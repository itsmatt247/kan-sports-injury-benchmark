#!/usr/bin/env python3
"""
KAN Variants vs MLP & XGBoost Benchmark
========================================
30-seed evaluation on 4 sports injury datasets.
Optimized for Yale Bouchet H200 GPUs.

Metrics: Accuracy, F1, AUC-ROC, Precision, Recall, Inference Time, Training Loss
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, recall_score
)

import xgboost as xgb

warnings.filterwarnings('ignore')

# Import local efficient_kan module (bundled in scripts folder)
from efficient_kan import KAN

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'n_seeds': 30,
    'epochs': 200,
    'batch_size': 1024,
    'learning_rate': 0.005,
    'test_size': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
}

RESULTS_DIR = Path(__file__).parent.parent / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
MODELS_DIR = RESULTS_DIR / 'models'
DATA_DIR = Path(__file__).parent.parent / 'data'

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class MLP(nn.Module):
    """Standard MLP baseline with BatchNorm and Dropout."""
    def __init__(self, n_features, hidden=[64, 32]):
        super().__init__()
        dims = [n_features] + hidden + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.extend([nn.BatchNorm1d(dims[i + 1]), nn.ReLU(), nn.Dropout(0.2)])
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class EfficientKANClassifier(nn.Module):
    """B-spline KAN from efficient-kan."""
    def __init__(self, n_features, hidden=20, grid_size=5):
        super().__init__()
        self.kan = KAN([n_features, hidden, 1], grid_size=grid_size, spline_order=3)
    
    def forward(self, x):
        return torch.sigmoid(self.kan(x))


class ChebyKANLayer(nn.Module):
    """Chebyshev polynomial layer."""
    def __init__(self, in_features, out_features, degree=4):
        super().__init__()
        self.degree = degree
        self.coeffs = nn.Parameter(torch.randn(in_features, out_features, degree + 1) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        x_norm = torch.tanh(x)
        T = [torch.ones_like(x_norm), x_norm]
        for _ in range(2, self.degree + 1):
            T.append(2 * x_norm * T[-1] - T[-2])
        T_stack = torch.stack(T, dim=-1)
        return torch.einsum('bid,iod->bo', T_stack, self.coeffs) + self.bias


class ChebyKAN(nn.Module):
    """ChebyKAN: Chebyshev polynomial-based KAN."""
    def __init__(self, n_features, hidden=[64, 32], degree=4):
        super().__init__()
        dims = [n_features] + hidden + [1]
        self.layers = nn.ModuleList([
            ChebyKANLayer(dims[i], dims[i + 1], degree) 
            for i in range(len(dims) - 1)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)


class RBFKANLayer(nn.Module):
    """Radial Basis Function layer for FastKAN."""
    def __init__(self, in_features, out_features, num_centers=16):
        super().__init__()
        self.centers = nn.Parameter(
            torch.linspace(-2, 2, num_centers).unsqueeze(0).unsqueeze(0).repeat(in_features, out_features, 1)
        )
        self.log_bw = nn.Parameter(torch.zeros(in_features, out_features, num_centers))
        self.weights = nn.Parameter(torch.randn(in_features, out_features, num_centers) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        x_exp = x.unsqueeze(2).unsqueeze(3)
        bw = torch.exp(self.log_bw) + 0.1
        rbf = torch.exp(-((x_exp - self.centers) ** 2) / (2 * bw ** 2))
        return (rbf * self.weights).sum(dim=-1).sum(dim=1) + self.bias


class FastKAN(nn.Module):
    """FastKAN: RBF-based KAN."""
    def __init__(self, n_features, hidden=[64, 32], num_centers=16):
        super().__init__()
        dims = [n_features] + hidden + [1]
        self.layers = nn.ModuleList([
            RBFKANLayer(dims[i], dims[i + 1], num_centers) 
            for i in range(len(dims) - 1)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)


class WaveletKANLayer(nn.Module):
    """Mexican Hat wavelet layer for WavKAN."""
    def __init__(self, in_features, out_features, num_wavelets=12):
        super().__init__()
        self.trans = nn.Parameter(
            torch.linspace(-3, 3, num_wavelets).unsqueeze(0).unsqueeze(0).repeat(in_features, out_features, 1)
        )
        self.log_scale = nn.Parameter(torch.zeros(in_features, out_features, num_wavelets))
        self.weights = nn.Parameter(torch.randn(in_features, out_features, num_wavelets) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def mexican_hat(self, x):
        return (1 - x ** 2) * torch.exp(-x ** 2 / 2)
    
    def forward(self, x):
        x_exp = x.unsqueeze(2).unsqueeze(3)
        scale = torch.exp(self.log_scale) + 0.1
        wav = self.mexican_hat((x_exp - self.trans) / scale)
        return (wav * self.weights).sum(dim=-1).sum(dim=1) + self.bias


class WavKAN(nn.Module):
    """WavKAN: Wavelet-based KAN (standardized to [64, 32])."""
    def __init__(self, n_features, hidden=[64, 32], num_wavelets=12):
        super().__init__()
        dims = [n_features] + hidden + [1]
        self.layers = nn.ModuleList([
            WaveletKANLayer(dims[i], dims[i + 1], num_wavelets) 
            for i in range(len(dims) - 1)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)


class FourierKANLayer(nn.Module):
    """Fourier series layer for FourierKAN."""
    def __init__(self, in_features, out_features, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        self.a = nn.Parameter(torch.randn(in_features, out_features, grid_size) * 0.1)
        self.b = nn.Parameter(torch.randn(in_features, out_features, grid_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        freqs = torch.arange(1, self.grid_size + 1, device=x.device).float()
        x_exp = x.unsqueeze(-1)
        cos_terms = torch.cos(freqs * x_exp * np.pi)
        sin_terms = torch.sin(freqs * x_exp * np.pi)
        return (
            torch.einsum('big,iog->bo', cos_terms, self.a) + 
            torch.einsum('big,iog->bo', sin_terms, self.b) + 
            self.bias
        )


class FourierKAN(nn.Module):
    """FourierKAN: Fourier series-based KAN."""
    def __init__(self, n_features, hidden=[64, 32], grid_size=8):
        super().__init__()
        dims = [n_features] + hidden + [1]
        self.layers = nn.ModuleList([
            FourierKANLayer(dims[i], dims[i + 1], grid_size) 
            for i in range(len(dims) - 1)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)


# =============================================================================
# MODEL FACTORY
# =============================================================================

def get_model_configs():
    """Return model factory functions with standardized architectures."""
    return {
        'MLP': lambda n: MLP(n, hidden=[64, 32]),
        'efficient-kan': lambda n: EfficientKANClassifier(n, hidden=20, grid_size=5),
        'ChebyKAN': lambda n: ChebyKAN(n, hidden=[64, 32], degree=4),
        'FastKAN': lambda n: FastKAN(n, hidden=[64, 32], num_centers=16),
        'WavKAN': lambda n: WavKAN(n, hidden=[64, 32], num_wavelets=12),
        'FourierKAN': lambda n: FourierKAN(n, hidden=[64, 32], grid_size=8),
    }


def count_parameters(model):
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_datasets():
    """Load all 4 sports injury datasets (FULL, no subsampling)."""
    datasets = {}
    
    # Dataset 1: Collegiate Athlete Injury Dataset (200 samples)
    df = pd.read_csv(DATA_DIR / 'collegiate_athlete_injury_dataset.csv')
    for col in ['Gender', 'Position']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    if 'Athlete_ID' in df.columns:
        df = df.drop('Athlete_ID', axis=1)
    X = df.drop('Injury_Indicator', axis=1).values
    y = df['Injury_Indicator'].values
    datasets['Collegiate Athlete'] = {
        'X': StandardScaler().fit_transform(X), 
        'y': y, 
        'n_features': X.shape[1]
    }
    print(f"✓ Collegiate Athlete: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Dataset 2: Injury Data (1000 samples)
    df = pd.read_csv(DATA_DIR / 'injury_data.csv')
    X = df.drop('Likelihood_of_Injury', axis=1).values
    y = df['Likelihood_of_Injury'].values
    datasets['Injury Data'] = {
        'X': StandardScaler().fit_transform(X), 
        'y': y, 
        'n_features': X.shape[1]
    }
    print(f"✓ Injury Data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Dataset 3: Day Timeseries (42k samples - FULL)
    df = pd.read_csv(DATA_DIR / 'archive' / 'day_approach_maskedID_timeseries.csv')
    feature_cols = [c for c in df.columns if c not in ['injury', 'Date', 'Athlete ID']]
    X = np.nan_to_num(df[feature_cols].values, nan=0.0)
    y = df['injury'].values
    datasets['Day Timeseries'] = {
        'X': StandardScaler().fit_transform(X), 
        'y': y, 
        'n_features': X.shape[1]
    }
    print(f"✓ Day Timeseries: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Dataset 4: Multimodal Sports (15k samples - FULL)
    df = pd.read_csv(DATA_DIR / 'archive-2' / 'multimodal_sports_injury_dataset.csv')
    for col in ['sport_type', 'gender', 'playing_surface']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    drop_cols = ['athlete_id', 'session_id', 'injury_occurred']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = np.nan_to_num(df[feature_cols].values, nan=0.0)
    y = (df['injury_occurred'].values > 0).astype(int)  # Binary
    datasets['Multimodal Sports'] = {
        'X': StandardScaler().fit_transform(X), 
        'y': y, 
        'n_features': X.shape[1]
    }
    print(f"✓ Multimodal Sports: {X.shape[0]} samples, {X.shape[1]} features")
    
    return datasets


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_torch_model(model, X_train, y_train, device, epochs=200, lr=0.005):
    """Train a PyTorch model and return loss history."""
    model = model.to(device)
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    loader = DataLoader(
        TensorDataset(X_t, y_t), 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=6,
        persistent_workers=True
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        loss_history.append(epoch_loss / n_batches)
    
    return model, loss_history


def evaluate_torch_model(model, X_test, y_test, device):
    """Evaluate a PyTorch model and return all metrics."""
    model.eval()
    X_t = torch.FloatTensor(X_test).to(device)
    
    # Inference time
    start = time.perf_counter()
    with torch.no_grad():
        y_prob = model(X_t).cpu().numpy().flatten()
    inference_time = time.perf_counter() - start
    
    y_pred = (y_prob > 0.5).astype(int)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'inference_time': inference_time
    }


def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_xgboost(model, X_test, y_test):
    """Evaluate XGBoost model."""
    start = time.perf_counter()
    y_prob = model.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - start
    
    y_pred = (y_prob > 0.5).astype(int)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'inference_time': inference_time
    }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(datasets, n_seeds=30):
    """Run full benchmark across all datasets and models."""
    device = torch.device(CONFIG['device'])
    print(f"\n🖥️  Device: {device}")
    print(f"🎲 Seeds: {n_seeds}")
    print(f"📊 Datasets: {len(datasets)}")
    
    model_configs = get_model_configs()
    all_results = []
    all_loss_histories = {}
    model_params = {}
    
    for dataset_name, data in datasets.items():
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {dataset_name}")
        print(f"{'='*70}")
        
        X, y = data['X'], data['y']
        n_features = data['n_features']
        
        for seed in range(1, n_seeds + 1):
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=CONFIG['test_size'], 
                random_state=seed, stratify=y
            )
            
            seed_results = {}
            
            # Train PyTorch models (KANs + MLP)
            for model_name, model_fn in model_configs.items():
                start_time = time.perf_counter()
                model = model_fn(n_features)
                
                # Store parameter count (once)
                if model_name not in model_params:
                    model_params[model_name] = count_parameters(model)
                
                model, loss_history = train_torch_model(
                    model, X_train, y_train, device, 
                    epochs=CONFIG['epochs'], lr=CONFIG['learning_rate']
                )
                train_time = time.perf_counter() - start_time
                
                metrics = evaluate_torch_model(model, X_test, y_test, device)
                metrics['train_time'] = train_time
                metrics['model'] = model_name
                metrics['dataset'] = dataset_name
                metrics['seed'] = seed
                metrics['n_params'] = model_params[model_name]
                
                all_results.append(metrics)
                seed_results[model_name] = metrics['accuracy']
                
                # Store loss history and save model weights (first seed only)
                if seed == 1:
                    key = f"{dataset_name}_{model_name}"
                    all_loss_histories[key] = loss_history
                    
                    # Save model weights for formula extraction
                    model_path = MODELS_DIR / f"{dataset_name.replace(' ', '_').lower()}_{model_name.lower()}.pt"
                    torch.save(model.state_dict(), model_path)
            
            # Train XGBoost
            start_time = time.perf_counter()
            xgb_model = train_xgboost(X_train, y_train)
            train_time = time.perf_counter() - start_time
            
            metrics = evaluate_xgboost(xgb_model, X_test, y_test)
            metrics['train_time'] = train_time
            metrics['model'] = 'XGBoost'
            metrics['dataset'] = dataset_name
            metrics['seed'] = seed
            metrics['n_params'] = xgb_model.get_booster().num_features() * 100  # Approximate
            
            if 'XGBoost' not in model_params:
                model_params['XGBoost'] = metrics['n_params']
            
            all_results.append(metrics)
            seed_results['XGBoost'] = metrics['accuracy']
            
            # Print progress
            best = max(seed_results, key=seed_results.get)
            accs = ' | '.join([f"{k[:4]}={v:.3f}" for k, v in seed_results.items()])
            print(f"Seed {seed:2d}: {accs} → {best}")
    
    return pd.DataFrame(all_results), all_loss_histories, model_params


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_statistics(df):
    """Compute summary statistics with Bonferroni-corrected p-values."""
    datasets = df['dataset'].unique()
    models = [m for m in df['model'].unique() if m not in ['MLP', 'XGBoost']]
    
    summary_stats = []
    pvalues_mlp = []
    pvalues_xgb = []
    
    for dataset in datasets:
        df_d = df[df['dataset'] == dataset]
        
        mlp_accs = df_d[df_d['model'] == 'MLP']['accuracy'].values
        xgb_accs = df_d[df_d['model'] == 'XGBoost']['accuracy'].values
        
        for model in df['model'].unique():
            accs = df_d[df_d['model'] == model]['accuracy'].values
            f1s = df_d[df_d['model'] == model]['f1'].values
            aucs = df_d[df_d['model'] == model]['auc_roc'].values
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            ci = stats.t.interval(0.95, len(accs)-1, loc=mean_acc, scale=std_acc/np.sqrt(len(accs)))
            
            summary_stats.append({
                'dataset': dataset,
                'model': model,
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc,
                'accuracy_ci_low': ci[0],
                'accuracy_ci_high': ci[1],
                'f1_mean': np.mean(f1s),
                'f1_std': np.std(f1s),
                'auc_roc_mean': np.mean(aucs),
                'auc_roc_std': np.std(aucs),
            })
            
            # P-values vs MLP
            if model not in ['MLP', 'XGBoost']:
                _, p_mlp = stats.ttest_rel(accs, mlp_accs)
                pvalues_mlp.append({
                    'dataset': dataset,
                    'model': model,
                    'p_value_raw': p_mlp,
                })
                
                # P-values vs XGBoost
                _, p_xgb = stats.ttest_rel(accs, xgb_accs)
                pvalues_xgb.append({
                    'dataset': dataset,
                    'model': model,
                    'p_value_raw': p_xgb,
                })
    
    # Bonferroni correction
    n_comparisons = len(pvalues_mlp)
    for p in pvalues_mlp:
        p['p_value_bonferroni'] = min(p['p_value_raw'] * n_comparisons, 1.0)
        p['significant'] = p['p_value_bonferroni'] < 0.05
    
    for p in pvalues_xgb:
        p['p_value_bonferroni'] = min(p['p_value_raw'] * n_comparisons, 1.0)
        p['significant'] = p['p_value_bonferroni'] < 0.05
    
    return (
        pd.DataFrame(summary_stats),
        pd.DataFrame(pvalues_mlp),
        pd.DataFrame(pvalues_xgb)
    )


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def create_figures(df, summary_df, pvalues_mlp_df, pvalues_xgb_df, loss_histories, model_params):
    """Generate all publication-quality figures."""
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Color palette
    colors_kan = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']
    color_mlp = '#e74c3c'
    color_xgb = '#34495e'
    
    kan_models = ['efficient-kan', 'ChebyKAN', 'FastKAN', 'WavKAN', 'FourierKAN']
    all_models_mlp = ['MLP'] + kan_models
    all_models_xgb = ['XGBoost'] + kan_models
    
    datasets = df['dataset'].unique()
    
    # =========================================================================
    # SECTION 1: KAN vs MLP
    # =========================================================================
    
    # Figure 1: Bar chart - Accuracy vs MLP
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for ax, dataset in zip(axes, datasets):
        df_d = summary_df[summary_df['dataset'] == dataset]
        means = [df_d[df_d['model'] == m]['accuracy_mean'].values[0] for m in all_models_mlp]
        stds = [df_d[df_d['model'] == m]['accuracy_std'].values[0] for m in all_models_mlp]
        
        bar_colors = [color_mlp] + colors_kan
        bars = ax.bar(range(len(all_models_mlp)), means, yerr=stds, 
                      color=bar_colors, capsize=5, edgecolor='black', linewidth=1.2)
        
        ax.set_xticks(range(len(all_models_mlp)))
        ax.set_xticklabels(all_models_mlp, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(dataset, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)
        
        for bar, val, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, val + std + 0.02, 
                    f'{val:.1%}', ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('KAN Variants vs MLP: Accuracy (30-Seed Average ± Std)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'kan_vs_mlp_accuracy.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: kan_vs_mlp_accuracy.png")
    
    # Figure 2: Table - Params, Time, Accuracy vs MLP
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    table_data = []
    for model in all_models_mlp:
        row = [model, f"{model_params.get(model, 0):,}"]
        for dataset in datasets:
            df_m = summary_df[(summary_df['model'] == model) & (summary_df['dataset'] == dataset)]
            if len(df_m) > 0:
                acc = df_m['accuracy_mean'].values[0]
                row.append(f"{acc:.3f}")
            else:
                row.append("-")
        # Average training time
        avg_time = df[df['model'] == model]['train_time'].mean()
        row.append(f"{avg_time:.2f}s")
        table_data.append(row)
    
    col_labels = ['Model', 'Params'] + list(datasets) + ['Avg Time']
    table = ax.table(cellText=table_data, colLabels=col_labels, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.title('KAN vs MLP: Parameters, Accuracy, and Training Time', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(FIGURES_DIR / 'kan_vs_mlp_table.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: kan_vs_mlp_table.png")
    
    # Figure 3: Scatter - Overall Accuracy vs Parameters
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, model in enumerate(all_models_mlp):
        avg_acc = summary_df[summary_df['model'] == model]['accuracy_mean'].mean()
        params = model_params.get(model, 0)
        color = color_mlp if model == 'MLP' else colors_kan[i-1]
        ax.scatter(params, avg_acc, s=200, c=color, edgecolor='black', linewidth=2, label=model)
        ax.annotate(model, (params, avg_acc), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Average Accuracy (across all datasets)', fontsize=12)
    ax.set_title('KAN vs MLP: Accuracy vs Model Complexity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'kan_vs_mlp_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: kan_vs_mlp_scatter.png")
    
    # =========================================================================
    # SECTION 2: KAN vs XGBoost
    # =========================================================================
    
    # Figure 4: Bar chart - Accuracy vs XGBoost
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for ax, dataset in zip(axes, datasets):
        df_d = summary_df[summary_df['dataset'] == dataset]
        means = [df_d[df_d['model'] == m]['accuracy_mean'].values[0] for m in all_models_xgb]
        stds = [df_d[df_d['model'] == m]['accuracy_std'].values[0] for m in all_models_xgb]
        
        bar_colors = [color_xgb] + colors_kan
        bars = ax.bar(range(len(all_models_xgb)), means, yerr=stds, 
                      color=bar_colors, capsize=5, edgecolor='black', linewidth=1.2)
        
        ax.set_xticks(range(len(all_models_xgb)))
        ax.set_xticklabels(all_models_xgb, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(dataset, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)
        
        for bar, val, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, val + std + 0.02, 
                    f'{val:.1%}', ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('KAN Variants vs XGBoost: Accuracy (30-Seed Average ± Std)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'kan_vs_xgboost_accuracy.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: kan_vs_xgboost_accuracy.png")
    
    # Figure 5: Table - Params, Time, Accuracy vs XGBoost
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    table_data = []
    for model in all_models_xgb:
        row = [model, f"{model_params.get(model, 0):,}"]
        for dataset in datasets:
            df_m = summary_df[(summary_df['model'] == model) & (summary_df['dataset'] == dataset)]
            if len(df_m) > 0:
                acc = df_m['accuracy_mean'].values[0]
                row.append(f"{acc:.3f}")
            else:
                row.append("-")
        avg_time = df[df['model'] == model]['train_time'].mean()
        row.append(f"{avg_time:.2f}s")
        table_data.append(row)
    
    col_labels = ['Model', 'Params'] + list(datasets) + ['Avg Time']
    table = ax.table(cellText=table_data, colLabels=col_labels, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.title('KAN vs XGBoost: Parameters, Accuracy, and Training Time', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(FIGURES_DIR / 'kan_vs_xgboost_table.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: kan_vs_xgboost_table.png")
    
    # Figure 6: Scatter - Overall Accuracy vs Parameters
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, model in enumerate(all_models_xgb):
        avg_acc = summary_df[summary_df['model'] == model]['accuracy_mean'].mean()
        params = model_params.get(model, 0)
        color = color_xgb if model == 'XGBoost' else colors_kan[i-1]
        ax.scatter(params, avg_acc, s=200, c=color, edgecolor='black', linewidth=2, label=model)
        ax.annotate(model, (params, avg_acc), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Average Accuracy (across all datasets)', fontsize=12)
    ax.set_title('KAN vs XGBoost: Accuracy vs Model Complexity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'kan_vs_xgboost_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: kan_vs_xgboost_scatter.png")
    
    # =========================================================================
    # P-VALUE TABLES
    # =========================================================================
    
    # Figure 7: P-values vs MLP
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    table_data = []
    for _, row in pvalues_mlp_df.iterrows():
        sig = "✓" if row['significant'] else ""
        table_data.append([
            row['dataset'], row['model'], 
            f"{row['p_value_raw']:.4f}", 
            f"{row['p_value_bonferroni']:.4f}", 
            sig
        ])
    
    col_labels = ['Dataset', 'KAN Variant', 'p-value (raw)', 'p-value (Bonferroni)', 'Significant']
    table = ax.table(cellText=table_data, colLabels=col_labels, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#27ae60')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Paired t-test: KAN Variants vs MLP (Bonferroni Corrected)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(FIGURES_DIR / 'pvalues_table_mlp.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: pvalues_table_mlp.png")
    
    # Figure 8: P-values vs XGBoost
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    table_data = []
    for _, row in pvalues_xgb_df.iterrows():
        sig = "✓" if row['significant'] else ""
        table_data.append([
            row['dataset'], row['model'], 
            f"{row['p_value_raw']:.4f}", 
            f"{row['p_value_bonferroni']:.4f}", 
            sig
        ])
    
    col_labels = ['Dataset', 'KAN Variant', 'p-value (raw)', 'p-value (Bonferroni)', 'Significant']
    table = ax.table(cellText=table_data, colLabels=col_labels, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#8e44ad')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Paired t-test: KAN Variants vs XGBoost (Bonferroni Corrected)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(FIGURES_DIR / 'pvalues_table_xgboost.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: pvalues_table_xgboost.png")
    
    # =========================================================================
    # TRAINING LOSS CURVES
    # =========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, dataset in zip(axes, datasets):
        for i, model in enumerate(kan_models + ['MLP']):
            key = f"{dataset}_{model}"
            if key in loss_histories:
                color = colors_kan[i] if i < len(colors_kan) else color_mlp
                ax.plot(loss_histories[key], label=model, color=color, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(dataset, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Training Loss Curves (Seed 1)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'training_loss_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: training_loss_curves.png")
    
    print(f"\n📁 All figures saved to: {FIGURES_DIR}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("KAN VARIANTS vs MLP & XGBoost BENCHMARK")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📂 Loading datasets...")
    datasets = load_datasets()
    
    # Run benchmark
    print("\n🚀 Running benchmark...")
    start_time = time.time()
    results_df, loss_histories, model_params = run_benchmark(datasets, n_seeds=CONFIG['n_seeds'])
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total benchmark time: {total_time/60:.1f} minutes")
    
    # Save raw results
    results_df.to_csv(RESULTS_DIR / 'benchmark_results.csv', index=False)
    print(f"✓ Saved: benchmark_results.csv")
    
    # Compute statistics
    print("\n📊 Computing statistics...")
    summary_df, pvalues_mlp_df, pvalues_xgb_df = compute_statistics(results_df)
    
    summary_df.to_csv(RESULTS_DIR / 'summary_stats.csv', index=False)
    pvalues_mlp_df.to_csv(RESULTS_DIR / 'pvalues_vs_mlp.csv', index=False)
    pvalues_xgb_df.to_csv(RESULTS_DIR / 'pvalues_vs_xgboost.csv', index=False)
    
    # Save model params
    pd.DataFrame([model_params]).to_csv(RESULTS_DIR / 'model_params.csv', index=False)
    
    print("✓ Saved: summary_stats.csv")
    print("✓ Saved: pvalues_vs_mlp.csv")
    print("✓ Saved: pvalues_vs_xgboost.csv")
    print("✓ Saved: model_params.csv")
    
    # Generate figures
    print("\n🎨 Generating figures...")
    create_figures(results_df, summary_df, pvalues_mlp_df, pvalues_xgb_df, loss_histories, model_params)
    
    # Final summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\n📁 Results saved to: {RESULTS_DIR}")
    print(f"📁 Figures saved to: {FIGURES_DIR}")
    
    # Print summary table
    print("\n📈 SUMMARY (Mean Accuracy ± Std):")
    pivot = summary_df.pivot_table(
        values='accuracy_mean', 
        index='model', 
        columns='dataset'
    )
    print(pivot.round(3).to_string())


if __name__ == '__main__':
    main()

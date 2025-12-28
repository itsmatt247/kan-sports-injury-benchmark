#!/usr/bin/env python3
"""
TMLR Paper Figure Generator
===========================
Generates publication-quality figures specifically for the TMLR submission.
Focuses on KAN vs XGBoost comparison on 'Day Timeseries' and 'Multimodal Sports'.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

# Config
RESULTS_DIR = Path(__file__).parent.parent / 'results'
OUTPUT_DIR = RESULTS_DIR / 'figures_tmlr'
CSV_PATH = RESULTS_DIR / 'benchmark_results.csv'

def setup_style():
    """Set publication-quality plotting style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

def load_data():
    """Load and filter data for relevant datasets."""
    df = pd.read_csv(CSV_PATH)
    datasets_of_interest = ['Day Timeseries', 'Multimodal Sports']
    return df[df['dataset'].isin(datasets_of_interest)]

def plot_comparative_accuracy(df):
    """Figure 1: Side-by-side Accuracy comparison."""
    plt.figure(figsize=(12, 6))
    
    # Filter to main models for cleaner plot
    models_to_show = ['efficient-kan', 'ChebyKAN', 'FastKAN', 'WavKAN', 'XGBoost']
    df_filtered = df[df['model'].isin(models_to_show)]
    
    # Custom palette
    colors = {
        'XGBoost': '#2c3e50',     # Dark Blue
        'efficient-kan': '#3498db', # Blue
        'ChebyKAN': '#9b59b6',    # Purple
        'FastKAN': '#2ecc71',     # Green
        'WavKAN': '#f1c40f'       # Yellow
    }
    
    dataset_order = ['Day Timeseries', 'Multimodal Sports']
    
    ax = sns.barplot(
        data=df_filtered, 
        x='dataset', 
        y='accuracy', 
        hue='model',
        hue_order=models_to_show,
        order=dataset_order,
        palette=colors,
        errorbar='sd',
        capsize=0.05,
        edgecolor='black'
    )
    
    # Calculate stats for dynamic label positioning (above error bar)
    stats = df_filtered.groupby(['model', 'dataset'])['accuracy'].agg(['mean', 'std'])
    
    # Add numbers on top of error bars
    for i, container in enumerate(ax.containers):
        if i >= len(models_to_show): break
        model = models_to_show[i]
        
        for j, bar in enumerate(container):
            dataset = dataset_order[j]
            try:
                rec = stats.loc[(model, dataset)]
                mean_val = rec['mean']
                std_val = rec['std']
                if pd.isna(std_val): std_val = 0
                
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_val + std_val + 0.005, # Position above error bar
                    f'{mean_val:.3f}',
                    ha='center', 
                    va='bottom', 
                    fontsize=8,
                    fontweight='bold'
                )
            except KeyError:
                pass
    
    plt.ylim(0.5, 1.05)
    plt.ylabel('Test Accuracy')
    plt.xlabel('')
    plt.title('KAN Variants vs Baselines: Performance by Dataset')
    plt.legend(title='Model architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_accuracy_comparison.png', bbox_inches='tight')
    print("✓ Saved Figure 1: Accuracy Comparison")

def plot_efficiency_tradeoff(df):
    """Figure 2: Accuracy vs Inference Time trade-off."""
    plt.figure(figsize=(10, 8))
    
    # Aggregate data
    summary = df.groupby(['model', 'dataset']).agg({
        'accuracy': 'mean',
        'inference_time': 'mean'
    }).reset_index()
    
    datasets = summary['dataset'].unique()
    markers = {'Day Timeseries': 'o', 'Multimodal Sports': 's'}
    
    summary = summary[summary['model'] != 'FourierKAN']
    summary = summary[summary['model'] != 'MLP']  # Exclude MLP
    
    # Custom palette (same as Fig 1)
    colors = {
        'XGBoost': '#2c3e50',     # Dark Blue
        'efficient-kan': '#3498db', # Blue
        'ChebyKAN': '#9b59b6',    # Purple
        'FastKAN': '#2ecc71',     # Green
        'WavKAN': '#f1c40f'       # Yellow
    }
    
    for i, row in summary.iterrows():
        color = colors.get(row['model'], '#333333')
        
        plt.scatter(
            row['inference_time'], 
            row['accuracy'], 
            marker=markers[row['dataset']],
            s=150,
            color=color,
            edgecolor='black',
            label=row['model'] if i < 10 else "", 
            alpha=0.8,
            zorder=3
        )
        
        # Annotate all points
        plt.annotate(
            row['model'], 
            (row['inference_time'], row['accuracy']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, fontweight='bold'
        )

    # Manual Legend (Datasets only, models are labeled)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Time Series', markersize=10),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', label='Multimodal', markersize=10),
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.xscale('log')
    plt.xlabel('Inference Time (seconds, log scale)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Efficiency Trade-off')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_efficiency.png')
    print("✓ Saved Figure 2: Efficiency Trade-off")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    
    try:
        df = load_data()
        print(f"Loaded {len(df)} rows from benchmark results.")
        
        plot_comparative_accuracy(df)
        plot_efficiency_tradeoff(df)
        plot_pvalue_table()
        
        print("\n✅ All TMLR figures generated successfully in:")
        print(f"   {OUTPUT_DIR}")
        
    except FileNotFoundError:
        print("❌ Error: benchmark_results.csv not found.")
        print("   Please ensure you have downloaded the results from the cluster.")

def plot_pvalue_table():
    """Figure 3: Formatted P-Value Table (KAN vs XGBoost)."""
    try:
        df = pd.read_csv(RESULTS_DIR / 'pvalues_vs_xgboost.csv')
        # Filter for TMLR datasets
        datasets = ['Day Timeseries', 'Multimodal Sports']
        df = df[df['dataset'].isin(datasets)]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        
        table_data = []
        for _, row in df.iterrows():
            # Bold checkmark for significance
            sig = "YES" if row['significant'] else "No"
            # Highlight small p-values
            pval = row['p_value_bonferroni']
            pval_str = "< 0.001" if pval < 0.001 else f"{pval:.4f}"
            
            table_data.append([
                row['dataset'], 
                row['model'], 
                pval_str, 
                sig
            ])
            
        col_labels = ['Dataset', 'KAN Comparison', 'P-Value (Adj)', 'Significant?']
        table = ax.table(
            cellText=table_data, 
            colLabels=col_labels, 
            loc='center', 
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Styling
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', fontweight='bold')
            elif row % 2 == 0:
                cell.set_facecolor('#f2f2f2')

        plt.title('Statistical Significance: KAN Variants vs XGBoost', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'fig3_pvalues_table.png', bbox_inches='tight', dpi=300)
        print("✓ Saved Figure 3: P-Value Table")
        
    except FileNotFoundError:
        print("⚠ Could not find pvalues_vs_xgboost.csv, skipping table.")

if __name__ == "__main__":
    main()

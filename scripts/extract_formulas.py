#!/usr/bin/env python3
"""
KAN Symbolic Formula Extractor
==============================
This script loads trained KAN models and attempts to extract human-readable 
mathematical formulas. This supports the "Interpretability & Trust" paper.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

# Add scripts folder to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_benchmark import EfficientKANClassifier, ChebyKAN, FastKAN, WavKAN, FourierKAN

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / 'results'
MODELS_DIR = RESULTS_DIR / 'models'
DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = RESULTS_DIR / 'formulas'

# Common functions to fit
def f_linear(x, a, b): return a * x + b
def f_quadratic(x, a, b, c): return a * x**2 + b * x + c
def f_sin(x, a, b, c): return a * np.sin(b * x + c)
def f_exp(x, a, b, c): return a * np.exp(b * x) + c
def f_log(x, a, b, c): return a * np.log(np.abs(x) + 0.1) + c

def get_best_fit(x, y):
    """Try to fit common functions and return the best one."""
    funcs = [
        ('linear', f_linear),
        ('quadratic', f_quadratic),
        ('sin', f_sin),
        ('exp', f_exp),
        ('log', f_log)
    ]
    
    best_func_name = 'unknown'
    best_r2 = -float('inf')
    best_formula = ""
    
    for name, func in funcs:
        try:
            popt, _ = curve_fit(func, x, y, maxfev=2000)
            y_pred = func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            if r2 > best_r2:
                best_r2 = r2
                best_func_name = name
                if name == 'linear':
                    best_formula = f"{popt[0]:.3f}*x + {popt[1]:.3f}"
                elif name == 'quadratic':
                    best_formula = f"{popt[0]:.3f}*x^2 + {popt[1]:.3f}*x + {popt[2]:.3f}"
                elif name == 'sin':
                    best_formula = f"{popt[0]:.3f}*sin({popt[1]:.3f}*x + {popt[2]:.3f})"
                elif name == 'exp':
                    best_formula = f"{popt[0]:.3f}*exp({popt[1]:.3f}*x) + {popt[2]:.3f}"
                elif name == 'log':
                    best_formula = f"{popt[0]:.3f}*log(|x|+0.1) + {popt[1]:.3f}"
        except:
            continue
            
    return best_func_name, best_formula, best_r2

def extract_efficient_kan_formulas(model, n_features):
    """Extract formulas from the B-spline KAN (efficient-kan)."""
    formulas = []
    # Sample range
    x_sample = np.linspace(-1.5, 1.5, 100)
    x_tensor = torch.FloatTensor(x_sample).unsqueeze(-1).repeat(1, n_features)
    
    # We'll analyze the first layer mainly for interpretable feature relationships
    first_layer = model.kan.layers[0]
    
    with torch.no_grad():
        # Evaluate base activation + spline contribution for each feature
        for i in range(n_features):
            # Input where only feature i varies
            x_input = torch.zeros(100, n_features)
            x_input[:, i] = torch.FloatTensor(x_sample)
            
            # The output of KANLinear is sum_j(phi_ij(x_i))
            # We want to see how the j-th output neuron depends on the i-th input
            # For simplicity, we'll look at the first hidden neuron
            y_output = first_layer(x_input)[:, 0].numpy()
            
            name, formula, r2 = get_best_fit(x_sample, y_output)
            formulas.append({
                'feature_index': i,
                'type': name,
                'formula': formula,
                'r2': r2
            })
            
    return formulas

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("KAN SYMBOLIC FORMULA EXTRACTOR")
    print("="*60)
    
    model_files = list(MODELS_DIR.glob("*.pt"))
    if not model_files:
        print(f"❌ No models found in {MODELS_DIR}")
        print("Please run the benchmark first to generate Seed 1 models.")
        return

    all_formulas = []

    for model_path in model_files:
        print(f"\nProcessing: {model_path.name}")
        parts = model_path.stem.split('_')
        # Expecting dataset_name_modelname.pt
        model_name = parts[-1]
        
        # Determine n_features based on filename (rough heuristic or load dummy)
        # For this script, we'll assume standard n_features from collegiate or hardcode
        # In a real run, you'd pass the dataset metadata
        n_features = 12 # Default guess for collegiate athlete
        
        if 'efficient-kan' in model_path.name:
            model = EfficientKANClassifier(n_features)
            model.load_state_dict(torch.load(model_path, map_utils='cpu'))
            formulas = extract_efficient_kan_formulas(model, n_features)
            
            df = pd.DataFrame(formulas)
            df['model'] = model_name
            df['dataset'] = '_'.join(parts[:-1])
            all_formulas.append(df)
            
            print(f"✓ Extracted formulas for {len(formulas)} features")
            print(df[['feature_index', 'type', 'r2']].head())

    if all_formulas:
        final_df = pd.concat(all_formulas)
        final_df.to_csv(OUTPUT_DIR / 'extracted_formulas.csv', index=False)
        print(f"\n✅ All formulas saved to: {OUTPUT_DIR / 'extracted_formulas.csv'}")

if __name__ == "__main__":
    main()

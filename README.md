# KAN Benchmarking for Sports Injury Prediction

Comparative evaluation of **Kolmogorov-Arnold Networks (KANs)** vs. traditional ML models for sports injury prediction tasks.

## 🏆 Key Results (5-Seed Verified)

| Dataset | XGBoost | MLP | ChebyKAN |
|---------|---------|-----|----------|
| Synthetic (HASI) | **97.8%** | 87.8% | 88.0% |
| Real (Premier League) | 69.9% | **71.7%** | 66.9% |

### Key Findings
- **Synthetic Data**: XGBoost excels (97.8%); ChebyKAN competitive with MLP (~88%)
- **Real Data**: MLP performs best (71.7%); all models within 5% of each other
- KANs are **competitive** with traditional baselines on tabular data

## 📁 Project Structure

```
├── notebooks/
│   ├── verified_final_benchmark.ipynb    # Final 5-seed evaluation
│   ├── tune_all_kan_variants.ipynb       # Hyperparameter tuning
│   ├── final_comprehensive_analysis.ipynb
│   ├── statistical_rigor_analysis.ipynb  # K-fold CV, p-values
│   └── benchmark_tables.ipynb            # Publication-style tables
├── data/
│   ├── High_Accuracy_Sport_Injury_Dataset.xlsx
│   └── player_injuries_impact.csv
├── figures/
│   └── final_benchmark.png
├── references/
│   └── ReferenceKAN.pdf
└── requirements.txt
```

## 🧠 Models Implemented

### Baselines
- XGBoost
- MLP (Multi-Layer Perceptron)

### KAN Variants
- **ChebyKAN** (Chebyshev polynomials)
- FastKAN (Radial Basis Functions)
- FourierKAN (Fourier series)
- WavKAN (Wavelets)
- efficient-kan (B-splines)

## 📊 Datasets

1. **High Accuracy Sport Injury Dataset** (Synthetic)
   - 600 samples, 15 features
   - Task: Injury Risk Classification

2. **Player Injuries Impact** (Real - English Premier League)
   - 503 samples, 6 engineered features
   - Task: Predict player performance recovery post-injury

## 🚀 Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/verified_final_benchmark.ipynb
```

## 📈 Statistical Analysis

- **5-seed evaluation** for robust results
- **95% confidence intervals** reported
- **Paired t-tests** for statistical significance
- **Feature importance** comparison (gradient-based for KANs)

## 📚 References

- [TabKAN Paper](references/ReferenceKAN.pdf)
- [Awesome KAN Repository](https://github.com/mintisan/awesome-kan)

## License

MIT

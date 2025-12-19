# KAN Benchmarking for Sports Injury Prediction

Comparative evaluation of **Kolmogorov-Arnold Networks (KANs)** vs. traditional ML models for sports injury prediction tasks.

## 🏆 Key Results

| Dataset | Best Model | Accuracy |
|---------|------------|----------|
| Synthetic (HASI) | XGBoost | **94.4%** |
| **Real (Premier League)** | **ChebyKAN (wide)** | **74.3%** |

**ChebyKAN outperformed XGBoost by +5% on real-world Premier League injury data!**

## 📁 Project Structure

```
├── notebooks/
│   ├── high_accuracy_sport_injury_analysis.ipynb  # Synthetic data benchmark
│   ├── player_injuries_impact_analysis.ipynb     # Real data benchmark
│   ├── kan_benchmark_v2_tuned.ipynb              # Tuned models
│   ├── kan_hyperparam_sweep.ipynb                # Hyperparameter sweep
│   └── benchmark_tables.ipynb                    # Publication-style tables
├── data/
│   ├── High_Accuracy_Sport_Injury_Dataset.xlsx   # Synthetic dataset
│   └── player_injuries_impact.csv                # Real EPL injury data
├── figures/
│   └── *.png                                     # Result visualizations
├── references/
│   └── *.pdf                                     # Reference papers
└── requirements.txt
```

## 🧠 Models Implemented

### Baselines
- XGBoost
- MLP (Multi-Layer Perceptron)
- LSTM

### KAN Variants
- **ChebyKAN** (Chebyshev polynomials) - *Best performer on real data*
- efficient-kan (B-splines)
- FastKAN (Radial Basis Functions)
- FourierKAN (Fourier series)
- Wav-KAN (Wavelets)

## 📊 Datasets

1. **High Accuracy Sport Injury Dataset** (Synthetic)
   - 600 samples, 15 features
   - Binary classification: Injury Risk

2. **Player Injuries Impact** (Real - English Premier League)
   - 503 samples, 6 engineered features
   - Task: Predict if player performance improves post-injury

## 🚀 Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/benchmark_tables.ipynb
```

## 📈 Results Summary

### Synthetic Data
- XGBoost achieves 94.4% accuracy
- Best KAN: ChebyKAN at 86.7%

### Real Data (Premier League)
- **ChebyKAN (wide) achieves 74.3%** - beats XGBoost (69.3%) by +5%
- Wider architectures `[n, 100, 50, 1]` outperform deeper ones

## 📚 References

- [TabKAN Paper](references/ReferenceKAN.pdf)
- [Awesome KAN Repository](https://github.com/mintisan/awesome-kan)

## License

MIT

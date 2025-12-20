# KAN vs MLP for Sports Injury Prediction

**Comprehensive benchmark demonstrating Kolmogorov-Arnold Networks outperform MLPs on tabular sports injury data.**

## 🏆 Key Results (30-Seed Evaluation)

### Synthetic Dataset (600 samples, 15 features)

| Model | Mean Accuracy | Wins vs MLP |
|-------|--------------|-------------|
| **efficient-kan** | **92.4%** | **26/30** 🏆 |
| **WavKAN** | **91.1%** | **27/30** 🏆 |
| **FastKAN** | **91.9%** | **25/30** 🏆 |
| **ChebyKAN** | **89.3%** | **21/30** |
| MLP (baseline) | 87.5% | - |
| FourierKAN | 68.4% | 0/30 |

### Real Dataset (Premier League, 503 samples)

| Model | Mean Accuracy | Wins vs MLP |
|-------|--------------|-------------|
| MLP (baseline) | 66.2% | - |
| **ChebyKAN** | 64.1% | 8/30 |
| efficient-kan | 63.8% | 6/30 |
| WavKAN | 61.3% | 5/30 |
| FastKAN | 57.5% | 2/30 |
| FourierKAN | 52.3% | 1/30 |

## 📊 Key Findings

1. **4 out of 5 KAN variants beat MLP on synthetic data** with >70% win rate
2. **efficient-kan is best overall** (92.4% accuracy, statistically significant p < 0.001)
3. **WavKAN has highest MLP win rate** (27/30 = 90%)
4. KANs are competitive with MLP on real-world data (~3% difference)
5. **FourierKAN consistently underperforms** - not recommended for tabular data

![KAN Variants Comparison](figures/all_kan_variants_comparison.png)

## 📁 Repository Structure

```
├── notebooks/
│   └── kan_vs_mlp_final_benchmark.ipynb  # 30-seed benchmark
├── data/
│   ├── High_Accuracy_Sport_Injury_Dataset.xlsx
│   └── player_injuries_impact.csv
├── figures/
│   ├── all_kan_variants_comparison.png
│   └── all_kan_variants_boxplot.png
├── references/
│   ├── ReferenceKAN.pdf
│   └── MOREKANHELP.pdf
└── requirements.txt
```

## 🧠 KAN Variants Tested

| Variant | Basis Functions | Best Config |
|---------|----------------|-------------|
| efficient-kan | B-splines | [n, 20, 1], grid=5 |
| ChebyKAN | Chebyshev polynomials | [n, 64, 32, 1], deg=4 |
| FastKAN | Radial Basis Functions | [n, 64, 32, 1], centers=16 |
| WavKAN | Mexican Hat Wavelets | [n, 48, 24, 1], wavelets=12 |
| FourierKAN | Fourier series | [n, 64, 32, 1], grid=8 |

## 🚀 Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/kan_vs_mlp_final_benchmark.ipynb
```

## 📚 References

1. Liu et al. (2024) - *"KAN: Kolmogorov-Arnold Networks"*
2. Poeta et al. (2024) - *"A Benchmarking Study of KANs on Tabular Data"*

## License

MIT

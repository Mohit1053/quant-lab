# AI Quant Research Lab — Results

## Overview

End-to-end quantitative research platform for Indian equities (NIFTY 50).
Pipeline: Data Ingestion → Feature Engineering → Model Training → Regime Detection → Backtesting → Reporting.

---

## Data Pipeline

| Stage | Details |
|-------|---------|
| **Universe** | 49 NIFTY 50 stocks (1 delisted from index) |
| **Date Range** | Jan 2010 – Dec 2024 (~3,700 trading days) |
| **Features** | 15 engineered features per stock per day |
| **Train** | Jan 2010 – Dec 2021 (2,962 days) |
| **Validation** | Jan 2022 – Jun 2023 (370 days) |
| **Test (OOS)** | Jul 2023 – Dec 2024 (368 days) |

### Feature Families (15 total)
- **Log Returns**: 1d, 5d, 21d, 63d windows
- **Realized Volatility**: 5d, 21d, 63d rolling standard deviation
- **Momentum**: 5d, 21d, 63d cumulative return
- **Max Drawdown**: 21d, 63d rolling drawdown from peak
- Normalized via z-score per ticker

---

## Model Training Results

### Loss Function
Multi-task loss = Gaussian NLL + 0.3 × Direction Cross-Entropy + 0.3 × Volatility MSE

Gaussian NLL is negative when the model assigns high probability to actual targets.
**More negative = better fit** (e.g., -3.3 is better than -3.0).

### Pre-trained Masked Encoder (BERT-like)
| Metric | Value |
|--------|-------|
| Hardware | H100 80GB (Colab Pro+) |
| Architecture | Patch tokenizer (size=5), d_model=256, 6 layers, 8 heads |
| Epochs | 100 (full) |
| Final MSE Loss | 0.0505 |
| Training Time | 20.8 min |
| Embeddings | (146,816 × 256) |
| Model Size | 23.8 MB |

### Transformer Forecaster
| Metric | Value |
|--------|-------|
| Hardware | H100 80GB (Colab Pro+) |
| Architecture | Pre-norm encoder, CLS token, d_model=256, 6 layers, 8 heads |
| Prediction Heads | Gaussian distribution + 3-class direction + volatility |
| Epochs | 33 / 100 (early stopped, patience=15) |
| Train Loss | -3.137 |
| **Best Val Loss** | **-3.305** |
| Training Time | 6.2 min |
| Model Size | 59.1 MB (best checkpoint) |

### Temporal Fusion Transformer (TFT)
| Metric | Value |
|--------|-------|
| Hardware | A100 80GB (Colab Pro+) |
| Architecture | GRN blocks, Variable Selection, LSTM encoder, d_model=256 |
| Epochs | 38 / 100 (early stopped, patience=15) |
| Train Loss | -3.038 |
| **Best Val Loss** | **-3.235** |
| Training Time | 40.1 min |
| Model Size | 427.5 MB (best checkpoint) |

**Transformer vs TFT**: Transformer achieves slightly better val loss (-3.305 vs -3.235) in 6x less training time. Both models converge well; early stopping prevents overfitting.

---

## RL Portfolio Allocation

Starting capital: INR 10,00,000 (10 lakh)

### PPO (Proximal Policy Optimization)
| Metric | Colab (A100, 2M steps) | Local (RTX 4080, 1M steps) |
|--------|------------------------|---------------------------|
| Train Final Value | ₹96.87 lakh (+869%) | ₹1.07 crore (+967%) |
| Val Final Value | ₹12.20 lakh (+22%) | ₹11.99 lakh (+20%) |
| Training Time | 65.4 min | 62 min |
| Model Size | 1.4 MB | 1.4 MB |

### SAC (Soft Actor-Critic)
| Metric | Local (RTX 4080, 100K steps) |
|--------|------------------------------|
| Training Time | ~80 min |
| Model Size | 12.5 MB |

**Key Insight**: Both PPO runs show massive training returns (~10x) but modest validation returns (~20%). This is expected — RL agents overfit to training market dynamics. The positive validation returns confirm the agent learned generalizable portfolio allocation behavior.

---

## Backtest Performance (Out-of-Sample)

**Strategy**: Transformer-based signal → Top-5 stock selection → Rebalance every 5 days
**Test Period**: Jul 2023 – Dec 2024 (fully out-of-sample)
**Transaction Costs**: 10 bps commission + 5 bps slippage + 5 bps spread

### Key Metrics

| Metric | Our Strategy | NIFTY 50 (approx.) | Interpretation |
|--------|-------------|--------------------|----|
| **CAGR** | **32.04%** | ~15–20% | +12–17% alpha |
| **Sharpe Ratio** | **1.38** | ~1.0–1.2 | Better risk-adjusted returns |
| **Sortino Ratio** | **1.67** | ~1.0–1.3 | Better downside protection |
| **Max Drawdown** | **-11.54%** | ~-8 to -12% | Comparable to benchmark |
| **Calmar Ratio** | **2.78** | ~1.5–2.0 | Excellent risk/reward |
| **Total Return** | **50.07%** | ~25–35% | ~2× benchmark |
| **Volatility** | 17.78% | ~13–15% | Slightly higher (concentrated portfolio) |
| **Annual Turnover** | 69.98% | N/A | Moderate rebalancing |

### Strategy Parameters
- **Signal**: Transformer predicted expected return (Gaussian mean)
- **Stock Selection**: Top 5 by predicted return each rebalance
- **Max Position**: 20% per stock (avoid concentration risk)
- **Rebalance Freq**: Every 5 trading days
- **Initial Capital**: ₹10,00,000
- **Costs**: Realistic Indian market execution costs

---

## Regime Detection

4 market regimes identified via KMeans clustering + Gaussian HMM:

| Regime | Label | Mean Daily Return | Daily Volatility | Frequency | Avg Duration |
|--------|-------|-------------------|------------------|-----------|-------------|
| 0 | **Bear** | -1.15% | 1.92% | 22.6% | 1.5 days |
| 1 | **High-Vol Bull** | +1.01% | 2.06% | 25.2% | 2.0 days |
| 2 | **Transition** | +0.16% | 1.55% | 51.4% | 4.0 days |
| 3 | **Rare Transition** | +0.91% | 5.50% | 0.8% | 15.0 days |

### Strategy Performance by Regime (Backtest Period)

| Regime | Days | % of Time | Ann. Return | Volatility | Sharpe | Max DD |
|--------|------|-----------|-------------|------------|--------|--------|
| 0 (Bear) | 99 | 27% | **48.81%** | 17.56% | **2.78** | -6.37% |
| 1 (Bull) | 111 | 30% | **51.16%** | 18.92% | **2.70** | -8.54% |
| 2 (Transition) | 157 | 43% | 1.95% | 16.87% | 0.12 | -14.42% |

**Key Insight**: The strategy excels in trending markets (both Bear and Bull, Sharpe ~2.7) but struggles in sideways/transition periods (Sharpe 0.12). Most of the -11.54% max drawdown comes from transition regimes. This is actionable — regime-conditional position sizing could improve results further.

---

## HTML Backtest Report

Interactive report at `outputs/backtests/report.html` contains:

1. **Equity Curve** — Portfolio value over time with regime-colored backgrounds
2. **Drawdown Chart** — Underwater plot showing loss from peak
3. **Rolling Returns** — Time-varying return visualization
4. **Position Weights** — Top-10 asset allocation history
5. **Regime Summary Table** — Cluster characteristics
6. **Regime Performance Table** — Strategy metrics per regime

All charts are interactive (Plotly) — hover for details, zoom, pan.

---

## Output Artifacts

```
outputs/
├── models/
│   ├── pretrained/masked_encoder.pt         23.8 MB   (BERT-like encoder)
│   ├── transformer/{best,final,last}.pt     59/21/59 MB (Transformer forecaster)
│   ├── tft/{best,final,last}.pt             427/142/427 MB (TFT forecaster)
│   └── rl/
│       ├── ppo/ppo_agent.zip                1.4 MB    (PPO portfolio agent)
│       └── sac/sac_agent.zip                12.5 MB   (SAC portfolio agent)
├── backtests/report.html                    234 KB    (Interactive HTML report)
└── regimes/
    ├── regime_labels.parquet                30 KB     (3,695 daily labels)
    └── regime_summary.parquet               4 KB      (Regime characteristics)
```

---

## Infrastructure

| Component | Details |
|-----------|---------|
| Tests | 362 passing (pytest) |
| Config | Hydra-based, 13 YAML files |
| Tracking | MLflow experiment tracking |
| Cloud | Colab Pro+ (H100/A100), 3 parallel notebooks |
| Local | RTX 4080 12GB, 32GB RAM |
| Data | All intermediates persisted as Parquet |

# AI Quant Research Lab - Master Implementation Plan

> Institution-grade Quantitative Finance Research Infrastructure
> Target: NIFTY 50 → NIFTY 500 → Full Indian Stock Market
> Hardware: RTX 4080 (12GB) local + Colab Pro+ (H100/A100/TPU)

---

## Table of Contents

1. [Vision & Goals](#vision--goals)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Data Flow](#data-flow)
5. [Tech Stack](#tech-stack)
6. [Phase 1: Foundation & End-to-End Skeleton](#phase-1-foundation--end-to-end-skeleton) ✅ COMPLETE
7. [Phase 2: Transformer Forecasting](#phase-2-transformer-forecasting) ✅ COMPLETE
8. [Phase 3: Feature Expansion + Representation Learning](#phase-3-feature-expansion--representation-learning)
9. [Phase 4: Temporal Fusion Transformer](#phase-4-temporal-fusion-transformer)
10. [Phase 5: RL Portfolio Allocation](#phase-5-rl-portfolio-allocation)
11. [Phase 6: Regime Detection + Reporting](#phase-6-regime-detection--reporting)
12. [Key Architectural Decisions](#key-architectural-decisions)
13. [Verification Plan](#verification-plan)

---

## Vision & Goals

Build a **modular, config-driven, reproducible** quantitative research platform that spans the full pipeline:

1. **Data Ingestion** — Multi-source data acquisition with quality validation
2. **Feature Engineering** — Multi-horizon price, cross-asset, and regime features
3. **Representation Learning** — Self-supervised pre-training (BERT-like for time series)
4. **Forecasting** — Transformer and TFT models with multi-task prediction
5. **Portfolio Allocation** — RL-based dynamic allocation (PPO/SAC)
6. **Regime Detection** — Unsupervised market regime identification
7. **Backtesting** — Realistic simulation with transaction costs and lookahead protection
8. **Experiment Tracking** — MLflow/W&B for full reproducibility

**Long-term target**: A research infrastructure that can be maintained and extended over 3-5 years, supporting systematic research into deep learning for Indian equity markets.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CONFIG LAYER (Hydra)                        │
│  configs/config.yaml → data/ features/ model/ backtest/ experiment/ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: DATA INGESTION & CLEANING                │
│                                                                      │
│  yfinance API ───► raw/*.parquet ───► CleaningPipeline ───►          │
│  CSV feeds ──┘     (OHLCV)           (validate, ffill,    cleaned/   │
│                                       cap outliers)       *.parquet  │
│                                                                      │
│  Key classes: YFinanceSource, CSVSource, CleaningPipeline,           │
│               ParquetStore, Universe (NIFTY 50/500)                  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: FEATURE ENGINEERING                       │
│                                                                      │
│  @register_feature decorator ───► FeatureEngine.compute() ───►       │
│                                                                      │
│  Price features:  log_returns, realized_volatility, momentum,        │
│                   max_drawdown                                       │
│  Cross-asset:     rolling_correlation, rolling_beta,                 │
│                   relative_strength, cross_sectional_rank            │
│  Regime signals:  vol_regime, volume_shock, gap_stats,               │
│                   breadth_indicators                                 │
│                                                                      │
│  Output: features/*.parquet (per-ticker, all features)               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼──────────────────┐
                    ▼             ▼                   ▼
┌──────────────────────┐ ┌──────────────┐ ┌─────────────────────┐
│ LAYER 3: REPR LEARN  │ │ LAYER 4:     │ │ LAYER 6: REGIME     │
│                      │ │ FORECASTING  │ │                     │
│ Masked time-series   │ │              │ │ K-Means/GMM/HDBSCAN │
│ encoder (BERT-like)  │ │ Transformer  │ │ HMM transitions     │
│ Patch tokenization   │ │ TFT          │ │ Interpretable labels│
│ MSE reconstruction   │ │ Multi-task   │ │ (Bull/Bear/Crisis)  │
│                      │ │ heads        │ │                     │
│ Output: embeddings/  │ │              │ │ Output: regime_labels│
│ *.parquet            │ │              │ │                     │
└──────────┬───────────┘ └──────┬───────┘ └──────────┬──────────┘
           │                    │                     │
           └────────┬───────────┘                     │
                    ▼                                 │
┌─────────────────────────────────────────────────────┤
│              LAYER 5: RL PORTFOLIO ALLOCATION        │
│                                                      │
│  Gymnasium PortfolioEnv                              │
│  Obs = [embeddings, forecasts, current_weights]      │
│  Action = target weight vector                       │
│  Reward = Sharpe - λ*MaxDD - costs - turnover        │
│                                                      │
│  Agents: PPO, SAC (via Stable-Baselines3)            │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 7: REALISTIC BACKTESTING                     │
│                                                                      │
│  BacktestEngine (day-by-day simulation)                              │
│  ExecutionModel (commission 10bps + slippage 5bps + spread 5bps)     │
│  LookaheadGuard (prevent future data leaks)                          │
│  Metrics: CAGR, Sharpe, Sortino, MaxDD, Calmar, Turnover            │
│  Regime-conditional performance breakdown                            │
│                                                                      │
│  Output: HTML report + MLflow artifacts                              │
└─────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 8: EXPERIMENT TRACKING                       │
│                                                                      │
│  MLflow (primary): params, metrics, artifacts, model registry        │
│  W&B (optional): rich dashboards, hyperparameter sweeps              │
│  ArtifactManager: checkpoint versioning, config snapshots            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Recommendation_System/
├── PLAN.md                              # This file
├── pyproject.toml                       # Build config, ruff, pytest, mypy settings
├── requirements.txt                     # Core production dependencies
├── requirements-dev.txt                 # Dev tools (pytest, ruff, mypy, pre-commit)
├── Makefile                             # Common commands (install, lint, test, etc.)
├── .gitignore                           # data/, outputs/, __pycache__, .env, mlruns/
├── .env.example                         # Environment variable template
├── .pre-commit-config.yaml              # Ruff lint + format hooks
│
├── configs/                             # Hydra-compatible YAML configs
│   ├── config.yaml                      # Master config (composes all sub-configs)
│   ├── data/
│   │   ├── default.yaml                 # Default: NIFTY 50, 2010-2024, split dates
│   │   ├── nifty50.yaml                 # 50 large-cap Indian equities
│   │   ├── nifty500.yaml                # 500 Indian equities (Phase 2+)
│   │   └── indian_market.yaml           # Full Indian universe (future)
│   ├── features/
│   │   └── default.yaml                 # Enabled features, windows, normalization
│   ├── model/
│   │   ├── transformer.yaml             # Vanilla transformer architecture
│   │   └── tft.yaml                     # Temporal Fusion Transformer
│   ├── representation/
│   │   └── masked_ts.yaml               # Masked time-series pre-training
│   ├── rl/
│   │   ├── ppo.yaml                     # PPO hyperparameters
│   │   └── sac.yaml                     # SAC hyperparameters
│   ├── backtest/
│   │   └── default.yaml                 # Execution costs, portfolio constraints
│   └── experiment/
│       └── default.yaml                 # MLflow/W&B tracking settings
│
├── src/quant_lab/                       # Main Python package
│   ├── __init__.py
│   │
│   ├── data/                            # Layer 1: Data Ingestion & Cleaning
│   │   ├── __init__.py
│   │   ├── sources/
│   │   │   ├── __init__.py
│   │   │   ├── base_source.py           # Abstract BaseDataSource (fetch, validate_schema)
│   │   │   ├── yfinance_source.py       # YFinanceSource with retry, batch download
│   │   │   └── csv_source.py            # CSVSource for professional data feeds
│   │   ├── cleaning/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py              # CleaningPipeline + CleaningConfig dataclass
│   │   │   ├── validators.py            # validate_ohlc, validate_positive_prices, etc.
│   │   │   └── transformers.py          # forward_fill, cap_outliers, remove_low_history
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── parquet_store.py         # ParquetStore (save/load/exists/list)
│   │   │   └── cache.py                 # LRU parquet cache with mtime invalidation
│   │   ├── datasets.py                  # TimeSeriesDataset, TemporalSplit, create_flat_datasets
│   │   ├── datamodule.py                # QuantDataModule (wraps datasets into DataLoaders)
│   │   └── universe.py                  # NIFTY50_TICKERS, Universe dataclass, get_universe()
│   │
│   ├── features/                        # Layer 2: Feature Engineering
│   │   ├── __init__.py
│   │   ├── registry.py                  # FEATURE_REGISTRY dict + @register_feature decorator
│   │   ├── engine.py                    # FeatureEngine (compute, normalize, get_feature_columns)
│   │   ├── price_features.py            # log_returns, realized_volatility, momentum, max_drawdown
│   │   ├── cross_asset_features.py      # rolling_correlation, rolling_beta, relative_strength
│   │   ├── regime_features.py           # vol_regime, volume_shock, gap_stats, breadth
│   │   └── feature_store.py             # FeatureStore (save/load features to parquet)
│   │
│   ├── representation/                  # Layer 3: Self-Supervised Pre-Training
│   │   ├── __init__.py
│   │   ├── masked_encoder.py            # MaskedTimeSeriesEncoder (BERT-like)
│   │   ├── embedding_space.py           # Extract & store market embeddings
│   │   ├── pretraining.py               # Pre-training loop (MSE reconstruction loss)
│   │   └── tokenizer.py                 # PatchTokenizer (non-overlapping patches)
│   │
│   ├── models/                          # Layer 4: Forecasting
│   │   ├── __init__.py
│   │   ├── base_model.py                # Abstract BaseForecaster (fit/predict/save/load/evaluate)
│   │   ├── linear_baseline.py           # RidgeBaseline (sklearn Ridge + StandardScaler)
│   │   ├── transformer/
│   │   │   ├── __init__.py              # Exports TransformerForecaster, TransformerConfig
│   │   │   ├── attention.py             # MultiHeadSelfAttention (SDPA), PositionalEncoding
│   │   │   ├── encoder.py              # TransformerEncoderLayer (pre-norm), TransformerEncoder
│   │   │   ├── decoder.py              # ForecastDecoder (cls/last/mean pooling)
│   │   │   └── model.py                # TransformerForecaster, TransformerConfig, MultiTaskLoss
│   │   ├── tft/
│   │   │   ├── __init__.py
│   │   │   ├── gated_residual.py        # GatedResidualNetwork
│   │   │   ├── variable_selection.py    # VariableSelectionNetwork
│   │   │   └── model.py                # TemporalFusionTransformer
│   │   └── heads/
│   │       ├── __init__.py              # Exports all heads
│   │       ├── distribution_head.py     # GaussianHead, StudentTHead
│   │       ├── volatility_head.py       # VolatilityHead (Softplus output)
│   │       └── direction_head.py        # DirectionHead (3-class: down/flat/up)
│   │
│   ├── rl/                              # Layer 5: Portfolio Allocation
│   │   ├── __init__.py
│   │   ├── environments/
│   │   │   ├── __init__.py
│   │   │   ├── portfolio_env.py         # PortfolioEnv(gymnasium.Env)
│   │   │   ├── reward.py                # RewardFunction (Sharpe-based composite)
│   │   │   └── action_space.py          # WeightConstraints, cash allocation
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── ppo_agent.py             # PPOAgent (SB3 wrapper)
│   │   │   └── sac_agent.py             # SACAgent (SB3 wrapper)
│   │   └── training.py                  # RL training loop with evaluation
│   │
│   ├── regime/                          # Layer 6: Regime Detection
│   │   ├── __init__.py
│   │   ├── detector.py                  # RegimeDetector orchestrator
│   │   ├── clustering.py                # ClusteringModel (KMeans/GMM/HDBSCAN)
│   │   ├── hmm.py                       # HMMRegimeModel (hmmlearn wrapper)
│   │   └── labels.py                    # RegimeLabeler (map clusters to names)
│   │
│   ├── backtest/                        # Layer 7: Realistic Backtesting
│   │   ├── __init__.py
│   │   ├── engine.py                    # BacktestEngine, BacktestConfig, BacktestResult
│   │   ├── execution.py                 # ExecutionModel (commission/slippage/spread)
│   │   ├── metrics.py                   # compute_cagr/sharpe/sortino/maxdd/calmar/turnover
│   │   ├── lookahead_guard.py           # LookaheadGuard, assert_no_lookahead
│   │   └── report.py                    # HTMLReportGenerator (Plotly + Jinja2)
│   │
│   ├── tracking/                        # Layer 8: Experiment Tracking
│   │   ├── __init__.py
│   │   ├── base_tracker.py              # Abstract BaseTracker interface
│   │   ├── mlflow_tracker.py            # MLflowTracker implementation
│   │   ├── wandb_tracker.py             # WandBTracker (optional)
│   │   └── artifact_manager.py          # ArtifactManager (checkpoint versioning)
│   │
│   ├── training/                        # Training Orchestration
│   │   ├── __init__.py
│   │   ├── trainer.py                   # Trainer (AMP, grad clip, logging, tracker)
│   │   ├── callbacks.py                 # EarlyStopping, ModelCheckpoint
│   │   └── schedulers.py                # cosine_warmup_scheduler, linear_warmup_scheduler
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seed.py                      # set_global_seed (Python/NumPy/PyTorch/CUDA)
│       ├── device.py                    # get_device, get_dtype, get_device_info
│       ├── logging.py                   # setup_logging (structlog)
│       ├── timer.py                     # @timed decorator, timer() context manager
│       └── math_utils.py               # log_returns, simple_returns, rolling_volatility, etc.
│
├── scripts/                             # CLI entry points (all Hydra-enabled)
│   ├── ingest_data.py                   # Download + clean data
│   ├── compute_features.py              # Compute + normalize features
│   ├── pretrain.py                      # Pre-train masked encoder
│   ├── train_forecaster.py              # Train transformer/TFT
│   ├── train_rl.py                      # Train RL agent
│   ├── detect_regimes.py                # Run regime detection
│   ├── run_backtest.py                  # Run backtest with saved model
│   ├── run_pipeline.py                  # End-to-end orchestration
│   └── generate_report.py              # Generate HTML backtest report
│
├── notebooks/                           # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_embedding_visualization.ipynb
│   ├── 04_model_analysis.ipynb
│   └── 05_backtest_analysis.ipynb
│
├── tests/                               # Test suite (mirrors src structure)
│   ├── conftest.py                      # Shared fixtures (synthetic OHLCV, features)
│   ├── test_data/                       # Tests for data layer
│   ├── test_features/                   # Tests for feature engine
│   ├── test_models/                     # Tests for transformer, TFT, heads
│   ├── test_training/                   # Tests for trainer, callbacks, schedulers
│   ├── test_rl/                         # Tests for RL env and agents
│   ├── test_backtest/                   # Tests for backtest engine, metrics
│   └── test_integration/               # End-to-end integration tests
│
├── data/                                # Gitignored, local artifacts only
│   ├── raw/                             # Raw OHLCV parquet files
│   ├── cleaned/                         # Cleaned parquet files
│   ├── features/                        # Feature parquet files
│   └── embeddings/                      # Pre-trained embeddings
│
└── outputs/                             # Gitignored, experiment outputs
    ├── models/                          # Saved model checkpoints
    ├── backtests/                       # Backtest results and reports
    ├── mlruns/                          # MLflow tracking data
    └── logs/                            # Structured log files
```

---

## Data Flow

```
yfinance API ──► data/raw/*.parquet
                      │
              cleaning/pipeline.py
              (validate OHLC, ffill gaps,
               cap outliers, remove short history)
                      │
                data/cleaned/*.parquet
                      │
              features/engine.py
              (price + cross-asset + regime features,
               rolling z-score normalization)
                      │
                data/features/*.parquet
                      │
         ┌────────────┼────────────────────┐
         │            │                    │
  representation/   datasets.py         regime/
  masked_encoder     (sliding windows,   detector
  (pre-train on      temporal splits)    (clustering on
   all data)          │                   embeddings)
         │            │                    │
  data/embeddings/  Transformer/TFT     Regime labels
  *.parquet          (multi-task          │
         │            prediction)         │
         └──────┬─────┘                    │
                │                          │
         rl/portfolio_env                  │
         (state = embeddings +             │
          forecasts + current weights)     │
                │                          │
         backtest/engine.py ◄──────────────┘
         (regime-conditional analysis)
                │
         outputs/backtests/report.html
         tracking/mlflow (all metrics logged)
```

**Key design principle**: Every intermediate artifact persists to Parquet. Any stage can rerun independently without recomputing upstream. This enables rapid experimentation.

---

## Tech Stack

| Category | Choice | Rationale |
|----------|--------|-----------|
| **DL Framework** | PyTorch 2.2+ | Native Flash Attention via SDPA, BF16, compile() |
| **Config** | Hydra + OmegaConf | YAML composition, CLI overrides, multirun sweeps |
| **Data Source** | yfinance → Parquet (pyarrow) | Free data, Parquet preserves dtypes, 10-50x compression |
| **RL** | Stable-Baselines3 + Gymnasium | Battle-tested PPO/SAC, standard env interface |
| **Tracking** | MLflow (primary), W&B (optional) | Free, local-first, model registry, artifact versioning |
| **Clustering** | scikit-learn + hdbscan + hmmlearn | Standard, well-documented libraries |
| **Reporting** | Plotly + Jinja2 | Interactive HTML charts with drill-down |
| **Logging** | structlog | Structured, machine-parseable JSON logs |
| **Validation** | pydantic | Runtime config validation at system boundaries |
| **Linting** | ruff | Fast Python linter + formatter (replaces flake8+black+isort) |
| **Testing** | pytest + pytest-cov + pytest-xdist | Parallel test execution, coverage tracking |

---

## Phase 1: Foundation & End-to-End Skeleton ✅ COMPLETE

**Goal**: Working pipeline from data download through backtest with a linear baseline model. Validates all infrastructure without GPU.

**Status**: Complete. 53 tests passing. Initial commit made.

### What Was Built

#### 1.1 Project Scaffolding
- `pyproject.toml` — Build config (setuptools), ruff linting (E/W/F/I/N/UP/B/SIM/TCH), pytest markers (slow/integration/gpu), mypy config
- `requirements.txt` — Core deps: numpy, pandas, scipy, torch>=2.2.0, yfinance, pyarrow, hydra-core, mlflow, scikit-learn, structlog, pydantic, rich
- `requirements-dev.txt` — Dev deps: pytest, pytest-cov, pytest-xdist, ruff, mypy, pre-commit
- `Makefile` — Commands: install, install-dev, lint, format, test, test-unit, test-integration, ingest, features, pipeline, mlflow, clean
- `.gitignore` — Ignores data/, outputs/, __pycache__, .venv, .env, mlruns/, .claude/
- `.pre-commit-config.yaml` — Ruff lint (--fix) + ruff-format hooks
- `.env.example` — Template for environment variables

#### 1.2 Hydra Configuration (10 YAML files)
- `configs/config.yaml` — Master config composing data/features/model/backtest/experiment
- `configs/data/default.yaml` — NIFTY 50 universe, date range 2010-2024, train/val/test split dates, cleaning params
- `configs/features/default.yaml` — Enabled features (log_returns, realized_volatility, momentum, max_drawdown), window sizes (short: [1,5], medium: [21,63], long: [126,252]), normalization (rolling_zscore, lookback=252)
- `configs/model/transformer.yaml` — d_model=128, nhead=8, 4 layers, dim_ff=512, GELU, pre-norm, sequence_length=63, Gaussian/direction/volatility heads, training hyperparams
- `configs/model/tft.yaml` — TFT architecture with GRN, LSTM, variable selection
- `configs/backtest/default.yaml` — Execution costs (commission 10bps, slippage 5bps, spread 5bps), portfolio constraints (top_n=5, max_position=0.20, min_position=0.02)
- `configs/experiment/default.yaml` — MLflow tracking URI, experiment name
- `configs/representation/masked_ts.yaml` — Patch size 5, mask ratio 15%, d_model 128
- `configs/rl/ppo.yaml` + `configs/rl/sac.yaml` — RL hyperparameters

#### 1.3 Utils Module
- `utils/seed.py` — `set_global_seed(seed)`: seeds Python random, NumPy, PyTorch CPU/CUDA, sets PYTHONHASHSEED, enables deterministic cuDNN
- `utils/device.py` — `get_device()`: auto-detects CUDA/MPS/CPU; `get_dtype()`: selects BF16 for Ampere+, FP16 for older; `get_device_info()`: returns dict of hardware details
- `utils/logging.py` — `setup_logging()`: configures structlog with console renderer, dev-friendly output
- `utils/timer.py` — `@timed` decorator for function timing, `timer()` context manager
- `utils/math_utils.py` — `log_returns()`, `simple_returns()`, `rolling_volatility()`, `rolling_max_drawdown()`, `rolling_sharpe()`

#### 1.4 Data Layer
- **Sources**: `BaseDataSource` (ABC with fetch/validate_schema), `YFinanceSource` (batch download with retry, reshape multi-ticker format to long DataFrame), `CSVSource` (load from CSV files)
- **Cleaning**: `CleaningConfig` (max_missing_pct=0.20, ffill_limit=5, outlier_sigma=10.0, min_history_days=252), `CleaningPipeline` (validate positive prices → validate OHLC → forward fill → cap outliers → remove high-missing tickers → remove low-history tickers)
- **Validators**: `validate_ohlc_relationships()`, `validate_positive_prices()`, `validate_volume()`, `check_missing_rate()`
- **Transformers**: `forward_fill_missing()`, `cap_outliers()`, `remove_low_history_tickers()`, `remove_high_missing_tickers()`
- **Storage**: `ParquetStore` (save/load/exists/list_files), LRU `cached_parquet_load()` with file mtime invalidation
- **Universe**: `NIFTY50_TICKERS` (50 `.NS`-suffixed tickers), `Universe` dataclass, `get_universe(name)` function
- **Datasets**: `TemporalSplit` dataclass (train_end, val_end), `TimeSeriesDataset` (sliding window, NaN filtering, returns (seq_len, num_features) tensor + target dict), `create_temporal_splits()` (per-ticker datasets), `create_flat_datasets()` (flat arrays for sklearn)

#### 1.5 Feature Engine
- **Registry**: `FEATURE_REGISTRY` dict + `@register_feature(name, description)` decorator + `get_feature_func()`, `list_features()`
- **Engine**: `FeatureEngine` class with `compute()` (runs registered features per-ticker), `normalize()` (rolling z-score or cross-sectional rank), `get_feature_columns()` (auto-detects)
- **Price Features** (4 registered):
  - `log_returns` — Multi-horizon log returns (windows: [1, 5, 21, 63, 126, 252])
  - `realized_volatility` — Rolling annualized volatility (guard: window >= 2, min_periods >= 2)
  - `momentum` — Price momentum via pct_change
  - `max_drawdown` — Rolling max drawdown with custom helper function
- **Feature Store**: `FeatureStore` wrapping `ParquetStore` for feature persistence

#### 1.6 Models
- **BaseForecaster** (ABC): fit(), predict(), save(), load(), evaluate() (MSE, MAE, direction_accuracy, IC via Spearman rank)
- **RidgeBaseline**: sklearn Ridge + StandardScaler, NaN filtering, feature importance via coefficients

#### 1.7 Backtest Engine
- **ExecutionModel**: commission_bps=10, slippage_bps=5, spread_bps=5, execution_delay_bars=1, `compute_trade_cost(turnover)`
- **BacktestEngine**: day-by-day simulation with wide-format pivot, execution delay, top-N equal-weight signals, max position cap
- **BacktestConfig**: initial_capital, rebalance_frequency, max_position_size, top_n, risk_free_rate
- **BacktestResult**: equity_curve, returns, weights_history, metrics dict, trades list
- **Metrics**: compute_cagr, compute_sharpe, compute_sortino, compute_max_drawdown, compute_calmar, compute_turnover, compute_annual_turnover, compute_all_metrics
- **LookaheadGuard**: wrapper class + assert_no_lookahead() function with LookaheadError exception

#### 1.8 Tracking
- **BaseTracker** (ABC): start_run, end_run, log_params, log_metrics, log_artifact, log_config (flattens nested dicts)
- **MLflowTracker**: Full MLflow integration (experiment creation, param/metric/artifact logging)

#### 1.9 Scripts
- `scripts/ingest_data.py` — Hydra CLI: download via yfinance → clean → save parquet
- `scripts/compute_features.py` — Hydra CLI: load cleaned data → compute features → normalize → save parquet
- `scripts/run_pipeline.py` — End-to-end: ingest → features → split → train Ridge → backtest → display → MLflow log

#### 1.10 Tests (53 tests)
- `test_cleaning.py` — 9 tests (validators, transformers, pipeline)
- `test_datasets.py` — 5 tests (TimeSeriesDataset, temporal splits, shapes)
- `test_price_features.py` — 8 tests (log returns, volatility, momentum, drawdown)
- `test_registry.py` — 5 tests (registration, lookup, listing)
- `test_metrics.py` — 12 tests (CAGR, Sharpe, Sortino, MaxDD, Calmar, turnover)
- `test_execution.py` — 5 tests (default costs, trade cost computation)
- `test_engine.py` — 5 tests (backtest runs, initial capital, metrics, cost comparison)
- `test_end_to_end.py` — 1 integration test (full pipeline with synthetic data)
- `conftest.py` — Shared fixtures: set_seed, synthetic_ohlcv (504 bdays, 5 tickers), synthetic_features, tiny_model_config

**Exit Criteria**: ✅ `python scripts/run_pipeline.py` works end-to-end. 53/53 tests pass.

---

## Phase 2: Transformer Forecasting ✅ COMPLETE

**Goal**: Multi-task transformer model with mixed precision training, replacing the Ridge baseline.

**Status**: Complete. 114 tests passing (53 Phase 1 + 61 Phase 2).

### What Was Built

#### 2.1 Attention Module (`models/transformer/attention.py`)
- **PositionalEncoding**: Sinusoidal positional encoding (Vaswani et al., 2017). Pre-computed buffer of shape (1, max_len, d_model), added to input with dropout.
- **MultiHeadSelfAttention**: Uses `F.scaled_dot_product_attention` (PyTorch 2.x SDPA) which automatically selects Flash Attention, Memory-Efficient Attention, or math-based attention depending on hardware and input size. Single QKV projection (3x faster than separate), output projection.

#### 2.2 Encoder (`models/transformer/encoder.py`)
- **TransformerEncoderLayer**: Pre-norm architecture (LN → Attn → +residual → LN → FFN → +residual). More stable training than post-norm, especially for deeper models. FFN: Linear → GELU → Dropout → Linear → Dropout.
- **TransformerEncoder**: Input projection (num_features → d_model), sinusoidal positional encoding, learnable [CLS] token prepended to sequence, N stacked encoder layers, final LayerNorm. Output: (batch, seq_len+1, d_model).

#### 2.3 Decoder (`models/transformer/decoder.py`)
- **ForecastDecoder**: Extracts fixed-size representation from encoder output. Three pooling strategies: `cls` (CLS token at position 0, default), `last` (last time step), `mean` (mean over time steps excluding CLS). Optional projection layer.

#### 2.4 Prediction Heads (`models/heads/`)
- **GaussianHead**: Predicts mean + log_variance for Gaussian return distribution. Optional hidden layer.
- **StudentTHead**: Predicts location, log_scale, log_df for heavy-tailed Student-t distribution. Ensures df > 2 for finite variance.
- **DirectionHead**: 3-class classification (down=0, flat=1, up=2). Raw logits output (softmax in loss).
- **VolatilityHead**: Positive scalar output via Softplus activation. Target: absolute return |r| as volatility proxy.

#### 2.5 Full Model (`models/transformer/model.py`)
- **TransformerConfig**: Dataclass with all architecture, head, and loss weight parameters. `from_hydra()` classmethod for Hydra config integration.
- **TransformerForecaster**: Composes encoder + decoder + heads. `forward()` returns dict of head outputs. `predict_returns()` convenience method. `save()`/`load()` with config persistence. `count_parameters()`.
- **MultiTaskLoss**: Weighted combination of Gaussian NLL (distribution), cross-entropy (direction), MSE (volatility). Direction labels computed on-the-fly from returns using configurable threshold. No dataset changes needed. Student-t NLL supported.

**Model size at default config**: ~790K parameters (d_model=128, 4 layers, 8 heads, dim_ff=512). Fits easily on RTX 4080.

#### 2.6 Training Infrastructure (`training/`)
- **Trainer**: Full training loop with:
  - Mixed precision (BF16 on Ampere+, FP16 fallback, disabled on CPU)
  - GradScaler for FP16 stability
  - Gradient clipping (max_grad_norm=1.0)
  - Cosine warmup LR schedule
  - Per-step and per-epoch structured logging
  - MLflow tracker integration (optional)
  - Early stopping + model checkpointing
- **EarlyStopping**: Configurable patience, min_delta, min/max mode. Tracks best score and counter.
- **ModelCheckpoint**: Saves best.pt and last.pt. Includes model state, optimizer state, epoch, config.
- **cosine_warmup_scheduler**: Linear warmup from 0 → LR, then cosine decay to min_lr_ratio * LR.
- **linear_warmup_scheduler**: Linear warmup, then constant LR.

#### 2.7 DataModule (`data/datamodule.py`)
- **QuantDataModule**: Wraps feature DataFrame + temporal split into train/val/test DataLoaders. Uses `create_temporal_splits()` to create per-ticker `TimeSeriesDataset`s, then `ConcatDataset` to merge. Configurable batch_size, num_workers, pin_memory.

#### 2.8 Training Script (`scripts/train_forecaster.py`)
- Hydra CLI with full config override support
- Loads feature parquet, auto-detects feature columns
- Builds TransformerConfig from Hydra config
- Sets up MLflow tracking (optional)
- Runs training with Trainer
- Saves final model + prints summary

#### 2.9 Tests (61 new, 114 total)
- `test_attention.py` — 8 tests: PE shape/content, MHSA shape/batch independence/causal mask/gradient flow
- `test_encoder.py` — 7 tests: layer shape, CLS token, variable features, gradient flow, deterministic eval
- `test_heads.py` — 11 tests: Gaussian/StudentT/Direction/Volatility output shapes and properties
- `test_transformer_model.py` — 13 tests: forward, shapes, predict_returns, save/load, variable seq lengths, loss computation, direction labels, loss decreases with training, Student-t loss
- `test_callbacks.py` — 9 tests: early stopping (improve/patience/reset/max mode/min delta), checkpointing (best/last/improvement/optimizer state)
- `test_schedulers.py` — 5 tests: warmup start/peak/decay, non-negative LR, linear warmup constant after
- `test_trainer.py` — 4 tests: fit runs, loss decreases, predict, early stopping triggers
- `test_datamodule.py` — 4 tests: setup creates loaders, num_features, batch shape, temporal safety

**Exit Criteria**: ✅ All 114 tests pass. Transformer model builds, trains, and produces predictions.

---

## Phase 3: Feature Expansion + Representation Learning

**Goal**: Expand the feature set with cross-asset and regime signals, then build a self-supervised pre-training system (BERT-like masked time-series encoder) that produces market embeddings.

### 3.1 Cross-Asset Features (`features/cross_asset_features.py`)

New registered features that capture inter-stock relationships:

#### `rolling_correlation`
- Compute pairwise rolling Pearson correlation between each stock's returns and the market benchmark (^NSEI)
- Windows: [21, 63, 126] days
- Output columns: `correlation_21d`, `correlation_63d`, `correlation_126d`
- Implementation: per-ticker rolling correlation against benchmark returns

#### `rolling_beta`
- Rolling CAPM beta: Cov(Ri, Rm) / Var(Rm) where Rm = benchmark returns
- Windows: [63, 126, 252] days
- Output columns: `beta_63d`, `beta_126d`, `beta_252d`
- Implementation: rolling covariance / rolling variance, both against benchmark

#### `relative_strength`
- Relative strength vs benchmark: stock_return / benchmark_return over rolling window
- Windows: [21, 63, 126] days
- Output columns: `rel_strength_21d`, `rel_strength_63d`, `rel_strength_126d`
- Implementation: ratio of cumulative returns over window

#### `cross_sectional_rank`
- Cross-sectional percentile rank of each stock's feature values at each date
- Applied to: returns, volatility, momentum
- Output columns: `rank_return_1d`, `rank_volatility_21d`, `rank_momentum_63d`
- Implementation: per-date groupby rank transformation, scaled to [0, 1]

### 3.2 Regime Features (`features/regime_features.py`)

Features that signal market regime changes:

#### `vol_regime`
- Rolling volatility relative to its own longer-term average
- Computation: vol_21d / vol_126d (short-term vol / long-term vol)
- High values (>1.5) indicate regime shift to high volatility
- Output: `vol_regime_ratio`

#### `volume_shock`
- Abnormal volume detection: current volume / rolling average volume
- Windows: [5, 21] day rolling average
- Output: `volume_shock_5d`, `volume_shock_21d`
- Values > 2.0 indicate abnormal volume events

#### `gap_stats`
- Overnight gap (open vs previous close): `gap = (open_t - close_{t-1}) / close_{t-1}`
- Rolling statistics of gaps: mean, std, max absolute gap
- Window: [21] days
- Output: `gap_mean_21d`, `gap_std_21d`, `gap_max_21d`

#### `breadth_indicators`
- Market breadth: what fraction of stocks are up on a given day
- Advance-decline ratio and line
- Rolling average of breadth
- Output: `market_breadth`, `adv_decline_ratio`, `breadth_ma_21d`
- Implementation: requires cross-sectional data (all tickers at each date)

### 3.3 Feature Engine Updates

- Update `FeatureEngine.compute()` to handle cross-asset features (need benchmark data)
- Add `benchmark_ticker` parameter to FeatureEngine
- Update `configs/features/default.yaml` with new feature groups
- Maintain backward compatibility (new features are opt-in via config)

### 3.4 Masked Time-Series Encoder (`representation/`)

Self-supervised pre-training inspired by BERT, adapted for continuous time series:

#### `tokenizer.py` — PatchTokenizer
- **Input**: (batch, seq_len, num_features) raw feature sequences
- **Patching**: Divide sequence into non-overlapping patches of size `patch_size` (default: 5 days = 1 trading week)
- **Output**: (batch, num_patches, patch_size * num_features) flattened patch tokens
- Learnable linear projection: (patch_size * num_features) → d_model
- Positional encoding for patch positions

```python
class PatchTokenizer(nn.Module):
    def __init__(self, num_features, patch_size=5, d_model=128):
        self.patch_proj = nn.Linear(patch_size * num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

    def forward(self, x):  # (B, T, F) -> (B, num_patches, d_model)
        patches = x.unfold(1, self.patch_size, self.patch_size)  # (B, num_patches, F, patch_size)
        patches = patches.permute(0, 1, 3, 2).flatten(2)  # (B, num_patches, patch_size * F)
        return self.pos_encoding(self.patch_proj(patches))
```

#### `masked_encoder.py` — MaskedTimeSeriesEncoder
- **Architecture**: Same TransformerEncoder as forecasting model, but with:
  - Patch tokenization instead of raw feature input
  - Random masking of 15-30% of patches (configurable `mask_ratio`)
  - Learnable [MASK] token replacing masked patches
  - Reconstruction head: Linear(d_model → patch_size * num_features)
- **Forward**: tokenize → mask → encode → reconstruct masked patches
- **Loss**: MSE between reconstructed and original patch values (only on masked positions)

```python
class MaskedTimeSeriesEncoder(nn.Module):
    def __init__(self, config):
        self.tokenizer = PatchTokenizer(...)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.encoder = TransformerEncoder(...)  # Reuse existing encoder
        self.reconstruction_head = nn.Linear(d_model, patch_size * num_features)

    def forward(self, x, mask_ratio=0.15):
        tokens = self.tokenizer(x)
        masked_tokens, mask = self._apply_mask(tokens, mask_ratio)
        encoded = self.encoder(masked_tokens)
        reconstructed = self.reconstruction_head(encoded)
        return reconstructed, mask  # Loss computed externally
```

#### `pretraining.py` — Pre-Training Loop
- Uses existing `Trainer` infrastructure with custom loss (MSE on masked patches)
- Pre-trains on ALL available data (no train/val split needed for self-supervised)
- Trains for 50-100 epochs on full NIFTY 50 history
- Saves encoder weights for transfer to forecaster
- Logs reconstruction loss to MLflow

#### `embedding_space.py` — Market Embedding Extraction
- Load pre-trained encoder (without reconstruction head)
- Run forward pass on all data to extract CLS token embeddings
- Store embeddings as Parquet: (date, ticker, embedding_0, ..., embedding_d_model)
- Embeddings serve as input to:
  - Regime detection (clustering)
  - RL portfolio agent (observation space)
  - Forecaster initialization (transfer learning)

### 3.5 Transfer Learning
- After pre-training, initialize the forecaster's encoder from pre-trained weights
- Fine-tune with lower learning rate for encoder, higher for heads
- Compare performance: random init vs pre-trained init

### 3.6 New Scripts
- `scripts/pretrain.py` — Pre-train masked encoder on full dataset
- Update `scripts/train_forecaster.py` — Add `--pretrained` flag to load pre-trained weights

### 3.7 New Tests
- `test_features/test_cross_asset_features.py` — Tests for correlation, beta, relative strength, rank
- `test_features/test_regime_features.py` — Tests for vol regime, volume shock, gap stats, breadth
- `test_representation/test_tokenizer.py` — Patch tokenization shapes, masking
- `test_representation/test_masked_encoder.py` — Forward pass, reconstruction loss, embedding extraction
- `test_representation/test_pretraining.py` — Pre-training loop runs, loss decreases

### 3.8 Exit Criteria
- All new features compute correctly on synthetic data
- Masked encoder pre-trains and produces embeddings
- Embeddings stored as Parquet
- Transfer learning pathway works (pre-trained → fine-tuned forecaster)
- All tests pass

---

## Phase 4: Temporal Fusion Transformer

**Goal**: Implement the Temporal Fusion Transformer (TFT) as an alternative forecasting architecture, offering interpretable attention and automatic feature selection.

### 4.1 Gated Residual Network (`models/tft/gated_residual.py`)

The core building block of TFT:

```python
class GatedResidualNetwork(nn.Module):
    """
    GRN: context-aware feature transformation with gating.

    Architecture:
        1. Linear(input, hidden) → ELU → Linear(hidden, hidden)
        2. If context provided: Linear(context, hidden) added to step 1
        3. GLU gate: sigmoid(Linear(hidden)) * Linear(hidden)
        4. LayerNorm(gate_output + residual)
    """
    def __init__(self, d_input, d_hidden, d_output, d_context=None, dropout=0.1):
        # Primary path
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        # Optional context injection
        self.context_proj = nn.Linear(d_context, d_hidden) if d_context else None
        # Gated Linear Unit
        self.gate = nn.Linear(d_hidden, d_output)
        self.gate_sigmoid = nn.Linear(d_hidden, d_output)
        # Residual + norm
        self.residual_proj = nn.Linear(d_input, d_output) if d_input != d_output else nn.Identity()
        self.layer_norm = nn.LayerNorm(d_output)

    def forward(self, x, context=None):
        hidden = F.elu(self.fc1(x))
        hidden = self.fc2(hidden)
        if context is not None:
            hidden = hidden + self.context_proj(context)
        gated = torch.sigmoid(self.gate_sigmoid(hidden)) * self.gate(hidden)
        return self.layer_norm(gated + self.residual_proj(x))
```

### 4.2 Variable Selection Network (`models/tft/variable_selection.py`)

Automatic feature importance weighting:

```python
class VariableSelectionNetwork(nn.Module):
    """
    VSN: learns which features are most relevant.

    For each input variable:
        1. Pass through individual GRN to get transformed representation
        2. Softmax weights across variables (via joint GRN on flattened input)
        3. Weighted sum of transformed variables

    Provides interpretable feature importance weights.
    """
    def __init__(self, d_input, num_vars, d_hidden, d_context=None, dropout=0.1):
        # Individual variable GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(d_input, d_hidden, d_hidden, dropout=dropout)
            for _ in range(num_vars)
        ])
        # Softmax weight GRN
        self.weight_grn = GatedResidualNetwork(
            d_input * num_vars, d_hidden, num_vars, d_context=d_context, dropout=dropout
        )

    def forward(self, inputs, context=None):
        # inputs: list of (batch, seq_len, d_input) tensors, one per variable
        transformed = [grn(inp) for grn, inp in zip(self.var_grns, inputs)]
        # Compute weights
        flattened = torch.cat(inputs, dim=-1)
        weights = F.softmax(self.weight_grn(flattened, context), dim=-1)
        # Weighted combination
        combined = sum(w.unsqueeze(-1) * t for w, t in zip(weights.unbind(-1), transformed))
        return combined, weights  # weights are interpretable
```

### 4.3 Full TFT Model (`models/tft/model.py`)

```python
class TemporalFusionTransformer(nn.Module):
    """
    Full TFT architecture:
        1. Variable Selection (input) — which features matter
        2. LSTM Encoder — sequential processing with gating
        3. Static Enrichment — inject static covariates (ticker identity)
        4. Temporal Self-Attention — multi-head attention with interpretable weights
        5. Position-wise Feedforward — final processing
        6. Prediction Heads — same multi-task heads as vanilla transformer
    """
    def __init__(self, config: TFTConfig):
        # Variable selection for time-varying known inputs
        self.variable_selection = VariableSelectionNetwork(...)
        # LSTM encoder
        self.lstm_encoder = nn.LSTM(d_model, d_model, num_layers, batch_first=True, dropout=dropout)
        # Static enrichment
        self.static_enrichment = GatedResidualNetwork(d_model, d_hidden, d_model, d_context=d_static)
        # Interpretable multi-head attention
        self.self_attention = InterpretableMultiHeadAttention(d_model, nhead)
        self.attn_gate = GatedResidualNetwork(d_model, d_hidden, d_model)
        # Position-wise feedforward
        self.positionwise_grn = GatedResidualNetwork(d_model, d_hidden, d_model)
        # Prediction heads (shared with vanilla transformer)
        self.heads = nn.ModuleDict({...})

    def forward(self, x, static_covariates=None):
        # 1. Variable selection
        selected, var_weights = self.variable_selection(x)
        # 2. LSTM encoding
        lstm_out, _ = self.lstm_encoder(selected)
        # 3. Static enrichment
        enriched = self.static_enrichment(lstm_out, static_covariates)
        # 4. Temporal attention
        attn_out, attn_weights = self.self_attention(enriched)
        attn_out = self.attn_gate(attn_out) + enriched
        # 5. Feedforward
        output = self.positionwise_grn(attn_out)
        # 6. Pool and predict
        pooled = output[:, -1]  # Last time step
        return {name: head(pooled) for name, head in self.heads.items()}, var_weights, attn_weights
```

### 4.4 Interpretable Multi-Head Attention
- Modified attention that exposes attention weights for interpretability
- Each head produces weights that can be averaged and visualized
- Shows which time steps the model attends to for its prediction

### 4.5 Config Integration
- Make `configs/config.yaml` support `model: transformer` or `model: tft` via Hydra defaults
- `scripts/train_forecaster.py` auto-selects model class based on `cfg.model.type`
- Both models share the same prediction heads and loss function

### 4.6 New Tests
- `test_models/test_gated_residual.py` — GRN forward, gating, context injection
- `test_models/test_variable_selection.py` — VSN weights sum to 1, variable importance
- `test_models/test_tft.py` — Full TFT forward, shapes, gradient flow, interpretability outputs
- Compare TFT vs vanilla transformer on synthetic data (both should train)

### 4.7 Exit Criteria
- TFT trains end-to-end with same training infrastructure as vanilla transformer
- Variable importance weights are interpretable
- Attention weights visualizable
- Config-switchable: `model=transformer` or `model=tft`
- All tests pass

---

## Phase 5: RL Portfolio Allocation

**Goal**: Build a reinforcement learning system that learns dynamic portfolio allocation using forecasts, embeddings, and market state as observations.

### 5.1 Portfolio Environment (`rl/environments/portfolio_env.py`)

Gymnasium-compatible environment:

```python
class PortfolioEnv(gymnasium.Env):
    """
    Observation space (Box):
        - Market embeddings: (num_assets, d_model)  [from pre-trained encoder]
        - Return forecasts: (num_assets,)            [from transformer/TFT]
        - Current weights: (num_assets + 1,)         [includes cash]
        - Regime features: (num_regime_features,)    [vol regime, etc.]

    Action space (Box):
        - Target portfolio weights: (num_assets + 1,)
        - Constrained: weights >= 0, sum = 1 (long-only, no leverage)

    State transitions:
        1. Agent outputs target weights
        2. Execute rebalancing (apply transaction costs)
        3. Advance one day: portfolio value changes based on actual returns
        4. Compute reward

    Episode:
        - One episode = one walk through the test period
        - Each step = one trading day
    """
    def __init__(self, config):
        self.prices = config.prices          # (T, N) daily prices
        self.forecasts = config.forecasts    # (T, N) return predictions
        self.embeddings = config.embeddings  # (T, N, D) market embeddings
        self.execution_model = ExecutionModel(...)

        # Spaces
        self.observation_space = gymnasium.spaces.Box(...)
        self.action_space = gymnasium.spaces.Box(low=0, high=1, shape=(num_assets + 1,))

    def reset(self, seed=None):
        self.current_step = 0
        self.current_weights = np.zeros(self.num_assets + 1)
        self.current_weights[-1] = 1.0  # Start 100% cash
        self.portfolio_value = self.initial_capital
        return self._get_observation(), {}

    def step(self, action):
        target_weights = self._normalize_weights(action)
        # Compute turnover and transaction costs
        turnover = np.abs(target_weights - self.current_weights).sum()
        cost = self.execution_model.compute_trade_cost(turnover)
        # Apply actual returns
        daily_returns = self._get_daily_returns()
        portfolio_return = np.dot(target_weights[:-1], daily_returns) - cost
        self.portfolio_value *= (1 + portfolio_return)
        self.current_weights = target_weights
        self.current_step += 1
        reward = self._compute_reward()
        done = self.current_step >= len(self.prices) - 1
        return self._get_observation(), reward, done, False, {}
```

### 5.2 Reward Function (`rl/environments/reward.py`)

```python
class RewardFunction:
    """
    Composite reward designed for financial portfolio management:

    reward = differential_sharpe
             - lambda_dd * drawdown_penalty
             - lambda_cost * transaction_cost
             - lambda_turnover * excess_turnover

    Components:
        1. Differential Sharpe Ratio: online Sharpe approximation
           (avoids need to compute over full episode)
        2. Drawdown penalty: penalizes being in drawdown
        3. Transaction cost: actual cost of rebalancing
        4. Turnover penalty: discourages excessive trading
    """
    def __init__(self, lambda_dd=0.5, lambda_cost=1.0, lambda_turnover=0.1):
        self.lambda_dd = lambda_dd
        self.lambda_cost = lambda_cost
        self.lambda_turnover = lambda_turnover
        # Running stats for differential Sharpe
        self.A_prev = 0.0  # EMA of returns
        self.B_prev = 0.0  # EMA of squared returns
        self.eta = 0.01    # EMA decay rate

    def compute(self, portfolio_return, cost, turnover, drawdown):
        # Differential Sharpe (Moody & Saffell, 2001)
        A = self.A_prev + self.eta * (portfolio_return - self.A_prev)
        B = self.B_prev + self.eta * (portfolio_return**2 - self.B_prev)
        denom = (B - A**2)**(3/2)
        if abs(denom) > 1e-10:
            dS = (B * (portfolio_return - A) - 0.5 * A * (portfolio_return**2 - B)) / denom
        else:
            dS = 0.0
        self.A_prev, self.B_prev = A, B

        reward = (dS
                  - self.lambda_dd * max(0, drawdown)
                  - self.lambda_cost * cost
                  - self.lambda_turnover * max(0, turnover - 0.1))  # Penalize > 10% daily turnover
        return reward
```

### 5.3 Action Space (`rl/environments/action_space.py`)

```python
class WeightConstraints:
    """
    Portfolio weight constraints:
        - Long only: all weights >= 0
        - Fully invested: weights sum to 1 (with cash as last asset)
        - Max position: no single stock > max_weight (e.g., 20%)
        - Min position: positions below min_weight are zeroed out
        - Cash allocation: always available as risk-free asset
    """
    def __init__(self, num_assets, max_weight=0.20, min_weight=0.02):
        ...

    def normalize(self, raw_action):
        """Convert raw RL output to valid portfolio weights."""
        weights = F.softmax(torch.tensor(raw_action), dim=0).numpy()
        weights = np.clip(weights, 0, self.max_weight)
        # Zero out tiny positions
        weights[weights[:-1] < self.min_weight] = 0  # Don't zero cash
        # Renormalize to sum to 1
        weights /= weights.sum()
        return weights
```

### 5.4 PPO Agent (`rl/agents/ppo_agent.py`)

```python
class PPOAgent:
    """
    PPO agent wrapper around Stable-Baselines3.

    Architecture:
        - Policy network: MLP [256, 256] with tanh activation
        - Value network: MLP [256, 256] separate from policy
        - Both receive the flattened observation

    Key hyperparameters (from configs/rl/ppo.yaml):
        - learning_rate: 3e-4
        - n_steps: 2048 (steps per update)
        - batch_size: 64
        - n_epochs: 10 (PPO epochs per update)
        - gamma: 0.99
        - gae_lambda: 0.95
        - clip_range: 0.2
        - ent_coef: 0.01 (entropy bonus for exploration)
    """
    def __init__(self, env, config):
        self.model = PPO(
            "MlpPolicy", env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            verbose=1,
        )

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action
```

### 5.5 SAC Agent (`rl/agents/sac_agent.py`)

```python
class SACAgent:
    """
    SAC agent for continuous action spaces (portfolio weights).

    SAC advantages over PPO for portfolio management:
        - Better sample efficiency
        - Entropy-regularized (automatically explores)
        - Off-policy (can reuse old data)

    Key hyperparameters (from configs/rl/sac.yaml):
        - learning_rate: 3e-4
        - buffer_size: 100000
        - batch_size: 256
        - tau: 0.005 (soft update coefficient)
        - gamma: 0.99
        - train_freq: 1
        - gradient_steps: 1
    """
    def __init__(self, env, config):
        self.model = SAC("MlpPolicy", env, ...)
```

### 5.6 RL Training Loop (`rl/training.py`)

```python
class RLTrainer:
    """
    Handles RL training with proper financial backtesting evaluation.

    Training procedure:
        1. Create train/val environments from different time periods
        2. Train agent on train environment
        3. Evaluate on val environment (full episode backtest)
        4. Compare against baselines:
           - Equal-weight portfolio
           - Signal-only portfolio (top-N by forecast, equal weight)
           - Buy-and-hold benchmark (NIFTY 50)
        5. Log metrics to MLflow
    """
    def train(self, agent, train_env, val_env, total_timesteps):
        for epoch in range(num_epochs):
            agent.train(steps_per_epoch)
            # Evaluate
            val_result = self.evaluate(agent, val_env)
            baseline_result = self.evaluate_baseline(val_env)
            # Log
            tracker.log_metrics({
                "rl_sharpe": val_result.sharpe,
                "baseline_sharpe": baseline_result.sharpe,
                "rl_cagr": val_result.cagr,
                ...
            })
```

### 5.7 New Script
- `scripts/train_rl.py` — Hydra CLI for RL training with env creation, agent selection, baseline comparison

### 5.8 New Tests
- `test_rl/test_portfolio_env.py` — Env reset/step, observation/action spaces, episode termination
- `test_rl/test_reward.py` — Differential Sharpe computation, penalty terms
- `test_rl/test_action_space.py` — Weight normalization, constraints
- `test_rl/test_agents.py` — PPO/SAC create and take actions (short training on simple env)

### 5.9 Exit Criteria
- Portfolio environment passes Gymnasium API checks
- PPO and SAC agents train without crashing
- RL agent beats equal-weight baseline on training period
- Metrics logged to MLflow (RL Sharpe vs baseline Sharpe)
- All tests pass

---

## Phase 6: Regime Detection + Reporting

**Goal**: Unsupervised market regime detection using embeddings, with interpretable labels. Generate comprehensive HTML backtest reports with regime-conditional performance breakdowns.

### 6.1 Clustering Models (`regime/clustering.py`)

```python
class ClusteringModel:
    """
    Flexible clustering on market embeddings.

    Supported algorithms:
        - KMeans: Fast, assumes spherical clusters. Good baseline.
        - GMM (Gaussian Mixture): Soft assignments, handles elliptical clusters.
        - HDBSCAN: No need to specify K, handles noise, density-based.

    Pipeline:
        1. Load embeddings from Parquet
        2. Optionally reduce dimensions (PCA/UMAP to 10-20 dims)
        3. Fit clustering model
        4. Assign regime labels to each (date, ticker) pair
    """
    def __init__(self, method="kmm", n_clusters=5, min_cluster_size=50):
        if method == "kmeans":
            self.model = KMeans(n_clusters=n_clusters)
        elif method == "gmm":
            self.model = GaussianMixture(n_components=n_clusters)
        elif method == "hdbscan":
            self.model = HDBSCAN(min_cluster_size=min_cluster_size)

    def fit_predict(self, embeddings):
        # Optional dimensionality reduction
        if embeddings.shape[1] > 20:
            from sklearn.decomposition import PCA
            embeddings = PCA(n_components=20).fit_transform(embeddings)
        return self.model.fit_predict(embeddings)
```

### 6.2 Hidden Markov Model (`regime/hmm.py`)

```python
class HMMRegimeModel:
    """
    Hidden Markov Model for regime transition modeling.

    Advantages over clustering:
        - Models temporal dependencies (regimes are "sticky")
        - Transition probabilities between regimes
        - Handles gradual regime shifts

    Implementation:
        - Uses hmmlearn.GaussianHMM
        - Input: aggregate market features (market vol, breadth, momentum)
        - States: N regimes (typically 3-5)
        - Output: regime sequence + transition matrix
    """
    def __init__(self, n_regimes=4, n_iter=100):
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
        )

    def fit(self, features):
        self.model.fit(features)
        return self

    def predict(self, features):
        return self.model.predict(features)

    def get_transition_matrix(self):
        return self.model.transmat_
```

### 6.3 Regime Labeler (`regime/labels.py`)

```python
class RegimeLabeler:
    """
    Map numeric cluster IDs to interpretable regime names.

    Process:
        1. For each cluster, compute aggregate statistics:
           - Mean return, mean volatility, mean breadth
        2. Classify based on rules:
           - Low-Vol Bull: positive returns, low vol
           - High-Vol Bull: positive returns, high vol
           - Bear: negative returns
           - Crisis: very negative returns, very high vol
           - Transition: moderate vol, near-zero returns

    Labels (standard set):
        0: "Low-Vol Bull"    — Steady uptrend, low volatility
        1: "High-Vol Bull"   — Rising with turbulence
        2: "Bear"            — Declining market
        3: "Crisis"          — Sharp decline, extreme vol (2008, COVID)
        4: "Transition"      — Regime change, uncertain direction
    """
    LABEL_MAP = {
        "low_vol_bull": "Low-Vol Bull",
        "high_vol_bull": "High-Vol Bull",
        "bear": "Bear",
        "crisis": "Crisis",
        "transition": "Transition",
    }

    def assign_labels(self, cluster_ids, returns, volatility):
        # Compute per-cluster statistics
        cluster_stats = {}
        for cid in np.unique(cluster_ids):
            mask = cluster_ids == cid
            cluster_stats[cid] = {
                "mean_return": returns[mask].mean(),
                "mean_vol": volatility[mask].mean(),
            }
        # Rule-based labeling
        labels = {}
        for cid, stats in cluster_stats.items():
            if stats["mean_return"] > 0 and stats["mean_vol"] < vol_median:
                labels[cid] = "Low-Vol Bull"
            elif stats["mean_return"] > 0 and stats["mean_vol"] >= vol_median:
                labels[cid] = "High-Vol Bull"
            elif stats["mean_return"] < -threshold and stats["mean_vol"] > vol_q75:
                labels[cid] = "Crisis"
            elif stats["mean_return"] < 0:
                labels[cid] = "Bear"
            else:
                labels[cid] = "Transition"
        return labels
```

### 6.4 Regime Detector Orchestrator (`regime/detector.py`)

```python
class RegimeDetector:
    """
    Orchestrates regime detection pipeline:
        1. Load embeddings (or compute from features)
        2. Run clustering (KMeans/GMM/HDBSCAN)
        3. Run HMM for temporal smoothing
        4. Assign interpretable labels
        5. Validate against known events (2008, COVID, etc.)
        6. Save regime labels to Parquet
    """
    def run(self, embeddings_df, features_df):
        # Clustering
        cluster_ids = self.clustering.fit_predict(embeddings)
        # HMM smoothing
        hmm_regimes = self.hmm.fit(market_features).predict(market_features)
        # Combine: use HMM for temporal consistency, clustering for fine-grained
        # Label assignment
        labels = self.labeler.assign_labels(hmm_regimes, returns, volatility)
        # Save
        regime_df = pd.DataFrame({"date": dates, "regime_id": hmm_regimes, "regime_label": ...})
        return regime_df
```

### 6.5 Historical Validation

Validate regime labels against known market events:
- **2008 Financial Crisis**: Should show "Crisis" regime
- **2015-2016 Chinese market turbulence**: Should show "Bear" or "High-Vol Bull"
- **March 2020 COVID crash**: Should show "Crisis" regime
- **2020-2021 Recovery**: Should show "Low-Vol Bull" → "High-Vol Bull"
- **2022 Inflation/Rate hikes**: Should show "Bear"

### 6.6 HTML Backtest Report (`backtest/report.py`)

```python
class HTMLReportGenerator:
    """
    Generate interactive HTML backtest reports using Plotly + Jinja2.

    Report sections:
        1. Executive Summary
           - Strategy name, period, key metrics table
           - Equity curve (strategy vs benchmark)

        2. Performance Metrics
           - Full metrics table: CAGR, Sharpe, Sortino, MaxDD, Calmar, Turnover
           - Monthly returns heatmap
           - Rolling 12-month Sharpe

        3. Regime Analysis
           - Equity curve with regime overlay (colored bands)
           - Per-regime performance table
           - Regime transition timeline

        4. Risk Analysis
           - Drawdown chart
           - Rolling volatility
           - VaR / CVaR estimates

        5. Portfolio Analysis
           - Position concentration over time
           - Sector exposure (if available)
           - Turnover over time

        6. Trade Analysis
           - Number of trades per period
           - Average holding period
           - Win rate and profit factor
    """
    def __init__(self, template_dir="templates"):
        self.env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))

    def generate(self, backtest_result, regime_labels=None, output_path="report.html"):
        # Create Plotly figures
        equity_fig = self._plot_equity_curve(backtest_result)
        drawdown_fig = self._plot_drawdowns(backtest_result)
        monthly_fig = self._plot_monthly_returns(backtest_result)
        regime_fig = self._plot_regime_overlay(backtest_result, regime_labels)
        # Render template
        template = self.env.get_template("backtest_report.html")
        html = template.render(
            metrics=backtest_result.metrics,
            equity_fig=equity_fig.to_html(),
            drawdown_fig=drawdown_fig.to_html(),
            monthly_fig=monthly_fig.to_html(),
            regime_fig=regime_fig.to_html() if regime_fig else "",
        )
        Path(output_path).write_text(html)
```

### 6.7 Regime Conditioning in RL

Extend the RL portfolio environment to include regime information:
- Add current regime label to observation space
- Agent can learn regime-conditional strategies (e.g., reduce risk in Crisis)
- Compare: regime-aware agent vs regime-blind agent

### 6.8 New Scripts
- `scripts/detect_regimes.py` — Run regime detection pipeline
- `scripts/generate_report.py` — Generate HTML backtest report

### 6.9 New Tests
- `test_regime/test_clustering.py` — KMeans/GMM/HDBSCAN produce valid cluster IDs
- `test_regime/test_hmm.py` — HMM fit/predict, transition matrix sums to 1
- `test_regime/test_labels.py` — Label assignment logic
- `test_regime/test_detector.py` — Full pipeline runs
- `test_backtest/test_report.py` — HTML report generates valid HTML

### 6.10 Exit Criteria
- Regime detection produces interpretable labels on historical NIFTY 50 data
- Known events (2008, COVID) are correctly identified
- HTML report generates with all sections
- Regime-conditional backtest analysis works
- All tests pass

---

## Key Architectural Decisions

1. **Parquet between stages** — Not CSV. Preserves dtypes (datetime, float64), 10-50x compression ratio, efficient column selection. Every pipeline stage reads/writes Parquet.

2. **Custom backtest engine** — Off-the-shelf libraries (Backtrader, Zipline, VectorBT) hide execution details. Full control over cost modeling, execution delay, and lookahead prevention is non-negotiable for research credibility.

3. **PyTorch 2.x native SDPA** — Uses `F.scaled_dot_product_attention` instead of the `flash-attn` package. Avoids CUDA compilation pain. Works on RTX 4080 (Ada Lovelace, sm_89) locally and H100/A100 on Colab without any custom builds.

4. **Ridge baseline in Phase 1** — Validates all infrastructure (data, features, backtest, tracking) without requiring GPU. If the pipeline works end-to-end with Ridge, it works with anything.

5. **NIFTY-first universe expansion** — Start with 50 liquid large-cap stocks for fast iteration (< 1 min data download, < 5 min feature computation). Configs are pre-built for NIFTY 500 and full Indian market - just swap the config file.

6. **Feature registry pattern** — Decorator-based `@register_feature` makes adding new features trivial. No need to modify the engine or any orchestration code. Just write the function, decorate it, and it's available.

7. **Hydra config** — Research labs need rapid experimentation. Override any parameter from CLI: `python train.py model.architecture.d_model=256 model.training.lr=0.001 data=nifty500`.

8. **Pre-norm transformer** — More stable training than post-norm, especially for deeper models. The final LayerNorm ensures the output is properly normalized regardless of depth.

9. **Multi-task learning** — A single model predicts return distribution + direction + volatility simultaneously. Auxiliary tasks act as regularizers and provide additional trading signals.

10. **Differential Sharpe reward** — For RL, using episode-level Sharpe as reward is too sparse. Differential Sharpe (Moody & Saffell, 2001) provides dense, per-step reward that approximates the Sharpe gradient.

---

## Verification Plan

### Per-Phase Verification

```bash
# Phase 1: Foundation
python scripts/ingest_data.py
python scripts/compute_features.py
python scripts/run_pipeline.py
python -m pytest tests/ -v --cov=src/quant_lab
# Expected: Full pipeline runs, Ridge baseline backtested, MLflow logged

# Phase 2: Transformer
python scripts/train_forecaster.py model.training.epochs=5  # Quick test
python -m pytest tests/ -v
# Expected: Transformer trains, loss decreases, all tests pass

# Phase 3: Features + Representation
python scripts/compute_features.py  # Now computes cross-asset + regime features
python scripts/pretrain.py  # Pre-train masked encoder
python -m pytest tests/ -v
# Expected: New features computed, embeddings extracted, transfer learning works

# Phase 4: TFT
python scripts/train_forecaster.py model=tft model.training.epochs=5
python -m pytest tests/ -v
# Expected: TFT trains with variable selection weights, interpretable attention

# Phase 5: RL
python scripts/train_rl.py rl=ppo rl.total_timesteps=10000  # Quick test
python -m pytest tests/ -v
# Expected: RL agent trains, beats equal-weight on training period

# Phase 6: Regime + Reporting
python scripts/detect_regimes.py
python scripts/generate_report.py
python -m pytest tests/ -v
# Expected: Regimes detected, HTML report generated with regime overlay
```

### Ongoing Verification
- **Unit tests**: `python -m pytest tests/ -v --ignore=tests/test_integration` (< 30s, no GPU, no network)
- **Integration**: `python -m pytest tests/test_integration/` (uses synthetic data, no external deps)
- **All tests deterministic** via `seed.py` (set_global_seed)
- **Lookahead guard** assertions run in every backtest (no future data leaks)
- **Pre-commit hooks** run ruff lint + format on every commit

### GPU Scaling Guide
| Task | RTX 4080 (12GB) | Colab H100 (80GB) |
|------|------------------|--------------------|
| Transformer training | seq_len=63, batch=64, BF16 | seq_len=512, batch=256, BF16 |
| Pre-training (masked) | seq_len=128, batch=32 | seq_len=512, batch=128 |
| TFT training | seq_len=63, batch=64 | seq_len=256, batch=128 |
| RL training | CPU is sufficient | CPU is sufficient |
| Regime detection | CPU (clustering) | CPU (clustering) |

---

## Progress Tracker

| Phase | Status | Tests | Key Deliverable |
|-------|--------|-------|-----------------|
| Phase 1: Foundation | ✅ Complete | 53 | End-to-end pipeline with Ridge baseline |
| Phase 2: Transformer | ✅ Complete | 114 | Multi-task transformer + training infrastructure |
| Phase 3: Features + Repr | ⬜ Pending | — | Cross-asset features + masked encoder + embeddings |
| Phase 4: TFT | ⬜ Pending | — | Interpretable TFT + variable selection |
| Phase 5: RL | ⬜ Pending | — | PPO/SAC portfolio agents |
| Phase 6: Regime + Report | ⬜ Pending | — | Regime detection + HTML reports |

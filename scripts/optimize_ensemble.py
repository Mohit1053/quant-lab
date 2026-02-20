"""Optimize ensemble weights across all available models.

Tests three strategies:
  1. Simple average
  2. Dirichlet-optimized weights (on val data)
  3. Regime-conditional weights (per-regime optimization)

Usage:
    python scripts/optimize_ensemble.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.seed import set_global_seed
from quant_lab.utils.device import get_device
from quant_lab.data.datasets import TemporalSplit, create_flat_datasets
from quant_lab.data.datamodule import QuantDataModule, DataModuleConfig
from quant_lab.features.feature_store import FeatureStore
from quant_lab.models.linear_baseline import RidgeBaseline
from quant_lab.models.ensemble import (
    EnsembleStrategy,
    EnsembleConfig,
    CombinationMethod,
    optimize_ensemble_weights,
)
from quant_lab.backtest.engine import BacktestEngine, BacktestConfig
from quant_lab.backtest.execution import ExecutionModel

logger = structlog.get_logger(__name__)

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
DATA_DIR = Path("data")
UNIVERSE = "nifty50"
SPLIT = TemporalSplit(
    train_end="2022-06-30",
    val_end="2023-06-30",
)
SEQ_LEN = 63
TARGET_COL = "log_return_1d"


def load_feature_data() -> tuple[pd.DataFrame, list[str]]:
    """Load feature DataFrame and identify feature columns."""
    store = FeatureStore(DATA_DIR / "features")
    feature_name = f"{UNIVERSE}_features"
    feature_df = store.load_features(feature_name)

    base_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
    feature_cols = [c for c in feature_df.columns if c not in base_cols]

    if TARGET_COL not in feature_df.columns:
        feature_df[TARGET_COL] = feature_df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    return feature_df, feature_cols


def generate_ridge_signals(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    split: TemporalSplit,
    period: str = "test",
) -> pd.DataFrame:
    """Generate Ridge model signals for specified period."""
    model_path = Path("outputs/models/ridge_baseline.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Ridge model not found at {model_path}")

    model = RidgeBaseline()
    model.load(model_path)

    datasets = create_flat_datasets(feature_df, feature_cols, split, TARGET_COL)
    X, _, meta = datasets[period]
    preds = model.predict(X)

    signals = meta.copy()
    signals["signal"] = preds
    return signals


def generate_transformer_signals(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    split: TemporalSplit,
    device: torch.device,
    period: str = "test",
) -> pd.DataFrame | None:
    """Generate Transformer signals for specified period."""
    model_path = Path("outputs/models/transformer/final_model.pt")
    if not model_path.exists():
        print(f"  [SKIP] Transformer model not found at {model_path}")
        return None

    from quant_lab.models.transformer.model import TransformerForecaster

    model = TransformerForecaster.load(model_path, map_location=str(device))
    model = model.to(device)
    model.eval()

    dm = QuantDataModule(
        feature_df, feature_cols, split,
        DataModuleConfig(
            sequence_length=SEQ_LEN,
            target_col=TARGET_COL,
            batch_size=256,
            num_workers=0,
        ),
    )
    dm.setup()

    if period == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.test_dataloader()

    if loader is None:
        return None

    all_preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            preds = model.predict_returns(x)
            all_preds.append(preds.cpu().numpy())

    test_preds = np.concatenate(all_preds)

    # Align with flat metadata
    datasets = create_flat_datasets(feature_df, feature_cols, split, TARGET_COL)
    _, _, meta = datasets[period]
    meta = meta.iloc[-len(test_preds):]

    signals = meta.copy()
    signals["signal"] = test_preds
    return signals


def generate_tft_small_signals(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    split: TemporalSplit,
    device: torch.device,
    period: str = "test",
) -> pd.DataFrame | None:
    """Generate TFT-small signals."""
    model_path = Path("outputs/models/tft_small/best.pt")
    if not model_path.exists():
        print(f"  [SKIP] TFT-small model not found at {model_path}")
        return None

    from quant_lab.models.tft.model import TFTForecaster

    try:
        model = TFTForecaster.load(model_path, map_location=str(device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"  [SKIP] TFT-small failed to load: {e}")
        return None

    dm = QuantDataModule(
        feature_df, feature_cols, split,
        DataModuleConfig(
            sequence_length=SEQ_LEN,
            target_col=TARGET_COL,
            batch_size=256,
            num_workers=0,
        ),
    )
    dm.setup()

    if period == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.test_dataloader()

    if loader is None:
        return None

    all_preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            preds = model.predict_returns(x)
            all_preds.append(preds.cpu().numpy())

    test_preds = np.concatenate(all_preds)

    datasets = create_flat_datasets(feature_df, feature_cols, split, TARGET_COL)
    _, _, meta = datasets[period]
    meta = meta.iloc[-len(test_preds):]

    signals = meta.copy()
    signals["signal"] = test_preds

    # Check for mode collapse
    sig_std = np.std(test_preds)
    if sig_std < 1e-6:
        print(f"  [WARN] TFT-small signal std={sig_std:.2e} — mode collapse detected, excluding")
        return None

    print(f"  TFT-small signal std={sig_std:.6f} (healthy)")
    return signals


def run_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    regime_labels: pd.Series | None = None,
    label: str = "",
) -> dict:
    """Run backtest and return metrics dict."""
    execution_model = ExecutionModel(
        commission_bps=10, slippage_bps=10, spread_bps=5, execution_delay_bars=1,
    )
    config = BacktestConfig(
        initial_capital=10_000_000,
        rebalance_frequency=5,
        max_position_size=0.05,
        top_n=10,
        risk_free_rate=0.065,
    )
    engine = BacktestEngine(execution_model=execution_model, config=config)
    result = engine.run(prices, signals, regime_labels=regime_labels)
    return result.metrics


def optimize_regime_weights(
    val_signals: dict[str, pd.DataFrame],
    val_prices: pd.DataFrame,
    regime_labels: pd.Series,
    n_trials: int = 200,
) -> dict[int, dict[str, float]]:
    """Optimize per-regime weights by running backtest trials per regime."""
    from quant_lab.backtest.engine import BacktestEngine, BacktestConfig

    model_names = list(val_signals.keys())
    n_models = len(model_names)

    # Get unique regimes in val period
    merged = None
    for name, df in val_signals.items():
        sig = df[["date", "ticker", "signal"]].rename(columns={"signal": f"signal_{name}"})
        if merged is None:
            merged = sig
        else:
            merged = merged.merge(sig, on=["date", "ticker"], how="inner")

    merged["regime"] = merged["date"].map(regime_labels)
    regimes = merged["regime"].dropna().unique()

    print(f"\n  Optimizing weights for {len(regimes)} regimes: {sorted(regimes)}")

    regime_weights = {}
    rng = np.random.RandomState(SEED)

    config = BacktestConfig(
        initial_capital=10_000_000,
        rebalance_frequency=5,
        max_position_size=0.05,
        top_n=10,
        risk_free_rate=0.065,
    )
    engine = BacktestEngine(
        execution_model=ExecutionModel(
            commission_bps=10, slippage_bps=10, spread_bps=5, execution_delay_bars=1,
        ),
        config=config,
    )

    for regime_id in sorted(regimes):
        regime_id = int(regime_id)
        best_score = -np.inf
        best_w = {name: 1.0 / n_models for name in model_names}

        # Filter to dates in this regime
        regime_dates = set(
            merged.loc[merged["regime"] == regime_id, "date"].unique()
        )
        if len(regime_dates) < 10:
            print(f"    Regime {regime_id}: too few dates ({len(regime_dates)}), using equal weights")
            regime_weights[regime_id] = best_w
            continue

        for trial in range(n_trials):
            raw = rng.dirichlet(np.ones(n_models))
            trial_weights = dict(zip(model_names, raw))

            ens_cfg = EnsembleConfig(
                method=CombinationMethod.WEIGHTED_AVERAGE,
                weights=trial_weights,
            )
            ens = EnsembleStrategy(ens_cfg)
            combined = ens.combine(val_signals)

            # Filter combined to regime dates only
            combined_regime = combined[combined["date"].isin(regime_dates)]
            if len(combined_regime) < 5:
                continue

            regime_prices = val_prices[val_prices["date"].isin(regime_dates)]
            if len(regime_prices) < 5:
                continue

            try:
                result = engine.run(regime_prices, combined_regime)
                score = result.metrics.get("sharpe", 0.0)
                if score > best_score:
                    best_score = score
                    best_w = trial_weights
            except Exception:
                continue

        regime_weights[regime_id] = best_w
        w_str = ", ".join(f"{k}: {v:.3f}" for k, v in best_w.items())
        print(f"    Regime {regime_id}: best Sharpe={best_score:.4f} -> [{w_str}]")

    return regime_weights


def main():
    set_global_seed(SEED)
    device = get_device()

    print("=" * 70)
    print("ENSEMBLE WEIGHT OPTIMIZATION")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading feature data...")
    feature_df, feature_cols = load_feature_data()
    print(f"  {len(feature_df)} rows, {len(feature_cols)} features")

    # Load regime labels
    regime_labels = None
    regime_path = Path("outputs/regimes/regime_labels.parquet")
    if regime_path.exists():
        regime_df = pd.read_parquet(regime_path)
        regime_df["date"] = pd.to_datetime(regime_df["date"])
        regime_labels = regime_df.set_index("date")["regime_label"]
        print(f"  Loaded {len(regime_labels)} regime labels")

    # Test prices
    test_dates = feature_df[
        feature_df["date"] > pd.Timestamp(SPLIT.val_end)
    ]["date"].unique()
    test_prices = feature_df[feature_df["date"].isin(test_dates)][
        ["date", "ticker", "adj_close"]
    ].copy()

    # Val prices
    val_dates = feature_df[
        (feature_df["date"] > pd.Timestamp(SPLIT.train_end))
        & (feature_df["date"] <= pd.Timestamp(SPLIT.val_end))
    ]["date"].unique()
    val_prices = feature_df[feature_df["date"].isin(val_dates)][
        ["date", "ticker", "adj_close"]
    ].copy()

    # Generate signals from all available models
    print("\n[2/5] Generating model signals...")
    test_signals: dict[str, pd.DataFrame] = {}
    val_signals: dict[str, pd.DataFrame] = {}

    # Ridge
    print("  Loading Ridge...")
    try:
        test_signals["ridge"] = generate_ridge_signals(feature_df, feature_cols, SPLIT, "test")
        val_signals["ridge"] = generate_ridge_signals(feature_df, feature_cols, SPLIT, "val")
        print(f"    Ridge: {len(test_signals['ridge'])} test, {len(val_signals['ridge'])} val signals")
    except Exception as e:
        print(f"    Ridge FAILED: {e}")

    # Transformer
    print("  Loading Transformer...")
    t_test = generate_transformer_signals(feature_df, feature_cols, SPLIT, device, "test")
    if t_test is not None:
        test_signals["transformer"] = t_test
        t_val = generate_transformer_signals(feature_df, feature_cols, SPLIT, device, "val")
        if t_val is not None:
            val_signals["transformer"] = t_val
        print(f"    Transformer: {len(t_test)} test" +
              (f", {len(t_val)} val" if t_val is not None else ", no val"))

    # TFT-small (if available)
    print("  Loading TFT-small...")
    tft_test = generate_tft_small_signals(feature_df, feature_cols, SPLIT, device, "test")
    if tft_test is not None:
        test_signals["tft_small"] = tft_test
        tft_val = generate_tft_small_signals(feature_df, feature_cols, SPLIT, device, "val")
        if tft_val is not None:
            val_signals["tft_small"] = tft_val

    available = list(test_signals.keys())
    print(f"\n  Available models: {available}")

    if len(available) < 2:
        print("  ERROR: Need at least 2 models for ensemble. Aborting.")
        return

    # Strategy 1: Simple Average
    print("\n[3/5] Strategy 1: Simple Average...")
    sa_cfg = EnsembleConfig(method=CombinationMethod.SIMPLE_AVERAGE)
    sa_ens = EnsembleStrategy(sa_cfg)
    sa_combined = sa_ens.combine(test_signals)
    sa_metrics = run_backtest(sa_combined, test_prices, regime_labels, "simple_avg")

    # Strategy 2: Dirichlet-optimized weights
    print("\n[4/5] Strategy 2: Optimized Weights (Dirichlet sampling on val)...")
    if len(val_signals) >= 2:
        np.random.seed(SEED)
        opt_weights = optimize_ensemble_weights(
            val_signals, val_prices, metric="sharpe", n_trials=500,
        )
        print(f"  Optimal weights: { {k: f'{v:.3f}' for k, v in opt_weights.items()} }")

        wa_cfg = EnsembleConfig(
            method=CombinationMethod.WEIGHTED_AVERAGE,
            weights=opt_weights,
        )
        wa_ens = EnsembleStrategy(wa_cfg)
        wa_combined = wa_ens.combine(test_signals)
        wa_metrics = run_backtest(wa_combined, test_prices, regime_labels, "optimized")
    else:
        print("  SKIP: Not enough val signals")
        opt_weights = {}
        wa_metrics = {}

    # Strategy 3: Regime-conditional weights
    print("\n[5/5] Strategy 3: Regime-Conditional Weights...")
    if regime_labels is not None and len(val_signals) >= 2:
        regime_w = optimize_regime_weights(
            val_signals, val_prices, regime_labels, n_trials=300,
        )
        print(f"\n  Regime weights:")
        for rid, w in sorted(regime_w.items()):
            w_str = ", ".join(f"{k}: {v:.3f}" for k, v in w.items())
            print(f"    Regime {rid}: [{w_str}]")

        rc_cfg = EnsembleConfig(
            method=CombinationMethod.REGIME_CONDITIONAL,
            weights=opt_weights,  # Fallback for unknown regimes
            regime_weights=regime_w,
        )
        rc_ens = EnsembleStrategy(rc_cfg)
        rc_combined = rc_ens.combine(test_signals, regime_labels=regime_labels)
        rc_metrics = run_backtest(rc_combined, test_prices, regime_labels, "regime_cond")
    else:
        print("  SKIP: No regime labels or not enough val signals")
        regime_w = {}
        rc_metrics = {}

    # Individual model backtests for comparison
    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL BACKTESTS")
    print("=" * 70)
    individual_metrics = {}
    for name, sig_df in test_signals.items():
        try:
            m = run_backtest(sig_df, test_prices, regime_labels, name)
            individual_metrics[name] = m
        except Exception as e:
            print(f"  {name} backtest failed: {e}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    all_results = {}
    for name, m in individual_metrics.items():
        all_results[name] = m
    if sa_metrics:
        all_results["ensemble_simple_avg"] = sa_metrics
    if wa_metrics:
        all_results["ensemble_optimized"] = wa_metrics
    if rc_metrics:
        all_results["ensemble_regime_cond"] = rc_metrics

    metrics_to_show = ["cagr", "sharpe", "sortino", "max_drawdown", "volatility", "total_return"]

    header = f"{'Metric':<20}"
    for name in all_results:
        header += f" {name:>18}"
    print(header)
    print("-" * len(header))

    for metric in metrics_to_show:
        row = f"  {metric:<18}"
        for name in all_results:
            val = all_results[name].get(metric, 0)
            if "return" in metric or "cagr" in metric or "drawdown" in metric:
                row += f" {val:>17.2%}"
            else:
                row += f" {val:>17.4f}"
        print(row)

    # Model contributions
    print("\n" + "=" * 70)
    print("MODEL CONTRIBUTIONS")
    print("=" * 70)
    contrib = sa_ens.get_model_contributions(test_signals)
    print(contrib.to_string(index=False))

    # Save optimized config
    output_dir = Path("outputs/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_out = {
        "available_models": available,
        "simple_avg_metrics": {k: float(v) for k, v in sa_metrics.items()} if sa_metrics else {},
        "optimized_weights": {k: float(v) for k, v in opt_weights.items()},
        "optimized_metrics": {k: float(v) for k, v in wa_metrics.items()} if wa_metrics else {},
        "regime_weights": {str(k): {mk: float(mv) for mk, mv in v.items()} for k, v in regime_w.items()},
        "regime_cond_metrics": {k: float(v) for k, v in rc_metrics.items()} if rc_metrics else {},
        "individual_metrics": {
            name: {k: float(v) for k, v in m.items()}
            for name, m in individual_metrics.items()
        },
    }
    with open(output_dir / "ensemble_optimization_results.json", "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"\nResults saved to {output_dir / 'ensemble_optimization_results.json'}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

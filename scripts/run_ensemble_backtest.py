"""Run ensemble backtest combining multiple model signals.

Usage:
    python scripts/run_ensemble_backtest.py
    python scripts/run_ensemble_backtest.py ensemble.method=regime_conditional
    python scripts/run_ensemble_backtest.py ensemble.optimize_weights=true
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import structlog
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.seed import set_global_seed
from quant_lab.utils.device import get_device
from quant_lab.data.datasets import TemporalSplit, create_flat_datasets
from quant_lab.data.datamodule import QuantDataModule, DataModuleConfig
from quant_lab.features.engine import FeatureEngine
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


def _generate_ridge_signals(
    model_path: Path,
    X_test: np.ndarray,
    meta_test: pd.DataFrame,
) -> pd.DataFrame:
    """Generate signals from Ridge model."""
    model = RidgeBaseline()
    model.load(model_path)
    preds = model.predict(X_test)
    signals = meta_test.copy()
    signals["signal"] = preds
    return signals


def _generate_dl_signals(
    model_cls,
    model_path: Path,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    split: TemporalSplit,
    sequence_length: int,
    device: torch.device,
) -> pd.DataFrame | None:
    """Generate signals from a DL model (Transformer or TFT)."""
    model = model_cls.load(model_path, map_location=str(device))
    model = model.to(device)
    model.eval()

    dm = QuantDataModule(
        feature_df,
        feature_cols,
        split,
        DataModuleConfig(
            sequence_length=sequence_length,
            target_col="log_return_1d",
            batch_size=256,
            num_workers=0,
        ),
    )
    dm.setup()
    test_loader = dm.test_dataloader()

    if test_loader is None:
        return None

    all_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            preds = model.predict_returns(x)
            all_preds.append(preds.cpu().numpy())

    test_preds = np.concatenate(all_preds)

    # Align with flat metadata
    datasets = create_flat_datasets(
        feature_df, feature_cols, split, target_col="log_return_1d"
    )
    _, _, meta_test = datasets["test"]
    meta_test = meta_test.iloc[-len(test_preds):]

    signals = meta_test.copy()
    signals["signal"] = test_preds
    return signals


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run ensemble backtest."""
    set_global_seed(cfg.project.seed)
    device = get_device()

    data_dir = Path(cfg.paths.data_dir)
    universe_name = cfg.data.universe.name

    # Load feature data
    feature_store = FeatureStore(data_dir / "features")
    feature_name = f"{universe_name}_features"
    if not feature_store.has_features(feature_name):
        logger.error("No feature data. Run 'python scripts/compute_features.py' first.")
        return

    feature_df = feature_store.load_features(feature_name)

    # Feature columns
    base_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
    feature_cols = [c for c in feature_df.columns if c not in base_cols]

    # Ensure target
    target_col = "log_return_1d"
    if target_col not in feature_df.columns:
        feature_df[target_col] = feature_df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    # Split
    split = TemporalSplit(
        train_end=cfg.data.date_range.train_end,
        val_end=cfg.data.date_range.val_end,
    )
    datasets = create_flat_datasets(feature_df, feature_cols, split, target_col=target_col)
    X_test, y_test, meta_test = datasets["test"]

    # Generate per-model signals
    all_signals: dict[str, pd.DataFrame] = {}
    individual_results = {}

    # Ridge
    ridge_path = Path("outputs/models/ridge_baseline.pkl")
    if ridge_path.exists():
        logger.info("generating_ridge_signals")
        all_signals["ridge"] = _generate_ridge_signals(ridge_path, X_test, meta_test)

    # Transformer
    transformer_path = Path("outputs/models/transformer/final_model.pt")
    if transformer_path.exists():
        logger.info("generating_transformer_signals")
        from quant_lab.models.transformer.model import TransformerForecaster

        sig = _generate_dl_signals(
            TransformerForecaster,
            transformer_path,
            feature_df,
            feature_cols,
            split,
            cfg.model.input.sequence_length,
            device,
        )
        if sig is not None:
            all_signals["transformer"] = sig

    # TFT
    tft_path = Path("outputs/models/tft/final_model.pt")
    if tft_path.exists():
        logger.info("generating_tft_signals")
        from quant_lab.models.tft.model import TFTForecaster

        sig = _generate_dl_signals(
            TFTForecaster,
            tft_path,
            feature_df,
            feature_cols,
            split,
            cfg.model.input.sequence_length,
            device,
        )
        if sig is not None:
            all_signals["tft"] = sig

    if not all_signals:
        logger.error("No trained models found in outputs/models/")
        return

    logger.info("models_loaded", models=list(all_signals.keys()))

    # Test prices
    test_dates = meta_test["date"].unique()
    test_prices = feature_df[feature_df["date"].isin(test_dates)][
        ["date", "ticker", "adj_close"]
    ].copy()

    # Build backtest engine
    execution_model = ExecutionModel(
        commission_bps=cfg.backtest.execution.commission_bps,
        slippage_bps=cfg.backtest.execution.slippage_bps,
        spread_bps=cfg.backtest.execution.spread_bps,
        execution_delay_bars=cfg.backtest.execution.execution_delay_bars,
    )
    backtest_config = BacktestConfig(
        initial_capital=cfg.backtest.portfolio.initial_capital,
        rebalance_frequency=cfg.backtest.portfolio.rebalance_frequency,
        max_position_size=cfg.backtest.portfolio.max_position_size,
        top_n=cfg.backtest.portfolio.top_n,
        risk_free_rate=cfg.backtest.metrics.risk_free_rate,
    )

    # Load regime labels
    regime_labels = None
    regime_path = Path("outputs/regimes/regime_labels.parquet")
    if regime_path.exists():
        regime_df = pd.read_parquet(regime_path)
        regime_df["date"] = pd.to_datetime(regime_df["date"])
        regime_labels = regime_df.set_index("date")["regime_label"]

    # Run individual backtests for comparison
    engine = BacktestEngine(execution_model=execution_model, config=backtest_config)
    for name, sig_df in all_signals.items():
        try:
            result = engine.run(test_prices, sig_df, regime_labels=regime_labels)
            individual_results[name] = result.metrics
        except Exception as e:
            logger.warning("individual_backtest_failed", model=name, error=str(e))

    # Configure ensemble
    ens_cfg = cfg.get("ensemble", {})
    method = CombinationMethod(ens_cfg.get("method", "simple_average"))

    weights = {}
    if ens_cfg.get("weights"):
        weights = {str(k): float(v) for k, v in ens_cfg.weights.items()}

    regime_weights = {}
    if ens_cfg.get("regime_weights"):
        for rid, w in ens_cfg.regime_weights.items():
            regime_weights[int(rid)] = {str(k): float(v) for k, v in w.items()}

    # Optimize weights if requested
    if ens_cfg.get("optimize_weights", False):
        logger.info("optimizing_ensemble_weights")
        val_datasets = create_flat_datasets(feature_df, feature_cols, split, target_col)
        _, _, val_meta = val_datasets["val"]
        val_dates = val_meta["date"].unique()
        val_prices = feature_df[feature_df["date"].isin(val_dates)][
            ["date", "ticker", "adj_close"]
        ].copy()

        # Generate val-period signals (reuse test signals with val dates)
        val_signals = {}
        for name, sig_df in all_signals.items():
            val_sig = sig_df[sig_df["date"].isin(val_dates)]
            if len(val_sig) > 0:
                val_signals[name] = val_sig

        if val_signals:
            weights = optimize_ensemble_weights(
                val_signals,
                val_prices,
                metric=ens_cfg.get("optimize_metric", "sharpe"),
                n_trials=100,
            )

    ensemble_config = EnsembleConfig(
        method=method,
        weights=weights,
        regime_weights=regime_weights,
    )
    ensemble = EnsembleStrategy(ensemble_config)
    combined_signals = ensemble.combine(all_signals, regime_labels=regime_labels)

    # Run ensemble backtest
    ensemble_result = engine.run(
        test_prices, combined_signals, regime_labels=regime_labels
    )

    # Model contributions
    contributions = ensemble.get_model_contributions(all_signals)

    # Print results
    print("\n" + "=" * 70)
    print("ENSEMBLE BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nMethod: {method.value}")
    print(f"Models: {list(all_signals.keys())}")
    if weights:
        print(f"Weights: {weights}")

    print(f"\n{'Metric':<25} {'Ensemble':>12}", end="")
    for name in all_signals:
        print(f" {name:>12}", end="")
    print()
    print("-" * (25 + 12 * (1 + len(all_signals))))

    for metric in ["cagr", "sharpe", "sortino", "max_drawdown", "total_return", "volatility"]:
        ens_val = ensemble_result.metrics.get(metric, 0)
        if "return" in metric or "cagr" in metric or "drawdown" in metric:
            print(f"  {metric:<23} {ens_val:>11.2%}", end="")
        else:
            print(f"  {metric:<23} {ens_val:>11.4f}", end="")

        for name in all_signals:
            ind_val = individual_results.get(name, {}).get(metric, 0)
            if "return" in metric or "cagr" in metric or "drawdown" in metric:
                print(f" {ind_val:>11.2%}", end="")
            else:
                print(f" {ind_val:>11.4f}", end="")
        print()

    print(f"\nModel Contributions:")
    print(contributions.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()

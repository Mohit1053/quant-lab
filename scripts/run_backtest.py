"""Run standalone backtest with a trained model.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py backtest.portfolio.top_n=10
    python scripts/run_backtest.py model_path=outputs/models/transformer/final_model.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.seed import set_global_seed
from quant_lab.utils.device import get_device
from quant_lab.data.datasets import TemporalSplit, create_flat_datasets
from quant_lab.data.storage.parquet_store import ParquetStore
from quant_lab.features.engine import FeatureEngine
from quant_lab.features.feature_store import FeatureStore
from quant_lab.models.linear_baseline import RidgeBaseline
from quant_lab.backtest.engine import BacktestEngine, BacktestConfig
from quant_lab.backtest.execution import ExecutionModel

logger = structlog.get_logger(__name__)


def _load_model(model_path: str | None, cfg: DictConfig):
    """Load the best available model."""
    # Try transformer model first
    if model_path and Path(model_path).exists():
        logger.info("loading_custom_model", path=model_path)
        if model_path.endswith(".pkl"):
            model = RidgeBaseline()
            model.load(Path(model_path))
            return model, "ridge"
        else:
            import torch
            from quant_lab.models.transformer.model import TransformerForecaster
            model = TransformerForecaster.load(Path(model_path))
            return model, "transformer"

    # Try default paths
    default_paths = [
        ("outputs/models/transformer/final_model.pt", "transformer"),
        ("outputs/models/ridge_baseline.pkl", "ridge"),
    ]
    for path, model_type in default_paths:
        if Path(path).exists():
            logger.info("loading_model", path=path, type=model_type)
            if model_type == "transformer":
                import torch
                from quant_lab.models.transformer.model import TransformerForecaster
                model = TransformerForecaster.load(Path(path))
                return model, model_type
            elif model_type == "ridge":
                model = RidgeBaseline()
                model.load(Path(path))
                return model, model_type

    # No trained model found - train a quick Ridge baseline
    logger.warning("no_trained_model_found, training Ridge baseline on the fly")
    return None, "ridge_adhoc"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run backtest on test period."""
    set_global_seed(cfg.project.seed)

    data_dir = Path(cfg.paths.data_dir)
    universe_name = cfg.data.universe.name

    # Load feature data
    feature_store = FeatureStore(data_dir / "features")
    if not feature_store.has_features(f"{universe_name}_features"):
        logger.error("No feature data. Run 'python scripts/compute_features.py' first.")
        return

    feature_df = feature_store.load_features(f"{universe_name}_features")
    logger.info("data_loaded", rows=len(feature_df), tickers=feature_df["ticker"].nunique())

    # Feature columns
    base_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
    feature_cols = [c for c in feature_df.columns if c not in base_cols]

    # Ensure target column
    target_col = "log_return_1d"
    if target_col not in feature_df.columns:
        feature_df[target_col] = feature_df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    # Split data
    split = TemporalSplit(
        train_end=cfg.data.date_range.train_end,
        val_end=cfg.data.date_range.val_end,
    )
    datasets = create_flat_datasets(feature_df, feature_cols, split, target_col=target_col)
    X_train, y_train, meta_train = datasets["train"]
    X_test, y_test, meta_test = datasets["test"]

    # Load or train model
    model_path = cfg.get("model_path", None)
    model, model_type = _load_model(model_path, cfg)

    if model is None:
        model = RidgeBaseline(alpha=1.0)
        model.fit(X_train, y_train)
        model_type = "ridge_adhoc"

    # Generate signals
    if model_type in ("transformer", "tft"):
        # Transformer needs sequence input via DataModule
        import torch
        from quant_lab.data.datamodule import QuantDataModule, DataModuleConfig

        dm = QuantDataModule(
            feature_df, feature_cols, split,
            DataModuleConfig(
                sequence_length=cfg.model.input.sequence_length,
                target_col=target_col,
                batch_size=256,
                num_workers=0,
            ),
        )
        dm.setup()
        test_loader = dm.test_dataloader()
        device = get_device()
        model = model.to(device)
        model.eval()
        all_preds = []
        with torch.no_grad():
            for x, _targets in test_loader:
                x = x.to(device)
                preds = model.predict_returns(x)
                all_preds.append(preds.cpu().numpy())
        test_preds = np.concatenate(all_preds)
        # Align with meta_test (DataModule may have fewer samples due to seq_len)
        meta_test = meta_test.iloc[-len(test_preds):]
        y_test = y_test[-len(test_preds):]
    else:
        test_preds = model.predict(X_test)
    signals_df = meta_test.copy()
    signals_df["signal"] = test_preds

    # Get test prices
    test_dates = meta_test["date"].unique()
    test_prices = feature_df[feature_df["date"].isin(test_dates)][
        ["date", "ticker", "adj_close"]
    ].copy()

    # Build backtest
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

    # Load regime labels if available (used for both sizing and reporting)
    regime_labels = None
    regime_series = None
    regime_path = Path("outputs/regimes/regime_labels.parquet")
    if regime_path.exists():
        regime_df = pd.read_parquet(regime_path)
        regime_df["date"] = pd.to_datetime(regime_df["date"])
        regime_labels = regime_df["regime_label"].values
        regime_series = regime_df.set_index("date")["regime_label"]
        logger.info("regime_labels_loaded", n_regimes=len(set(regime_labels[regime_labels >= 0])))

    # Build regime_size_map from config if present
    regime_size_map = None
    if cfg.backtest.portfolio.get("regime_size_map"):
        regime_size_map = {int(k): float(v) for k, v in cfg.backtest.portfolio.regime_size_map.items()}
        backtest_config.regime_size_map = regime_size_map
        logger.info("regime_sizing_enabled", regime_size_map=regime_size_map)

    engine = BacktestEngine(execution_model=execution_model, config=backtest_config)
    result = engine.run(prices=test_prices, signals=signals_df, regime_labels=regime_series)

    # Generate HTML report
    try:
        from quant_lab.backtest.report import BacktestReport, ReportConfig

        regime_summary = None
        regime_summary_path = Path("outputs/regimes/regime_summary.parquet")
        if regime_summary_path.exists():
            regime_summary = pd.read_parquet(regime_summary_path)

        report_config = ReportConfig(
            title=f"Backtest Report - {model_type.title()}",
            output_dir=str(cfg.backtest.get("report_dir", "outputs/backtests")),
        )
        report = BacktestReport(report_config)
        report_path = report.generate(
            portfolio_values=result.equity_curve.values,
            dates=result.equity_curve.index,
            metrics=result.metrics,
            weights_history=result.weights_history.values,
            regime_labels=regime_labels[:len(result.equity_curve)] if regime_labels is not None else None,
            regime_summary=regime_summary,
        )
        logger.info("report_saved", path=report_path)
    except Exception as e:
        logger.warning("report_generation_failed", error=str(e))
        report_path = None

    # Log to MLflow
    try:
        from quant_lab.tracking.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(
            experiment_name=cfg.experiment.tracking.get("experiment_name", "backtest"),
            tracking_uri=cfg.experiment.mlflow.get("tracking_uri", "mlruns"),
        )
        tracker.start_run(run_name=f"backtest_{model_type}")
        tracker.log_params({"model_type": model_type})
        tracker.log_metrics({f"bt_{k}": v for k, v in result.metrics.items()})
        tracker.end_run()
    except Exception as e:
        logger.warning("mlflow_failed", error=str(e))

    # Print results
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS ({model_type.upper()})")
    print("=" * 60)
    for metric, value in result.metrics.items():
        if "return" in metric or "cagr" in metric or "drawdown" in metric:
            print(f"  {metric:25s}: {value:>10.2%}")
        else:
            print(f"  {metric:25s}: {value:>10.4f}")
    if report_path:
        print(f"\n  Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

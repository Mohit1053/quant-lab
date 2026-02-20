"""Script: End-to-end pipeline - ingest, features, train, backtest, track."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.logging import setup_logging
from quant_lab.utils.seed import set_global_seed
from quant_lab.utils.timer import timed
from quant_lab.data.sources.yfinance_source import YFinanceSource
from quant_lab.data.cleaning.pipeline import CleaningPipeline, CleaningConfig
from quant_lab.data.storage.parquet_store import ParquetStore
from quant_lab.data.datasets import TemporalSplit, create_flat_datasets
from quant_lab.data.universe import get_universe
from quant_lab.features.engine import FeatureEngine
from quant_lab.features.feature_store import FeatureStore
from quant_lab.models.linear_baseline import RidgeBaseline
from quant_lab.backtest.engine import BacktestEngine, BacktestConfig
from quant_lab.backtest.execution import ExecutionModel
from quant_lab.tracking.mlflow_tracker import MLflowTracker


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_global_seed(cfg.project.seed)

    import structlog

    logger = structlog.get_logger(__name__)

    logger.info("=" * 60)
    logger.info("  AI Quant Research Lab - Phase 1 Pipeline")
    logger.info("=" * 60)

    data_dir = Path(cfg.paths.data_dir)
    universe_name = cfg.data.universe.name

    # ── Step 1: Data Ingestion ─────────────────────────────────
    logger.info("STEP 1: Data Ingestion")

    clean_store = ParquetStore(data_dir / "cleaned")
    if clean_store.exists(f"{universe_name}_cleaned"):
        logger.info("using_cached_data")
        df = clean_store.load(f"{universe_name}_cleaned")
    else:
        universe = get_universe(universe_name)

        # Auto-load NIFTY 500 tickers from NSE if universe is empty
        if not universe.tickers and universe_name in ("nifty500", "indian_market"):
            from quant_lab.data.universe import load_nifty500_tickers

            tickers = load_nifty500_tickers(
                cache_dir=str(data_dir / "universe_cache"),
            )
            universe = get_universe(universe_name, tickers_override=tickers)
            logger.info(
                "dynamic_tickers_loaded",
                universe=universe_name,
                count=len(tickers),
            )

        source = YFinanceSource()
        raw_df = source.fetch(
            tickers=list(universe.tickers),
            start=cfg.data.date_range.start,
            end=cfg.data.date_range.end,
        )

        raw_store = ParquetStore(data_dir / "raw")
        raw_store.save(raw_df, f"{universe_name}_raw")

        cleaning_cfg = CleaningConfig(
            max_missing_pct=cfg.data.cleaning.max_missing_pct,
            ffill_limit=cfg.data.cleaning.ffill_limit,
            outlier_sigma=cfg.data.cleaning.outlier_sigma,
            min_history_days=cfg.data.cleaning.min_history_days,
        )
        pipeline = CleaningPipeline(cleaning_cfg)
        df = pipeline.run(raw_df)
        clean_store.save(df, f"{universe_name}_cleaned")

    logger.info(
        "data_ready",
        tickers=df["ticker"].nunique(),
        trading_days=df["date"].nunique(),
        date_range=f"{df['date'].min()} to {df['date'].max()}",
    )

    # ── Step 2: Feature Computation ────────────────────────────
    logger.info("STEP 2: Feature Computation")

    feature_store = FeatureStore(data_dir / "features")
    if feature_store.has_features(f"{universe_name}_features"):
        logger.info("using_cached_features")
        df = feature_store.load_features(f"{universe_name}_features")
    else:
        windows = OmegaConf.to_container(cfg.features.windows, resolve=True)
        normalization = OmegaConf.to_container(cfg.features.normalization, resolve=True)

        engine = FeatureEngine(
            enabled_features=list(cfg.features.enabled_features),
            windows=windows,
            normalization=normalization,
        )
        df = engine.compute(df)
        df = engine.normalize(df)
        feature_store.save_features(df, f"{universe_name}_features")

    # Identify feature columns
    base_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
    feature_cols = [c for c in df.columns if c not in base_cols]
    logger.info("features_ready", num_features=len(feature_cols))

    # ── Step 3: Train/Val/Test Split ───────────────────────────
    logger.info("STEP 3: Train/Val/Test Split")

    target_col = "log_return_1d"
    if target_col not in df.columns:
        df[target_col] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    split = TemporalSplit(
        train_end=cfg.data.date_range.train_end,
        val_end=cfg.data.date_range.val_end,
    )
    datasets = create_flat_datasets(df, feature_cols, split, target_col=target_col)

    X_train, y_train, meta_train = datasets["train"]
    X_val, y_val, meta_val = datasets["val"]
    X_test, y_test, meta_test = datasets["test"]

    # ── Step 4: Train Model ────────────────────────────────────
    logger.info("STEP 4: Train Ridge Baseline")

    model = RidgeBaseline(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate on validation
    val_metrics = model.evaluate(X_val, y_val)
    logger.info("validation_metrics", **{k: f"{v:.6f}" for k, v in val_metrics.items()})

    # Evaluate on test
    test_metrics = model.evaluate(X_test, y_test)
    logger.info("test_metrics", **{k: f"{v:.6f}" for k, v in test_metrics.items()})

    # Feature importance
    importance = model.get_feature_importance(feature_cols)
    top_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    logger.info("top_features", features=top_features)

    # ── Step 5: Generate Signals & Run Backtest ────────────────
    logger.info("STEP 5: Backtest")

    # Generate predictions on test set as signals
    test_preds = model.predict(X_test)
    signals_df = meta_test.copy()
    signals_df["signal"] = test_preds

    # Get prices for the test period
    test_dates = meta_test["date"].unique()
    test_prices = df[df["date"].isin(test_dates)][["date", "ticker", "adj_close"]].copy()

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

    engine = BacktestEngine(execution_model=execution_model, config=backtest_config)
    result = engine.run(prices=test_prices, signals=signals_df)

    # ── Step 6: Display Results ────────────────────────────────
    logger.info("=" * 60)
    logger.info("  BACKTEST RESULTS")
    logger.info("=" * 60)
    for metric, value in result.metrics.items():
        if "return" in metric or "cagr" in metric or "drawdown" in metric:
            logger.info(f"  {metric}: {value:.2%}")
        else:
            logger.info(f"  {metric}: {value:.4f}")

    # ── Step 7: Log to MLflow ──────────────────────────────────
    logger.info("STEP 7: MLflow Tracking")

    tracker = MLflowTracker(
        experiment_name=cfg.experiment.tracking.experiment_name,
        tracking_uri=cfg.experiment.mlflow.tracking_uri,
    )
    tracker.start_run(run_name=f"phase1_ridge_{universe_name}")

    # Log config
    tracker.log_config(OmegaConf.to_container(cfg, resolve=True))

    # Log model metrics
    prefixed_val = {f"val_{k}": v for k, v in val_metrics.items()}
    prefixed_test = {f"test_{k}": v for k, v in test_metrics.items()}
    tracker.log_metrics(prefixed_val)
    tracker.log_metrics(prefixed_test)

    # Log backtest metrics
    bt_metrics = {f"bt_{k}": v for k, v in result.metrics.items()}
    tracker.log_metrics(bt_metrics)

    # Save model
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ridge_baseline.pkl"
    model.save(model_path)
    tracker.log_artifact(model_path)

    tracker.end_run()

    logger.info("=" * 60)
    logger.info("  Pipeline Complete!")
    logger.info("  Run 'mlflow ui' to view experiment results.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

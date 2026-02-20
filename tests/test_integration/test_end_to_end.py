"""Integration test: end-to-end pipeline with synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.data.cleaning.pipeline import CleaningPipeline, CleaningConfig
from quant_lab.features.engine import FeatureEngine
from quant_lab.data.datasets import TemporalSplit, create_flat_datasets
from quant_lab.models.linear_baseline import RidgeBaseline
from quant_lab.backtest.engine import BacktestEngine, BacktestConfig
from quant_lab.backtest.execution import ExecutionModel


@pytest.mark.integration
class TestEndToEnd:
    """Full pipeline test using synthetic data (no network, no GPU)."""

    def test_full_pipeline(self, synthetic_ohlcv):
        # Step 1: Clean data
        pipeline = CleaningPipeline(CleaningConfig(min_history_days=100))
        clean_df = pipeline.run(synthetic_ohlcv)
        assert len(clean_df) > 0

        # Step 2: Compute features (use short windows for synthetic data)
        engine = FeatureEngine(
            enabled_features=["log_returns", "realized_volatility", "momentum"],
            windows={"short": [1, 5], "medium": [21]},
            normalization={"method": "rolling_zscore", "lookback": 63},
        )
        feature_df = engine.compute(clean_df)
        feature_cols = engine.get_feature_columns(feature_df)
        assert len(feature_cols) > 0

        # Normalize with short lookback suitable for synthetic data
        feature_df = engine.normalize(feature_df)

        # Step 3: Create train/test split
        target_col = "log_return_1d"
        # Synthetic data: 504 bdays from 2020-01-01 (~2020-01 to 2021-12)
        split = TemporalSplit(train_end="2020-12-31", val_end="2021-06-30")
        datasets = create_flat_datasets(feature_df, feature_cols, split, target_col=target_col)

        X_train, y_train, meta_train = datasets["train"]
        X_test, y_test, meta_test = datasets["test"]

        assert len(X_train) > 0, "No training data"
        assert len(X_test) > 0, "No test data"

        # Step 4: Train model
        model = RidgeBaseline(alpha=1.0)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        assert "mse" in metrics
        assert "direction_accuracy" in metrics
        assert 0 <= metrics["direction_accuracy"] <= 1

        # Step 5: Generate signals and backtest
        preds = model.predict(X_test)
        signals_df = meta_test.copy()
        signals_df["signal"] = preds

        test_dates = meta_test["date"].unique()
        test_prices = feature_df[feature_df["date"].isin(test_dates)][
            ["date", "ticker", "adj_close"]
        ].copy()

        bt_engine = BacktestEngine(
            execution_model=ExecutionModel(),
            config=BacktestConfig(
                initial_capital=1_000_000,
                rebalance_frequency=5,
                top_n=3,
            ),
        )
        result = bt_engine.run(prices=test_prices, signals=signals_df)

        # Validate results
        assert result.equity_curve.iloc[0] == 1_000_000
        assert len(result.metrics) > 0
        assert np.isfinite(result.metrics["sharpe"])
        assert np.isfinite(result.metrics["cagr"])
        assert result.metrics["max_drawdown"] <= 0

        # The pipeline ran end-to-end without crashing!

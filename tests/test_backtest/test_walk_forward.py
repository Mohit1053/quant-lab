"""Tests for walk-forward analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.data.datasets import TemporalSplit, create_flat_datasets
from quant_lab.models.linear_baseline import RidgeBaseline
from quant_lab.backtest.walk_forward import (
    WalkForwardSplitter,
    WalkForwardConfig,
    WalkForwardEngine,
    WindowType,
)
from quant_lab.backtest.engine import BacktestConfig


class TestWalkForwardSplitter:
    def test_expanding_generates_correct_folds(self):
        """504 bdays with small windows should produce multiple folds."""
        dates = pd.bdate_range("2020-01-01", periods=504)
        config = WalkForwardConfig(
            window_type=WindowType.EXPANDING,
            train_days=200,
            val_days=50,
            test_days=50,
            step_days=50,
            min_train_days=100,
        )
        splitter = WalkForwardSplitter(config)
        splits = splitter.generate_splits(pd.DatetimeIndex(dates))

        assert len(splits) >= 3
        for split in splits:
            assert split.train_end < split.val_end

    def test_rolling_generates_folds(self):
        dates = pd.bdate_range("2020-01-01", periods=504)
        config = WalkForwardConfig(
            window_type=WindowType.ROLLING,
            train_days=100,
            val_days=50,
            test_days=50,
            step_days=50,
            min_train_days=50,
        )
        splitter = WalkForwardSplitter(config)
        splits = splitter.generate_splits(pd.DatetimeIndex(dates))
        assert len(splits) >= 4

    def test_too_short_data_returns_empty(self):
        dates = pd.bdate_range("2020-01-01", periods=50)
        config = WalkForwardConfig(
            train_days=200,
            val_days=50,
            test_days=50,
        )
        splitter = WalkForwardSplitter(config)
        splits = splitter.generate_splits(pd.DatetimeIndex(dates))
        assert len(splits) == 0

    def test_val_end_strictly_increasing(self):
        """Consecutive folds should have strictly increasing val_end."""
        dates = pd.bdate_range("2020-01-01", periods=504)
        config = WalkForwardConfig(
            train_days=200,
            val_days=50,
            test_days=50,
            step_days=50,
        )
        splitter = WalkForwardSplitter(config)
        splits = splitter.generate_splits(pd.DatetimeIndex(dates))

        for i in range(len(splits) - 1):
            assert splits[i].val_end < splits[i + 1].val_end

    def test_min_train_days_respected(self):
        """Folds with fewer than min_train_days training data should be skipped."""
        dates = pd.bdate_range("2020-01-01", periods=504)
        config = WalkForwardConfig(
            window_type=WindowType.EXPANDING,
            train_days=100,
            val_days=50,
            test_days=50,
            step_days=50,
            min_train_days=100,
        )
        splitter = WalkForwardSplitter(config)
        splits = splitter.generate_splits(pd.DatetimeIndex(dates))

        # All folds should have at least min_train_days
        for split in splits:
            train_end = pd.Timestamp(split.train_end)
            # train_start is dates[0] for expanding
            assert train_end >= dates[config.min_train_days - 1]

    def test_single_fold_possible(self):
        """Just enough data for exactly one fold."""
        # Need: train_days + val_days + at least 1 test day
        total = 200 + 50 + 10
        dates = pd.bdate_range("2020-01-01", periods=total)
        config = WalkForwardConfig(
            train_days=200,
            val_days=50,
            test_days=10,
            step_days=50,
            min_train_days=100,
        )
        splitter = WalkForwardSplitter(config)
        splits = splitter.generate_splits(pd.DatetimeIndex(dates))
        assert len(splits) >= 1


class TestWalkForwardEngine:
    @pytest.fixture
    def walk_forward_data(self, synthetic_features):
        """Prepare data suitable for walk-forward tests."""
        df = synthetic_features.copy()
        df["date"] = pd.to_datetime(df["date"])
        feature_cols = [
            "log_return_1d",
            "log_return_5d",
            "volatility_21d",
            "momentum_21d",
        ]
        prices = df[["date", "ticker", "adj_close"]].copy()
        return df, feature_cols, prices

    def test_engine_runs_with_ridge(self, walk_forward_data):
        df, feature_cols, prices = walk_forward_data

        config = WalkForwardConfig(
            window_type=WindowType.EXPANDING,
            train_days=200,
            val_days=50,
            test_days=50,
            step_days=50,
            min_train_days=100,
        )

        def ridge_factory(split, feature_df, feat_cols):
            datasets = create_flat_datasets(
                feature_df, feat_cols, split, target_col="log_return_1d"
            )
            X_tr, y_tr, _ = datasets["train"]
            X_te, y_te, meta_te = datasets["test"]

            model = RidgeBaseline(alpha=1.0)
            model.fit(X_tr, y_tr)

            signals = meta_te.copy()
            signals["signal"] = model.predict(X_te)
            return model, signals

        backtest_cfg = BacktestConfig(
            initial_capital=1_000_000,
            rebalance_frequency=5,
            top_n=5,
        )
        engine = WalkForwardEngine(config, backtest_cfg)
        result = engine.run(df, feature_cols, prices, ridge_factory)

        assert len(result.fold_results) >= 1
        assert len(result.aggregate_equity) > 0
        assert "sharpe" in result.aggregate_metrics
        assert "cagr" in result.aggregate_metrics
        assert len(result.per_fold_metrics) == len(result.fold_results)

    def test_rolling_window_engine(self, walk_forward_data):
        df, feature_cols, prices = walk_forward_data

        config = WalkForwardConfig(
            window_type=WindowType.ROLLING,
            train_days=100,
            val_days=50,
            test_days=50,
            step_days=50,
            min_train_days=50,
        )

        def ridge_factory(split, feature_df, feat_cols):
            datasets = create_flat_datasets(
                feature_df, feat_cols, split, target_col="log_return_1d"
            )
            X_tr, y_tr, _ = datasets["train"]
            X_te, y_te, meta_te = datasets["test"]

            model = RidgeBaseline(alpha=1.0)
            model.fit(X_tr, y_tr)

            signals = meta_te.copy()
            signals["signal"] = model.predict(X_te)
            return model, signals

        engine = WalkForwardEngine(config)
        result = engine.run(df, feature_cols, prices, ridge_factory)
        assert len(result.fold_results) >= 2

    def test_per_fold_metrics_has_all_folds(self, walk_forward_data):
        df, feature_cols, prices = walk_forward_data

        config = WalkForwardConfig(
            train_days=200,
            val_days=50,
            test_days=50,
            step_days=50,
            min_train_days=100,
        )

        def ridge_factory(split, feature_df, feat_cols):
            datasets = create_flat_datasets(
                feature_df, feat_cols, split, target_col="log_return_1d"
            )
            X_tr, y_tr, _ = datasets["train"]
            X_te, y_te, meta_te = datasets["test"]
            model = RidgeBaseline(alpha=1.0)
            model.fit(X_tr, y_tr)
            signals = meta_te.copy()
            signals["signal"] = model.predict(X_te)
            return model, signals

        engine = WalkForwardEngine(config)
        result = engine.run(df, feature_cols, prices, ridge_factory)

        assert "fold" in result.per_fold_metrics.columns
        assert "test_start" in result.per_fold_metrics.columns
        assert "sharpe" in result.per_fold_metrics.columns

    def test_no_valid_splits_raises(self, walk_forward_data):
        df, feature_cols, prices = walk_forward_data

        config = WalkForwardConfig(train_days=9999)
        engine = WalkForwardEngine(config)

        with pytest.raises(ValueError, match="No valid walk-forward"):
            engine.run(
                df, feature_cols, prices,
                lambda split, fd, fc: (None, pd.DataFrame()),
            )

    def test_aggregate_equity_starts_at_initial_capital(self, walk_forward_data):
        df, feature_cols, prices = walk_forward_data

        initial_capital = 500_000
        config = WalkForwardConfig(
            train_days=200,
            val_days=50,
            test_days=50,
            step_days=50,
            min_train_days=100,
        )
        backtest_cfg = BacktestConfig(initial_capital=initial_capital)

        def ridge_factory(split, feature_df, feat_cols):
            datasets = create_flat_datasets(
                feature_df, feat_cols, split, target_col="log_return_1d"
            )
            X_tr, y_tr, _ = datasets["train"]
            X_te, y_te, meta_te = datasets["test"]
            model = RidgeBaseline(alpha=1.0)
            model.fit(X_tr, y_tr)
            signals = meta_te.copy()
            signals["signal"] = model.predict(X_te)
            return model, signals

        engine = WalkForwardEngine(config, backtest_cfg)
        result = engine.run(df, feature_cols, prices, ridge_factory)
        assert result.aggregate_equity.iloc[0] == initial_capital

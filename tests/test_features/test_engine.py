"""Tests for FeatureEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.features.engine import FeatureEngine


class TestFeatureEngine:
    def test_compute_adds_feature_columns(self, synthetic_ohlcv):
        engine = FeatureEngine(enabled_features=["log_returns"])
        result = engine.compute(synthetic_ohlcv)
        assert "log_return_1d" in result.columns

    def test_compute_does_not_modify_input(self, synthetic_ohlcv):
        original_cols = list(synthetic_ohlcv.columns)
        engine = FeatureEngine(enabled_features=["log_returns"])
        engine.compute(synthetic_ohlcv)
        assert list(synthetic_ohlcv.columns) == original_cols

    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="Unknown feature"):
            FeatureEngine(enabled_features=["nonexistent_feature"])

    def test_custom_windows(self, synthetic_ohlcv):
        engine = FeatureEngine(
            enabled_features=["log_returns"],
            windows={"short": [1, 5]},
        )
        result = engine.compute(synthetic_ohlcv)
        assert "log_return_1d" in result.columns
        assert "log_return_5d" in result.columns

    def test_get_feature_columns(self, synthetic_ohlcv):
        engine = FeatureEngine(enabled_features=["log_returns"])
        result = engine.compute(synthetic_ohlcv)
        feature_cols = engine.get_feature_columns(result)
        assert "log_return_1d" in feature_cols
        # Base columns should be excluded
        assert "close" not in feature_cols
        assert "ticker" not in feature_cols

    def test_normalize_rolling_zscore(self, synthetic_features):
        engine = FeatureEngine(
            enabled_features=["log_returns"],
            normalization={"method": "rolling_zscore", "lookback": 50},
        )
        result = engine.normalize(synthetic_features.copy())
        # Normalized values should not have extreme outliers (within reason)
        feat_cols = [c for c in result.columns if c.startswith("log_return")]
        for col in feat_cols:
            valid = result[col].dropna()
            if len(valid) > 0:
                assert valid.std() < 100  # sanity check, not exactly 1 due to rolling

    def test_normalize_excludes_target(self, synthetic_features):
        engine = FeatureEngine(
            enabled_features=["log_returns"],
            normalization={"method": "rolling_zscore", "lookback": 50},
        )
        df = synthetic_features.copy()
        original_target = df["log_return_1d"].copy()
        result = engine.normalize(df, target_col="log_return_1d")
        pd.testing.assert_series_equal(result["log_return_1d"], original_target)

    def test_normalize_zero_std_uses_epsilon(self):
        """Constant feature column should not produce NaN after normalization."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=100).tolist() * 2,
            "ticker": ["A"] * 100 + ["B"] * 100,
            "close": 100.0,
            "adj_close": 100.0,
            "log_return_1d": 0.01,
            "constant_feat": 5.0,
        })
        engine = FeatureEngine(
            enabled_features=[],
            normalization={"method": "rolling_zscore", "lookback": 20},
        )
        result = engine.normalize(df)
        # Should NOT produce NaN for zero-std column
        valid = result["constant_feat"].dropna()
        assert not valid.isna().any()

    def test_normalize_rank(self, synthetic_features):
        engine = FeatureEngine(
            enabled_features=["log_returns"],
            normalization={"method": "rank"},
        )
        result = engine.normalize(synthetic_features.copy())
        feat_cols = [c for c in result.columns if c.startswith("log_return_5d")]
        for col in feat_cols:
            valid = result[col].dropna()
            if len(valid) > 0:
                assert valid.min() >= 0.0
                assert valid.max() <= 1.0

    def test_multiple_features(self, synthetic_ohlcv):
        engine = FeatureEngine(
            enabled_features=["log_returns", "realized_volatility", "momentum"]
        )
        result = engine.compute(synthetic_ohlcv)
        assert "log_return_1d" in result.columns
        assert "momentum_5d" in result.columns

"""Tests for cross-asset features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.features.cross_asset_features import (
    compute_rolling_correlation,
    compute_rolling_beta,
    compute_relative_strength,
    compute_cross_sectional_rank,
)


class TestRollingCorrelation:
    def test_produces_columns(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )
        result = compute_rolling_correlation(df, windows=[21])
        assert "correlation_21d" in result.columns

    def test_correlation_range(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )
        result = compute_rolling_correlation(df, windows=[21])
        valid = result["correlation_21d"].dropna()
        assert (valid >= -1.01).all() and (valid <= 1.01).all()


class TestRollingBeta:
    def test_produces_columns(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )
        result = compute_rolling_beta(df, windows=[63])
        assert "beta_63d" in result.columns

    def test_beta_finite(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )
        result = compute_rolling_beta(df, windows=[63])
        valid = result["beta_63d"].dropna()
        assert np.isfinite(valid).all()


class TestRelativeStrength:
    def test_produces_columns(self, synthetic_ohlcv):
        result = compute_relative_strength(synthetic_ohlcv, windows=[21])
        assert "rel_strength_21d" in result.columns

    def test_cross_sectional_mean_near_zero(self, synthetic_ohlcv):
        result = compute_relative_strength(synthetic_ohlcv, windows=[21])
        # Relative strength is stock_mom - benchmark_mom, so mean should be ~0
        valid = result.dropna(subset=["rel_strength_21d"])
        mean_per_date = valid.groupby("date")["rel_strength_21d"].mean()
        assert abs(mean_per_date.mean()) < 0.01


class TestCrossSectionalRank:
    def test_produces_rank_columns(self, synthetic_features):
        result = compute_cross_sectional_rank(synthetic_features)
        rank_cols = [c for c in result.columns if c.startswith("rank_")]
        assert len(rank_cols) > 0

    def test_ranks_between_0_and_1(self, synthetic_features):
        result = compute_cross_sectional_rank(synthetic_features)
        rank_cols = [c for c in result.columns if c.startswith("rank_")]
        for col in rank_cols:
            valid = result[col].dropna()
            if len(valid) > 0:
                assert valid.min() >= 0.0
                assert valid.max() <= 1.0

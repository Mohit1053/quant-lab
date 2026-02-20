"""Tests for regime signal features."""

from __future__ import annotations

import numpy as np

from quant_lab.features.regime_features import (
    compute_vol_regime,
    compute_volume_shock,
    compute_gap_stats,
    compute_breadth,
)


class TestVolRegime:
    def test_produces_column(self, synthetic_ohlcv):
        result = compute_vol_regime(synthetic_ohlcv)
        assert "vol_regime_ratio" in result.columns

    def test_ratio_positive(self, synthetic_ohlcv):
        result = compute_vol_regime(synthetic_ohlcv)
        valid = result["vol_regime_ratio"].dropna()
        assert (valid > 0).all()


class TestVolumeShock:
    def test_produces_columns(self, synthetic_ohlcv):
        result = compute_volume_shock(synthetic_ohlcv, windows=[5, 21])
        assert "volume_shock_5d" in result.columns
        assert "volume_shock_21d" in result.columns

    def test_shock_positive(self, synthetic_ohlcv):
        result = compute_volume_shock(synthetic_ohlcv, windows=[5])
        valid = result["volume_shock_5d"].dropna()
        assert (valid > 0).all()


class TestGapStats:
    def test_produces_columns(self, synthetic_ohlcv):
        result = compute_gap_stats(synthetic_ohlcv)
        assert "gap_mean_21d" in result.columns
        assert "gap_std_21d" in result.columns
        assert "gap_max_abs_21d" in result.columns

    def test_gap_std_non_negative(self, synthetic_ohlcv):
        result = compute_gap_stats(synthetic_ohlcv)
        valid = result["gap_std_21d"].dropna()
        assert (valid >= 0).all()

    def test_gap_max_abs_non_negative(self, synthetic_ohlcv):
        result = compute_gap_stats(synthetic_ohlcv)
        valid = result["gap_max_abs_21d"].dropna()
        assert (valid >= 0).all()


class TestBreadth:
    def test_produces_columns(self, synthetic_ohlcv):
        result = compute_breadth(synthetic_ohlcv)
        assert "market_breadth" in result.columns
        assert "adv_decline_ratio" in result.columns
        assert "breadth_ma_21d" in result.columns

    def test_breadth_between_0_and_1(self, synthetic_ohlcv):
        result = compute_breadth(synthetic_ohlcv)
        valid = result["market_breadth"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_adv_decline_non_negative(self, synthetic_ohlcv):
        result = compute_breadth(synthetic_ohlcv)
        valid = result["adv_decline_ratio"].dropna()
        assert (valid >= 0).all()

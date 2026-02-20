"""Tests for price-based features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.features.price_features import (
    compute_log_returns,
    compute_realized_volatility,
    compute_momentum,
    compute_max_drawdown,
)


@pytest.fixture
def simple_price_df():
    """Simple known-value DataFrame for testing."""
    dates = pd.bdate_range("2020-01-01", periods=30)
    prices = [100 + i for i in range(30)]  # Linear price increase

    return pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "open": prices,
        "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices],
        "close": prices,
        "volume": [1000000] * 30,
        "adj_close": prices,
    })


class TestLogReturns:
    def test_log_returns_computed(self, simple_price_df):
        result = compute_log_returns(simple_price_df, windows=[1, 5])
        assert "log_return_1d" in result.columns
        assert "log_return_5d" in result.columns

    def test_log_returns_first_values_nan(self, simple_price_df):
        result = compute_log_returns(simple_price_df, windows=[1])
        assert pd.isna(result["log_return_1d"].iloc[0])

    def test_log_returns_known_values(self, simple_price_df):
        result = compute_log_returns(simple_price_df, windows=[1])
        # From 100 to 101: log(101/100) ≈ 0.00995
        expected = np.log(101 / 100)
        actual = result["log_return_1d"].iloc[1]
        assert abs(actual - expected) < 1e-6


class TestRealizedVolatility:
    def test_volatility_computed(self, synthetic_ohlcv):
        result = compute_realized_volatility(synthetic_ohlcv, windows=[21])
        assert "volatility_21d" in result.columns

    def test_volatility_positive(self, synthetic_ohlcv):
        result = compute_realized_volatility(synthetic_ohlcv, windows=[21])
        valid = result["volatility_21d"].dropna()
        assert (valid >= 0).all()


class TestMomentum:
    def test_momentum_computed(self, simple_price_df):
        result = compute_momentum(simple_price_df, windows=[5])
        assert "momentum_5d" in result.columns

    def test_momentum_known_values(self, simple_price_df):
        result = compute_momentum(simple_price_df, windows=[5])
        # From 100 to 105: pct_change = 0.05
        expected = 5 / 100
        actual = result["momentum_5d"].iloc[5]
        assert abs(actual - expected) < 1e-6


class TestMaxDrawdown:
    def test_drawdown_computed(self, synthetic_ohlcv):
        result = compute_max_drawdown(synthetic_ohlcv, windows=[21])
        assert "max_drawdown_21d" in result.columns

    def test_drawdown_non_positive(self, synthetic_ohlcv):
        result = compute_max_drawdown(synthetic_ohlcv, windows=[21])
        valid = result["max_drawdown_21d"].dropna()
        assert (valid <= 0).all()

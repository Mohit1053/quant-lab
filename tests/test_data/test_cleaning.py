"""Tests for data cleaning pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.data.cleaning.pipeline import CleaningPipeline, CleaningConfig
from quant_lab.data.cleaning.validators import (
    validate_ohlc_relationships,
    validate_positive_prices,
)
from quant_lab.data.cleaning.transformers import (
    forward_fill_missing,
    remove_high_missing_tickers,
    remove_illiquid_tickers,
    remove_low_history_tickers,
)


class TestValidators:
    def test_validate_positive_prices_removes_negatives(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        df.loc[0, "close"] = -10.0
        result = validate_positive_prices(df)
        assert (result[["open", "high", "low", "close", "adj_close"]] > 0).all().all()

    def test_validate_ohlc_fixes_inconsistencies(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        # Make high < low
        df.loc[0, "high"] = 50.0
        df.loc[0, "low"] = 100.0
        result = validate_ohlc_relationships(df)
        assert (result["high"] >= result["low"]).all()


class TestTransformers:
    def test_forward_fill_fills_gaps(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        # Create a gap
        ticker_a = df[df["ticker"] == "TICKER_A"]
        idx = ticker_a.index[10]
        df.loc[idx, "close"] = np.nan
        result = forward_fill_missing(df, limit=5)
        # The NaN should be filled
        assert not result.loc[idx, "close"] != result.loc[idx, "close"]  # not NaN

    def test_remove_low_history_keeps_valid_tickers(self, synthetic_ohlcv):
        result = remove_low_history_tickers(synthetic_ohlcv, min_days=100)
        # All tickers have 504 days, so all should remain
        assert result["ticker"].nunique() == synthetic_ohlcv["ticker"].nunique()

    def test_remove_low_history_removes_short_tickers(self, synthetic_ohlcv):
        result = remove_low_history_tickers(synthetic_ohlcv, min_days=600)
        # No ticker has 600 days, so all should be removed
        assert len(result) == 0

    def test_remove_high_missing_tickers(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        # Make one ticker mostly NaN
        mask = df["ticker"] == "TICKER_A"
        nan_indices = df[mask].index[:400]
        df.loc[nan_indices, "close"] = np.nan
        result = remove_high_missing_tickers(df, max_missing_pct=0.20)
        assert "TICKER_A" not in result["ticker"].values


class TestLiquidityFilter:
    def test_removes_low_volume_tickers(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        # Set one ticker to have very low volume
        mask = df["ticker"] == "TICKER_A"
        df.loc[mask, "volume"] = 100
        result = remove_illiquid_tickers(df, min_avg_daily_volume=10_000)
        assert "TICKER_A" not in result["ticker"].values
        # Other tickers should survive
        assert result["ticker"].nunique() >= 3

    def test_removes_penny_stocks(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        # Set one ticker to penny stock prices
        mask = df["ticker"] == "TICKER_B"
        df.loc[mask, "close"] = 2.0
        result = remove_illiquid_tickers(df, min_median_price=5.0, min_avg_daily_volume=0)
        assert "TICKER_B" not in result["ticker"].values

    def test_removes_sparse_tickers(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        # Remove most data for one ticker (keep only 10%)
        mask = df["ticker"] == "TICKER_C"
        indices = df[mask].index
        drop_count = int(len(indices) * 0.92)
        df = df.drop(indices[:drop_count])
        result = remove_illiquid_tickers(
            df, min_trading_days_pct=0.50, min_avg_daily_volume=0,
        )
        assert "TICKER_C" not in result["ticker"].values

    def test_no_filter_keeps_all(self, synthetic_ohlcv):
        result = remove_illiquid_tickers(
            synthetic_ohlcv,
            min_avg_daily_volume=0,
            min_median_price=0.0,
            min_trading_days_pct=0.0,
        )
        assert result["ticker"].nunique() == synthetic_ohlcv["ticker"].nunique()

    def test_pipeline_with_liquidity_filter(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        # Set one ticker to penny stock prices
        mask = df["ticker"] == "TICKER_A"
        df.loc[mask, "close"] = 1.0
        config = CleaningConfig(
            min_history_days=100,
            min_median_price=5.0,
        )
        pipeline = CleaningPipeline(config)
        result = pipeline.run(df)
        assert "TICKER_A" not in result["ticker"].values


class TestCleaningPipeline:
    def test_pipeline_runs_without_error(self, synthetic_ohlcv):
        config = CleaningConfig(min_history_days=100)
        pipeline = CleaningPipeline(config)
        result = pipeline.run(synthetic_ohlcv)
        assert len(result) > 0
        assert result["ticker"].nunique() > 0

    def test_pipeline_sorts_output(self, synthetic_ohlcv):
        config = CleaningConfig(min_history_days=100)
        pipeline = CleaningPipeline(config)
        result = pipeline.run(synthetic_ohlcv)
        # Check sorted by ticker, then date
        for ticker in result["ticker"].unique():
            ticker_dates = result[result["ticker"] == ticker]["date"].values
            assert (np.diff(ticker_dates.astype(np.int64)) >= 0).all()

    def test_pipeline_with_dirty_data(self, synthetic_ohlcv_with_issues):
        config = CleaningConfig(min_history_days=100)
        pipeline = CleaningPipeline(config)
        result = pipeline.run(synthetic_ohlcv_with_issues)
        # Should still produce valid output
        assert len(result) > 0
        assert not result["close"].isna().any()

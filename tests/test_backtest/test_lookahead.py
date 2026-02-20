"""Tests for lookahead bias prevention."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_lab.backtest.lookahead_guard import (
    LookaheadGuard,
    LookaheadError,
    assert_no_lookahead,
)


def _make_sample_df():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    return pd.DataFrame({
        "date": dates.tolist() * 2,
        "ticker": ["AAPL"] * 10 + ["GOOG"] * 10,
        "close": list(range(100, 110)) + list(range(200, 210)),
    })


class TestLookaheadGuard:
    def test_get_data_filters_future(self):
        df = _make_sample_df()
        guard = LookaheadGuard(df)
        guard.set_current_date("2020-01-06")

        result = guard.get_data()
        assert (result["date"] <= pd.Timestamp("2020-01-06")).all()

    def test_get_data_without_date_raises(self):
        guard = LookaheadGuard(_make_sample_df())
        with pytest.raises(RuntimeError, match="No current date"):
            guard.get_data()

    def test_get_data_with_as_of(self):
        guard = LookaheadGuard(_make_sample_df())
        result = guard.get_data(as_of=pd.Timestamp("2020-01-03"))
        assert (result["date"] <= pd.Timestamp("2020-01-03")).all()

    def test_get_latest_per_ticker(self):
        guard = LookaheadGuard(_make_sample_df())
        guard.set_current_date("2020-01-06")
        latest = guard.get_latest()
        assert len(latest) == 2  # One row per ticker
        assert set(latest["ticker"]) == {"AAPL", "GOOG"}

    def test_get_data_returns_copy(self):
        df = _make_sample_df()
        guard = LookaheadGuard(df)
        guard.set_current_date("2020-01-10")
        result = guard.get_data()
        result["close"] = 999
        # Original should be unmodified
        result2 = guard.get_data()
        assert (result2["close"] != 999).any()

    def test_early_date_returns_empty_or_first(self):
        guard = LookaheadGuard(_make_sample_df())
        result = guard.get_data(as_of=pd.Timestamp("2019-01-01"))
        assert len(result) == 0


class TestAssertNoLookahead:
    def test_no_error_when_valid(self):
        assert_no_lookahead(
            signal_date=pd.Timestamp("2020-01-05"),
            data_date=pd.Timestamp("2020-01-03"),
        )

    def test_no_error_when_same_date(self):
        assert_no_lookahead(
            signal_date=pd.Timestamp("2020-01-05"),
            data_date=pd.Timestamp("2020-01-05"),
        )

    def test_raises_on_future_data(self):
        with pytest.raises(LookaheadError, match="Lookahead bias"):
            assert_no_lookahead(
                signal_date=pd.Timestamp("2020-01-05"),
                data_date=pd.Timestamp("2020-01-06"),
            )

    def test_context_in_error_message(self):
        with pytest.raises(LookaheadError, match="backtest"):
            assert_no_lookahead(
                signal_date=pd.Timestamp("2020-01-01"),
                data_date=pd.Timestamp("2020-01-02"),
                context="backtest",
            )

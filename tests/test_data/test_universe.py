"""Tests for universe definitions."""

from __future__ import annotations

import pytest

from quant_lab.data.universe import get_universe, UNIVERSES, Universe


class TestGetUniverse:
    def test_nifty50(self):
        u = get_universe("nifty50")
        assert u.name == "nifty50"
        assert len(u.tickers) == 50
        assert u.benchmark == "^NSEI"

    def test_nifty50_tickers_have_ns_suffix(self):
        u = get_universe("nifty50")
        for ticker in u.tickers:
            assert ticker.endswith(".NS"), f"{ticker} missing .NS suffix"

    def test_unknown_universe_raises(self):
        with pytest.raises(ValueError, match="Unknown universe"):
            get_universe("nonexistent_universe")

    def test_tickers_override(self):
        custom = ["AAPL", "GOOG"]
        u = get_universe("nifty50", tickers_override=custom)
        assert u.tickers == custom
        assert u.name == "nifty50"  # Name preserved

    def test_num_assets(self):
        u = get_universe("nifty50")
        assert u.num_assets == 50

    def test_nifty500_exists(self):
        u = get_universe("nifty500")
        assert u.name == "nifty500"

    def test_indian_market_exists(self):
        u = get_universe("indian_market")
        assert u.name == "indian_market"

    def test_all_universes_have_benchmark(self):
        for name in UNIVERSES:
            u = get_universe(name)
            assert u.benchmark, f"Universe {name} missing benchmark"

"""Tests for NSE constituent fetcher."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from quant_lab.data.sources.nse_source import (
    NSEConstituentFetcher,
    NSEFetchConfig,
)


@pytest.fixture
def mock_nse_response():
    """Simulates NSE API JSON response."""
    return {
        "data": [
            {"symbol": "NIFTY500"},  # index row, should be skipped
            {"symbol": "RELIANCE"},
            {"symbol": "TCS"},
            {"symbol": "INFY"},
            {"symbol": "HDFCBANK"},
        ]
    }


class TestNSEConstituentFetcher:
    def test_fetch_parses_symbols(self, tmp_path, mock_nse_response):
        config = NSEFetchConfig(cache_dir=str(tmp_path))
        fetcher = NSEConstituentFetcher(config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_nse_response
        mock_resp.raise_for_status = MagicMock()

        with patch("quant_lab.data.sources.nse_source.requests.Session") as MockSession:
            session = MockSession.return_value
            session.get.return_value = mock_resp
            session.headers = {}

            tickers = fetcher.fetch_constituents("NIFTY 500")

        assert "RELIANCE.NS" in tickers
        assert "TCS.NS" in tickers
        assert "INFY.NS" in tickers
        assert "HDFCBANK.NS" in tickers
        assert all(t.endswith(".NS") for t in tickers)
        # Index row should be filtered out
        assert "NIFTY500.NS" not in tickers

    def test_tickers_are_sorted(self, tmp_path, mock_nse_response):
        config = NSEFetchConfig(cache_dir=str(tmp_path))
        fetcher = NSEConstituentFetcher(config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_nse_response
        mock_resp.raise_for_status = MagicMock()

        with patch("quant_lab.data.sources.nse_source.requests.Session") as MockSession:
            session = MockSession.return_value
            session.get.return_value = mock_resp
            session.headers = {}

            tickers = fetcher.fetch_constituents("NIFTY 500")

        assert tickers == sorted(tickers)

    def test_cache_roundtrip(self, tmp_path):
        config = NSEFetchConfig(cache_dir=str(tmp_path), cache_ttl_days=30)
        fetcher = NSEConstituentFetcher(config)

        tickers = ["HDFCBANK.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS"]
        cache_path = fetcher._cache_path("NIFTY 500")
        fetcher._save_cache(tickers, cache_path, "NIFTY 500")
        loaded = fetcher._load_cache(cache_path)
        assert loaded == tickers

    def test_uses_fresh_cache(self, tmp_path):
        config = NSEFetchConfig(cache_dir=str(tmp_path), cache_ttl_days=30)
        fetcher = NSEConstituentFetcher(config)

        # Pre-populate cache
        cache_path = fetcher._cache_path("NIFTY 500")
        fetcher._save_cache(["CACHED.NS"], cache_path, "NIFTY 500")

        # Should use cache without making any network call
        tickers = fetcher.fetch_constituents("NIFTY 500")
        assert tickers == ["CACHED.NS"]

    def test_falls_back_to_stale_cache(self, tmp_path):
        config = NSEFetchConfig(cache_dir=str(tmp_path), cache_ttl_days=0)
        fetcher = NSEConstituentFetcher(config)

        # Pre-populate stale cache (ttl=0 means always stale)
        cache_path = fetcher._cache_path("NIFTY 500")
        fetcher._save_cache(["STALE.NS"], cache_path, "NIFTY 500")

        with patch("quant_lab.data.sources.nse_source.requests.Session") as MockSession:
            session = MockSession.return_value
            session.get.side_effect = Exception("Network error")
            session.headers = {}

            tickers = fetcher.fetch_constituents("NIFTY 500")

        assert tickers == ["STALE.NS"]

    def test_no_cache_no_network_raises(self, tmp_path):
        config = NSEFetchConfig(cache_dir=str(tmp_path))
        fetcher = NSEConstituentFetcher(config)

        with patch("quant_lab.data.sources.nse_source.requests.Session") as MockSession:
            session = MockSession.return_value
            session.get.side_effect = Exception("Network error")
            session.headers = {}

            with pytest.raises(Exception, match="Network error"):
                fetcher.fetch_constituents("NIFTY 500")

    def test_empty_response_raises(self, tmp_path):
        config = NSEFetchConfig(cache_dir=str(tmp_path))
        fetcher = NSEConstituentFetcher(config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("quant_lab.data.sources.nse_source.requests.Session") as MockSession:
            session = MockSession.return_value
            session.get.return_value = mock_resp
            session.headers = {}

            with pytest.raises(ValueError, match="No constituents found"):
                fetcher.fetch_constituents("NIFTY 500")

    def test_cache_path_safe_name(self, tmp_path):
        config = NSEFetchConfig(cache_dir=str(tmp_path))
        fetcher = NSEConstituentFetcher(config)

        path = fetcher._cache_path("NIFTY 500")
        assert "nifty_500" in path.name
        assert path.suffix == ".csv"

    def test_is_cache_fresh_no_file(self, tmp_path):
        config = NSEFetchConfig(cache_dir=str(tmp_path))
        fetcher = NSEConstituentFetcher(config)
        assert not fetcher._is_cache_fresh(tmp_path / "nonexistent.csv")


class TestLoadNifty500Tickers:
    def test_load_with_cache(self, tmp_path):
        """Test the convenience function with a pre-populated cache."""
        cache_dir = tmp_path / "universe_cache"
        cache_dir.mkdir()
        pd.DataFrame(
            {"ticker": ["RELIANCE.NS", "TCS.NS"], "index": "NIFTY 500"}
        ).to_csv(cache_dir / "nifty_500_constituents.csv", index=False)

        from quant_lab.data.universe import load_nifty500_tickers

        tickers = load_nifty500_tickers(str(cache_dir))
        assert tickers == ["RELIANCE.NS", "TCS.NS"]

"""Tests for data source modules."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_lab.data.sources.base_source import BaseDataSource
from quant_lab.data.sources.csv_source import CSVSource


class ConcreteSource(BaseDataSource):
    """Minimal concrete implementation for testing the abstract base."""

    def fetch(self, tickers, start, end):
        return pd.DataFrame({
            "date": ["2020-01-01"] * len(tickers),
            "ticker": tickers,
            "open": [100.0] * len(tickers),
            "high": [105.0] * len(tickers),
            "low": [95.0] * len(tickers),
            "close": [102.0] * len(tickers),
            "volume": [1000000] * len(tickers),
            "adj_close": [102.0] * len(tickers),
        })


class TestBaseDataSource:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseDataSource()

    def test_concrete_fetch(self):
        source = ConcreteSource()
        df = source.fetch(["AAPL", "GOOG"], "2020-01-01", "2020-12-31")
        assert len(df) == 2
        assert set(df["ticker"]) == {"AAPL", "GOOG"}

    def test_validate_schema_passes(self):
        source = ConcreteSource()
        df = source.fetch(["AAPL"], "2020-01-01", "2020-12-31")
        validated = source.validate_schema(df)
        assert pd.api.types.is_datetime64_any_dtype(validated["date"])

    def test_validate_schema_missing_columns(self):
        source = ConcreteSource()
        df = pd.DataFrame({"date": ["2020-01-01"], "ticker": ["AAPL"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            source.validate_schema(df)

    def test_required_columns_list(self):
        assert "date" in BaseDataSource.REQUIRED_COLUMNS
        assert "ticker" in BaseDataSource.REQUIRED_COLUMNS
        assert "adj_close" in BaseDataSource.REQUIRED_COLUMNS


class TestCSVSource:
    def test_csv_source_instantiates(self):
        source = CSVSource(data_dir="/nonexistent")
        assert source is not None

    def test_csv_source_fetch_missing_files_returns_empty(self, tmp_path):
        data_dir = tmp_path / "csv_data"
        data_dir.mkdir()
        source = CSVSource(data_dir=str(data_dir))
        df = source.fetch(["AAPL"], "2020-01-01", "2020-12-31")
        assert len(df) == 0

    def test_csv_source_loads_from_file(self, tmp_path):
        data_dir = tmp_path / "csv_data"
        data_dir.mkdir()
        # Create a sample CSV
        csv_df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5, freq="B"),
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "volume": [1e6] * 5,
            "adj_close": [102, 103, 104, 105, 106],
        })
        csv_df.to_csv(data_dir / "TEST.csv", index=False)

        source = CSVSource(data_dir=str(data_dir))
        df = source.fetch(["TEST"], "2020-01-01", "2020-12-31")
        assert len(df) == 5
        assert "ticker" in df.columns

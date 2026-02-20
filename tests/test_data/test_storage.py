"""Tests for Parquet storage and caching."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_lab.data.storage.parquet_store import ParquetStore
from quant_lab.data.storage.cache import load_cached, clear_cache


class TestParquetStore:
    def test_save_and_load(self, tmp_path):
        store = ParquetStore(tmp_path / "store")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        store.save(df, "test_data")

        loaded = store.load("test_data")
        assert len(loaded) == 3
        assert list(loaded.columns) == ["a", "b"]

    def test_exists(self, tmp_path):
        store = ParquetStore(tmp_path / "store")
        assert not store.exists("missing")

        store.save(pd.DataFrame({"x": [1]}), "present")
        assert store.exists("present")

    def test_load_nonexistent_raises(self, tmp_path):
        store = ParquetStore(tmp_path / "store")
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_list_files(self, tmp_path):
        store = ParquetStore(tmp_path / "store")
        store.save(pd.DataFrame({"x": [1]}), "file_a")
        store.save(pd.DataFrame({"y": [2]}), "file_b")

        files = store.list_files()
        assert "file_a" in files
        assert "file_b" in files

    def test_creates_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        store = ParquetStore(nested)
        assert nested.exists()

    def test_round_trip_dtypes(self, tmp_path):
        store = ParquetStore(tmp_path / "store")
        df = pd.DataFrame({
            "int_col": [1, 2],
            "float_col": [1.5, 2.5],
            "str_col": ["a", "b"],
        })
        store.save(df, "typed")
        loaded = store.load("typed")
        assert loaded["int_col"].dtype in ("int64", "int32")
        assert loaded["float_col"].dtype == "float64"


class TestCache:
    def test_load_cached_returns_df(self, tmp_path):
        clear_cache()
        df = pd.DataFrame({"x": [1, 2, 3]})
        path = tmp_path / "cached.parquet"
        df.to_parquet(path)

        loaded = load_cached(path)
        assert len(loaded) == 3

    def test_clear_cache_works(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        path = tmp_path / "cached2.parquet"
        df.to_parquet(path)

        load_cached(path)
        clear_cache()
        # Should not raise after clear
        loaded = load_cached(path)
        assert len(loaded) == 1

"""In-memory LRU cache for frequently accessed data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=16)
def _load_parquet_cached(path: str, mtime: float) -> pd.DataFrame:
    """Load parquet with cache keyed on path + modification time."""
    return pd.read_parquet(path, engine="pyarrow")


def load_cached(path: str | Path) -> pd.DataFrame:
    """Load a parquet file with LRU caching. Cache invalidates on file change."""
    path = Path(path)
    mtime = path.stat().st_mtime
    return _load_parquet_cached(str(path), mtime)


def clear_cache() -> None:
    """Clear the LRU cache."""
    _load_parquet_cached.cache_clear()

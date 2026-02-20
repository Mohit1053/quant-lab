"""Persist and load computed features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

from quant_lab.data.storage.parquet_store import ParquetStore

logger = structlog.get_logger(__name__)


class FeatureStore:
    """Manages storage of computed feature matrices."""

    def __init__(self, base_dir: str | Path):
        self.store = ParquetStore(base_dir)

    def save_features(self, df: pd.DataFrame, name: str = "features") -> Path:
        """Save the full feature DataFrame."""
        return self.store.save(df, name)

    def load_features(self, name: str = "features") -> pd.DataFrame:
        """Load a previously saved feature DataFrame."""
        return self.store.load(name)

    def has_features(self, name: str = "features") -> bool:
        """Check if features have been computed and saved."""
        return self.store.exists(name)

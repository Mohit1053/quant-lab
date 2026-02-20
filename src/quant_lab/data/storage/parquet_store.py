"""Parquet-based data storage for all pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class ParquetStore:
    """Read/write DataFrames as Parquet files with consistent conventions."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, df: pd.DataFrame, name: str) -> Path:
        """Save a DataFrame as a Parquet file."""
        path = self.base_dir / f"{name}.parquet"
        df.to_parquet(path, engine="pyarrow", index=False)
        logger.info("parquet_saved", path=str(path), rows=len(df), cols=len(df.columns))
        return path

    def load(self, name: str) -> pd.DataFrame:
        """Load a DataFrame from a Parquet file."""
        path = self.base_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("parquet_loaded", path=str(path), rows=len(df), cols=len(df.columns))
        return df

    def exists(self, name: str) -> bool:
        """Check if a Parquet file exists."""
        return (self.base_dir / f"{name}.parquet").exists()

    def list_files(self) -> list[str]:
        """List all available Parquet files."""
        return [p.stem for p in self.base_dir.glob("*.parquet")]

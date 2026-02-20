"""CSV file data source for professional data feeds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

from quant_lab.data.sources.base_source import BaseDataSource

logger = structlog.get_logger(__name__)


class CSVSource(BaseDataSource):
    """Loads OHLCV data from CSV files.

    Expected file structure: one CSV per ticker, or a single CSV with ticker column.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Load data from CSV files for the given tickers and date range."""
        records = []
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        for ticker in tickers:
            csv_path = self.data_dir / f"{ticker}.csv"
            if not csv_path.exists():
                logger.warning("csv_not_found", ticker=ticker, path=str(csv_path))
                continue

            df = pd.read_csv(csv_path, parse_dates=["date"])
            df["ticker"] = ticker
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
            records.append(df)

        if not records:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        result = pd.concat(records, ignore_index=True)
        return self.validate_schema(result)

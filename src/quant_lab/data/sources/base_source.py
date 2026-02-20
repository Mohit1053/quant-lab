"""Abstract base class for all data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseDataSource(ABC):
    """Interface for market data sources.

    All data sources must return a DataFrame with columns:
    [date, ticker, open, high, low, close, volume, adj_close]
    """

    REQUIRED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"]

    @abstractmethod
    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for given tickers and date range.

        Returns:
            DataFrame with columns: date, ticker, open, high, low, close, volume, adj_close
        """
        ...

    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that the DataFrame has the required columns."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "adj_close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

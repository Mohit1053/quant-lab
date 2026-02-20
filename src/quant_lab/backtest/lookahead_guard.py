"""Lookahead bias prevention utilities."""

from __future__ import annotations

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class LookaheadGuard:
    """Prevents accidental use of future data in backtesting.

    Wraps a DataFrame and only allows access to data up to the current date.
    """

    def __init__(self, df: pd.DataFrame, date_col: str = "date"):
        self._df = df.copy()
        self._df[date_col] = pd.to_datetime(self._df[date_col])
        self._date_col = date_col
        self._current_date: pd.Timestamp | None = None

    def set_current_date(self, date: pd.Timestamp) -> None:
        """Set the current simulation date."""
        self._current_date = pd.Timestamp(date)

    def get_data(self, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """Get data available as of the given date (no future data)."""
        date = as_of or self._current_date
        if date is None:
            raise RuntimeError("No current date set. Call set_current_date() first.")
        return self._df[self._df[self._date_col] <= date].copy()

    def get_latest(self, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """Get the most recent row per ticker, as of the given date."""
        data = self.get_data(as_of)
        if "ticker" in data.columns:
            return data.groupby("ticker").last().reset_index()
        return data.iloc[[-1]]


def assert_no_lookahead(
    signal_date: pd.Timestamp,
    data_date: pd.Timestamp,
    context: str = "",
) -> None:
    """Assert that signal_date does not use future data.

    signal_date: The date the signal is generated for.
    data_date: The latest data date used to generate the signal.
    """
    if data_date > signal_date:
        raise LookaheadError(
            f"Lookahead bias detected{' in ' + context if context else ''}: "
            f"signal_date={signal_date.date()}, data_date={data_date.date()}"
        )


class LookaheadError(Exception):
    """Raised when lookahead bias is detected."""

    pass

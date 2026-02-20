"""Data quality validation checks."""

from __future__ import annotations

import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def validate_ohlc_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLC price relationships: H >= L, H >= O, H >= C, etc.

    Rows with invalid relationships are flagged but not removed.
    """
    invalid_mask = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        logger.warning("invalid_ohlc", count=int(n_invalid))
        # Fix: clamp high/low to contain open/close
        df.loc[invalid_mask, "high"] = df.loc[invalid_mask, ["open", "high", "low", "close"]].max(
            axis=1
        )
        df.loc[invalid_mask, "low"] = df.loc[invalid_mask, ["open", "high", "low", "close"]].min(
            axis=1
        )

    return df


def validate_positive_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with non-positive prices."""
    price_cols = ["open", "high", "low", "close", "adj_close"]
    mask = (df[price_cols] > 0).all(axis=1)
    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.warning("non_positive_prices_removed", count=int(n_removed))
    return df[mask].copy()


def validate_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Replace negative volumes with NaN."""
    neg_mask = df["volume"] < 0
    if neg_mask.any():
        logger.warning("negative_volume_fixed", count=int(neg_mask.sum()))
        df.loc[neg_mask, "volume"] = np.nan
    return df


def check_missing_rate(df: pd.DataFrame, ticker: str) -> float:
    """Return the fraction of missing values for a ticker's close prices."""
    ticker_data = df[df["ticker"] == ticker]["close"]
    if len(ticker_data) == 0:
        return 1.0
    return ticker_data.isna().mean()

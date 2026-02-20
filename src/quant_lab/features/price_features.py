"""Price-based features: returns, volatility, momentum, drawdown."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.features.registry import register_feature


@register_feature("log_returns", "Log returns at multiple horizons")
def compute_log_returns(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Compute log returns for each ticker at specified horizons."""
    if windows is None:
        windows = [1, 5, 21]

    df = df.sort_values(["ticker", "date"])

    for w in windows:
        col_name = f"log_return_{w}d"
        df[col_name] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(w))
        )

    return df


@register_feature("realized_volatility", "Rolling realized volatility")
def compute_realized_volatility(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Compute annualized rolling volatility of daily log returns."""
    if windows is None:
        windows = [21]

    df = df.sort_values(["ticker", "date"])

    # Ensure we have daily log returns
    if "log_return_1d" not in df.columns:
        df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    for w in windows:
        if w < 2:
            continue  # std requires at least 2 observations
        col_name = f"volatility_{w}d"
        df[col_name] = df.groupby("ticker")["log_return_1d"].transform(
            lambda s: s.rolling(window=w, min_periods=max(w // 2, 2)).std() * np.sqrt(252)
        )

    return df


@register_feature("momentum", "Price momentum at multiple horizons")
def compute_momentum(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Compute momentum as cumulative return over the lookback window."""
    if windows is None:
        windows = [5, 21, 63]

    df = df.sort_values(["ticker", "date"])

    for w in windows:
        col_name = f"momentum_{w}d"
        df[col_name] = df.groupby("ticker")["adj_close"].transform(
            lambda s: s.pct_change(periods=w)
        )

    return df


@register_feature("max_drawdown", "Rolling maximum drawdown")
def compute_max_drawdown(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Compute rolling maximum drawdown for each ticker."""
    if windows is None:
        windows = [63]

    df = df.sort_values(["ticker", "date"])

    for w in windows:
        col_name = f"max_drawdown_{w}d"
        df[col_name] = df.groupby("ticker")["adj_close"].transform(
            lambda s: _rolling_max_drawdown(s, w)
        )

    return df


def _rolling_max_drawdown(prices: pd.Series, window: int) -> pd.Series:
    """Compute rolling max drawdown over a window (vectorized)."""
    rolling_max = prices.rolling(window=window, min_periods=1).max()
    drawdown = (prices - rolling_max) / rolling_max.replace(0, np.nan)
    min_periods = min(2, window)
    rolling_min_dd = drawdown.rolling(window=window, min_periods=min_periods).min()
    return rolling_min_dd.fillna(0.0)

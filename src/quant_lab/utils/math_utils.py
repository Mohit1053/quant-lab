"""Financial math helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute log returns from a price series."""
    return np.log(prices / prices.shift(1))


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute simple (arithmetic) returns from a price series."""
    return prices.pct_change()


def rolling_volatility(
    returns: pd.Series, window: int = 21, annualize: bool = True
) -> pd.Series:
    """Compute rolling annualized volatility."""
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def rolling_max_drawdown(prices: pd.Series, window: int = 63) -> pd.Series:
    """Compute rolling maximum drawdown over a window."""
    rolling_max = prices.rolling(window=window, min_periods=1).max()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown.rolling(window=window, min_periods=1).min()


def rolling_sharpe(
    returns: pd.Series, window: int = 252, risk_free_daily: float = 0.0
) -> pd.Series:
    """Compute rolling Sharpe ratio."""
    excess = returns - risk_free_daily
    mean = excess.rolling(window=window).mean()
    std = excess.rolling(window=window).std()
    return (mean / std.replace(0, np.nan)) * np.sqrt(252)

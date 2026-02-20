"""Regime signal features: volatility regime, volume shocks, gaps, breadth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.features.registry import register_feature


@register_feature("vol_regime", "Volatility regime ratio (short-term / long-term vol)")
def compute_vol_regime(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Short-term vol / long-term vol ratio. High values signal regime shift."""
    df = df.sort_values(["ticker", "date"])

    if "log_return_1d" not in df.columns:
        df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    short_w, long_w = 21, 126

    short_vol = df.groupby("ticker")["log_return_1d"].transform(
        lambda s: s.rolling(short_w, min_periods=short_w // 2).std()
    )
    long_vol = df.groupby("ticker")["log_return_1d"].transform(
        lambda s: s.rolling(long_w, min_periods=long_w // 2).std()
    )
    df["vol_regime_ratio"] = short_vol / long_vol.replace(0, np.nan)

    return df


@register_feature("volume_shock", "Abnormal volume detection")
def compute_volume_shock(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Current volume / rolling average volume. Values > 2.0 are abnormal."""
    if windows is None:
        windows = [5, 21]

    df = df.sort_values(["ticker", "date"])

    for w in windows:
        if w < 2:
            continue
        col_name = f"volume_shock_{w}d"
        avg_vol = df.groupby("ticker")["volume"].transform(
            lambda s: s.rolling(w, min_periods=max(w // 2, 2)).mean()
        )
        df[col_name] = df["volume"] / avg_vol.replace(0, np.nan)

    return df


@register_feature("gap_stats", "Overnight gap statistics")
def compute_gap_stats(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Overnight gap (open vs prev close) rolling statistics."""
    df = df.sort_values(["ticker", "date"])

    # Compute overnight gap
    prev_close = df.groupby("ticker")["close"].shift(1)
    gap = (df["open"] - prev_close) / prev_close.replace(0, np.nan)

    w = 21  # Fixed window for gap statistics
    df["_gap"] = gap
    df["gap_mean_21d"] = df.groupby("ticker")["_gap"].transform(
        lambda s: s.rolling(w, min_periods=w // 2).mean()
    )
    df["gap_std_21d"] = df.groupby("ticker")["_gap"].transform(
        lambda s: s.rolling(w, min_periods=max(w // 2, 2)).std()
    )
    df["gap_max_abs_21d"] = df.groupby("ticker")["_gap"].transform(
        lambda s: s.abs().rolling(w, min_periods=w // 2).max()
    )
    df = df.drop(columns=["_gap"])

    return df


@register_feature("breadth", "Market breadth indicators")
def compute_breadth(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Market breadth: fraction of stocks with positive daily returns."""
    df = df.sort_values(["ticker", "date"])

    if "log_return_1d" not in df.columns:
        df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    # Fraction of stocks positive each day
    breadth = df.groupby("date")["log_return_1d"].transform(
        lambda x: (x > 0).mean()
    )
    df["market_breadth"] = breadth

    # Advance-decline ratio
    adv = df.groupby("date")["log_return_1d"].transform(lambda x: (x > 0).sum())
    dec = df.groupby("date")["log_return_1d"].transform(lambda x: (x < 0).sum())
    df["adv_decline_ratio"] = adv / dec.replace(0, np.nan)

    # Smoothed breadth
    df["breadth_ma_21d"] = df.groupby("ticker")["market_breadth"].transform(
        lambda s: s.rolling(21, min_periods=10).mean()
    )

    return df

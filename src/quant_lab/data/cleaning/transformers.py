"""Data cleaning transformations - fill, cap, filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def forward_fill_missing(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Forward-fill missing prices within each ticker, up to a limit."""
    price_cols = ["open", "high", "low", "close", "adj_close"]
    df = df.sort_values(["ticker", "date"])

    for col in price_cols:
        df[col] = df.groupby("ticker")[col].transform(
            lambda s: s.ffill(limit=limit)
        )

    # Forward-fill volume separately (can have more gaps)
    df["volume"] = df.groupby("ticker")["volume"].transform(
        lambda s: s.ffill(limit=limit)
    )

    return df


def cap_outliers(df: pd.DataFrame, sigma: float = 10.0) -> pd.DataFrame:
    """Cap extreme daily returns beyond sigma standard deviations.

    Computes returns per ticker, caps at +/- sigma * std, then reconstructs prices.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    returns = df.groupby("ticker")["close"].pct_change()

    # Compute per-ticker stats
    grouped = returns.groupby(df["ticker"])
    mean = grouped.transform("mean")
    std = grouped.transform("std")

    upper = mean + sigma * std
    lower = mean - sigma * std

    outlier_mask = (returns > upper) | (returns < lower)
    n_capped = outlier_mask.sum()

    if n_capped > 0:
        logger.info("outliers_capped", count=int(n_capped), sigma=sigma)
        capped_returns = returns.clip(lower=lower, upper=upper)

        # Reconstruct close prices from capped returns
        for ticker in df["ticker"].unique():
            mask = df["ticker"] == ticker
            ticker_idx = df.index[mask]
            first_close = df.loc[ticker_idx[0], "close"]
            ticker_capped = capped_returns.loc[ticker_idx].fillna(0.0)
            reconstructed = first_close * (1 + ticker_capped).cumprod()
            reconstructed.iloc[0] = first_close
            df.loc[ticker_idx, "close"] = reconstructed

            # Scale adj_close proportionally
            if "adj_close" in df.columns:
                orig_close = df.loc[ticker_idx, "close"]
                # Avoid division by zero
                safe_orig = orig_close.replace(0, np.nan)
                ratio = df.loc[ticker_idx, "adj_close"] / safe_orig
                df.loc[ticker_idx, "adj_close"] = reconstructed * ratio.fillna(1.0)

    return df


def remove_low_history_tickers(
    df: pd.DataFrame, min_days: int = 252
) -> pd.DataFrame:
    """Remove tickers with fewer than min_days of data."""
    counts = df.groupby("ticker")["date"].count()
    valid_tickers = counts[counts >= min_days].index.tolist()
    removed = counts[counts < min_days].index.tolist()

    if removed:
        logger.info("tickers_removed_low_history", removed=removed, min_days=min_days)

    return df[df["ticker"].isin(valid_tickers)].copy()


def remove_high_missing_tickers(
    df: pd.DataFrame, max_missing_pct: float = 0.20
) -> pd.DataFrame:
    """Remove tickers with too many missing close prices."""
    missing_rates = df.groupby("ticker")["close"].apply(lambda s: s.isna().mean())
    valid_tickers = missing_rates[missing_rates <= max_missing_pct].index.tolist()
    removed = missing_rates[missing_rates > max_missing_pct].index.tolist()

    if removed:
        logger.info(
            "tickers_removed_missing",
            removed=removed,
            max_missing_pct=max_missing_pct,
        )

    return df[df["ticker"].isin(valid_tickers)].copy()


def remove_illiquid_tickers(
    df: pd.DataFrame,
    min_avg_daily_volume: float = 10_000,
    min_median_price: float = 5.0,
    min_trading_days_pct: float = 0.50,
) -> pd.DataFrame:
    """Remove illiquid tickers based on volume, price, and trading day filters.

    Particularly important when expanding to full market (2000+ stocks)
    where many micro-caps have negligible liquidity.

    Args:
        df: DataFrame with columns [ticker, date, close, volume].
        min_avg_daily_volume: Minimum average daily volume across history.
        min_median_price: Minimum median close price (filters penny stocks).
        min_trading_days_pct: Minimum fraction of total trading days the
            ticker must have data for (vs the most active ticker).
    """
    stats = df.groupby("ticker").agg(
        avg_volume=("volume", "mean"),
        median_price=("close", "median"),
        n_days=("date", "count"),
    )

    max_days = stats["n_days"].max()
    stats["trading_pct"] = stats["n_days"] / max_days

    # Apply all three filters
    valid = stats[
        (stats["avg_volume"] >= min_avg_daily_volume)
        & (stats["median_price"] >= min_median_price)
        & (stats["trading_pct"] >= min_trading_days_pct)
    ].index.tolist()

    removed_volume = stats[stats["avg_volume"] < min_avg_daily_volume].index.tolist()
    removed_price = stats[stats["median_price"] < min_median_price].index.tolist()
    removed_trading = stats[stats["trading_pct"] < min_trading_days_pct].index.tolist()

    n_removed = len(stats) - len(valid)
    if n_removed > 0:
        logger.info(
            "tickers_removed_illiquid",
            total_removed=n_removed,
            remaining=len(valid),
            low_volume=len(removed_volume),
            penny_stock=len(removed_price),
            low_trading_days=len(removed_trading),
            min_avg_volume=min_avg_daily_volume,
            min_median_price=min_median_price,
            min_trading_pct=min_trading_days_pct,
        )

    return df[df["ticker"].isin(valid)].copy()

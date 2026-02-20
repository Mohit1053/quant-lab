"""Cross-asset features: correlations, beta, relative strength, ranks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.features.registry import register_feature


def _get_benchmark_returns(df: pd.DataFrame) -> pd.Series:
    """Compute equal-weight market benchmark returns from all tickers.

    Uses the cross-sectional mean of daily log returns as the benchmark.
    This avoids needing external benchmark data.
    """
    if "log_return_1d" not in df.columns:
        df = df.sort_values(["ticker", "date"])
        df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    benchmark = df.groupby("date")["log_return_1d"].mean()
    benchmark.name = "_benchmark_return"
    return benchmark


@register_feature("rolling_correlation", "Rolling correlation with market benchmark")
def compute_rolling_correlation(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Rolling Pearson correlation between each stock and the market benchmark."""
    if windows is None:
        windows = [21, 63]

    df = df.sort_values(["ticker", "date"])

    if "log_return_1d" not in df.columns:
        df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    benchmark = _get_benchmark_returns(df)
    df = df.merge(benchmark.reset_index(), on="date", how="left")

    for w in windows:
        if w < 5:
            continue
        col_name = f"correlation_{w}d"
        # Per-ticker rolling correlation with benchmark
        df[col_name] = df.groupby("ticker").apply(
            lambda g: g["log_return_1d"].rolling(w, min_periods=w // 2).corr(g["_benchmark_return"]),
            include_groups=False,
        ).reset_index(level=0, drop=True)

    df = df.drop(columns=["_benchmark_return"], errors="ignore")
    return df


@register_feature("rolling_beta", "Rolling CAPM beta against market")
def compute_rolling_beta(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Rolling beta: Cov(Ri, Rm) / Var(Rm) against equal-weight benchmark."""
    if windows is None:
        windows = [63]

    df = df.sort_values(["ticker", "date"])

    if "log_return_1d" not in df.columns:
        df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    benchmark = _get_benchmark_returns(df)
    df = df.merge(benchmark.reset_index(), on="date", how="left")

    for w in windows:
        if w < 10:
            continue
        col_name = f"beta_{w}d"

        def _rolling_beta(g):
            cov = g["log_return_1d"].rolling(w, min_periods=w // 2).cov(g["_benchmark_return"])
            var = g["_benchmark_return"].rolling(w, min_periods=w // 2).var()
            return cov / var.replace(0, np.nan)

        df[col_name] = df.groupby("ticker").apply(
            _rolling_beta, include_groups=False,
        ).reset_index(level=0, drop=True)

    df = df.drop(columns=["_benchmark_return"], errors="ignore")
    return df


@register_feature("relative_strength", "Relative strength vs market benchmark")
def compute_relative_strength(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Relative strength: stock cumulative return / benchmark cumulative return."""
    if windows is None:
        windows = [21, 63]

    df = df.sort_values(["ticker", "date"])

    for w in windows:
        if w < 2:
            continue
        col_name = f"rel_strength_{w}d"

        # Stock momentum
        stock_mom = df.groupby("ticker")["adj_close"].transform(
            lambda s: s.pct_change(periods=w)
        )
        # Benchmark momentum (equal-weight average of all stocks' momentum)
        bm_mom = stock_mom.groupby(df["date"]).transform("mean")
        df[col_name] = (stock_mom - bm_mom)

    return df


@register_feature("cross_sectional_rank", "Cross-sectional percentile rank of features")
def compute_cross_sectional_rank(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Rank each stock's features relative to peers at each date.

    Computes percentile rank [0, 1] for key features cross-sectionally.
    """
    df = df.sort_values(["ticker", "date"])

    # Rank available return/momentum/volatility columns
    rank_targets = [
        c for c in df.columns
        if c.startswith(("log_return_", "momentum_", "volatility_"))
        and not c.startswith("rank_")
    ]

    for col in rank_targets:
        rank_col = f"rank_{col}"
        if rank_col not in df.columns:
            df[rank_col] = df.groupby("date")[col].rank(pct=True)

    return df

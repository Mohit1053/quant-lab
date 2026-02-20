"""Shared test fixtures - synthetic data for all tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for all tests (numpy + torch)."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate 504 days (2 years) of synthetic OHLCV data for 5 tickers."""
    np.random.seed(42)
    n_days = 504
    tickers = ["TICKER_A", "TICKER_B", "TICKER_C", "TICKER_D", "TICKER_E"]
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)

    records = []
    for ticker in tickers:
        # Random walk for price
        returns = np.random.normal(0.0005, 0.02, n_days)
        price = 100.0 * np.exp(np.cumsum(returns))

        for i, date in enumerate(dates):
            p = price[i]
            high = p * (1 + abs(np.random.normal(0, 0.01)))
            low = p * (1 - abs(np.random.normal(0, 0.01)))
            open_p = p * (1 + np.random.normal(0, 0.005))
            # Ensure OHLC consistency
            high = max(high, open_p, p)
            low = min(low, open_p, p)

            records.append({
                "date": date,
                "ticker": ticker,
                "open": round(open_p, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(p, 2),
                "volume": int(np.random.uniform(100000, 10000000)),
                "adj_close": round(p, 2),
            })

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_ohlcv_with_issues(synthetic_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Synthetic data with quality issues for cleaning tests."""
    df = synthetic_ohlcv.copy()

    # Add some NaN values
    nan_indices = np.random.choice(len(df), size=20, replace=False)
    df.loc[nan_indices, "close"] = np.nan

    # Add a negative price
    df.loc[5, "close"] = -10.0

    # Add OHLC inconsistency
    df.loc[10, "high"] = df.loc[10, "low"] - 5.0

    return df


@pytest.fixture
def synthetic_features(synthetic_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Synthetic data with pre-computed features."""
    df = synthetic_ohlcv.copy()
    df = df.sort_values(["ticker", "date"])

    # Add basic features
    df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    df["log_return_5d"] = df.groupby("ticker")["adj_close"].transform(
        lambda s: np.log(s / s.shift(5))
    )
    df["volatility_21d"] = df.groupby("ticker")["log_return_1d"].transform(
        lambda s: s.rolling(21).std() * np.sqrt(252)
    )
    df["momentum_21d"] = df.groupby("ticker")["adj_close"].transform(
        lambda s: s.pct_change(21)
    )

    return df


@pytest.fixture
def tiny_model_config() -> dict:
    """Small model config for fast testing."""
    return {
        "d_model": 16,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
    }

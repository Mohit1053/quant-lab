"""Data cleaning pipeline - orchestrates all cleaning steps."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import structlog

from quant_lab.data.cleaning.validators import (
    validate_ohlc_relationships,
    validate_positive_prices,
    validate_volume,
)
from quant_lab.data.cleaning.transformers import (
    cap_outliers,
    forward_fill_missing,
    remove_high_missing_tickers,
    remove_illiquid_tickers,
    remove_low_history_tickers,
)

logger = structlog.get_logger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for the cleaning pipeline."""

    max_missing_pct: float = 0.20
    ffill_limit: int = 5
    outlier_sigma: float = 10.0
    min_history_days: int = 252
    # Liquidity filters (active when any value > 0)
    min_avg_daily_volume: float = 0
    min_median_price: float = 0.0
    min_trading_days_pct: float = 0.0


class CleaningPipeline:
    """Sequential data cleaning pipeline."""

    def __init__(self, config: CleaningConfig | None = None):
        self.config = config or CleaningConfig()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute all cleaning steps in order."""
        initial_tickers = df["ticker"].nunique()
        initial_rows = len(df)

        logger.info(
            "cleaning_start",
            tickers=initial_tickers,
            rows=initial_rows,
        )

        # Step 1: Basic validation
        df = validate_positive_prices(df)
        df = validate_volume(df)
        df = validate_ohlc_relationships(df)

        # Step 2: Remove tickers with too much missing data
        df = remove_high_missing_tickers(df, self.config.max_missing_pct)

        # Step 3: Forward-fill small gaps
        df = forward_fill_missing(df, self.config.ffill_limit)

        # Step 4: Cap extreme outliers
        df = cap_outliers(df, self.config.outlier_sigma)

        # Step 5: Remove tickers with insufficient history
        df = remove_low_history_tickers(df, self.config.min_history_days)

        # Step 6: Liquidity filter (only if thresholds are set)
        if (
            self.config.min_avg_daily_volume > 0
            or self.config.min_median_price > 0
            or self.config.min_trading_days_pct > 0
        ):
            df = remove_illiquid_tickers(
                df,
                min_avg_daily_volume=self.config.min_avg_daily_volume,
                min_median_price=self.config.min_median_price,
                min_trading_days_pct=self.config.min_trading_days_pct,
            )

        # Step 7: Drop any remaining NaN rows in close price
        df = df.dropna(subset=["close"])

        # Sort for consistent output
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        final_tickers = df["ticker"].nunique()
        final_rows = len(df)

        logger.info(
            "cleaning_complete",
            tickers_before=initial_tickers,
            tickers_after=final_tickers,
            rows_before=initial_rows,
            rows_after=final_rows,
        )

        return df

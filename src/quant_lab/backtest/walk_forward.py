"""Walk-forward analysis: rolling retrain + backtest."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
import structlog

from quant_lab.data.datasets import TemporalSplit
from quant_lab.backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from quant_lab.backtest.metrics import compute_all_metrics

logger = structlog.get_logger(__name__)


class WindowType(str, Enum):
    EXPANDING = "expanding"
    ROLLING = "rolling"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""

    window_type: WindowType = WindowType.EXPANDING
    train_days: int = 756  # ~3 years
    val_days: int = 126  # ~6 months
    test_days: int = 126  # ~6 months per fold
    step_days: int = 126  # advance each fold
    min_train_days: int = 504  # ~2 years minimum


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_idx: int
    split: TemporalSplit
    test_start: str
    test_end: str
    backtest_result: BacktestResult
    train_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated results across all folds."""

    fold_results: list[FoldResult]
    aggregate_equity: pd.Series
    aggregate_returns: pd.Series
    aggregate_metrics: dict[str, float]
    per_fold_metrics: pd.DataFrame


# Type alias: factory receives (split, feature_df, feature_cols)
# and returns (model, signals_df_for_test_period).
ModelFactory = Callable[
    [TemporalSplit, pd.DataFrame, list[str]],
    tuple[Any, pd.DataFrame],
]


class WalkForwardSplitter:
    """Generates a sequence of TemporalSplit instances for walk-forward."""

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def generate_splits(
        self,
        all_dates: pd.DatetimeIndex,
    ) -> list[TemporalSplit]:
        """Generate walk-forward splits from available business dates.

        Args:
            all_dates: Sorted unique business dates available in the data.

        Returns:
            List of TemporalSplit instances for each fold.
        """
        dates = sorted(all_dates)
        n = len(dates)
        cfg = self.config
        splits = []

        train_end_idx = cfg.train_days - 1
        fold_idx = 0

        while True:
            val_end_idx = train_end_idx + cfg.val_days

            # Need room for test period
            if val_end_idx + 1 >= n:
                break

            test_end_idx = min(val_end_idx + cfg.test_days, n - 1)
            if test_end_idx <= val_end_idx:
                break

            # For rolling window, check effective train size
            if cfg.window_type == WindowType.ROLLING:
                train_start_idx = max(0, train_end_idx - cfg.train_days + 1)
            else:
                train_start_idx = 0

            actual_train_days = train_end_idx - train_start_idx + 1
            if actual_train_days < cfg.min_train_days:
                train_end_idx += cfg.step_days
                continue

            split = TemporalSplit(
                train_end=str(dates[train_end_idx].date()),
                val_end=str(dates[val_end_idx].date()),
            )
            splits.append(split)

            logger.info(
                "walk_forward_fold",
                fold=fold_idx,
                train_start=str(dates[train_start_idx].date()),
                train_end=split.train_end,
                val_end=split.val_end,
                test_end=str(dates[test_end_idx].date()),
                train_days=actual_train_days,
            )

            train_end_idx += cfg.step_days
            fold_idx += 1

        logger.info("walk_forward_splits_generated", n_folds=len(splits))
        return splits


class WalkForwardEngine:
    """Runs walk-forward analysis across multiple folds.

    The model_factory is responsible for:
    1. Creating and training a fresh model on the fold's train period
    2. Generating test-period signals
    3. Returning (trained_model, signals_df)

    This keeps WalkForwardEngine model-agnostic.
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        backtest_config: BacktestConfig | None = None,
    ):
        self.config = config
        self.backtest_config = backtest_config or BacktestConfig()
        self.splitter = WalkForwardSplitter(config)

    def run(
        self,
        feature_df: pd.DataFrame,
        feature_cols: list[str],
        prices_df: pd.DataFrame,
        model_factory: ModelFactory,
        regime_labels: pd.Series | None = None,
    ) -> WalkForwardResult:
        """Execute walk-forward analysis.

        Args:
            feature_df: Full feature DataFrame (all dates).
            feature_cols: List of feature column names.
            prices_df: Full prices DataFrame[date, ticker, adj_close].
            model_factory: Callable that trains model and returns signals.
            regime_labels: Optional regime labels for conditional sizing.

        Returns:
            WalkForwardResult with per-fold and aggregate metrics.
        """
        feature_df = feature_df.copy()
        feature_df["date"] = pd.to_datetime(feature_df["date"])
        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        all_dates = pd.DatetimeIndex(sorted(feature_df["date"].unique()))
        splits = self.splitter.generate_splits(all_dates)

        if not splits:
            raise ValueError(
                "No valid walk-forward splits. "
                "Check data length vs config window sizes."
            )

        fold_results = []
        engine = BacktestEngine(config=self.backtest_config)

        for fold_idx, split in enumerate(splits):
            logger.info("walk_forward_fold_start", fold=fold_idx)

            try:
                model, test_signals = model_factory(
                    split, feature_df, feature_cols
                )
            except Exception as e:
                logger.warning(
                    "fold_model_factory_failed", fold=fold_idx, error=str(e)
                )
                continue

            if test_signals is None or test_signals.empty:
                logger.warning("fold_skipped_no_signals", fold=fold_idx)
                continue

            # Get test-period prices
            test_dates = test_signals["date"].unique()
            test_prices = prices_df[
                prices_df["date"].isin(test_dates)
            ][["date", "ticker", "adj_close"]].copy()

            if test_prices.empty:
                logger.warning("fold_skipped_no_prices", fold=fold_idx)
                continue

            try:
                result = engine.run(
                    prices=test_prices,
                    signals=test_signals,
                    regime_labels=regime_labels,
                )
            except Exception as e:
                logger.warning(
                    "fold_backtest_failed", fold=fold_idx, error=str(e)
                )
                continue

            fold_result = FoldResult(
                fold_idx=fold_idx,
                split=split,
                test_start=str(pd.Timestamp(test_dates.min()).date()),
                test_end=str(pd.Timestamp(test_dates.max()).date()),
                backtest_result=result,
            )
            fold_results.append(fold_result)

            logger.info(
                "walk_forward_fold_complete",
                fold=fold_idx,
                sharpe=f"{result.metrics.get('sharpe', 0):.4f}",
                total_return=f"{result.metrics.get('total_return', 0):.4f}",
            )

        if not fold_results:
            raise ValueError("All walk-forward folds failed. Check data and model.")

        return self._aggregate_results(fold_results)

    def _aggregate_results(
        self, fold_results: list[FoldResult]
    ) -> WalkForwardResult:
        """Aggregate per-fold results into overall statistics."""
        # Concatenate per-fold returns (skip initial zero return per fold)
        all_returns = []
        for fr in fold_results:
            fold_returns = fr.backtest_result.returns.iloc[1:]
            all_returns.append(fold_returns)

        aggregate_returns = pd.concat(all_returns)
        aggregate_returns = aggregate_returns[
            ~aggregate_returns.index.duplicated(keep="first")
        ].sort_index()

        # Build aggregate equity curve
        initial = self.backtest_config.initial_capital
        cumulative = (1 + aggregate_returns).cumprod()
        equity_values = initial * cumulative

        # Prepend initial value
        first_date = aggregate_returns.index[0] - pd.Timedelta(days=1)
        aggregate_equity = pd.concat([
            pd.Series([initial], index=[first_date]),
            equity_values,
        ])
        aggregate_equity.name = "equity"

        aggregate_metrics = compute_all_metrics(
            equity_curve=aggregate_equity,
            returns=aggregate_returns,
            risk_free_annual=self.backtest_config.risk_free_rate,
        )

        # Per-fold metrics
        per_fold_records = []
        for fr in fold_results:
            row = {
                "fold": fr.fold_idx,
                "test_start": fr.test_start,
                "test_end": fr.test_end,
            }
            row.update(fr.backtest_result.metrics)
            per_fold_records.append(row)
        per_fold_df = pd.DataFrame(per_fold_records)

        logger.info(
            "walk_forward_complete",
            n_folds=len(fold_results),
            aggregate_sharpe=f"{aggregate_metrics.get('sharpe', 0):.4f}",
            aggregate_return=f"{aggregate_metrics.get('total_return', 0):.4f}",
        )

        return WalkForwardResult(
            fold_results=fold_results,
            aggregate_equity=aggregate_equity,
            aggregate_returns=aggregate_returns,
            aggregate_metrics=aggregate_metrics,
            per_fold_metrics=per_fold_df,
        )

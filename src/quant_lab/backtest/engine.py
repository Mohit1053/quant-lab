"""Backtest engine - day-by-day portfolio simulation with realistic execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

from quant_lab.backtest.execution import ExecutionModel
from quant_lab.backtest.lookahead_guard import assert_no_lookahead
from quant_lab.backtest.metrics import compute_all_metrics

logger = structlog.get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 1_000_000.0
    rebalance_frequency: int = 5  # days
    max_position_size: float = 0.20
    top_n: int = 5  # Number of top-signal assets to hold
    risk_free_rate: float = 0.05
    regime_size_map: dict[int, float] | None = None  # regime_id -> max_position_size


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    equity_curve: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    metrics: dict[str, float]
    trades: list[dict] = field(default_factory=list)


class BacktestEngine:
    """Realistic day-by-day backtest simulation."""

    def __init__(
        self,
        execution_model: ExecutionModel | None = None,
        config: BacktestConfig | None = None,
    ):
        self.execution = execution_model or ExecutionModel()
        self.config = config or BacktestConfig()

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        regime_labels: pd.Series | None = None,
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            prices: DataFrame with columns [date, ticker, adj_close].
            signals: DataFrame with columns [date, ticker, signal].
                     Higher signal = more bullish.
            regime_labels: Optional Series indexed by date with integer regime IDs.
                          Used with regime_size_map for conditional position sizing.
        """
        # Pivot to wide format
        price_wide = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
        signal_wide = signals.pivot(index="date", columns="ticker", values="signal").sort_index()

        tickers = sorted(set(price_wide.columns) & set(signal_wide.columns))
        price_wide = price_wide[tickers]
        signal_wide = signal_wide[tickers]

        # Compute daily returns
        returns_wide = price_wide.pct_change()

        # Simulation
        dates = price_wide.index.tolist()
        n_assets = len(tickers)

        # Initialize
        portfolio_value = self.config.initial_capital
        current_weights = np.zeros(n_assets)
        equity_history = []
        returns_history = []
        weights_history = []
        trades = []

        rebalance_counter = 0

        for i, date in enumerate(dates):
            if i == 0:
                equity_history.append(portfolio_value)
                returns_history.append(0.0)
                weights_history.append(current_weights.copy())
                continue

            # Daily return from current positions
            day_returns = returns_wide.loc[date].values
            nan_count = int(np.isnan(day_returns).sum())
            if nan_count > 0:
                logger.warning("nan_returns_filled", date=str(date), nan_count=nan_count)
            day_returns = np.nan_to_num(day_returns, nan=0.0)

            portfolio_return = np.dot(current_weights, day_returns)

            # Check if rebalance day
            rebalance_counter += 1
            if rebalance_counter >= self.config.rebalance_frequency:
                rebalance_counter = 0

                # Get signals with execution delay
                signal_date_idx = max(0, i - self.execution.execution_delay_bars)
                signal_date = dates[signal_date_idx]

                if signal_date in signal_wide.index:
                    # Guard: signal generated FOR current date, USING data from signal_date
                    assert_no_lookahead(
                        signal_date=pd.Timestamp(date),
                        data_date=pd.Timestamp(signal_date),
                        context="backtest_engine",
                    )
                    day_signals = signal_wide.loc[signal_date].values

                    # Get current regime for conditional sizing
                    current_regime = None
                    if regime_labels is not None:
                        sig_ts = pd.Timestamp(signal_date)
                        if sig_ts in regime_labels.index:
                            current_regime = int(regime_labels.loc[sig_ts])

                    target_weights = self._compute_target_weights(
                        day_signals, regime_id=current_regime,
                    )

                    # Compute turnover and costs
                    turnover = np.sum(np.abs(target_weights - current_weights))
                    trade_cost = self.execution.compute_trade_cost(turnover)

                    portfolio_return -= trade_cost

                    if turnover > 0.01:  # Only log meaningful trades
                        trades.append({
                            "date": date,
                            "turnover": float(turnover),
                            "cost": float(trade_cost),
                        })

                    current_weights = target_weights

            # Update portfolio value
            portfolio_value *= 1 + portfolio_return

            equity_history.append(portfolio_value)
            returns_history.append(portfolio_return)
            weights_history.append(current_weights.copy())

        equity_curve = pd.Series(equity_history, index=dates, name="equity")
        returns_series = pd.Series(returns_history, index=dates, name="returns")
        weights_df = pd.DataFrame(weights_history, index=dates, columns=tickers)

        # Exclude day-0 zero return from metrics computation
        metrics = compute_all_metrics(
            equity_curve=equity_curve,
            returns=returns_series.iloc[1:],  # Skip initial 0.0 return
            weights_history=weights_df,
            risk_free_annual=self.config.risk_free_rate,
        )

        logger.info("backtest_complete", **{k: f"{v:.4f}" for k, v in metrics.items()})

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns_series,
            weights_history=weights_df,
            metrics=metrics,
            trades=trades,
        )

    def _compute_target_weights(
        self,
        signals: np.ndarray,
        regime_id: int | None = None,
    ) -> np.ndarray:
        """Convert signals to portfolio weights.

        Strategy: equal-weight top N by signal, capped at max_position_size.
        When regime_size_map is configured, max position size varies by regime.
        """
        n = len(signals)
        weights = np.zeros(n)

        # Handle NaNs
        valid_mask = ~np.isnan(signals)
        if not valid_mask.any():
            return weights

        # Determine max position size (regime-conditional or default)
        if regime_id is not None and self.config.regime_size_map:
            max_pos = self.config.regime_size_map.get(
                regime_id, self.config.max_position_size,
            )
        else:
            max_pos = self.config.max_position_size

        # Rank by signal (higher = better)
        valid_signals = np.where(valid_mask, signals, -np.inf)
        top_n = min(self.config.top_n, valid_mask.sum())

        if top_n == 0:
            return weights

        top_indices = np.argsort(valid_signals)[-top_n:]

        # Equal weight, capped
        weight_per_asset = min(1.0 / top_n, max_pos)
        weights[top_indices] = weight_per_asset

        # Normalize to sum <= 1.0
        total = weights.sum()
        if total > 1.0:
            weights = weights / total

        return weights

"""Tests for backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.backtest.engine import BacktestEngine, BacktestConfig
from quant_lab.backtest.execution import ExecutionModel


@pytest.fixture
def simple_backtest_data():
    """Simple data for backtesting: 5 assets, 50 days."""
    dates = pd.bdate_range("2023-01-01", periods=50)
    tickers = ["A", "B", "C", "D", "E"]

    prices_records = []
    signals_records = []

    np.random.seed(42)
    for ticker in tickers:
        base_price = 100.0
        for i, date in enumerate(dates):
            price = base_price * (1 + np.random.normal(0.001, 0.01))
            base_price = price
            prices_records.append({
                "date": date,
                "ticker": ticker,
                "adj_close": price,
            })
            signals_records.append({
                "date": date,
                "ticker": ticker,
                "signal": np.random.normal(0, 1),
            })

    return pd.DataFrame(prices_records), pd.DataFrame(signals_records)


class TestBacktestEngine:
    def test_engine_runs(self, simple_backtest_data):
        prices, signals = simple_backtest_data
        engine = BacktestEngine()
        result = engine.run(prices, signals)
        assert len(result.equity_curve) > 0
        assert len(result.returns) > 0
        assert len(result.metrics) > 0

    def test_initial_capital_correct(self, simple_backtest_data):
        prices, signals = simple_backtest_data
        config = BacktestConfig(initial_capital=1_000_000)
        engine = BacktestEngine(config=config)
        result = engine.run(prices, signals)
        assert result.equity_curve.iloc[0] == 1_000_000

    def test_metrics_keys_present(self, simple_backtest_data):
        prices, signals = simple_backtest_data
        engine = BacktestEngine()
        result = engine.run(prices, signals)
        expected_keys = {"cagr", "sharpe", "sortino", "max_drawdown", "calmar"}
        assert expected_keys.issubset(set(result.metrics.keys()))

    def test_zero_cost_no_drag(self, simple_backtest_data):
        prices, signals = simple_backtest_data
        zero_cost = ExecutionModel(
            commission_bps=0, slippage_bps=0, spread_bps=0, execution_delay_bars=0
        )
        config = BacktestConfig(rebalance_frequency=1)
        engine = BacktestEngine(execution_model=zero_cost, config=config)
        result = engine.run(prices, signals)
        # Should complete without error
        assert result.equity_curve.iloc[-1] > 0

    def test_high_cost_reduces_returns(self, simple_backtest_data):
        prices, signals = simple_backtest_data
        config = BacktestConfig(rebalance_frequency=1)

        low_cost = ExecutionModel(commission_bps=0, slippage_bps=0, spread_bps=0)
        high_cost = ExecutionModel(commission_bps=100, slippage_bps=50, spread_bps=50)

        result_low = BacktestEngine(execution_model=low_cost, config=config).run(prices, signals)
        result_high = BacktestEngine(execution_model=high_cost, config=config).run(prices, signals)

        # Higher costs should reduce final equity
        assert result_high.equity_curve.iloc[-1] <= result_low.equity_curve.iloc[-1]

    def test_regime_conditional_sizing(self, simple_backtest_data):
        """Regime-conditional sizing uses regime_size_map when provided."""
        prices, signals = simple_backtest_data
        dates = sorted(prices["date"].unique())

        # Create regime labels: first half = regime 0, second half = regime 1
        mid = len(dates) // 2
        regime_labels = pd.Series(
            [0] * mid + [1] * (len(dates) - mid),
            index=dates,
        )

        # Regime 0 = aggressive (25%), regime 1 = defensive (5%)
        config = BacktestConfig(
            rebalance_frequency=1,
            max_position_size=0.20,
            top_n=5,
            regime_size_map={0: 0.25, 1: 0.05},
        )
        zero_cost = ExecutionModel(
            commission_bps=0, slippage_bps=0, spread_bps=0, execution_delay_bars=0
        )
        engine = BacktestEngine(execution_model=zero_cost, config=config)
        result = engine.run(prices, signals, regime_labels=regime_labels)

        # Second half weights should be smaller (5% cap vs 25%)
        weights_first_half = result.weights_history.iloc[1:mid]
        weights_second_half = result.weights_history.iloc[mid:]
        max_weight_first = weights_first_half.max().max()
        max_weight_second = weights_second_half.max().max()
        assert max_weight_second <= 0.05 + 1e-10
        assert max_weight_first <= 0.25 + 1e-10

    def test_regime_sizing_none_is_default(self, simple_backtest_data):
        """Without regime_labels, sizing uses default max_position_size."""
        prices, signals = simple_backtest_data
        config = BacktestConfig(
            rebalance_frequency=1,
            max_position_size=0.20,
            regime_size_map={0: 0.05},  # map exists but no labels passed
        )
        engine = BacktestEngine(config=config)
        result = engine.run(prices, signals)  # no regime_labels
        # Should use default 0.20 everywhere
        max_weight = result.weights_history.max().max()
        assert max_weight <= 0.20 + 1e-10

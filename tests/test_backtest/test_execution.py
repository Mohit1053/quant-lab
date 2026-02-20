"""Tests for execution model."""

from __future__ import annotations

import pytest

from quant_lab.backtest.execution import ExecutionModel


class TestExecutionModel:
    def test_default_costs(self):
        model = ExecutionModel()
        assert model.commission_bps == 10.0
        assert model.slippage_bps == 5.0
        assert model.spread_bps == 5.0

    def test_total_cost_bps(self):
        model = ExecutionModel(commission_bps=10, slippage_bps=5, spread_bps=5)
        # Total one-way = (10 + 5 + 5) / 2 = 10 bps
        assert model.total_cost_bps == 10.0

    def test_trade_cost_zero_turnover(self):
        model = ExecutionModel()
        cost = model.compute_trade_cost(turnover=0.0)
        assert cost == 0.0

    def test_trade_cost_full_turnover(self):
        model = ExecutionModel(commission_bps=10, slippage_bps=0, spread_bps=0)
        # Turnover of 2.0 (full rebalance) with 5 bps one-way cost
        cost = model.compute_trade_cost(turnover=2.0)
        # cost = 2.0 * (10/2) / 10000 = 2.0 * 5 / 10000 = 0.001
        assert abs(cost - 0.001) < 1e-10

    def test_custom_costs(self):
        model = ExecutionModel(commission_bps=20, slippage_bps=10, spread_bps=10)
        assert model.total_cost_bps == 20.0

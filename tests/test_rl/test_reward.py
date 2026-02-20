"""Tests for reward function."""

from __future__ import annotations

import pytest

from quant_lab.rl.environments.reward import RewardFunction, RewardConfig


class TestRewardFunction:
    def test_positive_return_positive_reward(self):
        rf = RewardFunction(RewardConfig(lambda_mdd=0, lambda_turnover=0,
                                          commission_bps=0, slippage_bps=0, spread_bps=0))
        reward, info = rf.compute(portfolio_return=0.01, turnover=0.0)
        assert reward > 0

    def test_negative_return_negative_reward(self):
        rf = RewardFunction(RewardConfig(lambda_mdd=0, lambda_turnover=0,
                                          commission_bps=0, slippage_bps=0, spread_bps=0))
        reward, info = rf.compute(portfolio_return=-0.01, turnover=0.0)
        assert reward < 0

    def test_trading_cost_reduces_reward(self):
        rf_no_cost = RewardFunction(RewardConfig(
            lambda_mdd=0, lambda_turnover=0,
            commission_bps=0, slippage_bps=0, spread_bps=0
        ))
        rf_with_cost = RewardFunction(RewardConfig(
            lambda_mdd=0, lambda_turnover=0,
            commission_bps=10, slippage_bps=5, spread_bps=5
        ))

        r_no_cost, _ = rf_no_cost.compute(0.01, turnover=0.5)
        r_with_cost, _ = rf_with_cost.compute(0.01, turnover=0.5)
        assert r_with_cost < r_no_cost

    def test_drawdown_penalty(self):
        rf = RewardFunction(RewardConfig(
            lambda_mdd=1.0, lambda_turnover=0,
            commission_bps=0, slippage_bps=0, spread_bps=0
        ))
        # First step: positive
        rf.compute(portfolio_return=0.05, turnover=0.0)
        # Second step: negative (creates drawdown)
        reward, info = rf.compute(portfolio_return=-0.10, turnover=0.0)
        assert info["drawdown"] > 0
        assert info["drawdown_penalty"] > 0

    def test_turnover_penalty(self):
        rf = RewardFunction(RewardConfig(
            lambda_mdd=0, lambda_turnover=0.1,
            commission_bps=0, slippage_bps=0, spread_bps=0
        ))
        r_no_turn, _ = rf.compute(0.01, turnover=0.0)
        rf.reset()
        r_high_turn, _ = rf.compute(0.01, turnover=1.0)
        assert r_high_turn < r_no_turn

    def test_reset(self):
        rf = RewardFunction()
        rf.compute(0.05, 0.0)
        rf.reset(initial_value=100.0)
        assert rf.peak_value == 100.0
        assert rf.current_value == 100.0

    def test_info_dict_keys(self):
        rf = RewardFunction()
        _, info = rf.compute(0.01, 0.1)
        assert "portfolio_return" in info
        assert "drawdown" in info
        assert "trading_cost" in info
        assert "turnover" in info
        assert "portfolio_value" in info

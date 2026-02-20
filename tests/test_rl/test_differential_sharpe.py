"""Tests for differential Sharpe ratio reward."""

from __future__ import annotations

import numpy as np
import pytest

from quant_lab.rl.environments.differential_sharpe import (
    DifferentialSharpeReward,
    DifferentialSharpeConfig,
)


class TestDifferentialSharpeReward:
    def test_first_step_reward_zero(self):
        reward_fn = DifferentialSharpeReward()
        reward, info = reward_fn.compute(portfolio_return=0.01, turnover=0.0)
        # First step diff_sharpe is 0 (no prior EMA)
        assert info["diff_sharpe"] == 0.0

    def test_diff_sharpe_nonzero_after_warmup(self):
        reward_fn = DifferentialSharpeReward(DifferentialSharpeConfig(eta=0.5))
        # Need varied returns to build up variance in B - A^2
        reward_fn.compute(0.01, 0.0)
        reward_fn.compute(-0.005, 0.0)  # Create variance in EMA
        _, info = reward_fn.compute(0.02, 0.0)
        # With variance established, a return surprise produces nonzero diff_sharpe
        assert info["diff_sharpe"] != 0.0

    def test_turnover_penalty_applied(self):
        config = DifferentialSharpeConfig(lambda_turnover=0.1)
        reward_fn = DifferentialSharpeReward(config)
        reward_fn.compute(0.01, 0.0)

        reward_no_turn, _ = reward_fn.compute(0.01, 0.0)
        reward_fn.reset()
        reward_fn.compute(0.01, 0.0)
        reward_with_turn, info = reward_fn.compute(0.01, 0.5)

        assert info["turnover_penalty"] > 0
        assert info["turnover"] == 0.5

    def test_trading_cost_computed(self):
        config = DifferentialSharpeConfig(
            commission_bps=10, slippage_bps=5, spread_bps=5
        )
        reward_fn = DifferentialSharpeReward(config)
        reward_fn.compute(0.01, 0.0)
        _, info = reward_fn.compute(0.01, 1.0)
        expected_cost = 1.0 * (10 + 5 + 5) / 10000
        assert abs(info["trading_cost"] - expected_cost) < 1e-8

    def test_reset_clears_state(self):
        reward_fn = DifferentialSharpeReward()
        reward_fn.compute(0.01, 0.0)
        reward_fn.compute(0.02, 0.0)

        reward_fn.reset()
        assert reward_fn._A == 0.0
        assert reward_fn._B == 0.0

    def test_ema_updates(self):
        config = DifferentialSharpeConfig(eta=0.5)
        reward_fn = DifferentialSharpeReward(config)
        reward_fn.compute(0.01, 0.0)
        assert reward_fn._A == 0.01

        reward_fn.compute(0.03, 0.0)
        # EMA: 0.01 + 0.5 * (0.03 - 0.01) = 0.02
        assert abs(reward_fn._A - 0.02) < 1e-8

    def test_sequence_of_rewards(self):
        reward_fn = DifferentialSharpeReward(DifferentialSharpeConfig(eta=0.01))
        rewards = []
        for r in [0.01, 0.02, -0.01, 0.005, 0.015]:
            reward, info = reward_fn.compute(r, 0.0)
            rewards.append(reward)
        assert len(rewards) == 5
        # All rewards should be finite
        assert all(np.isfinite(r) for r in rewards)

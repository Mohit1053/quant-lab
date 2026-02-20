"""Reward functions for portfolio RL environments.

Reward = Sharpe-like return - lambda_mdd * drawdown_penalty - turnover_cost
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:
    """Reward function configuration."""

    lambda_mdd: float = 0.5
    lambda_turnover: float = 0.01
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    spread_bps: float = 5.0
    risk_free_rate: float = 0.0


class RewardFunction:
    """Compute step-wise reward for portfolio allocation.

    reward = portfolio_return - lambda_mdd * drawdown_term - trading_costs
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.peak_value = 1.0
        self.current_value = 1.0

    def reset(self, initial_value: float = 1.0) -> None:
        self.peak_value = initial_value
        self.current_value = initial_value

    def compute(
        self,
        portfolio_return: float,
        turnover: float,
    ) -> tuple[float, dict[str, float]]:
        """Compute reward for a single step.

        Args:
            portfolio_return: Single-step portfolio return (e.g., 0.01 for 1%).
            turnover: Sum of absolute weight changes (0 to 2).

        Returns:
            (reward, info_dict) with breakdown of reward components.
        """
        # Update portfolio value
        self.current_value *= (1.0 + portfolio_return)
        self.peak_value = max(self.peak_value, self.current_value)

        # Drawdown penalty
        drawdown = (self.peak_value - self.current_value) / self.peak_value
        drawdown_penalty = self.config.lambda_mdd * drawdown

        # Trading cost (realistic cost model: commission + slippage + spread)
        total_cost_bps = (
            self.config.commission_bps
            + self.config.slippage_bps
            + self.config.spread_bps
        )
        trading_cost = turnover * total_cost_bps / 10000.0

        # Turnover penalty (RL regularization, separate from realistic trading costs)
        turnover_penalty = self.config.lambda_turnover * turnover

        # Total reward
        reward = portfolio_return - drawdown_penalty - trading_cost - turnover_penalty

        info = {
            "portfolio_return": portfolio_return,
            "drawdown": drawdown,
            "drawdown_penalty": drawdown_penalty,
            "trading_cost": trading_cost,
            "turnover_penalty": turnover_penalty,
            "turnover": turnover,
            "portfolio_value": self.current_value,
        }

        return reward, info

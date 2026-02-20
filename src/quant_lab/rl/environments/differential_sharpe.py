"""Differential Sharpe ratio reward for RL portfolio optimization.

The differential Sharpe ratio provides the incremental change in the
Sharpe ratio at each step, offering more stable training signals
compared to raw return-based rewards.

Reference: Moody & Saffell (2001) "Learning to Trade via Direct Reinforcement"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DifferentialSharpeConfig:
    """Configuration for differential Sharpe reward."""

    eta: float = 0.01  # Exponential moving average decay rate
    annualization_factor: float = 252.0  # Trading days per year
    lambda_turnover: float = 0.01
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    spread_bps: float = 5.0


class DifferentialSharpeReward:
    """Compute differential Sharpe ratio as step-wise RL reward.

    Maintains exponential moving averages of the first and second moments
    of returns to compute the instantaneous change in Sharpe ratio at
    each step. This gives the agent a gradient-like signal toward
    maximizing the overall Sharpe ratio.

    The reward at step t is:
        dS_t = (B_{t-1} * delta_A_t - 0.5 * A_{t-1} * delta_B_t) / (B_{t-1} - A_{t-1}^2)^{3/2}

    Where:
        A_t = EMA of returns (first moment)
        B_t = EMA of squared returns (second moment)
        delta_A_t = r_t - A_{t-1}
        delta_B_t = r_t^2 - B_{t-1}
    """

    def __init__(self, config: DifferentialSharpeConfig | None = None):
        self.config = config or DifferentialSharpeConfig()
        self._A = 0.0  # EMA of returns
        self._B = 0.0  # EMA of squared returns
        self._initialized = False

    def reset(self, **kwargs) -> None:
        """Reset the reward state."""
        self._A = 0.0
        self._B = 0.0
        self._initialized = False

    def compute(
        self,
        portfolio_return: float,
        turnover: float,
    ) -> tuple[float, dict[str, float]]:
        """Compute differential Sharpe reward for a single step.

        Args:
            portfolio_return: Single-step portfolio return.
            turnover: Sum of absolute weight changes.

        Returns:
            (reward, info_dict) with component breakdown.
        """
        eta = self.config.eta
        r = portfolio_return

        if not self._initialized:
            self._A = r
            self._B = r ** 2
            self._initialized = True
            diff_sharpe = 0.0
        else:
            delta_A = r - self._A
            delta_B = r ** 2 - self._B

            denominator = self._B - self._A ** 2
            if denominator > 1e-8:
                diff_sharpe = (
                    self._B * delta_A - 0.5 * self._A * delta_B
                ) / (denominator ** 1.5)
            else:
                diff_sharpe = 0.0

            # Update exponential moving averages
            self._A = self._A + eta * delta_A
            self._B = self._B + eta * delta_B

        # Trading cost penalty
        total_cost_bps = (
            self.config.commission_bps
            + self.config.slippage_bps
            + self.config.spread_bps
        )
        trading_cost = turnover * total_cost_bps / 10000.0
        turnover_penalty = self.config.lambda_turnover * turnover

        reward = diff_sharpe - trading_cost - turnover_penalty

        info = {
            "portfolio_return": portfolio_return,
            "diff_sharpe": diff_sharpe,
            "trading_cost": trading_cost,
            "turnover_penalty": turnover_penalty,
            "turnover": turnover,
            "ema_return": self._A,
            "ema_return_sq": self._B,
        }

        return reward, info

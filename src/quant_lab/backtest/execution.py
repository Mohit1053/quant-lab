"""Transaction cost and execution models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExecutionModel:
    """Models real-world execution costs."""

    commission_bps: float = 10.0  # Round-trip commission in basis points
    slippage_bps: float = 5.0  # Market impact slippage
    spread_bps: float = 5.0  # Bid-ask spread cost
    execution_delay_bars: int = 1  # Bars delay between signal and execution

    @property
    def total_cost_bps(self) -> float:
        """Total one-way transaction cost in basis points."""
        return (self.commission_bps + self.slippage_bps + self.spread_bps) / 2.0

    def compute_trade_cost(self, turnover: float) -> float:
        """Compute cost as a fraction of portfolio value for a given turnover.

        Args:
            turnover: Sum of absolute weight changes (0 to 2).

        Returns:
            Cost as a decimal fraction (e.g., 0.001 = 10 bps).
        """
        return turnover * self.total_cost_bps / 10000.0

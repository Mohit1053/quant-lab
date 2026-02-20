"""Action space utilities for portfolio environments.

Handles weight normalization, constraints, and cash allocation.
"""

from __future__ import annotations

import numpy as np


class ActionProcessor:
    """Process raw actions into valid portfolio weights.

    Handles:
    - Softmax normalization to ensure weights sum to <= 1
    - Per-asset weight constraints (min/max)
    - Cash allocation (1 - sum of weights)
    - Long-only constraints (no shorting)

    Args:
        num_assets: Number of tradeable assets.
        min_weight: Minimum weight per asset (default 0.0).
        max_weight: Maximum weight per asset (default 0.2).
        cash_weight: Whether to include cash as implicit residual.
        max_leverage: Maximum total absolute exposure.
    """

    def __init__(
        self,
        num_assets: int,
        min_weight: float = 0.0,
        max_weight: float = 0.20,
        cash_weight: bool = True,
        max_leverage: float = 1.0,
    ):
        self.num_assets = num_assets
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.cash_weight = cash_weight
        self.max_leverage = max_leverage

    def process(self, raw_action: np.ndarray) -> np.ndarray:
        """Convert raw action to valid portfolio weights.

        Args:
            raw_action: (num_assets,) raw action from agent ([0, 1]).

        Returns:
            (num_assets,) valid portfolio weights.
        """
        # Clip to valid range
        weights = np.clip(raw_action, 0.0, 1.0)

        # Apply softmax-like normalization
        weights = self._normalize(weights)

        # Iterative clipping: enforce both per-asset max and total leverage
        for _ in range(5):
            weights = np.clip(weights, self.min_weight, self.max_weight)
            total = np.sum(weights)
            if total <= self.max_leverage + 1e-9:
                break
            weights = weights * (self.max_leverage / total)

        return weights

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        """Softmax normalization to ensure weights are valid."""
        total = np.sum(weights)
        if total > 0:
            return weights / total * self.max_leverage
        return np.ones(self.num_assets) / self.num_assets * self.max_leverage

    def compute_turnover(
        self, old_weights: np.ndarray, new_weights: np.ndarray
    ) -> float:
        """Compute total turnover between two weight vectors.

        Returns:
            Sum of absolute weight changes (0 to 2 * max_leverage).
        """
        return float(np.sum(np.abs(new_weights - old_weights)))

    def get_cash_weight(self, weights: np.ndarray) -> float:
        """Compute implied cash position."""
        return max(0.0, 1.0 - np.sum(weights))

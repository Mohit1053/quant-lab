"""Gymnasium-compatible portfolio allocation environment.

Observation: [embeddings/features, forecasts, current_weights, regime_probs]
Action: target portfolio weights (continuous)
Reward: risk-adjusted return minus costs
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quant_lab.rl.environments.reward import RewardFunction, RewardConfig
from quant_lab.rl.environments.action_space import ActionProcessor


@dataclass
class PortfolioEnvConfig:
    """Portfolio environment configuration."""

    initial_cash: float = 1_000_000.0
    max_leverage: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 0.20
    cash_weight: bool = True
    rebalance_frequency: int = 5  # Rebalance every N steps
    use_regime_conditioning: bool = False  # Include regime features in obs


class PortfolioEnv(gym.Env):
    """Portfolio allocation environment for RL agents.

    At each step, the agent observes:
    - Feature vectors (embeddings, forecasts) for each asset
    - Current portfolio weights
    - (Optional) Regime probability vector for regime-conditioned policies

    The agent outputs target portfolio weights. The environment
    computes portfolio returns using the actual asset returns and
    calculates the reward including trading costs and risk penalties.

    Args:
        features: (num_steps, num_assets, feature_dim) feature array.
        returns: (num_steps, num_assets) return array.
        config: Environment configuration.
        reward_config: Reward function configuration.
        regime_probs: Optional (num_steps, n_regimes) regime probability array.
            Only used when config.use_regime_conditioning is True.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        config: PortfolioEnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        regime_probs: np.ndarray | None = None,
    ):
        super().__init__()

        self.config = config or PortfolioEnvConfig()
        self.features = features.astype(np.float32)
        self.returns = returns.astype(np.float32)

        self.num_steps, self.num_assets, self.feature_dim = features.shape
        assert returns.shape == (self.num_steps, self.num_assets)

        # Regime conditioning: (num_steps, n_regimes) probability vectors
        self.regime_probs = None
        self.n_regime_features = 0
        if regime_probs is not None and self.config.use_regime_conditioning:
            self.regime_probs = regime_probs.astype(np.float32)
            assert regime_probs.shape[0] == self.num_steps
            assert regime_probs.shape[1] > 0, "Regime probabilities must have at least 1 regime"
            self.n_regime_features = regime_probs.shape[1]

        # Action: target weights for each asset (long-only: [0, 1])
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.num_assets,),
            dtype=np.float32,
        )

        # Observation: features + current weights + optional regime probs
        obs_dim = self.num_assets * self.feature_dim + self.num_assets + self.n_regime_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_processor = ActionProcessor(
            num_assets=self.num_assets,
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
            cash_weight=self.config.cash_weight,
            max_leverage=self.config.max_leverage,
        )

        self.reward_fn = RewardFunction(reward_config)

        # State
        self._current_step = 0
        self._current_weights = np.zeros(self.num_assets, dtype=np.float32)
        self._portfolio_value = self.config.initial_cash

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        self._current_step = 0
        self._current_weights = np.zeros(self.num_assets, dtype=np.float32)
        self._portfolio_value = self.config.initial_cash
        self.reward_fn.reset(initial_value=self.config.initial_cash)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        """Execute one step.

        Args:
            action: (num_assets,) raw action from agent.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # Process action to valid weights
        new_weights = self.action_processor.process(action)
        turnover = self.action_processor.compute_turnover(
            self._current_weights, new_weights
        )

        # Advance step first, then use next period's returns (no lookahead)
        self._current_step += 1

        # Compute portfolio return using NEXT period's returns
        if self._current_step < self.num_steps:
            step_returns = self.returns[self._current_step]  # (num_assets,)
        else:
            step_returns = np.zeros(self.num_assets, dtype=np.float32)
        portfolio_return = float(np.sum(new_weights * step_returns))

        # Add cash return (0 for simplicity)
        cash_weight = self.action_processor.get_cash_weight(new_weights)

        # Update portfolio value
        self._portfolio_value *= (1.0 + portfolio_return)

        # Compute reward
        reward, reward_info = self.reward_fn.compute(portfolio_return, turnover)

        # Update state
        self._current_weights = new_weights.astype(np.float32)

        # Check termination
        terminated = False
        truncated = self._current_step >= self.num_steps - 1

        info = {
            **reward_info,
            "step": self._current_step,
            "portfolio_value": self._portfolio_value,
            "cash_weight": cash_weight,
            "weights": new_weights.copy(),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        step = min(self._current_step, self.num_steps - 1)
        features_flat = self.features[step].flatten()  # (num_assets * feature_dim,)
        parts = [features_flat, self._current_weights]
        if self.regime_probs is not None:
            parts.append(self.regime_probs[step])
        obs = np.concatenate(parts)
        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "step": self._current_step,
            "portfolio_value": self._portfolio_value,
            "weights": self._current_weights.copy(),
        }

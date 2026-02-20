"""Tests for regime conditioning in RL environment."""

from __future__ import annotations

import numpy as np
import pytest

from quant_lab.rl.environments.portfolio_env import PortfolioEnv, PortfolioEnvConfig


def _make_env_data(n_steps=50, n_assets=3, feature_dim=4, n_regimes=3):
    np.random.seed(42)
    features = np.random.randn(n_steps, n_assets, feature_dim).astype(np.float32)
    returns = np.random.randn(n_steps, n_assets).astype(np.float32) * 0.01
    # Create regime probabilities (soft assignments)
    regime_probs = np.random.dirichlet(np.ones(n_regimes), size=n_steps).astype(np.float32)
    return features, returns, regime_probs


class TestRegimeConditionedEnv:
    def test_obs_includes_regime_probs(self):
        features, returns, regime_probs = _make_env_data()
        config = PortfolioEnvConfig(use_regime_conditioning=True)
        env = PortfolioEnv(features, returns, config=config, regime_probs=regime_probs)

        obs, info = env.reset()
        # obs_dim = n_assets * feature_dim + n_assets + n_regimes
        expected_dim = 3 * 4 + 3 + 3
        assert obs.shape == (expected_dim,)

    def test_obs_without_regime_conditioning(self):
        features, returns, regime_probs = _make_env_data()
        config = PortfolioEnvConfig(use_regime_conditioning=False)
        env = PortfolioEnv(features, returns, config=config, regime_probs=regime_probs)

        obs, info = env.reset()
        # Without regime: obs_dim = n_assets * feature_dim + n_assets
        expected_dim = 3 * 4 + 3
        assert obs.shape == (expected_dim,)

    def test_regime_probs_none_works(self):
        features, returns, _ = _make_env_data()
        config = PortfolioEnvConfig(use_regime_conditioning=True)
        env = PortfolioEnv(features, returns, config=config, regime_probs=None)

        obs, info = env.reset()
        expected_dim = 3 * 4 + 3  # No regime features added
        assert obs.shape == (expected_dim,)

    def test_step_with_regime(self):
        features, returns, regime_probs = _make_env_data()
        config = PortfolioEnvConfig(use_regime_conditioning=True)
        env = PortfolioEnv(features, returns, config=config, regime_probs=regime_probs)

        obs, info = env.reset()
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2.shape == obs.shape

    def test_full_episode_with_regime(self):
        features, returns, regime_probs = _make_env_data(n_steps=20)
        config = PortfolioEnvConfig(use_regime_conditioning=True)
        env = PortfolioEnv(features, returns, config=config, regime_probs=regime_probs)

        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        # n_steps=20 gives 19 tradeable periods (observe at t, earn return at t+1)
        assert steps == 19

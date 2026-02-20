"""Tests for portfolio environment."""

from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym

from quant_lab.rl.environments.portfolio_env import PortfolioEnv, PortfolioEnvConfig
from quant_lab.rl.environments.reward import RewardConfig


def _make_dummy_env(num_steps=50, num_assets=5, feature_dim=4):
    """Create a simple test environment."""
    features = np.random.randn(num_steps, num_assets, feature_dim).astype(np.float32)
    returns = np.random.randn(num_steps, num_assets).astype(np.float32) * 0.01
    config = PortfolioEnvConfig(
        initial_cash=100_000,
        max_weight=0.30,
    )
    reward_config = RewardConfig(
        lambda_mdd=0.0,
        lambda_turnover=0.0,
        commission_bps=0,
        slippage_bps=0,
        spread_bps=0,
    )
    return PortfolioEnv(features, returns, config, reward_config)


class TestPortfolioEnv:
    def test_reset_returns_obs_and_info(self):
        env = _make_dummy_env()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_observation_shape(self):
        env = _make_dummy_env(num_assets=5, feature_dim=4)
        obs, _ = env.reset()
        expected_dim = 5 * 4 + 5  # features + weights
        assert obs.shape == (expected_dim,)

    def test_step_returns_correct_format(self):
        env = _make_dummy_env()
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_terminates(self):
        env = _make_dummy_env(num_steps=10)
        env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        # num_steps=10 gives 9 tradeable periods (observe at t, earn return at t+1)
        assert steps == 9

    def test_info_contains_portfolio_value(self):
        env = _make_dummy_env()
        env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "portfolio_value" in info
        assert "weights" in info
        assert info["portfolio_value"] > 0

    def test_gymnasium_check_env(self):
        """Verify env passes Gymnasium's check_env."""
        from gymnasium.utils.env_checker import check_env
        env = _make_dummy_env()
        # check_env will raise if something is wrong
        check_env(env, skip_render_check=True)

    def test_action_space_shape(self):
        env = _make_dummy_env(num_assets=5)
        assert env.action_space.shape == (5,)

    def test_obs_space_shape(self):
        env = _make_dummy_env(num_assets=5, feature_dim=4)
        expected_dim = 5 * 4 + 5
        assert env.observation_space.shape == (expected_dim,)

    def test_multiple_episodes(self):
        env = _make_dummy_env(num_steps=10)
        for _ in range(3):
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

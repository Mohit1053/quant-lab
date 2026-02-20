"""Tests for RL agents (PPO, SAC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_lab.rl.environments.portfolio_env import PortfolioEnv, PortfolioEnvConfig
from quant_lab.rl.environments.reward import RewardConfig
from quant_lab.rl.agents.ppo_agent import PPOAgent, PPOConfig
from quant_lab.rl.agents.sac_agent import SACAgent, SACConfig
from quant_lab.rl.training import evaluate_agent


def _make_env(num_steps=50, num_assets=3, feature_dim=4):
    features = np.random.randn(num_steps, num_assets, feature_dim).astype(np.float32)
    returns = np.random.randn(num_steps, num_assets).astype(np.float32) * 0.01
    return PortfolioEnv(features, returns)


class TestPPOAgent:
    def test_create_agent(self):
        env = _make_env()
        config = PPOConfig(n_steps=50)  # Small for testing
        agent = PPOAgent(env, config=config, device="cpu")
        assert agent.model is not None

    def test_predict(self):
        env = _make_env()
        config = PPOConfig(n_steps=50)
        agent = PPOAgent(env, config=config, device="cpu")
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert action.shape == (3,)

    def test_short_training(self):
        env = _make_env()
        config = PPOConfig(n_steps=50, batch_size=25)
        agent = PPOAgent(env, config=config, device="cpu")
        agent.train(total_timesteps=100)  # Very short training

    def test_save_and_load(self, tmp_path):
        env = _make_env()
        config = PPOConfig(n_steps=50)
        agent = PPOAgent(env, config=config, device="cpu")
        save_path = tmp_path / "ppo_test"
        agent.save(save_path)
        agent.load(save_path)
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert action.shape == (3,)


class TestSACAgent:
    def test_create_agent(self):
        env = _make_env()
        config = SACConfig(learning_starts=10, buffer_size=1000, batch_size=32)
        agent = SACAgent(env, config=config, device="cpu")
        assert agent.model is not None

    def test_predict(self):
        env = _make_env()
        config = SACConfig(learning_starts=10, buffer_size=1000, batch_size=32)
        agent = SACAgent(env, config=config, device="cpu")
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert action.shape == (3,)

    def test_short_training(self):
        env = _make_env()
        config = SACConfig(learning_starts=10, buffer_size=1000, batch_size=32)
        agent = SACAgent(env, config=config, device="cpu")
        agent.train(total_timesteps=100)

    def test_save_and_load(self, tmp_path):
        env = _make_env()
        config = SACConfig(learning_starts=10, buffer_size=1000, batch_size=32)
        agent = SACAgent(env, config=config, device="cpu")
        save_path = tmp_path / "sac_test"
        agent.save(save_path)
        agent.load(save_path)
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert action.shape == (3,)


class TestEvaluateAgent:
    def test_evaluate_returns_metrics(self):
        env = _make_env()
        config = PPOConfig(n_steps=50)
        agent = PPOAgent(env, config=config, device="cpu")
        metrics = evaluate_agent(agent, env, n_episodes=2)
        assert "mean_reward" in metrics
        assert "mean_final_value" in metrics

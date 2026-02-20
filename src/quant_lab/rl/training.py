"""RL training orchestration for portfolio allocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from quant_lab.rl.environments.portfolio_env import PortfolioEnv, PortfolioEnvConfig
from quant_lab.rl.environments.reward import RewardConfig
from quant_lab.rl.agents.ppo_agent import PPOAgent, PPOConfig
from quant_lab.rl.agents.sac_agent import SACAgent, SACConfig

logger = structlog.get_logger(__name__)


@dataclass
class RLTrainingConfig:
    """RL training configuration."""

    algorithm: str = "ppo"
    total_timesteps: int = 100_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    checkpoint_dir: str = "outputs/models/rl"


def create_env(
    features: np.ndarray,
    returns: np.ndarray,
    env_config: PortfolioEnvConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> PortfolioEnv:
    """Create a portfolio environment."""
    return PortfolioEnv(
        features=features,
        returns=returns,
        config=env_config,
        reward_config=reward_config,
    )


def create_agent(
    env: PortfolioEnv,
    algorithm: str = "ppo",
    ppo_config: PPOConfig | None = None,
    sac_config: SACConfig | None = None,
    device: str = "auto",
) -> PPOAgent | SACAgent:
    """Create an RL agent.

    Args:
        env: Portfolio environment.
        algorithm: "ppo" or "sac".
        ppo_config: PPO hyperparameters (if algorithm="ppo").
        sac_config: SAC hyperparameters (if algorithm="sac").
        device: Device for training.

    Returns:
        Configured agent.
    """
    if algorithm == "ppo":
        return PPOAgent(env, config=ppo_config, device=device)
    elif algorithm == "sac":
        return SACAgent(env, config=sac_config, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'ppo' or 'sac'.")


def evaluate_agent(
    agent: PPOAgent | SACAgent,
    env: PortfolioEnv,
    n_episodes: int = 5,
) -> dict[str, float]:
    """Evaluate an agent over multiple episodes.

    Returns:
        Dict with mean/std reward and portfolio metrics.
    """
    episode_rewards = []
    final_values = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)
        final_values.append(info["portfolio_value"])

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_final_value": float(np.mean(final_values)),
        "std_final_value": float(np.std(final_values)),
    }


def train_rl(
    train_features: np.ndarray,
    train_returns: np.ndarray,
    val_features: np.ndarray | None = None,
    val_returns: np.ndarray | None = None,
    config: RLTrainingConfig | None = None,
    env_config: PortfolioEnvConfig | None = None,
    reward_config: RewardConfig | None = None,
    device: str = "auto",
) -> dict:
    """End-to-end RL training.

    Args:
        train_features: (T, N, D) training features.
        train_returns: (T, N) training returns.
        val_features: Optional validation features.
        val_returns: Optional validation returns.
        config: Training configuration.
        env_config: Environment configuration.
        reward_config: Reward configuration.
        device: Training device.

    Returns:
        Dict with agent, training history, and eval metrics.
    """
    config = config or RLTrainingConfig()

    # Create environment
    train_env = create_env(train_features, train_returns, env_config, reward_config)

    # Create agent
    agent = create_agent(train_env, algorithm=config.algorithm, device=device)

    logger.info(
        "rl_training_start",
        algorithm=config.algorithm,
        total_timesteps=config.total_timesteps,
        num_assets=train_env.num_assets,
        num_steps=train_env.num_steps,
    )

    # Train
    agent.train(total_timesteps=config.total_timesteps)

    # Evaluate on training env
    train_metrics = evaluate_agent(agent, train_env, n_episodes=config.n_eval_episodes)
    logger.info("rl_train_eval", **train_metrics)

    result = {
        "agent": agent,
        "train_metrics": train_metrics,
    }

    # Evaluate on validation env if provided
    if val_features is not None and val_returns is not None:
        val_env = create_env(val_features, val_returns, env_config, reward_config)
        val_metrics = evaluate_agent(agent, val_env, n_episodes=config.n_eval_episodes)
        logger.info("rl_val_eval", **val_metrics)
        result["val_metrics"] = val_metrics

    # Save
    save_dir = Path(config.checkpoint_dir)
    agent.save(save_dir / f"{config.algorithm}_agent")

    logger.info("rl_training_complete")
    return result

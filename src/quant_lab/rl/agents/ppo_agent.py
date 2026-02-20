"""PPO agent wrapper using Stable-Baselines3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from stable_baselines3 import PPO

logger = structlog.get_logger(__name__)


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy: str = "MlpPolicy"


class PPOAgent:
    """PPO agent wrapper for portfolio allocation.

    Wraps SB3 PPO with config-driven setup, save/load, and evaluation.
    """

    def __init__(self, env, config: PPOConfig | None = None, device: str = "auto"):
        self.config = config or PPOConfig()
        self.env = env

        self.model = PPO(
            policy=self.config.policy,
            env=env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            device=device,
            verbose=0,
        )

        logger.info(
            "ppo_agent_created",
            policy=self.config.policy,
            lr=self.config.learning_rate,
        )

    def train(self, total_timesteps: int, **kwargs) -> None:
        """Train the agent."""
        logger.info("ppo_training_start", total_timesteps=total_timesteps)
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        logger.info("ppo_training_complete")

    def predict(self, obs, deterministic: bool = True):
        """Get action for observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("ppo_saved", path=str(path))

    def load(self, path: str | Path) -> None:
        self.model = PPO.load(str(path), env=self.env)
        logger.info("ppo_loaded", path=str(path))

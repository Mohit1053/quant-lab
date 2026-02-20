"""SAC agent wrapper using Stable-Baselines3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from stable_baselines3 import SAC

logger = structlog.get_logger(__name__)


@dataclass
class SACConfig:
    """SAC hyperparameters."""

    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    tau: float = 0.005
    gamma: float = 0.99
    ent_coef: str = "auto"
    policy: str = "MlpPolicy"


class SACAgent:
    """SAC agent wrapper for portfolio allocation.

    Wraps SB3 SAC with config-driven setup, save/load, and evaluation.
    SAC is well-suited for continuous action spaces like portfolio weights.
    """

    def __init__(self, env, config: SACConfig | None = None, device: str = "auto"):
        self.config = config or SACConfig()
        self.env = env

        self.model = SAC(
            policy=self.config.policy,
            env=env,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            tau=self.config.tau,
            gamma=self.config.gamma,
            ent_coef=self.config.ent_coef,
            device=device,
            verbose=0,
        )

        logger.info(
            "sac_agent_created",
            policy=self.config.policy,
            lr=self.config.learning_rate,
        )

    def train(self, total_timesteps: int, **kwargs) -> None:
        """Train the agent."""
        logger.info("sac_training_start", total_timesteps=total_timesteps)
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        logger.info("sac_training_complete")

    def predict(self, obs, deterministic: bool = True):
        """Get action for observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("sac_saved", path=str(path))

    def load(self, path: str | Path) -> None:
        self.model = SAC.load(str(path), env=self.env)
        logger.info("sac_loaded", path=str(path))

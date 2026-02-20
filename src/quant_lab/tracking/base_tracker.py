"""Abstract base class for experiment trackers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseTracker(ABC):
    """Interface for experiment tracking backends."""

    @abstractmethod
    def start_run(self, run_name: str | None = None) -> None:
        """Start a new tracking run."""
        ...

    @abstractmethod
    def end_run(self) -> None:
        """End the current tracking run."""
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics."""
        ...

    @abstractmethod
    def log_artifact(self, path: str | Path, artifact_name: str | None = None) -> None:
        """Log a file artifact."""
        ...

    def log_config(self, config: dict) -> None:
        """Log a nested config dict by flattening it."""
        flat = self._flatten_dict(config)
        self.log_params(flat)

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, str]:
        """Flatten a nested dict into dot-separated keys."""
        items: list[tuple[str, str]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(BaseTracker._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)

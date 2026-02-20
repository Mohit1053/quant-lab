"""Training callbacks: early stopping, checkpointing."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import structlog

logger = structlog.get_logger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' (loss) or 'max' (metric like accuracy).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            (score < self.best_score - self.min_delta)
            if self.mode == "min"
            else (score > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "early_stopping",
                    patience=self.patience,
                    best_score=self.best_score,
                )
                return True

        return False


class ModelCheckpoint:
    """Save model checkpoints when validation improves.

    Args:
        save_dir: Directory to save checkpoints.
        mode: 'min' (loss) or 'max' (metric).
        save_last: Also save the latest checkpoint every epoch.
    """

    def __init__(
        self,
        save_dir: str | Path,
        mode: str = "min",
        save_last: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.save_last = save_last
        self.best_score: float | None = None

    def __call__(
        self,
        score: float,
        model: nn.Module,
        epoch: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> bool:
        """Save checkpoint if score improved.

        Returns:
            True if this was the best checkpoint.
        """
        is_best = False
        if self.best_score is None:
            is_best = True
        elif self.mode == "min" and score < self.best_score:
            is_best = True
        elif self.mode == "max" and score > self.best_score:
            is_best = True

        if is_best:
            self.best_score = score
            self._save(model, optimizer, epoch, self.save_dir / "best.pt")
            logger.info("checkpoint_saved", path="best.pt", score=score, epoch=epoch)

        if self.save_last:
            self._save(model, optimizer, epoch, self.save_dir / "last.pt")

        return is_best

    def _save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        path: Path,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        # Save config if model has it
        if hasattr(model, "config"):
            checkpoint["config"] = model.config
        torch.save(checkpoint, path)

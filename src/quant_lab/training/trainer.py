"""Training loop with mixed precision, gradient clipping, and callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import structlog

from quant_lab.training.callbacks import EarlyStopping, ModelCheckpoint
from quant_lab.training.schedulers import cosine_warmup_scheduler
from quant_lab.utils.device import get_device, get_dtype

logger = structlog.get_logger(__name__)


@dataclass
class TrainerConfig:
    """Training configuration."""

    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    patience: int = 10
    mixed_precision: bool = True
    checkpoint_dir: str = "outputs/models"
    log_every_n_steps: int = 50


class Trainer:
    """Handles the training loop for transformer-based forecasters.

    Features:
    - Mixed precision training (BF16/FP16)
    - Gradient clipping
    - Cosine warmup learning rate schedule
    - Early stopping
    - Model checkpointing
    - Per-step and per-epoch logging
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainerConfig | None = None,
        device: torch.device | None = None,
        tracker: Any | None = None,
    ):
        self.config = config or TrainerConfig()
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.tracker = tracker

        # Determine mixed precision dtype
        self.use_amp = self.config.mixed_precision and self.device.type != "cpu"
        if self.use_amp:
            self.amp_dtype = get_dtype()
        else:
            self.amp_dtype = torch.float32

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # GradScaler for FP16 (not needed for BF16 but harmless)
        self.scaler = GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)

        # Callbacks
        self.early_stopping = EarlyStopping(patience=self.config.patience)
        self.checkpoint = ModelCheckpoint(
            save_dir=self.config.checkpoint_dir,
            mode="min",
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """Train the model.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.

        Returns:
            History dict with per-epoch 'train_loss', 'val_loss'.
        """
        # Set up scheduler (clamp warmup to not exceed total)
        total_steps = self.config.epochs * len(train_loader)
        warmup_steps = min(self.config.warmup_steps, total_steps // 2)
        self.scheduler = cosine_warmup_scheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        global_step = 0

        logger.info(
            "training_start",
            epochs=self.config.epochs,
            total_steps=total_steps,
            device=str(self.device),
            amp_dtype=str(self.amp_dtype),
            parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )

        for epoch in range(self.config.epochs):
            # Training
            train_loss, global_step = self._train_epoch(train_loader, epoch, global_step)
            history["train_loss"].append(train_loss)

            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                history["val_loss"].append(val_loss)

            logger.info(
                "epoch_complete",
                epoch=epoch + 1,
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.6f}" if val_loss is not None else "N/A",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

            # Log to tracker
            if self.tracker is not None:
                metrics = {"train_loss": train_loss, "epoch": epoch + 1}
                if val_loss is not None:
                    metrics["val_loss"] = val_loss
                self.tracker.log_metrics(metrics, step=epoch)

            # Checkpointing and early stopping
            checkpoint_score = val_loss if val_loss is not None else train_loss
            self.checkpoint(checkpoint_score, self.model, epoch, self.optimizer)
            if val_loss is not None:
                if self.early_stopping(val_loss):
                    logger.info("early_stopping_triggered", epoch=epoch + 1)
                    break

        logger.info("training_complete", epochs_run=epoch + 1)
        return history

    def _train_epoch(
        self, loader: DataLoader, epoch: int, global_step: int
    ) -> tuple[float, int]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (features, targets) in enumerate(loader):
            features = features.to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}

            # Forward with mixed precision
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(features)
                loss, loss_dict = self.loss_fn(outputs, targets)

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % self.config.log_every_n_steps == 0:
                logger.debug(
                    "train_step",
                    step=global_step,
                    loss=f"{loss.item():.6f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

        return total_loss / max(n_batches, 1), global_step

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for features, targets in loader:
            features = features.to(self.device, non_blocking=True)
            targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(features)
                loss, _ = self.loss_fn(outputs, targets)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> torch.Tensor:
        """Generate predictions for a dataset."""
        self.model.eval()
        predictions = []

        for features, _ in loader:
            features = features.to(self.device, non_blocking=True)
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                preds = self.model.predict_returns(features)
            predictions.append(preds.cpu())

        return torch.cat(predictions, dim=0)

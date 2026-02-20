"""Pre-training loop for masked time-series encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import structlog

from quant_lab.representation.masked_encoder import MaskedTimeSeriesEncoder
from quant_lab.training.callbacks import EarlyStopping, ModelCheckpoint
from quant_lab.training.schedulers import cosine_warmup_scheduler
from quant_lab.utils.device import get_device, get_dtype

logger = structlog.get_logger(__name__)


@dataclass
class PretrainConfig:
    """Pre-training configuration."""

    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    mask_ratio: float = 0.15
    mixed_precision: bool = True
    checkpoint_dir: str = "outputs/models/pretrained"
    log_every_n_steps: int = 50


class PreTrainer:
    """Pre-training loop for the masked time-series encoder.

    Uses MSE reconstruction loss on masked patches.
    """

    def __init__(
        self,
        model: MaskedTimeSeriesEncoder,
        config: PretrainConfig | None = None,
        device: torch.device | None = None,
        tracker=None,
    ):
        self.config = config or PretrainConfig()
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.tracker = tracker

        self.use_amp = self.config.mixed_precision and self.device.type != "cpu"
        self.amp_dtype = get_dtype() if self.use_amp else torch.float32

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)

        self.early_stopping = EarlyStopping(patience=10)
        self.checkpoint = ModelCheckpoint(
            save_dir=self.config.checkpoint_dir, mode="min",
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """Run pre-training.

        DataLoader should yield (features, targets) where features is
        (batch, seq_len, num_features). Targets are ignored for pre-training.
        """
        total_steps = self.config.epochs * len(train_loader)
        self.scheduler = cosine_warmup_scheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        global_step = 0

        logger.info(
            "pretrain_start",
            epochs=self.config.epochs,
            mask_ratio=self.config.mask_ratio,
            parameters=self.model.count_parameters(),
        )

        for epoch in range(self.config.epochs):
            train_loss, global_step = self._train_epoch(train_loader, epoch, global_step)
            history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                history["val_loss"].append(val_loss)

            logger.info(
                "pretrain_epoch",
                epoch=epoch + 1,
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.6f}" if val_loss is not None else "N/A",
            )

            if self.tracker is not None:
                metrics = {"pretrain_train_loss": train_loss, "pretrain_epoch": epoch + 1}
                if val_loss is not None:
                    metrics["pretrain_val_loss"] = val_loss
                self.tracker.log_metrics(metrics, step=epoch)

            if val_loss is not None:
                self.checkpoint(val_loss, self.model, epoch, self.optimizer)
                if self.early_stopping(val_loss):
                    logger.info("pretrain_early_stop", epoch=epoch + 1)
                    break

        logger.info("pretrain_complete", epochs_run=epoch + 1)
        return history

    def _train_epoch(self, loader, epoch, global_step):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for features, _ in loader:
            features = features.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                reconstructed, original, mask = self.model(
                    features, mask_ratio=self.config.mask_ratio
                )
                loss = F.mse_loss(reconstructed, original)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

        return total_loss / max(n_batches, 1), global_step

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for features, _ in loader:
            features = features.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                reconstructed, original, mask = self.model(
                    features, mask_ratio=self.config.mask_ratio
                )
                loss = F.mse_loss(reconstructed, original)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

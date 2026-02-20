"""DataModule: wraps datasets into train/val/test DataLoaders."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

import structlog

from quant_lab.data.datasets import TimeSeriesDataset, TemporalSplit, create_temporal_splits

logger = structlog.get_logger(__name__)


@dataclass
class DataModuleConfig:
    """DataModule configuration."""

    sequence_length: int = 63
    target_col: str = "log_return_1d"
    batch_size: int = 64
    num_workers: int = field(
        default_factory=lambda: 0 if sys.platform == "win32" else min(4, os.cpu_count() or 1)
    )
    pin_memory: bool = True


class QuantDataModule:
    """Creates train/val/test DataLoaders from feature DataFrame.

    Handles temporal splitting to prevent data leakage,
    per-ticker sliding window datasets, and DataLoader configuration.
    """

    def __init__(
        self,
        feature_df: pd.DataFrame,
        feature_cols: list[str],
        split: TemporalSplit,
        config: DataModuleConfig | None = None,
    ):
        self.feature_df = feature_df
        self.feature_cols = feature_cols
        self.split = split
        self.config = config or DataModuleConfig()

        self._train_ds: ConcatDataset | None = None
        self._val_ds: ConcatDataset | None = None
        self._test_ds: ConcatDataset | None = None

    def setup(self) -> None:
        """Create datasets with temporal splits."""
        splits = create_temporal_splits(
            df=self.feature_df,
            feature_cols=self.feature_cols,
            split=self.split,
            sequence_length=self.config.sequence_length,
            target_col=self.config.target_col,
        )

        self._train_ds = ConcatDataset(splits["train"]) if splits["train"] else None
        self._val_ds = ConcatDataset(splits["val"]) if splits["val"] else None
        self._test_ds = ConcatDataset(splits["test"]) if splits["test"] else None

        logger.info(
            "datamodule_setup",
            train_samples=len(self._train_ds) if self._train_ds else 0,
            val_samples=len(self._val_ds) if self._val_ds else 0,
            test_samples=len(self._test_ds) if self._test_ds else 0,
            num_features=len(self.feature_cols),
            sequence_length=self.config.sequence_length,
        )

    @property
    def num_features(self) -> int:
        return len(self.feature_cols)

    def train_dataloader(self) -> DataLoader | None:
        if self._train_ds is None or len(self._train_ds) == 0:
            return None
        return DataLoader(
            self._train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._val_ds is None or len(self._val_ds) == 0:
            return None
        return DataLoader(
            self._val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader | None:
        if self._test_ds is None or len(self._test_ds) == 0:
            return None
        return DataLoader(
            self._test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
        )

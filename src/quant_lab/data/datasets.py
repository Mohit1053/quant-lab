"""PyTorch Dataset classes for time-series sliding windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TemporalSplit:
    """Date-based train/val/test split configuration."""

    train_end: str
    val_end: str
    # test = everything after val_end


class TimeSeriesDataset(Dataset):
    """Sliding window dataset for multi-asset time-series.

    Each sample is a (features_window, target) pair:
    - features_window: (sequence_length, num_features) tensor
    - target: dict with 'returns' (next-day log return) for the ticker
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        sequence_length: int = 63,
        target_col: str = "log_return_1d",
        ticker: str | None = None,
    ):
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Filter to single ticker if specified
        if ticker is not None:
            df = df[df["ticker"] == ticker].copy()

        df = df.sort_values("date").reset_index(drop=True)

        # Extract feature matrix and target vector
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32) if target_col in df.columns else None
        self.dates = df["date"].values
        self.tickers = df["ticker"].values if "ticker" in df.columns else None

        # Build valid indices (where we have a full window + 1 target)
        self.valid_indices = []
        for i in range(sequence_length, len(df)):
            # Check no NaN in the window features or target
            window = self.features[i - sequence_length : i]
            if not np.any(np.isnan(window)):
                if self.targets is not None and not np.isnan(self.targets[i]):
                    self.valid_indices.append(i)

        logger.debug(
            "dataset_created",
            total_rows=len(df),
            valid_samples=len(self.valid_indices),
            sequence_length=sequence_length,
            num_features=len(feature_cols),
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        i = self.valid_indices[idx]

        # Feature window: (seq_len, num_features)
        window = self.features[i - self.sequence_length : i]
        x = torch.from_numpy(window)

        target = {
            "returns": torch.tensor(self.targets[i], dtype=torch.float32),
        }

        return x, target


def create_temporal_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: TemporalSplit,
    sequence_length: int = 63,
    target_col: str = "log_return_1d",
) -> dict[str, list[TimeSeriesDataset]]:
    """Create train/val/test datasets with proper temporal separation.

    Returns a dict mapping split name to a list of per-ticker datasets.
    This prevents any data leakage across time periods.
    """
    train_end = pd.Timestamp(split.train_end)
    val_end = pd.Timestamp(split.val_end)

    df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["date"] <= train_end]
    val_df = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test_df = df[df["date"] > val_end]

    tickers = df["ticker"].unique()

    splits = {"train": [], "val": [], "test": []}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for ticker in tickers:
            ds = TimeSeriesDataset(
                df=split_df,
                feature_cols=feature_cols,
                sequence_length=sequence_length,
                target_col=target_col,
                ticker=ticker,
            )
            if len(ds) > 0:
                splits[split_name].append(ds)

    for name, datasets in splits.items():
        total_samples = sum(len(ds) for ds in datasets)
        logger.info(f"split_{name}", num_tickers=len(datasets), total_samples=total_samples)

    return splits


def create_flat_datasets(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: TemporalSplit,
    target_col: str = "log_return_1d",
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """Create flat (non-sequential) train/val/test arrays for sklearn models.

    Returns dict mapping split name to (X, y, metadata_df) tuples.
    metadata_df contains date and ticker for each row.
    """
    train_end = pd.Timestamp(split.train_end)
    val_end = pd.Timestamp(split.val_end)

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=feature_cols + [target_col])

    result = {}
    for name, mask in [
        ("train", df["date"] <= train_end),
        ("val", (df["date"] > train_end) & (df["date"] <= val_end)),
        ("test", df["date"] > val_end),
    ]:
        split_df = df[mask]
        X = split_df[feature_cols].values.astype(np.float32)
        y = split_df[target_col].values.astype(np.float32)
        meta = split_df[["date", "ticker"]].reset_index(drop=True)
        result[name] = (X, y, meta)
        logger.info(f"flat_split_{name}", samples=len(X))

    return result

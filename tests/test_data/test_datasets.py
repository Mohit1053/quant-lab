"""Tests for PyTorch dataset and temporal splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.data.datasets import (
    TimeSeriesDataset,
    TemporalSplit,
    create_flat_datasets,
)


class TestTimeSeriesDataset:
    def test_dataset_creation(self, synthetic_features):
        feature_cols = ["log_return_1d", "log_return_5d", "volatility_21d", "momentum_21d"]
        ds = TimeSeriesDataset(
            df=synthetic_features,
            feature_cols=feature_cols,
            sequence_length=21,
            target_col="log_return_1d",
            ticker="TICKER_A",
        )
        assert len(ds) > 0

    def test_dataset_output_shape(self, synthetic_features):
        feature_cols = ["log_return_1d", "log_return_5d", "volatility_21d", "momentum_21d"]
        ds = TimeSeriesDataset(
            df=synthetic_features,
            feature_cols=feature_cols,
            sequence_length=21,
            target_col="log_return_1d",
            ticker="TICKER_A",
        )
        x, target = ds[0]
        assert x.shape == (21, 4)  # (seq_len, num_features)
        assert "returns" in target
        assert target["returns"].ndim == 0  # scalar

    def test_dataset_no_nan(self, synthetic_features):
        feature_cols = ["log_return_1d", "log_return_5d", "volatility_21d", "momentum_21d"]
        ds = TimeSeriesDataset(
            df=synthetic_features,
            feature_cols=feature_cols,
            sequence_length=21,
            target_col="log_return_1d",
            ticker="TICKER_A",
        )
        for i in range(min(10, len(ds))):
            x, target = ds[i]
            assert not x.isnan().any(), f"NaN in features at index {i}"
            assert not target["returns"].isnan(), f"NaN in target at index {i}"


class TestFlatDatasets:
    def test_temporal_split_no_leakage(self, synthetic_features):
        feature_cols = ["log_return_1d", "log_return_5d", "volatility_21d", "momentum_21d"]
        # Synthetic data: 504 bdays from 2020-01-01 (~2020-01 to 2021-12)
        split = TemporalSplit(train_end="2020-12-31", val_end="2021-06-30")

        datasets = create_flat_datasets(
            synthetic_features, feature_cols, split, target_col="log_return_1d"
        )

        X_train, y_train, meta_train = datasets["train"]
        X_val, y_val, meta_val = datasets["val"]
        X_test, y_test, meta_test = datasets["test"]

        # All splits should have data
        assert len(X_train) > 0, "No training data"
        assert len(X_val) > 0, "No validation data"
        assert len(X_test) > 0, "No test data"

        # Check no temporal overlap
        train_max = pd.Timestamp(meta_train["date"].max())
        val_min = pd.Timestamp(meta_val["date"].min())
        val_max = pd.Timestamp(meta_val["date"].max())
        test_min = pd.Timestamp(meta_test["date"].min())

        assert train_max < val_min, "Train/val overlap"
        assert val_max < test_min, "Val/test overlap"

    def test_flat_datasets_shapes(self, synthetic_features):
        feature_cols = ["log_return_1d", "log_return_5d", "volatility_21d", "momentum_21d"]
        split = TemporalSplit(train_end="2020-12-31", val_end="2021-06-30")

        datasets = create_flat_datasets(
            synthetic_features, feature_cols, split, target_col="log_return_1d"
        )

        for name in ["train", "val", "test"]:
            X, y, meta = datasets[name]
            assert X.shape[1] == len(feature_cols)
            assert len(X) == len(y)
            assert len(X) == len(meta)

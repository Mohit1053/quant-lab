"""Tests for the DataModule."""

from __future__ import annotations

from quant_lab.data.datasets import TemporalSplit
from quant_lab.data.datamodule import QuantDataModule, DataModuleConfig


class TestQuantDataModule:
    def test_setup_creates_loaders(self, synthetic_features):
        feature_cols = ["log_return_1d", "log_return_5d", "volatility_21d", "momentum_21d"]
        split = TemporalSplit(train_end="2020-12-31", val_end="2021-06-30")
        config = DataModuleConfig(sequence_length=10, batch_size=4)

        dm = QuantDataModule(synthetic_features, feature_cols, split, config)
        dm.setup()

        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

    def test_num_features(self, synthetic_features):
        feature_cols = ["log_return_1d", "log_return_5d", "volatility_21d", "momentum_21d"]
        split = TemporalSplit(train_end="2020-12-31", val_end="2021-06-30")

        dm = QuantDataModule(synthetic_features, feature_cols, split)
        assert dm.num_features == 4

    def test_batch_shape(self, synthetic_features):
        feature_cols = ["log_return_1d", "log_return_5d", "volatility_21d", "momentum_21d"]
        split = TemporalSplit(train_end="2020-12-31", val_end="2021-06-30")
        config = DataModuleConfig(sequence_length=10, batch_size=4)

        dm = QuantDataModule(synthetic_features, feature_cols, split, config)
        dm.setup()

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        features, targets = batch

        assert features.shape[0] == 4  # batch_size
        assert features.shape[1] == 10  # sequence_length
        assert features.shape[2] == 4  # num_features
        assert "returns" in targets
        assert targets["returns"].shape == (4,)

    def test_no_temporal_leakage(self, synthetic_features):
        feature_cols = ["log_return_1d", "log_return_5d", "volatility_21d", "momentum_21d"]
        split = TemporalSplit(train_end="2020-12-31", val_end="2021-06-30")
        config = DataModuleConfig(sequence_length=10, batch_size=4)

        dm = QuantDataModule(synthetic_features, feature_cols, split, config)
        dm.setup()

        for loader in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
            assert loader is not None
            batch = next(iter(loader))
            assert batch[0].shape[2] == 4

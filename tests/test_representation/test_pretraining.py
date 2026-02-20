"""Tests for pre-training loop."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from quant_lab.representation.masked_encoder import MaskedTimeSeriesEncoder, MaskedEncoderConfig
from quant_lab.representation.pretraining import PreTrainer, PretrainConfig


def _make_dummy_loader(batch_size=8, n_samples=32, seq_len=50, n_features=8):
    """Create dummy DataLoader for pre-training."""
    features = torch.randn(n_samples, seq_len, n_features)
    returns = torch.randn(n_samples) * 0.01

    class DummyDS(torch.utils.data.Dataset):
        def __init__(self, feats, rets):
            self.feats = feats
            self.rets = rets

        def __len__(self):
            return len(self.feats)

        def __getitem__(self, idx):
            return self.feats[idx], {"returns": self.rets[idx]}

    return DataLoader(DummyDS(features, returns), batch_size=batch_size, drop_last=True)


class TestPreTrainer:
    def test_fit_runs(self):
        config = MaskedEncoderConfig(
            num_features=8, patch_size=5, d_model=16, nhead=2,
            num_encoder_layers=1, dim_feedforward=32, dropout=0.0, mask_ratio=0.3,
        )
        model = MaskedTimeSeriesEncoder(config)

        pt_config = PretrainConfig(
            epochs=2, learning_rate=1e-3, warmup_steps=2,
            mixed_precision=False, checkpoint_dir="outputs/test_pretrain",
        )
        pretrainer = PreTrainer(model, pt_config, device=torch.device("cpu"))

        train_loader = _make_dummy_loader()
        history = pretrainer.fit(train_loader)

        assert len(history["train_loss"]) == 2

    def test_loss_decreases(self):
        torch.manual_seed(42)
        config = MaskedEncoderConfig(
            num_features=8, patch_size=5, d_model=32, nhead=2,
            num_encoder_layers=1, dim_feedforward=64, dropout=0.0, mask_ratio=0.3,
        )
        model = MaskedTimeSeriesEncoder(config)

        pt_config = PretrainConfig(
            epochs=30, learning_rate=5e-3, warmup_steps=5,
            mixed_precision=False, checkpoint_dir="outputs/test_pretrain2",
        )
        pretrainer = PreTrainer(model, pt_config, device=torch.device("cpu"))

        train_loader = _make_dummy_loader(n_samples=64)
        history = pretrainer.fit(train_loader)

        # Compare average of first 5 vs last 5 epochs to smooth noise
        avg_first = sum(history["train_loss"][:5]) / 5
        avg_last = sum(history["train_loss"][-5:]) / 5
        assert avg_last < avg_first

    def test_fit_with_validation(self):
        config = MaskedEncoderConfig(
            num_features=8, patch_size=5, d_model=16, nhead=2,
            num_encoder_layers=1, dim_feedforward=32, dropout=0.0, mask_ratio=0.3,
        )
        model = MaskedTimeSeriesEncoder(config)

        pt_config = PretrainConfig(
            epochs=3, learning_rate=1e-3, warmup_steps=2,
            mixed_precision=False, checkpoint_dir="outputs/test_pretrain3",
        )
        pretrainer = PreTrainer(model, pt_config, device=torch.device("cpu"))

        train_loader = _make_dummy_loader()
        val_loader = _make_dummy_loader(n_samples=16)
        history = pretrainer.fit(train_loader, val_loader)

        assert len(history["val_loss"]) == 3

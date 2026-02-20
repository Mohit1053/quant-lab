"""Tests for the training loop."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from quant_lab.models.transformer.model import TransformerForecaster, TransformerConfig, MultiTaskLoss
from quant_lab.training.trainer import Trainer, TrainerConfig


def _make_dummy_loader(batch_size: int = 8, n_samples: int = 32, seq_len: int = 10, n_features: int = 8):
    """Create a dummy DataLoader mimicking TimeSeriesDataset output."""
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

    ds = DummyDS(features, returns)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


class TestTrainer:
    def test_fit_runs(self):
        config = TransformerConfig(
            num_features=8, d_model=16, nhead=2, num_encoder_layers=1,
            dim_feedforward=32, dropout=0.0,
        )
        model = TransformerForecaster(config)
        loss_fn = MultiTaskLoss(config)

        trainer_config = TrainerConfig(
            epochs=2, learning_rate=1e-3, warmup_steps=5,
            patience=10, mixed_precision=False,
            checkpoint_dir="outputs/test_checkpoints",
        )
        trainer = Trainer(model, loss_fn, trainer_config, device=torch.device("cpu"))

        train_loader = _make_dummy_loader()
        val_loader = _make_dummy_loader(n_samples=16)

        history = trainer.fit(train_loader, val_loader)
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_train_loss_decreases(self):
        config = TransformerConfig(
            num_features=8, d_model=16, nhead=2, num_encoder_layers=1,
            dim_feedforward=32, dropout=0.0,
        )
        model = TransformerForecaster(config)
        loss_fn = MultiTaskLoss(config)

        trainer_config = TrainerConfig(
            epochs=10, learning_rate=1e-3, warmup_steps=2,
            patience=20, mixed_precision=False,
            checkpoint_dir="outputs/test_checkpoints2",
        )
        trainer = Trainer(model, loss_fn, trainer_config, device=torch.device("cpu"))

        train_loader = _make_dummy_loader(n_samples=64)
        history = trainer.fit(train_loader)

        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_predict(self):
        config = TransformerConfig(
            num_features=8, d_model=16, nhead=2, num_encoder_layers=1,
            dim_feedforward=32, dropout=0.0,
        )
        model = TransformerForecaster(config)
        loss_fn = MultiTaskLoss(config)

        trainer_config = TrainerConfig(
            epochs=1, mixed_precision=False,
            checkpoint_dir="outputs/test_checkpoints3",
        )
        trainer = Trainer(model, loss_fn, trainer_config, device=torch.device("cpu"))

        test_loader = _make_dummy_loader(n_samples=16, batch_size=8)
        preds = trainer.predict(test_loader)
        assert preds.shape == (16,)

    def test_early_stopping(self):
        config = TransformerConfig(
            num_features=8, d_model=16, nhead=2, num_encoder_layers=1,
            dim_feedforward=32, dropout=0.0,
        )
        model = TransformerForecaster(config)
        loss_fn = MultiTaskLoss(config)

        trainer_config = TrainerConfig(
            epochs=100,
            learning_rate=0.0,  # Zero LR = no learning = val loss won't improve
            warmup_steps=0,
            patience=3,
            mixed_precision=False,
            checkpoint_dir="outputs/test_checkpoints4",
        )
        trainer = Trainer(model, loss_fn, trainer_config, device=torch.device("cpu"))

        train_loader = _make_dummy_loader()
        val_loader = _make_dummy_loader(n_samples=16)

        history = trainer.fit(train_loader, val_loader)

        # Should stop well before 100 epochs due to early stopping
        # (shuffle=True causes tiny floating point loss variations, so allow some slack)
        assert len(history["train_loss"]) < 100

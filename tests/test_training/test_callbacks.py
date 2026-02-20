"""Tests for training callbacks."""

from __future__ import annotations

import torch
import torch.nn as nn

from quant_lab.training.callbacks import EarlyStopping, ModelCheckpoint


class TestEarlyStopping:
    def test_no_stop_when_improving(self):
        es = EarlyStopping(patience=3, mode="min")
        assert not es(10.0)
        assert not es(9.0)
        assert not es(8.0)
        assert not es(7.0)
        assert not es.should_stop

    def test_stop_after_patience(self):
        es = EarlyStopping(patience=3, mode="min")
        es(10.0)
        es(11.0)  # worse
        es(12.0)  # worse
        result = es(13.0)  # worse -> stop
        assert result is True
        assert es.should_stop

    def test_reset_on_improvement(self):
        es = EarlyStopping(patience=3, mode="min")
        es(10.0)
        es(11.0)  # worse (counter=1)
        es(12.0)  # worse (counter=2)
        es(8.0)   # improved -> reset counter
        assert es.counter == 0
        assert not es.should_stop

    def test_max_mode(self):
        es = EarlyStopping(patience=2, mode="max")
        es(0.5)
        es(0.6)  # improved
        es(0.55)  # worse
        result = es(0.55)  # worse -> stop
        assert result is True

    def test_min_delta(self):
        es = EarlyStopping(patience=2, mode="min", min_delta=0.1)
        es(10.0)
        es(9.95)  # improved by 0.05 < min_delta, counts as no improvement
        result = es(9.95)  # still no improvement -> stop
        assert result is True


class TestModelCheckpoint:
    def test_saves_best(self, tmp_path):
        ckpt = ModelCheckpoint(save_dir=tmp_path, mode="min", save_last=False)
        model = nn.Linear(10, 1)

        is_best = ckpt(10.0, model, epoch=0)
        assert is_best
        assert (tmp_path / "best.pt").exists()

    def test_saves_last(self, tmp_path):
        ckpt = ModelCheckpoint(save_dir=tmp_path, mode="min", save_last=True)
        model = nn.Linear(10, 1)

        ckpt(10.0, model, epoch=0)
        assert (tmp_path / "last.pt").exists()

    def test_only_saves_best_on_improvement(self, tmp_path):
        ckpt = ModelCheckpoint(save_dir=tmp_path, mode="min", save_last=False)
        model = nn.Linear(10, 1)

        assert ckpt(10.0, model, epoch=0)  # best
        assert not ckpt(11.0, model, epoch=1)  # worse
        assert ckpt(9.0, model, epoch=2)  # new best

    def test_saves_optimizer_state(self, tmp_path):
        ckpt = ModelCheckpoint(save_dir=tmp_path, mode="min", save_last=False)
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())

        ckpt(10.0, model, epoch=0, optimizer=optimizer)
        loaded = torch.load(tmp_path / "best.pt", weights_only=False)
        assert "optimizer_state_dict" in loaded

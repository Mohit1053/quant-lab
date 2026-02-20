"""Tests for learning rate schedulers."""

from __future__ import annotations

import torch.nn as nn
import torch.optim

from quant_lab.training.schedulers import cosine_warmup_scheduler, linear_warmup_scheduler


class TestCosineWarmupScheduler:
    def test_warmup_starts_at_zero(self):
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=100, total_steps=1000)

        # Step 0: LR should be near 0
        assert optimizer.param_groups[0]["lr"] < 1e-5

    def test_warmup_reaches_peak(self):
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=100, total_steps=1000)

        for _ in range(100):
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        assert abs(lr - 1e-3) < 1e-5

    def test_cosine_decay_after_warmup(self):
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=100, total_steps=1000)

        for _ in range(200):
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        assert lr < 1e-3

    def test_lr_never_negative(self):
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=10, total_steps=100)

        for _ in range(150):
            scheduler.step()
            assert optimizer.param_groups[0]["lr"] >= 0


class TestLinearWarmupScheduler:
    def test_warmup_then_constant(self):
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = linear_warmup_scheduler(optimizer, warmup_steps=10)

        for _ in range(10):
            scheduler.step()

        lr_at_warmup_end = optimizer.param_groups[0]["lr"]

        for _ in range(50):
            scheduler.step()

        lr_after = optimizer.param_groups[0]["lr"]
        assert abs(lr_at_warmup_end - lr_after) < 1e-8

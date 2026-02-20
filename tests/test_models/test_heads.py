"""Tests for prediction heads."""

from __future__ import annotations

import torch

from quant_lab.models.heads.distribution_head import GaussianHead, StudentTHead
from quant_lab.models.heads.direction_head import DirectionHead
from quant_lab.models.heads.volatility_head import VolatilityHead


class TestGaussianHead:
    def test_output_keys(self):
        head = GaussianHead(d_input=32)
        x = torch.randn(4, 32)
        out = head(x)
        assert "mean" in out
        assert "log_var" in out

    def test_output_shapes(self):
        head = GaussianHead(d_input=32)
        x = torch.randn(4, 32)
        out = head(x)
        assert out["mean"].shape == (4,)
        assert out["log_var"].shape == (4,)

    def test_with_hidden_dim(self):
        head = GaussianHead(d_input=32, hidden_dim=16)
        x = torch.randn(4, 32)
        out = head(x)
        assert out["mean"].shape == (4,)


class TestStudentTHead:
    def test_output_keys(self):
        head = StudentTHead(d_input=32)
        x = torch.randn(4, 32)
        out = head(x)
        assert "loc" in out
        assert "log_scale" in out
        assert "log_df" in out

    def test_output_shapes(self):
        head = StudentTHead(d_input=32)
        x = torch.randn(4, 32)
        out = head(x)
        assert out["loc"].shape == (4,)
        assert out["log_scale"].shape == (4,)
        assert out["log_df"].shape == (4,)


class TestDirectionHead:
    def test_output_shape(self):
        head = DirectionHead(d_input=32, num_classes=3)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4, 3)

    def test_logits_sum_not_one(self):
        head = DirectionHead(d_input=32, num_classes=3)
        x = torch.randn(4, 32)
        out = head(x)
        assert not torch.allclose(out.sum(dim=1), torch.ones(4))

    def test_custom_num_classes(self):
        head = DirectionHead(d_input=32, num_classes=5)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4, 5)


class TestVolatilityHead:
    def test_output_shape(self):
        head = VolatilityHead(d_input=32)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4,)

    def test_always_positive(self):
        head = VolatilityHead(d_input=32)
        x = torch.randn(100, 32) * 10
        out = head(x)
        assert (out > 0).all()

    def test_gradient_flow(self):
        head = VolatilityHead(d_input=32)
        x = torch.randn(4, 32, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

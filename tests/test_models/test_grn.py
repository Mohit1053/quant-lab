"""Tests for Gated Residual Network components."""

from __future__ import annotations

import torch
import pytest

from quant_lab.models.tft.gated_residual import GatedLinearUnit, GatedResidualNetwork


class TestGatedLinearUnit:
    def test_output_shape(self):
        glu = GatedLinearUnit(32, 32)
        x = torch.randn(4, 32)
        out = glu(x)
        assert out.shape == (4, 32)

    def test_different_output_dim(self):
        glu = GatedLinearUnit(32, 16)
        x = torch.randn(4, 32)
        out = glu(x)
        assert out.shape == (4, 16)

    def test_gradient_flow(self):
        glu = GatedLinearUnit(32, 32)
        x = torch.randn(4, 32, requires_grad=True)
        out = glu(x)
        out.sum().backward()
        assert x.grad is not None

    def test_3d_input(self):
        glu = GatedLinearUnit(32, 32)
        x = torch.randn(4, 10, 32)
        out = glu(x)
        assert out.shape == (4, 10, 32)


class TestGatedResidualNetwork:
    def test_output_shape(self):
        grn = GatedResidualNetwork(32, d_hidden=64, dropout=0.0)
        x = torch.randn(4, 32)
        out = grn(x)
        assert out.shape == (4, 32)

    def test_with_context(self):
        grn = GatedResidualNetwork(32, d_hidden=64, d_context=16, dropout=0.0)
        x = torch.randn(4, 32)
        ctx = torch.randn(4, 16)
        out = grn(x, context=ctx)
        assert out.shape == (4, 32)

    def test_without_context_when_supported(self):
        grn = GatedResidualNetwork(32, d_hidden=64, d_context=16, dropout=0.0)
        x = torch.randn(4, 32)
        # Should still work without context (context_proj ignored)
        out = grn(x, context=None)
        assert out.shape == (4, 32)

    def test_residual_connection(self):
        """GRN output should differ from input (non-trivial transformation)."""
        grn = GatedResidualNetwork(32, d_hidden=64, dropout=0.0)
        x = torch.randn(4, 32)
        out = grn(x)
        assert not torch.allclose(x, out)

    def test_gradient_flow(self):
        grn = GatedResidualNetwork(32, d_hidden=64, dropout=0.0)
        x = torch.randn(4, 32, requires_grad=True)
        out = grn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_3d_input(self):
        grn = GatedResidualNetwork(32, d_hidden=64, dropout=0.0)
        x = torch.randn(4, 10, 32)
        out = grn(x)
        assert out.shape == (4, 10, 32)

    def test_default_hidden_dim(self):
        grn = GatedResidualNetwork(32, dropout=0.0)
        x = torch.randn(4, 32)
        out = grn(x)
        assert out.shape == (4, 32)

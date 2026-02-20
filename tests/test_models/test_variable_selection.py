"""Tests for Variable Selection Network."""

from __future__ import annotations

import torch
import pytest

from quant_lab.models.tft.variable_selection import VariableSelectionNetwork


class TestVariableSelectionNetwork:
    def test_output_shapes(self):
        vsn = VariableSelectionNetwork(
            num_features=8, d_model=32, d_hidden=16, dropout=0.0
        )
        x = torch.randn(4, 8)
        selected, weights = vsn(x)
        assert selected.shape == (4, 32)
        assert weights.shape == (4, 8)

    def test_weights_sum_to_one(self):
        vsn = VariableSelectionNetwork(
            num_features=8, d_model=32, d_hidden=16, dropout=0.0
        )
        x = torch.randn(4, 8)
        _, weights = vsn(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_weights_non_negative(self):
        vsn = VariableSelectionNetwork(
            num_features=8, d_model=32, d_hidden=16, dropout=0.0
        )
        x = torch.randn(4, 8)
        _, weights = vsn(x)
        assert (weights >= 0).all()

    def test_3d_input(self):
        """VSN applied per time step."""
        vsn = VariableSelectionNetwork(
            num_features=8, d_model=32, d_hidden=16, dropout=0.0
        )
        x = torch.randn(4, 10, 8)  # (batch, time, features)
        selected, weights = vsn(x)
        assert selected.shape == (4, 10, 32)
        assert weights.shape == (4, 10, 8)

    def test_gradient_flow(self):
        vsn = VariableSelectionNetwork(
            num_features=8, d_model=32, d_hidden=16, dropout=0.0
        )
        x = torch.randn(4, 8, requires_grad=True)
        selected, weights = vsn(x)
        selected.sum().backward()
        assert x.grad is not None

    def test_with_context(self):
        vsn = VariableSelectionNetwork(
            num_features=8, d_model=32, d_hidden=16, d_context=24, dropout=0.0
        )
        x = torch.randn(4, 8)
        ctx = torch.randn(4, 24)
        selected, weights = vsn(x, context=ctx)
        assert selected.shape == (4, 32)
        assert weights.shape == (4, 8)

    def test_different_num_features(self):
        for n_feat in [3, 5, 12]:
            vsn = VariableSelectionNetwork(
                num_features=n_feat, d_model=16, d_hidden=8, dropout=0.0
            )
            x = torch.randn(2, n_feat)
            selected, weights = vsn(x)
            assert selected.shape == (2, 16)
            assert weights.shape == (2, n_feat)

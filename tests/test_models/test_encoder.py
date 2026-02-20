"""Tests for transformer encoder."""

from __future__ import annotations

import torch

from quant_lab.models.transformer.encoder import TransformerEncoderLayer, TransformerEncoder


class TestTransformerEncoderLayer:
    def test_output_shape(self):
        layer = TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0)
        x = torch.randn(2, 10, 32)
        out = layer(x)
        assert out.shape == (2, 10, 32)

    def test_residual_connection(self):
        layer = TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0)
        x = torch.randn(2, 10, 32)
        out = layer(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestTransformerEncoder:
    def test_output_shape_with_cls(self):
        enc = TransformerEncoder(
            num_features=8, d_model=32, nhead=4, num_layers=2,
            dim_feedforward=64, dropout=0.0,
        )
        x = torch.randn(2, 20, 8)  # (batch=2, seq_len=20, features=8)
        out = enc(x)
        # Output includes CLS token: (batch, seq_len + 1, d_model)
        assert out.shape == (2, 21, 32)

    def test_cls_token_at_position_zero(self):
        enc = TransformerEncoder(
            num_features=8, d_model=32, nhead=4, num_layers=1,
            dim_feedforward=64, dropout=0.0,
        )
        enc.eval()
        x = torch.randn(2, 10, 8)
        out = enc(x)
        cls_out = out[:, 0]
        assert cls_out.shape == (2, 32)

    def test_different_num_features(self):
        for nf in [4, 10, 50]:
            enc = TransformerEncoder(
                num_features=nf, d_model=32, nhead=4, num_layers=1,
                dim_feedforward=64, dropout=0.0,
            )
            x = torch.randn(1, 10, nf)
            out = enc(x)
            assert out.shape == (1, 11, 32)

    def test_gradient_flow(self):
        enc = TransformerEncoder(
            num_features=8, d_model=32, nhead=4, num_layers=2,
            dim_feedforward=64, dropout=0.0,
        )
        x = torch.randn(2, 10, 8, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_deterministic_eval(self):
        enc = TransformerEncoder(
            num_features=8, d_model=32, nhead=4, num_layers=2,
            dim_feedforward=64, dropout=0.1,
        )
        enc.eval()
        x = torch.randn(2, 10, 8)
        out1 = enc(x)
        out2 = enc(x)
        assert torch.allclose(out1, out2)

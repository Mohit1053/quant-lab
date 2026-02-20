"""Tests for multi-head self-attention module."""

from __future__ import annotations

import torch
import pytest

from quant_lab.models.transformer.attention import MultiHeadSelfAttention, PositionalEncoding


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=32, max_len=100, dropout=0.0)
        x = torch.randn(2, 50, 32)
        out = pe(x)
        assert out.shape == (2, 50, 32)

    def test_adds_positional_info(self):
        pe = PositionalEncoding(d_model=32, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, 32)
        out = pe(x)
        # Output should not be all zeros (PE was added)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_different_positions_get_different_encodings(self):
        pe = PositionalEncoding(d_model=32, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, 32)
        out = pe(x)
        # Different positions should have different encodings
        assert not torch.allclose(out[0, 0], out[0, 1])


class TestMultiHeadSelfAttention:
    def test_output_shape(self):
        attn = MultiHeadSelfAttention(d_model=32, nhead=4, dropout=0.0)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert out.shape == (2, 10, 32)

    def test_batch_independence(self):
        attn = MultiHeadSelfAttention(d_model=32, nhead=4, dropout=0.0)
        attn.eval()
        x1 = torch.randn(1, 10, 32)
        x2 = torch.randn(1, 10, 32)
        x_batch = torch.cat([x1, x2], dim=0)
        out_batch = attn(x_batch)
        out_single = attn(x1)
        assert torch.allclose(out_batch[0], out_single[0], atol=1e-5)

    def test_causal_mask(self):
        attn = MultiHeadSelfAttention(d_model=32, nhead=4, dropout=0.0)
        attn.eval()
        x = torch.randn(1, 5, 32)
        out = attn(x, is_causal=True)
        assert out.shape == (1, 5, 32)

    def test_d_model_nhead_validation(self):
        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(d_model=32, nhead=5)  # 32 not divisible by 5

    def test_gradient_flow(self):
        attn = MultiHeadSelfAttention(d_model=32, nhead=4, dropout=0.0)
        x = torch.randn(2, 10, 32, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

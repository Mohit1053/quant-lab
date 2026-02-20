"""Tests for patch tokenizer."""

from __future__ import annotations

import torch

from quant_lab.representation.tokenizer import PatchTokenizer


class TestPatchTokenizer:
    def test_output_shape(self):
        tok = PatchTokenizer(num_features=8, patch_size=5, d_model=32, dropout=0.0)
        x = torch.randn(2, 50, 8)  # 50 time steps -> 10 patches of size 5
        out = tok(x)
        assert out.shape == (2, 10, 32)

    def test_trimming(self):
        """If seq_len not divisible by patch_size, extra steps are trimmed."""
        tok = PatchTokenizer(num_features=8, patch_size=5, d_model=32, dropout=0.0)
        x = torch.randn(2, 53, 8)  # 53 // 5 = 10 patches, 3 extra trimmed
        out = tok(x)
        assert out.shape == (2, 10, 32)

    def test_different_patch_sizes(self):
        for ps in [3, 5, 10]:
            tok = PatchTokenizer(num_features=4, patch_size=ps, d_model=16, dropout=0.0)
            x = torch.randn(1, 60, 4)
            out = tok(x)
            expected_patches = 60 // ps
            assert out.shape == (1, expected_patches, 16)

    def test_gradient_flow(self):
        tok = PatchTokenizer(num_features=8, patch_size=5, d_model=32, dropout=0.0)
        x = torch.randn(2, 50, 8, requires_grad=True)
        out = tok(x)
        out.sum().backward()
        assert x.grad is not None

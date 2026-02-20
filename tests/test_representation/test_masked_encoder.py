"""Tests for masked time-series encoder."""

from __future__ import annotations

import torch
import pytest

from quant_lab.representation.masked_encoder import MaskedTimeSeriesEncoder, MaskedEncoderConfig


@pytest.fixture
def tiny_encoder_config():
    return MaskedEncoderConfig(
        num_features=8,
        patch_size=5,
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        mask_ratio=0.3,
    )


@pytest.fixture
def tiny_encoder(tiny_encoder_config):
    return MaskedTimeSeriesEncoder(tiny_encoder_config)


class TestMaskedTimeSeriesEncoder:
    def test_forward_returns_three_tensors(self, tiny_encoder):
        x = torch.randn(4, 50, 8)  # 50 time steps -> 10 patches
        reconstructed, original, mask = tiny_encoder(x)
        assert reconstructed.ndim == 2  # (total_masked, patch_size * features)
        assert original.ndim == 2
        assert mask.shape == (4, 10)  # (batch, num_patches)

    def test_mask_ratio_controls_num_masked(self, tiny_encoder):
        x = torch.randn(4, 50, 8)  # 10 patches
        _, _, mask = tiny_encoder(x, mask_ratio=0.3)
        # With 10 patches and 0.3 ratio, expect 3 masked per sample
        assert mask.sum(dim=1).float().mean().item() == pytest.approx(3.0, abs=0.1)

    def test_reconstructed_matches_masked_count(self, tiny_encoder):
        x = torch.randn(4, 50, 8)
        reconstructed, original, mask = tiny_encoder(x, mask_ratio=0.3)
        total_masked = mask.sum().item()
        assert reconstructed.shape[0] == total_masked
        assert original.shape[0] == total_masked

    def test_reconstruction_dimension(self, tiny_encoder_config, tiny_encoder):
        x = torch.randn(4, 50, 8)
        reconstructed, _, _ = tiny_encoder(x)
        patch_dim = tiny_encoder_config.patch_size * tiny_encoder_config.num_features
        assert reconstructed.shape[1] == patch_dim

    def test_encode_produces_cls_embedding(self, tiny_encoder):
        x = torch.randn(4, 50, 8)
        emb = tiny_encoder.encode(x)
        assert emb.shape == (4, 16)  # (batch, d_model)

    def test_gradient_flow(self, tiny_encoder):
        x = torch.randn(4, 50, 8, requires_grad=True)
        reconstructed, original, mask = tiny_encoder(x)
        loss = torch.nn.functional.mse_loss(reconstructed, original.detach())
        loss.backward()
        assert x.grad is not None

    def test_save_and_load(self, tiny_encoder, tmp_path):
        save_path = tmp_path / "encoder.pt"
        tiny_encoder.save(save_path)

        loaded = MaskedTimeSeriesEncoder.load(save_path)
        tiny_encoder.eval()
        loaded.eval()

        x = torch.randn(2, 50, 8)
        emb_orig = tiny_encoder.encode(x)
        emb_loaded = loaded.encode(x)
        assert torch.allclose(emb_orig, emb_loaded)

    def test_get_encoder_state_dict(self, tiny_encoder):
        state = tiny_encoder.get_encoder_state_dict()
        assert isinstance(state, dict)
        assert len(state) > 0

    def test_different_sequence_lengths(self, tiny_encoder):
        tiny_encoder.eval()
        for seq_len in [25, 50, 100]:
            x = torch.randn(2, seq_len, 8)
            emb = tiny_encoder.encode(x)
            assert emb.shape == (2, 16)

    def test_loss_decreases(self, tiny_encoder):
        """MSE reconstruction loss should decrease with training."""
        torch.manual_seed(42)
        optimizer = torch.optim.Adam(tiny_encoder.parameters(), lr=1e-3)
        x = torch.randn(8, 50, 8)

        losses = []
        for _ in range(50):
            reconstructed, original, mask = tiny_encoder(x, mask_ratio=0.3)
            loss = torch.nn.functional.mse_loss(reconstructed, original)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Compare average of first 5 vs last 5 to smooth noise
        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last < avg_first, f"Loss did not decrease: {avg_first:.4f} -> {avg_last:.4f}"

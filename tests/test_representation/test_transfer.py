"""Tests for transfer learning utility."""

from __future__ import annotations

import torch
import pytest

from quant_lab.representation.masked_encoder import MaskedTimeSeriesEncoder, MaskedEncoderConfig
from quant_lab.models.transformer.model import TransformerForecaster, TransformerConfig
from quant_lab.representation.transfer import transfer_encoder_weights, load_and_transfer


class TestTransferEncoderWeights:
    def _make_models(self, d_model=32, num_features=10, patch_size=5):
        enc_config = MaskedEncoderConfig(
            num_features=num_features,
            patch_size=patch_size,
            d_model=d_model,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=64,
        )
        pretrained = MaskedTimeSeriesEncoder(enc_config)

        fc_config = TransformerConfig(
            num_features=num_features,
            d_model=d_model,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=64,
        )
        forecaster = TransformerForecaster(fc_config)

        return pretrained, forecaster

    def test_transfer_returns_keys(self):
        pretrained, forecaster = self._make_models()
        transferred, skipped = transfer_encoder_weights(pretrained, forecaster)
        assert len(transferred) > 0

    def test_transferred_weights_match(self):
        pretrained, forecaster = self._make_models()
        torch.manual_seed(42)
        # Re-init pretrained with known weights
        pretrained = MaskedTimeSeriesEncoder(pretrained.config)

        transfer_encoder_weights(pretrained, forecaster)

        # Check that shared layer weights match
        for key in ["cls_token", "final_norm.weight", "final_norm.bias"]:
            pre_val = pretrained.encoder.state_dict()[key]
            fc_val = forecaster.encoder.state_dict()[key]
            assert torch.allclose(pre_val, fc_val), f"Mismatch on {key}"

    def test_input_proj_skipped_when_different(self):
        # d_model != num_features -> input_proj shapes differ between pretrained
        # (d_model -> d_model) and forecaster (num_features -> d_model)
        pretrained, forecaster = self._make_models(d_model=32, num_features=10)
        _, skipped = transfer_encoder_weights(pretrained, forecaster)

        # input_proj has different input dimension so should be skipped
        proj_skipped = [k for k in skipped if "input_proj" in k]
        assert len(proj_skipped) > 0

    def test_freeze_encoder(self):
        pretrained, forecaster = self._make_models()
        transfer_encoder_weights(pretrained, forecaster, freeze_encoder=True)

        # Check that some parameters are frozen
        frozen = sum(1 for p in forecaster.encoder.parameters() if not p.requires_grad)
        assert frozen > 0

    def test_heads_still_trainable(self):
        pretrained, forecaster = self._make_models()
        transfer_encoder_weights(pretrained, forecaster, freeze_encoder=True)

        # Heads should still be trainable
        for name, param in forecaster.heads.named_parameters():
            assert param.requires_grad, f"Head param {name} should be trainable"

    def test_load_and_transfer(self, tmp_path):
        pretrained, forecaster = self._make_models()
        pretrained.save(tmp_path / "encoder.pt")

        transferred, skipped = load_and_transfer(
            tmp_path / "encoder.pt", forecaster
        )
        assert len(transferred) > 0

    def test_strict_mode_raises_on_mismatch(self):
        pretrained, forecaster = self._make_models(d_model=32, num_features=10)
        with pytest.raises(ValueError, match="Strict mode"):
            transfer_encoder_weights(pretrained, forecaster, strict=True)

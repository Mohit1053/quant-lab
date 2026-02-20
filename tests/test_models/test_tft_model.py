"""Tests for Temporal Fusion Transformer model."""

from __future__ import annotations

import torch
import pytest

from quant_lab.models.tft.model import TFTConfig, TFTForecaster, InterpretableMultiHeadAttention


class TestInterpretableMultiHeadAttention:
    def test_output_shapes(self):
        attn = InterpretableMultiHeadAttention(d_model=32, nhead=4, dropout=0.0)
        q = k = v = torch.randn(2, 10, 32)
        out, weights = attn(q, k, v)
        assert out.shape == (2, 10, 32)
        assert weights.shape == (2, 4, 10, 10)

    def test_attention_weights_sum_to_one(self):
        attn = InterpretableMultiHeadAttention(d_model=32, nhead=4, dropout=0.0)
        q = k = v = torch.randn(2, 10, 32)
        _, weights = attn(q, k, v)
        # Each query position should sum to 1 over keys
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flow(self):
        attn = InterpretableMultiHeadAttention(d_model=32, nhead=4, dropout=0.0)
        x = torch.randn(2, 10, 32, requires_grad=True)
        out, _ = attn(x, x, x)
        out.sum().backward()
        assert x.grad is not None


class TestTFTForecaster:
    @pytest.fixture
    def tiny_config(self):
        return TFTConfig(
            num_features=8,
            sequence_length=20,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            lstm_layers=1,
            lstm_hidden=16,
            dropout=0.0,
            grn_hidden=8,
        )

    @pytest.fixture
    def tiny_tft(self, tiny_config):
        return TFTForecaster(tiny_config)

    def test_forward_output_keys(self, tiny_tft):
        x = torch.randn(4, 20, 8)
        outputs = tiny_tft(x)
        assert "distribution" in outputs
        assert "direction" in outputs
        assert "volatility" in outputs
        assert "attention_weights" in outputs
        assert "feature_weights" in outputs

    def test_forward_shapes(self, tiny_tft, tiny_config):
        x = torch.randn(4, 20, 8)
        outputs = tiny_tft(x)

        # Distribution: Gaussian mean and log_var
        dist = outputs["distribution"]
        assert dist["mean"].shape == (4,)
        assert dist["log_var"].shape == (4,)

        # Direction: 3-class logits
        assert outputs["direction"].shape == (4, 3)

        # Volatility: positive scalar
        assert outputs["volatility"].shape == (4,)

        # Feature weights: (batch, num_features)
        assert outputs["feature_weights"].shape == (4, 8)

        # Attention weights: list of (B, nhead, T, T)
        assert len(outputs["attention_weights"]) == tiny_config.num_encoder_layers

    def test_predict_returns(self, tiny_tft):
        x = torch.randn(4, 20, 8)
        preds = tiny_tft.predict_returns(x)
        assert preds.shape == (4,)

    def test_feature_importance(self, tiny_tft):
        x = torch.randn(4, 20, 8)
        importance = tiny_tft.get_feature_importance(x)
        assert importance.shape == (4, 8)
        # Should sum to ~1 (it's an average of per-timestep softmax weights)
        sums = importance.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_count_parameters(self, tiny_tft):
        n_params = tiny_tft.count_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_student_t_distribution(self):
        config = TFTConfig(
            num_features=8, d_model=16, nhead=2, num_encoder_layers=1,
            lstm_layers=1, lstm_hidden=16, dropout=0.0, grn_hidden=8,
            distribution_type="student_t",
        )
        model = TFTForecaster(config)
        x = torch.randn(4, 20, 8)
        outputs = model(x)
        dist = outputs["distribution"]
        assert "loc" in dist
        assert "log_scale" in dist
        assert "log_df" in dist

    def test_no_volatility_head(self):
        config = TFTConfig(
            num_features=8, d_model=16, nhead=2, num_encoder_layers=1,
            lstm_layers=1, lstm_hidden=16, dropout=0.0, grn_hidden=8,
            volatility_enabled=False,
        )
        model = TFTForecaster(config)
        x = torch.randn(4, 20, 8)
        outputs = model(x)
        assert "volatility" not in outputs

    def test_gradient_flow_end_to_end(self, tiny_tft):
        x = torch.randn(4, 20, 8, requires_grad=True)
        outputs = tiny_tft(x)
        loss = outputs["distribution"]["mean"].sum()
        loss.backward()
        assert x.grad is not None

    def test_save_and_load(self, tiny_tft, tmp_path):
        save_path = tmp_path / "tft.pt"
        tiny_tft.save(save_path)

        loaded = TFTForecaster.load(save_path)
        tiny_tft.eval()
        loaded.eval()

        x = torch.randn(2, 20, 8)
        pred_orig = tiny_tft.predict_returns(x)
        pred_loaded = loaded.predict_returns(x)
        assert torch.allclose(pred_orig, pred_loaded)

    def test_different_sequence_lengths(self, tiny_tft):
        tiny_tft.eval()
        for seq_len in [10, 20, 50]:
            x = torch.randn(2, seq_len, 8)
            preds = tiny_tft.predict_returns(x)
            assert preds.shape == (2,)

    def test_loss_decreases(self, tiny_tft):
        """TFT should be able to overfit on a small batch."""
        from quant_lab.models.transformer.model import MultiTaskLoss

        torch.manual_seed(42)
        # Use TransformerConfig-compatible loss (just needs the weight attributes)
        from dataclasses import dataclass

        @dataclass
        class LossConfig:
            distribution_weight: float = 1.0
            direction_weight: float = 0.3
            volatility_weight: float = 0.3
            direction_threshold: float = 0.005
            volatility_enabled: bool = True
            distribution_type: str = "gaussian"

        loss_fn = MultiTaskLoss(LossConfig())
        optimizer = torch.optim.Adam(tiny_tft.parameters(), lr=1e-3)

        x = torch.randn(8, 20, 8)
        targets = {"returns": torch.randn(8) * 0.01}

        losses = []
        for _ in range(30):
            outputs = tiny_tft(x)
            loss, _ = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last < avg_first, f"Loss did not decrease: {avg_first:.4f} -> {avg_last:.4f}"

    def test_multi_layer_attention(self):
        config = TFTConfig(
            num_features=8, d_model=16, nhead=2, num_encoder_layers=3,
            lstm_layers=1, lstm_hidden=16, dropout=0.0, grn_hidden=8,
        )
        model = TFTForecaster(config)
        x = torch.randn(2, 20, 8)
        outputs = model(x)
        assert len(outputs["attention_weights"]) == 3

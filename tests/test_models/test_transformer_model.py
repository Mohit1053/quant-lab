"""Tests for the full transformer forecaster model."""

from __future__ import annotations

import torch
import pytest

from quant_lab.models.transformer.model import (
    TransformerForecaster,
    TransformerConfig,
    MultiTaskLoss,
)


@pytest.fixture
def tiny_transformer_config():
    """Small config for fast testing."""
    return TransformerConfig(
        num_features=8,
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        distribution_type="gaussian",
        direction_num_classes=3,
        direction_threshold=0.005,
        volatility_enabled=True,
        distribution_weight=1.0,
        direction_weight=0.3,
        volatility_weight=0.3,
    )


@pytest.fixture
def tiny_model(tiny_transformer_config):
    return TransformerForecaster(tiny_transformer_config)


class TestTransformerForecaster:
    def test_forward_output_keys(self, tiny_model):
        x = torch.randn(4, 20, 8)
        out = tiny_model(x)
        assert "distribution" in out
        assert "direction" in out
        assert "volatility" in out

    def test_forward_shapes(self, tiny_model):
        x = torch.randn(4, 20, 8)
        out = tiny_model(x)
        assert out["distribution"]["mean"].shape == (4,)
        assert out["distribution"]["log_var"].shape == (4,)
        assert out["direction"].shape == (4, 3)
        assert out["volatility"].shape == (4,)

    def test_predict_returns(self, tiny_model):
        x = torch.randn(4, 20, 8)
        preds = tiny_model.predict_returns(x)
        assert preds.shape == (4,)

    def test_count_parameters(self, tiny_model):
        count = tiny_model.count_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_student_t_distribution(self):
        config = TransformerConfig(
            num_features=8, d_model=16, nhead=2, num_encoder_layers=1,
            dim_feedforward=32, dropout=0.0, distribution_type="student_t",
        )
        model = TransformerForecaster(config)
        x = torch.randn(4, 20, 8)
        out = model(x)
        assert "loc" in out["distribution"]
        assert "log_scale" in out["distribution"]
        assert "log_df" in out["distribution"]

    def test_no_volatility_head(self):
        config = TransformerConfig(
            num_features=8, d_model=16, nhead=2, num_encoder_layers=1,
            dim_feedforward=32, dropout=0.0, volatility_enabled=False,
        )
        model = TransformerForecaster(config)
        x = torch.randn(4, 20, 8)
        out = model(x)
        assert "volatility" not in out

    def test_gradient_flow_end_to_end(self, tiny_model):
        x = torch.randn(4, 20, 8, requires_grad=True)
        out = tiny_model(x)
        loss = out["distribution"]["mean"].sum() + out["direction"].sum() + out["volatility"].sum()
        loss.backward()
        assert x.grad is not None
        for p in tiny_model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_save_and_load(self, tiny_model, tmp_path):
        save_path = tmp_path / "model.pt"
        tiny_model.save(save_path)

        loaded = TransformerForecaster.load(save_path)

        tiny_model.eval()
        loaded.eval()
        x = torch.randn(2, 10, 8)
        out_original = tiny_model(x)
        out_loaded = loaded(x)
        assert torch.allclose(
            out_original["distribution"]["mean"],
            out_loaded["distribution"]["mean"],
        )

    def test_different_sequence_lengths(self, tiny_model):
        tiny_model.eval()
        for seq_len in [5, 20, 63, 128]:
            x = torch.randn(2, seq_len, 8)
            out = tiny_model(x)
            assert out["distribution"]["mean"].shape == (2,)


class TestMultiTaskLoss:
    def test_gaussian_loss_computation(self, tiny_transformer_config):
        loss_fn = MultiTaskLoss(tiny_transformer_config)
        outputs = {
            "distribution": {"mean": torch.randn(4), "log_var": torch.randn(4)},
            "direction": torch.randn(4, 3),
            "volatility": torch.abs(torch.randn(4)),
        }
        targets = {"returns": torch.randn(4)}
        total_loss, loss_dict = loss_fn(outputs, targets)

        assert torch.isfinite(total_loss)
        assert "distribution" in loss_dict
        assert "direction" in loss_dict
        assert "volatility" in loss_dict
        assert "total" in loss_dict

    def test_direction_labels(self, tiny_transformer_config):
        loss_fn = MultiTaskLoss(tiny_transformer_config)
        returns = torch.tensor([0.01, -0.01, 0.001, -0.001, 0.0])
        labels = loss_fn._returns_to_direction(returns)
        assert labels[0] == 2  # up (0.01 > 0.005)
        assert labels[1] == 0  # down (-0.01 < -0.005)
        assert labels[2] == 1  # flat
        assert labels[3] == 1  # flat
        assert labels[4] == 1  # flat

    def test_loss_decreases_with_training(self, tiny_model, tiny_transformer_config):
        loss_fn = MultiTaskLoss(tiny_transformer_config)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        x = torch.randn(8, 20, 8)
        targets = {"returns": torch.randn(8) * 0.01}

        losses = []
        for _ in range(20):
            outputs = tiny_model(x)
            loss, _ = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_student_t_loss(self):
        config = TransformerConfig(
            num_features=8, d_model=16, nhead=2, num_encoder_layers=1,
            dim_feedforward=32, dropout=0.0, distribution_type="student_t",
        )
        loss_fn = MultiTaskLoss(config)
        outputs = {
            "distribution": {
                "loc": torch.randn(4),
                "log_scale": torch.randn(4),
                "log_df": torch.randn(4),
            },
            "direction": torch.randn(4, 3),
            "volatility": torch.abs(torch.randn(4)),
        }
        targets = {"returns": torch.randn(4)}
        total_loss, loss_dict = loss_fn(outputs, targets)
        assert torch.isfinite(total_loss)

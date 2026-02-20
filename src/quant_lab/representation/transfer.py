"""Transfer learning: load pre-trained encoder weights into forecasting models."""

from __future__ import annotations

from pathlib import Path

import torch
import structlog

from quant_lab.representation.masked_encoder import MaskedTimeSeriesEncoder
from quant_lab.models.transformer.model import TransformerForecaster

logger = structlog.get_logger(__name__)


def transfer_encoder_weights(
    pretrained: MaskedTimeSeriesEncoder,
    forecaster: TransformerForecaster,
    freeze_encoder: bool = False,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Transfer pre-trained encoder weights to a forecasting model.

    Both models share the same TransformerEncoder architecture (layers,
    cls_token, positional encoding, final_norm). The input_proj layer
    differs (pre-trained uses d_model->d_model after tokenization, while
    forecaster uses num_features->d_model) so it is skipped by default.

    Args:
        pretrained: Fitted MaskedTimeSeriesEncoder.
        forecaster: Target TransformerForecaster to initialize.
        freeze_encoder: If True, freeze transferred encoder layers
            (only train heads + input_proj).
        strict: If True, require all encoder keys to match exactly.

    Returns:
        (transferred_keys, skipped_keys) for logging/verification.
    """
    pretrained_state = pretrained.encoder.state_dict()
    forecaster_state = forecaster.encoder.state_dict()

    transferred = []
    skipped = []

    for key, value in pretrained_state.items():
        if key in forecaster_state:
            target_shape = forecaster_state[key].shape
            if value.shape == target_shape:
                forecaster_state[key] = value
                transferred.append(key)
            else:
                skipped.append(f"{key} (shape mismatch: {value.shape} vs {target_shape})")
        else:
            skipped.append(f"{key} (not in forecaster)")

    if strict and skipped:
        raise ValueError(f"Strict mode: skipped keys: {skipped}")

    forecaster.encoder.load_state_dict(forecaster_state)

    if freeze_encoder:
        _freeze_transferred_params(forecaster.encoder, transferred)

    logger.info(
        "transfer_complete",
        transferred=len(transferred),
        skipped=len(skipped),
        frozen=freeze_encoder,
    )

    return transferred, skipped


def _freeze_transferred_params(encoder: torch.nn.Module, transferred_keys: list[str]) -> None:
    """Freeze parameters that were transferred (keep input_proj trainable)."""
    frozen_count = 0
    for name, param in encoder.named_parameters():
        if name in transferred_keys:
            param.requires_grad = False
            frozen_count += 1
    logger.info("params_frozen", count=frozen_count)


def load_and_transfer(
    pretrained_path: str | Path,
    forecaster: TransformerForecaster,
    freeze_encoder: bool = False,
    map_location: str = "cpu",
) -> tuple[list[str], list[str]]:
    """Load a pre-trained encoder from disk and transfer weights.

    Convenience function combining load + transfer in one call.

    Args:
        pretrained_path: Path to saved MaskedTimeSeriesEncoder checkpoint.
        forecaster: Target forecasting model.
        freeze_encoder: Whether to freeze transferred layers.
        map_location: Device mapping for torch.load.

    Returns:
        (transferred_keys, skipped_keys)
    """
    pretrained = MaskedTimeSeriesEncoder.load(pretrained_path, map_location=map_location)
    return transfer_encoder_weights(pretrained, forecaster, freeze_encoder=freeze_encoder)

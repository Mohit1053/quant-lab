"""BERT-like masked time-series encoder for self-supervised pre-training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from quant_lab.representation.tokenizer import PatchTokenizer
from quant_lab.models.transformer.encoder import TransformerEncoder


@dataclass
class MaskedEncoderConfig:
    """Configuration for masked time-series encoder."""

    num_features: int = 10
    patch_size: int = 5
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    mask_ratio: float = 0.15


class MaskedTimeSeriesEncoder(nn.Module):
    """Self-supervised masked encoder for financial time series.

    Inspired by BERT's masked language modeling, adapted for continuous
    time series. Randomly masks patches of the input sequence and trains
    the model to reconstruct the original values.

    Pre-training pipeline:
        1. Tokenize input into patches (PatchTokenizer)
        2. Randomly mask a fraction of patches (replace with learnable [MASK] token)
        3. Encode all tokens (including masked) with transformer encoder
        4. Reconstruct original patch values at masked positions
        5. Loss = MSE between reconstructed and original (masked positions only)

    After pre-training:
        - Extract the encoder (without reconstruction head) for transfer learning
        - Use CLS token output as market embedding
    """

    def __init__(self, config: MaskedEncoderConfig):
        super().__init__()
        self.config = config

        # Patch tokenizer
        self.tokenizer = PatchTokenizer(
            num_features=config.num_features,
            patch_size=config.patch_size,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Transformer encoder (reuses the same architecture as forecasting model)
        # We pass d_model as num_features since tokenizer already projects to d_model
        self.encoder = TransformerEncoder(
            num_features=config.d_model,  # Input is already projected patch tokens
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )

        # Reconstruction head: d_model -> patch_size * num_features
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, config.patch_size * config.num_features),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for pre-training.

        Args:
            x: (batch, seq_len, num_features) raw feature sequences
            mask_ratio: fraction of patches to mask (default: config value)

        Returns:
            reconstructed: (batch, num_masked, patch_size * num_features)
            original_patches: (batch, num_masked, patch_size * num_features) ground truth
            mask: (batch, num_patches) boolean mask (True = masked)
        """
        if mask_ratio is None:
            mask_ratio = self.config.mask_ratio

        B, T, F = x.shape

        # Tokenize into patches
        tokens = self.tokenizer(x)  # (B, num_patches, d_model)
        num_patches = tokens.size(1)

        # Get original patch values for loss computation
        trimmed_len = num_patches * self.config.patch_size
        x_trimmed = x[:, :trimmed_len, :]
        original_patches = x_trimmed.reshape(
            B, num_patches, self.config.patch_size * F
        )  # (B, num_patches, patch_size * F)

        # Random masking
        mask = self._generate_mask(B, num_patches, mask_ratio, x.device)  # (B, num_patches)

        # Replace masked tokens with [MASK] token
        masked_tokens = tokens.clone()
        mask_expanded = mask.unsqueeze(-1).expand_as(tokens)
        masked_tokens[mask_expanded] = self.mask_token.expand(B, num_patches, -1)[mask_expanded]

        # Encode
        encoded = self.encoder(masked_tokens)  # (B, num_patches + 1, d_model) with CLS

        # Extract encoded representations at masked positions (skip CLS at position 0)
        encoded_patches = encoded[:, 1:, :]  # (B, num_patches, d_model)

        # Reconstruct only masked positions
        masked_encoded = encoded_patches[mask]  # (total_masked, d_model)
        reconstructed = self.reconstruction_head(masked_encoded)  # (total_masked, patch_size * F)

        # Get original values at masked positions
        original_masked = original_patches[mask]  # (total_masked, patch_size * F)

        return reconstructed, original_masked, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embedding (for transfer learning / embedding extraction).

        Args:
            x: (batch, seq_len, num_features)

        Returns:
            (batch, d_model) CLS token embedding
        """
        tokens = self.tokenizer(x)
        encoded = self.encoder(tokens)
        return encoded[:, 0, :]  # CLS token

    def get_encoder_state_dict(self) -> dict:
        """Get the encoder's state dict for transfer to forecasting model."""
        return self.encoder.state_dict()

    def _generate_mask(
        self, batch_size: int, num_patches: int, mask_ratio: float, device: torch.device
    ) -> torch.Tensor:
        """Generate random mask for patches.

        Returns:
            (batch_size, num_patches) boolean tensor (True = masked)
        """
        num_mask = max(1, int(num_patches * mask_ratio))
        # Random permutation per batch
        noise = torch.rand(batch_size, num_patches, device=device)
        # Sort and take top-k as masked
        _, indices = noise.topk(num_mask, dim=1)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, indices, True)
        return mask

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": self.config, "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> MaskedTimeSeriesEncoder:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model

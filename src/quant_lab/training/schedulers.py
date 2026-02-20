"""Learning rate schedulers: cosine warmup, linear warmup."""

from __future__ import annotations

import math

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Cosine annealing with linear warmup.

    Learning rate schedule:
    1. Linear warmup from 0 to lr over warmup_steps
    2. Cosine decay from lr to min_lr_ratio * lr over remaining steps

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak LR.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / max(1, warmup_steps)
        # Cosine decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def linear_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
) -> LambdaLR:
    """Linear warmup only (constant LR after warmup).

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)

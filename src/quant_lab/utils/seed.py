"""Reproducibility utilities - seed everything."""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set seed for Python, NumPy, PyTorch, and CUDA for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

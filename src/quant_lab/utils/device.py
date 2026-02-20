"""GPU/CPU device management and mixed precision setup."""

from __future__ import annotations

import os

import torch
import structlog

logger = structlog.get_logger(__name__)


def get_device(override: str | None = None) -> torch.device:
    """Detect and return the best available device.

    Priority: override env var > CUDA > CPU.
    """
    if override:
        device = torch.device(override)
    elif env_device := os.environ.get("QUANT_LAB_DEVICE"):
        device = torch.device(env_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        logger.info("using_gpu", name=gpu_name, memory_gb=f"{gpu_mem:.1f}")
    else:
        logger.info("using_cpu")

    return device


def get_dtype(prefer_bf16: bool = True) -> torch.dtype:
    """Return the best mixed-precision dtype for the current hardware.

    BF16 is preferred on Ampere+ (RTX 30xx/40xx, A100, H100).
    Falls back to FP16 if BF16 is unavailable.
    """
    if not torch.cuda.is_available():
        return torch.float32

    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def get_device_info() -> dict:
    """Return a dict of device information for experiment logging."""
    info = {
        "device_type": "cpu",
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update({
            "device_type": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            ),
            "cuda_version": torch.version.cuda or "N/A",
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })

    info["pytorch_version"] = torch.__version__
    return info

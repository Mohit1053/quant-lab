"""Tests for utility modules: seed, device, timer."""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import pytest

from quant_lab.utils.seed import set_global_seed
from quant_lab.utils.device import get_device, get_dtype, get_device_info
from quant_lab.utils.timer import timer, timed


class TestSetGlobalSeed:
    def test_numpy_deterministic(self):
        set_global_seed(123)
        a = np.random.randn(10)
        set_global_seed(123)
        b = np.random.randn(10)
        np.testing.assert_array_equal(a, b)

    def test_python_random_deterministic(self):
        set_global_seed(123)
        a = random.random()
        set_global_seed(123)
        b = random.random()
        assert a == b

    def test_torch_deterministic(self):
        set_global_seed(123)
        a = torch.randn(10)
        set_global_seed(123)
        b = torch.randn(10)
        torch.testing.assert_close(a, b)

    def test_different_seeds_differ(self):
        set_global_seed(1)
        a = np.random.randn(10)
        set_global_seed(2)
        b = np.random.randn(10)
        assert not np.array_equal(a, b)


class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_override(self):
        device = get_device(override="cpu")
        assert device.type == "cpu"

    def test_device_type_valid(self):
        device = get_device()
        assert device.type in ("cpu", "cuda")


class TestGetDtype:
    def test_returns_dtype(self):
        dtype = get_dtype()
        assert dtype in (torch.float16, torch.bfloat16, torch.float32)

    def test_cpu_returns_float32(self):
        # If CUDA not available, should return float32
        if not torch.cuda.is_available():
            assert get_dtype() == torch.float32


class TestGetDeviceInfo:
    def test_returns_dict(self):
        info = get_device_info()
        assert isinstance(info, dict)
        assert "device_type" in info
        assert "pytorch_version" in info
        assert "cuda_available" in info


class TestTimer:
    def test_timer_context_manager(self):
        with timer("test_block"):
            time.sleep(0.01)
        # No error means success

    def test_timed_decorator(self):
        @timed
        def slow_func():
            time.sleep(0.01)
            return 42

        result = slow_func()
        assert result == 42

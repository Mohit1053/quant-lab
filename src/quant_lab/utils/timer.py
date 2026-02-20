"""Performance timing utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


@contextmanager
def timer(label: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info("timer", label=label, elapsed_seconds=round(elapsed, 3))


def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("timer", function=func.__name__, elapsed_seconds=round(elapsed, 3))
        return result

    return wrapper

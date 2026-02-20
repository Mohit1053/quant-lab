"""Feature registry - decorator-based feature function registration."""

from __future__ import annotations

from typing import Callable

import pandas as pd

# Global registry: name -> (function, description)
FEATURE_REGISTRY: dict[str, tuple[Callable, str]] = {}


def register_feature(name: str, description: str = ""):
    """Decorator to register a feature computation function.

    The decorated function must have signature:
        func(df: pd.DataFrame, **kwargs) -> pd.DataFrame

    It receives the full cleaned DataFrame and must return a DataFrame
    with new feature columns added (MultiIndex: date, ticker).
    """

    def decorator(func: Callable) -> Callable:
        FEATURE_REGISTRY[name] = (func, description)
        return func

    return decorator


def get_feature_func(name: str) -> Callable:
    """Get a registered feature function by name."""
    if name not in FEATURE_REGISTRY:
        available = ", ".join(sorted(FEATURE_REGISTRY.keys()))
        raise ValueError(f"Unknown feature '{name}'. Available: {available}")
    return FEATURE_REGISTRY[name][0]


def list_features() -> list[dict[str, str]]:
    """List all registered features with their descriptions."""
    return [
        {"name": name, "description": desc}
        for name, (_, desc) in sorted(FEATURE_REGISTRY.items())
    ]

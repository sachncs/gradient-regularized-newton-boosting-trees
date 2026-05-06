"""Utility helpers for norms, candidate split thresholds, and training history."""

from typing import Optional

import numpy as np


def empirical_norm(v: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Compute the empirical L^2 norm of a vector.

    Without weights this is ||v|| / sqrt(N), i.e. the root-mean-square.
    With weights it is sqrt(sum(weights * v^2)).

    Args:
        v: Vector of shape (n,) or higher (flattened).
        weights: Optional element-wise non-negative weights. Must match
            the flattened size of v.

    Returns:
        Scalar norm value.

    Raises:
        TypeError: If v is not a NumPy array.
        ValueError: If v is empty, contains NaN/Inf, or weights have wrong size.
    """
    if not isinstance(v, np.ndarray):
        raise TypeError(f"v must be a numpy.ndarray, got {type(v).__name__}")
    if v.size == 0:
        raise ValueError("Input vector must not be empty.")
    if np.any(np.isnan(v)):
        raise ValueError("Input vector contains NaN values.")
    if np.any(np.isinf(v)):
        raise ValueError("Input vector contains infinite values.")

    if weights is not None:
        if not isinstance(weights, np.ndarray):
            raise TypeError(
                f"weights must be a numpy.ndarray, got {type(weights).__name__}"
            )
        if weights.shape != v.shape:
            raise ValueError(
                f"weights shape {weights.shape} does not match v shape {v.shape}"
            )
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        if np.any(np.isnan(weights)):
            raise ValueError("weights contain NaN values.")
        if np.any(np.isinf(weights)):
            raise ValueError("weights contain infinite values.")
        return float(np.sqrt(np.sum(weights * v**2)))

    return float(np.linalg.norm(v) / np.sqrt(v.size))


def unique_thresholds(x: np.ndarray) -> np.ndarray:
    """Return midpoints between sorted unique values of a 1-D array.

    These are standard candidate thresholds for greedy axis-aligned splits.

    Args:
        x: 1-D feature column.

    Returns:
        1-D array of threshold candidates; empty if fewer than 2 unique values.

    Raises:
        TypeError: If x is not a NumPy array.
        ValueError: If x is not 1-D or contains NaN/Inf.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be a numpy.ndarray, got {type(x).__name__}")
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("x must not be empty.")
    if np.any(np.isnan(x)):
        raise ValueError("x contains NaN values.")
    if np.any(np.isinf(x)):
        raise ValueError("x contains infinite values.")

    vals = np.unique(x)
    if vals.size <= 1:
        return np.asarray(np.array([], dtype=float), dtype=float)
    return np.asarray((vals[:-1] + vals[1:]) / 2.0, dtype=float)


class History:
    """Simple key-value logger for scalar training metrics.

    Thread-safe for single-threaded use (no external locking required).
    All values are coerced to Python float for serialization safety.
    """

    def __init__(self) -> None:
        """Initialize empty history."""
        self._data: dict[str, list[float]] = {}

    def log(self, key: str, value: float) -> None:
        """Append a scalar value to a named series.

        Args:
            key: Metric name. Must be a non-empty string.
            value: Scalar value to record. Will be cast to float.

        Raises:
            TypeError: If key is not a string.
            ValueError: If key is empty or value is not finite.
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key).__name__}")
        if not key:
            raise ValueError("key must not be empty.")
        val_float = float(value)
        if not np.isfinite(val_float):
            raise ValueError(f"value must be finite, got {value}")
        self._data.setdefault(key, []).append(val_float)

    def get(self, key: str) -> list[float]:
        """Retrieve a recorded series.

        Args:
            key: Metric name.

        Returns:
            List of recorded scalars; empty if key unknown.

        Raises:
            TypeError: If key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key).__name__}")
        return self._data.get(key, [])

    def keys(self) -> list[str]:
        """Return all recorded metric names.

        Returns:
            Sorted list of metric names.
        """
        return sorted(self._data.keys())

    def as_dict(self) -> dict[str, list[float]]:
        """Return a shallow copy of the full history as a dictionary.

        Returns:
            Dictionary mapping metric names to lists of values.
        """
        return {k: list(v) for k, v in self._data.items()}

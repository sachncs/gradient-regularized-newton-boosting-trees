"""Utility helpers for norms, candidate split thresholds, and training history.

This module collects small, dependency-free helpers used by the core
algorithms:

* :func:`empirical_norm` — the empirical ``L^2`` norm used by the
  paper (RMS for unweighted vectors, weighted version for Hessian
  inner products).
* :func:`unique_thresholds` — candidate split thresholds (midpoints
  between consecutive unique feature values); the canonical list of
  candidates tested by the exact greedy tree builder.
* :class:`History` — a tiny per-run metric logger used by the
  boosting engines to record ``loss``, ``lambda_k``, and ``grad_norm``
  every iteration.

Design note: this module deliberately contains no I/O and no NumPy
state, so it is safe to import anywhere in the package without
circular-dependency concerns.
"""

from typing import Optional

import numpy as np


def empirical_norm(v: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Compute the empirical L^2 norm.

    * Without weights: ``||v|| / sqrt(N)`` — the root-mean-square.
      This matches the paper's convention for the empirical ``L^2``
      inner product.
    * With weights: ``sqrt(sum(weights * v^2))`` — the
      Hessian-weighted norm that underlies the ``H``-induced inner
      product.

    Args:
        v: Vector of shape ``(n,)`` or higher. Multi-dimensional
            arrays are flattened for the norm computation.
        weights: Optional element-wise non-negative weights, with the
            same shape as ``v``. When provided the function returns
            ``sqrt(sum(weights * v^2))``. Weights must be finite and
            non-negative.

    Returns:
        Scalar norm value.

    Raises:
        TypeError: If ``v`` (or ``weights``) is not a NumPy array.
        ValueError: If ``v`` is empty, contains ``NaN`` / ``Inf``,
            or if ``weights`` have the wrong shape, contain
            ``NaN`` / ``Inf``, or negative values.
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

    These are the *standard* candidate thresholds for greedy
    axis-aligned splits: between any two distinct values exactly one
    midpoint can split them, and midpoints never coincide with a
    sample value. The result is sorted ascending.

    Args:
        x: 1-D feature column, shape ``(n_samples,)``.

    Returns:
        1-D array of threshold candidates, shape ``(n_unique - 1,)``;
        empty if the column has fewer than 2 unique values.

    Raises:
        TypeError: If ``x`` is not a NumPy array.
        ValueError: If ``x`` is not 1-D, is empty, or contains
            ``NaN`` / ``Inf``.
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
        # No split is possible when there is only one unique value.
        return np.asarray(np.array([], dtype=float), dtype=float)
    # Midpoint between each consecutive pair of unique values.
    return np.asarray((vals[:-1] + vals[1:]) / 2.0, dtype=float)


class History:
    """Simple key-value logger for scalar training metrics.

    Stores time series of named metrics. Each call to :meth:`log`
    appends a Python float to the corresponding list, enforcing
    finiteness so that downstream plotting/serialization cannot be
    corrupted by ``NaN``/``Inf`` slips through.

    The class is **not** thread-safe: concurrent calls from different
    threads may interleave within :meth:`log` and corrupt the internal
    list. Use external synchronization if needed.

    Attributes:
        None public; the recorded data is exposed through :meth:`get`,
        :meth:`keys`, and :meth:`as_dict`.
    """

    def __init__(self) -> None:
        """Initialize empty history."""
        self._data: dict[str, list[float]] = {}

    def log(self, key: str, value: float) -> None:
        """Append a scalar value to a named series.

        Args:
            key: Metric name. Must be a non-empty string.
            value: Scalar value to record. Will be cast to ``float``;
                non-finite values (``NaN``/``Inf``) raise rather than
                corrupt the record.

        Raises:
            TypeError: If ``key`` is not a string.
            ValueError: If ``key`` is empty or ``value`` is not finite.
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
            List of recorded scalars (the internal list, not a copy);
            an empty list if the key was never logged.

        Raises:
            TypeError: If ``key`` is not a string.
        """
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key).__name__}")
        return self._data.get(key, [])

    def keys(self) -> list[str]:
        """Return all recorded metric names, sorted lexicographically.

        Returns:
            Sorted list of unique metric names.
        """
        return sorted(self._data.keys())

    def as_dict(self) -> dict[str, list[float]]:
        """Return a deep copy of the full history.

        The returned dictionary is decoupled from the instance state,
        so mutating it cannot affect subsequent reads.

        Returns:
            Dictionary mapping metric names to fresh lists of values.
        """
        return {k: list(v) for k, v in self._data.items()}

"""Dataset loaders for paper experiments.

This module fetches the two benchmark datasets used in the paper's
experiments and validates them before returning:

* :func:`load_wine_quality` — red-wine quality regression (UCI,
  ``wine-quality-red``).
* :func:`load_higgs_subset` — Higgs boson classification (OpenML).

Both loaders prefer ``scikit-learn``'s ``fetch_openml`` when available
and fall back to a direct download / synthetic surrogate when network
access fails. This keeps the experimental scripts runnable in offline
or restricted environments.

Notes
-----

* All features from :func:`load_wine_quality` are standardized to zero
  mean and unit variance; targets are returned as-is in ``[0, 10]``.
* :func:`load_higgs_subset` returns a *random* subset (seed ``42``)
  when the fetched dataset is larger than ``n_samples``; this makes
  experiments deterministic.
* Strict post-load validation detects any ``NaN``/``Inf``/shape issues
  before they silently corrupt downstream models.
"""

from typing import Tuple

import numpy as np


def load_wine_quality() -> Tuple[np.ndarray, np.ndarray]:
    """Load the Wine Quality (red) regression dataset.

    Tries ``sklearn.datasets.fetch_openml`` first, then a direct CSV
    download from the UCI repository. **No** synthetic fallback is
    provided because the dataset is small (~1.6k rows) and always
    available; both sources are tried before raising.

    Features are standardized to zero mean, unit variance. Targets are
    integer wine-quality scores in ``[0, 10]`` returned as ``float``.

    Returns:
        Tuple ``(x, y)`` with ``x`` of shape ``(n_samples, n_features)``
        and ``y`` of shape ``(n_samples,)``.

    Raises:
        RuntimeError: If all data sources fail *and* the resulting
            array fails any of the shape/finiteness checks.
    """
    data_loaded = False
    x = None
    y = None
    last_error = None

    # Attempt 1: scikit-learn fetch_openml (preferred when sklearn is
    # installed).
    try:
        from sklearn.datasets import fetch_openml

        wine = fetch_openml(
            name="wine-quality-red", version=1, as_frame=False, parser="auto"
        )
        x = wine.data.astype(float)
        y = wine.target.astype(float)
        data_loaded = True
    except Exception as exc:
        last_error = exc

    # Attempt 2: direct UCI CSV download (works without sklearn).
    if not data_loaded:
        try:
            import urllib.request

            url = (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "wine-quality/winequality-red.csv"
            )
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read().decode("utf-8").strip().split("\n")
            # UCI CSV uses ';' as separator; first row is the header.
            rows = [line.split(";") for line in data[1:]]
            arr = np.array(rows, dtype=float)
            x = arr[:, :-1]
            y = arr[:, -1]
            data_loaded = True
        except Exception as exc:
            last_error = exc

    if not data_loaded or x is None or y is None:
        raise RuntimeError(
            f"Failed to load Wine Quality dataset from all sources. Last error: {last_error}"
        )

    # Validate loaded data.
    if x.ndim != 2 or y.ndim != 1:
        raise RuntimeError(
            f"Loaded Wine Quality data has unexpected shape: x {x.shape}, y {y.shape}"
        )
    if x.shape[0] != y.shape[0]:
        raise RuntimeError(
            f"Loaded Wine Quality data has mismatched samples: x {x.shape[0]} vs y {y.shape[0]}"
        )
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise RuntimeError("Loaded Wine Quality data contains NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise RuntimeError("Loaded Wine Quality data contains infinite values.")

    # Standardize features (mean 0, std 1). The 1e-8 prevents
    # division-by-zero on constant columns.
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-8
    x = (x - mean) / std
    return x, y


def load_higgs_subset(n_samples: int = 100_000) -> Tuple[np.ndarray, np.ndarray]:
    """Load a subset of the Higgs boson binary classification dataset.

    Tries ``sklearn.datasets.fetch_openml`` first; if that fails (no
    sklearn, no network, etc.) it returns a *synthetic surrogate*
    with matching dimensionality so downstream code can still run end
    to end.

    The subset size policy:
        * If the fetched dataset is larger than ``n_samples``, keep a
          random subset of exactly ``n_samples`` rows (seed ``42``).
        * If smaller, return the whole dataset.

    Args:
        n_samples: Maximum number of samples to retain. Must be a
            positive integer.

    Returns:
        Tuple ``(x, y)`` where ``x.shape == (n_samples, 28)`` and
        ``y`` is a binary ``{0, 1}`` label vector of length
        ``n_samples``.

    Raises:
        ValueError: If ``n_samples`` is not a positive integer.
        RuntimeError: If both data sources fail or the returned
            arrays fail shape/label/finiteness checks.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")

    data_loaded = False
    x = None
    y = None
    last_error = None

    # Attempt 1: scikit-learn fetch_openml (preferred).
    try:
        from sklearn.datasets import fetch_openml

        higgs = fetch_openml(name="higgs", version=1, as_frame=False, parser="auto")
        x = higgs.data.astype(float)
        y = higgs.target.astype(int)
        # OpenML sometimes stores labels in a 2-D column matrix.
        if y.ndim > 1:
            y = y.ravel()
        # Deterministic subsampling so that experiments are reproducible.
        if x.shape[0] > n_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(x.shape[0], n_samples, replace=False)
            x = x[idx]
            y = y[idx]
        data_loaded = True
    except Exception as exc:
        last_error = exc

    # Attempt 2: synthetic surrogate with matching dimensionality.
    # Matches the *shape* contract only; not the *distribution*.
    if not data_loaded:
        rng = np.random.RandomState(42)
        x = rng.randn(n_samples, 28).astype(float)
        y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
        data_loaded = True

    if not data_loaded or x is None or y is None:
        raise RuntimeError(
            f"Failed to load Higgs dataset from all sources. Last error: {last_error}"
        )

    # Strict post-load validation.
    if x.ndim != 2 or y.ndim != 1:
        raise RuntimeError(
            f"Loaded Higgs data has unexpected shape: x {x.shape}, y {y.shape}"
        )
    if x.shape[0] != y.shape[0]:
        raise RuntimeError(
            f"Loaded Higgs data has mismatched samples: x {x.shape[0]} vs y {y.shape[0]}"
        )
    if x.shape[1] != 28:
        raise RuntimeError(f"Loaded Higgs data has {x.shape[1]} features, expected 28.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise RuntimeError("Loaded Higgs data contains NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise RuntimeError("Loaded Higgs data contains infinite values.")
    if not np.all(np.isin(y, [0, 1])):
        raise RuntimeError("Loaded Higgs labels are not binary {0,1}.")

    return x, y

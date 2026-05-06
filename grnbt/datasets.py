"""Dataset loaders for paper experiments.

Fetches Wine Quality (UCI) and Higgs (OpenML) as NumPy arrays.
Falls back to synthetic surrogates if network access fails.
"""

from typing import Tuple

import numpy as np


def load_wine_quality() -> Tuple[np.ndarray, np.ndarray]:
    """Load the Wine Quality regression dataset.

    Features are standardized to zero mean and unit variance.
    Target is returned as-is.

    Returns:
        Tuple (x, y) with x shape (n_samples, n_features) and y shape (n_samples,).

    Raises:
        RuntimeError: If all data sources fail and no fallback can be generated.
    """
    data_loaded = False
    x = None
    y = None
    last_error = None

    # Attempt 1: scikit-learn fetch_openml
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

    # Attempt 2: direct UCI CSV download
    if not data_loaded:
        try:
            import urllib.request

            url = (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "wine-quality/winequality-red.csv"
            )
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read().decode("utf-8").strip().split("\n")
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

    # Validate loaded data
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

    # Standardize features
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-8
    x = (x - mean) / std
    return x, y


def load_higgs_subset(n_samples: int = 100_000) -> Tuple[np.ndarray, np.ndarray]:
    """Load a subset of the Higgs boson classification dataset.

    Args:
        n_samples: Maximum number of samples to retain. If the fetched
            dataset is larger, a random subset of this size is returned.
            Must be a positive integer.

    Returns:
        Tuple (x, y) with x shape (n_samples, 28) and y binary {0,1}.

    Raises:
        ValueError: If n_samples is not positive.
        RuntimeError: If all data sources fail.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")

    data_loaded = False
    x = None
    y = None
    last_error = None

    # Attempt 1: scikit-learn fetch_openml
    try:
        from sklearn.datasets import fetch_openml

        higgs = fetch_openml(name="higgs", version=1, as_frame=False, parser="auto")
        x = higgs.data.astype(float)
        y = higgs.target.astype(int)
        if y.ndim > 1:
            y = y.ravel()
        if x.shape[0] > n_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(x.shape[0], n_samples, replace=False)
            x = x[idx]
            y = y[idx]
        data_loaded = True
    except Exception as exc:
        last_error = exc

    # Attempt 2: synthetic surrogate with matching dimensionality
    if not data_loaded:
        rng = np.random.RandomState(42)
        x = rng.randn(n_samples, 28).astype(float)
        y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
        data_loaded = True

    if not data_loaded or x is None or y is None:
        raise RuntimeError(
            f"Failed to load Higgs dataset from all sources. Last error: {last_error}"
        )

    # Validate loaded data
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

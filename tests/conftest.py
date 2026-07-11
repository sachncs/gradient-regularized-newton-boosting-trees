"""Shared pytest fixtures for GRNBT tests.

These fixtures provide small synthetic datasets used across multiple
test modules. They are sized to keep the entire test suite under a
few seconds while still exercising non-trivial learning dynamics.
"""

import numpy as np
import pytest


@pytest.fixture
def synthetic_regression():
    """Small synthetic regression dataset.

    Generates ``n=64`` samples with ``d=4`` features using a fixed
    ``RandomState(42)``. The targets are a mild nonlinear function
    of the first two features plus light Gaussian noise.

    Returns:
        Tuple ``(x, y)`` with shapes ``(64, 4)`` and ``(64,)``.
    """
    rng = np.random.RandomState(42)
    n_samples, n_features = 64, 4
    x = rng.randn(n_samples, n_features)
    y = x[:, 0] + 0.5 * x[:, 1] ** 2 + rng.randn(n_samples) * 0.1
    return x, y


@pytest.fixture
def synthetic_binary():
    """Small synthetic binary classification dataset.

    Generates ``n=64`` samples with ``d=4`` features. The decision
    boundary is the half-space ``x[:, 0] > 0.5 * x[:, 1]``.

    Returns:
        Tuple ``(x, y)`` with shapes ``(64, 4)`` and ``(64,)``;
        ``y`` contains values in ``{0, 1}``.
    """
    rng = np.random.RandomState(42)
    n_samples, n_features = 64, 4
    x = rng.randn(n_samples, n_features)
    logits = x[:, 0] - 0.5 * x[:, 1]
    y = (logits > 0).astype(int)
    return x, y


@pytest.fixture
def synthetic_multiclass():
    """Small synthetic 3-class classification dataset.

    Generates ``n=64`` samples with ``d=4`` features whose labels are
    the argmax of a three-dimensional linear-logit function.

    Returns:
        Tuple ``(x, y, n_classes)`` with shapes ``(64, 4)``,
        ``(64,)`` and ``n_classes == 3``.
    """
    rng = np.random.RandomState(42)
    n_samples, n_features, n_classes = 64, 4, 3
    x = rng.randn(n_samples, n_features)
    logits = np.stack([x[:, 0], -x[:, 0] + 0.5 * x[:, 1], 0.1 * x[:, 2]], axis=1)
    y = np.argmax(logits, axis=1)
    return x, y, n_classes

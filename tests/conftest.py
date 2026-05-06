"""Shared pytest fixtures for GRNBT tests."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_regression():
    """Small synthetic regression dataset."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 64, 4
    x = rng.randn(n_samples, n_features)
    y = x[:, 0] + 0.5 * x[:, 1] ** 2 + rng.randn(n_samples) * 0.1
    return x, y


@pytest.fixture
def synthetic_binary():
    """Small synthetic binary classification dataset."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 64, 4
    x = rng.randn(n_samples, n_features)
    logits = x[:, 0] - 0.5 * x[:, 1]
    y = (logits > 0).astype(int)
    return x, y


@pytest.fixture
def synthetic_multiclass():
    """Small synthetic 3-class classification dataset."""
    rng = np.random.RandomState(42)
    n_samples, n_features, n_classes = 64, 4, 3
    x = rng.randn(n_samples, n_features)
    logits = np.stack([x[:, 0], -x[:, 0] + 0.5 * x[:, 1], 0.1 * x[:, 2]], axis=1)
    y = np.argmax(logits, axis=1)
    return x, y, n_classes

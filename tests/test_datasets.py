"""Tests for dataset loaders."""

import numpy as np
import pytest

from grnbt.datasets import load_higgs_subset, load_wine_quality


def test_wine_quality_shape():
    """Wine Quality must load with expected dimensions."""
    x, y = load_wine_quality()
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] > 0
    assert x.shape[1] > 0


def test_wine_quality_standardized():
    """Wine Quality features must have zero mean and unit variance."""
    x, y = load_wine_quality()
    np.testing.assert_allclose(np.mean(x, axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(x, axis=0), 1.0, atol=1e-4)


def test_wine_quality_no_nan():
    """Wine Quality must not contain NaN or Inf."""
    x, y = load_wine_quality()
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(x))
    assert not np.any(np.isinf(y))


def test_higgs_shape():
    """Higgs subset must load with expected dimensions."""
    x, y = load_higgs_subset(n_samples=1000)
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == 1000
    assert x.shape[1] == 28
    assert x.shape[0] == y.shape[0]


def test_higgs_binary_labels():
    """Higgs labels must be binary {0, 1}."""
    x, y = load_higgs_subset(n_samples=500)
    assert np.all(np.isin(y, [0, 1]))


def test_higgs_no_nan():
    """Higgs must not contain NaN or Inf."""
    x, y = load_higgs_subset(n_samples=500)
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(x))
    assert not np.any(np.isinf(y))


def test_higgs_invalid_n_samples():
    """Higgs must reject invalid n_samples."""
    with pytest.raises(ValueError):
        load_higgs_subset(n_samples=0)
    with pytest.raises(ValueError):
        load_higgs_subset(n_samples=-1)

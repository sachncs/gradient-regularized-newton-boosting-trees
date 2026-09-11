"""Tests for the dataset loaders.

Two cases are covered per loader:

* shape, finiteness, and (for wine) standardization — these always
  run in offline mode by relying on the synthetic fallback for
  Higgs and a (typically cached) network fetch for wine.
* input validation — :func:`~grnbt.datasets.load_higgs_subset`
  rejects non-positive ``n_samples``.

These tests do not assert exact row counts because the underlying
UCI / OpenML datasets can change over time. They do assert that
every returned array is finite and has the expected rank.
"""

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


def test_load_wine_quality_local_path(tmp_path):
    """load_wine_quality accepts a pre-downloaded CSV via local_path."""
    import numpy as np
    from grnbt.datasets import load_wine_quality

    csv_path = tmp_path / "wine.csv"
    # 5 synthetic rows; UCI uses ';' separator; target is last column.
    rows = ["f1;f2;f3;f4;f5;f6;f7;f8;f9;f10;f11;target"]
    rng = np.random.RandomState(0)
    for _ in range(5):
        feats = ";".join(f"{v:.4f}" for v in rng.randn(11))
        rows.append(f"{feats};{rng.randint(0, 11)}")
    csv_path.write_text("\n".join(rows))

    x, y = load_wine_quality(local_path=str(csv_path))
    assert x.shape == (5, 11)
    assert y.shape == (5,)
    # Standardization sanity: mean ~ 0, std ~ 1 per column.
    assert np.allclose(x.mean(axis=0), 0.0, atol=1e-6)


def test_load_higgs_subset_local_path(tmp_path):
    """load_higgs_subset accepts a pre-downloaded CSV via local_path."""
    import numpy as np
    from grnbt.datasets import load_higgs_subset

    csv_path = tmp_path / "higgs.csv"
    rng = np.random.RandomState(0)
    arr = np.hstack([rng.randn(8, 28), rng.randint(0, 2, size=(8, 1)).astype(float)])
    np.savetxt(csv_path, arr, delimiter=",")

    x, y = load_higgs_subset(n_samples=8, local_path=str(csv_path))
    assert x.shape == (8, 28)
    assert y.shape == (8,)
    assert set(np.unique(y).tolist()).issubset({0, 1})

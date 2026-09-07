"""Tests for optional extensions (not part of the paper baseline).

These tests cover
:class:`~grnbt.extensions.histogram_tree.HistogramNewtonTree`:

* it must fit and predict with arbitrary ``n_bins`` (including the
  trivial ``n_bins == 1`` case which produces a single leaf);
* its predictions must be positively correlated with those of the
  exact :class:`~grnbt.tree.NewtonTree` (not identical because the
  threshold candidates are different);
* invalid ``n_bins`` values are rejected.

The tests are deliberately small because the histogram tree is only
used as a downstream speed-up, not in the paper reproduction.
"""

import numpy as np
import pytest

from grnbt.extensions.histogram_tree import HistogramNewtonTree
from grnbt.tree import NewtonTree


def test_histogram_tree_runs():
    """Histogram tree must fit and predict without error."""
    rng = np.random.RandomState(3)
    n, d = 64, 3
    x = rng.randn(n, d)
    g = rng.randn(n)
    h = np.ones(n)
    tree = HistogramNewtonTree(max_depth=2, min_samples_leaf=2, n_bins=8)
    tree.fit(x, g, h, 0.1)
    preds = tree.predict(x)
    assert preds.shape == (n,)


def test_histogram_tree_approximates_exact():
    """Histogram tree predictions should be close to exact tree on coarse data.

    The correlation is positive but not perfect because histogram
    binning approximates the exact threshold search. With sufficient
    bins (``n_bins >= n_samples``), the histogram tree converges to
    the exact tree.
    """
    rng = np.random.RandomState(4)
    n, d = 32, 2
    x = rng.randn(n, d)
    g = rng.randn(n)
    h = np.ones(n)
    lam = 0.5

    exact_tree = NewtonTree(max_depth=2, min_samples_leaf=2)
    exact_tree.fit(x, g, h, lam)
    hist_tree = HistogramNewtonTree(max_depth=2, min_samples_leaf=2, n_bins=16)
    hist_tree.fit(x, g, h, lam)

    exact_preds = exact_tree.predict(x)
    hist_preds = hist_tree.predict(x)
    # Not identical, but should be positively correlated
    corr = np.corrcoef(exact_preds, hist_preds)[0, 1]
    assert corr > 0.5


def test_histogram_tree_single_bin():
    """Histogram tree with 1 bin must still run (no splits possible)."""
    rng = np.random.RandomState(5)
    n, d = 10, 2
    x = rng.randn(n, d)
    g = rng.randn(n)
    h = np.ones(n)
    tree = HistogramNewtonTree(max_depth=2, min_samples_leaf=1, n_bins=1)
    tree.fit(x, g, h, 0.0)
    preds = tree.predict(x)
    assert preds.shape == (n,)
    assert np.allclose(preds, -np.sum(g) / np.sum(h))


def test_histogram_tree_invalid_n_bins():
    """Histogram tree must reject invalid n_bins."""
    with pytest.raises(ValueError):
        HistogramNewtonTree(n_bins=0)

"""Tests for the :class:`grnbt.tree.NewtonTree` weak learner.

The cases span:

* closed-form leaf weight (``w = -Σg / (Σh + λ)``);
* hyperparameter enforcement (``max_depth``, ``min_samples_leaf``,
  ``min_gain``, ``λ >= 0``);
* edge cases: single sample, constant features, near-zero Hessians,
  empty input, ``NaN``/``Inf``;
* surrogate loss monotonicity with depth (deeper trees achieve
  equal or lower quadratic surrogate ``Q(f) = ⟨g, f⟩ + 0.5 ⟨f, H f⟩``);
* sample-count invariant: every leaf covers at least
  ``min_samples_leaf`` samples.
"""

import numpy as np
import pytest

from grnbt.tree import NewtonTree


def _max_depth(node) -> int:
    """Helper to compute maximum depth below a node."""
    if node.is_leaf:
        return 0
    return 1 + max(_max_depth(node.left), _max_depth(node.right))


def test_tree_leaf_weight_formula():
    """Leaf weight must equal the closed-form Newton update."""
    rng = np.random.RandomState(0)
    n = 8
    g = rng.randn(n)
    h = np.ones(n) * 2.0
    lam = 1.0
    expected_w = -np.sum(g) / (np.sum(h) + lam)
    tree = NewtonTree(max_depth=0, min_samples_leaf=1)
    tree.fit(np.zeros((n, 1)), g, h, lam)
    preds = tree.predict(np.zeros((n, 1)))
    assert np.allclose(preds, expected_w)


def test_tree_depth_constraint():
    """Tree must respect max_depth."""
    rng = np.random.RandomState(1)
    n, d = 32, 2
    x = rng.randn(n, d)
    g = rng.randn(n)
    h = np.ones(n)
    tree = NewtonTree(max_depth=2, min_samples_leaf=1)
    tree.fit(x, g, h, 0.0)
    assert _max_depth(tree.root) <= 2


def test_tree_prediction_shape():
    """Predictions must match input sample count."""
    rng = np.random.RandomState(2)
    n, d = 16, 3
    x = rng.randn(n, d)
    g = rng.randn(n)
    h = np.ones(n)
    tree = NewtonTree(max_depth=2, min_samples_leaf=2)
    tree.fit(x, g, h, 0.1)
    preds = tree.predict(x)
    assert preds.shape == (n,)


def test_tree_single_sample():
    """Tree must handle a single sample gracefully."""
    x = np.array([[1.0]])
    g = np.array([2.0])
    h = np.array([1.0])
    tree = NewtonTree(max_depth=3, min_samples_leaf=1)
    tree.fit(x, g, h, 0.0)
    preds = tree.predict(x)
    assert preds.shape == (1,)
    assert np.isclose(preds[0], -2.0)


def test_tree_constant_feature():
    """Tree must handle constant features (no split possible)."""
    x = np.array([[1.0], [1.0], [1.0], [1.0]])
    g = np.array([1.0, -1.0, 2.0, -2.0])
    h = np.ones(4)
    tree = NewtonTree(max_depth=3, min_samples_leaf=1)
    tree.fit(x, g, h, 0.0)
    preds = tree.predict(x)
    assert np.allclose(preds, np.mean(preds))


def test_tree_all_constant_features():
    """Tree with all constant features must become a single leaf."""
    x = np.ones((10, 3))
    g = np.arange(10, dtype=float)
    h = np.ones(10)
    tree = NewtonTree(max_depth=3, min_samples_leaf=1)
    tree.fit(x, g, h, 0.5)
    preds = tree.predict(x)
    assert np.allclose(preds, -np.sum(g) / (np.sum(h) + 0.5))


def test_tree_negative_hessian_guard():
    """Tree must handle near-zero Hessians without divide-by-zero."""
    rng = np.random.RandomState(5)
    n, d = 8, 2
    x = rng.randn(n, d)
    g = rng.randn(n)
    h = np.full(n, 1e-15)
    tree = NewtonTree(max_depth=1, min_samples_leaf=1)
    tree.fit(x, g, h, 0.0)
    preds = tree.predict(x)
    assert np.all(np.isfinite(preds))


def test_tree_empty_data_raises():
    """Tree must raise on empty data."""
    tree = NewtonTree(max_depth=1)
    with pytest.raises(ValueError):
        tree.fit(np.empty((0, 2)), np.array([]), np.array([]), 0.0)


def test_tree_mismatched_grad_hessian_raises():
    """Tree must raise when g and h have different lengths."""
    tree = NewtonTree(max_depth=1)
    with pytest.raises(ValueError):
        tree.fit(np.ones((3, 1)), np.ones(3), np.ones(2), 0.0)


def test_tree_predict_before_fit_raises():
    """Tree must raise if predict is called before fit."""
    tree = NewtonTree(max_depth=1)
    with pytest.raises(ValueError):
        tree.predict(np.ones((2, 1)))


def test_tree_invalid_hyperparameters():
    """Tree must validate hyperparameters."""
    with pytest.raises(ValueError):
        NewtonTree(max_depth=-1)
    with pytest.raises(ValueError):
        NewtonTree(min_samples_leaf=0)
    with pytest.raises(ValueError):
        NewtonTree(min_gain=-1.0)


def test_tree_lam_negative_raises():
    """Tree must raise on negative regularization."""
    tree = NewtonTree(max_depth=1)
    with pytest.raises(ValueError):
        tree.fit(np.ones((3, 1)), np.ones(3), np.ones(3), -1.0)


def test_tree_nan_inputs_raises():
    """Tree must raise on NaN inputs."""
    tree = NewtonTree(max_depth=1)
    with pytest.raises(ValueError):
        tree.fit(np.array([[np.nan]]), np.array([1.0]), np.array([1.0]), 0.0)


def test_tree_inf_inputs_raises():
    """Tree must raise on infinite inputs."""
    tree = NewtonTree(max_depth=1)
    with pytest.raises(ValueError):
        tree.fit(np.array([[np.inf]]), np.array([1.0]), np.array([1.0]), 0.0)


def test_tree_gain_monotonicity():
    """Deeper trees should achieve equal or lower training surrogate loss."""
    rng = np.random.RandomState(7)
    n, d = 64, 4
    x = rng.randn(n, d)
    g = rng.randn(n)
    h = np.ones(n)
    lam = 0.1

    # Compute surrogate Q(f) = <g, f> + 0.5 <f, H f> for each tree
    def surrogate(tree):
        f = tree.predict(x)
        return float(np.sum(g * f) + 0.5 * np.sum(h * f**2))

    tree_depth1 = NewtonTree(max_depth=1, min_samples_leaf=1)
    tree_depth1.fit(x, g, h, lam)
    q1 = surrogate(tree_depth1)

    tree_depth3 = NewtonTree(max_depth=3, min_samples_leaf=1)
    tree_depth3.fit(x, g, h, lam)
    q3 = surrogate(tree_depth3)

    # Deeper tree should achieve lower (or equal) surrogate
    assert q3 <= q1 + 1e-6


def test_tree_min_samples_leaf_respected():
    """All leaves must have at least min_samples_leaf samples."""
    rng = np.random.RandomState(8)
    n, d = 32, 2
    x = rng.randn(n, d)
    g = rng.randn(n)
    h = np.ones(n)
    tree = NewtonTree(max_depth=4, min_samples_leaf=5)
    tree.fit(x, g, h, 0.0)

    def _check_leaf_sizes(node, mask):
        if node.is_leaf:
            assert np.count_nonzero(mask) >= 5
            return
        left_mask = mask & (x[:, node.feature_idx] <= node.threshold)
        right_mask = mask & ~left_mask
        _check_leaf_sizes(node.left, left_mask)
        _check_leaf_sizes(node.right, right_mask)

    _check_leaf_sizes(tree.root, np.ones(n, dtype=bool))

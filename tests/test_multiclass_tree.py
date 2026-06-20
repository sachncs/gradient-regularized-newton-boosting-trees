"""Tests for MultiClassNewtonTree weak learner."""

import numpy as np
import pytest

from grnbt.tree import MultiClassNewtonTree


def _max_depth(node) -> int:
    """Helper to compute maximum depth below a node."""
    if node.is_leaf:
        return 0
    return 1 + max(_max_depth(node.left), _max_depth(node.right))


def test_multiclass_tree_leaf_weight_formula():
    """Leaf weight must equal the closed-form Newton update per class."""
    rng = np.random.RandomState(0)
    n, k = 8, 3
    g = rng.randn(n, k)
    h = np.ones((n, k)) * 2.0
    lam = 1.0
    expected_w = -np.sum(g, axis=0) / (np.sum(h, axis=0) + lam)
    tree = MultiClassNewtonTree(n_classes=k, max_depth=0, min_samples_leaf=1)
    tree.fit(np.zeros((n, 1)), g, h, lam)
    preds = tree.predict(np.zeros((n, 1)))
    assert preds.shape == (n, k)
    assert np.allclose(preds, expected_w)


def test_multiclass_tree_depth_constraint():
    """Tree must respect max_depth."""
    rng = np.random.RandomState(1)
    n, d, k = 32, 2, 3
    x = rng.randn(n, d)
    g = rng.randn(n, k)
    h = np.ones((n, k))
    tree = MultiClassNewtonTree(n_classes=k, max_depth=2, min_samples_leaf=1)
    tree.fit(x, g, h, 0.0)
    assert _max_depth(tree.root) <= 2


def test_multiclass_tree_prediction_shape():
    """Predictions must match input sample count and class count."""
    rng = np.random.RandomState(2)
    n, d, k = 16, 3, 4
    x = rng.randn(n, d)
    g = rng.randn(n, k)
    h = np.ones((n, k))
    tree = MultiClassNewtonTree(n_classes=k, max_depth=2, min_samples_leaf=2)
    tree.fit(x, g, h, 0.1)
    preds = tree.predict(x)
    assert preds.shape == (n, k)


def test_multiclass_tree_single_sample():
    """Tree must handle a single sample gracefully."""
    x = np.array([[1.0]])
    g = np.array([[2.0, -1.0]])
    h = np.array([[1.0, 1.0]])
    tree = MultiClassNewtonTree(n_classes=2, max_depth=3, min_samples_leaf=1)
    tree.fit(x, g, h, 0.0)
    preds = tree.predict(x)
    assert preds.shape == (1, 2)
    assert np.allclose(preds[0], -g[0] / (h[0] + 1e-12))


def test_multiclass_tree_constant_feature():
    """Tree must handle constant features (no split possible)."""
    x = np.array([[1.0], [1.0], [1.0], [1.0]])
    g = np.array([[1.0, -1.0], [-1.0, 2.0], [2.0, -2.0], [-2.0, 1.0]])
    h = np.ones((4, 2))
    tree = MultiClassNewtonTree(n_classes=2, max_depth=3, min_samples_leaf=1)
    tree.fit(x, g, h, 0.0)
    preds = tree.predict(x)
    expected = -np.sum(g, axis=0) / (np.sum(h, axis=0) + 1e-12)
    assert np.allclose(preds, expected)


def test_multiclass_tree_gain_sums_across_classes():
    """Split gain must be the sum of gains across all classes."""
    rng = np.random.RandomState(3)
    n, d, k = 32, 2, 3
    x = rng.randn(n, d)
    g = rng.randn(n, k)
    h = np.ones((n, k))
    lam = 0.1

    tree_depth0 = MultiClassNewtonTree(n_classes=k, max_depth=0, min_samples_leaf=1)
    tree_depth0.fit(x, g, h, lam)

    tree_depth2 = MultiClassNewtonTree(n_classes=k, max_depth=2, min_samples_leaf=1)
    tree_depth2.fit(x, g, h, lam)

    # Deeper tree should achieve equal or lower surrogate
    f0 = tree_depth0.predict(x)
    f2 = tree_depth2.predict(x)
    q0 = float(np.sum(g * f0) + 0.5 * np.sum(h * f0**2))
    q2 = float(np.sum(g * f2) + 0.5 * np.sum(h * f2**2))
    assert q2 <= q0 + 1e-6


def test_multiclass_tree_empty_data_raises():
    """Tree must raise on empty data."""
    tree = MultiClassNewtonTree(n_classes=3, max_depth=1)
    with pytest.raises(ValueError):
        tree.fit(np.empty((0, 2)), np.empty((0, 3)), np.empty((0, 3)), 0.0)


def test_multiclass_tree_mismatched_shapes_raises():
    """Tree must raise when shapes mismatch."""
    tree = MultiClassNewtonTree(n_classes=3, max_depth=1)
    with pytest.raises(ValueError):
        tree.fit(np.ones((3, 1)), np.ones((3, 2)), np.ones((3, 3)), 0.0)


def test_multiclass_tree_predict_before_fit_raises():
    """Tree must raise if predict is called before fit."""
    tree = MultiClassNewtonTree(n_classes=3, max_depth=1)
    with pytest.raises(ValueError):
        tree.predict(np.ones((2, 1)))


def test_multiclass_tree_invalid_n_classes():
    """Tree must validate n_classes."""
    with pytest.raises(ValueError):
        MultiClassNewtonTree(n_classes=1)
    with pytest.raises(ValueError):
        MultiClassNewtonTree(n_classes=-1)


def test_multiclass_tree_lam_negative_raises():
    """Tree must raise on negative regularization."""
    tree = MultiClassNewtonTree(n_classes=2, max_depth=1)
    with pytest.raises(ValueError):
        tree.fit(np.ones((3, 1)), np.ones((3, 2)), np.ones((3, 2)), -1.0)


def test_multiclass_tree_nan_inputs_raises():
    """Tree must raise on NaN inputs."""
    tree = MultiClassNewtonTree(n_classes=2, max_depth=1)
    with pytest.raises(ValueError):
        tree.fit(
            np.array([[np.nan]]),
            np.array([[1.0, 1.0]]),
            np.array([[1.0, 1.0]]),
            0.0,
        )


def test_multiclass_tree_inf_inputs_raises():
    """Tree must raise on infinite inputs."""
    tree = MultiClassNewtonTree(n_classes=2, max_depth=1)
    with pytest.raises(ValueError):
        tree.fit(
            np.array([[np.inf]]),
            np.array([[1.0, 1.0]]),
            np.array([[1.0, 1.0]]),
            0.0,
        )


def test_multiclass_tree_2d_gradient_required():
    """Tree must require 2-D gradient input."""
    tree = MultiClassNewtonTree(n_classes=2, max_depth=1)
    with pytest.raises(ValueError):
        tree.fit(np.ones((3, 1)), np.ones(3), np.ones((3, 2)), 0.0)

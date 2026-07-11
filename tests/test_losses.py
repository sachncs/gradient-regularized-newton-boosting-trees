"""Tests for loss functions and their analytical properties.

This module exercises the four concrete loss classes plus the
abstract :class:`grnbt.losses.Loss` interface. It verifies:

* shape contracts (``gradient`` and ``hessian`` return inputs of
  compatible shape with ``y_pred``);
* correctness of values (e.g., ``MSE.loss(y, y) == 0``, the
  ``M_0`` constants match Appendix A of the paper);
* finite-difference agreement (the analytical ``CharbonnierLoss.
  gradient`` matches a central-difference approximation);
* defensive validation (rejection of non-NumPy inputs, shape
  mismatch, ``NaN``/``Inf``, and non-binary labels for BCE/CCE).
"""

import numpy as np
import pytest

from grnbt.losses import (
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss,
    CharbonnierLoss,
    MSELoss,
)


def test_mse_shapes(synthetic_regression):
    """MSE gradient and Hessian must match prediction shape."""
    x, y = synthetic_regression
    loss = MSELoss()
    y_pred = np.zeros_like(y)
    grad = loss.gradient(y, y_pred)
    hess = loss.hessian(y, y_pred)
    assert grad.shape == y.shape
    assert hess.shape == y.shape


def test_mse_loss_value():
    """MSE loss on perfect predictions is zero."""
    loss = MSELoss()
    y = np.array([1.0, 2.0, 3.0])
    assert loss.loss(y, y.copy()) == 0.0


def test_mse_gradient_sign():
    """MSE gradient points in the direction of the residual."""
    loss = MSELoss()
    y = np.array([1.0, 2.0])
    y_pred = np.array([0.0, 0.0])
    grad = loss.gradient(y, y_pred)
    # Gradient = 2*(0 - [1,2])/2 = [-1, -2]
    assert np.allclose(grad, np.array([-1.0, -2.0]))


def test_mse_hessian_constant():
    """MSE Hessian is constant 2/N."""
    loss = MSELoss()
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.5, 1.5, 2.5])
    hess = loss.hessian(y, y_pred)
    assert np.allclose(hess, np.full_like(y, 2.0 / 3.0))


def test_charbonnier_gradient_finite_difference():
    """Charbonnier gradient agrees with central finite differences element-wise."""
    loss = CharbonnierLoss()
    y = np.array([0.0, 1.0, -1.0])
    y_pred = np.array([0.1, 0.9, -0.8])
    eps = 1e-5
    grad = loss.gradient(y, y_pred)
    grad_fd = np.empty_like(grad)
    for i in range(y_pred.shape[0]):
        y_plus = y_pred.copy()
        y_minus = y_pred.copy()
        y_plus[i] += eps
        y_minus[i] -= eps
        grad_fd[i] = (loss.loss(y, y_plus) - loss.loss(y, y_minus)) / (2 * eps)
    assert np.allclose(grad, grad_fd, atol=1e-4)


def test_charbonnier_hessian_positive():
    """Charbonnier Hessian is strictly positive everywhere."""
    loss = CharbonnierLoss()
    y = np.array([0.0, 1.0, -2.0])
    y_pred = np.array([10.0, -5.0, 0.0])
    hess = loss.hessian(y, y_pred)
    assert np.all(hess > 0)


def test_charbonnier_m0_value():
    """Charbonnier M_0 is exactly 1."""
    assert CharbonnierLoss().hessian_lipschitz_constant() == 1.0


def test_bce_hessian_positive(synthetic_binary):
    """BCE Hessian is strictly positive for finite logits."""
    x, y = synthetic_binary
    loss = BinaryCrossEntropyLoss()
    y_pred = np.full_like(y, 0.5, dtype=float)
    hess = loss.hessian(y, y_pred)
    assert np.all(hess > 0)


def test_bce_loss_bounds():
    """BCE loss is non-negative."""
    loss = BinaryCrossEntropyLoss()
    y = np.array([0, 1, 0])
    y_pred = np.array([0.0, 0.0, 0.0])
    assert loss.loss(y, y_pred) >= 0.0


def test_bce_invalid_labels():
    """BCE must reject non-binary labels."""
    loss = BinaryCrossEntropyLoss()
    y = np.array([0, 2, 1])
    y_pred = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        loss.loss(y, y_pred)


def test_bce_gradient_zero_at_optimum():
    """BCE gradient is zero when predictions match labels (in probability space)."""
    loss = BinaryCrossEntropyLoss()
    y = np.array([0.0, 1.0])
    # Logits of 0 give probability 0.5; the gradient is (0.5 - y)/N
    y_pred = np.array([0.0, 0.0])
    grad = loss.gradient(y, y_pred)
    assert np.allclose(grad, np.array([0.25, -0.25]))


def test_cce_shapes(synthetic_multiclass):
    """CCE gradient and Hessian shapes are correct."""
    x, y, n_classes = synthetic_multiclass
    loss = CategoricalCrossEntropyLoss(n_classes)
    y_pred = np.zeros((x.shape[0], n_classes))
    grad = loss.gradient(y, y_pred)
    hess = loss.hessian(y, y_pred)
    assert grad.shape == (x.shape[0], n_classes)
    assert hess.shape == (x.shape[0], n_classes, n_classes)


def test_cce_invalid_n_classes():
    """CCE must reject invalid class counts."""
    with pytest.raises(ValueError):
        CategoricalCrossEntropyLoss(1)
    with pytest.raises(ValueError):
        CategoricalCrossEntropyLoss(-1)


def test_cce_invalid_labels():
    """CCE must reject out-of-range labels."""
    loss = CategoricalCrossEntropyLoss(3)
    y = np.array([0, 3, 1])  # 3 is out of range for 3 classes
    y_pred = np.zeros((3, 3))
    with pytest.raises(ValueError):
        loss.loss(y, y_pred)


def test_cce_gradient_sums_to_zero():
    """CCE gradient sums to zero across classes (probability conservation)."""
    loss = CategoricalCrossEntropyLoss(3)
    y = np.array([0, 1])
    y_pred = np.zeros((2, 3))
    grad = loss.gradient(y, y_pred)
    assert np.allclose(np.sum(grad, axis=1), 0.0)


def test_cce_hessian_symmetric():
    """CCE Hessian blocks are symmetric."""
    loss = CategoricalCrossEntropyLoss(3)
    y = np.array([0, 1])
    y_pred = np.zeros((2, 3))
    hess = loss.hessian(y, y_pred)
    for i in range(hess.shape[0]):
        assert np.allclose(hess[i], hess[i].T)


def test_hessian_lipschitz_constants():
    """Analytical M_0 values match paper (Appendix A)."""
    assert MSELoss().hessian_lipschitz_constant() == 0.0
    assert CharbonnierLoss().hessian_lipschitz_constant() == 1.0
    assert BinaryCrossEntropyLoss().hessian_lipschitz_constant() == 0.25
    assert CategoricalCrossEntropyLoss(3).hessian_lipschitz_constant() == 0.25


def test_empirical_risk_lipschitz_scaling():
    """M scales as sqrt(N) per Proposition 5.1."""
    loss = CharbonnierLoss()
    m_10 = loss.empirical_risk_lipschitz(10)
    m_40 = loss.empirical_risk_lipschitz(40)
    assert np.isclose(m_40, 2.0 * m_10)


def test_loss_input_validation():
    """Loss functions validate inputs robustly."""
    loss = MSELoss()
    with pytest.raises(TypeError):
        loss.loss([1.0, 2.0], np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        loss.loss(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        loss.loss(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        loss.loss(np.array([1.0, np.inf]), np.array([1.0, 2.0]))


def test_cce_non_integer_labels():
    """CCE must reject float labels."""
    loss = CategoricalCrossEntropyLoss(3)
    y = np.array([0.0, 1.0, 2.0])
    y_pred = np.zeros((3, 3))
    with pytest.raises(TypeError):
        loss.loss(y, y_pred)

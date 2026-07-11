"""Tests for Hilbert-space diagnostics.

Verifies the *exact* values of the empirical-Hilbert-space quantities
introduced in the paper:

* :func:`exact_newton_direction` returns ``-g / (h + λ)`` and
  gracefully handles zero Hessians (via ``1e-12`` epsilon).
* :func:`cosine_angle_theta` returns ``1`` for aligned vectors,
  ``0`` for ``H``-orthogonal vectors, ``-1`` for antiparallel
  vectors, and the safe ``1`` default for zero directions.
* :func:`weak_gradient_edge_gamma` recovers ``γ = 1`` when the
  weak step equals the exact Newton step.
* :func:`verify_lemma_4_2` confirms both identities of Lemma 4.2
  for the regularized exact Newton direction.

All tests also exercise input validation (shape, emptiness,
negativity of ``λ``, ``NaN``/``Inf``).
"""

import numpy as np
import pytest

from grnbt.diagnostics import (
    cosine_angle_theta,
    exact_newton_direction,
    verify_lemma_4_2,
    weak_gradient_edge_gamma,
)


def test_exact_newton_scalar():
    """Exact Newton direction matches element-wise formula."""
    g = np.array([1.0, -2.0, 3.0])
    h = np.array([1.0, 2.0, 3.0])
    lam = 0.5
    f = exact_newton_direction(g, h, lam)
    expected = -g / (h + lam)
    assert np.allclose(f, expected)


def test_exact_newton_zero_hessian():
    """Exact Newton handles zero Hessian with epsilon."""
    g = np.array([1.0, 2.0])
    h = np.array([0.0, 0.0])
    lam = 0.0
    f = exact_newton_direction(g, h, lam)
    assert np.all(np.isfinite(f))
    assert np.allclose(f, -g / 1e-12)


def test_exact_newton_negative_lam_raises():
    """Exact Newton must reject negative lambda."""
    with pytest.raises(ValueError):
        exact_newton_direction(np.ones(2), np.ones(2), -1.0)


def test_exact_newton_empty_raises():
    """Exact Newton must reject empty inputs."""
    with pytest.raises(ValueError):
        exact_newton_direction(np.array([]), np.array([]), 0.0)


def test_exact_newton_nan_raises():
    """Exact Newton must reject NaN inputs."""
    with pytest.raises(ValueError):
        exact_newton_direction(np.array([np.nan]), np.array([1.0]), 0.0)


def test_cosine_angle_perfect_alignment():
    """Identical directions give cos(Θ) = 1."""
    g = np.array([1.0, 2.0])
    h = np.ones(2)
    lam = 0.0
    f_exact = exact_newton_direction(g, h, lam)
    theta = cosine_angle_theta(g, h, f_exact, f_exact)
    assert np.isclose(theta, 1.0)


def test_cosine_angle_orthogonal():
    """Orthogonal directions (in H-norm) give cos(Θ) = 0."""
    # h = [1, 1], f_exact = [1, -1], f_weak = [1, 1]
    # <f_exact, f_weak>_H = 1*1*1 + 1*1*(-1) = 0
    g = np.array([2.0, 0.0])
    h = np.ones(2)
    f_exact = np.array([1.0, -1.0])
    f_weak = np.array([1.0, 1.0])
    theta = cosine_angle_theta(g, h, f_exact, f_weak)
    assert np.isclose(theta, 0.0, atol=1e-6)


def test_cosine_angle_opposite():
    """Opposite directions give cos(Θ) = -1."""
    g = np.array([1.0, 1.0])
    h = np.ones(2)
    f_exact = np.array([1.0, 1.0])
    f_weak = np.array([-1.0, -1.0])
    theta = cosine_angle_theta(g, h, f_exact, f_weak)
    assert np.isclose(theta, -1.0)


def test_cosine_angle_zero_direction():
    """Zero direction gives cos(Θ) = 1 as a safe default."""
    g = np.array([1.0])
    h = np.ones(1)
    f_exact = np.array([0.0])
    f_weak = np.array([1.0])
    theta = cosine_angle_theta(g, h, f_exact, f_weak)
    assert theta == 1.0


def test_cosine_angle_validation():
    """Cosine angle validates input shapes."""
    with pytest.raises(ValueError):
        cosine_angle_theta(np.ones(2), np.ones(2), np.ones(3), np.ones(2))
    with pytest.raises(TypeError):
        cosine_angle_theta([1.0], np.ones(1), np.ones(1), np.ones(1))


def test_weak_gradient_edge_perfect():
    """Exact weak step gives γ = 1."""
    g = np.array([1.0, 2.0])
    h = np.ones(2)
    lam = 0.0
    f_weak = exact_newton_direction(g, h, lam)
    gamma = weak_gradient_edge_gamma(g, h, lam, f_weak)
    assert np.isclose(gamma, 1.0)


def test_weak_gradient_edge_zero_gradient():
    """Zero gradient gives γ = 1 as a safe default."""
    g = np.array([0.0, 0.0])
    h = np.ones(2)
    lam = 0.0
    f_weak = np.array([1.0, -1.0])
    gamma = weak_gradient_edge_gamma(g, h, lam, f_weak)
    assert gamma == 1.0


def test_weak_gradient_edge_orthogonal():
    """A weak step orthogonal to the gradient gives a low edge."""
    g = np.array([1.0, 0.0])
    h = np.ones(2)
    lam = 0.0
    f_weak = np.array([0.0, 1.0])
    gamma = weak_gradient_edge_gamma(g, h, lam, f_weak)
    assert gamma < 1.0


def test_weak_gradient_edge_validation():
    """Weak gradient edge validates inputs."""
    with pytest.raises(ValueError):
        weak_gradient_edge_gamma(np.ones(2), np.ones(2), -1.0, np.ones(2))
    with pytest.raises(ValueError):
        weak_gradient_edge_gamma(np.ones(2), np.ones(3), 0.0, np.ones(2))


def test_lemma_4_2_identities():
    """Lemma 4.2 must hold for the exact Newton step."""
    g = np.array([1.0, -1.0, 0.5])
    h = np.array([1.0, 2.0, 1.5])
    lam = 0.3
    f_weak = exact_newton_direction(g, h, lam)
    checks = verify_lemma_4_2(g, h, lam, f_weak)
    assert checks["lambda_norm_bound"]
    assert checks["k_norm_identity"]


def test_lemma_4_2_zero_gradient():
    """Lemma 4.2 holds trivially when gradient is zero."""
    g = np.array([0.0, 0.0])
    h = np.ones(2)
    lam = 0.5
    f_weak = np.array([0.0, 0.0])
    checks = verify_lemma_4_2(g, h, lam, f_weak)
    assert checks["lambda_norm_bound"]
    assert checks["k_norm_identity"]


def test_lemma_4_2_negative_lam_raises():
    """Lemma 4.2 must reject negative lambda."""
    with pytest.raises(ValueError):
        verify_lemma_4_2(np.ones(2), np.ones(2), -1.0, np.ones(2))

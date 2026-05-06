"""Hilbert-space diagnostics for Newton boosting.

Provides exact Newton directions, cosine angles, weak gradient edges,
and numerical checks for Lemma 4.2 identities.
"""

from typing import Dict

import numpy as np


def _validate_diagonal_inputs(
    g: np.ndarray, h: np.ndarray, f_exact: np.ndarray, f_weak: np.ndarray
) -> None:
    """Validate inputs for diagonal-Hessian diagnostics.

    Raises:
        TypeError: For non-array inputs.
        ValueError: For shape mismatches, emptiness, or NaN/Inf.
    """
    for name, arr in [("g", g), ("h", h), ("f_exact", f_exact), ("f_weak", f_weak)]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
        if arr.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN values.")
        if np.any(np.isinf(arr)):
            raise ValueError(f"{name} contains infinite values.")
    if g.shape != h.shape:
        raise ValueError(f"Shape mismatch: g {g.shape} vs h {h.shape}")
    if g.shape != f_exact.shape:
        raise ValueError(f"Shape mismatch: g {g.shape} vs f_exact {f_exact.shape}")
    if g.shape != f_weak.shape:
        raise ValueError(f"Shape mismatch: g {g.shape} vs f_weak {f_weak.shape}")


def exact_newton_direction(g: np.ndarray, h: np.ndarray, lam: float) -> np.ndarray:
    """Compute the exact Newton direction for diagonal Hessians.

    For scalar-output losses the Hessian is diagonal (pointwise), so the
    exact step is simply f = -g / (h + lam) element-wise. A small epsilon
    is added to the denominator to prevent division by zero.

    Args:
        g: Gradient vector of shape (n_samples,) or (n_samples, n_classes).
        h: Hessian diagonal of shape (n_samples,) or (n_samples, n_classes).
        lam: Regularization parameter lambda_k. Must be non-negative.

    Returns:
        Exact Newton direction with same shape as g.

    Raises:
        TypeError: If g or h are not NumPy arrays.
        ValueError: If shapes mismatch, arrays are empty, or lam is negative.
    """
    if not isinstance(g, np.ndarray):
        raise TypeError(f"g must be a numpy.ndarray, got {type(g).__name__}")
    if not isinstance(h, np.ndarray):
        raise TypeError(f"h must be a numpy.ndarray, got {type(h).__name__}")
    if g.shape != h.shape:
        raise ValueError(f"Shape mismatch: g {g.shape} vs h {h.shape}")
    if g.size == 0:
        raise ValueError("Gradient and Hessian must not be empty.")
    if lam < 0:
        raise ValueError(f"Regularization lambda must be non-negative, got {lam}")
    if np.any(np.isnan(g)) or np.any(np.isnan(h)):
        raise ValueError("Inputs contain NaN values.")
    if np.any(np.isinf(g)) or np.any(np.isinf(h)):
        raise ValueError("Inputs contain infinite values.")

    result: np.ndarray = -g / (h + lam + 1e-12)
    return np.asarray(result, dtype=float)


def cosine_angle_theta(
    g: np.ndarray,
    h: np.ndarray,
    f_exact: np.ndarray,
    f_weak: np.ndarray,
) -> float:
    """Compute the cosine angle Θ_k in the H-induced inner product.

    The inner product is defined as <u, v>_H = <u, H v>.
    For a diagonal Hessian this reduces to sum(h * u * v).

    Args:
        g: Gradient vector.
        h: Hessian diagonal weights.
        f_exact: Exact Newton direction.
        f_weak: Weak learner direction.

    Returns:
        Cosine value in [-1, 1]. Returns 1.0 if either direction is zero.

    Raises:
        TypeError: For non-array inputs.
        ValueError: For shape mismatches or empty inputs.
    """
    _validate_diagonal_inputs(g, h, f_exact, f_weak)

    num: float = float(np.sum(h * f_exact * f_weak))
    den: float = float(np.sqrt(np.sum(h * f_exact**2) * np.sum(h * f_weak**2)))
    if den == 0.0:
        return 1.0
    cos_val = num / den
    cos_val = float(np.clip(cos_val, -1.0, 1.0))
    return cos_val


def weak_gradient_edge_gamma(
    g: np.ndarray,
    h: np.ndarray,
    lam: float,
    f_weak: np.ndarray,
) -> float:
    """Compute the weak gradient edge γ_k.

    The implied weak gradient is g^w = -(H + lam I) f_weak.
    The edge condition is ||g^w - g||^2 <= (1 - γ^2) ||g||^2.
    For diagonal H, g^w = -(h + lam) * f_weak.

    Args:
        g: Gradient vector.
        h: Hessian diagonal.
        lam: Regularization parameter. Must be non-negative.
        f_weak: Weak learner direction.

    Returns:
        Edge value in [0, 1]. Returns 1.0 if gradient is zero.

    Raises:
        TypeError: For non-array inputs.
        ValueError: For shape mismatches, empty inputs, or negative lam.
    """
    if not isinstance(g, np.ndarray):
        raise TypeError(f"g must be a numpy.ndarray, got {type(g).__name__}")
    if not isinstance(h, np.ndarray):
        raise TypeError(f"h must be a numpy.ndarray, got {type(h).__name__}")
    if not isinstance(f_weak, np.ndarray):
        raise TypeError(f"f_weak must be a numpy.ndarray, got {type(f_weak).__name__}")
    if g.shape != h.shape:
        raise ValueError(f"Shape mismatch: g {g.shape} vs h {h.shape}")
    if g.shape != f_weak.shape:
        raise ValueError(f"Shape mismatch: g {g.shape} vs f_weak {f_weak.shape}")
    if g.size == 0:
        raise ValueError("Inputs must not be empty.")
    if lam < 0:
        raise ValueError(f"Regularization lambda must be non-negative, got {lam}")
    if np.any(np.isnan(g)) or np.any(np.isnan(h)) or np.any(np.isnan(f_weak)):
        raise ValueError("Inputs contain NaN values.")
    if np.any(np.isinf(g)) or np.any(np.isinf(h)) or np.any(np.isinf(f_weak)):
        raise ValueError("Inputs contain infinite values.")

    g_weak = -(h + lam) * f_weak
    diff_norm_sq: float = float(np.sum((g_weak - g) ** 2))
    g_norm_sq: float = float(np.sum(g**2))
    if g_norm_sq == 0.0:
        return 1.0
    gamma_sq = 1.0 - diff_norm_sq / g_norm_sq
    gamma_sq = float(np.clip(gamma_sq, 0.0, 1.0))
    return float(np.sqrt(gamma_sq))


def verify_lemma_4_2(
    g: np.ndarray,
    h: np.ndarray,
    lam: float,
    f_weak: np.ndarray,
) -> Dict[str, bool]:
    """Numerically verify Lemma 4.2 identities.

    Identity (i):  λ ||f_weak|| <= ||g||
    Identity (ii): ||f_weak||^2_K = -<g, f_weak> where K = H + λ I.

    Args:
        g: Gradient vector.
        h: Hessian diagonal.
        lam: Regularization parameter. Must be non-negative.
        f_weak: Weak learner direction.

    Returns:
        Dictionary with boolean flags for each identity.

    Raises:
        TypeError: For non-array inputs.
        ValueError: For shape mismatches, empty inputs, or negative lam.
    """
    if not isinstance(g, np.ndarray):
        raise TypeError(f"g must be a numpy.ndarray, got {type(g).__name__}")
    if not isinstance(h, np.ndarray):
        raise TypeError(f"h must be a numpy.ndarray, got {type(h).__name__}")
    if not isinstance(f_weak, np.ndarray):
        raise TypeError(f"f_weak must be a numpy.ndarray, got {type(f_weak).__name__}")
    if g.shape != h.shape:
        raise ValueError(f"Shape mismatch: g {g.shape} vs h {h.shape}")
    if g.shape != f_weak.shape:
        raise ValueError(f"Shape mismatch: g {g.shape} vs f_weak {f_weak.shape}")
    if g.size == 0:
        raise ValueError("Inputs must not be empty.")
    if lam < 0:
        raise ValueError(f"Regularization lambda must be non-negative, got {lam}")
    if np.any(np.isnan(g)) or np.any(np.isnan(h)) or np.any(np.isnan(f_weak)):
        raise ValueError("Inputs contain NaN values.")
    if np.any(np.isinf(g)) or np.any(np.isinf(h)) or np.any(np.isinf(f_weak)):
        raise ValueError("Inputs contain infinite values.")

    norm_f = float(np.linalg.norm(f_weak))
    norm_g = float(np.linalg.norm(g))
    bound_ok = lam * norm_f <= norm_g + 1e-6

    k_weighted_norm_sq = float(np.sum((h + lam) * f_weak**2))
    inner = float(-np.sum(g * f_weak))
    identity_ok = np.isclose(k_weighted_norm_sq, inner, atol=1e-5)

    return {
        "lambda_norm_bound": bool(bound_ok),
        "k_norm_identity": bool(identity_ok),
    }

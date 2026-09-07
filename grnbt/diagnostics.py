"""Hilbert-space diagnostics for Newton boosting.

This module computes the empirical-Hilbert-space quantities that the
paper uses to characterize the *quality* of the weak learner at each
boosting iteration:

* :func:`exact_newton_direction` — the closed-form step
  ``f = -(H + λ I)^{-1} g`` for diagonal ``H``.
* :func:`cosine_angle_theta` — the cosine angle ``Θ_k`` between the
  exact step and the actual weak learner step, measured in the
  ``H``-induced inner product.
* :func:`weak_gradient_edge_gamma` — the edge ``γ_k`` that quantifies
  how well the weak learner matches the true gradient in
  ``H``-contraction.
* :func:`verify_lemma_4_2` — numerical checker for the two identities
  that constitute Lemma 4.2 of the paper.

These diagnostics operate on the empirical ``R^N`` (or ``R^{NK}``)
vectors representing the Hilbert-space element ``F`` at a single
iteration. The formalism mirrors the analytical definitions in
Sections 3–4 of the paper.

Examples
--------

>>> import numpy as np
>>> from grnbt.diagnostics import (
...     exact_newton_direction, cosine_angle_theta, weak_gradient_edge_gamma,
... )
>>> g = np.array([1.0, -2.0, 0.5])
>>> h = np.array([1.0, 1.0, 1.0])
>>> f_exact = exact_newton_direction(g, h, lam=0.0)
>>> f_exact
array([-1. ,  2. , -0.5])
>>> cosine_angle_theta(g, h, f_exact, f_exact)   # self-alignment
1.0
"""

from typing import Dict

import numpy as np


def validate_diagonal_inputs(
    g: np.ndarray, h: np.ndarray, f_exact: np.ndarray, f_weak: np.ndarray
) -> None:
    """Validate inputs for the diagonal-Hessian diagnostics.

    Centralizes the type, rank, finiteness, and shape checks used by
    :func:`cosine_angle_theta`. The remaining diagnostic functions
    inline their checks because they accept a different subset of
    arguments.

    Raises:
        TypeError: If any input is not a NumPy array.
        ValueError: If any array is empty, contains ``NaN``/``Inf``,
            or if shapes disagree.
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

    For a diagonal Hessian ``H`` with pointwise entries ``h_i``, the
    regularized Newton step is

        f_i = - g_i / (h_i + λ)

    computed element-wise. This is the **unrestricted** minimizer of
    the second-order surrogate, which is the reference direction used
    to benchmark the weak learner.

    Args:
        g: Gradient vector of shape ``(n_samples,)`` or
            ``(n_samples, n_classes)``.
        h: Hessian diagonal of the same shape as ``g``.
        lam: Regularization parameter ``λ_k``. Must be non-negative.

    Returns:
        Exact Newton direction with the same shape as ``g``.

    Raises:
        TypeError: If ``g`` or ``h`` is not a NumPy array.
        ValueError: If shapes mismatch, arrays are empty, ``lam`` is
            negative, or any element is ``NaN`` / ``Inf``.

    Notes:
        A ``1e-12`` epsilon is added to the denominator to prevent
        division by zero when both ``h_i`` and ``λ`` are numerically
        zero. This does not change the closed-form solution in any
        well-conditioned setting.
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

    # Closed-form element-wise direction with ε to guard against zero
    # denominators in degenerate regimes.
    result: np.ndarray = -g / (h + lam + 1e-12)
    return np.asarray(result, dtype=float)


def cosine_angle_theta(
    g: np.ndarray,
    h: np.ndarray,
    f_exact: np.ndarray,
    f_weak: np.ndarray,
) -> float:
    r"""Compute the cosine angle ``Θ_k`` in the ``H``-induced inner product.

    The empirical inner product induced by the diagonal Hessian ``H``
    is

        <u, v>_H = Σ_i h_i * u_i * v_i.

    The cosine angle between two directions is then

        cos Θ = <f_exact, f_weak>_H / (||f_exact||_H * ||f_weak||_H).

    A value of ``1`` means the weak learner moves in the same
    ``H``-direction as the exact Newton step; a value of ``0`` means
    ``H``-orthogonality; ``-1`` means opposite directions.

    Args:
        g: Gradient vector (used only for shape validation).
        h: Hessian diagonal weights ``h_i``.
        f_exact: Exact Newton direction ``-g / (h + λ)``.
        f_weak: Weak learner direction ``tree.predict(x)``.

    Returns:
        Cosine value in ``[-1, 1]``. Returns ``1.0`` if either direction
        is exactly zero (this is the convention used by the paper to
        avoid 0/0 ambiguities at convergence).

    Raises:
        TypeError: For non-array inputs.
        ValueError: For shape mismatches, empty inputs, or non-finite
            values.
    """
    validate_diagonal_inputs(g, h, f_exact, f_weak)

    # Numerator: <f_exact, f_weak>_H = Σ_i h_i * f_exact_i * f_weak_i.
    num: float = float(np.sum(h * f_exact * f_weak))
    # Denominator: product of H-norms.
    den: float = float(np.sqrt(np.sum(h * f_exact**2) * np.sum(h * f_weak**2)))
    if den == 0.0:
        # Convention: zero directions align with everything (paper, §4).
        return 1.0
    cos_val = num / den
    # Clip to [-1, 1] to absorb floating-point rounding outside the
    # closed unit interval.
    cos_val = float(np.clip(cos_val, -1.0, 1.0))
    return cos_val


def weak_gradient_edge_gamma(
    g: np.ndarray,
    h: np.ndarray,
    lam: float,
    f_weak: np.ndarray,
) -> float:
    r"""Compute the weak gradient edge ``γ_k``.

    The implied weak gradient at iteration ``k`` is

        g^w = -(H + λ I) f_weak,

    and the edge ``γ_k`` captures how well the weak step recovers the
    true gradient in ``L^2``:

        ||g^w - g||^2  <=  (1 - γ^2) ||g||^2

        ⟹  γ_k = sqrt(1 - ||g^w - g||^2 / ||g||^2).

    The edge is small when ``f_weak`` does not align with the exact
    gradient and equals ``1`` for the exact Newton step.

    Args:
        g: Gradient vector.
        h: Hessian diagonal.
        lam: Regularization parameter ``λ_k``. Must be non-negative.
        f_weak: Weak learner direction.

    Returns:
        Edge value in ``[0, 1]``. Returns ``1.0`` if the gradient is
        exactly zero (degenerate case).

    Raises:
        TypeError: For non-array inputs.
        ValueError: For shape mismatches, empty inputs, negative ``λ``,
            or non-finite values.
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

    # g^w = -(h + λ) * f_weak (diagonal Hessian specialization).
    g_weak = -(h + lam) * f_weak
    diff_norm_sq: float = float(np.sum((g_weak - g) ** 2))
    g_norm_sq: float = float(np.sum(g**2))
    if g_norm_sq == 0.0:
        # Perfect "contraction" trivially — already at a stationary point.
        return 1.0
    # γ^2 = 1 - ||g^w - g||^2 / ||g||^2; clip to [0, 1] to absorb
    # floating-point rounding.
    gamma_sq = 1.0 - diff_norm_sq / g_norm_sq
    gamma_sq = float(np.clip(gamma_sq, 0.0, 1.0))
    return float(np.sqrt(gamma_sq))


def verify_lemma_4_2(
    g: np.ndarray,
    h: np.ndarray,
    lam: float,
    f_weak: np.ndarray,
) -> Dict[str, bool]:
    """Numerically verify the two identities of Lemma 4.2.

    For the *regularized exact* Newton step, the paper proves:

      (i)  ``λ_k ||f|| <= ||g_k||``
      (ii) ``||f||^2_K = -<g_k, f>``, where ``K = H + λ I``

    This function is intended for ``f_weak = f_exact`` (the closed-form
    step) so that the identities hold exactly. With arbitrary weak
    learners they fail; in that case use this as a learning tool by
    inspecting :func:`cosine_angle_theta` and
    :func:`weak_gradient_edge_gamma` instead.

    Args:
        g: Gradient vector.
        h: Hessian diagonal.
        lam: Regularization parameter ``λ_k``. Must be non-negative.
        f_weak: Weak learner direction (use ``f_exact`` for the
            identities to hold).

    Returns:
        Dictionary with two boolean flags:

        * ``"lambda_norm_bound"`` — ``True`` iff
          ``λ ||f_weak|| <= ||g|| + 1e-6``.
        * ``"k_norm_identity"`` — ``True`` iff
          ``<g, f_weak>_K ≈ -<g, f_weak>`` to within ``1e-5``.

    Raises:
        TypeError: For non-array inputs.
        ValueError: For shape mismatches, empty inputs, negative ``λ``,
            or non-finite values.
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
    # Identity (i): λ ||f|| <= ||g||. The +1e-6 absorbs floating-point
    # rounding on near-equality.
    bound_ok = lam * norm_f <= norm_g + 1e-6

    # Identity (ii): ||f||^2_K = -<g, f>, with K = H + λ I.
    # Left-hand side: <f, (H + λ I) f> = Σ (h_i + λ) * f_i^2.
    # Right-hand side: -<g, f> = -Σ g_i * f_i.
    k_weighted_norm_sq = float(np.sum((h + lam) * f_weak**2))
    inner = float(-np.sum(g * f_weak))
    identity_ok = np.isclose(k_weighted_norm_sq, inner, atol=1e-5)

    return {
        "lambda_norm_bound": bool(bound_ok),
        "k_norm_identity": bool(identity_ok),
    }

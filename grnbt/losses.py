"""Loss functions for Gradient Regularized Newton Boosting Trees (GRNBT).

This module defines the pointwise losses used by the Newton's method boosting
engines. Each loss exposes:

1. ``loss(y_true, y_pred)`` — the empirical risk
   ``L(F) = (1/N) sum_i l(F(x_i), y_i)``.
2. ``gradient(y_true, y_pred)`` — the gradient of the *empirical* risk with
   respect to the predictions ``F`` (NOT the per-sample loss derivative).
3. ``hessian(y_true, y_pred)`` — the second-order information, either as a
   pointwise diagonal or as a block-diagonal operator.
4. ``hessian_lipschitz_constant()`` — the analytically derived constant ``M_0``
   used by **Proposition 5.1** of the paper to scale the adaptive
   regularization ``λ_k`` of the Gradient Regularized Newton (GRN) algorithm.

Conventions
-----------

* For scalar-output losses the Hessian is the per-sample scalar
  ``d^2 l / d y_pred^2`` divided by ``N``; this matches the diagonal Hessian
  used by ``NewtonTree`` and by ``grnbt.diagnostics``.
* For multi-class losses (CCE) the Hessian is the ``(K, K)`` block
  ``diag(p) - p p^T`` divided by ``N``, returned as a dense
  ``(N, K, K)`` array. The diagonal extraction used by ``MultiClassNewtonTree``
  is implemented in :class:`grnbt.boosting.MultiClassNewtonBoosting`.
* Numerical stability is handled with ``np.hypot`` for the Charbonnier loss,
  ``log(x + eps)`` for BCE/CCE, and an ``eps`` in ``softmax`` denominators.

References
----------

Zozoulenko, N., Falkowski, D., Cass, T., Gonon, L. (2026).
*Gradient Regularized Newton Boosting Trees with Global Convergence.*
arXiv:2605.00581v1 — Section 4 (loss gradients), Appendix A (``M_0`` derivations),
and Proposition 5.1 (scaling of empirical Hessian Lipschitz constants).

Examples
--------

>>> import numpy as np
>>> from grnbt.losses import MSELoss, CharbonnierLoss
>>> y = np.array([1.0, 2.0, 3.0])
>>> y_pred = np.array([0.9, 2.1, 2.95])
>>> MSELoss().loss(y, y_pred)
0.0101...
>>> CharbonnierLoss().hessian_lipschitz_constant()
1.0
"""

from abc import ABC, abstractmethod

import numpy as np


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate common scalar-output loss inputs.

    The MSE/Charbonnier/BCE losses share the same shape, type, and finiteness
    requirements. This helper centralizes the checks so that each public loss
    method calls it as its first statement. The categorical cross-entropy
    loss has stricter requirements (integer labels, fixed class count) and
    performs its own validation inline because it cannot share a function
    across all subclasses.

    Args:
        y_true: Ground-truth targets, shape ``(n_samples,)``.
        y_pred: Model predictions, shape ``(n_samples,)``.

    Raises:
        TypeError: If either input is not a NumPy array, if ``y_pred`` is
            not floating-point, or if ``y_true`` is neither integer nor
            floating-point.
        ValueError: If the shapes differ, if either array is empty, or
            if either contains ``NaN`` / ``Inf``.
    """
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"y_true must be a numpy.ndarray, got {type(y_true).__name__}")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"y_pred must be a numpy.ndarray, got {type(y_pred).__name__}")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if not np.issubdtype(y_true.dtype, np.floating) and not np.issubdtype(
        y_true.dtype, np.integer
    ):
        raise TypeError(f"y_true must be numeric, got dtype {y_true.dtype}")
    if not np.issubdtype(y_pred.dtype, np.floating):
        raise TypeError(f"y_pred must be floating-point, got dtype {y_pred.dtype}")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("Inputs contain NaN values.")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Inputs contain infinite values.")


class Loss(ABC):
    """Abstract base class for a pointwise loss ``l(y_pred, y_true)``.

    Subclasses must implement four primitives:

    * :meth:`loss` — empirical risk value.
    * :meth:`gradient` — gradient of the empirical risk, shape-preserving.
    * :meth:`hessian` — second-order information; either a pointwise diagonal
      array of shape ``(n_samples,)`` for scalar-output losses, or a block
      diagonal of shape ``(n_samples, K, K)`` for the multi-class case.
    * :meth:`hessian_lipschitz_constant` — analytical ``M_0`` from Appendix A.

    The :meth:`empirical_risk_lipschitz` convenience method applies the
    ``sqrt(N)`` scaling from Proposition 5.1 automatically.

    Subclasses are stateless from the standpoint of the gradient/Hessian:
    all per-iteration quantities are functions only of the current
    predictions, not of internal counters.
    """

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the empirical average loss.

        Implements ``L(F) = (1/N) * sum_i l(y_pred_i, y_true_i)``.

        Args:
            y_true: Ground-truth targets, shape ``(n_samples,)`` for
                scalar-output losses, or shape ``(n_samples, n_classes)``
                for multi-class losses.
            y_pred: Model predictions with the same shape as ``y_true``.

        Returns:
            Scalar empirical risk value. A Python ``float``, never an
            array scalar, to keep JSON-serializable history logging
            simple.
        """

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient of the empirical risk with respect to ``y_pred``.

        Note that this is the empirical mean of the per-sample derivatives,
        i.e. ``dL/dy_pred[i] = (1/N) * d l / d y_pred[i]``. This is the
        ``g_k`` used by the Newton's method boosting engines.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.

        Returns:
            Gradient array with the same shape as ``y_pred``. For
            multi-class losses this is a matrix of shape
            ``(n_samples, n_classes)``.
        """

    @abstractmethod
    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Pointwise (or block) Hessian of the empirical risk.

        For scalar-output losses this returns an array of shape
        ``(n_samples,)`` containing the per-sample diagonal entries
        ``d^2 l / d y_pred^2 / N``. This is what the tree builder
        consumes as ``h`` and what the diagnostics interpret as the
        ``H``-induced inner product weights.

        For multi-class losses this returns a dense ``(n_samples, K, K)``
        block diagonal of the empirical risk Hessian. The multi-class
        engine extracts only the diagonal entries for the tree builder.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.

        Returns:
            Hessian representation, shape-preserving with ``y_pred`` for
            scalar losses or ``(n_samples, K, K)`` for multi-class.
        """

    @abstractmethod
    def hessian_lipschitz_constant(self) -> float:
        """Return ``M_0``, the pointwise Hessian Lipschitz constant.

        This is the analytical constant from Appendix A of the paper,
        defining how fast the Hessian changes as a function of the
        prediction. Proposition 5.1 uses it to derive the empirical-risk
        constant ``M = M_0 * sqrt(N)`` consumed by the adaptive
        ``λ_k = λ_base + sqrt(M ||g_k||)`` formula of GRN.

        Returns:
            Non-negative scalar ``M_0``. Zero means the Hessian is
            constant (as for MSE).
        """

    def empirical_risk_lipschitz(self, n_samples: int) -> float:
        """Compute the empirical-risk Hessian Lipschitz constant M.

        Following Proposition 5.1 in the paper: M = M_0 * sqrt(N).

        Args:
            n_samples: Number of training samples N. Must be positive.

        Returns:
            Empirical risk Lipschitz constant M.

        Raises:
            ValueError: If n_samples is not positive.
        """
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
        return float(self.hessian_lipschitz_constant() * np.sqrt(n_samples))


class MSELoss(Loss):
    """Mean squared error empirical risk.

    Defined as ``L(F) = (1/N) * sum_i (y_pred[i] - y_true[i])^2``.

    This is the canonical strongly convex surrogate used to validate
    Newton-boosting theory. Its Hessian is constant, so the pointwise
    Lipschitz constant is ``M_0 = 0`` — the loss is "exactly quadratic"
    in the prediction and gradient-descent/Newton coincide.

    Attributes:
        None (stateless).
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the MSE empirical risk.

        Formula: ``(1/N) * sum((y_true - y_pred) ** 2)``.

        Args:
            y_true: Ground-truth targets, shape ``(n_samples,)``.
            y_pred: Model predictions, shape ``(n_samples,)``.

        Returns:
            Non-negative scalar risk.

        Raises:
            TypeError: If either input is not a NumPy array or has the
                wrong dtype.
            ValueError: On shape mismatch, empty arrays, or non-finite values.
        """
        _validate_inputs(y_true, y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient of the empirical risk: ``2 * (y_pred - y_true) / N``.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.

        Returns:
            Gradient array, same shape as ``y_pred``.

        Raises:
            TypeError: If either input is not a NumPy array or has the
                wrong dtype.
            ValueError: On shape mismatch, empty arrays, or non-finite values.
        """
        _validate_inputs(y_true, y_pred)
        return np.asarray(2.0 * (y_pred - y_true) / y_true.shape[0], dtype=float)

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Hessian diagonal: constant ``2 / N`` per sample.

        The second derivative does not depend on the prediction, which is
        why the Hessian is identical for every input and ``M_0 = 0``.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.

        Returns:
            Constant array of shape ``(n_samples,)`` filled with ``2/N``.

        Raises:
            TypeError: If either input is not a NumPy array or has the
                wrong dtype.
            ValueError: On shape mismatch, empty arrays, or non-finite values.
        """
        _validate_inputs(y_true, y_pred)
        return np.asarray(np.full_like(y_true, 2.0 / y_true.shape[0]), dtype=float)

    def hessian_lipschitz_constant(self) -> float:
        """Return ``M_0 = 0`` for the constant Hessian case.

        A constant Hessian has zero Lipschitz constant because it does
        not change as a function of the prediction. Consequently, the
        empirical-risk constant ``M`` also vanishes and the GRN
        regularization reduces to the static ``λ_base``.
        """
        return 0.0


class CharbonnierLoss(Loss):
    r"""Charbonnier pseudo-Huber loss: ``l(d) = sqrt(1 + d^2) - 1``.

    Here ``d = y_pred - y_true``. The Charbonnier loss is a smooth,
    everywhere-differentiable approximation to ``|d|`` that behaves
    quadratically for ``|d| << 1`` and linearly for ``|d| >> 1``.

    Used in the paper to demonstrate vanilla Newton divergence: the
    Hessian depends on the prediction (unlike MSE), making the loss
    no longer exactly quadratic. The third derivative of ``sqrt(1+d^2)``
    is bounded in absolute value by 1, so ``M_0 = 1`` (see Appendix A).

    Implementation notes:
        ``np.hypot(1.0, d)`` evaluates ``sqrt(1 + d^2)`` accurately even
        for ``|d|`` of order ``1e8`` where naive ``np.sqrt`` may overflow.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Charbonnier empirical risk.

        Implemented with ``np.hypot`` for numerical stability against
        overflow on large residuals.

        Args:
            y_true: Ground-truth targets, shape ``(n_samples,)``.
            y_pred: Model predictions, shape ``(n_samples,)``.

        Returns:
            Non-negative scalar risk.

        Raises:
            TypeError: If either input is not a NumPy array or has the
                wrong dtype.
            ValueError: On shape mismatch, empty arrays, or non-finite values.
        """
        _validate_inputs(y_true, y_pred)
        d = y_pred - y_true
        return float(np.mean(np.hypot(1.0, d) - 1.0))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient: ``d / (sqrt(1 + d^2) * N)``.

        As a function of ``d``, this saturates at ``±1/N``, giving
        Lipschitz-stable gradients at large residuals.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.

        Returns:
            Gradient array, same shape as ``y_pred``.

        Raises:
            TypeError: If either input is not a NumPy array or has the
                wrong dtype.
            ValueError: On shape mismatch, empty arrays, or non-finite values.
        """
        _validate_inputs(y_true, y_pred)
        d = y_pred - y_true
        denom = np.hypot(1.0, d) * y_true.shape[0]
        return np.asarray(d / denom, dtype=float)

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Hessian: ``1 / ((1 + d^2)^{3/2} * N)``.

        Always strictly positive, confirming strong convexity. The
        Hessian is bounded by ``1/N`` and attains its maximum at
        ``d = 0``, decaying to zero as ``|d| → ∞``.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.

        Returns:
            Hessian array of shape ``(n_samples,)`` with strictly
            positive entries.

        Raises:
            TypeError: If either input is not a NumPy array or has the
                wrong dtype.
            ValueError: On shape mismatch, empty arrays, or non-finite values.
        """
        _validate_inputs(y_true, y_pred)
        d = y_pred - y_true
        hypot_d = np.hypot(1.0, d)
        return np.asarray(1.0 / (hypot_d**3 * y_true.shape[0]), dtype=float)

    def hessian_lipschitz_constant(self) -> float:
        """Return ``M_0 = 1`` from the analytical bound of Appendix A.

        The third derivative of ``sqrt(1 + d^2) - 1`` is bounded in
        absolute value by ``1`` for all real ``d``, giving the
        ``M_0 = 1`` constant.
        """
        return 1.0


class BinaryCrossEntropyLoss(Loss):
    """Binary cross-entropy (logistic) loss on logits.

    Operates on raw logits ``z`` (NOT probabilities). Labels must live in
    ``{0, 1}``. With ``p = σ(z) = 1 / (1 + e^{-z})``:

    * Gradient: ``(p - y) / N``
    * Hessian:  ``p * (1 - p) / N``
    * Lipschitz constant: ``M_0 = 1/4``

    The Hessian is strictly positive for finite logits, guaranteeing
    strong convexity. The ``1/4`` bound follows from the identity
    ``sup_x p(1-p)(1 - 2p) = 1/4``.

    Raises:
        ValueError: If ``y_true`` contains labels outside ``{0, 1}``.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """BCE empirical risk with finite-logit stability.

        Adds ``eps`` inside both ``log`` terms to keep the result
        well-defined when the sigmoid saturates near ``0`` or ``1``.

        Args:
            y_true: Ground-truth binary labels, shape ``(n_samples,)``.
            y_pred: Logits, shape ``(n_samples,)``.

        Returns:
            Non-negative scalar risk.

        Raises:
            TypeError: For non-arrays or wrong dtypes.
            ValueError: For shape mismatch, empty arrays, non-finite values,
                or non-binary labels.
        """
        _validate_inputs(y_true, y_pred)
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true for BCE must contain only 0 and 1.")
        p = 1.0 / (1.0 + np.exp(-y_pred))
        eps = 1e-15
        return -float(
            np.mean(y_true * np.log(p + eps) + (1.0 - y_true) * np.log(1.0 - p + eps))
        )

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient: ``(σ(y_pred) - y_true) / N``.

        For a model at the optimum (where ``σ(z) = y``), the gradient
        is exactly zero. Since the BCE is strictly convex, the unique
        minimizer satisfies this condition.

        Args:
            y_true: Binary labels.
            y_pred: Logits.

        Returns:
            Gradient array, shape ``(n_samples,)``.

        Raises:
            TypeError: For non-arrays or wrong dtypes.
            ValueError: For shape mismatch, empty arrays, non-finite values,
                or non-binary labels.
        """
        _validate_inputs(y_true, y_pred)
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true for BCE must contain only 0 and 1.")
        p = 1.0 / (1.0 + np.exp(-y_pred))
        return np.asarray((p - y_true) / y_true.shape[0], dtype=float)

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Hessian: ``p * (1 - p) / N``.

        Strictly positive for any finite logits, vanishing only in the
        ``|y_pred| → ∞`` limit where the prediction becomes saturated.

        Args:
            y_true: Binary labels.
            y_pred: Logits.

        Returns:
            Strictly positive Hessian array.

        Raises:
            TypeError: For non-arrays or wrong dtypes.
            ValueError: For shape mismatch, empty arrays, non-finite values,
                or non-binary labels.
        """
        _validate_inputs(y_true, y_pred)
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true for BCE must contain only 0 and 1.")
        p = 1.0 / (1.0 + np.exp(-y_pred))
        return np.asarray((p * (1.0 - p)) / y_true.shape[0], dtype=float)

    def hessian_lipschitz_constant(self) -> float:
        """Return ``M_0 = 1/4`` for binary cross-entropy.

        Bounds the derivative of ``p (1 - p)`` (w.r.t. ``z``) by ``1/4``,
        which is the maximum of the derivative
        ``p(1-p)(1 - 2p)`` at ``z = 0`` (where ``p = 1/2``). Reference:
        Appendix A of the paper.
        """
        return 0.25


class CategoricalCrossEntropyLoss(Loss):
    """Categorical cross-entropy (softmax) loss on logits.

    Operates on logits ``z ∈ R^K`` and integer labels
    ``y ∈ {0, …, K - 1}``. With probabilities
    ``p_j = exp(z_j) / sum_k exp(z_k)``:

    * Gradient:  ``(p - 1_y) / N``  (matrix)
    * Hessian:   ``(diag(p) - p p^T) / N``  per sample, returned as a
      dense ``(N, K, K)`` block-diagonal tensor
    * Lipschitz constant: ``M_0 = 1/4`` (same bound as BCE)

    The dense ``(N, K, K)`` Hessian is mathematically exact but uses
    ``O(N K^2)`` memory. For large ``K``, a block-diagonal operator
    would suffice; consumers (e.g. the multi-class tree builder) only
    need the diagonal entries, which are exposed via the ``gradient``
    output ``p * (1 - p)``.

    Attributes:
        n_classes: Number of classes ``K`` (>= 2).
    """

    def __init__(self, n_classes: int) -> None:
        """Initialize with the number of classes.

        Args:
            n_classes: Number of target classes ``K``. Must be an
                integer ``>= 2``.

        Raises:
            ValueError: If ``n_classes`` is not an integer ``>= 2``.
        """
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError(f"n_classes must be an integer >= 2, got {n_classes}")
        self.n_classes = n_classes

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable row-wise softmax.

        Subtracts the per-row maximum before exponentiation. This shift
        is mathematically a no-op (``softmax`` is shift-invariant) but
        prevents ``exp`` from overflowing for large logits.

        Args:
            z: Logits array of shape ``(n_samples, n_classes)``.

        Returns:
            Probability array of shape ``(n_samples, n_classes)`` with
            each row summing to 1.
        """
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return np.asarray(exp_z / np.sum(exp_z, axis=1, keepdims=True), dtype=float)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the categorical cross-entropy empirical risk.

        Args:
            y_true: Integer class labels in ``{0, …, K-1}``,
                shape ``(n_samples,)``.
            y_pred: Logits matrix, shape ``(n_samples, n_classes)``.

        Returns:
            Scalar cross-entropy loss.

        Raises:
            TypeError: For non-arrays or wrong dtypes.
            ValueError: For shape/rank mismatch, empty arrays, non-finite
                values, or labels outside ``[0, K-1]``.
        """
        if not isinstance(y_true, np.ndarray):
            raise TypeError(
                f"y_true must be a numpy.ndarray, got {type(y_true).__name__}"
            )
        if not isinstance(y_pred, np.ndarray):
            raise TypeError(
                f"y_pred must be a numpy.ndarray, got {type(y_pred).__name__}"
            )
        if y_true.ndim != 1:
            raise ValueError(f"y_true must be 1-D, got shape {y_true.shape}")
        if y_pred.ndim != 2 or y_pred.shape[1] != self.n_classes:
            raise ValueError(
                f"y_pred must have shape (n_samples, {self.n_classes}), got {y_pred.shape}"
            )
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Batch size mismatch: y_true {y_true.shape[0]} vs y_pred {y_pred.shape[0]}"
            )
        if y_true.size == 0:
            raise ValueError("Input arrays must not be empty.")
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Inputs contain NaN values.")
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            raise ValueError("Inputs contain infinite values.")
        if not np.issubdtype(y_true.dtype, np.integer):
            raise TypeError(f"y_true must be integer typed, got {y_true.dtype}")
        if np.any(y_true < 0) or np.any(y_true >= self.n_classes):
            raise ValueError(
                f"y_true labels must be in [0, {self.n_classes - 1}], got range "
                f"[{y_true.min()}, {y_true.max()}]"
            )

        p = self._softmax(y_pred)
        n = y_true.shape[0]
        log_p = np.log(p + 1e-15)
        return -float(np.sum(log_p[np.arange(n), y_true]) / n)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient of the empirical risk: ``(p - 1_y) / N``.

        Sums to zero across the class dimension for every sample,
        reflecting probability conservation.

        Args:
            y_true: Integer class labels, shape ``(n_samples,)``.
            y_pred: Logits matrix, shape ``(n_samples, n_classes)``.

        Returns:
            Gradient matrix of shape ``(n_samples, n_classes)``.

        Raises:
            TypeError: For non-arrays or wrong dtypes.
            ValueError: For rank/shape mismatch, empty arrays, non-finite
                values, or labels outside ``[0, K-1]``.
        """
        if not isinstance(y_true, np.ndarray):
            raise TypeError(
                f"y_true must be a numpy.ndarray, got {type(y_true).__name__}"
            )
        if not isinstance(y_pred, np.ndarray):
            raise TypeError(
                f"y_pred must be a numpy.ndarray, got {type(y_pred).__name__}"
            )
        if y_true.ndim != 1:
            raise ValueError(f"y_true must be 1-D, got shape {y_true.shape}")
        if y_pred.ndim != 2 or y_pred.shape[1] != self.n_classes:
            raise ValueError(
                f"y_pred must have shape (n_samples, {self.n_classes}), got {y_pred.shape}"
            )
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Batch size mismatch: y_true {y_true.shape[0]} vs y_pred {y_pred.shape[0]}"
            )
        if y_true.size == 0:
            raise ValueError("Input arrays must not be empty.")
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Inputs contain NaN values.")
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            raise ValueError("Inputs contain infinite values.")
        if not np.issubdtype(y_true.dtype, np.integer):
            raise TypeError(f"y_true must be integer typed, got {y_true.dtype}")
        if np.any(y_true < 0) or np.any(y_true >= self.n_classes):
            raise ValueError(
                f"y_true labels must be in [0, {self.n_classes - 1}], got range "
                f"[{y_true.min()}, {y_true.max()}]"
            )

        p = self._softmax(y_pred)
        n = y_true.shape[0]
        grad = p.copy()
        grad[np.arange(n), y_true] -= 1.0
        return np.asarray(grad / n, dtype=float)

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Hessian: block-diagonal ``(diag(p) - p p^T) / N`` per sample.

        Returns a dense ``(N, K, K)`` tensor; each block is positive
        semidefinite because ``diag(p) - p p^T`` is the Fisher
        information of the softmax model. The loop is intentionally
        kept sequential for clarity; the diagonal extraction in
        :class:`grnbt.boosting.MultiClassNewtonBoosting` is what the
        tree builder actually consumes.

        Args:
            y_true: Integer class labels, shape ``(n_samples,)``.
            y_pred: Logits matrix, shape ``(n_samples, n_classes)``.

        Returns:
            Hessian tensor of shape ``(n_samples, n_classes, n_classes)``.

        Raises:
            TypeError: For non-arrays or wrong dtypes.
            ValueError: For rank/shape mismatch, empty arrays, non-finite
                values, or labels outside ``[0, K-1]``.
        """
        if not isinstance(y_true, np.ndarray):
            raise TypeError(
                f"y_true must be a numpy.ndarray, got {type(y_true).__name__}"
            )
        if not isinstance(y_pred, np.ndarray):
            raise TypeError(
                f"y_pred must be a numpy.ndarray, got {type(y_pred).__name__}"
            )
        if y_true.ndim != 1:
            raise ValueError(f"y_true must be 1-D, got shape {y_true.shape}")
        if y_pred.ndim != 2 or y_pred.shape[1] != self.n_classes:
            raise ValueError(
                f"y_pred must have shape (n_samples, {self.n_classes}), got {y_pred.shape}"
            )
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Batch size mismatch: y_true {y_true.shape[0]} vs y_pred {y_pred.shape[0]}"
            )
        if y_true.size == 0:
            raise ValueError("Input arrays must not be empty.")
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Inputs contain NaN values.")
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            raise ValueError("Inputs contain infinite values.")
        if not np.issubdtype(y_true.dtype, np.integer):
            raise TypeError(f"y_true must be integer typed, got {y_true.dtype}")
        if np.any(y_true < 0) or np.any(y_true >= self.n_classes):
            raise ValueError(
                f"y_true labels must be in [0, {self.n_classes - 1}], got range "
                f"[{y_true.min()}, {y_true.max()}]"
            )

        p = self._softmax(y_pred)
        n = y_true.shape[0]
        k = self.n_classes
        hess = np.zeros((n, k, k))
        for i in range(n):
            pi = p[i]
            hess[i] = (np.diag(pi) - np.outer(pi, pi)) / n
        return np.asarray(hess, dtype=float)

    def hessian_lipschitz_constant(self) -> float:
        """Return ``M_0 = 1/4`` for categorical cross-entropy.

        Same bound as binary cross-entropy because the off-diagonal
        entries share the same derivative magnitude as the diagonal
        terms ``p_k(1 - p_k)``.
        """
        return 0.25

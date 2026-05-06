"""Loss functions for Gradient Regularized Newton Boosting Trees.

Implements MSE, Charbonnier, binary cross entropy, and categorical cross entropy
with pointwise gradients, Hessians, and analytically known Hessian Lipschitz
constants M_0 (used for Proposition 5.1 adaptive regularization).
"""

from abc import ABC, abstractmethod

import numpy as np


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate common loss-function inputs.

    Args:
        y_true: Ground-truth targets.
        y_pred: Model predictions.

    Raises:
        TypeError: If inputs are not NumPy arrays.
        ValueError: If arrays contain NaN/Inf or have mismatched shapes.
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
    """Abstract base class for a pointwise loss l(y_pred, y_true)."""

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the empirical average loss.

        Args:
            y_true: Ground-truth targets, shape (n_samples,) or (n_samples, n_classes).
            y_pred: Model predictions, same shape as y_true.

        Returns:
            Scalar empirical risk value.
        """

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute the pointwise gradient of the empirical risk w.r.t. y_pred.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.

        Returns:
            Gradient array with same shape as y_pred.
        """

    @abstractmethod
    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute the pointwise Hessian (or its diagonal) of the empirical risk.

        For scalar-output losses this returns an (n_samples,) array of second
        derivatives. For multi-class losses it returns a full block-diagonal
        representation.

        Args:
            y_true: Ground-truth targets.
            y_pred: Model predictions.

        Returns:
            Hessian representation compatible with the tree builder.
        """

    @abstractmethod
    def hessian_lipschitz_constant(self) -> float:
        """Return M_0, the pointwise Hessian Lipschitz constant.

        This is used by Proposition 5.1 to scale the adaptive regularization:
        M = M_0 * sqrt(n_samples).

        Returns:
            Non-negative scalar M_0.
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
    """Mean squared error loss."""

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MSE empirical risk.

        Formula: (1/N) * sum((y_true - y_pred)^2).
        """
        _validate_inputs(y_true, y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient: 2 (y_pred - y_true) / N.

        This is the derivative of the empirical risk, not the per-sample loss.
        """
        _validate_inputs(y_true, y_pred)
        return np.asarray(2.0 * (y_pred - y_true) / y_true.shape[0], dtype=float)

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Hessian: constant 2 / N.

        The second derivative is constant for MSE, which implies M_0 = 0.
        """
        _validate_inputs(y_true, y_pred)
        return np.asarray(np.full_like(y_true, 2.0 / y_true.shape[0]), dtype=float)

    def hessian_lipschitz_constant(self) -> float:
        """M_0 = 0 because the Hessian is constant.

        A constant Hessian has zero Lipschitz constant since it does not change
        with the prediction.
        """
        return 0.0


class CharbonnierLoss(Loss):
    """Charbonnier loss: l(d) = sqrt(1 + d^2) - 1, where d = y_pred - y_true.

    A smooth, convex alternative to absolute loss that is differentiable
    everywhere. The Hessian is Lipschitz with constant M_0 = 1.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Charbonnier empirical risk using numerically stable hypot."""
        _validate_inputs(y_true, y_pred)
        d = y_pred - y_true
        return float(np.mean(np.hypot(1.0, d) - 1.0))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient: d / (sqrt(1 + d^2) * N).

        Derivative of the empirical Charbonnier risk w.r.t. predictions.
        """
        _validate_inputs(y_true, y_pred)
        d = y_pred - y_true
        denom = np.hypot(1.0, d) * y_true.shape[0]
        return np.asarray(d / denom, dtype=float)

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Hessian: 1 / ((1 + d^2)^{3/2} * N).

        Always positive, ensuring convexity. The Hessian is bounded by 1/N
        and attains its maximum at d = 0.
        """
        _validate_inputs(y_true, y_pred)
        d = y_pred - y_true
        hypot_d = np.hypot(1.0, d)
        return np.asarray(1.0 / (hypot_d**3 * y_true.shape[0]), dtype=float)

    def hessian_lipschitz_constant(self) -> float:
        """M_0 = 1 (analytically derived from the Charbonnier formula).

        The third derivative of the Charbonnier loss is bounded by 1 in
        absolute value, giving a Hessian Lipschitz constant of 1.
        """
        return 1.0


class BinaryCrossEntropyLoss(Loss):
    """Binary cross-entropy (logistic) loss on logits.

    Operates on raw logits (not probabilities). The labels y_true must be
    in {0, 1}. The Hessian is p(1-p)/N where p = sigmoid(y_pred), bounded
    by 1/(4N), giving M_0 = 1/4.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """BCE empirical risk.

        Uses log-sum-exp stability trick internally via sigmoid.
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
        """Gradient: (sigma(y_pred) - y_true) / N."""
        _validate_inputs(y_true, y_pred)
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true for BCE must contain only 0 and 1.")
        p = 1.0 / (1.0 + np.exp(-y_pred))
        return np.asarray((p - y_true) / y_true.shape[0], dtype=float)

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Hessian: p (1 - p) / N.

        Strictly positive for finite logits, ensuring strict convexity.
        """
        _validate_inputs(y_true, y_pred)
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true for BCE must contain only 0 and 1.")
        p = 1.0 / (1.0 + np.exp(-y_pred))
        return np.asarray((p * (1.0 - p)) / y_true.shape[0], dtype=float)

    def hessian_lipschitz_constant(self) -> float:
        """M_0 = 1/4 for binary cross-entropy (see Appendix A of paper).

        The derivative of p(1-p) is p(1-p)(1-2p), bounded by 1/4 in absolute
        value.
        """
        return 0.25


class CategoricalCrossEntropyLoss(Loss):
    """Categorical cross-entropy (softmax) loss on logits.

    The Hessian is returned as a full (n_samples, K, K) block-diagonal array,
    where K is the number of classes. This is exact but O(N K^2) in memory.
    For large K, a block-diagonal operator would be more efficient.
    """

    def __init__(self, n_classes: int) -> None:
        """Initialize with number of classes.

        Args:
            n_classes: Number of target classes K. Must be >= 2.

        Raises:
            ValueError: If n_classes < 2.
        """
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError(f"n_classes must be an integer >= 2, got {n_classes}")
        self.n_classes = n_classes

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax.

        Subtracts the per-row maximum before exponentiation to avoid overflow.

        Args:
            z: Logits array of shape (n_samples, n_classes).

        Returns:
            Probability array of shape (n_samples, n_classes).
        """
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return np.asarray(exp_z / np.sum(exp_z, axis=1, keepdims=True), dtype=float)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """CCE empirical risk.

        Args:
            y_true: Integer class labels in {0, ..., K-1}, shape (n_samples,).
            y_pred: Logits matrix, shape (n_samples, n_classes).

        Returns:
            Scalar cross-entropy loss.
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
        """Gradient: (p - one_hot(y)) / N.

        Args:
            y_true: Integer class labels, shape (n_samples,).
            y_pred: Logits matrix, shape (n_samples, n_classes).

        Returns:
            Gradient matrix of shape (n_samples, n_classes).
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
        """Hessian: block-diagonal (diag(p) - p p^T) / N per sample.

        Args:
            y_true: Integer class labels, shape (n_samples,).
            y_pred: Logits matrix, shape (n_samples, n_classes).

        Returns:
            Hessian array of shape (n_samples, n_classes, n_classes).
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
        """M_0 = 1/4 for categorical cross-entropy.

        Same bound as binary cross-entropy since each class probability
        p_k(1-p_k) has derivative bounded by 1/4.
        """
        return 0.25

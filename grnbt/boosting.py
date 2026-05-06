"""Boosting engines: Vanilla Newton Boosting and Gradient Regularized Newton Boosting.

Both follow the Restricted Newton Descent framework from the paper:
  1. Compute gradient g_k and Hessian h_k at current ensemble F_k.
  2. Fit a NewtonTree weak learner minimizing the second-order surrogate.
  3. Update F_{k+1} = F_k + η * f_weak.

GRN differs only in the adaptive regularization:
  λ_k = λ_base + sqrt(M * ||g_k||)
where M = M_0 * sqrt(N) per Proposition 5.1.
"""

from typing import Optional

import numpy as np

from grnbt.losses import Loss
from grnbt.tree import NewtonTree
from grnbt.utils import History


class BaseBoosting:
    """Base class for Newton boosting engines.

    Attributes:
        loss: Loss function instance.
        n_estimators: Number of boosting iterations.
        learning_rate: Step-size η (learning rate).
        max_depth: Maximum depth of each weak learner tree.
        min_samples_leaf: Minimum samples per leaf.
        lam_base: Static L2 regularization λ_base.
        verbose: Whether to print iteration progress.
        trees: List of fitted NewtonTree instances.
        F0: Initial prediction (scalar or vector).
        history: Training history container.
    """

    def __init__(
        self,
        loss: Loss,
        n_estimators: int = 100,
        learning_rate: float = 1.0,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        lam_base: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """Initialize boosting hyperparameters.

        Args:
            loss: Loss function with gradient, hessian, and M_0.
            n_estimators: Number of boosting rounds. Must be >= 1.
            learning_rate: Learning rate η (default 1.0 as in paper). Must be > 0.
            max_depth: Max depth of each weak learner. Must be >= 0.
            min_samples_leaf: Minimum samples in a leaf node. Must be >= 1.
            lam_base: Base L2 regularization (static component). Must be >= 0.
            verbose: If True, print loss every 10 iterations.

        Raises:
            TypeError: If loss is not a Loss instance.
            ValueError: If any hyperparameter is out of range.
        """
        if not isinstance(loss, Loss):
            raise TypeError(f"loss must be a Loss instance, got {type(loss).__name__}")
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError(
                f"max_depth must be a non-negative integer, got {max_depth}"
            )
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
        if lam_base < 0:
            raise ValueError(f"lam_base must be non-negative, got {lam_base}")

        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.lam_base = lam_base
        self.verbose = verbose
        self.trees: list[NewtonTree] = []
        self.F0: Optional[np.ndarray] = None
        self.history = History()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaseBoosting":
        """Fit the boosting ensemble.

        Args:
            x: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).

        Returns:
            Self, for chaining.

        Raises:
            TypeError: If inputs are not NumPy arrays.
            ValueError: If shapes mismatch, arrays are empty, or contain NaN/Inf.
        """
        self._validate_fit_inputs(x, y)
        self.F0 = self._init_prediction(y)
        f_current = self.F0.copy()
        n = x.shape[0]
        for k in range(self.n_estimators):
            g = self.loss.gradient(y, f_current)
            h = self.loss.hessian(y, f_current)
            lam_k = self._compute_lambda(g, h, n)
            tree = NewtonTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(x, g, h, lam_k)
            f_weak = tree.predict(x)
            f_current += self.learning_rate * f_weak
            self.trees.append(tree)
            loss_val = self.loss.loss(y, f_current)
            self.history.log("loss", loss_val)
            self.history.log("lambda_k", lam_k)
            self.history.log("grad_norm", float(np.linalg.norm(g)))
            if self.verbose and k % 10 == 0:
                print(f"Iter {k}: loss={loss_val:.6f} lambda={lam_k:.6f}")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict on new data.

        Args:
            x: Feature matrix of shape (n_samples, n_features).

        Returns:
            Prediction vector of shape (n_samples,).

        Raises:
            ValueError: If the model has not been fitted.
            TypeError: If x is not a NumPy array.
            ValueError: If x is not 2-D.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy.ndarray, got {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if self.F0 is None:
            raise ValueError(
                "Model has not been fitted yet. Call fit() before predict()."
            )

        # F0 is stored as the full initial prediction vector for the training set.
        # For prediction on new data we assume a constant initial prediction
        # equal to the first element (the common case when F0 is constant).
        f_out: np.ndarray
        if np.isscalar(self.F0):
            f_out = np.full(x.shape[0], float(self.F0), dtype=float)
        else:
            # Use the mean of F0 as the constant baseline for new samples.
            baseline = float(np.mean(self.F0))
            f_out = np.full(x.shape[0], baseline, dtype=float)

        for tree in self.trees:
            f_out += self.learning_rate * tree.predict(x)
        return np.asarray(f_out, dtype=float)

    def _init_prediction(self, y: np.ndarray) -> np.ndarray:
        """Initialize the ensemble prediction.

        Default is zeros; subclasses may override (e.g., mean for MSE).

        Args:
            y: Target vector.

        Returns:
            Initial prediction array with same shape as y.
        """
        result: np.ndarray = np.zeros_like(y, dtype=float)
        return np.asarray(result, dtype=float)

    def _compute_lambda(self, g: np.ndarray, h: np.ndarray, n: int) -> float:
        """Compute the per-iteration regularization λ_k.

        Args:
            g: Gradient vector.
            h: Hessian vector.
            n: Number of samples.

        Returns:
            Scalar regularization value.
        """
        raise NotImplementedError

    def _validate_fit_inputs(self, x: np.ndarray, y: np.ndarray) -> None:
        """Validate inputs to fit().

        Raises:
            TypeError: For wrong input types.
            ValueError: For invalid shapes, empty data, or NaN/Inf.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy.ndarray, got {type(x).__name__}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be a numpy.ndarray, got {type(y).__name__}")
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: x {x.shape[0]} vs y {y.shape[0]}")
        if x.shape[0] == 0:
            raise ValueError("Cannot fit on empty data (n_samples = 0).")
        if x.shape[1] == 0:
            raise ValueError("Cannot fit on data with zero features (n_features = 0).")
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            raise ValueError("Inputs contain NaN values.")
        if np.any(np.isinf(x)) or np.any(np.isinf(y)):
            raise ValueError("Inputs contain infinite values.")


class VanillaNewtonBoosting(BaseBoosting):
    """Vanilla Restricted Newton Boosting (Algorithm 1).

    Uses a static regularization λ_k = λ_base.
    """

    def _compute_lambda(self, g: np.ndarray, h: np.ndarray, n: int) -> float:
        """Static regularization.

        Returns the base regularization without any adaptive component.
        """
        del g, h, n  # Unused in vanilla variant.
        return self.lam_base


class GradientRegularizedNewtonBoosting(BaseBoosting):
    """Gradient Regularized Newton Boosting (Algorithm 2).

    Adaptive regularization per iteration:
        λ_k = λ_base + sqrt(M * ||g_k||)
    where M = M_0 * sqrt(N) from Proposition 5.1.
    """

    def _compute_lambda(self, g: np.ndarray, h: np.ndarray, n: int) -> float:
        """Adaptive regularization based on gradient norm.

        The additional term grows with the gradient magnitude, providing
        stronger regularization when the iterate is far from optimality.
        """
        del h  # Hessian not needed for λ_k computation.
        m = self.loss.empirical_risk_lipschitz(n)
        grad_norm = float(np.linalg.norm(g))
        lam_adaptive = np.sqrt(m * grad_norm)
        return float(self.lam_base + lam_adaptive)

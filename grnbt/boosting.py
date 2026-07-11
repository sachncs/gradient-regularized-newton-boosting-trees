"""Boosting engines for GRNBT.

This module implements the *Newton boosting engines* that drive training:

* :class:`VanillaNewtonBoosting` — Algorithm 1 of the paper; static
  regularization ``λ_k = λ_base``.
* :class:`GradientRegularizedNewtonBoosting` — Algorithm 2; adaptive
  ``λ_k = λ_base + sqrt(M * ||g_k||)`` with
  ``M = M_0 * sqrt(N)`` (Proposition 5.1).
* :class:`MultiClassNewtonBoosting` — K-class extension with softmax
  output and vector-valued trees.

All engines share the **Restricted Newton Descent** framework:

    for k in 0, ..., K-1:
        g = loss.gradient(y, F_k)        # gradient of empirical risk
        h = loss.hessian(y, F_k)         # Hessian (or its diagonal)
        λ_k = compute_lambda(g, h, N)    # static or adaptive
        tree.fit(x, g, h, λ_k)
        F_{k+1} = F_k + η * tree.predict(x)

The differences between engines live only in (a) how ``λ_k`` is set and
(b) how the predictions are post-processed (softmax for multi-class).

Architectural notes
-------------------

* ``BaseBoosting`` captures the shared training loop. Subclasses only
  override ``_compute_lambda``; the multi-class subclass additionally
  overrides ``fit`` and ``predict`` because of the multi-output nature
  of ``F``.
* The initial prediction ``F0`` is stored as the **constant** baseline
  used at predict-time so that the model can be applied to new data of
  arbitrary size.
* A ``History`` instance logs ``loss``, ``λ_k`` and ``grad_norm`` at every
  iteration. This is the canonical way to reproduce paper figures.

References
----------

Zozoulenko et al. (2026), Algorithm 1, Algorithm 2, and Proposition 5.1.
"""

from typing import Optional

import numpy as np

from grnbt.losses import Loss
from grnbt.tree import MultiClassNewtonTree, NewtonTree
from grnbt.utils import History


class BaseBoosting:
    """Base class for Newton boosting engines.

    Holds the training loop and prediction logic shared by all
    sub-classes. Subclasses must implement ``_compute_lambda`` to
    decide the per-iteration regularization ``λ_k``.

    Lifecycle:

    1. ``__init__`` — validate and store hyperparameters.
    2. ``fit(x, y)`` — run the training loop, populating ``self.trees``
       and ``self.history``.
    3. ``predict(x)`` — replay the ensemble on new data.

    Attributes:
        loss: Loss function used for gradients/Hessians.
        n_estimators: Number of boosting iterations ``K``.
        learning_rate: Step size ``η``.
        max_depth: Maximum depth of each weak tree.
        min_samples_leaf: Minimum samples per leaf.
        lam_base: Static ``λ_base`` component.
        verbose: If ``True``, prints iteration metrics every 10 rounds.
        trees: List of fitted :class:`NewtonTree` (or
            :class:`MultiClassNewtonTree`) instances.
        F0: Initial prediction (scalar or vector). Set by
            :meth:`fit`; ``None`` before fitting.
        history: :class:`grnbt.utils.History` instance recording per-round
            ``loss``, ``lambda_k``, and ``grad_norm`` scalars.

    Thread safety:
        Instances are not thread-safe; create separate instances for
        parallel training runs.
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
            loss: Loss function with first/second-order primitives and
                ``M_0`` (the Hessian Lipschitz constant).
            n_estimators: Number of boosting rounds. Must be ``>= 1``.
            learning_rate: Learning rate ``η`` (default ``1.0`` as in the
                paper). Must be ``> 0``.
            max_depth: Max depth of each weak learner. Must be ``>= 0``.
            min_samples_leaf: Minimum samples in a leaf node. Must be
                ``>= 1``.
            lam_base: Base L2 regularization (static component). Must be
                ``>= 0``. Use ``0.0`` for the paper's "no static
                regularization" baseline.
            verbose: If ``True``, print loss and ``λ_k`` every 10
                iterations; useful for live monitoring.

        Raises:
            TypeError: If ``loss`` is not a :class:`Loss` instance.
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
        """Fit the scalar-output boosting ensemble.

        Implements Algorithm 1 of the paper with the loop
        ::

            for k in range(n_estimators):
                g = loss.gradient(y, F_k)
                h = loss.hessian(y, F_k)
                λ_k = compute_lambda(g, h, N)
                tree = NewtonTree(...).fit(x, g, h, λ_k)
                F_{k+1} = F_k + η * tree.predict(x)

        The initial ``F_0`` is set by :meth:`_init_prediction` (default
        zero vector). Loss, ``λ_k`` and ``grad_norm`` are logged at every
        iteration.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.

        Returns:
            Self, for chaining.

        Raises:
            TypeError: If ``x`` or ``y`` is not a NumPy array.
            ValueError: If shapes mismatch, arrays are empty, or any
                element is ``NaN`` / ``Inf``.
        """
        self._validate_fit_inputs(x, y)
        self.F0 = self._init_prediction(y)
        f_current = self.F0.copy()
        n = x.shape[0]
        for k in range(self.n_estimators):
            # 1. Compute empirical-risk first/second-order information.
            g = self.loss.gradient(y, f_current)
            h = self.loss.hessian(y, f_current)
            # 2. Decide λ_k (override point for GRN vs. vanilla).
            lam_k = self._compute_lambda(g, h, n)
            # 3. Fit a depth-limited NewtonTree to the surrogate.
            tree = NewtonTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(x, g, h, lam_k)
            # 4. Apply the weak learner with step size η.
            f_weak = tree.predict(x)
            f_current += self.learning_rate * f_weak
            self.trees.append(tree)
            # 5. Log scalar metrics for plotting/diagnostics.
            loss_val = self.loss.loss(y, f_current)
            self.history.log("loss", loss_val)
            self.history.log("lambda_k", lam_k)
            self.history.log("grad_norm", float(np.linalg.norm(g)))
            if self.verbose and k % 10 == 0:
                print(f"Iter {k}: loss={loss_val:.6f} lambda={lam_k:.6f}")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict on new data.

        Replays the fitted ensemble on ``x`` by starting from a
        constant baseline (the mean of ``self.F0`` when stored as an
        array) and adding each tree's prediction scaled by ``η``.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Prediction vector of shape ``(n_samples,)``.

        Raises:
            TypeError: If ``x`` is not a NumPy array.
            ValueError: If ``x`` is not 2-D, or if the model has not
                been fitted yet.
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

        The default is a zero vector. Subclasses may override
        (e.g., setting ``F_0`` to the mean of ``y`` for MSE warm-start).
        The returned array is the *training-set* baseline; only its
        shape is preserved for ``predict``, while its scalar
        ``mean`` is used as the new-sample baseline.

        Args:
            y: Target vector, shape ``(n_samples,)``.

        Returns:
            Initial prediction array with the same shape as ``y``.
        """
        result: np.ndarray = np.zeros_like(y, dtype=float)
        return np.asarray(result, dtype=float)

    def _compute_lambda(self, g: np.ndarray, h: np.ndarray, n: int) -> float:
        """Compute the per-iteration regularization ``λ_k``.

        Subclasses override this method: vanilla boosting returns the
        static ``self.lam_base``; GRN adds an adaptive
        ``sqrt(M * ||g||)`` term where ``M = M_0 * sqrt(N)``.

        Args:
            g: Gradient vector at iterate ``F_k``.
            h: Hessian vector at iterate ``F_k``. Unused in the
                default vanilla implementation but kept in the signature
                for symmetry with adaptive variants.
            n: Number of training samples ``N``.

        Returns:
            Scalar regularization value ``λ_k >= 0``.
        """
        raise NotImplementedError

    def _validate_fit_inputs(self, x: np.ndarray, y: np.ndarray) -> None:
        """Validate the inputs to :meth:`fit`.

        Raises:
            TypeError: If either input is not a NumPy array.
            ValueError: On rank mismatch (``x`` must be 2-D, ``y`` 1-D),
                on inconsistent sample counts, on empty data, on
                zero-feature data, or on ``NaN`` / ``Inf``.
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

    Uses a static per-iteration regularization ``λ_k = λ_base``. This is
    the "textbook" Newton boosting baseline against which the paper
    benchmarks GRN: it diverges on losses whose Hessian depends on the
    prediction (e.g., Charbonnier) and converges only under strong-
    convexity / smoothness assumptions.
    """

    def _compute_lambda(self, g: np.ndarray, h: np.ndarray, n: int) -> float:
        """Return the static ``λ_base`` unchanged for every iteration.

        Args:
            g: Gradient (unused).
            h: Hessian (unused).
            n: Number of samples (unused).

        Returns:
            ``self.lam_base`` (a non-negative scalar).
        """
        del g, h, n  # Unused in vanilla variant.
        return self.lam_base


class GradientRegularizedNewtonBoosting(BaseBoosting):
    """Gradient Regularized Newton Boosting (Algorithm 2).

    Adaptive per-iteration regularization:

        λ_k = λ_base + sqrt(M * ||g_k||_H)

    where ``M = M_0 * sqrt(N)`` per Proposition 5.1 and ``||g_k||_H``
    is the empirical ``L^2`` (RMS) norm of the gradient. The adaptive
    term grows with the gradient magnitude — i.e., when the iterate is
    far from the optimum — and decays toward ``λ_base`` as optimization
    progresses. This yields the paper's ``O(1/k^2)`` global rate for
    convex losses with Lipschitz Hessian.
    """

    def _compute_lambda(self, g: np.ndarray, h: np.ndarray, n: int) -> float:
        """Compute the adaptive ``λ_k`` from Proposition 5.1.

        Args:
            g: Gradient vector at iterate ``F_k``. Used for its norm.
            h: Hessian vector (unused; kept for API symmetry).
            n: Number of training samples.

        Returns:
            ``λ_k = λ_base + sqrt(M * ||g||)`` with ``M = M_0 * sqrt(N)``.
            A Python float.
        """
        del h  # Hessian not needed for λ_k computation.
        m = self.loss.empirical_risk_lipschitz(n)
        grad_norm = float(np.linalg.norm(g))
        lam_adaptive = np.sqrt(m * grad_norm)
        return float(self.lam_base + lam_adaptive)


class MultiClassNewtonBoosting(BaseBoosting):
    """Multi-class Newton Boosting with gradient regularization support.

    Builds K-class classification models using Newton-type gradient
    boosting. Each boosting round fits a **single** tree with
    K-dimensional leaf weights (see :class:`MultiClassNewtonTree`); all
    classes share the same split structure but have independent
    per-class leaf weights. The split criterion sums Newton gains
    across all ``K`` classes.

    The ensemble output is a logits matrix ``F`` of shape
    ``(n_samples, K)``. Predictions can be obtained either as
    probabilities (default) or as raw logits via ``softmax_output``.

    Like :class:`BaseBoosting`, the only difference between vanilla
    multi-class and GRN multi-class is the per-iteration ``λ_k``
    recipe, controlled by :meth:`_compute_lambda_for_multiclass`.

    Attributes:
        n_classes: Number of target classes ``K``.
        softmax_output: If ``True`` (default), :meth:`predict` returns
            softmax probabilities; if ``False``, it returns raw logits.
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
        n_classes: int = 2,
        softmax_output: bool = True,
    ) -> None:
        """Initialize multi-class boosting hyperparameters.

        Args:
            loss: Loss function with first/second-order primitives and
                ``M_0``.
            n_estimators: Number of boosting rounds. Must be ``>= 1``.
            learning_rate: Learning rate ``η``. Must be ``> 0``.
            max_depth: Max depth of each weak learner. Must be ``>= 0``.
            min_samples_leaf: Minimum samples in a leaf. Must be ``>= 1``.
            lam_base: Base L2 regularization. Must be ``>= 0``.
            verbose: If ``True``, print loss every 10 iterations.
            n_classes: Number of classes ``K``. Must be ``>= 2``.
            softmax_output: If ``True``, :meth:`predict` returns
                probabilities; if ``False``, it returns logits.

        Raises:
            TypeError: If ``loss`` is not a :class:`Loss` instance.
            ValueError: If any hyperparameter is out of range.
        """
        super().__init__(
            loss=loss,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            lam_base=lam_base,
            verbose=verbose,
        )
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.n_classes = n_classes
        self.softmax_output = softmax_output
        self.trees: list[MultiClassNewtonTree] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MultiClassNewtonBoosting":
        """Fit the multi-class boosting ensemble.

        Mirrors :meth:`BaseBoosting.fit` but threads ``K`` parallel
        logits through every iteration. In each round:

        1. ``g ∈ (N, K)`` and ``H ∈ (N, K, K)`` are computed at the
           current logits ``F_k``.
        2. The Hessian is reduced to its per-class diagonal
           ``h_diag ∈ (N, K)`` for the tree builder.
        3. ``λ_k`` is computed via :meth:`_compute_lambda_for_multiclass`.
        4. A single :class:`MultiClassNewtonTree` fits
           ``(x, g, h_diag, λ_k)`` and produces a ``K``-dimensional
           weak learner.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.
            y: Integer class labels in ``{0, …, K-1}``,
                shape ``(n_samples,)``.

        Returns:
            Self, for chaining.

        Raises:
            TypeError: If ``x`` or ``y`` is not a NumPy array, or if
                ``y`` is not integer-typed.
            ValueError: On shape mismatch, empty data, out-of-range
                labels, or non-finite values.
        """
        self._validate_fit_inputs(x, y)
        self._validate_multiclass_labels(y)
        n = x.shape[0]
        self.F0 = np.zeros((n, self.n_classes), dtype=float)
        f_current = self.F0.copy()

        for k in range(self.n_estimators):
            g = self.loss.gradient(y, f_current)
            h = self.loss.hessian(y, f_current)
            # g is (n, K); use the full Frobenius norm for λ scaling.
            g_norm = float(np.linalg.norm(g))
            # Tree builder consumes per-class Hessians, so we extract
            # the diagonal of the (n, K, K) block Hessian.
            h_diag = self._extract_hessian_diagonal(h)
            lam_k = self._compute_lambda_for_multiclass(g_norm, h_diag, n)

            tree = MultiClassNewtonTree(
                n_classes=self.n_classes,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(x, g, h_diag, lam_k)
            f_weak = tree.predict(x)
            f_current = f_current + self.learning_rate * f_weak
            self.trees.append(tree)
            loss_val = self.loss.loss(y, f_current)
            self.history.log("loss", loss_val)
            self.history.log("lambda_k", lam_k)
            self.history.log("grad_norm", g_norm)
            if self.verbose and k % 10 == 0:
                print(f"Iter {k}: loss={loss_val:.6f} lambda={lam_k:.6f}")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict on new data.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            If ``softmax_output`` is ``True``: probability matrix of
            shape ``(n_samples, n_classes)`` (rows sum to 1).
            If ``softmax_output`` is ``False``: raw logits matrix of
            shape ``(n_samples, n_classes)``.

        Raises:
            TypeError: If ``x`` is not a NumPy array.
            ValueError: If ``x`` is not 2-D, or if the model has not
                been fitted.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy.ndarray, got {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if self.F0 is None:
            raise ValueError(
                "Model has not been fitted yet. Call fit() before predict()."
            )

        # Multi-class baseline is the zero-logits matrix (uniform
        # softmax probability across classes).
        f_out = np.zeros((x.shape[0], self.n_classes), dtype=float)
        for tree in self.trees:
            f_out = f_out + self.learning_rate * tree.predict(x)

        if self.softmax_output:
            return self._softmax(f_out)
        return f_out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Equivalent to :meth:`predict` with ``softmax_output=True`` and
        always returns probabilities (it applies softmax if necessary).

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Probability matrix of shape ``(n_samples, n_classes)`` with
            each row summing to 1.
        """
        logits = self.predict(x)
        if self.softmax_output:
            return logits
        return self._softmax(logits)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable row-wise softmax.

        Subtracts the per-row maximum before exponentiation to avoid
        overflow for large logits.

        Args:
            z: Logits matrix of shape ``(n_samples, n_classes)``.

        Returns:
            Probability matrix of shape ``(n_samples, n_classes)``
            whose rows sum to 1.
        """
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return np.asarray(exp_z / np.sum(exp_z, axis=1, keepdims=True), dtype=float)

    def _extract_hessian_diagonal(self, h: np.ndarray) -> np.ndarray:
        """Extract the diagonal of a (possibly block) Hessian.

        For scalar Hessians (i.e. when ``n_classes == 1``) the input is
        already 2-D and is returned unchanged. For multi-class losses
        the Hessian is a block-diagonal ``(N, K, K)`` operator and the
        diagonal is consumed by the tree builder.

        Args:
            h: Hessian array of shape ``(n_samples, n_classes)`` or
                ``(n_samples, n_classes, n_classes)``.

        Returns:
            Per-sample diagonal of shape ``(n_samples, n_classes)``.
        """
        if h.ndim == 2:
            return h
        if h.ndim == 3:
            return np.asarray(np.diagonal(h, axis1=1, axis2=2), dtype=float)
        raise ValueError(f"Unexpected Hessian shape: {h.shape}")

    def _compute_lambda_for_multiclass(
        self, grad_norm: float, h_diag: np.ndarray, n: int
    ) -> float:
        """Compute ``λ_k`` for the multi-class iteration.

        Args:
            grad_norm: Frobenius norm ``||g_k||`` over the
                ``(n, K)`` gradient matrix.
            h_diag: Per-class diagonal Hessian, shape ``(n, K)``.
                Currently unused; kept for future variants that scale
                by ``sum(h_diag)``.
            n: Number of samples.

        Returns:
            ``λ_k = λ_base + sqrt(M * ||g||)`` where
            ``M = M_0 * sqrt(N)`` per Proposition 5.1.
        """
        del h_diag  # Reserved for future Hessian-aware λ scaling.
        m = self.loss.empirical_risk_lipschitz(n)
        lam_adaptive = np.sqrt(m * grad_norm)
        return float(self.lam_base + lam_adaptive)

    def _validate_multiclass_labels(self, y: np.ndarray) -> None:
        """Validate that ``y`` contains valid integer multi-class labels.

        Raises:
            TypeError: If ``y`` is not an integer-typed array.
            ValueError: If any label lies outside ``[0, n_classes - 1]``.
        """
        if not np.issubdtype(y.dtype, np.integer):
            raise TypeError(f"y must be integer typed for multi-class, got {y.dtype}")
        if np.any(y < 0) or np.any(y >= self.n_classes):
            raise ValueError(
                f"y labels must be in [0, {self.n_classes - 1}], "
                f"got range [{y.min()}, {y.max()}]"
            )

    def _validate_fit_inputs(self, x: np.ndarray, y: np.ndarray) -> None:
        """Validate the inputs to multi-class :meth:`fit`.

        In addition to the standard checks, ``y`` is validated to be
        integer-typed (since :class:`CategoricalCrossEntropyLoss`
        consumes integer class labels).

        Raises:
            TypeError: If either input is not a NumPy array.
            ValueError: On rank mismatch (``x`` 2-D, ``y`` 1-D),
                inconsistent sample counts, empty data, zero features,
                or non-finite values.
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
        if np.any(np.isnan(x)) or np.any(np.isnan(y.astype(float))):
            raise ValueError("Inputs contain NaN values.")
        if np.any(np.isinf(x)):
            raise ValueError("Inputs contain infinite values.")

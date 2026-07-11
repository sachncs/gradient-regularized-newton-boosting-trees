"""Newton-boosting decision tree with exact greedy split finding.

This module implements the weak learner used by the boosting engines in
``grnbt.boosting``. The trees are *axis-aligned* decision trees fit to a
*second-order* surrogate of the empirical risk:

    Q_k(f) = sum_i (g_i * f(x_i) + 0.5 * h_i * f(x_i)^2) + (λ/2) * ||f||^2

Leaf weights are closed-form optimal for this quadratic surrogate:

    w_leaf = - sum_{i in leaf} g_i / (sum_{i in leaf} h_i + λ)

Split quality is measured by the **exact Newton gain** from the paper:

    gain = 0.5 * [ (G_L)^2 / (H_L + λ)
                 + (G_R)^2 / (H_R + λ)
                 - (G_total)^2 / (H_total + λ) ]

Two tree variants are provided:

* :class:`NewtonTree`            — scalar-output regression / binary
                                   classification; leaf weight is a scalar.
* :class:`MultiClassNewtonTree`  — vector-valued outputs with K classes;
                                   leaf weight is a K-dimensional vector and
                                   split gain sums across classes.

Design notes
------------

* The trees are the "idealized" weak learner from the paper's analysis
  (exact greedy search over every threshold in the current node). For
  fast approximation, see :class:`grnbt.extensions.histogram_tree.
  HistogramNewtonTree`.
* ``min_gain = 1e-7`` defaults to a tiny non-zero value to discard
  numerically meaningless splits while keeping the algorithm eager.
* ``1e-12`` epsilons are added to leaf denominators (``sum(h) + λ``)
  to guard against pathological zero-Hessian cases.
* Constant features are skipped because no valid split exists on them.

Algorithm complexity
--------------------

For one tree with ``N`` samples, ``D`` features and depth ``d``:
* O(N log N * D) per inner node for sorting, but the cumulative sums are
  reused, giving an effective O(N * D * d) cost.
* O(N) memory for the leaf weight computation.

References
----------

Zozoulenko et al. (2026), Section 5 (weak learner equations) and Appendix B
(pseudocode for greedy split search).

Examples
--------

>>> import numpy as np
>>> from grnbt.tree import NewtonTree
>>> rng = np.random.RandomState(0)
>>> x = rng.randn(20, 3)
>>> g = rng.randn(20)
>>> h = np.ones(20)
>>> tree = NewtonTree(max_depth=2)
>>> tree.fit(x, g, h, lam=0.5)
<...>
>>> tree.predict(x).shape
(20,)
"""

from typing import Optional

import numpy as np


class _Node:
    """Recursive scalar-output tree node.

    ``_Node`` is the minimal mutable record used by :class:`NewtonTree`.
    It is intentionally kept private (no public API exposure) because the
    tree itself is the public interface; consumers should use
    :meth:`NewtonTree.predict` rather than walking nodes directly.

    Attributes:
        is_leaf: ``True`` for terminal nodes, ``False`` for split nodes.
        weight: Scalar leaf weight set to the closed-form Newton value
            ``-sum(g) / (sum(h) + λ)``. Ignored for split nodes.
        feature_idx: Index of the feature used for splitting (``None``
            for leaves).
        threshold: Split threshold (samples with
            ``x[feature_idx] <= threshold`` go left). ``None`` for leaves.
        left: Left child (samples passing the threshold test).
        right: Right child.
    """

    def __init__(self, is_leaf: bool = True) -> None:
        self.is_leaf = is_leaf
        self.weight: float = 0.0
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None


class NewtonTree:
    """Decision tree weak learner optimized for Newton boosting.

    Builds a greedy axis-aligned tree by exhaustively searching for the
    threshold that maximizes the second-order Newton gain. Leaf
    weights are closed-form optimal for the quadratic surrogate that
    the boosting engine optimizes.

    The tree is stateless across calls: :meth:`fit` replaces the
    existing ``root`` and :meth:`predict` is read-only. Two :meth:`fit`
    calls with different ``(g, h, λ)`` produce independent trees.

    Attributes:
        max_depth: Maximum tree depth (root is depth ``0``).
        min_samples_leaf: Minimum samples required in each leaf.
        min_gain: Minimum gain required to accept a split.
        root: Fitted root node; ``None`` before :meth:`fit` is called.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        min_gain: float = 1e-7,
    ) -> None:
        """Initialize tree hyperparameters.

        Args:
            max_depth: Maximum depth of the tree. Root depth is ``0``,
                so ``max_depth=0`` produces a single-leaf tree.
                Must be ``>= 0``.
            min_samples_leaf: Minimum number of samples in each leaf
                after a split. Must be ``>= 1``. Higher values reduce
                overfitting but may starve leaves of data.
            min_gain: Minimum gain required to accept a split. Splits
                whose closed-form Newton gain is below this threshold
                are discarded. Must be ``>= 0``. A small positive value
                filters out noise-driven splits.

        Raises:
            ValueError: If any hyperparameter is out of range or has
                the wrong Python type.
        """
        if not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError(
                f"max_depth must be a non-negative integer, got {max_depth}"
            )
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be a positive integer, got {min_samples_leaf}"
            )
        if min_gain < 0:
            raise ValueError(f"min_gain must be non-negative, got {min_gain}")

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.root: Optional[_Node] = None

    def fit(
        self,
        x: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        lam: float,
    ) -> "NewtonTree":
        """Fit the tree to second-order gradients ``(g, h)``.

        Replaces any previously stored tree. The recursive greedy
        search runs in O(N log N * D) per node; the cumulative-sum
        trick keeps the inner loop linear in the number of candidate
        thresholds.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.
            g: First-order gradients, shape ``(n_samples,)``. These are
                the empirical-risk gradients ``g_i = (1/N) * d l / d y_pred``.
            h: Second-order Hessians, shape ``(n_samples,)``. Must be
                non-negative for typical losses (negative entries would
                yield a non-convex surrogate).
            lam: L2 regularization parameter (``λ_base`` for vanilla
                Newton, ``λ_k`` for GRN). Must be non-negative.

        Returns:
            Self, for chaining.

        Raises:
            TypeError: If any input is not a NumPy array.
            ValueError: If shapes mismatch, the data is empty, ``lam``
                is negative, or any input contains ``NaN``/``Inf``.
        """
        self._validate_fit_inputs(x, g, h, lam)
        self.root = self._build(x, g, h, lam, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict leaf weights for each sample.

        For each row of ``x`` the function walks the tree following
        ``x[feature_idx] <= threshold`` comparisons and returns the
        weight of the reached leaf.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Prediction vector of shape ``(n_samples,)``.

        Raises:
            TypeError: If ``x`` is not a NumPy array.
            ValueError: If ``x`` is not 2-D, or if the tree has not
                been fitted.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy.ndarray, got {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if self.root is None:
            raise ValueError(
                "Tree has not been fitted yet. Call fit() before predict()."
            )
        # Preallocate output to avoid per-row appends.
        out: np.ndarray = np.empty(x.shape[0])
        assert self.root is not None
        for i in range(x.shape[0]):
            out[i] = self._predict_one(self.root, x[i])
        return out

    def _predict_one(self, node: _Node, sample: np.ndarray) -> float:
        """Traverse the tree for a single sample.

        Recursive descent following the threshold test; ``<=``
        (rather than ``<``) is used so that samples exactly equal to
        the threshold go left, which keeps the tree deterministic when
        thresholds are midpoints.
        """
        if node.is_leaf:
            return node.weight
        assert node.left is not None
        assert node.right is not None
        if sample[node.feature_idx] <= node.threshold:
            return self._predict_one(node.left, sample)
        return self._predict_one(node.right, sample)

    def _build(
        self,
        x: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        lam: float,
        depth: int,
    ) -> _Node:
        """Recursively build the tree via exhaustive greedy search.

        Algorithm (per recursive call):
            1. Initialize the node as a leaf with the closed-form
               weight ``w = -Σg / (Σh + λ)``.
            2. Stop if ``depth >= max_depth`` or fewer than ``2 * min_samples_leaf``
               samples remain, or no features exist.
            3. For every non-constant feature, sort by value, build
               cumulative sums of ``g`` and ``h``, and evaluate the
               gain for every distinct threshold.
            4. Keep the split with the largest gain above ``min_gain``.
            5. Recurse on the chosen left/right partitions.

        The returned node is fully self-contained (its ``left`` and
        ``right`` children own their own sub-arrays).
        """
        node = _Node(is_leaf=True)
        sum_h: float = float(np.sum(h))
        # Closed-form leaf weight. The epsilon guards division-by-zero when
        # both Σh and λ are numerically zero (rare but possible with
        # pathological, e.g., all-zero gradients).
        node.weight = -float(np.sum(g)) / (sum_h + lam + 1e-12)

        # Stop if max depth reached, or not enough samples to split.
        if depth >= self.max_depth or x.shape[0] <= self.min_samples_leaf * 2:
            return node

        # Stop if all features are constant (no valid split possible).
        if x.shape[1] == 0:
            return node

        best_gain = self.min_gain
        best_feat: Optional[int] = None
        best_thresh: Optional[float] = None
        best_left_idx: Optional[np.ndarray] = None
        best_right_idx: Optional[np.ndarray] = None

        for feat in range(x.shape[1]):
            x_col = x[:, feat]
            # Skip constant features (no information gain is possible).
            if np.min(x_col) == np.max(x_col):
                continue

            # Sort by feature value to enable cumulative-sum gain evaluation.
            order = np.argsort(x_col)
            x_sorted = x_col[order]
            g_sorted = g[order]
            h_sorted = h[order]
            g_cum = np.cumsum(g_sorted)
            h_cum = np.cumsum(h_sorted)
            g_total = g_cum[-1]
            h_total = h_cum[-1]

            for j in range(x_sorted.shape[0] - 1):
                # Skip duplicate feature values; the midpoint would be
                # identical to the next candidate, producing the same split.
                if x_sorted[j] == x_sorted[j + 1]:
                    continue
                n_left = j + 1
                n_right = x_sorted.shape[0] - n_left
                # Enforce min_samples_leaf on both sides of the split.
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gl = g_cum[j]
                hl = h_cum[j]
                gr = g_total - gl
                hr = h_total - hl
                # Closed-form Newton gain (paper Section 5) with epsilons
                # guarding against zero denominators.
                gain = 0.5 * (
                    gl**2 / (hl + lam + 1e-12)
                    + gr**2 / (hr + lam + 1e-12)
                    - g_total**2 / (h_total + lam + 1e-12)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    # Midpoint threshold: equidistant from the two adjacent
                    # unique values so neither value sits exactly on the cut.
                    best_thresh = (x_sorted[j] + x_sorted[j + 1]) / 2.0
                    best_left_idx = order[: j + 1]
                    best_right_idx = order[j + 1 :]

        # If no split yields sufficient gain, keep the node as a leaf.
        if best_feat is None:
            return node

        node.is_leaf = False
        node.feature_idx = best_feat
        node.threshold = best_thresh
        node.left = self._build(
            x[best_left_idx],
            g[best_left_idx],
            h[best_left_idx],
            lam,
            depth + 1,
        )
        node.right = self._build(
            x[best_right_idx],
            g[best_right_idx],
            h[best_right_idx],
            lam,
            depth + 1,
        )
        return node

    def _validate_fit_inputs(
        self,
        x: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        lam: float,
    ) -> None:
        """Validate the inputs to :meth:`fit`.

        Raises:
            TypeError: If any input is not a NumPy array.
            ValueError: On rank mismatch (``x`` must be 2-D, ``g``
                and ``h`` must be 1-D), on inconsistent sample counts,
                on empty data, on negative ``lam``, or on ``NaN`` /
                ``Inf`` in any of ``x``, ``g``, ``h``.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy.ndarray, got {type(x).__name__}")
        if not isinstance(g, np.ndarray):
            raise TypeError(f"g must be a numpy.ndarray, got {type(g).__name__}")
        if not isinstance(h, np.ndarray):
            raise TypeError(f"h must be a numpy.ndarray, got {type(h).__name__}")
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if g.ndim != 1:
            raise ValueError(f"g must be 1-D, got shape {g.shape}")
        if h.ndim != 1:
            raise ValueError(f"h must be 1-D, got shape {h.shape}")
        if x.shape[0] != g.shape[0]:
            raise ValueError(f"Sample count mismatch: x {x.shape[0]} vs g {g.shape[0]}")
        if x.shape[0] != h.shape[0]:
            raise ValueError(f"Sample count mismatch: x {x.shape[0]} vs h {h.shape[0]}")
        if x.shape[0] == 0:
            raise ValueError("Cannot fit on empty data (n_samples = 0).")
        if lam < 0:
            raise ValueError(f"Regularization lambda must be non-negative, got {lam}")
        if np.any(np.isnan(x)) or np.any(np.isnan(g)) or np.any(np.isnan(h)):
            raise ValueError("Inputs contain NaN values.")
        if np.any(np.isinf(x)) or np.any(np.isinf(g)) or np.any(np.isinf(h)):
            raise ValueError("Inputs contain infinite values.")


class _MultiClassNode:
    """Recursive multi-class tree node used by :class:`MultiClassNewtonTree`.

    Mirrors :class:`_Node` but stores a vector leaf weight of shape
    ``(K,)`` so that all classes share the same split structure. The
    multi-class ensemble ``F`` is therefore a stack of ``K`` real-valued
    functions, all derived from a single greedy tree.

    Attributes:
        is_leaf: ``True`` for terminal nodes, ``False`` for split nodes.
        weight: ``(K,)`` leaf weight vector; ``None`` until the node
            is built.
        feature_idx: Split feature index (``None`` for leaves).
        threshold: Split threshold (``None`` for leaves).
        left: Left child.
        right: Right child.
    """

    def __init__(self, is_leaf: bool = True) -> None:
        self.is_leaf = is_leaf
        self.weight: Optional[np.ndarray] = None
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["_MultiClassNode"] = None
        self.right: Optional["_MultiClassNode"] = None


class MultiClassNewtonTree:
    """Decision tree weak learner for multi-class Newton boosting.

    Builds a greedy axis-aligned tree with vector-valued leaf weights.
    Split quality is measured by the **sum** of Newton gains across all
    ``K`` classes — every class uses the same split but ends up with
    its own scalar weight in the leaf:

        w_k = - sum_{i in leaf} g_{i,k} / (sum_{i in leaf} h_{i,k} + λ)

    This is the natural extension of :class:`NewtonTree` to multi-class
    logits. The split is selected to maximize the *aggregate* surrogate
    reduction across classes, and the resulting leaves hold vectors
    ``w ∈ R^K``.

    Attributes:
        n_classes: Number of classes ``K``.
        max_depth: Maximum tree depth (root is depth ``0``).
        min_samples_leaf: Minimum samples required in each leaf.
        min_gain: Minimum gain required to accept a split.
        root: Fitted root node; ``None`` before :meth:`fit` is called.
    """

    def __init__(
        self,
        n_classes: int,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        min_gain: float = 1e-7,
    ) -> None:
        """Initialize multi-class tree hyperparameters.

        Args:
            n_classes: Number of classes ``K``. Must be ``>= 2``.
            max_depth: Maximum depth of the tree (root depth ``0``).
                Must be ``>= 0``.
            min_samples_leaf: Minimum samples in each leaf. Must be
                ``>= 1``.
            min_gain: Minimum gain threshold to accept a split.
                Must be ``>= 0``.

        Raises:
            ValueError: If any hyperparameter is out of range or has the
                wrong Python type.
        """
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError(f"n_classes must be an integer >= 2, got {n_classes}")
        if not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError(
                f"max_depth must be a non-negative integer, got {max_depth}"
            )
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be a positive integer, got {min_samples_leaf}"
            )
        if min_gain < 0:
            raise ValueError(f"min_gain must be non-negative, got {min_gain}")

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.root: Optional[_MultiClassNode] = None

    def fit(
        self,
        x: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        lam: float,
    ) -> "MultiClassNewtonTree":
        """Fit the tree to multi-class second-order gradients.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.
            g: First-order gradients, shape ``(n_samples, n_classes)``.
            h: Second-order Hessians, shape ``(n_samples, n_classes)``.
                In practice the diagonal of the block Hessian is used;
                see :class:`grnbt.boosting.MultiClassNewtonBoosting`
                for the extraction step.
            lam: L2 regularization parameter. Must be non-negative.

        Returns:
            Self, for chaining.

        Raises:
            TypeError: If any input is not a NumPy array.
            ValueError: If shapes mismatch, the data is empty, ``lam``
                is negative, or any input contains ``NaN`` / ``Inf``.
        """
        self._validate_fit_inputs(x, g, h, lam)
        self.root = self._build(x, g, h, lam, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict leaf weight vectors for each sample.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Prediction matrix of shape ``(n_samples, n_classes)``.
            Each row is the ``K``-dimensional weight vector of the
            leaf reached by the corresponding sample.

        Raises:
            TypeError: If ``x`` is not a NumPy array.
            ValueError: If ``x`` is not 2-D, or if the tree has not been
                fitted.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy.ndarray, got {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if self.root is None:
            raise ValueError(
                "Tree has not been fitted yet. Call fit() before predict()."
            )
        out = np.empty((x.shape[0], self.n_classes))
        assert self.root is not None
        for i in range(x.shape[0]):
            out[i] = self._predict_one(self.root, x[i])
        return out

    def _predict_one(self, node: _MultiClassNode, sample: np.ndarray) -> np.ndarray:
        """Traverse tree for a single sample, returning the K-dim weight vector."""
        if node.is_leaf:
            assert node.weight is not None
            return node.weight
        assert node.left is not None
        assert node.right is not None
        if sample[node.feature_idx] <= node.threshold:
            return self._predict_one(node.left, sample)
        return self._predict_one(node.right, sample)

    def _build(
        self,
        x: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        lam: float,
        depth: int,
    ) -> _MultiClassNode:
        """Recursively build the multi-class tree.

        Mirrors :meth:`NewtonTree._build` but evaluates gains on a
        per-class vector basis. The split is selected to maximize the
        *aggregate* second-order gain across all classes; the same
        split structure is then used for every class's scalar weight.

        Algorithm:

        1. Initialize the node as a leaf with per-class closed-form
           weights ``w_k = -sum(g_k) / (sum(h_k) + λ)``.
        2. Stop if ``depth >= max_depth`` or fewer than
           ``2 * min_samples_leaf`` samples remain.
        3. For every non-constant feature, sort by feature value,
           compute per-class cumulative sums, and evaluate the gain
           ``sum_k gain_k`` at every distinct threshold.
        4. Keep the split with the largest aggregate gain above
           ``min_gain`` and recurse on both partitions.
        """
        node = _MultiClassNode(is_leaf=True)
        # Per-class leaf weights: w_k = -Σ g_k / (Σ h_k + λ + eps).
        sum_h = np.sum(h, axis=0)  # (K,)
        node.weight = -np.sum(g, axis=0) / (sum_h + lam + 1e-12)  # (K,)

        if depth >= self.max_depth or x.shape[0] <= self.min_samples_leaf * 2:
            return node

        if x.shape[1] == 0:
            return node

        best_gain = self.min_gain
        best_feat: Optional[int] = None
        best_thresh: Optional[float] = None
        best_left_idx: Optional[np.ndarray] = None
        best_right_idx: Optional[np.ndarray] = None

        # Pre-compute per-class totals to avoid recomputation inside the
        # threshold loop.
        g_total = np.sum(g, axis=0)  # (K,)
        h_total = np.sum(h, axis=0)  # (K,)

        for feat in range(x.shape[1]):
            x_col = x[:, feat]
            if np.min(x_col) == np.max(x_col):
                continue

            order = np.argsort(x_col)
            x_sorted = x_col[order]
            g_sorted = g[order]
            h_sorted = h[order]
            g_cum = np.cumsum(g_sorted, axis=0)  # (N, K)
            h_cum = np.cumsum(h_sorted, axis=0)  # (N, K)

            for j in range(x_sorted.shape[0] - 1):
                if x_sorted[j] == x_sorted[j + 1]:
                    continue
                n_left = j + 1
                n_right = x_sorted.shape[0] - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gl = g_cum[j]  # (K,)
                hl = h_cum[j]  # (K,)
                gr = g_total - gl  # (K,)
                hr = h_total - hl  # (K,)
                # Aggregate Newton gain across classes. The same split
                # structure is shared, so we sum per-class contributions.
                gain = float(
                    np.sum(
                        0.5
                        * (
                            gl**2 / (hl + lam + 1e-12)
                            + gr**2 / (hr + lam + 1e-12)
                            - g_total**2 / (h_total + lam + 1e-12)
                        )
                    )
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = (x_sorted[j] + x_sorted[j + 1]) / 2.0
                    best_left_idx = order[: j + 1]
                    best_right_idx = order[j + 1 :]

        if best_feat is None:
            return node

        node.is_leaf = False
        node.feature_idx = best_feat
        node.threshold = best_thresh
        node.left = self._build(
            x[best_left_idx],
            g[best_left_idx],
            h[best_left_idx],
            lam,
            depth + 1,
        )
        node.right = self._build(
            x[best_right_idx],
            g[best_right_idx],
            h[best_right_idx],
            lam,
            depth + 1,
        )
        return node

    def _validate_fit_inputs(
        self,
        x: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        lam: float,
    ) -> None:
        """Validate the inputs to the multi-class :meth:`fit`.

        In addition to the scalar checks, this method enforces that
        ``g`` and ``h`` are 2-D with ``n_classes`` columns.

        Raises:
            TypeError: If any input is not a NumPy array.
            ValueError: On rank mismatch (``x`` must be 2-D, ``g``
                and ``h`` must be 2-D with ``n_classes`` columns), on
                inconsistent sample counts, on empty data, on
                negative ``lam``, or on ``NaN`` / ``Inf`` in any
                input.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy.ndarray, got {type(x).__name__}")
        if not isinstance(g, np.ndarray):
            raise TypeError(f"g must be a numpy.ndarray, got {type(g).__name__}")
        if not isinstance(h, np.ndarray):
            raise TypeError(f"h must be a numpy.ndarray, got {type(h).__name__}")
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if g.ndim != 2:
            raise ValueError(f"g must be 2-D, got shape {g.shape}")
        if h.ndim != 2:
            raise ValueError(f"h must be 2-D, got shape {h.shape}")
        if g.shape[1] != self.n_classes:
            raise ValueError(f"g must have {self.n_classes} columns, got {g.shape[1]}")
        if h.shape[1] != self.n_classes:
            raise ValueError(f"h must have {self.n_classes} columns, got {h.shape[1]}")
        if x.shape[0] != g.shape[0]:
            raise ValueError(f"Sample count mismatch: x {x.shape[0]} vs g {g.shape[0]}")
        if x.shape[0] != h.shape[0]:
            raise ValueError(f"Sample count mismatch: x {x.shape[0]} vs h {h.shape[0]}")
        if x.shape[0] == 0:
            raise ValueError("Cannot fit on empty data (n_samples = 0).")
        if lam < 0:
            raise ValueError(f"Regularization lambda must be non-negative, got {lam}")
        if np.any(np.isnan(x)) or np.any(np.isnan(g)) or np.any(np.isnan(h)):
            raise ValueError("Inputs contain NaN values.")
        if np.any(np.isinf(x)) or np.any(np.isinf(g)) or np.any(np.isinf(h)):
            raise ValueError("Inputs contain infinite values.")

"""Newton-boosting decision tree with exact greedy split finding.

Leaf weights follow the standard second-order formula:
    w = - sum(g) / (sum(h) + lambda)

Split quality is measured by the exact Newton gain from the paper.
"""

from typing import Optional

import numpy as np


class _Node:
    """Internal node representation for the decision tree."""

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
    threshold that maximizes the second-order Newton gain.  Leaf weights
    are closed-form optimal values for the quadratic surrogate.

    Attributes:
        max_depth: Maximum tree depth (root is depth 0).
        min_samples_leaf: Minimum samples required in each leaf.
        min_gain: Minimum gain required to perform a split.
        root: Fitted root node; None before fit().
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        min_gain: float = 1e-7,
    ) -> None:
        """Initialize tree hyperparameters.

        Args:
            max_depth: Maximum depth of the tree. Must be >= 0.
            min_samples_leaf: Minimum number of samples in a leaf. Must be >= 1.
            min_gain: Minimum gain threshold to accept a split. Must be >= 0.

        Raises:
            ValueError: If any hyperparameter is invalid.
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
        """Fit the tree to second-order gradients.

        Args:
            x: Feature matrix of shape (n_samples, n_features).
            g: First-order gradients, shape (n_samples,).
            h: Second-order Hessians, shape (n_samples,).
            lam: L2 regularization parameter (static or adaptive lambda_k).
                Must be non-negative.

        Returns:
            Self, for chaining.

        Raises:
            TypeError: If inputs are not NumPy arrays.
            ValueError: If shapes mismatch, arrays are empty, or lam is negative.
        """
        self._validate_fit_inputs(x, g, h, lam)
        self.root = self._build(x, g, h, lam, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict leaf weights for each sample.

        Args:
            x: Feature matrix of shape (n_samples, n_features).

        Returns:
            Prediction vector of shape (n_samples,).

        Raises:
            ValueError: If the tree has not been fitted.
            TypeError: If x is not a NumPy array.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy.ndarray, got {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        if self.root is None:
            raise ValueError(
                "Tree has not been fitted yet. Call fit() before predict()."
            )
        out: np.ndarray = np.empty(x.shape[0])
        assert self.root is not None
        for i in range(x.shape[0]):
            out[i] = self._predict_one(self.root, x[i])
        return out

    def _predict_one(self, node: _Node, sample: np.ndarray) -> float:
        """Traverse tree for a single sample."""
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
        """Recursively build the tree via exhaustive greedy search."""
        node = _Node(is_leaf=True)
        sum_h: float = float(np.sum(h))
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
            # Skip constant features.
            if np.min(x_col) == np.max(x_col):
                continue

            order = np.argsort(x_col)
            x_sorted = x_col[order]
            g_sorted = g[order]
            h_sorted = h[order]
            g_cum = np.cumsum(g_sorted)
            h_cum = np.cumsum(h_sorted)
            g_total = g_cum[-1]
            h_total = h_cum[-1]

            for j in range(x_sorted.shape[0] - 1):
                if x_sorted[j] == x_sorted[j + 1]:
                    continue
                n_left = j + 1
                n_right = x_sorted.shape[0] - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gl = g_cum[j]
                hl = h_cum[j]
                gr = g_total - gl
                hr = h_total - hl
                # Guard against pathological zero denominators.
                gain = 0.5 * (
                    gl**2 / (hl + lam + 1e-12)
                    + gr**2 / (hr + lam + 1e-12)
                    - g_total**2 / (h_total + lam + 1e-12)
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
        """Validate inputs to fit().

        Raises:
            TypeError: For wrong input types.
            ValueError: For invalid shapes, empty data, or negative lam.
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

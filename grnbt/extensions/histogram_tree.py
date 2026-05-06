"""Histogram-based NewtonTree approximation for faster split finding.

Bins each feature into a fixed number of buckets, then evaluates splits on
bucket boundaries rather than individual sample thresholds. This reduces
complexity from O(N log N) per split to O(N + B) where B is the number of bins.

This is an efficiency extension only; leaf weights and the gain formula are
identical to the exact `NewtonTree`.
"""

from typing import Optional

import numpy as np

from grnbt.tree import NewtonTree, _Node


class HistogramNewtonTree(NewtonTree):
    """NewtonTree variant using histogram binning for approximate splits.

    Attributes:
        n_bins: Number of histogram bins per feature.
        (inherited from NewtonTree: max_depth, min_samples_leaf, min_gain)
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        min_gain: float = 1e-7,
        n_bins: int = 64,
    ) -> None:
        """Initialize histogram tree.

        Args:
            max_depth: Maximum depth.
            min_samples_leaf: Minimum samples per leaf.
            min_gain: Minimum gain to split.
            n_bins: Number of histogram bins per feature.
        """
        super().__init__(max_depth, min_samples_leaf, min_gain)
        if not isinstance(n_bins, int) or n_bins < 1:
            raise ValueError(f"n_bins must be a positive integer, got {n_bins}")
        self.n_bins = n_bins

    def _build(
        self,
        x: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        lam: float,
        depth: int,
    ) -> _Node:
        """Recursively build the tree using histogram splits."""
        node = _Node(is_leaf=True)
        node.weight = -np.sum(g) / (np.sum(h) + lam + 1e-12)

        if depth >= self.max_depth or x.shape[0] <= self.min_samples_leaf * 2:
            return node

        best_gain = self.min_gain
        best_feat: Optional[int] = None
        best_thresh: Optional[float] = None
        best_left_idx: Optional[np.ndarray] = None
        best_right_idx: Optional[np.ndarray] = None

        for feat in range(x.shape[1]):
            x_col = x[:, feat]
            # Build histogram
            x_min: float = float(np.min(x_col))
            x_max: float = float(np.max(x_col))
            if x_min == x_max:
                continue
            bin_edges = np.linspace(x_min, x_max, self.n_bins + 1)
            bin_idx = np.digitize(x_col, bin_edges[1:-1])
            # Evaluate splits at bin boundaries
            g_total: float = float(np.sum(g))
            h_total: float = float(np.sum(h))
            for b in range(self.n_bins - 1):
                left_mask = bin_idx <= b
                right_mask = ~left_mask
                n_left = np.count_nonzero(left_mask)
                n_right = x.shape[0] - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                gl: float = float(np.sum(g[left_mask]))
                hl: float = float(np.sum(h[left_mask]))
                gr = g_total - gl
                hr = h_total - hl
                gain = 0.5 * (
                    gl**2 / (hl + lam + 1e-12)
                    + gr**2 / (hr + lam + 1e-12)
                    - g_total**2 / (h_total + lam + 1e-12)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = bin_edges[b + 1]
                    best_left_idx = np.where(left_mask)[0]
                    best_right_idx = np.where(right_mask)[0]

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

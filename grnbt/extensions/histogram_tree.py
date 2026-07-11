"""Histogram-based NewtonTree approximation for faster split finding.

Bins each feature into a fixed number of buckets, then evaluates
candidate splits on bucket *boundaries* rather than on every distinct
sample threshold. This reduces per-node complexity from
``O(N log N)`` (sort + cumulative sum) to ``O(N + B)`` where ``B`` is
the number of bins.

Important
---------

* This module is part of ``grnbt.extensions`` and is **not** part of
  the faithful paper reproduction. It exists to demonstrate how the
  algorithm scales to larger datasets.
* Leaf weights and the gain formula are identical to :class:`NewtonTree`
  so that downstream consumers (boosting engines, diagnostics) do not
  need to special-case the variant.
"""

from typing import Optional

import numpy as np

from grnbt.tree import NewtonTree, _Node


class HistogramNewtonTree(NewtonTree):
    """Histogram-binned variant of :class:`NewtonTree`.

    Subclasses :class:`NewtonTree` and overrides :meth:`_build` to use
    histogram bin boundaries as candidate split points. Prediction,
    leaf weight formulas, gain expressions, and the ``min_gain`` /
    ``min_samples_leaf`` semantics are inherited unchanged.

    Trade-offs versus the exact tree:

    * Faster on large ``N`` (e.g., ``N >= 10^4``) because the inner
      loop runs ``n_bins - 1`` times instead of ``N - 1``.
    * Approximate: splits may be slightly suboptimal because the
      "best" exact threshold can fall inside a wide bin.

    Attributes:
        n_bins: Number of histogram bins per feature.
        (Also inherits ``max_depth``, ``min_samples_leaf``, and
        ``min_gain`` from :class:`NewtonTree`.)
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
            n_bins: Number of histogram bins per feature. Must be a
                positive integer. With ``n_bins=1`` no split is
                possible so the tree is a single leaf.

        Raises:
            ValueError: If ``n_bins`` is not a positive integer.
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
        """Recursively build the tree using histogram splits.

        The algorithm mirrors :meth:`NewtonTree._build` but replaces
        the sort-based threshold loop with a ``np.digitize`` binning
        step: each sample is assigned a bin index in ``[0, n_bins-1]``
        and splits are evaluated only at the ``n_bins - 1`` bin
        boundaries.

        Algorithmic complexity per recursive call:
            * O(N) for ``digitize`` plus ``O(B)`` threshold evaluation,
              where ``B = n_bins``.
        """
        node = _Node(is_leaf=True)
        # Closed-form leaf weight is unchanged from the exact builder.
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
            # Build histogram. Skip constant features (no info).
            x_min: float = float(np.min(x_col))
            x_max: float = float(np.max(x_col))
            if x_min == x_max:
                continue
            # Equal-width bins over the observed feature range.
            bin_edges = np.linspace(x_min, x_max, self.n_bins + 1)
            # `bin_edges[1:-1]` are the *interior* boundaries used by
            # `np.digitize`; bin 0 is the leftmost.
            bin_idx = np.digitize(x_col, bin_edges[1:-1])
            # Pre-compute totals so the per-boundary loop is O(B).
            g_total: float = float(np.sum(g))
            h_total: float = float(np.sum(h))
            # Evaluate splits at each interior bin boundary.
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
                # Same gain formula as the exact builder.
                gain = 0.5 * (
                    gl**2 / (hl + lam + 1e-12)
                    + gr**2 / (hr + lam + 1e-12)
                    - g_total**2 / (h_total + lam + 1e-12)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    # Threshold is the actual feature value at the
                    # bin boundary, which keeps splits on the same
                    # scale as the input.
                    best_thresh = bin_edges[b + 1]
                    best_left_idx = np.where(left_mask)[0]
                    best_right_idx = np.where(right_mask)[0]

        # No split cleared the gain threshold; remain a leaf.
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

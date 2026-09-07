"""Gradient Regularized Newton Boosting Trees (GRNBT).

Pure-Python reproduction of:

    Zozoulenko, N., Falkowski, D., Cass, T., Gonon, L. (2026).
    *Gradient Regularized Newton Boosting Trees with Global Convergence.*
    arXiv:2605.00581v1.

The package contains a faithful, dependency-light implementation of
the paper's two boosting algorithms (vanilla Newton boosting, gradient
regularized Newton boosting) plus a multi-class extension. It depends
*only* on NumPy at runtime; scikit-learn and matplotlib are only
required for the optional dataset loaders and experiment plots.

Quick start
-----------

>>> import numpy as np
>>> from grnbt import (
...     GradientRegularizedNewtonBoosting, CharbonnierLoss,
... )
>>> rng = np.random.RandomState(0)
>>> x = rng.randn(64, 4)
>>> y = x[:, 0] + 0.1 * rng.randn(64)
>>> model = GradientRegularizedNewtonBoosting(
...     loss=CharbonnierLoss(), n_estimators=20, max_depth=3, lam_base=0.0,
... )
>>> model.fit(x, y)
<...>
>>> preds = model.predict(x)  # shape (64,)

Package layout
--------------

* :mod:`grnbt.losses` — pointwise losses with analytical ``M_0``
  constants (MSE, Charbonnier, BCE, CCE).
* :mod:`grnbt.tree` — :class:`NewtonTree` and
  :class:`MultiClassNewtonTree` weak learners.
* :mod:`grnbt.boosting` — :class:`VanillaNewtonBoosting`,
  :class:`GradientRegularizedNewtonBoosting`, and the multi-class
  extension.
* :mod:`grnbt.diagnostics` — Hilbert-space quantities (cosine angle,
  weak gradient edge, Lemma 4.2 checker).
* :mod:`grnbt.datasets` — Wine Quality and Higgs loaders.
* :mod:`grnbt.utils` — norms, threshold candidates, and the training
  metric logger.
* :mod:`grnbt.extensions` — optional improvements (e.g. histogram
  tree) that are **not** part of the paper baseline.

References
----------

Paper: arXiv:2605.00581v1 (Sections 4–6 for algorithms and diagnostics,
Appendix A for ``M_0`` derivations, Proposition 5.1 for empirical
Hessian Lipschitz scaling).
"""

from grnbt.boosting import (
    GradientRegularizedNewtonBoosting,
    MultiClassNewtonBoosting,
    VanillaNewtonBoosting,
)
from grnbt.datasets import load_higgs_subset, load_wine_quality
from grnbt.diagnostics import (
    cosine_angle_theta,
    exact_newton_direction,
    validate_diagonal_inputs,
    verify_lemma_4_2,
    weak_gradient_edge_gamma,
)
from grnbt.losses import (
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss,
    CharbonnierLoss,
    MSELoss,
    validate_inputs,
)
from grnbt.tree import MultiClassNewtonTree, MultiClassNode, NewtonTree, Node

__all__ = [
    # Losses.
    "MSELoss",
    "CharbonnierLoss",
    "BinaryCrossEntropyLoss",
    "CategoricalCrossEntropyLoss",
    "validate_inputs",
    # Weak learners.
    "NewtonTree",
    "MultiClassNewtonTree",
    "Node",
    "MultiClassNode",
    # Boosting engines.
    "VanillaNewtonBoosting",
    "GradientRegularizedNewtonBoosting",
    "MultiClassNewtonBoosting",
    # Diagnostics.
    "exact_newton_direction",
    "cosine_angle_theta",
    "weak_gradient_edge_gamma",
    "verify_lemma_4_2",
    "validate_diagonal_inputs",
    # Datasets.
    "load_wine_quality",
    "load_higgs_subset",
]

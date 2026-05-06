"""Gradient Regularized Newton Boosting Trees (GRNBT).

Pure Python reproduction of:
  Zozoulenko et al., "Gradient Regularized Newton Boosting Trees with
  Global Convergence", arXiv:2605.00581v1.
"""

from grnbt.boosting import GradientRegularizedNewtonBoosting, VanillaNewtonBoosting
from grnbt.datasets import load_higgs_subset, load_wine_quality
from grnbt.diagnostics import (
    cosine_angle_theta,
    exact_newton_direction,
    verify_lemma_4_2,
    weak_gradient_edge_gamma,
)
from grnbt.losses import (
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss,
    CharbonnierLoss,
    MSELoss,
)
from grnbt.tree import NewtonTree

__all__ = [
    "MSELoss",
    "CharbonnierLoss",
    "BinaryCrossEntropyLoss",
    "CategoricalCrossEntropyLoss",
    "NewtonTree",
    "VanillaNewtonBoosting",
    "GradientRegularizedNewtonBoosting",
    "exact_newton_direction",
    "cosine_angle_theta",
    "weak_gradient_edge_gamma",
    "verify_lemma_4_2",
    "load_wine_quality",
    "load_higgs_subset",
]

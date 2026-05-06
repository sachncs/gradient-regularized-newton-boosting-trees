"""Reproduce Higgs weak-learner diagnostics (Paper Section 6, Fig. 2).

Computes cosine angle Θ_k and weak gradient edge γ_k per iteration
for tree depths 2, 4, and 6.
Expected result: deeper trees produce larger angles/edges; both plateau positive.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from grnbt.boosting import GradientRegularizedNewtonBoosting
from grnbt.datasets import load_higgs_subset
from grnbt.diagnostics import (
    cosine_angle_theta,
    exact_newton_direction,
    weak_gradient_edge_gamma,
)
from grnbt.losses import BinaryCrossEntropyLoss


def main() -> None:
    """Run Higgs diagnostics experiment and save results."""
    x, y = load_higgs_subset(n_samples=10_000)
    n_estimators = 50
    depths = [2, 4, 6]
    eta = 1.0
    results = {d: {"theta": [], "gamma": []} for d in depths}

    for depth in depths:
        model = GradientRegularizedNewtonBoosting(
            loss=BinaryCrossEntropyLoss(),
            n_estimators=n_estimators,
            learning_rate=eta,
            max_depth=depth,
            lam_base=1e-3,
            verbose=False,
        )
        model.fit(x, y)
        f_current = model.F0.copy()
        for k, tree in enumerate(model.trees):
            g = model.loss.gradient(y, f_current)
            h = model.loss.hessian(y, f_current)
            lam = model.history.get("lambda_k")[k]
            f_exact = exact_newton_direction(g, h, lam)
            f_weak = tree.predict(x)
            theta = cosine_angle_theta(g, h, f_exact, f_weak)
            gamma = weak_gradient_edge_gamma(g, h, lam, f_weak)
            results[depth]["theta"].append(theta)
            results[depth]["gamma"].append(gamma)
            f_current += model.learning_rate * f_weak

    save_dict = {}
    for depth, vals in results.items():
        save_dict[f"theta_depth_{depth}"] = np.array(vals["theta"])
        save_dict[f"gamma_depth_{depth}"] = np.array(vals["gamma"])
    np.savez("experiments/higgs_diagnostics_results.npz", **save_dict)
    print("Saved results to experiments/higgs_diagnostics_results.npz")


if __name__ == "__main__":
    main()

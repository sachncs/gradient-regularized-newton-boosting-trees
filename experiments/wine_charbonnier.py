"""Reproduce the Charbonnier-loss experiment on Wine Quality (Paper Section 6).

Compares Vanilla Newton Boosting, GRN, and a static high-lambda baseline.
Expected result: vanilla diverges, GRN converges fastest, static lambda is biased.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from grnbt.boosting import GradientRegularizedNewtonBoosting, VanillaNewtonBoosting
from grnbt.datasets import load_wine_quality
from grnbt.losses import CharbonnierLoss


def main() -> None:
    """Run Wine Quality Charbonnier experiment and save results."""
    x, y = load_wine_quality()
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)

    n_estimators = 100
    depth = 4
    eta = 1.0

    models = {
        "Vanilla": VanillaNewtonBoosting(
            loss=CharbonnierLoss(),
            n_estimators=n_estimators,
            learning_rate=eta,
            max_depth=depth,
            lam_base=0.0,
            verbose=True,
        ),
        "GRN": GradientRegularizedNewtonBoosting(
            loss=CharbonnierLoss(),
            n_estimators=n_estimators,
            learning_rate=eta,
            max_depth=depth,
            lam_base=0.0,
            verbose=True,
        ),
        "StaticHighLam": VanillaNewtonBoosting(
            loss=CharbonnierLoss(),
            n_estimators=n_estimators,
            learning_rate=eta,
            max_depth=depth,
            lam_base=10.0,
            verbose=True,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(x, y)
        results[name] = model.history.get("loss")

    np.savez("experiments/wine_charbonnier_results.npz", **results)
    print("Saved results to experiments/wine_charbonnier_results.npz")

    try:
        import matplotlib.pyplot as plt

        for name, losses in results.items():
            plt.plot(losses, label=name)
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Charbonnier Loss")
        plt.legend()
        plt.title("Wine Quality — Charbonnier Loss")
        plt.savefig("experiments/wine_charbonnier.png")
        print("Saved plot to experiments/wine_charbonnier.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()

"""Hyperparameter ablations on synthetic data.

Vary loss, learning rate η, tree depth, λ_base, and engine type,
recording final loss after 50 iterations.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from grnbt.boosting import GradientRegularizedNewtonBoosting, VanillaNewtonBoosting
from grnbt.losses import CharbonnierLoss, MSELoss


def _make_synthetic(n_samples: int = 200, n_features: int = 5, seed: int = 42):
    """Generate a small synthetic regression dataset."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples, n_features)
    y = x[:, 0] + 0.5 * x[:, 1] ** 2 + rng.randn(n_samples) * 0.1
    return x, y


def main() -> None:
    """Run ablation grid and write CSV."""
    x, y = _make_synthetic()
    rows = []
    for loss_cls, loss_name in [(MSELoss, "MSE"), (CharbonnierLoss, "Charbonnier")]:
        for eta in [0.1, 0.5, 1.0]:
            for depth in [2, 4, 6]:
                for lam_base in [0.0, 0.1, 1.0]:
                    for engine_cls, engine_name in [
                        (VanillaNewtonBoosting, "Vanilla"),
                        (GradientRegularizedNewtonBoosting, "GRN"),
                    ]:
                        model = engine_cls(
                            loss=loss_cls(),
                            n_estimators=50,
                            learning_rate=eta,
                            max_depth=depth,
                            lam_base=lam_base,
                        )
                        model.fit(x, y)
                        final_loss = model.history.get("loss")[-1]
                        rows.append(
                            {
                                "loss": loss_name,
                                "eta": eta,
                                "depth": depth,
                                "lam_base": lam_base,
                                "engine": engine_name,
                                "final_loss": final_loss,
                            }
                        )

    path = "experiments/ablations.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ablation results to {path}")


if __name__ == "__main__":
    main()

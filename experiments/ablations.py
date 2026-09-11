"""Hyperparameter ablations on synthetic data.

Systematically varies five axes of the Newton-boosting algorithm:

* loss function (MSE, Charbonnier);
* learning rate ``η ∈ {0.1, 0.5, 1.0}``;
* tree depth ``∈ {2, 4, 6}``;
* static regularization ``λ_base ∈ {0.0, 0.1, 1.0}``;
* engine type (:class:`VanillaNewtonBoosting`,
  :class:`GradientRegularizedNewtonBoosting`).

For each of the 108 configurations the script trains for 50
boosting rounds on a fixed synthetic regression dataset and records
the final loss. Each configuration is run with three seeds (42, 43,
44) so the CSV captures both the central value and the seed-to-seed
variance. Results are written to ``experiments/ablations.csv`` and
meant to be analyzed with pandas:

.. code-block:: python

    import pandas as pd
    df = pd.read_csv("experiments/ablations.csv")
    grouped = df.groupby(["loss", "engine"])["final_loss"].agg(["mean", "std"])
    print(grouped)

Run with ``python experiments/ablations.py`` from the project root.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from grnbt.boosting import GradientRegularizedNewtonBoosting, VanillaNewtonBoosting
from grnbt.losses import CharbonnierLoss, MSELoss


SEEDS = (42, 43, 44)


def make_synthetic(n_samples: int = 200, n_features: int = 5, seed: int = 42):
    """Generate a small synthetic regression dataset.

    Args:
        n_samples: Number of rows.
        n_features: Number of input columns.
        seed: Random seed (used to make the ablation reproducible).

    Returns:
        Tuple ``(x, y)`` with shapes ``(n_samples, n_features)`` and
        ``(n_samples,)``. Targets are a mildly nonlinear function of
        the first two features plus light Gaussian noise.
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples, n_features)
    y = x[:, 0] + 0.5 * x[:, 1] ** 2 + rng.randn(n_samples) * 0.1
    return x, y


def main() -> None:
    """Run the ablation grid search and write the results.

    Side effects:
        Writes ``experiments/ablations.csv`` with one row per
        (configuration, seed) pair. The total runtime is approximately
        6-9 minutes on a modern CPU (3 seeds × 108 configs).
    """
    rows = []
    for seed in SEEDS:
        x, y = make_synthetic(seed=seed)
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
                                    "seed": seed,
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
    print(f"Saved {len(rows)} ablation rows to {path}")


if __name__ == "__main__":
    main()

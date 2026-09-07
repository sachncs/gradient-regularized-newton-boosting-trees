# Experiments

This document describes the reproduction of the paper's numerical experiments,
expected outcomes, and how to run them.

---

## 1. Wine Quality — Charbonnier Loss

**Paper reference:** Section 6, Figure 1.

**Goal:** Demonstrate that vanilla Newton boosting diverges on the Charbonnier loss
while GRN converges, and that static high regularization is biased.

**Setup:**
- Dataset: Wine Quality (red wine, UCI repository)
- Loss: Charbonnier (`M_0 = 1`)
- Tree depth: 4
- Learning rate: `η = 1.0`
- Iterations: 100
- Models compared:
  1. **Vanilla Newton** (`λ_base = 0.0`)
  2. **GRN** (`λ_base = 0.0`)
  3. **Static High Lambda** (`λ_base = 10.0`)

**Expected behavior:**
- Vanilla: loss explodes (divergence).
- GRN: loss decreases steadily, converging to the unregularized optimum.
- Static High Lambda: loss decreases but plateaus at a higher value (biased optimum).

**Run:**
```bash
python experiments/wine_charbonnier.py
```

**Output:**
- `experiments/wine_charbonnier_results.npz` — loss curves for all three models.
- `experiments/wine_charbonnier.png` — plot (if matplotlib is installed).

---

## 2. Higgs — Weak Learner Diagnostics

**Paper reference:** Section 6, Figure 2.

**Goal:** Compute and visualize `Θ_k` (cosine angle) and `γ_k` (weak gradient edge)
per iteration for different tree depths.

**Setup:**
- Dataset: Higgs subset (`n_samples = 10,000`)
- Loss: Binary Cross-Entropy (`M_0 = 0.25`)
- Tree depths: 2, 4, 6
- Learning rate: `η = 1.0`
- `λ_base = 1e-3`
- Iterations: 50

**Expected behavior:**
- Deeper trees produce larger `Θ_k` and `γ_k` (stronger weak learners).
- Both metrics are highest early in training and then plateau at positive values.
- `Θ_k` is typically in `[0.3, 0.8]` and `γ_k` in `[0.5, 0.95]` depending on depth.

**Run:**
```bash
python experiments/higgs_diagnostics.py
```

**Output:**
- `experiments/higgs_diagnostics_results.npz` — `theta_depth_*` and `gamma_depth_*` arrays.

**Analysis:**
```python
import numpy as np

data = np.load("experiments/higgs_diagnostics_results.npz")
print(data["theta_depth_2"].mean(), data["theta_depth_4"].mean(), data["theta_depth_6"].mean())
print(data["gamma_depth_2"].mean(), data["gamma_depth_4"].mean(), data["gamma_depth_6"].mean())
```

---

## 3. Ablations

**Goal:** Systematically vary hyperparameters on synthetic data to understand
their interaction with loss and engine type.

**Grid:**
- Losses: MSE, Charbonnier
- `η`: 0.1, 0.5, 1.0
- Tree depth: 2, 4, 6
- `λ_base`: 0.0, 0.1, 1.0
- Engines: Vanilla, GRN

**Dataset:**
- Synthetic regression: `y = x_0 + 0.5 x_1^2 + ε` with `N=200, D=5`

**Run:**
```bash
python experiments/ablations.py
```

**Output:**
- `experiments/ablations.csv` — one row per configuration with `final_loss`.

**Analysis:**
```python
import pandas as pd

df = pd.read_csv("experiments/ablations.csv")
print(df.groupby(["loss", "engine"])["final_loss"].mean())
```

---

## 4. Reproducibility Notes

### Random Seeds

All synthetic data generators use a fixed `RandomState(seed=42)` for reproducibility.
Dataset loaders use `seed=42` for subsampling.

### Floating Point

Results may vary slightly across platforms and NumPy versions due to:
- Summation order in `np.sum`.
- `np.exp` and `np.log` implementations.
- Softmax numerical stability (max-subtraction is deterministic but order-dependent).

Differences are typically below `1e-6` per iteration and do not change qualitative behavior.

### Missing Paper Details

| Parameter | Paper Spec | Our Choice |
|-----------|-----------|------------|
| `λ_base` (Wine) | Unstated | `0.0` |
| `λ_base` (Higgs) | Unstated | `1e-3` |
| `min_samples_leaf` | Unstated | `1` |
| Higgs subset size | "subset" | `10,000` diagnostics, `100,000` full |
| Iteration count | Unstated | `100` Wine, `50` Higgs |

These are explicitly documented as assumptions in `docs/fidelity.md`.

### Hardware Requirements

- Wine experiment: ~10 seconds on a modern CPU.
- Higgs diagnostics (10k samples): ~1-2 minutes.
- Ablations: ~2-3 minutes (grid of 108 configurations × 50 iterations).
- Memory: < 500 MB for all experiments.

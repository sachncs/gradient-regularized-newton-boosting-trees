# GRNBT — Gradient Regularized Newton Boosting Trees

Pure Python reproduction of:

> N. Zozoulenko, D. Falkowski, T. Cass, L. Gonon,
> "Gradient Regularized Newton Boosting Trees with Global Convergence",
> arXiv:2605.00581v1.

## Overview

This repository reproduces the core algorithmic contributions of the paper:

1. **Vanilla Newton Boosting** — standard second-order GBDT with static L2 regularization.
2. **Gradient Regularized Newton (GRN)** — adaptive regularization
   `λ_k = λ_base + sqrt(M ||g_k||)` where `M = M_0 sqrt(N)` (Proposition 5.1).
3. **Hilbert-space diagnostics** — exact Newton directions, cosine angles `Θ_k`,
   weak gradient edges `γ_k`, and numerical checks for Lemma 4.2.
4. **Loss functions** — MSE, Charbonnier, binary cross-entropy, and categorical
   cross-entropy with analytically known `M_0` constants.

## Setup

Install in editable mode with all development dependencies:

```bash
pip install -e ".[dev]"
```

Or install only core dependencies:

```bash
pip install -e "."
```

Dependencies:
- `numpy` — required (vector / matrix operations)
- `scikit-learn` — optional (dataset fetching convenience)
- `matplotlib` — optional (experiment plotting)
- `pytest` — optional (test runner)

Configuration is in `pyproject.toml`. There is no `requirements.txt`.

## Tests

Run the full test suite:

```bash
pytest tests/ -v
```

Tests cover:
- Gradient / Hessian correctness via finite differences
- Loss `M_0` values against paper Appendix A
- Tree leaf weight formula and depth constraints
- Boosting loop invariants and loss decrease
- Diagnostic identities (Lemma 4.2, `Θ_k`, `γ_k`)
- Input validation and edge cases (NaN, Inf, empty data, mismatched shapes)
- Dataset loading and standardization

## Experiments

### 1. Wine Quality — Charbonnier Loss

Reproduces Section 6, Figure 1:

```bash
python experiments/wine_charbonnier.py
```

Compares Vanilla Newton, GRN, and a static high-lambda baseline.
Results are saved to `experiments/wine_charbonnier_results.npz`.

### 2. Higgs — Weak Learner Diagnostics

Reproduces Section 6, Figure 2:

```bash
python experiments/higgs_diagnostics.py
```

Computes `Θ_k` and `γ_k` per iteration for tree depths 2, 4, 6.
Results saved to `experiments/higgs_diagnostics_results.npz`.

### 3. Ablations

```bash
python experiments/ablations.py
```

Grid search over loss, η, depth, λ_base, and engine type.
Results saved to `experiments/ablations.csv`.

## Project Structure

```
grnbt/
  __init__.py              — Package exports
  losses.py                — Loss base + MSE, Charbonnier, BCE, CCE
  tree.py                  — NewtonTree weak learner (exact greedy splits)
  boosting.py              — VanillaNewtonBoosting + GradientRegularizedNewtonBoosting
  diagnostics.py           — Θ_k, γ_k, exact Newton, Lemma 4.2 checks
  datasets.py              — Wine Quality and Higgs loaders
  utils.py                 — Norms, thresholds, history logging
  extensions/              — Optional improvements (isolated from baseline)
    __init__.py
    histogram_tree.py      — Histogram-based split finding (non-paper)
tests/
  conftest.py              — Shared fixtures
  test_losses.py           — Loss correctness tests (16 cases)
  test_tree.py             — Tree structure tests (12 cases)
  test_boosting.py         — Engine loop tests (14 cases)
  test_diagnostics.py      — Diagnostic identity tests (13 cases)
  test_datasets.py         — Dataset loading tests (7 cases)
  test_utils.py            — Utility tests (14 cases)
  test_extensions.py       — Extension tests (4 cases)
experiments/
  wine_charbonnier.py      — Paper Fig 1 reproduction
  higgs_diagnostics.py     — Paper Fig 2 reproduction
  ablations.py             — Hyperparameter ablations
docs/
  math.md                  — Mathematical foundations, notation, formulas
  architecture.md          — System architecture and design decisions
  api.md                   — Complete API reference
  experiments.md           — Experiment descriptions and reproducibility notes
  fidelity.md              — Section-by-section fidelity report
README.md                  — Setup, usage, fidelity notes
FIDELITY_REPORT.md         — Legacy fidelity report (kept for reference)
pyproject.toml             — Project metadata and dependencies
.gitignore                 — Git ignore patterns
```

## Documentation

- **Math:** `docs/math.md` — Restated problem definition, loss formulas, tree
  equations, convergence theorems.
- **Architecture:** `docs/architecture.md` — Module responsibilities, data flow,
  design decisions, extension points.
- **API:** `docs/api.md` — Complete reference for all public classes and functions.
- **Experiments:** `docs/experiments.md` — Expected behavior, analysis scripts,
  reproducibility notes.
- **Fidelity:** `docs/fidelity.md` — Exact/approximate/assumed component tracking.

## Fidelity Notes

See `docs/fidelity.md` for a section-by-section comparison against the paper.
Key points:

- **Exact implementations:** leaf weight formula, split gain, adaptive `λ_k`,
  gradient/Hessian derivations (Appendix A), Proposition 5.1 scaling,
  exact Newton direction, `Θ_k`, `γ_k`, Lemma 4.2 checks.
- **Assumed / unspecified:** `λ_base` values (we use 0 for vanilla, 1e-3 for Higgs
  GRN), `min_samples_leaf` (default 1), exact Higgs subset size (10k diagnostics).
- **Extensions (not part of paper):** histogram-based split finding in
  `extensions/`, optional plotting, early-stopping helpers.

## License

MIT (reproduction code only; paper contents remain property of the authors).

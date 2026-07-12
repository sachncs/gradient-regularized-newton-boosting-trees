<p align="center">
  <h1 align="center">GRNBT</h1>
  <p align="center">Pure-Python reproduction of Gradient Regularized Newton Boosting Trees with global convergence.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachn-cs/gradient-regularized-newton-boosting-trees/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachn-cs/gradient-regularized-newton-boosting-trees/ci.yml?branch=main" alt="CI"></a>
    <a href="https://codecov.io/gh/sachn-cs/gradient-regularized-newton-boosting-trees"><img src="https://codecov.io/gh/sachn-cs/gradient-regularized-newton-boosting-trees/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://pypi.org/project/grnbt/"><img src="https://img.shields.io/pypi/v/grnbt.svg" alt="PyPI"></a>
    <a href="https://github.com/sachn-cs/gradient-regularized-newton-boosting-trees/stargazers"><img src="https://img.shields.io/github/stars/sachn-cs/gradient-regularized-newton-boosting-trees" alt="Stars"></a>
    <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/mypy-strict-green.svg" alt="Checked with mypy"></a>
  </p>
</p>

**GRNBT** is a pure-Python/NumPy reproduction of:

> N. Zozoulenko, D. Falkowski, T. Cass, L. Gonon,
> *Gradient Regularized Newton Boosting Trees with Global Convergence*,
> [arXiv:2605.00581v1](https://arxiv.org/abs/2605.00581v1)

It implements both **Vanilla Restricted Newton Boosting** (Algorithm 1)
and **Gradient Regularized Newton Boosting** (Algorithm 2) with a
multi-class extension, four analytic losses, a Newton tree weak learner,
and Hilbert-space diagnostics for verifying the paper's identities.
The package has **no framework dependencies** at runtime — only NumPy —
making it ideal as a faithful, transparent reference implementation.

---

## Features

- **Vanilla Newton Boosting** — Static ``λ_k = λ_base`` second-order GBDT, Algorithm 1.
- **Gradient Regularized Newton Boosting (GRN)** — Adaptive
  ``λ_k = λ_base + sqrt(M · ||g_k||)`` with ``M = M_0 · √N``
  (Proposition 5.1); achieves the paper's ``O(1/k²)`` global rate.
- **Multi-Class Boosting** — ``K``-class softmax output with
  vector-valued tree leaves and shared split structure across classes.
- **Four Loss Functions** — MSE, Charbonnier, Binary Cross-Entropy,
  Categorical Cross-Entropy; all with analytically known ``M_0``.
- **Hilbert-Space Diagnostics** — Exact Newton direction ``f = -g / (h + λ)``,
  cosine angle ``Θ_k``, weak gradient edge ``γ_k``, and Lemma 4.2 checks.
- **Exact Greedy Splits** — Closed-form leaf weights
  ``w = -Σg / (Σh + λ)`` and exact Newton gain, matching the paper.
- **Pure NumPy Implementation** — No C extensions, no framework
  dependencies; from-scratch research reproduction.
- **Comprehensive Test Suite** — 126 tests covering correctness,
  edge cases, mathematical identities, and validation.
- **Reproducible Experiments** — Scripts to reproduce Figures 1 and 2
  of the paper on Wine Quality and Higgs.
- **Optional Extensions** — Histogram-based split finding (not part of
  paper baseline) under ``grnbt.extensions``.

---

## Installation

### From PyPI

```bash
pip install grnbt
```

### From source

```bash
git clone https://github.com/sachn-cs/gradient-regularized-newton-boosting-trees.git
cd gradient-regularized-newton-boosting-trees
pip install -e .
```

### With dev dependencies

```bash
pip install -e ".[dev]"
```

**Requirements**: Python ≥ 3.9, NumPy ≥ 1.21.
Optional dependencies: scikit-learn ≥ 1.0 (datasets), matplotlib ≥ 3.4
(experiment plots), pytest ≥ 7 (development).

---

## Quick Start

### Vanilla Newton Boosting (Regression)

```python
import numpy as np
from grnbt import VanillaNewtonBoosting, MSELoss

rng = np.random.default_rng(42)
X = rng.standard_normal((500, 8))
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(500)

model = VanillaNewtonBoosting(
    loss=MSELoss(),
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
)
model.fit(X, y)
preds = model.predict(X)
print(f"final loss: {model.history.get('loss')[-1]:.6f}")
```

### Gradient Regularized Newton Boosting

```python
from grnbt import GradientRegularizedNewtonBoosting, CharbonnierLoss

model = GradientRegularizedNewtonBoosting(
    loss=CharbonnierLoss(),
    n_estimators=100,
    learning_rate=1.0,
    max_depth=4,
    lam_base=1e-3,
)
model.fit(X, y)
preds = model.predict(X)
```

### Multi-Class Classification

```python
import numpy as np
from grnbt import MultiClassNewtonBoosting, CategoricalCrossEntropyLoss

rng = np.random.default_rng(42)
X = rng.standard_normal((500, 8))
logits = np.stack([X[:, 0], -X[:, 0] + X[:, 1], X[:, 2]], axis=1)
y = np.argmax(logits, axis=1)

model = MultiClassNewtonBoosting(
    loss=CategoricalCrossEntropyLoss(n_classes=3),
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    lam_base=1e-3,
    n_classes=3,
)
model.fit(X, y)
probs = model.predict_proba(X)        # shape (500, 3), rows sum to 1
```

### Hilbert-Space Diagnostics

```python
from grnbt import GradientRegularizedNewtonBoosting, CharbonnierLoss
from grnbt.diagnostics import (
    cosine_angle_theta,
    exact_newton_direction,
    verify_lemma_4_2,
    weak_gradient_edge_gamma,
)

# Build a small ensemble and inspect Θ_k, γ_k per iteration.
model = GradientRegularizedNewtonBoosting(
    loss=CharbonnierLoss(), n_estimators=20, max_depth=3, lam_base=0.0,
)
model.fit(X, y)

# Replay the last iteration to compute diagnostics on demand.
F_last = np.zeros_like(y)
for k, tree in enumerate(model.trees):
    g = model.loss.gradient(y, F_last)
    h = model.loss.hessian(y, F_last)
    lam_k = model.history.get("lambda_k")[k]
    f_exact = exact_newton_direction(g, h, lam_k)
    f_weak = tree.predict(X)
    theta_k = cosine_angle_theta(g, h, f_exact, f_weak)
    gamma_k = weak_gradient_edge_gamma(g, h, lam_k, f_weak)
    F_last += model.learning_rate * f_weak
```

---

## Configuration

### Boosting Hyperparameters

| Parameter         | Default | Description |
|-------------------|---------|-------------|
| `loss`            | *required* | Loss function instance (`MSELoss`, `CharbonnierLoss`, `BinaryCrossEntropyLoss`, `CategoricalCrossEntropyLoss`). |
| `n_estimators`    | `100`    | Number of boosting rounds ``K``. |
| `learning_rate`   | `1.0`    | Step size ``η`` (paper default). |
| `max_depth`       | `3`      | Maximum tree depth (root depth `0`). |
| `min_samples_leaf`| `1`      | Minimum samples per leaf. |
| `lam_base`        | `0.0`    | Static ``λ_base`` regularization component. |
| `verbose`         | `False`  | Print metrics every 10 iterations. |

### Adaptive Regularization (GRN only)

GRN adds ``sqrt(M · ||g_k||)`` to ``λ_base``, where ``M = M_0 · √N``
and ``M_0`` is supplied by the loss (Appendix A):

| Loss                          | ``M_0`` |
|-------------------------------|---------|
| `MSELoss`                     | `0.0`   |
| `CharbonnierLoss`             | `1.0`   |
| `BinaryCrossEntropyLoss`      | `0.25`  |
| `CategoricalCrossEntropyLoss` | `0.25`  |

### Multi-Class Only

| Parameter       | Default | Description |
|-----------------|---------|-------------|
| `n_classes`     | `2`     | Number of target classes ``K`` (≥ 2). |
| `softmax_output`| `True`  | `predict()` returns probabilities; if `False`, returns logits. |

---

## Reproducible Experiments

```bash
# Wine Quality — Charbonnier Loss (Figure 1)
python experiments/wine_charbonnier.py
# produces experiments/wine_charbonnier_results.npz and .png

# Higgs — Weak Learner Diagnostics (Figure 2)
python experiments/higgs_diagnostics.py
# produces experiments/higgs_diagnostics_results.npz

# Hyperparameter Ablations
python experiments/ablations.py
# produces experiments/ablations.csv
```

See [`docs/experiments.md`](docs/experiments.md) for expected
behavior, analysis snippets, and reproducibility notes.

---

## Project Structure

```
gradient-regularized-newton-boosting-trees/
├── grnbt/                       # Core package (pure NumPy)
│   ├── __init__.py              # Public API exports
│   ├── losses.py                # MSE/Charbonnier/BCE/CCE + M_0
│   ├── tree.py                  # NewtonTree + MultiClassNewtonTree
│   ├── boosting.py              # Vanilla / GRN / Multi-class engines
│   ├── diagnostics.py           # Θ_k, γ_k, exact Newton, Lemma 4.2
│   ├── datasets.py              # Wine Quality and Higgs loaders
│   ├── utils.py                 # Norms, thresholds, history logger
│   └── extensions/
│       ├── __init__.py
│       └── histogram_tree.py    # Bin-based split finding (non-paper)
├── tests/                       # Test suite (126 cases)
│   ├── conftest.py              # Shared fixtures
│   ├── test_losses.py           # 20 cases — losses, M_0
│   ├── test_tree.py             # 16 cases — single-tree logic
│   ├── test_boosting.py         # 17 cases — scalar engines
│   ├── test_multiclass_tree.py  # 14 cases — K-class trees
│   ├── test_multiclass_boosting.py  # 13 cases — K-class engines
│   ├── test_diagnostics.py      # 17 cases — Θ, γ, Lemma 4.2
│   ├── test_datasets.py         # 7 cases  — Wine/Higgs loaders
│   ├── test_utils.py            # 18 cases — utilities
│   └── test_extensions.py       # 4 cases  — histogram tree
├── experiments/                 # Paper reproductions
│   ├── wine_charbonnier.py      # Fig. 1 (Wine + Charbonnier)
│   ├── higgs_diagnostics.py     # Fig. 2 (Higgs + BCE)
│   └── ablations.py             # Loss / η / depth / λ_base grid
├── docs/                        # Documentation
│   ├── math.md                  # Mathematical foundations
│   ├── architecture.md          # Module responsibilities & data flow
│   ├── api.md                   # Complete API reference
│   ├── experiments.md           # Experiment descriptions
│   └── fidelity.md              # Paper-section fidelity report
├── pyproject.toml               # Build & tool config
└── README.md
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest tests/ -v --cov=grnbt --cov-report=term-missing

# Check import sorting
isort --check-only --diff grnbt tests experiments

# Check formatting
black --check --diff grnbt tests experiments

# Type check
mypy grnbt --ignore-missing-imports

# All checks
isort --check-only grnbt tests experiments \
  && black --check grnbt tests experiments \
  && mypy grnbt --ignore-missing-imports \
  && pytest tests/ -v
```

### Code Style

- Line length: 88 (black default)
- Quotes: double (`"`)
- Formatting: [black](https://black.readthedocs.io) + [isort](https://pycqa.github.io/isort/)
- Type hints: required on all public signatures; `mypy --strict`
- Docstrings: Google-style with **what** and **why**
- No semi-private naming (`_foo`) — all identifiers are public
- Numerical conventions documented in [`docs/math.md`](docs/math.md)

### Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add multi-class CCE Hessian extraction
fix: handle zero-divisor in Charbonnier loss with hypot
docs: document Proposition 5.1 scaling
refactor: split boosting.py into per-engine modules
test: add Lemma 4.2 identity parity tests
chore: update pre-commit hooks
```

---

## Mathematical Guarantees

This codebase is a **byte-faithful** reproduction of the paper's formulas
(verified in [`docs/fidelity.md`](docs/fidelity.md)):

1. **Closed-form leaf weight** — `w = -Σg / (Σh + λ)` (Algorithm 1 line 4).
2. **Exact Newton gain** — `½[ (G_L)²/(H_L+λ) + (G_R)²/(H_R+λ) - (G)²/(H+λ) ]`
   used for greedy split selection.
3. **Adaptive ``λ_k`` (Proposition 5.1)** —
   ``λ_k = λ_base + sqrt(M · ||g_k||)`` with ``M = M_0 · √N``.
4. **Lemma 4.2 identities** — Both ``λ ||f|| ≤ ||g||`` and
   ``||f||²_K = -⟨g, f⟩`` are numerically checked by
   `verify_lemma_4_2`.
5. **Analytic Hessian Lipschitz constants** — ``M_0 ∈ {0, 1/4, 1}``
   for the four losses (Appendix A).
6. **Multi-class block Hessian** — ``diag(p) - p p^T`` per sample,
   extracted to diagonals for the K-class tree builder.

### Documentation

| Document | Description |
|----------|-------------|
| [Mathematical Foundations](docs/math.md) | Restated problem, loss formulas, tree equations, convergence theorems. |
| [Architecture](docs/architecture.md) | Module responsibilities, data flow, design decisions, extension points. |
| [API Reference](docs/api.md) | Complete reference for every public class and function. |
| [Experiments](docs/experiments.md) | Expected behavior, analysis, reproducibility notes. |
| [Fidelity Report](docs/fidelity.md) | Section-by-section fidelity report against the paper. |

---

## Tech Stack

| Category        | Technology                                   |
|-----------------|----------------------------------------------|
| Language        | Python 3.9+                                  |
| Numerical       | [NumPy](https://numpy.org/) ≥ 1.21           |
| Datasets (opt.) | [scikit-learn](https://scikit-learn.org/) ≥ 1.0 |
| Plotting (opt.) | [matplotlib](https://matplotlib.org/) ≥ 3.4  |
| Lint/Format     | [black](https://black.readthedocs.io/) + [isort](https://pycqa.github.io/isort/) |
| Type Check      | [mypy](https://mypy-lang.org/) (strict)        |
| Testing         | [pytest](https://docs.pytest.org/) + pytest-cov |
| Build           | setuptools (PEP 621)                         |
| Pre-commit      | [pre-commit](https://pre-commit.com) hooks    |

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and milestones.

- **v0.1.0** — Current release: paper-faithful reproduction, 126 tests,
  three experiment scripts, full docs.
- **v0.2.0** — Column subsampling, row subsampling, early stopping,
  warm-start support.
- **v0.3.0** — Parallel tree construction, additional loss functions.
- **v1.0.0** — Stable API, scikit-learn-compatible API, PyPI release
  automation, GPU optional backends.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup (PEP 621, black, isort, mypy, pytest)
- Pull request process and review checklist
- Coding standards and paper-fidelity expectations
- Test expectations (≥ 90% coverage on new code)

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

## Security

Report vulnerabilities to **sachncs@gmail.com** — see [SECURITY.md](SECURITY.md).

## Citation

If you use this software in your research, please cite:

```bibtex
@article{zozoulenko2026grnbt,
  title  = {Gradient Regularized Newton Boosting Trees with Global Convergence},
  author = {Zozoulenko, Nikita and Falkowski, Daniel and Cass, Thomas and Gonon, Lucien},
  journal = {arXiv preprint arXiv:2605.00581v1},
  year   = {2026}
}
```

## License

[MIT](LICENSE) © 2026 Sachin.

**Note:** This is reproduction code only; the paper contents remain
property of the original authors.

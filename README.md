# GRNBT — Gradient Regularized Newton Boosting Trees

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/sachn-cs/gradient-regularized-newton-boosting-trees/actions/workflows/ci.yml/badge.svg)](https://github.com/sachn-cs/gradient-regularized-newton-boosting-trees/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sachn-cs/gradient-regularized-newton-boosting-trees/branch/main/graph/badge.svg)](https://codecov.io/gh/sachn-cs/gradient-regularized-newton-boosting-trees)
[![PyPI version](https://img.shields.io/pypi/v/grnbt.svg)](https://pypi.org/project/grnbt/)

Pure Python reproduction of:

> N. Zozoulenko, D. Falkowski, T. Cass, L. Gonon,
> "Gradient Regularized Newton Boosting Trees with Global Convergence",
> [arXiv:2605.00581v1](https://arxiv.org/abs/2605.00581v1)

## Features

- **Vanilla Newton Boosting** — Standard second-order GBDT with static L2 regularization
- **Gradient Regularized Newton (GRN)** — Adaptive regularization `λ_k = λ_base + sqrt(M ||g_k||)` achieving O(1/k²) convergence
- **Multi-class classification** — K-class boosting with vector-valued trees and softmax output
- **Four loss functions** — MSE, Charbonnier, Binary Cross-Entropy, Categorical Cross-Entropy with analytically known M₀ constants
- **Hilbert-space diagnostics** — Exact Newton directions, cosine angles Θ_k, weak gradient edges γ_k, and Lemma 4.2 verification
- **Pure NumPy implementation** — No framework dependencies, from-scratch research reproduction
- **Comprehensive test suite** — 126 tests covering correctness, edge cases, and mathematical identities
- **Reproducible experiments** — Scripts to reproduce Figures 1 and 2 from the paper

## Installation

### Quick Install

```bash
pip install -e ".[dev]"
```

### From Source

```bash
git clone https://github.com/sachn-cs/gradient-regularized-newton-boosting-trees.git
cd gradient-regularized-newton-boosting-trees
pip install -e ".[dev]"
```

### Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `numpy>=1.21.0` | Yes | Core numerical operations |
| `scikit-learn>=1.0.0` | No | Dataset fetching |
| `matplotlib>=3.4.0` | No | Experiment plotting |
| `pytest>=7.0.0` | No | Test runner |

## Usage

### Basic Example

```python
import numpy as np
from grnbt import GradientRegularizedNewtonBoosting, CharbonnierLoss

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = X @ np.random.randn(10) + 0.1 * np.random.randn(1000)

# Train GRN boosting model
model = GradientRegularizedNewtonBoosting(
    loss=CharbonnierLoss(),
    n_rounds=100,
    learning_rate=0.1,
    max_depth=4,
    lambda_base=1e-3,
)

model.fit(X, y)
predictions = model.predict(X)
```

### Vanilla Newton Boosting

```python
from grnbt import VanillaNewtonBoosting, MSELoss

model = VanillaNewtonBoosting(
    loss=MSELoss(),
    n_rounds=100,
    learning_rate=0.1,
    max_depth=4,
)

model.fit(X, y)
```

### Diagnostics

```python
from grnbt import GradientRegularizedNewtonBoosting, CharbonnierLoss
from grnbt.diagnostics import compute_cosine_angle, compute_weak_gradient_edge

model = GradientRegularizedNewtonBoosting(
    loss=CharbonnierLoss(),
    n_rounds=50,
    learning_rate=0.1,
    max_depth=4,
    lambda_base=1e-3,
    record_diagnostics=True,
)

model.fit(X, y)

# Compute diagnostics
theta_k = compute_cosine_angle(model)
gamma_k = compute_weak_gradient_edge(model)
```

### Multi-Class Classification

```python
import numpy as np
from grnbt import MultiClassNewtonBoosting, CategoricalCrossEntropyLoss

# Generate synthetic 3-class data
np.random.seed(42)
X = np.random.randn(500, 8)
logits = np.stack([X[:, 0], -X[:, 0] + X[:, 1], X[:, 2]], axis=1)
y = np.argmax(logits, axis=1)

# Train multi-class GRN boosting model
model = MultiClassNewtonBoosting(
    loss=CategoricalCrossEntropyLoss(n_classes=3),
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    lam_base=1e-3,
    n_classes=3,
)

model.fit(X, y)

# Get probabilities
probs = model.predict(X)  # (n_samples, 3) probabilities summing to 1
logits = model.predict_proba(X)  # Same as predict with softmax_output=True
```

## Experiments

### Wine Quality — Charbonnier Loss (Figure 1)

```bash
python experiments/wine_charbonnier.py
```

Compares Vanilla Newton, GRN, and static high-lambda baseline. Results saved to `experiments/wine_charbonnier_results.npz`.

### Higgs — Weak Learner Diagnostics (Figure 2)

```bash
python experiments/higgs_diagnostics.py
```

Computes Θ_k and γ_k per iteration for tree depths 2, 4, 6. Results saved to `experiments/higgs_diagnostics_results.npz`.

### Ablations

```bash
python experiments/ablations.py
```

Grid search over loss, η, depth, λ_base, and engine type. Results saved to `experiments/ablations.csv`.

## Project Structure

```
grnbt/
├── __init__.py          # Package exports
├── losses.py            # Loss base + MSE, Charbonnier, BCE, CCE
├── tree.py              # NewtonTree + MultiClassNewtonTree weak learners
├── boosting.py          # VanillaNewton + GRN + MultiClassNewton boosting engines
├── diagnostics.py       # Θ_k, γ_k, exact Newton, Lemma 4.2 checks
├── datasets.py          # Wine Quality and Higgs loaders
├── utils.py             # Norms, thresholds, history logging
└── extensions/
    ├── __init__.py
    └── histogram_tree.py  # Histogram-based split finding (non-paper)
tests/
├── conftest.py              # Shared fixtures
├── test_losses.py           # Loss correctness tests (16 cases)
├── test_tree.py             # Tree structure tests (12 cases)
├── test_boosting.py         # Engine loop tests (14 cases)
├── test_multiclass_tree.py  # Multi-class tree tests (14 cases)
├── test_multiclass_boosting.py  # Multi-class boosting tests (13 cases)
├── test_diagnostics.py      # Diagnostic identity tests (13 cases)
├── test_datasets.py         # Dataset loading tests (7 cases)
├── test_utils.py            # Utility tests (14 cases)
└── test_extensions.py       # Extension tests (4 cases)
experiments/
├── wine_charbonnier.py  # Paper Fig 1 reproduction
├── higgs_diagnostics.py # Paper Fig 2 reproduction
└── ablations.py         # Hyperparameter ablations
docs/
├── math.md              # Mathematical foundations, notation, formulas
├── architecture.md      # System architecture and design decisions
├── api.md               # Complete API reference
├── experiments.md       # Experiment descriptions and reproducibility notes
└── fidelity.md          # Section-by-section fidelity report
```

## Documentation

| Document | Description |
|----------|-------------|
| [Mathematical Foundations](docs/math.md) | Restated problem definition, loss formulas, tree equations, convergence theorems |
| [Architecture](docs/architecture.md) | Module responsibilities, data flow, design decisions, extension points |
| [API Reference](docs/api.md) | Complete reference for all public classes and functions |
| [Experiments](docs/experiments.md) | Expected behavior, analysis scripts, reproducibility notes |
| [Fidelity Report](docs/fidelity.md) | Exact/approximate/assumed component tracking against the paper |

## Development

### Commands

| Command | Description |
|---------|-------------|
| `pip install -e ".[dev]"` | Install with development dependencies |
| `pytest tests/ -v` | Run test suite |
| `isort --check-only --diff grnbt tests experiments` | Check import sorting |
| `black --check --diff grnbt tests experiments` | Check code formatting |
| `mypy grnbt --ignore-missing-imports` | Type checking |
| `isort grnbt tests experiments` | Auto-sort imports |
| `black grnbt tests experiments` | Auto-format code |

### Running All Checks

```bash
isort --check-only --diff grnbt tests experiments
black --check --diff grnbt tests experiments
mypy grnbt --ignore-missing-imports
pytest tests/ -v
```

### Auto-Format

```bash
isort grnbt tests experiments
black grnbt tests experiments
```

## Tech Stack

- **Language:** Python 3.9+
- **Core Dependency:** NumPy
- **Build System:** setuptools (PEP 621)
- **Testing:** pytest, pytest-cov
- **Linting:** black, isort
- **Type Checking:** mypy (strict mode)
- **CI:** GitHub Actions (lint, type-check, tests, coverage)
- **Pre-commit:** pre-commit hooks for code quality

## Roadmap

- [x] Multi-output / multi-class tree support
- [ ] Column subsampling
- [ ] Row subsampling
- [ ] Early stopping
- [ ] Warm-start support
- [ ] Parallel tree construction
- [ ] Additional loss functions
- [ ] Scikit-learn compatible API
- [ ] PyPI publishing via automated releases

## Fidelity Notes

See [docs/fidelity.md](docs/fidelity.md) for a section-by-section comparison against the paper.

Key points:
- **Exact implementations:** leaf weight formula, split gain, adaptive λ_k, gradient/Hessian derivations (Appendix A), Proposition 5.1 scaling, exact Newton direction, Θ_k, γ_k, Lemma 4.2 checks
- **Assumed / unspecified:** λ_base values (we use 0 for vanilla, 1e-3 for Higgs GRN), min_samples_leaf (default 1), exact Higgs subset size (10k diagnostics)
- **Extensions (not part of paper):** histogram-based split finding in `extensions/`, optional plotting, early-stopping helpers

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on:

- Fork and branch workflow
- Commit conventions
- Pull request process
- Coding standards
- Testing requirements

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Security

For reporting security vulnerabilities, please see our [Security Policy](SECURITY.md).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

**Note:** This is reproduction code only; the paper contents remain property of the original authors.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{zozoulenko2026grnbt,
  title={Gradient Regularized Newton Boosting Trees with Global Convergence},
  author={Zozoulenko, Nikita and Falkowski, Daniel and Cass, Thomas and Gonon, Lucien},
  journal={arXiv preprint arXiv:2605.00581v1},
  year={2026}
}
```

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive usage examples in docstrings for boosting engines (`VanillaNewtonBoosting`, `GradientRegularizedNewtonBoosting`), losses (`MSELoss`), and utilities (`empirical_norm`, `History`)
- Docstrings for `Node.__init__` and `MultiClassNode.__init__`
- Public re-exports from `grnbt.__init__`: `Node`, `MultiClassNode`, `validate_inputs`, `validate_diagonal_inputs`
- Enhanced test docstrings with algorithmic explanations, paper references, and full cross-references using `:class:`, `:func:`, `:meth:` roles

### Changed

- Renamed all semi-private (`_`-prefixed) helpers to public API:
  - `_Node` -> `Node`, `_MultiClassNode` -> `MultiClassNode`
  - `_build` -> `build`, `_predict_one` -> `predict_one`, `_validate_fit_inputs` -> `validate_fit_inputs`
  - `_compute_lambda` -> `compute_lambda`, `_init_prediction` -> `init_prediction`
  - `_softmax` -> `softmax`, `_extract_hessian_diagonal` -> `extract_hessian_diagonal`
  - `_compute_lambda_for_multiclass` -> `compute_lambda_for_multiclass`
  - `_validate_multiclass_labels` -> `validate_multiclass_labels`
  - `_validate_inputs` -> `validate_inputs`, `_validate_diagonal_inputs` -> `validate_diagonal_inputs`
  - `_data` -> `data` attribute on `History`
  - `_max_depth` -> `max_depth`, `_check_leaf_sizes` -> `check_leaf_sizes` (tests)
  - `_make_synthetic` -> `make_synthetic` (experiments)
- Switched docstring style recommendation from NumPy to Google in CONTRIBUTING.md

### Atomic commits in this release

| Commit | Date (UTC+05:30) | Subject |
|--------|------------------|---------|
| `c8e657a` | 2026-07-12 14:11:03 +0530 | docs: add usage examples, docstrings, and improve cross-references |
| `c4bcb9f` | 2026-07-12 14:11:14 +0530 | refactor: promote semi-private helpers to public API |

## [0.1.0] - 2026-05-06

### Added

- Initial release of GRNBT (Gradient Regularized Newton Boosting Trees)
- Pure Python reproduction of arXiv:2605.00581v1
- Four loss functions: MSE, Charbonnier, BCE, CCE with analytical M_0 constants
- NewtonTree weak learner with exact greedy split finding
- Vanilla Newton Boosting engine with static L2 regularization
- Gradient Regularized Newton Boosting with adaptive regularization
- Hilbert-space diagnostics: exact Newton directions, cosine angles, weak gradient edges
- Numerical verification of Lemma 4.2 identities
- Dataset loaders for Wine Quality and Higgs datasets
- Three experiment scripts reproducing paper figures
- Comprehensive test suite with 80 test cases
- CI pipeline with lint, type-check, tests, and experiment smoke tests
- Complete API documentation
- Mathematical foundations documentation
- Architecture and design documentation
- Section-by-section fidelity report against the paper

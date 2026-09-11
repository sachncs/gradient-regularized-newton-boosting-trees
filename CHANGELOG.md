# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `grnbt/py.typed` marker and `[tool.setuptools.package-data]` in
  `pyproject.toml` so mypy downstream consumers can resolve strict
  annotations (PEP 561).
- `__repr__` methods on `Loss` and every subclass, `Node` and
  `MultiClassNode`, `NewtonTree` and `MultiClassNewtonTree`,
  `BaseBoosting` and the three subclasses, and `History`.
- `warn` keyword-only flag on `cosine_angle_theta` and
  `weak_gradient_edge_gamma` to emit `UserWarning` when the
  zero-denominator fallback fires.
- `Loss.hessian_diagonal` (with `CategoricalCrossEntropyLoss` override
  that computes `p * (1 - p)` without allocating the full `(N, K, K)`
  block).
- `local_path` offline override on `load_wine_quality` and
  `load_higgs_subset` for environments without network access.
- `.github/ISSUE_TEMPLATE/question.md` so the issue chooser offers a
  dedicated usage-question template.
- `ROADMAP.md` resolving the previously-broken README link.
- Three regression tests asserting the paper's headline Charbonnier
  behaviour: `test_vanilla_diverges_on_charbonnier`,
  `test_grn_converges_on_charbonnier`, and
  `test_static_high_lambda_biased_vs_grn`.

### Changed

- Replaced unverified arXiv ID `2605.00581v1` with a `(preprint, id
  verification pending)` placeholder across `README.md`,
  `pyproject.toml`, `CHANGELOG.md`, `docs/fidelity.md`,
  `docs/math.md`, `grnbt/__init__.py`, and `grnbt/losses.py`. The
  cited id could not be verified against arXiv.org from this audit;
  once the correct id is confirmed, update all eight locations in a
  single follow-up commit.
- `pyproject.toml` `[project.urls]` Documentation and Changelog URLs
  now use `/master/` instead of `/main/` to match the actual default
  branch.
- Adaptive `λ_k` now uses the empirical-RMS gradient norm
  `||g_k||_H = ||g_k|| / √N` matching the paper's notation (was:
  standard L²). `grnbt.utils.empirical_norm` is wired into
  `compute_lambda` and `compute_lambda_for_multiclass`.
- `BaseBoosting.fit` is now a template method exposing seven override
  points (`_init_f0`, `_compute_lambda`, `_make_tree`, `_tree_input`,
  `_predict_tree`, `_update_f`, `_grad_norm`). `MultiClassNewtonBoosting`
  inherits the loop and only overrides the multi-class-specific hooks.
- `MultiClassNewtonBoosting.extract_hessian_diagonal` now strictly
  requires a 3-D tensor; the unreachable 2-D passthrough branch is
  removed.
- `MultiClassNewtonBoosting.validate_fit_inputs` moves the integer-dtype
  check above the NaN check and drops the silent `.astype(float)`
  coercion.
- `BaseBoosting.predict` drops the unreachable `F0.ndim == 0` branch
  and documents that `init_prediction`'s mean is what predict-time
  consumes.
- `experiments/ablations.py` now runs each of the 108 configurations
  with three seeds (`42`, `43`, `44`) and writes one row per
  `(config, seed)` pair to `experiments/ablations.csv`. The previous
  single-seed run could not distinguish stable results from lucky
  draws.
- `experiments/wine_charbonnier.py` prints a one-line message when
  matplotlib is missing instead of silently skipping the plot.
- Test suite now contains 139 cases (up from 126). The README's hero
  "126 tests" claim is replaced with the up-to-date count.

## [0.1.0] - 2026-05-06

### Added

- Comprehensive usage examples in docstrings for boosting engines
  (`VanillaNewtonBoosting`, `GradientRegularizedNewtonBoosting`),
  losses (`MSELoss`), and utilities (`empirical_norm`, `History`).
- Docstrings for `Node.__init__` and `MultiClassNode.__init__`.
- Public re-exports from `grnbt.__init__`: `Node`, `MultiClassNode`,
  `validate_inputs`, `validate_diagonal_inputs`.
- Enhanced test docstrings with algorithmic explanations, paper
  references, and full cross-references using `:class:`,`,
  `:meth:` roles.
- Initial release of GRNBT (Gradient Regularized Newton Boosting
  Trees) — pure-Python reproduction of Zozoulenko et al. (2026)
  preprint.
- Four loss functions: MSE, Charbonnier, BCE, CCE with analytical
  `M_0` constants.
- `NewtonTree` weak learner with exact greedy split finding.
- Vanilla Newton Boosting engine with static L2 regularization.
- Gradient Regularized Newton Boosting with adaptive regularization.
- Hilbert-space diagnostics: exact Newton directions, cosine angles,
  weak gradient edges.
- Numerical verification of Lemma 4.2 identities.
- Dataset loaders for Wine Quality and Higgs datasets.
- Three experiment scripts reproducing paper figures.
- Comprehensive test suite with 80 test cases.
- CI pipeline with lint, type-check, tests, and experiment smoke
  tests.
- Complete API documentation, mathematical foundations
  documentation, architecture and design documentation, and
  section-by-section fidelity report against the paper.

### Changed

- Renamed all semi-private (`_`-prefixed) helpers to public API:
  - `_Node` -> `Node`, `_MultiClassNode` -> `MultiClassNode`
  - `_build` -> `build`, `_predict_one` -> `predict_one`,
    `_validate_fit_inputs` -> `validate_fit_inputs`
  - `_compute_lambda` -> `compute_lambda`,
    `_init_prediction` -> `init_prediction`
  - `_softmax` -> `softmax`,
    `_extract_hessian_diagonal` -> `extract_hessian_diagonal`
  - `_compute_lambda_for_multiclass` -> `compute_lambda_for_multiclass`
  - `_validate_multiclass_labels` -> `validate_multiclass_labels`
  - `_validate_inputs` -> `validate_inputs`,
    `_validate_diagonal_inputs` -> `validate_diagonal_inputs`
  - `_data` -> `data` attribute on `History`
  - `_max_depth` -> `max_depth`,
    `_check_leaf_sizes` -> `check_leaf_sizes` (tests)
  - `_make_synthetic` -> `make_synthetic` (experiments)
- Switched docstring style recommendation from NumPy to Google in
  `CONTRIBUTING.md`.

### Atomic commits in this release

| Commit | Date (UTC+05:30) | Subject |
|--------|------------------|---------|
| `c8e657a` | 2026-07-12 14:11:03 +0530 | docs: add usage examples, docstrings, and improve cross-references |
| `c4bcb9f` | 2026-07-12 14:11:14 +0530 | refactor: promote semi-private helpers to public API |
| `601f740` | 2026-07-12 14:11:57 +0530 | docs: update CHANGELOG.md with atomic commits for this release |
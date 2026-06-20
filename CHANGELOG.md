# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Multi-class Newton boosting with `MultiClassNewtonTree` and `MultiClassNewtonBoosting`
- PyPI release automation workflow (`.github/workflows/release.yml`)
- GitHub Pages documentation hosting workflow (`.github/workflows/docs.yml`)
- Dependabot auto-merge workflow for minor/patch updates
- Test coverage reporting with pytest-cov and Codecov integration
- Pre-commit hooks configuration (`.pre-commit-config.yaml`)
- MIT License file
- CONTRIBUTING.md with development guidelines
- CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
- SECURITY.md with vulnerability reporting process
- .editorconfig for consistent formatting
- .gitattributes for line ending normalization
- GitHub issue templates (bug report, feature request)
- Pull request template
- Dependabot configuration for automated dependency updates
- GitHub funding configuration
- CHANGELOG.md
- 46 new tests for multi-class tree and boosting (total: 126)

### Changed

- Updated pyproject.toml with correct repository URLs and author information
- Improved README.md with world-class documentation standards
- Updated CI workflow with coverage reporting and Codecov integration
- Added pytest-cov to dev dependencies

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

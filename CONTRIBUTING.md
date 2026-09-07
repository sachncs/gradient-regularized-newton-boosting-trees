# Contributing to GRNBT

Thank you for considering contributing to GRNBT! This document outlines the guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Branch Naming](#branch-naming)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/gradient-regularized-newton-boosting-trees.git
   cd gradient-regularized-newton-boosting-trees
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/sachncs/gradient-regularized-newton-boosting-trees.git
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feat/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check [existing issues](https://github.com/sachncs/gradient-regularized-newton-boosting-trees/issues) to avoid duplicates.

When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, package version)
- Minimal code example if applicable

### Suggesting Features

Feature requests are welcome. Please open an issue with:

- Clear description of the proposed feature
- Motivation or use case
- Possible implementation approach if you have one

### Submitting Changes

1. Ensure your code follows the project's coding standards
2. Add or update tests for your changes
3. Update documentation if needed
4. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

1. Install the package in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Verify your setup:
   ```bash
   pytest tests/ -v
   ```

### Development Tools

The project uses the following tools (all configured in `pyproject.toml`):

- **black** - Code formatting
- **isort** - Import sorting
- **mypy** - Type checking
- **pytest** - Testing

Run all checks:
```bash
isort --check-only --diff grnbt tests experiments
black --check --diff grnbt tests experiments
mypy grnbt --ignore-missing-imports
pytest tests/ -v
```

Auto-format code:
```bash
isort grnbt tests experiments
black grnbt tests experiments
```

## Branch Naming

Use descriptive branch names with prefixes:

| Prefix | Purpose |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance tasks |

Example: `feat/add-early-stopping`

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, missing semicolons, etc.) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or correcting tests |
| `chore` | Maintenance tasks (dependencies, CI, etc.) |
| `revert` | Reverts a previous commit |

### Examples

```
feat: add early stopping support

fix: correct Hessian computation for Charbonnier loss

docs: update API reference for NewtonTree

test: add edge case tests for empty input arrays

chore: update numpy minimum version to 1.24
```

## Pull Request Process

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Ensure all checks pass**:
   ```bash
   isort --check-only --diff grnbt tests experiments
   black --check --diff grnbt tests experiments
   mypy grnbt --ignore-missing-imports
   pytest tests/ -v
   ```

3. **Push your changes** and create a pull request on GitHub

4. **Fill out the PR template** with:
   - Summary of changes
   - Related issue (if any)
   - Description of testing performed
   - Checklist completion

5. **Respond to review feedback** promptly

### PR Guidelines

- Keep PRs focused on a single change
- Write clear commit messages
- Add tests for new functionality
- Update documentation as needed
- Ensure CI passes before requesting review

## Coding Standards

### Python Style

- Follow PEP 8 (enforced by black with 88-character line length)
- Use type hints for all public functions (enforced by mypy strict mode)
- Write docstrings for public APIs (Google style)
- Keep functions focused and reasonably sized

### Imports

- Sort imports with isort (black-compatible profile)
- Group imports: standard library, third-party, local
- Avoid wildcard imports

### Naming Conventions

- `snake_case` for functions, methods, and variables
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- Descriptive names over abbreviated ones

### Error Handling

- Use descriptive error messages
- Validate inputs at public API boundaries
- Use appropriate exception types

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_<module>.py`
- Name test functions `test_<description>.py`
- Use descriptive test names
- One assertion per test when possible
- Use fixtures from `conftest.py` for shared setup

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_losses.py -v

# Run with coverage
pytest tests/ -v --cov=grnbt
```

### Test Categories

- **Correctness**: Gradient/Hessian accuracy via finite differences
- **Properties**: Mathematical invariants and identities
- **Edge cases**: Empty inputs, NaN, Inf, mismatched shapes
- **Integration**: Full boosting loop behavior

## Documentation

- Update `docs/api.md` for new public APIs
- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update README.md if adding new features or changing setup instructions
- Include examples in docstrings where helpful

### Docstring Format

```python
def function_name(param1: int, param2: str = "default") -> bool:
    """Short description of the function.

    Longer description if needed, explaining the purpose
    and behavior in more detail.

    Args:
        param1: Description of param1.
        param2: Description of param2. Default is "default".

    Returns:
        Description of return value.

    Raises:
        ValueError: If param1 is negative.

    Examples:
        >>> result = function_name(42, "hello")
        >>> print(result)
        True
    """
```

## Questions?

If you have questions about contributing, feel free to open an issue with the "question" label.

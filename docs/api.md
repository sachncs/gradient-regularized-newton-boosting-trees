# API Reference

Complete reference for all public classes and functions in `grnbt`.

---

## `grnbt.losses`

### `class Loss(ABC)`

Abstract base class for all loss functions.

#### Methods

- `loss(y_true: np.ndarray, y_pred: np.ndarray) -> float`
  - Compute the empirical average loss.

- `gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray`
  - Gradient of the empirical risk w.r.t. `y_pred`.

- `hessian(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray`
  - Hessian (or diagonal) of the empirical risk.

- `hessian_lipschitz_constant() -> float`
  - Return `M_0`, the pointwise Hessian Lipschitz constant.

- `empirical_risk_lipschitz(n_samples: int) -> float`
  - Return `M = M_0 * sqrt(n_samples)` per Proposition 5.1.

### `class MSELoss`

Mean squared error. `M_0 = 0.0`.

### `class CharbonnierLoss`

Charbonnier loss: `sqrt(1 + d^2) - 1` where `d = y_pred - y_true`.
`M_0 = 1.0`.

### `class BinaryCrossEntropyLoss`

Binary cross-entropy on logits. Labels must be `{0, 1}`.
`M_0 = 0.25`.

### `class CategoricalCrossEntropyLoss`

Categorical cross-entropy on logits. Labels must be integers in `[0, n_classes-1]`.

**Constructor:** `CategoricalCrossEntropyLoss(n_classes: int)`
- `n_classes`: number of classes `K >= 2`.

`M_0 = 0.25`.

---

## `grnbt.tree`

### `class NewtonTree`

Decision tree weak learner for Newton boosting.

**Constructor:** `NewtonTree(max_depth=3, min_samples_leaf=1, min_gain=1e-7)`
- `max_depth`: maximum depth (root = depth 0).
- `min_samples_leaf`: minimum samples in any leaf.
- `min_gain`: minimum gain to accept a split.

#### Methods

- `fit(x: np.ndarray, g: np.ndarray, h: np.ndarray, lam: float) -> NewtonTree`
  - Fit the tree to gradients `g`, Hessians `h`, with regularization `lam >= 0`.

- `predict(x: np.ndarray) -> np.ndarray`
  - Predict leaf weights for samples in `x`. Returns shape `(n_samples,)`.

**Raises:**
- `TypeError` if inputs are not NumPy arrays.
- `ValueError` for shape mismatches, empty data, negative `lam`, NaN/Inf.

---

## `grnbt.boosting`

### `class BaseBoosting`

Base class for boosting engines. Not instantiated directly.

**Constructor:** `BaseBoosting(loss, n_estimators=100, learning_rate=1.0, max_depth=3, min_samples_leaf=1, lam_base=0.0, verbose=False)`

#### Methods

- `fit(x: np.ndarray, y: np.ndarray) -> BaseBoosting`
  - Fit the ensemble. Returns self.

- `predict(x: np.ndarray) -> np.ndarray`
  - Predict on new data. Returns shape `(n_samples,)`.

#### Attributes

- `trees`: list of fitted `NewtonTree` instances.
- `F0`: initial prediction vector.
- `history`: `History` object with logged metrics.

### `class VanillaNewtonBoosting(BaseBoosting)`

Vanilla Restricted Newton Boosting. Uses static `λ_k = lam_base`.

### `class GradientRegularizedNewtonBoosting(BaseBoosting)`

Gradient Regularized Newton Boosting. Uses adaptive `λ_k = lam_base + sqrt(M ||g_k||)`.

---

## `grnbt.diagnostics`

### `exact_newton_direction(g, h, lam) -> np.ndarray`

Compute `f = -g / (h + lam)` element-wise. Adds `1e-12` epsilon to denominator.

**Args:**
- `g`: gradient vector.
- `h`: Hessian diagonal.
- `lam`: regularization (must be >= 0).

**Returns:** exact Newton direction.

### `cosine_angle_theta(g, h, f_exact, f_weak) -> float`

Compute `Θ_k`, the cosine angle in the H-induced inner product.

**Returns:** value in `[-1, 1]`. Returns `1.0` for zero directions.

### `weak_gradient_edge_gamma(g, h, lam, f_weak) -> float`

Compute `γ_k`, the weak gradient edge.

**Returns:** value in `[0, 1]`. Returns `1.0` for zero gradient.

### `verify_lemma_4_2(g, h, lam, f_weak) -> dict[str, bool]`

Numerically verify Lemma 4.2 identities.

**Returns:** `{"lambda_norm_bound": bool, "k_norm_identity": bool}`

---

## `grnbt.datasets`

### `load_wine_quality() -> tuple[np.ndarray, np.ndarray]`

Load and standardize the Wine Quality dataset.

**Returns:** `(x, y)` where `x` is standardized.

**Raises:** `RuntimeError` if all data sources fail.

### `load_higgs_subset(n_samples=100_000) -> tuple[np.ndarray, np.ndarray]`

Load a subset of the Higgs dataset.

**Args:**
- `n_samples`: positive integer.

**Returns:** `(x, y)` with `x.shape[1] == 28` and binary `y`.

**Raises:** `ValueError` if `n_samples <= 0`; `RuntimeError` if loading fails.

---

## `grnbt.utils`

### `empirical_norm(v, weights=None) -> float`

Compute the empirical L2 norm.

- Without weights: `||v|| / sqrt(N)` (RMS).
- With weights: `sqrt(sum(weights * v^2))`.

### `unique_thresholds(x) -> np.ndarray`

Return midpoints between sorted unique values of a 1-D array.

**Returns:** empty array if fewer than 2 unique values.

### `class History`

Lightweight metric logger.

#### Methods

- `log(key: str, value: float) -> None`
  - Append a value. Key must be non-empty string; value must be finite.

- `get(key: str) -> list[float]`
  - Retrieve series. Empty list for unknown keys.

- `keys() -> list[str]`
  - Sorted list of recorded metric names.

- `as_dict() -> dict[str, list[float]]`
  - Shallow copy of all recorded data.

# System Architecture

This document describes the package structure, module responsibilities, data flow,
and design decisions.

---

## 1. Package Overview

```
grnbt/
  __init__.py              — Public API exports
  losses.py                — Loss function implementations
  tree.py                  — NewtonTree weak learner
  boosting.py              — Boosting engines (Vanilla + GRN)
  diagnostics.py           — Hilbert-space theory diagnostics
  datasets.py              — Dataset loaders
  utils.py                 — Norms, thresholds, history logging
  extensions/
    __init__.py
    histogram_tree.py      — Approximate histogram-based tree (non-paper)
```

---

## 2. Module Responsibilities

### 2.1 `grnbt.losses`

**Responsibility:** Provide gradient, Hessian, and \(M_0\) for each loss.

**Design:**
- Abstract base class `Loss` enforces a uniform interface.
- Each concrete loss validates inputs (shape, NaN, Inf, label ranges) before computation.
- `empirical_risk_lipschitz(n_samples)` implements Proposition 5.1 scaling automatically.

**Classes:**
- `MSELoss`
- `CharbonnierLoss`
- `BinaryCrossEntropyLoss`
- `CategoricalCrossEntropyLoss`

### 2.2 `grnbt.tree`

**Responsibility:** Build a single decision tree that minimizes the second-order surrogate.

**Design:**
- `_Node` is a private recursive node class.
- `NewtonTree.fit()` performs exhaustive greedy split search over all features and thresholds.
- `NewtonTree.predict()` traverses the tree for each sample.
- Split gain and leaf weight formulas match the paper exactly.
- Numerical stability: a small epsilon prevents division by zero in pathological cases.

**Key Invariants:**
- Every leaf satisfies `min_samples_leaf`.
- Tree depth never exceeds `max_depth`.
- Constant features are skipped (no split possible).

### 2.3 `grnbt.boosting`

**Responsibility:** Orchestrate the boosting loop for both vanilla and GRN variants.

**Design:**
- `BaseBoosting` contains the shared training loop and prediction logic.
- `VanillaNewtonBoosting` overrides only `_compute_lambda` to return a static value.
- `GradientRegularizedNewtonBoosting` computes the adaptive `λ_k` from the current gradient norm.
- History logging is automatic and includes `loss`, `lambda_k`, and `grad_norm`.

**Training Loop:**
```
for k in range(n_estimators):
    g = loss.gradient(y, F_k)
    h = loss.hessian(y, F_k)
    lam_k = compute_lambda(g, h, N)
    tree = NewtonTree(...)
    tree.fit(x, g, h, lam_k)
    f_weak = tree.predict(x)
    F_{k+1} = F_k + eta * f_weak
```

### 2.4 `grnbt.diagnostics`

**Responsibility:** Compute and verify Hilbert-space quantities from the paper.

**Functions:**
- `exact_newton_direction(g, h, lam)` — element-wise exact step.
- `cosine_angle_theta(g, h, f_exact, f_weak)` — `Θ_k` in H-norm.
- `weak_gradient_edge_gamma(g, h, lam, f_weak)` — `γ_k` from gradient contraction.
- `verify_lemma_4_2(g, h, lam, f_weak)` — numerical identity checks.

**Scope:** All diagnostics operate on the empirical `ℝ^N` (or `ℝ^{NK}`) vector
representation, which is the finite-sample instantiation of the paper's Hilbert space.

### 2.5 `grnbt.datasets`

**Responsibility:** Load benchmark datasets used in the paper's experiments.

**Design:**
- Primary source: `scikit-learn`'s `fetch_openml` (optional dependency).
- Fallback: direct UCI CSV download for Wine Quality; synthetic surrogate for Higgs.
- All loaded data is validated for shape, NaN/Inf, and label range.

### 2.6 `grnbt.utils`

**Responsibility:** Small, reusable helpers with no external dependencies.

**Functions:**
- `empirical_norm(v, weights=None)` — RMS or weighted norm.
- `unique_thresholds(x)` — candidate split thresholds.
- `History` — lightweight metric logger.

---

## 3. Data Flow

```
Dataset (X, y)
    │
    ▼
Loss.gradient(y, F_k) ──► g_k  (N-vector)
Loss.hessian(y, F_k)  ──► h_k  (N-vector or block-diagonal)
    │
    ├─► Boosting Engine (Vanilla or GRN)
    │      │
    │      ▼
    │   λ_k = λ_base + sqrt(M * ||g_k||)   [GRN only]
    │      │
    │      ▼
    │   Tree.fit(X, g_k, h_k, λ_k)
    │      │   greedy split using g/h weighted by λ
    │      ▼
    │   f_weak = Tree.predict(X)  (N-vector)
    │      │
    │      ▼
    │   F_{k+1} = F_k + η * f_weak
    │      │
    │      ▼
    ▼
Diagnostics (optional, every T iterations)
    │
    ▼
Exact Newton direction f_exact = -g_k / (h_k + λ_k)
    │
    ├─► Θ_k = angle(f_exact, f_weak) in H-norm
    ├─► γ_k = edge from g_weak = -(h_k + λ_k) * f_weak
    └─► Invariant checks (Lemma 4.2, dominance, etc.)
```

---

## 4. Design Decisions

### 4.1 Pure Python + NumPy

The core algorithm uses only NumPy arrays and Python loops. This maximizes readability
and allows direct comparison with the paper's formulas. Performance-critical inner loops
(e.g., tree traversal) use Python `for` loops; for large-scale production use the
histogram extension or a compiled backend would be appropriate.

### 4.2 Exact vs. Approximate Split Finding

The baseline `NewtonTree` implements **exhaustive greedy search** over all thresholds.
This matches the paper's idealized weak-learner analysis. `HistogramNewtonTree` (in
`extensions/`) provides an approximate alternative for speed, but is not part of the
faithful reproduction.

### 4.3 Scalar-Output Focus

The current tree builder assumes scalar gradients/Hessians (shape `(N,)`). Multi-class
logits require a separate vector-valued tree builder (not yet implemented). The loss
functions (`CategoricalCrossEntropyLoss`) and diagnostics support multi-class vectors,
but the tree itself would need one tree per class or a multi-output extension.

### 4.4 Input Validation Strategy

Every public function validates:
- Type (must be `np.ndarray` where expected)
- Shape (correct rank and compatibility)
- Emptiness (non-zero size)
- NaN / Inf (no non-finite values)
- Range (e.g., non-negative lambda, binary labels)

Validation is strict to catch errors early and prevent silent numerical corruption.

### 4.5 Numerical Stability

- **Charbonnier:** `np.hypot(1.0, d)` avoids overflow for large residuals.
- **Tree denominators:** `1e-12` epsilon prevents division by zero when `sum(h) + λ` is zero.
- **BCE/CCE:** `1e-15` epsilon inside `log()` prevents `log(0)`.
- **Diagnostics:** `np.clip` ensures `cos Θ ∈ [-1, 1]` and `γ ∈ [0, 1]` despite rounding.

---

## 5. Extension Points

The following are cleanly isolated and do not affect baseline behavior:

1. **Histogram tree** (`extensions/histogram_tree.py`) — binning-based split search.
2. **Multi-output trees** — would subclass `NewtonTree` with vector leaf weights.
3. **Column subsampling** — would add a `feature_fraction` parameter to `NewtonTree.fit`.
4. **Row subsampling** — would sample a subset of rows before each tree fit.
5. **Early stopping** — would monitor validation loss in `BaseBoosting.fit`.

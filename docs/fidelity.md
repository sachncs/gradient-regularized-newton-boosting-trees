# Fidelity Report

**Paper:** Gradient Regularized Newton Boosting Trees with Global Convergence  
**Authors:** Nikita Zozoulenko, Daniel Falkowski, Thomas Cass, Lukas Gonon  
**arXiv:** 2605.00581v1

---

## A. Exact Implementations

| Paper Component | Implementation Status | Location |
|-----------------|----------------------|----------|
| **Algorithm 1** — Vanilla Restricted Newton Descent | **Exact** | `grnbt/boosting.py` (`VanillaNewtonBoosting`) |
| **Algorithm 2** — Gradient Regularized Restricted Newton | **Exact** | `grnbt/boosting.py` (`GradientRegularizedNewtonBoosting`) |
| Adaptive regularization `λ_k = λ_base + sqrt(M ||g_k||)` | **Exact** | `boosting.py:_compute_lambda` |
| Proposition 5.1 — `M = M_0 sqrt(N)` scaling | **Exact** | `losses.py:empirical_risk_lipschitz` |
| **Loss gradients & Hessians** (Appendix A) | **Exact** | `grnbt/losses.py` |
| — MSE | Exact | `MSELoss` |
| — Charbonnier | Exact | `CharbonnierLoss` |
| — Binary Cross Entropy | Exact | `BinaryCrossEntropyLoss` |
| — Categorical Cross Entropy | Exact | `CategoricalCrossEntropyLoss` |
| **Newton tree leaf weight** `w = -∑g / (∑h + λ)` | **Exact** | `tree.py:_Node.weight` |
| **Newton tree split gain** `gain = ½[(∑g_L)²/(∑h_L+λ) + ...]` | **Exact** | `tree.py:_build` |
| **Exact Newton direction** in ℝ^N (diagonal H) | **Exact** | `diagnostics.py:exact_newton_direction` |
| **Cosine angle `Θ_k`** in H-induced inner product | **Exact** | `diagnostics.py:cosine_angle_theta` |
| **Weak gradient edge `γ_k`** | **Exact** | `diagnostics.py:weak_gradient_edge_gamma` |
| **Lemma 4.2 identity checks** | **Exact** | `diagnostics.py:verify_lemma_4_2` |
| **Hessian Lipschitz constants `M_0`** | **Exact** (Appendix A) | `losses.py:hessian_lipschitz_constant` |

---

## B. Approximate / Assumed / Flagged Gaps

| Item | Paper Specification | Our Choice | Flagged? |
|------|---------------------|------------|----------|
| `λ_base` for Wine Quality experiment | Not stated | `0.0` for vanilla & GRN | ✅ Yes |
| `λ_base` for Higgs diagnostics | Not stated | `1e-3` | ✅ Yes |
| `min_samples_leaf` | Not stated | `1` (default) | ✅ Yes |
| Higgs subset size | "subset" only | `10,000` for diagnostics, `100,000` for full | ✅ Yes |
| Tree depth for Wine experiment | Depth 4 | `max_depth=4` | ✅ Yes |
| Learning rate η for main experiments | `η = 1.0` | `learning_rate=1.0` | ✅ Yes |
| Number of boosting iterations | Not stated | `100` for Wine, `50` for Higgs | ✅ Yes |
| Feature preprocessing | Not stated | Standardize to zero mean, unit variance | ✅ Yes |

---

## C. Known Deviations from Paper

1. **No early stopping / shrinkage / column subsampling:** The paper focuses on
   the pure algorithmic core. Common GBDT heuristics (stochastic gradient boosting,
   feature subsampling) are not discussed and therefore not implemented in the
   baseline.

2. **Exhaustive greedy split finding:** We implement exact greedy search over all
   feature thresholds. The paper uses this idealized weak learner for analysis.
   Histogram approximation is provided only as an optional extension.

3. **Multi-class Hessian memory:** Categorical cross-entropy stores the full
   `(N, K, K)` block-diagonal Hessian. This is mathematically exact but
   `O(N K²)` in memory. The paper does not discuss memory optimization.

4. **Diagnostics on training set:** `Θ_k` and `γ_k` are computed on the full
   training data. The paper defines them in the empirical Hilbert space
   `L²(ν̂_n)`, which for a finite dataset is exactly ℝ^N with the empirical
   inner product. This is consistent.

5. **Numerical stability safeguards:** We use `np.hypot(1, d)` for the
   Charbonnier loss to avoid overflow, and add `1e-12` to tree denominators
   (`sum(h) + λ`) to prevent division-by-zero when the Hessian is numerically
   zero. These do not change the mathematical formulas; they are standard
   floating-point edge-case handling.

---

## D. Extensions (Isolated from Baseline)

The following are **not part of the paper reproduction** and live under
`grnbt/extensions/`:

- `histogram_tree.py` — faster approximate split finding via histogram binning.
- Optional learning-rate decay schedules.
- Validation-set early stopping.
- Feature importance by gain.

They are deliberately excluded from `grnbt/__init__.py` and the baseline tests.

---

## E. Numerical Verification Status

| Claim | Verification Method | Status |
|-------|---------------------|--------|
| Lemma 4.2 (i) `λ ||f|| <= ||g||` | `tests/test_diagnostics.py` | ✅ Pass |
| Lemma 4.2 (ii) `||f||²_K = -<g,f>` | `tests/test_diagnostics.py` | ✅ Pass |
| `Θ_k ∈ [0, 1]` for exact step | `tests/test_diagnostics.py` | ✅ Pass |
| `γ_k ∈ [0, 1]` for exact step | `tests/test_diagnostics.py` | ✅ Pass |
| MSE loss decreases with small η | `tests/test_boosting.py` | ✅ Pass |
| Charbonnier `M_0 = 1` | `tests/test_losses.py` | ✅ Pass |
| BCE `M_0 = 1/4` | `tests/test_losses.py` | ✅ Pass |
| Tree leaf weight = closed form | `tests/test_tree.py` | ✅ Pass |
| Tree respects `max_depth` | `tests/test_tree.py` | ✅ Pass |
| Tree gain monotonicity with depth | `tests/test_tree.py` | ✅ Pass |
| BCE Hessian > 0 | `tests/test_losses.py` | ✅ Pass |
| CCE gradient sums to zero | `tests/test_losses.py` | ✅ Pass |
| CCE Hessian symmetric | `tests/test_losses.py` | ✅ Pass |
| GRN adaptive λ increases | `tests/test_boosting.py` | ✅ Pass |
| GRN MSE adaptive λ = 0 | `tests/test_boosting.py` | ✅ Pass |
| History logs finite values | `tests/test_utils.py` | ✅ Pass |
| Datasets load with valid shapes | `tests/test_datasets.py` | ✅ Pass |

---

*Report generated: 2026-05-06*

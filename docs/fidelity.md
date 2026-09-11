# Fidelity Report

**Paper:** Gradient Regularized Newton Boosting Trees with Global Convergence  
**Authors:** Nikita Zozoulenko, Daniel Falkowski, Thomas Cass, Lukas Gonon  
**arXiv:** (preprint, id verification pending)

---

## A. Exact Implementations

| Paper Component | Implementation Status | Location |
|-----------------|----------------------|----------|
| **Algorithm 1** — Vanilla Restricted Newton Descent | **Exact** | `grnbt/boosting.py` (`VanillaNewtonBoosting`) |
| **Algorithm 2** — Gradient Regularized Restricted Newton | **Exact** | `grnbt/boosting.py` (`GradientRegularizedNewtonBoosting`) |
| Adaptive regularization `λ_k = λ_base + sqrt(M ‖g_k‖_H)` with empirical-RMS norm `‖g_k‖_H = ‖g_k‖ / √N` | **Exact** | `boosting.py:compute_lambda` |
| Proposition 5.1 — `M = M_0 sqrt(N)` scaling | **Exact** | `losses.py:empirical_risk_lipschitz` |
| **Loss gradients & Hessians** (Appendix A) | **Exact** | `grnbt/losses.py` |
| — MSE | Exact | `MSELoss` |
| — Charbonnier | Exact | `CharbonnierLoss` |
| — Binary Cross Entropy | Exact | `BinaryCrossEntropyLoss` |
| — Categorical Cross Entropy | Exact | `CategoricalCrossEntropyLoss` |
| **Newton tree leaf weight** `w = -∑g / (∑h + λ)` | **Exact** | `tree.py:Node.weight` |
| **Newton tree split gain** `gain = ½[(∑g_L)²/(∑h_L+λ) + ...]` | **Exact** | `tree.py:build` |
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

| Claim | Verification Method | Tolerance | Status |
|-------|---------------------|-----------|--------|
| Lemma 4.2 (i) `λ ||f|| <= ||g||` | `tests/test_diagnostics.py::test_lemma_4_2_identities` | `1e-6` | ✅ Pass |
| Lemma 4.2 (ii) `||f||²_K = -<g,f>` | `tests/test_diagnostics.py::test_lemma_4_2_identities` | `1e-5` | ✅ Pass |
| `Θ_k ∈ [0, 1]` for exact step | `tests/test_diagnostics.py::test_cosine_angle_perfect_alignment` | `1e-12` | ✅ Pass |
| `γ_k ∈ [0, 1]` for exact step | `tests/test_diagnostics.py::test_weak_gradient_edge_perfect` | `1e-12` | ✅ Pass |
| MSE loss decreases | `tests/test_boosting.py::test_loss_decreases_for_strongly_convex` | strict `<` | ✅ Pass |
| Charbonnier `M_0 = 1` | `tests/test_losses.py::test_hessian_lipschitz_constants` | exact `==` | ✅ Pass |
| BCE `M_0 = 1/4` | `tests/test_losses.py::test_hessian_lipschitz_constants` | exact `==` | ✅ Pass |
| Tree leaf weight = closed form | `tests/test_tree.py::test_leaf_weight_closed_form` | `1e-9` | ✅ Pass |
| Tree respects `max_depth` | `tests/test_tree.py::test_respects_max_depth` | exact `==` | ✅ Pass |
| Tree gain monotonicity with depth | `tests/test_tree.py::test_gain_monotonic_with_depth` | `1e-9` | ✅ Pass |
| BCE Hessian > 0 | `tests/test_losses.py::test_bce_hessian_positive` | strict `>` | ✅ Pass |
| CCE gradient sums to zero | `tests/test_losses.py::test_cce_gradient_sums_to_zero` | `1e-12` | ✅ Pass |
| CCE Hessian symmetric | `tests/test_losses.py::test_cce_hessian_symmetric` | `1e-12` | ✅ Pass |
| GRN adaptive λ increases | `tests/test_boosting.py::test_grn_adaptive_lambda_increases` | strict `>` | ✅ Pass |
| GRN MSE adaptive λ = 0 | `tests/test_boosting.py::test_boosting_grn_lam_base_zero_mse` | `1e-9` | ✅ Pass |
| Vanilla diverges on Charbonnier | `tests/test_boosting.py::test_vanilla_diverges_on_charbonnier` | ratio `>10` | ✅ Pass |
| GRN converges on Charbonnier | `tests/test_boosting.py::test_grn_converges_on_charbonnier` | strict `<` | ✅ Pass |
| History logs finite values | `tests/test_utils.py::test_history_non_finite_raises` | exact `==` | ✅ Pass |
| Datasets load with valid shapes | `tests/test_datasets.py::test_wine_quality_shape` | exact `==` | ✅ Pass |

---

*Tolerance column gives the numerical bound the test uses. Status is
re-validated by `pytest tests/` on every CI run; this table is regenerated
in step with the test suite, not the 0.1.0 release date.*

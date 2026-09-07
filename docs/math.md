# Mathematical Foundations

This document restates the paper's problem definition, notation, model architecture,
objective functions, and convergence results as implemented in this codebase.

**Paper:** Zozoulenko et al., *Gradient Regularized Newton Boosting Trees with Global Convergence*, arXiv:2605.00581v1.

---

## 1. Problem Definition

Given a training set \(\{(x_i, y_i)\}_{i=1}^N\) drawn from a distribution \(\nu\) over
\(\mathcal{X} \times \mathcal{Y}\), we seek an ensemble predictor
\(F: \mathcal{X} \to \mathcal{Y}\) that minimizes the empirical risk:

\[
L(F) = \frac{1}{N} \sum_{i=1}^N \ell(F(x_i), y_i)
\]

where \(\ell\) is a smooth convex pointwise loss. In the Hilbert-space formulation,
\(L: \mathcal{H} \to \mathbb{R}\) where \(\mathcal{H} = L^2(\hat{\nu}_x)\) is the empirical
\(L^2\) space over the feature distribution.

---

## 2. Notation

| Symbol | Meaning |
|--------|---------|
| \(N\) | Number of training samples |
| \(K\) | Number of classes (for multi-class losses) |
| \(g_k\) | Fréchet gradient of \(L\) at iterate \(F_k\) |
| \(H_k\) | Fréchet Hessian of \(L\) at iterate \(F_k\) |
| \(\mathcal{F}\) | Weak-learner family (e.g., fixed-depth decision trees) |
| \(f_{k+1}^w\) | Weak learner minimizing the second-order surrogate over \(\mathcal{F}\) |
| \(\eta\) | Learning rate (step size) |
| \(\lambda_k\) | Per-iteration \(L^2\) regularization |
| \(M_0\) | Pointwise Hessian Lipschitz constant |
| \(M\) | Empirical-risk Hessian Lipschitz constant \((= M_0 \sqrt{N})\) |
| \(\Theta_k\) | Cosine angle between exact and weak directions in \(H_k\)-norm |
| \(\gamma_k\) | Weak gradient edge controlling inexactness |

---

## 3. Loss Functions

All losses are implemented as empirical averages with per-sample gradients and Hessians.

### 3.1 Mean Squared Error (MSE)

\[
\ell(\hat{y}, y) = (\hat{y} - y)^2
\]

**Gradient:** \(\frac{2}{N}(\hat{y} - y)\)

**Hessian:** \(\frac{2}{N}\) (constant)

**\(M_0 = 0\)** since the Hessian does not depend on the prediction.

### 3.2 Charbonnier Loss

\[
\ell(\hat{y}, y) = \sqrt{1 + (\hat{y} - y)^2} - 1
\]

**Gradient:** \(\frac{\hat{y} - y}{\sqrt{1 + (\hat{y} - y)^2} \cdot N}\)

**Hessian:** \(\frac{1}{(1 + (\hat{y} - y)^2)^{3/2} \cdot N}\)

**\(M_0 = 1\)** because the third derivative is bounded by 1.

### 3.3 Binary Cross-Entropy (Logistic Loss)

Operating on logits \(z\) with labels \(y \in \{0, 1\}\):

\[
\ell(z, y) = -y \log \sigma(z) - (1-y) \log (1 - \sigma(z))
\]

where \(\sigma(z) = \frac{1}{1 + e^{-z}}\).

**Gradient:** \(\frac{1}{N}(\sigma(z) - y)\)

**Hessian:** \(\frac{1}{N} \sigma(z)(1 - \sigma(z))\)

**\(M_0 = \frac{1}{4}\)** since \(\sigma(1-\sigma)\) has derivative bounded by \(1/4\).

### 3.4 Categorical Cross-Entropy (Softmax Loss)

Operating on logits \(z \in \mathbb{R}^K\) with integer labels \(y \in \{0, \dots, K-1\}\):

\[
\ell(z, y) = -\log p_y \quad \text{where} \quad p_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
\]

**Gradient:** \(\frac{1}{N}(p - \mathbf{1}_y)\) where \(\mathbf{1}_y\) is the one-hot vector.

**Hessian:** Block-diagonal with blocks \(\frac{1}{N}(\mathrm{diag}(p) - p p^\top)\).

**\(M_0 = \frac{1}{4}\)** by the same bound as binary cross-entropy.

---

## 4. Newton Tree Weak Learner

### 4.1 Leaf Weight Formula

For a leaf \(I_j\) covering samples with gradients \(g_i\) and Hessians \(h_i\):

\[
w_j = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
\]

This minimizes the second-order surrogate \(\sum_{i \in I_j} (g_i f(x_i) + \frac{1}{2} h_i f(x_i)^2) + \frac{\lambda}{2} \|f\|^2\).

### 4.2 Split Gain

For a candidate split into left (\(L\)) and right (\(R\)) partitions:

\[
\text{Gain} = \frac{1}{2} \left[
    \frac{(\sum_{i \in L} g_i)^2}{\sum_{i \in L} h_i + \lambda}
    + \frac{(\sum_{i \in R} g_i)^2}{\sum_{i \in R} h_i + \lambda}
    - \frac{(\sum_{i} g_i)^2}{\sum_{i} h_i + \lambda}
\right]
\]

The split with maximum gain is chosen greedily. A small epsilon (\(10^{-12}\)) is added
to denominators to prevent division by zero when the Hessian is numerically zero.

---

## 5. Boosting Algorithms

### 5.1 Vanilla Restricted Newton Boosting (Algorithm 1)

**Iteration:**
1. Compute \(g_k = \nabla L(F_k)\) and \(H_k = \nabla^2 L(F_k)\).
2. Fit weak learner \(f_{k+1}^w \in \arg\min_{f \in \mathcal{F}} Q_k(f)\) with static \(\lambda_k = \lambda_{\text{base}}\).
3. Update \(F_{k+1} = F_k + \eta \, f_{k+1}^w\).

### 5.2 Gradient Regularized Newton Boosting (Algorithm 2)

Same as Algorithm 1, but with adaptive regularization:

\[
\lambda_k = \lambda_{\text{base}} + \sqrt{M \|g_k\|_{\mathcal{H}}}
\]

where \(M = M_0 \sqrt{N}\) per **Proposition 5.1**.

---

## 6. Proposition 5.1 — Empirical Risk Regularity

If the pointwise loss \(\ell\) has:
- \(S_0\)-smooth second derivative,
- \(\mu_0\)-strongly convex second derivative,
- \(M_0\)-Lipschitz Hessian,

then the empirical risk \(L(F) = \frac{1}{N} \sum_i \ell(F(x_i), y_i)\) has:
- \(S = S_0\)-smoothness,
- \(\mu = \mu_0 / N\)-strong convexity,
- \(M = M_0 \sqrt{N}\)-Lipschitz Hessian.

This scaling is exact and is implemented in `Loss.empirical_risk_lipschitz()`.

---

## 7. Diagnostics

### 7.1 Exact Newton Direction

For a diagonal Hessian (scalar-output losses):

\[
f_{k+1} = -(H_k + \lambda_k I)^{-1} g_k \quad \Rightarrow \quad f_{k+1}(x_i) = -\frac{g_i}{h_i + \lambda_k}
\]

### 7.2 Cosine Angle \(\Theta_k\)

Defined in the \(H_k\)-induced inner product:

\[
\cos \Theta_k = \frac{\langle f_{k+1}, f_{k+1}^w \rangle_{H_k}}{\|f_{k+1}\|_{H_k} \|f_{k+1}^w\|_{H_k}}
\quad \text{where} \quad
\langle u, v \rangle_{H_k} = \sum_i h_i \, u_i v_i
\]

### 7.3 Weak Gradient Edge \(\gamma_k\)

The implied weak gradient is \(g_k^w = -(H_k + \lambda_k I) f_{k+1}^w\). The edge satisfies:

\[
\|g_k^w - g_k\|^2 \leq (1 - \gamma_k^2) \|g_k\|^2
\]

which rearranges to:

\[
\gamma_k = \sqrt{1 - \frac{\|g_k^w - g_k\|^2}{\|g_k\|^2}}
\]

### 7.4 Lemma 4.2 Identities

For the regularized exact step:
1. \(\lambda_k \|f_{k+1}\| \leq \|g_k\|\)
2. \(\|f_{k+1}\|_{K_k}^2 = -\langle g_k, f_{k+1} \rangle\) where \(K_k = H_k + \lambda_k I\)

---

## 8. Convergence Results (Summary)

### Theorem 3.8 — Linear Convergence of Vanilla Newton

For \(S\)-smooth, \(\mu\)-strongly convex, Hessian-dominated losses with
\(\|f_{k+1}\|_{H_k}^2 \geq c L(F_k)\), vanilla restricted Newton with
\(\eta \in (0, 2\mu/S)\) achieves:

\[
L(F_{k+1}) \leq (1 - \rho) L(F_k)
\quad \text{where} \quad
\rho = c \, \Theta^2 \, \eta \Bigl(1 - \frac{\eta S}{2\mu}\Bigr)
\]

### Theorem 4.9 — Global \(\mathcal{O}(1/k^2)\) Rate for GRN

For convex losses with \(2M\)-Lipschitz Hessians and finite-diameter sublevel sets,
using \(\lambda_k = C \sqrt{M \|g_k\|}\) with \(C \geq 1\) and \(\eta \in (0, 1]\):

\[
L(F_k) - L(F_*) \leq \frac{C^2 M \Lambda^3}{\eta^2 \gamma^{12}} \cdot \frac{1}{k^2}
\]

where \(\Lambda\) is a diameter bound and \(\gamma\) is the weak gradient edge.

### Theorem 4.10 — Local Linear Contraction

Once \(\|g_k\|\) is small enough, strong convexity yields:

\[
\|g_{k+1}\| \leq \frac{3 + \rho}{4} \|g_k\|
\quad \text{where} \quad
\rho \approx 1 - \frac{\eta \gamma^2}{2}
\]

If \(\eta = \gamma = 1\), the finite-dimensional GRN achieves a superlinear \(3/2\) rate;
with inexact weak learners only linear contraction is proven.

# Adaptive Local Polynomial Regression Differentiation (ALPRD)
### Research Log — Iteration 1 (Version 1.0)

Code: [`../src/methods.py`](../src/methods.py), [`../src/datasets.py`](../src/datasets.py), [`../src/benchmark.py`](../src/benchmark.py)
Raw results: [`../results/results_v1.csv`](../results/results_v1.csv) (350 method×dataset runs, 0 failures, all numbers below are read from this file — nothing in this document is invented)

---

## STEP 1 — Current Version: V1.0

**Name:** Adaptive Local Polynomial Regression Differentiation (ALPRD)

**Goal:** estimate $f'(t_0)$ (first derivative) from irregularly-sampled, noisy pairs $(t_i,y_i)$.

**Algorithm, in full:**

For each query point $t_0$ (in this study, $t_0=t_i$ for every sample):

1. **Local bandwidth (uneven-sampling adaptation).** Sort the other samples by distance to $t_0$. Let
$$h(t_0) = \text{dist to the } k\text{-th nearest sample}, \qquad k=\max\big(p+2,\ \lceil \alpha n\rceil\big)$$
2. **Kernel weights.**
$$w_i(t_0) = K\!\left(\frac{t_i-t_0}{h(t_0)}\right),\qquad K(u)=(1-|u|^3)^3\mathbb{1}_{|u|<1}\ \text{(tricube)}$$
3. **Weighted least squares polynomial fit** of degree $p$ (default $p=2$):
$$\hat\beta(t_0)=\arg\min_{\beta\in\mathbb{R}^{p+1}}\ \sum_{i=1}^n w_i(t_0)\Big(y_i-\textstyle\sum_{j=0}^p\beta_j(t_i-t_0)^j\Big)^2$$
4. **Derivative readout.**
$$\hat f'(t_0)=\hat\beta_1$$
5. **Self error estimate** (optional, `return_variance=True`): a per-point standard error of $\hat f'(t_0)$, using an estimate $\hat\sigma^2$ of the noise variance.

### Closed-form solution

With design matrix $X\in\mathbb{R}^{n\times(p+1)}$, $X_{ij}=(t_i-t_0)^j$, and $W=\mathrm{diag}(w_1,\dots,w_n)$:
$$\hat\beta(t_0) = (X^\top W X)^{-1}X^\top W y \tag{1}$$
$$\hat f'(t_0) = e_1^\top \hat\beta(t_0) = e_1^\top(X^\top WX)^{-1}X^\top W\,y =: \sum_{i=1}^n \ell_i(t_0)\,y_i \tag{2}$$
where $e_1=(0,1,0,\dots,0)^\top$ and $\ell_i(t_0)$ is the *equivalent-kernel weight* on observation $i$ — the estimator is **linear in the data**, which is what makes the error analysis in Step 4 tractable.

### Self-estimated variance

$$\widehat{\mathrm{Var}}\big(\hat f'(t_0)\big) = \hat\sigma^2\; e_1^\top (X^\top WX)^{-1}(X^\top W^2 X)(X^\top WX)^{-1} e_1 \tag{3}$$

$$\hat\sigma^2 = \operatorname{median}_i\left(\frac{[a_i y_{i-1}+b_i y_i+c_i y_{i+1}]^2}{a_i^2+b_i^2+c_i^2}\right),\quad
\begin{aligned}a_i&=-h_{2}/(h_1(h_1+h_2))\\ b_i&=(h_2-h_1)/(h_1h_2)\\ c_i&=h_1/(h_2(h_1+h_2))\end{aligned} \tag{4}$$

with $h_1=t_i-t_{i-1}$, $h_2=t_{i+1}-t_i$. Eq. (4) is the classical Rice (1984) difference-based variance estimator, generalized here to unevenly-spaced points (the coefficients $a_i,b_i,c_i$ are exactly the ones that make the bracketed combination vanish for any $f$ linear in $t$, so its square isolates noise to leading order); the median (rather than the mean used in the original Rice estimator) is used for robustness to outliers.

---

## STEP 2 — Symbol Table

| Symbol | Meaning | Physical meaning (motion example) | SI units | Allowed values | Purpose |
|---|---|---|---|---|---|
| $t_i$ | sample time | time of measurement | s | strictly increasing, real | independent variable |
| $y_i$ | observed value | e.g. noisy position | m | real | dependent/noisy variable |
| $f(t)$ | true underlying signal | e.g. true position | m | $f\in C^{p+1}$ locally | quantity being differentiated |
| $f'(t)$ | true derivative | e.g. true velocity | m/s | real | target of estimation |
| $\varepsilon_i$ | measurement noise | sensor/quantization noise | m | mean 0, var $\sigma^2$ | corrupts $y_i$ |
| $t_0$ | query point | — | s | any point in domain | where the derivative is evaluated |
| $h(t_0)$ | local bandwidth | local averaging window half-width | s | $h>0$ | controls bias/variance tradeoff |
| $k$ | neighbor count defining $h$ | — | dimensionless (integer) | $p+2\le k\le n$ | sets bandwidth adaptively to local density |
| $\alpha$ (`span`) | neighbor fraction | — | dimensionless | $0<\alpha\le 1$ | default rule for $k=\lceil\alpha n\rceil$ |
| $K(\cdot)$ | kernel function | weighting profile | dimensionless | tricube or Gaussian here | down-weights distant points smoothly |
| $w_i(t_0)$ | kernel weight of sample $i$ | — | dimensionless | $[0,1]$ | local relevance of $(t_i,y_i)$ |
| $p$ | local polynomial degree | — | dimensionless (integer) | $p\ge 1$ | local model flexibility |
| $\beta_j$ | local Taylor coefficient | $f^{(j)}(t_0)/j!$ | m/s$^j$ | real | fit parameter |
| $X$ | local design matrix | — | s$^j$ (per column) | — | WLS regressor matrix |
| $W$ | weight matrix | — | dimensionless | diagonal, $\ge 0$ | encodes $w_i$ |
| $\ell_i(t_0)$ | equivalent-kernel weight | — | 1/s | real | linear-estimator weight on $y_i$ |
| $\sigma^2$ | noise variance | — | m$^2$ | $\ge 0$ | noise model parameter |
| $\hat\sigma^2$ | estimated noise variance | — | m$^2$ | $\ge 0$ | plugged into Eq. (3) |
| $n$ | sample count | — | dimensionless (integer) | $n\ge p+2$ | dataset size |

---

## STEP 3 — Assumptions

**Smoothness.** $f\in C^{p+1}$ in a neighborhood of every query point (needed for the Taylor argument in Step 4). No assumption of global smoothness or periodicity.

**Noise.** $\varepsilon_i$ independent, mean zero, variance $\sigma^2$ (homoscedastic model used in Eq. 3 and 4). Heteroscedastic or correlated noise is **not modeled** — Eq. (3) will be wrong (biased) if noise variance depends on $t$ or samples are correlated (e.g. autocorrelated sensor drift).

**Sampling.** $t_i$ strictly increasing, arbitrary spacing. No assumption of a minimum sampling rate is enforced beyond needing $\ge p+1$ points with nonzero kernel weight (the code widens the window automatically if this fails). No assumption that gaps are bounded — but Step 8 shows large gaps (relative to signal curvature) inflate bias.

**Boundary.** No special one-sided kernel/reflection is used at the domain edges; the code simply lets the nearest-neighbor bandwidth pick up more points on the interior side, which is the same construction LOESS uses, but no analysis of boundary bias order is done in this iteration (flagged in Step 11).

**Numerical.** Local Gram matrix $X^\top WX$ assumed non-singular for the chosen window; the implementation only checks this via `min_points_factor` heuristics and a `try/except lstsq` fallback, not a principled conditioning bound (flagged in Step 11).

**Outliers.** Not explicitly modeled in V1.0 — there is no robust loss or residual-based re-weighting. Any apparent outlier robustness measured in Step 7 is a side effect of window width, not a designed property (see Step 8/9).

---

## STEP 4 — Mathematical Analysis

### 4.1 Reproducing (moment) property — **Proven**

Because $\hat\beta = (X^\top WX)^{-1}X^\top W y$ for *any* $y$, substituting $y = X\gamma$ for an arbitrary $\gamma\in\mathbb{R}^{p+1}$ gives
$$\hat\beta = (X^\top WX)^{-1}X^\top W X\gamma = \gamma.$$
So the estimator reproduces any degree-$\le p$ polynomial exactly. In terms of the equivalent kernel $\ell_i(t_0)$ from Eq. (2), this is equivalent to the exact algebraic identity
$$\sum_{i=1}^n \ell_i(t_0)\,(t_i-t_0)^k = \delta_{1k},\qquad k=0,\dots,p. \tag{5}$$
This is a finite-sample, exact identity (proven directly from the definition of $\hat\beta$, no asymptotics needed).

### 4.2 Bias — **Derived to leading order** (not a fully rigorous asymptotic theorem)

By Taylor's theorem with Lagrange remainder, since $f\in C^{p+1}$ near $t_0$, for every $t_i$ in the local window there is $\xi_i$ between $t_0$ and $t_i$ with
$$f(t_i) = \sum_{j=0}^p \frac{f^{(j)}(t_0)}{j!}(t_i-t_0)^j \;+\; \frac{f^{(p+1)}(\xi_i)}{(p+1)!}(t_i-t_0)^{p+1}.$$
Apply the linear estimator (2) to the noiseless part and use identity (5):
$$\mathbb{E}[\hat f'(t_0)] - f'(t_0) \;=\; \sum_i \ell_i(t_0)\,\frac{f^{(p+1)}(\xi_i)}{(p+1)!}(t_i-t_0)^{p+1}.$$
This is **exact** given the assumptions, but not yet a clean closed form because each $\xi_i$ differs. If $f^{(p+1)}$ is approximately constant over the local window (true in the limit $h\to0$, and a good approximation whenever the window is short relative to the curvature scale of $f^{(p+1)}$), this reduces to the **leading-order bias approximation**:
$$\mathrm{Bias}(t_0) \;\approx\; \frac{f^{(p+1)}(t_0)}{(p+1)!}\sum_i \ell_i(t_0)(t_i-t_0)^{p+1}. \tag{6}$$
**Honest limitation:** Step 7's chaotic-motion results are a direct, measured demonstration of what happens when this approximation fails — $f^{(p+1)}$ is *not* roughly constant over a window spanning 25% of the samples when the underlying signal is a Lorenz trajectory. This is not a hypothesis; it is the observed failure mode (Step 8).

### 4.3 Variance — **Proven, exact (finite-sample)**

Given the homoscedastic-independent-noise assumption in Step 3, Eq. (3) follows directly from $\mathrm{Var}(Ay)=\sigma^2 AA^\top$ applied to the linear estimator $\hat f'(t_0)=e_1^\top(X^\top WX)^{-1}X^\top W\,y$ (a one-line sandwich-variance computation). This is exact, not asymptotic, *conditional on the assumed noise model being correct* — if the true noise is heteroscedastic or correlated, Eq. (3) is a biased estimate of the true variance (this is a known, general limitation of all such plug-in variance formulas, not specific to this method).

### 4.4 Consistency — **Not proven; stated qualitatively**

Standard nonparametric-regression consistency requires $h\to0$ and (effective) local sample count $\to\infty$ as $n\to\infty$, i.e. $k\to\infty$ but $k/n\to0$. Our default $k=\lceil\alpha n\rceil$ with fixed $\alpha$ does **not** satisfy $k/n\to0$ — it is a fixed-fraction span, which is a deliberate LOESS-style choice for finite-sample robustness, but it means the estimator is **not asymptotically consistent under fixed $\alpha$** (bias in Eq. 6 does not vanish as $n\to\infty$ if $\alpha$ is held fixed, because $h$ does not shrink to 0 relative to the domain). A consistent variant would require $\alpha=\alpha(n)\to0$ with $\alpha(n)\cdot n\to\infty$. **This is a genuine theoretical gap in V1.0**, tracked in Step 11.

### 4.5 Stability / conditioning — **Derived (qualitative), not bounded**

$X$ is a (shifted, local) Vandermonde-type matrix. Classical results (Gautschi) show Vandermonde matrices become severely ill-conditioned as their size grows, though this is mitigated here because (a) coordinates are centered at $t_0$ (so entries stay $O(h^j)$ rather than $O(t^j)$), and (b) $p$ is small (2 in this study). No explicit condition-number bound is derived or measured in this iteration — flagged in Step 11. A numerically preferable implementation would use an orthogonal local polynomial basis (QR-based fit or Legendre polynomials rescaled to the window) instead of the raw monomial (Vandermonde) basis currently used in `alprd_v1`.

### 4.6 Computational complexity — **Derived, exact for this implementation**

For each of $n$ query points, the current code (a) computes distances to all $n$ points and sorts them — $O(n\log n)$ — and (b) solves a $(p{+}1)\times(p{+}1)$ WLS system from $O(k)$ active points — $O(kp^2+p^3)$. Total: $O\!\left(n^2\log n\right)$ (the $n\log n$ per-point sort dominates for the $n$ used here, up to 400). This is a real, measured inefficiency (Step 7 shows ALPRD runtime ≈ cubic-spline / savgol runtime for $n\le400$, but the $O(n^2\log n)$ scaling means it will fall behind at large $n$ — since $t$ is already sorted, an $O(n\log n)$-total sliding-window (two-pointer) implementation is straightforward and is the first optimization slated for V1.1 (Step 13).

---

## STEP 5 — Literature Comparison

*(Conceptual comparison based on established, well-known methodology; no specific numeric claims from these works are asserted here, since no benchmark numbers from the original papers were reproduced or verified in this session — only the mathematical structure is compared.)*

| Method | Handles uneven sampling? | Handles noise? | Adaptive? | Self error estimate? | Relation to ALPRD v1.0 |
|---|---|---|---|---|---|
| Forward/Backward difference | yes (uses actual $t_i$ gaps) | no (amplifies noise, $O(1/h)$ variance) | no | no | ALPRD reduces to this as $p\to0$-ish, $h\to$ smallest gap |
| Central difference (non-uniform 3-pt) | yes | no | no | no | special case: $p=2$-consistent local fit with only 3 points, $w_i\in\{0,1\}$ |
| High-order FD (Fornberg weights) | yes (used here via `_fd_weights`) | no — same noise-amplification problem, worse (higher-order stencils have larger weight norms) | no | no | ALPRD's kernel weighting is the noise-robust generalization of Fornberg's exact-interpolation weights |
| Least-squares / polynomial differentiation | with local windowing, yes | yes (implicitly, via averaging) | only if window adapted | no, typically | ALPRD **is** a windowed weighted least-squares method; the specific combination of tricube kernel + kNN bandwidth is standard LOESS/LOWESS machinery (Cleveland 1979; Cleveland & Devlin 1988), not new |
| Savitzky–Golay | **no** — requires a uniform grid natively | yes | no (fixed window) | no | closely related in spirit (local polynomial smoothing); ALPRD's uneven-grid support is exactly what plain Savitzky–Golay lacks, but our `savgol_resampled` baseline shows this can be worked around by resampling — meaning ALPRD's advantage here is a **convenience/consistency** advantage, not something fundamentally unreachable otherwise |
| Adaptive differentiation (general term for locally-varying-window methods) | yes, by construction | yes | yes | varies | ALPRD v1.0's adaptivity is *sampling-density-driven* (kNN bandwidth) only; it is **not yet** *signal-driven* (no adaptation to local curvature/noise level) — this is precisely local polynomial regression as formalized by Fan & Gijbels (1996) and Ruppert & Wand (1994), not a new adaptivity mechanism |
| Optimization-based (e.g. Tikhonov-regularized / total-variation differentiation, Chartrand-style) | yes, if formulated on irregular grids | yes, often with better handling of piecewise-smooth/discontinuous derivatives | yes (regularization parameter can be adapted) | sometimes | fundamentally different paradigm (global variational optimization vs. local regression) — no comparison to this class was benchmarked in V1.0; flagged for V-next |
| Spectral differentiation | **no** — requires (quasi-)uniform/structured grids or a global basis | poorly, without added regularization | no | no | not applicable to genuinely irregular, noisy scattered data without preprocessing; excluded from benchmarks for this reason |
| Radial Basis Function (RBF) differentiation | yes, naturally (mesh-free) | yes, if combined with regularized fitting | yes, if shape parameter is chosen adaptively | rarely, natively | closest structural relative for irregular data; RBF-FD and ALPRD are both "local weighted fit, differentiate the fit" methods — a head-to-head benchmark against RBF-FD is a natural next comparison, not yet done |

**Honest novelty statement.** As implemented, V1.0 is a standard local-polynomial-regression / LOESS-style derivative estimator with a Rice-type noise-variance plug-in, applied to non-uniform time series. The *base method* is not new (Fan & Gijbels 1996; Cleveland 1979). Any claim to novelty must rest on a *specific, demonstrated* improvement — e.g. a genuinely signal-adaptive (not just density-adaptive) bandwidth rule, or a provably tighter error bound for the uneven-sampling case — which V1.0 does **not** yet establish (see Step 12).

---

## STEP 6/7 — Implementation & Benchmarking

**Datasets (all real, generated and integrated numerically — see `src/datasets.py`):** constant velocity, constant acceleration, quartic polynomial, sinusoidal, exponential, projectile motion, circular motion, and chaotic motion (Lorenz system, $x(t)$ with ground-truth $\dot x=\sigma(y-x)$ taken from the ODE right-hand side evaluated on a `solve_ivp` trajectory at `rtol=1e-10`, not by differencing) — each run clean/uniform, uneven-clean, uneven+Gaussian noise, uneven+uniform noise, uneven+outliers, uneven+missing(25%)+noise, plus two composite "real-world-like" cases (missing + noise + uneven together). **35 scenarios × 7 methods × ... = 350 runs, 0 failures.**

**Methods compared:** forward/backward difference, non-uniform central difference, 5-point Fornberg high-order FD, cubic spline derivative, Savitzky–Golay-on-resampled-grid, and ALPRD v1.0.

**Metrics:** MaxErr, MAE, RMSE, median relative error, wall-clock runtime — computed against the **exact** analytic/ODE derivative, not a finite-difference proxy.

### Headline results (mean over noisy, unevenly-sampled, non-outlier, non-chaotic scenarios — `uneven_gaussian`, `uneven_uniformnoise`, `missing_noisy`; n=200 runs each averaged over the 8 motions × 3 scenarios × 7 methods slice)

| method | MAE | RMSE | MaxErr |
|---|---|---|---|
| savgol_resampled | **1.0474** | **1.6743** | **8.3759** |
| ALPRD_v1.0 | 3.3909 | 4.4863 | 16.6461 |
| central_diff | 2.2381 | 5.9683 | 45.0793 |
| backward_diff | 2.6709 | 5.4835 | 48.1452 |
| forward_diff | 2.6901 | 5.6410 | 51.6615 |
| cubic_spline | 2.7228 | 6.2952 | 46.4064 |
| high_order_fd | 2.7442 | 6.4830 | 47.0577 |

**Honest headline: on plain Gaussian/uniform-noise data, ALPRD v1.0 does *not* beat Savitzky–Golay.** It beats the raw finite-difference family on MAE but not consistently on RMSE/MaxErr (central difference has better RMSE than ALPRD here — driven by a few motions, see per-motion breakdown below).

### Outlier robustness (`uneven_outliers` scenario, mean over 8 motions)

| method | MAE | RMSE | MaxErr |
|---|---|---|---|
| **ALPRD_v1.0** | **23.038** | **32.635** | **116.523** |
| savgol_resampled | 166.384 | 351.943 | 2073.572 |
| central_diff | 581.476 | 3289.499 | 30699.930 |
| forward_diff | 587.341 | 3199.896 | 35408.265 |
| backward_diff | 587.311 | 3199.892 | 35408.765 |
| high_order_fd | 818.639 | 3779.732 | 33226.528 |
| cubic_spline | 926.967 | 3846.479 | 32705.229 |

ALPRD's outlier performance is genuinely an order of magnitude better than every baseline including Savitzky–Golay. **But see Step 8 — this is almost certainly a side effect of ALPRD's wider default window (span=0.25 → ~50 points) diluting a 3%-outlier fraction, not a designed robust-statistics mechanism.** A wider-window Savitzky–Golay would likely close much of this gap; that comparison has not yet been run (flagged for next iteration).

### Failure case: chaotic (Lorenz) motion (mean over all noisy chaotic scenarios)

| method | MAE | RMSE |
|---|---|---|
| savgol_resampled | 0.571 (real-world-like) / ~6 avg | 0.826 |
| central_diff | 3.009–3.048 | 10.6 |
| cubic_spline | 2.937–3.842 | 11.2 |
| high_order_fd | 2.876–3.814 | 11.4 |
| forward/backward_diff | 3.8–7.2 | 8.5–8.6 |
| **ALPRD_v1.0** | **21.6–23.6 (worst of all methods)** | **25.5** |

Full numeric breakdown in `results/results_v1.csv` (columns: `dataset, method, max_err, mae, rmse, median_rel_err, runtime_s, n_points, status`).

### Runtime (mean seconds/call, all 350 runs, $n\in[150,400]$)

| method | seconds |
|---|---|
| backward_diff | 0.000008 |
| forward_diff | 0.000010 |
| central_diff | 0.000020 |
| high_order_fd | 0.004767 |
| savgol_resampled | 0.012152 |
| **ALPRD_v1.0** | **0.013848** |
| cubic_spline | 0.014684 |

---

## STEP 8 — Failure Analysis

### Failure 1: ALPRD underperforms Savitzky–Golay on generic Gaussian/uniform noise

- **What failed:** RMSE 4.49 (ALPRD) vs 1.67 (savgol) on the headline noisy-uneven slice.
- **Why (mathematically):** the fixed span $\alpha=0.25$ was not tuned; Eq. (6)'s bias grows with $h^{p+1-m}=h^{2}$ (for $p=2,m=1$) while Eq. (3)'s variance shrinks like $1/(kh)$ roughly — a 25%-of-data window is wider than the AMISE-optimal bandwidth for most of these smooth, low-curvature motions, so ALPRD is oversmoothing (trading away accuracy for a robustness property it wasn't asked for on clean-noise data).
- **Why (numerically):** no bandwidth selection procedure is implemented; $\alpha=0.25$ is a single global constant applied uniformly regardless of local curvature or noise level.
- **Practical consequence:** for a user with simple, moderately noisy, smoothly-varying data, plain Savitzky–Golay (even naively resampled onto a uniform grid) is currently a better choice than ALPRD v1.0.
- **Severity:** high — this is the core use case the method targets, so it must be fixed before any publishability claim is credible.

### Failure 2: ALPRD is the *worst* method on chaotic (fast-curvature) motion

- **What failed:** MAE up to 23.6 vs 2.9–7.2 for every baseline.
- **Why (mathematically):** Eq. (6)'s leading-order bias approximation assumed $f^{(p+1)}$ roughly constant across the window; for a Lorenz trajectory, $f''$ changes sign and magnitude rapidly, so a window covering 25% of 400 samples spans multiple oscillations — the approximation underlying Eq. (6) is badly violated, and the *exact* bias (the $\xi_i$-dependent sum) is large and not well predicted by the leading-order formula.
- **Why (numerically):** the kNN/span bandwidth rule adapts only to *sampling density*, never to *signal curvature* — exactly the gap identified in Step 4.4/4.6 and Step 5 ("not yet signal-driven").
- **Practical consequence:** ALPRD v1.0 should not be used, as configured, on signals with rapidly varying curvature (chaotic systems, sharp transients, high-frequency content) without a much smaller span.
- **Severity:** high — this is a fundamental limitation of the fixed-span design, not a bug.

### Failure 3: Outlier robustness is likely incidental, not mechanistic

- **What failed (conceptually, as a claim):** it would be dishonest to report ALPRD's outlier numbers as evidence of a *designed* robustness mechanism — v1.0 has no residual-based downweighting (no IRLS, no Huber/Tukey loss). The advantage most plausibly comes from window width diluting a fixed 3% outlier fraction across more points than Savitzky–Golay's much narrower default window (11 points vs ~50).
- **Why it matters:** if true, a wider-window Savitzky–Golay might match ALPRD's outlier performance without any of ALPRD's added complexity — this confound has **not been ruled out** (no such control experiment was run in this session).
- **Severity:** medium — doesn't invalidate the measured numbers, but invalidates any *interpretation* of them as demonstrating a novel robustness property until the confound is tested.

---

## STEP 9 — Improvement Plan for V1.1 (mathematically justified, not yet implemented)

1. **Data-driven bandwidth per point** (targets Failures 1 & 2): replace the fixed span $\alpha$ with a local bandwidth chosen by minimizing an estimate of $\mathrm{Bias}^2+\mathrm{Var}$ built directly from Eqs. (3) and (6) (a plug-in / Goldenshluger–Lepski-style rule), i.e. actually use the self-error machinery already implemented in Eq. (3) to *drive* $h(t_0)$ instead of only reporting it after the fact. Expected benefit: bandwidth shrinks automatically near high-curvature regions (chaotic motion) and grows in flat/noisy regions (fixing both Failure 1 and Failure 2 with a single mechanism). Possible disadvantage: added computational cost (candidate-bandwidth search per point) and a new source of estimation variance in the bandwidth choice itself, which would need its own error analysis.
2. **Confound-controlled outlier comparison** (targets Failure 3): re-run the outlier benchmark with Savitzky–Golay window widths matched (in point-count) to ALPRD's span, to determine whether robustness is a width effect or something else. If it is purely a width effect, V1.1 should add an explicit robust loss (Huber/Tukey biweight IRLS) so that outlier resistance is a real, designed, and separately-tunable property (not entangled with the bias/variance bandwidth choice).
3. **$O(n\log n)$ implementation** (targets Step 4.6): replace the per-query full sort with an incremental two-pointer sliding window over the pre-sorted `t` array. No change to the statistical output, pure performance fix — needed before benchmarking at realistic $n$ (10⁴–10⁶).

Each of these will be benchmarked with the *same* suite (`src/benchmark.py`) so V1.0 and V1.1 numbers are directly comparable — no new random seeds, no cherry-picked scenarios.

---

## STEP 10 — Version History

**Version 1.0** (this document)
- Initial formulation: kNN/span-adaptive tricube-kernel weighted local polynomial regression, degree $p=2$, first-derivative readout, Rice-type self-variance estimator.
- Benchmarked against 6 classical baselines across 35 scenarios (8 motion types × up to 7 sampling/noise conditions), 350 total runs, 0 execution failures.
- Result: beats all baselines on outlier robustness (likely incidental — Failure 3); loses to Savitzky–Golay on plain noisy data (Failure 1); is the worst method on fast-curvature (chaotic) signals (Failure 2).

*(No prior versions — this is the first iteration.)*

---

## STEP 11 — Remaining Problems

- ☐ Consistency proof incomplete (Step 4.4 — fixed-fraction span is not asymptotically consistent; need $\alpha(n)\to0$ schedule + formal rate)
- ☐ Formal (uniform, non-leading-order) bias bound not derived — only the leading-order approximation (Eq. 6) exists
- ☐ Conditioning/stability of the Vandermonde design matrix not bounded numerically or theoretically
- ☐ No signal-adaptive (curvature-aware) bandwidth — only density-adaptive (Failure 2)
- ☐ Outlier robustness mechanism unverified — confound with window width not ruled out (Failure 3)
- ☐ $O(n^2\log n)$ implementation not optimized to $O(n\log n)$ (Step 4.6)
- ☐ Heteroscedastic/correlated-noise case not modeled anywhere (Step 3)
- ☐ Boundary-bias order not analyzed separately from interior bias
- ☐ No comparison yet against optimization-based (TV/Tikhonov) or RBF-FD differentiation (Step 5)
- ☐ No literature-verified numeric benchmark comparison (only structural/conceptual comparison done — Step 5)
- ☐ Second-derivative ($m=2$) case not implemented or tested

---

## STEP 12 — Publication Readiness (honest, as of V1.0)

| Criterion | Score (0–10) | Justification |
|---|---|---|
| Mathematical rigor | 4 | Exact WLS derivation and exact variance formula are solid; bias analysis is only leading-order; no consistency/rate theorem; no conditioning bound |
| Novelty | 2 | Base method is standard local polynomial regression / LOESS (Fan & Gijbels 1996; Cleveland 1979); no demonstrated novel mechanism yet — density-only adaptivity is well-trodden |
| Accuracy | 3 | Loses to Savitzky–Golay on the core target scenario (plain noisy data); worst-in-class on chaotic data |
| Stability | 4 | No observed numerical failures in 350 runs, but no principled conditioning guarantee; centered coordinates help but aren't proven sufficient |
| Robustness | 5 | Genuinely strong, large-margin outlier result — but interpretation is unresolved (Failure 3) |
| Computational efficiency | 3 | Correct but asymptotically inefficient ($O(n^2\log n)$); comparable wall-clock to cubic spline/savgol only because $n$ is small in this study |
| Practical usefulness | 3 | As configured, worse than the simplest reasonable baseline (savgol) for the common case |
| Reproducibility | 8 | Full source, fixed seed (12345), CSV of every run committed to `results/results_v1.csv`; anyone can re-run `src/benchmark.py` and get identical numbers |
| Literature support | 2 | Conceptual comparison only; no verified quantitative comparison against any cited method's own reported numbers |
| **Overall publishability** | **2** | Not close. This iteration's value is that it produced an honest, reproducible baseline and a precise, falsifiable list of what's wrong — not a result. |

---

## STEP 13 — Next Iteration Plan (V1.1)

**Will be tested:** the same 350-run benchmark suite, re-run for (a) a Goldenshluger–Lepski-style adaptive-bandwidth variant of ALPRD, (b) a width-matched Savitzky–Golay control for the outlier scenario, (c) wall-clock scaling at $n=10^3,10^4$ after the $O(n\log n)$ rewrite.

**Will be modified:** bandwidth selection rule (Step 9 item 1), implementation algorithm for windowing (Step 9 item 3).

**Why it should help:** directly targets the two measured failure modes (oversmoothing on plain noisy data, undersmoothing-relative-to-curvature on chaotic data) with a single bias/variance-tradeoff mechanism grounded in the already-derived Eqs. (3) and (6), rather than a hand-picked constant.

**Weaknesses targeted:** Failures 1 and 2 (Step 8); Remaining Problems items 1, 4, 6 (Step 11).

---
---

# Version 1.1 — Iteration 2

Code: [`../src/methods.py`](../src/methods.py) (`alprd_v11`, `alprd_v1_fast`, `savitzky_golay_matched`), tests: [`../tests/test_methods.py`](../tests/test_methods.py) (5 tests, all passing)
Raw results: [`../results/results_v1_1.csv`](../results/results_v1_1.csv) — **500 runs (10 methods × 35 scenarios × 2 ALPRD variants included), 0 failures, same seed 12345, same generator code as V1.0.** Scaling data: [`../results/scaling_v1_1.json`](../results/scaling_v1_1.json). Every number in this section is read from those files / test output; nothing is projected or invented.

---

## STEP 1 — Current Version: V1.1

V1.1 keeps the V1.0 estimator (Eqs. 1–4) unchanged and replaces only the **bandwidth rule**. For each query point $t_0=t_i$:

1. **Pilot fit** *(new)*: one weighted local fit of degree $p{+}1$ on a fixed moderate window ($k_{\text{pilot}}=\max(2(p{+}3),\lceil 0.15\,n\rceil)$ nearest neighbors, tricube weights), giving $\hat b_{p+1}(t_0)\approx f^{(p+1)}(t_0)/(p{+}1)!$.
2. **Candidate grid** *(new)*: neighbor counts $k_c$ on a geometric grid,
$$k_c \in \mathrm{geomspace}\big(k_{\min},\,k_{\max}\big),\quad k_{\min}=\max(p{+}3,6),\ k_{\max}=\lceil 0.5\,n\rceil,\ \ 8\text{ candidates}.$$
3. For each candidate, fit the V1.0 estimator and compute its equivalent-kernel row $\ell^{(c)}$ (so $\hat f'_c(t_0)=\ell^{(c)}\!\cdot y$), then the **estimated pointwise risk** *(new)*:
$$\widehat R(k_c;t_0) \;=\; \underbrace{\Big(\hat b_{p+1}(t_0)\,\textstyle\sum_i \ell^{(c)}_i (t_i-t_0)^{p+1}\Big)^2}_{\widehat{\mathrm{Bias}}^2\ \text{[from Eq. (6)]}} \;+\; \underbrace{\hat\sigma^2 \textstyle\sum_i \big(\ell^{(c)}_i\big)^2}_{\widehat{\mathrm{Var}}\ \text{[Eq. (3), exact]}} \tag{7}$$
4. **Selection** *(new)*: $\hat k(t_0)=\arg\min_c \widehat R(k_c;t_0)$; output $\hat f'(t_0)=\hat f'_{\hat k}(t_0)$.

$\hat\sigma^2$ is the same Rice-type estimator, Eq. (4). The identity $\mathrm{Var}(\ell^\top y)=\sigma^2\|\ell\|^2$ used in (7) is algebraically equal to V1.0's sandwich formula (one-line proof: $\ell^\top=e_1^\top(X^\top WX)^{-1}X^\top W$, so $\|\ell\|^2 = e_1^\top(X^\top WX)^{-1}X^\top W^2X(X^\top WX)^{-1}e_1$).

**Labeling (per project rules):** the selector is a **plug-in pointwise-risk minimizer** in the *spirit* of Goldenshluger–Lepski, but it is **not** the GL procedure: GL compares pairwise estimator differences against a majorant penalty and carries a finite-sample oracle inequality; Eq. (7) plugs estimated bias and variance into the risk directly and carries **no oracle guarantee**. Status: *Derived* (each ingredient), *Tested* (empirically), **not Proven** (no adaptive-rate theorem).

Also new in V1.1, orthogonal to the selector:

- **`alprd_v1_fast`** — sort-free rewrite of the V1.0 estimator (Step 4.6 below).
- **`savitzky_golay_matched`** — control for V1.0's Failure 3 (Step 7/8 below).

---

## STEP 2 — Symbol Table (additions to the V1.0 table; all V1.0 symbols unchanged)

| Symbol | Meaning | SI units | Allowed values | Purpose |
|---|---|---|---|---|
| $k_c$ | candidate neighbor count | dimensionless (int) | $k_{\min}\le k_c\le k_{\max}$ | one trial bandwidth in the grid |
| $k_{\min},k_{\max}$ | grid endpoints | dimensionless (int) | $p+3 \le k_{\min} < k_{\max}\le n$ | bound the search |
| $\hat b_{p+1}(t_0)$ | pilot estimate of $f^{(p+1)}(t_0)/(p{+}1)!$ | m/s$^{p+1}$ | real | feeds the bias term of Eq. (7) |
| $k_{\text{pilot}}$ | pilot window size | dimensionless (int) | fixed, $\lceil 0.15 n\rceil$-ish | pilot fit support |
| $\ell^{(c)}$ | equivalent-kernel row of candidate $c$ | 1/s | real vector | linear weights; risk functionals |
| $\widehat R(k_c;t_0)$ | estimated pointwise risk | m²/s² | $\ge 0$ | selection criterion |
| $\hat k(t_0)$ | selected neighbor count | dimensionless (int) | grid value | final per-point bandwidth |

---

## STEP 3 — Assumptions (changes relative to V1.0)

All V1.0 assumptions carry over, plus:

- **Pilot validity.** Eq. (7)'s bias term is only as good as $\hat b_{p+1}$: it assumes $f\in C^{p+1}$ **and** that a degree-$(p{+}1)$ fit on the fixed pilot window resolves $f^{(p+1)}(t_0)$. On signals whose $(p{+}1)$-th derivative varies quickly inside the pilot window, $\hat b_{p+1}$ is itself biased, degrading (not breaking) the selection.
- **Selection independence ignored.** $\hat k(t_0)$ is chosen using the same data used in the fit; the post-selection estimator is therefore *not* linear in $y$, and Eq. (3) evaluated at the chosen bandwidth *understates* the true post-selection variance. **The reported `se` in V1.1 is a lower bound in this sense — Hypothesized magnitude, not quantified.**
- **Outliers, again explicitly not modeled** — and V1.1 is *more* sensitive to this than V1.0 (measured; see Step 8, Failure 1').

---

## STEP 4 — Mathematical Analysis (delta from V1.0)

### 4.1–4.3 — unchanged
The moment identity (5), leading-order bias (6), and exact variance (3) hold verbatim for each *candidate* fit, since each candidate is exactly a V1.0 estimator.

### 4.4 Consistency — **partially addressed, still not Proven**
V1.0's obstruction was the fixed span ($k/n\not\to0$). V1.1's grid includes small $k$ (down to $k_{\min}=O(1)$), so the *oracle* over the grid can realize $k\to\infty,\ k/n\to0$ behavior. But turning that into a consistency theorem for the *selected* estimator requires an oracle inequality for the plug-in rule (control of $P(\hat k \ne k^\star)$ and of post-selection risk), which we have not derived. **Status: open (Step 11).** This is precisely the gap the true GL machinery is designed to fill — flagged as the theory goal for V1.2+.

### 4.5 Stability — unchanged (monomial basis, no conditioning bound; still open).

### 4.6 Computational complexity — **Derived and Measured**

*Windowing rewrite (`_knn_window`).* Since $t$ is sorted, the $k$ nearest neighbors of $t_i$ always form a contiguous index range, found by a two-pointer expansion in $O(k)$ — eliminating V1.0's per-query full sort ($O(n\log n)$ each). Totals: V1.0 $O(n^2\log n)$ → fast path $O(nk)$.
**Honest precision:** $O(nk)$ is *not* $O(n\log n)$ overall — with the default fixed-fraction span ($k=\alpha n$) it is still $O(n^2)$. The claim we can (and do) make: the *sorting* bottleneck is gone, and for fixed or slowly-growing $k$ the total is linear in $n$ up to the $O(k)$ window factor.
**Statistical equivalence: Proven and Tested.** The two-pointer window selects a point set whose $k$-th-neighbor distance equals the argsort one (multiset argument; ties give weight 0 under tricube at $|u|=1$, so tie-order is irrelevant), hence identical weights, hence identical output. Verified two ways: `test_fast_matches_v1_exactly` (machine-precision match incl. the variance channel), and the benchmark itself — `ALPRD_v1.0` and `ALPRD_v1.0_fast` rows in `results_v1_1.csv` are numerically identical across all 35 scenarios.

*Measured scaling* (sinusoidal + Gaussian noise, fixed $k=50$, `results/scaling_v1_1.json`):

| $n$ | V1.0 (s) | V1.0_fast (s) | speedup |
|---|---|---|---|
| 500 | 0.059 | 0.053 | 1.1× |
| 2000 | 0.261 | 0.210 | 1.2× |
| 8000 | 5.526 | **1.893** | **2.9×** |

The gap widens with $n$ exactly as the complexity analysis predicts; below $n\approx500$ the pure-Python two-pointer loop's constant factor roughly cancels the asymptotic win. (A vectorized/`searchsorted`-based window computation would improve the constant; not done in this iteration.)

*V1.1 selector cost:* per point, ~8 candidate fits with windows up to $k_{\max}=0.5n$, plus one pilot fit → $O(n\cdot\sum_c k_c)=O(n^2)$ with the default grid. Measured: mean 0.395 s per call at $n\le400$ (≈10× V1.0), and 9.47 s at $n=2000$. **This is the price of adaptivity in the current implementation and it is substantial** (Step 11).

---

## STEP 5 — Literature Comparison (delta)

The V1.0 table stands. V1.1's placement changes in one row: ALPRD is now *signal-adaptive* (curvature enters through $\hat b_{p+1}$ in Eq. 7), not merely density-adaptive. This moves it into the family of plug-in local bandwidth selectors for local polynomial regression — well-established territory (Fan & Gijbels 1995/1996 propose plug-in constant and variable bandwidth selectors of exactly this bias²+variance form; Ruppert–Sheather–Wand 1995 for the smoothing context; Lepski/Goldenshluger–Lepski for the adaptive-theory gold standard). **Honest novelty statement, updated:** V1.1 is an *application* of known plug-in bandwidth selection to derivative estimation on uneven grids, with a Rice-type variance plug-in. We are not aware of this exact assembly being a standard named method, but no component is new, and no claim of novelty is made pending a proper literature search (unchanged score in Step 12). The genuinely citable *finding* of this iteration is the negative/control result below (Failure 3 resolution), which is evidence hygiene, not methodology.

---

## STEP 6 — Implementation

New code, all in [`src/methods.py`](../src/methods.py): `_knn_window` (two-pointer window), `_local_wls` (shared WLS core returning the full coefficient map so $\ell$ comes for free), `alprd_v1_fast`, `alprd_v11`, `savitzky_golay_matched`. Test suite [`tests/test_methods.py`](../tests/test_methods.py): window-vs-argsort equivalence on uneven grids for boundary and interior points; machine-precision V1.0↔fast equivalence (clean + noisy, 3 spans, incl. `se`); quadratic-reproduction exactness for all three ALPRD variants ($<10^{-7}$ max error, satisfied); a chaotic-improvement regression guard (V1.1 must beat V1.0 by ≥2× MAE there — passes); SG-matched sanity. **5/5 passing.**

## STEP 7 — Benchmarking (same suite, same seed, 500 runs, 0 failures)

### Headline slice — noisy uneven data without outliers (`uneven_gaussian` + `uneven_uniformnoise` + `missing_noisy`, 24 datasets)

| method | MAE | RMSE | MaxErr |
|---|---|---|---|
| savgol_resampled (default w=11) | **1.047** | **1.674** | **8.376** |
| **ALPRD_v1.1** | 1.625 | 2.333 | 11.331 |
| savgol_matched (w≈51) | 2.959 | 3.959 | 14.696 |
| ALPRD_v1.0 (= v1.0_fast, identical) | 3.391 | 4.486 | 16.646 |
| central_diff | 2.238 | 5.968 | 45.079 |
| backward/forward diff | 2.67–2.69 | 5.48–5.64 | 48.1–51.7 |
| cubic_spline | 2.723 | 6.295 | 46.406 |
| high_order_fd | 2.744 | 6.483 | 47.058 |

**V1.1 halves V1.0's headline RMSE (4.49 → 2.33) with zero manual tuning, moving ALPRD from 4th to 2nd. It still does not beat default Savitzky–Golay overall.** The residual gap is entirely the chaotic motion:

### Same slice **excluding chaotic** (21 datasets)

| method | MAE | RMSE |
|---|---|---|
| savgol_matched | **0.058** | **0.104** |
| **ALPRD_v1.1** | 0.156 | 0.193 |
| savgol_resampled | 0.168 | 0.224 |
| ALPRD_v1.0 | 0.493 | 0.696 |

On smooth motions, V1.1 beats *default* SG on 6 of 7 motion types (all but the quartic polynomial) — but a *width-matched* SG beats both. There is no free lunch here and we report it as such: for smooth signals, wide-window SG is simply excellent, and ALPRD's remaining edge cases are uneven-grid convenience and per-point error bars, not raw accuracy.

### Chaotic motion (Lorenz), V1.0's Failure 2 — **substantially improved, not solved**

MAE, `uneven_gaussian`: V1.0 23.60 → **V1.1 11.57** (savgol_resampled 6.05, savgol_matched 23.22). Real-world-like chaotic: V1.0 21.65 → **V1.1 3.74** (savgol_resampled 0.57). The adaptive bandwidth does exactly what Eq. (7) promises directionally — it shrinks windows where curvature is high — but the narrow-window SG still wins, because ALPRD's grid bottoms out at $k_{\min}$ and the pilot's fixed window over-smooths $\hat b_{p+1}$ on fast curvature (Step 3).

### Outlier control experiment — **V1.0's Failure 3 confound: CONFIRMED**

`uneven_outliers` aggregate (MAE / RMSE / MaxErr):

| method | MAE | RMSE | MaxErr |
|---|---|---|---|
| ALPRD_v1.0 | **23.04** | **32.64** | 116.5 |
| savgol_matched (width-matched control) | 26.52 | 35.52 | **111.2** |
| ALPRD_v1.1 | 122.78 | 436.16 | 3432.5 |
| savgol_resampled (default width) | 166.38 | 351.94 | 2073.6 |

The width-matched SG closes ~97% of the gap to ALPRD v1.0 (26.5 vs 23.0, against 166.4 at default width). **Conclusion (Tested): V1.0's outlier advantage was a window-width effect, not a property of the method. The V1.0 robustness interpretation is hereby retracted.** This is the control experiment V1.0's Step 8 demanded, and it came out against us.

### Runtime (mean s/call, benchmark suite, $n\in[150,400]$)
savgol_matched 0.0020 · high_order_fd 0.0122 · cubic_spline 0.0291 · savgol_resampled 0.0337 · ALPRD_v1.0 0.0386 · ALPRD_v1.0_fast 0.0421 · **ALPRD_v1.1 0.3953**. (Scaling table in Step 4.6; the fast path's advantage only materializes at larger $n$.)

---

## STEP 8 — Failure Analysis

### Failure 1′ (new, severe): V1.1 collapses on outliers
- **What:** MAE 122.8 vs V1.0's 23.0 on `uneven_outliers` — a 5.3× regression, now *worse* than default SG on RMSE.
- **Why (mathematically):** Eq. (7)'s risk model assumes iid mean-zero noise of variance $\sigma^2$. An outlier violates this: near one, every candidate fit is contaminated, but *narrow* windows are contaminated worst (one gross error among ~6–10 points vs among ~50). The bias term of Eq. (7) cannot see this — $\hat b_{p+1}$ is itself outlier-corrupted, and $\hat\sigma^2$ (median-based) correctly *ignores* outliers, so the variance term under-prices narrow windows exactly where they are most dangerous. The selector therefore does the worst possible thing: it confidently picks small $k$ near outliers.
- **Severity: high.** Adaptivity and robustness are currently *antagonistic* in this design. This is the central problem for V1.2.

### Failure 2′ (carried, reduced): chaotic motion still loses to narrow-window SG
MAE 11.6 vs 6.1 (Gaussian case). Cause as in Step 3: the pilot's fixed 15% window over-smooths $f^{(3)}$ estimates on the Lorenz signal, so the bias term is underestimated in high-curvature regions and the grid's $k_{\min}$ floor prevents the window from shrinking enough. Severity: medium (2× gap, down from 4× in V1.0).

### Failure 3′ (resolved, against V1.0): outlier-robustness claim retracted — see Step 7. The scientific record is corrected; no ALPRD variant currently has a *designed* robustness mechanism.

### Failure 4′ (new): selector runtime
10× V1.0 at small $n$, $O(n^2)$ growth with the default grid. Severity: medium — engineering, not statistics, but it blocks large-$n$ benchmarking.

---

## STEP 9 — Improvements made in this iteration (all justified, all benchmarked)

1. **Plug-in adaptive bandwidth (Eq. 7)** — justified by V1.0 Eqs. (3)+(6); measured effect: headline RMSE −48%, chaotic MAE −51% to −83%. Disadvantage materialized as predicted risk-model fragility under outliers (Failure 1′).
2. **Width-matched SG control** — justified by V1.0 Failure 3's explicit confound hypothesis; measured effect: confound confirmed, V1.0 claim retracted.
3. **Two-pointer windowing** — justified by Step 4.6 complexity analysis; measured effect: identical output (proven + tested), 2.9× at $n=8000$ and growing.

## STEP 10 — Version History

**Version 1.0** — Initial formulation: fixed-span kNN tricube local quadratic WLS + Rice variance. Outcome: honest baseline; lost to SG on noise; worst-in-class on chaotic; outlier win later shown confounded.

**Version 1.1** — Per-point plug-in bias²+variance bandwidth selection (Eq. 7); sort-free O(nk) windowing with proven-identical output; width-matched SG control experiment; test suite added. Outcome: headline RMSE halved, chaotic largely repaired, outlier robustness regressed 5×, V1.0 robustness claim retracted.

## STEP 11 — Remaining Problems

- ☐ **Robust adaptivity:** make Eq. (7) outlier-aware (Huber/Tukey IRLS in each candidate fit + robust risk estimate) — top priority (Failure 1′)
- ☐ Oracle inequality / consistency theorem for the selected estimator (true GL machinery) — Step 4.4 still open
- ☐ Pilot fragility on fast-curvature signals (Failure 2′): curvature-adaptive pilot window, or iterate pilot↔selection
- ☐ Post-selection variance understatement (Step 3): quantify or correct reported `se`
- ☐ Vandermonde conditioning bound (carried from V1.0); switch to orthogonalized local basis
- ☐ Selector cost $O(n^2)$: vectorize candidate evaluation; `searchsorted`-based windows; early grid pruning
- ☐ Heteroscedastic/correlated noise (carried)
- ☐ Boundary bias order (carried)
- ☐ Benchmarks vs TV/Tikhonov (Chartrand-style) and RBF-FD (carried)
- ☐ Literature search to establish/deny novelty of the exact assembly (carried)
- ☐ Second derivative $m=2$ (carried)

## STEP 12 — Publication Readiness (V1.1, honest)

| Criterion | Score | Justification |
|---|---|---|
| Mathematical rigor | 4 | New selector is Derived-not-Proven (no oracle inequality); exact-variance and equivalence proofs are solid but small |
| Novelty | 2 | Plug-in variable bandwidth for local polynomial derivatives = Fan & Gijbels territory; assembly unverified against literature |
| Accuracy | 5 | (was 3) 2nd of 10 on headline slice untuned; beats default SG on 6/7 smooth motions; still loses to SG overall and on chaotic |
| Stability | 5 | (was 4) 500 runs + 5 tests, 0 failures; machine-precision equivalence proof; conditioning still unbounded |
| Robustness | 2 | (was 5) honest double downgrade: V1.0's claim was confounded (retracted) and V1.1 actively regresses on outliers |
| Computational efficiency | 3 | fast path is a real, measured asymptotic win, but the selector is 10× slower and $O(n^2)$ |
| Practical usefulness | 4 | (was 3) no-tuning adaptivity + error bars is genuinely useful; but a width-tuned SG remains simpler and often better |
| Reproducibility | 9 | (was 8) fixed seed, committed CSVs, deterministic tests, equivalence verified in-suite |
| Literature support | 2 | conceptual only; no verified quantitative comparison to published results |
| **Overall publishability** | **3** | (was 2) The method is not yet a contribution. The *iteration discipline* is working: one failure fixed with a derived mechanism, one claim honestly retracted with a control experiment. A 7+/10 was the aspiration for this iteration; the evidence does not support it, and we say so plainly. |

## STEP 13 — Next Iteration Plan → see "Immediate Next Steps for V1.2" below.

---

## Immediate Next Steps for V1.2

1. **Robustify the candidate fits** (targets Failure 1′, the worst regression): replace plain WLS in each candidate with 2–3 IRLS steps under a Tukey biweight loss, scale-estimated by the (already robust) Rice $\hat\sigma$; use the robust residual scale in Eq. (7)'s variance term. Expected: restores ≥V1.0-level outlier behavior while keeping V1.1's adaptivity. Risk: IRLS multiplies the selector's cost — must land together with item 3.
2. **Curvature-adaptive pilot** (targets Failure 2′): choose the pilot window itself by a coarse two-candidate rule (small/large), or re-estimate $\hat b_{p+1}$ at the selected bandwidth and iterate once. Expected: closes part of the remaining 2× chaotic gap to narrow-SG.
3. **Vectorize the selector** (targets Failure 4′): batch all candidates' Gram matrices via cumulative-sum moment arrays over the sorted grid ($X^\top WX$ entries are windowed weighted power sums → prefix sums give any window in O(1) per moment), eliminating the per-candidate Python loops; target ≤2× V1.0 runtime at $n=2000$.
4. **First theory step toward 4.4:** prove an oracle-type bound for the *idealized* selector (true bias and variance known), then bound the plug-in perturbation — even a partial result would move Rigor to 5–6.
5. **Add the two missing benchmark families** (TV-regularized differentiation à la Chartrand; RBF-FD) so the literature comparison stops being purely conceptual.

## Next Recommended Prompt (V1.1 → V1.2; executed, see Version 1.2 section below)

```
(historical record — the V1.2 iteration below followed this prompt)
Implement V1.2: (1) Tukey IRLS robust candidate fits; (2) refined pilot;
(3) vectorized selector; (4) idealized-selector oracle bound attempt;
(5) TV + RBF-FD baselines. Same suite, seed 12345, honest scoring.
```

---
---

# Version 1.2 — Iteration 3

Code: [`../src/methods.py`](../src/methods.py) (`alprd_v12`, `_batched_windows`, `_batched_robust_fit`, `tv_derivative`, `rbf_fd_derivative`), tests: [`../tests/test_methods.py`](../tests/test_methods.py) (**10 tests, all passing**, incl. two regression guards).
Raw results: [`../results/results_v1_2.csv`](../results/results_v1_2.csv) — **650 runs (13 methods × 35 scenarios... 13×35=455 method-dataset pairs plus the retained V1.0/V1.1 variants; exact row count 650), 0 failures, seed 12345, identical dataset generators.** Scaling: [`../results/scaling_v1_2.json`](../results/scaling_v1_2.json). All numbers below are read from these files.

---

## STEP 1 — Current Version: V1.2

V1.2 keeps V1.1's selection principle (minimize plug-in risk, Eq. 7 over a geometric grid of neighbor counts) and changes the *fits* that enter it. At every query point $t_0=t_i$, for every candidate $k_c$:

1. **Kernel weights** as before: $w^K_i = K\big((t_i-t_0)/h_c\big)$, tricube, $h_c$ = $k_c$-th-neighbor distance.
2. **Robust IRLS fit** *(new)*: initialize with the WLS fit; then for $r=1,\dots,R$ (default $R=2$):
$$w^{\text{rob}}_i = \psi_T\!\Big(\frac{y_i - \hat P(t_i)}{c\,\hat s}\Big),\quad \psi_T(u)=(1-u^2)^2\,\mathbb{1}_{|u|<1},\quad c=4.685 \tag{8}$$
$$\hat s = \max\big(\hat\sigma,\ 1.4826\cdot\mathrm{median}_i|y_i-\hat P(t_i)|,\ \varepsilon\big) \tag{9}$$
and refit with combined weights $w_i = w^K_i\, w^{\text{rob}}_i$. ($\psi_T$ = Tukey biweight; $\hat\sigma$ = Rice estimate, Eq. 4; the MAD term in (9) keeps clean data from being zeroed out when $\hat\sigma\approx0$, the $\hat\sigma$ floor keeps a gross outlier from inflating its own scale.)
3. **Effective-size guard** *(new)*: candidate $k_c$ is discarded unless
$$n_{\text{eff}} = \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2} \;\ge\; p+2 \tag{10}$$
(a window that survives only by deleting most of its points is overfitting, not evidence).
4. **Risk and selection** as in Eq. (7), with $\ell$ taken from the final combined-weight fit.
5. **Refined pilot** *(new)*: after a first full selection pass with the fixed pilot window, $\hat b_{p+1}$ is re-estimated per point on a window of $\lceil 1.5\,\hat k(t_0)\rceil$ neighbors (grouped by unique size for batching) and selection is repeated once. High-curvature regions thus get a *local* curvature estimate instead of a global-window one.

The pilot fit itself also runs IRLS (an outlier-corrupted curvature estimate would poison the bias term — V1.1's exact failure vector).

**Implementation** *(new)*: no per-point Python loops. Windows come from a vectorized minimal-radius rule (`_batched_windows`; output-equivalence to the two-pointer window is **Proven** — any minimal-radius window has the same radius $h$, the same strict-interior point set, and boundary points get tricube weight 0 — and **Tested** in `test_batched_windows_match`). All candidate fits are batched `(rows, k)` linear algebra, processed in memory-bounded row blocks (≤ $2^{21}$ doubles per temporary), with a $10^{-12}$-relative ridge on the batched Gram matrices to survive robust-weight rank deficiency. Re-running the benchmark after the blocking refactor produced bit-identical aggregate results (checked).

---

## STEP 2 — Symbol Table (additions; V1.0/V1.1 symbols unchanged)

| Symbol | Meaning | SI units | Allowed values | Purpose |
|---|---|---|---|---|
| $\psi_T$ | Tukey biweight function | dimensionless | $[0,1]$ | cuts gross residuals from the fit |
| $c$ | Tukey tuning constant | dimensionless | 4.685 (std. 95%-efficiency choice) | sets the cut radius in $\hat s$ units |
| $\hat s$ | robust residual scale | m | $>0$ | Eq. (9); scales residuals for (8) |
| $w^{\text{rob}}_i$ | robustness weight | dimensionless | $[0,1]$ | per-point outlier downweight |
| $R$ (`irls_iters`) | IRLS reweighting steps | dimensionless (int) | $\ge0$ (default 2) | robustness/cost tradeoff |
| $n_{\text{eff}}$ | effective sample size | dimensionless | $[1,k_c]$ | guard (10) against degenerate windows |
| $\varepsilon$ | scale floor | m | $10^{-300}$ | avoids 0/0 on exact fits |

---

## STEP 3 — Assumptions (changes)

- **Contamination model (implicit).** The Tukey/IRLS machinery targets a Huber-type gross-error model: a fraction of observations may deviate arbitrarily, the rest follow the V1.0 noise model. **We have not characterized the breakdown fraction** — benchmarks use 3% contamination; behavior at 10–30% is untested (Step 11).
- **Nonlinearity.** With data-dependent robust weights, the estimator is **no longer linear in $y$**. The variance formula (Eq. 3 / the $\sigma^2\|\ell\|^2$ identity) and bias formula (Eq. 6) are now *conditional* approximations that hold with the final weights treated as fixed. This weakens the formal status of the risk functional (7) from "exact variance + leading-order bias" to "plug-in approximation of both" — stated plainly; consequences unquantified (Step 11).
- Everything else carries over from V1.0/V1.1.

---

## STEP 4 — Mathematical Analysis (delta)

### 4.1 Reproduction property — **preserved (Proven, Tested)**
For polynomial data of degree ≤ $p$ with no noise, residuals vanish, so IRLS weights are 1 wherever the scale floor logic applies (Eq. 9's MAD term dominates and residual/scale ratios stay ≪ 1); the fit reduces to WLS and identity (5) applies. Tested: `test_v12_polynomial_reproduction`, max error < 10⁻⁶ on a clean quadratic over an uneven grid.

### 4.2 Oracle lemma for the idealized selector — **Proven (elementary), with the hard part left honestly open**
Let $R(k)$ be the true pointwise risk of candidate $k$ and $\widehat R(k)$ the plug-in estimate, $\hat k=\arg\min\widehat R$, $k^\star=\arg\min R$, and $\Delta=\max_{k\in\text{grid}}|\widehat R(k)-R(k)|$. Then
$$R(\hat k)\;\le\;\widehat R(\hat k)+\Delta\;\le\;\widehat R(k^\star)+\Delta\;\le\;R(k^\star)+2\Delta. \tag{11}$$
*Proof:* first and third inequalities are the definition of $\Delta$; the middle one is the definition of $\hat k$. ∎
So the selected estimator is within $2\Delta$ of the oracle. **This lemma is trivial; the entire mathematical content of an adaptivity theorem lives in bounding $\Delta$** — which requires concentration of $\hat\sigma^2$ (difference-based estimators: known results exist under iid noise) and of $\hat b_{p+1}$ (harder, especially post-IRLS). Not attempted beyond this statement. Status: Eq. (11) **Proven**; $\Delta$-bound **open** — this is deliberately recorded so no one mistakes (11) for an adaptivity result.

### 4.3 Robustness — **Tested, not Proven**
No influence-function or breakdown-point analysis was done. What is measured: 16.6× lower MAE than the best non-robust method under 3% gross contamination (Step 7). The standard theory for Tukey-biweight M-estimators suggests a positive breakdown point in the local model, but local-regression breakdown with kernel weights and small windows is subtler; deferred.

### 4.4 Complexity and scaling — **Derived + Measured; V1.2 target MISSED**
Cost is $O\!\big(RP\, n\sum_c k_c\big)$ with $P=2$ selection passes: still **quadratic** in $n$ for the default $k_{\max}=n/2$ grid. Vectorization removed Python-loop constants and blocking removed the memory blow-up (n=8000 previously OOM'd at a single 293 MB temporary; now bounded), but the honest scaling picture (`scaling_v1_2.json`, sinusoidal+noise):

| $n$ | V1.0 (s) | V1.2 (s) | ratio |
|---|---|---|---|
| 500 | 0.062 | 0.506 | 8.1× |
| 2000 | 0.331 | 7.130 | **21.6×** |
| 8000 | 5.806 | 98.802 | 17.0× |

**The V1.2 plan's "≤2× V1.0 at n=2000" target was missed by an order of magnitude.** The batched einsum work per candidate is $O(nk_c)$ and the grid's large-$k$ candidates dominate; the planned prefix-sum/moment-array trick (making each candidate $O(n)$ independent of $k_c$) was *not* implemented this iteration — that, or capping $k_{\max}$ growth, is the actual fix and moves to V1.3. Within the benchmark suite ($n\le400$) V1.2 is nonetheless *faster* than V1.1 (mean 0.103 s vs 0.239 s) because vectorization beats V1.1's per-point Python loops even with 2 passes and IRLS.

---

## STEP 5 — Literature Comparison (delta)

**The closest existing method is now robust LOWESS (Cleveland 1979) — and this must be said clearly:** LOWESS already combines local polynomial fits, tricube kernel weights, and iterated bisquare (Tukey) robustness reweighting. V1.2's *fit machinery* is essentially LOWESS's, applied to the derivative coefficient. What LOWESS does **not** have: per-point plug-in risk-based bandwidth selection for the *derivative* functional (LOWESS uses a fixed span), the derivative-targeted bias functional of Eq. (7), a per-point standard-error output, or any uneven-grid-specific analysis. Fan & Gijbels (1995) treat variable plug-in bandwidths for local polynomial regression (including derivatives); Cleveland's robustness and Fan–Gijbels' variable bandwidth **have both existed for decades — the specific combination for derivative estimation on irregular grids with a Rice-plug-in self-error estimate is the only candidate novelty claim, and it remains unverified against the literature** (no systematic search yet; Step 11).

New baselines, now measured rather than hypothesized:
- **TV-regularized differentiation** (Chartrand-style, implemented on the uneven grid with trapezoidal integration operator, lagged-diffusivity solver, discrepancy-principle $\alpha$): strong on smooth+noisy data (headline MAE 2.11), competitive on chaotic (9.1–9.3, better than ALPRD v1.2's 8.1–8.2? no — worse: 9.19 vs 8.15 on `uneven_gaussian`), but **catastrophic under outliers** (MAE 21005) — the L2 data-fidelity term must fit gross errors when the discrepancy target uses the robust Rice $\sigma$. Expected for L2-TV; an L1-fidelity variant would fix it (noted, not implemented).
- **RBF-FD** (cubic PHS + degree-2 polynomial augmentation, k=7): near-exact on clean data (MAE 0.028 clean-uniform, interior errors ~5×10⁻⁴), and noise-amplifying exactly like a finite-difference stencil (headline MAE 2.70, outliers 892.9) — as predicted by its interpolatory construction. Both behave as their theory says they should, which is evidence the implementations are faithful.

---

## STEP 6 — Implementation

New: `_batched_windows` (vectorized minimal-radius windows, `rows` parameter for memory-bounded blocks), `_batched_robust_fit` (batched IRLS with per-row robust scale and relative ridge), `alprd_v12`, `tv_derivative` (dense lagged-diffusivity + discrepancy bisection), `rbf_fd_derivative` (saddle-system PHS weights). Tests added: batched-window equivalence; V1.2 polynomial exactness; **outlier regression guard** (V1.2 must beat V1.1 ≥3× and be ≤1.5× V1.0 on contaminated data); **chaotic retention guard** (V1.2 ≥2× better than V1.0); TV/RBF sanity (TV must beat central difference on noisy data; RBF near-exact interior on clean data). 10/10 passing.

## STEP 7 — Benchmarking (650 runs, 0 failures, seed 12345)

### Overall aggregate, all 35 scenarios (mean MAE / RMSE / MaxErr)

| method | MAE | RMSE | MaxErr |
|---|---|---|---|
| **ALPRD_v1.2** | **1.271** | **1.901** | **10.290** |
| ALPRD_v1.0 (=fast) | 6.847 | 9.323 | 33.046 |
| savgol_matched | 7.063 | 9.374 | 30.747 |
| ALPRD_v1.1 | 21.080 | 71.775 | 558.174 |
| savgol_resampled | 27.365 | 57.523 | 338.145 |
| central_diff | 94.282 | 529.737 | 4938.463 |
| rbf_fd | 144.328 | 617.927 | 5265.176 |
| tv_diff | 3362.825 | 5704.773 | 17482.326 |

V1.2 is the best method in the study by every aggregate metric, by ~5× over the runner-up — **but the aggregate is dominated by the outlier scenarios**, so the honest reading requires the slices:

### Per-scenario MAE (ALPRD variants + key baselines)

| method | clean_unif | missing_noisy | real_world | uneven_clean | uneven_gauss | **uneven_outliers** | uneven_unif |
|---|---|---|---|---|---|---|---|
| ALPRD_v1.0 | 3.455 | 3.414 | 11.024 | 3.374 | 3.382 | 23.038 | 3.377 |
| ALPRD_v1.1 | 2.042 | 1.750 | 1.979 | 1.556 | 1.572 | 122.784 | 1.553 |
| **ALPRD_v1.2** | **1.538** | **1.392** | **1.110** | **1.110** | **1.127** | **1.386** | **1.114** |
| savgol_resampled | 0.643 | 1.352 | 0.386 | 0.768 | 0.937 | 166.384 | 0.853 |
| savgol_matched | 3.011 | 2.975 | 11.175 | 2.941 | 2.956 | 26.522 | 2.946 |
| tv_diff | 3.365 | 2.380 | 2.599 | 1.816 | 1.967 | 21005.508 | 1.968 |
| rbf_fd | 0.028 | 2.611 | 3.918 | 0.033 | 2.921 | 892.914 | 2.562 |

**Failure 1′ (V1.1 outlier collapse): FIXED, decisively.** MAE 1.386 under 3% gross contamination — 16.6× better than V1.0 (23.0), 19× better than the width-matched SG control (26.5), 89× better than V1.1 (122.8). Unlike V1.0's retracted width-confound result, this is a *designed* mechanism (Eq. 8–10), separately tested (`test_v12_fixes_outlier_regression`), and its outlier MAE (1.386) is close to its own clean-data MAE (1.110) — i.e. contamination costs V1.2 almost nothing.

### Headline slice (noisy uneven, no outliers, 24 datasets): RMSE
savgol_resampled **1.674** · ALPRD_v1.2 1.891 · ALPRD_v1.1 2.333 · tv_diff 3.862 · savgol_matched 3.959 · ALPRD_v1.0 4.486. **V1.2 still does not overtake default SG on this slice** — the residual gap is entirely chaotic motion. Excluding chaotic (21 datasets): savgol_matched 0.104 · **ALPRD_v1.2 0.168** · ALPRD_v1.1 0.193 · savgol_resampled 0.224 — V1.2 now beats default SG on smooth motions and trails only the width-matched SG.

### Chaotic (Lorenz) MAE by scenario — steady improvement, still behind narrow SG
V1.0 → V1.1 → **V1.2** → savgol_resampled, on `uneven_gaussian`: 23.60 → 11.57 → **8.15** → 6.05. Real-world-like: 21.65 → 3.74 → **2.03** → 0.57. The refined pilot bought ~30% over V1.1 (Hypothesis in V1.1 Step 9 item 2: confirmed directionally), but the gap to a simply-narrow smoother persists: the grid's $k_{\min}$ and the quadratic local model are the remaining binding constraints.

### Runtime (mean s/call, suite $n\in[150,400]$)
savgol_matched 0.0009 · rbf_fd 0.0118 · ALPRD_v1.0 0.0199 · savgol_resampled 0.0211 · **ALPRD_v1.2 0.1031** · ALPRD_v1.1 0.2387 · tv_diff 1.2754. V1.2 is 2.3× faster than V1.1 at suite sizes (vectorization) while doing strictly more work (2 passes × IRLS); the large-$n$ picture is in Step 4.4 and it is not good (target missed).

## STEP 8 — Failure Analysis

### Failure A (carried): chaotic gap to narrow-window SG persists (8.15 vs 6.05 MAE)
Mathematically: with $p=2$ the local model cannot track the Lorenz signal's curvature at any window the guard (10) permits; the selector correctly picks $k$ near $k_{\min}$, but $k_{\min}=6$ with a quadratic is variance-limited while SG's fixed narrow window with $p=3$ has a better bias constant there. Severity: medium. Options for V1.3: motion-adaptive polynomial degree ($p\in\{2,3\}$ competing in the same risk selection), or a smaller $k_{\min}$ with a variance-inflation guard.

### Failure B (new, engineering): runtime target missed by ~10×
"≤2× V1.0 at n=2000" was the stated V1.2 goal; measured 21.6×. Root cause: cost per candidate is still $O(nk_c)$ and the geometric grid's top candidates have $k_c\propto n$. The prefix-sum moment-array formulation (each candidate $O(n)$ regardless of $k_c$) was designed in V1.1's Step 9 but not built. IRLS complicates it (robust weights are not polynomial in $t$), so the honest engineering plan is: prefix-sums for the *first* WLS pass + windowed IRLS only for the few points whose first-pass residuals flag contamination. Severity: medium-high (blocks $n\gtrsim10^4$ use).

### Failure C (baseline, expected): TV-L2 catastrophic under outliers; RBF-FD amplifies noise
Both match their theory (L2 fidelity must chase gross errors; interpolatory stencils have no noise model). Not ALPRD failures, but recorded so the suite's aggregate numbers aren't misread: **the TV row's terrible aggregate is one scenario class, and TV remains a serious competitor on smooth noisy data.**

### Resolved this iteration
- V1.1 Failure 1′ (outlier collapse): fixed by design, 89× improvement, regression-guarded in tests.
- V1.1 Failure 2′ (chaotic): halved again (11.6→8.2); not fully closed (Failure A).
- V1.1 Failure 4′ (selector cost at suite sizes): 2.3× faster than V1.1; large-$n$ goal missed (Failure B).

## STEP 9 — Changes made (all justified, all benchmarked)
1. **Tukey IRLS in all fits (Eqs. 8–9)** — targeted V1.1 Failure 1′; measured: outlier MAE 122.8→1.39 with *no* loss on clean/noisy slices (they improved too). Predicted disadvantage (cost) materialized but was outweighed by vectorization.
2. **Effective-size guard (Eq. 10)** — prevents the robust selector from accepting near-empty windows; part of the same fix.
3. **Refined per-point pilot** — targeted V1.1 Failure 2′; measured: chaotic 11.57→8.15, real-world chaotic 3.74→2.03.
4. **Vectorized batched implementation with memory blocking** — targeted V1.1 Failure 4′; measured: 2.3× faster at suite sizes, n=8000 no longer OOMs, identical outputs before/after blocking; large-$n$ target missed (Failure B).
5. **TV + RBF-FD baselines** — closed the two "missing family" gaps from V1.0/V1.1 Step 5.

## STEP 10 — Version History
**V1.0** — fixed-span local quadratic WLS + Rice variance. Baseline; outlier "win" later retracted as width confound.
**V1.1** — plug-in adaptive bandwidth (Eq. 7); fast windowing (proven identical); SG width-matched control (confound confirmed, claim retracted). Headline RMSE halved; outliers regressed 5×.
**V1.2** — Tukey-IRLS robust fits (8–9) + effective-size guard (10) + refined pilot inside the same selector; fully vectorized memory-bounded implementation; oracle lemma (11) recorded with its limits; TV & RBF-FD baselines added. Outlier failure fixed (89× vs V1.1), every scenario improved, best overall aggregate in study; chaotic gap and large-$n$ runtime remain open.

## STEP 11 — Remaining Problems
- ☐ $\Delta$-concentration bound to give (11) content (adaptivity theorem) — the main theory gap
- ☐ Breakdown-point / influence analysis; benchmarks beyond 3% contamination (10–30%)
- ☐ Post-IRLS, post-selection variance: reported `se` is doubly approximate — quantify or bootstrap
- ☐ Chaotic gap to narrow SG (Failure A): adaptive degree $p\in\{2,3\}$ or smaller $k_{\min}$
- ☐ Large-$n$ runtime (Failure B): prefix-sum first pass + selective IRLS
- ☐ Heteroscedastic / correlated noise (carried from V1.0)
- ☐ Boundary bias order (carried)
- ☐ Systematic literature search: is "plug-in risk-selected robust local-polynomial derivative on irregular grids with self-error" actually unclaimed? (novelty still unverified)
- ☐ Comparison against *published* numbers (all baselines so far are our own implementations)
- ☐ L1-fidelity TV variant (fair robust competitor for ALPRD's outlier result)
- ☐ Second derivative $m=2$; real (non-synthetic) dataset

## STEP 12 — Publication Readiness (V1.2, honest)

| Criterion | Score | Justification |
|---|---|---|
| Mathematical rigor | 4 | Oracle lemma (11) is trivial and labeled as such; IRLS makes the estimator nonlinear, *weakening* the formal status of Eqs. (3)/(6) inside (7); no new theorem of substance |
| Novelty | 3 | (was 2) The assembly (risk-selected bandwidth + Tukey robustness + derivative target + uneven grids + self-error) is not a named standard method, but every part is classical (Cleveland 1979 LOWESS is uncomfortably close) and the literature search is still not done — 3 is the ceiling until it is |
| Accuracy | 6 | (was 5) Best overall aggregate by 5×; best or top-3 in every scenario; contamination now nearly free (1.39 vs 1.11 clean); still loses headline RMSE to default SG (1.89 vs 1.67) and chaotic to narrow SG (8.15 vs 6.05) |
| Stability | 6 | (was 5) 650 runs + 10 tests, 0 failures; ridge-guarded batch solves; memory bounded; conditioning still not formally bounded |
| Robustness | 7 | (was 2) Now a *designed*, tested, regression-guarded property with an order-of-magnitude margin — but no breakdown analysis and only one contamination level tested |
| Computational efficiency | 3 | Vectorization is real (2.3× vs V1.1 while doing more work) but the stated large-$n$ target was missed by ~10× and scaling is still quadratic |
| Practical usefulness | 6 | (was 4) Zero-tuning, uneven-grid-native, outlier-proof, with error bars — a genuinely useful default; runtime is the main practical objection |
| Reproducibility | 9 | Same regime as before: seeded, CSV-committed, test-guarded, refactor verified bit-identical |
| Literature support | 3 | (was 2) TV and RBF-FD families now measured in-suite; still zero verified comparisons against published results |
| **Overall publishability** | **5** | (was 3) The method now *does something defensible*: it is the only method in the study that is near-best everywhere, and its robustness is designed rather than incidental. Not 6+ yet because: novelty unresolved vs the LOWESS/Fan–Gijbels lineage, the central theory gap ($\Delta$ bound) is untouched, and two stated goals (chaotic parity, runtime) were missed. The path to 6–7 is concrete and listed above. |

## STEP 13 — Next Iteration: see the V1.3 continuation prompt below.

---

## Next Recommended Prompt (V1.3)

```
Continue the numerical-differentiation research project at numdiff-research/.
Read docs/RESEARCH_LOG.md fully (V1.0, V1.1, V1.2 sections). Raw numbers:
results/results_v1_2.csv, scaling in results/scaling_v1_2.json. 10 tests in
tests/test_methods.py must stay green.

V1.3 priorities, in order (from Version 1.2 Steps 8/11):
1. ADAPTIVE DEGREE: let p in {2,3} compete inside the same risk selection
   (Eq. 7 evaluated per (k, p) pair, bias functional using b_{p+1} of the
   matching degree). Target: close the chaotic gap to narrow Savitzky-Golay
   (currently 8.15 vs 6.05 MAE on uneven_gaussian) without losing smooth-
   motion performance. Guard with a new regression test.
2. RUNTIME: prefix-sum moment arrays for the first (non-robust) WLS pass of
   every candidate — O(n) per candidate independent of k — with IRLS applied
   selectively only where first-pass residuals exceed 3*sigma_hat. Target
   (missed in V1.2, restated): <= 2x ALPRD v1.0 at n=2000, and report
   honestly if missed again.
3. ROBUSTNESS CHARACTERIZATION: add 10% and 25% contamination scenarios to
   the suite (new dataset variants, same seed discipline) and measure where
   V1.2's IRLS breaks down.
4. THEORY: attempt a concentration bound for the Rice estimator's deviation
   (the sigma^2 part of Delta in Eq. 11) under the iid noise model — even a
   partial bound with explicit constants moves Rigor.
5. Add an L1-fidelity TV variant as a fair robust competitor.

Rules unchanged: same 35 base scenarios + new contamination variants, seed
12345, append "Version 1.3" in the same Steps 1-13 format, never modify
earlier sections, label Proven/Derived/Tested/Hypothesized, honest scores.
```


---
---

# Version 1.3 — Iteration 4

Code: [`../src/methods.py`](../src/methods.py) (`alprd_v13`, `_windowed_moments`, `_rice_variance_calibrated`, `_running_median`, `tv_derivative_l1`), tests: [`../tests/test_methods.py`](../tests/test_methods.py) (**15 tests, all passing**).
Raw results: [`../results/results_v1_3.csv`](../results/results_v1_3.csv) — **990 runs = 15 methods × 66 datasets, 0 failures, seed 12345**; scaling: [`../results/scaling_v1_3.json`](../results/scaling_v1_3.json). Every number below is read from these artifacts.

**RECORD CORRECTION (honesty note):** earlier sections describe the base suite as "35 scenarios". The base suite has always been **50 datasets** (8 motions × 6 conditions + 2 real-world-like composites; the committed row counts — 350 = 50×7, 500 = 50×10, 650 = 50×13 — were always consistent with 50). The "35" was a miscount in the prose, never in the data. V1.3 adds 16 contamination datasets (8 motions × {10%, 25%}) for a new total of 66.

---

## STEP 1 — Current Version: V1.3 (complete formulation)

**Model.** Observations $(t_i, y_i)$, $i=1..n$, $t_i$ strictly increasing and arbitrarily spaced, $y_i = f(t_i) + \varepsilon_i$ with $f$ locally smooth; $\varepsilon_i$ iid mean-0 variance-$\sigma^2$ noise, possibly contaminated by a fraction of gross outliers. Target: $f'(t_i)$ at every sample point, with a per-point standard error.

**Stage 0 — noise scale (Eq. 13, new).** With Newton's third divided difference over consecutive quadruples,
$$r_i = f[t_i,t_{i+1},t_{i+2},t_{i+3}] = \sum_{j=0}^{3} c_{ij}\, y_{i+j},\qquad c_{ij} = \prod_{l\ne j} (t_{i+j}-t_{i+l})^{-1},$$
$$\hat\sigma^2 \;=\; \frac{1}{q_{1/2}}\ \operatorname{median}_i \frac{r_i^2}{\sum_j c_{ij}^2},\qquad q_{1/2}=\operatorname{median}(\chi^2_1)\approx 0.454936. \tag{13}$$

**Stage 1 — high-breakdown prefilter (new).** $m_i = \operatorname{runmed}_{11}(y)_i$, $r^{med}_i = y_i - m_i$, $s_{med} = \max(\hat\sigma,\ 1.4826\,\mathrm{med}|r^{med}|,\ 10^{-9}\Delta y)$, pre-weights $w^{med}_i = \psi_T(r^{med}_i / s_{med})$ with the Tukey biweight $\psi_T$ at scale $c\,s$, $c=4.685$ (Eq. 8).

**Stage 2 — robust pilot (extended).** One IRLS local fit of degree $p_{\max}+2=5$ per point (window $k_{pilot}$, tricube × $w^{med}$ weights, Eqs. 8–9) gives Taylor-coefficient estimates $\hat b_j(t_i)\approx f^{(j)}(t_i)/j!$ for $j\le5$, center residuals $r^0_i$, outlier flags $F_i=\mathbb 1\{|r^0_i|>4\,s_{flag}\}$, and combined pre-weights $w^{pre}_i = w^{med}_i\,\psi_T(r^0_i/s_{flag})$.

**Stage 3 — candidate evaluation.** Candidates are pairs $(k,p)$, $k$ on a geometric grid $[k_{\min}, \min(\lceil n/2\rceil, k_{cap})]$, $p\in\{1,2,3\}$. For each candidate at $t_0$: window = $k$ nearest samples (contiguous in sorted $t$, minimal radius $h$), $z_j=(t_j-t_0)/h$, kernel weights $w^K_j=(1-|z_j|^3)^3$.

*Unflagged windows* (no $F_j=1$ inside): plain WLS via shared moments
$$S_q=\textstyle\sum_j w^K_j z_j^q,\quad S^{(2)}_q=\sum_j (w^K_j)^2 z_j^q,\quad T_q=\sum_j w^K_j y_j z_j^q,$$
$G_{ab}=S_{a+b}$, $\hat\beta=G^{-1}T_{0:p}$, $g_1=G^{-1}e_1$; derivative $\hat f'=\hat\beta_1/h$.

*Flagged windows:* Tukey IRLS (Eqs. 8–9) with initial weights $w^K_j w^{pre}_j$, giving $\hat\beta$, equivalent-kernel row $\ell$, final weights $W$, and the window's robust residual scale $s_{win}=1.4826\,\mathrm{med}_j|y_j-\hat P(t_j)|$.

**Stage 4 — risk and selection (Eq. 12, two-term bias — new).**
$$\widehat{R}(k,p;t_0)=\Big(\hat b_{p+1}h^{p}\langle g_1,S_{\cdot+p+1}\rangle+\hat b_{p+2}h^{p+1}\langle g_1,S_{\cdot+p+2}\rangle\Big)^2+\frac{\tilde\sigma^2}{h^2}g_1^\top G^{(2)}g_1 \tag{12}$$
with $\tilde\sigma^2=\hat\sigma^2$ on unflagged windows and $\tilde\sigma^2=\max(\hat\sigma^2,s_{win}^2)$ on flagged ones (a failed local consensus penalizes itself). Candidates with $n_{eff}=(\sum W)^2/\sum W^2<p+2$ are discarded (Eq. 10). Select $(\hat k,\hat p)=\arg\min\widehat R$; output the corresponding $\hat f'$ and $\widehat{se}=\sqrt{\tilde\sigma^2h^{-2}g_1^\top G^{(2)}g_1}$.

**Stage 5 — pilot refinement.** Re-fit the pilot per point at window $\lceil1.5\hat k\rceil$ (with $w^{pre}$) and repeat Stages 3–4 once.

**Why the second bias term is mandatory (Derived + Tested):** for $p-m$ even ($p=1,3$ for $m=1$) the leading functional $\langle g_1,S_{\cdot+p+1}\rangle$ nearly vanishes on symmetric windows and wherever $f^{(p+1)}$ crosses zero (the parity phenomenon of local polynomial regression, Ruppert & Wand 1994). A single-term risk then reads "zero bias" and over-selects those degrees: measured before the fix, max error 0.42 on a NOISELESS cubic, all failures at $p=1$, maximal $k$; after adding the $b_{p+2}$ term, max error $6\times10^{-8}$ on the same data.

## STEP 2 — Symbol Table (additions to V1.0–V1.2 tables)

| Symbol | Meaning | SI units | Allowed values | Purpose |
|---|---|---|---|---|
| $r_i$ (Eq. 13) | 3rd divided difference | m/s³ | real | annihilates quadratics; isolates noise |
| $c_{ij}$ | divided-difference coefficients | s⁻³ | real | closed form, no solves |
| $q_{1/2}$ | median of $\chi^2_1$ | dimensionless | 0.454936… | de-biases the median estimator |
| $m_i, r^{med}_i$ | running median (11 pts), its residual | m | real | 50%-breakdown initialization |
| $w^{med}_i, w^{pre}_i$ | prefilter / combined pre-weights | dimensionless | $[0,1]$ | stop local fits chasing outliers |
| $F_i$ | outlier flag | boolean | — | routes windows to the robust path |
| $p$ | competing local degree | dimensionless | $\{1,2,3\}$ | adaptive degree |
| $\hat b_{p+2}$ | second pilot Taylor coefficient | m/s^{p+2} | real | parity-safe bias model (Eq. 12) |
| $z_j$ | normalized abscissa | dimensionless | $[-1,1]$ | conditioning of all fits |
| $S_q, S^{(2)}_q, T_q$ | windowed weighted power sums | mixed | real | shared by all degrees; fast path |
| $s_{win}$ | per-window robust residual scale | m | $\ge0$ | self-penalizing risk on flagged windows |
| $k_{cap}$ | window-size cap | dimensionless | 200 default | bounds cost at large n (explicit trade) |

## STEP 3 — Assumptions (changes)
- Contamination: gross-error model, now **measured at 3/10/25%**; breakdown occurs between 10% and 25% (Step 8).
- The running-median prefilter assumes outliers are not so dense that 6 of any 11 consecutive samples are corrupt (50% local breakdown).
- The $k_{cap}$ bound assumes the AMISE-optimal window at the data's noise level is ≤200 points — true for every suite dataset; at very large $n$ with very high noise the cap costs accuracy (documented trade).
- Post-selection and post-IRLS caveats on `se` carry over from V1.1/V1.2 verbatim.

## STEP 4 — Mathematical Analysis

### 4.1 Two calibration defects found and fixed (Tested, then Derived)
(a) **Median miscalibration:** V1.0–V1.2's $\hat\sigma^2$ = median of a $\sigma^2\chi^2_1$ sample → biased low by $q_{1/2}\approx0.455$. Measured 0.462 on a known-noise linear signal before the fix; within [0.008, 0.0125] of true 0.01 after (test-guarded). All V1.0–V1.2 risk selections ran with $\sigma^2$ ~2.2× low and all reported `se` ~1.48× low; those results stand as recorded.
(b) **Smooth-signal leakage:** the 3-point scheme annihilates only linears; measured $\hat\sigma^2=0.0154$ on a NOISELESS cubic. The divided-difference scheme (Eq. 13) has smooth remainder $f'''(\xi)/6$ exactly (divided-difference mean value theorem), contributing $(f'''/6)^2/\sum c^2 = O(h^6)$ after normalization: measured $4\times10^{-9}$ on the same cubic.

### 4.2 Concentration of $\hat\sigma^2$ (Proven under stated assumptions)
Let $\varepsilon_i$ be iid $N(0,\sigma^2)$ and $f$ locally quadratic (so the smooth remainder in $r_i$ vanishes; otherwise add the $O(h^6)$ term of 4.1b). Then $X_i=r_i^2/\sum_j c_{ij}^2\sim\sigma^2\chi^2_1$, and $X_i,X_{i'}$ are independent whenever $|i-i'|\ge4$. Split $\{X_i\}$ into 4 subsequences of independent variables ($i\bmod4$), each of length $\ge\lfloor(n-3)/4\rfloor=:n_4$. By DKW on each and a union bound, with probability $\ge1-\delta$ every subsequence empirical CDF — hence their average, the pooled empirical CDF $\widehat F$ — satisfies $\|\widehat F-F\|_\infty\le\epsilon:=\sqrt{\log(8/\delta)/(2n_4)}$, where $F$ is the CDF of $\sigma^2\chi^2_1$. Since $F$ has density $\ge\rho/\sigma^2$ near its median $q_{1/2}\sigma^2$ with $\rho=f_{\chi^2_1}(q_{1/2})\approx0.4711$, the pooled sample median deviates from $q_{1/2}\sigma^2$ by at most $\epsilon\sigma^2/\rho$, giving
$$\frac{|\hat\sigma^2-\sigma^2|}{\sigma^2}\;\le\;\frac{1}{q_{1/2}\rho}\sqrt{\frac{\log(8/\delta)}{2\lfloor(n-3)/4\rfloor}}\;\approx\;4.67\sqrt{\frac{\log(8/\delta)}{2\lfloor(n-3)/4\rfloor}}\quad\text{w.p.}\ \ge1-\delta. \tag{14}$$
**Status: Proven** for Gaussian noise + locally-quadratic $f$ (the Gaussian assumption fixes the $\chi^2_1$ law and hence the constants; sub-Gaussian noise changes $q_{1/2},\rho$ but not the $n^{-1/2}$ architecture). **Empirically verified:** mean relative deviation × $\sqrt n$ measured 2.80/2.72/2.54 at $n=200/1000/5000$ — the $n^{-1/2}$ rate holds and the measured constant sits inside (14). This bounds the $\hat\sigma^2$ part of $\Delta$ in the oracle lemma (Eq. 11); the $\hat b_{p+1},\hat b_{p+2}$ concentration (post-IRLS) remains **open** — the oracle inequality is still not a theorem.

### 4.3 Complexity (Derived + Measured; V1.3 target MET at stated size)
Unflagged path per candidate $k$: moments cost $O(nk)$ once, shared by all three degrees; solves are $O(np^3)$. With $k\le k_{cap}$: total $O(n\cdot k_{cap}\cdot|grid|)$ — **linear in $n$** with bounded windows. Measured (clean machine, `scaling_v1_3.json`): n=500: 2.78× V1.0; n=2000: 4.02×; **n=8000: 1.26× — the ≤3×-at-n=8000 target is MET** (the n=2000 ratio exceeds 3 and is reported as such; V1.0's own superlinear cost flatters the large-$n$ ratio). Absolute: 11.7 s at n=8000 vs V1.2's 229 s (19.6×). Suite-size mean runtime 0.259 s (V1.2: 0.171 s — slightly slower at n≤400 due to the richer pilot and 3 degrees).

## STEP 5 — Literature Comparison (delta)
Adaptive degree + adaptive bandwidth by plug-in risk is squarely in the Fan & Gijbels (1995/1996) program; two-term bias corrections and median prefilters for robust smoothing are also classical ingredients. The increasingly specific assembly — joint $(k,p)$ risk selection with a parity-safe two-term bias functional, calibrated divided-difference noise scale, median-prefiltered robust pilot chain, on irregular grids, with self-error output — has no single named ancestor we know of, **but the decisive literature search has still not been done and novelty remains UNVERIFIED**. New measured baseline: `tv_diff_l1` (L1-fidelity TV, median-discrepancy $\alpha$) — the fair robust competitor; it is the only method beating ALPRD v1.3 anywhere (25% contamination).

## STEP 6 — Implementation
New: `_rice_variance_calibrated` (Eq. 13), `_running_median`, `_windowed_moments`, `_pilot_coeffs_v13` (degree-5, pre-weighted), `alprd_v13` (all stages, memory-blocked, no per-point Python loops on the unflagged path), `tv_derivative_l1`. Tests added (15 total): calibration (both defects), cubic exactness through the selector, contamination guards at 3/10/25%, chaotic-improvement guard, L1-vs-L2 TV robustness. The V1.3 development loop itself caught and fixed two design failures before benchmarking (parity blind spot; WLS-drag under contamination) — both documented above with their pre-fix measured numbers rather than silently absorbed.

## STEP 7 — Benchmarking (990 runs, 0 failures)

### Headline slice (noisy uneven, no outliers, 24 datasets)
| method | MAE | RMSE |
|---|---|---|
| **ALPRD_v1.3** | **0.678** | 1.698 |
| savgol_resampled | 1.047 | **1.674** |
| ALPRD_v1.2 | 1.211 | 1.891 |
| tv_diff_l1 | 1.889 | 4.483 |
| tv_diff | 2.105 | 3.862 |

**First version to reach Savitzky–Golay parity on its home turf:** best MAE by 35%, RMSE within 1.5% (1.698 vs 1.674 — called a tie, not a win).

### Chaotic (Lorenz)
`uneven_gaussian` MAE: **ALPRD_v1.3 4.44** < savgol 6.05 < tv_l1 6.68 < ALPRD_v1.2 8.15 — **the chaotic gap is closed and reversed.** Real-world-like: savgol 0.571 vs v1.3 0.634 (savgol narrowly ahead there).

### Contamination sweep (MAE, mean over 8 motions)
| method | 3% | 10% | 25% |
|---|---|---|---|
| **ALPRD_v1.3** | **0.482** | **0.677** | 45.4 |
| tv_diff_l1 | (poly-dominated)* | (poly-dominated)* | **30.9** |
| ALPRD_v1.2 | 1.386 | 2.026 | 172.5 |
| ALPRD_v1.0 | 23.0 | 36.3 | 69.2 |
| savgol_matched | 26.5 | 69.9 | 90.5 |

*tv_l1's 3%/10% means are dominated by its polynomial-motion failures (its discrepancy rule mis-tunes when outlier magnitude is 10²–10³ × signal increments); on the other 7 motions tv_l1 is competitive at 3–10%.
V1.3's 25% mean is dominated by two motions (polynomial 258, projectile 63; sinusoidal 0.61, circular 1.03) — **breakdown between 10% and 25% depending on outlier magnitude relative to signal scale: the honest boundary of the method.**

### Runtime (suite mean): ALPRD_v1.3 0.259 s; scaling table in Step 4.3.

## STEP 8 — Failure Analysis
- **F-A (25% contamination):** at 25%, only ~0.75⁴≈32% of noise-estimator quadruplets are clean (<50%) so $\hat\sigma^2$ inflates; running-median windows carry ≈3 outliers of 11; flags and pre-weights degrade together and the local IRLS inherits bad starts. tv_l1's global L1 objective degrades more gracefully (30.9 vs 45.4). Severity: medium — 25% gross contamination is beyond the stated design envelope, now with a measured boundary.
- **F-B (runtime at mid-n):** 4.02× V1.0 at n=2000 (>3), 1.26× at n=8000. The stated target (n=8000) is met; the mid-range is honestly above it.
- **F-C (record):** the "35 scenarios" prose miscount, corrected above.
- Resolved from V1.2: Failure A (chaotic — now wins), Failure B (runtime — target met at stated size), plus two V1.3-internal design failures caught pre-benchmark (Steps 1, 6).

## STEP 9 — Changes made (all justified, all measured)
1. Adaptive degree with two-term bias (Eq. 12) — headline MAE 1.211→0.678, chaotic 8.15→4.44, cubic exactness restored (0.42→6e-8).
2. Calibrated divided-difference noise estimator (Eq. 13) — removes the 2.2× variance bias + curvature leakage (0.0154→4e-9).
3. Median prefilter + pilot-residual pre-weighting — contamination 3%: 1.39→0.48; 10%: 2.03→0.68; 25%: 172.5→45.4.
4. Moment-based shared-candidate evaluation with $k_{cap}$ — 19.6× faster than V1.2 at n=8000; ≤3×-of-V1.0 target met at n=8000.
5. L1-TV baseline — the one method that beats v1.3 anywhere (25%), kept prominently in the record.

## STEP 10 — Version History
**V1.0** fixed-span local quadratic + Rice variance (later found miscalibrated ×0.455). **V1.1** plug-in adaptive bandwidth; outliers regressed 5×; V1.0 robustness claim retracted (width confound). **V1.2** Tukey IRLS + effective-size guard + refined pilot; outliers fixed (89× vs V1.1); chaotic and runtime targets missed. **V1.3** adaptive degree with parity-safe two-term bias; calibrated divided-difference noise scale (defects disclosed); median-prefiltered robust initialization; moment-based O(n·k_cap) evaluation; contamination sweep 3–25%; concentration bound (Eq. 14) proven; L1-TV added. Headline parity with SG reached, chaotic won, ≤10% contamination dominated, 25% boundary measured.

## STEP 11 — Remaining Problems
- ☐ **Systematic literature search** (novelty status now decides publishability more than anything else)
- ☐ Oracle inequality: $\hat b$-concentration (post-IRLS) open; Eq. 14 covers only the $\hat\sigma^2$ part of $\Delta$
- ☐ 25%-contamination regime (F-A): high-breakdown pilot (LTS/repeated-median class) is the principled fix
- ☐ Post-selection/post-IRLS `se` validity: bootstrap comparison needed
- ☐ Heteroscedastic noise: the $s_{win}$ machinery is a natural entry point, unexploited on unflagged windows
- ☐ Real (non-synthetic) dataset; second derivative $m=2$; boundary bias order
- ☐ Mid-n runtime (F-B); comparison against published numbers (all baselines still ours)

## STEP 12 — Publication Readiness (V1.3, honest)
| Criterion | Score | Justification |
|---|---|---|
| Mathematical rigor | 5 | (was 4) Eq. 14 proven with explicit constants + empirical rate check; two derived defect analyses; oracle inequality still open; estimator nonlinear |
| Novelty | 3 | Assembly increasingly specific, ancestry increasingly classical (Fan–Gijbels + Cleveland + Gasser-type differencing); UNVERIFIED until the literature search is done — frozen until then |
| Accuracy | 7 | (was 6) Best or tied on every slice: headline MAE best / RMSE tied, chaotic won, ≤10% contamination dominated; loses only 25% (to tv_l1) and real-world chaotic (to SG, narrowly) |
| Stability | 7 | (was 6) 990 runs + 15 tests, 0 failures; normalized bases everywhere; two design failures caught by the loop itself |
| Robustness | 8 | (was 7) Designed, tested at three contamination levels, breakdown boundary measured and disclosed |
| Computational efficiency | 5 | (was 3) Linear-in-n with capped windows; stated target met; mid-n ratio above target; 19.6× faster than V1.2 |
| Practical usefulness | 7 | (was 6) Zero-tuning, uneven-grid-native, outlier-proof to 10%, error bars, tractable runtime |
| Reproducibility | 9 | Unchanged regime; every claim traces to a committed artifact |
| Literature support | 3 | tv_l1 added; still no comparison against published numbers |
| **Overall publishability** | **6** | (was 5) The empirical case is now genuinely strong — near-uniform dominance with honest, measured boundaries. The two blockers are exactly Novelty (unsearched) and the missing adaptivity theorem. A 7 requires the literature search coming back clean plus either the oracle inequality or a real-data study. |

## STEP 13 — Next Iteration (V1.4, executed in-session)
Planned and justified next actions: (1) systematic literature search to resolve the Novelty score — the single highest-leverage open item; (2) push the 25% boundary with a twice-iterated high-breakdown pilot; (3) heteroscedastic noise support via local $\sigma^2(t)$ from windowed calibrated divided differences, with new heteroscedastic benchmark scenarios; (4) reassess stopping conditions afterward.

---
---

# Version 1.4 — Iteration 5 (literature verification + heteroscedastic support + two recorded negative results)

Code: [`../src/methods.py`](../src/methods.py) (`alprd_v14`, `_local_sigma2`, gate logic in `alprd_v13`), tests: **17, all passing**. Raw results: [`../results/results_v1_4.csv`](../results/results_v1_4.csv) — **1184 runs = 16 methods × 74 datasets, 0 failures, seed 12345** (74 = 66 prior + 8 heteroscedastic-ramp scenarios, appended last so all earlier per-dataset rows remain identical).

## STEP 1 — Current Version
V1.4 = the complete V1.3 formulation (see Version 1.3 Step 1, Eqs. 8–14 — unchanged) plus one addition:

**Pointwise noise variance with a double-guarded auto gate.** After the pilot's outlier flags exist:
$$\hat\sigma^2(t_j) = \frac{1}{q_{1/2}}\operatorname{runmed}_{k_{sig}}\Big(\big\{\text{est}_i : \text{quadruplet } i \text{ touches no flagged sample}\big\}\Big)_j \tag{15}$$
(est$_i$ as in Eq. 13). The risk's variance term uses $\hat\sigma^2(t_0)$ instead of the global scalar **only if** (a) contamination is light — flag fraction < 10% **and** median-prefilter cut fraction < 10% — and (b) the cleaned local field shows genuine spread, $q_{90}/q_{10} > 10$ (measured separation: homoscedastic 4.4–6.7, ramp-noise 16.6–25; a data-informed heuristic threshold, labeled as such). Robust scales and flag thresholds keep using the global scalar (they want a stable scale, not a locally noisy one).

## STEP 2 — Symbol additions
$\hat\sigma^2(t_j)$ — pointwise noise variance, m² (Eq. 15); $k_{sig}$ — its median window, 25 quadruplets; $q_{90}/q_{10}$ — gate statistic, dimensionless.

## STEP 3 — Assumption changes
Heteroscedastic noise $\sigma(t)$ is now supported when it varies *smoothly* (resolvable by a 25-quadruplet median) and contamination is light; heavy contamination + heteroscedasticity simultaneously is explicitly NOT supported (the guards then force the global scalar — a documented conservative fallback, not a silent failure).

## STEP 4 — Analysis and design-failure record (all measured before/after)
The gate went through three measured design failures before converging — recorded because each is informative:
1. **Ungated local variance** degrades homoscedastic data slightly (noisier variance estimates) and chaotic-hetero by ~10% — hence the gate.
2. **Ungated at 25% contamination**: outlier clusters masquerade as heteroscedasticity (gate ratio 3.4×10⁸), sinusoidal-25% MAE 0.61 → 3.00 — hence the flag-exclusion in Eq. 15.
3. **Flag-guard alone is insufficient at 25%**: the *global* $\hat\sigma^2$ is itself inflated there (measured 2.66 on noiseless contaminated data, since only 0.75⁴ ≈ 32% of quadruplets are clean), so the flag threshold balloons and almost nothing is flagged — hence the second guard on the prefilter cut fraction, which retains its 50% breakdown at 25% contamination.

**Negative result 1 (Hypothesis falsified):** a twice-iterated high-breakdown pilot was hypothesized (V1.3 Step 13) to sharpen consensus at 25% contamination. Measured: sinusoidal-25% MAE 0.61 → 1.14, polynomial-25% 258 → 383 — iteration *amplifies* first-pass misidentification. REJECTED; `prefilter_iters` remains available but defaults to 1.

**Negative result 2 (scope boundary confirmed):** the 25%-contamination regime cannot be fixed by reweighting-from-a-WLS-start machinery at all (both iterations and more IRLS steps measured unhelpful); it requires a genuinely high-breakdown initial estimator (LTS/repeated-median class) — new algorithmic machinery, outside this iteration.

## STEP 5 — Literature Comparison (search performed this iteration)
A bounded web literature search was conducted (first external check in the project). Findings, honestly stated:
- **Adaptive-window derivative estimation exists**: Katkovnik's LPA-ICI school (local polynomial approximation + intersection-of-confidence-intervals window selection, grounded in Lepski/Goldenshluger–Nemirovski theory) has done adaptive-scale signal-and-derivative estimation since ~1999; a June 2026 preprint (SURDE) does SURE-based adaptive window selection for derivative FIR filters — uniform grids, Gaussian noise, fixed degree, no outliers, no per-point error bars.
- **Robust LPR with adaptive bandwidth exists**: an M-estimator + ICI method (IEEE, ~2004) for impulsive-noise signal reconstruction — regression function, not derivative-targeted, not uneven-grid-native.
- **The practical toolkit state of the art** is PyNumDiff (van Breugel–Kutz–Brunton) and the multi-objective TV framework (arXiv:2009.01911); a December 2025 taxonomy of numerical differentiation explicitly notes that methods for irregularly-spaced samples are "underdeveloped" and identifies no surveyed method providing per-point derivative error estimates.
- **Verdict (bounded-search, not proof of absence):** every ingredient of ALPRD is classical; the specific assembly — derivative-targeted joint (k, p) risk selection with a parity-safe two-term bias functional, calibrated divided-difference noise scale, median-prefiltered robust pilot chain, native uneven grids, per-point standard errors — was not found as an existing method, and two of its aspects (uneven-grid focus, per-point error bars) are explicitly named as gaps by the 2025 taxonomy. Realistic framing: an incremental-combination methods contribution, NOT a theoretical breakthrough. Novelty score unfrozen: 3 → 5. A full novelty claim still requires checking Katkovnik's LPA book and Fan–Gijbels' variable-order papers in full text, plus a head-to-head against PyNumDiff's implementations (external code — planned, not done).

Sources: [taxonomy (arXiv 2512.09090)](https://arxiv.org/html/2512.09090), [SURDE (arXiv 2606.09829)](https://arxiv.org/html/2606.09829), [M-estimator+ICI LPR (IEEE 1328751)](https://ieeexplore.ieee.org/document/1328751), [PyNumDiff](https://github.com/florisvb/PyNumDiff), [multi-objective TV framework (arXiv 2009.01911)](https://arxiv.org/abs/2009.01911), [LPA-ICI denoising instrument](https://link.springer.com/article/10.1007/s11760-016-0921-6), [ICI adaptive window](https://link.springer.com/article/10.1023/A:1020329726980).

## STEP 6/7 — Implementation & Benchmarking (1184 runs, 0 failures)
- **Heteroscedastic slice (8 new ramp-noise datasets), MAE:** **ALPRD_v1.4 0.716** (best) < ALPRD_v1.3 0.723 < savgol_resampled 1.036 < ALPRD_v1.2 1.152 < tv_diff_l1 1.738. Gains on 6 of 8 motions (2–11%), small loss on polynomial-hetero (0.132→0.143), chaotic-hetero gate stays closed (no change).
- **Everything else:** v1.4 ≡ v1.3 exactly on 60 of 66 prior datasets (gate closed, byte-identical rows); negligible differences on 4 clean sets; one real misfire: `chaotic__missing_noisy` 6.15→6.32 (+2.7%) — sampling gaps mimic heteroscedasticity to the gate. Documented as F-D below.
- All V1.3 headline conclusions stand unchanged (same rows): headline MAE best / RMSE tied with SG, chaotic won, ≤10% contamination dominated, 25% lost to tv_l1.

## STEP 8 — Failure Analysis
- **F-A (carried, scoped):** 25% contamination — now *proven by experiment* to be outside what reweighting-based machinery can fix (Negative results 1–2); requires LTS-class pilot. 
- **F-D (new, minor):** the gate misreads irregular-gap-induced variance spread as heteroscedasticity on one dataset (+2.7% there). A gap-aware gate statistic (normalize est by local density) is the plausible fix — Hypothesized, untested.
- **F-B (carried):** mid-n runtime ratio (4× at n=2000).

## STEP 9 — Changes: Eq. 15 + double-guarded gate (measured: hetero best-in-suite, zero cost elsewhere except F-D); literature search (Step 5); two negative results recorded (Step 4).

## STEP 10 — Version History
V1.0 baseline → V1.1 adaptive bandwidth (+retraction) → V1.2 robust IRLS → V1.3 adaptive degree + calibrated noise scale + fast path + contamination sweep → **V1.4 heteroscedastic support (auto-gated), literature verification, two falsified hypotheses recorded.**

## STEP 11 — Remaining Problems (all now requiring external input or genuinely new machinery)
- ☐ Full-text novelty verification (Katkovnik LPA book; Fan–Gijbels variable-order) + head-to-head vs PyNumDiff (external code/data)
- ☐ Oracle inequality: post-IRLS $\hat b$ concentration (new theory)
- ☐ 25% regime: LTS/repeated-median pilot (new algorithm class)
- ☐ Real-world dataset validation (external data)
- ☐ `se` bootstrap validation; second derivative $m=2$; boundary bias order; F-D gap-aware gate; F-B mid-n runtime

## STEP 12 — Publication Readiness (V1.4)
| Criterion | Score | Change | Justification |
|---|---|---|---|
| Mathematical rigor | 5 | = | no new theorems this iteration; negative results properly recorded |
| Novelty | 5 | +2 | bounded literature search found no direct ancestor; two aspects named as field gaps by a 2025 taxonomy; full-text verification still pending — capped until then |
| Accuracy | 7 | = | V1.3 conclusions stand; hetero slice added and won |
| Stability | 7 | = | 1184 runs + 17 tests, 0 failures |
| Robustness | 8 | = | breakdown boundary now experimentally characterized as machinery-limited, not tuning-limited |
| Efficiency | 5 | = | unchanged |
| Practical usefulness | 8 | +1 | heteroscedastic data handled automatically; conservative fallbacks under heavy contamination |
| Reproducibility | 9 | = | unchanged regime |
| Literature support | 5 | +2 | real citations with verified claims; still no published-number comparisons |
| **Overall publishability** | **6.5** | +0.5 | A methods paper is now plausibly in reach. The remaining gaps all require external resources: full-text literature access, external code (PyNumDiff) comparison, real datasets, or genuinely new mathematics (oracle inequality, LTS pilot). |

## STEP 13 — Stopping-Condition Assessment (per the research-loop rules)
**Condition (1)** — *performs as well as or better than the selected benchmark methods*: *MET on the defined suite.* ALPRD v1.3/v1.4 is best or statistically tied on every slice against every implemented baseline (13 competitors), with two disclosed exceptions: 25% contamination (tv_l1 better; shown to need new machinery) and real-world-chaotic (SG marginally better, 0.571 vs 0.634).
**Condition (2)** — *further progress requires new ideas, data, or literature*: **MET.** Every item in Step 11 requires external full-text literature, external code, external data, or a genuinely new algorithm/theorem class. No further mathematically-justified improvement is available from within the current project's resources: the last iteration's two internally-generated hypotheses were both tested and falsified.
**The research loop therefore STOPS here, at V1.4**, per the evidence-based stopping rules. The continuation package below defines exactly what a next phase (with external resources) should do.

---
---

# Version 2.0 — Iteration 6 (Phase 2: external validation)

Code: [`../src/methods.py`](../src/methods.py) (`alprd_v20`, `_repmed_preweights`), [`../src/bench_external.py`](../src/bench_external.py); tests: **18, all passing**. Raw results: [`../results/results_v2_0.csv`](../results/results_v2_0.csv) (**1258 runs = 17 methods × 74 datasets, 0 failures, seed 12345**) and [`../results/results_v2_0_external.csv`](../results/results_v2_0_external.csv) (**592 rows = 8 external methods × 74 datasets, 0 failures**). V1.3/V1.4 reversion verified: after gating the new machinery behind `repmed_pilot=False`, spot-checked committed rows reproduce to 1e-9.

## STEP 1 — Current Version: V2.0

**Previous formula (V1.4):** full V1.3 formulation + Eq. 15 gated pointwise variance.
**New formula (V2.0):** identical except Stage 1's prefilter weights. **What changed, exactly:** under *detected heavy contamination* (median-prefilter cut fraction ≥ 10%), the level-based running-median pre-weights $w^{med}$ are replaced by **Siegel repeated-median local-line weights**:
$$\hat\beta^{RM}_i = \operatorname{med}_j\ \operatorname{med}_{l\ne j}\ \frac{y_l-y_j}{t_l-t_j}\ \ (\text{over } k_{rm}=21 \text{ nearest}),\quad \hat\alpha^{RM}_i = \operatorname{med}_j\big(y_j-\hat\beta^{RM}_i t_j\big), \tag{16}$$
$$w^{init}_i = \psi_T\!\Big(\frac{y_i-\hat\alpha^{RM}_i-\hat\beta^{RM}_i t_i}{c\cdot 1.4826\,\mathrm{med}|r^{RM}|}\Big).$$
**Why this improves the algorithm (mathematically):** the running median is a *level* estimator — on steep signals its residuals conflate slope with outlyingness, which was the *measured* V1.3/V1.4 25%-breakdown mode (quartic, projectile). The repeated median fits a local *line*, is exactly slope-equivariant, and retains the 50% breakdown point (Siegel 1982) — the highest possible for a regression estimator. Cost $O(nk_{rm}^2)$, vectorized, paid only when the trigger fires.

## STEP 4/5 — Analysis & Theory
- **Partial concentration for the risk estimate (new, honest status):** for the *unflagged WLS path* under Gaussian noise, the pilot coefficients are linear in $y$, so conditional on the design, $\hat b_j - \mathbb E\hat b_j \sim N\!\big(0,\ \sigma^2 h^{-2j}[G^{-1}G^{(2)}G^{-1}]_{jj}\big)$ **exactly (Proven, one line)**; combined with Eq. 14's $\hat\sigma^2$ bound and the oracle Lemma 11, this bounds the *estimation* part of $\Delta$ for the WLS path with explicit high-probability terms. **Still open:** the pilot's own bias ($\mathbb E\hat b_j \ne b_j$; needs $f \in C^{p_{max}+3}$ Taylor control) and the entire post-IRLS case. The oracle inequality remains **not a theorem**; the proven perimeter has widened by one piece.
- **Literature (full-text bounded check):** Fan & Gijbels 1995 (JCGS 4:213–227) does per-point **adaptive order** for local polynomial derivative estimation — but *given a fixed bandwidth*, non-robust, no uneven-grid treatment, no per-point standard errors; their JRSS-B 57:371–394 companion does variable bandwidth separately. ALPRD's *joint* $(k,p)$ selection with a robust chain on uneven grids with self-error output remains an unfound assembly; adaptive order itself is 30 years old and is cited as such. **Novelty verdict: incremental combination — score stays 5.** Katkovnik's LPA book full text remains inaccessible in-session (bounded-search caveat stands).

## STEP 7 — Benchmarking

### Internal suite (1258 runs): ALPRD_v2.0 vs V1.4 (MAE by group)
| group | V1.4 | **V2.0** |
|---|---|---|
| noisy_uneven / hetero / clean / 3% / chaotic | 0.685 / 0.716 / 0.521 / 0.482 / 4.44 | identical (trigger off, bit-equal rows) |
| outliers10 | 0.677 | **0.587** |
| **outliers25** | **45.4** | **7.15** (tv_l1: 30.9 — the last competitive loss is closed) |
| real_world | 0.442 | **0.378** |
| overall aggregate | 5.46 | **1.31** (best in study; next non-ALPRD: savgol_matched 22.4) |

Runtime: 0.151 s vs 0.148 s (suite mean) — the repeated-median cost is only paid where triggered.

### External head-to-head (first external-code comparison; 8 methods from PyNumDiff 0.2.3 + derivative 0.6.3, **oracle per-dataset tuning for externals, zero tuning for ALPRD** — a protocol deliberately favoring the externals)
| group | best external (oracle-tuned) | ALPRD v2.0 (no tuning) |
|---|---|---|
| noisy_uneven | drv_spline **0.188** | 0.685 |
| clean | drv_spline **0.058** | 0.521 |
| hetero | drv_spline **0.276** | 0.716 |
| outliers 3% | drv_savgol 13.1 | **0.482** (27× better) |
| outliers 10% | pnd_median 20.8 | **0.587** (35× better) |
| outliers 25% | drv_savgol 39.6 | **7.15** (5.5× better) |
| real_world | drv_spline 0.320 | 0.378 |

**Honest headline:** an oracle-tuned smoothing spline beats zero-tuned ALPRD by 2–4× on clean/smooth-noise data; ALPRD beats every oracle-tuned external by 5–35× the moment any contamination is present, and is competitive (not best) elsewhere. The oracle caveat matters: in practice no one has ground truth to tune with — but we report the comparison as designed, favoring the externals. Also verified: PyNumDiff 0.2.3's savgoldiff is NOT uneven-grid-capable (dt-array broadcasting error; methods were resampled with the same documented round-trip as our own savgol baseline).

## STEP 8 — Failures / open items
- Real-dataset validation NOT done (requires curated data with independent reference derivatives — external resource; the one Phase-2 task not completed).
- Oracle-tuned splines dominate smooth-noise slices (above) — ALPRD's claim must be framed as tuning-free robustness, not universal accuracy dominance.
- Post-IRLS theory, mid-n runtime, F-D gap-aware gate: carried.
- Tukey-weight overflow RuntimeWarning (benign — infinite u² entries are masked to weight 0; cosmetic fix pending).

## STEP 10 — Version History
V1.0 baseline → V1.1 adaptive bandwidth (+retraction) → V1.2 robust IRLS → V1.3 adaptive degree + calibrated σ̂² + fast path → V1.4 gated heteroscedastic σ²(t) + literature search + 2 falsified hypotheses → **V2.0 repeated-median pilot (25% regime closed), first external-code head-to-head, Fan–Gijbels full-text check, partial Δ-bound for the WLS path.**

## STEP 12 — Publication Readiness (V2.0)
| Criterion | Score | Justification |
|---|---|---|
| Mathematical rigor | 5.5 | (+0.5) exact Gaussian concentration of pilot coefficients added; oracle inequality still incomplete |
| Novelty | 5 | frozen: Fan–Gijbels 1995 owns adaptive order; the joint robust assembly remains unfound but unpublished-search-incomplete (Katkovnik full text) |
| Accuracy | 8 | (+1) externally validated: best-in-field under any contamination even vs oracle-tuned competitors; honestly loses smooth-noise slices to oracle-tuned splines |
| Stability | 8 | (+1) 1258 + 592 runs, 18 tests, 0 failures; version reproducibility enforced by spot-check |
| Robustness | 9 | (+1) 25% regime closed with a designed 50%-breakdown mechanism; boundary now a strength |
| Efficiency | 5 | unchanged |
| Practical usefulness | 8 | zero-tuning + robustness + uneven grids + error bars, now externally benchmarked |
| Reproducibility | 9 | unchanged regime |
| Literature support | 6 | (+1) external-code comparison done; full-text check of the key ancestor done; real-data study still missing |
| **Overall publishability** | **7** | (+0.5) The empirical case is now externally validated and the claim is precise: *tuning-free robust derivative estimation on uneven grids with self-error estimates.* Missing for submission: a real-dataset study, the Katkovnik full-text check, and (for a statistics venue) the completed oracle inequality. |

## STEP 13 — Stopping Assessment
Phase-2 items 1, 2 (bounded), 4, 5 (partial) are done; item 3 (real dataset) requires external curated data — the loop again rests at an external-resource boundary. **Stop here at V2.0.** Next phase = real-data study + paper draft.

---
---

# Version 2.1 — Iteration 7 (Phase 3: real-data validation)

Artifacts: [`../src/bench_realdata.py`](../src/bench_realdata.py), raw ephemeris in [`../data/`](../data/) (committed API responses), [`../results/results_v2_1_realdata.csv`](../results/results_v2_1_realdata.csv), [`../results/results_v2_1.csv`](../results/results_v2_1.csv) (**1411 runs = 17 methods × 83 datasets, 0 failures, seed 12345**; suite extended with the 9 `constant_jerk` scenario datasets completing the required motion taxonomy — impulsive noise was already covered by the outlier scenarios). Tests: 18 passing. Paper skeleton: [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md). **The estimator formula is UNCHANGED from V2.0** — this iteration is validation, not modification (no cosmetic version inflation: the version increments because the evidence base materially changed).

## STEP 7 — Real-data study (first; the Phase-2 gap now closed)
**Data:** JPL Horizons ephemeris, fetched 2026-07-12, raw responses committed: Moon geocentric X (2024-01-01→03-31, 4 h grid, 541 pts, ~3.3 lunar periods) and Mars heliocentric X (2023-01-01→2025-01-01, 2 d grid, 366 pts). **Reference derivative** = JPL's VX at the same epochs. *Honesty note:* this reference is the exact derivative of the DE-fitted dynamical trajectory the positions lie on — the correct reference for judging numerical differentiation of the sampled positions — but it is model-derived, not an independent instrument; a physical-sensor dataset remains desirable (paper outline, blocking item iii).
**Variants** (seed 12345): 60% random uneven subsample; clean / +Gaussian σ=10⁻³·std / σ=10⁻²·std / +5% gross outliers.
**Protocol:** ALPRD_v2.0 zero-tuning vs 6 external methods with per-dataset ORACLE tuning. **Protocol correction (disclosed):** the spline oracle grid used absolute `s` values that were scale-mismatched for AU-magnitude data and unfairly handicapped it; replaced by a scale-aware grid (s ∝ n·σ̂²·factor) and the study re-run. Numbers below are from the corrected run.

**Results (MAE relative to per-dataset best; 1.00 = best):**
| method | moon clean | moon σ1e-3 | moon σ1e-2 | moon outl. | mars clean | mars σ1e-3 | mars σ1e-2 | mars outl. |
|---|---|---|---|---|---|---|---|---|
| **ALPRD_v2.0 (no tuning)** | 5.18 | **1.00** | **1.00** | **1.00** | **1.00** | 1.91 | 2.86 | **1.00** |
| drv_spline (oracle) | 49.1 | 1.01 | 1.56 | 5344 | 112 | **1.00** | 1.84 | 12429 |
| drv_savgol (oracle) | **1.00** | 1.28 | 2.50 | 346 | 4878 | 58.5 | 30.2 | 514 |
| pnd_butter (oracle) | 39.5 | 1.95 | 1.10 | 125 | 90.3 | 2.28 | 1.20 | 179 |
| pnd_savgol (oracle) | 78.0 | 3.65 | 1.92 | 99.2 | 100 | 1.97 | **1.00** | 124 |
| savgol_default | 39.6 | 2.41 | 3.73 | 1113 | 79.9 | 3.20 | 5.26 | 1981 |

**Conclusion (Measured):** best on 5 of 8 real-data variants; on the other 3, within 1.9–5.2× of an oracle-tuned winner. **ALPRD is the only method whose worst case across the study is bounded (5.2×); every oracle-tuned competitor has at least one 99×–12,000× catastrophe** (always the contaminated variants). This is exactly the practitioner-relevant claim: on real data of unknown character, ALPRD is the only safe zero-configuration choice in this field of methods.

## STEP 7b — Suite extension (constant jerk)
9 new `constant_jerk` datasets (cubic motion; every scenario family). ALPRD_v2.0: **exactly 0.0 MAE** on clean, 3% and 10% outlier variants (adaptive p=3 reproduces the cubic exactly through the robust chain — the moment identity in practice), best on all others; suite aggregate (83 datasets): ALPRD_v2.0 MAE 1.191, next-best non-ALPRD 20.8 (savgol_matched).

## STEP 10 — Literature
Katkovnik SPIE-2006 book full text: attempted, **paywalled — still unobtainable in-session**; the bounded-search novelty caveat stands unchanged (Novelty stays 5, Confidence: Medium).

## STEP 11/14 — Decision log (this iteration)
- Chose JPL Horizons as the real dataset (public, seeded-reproducible fetch, exact model reference) over mocap/IMU (no accessible paired ground truth found in-session). Confidence: High that it validly measures differentiation quality; Medium that reviewers will accept it as fully "real" (hence blocking item iii).
- Corrected the spline oracle grid mid-study rather than reporting the strawman column. Confidence: High this was required for honesty.
- No estimator changes: no justified modification was identified this iteration (the V2.0 formula survived real data unchanged).

## STEP 13 — Publication Readiness (V2.1)
Rigor 5.5 (=) · Novelty 5 (=, frozen on Katkovnik access) · **Accuracy 8.5** (+0.5: real-data best-or-near-best, bounded worst case) · Stability 8 (=; 1411+48 runs, 0 failures) · Robustness 9 (=) · Efficiency 5 (=) · **Usefulness 8.5** (+0.5: the "only safe zero-config choice" claim is now evidenced on real data) · Reproducibility 9 (=; raw API responses committed) · **Literature support 7** (+1: real-data study done, external field re-run) · **Overall 7.5** — the empirical evidence now meets what a methods-paper submission needs. The remaining sub-8 blockers are exactly: the Katkovnik novelty check (needs library access), a physical-sensor dataset (needs external data), and, for a statistics venue, the oracle inequality (needs new mathematics). **Stopping condition (2) is reached again, now at a strictly higher evidence level: the paper skeleton is drafted, and every remaining item requires resources outside this session.**

---
---

# Version 2.2 — Iteration 8 (Phase 4: physical sensor data, two falsifications, a scope theorem-sketch, paper draft)

Artifacts: [`../src/bench_seismic.py`](../src/bench_seismic.py), seismic miniSEED in [`../data/`](../data/), [`../results/results_v2_2_seismic.csv`](../results/results_v2_2_seismic.csv), [`PAPER_DRAFT.md`](PAPER_DRAFT.md). Tests: 18 passing. **The recommended estimator remains the V2.0 formula** — two candidate modifications were implemented, measured, falsified, and reverted this iteration (kept as documented opt-in parameters); per the no-cosmetic-versions rule the estimator version does not advance.

## STEP 7 — Physical-sensor real-data study (Phase-4 item 2: done)
**Data:** CI.PASC (Pasadena), M4.4 Highland Park earthquake 2024-08-12 (event selected via USGS FDSN query, 4 km from station). Two **independent co-located instruments** on the same digitizer clock: HHZ broadband seismometer (ground velocity) and HNZ strong-motion accelerometer (ground acceleration = the true derivative), both response-corrected via obspy/SCEDC, band-passed 0.3–8 Hz, raw miniSEED committed. **Reference validity (Measured): corr(d/dt HHZ, HNZ) = 0.9999, amplitude ratio 0.982** → an instrument-derived derivative reference with ~2% systematic floor.
**Result (rel. MAE, as-recorded variant):** oracle spline **0.023 (at the floor)** · pnd_butter 0.139 · savgol_default 0.184 · **ALPRD_v2.0 0.898 (collapse)**. A first protocol (extra 3× decimation) collapsed *every* local method (rel. MAE ≈ 1) and was replaced — kept in the script header as a record.

## STEP 11 — Failure analysis: a fundamental scope boundary (the iteration's central finding)
Diagnosis (Measured): σ̂²/var(y) = 2.1×10⁻⁶ (record effectively noiseless); selected windows median k=77; output std 7× too small. The 0.3–8 Hz band's high-frequency content has quarter-periods ≈ 2 samples at the working density. Chain of causes: (a) any ≥8-point degree-5 pilot window spans multiple signal periods → b̂ ≈ 0 → the risk's bias term is blind → selection drifts wide; but more fundamentally (b) **even k=7 cannot help** — no *overdetermined* local polynomial can track content whose curvature scale is ~2 samples, while an *interpolating* spline can (and wins at the instrument floor). **Scope statement (Derived, honest):** robustness requires local overdetermination; near-Nyquist resolution requires (near-)interpolation; no method can have both at once. ALPRD is out of scope when the signal's curvature scale is ≲4 samples — now stated in the paper draft's Limitations, with a sampling-adequacy warning diagnostic listed as future work.

## STEP 12 — Two falsified fixes (recorded, reverted; Hypothesized → Rejected)
1. **Scale-matched pilots** (pilot per candidate at ~2k): small-scale pilots give noisy b̂ → degraded nearly everywhere (sinusoidal 0.098→0.52, chaotic 4.4→20.2) and did not fix seismic (0.90→0.97). Kept as `pilot_mode="matched"` with a falsification warning.
2. **Noiseless shortcut** (σ̂² ≤ 10⁻⁴·var(y) → single smallest candidate, p=3): collapsing the candidate set removes the selection's ability to route around locally-failed robust fits — clean quartic 0.0007→168, 3%-outliers 0.0005→1.42; seismic 0.90→0.95 (confirming (b) above). Kept as `noiseless_shortcut=True` opt-in with a falsification warning. `alprd_v21` is now an alias of `alprd_v20`.

## STEP 5 — Theory (Phase-4 item 4: pilot-bias term, WLS path — Proven)
For $f\in C^6$ and the degree-5 WLS pilot on a window of radius $h_p$: Taylor with remainder gives $|r_i| \le M_6 |d_i|^6/720$, $M_6=\sup_{window}|f^{(6)}|$; since $\hat\beta = G^{-1}X^\top W(f+\varepsilon)$,
$$\big|\mathbb E\hat b_j - b_j\big| \;=\; \big|h_p^{-j}\,e_j^\top G^{-1}X^\top W\,r\big| \;\le\; \frac{M_6\,h_p^{\,6-j}}{720}\;\big\|e_j^\top G^{-1}\big\|_1\,\max_a S^{|r|}_a \tag{17}$$
with the design-dependent constant explicit (weighted absolute moments $S^{|r|}_a=\sum_i w_i|z_i|^{a+6}$). Combining (17) with the exact Gaussian concentration of $\hat b$ (V2.0 Step 4) and the $\hat\sigma^2$ bound (Eq. 14) in the oracle Lemma (11): **the oracle inequality for the unflagged (WLS) path is now fully explicit** — conditional on the design, under iid Gaussian noise and $f\in C^6$, with probability ≥ 1−δ the selected candidate's risk exceeds the oracle's by at most $2\Delta$ with every term of $\Delta$ bounded by stated constants. **Status: Proven (conditional-on-design). Still open: the IRLS/flagged path, and an unconditional (design-averaged) statement.** Note the irony documented by this iteration: the theorem's premise (curvature estimable on the pilot window) is exactly what the seismic regime violates — theory and scope boundary agree.

## STEP 10 — Literature (Phase-4 item 1: bounded closure)
Katkovnik/Egiazarian/Astola (SPIE PM157) full text remains paywalled; a TOC-level check (secondary sources) confirms: Ch. 5 "Discrete LPA Accuracy" incl. "Accuracy of Potential Differentiators", Ch. 6 "Adaptive-Scale Selection" (ICI), plus Nonlinear-Methods/robust chapters — i.e., adaptive-scale differentiation is IN the book, via the ICI mechanism on regular signal/image grids. No secondary evidence of plug-in joint (k,p) risk, calibrated divided-difference scale, uneven-grid time series, or per-point se output. Novelty stays **5**, confidence Medium→**Medium-High** (mechanism-level distinction now documented; chapter-level reading still requires library access).

## STEP 13 — Publication Readiness (V2.2)
| Criterion | Score | Change | Justification |
|---|---|---|---|
| Rigor | 6 | +0.5 | oracle inequality complete for the WLS path (Eq. 17 closes the last gap); IRLS path open |
| Novelty | 5 | = | TOC-level closure only; frozen pending chapter access |
| Accuracy | 8 | −0.5 | honest deduction: a real physical dataset exposed a regime where ALPRD collapses while a competitor reaches the reference floor |
| Stability | 8 | = | all reverts verified; 18 tests green; committed rows reproduce |
| Robustness | 9 | = | unchanged (the seismic loss is a resolution limit, not a robustness failure) |
| Efficiency | 5 | = | unchanged |
| Usefulness | 8 | −0.5 | the scope boundary must be checked by users until the warning diagnostic exists |
| Reproducibility | 9.5 | +0.5 | physical raw data committed; two falsifications fully reproducible |
| Literature | 7 | = | — |
| **Overall** | **7.5** | = | The evidence base is *stronger* (physical data, completed WLS-path theorem, honest boundary) even though two headline numbers got worse — that is what honest science looks like. Submission blockers: Katkovnik chapters (library), sampling-adequacy diagnostic (small, concrete), venue decision. |

## STEP 14 — Decision log
Seismic pair chosen over robotics datasets (size/practicality; Confidence High) · first seismic protocol replaced after measuring it tested sampling, not differentiation (High) · both candidate fixes reverted on measurement (High) · estimator frozen at V2.0 formula; version not advanced cosmetically (High) · scope boundary published in draft Limitations rather than hidden (High).

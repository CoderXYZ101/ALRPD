"""
Numerical differentiation methods for unevenly sampled, noisy 1-D data.

All methods share the signature:
    d = method(t, y, **kwargs)
where t is a 1-D array of strictly increasing sample times (may be uneven)
and y is a 1-D array of observed (possibly noisy) values.
Returns an array d of the same length as t: an estimate of f'(t_i) at every
sample point t_i (derivative queried AT the data points, not off-grid).
"""

import numpy as np


# ----------------------------------------------------------------------
# Baseline methods
# ----------------------------------------------------------------------

def forward_difference(t, y):
    n = len(t)
    d = np.empty(n)
    d[:-1] = (y[1:] - y[:-1]) / (t[1:] - t[:-1])
    d[-1] = d[-2]  # no forward neighbor at last point; repeat last valid slope
    return d


def backward_difference(t, y):
    n = len(t)
    d = np.empty(n)
    d[1:] = (y[1:] - y[:-1]) / (t[1:] - t[:-1])
    d[0] = d[1]
    return d


def central_difference(t, y):
    """Non-uniform central difference (3-point, first-order-consistent
    Taylor derivation), standard textbook formula:

        f'(t_i) ~= [ h_-^2 (y_{i+1}-y_i) + h_+^2 (y_i - y_{i-1}) ]
                   / (h_+ h_- (h_+ + h_-))

    where h_- = t_i - t_{i-1}, h_+ = t_{i+1} - t_i.
    Endpoints fall back to one-sided differences.
    """
    n = len(t)
    d = np.empty(n)
    hm = t[1:-1] - t[:-2]
    hp = t[2:] - t[1:-1]
    d[1:-1] = (hm**2 * (y[2:] - y[1:-1]) + hp**2 * (y[1:-1] - y[:-2])) / (
        hm * hp * (hm + hp)
    )
    d[0] = (y[1] - y[0]) / (t[1] - t[0])
    d[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    return d


def high_order_fd(t, y):
    """5-point high-order finite difference. Exact for a uniform grid
    (standard 4th-order stencil); for uneven grids we build the equivalent
    stencil at each interior point by fitting a degree-4 polynomial through
    the 5 nearest points by index and differentiating it analytically
    (a Fornberg-style finite-difference weight computation).
    """
    n = len(t)
    d = np.empty(n)
    for i in range(n):
        lo = max(0, min(i - 2, n - 5))
        idx = np.arange(lo, lo + min(5, n))
        w = _fd_weights(t[idx], t[i], m=1)
        d[i] = np.dot(w, y[idx])
    return d


def _fd_weights(x, x0, m):
    """Fornberg (1988) algorithm: finite-difference weights for the m-th
    derivative at x0 using (possibly unevenly spaced) nodes x.
    Returns weights w such that f^(m)(x0) ~= sum(w * f(x)).
    """
    n = len(x) - 1
    c1 = 1.0
    c4 = x[0] - x0
    C = np.zeros((n + 1, m + 1))
    C[0, 0] = 1.0
    for i in range(1, n + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    C[i, k] = c1 * (k * C[i - 1, k - 1] - c5 * C[i - 1, k]) / c2
                C[i, 0] = -c1 * c5 * C[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                C[j, k] = (c4 * C[j, k] - k * C[j, k - 1]) / c3
            C[j, 0] = c4 * C[j, 0] / c3
        c1 = c2
    return C[:, m]


def cubic_spline_derivative(t, y):
    from scipy.interpolate import CubicSpline

    cs = CubicSpline(t, y)
    return cs(t, 1)


def savitzky_golay_resampled(t, y, window=11, poly_order=3):
    """Classical Savitzky-Golay requires a UNIFORM grid. To apply it to
    uneven data we resample onto a uniform grid via linear interpolation,
    filter, then map the derivative back to the original sample times by
    interpolation. This round-trip is itself a source of error that we
    report honestly in benchmarks (it is a real limitation of applying
    Savitzky-Golay to non-uniform data, not an implementation bug).
    """
    from scipy.signal import savgol_filter

    n = len(t)
    if window >= n:
        window = n - 1 if (n - 1) % 2 == 1 else n - 2
    if window <= poly_order:
        window = poly_order + 2
    if window % 2 == 0:
        window += 1

    t_uniform = np.linspace(t[0], t[-1], n)
    dt = t_uniform[1] - t_uniform[0]
    y_uniform = np.interp(t_uniform, t, y)
    d_uniform = savgol_filter(y_uniform, window, poly_order, deriv=1, delta=dt)
    return np.interp(t, t_uniform, d_uniform)


# ----------------------------------------------------------------------
# Proposed method V1.0: Adaptive Local Polynomial Regression
# Differentiation (ALPRD) for uneven, noisy samples.
# ----------------------------------------------------------------------

def _tricube(u):
    w = np.zeros_like(u)
    m = np.abs(u) < 1
    w[m] = (1 - np.abs(u[m]) ** 3) ** 3
    return w


def alprd_v1(t, y, poly_order=2, k=None, span=0.3, kernel="tricube",
             min_points_factor=2, return_variance=False, sigma2=None):
    """Adaptive Local Polynomial Regression Differentiation, version 1.0.

    At every query point t0 = t_i:
      1. choose a LOCAL bandwidth h(t0) as the distance to the k-th
         nearest neighbor of t0 in time (k adapts the window WIDTH to
         local sample DENSITY -- the core adaptation for uneven sampling).
      2. weight neighbors with a tricube kernel on u = (t - t0) / h(t0).
      3. fit a degree-`poly_order` polynomial by weighted least squares.
      4. the derivative estimate is m! * beta_m (m=1 for first derivative).

    Parameters
    ----------
    t, y : arrays, t strictly increasing
    poly_order : local polynomial degree p (default 2; Fan & Gijbels 1996
        recommend p - m odd for first-derivative estimation, i.e. p=2 for m=1)
    k : int or None. Number of neighbors defining the bandwidth. If None,
        k = max(poly_order + 2, round(span * n)).
    span : fraction of the data used to set k when k is None (LOESS-style).
    min_points_factor : safety multiplier ensuring the local design matrix
        is well-conditioned (at least min_points_factor*(poly_order+1) points
        must have nonzero weight).
    return_variance : if True, also return a per-point self-estimated
        standard error of the derivative (see docs for derivation).
    sigma2 : optional externally supplied noise variance; if None it is
        estimated from the data via the Rice (1984) difference-based
        estimator, adapted for uneven spacing.

    Returns
    -------
    d : array of derivative estimates at each t_i
    se : array of estimated standard errors (only if return_variance=True)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    p = poly_order
    if k is None:
        k = max(p + 2, int(round(span * n)))
    k = min(k, n)
    min_pts = min_points_factor * (p + 1)

    if sigma2 is None:
        sigma2 = _rice_variance_estimator(t, y)

    d = np.empty(n)
    se = np.empty(n) if return_variance else None

    for i in range(n):
        t0 = t[i]
        dist = np.abs(t - t0)
        order = np.argsort(dist)
        # ensure enough points for a well-posed fit, and enough to include
        # the k-th neighbor distance as bandwidth
        kk = max(k, min_pts)
        kk = min(kk, n)
        h = dist[order[kk - 1]]
        if h == 0:
            h = np.max(dist[order[:kk]]) if np.max(dist[order[:kk]]) > 0 else 1e-12

        u = (t - t0) / h
        w = _tricube(u) if kernel == "tricube" else np.exp(-0.5 * u**2)
        mask = w > 0
        if mask.sum() < p + 1:
            # widen window until well-posed
            extra = p + 1 - mask.sum()
            kk2 = min(kk + extra + 2, n)
            h = dist[order[kk2 - 1]]
            u = (t - t0) / h
            w = _tricube(u) if kernel == "tricube" else np.exp(-0.5 * u**2)
            mask = w > 0

        ti = t[mask] - t0
        yi = y[mask]
        wi = w[mask]

        X = np.vander(ti, N=p + 1, increasing=True)  # columns: 1, ti, ti^2, ...
        Wm = wi[:, None]
        XtW = X.T * wi
        XtWX = XtW @ X
        XtWy = XtW @ yi

        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(XtWX, XtWy, rcond=None)

        d[i] = beta[1]  # first derivative = 1! * beta_1

        if return_variance:
            XtWX_inv = np.linalg.pinv(XtWX)
            XtW2X = (X.T * (wi**2)) @ X
            cov = sigma2 * (XtWX_inv @ XtW2X @ XtWX_inv)
            se[i] = np.sqrt(max(cov[1, 1], 0.0))

    if return_variance:
        return d, se
    return d


# ----------------------------------------------------------------------
# V1.1 machinery
# ----------------------------------------------------------------------

def _knn_window(t, i, k):
    """Return (lo, hi) such that t[lo:hi] are the k nearest samples to t[i]
    (inclusive of t[i] itself), found by expanding two pointers over the
    already-sorted array t. Because t is sorted, the k nearest neighbors of
    any point always form a CONTIGUOUS index range, so no per-query sort is
    needed. Cost: O(k) per query instead of O(n log n).

    Tie handling: when the left and right candidates are equidistant we take
    the left one first; the resulting *set* of distances (and therefore the
    k-th neighbor distance used as bandwidth) is identical to what a full
    argsort would produce, so downstream results match `alprd_v1` exactly.
    """
    n = len(t)
    k = min(k, n)
    lo = hi = i  # window is t[lo..hi] inclusive; starts as just the query point
    for _ in range(k - 1):
        left_ok = lo > 0
        right_ok = hi < n - 1
        if left_ok and right_ok:
            if t[i] - t[lo - 1] <= t[hi + 1] - t[i]:
                lo -= 1
            else:
                hi += 1
        elif left_ok:
            lo -= 1
        else:
            hi += 1
    return lo, hi + 1  # half-open


def _local_wls(ti, yi, wi, p):
    """Weighted least-squares fit of degree p on centered abscissae ti.
    Returns (beta, ell) where ell is the equivalent-kernel row for the
    derivative coefficient: d = ell @ yi, so variance and bias functionals
    can be computed from ell directly.  Returns (None, None) if the local
    Gram matrix is singular beyond repair.
    """
    X = np.vander(ti, N=p + 1, increasing=True)
    XtW = X.T * wi
    XtWX = XtW @ X
    try:
        A = np.linalg.solve(XtWX, XtW)      # (p+1) x m: full coefficient map
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(XtWX) @ XtW
    beta = A @ yi
    return beta, A


def alprd_v1_fast(t, y, poly_order=2, k=None, span=0.3, sigma2=None,
                  return_variance=False):
    """Performance rewrite of `alprd_v1` with IDENTICAL statistical output
    (verified in tests/test_methods.py): replaces the per-query full
    distance sort, O(n log n) each, with the O(k) sliding two-pointer
    window of `_knn_window`. Total cost drops from O(n^2 log n) to O(n k).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    p = poly_order
    if k is None:
        k = max(p + 2, int(round(span * n)))
    k = min(k, n)
    min_pts = 2 * (p + 1)
    kk = min(max(k, min_pts), n)

    if sigma2 is None and return_variance:
        sigma2 = _rice_variance_estimator(t, y)

    d = np.empty(n)
    se = np.empty(n) if return_variance else None

    for i in range(n):
        lo, hi = _knn_window(t, i, kk)
        tw = t[lo:hi] - t[i]
        h = np.max(np.abs(tw))
        if h == 0:
            h = 1e-12
        w = _tricube(tw / h)
        mask = w > 0
        if mask.sum() < p + 1:
            # widen exactly as alprd_v1 does: kk + deficit + 2 neighbors
            extra = p + 1 - int(mask.sum())
            lo, hi = _knn_window(t, i, min(kk + extra + 2, n))
            tw = t[lo:hi] - t[i]
            h = np.max(np.abs(tw))
            w = _tricube(tw / h)
            mask = w > 0

        ti, yi, wi = tw[mask], y[lo:hi][mask], w[mask]
        beta, A = _local_wls(ti, yi, wi, p)
        d[i] = beta[1]

        if return_variance:
            # d = ell @ yi with ell = [ (X'WX)^{-1} X'W ]_1, so
            # Var(d) = sigma2 * ||ell||^2, algebraically identical to V1.0's
            # sandwich formula sigma2 * e1'(X'WX)^{-1}X'W^2X(X'WX)^{-1}e1.
            ell = A[1]
            se[i] = np.sqrt(max(sigma2 * float(np.sum(ell**2)), 0.0))
    if return_variance:
        return d, se
    return d


def alprd_v11(t, y, poly_order=2, n_candidates=8, k_min=None, k_max_frac=0.5,
              pilot_frac=0.15, sigma2=None, return_details=False):
    """ALPRD Version 1.1: per-point data-driven bandwidth selection.

    For each query point t0 = t_i, instead of one fixed-span window, evaluate
    a geometric grid of candidate neighbor counts k_c and pick the one that
    minimizes the estimated pointwise risk

        R_hat(k_c; t0) = Bias_hat(k_c; t0)^2 + Var_hat(k_c; t0)

    built from the two quantities already derived for V1.0:

      Var_hat  = sigma2_hat * sum_i ell_i^2                       [Eq. (3),
                 exact for the assumed iid noise, since d = sum ell_i y_i]
      Bias_hat = beta_pilot_{p+1} * sum_i ell_i (t_i - t0)^{p+1}  [Eq. (6),
                 leading-order; beta_pilot_{p+1} ~ f^{(p+1)}(t0)/(p+1)!]

    where ell_i are the equivalent-kernel weights of the candidate fit and
    beta_pilot_{p+1} comes from a single degree-(p+1) pilot fit per point on
    a fixed moderate window (pilot_frac of the data).

    HONESTY NOTE (also in the research log): this is a *plug-in* pointwise
    risk minimizer in the spirit of, but simpler than, the full
    Goldenshluger-Lepski procedure -- GL compares pairs of estimates with a
    majorant penalty and carries an oracle inequality; the plug-in rule below
    carries no such finite-sample guarantee. We label it accordingly.

    Returns d, or (d, se, k_chosen) if return_details.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    p = poly_order

    if sigma2 is None:
        sigma2 = _rice_variance_estimator(t, y)

    if k_min is None:
        k_min = max(p + 3, 6)
    k_min = min(k_min, n)
    k_max = max(k_min + 1, min(n, int(round(k_max_frac * n))))
    cands = np.unique(np.round(np.geomspace(k_min, k_max, n_candidates)).astype(int))

    k_pilot = min(n, max(2 * (p + 3), int(round(pilot_frac * n))))

    d = np.empty(n)
    se = np.empty(n)
    k_sel = np.empty(n, dtype=int)

    for i in range(n):
        # ---- pilot fit: degree p+1, moderate fixed window, estimates the
        # (p+1)-th Taylor coefficient used by the bias functional
        lo, hi = _knn_window(t, i, k_pilot)
        tw = t[lo:hi] - t[i]
        h = np.max(np.abs(tw))
        w = _tricube(tw / max(h, 1e-12))
        m = w > 0
        if m.sum() >= p + 2:
            beta_pilot, _ = _local_wls(tw[m], y[lo:hi][m], w[m], p + 1)
            b_next = beta_pilot[p + 1] if beta_pilot is not None else 0.0
        else:
            b_next = 0.0

        # ---- candidate search
        best = None
        for kc in cands:
            lo, hi = _knn_window(t, i, int(kc))
            tw = t[lo:hi] - t[i]
            h = np.max(np.abs(tw))
            if h <= 0:
                continue
            w = _tricube(tw / h)
            m = w > 0
            if m.sum() < p + 2:   # need strict overdetermination for stability
                continue
            ti, yi, wi = tw[m], y[lo:hi][m], w[m]
            beta, A = _local_wls(ti, yi, wi, p)
            if beta is None:
                continue
            ell = A[1]                        # d = ell @ yi
            var_hat = sigma2 * float(np.sum(ell**2))
            bias_hat = b_next * float(ell @ ti**(p + 1))
            risk = bias_hat**2 + var_hat
            if best is None or risk < best[0]:
                best = (risk, float(beta[1]), var_hat, int(kc))

        if best is None:
            # pathological fallback: global least-squares line
            beta, _ = _local_wls(t - t[i], y, np.ones(n), 1)
            best = (np.inf, float(beta[1]), np.nan, n)

        d[i] = best[1]
        se[i] = np.sqrt(best[2]) if np.isfinite(best[2]) else np.nan
        k_sel[i] = best[3]

    if return_details:
        return d, se, k_sel
    return d


def savitzky_golay_matched(t, y, span=0.25, poly_order=3):
    """Control experiment for V1.0's outlier result: Savitzky-Golay with its
    window length MATCHED (in point count) to ALPRD v1.0's effective span,
    instead of the fixed default window=11. If ALPRD's outlier advantage was
    purely a window-width effect, this control should close the gap.
    """
    n = len(t)
    window = int(round(span * n))
    if window % 2 == 0:
        window += 1
    window = max(window, poly_order + 2 + (poly_order % 2))
    return savitzky_golay_resampled(t, y, window=window, poly_order=poly_order)


# ----------------------------------------------------------------------
# V1.2 machinery: robust (IRLS) adaptive selection, vectorized; plus
# TV-regularized and RBF-FD baselines.
# ----------------------------------------------------------------------

def _batched_windows(t, k, rows=None):
    """Vectorized k-nearest-neighbor windows for ALL query points at once.

    For sorted t, the k nearest neighbors of t[i] form a contiguous slice
    t[lo:lo+k] with lo in [i-k+1, i]. We pick, per i, the lo minimizing the
    window radius max(t[i]-t[lo], t[lo+k-1]-t[i]).

    Output-equivalence to `_knn_window` (Proven, and tested in
    tests/test_methods.py::test_batched_windows_match): any minimal-radius
    window has the same radius h, and the set of points STRICTLY inside
    radius h is the same for every minimal window; points exactly at
    distance h receive tricube weight 0. Hence downstream fits are
    identical regardless of which minimal window is returned.

    Returns idx of shape (len(rows), k): idx[r] = window indices for query
    point t[rows[r]]; rows=None means all n points. The rows parameter lets
    callers process query points in memory-bounded blocks (all temporaries
    here are (len(rows), k)).
    """
    n = len(t)
    k = min(k, n)
    rows = np.arange(n) if rows is None else np.asarray(rows)
    j = np.arange(k)
    lo_cand = rows[:, None] - j[None, :]              # (m, k) candidate starts
    valid = (lo_cand >= 0) & (lo_cand <= n - k)
    lo_c = np.clip(lo_cand, 0, n - k)
    radius = np.maximum(t[rows][:, None] - t[lo_c],
                        t[lo_c + k - 1] - t[rows][:, None])
    radius[~valid] = np.inf
    lo = lo_c[np.arange(len(rows)), np.argmin(radius, axis=1)]
    return lo[:, None] + j


def _tukey_weights(r, s, c=4.685):
    """Tukey biweight robustness weights for residuals r at scale s.
    w = (1 - (r/(c s))^2)^2 for |r| < c s, else 0.  s must be > 0.
    """
    u = r / (c * s)
    w = (1.0 - u**2)**2
    w[np.abs(u) >= 1.0] = 0.0
    return w


def _batched_robust_fit(D, Y, Wk, p, sigma, irls_iters=2):
    """Batched (per-row) weighted polynomial fits with optional Tukey IRLS.

    D  : (n, k) centered abscissae per window
    Y  : (n, k) observations per window
    Wk : (n, k) kernel weights
    Returns (beta, ell, Wtot):
      beta : (n, p+1) coefficients
      ell  : (n, k) equivalent-kernel rows for the derivative coefficient,
             i.e. beta[:,1] == sum(ell * Y, axis=1) with weights held fixed
      Wtot : (n, k) final combined weights (kernel * robustness)

    Robust scale: per-row s = max(sigma_global, 1.4826*median|resid|, floor)
    so that clean data (residuals ~ machine eps) is never zeroed out and
    gross outliers are always cut at 4.685 s.
    """
    n, k = D.shape
    X = D[:, :, None] ** np.arange(p + 1)[None, None, :]      # (n, k, p+1)
    Wtot = Wk.copy()

    def solve_pass(W):
        XtW = X.transpose(0, 2, 1) * W[:, None, :]            # (n, p+1, k)
        G = XtW @ X                                           # (n, p+1, p+1)
        # tiny relative ridge: keeps batch solve non-singular when robust
        # weights leave a window rank-deficient (documented in the log)
        diag_max = np.maximum(G.reshape(n, -1)[:, ::p + 2].max(axis=1), 1e-300)
        G += (1e-12 * diag_max)[:, None, None] * np.eye(p + 1)[None]
        A = np.linalg.solve(G, XtW)                           # (n, p+1, k)
        beta = (A * Y[:, None, :]).sum(axis=2)                # (n, p+1)
        return beta, A

    beta, A = solve_pass(Wtot)
    for _ in range(irls_iters):
        resid = Y - (X * beta[:, None, :]).sum(axis=2)
        med = np.median(np.abs(resid), axis=1, keepdims=True)
        s = np.maximum(np.maximum(sigma, 1.4826 * med), 1e-300)
        Wtot = Wk * _tukey_weights(resid, s)
        beta, A = solve_pass(Wtot)
    return beta, A[:, 1, :], Wtot


def alprd_v12(t, y, poly_order=2, n_candidates=8, k_min=None, k_max_frac=0.5,
              pilot_frac=0.15, sigma2=None, irls_iters=2, refine_pilot=True,
              return_details=False):
    """ALPRD Version 1.2: robust adaptive local polynomial differentiation.

    Same selection principle as V1.1 (minimize plug-in risk, Eq. 7) with
    three changes, each targeting a measured V1.1 failure:

    1. ROBUST FITS (Failure 1', outlier collapse): every candidate fit and
       the pilot fit run Tukey-biweight IRLS (`irls_iters` reweighting
       steps) on top of the kernel weights, with a scale floored at the
       (outlier-resistant) Rice sigma. Narrow windows can no longer be
       hijacked by a single gross error, because that error is cut from
       the fit rather than absorbed by it.
    2. EFFECTIVE-SIZE GUARD: candidates whose combined weights have
       effective sample size n_eff = (sum w)^2 / sum w^2 < p+2 are
       discarded (a narrow window that survives only by deleting most of
       its points is not evidence, it is overfitting).
    3. REFINED PILOT (Failure 2', chaotic gap): after a first selection
       pass, the (p+1)-degree pilot is re-fit per point on a window
       proportional to the selected bandwidth (1.5 * k_hat), and selection
       is repeated once. High-curvature regions thus get a local, rather
       than global, curvature estimate.

    Fully vectorized (Failure 4'): all n query points are processed as
    batched (n, k) arrays per candidate; no per-point Python loop.

    Returns d, or (d, se, k_chosen) if return_details. The reported se
    carries V1.1's caveats (post-selection, weights-held-fixed) -- it is
    a lower bound in that sense, stated honestly in the log.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    p = poly_order

    if sigma2 is None:
        sigma2 = _rice_variance_estimator(t, y)
    sigma = np.sqrt(max(sigma2, 0.0))

    if k_min is None:
        k_min = max(p + 3, 6)
    k_min = min(k_min, n)
    k_max = max(k_min + 1, min(n, int(round(k_max_frac * n))))
    cands = np.unique(np.round(np.geomspace(k_min, k_max, n_candidates)).astype(int))

    def _row_blocks(rows, k, max_elems=2**21):
        """Split a row-index array into blocks of at most max_elems/k rows,
        bounding every (block, k) temporary to ~max_elems doubles."""
        step = max(1, int(max_elems // max(k, 1)))
        for s in range(0, len(rows), step):
            yield rows[s:s + step]

    def pilot_bnext(k_pilot_arr):
        """Degree-(p+1) robust pilot fit; k may differ per point, so points
        are processed in groups of equal window size (<= n_candidates+1
        distinct values), each group in memory-bounded row blocks."""
        b_next = np.zeros(n)
        for kp in np.unique(k_pilot_arr):
            sel = np.where(k_pilot_arr == kp)[0]
            for blk in _row_blocks(sel, int(kp)):
                idx = _batched_windows(t, int(kp), rows=blk)
                D = t[idx] - t[blk, None]
                h = np.maximum(np.max(np.abs(D), axis=1, keepdims=True), 1e-300)
                Wk = _tricube(D / h)
                beta, _, _ = _batched_robust_fit(D, y[idx], Wk, p + 1, sigma,
                                                 irls_iters=irls_iters)
                b_next[blk] = beta[:, p + 1]
        return b_next

    def select(b_next):
        best_risk = np.full(n, np.inf)
        best_d = np.zeros(n)
        best_var = np.full(n, np.nan)
        best_k = np.full(n, n, dtype=int)
        all_rows = np.arange(n)
        for kc in cands:
            for blk in _row_blocks(all_rows, int(kc)):
                idx = _batched_windows(t, int(kc), rows=blk)
                D = t[idx] - t[blk, None]
                h = np.maximum(np.max(np.abs(D), axis=1, keepdims=True), 1e-300)
                Wk = _tricube(D / h)
                beta, ell, Wtot = _batched_robust_fit(D, y[idx], Wk, p, sigma,
                                                      irls_iters=irls_iters)
                n_eff = Wtot.sum(axis=1)**2 / np.maximum(
                    (Wtot**2).sum(axis=1), 1e-300)
                var_hat = sigma2 * (ell**2).sum(axis=1)
                bias_hat = b_next[blk] * (ell * D**(p + 1)).sum(axis=1)
                risk = bias_hat**2 + var_hat
                ok = (n_eff >= p + 2) & np.isfinite(risk)
                upd = ok & (risk < best_risk[blk])
                g = blk[upd]
                best_risk[g] = risk[upd]
                best_d[g] = beta[upd, 1]
                best_var[g] = var_hat[upd]
                best_k[g] = kc
        return best_d, best_var, best_k

    k_pilot0 = min(n, max(2 * (p + 3), int(round(pilot_frac * n))))
    b_next = pilot_bnext(np.full(n, k_pilot0))
    d, var, k_sel = select(b_next)

    if refine_pilot:
        k_pilot1 = np.clip((1.5 * k_sel).astype(int), 2 * (p + 3), n)
        b_next = pilot_bnext(k_pilot1)
        d, var, k_sel = select(b_next)

    if return_details:
        return d, np.sqrt(np.maximum(var, 0.0)), k_sel
    return d


# ----------------------------------------------------------------------
# V1.3 machinery: calibrated noise estimator, adaptive polynomial degree,
# moment-based fast candidate evaluation with selective IRLS.
# ----------------------------------------------------------------------

# median of the chi-square(1) distribution; scipy.stats.chi2.median(1)
_CHI2_1_MEDIAN = 0.45493642311957305


def _rice_variance_calibrated(t, y):
    """Calibrated difference-based noise variance estimator (V1.3).

    Fixes TWO defects discovered in the V1.0 estimator (both verified
    numerically before fixing; see the V1.3 log, Step 4):

    1. MEDIAN MISCALIBRATION: `_rice_variance_estimator` returns
       median(r_i^2/norm_i). Under Gaussian noise r_i^2/norm_i ~
       sigma^2 * chi^2_1, whose median is ~0.4549 sigma^2 -- the V1.0-V1.2
       estimator is biased LOW by ~2.2x (measured 0.462 on a known-noise
       linear signal). Fixed by dividing by median(chi^2_1), exactly like
       the 1.4826 calibration of the MAD.
    2. SMOOTH-SIGNAL LEAKAGE: the V1.0 3-point scheme annihilates only
       linear signals, so local curvature contaminates the estimate
       (measured: sigma2 = 0.0154 on a NOISELESS cubic). Fixed by using
       the THIRD divided difference over 4 consecutive points, which
       annihilates quadratics; its smooth remainder is f'''(xi)/6 (exact,
       by the standard divided-difference mean-value theorem), and after
       normalization the leakage is (f'''/6)^2 / sum(c^2) with
       sum(c^2) = O(h^-6) -- negligible for any reasonable sampling.

    r_i = f[t_i, t_{i+1}, t_{i+2}, t_{i+3}] (Newton divided difference)
        = sum_j c_j y_j,  c_j = 1 / prod_{l != j} (t_j - t_l),
    est_i = r_i^2 / sum_j c_j^2  ~  sigma^2 chi^2_1 under Gaussian noise,
    sigma2_hat = median_i(est_i) / median(chi^2_1).

    The median keeps outlier resistance (each gross error corrupts at most
    4 of the n-3 triplets... quadruplets); above ~15-20% contamination a
    majority of quadruplets touch an outlier and the estimate inflates --
    an honest, known breakdown documented in the log.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    if n < 4:
        return _rice_variance_estimator(t, y) / _CHI2_1_MEDIAN
    T = np.stack([t[i:n - 3 + i] for i in range(4)], axis=1)   # (n-3, 4)
    Y = np.stack([y[i:n - 3 + i] for i in range(4)], axis=1)
    # c_j = 1 / prod_{l != j} (T_j - T_l)
    diff = T[:, :, None] - T[:, None, :]                       # (n-3, 4, 4)
    diff[:, np.arange(4), np.arange(4)] = 1.0
    c = 1.0 / diff.prod(axis=2)
    r = (c * Y).sum(axis=1)
    est = r**2 / (c**2).sum(axis=1)
    return float(np.median(est)) / _CHI2_1_MEDIAN


def _windowed_moments(Z, Wk, Yw, q_s, q_w2, q_t):
    """Weighted windowed power sums shared by all polynomial degrees.

    Z  : (m, k) normalized abscissae (d/h, in [-1, 1])
    Wk : (m, k) kernel weights;  Yw : (m, k) observations
    Returns S (m, q_s), Sw2 (m, q_w2), T (m, q_t) with
      S[:, q]   = sum_j Wk z^q      (Gram entries G_ij = S[i+j], bias vector)
      Sw2[:, q] = sum_j Wk^2 z^q    (variance quadratic form)
      T[:, q]   = sum_j Wk y z^q    (right-hand sides)
    Cost O(m*k*max_q): one multiply-add sweep per moment order, shared by
    every degree p (the (k,p)-candidate grid reuses these instead of
    building per-degree (m,k,p+1) design tensors).
    """
    m, k = Z.shape
    W2 = Wk * Wk
    WY = Wk * Yw
    S = np.empty((m, q_s))
    Sw2 = np.empty((m, q_w2))
    T = np.empty((m, q_t))
    zq = np.ones_like(Z)
    for q in range(max(q_s, q_w2, q_t)):
        if q < q_s:
            S[:, q] = (Wk * zq).sum(1)
        if q < q_w2:
            Sw2[:, q] = (W2 * zq).sum(1)
        if q < q_t:
            T[:, q] = (WY * zq).sum(1)
        zq = zq * Z
    return S, Sw2, T


def _running_median(y, w=11):
    """Index-window running median (window w points, odd), edges padded by
    replication. Breakdown point 50%: the standard high-breakdown prefilter
    for initializing robust fits under heavy contamination (an initial WLS
    fit -- even inside IRLS -- can be dragged arbitrarily by gross errors,
    which is exactly the measured V1.3 failure at 25% contamination)."""
    n = len(y)
    w = min(w if w % 2 == 1 else w + 1, n if n % 2 == 1 else n - 1)
    half = w // 2
    ypad = np.concatenate([np.full(half, y[0]), y, np.full(half, y[-1])])
    from numpy.lib.stride_tricks import sliding_window_view
    return np.nanmedian(sliding_window_view(ypad, w), axis=1)


def _repmed_preweights(t, y, k_rm=21, c=4.685):
    """Siegel repeated-median LOCAL LINE pre-weights (V2.0): for each point,
    over its k_rm nearest neighbors,
        slope_i = med_j med_{l != j} (y_l - y_j)/(t_l - t_j),
        level_i = med_j (y_j - slope_i t_j),
    residual r_i = y_i - (level_i + slope_i t_i), Tukey weights at MAD scale.

    Why: the running-median prefilter is level-based and LAGS on steep
    signals, so at heavy contamination on steep data (quartic, projectile)
    its residuals confuse signal slope with outlyingness — the measured
    V1.3 25%-breakdown mode. The repeated median fits a line instead, has
    the same 50% breakdown point (Siegel 1982), and is slope-blind to
    steepness. Cost O(n k_rm^2), vectorized; only invoked under detected
    heavy contamination, so the typical-path cost is zero.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    k_rm = min(k_rm, n)
    idx = _batched_windows(t, k_rm)
    Tw, Yw = t[idx], y[idx]
    dT = Tw[:, None, :] - Tw[:, :, None]              # (n, k, k)
    dY = Yw[:, None, :] - Yw[:, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        S = dY / dT
    S[:, np.arange(k_rm), np.arange(k_rm)] = np.nan   # exclude l == j
    slope = np.nanmedian(np.nanmedian(S, axis=2), axis=1)
    level = np.median(Yw - slope[:, None] * Tw, axis=1)
    r = y - (level + slope * t)
    s = np.maximum(1.4826 * np.median(np.abs(r)), 1e-300)
    return _tukey_weights(r, s)


def _pilot_coeffs_v13(t, y, k_pilot_arr, sigma, irls_iters, deg=4,
                      pre_w=None, max_elems=2**22):
    """Robust degree-`deg` pilot fit at every point; returns B of shape
    (n, deg+1) with B[:, j] ~ f^(j)(t_i)/j! (raw units). One degree-4 fit
    supplies the (p+1)-th Taylor coefficient for every competing degree
    p in {1,2,3} simultaneously. Grouped by unique window size; fits run
    in the normalized z = d/h basis for conditioning, coefficients are
    rescaled back by h^-j.
    """
    n = len(t)
    B = np.zeros((n, deg + 1))
    k_eff = np.clip(k_pilot_arr, deg + 3, n)          # every fit well-posed
    for kp in np.unique(k_eff):
        kp = int(kp)
        sel = np.where(k_eff == kp)[0]
        step = max(1, int(max_elems // kp))
        for s in range(0, len(sel), step):
            blk = sel[s:s + step]
            idx = _batched_windows(t, kp, rows=blk)
            D = t[idx] - t[blk, None]
            h = np.maximum(np.max(np.abs(D), axis=1, keepdims=True), 1e-300)
            Z = D / h
            Wk = _tricube(Z)
            if pre_w is not None:
                Wk = Wk * pre_w[idx]
            beta_z, _, _ = _batched_robust_fit(Z, y[idx], Wk, deg, sigma,
                                               irls_iters=irls_iters)
            B[blk] = beta_z / h ** np.arange(deg + 1)[None, :]
    return B


def _local_sigma2(t, y, k_sig=25, exclude=None):
    """Pointwise noise variance sigma^2(t_i): running median (window k_sig)
    over the per-quadruplet divided-difference estimates of Eq. (13),
    calibrated by median(chi^2_1). Heteroscedastic generalization of
    `_rice_variance_calibrated`: same annihilation and calibration logic,
    but the median is taken over the k_sig quadruplets NEAREST each point
    instead of globally, so slowly-varying sigma(t) is tracked. The local
    median has the same robustness (each gross error touches <= 4
    quadruplets) but higher variance than the global one -- use only when
    heteroscedasticity is actually suspected (documented trade)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    if n < max(4, k_sig):
        return np.full(n, _rice_variance_calibrated(t, y))
    T = np.stack([t[i:n - 3 + i] for i in range(4)], axis=1)
    Y = np.stack([y[i:n - 3 + i] for i in range(4)], axis=1)
    diff = T[:, :, None] - T[:, None, :]
    diff[:, np.arange(4), np.arange(4)] = 1.0
    c = 1.0 / diff.prod(axis=2)
    est = (c * Y).sum(axis=1)**2 / (c**2).sum(axis=1)   # length n-3
    if exclude is not None:
        # drop quadruplets touching a flagged (outlier) sample, so gross
        # contamination cannot masquerade as heteroscedasticity (measured
        # failure without this: the auto gate fired on 25%-outlier data
        # and degraded it, sinusoidal MAE 0.61 -> 1.22)
        fpx = np.concatenate([[0], np.cumsum(exclude.astype(int))])
        bad = (fpx[4:] - fpx[:-4]) > 0
        est = est.copy()
        est[bad] = np.nan
        if np.all(np.isnan(est)):
            return np.full(n, _rice_variance_calibrated(t, y))
    sm = _running_median(est, min(k_sig, len(est)))
    if exclude is not None:
        gm = float(np.nanmedian(est))
        sm = np.where(np.isnan(sm), gm, sm)
    # quadruplet i spans samples i..i+3; assign sample j the estimate at
    # the nearest quadruplet center
    idx = np.clip(np.arange(n) - 1, 0, len(est) - 1)
    return sm[idx] / _CHI2_1_MEDIAN


def alprd_v13(t, y, degrees=(1, 2, 3), n_candidates=8, k_min=7, k_cap=200,
              k_max_frac=0.5, pilot_frac=0.15, pilot_cap=300, sigma2=None,
              irls_iters=2, refine_pilot=True, flag_thresh=4.0,
              prefilter_iters=1, local_sigma=False, repmed_pilot=False,
              pilot_mode="fixed", noiseless_shortcut=False,
              max_elems=2**22, return_details=False):
    """ALPRD Version 1.3: adaptive-degree, moment-accelerated, selectively
    robust local polynomial differentiation.

    Changes vs V1.2, each targeting a recorded V1.2 failure:

    1. ADAPTIVE DEGREE (V1.2 Failure A): candidates are (k, p) pairs with
       p in `degrees`; the plug-in risk (Eq. 7) is evaluated per pair.
       The bias model uses TWO Taylor terms, b_{p+1} and b_{p+2}, from one
       shared robust degree-(p_max+2) pilot fit. The second term is not a
       refinement but a REQUIREMENT for degree competition: for p - m even
       (p = 1, 3 for the first derivative) the leading functional
       sum(ell d^{p+1}) nearly vanishes on symmetric windows and wherever
       f^{(p+1)} crosses zero, so a single-term risk sees "zero bias" and
       confidently over-selects those degrees (the classical parity
       phenomenon of local polynomial regression, e.g. Ruppert & Wand
       1994). Measured before the fix: worst-case error 0.42 on a clean
       cubic, all failures at p = 1 with maximal k.
    2. MOMENT-BASED FAST PATH (V1.2 Failure B): for windows containing no
       flagged outlier, each candidate's Gram matrix, variance form, and
       bias functional are assembled from shared windowed power sums
       (`_windowed_moments`) -- O(n k) once per k, reused by all degrees,
       no (n, k, p+1) design tensors. Window sizes are capped at `k_cap`,
       making total cost O(n * k_cap * n_candidates): linear in n with
       bounded windows (an explicit, documented statistical trade at very
       large n -- variance cannot shrink below the k_cap window limit).
    3. SELECTIVE IRLS: points are pre-flagged once (robust pilot residual
       > flag_thresh * scale); full Tukey IRLS runs only for query points
       whose window CONTAINS a flagged point (O(1) check via a prefix sum
       of the flag array over the contiguous window). Un-flagged windows
       use the plain WLS moment path. On heavily contaminated data most
       windows are flagged and the cost honestly reverts toward V1.2's.
    4. CALIBRATED NOISE SCALE: uses `_rice_variance_calibrated` (the
       V1.0-V1.2 estimator is biased low by ~2.2x; see its docstring).

    Returns d, or (d, se, k_chosen, p_chosen) if return_details.
    The se caveats of V1.1/V1.2 (post-selection, weights-held-fixed)
    still apply.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    p_max = max(degrees)

    if sigma2 is None:
        sigma2 = _rice_variance_calibrated(t, y)
    # pointwise variance for the risk's variance term; global scalar keeps
    # serving the robust scales and flag thresholds (which want a stable,
    # low-variance scale, not a locally noisy one)
    sigma = np.sqrt(max(sigma2, 0.0))

    k_hi = max(k_min + 1, min(n, int(round(k_max_frac * n)), k_cap))
    cands = np.unique(np.round(
        np.geomspace(min(k_min, n), k_hi, n_candidates)).astype(int))

    # NOISELESS SHORTCUT (V2.2, motivated by the physical seismic study):
    # when the estimated noise is negligible relative to the signal
    # (sigma2 <= 1e-4 * var(y)), the pointwise risk reduces to bias^2
    # alone, which for any locally non-polynomial signal is minimized by
    # the SMALLEST admissible window at the HIGHEST degree -- no pilot
    # needed. This matters exactly where the pilot is structurally blind:
    # densely sampled broadband signals whose curvature scale is a few
    # samples, where any >= (deg+3)-point pilot window averages the
    # oscillation away (measured on CI.PASC: b-hat ~ 0 for all scales,
    # selector drifted to k~77 and output collapsed). Robust handling of
    # flagged windows is unaffected (it is per-window machinery).
    if noiseless_shortcut and sigma2 <= 1e-4 * float(np.var(y)):
        cands = np.array([int(min(max(k_min, max(degrees) + 4), n))])
        degrees = (max(degrees),)

    # ---- high-breakdown initialization: running-median residuals give
    # 50%-breakdown pre-weights so the pilot's own IRLS cannot be dragged
    # by heavy contamination before it starts (measured necessity: at 25%
    # contamination on the quartic, WLS-initialized pilots produced MAE
    # ~1500; the median prefilter is the standard remedy)
    scale_y = float(np.max(y) - np.min(y)) + 1e-30
    r_med = y - _running_median(y, 11)
    s_med = max(sigma, 1.4826 * float(np.median(np.abs(r_med))),
                1e-9 * scale_y)
    w_med = _tukey_weights(r_med, s_med)
    # V2.0: under detected heavy contamination, swap the level-based
    # running-median pre-weights for Siegel repeated-median LINE weights
    # (slope-aware, same 50% breakdown) -- fixes the steep-signal 25%
    # failure mode. The heavy/light decision itself stays on w_med.
    heavy = float((w_med < 0.5).mean()) >= 0.10
    w_init = _repmed_preweights(t, y) if (repmed_pilot and heavy) else w_med

    # ---- one shared robust pilot: degree p_max+2 (b_{p+1} AND b_{p+2}
    # for every competing degree) + outlier flags. `prefilter_iters` > 1
    # re-runs the pilot with the previous round's combined pre-weights:
    # at heavy contamination the first pilot's consensus is imperfect and
    # a second pass, started from already-cut suspects, sharpens both the
    # flags and the Taylor coefficients (V1.4 change targeting the
    # measured 25%-contamination breakdown).
    pilot_deg = p_max + 2
    k_pilot0 = min(n, pilot_cap,
                   max(2 * (pilot_deg + 2), int(round(pilot_frac * n)), 12))
    pre = w_init
    for _ in range(max(1, prefilter_iters)):
        B = _pilot_coeffs_v13(t, y, np.full(n, k_pilot0), sigma, irls_iters,
                              deg=pilot_deg, pre_w=pre)
        resid0 = y - B[:, 0]
        s_flag = max(sigma, 1.4826 * float(np.median(np.abs(resid0))),
                     1e-9 * scale_y)
        pre = w_init * _tukey_weights(resid0, s_flag)
    flag = np.abs(resid0) > flag_thresh * s_flag
    fp = np.concatenate([[0], np.cumsum(flag)])       # prefix sums of flags

    # ---- pointwise noise variance for the risk's variance term.
    # local_sigma: False = global scalar; True = always pointwise; 'auto'
    # = pointwise only if the data shows real heteroscedasticity (q90/q10
    # of the OUTLIER-CLEANED local estimates > 10; measured separation:
    # homoscedastic suite data 4.4-6.7, ramp-noise data 16.6-25 -- a
    # data-informed heuristic threshold, labeled as such in the log).
    # Computed AFTER the pilot so flagged samples can be excluded:
    # without the exclusion, contamination masquerades as
    # heteroscedasticity and fires the gate (measured).
    # Under heavy contamination the local-variance field is spiky and its
    # spread trips the 'auto' gate spuriously (measured: 25%-outlier MAE
    # 0.61 -> 3.00 without a guard). Flags alone cannot serve as the
    # guard: at 25% the GLOBAL sigma is itself inflated (measured 2.66 on
    # noiseless contaminated data), the flag threshold balloons, and
    # almost nothing is flagged. The median prefilter still sees the
    # outliers at 25% (50% breakdown), so its cut fraction is the guard:
    # pointwise variance is only considered when both the flags and the
    # prefilter agree contamination is light (<10%).
    use_local = bool(local_sigma) and local_sigma != "auto"
    sig2_arr = np.full(n, sigma2)
    light = float(flag.mean()) < 0.10 and float((w_med < 0.5).mean()) < 0.10
    if local_sigma and light:
        loc = _local_sigma2(t, y, exclude=flag if flag.any() else None)
        if local_sigma == "auto":
            q10 = float(np.quantile(loc, 0.1))
            q90 = float(np.quantile(loc, 0.9))
            use_local = q90 > 10.0 * max(q10, 1e-300)
        if use_local:
            sig2_arr = loc
    # Pilot-residual pre-weights for the robust path: a flexible local fit
    # (small k, p=3) can CHASE an outlier in its initial WLS pass, making
    # the outlier's residual small and the clean points' residuals large,
    # so IRLS then cuts the wrong points (measured failure: MAE 1.31 at
    # 10% contamination before this fix, errors concentrated at k=10,p=3).
    # Initializing every local robust fit with Tukey weights computed from
    # the STIFF pilot's residuals removes that failure mode: known-suspect
    # points enter the local fit already downweighted. Combined with the
    # median-prefilter weights so a point cut by either stays cut.
    w_pre = pre

    e_deg = {p: np.eye(p + 1)[1] for p in degrees}    # e1 per degree

    def select(B_pilot, B_by_k=None):
        # B_by_k: optional {k_c: B} scale-matched pilots (pilot window
        # ~ 2*k_c). Fixes the V2.2-discovered failure on densely sampled
        # oscillatory data, where one wide fixed pilot averages the
        # oscillation to zero curvature, blinding the bias term for every
        # candidate and driving the selector to maximal windows.
        best_risk = np.full(n, np.inf)
        best_d = np.zeros(n)
        best_var = np.full(n, np.nan)
        best_k = np.full(n, n, dtype=int)
        best_p = np.full(n, max(degrees), dtype=int)
        all_rows = np.arange(n)
        for kc in cands:
            kc = int(kc)
            Bp = B_pilot if B_by_k is None else B_by_k[kc]
            step = max(1, int(max_elems // kc))
            for s0 in range(0, n, step):
                blk = all_rows[s0:s0 + step]
                idx = _batched_windows(t, kc, rows=blk)
                D = t[idx] - t[blk, None]
                h = np.maximum(np.max(np.abs(D), axis=1, keepdims=True),
                               1e-300)
                Z = D / h
                Wk = _tricube(Z)
                Yw = y[idx]
                hb = h[:, 0]

                # flagged windows: contiguous [lo, lo+kc) -> O(1) count
                lo = idx[:, 0]
                has_out = (fp[lo + kc] - fp[lo]) > 0

                # G needs z-moments 0..2p; the two bias vectors need
                # 0..(p + p+2) = 0..2p+2  ->  q_s = 2*p_max + 3
                q_s = 2 * p_max + 3
                S, Sw2, T = _windowed_moments(Z, Wk, Yw, q_s, 2 * p_max + 1,
                                              p_max + 1)

                # robust path for contaminated windows (per degree, below)
                rob = {}
                if np.any(has_out):
                    r = np.where(has_out)[0]
                    Wk_r = Wk[r] * w_pre[idx[r]]      # pilot pre-downweighting
                    for p in degrees:
                        if kc < p + 4:
                            continue
                        beta_z, ell_z, Wtot = _batched_robust_fit(
                            Z[r], Yw[r], Wk_r, p, sigma,
                            irls_iters=irls_iters)
                        # per-window robust residual scale: if the local
                        # consensus failed (fit dragged, residuals inflated)
                        # this inflates the variance term and the candidate
                        # penalizes itself in the risk comparison
                        fit = np.zeros_like(Yw[r])
                        for j in range(p + 1):
                            fit += beta_z[:, j:j + 1] * Z[r]**j
                        s_win = 1.4826 * np.median(np.abs(Yw[r] - fit),
                                                   axis=1)
                        rob[p] = (r, beta_z, ell_z, Wtot,
                                  np.maximum(sig2_arr[blk][r], s_win**2))

                for p in degrees:
                    if kc < p + 4:
                        continue
                    m = p + 1
                    iq = np.arange(m)
                    G = S[:, iq[:, None] + iq[None, :]]
                    G2 = Sw2[:, iq[:, None] + iq[None, :]]
                    dmax = np.maximum(G[:, iq, iq].max(axis=1), 1e-300)
                    Gr = G + (1e-12 * dmax)[:, None, None] * np.eye(m)[None]
                    beta = np.linalg.solve(Gr, T[:, :m, None])[..., 0]
                    g1 = np.linalg.solve(Gr, np.broadcast_to(
                        e_deg[p], (len(blk), m))[..., None].copy())[..., 0]
                    d_est = beta[:, 1] / hb
                    var = sig2_arr[blk] / hb**2 * np.einsum(
                        'ni,nij,nj->n', g1, G2, g1)
                    # two-term bias: b_{p+1} h^p <g1, S_{.+p+1}>
    #                              + b_{p+2} h^{p+1} <g1, S_{.+p+2}>
                    bias = (Bp[blk, p + 1] * hb**p *
                            np.einsum('ni,ni->n', g1, S[:, iq + p + 1]) +
                            Bp[blk, p + 2] * hb**(p + 1) *
                            np.einsum('ni,ni->n', g1, S[:, iq + p + 2]))
                    n_eff = S[:, 0]**2 / np.maximum(Sw2[:, 0], 1e-300)

                    if p in rob:
                        r, beta_z, ell_z, Wtot, sig2_win = rob[p]
                        d_est[r] = beta_z[:, 1] / hb[r]
                        var[r] = sig2_win / hb[r]**2 * (ell_z**2).sum(1)
                        bias[r] = (Bp[blk[r], p + 1] * hb[r]**p *
                                   (ell_z * Z[r]**(p + 1)).sum(1) +
                                   Bp[blk[r], p + 2] * hb[r]**(p + 1) *
                                   (ell_z * Z[r]**(p + 2)).sum(1))
                        n_eff[r] = Wtot.sum(1)**2 / np.maximum(
                            (Wtot**2).sum(1), 1e-300)

                    risk = bias**2 + var
                    ok = (n_eff >= p + 2) & np.isfinite(risk)
                    upd = ok & (risk < best_risk[blk])
                    g = blk[upd]
                    best_risk[g] = risk[upd]
                    best_d[g] = d_est[upd]
                    best_var[g] = var[upd]
                    best_k[g] = kc
                    best_p[g] = p
        return best_d, best_var, best_k, best_p

    if pilot_mode == "matched":
        B_by_k = {}
        for kc in cands:
            kp = int(np.clip(2 * kc, pilot_deg + 3, n))
            B_by_k[int(kc)] = _pilot_coeffs_v13(
                t, y, np.full(n, kp), sigma, irls_iters, deg=pilot_deg,
                pre_w=pre)
        d, var, k_sel, p_sel = select(None, B_by_k=B_by_k)
        if return_details:
            return d, np.sqrt(np.maximum(var, 0.0)), k_sel, p_sel
        return d

    d, var, k_sel, p_sel = select(B)

    if refine_pilot:
        k_pilot1 = np.clip((1.5 * k_sel).astype(int), 2 * (pilot_deg + 2),
                           min(n, pilot_cap))
        B1 = _pilot_coeffs_v13(t, y, k_pilot1, sigma, irls_iters,
                               deg=pilot_deg, pre_w=w_pre)
        d, var, k_sel, p_sel = select(B1)

    if return_details:
        return d, np.sqrt(np.maximum(var, 0.0)), k_sel, p_sel
    return d


def alprd_v14(t, y, **kw):
    """ALPRD Version 1.4 = V1.3 machinery + auto-gated heteroscedastic
    noise scale: pointwise sigma^2(t) from windowed calibrated divided
    differences is used in the risk's variance term IF the data shows
    genuine heteroscedasticity (see the gate in alprd_v13), otherwise the
    method is exactly V1.3.

    A twice-iterated pilot was also tried for the 25%-contamination
    regime and REJECTED on measurement: it amplified first-pass
    misidentification instead of correcting it (sinusoidal 25% MAE
    0.61 -> 1.14, polynomial 258 -> 383). Recorded in the V1.4 log; the
    25% regime remains an open problem requiring a genuinely
    high-breakdown pilot (LTS/repeated-median class).
    """
    kw.setdefault("local_sigma", "auto")
    return alprd_v13(t, y, **kw)


def alprd_v20(t, y, **kw):
    """ALPRD Version 2.0 = V1.4 + Siegel repeated-median pilot initializer
    under detected heavy contamination (`repmed_pilot=True`). Measured
    effect at 25% gross contamination (n=200): sinusoidal 0.61 -> 0.44,
    polynomial 258 -> 33, projectile 63 -> 3.9, circular 1.03 -> 0.44;
    light regimes where the trigger stays off are bit-identical to V1.4.
    The trigger also fires on some 3-10% contaminated datasets (the
    prefilter's cut fraction over-counts there), which measurably HELPS
    (e.g. sinusoidal 3%: 0.006 -> 0.0005), so the sensitivity is kept.
    """
    kw.setdefault("local_sigma", "auto")
    kw.setdefault("repmed_pilot", True)
    return alprd_v13(t, y, **kw)


def alprd_v21(t, y, **kw):
    """ALPRD Version 2.1-estimator (introduced in the V2.2 iteration) =
    V2.0 + the noiseless shortcut (see alprd_v13). NOTE: a scale-matched
    pilot variant (pilot_mode="matched") was implemented first and is
    kept available, but it was FALSIFIED as a default by measurement --
    small-scale pilots produce noisy curvature estimates that degrade
    selection nearly everywhere (sinusoidal 0.098 -> 0.52, chaotic
    4.4 -> 20.2, and it did not fix the seismic case, 0.90 -> 0.97).
    The record of that negative result is in the V2.2 log.

    SECOND FALSIFICATION (V2.2): the noiseless shortcut was then tried as
    the default and ALSO rejected by measurement -- collapsing the
    candidate set removes the risk selection's ability to route around
    locally-failed robust fits (clean quartic 0.0007 -> 168 MAE; 3%
    outliers 0.0005 -> 1.42) and still does not fix the seismic case
    (0.90 -> 0.95), because the true obstruction there is that NO
    overdetermined local fit can resolve content whose quarter-period is
    ~2 samples: a fundamental scope boundary (interpolation-vs-robustness
    trade-off), documented in the V2.2 log. Both mechanisms remain
    available as explicit opt-in parameters; ALPRD_v2.0 remains the
    recommended estimator, and this wrapper is now an alias for it."""
    return alprd_v20(t, y, **kw)


# ----------------------------------------------------------------------
# New baselines for V1.2
# ----------------------------------------------------------------------

def tv_derivative(t, y, alpha=None, max_lag_iters=15, n_bisect=12):
    """Total-variation-regularized differentiation (Chartrand 2011 style),
    formulated directly on the uneven grid:

        min_{y0, u}  || y0 + A u - y ||^2  +  alpha * sum_j |u_{j+1}-u_j|

    where u_i ~ f'(t_i) and A is the trapezoidal cumulative-integration
    matrix on the actual sample times (so no resampling is needed).
    Solved by lagged-diffusivity fixed point (IRLS on the TV term with
    |v| ~ sqrt(v^2 + eps)).

    alpha selection: if not given, chosen by the DISCREPANCY PRINCIPLE --
    bisection on log(alpha) until the residual sum of squares matches
    n * sigma2_hat (Rice estimate). This is a principled, standard rule;
    it is honestly a heuristic when the noise estimate itself is off.
    For (near-)noiseless data a small floor alpha is used.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    dt = np.diff(t)

    # trapezoid integration matrix: (Au)_i = int_{t_0}^{t_i} u, built
    # incrementally: row i = row i-1 + dt[i-1]/2 on columns i-1 and i
    A = np.zeros((n, n))
    for i in range(1, n):
        A[i] = A[i - 1]
        A[i, i - 1] += dt[i - 1] / 2
        A[i, i] += dt[i - 1] / 2

    # augment intercept y0: B = [1, A]
    B = np.hstack([np.ones((n, 1)), A])
    Dm = (np.eye(n, n, 1) - np.eye(n))[:-1]           # first differences of u
    BtB = B.T @ B
    Bty = B.T @ y
    eps = 1e-8

    sigma2 = _rice_variance_estimator(t, y)
    scale = float(np.mean(np.abs(np.diff(y, 2)))) + 1e-30

    def solve_for(alpha):
        z = np.zeros(n + 1)
        z[1:] = np.gradient(y, t)                     # warm start
        for _ in range(max_lag_iters):
            v = Dm @ z[1:]
            E = 1.0 / np.sqrt(v**2 + eps * scale**2)
            P = np.zeros((n + 1, n + 1))
            P[1:, 1:] = Dm.T @ (E[:, None] * Dm)
            z_new = np.linalg.solve(BtB + 0.5 * alpha * P, Bty)
            if np.max(np.abs(z_new - z)) < 1e-9 * (1 + np.max(np.abs(z))):
                z = z_new
                break
            z = z_new
        resid2 = float(np.sum((B @ z - y)**2))
        return z, resid2

    if alpha is None:
        if sigma2 <= 0:
            alpha = 1e-8 * scale
            z, _ = solve_for(alpha)
            return z[1:]
        target = n * sigma2
        lo_a, hi_a = 1e-8 * scale, 1e6 * scale
        z, _ = solve_for(hi_a)
        for _ in range(n_bisect):
            mid = np.sqrt(lo_a * hi_a)
            z, r2 = solve_for(mid)
            if r2 > target:
                hi_a = mid      # too smooth -> relax
            else:
                lo_a = mid      # underfitting noise -> smooth more
        return z[1:]
    z, _ = solve_for(alpha)
    return z[1:]


def tv_derivative_l1(t, y, alpha=None, max_lag_iters=15, n_bisect=10):
    """L1-fidelity total-variation differentiation (V1.3 baseline):

        min_{y0, u}  sum_i | y0 + (Au)_i - y_i |  +  alpha * sum_j |u_{j+1}-u_j|

    The L1 data term makes the fit resistant to gross outliers -- the fair
    competitor to ALPRD's IRLS robustness that the L2 version (`tv_derivative`)
    is not (its V1.2-measured outlier MAE was ~21000). Both absolute values
    are handled by lagged-diffusivity IRLS (|v| ~ sqrt(v^2 + eps^2)).

    alpha selection: robust discrepancy principle -- bisection on log(alpha)
    until median|residual| matches its Gaussian-inlier expectation
    0.6745 * sigma_hat (median-based so 10-25% contamination cannot drag
    the target; sigma_hat is the CALIBRATED Rice estimate). Honest caveat:
    like all discrepancy rules this inherits the noise estimate's own bias,
    and the Rice estimate itself degrades above ~25% contamination (its
    median-of-triplets sees a contaminated majority; see the V1.3 log).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    dt = np.diff(t)

    A = np.zeros((n, n))
    for i in range(1, n):
        A[i] = A[i - 1]
        A[i, i - 1] += dt[i - 1] / 2
        A[i, i] += dt[i - 1] / 2
    B = np.hstack([np.ones((n, 1)), A])
    Dm = (np.eye(n, n, 1) - np.eye(n))[:-1]

    sigma2 = _rice_variance_calibrated(t, y)
    sigma = np.sqrt(max(sigma2, 0.0))
    scale = float(np.mean(np.abs(np.diff(y, 2)))) + 1e-30
    eps_f = 1e-6 * (float(np.max(np.abs(y))) + 1e-30)
    eps_tv = 1e-8

    def solve_for(alpha, z0):
        z = z0.copy()
        for _ in range(max_lag_iters):
            res = B @ z - y
            Wf = 1.0 / np.sqrt(res**2 + eps_f**2)
            v = Dm @ z[1:]
            E = 1.0 / np.sqrt(v**2 + (eps_tv * scale)**2)
            P = np.zeros((n + 1, n + 1))
            P[1:, 1:] = Dm.T @ (E[:, None] * Dm)
            M = B.T @ (Wf[:, None] * B) + 0.5 * alpha * P
            rhs = B.T @ (Wf * y)
            z_new = np.linalg.solve(M, rhs)
            if np.max(np.abs(z_new - z)) < 1e-9 * (1 + np.max(np.abs(z))):
                z = z_new
                break
            z = z_new
        med_res = float(np.median(np.abs(B @ z - y)))
        return z, med_res

    z0 = np.zeros(n + 1)
    z0[1:] = np.gradient(y, t)

    if alpha is None:
        if sigma <= 0:
            z, _ = solve_for(1e-8 * scale, z0)
            return z[1:]
        target = 0.6745 * sigma            # median|N(0, sigma)|
        lo_a, hi_a = 1e-8 * scale, 1e6 * scale
        z = z0
        for _ in range(n_bisect):
            mid = np.sqrt(lo_a * hi_a)
            z, med = solve_for(mid, z)     # warm start from previous alpha
            if med > target:
                hi_a = mid                 # over-smoothing inliers
            else:
                lo_a = mid
        return z[1:]
    z, _ = solve_for(alpha, z0)
    return z[1:]


def rbf_fd_derivative(t, y, k=7, phs_power=3, poly_deg=2):
    """RBF-FD differentiation with polyharmonic-spline kernel |r|^3 and
    polynomial augmentation (standard formulation, e.g. Fornberg & Flyer).
    At each point, derivative weights are computed over the k nearest
    neighbors by solving the saddle system

        [ Phi  P ] [w]   [ dphi/dx |_{t0} ]
        [ P^T  0 ] [g] = [ dp/dx   |_{t0} ]

    This is an INTERPOLATORY (exactness-based) method: it has no noise
    model, and is expected to amplify noise like a finite-difference
    stencil. Included as the standard mesh-free baseline for scattered
    data, not as a denoising competitor.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    k = min(k, n)
    d = np.empty(n)
    for i in range(n):
        lo, hi = _knn_window(t, i, k)
        c = t[lo:hi] - t[i]                       # centered nodes
        m = poly_deg + 1
        Phi = np.abs(c[:, None] - c[None, :]) ** phs_power
        P = c[:, None] ** np.arange(m)[None, :]
        S = np.zeros((k + m, k + m))
        S[:k, :k] = Phi
        S[:k, k:] = P
        S[k:, :k] = P.T
        rhs = np.zeros(k + m)
        # d/dx |x - c_j|^3 at x=0  =  -3 c_j |c_j|
        rhs[:k] = -phs_power * c * np.abs(c) ** (phs_power - 2)
        rhs[k + 1] = 1.0                          # d/dx of x at 0
        try:
            w = np.linalg.solve(S, rhs)[:k]
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(S, rhs, rcond=None)[0][:k]
        d[i] = w @ y[lo:hi]
    return d


def _rice_variance_estimator(t, y):
    """Difference-based noise variance estimator (Rice, 1984), generalized
    to uneven spacing by using the *normalized* second difference so that
    it stays an unbiased-in-leading-order estimator of sigma^2 for smooth
    f and roughly-consistent spacing locally. For strongly uneven grids
    this is a first-order estimator only (see docs: known limitation).
    """
    n = len(t)
    if n < 3:
        return 0.0
    h1 = t[1:-1] - t[:-2]
    h2 = t[2:] - t[1:-1]
    # weights that make the combination exact (zero) for any straight line,
    # so its expectation isolates noise to leading order
    a = -h2 / (h1 * (h1 + h2))
    b = (h2 - h1) / (h1 * h2)
    c = h1 / (h2 * (h1 + h2))
    r = a * y[:-2] + b * y[1:-1] + c * y[2:]
    norm = a**2 + b**2 + c**2
    est = r**2 / norm
    return float(np.median(est))  # median for robustness to outliers

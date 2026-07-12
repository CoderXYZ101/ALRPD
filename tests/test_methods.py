"""
Correctness tests for the V1.1 changes.

Run with:  python tests/test_methods.py   (plain asserts, no pytest needed)
       or: python -m pytest tests/ -q     (if pytest is installed)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from methods import (
    alprd_v1, alprd_v1_fast, alprd_v11, savitzky_golay_matched,
    _knn_window, central_difference,
    alprd_v12, _batched_windows, tv_derivative, rbf_fd_derivative,
    alprd_v13, tv_derivative_l1, _rice_variance_calibrated,
)
from datasets import make_dataset


def test_knn_window_matches_argsort():
    """The two-pointer window must select a point set whose k-th neighbor
    distance equals the argsort-based one, for uneven grids and every k."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0, 10, 60))
    for i in [0, 1, 17, 30, 58, 59]:
        dist = np.abs(t - t[i])
        order = np.argsort(dist)
        for k in [3, 5, 10, 25, 60]:
            lo, hi = _knn_window(t, i, k)
            assert hi - lo == min(k, len(t)), (i, k, lo, hi)
            h_fast = np.max(np.abs(t[lo:hi] - t[i]))
            h_ref = dist[order[min(k, len(t)) - 1]]
            assert np.isclose(h_fast, h_ref), (i, k, h_fast, h_ref)


def test_fast_matches_v1_exactly():
    """alprd_v1_fast must reproduce alprd_v1 to machine precision on both
    clean and noisy uneven data, including the variance channel."""
    for noise in ["none", "gaussian"]:
        ds = make_dataset(kind="sinusoidal", n=120, uneven=True,
                          noise=noise, noise_level=0.05, seed=99)
        t, y = ds["t"], ds["y"]
        for span in [0.1, 0.25, 0.4]:
            d_ref, se_ref = alprd_v1(t, y, poly_order=2, span=span,
                                     return_variance=True)
            d_fast, se_fast = alprd_v1_fast(t, y, poly_order=2, span=span,
                                            return_variance=True)
            assert np.allclose(d_ref, d_fast, rtol=1e-10, atol=1e-12), \
                f"derivative mismatch (noise={noise}, span={span})"
            assert np.allclose(se_ref, se_fast, rtol=1e-8, atol=1e-12), \
                f"se mismatch (noise={noise}, span={span})"


def test_polynomial_reproduction():
    """Exactness on polynomials of degree <= p (the moment identity, Eq. 5):
    with clean data, both ALPRD versions must recover the derivative of a
    quadratic to near machine precision at interior AND boundary points."""
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 5, 80))
    y = 1.5 * t**2 - 2.0 * t + 0.7
    d_true = 3.0 * t - 2.0
    for fn in [lambda: alprd_v1(t, y, poly_order=2, span=0.3),
               lambda: alprd_v1_fast(t, y, poly_order=2, span=0.3),
               lambda: alprd_v11(t, y, poly_order=2)]:
        d = fn()
        assert np.max(np.abs(d - d_true)) < 1e-7, np.max(np.abs(d - d_true))


def test_v11_beats_fixed_span_on_chaotic():
    """Regression guard for V1.0 Failure 2: on noisy chaotic data the
    adaptive bandwidth must at least substantially improve on the fixed
    span. (Threshold deliberately loose: 2x, not a tuned number.)"""
    ds = make_dataset(kind="chaotic", n=300, t_max=8.0, uneven=True,
                      noise="gaussian", noise_level=0.02, seed=12345)
    t, y, d_true = ds["t"], ds["y"], ds["dydt_true"]
    mae_v1 = np.mean(np.abs(alprd_v1(t, y, poly_order=2, span=0.25) - d_true))
    mae_v11 = np.mean(np.abs(alprd_v11(t, y, poly_order=2) - d_true))
    assert mae_v11 < mae_v1 / 2, (mae_v1, mae_v11)


def test_sg_matched_window_size():
    ds = make_dataset(kind="sinusoidal", n=200, uneven=True,
                      noise="gaussian", noise_level=0.05)
    d = savitzky_golay_matched(ds["t"], ds["y"], span=0.25)
    assert d.shape == ds["t"].shape
    assert np.all(np.isfinite(d))


def test_batched_windows_match():
    """The batched window selector must give the same k-th neighbor
    distance (hence identical tricube weights) as the sequential
    two-pointer version, for every point and several k, on uneven grids."""
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(0, 10, 73))
    for k in [3, 7, 20, 50, 73]:
        idx = _batched_windows(t, k)
        for i in range(len(t)):
            lo, hi = _knn_window(t, i, k)
            h_seq = np.max(np.abs(t[lo:hi] - t[i]))
            h_bat = np.max(np.abs(t[idx[i]] - t[i]))
            assert np.isclose(h_seq, h_bat), (i, k, h_seq, h_bat)


def test_v12_polynomial_reproduction():
    """V1.2 (robust path on) must still recover a quadratic's derivative
    to high precision on clean data — the IRLS scale floor must not
    zero out machine-epsilon residuals."""
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 5, 80))
    y = 1.5 * t**2 - 2.0 * t + 0.7
    d_true = 3.0 * t - 2.0
    d = alprd_v12(t, y, poly_order=2)
    assert np.max(np.abs(d - d_true)) < 1e-6, np.max(np.abs(d - d_true))


def test_v12_fixes_outlier_regression():
    """Regression guard for V1.1 Failure 1': on outlier-contaminated data
    V1.2 must be at least 3x better than V1.1 in MAE, and no worse than
    1.5x ALPRD v1.0 (the wide-window incumbent)."""
    ds = make_dataset(kind="sinusoidal", n=200, uneven=True,
                      noise="outliers", noise_level=0.0, seed=12345)
    t, y, d_true = ds["t"], ds["y"], ds["dydt_true"]
    mae = lambda d: np.mean(np.abs(d - d_true))
    m_v10 = mae(alprd_v1(t, y, poly_order=2, span=0.25))
    m_v11 = mae(alprd_v11(t, y, poly_order=2))
    m_v12 = mae(alprd_v12(t, y, poly_order=2))
    assert m_v12 < m_v11 / 3, (m_v11, m_v12)
    assert m_v12 < 1.5 * m_v10, (m_v10, m_v12)


def test_v12_keeps_chaotic_gains():
    """V1.2 must not give back V1.1's chaotic improvement."""
    ds = make_dataset(kind="chaotic", n=300, t_max=8.0, uneven=True,
                      noise="gaussian", noise_level=0.02, seed=12345)
    t, y, d_true = ds["t"], ds["y"], ds["dydt_true"]
    mae = lambda d: np.mean(np.abs(d - d_true))
    m_v1 = mae(alprd_v1(t, y, poly_order=2, span=0.25))
    m_v12 = mae(alprd_v12(t, y, poly_order=2))
    assert m_v12 < m_v1 / 2, (m_v1, m_v12)


def test_tv_and_rbf_sane():
    """New baselines: finite output; TV beats raw central difference on
    noisy data (that is its whole job); RBF-FD is near-exact on clean
    smooth data (it is an exactness-based stencil method)."""
    ds = make_dataset(kind="sinusoidal", n=150, uneven=True,
                      noise="gaussian", noise_level=0.05, seed=5)
    t, y, d_true = ds["t"], ds["y"], ds["dydt_true"]
    d_tv = tv_derivative(t, y)
    assert np.all(np.isfinite(d_tv))
    mae_tv = np.mean(np.abs(d_tv - d_true))
    mae_cd = np.mean(np.abs(central_difference(t, y) - d_true))
    assert mae_tv < mae_cd, (mae_tv, mae_cd)

    ds0 = make_dataset(kind="sinusoidal", n=150, uneven=True,
                       noise="none", noise_level=0.0, seed=5)
    d_rbf = rbf_fd_derivative(ds0["t"], ds0["y"])
    err = np.abs(d_rbf - ds0["dydt_true"])
    assert np.all(np.isfinite(err))
    # interior points: near-exact; boundary stencils are one-sided and
    # measurably worse (expected for stencil methods, not asserted tightly)
    assert np.mean(err[5:-5]) < 1e-3, np.mean(err[5:-5])


def test_rice_calibrated():
    """The V1.3 noise estimator must (a) be approximately unbiased under
    Gaussian noise on a linear signal (the V1.0 estimator returned ~0.46x),
    and (b) not leak smooth-signal curvature (the V1.0 3-point scheme
    returned 0.0154 on a NOISELESS cubic)."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 10, 5000))
    y = 2.0 * t + rng.normal(0, 0.1, 5000)
    est = _rice_variance_calibrated(t, y)
    assert 0.008 < est < 0.0125, est          # true sigma2 = 0.01

    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 5, 80))
    y = 0.5 * t**3 - 1.5 * t**2 + 2 * t       # noiseless cubic
    assert _rice_variance_calibrated(t, y) < 1e-6


def test_v13_cubic_exactness():
    """With p=3 among the competing degrees, a clean cubic's derivative
    must be recovered near-exactly (two-term bias risk must not let the
    parity blind spot select a lower degree; measured pre-fix error 0.42)."""
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 5, 80))
    y = 0.5 * t**3 - 1.5 * t**2 + 2 * t
    d_true = 1.5 * t**2 - 3 * t + 2
    assert np.max(np.abs(alprd_v13(t, y) - d_true)) < 1e-6


def test_v13_contamination_sweep():
    """Regression guards from measured V1.3 results: at 3% and 10% gross
    contamination V1.3 must stay far below V1.2 (measured 0.006/0.011 vs
    0.09/0.106); at 25% it must at least not be worse than V1.2
    (breakdown region, measured 0.61 vs 2.29). Loose factors so noise in
    future refactors doesn't flap the test."""
    for frac, cap in [(0.03, 0.05), (0.10, 0.06), (0.25, 2.0)]:
        ds = make_dataset(kind="sinusoidal", n=200, uneven=True,
                          noise="outliers", noise_level=0.0,
                          outlier_frac=frac, seed=12345)
        m = np.mean(np.abs(alprd_v13(ds["t"], ds["y"]) - ds["dydt_true"]))
        assert m < cap, (frac, m)


def test_v13_chaotic_improves_on_v12():
    """The adaptive degree + calibrated sigma must keep improving the
    chaotic case (V1.2 measured 8.15; V1.3 measured 4.44)."""
    ds = make_dataset(kind="chaotic", n=200, uneven=True,
                      noise="gaussian", noise_level=0.05, seed=12345)
    m13 = np.mean(np.abs(alprd_v13(ds["t"], ds["y"]) - ds["dydt_true"]))
    m12 = np.mean(np.abs(alprd_v12(ds["t"], ds["y"]) - ds["dydt_true"]))
    assert m13 < m12, (m13, m12)


def test_tv_l1_outlier_robust():
    """The L1-fidelity TV variant must be dramatically better than the L2
    version under contamination (that is its purpose)."""
    ds = make_dataset(kind="sinusoidal", n=200, uneven=True,
                      noise="outliers", noise_level=0.0, outlier_frac=0.10,
                      seed=12345)
    m_l1 = np.mean(np.abs(tv_derivative_l1(ds["t"], ds["y"]) - ds["dydt_true"]))
    m_l2 = np.mean(np.abs(tv_derivative(ds["t"], ds["y"]) - ds["dydt_true"]))
    assert m_l1 < m_l2 / 10, (m_l1, m_l2)


def test_v14_hetero_gate():
    """V1.4's auto-gated local variance must (a) improve on V1.3 for
    genuinely heteroscedastic noise, and (b) leave homoscedastic and
    contaminated data EXACTLY unchanged (gate closed)."""
    from methods import alprd_v14
    ds = make_dataset(kind="sinusoidal", n=200, uneven=True,
                      noise="gaussian_ramp", noise_level=0.08, seed=12345)
    m13 = np.mean(np.abs(alprd_v13(ds["t"], ds["y"]) - ds["dydt_true"]))
    m14 = np.mean(np.abs(alprd_v14(ds["t"], ds["y"]) - ds["dydt_true"]))
    assert m14 < m13, (m13, m14)

    for kw in [dict(noise="gaussian", noise_level=0.05),
               dict(noise="outliers", noise_level=0.0, outlier_frac=0.25)]:
        ds = make_dataset(kind="sinusoidal", n=200, uneven=True,
                          seed=12345, **kw)
        d13 = alprd_v13(ds["t"], ds["y"])
        d14 = alprd_v14(ds["t"], ds["y"])
        assert np.array_equal(d13, d14), kw


def test_v20_repmed_pilot():
    """V2.0 guards from measured results: repeated-median pilot must hold
    the 25%-contamination gains (sinusoidal 0.44, polynomial 33 measured;
    loose caps) and leave light-contamination Gaussian data bit-identical
    to V1.4 (trigger off)."""
    from methods import alprd_v14, alprd_v20
    ds = make_dataset(kind="sinusoidal", n=200, uneven=True,
                      noise="outliers", noise_level=0.0, outlier_frac=0.25,
                      seed=12345)
    m = np.mean(np.abs(alprd_v20(ds["t"], ds["y"]) - ds["dydt_true"]))
    assert m < 1.5, m
    ds = make_dataset(kind="polynomial", n=200, uneven=True,
                      noise="outliers", noise_level=0.0, outlier_frac=0.25,
                      seed=12345)
    m = np.mean(np.abs(alprd_v20(ds["t"], ds["y"]) - ds["dydt_true"]))
    assert m < 100, m
    ds = make_dataset(kind="sinusoidal", n=200, uneven=True,
                      noise="gaussian", noise_level=0.05, seed=12345)
    assert np.array_equal(alprd_v14(ds["t"], ds["y"]),
                          alprd_v20(ds["t"], ds["y"]))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tests passed")

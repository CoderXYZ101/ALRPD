# ALPRD: tuning-free robust derivative estimation on irregular grids with per-point error estimates

*(JOSS-style draft, V2.2. Every number cites a committed CSV in `results/`.
Submission blockers listed at the end — this is a draft, not a submission.)*

## Summary

ALPRD estimates the first derivative of a 1-D signal from unevenly sampled,
noisy, and possibly outlier-contaminated measurements, with no user-tuned
parameters and a per-point standard-error output. At each sample it selects,
by a plug-in bias–variance risk, both a local window size and a polynomial
degree (p ∈ {1,2,3}); all fits are Tukey-robust, initialized from a
high-breakdown median/repeated-median prefilter chain; the noise scale comes
from a calibrated third-divided-difference estimator that annihilates
quadratic trend and corrects the classical median-χ² bias (a 2.2×
miscalibration we found in our own earlier versions and disclose).

## Statement of need

Practitioners differentiate noisy time series with tools that assume uniform
grids (Savitzky–Golay), require per-dataset tuning (splines, TV), or lack
uncertainty output; a recent taxonomy [arXiv:2512.09090] explicitly names
irregular sampling and pointwise error estimates as underdeveloped. ALPRD
targets exactly that gap.

## Evidence (all measured, seed 12345, committed per-run CSVs)

- **Synthetic suite** (83 datasets: 9 motion families × sampling/noise/
  contamination/heteroscedastic conditions; 17 methods; 1411 runs, 0
  failures): ALPRD best aggregate MAE 1.19; next non-ALPRD method 20.8.
- **Against external code with per-dataset ORACLE tuning** (PyNumDiff 0.2.3,
  derivative 0.6.3; a protocol deliberately favoring the externals): ALPRD
  zero-tuning wins all contamination levels by 5.5–35×; loses smooth-noise
  slices to oracle-tuned splines by 2–4×.
- **Real ephemeris data** (JPL Horizons, Moon + Mars, model-exact reference
  velocities): best on 5 of 8 variants; the only method whose worst case is
  bounded (5.2× vs ≥345× for every oracle-tuned competitor).
- **Real physical-sensor data** (CI.PASC seismic station: broadband
  seismometer velocity differentiated against an independent co-located
  accelerometer, corr 0.9999, ~2% reference floor): **a negative result we
  report prominently** — see Limitations.

## Limitations (disclosed)

1. **Near-Nyquist content is out of scope.** On the seismic record
   (0.3–8 Hz content, quarter-periods of ~2 samples), no overdetermined
   local fit can track the signal; near-interpolatory methods (smoothing
   spline, s→0) reach the ~2% instrument floor while ALPRD and all local
   competitors collapse. This is a structural trade-off: robustness
   requires local overdetermination, which forfeits near-Nyquist
   resolution. ALPRD should not be used when the curvature scale is
   ≲4 samples.
2. Oracle-tuned splines beat ALPRD 2–4× on smooth, homoscedastic,
   adequately-sampled data — the price of zero tuning and robustness.
3. Gross contamination beyond ~25% exceeds the design envelope.
4. Reported standard errors are post-selection lower bounds.
5. Theory: the oracle inequality is complete only for the non-robust path
   (Gaussian noise, f ∈ C⁶); the IRLS path has no such theorem.

## Availability

Python, numpy/scipy only; 18 seeded tests; every figure/table regenerates
from `src/benchmark.py`, `src/bench_external.py`, `src/bench_realdata.py`,
`src/bench_seismic.py` with committed raw data fetches.

## Submission blockers (honest)

(i) chapter-level check of Katkovnik et al. (SPIE PM157) still pending
(TOC-level check done); (ii) a sampling-adequacy diagnostic should be added
so the method can warn users in the near-Nyquist regime (limitation 1);
(iii) venue decision (JOSS vs DSP-class) affects required theory depth.

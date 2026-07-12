# numdiff-research

Adaptive Local Polynomial Regression Differentiation (**ALPRD**) — an ongoing,
honesty-first research project on numerical differentiation of **unevenly
sampled, noisy** 1-D data.

**Status: Phase 4 concluded at V2.2** (estimator frozen at the V2.0 formula; V2.2 added physical-sensor validation, a completed WLS-path oracle inequality, two falsified-and-reverted fixes, a documented scope boundary, and the paper draft) (evidence-based stopping
condition: all remaining improvements require external literature, external
code/data, or genuinely new mathematics — see the V1.4 log section). Current
self-assessed publication readiness: **7.5/10**. Claims are strictly
separated into Proven / Derived / Tested / Hypothesized, and negative results
are kept in the record (V1.0's outlier-robustness claim was retracted after a
control experiment; V1.0-V1.2's noise estimator was found miscalibrated by
2.2x and fixed in V1.3; two V1.4 hypotheses were tested, falsified, and
recorded).

## Layout

```
docs/RESEARCH_LOG.md      the full record: derivations, assumptions, benchmarks,
                          failure analyses, version history (V1.0, V1.1, ...)
src/methods.py            ALPRD variants + all baseline methods
src/datasets.py           benchmark dataset generators (exact ground-truth
                          derivatives, incl. a Lorenz trajectory via solve_ivp)
src/benchmark.py          the full 35-scenario suite; writes results/*.csv
tests/test_methods.py     correctness + equivalence tests (plain python, no
                          pytest required)
results/                  raw per-run CSVs and scaling measurements — every
                          number cited in the log traces to a file here
```

## Reproduce

```bash
pip install numpy scipy pandas
python tests/test_methods.py     # 5 tests, all must pass
python src/benchmark.py          # ~2 min; rewrites results/results_v1_1.csv
```

Everything is seeded (`seed=12345` throughout); re-running the suite must
reproduce the CSVs bit-for-bit on the same numpy/scipy versions
(numpy 2.5.1 / scipy 1.18.0 were used for the committed results).

## Method in one paragraph

At each query point, ALPRD fits a weighted local polynomial (tricube kernel,
degree 2) over a k-nearest-neighbor window of the *actual* sample times — no
resampling to a uniform grid — and reads the derivative off the linear
coefficient, together with a finite-sample variance estimate. V1.1 selects k
per point by minimizing an estimated Bias² + Variance (plug-in rule built
from the derived leading-order bias and exact variance functionals). V1.2
makes every fit Tukey-biweight-robust (IRLS), refines the curvature pilot at
the selected bandwidth, and vectorizes the whole selector; under 3% gross
contamination its error is now ~16x lower than any non-robust method in the
suite while remaining near-best on clean and noisy data. Known open problems,
failure modes, and the improvement queue are tracked in
`docs/RESEARCH_LOG.md` Steps 8–13 of each version.

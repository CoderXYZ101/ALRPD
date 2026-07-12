# Paper skeleton (drafted at V2.1; submit only after remaining gaps close)

**Working title:** Tuning-free robust derivative estimation on irregular grids
with per-point error estimates (ALPRD)

**Target venue class:** computational statistics / signal processing methods
(e.g., CSDA, Digital Signal Processing) or JOSS (software-first) — decide
after the novelty question is fully settled.

1. **Introduction.** Problem: derivatives from unevenly sampled, noisy,
   possibly contaminated data; practitioners lack a no-tuning method with
   uncertainty output (cite the 2025 taxonomy's explicit gaps).
2. **Method.** The V2.0 formulation (log Steps 1 of V1.3/V1.4/V2.0):
   calibrated divided-difference noise scale; median/repeated-median
   prefilter chain; robust degree-5 pilot; joint (k, p) plug-in risk with
   parity-safe two-term bias; gated pointwise variance; se output.
3. **Theory.** Exact moment identity; exact conditional variance; oracle
   lemma; sigma-hat concentration (Eq. 14, proven); pilot-coefficient
   Gaussian concentration (WLS path, proven); openly stated gaps
   (pilot bias term, post-IRLS, consistency rate).
4. **Experiments.** (a) 83-dataset synthetic suite, 17 methods, seeded;
   (b) oracle-tuned external field (PyNumDiff, derivative packages) —
   ALPRD wins all contamination levels 5.5-35x, loses smooth slices to
   oracle-tuned splines 2-4x (stated plainly); (c) real JPL ephemeris
   study — best on 5/8 variants, only method with bounded worst-case
   (5.2x vs >=345x for every competitor).
5. **Limitations.** Oracle-spline dominance on smooth homoscedastic data;
   >25% contamination; mid-n runtime; se caveats (post-selection).
6. **Reproducibility.** Full code, seeds, committed per-run CSVs, 19 tests.

**Blocking items before submission:** (i) Katkovnik book full-text check
(novelty), (ii) decide venue, (iii) second real dataset from a physical
sensor (the ephemeris reference is model-derived, not instrument-derived),
(iv) optional: finish oracle inequality for a statistics venue.

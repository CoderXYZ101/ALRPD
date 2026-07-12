"""
V2.1 real-data study: JPL Horizons ephemeris (fetched 2026-07-12, raw API
responses committed under data/ for reproducibility).

Series:
  moon : geocentric Moon X-coordinate, 2024-01-01..2024-03-31, 4 h grid
         (fast oscillatory signal, ~3.3 lunar periods)
  mars : heliocentric Mars X-coordinate, 2023-01-01..2025-01-01, 2 d grid
         (slow, gently curved signal)
Units: AU for X, AU/day for VX; time in days (JD - first epoch).

Reference derivative: JPL's VX at the same epochs. HONESTY NOTE: this is
the exact derivative of the DE-ephemeris trajectory the positions lie on
(from the fitted dynamical model), not an independent instrument -- the
right reference for judging *numerical differentiation of the sampled
positions*, and stated as such in the log.

Variants per series (seed 12345):
  clean-uneven      : random 60% subsample of epochs (uneven grid), no noise
  noisy-uneven      : + Gaussian noise, sigma = 1e-3 * std(x)
  noisy2-uneven     : + Gaussian noise, sigma = 1e-2 * std(x)
  outliers-uneven   : + 5% gross outliers (as in the synthetic suite)

Methods: ALPRD_v2.0 (zero tuning) vs the strongest external methods with
ORACLE per-dataset tuning (same protocol as bench_external.py), plus the
suite's own savgol_resampled default.
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from methods import alprd_v20, savitzky_golay_resampled
from bench_external import build_grids

SEED = 12345
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def parse_horizons(path):
    txt = open(path).read()
    body = txt.split("$$SOE")[1].split("$$EOE")[0]
    jd, x, vx = [], [], []
    for chunk in re.finditer(
            r"^(24\d+\.\d+) = .*?\n"
            r" X =\s*([-+0-9.E]+) Y =.*?\n"
            r" VX=\s*([-+0-9.E]+) VY=", body, re.M | re.S):
        jd.append(float(chunk.group(1)))
        x.append(float(chunk.group(2)))
        vx.append(float(chunk.group(3)))
    jd = np.array(jd)
    return jd - jd[0], np.array(x), np.array(vx)


def make_variants(t, x, vx, rng):
    keep = np.sort(rng.choice(len(t), size=int(0.6 * len(t)), replace=False))
    keep[0], keep[-1] = 0, len(t) - 1
    keep = np.unique(keep)
    tt, xx, vv = t[keep], x[keep], vx[keep]
    s = np.std(xx)
    out = {"clean": (tt, xx, vv)}
    out["noise1e3"] = (tt, xx + rng.normal(0, 1e-3 * s, len(tt)), vv)
    out["noise1e2"] = (tt, xx + rng.normal(0, 1e-2 * s, len(tt)), vv)
    y = xx + rng.normal(0, 1e-3 * s, len(tt))
    n_out = max(1, int(0.05 * len(tt)))
    idx = rng.choice(len(tt), size=n_out, replace=False)
    y[idx] += rng.choice([-1, 1], n_out) * rng.uniform(5, 15, n_out) * 0.5 * s
    out["outliers5pct"] = (tt, y, vv)
    return out


def main():
    rng = np.random.default_rng(SEED)
    grids = build_grids()
    externals = {k: grids[k] for k in
                 ["drv_savgol", "drv_spline", "drv_kalman", "pnd_butter",
                  "pnd_savgol", "pnd_median"]}
    rows = []
    for series, fname in [("moon", "moon_geocentric.txt"),
                          ("mars", "mars_heliocentric.txt")]:
        t, x, vx = parse_horizons(os.path.join(DATA, fname))
        for vname, (tt, yy, d_true) in make_variants(t, x, vx, rng).items():
            tag = f"{series}__{vname}"
            # ALPRD v2.0, zero tuning
            d = alprd_v20(tt, yy)
            err = d - d_true
            rows.append({"dataset": tag, "method": "ALPRD_v2.0",
                         "mae": float(np.mean(np.abs(err))),
                         "rmse": float(np.sqrt(np.mean(err**2))),
                         "max_err": float(np.max(np.abs(err))),
                         "n": len(tt)})
            # internal default baseline
            d = savitzky_golay_resampled(tt, yy)
            err = d - d_true
            rows.append({"dataset": tag, "method": "savgol_default",
                         "mae": float(np.mean(np.abs(err))),
                         "rmse": float(np.sqrt(np.mean(err**2))),
                         "max_err": float(np.max(np.abs(err))),
                         "n": len(tt)})
            # oracle-tuned externals
            for m, configs in externals.items():
                best = None
                for fn in configs:
                    try:
                        d = np.asarray(fn(tt, yy), float).ravel()[:len(tt)]
                        if d.shape != tt.shape or not np.all(np.isfinite(d)):
                            continue
                        err = d - d_true
                        mae = float(np.mean(np.abs(err)))
                        if best is None or mae < best["mae"]:
                            best = {"mae": mae,
                                    "rmse": float(np.sqrt(np.mean(err**2))),
                                    "max_err": float(np.max(np.abs(err)))}
                    except Exception:
                        continue
                rows.append({"dataset": tag, "method": m + "_oracle",
                             "n": len(tt),
                             **(best or {"mae": np.nan, "rmse": np.nan,
                                         "max_err": np.nan})})
            print("done", tag, flush=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "results_v2_1_realdata.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()

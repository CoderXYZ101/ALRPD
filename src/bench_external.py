"""
V2.0 external-code comparison: PyNumDiff and `derivative` package methods
vs ALPRD v1.4, on the identical 74-dataset suite (seed 12345).

PROTOCOL (deliberately favors the external methods):
- External methods receive ORACLE parameter selection: for each dataset,
  every config in a small per-method grid is scored against the TRUE
  derivative and the best is kept. This upper-bounds their achievable
  accuracy. ALPRD v1.4 runs with zero tuning, as always.
- `derivative`-package methods run natively on the uneven grid.
- pynumdiff methods are uniform-grid methods (their savgoldiff divides by
  a scalar/array dt in a way that breaks on uneven grids in v0.2.3), so
  they get the same linear-interpolation resample round-trip as the
  suite's own savgol_resampled baseline. This round-trip cost is part of
  what "applying a uniform-grid method to uneven data" means, and is
  documented, not hidden.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from datasets import make_dataset, all_dataset_specs

from derivative import dxdt
import pynumdiff


def _resampled(fn):
    """Wrap a uniform-grid pynumdiff method for uneven t (interp round trip)."""
    def run(t, y):
        n = len(t)
        tu = np.linspace(t[0], t[-1], n)
        dt = tu[1] - tu[0]
        yu = np.interp(tu, t, y)
        _, dh = fn(yu, dt)
        return np.interp(t, tu, dh)
    return run


# name -> list of (callable(t, y) -> d) configs
def build_grids():
    grids = {}

    grids["drv_savgol"] = [
        (lambda t, y, w=w, o=o: dxdt(y, t, kind="savitzky_golay",
                                     left=w, right=w, order=o))
        for w in [0.25, 0.5, 1.0, 2.0] for o in [2, 3]
    ]
    # scale-aware smoothing grid: scipy's spline `s` is an absolute residual
    # budget, so absolute values only suit O(1)-scale data. The classical
    # choice is s ~ n * sigma^2; we bracket it by factors on both sides
    # using a rough difference-based sigma estimate from the data itself
    # (protocol correction made during the V2.1 real-data study, where the
    # original absolute grid was scale-mismatched for AU-magnitude data
    # and unfairly handicapped the spline; documented in the log).
    def _spline_cfg(fac):
        def run(t, y):
            sig2 = np.median(np.diff(y, 2)**2) / 6.0
            s = max(len(y) * sig2 * fac, 1e-300)
            return dxdt(y, t, kind="spline", s=s)
        return run
    grids["drv_spline"] = [_spline_cfg(f)
                           for f in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]]
    grids["drv_kalman"] = [
        (lambda t, y, a=a: dxdt(y, t, kind="kalman", alpha=a))
        for a in [0.01, 0.05, 0.2, 1.0]
    ]

    grids["pnd_savgol"] = [
        _resampled(lambda x, dt, w=w: pynumdiff.savgoldiff(
            x, dt, degree=3, window_size=w, smoothing_win=w))
        for w in [9, 15, 25, 41, 61]
    ]
    grids["pnd_butter"] = [
        _resampled(lambda x, dt, c=c: pynumdiff.butterdiff(
            x, dt, filter_order=2, cutoff_freq=c))
        for c in [0.05, 0.1, 0.2, 0.4]
    ]
    grids["pnd_spectral"] = [
        _resampled(lambda x, dt, c=c: pynumdiff.spectraldiff(
            x, dt, high_freq_cutoff=c))
        for c in [0.05, 0.1, 0.2, 0.4]
    ]
    grids["pnd_median"] = [
        _resampled(lambda x, dt, w=w: pynumdiff.mediandiff(
            x, dt, window_size=w, num_iterations=2))
        for w in [5, 11, 21, 41]
    ]
    grids["pnd_tvr"] = [
        _resampled(lambda x, dt, g=g: pynumdiff.tvrdiff(
            x, dt, order=1, gamma=g))
        for g in [0.01, 0.1, 1.0]
    ]
    return grids


def main():
    grids = build_grids()
    rows = []
    for name, kwargs in all_dataset_specs():
        ds = make_dataset(**kwargs)
        t, y, d_true = ds["t"], ds["y"], ds["dydt_true"]
        for method, configs in grids.items():
            best = None
            t0 = time.perf_counter()
            for ci, fn in enumerate(configs):
                try:
                    d = np.asarray(fn(t, y), dtype=float).ravel()[:len(t)]
                    if d.shape != t.shape or not np.all(np.isfinite(d)):
                        continue
                    err = d - d_true
                    mae = float(np.mean(np.abs(err)))
                    if best is None or mae < best["mae"]:
                        best = {"mae": mae,
                                "rmse": float(np.sqrt(np.mean(err**2))),
                                "max_err": float(np.max(np.abs(err))),
                                "config": ci}
                except Exception:
                    continue
            elapsed = time.perf_counter() - t0
            row = {"dataset": name, "method": method,
                   "grid_runtime_s": elapsed, "n_points": len(t)}
            if best is None:
                row.update({"status": "ALL_CONFIGS_FAILED", "mae": np.nan,
                            "rmse": np.nan, "max_err": np.nan})
            else:
                row.update({"status": "ok", **best})
            rows.append(row)
        print(f"done {name}", flush=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "results_v2_0_external.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}, {len(rows)} rows")


if __name__ == "__main__":
    main()

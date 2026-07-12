"""
V2.2 physical-sensor real-data study.

Data: CI.PASC (Pasadena), 2024-08-12 M4.4 Highland Park earthquake,
19:20:25-19:21:15 UTC. TWO INDEPENDENT INSTRUMENTS at the same site and
sample clock: HHZ broadband seismometer (ground VELOCITY) and HNZ
strong-motion accelerometer (ground ACCELERATION). Both response-corrected
via obspy/SCEDC and identically band-passed 0.3-8 Hz; raw miniSEED
committed under data/.

Reference validity (measured): corr(d/dt HHZ, HNZ) = 0.9999, amplitude
ratio 0.982 -- i.e. the accelerometer supplies an instrument-derived
derivative reference with ~2% systematic uncertainty. Consequence: no
method can meaningfully score below ~2% relative error here; differences
below that floor are not interpretable (stated in the log).

Task: differentiate the SEISMOMETER's velocity samples (unevenly
subsampled) and compare against the ACCELEROMETER at the kept times
(linear interp from the dense 100 Hz grid; band edge 8 Hz => interp error
negligible relative to the 2% reference floor).
"""

import os
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


def load():
    from obspy import read
    v = read(os.path.join(DATA, "pasc_hhz_vel.mseed"))[0]
    a = read(os.path.join(DATA, "pasc_hnz_acc.mseed"))[0]
    dt = v.stats.delta
    t = np.arange(len(v.data)) * dt
    return t, v.data.astype(float), a.data.astype(float)


def main():
    rng = np.random.default_rng(SEED)
    t, vel, acc = load()
    # PROTOCOL NOTE (first attempt, kept for the record): decimating to
    # 33 Hz before the 60% thinning left ~20 Hz mean sampling against an
    # 8 Hz band edge -- every local method (ALPRD and all local externals)
    # collapsed to near-zero output (rel MAE ~ 1.0) because any >=7-point
    # window spans multiple signal periods; only the global spline
    # degraded gracefully (0.145). That configuration measures sampling
    # inadequacy, not differentiation quality. The study below therefore
    # keeps the full 100 Hz grid (mean ~60 Hz after thinning), which
    # resolves the 0.3-8 Hz band for local windows.
    t3, v3 = t, vel
    keep = np.sort(rng.choice(len(t3), size=int(0.6 * len(t3)),
                              replace=False))
    keep[0], keep[-1] = 0, len(t3) - 1
    keep = np.unique(keep)
    tt, yy = t3[keep], v3[keep]
    d_true = np.interp(tt, t, acc)          # independent-instrument reference
    s = np.std(yy)

    variants = {"asrecorded": yy.copy()}
    variants["noise5pct"] = yy + rng.normal(0, 0.05 * s, len(yy))
    y = yy + rng.normal(0, 0.01 * s, len(yy))
    n_out = max(1, int(0.05 * len(yy)))
    idx = rng.choice(len(yy), size=n_out, replace=False)
    y[idx] += rng.choice([-1, 1], n_out) * rng.uniform(5, 15, n_out) * 0.5 * s
    variants["outliers5pct"] = y

    grids = build_grids()
    externals = {k: grids[k] for k in
                 ["drv_savgol", "drv_spline", "pnd_butter", "pnd_savgol"]}
    rows = []
    for vname, yv in variants.items():
        tag = f"seismic__{vname}"
        d = alprd_v20(tt, yv)
        err = d - d_true
        rows.append({"dataset": tag, "method": "ALPRD_v2.0",
                     "mae": float(np.mean(np.abs(err))),
                     "rmse": float(np.sqrt(np.mean(err**2))),
                     "rel_mae": float(np.mean(np.abs(err)) / np.mean(np.abs(d_true))),
                     "n": len(tt)})
        d = savitzky_golay_resampled(tt, yv)
        err = d - d_true
        rows.append({"dataset": tag, "method": "savgol_default",
                     "mae": float(np.mean(np.abs(err))),
                     "rmse": float(np.sqrt(np.mean(err**2))),
                     "rel_mae": float(np.mean(np.abs(err)) / np.mean(np.abs(d_true))),
                     "n": len(tt)})
        for m, configs in externals.items():
            best = None
            for fn in configs:
                try:
                    d = np.asarray(fn(tt, yv), float).ravel()[:len(tt)]
                    if d.shape != tt.shape or not np.all(np.isfinite(d)):
                        continue
                    err = d - d_true
                    mae = float(np.mean(np.abs(err)))
                    if best is None or mae < best["mae"]:
                        best = {"mae": mae,
                                "rmse": float(np.sqrt(np.mean(err**2))),
                                "rel_mae": mae / float(np.mean(np.abs(d_true)))}
                except Exception:
                    continue
            rows.append({"dataset": tag, "method": m + "_oracle",
                         "n": len(tt),
                         **(best or {"mae": np.nan, "rmse": np.nan,
                                     "rel_mae": np.nan})})
        print("done", tag, flush=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "results_v2_2_seismic.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()

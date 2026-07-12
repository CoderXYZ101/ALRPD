"""
Runs every differentiation method against every benchmark dataset and
computes error / runtime metrics. Produces a CSV of raw results.
"""

import time
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from datasets import make_dataset, all_dataset_specs
from methods import (
    forward_difference, backward_difference, central_difference,
    high_order_fd, cubic_spline_derivative, savitzky_golay_resampled,
    alprd_v1, alprd_v1_fast, alprd_v11, savitzky_golay_matched,
    alprd_v12, tv_derivative, rbf_fd_derivative,
    alprd_v13, tv_derivative_l1, alprd_v14, alprd_v20,
)

METHODS = {
    "forward_diff": forward_difference,
    "backward_diff": backward_difference,
    "central_diff": central_difference,
    "high_order_fd": high_order_fd,
    "cubic_spline": cubic_spline_derivative,
    "savgol_resampled": savitzky_golay_resampled,
    "savgol_matched": lambda t, y: savitzky_golay_matched(t, y, span=0.25),
    "ALPRD_v1.0": lambda t, y: alprd_v1(t, y, poly_order=2, span=0.25),
    "ALPRD_v1.0_fast": lambda t, y: alprd_v1_fast(t, y, poly_order=2, span=0.25),
    "ALPRD_v1.1": lambda t, y: alprd_v11(t, y, poly_order=2),
    "ALPRD_v1.2": lambda t, y: alprd_v12(t, y, poly_order=2),
    "ALPRD_v1.3": alprd_v13,
    "ALPRD_v1.4": alprd_v14,
    "ALPRD_v2.0": alprd_v20,
    "tv_diff": tv_derivative,
    "tv_diff_l1": tv_derivative_l1,
    "rbf_fd": rbf_fd_derivative,
}


def compute_metrics(d_est, d_true, elapsed, mem_bytes):
    err = d_est - d_true
    abs_err = np.abs(err)
    scale = np.maximum(np.abs(d_true), 1e-8)
    rel_err = abs_err / scale
    return {
        "max_err": float(np.max(abs_err)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_rel_err": float(np.median(rel_err)),
        "runtime_s": elapsed,
        "mem_bytes": mem_bytes,
    }


def run_suite(out_csv):
    specs = all_dataset_specs()
    rows = []
    for name, kwargs in specs:
        ds = make_dataset(**kwargs)
        t, y, d_true = ds["t"], ds["y"], ds["dydt_true"]
        for method_name, fn in METHODS.items():
            try:
                t0 = time.perf_counter()
                d_est = fn(t, y)
                elapsed = time.perf_counter() - t0
                mem = d_est.nbytes + t.nbytes + y.nbytes
                m = compute_metrics(np.asarray(d_est), d_true, elapsed, mem)
                m.update({"dataset": name, "method": method_name,
                          "n_points": len(t), "status": "ok"})
            except Exception as e:
                m = {"dataset": name, "method": method_name, "n_points": len(t),
                     "status": f"FAILED: {type(e).__name__}: {e}",
                     "max_err": np.nan, "mae": np.nan, "rmse": np.nan,
                     "median_rel_err": np.nan, "runtime_s": np.nan, "mem_bytes": np.nan}
            rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "results", "results_v2_1.csv")
    df = run_suite(out)
    n_fail = (df["status"] != "ok").sum()
    print(f"Completed {len(df)} runs, {n_fail} failures. Saved to {out}")
    print(df.groupby("method")[["mae", "rmse", "max_err"]].mean(numeric_only=True))

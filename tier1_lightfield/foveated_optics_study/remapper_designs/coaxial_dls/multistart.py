"""Multi-start damped least squares over coaxial topologies.

    python multistart.py <out_dir> <n_trials> [n_workers]

Each trial draws a random 4-6 element seed (random curvatures, spacings and
real-material indices, no aspheres) and runs scipy least_squares ('trf', bounded,
x_scale='jac') on fast_merit.residuals in two stages: spacings + curvatures +
conics, then also r^4..r^8 aspheres. Trials are independent, one per worker.
Every trial's final prescription and merit are written to <out_dir>.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import least_squares

import fast_merit as fm

INDICES = (1.49, 1.53, 1.585, 1.64, 1.72, 1.80)
SAGITTA_BOUND_MM = 50.0          # paraxial sag at R0: |c| up to 1/4 mm^-1
ASPHERE_BOUND = (8.0, 8.0, 8.0)  # rim sag of each asphere term at R0, mm


SEED_RX = Path(__file__).resolve().parent / "stage1_shape.json"


def random_seed(rng):
    """A perturbation of a known-sane design (every ray survives), optionally
    with one extra thin, weak element inserted before the MLA as a field
    flattener candidate. Fully random curvatures leave most rays dead, and dead
    rays give a flat, gradient-free merit."""
    rx = json.loads(SEED_RX.read_text())
    if rng.random() < 0.5:
        last = rx["elements"][-1]
        gap = last["gap_after_mm"]
        last["gap_after_mm"] = 0.4 * gap
        rx["elements"].append({"index": float(rng.choice(INDICES)), "thickness_mm": 2.0,
                               "gap_after_mm": max(0.6 * gap - 2.0, 0.5),
                               "front": {"radius_mm": float(rng.choice([-1, 1]) * rng.uniform(40, 120)),
                                         "conic": 0.0, "coefficients": [0.0, 0.0, 0.0, 0.0]},
                               "back": {"radius_mm": math.inf, "conic": 0.0, "coefficients": [0.0, 0.0, 0.0, 0.0]}})
    indices = [e["index"] for e in rx["elements"]]
    x = fm.to_vector(rx)
    n = len(indices)
    shape = x[1 + 2 * n:].reshape(n, 2, 5)
    shape[:, :, 0] *= rng.uniform(0.9, 1.1, size=(n, 2))
    shape[:, :, 1] += rng.normal(0.0, 0.3, size=(n, 2))
    x[1:1 + 2 * n] *= rng.uniform(0.9, 1.1, size=2 * n)
    return indices, x


def bounds(n):
    lo, hi = [20.5], [30.0]
    for i in range(n):
        lo += [1.5, 0.2]
        hi += [25.0, 40.0 if i < n - 1 else 80.0]
    for _ in range(2 * n):
        lo += [-SAGITTA_BOUND_MM, -30.0] + [-b for b in ASPHERE_BOUND]
        hi += [SAGITTA_BOUND_MM, 30.0] + list(ASPHERE_BOUND)
    return np.array(lo), np.array(hi)


def asphere_mask(n):
    m = np.zeros(1 + 2 * n + 10 * n, dtype=bool)
    for s in range(2 * n):
        m[1 + 2 * n + 5 * s + 2: 1 + 2 * n + 5 * s + 5] = True
    return m


def solve(x, indices, free, max_nfev):
    lo, hi = bounds(len(indices))
    fixed = x.copy()

    def f(z):
        full = fixed.copy()
        full[free] = z
        return fm.residuals(full, indices)

    z0 = np.clip(x[free], lo[free] + 1e-12, hi[free] - 1e-12)
    res = least_squares(f, z0, bounds=(lo[free], hi[free]), method="trf", x_scale="jac", max_nfev=max_nfev)
    out = fixed.copy()
    out[free] = res.x
    return out, float(np.sum(res.fun**2))


def trial(args):
    k, out_dir, seed_value = args
    rng = np.random.default_rng(seed_value)
    indices, x = random_seed(rng)
    t0 = time.time()
    m_asph = asphere_mask(len(indices))
    x, m1 = solve(x, indices, ~m_asph, max_nfev=300)
    x, m2 = solve(x, indices, np.ones_like(m_asph), max_nfev=500)
    rx = fm.to_prescription(x, indices)
    result = {"trial": k, "seed": seed_value, "n_elements": len(indices), "merit_stage1": m1,
              "merit": m2, "seconds": time.time() - t0, "prescription": rx}
    (Path(out_dir) / f"trial_{k:03d}.json").write_text(json.dumps(result, indent=1))
    return k, len(indices), m1, m2, time.time() - t0


def main():
    out_dir, n_trials = Path(sys.argv[1]), int(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, os.cpu_count() - 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(k, str(out_dir), 1000 + k) for k in range(n_trials)]
    with mp.get_context("fork").Pool(workers) as pool:
        for k, n, m1, m2, dt in pool.imap_unordered(trial, jobs):
            print(f"trial {k:3d}  elements {n}  merit {m1:10.4f} -> {m2:10.4f}  ({dt:6.0f} s)", flush=True)
    best = min((json.loads(p.read_text()) for p in out_dir.glob("trial_*.json")), key=lambda r: r["merit"])
    (out_dir / "best.json").write_text(json.dumps(best, indent=1))
    x = fm.to_vector(best["prescription"])
    indices = [e["index"] for e in best["prescription"]["elements"]]
    print(f"BEST trial {best['trial']} ({best['n_elements']} elements) merit {best['merit']:.4f}")
    print(fm.spot_table(x, indices))


if __name__ == "__main__":
    main()

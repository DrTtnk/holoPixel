"""Adam optimisation of the two-mirror + panel-pose system against the R = 6
foveation target. Run as a script: writes the optimised params to a .pt file
and prints a metrics summary every `log_every` iterations.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(8)  # be a reasonable neighbour: two other agents share this CPU/GPU

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout import all_leaves, build_params, solve_initial_layout  # noqa: E402
from system import field_grid, hex_pupil_samples, loss_and_metrics  # noqa: E402

APERTURE = ((38.0, 24.0), (22.0, 22.0))  # (half-x, half-y) mm, local, for M1 then M2


def run(iters=2500, nx=17, nz=13, pupil_spacing=0.75, lr=0.02, log_every=100, seed=0,
        a1_deg=48.0, y1=21.0, d12=18.0, panel_target=(14.0, -9.0), f1=150.0):
    layout = solve_initial_layout(a1_deg=a1_deg, y1=y1, d12=d12, panel_target=panel_target, f1=f1)
    params = build_params(layout, APERTURE, seed=seed)
    leaves = all_leaves(params)
    opt = torch.optim.Adam(leaves, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.03)

    tx, tz = field_grid(nx, nz)
    pupil_xy = hex_pupil_samples(pupil_spacing)
    print(f"field points: {len(tx)}, pupil samples: {len(pupil_xy)}, rays/iter: {len(tx)*len(pupil_xy)}")

    t0 = time.time()
    for it in range(iters):
        # ramp the spot/tele weight in: first fix the chief-ray mapping, then focus.
        ramp = min(1.0, it / (0.2 * iters))
        weights = dict(pos=1.0, spot=1.0 + 15.0 * ramp, tele=0.2 + 6.0 * ramp,
                       invalid=1.0, aperture=0.1, reg=1e-6)
        opt.zero_grad()
        loss, metrics, _ = loss_and_metrics(params, tx, tz, pupil_xy, aperture=APERTURE, weights=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(leaves, 2.0)
        opt.step()
        sched.step()
        if it % log_every == 0 or it == iters - 1:
            print(f"it {it:5d} loss {loss.item():.5f} pos_rms {metrics['pos_err_mm'].mean():.4f}mm "
                 f"spot_rms {metrics['spot_rms_mm'].mean():.4f}mm tele {metrics['tele_1mcos'].mean():.5f} "
                 f"valid {metrics['frac_valid'].mean():.4f} relief {metrics['eye_relief_mm'].item():.2f}mm "
                 f"({time.time()-t0:.0f}s)")
    return params, (tx, tz, pupil_xy)


def save_params(params, path):
    """Detach every tensor leaf; pass through plain floats (u_scale, v_scale)
    unchanged, since torch.save handles a mixed dict of tensors/floats fine."""
    def clean(v):
        return v.detach() if torch.is_tensor(v) else v

    out = {}
    for k, v in params.items():
        out[k] = {kk: clean(vv) for kk, vv in v.items()} if isinstance(v, dict) else clean(v)
    torch.save(out, path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "opt_state.pt"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    params, _ = run(iters=iters)
    save_params(params, out)
    print(f"saved {out}")

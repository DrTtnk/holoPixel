import time
import numpy as np
import torch
import design as ds
import raytrace as rt

torch.manual_seed(0)
z = np.load("/tmp/phase1_adam.npz")
d = ds.Design()
d.pose = torch.tensor(z["pose"], dtype=ds.DTYPE, requires_grad=True)
d.coeff = torch.tensor(z["coeff"], dtype=ds.DTYPE, requires_grad=True)

tx, tz = ds.mapping_grid()
ecc = np.radians(np.hypot(tx, tz))
target_r = ds.ft.panel_radius_mm(ecc)
weight = 1.0 / ds.ft.local_focal_mm(ecc)
pupil_xy = torch.tensor(ds.pupil_ring(n=7), dtype=ds.DTYPE)
print(f"n_field={len(tx)} n_pupil={len(pupil_xy)}", flush=True)

opt = torch.optim.Adam([
    {"params": [d.pose], "lr": 0.004},
    {"params": [d.coeff], "lr": 0.003},
])
t0 = time.time()
n_iters = 900
best = None
for it in range(n_iters):
    opt.zero_grad()
    l_map, l_spot, l_margin, valid, r_ach, spot_rms = ds.loss_fn(d, tx, tz, target_r, weight, pupil_xy, margin_deg=2.0)
    # No spot term this pass: it destabilised the optimisation and fought
    # the mapping objective. Get the mapping right with full freeform terms
    # first; spot quality is judged (and possibly polished) afterwards.
    loss = 1.0 * l_map + 0.5 * l_margin + 5e-3 * d.coeff.pow(2).sum()
    if not torch.isfinite(loss):
        print(it, "NON-FINITE LOSS, stopping", flush=True)
        break
    loss.backward()
    torch.nn.utils.clip_grad_norm_([d.pose, d.coeff], 1.0)
    opt.step()
    if it % 25 == 0 or it == n_iters - 1:
        v = float(valid)
        print(it, "loss", float(loss), "map", float(l_map),
             "margin", float(l_margin), "valid", v,
             "spot_rms_um", float(spot_rms.mean()) * 1000, "t", time.time() - t0, flush=True)
        if v > 0.97 and torch.isfinite(loss):
            best = (float(l_map), d.pose.detach().clone(), d.coeff.detach().clone())
            np.savez("/tmp/phase2c_adam.npz", pose=best[1].numpy(), coeff=best[2].numpy(), index=d.index)

print("done, best map loss:", best[0] if best else None, flush=True)

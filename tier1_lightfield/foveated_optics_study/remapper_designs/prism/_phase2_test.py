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
    {"params": [d.coeff], "lr": 0.006},
])
t0 = time.time()
n_iters = 1200
for it in range(n_iters):
    opt.zero_grad()
    l_map, l_spot, l_margin, valid, r_ach, spot_rms = ds.loss_fn(d, tx, tz, target_r, weight, pupil_xy, margin_deg=2.0)
    loss = 1.0 * l_map + 0.15 * l_spot + 0.5 * l_margin + 2e-3 * d.coeff.pow(2).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([d.pose, d.coeff], 1.0)
    opt.step()
    if it % 100 == 0 or it == n_iters - 1:
        print(it, "loss", float(loss), "map", float(l_map), "spot", float(l_spot),
             "margin", float(l_margin), "valid", float(valid),
             "spot_rms_um", float(spot_rms.mean()) * 1000, "t", time.time() - t0, flush=True)
        np.savez("/tmp/phase2_adam.npz", pose=d.pose.detach().numpy(), coeff=d.coeff.detach().numpy(), index=d.index)

np.savez("/tmp/phase2_adam.npz", pose=d.pose.detach().numpy(), coeff=d.coeff.detach().numpy(), index=d.index)
print("done", flush=True)

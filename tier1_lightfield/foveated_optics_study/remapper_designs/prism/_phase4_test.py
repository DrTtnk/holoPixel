import time
import numpy as np
import torch
import design as ds
import raytrace as rt

torch.manual_seed(0)
z = np.load("/tmp/phase2c_adam.npz")
d = ds.Design()
d.pose = torch.tensor(z["pose"], dtype=ds.DTYPE, requires_grad=False)  # frozen: protect the mapping
d.coeff = torch.tensor(z["coeff"], dtype=ds.DTYPE, requires_grad=True)

tx, tz = ds.mapping_grid()
ecc = np.radians(np.hypot(tx, tz))
target_r = ds.ft.panel_radius_mm(ecc)
weight = 1.0 / ds.ft.local_focal_mm(ecc)
pupil_xy = torch.tensor(ds.pupil_ring(n=7), dtype=ds.DTYPE)

opt = torch.optim.Adam([d.coeff], lr=0.0003)
t0 = time.time()
n_iters = 500
best = None
map0 = None
for it in range(n_iters):
    opt.zero_grad()
    l_map, l_spot, l_margin, valid, r_ach, spot_rms = ds.loss_fn(d, tx, tz, target_r, weight, pupil_xy, margin_deg=2.0)
    if map0 is None:
        map0 = float(l_map)
    # mapping stays a hard-ish constraint (large weight, small allowed drift),
    # spot gets to improve within that budget.
    loss = 3.0 * l_map + 0.01 * l_spot + 0.5 * l_margin + 5e-3 * d.coeff.pow(2).sum()
    if not torch.isfinite(loss):
        print(it, "NON-FINITE LOSS, stopping", flush=True)
        break
    loss.backward()
    torch.nn.utils.clip_grad_norm_([d.coeff], 0.2)
    opt.step()
    if it % 25 == 0 or it == n_iters - 1:
        v = float(valid)
        print(it, "loss", float(loss), "map", float(l_map), "spot", float(l_spot),
             "margin", float(l_margin), "valid", v,
             "spot_rms_um", float(spot_rms.mean()) * 1000, "t", time.time() - t0, flush=True)
        if v > 0.97 and float(l_map) < 1.5 * map0 and torch.isfinite(loss):
            best = (float(l_map), float(l_spot), d.coeff.detach().clone())
            np.savez("/tmp/phase4_adam.npz", pose=d.pose.detach().numpy(), coeff=best[2].numpy(), index=d.index)

print("done, map0", map0, "best map/spot loss:", (best[0], best[1]) if best else None, flush=True)

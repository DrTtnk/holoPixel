import time
import numpy as np
import torch
import design as ds
import raytrace as rt

torch.manual_seed(0)
z = np.load("/tmp/phase5_adam.npz")
d = ds.Design()
d.pose = torch.tensor(z["pose"], dtype=ds.DTYPE, requires_grad=True)
d.coeff = torch.tensor(z["coeff"], dtype=ds.DTYPE, requires_grad=True)

tx, tz = ds.mapping_grid()
ecc = np.radians(np.hypot(tx, tz))
target_r = ds.ft.panel_radius_mm(ecc)
weight = 1.0 / ds.ft.local_focal_mm(ecc)
pupil_xy = torch.tensor(ds.pupil_ring(n=7), dtype=ds.DTYPE)

onaxis_o = torch.tensor([[0.0, ds.PUPIL_Y_MM, 0.0]], dtype=ds.DTYPE)
onaxis_d = torch.tensor([[0.0, 1.0, 0.0]], dtype=ds.DTYPE)


def eye_relief_mm(design):
    s1, s2, s3 = design.surfaces()
    t1, p1, n1 = rt.intersect(s1, onaxis_o, onaxis_d)
    return t1[0]


opt = torch.optim.Adam([
    {"params": [d.pose], "lr": 0.001},
    {"params": [d.coeff], "lr": 0.0006},
])
t0 = time.time()
n_iters = 400
best = None
for it in range(n_iters):
    opt.zero_grad()
    l_map, l_spot, l_margin, valid, r_ach, spot_rms = ds.loss_fn(d, tx, tz, target_r, weight, pupil_xy, margin_deg=2.0)
    er = eye_relief_mm(d)
    l_er = torch.relu(22.0 - er).pow(2)  # target comfortably above 20 mm
    loss = 1.0 * l_map + 0.5 * l_margin + 5e-3 * d.coeff.pow(2).sum() + 5.0 * l_er
    if not torch.isfinite(loss):
        print(it, "NON-FINITE LOSS, stopping", flush=True)
        break
    loss.backward()
    torch.nn.utils.clip_grad_norm_([d.pose, d.coeff], 0.5)
    opt.step()
    if it % 20 == 0 or it == n_iters - 1:
        v = float(valid)
        print(it, "loss", float(loss), "map", float(l_map), "margin", float(l_margin),
             "eye_relief_mm", float(er), "valid", v, "t", time.time() - t0, flush=True)
        if v > 0.97 and torch.isfinite(loss):
            best = (float(l_map), float(er), d.pose.detach().clone(), d.coeff.detach().clone())
            np.savez("/tmp/phase5_adam.npz", pose=best[2].numpy(), coeff=best[3].numpy(), index=d.index)

print("done, best map/eye_relief:", (best[0], best[1]) if best else None, flush=True)

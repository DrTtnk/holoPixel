import torch, numpy as np, math, time
import design as ds
import raytrace as rt
torch.manual_seed(0)
d = ds.Design()
tx, tz = ds.mapping_grid()
ecc = np.radians(np.hypot(tx, tz))
target_r = ds.ft.panel_radius_mm(ecc)
weight = 1.0/ds.ft.local_focal_mm(ecc)
pupil_xy = torch.tensor(ds.pupil_ring(), dtype=ds.DTYPE)[:1]  # chief ray only, phase 1

low_order = {(0,2),(2,0)}
mask = torch.tensor([1.0 if t in low_order else 0.0 for t in rt.TERMS], dtype=ds.DTYPE)

opt = torch.optim.Adam([
    {"params":[d.pose], "lr":0.01},
    {"params":[d.coeff], "lr":0.01},
])
t0=time.time()
for it in range(1200):
    opt.zero_grad()
    l_map,l_spot,l_margin,valid,r_ach,spot_rms = ds.loss_fn(d, tx, tz, target_r, weight, pupil_xy, margin_deg=3.0)
    loss = 1.0*l_map + 0.5*l_margin + 2e-3*d.coeff.pow(2).sum()
    loss.backward()
    d.coeff.grad *= mask
    torch.nn.utils.clip_grad_norm_([d.pose,d.coeff], 1.0)
    opt.step()
    if it%150==0 or it==1199:
        print(it, "loss",float(loss),"map",float(l_map),"margin",float(l_margin),"valid",float(valid),"t",time.time()-t0, flush=True)
print("pose", d.pose.detach().numpy())
np.savez("/tmp/phase1_adam.npz", pose=d.pose.detach().numpy(), coeff=d.coeff.detach().numpy(), index=d.index)

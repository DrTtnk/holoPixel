"""Optimizes the freeform TIR prism: S1 (eye-facing, used twice), S2 (mirror),
S3 (entry towards the MLA). Traced in reverse, pupil -> panel, exactly the
order Cycles' camera rays see: S1 refract in, S2 mirror, S1 TIR, S3 refract
out, then straight-line propagation in air to the flat MLA plane (the
lenslet sag is a few microns, negligible next to the 30 um pitch for this
distortion-mapping optimisation -- lf_evaluate.py is the real judge).

Merit per field/pupil sample:
  mapping  -- achieved landing radius on the MLA plane vs foveation_target's
              panel_radius_mm(eccentricity), weighted by 1/local_focal_mm so
              the residual is an angular (not linear) error -- this is what
              the evaluator's sampling-ratio check actually measures.
  spot     -- RMS transverse spread across the pupil at fixed field angle
              (rays from an object at infinity should refocus to one point).
  margin   -- TIR must hold at S1's second hit, refraction (not TIR) must
              hold at S1's exit hit and at S3.
  positive -- every ray/surface hit must be in front of the ray (t > 0).

Run: python design.py [--iters N] [--out params.npz]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import raytrace as rt  # noqa: E402
import foveation_target as ft  # noqa: E402

DTYPE = rt.DTYPE
PUPIL_Y_MM = -3.6
PUPIL_RADIUS_MM = 2.0
INDEX = 1.9
FOV_X_DEG, FOV_Z_DEG = 35.0, 22.5

INIT = dict(y1=19.74158776362268, z1=-1.6135831162865588, th1=86.02294184881563,
            y2=24.413516701857642, z2=-1.3607875241396883, th2=241.33191295628885,
            y3=40.645344748528714, z3=33.22125324561914, th3=-49.96032587057946)


def field_direction(tx_deg, tz_deg):
    d = torch.stack([torch.tan(torch.deg2rad(tx_deg)),
                     torch.ones_like(tx_deg),
                     torch.tan(torch.deg2rad(tz_deg))], dim=-1)
    return d / d.norm(dim=-1, keepdim=True)


def mapping_grid():
    """(tx, tz) samples covering the field, denser near the axis where the
    R=6 target curves fastest, clipped to the rectangular FOV per azimuth."""
    eccs = np.concatenate([np.linspace(0.0, 10.0, 11), np.linspace(12.0, 35.0, 10)])
    azs = np.radians(np.arange(0.0, 360.0, 22.5))
    tx, tz = [], []
    for e in eccs:
        for a in azs:
            x, z = e * math.cos(a), e * math.sin(a)
            if abs(x) <= FOV_X_DEG and abs(z) <= FOV_Z_DEG:
                tx.append(x)
                tz.append(z)
    return np.array(tx), np.array(tz)


def pupil_ring(n=7, radius=PUPIL_RADIUS_MM):
    """Centre plus (n - 1) points evenly spaced on the pupil rim."""
    pts = [(0.0, 0.0)]
    for k in range(n - 1):
        a = 2.0 * math.pi * k / (n - 1)
        pts.append((radius * math.cos(a), radius * math.sin(a)))
    return np.array(pts)


class Design:
    def __init__(self, init=INIT, index=INDEX, requires_grad=True):
        self.index = index
        names = ["y1", "z1", "th1", "y2", "z2", "th2", "y3", "z3", "th3"]
        self.pose = torch.tensor([init[k] for k in names], dtype=DTYPE, requires_grad=requires_grad)
        self.coeff = torch.zeros(3, rt.N_COEFF, dtype=DTYPE, requires_grad=requires_grad)

    def params(self):
        return [self.pose, self.coeff]

    def surfaces(self):
        y1, z1, th1, y2, z2, th2, y3, z3, th3 = self.pose
        zt = torch.zeros((), dtype=DTYPE)
        s1 = rt.Surface(torch.stack([zt, y1, z1]), torch.deg2rad(th1), self.coeff[0])
        s2 = rt.Surface(torch.stack([zt, y2, z2]), torch.deg2rad(th2), self.coeff[1])
        s3 = rt.Surface(torch.stack([zt, y3, z3]), torch.deg2rad(th3), self.coeff[2])
        return s1, s2, s3

    def prism(self):
        s1, s2, s3 = self.surfaces()
        return rt.Prism(s1, s2, s3, self.index)

    def panel_frame(self, gap_mm=3.0):
        """Boresight (on-axis, pupil-centre) chief ray fixes the panel pose:
        the flat MLA face sits `gap_mm` beyond S3's exit point, facing back
        towards the prism, sharing the world-X axis (plane symmetry)."""
        prism = self.prism()
        origin = torch.tensor([[0.0, PUPIL_Y_MM, 0.0]], dtype=DTYPE)
        direction = torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE)
        out = prism.trace(origin, direction)
        p4, d4 = out["p_out"][0], out["d_out"][0]
        panel_origin = p4 + gap_mm * d4
        w = -d4 / d4.norm()
        u = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
        v = torch.cross(w, u, dim=-1)
        v = v / v.norm()
        u = torch.cross(v, w, dim=-1)
        return panel_origin, torch.stack([u, v, w])

    def trace_to_panel(self, tx_deg, tz_deg, pupil_xy, gap_mm=3.0, iters=8):
        """tx_deg, tz_deg: (F,) field samples. pupil_xy: (P, 2) mm offsets.
        Returns landing (u, v) mm in the panel frame, shape (F, P), plus the
        prism trace dict (flattened over F*P) for margin/validity losses."""
        prism = self.prism()
        origin_panel, basis = self.panel_frame(gap_mm)
        n_f, n_p = tx_deg.shape[0], pupil_xy.shape[0]
        d = field_direction(tx_deg, tz_deg)                       # (F, 3)
        d = d[:, None, :].expand(n_f, n_p, 3).reshape(-1, 3)
        px, pz = pupil_xy[:, 0], pupil_xy[:, 1]
        origin = torch.stack([px, torch.full_like(px, PUPIL_Y_MM), pz], dim=-1)
        origin = origin[None, :, :].expand(n_f, n_p, 3).reshape(-1, 3)
        out = prism.trace(origin, d, iters=iters)
        rel = out["p_out"] - origin_panel[None, :]
        uv = torch.stack([(rel * basis[0]).sum(-1), (rel * basis[1]).sum(-1)], dim=-1)
        return uv.reshape(n_f, n_p, 2), out


RAD2DEG = 180.0 / math.pi
SPOT_TARGET_MM = 0.005  # 5 um


def loss_fn(design, tx, tz, target_r, weight, pupil_xy, margin_deg=3.0):
    n_g = design.index
    crit = math.degrees(math.asin(1.0 / n_g))
    uv, out = design.trace_to_panel(torch.tensor(tx, dtype=DTYPE), torch.tensor(tz, dtype=DTYPE), pupil_xy)
    r_achieved = uv[:, 0, :].norm(dim=-1)  # pupil-centre ray only (index 0 == (0,0))
    w = torch.tensor(weight, dtype=DTYPE)
    # map error in an angular unit (rad, via the 1/local_focal weight), reported in deg^2
    loss_map = (((r_achieved - torch.tensor(target_r, dtype=DTYPE)) * w) * RAD2DEG).pow(2).mean()

    centroid = uv.mean(dim=1, keepdim=True)
    spot_rms = (uv - centroid).pow(2).sum(-1).mean(dim=1).sqrt()
    loss_spot = (spot_rms / SPOT_TARGET_MM).pow(2).mean()

    inc3, inc4 = out["inc3_deg"], out["inc4_deg"]
    margin_tir = inc3 - crit
    margin_exit = crit - inc4
    loss_margin = (torch.relu(margin_deg - margin_tir).pow(2).mean()
                   + torch.relu(margin_deg - margin_exit).pow(2).mean())

    valid = out["valid"].float().mean()
    return loss_map, loss_spot, loss_margin, valid, r_achieved, spot_rms


def run(iters, out_path, lr_pose=0.01, lr_coeff=0.02, log_every=200, resume=None,
       unlock_at=0, w_map=1.0, w_spot=0.3, w_margin=0.5, w_reg=2e-4):
    design = Design()
    if resume is not None:
        z = np.load(resume)
        design.pose = torch.tensor(z["pose"], dtype=DTYPE, requires_grad=True)
        design.coeff = torch.tensor(z["coeff"], dtype=DTYPE, requires_grad=True)
    tx, tz = mapping_grid()
    ecc = np.radians(np.hypot(tx, tz))
    target_r = ft.panel_radius_mm(ecc)
    weight = 1.0 / ft.local_focal_mm(ecc)
    pupil_xy = torch.tensor(pupil_ring(), dtype=DTYPE)
    print(f"mapping grid: {len(tx)} field points, pupil ring: {len(pupil_xy)} points")

    # curriculum: for the first `unlock_at` iterations, only the base conic
    # terms (x^2, u^2) may move -- gets the paraxial power roughly right
    # before the higher-order barrel-distortion terms engage.
    low_order = {(0, 2), (2, 0)}
    mask = torch.tensor([1.0 if t in low_order else 0.0 for t in rt.TERMS], dtype=DTYPE)

    opt = torch.optim.Adam([
        {"params": [design.pose], "lr": lr_pose},
        {"params": [design.coeff], "lr": lr_coeff},
    ])
    t0 = time.time()
    for it in range(iters):
        opt.zero_grad()
        l_map, l_spot, l_margin, valid, r_ach, spot_rms = loss_fn(design, tx, tz, target_r, weight, pupil_xy)
        loss = w_map * l_map + w_spot * l_spot + w_margin * l_margin + w_reg * design.coeff.pow(2).sum()
        loss.backward()
        if it < unlock_at:
            design.coeff.grad *= mask
        torch.nn.utils.clip_grad_norm_(design.params(), 2.0)
        opt.step()
        if it % log_every == 0 or it == iters - 1:
            print(f"it {it:5d} loss {float(loss):.5f} map_deg2 {float(l_map):.6f} spot {float(l_spot):.6f} "
                 f"margin {float(l_margin):.4f} valid {float(valid):.4f} "
                 f"spot_rms_um {float(spot_rms.mean())*1000:.2f} t={time.time()-t0:.0f}s")
    np.savez(out_path, pose=design.pose.detach().numpy(), coeff=design.coeff.detach().numpy(),
            index=design.index)
    print(f"saved {out_path}")
    return design


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "params.npz"))
    ap.add_argument("--lr_pose", type=float, default=0.01)
    ap.add_argument("--lr_coeff", type=float, default=0.02)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--unlock_at", type=int, default=0)
    args = ap.parse_args()
    run(args.iters, args.out, args.lr_pose, args.lr_coeff, resume=args.resume, unlock_at=args.unlock_at)

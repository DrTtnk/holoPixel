"""The two-mirror + flat-panel system: builds Freeform surfaces from a
parameter dict, reverse-traces pupil-side rays to the panel plane, and scores
them against the R = 6 foveation target.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from surfaces import DTYPE, Freeform, reflect, rot_x_frame  # noqa: E402
import foveation_target as ft  # noqa: E402
from layout import TERMS, PUPIL_Y_MM  # noqa: E402

FOV_X_DEG, FOV_Z_DEG = 35.0, 22.5
PUPIL_RADIUS_MM = 2.0


def make_mirror(p):
    return Freeform(p["V"], p["tilt"], p["c"], p["k"], TERMS, p["coeffs"],
                    u_scale=p["u_scale"], v_scale=p["v_scale"])


def panel_frame(panel):
    u, v, w = rot_x_frame(panel["tilt"])
    return u, v, w


def hex_pupil_samples(spacing_mm, radius_mm=PUPIL_RADIUS_MM):
    n = int(np.ceil(radius_mm / spacing_mm)) + 1
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    x = spacing_mm * (i + 0.5 * j)
    z = spacing_mm * (np.sqrt(3.0) / 2.0) * j
    keep = np.hypot(x, z) <= radius_mm + 1e-9
    return np.column_stack([x[keep], z[keep]])


def field_grid(nx, nz, fov_x=FOV_X_DEG, fov_z=FOV_Z_DEG):
    tx = np.radians(np.linspace(-fov_x, fov_x, nx))
    tz = np.radians(np.linspace(-fov_z, fov_z, nz))
    TX, TZ = np.meshgrid(tx, tz)
    return TX.ravel(), TZ.ravel()


def trace(params, tx, tz, pupil_xy, aperture=None):
    """tx, tz: (F,) numpy radians. pupil_xy: (S, 2) mm. Returns dict of
    (F, S, ...) torch tensors: panel_uv, hit, dir_out (direction arriving at
    the panel), plus (F, S) validity mask."""
    m1, m2 = make_mirror(params["m1"]), make_mirror(params["m2"])
    panel = params["panel"]

    F, S = len(tx), len(pupil_xy)
    tx_t = torch.tensor(tx, dtype=DTYPE)
    tz_t = torch.tensor(tz, dtype=DTYPE)
    D = torch.stack([torch.tan(tx_t), torch.ones(F, dtype=DTYPE), torch.tan(tz_t)], dim=1)
    D = D / torch.linalg.norm(D, dim=1, keepdim=True)  # (F, 3)
    D = D.repeat_interleave(S, dim=0)  # (F*S, 3)

    px = torch.tensor(pupil_xy[:, 0], dtype=DTYPE)
    pz = torch.tensor(pupil_xy[:, 1], dtype=DTYPE)
    O = torch.stack([px, torch.full((S,), PUPIL_Y_MM, dtype=DTYPE), pz], dim=1)
    O = O.repeat(F, 1)  # (F*S, 3)

    t1, P1, hit1 = m1.intersect(O, D)
    x1, y1 = m1.local_coords(P1)[:2]
    n1 = m1.normal_world(x1, y1)
    D1 = reflect(D, n1)

    t2, P2, hit2 = m2.intersect(P1, D1)
    x2, y2 = m2.local_coords(P2)[:2]
    n2 = m2.normal_world(x2, y2)
    D2 = reflect(D1, n2)

    u_p, v_p, w_p = panel_frame(panel)
    denom = D2 @ w_p
    t3 = ((panel["origin"] - P2) @ w_p) / denom
    P3 = P2 + t3[:, None] * D2
    up = (P3 - panel["origin"]) @ u_p
    vp = (P3 - panel["origin"]) @ v_p

    valid = hit1 & hit2 & (t2 > 1e-6) & (t3 > 1e-6)
    # A hard aperture cutoff makes rays that drift outside it INVISIBLE to the
    # optimiser (a boolean mask has zero gradient), which was observed to let
    # the valid fraction decay over iterations with nothing pulling it back.
    # Report a smooth (differentiable) excess-beyond-aperture instead, so a
    # soft penalty in the loss can pull the beam back in; `valid` here stays
    # purely physical (did the ray actually hit both mirrors and the panel).
    aperture_excess = torch.zeros_like(x1)
    if aperture is not None:
        (ax1, ay1), (ax2, ay2) = aperture
        aperture_excess = (torch.relu(x1.abs() - ax1) ** 2 + torch.relu(y1.abs() - ay1) ** 2
                          + torch.relu(x2.abs() - ax2) ** 2 + torch.relu(y2.abs() - ay2) ** 2)

    shp = (F, S)
    return dict(up=up.reshape(shp), vp=vp.reshape(shp), dir_out=D2.reshape(*shp, 3),
               valid=valid.reshape(shp), w_p=w_p, u_p=u_p, v_p=v_p,
               x1=x1.reshape(shp), y1=y1.reshape(shp), x2=x2.reshape(shp), y2=y2.reshape(shp),
               aperture_excess=aperture_excess.reshape(shp),
               P1=P1.reshape(*shp, 3), P2=P2.reshape(*shp, 3), P3=P3.reshape(*shp, 3))


def axial_eye_relief_mm(params):
    """t at which the single on-axis ray (straight ahead from the pupil
    centre) first meets M1. A hinge hook for a >= 20 mm constraint."""
    m1 = make_mirror(params["m1"])
    O = torch.tensor([[0.0, PUPIL_Y_MM, 0.0]], dtype=DTYPE)
    D = torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE)
    t1, _, _ = m1.intersect(O, D)
    return t1[0]


def measure_footprint(params, cap, nx=25, nz=19, pupil_spacing=0.6, margin=1.15, pct=97.0):
    """Actual (x1, y1, x2, y2) extent the CURRENT (e.g. just-optimised)
    system needs over the full field x pupil grid, physically-valid rays
    only. Used to size the export mesh aperture from what the design really
    needs, rather than trusting the training-time soft-penalty aperture.

    A PERCENTILE, not the max, and a hard `cap` (the training-time soft
    aperture, which the freeform polynomial was normalised against and is
    known to be a physically sane mirror size): a handful of field corners
    can produce runaway rays outside the well-behaved region (the
    polynomial is only softly discouraged, not forbidden, from exceeding
    its normalisation range during training), and sizing the exported mesh
    to their raw max was observed to fold the mesh back on itself -- one
    vertex landed 6 mm from the pupil centre, failing the clearance check
    outright, instead of the ~20 mm the well-behaved region needs."""
    tx, tz = field_grid(nx, nz)
    pupil = hex_pupil_samples(pupil_spacing)
    with torch.no_grad():
        out = trace(params, tx, tz, pupil, aperture=None)
    v = out["valid"]
    (cx1, cy1), (cx2, cy2) = cap

    def bound(t, c):
        return min(float(np.percentile(t[v].abs().numpy(), pct)) * margin, c)

    ax1, ay1 = bound(out["x1"], cx1), bound(out["y1"], cy1)
    ax2, ay2 = bound(out["x2"], cx2), bound(out["y2"], cy2)
    return (ax1, ay1), (ax2, ay2)


def targets(tx, tz):
    u, v = ft.field_to_panel_mm(tx, tz)
    return torch.tensor(u, dtype=DTYPE), torch.tensor(v, dtype=DTYPE)


def loss_and_metrics(params, tx, tz, pupil_xy, aperture=None, weights=None):
    weights = weights or dict(pos=1.0, spot=4.0, tele=0.3, invalid=8.0, reg=1e-6)
    out = trace(params, tx, tz, pupil_xy, aperture=aperture)
    tu, tv = targets(tx, tz)

    valid = out["valid"]
    n_valid = valid.sum(dim=1).clamp(min=1)
    vf = valid.to(DTYPE)

    cu = (out["up"] * vf).sum(dim=1) / n_valid
    cv = (out["vp"] * vf).sum(dim=1) / n_valid
    pos_err2 = (cu - tu) ** 2 + (cv - tv) ** 2

    du = out["up"] - cu[:, None]
    dv = out["vp"] - cv[:, None]
    spot_var = ((du**2 + dv**2) * vf).sum(dim=1) / n_valid

    cos_i = -(out["dir_out"] * out["w_p"]).sum(dim=-1)
    tele = ((1.0 - cos_i) * vf).sum(dim=1) / n_valid

    frac_valid = vf.mean(dim=1)
    invalid_pen = (1.0 - frac_valid) ** 2  # reported, but ~zero-gradient (valid is a hard mask)
    aperture_pen = out["aperture_excess"].mean(dim=1)  # smooth: DOES have gradient

    per_field = (weights["pos"] * pos_err2 + weights["spot"] * spot_var
                + weights["tele"] * tele + weights["invalid"] * invalid_pen
                + weights.get("aperture", 0.05) * aperture_pen)
    reg = sum((params[g]["coeffs"] ** 2).sum() for g in ("m1", "m2"))
    eye_relief = axial_eye_relief_mm(params)
    relief_pen = torch.clamp(22.0 - eye_relief, min=0.0) ** 2  # 2 mm margin over the 20 mm floor
    loss = per_field.mean() + weights["reg"] * reg + weights.get("relief", 2.0) * relief_pen

    metrics = dict(pos_err_mm=torch.sqrt(pos_err2 + 1e-30).detach(),
                   spot_rms_mm=torch.sqrt(spot_var + 1e-30).detach(),
                   tele_1mcos=tele.detach(), frac_valid=frac_valid.detach(),
                   cu=cu.detach(), cv=cv.detach(), tu=tu, tv=tv,
                   eye_relief_mm=eye_relief.detach())
    return loss, metrics, out

"""Foveal inset feasibility: how sharp can a cheap coaxial magnifier show the
whole panel over only the central field (brainstorm idea: the panel, with no
lenslets, spans +/- 10 deg, so a pixel is 7.2 um / 52 mm = 0.47 arcmin, the
foveal retinal pitch)?

    python inset_study.py <out_dir> [--starts 40]

Scored in colour: the panel emits three narrow channels (B, G, R). Lateral
colour and distortion are corrected in rendering (per-channel pre-warp), so
each channel's rays are scored about their own centroid; axial colour, which
no rendering can undo, stays in the blur. Blur is the RMS angular radius of a
field's spot, landing spread over the local focal length, against
foveation_target.blur_tolerance_rad. Materials: moulded PMMA, and a PMMA +
polycarbonate achromatic doublet (Cauchy dispersion from nd and Abbe number).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
sys.path.insert(0, str(HERE.parent / "coaxial_dls"))

import foveation_target as ft  # noqa: E402
import gpu_tracer as gt  # noqa: E402
import screen_spec as spec  # noqa: E402

MATERIALS = {"PMMA": (1.4917, 57.4), "PC": (1.5855, 29.9)}      # nd, Abbe number
CHANNELS_UM = (0.460, 0.530, 0.620)                             # OLED B, G, R peaks
HALF_FIELD_DEG = 10.0                                           # the panel's half-width
FIELDS_DEG = np.array([0.0, 1.0, 2.0, 3.5, 5.0, 6.5, 8.0, 9.0, 10.0, 11.5, 13.0, 14.14])   # to the corner
EYE_RELIEF_MIN_MM = 15.0
MIN_GLASS_MM, MIN_AIR_MM = 1.0, 0.3
R0_MM = 10.0
W_MAP, W_SHAPE = 30.0, 100.0


def index(material, wavelength_um):
    nd, v = MATERIALS[material]
    lf, ld, lc = 0.4861, 0.5876, 0.6563
    b = (nd - 1.0) / v / (1.0 / lf**2 - 1.0 / lc**2)
    return nd - b / ld**2 + b / wavelength_um**2


def pupil_samples():
    pts = [(0.0, 0.0)]
    for r, n in ((0.35, 6), (0.7, 10), (1.0, 14)):
        pts += [(r * math.cos(a), r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    return torch.tensor(pts, dtype=torch.float64)


def layout(materials):
    n = len(materials)
    return {"materials": materials, "size": 1 + 2 * n + 10 * n}


def to_batches(x, lay):
    """Parameter vector (D,) -> one gt.Batch per colour channel (differentiable).
    x = [eye_relief, (thickness, gap_after) per element, (s2, k, b4, b6, b8) per surface]."""
    n = len(lay["materials"])
    steps = [x[0]]
    for e in range(n):
        steps += [x[1 + 2 * e], x[2 + 2 * e]]
    z = torch.cumsum(torch.stack(steps), 0)[None]
    sh = x[1 + 2 * n:].reshape(2 * n, 5)
    c = 2.0 * sh[:, 0] / R0_MM**2
    a = torch.stack([sh[:, 2] / R0_MM**4, sh[:, 3] / R0_MM**6, sh[:, 4] / R0_MM**8], -1)
    zero = torch.zeros(1, dtype=x.dtype)
    batches = []
    for lam in CHANNELS_UM:
        nn = []
        for m in lay["materials"]:
            nn += [index(m, lam), 1.0]
        batches.append(gt.Batch(z=z, c=torch.cat([c, zero])[None], k=torch.cat([sh[:, 1], zero])[None],
                                a=torch.cat([a, torch.zeros(1, 3, dtype=x.dtype)])[None],
                                n=torch.tensor(nn + [1.0], dtype=x.dtype)[None]))
    return batches


TOL = torch.tensor(ft.blur_tolerance_rad(np.radians(FIELDS_DEG), np.zeros_like(FIELDS_DEG)), dtype=torch.float64)
FD = 0.05


def residuals(x, lay):
    pupil = pupil_samples()
    fields = torch.tensor(FIELDS_DEG, dtype=torch.float64)
    res, info = [], {}
    batches = to_batches(x, lay)
    chief = torch.zeros(1, 2, dtype=torch.float64)
    g_land, _, g_ok = gt.trace(batches[1], torch.cat([fields, fields + FD]), chief)
    h = g_land[0, :, 0, 1]
    focal = (h[len(FIELDS_DEG):] - h[:len(FIELDS_DEG)]) / math.radians(FD)          # local focal length, mm/rad
    blur = []
    for b in batches:
        land, _, ok = gt.trace(b, fields, pupil)
        w = ok[0].double()
        n = w.sum(1).clamp(min=1.0)
        cen = (land[0] * w[..., None]).sum(1) / n[:, None]
        dev = (land[0] - cen[:, None]) * w[..., None]
        scale = (TOL * focal.abs().clamp(min=1.0))[:, None, None]
        r = dev / scale / math.sqrt(len(pupil))
        res.append(r.reshape(-1))
        res.append(10.0 * (1.0 - w).reshape(-1))                         # a lost ray costs 10 tolerances
        rms = torch.sqrt((dev**2).sum(-1).sum(1) / n) / focal.abs().clamp(min=1.0) / TOL
        blur.append(torch.where(w.all(1), rms, torch.full_like(rms, math.inf)))   # a lost ray: no image
    i10 = int(np.argmin(np.abs(FIELDS_DEG - HALF_FIELD_DEG)))
    res.append(math.sqrt(W_MAP) * (h[i10] - spec.PANEL_MM / 2)[None])   # the panel edge at 10 deg
    # the corner of the square field lands on the panel corner or inside
    res.append(math.sqrt(W_MAP) * torch.relu(h[len(FIELDS_DEG) - 1] - spec.PANEL_MM / math.sqrt(2.0))[None])
    n_el = len(lay["materials"])
    zs = batches[1].z[0]
    res.append(math.sqrt(W_SHAPE) * torch.relu(EYE_RELIEF_MIN_MM - zs[0])[None])
    used = 1.2 * (2.0 + zs[-1] * math.tan(math.radians(FIELDS_DEG[-1])))            # generous aperture
    for s in range(2 * n_el):
        c0, k0, a0 = batches[1].c[0, s], batches[1].k[0, s], batches[1].a[0, s]
        c1, k1, a1 = batches[1].c[0, s + 1], batches[1].k[0, s + 1], batches[1].a[0, s + 1]
        r2 = torch.as_tensor(used**2, dtype=torch.float64)
        s0, _ = gt.sag(r2, c0, k0, a0)
        s1, _ = gt.sag(r2, c1, k1, a1)
        edge = zs[s + 1] + s1 - zs[s] - s0
        centre = zs[s + 1] - zs[s]
        floor = MIN_GLASS_MM if s % 2 == 0 else MIN_AIR_MM
        res.append(math.sqrt(W_SHAPE) * torch.relu(floor - torch.stack([edge, centre])))
    info["blur"] = torch.stack(blur)                                     # (3, F) in tolerance units
    info["focal"] = focal
    info["alive"] = bool(g_ok.all())
    return torch.cat(res), info


def seed_radii(materials, focal_mm=52.3):
    """Thin-lens seed: a biconvex singlet, or a near-cemented achromat whose
    element powers split as V1 / (V1 - V2) and -V2 / (V1 - V2)."""
    if len(materials) == 1:
        r = 2.0 * focal_mm * (MATERIALS[materials[0]][0] - 1.0)
        return [r, -r]
    (n1, v1), (n2, v2) = MATERIALS[materials[0]], MATERIALS[materials[1]]
    phi = 1.0 / focal_mm
    phi1, phi2 = phi * v1 / (v1 - v2), -phi * v2 / (v1 - v2)
    r1 = 2.0 * (n1 - 1.0) / phi1                                         # element 1 equiconvex (or concave)
    r3 = -r1                                                             # element 2 front matches element 1 back
    r4 = 1.0 / (1.0 / r3 - phi2 / (n2 - 1.0))
    return [r1, -r1, r3, r4]


def fit(lay, rng, starts):
    """Multi-start least squares; returns the best result."""
    n = len(lay["materials"])

    def fun(v):
        with torch.no_grad():
            return residuals(torch.tensor(v, dtype=torch.float64), lay)[0].numpy()

    def jac(v):
        return torch.autograd.functional.jacobian(lambda t: residuals(t, lay)[0], torch.tensor(v, dtype=torch.float64),
                                                  vectorize=True, strategy="forward-mode").numpy()

    results = []
    for _ in range(starts):
        x0 = np.zeros(lay["size"])
        x0[0] = rng.uniform(16.0, 22.0)
        for e in range(n):
            x0[1 + 2 * e] = rng.uniform(3.0, 9.0)
            x0[2 + 2 * e] = rng.uniform(0.5, 3.0) if e < n - 1 else rng.uniform(35.0, 55.0)
        sh = rng.normal(0.0, 0.1, (2 * n, 5))
        sh[:, 1] = rng.normal(0.0, 0.3, 2 * n)
        sh[:, 0] = [R0_MM**2 / (2 * r) * rng.uniform(0.8, 1.2) for r in seed_radii(lay["materials"])]
        x0[1 + 2 * n:] = sh.ravel()
        results.append(least_squares(fun, x0, jac=jac, method="trf", max_nfev=300, x_scale="jac"))
    return min(results, key=lambda r: r.cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--starts", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    report = {}
    for name, mats in (("singlet_PMMA", ["PMMA"]), ("doublet_PMMA_PC", ["PMMA", "PC"]),
                       ("doublet_PC_PMMA", ["PC", "PMMA"])):
        lay = layout(mats)
        best = fit(lay, np.random.default_rng(0), args.starts)
        with torch.no_grad():
            _, info = residuals(torch.tensor(best.x, dtype=torch.float64), lay)
        blur = info["blur"].numpy()
        report[name] = {"cost": float(best.cost), "x": best.x.tolist(), "alive": info["alive"],
                        "fields_deg": FIELDS_DEG.tolist(), "local_focal_mm": info["focal"].numpy().tolist(),
                        "blur_over_tolerance_BGR": blur.tolist(),
                        "worst_channel_blur": blur.max(0).tolist()}
        print(name, "cost", round(float(best.cost), 3), "worst-channel blur/tolerance per field:",
              " ".join(f"{b:.2f}" for b in blur.max(0)), "focal(0) mm", round(float(info["focal"][0]), 2),
              flush=True)
    (out / "inset_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

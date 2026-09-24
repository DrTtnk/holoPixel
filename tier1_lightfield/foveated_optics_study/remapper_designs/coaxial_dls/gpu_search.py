"""Global search for coaxial remappers on the GPU: thousands of random designs
optimised together with Adam through the differentiable tracer (gpu_tracer).

    python gpu_search.py <out_dir> --elements 4 --designs 2048 [--curved-image] [--steps 2500]

Parameter vector per design (sags in mm at R0, as in fast_merit):
    [eye_relief, (t_glass, gap_after) per element, (s2, k, b4, b6, b8) per surface,
     (s2_img, k_img) if the image surface is curved (fibre-optic faceplate)]

Target: the round, temporal-meridian map of foveation_target_radial (a round
design cannot follow the anamorphic, left-right symmetric foveation_target).
Image surface: flat, a uniform lenslet array of foveation_target_radial.lenslet_focal_um().

Merit, per design (all landing errors in units of the blur the eye tolerates
there, foveation_target.blur_tolerance_rad times the local focal length):
  land   per-pixel blur: every pupil ray's miss from the centroid of its pupil
         cell (the part of the pupil one pixel sees, 0.8 mm with these
         lenslets, as in freeform_mirror/fold_search.py); a lost ray costs LOST
  chief  the chief ray's height miss (the mapping itself), weighted up
  tilt   chief-ray angle at the image above TILT_MAX_DEG
  shape  glass thinner than 1 mm or air thinner than 0.3 mm at the radius the rays use
  margin anticipatory: sin^2 of refraction past 0.85 (near TIR), sag domain below 0.05
  track  total length (eye to image) beyond TRACK_MAX_MM
The mapping sign is free (the encoder calibrates it), so the target point is
+/- r(theta) with the sign each design's chief ray picks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import foveation_target_radial as ft  # noqa: E402  (a round design can only follow a round map)
import screen_spec as spec  # noqa: E402

import gpu_tracer as gt  # noqa: E402

R0_MM = 20.0
INDICES = (1.49, 1.51, 1.53)   # 3D-printable clear resins and PMMA
MAX_FIELD_DEG = 41.6852
TILT_MAX_DEG = 30.0
TRACK_MAX_MM = 80.0
MIN_GLASS_MM, MIN_AIR_MM = 1.0, 0.3
LOST = 100.0                  # squared residual of a lost ray, in tolerance units
W_CHIEF, W_TILT, W_SHAPE, W_MARGIN, W_TRACK = 9.0, 1.0, 100.0, 50.0, 1.0
PUPIL_SPACING_MM = 0.4
def pupil_samples(spacing_mm=PUPIL_SPACING_MM):
    """Hexagonal grid inside the pupil disc, normalised to its radius."""
    r = spec.PUPIL_DIAMETER_MM / 2.0
    n = int(math.ceil(r / spacing_mm)) + 1
    pts = [(i * spacing_mm + (j % 2) * spacing_mm / 2, j * spacing_mm * math.sqrt(3) / 2)
           for i in range(-n, n + 1) for j in range(-n, n + 1)]
    return np.array([p for p in pts if math.hypot(*p) <= r + 1e-9]) / r


def pupil_cells(fields_deg, pupil):
    """(F, P, K) one-hot: the pupil cell each ray belongs to, per field; a cell is
    a pixel's footprint on the pupil, pixel * F(theta) / f_lenslet(theta)."""
    th = np.radians(fields_deg)
    side = spec.PIXEL_UM * ft.local_focal_mm(th) / ft.lenslet_focal_um()
    uv = pupil * spec.PUPIL_DIAMETER_MM / 2.0 + spec.PUPIL_DIAMETER_MM / 2.0
    ids = np.zeros((len(fields_deg), len(pupil)), dtype=np.int64)
    for f, a in enumerate(side):
        n = max(1, int(math.ceil(spec.PUPIL_DIAMETER_MM / a)))
        cell = np.minimum((uv / a).astype(np.int64), n - 1)
        _, ids[f] = np.unique(cell[:, 0] * n + cell[:, 1], return_inverse=True)
    return np.eye(int(ids.max()) + 1)[ids]


def cell_spread(land, alive, onehot):
    """Each ray's landing offset from its pupil cell's centroid (B, F, P, 2), zero if lost."""
    w = alive.double()
    count = torch.einsum("bfp,fpk->bfk", w, onehot).clamp(min=1.0)
    centroid = torch.einsum("bfpc,fpk->bfkc", land * w[..., None], onehot) / count[..., None]
    return (land - torch.einsum("bfkc,fpk->bfpc", centroid, onehot)) * w[..., None]


def field_grid(n=15):
    return np.linspace(0.0, 1.0, n) ** 1.5 * MAX_FIELD_DEG   # denser near the fovea


def tolerance_mm(fields_deg):
    th = np.radians(fields_deg)
    return ft.blur_tolerance_rad(th, np.zeros_like(th)) * ft.local_focal_mm(th)


def layout(n_el, curved, lenslets="uniform"):
    """lenslets: "uniform" (a flat image surface), the only kind a round design uses."""
    n_shape = 2 * n_el * 5
    return {"n_el": n_el, "curved": curved, "lenslets": lenslets,
            "size": 1 + 2 * n_el + n_shape + (2 if curved else 0)}


def bounds(lay, device):
    n = lay["n_el"]
    lo, hi = [20.5], [30.0]
    for i in range(n):
        lo += [1.5, 0.3]
        hi += [20.0, 30.0 if i < n - 1 else 60.0]
    for _ in range(2 * n):
        lo += [-40.0, -20.0, -8.0, -8.0, -8.0]
        hi += [40.0, 20.0, 8.0, 8.0, 8.0]
    if lay["curved"]:
        lo += [-15.0, -5.0]
        hi += [15.0, 5.0]
    return torch.tensor(lo, dtype=torch.float64, device=device), torch.tensor(hi, dtype=torch.float64, device=device)


def random_designs(lay, count, rng, device):
    n = lay["n_el"]
    x = np.zeros((count, lay["size"]))
    x[:, 0] = rng.uniform(20.5, 23.0, count)
    for i in range(n):
        x[:, 1 + 2 * i] = rng.uniform(2.0, 8.0, count)
        x[:, 2 + 2 * i] = rng.uniform(0.5, 8.0, count) if i < n - 1 else rng.uniform(2.0, 25.0, count)
    shape = x[:, 1 + 2 * n:1 + 2 * n + 10 * n].reshape(count, 2 * n, 5)
    shape[:, :, 0] = rng.uniform(-8.0, 8.0, (count, 2 * n))
    shape[:, :, 1] = rng.normal(0.0, 0.5, (count, 2 * n))
    x[:, 1 + 2 * n:1 + 2 * n + 10 * n] = shape.reshape(count, -1)
    indices = rng.choice(INDICES, size=(count, n))
    return torch.tensor(x, dtype=torch.float64, device=device), torch.tensor(indices, dtype=torch.float64, device=device)


def to_batch(x, indices, lay):
    """Parameter vectors (B, D) -> gt.Batch (differentiable)."""
    B, n = x.shape[0], lay["n_el"]
    steps = [x[:, 0]]
    for i in range(n):
        steps += [x[:, 1 + 2 * i], x[:, 2 + 2 * i]]
    z = torch.cumsum(torch.stack(steps, 1), dim=1)               # (B, 2n+1): surfaces then image
    shape = x[:, 1 + 2 * n:1 + 2 * n + 10 * n].reshape(B, 2 * n, 5)
    c = 2.0 * shape[:, :, 0] / R0_MM**2
    k = shape[:, :, 1]
    a = torch.stack([shape[:, :, 2] / R0_MM**4, shape[:, :, 3] / R0_MM**6, shape[:, :, 4] / R0_MM**8], -1)
    if lay["curved"]:
        c_img = (2.0 * x[:, -2] / R0_MM**2)[:, None]
        k_img = x[:, -1][:, None]
    else:
        c_img = torch.zeros(B, 1, dtype=x.dtype, device=x.device)
        k_img = torch.zeros(B, 1, dtype=x.dtype, device=x.device)
    if lay["lenslets"] != "uniform":
        raise ValueError(f"a coaxial design uses a uniform lenslet array, not {lay['lenslets']!r}")
    zeros3 = torch.zeros(B, 1, 3, dtype=x.dtype, device=x.device)
    n_after = torch.stack([indices, torch.ones_like(indices)], -1).reshape(B, 2 * n)
    return gt.Batch(z=z, c=torch.cat([c, c_img], 1), k=torch.cat([k, k_img], 1),
                    a=torch.cat([a, zeros3], 1), n=torch.cat([n_after, torch.ones(B, 1, dtype=x.dtype,
                                                                                    device=x.device)], 1))


def merit(x, indices, lay, fields, pupil, tol, target, onehot):
    batch = to_batch(x, indices, lay)
    land, d, alive, diag = gt.trace(batch, fields, pupil, diagnostics=True)
    B = x.shape[0]
    chief_y = land[:, :, 0, 1]
    sign = torch.sign(chief_y[:, -3].detach())
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    ty = sign[:, None] * target[None, :]                           # (B, F)
    err2 = (cell_spread(land, alive, onehot) ** 2).sum(-1) / tol[None, :, None] ** 2
    err2 = torch.where(alive, err2, torch.full_like(err2, LOST))
    land_term = err2.mean(dim=(1, 2))
    chief_err = torch.where(alive[:, :, 0], (chief_y - ty) / tol[None, :], torch.full_like(chief_y, 10.0))
    chief_term = W_CHIEF * (chief_err**2).mean(1)
    tilt = torch.rad2deg(torch.acos(torch.clamp(d[:, :, 0, 2], -1.0 + 1e-12, 1.0 - 1e-12)))
    tilt = torch.where(alive[:, :, 0], tilt, torch.zeros_like(tilt))
    tilt_term = W_TILT * (torch.relu(tilt - TILT_MAX_DEG) ** 2).mean(1)

    r = diag["radius"]                                             # (B, S, F, P)
    used = torch.where(alive[:, None], r, torch.zeros_like(r)).amax(dim=(2, 3)).detach() * 1.05
    S = batch.z.shape[1]
    shape_term = torch.zeros(B, dtype=x.dtype, device=x.device)
    for s in range(S - 1):
        rr = torch.maximum(used[:, s], used[:, s + 1]) ** 2
        s0, _ = gt.sag(rr, batch.c[:, s], batch.k[:, s], batch.a[:, s])
        s1, _ = gt.sag(rr, batch.c[:, s + 1], batch.k[:, s + 1], batch.a[:, s + 1])
        thick = batch.z[:, s + 1] + s1 - batch.z[:, s] - s0
        floor = MIN_GLASS_MM if (s % 2 == 0 and s < S - 1) else MIN_AIR_MM
        shape_term = shape_term + W_SHAPE * torch.relu(floor - thick) ** 2
    live = alive[:, None].double()
    margin = (torch.relu(diag["sin2t"] - 0.85) ** 2 * live[:, :diag["sin2t"].shape[1]]).mean(dim=(1, 2, 3)) + \
        (torch.relu(0.05 - diag["domain"]) ** 2 * live).mean(dim=(1, 2, 3))
    track = torch.relu(batch.z[:, -1] - TRACK_MAX_MM) ** 2
    total = land_term + chief_term + tilt_term + shape_term + W_MARGIN * margin + W_TRACK * track
    return total, {"land": land_term, "chief": chief_term, "alive": alive.float().mean(dim=(1, 2)),
                   "shape": shape_term, "tilt_max": tilt.amax(1)}


def spot_in_tolerance(x, indices, lay, fields, pupil, tol, onehot):
    """Worst pupil cell's RMS spot radius (per-pixel blur), per field, in tolerance units."""
    with torch.no_grad():
        land, _, alive = gt.trace(to_batch(x, indices, lay), fields, pupil)
        sq = torch.einsum("bfp,fpk->bfk", (cell_spread(land, alive, onehot) ** 2).sum(-1), onehot)
        count = torch.einsum("bfp,fpk->bfk", alive.double(), onehot)
        rms = torch.sqrt(sq / count.clamp(min=1.0))
        return torch.where(count > 0, rms, torch.zeros_like(rms)).amax(-1) / tol[None, :]


def run(out_dir, n_el, curved, count, steps, seed, device):
    lay = layout(n_el, curved)
    rng = np.random.default_rng(seed)
    x, indices = random_designs(lay, count, rng, device)
    lo, hi = bounds(lay, device)
    x = torch.clamp(x, lo, hi).requires_grad_(True)
    fields_np = field_grid()
    fields = torch.tensor(fields_np, dtype=torch.float64, device=device)
    pupil = torch.tensor(pupil_samples(), dtype=torch.float64, device=device)
    onehot = torch.tensor(pupil_cells(fields_np, pupil_samples()), dtype=torch.float64, device=device)
    tol = torch.tensor(tolerance_mm(fields_np), dtype=torch.float64, device=device)
    target = torch.tensor(ft.panel_radius_mm(np.radians(fields_np)), dtype=torch.float64, device=device)
    asph = torch.zeros(lay["size"], dtype=torch.bool, device=device)
    for s in range(2 * n_el):
        asph[1 + 2 * n_el + 5 * s + 2: 1 + 2 * n_el + 5 * s + 5] = True
    opt = torch.optim.Adam([x], lr=0.02)
    t0 = time.time()
    for it in range(steps):
        opt.zero_grad()
        loss, info = merit(x, indices, lay, fields, pupil, tol, target, onehot)
        loss.sum().backward()
        if it < steps // 3:
            x.grad[:, asph] = 0.0                                  # spheres + conics first
        if not torch.isfinite(x.grad).all():
            raise FloatingPointError(f"non-finite gradient at step {it}")
        opt.step()
        with torch.no_grad():
            x.clamp_(lo, hi)
        if it % 250 == 0 or it == steps - 1:
            ld = loss.detach()
            if not torch.isfinite(ld).all():
                raise FloatingPointError(f"non-finite merit for {int((~torch.isfinite(ld)).sum())} designs at step {it}")
            best = int(torch.argmin(ld))
            print(f"[{n_el} el, {'curved' if curved else 'flat'} image] it {it:5d}  best {float(ld[best]):10.3f}  "
                  f"median {float(ld.median()):10.3f}  alive(best) {float(info['alive'][best]):.2f}  "
                  f"{time.time() - t0:6.0f} s", flush=True)
    with torch.no_grad():
        loss, info = merit(x, indices, lay, fields, pupil, tol, target, onehot)
    spot = spot_in_tolerance(x.detach(), indices, lay, fields, pupil, tol, onehot)
    order = torch.argsort(loss)[:20]
    out = []
    for rank, b in enumerate(order.tolist()):
        out.append({"rank": rank, "loss": float(loss[b]), "alive": float(info["alive"][b]),
                    "shape_penalty": float(info["shape"][b]), "tilt_max_deg": float(info["tilt_max"][b]),
                    "spot_in_tolerance_per_field": spot[b].tolist(), "fields_deg": fields_np.tolist(),
                    "indices": indices[b].tolist(), "x": x[b].detach().tolist(), "layout": lay})
    tag = f"el{n_el}_{'curved' if curved else 'flat'}"
    (Path(out_dir) / f"best_{tag}.json").write_text(json.dumps(out, indent=1))
    top = out[0]
    print(f"TOP {tag}: loss {top['loss']:.3f} alive {top['alive']:.2f} shape {top['shape_penalty']:.3f} "
          f"tilt {top['tilt_max_deg']:.1f}  spot/tolerance per field: "
          + " ".join(f"{v:.2f}" for v in top["spot_in_tolerance_per_field"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--elements", type=int, required=True)
    ap.add_argument("--designs", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--curved-image", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(args.out_dir, args.elements, args.curved_image, args.designs, args.steps, args.seed, torch.device("cuda"))


if __name__ == "__main__":
    main()

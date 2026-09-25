"""Pancake (polarisation-folded) light-field remapper, searched with fold_search's
merit and optimiser.

    python pancake_search.py <out_dir> --elements 1 --material resin [--designs 128] [--iters 90]

Layout, eye -> panel, all surfaces on the eye's axis (no tilt; plane-symmetric
XY polynomials, so the anamorphic target can be followed):
  P    flat reflective polariser at z_p (the first pass through it changes
       nothing and is not traced)
  HM   the half-mirror coating on the front of lens 1, at z_p + gap: reflects
       the ray back towards the eye
  P    reflects it forward again
  HM   now transmits it into lens 1 (index n1); lens 1's back surface follows
  --   optional correctors, then the image surface: the variable-focal
       lenslet array's vertex surface
The polarisation bookkeeping (quarter-wave plate, 25 % of polarised light at
best) is ideal and not traced. Cycles has no polarisation either, but renders
the same ideal path with shaders switched by the ray's glossy-bounce count
(prototype checked against an independent numpy trace: 0 ghost pixels, max
landing error 0.0125 mm).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import fold_search as fs
import offaxis_tracer as ot

EYE_RELIEF_MIN_MM = 15.0
TRACK_MAX_MM = 40.0          # pupil to lenslet array: the point of a pancake
HM_R0_MM, LENS_R0_MM = 20.0, 10.0
LENS_SHAPE_MM = 20.0         # bound of the lens shape parameters: the 70 x 45 best lens back sat at +-10


def layout(n_el, flip_u=1, flip_v=1):
    """n_el lenses: lens 1 carries the half-mirror, the others are correctors."""
    i = 0

    def take(n):
        nonlocal i
        s = slice(i, i + n)
        i += n
        return s

    lay = {"family": "pancake", "n_el": n_el, "flip_u": flip_u, "flip_v": flip_v, "cavity": take(2),
           "hm_shape": take(fs.N_SHAPE), "lens1": {"thickness": take(1), "back": take(fs.N_SHAPE)}, "elements": []}
    for _ in range(n_el - 1):
        lay["elements"].append({"pose": take(2), "front": take(fs.N_SHAPE), "back": take(fs.N_SHAPE)})
    lay["image"] = take(1)
    lay["size"] = i
    lay["shapes"] = ([lay["hm_shape"], lay["lens1"]["back"]]
                     + [el[side] for el in lay["elements"] for side in ("front", "back")])
    return lay


def bounds(lay, device):
    lo, hi = np.full(lay["size"], -np.inf), np.full(lay["size"], np.inf)

    def put(s, low, high):
        lo[s], hi[s] = low, high

    put(lay["cavity"], [EYE_RELIEF_MIN_MM, 1.0], [30.0, 20.0])            # z_p, polariser -> half-mirror gap
    put(lay["hm_shape"], -10.0, 10.0)
    put(lay["lens1"]["thickness"], fs.MIN_GLASS_MM, 10.0)
    put(lay["lens1"]["back"], -LENS_SHAPE_MM, LENS_SHAPE_MM)
    for el in lay["elements"]:
        put(el["pose"], [0.3, fs.MIN_GLASS_MM], [20.0, 8.0])             # gap before, thickness
        put(el["front"], -LENS_SHAPE_MM, LENS_SHAPE_MM)
        put(el["back"], -LENS_SHAPE_MM, LENS_SHAPE_MM)
    put(lay["image"], 0.3, 30.0)
    t = lambda a: torch.tensor(a, dtype=torch.float64, device=device)  # noqa: E731
    return t(lo), t(hi)


def random_designs(lay, count, rng, device):
    x = np.zeros((count, lay["size"]))
    x[:, lay["cavity"]] = np.column_stack([rng.uniform(15.0, 19.0, count), rng.uniform(3.0, 10.0, count)])
    hm = np.zeros((count, fs.N_SHAPE))
    radius = -rng.uniform(40.0, 120.0, count)                              # concave towards the eye
    hm[:, 0] = HM_R0_MM**2 / (2.0 * radius)
    x[:, lay["hm_shape"]] = hm
    x[:, lay["lens1"]["thickness"]] = rng.uniform(2.0, 5.0, count)[:, None]
    back = np.zeros((count, fs.N_SHAPE))
    back[:, 0] = rng.normal(0.0, 0.5, count)
    x[:, lay["lens1"]["back"]] = back
    for el in lay["elements"]:
        x[:, el["pose"]] = np.column_stack([rng.uniform(0.5, 3.0, count), rng.uniform(1.5, 4.0, count)])
        for side in ("front", "back"):
            sh = np.zeros((count, fs.N_SHAPE))
            sh[:, 0] = rng.normal(0.0, 0.5, count)
            x[:, el[side]] = sh
    x[:, lay["image"]] = rng.uniform(3.0, 15.0, count)[:, None]
    return torch.tensor(x, dtype=torch.float64, device=device)


def to_batch(x, indices, lay):
    B = x.shape[0]
    zero = torch.zeros(B, dtype=x.dtype, device=x.device)
    one = torch.ones_like(zero)
    z_p, gap = x[:, lay["cavity"]].unbind(1)
    z_hm = z_p + gap
    hm = fs._shape(x[:, lay["hm_shape"]], HM_R0_MM)
    flat = (zero, zero, torch.zeros(B, len(fs.TERMS), dtype=x.dtype, device=x.device))
    zs, cs, ks, Cs, ns = [], [], [], [], []

    def add(z, shape, n_after):
        zs.append(z)
        cs.append(shape[0])
        ks.append(shape[1])
        Cs.append(shape[2])
        ns.append(n_after)

    add(z_hm, hm, one)                                                   # half-mirror: reflect
    add(z_p, flat, one)                                                  # polariser: reflect
    add(z_hm, hm, indices[:, 0])                                         # half-mirror: transmit into lens 1
    z = z_hm + x[:, lay["lens1"]["thickness"]][:, 0]
    add(z, fs._shape(x[:, lay["lens1"]["back"]], fs.LENS_R0_MM), one)
    for e, el in enumerate(lay["elements"]):
        g, t = x[:, el["pose"]].unbind(1)
        z = z + g
        add(z, fs._shape(x[:, el["front"]], fs.LENS_R0_MM), indices[:, e + 1])
        z = z + t
        add(z, fs._shape(x[:, el["back"]], fs.LENS_R0_MM), one)
    add(z + x[:, lay["image"]][:, 0], flat, one)
    st = lambda v: torch.stack(v, 1)  # noqa: E731
    S = len(zs)
    return ot.Batch(y=torch.zeros(B, S, dtype=x.dtype, device=x.device), z=st(zs),
                    rx=torch.zeros(B, S, dtype=x.dtype, device=x.device), c=st(cs), k=st(ks), xy=st(Cs), n=st(ns),
                    mirror=(True, True) + (False,) * (S - 2), terms=fs.TERMS,
                    image_sag=fs.bowl(x.device, lay["flip_v"], light_along_z=1))


def constraints(batch, diag, alive, lay, x):
    """Space: the lenslet array within TRACK_MAX_MM of the pupil. Barriers: air
    and glass floors along every ray, TIR and sag-domain margins."""
    pts = diag["points"]                                                 # (B, S, F, P, 3)
    live = alive[:, None].double()
    space = torch.relu(batch.z[:, -1] - TRACK_MAX_MM) ** 2 * 100.0
    seg = torch.linalg.norm(pts[:, 1:] - pts[:, :-1], dim=-1)            # HM -> P -> HM -> back -> ... -> image
    floor = [fs.MIN_AIR_MM, fs.MIN_AIR_MM, fs.MIN_GLASS_MM] + [fs.MIN_AIR_MM, fs.MIN_GLASS_MM] * (lay["n_el"] - 1) \
        + [fs.MIN_AIR_MM]
    floor = torch.tensor(floor, dtype=x.dtype, device=x.device)
    barrier = (torch.relu(floor[None, :, None, None] - seg) ** 2 * live).sum((1, 2, 3))
    barrier = barrier + (torch.relu(diag["sin2t"][:, :-1] - fs.SIN2_MAX) ** 2 * live).sum((1, 2, 3))
    barrier = barrier + (torch.relu(fs.DOMAIN_MIN - diag["domain"]) ** 2 * alive[:, None].double()).sum((1, 2, 3))
    return space, barrier


fs.register("pancake", layout=layout, bounds=bounds, random_designs=random_designs, to_batch=to_batch,
            constraints=constraints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--elements", type=int, required=True)
    ap.add_argument("--material", choices=sorted(fs.INDEX_SETS), required=True)
    ap.add_argument("--designs", type=int, default=128)
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-from", nargs="*", default=[], help="best_*.json files whose designs join the seeds")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    ap.add_argument("--spline-cells", type=int, default=0,
                    help="add a B-spline of this many cells across to the half-mirror (needs --seed-from)")
    ap.add_argument("--seed-other-field", action="store_true",
                    help="accept --seed-from designs searched for another field (a field continuation)")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    fs.run(args.out_dir, args.elements, args.material, args.designs, args.iters, args.seed, torch.device("cuda"),
           family="pancake", seed_from=args.seed_from, tag_suffix=args.tag, spline_cells=args.spline_cells,
           seed_other_field=args.seed_other_field)


if __name__ == "__main__":
    main()

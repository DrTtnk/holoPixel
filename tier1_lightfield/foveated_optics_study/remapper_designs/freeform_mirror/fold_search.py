"""GPU search for a folded remapper: panel above the eye facing down, a tilted
concave freeform mirror in front of the eye, and 1-2 freeform corrector
elements in the return path (the optimiser places them).

    python fold_search.py <out_dir> --elements 1 --material resin [--designs 128] [--iters 90]

Frame (offaxis_optiland.py): eye pupil at the origin, z forward, y up. The
mirror at (y_m, z_m), tilted alpha about x, sends the axis up and back along
u = (sin 2 alpha, -cos 2 alpha) in (y, z). Correctors and the image surface
(the lenslet array face) sit along u at free distances, lateral offsets and
tilts. Plane-symmetric: every surface is an XY polynomial even in x.

Merit, in the units the acceptance evaluator (lf_evaluate.py) uses:
  blur    per-pixel beam width. A pixel behind a lenslet collects the rays from
          one pupil cell of size pixel * F(theta) / f_lenslet (F: local focal
          length of the retina-matched target; the variable-focal lenslets make
          every cell pixel * pupil / pitch = 0.8 mm); the rays of a cell must
          land together on the image surface, the lens-vertex bowl.
          Their landing spread, mapped to field angle by the design's own
          local Jacobian, is scored against foveation_target.blur_tolerance_rad.
  ratio   the local magnification (singular values of the Jacobian) over the
          target F(theta) must stay in RATIO_BAND, isotropically.
  panel   the whole field lands on the 18.4 mm panel.
  tilt    chief rays reach the lenslet array within TILT_MAX_DEG of its normal.
  space   no hardware (corrector hits, panel corners) inside the eye's view
          cone to the mirror, nearer the pupil than CLEARANCE_MM, or behind
          the face plane MIN_Z_MM; glass at least MIN_GLASS_MM along every ray.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

import foveation_target as ft  # noqa: E402
import variable_lenslets as vl  # noqa: E402
import offaxis_tracer as ot  # noqa: E402
import screen_spec as spec  # noqa: E402

INDEX_SETS = {"resin": (1.49, 1.51, 1.53), "glass": (1.72, 1.80, 1.90)}
FIELDS_X_DEG = (0.0, 1.0, 2.5, 5.0, 10.0, 17.0, 24.0, 30.0, 35.0)
FIELDS_Y_DEG = (-22.5, -15.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 15.0, 22.5)
FD_DEG = 0.1
PUPIL_SPACING_MM = 0.4
LENSLETS = "variable_retina"     # lens focal length follows the local F (foveation_target)
TOP_FIELD_DEG = 22.5
CONE_MARGIN_MM = 2.0            # rim of an element beyond its footprint, plus air
BAFFLE_OFFSET_MM = 0.5          # baffle plane above the top ray of the view cone
BAFFLE_CLEAR_MM = 1.0           # return rays cross the baffle plane this far beyond the hardware
CLEARANCE_MM, MIN_Z_MM = 15.0, 12.0
MIN_GLASS_MM, MIN_AIR_MM = 1.0, 0.5
TILT_MAX_DEG = 30.0
SIN2_MAX, DOMAIN_MIN = 0.9, 0.05  # TIR and sag-domain barriers start here
RATIO_BAND = (0.85, 1.2)
PANEL_HALF_MM = spec.PANEL_MM / 2.0 - 0.1
MIRROR_R0_MM, LENS_R0_MM = 20.0, 10.0
TERMS = ((0, 2), (2, 0), (0, 3), (2, 1), (0, 4), (2, 2), (4, 0), (0, 5), (2, 3), (4, 1), (0, 6), (2, 4), (4, 2),
         (6, 0))
LOW_ORDER = 4                   # (0,2) (2,0) (0,3) (2,1): free from the first step
N_SHAPE = 2 + len(TERMS)        # base sag at R0, conic, polynomial sags at R0
POLY = 7
MAP_SCALE_MM = 0.1               # chief landing vs the target map's panel point
W = {"blur": 1.0, "hinge": 10.0, "ratio": 30.0, "panel": 30.0, "tilt": 1.0, "space": 3.0, "barrier": 30.0,
     "map": 10.0}


def fold_layout(n_el, flip_u=1, flip_v=1):
    """Slices of the parameter vector, and the image orientation of this layout
    family relative to the target map (+1 upright, -1 mirrored), measured by
    orientation() on its seeds."""
    i = 0

    def take(n):
        nonlocal i
        s = slice(i, i + n)
        i += n
        return s

    lay = {"family": "fold", "n_el": n_el, "flip_u": flip_u, "flip_v": flip_v, "mirror_pose": take(3),
           "mirror_shape": take(N_SHAPE), "elements": []}
    for _ in range(n_el):
        lay["elements"].append({"pose": take(5), "front": take(N_SHAPE), "back": take(N_SHAPE)})
    lay["image"] = take(3)
    lay["size"] = i
    lay["shapes"] = [lay["mirror_shape"]] + [el[side] for el in lay["elements"] for side in ("front", "back")]
    return lay


def fold_bounds(lay, device):
    lo, hi = np.full(lay["size"], -np.inf), np.full(lay["size"], np.inf)

    def put(s, low, high):
        lo[s], hi[s] = low, high

    put(lay["mirror_pose"], [25.0, -10.0, 5.0], [60.0, 10.0, 40.0])     # z_m, y_m, alpha (deg)
    put(lay["mirror_shape"], -20.0, 20.0)
    for el in lay["elements"]:
        put(el["pose"], [0.3, -15.0, -40.0, MIN_GLASS_MM, -30.0], [60.0, 15.0, 40.0, 15.0, 30.0])
        put(el["front"], -10.0, 10.0)
        put(el["back"], -10.0, 10.0)
    put(lay["image"], [0.3, -15.0, -45.0], [80.0, 15.0, 45.0])            # gap, lateral, tilt (deg)
    t = lambda a: torch.tensor(a, dtype=torch.float64, device=device)  # noqa: E731
    return t(lo), t(hi)


def fold_random_designs(lay, count, rng, device):
    x = np.zeros((count, lay["size"]))
    s_img = rng.uniform(22.0, 40.0, count)
    x[:, lay["mirror_pose"]] = np.column_stack([rng.uniform(30.0, 45.0, count), rng.normal(0.0, 2.0, count),
                                                rng.uniform(20.0, 35.0, count)])
    ms = np.zeros((count, N_SHAPE))
    ms[:, 0] = -MIRROR_R0_MM**2 / (4.0 * s_img) + rng.normal(0.0, 0.3, count)   # sphere R = -2 s_img
    ms[:, 1] = rng.normal(0.0, 0.3, count)
    x[:, lay["mirror_shape"]] = ms
    pos = rng.uniform(0.65, 0.85, count) * s_img        # near the panel, where the beam is narrow
    for el in lay["elements"]:
        t = rng.uniform(2.0, 5.0, count)
        x[:, el["pose"]] = np.column_stack([pos, rng.normal(0.0, 1.0, count), rng.normal(0.0, 3.0, count), t,
                                            rng.normal(0.0, 2.0, count)])
        for side in ("front", "back"):
            sh = np.zeros((count, N_SHAPE))
            sh[:, 0] = rng.normal(0.0, 0.5, count)
            x[:, el[side]] = sh
        pos = rng.uniform(0.3, 3.0, count)
    x[:, lay["image"]] = np.column_stack([rng.uniform(1.0, 5.0, count), rng.normal(0.0, 1.0, count),
                                          rng.normal(0.0, 5.0, count)])
    return torch.tensor(x, dtype=torch.float64, device=device)


def _shape(p, r0):
    """(B, N_SHAPE) -> curvature, conic, (B, POLY, POLY) coefficients."""
    B = p.shape[0]
    C = torch.zeros(B, POLY, POLY, dtype=p.dtype, device=p.device)
    for n, (i, j) in enumerate(TERMS):
        C[:, i, j] = p[:, 2 + n] / r0 ** (i + j)
    return 2.0 * p[:, 0] / r0**2, p[:, 1], C


def fold_to_batch(x, indices, lay):
    B = x.shape[0]
    zero = torch.zeros(B, dtype=x.dtype, device=x.device)
    z_m, y_m, a_deg = x[:, lay["mirror_pose"]].unbind(1)
    a = torch.deg2rad(a_deg)
    u = torch.stack([torch.sin(2 * a), -torch.cos(2 * a)], 1)            # (y, z)
    up = torch.stack([torch.cos(2 * a), torch.sin(2 * a)], 1)
    M = torch.stack([y_m, z_m], 1)
    ys, zs, rxs, cs, ks, Cs, ns = [], [], [], [], [], [], []

    def add(pos, rx, shape, n_after):
        ys.append(pos[:, 0])
        zs.append(pos[:, 1])
        rxs.append(rx)
        cs.append(shape[0])
        ks.append(shape[1])
        Cs.append(shape[2])
        ns.append(n_after)

    add(M, a, _shape(x[:, lay["mirror_shape"]], MIRROR_R0_MM), torch.ones_like(zero))
    s = zero
    for e, el in enumerate(lay["elements"]):
        gap, lat, tilt, thick, back_tilt = x[:, el["pose"]].unbind(1)
        s = s + gap
        front = M + s[:, None] * u + lat[:, None] * up
        rx = 2 * a + torch.deg2rad(tilt)
        add(front, rx, _shape(x[:, el["front"]], LENS_R0_MM), indices[:, e])
        add(front + thick[:, None] * u, rx + torch.deg2rad(back_tilt), _shape(x[:, el["back"]], LENS_R0_MM),
            torch.ones_like(zero))
        s = s + thick
    gap, lat, tilt = x[:, lay["image"]].unbind(1)
    s = s + gap
    flat = (zero, zero, torch.zeros(B, POLY, POLY, dtype=x.dtype, device=x.device))
    add(M + s[:, None] * u + lat[:, None] * up, 2 * a + torch.deg2rad(tilt), flat, torch.ones_like(zero))
    st = lambda v: torch.stack(v, 1)  # noqa: E731
    return ot.Batch(y=st(ys), z=st(zs), rx=st(rxs), c=st(cs), k=st(ks), xy=st(Cs), n=st(ns),
                    mirror=(True,) + (False,) * (2 * lay["n_el"] + 1), image_sag=bowl(x.device, lay["flip_v"]))


FAMILIES = {}


def register(name, **functions):
    """An optical family: layout, bounds, random_designs, to_batch, constraints."""
    FAMILIES[name] = functions


def layout(n_el, flip_u=1, flip_v=1, family="fold"):
    return FAMILIES[family]["layout"](n_el, flip_u, flip_v)


def bounds(lay, device):
    return FAMILIES[lay["family"]]["bounds"](lay, device)


def random_designs(lay, count, rng, device):
    return FAMILIES[lay["family"]]["random_designs"](lay, count, rng, device)


def to_batch(x, indices, lay):
    return FAMILIES[lay["family"]]["to_batch"](x, indices, lay)


_BOWLS = {}


def bowl(device, flip_v):
    """The image surface: the vertex surface of the variable-focal lenslet array
    (variable_lenslets), tabulated in the image surface's local (x, y) = panel
    (u, v); the vertices stand up towards the light, which arrives along -z."""
    key = (str(device), flip_v)
    if key not in _BOWLS:
        gu, gv, h = vl.bowl_table(flip_v)
        t = lambda a: torch.tensor(a, dtype=torch.float64, device=device)  # noqa: E731
        _BOWLS[key] = (t(gu), t(gv), t(h))
    return _BOWLS[key]


def pack(prescriptions, device):
    """offaxis_tracer.pack for fold prescriptions: their image surface is the bowl."""
    if any(rx["image_surface"] != LENSLETS for rx in prescriptions):
        raise ValueError(f"fold prescriptions must declare image_surface {LENSLETS!r}")
    flips = {rx["lenslet_flip_v"] for rx in prescriptions}
    if len(flips) != 1:
        raise ValueError("a batch needs one lenslet orientation")
    return ot.pack(prescriptions, device)._replace(image_sag=bowl(device, flips.pop()))


def to_prescription(x_row, indices_row, lay):
    """One design (1-D tensors) -> offaxis_optiland JSON."""
    b = to_batch(x_row[None], indices_row[None], lay)
    out = []
    for s in range(b.z.shape[1]):
        c = float(b.c[0, s])
        out.append({"y_mm": float(b.y[0, s]), "z_mm": float(b.z[0, s]), "rx_deg": math.degrees(float(b.rx[0, s])),
                    "radius_mm": math.inf if c == 0.0 else 1.0 / c, "conic": float(b.k[0, s]),
                    "xy": b.xy[0, s].tolist(), "mirror": b.mirror[s], "index_after": float(b.n[0, s])})
    return {"surfaces": out, "image_surface": LENSLETS, "lenslet_flip_v": lay["flip_v"]}


def pupil_samples(spacing_mm=PUPIL_SPACING_MM):
    """Hexagonal grid inside the pupil disc, normalised to its radius."""
    r = spec.PUPIL_DIAMETER_MM / 2.0
    n = int(math.ceil(r / spacing_mm)) + 1
    pts = [(i * spacing_mm + (j % 2) * spacing_mm / 2, j * spacing_mm * math.sqrt(3) / 2)
           for i in range(-n, n + 1) for j in range(-n, n + 1)]
    pts = np.array([p for p in pts if math.hypot(*p) <= r + 1e-9])
    return pts / r


def field_grid():
    return np.array([(fx, fy) for fx in FIELDS_X_DEG for fy in FIELDS_Y_DEG])


def pupil_cells(fields_deg, pupil):
    """Cell of each pupil point, per field: squares of side pixel * F / f_lenslet
    (whole pupil in one cell where that exceeds it). Returns (F, P) ids and the
    count of cells."""
    tx, tz = np.radians(fields_deg[:, 0]), np.radians(fields_deg[:, 1])
    u, v = ft.field_to_panel_mm(tx, tz)
    focal_um = ft.lenslet_focal_um(u * 1e3, v * 1e3)                          # the lens this field lands on
    side = spec.PIXEL_UM * ft.local_focal_mm(tx, tz) / focal_um                # mm on the pupil
    uv = pupil * spec.PUPIL_DIAMETER_MM / 2.0 + spec.PUPIL_DIAMETER_MM / 2.0   # 0 .. D
    ids = np.zeros((len(fields_deg), len(pupil)), dtype=np.int64)
    for f, a in enumerate(side):
        n = max(1, int(math.ceil(spec.PUPIL_DIAMETER_MM / a)))
        cell = np.minimum((uv / a).astype(np.int64), n - 1)
        _, ids[f] = np.unique(cell[:, 0] * n + cell[:, 1], return_inverse=True)
    return ids, int(ids.max()) + 1


def context(device, weights=W):
    fields = field_grid()
    pupil = pupil_samples()
    ids, n_cells = pupil_cells(fields, pupil)
    tx, tz = np.radians(fields[:, 0]), np.radians(fields[:, 1])
    jac_t = ft.jacobian_mm_per_rad(tx, tz)
    panel_t = np.stack(ft.field_to_panel_mm(tx, tz), -1)
    t = lambda a, dt=torch.float64: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
    onehot = torch.nn.functional.one_hot(t(ids, torch.int64), n_cells).double()   # (F, P, K)
    fd = np.concatenate([fields, fields + [FD_DEG, 0.0], fields + [0.0, FD_DEG]])
    return {"weights": dict(weights), "fields": t(fields), "fd_fields": t(fd), "pupil": t(pupil), "onehot": onehot,
            "tol": t(ft.blur_tolerance_rad(np.radians(fields[:, 0]), np.radians(fields[:, 1]))),
            "jac_target": t(jac_t), "sv_target": t(np.linalg.svd(jac_t, compute_uv=False)),
            "panel_target": t(panel_t), "n_fields": len(fields)}


def jacobian(land_fd, n_fields):
    """Chief landings at theta, theta + dx, theta + dy -> (B, F, 2, 2) mm per rad."""
    h0, hx, hy = land_fd[:, :n_fields], land_fd[:, n_fields:2 * n_fields], land_fd[:, 2 * n_fields:]
    step = math.radians(FD_DEG)
    return torch.stack([(hx - h0) / step, (hy - h0) / step], -1)


def cell_offsets(land, alive, onehot, jac):
    """Each ray's landing offset from its pupil cell's centroid, mapped to field
    angle through the inverse local Jacobian (a fixed metric): (B, F, P, 2) rad,
    zero for lost rays. Also the ray count per (B, F, cell)."""
    w = alive.double()
    count = torch.einsum("bfp,fpk->bfk", w, onehot)
    centroid = torch.einsum("bfpc,fpk->bfkc", land * w[..., None], onehot) / count.clamp(min=1.0)[..., None]
    dev = land - torch.einsum("bfkc,fpk->bfpc", centroid, onehot)
    ang = torch.linalg.solve(jac.detach()[:, :, None], dev[..., None])[..., 0]
    return ang * w[..., None], count


def cell_blur2(land, alive, onehot, jac):
    """Per-pixel blur squared in rad^2, per (B, F, cell), and the ray count per cell."""
    ang, count = cell_offsets(land, alive, onehot, jac)
    sq = torch.einsum("bfp,fpk->bfk", (ang**2).sum(-1), onehot)
    return sq / count.clamp(min=1.0), count


def residuals(x, indices, lay, ctx):
    """The merit as least-squares residuals, (B, R): the merit is their sum of
    squares. Per-ray blur residuals, per-cell hinge, per-field ratio, panel and
    tilt; each constraint family is one residual, the norm of its violations
    (a SUM over rays: one ray through a wall must count)."""
    batch = to_batch(x, indices, lay)
    land, d_img, alive, diag = ot.trace(batch, ctx["fields"], ctx["pupil"], diagnostics=True)
    chief = torch.zeros(1, 2, dtype=x.dtype, device=x.device)
    land_fd, dir_fd, alive_fd = ot.trace(batch, ctx["fd_fields"], chief)
    F = ctx["n_fields"]
    B = x.shape[0]
    ok_c = alive_fd[:, :F, 0] & alive_fd[:, F:2 * F, 0] & alive_fd[:, 2 * F:, 0]
    okd = ok_c.double()
    jac = jacobian(land_fd[:, :, 0], F)
    eye = torch.eye(2, dtype=x.dtype, device=x.device)
    jac = torch.where(ok_c[..., None, None], jac, ctx["jac_target"][None])

    ang, count = cell_offsets(land, alive, ctx["onehot"], jac)
    used_cell = (count >= 2).double()                                  # (B, F, K)
    used_ray = torch.einsum("bfk,fpk->bfp", used_cell, ctx["onehot"])
    n_used = used_ray.sum((1, 2)).clamp(min=1.0)
    r_blur = (ang / ctx["tol"][None, :, None, None] * torch.sqrt(ctx["weights"]["blur"] * used_ray / n_used[:, None, None])[..., None])
    b2 = torch.einsum("bfp,fpk->bfk", (ang**2).sum(-1), ctx["onehot"]) / count.clamp(min=1.0)
    b2 = b2 / ctx["tol"][None, :, None] ** 2
    n_cells = used_cell.sum((1, 2)).clamp(min=1.0)
    r_hinge = torch.relu(torch.sqrt(b2 + 1e-18) - 0.6) * torch.sqrt(ctx["weights"]["hinge"] * used_cell / n_cells[:, None, None])

    # magnification: the design's singular values against the target map's (largest with largest)
    T = (jac**2).sum((-1, -2))
    D = torch.linalg.det(jac) ** 2
    disc = torch.sqrt(torch.clamp(T**2 - 4 * D, min=0.0))
    sv2 = torch.clamp(torch.stack([(T + disc) / 2, (T - disc) / 2], -1), min=1e-12)
    log_s = 0.5 * torch.log(sv2) - torch.log(ctx["sv_target"])[None]
    lo, hi = math.log(RATIO_BAND[0]), math.log(RATIO_BAND[1])
    r_ratio = torch.cat([torch.relu(lo - log_s), torch.relu(log_s - hi)], -1) * torch.sqrt(ctx["weights"]["ratio"] * okd / F)[..., None]

    h0 = land_fd[:, :F, 0]
    r_panel = torch.relu(h0.abs() - PANEL_HALF_MM) * torch.sqrt(ctx["weights"]["panel"] * okd / F)[..., None]
    flip = torch.tensor([lay["flip_u"], lay["flip_v"]], dtype=x.dtype, device=x.device)
    r_map = ((h0 - flip * ctx["panel_target"][None]) / MAP_SCALE_MM
             * torch.sqrt(ctx["weights"]["map"] * okd / F)[..., None])
    cos_t = dir_fd[:, :F, 0, 2].abs()
    r_tilt = (torch.relu(torch.rad2deg(torch.acos(torch.clamp(cos_t, max=1.0 - 1e-12))) - TILT_MAX_DEG)
              * torch.sqrt(ctx["weights"]["tilt"] * okd / F))

    space, barrier = FAMILIES[lay["family"]]["constraints"](batch, diag, alive, lay, x)
    norm = lambda v: torch.sqrt(v + 1e-30)  # noqa: E731
    r = torch.cat([r_blur.reshape(B, -1), r_hinge.reshape(B, -1), r_ratio.reshape(B, -1), r_panel.reshape(B, -1),
                   r_map.reshape(B, -1),
                   r_tilt, norm(ctx["weights"]["space"] * space)[:, None], norm(ctx["weights"]["barrier"] * barrier)[:, None]], 1)
    alive_all = (alive.double().sum((1, 2)) + okd.sum(1)) / (alive[0].numel() + ok_c.shape[1])
    sq = lambda t: (t**2).reshape(B, -1).sum(1)  # noqa: E731
    return r, {"blur": sq(r_blur), "hinge": sq(r_hinge), "ratio": sq(r_ratio), "panel": sq(r_panel),
               "map": sq(r_map),
               "tilt": sq(r_tilt), "space": ctx["weights"]["space"] * space, "barrier": ctx["weights"]["barrier"] * barrier,
               "alive_all": alive_all}


def fold_constraints(batch, diag, alive, lay, x):
    """Space and barrier sums (squared, over rays) of the fold family: no hardware
    in the eye's view cone, near the pupil or behind the face plane; return rays
    cross the baffle beyond the hardware; glass and air floors; TIR and sag-domain
    margins."""
    pts = diag["points"]                                               # (B, S, F, P, 3)
    live = alive[:, None].double()
    hw = pts[:, 1:]
    corners = panel_corners(batch)                                     # (B, 4, 3)
    cone = lambda p: p[..., 2] * math.tan(math.radians(TOP_FIELD_DEG)) + spec.PUPIL_DIAMETER_MM / 2 \
        + CONE_MARGIN_MM  # noqa: E731

    def violation(p):
        return (torch.relu(cone(p) - p[..., 1]) ** 2 + torch.relu(CLEARANCE_MM - torch.linalg.norm(p, dim=-1)) ** 2
                + torch.relu(MIN_Z_MM - p[..., 2]) ** 2)

    space = (violation(hw) * live).sum((1, 2, 3)) + violation(corners).sum(1)
    # Baffle (baffle_geometry): a black plate on the plane just above the view
    # cone, from the face plane to the far edge of the hardware, and a wall in the
    # face plane. The return rays (mirror -> first corrector) must cross the plane
    # beyond that far edge, or the plate would block them.
    z_far = torch.maximum((torch.where(alive[:, None], hw[..., 2], torch.zeros_like(hw[..., 2]))).amax((1, 2, 3)),
                          corners[..., 2].amax(1))
    z_cross = baffle_crossing_z(pts[:, 0], pts[:, 1])
    space = space + (torch.relu(z_far[:, None, None] + BAFFLE_CLEAR_MM - z_cross) ** 2 * alive).sum((1, 2))
    # barriers: a ray is penalised before it dies (no gradient reaches a lost ray)
    seg = torch.linalg.norm(pts[:, 1:] - pts[:, :-1], dim=-1)          # (B, S-1, F, P) mirror -> ... -> image
    floor = torch.tensor([MIN_AIR_MM] + [MIN_GLASS_MM, MIN_AIR_MM] * lay["n_el"], dtype=x.dtype, device=x.device)
    barrier = (torch.relu(floor[None, :, None, None] - seg) ** 2 * live).sum((1, 2, 3))
    barrier = barrier + (torch.relu(diag["sin2t"][:, :-1] - SIN2_MAX) ** 2 * live).sum((1, 2, 3))
    barrier = barrier + (torch.relu(DOMAIN_MIN - diag["domain"]) ** 2 * alive[:, None].double()).sum((1, 2, 3))
    return space, barrier


def merit(x, indices, lay, ctx):
    r, info = residuals(x, indices, lay, ctx)
    return (r**2).sum(1), info


def baffle_plane(p):
    """Signed height above the baffle plane y = z tan(top field) + pupil radius + offset."""
    return p[..., 1] - (p[..., 2] * math.tan(math.radians(TOP_FIELD_DEG)) + spec.PUPIL_DIAMETER_MM / 2
                        + BAFFLE_OFFSET_MM)


def baffle_crossing_z(a, b):
    """z where the segment a -> b (a below the plane, b above) crosses the baffle plane."""
    fa, fb = baffle_plane(a), baffle_plane(b)
    t = torch.clamp(fa / (fa - fb + 1e-30), 0.0, 1.0)
    return a[..., 2] + t * (b[..., 2] - a[..., 2])


def panel_corners(batch):
    """Global corners of the physical panel on the image surface (B, 4, 3)."""
    h = spec.PANEL_MM / 2.0
    local = torch.tensor([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]], dtype=batch.z.dtype,
                         device=batch.z.device)
    a = batch.rx[:, -1, None]
    ca, sa = torch.cos(a), torch.sin(a)
    y = local[None, :, 1] * ca - local[None, :, 2] * sa + batch.y[:, -1, None]
    z = local[None, :, 1] * sa + local[None, :, 2] * ca + batch.z[:, -1, None]
    return torch.stack([local[None, :, 0].expand_as(y), y, z], -1)


def per_field_blur(x, indices, lay, ctx):
    """Worst-cell per-pixel blur per field, in tolerance units (no gradients)."""
    with torch.no_grad():
        batch = to_batch(x, indices, lay)
        land, _, alive = ot.trace(batch, ctx["fields"], ctx["pupil"])
        chief = torch.zeros(1, 2, dtype=x.dtype, device=x.device)
        land_fd, _, _ = ot.trace(batch, ctx["fd_fields"], chief)
        b2, count = cell_blur2(land, alive, ctx["onehot"], jacobian(land_fd[:, :, 0], ctx["n_fields"]))
        b = torch.sqrt(b2) / ctx["tol"][None, :, None]
        return torch.where(count >= 2, b, torch.zeros_like(b)).amax(-1)


def orientation(x, indices, lay):
    """Per design, the sign of d(image x)/d(theta_x) and d(image y)/d(theta_z) at
    the fovea: +1 when the image is upright relative to the target map."""
    f = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], dtype=x.dtype, device=x.device)
    with torch.no_grad():
        land, _, _ = ot.trace(to_batch(x, indices, lay), f, torch.zeros(1, 2, dtype=x.dtype, device=x.device))
    return torch.sign(land[:, 1, 0, 0] - land[:, 0, 0, 0]), torch.sign(land[:, 3, 0, 1] - land[:, 2, 0, 1])


def family_orientation(n_el, material, rng, device, lo, hi, count=512, family="fold"):
    """The image orientation most random designs of this layout produce."""
    lay = layout(n_el, family=family)
    x = torch.clamp(random_designs(lay, count, rng, device), lo, hi)
    idx = torch.tensor(rng.choice(INDEX_SETS[material], size=(count, n_el)), dtype=torch.float64, device=device)
    su, sv = orientation(x, idx, lay)
    return int(torch.sign(su.sum())), int(torch.sign(sv.sum()))


def live_seeds(lay, count, material, rng, device, ctx, lo, hi, chunk=512):
    """Random designs in which every traced ray (and every chief ray of the
    Jacobian) reaches the panel. The search then never lets a ray die."""
    xs, idxs, tried = [], [], 0
    while sum(len(x) for x in xs) < count:
        x = torch.clamp(random_designs(lay, chunk, rng, device), lo, hi)
        idx = torch.tensor(rng.choice(INDEX_SETS[material], size=(chunk, lay["n_el"])), dtype=torch.float64,
                           device=device)
        with torch.no_grad():
            _, info = merit(x, idx, lay, ctx)
        su, sv = orientation(x, idx, lay)
        keep = (info["alive_all"] == 1.0) & (su == lay["flip_u"]) & (sv == lay["flip_v"])
        xs.append(x[keep])
        idxs.append(idx[keep])
        tried += chunk
        if tried > 200 * count:
            raise RuntimeError(f"only {sum(len(x) for x in xs)} live seeds in {tried} random designs")
    return torch.cat(xs)[:count], torch.cat(idxs)[:count], tried


def fd_jacobian(x, indices, lay, ctx, r0, free, lo, hi, h=1e-5):
    """Forward-difference Jacobian of the residuals, (B, R, D); zero columns for
    frozen parameters. The step points inwards at a bound."""
    J = torch.zeros(x.shape[0], r0.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
    for d in torch.nonzero(free).flatten().tolist():
        step = torch.where(x[:, d] + h > hi[d], -h, h)
        xp = x.clone()
        xp[:, d] += step
        rp, _ = residuals(xp, indices, lay, ctx)
        J[:, :, d] = (rp - r0) / step[:, None]
    return J


def run(out_dir, n_el, material, count, iters, seed, device, ratio_weight=W["ratio"], tag_suffix="", family="fold"):
    """Batched Levenberg-Marquardt (damped least squares), one damping per
    design. Each iteration tries four dampings and keeps the best step that
    lowers the merit and loses no ray. High-order terms are frozen for the
    first third of the iterations."""
    rng = np.random.default_rng(seed)
    lo, hi = bounds(layout(n_el, family=family), device)
    flip_u, flip_v = family_orientation(n_el, material, rng, device, lo, hi, family=family)
    lay = layout(n_el, flip_u, flip_v, family=family)
    print(f"image orientation of this layout: flip_u {flip_u}, flip_v {flip_v}", flush=True)
    ctx = context(device, {**W, "ratio": ratio_weight})
    with torch.no_grad():
        x, indices, tried = live_seeds(lay, count, material, rng, device, ctx, lo, hi)
        print(f"{count} live seeds from {tried} random designs", flush=True)
        high = torch.zeros(lay["size"], dtype=torch.bool, device=device)
        for s in lay["shapes"]:
            high[s.start + 2 + LOW_ORDER:s.stop] = True
        lam = torch.full((count,), 1e-2, dtype=x.dtype, device=device)
        tries = torch.tensor([0.25, 1.0, 4.0, 16.0], dtype=x.dtype, device=device)
        r, info = residuals(x, indices, lay, ctx)
        loss = (r**2).sum(1)
        t0 = time.time()
        for it in range(iters):
            free = ~high if it < iters // 3 else torch.ones_like(high)
            J = fd_jacobian(x, indices, lay, ctx, r, free, lo, hi)
            A = J.transpose(1, 2) @ J
            g = (J.transpose(1, 2) @ r[..., None])[..., 0]
            frozen = ~free
            A[:, frozen, :] = 0.0
            A[:, :, frozen] = 0.0
            A[:, frozen, frozen] = 1.0
            diag = torch.diagonal(A, dim1=1, dim2=2)
            floor = 1e-9 * diag.amax(1, keepdim=True)
            best_loss, best_x, best_lam = loss.clone(), x.clone(), lam * 16.0 * 4.0
            for f in tries:
                damp = (lam * f)[:, None] * torch.maximum(diag, floor)
                delta = -torch.linalg.solve(A + torch.diag_embed(damp), g[..., None])[..., 0]
                xc = torch.clamp(x + delta, lo, hi)
                lc, ic = merit(xc, indices, lay, ctx)
                ok = (ic["alive_all"] == 1.0) & torch.isfinite(lc) & (lc < best_loss)
                best_loss = torch.where(ok, lc, best_loss)
                best_x[ok] = xc[ok]
                best_lam = torch.where(ok, lam * f / 3.0, best_lam)
            moved = best_loss < loss
            x = best_x
            lam = torch.clamp(best_lam, 1e-9, 1e9)
            r, info = residuals(x, indices, lay, ctx)
            loss = (r**2).sum(1)
            if it % 5 == 0 or it == iters - 1:
                b = int(torch.argmin(loss))
                print(f"[{n_el} el {material}] it {it:4d} best {float(loss[b]):9.4f} median {float(loss.median()):10.3f} "
                      f"moved {int(moved.sum()):4d}  " + " ".join(f"{k} {float(v[b]):.3g}" for k, v in info.items())
                      + f"  {time.time() - t0:5.0f} s", flush=True)
    with torch.no_grad():
        loss, info = merit(x, indices, lay, ctx)
    worst = per_field_blur(x.detach(), indices, lay, ctx)
    out = []
    for rank, b in enumerate(torch.argsort(loss)[:20].tolist()):
        out.append({"rank": rank, "loss": float(loss[b]), **{k: float(v[b]) for k, v in info.items()},
                    "worst_cell_blur_per_field": worst[b].tolist(), "fields_deg": ctx["fields"].tolist(),
                    "indices": indices[b].tolist(), "x": x[b].detach().tolist(), "n_el": n_el,
                    "flip_u": lay["flip_u"], "flip_v": lay["flip_v"],
                    "material": material, "prescription": to_prescription(x[b].detach(), indices[b], lay)})
    tag = f"{family}_el{n_el}_{material}{tag_suffix}"
    (Path(out_dir) / f"best_{tag}.json").write_text(json.dumps(out, indent=1))
    top = out[0]
    print(f"TOP {tag}: loss {top['loss']:.3f} blur {top['blur']:.3f} alive {top['alive_all']:.3f} "
          f"worst-cell blur p90 {np.percentile(top['worst_cell_blur_per_field'], 90):.2f}", flush=True)


register("fold", layout=fold_layout, bounds=fold_bounds, random_designs=fold_random_designs,
         to_batch=fold_to_batch, constraints=fold_constraints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--elements", type=int, required=True)
    ap.add_argument("--material", choices=sorted(INDEX_SETS), required=True)
    ap.add_argument("--designs", type=int, default=256)
    ap.add_argument("--iters", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio-weight", type=float, default=W["ratio"],
                    help="weight of the magnification target; 0 leaves the mapping free (diagnostic)")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(args.out_dir, args.elements, args.material, args.designs, args.iters, args.seed, torch.device("cuda"),
        ratio_weight=args.ratio_weight, tag_suffix=args.tag)


if __name__ == "__main__":
    main()

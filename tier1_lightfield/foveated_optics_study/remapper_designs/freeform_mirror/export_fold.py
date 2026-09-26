"""Export one folded-remapper design to the shared evaluator contract (lf_evaluate.py).

    python export_fold.py <best_fold_*.json> <out_dir> [--rank 0]

The mirror becomes a freeform sheet, each corrector a closed freeform solid
(the back surface is sampled along the front surface's local axis), both with
exact loop normals and sized to the traced footprint plus MARGIN_MM. The
lenslet array is the variable-focal one: its vertex bowl is the design's image
surface, the centre lens's vertex at the image surface's origin.
World frame (lf_blender.py): +Y forward, +Z up, pupil at y = PUPIL_Y_MM; the
tracer frame (z forward, y up) maps by the proper rotation (x, y, z) ->
(-x, PUPIL_Y_MM + z, y).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

import fold_search as fs  # noqa: E402
import lf_pipeline as lp  # noqa: E402
import variable_lenslets as vl  # noqa: E402
import offaxis_tracer as ot  # noqa: E402
import screen_spec as spec  # noqa: E402

MARGIN_MM = 1.0
GRID = 161
TRACER_TO_WORLD = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])   # rows: tracer x, y, z


def to_world(p):
    return np.asarray(p) @ TRACER_TO_WORLD + np.array([0.0, lp.PUPIL_Y_MM, 0.0])


def _frame(batch, s):
    """Origin and local axes (rows) of surface s in the tracer frame."""
    a = float(batch.rx[0, s])
    o = np.array([0.0, float(batch.y[0, s]), float(batch.z[0, s])])
    axes = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), np.sin(a)], [0.0, -np.sin(a), np.cos(a)]])
    return o, axes


def _sag(batch, s, x, y):
    t = lambda v: torch.as_tensor(v, dtype=torch.float64)  # noqa: E731
    f, sx, sy, ok = ot.sag(t(x), t(y), batch.c[0, s].cpu(), batch.k[0, s].cpu(), batch.xy[0, s].cpu(),
                         batch.terms)
    if not bool(ok.all()):
        raise ValueError(f"surface {s}: the exported region leaves the sag domain")
    if len(batch.spline) == 3 and s in batch.spline[0]:
        shape = f.shape
        e, ex, ey = ot.bspline_sag(t(x).reshape(1, -1), t(y).reshape(1, -1), batch.spline[1], batch.spline[2][:1].cpu())
        f, sx, sy = f + e.reshape(shape), sx + ex.reshape(shape), sy + ey.reshape(shape)
    return f.numpy(), sx.numpy(), sy.numpy()


def cpu_batch(batch):
    """A detached CPU copy of a one-design batch, its spline included."""
    spline = (batch.spline[0], batch.spline[1], batch.spline[2].detach().cpu()) if batch.spline else ()
    return ot.Batch(*(t.detach().cpu() for t in batch[:7]), mirror=batch.mirror, terms=batch.terms, spline=spline)


def _local(batch, s, p):
    o, axes = _frame(batch, s)
    return (p - o) @ axes.T


def _global(batch, s, p):
    o, axes = _frame(batch, s)
    return p @ axes + o


def _grid(batch, s, hits, disc=False):
    """Rectangle over the footprint (+ margin) in surface s's local x, y. The
    search traces only the theta_x >= 0 half field (plane symmetry), so the
    rectangle is made symmetric in x. With disc, the outline is the circle
    about the rectangle's centre that holds the footprint: the square grid is
    mapped onto it (elliptical grid mapping), so the topology stays."""
    loc = _local(batch, s, hits)
    lo, hi = loc[:, :2].min(0) - MARGIN_MM, loc[:, :2].max(0) + MARGIN_MM
    half_x = max(-lo[0], hi[0])
    lo[0], hi[0] = -half_x, half_x
    if not disc:
        return np.meshgrid(np.linspace(lo[0], hi[0], GRID), np.linspace(lo[1], hi[1], GRID), indexing="ij")
    cy = 0.5 * (lo[1] + hi[1])
    radius = np.hypot(loc[:, 0], loc[:, 1] - cy).max() + MARGIN_MM
    a, b = np.meshgrid(np.linspace(-1.0, 1.0, GRID), np.linspace(-1.0, 1.0, GRID), indexing="ij")
    return radius * a * np.sqrt(1.0 - b**2 / 2.0), cy + radius * b * np.sqrt(1.0 - a**2 / 2.0)


def _surface(batch, s, gx, gy, sag_limits=(-np.inf, np.inf)):
    """Mesh points, normals and the clamped-vertex mask of surface s over the
    grid. sag_limits (local sag, mm) clamps it flat outside its optical zone,
    where its polynomial would otherwise run away."""
    f, sx, sy = _sag(batch, s, gx, gy)
    clamped = (f < sag_limits[0]) | (f > sag_limits[1])
    f = np.clip(f, *sag_limits)
    sx, sy = np.where(clamped, 0.0, sx), np.where(clamped, 0.0, sy)
    loc = np.stack([gx, gy, f], -1)
    n = np.stack([-sx, -sy, np.ones_like(sx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    _, axes = _frame(batch, s)
    return _global(batch, s, loc.reshape(-1, 3)), (n.reshape(-1, 3) @ axes), clamped.reshape(-1)


def _quads(n0, n1, base=0):
    idx = np.arange(n0 * n1).reshape(n0, n1) + base
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    return np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])


def _orient(verts, faces, vertex_normals):
    """Loop normals that agree with the face winding (flip the analytic ones if needed)."""
    geo = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    n = vertex_normals[faces]                                            # (M, 3, 3)
    sign = np.sign(np.einsum("mc,mkc->m", geo, n).sum())
    return (sign * n).reshape(-1, 3)


def _back_along_front_axis(batch, s_front, s_back, gx, gy):
    """Points of the back surface on the lines through the front grid along the
    front's local z axis, as local (x, y) of the back surface."""
    f, _, _ = _sag(batch, s_front, gx, gy)
    start = _global(batch, s_front, np.stack([gx, gy, f], -1).reshape(-1, 3))
    _, axes_f = _frame(batch, s_front)
    o = torch.as_tensor(_local(batch, s_back, start))
    _, axes_b = _frame(batch, s_back)
    d = torch.as_tensor(np.broadcast_to(axes_f[2] @ axes_b.T, o.shape).copy())
    c, k, C = batch.c[0, s_back].cpu(), batch.k[0, s_back].cpu(), batch.xy[0, s_back].cpu()
    t = -o[:, 2] / d[:, 2]
    for _ in range(ot.NEWTON_STEPS):
        t = ot._newton(o, d, c, k, C, batch.terms, t)
    p = o + t[:, None] * d
    f_b, _, _ = _sag(batch, s_back, p[:, 0].numpy(), p[:, 1].numpy())
    if np.max(np.abs(p[:, 2].numpy() - f_b)) > 1e-9:
        raise ValueError(f"surface {s_back}: back surface not reached along the front axis")
    return p[:, 0].numpy().reshape(gx.shape), p[:, 1].numpy().reshape(gx.shape)


def _sag_range(batch, s, hits):
    """The local sag range of a surface's traced hits, plus MARGIN_MM each way:
    beyond it the exported surface is flat."""
    z = _local(batch, s, hits)[:, 2]
    return float(z.min()) - MARGIN_MM, float(z.max()) + MARGIN_MM


def beyond_footprint(batch, s, points, hits):
    """Mask of the points whose local x, y on surface s lie outside the convex
    hull of the hits and of their mirror images (x -> -x: the search traces
    one half of the field)."""
    both = np.concatenate([hits, hits * np.array([-1.0, 1.0, 1.0])])
    return Delaunay(_local(batch, s, both)[:, :2]).find_simplex(_local(batch, s, points)[:, :2]) < 0


def _solid(batch, s_front, s_back, hits_front, hits_back, disc=False, front_floor=-np.inf):
    # the outline must hold both footprints (rays cross the glass obliquely),
    # so project the back hits onto the front's local x, y along its axis; each
    # surface is flat beyond its own hits' sag range (its polynomial would run
    # away over the rim and can cross the other surface); front_floor, a local
    # sag, keeps the front off something just before it (a pancake's polariser)
    gx, gy = _grid(batch, s_front, np.concatenate([hits_front, hits_back]), disc)
    lo, hi = _sag_range(batch, s_front, hits_front)
    vf, nf, cf = _surface(batch, s_front, gx, gy, (max(lo, front_floor), hi))
    bx, by = _back_along_front_axis(batch, s_front, s_back, gx, gy)
    vb, nb, cb = _surface(batch, s_back, bx, by, _sag_range(batch, s_back, hits_back))
    _, axes_f = _frame(batch, s_front)
    thickness = (vb - vf) @ axes_f[2]                                  # one sign: the light may run either way
    # beyond the footprint nothing constrains the glass: where it thins below
    # MIN_GLASS_MM the back is set MIN_GLASS_MM behind the front (a flat facet)
    sign = np.sign(np.median(thickness))
    thin = beyond_footprint(batch, s_front, vf, np.concatenate([hits_front, hits_back])) \
        & (sign * thickness < fs.MIN_GLASS_MM)
    vb[thin] = vf[thin] + sign * fs.MIN_GLASS_MM * axes_f[2]
    cb = cb | thin
    thickness = (vb - vf) @ axes_f[2]
    if not (thickness.min() > 0.0 or thickness.max() < 0.0):
        raise ValueError(f"surfaces {s_front}, {s_back}: the back crosses the front")
    N = GRID * GRID
    ff, fb = _quads(GRID, GRID), _quads(GRID, GRID, N)[:, ::-1]
    ring = np.concatenate([np.arange(GRID) * GRID, (GRID - 1) * GRID + np.arange(GRID),
                           (np.arange(GRID) * GRID + GRID - 1)[::-1], (np.arange(GRID))[::-1]])
    ring = np.array([r for i, r in enumerate(ring) if i == 0 or r != ring[i - 1]])
    ring = ring[:-1] if ring[-1] == ring[0] else ring
    a, b = ring, np.roll(ring, -1)
    # the front grid's boundary edges run a -> b along the ring, so the walls use b -> a
    side = np.concatenate([np.stack([b, a, a + N], -1), np.stack([b, a + N, b + N], -1)])
    verts = np.concatenate([vf, vb])
    faces = np.concatenate([ff, fb, side])
    vol = np.einsum("ij,ij->i", verts[faces[:, 0]], np.cross(verts[faces[:, 1]], verts[faces[:, 2]])).sum()
    if vol < 0:
        faces = faces[:, ::-1]
    n_opt = len(ff) + len(fb)
    loops_f = _orient(verts, faces[:len(ff)], np.concatenate([nf, nb]))
    loops_b = _orient(verts, faces[len(ff):n_opt], np.concatenate([nf, nb]))
    geo = np.cross(verts[faces[n_opt:, 1]] - verts[faces[n_opt:, 0]], verts[faces[n_opt:, 2]] - verts[faces[n_opt:, 0]])
    geo /= np.linalg.norm(geo, axis=1, keepdims=True)
    # a face touching a clamped vertex is a flat facet across the kink: its own normal
    opt = faces[:n_opt]
    geo_opt = np.cross(verts[opt[:, 1]] - verts[opt[:, 0]], verts[opt[:, 2]] - verts[opt[:, 0]])
    geo_opt /= np.linalg.norm(geo_opt, axis=1, keepdims=True)
    loops_opt = np.concatenate([loops_f, loops_b]).reshape(n_opt, 3, 3)
    kink = np.concatenate([cf, cb])[opt].any(1)
    loops_opt[kink] = geo_opt[kink][:, None, :]
    loops = np.concatenate([loops_opt.reshape(-1, 3), np.repeat(geo, 3, axis=0)])
    return verts, faces, loops


def baffle_geometry(hardware):
    """L-shaped black shield in the tracer frame (fold_search's baffle): a plate
    on the baffle plane from the face plane to just beyond the hardware's far
    edge, and a wall in the face plane up to above the hardware. Every line of
    sight from the pupil to hardware above the plane crosses one of them."""
    tan = np.tan(np.radians(fs.TOP_FIELD_DEG))
    height = lambda z: z * tan + fs.spec.PUPIL_DIAMETER_MM / 2 + fs.BAFFLE_OFFSET_MM  # noqa: E731
    z0, z1 = fs.MIN_Z_MM, hardware[:, 2].max() + fs.BAFFLE_CLEAR_MM / 2
    y_top = hardware[:, 1].max() + 1.0
    w = np.abs(hardware[:, 0]).max() + 3.0
    verts = np.array([[-w, height(z0), z0], [w, height(z0), z0], [w, height(z1), z1], [-w, height(z1), z1],
                      [-w, y_top, z0], [w, y_top, z0]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 1]])
    return verts, faces


def export(best_json, out_dir, rank=0, device="cuda"):
    entry = json.loads(Path(best_json).read_text())[rank]
    return export_entry(entry, out_dir, device, source={"design": Path(best_json).name, "rank": rank})


def export_entry(entry, out_dir, device="cuda", source={}):
    dev = torch.device(device)
    lay = fs.entry_layout(entry)
    x = torch.tensor([entry["x"]], dtype=torch.float64, device=dev)
    idx = torch.tensor([entry["indices"]], dtype=torch.float64, device=dev)
    batch = fs.to_batch(x, idx, lay)
    ctx = fs.context(dev)
    with torch.no_grad():
        _, d_img, alive, diag = ot.trace(batch, ctx["fields"], ctx["pupil"], diagnostics=True)
    if not bool(alive.all()):
        raise ValueError("the design loses rays; it cannot be exported")
    pts = diag["points"][0].cpu().numpy().reshape(batch.z.shape[1], -1, 3)
    batch = cpu_batch(batch)
    data = {}
    gx, gy = _grid(batch, 0, pts[0])
    vm, nm, _ = _surface(batch, 0, gx, gy)
    fm = _quads(GRID, GRID)
    surfaces = [("mirror", vm, fm, _orient(vm, fm, nm), None)]
    for e in range(entry["n_el"]):
        v, f, n = _solid(batch, 1 + 2 * e, 2 + 2 * e, pts[1 + 2 * e], pts[2 + 2 * e])
        surfaces.append(("glass", v, f, n, entry["indices"][e]))
    for k, (kind, v, f, n, index) in enumerate(surfaces):
        data.update({f"surf{k}_kind": kind, f"surf{k}_verts": to_world(v), f"surf{k}_faces": f,
                     f"surf{k}_normals": n @ TRACER_TO_WORLD})
        if index is not None:
            data[f"surf{k}_index"] = index
    hardware = np.concatenate([np.concatenate([s[1] for s in surfaces[1:]]),
                               fs.panel_corners(batch)[0].numpy()])
    bv, bf = baffle_geometry(hardware)
    k = len(surfaces)
    data.update({f"surf{k}_kind": "absorber", f"surf{k}_verts": to_world(bv), f"surf{k}_faces": bf})
    data["n_surfaces"] = k + 1
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "remapper.npz", **data)

    s_img = batch.z.shape[1] - 1
    o, axes = _frame(batch, s_img)
    if not bool((d_img[0, ..., 2] < 0).all()):
        raise ValueError("light must reach the image surface along its local -z (the bowl faces +z)")
    toward_light = axes[2]
    u = axes[0]
    v = np.cross(toward_light, u)
    basis = np.stack([u, v, toward_light]) @ TRACER_TO_WORLD
    # the centre lens's vertex sits on the image surface's origin; the bowl follows
    centre_height_um = float(vl.vertex_height_um(0.0, 0.0, lay["flip_v"]))
    origin = to_world(o) - centre_height_um * 1e-3 * basis[2]
    design = {"lenslets": fs.LENSLETS, "lenslet_flip_v": lay["flip_v"], "remapper_npz": "remapper.npz",
              "panel_pose": {"origin_mm": origin.tolist(), "basis": basis.tolist()},
              "field_deg": entry["field_deg"], "source": {**source, "material": entry["material"]}}
    (out / "design.json").write_text(json.dumps(design, indent=1))
    np.savez(out / "rays.npz", **fans(entry, dev))
    return out


_HX, _HZ = spec.FIELD_HALF_DEG
_CORNER = 1.0 if spec.FIELD_SHAPE == "rect" else math.sqrt(0.5)     # the field's corner, or the ellipse's diagonal
FAN_FIELDS = ((0.0, 0.0), (0.0, _HZ), (0.0, -_HZ), (_HX, 0.0), (_CORNER * _HX, _CORNER * _HZ),
              (_CORNER * _HX, -_CORNER * _HZ), (17.0 * _HX / 35.0, 0.0))
FAN_PUPIL = tuple((0.0, py) for py in (-1.0, -0.5, 0.0, 0.5, 1.0))


def fans(entry, dev, family="fold"):
    """World-frame polylines (pupil -> every surface) for a few fields, for viewing."""
    lay = fs.entry_layout(entry, family)
    x = torch.tensor([entry["x"]], dtype=torch.float64, device=dev)
    idx = torch.tensor([entry["indices"]], dtype=torch.float64, device=dev)
    t = lambda a: torch.tensor(a, dtype=torch.float64, device=dev)  # noqa: E731
    with torch.no_grad():
        _, _, alive, diag = ot.trace(fs.to_batch(x, idx, lay), t(FAN_FIELDS), t(FAN_PUPIL), diagnostics=True)
    pts = diag["points"][0].cpu().numpy()                               # (S, F, P, 3)
    start = np.array([[0.0, py * 2.0, 0.0] for _, py in FAN_PUPIL])
    paths = np.concatenate([np.broadcast_to(start, (len(FAN_FIELDS),) + start.shape)[None], pts], 0)
    return {"paths": to_world(np.moveaxis(paths, 0, 2)), "alive": alive[0].cpu().numpy(),
            "fields": np.array(FAN_FIELDS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("best_json")
    ap.add_argument("out_dir")
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()
    print(export(args.best_json, args.out_dir, args.rank))


if __name__ == "__main__":
    main()

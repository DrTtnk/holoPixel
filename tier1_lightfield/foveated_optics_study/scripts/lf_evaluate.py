"""Shared, design-independent acceptance evaluator for a light-field screen.

    python lf_evaluate.py <design_dir> [--work <dir>] [--pixels 2044]

<design_dir>/design.json:
    {"focal_um": lenslet focal length,
     "panel_pose": {"origin_mm": [x, y, z], "basis": [u_world, v_world, w_world]},
     "remapper_npz": "remapper.npz"}          (relative to design_dir)

remapper.npz: n_surfaces, and per surface k: surf{k}_verts (N, 3) world mm,
surf{k}_faces (M, 3), surf{k}_normals (3M, 3) optional loop normals,
surf{k}_kind in {glass, mirror, absorber}, surf{k}_index (glass only).

Every number comes from Cycles: each pupil camera ray is traced through the
real geometry to a panel pixel, and the pixel's owning lens is read back.
Per lens, the rays that reach its pixels give its field direction (mean), its
angular blur (RMS spread over the whole pupil), its pupil fill (fraction of
pupil points that see it) and its ghosts (rays more than 3 pitches off).
Pitches are the R = 6 target pitch at the lens's own field direction, so a
perfectly focused lens has blur 0.373 pitch: the RMS radius of its own
hexagonal aperture (flat-to-flat = one pitch).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import foveation_target as ft
import lf_pipeline as lp
import mla_design as mla
import mla_mesh

SIDE_UM, PIXEL_UM, INDEX, MIN_THICKNESS_UM = 17.37, 4.0, 1.5, 10.0
FOV_X_DEG, FOV_Z_DEG = 35.0, 22.5
HEX_RMS_PITCH = math.sqrt(5.0 / 12.0) / math.sqrt(3.0)

ACCEPT = {
    "coverage_min": 0.98,
    "ratio_median_range": (0.8, 1.25),
    "ratio_p5_p95_range": (0.67, 1.5),
    "blur_p90_max_pitch": 0.75,
    "fill_p10_min": 0.8,
    "ghost_mean_max": 0.05,
    "throughput_min": 0.9,
    "eye_relief_min_mm": 20.0,
    "clearance_min_mm": 15.0,
}


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _field_deg(d):
    return np.degrees(np.arctan2(d[..., 0], d[..., 1])), np.degrees(np.arctan2(d[..., 2], d[..., 1]))


def _in_fov(d):
    tx, tz = _field_deg(d)
    return (np.abs(tx) <= FOV_X_DEG) & (np.abs(tz) <= FOV_Z_DEG) & (d[..., 1] > 0)


def _ecc(d):
    return np.arccos(np.clip(d[..., 1], -1.0, 1.0))


def _axis_hit_mm(verts, faces, origin, direction):
    """Nearest positive ray/triangle hit (Moller-Trumbore), inf when none."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e1, e2 = b - a, c - a
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - a
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = np.einsum("j,ij->i", direction, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
    return float(t[hit].min()) if hit.any() else math.inf


def validate_surfaces(remapper):
    """Glass must be a closed solid wound outwards, or Cycles refracts with the
    index inverted. Fail before rendering, naming the surface."""
    r = np.load(remapper)
    for k in range(int(r["n_surfaces"])):
        kind, v, f = str(r[f"surf{k}_kind"]), r[f"surf{k}_verts"], r[f"surf{k}_faces"]
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"surface {k}: faces must be (M, 3) triangles")
        if f"surf{k}_normals" in r and r[f"surf{k}_normals"].shape != (3 * len(f), 3):
            raise ValueError(f"surface {k}: normals must be (3M, 3) loop normals")
        if kind != "glass":
            continue
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        if not np.all(counts == 2):
            raise ValueError(f"surface {k}: glass is not closed ({int(np.sum(counts != 2))} open edges)")
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        if np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) <= 0:
            raise ValueError(f"surface {k}: glass is wound inwards (negative signed volume)")


def evaluate(design_dir, work, panel_pixels=2044, view_spacing_mm=0.5, resolution=2048, fov_deg=76.0):
    design_dir, work = Path(design_dir), Path(work)
    design = json.loads((design_dir / "design.json").read_text())
    remapper = (design_dir / design["remapper_npz"]).resolve()
    validate_surfaces(remapper)
    focal = float(design["focal_um"])
    radius = mla.radius_for_focal_length_um(focal, INDEX)
    panel_um = panel_pixels * PIXEL_UM
    mesh = mla_mesh.build(panel_um=panel_um, side_um=SIDE_UM, radius_um=radius,
                          min_thickness_um=MIN_THICKNESS_UM, subdivisions=3)
    gap = mla.back_focal_gap_um(radius, INDEX, mesh.centre_thickness)
    work.mkdir(parents=True, exist_ok=True)
    np.savez(work / "mla_mesh.npz", verts=mesh.verts, faces=mesh.faces, loop_normals=mesh.loop_normals)
    k = np.arange(panel_pixels)
    pu, pv = np.meshgrid((k + 0.5) * PIXEL_UM - panel_um / 2, (k + 0.5) * PIXEL_UM - panel_um / 2)
    _, owner = cKDTree(mesh.centres).query(np.column_stack([pu.ravel(), pv.ravel()]))
    np.save(work / "pixel_owner.npy", owner.reshape(panel_pixels, panel_pixels).astype(np.int32))
    views = lp.hex_views_mm(view_spacing_mm, 2.0)
    cfg = {
        "mode": "evaluate", "mla_npz": str(work / "mla_mesh.npz"), "index": INDEX,
        "panel_pose": design["panel_pose"], "gap_um": gap, "panel_pixels": panel_pixels,
        "pixel_um": PIXEL_UM, "camera": {"resolution": resolution, "fov_deg": fov_deg},
        "views_mm": views.tolist(), "tmp_dir": str(work / "views"), "remapper_npz": str(remapper),
        "pixel_owner_npy": str(work / "pixel_owner.npy"), "out_npz": str(work / "evaluate.npz"),
    }
    lp.run_blender(cfg, work)
    report = metrics(work / "views", views, mesh.centres)
    report["geometry"] = geometry(design, remapper, mesh, gap)
    report["design"] = {"focal_um": focal, "gap_um": gap, "radius_um": radius,
                        "centre_thickness_um": mesh.centre_thickness, "n_views": len(views),
                        "panel_pixels": panel_pixels, "resolution": resolution}
    report["accept"] = acceptance(report)
    (work / "report.json").write_text(json.dumps(report, indent=1))
    return report


def metrics(view_dir, views, centres):
    """A lens's field direction is where it is seen from the pupil centre (its
    chief direction). Blur and ghosts are measured about it over the whole pupil."""
    d = _unit(np.load(view_dir / "direction.npy").astype(np.float64)).reshape(-1, 3)
    n_lens, n_views = len(centres), len(views)
    lens_maps = [np.load(view_dir / f"lens_{k}.npy").ravel() for k in range(n_views)]
    centre = int(np.argmin(np.linalg.norm(views, axis=1)))
    if np.linalg.norm(views[centre]) > 1e-9:
        raise ValueError("the view list must contain the pupil centre")

    lm = lens_maps[centre]
    ok = lm >= 0
    count = np.bincount(lm[ok], minlength=n_lens)
    mean = np.stack([np.bincount(lm[ok], weights=d[ok, c], minlength=n_lens) for c in range(3)], axis=1)
    chief = count > 0
    mean[chief] = _unit(mean[chief])
    pitch = np.where(chief, ft.target_pitch_rad(_ecc(mean)), np.inf)

    seen, inl, sq, total, sq_all, inlier_views = (np.zeros(n_lens) for _ in range(6))
    for lm in lens_maps:
        ok = lm >= 0
        lens, dirs = lm[ok], d[ok]
        seen[np.unique(lens)] += 1
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", dirs, mean[lens]), -1.0, 1.0))
        keep = ang <= 3.0 * pitch[lens]
        total += np.bincount(lens, minlength=n_lens)
        sq_all += np.bincount(lens, weights=ang**2, minlength=n_lens)
        inl += np.bincount(lens[keep], minlength=n_lens)
        sq += np.bincount(lens[keep], weights=ang[keep] ** 2, minlength=n_lens)
        inlier_views[np.unique(lens[keep])] += 1

    valid = chief & (inl > 0) & _in_fov(mean)
    blur_rad = np.sqrt(sq[valid] / inl[valid])
    # A spread needs rays from at least two pupil points: a lens whose inliers
    # all come from one view would score a perfect (zero) blur, so it fails.
    blur_rad[inlier_views[valid] < 2] = np.inf
    blur = blur_rad / pitch[valid]
    spread_all_rad = np.sqrt(sq_all[valid] / total[valid])
    fill = seen[valid] / n_views
    ghost = 1.0 - inl[valid] / total[valid]

    pairs = cKDTree(centres).query_pairs(r=1.01 * math.sqrt(3.0) * SIDE_UM, output_type="ndarray")
    both = chief[pairs[:, 0]] & chief[pairs[:, 1]]
    pairs = pairs[both]
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", mean[pairs[:, 0]], mean[pairs[:, 1]]), -1.0, 1.0))
    spacing_sum = np.bincount(pairs.ravel(), weights=np.repeat(ang, 2), minlength=n_lens)
    spacing_n = np.bincount(pairs.ravel(), minlength=n_lens)
    has = valid & (spacing_n > 0)
    ratio = spacing_sum[has] / spacing_n[has] / pitch[has]

    gx, gz = np.meshgrid(np.radians(np.arange(-FOV_X_DEG, FOV_X_DEG + 1e-9, 0.25)),
                         np.radians(np.arange(-FOV_Z_DEG, FOV_Z_DEG + 1e-9, 0.25)))
    grid = _unit(np.stack([np.tan(gx), np.ones_like(gx), np.tan(gz)], axis=-1).reshape(-1, 3))
    dist, _ = cKDTree(mean[chief]).query(grid)
    coverage = float(np.mean(dist <= ft.target_pitch_rad(_ecc(grid))))

    fov_rays = _in_fov(d)
    throughput = float(np.mean([np.mean(lm[fov_rays] >= 0) for lm in lens_maps]))

    tx, tz = _field_deg(mean[valid])
    np.savez_compressed(view_dir.parent / "per_lens.npz", tx=tx, tz=tz, blur=blur, blur_rad=blur_rad,
                        spread_all_rad=spread_all_rad, fill=fill, ghost=ghost, lens=np.nonzero(valid)[0],
                        ratio_lens=np.nonzero(has)[0], ratio=ratio,
                        centres=centres, mean=mean, inliers=inl)

    def pct(x, q):
        # nearest rank: interpolating between inf entries would give NaN
        return float(np.percentile(x, q, method="nearest"))

    return {
        "lenses_in_fov": int(valid.sum()),
        "coverage": coverage,
        "throughput": throughput,
        "ratio": {"median": pct(ratio, 50), "p5": pct(ratio, 5), "p95": pct(ratio, 95)},
        "blur_pitch": {"median": pct(blur, 50), "p90": pct(blur, 90), "ideal": HEX_RMS_PITCH,
                       "single_view_fraction": float(np.mean(np.isinf(blur)))},
        "fill": {"p10": pct(fill, 10), "median": pct(fill, 50)},
        "ghost": {"mean": float(ghost.mean()), "p90": pct(ghost, 90)},
    }


def geometry(design, remapper, mesh, gap):
    origin, basis = np.asarray(design["panel_pose"]["origin_mm"]), np.asarray(design["panel_pose"]["basis"])

    def to_world(uvw_um):
        return origin + np.asarray(uvw_um) * 1e-3 @ basis

    pupil = np.array([0.0, lp.PUPIL_Y_MM, 0.0])
    axis = np.array([0.0, 1.0, 0.0])
    verts, faces, off = [to_world(mesh.verts)], [mesh.faces], len(mesh.verts)
    r = np.load(remapper)
    for k in range(int(r["n_surfaces"])):
        verts.append(r[f"surf{k}_verts"])
        faces.append(r[f"surf{k}_faces"] + off)
        off += len(r[f"surf{k}_verts"])
    half = mesh.verts[:, 0].max()
    panel = to_world(np.array([[-half, -half, -gap], [half, -half, -gap], [half, half, -gap], [-half, half, -gap]]))
    verts.append(panel)
    faces.append(np.array([[0, 1, 2], [0, 2, 3]]) + off)
    v, f = np.concatenate(verts), np.concatenate(faces)
    return {"eye_relief_mm": _axis_hit_mm(v, f, pupil, axis),
            "clearance_mm": float(np.min(np.linalg.norm(v - pupil, axis=1)))}


def acceptance(rep):
    a = ACCEPT
    checks = {
        "coverage": rep["coverage"] >= a["coverage_min"],
        "ratio_median": a["ratio_median_range"][0] <= rep["ratio"]["median"] <= a["ratio_median_range"][1],
        "ratio_spread": (rep["ratio"]["p5"] >= a["ratio_p5_p95_range"][0]
                         and rep["ratio"]["p95"] <= a["ratio_p5_p95_range"][1]),
        "blur": rep["blur_pitch"]["p90"] <= a["blur_p90_max_pitch"],
        "fill": rep["fill"]["p10"] >= a["fill_p10_min"],
        "ghost": rep["ghost"]["mean"] <= a["ghost_mean_max"],
        "throughput": rep["throughput"] >= a["throughput_min"],
        "eye_relief": rep["geometry"]["eye_relief_mm"] >= a["eye_relief_min_mm"],
        "clearance": rep["geometry"]["clearance_mm"] >= a["clearance_min_mm"],
    }
    return {"checks": checks, "passed": all(checks.values()), "thresholds": {k: list(v) if isinstance(v, tuple)
                                                                                 else v for k, v in a.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design_dir")
    ap.add_argument("--work", default="")
    ap.add_argument("--pixels", type=int, default=2044)
    args = ap.parse_args()
    work = Path(args.work) if args.work else Path(args.design_dir) / "evaluation"
    rep = evaluate(args.design_dir, work, panel_pixels=args.pixels)
    print(json.dumps({k: rep[k] for k in rep if k != "design"}, indent=1))
    print("ACCEPTED" if rep["accept"]["passed"] else "REJECTED")


if __name__ == "__main__":
    main()

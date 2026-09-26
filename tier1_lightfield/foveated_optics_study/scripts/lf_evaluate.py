"""Shared, design-independent acceptance evaluator for a light-field screen.

    python lf_evaluate.py <design_dir> [--work <dir>] [--pixels 2044]

<design_dir>/design.json:
    {"focal_um": lenslet focal length        (a uniform array), or
     "lenslets": "variable_retina",          (variable_lenslets, with "lenslet_flip_v": +1 or -1)
     "panel_pose": {"origin_mm": [x, y, z], "basis": [u_world, v_world, w_world]},
     "remapper_npz": "remapper.npz",          (relative to design_dir)
     "mask_npz": "mask.npz"}                  (exactly when HOLOPIXEL_FIELD_SHAPE=ellipse: the black
                                              mask over the lenslets: grid_u, grid_v, open)

remapper.npz: n_surfaces, and per surface k: surf{k}_verts (N, 3) world mm,
surf{k}_faces (M, 3), surf{k}_normals (3M, 3) optional loop normals,
surf{k}_kind in {glass, mirror, absorber}, surf{k}_index (glass only).

Every number comes from Cycles: each pupil camera ray is traced through the
real geometry to a panel pixel, and the same render reports the lens the ray
entered (lf_blender.entry_marking_glass). Per lens, the rays that enter it give its field direction (pupil
centre), its angular blur (RMS width of the beams of its pixels, what the eye
perceives; the RMS spread over the whole pupil is kept as a diagnostic), its pupil fill
(fraction of pupil points that see it) and its landing offset at the panel
(the chief ray's tilt there). A ghost is a ray landing on a pixel that is also
reached through a lens more than 3 pitches away, or by stray light (a ray
through no lens top or through two, or more than STRAY_DEG from its lens's direction). Pitches are the foveation
target's lens pitch at the lens's own field direction. Blur is reported in
units of what the eye could see there (foveation_target.blur_tolerance_rad:
the retinal pitch, clipped at the RMS diffraction blur of the 4 mm pupil).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import foveation_target as ft
import lf_pipeline as lp
import mla_design as mla
import mla_mesh
import screen_spec as spec
import variable_lenslets as vl

SIDE_UM, PIXEL_UM, INDEX, MIN_THICKNESS_UM = (spec.LENS_SIDE_UM, spec.PIXEL_UM, spec.LENS_INDEX,
                                              spec.LENS_MIN_THICKNESS_UM)
FOV_X_DEG, FOV_Z_DEG = spec.FIELD_HALF_DEG
# The evaluation camera (square, rectilinear) sees the field plus 3 deg, with the
# pixel angle at its centre of 76 deg over 2048 px (its size at 70 x 45).
CAMERA_FOV_DEG = 2.0 * (max(FOV_X_DEG, FOV_Z_DEG) + 3.0)
CAMERA_RESOLUTION = round(2048 * math.tan(math.radians(CAMERA_FOV_DEG / 2)) / math.tan(math.radians(38.0)))
HEX_RMS_PITCH = math.sqrt(5.0 / 12.0) / math.sqrt(3.0)
# A ray this far from its lens's chief direction is stray light, not the lens's
# light field: pupil parallax keeps a lens's own rays within ~5 deg (99.99 % of
# them on the spline pancake), and 10 deg would need a virtual image nearer
# than ~11 mm; the stray rays measured there sit at 20 to 63 deg.
STRAY_DEG = 10.0

ACCEPT = {
    "coverage_min": 0.98,
    "ratio_median_range": (0.8, 1.25),
    "ratio_p5_p95_range": (0.67, 1.5),
    "blur_p90_max_pitch": 0.75,
    "fill_p10_min": 0.8,
    "ghost_mean_max": 0.05,
    "throughput_min": 0.9,
    "eye_relief_min_mm": 15.0,        # the pancake search minimum (pancake_search.EYE_RELIEF_MIN_MM)
    "clearance_min_mm": 15.0,
}


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _field_deg(d):
    return np.degrees(np.arctan2(d[..., 0], d[..., 1])), np.degrees(np.arctan2(d[..., 2], d[..., 1]))


def _in_fov(d):
    return spec.in_field(*_field_deg(d)) & (d[..., 1] > 0)


def coverage_grid():
    """The field directions coverage is scored on: a 0.25 deg grid inside the
    field, (tx, tz) rad, flat."""
    gx, gz = np.meshgrid(np.arange(-FOV_X_DEG, FOV_X_DEG + 1e-9, 0.25), np.arange(-FOV_Z_DEG, FOV_Z_DEG + 1e-9, 0.25))
    keep = spec.in_field(gx, gz)
    return np.radians(gx[keep]), np.radians(gz[keep])


def _ecc(d):
    return np.arccos(np.clip(d[..., 1], -1.0, 1.0))


def _axis_hit_mm(verts, faces, origin, direction):
    """Nearest positive ray/triangle hit (Moller-Trumbore), inf when none. The
    barycentric test is widened by 1e-9 so a ray exactly on a shared edge
    cannot pass between its two triangles by rounding."""
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
    eps = 1e-9
    hit = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps) & (t > 1e-9)
    return float(t[hit].min()) if hit.any() else math.inf


def validate_surfaces(remapper):
    """Glass must be a closed solid, wound consistently and outwards, or Cycles
    refracts with the index inverted. Fail before rendering, naming the surface."""
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
        directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        _, dcounts = np.unique(directed, axis=0, return_counts=True)
        if not np.all(dcounts == 1):
            raise ValueError(f"surface {k}: glass winding is inconsistent ({int(np.sum(dcounts != 1))} edges "
                             "run the same way in two faces)")
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        if np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) <= 0:
            raise ValueError(f"surface {k}: glass is wound inwards (negative signed volume)")


def render_views(design_dir, work, panel_pixels=spec.PANEL_PIXELS, view_spacing_mm=0.5,
                 resolution=CAMERA_RESOLUTION, fov_deg=CAMERA_FOV_DEG, subdivisions=2, mla_index=INDEX):
    """The Cycles pass alone: per pupil view, the panel pixel and the entered lens
    of every camera ray (work/views). The lenslets are built for INDEX and render
    at mla_index (another wavelength). Returns the design, remapper path, lenslet
    mesh, air gap, lenslet summary and the views."""
    design_dir, work = Path(design_dir), Path(work)
    design = json.loads((design_dir / "design.json").read_text())
    if design["field_deg"] != spec.FIELD_DEG:
        raise ValueError(f"{design_dir}: the design was made for a {design['field_deg']} deg field, "
                         f"not this run's {spec.FIELD_DEG} (HOLOPIXEL_FIELD_DEG)")
    if ("mask_npz" in design) != (spec.FIELD_SHAPE == "ellipse"):
        raise ValueError(f"{design_dir}: a design has a black mask (design.json mask_npz) exactly when its "
                         f"field is elliptic; this field is {spec.FIELD_SHAPE}")
    remapper = (design_dir / design["remapper_npz"]).resolve()
    validate_surfaces(remapper)
    mesh, gap, lenslets = lenslet_array(design, design_dir, panel_pixels * PIXEL_UM, subdivisions)
    work.mkdir(parents=True, exist_ok=True)
    np.savez(work / "mla_mesh.npz", verts=mesh.verts, faces=mesh.faces, loop_normals=mesh.loop_normals,
             face_lens=mesh.face_lens, face_wall=mesh.face_lens == mla_mesh.WALL)
    views = lp.hex_views_mm(view_spacing_mm, 2.0)
    cfg = {
        "mode": "evaluate", "mla_npz": str(work / "mla_mesh.npz"), "index": mla_index,
        "panel_pose": design["panel_pose"], "gap_um": gap, "panel_pixels": panel_pixels,
        "pixel_um": PIXEL_UM, "camera": {"resolution": resolution, "fov_deg": fov_deg},
        "views_mm": views.tolist(), "tmp_dir": str(work / "views"), "remapper_npz": str(remapper),
        "out_npz": str(work / "evaluate.npz"),
    }
    lp.run_blender(cfg, work)
    return design, remapper, mesh, gap, lenslets, views


def evaluate(design_dir, work, panel_pixels=spec.PANEL_PIXELS, view_spacing_mm=0.5,
             resolution=CAMERA_RESOLUTION, fov_deg=CAMERA_FOV_DEG, subdivisions=2):
    work = Path(work)
    design, remapper, mesh, gap, lenslets, views = render_views(design_dir, work, panel_pixels, view_spacing_mm,
                                                                resolution, fov_deg, subdivisions)
    report = metrics(work / "views", views, mesh.centres, panel_pixels)
    report["geometry"] = geometry(design, remapper, mesh, gap)
    report["design"] = {**lenslets, "gap_um": gap, "centre_thickness_um": mesh.centre_thickness,
                        "n_views": len(views),
                        "panel_pixels": panel_pixels, "resolution": resolution}
    report["accept"] = acceptance(report)
    (work / "report.json").write_text(json.dumps(report, indent=1))
    return report


def lenslet_array(design, design_dir, panel_um, subdivisions):
    """The design's lenslet array: exactly one of "focal_um" (uniform) or
    "lenslets": "variable_retina", the latter with the design's black mask
    (variable_lenslets.with_mask) when it names one ("mask_npz": grid_u, grid_v,
    open). Returns the mesh, the air gap and a summary."""
    kinds = [k for k in ("focal_um", "lenslets") if k in design]
    if len(kinds) != 1:
        raise ValueError(f"design.json needs exactly one of focal_um, lenslets; has {kinds}")
    if kinds[0] == "focal_um":
        focal = float(design["focal_um"])
        radius = mla.radius_for_focal_length_um(focal, INDEX)
        mesh = mla_mesh.build(panel_um=panel_um, side_um=SIDE_UM, radius_um=radius,
                              min_thickness_um=MIN_THICKNESS_UM, subdivisions=subdivisions)
        return mesh, mla.back_focal_gap_um(radius, INDEX, mesh.centre_thickness), {"focal_um": focal}
    if design["lenslets"] != "variable_retina":
        raise ValueError(f"unknown lenslets {design['lenslets']!r}")
    if abs(panel_um - vl.PANEL_UM) > 1e-6:
        raise ValueError("the retina-matched lenslet array is defined on the full panel")
    v = vl.build(int(design["lenslet_flip_v"]), subdivisions)
    if "mask_npz" in design:
        m = np.load(Path(design_dir) / design["mask_npz"])
        v = vl.with_mask(v, (m["grid_u"], m["grid_v"], m["open"]))
    return v.mesh, v.gap_um, {"lenslets": "variable_retina", "lenslet_flip_v": int(design["lenslet_flip_v"]),
                              "focal_um_range": [float(v.focal_um.min()), float(v.focal_um.max())]}


def _group_median(keys, values, n):
    """Per key in range(n), the median of its values; 0 for a key without any."""
    order = np.lexsort((values, keys))
    k, v = keys[order], values[order]
    count = np.bincount(k, minlength=n)
    start = np.cumsum(count) - count
    has = count > 0
    out = np.zeros(n)
    out[has] = 0.5 * (v[(start + (count - 1) // 2)[has]] + v[(start + count // 2)[has]])
    return out


def metrics(view_dir, views, centres, panel_pixels):
    """Rays are grouped by the lens they ENTER (read from Cycles), not by the lens
    above the pixel they land on: with a tilted chief ray a lens's pupil cone
    lands under its neighbour, and that is correct, not crosstalk.

    Per lens: chief direction (median of the pupil-centre rays, so one stray ray
    cannot move it), blur (ray-weighted
    RMS width of the beams of the pixels it lights, what the eye perceives),
    lens spread (RMS angle of all its pupil rays about its direction: full-pupil
    focus, a diagnostic that is stricter than perception whenever each pixel
    covers only part of the pupil), fill (share of pupil points that see it),
    landing offset (pupil-centre landing point minus lens centre: the chief
    ray's tilt at the panel, in um). A ghost is a ray whose pixel is also reached
    through a lens more than 3 pitches away: that pixel must serve two field
    directions, so one of them sees wrong content. Stray light (a ray that
    reaches a pixel through no lens top or through two (entered -1), or more
    than STRAY_DEG from its lens's chief direction) is left out of every lens metric, makes its pixel a ghost
    pixel, and counts as a ghost of the lens it entered.

    The per-view passes run on the GPU (fp64). Their float sums are atomic, in
    no fixed order: results agree with a CPU run to ~1e-8, not bit for bit."""
    d = _unit(np.load(view_dir / "direction.npy").astype(np.float64)).reshape(-1, 3)
    n_lens, n_views = len(centres), len(views)
    order = np.argsort(np.linalg.norm(views, axis=1))
    if np.linalg.norm(views[order[0]]) > 1e-9:
        raise ValueError("the view list must contain the pupil centre")

    def rays(k):
        pix = np.load(view_dir / f"pix_{k}.npy").ravel()
        ent = np.load(view_dir / f"entered_{k}.npy").ravel()
        ok = pix >= 0
        return np.nonzero(ok)[0], pix[ok], ent[ok]

    def stray(di, ent):
        """Rays (directions di) through no lens, or far from their lens's chief direction."""
        lens = np.maximum(ent, 0)
        far = chief[lens] & (np.einsum("ij,ij->i", di, mean[lens]) < math.cos(math.radians(STRAY_DEG)))
        return (ent < 0) | far

    idx, pix, ent = rays(order[0])
    lensed = ent >= 0
    mean = np.stack([_group_median(ent[lensed], d[idx[lensed], c], n_lens) for c in range(3)], axis=1)
    chief = np.bincount(ent[lensed], minlength=n_lens) > 0
    mean[chief] = _unit(mean[chief])
    good = ~stray(d[idx], ent)
    idx, pix, ent = idx[good], pix[good], ent[good]
    count = np.bincount(ent, minlength=n_lens)
    chief &= count > 0                                                 # every centre ray stray: no direction
    pitch = np.where(chief, ft.target_pitch_rad(*np.radians(_field_deg(mean))), np.inf)
    uv = np.column_stack([(pix % panel_pixels + 0.5) * PIXEL_UM, (pix // panel_pixels + 0.5) * PIXEL_UM])
    uv -= panel_pixels * PIXEL_UM / 2
    offset = np.stack([np.bincount(ent, weights=uv[:, c], minlength=n_lens) for c in range(2)], axis=1)
    offset[chief] = offset[chief] / count[chief, None] - centres[chief]

    dev = torch.device("cuda")
    gpu = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    n_pix = panel_pixels**2
    d_g, mean_g, chief_g, pitch_g = gpu(d), gpu(mean), gpu(chief), gpu(pitch)
    cos_stray = math.cos(math.radians(STRAY_DEG))

    def rays_gpu(k):
        pix_k = gpu(np.load(view_dir / f"pix_{k}.npy").ravel())
        ent_k = gpu(np.load(view_dir / f"entered_{k}.npy").ravel())
        ok = pix_k >= 0
        return torch.nonzero(ok)[:, 0], pix_k[ok].long(), ent_k[ok].long()

    first = torch.full((n_pix,), -1, dtype=torch.int64, device=dev)
    conflict = torch.zeros(n_pix, dtype=torch.bool, device=dev)
    pix_sum = torch.zeros(n_pix, 3, dtype=torch.float64, device=dev)
    pix_n = torch.zeros(n_pix, dtype=torch.float64, device=dev)
    seen, total, sq_all, delivered = (torch.zeros(n_lens, dtype=torch.float64, device=dev) for _ in range(4))
    n_stray = n_reach = 0
    in_fov = gpu(_in_fov(d))
    n_fov = int(in_fov.sum())
    reached = []                                                       # throughput: in-field rays through a lens
    for k in order:
        idx, pix, ent = rays_gpu(k)
        lensed = ent >= 0
        reached.append(int(in_fov[idx[lensed]].sum()) / n_fov)
        di = d_g[idx]
        lens = ent.clamp(min=0)
        bad = ~lensed | (chief_g[lens] & ((di * mean_g[lens]).sum(-1) < cos_stray))   # stray()
        conflict[pix[bad]] = True
        n_stray, n_reach = n_stray + int(bad.sum()), n_reach + len(pix)
        delivered += torch.bincount(ent[lensed], minlength=n_lens)    # every ray a lens sends, stray or not
        di, pix, ent = di[~bad], pix[~bad], ent[~bad]
        pix_sum.index_add_(0, pix, di)
        pix_n += torch.bincount(pix, minlength=n_pix)
        per_lens = torch.bincount(ent, minlength=n_lens)
        seen += per_lens > 0
        total += per_lens
        ang = torch.arccos(torch.clamp((di * mean_g[ent]).sum(-1), -1.0, 1.0))
        sq_all.index_add_(0, ent, ang**2)
        # a pixel's first lens; among one view's rays on a new pixel the last one
        # in raster order wins, as a numpy assignment would pick it
        new = first[pix] < 0
        last = torch.full((n_pix,), -1, dtype=torch.int64, device=dev)
        last.scatter_reduce_(0, pix[new], torch.nonzero(new)[:, 0], "amax")
        taken = last >= 0
        first[taken] = ent[last[taken]]
        other = first[pix]
        differs = (other != ent) & chief_g[ent] & chief_g[other]
        e, o = ent[differs], other[differs]
        far = torch.arccos(torch.clamp((mean_g[e] * mean_g[o]).sum(-1), -1.0, 1.0)) > 3.0 * pitch_g[e]
        conflict[pix[differs][far]] = True
    ghosts = torch.zeros(n_lens, dtype=torch.float64, device=dev)
    for k in order:
        _, pix, ent = rays_gpu(k)
        hit_ghost = (ent >= 0) & conflict[pix]
        ghosts += torch.bincount(ent[hit_ghost], minlength=n_lens)
    first, pix_sum, pix_n, seen, total, sq_all, delivered, ghosts = (
        t.cpu().numpy() for t in (first, pix_sum, pix_n, seen, total, sq_all, delivered, ghosts))

    valid = chief & (total > 0) & _in_fov(mean)
    lens_spread_rad = np.sqrt(sq_all[valid] / total[valid])
    lens_spread_rad[seen[valid] < 2] = np.inf

    # Perceived blur: the angular width of each PIXEL's beam at the eye. Every
    # pixel is encoded for its own mean direction, so a lens whose pixels point in
    # slightly different directions (a sheared light field) is not blurred; only
    # the spread of the rays that reach ONE pixel is. RMS angle about the pixel's
    # mean direction: for unit vectors, 2 * (1 - |mean vector|) to second order,
    # times n / (n - 1) because the mean is estimated from the same n rays. One
    # ray gives no spread, so such a pixel is not measured.
    hit = pix_n >= 2
    width2 = np.zeros(panel_pixels**2)
    n_hit = pix_n[hit]
    width2[hit] = 2.0 * (1.0 - np.linalg.norm(pix_sum[hit], axis=1) / n_hit) * n_hit / (n_hit - 1.0)
    lens_of_pixel = np.where(hit, first, -1)
    use = lens_of_pixel >= 0
    wsum = np.bincount(lens_of_pixel[use], weights=(width2 * pix_n)[use], minlength=n_lens)
    wn = np.bincount(lens_of_pixel[use], weights=pix_n[use], minlength=n_lens)
    blur_rad = np.full(int(valid.sum()), np.inf)
    measured = wn[valid] > 0
    blur_rad[measured] = np.sqrt(np.maximum(wsum[valid][measured], 0.0) / wn[valid][measured])
    # Blur is judged against what the eye could see there: the larger of the
    # retina's own pitch and the display's sampling pitch in that direction.
    tx_rad, tz_rad = (np.radians(a) for a in _field_deg(mean[valid]))
    tolerance = ft.blur_tolerance_rad(tx_rad, tz_rad)
    blur = blur_rad / tolerance
    lens_spread = lens_spread_rad / tolerance
    fill = seen[valid] / n_views
    ghost = ghosts[valid] / delivered[valid]
    landing_um = np.linalg.norm(offset[valid], axis=1)
    inl = total

    pairs = cKDTree(centres).query_pairs(r=1.01 * math.sqrt(3.0) * SIDE_UM, output_type="ndarray")
    both = chief[pairs[:, 0]] & chief[pairs[:, 1]]
    pairs = pairs[both]
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", mean[pairs[:, 0]], mean[pairs[:, 1]]), -1.0, 1.0))
    spacing_sum = np.bincount(pairs.ravel(), weights=np.repeat(ang, 2), minlength=n_lens)
    spacing_n = np.bincount(pairs.ravel(), minlength=n_lens)
    has = valid & (spacing_n > 0)
    ratio = spacing_sum[has] / spacing_n[has] / pitch[has]

    gx, gz = coverage_grid()
    grid = _unit(np.stack([np.tan(gx), np.ones_like(gx), np.tan(gz)], axis=-1))
    dist, _ = cKDTree(mean[chief]).query(grid)
    coverage = float(np.mean(dist <= ft.target_pitch_rad(gx, gz)))

    throughput = float(np.mean(reached))

    tx, tz = _field_deg(mean[valid])
    np.savez_compressed(view_dir.parent / "per_lens.npz", tx=tx, tz=tz, blur=blur, blur_rad=blur_rad,
                        lens_spread=lens_spread, pixel_width_rad=np.sqrt(np.maximum(width2, 0.0))[hit],
                        lens_spread_rad=lens_spread_rad, fill=fill, ghost=ghost, lens=np.nonzero(valid)[0],
                        ratio_lens=np.nonzero(has)[0], ratio=ratio, landing_um=landing_um,
                        centres=centres, mean=mean, inliers=inl)

    def pct(x, q):
        # nearest rank: interpolating between inf entries would give NaN
        if len(x) == 0:
            raise ValueError("no lens has a lit neighbour: the camera is too coarse for the lens pitch")
        return float(np.percentile(x, q, method="nearest"))

    return {
        "lenses_in_fov": int(valid.sum()),
        "coverage": coverage,
        "throughput": throughput,
        "ratio": {"median": pct(ratio, 50), "p5": pct(ratio, 5), "p95": pct(ratio, 95)},
        "blur_pitch": {"median": pct(blur, 50), "p90": pct(blur, 90),
                       "single_view_fraction": float(np.mean(np.isinf(blur)))},
        "lens_spread_pitch": {"median": pct(lens_spread, 50), "p90": pct(lens_spread, 90)},
        "fill": {"p10": pct(fill, 10), "median": pct(fill, 50)},
        "ghost": {"mean": float(ghost.mean()), "p90": pct(ghost, 90)},
        "stray": {"fraction": n_stray / n_reach},
        "landing_offset_um": {"median": pct(landing_um, 50), "p90": pct(landing_um, 90)},
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

"""Put the best GPU-search candidates side by side in one Blender scene, to eyeball.

    python show_candidates.py <out.blend> <best_*.json> [<best_*.json> ...]

Each candidate: its lenses as closed solids of revolution (sized to the radius
its rays actually use, plus 1 mm), the flat panel, a schematic eye for scale,
and meridional ray fans at 0, 10, 20, 30 deg and the field corner, colour-coded
by field. Candidates sit 70 mm apart along X, each labelled with its element
count and its worst per-field blur relative to the eye's tolerance. The world
frame is the eye's (+Y optical axis, pupil at y = -3.6 mm).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import gpu_search as gs
import gpu_tracer as gt

HERE = Path(__file__).resolve().parent
PUPIL_Y = -3.6
SPACING_MM = 70.0
FAN_FIELDS = [0.0, 10.0, 20.0, 30.0, gs.MAX_FIELD_DEG]
FAN_PUPIL = [(0.0, py) for py in (-1.0, -0.5, 0.0, 0.5, 1.0)]


def _sag_and_slope(r2, surf):
    c, k, a = (torch.as_tensor(np.asarray(q, dtype=np.float64)) for q in surf)
    r2 = torch.as_tensor(r2)
    s, ok = gt.sag(r2, c, k, a)
    if not bool(ok.all()):
        raise ValueError("the aperture reaches outside the sag domain of the surface")
    return s.numpy(), gt.dsag_dr2(r2, c, k, a).numpy()


def lathe(z_front, z_back, surf_front, surf_back, aperture, n_r=48, n_phi=96):
    """Closed solid of revolution between two even-asphere surfaces (axis z),
    wound outwards. Returns verts, triangles and exact loop normals (3 per
    triangle): the analytic surface normals on the faces, radial on the rim."""
    r = np.linspace(0.0, aperture, n_r)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    rings = np.stack([np.cos(phi), np.sin(phi)], -1)
    verts, normals = [], []
    for z0, surf, side in ((z_front, surf_front, -1.0), (z_back, surf_back, 1.0)):
        s, g = _sag_and_slope(r**2, surf)
        xy = np.concatenate([[[0.0, 0.0]], (r[1:, None, None] * rings[None]).reshape(-1, 2)])
        ri = np.concatenate([[0], np.repeat(np.arange(1, n_r), n_phi)])
        verts.append(np.column_stack([xy, z0 + s[ri]]))
        # z - sag(r^2) = 0 has gradient (-2 g x, -2 g y, 1); outward is -z at the front
        n = side * np.column_stack([-2 * g[ri, None] * xy, np.ones(len(xy))])
        normals.append(n / np.linalg.norm(n, axis=1, keepdims=True))
    v = np.concatenate(verts)
    face_n = np.concatenate(normals)
    ring = lambda base, i, j: base + 1 + i * n_phi + (j % n_phi)  # noqa: E731
    b0 = len(verts[0])
    tri, rim = [], []
    for base, flip in ((0, True), (b0, False)):
        for j in range(n_phi):
            t = (base, ring(base, 0, j), ring(base, 0, j + 1))
            tri.append(t[::-1] if flip else t)
        for i in range(n_r - 2):
            for j in range(n_phi):
                q = (ring(base, i, j), ring(base, i + 1, j), ring(base, i + 1, j + 1), ring(base, i, j + 1))
                q = q[::-1] if flip else q
                tri += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    last = n_r - 2
    for j in range(n_phi):
        q = (ring(0, last, j), ring(0, last, j + 1), ring(b0, last, j + 1), ring(b0, last, j))
        rim += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    t = np.array(tri + rim)
    loop_n = face_n[t].reshape(-1, 3)
    rim_corners = v[np.array(rim)].reshape(-1, 3)
    radial = np.column_stack([rim_corners[:, :2], np.zeros(len(rim_corners))])
    loop_n[3 * len(tri):] = radial / np.linalg.norm(radial, axis=1, keepdims=True)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    if np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) <= 0:
        raise ValueError("lathe solid is wound inwards")
    return v, t, loop_n


def candidate_solids(entry, device, n_r=48, n_phi=96):
    """Lens solids of a GPU-search candidate in the tracer frame, each sized to
    the radius its rays use over the whole field grid plus 1 mm, and the batch."""
    lay = entry["layout"]
    x = torch.tensor([entry["x"]], dtype=torch.float64, device=device)
    idx = torch.tensor([entry["indices"]], dtype=torch.float64, device=device)
    batch = gs.to_batch(x, idx, lay)
    with torch.no_grad():
        all_f = torch.tensor(gs.field_grid(), dtype=torch.float64, device=device)
        all_p = torch.tensor(gs.pupil_samples(), dtype=torch.float64, device=device)
        _, _, alive_all, diag_all = gt.trace(batch, all_f, all_p, diagnostics=True)
    used = torch.where(alive_all[:, None], diag_all["radius"], torch.zeros_like(diag_all["radius"]))
    used = used.amax(dim=(2, 3))[0].cpu().numpy()
    z, c, k, a = (t[0].cpu().numpy() for t in (batch.z, batch.c, batch.k, batch.a))
    solids = []
    for e in range(lay["n_el"]):
        s0, s1 = 2 * e, 2 * e + 1
        ap = max(used[s0], used[s1]) + 1.0
        v, t, n = lathe(z[s0], z[s1], (c[s0], k[s0], a[s0]), (c[s1], k[s1], a[s1]), ap, n_r, n_phi)
        solids.append({"verts": v, "faces": t, "normals": n, "index": entry["indices"][e], "aperture": ap,
                       "z_front_rim": z[s0] + _sag_and_slope(np.array([ap**2]), (c[s0], k[s0], a[s0]))[0][0]})
    return solids, batch


TRACER_TO_WORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # rows: tracer x, y, z


def to_world(p, offset_x):
    """Tracer frame (x, y meridian, z along the axis from the pupil) -> eye world
    frame. A proper rotation, so a solid keeps its outward winding."""
    return np.asarray(p) @ TRACER_TO_WORLD + np.array([offset_x, PUPIL_Y, 0.0])


def candidate(entry, offset_x, device):
    lay = entry["layout"]
    lens_solids, batch = candidate_solids(entry, device)
    fields = torch.tensor(FAN_FIELDS, dtype=torch.float64, device=device)
    pupil = torch.tensor(FAN_PUPIL, dtype=torch.float64, device=device)
    with torch.no_grad():
        _, _, alive, diag = gt.trace(batch, fields, pupil, diagnostics=True)
    z = batch.z[0].cpu().numpy()
    solids = [{"verts": to_world(s["verts"], offset_x).tolist(), "faces": s["faces"].tolist(),
               "index": s["index"]} for s in lens_solids]
    pts = diag["points"][0].cpu().numpy()                                # (S, F, P, 3)
    start = np.stack([np.array([px * 2.0, py * 2.0, 0.0]) for px, py in FAN_PUPIL])
    rays = []
    ok = alive[0].cpu().numpy()
    for fi in range(len(FAN_FIELDS)):
        for pi in range(len(FAN_PUPIL)):
            if not ok[fi, pi]:
                continue
            path = np.concatenate([start[pi][None], pts[:, fi, pi]], 0)
            rays.append({"field": fi, "points": to_world(path, offset_x).tolist()})
    worst = float(np.max(entry["spot_in_tolerance_per_field"]))
    label = f"{lay['n_el']} elements  worst blur {worst:.1f}x tolerance"
    return {"offset_x": offset_x, "panel_y": float(PUPIL_Y + z[-1]), "solids": solids, "rays": rays,
            "label": label, "track_mm": float(z[-1])}


def main():
    out = Path(sys.argv[1]).resolve()
    files = [Path(p) for p in sys.argv[2:]]
    device = torch.device("cuda")
    scene = {"candidates": [], "fields": FAN_FIELDS, "out_blend": str(out)}
    for i, f in enumerate(files):
        best = json.loads(f.read_text())[0]
        scene["candidates"].append(candidate(best, i * SPACING_MM, device))
        print(f"{f.name}: {scene['candidates'][-1]['label']}, track {scene['candidates'][-1]['track_mm']:.1f} mm")
    spec = out.with_suffix(".json")
    spec.write_text(json.dumps(scene))
    log = out.with_suffix(".log")
    with open(log, "w") as fh:
        proc = subprocess.run(["blender", "-b", "--factory-startup", "--python", str(HERE / "show_candidates_blender.py"),
                               "--", str(spec)], stdout=fh, stderr=subprocess.STDOUT)
    text = log.read_text()
    if proc.returncode != 0 or "SHOW_DONE" not in text or "Traceback" in text:
        raise RuntimeError(f"Blender failed, log {log}:\n{text}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

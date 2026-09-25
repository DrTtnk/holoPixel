"""Tessellate the optimised two-mirror system into the evaluator's mesh
contract: design.json + remapper.npz (surf{k}_verts/faces/normals/kind).

Each mirror is meshed on a regular (x, y) grid in its own local aperture,
with EXACT analytic loop normals (the evaluator reflects about these, so
tessellation only affects position, per its docstring). Triangulation must be
consistently wound so triangle normals point the same way as the analytic
ones (their sign does not matter for a "mirror" kind, but a consistent
right-handed winding is still checked by a test).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surfaces import DTYPE  # noqa: E402
from system import make_mirror, panel_frame  # noqa: E402


def mesh_surface(mirror, half_x, half_y, nx, ny):
    """Regular grid mesh of a Freeform surface's local (x, y) rectangle,
    clipped to an inscribed ellipse (the physical mirror is not square)."""
    xs = np.linspace(-half_x, half_x, nx)
    ys = np.linspace(-half_y, half_y, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    inside = (X / half_x) ** 2 + (Y / half_y) ** 2 <= 1.0 + 1e-9
    idx = -np.ones(X.shape, dtype=np.int64)
    idx[inside] = np.arange(int(inside.sum()))
    x_t = torch.tensor(X[inside], dtype=DTYPE)
    y_t = torch.tensor(Y[inside], dtype=DTYPE)
    with torch.no_grad():
        s = mirror.sag(x_t, y_t)
        P = mirror.to_world(x_t, y_t, s)
        n = mirror.normal_world(x_t, y_t)
    verts = P.numpy()
    normals_at_vert = n.numpy()

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            if min(a, b, c, d) < 0:
                continue
            faces.append((a, b, c))
            faces.append((a, c, d))
    faces = np.asarray(faces, dtype=np.int64)

    # orient consistently outward (world -Y-ish is "towards the eye" for M1;
    # sign does not affect physics but a flipped winding would give inward
    # face normals inconsistent with the loop normals, which the evaluator
    # only checks for "glass" kind -- still keep it right for hygiene).
    tri = verts[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vert_n = normals_at_vert[faces].mean(axis=1)
    flip = np.einsum("ij,ij->i", face_n, vert_n) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]

    loop_normals = normals_at_vert[faces].reshape(-1, 3)
    return verts, faces, loop_normals


def build_remapper_npz(params, apertures, path, grid=(161, 161)):
    m1 = make_mirror(params["m1"])
    m2 = make_mirror(params["m2"])
    (ax1, ay1), (ax2, ay2) = apertures
    v1, f1, n1 = mesh_surface(m1, ax1, ay1, *grid)
    v2, f2, n2 = mesh_surface(m2, ax2, ay2, *grid)
    np.savez(path, n_surfaces=2,
             surf0_verts=v1, surf0_faces=f1, surf0_normals=n1, surf0_kind="mirror",
             surf1_verts=v2, surf1_faces=f2, surf1_normals=n2, surf1_kind="mirror")
    return v1, f1, v2, f2


def build_design_json(params, design_dir, remapper_name="remapper.npz"):
    u_p, v_p, w_p = panel_frame(params["panel"])
    basis = torch.stack([u_p, v_p, w_p]).detach().numpy().tolist()
    origin = params["panel"]["origin"].detach().numpy().tolist()
    focal_um = float(params["focal_um"].detach())
    design = {"focal_um": focal_um, "panel_pose": {"origin_mm": origin, "basis": basis},
             "remapper_npz": remapper_name}
    design["field_deg"] = [70.0, 45.0]  # searched for the 70 x 45 deg field (lf_evaluate refuses another)
    (Path(design_dir) / "design.json").write_text(json.dumps(design, indent=1))
    return design

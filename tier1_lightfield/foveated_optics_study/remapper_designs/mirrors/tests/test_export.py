import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surfaces import DTYPE, Freeform  # noqa: E402
from export import mesh_surface  # noqa: E402


def _toy_mirror():
    terms = [(0, 2), (2, 0), (0, 3), (2, 1)]
    coeffs = torch.tensor([1e-3, 2e-3, -5e-4, 3e-4], dtype=DTYPE)
    return Freeform(torch.tensor([0.0, 20.0, 0.0], dtype=DTYPE), torch.tensor(0.4, dtype=DTYPE),
                    torch.tensor(1.0 / 150.0, dtype=DTYPE), torch.tensor(-0.3, dtype=DTYPE),
                    terms, coeffs, u_scale=15.0, v_scale=12.0)


def test_mesh_contract_shapes():
    mirror = _toy_mirror()
    verts, faces, loop_normals = mesh_surface(mirror, 15.0, 12.0, 25, 21)
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert loop_normals.shape == (3 * len(faces), 3)
    assert np.isfinite(verts).all()
    assert faces.min() >= 0 and faces.max() < len(verts)


def test_loop_normals_are_unit_and_match_analytic_sag():
    mirror = _toy_mirror()
    verts, faces, loop_normals = mesh_surface(mirror, 15.0, 12.0, 25, 21)
    norms = np.linalg.norm(loop_normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)

    # Recompute the normal at a few face-corner vertices directly from the
    # surface (independent of the mesh's own bookkeeping) and compare.
    corners = verts[faces].reshape(-1, 3)
    x, y, z = mirror.local_coords(torch.tensor(corners, dtype=DTYPE))
    with torch.no_grad():
        n_direct = mirror.normal_world(x, y).numpy()
    assert np.allclose(np.abs((n_direct * loop_normals).sum(axis=1)), 1.0, atol=1e-6)


def test_no_degenerate_triangles():
    mirror = _toy_mirror()
    verts, faces, _ = mesh_surface(mirror, 15.0, 12.0, 20, 17)
    tri = verts[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    assert np.all(area2 > 1e-9)

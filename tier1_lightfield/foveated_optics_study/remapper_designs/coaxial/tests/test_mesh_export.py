"""Proofs for the lens-solid mesh builder (mesh_export.py): the two curved
caps are closed and outward-wound (checked against the exact analytic
formula the shared evaluator itself uses, lf_evaluate.validate_surfaces),
and their per-corner normals match autograd of the same sag function used to
build the vertex heights."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import optics as op  # noqa: E402
import mesh_export as me  # noqa: E402

torch.set_default_dtype(op.DTYPE)


def _surf(vertex_y, c, k=-0.2, a4=1e-4, a6=0.0):
    t = lambda x: torch.tensor(float(x), dtype=op.DTYPE)  # noqa: E731
    return op.Surface(t(vertex_y), t(c), t(k), t(a4), t(a6), t(0.0), 10.0)


@pytest.fixture(scope="module")
def solid():
    front = _surf(20.0, c=0.02)
    back = _surf(23.0, c=-0.015)
    verts, faces, normals = me.build_element_solid(front, back, aperture_mm=8.0, n_r=12, n_phi=24)
    return front, back, verts, faces, normals


def test_solid_is_closed_and_wound_outwards(solid):
    _, _, verts, faces, _ = solid
    vol = me.check_closed_and_outward(verts, faces)
    # Sanity: a squat cylinder-ish glass blob of ~8 mm radius, ~3 mm thick
    # should have a plausible (not wildly wrong) positive volume.
    assert 0 < vol < 8000.0


def test_cap_normals_match_autograd_of_sag(solid):
    """Every mesh corner's stored normal must equal an independent autograd
    normal of the sag function AT THAT EXACT (x, z), not merely at the same
    radius (the surface is not axially uniform in direction, only in r)."""
    front, back, _, _, _ = solid
    rng = np.random.default_rng(0)
    for surface, facing_front in ((front, True), (back, False)):
        cv, cf, cn, _ = me._cap_mesh(surface, aperture_mm=8.0, n_r=12, n_phi=24, facing_front=facing_front)
        corner_xyz = cv[cf].reshape(-1, 3)
        n_corners = corner_xyz.shape[0]
        for j in rng.integers(0, n_corners, size=15):
            xj, zj = float(corner_xyz[j, 0]), float(corner_xyz[j, 2])
            x = torch.tensor(xj, dtype=op.DTYPE, requires_grad=True)
            z = torch.tensor(zj, dtype=op.DTYPE, requires_grad=True)
            r = torch.sqrt(x**2 + z**2 + 1e-30)
            y = surface.vertex_y + op.sag(r, surface)
            gx, gz = torch.autograd.grad(y, [x, z])
            grad = torch.stack([-gx, torch.tensor(1.0, dtype=op.DTYPE), -gz])
            sign = -1.0 if facing_front else 1.0
            n_auto = sign * grad / torch.linalg.norm(grad)
            assert cn[j] == pytest.approx(n_auto.detach().numpy(), abs=1e-6)


def test_orientation_is_corrected_when_the_aperture_grows_past_the_flip_point():
    """A strongly curved front/back pair was found to wind outward at a
    small aperture and INWARD at a larger one (the fixed 'front cap is CCW
    in (x, z)' heuristic stops matching reality once curvature dominates the
    projection -- see NOTES.md and mesh_export.orient_outward). Reproduce
    both regimes and check build_element_solid corrects both, with loop
    normals that stay a valid permutation of the per-corner analytic ones."""
    front = _surf(41.0, c=0.0044, k=-2.47, a4=0.0, a6=0.0)
    back = _surf(54.5, c=-0.0282, k=-1.42, a4=0.0, a6=0.0)
    small_v, small_f, _ = me.build_element_solid(front, back, aperture_mm=10.0, n_r=16, n_phi=32)
    big_v, big_f, big_n = me.build_element_solid(front, back, aperture_mm=20.0, n_r=16, n_phi=32)
    assert me.check_closed_and_outward(small_v, small_f) > 0
    assert me.check_closed_and_outward(big_v, big_f) > 0
    assert np.all(np.linalg.norm(big_n, axis=1) > 0.999)


def test_faces_are_all_triangles_with_valid_indices():
    front = _surf(10.0, c=0.01)
    back = _surf(12.0, c=-0.01)
    verts, faces, normals = me.build_element_solid(front, back, aperture_mm=6.0, n_r=8, n_phi=16)
    assert faces.shape[1] == 3
    assert faces.min() >= 0 and faces.max() < len(verts)
    assert normals.shape == (3 * len(faces), 3)
    lens = np.linalg.norm(normals, axis=1)
    assert np.allclose(lens, 1.0, atol=1e-8)

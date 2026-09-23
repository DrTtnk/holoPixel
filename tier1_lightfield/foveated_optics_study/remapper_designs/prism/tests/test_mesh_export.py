"""Proofs for mesh_export.py's watertight-solid construction, independent of
whatever the optimizer converges to: build a small mesh from a fixed pose
(the validated first-order wedge, INIT in design.py) and check the exact
properties lf_evaluate.validate_surfaces requires, plus a per-face winding
check (validate_surfaces only checks the mesh-wide sum, not each triangle).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mesh_export as me  # noqa: E402
from design import Design, INIT  # noqa: E402

DTYPE = me.DTYPE


def small_design(coeff_scale=0.0):
    d = Design(requires_grad=False)
    if coeff_scale:
        rng = np.random.default_rng(0)
        d.coeff = torch.tensor(rng.uniform(-coeff_scale, coeff_scale, d.coeff.shape), dtype=DTYPE)
    return d


@pytest.mark.parametrize("coeff_scale", [0.0, 0.02])
def test_solid_is_closed_and_outward(coeff_scale):
    design = small_design(coeff_scale)
    verts, faces, loop_normals, mirror_faces = me.build_solid(design, nx=13, nu=13)

    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), f"{int(np.sum(counts != 2))} open edges"

    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    signed_vol = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c)))
    assert signed_vol > 0, "glass must be wound outwards (positive signed volume)"


@pytest.mark.parametrize("coeff_scale", [0.0, 0.02])
def test_every_face_winding_matches_its_loop_normals(coeff_scale):
    """validate_surfaces only checks the mesh-wide sum; a real renderer needs
    every individual triangle's winding to agree with its stored normals, so
    check that directly (mean cosine well above 0, no just-barely-positive
    aggregate hiding flipped faces)."""
    design = small_design(coeff_scale)
    verts, faces, loop_normals, mirror_faces = me.build_solid(design, nx=13, nu=13)
    p = verts[faces]
    flat = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    corner_normals = loop_normals.reshape(-1, 3, 3)
    cos = np.einsum("ij,ij->i", flat, corner_normals.mean(axis=1))
    assert np.all(cos > 0.05), f"{int(np.sum(cos <= 0.05))} faces wound against their own normals"


def test_loop_normals_shape_and_unit_length():
    design = small_design(0.02)
    verts, faces, loop_normals, mirror_faces = me.build_solid(design, nx=11, nu=11)
    assert loop_normals.shape == (3 * len(faces), 3)
    assert mirror_faces.shape == (len(faces),)
    norms = np.linalg.norm(loop_normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_mirror_faces_flag_only_on_s2():
    """S2 is the only mirror-coated face; every flagged triangle's vertices
    must lie exactly on the S2 surface (sag matches, to Newton precision)."""
    design = small_design(0.02)
    verts, faces, loop_normals, mirror_faces = me.build_solid(design, nx=11, nu=11)
    assert 0 < mirror_faces.sum() < len(faces)
    s2 = design.surfaces()[1]
    coated = faces[mirror_faces]
    pts = torch.tensor(verts[coated.ravel()], dtype=DTYPE)
    x_l, u_l, n_l = s2.local(pts)
    s, _, _ = s2.sag_and_grad(x_l, u_l)
    assert torch.allclose(n_l, s, atol=1e-6)


def test_panel_frame_is_a_proper_rotation():
    design = small_design(0.02)
    origin, basis = design.panel_frame(gap_mm=3.0)
    basis = basis.detach().numpy()
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(basis) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

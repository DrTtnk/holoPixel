"""Watertight mesh of the plano-convex hex MLA, in the panel frame (um).

u, v span the panel; w points from the panel towards the eye. The flat face is
w = 0 and the convex faces peak at w = centre thickness.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import mla_design as mla  # noqa: E402
import mla_mesh  # noqa: E402

SIDE, RADIUS, T_MIN, PANEL = 17.37, 75.0, 10.0, 200.0


@pytest.fixture(scope="module")
def mesh():
    return mla_mesh.build(panel_um=PANEL, side_um=SIDE, radius_um=RADIUS,
                          min_thickness_um=T_MIN, subdivisions=3)


def test_short_focus_lenses_are_defined_over_the_whole_panel():
    """A sphere of radius below the border distance used to raise: every point
    must lie within one circumradius of a lens centre."""
    short = mla_mesh.build(panel_um=PANEL, side_um=SIDE, radius_um=21.5, min_thickness_um=T_MIN,
                           subdivisions=3)
    uv = short.verts[:, :2]
    d = np.min(np.linalg.norm(uv[:, None, :] - short.centres[None, :, :], axis=2), axis=1)
    assert d.max() <= SIDE + 1e-9


def test_every_edge_is_shared_by_exactly_two_faces(mesh):
    tris = mesh.faces
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)


def test_linear_interpolation_of_a_paraboloid_overshoots_by_a_squared_over_four():
    """On an equilateral triangle of side a, the mean of the linear interpolant
    of |x|^2 exceeds the mean of |x|^2 by a^2/4. For a sphere cap z ~ -r^2/(2R)
    the mesh therefore sits a^2/(8R) below the true surface on average."""
    a, x, y = sp.symbols("a x y", positive=True)
    h = sp.sqrt(3) * a / 2
    verts = [(0, 0), (a, 0), (a / 2, h)]
    vertex_mean = sum(vx**2 + vy**2 for vx, vy in verts) / 3
    left, right = x * a / (2 * h), a - x * a / (2 * h)
    true_mean = sp.integrate(sp.integrate(x**2 + y**2, (x, left.subs(x, y), right.subs(x, y))),
                             (y, 0, h)) / (sp.sqrt(3) * a**2 / 4)
    assert sp.simplify(vertex_mean - true_mean - a**2 / 4) == 0


def test_faces_are_wound_outwards_and_enclose_the_lens_volume(mesh):
    v, f = mesh.verts, mesh.faces
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    signed_volume = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    n = 2000
    grid = (np.arange(n) + 0.5) / n * PANEL - PANEL / 2
    gu, gv = np.meshgrid(grid, grid)
    heights = mla_mesh.surface_height_um(np.column_stack([gu.ravel(), gv.ravel()]),
                                         mesh.centres, RADIUS, mesh.centre_thickness)
    exact = heights.mean() * PANEL**2
    spacing = SIDE / 3
    chord_deficit = PANEL**2 * spacing**2 / (8 * RADIUS)
    assert signed_volume == pytest.approx(exact - chord_deficit, rel=2e-4)


def test_top_vertices_lie_on_their_nearest_lens_sphere(mesh):
    top = mesh.verts[mesh.verts[:, 2] > 0]
    expected = mla_mesh.surface_height_um(top[:, :2], mesh.centres, RADIUS, mesh.centre_thickness)
    assert np.max(np.abs(top[:, 2] - expected)) < 1e-9


def test_thinnest_glass_is_at_the_hex_corners(mesh):
    top = mesh.verts[mesh.verts[:, 2] > 0]
    inside = np.all(np.abs(top[:, :2]) < PANEL / 2 - 2 * SIDE, axis=1)
    assert top[inside, 2].min() == pytest.approx(T_MIN, abs=1e-9)
    assert mesh.centre_thickness == pytest.approx(T_MIN + mla.sag_um(SIDE, RADIUS))


def test_loop_normals_are_the_exact_sphere_normals_of_the_owning_lens(mesh):
    n = mesh.loop_normals.reshape(-1, 3, 3)
    assert np.allclose(np.linalg.norm(n, axis=2), 1.0)
    top_faces = np.all(mesh.verts[mesh.faces][:, :, 2] > 0, axis=1)
    owners = mesh.face_lens[top_faces]
    duv = mesh.verts[mesh.faces[top_faces]][:, :, :2] - mesh.centres[owners][:, None, :]
    rise = np.sqrt(RADIUS**2 - np.sum(duv**2, axis=2))
    expected = np.concatenate([duv, rise[:, :, None]], axis=2) / RADIUS
    assert np.allclose(n[top_faces], expected, atol=1e-12)


def test_the_crease_is_exact_on_hex_edges(mesh):
    """Away from the panel border, every top face lies inside one lens's
    hexagon: its three corners are all at least as close to the owning centre
    as to any other centre. (At the border the mesh closes onto the straight
    panel edge, which does not follow hex edges.)"""
    corners = mesh.verts[mesh.faces]
    top = np.all(corners[:, :, 2] > 0, axis=1)
    interior = np.all(np.abs(corners[:, :, :2]) < PANEL / 2 - 2 * SIDE, axis=(1, 2))
    chosen = top & interior
    assert chosen.sum() > 1000
    p = corners[chosen][:, :, :2].reshape(-1, 2)
    d = np.linalg.norm(p[:, None, :] - mesh.centres[None, :, :], axis=2)
    owner = np.repeat(mesh.face_lens[chosen], 3)
    assert np.all(d[np.arange(len(p)), owner] <= d.min(axis=1) + 1e-9)

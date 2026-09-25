"""Variable-focal hex MLA: every lens focuses on the one flat panel, so each lens
stands on its own column of glass and neighbours meet at vertical steps. The
mesh must stay a closed, consistently wound solid."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import mla_design as mla  # noqa: E402
import mla_mesh  # noqa: E402

SIDE, T_MIN, PANEL, INDEX = 17.37, 10.0, 240.0, 1.5


def focal(u_um, v_um):
    """A steep, anisotropic, off-centre focal-length profile: 400 um at its peak to 60 um."""
    return 60.0 + 340.0 * np.exp(-((np.asarray(u_um) - 10.0) / 40.0) ** 2 - (np.asarray(v_um) / 25.0) ** 2)


@pytest.fixture(scope="module")
def mesh():
    return mla_mesh.build_variable(panel_um=PANEL, side_um=SIDE, focal_of_position=focal, index=INDEX,
                                   min_thickness_um=T_MIN, subdivisions=3)


def _directed_edge_counts(faces):
    d = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(d, axis=0, return_counts=True)
    return counts


def test_the_solid_is_closed_and_consistently_wound(mesh):
    assert np.all(_directed_edge_counts(mesh.mesh.faces) == 1)
    e = np.sort(np.concatenate([mesh.mesh.faces[:, [0, 1]], mesh.mesh.faces[:, [1, 2]],
                                mesh.mesh.faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    assert np.all(counts == 2)


def test_it_is_wound_outwards_and_has_no_degenerate_faces(mesh):
    v, f = mesh.mesh.verts, mesh.mesh.faces
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    assert np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) > 0
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    assert area.min() > 1e-9


def test_every_lens_focuses_on_the_panel(mesh):
    """Back focal distance of each plano-convex lens, f - t / n, equals the one air gap."""
    f = focal(mesh.mesh.centres[:, 0], mesh.mesh.centres[:, 1])
    np.testing.assert_allclose(mesh.focal_um, f, rtol=1e-12)
    gaps = mla.back_focal_gap_um(mesh.radius_um, INDEX, mesh.centre_height_um)
    np.testing.assert_allclose(gaps, mesh.gap_um, atol=1e-9)


def test_the_thinnest_glass_is_the_minimum_thickness(mesh):
    """The gap is chosen so that no lens is thinner than T_MIN at its hex corners."""
    corner = mesh.centre_height_um - mla.sag_um(SIDE, mesh.radius_um)
    assert corner.min() == pytest.approx(T_MIN, abs=1e-9)
    assert np.all(corner >= T_MIN - 1e-9)


def test_top_faces_lie_on_their_own_lens_sphere(mesh):
    m = mesh.mesh
    top = m.face_lens >= 0
    corners = m.verts[m.faces[top]]                                     # (T, 3, 3)
    owner = m.face_lens[top]
    duv = corners[:, :, :2] - m.centres[owner][:, None, :]
    expected = mesh.centre_height_um[owner][:, None] - mla.sag_um(np.linalg.norm(duv, axis=2),
                                                                  mesh.radius_um[owner][:, None])
    assert np.max(np.abs(corners[:, :, 2] - expected)) <= mla_mesh.MERGE_UM   # near-equal heights share a vertex


def test_steps_between_neighbours_are_vertical_walls(mesh):
    """Every face that is not a lens cap, the outer wall or the floor is a vertical wall."""
    m = mesh.mesh
    v = m.verts[m.faces]
    normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    floor = np.all(v[:, :, 2] == 0.0, axis=1)
    walls = (m.face_lens < 0) & ~floor
    assert walls.sum() > 0
    assert np.max(np.abs(normal[walls, 2])) < 1e-9


def test_a_uniform_profile_builds_the_uniform_array_away_from_the_border():
    """With one focal length there is no step: in the interior the surface is the
    one build() makes and no wall is needed. (At the panel border the lattice
    triangles cross hex edges; build() gives each vertex its nearest lens's
    height, build_variable keeps every face on its own lens and adds a wall.)"""
    uniform = mla_mesh.build_variable(panel_um=PANEL, side_um=SIDE, focal_of_position=lambda u, v: np.full_like(u, 150.0),
                                      index=INDEX, min_thickness_um=T_MIN, subdivisions=3)
    radius = mla.radius_for_focal_length_um(150.0, INDEX)
    ref = mla_mesh.build(panel_um=PANEL, side_um=SIDE, radius_um=radius, min_thickness_um=T_MIN, subdivisions=3)
    m = uniform.mesh
    interior = np.all(np.abs(m.verts[:, :2]) < PANEL / 2 - 2 * SIDE, axis=1) & (m.verts[:, 2] > 0)
    expected = mla_mesh.surface_height_um(m.verts[interior, :2], ref.centres, radius, ref.centre_thickness)
    assert np.max(np.abs(m.verts[interior, 2] - expected)) < 1e-9
    v = m.verts[m.faces]
    walls = (m.face_lens < 0) & ~np.all(v[:, :, 2] == 0.0, axis=1)
    inner_walls = walls & np.all(np.abs(v[:, :, :2]) < PANEL / 2 - 2 * SIDE, axis=(1, 2))
    assert inner_walls.sum() == 0
    assert uniform.gap_um == pytest.approx(mla.back_focal_gap_um(radius, INDEX, ref.centre_thickness), abs=1e-9)


def test_the_vertex_profile_passes_through_every_lens_vertex(mesh):
    """The smooth profile the remapper must focus on: n (f(r) - gap) at each centre."""
    c = mesh.mesh.centres
    profile = mla_mesh.variable_vertex_profile_um(c[:, 0], c[:, 1], PANEL, SIDE, focal, INDEX, T_MIN)
    np.testing.assert_allclose(profile, mesh.centre_height_um, atol=1e-9)


def test_every_wall_is_labelled_a_wall_and_only_the_floor_is_not(mesh):
    """The steps between neighbouring lenses and the outer border carry
    mla_mesh.WALL (the renderer blackens them: a black matrix and a black mount);
    only the floor, facing the panel, carries -1."""
    m = mesh.mesh
    v = m.verts[m.faces]
    normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    wall = m.face_lens == mla_mesh.WALL
    assert wall.sum() > 0 and np.max(np.abs(normal[wall, 2])) < 1e-9
    border = np.all(np.abs(v[:, :, :2]).max(axis=2) >= PANEL / 2 - 1e-9, axis=1) & (m.face_lens < 0)
    floor = np.all(v[:, :, 2] == 0.0, axis=1)
    assert border.sum() > 0 and np.all(m.face_lens[border] == mla_mesh.WALL)
    assert np.all(m.face_lens[floor] == -1) and np.all(floor[m.face_lens == -1])


def test_the_uniform_array_border_is_a_wall_too():
    radius = mla.radius_for_focal_length_um(150.0, INDEX)
    m = mla_mesh.build(panel_um=PANEL, side_um=SIDE, radius_um=radius, min_thickness_um=T_MIN, subdivisions=3)
    v = m.verts[m.faces]
    floor = np.all(v[:, :, 2] == 0.0, axis=1)
    assert np.all(m.face_lens[(m.face_lens < 0) & ~floor] == mla_mesh.WALL)
    assert np.all(m.face_lens[floor] == -1)

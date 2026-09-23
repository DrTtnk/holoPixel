"""Microlens-array prescription for the physical light-field screen model.

The panel must sit at the lenslet's front focal point, measured through the
air gap and the flat substrate, so that every pixel leaves its lenslet as one
collimated beam. The gap formula is re-derived here from ray-transfer matrices
rather than trusted.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import mla_design as mla  # noqa: E402


def _abcd_pixel_to_exit():
    """Reduced-angle ABCD from a pixel, through the air gap, the plane face,
    the glass, and out of the convex face (centre of curvature inside glass)."""
    g, t, n, R = sp.symbols("g t n R", positive=True)
    def travel(d, index):
        return sp.Matrix([[1, d / index], [0, 1]])
    def surface(power):
        return sp.Matrix([[1, 0], [-power, 1]])
    signed_radius = -R
    convex_power = (1 - n) / signed_radius
    system = surface(convex_power) * travel(t, n) * surface(0) * travel(g, 1)
    return system, (g, t, n, R)


def test_back_focal_gap_collimates_a_pixel():
    system, (g, t, n, R) = _abcd_pixel_to_exit()
    gap = sp.solve(sp.Eq(system[1, 1], 0), g)
    assert len(gap) == 1
    rng = np.random.default_rng(7)
    for _ in range(20):
        vals = {R: rng.uniform(20, 200), n: rng.uniform(1.4, 1.8), t: rng.uniform(5, 60)}
        exact = float(gap[0].subs(vals))
        assert mla.back_focal_gap_um(vals[R], vals[n], vals[t]) == pytest.approx(exact, rel=1e-12)


def test_exit_angle_per_pixel_offset_is_offset_over_focal_length():
    system, (g, t, n, R) = _abcd_pixel_to_exit()
    rng = np.random.default_rng(11)
    for _ in range(20):
        vals = {R: rng.uniform(20, 200), n: rng.uniform(1.4, 1.8), t: rng.uniform(5, 60)}
        vals[g] = mla.back_focal_gap_um(vals[R], vals[n], vals[t])
        slope = float(system[1, 0].subs(vals))
        assert -slope == pytest.approx(1.0 / mla.focal_length_um(vals[R], vals[n]), rel=1e-12)


def test_radius_round_trips_through_focal_length():
    for f in (80.0, 150.0, 400.0):
        assert mla.focal_length_um(mla.radius_for_focal_length_um(f, 1.5), 1.5) == pytest.approx(f)


def test_sag_is_the_exact_sphere_not_the_paraxial_parabola():
    R, r = 75.0, 17.37
    assert mla.sag_um(r, R) == pytest.approx(R - np.sqrt(R * R - r * r), rel=1e-15)
    assert mla.sag_um(0.0, R) == 0.0
    with pytest.raises(ValueError):
        mla.sag_um(R * 1.01, R)


def test_flat_top_lattice_matches_the_study_spacing():
    side = 17.37
    assert mla.column_pitch_um(side) == pytest.approx(26.06, abs=0.01)
    assert mla.row_pitch_um(side) == pytest.approx(30.09, abs=0.01)


def test_flat_top_lattice_has_six_equal_neighbours_and_no_overlap():
    side = 10.0
    centres = mla.hex_centres_um(panel_um=400.0, side_um=side)
    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    flat_to_flat = np.sqrt(3.0) * side
    assert d.min() == pytest.approx(flat_to_flat, rel=1e-12)
    interior = np.all(np.abs(centres) < 150.0, axis=1)
    neighbours = np.sum(np.isclose(d, flat_to_flat, rtol=1e-9), axis=1)
    assert interior.sum() > 50
    assert np.all(neighbours[interior] == 6)


def test_lattice_cell_count_matches_the_study_csv():
    centres = mla.hex_centres_um(panel_um=8176.0, side_um=17.37)
    assert len(centres) == pytest.approx(84688, rel=0.005)


def test_neighbouring_lens_surfaces_meet_on_the_shared_edge():
    side, R = 17.37, 75.0
    centres = mla.hex_centres_um(panel_um=400.0, side_um=side)
    a = centres[len(centres) // 2]
    d = np.linalg.norm(centres - a, axis=1)
    b = centres[np.argsort(d)[1]]
    midpoint = 0.5 * (a + b)
    assert mla.sag_um(np.linalg.norm(midpoint - a), R) == pytest.approx(
        mla.sag_um(np.linalg.norm(midpoint - b), R), rel=1e-12)

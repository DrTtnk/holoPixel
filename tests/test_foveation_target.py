"""Retina-matched radial foveation target for the remapper: panel radius as a
function of field eccentricity, with local focal length following the Watson
pitch falloff everywhere (no cap), scaled so the field edge meets the panel edge."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import foveation_target as ft  # noqa: E402
import screen_spec as spec  # noqa: E402
from retina_model import one_mosaic_spacing_deg, square_equivalent_pitch_deg  # noqa: E402


def test_horizontal_field_edge_lands_on_the_panel_edge():
    assert ft.panel_radius_mm(np.radians(35.0)) == pytest.approx(spec.PANEL_MM / 2, rel=1e-6)
    assert spec.PANEL_MM == pytest.approx(18.432)


def test_focal_length_follows_watson_over_the_whole_field():
    th = np.radians(np.array([0.5, 1.0, 2.0, 10.0, 25.0, 35.0]))
    p = one_mosaic_spacing_deg(np.degrees(th), "temporal")
    p0 = one_mosaic_spacing_deg(0.0, "temporal")
    assert ft.local_focal_mm(th) == pytest.approx(ft.local_focal_mm(0.0) * p0 / p, rel=1e-6)


def test_radius_is_the_integral_of_local_focal_length():
    th = np.radians(np.linspace(0, 40, 40001))
    r = ft.panel_radius_mm(th)
    numeric = np.concatenate([[0.0], np.cumsum(0.5 * (ft.local_focal_mm(th[1:]) + ft.local_focal_mm(th[:-1]))
                                                * np.diff(th))])
    assert np.max(np.abs(r - numeric)) < 1e-6


def test_inverse_mapping_round_trips():
    th = np.radians(np.linspace(0, 41.6, 500))
    assert ft.eccentricity_rad(ft.panel_radius_mm(th)) == pytest.approx(th, abs=1e-9)


def test_whole_70_by_45_field_fits_on_the_panel():
    tx, ty = np.meshgrid(np.radians(np.linspace(-35, 35, 141)), np.radians(np.linspace(-22.5, 22.5, 91)))
    x, y = ft.field_to_panel_mm(tx, ty)
    assert np.max(np.abs(x)) <= spec.PANEL_MM / 2 + 1e-9
    assert np.max(np.abs(y)) <= spec.PANEL_MM / 2 + 1e-9


def test_target_neighbour_pitch_is_lens_spacing_over_local_focal_length():
    assert ft.target_pitch_rad(0.0) == pytest.approx(spec.LENS_PITCH_UM * 1e-3 / ft.local_focal_mm(0.0), rel=1e-12)


def test_the_display_samples_the_retina_at_one_constant_factor():
    """Matched to the retina everywhere, the 36 um lenses on this panel sample the
    field at one fixed multiple of the retinal pitch: 2560 pixels over 70 deg
    cannot do better, and about 2x is what fits."""
    th = np.radians(np.array([0.0, 1.0, 5.0, 20.0, 35.0]))
    factor = ft.target_pitch_rad(th) / np.radians(one_mosaic_spacing_deg(np.degrees(th), "temporal"))
    assert factor == pytest.approx(factor[0], rel=1e-6)
    assert factor[0] == pytest.approx(ft.SAMPLING_OVER_RETINA, rel=1e-12)
    assert 1.8 < factor[0] < 2.2


def test_blur_tolerance_is_the_retina_clipped_at_the_diffraction_limit():
    """The retinal pitch, but never below the Airy radius of the design pupil
    (1.22 lambda / D, 0.58 arcmin for 4 mm at 550 nm): no optics can do better."""
    tx = np.radians(np.array([0.0, 0.5, 5.0, 20.0, 35.0]))
    tz = np.zeros_like(tx)
    retina = np.radians(square_equivalent_pitch_deg(np.degrees(tx), 0.0))
    airy = 1.22 * 0.55e-3 / spec.PUPIL_DIAMETER_MM
    assert ft.blur_tolerance_rad(tx, tz) == pytest.approx(np.maximum(retina, airy), rel=1e-12)
    assert ft.blur_tolerance_rad(0.0, 0.0) == pytest.approx(airy, rel=1e-12)     # the fovea is diffraction-limited
    assert np.all(ft.blur_tolerance_rad(tx[2:], tz[2:]) == pytest.approx(retina[2:], rel=1e-12))


def test_lenslet_focal_length_lets_the_pupil_fill_one_lens_at_the_field_edge():
    """The pupil's image under a lenslet is f D / F: at the edge (smallest F) it
    must fit one lens pitch, so f = pitch F_edge / D."""
    f = ft.lenslet_focal_um()
    edge = ft.local_focal_mm(np.radians(35.0))
    assert f * spec.PUPIL_DIAMETER_MM / edge == pytest.approx(spec.LENS_PITCH_UM, rel=1e-12)

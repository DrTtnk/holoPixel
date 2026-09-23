"""R = 6 radial foveation target for the remapper: panel radius as a function of
field eccentricity, with local focal length following the Watson pitch falloff
capped at 1/6 of its foveal value."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import foveation_target as ft  # noqa: E402
from retina_model import one_mosaic_spacing_deg  # noqa: E402


def test_horizontal_field_edge_lands_on_the_panel_edge():
    assert ft.panel_radius_mm(np.radians(35.0)) == pytest.approx(4.088, rel=1e-6)


def test_centre_to_edge_focal_length_ratio_is_six():
    assert ft.local_focal_mm(0.0) / ft.local_focal_mm(np.radians(35.0)) == pytest.approx(6.0, rel=1e-3)


def test_focal_length_follows_watson_until_the_cap():
    th = np.radians(np.array([0.5, 1.0, 2.0]))
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
    assert np.max(np.abs(x)) <= 4.088 + 1e-9
    assert np.max(np.abs(y)) <= 4.088 + 1e-9


def test_target_neighbour_pitch_is_lens_spacing_over_local_focal_length():
    assert ft.target_pitch_rad(0.0) == pytest.approx(30.09e-3 / ft.local_focal_mm(0.0), rel=1e-3)

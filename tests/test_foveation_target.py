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


def test_the_diffraction_floor_is_the_rms_radius_of_the_best_gaussian_fit_to_airy():
    """The blur metric is an RMS radius, and the Airy pattern has none (its tail
    makes the second moment diverge), so the floor is the RMS radius sqrt(2) sigma
    of the least-squares Gaussian fit to the Airy intensity. Fit it numerically."""
    from scipy.optimize import curve_fit
    from scipy.special import j1
    lam, D = ft.WAVELENGTH_MM, spec.PUPIL_DIAMETER_MM
    theta = np.linspace(1e-9, 3.0 * lam / D, 4001)                      # out to ~2.5 dark rings
    x = np.pi * D * theta / lam
    airy = (2.0 * j1(x) / x) ** 2
    weight = theta                                                       # least squares over the 2D plane
    (amp, sigma), _ = curve_fit(lambda t, a, s: a * np.exp(-t**2 / (2 * s**2)), theta, airy,
                                p0=(1.0, 0.4 * lam / D), sigma=1.0 / np.sqrt(weight))
    assert sigma / (lam / D) == pytest.approx(0.42, rel=0.03)
    assert ft.DIFFRACTION_RMS_RAD == pytest.approx(np.sqrt(2.0) * sigma, rel=0.03)


def test_blur_tolerance_is_the_retina_clipped_at_the_rms_diffraction_floor():
    """The retinal pitch, never below the diffraction floor. With a 4 mm pupil the
    floor (0.28 arcmin) is below the finest retinal pitch (0.46 arcmin), so the
    retina sets the tolerance everywhere."""
    tx = np.radians(np.array([0.0, 0.5, 5.0, 20.0, 35.0]))
    tz = np.zeros_like(tx)
    retina = np.radians(square_equivalent_pitch_deg(np.degrees(tx), 0.0))
    assert ft.blur_tolerance_rad(tx, tz) == pytest.approx(np.maximum(retina, ft.DIFFRACTION_RMS_RAD), rel=1e-12)
    assert np.all(retina > ft.DIFFRACTION_RMS_RAD)


def test_lenslet_focal_length_lets_the_pupil_fill_one_lens_at_the_field_edge():
    """The pupil's image under a lenslet is f D / F: at the edge (smallest F) it
    must fit one lens pitch, so f = pitch F_edge / D."""
    f = ft.lenslet_focal_um()
    edge = ft.local_focal_mm(np.radians(35.0))
    assert f * spec.PUPIL_DIAMETER_MM / edge == pytest.approx(spec.LENS_PITCH_UM, rel=1e-12)


def test_variable_lenslets_follow_the_local_focal_length_down_to_a_hemisphere_floor():
    """Lens at panel radius r: f = pitch F(theta(r)) / D, so the pupil fills one
    lens pitch everywhere; but a plano-convex lens cannot be steeper than a
    sphere reaching past its hex corner, so f stops at the floor."""
    import mla_design as mla
    r_mm = np.array([0.0, 1.0, 5.0, 9.216, 12.0])
    f = ft.lenslet_focal_of_radius_um(r_mm * 1e3)
    wanted = spec.LENS_PITCH_UM * ft.local_focal_mm(ft.eccentricity_rad(r_mm)) / spec.PUPIL_DIAMETER_MM
    floor = mla.focal_length_um(ft.LENSLET_MIN_RADIUS_OVER_SIDE * spec.LENS_SIDE_UM, spec.LENS_INDEX)
    assert f == pytest.approx(np.maximum(wanted, floor), rel=1e-12)
    assert f[0] == pytest.approx(spec.LENS_PITCH_UM * ft.F0_MM / spec.PUPIL_DIAMETER_MM, rel=1e-12)
    assert f[3] == pytest.approx(ft.lenslet_focal_um(), rel=1e-9)       # the field edge is not floored
    assert f[4] == pytest.approx(floor, rel=1e-12)                       # the panel corner is

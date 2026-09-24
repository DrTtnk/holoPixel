"""Retina-matched, left-right symmetric foveation target: along every azimuth the
panel radius grows as the integral of 1 / retinal pitch (temporal pitch on both
sides, superior and inferior as they are), and two constant scales stretch the
70 x 45 deg field over the whole square panel."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import foveation_target as ft  # noqa: E402
import screen_spec as spec  # noqa: E402
from retina_model import one_mosaic_spacing_deg, square_equivalent_pitch_deg  # noqa: E402

HALF = spec.PANEL_MM / 2.0


def _field_grid(n=121):
    tx, tz = np.meshgrid(np.radians(np.linspace(-35, 35, n)), np.radians(np.linspace(-22.5, 22.5, n)))
    return tx.ravel(), tz.ravel()


def test_the_field_fills_the_whole_square_panel():
    u, v = ft.field_to_panel_mm(*_field_grid())
    assert u.min() == pytest.approx(-HALF, abs=1e-4) and u.max() == pytest.approx(HALF, abs=1e-4)   # 0.1 um
    assert v.min() == pytest.approx(-HALF, abs=1e-4) and v.max() == pytest.approx(HALF, abs=1e-4)


def test_the_map_is_left_right_symmetric():
    tx, tz = _field_grid(41)
    u1, v1 = ft.field_to_panel_mm(tx, tz)
    u2, v2 = ft.field_to_panel_mm(-tx, tz)
    assert u2 == pytest.approx(-u1, abs=1e-9) and v2 == pytest.approx(v1, abs=1e-9)


def test_the_symmetric_retina_uses_the_temporal_pitch_on_both_sides():
    th = np.array([1.0, 5.0, 20.0, 35.0])
    temporal = np.radians(square_equivalent_pitch_deg(th, 0.0))
    assert ft.design_retina_pitch_rad(np.radians(th), 0.0) == pytest.approx(temporal, rel=1e-12)
    assert ft.design_retina_pitch_rad(-np.radians(th), 0.0) == pytest.approx(temporal, rel=1e-12)
    assert ft.design_retina_pitch_rad(0.0, np.radians(20.0)) == pytest.approx(
        np.radians(square_equivalent_pitch_deg(0.0, 20.0)), rel=1e-12)       # superior, as it is


def test_along_each_azimuth_the_radius_grows_as_one_over_the_retinal_pitch():
    """Unscale the panel point (u / gx, (v - v0) / gz): its distance from the fovea
    grows with eccentricity at the rate p(0) / p(theta, azimuth)."""
    for az_deg in (0.0, 30.0, 90.0, 150.0, 250.0):
        az = np.radians(az_deg)
        th = np.radians(np.array([2.0, 8.0, 15.0]))
        h = 1e-6

        def unscaled(t):
            tx, tz = np.arctan(np.tan(t) * np.cos(az)), np.arctan(np.tan(t) * np.sin(az))
            u, v = ft.field_to_panel_mm(tx, tz)
            return np.hypot(u / ft.GX_MM, (v - ft.V0_MM) / ft.GZ_MM)

        rate = (unscaled(th + h) - unscaled(th - h)) / (2 * h)
        x, y = np.degrees(th) * np.cos(az), np.degrees(th) * np.sin(az)
        p0 = one_mosaic_spacing_deg(0.0, "temporal") * np.sqrt(3.0) / 2.0
        expected = p0 / np.degrees(ft.design_retina_pitch_rad(np.radians(x), np.radians(y)))
        assert rate == pytest.approx(expected, rel=2e-3), az_deg


def test_panel_to_field_inverts_field_to_panel():
    tx, tz = _field_grid(61)
    u, v = ft.field_to_panel_mm(tx, tz)
    bx, bz = ft.panel_to_field_rad(u, v)
    # 1e-5 rad = 0.03 arcmin, 0.05 um on the panel at the edge: the inverse table's resolution
    assert bx == pytest.approx(tx, abs=1e-5) and bz == pytest.approx(tz, abs=1e-5)


def test_local_focal_length_is_the_areal_scale_of_the_map_and_sets_the_lens_pitch():
    tx, tz = np.radians(np.array([0.0, 10.0, -25.0])), np.radians(np.array([0.0, 5.0, -15.0]))
    J = ft.jacobian_mm_per_rad(tx, tz)
    assert ft.local_focal_mm(tx, tz) == pytest.approx(np.sqrt(np.abs(np.linalg.det(J))), rel=1e-12)
    assert ft.target_pitch_rad(tx, tz) == pytest.approx(spec.LENS_PITCH_UM * 1e-3 / ft.local_focal_mm(tx, tz),
                                                        rel=1e-12)


def test_the_diffraction_floor_is_the_rms_radius_of_the_best_gaussian_fit_to_airy():
    """The blur metric is an RMS radius, and the Airy pattern has none (its tail
    makes the second moment diverge), so the floor is the RMS radius sqrt(2) sigma
    of the least-squares Gaussian fit to the Airy intensity. Fit it numerically."""
    from scipy.optimize import curve_fit
    from scipy.special import j1
    lam, D = ft.WAVELENGTH_MM, spec.PUPIL_DIAMETER_MM
    theta = np.linspace(1e-9, 3.0 * lam / D, 4001)
    x = np.pi * D * theta / lam
    airy = (2.0 * j1(x) / x) ** 2
    (amp, sigma), _ = curve_fit(lambda t, a, s: a * np.exp(-t**2 / (2 * s**2)), theta, airy,
                                p0=(1.0, 0.4 * lam / D), sigma=1.0 / np.sqrt(theta))
    assert sigma / (lam / D) == pytest.approx(0.42, rel=0.03)
    assert ft.DIFFRACTION_RMS_RAD == pytest.approx(np.sqrt(2.0) * sigma, rel=0.03)


def test_blur_tolerance_is_the_retina_clipped_at_the_rms_diffraction_floor():
    tx = np.radians(np.array([0.0, 0.5, 5.0, 20.0, 35.0]))
    tz = np.zeros_like(tx)
    retina = np.radians(square_equivalent_pitch_deg(np.degrees(tx), 0.0))
    assert ft.blur_tolerance_rad(tx, tz) == pytest.approx(np.maximum(retina, ft.DIFFRACTION_RMS_RAD), rel=1e-12)
    assert np.all(retina > ft.DIFFRACTION_RMS_RAD)


def test_views_per_lens_follow_diffraction_and_the_lens_size():
    """A pixel sees a pupil cell of side d; its view is diffraction-blurred to
    0.59 lambda / d RMS, which must not exceed the retinal pitch, so
    d >= 0.59 lambda / p. The pupil's image under a lens (5 x 5 pixels at most
    along the tighter axis) caps the views too. f = pixel * F / d."""
    for tx_deg, tz_deg in ((0.0, 0.0), (0.5, 0.2), (10.0, 0.0), (30.0, 15.0)):
        tx, tz = np.radians(tx_deg), np.radians(tz_deg)
        u, v = ft.field_to_panel_mm(tx, tz)
        f = ft.lenslet_focal_um(u * 1e3, v * 1e3)
        F = ft.local_focal_mm(tx, tz)
        s = np.linalg.svd(ft.jacobian_mm_per_rad(tx, tz), compute_uv=False)
        d_diff = np.sqrt(2.0) * 0.42 * ft.WAVELENGTH_MM / ft.design_retina_pitch_rad(tx, tz)
        d_lens = spec.PUPIL_DIAMETER_MM * F / (spec.LENS_PITCH_UM / spec.PIXEL_UM * s.min())
        d = min(spec.PUPIL_DIAMETER_MM, max(d_diff, d_lens))
        assert f == pytest.approx(max(spec.PIXEL_UM * F / d, ft.LENSLET_FLOOR_UM), rel=1e-6), (tx_deg, tz_deg)
    u0, v0 = ft.field_to_panel_mm(0.0, 0.0)
    views_fovea = spec.PUPIL_DIAMETER_MM / (spec.PIXEL_UM * ft.local_focal_mm(0.0, 0.0) / ft.lenslet_focal_um(u0 * 1e3, v0 * 1e3))
    assert views_fovea < 2.0                                               # about one view at the fovea


def test_sampling_over_retina_is_reported_over_the_field():
    tx, tz = _field_grid(61)
    ratio = ft.target_pitch_rad(tx, tz) / ft.retina_pitch_rad(tx, tz)
    assert ft.SAMPLING_OVER_RETINA_RANGE == pytest.approx((ratio.min(), ratio.max()), rel=1e-2)

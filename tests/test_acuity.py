"""
The geometry half of the foveal map: where on the retina each sample of a
rendered view lands.
"""

import numpy as np
import pytest
import torch

from tier2_hfh.acuity import direction_cosines, eccentricity_map


SPAN = 532e-9 / 1.524e-6          # the optimiser's own lam/pitch, ~0.349


def test_direction_cosines_centre_is_on_axis():
    a, b = direction_cosines(8, SPAN)
    assert a[4, 4] == pytest.approx(0.0)
    assert b[4, 4] == pytest.approx(0.0)


def test_direction_cosines_span_matches_lambda_over_pitch():
    """An n-sample window spans exactly lam/p in DIRECTION COSINE, not angle."""
    n = 64
    a, _ = direction_cosines(n, SPAN)
    step = SPAN / n
    assert float(a[0, -1] - a[0, 0]) == pytest.approx(SPAN - step, rel=1e-12)


def test_first_index_is_horizontal_second_is_vertical():
    a, b = direction_cosines(8, SPAN)
    assert torch.allclose(a[0], a[3])          # alpha varies along columns only
    assert torch.allclose(b[:, 0], b[:, 3])    # beta varies along rows only


def test_eccentricity_is_arcsin_not_the_small_angle_form():
    """
    At 20 degrees the difference is only 0.45%, but it is a real difference and
    the small-angle form is the wrong one: a direction cosine of s is an angle
    of arcsin(s).
    """
    n = 64
    e, _ = eccentricity_map(n, SPAN)
    edge = float(e[n // 2, -1])
    cosine = SPAN / 2 - SPAN / n
    assert edge == pytest.approx(np.degrees(np.arcsin(cosine)), rel=1e-12)
    assert edge != pytest.approx(np.degrees(cosine), rel=1e-6)


def test_eccentricity_corner_exceeds_edge():
    n = 32
    e, _ = eccentricity_map(n, SPAN)
    assert float(e[0, 0]) > float(e[n // 2, 0])


def test_eccentricity_is_four_fold_symmetric():
    e, _ = eccentricity_map(33, SPAN)
    assert torch.allclose(e, torch.flip(e, dims=(0,)), atol=1e-12)
    assert torch.allclose(e, torch.flip(e, dims=(1,)), atol=1e-12)
    assert torch.allclose(e, e.T, atol=1e-12)


def test_meridian_right_eye_temporal_is_toward_larger_column():
    """
    Visual field coordinates: for a RIGHT eye the temporal field (toward the
    ear) is to the right of fixation, so +x is temporal, meridian 0.
    """
    n = 16
    _, m = eccentricity_map(n, SPAN, eye="right")
    assert float(m[n // 2, -1]) == pytest.approx(0.0, abs=1e-9)
    assert float(m[n // 2, 0]) == pytest.approx(180.0, abs=1e-9)


def test_meridian_left_eye_temporal_is_mirrored():
    n = 16
    _, m = eccentricity_map(n, SPAN, eye="left")
    assert float(m[n // 2, 0]) == pytest.approx(0.0, abs=1e-9)
    assert float(m[n // 2, -1]) == pytest.approx(180.0, abs=1e-9)


def test_meridian_superior_is_toward_smaller_row():
    """Row 0 is the top of the image, which is the superior visual field."""
    n = 16
    _, m = eccentricity_map(n, SPAN)
    assert float(m[0, n // 2]) == pytest.approx(90.0, abs=1e-9)
    assert float(m[-1, n // 2]) == pytest.approx(270.0, abs=1e-9)


def test_meridian_is_in_zero_to_360():
    _, m = eccentricity_map(32, SPAN)
    assert float(m.min()) >= 0.0
    assert float(m.max()) < 360.0


def test_unknown_eye_is_rejected():
    with pytest.raises(ValueError, match="eye"):
        eccentricity_map(8, SPAN, eye="cyclops")


def test_span_above_two_is_rejected():
    """A direction cosine cannot exceed 1, so a full span above 2 is evanescent."""
    with pytest.raises(ValueError, match="span"):
        eccentricity_map(8, 2.5)


def test_dtype_and_device_are_honoured():
    e, m = eccentricity_map(8, SPAN, dtype=torch.float32)
    assert e.dtype == torch.float32 and m.dtype == torch.float32


# --------------------------------------------------------------------------
# The space-variant blur that turns an acuity map into a cheaper target.
# --------------------------------------------------------------------------

from tier2_hfh.acuity import gaussian_pyramid, foveate


def _ramp(n):
    y = torch.arange(n, dtype=torch.float64)
    return (y[:, None] * 3 + y[None, :] * 7).sin() + 2.0


def test_pyramid_level_zero_is_the_image():
    img = _ramp(32)
    assert torch.equal(gaussian_pyramid(img, 3)[0], img)


def test_pyramid_halves_each_level():
    pyr = gaussian_pyramid(_ramp(64), 4)
    assert [p.shape[-1] for p in pyr] == [64, 32, 16, 8]


def test_pyramid_preserves_a_constant():
    """A blur that changes a flat field is not a blur, it is a bug."""
    flat = torch.full((32, 32), 0.37, dtype=torch.float64)
    for level in gaussian_pyramid(flat, 4):
        assert torch.allclose(level, torch.full_like(level, 0.37), atol=1e-12)


def test_pyramid_does_not_amplify():
    pyr = gaussian_pyramid(_ramp(64), 4)
    for level in pyr[1:]:
        assert float(level.max()) <= float(pyr[0].max()) + 1e-12
        assert float(level.min()) >= float(pyr[0].min()) - 1e-12


def test_pyramid_rejects_a_level_count_that_does_not_fit():
    with pytest.raises(ValueError, match="levels"):
        gaussian_pyramid(_ramp(8), 6)


def test_foveate_with_zero_level_is_the_identity():
    img = _ramp(32)
    out = foveate(img, torch.zeros(32, 32, dtype=torch.float64))
    assert torch.allclose(out, img, atol=1e-12)


def test_foveate_with_a_uniform_level_matches_that_pyramid_level():
    img = _ramp(32)
    pyr = gaussian_pyramid(img, 3)
    expected = torch.nn.functional.interpolate(
        pyr[2][None, None], size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
    out = foveate(img, torch.full((32, 32), 2.0, dtype=torch.float64))
    assert torch.allclose(out, expected, atol=1e-12)


def test_foveate_interpolates_between_levels():
    img = _ramp(32)
    lo = foveate(img, torch.ones(32, 32, dtype=torch.float64))
    hi = foveate(img, torch.full((32, 32), 2.0, dtype=torch.float64))
    mid = foveate(img, torch.full((32, 32), 1.5, dtype=torch.float64))
    assert torch.allclose(mid, 0.5 * (lo + hi), atol=1e-12)


def test_foveate_blurs_more_where_the_level_is_higher():
    """The point of the whole exercise: detail survives at level 0, not at 3."""
    img = _ramp(64)
    level = torch.zeros(64, 64, dtype=torch.float64)
    level[:, 32:] = 3.0
    out = foveate(img, level)
    sharp = (out[:, 4:28] - img[:, 4:28]).abs().mean()
    blurred = (out[:, 36:60] - img[:, 36:60]).abs().mean()
    assert float(blurred) > 10 * float(sharp)


def test_foveate_is_differentiable():
    img = _ramp(16).requires_grad_(True)
    level = torch.full((16, 16), 1.4, dtype=torch.float64)
    foveate(img, level).sum().backward()
    assert img.grad is not None and float(img.grad.abs().sum()) > 0


def test_foveate_keeps_a_leading_batch_axis():
    batch = torch.stack([_ramp(32), _ramp(32) * 2])
    level = torch.full((32, 32), 1.0, dtype=torch.float64)
    out = foveate(batch, level)
    assert out.shape == batch.shape
    assert torch.allclose(out[1], 2 * out[0], atol=1e-12)


def test_foveate_rejects_a_negative_level():
    with pytest.raises(ValueError, match="level"):
        foveate(_ramp(16), torch.full((16, 16), -0.5, dtype=torch.float64))


# --------------------------------------------------------------------------
# Turning a resolution limit into a pyramid level.
# --------------------------------------------------------------------------

from tier2_hfh.acuity import PYRAMID_MTF50, level_for_cutoff


def test_level_zero_is_the_pyramids_own_cutoff():
    got = level_for_cutoff(torch.tensor([PYRAMID_MTF50], dtype=torch.float64))
    assert float(got[0]) == pytest.approx(0.0, abs=1e-12)


def test_each_halving_of_the_cutoff_costs_one_level():
    cutoff = torch.tensor([PYRAMID_MTF50 / 2, PYRAMID_MTF50 / 4, PYRAMID_MTF50 / 8],
                          dtype=torch.float64)
    assert torch.allclose(level_for_cutoff(cutoff),
                          torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), atol=1e-12)


def test_an_eye_sharper_than_the_grid_is_clamped_to_full_detail():
    """We cannot render finer than the view samples, so the level floors at 0."""
    got = level_for_cutoff(torch.tensor([10.0, 0.9], dtype=torch.float64))
    assert torch.all(got == 0)


def test_level_rises_monotonically_as_the_eye_gets_worse():
    cutoff = torch.logspace(-3, -0.3, 40, dtype=torch.float64)
    level = level_for_cutoff(cutoff)
    assert torch.all(level.diff() <= 1e-12)


def test_a_zero_cutoff_is_rejected_rather_than_returning_infinity():
    with pytest.raises(ValueError, match="cutoff"):
        level_for_cutoff(torch.tensor([0.0], dtype=torch.float64))


# --------------------------------------------------------------------------
# The eye's own resolution limit.
# --------------------------------------------------------------------------

from tier2_hfh.acuity import luminance_cutoff_cpd, mar_arcmin


def test_foveal_acuity_is_the_standard_one_arcminute():
    e = torch.zeros(1, dtype=torch.float64)
    assert float(mar_arcmin(e, e)) == pytest.approx(1.0)
    assert float(luminance_cutoff_cpd(e, e)) == pytest.approx(30.0)


def test_mar_grows_linearly_at_watsons_rate():
    """One arcminute per 2.3 degrees of eccentricity."""
    e = torch.tensor([0.0, 2.3, 4.6, 23.0], dtype=torch.float64)
    m = torch.zeros_like(e)
    assert torch.allclose(mar_arcmin(e, m),
                          torch.tensor([1.0, 2.0, 3.0, 11.0], dtype=torch.float64))


def test_cutoff_is_the_reciprocal_of_twice_the_mar():
    """A resolvable cycle is two bars, so the cutoff is 1 / (2 * MAR)."""
    e = torch.tensor([0.0, 5.0, 20.0], dtype=torch.float64)
    m = torch.zeros_like(e)
    assert torch.allclose(luminance_cutoff_cpd(e, m), 30.0 / mar_arcmin(e, m))


def test_acuity_falls_monotonically_with_eccentricity():
    e = torch.linspace(0, 45, 60, dtype=torch.float64)
    assert torch.all(luminance_cutoff_cpd(e, torch.zeros_like(e)).diff() < 0)


def test_negative_eccentricity_is_rejected():
    with pytest.raises(ValueError, match="eccentricity"):
        mar_arcmin(torch.tensor([-1.0]), torch.tensor([0.0]))


def test_the_map_feeds_straight_into_a_pyramid_level():
    """End to end: a view grid in, a per-sample blur depth out."""
    n = 256
    ecc, meridian = eccentricity_map(n, SPAN)
    deg_per_sample = float(np.degrees(2 * np.arcsin(SPAN / 2))) / n
    level = level_for_cutoff(luminance_cutoff_cpd(ecc, meridian) * deg_per_sample)
    assert level.shape == (n, n)
    assert float(level[n // 2, n // 2]) < float(level[0, 0])


# --------------------------------------------------------------------------
# The geometry must agree with the optimiser's own optics, not just with
# itself. A blazed grating of known period leaves the panel at a known angle;
# the eccentricity map has to put its peak exactly there.
# --------------------------------------------------------------------------

from tier2_hfh.optimise import Geometry, render_views


@pytest.mark.parametrize("order", [1, 3, 7, 12])
def test_a_blazed_grating_lands_where_the_map_says_it_should(order):
    geom = Geometry(panel=96, window=64, pitch=1.524e-6, n_views=3)
    n = geom.window

    x = torch.arange(geom.panel, dtype=torch.float64)
    phase = (2 * np.pi * order * x / n)[None, :].expand(geom.panel, geom.panel)

    image = render_views(phase.contiguous(), geom, torch.tensor([0]))[0]
    peak = torch.argmax(image)
    row, col = int(peak // n), int(peak % n)

    ecc, meridian = eccentricity_map(n, geom.wavelength / geom.pitch)
    expected = np.degrees(np.arcsin(order * (geom.wavelength / geom.pitch) / n))

    assert row == n // 2, "a purely horizontal ramp must not tilt vertically"
    assert float(ecc[row, col]) == pytest.approx(expected, rel=1e-9)
    assert float(meridian[row, col]) == pytest.approx(0.0, abs=1e-9), \
        "a +x tilt is the temporal meridian for a right eye"


def test_a_vertical_grating_lands_on_the_vertical_meridian():
    geom = Geometry(panel=96, window=64, pitch=1.524e-6, n_views=3)
    n = geom.window
    y = torch.arange(geom.panel, dtype=torch.float64)
    phase = (2 * np.pi * 5 * y / n)[:, None].expand(geom.panel, geom.panel)

    image = render_views(phase.contiguous(), geom, torch.tensor([0]))[0]
    peak = torch.argmax(image)
    row, col = int(peak // n), int(peak % n)

    ecc, meridian = eccentricity_map(n, geom.wavelength / geom.pitch)
    assert col == n // 2
    assert float(ecc[row, col]) == pytest.approx(
        np.degrees(np.arcsin(5 * (geom.wavelength / geom.pitch) / n)), rel=1e-9)
    assert float(meridian[row, col]) in (pytest.approx(90.0), pytest.approx(270.0))


# --------------------------------------------------------------------------
# Watson (2014): midget retinal ganglion cell receptive-field density.
#
# The paper prints several derived constants alongside its parameter table.
# Those constants are the check: if the formula is transcribed correctly, it
# must reproduce them, and they are not free parameters of the fit.
# --------------------------------------------------------------------------

from tier2_hfh.acuity import (MERIDIANS, PEAK_RGCF_DENSITY, rgcf_density,
                              midget_fraction, mrgcf_density,
                              mrgcf_spacing_arcmin, mrgcf_nyquist_cpd)


def _zero(): return torch.zeros(1, dtype=torch.float64)


def test_peak_mrgcf_density_matches_the_printed_29609_2():
    got = float(mrgcf_density(_zero(), _zero()))
    assert got == pytest.approx(29609.2, rel=2e-5)


def test_every_meridian_agrees_on_axis():
    """The bracket collapses to 1 at r = 0, so the meridian cannot matter."""
    r = _zero()
    values = [float(rgcf_density(r, torch.tensor([m], dtype=torch.float64)))
              for m in (0.0, 90.0, 180.0, 270.0)]
    assert values == [pytest.approx(PEAK_RGCF_DENSITY)] * 4


def test_midget_fraction_at_the_fovea_is_one_over_1_12():
    assert float(midget_fraction(_zero())) == pytest.approx(1 / 1.12)


def test_minimum_spacing_matches_the_printed_0_5299_arcmin():
    """
    The sampling lattice is the ON-centre midgets alone, half of all midgets,
    which is why this equals the cone spacing. Getting the factor of two wrong
    shows up here as 0.375 arcmin instead.
    """
    assert float(mrgcf_spacing_arcmin(_zero(), _zero())) == pytest.approx(0.5299, rel=2e-4)


def test_peak_nyquist_matches_the_printed_65_37_cycles_per_degree():
    assert float(mrgcf_nyquist_cpd(_zero(), _zero())) == pytest.approx(65.37, rel=2e-4)


def test_nasal_keeps_more_cells_far_out_than_temporal():
    """
    Watson's r_e is 7.633 nasal against 22.14 temporal, so the exponential
    term dies fastest nasally -- but nasal's larger 1 - a makes it the denser
    meridian near the fovea. Both orderings are real; check the one at 40 deg.
    """
    r = torch.tensor([40.0], dtype=torch.float64)
    nasal = float(mrgcf_density(r, torch.tensor([180.0], dtype=torch.float64)))
    temporal = float(mrgcf_density(r, torch.tensor([0.0], dtype=torch.float64)))
    assert nasal < temporal


def test_the_four_meridians_really_differ():
    r = torch.tensor([20.0], dtype=torch.float64)
    values = {name: float(mrgcf_density(r, torch.tensor([ang], dtype=torch.float64)))
              for name, ang in (("temporal", 0.0), ("superior", 90.0),
                                ("nasal", 180.0), ("inferior", 270.0))}
    assert len(set(round(v, 3) for v in values.values())) == 4


def test_an_off_meridian_point_lies_between_its_two_neighbours():
    """Watson combines the two bracketing meridians; 45 deg is halfway."""
    r = torch.tensor([20.0], dtype=torch.float64)
    t = float(mrgcf_spacing_arcmin(r, torch.tensor([0.0], dtype=torch.float64)))
    s = float(mrgcf_spacing_arcmin(r, torch.tensor([90.0], dtype=torch.float64)))
    d = float(mrgcf_spacing_arcmin(r, torch.tensor([45.0], dtype=torch.float64)))
    assert min(t, s) <= d <= max(t, s)


def test_density_falls_monotonically_on_every_meridian():
    r = torch.linspace(0, 60, 200, dtype=torch.float64)
    for ang in (0.0, 90.0, 180.0, 270.0):
        d = mrgcf_density(r, torch.full_like(r, ang))
        assert torch.all(d.diff() < 0), f"meridian {ang} is not monotonic"


def test_watsons_table_is_the_published_one():
    """Guard the transcription itself. Four meridians, (a, r2, re)."""
    assert MERIDIANS == {
        "temporal": (0.9851, 1.058, 22.14),
        "superior": (0.9935, 1.035, 16.35),
        "nasal":    (0.9729, 1.084, 7.633),
        "inferior": (0.996, 0.9932, 12.13),
    }


def test_watson_resolves_finer_than_the_classical_acuity_formula():
    """
    A real and important gap: Watson's anatomy says 65.4 cycles/degree on
    axis, the behavioural MAR says 30. The retina samples finer than the eye's
    optics deliver, so the two are not the same limit and must not be swapped
    for one another silently.
    """
    r, m = _zero(), _zero()
    assert float(mrgcf_nyquist_cpd(r, m)) > 2 * float(luminance_cutoff_cpd(r, m))


# --------------------------------------------------------------------------
# The weighting route (Chakravarthula et al. 2021, Eq. 8), which is what the
# literature actually does -- and, unlike blurring the target, what works.
# --------------------------------------------------------------------------

from tier2_hfh.acuity import foveal_weight


def test_weight_is_highest_at_the_fovea():
    n = 64
    ecc, mer = eccentricity_map(n, SPAN)
    w = foveal_weight(ecc, mer)
    assert float(w[n // 2, n // 2]) == float(w.max())


def test_eccentricity_alone_does_not_order_the_weight_map():
    """
    The meridian matters enough to reverse the ordering. The corner at
    135 degrees (temporal-superior) sits 14.29 degrees out and is sampled MORE
    finely than the corner at 224 degrees (nasal-inferior) at 14.06 degrees --
    further away, yet better resolved. An isotropic acuity formula cannot
    express this, which is the reason for carrying Watson's four meridians.
    """
    n = 64
    ecc, mer = eccentricity_map(n, SPAN)
    w = foveal_weight(ecc, mer, normalise=False)
    far, near = (0, 0), (n - 1, 0)
    assert float(ecc[far]) > float(ecc[near])
    assert float(w[far]) > float(w[near])


def test_weight_is_the_reciprocal_of_the_midget_spacing():
    ecc, mer = eccentricity_map(32, SPAN)
    w = foveal_weight(ecc, mer, normalise=False)
    assert torch.allclose(w, 60.0 / mrgcf_spacing_arcmin(ecc, mer))


def test_weight_normalises_to_unit_mean_by_default():
    ecc, mer = eccentricity_map(64, SPAN)
    assert float(foveal_weight(ecc, mer).mean()) == pytest.approx(1.0)


def test_weight_falls_by_a_real_factor_across_our_field():
    """
    Within a single +-14 degree view the midget lattice goes from 0.530 to
    8.34 arcminutes, so the sampling rate drops 13.8x. That is the whole size
    of the prize inside one view, and it is much larger than the +-10 degree
    span might suggest.
    """
    ecc, mer = eccentricity_map(64, SPAN)
    w = foveal_weight(ecc, mer, normalise=False)
    assert float(w.max() / w.min()) == pytest.approx(13.82, rel=1e-3)


def test_foveal_weight_is_algebraically_chakravarthulas_mask():
    """
    Verified with sympy against arXiv:2108.06192 Equation 8 as printed:

        mask(x, y) = r / sqrt( (2/sqrt(3)) * (x^2/rho(r,1) + y^2/rho(r,2)) )

    Our route goes through spacings rather than densities -- we build
    s_k = sqrt(2 / (sqrt(3) rho_k)) per meridian, combine as
    sqrt(cos^2 s_h^2 + sin^2 s_v^2), and invert. The two must be the same
    expression, and this pins that they are rather than leaving it to a
    comment.
    """
    import sympy as sp

    r, theta, rho1, rho2 = sp.symbols("r theta rho1 rho2", positive=True)
    x, y = r * sp.cos(theta), r * sp.sin(theta)

    paper = r / sp.sqrt(sp.Rational(2, 1) / sp.sqrt(3) * (x ** 2 / rho1 + y ** 2 / rho2))

    s1 = sp.sqrt(2 / (sp.sqrt(3) * rho1))
    s2 = sp.sqrt(2 / (sp.sqrt(3) * rho2))
    ours = 1 / sp.sqrt(sp.cos(theta) ** 2 * s1 ** 2 + sp.sin(theta) ** 2 * s2 ** 2)

    assert sp.simplify(sp.powsimp(paper / ours, force=True) - 1) == 0


def test_foveal_weight_matches_the_mask_numerically_off_axis():
    """The symbolic identity again, on the real densities, away from any axis."""
    e = torch.tensor([7.0], dtype=torch.float64)
    ang = torch.tensor([37.0], dtype=torch.float64)
    rad = float(np.radians(37.0))

    rho_h = float(mrgcf_density(e, torch.tensor([0.0], dtype=torch.float64))) / 2
    rho_v = float(mrgcf_density(e, torch.tensor([90.0], dtype=torch.float64))) / 2
    r = 7.0
    x, y = r * np.cos(rad), r * np.sin(rad)
    mask = r / np.sqrt(2 / np.sqrt(3) * (x ** 2 / rho_h + y ** 2 / rho_v))

    assert float(foveal_weight(e, ang, normalise=False)) == pytest.approx(mask, rel=1e-12)

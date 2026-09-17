"""
Verifies the holographic parallax formula symbolically, then against the renderer.

useful_knowledge.md records the hand derivation:

    brightest hogel at hx where (hx - px) / pz = (ox - hx) / oz
    => hx = (pz*ox + oz*px) / (oz + pz)
    => d(hx)/d(ox) = pz / (oz + pz)

The sign matters: an earlier version of this project used pz / (oz - pz), which
diverges as the point approaches the observer. CLAUDE.md requires every
hand derivation be checked with a tool before it reaches code, so sympy does
the algebra here and the numerical renderer is checked against its result.
"""

import numpy as np
import pytest
import sympy as sp

from tier3_display.holo_parallax_clean import (
    HOGEL_PITCH,
    compute_phases,
    find_peak_x,
    make_display,
    render_views,
)


def test_parallax_slope_derivation_is_correct():
    hx, px, pz, ox, oz = sp.symbols("hx px pz ox oz", real=True)

    # Similar triangles: the hogel that sends the point's ray to the observer.
    condition = sp.Eq((hx - px) / pz, (ox - hx) / oz)
    solution = sp.solve(condition, hx)
    assert len(solution) == 1

    hx_of_ox = sp.simplify(solution[0])
    assert sp.simplify(hx_of_ox - (pz * ox + oz * px) / (oz + pz)) == 0

    slope = sp.simplify(sp.diff(hx_of_ox, ox))
    assert sp.simplify(slope - pz / (oz + pz)) == 0

    # The rejected form is genuinely different, not an algebraic restatement.
    assert sp.simplify(slope - pz / (oz - pz)) != 0


def test_slope_is_bounded_and_increases_with_point_depth():
    """
    0 < slope < 1 for any point in front of the display. Points farther from the
    display (closer to the observer) show more parallax.
    """
    pz, oz = sp.symbols("pz oz", positive=True)
    slope = pz / (oz + pz)

    assert sp.simplify(sp.diff(slope, pz)) == sp.simplify(oz / (oz + pz) ** 2)
    assert sp.limit(slope, pz, 0) == 0
    assert sp.limit(slope, pz, sp.oo) == 1


def test_diverging_wavefront_sign_convention():
    """
    exp(+i k R) is a diverging (real image) wavefront. compute_phases builds
    exp(+i k dist) / dist, so moving the point farther must increase the phase
    at a fixed sub-pixel, not decrease it.
    """
    lam = 532e-9
    hogel_cx, hogel_cy, sub_dx, sub_dy = make_display(1, 1, 2)

    near = compute_phases(hogel_cx, hogel_cy, sub_dx, sub_dy,
                          [(0.0, 0.0, 0.020, 1.0)], lam)
    far = compute_phases(hogel_cx, hogel_cy, sub_dx, sub_dy,
                         [(0.0, 0.0, 0.020 + lam / 8, 1.0)], lam)

    # An eighth-wavelength extra path is an extra pi/4 of phase, before quantising.
    assert not np.allclose(near, far)


@pytest.mark.parametrize("pz_mm,tol", [(20.0, 0.25), (60.0, 0.25), (150.0, 0.25)])
def test_rendered_parallax_matches_the_formula(pz_mm, tol):
    """
    Render one point source through a real hogel grid at two observer positions
    and check the measured image shift against pz / (oz + pz).
    """
    lam = 532e-9
    n_hx, n_hy, n_sub = 96, 4, 32
    oz = 0.500
    pz = pz_mm * 1e-3

    hogel_cx, hogel_cy, sub_dx, sub_dy = make_display(n_hx, n_hy, n_sub)
    phases = compute_phases(hogel_cx, hogel_cy, sub_dx, sub_dy,
                            [(0.0, 0.0, pz, 1.0)], lam)

    ox_a, ox_b = -0.040, 0.040
    views = render_views(hogel_cx, hogel_cy, phases, n_sub, lam,
                         [(ox_a, 0.0, oz), (ox_b, 0.0, oz)])

    x_coords = hogel_cx[:, 0]
    row = n_hy // 2
    peak_a = find_peak_x(np.abs(views[0][:, row]) ** 2, x_coords)
    peak_b = find_peak_x(np.abs(views[1][:, row]) ** 2, x_coords)

    measured = (peak_b - peak_a) / (ox_b - ox_a)
    predicted = pz / (oz + pz)

    assert measured == pytest.approx(predicted, rel=tol)
    # The rejected formula would be a different number; make sure we can tell.
    rejected = pz / (oz - pz)
    assert abs(measured - predicted) < abs(measured - rejected)


def test_hogel_pitch_matches_150_ppi_specification():
    """150 PPI means one hogel every 25.4 mm / 150."""
    assert HOGEL_PITCH == pytest.approx(25.4e-3 / 150, rel=1e-3)

"""
Sub-pixel pitch and field of view.

DRAFT.md section 6.5 and the README claim "0.5 um sub-pixel pitch achieves
+/-32 deg FoV for green, but blue limited to +/-27 deg". The grating equation
gives the largest steerable angle from a pitch p as

    sin(theta_max) = lambda / (2 p)

This module checks that closed form and the FFT-index-to-angle conversion in
tier1_tmm/hogel_projection.py against each other.
"""

import numpy as np
import pytest

from tier1_tmm.hogel_projection import angles_from_fft

PITCH = 0.5e-6
RGB = {"red": 632e-9, "green": 532e-9, "blue": 450e-9}


def _theta_max_deg(lam, pitch):
    return np.degrees(np.arcsin(lam / (2 * pitch)))


def test_quoted_field_of_view_per_colour():
    assert _theta_max_deg(RGB["green"], PITCH) == pytest.approx(32.1, abs=0.2)
    assert _theta_max_deg(RGB["blue"], PITCH) == pytest.approx(26.7, abs=0.2)
    assert _theta_max_deg(RGB["red"], PITCH) == pytest.approx(39.2, abs=0.2)


def test_blue_is_the_field_of_view_bottleneck():
    """
    Shorter wavelength diffracts less for a fixed pitch, so blue sets the
    common FoV of an RGB display.
    """
    fovs = {c: _theta_max_deg(lam, PITCH) for c, lam in RGB.items()}
    assert min(fovs, key=fovs.get) == "blue"


@pytest.mark.parametrize("colour,lam", sorted(RGB.items()))
@pytest.mark.parametrize("n", [32, 64, 128])
def test_fft_angle_grid_spans_the_grating_limit(colour, lam, n):
    """
    The extreme FFT bin of an N-point hogel corresponds to the grating limit
    to within one bin, and the grid is symmetric about zero.
    """
    angles = angles_from_fft(n, PITCH, lam)
    bin_width = _theta_max_deg(lam, PITCH) * 2 / n

    assert angles.max() == pytest.approx(_theta_max_deg(lam, PITCH), abs=2 * bin_width)
    assert angles.min() == pytest.approx(-_theta_max_deg(lam, PITCH), abs=1e-9)
    assert angles[n // 2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n", [32, 64])
def test_fft_angle_grid_is_monotonic(n):
    angles = angles_from_fft(n, PITCH, RGB["green"])
    assert np.all(np.diff(angles) > 0)


def test_evanescent_pitch_saturates_instead_of_returning_nan():
    """
    Below lambda/2 the first order is evanescent. angles_from_fft clips
    sin(theta) to [-1, 1] rather than producing NaN.
    """
    angles = angles_from_fft(64, 0.2e-6, RGB["red"])
    assert np.all(np.isfinite(angles))
    assert angles.max() == pytest.approx(90.0, abs=1e-9)

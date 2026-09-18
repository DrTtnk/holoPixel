"""
The windowed Fourier transform underneath hogel-free holography.

Everything the method produces sits on this operator, so it is checked before
anything is built on it: exactness of the round trip, and the conditioning of
the inverse, which the paper does not discuss and which turns out to decide
whether the seams between window positions are fragile.
"""

import numpy as np
import pytest

from tier2_hfh.wft import (
    WINDOWS,
    analyse,
    overlap_noise_gain,
    overlap_weight,
    synthesise,
    window_1d,
    window_2d,
)


@pytest.mark.parametrize("hop", [1, 2, 3, 4])
@pytest.mark.parametrize("name", sorted(WINDOWS))
def test_analysis_then_synthesis_returns_the_field_exactly(name, hop):
    """
    The whole method rests on this. Verified on a random complex field, not a
    convenient special case.
    """
    rng = np.random.default_rng(0)
    n, w = 24, 8
    field = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    window = window_2d(name, w)

    back = synthesise(analyse(field, window, hop), window, hop, field.shape)

    assert np.abs(back - field).max() / np.abs(field).max() < 1e-12


@pytest.mark.parametrize("name", ["hann", "blackman"])
def test_zero_endpoint_windows_cannot_tile_without_overlap(name):
    """
    Hann and Blackman are exactly zero at one endpoint. At hop == window width
    each sample is covered by exactly one frame, so those samples get no weight
    at all and are unrecoverable. This must fail loudly, not silently produce
    NaN.
    """
    w = 9
    assert window_1d(name, w)[0] == pytest.approx(0.0, abs=1e-12)

    with pytest.raises(ValueError, match="cannot be recovered"):
        overlap_noise_gain(window_1d(name, w), hop=w, n=180)


def test_hamming_is_the_tapered_window_that_survives_non_overlapping_tiling():
    """
    Of the common tapered windows only Hamming has non-zero endpoints, so only
    Hamming can tile without overlap. This is very likely why the paper chose
    it over Hann, which is the more usual default.
    """
    ends = {name: window_1d(name, 9)[0] for name in WINDOWS}

    assert ends["hamming"] == pytest.approx(0.08, abs=1e-9)
    assert ends["hann"] == pytest.approx(0.0, abs=1e-12)
    assert ends["blackman"] == pytest.approx(0.0, abs=1e-12)
    assert ends["rect"] == pytest.approx(1.0)


def test_non_overlapping_hamming_amplifies_light_field_error_twelvefold():
    """
    The paper's configuration: a 9-sample Hamming window matched to a 9x9 view
    light field. The round trip is exact, but the inverse divides by a weight
    that ranges over two orders of magnitude, so error in the light field is
    amplified at the frame seams. Exactness on clean input hides this entirely.
    """
    window = window_1d("hamming", 9)

    acc = overlap_weight(window, hop=9, n=180)
    assert acc.min() == pytest.approx(0.08 ** 2, rel=1e-9)
    assert overlap_noise_gain(window, hop=9, n=180) == pytest.approx(12.15, rel=1e-3)


@pytest.mark.parametrize("hop,expected_gain", [(9, 12.15), (5, 1.50), (3, 1.00), (1, 1.00)])
def test_overlap_removes_the_amplification(hop, expected_gain):
    """
    Overlapping the windows fixes the conditioning outright. Hop 3 reaches the
    ideal 1.0 for three times the frames; hop 5 reaches 1.5 for 1.8 times.
    This is the cheapest improvement available over the published method.
    """
    gain = overlap_noise_gain(window_1d("hamming", 9), hop=hop, n=180)

    assert gain == pytest.approx(expected_gain, rel=2e-3)


def test_rectangular_window_is_perfectly_conditioned_at_every_hop():
    """
    The rectangular window never amplifies, because its overlap weight is
    constant. It pays elsewhere: section 7 of the rendering equation measures
    it missing the Gabor bound by 17x, with -13 dB sidelobes spraying light at
    every angle. The two defects trade against each other.
    """
    for hop in (1, 3, 9):
        assert overlap_noise_gain(window_1d("rect", 9), hop=hop, n=180) == pytest.approx(1.0)


def test_hop_must_divide_the_panel():
    with pytest.raises(ValueError, match="must divide"):
        overlap_weight(window_1d("hamming", 9), hop=7, n=180)

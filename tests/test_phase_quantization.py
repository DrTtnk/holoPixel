"""
Phase quantisation efficiency.

DRAFT.md section 6.4 and the README both claim "3-bit phase (8 levels) is
sufficient: 95% diffraction efficiency". The closed form for an N-level
quantised blazed grating is

    eta_1 = ( sin(pi/N) / (pi/N) )^2

which gives 0.9496 at N=8. This module derives that closed form symbolically
with sympy, confirms it numerically with an FFT of a real staircase grating,
and then checks the two quantisers in the codebase against it.
"""

import numpy as np
import pytest
import sympy as sp

from tier1_tmm.partial_phase import quantize_phase


def _sinc_efficiency(n_levels):
    return (np.sin(np.pi / n_levels) / (np.pi / n_levels)) ** 2


def test_blazed_grating_efficiency_closed_form_derivation():
    """
    Fourier coefficient of order +1 for a staircase of N steps of height
    2*pi*m/N over a period of 1. Derived, not asserted.
    """
    n = sp.Symbol("N", integer=True, positive=True)
    m, x = sp.symbols("m x", real=True)

    # One step of the staircase: constant phase 2*pi*m/N over [m/N, (m+1)/N].
    step = sp.integrate(sp.exp(sp.I * 2 * sp.pi * m / n) * sp.exp(-sp.I * 2 * sp.pi * x),
                        (x, m / n, (m + 1) / n))

    for n_levels in (2, 3, 4, 8, 16):
        c1 = sum(complex(step.subs({n: n_levels, m: k})) for k in range(n_levels))
        eta = abs(c1) ** 2
        assert eta == pytest.approx(_sinc_efficiency(n_levels), rel=1e-9)


def test_three_bit_phase_gives_95_percent():
    """The headline DRAFT claim."""
    assert _sinc_efficiency(8) == pytest.approx(0.950, abs=0.001)
    assert _sinc_efficiency(4) == pytest.approx(0.811, abs=0.001)
    assert _sinc_efficiency(2) == pytest.approx(0.405, abs=0.001)


def _discrete_efficiency(n_levels, samples_per_level):
    """
    Exact order-1 efficiency of a staircase sampled with a finite number of
    points per step:  eta = [ sin(pi/N) / (M sin(pi/(N M))) ]^2.
    Tends to the sinc form as M -> infinity.
    """
    m = samples_per_level
    return (np.sin(np.pi / n_levels)
            / (m * np.sin(np.pi / (n_levels * m)))) ** 2


@pytest.mark.parametrize("samples_per_level", [4, 8, 64])
@pytest.mark.parametrize("n_levels", [2, 4, 8, 16, 32])
def test_numeric_blazed_grating_matches_discrete_closed_form(n_levels, samples_per_level):
    """
    Build an actual staircase blazed grating and read the +1 order off an FFT.
    The measured value must match the exact discrete formula, not merely the
    continuous sinc limit.
    """
    periods = 16
    n = periods * n_levels * samples_per_level

    x = np.arange(n) / n
    ramp = (x * periods) % 1.0
    phase = np.floor(ramp * n_levels) / n_levels * 2 * np.pi

    spectrum = np.abs(np.fft.fft(np.exp(1j * phase))) ** 2
    eta = spectrum[periods] / spectrum.sum()

    assert eta == pytest.approx(_discrete_efficiency(n_levels, samples_per_level),
                                rel=1e-9)


@pytest.mark.parametrize("n_levels", [2, 4, 8, 16, 32])
def test_discrete_efficiency_converges_to_the_sinc_limit(n_levels):
    coarse = _discrete_efficiency(n_levels, 8)
    fine = _discrete_efficiency(n_levels, 4096)
    limit = _sinc_efficiency(n_levels)

    assert abs(fine - limit) < abs(coarse - limit)
    assert fine == pytest.approx(limit, rel=1e-6)


@pytest.mark.parametrize("n_levels", [2, 4, 8])
def test_quantize_phase_produces_n_distinct_levels(n_levels):
    """
    A quantiser targeting n_levels over a full 2*pi range must emit n_levels
    physically distinct phases. Levels spaced by linspace(0, 2*pi, n) place a
    sample at both 0 and 2*pi, which are the same phase, so only n-1 survive.
    """
    rng = np.random.default_rng(0)
    phase = rng.uniform(0, 2 * np.pi, 20_000)

    quantised = quantize_phase(phase, 2 * np.pi, n_levels=n_levels)
    distinct = np.unique(np.round(np.mod(quantised, 2 * np.pi), 9))

    assert len(distinct) == n_levels


@pytest.mark.parametrize("n_levels", [2, 4, 8])
def test_quantize_phase_levels_are_evenly_spaced_modulo_two_pi(n_levels):
    rng = np.random.default_rng(1)
    phase = rng.uniform(0, 2 * np.pi, 20_000)

    quantised = quantize_phase(phase, 2 * np.pi, n_levels=n_levels)
    distinct = np.sort(np.unique(np.round(np.mod(quantised, 2 * np.pi), 9)))

    spacing = np.diff(np.append(distinct, distinct[0] + 2 * np.pi))
    assert np.allclose(spacing, 2 * np.pi / n_levels, atol=1e-6)

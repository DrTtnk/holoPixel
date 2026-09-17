"""
Locks the GTE (Gires-Tournois etalon) claims recorded in useful_knowledge.md
and docs/tier1_findings.md against the implementation in tier1_tmm/gte_definitive.py.

The claims under test:
  1. F = (1 + sqrt(R1)) / (1 - sqrt(R1)), and the DRAFT's F=14 corresponds to R1=0.75
     while the corrected requirement F>=30 corresponds to R1>=0.877.
  2. The reflected phase sweeps exactly 2*pi per free spectral range, whatever the
     finesse. The FSR expressed in refractive index is always lambda / (2 d).
  3. F is the small-signal phase sensitivity multiplier: dphi/ddelta at resonance
     is exactly -F. Derived here with sympy, not asserted.
  4. Resonance enhancement narrows the transition window as 1/F. The proportionality
     constant is derived in closed form, not fitted.

Sweeps start at an ANTI-resonance. useful_knowledge.md records why: starting at a
resonance puts the discontinuity of np.angle at the first sample and makes
np.unwrap produce a misleading curve.
"""

import numpy as np
import pytest
import sympy as sp

from tier1_tmm.gte_definitive import gte_phase, enhancement_factor


def _antiresonant_sweep(lam, d, n_points=400_001, n_hint=1.5):
    """
    One full FSR in index, starting at the anti-resonance below n_hint, so the
    resonance sits at the centre of the sweep.
    """
    fsr = lam / (2 * d)
    order = np.floor(2 * n_hint * d / lam - 0.5) + 0.5   # nearest half-integer below
    n_start = order * lam / (2 * d)
    return np.linspace(n_start, n_start + fsr, n_points), fsr


def test_enhancement_factor_matches_closed_form():
    r1 = sp.Symbol("R1", positive=True)
    closed_form = (1 + sp.sqrt(r1)) / (1 - sp.sqrt(r1))

    for value in ("0.5", "0.75", "0.877", "0.936", "0.99"):
        expected = float(closed_form.subs(r1, sp.Rational(value)))
        assert enhancement_factor(float(value)) == pytest.approx(expected, rel=1e-12)


def test_draft_finesse_corrections_are_the_quoted_reflectivities():
    """docs/tier1_findings.md quotes F=14 at R1=0.75 and F>=30 at R1>=0.877."""
    assert enhancement_factor(0.75) == pytest.approx(14.0, abs=0.5)
    assert enhancement_factor(0.877) == pytest.approx(30.0, abs=0.7)
    assert enhancement_factor(0.936) == pytest.approx(60.0, abs=1.5)


def test_finesse_is_the_small_signal_phase_sensitivity():
    """
    useful_knowledge.md: "F is the small-signal phase sensitivity multiplier".
    Differentiate the exact reflected phase symbolically and evaluate on resonance.
    """
    delta, r = sp.symbols("delta r", real=True, positive=True)
    z = sp.exp(-sp.I * delta)
    reflected = sp.expand_complex((r - z) / (1 - r * z))
    phi = sp.atan2(sp.im(reflected), sp.re(reflected))

    dphi = sp.simplify(sp.diff(phi, delta))
    assert sp.simplify(dphi - (r ** 2 - 1) / (r ** 2 - 2 * r * sp.cos(delta) + 1)) == 0

    on_resonance = sp.simplify(dphi.subs(delta, 0))
    assert sp.simplify(on_resonance + (1 + r) / (1 - r)) == 0    # i.e. -F


@pytest.mark.parametrize("r1", [0.3, 0.5, 0.75, 0.9, 0.95])
@pytest.mark.parametrize("lam,d", [(450e-9, 300e-9), (532e-9, 355e-9), (632e-9, 211e-9)])
def test_phase_sweeps_exactly_two_pi_per_fsr_regardless_of_finesse(r1, lam, d):
    """
    One FSR in index is dn = lambda / (2 d). Over that interval the unwrapped
    reflected phase must move by exactly -2*pi, independent of R1.
    """
    n, _ = _antiresonant_sweep(lam, d)
    phase = gte_phase(n, d, lam, r1)
    assert phase[-1] - phase[0] == pytest.approx(-2 * np.pi, abs=1e-3)


@pytest.mark.parametrize("r1", [0.3, 0.75, 0.95])
def test_phase_is_monotonically_decreasing_in_index(r1):
    n, _ = _antiresonant_sweep(532e-9, 355e-9)
    phase = gte_phase(n, d=355e-9, lam=532e-9, R1=r1)
    assert np.all(np.diff(phase) <= 1e-12)


def _window_10_90(r1, lam=532e-9, d=355e-9, n_points=2_000_001):
    n, fsr = _antiresonant_sweep(lam, d, n_points)
    phase = gte_phase(n, d, lam, r1)
    frac = (phase - phase[0]) / (phase[-1] - phase[0])
    lo = n[np.searchsorted(frac, 0.10)]
    hi = n[np.searchsorted(frac, 0.90)]
    return hi - lo, fsr


def test_higher_finesse_concentrates_the_transition():
    low, _ = _window_10_90(0.3)
    high, _ = _window_10_90(0.95)
    assert high < low / 15


def test_transition_width_constant_is_the_closed_form_value():
    """
    Near resonance the phase is phi = -(1+r)/sqrt(r) * arctan(sqrt(r)*u/(1-r)),
    so the 10%-90% width in delta is 2*tan(2*pi/5)*(1-r)/sqrt(r), and in index
    it is that times fsr/(2*pi). For r -> 1 this gives

        width * F / fsr  ->  2 * tan(2*pi/5) / pi  =  1.95932...

    Derive the constant with sympy, then check the implementation converges to it.
    """
    constant = sp.Rational(2, 1) * sp.tan(2 * sp.pi / 5) / sp.pi
    expected = float(constant)
    assert expected == pytest.approx(1.959324, abs=1e-5)

    for r1, tol in ((0.9, 2e-2), (0.95, 5e-3), (0.98, 1e-3), (0.99, 5e-4)):
        width, fsr = _window_10_90(r1)
        assert width * enhancement_factor(r1) / fsr == pytest.approx(expected, rel=tol)


def test_width_scales_as_one_over_finesse():
    """The scaling law itself, independent of the constant."""
    w_a, _ = _window_10_90(0.95)
    w_b, _ = _window_10_90(0.98)
    ratio_widths = w_a / w_b
    ratio_finesse = enhancement_factor(0.98) / enhancement_factor(0.95)
    assert ratio_widths == pytest.approx(ratio_finesse, rel=2e-3)

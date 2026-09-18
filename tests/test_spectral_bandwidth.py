"""
Spectral bandwidth of the Sb2Se3+DBR+AR phase LUT under laser-diode detuning.

DRAFT.md 8.4's 300nm-film phase range table shows green enhanced ~30% over
the bare double-pass value (3.23pi vs 2.48pi) while red and blue roughly
match their bare values (2.70pi/2.93pi, 2.05pi/2.09pi). The suspicion under
test: since resonance enhancement narrows spectral bandwidth roughly in
proportion to the enhancement (tests/test_gte_physics.py already proves
width ~ 1/F for a GTE), green's tolerance to a 1-2nm laser-diode linewidth
might be far worse than red's or blue's.

Measured verdict (see docs/notes_spectral_bandwidth.md for the full
writeup): the enhancement is real (pins the first test group below) but it
does *not* translate into a narrower bandwidth for green -- on the actual
device stack (DRAFT 8.5, with the 72nm MgF2 AR coat), green's pi/8
phase-error bandwidth is the *widest* of the three colours, not the
narrowest. The hypothesis is refuted for the real device.
"""

import numpy as np
import pytest

from tier2_rcwa.spectral_bandwidth import (
    RGB_NM,
    N_LEVELS,
    build_stack,
    bandwidth_report,
    design_phase_lut,
    phase_lut_error,
    find_threshold_crossing,
)


@pytest.fixture(scope="module")
def report_with_ar():
    return bandwidth_report(n_pairs_list=(4, 5, 6), dlam_max_nm=10.0, n_points=201, include_ar=True)


@pytest.fixture(scope="module")
def report_no_ar():
    return bandwidth_report(n_pairs_list=(4,), dlam_max_nm=10.0, n_points=201, include_ar=False)


# ---------------------------------------------------------------------------
# Sanity: the LUT designed at lam0 really is evenly spaced at lam0.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("colour,lam0", sorted(RGB_NM.items()))
def test_lut_is_evenly_spaced_at_its_own_design_wavelength(colour, lam0):
    n_list, d_list = build_stack(lam0, n_pairs=4, include_ar=True)
    corrected_n, _, _ = design_phase_lut(n_list, d_list, lam0, n_levels=N_LEVELS)

    err, _ = phase_lut_error(corrected_n, n_list, d_list, lam0, dlam_nm=0.0, n_levels=N_LEVELS)

    # design_phase_lut interpolates n(phase) off a 2000-point linear grid, so
    # this is exact only up to that grid's resolution -- about 1e-5 rad here,
    # four orders of magnitude below the pi/8 (~0.39 rad) tolerance that
    # actually matters for the bandwidth measurement.
    assert np.max(np.abs(err)) < 1e-4


# ---------------------------------------------------------------------------
# Resonance hypothesis: pin the DRAFT 8.4 enhancement numbers either way.
# ---------------------------------------------------------------------------

def test_green_enhancement_over_bare_double_pass_matches_draft_8_4(report_with_ar):
    """
    DRAFT 8.4 (DBR-only, no AR, 4 pairs): green 3.23pi vs bare 2.48pi is a
    genuine ~30% enhancement; blue and red are within 8% of their bare
    values. This is the mechanism the bandwidth suspicion rests on -- it
    must be real before the bandwidth consequence is even worth checking.
    """
    h = report_with_ar["hypothesis"]

    assert h["green"]["enhancement"] == pytest.approx(1.30, abs=0.02)
    assert h["blue"]["enhancement"] == pytest.approx(0.92, abs=0.02)
    assert h["red"]["enhancement"] == pytest.approx(0.98, abs=0.02)

    # Green is unambiguously the most enhanced of the three.
    assert h["green"]["enhancement"] > h["blue"]["enhancement"]
    assert h["green"]["enhancement"] > h["red"]["enhancement"]


def test_green_bandwidth_is_not_the_narrowest_on_the_real_device(report_with_ar):
    """
    The hypothesis under test, pinned either way: does green's larger
    phase-range enhancement translate into a narrower pi/8 phase-error
    bandwidth on the real DRAFT 8.5 stack (with the 72nm MgF2 AR coat)?

    Measured answer: no. Green's bandwidth is the *widest* of the three,
    not the narrowest -- the AR coating (added for amplitude-uniformity
    reasons in DRAFT 8.3) changes the cavity enough that the naive
    1/enhancement scaling from the DBR-only stack does not carry over.
    """
    sweeps = report_with_ar["sweeps"]

    def min_pi8_bandwidth(colour, n_pairs=4):
        r = sweeps[(colour, n_pairs)]
        neg, pos = find_threshold_crossing(r["dlam_nm"], r["max_err"], np.pi / 8, rising=True)
        return min(v for v in (neg, pos) if np.isfinite(v))

    bw_blue = min_pi8_bandwidth("blue")
    bw_green = min_pi8_bandwidth("green")
    bw_red = min_pi8_bandwidth("red")

    assert bw_green > bw_blue
    assert bw_green > bw_red


# ---------------------------------------------------------------------------
# The measured bandwidth numbers themselves, pinned with tolerance.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("colour,expected_min_bandwidth_nm", [
    ("blue", 3.0), ("green", 4.5), ("red", 3.0),
])
def test_measured_pi8_bandwidth_clears_a_1_2nm_diode_linewidth(
        report_with_ar, colour, expected_min_bandwidth_nm):
    """
    Every colour's pi/8 phase-error bandwidth on the real device (4-pair
    DBR + 72nm MgF2 AR) must be comfortably larger than a 1-2nm laser-diode
    linewidth, with margin to spare. This is the load-bearing number for
    the "does a plain laser diode work" verdict.
    """
    r = report_with_ar["sweeps"][(colour, 4)]
    neg, pos = find_threshold_crossing(r["dlam_nm"], r["max_err"], np.pi / 8, rising=True)
    bandwidth = min(v for v in (neg, pos) if np.isfinite(v))

    assert bandwidth >= expected_min_bandwidth_nm
    assert bandwidth > 2.0  # clears even the top of the diode linewidth range


# ---------------------------------------------------------------------------
# DBR pair-count trade: does adding pairs narrow the bandwidth as expected
# from "more pairs = higher reflectance = stronger resonance"?
# ---------------------------------------------------------------------------

def test_dbr_reflectance_rises_with_pair_count_but_bandwidth_barely_moves(report_with_ar):
    """
    Quantifies the pair-count trade DRAFT 8.5 leaves open (4-6 pairs).
    Reflectance rises substantially (this is the entire point of adding
    pairs), but because the front "mirror" here is the fixed, weak
    Sb2Se3/AR Fresnel reflection and the back DBR is already well above it
    at 4 pairs, the system is already close to the GTE limit where finesse
    is set by the front mirror alone (tests/test_gte_physics.py) --
    additional back-mirror reflectance buys little extra bandwidth
    narrowing. So "use fewer DBR pairs to fix green's bandwidth" is not a
    real fix: green's bandwidth was never the tight one, and pair count is
    not the lever that controls it.
    """
    sweeps = report_with_ar["sweeps"]

    r4 = sweeps[("green", 4)]
    r6 = sweeps[("green", 6)]

    assert r6["corrected_refl"].mean() > r4["corrected_refl"].mean() * 1.15

    def min_pi8_bandwidth(r):
        neg, pos = find_threshold_crossing(r["dlam_nm"], r["max_err"], np.pi / 8, rising=True)
        return min(v for v in (neg, pos) if np.isfinite(v))

    bw4 = min_pi8_bandwidth(r4)
    bw6 = min_pi8_bandwidth(r6)

    # Bandwidth changes by far less than reflectance does (within 20%,
    # while reflectance rose >15%) -- the pair-count lever is weak here.
    assert abs(bw6 - bw4) / bw4 < 0.20


# ---------------------------------------------------------------------------
# The AR coating's effect on bandwidth is itself worth pinning: without it,
# blue's bandwidth is close to a bare 1-2nm diode linewidth.
# ---------------------------------------------------------------------------

def test_ar_coating_widens_blue_bandwidth_substantially(report_with_ar, report_no_ar):
    r_with = report_with_ar["sweeps"][("blue", 4)]
    r_without = report_no_ar["sweeps"][("blue", 4)]

    def min_pi8_bandwidth(r):
        neg, pos = find_threshold_crossing(r["dlam_nm"], r["max_err"], np.pi / 8, rising=True)
        finite = [v for v in (neg, pos) if np.isfinite(v)]
        return min(finite) if finite else np.nan

    bw_with = min_pi8_bandwidth(r_with)
    bw_without = min_pi8_bandwidth(r_without)

    assert bw_without < 2.0   # marginal against a 1-2nm diode without the AR coat
    assert bw_with > 2 * bw_without  # the AR coat more than doubles it

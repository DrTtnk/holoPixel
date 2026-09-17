"""
Sb2Se3 switching-speed budget.

docs/notes_pcm_switching.md and tier1_tmm/pcm_switching_budget.py argue that
the "100 ns, non-volatile" Sb2Se3 spec only ever holds for the amorphization
(RESET) direction: every measured device we found crystallizes (SET) orders
of magnitude slower than it amorphizes. This module proves the two closed
forms the budget script leans on -- the uniform-random transition
probabilities and the "average case collapses onto worst case for a large
parallel array" argument -- with sympy, the way test_opa_power_budget.py
proves the PCM waveguide-loss closed form, then checks the budget script's
boundary behaviour and monotonicity.
"""

import numpy as np
import pytest
import sympy as sp

from tier1_tmm.pcm_switching_budget import (
    N_LEVELS,
    SCENARIOS,
    CHOI_SUBFRAMES,
    CHOI_PSNR_DB,
    CHOI_SSIM,
    SUBPIXELS_PER_HOGEL_SIDE,
    HOGELS_PER_SIDE,
    FANG_MEASURED_PIXEL_SIDE_M,
    FANG_SET_CURRENT_A,
    FANG_RESET_CURRENT_A,
    SUBPIXEL_PITCH_M,
    transition_probabilities,
    worst_case_frame_time,
    average_case_frame_time,
    prob_no_pixel_needs_slow_direction,
    subframes_in_budget,
    psnr_model,
    ssim_model,
    pixel_switching_energy,
    hogel_row_serial_time,
    global_serial_time,
    scale_current_by_area,
)


def test_transition_probability_closed_form_derivation_symbolic():
    """
    For (before, after) drawn independently and uniformly from n levels,
    #pairs with after > before, summed by fixing 'before'=i and counting
    the (n-1-i) values of 'after' above it, is

        sum_{i=0}^{n-1} (n-1-i) = n(n-1)/2

    out of n^2 total ordered pairs, i.e. P(after>before) = (n-1)/(2n).
    Derived here with sympy, not asserted.
    """
    n, i = sp.symbols("n i", positive=True, integer=True)
    count_gt = sp.summation(n - 1 - i, (i, 0, n - 1))
    assert sp.simplify(count_gt - n * (n - 1) / 2) == 0

    p_gt_symbolic = sp.simplify(count_gt / n ** 2)
    assert sp.simplify(p_gt_symbolic - (n - 1) / (2 * n)) == 0


@pytest.mark.parametrize("n_levels", [2, 3, 4, 8, 16, 32])
def test_transition_probabilities_match_brute_force_enumeration(n_levels):
    pairs = [(i, j) for i in range(n_levels) for j in range(n_levels)]
    n_gt = sum(1 for i, j in pairs if j > i)
    n_lt = sum(1 for i, j in pairs if j < i)
    n_eq = sum(1 for i, j in pairs if j == i)
    total = len(pairs)

    p_gt, p_lt, p_eq = transition_probabilities(n_levels)
    assert p_gt == pytest.approx(n_gt / total)
    assert p_lt == pytest.approx(n_lt / total)
    assert p_eq == pytest.approx(n_eq / total)


@pytest.mark.parametrize("n_levels", [2, 3, 4, 8, 16, 32, 256])
def test_transition_probabilities_sum_to_one(n_levels):
    p_gt, p_lt, p_eq = transition_probabilities(n_levels)
    assert p_gt + p_lt + p_eq == pytest.approx(1.0)


def test_transition_probabilities_symmetric_gt_lt():
    """Up-transitions and down-transitions are equally likely under a
    uniform draw, by symmetry of the level ordering."""
    p_gt, p_lt, _ = transition_probabilities(N_LEVELS)
    assert p_gt == p_lt


def test_eight_level_probability_is_seven_sixteenths():
    """The number the budget script actually uses: N_LEVELS=8."""
    p_gt, p_lt, p_eq = transition_probabilities(N_LEVELS)
    assert p_gt == pytest.approx(7 / 16)
    assert p_eq == pytest.approx(1 / 8)


def test_worst_case_frame_time_is_exactly_t_set():
    assert worst_case_frame_time(0.00123) == 0.00123


def test_average_case_time_is_between_reset_and_set():
    t_set, t_reset = 1.0, 1e-6
    avg = average_case_frame_time(t_set, t_reset)
    assert t_reset < avg < t_set


def test_average_case_time_matches_manual_weighted_sum():
    t_set, t_reset = 0.3, 40e-9
    p_gt, p_lt, p_eq = transition_probabilities(N_LEVELS)
    expected = p_gt * t_set + p_lt * t_reset + p_eq * 0.0
    assert average_case_frame_time(t_set, t_reset) == pytest.approx(expected)


def test_average_case_reduces_to_half_sum_when_symmetric():
    """p_gt == p_lt, so average = p_gt*(t_set+t_reset) exactly."""
    t_set, t_reset = 0.3, 40e-9
    p_gt, _, _ = transition_probabilities(N_LEVELS)
    assert average_case_frame_time(t_set, t_reset) == pytest.approx(p_gt * (t_set + t_reset))


@pytest.mark.parametrize("n_pixels", [0, 1, 10, 1000, 30_000_000_000])
def test_prob_no_pixel_needs_slow_direction_closed_form(n_pixels):
    p_gt, _, _ = transition_probabilities(N_LEVELS)
    assert prob_no_pixel_needs_slow_direction(n_pixels) == pytest.approx((1 - p_gt) ** n_pixels)


def test_prob_no_pixel_needs_slow_direction_boundary_n_zero():
    """Zero pixels: vacuously true that none of them need the slow direction."""
    assert prob_no_pixel_needs_slow_direction(0) == pytest.approx(1.0)


def test_prob_no_pixel_needs_slow_direction_is_monotonically_decreasing():
    ns = [1, 2, 5, 10, 50, 200, 1000]
    probs = [prob_no_pixel_needs_slow_direction(n) for n in ns]
    assert all(a > b for a, b in zip(probs, probs[1:]))


def test_prob_no_pixel_needs_slow_direction_vanishes_for_a_real_panel():
    """
    This is the load-bearing claim in Part 2.1: for a panel with anything
    like our 30 Gpixel count, 'worst case' and 'average case' frame time are
    the same number, because some pixel needing the slow direction is a
    certainty, not a tail risk.
    """
    n_pixels = 512 * 512 * 338 * 338
    assert prob_no_pixel_needs_slow_direction(n_pixels) < 1e-300


@pytest.mark.parametrize("frame_period,subframe_time,expected", [
    (1.0, 0.125, 8),          # exact multiple: must not round down to 7
    (1.0, 0.125 + 1e-12, 7),  # just over the exact multiple: rounds down
    (1.0, 0.125 - 1e-12, 8),  # just under: still 8 whole subframes fit
    (1.0 / 60.0, 0.30, 0),    # slower than the whole frame: zero subframes
    (1.0 / 60.0, 600e-6, 27),
])
def test_subframes_in_budget_boundary_cases(frame_period, subframe_time, expected):
    assert subframes_in_budget(frame_period, subframe_time) == expected


def test_subframes_in_budget_is_monotonically_nonincreasing_in_subframe_time():
    times = [1e-6, 1e-5, 1e-4, 1e-3]
    counts = [subframes_in_budget(1 / 60, t) for t in times]
    assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_psnr_model_reproduces_choi_measured_anchors():
    """The model must pass exactly through Choi et al.'s two measured points
    (arXiv:2205.02367): it is fit to them, not merely close to them."""
    assert psnr_model(CHOI_SUBFRAMES[0]) == pytest.approx(CHOI_PSNR_DB[0])
    assert psnr_model(CHOI_SUBFRAMES[1]) == pytest.approx(CHOI_PSNR_DB[1])
    assert ssim_model(CHOI_SUBFRAMES[0]) == pytest.approx(CHOI_SSIM[0])
    assert ssim_model(CHOI_SUBFRAMES[1]) == pytest.approx(CHOI_SSIM[1])


def test_psnr_model_is_monotonically_increasing_in_subframe_count():
    n = np.array([1, 2, 4, 8, 16, 32, 100])
    values = psnr_model(n)
    assert np.all(np.diff(values) > 0)


def test_ssim_model_is_capped_at_one():
    """SSIM has a hard physical ceiling the 2-point log-linear fit does not
    know about; the model must clip rather than report an unphysical value
    when extrapolated far past Choi et al.'s measured range."""
    assert ssim_model(1_000_000) == pytest.approx(1.0)
    assert ssim_model(CHOI_SUBFRAMES[1]) < 1.0


def test_pixel_switching_energy_reset_exceeds_set():
    """
    Physically required: RESET heats the pixel to the melting point (884 K),
    SET only to the (lower) crystallization temperature (473 K), both from
    the same room-temperature start, with the same mass and specific heat.
    """
    e_set, e_reset = pixel_switching_energy()
    assert e_reset > e_set > 0


def test_pixel_switching_energy_matches_independent_calculation():
    """Recompute Q = m*Cp*deltaT via an independent path (not calling the
    module's constants directly by name) and check the two agree, guarding
    against a silent unit error (kg vs g, K vs C) in the implementation."""
    volume_m3 = (0.5e-6) ** 2 * 300e-9
    mass_kg = 5840.0 * volume_m3
    e_set_expected = mass_kg * 150.0 * (473.0 - 300.0)
    e_reset_expected = mass_kg * 150.0 * (884.0 - 300.0)

    e_set, e_reset = pixel_switching_energy()
    assert e_set == pytest.approx(e_set_expected, rel=1e-9)
    assert e_reset == pytest.approx(e_reset_expected, rel=1e-9)


def test_pixel_switching_energy_is_picojoule_scale():
    """Sanity bound: a first-principles thermal estimate for a 0.5um x 0.5um
    x 300nm pixel must land in the pJ range, not nJ/uJ (macroscopic
    waveguide-heater scale) or fJ (implausibly small)."""
    e_set, e_reset = pixel_switching_energy()
    assert 1e-15 < e_set < 1e-9
    assert 1e-15 < e_reset < 1e-9


@pytest.mark.parametrize("key", SCENARIOS.keys())
def test_every_scenario_has_set_slower_than_reset(key):
    """The core empirical finding this whole module exists to check: in
    every measured Sb2Se3 device we found, crystallization (SET) is slower
    than amorphization (RESET). If a future scenario ever violates this, the
    'nanosecond switching' spec would actually be plausible and the verdict
    in docs/notes_pcm_switching.md would need to be revisited."""
    s = SCENARIOS[key]
    assert s["t_set"] > s["t_reset"]


def test_lawson_counterfactual_cannot_fit_a_single_frame_at_60hz_or_90hz():
    """Lawson et al. (arXiv:2111.13182) is kept only as a counterfactual for
    what OPTICAL writing would have cost us -- our actual design (DRAFT.md
    8.10: 2T1R sub-pixel, TiN heater, electrical Joule heating) is not this
    scenario. Even so, it is slower than a whole 60 Hz or 90 Hz frame
    period, let alone a subframe, which is the point of keeping it."""
    t_set = SCENARIOS["lawson_optical_counterfactual"]["t_set"]
    assert subframes_in_budget(1 / 60, t_set) == 0
    assert subframes_in_budget(1 / 90, t_set) == 0


def test_fang_is_the_primary_scenario_and_beats_lawson_counterfactual():
    """Fang et al. (arXiv:2307.12103) -- free-space, electrically switched
    by a doped-Si microheater -- is our closest architectural match per
    DRAFT.md 8.10 (2T1R + TiN heater, electrically written; free-space
    illumination is READOUT only). It must be dramatically faster than the
    optical-write counterfactual, and fast enough to clear a single 60 Hz
    frame on switching speed alone (that this is NOT the binding constraint
    is exactly the point of the row-serial-addressing tests below)."""
    fang_t_set = SCENARIOS["fang_electrical_freespace"]["t_set"]
    lawson_t_set = SCENARIOS["lawson_optical_counterfactual"]["t_set"]
    assert fang_t_set < lawson_t_set
    assert subframes_in_budget(1 / 60, fang_t_set) > 0


def test_hogel_row_serial_time_matches_manual_multiplication():
    assert hogel_row_serial_time(50e-6) == pytest.approx(338 * 50e-6)
    assert hogel_row_serial_time(50e-6, rows=100) == pytest.approx(100 * 50e-6)


def test_hogel_row_serial_time_reproduces_draft_numbers():
    """
    DRAFT.md 8.10 states '2.96 ms/row' and, at an assumed 100 ns PCM pulse,
    '29,586x margin per row'. Both numbers must fall out of
    1 s / SUBPIXELS_PER_HOGEL_SIDE, not be independently retyped constants:
    that 1 s is DRAFT's own stated 1 FPS refresh target (DRAFT.md line 19/208).
    """
    draft_row_time = 1.0 / SUBPIXELS_PER_HOGEL_SIDE
    assert draft_row_time == pytest.approx(2.96e-3, rel=5e-3)  # DRAFT rounds to 3 sig figs
    assert draft_row_time / 100e-9 == pytest.approx(29_586, rel=5e-3)


def test_draft_margin_scales_inversely_with_the_real_pulse_time():
    """
    DRAFT's 29,586x margin assumed a 100 ns pulse. Fang et al.'s measured
    SET (50 us) is exactly 500x slower than that assumption, so the
    corrected margin must be exactly 500x smaller: 29,586 / 500 ~= 59.
    """
    draft_row_time = 1.0 / SUBPIXELS_PER_HOGEL_SIDE
    draft_margin = draft_row_time / 100e-9
    fang_t_set = SCENARIOS["fang_electrical_freespace"]["t_set"]
    slowdown = fang_t_set / 100e-9
    corrected_margin = draft_row_time / fang_t_set
    assert slowdown == pytest.approx(500.0)
    assert corrected_margin == pytest.approx(draft_margin / slowdown)
    assert corrected_margin == pytest.approx(59.17, abs=0.01)


def test_global_serial_time_matches_manual_multiplication():
    total_rows = HOGELS_PER_SIDE * SUBPIXELS_PER_HOGEL_SIDE
    assert global_serial_time(50e-6) == pytest.approx(total_rows * 50e-6)


def test_global_serial_time_is_far_worse_than_hogel_parallel_time():
    """The pessimistic (no hogel-level parallelism) reading must be worse by
    exactly a factor of HOGELS_PER_SIDE, since it serialises the extra
    dimension that hogel-parallel scanning does not."""
    t_set = SCENARIOS["fang_electrical_freespace"]["t_set"]
    ratio = global_serial_time(t_set) / hogel_row_serial_time(t_set)
    assert ratio == pytest.approx(HOGELS_PER_SIDE)


@pytest.mark.parametrize("key", ["fang_electrical_freespace", "yu_wafer_electrical_pessimistic"])
def test_row_serial_time_is_monotonic_in_switching_speed(key):
    """A slower t_set must never produce a faster (smaller) row-serial time."""
    t_set = SCENARIOS[key]["t_set"]
    assert hogel_row_serial_time(t_set * 2) > hogel_row_serial_time(t_set)


def test_fang_primary_scenario_row_serial_time_exceeds_60hz_frame_budget():
    """
    The headline finding of this revision: even our best-matched, PRIMARY,
    electrically-switched scenario (Fang et al., 50 us measured SET) cannot
    complete a single un-multiplexed 8-level hogel/panel update inside one
    60 Hz frame once DRAFT.md's own row-at-a-time addressing (not an
    idealised fully-parallel array) is accounted for. This is a tighter
    constraint than raw switching speed alone (which, per
    test_fang_is_the_primary_scenario_and_beats_lawson_counterfactual,
    looks fine) and would bind even if switching were instantaneous, because
    it comes from the wiring topology (2 shared wires/pixel), not the
    material.
    """
    t_set = SCENARIOS["fang_electrical_freespace"]["t_set"]
    assert hogel_row_serial_time(t_set) > 1.0 / 60.0
    assert subframes_in_budget(1.0 / 60.0, hogel_row_serial_time(t_set)) == 0


def test_scale_current_by_area_matches_manual_ratio():
    scaled = scale_current_by_area(FANG_SET_CURRENT_A, FANG_MEASURED_PIXEL_SIDE_M, SUBPIXEL_PITCH_M)
    expected = FANG_SET_CURRENT_A * (SUBPIXEL_PITCH_M / FANG_MEASURED_PIXEL_SIDE_M) ** 2
    assert scaled == pytest.approx(expected)


def test_scale_current_by_area_is_area_ratio_not_linear_ratio():
    """A common mistake: scaling current linearly with side length instead
    of with area (side length squared). Guard against it explicitly."""
    scaled = scale_current_by_area(FANG_SET_CURRENT_A, FANG_MEASURED_PIXEL_SIDE_M, SUBPIXEL_PITCH_M)
    wrong_linear_scale = FANG_SET_CURRENT_A * (SUBPIXEL_PITCH_M / FANG_MEASURED_PIXEL_SIDE_M)
    assert scaled != pytest.approx(wrong_linear_scale)
    assert scaled < wrong_linear_scale  # area ratio (1/3600) is smaller than linear ratio (1/60)


def test_scale_current_by_area_reset_exceeds_set():
    """Physically required: Fang et al. measured more RESET current (10 mA)
    than SET current (4.1 mA) at their device scale; area-scaling by the
    same ratio must preserve that ordering."""
    set_scaled = scale_current_by_area(FANG_SET_CURRENT_A, FANG_MEASURED_PIXEL_SIDE_M)
    reset_scaled = scale_current_by_area(FANG_RESET_CURRENT_A, FANG_MEASURED_PIXEL_SIDE_M)
    assert reset_scaled > set_scaled


def test_row_current_is_milliamp_scale_not_a_delivery_bottleneck():
    """Sanity bound distinguishing the row-serial WIRING bottleneck (real,
    per the tests above) from a hypothetical CURRENT bottleneck (not real):
    a full row of 338 area-scaled pixels driven in parallel must land in the
    sub-10-mA range, comfortably within normal CMOS driver capability."""
    i_set = scale_current_by_area(FANG_SET_CURRENT_A, FANG_MEASURED_PIXEL_SIDE_M)
    row_current = i_set * SUBPIXELS_PER_HOGEL_SIDE
    assert 1e-6 < row_current < 1e-2

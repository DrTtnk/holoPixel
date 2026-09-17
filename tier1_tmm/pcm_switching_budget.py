"""
Sb2Se3 switching-speed budget: does "100 ns, non-volatile" survive contact
with the measured literature, and what does the real number cost us?

docs/research_holo_sota.md already flagged arXiv:2111.13182 (Lawson et al.)
as evidence that Sb2Se3 vitrification (RESET, amorphization) is nanosecond
but crystallization (SET) is five orders of magnitude slower. This module
turns that qualitative flag into numbers, using ONLY measured SET/RESET
times pulled from the arXiv literature (see docs/notes_pcm_switching.md for
the full paper-by-paper trace). No number here is assumed symmetric.

REVISION NOTE: an earlier version of this module used Lawson et al. as the
scenario "matching our design". That was wrong. DRAFT.md 8.10 specifies our
actual write mechanism: a 2T1R sub-pixel (select transistor + heater-driver
transistor + resistive TiN heater), "identical to Intel Optane PCM memory",
addressed row-at-a-time over 2 shared wires. Free-space laser illumination
in our design is READOUT only -- it never writes a pixel. We are an
electrically-switched device, not an optically-pumped one. Lawson (optical,
free-space write) is now kept only as a counterfactual for what optical
writing would have cost us; it does not describe our architecture.

Three scenarios, because switching time is not a material constant -- it is
dominated by thermal mass, heat-sinking, and (as of this revision) the
addressing topology itself:

  fang_electrical_freespace   PRIMARY scenario -- our closest architectural
                              match. Fang et al. 2023, arXiv:2307.12103,
                              MEASURED: free-space transmissive metasurface,
                              electrically switched by a doped-Si
                              microheater, 10 deterministic phase levels
                              spanning 2*pi (multi-level, cycled 9x). Same
                              concept as our design feature for feature:
                              free-space optics + electrical Joule heating +
                              multi-level phase, differing only in doped-Si
                              vs. TiN heater material and a metasurface
                              resonance vs. a plain thin-film stack.

  yu_wafer_electrical_pessimistic   PESSIMISTIC BOUND. Yu et al. 2026,
                              arXiv:2604.11649, "This work" row of their
                              Table 1, MEASURED: electrical (ITO Joule
                              heater), wafer-scale, waveguide (not
                              free-space) -- a less exact architectural
                              match than Fang, but the best-endurance
                              (1.4e8 binary cycles) electrically-switched
                              Sb2Se3 device in the literature, used here as
                              a conservative/what-if-we-need-more-margin
                              bound rather than the primary estimate.

  lawson_optical_counterfactual   COUNTERFACTUAL ONLY, not our design.
                              Lawson et al. 2021, arXiv:2111.13182,
                              MEASURED: optical (488 nm pump pulse), free-
                              space thin film. Kept to show what switching
                              speed WOULD have cost us if the panel were
                              optically written instead of electrically
                              heated -- i.e. exactly what our architecture
                              avoids by using a TiN heater under every
                              sub-pixel.

SET always means crystallization (amorphous -> crystalline, the slow
direction in every Sb2Se3 device we found). RESET always means
amorphization (the fast direction). Some source papers use "SET"/"RESET"
to mean the opposite of this (Yu et al. 2604.11649 label their amorphization
pulse "set" and their crystallization pulse "reset") -- values below have
been re-labelled to the crystallization/amorphization convention used
throughout this repository and docs/notes_pcm_switching.md.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

N_LEVELS = 8  # 3-bit / 8-level phase quantization, per the HoloPixel spec

SCENARIOS = {
    "fang_electrical_freespace": dict(
        label="Free-space metasurface, electrical (Fang et al. 2023) -- PRIMARY",
        source="arXiv:2307.12103",
        t_set=50e-6,       # s, crystallization pulse (measured: 6V, 50us, 30us trailing edge)
        t_reset=1.25e-6,   # s, amorphization pulse (measured: 15V, 1.25us, 8ns trailing edge)
        measured=True,
        role="primary",
    ),
    "yu_wafer_electrical_pessimistic": dict(
        label="Wafer-scale ITO heater, electrical (Yu et al. 2026) -- PESSIMISTIC BOUND",
        source="arXiv:2604.11649",
        t_set=600e-6,      # s, crystallization pulse (measured)
        t_reset=20e-6,     # s, amorphization pulse (measured)
        measured=True,
        role="bound",
    ),
    "lawson_optical_counterfactual": dict(
        label="Free-space thin film, optical (Lawson 2021) -- COUNTERFACTUAL, NOT our design",
        source="arXiv:2111.13182",
        t_set=0.30,        # s, crystallization onset (measured; reflectance
                           # still rising at the 1 s pulse duration used --
                           # this is a lower bound on the true t_set)
        t_reset=40e-9,     # s, vitrification quench time constant (measured,
                           # mean of 37-47 ns across five pulse lengths)
        measured=True,
        role="counterfactual",
    ),
}

# Fang et al. 2023 (arXiv:2307.12103) Methods: per-channel drive current for
# their 30um x 30um metasurface pixel -- SET 6V/~4.1 mA, RESET 15V/~10 mA.
# Their pixel is 3600x the area of our 0.5um sub-pixel; scaled by area ratio
# below for a rough peak-current estimate (Part 2, row-serial addressing).
# No device at our actual pixel scale exists to measure this directly.
FANG_MEASURED_PIXEL_SIDE_M = 30e-6
FANG_SET_CURRENT_A = 4.1e-3
FANG_RESET_CURRENT_A = 10e-3

# Choi et al., Time-Multiplexed Neural Holography, arXiv:2205.02367.
# Measured on a TI DLP6750, 4-bit/16-level, 1440 Hz [HW]. Only two operating
# points are reported to us: 1 subframe and 8 subframes. Everything else is
# a 2-point log-linear (dB-per-doubling) interpolation -- a MODEL, not a
# measurement -- flagged wherever it is used.
CHOI_SUBFRAMES = np.array([1, 8])
CHOI_PSNR_DB = np.array([18.33, 22.85])
CHOI_SSIM = np.array([0.652, 0.770])

# Thermal constants for the pixel-energy estimate (Part 2.4). Sb2Se3 melting
# point Tm and crystallization temperature Tc are cross-validated between two
# independent measured sources: Lawson (Tm=884 K) and Rios's Tc=200 degC
# =473 K, arXiv:2105.06010, agree with Lawson's own Tc=473 K to the kelvin.
# Specific heat Cp=0.15 J/(g K) is the value Lawson et al. used for their own
# COMSOL simulation (their ref. [40]). Density is NOT from our arXiv corpus:
# 5840 kg/m^3 is the standard crystallographic value for Pnma Sb2Se3 (e.g.
# CRC Handbook / Materials Project mp-2160); flagged separately because it
# was not independently re-measured by any paper in this survey.
T_ROOM_K = 300.0
T_MELT_K = 884.0
T_CRYST_K = 473.0
CP_J_PER_KG_K = 150.0
RHO_KG_PER_M3 = 5840.0

SUBPIXEL_PITCH_M = 0.5e-6
FILM_THICKNESS_M = 300e-9
HOGELS_PER_SIDE = 512
SUBPIXELS_PER_HOGEL_SIDE = 338
N_PIXELS_TOTAL = (HOGELS_PER_SIDE * SUBPIXELS_PER_HOGEL_SIDE) ** 2


def transition_probabilities(n_levels=N_LEVELS):
    """
    For (before, after) drawn independently and uniformly from n_levels
    discrete levels, the closed forms (proved with sympy in
    tests/test_pcm_switching.py) are:
        P(after > before) = P(after < before) = (n-1) / (2n)
        P(after == before) = 1/n
    """
    p_gt = (n_levels - 1) / (2 * n_levels)
    p_lt = p_gt
    p_eq = 1.0 / n_levels
    return p_gt, p_lt, p_eq


def worst_case_frame_time(t_set):
    """Every pixel that changes traverses the slow (SET) direction."""
    return t_set


def average_case_frame_time(t_set, t_reset, n_levels=N_LEVELS):
    """
    Mean per-pixel transition time under uniform-random level transitions.
    This is the literal "average case" the task asks for. It is NOT the
    array-wide frame time for a large parallel-addressed panel -- see
    prob_no_pixel_needs_slow_direction below for why.
    """
    p_gt, p_lt, p_eq = transition_probabilities(n_levels)
    return p_gt * t_set + p_lt * t_reset + p_eq * 0.0


def prob_no_pixel_needs_slow_direction(n_pixels, n_levels=N_LEVELS):
    """
    Probability that, across n_pixels independent uniform-random level
    transitions, NOT ONE requires the slow (SET/crystallization) direction.
    For a synchronous, parallel-addressed CMOS backplane the array cannot
    present a valid frame until every pixel that needs to switch has
    finished, so the frame time is bounded by the SLOWEST required pixel,
    not the average pixel. This collapses the "average case" onto the
    "worst case" the moment n_pixels is more than a few dozen.
    """
    p_gt, _, _ = transition_probabilities(n_levels)
    return (1.0 - p_gt) ** n_pixels


def subframes_in_budget(frame_period_s, subframe_time_s):
    return int(np.floor(frame_period_s / subframe_time_s))


def psnr_model(n_subframes):
    """2-point log2-linear fit through Choi et al.'s (1, 18.33) and
    (8, 22.85) dB measurements. A MODEL for interpolation/extrapolation,
    not itself a measured curve at other subframe counts. Not physically
    valid far outside [1, 8] -- averaging gain cannot grow forever -- so
    this is reported as-is with that caveat rather than capped at a made-up
    ceiling."""
    slope = (CHOI_PSNR_DB[1] - CHOI_PSNR_DB[0]) / np.log2(CHOI_SUBFRAMES[1] / CHOI_SUBFRAMES[0])
    return CHOI_PSNR_DB[0] + slope * np.log2(np.maximum(n_subframes, 1) / CHOI_SUBFRAMES[0])


def ssim_model(n_subframes):
    """Same 2-point log2-linear model, through Choi et al.'s SSIM pair.
    SSIM has a hard physical ceiling of 1.0 that the linear-in-log model
    does not know about -- clipped here, and any row that hits the clip is
    a sign the model is being extrapolated past where it means anything."""
    slope = (CHOI_SSIM[1] - CHOI_SSIM[0]) / np.log2(CHOI_SUBFRAMES[1] / CHOI_SUBFRAMES[0])
    raw = CHOI_SSIM[0] + slope * np.log2(np.maximum(n_subframes, 1) / CHOI_SUBFRAMES[0])
    return np.minimum(raw, 1.0)


def pixel_switching_energy():
    """
    Thermal LOWER BOUND on per-pixel SET/RESET energy: heat a
    0.5um x 0.5um x 300nm Sb2Se3 sub-pixel from room temperature to Tc
    (SET) or Tm (RESET). Excludes latent heat of the phase transition and
    any optical/electrical coupling inefficiency, both of which only make
    the real energy larger. No published device operates at this pixel
    scale (150 PPI CMOS-addressed 2D array does not exist yet, per
    docs/research_holo_sota.md section 6), so there is no measured number
    to use directly -- this first-principles estimate is what we have.
    """
    volume_m3 = SUBPIXEL_PITCH_M * SUBPIXEL_PITCH_M * FILM_THICKNESS_M
    mass_kg = RHO_KG_PER_M3 * volume_m3
    e_set = mass_kg * CP_J_PER_KG_K * (T_CRYST_K - T_ROOM_K)
    e_reset = mass_kg * CP_J_PER_KG_K * (T_MELT_K - T_ROOM_K)
    return e_set, e_reset


def hogel_row_serial_time(t_switch, rows=SUBPIXELS_PER_HOGEL_SIDE):
    """
    DRAFT.md 8.10 specifies 2 shared wires (row+col) per sub-pixel, 'Row-at-
    a-time (DRAM-style)' addressing -- exactly the crossbar/2T1R addressing
    scheme real PCM memory (Optane) uses, and for the same reason: with only
    a shared row-select and column-drive line, one physical row is active at
    a time. Within a row, all `rows`-wide pixels ARE driven in parallel by
    independent column drivers, so one row takes t_switch; a full hogel scan
    takes `rows` of those in strict sequence.

    Under hogel-level parallelism across the panel -- every hogel scanning
    its own rows independently and simultaneously with every other hogel --
    this is also the time to update every 8-level pixel on the WHOLE panel
    once. This is not an assumption invented for this script: it is the only
    reading under which DRAFT.md's own numbers are self-consistent (see
    docs/notes_pcm_switching.md and test_hogel_row_serial_time_reproduces_draft_numbers).
    """
    return rows * t_switch


def global_serial_time(t_switch, rows=HOGELS_PER_SIDE * SUBPIXELS_PER_HOGEL_SIDE):
    """
    Pessimistic alternative reading: ONE shared row-scanner for the entire
    panel (no per-hogel parallelism), so all of the panel's physical
    sub-pixel rows (512 hogels x 338 rows/hogel = 173,056) are scanned in a
    single strict sequence.
    """
    return rows * t_switch


def scale_current_by_area(measured_current_a, measured_side_m, target_side_m=SUBPIXEL_PITCH_M):
    """Scale a measured per-pixel drive current by the ratio of pixel areas.
    A rough estimate, not a measurement -- no device at our target pixel
    size exists."""
    return measured_current_a * (target_side_m / measured_side_m) ** 2


def print_switching_time_table():
    print("=" * 88)
    print("Part 2.1 -- measured SET (crystallization) / RESET (amorphization) times")
    print("=" * 88)
    header = f"{'scenario':<34}{'t_set':>12}{'t_reset':>12}{'ratio':>12}  source"
    print(header)
    for key, s in SCENARIOS.items():
        ratio = s["t_set"] / s["t_reset"]
        print(f"{key:<34}{s['t_set']:>12.3g}{s['t_reset']:>12.3g}{ratio:>12.3g}  {s['source']}")
    print()


def print_frame_rate_table():
    print("=" * 88)
    print("Part 2.1 -- max achievable frame rate for a full 8-level pixel update")
    print("=" * 88)
    header = f"{'scenario':<34}{'worst-case Hz':>16}{'avg-case Hz':>16}{'P(no SET pixel)':>18}"
    print(header)
    for key, s in SCENARIOS.items():
        t_worst = worst_case_frame_time(s["t_set"])
        t_avg = average_case_frame_time(s["t_set"], s["t_reset"])
        p_none = prob_no_pixel_needs_slow_direction(N_PIXELS_TOTAL)
        print(f"{key:<34}{1.0 / t_worst:>16.4g}{1.0 / t_avg:>16.4g}{p_none:>18.3g}")
    print()
    print("P(no SET pixel) is effectively zero for a 30 Gpixel panel: with any")
    print("real content, the worst case and the average case are the SAME number")
    print("for an array this size. The 'average case' column above is a per-pixel")
    print("mean, useful for the energy budget in Part 2.4, but it does not bound")
    print("the frame rate of a synchronous, parallel-addressed panel.")
    print()


def print_multiplexing_table():
    print("=" * 88)
    print("Part 2.2 -- time-multiplexed subframes per 60 Hz / 90 Hz frame")
    print("=" * 88)
    header = f"{'scenario':<34}{'@60Hz N':>10}{'PSNR dB':>10}{'SSIM':>8}{'@90Hz N':>10}{'PSNR dB':>10}{'SSIM':>8}"
    print(header)
    for key, s in SCENARIOS.items():
        n60 = subframes_in_budget(1.0 / 60.0, s["t_set"])
        n90 = subframes_in_budget(1.0 / 90.0, s["t_set"])
        p60, s60 = psnr_model(max(n60, 1)), ssim_model(max(n60, 1))
        p90, s90 = psnr_model(max(n90, 1)), ssim_model(max(n90, 1))
        print(f"{key:<34}{n60:>10d}{p60:>10.2f}{s60:>8.3f}{n90:>10d}{p90:>10.2f}{s90:>8.3f}")
    print()
    print("N=0 means not even ONE full 8-level frame fits in the budget; PSNR/SSIM")
    print("shown for N=0 rows use the N=1 model point as the best available floor,")
    print("i.e. even that number is optimistic.")
    print()
    print(f"Choi et al.'s own hardware (TI DLP6750, 1440 Hz) runs subframes at "
          f"{1/1440*1e6:.0f} us. fang_electrical_freespace's t_set (50 us) beats that")
    print("by ~14x -- our PRIMARY, best-matched scenario is not the blocker on raw")
    print("switching speed alone. yu_wafer_electrical_pessimistic's t_set (600 us) is")
    print("in the same ballpark as Choi's own hardware, not orders of magnitude ahead")
    print("of it. lawson_optical_counterfactual (NOT our design -- shown only to")
    print("quantify what optical writing would have cost) is catastrophically worse")
    print("than either electrically-switched scenario.")
    print()
    print("IMPORTANT: the subframe counts above assume every pixel in the array can")
    print("be addressed in parallel. Our design cannot do that -- see the row-serial")
    print("addressing analysis below, which supersedes this table.")
    print()


def print_non_volatility_table():
    print("=" * 88)
    print("Part 2.3 -- does non-volatility save the timing budget?")
    print("=" * 88)
    p_gt, p_lt, p_eq = transition_probabilities()
    f_dither = 1.0 - p_eq
    f_video_assumptions = [0.05, 0.15, 0.30]
    print(f"Intra-frame time-multiplexed dither subframes are near-independent draws")
    print(f"by design (decorrelation is the point of dithering for speckle averaging).")
    print(f"Uniform-random model: fraction of pixels changing level per dither step")
    print(f"= 1 - 1/{N_LEVELS} = {f_dither:.3f}. Non-volatility buys ~zero timing headroom")
    print(f"here: {f_dither:.0%} of pixels still need a fresh switch every subframe, and")
    print(f"per Part 2.1 even ONE pixel needing the slow direction gates the whole")
    print(f"array. Non-volatility DOES still cut energy for the ~{1-f_dither:.0%} that hold.")
    print()
    print(f"Frame-to-frame VIDEO content update is a different regime: natural scenes")
    print(f"have much higher temporal redundancy. f_video below is an ASSUMPTION, not")
    print(f"a literature-measured number -- swept for sensitivity:")
    for f in f_video_assumptions:
        print(f"  f_video={f:.2f}: {f*N_PIXELS_TOTAL/1e9:.2f} Gpixels change per video frame")
    print()


def print_row_serial_addressing_table():
    print("=" * 88)
    print("Part 2 (revised) -- row-serial addressing: the constraint that actually binds")
    print("=" * 88)
    draft_row_time = 1.0 / SUBPIXELS_PER_HOGEL_SIDE  # DRAFT's implicit "1 FPS / 338 rows"
    draft_assumed_pulse = 100e-9
    draft_margin = draft_row_time / draft_assumed_pulse
    print(f"DRAFT.md 8.10 states '2.96 ms/row' and '100 ns pulse, 29,586x margin per row'.")
    print(f"Reproduction: 1 s / {SUBPIXELS_PER_HOGEL_SIDE} rows = {draft_row_time * 1e3:.4f} ms/row "
          f"(DRAFT: 2.96 ms/row)")
    print(f"              margin at the assumed 100 ns pulse = {draft_margin:,.0f}x (DRAFT: 29,586x)")
    print("Both reproduce DRAFT's own numbers essentially exactly: DRAFT's row budget is")
    print("their stated 1 FPS refresh divided evenly across a hogel's 338 rows. That only")
    print("equals '1 FPS' for the WHOLE PANEL if every hogel scans its own 338 rows in")
    print("parallel with every other hogel -- the alternative (one shared scanner for all")
    print(f"512*338={HOGELS_PER_SIDE * SUBPIXELS_PER_HOGEL_SIDE} panel rows) would give DRAFT's own numbers a frame time of")
    print(f"{HOGELS_PER_SIDE * SUBPIXELS_PER_HOGEL_SIDE * draft_row_time:.1f} s, not 1 s. Hogel-level parallel scanning is therefore not an")
    print("assumption this script introduces -- it is the only reading under which")
    print("DRAFT's own published numbers are self-consistent.")
    print()

    header = f"{'scenario':<32}{'t_set':>10}{'row margin':>12}{'hogel-parallel Hz':>18}{'global-serial Hz':>18}"
    print(header)
    for key, s in SCENARIOS.items():
        margin = draft_row_time / s["t_set"]
        t_hogel = hogel_row_serial_time(s["t_set"])
        t_global = global_serial_time(s["t_set"])
        print(f"{key:<32}{s['t_set']:>10.2g}{margin:>12.2f}{1.0 / t_hogel:>18.3f}{1.0 / t_global:>18.5f}")
    print()
    print(f"DRAFT's 29,586x margin used the wrong (assumed, not measured) 100 ns pulse.")
    print(f"fang_electrical_freespace's real measured SET (50 us, 500x slower than that")
    print(f"assumption) cuts the margin to {draft_margin / 500:.1f}x -- still positive, but nowhere near")
    print(f"the 29,586x DRAFT reports, and note 29,586 / 500 = {draft_margin / 500:.1f} exactly, since the row")
    print(f"time is fixed and only the pulse assumption changed.")
    print()
    t_fang_hogel = hogel_row_serial_time(SCENARIOS["fang_electrical_freespace"]["t_set"])
    print(f"With the real number, one full hogel (and, under hogel-parallel scanning, the")
    print(f"WHOLE 30 Gpixel panel) takes {t_fang_hogel * 1e3:.2f} ms for a single un-multiplexed 8-level")
    print(f"update -- {'MORE' if t_fang_hogel > 1/60 else 'less'} than one 60 Hz frame period (16.67 ms) even in our best-matched,")
    print(f"PRIMARY scenario. Time-multiplexing needs MULTIPLE such passes inside one")
    print(f"frame; row-serial addressing does not have room for one, let alone several,")
    print(f"regardless of how fast the underlying material switches. This is a tighter,")
    print(f"more binding constraint than either raw switching speed (Part 2.2, which by")
    print(f"itself looked fine) or switching energy (Part 2.4, below) -- it would matter")
    print(f"even if the material switched instantaneously, because it is a wiring/topology")
    print(f"limit (2 shared wires, 5 routing tracks at 0.5um pitch per DRAFT 8.10), not a")
    print(f"materials limit.")
    print()

    i_set = scale_current_by_area(FANG_SET_CURRENT_A, FANG_MEASURED_PIXEL_SIDE_M)
    i_reset = scale_current_by_area(FANG_RESET_CURRENT_A, FANG_MEASURED_PIXEL_SIDE_M)
    row_i_set = i_set * SUBPIXELS_PER_HOGEL_SIDE
    row_i_reset = i_reset * SUBPIXELS_PER_HOGEL_SIDE
    n_hogels = HOGELS_PER_SIDE ** 2
    panel_i_set = row_i_set * n_hogels
    panel_i_reset = row_i_reset * n_hogels
    print("Is the row-serial bound actually a CURRENT limit in disguise (i.e. could we")
    print("drive more than one row at once if only we had enough current)? Scaling Fang")
    print("et al.'s measured per-pixel drive current (their 30umx30um pixel) down to our")
    print("0.5um sub-pixel by area:")
    print(f"  per-pixel:  SET {i_set * 1e6:.3f} uA,  RESET {i_reset * 1e6:.3f} uA  (area-scaled estimate, not measured)")
    print(f"  per row ({SUBPIXELS_PER_HOGEL_SIDE} pixels in parallel):  SET {row_i_set * 1e3:.3f} mA,  RESET {row_i_reset * 1e3:.3f} mA")
    print(f"  whole panel in lockstep ({n_hogels:,} hogels):  SET {panel_i_set:.1f} A,  RESET {panel_i_reset:.1f} A")
    print("A single row's current is trivial -- current delivery is NOT why addressing is")
    print("row-serial; DRAFT's own choice of only 2 shared wires per pixel is. But if")
    print("every hogel really does scan in lockstep, the AGGREGATE panel current (100s of")
    print("amps) is a real power-delivery-network design problem in its own right,")
    print("comparable to a high-end GPU's supply current -- solvable, but not free, and")
    print("separate from the per-pixel switching-energy budget in Part 2.4.")
    print()


def print_energy_table():
    print("=" * 88)
    print("Part 2.4 -- energy budget, 512x512 hogels x 338x338 sub-pixels")
    print("=" * 88)
    e_set, e_reset = pixel_switching_energy()
    print(f"N_PIXELS_TOTAL = {N_PIXELS_TOTAL:.4g} ({N_PIXELS_TOTAL/1e9:.2f} Gpixel)")
    print(f"Thermal-minimum switching energy per pixel: E_set={e_set*1e12:.2f} pJ, "
          f"E_reset={e_reset*1e12:.2f} pJ (excludes latent heat; lower bound)")
    print()
    p_gt, p_lt, p_eq = transition_probabilities()
    e_mean = p_gt * e_set + p_lt * e_reset  # per pixel that actually changes
    f_dither = 1.0 - p_eq

    header = f"{'scenario':<34}{'regime':<22}{'switches/s':>14}{'power (W)':>12}"
    print(header)
    for key, s in SCENARIOS.items():
        # Video-only: 60 Hz refresh, f_video=0.15 fraction of pixels change,
        # no time multiplexing (whether or not multiplexing is even possible
        # in this scenario is a separate question answered in Part 2.2).
        f_video = 0.15
        switches_video = f_video * N_PIXELS_TOTAL * 60.0
        power_video = switches_video * e_mean

        # Fully multiplexed: as many subframes as fit at 60 Hz assuming
        # fully-parallel addressing (Part 2.2's model), each one a
        # near-total dither redraw (f_dither). Part 2's row-serial analysis
        # shows this parallel-addressing assumption does not hold for our
        # actual 2-wire DRAM-style backplane -- these numbers are a
        # theoretical ceiling, not something reachable with the specified
        # wiring, kept here to show that EVEN IF the addressing bottleneck
        # were solved, power would be the next wall.
        n60 = max(subframes_in_budget(1.0 / 60.0, s["t_set"]), 0)
        switches_muxed = f_dither * N_PIXELS_TOTAL * n60 * 60.0
        power_muxed = switches_muxed * e_mean

        print(f"{key:<34}{'video-only 60Hz':<22}{switches_video:>14.3g}{power_video:>12.4g}")
        print(f"{key:<34}{f'muxed 60Hz x{n60} (theoretical)':<22}{switches_muxed:>14.3g}{power_muxed:>12.4g}")
    print()
    print("Any power figure above ~10 W for a headset-scale optical engine is not")
    print("survivable thermally -- flag those rows as ABSURD, not as a design point.")
    print("The 'muxed' rows are theoretical: Part 2's row-serial addressing result")
    print("shows the panel cannot actually attempt this many subframes in the first")
    print("place, so this table demonstrates a SECOND, independent way the naive")
    print("multiplexing plan fails, not the reason it fails first.")
    print()


def plot_switching_asymmetry():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = list(SCENARIOS.keys())
    t_set = [SCENARIOS[k]["t_set"] for k in labels]
    t_reset = [SCENARIOS[k]["t_reset"] for k in labels]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, t_set, width, label="SET (crystallization)", color="firebrick")
    ax.bar(x + width / 2, t_reset, width, label="RESET (amorphization)", color="steelblue")
    ax.set_yscale("log")
    ax.set_ylabel("time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.axhline(1 / 60, color="k", linestyle="--", linewidth=1, label="60 Hz frame period")
    ax.axhline(1 / 1440, color="gray", linestyle=":", linewidth=1, label="Choi DLP6750 subframe (1440 Hz)")
    ax.legend(fontsize=8)
    ax.set_title("Measured Sb2Se3 SET/RESET times vs. our display frame budget\n"
                 "(fang_electrical_freespace is our PRIMARY design match; lawson is a counterfactual)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pcm_switching_asymmetry.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'pcm_switching_asymmetry.png'}")


def plot_psnr_vs_subframes():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    n = np.logspace(0, 3, 200)
    ax.plot(n, psnr_model(n), "k-", label="2-point log-linear model")
    ax.scatter(CHOI_SUBFRAMES, CHOI_PSNR_DB, color="red", zorder=5, label="Choi et al. measured [HW]")
    for key, s in SCENARIOS.items():
        n60 = max(subframes_in_budget(1.0 / 60.0, s["t_set"]), 1)
        ax.axvline(n60, linestyle=":", alpha=0.6)
        ax.annotate(key, (n60, 19.0), fontsize=7, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("time-multiplexed subframes per displayed frame")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs. subframe count\n(scenarios overlaid at their 60Hz subframe budget)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pcm_psnr_vs_subframes.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'pcm_psnr_vs_subframes.png'}")


def main():
    print_switching_time_table()
    print_frame_rate_table()
    print_multiplexing_table()
    print_row_serial_addressing_table()
    print_non_volatility_table()
    print_energy_table()
    plot_switching_asymmetry()
    plot_psnr_vs_subframes()


if __name__ == "__main__":
    main()

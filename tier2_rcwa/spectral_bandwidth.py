"""
Spectral bandwidth of the Sb2Se3+DBR phase LUT under laser-diode detuning.

DRAFT.md 8.4 lists the phase range achieved by the full TMM stack (300nm
Sb2Se3 film + per-colour quarter-wave DBR, no AR coat) at each design
wavelength:

    blue  2.70 pi   (bare double-pass 4*pi*d*dn/lam = 2.93 pi)
    green 3.23 pi   (bare double-pass                 2.48 pi)   <- +30%
    red   2.05 pi   (bare double-pass                 2.09 pi)

Green is enhanced ~30% over the bare-film value while red and blue roughly
match it. Since resonance enhancement narrows spectral bandwidth roughly in
proportion to the enhancement (see tier1_tmm/gte_definitive.py and
tests/test_gte_physics.py: "the 10-90% transition width obeys
width*F/FSR = 2*tan(2*pi/5)/pi", i.e. width ~ 1/F), the suspicion is that
green's tolerance to a 1-2nm laser-diode linewidth is much worse than red's
or blue's.

This module tests that suspicion directly with two detuning metrics on the
*physical device* stack from DRAFT.md 8.5 (MgF2 AR + Sb2Se3 + DBR + Si
substrate), then sweeps DBR pair count (4/5/6) because more pairs raise
DBR reflectance, which raises resonance strength.

Reuses rather than reimplements:
  - tier2_rcwa.rcwa_optimization.design_phase_lut / stack_response, the
    TMM-corrected LUT machinery already validated by
    tests/test_tmm_phase_lut.py (it accepts an arbitrary n_list/d_list with
    a single `None` placeholder for the Sb2Se3 layer, so the DBR-only stack
    tested in DRAFT 8.4 and the full AR+DBR stack of DRAFT 8.5 are both just
    different n_list/d_list built with build_stack() below).
  - The Sb2Se3 constants (N_AMORPHOUS=3.0, N_CRYSTALLINE=4.1, K_SB2SE3=0.01)
    are the same numbers as tier1_tmm/actuation.py line 227, but that file
    keeps them in a function-local dict (inside pcm_analysis()), not at
    module scope, so there is nothing importable there. They are sourced
    here from tier2_rcwa.rcwa_optimization, which already centralises them
    and which the rest of the test suite (test_tmm_phase_lut.py) treats as
    the canonical values -- no literals are retyped.

Two independent bandwidth metrics, both vs. wavelength detuning dlam:

  (a) Phase LUT error. The 8 Sb2Se3 index values are designed once at lam0
      for even 2*pi/8 phase spacing. They are a physical PCM crystallinity
      state and cannot change when the illumination detunes, so the same
      corrected_n array is re-evaluated at lam0+dlam. The deviation from
      the intended even spacing, with the global piston (phase of level 0)
      subtracted off, is the LUT error. Tolerance: pi/8 max error (half an
      8-level step).

  (b) First-order diffraction efficiency. The 8-level LUT is a one-period
      blazed grating; an 8-point FFT of the complex per-level reflection
      coefficients sqrt(R_k)*exp(i*phi_k) gives the diffraction orders
      directly (same convention as tests/test_phase_quantization.py, which
      reads order +1 off spectrum[periods] for a 1-period, 1-sample-per-level
      staircase -- here periods=1 so order +1 is FFT bin 1). Reported
      bandwidth: the |dlam| at which the +1 order efficiency first drops 10%
      relative to its own peak.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from tier2_rcwa.rcwa_optimization import (
    N_AMORPHOUS,
    N_CRYSTALLINE,
    N_TIO2,
    N_SIO2,
    N_MGF2,
    design_phase_lut,
    stack_response,
)

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

RGB_NM = {"blue": 450.0, "green": 532.0, "red": 632.0}
COLOR_HEX = {"blue": "tab:blue", "green": "tab:green", "red": "tab:red"}
N_LEVELS = 8
FILM_NM = 300.0
AR_NM = 72.0
SUB_N = 1.5  # substrate index; see note in bandwidth_report() -- effect is <0.2% of phase range


def build_stack(lam0_nm, n_pairs, film_nm=FILM_NM, ar_nm=AR_NM, include_ar=True, sub_n=SUB_N):
    """
    DRAFT.md 8.5 stack: Air / [MgF2 AR] / Sb2Se3 (None placeholder) /
    n_pairs x (TiO2/SiO2 quarter-wave at lam0_nm) / substrate.

    include_ar=False reproduces DRAFT.md 8.4's DBR-only stack (used to
    pin the resonance hypothesis against the exact numbers quoted there).
    """
    d_tio2 = lam0_nm / (4 * N_TIO2)
    d_sio2 = lam0_nm / (4 * N_SIO2)

    if include_ar:
        n_list, d_list = [1.0, N_MGF2, None], [np.inf, ar_nm, film_nm]
    else:
        n_list, d_list = [1.0, None], [np.inf, film_nm]

    for _ in range(n_pairs):
        n_list.extend([N_TIO2, N_SIO2])
        d_list.extend([d_tio2, d_sio2])
    n_list.append(sub_n)
    d_list.append(np.inf)
    return n_list, d_list


def bare_double_pass_range(lam0_nm, film_nm=FILM_NM):
    """4*pi*d*dn/lam: phase range of a lossless double pass with no mirror
    reflection physics at all -- the naive linear model DRAFT 8.1 shows is
    wrong, used here only as the reference DRAFT 8.4 compares against."""
    dn = N_CRYSTALLINE - N_AMORPHOUS
    return 4 * np.pi * film_nm * dn / lam0_nm


def phase_lut_error(corrected_n, n_list, d_list, lam0_nm, dlam_nm, n_levels=N_LEVELS):
    """
    Max/RMS deviation from even 2*pi/n_levels phase spacing when the LUT
    designed at lam0_nm is illuminated at lam0_nm + dlam_nm, with the
    global piston (level-0 phase) removed.
    """
    target = np.arange(n_levels) * (2 * np.pi / n_levels)
    phase, refl = stack_response(n_list, d_list, lam0_nm + dlam_nm, corrected_n)
    delta = phase - phase[0]
    err = delta - target
    return err, refl


def first_order_efficiency(corrected_n, n_list, d_list, lam_nm, order_bin=1):
    """Absolute +1-order efficiency (fraction of incident power) from an
    8-point FFT of the complex per-level reflection coefficients."""
    phase, refl = stack_response(n_list, d_list, lam_nm, corrected_n)
    r = np.sqrt(refl) * np.exp(1j * phase)
    c = np.fft.fft(r) / len(r)
    return np.abs(c[order_bin]) ** 2, refl


def sweep_colour(colour, lam0_nm, n_pairs, dlam_range_nm, include_ar=True, n_levels=N_LEVELS):
    """Run both metrics vs detuning for one colour / DBR pair count."""
    n_list, d_list = build_stack(lam0_nm, n_pairs, include_ar=include_ar)
    corrected_n, corrected_refl, total_range = design_phase_lut(
        n_list, d_list, lam0_nm, n_levels=n_levels)

    max_err = np.empty_like(dlam_range_nm)
    rms_err = np.empty_like(dlam_range_nm)
    eta1 = np.empty_like(dlam_range_nm)

    for i, dlam in enumerate(dlam_range_nm):
        err, _ = phase_lut_error(corrected_n, n_list, d_list, lam0_nm, dlam, n_levels)
        max_err[i] = np.max(np.abs(err))
        rms_err[i] = np.sqrt(np.mean(err ** 2))
        eta1[i], _ = first_order_efficiency(corrected_n, n_list, d_list, lam0_nm + dlam)

    return {
        "colour": colour, "lam0_nm": lam0_nm, "n_pairs": n_pairs,
        "dlam_nm": dlam_range_nm, "max_err": max_err, "rms_err": rms_err,
        "eta1": eta1, "total_range": total_range, "corrected_n": corrected_n,
        "corrected_refl": corrected_refl,
    }


def find_threshold_crossing(dlam_nm, metric, threshold, rising=True):
    """
    First |dlam| (scanning outward from 0 in each direction) at which
    `metric` crosses `threshold`. dlam_nm must be sorted and include 0.
    Returns (neg_crossing, pos_crossing), each nan if never crossed.
    """
    zero_idx = np.searchsorted(dlam_nm, 0.0)

    def scan(indices):
        for idx in indices:
            crossed = metric[idx] >= threshold if rising else metric[idx] <= threshold
            if crossed:
                return abs(dlam_nm[idx])
        return np.nan

    pos = scan(range(zero_idx, len(dlam_nm)))
    neg = scan(range(zero_idx, -1, -1))
    return neg, pos


def bandwidth_report(n_pairs_list=(4, 5, 6), dlam_max_nm=10.0, n_points=401, include_ar=True):
    """
    Compute both metrics for all three colours and DBR pair counts, and the
    resonance-hypothesis comparison (DRAFT 8.4 DBR-only stack vs bare
    double pass). Returns a results dict; does not print or plot.

    include_ar controls the stack used for the *bandwidth sweep*
    (results["sweeps"]) -- True is the real DRAFT 8.5 device (MgF2 AR +
    Sb2Se3 + DBR). The hypothesis comparison (results["hypothesis"]) always
    uses the DBR-only stack because that is what DRAFT 8.4's quoted numbers
    are built from, regardless of include_ar.
    """
    dlam_range = np.linspace(-dlam_max_nm, dlam_max_nm, n_points)

    results = {"sweeps": {}, "hypothesis": {}}

    # Resonance hypothesis: reproduce DRAFT 8.4 exactly (DBR-only, 4 pairs, no AR).
    for colour, lam0 in RGB_NM.items():
        n_list, d_list = build_stack(lam0, n_pairs=4, include_ar=False)
        _, _, total_range = design_phase_lut(n_list, d_list, lam0, n_levels=N_LEVELS)
        bare = bare_double_pass_range(lam0)
        results["hypothesis"][colour] = {
            "stack_range_pi": total_range / np.pi,
            "bare_range_pi": bare / np.pi,
            "enhancement": total_range / bare,
        }

    # Bandwidth sweep over DBR pair count, on the requested stack.
    for n_pairs in n_pairs_list:
        for colour, lam0 in RGB_NM.items():
            key = (colour, n_pairs)
            results["sweeps"][key] = sweep_colour(
                colour, lam0, n_pairs, dlam_range, include_ar=include_ar)

    return results


def print_report(results):
    print("=" * 78)
    print("RESONANCE HYPOTHESIS: DRAFT 8.4 stack (DBR-only, 4 pairs, no AR) vs bare double pass")
    print("=" * 78)
    print(f"{'colour':<8}{'stack (pi)':<14}{'bare (pi)':<14}{'enhancement':<14}")
    for colour, h in results["hypothesis"].items():
        print(f"{colour:<8}{h['stack_range_pi']:<14.3f}{h['bare_range_pi']:<14.3f}{h['enhancement']:<14.3f}")

    print()
    print("=" * 78)
    print("BANDWIDTH: full device stack (MgF2 AR + Sb2Se3 + DBR), pi/8 phase-error tolerance")
    print("=" * 78)
    print(f"{'colour':<8}{'N_pairs':<9}{'range(pi)':<11}{'-dlam(nm)':<12}{'+dlam(nm)':<12}"
          f"{'eta1-10%(-)':<13}{'eta1-10%(+)':<13}")
    for (colour, n_pairs), r in sorted(results["sweeps"].items(), key=lambda kv: (kv[0][0], kv[0][1])):
        neg_p, pos_p = find_threshold_crossing(r["dlam_nm"], r["max_err"], np.pi / 8, rising=True)
        eta1_peak = r["eta1"].max()
        neg_e, pos_e = find_threshold_crossing(r["dlam_nm"], r["eta1"], 0.9 * eta1_peak, rising=False)
        print(f"{colour:<8}{n_pairs:<9}{r['total_range']/np.pi:<11.3f}"
              f"{neg_p:<12.2f}{pos_p:<12.2f}{neg_e:<13.2f}{pos_e:<13.2f}")


def plot_bandwidth(results, n_pairs_for_main=4, path=PLOT_DIR / "spectral_bandwidth.png"):
    """Phase error and +1-order efficiency vs detuning, 3 colours, with
    +-1nm and +-2nm diode-linewidth bands marked. Also a small pair-count
    comparison panel."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax_phase, ax_eff = axes[0]
    ax_phase_pairs, ax_eff_pairs = axes[1]

    for colour in RGB_NM:
        r = results["sweeps"][(colour, n_pairs_for_main)]
        ax_phase.plot(r["dlam_nm"], r["max_err"] / np.pi, color=COLOR_HEX[colour],
                      linewidth=2, label=f"{colour} (lam0={r['lam0_nm']:.0f}nm)")
        ax_eff.plot(r["dlam_nm"], r["eta1"], color=COLOR_HEX[colour], linewidth=2,
                    label=colour)

    ax_phase.axhline(1 / 8, color="k", linestyle="--", alpha=0.6, label="pi/8 tolerance")
    for ax in (ax_phase, ax_eff):
        for w, alpha in ((1, 0.18), (2, 0.10)):
            ax.axvspan(-w, w, color="gray", alpha=alpha)
    ax_phase.set_xlabel("Wavelength detuning dlam (nm)")
    ax_phase.set_ylabel("Max phase LUT error (x pi rad)")
    ax_phase.set_title(f"Phase LUT error vs detuning ({n_pairs_for_main}-pair DBR + 72nm MgF2 AR)")
    ax_phase.legend(fontsize=8)
    ax_phase.grid(True, alpha=0.3)

    ax_eff.set_xlabel("Wavelength detuning dlam (nm)")
    ax_eff.set_ylabel("+1 order efficiency (absolute)")
    ax_eff.set_title("First-order diffraction efficiency vs detuning")
    ax_eff.legend(fontsize=8)
    ax_eff.grid(True, alpha=0.3)

    # DBR pair-count comparison, green only (the resonance suspect)
    for n_pairs in (4, 5, 6):
        r = results["sweeps"][("green", n_pairs)]
        ax_phase_pairs.plot(r["dlam_nm"], r["max_err"] / np.pi, linewidth=2,
                            label=f"{n_pairs} pairs")
        ax_eff_pairs.plot(r["dlam_nm"], r["eta1"], linewidth=2, label=f"{n_pairs} pairs")
    ax_phase_pairs.axhline(1 / 8, color="k", linestyle="--", alpha=0.6, label="pi/8 tolerance")
    for ax in (ax_phase_pairs, ax_eff_pairs):
        for w, alpha in ((1, 0.18), (2, 0.10)):
            ax.axvspan(-w, w, color="gray", alpha=alpha)
    ax_phase_pairs.set_xlabel("Wavelength detuning dlam (nm)")
    ax_phase_pairs.set_ylabel("Max phase LUT error (x pi rad)")
    ax_phase_pairs.set_title("Green: DBR pair-count sweep (phase error)")
    ax_phase_pairs.legend(fontsize=8)
    ax_phase_pairs.grid(True, alpha=0.3)

    ax_eff_pairs.set_xlabel("Wavelength detuning dlam (nm)")
    ax_eff_pairs.set_ylabel("+1 order efficiency (absolute)")
    ax_eff_pairs.set_title("Green: DBR pair-count sweep (+1 order efficiency)")
    ax_eff_pairs.legend(fontsize=8)
    ax_eff_pairs.grid(True, alpha=0.3)

    fig.suptitle("Spectral bandwidth of the Sb2Se3+DBR+AR phase LUT\n"
                 "gray bands: +-1nm / +-2nm laser-diode linewidth", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    results = bandwidth_report()
    print_report(results)
    plot_bandwidth(results)

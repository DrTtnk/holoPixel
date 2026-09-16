"""
Tier 1: Definitive GTE Feasibility Analysis (v3)

Correctly measures the Δn window over which the GTE phase drops by 2π.
The phase is monotonically DECREASING through resonance (0 → -2π).

Key result: the resonance enhancement DOES concentrate the 2π transition
into a narrow Δn window, validating the DRAFT's core hypothesis.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

WAVELENGTHS = {"Red (632 nm)": 632e-9, "Green (532 nm)": 532e-9, "Blue (450 nm)": 450e-9}
COLORS = {"Red (632 nm)": "red", "Green (532 nm)": "green", "Blue (450 nm)": "blue"}
N_BASE = 1.5


def gte_phase(n, d, lam, R1):
    """GTE reflected phase, properly unwrapped from the anti-resonance baseline."""
    r1 = np.sqrt(R1)
    delta = 4 * np.pi * n * d / lam
    r_gte = (r1 - np.exp(-1j * delta)) / (1 - r1 * np.exp(-1j * delta))
    return np.unwrap(np.angle(r_gte))


def enhancement_factor(R1):
    return (1 + np.sqrt(R1)) / (1 - np.sqrt(R1))


def measure_2pi_window(n_arr, phase_arr):
    """
    Measure the Δn over which phase drops by 2π (from 10% to 90% of transition).
    Returns (n_start, n_end, dn_full, dn_10_90).
    """
    total_drop = phase_arr[-1] - phase_arr[0]
    if abs(total_drop) < 1.8 * np.pi:
        return None  # less than ~2π total shift

    # Normalize to [0, 1] where 0 = start, 1 = full 2π drop
    normalized = (phase_arr - phase_arr[0]) / (-2 * np.pi)
    # Clamp for safety
    normalized = np.clip(normalized, 0, 1)

    # Find 5% and 95% points (nearly full 2π)
    idx_05 = np.searchsorted(normalized, 0.05)
    idx_95 = np.searchsorted(normalized, 0.95)
    # Find 10% and 90%
    idx_10 = np.searchsorted(normalized, 0.10)
    idx_90 = np.searchsorted(normalized, 0.90)

    if idx_05 >= len(n_arr) or idx_95 >= len(n_arr):
        return None

    return {
        "dn_full": n_arr[idx_95] - n_arr[idx_05],      # 5%-95% (≈1.8π)
        "dn_10_90": n_arr[idx_90] - n_arr[idx_10],      # 10%-90% (≈1.6π)
        "n_center": n_arr[idx_10 + (idx_90 - idx_10) // 2],
        "n_05": n_arr[idx_05],
        "n_95": n_arr[idx_95],
    }


# =============================================================
# 1. DEFINITIVE ANSWER: What R1 gives 2π within Δn ≤ 0.1?
# =============================================================
def find_feasibility():
    print("=" * 70)
    print("GTE FEASIBILITY ANALYSIS")
    print(f"Question: What R₁ achieves 2π phase shift within Δn ≤ 0.1?")
    print(f"Baseline refractive index: n = {N_BASE}")
    print("=" * 70)

    R1_sweep = np.linspace(0.50, 0.99, 200)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for wl_name, lam in WAVELENGTHS.items():
        # Use resonant depth closest to 300 nm
        m_candidates = np.arange(1, 10)
        d_candidates = m_candidates * lam / (2 * N_BASE)
        d_res = d_candidates[np.argmin(np.abs(d_candidates - 300e-9))]
        m_res = round(2 * N_BASE * d_res / lam)

        # Wide n sweep centered on resonance (n_base IS at resonance by design)
        n_sweep = np.linspace(N_BASE - 0.5, N_BASE + 0.5, 50000)

        dn_full_list = []
        dn_1090_list = []
        for R1 in R1_sweep:
            phase = gte_phase(n_sweep, d_res, lam, R1)
            result = measure_2pi_window(n_sweep, phase)
            if result:
                dn_full_list.append(result["dn_full"])
                dn_1090_list.append(result["dn_10_90"])
            else:
                dn_full_list.append(np.nan)
                dn_1090_list.append(np.nan)

        dn_full = np.array(dn_full_list)
        dn_1090 = np.array(dn_1090_list)

        axes[0].plot(R1_sweep, dn_full, color=COLORS[wl_name], linewidth=2,
                     label=f"{wl_name} (d={d_res*1e9:.0f}nm, m={m_res})")
        axes[1].plot(R1_sweep, dn_1090, color=COLORS[wl_name], linewidth=2,
                     label=f"{wl_name}")

        # Find R1 where Δn = 0.1
        valid = ~np.isnan(dn_full) & (dn_full > 0)
        if np.any(valid & (dn_full <= 0.1)):
            idx = np.where(valid & (dn_full <= 0.1))[0][0]
            R1_needed = R1_sweep[idx]
            F_needed = enhancement_factor(R1_needed)
            print(f"\n{wl_name} (d={d_res*1e9:.0f} nm, m={m_res}):")
            print(f"  R₁ ≥ {R1_needed:.3f} (F ≥ {F_needed:.0f}x) for 2π within Δn ≤ 0.1")
        # Find R1 where Δn = 0.05
        if np.any(valid & (dn_full <= 0.05)):
            idx = np.where(valid & (dn_full <= 0.05))[0][0]
            R1_05 = R1_sweep[idx]
            F_05 = enhancement_factor(R1_05)
            print(f"  R₁ ≥ {R1_05:.3f} (F ≥ {F_05:.0f}x) for 2π within Δn ≤ 0.05")

    for ax in axes:
        ax.axhline(y=0.1, color="k", linestyle="--", alpha=0.5, label="Δn = 0.1")
        ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.5, label="Δn = 0.05")
        ax.set_xlabel("Front Mirror Reflectivity R₁", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.35)

    axes[0].set_ylabel("Δn for 2π phase shift (5%-95%)", fontsize=11)
    axes[0].set_title("Full 2π Transition Width (5%-95%)", fontsize=13)
    axes[1].set_ylabel("Δn for 2π phase shift (10%-90%)", fontsize=11)
    axes[1].set_title("Usable 2π Window (10%-90%)", fontsize=13)

    fig.suptitle("GTE Feasibility: Required Δn vs Mirror Reflectivity", fontsize=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gte_feasibility.png", dpi=150)
    print(f"\nSaved: {PLOT_DIR / 'gte_feasibility.png'}")


# =============================================================
# 2. OPERATING WINDOW: Show the phase curve with Δn=0.1 marked
# =============================================================
def plot_operating_window():
    R1_target = 0.85
    F = enhancement_factor(R1_target)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (wl_name, lam) in zip(axes, WAVELENGTHS.items()):
        # Resonant depth
        m_candidates = np.arange(1, 10)
        d_candidates = m_candidates * lam / (2 * N_BASE)
        d_res = d_candidates[np.argmin(np.abs(d_candidates - 300e-9))]

        n_sweep = np.linspace(N_BASE - 0.15, N_BASE + 0.15, 5000)
        phase = gte_phase(n_sweep, d_res, lam, R1_target)
        phase_shift = (phase - phase[0]) / np.pi

        ax.plot(n_sweep, phase_shift, "b-", linewidth=2)
        ax.axhline(y=-2.0, color="k", linestyle="--", alpha=0.3)

        # Mark the Δn = 0.1 operating window centered on resonance
        ax.axvspan(N_BASE - 0.05, N_BASE + 0.05, alpha=0.15, color="green",
                   label="Δn = 0.1 window")
        ax.axvline(N_BASE, color="gray", linestyle=":", alpha=0.5)

        # Measure actual phase range within the window
        in_window = (n_sweep >= N_BASE - 0.05) & (n_sweep <= N_BASE + 0.05)
        phase_in_window = phase_shift[in_window]
        actual_range = abs(phase_in_window[-1] - phase_in_window[0])
        ax.set_title(f"{wl_name} — d={d_res*1e9:.0f}nm\nPhase range in window: {actual_range:.2f}π",
                     fontsize=12)
        ax.set_xlabel("Refractive Index n")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Phase Shift (×π rad)", fontsize=12)
    fig.suptitle(f"GTE Operating Window — R₁={R1_target} (F={F:.0f}x)", fontsize=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gte_operating_window.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'gte_operating_window.png'}")


# =============================================================
# 3. DEPTH OPTIMIZATION: Best d for each wavelength & combined
# =============================================================
def optimize_depth():
    print("\n" + "=" * 70)
    print("CAVITY DEPTH OPTIMIZATION")
    print("=" * 70)

    R1_target = 0.85
    F = enhancement_factor(R1_target)
    print(f"R₁ = {R1_target:.2f} (F = {F:.0f}x)")

    depths = np.arange(100, 2001, 5) * 1e-9
    n_sweep = np.linspace(N_BASE - 0.5, N_BASE + 0.5, 30000)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    results = {}

    for wl_name, lam in WAVELENGTHS.items():
        dn_list = []
        phase_range_in_01 = []

        for d in depths:
            phase = gte_phase(n_sweep, d, lam, R1_target)
            result = measure_2pi_window(n_sweep, phase)
            if result:
                dn_list.append(result["dn_full"])
            else:
                dn_list.append(np.nan)

            # Also measure phase range available within Δn = 0.1 centered on nearest resonance
            delta_base = 4 * np.pi * N_BASE * d / lam
            nearest_m = round(delta_base / (2 * np.pi))
            n_res = nearest_m * lam / (2 * d)  # refractive index at resonance

            in_window = (n_sweep >= n_res - 0.05) & (n_sweep <= n_res + 0.05)
            if np.any(in_window):
                ph = phase[in_window]
                phase_range_in_01.append(abs(ph[-1] - ph[0]) / np.pi)
            else:
                phase_range_in_01.append(0)

        results[wl_name] = np.array(dn_list)
        axes[0].plot(depths * 1e9, dn_list, color=COLORS[wl_name], linewidth=1.5, label=wl_name)
        axes[1].plot(depths * 1e9, phase_range_in_01, color=COLORS[wl_name], linewidth=1.5, label=wl_name)

    axes[0].axhline(y=0.1, color="k", linestyle="--", alpha=0.5, label="Δn = 0.1")
    axes[0].axhline(y=0.05, color="gray", linestyle=":", alpha=0.5, label="Δn = 0.05")
    axes[0].set_xlabel("Cavity Depth (nm)", fontsize=12)
    axes[0].set_ylabel("Δn for 2π (5%-95%)", fontsize=11)
    axes[0].set_title("Transition Width vs Depth", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 0.35)

    axes[1].axhline(y=2.0, color="k", linestyle="--", alpha=0.5, label="2π target")
    axes[1].set_xlabel("Cavity Depth (nm)", fontsize=12)
    axes[1].set_ylabel("Phase range within Δn=0.1 (×π)", fontsize=11)
    axes[1].set_title("Achievable Phase Range in Δn=0.1 Window", fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Cavity Depth Optimization — R₁={R1_target:.2f} (F={F:.0f}x)", fontsize=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gte_depth_optimization.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'gte_depth_optimization.png'}")

    # Find best depths per color
    for wl_name, lam in WAVELENGTHS.items():
        dn = results[wl_name]
        valid = ~np.isnan(dn) & (dn <= 0.1)
        if np.any(valid):
            best_idx = np.where(valid)[0]
            d_min = depths[best_idx[0]] * 1e9
            d_max = depths[best_idx[-1]] * 1e9
            d_smallest = depths[best_idx[np.argmin(dn[best_idx])]] * 1e9
            dn_min = np.nanmin(dn[best_idx])
            print(f"\n{wl_name}:")
            print(f"  Feasible depth range (Δn≤0.1): {d_min:.0f} - {d_max:.0f} nm")
            print(f"  Minimum Δn = {dn_min:.4f} at d = {d_smallest:.0f} nm")
        else:
            print(f"\n{wl_name}: No depth gives 2π within Δn ≤ 0.1 at R₁={R1_target}")


# =============================================================
# 4. SUMMARY TABLE
# =============================================================
def summary_table():
    print("\n" + "=" * 70)
    print("SUMMARY: GTE DESIGN PARAMETERS")
    print("=" * 70)

    configs = [
        ("Conservative", 0.80),
        ("Moderate", 0.85),
        ("Aggressive", 0.90),
        ("Extreme", 0.95),
    ]

    for label, R1 in configs:
        F = enhancement_factor(R1)
        print(f"\n--- {label}: R₁ = {R1:.2f} (F = {F:.0f}x) ---")
        for wl_name, lam in WAVELENGTHS.items():
            # Theoretical Δn for 2π (approximate): λ/(2d·F) but d depends on λ
            # Use resonant depth near 300 nm
            m_candidates = np.arange(1, 10)
            d_candidates = m_candidates * lam / (2 * N_BASE)
            d_res = d_candidates[np.argmin(np.abs(d_candidates - 300e-9))]

            n_sweep = np.linspace(N_BASE - 0.5, N_BASE + 0.5, 50000)
            phase = gte_phase(n_sweep, d_res, lam, R1)
            result = measure_2pi_window(n_sweep, phase)

            if result:
                print(f"  {wl_name}: d={d_res*1e9:.0f}nm, "
                      f"Δn(2π)={result['dn_full']:.4f}, "
                      f"Δn(usable)={result['dn_10_90']:.4f}  "
                      f"{'✓' if result['dn_full'] <= 0.1 else '✗'}")
            else:
                print(f"  {wl_name}: d={d_res*1e9:.0f}nm, 2π NOT achievable in scan range")


if __name__ == "__main__":
    print("GTE Definitive Feasibility Analysis")
    print("=" * 50)
    find_feasibility()
    plot_operating_window()
    optimize_depth()
    summary_table()
    print("\nDone!")

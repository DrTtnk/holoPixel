"""
Tier 1: Corrected GTE Design — Per-Color Optimized Parameters

Based on findings from gte_definitive.py:
- Each color needs its own resonant cavity depth
- R₁ must be ≥ 0.90 (F ≥ 38x) for all colors to work within Δn ≤ 0.1
- Also explores dielectric Bragg reflector (DBR) as Ag replacement
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tmm import coh_tmm
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

N_BASE = 1.5
WAVELENGTHS = {"Red (632 nm)": 632e-9, "Green (532 nm)": 532e-9, "Blue (450 nm)": 450e-9}
COLORS = {"Red (632 nm)": "red", "Green (532 nm)": "green", "Blue (450 nm)": "blue"}

# --- Corrected design parameters ---
# Per-color resonant depths using m=2 for red (deeper cavity, easier to achieve)
DESIGNS = {
    "Red (632 nm)":   {"d": 421.3e-9, "m": 2, "lam": 632e-9},
    "Green (532 nm)": {"d": 354.7e-9, "m": 2, "lam": 532e-9},
    "Blue (450 nm)":  {"d": 300.0e-9, "m": 2, "lam": 450e-9},
}


def gte_phase(n, d, lam, R1):
    r1 = np.sqrt(R1)
    delta = 4 * np.pi * n * d / lam
    r_gte = (r1 - np.exp(-1j * delta)) / (1 - r1 * np.exp(-1j * delta))
    return np.unwrap(np.angle(r_gte))


def enhancement_factor(R1):
    return (1 + np.sqrt(R1)) / (1 - np.sqrt(R1))


# --- Silver optical constants ---
AG_DATA = np.array([
    [400, 0.075, 1.93], [450, 0.055, 2.42], [500, 0.050, 2.87],
    [532, 0.050, 3.11], [550, 0.055, 3.33], [600, 0.060, 3.75],
    [632, 0.056, 3.97], [650, 0.060, 4.18], [700, 0.075, 4.62],
])


def ag_index(wl_nm):
    n = np.interp(wl_nm, AG_DATA[:, 0], AG_DATA[:, 1])
    k = np.interp(wl_nm, AG_DATA[:, 0], AG_DATA[:, 2])
    return complex(n, k)


# =============================================================
# 1. Corrected per-color phase response
# =============================================================
def plot_corrected_design():
    R1_target = 0.90
    F = enhancement_factor(R1_target)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    n_sweep = np.linspace(N_BASE - 0.08, N_BASE + 0.08, 3000)

    for ax, (wl_name, design) in zip(axes, DESIGNS.items()):
        phase = gte_phase(n_sweep, design["d"], design["lam"], R1_target)
        phase_shift = (phase - phase[0]) / np.pi

        ax.plot(n_sweep, phase_shift, color=COLORS[wl_name], linewidth=2.5)
        ax.axhline(y=-2.0, color="k", linestyle="--", alpha=0.3)
        ax.axvspan(N_BASE - 0.05, N_BASE + 0.05, alpha=0.12, color="green")
        ax.axvline(N_BASE, color="gray", linestyle=":", alpha=0.5)

        # Measure phase in window
        in_win = (n_sweep >= N_BASE - 0.05) & (n_sweep <= N_BASE + 0.05)
        ph_range = abs(phase_shift[in_win][-1] - phase_shift[in_win][0])

        ax.set_title(f"{wl_name}\nd = {design['d']*1e9:.1f} nm (m={design['m']})\n"
                     f"Phase in Δn=0.1: {ph_range:.2f}π", fontsize=12)
        ax.set_xlabel("Refractive Index n")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2.5, 0.3)

    axes[0].set_ylabel("Phase Shift (×π rad)")
    fig.suptitle(f"CORRECTED GTE Design — R₁={R1_target} (F={F:.0f}x), Per-Color Depths",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gte_corrected_design.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'gte_corrected_design.png'}")


# =============================================================
# 2. Dielectric Bragg Reflector (DBR) — design & TMM
# =============================================================
def design_dbr_mirror(target_R, lam_center_nm, n_high=2.3, n_low=1.45):
    """
    Design a quarter-wave DBR stack to achieve target reflectivity.
    Uses TiO2 (n≈2.3) and SiO2 (n≈1.45).

    R ≈ ((n_high/n_low)^(2N) - 1)² / ((n_high/n_low)^(2N) + 1)²
    for N pairs on a substrate with n_sub ≈ n_low.
    """
    ratio = n_high / n_low
    # Solve for N: R = ((ratio^2N - 1)/(ratio^2N + 1))^2
    # ratio^2N = (1+√R)/(1-√R)
    target = (1 + np.sqrt(target_R)) / (1 - np.sqrt(target_R))
    N = np.log(target) / (2 * np.log(ratio))
    N = int(np.ceil(N))

    d_high = lam_center_nm / (4 * n_high)  # quarter-wave thickness
    d_low = lam_center_nm / (4 * n_low)

    return N, d_high, d_low


def simulate_dbr_gte(wl_name, design, R1_target=0.90):
    """Simulate GTE with DBR top mirror + Ag back mirror using TMM."""
    lam_nm = design["lam"] * 1e9
    d_cavity_nm = design["d"] * 1e9

    n_high, n_low = 2.3, 1.45
    N_pairs, d_high, d_low = design_dbr_mirror(R1_target, lam_nm, n_high, n_low)

    n_sweep = np.linspace(N_BASE - 0.08, N_BASE + 0.08, 500)
    phases_dbr = []
    reflectances_dbr = []

    for n_cav in n_sweep:
        # Stack: Air | [TiO2/SiO2]×N | EO cavity | Ag (thick) | substrate
        n_list = [1.0]
        d_list = [np.inf]

        # DBR top mirror (start with high-index)
        for _ in range(N_pairs):
            n_list.extend([n_high, n_low])
            d_list.extend([d_high, d_low])

        # Cavity
        n_list.append(n_cav)
        d_list.append(d_cavity_nm)

        # Ag back mirror (thick)
        n_ag = ag_index(lam_nm)
        n_list.append(n_ag)
        d_list.append(200)

        # Substrate
        n_list.append(1.5)
        d_list.append(np.inf)

        result = coh_tmm("s", n_list, d_list, 0, lam_nm)
        phases_dbr.append(np.angle(result["r"]))
        reflectances_dbr.append(abs(result["r"]) ** 2)

    return (n_sweep, np.unwrap(np.array(phases_dbr)),
            np.array(reflectances_dbr), N_pairs, d_high, d_low)


def plot_dbr_comparison():
    R1_target = 0.90
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (wl_name, design) in enumerate(DESIGNS.items()):
        n_sweep, phases_dbr, R_dbr, N_pairs, d_high, d_low = simulate_dbr_gte(
            wl_name, design, R1_target)

        phase_shift_dbr = (phases_dbr - phases_dbr[0]) / np.pi

        # Also compute Ag-only version for comparison
        lam_nm = design["lam"] * 1e9
        d_cavity_nm = design["d"] * 1e9
        phases_ag = []
        R_ag = []
        for n_cav in n_sweep:
            n_ag = ag_index(lam_nm)
            # Ag top mirror (35nm) + cavity + Ag back
            result = coh_tmm("s",
                             [1.0, n_ag, n_cav, n_ag, 1.5],
                             [np.inf, 35, d_cavity_nm, 200, np.inf], 0, lam_nm)
            phases_ag.append(np.angle(result["r"]))
            R_ag.append(abs(result["r"]) ** 2)
        phases_ag = np.unwrap(np.array(phases_ag))
        phase_shift_ag = (phases_ag - phases_ag[0]) / np.pi

        # Phase comparison
        axes[0, col].plot(n_sweep, phase_shift_dbr, color=COLORS[wl_name],
                          linewidth=2, label=f"DBR ({N_pairs} pairs)")
        axes[0, col].plot(n_sweep, phase_shift_ag, color="gray",
                          linewidth=2, linestyle="--", label="Ag 35nm")
        axes[0, col].axhline(y=-2.0, color="k", linestyle="--", alpha=0.3)
        axes[0, col].set_title(f"{wl_name} — d={d_cavity_nm:.0f}nm", fontsize=12)
        axes[0, col].legend(fontsize=9)
        axes[0, col].grid(True, alpha=0.3)

        # Reflectance comparison
        axes[1, col].plot(n_sweep, R_dbr, color=COLORS[wl_name],
                          linewidth=2, label=f"DBR ({N_pairs} pairs)")
        axes[1, col].plot(n_sweep, R_ag, color="gray",
                          linewidth=2, linestyle="--", label="Ag 35nm")
        axes[1, col].set_xlabel("Refractive Index n")
        axes[1, col].set_ylim(0.5, 1.02)
        axes[1, col].axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
        axes[1, col].legend(fontsize=9)
        axes[1, col].grid(True, alpha=0.3)

        print(f"{wl_name}: DBR = {N_pairs} pairs of TiO₂({d_high:.1f}nm)/SiO₂({d_low:.1f}nm)")
        print(f"  Total DBR thickness: {N_pairs * (d_high + d_low):.0f} nm")
        print(f"  Reflectance range: {R_dbr.min():.3f} - {R_dbr.max():.3f}")

    axes[0, 0].set_ylabel("Phase Shift (×π rad)")
    axes[1, 0].set_ylabel("Reflectance |r|²")
    fig.suptitle(f"DBR vs Ag Top Mirror — Target R₁={R1_target}", fontsize=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "dbr_vs_ag.png", dpi=150)
    print(f"\nSaved: {PLOT_DIR / 'dbr_vs_ag.png'}")


# =============================================================
# 3. DBR Reflectance bandwidth check
# =============================================================
def dbr_bandwidth():
    """Check if DBR mirrors designed for each color have sufficient bandwidth."""
    fig, ax = plt.subplots(figsize=(12, 6))
    wl_sweep = np.linspace(380, 780, 500)

    for wl_name, design in DESIGNS.items():
        lam_center_nm = design["lam"] * 1e9
        n_high, n_low = 2.3, 1.45
        N_pairs, d_high, d_low = design_dbr_mirror(0.90, lam_center_nm, n_high, n_low)

        R_spectrum = []
        for wl in wl_sweep:
            n_list = [1.0]
            d_list = [np.inf]
            for _ in range(N_pairs):
                n_list.extend([n_high, n_low])
                d_list.extend([d_high, d_low])
            n_list.append(1.5)
            d_list.append(np.inf)

            result = coh_tmm("s", n_list, d_list, 0, wl)
            R_spectrum.append(abs(result["r"]) ** 2)

        ax.plot(wl_sweep, R_spectrum, color=COLORS[wl_name], linewidth=2,
                label=f"{wl_name} DBR ({N_pairs} pairs)")

    ax.axhline(y=0.90, color="k", linestyle="--", alpha=0.3, label="R=0.90")
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance", fontsize=12)
    ax.set_title("DBR Mirror Bandwidth — TiO₂/SiO₂", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "dbr_bandwidth.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'dbr_bandwidth.png'}")


if __name__ == "__main__":
    print("Corrected GTE Design + DBR Analysis")
    print("=" * 50)
    plot_corrected_design()
    print()
    plot_dbr_comparison()
    print()
    dbr_bandwidth()
    print("\nDone!")

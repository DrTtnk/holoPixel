"""
Tier 1: TMM-Based GTE Simulation with Realistic Material Stack

Models a physical thin-film stack using the Transfer Matrix Method:
    Air | Ag (thin, partial reflector) | EO cavity | Ag (thick, perfect reflector)

Uses wavelength-dependent complex refractive indices for silver (Ag)
from Johnson & Christy data (interpolated).

Compares TMM results against the analytical GTE model.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tmm import coh_tmm
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# --- Silver (Ag) optical constants (Johnson & Christy, selected points) ---
# Format: wavelength_nm, n, k  (n + ik is the complex refractive index)
AG_DATA = np.array([
    [400, 0.075, 1.93],
    [450, 0.055, 2.42],
    [500, 0.050, 2.87],
    [532, 0.050, 3.11],
    [550, 0.055, 3.33],
    [600, 0.060, 3.75],
    [632, 0.056, 3.97],
    [650, 0.060, 4.18],
    [700, 0.075, 4.62],
    [750, 0.090, 5.05],
])


def ag_index(wavelength_nm):
    """Interpolate Ag complex refractive index at given wavelength."""
    n = np.interp(wavelength_nm, AG_DATA[:, 0], AG_DATA[:, 1])
    k = np.interp(wavelength_nm, AG_DATA[:, 0], AG_DATA[:, 2])
    return complex(n, k)


def simulate_gte_tmm(wavelength_nm, n_cavity, d_cavity_nm, d_ag_top_nm, d_ag_bottom_nm=200):
    """
    Simulate a GTE stack using TMM.
    
    Stack: Air | Ag_top | EO_cavity | Ag_bottom | Glass_substrate
    Returns the reflected phase in radians.
    """
    lam = wavelength_nm  # tmm uses consistent units (we'll use nm throughout)
    n_ag = ag_index(wavelength_nm)

    # Layer structure: [n_list], [d_list]
    # Semi-infinite layers have d = inf
    n_list = [1.0, n_ag, n_cavity, n_ag, 1.5]  # air, Ag top, cavity, Ag bottom, glass
    d_list = [np.inf, d_ag_top_nm, d_cavity_nm, d_ag_bottom_nm, np.inf]

    result = coh_tmm("s", n_list, d_list, 0, lam)  # normal incidence, s-pol
    return np.angle(result["r"])


# =============================================================
# Sweep 1: Phase vs n_cavity for different Ag top thicknesses
# =============================================================
def sweep_ag_thickness():
    """Find what Ag top-mirror thickness gives optimal phase range."""
    wavelengths = {"Red (632 nm)": 632, "Green (532 nm)": 532, "Blue (450 nm)": 450}
    ag_top_thicknesses = [20, 25, 30, 35, 40, 45, 50, 60]  # nm
    d_cavity = 300  # nm
    n_sweep = np.linspace(1.5, 1.6, 500)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (wl_name, lam) in zip(axes, wavelengths.items()):
        for d_ag in ag_top_thicknesses:
            phases = np.array([simulate_gte_tmm(lam, n, d_cavity, d_ag) for n in n_sweep])
            phases_unwrapped = np.unwrap(phases)
            phase_shift = phases_unwrapped - phases_unwrapped[0]
            ax.plot(n_sweep, phase_shift / np.pi, label=f"Ag={d_ag} nm")

        ax.set_title(wl_name, fontsize=14)
        ax.set_xlabel("Cavity Refractive Index n")
        ax.axhline(y=2.0, color="k", linestyle="--", alpha=0.5, label="2π target")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Phase Shift (×π rad)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"TMM: GTE Phase vs Ag Top Mirror Thickness — d_cavity={d_cavity} nm", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tmm_phase_vs_ag_thickness.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'tmm_phase_vs_ag_thickness.png'}")


# =============================================================
# Sweep 2: Compare TMM vs Analytical for a given configuration
# =============================================================
def compare_tmm_vs_analytical():
    """Side-by-side comparison: TMM realistic stack vs analytical GTE formula."""
    from tier1_tmm.gte_analytical import gte_reflected_phase

    wavelengths = {"Red (632 nm)": 632, "Green (532 nm)": 532, "Blue (450 nm)": 450}
    d_cavity = 300  # nm
    d_ag_top = 35   # nm — reasonable partial reflector
    n_sweep = np.linspace(1.5, 1.6, 500)

    # Estimate effective R1 from Ag film at each wavelength
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (wl_name, lam) in zip(axes, wavelengths.items()):
        # TMM simulation
        tmm_phases = np.array([simulate_gte_tmm(lam, n, d_cavity, d_ag_top) for n in n_sweep])
        tmm_unwrapped = np.unwrap(tmm_phases)
        tmm_shift = tmm_unwrapped - tmm_unwrapped[0]

        # Estimate R1: simulate Ag film reflectivity in isolation
        n_ag = ag_index(lam)
        r_ag = coh_tmm("s", [1.0, n_ag, 1.5], [np.inf, d_ag_top, np.inf], 0, lam)
        R1_eff = abs(r_ag["r"]) ** 2

        # Analytical GTE with estimated R1
        analytical_phases = gte_reflected_phase(n_sweep, d_cavity * 1e-9, lam * 1e-9, R1_eff)
        analytical_unwrapped = np.unwrap(analytical_phases)
        analytical_shift = analytical_unwrapped - analytical_unwrapped[0]

        ax.plot(n_sweep, tmm_shift / np.pi, "b-", linewidth=2, label=f"TMM (Ag={d_ag_top}nm)")
        ax.plot(n_sweep, analytical_shift / np.pi, "r--", linewidth=2, label=f"Analytical (R₁={R1_eff:.3f})")
        ax.set_title(f"{wl_name}\nR₁_eff = {R1_eff:.3f}", fontsize=12)
        ax.set_xlabel("Cavity Refractive Index n")
        ax.axhline(y=2.0, color="k", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Phase Shift (×π rad)")
    fig.suptitle("TMM vs Analytical GTE — Validation", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tmm_vs_analytical.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'tmm_vs_analytical.png'}")


# =============================================================
# Sweep 3: Reflectance spectrum — verify |r|≈1 (GTE property)
# =============================================================
def sweep_reflectance_spectrum():
    """Verify that total reflectance stays near 1 (lossless GTE condition)."""
    wavelength_sweep = np.linspace(400, 750, 500)
    d_cavity = 300
    n_cavity = 1.55  # mid-range
    ag_thicknesses = [25, 30, 35, 40, 50]

    fig, ax = plt.subplots(figsize=(10, 6))
    for d_ag in ag_thicknesses:
        R_spectrum = []
        for lam in wavelength_sweep:
            n_ag = ag_index(lam)
            result = coh_tmm("s", [1.0, n_ag, n_cavity, n_ag, 1.5],
                             [np.inf, d_ag, d_cavity, 200, np.inf], 0, lam)
            R_spectrum.append(abs(result["r"]) ** 2)
        ax.plot(wavelength_sweep, R_spectrum, label=f"Ag_top={d_ag} nm")

    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance |r|²", fontsize=12)
    ax.set_title(f"GTE Total Reflectance — d_cavity={d_cavity}nm, n={n_cavity}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "tmm_reflectance_spectrum.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'tmm_reflectance_spectrum.png'}")


if __name__ == "__main__":
    print("TMM GTE Simulation — Tier 1")
    print("=" * 50)
    sweep_ag_thickness()
    sweep_reflectance_spectrum()
    # compare_tmm_vs_analytical requires import from sibling — run from project root
    try:
        compare_tmm_vs_analytical()
    except ImportError:
        print("Run from project root to enable TMM vs Analytical comparison:")
        print("  cd holoPixel && python -m tier1_tmm.gte_tmm")
    print("\nDone! Check plots/ directory.")

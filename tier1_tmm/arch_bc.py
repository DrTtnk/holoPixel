"""
Architecture B: Trumpet Valve — Silicon Photonics MZI + Grating Coupler

Models a Mach-Zehnder Interferometer (MZI) with electro-optic phase arm,
followed by a grating coupler that redirects light vertically.

Key advantage: modulation happens HORIZONTALLY (plenty of space),
beam steering happens VERTICALLY (via grating coupler pitch).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def mzi_output(delta_phi):
    """MZI intensity output: I = cos²(Δφ/2), phase = Δφ/2."""
    I_out = np.cos(delta_phi / 2) ** 2
    # The output E-field phase shift is delta_phi (one arm modulated)
    return I_out, delta_phi


def grating_coupler_angle(lam, pitch, n_eff):
    """
    Grating coupler diffraction angle.
    n_eff·λ/Λ = sin(θ) + m  (simplified for m=1, normal incidence from waveguide)
    θ = arcsin(n_eff - λ/Λ)
    """
    sin_theta = n_eff - lam / pitch
    valid = np.abs(sin_theta) <= 1
    theta = np.where(valid, np.degrees(np.arcsin(sin_theta)), np.nan)
    return theta


# =============================================================
# 1. MZI Phase Modulation
# =============================================================
def mzi_analysis():
    print("=" * 70)
    print("ARCHITECTURE B: Silicon Photonics MZI")
    print("=" * 70)

    lam = 632e-9
    n_eo = 1.5  # baseline EO material index
    dn = 0.1

    # MZI arm length for 2π phase shift
    # Δφ = 2π·Δn·L/λ = 2π → L = λ/Δn
    L_2pi = lam / dn
    print(f"MZI arm length for 2π at Δn=0.1: {L_2pi*1e6:.1f} µm")
    print(f"Hogel size: 169.3 µm")
    print(f"Fits in hogel: {'YES' if L_2pi < 169.3e-6 else 'Needs folding'}")

    # Phase vs arm length for different Δn
    L_sweep = np.linspace(0, 20e-6, 500)
    dn_values = [0.01, 0.02, 0.05, 0.1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Phase shift vs arm length
    for dn_val in dn_values:
        phase = 2 * np.pi * dn_val * L_sweep / lam
        axes[0].plot(L_sweep * 1e6, phase / np.pi, linewidth=2, label=f"Δn={dn_val}")

    axes[0].axhline(y=2.0, color="k", linestyle="--", alpha=0.3, label="2π")
    axes[0].set_xlabel("Arm Length L (µm)")
    axes[0].set_ylabel("Phase Shift (×π)")
    axes[0].set_title("MZI Phase vs Arm Length")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Grating coupler angles
    pitches = np.linspace(300, 1500, 500) * 1e-9
    n_eff = 1.8  # typical for Si waveguide mode
    wavelengths = {"Red": 632e-9, "Green": 532e-9, "Blue": 450e-9}
    colors = {"Red": "red", "Green": "green", "Blue": "blue"}

    for wl_name, lam_gc in wavelengths.items():
        theta = grating_coupler_angle(lam_gc, pitches, n_eff)
        axes[1].plot(pitches * 1e9, theta, color=colors[wl_name], linewidth=2, label=wl_name)

    axes[1].axhline(y=30, color="k", linestyle="--", alpha=0.3, label="30° target")
    axes[1].axhline(y=0, color="gray", linestyle=":", alpha=0.3, label="Normal")
    axes[1].set_xlabel("Grating Pitch Λ (nm)")
    axes[1].set_ylabel("Diffraction Angle θ (°)")
    axes[1].set_title("Grating Coupler Angle vs Pitch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-30, 60)

    # Plot 3: Sub-pixel layout for hogel
    hogel = 169.3e-6
    wg_width = 0.5e-6
    wg_pitch = 1.0e-6  # sub-pixel pitch
    n_wg = int(hogel / wg_pitch)

    # How many sub-pixels can we fit with routing?
    mzi_lengths = [3, 5, 7, 10, 15]  # µm
    sub_pixel_counts = []
    for L in mzi_lengths:
        L_m = L * 1e-6
        # Each MZI needs: arm_length vertical + grating coupler horizontal
        # Compact layout: waveguides run vertically, gratings at bottom
        n_horizontal = int(hogel / wg_pitch)
        n_vertical = 1  # each MZI occupies one column
        sub_pixel_counts.append(n_horizontal)

    axes[2].bar([str(L) for L in mzi_lengths], sub_pixel_counts, color="steelblue")
    axes[2].set_xlabel("MZI Arm Length (µm)")
    axes[2].set_ylabel("Sub-pixels (1D)")
    axes[2].set_title(f"Sub-pixels per Row (hogel={hogel*1e6:.0f}µm)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Architecture B: Silicon Photonics (Trumpet Valve)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "arch_b_si_photonics.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'arch_b_si_photonics.png'}")

    # Key metrics
    for wl_name, lam_val in wavelengths.items():
        L = lam_val / dn
        print(f"{wl_name}: L_2π = {L*1e6:.1f} µm, "
              f"fits {int(hogel/wg_pitch)} waveguides per row")


# =============================================================
# Architecture C: Chladni Plate — Surface Acoustic Wave
# =============================================================
def saw_analysis():
    print("\n" + "=" * 70)
    print("ARCHITECTURE C: Surface Acoustic Waves (Chladni Plate)")
    print("=" * 70)

    # SAW parameters
    # LiNbO₃: acoustic velocity ~ 3488 m/s (Rayleigh wave)
    v_saw = 3488  # m/s

    # For λ_acoustic = 0.5 µm (matching optical sub-pixel pitch):
    lam_acoustic = 0.5e-6
    f_saw = v_saw / lam_acoustic
    print(f"SAW frequency for λ_a={lam_acoustic*1e6:.1f}µm: {f_saw/1e9:.2f} GHz")

    # SAW creates a periodic refractive index modulation via the photo-elastic effect
    # Δn = p · strain, where p is the photo-elastic coefficient
    # For LiNbO₃: p ≈ 0.12, and achievable strain ≈ 1e-4 to 1e-3
    # Δn ≈ 0.12 × 1e-3 = 1.2e-4 (TINY!)

    p_eo = 0.12  # photo-elastic coefficient
    strains = np.logspace(-5, -2, 100)
    dn_saw = p_eo * strains

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Achievable Δn vs strain
    axes[0].loglog(strains, dn_saw, "b-", linewidth=2)
    axes[0].axhline(y=0.1, color="r", linestyle="--", label="Δn=0.1 (GTE target)")
    axes[0].axhline(y=0.001, color="orange", linestyle="--", label="Δn=0.001")
    axes[0].axvspan(1e-4, 1e-3, alpha=0.1, color="green", label="Achievable strain")
    axes[0].set_xlabel("Acoustic Strain")
    axes[0].set_ylabel("Refractive Index Change Δn")
    axes[0].set_title("SAW: Δn vs Strain")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Diffraction efficiency (Raman-Nath regime)
    # For thin acoustic grating: η = sin²(π·Δn·L/λ)
    # For thick grating (Bragg): η = sin²(π·Δn·L/(λ·cos(θ)))
    lam = 632e-9
    L_interaction = np.linspace(0, 1000e-6, 500)  # interaction length
    dn_saw_typical = 1.2e-4

    eta = np.sin(np.pi * dn_saw_typical * L_interaction / lam) ** 2
    axes[1].plot(L_interaction * 1e6, eta * 100, "b-", linewidth=2)
    axes[1].set_xlabel("Interaction Length (µm)")
    axes[1].set_ylabel("Diffraction Efficiency (%)")
    axes[1].set_title(f"SAW Diffraction Efficiency (Δn={dn_saw_typical:.1e})")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=169.3, color="gray", linestyle=":", label="Hogel size")
    axes[1].legend()

    # Plot 3: SAW frequency vs required pitch
    pitches = np.linspace(0.3, 5, 100) * 1e-6
    freqs = v_saw / pitches

    axes[2].plot(pitches * 1e6, freqs / 1e9, "b-", linewidth=2)
    axes[2].axhline(y=f_saw / 1e9, color="r", linestyle="--",
                    label=f"0.5µm pitch ({f_saw/1e9:.1f} GHz)")
    axes[2].set_xlabel("Acoustic Wavelength (µm)")
    axes[2].set_ylabel("SAW Frequency (GHz)")
    axes[2].set_title("SAW Frequency Requirements")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Architecture C: SAW Acousto-Optics (Chladni Plate)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "arch_c_saw.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'arch_c_saw.png'}")

    # Key finding: SAW Δn is 1000× too small!
    print(f"\nSAW achievable Δn ≈ {dn_saw_typical:.1e} (via photo-elastic effect)")
    print(f"GTE requires Δn ≈ 0.1")
    print(f"Gap: {0.1/dn_saw_typical:.0f}× shortfall")
    print(f"\nSAW works differently: it creates a GRATING, not per-pixel phase shift")
    print(f"The diffraction efficiency at L={169.3}µm: {np.sin(np.pi*dn_saw_typical*169.3e-6/lam)**2*100:.2f}%")
    print(f"SAW is better suited as a BEAM STEERER, not a per-pixel modulator")
    print(f"Combine with a different phase modulator for hybrid architecture")


if __name__ == "__main__":
    print("Architecture B & C Analysis")
    print("=" * 50)
    mzi_analysis()
    saw_analysis()
    print("\nDone!")

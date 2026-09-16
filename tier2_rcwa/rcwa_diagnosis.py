"""
RCWA Diagnosis: Why is blazed grating efficiency only 44.5%?

Hypothesis: the simple linear phase model φ = 4πnd/λ is wrong.
The air-Sb₂Se₃-Al stack forms an implicit Fabry-Perot cavity
because of the high Fresnel reflection at the air-Sb₂Se₃ interface
(R ≈ 25-34% due to n=3.0-4.1).

Solution: compute the actual phase response using TMM, then choose
n values that give uniformly-spaced phases.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import grcwa

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Material constants
N_AMORPHOUS = 3.0
N_CRYSTALLINE = 4.1
K_SB2SE3 = 0.01
N_AL, K_AL = 0.92, 6.28
SUB_PIXEL_PITCH = 0.5  # µm
MIRROR_THICKNESS = 0.1  # µm


def eps_sb2se3(n, k=K_SB2SE3):
    return (n + 1j * k) ** 2


def eps_al():
    return (N_AL + 1j * K_AL) ** 2


# =============================================================
# Step 1: Compute actual phase vs n using TMM
# =============================================================
def tmm_phase_response(wavelength, film_thickness, n_range):
    """
    Compute reflected phase as function of Sb₂Se₃ refractive index
    for the stack: air | Sb₂Se₃ | Al
    
    Uses transfer matrix method for thin film on metal.
    """
    import tmm

    phases = []
    reflectances = []

    for n_sb in n_range:
        n_list = [1.0, n_sb + 1j * K_SB2SE3, N_AL + 1j * K_AL]
        d_list = [np.inf, film_thickness * 1e3, np.inf]  # tmm uses nm

        result = tmm.coh_tmm('s', n_list, d_list, 0, wavelength * 1e3)
        r = result['r']
        phases.append(np.angle(r))
        reflectances.append(abs(r) ** 2)

    return np.array(phases), np.array(reflectances)


def diagnose_phase():
    """Show the actual phase response and design correct n values."""
    print("=" * 70)
    print("DIAGNOSIS: Phase vs Index — TMM vs Linear Model")
    print("=" * 70)

    wavelength = 0.532  # µm

    thicknesses = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    n_range = np.linspace(N_AMORPHOUS, N_CRYSTALLINE, 500)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    best_d = None
    best_range = 0

    for idx, d in enumerate(thicknesses):
        phase_tmm, refl = tmm_phase_response(wavelength, d, n_range)
        phase_unwrap = np.unwrap(phase_tmm)

        # Linear model for comparison
        phase_linear = 4 * np.pi * n_range * d / wavelength

        # Shift to start at 0
        phase_tmm_shifted = phase_unwrap - phase_unwrap[0]
        phase_linear_shifted = phase_linear - phase_linear[0]

        # Phase range achieved
        total_range = abs(phase_tmm_shifted[-1])
        if total_range > best_range:
            best_range = total_range
            best_d = d

        ax = axes[idx // 3][idx % 3]
        ax.plot(n_range, phase_tmm_shifted / np.pi, "b-", linewidth=2,
               label="TMM (actual)")
        ax.plot(n_range, phase_linear_shifted / np.pi, "r--", linewidth=1.5,
               label="Linear model")
        ax.set_xlabel("Refractive index n")
        ax.set_ylabel("Phase shift (×π)")
        ax.set_title(f"d = {d*1e3:.0f}nm, range = {total_range/np.pi:.2f}π")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Mark 2π
        ax.axhline(y=2, color="green", linestyle=":", alpha=0.5)

        # Mark reflectance variation
        ax2 = ax.twinx()
        ax2.plot(n_range, refl, "gray", linewidth=1, alpha=0.5)
        ax2.set_ylabel("R", color="gray", fontsize=8)
        ax2.set_ylim(0, 1)

    fig.suptitle(f"Phase Response: air | Sb₂Se₃ | Al — λ={wavelength*1e3:.0f}nm\n"
                 f"Blue=TMM, Red=linear, Gray=reflectance",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rcwa_phase_diagnosis.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'rcwa_phase_diagnosis.png'}")

    print(f"\nPhase ranges achieved:")
    for d in thicknesses:
        phase_tmm, refl = tmm_phase_response(wavelength, d, n_range)
        phase_unwrap = np.unwrap(phase_tmm)
        total = abs(phase_unwrap[-1] - phase_unwrap[0])
        print(f"  d={d*1e3:.0f}nm: {total/np.pi:.2f}π  "
              f"R_range=[{refl.min():.3f}, {refl.max():.3f}]")

    return best_d


def design_corrected_levels(wavelength=0.532, film_thickness=0.3, n_levels=8):
    """
    Find the correct n values that give uniformly-spaced phases
    using the actual TMM response (not the linear model).
    """
    print(f"\n{'='*70}")
    print(f"DESIGNING CORRECTED PHASE LEVELS")
    print(f"{'='*70}")

    n_range = np.linspace(N_AMORPHOUS, N_CRYSTALLINE, 2000)
    phase_tmm, refl = tmm_phase_response(wavelength, film_thickness, n_range)
    phase_unwrap = np.unwrap(phase_tmm)
    phase_shifted = phase_unwrap - phase_unwrap[0]

    total_range = abs(phase_shifted[-1])
    print(f"Total phase range at d={film_thickness*1e3:.0f}nm: {total_range/np.pi:.2f}π")

    # Target phases: uniformly spaced over 2π (or over available range if < 2π)
    usable_range = min(total_range, 2 * np.pi)
    target_phases = np.linspace(0, usable_range * (1 - 1 / n_levels), n_levels)

    # Find n values for each target phase by interpolation
    # Phase is monotonic, so we can interpolate
    from scipy.interpolate import interp1d
    n_from_phase = interp1d(phase_shifted, n_range, kind='linear')

    corrected_n = n_from_phase(target_phases)
    actual_phases = np.interp(corrected_n, n_range, phase_shifted)
    actual_refl = np.interp(corrected_n, n_range, refl)

    print(f"\nCorrected n values (TMM-designed):")
    print(f"{'Level':<8} {'n_corrected':<14} {'n_linear':<14} {'Phase (×π)':<14} {'R':<8}")
    print("-" * 58)

    # Linear model values for comparison
    dn_step_linear = wavelength / (2 * n_levels * film_thickness)
    n_linear = N_AMORPHOUS + np.arange(n_levels) * dn_step_linear

    for k in range(n_levels):
        print(f"{k:<8} {corrected_n[k]:<14.4f} {n_linear[k]:<14.4f} "
              f"{actual_phases[k]/np.pi:<14.3f} {actual_refl[k]:<8.3f}")

    print(f"\nReflectance variation: {actual_refl.min():.3f} – {actual_refl.max():.3f}")
    print(f"This amplitude modulation corrupts the pure phase grating!")

    return corrected_n, actual_refl


def sim_blazed_corrected(corrected_n, n_levels=8, wavelength=0.532,
                         film_thickness=0.3, nG=101):
    """Run RCWA with TMM-corrected n values."""
    freq = 1.0 / wavelength
    period = n_levels * SUB_PIXEL_PITCH
    L1 = [period, 0]
    L2 = [0, SUB_PIXEL_PITCH]

    obj = grcwa.obj(nG, L1, L2, freq, 0, 0, verbose=0)
    Nx, Ny = n_levels * 20, 20

    obj.Add_LayerUniform(0, 1.0)
    obj.Add_LayerGrid(film_thickness, Nx, Ny)
    obj.Add_LayerUniform(MIRROR_THICKNESS, eps_al())
    obj.Add_LayerUniform(0, 1.0)

    obj.Init_Setup()
    obj.MakeExcitationPlanewave(0, 0, 1, 0, order=0)

    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    epgrid = np.ones((Nx, Ny), dtype=complex)
    for k in range(n_levels):
        x_lo = k / n_levels
        x_hi = (k + 1) / n_levels
        mask = (X >= x_lo) & (X < x_hi)
        epgrid[mask] = eps_sb2se3(corrected_n[k])

    obj.GridLayer_geteps(epgrid.flatten())

    R_total, T_total = obj.RT_Solve(normalize=1)
    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
    G = obj.G

    idx_1 = np.where((G[:, 0] == 1) & (G[:, 1] == 0))[0]
    idx_0 = np.where((G[:, 0] == 0) & (G[:, 1] == 0))[0]
    eta1 = Ri[idx_1[0]] / R_total if len(idx_1) > 0 else 0
    eta0 = Ri[idx_0[0]] / R_total if len(idx_0) > 0 else 0

    return eta1, eta0, R_total, Ri, G


def compare_linear_vs_corrected():
    """Compare linear model vs TMM-corrected phase levels."""
    print(f"\n{'='*70}")
    print("COMPARISON: Linear vs TMM-Corrected Design")
    print(f"{'='*70}")

    wavelength = 0.532
    film_thickness = 0.3

    # Sweep different film thicknesses to find optimal
    print(f"\n{'d (nm)':<10} {'η₁ linear':<14} {'η₁ corrected':<16} {'η₀ linear':<14} {'η₀ corrected':<16} {'R_total':<10}")
    print("-" * 80)

    thicknesses = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    results = []

    for d in thicknesses:
        # Linear design
        n_levels = 8
        dn_step = wavelength / (2 * n_levels * d)
        n_linear = N_AMORPHOUS + np.arange(n_levels) * dn_step

        if n_linear[-1] > N_CRYSTALLINE:
            n_linear = np.clip(n_linear, N_AMORPHOUS, N_CRYSTALLINE)

        # TMM-corrected design
        corrected_n, _ = design_corrected_levels(wavelength, d, n_levels)

        # RCWA simulations
        eta1_lin, eta0_lin, R_lin, _, _ = sim_blazed_corrected(
            n_linear, n_levels, wavelength, d)
        eta1_cor, eta0_cor, R_cor, _, _ = sim_blazed_corrected(
            corrected_n, n_levels, wavelength, d)

        print(f"{d*1e3:<10.0f} {eta1_lin:<14.4f} {eta1_cor:<16.4f} "
              f"{eta0_lin:<14.4f} {eta0_cor:<16.4f} {R_cor:<10.4f}")
        results.append((d, eta1_lin, eta1_cor, eta0_lin, eta0_cor, R_cor))

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ds = [r[0] * 1e3 for r in results]
    ax1.plot(ds, [r[1] for r in results], "rs--", linewidth=2, markersize=8,
            label="Linear model")
    ax1.plot(ds, [r[2] for r in results], "bo-", linewidth=2, markersize=8,
            label="TMM-corrected")
    ax1.axhline(y=np.sinc(1/8)**2, color="green", linestyle=":", alpha=0.7,
               label="Theoretical max (95%)")
    ax1.set_xlabel("Film Thickness (nm)")
    ax1.set_ylabel("First-Order Efficiency η₁")
    ax1.set_title("Linear vs TMM-Corrected Design")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ds, [r[3] for r in results], "rs--", linewidth=2, markersize=8,
            label="Linear η₀")
    ax2.plot(ds, [r[4] for r in results], "bo-", linewidth=2, markersize=8,
            label="Corrected η₀")
    ax2.set_xlabel("Film Thickness (nm)")
    ax2.set_ylabel("Zero-Order Leakage η₀")
    ax2.set_title("Zero-Order Leakage")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("RCWA: Fixing the Phase Design\n"
                 "Sb₂Se₃ on Al, 8-level grating, λ=532nm", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rcwa_corrected_design.png", dpi=150)
    print(f"\nSaved: {PLOT_DIR / 'rcwa_corrected_design.png'}")


def analyze_amplitude_modulation():
    """
    Even with correct phase levels, the reflectance varies with n.
    This amplitude modulation creates zero-order leakage.
    Quantify the penalty.
    """
    print(f"\n{'='*70}")
    print("AMPLITUDE MODULATION PENALTY")
    print(f"{'='*70}")

    wavelength = 0.532

    for d in [0.20, 0.25, 0.30]:
        n_range = np.linspace(N_AMORPHOUS, N_CRYSTALLINE, 500)
        _, refl = tmm_phase_response(wavelength, d, n_range)

        corrected_n, actual_refl = design_corrected_levels(wavelength, d)
        R_mean = actual_refl.mean()
        R_std = actual_refl.std()
        R_ratio = actual_refl.max() / actual_refl.min()

        # Theoretical penalty from amplitude variation
        # For a phase grating with amplitude modulation a_k:
        # η₁ ≈ sinc²(1/N) × (mean_a)² / (mean_a²)
        # If all a_k equal: ratio = 1 (no penalty)
        a = np.sqrt(actual_refl)
        penalty = a.mean() ** 2 / np.mean(a ** 2)

        print(f"\nd={d*1e3:.0f}nm:")
        print(f"  R range: {actual_refl.min():.3f} – {actual_refl.max():.3f} "
              f"(ratio {R_ratio:.2f})")
        print(f"  Amplitude penalty factor: {penalty:.3f}")
        print(f"  Expected η₁ with penalty: {np.sinc(1/8)**2 * penalty:.3f}")


if __name__ == "__main__":
    print("RCWA Phase Diagnosis & Correction")
    print("=" * 50)
    best_d = diagnose_phase()
    corrected_n, refl = design_corrected_levels()
    compare_linear_vs_corrected()
    analyze_amplitude_modulation()
    print("\nDone!")

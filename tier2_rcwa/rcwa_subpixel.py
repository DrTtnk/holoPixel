"""
Tier 2 RCWA: Periodic Sub-Pixel Array Simulation

Simulates a 2D array of Sb₂Se₃ phase-shifting cells on a reflective
backplane using Rigorous Coupled-Wave Analysis (grcwa).

Key questions:
  1. Diffraction efficiency into designed order vs zero-order leakage
  2. Fill-factor penalty from inter-cell gaps
  3. Effect of phase quantization on diffraction pattern
  4. Wavelength dependence (RGB)
  5. Angular bandwidth of diffracted beams
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import grcwa

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# =============================================================
# Material constants (at λ = 532nm unless noted)
# =============================================================
# Sb₂Se₃ refractive index range
N_AMORPHOUS = 3.0    # fully amorphous
N_CRYSTALLINE = 4.1  # fully crystalline
K_SB2SE3 = 0.01      # low loss

# Aluminum mirror at 532nm
N_AL = 0.92
K_AL = 6.28

# Physical dimensions (µm)
SUB_PIXEL_PITCH = 0.5  # µm
FILM_THICKNESS = 0.3   # µm (300nm Sb₂Se₃)
MIRROR_THICKNESS = 0.1  # µm (100nm Al, opaque)


def eps_sb2se3(n_real, k=K_SB2SE3):
    """Complex permittivity from n, k."""
    return (n_real + 1j * k) ** 2


def eps_al():
    """Complex permittivity of Al at 532nm."""
    return (N_AL + 1j * K_AL) ** 2


# =============================================================
# 1. Blazed Grating — Diffraction Efficiency
# =============================================================
def sim_blazed_grating(n_levels=8, wavelength=0.532, fill_factor=1.0, nG=101):
    """
    Simulate an 8-level blazed grating in reflection.
    Super-cell = n_levels × 0.5µm.
    
    Returns: dict with R per order, efficiency, zero-order, etc.
    """
    freq = 1.0 / wavelength  # grcwa frequency = 1/λ

    # Super-cell period
    period = n_levels * SUB_PIXEL_PITCH
    L1 = [period, 0]
    L2 = [0, SUB_PIXEL_PITCH]  # minimal y-period (quasi-1D)

    # Normal incidence
    theta, phi = 0.0, 0.0

    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)

    Nx, Ny = n_levels * 20, 20  # grid resolution

    # Layer stack: air | Sb₂Se₃ (patterned) | Al mirror | substrate
    obj.Add_LayerUniform(0, 1.0)          # air (incidence medium)
    obj.Add_LayerGrid(FILM_THICKNESS, Nx, Ny)  # Sb₂Se₃ patterned
    obj.Add_LayerUniform(MIRROR_THICKNESS, eps_al())  # Al mirror
    obj.Add_LayerUniform(0, 1.0)          # substrate (irrelevant)

    obj.Init_Setup()

    # Excitation: s-polarized plane wave
    obj.MakeExcitationPlanewave(0, 0, 1, 0, order=0)

    # Build permittivity grid
    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    epgrid = np.ones((Nx, Ny), dtype=complex) * 1.0  # air background (for gaps)

    # Phase target: 0 to 2π across the super-cell
    # Need to find n values that give the right phase in double-pass
    # Phase = 4π·n·d/λ → for step k: φ_k = 2πk/n_levels
    # n_k = n_0 + k·Δn_step, where Δn_step = λ/(2·n_levels·d)
    dn_step = wavelength / (2 * n_levels * FILM_THICKNESS)
    n_values = N_AMORPHOUS + np.arange(n_levels) * dn_step

    # Check range is within Sb₂Se₃ capability
    n_max_needed = n_values[-1]
    assert n_max_needed <= N_CRYSTALLINE, \
        f"Need n_max={n_max_needed:.2f} but Sb₂Se₃ max is {N_CRYSTALLINE}"

    for k in range(n_levels):
        # Fractional x range for this sub-pixel
        x_lo = k / n_levels
        x_hi = (k + fill_factor) / n_levels

        # Active region
        mask = (X >= x_lo) & (X < x_hi)
        epgrid[mask] = eps_sb2se3(n_values[k])

    obj.GridLayer_geteps(epgrid.flatten())

    # Solve
    R_total, T_total = obj.RT_Solve(normalize=1)
    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)

    # Identify diffraction orders
    G = obj.G  # (nG, 2) integer order pairs

    return {
        'R_total': R_total,
        'T_total': T_total,
        'Ri': Ri,
        'Ti': Ti,
        'G': G,
        'n_values': n_values,
        'period': period,
        'wavelength': wavelength,
    }


def analyze_blazed():
    """Analyze blazed grating: efficiency, fill-factor, wavelength."""
    print("=" * 70)
    print("BLAZED GRATING RCWA ANALYSIS")
    print("=" * 70)

    # --- Test 1: Basic 8-level grating at green ---
    result = sim_blazed_grating(n_levels=8, wavelength=0.532, fill_factor=1.0)

    G = result['G']
    Ri = result['Ri']

    # Find the designed first order (m=1, n=0 along x)
    # For a blazed grating with period = 8×0.5µm, the first diffraction order
    # steers light to angle sin(θ) = λ/period = 0.532/4.0 = 0.133 → θ = 7.64°
    print(f"\nBaseline: 8-level blazed grating at λ={result['wavelength']*1e3:.0f}nm")
    print(f"Period: {result['period']:.1f}µm, n range: {result['n_values'][0]:.3f} → {result['n_values'][-1]:.3f}")
    print(f"Total R: {result['R_total']:.4f}, Total T: {result['T_total']:.4f}")
    print(f"R + T = {result['R_total'] + result['T_total']:.4f}")

    # Top 10 orders by reflected power
    sorted_idx = np.argsort(Ri)[::-1]
    print(f"\nTop reflected orders:")
    print(f"{'Order (mx,my)':<16} {'R_frac':<12} {'Angle (°)':<12}")
    print("-" * 40)
    for i in sorted_idx[:10]:
        mx, my = G[i]
        # Diffraction angle: sin(θ) = mx·λ/period_x
        sin_theta = mx * result['wavelength'] / result['period']
        if abs(sin_theta) <= 1:
            angle = np.degrees(np.arcsin(sin_theta))
        else:
            angle = np.nan
        frac = Ri[i] / result['R_total'] if result['R_total'] > 0 else 0
        if Ri[i] > 1e-6:
            print(f"({mx:+3d}, {my:+3d})     {frac:<12.4f} {angle:<12.1f}")

    # Diffraction efficiency into first order
    idx_first = np.where((G[:, 0] == 1) & (G[:, 1] == 0))[0]
    if len(idx_first) > 0:
        eta_first = Ri[idx_first[0]] / result['R_total']
        print(f"\nFirst-order efficiency: η₁ = {eta_first:.1%}")
    idx_zero = np.where((G[:, 0] == 0) & (G[:, 1] == 0))[0]
    if len(idx_zero) > 0:
        eta_zero = Ri[idx_zero[0]] / result['R_total']
        print(f"Zero-order (leakage): η₀ = {eta_zero:.1%}")

    # --- Test 2: Sweep number of phase levels ---
    print(f"\n{'Levels':<10} {'η₁ (1st order)':<18} {'η₀ (zero order)':<18} {'R_total':<10}")
    print("-" * 55)

    levels_list = [2, 4, 6, 8, 12, 16]
    eta_1_list = []
    eta_0_list = []

    for n_lev in levels_list:
        r = sim_blazed_grating(n_levels=n_lev, wavelength=0.532, fill_factor=1.0)
        G = r['G']
        Ri = r['Ri']
        idx_1 = np.where((G[:, 0] == 1) & (G[:, 1] == 0))[0]
        idx_0 = np.where((G[:, 0] == 0) & (G[:, 1] == 0))[0]
        eta1 = Ri[idx_1[0]] / r['R_total'] if len(idx_1) > 0 else 0
        eta0 = Ri[idx_0[0]] / r['R_total'] if len(idx_0) > 0 else 0
        eta_1_list.append(eta1)
        eta_0_list.append(eta0)
        print(f"{n_lev:<10} {eta1:<18.4f} {eta0:<18.4f} {r['R_total']:<10.4f}")

    # Theoretical comparison
    theoretical_eta = [np.sinc(1 / n) ** 2 for n in levels_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(levels_list, eta_1_list, "bo-", linewidth=2, markersize=8,
            label="RCWA (2D)")
    ax1.plot(levels_list, theoretical_eta, "r--", linewidth=2,
            label="Theory: sinc²(1/N)")
    ax1.set_xlabel("Phase Levels")
    ax1.set_ylabel("First-Order Efficiency η₁")
    ax1.set_title("Diffraction Efficiency vs Phase Levels")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(levels_list, eta_0_list, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Phase Levels")
    ax2.set_ylabel("Zero-Order Leakage η₀")
    ax2.set_title("Zero-Order (Blinding) Light")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("RCWA Blazed Grating — Sb₂Se₃ on Al mirror, λ=532nm",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rcwa_blazed_levels.png", dpi=150)
    print(f"\nSaved: {PLOT_DIR / 'rcwa_blazed_levels.png'}")

    # --- Test 3: Fill-factor sweep ---
    print(f"\n{'Fill Factor':<15} {'η₁':<12} {'η₀':<12} {'R_total':<10}")
    print("-" * 50)

    fill_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    eta_1_ff = []
    eta_0_ff = []

    for ff in fill_factors:
        r = sim_blazed_grating(n_levels=8, wavelength=0.532, fill_factor=ff)
        G = r['G']
        Ri = r['Ri']
        idx_1 = np.where((G[:, 0] == 1) & (G[:, 1] == 0))[0]
        idx_0 = np.where((G[:, 0] == 0) & (G[:, 1] == 0))[0]
        eta1 = Ri[idx_1[0]] / r['R_total'] if len(idx_1) > 0 else 0
        eta0 = Ri[idx_0[0]] / r['R_total'] if len(idx_0) > 0 else 0
        eta_1_ff.append(eta1)
        eta_0_ff.append(eta0)
        print(f"{ff:<15.2f} {eta1:<12.4f} {eta0:<12.4f} {r['R_total']:<10.4f}")

    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fill_factors, eta_1_ff, "bo-", linewidth=2, markersize=8,
           label="1st order η₁")
    ax.plot(fill_factors, eta_0_ff, "ro-", linewidth=2, markersize=8,
           label="Zero order η₀")
    ax.set_xlabel("Fill Factor")
    ax.set_ylabel("Efficiency")
    ax.set_title("Fill Factor Impact — 8-level blazed grating, λ=532nm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "rcwa_fill_factor.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'rcwa_fill_factor.png'}")

    # --- Test 4: Wavelength sweep (RGB) ---
    print(f"\n{'Wavelength':<12} {'η₁':<12} {'θ₁ (°)':<12} {'η₀':<12} {'R_total':<10}")
    print("-" * 58)

    wavelengths = np.linspace(0.42, 0.68, 30)
    eta_1_wl = []
    eta_0_wl = []

    for wl in wavelengths:
        r = sim_blazed_grating(n_levels=8, wavelength=wl, fill_factor=1.0)
        G = r['G']
        Ri = r['Ri']
        idx_1 = np.where((G[:, 0] == 1) & (G[:, 1] == 0))[0]
        idx_0 = np.where((G[:, 0] == 0) & (G[:, 1] == 0))[0]
        eta1 = Ri[idx_1[0]] / r['R_total'] if (len(idx_1) > 0 and r['R_total'] > 0) else 0
        eta0 = Ri[idx_0[0]] / r['R_total'] if (len(idx_0) > 0 and r['R_total'] > 0) else 0
        eta_1_wl.append(eta1)
        eta_0_wl.append(eta0)

    # Print RGB values
    for wl_rgb, name in [(0.450, "Blue"), (0.532, "Green"), (0.632, "Red")]:
        r = sim_blazed_grating(n_levels=8, wavelength=wl_rgb, fill_factor=1.0)
        G = r['G']
        Ri = r['Ri']
        idx_1 = np.where((G[:, 0] == 1) & (G[:, 1] == 0))[0]
        idx_0 = np.where((G[:, 0] == 0) & (G[:, 1] == 0))[0]
        eta1 = Ri[idx_1[0]] / r['R_total'] if (len(idx_1) > 0 and r['R_total'] > 0) else 0
        eta0 = Ri[idx_0[0]] / r['R_total'] if (len(idx_0) > 0 and r['R_total'] > 0) else 0
        sin_t = wl_rgb / r['period']
        angle = np.degrees(np.arcsin(sin_t)) if abs(sin_t) <= 1 else float('nan')
        print(f"{name} {wl_rgb*1e3:.0f}nm  {eta1:<12.4f} {angle:<12.1f} {eta0:<12.4f} {r['R_total']:<10.4f}")

    fig3, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wavelengths * 1e3, eta_1_wl, "b-", linewidth=2, label="1st order η₁")
    ax.plot(wavelengths * 1e3, eta_0_wl, "r-", linewidth=2, label="Zero order η₀")
    for wl, col, name in [(450, "blue", "B"), (532, "green", "G"), (632, "red", "R")]:
        ax.axvline(x=wl, color=col, linestyle=":", alpha=0.5)
        ax.text(wl + 3, 0.5, name, color=col, fontsize=12)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Efficiency")
    ax.set_title("Wavelength Dependence — 8-level blazed grating")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(PLOT_DIR / "rcwa_wavelength.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'rcwa_wavelength.png'}")


# =============================================================
# 2. 2D Phase Pattern — Holographic Encoding
# =============================================================
def sim_2d_hologram():
    """
    Simulate a 2D phase pattern encoding a point at angle (15°, 10°).
    The super-cell is 8×8 sub-pixels with a 2D phase ramp.
    """
    print("\n" + "=" * 70)
    print("2D HOLOGRAM — RCWA")
    print("=" * 70)

    wavelength = 0.532
    freq = 1.0 / wavelength
    n_levels = 8
    n_cells = 8  # 8×8 super-cell

    period = n_cells * SUB_PIXEL_PITCH  # 4µm × 4µm
    L1 = [period, 0]
    L2 = [0, period]

    nG = 201  # more orders for 2D

    obj = grcwa.obj(nG, L1, L2, freq, 0, 0, verbose=0)

    Nx, Ny = n_cells * 20, n_cells * 20

    obj.Add_LayerUniform(0, 1.0)          # air
    obj.Add_LayerGrid(FILM_THICKNESS, Nx, Ny)  # Sb₂Se₃
    obj.Add_LayerUniform(MIRROR_THICKNESS, eps_al())  # Al
    obj.Add_LayerUniform(0, 1.0)          # substrate

    obj.Init_Setup()
    obj.MakeExcitationPlanewave(0, 0, 1, 0, order=0)

    # Target angle: (15°, 10°)
    target_tx, target_ty = 15, 10
    kx_target = np.sin(np.radians(target_tx))
    ky_target = np.sin(np.radians(target_ty))

    # Phase ramp across sub-pixels
    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Physical coordinates
    X_phys = X * period
    Y_phys = Y * period

    # Continuous phase
    phase = 2 * np.pi / wavelength * (kx_target * X_phys + ky_target * Y_phys)

    # Quantize to 8 levels
    phase_q = (phase % (2 * np.pi)) / (2 * np.pi) * n_levels
    phase_idx = np.floor(phase_q).astype(int) % n_levels

    # Map to refractive index
    dn_step = wavelength / (2 * n_levels * FILM_THICKNESS)
    n_values = N_AMORPHOUS + np.arange(n_levels) * dn_step

    epgrid = np.zeros((Nx, Ny), dtype=complex)
    for k in range(n_levels):
        mask = phase_idx == k
        epgrid[mask] = eps_sb2se3(n_values[k])

    obj.GridLayer_geteps(epgrid.flatten())

    R_total, T_total = obj.RT_Solve(normalize=1)
    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
    G = obj.G

    print(f"2D hologram: {n_cells}×{n_cells} sub-pixels, period={period:.1f}µm")
    print(f"Target angle: ({target_tx}°, {target_ty}°)")
    print(f"Total R: {R_total:.4f}")

    # Map diffraction orders to angles
    angles_x = np.degrees(np.arcsin(np.clip(G[:, 0] * wavelength / period, -1, 1)))
    angles_y = np.degrees(np.arcsin(np.clip(G[:, 1] * wavelength / period, -1, 1)))

    # Plot diffraction pattern
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Reflected power per order
    valid = Ri > 1e-8
    scatter = ax1.scatter(angles_x[valid], angles_y[valid],
                         c=Ri[valid] / R_total, cmap="hot",
                         s=50, edgecolors="k", linewidth=0.5)
    ax1.plot(target_tx, target_ty, "c+", markersize=20, markeredgewidth=3,
            label=f"Target ({target_tx}°,{target_ty}°)")
    plt.colorbar(scatter, ax=ax1, label="Fraction of reflected power")
    ax1.set_xlabel("θ_x (°)")
    ax1.set_ylabel("θ_y (°)")
    ax1.set_title("Diffraction Orders — Angular Space")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-40, 40)
    ax1.set_ylim(-40, 40)

    # Phase pattern
    ax2.imshow(phase_idx.T, cmap="twilight", origin="lower",
              extent=[0, period, 0, period])
    ax2.set_xlabel("x (µm)")
    ax2.set_ylabel("y (µm)")
    ax2.set_title(f"Phase Pattern (8 levels)\nn: {n_values[0]:.3f} → {n_values[-1]:.3f}")

    fig.suptitle(f"2D Holographic Phase Pattern — RCWA\n"
                 f"Sb₂Se₃ on Al, λ=532nm, 8×8 cells at 0.5µm pitch",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rcwa_2d_hologram.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'rcwa_2d_hologram.png'}")

    # Find efficiency into target order
    # Target order: mx such that sin(θ)=mx*λ/period → mx=period*sin(θ)/λ
    mx_target = round(period * kx_target / wavelength)
    my_target = round(period * ky_target / wavelength)
    idx_target = np.where((G[:, 0] == mx_target) & (G[:, 1] == my_target))[0]
    if len(idx_target) > 0:
        eta_target = Ri[idx_target[0]] / R_total
        print(f"Target order ({mx_target},{my_target}): η = {eta_target:.1%}")

    idx_zero = np.where((G[:, 0] == 0) & (G[:, 1] == 0))[0]
    if len(idx_zero) > 0:
        eta_zero = Ri[idx_zero[0]] / R_total
        print(f"Zero order: η₀ = {eta_zero:.1%}")


if __name__ == "__main__":
    print("Tier 2 RCWA: Sub-Pixel Array Simulation")
    print("=" * 50)
    analyze_blazed()
    sim_2d_hologram()
    print("\nDone!")

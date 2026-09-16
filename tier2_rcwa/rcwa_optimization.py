"""
RCWA Tier 2: Complete Stack Optimization

1. DBR mirror replacing Al → eliminate absorption
2. Anti-reflection coating → reduce Fresnel oscillations  
3. RGB validation → verify design works at all 3 wavelengths
4. E-beam alternative architecture analysis

Film stack: air | (AR coat) | Sb₂Se₃ | DBR mirror | substrate
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import grcwa
import tmm as tmm_lib

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Material constants
N_AMORPHOUS = 3.0
N_CRYSTALLINE = 4.1
K_SB2SE3 = 0.01

# DBR materials
N_TIO2 = 2.3
N_SIO2 = 1.45

# Al mirror
N_AL, K_AL = 0.92, 6.28

# AR coating
N_MGF2 = 1.38  # MgF₂

SUB_PIXEL_PITCH = 0.5  # µm


def eps(n, k=0):
    return (n + 1j * k) ** 2


# =============================================================
# TMM helper: compute phase & reflectance for arbitrary stack
# =============================================================
def stack_response(n_list, d_list, wavelength_nm, n_sb_range):
    """
    Compute phase and reflectance vs Sb₂Se₃ index.
    n_list template has a None placeholder for the Sb₂Se₃ layer.
    d_list has corresponding thicknesses (nm). inf for semi-infinite.
    """
    phases = []
    refls = []
    for n_sb in n_sb_range:
        nl = [n if n is not None else (n_sb + 1j * K_SB2SE3) for n in n_list]
        result = tmm_lib.coh_tmm('s', nl, d_list, 0, wavelength_nm)
        phases.append(np.angle(result['r']))
        refls.append(abs(result['r']) ** 2)
    return np.unwrap(np.array(phases)), np.array(refls)


def design_phase_lut(n_list, d_list, wavelength_nm, n_levels=8):
    """Design TMM-corrected phase LUT for given stack."""
    from scipy.interpolate import interp1d

    n_range = np.linspace(N_AMORPHOUS, N_CRYSTALLINE, 2000)
    phase, refl = stack_response(n_list, d_list, wavelength_nm, n_range)
    phase_shifted = phase - phase[0]

    total_range = abs(phase_shifted[-1])
    usable = min(total_range, 2 * np.pi)
    targets = np.linspace(0, usable * (1 - 1 / n_levels), n_levels)

    n_from_phase = interp1d(phase_shifted, n_range, kind='linear')
    corrected_n = n_from_phase(targets)
    corrected_refl = np.interp(corrected_n, n_range, refl)

    return corrected_n, corrected_refl, total_range


def rcwa_blazed(n_values, film_thickness, wavelength, mirror_eps_layers,
                ar_coat_layers=None, nG=101):
    """
    Run RCWA for blazed grating with arbitrary mirror stack.
    
    mirror_eps_layers: list of (thickness_um, epsilon) tuples for mirror.
    ar_coat_layers: list of (thickness_um, epsilon) tuples for AR coating.
    """
    n_levels = len(n_values)
    freq = 1.0 / wavelength
    period = n_levels * SUB_PIXEL_PITCH
    L1 = [period, 0]
    L2 = [0, SUB_PIXEL_PITCH]

    obj = grcwa.obj(nG, L1, L2, freq, 0, 0, verbose=0)
    Nx, Ny = n_levels * 20, 20

    # Build layer stack
    obj.Add_LayerUniform(0, 1.0)  # air incidence

    # AR coating (if any)
    if ar_coat_layers:
        for t, ep in ar_coat_layers:
            obj.Add_LayerUniform(t, ep)

    # Sb₂Se₃ patterned layer
    obj.Add_LayerGrid(film_thickness, Nx, Ny)

    # Mirror layers
    for t, ep in mirror_eps_layers:
        obj.Add_LayerUniform(t, ep)

    # Substrate
    obj.Add_LayerUniform(0, 1.0)

    obj.Init_Setup()
    obj.MakeExcitationPlanewave(0, 0, 1, 0, order=0)

    # Build permittivity grid
    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    epgrid = np.ones((Nx, Ny), dtype=complex)
    for k in range(n_levels):
        x_lo = k / n_levels
        x_hi = (k + 1) / n_levels
        mask = (X >= x_lo) & (X < x_hi)
        epgrid[mask] = eps(n_values[k], K_SB2SE3)

    obj.GridLayer_geteps(epgrid.flatten())

    R_total, T_total = obj.RT_Solve(normalize=1)
    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
    G = obj.G

    idx_1 = np.where((G[:, 0] == 1) & (G[:, 1] == 0))[0]
    idx_0 = np.where((G[:, 0] == 0) & (G[:, 1] == 0))[0]
    eta1 = Ri[idx_1[0]] / R_total if (len(idx_1) > 0 and R_total > 0) else 0
    eta0 = Ri[idx_0[0]] / R_total if (len(idx_0) > 0 and R_total > 0) else 0

    return {
        'eta1': eta1, 'eta0': eta0, 'R_total': R_total,
        'eta1_abs': Ri[idx_1[0]] if len(idx_1) > 0 else 0,
        'Ri': Ri, 'G': G,
    }


# =============================================================
# 1. DBR Mirror
# =============================================================
def sim_dbr_mirror():
    """Replace Al with DBR (TiO₂/SiO₂ pairs) to eliminate absorption."""
    print("=" * 70)
    print("1. DBR MIRROR — Eliminating Absorption")
    print("=" * 70)

    wavelength = 0.532  # µm
    wavelength_nm = 532
    film_d = 0.25  # µm (optimal from diagnosis)
    film_d_nm = 250

    # DBR quarter-wave thicknesses at 532nm
    d_tio2 = wavelength_nm / (4 * N_TIO2)  # nm
    d_sio2 = wavelength_nm / (4 * N_SIO2)  # nm
    d_tio2_um = d_tio2 / 1e3
    d_sio2_um = d_sio2 / 1e3

    print(f"DBR: TiO₂({d_tio2:.1f}nm) / SiO₂({d_sio2:.1f}nm)")

    configs = []

    # Al mirror baseline
    n_list_al = [1.0, None, N_AL + 1j * K_AL]
    d_list_al = [np.inf, film_d_nm, np.inf]
    n_cor_al, refl_al, range_al = design_phase_lut(n_list_al, d_list_al, wavelength_nm)
    mirror_al = [(0.1, eps(N_AL, K_AL))]
    r_al = rcwa_blazed(n_cor_al, film_d, wavelength, mirror_al)
    configs.append(("Al mirror", r_al, refl_al))
    print(f"\nAl mirror:  η₁={r_al['eta1']:.1%}  η₀={r_al['eta0']:.1%}  "
          f"R_total={r_al['R_total']:.3f}  η₁_abs={r_al['eta1_abs']:.3f}")

    # DBR mirrors: 2, 3, 4, 6 pairs
    for n_pairs in [2, 3, 4, 6]:
        # TMM stack: air | Sb₂Se₃ | (TiO₂ | SiO₂)×n_pairs | substrate
        n_list = [1.0, None]
        d_list = [np.inf, film_d_nm]
        mirror_layers = []
        for _ in range(n_pairs):
            n_list.extend([N_TIO2, N_SIO2])
            d_list.extend([d_tio2, d_sio2])
            mirror_layers.extend([
                (d_tio2_um, eps(N_TIO2)),
                (d_sio2_um, eps(N_SIO2)),
            ])
        n_list.append(1.5)  # glass substrate
        d_list.append(np.inf)

        n_cor, refl, phase_range = design_phase_lut(n_list, d_list, wavelength_nm)
        r = rcwa_blazed(n_cor, film_d, wavelength, mirror_layers)
        configs.append((f"DBR {n_pairs} pairs", r, refl))
        print(f"DBR {n_pairs} pairs: η₁={r['eta1']:.1%}  η₀={r['eta0']:.1%}  "
              f"R_total={r['R_total']:.3f}  η₁_abs={r['eta1_abs']:.3f}  "
              f"range={phase_range/np.pi:.2f}π")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels = [c[0] for c in configs]
    eta1_vals = [c[1]['eta1'] for c in configs]
    eta1_abs = [c[1]['eta1_abs'] for c in configs]
    r_totals = [c[1]['R_total'] for c in configs]
    eta0_vals = [c[1]['eta0'] for c in configs]

    x = range(len(labels))
    width = 0.35
    ax1.bar([i - width/2 for i in x], eta1_vals, width, label='η₁ (of reflected)',
           color='steelblue')
    ax1.bar([i + width/2 for i in x], eta1_abs, width, label='η₁ (absolute)',
           color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Efficiency")
    ax1.set_title("First-Order Efficiency")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.95, color='green', linestyle=':', alpha=0.5, label='95% theory')

    ax2.bar(x, r_totals, color='steelblue', alpha=0.7, label='R_total')
    ax2.bar(x, eta0_vals, color='red', alpha=0.7, label='η₀ (zero-order)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Fraction")
    ax2.set_title("Total Reflectance & Zero-Order Leakage")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle("DBR vs Al Mirror — RCWA, 8-level blazed grating, λ=532nm",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rcwa_dbr_comparison.png", dpi=150)
    print(f"\nSaved: {PLOT_DIR / 'rcwa_dbr_comparison.png'}")

    return configs


# =============================================================
# 2. Anti-Reflection Coating
# =============================================================
def sim_ar_coating():
    """Add MgF₂ AR coating on top of Sb₂Se₃ to reduce Fresnel loss."""
    print("\n" + "=" * 70)
    print("2. ANTI-REFLECTION COATING")
    print("=" * 70)

    wavelength = 0.532
    wavelength_nm = 532
    film_d = 0.25
    film_d_nm = 250

    # DBR mirror (4 pairs)
    d_tio2 = wavelength_nm / (4 * N_TIO2)
    d_sio2 = wavelength_nm / (4 * N_SIO2)
    d_tio2_um = d_tio2 / 1e3
    d_sio2_um = d_sio2 / 1e3
    n_pairs = 4

    mirror_layers = []
    for _ in range(n_pairs):
        mirror_layers.extend([
            (d_tio2_um, eps(N_TIO2)),
            (d_sio2_um, eps(N_SIO2)),
        ])

    # AR coating: MgF₂ quarter-wave
    # Optimal AR thickness depends on Sb₂Se₃ index (which varies!)
    # Design for mid-range n ≈ 3.5
    n_mid = 3.5
    # Ideal AR index = sqrt(n_air × n_sb) = sqrt(3.5) ≈ 1.87
    # MgF₂ (1.38) is not ideal but commonly available
    # Quarter-wave: d_AR = λ/(4×n_AR)
    d_ar = wavelength_nm / (4 * N_MGF2)  # nm
    d_ar_um = d_ar / 1e3

    print(f"MgF₂ AR coating: n={N_MGF2}, d={d_ar:.1f}nm")
    print(f"Ideal AR index for n_mid={n_mid}: {np.sqrt(n_mid):.2f}")

    # Test with different AR thicknesses
    ar_thicknesses = [0, d_ar * 0.5, d_ar * 0.75, d_ar, d_ar * 1.25, d_ar * 1.5]

    results = []
    print(f"\n{'AR thickness (nm)':<20} {'η₁':<10} {'η₀':<10} {'R_total':<10} {'R variation':<12}")
    print("-" * 62)

    for d_ar_test in ar_thicknesses:
        # TMM stack for phase LUT
        n_list = [1.0]
        d_list_tmm = [np.inf]
        ar_rcwa = []

        if d_ar_test > 0:
            n_list.append(N_MGF2)
            d_list_tmm.append(d_ar_test)
            ar_rcwa = [(d_ar_test / 1e3, eps(N_MGF2))]

        n_list.append(None)  # Sb₂Se₃
        d_list_tmm.append(film_d_nm)

        for _ in range(n_pairs):
            n_list.extend([N_TIO2, N_SIO2])
            d_list_tmm.extend([d_tio2, d_sio2])
        n_list.append(1.5)
        d_list_tmm.append(np.inf)

        n_cor, refl, phase_range = design_phase_lut(n_list, d_list_tmm, wavelength_nm)
        r = rcwa_blazed(n_cor, film_d, wavelength, mirror_layers, ar_rcwa)
        r_var = refl.max() / refl.min()

        results.append((d_ar_test, r, refl, r_var))
        print(f"{d_ar_test:<20.1f} {r['eta1']:<10.4f} {r['eta0']:<10.4f} "
              f"{r['R_total']:<10.4f} {r_var:<12.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ar_d = [r[0] for r in results]
    eta1 = [r[1]['eta1'] for r in results]
    eta1_abs = [r[1]['eta1_abs'] for r in results]
    r_total = [r[1]['R_total'] for r in results]

    ax.plot(ar_d, eta1, "bo-", linewidth=2, markersize=8, label="η₁ (of reflected)")
    ax.plot(ar_d, eta1_abs, "rs-", linewidth=2, markersize=8, label="η₁ (absolute)")
    ax.plot(ar_d, r_total, "g^-", linewidth=2, markersize=8, label="R_total")
    ax.set_xlabel("MgF₂ AR Coating Thickness (nm)")
    ax.set_ylabel("Efficiency")
    ax.set_title("Anti-Reflection Coating Effect — 4-pair DBR, λ=532nm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rcwa_ar_coating.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'rcwa_ar_coating.png'}")


# =============================================================
# 3. RGB Validation
# =============================================================
def sim_rgb():
    """Validate the TMM-corrected design at all three RGB wavelengths."""
    print("\n" + "=" * 70)
    print("3. RGB WAVELENGTH VALIDATION")
    print("=" * 70)

    wavelengths = {
        'Blue (450nm)': (0.450, 450),
        'Green (532nm)': (0.532, 532),
        'Red (632nm)': (0.632, 632),
    }
    colors = {'Blue (450nm)': 'blue', 'Green (532nm)': 'green', 'Red (632nm)': 'red'}

    # For each color, use a DBR designed for that wavelength
    n_pairs = 4
    film_thicknesses = [200, 250, 300]  # nm

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, film_d_nm in enumerate(film_thicknesses):
        film_d = film_d_nm / 1e3  # µm for RCWA

        print(f"\n--- Film thickness: {film_d_nm}nm ---")
        print(f"{'Color':<16} {'η₁':<10} {'η₀':<10} {'R_total':<10} {'η₁_abs':<10} "
              f"{'Phase range':<14}")
        print("-" * 70)

        for wl_name, (wl_um, wl_nm) in wavelengths.items():
            # DBR designed for this wavelength
            d_tio2 = wl_nm / (4 * N_TIO2)
            d_sio2 = wl_nm / (4 * N_SIO2)

            n_list = [1.0, None]
            d_list = [np.inf, film_d_nm]
            mirror_layers = []
            for _ in range(n_pairs):
                n_list.extend([N_TIO2, N_SIO2])
                d_list.extend([d_tio2, d_sio2])
                mirror_layers.extend([
                    (d_tio2 / 1e3, eps(N_TIO2)),
                    (d_sio2 / 1e3, eps(N_SIO2)),
                ])
            n_list.append(1.5)
            d_list.append(np.inf)

            n_cor, refl, phase_range = design_phase_lut(n_list, d_list, wl_nm)
            r = rcwa_blazed(n_cor, film_d, wl_um, mirror_layers)

            print(f"{wl_name:<16} {r['eta1']:<10.4f} {r['eta0']:<10.4f} "
                  f"{r['R_total']:<10.4f} {r['eta1_abs']:<10.4f} "
                  f"{phase_range/np.pi:<14.2f}π")

        # Plot for this thickness: wavelength sweep
        wl_sweep = np.linspace(0.42, 0.68, 25)
        for wl_name, (wl_design, wl_design_nm) in wavelengths.items():
            eta1_sweep = []
            # DBR designed for this specific color
            d_tio2 = wl_design_nm / (4 * N_TIO2)
            d_sio2 = wl_design_nm / (4 * N_SIO2)

            n_list = [1.0, None]
            d_list = [np.inf, film_d_nm]
            mirror_layers = []
            for _ in range(n_pairs):
                n_list.extend([N_TIO2, N_SIO2])
                d_list.extend([d_tio2, d_sio2])
                mirror_layers.extend([
                    (d_tio2 / 1e3, eps(N_TIO2)),
                    (d_sio2 / 1e3, eps(N_SIO2)),
                ])
            n_list.append(1.5)
            d_list.append(np.inf)

            # Use LUT designed at the design wavelength
            n_cor, _, _ = design_phase_lut(n_list, d_list, wl_design_nm)

            for wl_test in wl_sweep:
                r = rcwa_blazed(n_cor, film_d, wl_test, mirror_layers)
                eta1_sweep.append(r['eta1_abs'])

            axes[ax_idx].plot(wl_sweep * 1e3, eta1_sweep, '-',
                            color=colors[wl_name], linewidth=2,
                            label=f"DBR@{wl_design_nm}nm")

        axes[ax_idx].set_xlabel("Wavelength (nm)")
        axes[ax_idx].set_ylabel("Absolute η₁")
        axes[ax_idx].set_title(f"d={film_d_nm}nm")
        axes[ax_idx].legend(fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3)

    fig.suptitle("RGB Validation — Absolute First-Order Efficiency\n"
                 "8-level blazed grating, 4-pair DBR mirror, TMM-corrected LUT",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rcwa_rgb_validation.png", dpi=150)
    print(f"\nSaved: {PLOT_DIR / 'rcwa_rgb_validation.png'}")


if __name__ == "__main__":
    print("RCWA Tier 2: Complete Stack Optimization")
    print("=" * 50)
    sim_dbr_mirror()
    sim_ar_coating()
    sim_rgb()
    print("\nDone!")

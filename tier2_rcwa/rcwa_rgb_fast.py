"""
Fast RGB validation — efficiency table only (no wavelength sweep).
Uses the same RCWA infrastructure as rcwa_optimization.py with reduced nG.
"""
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rcwa_optimization import (
    design_phase_lut, rcwa_blazed, eps,
    N_TIO2, N_SIO2, SUB_PIXEL_PITCH, K_SB2SE3, PLOT_DIR,
)

WAVELENGTHS = {
    'Blue (450nm)': (0.450, 450),
    'Green (532nm)': (0.532, 532),
    'Red (632nm)': (0.632, 632),
}
N_PAIRS = 4
FILM_THICKNESSES_NM = [200, 250, 300]


def make_dbr_stack(wl_nm, n_pairs):
    d_tio2 = wl_nm / (4 * N_TIO2)
    d_sio2 = wl_nm / (4 * N_SIO2)
    n_list = [1.0, None]
    d_list = [np.inf, None]          # film_d_nm filled per call
    mirror_layers = []
    for _ in range(n_pairs):
        n_list.extend([N_TIO2, N_SIO2])
        d_list.extend([d_tio2, d_sio2])
        mirror_layers.append((d_tio2 / 1e3, eps(N_TIO2)))
        mirror_layers.append((d_sio2 / 1e3, eps(N_SIO2)))
    n_list.append(1.5)
    d_list.append(np.inf)
    return n_list, d_list, mirror_layers


def run():
    print("RGB WAVELENGTH VALIDATION (fast, nG=51)")
    print("=" * 70)

    for film_d_nm in FILM_THICKNESSES_NM:
        film_d = film_d_nm / 1e3
        print(f"\n--- Film thickness: {film_d_nm}nm ---")
        print(f"{'Color':<16} {'η₁':<10} {'η₀':<10} {'R_total':<10} "
              f"{'η₁_abs':<10} {'Phase range':<14}")
        print("-" * 70)

        for wl_name, (wl_um, wl_nm) in WAVELENGTHS.items():
            n_list, d_list, mirror_layers = make_dbr_stack(wl_nm, N_PAIRS)
            d_list[1] = film_d_nm  # set film thickness in nm for TMM

            n_cor, refl, phase_range = design_phase_lut(n_list, d_list, wl_nm)
            r = rcwa_blazed(n_cor, film_d, wl_um, mirror_layers, nG=51)

            print(f"{wl_name:<16} {r['eta1']:<10.4f} {r['eta0']:<10.4f} "
                  f"{r['R_total']:<10.4f} {r['eta1_abs']:<10.4f} "
                  f"{phase_range / np.pi:<14.2f}π")

    print("\nDone!")


if __name__ == "__main__":
    run()

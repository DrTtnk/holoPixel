"""
2D Hologram RCWA — Reconstruct an 'H' letter in the far field.

Uses a 16×16 sub-pixel super-cell with:
  - Gerchberg-Saxton phase retrieval for the 'H' target
  - TMM-corrected 8-level phase quantization (Sb₂Se₃)
  - 4-pair DBR mirror (TiO₂/SiO₂)
  - RCWA validation of the actual diffraction pattern
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from rcwa_optimization import (
    design_phase_lut, eps, N_TIO2, N_SIO2, N_AMORPHOUS,
    K_SB2SE3, SUB_PIXEL_PITCH, PLOT_DIR,
)
import grcwa

# ── Design parameters ──
WAVELENGTH_UM = 0.532
WAVELENGTH_NM = 532
FILM_D_NM = 300
FILM_D_UM = FILM_D_NM / 1e3
N_CELLS = 16  # 16×16 super-cell
N_LEVELS = 8
N_PAIRS = 4   # DBR pairs
N_GS_ITER = 500  # Gerchberg-Saxton iterations


def define_H_target(n_cells):
    """Define an 'H' letter on the diffraction order grid."""
    target = np.zeros((n_cells, n_cells))

    # 'H' on a 5×7 grid, centered in the 16×16 order space
    # Orders go from -8 to +7, so center is at index 8
    cx, cy = n_cells // 2, n_cells // 2

    # H letter: two vertical strokes + horizontal bar
    # x range: -2 to +2 (5 wide), y range: -3 to +3 (7 tall)
    h_pattern = np.array([
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
    ])

    # Place in center of order grid
    hh, hw = h_pattern.shape
    x0 = cx - hw // 2
    y0 = cy - hh // 2
    target[x0:x0 + hw, y0:y0 + hh] = h_pattern.T

    # Zero out DC (order 0,0) — we don't want a bright center
    target[cx, cy] = 0

    return target


def gerchberg_saxton(target_amp_centered, n_iter):
    """
    Gerchberg-Saxton algorithm for phase-only hologram.
    target_amp_centered: binary target with DC at array center (1=lit, 0=dark).
    Returns continuous phase pattern (0 to 2π).
    """
    # Convert target from centered (DC at N//2) to FFT convention (DC at 0)
    target_fft = np.fft.ifftshift(target_amp_centered)
    ny, nx = target_fft.shape
    N = ny * nx
    mask = target_fft > 0
    n_lit = int(mask.sum())

    # Amplitude normalization: FFT of unit-amplitude field has total energy N².
    # For all energy in n_lit pixels: each gets amplitude N/sqrt(n_lit).
    A_target = N / np.sqrt(n_lit)

    rng = np.random.default_rng(42)
    fourier = np.zeros((ny, nx), dtype=complex)
    fourier[mask] = A_target * np.exp(1j * rng.uniform(0, 2 * np.pi, n_lit))

    for i in range(n_iter):
        field = np.fft.ifft2(fourier)
        field = np.exp(1j * np.angle(field))  # phase-only constraint
        fourier = np.fft.fft2(field)
        # Replace amplitude on target pixels; leave non-target free
        fourier_new = fourier.copy()
        fourier_new[mask] = A_target * np.exp(1j * np.angle(fourier[mask]))
        fourier = fourier_new

        if i % 100 == 0:
            energy_in_target = np.sum(np.abs(fourier[mask]) ** 2)
            energy_total = np.sum(np.abs(fourier) ** 2)
            print(f"      iter {i:4d}: η = {energy_in_target / energy_total:.1%}")

    # Final phase pattern (from the phase-only field, not the constrained fourier)
    field = np.fft.ifft2(fourier)
    phase = np.angle(field) % (2 * np.pi)

    # Report final efficiency of the actual phase-only pattern
    field_final = np.exp(1j * phase)
    ft_final = np.fft.fft2(field_final)
    e_target = np.sum(np.abs(ft_final[mask]) ** 2)
    e_total = np.sum(np.abs(ft_final) ** 2)
    print(f"      final (phase-only): η = {e_target / e_total:.1%}")

    return phase


def quantize_phase(continuous_phase, n_corrected, phase_range):
    """Quantize continuous phase to TMM-corrected n levels."""
    usable = min(phase_range, 2 * np.pi)
    step = usable / len(n_corrected)
    level_idx = np.floor(continuous_phase / step).astype(int)
    level_idx = np.clip(level_idx, 0, len(n_corrected) - 1)
    return level_idx


def build_dbr_stack(wl_nm, n_pairs):
    """Build DBR mirror layer list for RCWA."""
    d_tio2 = wl_nm / (4 * N_TIO2)  # nm
    d_sio2 = wl_nm / (4 * N_SIO2)  # nm
    layers = []
    for _ in range(n_pairs):
        layers.append((d_tio2 / 1e3, eps(N_TIO2)))
        layers.append((d_sio2 / 1e3, eps(N_SIO2)))
    return layers


def build_tmm_stack(wl_nm, film_d_nm, n_pairs):
    """Build TMM stack for phase LUT design."""
    d_tio2 = wl_nm / (4 * N_TIO2)
    d_sio2 = wl_nm / (4 * N_SIO2)
    n_list = [1.0, None]  # air + Sb₂Se₃ placeholder
    d_list = [np.inf, film_d_nm]
    for _ in range(n_pairs):
        n_list.extend([N_TIO2, N_SIO2])
        d_list.extend([d_tio2, d_sio2])
    n_list.append(1.5)  # substrate
    d_list.append(np.inf)
    return n_list, d_list


def rcwa_2d(n_pattern, film_d_um, wavelength_um, mirror_layers, nG=301):
    """
    Run RCWA for a 2D phase pattern.
    n_pattern: NxN array of refractive indices.
    """
    n_cells = n_pattern.shape[0]
    period = n_cells * SUB_PIXEL_PITCH
    freq = 1.0 / wavelength_um

    obj = grcwa.obj(nG, [period, 0], [0, period], freq, 0, 0, verbose=0)
    Nx = Ny = n_cells * 10  # 10 grid points per sub-pixel

    # Layer stack: air → patterned Sb₂Se₃ → DBR → substrate
    obj.Add_LayerUniform(0, 1.0)
    obj.Add_LayerGrid(film_d_um, Nx, Ny)
    for t, ep in mirror_layers:
        obj.Add_LayerUniform(t, ep)
    obj.Add_LayerUniform(0, 1.0)

    obj.Init_Setup()
    obj.MakeExcitationPlanewave(0, 0, 1, 0, order=0)

    # Build permittivity grid from n_pattern
    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    epgrid = np.zeros((Nx, Ny), dtype=complex)
    ppx = 1.0 / n_cells  # fractional pitch per sub-pixel
    for ix in range(n_cells):
        for iy in range(n_cells):
            mask = ((X >= ix * ppx) & (X < (ix + 1) * ppx) &
                    (Y >= iy * ppx) & (Y < (iy + 1) * ppx))
            epgrid[mask] = eps(n_pattern[ix, iy], K_SB2SE3)

    obj.GridLayer_geteps(epgrid.flatten())

    R_total, T_total = obj.RT_Solve(normalize=1)
    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
    G = obj.G

    return R_total, Ri, G


def orders_to_angles(G, wavelength_um, period):
    """Convert grating orders to angles."""
    sin_x = G[:, 0] * wavelength_um / period
    sin_y = G[:, 1] * wavelength_um / period
    valid = (np.abs(sin_x) < 1) & (np.abs(sin_y) < 1)
    angles_x = np.full(len(G), np.nan)
    angles_y = np.full(len(G), np.nan)
    angles_x[valid] = np.degrees(np.arcsin(sin_x[valid]))
    angles_y[valid] = np.degrees(np.arcsin(sin_y[valid]))
    return angles_x, angles_y


def main():
    print("2D Hologram RCWA — 'H' Letter Reconstruction")
    print("=" * 60)

    # 1. Design TMM-corrected phase LUT
    print("\n1. Designing TMM-corrected phase LUT...")
    n_list, d_list = build_tmm_stack(WAVELENGTH_NM, FILM_D_NM, N_PAIRS)
    n_corrected, refl_corrected, phase_range = design_phase_lut(
        n_list, d_list, WAVELENGTH_NM, N_LEVELS
    )
    print(f"   Phase range: {phase_range / np.pi:.2f}π")
    print(f"   n values: {n_corrected}")

    # 2. Define 'H' target
    print("\n2. Defining 'H' target pattern...")
    target = define_H_target(N_CELLS)
    n_lit = int(target.sum())
    print(f"   'H' has {n_lit} lit pixels in {N_CELLS}×{N_CELLS} order grid")

    # 3. Gerchberg-Saxton phase retrieval
    print(f"\n3. Running Gerchberg-Saxton ({N_GS_ITER} iterations)...")
    continuous_phase = gerchberg_saxton(target, N_GS_ITER)

    # Verify GS quality (analytical, no RCWA)
    field_gs = np.exp(1j * continuous_phase)
    # fftshift moves DC to center — matching target which also has DC at center
    ft_gs = np.fft.fftshift(np.fft.fft2(field_gs))
    gs_intensity = np.abs(ft_gs) ** 2
    gs_intensity /= gs_intensity.sum()
    mask_h = target > 0  # target already has DC at center
    eta_gs = gs_intensity[mask_h].sum()
    print(f"   GS efficiency into 'H': {eta_gs:.1%}")

    # 4. Quantize to 8 TMM-corrected levels
    print("\n4. Quantizing to 8 TMM-corrected levels...")
    level_idx = quantize_phase(continuous_phase, n_corrected, phase_range)
    n_pattern = n_corrected[level_idx]
    print(f"   Level distribution: {[int((level_idx == k).sum()) for k in range(N_LEVELS)]}")

    # Quantized analytical check
    field_q = np.exp(1j * level_idx * min(phase_range, 2 * np.pi) / N_LEVELS)
    ft_q = np.fft.fftshift(np.fft.fft2(field_q))
    q_intensity = np.abs(ft_q) ** 2
    q_intensity /= q_intensity.sum()
    eta_q = q_intensity[mask_h].sum()  # mask_h from target (DC at center)
    print(f"   Quantized analytical efficiency into 'H': {eta_q:.1%}")

    # 5. RCWA simulation
    print(f"\n5. Running RCWA (nG=301, {N_CELLS}×{N_CELLS} pattern)...")
    mirror_layers = build_dbr_stack(WAVELENGTH_NM, N_PAIRS)
    R_total, Ri, G = rcwa_2d(n_pattern, FILM_D_UM, WAVELENGTH_UM, mirror_layers)
    print(f"   Total reflectance: {R_total:.4f}")

    # Map RCWA orders to 'H' target
    # Target has DC at center (index N_CELLS//2, N_CELLS//2)
    # RCWA orders (gx, gy) map to target index: (gx + N_CELLS//2, gy + N_CELLS//2)
    period = N_CELLS * SUB_PIXEL_PITCH
    half = N_CELLS // 2
    h_power = 0.0
    zero_power = 0.0
    for i, (gx, gy) in enumerate(G):
        ix = int(gx) + half
        iy = int(gy) + half
        if 0 <= ix < N_CELLS and 0 <= iy < N_CELLS:
            if target[ix, iy] > 0:
                h_power += Ri[i]
        if gx == 0 and gy == 0:
            zero_power = Ri[i]

    eta_rcwa = h_power / R_total if R_total > 0 else 0
    eta_rcwa_abs = h_power
    print(f"   Efficiency into 'H' orders (relative): {eta_rcwa:.1%}")
    print(f"   Efficiency into 'H' orders (absolute): {eta_rcwa_abs:.4f}")
    print(f"   Zero-order power: {zero_power / R_total:.1%}")

    # 6. Build far-field image from RCWA orders
    angles_x, angles_y = orders_to_angles(G, WAVELENGTH_UM, period)
    rcwa_image = np.zeros((N_CELLS, N_CELLS))
    for i, (gx, gy) in enumerate(G):
        ix = int(gx) + half
        iy = int(gy) + half
        if 0 <= ix < N_CELLS and 0 <= iy < N_CELLS:
            rcwa_image[ix, iy] += Ri[i]

    # 7. Plot results
    print("\n6. Generating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Target 'H'
    axes[0, 0].imshow(target.T, cmap='gray', origin='lower',
                      extent=[-N_CELLS // 2, N_CELLS // 2,
                              -N_CELLS // 2, N_CELLS // 2])
    axes[0, 0].set_title("Target: 'H' in order space")
    axes[0, 0].set_xlabel("Order m_x")
    axes[0, 0].set_ylabel("Order m_y")

    # GS continuous phase
    axes[0, 1].imshow(continuous_phase.T, cmap='twilight', origin='lower',
                      extent=[0, period, 0, period])
    axes[0, 1].set_title(f"GS phase (continuous)\nη into H = {eta_gs:.1%}")
    axes[0, 1].set_xlabel("x (µm)")
    axes[0, 1].set_ylabel("y (µm)")

    # Quantized phase (level index)
    im = axes[0, 2].imshow(level_idx.T, cmap='twilight', origin='lower',
                           extent=[0, period, 0, period], vmin=0, vmax=N_LEVELS - 1)
    axes[0, 2].set_title(f"Quantized phase ({N_LEVELS} levels)\nη into H = {eta_q:.1%}")
    axes[0, 2].set_xlabel("x (µm)")
    axes[0, 2].set_ylabel("y (µm)")
    plt.colorbar(im, ax=axes[0, 2], label="Phase level")

    # GS analytical far-field
    axes[1, 0].imshow(np.fft.fftshift(gs_intensity).T, cmap='hot', origin='lower',
                      extent=[-N_CELLS // 2, N_CELLS // 2,
                              -N_CELLS // 2, N_CELLS // 2])
    axes[1, 0].set_title("GS analytical far-field")
    axes[1, 0].set_xlabel("Order m_x")
    axes[1, 0].set_ylabel("Order m_y")

    # Quantized analytical far-field
    axes[1, 1].imshow(np.fft.fftshift(q_intensity).T, cmap='hot', origin='lower',
                      extent=[-N_CELLS // 2, N_CELLS // 2,
                              -N_CELLS // 2, N_CELLS // 2])
    axes[1, 1].set_title("Quantized analytical far-field")
    axes[1, 1].set_xlabel("Order m_x")
    axes[1, 1].set_ylabel("Order m_y")

    # RCWA far-field (already centered via ix = gx + half mapping)
    axes[1, 2].imshow(rcwa_image.T, cmap='hot', origin='lower',
                      extent=[-N_CELLS // 2, N_CELLS // 2,
                              -N_CELLS // 2, N_CELLS // 2])
    axes[1, 2].set_title(f"RCWA far-field\nη into H = {eta_rcwa:.1%} "
                         f"(abs: {eta_rcwa_abs:.3f})")
    axes[1, 2].set_xlabel("Order m_x")
    axes[1, 2].set_ylabel("Order m_y")

    fig.suptitle(
        f"2D Hologram: 'H' Letter Reconstruction\n"
        f"16×16 super-cell, 300nm Sb₂Se₃, 4-pair DBR, λ=532nm, 8 phase levels",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "rcwa_2d_hologram_H.png", dpi=150)
    print(f"\nSaved: {PLOT_DIR / 'rcwa_2d_hologram_H.png'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Target: 'H' letter ({n_lit} lit orders)")
    print(f"Phase range: {phase_range / np.pi:.2f}π")
    print(f"GS continuous η: {eta_gs:.1%}")
    print(f"Quantized (8-level) η: {eta_q:.1%}")
    print(f"RCWA relative η into H: {eta_rcwa:.1%}")
    print(f"RCWA absolute η into H: {eta_rcwa_abs:.4f}")
    print(f"RCWA total reflectance: {R_total:.4f}")
    print(f"Zero-order leakage: {zero_power / R_total:.1%}")


if __name__ == "__main__":
    main()

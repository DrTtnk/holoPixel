"""
Holographic Display — Sub-pixel Count Comparison

Shows how reconstruction quality improves with more sub-pixels per holoxel.
Fixed sub-pixel pitch (0.5µm), varying holoxel size → DPI/angular tradeoff.

Configs (all with ±32° FoV due to fixed 0.5µm pitch):
  32×32  → 16µm holoxel, 16 angular bins per side
  64×64  → 32µm holoxel, 32 angular bins
  128×128 → 64µm holoxel, 64 angular bins
  338×338 → 169µm holoxel, 169 angular bins (credit-card spec)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

SUBPIXEL_PITCH = 0.5e-6
N_PHASE_LEVELS = 8
LAMBDA = 532e-9


def compute_single_hogel_farfield(n_sub, scene_points):
    """Compute the far-field of one holoxel at display center encoding full scene."""
    hogel_size = n_sub * SUBPIXEL_PITCH
    k = 2 * np.pi / LAMBDA

    si = (np.arange(n_sub) - (n_sub - 1) / 2) * SUBPIXEL_PITCH
    sub_x, sub_y = np.meshgrid(si, si, indexing='ij')

    E = np.zeros((n_sub, n_sub), dtype=np.complex128)
    for (px, py, pz, amp) in scene_points:
        dist = np.sqrt((sub_x - px) ** 2 + (sub_y - py) ** 2 + pz ** 2)
        E += amp * np.exp(1j * k * dist) / dist

    # Phase-only + quantization
    phase = np.angle(E) % (2 * np.pi)
    levels = np.round(phase / (2 * np.pi) * N_PHASE_LEVELS) % N_PHASE_LEVELS
    quantized = levels * (2 * np.pi / N_PHASE_LEVELS)

    # Far-field via zero-padded FFT for smooth angular plot
    pad = max(512, n_sub * 2)
    E_q = np.exp(1j * quantized)
    ff = np.fft.fftshift(np.fft.fft2(E_q, s=(pad, pad)))
    I_ff = np.abs(ff) ** 2

    # Angular axes
    freq = np.fft.fftshift(np.fft.fftfreq(pad, d=SUBPIXEL_PITCH))
    sin_theta = freq * LAMBDA  # sin(θ) = m·λ/d for each FFT bin
    theta_deg = np.degrees(np.arcsin(np.clip(sin_theta, -1, 1)))

    return I_ff, theta_deg, quantized


def make_display(n_hx, n_hy, n_sub):
    pitch = n_sub * SUBPIXEL_PITCH
    hx = (np.arange(n_hx) - (n_hx - 1) / 2) * pitch
    hy = (np.arange(n_hy) - (n_hy - 1) / 2) * pitch
    hogel_cx, hogel_cy = np.meshgrid(hx, hy, indexing='ij')
    si = (np.arange(n_sub) - (n_sub - 1) / 2) * SUBPIXEL_PITCH
    sub_dx, sub_dy = np.meshgrid(si, si, indexing='ij')
    return hogel_cx, hogel_cy, sub_dx, sub_dy


def compute_and_render_chunked(n_hx, n_hy, n_sub, scene_points, observers,
                                chunk_rows=16):
    """Compute phases and render views, processing in row chunks to limit memory."""
    hogel_cx, hogel_cy, sub_dx, sub_dy = make_display(n_hx, n_hy, n_sub)
    hogel_size = n_sub * SUBPIXEL_PITCH
    k = 2 * np.pi / LAMBDA

    n_obs = len(observers)
    images = [np.zeros((n_hx, n_hy), dtype=np.complex128) for _ in range(n_obs)]

    for row_start in range(0, n_hx, chunk_rows):
        row_end = min(row_start + chunk_rows, n_hx)
        n_rows = row_end - row_start

        cx_chunk = hogel_cx[row_start:row_end]  # (n_rows, n_hy)
        cy_chunk = hogel_cy[row_start:row_end]

        # Compute phases for this chunk
        E = np.zeros((n_rows, n_hy, n_sub, n_sub), dtype=np.complex64)
        for (px, py, pz, amp) in scene_points:
            dx = cx_chunk[:, :, None, None] + sub_dx[None, None, :, :] - px
            dy = cy_chunk[:, :, None, None] + sub_dy[None, None, :, :] - py
            dist = np.sqrt(dx ** 2 + dy ** 2 + pz ** 2)
            E += (amp * np.exp(1j * k * dist) / dist).astype(np.complex64)

        phase = np.angle(E) % (2 * np.pi)
        levels = np.round(phase / (2 * np.pi) * N_PHASE_LEVELS) % N_PHASE_LEVELS
        quantized = (levels * (2 * np.pi / N_PHASE_LEVELS)).astype(np.float32)

        # FFT per holoxel
        E_q = np.exp(1j * quantized.astype(np.float64))
        ff = np.fft.fft2(E_q, axes=(-2, -1))

        # Sample far-field at each observer direction
        for oi, (ox, oy, oz) in enumerate(observers):
            dx_obs = ox - cx_chunk
            dy_obs = oy - cy_chunk
            dist_obs = np.sqrt(dx_obs ** 2 + dy_obs ** 2 + oz ** 2)
            m_x = (np.round(dx_obs / dist_obs * hogel_size / LAMBDA).astype(int)) % n_sub
            m_y = (np.round(dy_obs / dist_obs * hogel_size / LAMBDA).astype(int)) % n_sub
            idx_r = np.arange(n_rows)[:, None]
            idx_c = np.arange(n_hy)[None, :]
            images[oi][row_start:row_end] = ff[idx_r, idx_c, m_x, m_y]

    return images, hogel_cx, hogel_cy


def main():
    # ── Scene ──
    scene_points = [
        (2e-3, 0.0, 30e-3, 1.0),   # NEAR
        (0.0, 0.0, 80e-3, 1.0),     # MID
        (-2e-3, 0.0, 250e-3, 1.0),  # FAR
    ]

    configs = [
        (32, 'tab:blue'),
        (64, 'tab:orange'),
        (128, 'tab:green'),
        (338, 'tab:red'),
    ]

    # ═══════════════════════════════════════════════════
    # Part 1: Single holoxel far-field comparison
    # ═══════════════════════════════════════════════════
    print("=" * 50)
    print("Part 1: Single holoxel far-field")
    print("=" * 50)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for col, (n_sub, color) in enumerate(configs):
        print(f"  n_sub={n_sub}...", end=" ", flush=True)
        t0 = time.time()
        I_ff, theta_deg, phase_map = compute_single_hogel_farfield(n_sub, scene_points)
        dt = time.time() - t0
        print(f"{dt:.2f}s")

        hogel_um = n_sub * SUBPIXEL_PITCH * 1e6
        n_bins = n_sub // 2

        # Top: far-field 2D
        ax = axes[0, col]
        extent = [theta_deg[0], theta_deg[-1], theta_deg[0], theta_deg[-1]]
        vmax = np.percentile(I_ff, 99.5)
        ax.imshow(I_ff.T, cmap='inferno', aspect='equal',
                  extent=extent, origin='lower', vmin=0, vmax=vmax * 0.3)
        ax.set_xlim(-35, 35)
        ax.set_ylim(-35, 35)
        ax.set_title(f'{n_sub}×{n_sub} sub-px\n'
                     f'hogel={hogel_um:.0f}µm, {n_bins} bins/side',
                     fontsize=10, color=color, fontweight='bold')
        if col == 0:
            ax.set_ylabel('θ_y (°)', fontsize=10)
        ax.set_xlabel('θ_x (°)', fontsize=10)

        # Mark expected point directions from center holoxel
        for (px, py, pz, _) in scene_points:
            theta_x = np.degrees(np.arctan2(px, pz))
            theta_y = np.degrees(np.arctan2(py, pz))
            ax.plot(theta_x, theta_y, '+', color='white', markersize=10,
                    markeredgewidth=1.5)

        # Bottom: horizontal slice through θ_y=0
        ax = axes[1, col]
        mid = I_ff.shape[1] // 2
        # Average a few rows around center for smoother slice
        I_slice = I_ff[:, mid - 2:mid + 3].mean(axis=1)
        I_slice_norm = I_slice / I_slice.max()
        ax.plot(theta_deg, I_slice_norm, color=color, linewidth=1.5)
        ax.set_xlim(-35, 35)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('θ_x (°)', fontsize=10)
        if col == 0:
            ax.set_ylabel('Normalized intensity', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Mark point directions
        for (px, py, pz, _) in scene_points:
            theta_x = np.degrees(np.arctan2(px, pz))
            ax.axvline(theta_x, color='gray', linestyle='--', alpha=0.5)

        # Measure peak widths
        for (px, py, pz, _) in scene_points:
            theta_target = np.degrees(np.arctan2(px, pz))
            idx = np.argmin(np.abs(theta_deg - theta_target))
            # Find FWHM around this peak
            peak_val = I_slice[idx]
            half = peak_val / 2
            left = idx
            while left > 0 and I_slice[left] > half:
                left -= 1
            right = idx
            while right < len(I_slice) - 1 and I_slice[right] > half:
                right += 1
            fwhm = theta_deg[right] - theta_deg[left]
            if abs(theta_target) < 1:  # Only annotate center peak
                ax.annotate(f'FWHM={fwhm:.1f}°', xy=(theta_target, 0.5),
                            fontsize=8, ha='center', color=color)

    plt.suptitle('Single Holoxel Far-Field — More Sub-pixels = Sharper Peaks\n'
                 'White + marks = expected point directions | '
                 'Dashed lines = point angles',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out1 = PLOT_DIR / "holo_subpixel_comparison_farfield.png"
    fig.savefig(out1, dpi=150)
    plt.close()
    print(f"✅ Saved: {out1}")

    # ═══════════════════════════════════════════════════
    # Part 2: Full display comparison (small display)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("Part 2: Full display reconstruction")
    print("=" * 50)

    # Use ~4mm display width for all configs
    display_width_target = 4e-3  # 4mm
    obs_z = 400e-3
    observers = [
        (-10e-3, 0.0, obs_z),
        (0.0, 0.0, obs_z),
        (10e-3, 0.0, obs_z),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    for col, (n_sub, color) in enumerate(configs):
        hogel_pitch = n_sub * SUBPIXEL_PITCH
        n_hx = max(8, int(display_width_target / hogel_pitch))
        n_hy = max(6, int(n_hx * 9 / 16))  # 16:9 aspect
        actual_width = n_hx * hogel_pitch * 1e3
        total_sub = n_hx * n_hy * n_sub * n_sub
        dpi = 25.4e-3 / hogel_pitch

        print(f"  n_sub={n_sub}: {n_hx}×{n_hy} holoxels, "
              f"{total_sub/1e6:.1f}M sub-pixels, {dpi:.0f} DPI...",
              end=" ", flush=True)

        t0 = time.time()
        chunk = max(1, min(n_hx, 64 * 64 // (n_sub * n_sub // 64 + 1)))
        chunk = min(chunk, n_hx)
        imgs, hcx, hcy = compute_and_render_chunked(
            n_hx, n_hy, n_sub, scene_points, observers,
            chunk_rows=max(1, min(n_hx, 16))
        )
        dt = time.time() - t0
        print(f"{dt:.1f}s")

        hx_mm = hcx[:, 0] * 1e3
        hy_mm = hcy[0, :] * 1e3
        extent = [hx_mm[0], hx_mm[-1], hy_mm[0], hy_mm[-1]]

        for row, (img, (ox, oy, oz)) in enumerate(zip(imgs, observers)):
            ax = axes[row, col]
            I = np.abs(img.T) ** 2
            vmax = np.percentile(I, 99) if I.max() > 0 else 1
            ax.imshow(I, cmap='inferno', aspect='equal',
                      extent=extent, origin='lower', vmin=0,
                      vmax=max(vmax * 0.5, 1e-30))
            if col == 0:
                ax.set_ylabel(f'obs x={ox*1e3:+.0f}mm', fontsize=10)
            if row == 0:
                ax.set_title(f'{n_sub}×{n_sub}\n{n_hx}×{n_hy} hogels\n'
                             f'{dpi:.0f} DPI', fontsize=9,
                             color=color, fontweight='bold')

    plt.suptitle(f'Display Reconstruction — Same ~4mm Display Area\n'
                 f'More sub-pixels → fewer holoxels (lower DPI) but sharper '
                 f'angular control | 3 points at z=30/80/250mm',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out2 = PLOT_DIR / "holo_subpixel_comparison_display.png"
    fig.savefig(out2, dpi=150)
    plt.close()
    print(f"✅ Saved: {out2}")

    # ═══════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUB-PIXEL COUNT COMPARISON")
    print("=" * 70)
    print(f"{'Sub-px':>8} {'Hogel':>8} {'DPI':>8} {'Ang bins':>10} "
          f"{'FoV':>8} {'Ang res':>10}")
    print("-" * 60)
    for n_sub, _ in configs:
        hogel_um = n_sub * SUBPIXEL_PITCH * 1e6
        dpi = 25.4e-3 / (n_sub * SUBPIXEL_PITCH)
        n_bins = n_sub // 2
        fov = np.degrees(np.arcsin(min(1, n_bins * LAMBDA / (n_sub * SUBPIXEL_PITCH))))
        ang_res = np.degrees(np.arcsin(LAMBDA / (n_sub * SUBPIXEL_PITCH)))
        print(f"{n_sub:>5}²  {hogel_um:>6.0f}µm {dpi:>7.0f} {n_bins:>6}/side "
              f"  ±{fov:>4.1f}°   {ang_res:>6.2f}°")

    print(f"\nKey insight: sub-pixel pitch is fixed at {SUBPIXEL_PITCH*1e6:.1f}µm")
    print(f"  More sub-pixels → bigger holoxel → lower DPI → fewer pixels")
    print(f"  BUT: each pixel has better angular control → sharper 3D points")
    print(f"  This is the fundamental SPATIAL vs ANGULAR resolution tradeoff")
    print(f"\n  Credit card spec (150 DPI) → 338×338 sub-pixels per holoxel")
    print(f"  506×319 holoxels = 161,714 pixels (≈144p) with 169 angular bins")


if __name__ == "__main__":
    main()

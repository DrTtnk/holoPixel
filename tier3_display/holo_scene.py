"""
Holographic Display — Full Scene Simulation

Scene: Letters at different depths, rendered as point-source holograms.
Uses Fresnel separable approximation for fast computation.
Grayscale (monochromatic λ=532nm) rendering.

Credit card spec: 506×319 holoxels, 338×338 sub-pixels, 150 DPI.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

SUBPIXEL_PITCH = 0.5e-6       # 0.5 µm
N_PHASE_LEVELS = 8             # 3-bit quantization
LAMBDA = 532e-9                # green laser

# ────────────────────────────────────────────────────────
# Letter bitmaps (each row is a string, '#' = lit pixel)
# ────────────────────────────────────────────────────────
LETTER_H = [
    "#....#",
    "#....#",
    "#....#",
    "######",
    "#....#",
    "#....#",
    "#....#",
    "#....#",
]

LETTER_D = [
    "####..",
    "#...#.",
    "#....#",
    "#....#",
    "#....#",
    "#....#",
    "#...#.",
    "####..",
]

LETTER_O = [
    ".####.",
    "#....#",
    "#....#",
    "#....#",
    "#....#",
    "#....#",
    "#....#",
    ".####.",
]

LETTER_L = [
    "#.....",
    "#.....",
    "#.....",
    "#.....",
    "#.....",
    "#.....",
    "#.....",
    "######",
]


def bitmap_to_points(bitmap, center_x, center_y, z, amp, spacing_mm=1.5):
    """Convert a letter bitmap to 3D point sources."""
    rows = len(bitmap)
    cols = len(bitmap[0])
    spacing = spacing_mm * 1e-3

    points = []
    for r, row in enumerate(bitmap):
        for c, ch in enumerate(row):
            if ch == '#':
                x = center_x + (c - (cols - 1) / 2) * spacing
                y = center_y + ((rows - 1) / 2 - r) * spacing  # flip y
                points.append((x, y, z, amp))
    return points


def make_display(n_hx, n_hy, n_sub):
    hogel_pitch = n_sub * SUBPIXEL_PITCH
    hx = (np.arange(n_hx) - (n_hx - 1) / 2) * hogel_pitch
    hy = (np.arange(n_hy) - (n_hy - 1) / 2) * hogel_pitch
    hogel_cx, hogel_cy = np.meshgrid(hx, hy, indexing='ij')
    return hogel_cx, hogel_cy, hogel_pitch


def compute_and_render(n_hx, n_hy, n_sub, scene_points, observers,
                       chunk_rows=4):
    """
    Compute holographic phases using Fresnel separable approximation,
    then render observer views via single-bin DFT.

    Returns list of (n_hx, n_hy) complex images, one per observer.
    """
    hogel_cx, hogel_cy, hogel_pitch = make_display(n_hx, n_hy, n_sub)
    hogel_size = n_sub * SUBPIXEL_PITCH
    k = 2 * np.pi / LAMBDA

    # Sub-pixel 1D arrays
    dx_arr = (np.arange(n_sub) - (n_sub - 1) / 2) * SUBPIXEL_PITCH
    dy_arr = dx_arr.copy()
    dx2 = dx_arr ** 2
    dy2 = dy_arr ** 2

    n_obs = len(observers)
    images = [np.zeros((n_hx, n_hy), dtype=np.complex128) for _ in range(n_obs)]

    n_pts = len(scene_points)
    total_chunks = (n_hx + chunk_rows - 1) // chunk_rows
    t_start = time.time()

    for ci, row_start in enumerate(range(0, n_hx, chunk_rows)):
        row_end = min(row_start + chunk_rows, n_hx)
        n_rows = row_end - row_start

        cx = hogel_cx[row_start:row_end]  # (n_rows, n_hy)
        cy = hogel_cy[row_start:row_end]

        # Accumulate complex field using Fresnel separable approximation
        E = np.zeros((n_rows, n_hy, n_sub, n_sub), dtype=np.complex64)

        for px, py, pz, amp in scene_points:
            Dx = cx - px  # (n_rows, n_hy)
            Dy = cy - py
            R0 = np.sqrt(Dx ** 2 + Dy ** 2 + pz ** 2)
            sin_tx = Dx / R0
            sin_ty = Dy / R0
            inv_2R0 = 0.5 / R0

            # Scalar per holoxel
            scalar = (amp / R0 * np.exp(1j * k * R0)).astype(np.complex64)

            # 1D phase arrays: (n_rows, n_hy, n_sub)
            phase_x = k * (sin_tx[:, :, None] * dx_arr[None, None, :]
                           + inv_2R0[:, :, None] * dx2[None, None, :])
            fx = np.exp(1j * phase_x).astype(np.complex64)

            phase_y = k * (sin_ty[:, :, None] * dy_arr[None, None, :]
                           + inv_2R0[:, :, None] * dy2[None, None, :])
            fy = np.exp(1j * phase_y).astype(np.complex64)

            # Rank-1 accumulation: E += scalar * outer(fx, fy)
            E += scalar[:, :, None, None] * fx[:, :, :, None] * fy[:, :, None, :]

        # Phase-only quantization
        phase = np.angle(E) % (2 * np.pi)
        levels = np.round(phase / (2 * np.pi) * N_PHASE_LEVELS) % N_PHASE_LEVELS
        quantized = (levels * (2 * np.pi / N_PHASE_LEVELS)).astype(np.float32)
        del E, phase, levels  # free memory

        # Render: FFT + sample at observer direction
        E_q = np.exp(1j * quantized.astype(np.float64))
        del quantized
        ff = np.fft.fft2(E_q, axes=(-2, -1))
        del E_q

        for oi, (ox, oy, oz) in enumerate(observers):
            dx_obs = ox - cx
            dy_obs = oy - cy
            dist_obs = np.sqrt(dx_obs ** 2 + dy_obs ** 2 + oz ** 2)
            m_x = (np.round(dx_obs / dist_obs * hogel_size / LAMBDA).astype(int)
                   ) % n_sub
            m_y = (np.round(dy_obs / dist_obs * hogel_size / LAMBDA).astype(int)
                   ) % n_sub
            idx_r = np.arange(n_rows)[:, None]
            idx_c = np.arange(n_hy)[None, :]
            images[oi][row_start:row_end] = ff[idx_r, idx_c, m_x, m_y]

        del ff

        elapsed = time.time() - t_start
        eta = elapsed / (ci + 1) * (total_chunks - ci - 1)
        print(f"\r  Chunk {ci+1}/{total_chunks} | "
              f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s", end="", flush=True)

    print()
    return images, hogel_cx, hogel_cy


def validate_fresnel(n_sub=64):
    """Quick check that Fresnel matches exact computation for one holoxel."""
    k = 2 * np.pi / LAMBDA
    dx_arr = (np.arange(n_sub) - (n_sub - 1) / 2) * SUBPIXEL_PITCH
    sub_x, sub_y = np.meshgrid(dx_arr, dx_arr, indexing='ij')

    px, py, pz = 5e-3, 2e-3, 50e-3

    # Exact
    dist = np.sqrt((sub_x - px) ** 2 + (sub_y - py) ** 2 + pz ** 2)
    E_exact = np.exp(1j * k * dist) / dist

    # Fresnel separable (holoxel at origin)
    R0 = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    sin_tx = -px / R0  # Dx = 0 - px
    sin_ty = -py / R0
    inv_2R0 = 0.5 / R0
    scalar = np.exp(1j * k * R0) / R0

    phase_x = k * (sin_tx * dx_arr + inv_2R0 * dx_arr ** 2)
    fx = np.exp(1j * phase_x)
    phase_y = k * (sin_ty * dx_arr + inv_2R0 * dx_arr ** 2)
    fy = np.exp(1j * phase_y)
    E_fresnel = scalar * fx[:, None] * fy[None, :]

    # Compare phases (what matters for phase-only modulation)
    phase_exact = np.angle(E_exact)
    phase_fresnel = np.angle(E_fresnel)
    phase_error = np.angle(np.exp(1j * (phase_exact - phase_fresnel)))
    rms_error = np.sqrt(np.mean(phase_error ** 2))
    max_error = np.max(np.abs(phase_error))

    print(f"Fresnel validation (n_sub={n_sub}, point at 5mm,2mm,50mm):")
    print(f"  Phase RMS error: {np.degrees(rms_error):.3f}°")
    print(f"  Phase max error: {np.degrees(max_error):.3f}°")
    return rms_error < 0.1  # < 5.7° is fine


def main():
    # ── Validate Fresnel approximation ──
    print("=" * 60)
    print("Fresnel approximation validation")
    print("=" * 60)
    assert validate_fresnel(64), "Fresnel approximation failed!"
    assert validate_fresnel(338), "Fresnel approximation failed at full scale!"
    print("✅ Fresnel separable approximation verified\n")

    # ── Scene: Letters "HOLO" at two depths ──
    print("=" * 60)
    print("Creating scene")
    print("=" * 60)

    # "HO" near the display (z=50mm) → low parallax
    pts_H = bitmap_to_points(LETTER_H, -12e-3, 0, 50e-3, 1.0, spacing_mm=1.8)
    pts_O_near = bitmap_to_points(LETTER_O, -2e-3, 0, 50e-3, 1.0, spacing_mm=1.8)

    # "LO" far from display (z=200mm) → high parallax
    pts_L = bitmap_to_points(LETTER_L, 8e-3, 0, 200e-3, 1.0, spacing_mm=1.8)
    pts_O_far = bitmap_to_points(LETTER_O, 18e-3, 0, 200e-3, 1.0, spacing_mm=1.8)

    scene_points = pts_H + pts_O_near + pts_L + pts_O_far

    print(f"  H: {len(pts_H)} points at z=50mm")
    print(f"  O: {len(pts_O_near)} points at z=50mm")
    print(f"  L: {len(pts_L)} points at z=200mm")
    print(f"  O: {len(pts_O_far)} points at z=200mm")
    print(f"  Total: {len(scene_points)} point sources")

    # ── Display configuration ──
    n_sub = 338  # credit card spec
    hogel_pitch = n_sub * SUBPIXEL_PITCH  # 169µm
    dpi = 25.4e-3 / hogel_pitch

    # Start with 256×144 (manageable), then offer to scale
    n_hx, n_hy = 256, 144
    display_w = n_hx * hogel_pitch * 1e3
    display_h = n_hy * hogel_pitch * 1e3
    total_sub = n_hx * n_hy * n_sub ** 2

    print(f"\n  Display: {n_hx}×{n_hy} holoxels at {dpi:.0f} DPI")
    print(f"  Size: {display_w:.1f} × {display_h:.1f} mm")
    print(f"  Sub-pixels: {n_sub}×{n_sub} per holoxel = {total_sub/1e9:.2f}B total")

    # ── Observer positions ──
    obs_z = 400e-3
    obs_positions = [
        (-20e-3, 0, obs_z),
        (-10e-3, 0, obs_z),
        (0, 0, obs_z),
        (10e-3, 0, obs_z),
        (20e-3, 0, obs_z),
    ]

    # ── Compute ──
    print(f"\n{'='*60}")
    print(f"Computing holographic phases + rendering")
    print(f"{'='*60}")

    t0 = time.time()
    # Chunk size: balance memory and speed
    # Each chunk: chunk_rows × 144 × 338 × 338 × 8 bytes (complex64)
    # chunk_rows=8: 8 × 144 × 114K × 8 = 1.05 GB per chunk
    images, hogel_cx, hogel_cy = compute_and_render(
        n_hx, n_hy, n_sub, scene_points, obs_positions,
        chunk_rows=8
    )
    dt = time.time() - t0
    print(f"  Total: {dt:.1f}s ({dt/60:.1f} min)")

    # ── Plot 1: Observer views (grayscale) ──
    hx_mm = hogel_cx[:, 0] * 1e3
    hy_mm = hogel_cy[0, :] * 1e3
    extent = [hx_mm[0], hx_mm[-1], hy_mm[0], hy_mm[-1]]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, (img, (ox, oy, oz)) in enumerate(zip(images, obs_positions)):
        ax = axes[i]
        I = np.abs(img.T) ** 2
        vmax = np.percentile(I, 99.5)
        ax.imshow(I, cmap='gray', aspect='equal', extent=extent,
                  origin='lower', vmin=0, vmax=vmax * 0.4)
        ax.set_title(f'obs x={ox*1e3:+.0f}mm', fontsize=11)
        ax.set_xlabel('x (mm)')
        if i == 0:
            ax.set_ylabel('y (mm)')

    plt.suptitle(f'Holographic Display — "HOLO" Scene\n'
                 f'"HO" at z=50mm (near, steady) | "LO" at z=200mm (far, shifts)\n'
                 f'{n_hx}×{n_hy} holoxels, {n_sub}×{n_sub} sub-px, '
                 f'λ=532nm monochromatic',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out1 = PLOT_DIR / "holo_scene_views.png"
    fig.savefig(out1, dpi=150)
    plt.close()
    print(f"✅ Saved: {out1}")

    # ── Plot 2: Parallax cross-section ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: horizontal slices at y=0 for all observer positions
    ax = axes[0]
    mid_y = n_hy // 2
    for img, (ox, oy, oz) in zip(images, obs_positions):
        I_row = np.abs(img[:, mid_y]) ** 2
        I_norm = I_row / I_row.max() if I_row.max() > 0 else I_row
        ax.plot(hx_mm, I_norm, linewidth=1.5,
                label=f'x={ox*1e3:+.0f}mm')
    ax.set_xlabel('Display x (mm)', fontsize=11)
    ax.set_ylabel('Normalized intensity', fontsize=11)
    ax.set_title('Horizontal Cross-Section (y=0)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: difference between left and right views
    ax = axes[1]
    I_left = np.abs(images[0].T) ** 2
    I_right = np.abs(images[-1].T) ** 2
    I_center = np.abs(images[2].T) ** 2
    vmax = np.percentile(I_center, 99.5)

    diff = I_right - I_left
    vabs = np.percentile(np.abs(diff), 99)
    ax.imshow(diff, cmap='RdBu_r', aspect='equal', extent=extent,
              origin='lower', vmin=-vabs, vmax=vabs)
    ax.set_title('Right − Left view (red=brighter right, blue=brighter left)',
                 fontsize=10)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    plt.suptitle(f'Parallax Analysis — "HOLO"\n'
                 f'Near letters (z=50mm) barely shift | '
                 f'Far letters (z=200mm) shift significantly',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out2 = PLOT_DIR / "holo_scene_parallax.png"
    fig.savefig(out2, dpi=150)
    plt.close()
    print(f"✅ Saved: {out2}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SCENE SUMMARY")
    print(f"{'='*60}")
    print(f"Display: {n_hx}×{n_hy} holoxels ({display_w:.1f}×{display_h:.1f}mm)")
    print(f"Sub-pixels: {n_sub}×{n_sub} per holoxel ({dpi:.0f} DPI)")
    print(f"Total sub-pixels: {total_sub/1e9:.2f} billion")
    print(f"Scene: {len(scene_points)} point sources")
    print(f"  NEAR (z=50mm): 'HO' — parallax slope = "
          f"{50/(obs_z*1e3+50):.4f}")
    print(f"  FAR (z=200mm): 'LO' — parallax slope = "
          f"{200/(obs_z*1e3+200):.4f}")
    print(f"Computation: {dt:.1f}s")


if __name__ == "__main__":
    main()

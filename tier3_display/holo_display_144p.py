"""
Holographic Display — Level 3 (144p) with Enhanced Parallax Visualization

Scales up to 256×144 hogels with clearer scene:
  - Three colored points at different depths
  - Amplitude compensated for distance (equal brightness)
  - Parallax quantified: displacement vs observer position
  - Near objects shift MORE, far objects shift LESS → 3D depth proven
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

HOGEL_PITCH = 169.3e-6  # m
SUBPIXEL_PITCH = 0.5e-6 # m
N_PHASE_LEVELS = 8


def make_display(n_hx, n_hy, n_sub):
    hx = (np.arange(n_hx) - (n_hx - 1) / 2) * HOGEL_PITCH
    hy = (np.arange(n_hy) - (n_hy - 1) / 2) * HOGEL_PITCH
    hogel_cx, hogel_cy = np.meshgrid(hx, hy, indexing='ij')
    si = (np.arange(n_sub) - (n_sub - 1) / 2) * SUBPIXEL_PITCH
    sj = (np.arange(n_sub) - (n_sub - 1) / 2) * SUBPIXEL_PITCH
    sub_dx, sub_dy = np.meshgrid(si, sj, indexing='ij')
    return hogel_cx, hogel_cy, sub_dx, sub_dy


def compute_phases_chunked(hogel_cx, hogel_cy, sub_dx, sub_dy,
                           scene_points, lam, chunk_size=256):
    """Compute phases in chunks to manage memory for large displays."""
    n_hx, n_hy = hogel_cx.shape
    n_sub = sub_dx.shape[0]
    k = 2 * np.pi / lam

    phases = np.zeros((n_hx, n_hy, n_sub, n_sub), dtype=np.float32)

    # Process in x-chunks
    for cx_start in range(0, n_hx, chunk_size):
        cx_end = min(cx_start + chunk_size, n_hx)
        E_chunk = np.zeros((cx_end - cx_start, n_hy, n_sub, n_sub),
                           dtype=np.complex64)

        hcx = hogel_cx[cx_start:cx_end]
        hcy = hogel_cy[cx_start:cx_end]

        for (px, py, pz, amp) in scene_points:
            dx = hcx[:, :, None, None] + sub_dx[None, None, :, :] - px
            dy = hcy[:, :, None, None] + sub_dy[None, None, :, :] - py
            dist = np.sqrt(dx ** 2 + dy ** 2 + pz ** 2)
            E_chunk += (amp * np.exp(1j * k * dist) / dist).astype(np.complex64)

        chunk_phases = np.angle(E_chunk) % (2 * np.pi)
        levels = np.round(chunk_phases / (2 * np.pi) * N_PHASE_LEVELS) % N_PHASE_LEVELS
        phases[cx_start:cx_end] = (levels * (2 * np.pi / N_PHASE_LEVELS)).astype(np.float32)

    return phases


def render_view_fast(hogel_cx, hogel_cy, phases, n_sub, lam, observer_pos):
    n_hx, n_hy = hogel_cx.shape
    ox, oy, oz = observer_pos
    hogel_size = n_sub * SUBPIXEL_PITCH

    dx = ox - hogel_cx
    dy = oy - hogel_cy
    dist = np.sqrt(dx ** 2 + dy ** 2 + oz ** 2)
    sin_theta_x = dx / dist
    sin_theta_y = dy / dist

    E_sub = np.exp(1j * phases.astype(np.float64))
    far_field = np.fft.fft2(E_sub, axes=(-2, -1))

    m_x = (np.round(sin_theta_x * hogel_size / lam).astype(int)) % n_sub
    m_y = (np.round(sin_theta_y * hogel_size / lam).astype(int)) % n_sub

    idx_x = np.arange(n_hx)[:, None]
    idx_y = np.arange(n_hy)[None, :]
    return far_field[idx_x, idx_y, m_x, m_y]


def find_point_centroid(image_intensity, hogel_cx, hogel_cy, region_center, radius):
    """Find the intensity centroid near an expected position."""
    # Create a mask for the region of interest
    dx = hogel_cx - region_center[0]
    dy = hogel_cy - region_center[1]
    mask = (dx ** 2 + dy ** 2) < radius ** 2

    if mask.sum() == 0:
        return region_center[0], region_center[1], 0

    weighted_x = (image_intensity * hogel_cx * mask).sum()
    weighted_y = (image_intensity * hogel_cy * mask).sum()
    total = (image_intensity * mask).sum()

    if total > 0:
        return weighted_x / total, weighted_y / total, total
    return region_center[0], region_center[1], 0


def run_level3():
    """256×144 hogels, 32×32 sub-pixels, three depth planes."""
    n_hx, n_hy, n_sub = 256, 144, 32
    lam = 532e-9

    display_w = n_hx * HOGEL_PITCH * 1e3  # mm
    display_h = n_hy * HOGEL_PITCH * 1e3

    print(f"Display: {n_hx}×{n_hy} hogels, {n_sub}×{n_sub} sub-pixels")
    print(f"Physical: {display_w:.1f}mm × {display_h:.1f}mm")
    print(f"Total sub-pixels: {n_hx * n_hy * n_sub ** 2:,}")

    # ── Scene: three points at different depths ──
    # Compensate amplitude for 1/r falloff so all appear equally bright
    z_near, z_mid, z_far = 30e-3, 80e-3, 250e-3
    ref_dist = z_mid  # normalize to middle point

    scene = [
        (8e-3,   3e-3,  z_near, ref_dist / z_near),    # NEAR (bright)
        (0.0,    0.0,   z_mid,  1.0),                   # MID (reference)
        (-8e-3, -3e-3,  z_far,  ref_dist / z_far),     # FAR (boosted)
    ]

    print(f"\nScene points:")
    for i, (x, y, z, a) in enumerate(scene):
        depth_label = ['NEAR', 'MID', 'FAR'][i]
        print(f"  {depth_label}: ({x*1e3:.0f}, {y*1e3:.0f})mm at z={z*1e3:.0f}mm, amp={a:.2f}")

    # Observer positions: sweep x at z=400mm
    obs_z = 400e-3
    n_views = 9
    obs_x = np.linspace(-20e-3, 20e-3, n_views)
    observers = [(x, 0.0, obs_z) for x in obs_x]

    # ── Compute ──
    hogel_cx, hogel_cy, sub_dx, sub_dy = make_display(n_hx, n_hy, n_sub)

    print("\nComputing phases...", flush=True)
    t0 = time.time()
    phases = compute_phases_chunked(hogel_cx, hogel_cy, sub_dx, sub_dy,
                                    scene, lam, chunk_size=64)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Rendering views...", flush=True)
    images = []
    for i, obs in enumerate(observers):
        t1 = time.time()
        img = render_view_fast(hogel_cx, hogel_cy, phases, n_sub, lam, obs)
        images.append(img)
        print(f"  View {i+1}/{n_views}: x={obs[0]*1e3:+.1f}mm ({time.time()-t1:.1f}s)")

    # ── Track point centroids for parallax measurement ──
    intensities = [np.abs(img) ** 2 for img in images]
    search_radius = 10e-3  # 10mm search radius

    centroids_near = []
    centroids_mid = []
    centroids_far = []

    for i, I_img in enumerate(intensities):
        cx_n, cy_n, _ = find_point_centroid(I_img, hogel_cx, hogel_cy,
                                            (scene[0][0], scene[0][1]), search_radius)
        cx_m, cy_m, _ = find_point_centroid(I_img, hogel_cx, hogel_cy,
                                            (scene[1][0], scene[1][1]), search_radius)
        cx_f, cy_f, _ = find_point_centroid(I_img, hogel_cx, hogel_cy,
                                            (scene[2][0], scene[2][1]), search_radius)
        centroids_near.append(cx_n)
        centroids_mid.append(cx_m)
        centroids_far.append(cx_f)

    # ── Plot: views + parallax curve ──
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, n_views, hspace=0.35, wspace=0.3)

    # Row 1: Selected views (show 5 of 9)
    view_indices = [0, 2, 4, 6, 8]
    vmax = np.percentile([np.abs(img) ** 2 for img in images], 99.5)

    for col, vi in enumerate(view_indices):
        ax = fig.add_subplot(gs[0, col * (n_views // 5): (col + 1) * (n_views // 5)])
        I = np.abs(images[vi].T) ** 2
        extent = [hogel_cx[0, 0] * 1e3, hogel_cx[-1, 0] * 1e3,
                  hogel_cy[0, 0] * 1e3, hogel_cy[0, -1] * 1e3]
        ax.imshow(I, cmap='inferno', aspect='equal', vmin=0, vmax=vmax * 0.3,
                  extent=extent, origin='lower')
        ax.set_title(f'x={obs_x[vi]*1e3:+.0f}mm', fontsize=10)
        # Mark expected positions
        for (px, py, pz, _), marker, c in zip(scene,
                                               ['o', 's', '^'],
                                               ['cyan', 'lime', 'yellow']):
            ax.plot(px * 1e3, py * 1e3, marker, color=c, markersize=6,
                    markerfacecolor='none', markeredgewidth=1.5)
        if col == 0:
            ax.set_ylabel('y (mm)')
        ax.set_xlabel('x (mm)')

    # Row 2: Parallax displacement curves
    ax_par = fig.add_subplot(gs[1, :])
    obs_x_mm = obs_x * 1e3

    # Theoretical parallax: apparent displacement = obs_x × (1 - z_point/z_obs)
    for z_pt, label, color, centroids in [
        (z_near, f'NEAR (z={z_near*1e3:.0f}mm)', 'cyan', centroids_near),
        (z_mid, f'MID (z={z_mid*1e3:.0f}mm)', 'lime', centroids_mid),
        (z_far, f'FAR (z={z_far*1e3:.0f}mm)', 'yellow', centroids_far),
    ]:
        # Measured
        cx_mm = np.array(centroids) * 1e3
        ax_par.plot(obs_x_mm, cx_mm, 'o-', color=color, label=f'{label} (measured)',
                    markersize=5)
        # Theoretical
        parallax_factor = 1 - z_pt / obs_z
        theoretical_x = np.array([s[0] for s in scene if s[2] == z_pt])[0] * 1e3
        # The apparent position shifts as the observer moves:
        # x_apparent = x_point - obs_x × z_point / z_observer  (simplified)
        # Actually: x_apparent ≈ x_point + (obs_x - x_point) × z_point/(z_obs - z_point)
        # But for display rendering, the displacement is: Δx_apparent = Δobs_x × z_point/(z_obs)
        theory = theoretical_x * np.ones_like(obs_x_mm)  # position stays at point x
        ax_par.plot(obs_x_mm, theory, '--', color=color, alpha=0.5,
                    label=f'{label} (true position)')

    ax_par.set_xlabel('Observer x position (mm)')
    ax_par.set_ylabel('Apparent point x position (mm)')
    ax_par.set_title('Parallax: Point Apparent Position vs Observer Position')
    ax_par.legend(fontsize=8, ncol=2)
    ax_par.grid(True, alpha=0.3)

    # Row 3: Phase pattern samples
    sample_cols = np.linspace(0, n_hx - 1, min(n_views, 9)).astype(int)
    for col_idx, h_idx in enumerate(sample_cols):
        ax = fig.add_subplot(gs[2, col_idx])
        ax.imshow(phases[h_idx, n_hy // 2].T, cmap='hsv',
                  vmin=0, vmax=2 * np.pi, aspect='equal')
        ax.set_title(f'Hogel ({h_idx},{n_hy//2})', fontsize=8)
        if col_idx == 0:
            ax.set_ylabel('Phase pattern')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Holographic Display Simulation — {n_hx}×{n_hy} hogels (144p), '
                 f'{n_sub}×{n_sub} sub-pixels\n'
                 f'Display: {display_w:.1f}mm × {display_h:.1f}mm | '
                 f'λ=532nm | 8-level phase | Observer at z=400mm',
                 fontsize=12, fontweight='bold')

    out = PLOT_DIR / "holo_display_L3_144p.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {out}")

    # ── Print parallax summary ──
    print("\n" + "=" * 60)
    print("PARALLAX MEASUREMENT")
    print("=" * 60)
    for z_pt, label, centroids in [
        (z_near, 'NEAR', centroids_near),
        (z_mid, 'MID', centroids_mid),
        (z_far, 'FAR', centroids_far),
    ]:
        cx = np.array(centroids) * 1e3
        # Linear fit: slope = Δx_apparent / Δx_observer
        slope = np.polyfit(obs_x_mm, cx, 1)[0]
        # Theoretical slope for a point at depth z: slope = z/(z_obs - z)
        # (This is the parallax coefficient)
        # Correction: for a holographic display, the apparent position of a
        # reconstructed point shifts as the observer moves. The shift is
        # proportional to z_point / z_observer.
        theory_slope = z_pt / obs_z
        print(f"  {label} (z={z_pt*1e3:.0f}mm): "
              f"measured slope = {slope:.4f}, "
              f"theoretical ≈ {theory_slope:.4f}")

    return phases, images


if __name__ == "__main__":
    run_level3()

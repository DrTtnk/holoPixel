"""
Holographic Display — Clean Parallax Demonstration

Renders each point source INDEPENDENTLY to cleanly measure parallax.
Then combines them for the full scene view.

Key visualization: horizontal cross-sections showing how each point
shifts at a different rate as the observer moves → proving 3D depth.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

HOGEL_PITCH = 169.3e-6
SUBPIXEL_PITCH = 0.5e-6
N_PHASE_LEVELS = 8


def make_display(n_hx, n_hy, n_sub):
    hx = (np.arange(n_hx) - (n_hx - 1) / 2) * HOGEL_PITCH
    hy = (np.arange(n_hy) - (n_hy - 1) / 2) * HOGEL_PITCH
    hogel_cx, hogel_cy = np.meshgrid(hx, hy, indexing='ij')
    si = (np.arange(n_sub) - (n_sub - 1) / 2) * SUBPIXEL_PITCH
    sj = (np.arange(n_sub) - (n_sub - 1) / 2) * SUBPIXEL_PITCH
    sub_dx, sub_dy = np.meshgrid(si, sj, indexing='ij')
    return hogel_cx, hogel_cy, sub_dx, sub_dy


def compute_phases(hogel_cx, hogel_cy, sub_dx, sub_dy, scene_points, lam):
    n_hx, n_hy = hogel_cx.shape
    n_sub = sub_dx.shape[0]
    k = 2 * np.pi / lam
    E = np.zeros((n_hx, n_hy, n_sub, n_sub), dtype=np.complex64)

    for (px, py, pz, amp) in scene_points:
        dx = hogel_cx[:, :, None, None] + sub_dx[None, None, :, :] - px
        dy = hogel_cy[:, :, None, None] + sub_dy[None, None, :, :] - py
        dist = np.sqrt(dx ** 2 + dy ** 2 + pz ** 2)
        E += (amp * np.exp(1j * k * dist) / dist).astype(np.complex64)

    phases = np.angle(E) % (2 * np.pi)
    levels = np.round(phases / (2 * np.pi) * N_PHASE_LEVELS) % N_PHASE_LEVELS
    return (levels * (2 * np.pi / N_PHASE_LEVELS)).astype(np.float32)


def render_views(hogel_cx, hogel_cy, phases, n_sub, lam, observers):
    n_hx, n_hy = hogel_cx.shape
    hogel_size = n_sub * SUBPIXEL_PITCH
    E_sub = np.exp(1j * phases.astype(np.float64))
    far_field = np.fft.fft2(E_sub, axes=(-2, -1))

    images = []
    for ox, oy, oz in observers:
        dx = ox - hogel_cx
        dy = oy - hogel_cy
        dist = np.sqrt(dx ** 2 + dy ** 2 + oz ** 2)
        m_x = (np.round(dx / dist * hogel_size / lam).astype(int)) % n_sub
        m_y = (np.round(dy / dist * hogel_size / lam).astype(int)) % n_sub
        idx_x = np.arange(n_hx)[:, None]
        idx_y = np.arange(n_hy)[None, :]
        images.append(far_field[idx_x, idx_y, m_x, m_y])

    return images


def find_peak_x(intensity_row, x_coords):
    """Find peak position using weighted average around maximum."""
    idx_max = np.argmax(intensity_row)
    # Use 5-pixel window around max
    lo = max(0, idx_max - 2)
    hi = min(len(intensity_row), idx_max + 3)
    weights = intensity_row[lo:hi]
    total = weights.sum()
    if total > 0:
        return (weights * x_coords[lo:hi]).sum() / total
    return x_coords[idx_max]


def main():
    n_hx, n_hy, n_sub = 256, 144, 32
    lam = 532e-9

    hogel_cx, hogel_cy, sub_dx, sub_dy = make_display(n_hx, n_hy, n_sub)
    hx_mm = hogel_cx[:, 0] * 1e3  # x coordinates in mm

    # ── Scene: 3 points at different depths ──
    points = {
        'NEAR (z=30mm)': (6e-3, 0.0, 30e-3, 1.0),
        'MID (z=80mm)': (0.0, 0.0, 80e-3, 1.0),
        'FAR (z=250mm)': (-6e-3, 0.0, 250e-3, 1.0),
    }
    colors = {'NEAR (z=30mm)': 'cyan', 'MID (z=80mm)': 'lime', 'FAR (z=250mm)': 'gold'}

    # Observer sweep
    obs_z = 400e-3
    n_views = 11
    obs_x = np.linspace(-20e-3, 20e-3, n_views)
    observers = [(x, 0.0, obs_z) for x in obs_x]
    obs_x_mm = obs_x * 1e3

    # ── Render each point independently ──
    point_images = {}
    for name, (px, py, pz, amp) in points.items():
        print(f"Processing {name}...", flush=True)
        t0 = time.time()
        phases = compute_phases(hogel_cx, hogel_cy, sub_dx, sub_dy,
                                [(px, py, pz, amp)], lam)
        imgs = render_views(hogel_cx, hogel_cy, phases, n_sub, lam, observers)
        point_images[name] = imgs
        print(f"  Done ({time.time()-t0:.1f}s)")

    # ── Also render the combined scene ──
    print("Processing COMBINED...", flush=True)
    t0 = time.time()
    all_points = list(points.values())
    phases_all = compute_phases(hogel_cx, hogel_cy, sub_dx, sub_dy,
                                all_points, lam)
    imgs_all = render_views(hogel_cx, hogel_cy, phases_all, n_sub, lam, observers)
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ── Extract horizontal cross-sections and track peaks ──
    mid_row = n_hy // 2
    peak_tracks = {}

    for name, imgs in point_images.items():
        peaks = []
        for img in imgs:
            I_row = np.abs(img[:, mid_row]) ** 2
            peak_x = find_peak_x(I_row, hx_mm)
            peaks.append(peak_x)
        peak_tracks[name] = np.array(peaks)

    # ── PLOT 1: Cross-section waterfall ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax_idx, (name, imgs) in enumerate(point_images.items()):
        ax = axes[ax_idx]
        color = colors[name]

        for i, (img, ox_mm) in enumerate(zip(imgs, obs_x_mm)):
            I_row = np.abs(img[:, mid_row]) ** 2
            I_norm = I_row / I_row.max() if I_row.max() > 0 else I_row
            offset = i * 1.2  # vertical offset for waterfall
            ax.fill_between(hx_mm, offset, offset + I_norm * 1.0,
                            alpha=0.4, color=color)
            ax.plot(hx_mm, offset + I_norm * 1.0, color=color, linewidth=0.5)
            ax.text(hx_mm[-1] + 0.5, offset + 0.3, f'x={ox_mm:+.0f}',
                    fontsize=7, va='center')

        # Draw line connecting peaks
        peaks = peak_tracks[name]
        for i in range(len(peaks)):
            ax.plot(peaks[i], i * 1.2 + 0.5, 'k+', markersize=8, markeredgewidth=2)

        ax.set_xlabel('Display x (mm)')
        ax.set_ylabel('Observer position →')
        ax.set_title(name, fontsize=11, color=color, fontweight='bold')
        ax.set_yticks([])
        ax.set_xlim(hx_mm[0], hx_mm[-1] + 3)

    plt.suptitle('Holographic Parallax — Horizontal Cross-Sections\n'
                 'Each row = observer at different x position | '
                 '+ marks = peak intensity (should shift at different rates)',
                 fontsize=11)
    plt.tight_layout()
    out1 = PLOT_DIR / "holo_parallax_waterfall.png"
    fig.savefig(out1, dpi=150)
    plt.close()
    print(f"✅ Saved: {out1}")

    # ── PLOT 2: Parallax displacement curves (the money plot) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, peaks in peak_tracks.items():
        color = colors[name]
        # Displacement relative to center view
        center_idx = n_views // 2
        displacement = peaks - peaks[center_idx]
        ax.plot(obs_x_mm, displacement, 'o-', color=color, label=name,
                markersize=6, linewidth=2)

    # Theoretical lines
    for (name, (px, py, pz, _)) in points.items():
        # Phase pattern exp(+ikR) produces diverging (real-image) wavefront.
        # The brightest hogel for observer at ox satisfies:
        #   (hx - px)/pz = (ox - hx)/oz  →  hx = (pz·ox + oz·px)/(oz + pz)
        # Slope: d(hx)/d(ox) = pz / (oz + pz)
        slope = pz / (obs_z + pz)
        theory_disp = obs_x_mm * slope
        ax.plot(obs_x_mm, theory_disp, '--', color=colors[name], alpha=0.5, linewidth=1)

    ax.set_xlabel('Observer x position (mm)', fontsize=11)
    ax.set_ylabel('Peak displacement from center (mm)', fontsize=11)
    ax.set_title('Parallax: Peak Shift vs Observer Motion', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    # Add slope annotations
    for name, peaks in peak_tracks.items():
        disp = peaks - peaks[n_views // 2]
        slope_measured = np.polyfit(obs_x_mm, disp, 1)[0]
        pz = [v[2] for k, v in points.items() if k == name][0]
        slope_theory = pz / (obs_z + pz)
        ax.text(0.02, 0.98 - list(peak_tracks.keys()).index(name) * 0.08,
                f'{name}: slope={slope_measured:.4f} (theory={slope_theory:.4f})',
                transform=ax.transAxes, fontsize=8, va='top', color=colors[name])

    # ── Right panel: combined views (3 positions) ──
    ax = axes[1]
    view_indices = [0, n_views // 2, n_views - 1]
    for vi in view_indices:
        I_row = np.abs(imgs_all[vi][:, mid_row]) ** 2
        I_norm = I_row / I_row.max() if I_row.max() > 0 else I_row
        ax.plot(hx_mm, I_norm, linewidth=1.5,
                label=f'observer x={obs_x_mm[vi]:+.0f}mm')

    ax.set_xlabel('Display x (mm)', fontsize=11)
    ax.set_ylabel('Normalized intensity', fontsize=11)
    ax.set_title('Combined Scene: 3 Views', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Holographic Parallax Proof — 256×144 hogels, 32×32 sub-px\n'
                 f'slope = z_point / (z_obs + z_point) — steeper = farther from display',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out2 = PLOT_DIR / "holo_parallax_proof.png"
    fig.savefig(out2, dpi=150)
    plt.close()
    print(f"✅ Saved: {out2}")

    # ── PLOT 3: Side-by-side observer views ──
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    view_indices = [0, 2, 5, 8, 10]

    extent = [hx_mm[0], hx_mm[-1],
              hogel_cy[0, 0] * 1e3, hogel_cy[0, -1] * 1e3]

    # Row 1: individual point intensities summed
    for col, vi in enumerate(view_indices):
        ax = axes[0, col]
        I_combined = np.zeros((n_hx, n_hy))
        for name, imgs in point_images.items():
            I_combined += np.abs(imgs[vi]) ** 2
        vmax = np.percentile(I_combined, 99)
        ax.imshow(I_combined.T, cmap='inferno', aspect='equal',
                  extent=extent, origin='lower', vmin=0, vmax=vmax * 0.5)
        ax.set_title(f'x={obs_x_mm[vi]:+.0f}mm', fontsize=11)
        if col == 0:
            ax.set_ylabel('Individual sum')

        # Mark peak positions
        for name, imgs in point_images.items():
            I_row = np.abs(imgs[vi][:, mid_row]) ** 2
            peak = find_peak_x(I_row, hx_mm)
            ax.plot(peak, 0, 'v', color=colors[name], markersize=8,
                    markeredgecolor='white', markeredgewidth=0.5)

    # Row 2: holographic combined
    for col, vi in enumerate(view_indices):
        ax = axes[1, col]
        I = np.abs(imgs_all[vi].T) ** 2
        vmax = np.percentile(I, 99)
        ax.imshow(I, cmap='inferno', aspect='equal',
                  extent=extent, origin='lower', vmin=0, vmax=vmax * 0.5)
        if col == 0:
            ax.set_ylabel('Holographic combined')

    plt.suptitle('Observer Views — Top: individual point sum | '
                 'Bottom: holographic (phase-only) reconstruction\n'
                 'Arrows mark peak positions for each depth',
                 fontsize=11)
    plt.tight_layout()
    out3 = PLOT_DIR / "holo_parallax_views.png"
    fig.savefig(out3, dpi=150)
    plt.close()
    print(f"✅ Saved: {out3}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PARALLAX QUANTIFICATION")
    print("=" * 60)
    print(f"{'Point':<20} {'Depth':>8} {'Measured slope':>15} {'Theory':>10} {'Match':>8}")
    print("-" * 65)
    for name, peaks in peak_tracks.items():
        disp = peaks - peaks[n_views // 2]
        slope_m = np.polyfit(obs_x_mm, disp, 1)[0]
        pz = [v[2] for k, v in points.items() if k == name][0]
        slope_t = pz / (obs_z + pz)
        match = abs(slope_m - slope_t) / slope_t * 100
        status = "✅" if match < 30 else "⚠"
        print(f"{name:<20} {pz*1e3:>6.0f}mm {slope_m:>15.4f} {slope_t:>10.4f} "
              f"{status} {match:.0f}%")

    print(f"\nInterpretation:")
    print(f"  slope = z_p / (z_obs + z_p) — diverging wavefront (real image)")
    print(f"  slope > 0 → peak shifts WITH observer motion")
    print(f"  steeper slope → point is FARTHER from display (closer to observer)")
    print(f"  NEAR (30mm):  slope = 30/(400+30) = {30/430:.4f}")
    print(f"  MID (80mm):   slope = 80/(400+80) = {80/480:.4f}")
    print(f"  FAR (250mm):  slope = 250/(400+250) = {250/650:.4f}")
    print(f"  Ratio FAR/NEAR = {(250/650)/(30/430):.1f}× — far point shows "
          f"{(250/650)/(30/430):.1f}× more parallax")


if __name__ == "__main__":
    main()

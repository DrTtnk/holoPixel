"""
Holographic Display Simulator — v2

Improvements over v1:
  1. Continuous phase (no quantization) option for artifact-free rendering
  2. Multiprocessing across holoxel row chunks (24 cores)
  3. Anti-aliased observer rendering: average neighboring FFT bins
  4. Proper Cornell box scene with point light and shadows

Physics: Each holoxel encodes the full scene light field via phase-only
modulation (Fresnel separable CGH). The observer sees one angular sample
per holoxel → the display image.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import math
import time
import sys
from tqdm import tqdm
import numba

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

SUBPIXEL_PITCH = 0.5e-6
LAMBDA = 532e-9


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cornell Box Scene Generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_cornell_box(n_points_per_wall=400, box_size=15e-3, depth_center=100e-3,
                     light_pos=None):
    """
    Create a Cornell box scene as a point cloud.

    Box centered at (0, 0, depth_center) with given half-size.
    Point light at light_pos casts shadows via ray tracing.

    Returns list of (x, y, z, amplitude) tuples.
    """
    hs = box_size  # half-size
    z0 = depth_center

    if light_pos is None:
        light_pos = np.array([0, hs * 0.8, z0 - hs * 0.5])

    rng = np.random.RandomState(42)
    points = []

    # Wall definitions: (normal, point_on_wall, u_dir, v_dir, u_range, v_range, name)
    walls = [
        # Back wall (z = z0 + hs)
        ('back', np.array([0, 0, -1]), np.array([0, 0, z0 + hs]),
         np.array([1, 0, 0]), np.array([0, 1, 0]),
         (-hs, hs), (-hs, hs)),
        # Floor (y = -hs)
        ('floor', np.array([0, 1, 0]), np.array([0, -hs, z0]),
         np.array([1, 0, 0]), np.array([0, 0, 1]),
         (-hs, hs), (-hs, hs)),
        # Ceiling (y = +hs)
        ('ceiling', np.array([0, -1, 0]), np.array([0, hs, z0]),
         np.array([1, 0, 0]), np.array([0, 0, 1]),
         (-hs, hs), (-hs, hs)),
        # Left wall (x = -hs) - RED
        ('left', np.array([1, 0, 0]), np.array([-hs, 0, z0]),
         np.array([0, 1, 0]), np.array([0, 0, 1]),
         (-hs, hs), (-hs, hs)),
        # Right wall (x = +hs) - GREEN
        ('right', np.array([-1, 0, 0]), np.array([hs, 0, z0]),
         np.array([0, 1, 0]), np.array([0, 0, 1]),
         (-hs, hs), (-hs, hs)),
    ]

    # Tall box (rectangular prism) in the scene
    tall_box_center = np.array([-hs * 0.35, -hs + hs * 0.6, z0 + hs * 0.2])
    tall_box_size = np.array([hs * 0.3, hs * 0.6, hs * 0.3])

    # Short box
    short_box_center = np.array([hs * 0.35, -hs + hs * 0.25, z0 - hs * 0.1])
    short_box_size = np.array([hs * 0.3, hs * 0.25, hs * 0.3])

    # Box face definitions for the inner boxes
    inner_boxes = []
    for center, size, name in [(tall_box_center, tall_box_size, 'tall'),
                                (short_box_center, short_box_size, 'short')]:
        # 5 visible faces (skip bottom)
        bx, by, bz = size
        faces = [
            # Top
            (center + np.array([0, by, 0]), np.array([1, 0, 0]),
             np.array([0, 0, 1]), bx, bz),
            # Front (facing observer)
            (center + np.array([0, 0, -bz]), np.array([1, 0, 0]),
             np.array([0, 1, 0]), bx, by),
            # Back
            (center + np.array([0, 0, bz]), np.array([1, 0, 0]),
             np.array([0, 1, 0]), bx, by),
            # Left
            (center + np.array([-bx, 0, 0]), np.array([0, 1, 0]),
             np.array([0, 0, 1]), by, bz),
            # Right
            (center + np.array([bx, 0, 0]), np.array([0, 1, 0]),
             np.array([0, 0, 1]), by, bz),
        ]
        inner_boxes.append((faces, name))

    # Generate all occluder triangles for shadow testing
    all_occluder_boxes = [
        (tall_box_center, tall_box_size),
        (short_box_center, short_box_size),
    ]

    def is_in_shadow(point, light, occluder_boxes):
        """Test if point is shadowed by any box (simple AABB ray-box test)."""
        direction = light - point
        t_max = 1.0  # light is at t=1
        for center, size in occluder_boxes:
            box_min = center - size
            box_max = center + size
            # Ray-AABB slab test
            t_near = -np.inf
            t_far = np.inf
            for i in range(3):
                if abs(direction[i]) < 1e-15:
                    if point[i] < box_min[i] or point[i] > box_max[i]:
                        t_near = np.inf  # miss
                        break
                    continue
                inv_d = 1.0 / direction[i]
                t1 = (box_min[i] - point[i]) * inv_d
                t2 = (box_max[i] - point[i]) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_near = max(t_near, t1)
                t_far = min(t_far, t2)
                if t_near > t_far:
                    break
            else:
                # Check if intersection is between point and light
                if t_near < t_max and t_far > 1e-4:
                    return True
        return False

    def sample_wall_points(origin, u_dir, v_dir, u_range, v_range, n_pts,
                           normal_vec):
        """Sample points on a planar wall and compute lighting."""
        pts = []
        u_vals = rng.uniform(u_range[0], u_range[1], n_pts)
        v_vals = rng.uniform(v_range[0], v_range[1], n_pts)
        for u, v in zip(u_vals, v_vals):
            p = origin + u * u_dir + v * v_dir
            # Skip if inside either box
            for center, size in all_occluder_boxes:
                if all(abs(p[i] - center[i]) < size[i] + 1e-6 for i in range(3)):
                    break
            else:
                # Lighting: Lambert + distance falloff + shadow
                to_light = light_pos - p
                dist_to_light = np.linalg.norm(to_light)
                light_dir = to_light / dist_to_light
                cos_theta = max(0, np.dot(normal_vec, light_dir))
                if cos_theta > 0 and not is_in_shadow(p, light_pos,
                                                       all_occluder_boxes):
                    # Inverse-square falloff (normalized)
                    amplitude = cos_theta / (dist_to_light ** 2 + 1e-10)
                    pts.append((p[0], p[1], p[2], amplitude))
        return pts

    # Sample walls
    for name, normal, origin, u_dir, v_dir, u_range, v_range in walls:
        pts = sample_wall_points(origin, u_dir, v_dir, u_range, v_range,
                                 n_points_per_wall, normal)
        points.extend(pts)

    # Sample inner box faces
    for faces, name in inner_boxes:
        for face_center, u_dir, v_dir, u_half, v_half in faces:
            # Determine face normal from face position relative to box center
            if name == 'tall':
                bc = tall_box_center
            else:
                bc = short_box_center
            normal = face_center - bc
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normal = normal / norm_len
            else:
                normal = np.array([0, 1, 0])

            pts = sample_wall_points(face_center, u_dir, v_dir,
                                     (-u_half, u_half), (-v_half, v_half),
                                     n_points_per_wall // 3, normal)
            points.extend(pts)

    # Normalize amplitudes
    if points:
        amps = np.array([p[3] for p in points])
        max_amp = amps.max()
        if max_amp > 0:
            points = [(p[0], p[1], p[2], p[3] / max_amp) for p in points]

    return points, light_pos


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Holographic computation (Numba-accelerated)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@numba.njit(parallel=True, cache=True)
def _accumulate_fields(E, cx, cy, scene_pts, dx_arr, k):
    """Accumulate Fresnel separable fields — no temporary arrays."""
    n_rows, n_hy, n_sub, _ = E.shape
    dx2 = dx_arr * dx_arr
    n_pts = scene_pts.shape[0]
    n_total = n_rows * n_hy

    for idx in numba.prange(n_total):
        i = idx // n_hy
        j = idx % n_hy
        cx_ij = cx[i, j]
        cy_ij = cy[i, j]

        fy_r = np.empty(n_sub, dtype=np.float32)
        fy_i = np.empty(n_sub, dtype=np.float32)

        for p in range(n_pts):
            px = scene_pts[p, 0]
            py = scene_pts[p, 1]
            pz = scene_pts[p, 2]
            amp = scene_pts[p, 3]

            Dx = cx_ij - px
            Dy = cy_ij - py
            R0 = math.sqrt(Dx * Dx + Dy * Dy + pz * pz)
            if R0 < 1e-10:
                continue

            sin_tx = Dx / R0
            sin_ty = Dy / R0
            inv_2R0 = 0.5 / R0

            phase0 = k * R0
            s_r = np.float32(amp / R0 * math.cos(phase0))
            s_i = np.float32(amp / R0 * math.sin(phase0))

            # Precompute fy for all sub-pixel columns
            for sj in range(n_sub):
                py_sj = k * (sin_ty * dx_arr[sj] + inv_2R0 * dx2[sj])
                fy_r[sj] = np.float32(math.cos(py_sj))
                fy_i[sj] = np.float32(math.sin(py_sj))

            for si in range(n_sub):
                px_si = k * (sin_tx * dx_arr[si] + inv_2R0 * dx2[si])
                fx_r = np.float32(math.cos(px_si))
                fx_i = np.float32(math.sin(px_si))
                sfx_r = s_r * fx_r - s_i * fx_i
                sfx_i = s_r * fx_i + s_i * fx_r

                for sj in range(n_sub):
                    val_r = sfx_r * fy_r[sj] - sfx_i * fy_i[sj]
                    val_i = sfx_r * fy_i[sj] + sfx_i * fy_r[sj]
                    E[i, j, si, sj] += val_r + 1j * val_i


def compute_display(n_hx, n_hy, n_sub, scene_points, observers,
                    phase_levels=0, chunk_rows=8):
    """
    Compute holographic display using Numba-parallelized accumulation.

    phase_levels=0 → continuous phase (no quantization)
    phase_levels=8 → 3-bit quantization
    """
    hogel_pitch = n_sub * SUBPIXEL_PITCH
    hx = (np.arange(n_hx) - (n_hx - 1) / 2) * hogel_pitch
    hy = (np.arange(n_hy) - (n_hy - 1) / 2) * hogel_pitch
    hogel_cx, hogel_cy = np.meshgrid(hx, hy, indexing='ij')

    scene_arr = np.array(scene_points, dtype=np.float64)
    k = 2 * np.pi / LAMBDA
    dx_arr = (np.arange(n_sub, dtype=np.float64) - (n_sub - 1) / 2) * SUBPIXEL_PITCH
    hogel_size = n_sub * SUBPIXEL_PITCH
    observers_arr = np.array(observers, dtype=np.float64)

    n_obs = len(observers)
    images = np.zeros((n_obs, n_hx, n_hy), dtype=np.complex128)

    # Prepare chunks
    chunks = []
    for row_start in range(0, n_hx, chunk_rows):
        row_end = min(row_start + chunk_rows, n_hx)
        chunks.append((row_start, row_end))

    # JIT warmup on tiny input
    print(f"  {len(chunks)} chunks, Numba JIT compiling...", end="", flush=True)
    _warmup_E = np.zeros((1, 1, 2, 2), dtype=np.complex64)
    _warmup_cx = np.zeros((1, 1), dtype=np.float64)
    _warmup_cy = np.zeros((1, 1), dtype=np.float64)
    _warmup_pts = np.zeros((1, 4), dtype=np.float64)
    _warmup_dx = np.zeros(2, dtype=np.float64)
    _accumulate_fields(_warmup_E, _warmup_cx, _warmup_cy, _warmup_pts, _warmup_dx, k)
    print(" done", flush=True)

    for row_start, row_end in tqdm(chunks, desc="  Chunks", unit="chunk"):
        n_rows = row_end - row_start
        cx_chunk = hogel_cx[row_start:row_end]
        cy_chunk = hogel_cy[row_start:row_end]

        # Numba-parallel field accumulation
        E = np.zeros((n_rows, n_hy, n_sub, n_sub), dtype=np.complex64)
        _accumulate_fields(E, cx_chunk, cy_chunk, scene_arr, dx_arr, k)

        # Phase extraction + optional quantization
        phase = np.angle(E)
        if phase_levels > 0:
            phase_pos = phase % (2 * np.pi)
            levels = np.round(phase_pos / (2 * np.pi) * phase_levels) % phase_levels
            phase = levels * (2 * np.pi / phase_levels)
        del E

        # Far-field via FFT
        E_q = np.exp(1j * phase.astype(np.float64))
        del phase
        ff = np.fft.fft2(E_q, axes=(-2, -1))
        del E_q

        # Sample at observer directions with 3×3 averaging
        for oi in range(n_obs):
            ox, oy, oz = observers_arr[oi]
            dx_obs = ox - cx_chunk
            dy_obs = oy - cy_chunk
            dist_obs = np.sqrt(dx_obs ** 2 + dy_obs ** 2 + oz ** 2)
            m_x_f = dx_obs / dist_obs * hogel_size / LAMBDA
            m_y_f = dy_obs / dist_obs * hogel_size / LAMBDA

            acc = np.zeros((n_rows, n_hy), dtype=np.complex128)
            count = 0
            for dm_x in [-1, 0, 1]:
                for dm_y in [-1, 0, 1]:
                    mx = (np.round(m_x_f).astype(int) + dm_x) % n_sub
                    my = (np.round(m_y_f).astype(int) + dm_y) % n_sub
                    idx_r = np.arange(n_rows)[:, None]
                    idx_c = np.arange(n_hy)[None, :]
                    acc += ff[idx_r, idx_c, mx, my]
                    count += 1
            images[oi, row_start:row_end, :] = acc / count

    return [images[oi] for oi in range(n_obs)], hogel_cx, hogel_cy


def main():
    # ── Cornell Box Scene ──
    print("=" * 60)
    print("Generating Cornell Box scene")
    print("=" * 60)

    # Box: 15mm half-size centered at z=100mm
    scene_points, light_pos = make_cornell_box(
        n_points_per_wall=600,
        box_size=15e-3,
        depth_center=100e-3,
    )
    print(f"  Scene: {len(scene_points)} point sources")
    print(f"  Light at ({light_pos[0]*1e3:.1f}, {light_pos[1]*1e3:.1f}, "
          f"{light_pos[2]*1e3:.1f}) mm")

    # ── Display config ──
    n_sub = 338
    hogel_pitch = n_sub * SUBPIXEL_PITCH
    n_hx, n_hy = 256, 144  # 144p
    display_w = n_hx * hogel_pitch * 1e3
    display_h = n_hy * hogel_pitch * 1e3
    total_sub = n_hx * n_hy * n_sub ** 2

    print(f"\n  Display: {n_hx}×{n_hy} holoxels, {n_sub}×{n_sub} sub-px")
    print(f"  Size: {display_w:.1f}×{display_h:.1f} mm")
    print(f"  Sub-pixels: {total_sub / 1e9:.2f}B")

    # Observer positions
    obs_z = 400e-3
    observers = [
        (-20e-3, 0, obs_z),
        (-10e-3, 0, obs_z),
        (0, 0, obs_z),
        (10e-3, 0, obs_z),
        (20e-3, 0, obs_z),
    ]

    # ── Run with continuous phase (best quality) ──
    print(f"\n{'='*60}")
    print(f"Computing holographic display (continuous phase)")
    print(f"{'='*60}")

    t0 = time.time()
    images, hogel_cx, hogel_cy = compute_display(
        n_hx, n_hy, n_sub, scene_points, observers,
        phase_levels=0, chunk_rows=8
    )
    dt = time.time() - t0
    print(f"  Total: {dt:.1f}s ({dt/60:.1f} min)")

    # ── Plot 1: Observer views (grayscale) ──
    hx_mm = hogel_cx[:, 0] * 1e3
    hy_mm = hogel_cy[0, :] * 1e3
    extent = [hx_mm[0], hx_mm[-1], hy_mm[0], hy_mm[-1]]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for i, (img, (ox, oy, oz)) in enumerate(zip(images, observers)):
        ax = axes[i]
        I = np.abs(img.T) ** 2
        # Gamma correction for better visibility
        I_gamma = np.power(I / (I.max() + 1e-30), 0.4)
        ax.imshow(I_gamma, cmap='gray', aspect='equal', extent=extent,
                  origin='lower', vmin=0, vmax=1)
        ax.set_title(f'obs x={ox * 1e3:+.0f}mm', fontsize=11)
        ax.set_xlabel('x (mm)')
        if i == 0:
            ax.set_ylabel('y (mm)')

    plt.suptitle(f'Holographic Cornell Box — Continuous Phase\n'
                 f'{n_hx}×{n_hy} holoxels, {n_sub}×{n_sub} sub-px, '
                 f'{len(scene_points)} points, λ=532nm',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out1 = PLOT_DIR / "holo_cornell_views.png"
    fig.savefig(out1, dpi=150)
    plt.close()
    print(f"✅ Saved: {out1}")

    # ── Plot 2: Quality comparison (continuous vs 8-level vs 16-level) ──
    print(f"\n{'='*60}")
    print(f"Computing 8-level quantized version for comparison")
    print(f"{'='*60}")

    t0 = time.time()
    images_q8, _, _ = compute_display(
        n_hx, n_hy, n_sub, scene_points,
        [(0, 0, obs_z)],  # center view only
        phase_levels=8, chunk_rows=8
    )
    dt_q8 = time.time() - t0
    print(f"  Total: {dt_q8:.1f}s")

    print(f"\n{'='*60}")
    print(f"Computing 32-level quantized version")
    print(f"{'='*60}")

    t0 = time.time()
    images_q32, _, _ = compute_display(
        n_hx, n_hy, n_sub, scene_points,
        [(0, 0, obs_z)],
        phase_levels=32, chunk_rows=8
    )
    dt_q32 = time.time() - t0
    print(f"  Total: {dt_q32:.1f}s")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(axes,
                               [images_q8[0], images_q32[0], images[2]],
                               ['8-level (3-bit)', '32-level (5-bit)',
                                'Continuous']):
        I = np.abs(img.T) ** 2
        I_gamma = np.power(I / (I.max() + 1e-30), 0.4)
        ax.imshow(I_gamma, cmap='gray', aspect='equal', extent=extent,
                  origin='lower', vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x (mm)')

    axes[0].set_ylabel('y (mm)')
    plt.suptitle(f'Phase Quantization Comparison — Cornell Box (center view)\n'
                 f'More levels → fewer diffraction artifacts',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out2 = PLOT_DIR / "holo_cornell_quantization.png"
    fig.savefig(out2, dpi=150)
    plt.close()
    print(f"✅ Saved: {out2}")

    # ── Plot 3: Parallax difference ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left and right views
    I_left = np.abs(images[0].T) ** 2
    I_right = np.abs(images[-1].T) ** 2
    I_center = np.abs(images[2].T) ** 2

    ax = axes[0]
    I_gamma = np.power(I_center / (I_center.max() + 1e-30), 0.4)
    ax.imshow(I_gamma, cmap='gray', aspect='equal', extent=extent,
              origin='lower', vmin=0, vmax=1)
    ax.set_title('Center view (x=0)', fontsize=12)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    ax = axes[1]
    diff = I_right - I_left
    vabs = np.percentile(np.abs(diff), 99)
    ax.imshow(diff, cmap='RdBu_r', aspect='equal', extent=extent,
              origin='lower', vmin=-vabs, vmax=vabs)
    ax.set_title('Right − Left (parallax map)', fontsize=12)
    ax.set_xlabel('x (mm)')

    plt.suptitle('Cornell Box — Parallax Visualization\n'
                 'Red/blue = features that shift between viewpoints',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out3 = PLOT_DIR / "holo_cornell_parallax.png"
    fig.savefig(out3, dpi=150)
    plt.close()
    print(f"✅ Saved: {out3}")

    # ── Plot 4: Scene visualization (point cloud) ──
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    xs = np.array([p[0] for p in scene_points]) * 1e3
    ys = np.array([p[1] for p in scene_points]) * 1e3
    zs = np.array([p[2] for p in scene_points]) * 1e3
    amps = np.array([p[3] for p in scene_points])

    sc = ax.scatter(xs, zs, ys, c=amps, cmap='inferno', s=2, alpha=0.7)
    ax.scatter(*light_pos * 1e3, color='yellow', s=100, marker='*',
               label='Light', zorder=5)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_zlabel('y (mm)')
    ax.set_title(f'Cornell Box Scene — {len(scene_points)} points\n'
                 f'Color = brightness (Lambert shading + shadows)',
                 fontsize=12)
    plt.colorbar(sc, ax=ax, label='Amplitude', shrink=0.6)
    ax.legend()

    out4 = PLOT_DIR / "holo_cornell_scene.png"
    fig.savefig(out4, dpi=150)
    plt.close()
    print(f"✅ Saved: {out4}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"CORNELL BOX SUMMARY")
    print(f"{'='*60}")
    print(f"Scene: {len(scene_points)} points, {len(set(round(p[2]*1e6) for p in scene_points))} unique depths")
    print(f"Display: {n_hx}×{n_hy} holoxels, {n_sub}×{n_sub} sub-px")
    print(f"Phase: continuous (best quality)")
    print(f"Computation: {dt:.1f}s ({dt/60:.1f} min) with Numba ({numba.config.NUMBA_NUM_THREADS} threads)")
    print(f"Quantization comparison: 8-level {dt_q8:.1f}s, 32-level {dt_q32:.1f}s")


if __name__ == "__main__":
    main()

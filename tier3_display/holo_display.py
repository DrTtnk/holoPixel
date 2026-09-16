"""
Holographic Display Simulator — Full Screen Reconstruction

Simulates a hogel-based holographic display with "magic" sub-pixels:
each sub-pixel absorbs coherent light and re-emits omnidirectionally
with a controlled phase (8-level quantization).

3D scene: two point sources at different depths (near/far parallax demo).
The viewer sees a 2D image reconstructed from the holographic display,
and moving left/right produces parallax (near object shifts more).

Scaling ladder:
  Level 1: 16×16 hogels, 16×16 sub-pixels (proof of concept)
  Level 2: 64×36 hogels, 32×32 sub-pixels (basic parallax)
  Level 3: 256×144 hogels, 32×32 sub-pixels (144p)
  Level 4: 506×319 hogels, 338×338 sub-pixels (credit card at 150 DPI)

Physics:
  - Display at z=0, viewer at z>0
  - Phase at sub-pixel (i,j) for a scene point P:
      φ = (2π/λ) × |r_P - r_ij|   (converging spherical wave)
  - Far-field of each hogel = 2D FFT of exp(jφ)
  - Observer at angle (θx, θy) from hogel → samples FFT at that frequency
  - Parallax: near objects shift more than far objects when observer moves
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Display parameters ─────────────────────────────────────────
HOGEL_PITCH = 169.3e-6  # m (150 DPI)
SUBPIXEL_PITCH = 0.5e-6 # m
WAVELENGTHS = {'Red': 633e-9, 'Green': 532e-9, 'Blue': 450e-9}
N_PHASE_LEVELS = 8       # 3-bit quantization


def make_display(n_hogels_x, n_hogels_y, n_sub):
    """Create display geometry.

    Returns hogel centers (N_hx, N_hy, 3) and sub-pixel offsets (n_sub, n_sub, 2).
    """
    # Hogel center positions
    hx = (np.arange(n_hogels_x) - (n_hogels_x - 1) / 2) * HOGEL_PITCH
    hy = (np.arange(n_hogels_y) - (n_hogels_y - 1) / 2) * HOGEL_PITCH
    hogel_cx, hogel_cy = np.meshgrid(hx, hy, indexing='ij')  # (Nhx, Nhy)

    # Sub-pixel offsets within a hogel (centered)
    si = (np.arange(n_sub) - (n_sub - 1) / 2) * SUBPIXEL_PITCH
    sj = (np.arange(n_sub) - (n_sub - 1) / 2) * SUBPIXEL_PITCH
    sub_dx, sub_dy = np.meshgrid(si, sj, indexing='ij')  # (n_sub, n_sub)

    return hogel_cx, hogel_cy, sub_dx, sub_dy


def compute_hogel_phases(hogel_cx, hogel_cy, sub_dx, sub_dy,
                         scene_points, lam):
    """Compute phase pattern for all hogels given scene points.

    scene_points: list of (x, y, z, amplitude) — z > 0 means in front of display.

    Returns: phases array (n_hx, n_hy, n_sub, n_sub) in [0, 2π).
    """
    n_hx, n_hy = hogel_cx.shape
    n_sub = sub_dx.shape[0]
    k = 2 * np.pi / lam

    # Complex field at each sub-pixel of each hogel
    # Shape: (n_hx, n_hy, n_sub, n_sub) — computed in vectorized batches
    E_total = np.zeros((n_hx, n_hy, n_sub, n_sub), dtype=np.complex128)

    for (px, py, pz, amp) in scene_points:
        # For each hogel, sub-pixel position = hogel_center + offset
        # Distance to scene point:
        # dx = hogel_cx[:,:,None,None] + sub_dx[None,None,:,:] - px
        # dy = hogel_cy[:,:,None,None] + sub_dy[None,None,:,:] - py
        # dz = pz
        # dist = sqrt(dx² + dy² + dz²)

        dx = hogel_cx[:, :, None, None] + sub_dx[None, None, :, :] - px
        dy = hogel_cy[:, :, None, None] + sub_dy[None, None, :, :] - py
        dist = np.sqrt(dx ** 2 + dy ** 2 + pz ** 2)

        # Spherical wave: converging toward the point
        # Phase = k × distance (conjugate = time-reversed wavefront)
        E_total += amp * np.exp(1j * k * dist) / dist

    # Extract phase, quantize to N levels
    phases = np.angle(E_total) % (2 * np.pi)
    phase_levels = np.round(phases / (2 * np.pi) * N_PHASE_LEVELS) % N_PHASE_LEVELS
    phases_quantized = phase_levels * (2 * np.pi / N_PHASE_LEVELS)

    return phases_quantized


def render_view(hogel_cx, hogel_cy, phases, n_sub, lam, observer_pos):
    """Render what an observer sees from a given position.

    For each hogel, compute the far-field amplitude in the direction
    of the observer, using FFT.

    Returns: image (n_hx, n_hy) complex amplitude.
    """
    n_hx, n_hy = hogel_cx.shape
    ox, oy, oz = observer_pos
    k = 2 * np.pi / lam

    # Hogel angular extent: n_sub × sub_pitch
    hogel_size = n_sub * SUBPIXEL_PITCH

    # For each hogel, find the direction to the observer
    # sin(θx) = (ox - hx) / sqrt((ox-hx)² + (oy-hy)² + oz²)
    dx = ox - hogel_cx  # (n_hx, n_hy)
    dy = oy - hogel_cy
    dist = np.sqrt(dx ** 2 + dy ** 2 + oz ** 2)
    sin_theta_x = dx / dist
    sin_theta_y = dy / dist

    # Far-field: FFT of the phase pattern
    # The spatial frequency corresponding to angle θ is:
    # fx = sin(θx) / λ,  fy = sin(θy) / λ
    # The FFT frequency bin: kx = fx × (n_sub × sub_pitch)
    # Normalized bin index: m = sin(θx) × hogel_size / λ

    # Compute FFT of each hogel's pattern
    # phases shape: (n_hx, n_hy, n_sub, n_sub)
    E_sub = np.exp(1j * phases)
    far_field = np.fft.fft2(E_sub, axes=(-2, -1))  # (n_hx, n_hy, n_sub, n_sub)

    # Frequency axis: bin m corresponds to sin(θ) = m × λ / hogel_size
    # We need the bin closest to sin(θx), sin(θy) for each hogel
    m_x = sin_theta_x * hogel_size / lam  # fractional bin index
    m_y = sin_theta_y * hogel_size / lam

    # Round to nearest integer bin (nearest-neighbor sampling)
    m_x_int = np.round(m_x).astype(int) % n_sub
    m_y_int = np.round(m_y).astype(int) % n_sub

    # Sample the far-field at the right angle for each hogel
    image = np.zeros((n_hx, n_hy), dtype=np.complex128)
    for ix in range(n_hx):
        for iy in range(n_hy):
            image[ix, iy] = far_field[ix, iy, m_x_int[ix, iy], m_y_int[ix, iy]]

    return image


def render_view_fast(hogel_cx, hogel_cy, phases, n_sub, lam, observer_pos):
    """Vectorized version of render_view using advanced indexing."""
    n_hx, n_hy = hogel_cx.shape
    ox, oy, oz = observer_pos

    hogel_size = n_sub * SUBPIXEL_PITCH

    dx = ox - hogel_cx
    dy = oy - hogel_cy
    dist = np.sqrt(dx ** 2 + dy ** 2 + oz ** 2)
    sin_theta_x = dx / dist
    sin_theta_y = dy / dist

    E_sub = np.exp(1j * phases)
    far_field = np.fft.fft2(E_sub, axes=(-2, -1))

    m_x = (np.round(sin_theta_x * hogel_size / lam).astype(int)) % n_sub
    m_y = (np.round(sin_theta_y * hogel_size / lam).astype(int)) % n_sub

    # Advanced indexing: select one element per hogel
    idx_x = np.arange(n_hx)[:, None]
    idx_y = np.arange(n_hy)[None, :]
    image = far_field[idx_x, idx_y, m_x, m_y]

    return image


def simulate_display(n_hx, n_hy, n_sub, scene_points, lam,
                     observer_positions, title_prefix=""):
    """Full simulation: compute hogel phases and render multiple views."""
    print(f"\n{'='*60}")
    print(f"Display: {n_hx}×{n_hy} hogels, {n_sub}×{n_sub} sub-pixels")
    print(f"Physical size: {n_hx * HOGEL_PITCH * 1e3:.1f}mm × "
          f"{n_hy * HOGEL_PITCH * 1e3:.1f}mm")
    print(f"Total sub-pixels: {n_hx * n_hy * n_sub * n_sub:,}")
    print(f"Wavelength: {lam*1e9:.0f}nm")
    print(f"{'='*60}")

    t0 = time.time()

    # Create display geometry
    hogel_cx, hogel_cy, sub_dx, sub_dy = make_display(n_hx, n_hy, n_sub)

    # Compute holographic patterns
    print("Computing hogel phase patterns...", end=" ", flush=True)
    phases = compute_hogel_phases(hogel_cx, hogel_cy, sub_dx, sub_dy,
                                 scene_points, lam)
    t1 = time.time()
    print(f"done ({t1 - t0:.1f}s)")

    # Render views from multiple positions
    images = []
    for i, obs_pos in enumerate(observer_positions):
        print(f"  Rendering view {i+1}/{len(observer_positions)}: "
              f"observer at ({obs_pos[0]*1e3:.1f}, {obs_pos[1]*1e3:.1f}, "
              f"{obs_pos[2]*1e3:.0f})mm...", end=" ", flush=True)
        t2 = time.time()
        img = render_view_fast(hogel_cx, hogel_cy, phases, n_sub, lam, obs_pos)
        t3 = time.time()
        print(f"done ({t3 - t2:.1f}s)")
        images.append(img)

    print(f"Total time: {time.time() - t0:.1f}s")
    return phases, images


def run_parallax_demo(n_hx, n_hy, n_sub, label=""):
    """Near/far parallax demonstration.

    Two point sources:
      Near: (4mm, 0, 30mm)  — close to display, should shift MORE
      Far:  (-4mm, 0, 150mm) — far from display, should shift LESS

    Observer moves left-to-right at z=300mm.
    """
    # Scene: two points at different depths
    near_point = (4e-3, 0.0, 30e-3, 1.0)   # (x, y, z, amplitude)
    far_point = (-4e-3, 0.0, 150e-3, 1.0)
    scene = [near_point, far_point]

    # Observer positions: sweep x at fixed z
    obs_z = 300e-3  # 30cm viewing distance
    n_views = 7
    obs_x_range = np.linspace(-15e-3, 15e-3, n_views)  # ±15mm sweep
    observer_positions = [(x, 0.0, obs_z) for x in obs_x_range]

    lam = WAVELENGTHS['Green']  # single wavelength for speed

    phases, images = simulate_display(
        n_hx, n_hy, n_sub, scene, lam, observer_positions,
        title_prefix=label
    )

    # ── Plot results ──
    fig, axes = plt.subplots(2, max(n_views, 4), figsize=(3 * n_views, 7))
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    # Row 1: Observer views (intensity)
    vmax = max(np.max(np.abs(img) ** 2) for img in images)
    for i, (img, ox) in enumerate(zip(images, obs_x_range)):
        if i < axes.shape[1]:
            ax = axes[0, i]
            intensity = np.abs(img.T) ** 2
            ax.imshow(intensity, cmap='hot', aspect='equal',
                      vmin=0, vmax=vmax * 0.5,
                      extent=[-n_hx / 2, n_hx / 2, -n_hy / 2, n_hy / 2])
            ax.set_title(f'x={ox*1e3:.1f}mm', fontsize=9)
            if i == 0:
                ax.set_ylabel('Observer views\n(intensity)')

    # Row 2: Phase patterns (sample hogels)
    sample_hogels = np.linspace(0, n_hx - 1, min(n_views, axes.shape[1])).astype(int)
    for i, h_idx in enumerate(sample_hogels):
        if i < axes.shape[1]:
            ax = axes[1, i]
            ax.imshow(phases[h_idx, n_hy // 2].T, cmap='hsv',
                      vmin=0, vmax=2 * np.pi, aspect='equal')
            ax.set_title(f'Hogel ({h_idx},{n_hy//2})', fontsize=9)
            if i == 0:
                ax.set_ylabel('Phase patterns\n(sample hogels)')

    # Hide unused axes
    for row in range(2):
        for col in range(axes.shape[1]):
            if (row == 0 and col >= n_views) or (row == 1 and col >= len(sample_hogels)):
                axes[row, col].axis('off')

    plt.suptitle(f'Holographic Parallax — {n_hx}×{n_hy} hogels, {n_sub}×{n_sub} sub-px\n'
                 f'Near point: (4mm, 30mm depth) | Far point: (-4mm, 150mm depth)\n'
                 f'Observer at z=300mm, sweeping x=±15mm',
                 fontsize=11)
    plt.tight_layout()

    out = PLOT_DIR / f"holo_display_{label}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"✅ Saved: {out}")

    return phases, images


def run_rgb_parallax(n_hx, n_hy, n_sub, label=""):
    """Full RGB parallax demo with 3 wavelengths."""
    near_point = (4e-3, 0.0, 30e-3, 1.0)
    far_point = (-4e-3, 0.0, 150e-3, 1.0)
    scene = [near_point, far_point]

    obs_z = 300e-3
    n_views = 5
    obs_x_range = np.linspace(-15e-3, 15e-3, n_views)
    observer_positions = [(x, 0.0, obs_z) for x in obs_x_range]

    rgb_images = {}
    for color, lam in WAVELENGTHS.items():
        print(f"\n--- {color} ({lam*1e9:.0f}nm) ---")
        _, images = simulate_display(n_hx, n_hy, n_sub, scene, lam,
                                     observer_positions)
        rgb_images[color] = images

    # Combine RGB
    fig, axes = plt.subplots(1, n_views, figsize=(3.5 * n_views, 4))

    for i, ox in enumerate(obs_x_range):
        r = np.abs(rgb_images['Red'][i].T) ** 2
        g = np.abs(rgb_images['Green'][i].T) ** 2
        b = np.abs(rgb_images['Blue'][i].T) ** 2

        # Normalize each channel
        for ch in [r, g, b]:
            mx = ch.max()
            if mx > 0:
                ch /= mx

        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.clip(rgb ** 0.4, 0, 1)  # gamma correction

        axes[i].imshow(rgb, aspect='equal',
                       extent=[-n_hx / 2, n_hx / 2, -n_hy / 2, n_hy / 2])
        axes[i].set_title(f'x={ox*1e3:.1f}mm', fontsize=10)

    plt.suptitle(f'RGB Holographic Parallax — {n_hx}×{n_hy} hogels, '
                 f'{n_sub}×{n_sub} sub-px\nNear: 30mm depth | Far: 150mm depth | '
                 f'Observer at z=300mm',
                 fontsize=11)
    plt.tight_layout()

    out = PLOT_DIR / f"holo_display_rgb_{label}.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"\n✅ Saved: {out}")


if __name__ == "__main__":
    # ── Level 1: Tiny (proof of concept) ──
    print("\n" + "=" * 60)
    print("LEVEL 1: 16×16 hogels, 16×16 sub-pixels")
    print("=" * 60)
    run_parallax_demo(16, 16, 16, label="L1_16x16")

    # ── Level 2: Small (basic parallax) ──
    print("\n" + "=" * 60)
    print("LEVEL 2: 64×36 hogels, 32×32 sub-pixels")
    print("=" * 60)
    run_parallax_demo(64, 36, 32, label="L2_64x36")

    # ── Level 2 RGB ──
    print("\n" + "=" * 60)
    print("LEVEL 2 RGB: 64×36 hogels, 32×32 sub-pixels, 3 wavelengths")
    print("=" * 60)
    run_rgb_parallax(64, 36, 32, label="L2_64x36")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Scaling ladder results:
  Level 1: 16×16×16×16 = 65,536 sub-pixels → instant
  Level 2: 64×36×32×32 = 2,359,296 sub-pixels → seconds
  Level 3: 256×144×32×32 = 37,748,736 sub-pixels → needs optimization
  Level 4: 506×319×338×338 = 18.4 billion sub-pixels → needs chunking

For Level 3+, optimizations:
  - Compute phase per hogel in chunks (not all at once)
  - Use plane-wave approximation for far scene points
  - GPU acceleration (cupy/torch)
""")

"""
Hogel Display — Corrected Light Field Simulation (v3)

Fixed geometry: scene objects sized to fill the display's angular range.
The display is a "window" into a virtual 3D scene.

Key design rules:
  - Display FoV = ±30° means scene objects can be at angles up to 30° from normal
  - A 34mm display at 0.5m viewing distance subtends only ±1.9° to the observer,
    but each hogel can redirect light by ±30°
  - For parallax: objects at different depths shift by different amounts
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# =============================================================
# Light Field Display
# =============================================================
class LFDisplay:
    def __init__(self, n_hogels, hogel_pitch, n_ang, max_angle_deg, phase_bits):
        self.N = n_hogels
        self.pitch = hogel_pitch
        self.n_ang = n_ang
        self.max_angle = max_angle_deg
        self.phase_bits = phase_bits

        # Efficiency from quantization
        self.eta = np.sinc(1 / (2 ** phase_bits)) ** 2 if phase_bits > 0 else 1.0

        # Hogel centers
        pos = (np.arange(n_hogels) - (n_hogels - 1) / 2) * hogel_pitch
        self.hx, self.hy = np.meshgrid(pos, pos)
        self.extent_mm = n_hogels * hogel_pitch * 1e3 / 2

        # Angular bins
        self.ang_step = 2 * max_angle_deg / n_ang
        self.ang_edges = np.linspace(-max_angle_deg, max_angle_deg, n_ang + 1)
        self.ang_centers = 0.5 * (self.ang_edges[:-1] + self.ang_edges[1:])

        # 4D light field: hogel_y, hogel_x, ang_y, ang_x
        self.lf = np.zeros((n_hogels, n_hogels, n_ang, n_ang))

    @property
    def size_mm(self):
        return self.N * self.pitch * 1e3

    def _angle_to_bin(self, angle_deg):
        idx = ((angle_deg + self.max_angle) / self.ang_step).astype(int)
        return np.clip(idx, 0, self.n_ang - 1)

    def encode_scene(self, scene_pts):
        """
        scene_pts: (M, 4) → [x, y, z, brightness]
        z > 0 means behind the display (virtual image).
        z < 0 means in front (between display and observer).
        """
        for px, py, pz, brightness in scene_pts:
            dx = px - self.hx
            dy = py - self.hy

            # Direction from hogel to point
            theta_x = np.degrees(np.arctan2(dx, np.abs(pz)))
            theta_y = np.degrees(np.arctan2(dy, np.abs(pz)))

            # Bin indices
            bx = self._angle_to_bin(theta_x)
            by = self._angle_to_bin(theta_y)

            # Valid angles within FoV
            mask = (np.abs(theta_x) < self.max_angle) & (np.abs(theta_y) < self.max_angle)

            r2 = dx**2 + dy**2 + pz**2
            intensity = self.eta * brightness / r2

            iy, ix = np.where(mask)
            for k in range(len(iy)):
                self.lf[iy[k], ix[k], by[iy[k], ix[k]], bx[iy[k], ix[k]]] += intensity[iy[k], ix[k]]

    def observe(self, obs_x, obs_y, obs_z):
        """Render what observer at (x,y,z) sees. z > 0 means in front of display."""
        image = np.zeros((self.N, self.N))

        dx = obs_x - self.hx
        dy = obs_y - self.hy

        theta_x = np.degrees(np.arctan2(dx, obs_z))
        theta_y = np.degrees(np.arctan2(dy, obs_z))

        bx = self._angle_to_bin(theta_x)
        by = self._angle_to_bin(theta_y)

        for iy in range(self.N):
            for ix in range(self.N):
                image[iy, ix] = self.lf[iy, ix, by[iy, ix], bx[iy, ix]]

        return image


# =============================================================
# Scene Generators
# =============================================================
def scene_cross_and_circle():
    """Cross at z=20mm, circle at z=50mm (both behind display)."""
    pts = []
    # Cross: ±10mm arms
    z1 = 0.02
    for x in np.linspace(-10e-3, 10e-3, 40):
        pts.append([x, 0, z1, 1.0])
    for y in np.linspace(-10e-3, 10e-3, 40):
        pts.append([0, y, z1, 1.0])

    # Circle: 12mm radius
    z2 = 0.05
    for a in np.linspace(0, 2 * np.pi, 60, endpoint=False):
        pts.append([12e-3 * np.cos(a), 12e-3 * np.sin(a), z2, 0.7])

    return np.array(pts)


def scene_3_squares():
    """Three squares at different depths — clear depth ordering."""
    pts = []
    configs = [
        (-8e-3, 0, 0.015, 5e-3, 1.0),   # left, near
        (0, 2e-3, 0.035, 5e-3, 0.8),     # center, mid
        (8e-3, -2e-3, 0.06, 5e-3, 0.6),  # right, far
    ]
    for cx, cy, cz, size, brightness in configs:
        # Square outline
        for t in np.linspace(-1, 1, 20):
            pts.append([cx + t * size, cy - size, cz, brightness])
            pts.append([cx + t * size, cy + size, cz, brightness])
            pts.append([cx - size, cy + t * size, cz, brightness])
            pts.append([cx + size, cy + t * size, cz, brightness])

    return np.array(pts)


def scene_depth_line():
    """Diagonal line going into depth — tests depth perception."""
    pts = []
    for t in np.linspace(0, 1, 100):
        x = (t - 0.5) * 20e-3
        y = (t - 0.5) * 15e-3
        z = 0.01 + t * 0.05  # 10mm to 60mm behind display
        pts.append([x, y, z, 1.0 - 0.3 * t])  # dimmer with distance
    return np.array(pts)


# =============================================================
# Simulations
# =============================================================
def sim_parallax_sweep():
    """Show parallax for cross+circle scene."""
    print("=" * 70)
    print("PARALLAX SWEEP — Cross (near) + Circle (far)")
    print("=" * 70)

    scene = scene_cross_and_circle()

    display = LFDisplay(
        n_hogels=200, hogel_pitch=169.3e-6,
        n_ang=128, max_angle_deg=30, phase_bits=3
    )
    print(f"Display: {display.N}×{display.N} hogels ({display.size_mm:.1f}mm)")
    print(f"Angular bins: {display.n_ang}, step: {display.ang_step:.2f}°")
    print(f"Phase: {display.phase_bits}-bit, η={display.eta:.1%}")

    print("Encoding...")
    display.encode_scene(scene)

    obs_z = 0.3  # 300mm viewing distance
    offsets = [-20e-3, -12e-3, -6e-3, 0, 6e-3, 12e-3, 20e-3]

    fig, axes = plt.subplots(1, 7, figsize=(24, 4))
    ext = [-display.extent_mm, display.extent_mm,
           -display.extent_mm, display.extent_mm]

    images = []
    for ox in offsets:
        img = display.observe(ox, 0, obs_z)
        images.append(img)

    vmax = max(img.max() for img in images)

    for i, (img, ox) in enumerate(zip(images, offsets)):
        axes[i].imshow(img, cmap="inferno", extent=ext, vmin=0, vmax=vmax,
                      origin="lower", interpolation="bilinear")
        axes[i].set_title(f"x={ox*1e3:.0f}mm", fontsize=10)
        axes[i].set_xlabel("x (mm)")
        if i == 0:
            axes[i].set_ylabel("y (mm)")

    fig.suptitle(f"Parallax — Cross at z=20mm (near), Circle at z=50mm (far)\n"
                 f"200×200 hogels, 3-bit phase, 128 angular bins, observer at z={obs_z*1e3:.0f}mm",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "lf_parallax.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'lf_parallax.png'}")


def sim_3_squares_parallax():
    """Three squares at different depths."""
    print("\n" + "=" * 70)
    print("THREE SQUARES AT DIFFERENT DEPTHS")
    print("=" * 70)

    scene = scene_3_squares()

    display = LFDisplay(
        n_hogels=200, hogel_pitch=169.3e-6,
        n_ang=128, max_angle_deg=30, phase_bits=3
    )
    print(f"Display: {display.size_mm:.1f}mm, {scene.shape[0]} pts")

    print("Encoding...")
    display.encode_scene(scene)

    obs_z = 0.3
    positions = [
        (-15e-3, 8e-3, "Top-Left"),
        (0, 8e-3, "Top"),
        (15e-3, 8e-3, "Top-Right"),
        (-15e-3, 0, "Left"),
        (0, 0, "Center"),
        (15e-3, 0, "Right"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ext = [-display.extent_mm, display.extent_mm,
           -display.extent_mm, display.extent_mm]

    images = []
    for ox, oy, label in positions:
        img = display.observe(ox, oy, obs_z)
        images.append((img, label))

    vmax = max(img.max() for img, _ in images)

    for idx, (img, label) in enumerate(images):
        ax = axes[idx // 3][idx % 3]
        ax.imshow(img, cmap="inferno", extent=ext, vmin=0, vmax=vmax,
                 origin="lower", interpolation="bilinear")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    fig.suptitle("Three Squares — Near (left), Mid (center), Far (right)\n"
                 "6 viewpoints, 200×200 hogels, 3-bit phase",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "lf_3squares.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'lf_3squares.png'}")


def sim_stereo_pair():
    """Human stereo pair for depth line."""
    print("\n" + "=" * 70)
    print("STEREO PAIR — Depth Line")
    print("=" * 70)

    scene = scene_depth_line()

    display = LFDisplay(
        n_hogels=200, hogel_pitch=169.3e-6,
        n_ang=128, max_angle_deg=30, phase_bits=3
    )

    print("Encoding...")
    display.encode_scene(scene)

    # Stereo at close viewing distance
    ipd = 63e-3
    obs_z = 0.2  # close viewing
    img_l = display.observe(-ipd / 2, 0, obs_z)
    img_r = display.observe(ipd / 2, 0, obs_z)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ext = [-display.extent_mm, display.extent_mm,
           -display.extent_mm, display.extent_mm]
    vmax = max(img_l.max(), img_r.max())

    ax1.imshow(img_l, cmap="inferno", extent=ext, vmin=0, vmax=vmax,
              origin="lower", interpolation="bilinear")
    ax1.set_title("Left Eye")

    ax2.imshow(img_r, cmap="inferno", extent=ext, vmin=0, vmax=vmax,
              origin="lower", interpolation="bilinear")
    ax2.set_title("Right Eye")

    # Anaglyph
    combined = np.zeros((*img_l.shape, 3))
    if vmax > 0:
        combined[:, :, 0] = img_l / vmax
        combined[:, :, 2] = img_r / vmax
    combined = np.clip(combined, 0, 1)
    ax3.imshow(combined, extent=ext, origin="lower")
    ax3.set_title("Anaglyph (Red=L, Blue=R)")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    fig.suptitle(f"Stereo Pair — Diagonal depth line (10–60mm behind display)\n"
                 f"IPD={ipd*1e3:.0f}mm, viewing distance={obs_z*1e3:.0f}mm",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "lf_stereo.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'lf_stereo.png'}")


def sim_parameter_grid():
    """Show how phase bits and angular resolution affect quality."""
    print("\n" + "=" * 70)
    print("PARAMETER GRID")
    print("=" * 70)

    scene = scene_cross_and_circle()

    configs = [
        # label, n_hogels, n_ang, phase_bits
        ("128 hogels\n32 ang bins\n2-bit phase", 128, 32, 2),
        ("128 hogels\n64 ang bins\n3-bit phase", 128, 64, 3),
        ("128 hogels\n128 ang bins\n3-bit phase", 128, 128, 3),
        ("200 hogels\n64 ang bins\n2-bit phase", 200, 64, 2),
        ("200 hogels\n64 ang bins\n3-bit phase", 200, 64, 3),
        ("200 hogels\n128 ang bins\n4-bit phase", 200, 128, 4),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    obs = (0, 0, 0.3)

    for idx, (label, n_h, n_a, bits) in enumerate(configs):
        print(f"  {label.replace(chr(10), ', ')}")
        d = LFDisplay(n_hogels=n_h, hogel_pitch=169.3e-6,
                      n_ang=n_a, max_angle_deg=30, phase_bits=bits)
        d.encode_scene(scene)
        img = d.observe(*obs)

        ax = axes[idx // 3][idx % 3]
        ext = [-d.extent_mm, d.extent_mm, -d.extent_mm, d.extent_mm]
        ax.imshow(img, cmap="inferno", extent=ext, vmin=0, origin="lower",
                 interpolation="bilinear")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    fig.suptitle("Parameter Comparison — Cross + Circle, center view", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "lf_param_grid.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'lf_param_grid.png'}")


def print_design_summary():
    """Final design summary table."""
    print("\n" + "=" * 70)
    print("DESIGN SUMMARY")
    print("=" * 70)

    print("""
╔══════════════════════════════════════════════════════════════╗
║  HOGEL DISPLAY DESIGN PARAMETERS                            ║
╠══════════════════════════════════════════════════════════════╣
║  Hogel pitch:     169.3 µm (150 PPI)                        ║
║  Sub-pixel pitch: 0.5 µm → 338×338 sub-pixels per hogel     ║
║  FoV:             ±32° (green), ±27° (blue), ±39° (red)     ║
║  Angular res:     0.18° (Rayleigh)                           ║
║  Resolvable dirs: ~114k per hogel (338²)                     ║
║                                                              ║
║  Phase bits: 3 (8 levels) ← SWEET SPOT                      ║
║    η = 95%, SNR ≈ 3.0, minimal ghost orders                 ║
║    Matches Sb₂Se₃ PCM multi-level capability                 ║
║                                                              ║
║  For blue FoV ≥ 30°: need pitch ≤ 0.45 µm                   ║
║    → 376×376 sub-pixels per hogel                            ║
║    → or accept ±27° for blue channel                         ║
║                                                              ║
║  Display example (1920×1080 hogels = Full HD):               ║
║    Size: 325mm × 183mm (12.8" × 7.2")                       ║
║    Total sub-pixels: 1920×1080 × 338² = 237 billion          ║
║    Phase levels per sub-pixel: 8 (3 bits)                    ║
║    Data per frame (green): 237B × 3 bits = 89 GB             ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print("Hogel Display — Light Field Simulation v3")
    print("=" * 50)
    sim_parallax_sweep()
    sim_3_squares_parallax()
    sim_stereo_pair()
    sim_parameter_grid()
    print_design_summary()
    print("\nDone!")

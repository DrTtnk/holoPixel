"""
Dimensioned cross-section of the eye and the holographic screen.

Every number is the one the optical calculations in this project use, so the
drawing and the physics cannot drift apart. Anatomy is the standard schematic
eye; the screen geometry is derived in docs/holographic_rendering_equation.md
and today's etendue work.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

EYE_R, CORNEA_R, CORNEA_AP = 12.0, 7.8, 11.7
PUPIL_Y, PUPIL_D, ROT_Y, AXIAL = -3.6, 4.0, -13.5, -24.2
RELIEF, FOV, SUBSTRATE, EYEBOX = 20.0, 100.0, 0.7, 10.0
HALF = math.radians(FOV / 2)

CAP_AP = 2 * RELIEF * math.sin(HALF)
CAP_SAG = RELIEF * (1 - math.cos(HALF))
FLAT_D = 2 * RELIEF * math.tan(HALF)
CAP_AREA = 2 * math.pi * RELIEF ** 2 * (1 - math.cos(HALF))

fig, ax = plt.subplots(figsize=(14, 9))

# ── eyeball ────────────────────────────────────────────────────────────
cornea_half = math.asin((CORNEA_AP / 2) / CORNEA_R)
eye_cy = -math.sqrt(EYE_R ** 2 - (CORNEA_AP / 2) ** 2)
t = np.linspace(cornea_half if False else 0, 2 * math.pi, 600)
sclera_start = math.asin((CORNEA_AP / 2) / EYE_R)
th = np.linspace(sclera_start, 2 * math.pi - sclera_start, 400)
ax.plot(EYE_R * np.sin(th), eye_cy + EYE_R * np.cos(th), color="#444", lw=2,
        zorder=3, label="sclera")

tc = np.linspace(-cornea_half, cornea_half, 200)
ax.plot(CORNEA_R * np.sin(tc), -CORNEA_R + CORNEA_R * np.cos(tc),
        color="#3a7fb5", lw=2.5, zorder=4, label="cornea")

for s in (+1, -1):
    ax.plot([s * PUPIL_D / 2, s * CORNEA_AP / 2], [PUPIL_Y, PUPIL_Y],
            color="#2d5f7f", lw=6, solid_capstyle="butt", zorder=4)
ax.plot([-PUPIL_D / 2, PUPIL_D / 2], [PUPIL_Y, PUPIL_Y], color="#111", lw=3,
        zorder=5)
ax.add_patch(Circle((0, ROT_Y), 0.55, color="#e8602c", zorder=6))
ax.annotate("centre of rotation", (0.9, ROT_Y), fontsize=9, color="#e8602c",
            va="center")
ax.plot([0, 0], [AXIAL, RELIEF + PUPIL_Y + 4], color="#999", lw=0.8, ls=(0, (6, 4)),
        zorder=1)

# ── the screen, spherical cap centred on the pupil ─────────────────────
ts = np.linspace(-HALF, HALF, 300)
for off in (0, SUBSTRATE):
    ax.plot((RELIEF + off) * np.sin(ts), PUPIL_Y + (RELIEF + off) * np.cos(ts),
            color="#1f77b4", lw=2.5 if off == 0 else 1.2, zorder=4)
ax.fill_between((RELIEF) * np.sin(ts), PUPIL_Y + RELIEF * np.cos(ts),
                PUPIL_Y + (RELIEF + SUBSTRATE) * np.cos(ts),
                color="#1f77b4", alpha=0.25, zorder=2)

# ── the flat alternative, same field of view ───────────────────────────
ax.plot([-FLAT_D / 2, FLAT_D / 2], [PUPIL_Y + RELIEF] * 2, color="#e8602c",
        lw=2, ls="--", alpha=0.8, zorder=3)
ax.annotate(f"flat alternative, same {FOV:.0f}° field\n"
            f"{FLAT_D:.1f} mm wide — {FLAT_D/CAP_AP:.2f}× the curved one",
            (FLAT_D / 2 + 0.5, PUPIL_Y + RELIEF), fontsize=9, color="#e8602c",
            va="center")

# ── field of view rays from the pupil ──────────────────────────────────
for s in (+1, -1):
    ax.plot([0, s * (RELIEF + 6) * math.sin(HALF)],
            [PUPIL_Y, PUPIL_Y + (RELIEF + 6) * math.cos(HALF)],
            color="#c9a227", lw=1.1, ls=":", zorder=2)
ax.add_patch(Arc((0, PUPIL_Y), 26, 26, angle=90, theta1=-FOV / 2, theta2=FOV / 2,
                 color="#c9a227", lw=1.2))
ax.annotate(f"{FOV:.0f}° field of view", (0, PUPIL_Y + 13.6), fontsize=10,
            color="#8a6d12", ha="center")

# ── eyebox ─────────────────────────────────────────────────────────────
ax.plot([-EYEBOX / 2, EYEBOX / 2], [PUPIL_Y] * 2, color="#c9a227", lw=3,
        alpha=0.55, zorder=3)
ax.annotate(f"eyebox {EYEBOX:.0f} mm", (-EYEBOX / 2 - 0.6, PUPIL_Y - 1.6),
            fontsize=9, color="#8a6d12", ha="right")

# ── dimensions down the axis ───────────────────────────────────────────
def vdim(x, y0, y1, text, colour="#333"):
    ax.annotate("", (x, y0), (x, y1),
                arrowprops=dict(arrowstyle="<->", color=colour, lw=1.1))
    ax.annotate(text, (x + 0.45, (y0 + y1) / 2), fontsize=9, color=colour,
                va="center")

vdim(-17.5, PUPIL_Y, PUPIL_Y + RELIEF, f"eye relief {RELIEF:.0f} mm")
vdim(-22.0, AXIAL, 0.0, f"axial length {abs(AXIAL):.1f} mm")
vdim(19.0, PUPIL_Y + RELIEF * math.cos(HALF), PUPIL_Y + RELIEF,
     f"sagitta {CAP_SAG:.2f} mm", "#1f77b4")

ax.annotate("", (-CAP_AP / 2, PUPIL_Y + RELIEF * math.cos(HALF) - 2.6),
            (CAP_AP / 2, PUPIL_Y + RELIEF * math.cos(HALF) - 2.6),
            arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.2))
ax.annotate(f"cap aperture {CAP_AP:.1f} mm", (0, PUPIL_Y + RELIEF * math.cos(HALF) - 3.6),
            fontsize=10, color="#1f77b4", ha="center")

for y, label in ((0, "corneal apex  y = 0"),
                 (PUPIL_Y, f"entrance pupil  y = {PUPIL_Y}  (⌀{PUPIL_D:.0f} mm)"),
                 (AXIAL, f"retina  y = {AXIAL}")):
    ax.plot([-26, -24.5], [y, y], color="#888", lw=0.9)
    ax.annotate(label, (-26.4, y), fontsize=8.5, color="#555", ha="right",
                va="center")

ax.set_aspect("equal")
ax.set_xlim(-34, 34)
ax.set_ylim(-27, 21)
ax.set_xlabel("mm")
ax.set_ylabel("mm  (optical axis, cornea at 0)")
ax.grid(alpha=0.15)
ax.set_title("Eye and holographic screen — cross-section to scale\n"
             f"cap area {CAP_AREA:.0f} mm²   ·   substrate {SUBSTRATE} mm   ·   "
             f"cap centred on the pupil, so every point faces the eye",
             fontsize=12)
fig.tight_layout()
fig.savefig("plots/eye_screen_section.png", dpi=140, bbox_inches="tight")
print("saved plots/eye_screen_section.png")
print(f"cap aperture {CAP_AP:.2f}  sagitta {CAP_SAG:.2f}  area {CAP_AREA:.1f} mm2  "
      f"flat {FLAT_D:.2f}  nearest edge y={PUPIL_Y + RELIEF*math.cos(HALF):.2f}")

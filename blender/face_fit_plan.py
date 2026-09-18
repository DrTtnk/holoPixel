"""
Plan view: both eyes, the nose, and the screens. Does it fit a face?

Changes from the earlier section, both prompted by the obvious objections that
the eye moves and the nose is in the way:

  * the cap is centred on the CENTRE OF ROTATION, not the pupil. The pupil
    sits 9.9 mm in front of that centre, so a cap of radius relief + 9.9
    keeps the pupil-to-screen distance CONSTANT for every gaze direction.
    Centring on the pupil only holds for a fixed stare.

  * the face is drawn, so the nasal limit is visible rather than assumed.

Anthropometry, adult means: interpupillary distance 63 mm, nasal bridge
half-width 10 mm at eye level, temple 45 mm temporally, spectacle vertex
distance 12-14 mm.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IPD, NASAL_HALF, TEMPLE = 63.0, 10.0, 45.0
PUPIL_TO_ROT, EYE_R = 9.9, 12.0
FOV = 100.0

def cap(relief, fov_deg=FOV):
    R = relief + PUPIL_TO_ROT
    half = math.radians(fov_deg / 2)
    a = np.linspace(1e-6, math.pi / 2, 200001)
    seen = np.arctan2(R * np.sin(a), R * np.cos(a) - PUPIL_TO_ROT)
    alpha = a[np.searchsorted(seen, half)]
    return R, alpha, 2 * R * math.sin(alpha), 2 * math.pi * R ** 2 * (1 - math.cos(alpha))

fig, axes = plt.subplots(1, 2, figsize=(17, 8.2))

for ax, relief, title in ((axes[0], 20.0, "as designed — 20 mm relief"),
                          (axes[1], 14.0, "spectacle distance — 14 mm relief")):
    R, alpha, aperture, area = cap(relief)
    pitch = 2.0e-6 * math.sqrt(area / cap(20.0)[3])
    n_px = area * 1e-6 / pitch ** 2

    for s in (-1, +1):                                   # both eyes
        ax_x = s * IPD / 2
        rot_y = -PUPIL_TO_ROT                            # pupil at y = 0
        th = np.linspace(0, 2 * math.pi, 300)
        ax.plot(ax_x + EYE_R * np.cos(th), rot_y - 1.0 + EYE_R * np.sin(th),
                color="#666", lw=1.4)
        ax.plot(ax_x, 0, "o", color="#111", ms=5)

        t = np.linspace(-alpha, alpha, 300)
        ax.plot(ax_x + R * np.sin(t), rot_y + R * np.cos(t), color="#1f77b4", lw=3)
        for e in (-alpha, alpha):
            ax.plot([ax_x, ax_x + R * math.sin(e)], [0, rot_y + R * math.cos(e)],
                    color="#c9a227", lw=0.8, ls=":")

    # the nose, as a wedge from the midline
    nose_y = np.linspace(-6, 26, 100)
    ax.fill_betweenx(nose_y, -NASAL_HALF - nose_y * 0.18, NASAL_HALF + nose_y * 0.18,
                     color="#d9b48f", alpha=0.55, zorder=0)
    ax.plot([-NASAL_HALF, -NASAL_HALF - 26 * 0.18], [-6 + 0, 26], color="#a07a52", lw=1)
    ax.plot([NASAL_HALF, NASAL_HALF + 26 * 0.18], [-6 + 0, 26], color="#a07a52", lw=1)
    ax.annotate("nose", (0, 20), ha="center", fontsize=10, color="#8a6440")
    ax.axvline(0, color="#aaa", lw=0.7, ls=(0, (6, 4)))

    nasal_edge = IPD / 2 - aperture / 2
    clearance = nasal_edge - NASAL_HALF
    ax.annotate("", (NASAL_HALF, -14), (nasal_edge, -14),
                arrowprops=dict(arrowstyle="<->", color="#2a8", lw=1.4))
    ax.annotate(f"{clearance:.1f} mm clear of the nose", ((NASAL_HALF + nasal_edge) / 2, -16),
                ha="center", fontsize=9.5, color="#1a6")

    ax.set_aspect("equal")
    ax.set_xlim(-58, 58)
    ax.set_ylim(-30, 30)
    ax.grid(alpha=0.15)
    ax.set_xlabel("mm from the midline")
    ax.set_title(f"{title}\ncap radius {R:.1f} mm · aperture {aperture:.1f} mm · "
                 f"{area:.0f} mm²\npitch {pitch*1e6:.2f} µm · {n_px/1e9:.3f} Gpx per eye",
                 fontsize=11)

axes[0].set_ylabel("mm (pupil plane at 0, world ahead)")
fig.suptitle("Plan view: both eyes, the nose, and the screen — cap centred on the "
             "centre of rotation so the geometry survives eye movement", fontsize=13)
fig.tight_layout()
fig.savefig("plots/face_fit_plan.png", dpi=135, bbox_inches="tight")
print("saved plots/face_fit_plan.png")
for relief in (12, 14, 16, 20, 24):
    R, alpha, ap, area = cap(relief)
    pitch = 2.0e-6 * math.sqrt(area / cap(20.0)[3])
    print(f"  relief {relief:2.0f} mm: R {R:5.1f}  aperture {ap:5.1f}  area {area:6.0f} mm2  "
          f"pitch {pitch*1e6:5.3f} um  nasal clearance "
          f"{IPD/2 - ap/2 - NASAL_HALF:5.1f} mm")

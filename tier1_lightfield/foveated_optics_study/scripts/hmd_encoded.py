"""Pre-warp content onto the panel for a remapper design, then view it through
the whole simulated headset (hmd_view.py).

    python hmd_encoded.py <design_dir> <out_dir> --work <scratch_dir> [--name NAME] [--fovea]

Uses the Cycles evaluation of the design (<design_dir>/evaluation/views, from
lf_evaluate.py): for every pupil point, which panel pixel each camera ray
reaches. Every panel pixel is set to the mean colour of the content in the
directions of the rays that reach it, over all pupil points (the light-field
encoding). Content: an angle chart, a 5 deg labelled grid with eccentricity
circles every 5 deg and fine rings every 0.5 deg inside 3 deg. With --fovea: a
true-size crop of +/- 5 deg, from its own pupil views at 0.6 arcmin per pixel,
with square-wave gratings of 1-16 arcmin period (the blur ruler).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import hmd_view  # noqa: E402
import lf_evaluate as ev  # noqa: E402
import lf_pipeline as lp  # noqa: E402
import screen_spec as spec  # noqa: E402

HALF_DEG = max(45.0, 5.0 * math.ceil((spec.FIELD_HALF_DEG[0] + 10.0) / 5.0))   # the chart: the field and a margin
CHART_PX = 2700
FOVEAL_HALF_DEG = 5.0
FOVEAL_PX = 3600                 # 1/6 arcmin per chart pixel
PATCH_DEG = 1.2
FOVEA_VIEW_PX, FOVEA_VIEW_FOV_DEG = 1200, 12.0     # 0.6 arcmin per view pixel
# (period arcmin, axis the bars vary along, centre (theta_x, theta_z) deg)
FOVEAL_PATCHES = tuple((p, axis, (x, z)) for p, x in zip((1, 2, 4, 8, 16), (-3.0, -1.5, 0.0, 1.5, 3.0))
                       for axis, z in (("x", 1.0), ("z", -1.0)))


def angle_chart():
    """RGB image over theta_x, theta_z in [-HALF_DEG, HALF_DEG], row 0 at theta_z = -HALF_DEG."""
    fig = plt.figure(figsize=(CHART_PX / 300, CHART_PX / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-HALF_DEG, HALF_DEG)
    ax.set_ylim(-HALF_DEG, HALF_DEG)
    ax.axis("off")
    for (x0, y0), c in zip(((-HALF_DEG, -HALF_DEG), (0, -HALF_DEG), (-HALF_DEG, 0), (0, 0)),
                           ("#402020", "#203a20", "#202040", "#3a3a18")):
        ax.add_patch(plt.Rectangle((x0, y0), HALF_DEG, HALF_DEG, color=c))
    for g in np.arange(-HALF_DEG, HALF_DEG + 1e-9, 5.0):
        ax.plot([g, g], [-HALF_DEG, HALF_DEG], color="white", lw=0.8)
        ax.plot([-HALF_DEG, HALF_DEG], [g, g], color="white", lw=0.8)
    for gx in np.arange(-40.0, 40.1, 10.0):
        for gz in np.arange(-20.0, 20.1, 10.0):
            ax.text(gx + 2.5, gz + 2.5, f"{int(gx)},{int(gz)}", color="white", ha="center", va="center", fontsize=5)
    for r in np.arange(5.0, 45.1, 5.0):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#ff5050", lw=0.8))
    for r in np.arange(0.5, 3.01, 0.5):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#ffd040", lw=0.4))
    ax.text(0, 1.2, "FOVEA", color="white", ha="center", va="center", fontsize=2)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[::-1, :, :3].astype(np.float32) / 255.0
    plt.close(fig)
    return rgb


def foveal_chart():
    """RGB image over theta_x, theta_z in [-FOVEAL_HALF_DEG, FOVEAL_HALF_DEG] (row 0
    at -FOVEAL_HALF_DEG): square-wave gratings of FOVEAL_PATCHES on grey, red
    rings every 1 deg, a white cross through the fovea."""
    g = (np.arange(FOVEAL_PX) + 0.5) / FOVEAL_PX * 2 * FOVEAL_HALF_DEG - FOVEAL_HALF_DEG
    X, Z = np.meshgrid(g, g)
    rgb = np.full((FOVEAL_PX, FOVEAL_PX, 3), 0.35, dtype=np.float32)
    for period, axis, (cx, cz) in FOVEAL_PATCHES:
        inside = (np.abs(X - cx) < PATCH_DEG / 2) & (np.abs(Z - cz) < PATCH_DEG / 2)
        t = (X if axis == "x" else Z) * 60.0 / period
        rgb[inside] = np.where((t[inside] % 1.0) < 0.5, 1.0, 0.0)[:, None]
    r = np.hypot(X, Z)
    step = 2 * FOVEAL_HALF_DEG / FOVEAL_PX
    ring = (np.abs(r - np.round(r)) < step) & (np.round(r) > 0)
    rgb[ring] = (1.0, 0.3, 0.3)
    rgb[(np.abs(X) < step) | (np.abs(Z) < step)] = 1.0
    return rgb


def encode(views_dir, chart, half_deg=HALF_DEG):
    views_dir = Path(views_dir)
    d = np.load(views_dir / "direction.npy").astype(np.float64).reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    tx, tz = (np.degrees(a) for a in lp.field_angles(d))
    n = chart.shape[0]
    col = np.clip(((tx + half_deg) / (2 * half_deg) * n).astype(int), 0, n - 1)
    row = np.clip(((tz + half_deg) / (2 * half_deg) * n).astype(int), 0, n - 1)
    colour = chart[row, col]
    N = spec.PANEL_PIXELS
    total, count = np.zeros((N * N, 3)), np.zeros(N * N)
    k = 0
    while (views_dir / f"pix_{k}.npy").exists():
        pix = np.load(views_dir / f"pix_{k}.npy").ravel()
        ok = pix >= 0
        for c in range(3):
            total[:, c] += np.bincount(pix[ok], weights=colour[ok, c], minlength=N * N)
        count += np.bincount(pix[ok], minlength=N * N)
        k += 1
    if k == 0:
        raise FileNotFoundError(f"no evaluation views in {views_dir}")
    panel = np.where(count[:, None] > 0, total / np.maximum(count, 1)[:, None], 0.0)
    return panel.reshape(N, N, 3), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--work", required=True)
    ap.add_argument("--name", default="hmd")
    ap.add_argument("--fovea", action="store_true",
                    help="true-size foveal crop: its own pupil views over the fovea, a grating chart")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if args.fovea:
        views = Path(args.work) / "fovea_views"
        ev.render_views(args.design_dir, views, resolution=FOVEA_VIEW_PX, fov_deg=FOVEA_VIEW_FOV_DEG)
        chart, half, fov = foveal_chart(), FOVEAL_HALF_DEG, 2 * FOVEAL_HALF_DEG
        views = views / "views"
    else:
        views = Path(args.design_dir) / "evaluation" / "views"
        chart, half, fov = angle_chart(), HALF_DEG, hmd_view.FOV_DEG
    panel, n_views = encode(views, chart, half)
    plt.imsave(out / f"{args.name}_encoded_panel.png", panel[::-1])
    plt.imsave(out / f"{args.name}_target.png", chart[::-1])
    for aperture in (0.0, 4.0):
        print(*hmd_view.build(args.design_dir, out, args.work, f"{args.name}_encoded", aperture, panel_rgb=panel,
                              fov_deg=fov))
    print(f"encoded from {n_views} pupil views")


if __name__ == "__main__":
    main()

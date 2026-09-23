"""Pre-warp content onto the panel for a remapper design, then view it through
the whole simulated headset (hmd_view.py).

    python hmd_encoded.py <design_dir> <out_dir> --work <scratch_dir> [--name NAME]

Uses the Cycles evaluation of the design (<design_dir>/evaluation/views, from
lf_evaluate.py): for every pupil point, which panel pixel each camera ray
reaches. Every panel pixel is set to the mean colour of the content in the
directions of the rays that reach it, over all pupil points (the light-field
encoding). Content: an angle chart, a 5 deg labelled grid with eccentricity
circles every 5 deg and fine rings every 0.5 deg inside 3 deg.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import hmd_view  # noqa: E402
import lf_pipeline as lp  # noqa: E402
import screen_spec as spec  # noqa: E402

HALF_DEG = 45.0
CHART_PX = 2700


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


def encode(design_dir, chart):
    views_dir = Path(design_dir) / "evaluation" / "views"
    d = np.load(views_dir / "direction.npy").astype(np.float64).reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    tx, tz = (np.degrees(a) for a in lp.field_angles(d))
    n = chart.shape[0]
    col = np.clip(((tx + HALF_DEG) / (2 * HALF_DEG) * n).astype(int), 0, n - 1)
    row = np.clip(((tz + HALF_DEG) / (2 * HALF_DEG) * n).astype(int), 0, n - 1)
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
    args = ap.parse_args()
    out = Path(args.out_dir)
    chart = angle_chart()
    panel, n_views = encode(args.design_dir, chart)
    plt.imsave(out / f"{args.name}_encoded_panel.png", panel[::-1])
    plt.imsave(out / f"{args.name}_target.png", chart[::-1])
    for aperture in (0.0, 4.0):
        print(*hmd_view.build(args.design_dir, out, args.work, f"{args.name}_encoded", aperture, panel_rgb=panel))
    print(f"encoded from {n_views} pupil views")


if __name__ == "__main__":
    main()

"""Simulated headset: a remapper design (evaluator contract, lf_evaluate.py) with
the real lenslet array and panel in Cycles, the panel showing a raw test chart,
seen by a camera at the eye's pupil. Saves the scene and the rendered view.

    python hmd_view.py <design_dir> <out_dir> --work <scratch_dir> [--name NAME] [--aperture-mm 0] [--resolution 1024]

The chart is drawn straight onto the panel pixels (no pre-correction), so the
view shows everything the optics do to it: the remapping, its distortion, the
lens mosaic, blur, ghosts and stray light. Aperture 0 is a pinhole; 4 is the
design pupil, focused at infinity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import lf_evaluate as ev  # noqa: E402
import lf_pipeline as lp  # noqa: E402
import screen_spec as spec  # noqa: E402

FOV_DEG = 90.0


def test_chart(n=spec.PANEL_PIXELS):
    """Panel-pixel chart: coloured quadrants, a labelled 8 x 8 grid, circles about
    the centre every 1 mm of panel radius, a centre cross. Row j along +v."""
    fig = plt.figure(figsize=(n / 256, n / 256), dpi=256)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.axis("off")
    q = n / 2
    for (x0, y0), c in zip(((0, 0), (q, 0), (0, q), (q, q)), ("#402020", "#203a20", "#202040", "#3a3a18")):
        ax.add_patch(plt.Rectangle((x0, y0), q, q, color=c))
    step = n / 8
    for k in range(9):
        ax.plot([k * step] * 2, [0, n], color="white", lw=2)
        ax.plot([0, n], [k * step] * 2, color="white", lw=2)
    for i in range(8):
        for j in range(8):
            ax.text((i + 0.5) * step, (j + 0.5) * step, f"{'ABCDEFGH'[i]}{j + 1}", color="white",
                    ha="center", va="center", fontsize=28, weight="bold")
    px_per_mm = n / spec.PANEL_MM
    for r_mm in range(1, 10):
        ax.add_patch(plt.Circle((q, q), r_mm * px_per_mm, fill=False, color="#ff5050", lw=2))
    ax.plot([q - 60, q + 60], [q, q], color="yellow", lw=3)
    ax.plot([q, q], [q - 60, q + 60], color="yellow", lw=3)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[::-1, :, :3].astype(np.float32) / 255.0   # row 0 at v = -half
    plt.close(fig)
    return rgb


def build(design_dir, out_dir, work, name, aperture_mm=0.0, resolution=1024, samples=64, panel_rgb=None,
          fov_deg=FOV_DEG):
    """panel_rgb: (N, N, 3) panel image, row j along +v; default the raw test chart."""
    design_dir, out_dir = Path(design_dir), Path(out_dir)
    design = json.loads((design_dir / "design.json").read_text())
    remapper = (design_dir / design["remapper_npz"]).resolve()
    ev.validate_surfaces(remapper)
    work = Path(work) / name
    work.mkdir(parents=True, exist_ok=True)
    plain = design.get("lenslets") == "none"       # a plain display: the panel itself is the image
    if not plain:
        mesh, gap, _ = ev.lenslet_array(design, spec.PANEL_MM * 1e3, 2)
        np.savez(work / "mla_mesh.npz", verts=mesh.verts, faces=mesh.faces, loop_normals=mesh.loop_normals,
                 face_lens=mesh.face_lens)
    np.save(work / "panel.npy", test_chart() if panel_rgb is None else panel_rgb.astype(np.float32))
    blend = out_dir / f"{name}.blend"
    cfg = {"mode": "display", "index": spec.LENS_INDEX,
           "panel_pose": design["panel_pose"], "gap_um": 0.0 if plain else gap,
           "panel_pixels": spec.PANEL_PIXELS, "pixel_um": spec.PIXEL_UM,
           "camera": {"resolution": resolution, "fov_deg": fov_deg}, "views_mm": [[0.0, 0.0]],
           "tmp_dir": str(work / "exr"), "remapper_npz": str(remapper), "panel_image_npy": str(work / "panel.npy"),
           "save_blend": str(blend.resolve()), "out_npz": str(work / "display.npz"),
           "display": {"samples": samples, "filter_width_px": 1.0, "aperture_radius_mm": aperture_mm / 2.0,
                       "focus_distance_mm": 1e6}}
    if not plain:
        cfg["mla_npz"] = str(work / "mla_mesh.npz")
    image = lp.run_blender(cfg, work)["images"][0]
    png = out_dir / f"{name}_{'pinhole' if aperture_mm == 0 else f'pupil{aperture_mm:g}mm'}.png"
    plt.imsave(png, np.clip(image[::-1] ** (1 / 2.2), 0, 1))
    return blend, png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--work", required=True, help="scratch folder for the panel image and meshes")
    ap.add_argument("--name", default="hmd")
    ap.add_argument("--aperture-mm", type=float, default=0.0)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--fov-deg", type=float, default=FOV_DEG)
    args = ap.parse_args()
    print(*build(args.design_dir, args.out_dir, args.work, args.name, args.aperture_mm, args.resolution,
                 fov_deg=args.fov_deg))


if __name__ == "__main__":
    main()

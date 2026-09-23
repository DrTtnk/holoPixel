"""Build the direct-view (no foveation) light-field screen scene for inspection.

    python build_viz_scene.py --work <scratch dir> --content chart --out ../blender/direct_view_lightfield.blend
    python build_viz_scene.py --work <scratch dir> --content 3d    --out ../blender/direct_view_parallax.blend

Full scale: the screen_spec micro-OLED (2560^2 px of 7.2 um) under ~300k hex lenses
(f = 150 um), 20 mm in front
of the schematic eye's pupil. The panel shows content encoded through a Cycles
calibration from a 0.5 mm hex grid of pupil points, so the saved scene's pupil
camera sees it through the real lenses.

  chart  a 4 deg checker at infinity, tinted by quadrant so flips show
  3d     a sphere at 35 mm, a cube at 70 mm and a backdrop at 600 mm (inside
         the 1 m camera clip range). The target
         light field is Blender's own render of that content through the same
         pupil cameras, so parallax and occlusion come from real geometry.

Check renders go to ../renders/: held-out pupil views against their targets,
and a 4 mm aperture.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

import lf_pipeline as lp
import screen_spec as spec

HERE = Path(__file__).resolve().parent
SQUARE_DEG = 4.0
Y0 = lp.PUPIL_Y_MM
CONTENT_3D = [
    {"type": "sphere", "center": [-2.0, Y0 + 35.0, 0.8], "radius": 2.5, "color": [1.0, 0.45, 0.1],
     "checker_scale": 0.6},
    {"type": "cube", "center": [2.6, Y0 + 70.0, -1.2], "size": 7.0, "rotation": [0.35, 0.5, 0.2],
     "color": [0.15, 0.85, 1.0], "checker_scale": 0.35},
    {"type": "plane", "center": [0.0, Y0 + 600.0, 0.0], "size": 500.0, "rotation": [math.pi / 2, 0.0, 0.0],
     "color": [0.9, 0.9, 0.75], "checker_scale": 0.02},
]


def chart(direction):
    """Checker of 4 deg squares in field angle, tinted by quadrant so flips show."""
    tx, tz = (np.degrees(a) for a in lp.field_angles(direction))
    checker = (np.floor(tx / SQUARE_DEG) + np.floor(tz / SQUARE_DEG)) % 2
    tint = np.stack([0.35 + 0.65 * (tx > 0), 0.35 + 0.65 * (tz > 0), np.full_like(tx, 0.6)], axis=-1)
    return tint * (0.15 + 0.85 * checker[..., None])


def save_png(path, rgb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(path, np.clip(rgb[::-1], 0, 1))


def to_u8(rgb):
    return (np.clip(rgb[::-1], 0, 1) * 255).astype(np.uint8)


def save_parallax(renders, stem, views, shown, target):
    """Contact sheet of the held-out views (display over target), and a GIF
    sweeping the horizontal row of pupil points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    row = np.nonzero(np.abs(views[:, 1]) < 1e-6)[0]
    row = row[np.argsort(views[row, 0])]
    fig, ax = plt.subplots(2, len(row), figsize=(4 * len(row), 8.4))
    for c, k in enumerate(row):
        ax[0, c].imshow(to_u8(shown[k]))
        ax[0, c].set_title(f"screen, pupil x = {views[k, 0]:+.2f} mm")
        ax[1, c].imshow(to_u8(target[k]))
        ax[1, c].set_title("target (no screen)")
        for a in ax[:, c]:
            a.axis("off")
    plt.tight_layout()
    plt.savefig(renders / f"{stem}_parallax_row.png", dpi=60)
    plt.close(fig)
    frames = [Image.fromarray(to_u8(shown[k])) for k in list(row) + list(row[::-1][1:-1])]
    frames[0].save(renders / f"{stem}_parallax_sweep.gif", save_all=True, append_images=frames[1:],
                   duration=250, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--content", choices=("chart", "3d"), required=True)
    args = ap.parse_args()
    work, out = Path(args.work), Path(args.out).resolve()
    renders = out.parent.parent / "renders"
    renders.mkdir(exist_ok=True)

    screen = lp.Screen(panel_pixels=spec.PANEL_PIXELS)
    fov = math.degrees(2 * math.atan(screen.panel_um * 1e-3 / 2 * 1.1 / screen.eye_relief_mm))
    res = 1024
    cal_views = lp.hex_views_mm(0.5, 2.0)
    cal, mesh = lp.calibrate(screen, work / "cal", cal_views, res, fov)
    if args.content == "chart":
        wanted = chart(cal["direction"])
    else:
        wanted = lp.target(screen, work / "target", CONTENT_3D, cal_views, res, fov)
    panel, seen = lp.encode(cal["ids"], wanted, screen.panel_pixels)
    print(f"panel pixels seen by calibration: {seen.mean():.3f}")

    if args.content == "chart":
        pinhole = lp.display(screen, work / "disp", panel, [[0.0, 0.0], [1.2, 0.0]], res, fov, tag="pinhole")
        aperture = lp.display(screen, work / "disp", panel, [[0.0, 0.0]], res, fov, samples=512,
                              aperture_radius_mm=2.0, tag="aperture")[0]
        centre = int(np.argmin(np.linalg.norm(cal_views, axis=1)))
        save_png(renders / "direct_view_target.png", chart(cal["direction"][centre]))
        save_png(renders / "direct_view_pinhole_centre.png", pinhole[0])
        save_png(renders / "direct_view_pinhole_x1p2mm.png", pinhole[1])
        save_png(renders / "direct_view_aperture_4mm.png", aperture)
    else:
        views = lp.hex_views_mm(0.8486, 1.7)
        shown = lp.display(screen, work / "disp", panel, views, res, fov, tag="pinhole")
        target = lp.target(screen, work / "target_eval", CONTENT_3D, views, res, fov)
        save_parallax(renders, "parallax", views, shown, target)
        for depth in (35.0, 600.0):
            img = lp.display(screen, work / "disp", panel, [[0.0, 0.0]], res, fov, samples=512,
                             aperture_radius_mm=2.0, focus_distance_mm=depth, tag=f"aperture_{int(depth)}")[0]
            save_png(renders / f"parallax_aperture_4mm_focus_{int(depth)}mm.png", img)
        # Score only where the screen is seen: a white panel through the same cameras.
        white = lp.display(screen, work / "disp", np.ones_like(panel), views, res, fov, tag="white")
        on_screen = white[..., 0] > 0.5
        err = [float(np.abs(shown[k] - target[k])[on_screen[k]].mean()) for k in range(len(views))]
        print("mean abs error per held-out view, on the screen:", np.round(err, 4).tolist())
        print("fraction of frame on the screen:", np.round(on_screen.mean(axis=(1, 2)), 3).tolist())

    cfg, _ = lp.base_config(screen, work / "build", [[0.0, 0.0]], res, fov)
    image = work / "build" / "panel.npy"
    np.save(image, panel.astype(np.float32))
    cfg.update(mode="build", out_npz=str(work / "build" / "build.npz"), panel_image_npy=str(image),
               save_blend=str(work / "build" / "scene.blend"))
    lp.run_blender(cfg, work / "build")

    row = mesh.centres[np.abs(mesh.centres[:, 1]) < 1.0]
    row = row[np.argsort(row[:, 0])]
    picks = row[np.linspace(0, len(row) - 1, 9).astype(int)] * 1e-3
    viz = {
        "focal_um": screen.focal_um, "eye_relief_mm": screen.eye_relief_mm, "pupil_y_mm": lp.PUPIL_Y_MM,
        "y_vertex_mm": screen.y_vertex_mm, "y_panel_mm": screen.y_flat_mm + screen.gap_um * 1e-3,
        "fan_pupil_mm": [[-1.7, 0.0], [0.0, 0.0], [1.7, 0.0]],
        "fan_lenses_mm": [[float(u), float(v)] for u, v in picks],
        "views_mm": lp.hex_views_mm(0.8486, 1.7).tolist(), "ray_radius_mm": 0.004,
        "overview_span_mm": 50.0, "overview_y_range_mm": [-26.0, 18.0], "out_blend": str(out),
        "content": CONTENT_3D[:2] if args.content == "3d" else [],
    }
    viz_json = work / "viz.json"
    viz_json.write_text(json.dumps(viz, indent=1))
    log = work / "viz.log"
    with open(log, "w") as fh:
        proc = subprocess.run(["blender", "-b", str(work / "build" / "scene.blend"), "--factory-startup",
                               "--python", str(HERE / "lf_viz.py"), "--", str(viz_json)],
                              stdout=fh, stderr=subprocess.STDOUT)
    text = log.read_text()
    if proc.returncode != 0 or "LF_VIZ_DONE" not in text or "Traceback" in text:
        raise RuntimeError(f"viz layer failed (exit {proc.returncode}), log {log}:\n{text}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

"""Venv side of the physical light-field screen model: geometry, Blender runs,
encoding and comparison. Blender is the only optics: every mapping from a
camera ray to a panel pixel comes from a Cycles render, never from a formula.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

import mla_design as mla
import mla_mesh

HERE = Path(__file__).resolve().parent
BLENDER_SCRIPT = HERE / "lf_blender.py"
PUPIL_Y_MM = -3.6


@dataclass(frozen=True)
class Screen:
    panel_pixels: int
    pixel_um: float = 4.0
    side_um: float = 17.37
    focal_um: float = 150.0
    index: float = 1.5
    min_thickness_um: float = 10.0
    subdivisions: int = 3
    eye_relief_mm: float = 20.0
    gap_scale: float = 1.0

    @property
    def panel_um(self):
        return self.panel_pixels * self.pixel_um

    @property
    def radius_um(self):
        return mla.radius_for_focal_length_um(self.focal_um, self.index)

    @property
    def centre_thickness_um(self):
        return self.min_thickness_um + float(mla.sag_um(self.side_um, self.radius_um))

    @property
    def gap_um(self):
        return self.gap_scale * mla.back_focal_gap_um(self.radius_um, self.index, self.centre_thickness_um)

    @property
    def y_vertex_mm(self):
        return PUPIL_Y_MM + self.eye_relief_mm

    @property
    def y_flat_mm(self):
        return self.y_vertex_mm + self.centre_thickness_um * 1e-3


def direct_view_pose(screen):
    """Panel facing the eye on axis: u -> +X, v -> +Z, w (towards the eye) -> -Y."""
    return {"origin_mm": [0.0, screen.y_flat_mm, 0.0],
            "basis": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]}


def hex_views_mm(spacing_mm, radius_mm):
    """Hexagonal lattice of pupil points (x, z) within a disc."""
    n = int(np.ceil(radius_mm / spacing_mm)) + 1
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    x = spacing_mm * (i + 0.5 * j)
    z = spacing_mm * (np.sqrt(3.0) / 2.0) * j
    keep = np.hypot(x, z) <= radius_mm + 1e-9
    pts = np.column_stack([x[keep], z[keep]])
    return pts[np.lexsort((pts[:, 0], pts[:, 1]))]


def run_blender(cfg, work):
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / f"{cfg['mode']}.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))
    log = work / f"{cfg['mode']}.log"
    with open(log, "w") as fh:
        proc = subprocess.run(["blender", "-b", "--factory-startup", "--python", str(BLENDER_SCRIPT),
                               "--", str(cfg_path)], stdout=fh, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
    text = log.read_text()
    if proc.returncode != 0 or "LF_BLENDER_DONE" not in text or "Traceback" in text:
        raise RuntimeError(f"Blender {cfg['mode']} failed (exit {proc.returncode}), log {log}:\n{text}")
    return dict(np.load(cfg["out_npz"]))


def base_config(screen, work, views_mm, resolution, fov_deg):
    mesh = mla_mesh.build(panel_um=screen.panel_um, side_um=screen.side_um, radius_um=screen.radius_um,
                          min_thickness_um=screen.min_thickness_um, subdivisions=screen.subdivisions)
    work.mkdir(parents=True, exist_ok=True)
    npz = work / "mla_mesh.npz"
    np.savez(npz, verts=mesh.verts, faces=mesh.faces, loop_normals=mesh.loop_normals)
    return {
        "mla_npz": str(npz), "index": screen.index, "panel_pose": direct_view_pose(screen),
        "gap_um": screen.gap_um, "panel_pixels": screen.panel_pixels, "pixel_um": screen.pixel_um,
        "camera": {"resolution": resolution, "fov_deg": fov_deg, "aim_distance_mm": screen.eye_relief_mm},
        "views_mm": np.asarray(views_mm).tolist(), "tmp_dir": str(work / "exr"),
    }, mesh


def calibrate(screen, work, views_mm, resolution, fov_deg):
    cfg, mesh = base_config(screen, work, views_mm, resolution, fov_deg)
    cfg.update(mode="calibrate", out_npz=str(work / "calibration.npz"))
    return run_blender(cfg, work), mesh


def display(screen, work, panel_rgb, views_mm, resolution, fov_deg, samples=1,
            filter_width_px=0.01, aperture_radius_mm=0.0, focus_distance_mm=1e6, tag="display"):
    cfg, _ = base_config(screen, work, views_mm, resolution, fov_deg)
    image = work / f"{tag}_panel.npy"
    np.save(image, panel_rgb.astype(np.float32))
    cfg.update(mode="display", out_npz=str(work / f"{tag}.npz"), panel_image_npy=str(image),
               display={"samples": samples, "filter_width_px": filter_width_px,
                        "aperture_radius_mm": aperture_radius_mm, "focus_distance_mm": focus_distance_mm})
    return run_blender(cfg, work)["images"]


def target(screen, work, content, views_mm, resolution, fov_deg, tag="target"):
    """Target colour of every calibration ray: the virtual content rendered alone
    through the same pupil cameras (same position, shift, resolution, samples)."""
    cfg, _ = base_config(screen, work, views_mm, resolution, fov_deg)
    cfg.update(mode="target", content=content, out_npz=str(work / f"{tag}.npz"))
    return run_blender(cfg, work)["images"]


def field_angles(direction):
    """World direction (..., 3) -> (theta_x, theta_z) radians about the +Y axis."""
    return np.arctan2(direction[..., 0], direction[..., 1]), np.arctan2(direction[..., 2], direction[..., 1])


def chart_lookup(chart, theta_x, theta_z, half_fov_rad):
    """Nearest-texel lookup of an angle-indexed chart spanning +/- half_fov_rad."""
    h, w = chart.shape[:2]
    ci = np.clip(((theta_x / half_fov_rad + 1) * 0.5 * w).astype(int), 0, w - 1)
    rj = np.clip(((theta_z / half_fov_rad + 1) * 0.5 * h).astype(int), 0, h - 1)
    return chart[rj, ci]


def encode(ids, target_per_view, panel_pixels):
    """Panel value = mean of the targets of every calibrated camera ray that lands on it.

    ids: (V, H, W, 2) panel (i, j) per camera pixel, -1 where the ray hit no pixel.
    target_per_view: (V, H, W, C) what the eye should see along each camera ray.
    """
    i, j = ids[..., 0].ravel(), ids[..., 1].ravel()
    ok = i >= 0
    flat = j[ok] * panel_pixels + i[ok]
    t = target_per_view.reshape(-1, target_per_view.shape[-1])[ok]
    count = np.bincount(flat, minlength=panel_pixels**2)
    panel = np.stack([np.bincount(flat, weights=t[:, c], minlength=panel_pixels**2)
                      for c in range(t.shape[1])], axis=1)
    seen = count > 0
    panel[seen] /= count[seen, None]
    return panel.reshape(panel_pixels, panel_pixels, -1), seen.reshape(panel_pixels, panel_pixels)


def lens_of_ray(direction, view_mm, screen, centres_um):
    """Lens whose hexagon the ray from `view_mm` crosses at the lens vertex plane (um, panel frame)."""
    from scipy.spatial import cKDTree
    t = (screen.y_vertex_mm - PUPIL_Y_MM) / direction[..., 1]
    x = view_mm[0] + t * direction[..., 0]
    z = view_mm[1] + t * direction[..., 2]
    uv = np.stack([x, z], axis=-1).reshape(-1, 2) * 1e3
    _, lens = cKDTree(centres_um).query(uv)
    return lens.reshape(direction.shape[:-1])

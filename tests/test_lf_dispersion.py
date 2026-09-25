"""Dispersive glass in Cycles: a glass index given per channel (R, G, B) makes
each colour refract at its own index. A wedge prism between the pupil and a
panel line must put the line's red, green and blue images where Snell's law
puts them, traced here independently in the plane of the wedge."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import dispersion as dp  # noqa: E402
import lf_evaluate as ev  # noqa: E402
import lf_pipeline as lp  # noqa: E402

pytestmark = pytest.mark.blender

P = lp.PUPIL_Y_MM
Y0, T, APEX_DEG, HALF = P + 20.0, 3.0, 10.0, 10.0      # prism front plane, thickness at x = 0, apex, half-size
PANEL_Y, N, PIXEL_UM = P + 80.0, 400, 25.0
RES, FOV_DEG = 1024, 30.0
INDICES = dp.channel_indices(1.9, "nlasf46b")


def _prism():
    """Box x, z in [-HALF, HALF]; front face y = Y0, back face y = Y0 + T + x tan(apex)."""
    tan_a = math.tan(math.radians(APEX_DEG))
    v = []
    for x in (-HALF, HALF):
        for z in (-HALF, HALF):
            v.append((x, Y0, z))
            v.append((x, Y0 + T + x * tan_a, z))
    v = np.array(v)                                     # index: 4 * (x > 0) + 2 * (z > 0) + (back)
    quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6)]
    faces = np.array([f for a, b, c, d in quads for f in ((a, b, c), (a, c, d))])
    centre = v.mean(0)
    out = np.einsum("ij,ij->i", np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]),
                    v[faces].mean(1) - centre)
    return v, np.where((out < 0)[:, None], faces[:, ::-1], faces)


def _refract(d, n, eta):
    """Snell for unit d at a surface of unit normal n (against d), eta = n1 / n2."""
    c = -float(d @ n)
    k = 1.0 - eta**2 * (1.0 - c * c)
    return eta * d + (eta * c - math.sqrt(k)) * n


def _panel_x(theta, index):
    """Where the camera ray at angle theta (from +y, towards +x) meets the panel plane."""
    tan_a = math.tan(math.radians(APEX_DEG))
    d = np.array([math.sin(theta), math.cos(theta)])
    p = np.array([0.0, P]) + (Y0 - P) / d[1] * d
    d = _refract(d, np.array([0.0, -1.0]), 1.0 / index)
    nb = np.array([-tan_a, 1.0]) / math.hypot(tan_a, 1.0)          # back plane y - x tan_a = Y0 + T
    t = (Y0 + T - (p[1] - p[0] * tan_a)) / (d[1] - d[0] * tan_a)
    p = p + t * d
    d = _refract(d, -nb, index)
    return p[0] + (PANEL_Y - p[1]) / d[1] * d[0]


def _design(tmp_path, index):
    v, f = _prism()
    np.savez(tmp_path / "remapper.npz", n_surfaces=1, surf0_kind="glass", surf0_verts=v, surf0_faces=f,
             surf0_index=np.asarray(index))
    ev.validate_surfaces(tmp_path / "remapper.npz")
    panel = np.zeros((N, N, 3), np.float32)
    panel[:, N // 2 - 1:N // 2 + 1] = 1.0                                # a 2-pixel vertical line at x = 0
    np.save(tmp_path / "panel.npy", panel)
    return {"index": 1.5, "panel_pose": {"origin_mm": [0.0, PANEL_Y, 0.0],
                                         "basis": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]},
            "gap_um": 0.0, "panel_pixels": N, "pixel_um": PIXEL_UM,
            "camera": {"resolution": RES, "fov_deg": FOV_DEG}, "views_mm": [[0.0, 0.0]],
            "tmp_dir": str(tmp_path / "exr"), "remapper_npz": str(tmp_path / "remapper.npz"),
            "out_npz": str(tmp_path / "out.npz")}


def test_each_channel_refracts_at_its_own_index(tmp_path):
    cfg = {**_design(tmp_path, INDICES), "mode": "display", "panel_image_npy": str(tmp_path / "panel.npy"),
           "display": {"samples": 64, "filter_width_px": 1.0, "aperture_radius_mm": 0.0, "focus_distance_mm": 1e6}}
    image = lp.run_blender(cfg, tmp_path)["images"][0]                    # (row from the bottom, column, rgb)
    rows = slice(RES // 2 - 150, RES // 2 + 150)                         # through the prism, away from its edges
    tan_half = math.tan(math.radians(FOV_DEG / 2))
    col_angle = np.degrees(np.arctan(((np.arange(RES) + 0.5) - RES / 2) / (RES / 2) * tan_half))
    got = []
    for c in range(3):
        profile = image[rows, :, c].mean(0)
        assert profile.max() > 0.2, f"channel {c}: no line seen"
        peak = int(np.argmax(profile))
        win = slice(peak - 8, peak + 9)
        got.append(float((profile[win] * col_angle[win]).sum() / profile[win].sum()))
    want = [math.degrees(brentq(lambda th: _panel_x(th, n), math.radians(-14.9), math.radians(14.9))) for n in INDICES]
    pixel_deg = FOV_DEG / RES
    assert got == pytest.approx(want, abs=0.5 * pixel_deg)
    assert got[2] - got[0] == pytest.approx(want[2] - want[0], rel=0.05)
    assert abs(want[2] - want[0]) > 5 * pixel_deg                         # the test can see dispersion (8 px)


def test_a_pixel_id_render_refuses_a_dispersive_index(tmp_path):
    """Pixel and lens ids need one deterministic path per camera ray: a glass
    with three indices would pick one at random."""
    cfg = {**_design(tmp_path, INDICES), "mode": "calibrate"}
    cfg["camera"]["aim_distance_mm"] = PANEL_Y - P
    with pytest.raises(RuntimeError, match="dispersive"):
        lp.run_blender(cfg, tmp_path)

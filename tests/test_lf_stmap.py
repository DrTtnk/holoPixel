"""The ST-map panel: each panel pixel shows the content in the field direction
its channel's ST-map gives. With no optics, a panel at distance L and the map
of each pixel's own direction, the pupil pinhole must see the content in place;
a red map offset by 1 deg must move only the red image, by 1 deg; the foveal
texture must show only inside its window."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import lf_pipeline as lp  # noqa: E402

pytestmark = pytest.mark.blender

P, L = lp.PUPIL_Y_MM, 50.0
N, PIXEL_UM = 600, 50.0
WIDE_HALF, FOVEA_HALF = 45.0, 5.0
RES, FOV_DEG = 512, 30.0
LINE_TX, FOVEA_LINE_TZ, RED_SHIFT = 3.0, -1.0, 1.0


def _content():
    wide = np.zeros((900, 900, 3), np.float32)                        # 0.1 deg per texel
    col = int(round((LINE_TX + WIDE_HALF) / (2 * WIDE_HALF) * 900))
    wide[:, col - 2:col + 2] = 1.0                                   # vertical line at tx = +3 deg
    fovea = np.zeros((1000, 1000, 3), np.float32)                     # 0.01 deg per texel
    row = int(round((FOVEA_LINE_TZ + FOVEA_HALF) / (2 * FOVEA_HALF) * 1000))
    fovea[row - 8:row + 8, :] = 1.0                                   # horizontal line at tz = -1 deg
    return wide, fovea


def _stmap(shift_deg):
    x = ((np.arange(N) + 0.5) - N / 2) * PIXEL_UM * 1e-3            # column i along +u = +x
    tx = np.degrees(np.arctan(x / L))[None, :].repeat(N, 0) + shift_deg
    tz = np.degrees(np.arctan(x / L))[:, None].repeat(N, 1)           # row j along +v = +z
    return np.stack([(tx + WIDE_HALF) / (2 * WIDE_HALF), (tz + WIDE_HALF) / (2 * WIDE_HALF), np.ones_like(tx)],
                    -1).astype(np.float32)


def test_each_channel_shows_its_own_map_and_the_fovea_its_window(tmp_path):
    wide, fovea = _content()
    np.save(tmp_path / "wide.npy", wide)
    np.save(tmp_path / "fovea.npy", fovea)
    paths = []
    for c, shift in enumerate((RED_SHIFT, 0.0, 0.0)):
        paths.append(str(tmp_path / f"st{c}.npy"))
        np.save(paths[-1], _stmap(shift))
    cfg = {"mode": "display", "index": 1.5,
           "panel_pose": {"origin_mm": [0.0, P + L, 0.0], "basis": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]},
           "gap_um": 0.0, "panel_pixels": N, "pixel_um": PIXEL_UM,
           "camera": {"resolution": RES, "fov_deg": FOV_DEG}, "views_mm": [[0.0, 0.0]],
           "tmp_dir": str(tmp_path / "exr"), "out_npz": str(tmp_path / "out.npz"),
           "panel_stmap": {"stmaps": paths, "wide_npy": str(tmp_path / "wide.npy"), "wide_half_deg": WIDE_HALF,
                           "fovea_npy": str(tmp_path / "fovea.npy"), "fovea_half_deg": FOVEA_HALF},
           "display": {"samples": 16, "filter_width_px": 1.0, "aperture_radius_mm": 0.0, "focus_distance_mm": 1e6}}
    image = lp.run_blender(cfg, tmp_path)["images"][0]                # row from the bottom, column, rgb
    t = math.tan(math.radians(FOV_DEG / 2))
    angle = np.degrees(np.arctan(((np.arange(RES) + 0.5) - RES / 2) / (RES / 2) * t))
    px = FOV_DEG / RES

    def centroid(profile):
        peak = int(np.argmax(profile))
        win = slice(peak - 6, peak + 7)
        return float((profile[win] * angle[win]).sum() / profile[win].sum())

    band = slice(int(RES * 0.8), int(RES * 0.95))                      # rows well above the fovea
    vertical = [centroid(image[band, :, c].mean(0)) for c in range(3)]
    # the red map says each pixel shows 1 deg further right: the red line is seen 1 deg to the left
    assert vertical == pytest.approx([LINE_TX - RED_SHIFT, LINE_TX, LINE_TX], abs=px)
    inside = slice(RES // 2 - 20, RES // 2 + 20)                       # columns at |tx| < 1.2 deg: in the fovea
    assert centroid(image[:, inside, 1].mean(1)) == pytest.approx(FOVEA_LINE_TZ, abs=px)
    outside = slice(int(RES * 0.02), int(RES * 0.12))                  # columns at tx ~ -13 deg: wide texture only
    row = int(np.argmin(np.abs(angle - FOVEA_LINE_TZ)))
    assert image[row - 3:row + 4, outside, 1].max() < 0.05


@pytest.mark.parametrize("switch", [0.0, 1.0])
def test_the_content_switch_picks_the_grids_or_the_scene_panel(tmp_path, switch):
    """CONTENT_SWITCH 0 shows the ST-map grids, 1 the baked scene panel image
    (one texel per panel pixel)."""
    wide, fovea = np.zeros((90, 90, 3), np.float32), np.zeros((100, 100, 3), np.float32)
    wide[...] = (0.9, 0.1, 0.1)
    np.save(tmp_path / "wide.npy", wide)
    np.save(tmp_path / "fovea.npy", fovea + wide[:1, :1])
    scene = np.zeros((N, N, 3), np.float32)
    scene[...] = (0.1, 0.3, 0.8)
    np.save(tmp_path / "scene.npy", scene)
    paths = []
    for c in range(3):
        paths.append(str(tmp_path / f"st{c}.npy"))
        np.save(paths[-1], _stmap(0.0))
    cfg = {"mode": "display", "index": 1.5,
           "panel_pose": {"origin_mm": [0.0, P + L, 0.0], "basis": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]},
           "gap_um": 0.0, "panel_pixels": N, "pixel_um": PIXEL_UM,
           "camera": {"resolution": 64, "fov_deg": FOV_DEG}, "views_mm": [[0.0, 0.0]],
           "tmp_dir": str(tmp_path / "exr"), "out_npz": str(tmp_path / "out.npz"),
           "panel_stmap": {"stmaps": paths, "wide_npy": str(tmp_path / "wide.npy"), "wide_half_deg": WIDE_HALF,
                           "fovea_npy": str(tmp_path / "fovea.npy"), "fovea_half_deg": FOVEA_HALF,
                           "scene": {"npy": str(tmp_path / "scene.npy"), "switch": switch}},
           "display": {"samples": 4, "filter_width_px": 1.0, "aperture_radius_mm": 0.0, "focus_distance_mm": 1e6}}
    centre = lp.run_blender(cfg, tmp_path)["images"][0][32, 32]
    assert centre == pytest.approx((0.1, 0.3, 0.8) if switch else (0.9, 0.1, 0.1), abs=1e-3)

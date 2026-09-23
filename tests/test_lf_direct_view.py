"""Physical light-field screen, direct-view chain: panel -> hex MLA -> pupil.

Every optical claim is checked on Cycles renders through the real lens mesh:
the lenses collimate (one pixel per lens per pupil point), the pixel a pupil
point sees under a lens follows -f/L * p, and a light field encoded from one
set of pupil points is reproduced at pupil points that were never calibrated,
including by a 4 mm finite-aperture eye.

With the panel at the lens focal plane every pixel is one collimated beam, so
for content at infinity the display's angular resolution is one pixel bin,
pixel / f = 26.7 mrad, not the lens pitch. (A pinhole camera shows a sharp
lens mosaic that a real pupil never sees.) The target is therefore a smooth
linear light field, which this chain can represent: the only expected error
is the quantisation to pixel bins.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import lf_pipeline as lp  # noqa: E402

pytestmark = pytest.mark.blender

SCREEN = lp.Screen(panel_pixels=250)
HALF_FOV = math.atan(SCREEN.panel_um * 1e-3 / 2 / SCREEN.eye_relief_mm) * 1.1
RES = 512
CAL_VIEWS = lp.hex_views_mm(0.4, 2.0)
EVAL_VIEWS = lp.hex_views_mm(0.8486, 1.7)


RAMP_RAD = 0.3          # value changes by 1 over 0.3 rad
PIXEL_BIN_RAD = SCREEN.pixel_um / SCREEN.focal_um
QUANTUM = PIXEL_BIN_RAD / RAMP_RAD


def ramp(direction, offset_rad=0.0):
    """Linear light field at infinity: R follows theta_x, G follows theta_z."""
    tx, tz = lp.field_angles(direction)
    return np.stack([0.5 + (tx + offset_rad) / RAMP_RAD, 0.5 + (tz + offset_rad) / RAMP_RAD,
                     np.full_like(tx, 0.5)], axis=-1)


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return tmp_path_factory.mktemp("lf_direct")


@pytest.fixture(scope="module")
def calibration(work):
    views = np.concatenate([CAL_VIEWS, EVAL_VIEWS])
    cal, mesh = lp.calibrate(SCREEN, work / "cal", views, RES, math.degrees(2 * HALF_FOV))
    n = len(CAL_VIEWS)
    return {"ids": cal["ids"][:n], "dir": cal["direction"][:n], "eval_dir": cal["direction"][n:],
            "eval_ids": cal["ids"][n:], "centres": mesh.centres}


def _one_pixel_fraction(ids, direction, view, centres):
    lens = lp.lens_of_ray(direction, view, SCREEN, centres)
    ok = ids[..., 0] >= 0
    pix = ids[..., 1][ok] * SCREEN.panel_pixels + ids[..., 0][ok]
    pairs = np.unique(np.column_stack([lens[ok], pix]), axis=0)
    per_lens = np.bincount(pairs[:, 0])
    per_lens = per_lens[per_lens > 0]
    return np.mean(per_lens == 1)


def test_every_calibration_view_sees_the_panel(calibration):
    valid = (calibration["ids"][..., 0] >= 0).mean(axis=(1, 2))
    in_frame = (1 / 1.1) ** 2
    assert np.all(valid > 0.95 * in_frame)


def test_lenses_collimate_and_a_defocused_screen_does_not(calibration, work):
    view = (0.2667, 0.2667)  # central-lens pixel lands on a pixel centre, not a corner
    cal, mesh = lp.calibrate(SCREEN, work / "focus", [view], RES, math.degrees(2 * HALF_FOV))
    focused = _one_pixel_fraction(cal["ids"][0], cal["direction"][0], view, mesh.centres)
    blurred_screen = lp.Screen(panel_pixels=250, gap_scale=0.7)
    cal_b, _ = lp.calibrate(blurred_screen, work / "defocus", [view], RES, math.degrees(2 * HALF_FOV))
    defocused = _one_pixel_fraction(cal_b["ids"][0], cal_b["direction"][0], view, mesh.centres)
    assert focused > 0.6
    assert defocused < 0.3
    assert focused > 2.5 * defocused


def test_pixel_under_a_lens_follows_minus_f_over_l_times_pupil_offset(calibration):
    c = calibration["centres"]
    k0 = int(np.argmin(np.linalg.norm(c, axis=1)))
    slope = -SCREEN.focal_um / (SCREEN.eye_relief_mm * 1e3)
    worst = 0.0
    for v, view in enumerate(CAL_VIEWS):
        lens = lp.lens_of_ray(calibration["dir"][v], view, SCREEN, c)
        pix = calibration["ids"][v][(lens == k0) & (calibration["ids"][v][..., 0] >= 0)]
        assert len(pix) > 20
        offset = (pix.mean(axis=0) + 0.5) * SCREEN.pixel_um - SCREEN.panel_um / 2
        predicted = slope * np.asarray(view) * 1e3
        worst = max(worst, float(np.max(np.abs(offset - predicted))))
    # quantised to the 4 um pixel grid, plus the thick-lens/paraxial residual
    assert worst <= SCREEN.pixel_um / 2 + 0.6


@pytest.fixture(scope="module")
def displayed(calibration, work):
    panel, seen = lp.encode(calibration["ids"], ramp(calibration["dir"]), SCREEN.panel_pixels)
    fov = math.degrees(2 * HALF_FOV)
    pinhole = lp.display(SCREEN, work / "disp", panel, EVAL_VIEWS, RES, fov)
    aperture = lp.display(SCREEN, work / "disp", panel, [[0.0, 0.0]], RES, fov,
                          samples=1024, aperture_radius_mm=2.0, tag="aperture")[0]
    white = lp.display(SCREEN, work / "disp", np.ones_like(panel), [[0.0, 0.0]], RES, fov,
                       samples=1024, aperture_radius_mm=2.0, tag="vignetting")[0]
    reference = ramp(calibration["eval_dir"])
    mask = calibration["eval_ids"][..., 0] >= 0
    np.savez_compressed(work / "displayed.npz", pinhole=pinhole, aperture=aperture, white=white,
                        reference=reference, mask=mask, panel=panel, seen=seen)
    return pinhole, aperture, white, reference, mask, calibration["eval_dir"]


def test_rendered_views_are_not_blank(displayed):
    pinhole, aperture, white, *_ = displayed
    assert pinhole.max() > 0.5
    assert aperture.max() > 0.02 and white.max() > 0.02


def test_every_pupil_ray_that_meets_the_array_reaches_the_panel(displayed):
    """Etendue bookkeeping: near the frame centre the 4 mm pupil's footprint on
    the array fully contains the 1 mm panel, so a white panel must fill exactly
    (1 mm)^2 / (pi * (2 mm)^2) of the aperture. Lost or doubled rays would show."""
    _, _, white, *_ = displayed
    c = RES // 2
    expected = (SCREEN.panel_um * 1e-3) ** 2 / (math.pi * 2.0**2)
    assert white[c - 32:c + 32, c - 32:c + 32, 0].mean() == pytest.approx(expected, rel=0.03)


def test_held_out_pupil_points_see_the_light_field_to_one_pixel_bin(displayed):
    """Uniform quantisation has mean absolute error quantum / 4. The calibration
    rays do not sit at bin centres, so allow up to quantum / 2."""
    pinhole, _, _, reference, mask, eval_dir = displayed
    for v in range(len(EVAL_VIEWS)):
        m = mask[v]
        err = np.abs(pinhole[v] - reference[v])[m][:, :2].mean()
        control = np.abs(ramp(eval_dir[v], offset_rad=3 * PIXEL_BIN_RAD) - reference[v])[m][:, :2].mean()
        assert QUANTUM / 8 < err < QUANTUM / 2, (EVAL_VIEWS[v], err, QUANTUM)
        assert control > 5 * err


def test_a_four_millimetre_pupil_sees_the_linear_light_field_unblurred(displayed):
    """A symmetric blur leaves a linear function unchanged, so a finite pupil,
    once divided by its own vignetting (a white panel), must see the ramp."""
    _, aperture, white, reference, mask, _ = displayed
    centre = int(np.argmin(np.linalg.norm(EVAL_VIEWS, axis=1)))
    lit = white[..., 0] > 0.5 * white[..., 0].max()
    lit[:64] = lit[-64:] = False
    lit[:, :64] = lit[:, -64:] = False
    assert lit.sum() > 10000
    seen = aperture[lit] / white[lit]
    err = np.abs(seen - reference[centre][lit])[:, :2].mean()
    assert err < QUANTUM / 2, (err, QUANTUM)

"""Evaluator metrics on synthetic view maps (no Blender)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import lf_evaluate as ev  # noqa: E402
import lf_pipeline as lp  # noqa: E402

PIXELS = 20


def _write(tmp_path, views, direction, per_view):
    np.save(tmp_path / "direction.npy", direction)
    for k in range(len(views)):
        entered, pix = per_view(k)
        np.save(tmp_path / f"entered_{k}.npy", np.asarray(entered, dtype=np.int32))
        np.save(tmp_path / f"pix_{k}.npy", np.asarray(pix, dtype=np.int32))


def _centres():
    return np.array([[0.0, 0.0], [np.sqrt(3.0) * ev.SIDE_UM, 0.0]])


def test_a_lens_seen_from_one_pupil_point_fails_the_blur_check(tmp_path):
    """A lens whose rays all come from one pupil point has no measurable spread
    and must fail, not score a perfect zero blur."""
    views = lp.hex_views_mm(0.5, 2.0)
    centre = int(np.argmin(np.linalg.norm(views, axis=1)))
    direction = np.array([[[0.0, 1.0, 0.0], [1e-3, 1.0, 0.0]]])
    _write(tmp_path, views, direction,
           lambda k: ([[0, 1 if k == centre else -1]], [[5, 6 if k == centre else -1]]))
    rep = ev.metrics(tmp_path, views, _centres(), PIXELS)
    per = np.load(tmp_path.parent / "per_lens.npz")
    blur = dict(zip(per["lens"].tolist(), per["blur"].tolist()))
    assert blur[0] == 0.0
    assert np.isinf(blur[1])
    assert rep["blur_pitch"]["single_view_fraction"] == 0.5
    assert np.isinf(rep["blur_pitch"]["p90"])
    assert not ev.acceptance({**rep, "geometry": {"eye_relief_mm": 30.0, "clearance_mm": 30.0}})["checks"]["blur"]


def test_a_sheared_lens_is_sharp_per_pixel_but_spread_over_the_pupil(tmp_path):
    """Each pupil point sees lens 0 in its own direction, through its own pixel,
    with two rays 1e-4 rad apart: the eye sees sharp pixels, so the blur is the
    per-pixel width (1e-4 / 2 RMS about the pair's mean, times sqrt(2) for the
    two-ray sample) while the full-pupil lens spread is large."""
    views = lp.hex_views_mm(0.5, 2.0)
    n = len(views)
    angle = np.linspace(-0.02, 0.02, n)
    direction = np.stack([np.stack([np.sin(angle + dt), np.cos(angle + dt), np.zeros(n)], -1)
                          for dt in (0.0, 1e-4)])                    # camera (2, n)
    entered = np.full((2, n + 1), -1)
    pix = np.full((2, n + 1), -1)
    direction = np.concatenate([direction, np.array([[[0.0, 1.0, 0.0]]] * 2)], axis=1)
    entered[:, n], pix[:, n] = 1, 100                                # a neighbour, for the spacing ratio

    def per_view(k):
        e, p = entered.copy(), pix.copy()
        e[:, k], p[:, k] = 0, k
        return e, p

    _write(tmp_path, views, direction, per_view)
    ev.metrics(tmp_path, views, _centres(), PIXELS)
    per = np.load(tmp_path.parent / "per_lens.npz")
    assert per["blur_rad"][0] == pytest.approx(1e-4 / math.sqrt(2.0), rel=1e-3)
    assert per["lens_spread_rad"][0] > 100 * per["blur_rad"][0]


def test_landing_under_the_neighbour_is_not_a_ghost_but_sharing_a_pixel_is(tmp_path):
    """Lens 0's rays land on pixel 7, physically under lens 1: no ghost, as long
    as no other lens reaches pixel 7. When a far-off lens also reaches it, the
    pixel must serve two field directions: both rays are ghosts."""
    views = lp.hex_views_mm(0.5, 2.0)
    far = np.array([0.3, 1.0, 0.0]) / np.linalg.norm([0.3, 1.0, 0.0])
    direction = np.array([[[0.0, 1.0, 0.0], far]])
    _write(tmp_path, views, direction, lambda k: ([[0, 1]], [[7, 9]]))
    clean = ev.metrics(tmp_path, views, _centres(), PIXELS)
    assert clean["ghost"]["mean"] == 0.0
    _write(tmp_path, views, direction, lambda k: ([[0, 1]], [[7, 7]]))
    shared = ev.metrics(tmp_path, views, _centres(), PIXELS)
    assert shared["ghost"]["mean"] == 1.0


def test_landing_offset_is_the_pupil_centre_landing_point_minus_the_lens_centre(tmp_path):
    views = lp.hex_views_mm(0.5, 2.0)
    direction = np.array([[[0.0, 1.0, 0.0], [1e-3, 1.0, 0.0]]])
    half = PIXELS * ev.PIXEL_UM / 2
    i0 = int(np.floor((0.0 + half) / ev.PIXEL_UM))          # pixel column holding u = 0
    target = 3 * PIXELS // 4 * PIXELS + i0 + 2               # 2 columns right of lens 0, far up
    _write(tmp_path, views, direction, lambda k: ([[0, 1]], [[target, 3]]))
    ev.metrics(tmp_path, views, _centres(), PIXELS)
    per = np.load(tmp_path.parent / "per_lens.npz")
    u = (target % PIXELS + 0.5) * ev.PIXEL_UM - half
    v = (target // PIXELS + 0.5) * ev.PIXEL_UM - half
    got = dict(zip(per["lens"].tolist(), per["landing_um"].tolist()))
    assert got[0] == pytest.approx(np.hypot(u, v), rel=1e-12)


def test_glass_with_one_flipped_face_is_rejected(tmp_path):
    """Closed and with positive volume, but one face wound the wrong way: the
    edge check must catch it (Cycles would refract through it inverted)."""
    cube = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    outward = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    bad = outward.copy()
    bad[0] = bad[0][::-1]
    np.savez(tmp_path / "bad.npz", n_surfaces=1, surf0_kind="glass", surf0_index=1.5, surf0_verts=cube,
             surf0_faces=bad)
    with pytest.raises(ValueError, match="inconsistent"):
        ev.validate_surfaces(tmp_path / "bad.npz")


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# lens 0: four rays near +y on pixel 5 in every view; lens 1, its neighbour, one ray on pixel 9
_LENS0 = [_unit([0.0, 1.0, 0.0]), _unit([0.0, 1.0, 0.0]), _unit([-1e-4, 1.0, 0.0]), _unit([1e-4, 1.0, 0.0])]
_LENS1 = _unit([1e-3, 1.0, 0.0])
_STRAY = _unit([0.5, 1.0, 0.0])                                       # 27 deg away: no pupil parallax reaches it


def _stray_case(tmp_path, stray_lens, stray_pixel):
    """Every view: lens 0's four rays and lens 1's one; the pupil-centre view
    also has one ray from _STRAY that enters stray_lens (-1: no lens) and lands
    on stray_pixel. Returns the report and the per-lens arrays."""
    views = lp.hex_views_mm(0.5, 2.0)
    centre = int(np.argmin(np.linalg.norm(views, axis=1)))
    direction = np.array([_LENS0 + [_STRAY, _LENS1]])

    def per_view(k):
        at_centre = k == centre
        return ([[0, 0, 0, 0, stray_lens if at_centre else -1, 1]],
                [[5, 5, 5, 5, stray_pixel if at_centre else -1, 9]])

    _write(tmp_path, views, direction, per_view)
    rep = ev.metrics(tmp_path, views, _centres(), PIXELS)
    return rep, dict(np.load(tmp_path.parent / "per_lens.npz")), len(views)


def test_a_stray_ray_through_a_lens_is_a_ghost_and_does_not_move_the_lens(tmp_path):
    """A ray far from its lens's direction (more than STRAY_DEG) is stray light:
    its pixel shows wrong content, so it is a ghost of that lens even though no
    other lens reaches the pixel, and it must not pull the lens's direction."""
    rep, per, n = _stray_case(tmp_path, stray_lens=0, stray_pixel=11)
    assert per["mean"][0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    ghost = dict(zip(per["lens"].tolist(), per["ghost"].tolist()))
    assert ghost[0] == pytest.approx(1.0 / (4 * n + 1)) and ghost[1] == 0.0
    assert rep["stray"]["fraction"] == pytest.approx(1.0 / (5 * n + 1))


def test_a_ray_that_reaches_the_panel_through_no_lens_makes_its_pixel_a_ghost(tmp_path):
    """A ray that lands on lens 0's pixel without passing through any lens: the
    eye sees that pixel from a second, wrong direction, so all of lens 0's rays
    there are ghosts; the stray ray adds nothing to the lens's blur."""
    clean, clean_per, n = _stray_case(tmp_path, stray_lens=-1, stray_pixel=-1)
    rep, per, _ = _stray_case(tmp_path, stray_lens=-1, stray_pixel=5)
    ghost = dict(zip(per["lens"].tolist(), per["ghost"].tolist()))
    assert ghost[0] == 1.0 and ghost[1] == 0.0
    assert rep["stray"]["fraction"] == pytest.approx(1.0 / (5 * n + 1))
    assert clean["stray"]["fraction"] == 0.0
    assert np.array_equal(per["blur_rad"], clean_per["blur_rad"])


def test_group_median_is_numpys_median_of_each_group():
    rng = np.random.default_rng(5)
    keys = rng.integers(0, 40, 1000)
    values = rng.normal(size=1000)
    got = ev._group_median(keys, values, 45)
    for k in range(45):
        assert got[k] == (np.median(values[keys == k]) if (keys == k).any() else 0.0)


def test_a_lens_with_only_stray_centre_rays_has_no_direction(tmp_path):
    """Lens 2 is seen at the pupil centre by two rays 30 deg apart: their median
    lies 15 deg from both, so both are stray and the lens has no chief direction;
    it must drop out, not carry NaN offsets."""
    views = lp.hex_views_mm(0.5, 2.0)
    centre = int(np.argmin(np.linalg.norm(views, axis=1)))
    a, b = _unit([np.tan(np.radians(-15.0)), 1.0, 0.0]), _unit([np.tan(np.radians(15.0)), 1.0, 0.0])
    direction = np.array([_LENS0 + [_LENS1, a, b]])
    centres = np.array([[0.0, 0.0], [np.sqrt(3.0) * ev.SIDE_UM, 0.0], [0.0, 3.0 * ev.SIDE_UM]])

    def per_view(k):
        seen = k == centre
        return ([[0, 0, 0, 0, 1, 2 if seen else -1, 2 if seen else -1]],
                [[5, 5, 5, 5, 9, 13 if seen else -1, 14 if seen else -1]])

    _write(tmp_path, views, direction, per_view)
    ev.metrics(tmp_path, views, centres, PIXELS)
    per = np.load(tmp_path.parent / "per_lens.npz")
    assert 2 not in per["lens"].tolist()
    assert np.all(np.isfinite(per["landing_um"]))

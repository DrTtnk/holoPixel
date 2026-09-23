"""The shared acceptance evaluator, checked against analytic direct-view optics
and against a 45 degree flat-mirror fold that must not change any metric."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import foveation_target as ft  # noqa: E402
import lf_evaluate as ev  # noqa: E402
import lf_pipeline as lp  # noqa: E402
import mla_design as mla  # noqa: E402

pytestmark = pytest.mark.blender

PIXELS, FOCAL = 500, 150.0
RELIEF = 20.0
RADIUS = mla.radius_for_focal_length_um(FOCAL, ev.INDEX)
T_C = ev.MIN_THICKNESS_UM + float(mla.sag_um(ev.SIDE_UM, RADIUS))
MIRROR_Y = lp.PUPIL_Y_MM + 8.0


def write_design(path, pose, surfaces):
    path.mkdir(parents=True, exist_ok=True)
    data = {"n_surfaces": len(surfaces)}
    for k, (kind, verts, faces, *extra) in enumerate(surfaces):
        data.update({f"surf{k}_kind": kind, f"surf{k}_verts": verts, f"surf{k}_faces": faces})
        for key, value in (extra[0] if extra else {}).items():
            data[f"surf{k}_{key}"] = value
    np.savez(path / "remapper.npz", **data)
    (path / "design.json").write_text(json.dumps({"focal_um": FOCAL, "panel_pose": pose,
                                                  "remapper_npz": "remapper.npz"}))
    return path


def run(path, work):
    return ev.evaluate(path, work, panel_pixels=PIXELS, resolution=1024, fov_deg=24.0)


@pytest.fixture(scope="module")
def direct(tmp_path_factory):
    base = tmp_path_factory.mktemp("eval_direct")
    y_vertex = lp.PUPIL_Y_MM + RELIEF
    pose = {"origin_mm": [0.0, y_vertex + T_C * 1e-3, 0.0],
            "basis": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]}
    rep = run(write_design(base / "design", pose, []), base / "work")
    return rep, np.load(base / "work" / "per_lens.npz")


@pytest.fixture(scope="module")
def folded(tmp_path_factory):
    base = tmp_path_factory.mktemp("eval_fold")
    vertex = np.array([0.0, MIRROR_Y, RELIEF - 8.0])
    pose = {"origin_mm": (vertex + np.array([0.0, 0.0, T_C * 1e-3])).tolist(),
            "basis": [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]}
    c = np.array([0.0, MIRROR_Y, 0.0])
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0]) / math.sqrt(2.0)
    quad = np.array([c - 6 * e1 - 6 * e2, c + 6 * e1 - 6 * e2, c + 6 * e1 + 6 * e2, c - 6 * e1 + 6 * e2])
    rep = run(write_design(base / "design", pose, [("mirror", quad, np.array([[0, 1, 2], [0, 2, 3]]))]),
              base / "work")
    return rep, np.load(base / "work" / "per_lens.npz")


PRISM_ENTRY = 8.0   # pupil to prism entry face, air
PRISM_GLASS = 8.0   # glass path, the same for every height (right-angle prism)
PRISM_INDEX = 1.42  # critical angle 44.8 deg: half the pupil clearly fails TIR at 45 deg
PRISM_EXIT = RELIEF - PRISM_ENTRY - PRISM_GLASS / PRISM_INDEX  # unfolded: a + t/n + b = L


def prism_fold(tmp_path_factory, name, coated):
    """Right-angle glass prism (legs 8 mm): the axis enters its y = y0 face, meets
    the hypotenuse at 45 deg (above the 41.8 deg critical angle), is totally
    internally reflected up to +Z and leaves the top face towards the panel."""
    base = tmp_path_factory.mktemp(name)
    y0 = lp.PUPIL_Y_MM + PRISM_ENTRY
    tri = [(y0, -4.0), (y0, 4.0), (y0 + 8.0, 4.0)]     # A, C (right angle), D in (y, z)
    verts = np.array([(x, y, z) for x in (-4.0, 4.0) for (y, z) in tri])
    faces = np.array([[0, 1, 2], [3, 5, 4],           # end caps x = -4, x = +4
                      [0, 4, 1], [0, 3, 4],           # entry face A-C (y = y0)
                      [1, 5, 2], [1, 4, 5],           # top face C-D (z = 4)
                      [0, 5, 3], [0, 2, 5]])          # hypotenuse A-D
    extra = {"index": PRISM_INDEX}
    if coated:
        extra["mirror_faces"] = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
    vertex = np.array([0.0, y0 + 4.0, 4.0 + PRISM_EXIT])
    pose = {"origin_mm": (vertex + np.array([0.0, 0.0, T_C * 1e-3])).tolist(),
            "basis": [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]}
    rep = run(write_design(base / "design", pose, [("glass", verts, faces, extra)]), base / "work")
    return rep, np.load(base / "work" / "per_lens.npz")


def test_inward_or_open_glass_is_rejected_before_rendering(tmp_path):
    cube = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    outward = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    for name, faces, message in (("inward", outward[:, ::-1], "wound inwards"),
                                 ("open", outward[:-1], "not closed")):
        np.savez(tmp_path / f"{name}.npz", n_surfaces=1, surf0_kind="glass", surf0_index=1.5,
                 surf0_verts=cube, surf0_faces=faces)
        with pytest.raises(ValueError, match=message):
            ev.validate_surfaces(tmp_path / f"{name}.npz")
    np.savez(tmp_path / "good.npz", n_surfaces=1, surf0_kind="glass", surf0_index=1.5,
             surf0_verts=cube, surf0_faces=outward)
    ev.validate_surfaces(tmp_path / "good.npz")


@pytest.fixture(scope="module")
def tir_fold(tmp_path_factory):
    return prism_fold(tmp_path_factory, "eval_tir", coated=False)


@pytest.fixture(scope="module")
def coated_fold(tmp_path_factory):
    return prism_fold(tmp_path_factory, "eval_coated", coated=True)


def _central(per_lens, radius_deg=1.0):
    return np.hypot(per_lens["tx"], per_lens["tz"]) < radius_deg


def test_direct_view_blur_is_the_pupil_spread_plus_the_lens_aperture(direct):
    _, pl = direct
    views = lp.hex_views_mm(0.5, 2.0)
    pupil_rms = math.sqrt(np.mean(np.sum(views**2, axis=1))) / RELIEF
    aperture_rms = ev.HEX_RMS_PITCH * math.sqrt(3.0) * ev.SIDE_UM * 1e-3 / RELIEF
    expected = math.hypot(pupil_rms, aperture_rms)
    measured = np.median(pl["spread_all_rad"][_central(pl)])
    assert measured == pytest.approx(expected, rel=0.05)


def test_direct_view_pupil_spread_is_mostly_ghosts_for_the_acceptance_metric(direct):
    """The whole-pupil spread (~70 mrad) dwarfs 3 target pitches (~2.7 mrad), so
    the acceptance metric must call almost all of it ghosts: a direct-view
    screen is not a foveated remapper and must fail."""
    rep, pl = direct
    assert np.median(pl["ghost"][_central(pl)]) > 0.9
    assert not rep["accept"]["passed"]


def test_direct_view_is_seen_from_the_whole_pupil(direct):
    _, pl = direct
    assert np.median(pl["fill"][_central(pl)]) > 0.95


def test_direct_view_sampling_ratio_is_local_focal_length_over_eye_relief(direct):
    """Neighbouring lenses sit sqrt(3)*side apart, seen from L = 20 mm; the target
    spacing is sqrt(3)*side / F(theta), so the ratio is F(theta) / L, lens by lens."""
    _, pl = direct
    lenses = pl["ratio_lens"]
    ecc = np.arccos(np.clip(pl["mean"][lenses, 1], -1.0, 1.0))
    central = ecc < np.radians(2.0)
    expected = ft.local_focal_mm(ecc[central]) / RELIEF
    assert np.median(pl["ratio"][central] / expected) == pytest.approx(1.0, rel=0.03)


def test_direct_view_eye_relief_is_measured_to_the_lens_vertex(direct):
    rep, _ = direct
    assert rep["geometry"]["eye_relief_mm"] == pytest.approx(RELIEF, abs=0.01)


def test_a_flat_mirror_fold_changes_no_optical_metric(direct, folded):
    (rd, pd), (rf, pf) = direct, folded
    assert rf["lenses_in_fov"] == pytest.approx(rd["lenses_in_fov"], rel=0.02)
    for key in ("spread_all_rad", "fill"):
        assert np.median(pf[key]) == pytest.approx(np.median(pd[key]), rel=0.02), key
    assert np.median(pf["ratio"]) == pytest.approx(np.median(pd["ratio"]), rel=0.02)
    assert rf["geometry"]["eye_relief_mm"] == pytest.approx(8.0, abs=0.01)


def test_a_coated_glass_prism_fold_matches_the_unfolded_screen(direct, coated_fold):
    """The fold only reaches the panel through the coating, and the evaluator's
    integer-ID check fails unless the path has unit throughput. Unfolded, the
    prism is a glass block, so the screen must look like the direct view at
    L = a + t/n + b = 20 mm, seen from the same pupil."""
    (rd, pd), (rf, pf) = direct, coated_fold
    assert rf["lenses_in_fov"] == pytest.approx(rd["lenses_in_fov"], rel=0.02)
    assert np.median(pf["spread_all_rad"]) == pytest.approx(np.median(pd["spread_all_rad"]), rel=0.02)
    assert np.median(pf["ratio"]) == pytest.approx(np.median(pd["ratio"]), rel=0.02)
    assert np.median(pf["fill"]) == pytest.approx(np.median(pd["fill"]), abs=1.5 / 61)
    assert rf["geometry"]["eye_relief_mm"] == pytest.approx(PRISM_ENTRY, abs=0.01)


def _visible_from_each_view(work, lens):
    views = lp.hex_views_mm(0.5, 2.0)
    return np.array([lens in np.load(work / "views" / f"lens_{k}.npy") for k in range(len(views))])


def test_the_tir_fold_loses_exactly_the_pupil_points_below_the_critical_angle(direct, tir_fold, tmp_path_factory):
    """Uncoated, a pupil ray survives the hypotenuse only if it meets it above the
    critical angle asin(1/n) = 41.8 deg. For the lens on the axis, predict view by
    view (chief ray, Snell at the entry face) which pupil points still see it:
    exactly those the direct view sees AND that survive, apart from views whose
    chief ray lies within 0.3 deg of the critical angle."""
    (_, pd), _ = direct, tir_fold
    base = Path(str(tmp_path_factory.getbasetemp()))
    lens = int(np.argmin(np.linalg.norm(pd["centres"], axis=1)))
    seen_direct = _visible_from_each_view(base / "eval_direct0" / "work", lens)
    seen_tir = _visible_from_each_view(base / "eval_tir0" / "work", lens)
    views = lp.hex_views_mm(0.5, 2.0)
    d_air = np.column_stack([-views[:, 0] / RELIEF, np.ones(len(views)), -views[:, 1] / RELIEF])
    d_air /= np.linalg.norm(d_air, axis=1, keepdims=True)
    lateral = d_air[:, [0, 2]] / PRISM_INDEX                       # Snell at the y = y0 face
    d_glass = np.column_stack([lateral[:, 0], np.sqrt(1 - np.sum(lateral**2, axis=1)), lateral[:, 1]])
    hyp_normal = np.array([0.0, 1.0, -1.0]) / math.sqrt(2.0)
    incidence = np.degrees(np.arccos(np.abs(d_glass @ hyp_normal)))
    critical = np.degrees(np.arcsin(1.0 / PRISM_INDEX))
    decided = np.abs(incidence - critical) > 0.3
    predicted = seen_direct & (incidence > critical)
    assert np.sum(~(incidence > critical) & decided) >= 3
    assert np.array_equal(seen_tir[decided], predicted[decided])

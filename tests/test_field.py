"""The target field of view has one source, screen_spec.FIELD_HALF_DEG, set by
HOLOPIXEL_FIELD_DEG ("<width>x<height>" in degrees, default 70x45). Module-level
tables depend on it, so each case runs in a fresh interpreter."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study" / "scripts"


def _run(code, field=None, shape=None):
    env = {k: v for k, v in os.environ.items() if k not in ("HOLOPIXEL_FIELD_DEG", "HOLOPIXEL_FIELD_SHAPE")}
    if field is not None:
        env["HOLOPIXEL_FIELD_DEG"] = field
    if shape is not None:
        env["HOLOPIXEL_FIELD_SHAPE"] = shape
    out = subprocess.run([sys.executable, "-c", code], cwd=SCRIPTS, env=env, capture_output=True, text=True)
    return out


def test_the_default_field_is_70_by_45():
    out = _run("import screen_spec as s; print(list(s.FIELD_HALF_DEG))")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [35.0, 22.5]


def test_the_field_is_read_from_the_environment_and_derived_everywhere():
    code = ("import json, screen_spec as s, foveation_target as ft, lf_evaluate as ev; "
            "print(json.dumps([list(s.FIELD_HALF_DEG), list(ft.FIELD_HALF_DEG), [ev.FOV_X_DEG, ev.FOV_Z_DEG]]))")
    out = _run(code, "100x80")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [[50.0, 40.0]] * 3


@pytest.mark.parametrize("bad", ["100", "100x", "x80", "100 x 80", "-10x20", "0x45", "100x80x3", "abc"])
def test_a_malformed_field_is_refused(bad):
    out = _run("import screen_spec", bad)
    assert out.returncode != 0 and "HOLOPIXEL_FIELD_DEG" in out.stderr


def test_a_100_by_80_field_samples_the_fovea_at_1_02_arcmin():
    """The retina-matched map over a wider field: the fovea's lens pitch grows
    only by 1.02 / 0.89 (the trade study of 2026-09-25, from the exact foveal
    scale sqrt(GX GZ)); the module's own numeric Jacobian reads 0.90 arcmin at
    70 x 45, its polar table being least accurate at the pole."""
    code = ("import numpy as np, foveation_target as ft; "
            "print(np.degrees(ft.target_pitch_rad(0.0, 0.0)) * 60)")
    wide, base = _run(code, "100x80"), _run(code)
    assert wide.returncode == 0 and base.returncode == 0, wide.stderr + base.stderr
    assert float(base.stdout) == pytest.approx(0.90, abs=0.01)
    assert float(wide.stdout) / float(base.stdout) == pytest.approx(1.02 / 0.89, rel=0.01)


def test_the_search_fields_the_view_cone_and_the_chart_follow_the_field():
    code = ("import sys, json; sys.path.insert(0, '../remapper_designs/freeform_mirror'); "
            "import fold_search as fs, export_fold as ef, hmd_encoded as he; "
            "print(json.dumps([fs.FIELDS_X_DEG[0], fs.FIELDS_X_DEG[-1], fs.FIELDS_Y_DEG[0], fs.FIELDS_Y_DEG[-1], "
            "fs.TOP_FIELD_DEG, max(f[0] for f in ef.FAN_FIELDS), max(f[1] for f in ef.FAN_FIELDS), he.HALF_DEG]))")
    for field, (hx, hz, chart) in (("100x80", (50.0, 40.0, 60.0)), (None, (35.0, 22.5, 45.0))):
        out = _run(code, field)
        assert out.returncode == 0, out.stderr
        assert json.loads(out.stdout) == [0.0, hx, -hz, hz, hz, hx, hz, chart]


def test_the_evaluation_camera_sees_the_whole_field_at_the_same_central_pixel_angle():
    """The Cycles evaluation camera (a square rectilinear view) must see the field
    plus a margin in both axes (at 70 x 45 its old fixed 76 deg, 2048 px), with the
    pixel angle at its centre unchanged, so wider fields are not cut at +-38 deg."""
    code = ("import json, math, lf_evaluate as ev; "
            "print(json.dumps([ev.CAMERA_FOV_DEG, ev.CAMERA_RESOLUTION, "
            "math.tan(math.radians(ev.CAMERA_FOV_DEG / 2)) / ev.CAMERA_RESOLUTION]))")
    base, wide = _run(code), _run(code, "100x80")
    assert base.returncode == 0 and wide.returncode == 0, base.stderr + wide.stderr
    fov, res, pixel = json.loads(base.stdout)
    assert (fov, res) == (76.0, 2048)
    fov_w, res_w, pixel_w = json.loads(wide.stdout)
    assert fov_w == 2 * (50.0 + 3.0)
    assert pixel_w == pytest.approx(pixel, rel=1e-3)


def test_a_design_of_another_field_is_refused_before_rendering(tmp_path):
    """A design records its field; the evaluator must not score it against another."""
    code = ("import json, sys, lf_evaluate as ev; from pathlib import Path; "
            f"d = Path({str(tmp_path)!r}); "
            "(d / 'design.json').write_text(json.dumps({'field_deg': [100.0, 80.0], 'lenslets': 'variable_retina', "
            "'lenslet_flip_v': 1, 'panel_pose': {}, 'remapper_npz': 'r.npz'})); "
            "ev.render_views(d, d / 'work')")
    out = _run(code)
    assert out.returncode != 0 and "field" in out.stderr and "100" in out.stderr


@pytest.mark.parametrize("name, field, shape", [("fov84x61", "84x61.3", "rect"), ("fov88x66", "88x66", "rect"),
                                               ("fov100x80", "100x80", "rect"), ("ell100x80", "100x80", "ellipse")])
def test_the_stored_wide_pancakes_hold_under_their_own_field(name, field, shape):
    """The 2-lens field-continuation results keep every ray under their own field
    (stored_seeds refuses one that loses rays) and their Cycles-evaluated ranks 0
    and 1 can be built (no lens face leaves its sag domain on the exported disc)."""
    code = ("import sys, json, torch; sys.path.insert(0, '../remapper_designs/freeform_mirror'); "
            "import fold_search as fs, pancake_search as ps, offaxis_tracer as ot; "
            f"p = '../remapper_designs/freeform_mirror/results_pancake/best_pancake_el2_glass_{name}.json'; "
            "e = json.load(open(p))[0]; dev = torch.device('cuda'); lay = fs.entry_layout(e, 'pancake'); "
            "ctx = fs.context(dev); x, idx = fs.stored_seeds([p], lay, 'glass', dev, ctx); "
            "b = fs.to_batch(x[:2], idx[:2], lay); _, _, a, d = ot.trace(b, ctx['fields'], ctx['pupil'], diagnostics=True); "
            "print(json.dumps([fs.spec.FIELD_DEG, ps.lens_domain_violation(b, d['points'], a, lay).tolist()]))")
    out = _run(code, field, shape)
    assert out.returncode == 0, out.stderr
    got_field, violation = json.loads(out.stdout)
    assert got_field == [float(v) for v in field.split("x")] + ([] if shape == "rect" else [shape])
    assert violation == [0.0, 0.0]


def test_an_elliptic_field_is_read_and_recorded_with_its_shape():
    """HOLOPIXEL_FIELD_SHAPE=ellipse: the field is the ellipse inscribed in the
    <width>x<height> box. Designs record the shape with the size, so a design of
    one shape is refused under the other; a rectangular field records the size alone."""
    code = ("import json, screen_spec as s; "
            "print(json.dumps([s.FIELD_DEG, s.FIELD_SHAPE, s.in_field([55.9, 0.0, 50.0, 0.0], [0.0, 44.9, 40.0, 0.0]).tolist()]))")
    out = _run(code, "112x90", "ellipse")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [[112.0, 90.0, "ellipse"], "ellipse", [True, True, False, True]]
    base = _run(code, "112x90")
    assert base.returncode == 0, base.stderr
    assert json.loads(base.stdout) == [[112.0, 90.0], "rect", [True, True, True, True]]


def test_an_unknown_field_shape_is_refused():
    out = _run("import screen_spec", "112x90", "circle")
    assert out.returncode != 0 and "HOLOPIXEL_FIELD_SHAPE" in out.stderr


def test_the_target_map_stretches_the_ellipse_to_the_search_s_panel_limit():
    """The ellipse's image reaches the search's panel limit on the left, right, top
    and bottom (foveation_target.EDGE_MARGIN_MM inside the panel edges, the limit
    fold_search keeps the chief rays in); a rectangle's image keeps the whole panel,
    as its stored designs were made for."""
    code = ("import sys, json, numpy as np, screen_spec as s, foveation_target as ft; "
            "sys.path.insert(0, '../remapper_designs/freeform_mirror'); import fold_search as fs; "
            "tx, tz = np.radians(s.field_boundary_deg(2000)); u, v = ft.field_to_panel_mm(tx, tz); "
            "print(json.dumps([float(np.abs(u).max()), float(v.min()), float(v.max()), ft.HALF_PANEL_MM, "
            "ft.EDGE_MARGIN_MM, fs.PANEL_HALF_MM]))")
    out = _run(code, "112x90", "ellipse")
    assert out.returncode == 0, out.stderr
    u_max, v_min, v_max, half, margin, search_half = json.loads(out.stdout)
    assert margin == pytest.approx(0.1) and search_half == pytest.approx(half - margin)
    assert u_max == pytest.approx(half - margin, abs=1e-3)
    assert (v_min, v_max) == pytest.approx((-(half - margin), half - margin), abs=1e-3)
    rect = _run(code, "112x90")
    assert rect.returncode == 0, rect.stderr
    u_max, v_min, v_max, half, margin, search_half = json.loads(rect.stdout)
    assert search_half == pytest.approx(half - margin)
    assert max(u_max, -v_min, v_max) == pytest.approx(half, abs=1e-3)


_ELLIPSE_SEARCH = ("import sys, json, numpy as np; sys.path.insert(0, '../remapper_designs/freeform_mirror'); "
                   "import fold_search as fs, screen_spec as s; "
                   "rho = lambda f: np.hypot(f[:, 0] / s.FIELD_HALF_DEG[0], f[:, 1] / s.FIELD_HALF_DEG[1]); ")


def test_the_search_samples_the_ellipse_and_its_edge_and_rings_beyond_it():
    """Every sampled field lies in the ellipse; the grid points beyond it move onto
    its edge (so the diagonal edge is sampled too); the chief-ray fold grid keeps
    only its points inside; the rings of beyond-field directions surround the ellipse."""
    code = _ELLIPSE_SEARCH + (
        "f = fs.field_grid(); ring = fs.ring_fields(); d = fs.dense_grid(); m = fs.dense_in_field(d); "
        "diag = f[(f[:, 0] > 0.5 * s.FIELD_HALF_DEG[0]) & (np.abs(f[:, 1]) > 0.5 * s.FIELD_HALF_DEG[1])]; "
        "print(json.dumps([float(rho(f).max()), len(diag), float(rho(diag).max()) if len(diag) else 0.0, "
        "float(rho(ring).min()), bool(s.in_field(*d[m].T).all()), bool((~s.in_field(*d[~m].T)).all()), "
        "len(f) == len(np.unique(f, axis=0))]))")
    out = _run(code, "112x90", "ellipse")
    assert out.returncode == 0, out.stderr
    rho_max, n_diag, rho_diag, rho_ring, dense_in, dense_out, unique = json.loads(out.stdout)
    assert rho_max == pytest.approx(1.0, abs=1e-9)
    assert n_diag > 0 and rho_diag == pytest.approx(1.0, abs=1e-9)
    assert rho_ring > 1.0
    assert dense_in and dense_out and unique


def test_the_rectangular_search_fields_are_unchanged():
    """The rectangular field keeps the full product grid and every dense point."""
    code = ("import sys, json, numpy as np; sys.path.insert(0, '../remapper_designs/freeform_mirror'); "
            "import fold_search as fs; d = fs.dense_grid(); "
            "print(json.dumps([len(fs.field_grid()), len(fs.FIELDS_X_DEG) * len(fs.FIELDS_Y_DEG), "
            "bool(fs.dense_in_field(d).all())]))")
    out = _run(code)
    assert out.returncode == 0, out.stderr
    n, product, all_in = json.loads(out.stdout)
    assert n == product and all_in


def test_the_beyond_field_rays_must_land_outside_the_design_s_own_image_of_the_ellipse():
    """The elliptic field's mask opening is each design's own image of the ellipse:
    the polygon of its edge chief rays' landings (fold_search.opening_polygon, the
    x half mirrored). The outside term is the depth of the ring rays inside it,
    plus the margin; the panel term keeps the plain panel."""
    code = ("import sys, json, torch; sys.path.insert(0, '../remapper_designs/freeform_mirror'); "
            "import fold_search as fs, pancake_search as ps, offaxis_tracer as ot; "
            "p = '../remapper_designs/freeform_mirror/results_pancake/best_pancake_el2_glass_fov100x80.json'; "
            "e = json.load(open(p))[0]; dev = torch.device('cuda'); lay = fs.entry_layout(e, 'pancake'); "
            "ctx = fs.context(dev); x, idx = fs.stored_seeds([p], lay, 'glass', dev, ctx, other_field=True); "
            "_, info = fs.residuals(x, idx, lay, ctx); b = fs.to_batch(x, idx, lay); "
            "chief = torch.zeros(1, 2, dtype=torch.float64, device=dev); "
            ""
            "ring, _, ok = ot.trace(b, ctx['ring_fields'], chief); edge, _, _ = ot.trace(b, ctx['boundary_fields'], chief); poly = fs.opening_polygon(edge[:, :, 0]); "
            "out = fs.outside_violation_polygon(ring[:, :, 0], ok[:, :, 0], poly); "
            "exp = fs.W['outside'] * (out**2).sum(1) / out.shape[1]; n = edge.shape[1]; "
            "print(json.dumps([info['outside'].tolist(), exp.tolist(), float((poly[:, :n] - edge[:, :, 0]).abs().max()), "
            "float((poly[:, n:, 0] + edge[:, 1:-1, 0, 0].flip(1)).abs().max()), poly.shape[1] == 2 * n - 2]))")
    out = _run(code, "100x80", "ellipse")
    assert out.returncode == 0, out.stderr
    got, expected, same_half, mirrored, closed = json.loads(out.stdout)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert same_half == 0.0 and mirrored == 0.0 and closed


def test_an_elliptic_design_needs_its_mask_and_the_evaluator_scores_only_the_ellipse(tmp_path):
    """An elliptic design must bring its mask (design.json "mask_npz"), refused before
    rendering; the evaluator's field test and coverage grid are the ellipse's."""
    code = ("import json, numpy as np, screen_spec as s, lf_evaluate as ev; from pathlib import Path; "
            f"d = Path({str(tmp_path)!r}); "
            "gx, gz = ev.coverage_grid(); c = np.array([[np.tan(np.radians(50.0)), 1.0, np.tan(np.radians(40.0))]]); "
            "print(json.dumps([bool(s.in_field(np.degrees(gx), np.degrees(gz)).all()), gx.size, bool(ev._in_fov(c)[0])]), flush=True); "
            "(d / 'design.json').write_text(json.dumps({'field_deg': s.FIELD_DEG, 'lenslets': 'variable_retina', "
            "'lenslet_flip_v': 1, 'panel_pose': {}, 'remapper_npz': 'r.npz'})); "
            "ev.render_views(d, d / 'work')")
    ellipse, rect = _run(code, "112x90", "ellipse"), _run(code, "112x90")
    inside, n_e, corner_in = json.loads(ellipse.stdout)
    assert inside and not corner_in
    assert ellipse.returncode != 0 and "mask_npz" in ellipse.stderr
    inside, n_r, corner_in = json.loads(rect.stdout)
    assert inside and corner_in
    assert n_e / n_r == pytest.approx(math.pi / 4.0, rel=0.01)


def test_an_elliptic_export_writes_the_design_s_own_mask(tmp_path):
    """export_pancake under an elliptic field writes mask.npz (the design's own
    image of the ellipse, in array coordinates) and names it in design.json:
    open at the panel centre, black at the panel corners."""
    code = ("import sys, json, numpy as np; sys.path.insert(0, '../remapper_designs/freeform_mirror'); "
            "import export_pancake as ep, variable_lenslets as vl; "
            "p = '../remapper_designs/freeform_mirror/results_pancake/best_pancake_el2_glass_ell100x80.json'; "
            f"out = ep.export(p, {str(tmp_path / 'd')!r}, 0); d = json.load(open(out / 'design.json')); "
            "m = np.load(out / d['mask_npz']); t = (m['grid_u'], m['grid_v'], m['open']); h = 9000.0; "
            "print(json.dumps(vl.masked(np.array([[0.0, 0.0], [h, h], [-h, h], [h, -h], [-h, -h]]), t).tolist()))")
    out = _run(code, "100x80", "ellipse")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [False, True, True, True, True]

"""The target field of view has one source, screen_spec.FIELD_HALF_DEG, set by
HOLOPIXEL_FIELD_DEG ("<width>x<height>" in degrees, default 70x45). Module-level
tables depend on it, so each case runs in a fresh interpreter."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study" / "scripts"


def _run(code, field=None):
    env = {k: v for k, v in os.environ.items() if k != "HOLOPIXEL_FIELD_DEG"}
    if field is not None:
        env["HOLOPIXEL_FIELD_DEG"] = field
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


@pytest.mark.parametrize("name, field", [("fov84x61", "84x61.3"), ("fov88x66", "88x66")])
def test_the_stored_wide_pancakes_hold_under_their_own_field(name, field):
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
    out = _run(code, field)
    assert out.returncode == 0, out.stderr
    got_field, violation = json.loads(out.stdout)
    assert got_field == [float(v) for v in field.split("x")]
    assert violation == [0.0, 0.0]

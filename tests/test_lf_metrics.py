"""Evaluator metrics on synthetic view maps (no Blender): a lens whose rays all
come from one pupil point has no measurable spread and must fail, not score a
perfect zero blur."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "tier1_lightfield" / "foveated_optics_study" / "scripts"))

import lf_evaluate as ev  # noqa: E402
import lf_pipeline as lp  # noqa: E402


def test_a_lens_seen_from_one_pupil_point_fails_the_blur_check(tmp_path):
    views = lp.hex_views_mm(0.5, 2.0)
    centre = int(np.argmin(np.linalg.norm(views, axis=1)))
    step = 1e-3
    direction = np.array([[[0.0, 1.0, 0.0], [step, 1.0, 0.0]]])
    np.save(tmp_path / "direction.npy", direction)
    for k in range(len(views)):
        lens = np.array([[0, 1 if k == centre else -1]], dtype=np.int32)
        np.save(tmp_path / f"lens_{k}.npy", lens)
    centres = np.array([[0.0, 0.0], [ev.math.sqrt(3.0) * ev.SIDE_UM, 0.0]])
    rep = ev.metrics(tmp_path, views, centres)
    per = np.load(tmp_path.parent / "per_lens.npz")
    blur = dict(zip(per["lens"].tolist(), per["blur"].tolist()))
    assert blur[0] == 0.0
    assert np.isinf(blur[1])
    assert rep["blur_pitch"]["single_view_fraction"] == 0.5
    assert np.isinf(rep["blur_pitch"]["p90"])
    assert not ev.acceptance({**rep, "geometry": {"eye_relief_mm": 30.0, "clearance_mm": 30.0}})["checks"]["blur"]

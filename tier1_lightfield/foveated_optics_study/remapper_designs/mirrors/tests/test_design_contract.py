"""Exercise the whole export path (untrained layout -> npz/json) against the
shared evaluator's own `validate_surfaces`, so a contract break is caught
here rather than during a 1-3 minute Blender run."""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from layout import build_params, solve_initial_layout  # noqa: E402
from export import build_design_json, build_remapper_npz  # noqa: E402
import lf_evaluate as ev  # noqa: E402

APERTURE = ((38.0, 24.0), (22.0, 22.0))


def test_exported_mesh_passes_evaluator_validation(tmp_path):
    layout = solve_initial_layout()
    params = build_params(layout, APERTURE)
    build_remapper_npz(params, APERTURE, tmp_path / "remapper.npz", grid=(21, 21))
    design = build_design_json(params, tmp_path)
    ev.validate_surfaces(tmp_path / "remapper.npz")  # raises on any contract violation

    design = json.loads((tmp_path / "design.json").read_text())
    basis = np.asarray(design["panel_pose"]["basis"])
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(basis) > 0


def test_two_mirror_surfaces_exported():
    layout = solve_initial_layout()
    params = build_params(layout, APERTURE)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "remapper.npz"
        build_remapper_npz(params, APERTURE, path, grid=(11, 11))
        r = np.load(path)
        assert int(r["n_surfaces"]) == 2
        assert str(r["surf0_kind"]) == "mirror"
        assert str(r["surf1_kind"]) == "mirror"

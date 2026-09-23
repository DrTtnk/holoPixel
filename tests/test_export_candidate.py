"""The GPU-search candidate exporter writes solids the evaluator accepts, with
normals that are the true surface normals and vertices on the prescription."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

DLS = (Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study"
       / "remapper_designs" / "coaxial_dls")
sys.path.insert(0, str(DLS))
sys.path.insert(0, str(DLS.parents[1] / "scripts"))

import export_candidate as ec  # noqa: E402
import lf_evaluate as ev  # noqa: E402
import show_candidates as sc  # noqa: E402

CANDIDATE = DLS / "candidates" / "resin_el5.json"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = ec.export(CANDIDATE, tmp_path_factory.mktemp("export"), device="cpu")
    return out, np.load(out / "remapper.npz"), json.loads((out / "design.json").read_text())


def test_every_lens_is_a_closed_outward_solid(exported):
    out, r, _ = exported
    assert int(r["n_surfaces"]) == 5
    ev.validate_surfaces(out / "remapper.npz")


def test_loop_normals_are_outward_and_follow_the_faces(exported):
    """Catches a wrong sign or frame. The loop normals are exact by construction;
    a flat facet differs from them by at most 2.3 deg, on the steep rim margin."""
    _, r, _ = exported
    for k in range(int(r["n_surfaces"])):
        v, f, n = r[f"surf{k}_verts"], r[f"surf{k}_faces"], r[f"surf{k}_normals"].reshape(-1, 3, 3)
        geo = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        geo /= np.linalg.norm(geo, axis=1, keepdims=True)
        assert np.allclose(np.linalg.norm(n, axis=2), 1.0)
        assert np.min(np.einsum("ij,ikj->ik", geo, n)) > math.cos(math.radians(3.0))


def test_lens_vertices_and_lenslet_array_sit_where_the_tracer_puts_them(exported):
    _, r, design = exported
    entry = json.loads(CANDIDATE.read_text())[0]
    _, batch = sc.candidate_solids(entry, "cpu")
    z = batch.z[0].numpy()
    for k in range(int(r["n_surfaces"])):
        v = r[f"surf{k}_verts"]
        on_axis = v[np.hypot(v[:, 0], v[:, 2]) < 1e-12]
        assert sorted(on_axis[:, 1]) == pytest.approx([sc.PUPIL_Y + z[2 * k], sc.PUPIL_Y + z[2 * k + 1]], abs=1e-9)
    thickness_mm = design["panel_pose"]["origin_mm"][1] - (sc.PUPIL_Y + z[-1])
    assert 0.0 < thickness_mm < 0.1

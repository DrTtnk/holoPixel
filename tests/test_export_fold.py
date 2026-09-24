"""The folded-remapper exporter: valid glass, true normals, and meshes where
the tracer's rays actually meet the surfaces (checked by ray casting)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = (Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study"
        / "remapper_designs" / "freeform_mirror")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

import export_fold as ef  # noqa: E402
import fold_search as fs  # noqa: E402
import lf_evaluate as ev  # noqa: E402
import offaxis_tracer as ot  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(4)
    lo, hi = fs.bounds(fs.layout(1), dev)
    flip_u, flip_v = fs.family_orientation(1, "resin", rng, dev, lo, hi)
    lay = fs.layout(1, flip_u, flip_v)
    x, idx, _ = fs.live_seeds(lay, 1, "resin", rng, dev, fs.context(dev), lo, hi, chunk=64)
    entry = {"n_el": 1, "x": x[0].tolist(), "indices": idx[0].tolist(), "material": "resin",
             "flip_u": flip_u, "flip_v": flip_v}
    out = ef.export_entry(entry, tmp_path_factory.mktemp("fold"), device=str(dev))
    batch = fs.to_batch(x, idx, lay)
    ctx = fs.context(dev)
    _, _, alive, diag = ot.trace(batch, ctx["fields"], ctx["pupil"], diagnostics=True)
    return out, np.load(out / "remapper.npz"), diag["points"][0].cpu().numpy(), alive[0].cpu().numpy()


def test_the_corrector_is_valid_glass_the_mirror_a_sheet_and_the_baffle_black(exported):
    out, r, _, _ = exported
    assert [str(r[f"surf{k}_kind"]) for k in range(int(r["n_surfaces"]))] == ["mirror", "glass", "absorber"]
    ev.validate_surfaces(out / "remapper.npz")


def test_loop_normals_follow_the_faces(exported):
    _, r, _, _ = exported
    for k in range(int(r["n_surfaces"]) - 1):                              # the baffle is flat, no normals
        v, f, n = r[f"surf{k}_verts"], r[f"surf{k}_faces"], r[f"surf{k}_normals"].reshape(-1, 3, 3)
        geo = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        geo /= np.linalg.norm(geo, axis=1, keepdims=True)
        assert np.min(np.einsum("ij,ikj->ik", geo, n)) > math.cos(math.radians(3.0))


def test_traced_segments_meet_the_exported_meshes_where_the_tracer_says(exported):
    """Cast every traced segment (pupil -> mirror -> corrector front) against the
    world-frame meshes: the first hit must be at the traced length."""
    _, r, pts, alive = exported
    assert alive.all()
    rng = np.random.default_rng(0)
    F, P = pts.shape[1], pts.shape[2]
    for f, p in zip(rng.integers(0, F, 12), rng.integers(0, P, 12)):
        start = np.array([0.0, 0.0, 0.0])
        start[:2] = fs.pupil_samples()[p] * 2.0
        path = [start, pts[0, f, p], pts[1, f, p]]
        for (a, b), k in zip(zip(path[:-1], path[1:]), (0, 1)):
            wa, wb = ef.to_world(a), ef.to_world(b)
            d = (wb - wa) / np.linalg.norm(wb - wa)
            hit = ev._axis_hit_mm(r[f"surf{k}_verts"], r[f"surf{k}_faces"], wa + 1e-6 * d, d)
            assert hit == pytest.approx(np.linalg.norm(wb - wa), abs=2e-3)


def test_the_lens_vertices_lie_on_the_image_surface_and_face_the_light(exported):
    """The variable-focal array stands with its lens vertices on the design's
    image surface: every traced image point sits at w = h(r) above the array's
    flat face, h the array's own vertex profile, and the light arrives against w."""
    out, _, pts, _ = exported
    design = json.loads((out / "design.json").read_text())
    assert design["lenslets"] == "variable_retina" and "focal_um" not in design
    flip_v = design["lenslet_flip_v"]
    pose = design["panel_pose"]
    basis, origin = np.asarray(pose["basis"]), np.asarray(pose["origin_mm"])
    assert np.linalg.det(basis) == pytest.approx(1.0, abs=1e-12)          # right-handed: the MLA keeps its winding
    img = ef.to_world(pts[-1].reshape(-1, 3))
    local = (img - origin) @ basis.T                                     # (u, v, w) mm
    inside = (np.abs(local[:, 0]) < ef.vl.TABLE_HALF_MM) & (np.abs(local[:, 1]) < ef.vl.TABLE_HALF_MM)
    h_um = ef.vl.vertex_height_um(local[inside, 0] * 1e3, local[inside, 1] * 1e3, flip_v)
    # the tracer follows the bilinear table (20 um grid): nm-exact in bulk, ~2 um on the
    # crease where the lenslet rule switches from the diffraction to the lens-fit limit
    err = np.abs(local[inside, 2] * 1e3 - h_um)
    assert np.percentile(err, 99) < 0.1 and err.max() < 2.5
    incoming = img - ef.to_world(pts[-2].reshape(-1, 3))
    assert np.all(incoming @ basis[2] < 0)


def test_the_exported_optics_cover_both_halves_of_the_field(exported):
    """The search traces theta_x >= 0 only; the mirrored rays (x -> -x) must
    meet the exported meshes too."""
    _, r, pts, _ = exported
    rng = np.random.default_rng(1)
    F, P = pts.shape[1], pts.shape[2]
    flip = np.array([-1.0, 1.0, 1.0])
    for f, p in zip(rng.integers(0, F, 12), rng.integers(0, P, 12)):
        a, b = pts[0, f, p] * flip, pts[1, f, p] * flip
        wa, wb = ef.to_world(a), ef.to_world(b)
        d = (wb - wa) / np.linalg.norm(wb - wa)
        hit = ev._axis_hit_mm(r["surf1_verts"], r["surf1_faces"], wa + 1e-6 * d, d)
        assert hit == pytest.approx(np.linalg.norm(wb - wa), abs=2e-3)


def test_the_baffle_hides_the_hardware_from_the_pupil(exported):
    """Every line of sight from any pupil sample point to a hardware vertex that
    lies above the baffle plane and in front of the face plane meets the baffle first."""
    _, r, _, _ = exported
    k = int(r["n_surfaces"]) - 1
    assert str(r[f"surf{k}_kind"]) == "absorber"
    bv, bf = r[f"surf{k}_verts"], r[f"surf{k}_faces"]
    to_tracer = lambda w: (w - np.array([0.0, ef.lp.PUPIL_Y_MM, 0.0])) @ ef.TRACER_TO_WORLD.T  # noqa: E731
    rng = np.random.default_rng(2)
    hw = r["surf1_verts"]
    t = to_tracer(hw)
    above = (fs.baffle_plane(torch.tensor(t)).numpy() > 0) & (t[:, 2] >= fs.MIN_Z_MM)
    checked = 0
    pupil = fs.pupil_samples()
    for i in rng.choice(np.nonzero(above)[0], 25, replace=False):
        for px, py in pupil:
            start = ef.to_world(np.array([px * 2.0, py * 2.0, 0.0]))
            seg = hw[i] - start
            d = seg / np.linalg.norm(seg)
            assert ev._axis_hit_mm(bv, bf, start, d) < np.linalg.norm(seg) - 1e-6
            checked += 1
    assert checked == 25 * len(pupil)

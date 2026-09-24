"""The pancake exporter: lens 1 is round (a rectangle's corners leave the sag
domain of a strongly curved half-mirror), holds every traced hit, and the
black plate behind it leaves no gap beside the lens."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = (Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study"
        / "remapper_designs" / "freeform_mirror")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

import export_pancake as ep  # noqa: E402
import fold_search as fs  # noqa: E402
import offaxis_tracer as ot  # noqa: E402


@pytest.fixture(scope="module", params=["random_seed", "best_glass", "folded_resin", "spline_glass"])
def exported(request, tmp_path_factory):
    """A random live seed, the best glass design (a strongly curved half-mirror
    whose back surface runs away beyond its footprint) and a folded resin design
    (whose back runs through the front there)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = tmp_path_factory.mktemp("pancake_export")
    if request.param == "random_seed":
        rng = np.random.default_rng(21)
        lo, hi = fs.bounds(fs.layout(1, family="pancake"), dev)
        flips = fs.family_orientation(1, "resin", rng, dev, lo, hi, family="pancake")
        lay = fs.layout(1, *flips, family="pancake")
        x, idx, _ = fs.live_seeds(lay, 1, "resin", rng, dev, fs.context(dev), lo, hi, chunk=256)
        entry = {"n_el": 1, "x": x[0].tolist(), "indices": idx[0].tolist(), "material": "resin",
                 "flip_u": flips[0], "flip_v": flips[1], "spline": {}}
    elif request.param == "spline_glass":
        entry = json.loads((HERE / "results_pancake" / "best_pancake_el1_glass.json").read_text())[0]
        grid, shape = fs.mirror_spline_grid(entry, dev, cells=4, family="pancake")
        n = fs.with_mirror_spline(fs.entry_layout(entry, "pancake"), grid, shape)["size"] - len(entry["x"])
        entry = {**entry, "x": entry["x"] + np.random.default_rng(8).normal(0.0, 0.01, n).tolist(),
                 "spline": {"grid": list(grid), "shape": list(shape)}}
    else:
        name = {"best_glass": "best_pancake_el1_glass.json", "folded_resin": "best_pancake_el1_resin_folded.json"}
        entry = json.loads((HERE / "results_pancake" / name[request.param]).read_text())[0]
    lay = fs.entry_layout(entry, "pancake")
    x = torch.tensor([entry["x"]], dtype=torch.float64, device=dev)
    idx = torch.tensor([entry["indices"]], dtype=torch.float64, device=dev)
    (base / "best.json").write_text(json.dumps([entry]))
    out = ep.export(base / "best.json", base / "design", device=str(dev))
    ctx = fs.context(dev)
    _, _, alive, diag = ot.trace(fs.to_batch(x, idx, lay), ctx["fields"], ctx["pupil"], diagnostics=True)
    pts = diag["points"][0].cpu().numpy()[:, alive[0].cpu().numpy()]
    pts = np.concatenate([pts, pts * np.array([-1.0, 1.0, 1.0])], axis=1)   # both halves of the field
    return np.load(out / "remapper.npz"), pts, out


def _tracer(world):
    """Inverse of export_fold.to_world."""
    return (world - np.array([0.0, ep.ef.lp.PUPIL_Y_MM, 0.0])) @ ep.ef.TRACER_TO_WORLD.T


def _lens_front(data):
    """Tracer-frame (x, y) of lens 1's front grid (its half-mirror faces)."""
    v = _tracer(data["surf1_verts"])
    return v[:ep.ef.GRID ** 2, :2]


def test_lens_one_is_round(exported):
    data, _, _ = exported
    xy = _lens_front(data).reshape(ep.ef.GRID, ep.ef.GRID, 2)
    ring = np.concatenate([xy[0], xy[-1], xy[:, 0], xy[:, -1]])
    centre = xy[ep.ef.GRID // 2, ep.ef.GRID // 2]
    r = np.linalg.norm(ring - centre, axis=1)
    assert r.max() - r.min() == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(xy - centre, axis=-1).max() == pytest.approx(r.max(), abs=1e-9)


def test_lens_one_holds_every_traced_hit(exported):
    data, pts, _ = exported
    xy = _lens_front(data).reshape(ep.ef.GRID, ep.ef.GRID, 2)
    centre = xy[ep.ef.GRID // 2, ep.ef.GRID // 2]
    radius = np.linalg.norm(xy[0, 0] - centre)
    for s in (0, 2, 3):                                                   # half-mirror twice, lens back
        assert np.linalg.norm(pts[s, :, :2] - centre, axis=1).max() < radius - 0.5 * ep.ef.MARGIN_MM


def test_the_plate_behind_lens_one_leaves_no_gap(exported):
    data, _, _ = exported
    k = int(data["n_surfaces"]) - 1
    assert str(data[f"surf{k}_kind"]) == "absorber"
    xy = _lens_front(data).reshape(ep.ef.GRID, ep.ef.GRID, 2)
    centre = xy[ep.ef.GRID // 2, ep.ef.GRID // 2]
    lens_r = np.linalg.norm(xy[0, 0] - centre)
    plate = _tracer(data[f"surf{k}_verts"])
    r = np.linalg.norm(plate[:, :2] - centre, axis=1)
    rim = r[r < 0.75 * ep.PLATE_HALF_MM]                                   # the outer edge is at >= 40 mm
    assert rim.max() < lens_r                                             # the whole rim sits under the lens
    assert rim.min() > lens_r - ep.ef.MARGIN_MM


def test_the_ray_fans_go_forward_back_and_forward(exported):
    """rays.npz (for the side view): pupil, half-mirror, polariser, half-mirror,
    lens back, image; in the world frame the eye looks along +y."""
    _, _, out = exported
    rays = np.load(out / "rays.npz")
    y = rays["paths"][..., 1][rays["alive"]]                                # (rays, points)
    assert rays["alive"].all()
    assert (y[:, 1] > y[:, 2]).all() and (y[:, 3] > y[:, 2]).all() and (y[:, -1] > y[:, 3]).all()


def test_lens_one_stays_between_the_polariser_and_its_own_back_footprint(exported):
    """Outside the traced zones the surfaces are clamped flat: the lens neither
    comes in front of the polariser nor reaches past its back surface's hits
    (a polynomial evaluated beyond its footprint wraps the lens round the panel)."""
    data, pts, _ = exported
    z = _tracer(data["surf1_verts"])[:, 2]
    z_pol = _tracer(data["surf0_verts"])[:, 2].max()
    assert z.min() > z_pol
    assert z.max() <= pts[3, :, 2].max() + ep.ef.MARGIN_MM + 1e-9
    assert z.min() >= pts[[0, 2], :, 2].min() - ep.ef.MARGIN_MM - 1e-9
    plate = _tracer(data[f"surf{int(data['n_surfaces']) - 1}_verts"])[:, 2]
    assert plate.min() > z.max() and plate.max() < z.max() + 0.1         # just behind the lens

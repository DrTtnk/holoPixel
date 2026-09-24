"""An exported pancake renders in Cycles along the ideal fold: the glossy-depth
switched polariser and half-mirror send each pupil-centre camera ray to the
panel pixel our tracer predicts (to the few pixels the lenslets' own refraction
adds), and no ray reaches the panel outside the fold."""
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
import json  # noqa: E402
import lf_evaluate as ev  # noqa: E402
import lf_pipeline as lp  # noqa: E402
import offaxis_tracer as ot  # noqa: E402

pytestmark = pytest.mark.blender


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(21)
    lo, hi = fs.bounds(fs.layout(1, family="pancake"), dev)
    flips = fs.family_orientation(1, "resin", rng, dev, lo, hi, family="pancake")
    lay = fs.layout(1, *flips, family="pancake")
    x, idx, _ = fs.live_seeds(lay, 1, "resin", rng, dev, fs.context(dev), lo, hi, chunk=256)
    base = tmp_path_factory.mktemp("pancake_cycles")
    entry = {"n_el": 1, "x": x[0].tolist(), "indices": idx[0].tolist(), "material": "resin",
             "flip_u": flips[0], "flip_v": flips[1], "spline": {}}
    (base / "best.json").write_text(json.dumps([entry]))
    design = ep.export(base / "best.json", base / "design", device=str(dev))
    work = base / "work"
    ev.render_views(design, work, resolution=256, fov_deg=50.0, view_spacing_mm=2.0)
    return design, work, lay, x, idx, dev


def test_camera_rays_land_where_the_tracer_puts_them(rendered):
    design_dir, work, lay, x, idx, dev = rendered
    views = lp.hex_views_mm(2.0, 2.0)
    k = int(np.argmin(np.linalg.norm(views, axis=1)))
    pix = np.load(work / "views" / f"pix_{k}.npy").ravel()
    d = np.load(work / "views" / "direction.npy").reshape(-1, 3).astype(np.float64)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    tx, tz = np.degrees(np.arctan2(d[:, 0], d[:, 1])), np.degrees(np.arctan2(d[:, 2], d[:, 1]))
    lit = np.nonzero(pix >= 0)[0]
    assert len(lit) > 500
    sel = np.random.default_rng(0).choice(lit, 400, replace=False)
    fields = torch.tensor(np.stack([-tx[sel], tz[sel]], 1), dtype=torch.float64, device=dev)   # tracer x = -world x
    with torch.no_grad():
        _, _, alive, diag = ot.trace(fs.to_batch(x, idx, lay), fields,
                                     torch.zeros(1, 2, dtype=torch.float64, device=dev), diagnostics=True)
    ok = alive[0, :, 0].cpu().numpy()
    assert ok.mean() > 0.95                                               # Cycles' lit pixels are the fold's
    img = ep.ef.to_world(diag["points"][0, -1, :, 0].cpu().numpy())
    pose = json.loads((design_dir / "design.json").read_text())["panel_pose"]
    uv = (img - np.asarray(pose["origin_mm"])) @ np.asarray(pose["basis"])[:2].T
    n = ev.spec.PANEL_PIXELS
    pred = np.floor((uv + ev.spec.PANEL_MM / 2) / (ev.PIXEL_UM * 1e-3)).astype(int)
    got = np.stack([pix[sel] % n, pix[sel] // n], 1)
    err = np.linalg.norm(pred - got, axis=1)[ok]
    assert np.median(err) < 10 and np.percentile(err, 90) < 25            # lenslet refraction: a few pixels

"""Pancake remapper family: the three-pass fold traces like optiland, the rays
really go forward, back and forward, and the shared merit is finite."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = (Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study"
        / "remapper_designs" / "freeform_mirror")
sys.path.insert(0, str(HERE))

import fold_search as fs  # noqa: E402
import offaxis_optiland as oo  # noqa: E402
import offaxis_tracer as ot  # noqa: E402
import pancake_search as ps  # noqa: E402


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module", params=[1, 2])
def designs(request, device):
    rng = np.random.default_rng(10 + request.param)
    lo, hi = fs.bounds(fs.layout(request.param, family="pancake"), device)
    lay = fs.layout(request.param, *fs.family_orientation(request.param, "resin", rng, device, lo, hi,
                                                         family="pancake"), family="pancake")
    x, idx, _ = fs.live_seeds(lay, 4, "resin", rng, device, fs.context(device), lo, hi, chunk=256)
    return lay, x, idx


def test_the_three_passes_trace_like_optiland(designs, device):
    """Up to the last lens surface (optiland cannot hold the tabulated lenslet
    surface), every hit point matches optiland's to 1e-6 mm."""
    lay, x, idx = designs
    ctx = fs.context(device)
    fields, pupil = ctx["fields"][::7], ctx["pupil"][::9]
    _, _, alive, diag = ot.trace(fs.to_batch(x, idx, lay), fields, pupil, diagnostics=True)
    for b in range(len(x)):
        rx = fs.to_prescription(x[b], idx[b], lay)
        rx = {"surfaces": rx["surfaces"][:-1]}                             # drop the image surface
        lens = oo.trace(rx, fields.cpu().numpy(), pupil.cpu().numpy())
        ok = alive[b].cpu().numpy().ravel()
        for s in range(len(rx["surfaces"])):
            k = s + 2                                                       # optiland: object, stop, then ours
            ref = np.stack([np.asarray(lens.surfaces.x[k]), np.asarray(lens.surfaces.y[k]),
                            np.asarray(lens.surfaces.z[k])], -1)
            got = diag["points"][b, s].reshape(-1, 3).cpu().numpy()
            assert got[ok] == pytest.approx(ref[ok], abs=1e-6), (b, s)


def test_the_light_goes_forward_back_and_forward(designs, device):
    lay, x, idx = designs
    ctx = fs.context(device)
    _, _, alive, diag = ot.trace(fs.to_batch(x, idx, lay), ctx["fields"], ctx["pupil"], diagnostics=True)
    z = diag["points"][..., 2]                                           # (B, S, F, P)
    hm, pol, hm2, back = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    a = alive
    assert bool((pol[a] < hm[a]).all()) and bool((hm2[a] > pol[a]).all()) and bool((back[a] > hm2[a]).all())
    assert bool((pol[a] >= ps.EYE_RELIEF_MIN_MM).all())


def test_merit_and_its_gradient_are_finite(designs, device):
    lay, x, idx = designs
    xg = x.clone().requires_grad_(True)
    loss, info = fs.merit(xg, idx, lay, fs.context(device))
    assert torch.isfinite(loss).all() and bool((info["alive_all"] == 1.0).all())
    loss.sum().backward()
    assert torch.isfinite(xg.grad).all()

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
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

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


def test_the_lens_vertices_stand_up_towards_the_light(designs, device):
    """Light reaches the pancake's image surface along +z, so the lens vertex
    surface there is -(h - h0): a lens that stands taller sits nearer the eye."""
    from scipy.interpolate import RegularGridInterpolator
    lay, x, idx = designs
    ctx = fs.context(device)
    batch = fs.to_batch(x, idx, lay)
    _, d_img, alive, diag = ot.trace(batch, ctx["fields"], ctx["pupil"], diagnostics=True)
    assert bool((d_img[..., 2][alive] > 0).all())
    p = diag["points"][:, -1][alive].cpu().numpy()
    z_img = batch.z[:, -1, None, None].expand(alive.shape)[alive].cpu().numpy()
    gu, gv, h = fs.vl.bowl_table(lay["flip_v"])
    inside = (np.abs(p[:, 0]) < gu[-1]) & (np.abs(p[:, 1]) < gv[-1])
    expected = -RegularGridInterpolator((gu, gv), h)(p[inside, :2])
    assert (p[inside, 2] - z_img[inside]) == pytest.approx(expected, abs=1e-9)



STORED = HERE / "results_pancake" / "best_pancake_el1_glass.json"


def test_stored_designs_seed_the_search_exactly(device):
    import json
    entry = json.loads(STORED.read_text())[0]
    lay = fs.layout(1, entry["flip_u"], entry["flip_v"], family="pancake")
    x, idx = fs.stored_seeds([STORED], lay, "glass", device, fs.context(device))
    assert x[0].tolist() == entry["x"] and idx[0].tolist() == entry["indices"]


@pytest.mark.parametrize("material, flip_v", [("resin", 1), ("glass", -1)])
def test_a_stored_design_of_another_material_or_orientation_is_refused(device, material, flip_v):
    import json
    entry = json.loads(STORED.read_text())[0]
    lay = fs.layout(1, entry["flip_u"], flip_v * entry["flip_v"], family="pancake")
    with pytest.raises(ValueError):
        fs.stored_seeds([STORED], lay, material, device, fs.context(device))


def test_one_spline_shapes_both_passes_of_the_half_mirror(device):
    """The half-mirror is met twice (reflect, then transmit into lens 1): its
    spline must be the same on both slots, and zero controls change nothing."""
    import json
    entry = json.loads(STORED.read_text())[0]
    base = fs.entry_layout(entry, "pancake")
    lay = fs.with_mirror_spline(base, *fs.mirror_spline_grid(entry, device, cells=4, family="pancake"))
    ctx = fs.context(device)
    x0, idx = fs.stored_seeds([STORED], base, "glass", device, ctx)
    x, _ = fs.stored_seeds([STORED], lay, "glass", device, ctx)
    r0, _ = fs.residuals(x0, idx, base, ctx)
    r1, _ = fs.residuals(x, idx, lay, ctx)
    assert r1.cpu().numpy() == pytest.approx(r0.cpu().numpy(), abs=1e-9)
    x[0, lay["spline"]["slice"]] = 0.01
    batch = fs.to_batch(x, idx, lay)
    assert batch.spline[0] == (0, 2)
    _, _, alive, diag = ot.trace(batch, ctx["fields"], ctx["pupil"], diagnostics=True)
    for s in (0, 2):                                                      # both hits lie on the splined sag
        p = diag["points"][0, s][alive[0]]
        c, k, C = batch.c[:1, s], batch.k[:1, s], batch.xy[:1, s]
        f, _, _, _ = ot.sag(p[None, :, 0], p[None, :, 1], c[:, None], k[:, None], C[:, None])
        e, _, _ = ot.bspline_sag(p[None, :, 0], p[None, :, 1], batch.spline[1], batch.spline[2])
        assert (p[:, 2] - batch.z[0, s]).cpu().numpy() == pytest.approx((f + e)[0].cpu().numpy(), abs=1e-8)

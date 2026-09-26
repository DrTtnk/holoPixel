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


def test_the_jacobian_is_the_same_in_chunks_of_columns(designs, device):
    """fd_jacobian traces several perturbed columns in one batch: the columns (and
    the frozen zero ones) must not depend on the chunk size."""
    lay, x, idx = designs
    ctx = fs.context(device)
    lo, hi = fs.bounds(lay, device)
    x = torch.minimum(x, hi - 1e-6)
    x[0, 0] = hi[0] - 1e-6                                                # a step that turns back at the bound
    free = torch.ones(lay["size"], dtype=torch.bool, device=device)
    free[5] = False
    with torch.no_grad():
        r, _ = fs.residuals(x, idx, lay, ctx)
        one = fs.fd_jacobian(x, idx, lay, ctx, r, free, lo, hi, chunk=1)
        many = fs.fd_jacobian(x, idx, lay, ctx, r, free, lo, hi, chunk=4)
    assert not one[:, :, 5].any() and one[:, :, 0].abs().max() > 0
    assert many.cpu().numpy() == pytest.approx(one.cpu().numpy(), rel=1e-6, abs=1e-6 * float(one.abs().max()))


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
        f, _, _, _ = ot.sag(p[None, :, 0], p[None, :, 1], c[:, None], k[:, None], C[:, None], batch.terms)
        e, _, _ = ot.bspline_sag(p[None, :, 0], p[None, :, 1], batch.spline[1], batch.spline[2])
        assert (p[:, 2] - batch.z[0, s]).cpu().numpy() == pytest.approx((f + e)[0].cpu().numpy(), abs=1e-8)


SPLINED = HERE / "results_pancake" / "best_pancake_el1_glass_spline4.json"


def test_a_splined_seed_must_match_the_layouts_spline_grid(device):
    """A seed's spline controls only mean something on its own grid: a layout
    whose grid was laid out again (over the seed's own, changed footprint)
    must refuse it, not reinterpret the controls."""
    import json
    entry = json.loads(SPLINED.read_text())[0]
    own = fs.entry_layout(entry, "pancake")
    ctx = fs.context(device)
    x, _ = fs.stored_seeds([SPLINED], own, "glass", device, ctx)
    assert x[0].tolist() == entry["x"]
    grid, shape = fs.mirror_spline_grid(entry, device, cells=4, family="pancake")
    assert tuple(grid) != tuple(entry["spline"]["grid"])                  # the footprint moved in the search
    regrid = fs.with_mirror_spline(fs.layout(1, entry["flip_u"], entry["flip_v"], family="pancake"), grid, shape)
    with pytest.raises(ValueError, match="spline grid"):
        fs.stored_seeds([SPLINED], regrid, "glass", device, ctx)


def test_a_search_seeded_from_a_splined_design_keeps_its_grid(device):
    import json
    entry = json.loads(SPLINED.read_text())[0]
    base = fs.layout(1, entry["flip_u"], entry["flip_v"], family="pancake")
    lay = fs.seeded_layout(base, [SPLINED], 0, device)
    assert lay["spline"]["grid"] == tuple(entry["spline"]["grid"])
    assert lay["spline"]["shape"] == tuple(entry["spline"]["shape"])
    with pytest.raises(ValueError, match="already carries a spline"):
        fs.seeded_layout(base, [SPLINED], 4, device)
    smooth = fs.seeded_layout(base, [STORED], 4, device)                  # a smooth seed: a new grid over it
    assert smooth["spline"]["shape"] == tuple(fs.mirror_spline_grid(json.loads(STORED.read_text())[0], device,
                                                                     4, "pancake")[1])
    assert "spline" not in fs.seeded_layout(base, [STORED], 0, device)


def test_a_stored_design_of_another_field_is_refused(device):
    import json
    entry = json.loads(STORED.read_text())[0]
    lay = fs.layout(1, entry["flip_u"], entry["flip_v"], family="pancake")
    other = STORED.parent.parent / "other_field.json"
    other.write_text(json.dumps([{**entry, "field_deg": [100.0, 80.0]}]))
    try:
        with pytest.raises(ValueError, match="field"):
            fs.stored_seeds([other], lay, "glass", device, fs.context(device))
    finally:
        other.unlink()


def test_a_stored_design_of_another_field_seeds_a_continuation_when_asked(device):
    """Field continuation: a design searched for another field starts the search
    only on an explicit request; the search then scores it under this field."""
    import json
    entry = json.loads(STORED.read_text())[0]
    lay = fs.layout(1, entry["flip_u"], entry["flip_v"], family="pancake")
    other = STORED.parent.parent / "other_field_continue.json"
    other.write_text(json.dumps([{**entry, "field_deg": [100.0, 80.0]}]))
    try:
        x, idx = fs.stored_seeds([other], lay, "glass", device, fs.context(device), other_field=True)
    finally:
        other.unlink()
    assert x[0].tolist() == entry["x"] and idx[0].tolist() == entry["indices"]


def test_a_run_records_the_merit_weights_it_used(tmp_path, device):
    """The map weight is a search setting: every result entry records the weights
    its loss was computed with, so losses of runs with other weights are not
    compared blindly."""
    import json
    weights = fs.weight_overrides(["map=2.5", "tilt=30"])
    assert weights == {**fs.W, "map": 2.5, "tilt": 30.0}
    fs.run(tmp_path, 1, "resin", 4, 1, 0, device, family="pancake", weights=weights, tilt_max_deg=20.0)
    entries = json.loads((tmp_path / "best_pancake_el1_resin.json").read_text())
    assert all(e["weights"] == weights and e["tilt_max_deg"] == 20.0 for e in entries)


@pytest.mark.parametrize("bad", ["mapp=2", "map", "map=x", "=3"])
def test_a_weight_override_must_name_a_merit_term_and_a_number(bad):
    with pytest.raises(ValueError):
        fs.weight_overrides([bad])


def test_the_lens_domain_covers_the_exported_disc(device):
    """The exporter builds each lens over its front's footprint disc plus
    export_fold.MARGIN_MM, and every face of the lens must keep its sag domain
    there. The stored designs export, so their violation is zero; a lens back
    whose conic closes the domain inside the disc is penalised."""
    import json
    ctx = fs.context(device)
    for path in (STORED, SPLINED):
        entry = json.loads(path.read_text())[0]
        lay = fs.entry_layout(entry, "pancake")
        x = torch.tensor([entry["x"]], dtype=torch.float64, device=device)
        idx = torch.tensor([entry["indices"]], dtype=torch.float64, device=device)
        batch = fs.to_batch(x, idx, lay)
        _, _, alive, diag = ot.trace(batch, ctx["fields"], ctx["pupil"], diagnostics=True)
        assert float(ps.lens_domain_violation(batch, diag["points"], alive, lay)[0]) == 0.0
        closed = batch._replace(k=batch.k.clone())
        closed.k[0, 3] = 1.0 / (float(batch.c[0, 3]) ** 2 * 64.0) - 1.0          # domain edge at r = 8 mm
        assert float(ps.lens_domain_violation(closed, diag["points"], alive, lay)[0]) > 0.0


def test_the_evaluator_accepts_the_eye_relief_the_search_allows():
    """The user accepts 15-18 mm of eye relief (2026-09-26): the evaluator's pass
    mark and the pancake search's minimum are the same number."""
    import lf_evaluate as ev
    assert ev.ACCEPT["eye_relief_min_mm"] == ps.EYE_RELIEF_MIN_MM

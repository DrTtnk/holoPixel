"""The folded-remapper search: its parameter-to-prescription route, its pupil
cells and its per-pixel blur, checked against direct computations."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = (Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study"
        / "remapper_designs" / "freeform_mirror")
sys.path.insert(0, str(HERE))

import fold_search as fs  # noqa: E402
import offaxis_tracer as ot  # noqa: E402


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module", params=[1, 2])
def designs(request, device):
    rng = np.random.default_rng(request.param)
    lo, hi = fs.bounds(fs.layout(request.param), device)
    lay = fs.layout(request.param, *fs.family_orientation(request.param, "resin", rng, device, lo, hi))
    x = torch.clamp(fs.random_designs(lay, 6, rng, device), lo, hi)
    x[:, lay["mirror_shape"]] += torch.tensor(rng.normal(0, 0.05, (6, fs.N_SHAPE)), device=device)
    idx = torch.tensor(rng.choice(fs.INDEX_SETS["resin"], (6, request.param)), dtype=torch.float64, device=device)
    return lay, x, idx


def test_the_prescription_traces_exactly_like_the_parameter_batch(designs, device):
    lay, x, idx = designs
    ctx = fs.context(device)
    direct = ot.trace(fs.to_batch(x, idx, lay), ctx["fields"], ctx["pupil"])
    packed = fs.pack([fs.to_prescription(x[b], idx[b], lay) for b in range(len(x))], device)
    via = ot.trace(packed, ctx["fields"], ctx["pupil"])
    assert torch.equal(direct[2], via[2])
    ok = direct[2]
    assert torch.allclose(direct[0][ok], via[0][ok], atol=1e-9)


def test_merit_and_its_gradient_are_finite_on_random_seeds(designs, device):
    lay, x, idx = designs
    x = x.clone().requires_grad_(True)
    loss, _ = fs.merit(x, idx, lay, fs.context(device))
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_pupil_cells_follow_the_pixel_footprint_on_the_pupil():
    """A pixel sees a pupil square of side pixel * F / f_lens (F the target map's
    local focal length, f_lens the lens that field lands on). The lenslet rule
    makes it ~2.4 mm at the fovea (about one view), smaller further out (at
    least 0.8 mm: 5 views along the map's tighter axis); cells are those squares that hold hexagonal pupil points."""
    pupil = fs.pupil_samples()
    fields = np.array([[0.0, 0.0], [10.0, 0.0], [35.0, 0.0], [5.0, -15.0]])
    ids, _ = fs.pupil_cells(fields, pupil)
    uv = pupil * 2.0 + 2.0
    sides = []
    for f, (fx, fy) in enumerate(fields):
        tx, tz = np.radians(fx), np.radians(fy)
        u, v = fs.ft.field_to_panel_mm(tx, tz)
        side = 7.2 * float(fs.ft.local_focal_mm(tx, tz)) / float(fs.ft.lenslet_focal_um(u * 1e3, v * 1e3))
        sides.append(side)
        n = max(1, math.ceil(4.0 / side))                    # points on the far rim belong to the last cell
        expected = len({(min(int(a / side), n - 1), min(int(b / side), n - 1)) for a, b in uv})
        assert len(np.unique(ids[f])) == expected
    assert sides[0] > 2.0 and 0.8 <= sides[1] < sides[0]            # ~1 view at the fovea, more outside


def test_cell_blur_is_the_rms_angle_about_each_cells_centroid(device):
    """Random landings, random cells, a random Jacobian: compare with loops."""
    rng = np.random.default_rng(0)
    B, F, P, K = 2, 3, 20, 4
    land = rng.normal(0, 0.05, (B, F, P, 2))
    alive = rng.random((B, F, P)) > 0.2
    ids = rng.integers(0, K, (F, P))
    jac = rng.normal(0, 30.0, (B, F, 2, 2)) + 40.0 * np.eye(2)
    onehot = torch.nn.functional.one_hot(torch.tensor(ids), K).double().to(device)
    t = lambda a: torch.tensor(a, device=device)  # noqa: E731
    b2, count = fs.cell_blur2(t(land), t(alive), onehot, t(jac))
    for b in range(B):
        for f in range(F):
            for k in range(K):
                sel = alive[b, f] & (ids[f] == k)
                assert count[b, f, k] == sel.sum()
                if sel.sum() == 0:
                    continue
                dev = land[b, f, sel] - land[b, f, sel].mean(0)
                ang = np.linalg.solve(jac[b, f], dev.T).T
                assert float(b2[b, f, k]) == pytest.approx(np.mean(np.sum(ang**2, 1)), rel=1e-10)


def test_panel_corners_lie_on_the_image_plane(designs, device):
    lay, x, idx = designs
    batch = fs.to_batch(x, idx, lay)
    corners = fs.panel_corners(batch).cpu().numpy()
    a = batch.rx[:, -1].cpu().numpy()
    normal = np.stack([np.zeros_like(a), -np.sin(a), np.cos(a)], -1)
    origin = np.stack([np.zeros_like(a), batch.y[:, -1].cpu().numpy(), batch.z[:, -1].cpu().numpy()], -1)
    assert np.einsum("bkc,bc->bk", corners - origin[:, None], normal) == pytest.approx(0.0, abs=1e-12)
    side = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=-1)
    assert side == pytest.approx(18.432, abs=1e-9)


def test_the_image_surface_is_the_lens_vertex_bowl(designs, device):
    """Traced landings lie on the variable-focal array's vertex surface: local z
    of the hit = (h(x, y) - h(0, 0)) mm, the lenses standing up towards the light."""
    from scipy.interpolate import RegularGridInterpolator
    lay, x, idx = designs
    ctx = fs.context(device)
    batch = fs.to_batch(x, idx, lay)
    _, d_img, alive, diag = ot.trace(batch, ctx["fields"], ctx["pupil"], diagnostics=True)
    assert bool((d_img[..., 2][alive] < 0).all())                        # light arrives along -z
    p = diag["points"][:, -1]                                            # global
    a = batch.rx[:, -1, None, None]
    y = p[..., 1] - batch.y[:, -1, None, None]
    z = p[..., 2] - batch.z[:, -1, None, None]
    local = torch.stack([p[..., 0], y * torch.cos(a) + z * torch.sin(a), -y * torch.sin(a) + z * torch.cos(a)], -1)
    loc = local[alive].cpu().numpy()
    gu, gv, h = fs.vl.bowl_table(lay["flip_v"])
    inside = (np.abs(loc[:, 0]) < gu[-1]) & (np.abs(loc[:, 1]) < gv[-1])
    expected = RegularGridInterpolator((gu, gv), h)(loc[inside, :2])
    assert loc[inside, 2] == pytest.approx(expected, abs=1e-9)
    assert inside.mean() > 0.5


def _grid_map(fn, n=9):
    a = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    X, Y = torch.meshgrid(a, a, indexing="ij")
    return torch.stack(fn(X, Y), -1)[None]                               # (1, n, n, 2)


def test_the_fold_term_is_zero_for_the_target_and_positive_for_folds_and_collapse():
    target = _grid_map(lambda x, y: (x, y))
    ok = torch.ones(target.shape[:3], dtype=torch.bool)
    det_t = fs.cell_det(target)[0]
    assert float(fs.fold_violation(target, ok, det_t, 1.0).abs().max()) == 0.0
    assert float(fs.fold_violation(_grid_map(lambda x, y: (1.5 * x, y)), ok, det_t, 1.0).abs().max()) == 0.0
    folded = fs.fold_violation(_grid_map(lambda x, y: (x.abs(), y)), ok, det_t, 1.0)
    assert float(folded[0, :4].min()) > 0.0 and float(folded[0, 4:].abs().max()) == 0.0   # x < 0 half folds
    collapsed = fs.fold_violation(_grid_map(lambda x, y: (0.1 * x, y)), ok, det_t, 1.0)
    assert float(collapsed.min()) == pytest.approx(fs.Q_MIN - 0.1)
    mirrored = fs.fold_violation(_grid_map(lambda x, y: (-x, y)), ok, det_t, -1.0)   # a flipped image is fine
    assert float(mirrored.abs().max()) == 0.0
    dead = ok.clone()
    dead[0, 0, 0] = False
    assert float(fs.fold_violation(_grid_map(lambda x, y: (x.abs(), y)), dead, det_t, 1.0)[0, 0, 0]) == 0.0


def test_the_outside_term_is_the_depth_inside_the_panel_of_out_of_field_rays():
    h = fs.PANEL_HALF_MM + fs.OUT_MARGIN_MM
    land = torch.tensor([[[0.0, 0.0], [h - 1.0, 0.0], [0.0, -(h + 1.0)], [h + 3.0, 2.0]]], dtype=torch.float64)
    alive = torch.tensor([[True, True, True, False]])
    got = fs.outside_violation(land, alive)
    assert got[0].tolist() == pytest.approx([h, 1.0, 0.0, 0.0])


def test_the_dense_grid_and_the_ring_enter_the_merit(designs, device):
    lay, x, idx = designs
    ctx = fs.context(device)
    _, info = fs.residuals(x, idx, lay, ctx)
    batch = fs.to_batch(x, idx, lay)
    chief = torch.zeros(1, 2, dtype=torch.float64, device=device)
    land, _, alive = ot.trace(batch, ctx["dense_fields"], chief)
    shape = ctx["dense_shape"]
    L = land[:, :, 0].reshape(len(x), *shape, 2)
    ok = alive[:, :, 0].reshape(len(x), *shape)
    sign = lay["flip_u"] * lay["flip_v"] * ctx["dense_sign"]
    fold = fs.fold_violation(L, ok, ctx["dense_det"], sign)
    expected = fs.W["fold"] * (fold**2).sum((1, 2)) / fold[0].numel()
    assert info["fold"].detach().cpu().numpy() == pytest.approx(expected.cpu().numpy(), rel=1e-9, abs=1e-12)
    ring, _, ring_alive = ot.trace(batch, ctx["ring_fields"], chief)
    out = fs.outside_violation(ring[:, :, 0], ring_alive[:, :, 0])
    expected = fs.W["outside"] * (out**2).sum(1) / out.shape[1]
    assert info["outside"].detach().cpu().numpy() == pytest.approx(expected.cpu().numpy(), rel=1e-9, abs=1e-12)


FOLD_BEST = HERE / "results_fold" / "best_fold_el1_glass.json"                # anamorphic target map


@pytest.fixture(scope="module")
def splined(device):
    import json
    entry = json.loads(FOLD_BEST.read_text())[0]
    base = fs.layout(entry["n_el"], entry["flip_u"], entry["flip_v"])
    grid, shape = fs.mirror_spline_grid(entry, device, cells=4)
    return entry, base, fs.with_mirror_spline(base, grid, shape)


def test_a_zero_spline_leaves_a_stored_design_unchanged(splined, device):
    entry, base, lay = splined
    ctx = fs.context(device)
    x_base, idx = fs.stored_seeds([FOLD_BEST], base, entry["material"], device, ctx)
    x, idx2 = fs.stored_seeds([FOLD_BEST], lay, entry["material"], device, ctx)
    assert x.shape[1] == lay["size"] and float(x[:, lay["spline"]["slice"]].abs().max()) == 0.0
    r0, _ = fs.residuals(x_base, idx, base, ctx)
    r1, _ = fs.residuals(x, idx2, lay, ctx)
    assert r1.cpu().numpy() == pytest.approx(r0.cpu().numpy(), abs=1e-9)   # uncompiled Newton: other rounding


def test_the_spline_controls_are_mirror_symmetric_in_x(splined, device):
    entry, _, lay = splined
    x = torch.zeros(1, lay["size"], dtype=torch.float64, device=device)
    x[0, :len(entry["x"])] = torch.tensor(entry["x"], dtype=torch.float64)
    x[0, lay["spline"]["slice"]] = torch.linspace(-0.1, 0.2, lay["spline"]["slice"].stop - lay["spline"]["slice"].start)
    ctrl = fs.to_batch(x, torch.tensor([entry["indices"]], dtype=torch.float64, device=device), lay).spline[2][0]
    assert ctrl.cpu().numpy() == pytest.approx(ctrl.flip(0).cpu().numpy(), abs=0.0)
    assert float(ctrl.abs().max()) > 0.0


def test_the_spline_grid_covers_the_mirror_footprint_with_a_cell_to_spare(splined, device):
    entry, base, lay = splined
    (x0, y0, h), (nx, ny) = lay["spline"]["grid"], lay["spline"]["shape"]
    ctx = fs.context(device)
    x = torch.tensor([entry["x"]], dtype=torch.float64, device=device)
    idx = torch.tensor([entry["indices"]], dtype=torch.float64, device=device)
    batch = fs.to_batch(x, idx, base)
    _, _, alive, diag = ot.trace(batch, ctx["fields"], ctx["pupil"], diagnostics=True)
    p = diag["points"][0, 0][alive[0]].cpu().numpy()
    a = float(batch.rx[0, 0])
    loc_x = p[:, 0]
    loc_y = (p[:, 1] - float(batch.y[0, 0])) * math.cos(a) + (p[:, 2] - float(batch.z[0, 0])) * math.sin(a)
    assert x0 == pytest.approx(-(nx - 1) / 2 * h)
    e = 1e-9                                                                # the first free control sits on the edge
    assert x0 + h <= -np.abs(loc_x).max() + e and x0 + (nx - 2) * h >= np.abs(loc_x).max() - e
    assert y0 + h <= loc_y.min() + e and y0 + (ny - 2) * h >= loc_y.max() - e


@pytest.fixture(scope="module")
def knife_edge(device):
    """A searched spline fold whose lens thins to a knife edge just beyond its footprint."""
    import json
    entry = json.loads((HERE / "results_fold" / "best_fold_el1_glass_spline4.json").read_text())[0]
    lay = fs.entry_layout(entry)
    x = torch.tensor([entry["x"]], dtype=torch.float64, device=device)
    idx = torch.tensor([entry["indices"]], dtype=torch.float64, device=device)
    return lay, x, idx


def _traced(lay, x, idx, device):
    batch = fs.to_batch(x, idx, lay)
    ctx = fs.context(device)
    _, _, alive, diag = ot.trace(batch, ctx["fields"], ctx["pupil"], diagnostics=True)
    return batch, diag, alive


def test_the_rim_thickness_is_the_exporters_along_the_front_axis(knife_edge, device):
    sys.path.insert(0, str(HERE.parents[1] / "scripts"))
    import export_fold as ef
    lay, x, idx = knife_edge
    batch, diag, alive = _traced(lay, x, idx, device)
    rim, t, valid, _ = fs.rim_thickness(batch, diag["points"], alive, 1, 2)
    cb = ef.cpu_batch(batch)
    rim, t, valid = rim[0].detach().cpu().numpy(), t[0].detach().cpu().numpy(), valid[0].cpu().numpy()
    gx, gy = rim[valid, 0], rim[valid, 1]
    f, _, _ = ef._sag(cb, 1, gx, gy)
    bx, by = ef._back_along_front_axis(cb, 1, 2, gx, gy)
    fb, _, _ = ef._sag(cb, 2, bx, by)
    front = ef._global(cb, 1, np.stack([gx, gy, f], -1))
    back = ef._global(cb, 2, np.stack([bx, by, fb], -1))
    assert valid.mean() > 0.9
    assert np.allclose((back - front) @ ef._frame(cb, 1)[1][2], t[valid], atol=1e-9)


def test_the_space_term_sees_a_knife_edge_rim_and_not_a_thick_one(knife_edge, device):
    """The stored design's lens crosses itself just beyond its footprint; the
    same lens made 3 mm thicker keeps MIN_GLASS_MM on its whole rim."""
    lay, x, idx = knife_edge
    thick = x.clone()
    thick[:, lay["elements"][0]["pose"].start + 3] += 3.0
    rims = []
    for design in (x, thick):
        batch, diag, alive = _traced(lay, design, idx, device)
        _, t, valid, sign = fs.rim_thickness(batch, diag["points"], alive, 1, 2)
        rims.append(fs.rim_violation(t, valid, sign))
    assert float(rims[0]) > 0.0 and float(rims[1]) == 0.0
    batch, diag, alive = _traced(lay, x, idx, device)
    space, _ = fs.fold_constraints(batch, diag, alive, lay, x)
    assert float(space) >= float(rims[0])                              # the rim enters the space term


def test_the_polygon_depth_is_the_signed_distance_and_matches_the_mask_table():
    """fold_search.polygon_depth (torch, the search) is the exact signed distance,
    positive inside; variable_lenslets.mask_table (numpy, the exporter's mask) is
    open on the same side, off the edge."""
    sq = torch.tensor([[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]], dtype=torch.float64)
    p = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [3.0, 0.0], [2.0, 2.0], [0.0, -0.9]]], dtype=torch.float64)
    assert fs.polygon_depth(p, sq)[0].tolist() == pytest.approx([1.0, 0.5, -2.0, -math.sqrt(2.0), 0.1])
    rng = np.random.default_rng(3)
    a = np.sort(rng.uniform(0.0, 2 * np.pi, 40))
    r = rng.uniform(4.0, 8.0, 40)
    poly = np.column_stack([r * np.cos(a), r * np.sin(a)])
    gu, gv, is_open = fs.vl.mask_table(poly)
    q = rng.uniform(-9.0, 9.0, (500, 2))
    i, j = np.rint((q[:, 0] - gu[0]) / (gu[1] - gu[0])).astype(int), np.rint((q[:, 1] - gv[0]) / (gv[1] - gv[0])).astype(int)
    got = fs.polygon_depth(torch.tensor(np.column_stack([gu[i], gv[j]])[None]), torch.tensor(poly[None]))[0].numpy()
    off_edge = np.abs(got) > 0.03                                       # 1.5 table steps
    assert np.array_equal(got[off_edge] > 0, is_open[i, j][off_edge])
    assert 0 < (got[off_edge] > 0).sum() < off_edge.sum()


def test_the_polygon_outside_term_is_the_depth_inside_the_opening_plus_the_margin():
    sq = torch.tensor([[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]], dtype=torch.float64)
    land = torch.tensor([[[0.0, 0.0], [1.1, 0.0], [3.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    alive = torch.tensor([[True, True, True, False]])
    got = fs.outside_violation_polygon(land, alive, sq)
    assert got[0].tolist() == pytest.approx([1.0 + fs.OUT_MARGIN_MM, fs.OUT_MARGIN_MM - 0.1, 0.0, 0.0])

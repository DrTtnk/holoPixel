"""The off-axis GPU tracer must land every ray where optiland lands it, through
tilted, decentred freeform mirrors and lenses, and its gradients must match
finite differences."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = (Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study"
        / "remapper_designs" / "freeform_mirror")
sys.path.insert(0, str(HERE))

import offaxis_optiland as oo  # noqa: E402
import offaxis_tracer as ot  # noqa: E402

FIELDS = np.array([(0.0, 0.0), (0.0, 10.0), (0.0, -10.0), (12.0, 0.0), (12.0, 8.0)])
PUPIL = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-0.7, 0.7), (0.5, -0.3)])


def _fold(rng):
    """Eye -> tilted concave freeform mirror 40 mm ahead -> beam goes up and back
    -> a tilted freeform lens -> a tilted image plane. Random around that."""
    a = math.radians(rng.uniform(17.0, 23.0))
    up = np.array([math.sin(2 * a), -math.cos(2 * a)])            # reflected axis in (y, z)
    mirror = (0.0, 40.0)

    def along(dist):
        return mirror[0] + dist * up[0], mirror[1] + dist * up[1]

    def xy(scale):
        return [[0.0, rng.normal(0, scale), rng.normal(0, scale), rng.normal(0, scale**2)],
                [0.0] * 4,
                [rng.normal(0, scale), rng.normal(0, scale**2), 0.0, 0.0]]

    tilt = math.degrees(2 * a)
    y1, z1 = along(18.0 + rng.uniform(-1, 1))
    y2, z2 = along(22.0 + rng.uniform(-1, 1))
    yi, zi = along(34.0 + rng.uniform(-1, 1))
    return {"surfaces": [
        {"y_mm": rng.normal(0, 0.5), "z_mm": 40.0, "rx_deg": math.degrees(a) + rng.normal(0, 0.5),
         "radius_mm": -80.0 * rng.uniform(0.9, 1.1), "conic": rng.normal(0, 0.3), "xy": xy(1e-4),
         "mirror": True, "index_after": 1.0},
        {"y_mm": y1, "z_mm": z1, "rx_deg": tilt + rng.normal(0, 2), "radius_mm": 30.0 * rng.uniform(0.9, 1.1),
         "conic": 0.0, "xy": xy(1e-4), "mirror": False, "index_after": 1.5},
        {"y_mm": y2, "z_mm": z2, "rx_deg": tilt + rng.normal(0, 2), "radius_mm": -40.0 * rng.uniform(0.9, 1.1),
         "conic": 0.0, "xy": xy(1e-4), "mirror": False, "index_after": 1.0},
        {"y_mm": yi, "z_mm": zi, "rx_deg": tilt + rng.normal(0, 3), "radius_mm": math.inf, "conic": 0.0,
         "xy": [[0.0]], "mirror": False, "index_after": 1.0},
    ]}


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _local_on_image(lens, rx):
    """optiland's global hit points on the image surface, in its local frame."""
    s = rx["surfaces"][-1]
    img = len(rx["surfaces"]) + 1
    p = np.stack([np.asarray(lens.surfaces.x[img]), np.asarray(lens.surfaces.y[img]) - s["y_mm"],
                  np.asarray(lens.surfaces.z[img]) - s["z_mm"]], -1)
    a = -math.radians(s["rx_deg"])
    return np.stack([p[:, 0], p[:, 1] * math.cos(a) - p[:, 2] * math.sin(a)], -1)


def test_batched_landings_match_optiland(device):
    rng = np.random.default_rng(1)
    designs = [_fold(rng) for _ in range(4)]
    batch = ot.pack(designs, device)
    land, _, alive = ot.trace(batch, torch.tensor(FIELDS, device=device), torch.tensor(PUPIL, device=device))
    for b, rx in enumerate(designs):
        lens = oo.trace(rx, FIELDS, PUPIL)
        ref = _local_on_image(lens, rx).reshape(len(FIELDS), len(PUPIL), 2)
        ok = alive[b].cpu().numpy()
        assert ok.all()
        assert land[b].cpu().numpy() == pytest.approx(ref, abs=1e-6)


def test_the_mirror_reverses_the_axial_direction(device):
    """After the fold the rays travel back towards the eye (negative global z)."""
    rx = _fold(np.random.default_rng(2))
    _, _, alive, diag = ot.trace(ot.pack([rx], device), torch.tensor(FIELDS, device=device),
                                 torch.tensor(PUPIL, device=device), diagnostics=True)
    pts = diag["points"][0].cpu().numpy()                     # (S, F, P, 3)
    assert alive.all()
    assert np.all(pts[-1, ..., 2] < pts[0, ..., 2])
    assert np.all(pts[-1, ..., 1] > 10.0)


def test_total_internal_reflection_loses_the_ray(device):
    """A glass block whose exit face is tilted past the critical angle traps the ray."""
    rx = {"surfaces": [
        {"y_mm": 0.0, "z_mm": 20.0, "rx_deg": 0.0, "radius_mm": math.inf, "conic": 0.0, "xy": [[0.0]],
         "mirror": False, "index_after": 1.5},
        {"y_mm": 0.0, "z_mm": 25.0, "rx_deg": 50.0, "radius_mm": math.inf, "conic": 0.0, "xy": [[0.0]],
         "mirror": False, "index_after": 1.0},
        {"y_mm": 0.0, "z_mm": 40.0, "rx_deg": 0.0, "radius_mm": math.inf, "conic": 0.0, "xy": [[0.0]],
         "mirror": False, "index_after": 1.0}]}
    chief = torch.tensor([[0.0, 0.0]], dtype=torch.float64, device=device)
    _, _, alive = ot.trace(ot.pack([rx], device), torch.tensor([[0.0, 0.0]], device=device), chief)
    assert not bool(alive[0, 0, 0])
    rx["surfaces"][1]["rx_deg"] = 30.0                       # 30 deg < 41.8 deg: it passes
    _, _, alive = ot.trace(ot.pack([rx], device), torch.tensor([[0.0, 0.0]], device=device), chief)
    assert bool(alive[0, 0, 0])


def test_gradients_match_finite_differences(device):
    rx = _fold(np.random.default_rng(3))
    batch = ot.pack([rx], device)
    fields = torch.tensor([(8.0, 6.0)], dtype=torch.float64, device=device)
    pupil = torch.tensor([(0.4, 0.8)], dtype=torch.float64, device=device)

    def landing(b):
        return ot.trace(b, fields, pupil)[0][0, 0, 0]

    for name in ("c", "rx", "z", "xy"):
        leaf = getattr(batch, name).clone().requires_grad_(True)
        g = torch.autograd.grad(landing(batch._replace(**{name: leaf}))[1], leaf)[0].flatten()
        h = 1e-7
        fd = []
        for i in range(leaf.numel()):
            dp, dm = leaf.detach().clone().flatten(), leaf.detach().clone().flatten()
            dp[i] += h
            dm[i] -= h
            yp = landing(batch._replace(**{name: dp.reshape(leaf.shape)}))[1]
            ym = landing(batch._replace(**{name: dm.reshape(leaf.shape)}))[1]
            fd.append(float((yp - ym) / (2 * h)))
        assert g.cpu().numpy() == pytest.approx(np.array(fd), rel=1e-4, abs=1e-5), name


def test_a_tabulated_image_surface_lands_rays_like_the_analytic_one(device):
    """The variable-focal lenslet array needs a tabulated image surface (its
    vertex bowl). Tabulate a sphere on a 20 um (x, y) grid: bilinear sag differs
    from the sphere by ~curvature * step^2 / 8 = 1.4e-6 mm, so the landings match
    the analytic sphere image surface to 1e-5 mm."""
    rx = _fold(np.random.default_rng(5))
    radius = -35.0
    sphere = {**rx, "surfaces": rx["surfaces"][:-1] + [{**rx["surfaces"][-1], "radius_mm": radius}]}
    flat = ot.pack([rx], device)
    g = torch.linspace(-25.0, 25.0, 2501, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(g, g, indexing="ij")
    c = 1.0 / radius
    r2 = X**2 + Y**2
    table = c * r2 / (1.0 + torch.sqrt(1.0 - c * c * r2))
    tabulated = flat._replace(image_sag=(g, g, table))
    fields, pupil = torch.tensor(FIELDS, device=device), torch.tensor(PUPIL, device=device)
    ref, _, ok_ref = ot.trace(ot.pack([sphere], device), fields, pupil)
    got, _, ok = ot.trace(tabulated, fields, pupil)
    assert torch.equal(ok, ok_ref) and bool(ok.all())
    assert got.cpu().numpy() == pytest.approx(ref.cpu().numpy(), abs=1e-5)
    flat_land, _, _ = ot.trace(flat, fields, pupil)
    assert float((flat_land - got).abs().max()) > 0.1                    # the table is really used


@pytest.mark.parametrize("terms", [((0, 2), (2, 0), (0, 3), (2, 1), (0, 6), (2, 4), (4, 2), (6, 0)),   # even in x
                                   ((1, 0), (0, 1), (3, 2), (0, 0)),                               # odd rows, a gap
                                   ((4, 1),),
                                   ()])
def test_the_polynomial_sums_its_terms_and_its_slopes_are_exact(terms):
    """_poly evaluates only the listed terms, by Horner's rule (in x^2 when every
    power of x is even): it must equal the direct sum, and its slopes autograd's."""
    rng = np.random.default_rng(len(terms))
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    C = t(rng.normal(0.0, 1.0, (3, 1, len(terms))))
    x = t(rng.uniform(-1.5, 1.5, (3, 50))).requires_grad_(True)
    y = t(rng.uniform(-1.5, 1.5, (3, 50))).requires_grad_(True)
    ref = sum((C[..., n] * x**i * y**j for n, (i, j) in enumerate(terms)), torch.zeros_like(x))
    val, ddx, ddy = ot._poly(x, y, C, terms)
    assert val.detach().numpy() == pytest.approx(ref.detach().numpy(), abs=1e-12)
    if terms:
        gx, gy = torch.autograd.grad(ref.sum(), (x, y))
        assert ddx.detach().numpy() == pytest.approx(gx.numpy(), abs=1e-12)
        assert ddy.detach().numpy() == pytest.approx(gy.numpy(), abs=1e-12)
    else:
        assert not ddx.detach().numpy().any() and not ddy.detach().numpy().any()


def test_the_newton_early_exit_lands_the_rays_where_every_step_does(device, monkeypatch):
    """The free Newton steps stop once no ray moves more than NEWTON_TOL_MM; with a
    zero tolerance all of them run. Same landings, same lost rays (a TIR ray too)."""
    designs = [_fold(np.random.default_rng(s)) for s in range(3)]
    designs[1]["surfaces"][1]["rx_deg"] += 60.0                                # loses rays
    batch = ot.pack(designs, device)
    f, p = torch.tensor(FIELDS, device=device), torch.tensor(PUPIL, device=device)
    early, _, ok_early = ot.trace(batch, f, p)
    monkeypatch.setattr(ot, "NEWTON_TOL_MM", 0.0)
    full, _, ok_full = ot.trace(batch, f, p)
    assert torch.equal(ok_early, ok_full) and not bool(ok_full.all()) and bool(ok_full.any())
    assert early[ok_full].cpu().numpy() == pytest.approx(full[ok_full].cpu().numpy(), abs=1e-12)


def test_pack_keeps_the_nonzero_terms_of_the_prescriptions(device):
    rx = _fold(np.random.default_rng(1))
    batch = ot.pack([rx, _fold(np.random.default_rng(2))], device)
    assert batch.terms == ((0, 1), (0, 2), (0, 3), (2, 0), (2, 1))
    assert tuple(batch.xy.shape) == (2, len(rx["surfaces"]), len(batch.terms))
    assert float(batch.xy[0, 0, 3]) == rx["surfaces"][0]["xy"][2][0]


def _centred_cubic(t):
    a = np.abs(t)
    return np.where(a < 1, 2.0 / 3.0 - a**2 + a**3 / 2.0, np.where(a < 2, (2.0 - a) ** 3 / 6.0, 0.0))


def test_the_bspline_sag_is_the_sum_of_centred_cubics_and_its_slopes_are_exact():
    rng = np.random.default_rng(3)
    grid = (-7.5, -4.0, 2.5)
    ctrl = rng.normal(0.0, 0.05, (2, 7, 5))
    x = rng.uniform(-14.0, 14.0, (2, 400))
    y = rng.uniform(-10.0, 12.0, (2, 400))                                # also outside the control region
    ref = np.zeros_like(x)
    for j in range(7):
        for k in range(5):
            ref += ctrl[:, j, k, None] * _centred_cubic((x - grid[0]) / grid[2] - j) \
                * _centred_cubic((y - grid[1]) / grid[2] - k)
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    v, vx, vy = ot.bspline_sag(t(x), t(y), grid, t(ctrl))
    assert v.numpy() == pytest.approx(ref, abs=1e-13)
    e = 1e-6
    fx = (ot.bspline_sag(t(x + e), t(y), grid, t(ctrl))[0] - ot.bspline_sag(t(x - e), t(y), grid, t(ctrl))[0]) / (2 * e)
    fy = (ot.bspline_sag(t(x), t(y + e), grid, t(ctrl))[0] - ot.bspline_sag(t(x), t(y - e), grid, t(ctrl))[0]) / (2 * e)
    assert vx.numpy() == pytest.approx(fx.numpy(), abs=1e-8) and vy.numpy() == pytest.approx(fy.numpy(), abs=1e-8)


def _spline_twin(rng, device):
    """A fold whose mirror carries alpha x^2 + beta y^2 as polynomial terms, and its
    twin carrying the same quadratic as a B-spline (exact reproduction inside the
    fully supported region)."""
    rx = _fold(rng)
    alpha, beta = 2e-3, -1.5e-3
    rx["surfaces"][0]["xy"] = [[0.0, 0.0, beta], [0.0], [alpha]]
    poly = ot.pack([rx], device)
    h, n = 3.0, 14
    x0 = y0 = -(n - 1) / 2 * h
    xj = x0 + h * np.arange(n)
    ctrl = alpha * xj[:, None] ** 2 + beta * xj[None, :] ** 2 - (alpha + beta) * h**2 / 3.0
    rx["surfaces"][0]["xy"] = [[0.0]]
    base = ot.pack([rx], device)
    twin = base._replace(spline=((0,), (x0, y0, h), torch.tensor(ctrl[None], dtype=torch.float64, device=device)))
    return poly, base, twin


def test_a_spline_mirror_lands_rays_like_the_polynomial_it_reproduces(device):
    poly, _, twin = _spline_twin(np.random.default_rng(5), device)
    f, p = torch.tensor(FIELDS, device=device), torch.tensor(PUPIL, device=device)
    a, _, ok_a = ot.trace(poly, f, p)
    b, _, ok_b = ot.trace(twin, f, p)
    assert bool(ok_a.all()) and bool(ok_b.all())
    assert b.cpu().numpy() == pytest.approx(a.cpu().numpy(), abs=1e-9)


def test_zero_spline_controls_change_nothing(device):
    _, base, twin = _spline_twin(np.random.default_rng(6), device)
    twin = twin._replace(spline=((0,), twin.spline[1], torch.zeros_like(twin.spline[2])))
    f, p = torch.tensor(FIELDS, device=device), torch.tensor(PUPIL, device=device)
    assert ot.trace(twin, f, p)[0].cpu().numpy() == pytest.approx(ot.trace(base, f, p)[0].cpu().numpy(), abs=1e-12)

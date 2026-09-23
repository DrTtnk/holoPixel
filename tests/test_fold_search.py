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
    lay = fs.layout(request.param)
    rng = np.random.default_rng(request.param)
    lo, hi = fs.bounds(lay, device)
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
    """The variable-focal lenslets make the pupil fill one lens pitch (36 um =
    5 pixels) everywhere, so a pixel's footprint on the pupil is 4/5 mm at every
    field inside 35 deg: 5 x 5 squares, of which those that hold hexagonal pupil
    points survive."""
    pupil = fs.pupil_samples()
    fields = np.array([[0.0, 0.0], [10.0, 0.0], [35.0, 0.0]])
    ids, _ = fs.pupil_cells(fields, pupil)
    uv = pupil * 2.0 + 2.0
    for f, (fx, fy) in enumerate(fields):
        ecc = np.arctan(np.hypot(np.tan(np.radians(fx)), np.tan(np.radians(fy))))
        side = 7.2 * float(fs.ft.local_focal_mm(ecc)) / float(
            fs.ft.lenslet_focal_of_radius_um(fs.ft.panel_radius_mm(ecc) * 1e3))
        assert side == pytest.approx(4.0 / 5.0, rel=1e-6)
        n = max(1, math.ceil(4.0 / side))                    # points on the far rim belong to the last cell
        expected = len({(min(int(u / side), n - 1), min(int(v / side), n - 1)) for u, v in uv})
        assert len(np.unique(ids[f])) == expected


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
    """Traced landings lie on the variable-focal array's vertex profile: local z
    of the hit = (h(r) - h(0)) mm, the lenses standing up towards the light."""
    lay, x, idx = designs
    _, d_img, alive, diag = ot.trace(fs.to_batch(x, idx, lay), fs.context(device)["fields"],
                                     fs.context(device)["pupil"], diagnostics=True)
    assert bool((d_img[..., 2][alive] < 0).all())                        # light arrives along -z
    batch = fs.to_batch(x, idx, lay)
    p = diag["points"][:, -1]                                            # global
    a = batch.rx[:, -1, None, None]
    y = p[..., 1] - batch.y[:, -1, None, None]
    z = p[..., 2] - batch.z[:, -1, None, None]
    local = torch.stack([p[..., 0], y * torch.cos(a) + z * torch.sin(a), -y * torch.sin(a) + z * torch.cos(a)], -1)
    r = torch.linalg.norm(local[..., :2], dim=-1)[alive].cpu().numpy()
    expected = np.interp(r**2, fs.BOWL_R_MM**2, fs.BOWL_SAG_MM)
    assert local[..., 2][alive].cpu().numpy() == pytest.approx(expected, abs=1e-9)
    assert fs.BOWL_SAG_MM[0] == 0.0 and fs.BOWL_SAG_MM[-1] < -1.0          # ~1.5 mm deep

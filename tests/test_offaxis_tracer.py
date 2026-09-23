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

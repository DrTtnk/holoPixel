"""The batched GPU tracer must land every ray where optiland (the verified
reference, tests/test_remapper_optiland.py) lands it, for many designs at once."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

STUDY = Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study"
sys.path.insert(0, str(STUDY / "scripts"))
sys.path.insert(0, str(STUDY / "remapper_designs" / "coaxial_dls"))

import coaxial_optiland as co  # noqa: E402
import fast_merit as fm  # noqa: E402
import gpu_tracer as gt  # noqa: E402
import merit  # noqa: E402

FIELDS = [0.0, 7.0, 13.0, 19.0]   # fields where these seeds' glass is physical everywhere


def _designs(n, seed=0):
    """Mild random coaxial designs around a sane one (every ray survives)."""
    rng = np.random.default_rng(seed)
    base = json.loads((STUDY / "remapper_designs" / "coaxial_dls" / "stage1_shape.json").read_text())
    out = []
    for _ in range(n):
        rx = json.loads(json.dumps(base))
        for el in rx["elements"]:
            for side in ("front", "back"):
                s = el[side]
                s["radius_mm"] *= rng.uniform(0.95, 1.05)
                s["conic"] += rng.normal(0, 0.1)
                s["coefficients"] = [0.0, rng.normal(0, 1e-6), rng.normal(0, 1e-9), rng.normal(0, 1e-12)]
            el["thickness_mm"] *= rng.uniform(0.95, 1.05)
        out.append(rx)
    return out


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_batched_landings_match_optiland(device):
    designs = _designs(5)
    batch = gt.pack(designs, device=device)
    pupil = fm.pupil_samples()
    land, direction, alive = gt.trace(batch, torch.tensor(FIELDS, dtype=torch.float64, device=device),
                                      torch.tensor(pupil, dtype=torch.float64, device=device))
    for b, rx in enumerate(designs):
        lens, _ = fm.trace(rx, FIELDS)
        img = merit.image_surface(lens)
        X = np.asarray(lens.surfaces.x[img]).reshape(len(FIELDS), len(pupil))
        Y = np.asarray(lens.surfaces.y[img]).reshape(len(FIELDS), len(pupil))
        N = np.asarray(lens.surfaces.N[img]).reshape(len(FIELDS), len(pupil))
        ok = alive[b].cpu().numpy()
        assert ok.all()
        assert land[b, ..., 0].cpu().numpy()[ok] == pytest.approx(X[ok], abs=1e-6)
        assert land[b, ..., 1].cpu().numpy()[ok] == pytest.approx(Y[ok], abs=1e-6)
        # optiland's Newton tolerance (1e-10 mm) shows up at ~3e-9 in the normal
        assert direction[b, ..., 2].cpu().numpy()[ok] == pytest.approx(N[ok], abs=1e-7)


def test_a_ray_through_crossed_surfaces_is_lost(device):
    """At 33 deg these seeds' rays reach element 2's back surface BEHIND its
    front surface (the two cross at that radius: negative glass). Optiland
    steps backwards and carries on; the tracer must declare the ray lost."""
    rx = _designs(5)[1]
    batch = gt.pack([rx], device=device)
    fields = torch.tensor([33.0], dtype=torch.float64, device=device)
    chief = torch.tensor([[0.0, 0.0]], dtype=torch.float64, device=device)
    _, _, alive = gt.trace(batch, fields, chief)
    assert not bool(alive[0, 0, 0])
    lens, _ = fm.trace(rx, [33.0])
    assert float(np.asarray(lens.surfaces.intensity[merit.image_surface(lens)])[0]) > 0


def test_gradients_match_finite_differences(device):
    """Autograd through the tracer equals a central difference of the traced
    landing height with respect to one curvature."""
    rx = _designs(1, seed=3)[0]
    batch = gt.pack([rx], device=device)
    fields = torch.tensor([19.0], dtype=torch.float64, device=device)
    pupil = torch.tensor([[0.0, 1.0]], dtype=torch.float64, device=device)
    c = batch.c.clone().requires_grad_(True)
    land, _, _ = gt.trace(batch._replace(c=c), fields, pupil)
    land[0, 0, 0, 1].backward()
    h = 1e-7
    grad_fd = []
    for s in range(batch.c.shape[1]):
        dp, dm = batch.c.clone(), batch.c.clone()
        dp[0, s] += h
        dm[0, s] -= h
        yp = gt.trace(batch._replace(c=dp), fields, pupil)[0][0, 0, 0, 1]
        ym = gt.trace(batch._replace(c=dm), fields, pupil)[0][0, 0, 0, 1]
        grad_fd.append(float((yp - ym) / (2 * h)))
    assert c.grad[0].cpu().numpy() == pytest.approx(np.array(grad_fd), rel=1e-4, abs=1e-6)

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surfaces import DTYPE, Freeform, reflect, rot_x_frame  # noqa: E402

torch.manual_seed(0)


def test_reflection_law_hand_case():
    """A ray along +x hitting a mirror whose normal is +z (a plane y-x) should
    NOT flip (grazing); a ray along -z hitting a flat mirror normal to +z
    (normal incidence) must reverse exactly."""
    d = torch.tensor([[0.0, 0.0, -1.0]], dtype=DTYPE)
    n = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    out = reflect(d, n)
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE))

    # 45 degree incidence in the x-z plane off a z-normal mirror: d=(1,0,-1)/sqrt2
    d = torch.tensor([[1.0, 0.0, -1.0]], dtype=DTYPE) / math.sqrt(2.0)
    out = reflect(d, n)
    expect = torch.tensor([[1.0, 0.0, 1.0]], dtype=DTYPE) / math.sqrt(2.0)
    assert torch.allclose(out, expect, atol=1e-12)

    # normal sign must not matter
    out2 = reflect(d, -n)
    assert torch.allclose(out, out2, atol=1e-12)


def test_normal_sign_invariance_random():
    g = torch.Generator().manual_seed(1)
    d = torch.nn.functional.normalize(torch.randn(50, 3, dtype=DTYPE, generator=g), dim=1)
    n = torch.nn.functional.normalize(torch.randn(50, 3, dtype=DTYPE, generator=g), dim=1)
    assert torch.allclose(reflect(d, n), reflect(d, -n), atol=1e-12)


def _flat_mirror(tilt=0.0, V=(0.0, 20.0, 0.0)):
    terms = [(0, 2), (2, 0)]
    coeffs = torch.zeros(len(terms), dtype=DTYPE)
    return Freeform(torch.tensor(V, dtype=DTYPE), torch.tensor(tilt, dtype=DTYPE),
                    torch.tensor(0.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE), terms, coeffs)


def test_paraxial_spherical_focal_length():
    """A concave spherical mirror (c = 1/R) reflects a bundle of rays parallel
    to its axis back through the focal point at R/2, to paraxial accuracy."""
    R = 100.0
    terms = []
    coeffs = torch.zeros(0, dtype=DTYPE)
    mirror = Freeform(torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE),
                      torch.tensor(1.0 / R, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE), terms, coeffs)
    y = torch.tensor([0.0, 0.01, 0.02, -0.01, -0.02], dtype=DTYPE)  # mm, paraxial
    x = torch.zeros_like(y)
    O = torch.stack([x, y, torch.full_like(y, -50.0)], dim=1)
    D = torch.zeros_like(O)
    D[:, 2] = 1.0
    t, P, hit = mirror.intersect(O, D)
    assert hit.all()
    n = mirror.normal_world(*mirror.local_coords(P)[:2])
    Dout = reflect(D, n)
    # propagate to z = R/2 (focal plane) and check all rays land near y=0
    focal_z = R / 2.0
    tf = (focal_z - P[:, 2]) / Dout[:, 2]
    y_focal = P[:, 1] + tf * Dout[:, 1]
    assert torch.max(torch.abs(y_focal)).item() < 1e-3  # sub-micron at paraxial heights


def test_intersection_matches_independent_bisection():
    """Newton's intersection must agree with a plain bisection search on the
    same implicit surface equation, for a curved tilted freeform mirror."""
    terms = [(0, 2), (2, 0), (0, 3), (2, 1), (0, 4)]
    g = torch.Generator().manual_seed(2)
    coeffs = 1e-4 * torch.randn(len(terms), dtype=DTYPE, generator=g)
    mirror = Freeform(torch.tensor([0.0, 20.0, -5.0], dtype=DTYPE), torch.tensor(0.3, dtype=DTYPE),
                      torch.tensor(1.0 / 80.0, dtype=DTYPE), torch.tensor(-0.5, dtype=DTYPE), terms, coeffs)

    O = torch.tensor([[0.0, -3.6, 0.0]], dtype=DTYPE).repeat(20, 1)
    tx = torch.linspace(-0.3, 0.3, 20, dtype=DTYPE)
    D = torch.nn.functional.normalize(torch.stack(
        [torch.tan(tx), torch.ones(20, dtype=DTYPE), torch.tan(tx * 0.6)], dim=1), dim=1)

    t, P, hit = mirror.intersect(O, D)
    assert hit.all()

    def f(t_scalar, i):
        Pt = O[i] + t_scalar * D[i]
        x, y, z = mirror.local_coords(Pt[None, :])
        x, y, z = x[0], y[0], z[0]
        return (z - mirror.sag(x[None], y[None])[0]).item()

    # The surface is a general polynomial and can have more than one root
    # along a ray; bisection is bracketed around Newton's own root (not
    # picked to match it) to test that a genuine sign change -- an actual
    # root of the implicit equation -- sits there, independently of Newton.
    for i in range(20):
        t0 = t.detach()[i].item()
        lo, hi = t0 - 3.0, t0 + 3.0
        flo, fhi = f(torch.tensor(lo, dtype=DTYPE), i), f(torch.tensor(hi, dtype=DTYPE), i)
        assert flo * fhi < 0, f"ray {i}: bracket does not change sign ({flo}, {fhi})"
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fm = f(torch.tensor(mid, dtype=DTYPE), i)
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        t_bisect = 0.5 * (lo + hi)
        assert abs(t.detach()[i].item() - t_bisect) < 1e-6, (i, t.detach()[i].item(), t_bisect)


def test_freeform_normal_against_autograd():
    """The analytic sag gradient (used for the normal) must match torch's own
    autograd of `sag`, on random points and random polynomial coefficients."""
    terms = [(0, 1), (0, 2), (2, 0), (0, 3), (2, 1), (4, 0), (0, 4), (2, 2)]
    g = torch.Generator().manual_seed(3)
    coeffs = 1e-3 * torch.randn(len(terms), dtype=DTYPE, generator=g)
    mirror = Freeform(torch.zeros(3, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE),
                      torch.tensor(1.0 / 120.0, dtype=DTYPE), torch.tensor(0.2, dtype=DTYPE), terms, coeffs)

    x = torch.tensor(1.3, dtype=DTYPE, requires_grad=True)
    y = torch.tensor(-2.1, dtype=DTYPE, requires_grad=True)
    s = mirror.sag(x[None], y[None])[0]
    gx, gy = torch.autograd.grad(s, (x, y))

    dx, dy = mirror.sag_grad(x.detach()[None], y.detach()[None])
    assert torch.allclose(dx[0], gx, atol=1e-10)
    assert torch.allclose(dy[0], gy, atol=1e-10)


def test_rot_x_frame_orthonormal_right_handed():
    for a in [-1.2, 0.0, 0.3, 1.5]:
        u, v, w = rot_x_frame(torch.tensor(a, dtype=DTYPE))
        R = torch.stack([u, v, w], dim=1)
        assert torch.allclose(R.T @ R, torch.eye(3, dtype=DTYPE), atol=1e-12)
        assert torch.allclose(torch.linalg.cross(u, v), w, atol=1e-12)
        assert torch.linalg.det(R).item() == pytest.approx(1.0, abs=1e-12)


def test_x_even_polynomial_gives_plane_symmetric_surface():
    """Sag with only even-x terms must satisfy s(-x, y) == s(x, y)."""
    terms = [(0, 1), (2, 0), (0, 2), (2, 1), (4, 0)]
    g = torch.Generator().manual_seed(4)
    coeffs = 1e-3 * torch.randn(len(terms), dtype=DTYPE, generator=g)
    mirror = Freeform(torch.zeros(3, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE),
                      torch.tensor(1.0 / 90.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE), terms, coeffs)
    x = torch.linspace(-5, 5, 11, dtype=DTYPE)
    y = torch.linspace(-3, 7, 11, dtype=DTYPE)
    assert torch.allclose(mirror.sag(x, y), mirror.sag(-x, y), atol=1e-12)

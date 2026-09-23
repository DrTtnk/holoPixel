"""Proofs for the coaxial ray tracer (remapper_designs/coaxial/src/optics.py):
- asphere derivative against sympy (also checked once at design time)
- Newton intersection against independent bisection
- surface normal against autograd of the same sag function
- vector Snell's law against the scalar law (angle form)
- a paraxial thin lens against its ABCD prediction
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import optics as op  # noqa: E402

torch.set_default_dtype(op.DTYPE)


def test_asphere_derivative_matches_sympy():
    r, c, k, a4, a6, a8 = sp.symbols("r c k a4 a6 a8", real=True)
    sag = c * r**2 / (1 + sp.sqrt(1 - (1 + k) * c**2 * r**2)) + a4 * r**4 + a6 * r**6 + a8 * r**8
    d = sp.diff(sag, r)
    proposed = c * r / sp.sqrt(1 - (1 + k) * c**2 * r**2) + 4 * a4 * r**3 + 6 * a6 * r**5 + 8 * a8 * r**7
    assert sp.simplify(d - proposed) == 0

    rng = np.random.default_rng(0)
    for _ in range(20):
        vals = {r: rng.uniform(0.1, 3.0), c: rng.uniform(-0.1, 0.1), k: rng.uniform(-3, 3),
                a4: rng.uniform(-1e-4, 1e-4), a6: rng.uniform(-1e-6, 1e-6), a8: rng.uniform(-1e-8, 1e-8)}
        lhs = float(d.subs(vals))
        rhs = float(proposed.subs(vals))
        assert lhs == pytest.approx(rhs, abs=1e-9)


def _surf(vertex_y=10.0, c=0.05, k=-0.3, a4=1e-5, a6=0.0, a8=0.0, aperture=8.0):
    t = lambda x: torch.tensor(float(x), dtype=op.DTYPE)  # noqa: E731
    return op.Surface(t(vertex_y), t(c), t(k), t(a4), t(a6), t(a8), aperture)


def test_intersection_matches_independent_bisection():
    rng = np.random.default_rng(1)
    s = _surf()
    for _ in range(30):
        o = torch.tensor([rng.uniform(-1, 1), -3.6, rng.uniform(-1, 1)], dtype=op.DTYPE)
        theta = rng.uniform(-0.3, 0.3)
        phi = rng.uniform(-0.3, 0.3)
        d = torch.tensor([math.sin(phi), math.cos(theta) * math.cos(phi), math.sin(theta)], dtype=op.DTYPE)
        d = d / torch.linalg.norm(d)
        t = op.intersect(o, d, s)

        def f(tt):
            p = o.numpy() + tt * d.numpy()
            r = math.hypot(p[0], p[2])
            sg = s.c.item() * r**2 / (1 + math.sqrt(max(1e-9, 1 - (1 + s.k.item()) * s.c.item()**2 * r**2)))
            sg += s.a4.item() * r**4
            return p[1] - s.vertex_y.item() - sg

        lo, hi = 1e-6, 20.0
        flo, fhi = f(lo), f(hi)
        assert flo * fhi < 0, "bracket must change sign"
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        t_bisect = 0.5 * (lo + hi)
        assert t.item() == pytest.approx(t_bisect, abs=1e-6)


def test_normal_matches_autograd_of_the_implicit_function():
    s = _surf()
    rng = np.random.default_rng(2)
    for _ in range(20):
        x = torch.tensor(rng.uniform(-3, 3), dtype=op.DTYPE, requires_grad=True)
        z = torch.tensor(rng.uniform(-3, 3), dtype=op.DTYPE, requires_grad=True)
        r = torch.sqrt(x**2 + z**2)
        y = s.vertex_y + op.sag(r, s)
        gy = torch.tensor(1.0, dtype=op.DTYPE)
        gx, gz = torch.autograd.grad(y, [x, z])
        n_auto = torch.stack([-gx, gy, -gz])
        n_auto = n_auto / torch.linalg.norm(n_auto)

        o = torch.tensor([0.0, -3.6, 0.0], dtype=op.DTYPE)
        target = torch.tensor([x.item(), y.item(), z.item()], dtype=op.DTYPE)
        d = target - o
        d = d / torch.linalg.norm(d)
        _, p, n = op.surface_point_normal(o, d, s)
        assert torch.allclose(p, target, atol=1e-6)
        assert torch.allclose(n.abs(), n_auto.abs(), atol=1e-6)


def test_vector_snell_matches_scalar_snell():
    rng = np.random.default_rng(3)
    for _ in range(30):
        n1, n2 = rng.uniform(1.0, 1.9), rng.uniform(1.0, 1.9)
        theta_i = rng.uniform(0.0, 1.0)
        normal = torch.tensor([0.0, 1.0, 0.0], dtype=op.DTYPE)
        d = torch.tensor([math.sin(theta_i), -math.cos(theta_i), 0.0], dtype=op.DTYPE)
        out, tir, _ = op.refract(d, normal, n1, n2)
        sin_t = n1 / n2 * math.sin(theta_i)
        if sin_t > 1.0:
            assert bool(tir)
        else:
            theta_t = math.asin(sin_t)
            assert not bool(tir)
            measured = math.atan2(out[0].item(), -out[1].item())
            assert measured == pytest.approx(theta_t, abs=1e-8)


def test_snell_total_internal_reflection_matches_law_of_reflection():
    normal = torch.tensor([0.0, 1.0, 0.0], dtype=op.DTYPE)
    theta_i = math.radians(60.0)  # > critical angle for n1=1.6 -> n2=1.0
    d = torch.tensor([math.sin(theta_i), -math.cos(theta_i), 0.0], dtype=op.DTYPE)
    out, tir, _ = op.refract(d, normal, 1.6, 1.0)
    assert bool(tir)
    assert out[0].item() == pytest.approx(d[0].item(), abs=1e-9)
    assert out[1].item() == pytest.approx(-d[1].item(), abs=1e-9)


def test_thin_lens_matches_paraxial_abcd():
    """A weak biconvex singlet, paraxial rays only: image distance from a thin
    lens of power (n-1)(1/R1 - 1/R2) at object distance u obeys 1/v = 1/u + 1/f
    (ABCD system matrix), independent of the ray-tracer's own derivation. The
    lens is built vanishingly thin so the thin-lens (single-plane) formula
    applies without a thickness correction."""
    n, R1, R2, u = 1.5, 100.0, -100.0, 40.0
    f = 1.0 / ((n - 1.0) * (1.0 / R1 - 1.0 / R2))
    v_abcd = 1.0 / (1.0 / f - 1.0 / u)

    t = lambda x: torch.tensor(float(x), dtype=op.DTYPE)  # noqa: E731
    thickness = 1e-3
    front = op.Surface(t(u), t(1.0 / R1), t(0.0), t(0.0), t(0.0), t(0.0), 5.0)
    back = op.Surface(t(u + thickness), t(1.0 / R2), t(0.0), t(0.0), t(0.0), t(0.0), 5.0)
    el = op.Element(front, back, n)

    alpha = 1e-4  # paraxial: tiny ray angle from an on-axis object at y = 0
    o = torch.tensor([0.0, 0.0, 0.0], dtype=op.DTYPE)
    d = torch.tensor([0.0, math.cos(alpha), math.sin(alpha)], dtype=op.DTYPE)
    hit_far, dir_far, alive, _, _, _, _ = op.trace_system(o, d, [el], image_y=u + 1000.0)
    assert bool(alive)
    # Back-propagate the outgoing ray to where it crosses the axis (z = 0).
    t_cross = -hit_far[2].item() / dir_far[2].item()
    y_cross = hit_far[1].item() + t_cross * dir_far[1].item()
    v_measured = y_cross - u
    assert v_measured == pytest.approx(v_abcd, rel=2e-3)

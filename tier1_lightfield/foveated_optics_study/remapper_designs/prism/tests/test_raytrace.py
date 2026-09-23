"""Unit proofs for raytrace.py: intersection, Snell's law, TIR, normals, and
a known flat right-angle-prism fold. Independent of the optimizer / export.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import raytrace as rt  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
DTYPE = rt.DTYPE


def random_surface(n=1):
    return rt.Surface(
        origin=torch.zeros(3, dtype=DTYPE),
        theta=torch.tensor(0.3, dtype=DTYPE),
        coeff=torch.tensor(np.random.uniform(-2e-3, 2e-3, rt.N_COEFF), dtype=DTYPE),
    )


def test_intersection_matches_bisection():
    """Newton hit vs an independent 1-D bisection on the ray parameter t.

    Coefficients are kept small enough that the surface is a graph over a
    wide local aperture (no folds), so the residual is monotonic near the
    true root and a bracket found by geometric expansion around the Newton
    root is fair (not seeded from the Newton answer itself)."""
    surf = random_surface()
    surf.coeff = surf.coeff * 1e-2  # keep u^6 terms from blowing up off-aperture
    origin = torch.tensor(np.random.uniform(-1, 1, (20, 3)) + [0, -30, 0], dtype=DTYPE)
    direction = torch.zeros(20, 3, dtype=DTYPE)
    direction[:, 1] = 1.0
    direction += torch.tensor(np.random.uniform(-0.05, 0.05, (20, 3)), dtype=DTYPE)
    direction = direction / direction.norm(dim=-1, keepdim=True)

    t_newton, p_newton, _ = rt.intersect(surf, origin, direction)

    def residual(t, k):
        p = origin[k] + t * direction[k]
        x_l, u_l, n_l = surf.local(p[None, :])
        s, _, _ = surf.sag_and_grad(x_l, u_l)
        return float((n_l - s).item())

    for k in range(20):
        # A degree-6 sag polynomial can have several real roots along a ray,
        # so search for the bracket nearest the ray/tangent-plane crossing
        # (the same, easily-derived first-order guess Newton starts from) by
        # dense sampling, then bisect it -- independent of the Newton path.
        rel0 = (origin[k] - surf.origin)
        n0 = float((rel0 * surf.frame()[2]).sum())
        dn = float((direction[k] * surf.frame()[2]).sum())
        t_guess = -n0 / dn
        ts = t_guess + np.linspace(-6.0, 6.0, 4001)
        vals = np.array([residual(torch.tensor(t, dtype=DTYPE), k) for t in ts])
        sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
        nearest = sign_changes[np.argmin(np.abs(ts[sign_changes] - t_guess))]
        lo, hi = ts[nearest], ts[nearest + 1]
        flo = residual(torch.tensor(lo, dtype=DTYPE), k)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = residual(torch.tensor(mid, dtype=DTYPE), k)
            if fmid * flo <= 0:
                hi = mid
            else:
                lo, flo = mid, fmid
        t_bisect = 0.5 * (lo + hi)
        assert abs(t_bisect - float(t_newton[k])) < 1e-6


def test_surface_normal_matches_autograd():
    """Analytic normal (hand-coded sag gradient) vs torch.autograd on the same sag."""
    surf = random_surface()
    x_l = torch.tensor(np.random.uniform(-0.5, 0.5, 15), dtype=DTYPE, requires_grad=True)
    u_l = torch.tensor(np.random.uniform(-0.5, 0.5, 15), dtype=DTYPE, requires_grad=True)
    s, dsdx, dsdu = surf.sag_and_grad(x_l, u_l)
    g_x, g_u = torch.autograd.grad(s.sum(), [x_l, u_l])
    assert torch.allclose(dsdx, g_x, atol=1e-10)
    assert torch.allclose(dsdu, g_u, atol=1e-10)

    e_x, e_u, e_n = surf.frame()
    normal = surf.normal(x_l.detach(), u_l.detach())
    expected = -g_x[:, None] * e_x - g_u[:, None] * e_u + e_n
    expected = expected / expected.norm(dim=-1, keepdim=True)
    assert torch.allclose(normal, expected, atol=1e-10)


def test_vector_snell_matches_scalar_law():
    """Vector refraction vs the scalar sin law n1 sin(theta_i) = n2 sin(theta_t)."""
    n1, n2 = 1.0, 1.585
    normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE).repeat(50, 1)
    theta_i = torch.tensor(np.random.uniform(0.0, 60.0, 50), dtype=DTYPE)
    az = torch.tensor(np.random.uniform(0.0, 2 * math.pi, 50), dtype=DTYPE)
    ti = torch.deg2rad(theta_i)
    direction = torch.stack([
        torch.sin(ti) * torch.cos(az), torch.sin(ti) * torch.sin(az), -torch.cos(ti),
    ], dim=-1)
    out, tir = rt.refract_or_reflect(direction, normal, n1, n2)
    assert not tir.any()
    theta_t = torch.rad2deg(torch.acos((-out[:, 2]).clamp(-1, 1)))
    lhs = n1 * torch.sin(ti)
    rhs = n2 * torch.sin(torch.deg2rad(theta_t))
    assert torch.allclose(lhs, rhs, atol=1e-9)
    # refracted ray stays in the plane of incidence (compare azimuths on the
    # circle, not linearly: remainder(x, 2pi) can land at ~2pi by rounding)
    out_az = torch.atan2(out[:, 1], out[:, 0])
    wrapped = torch.remainder(out_az - az + math.pi, 2 * math.pi) - math.pi
    assert torch.allclose(wrapped, torch.zeros_like(wrapped), atol=1e-8)


def test_tir_threshold_matches_asin_1_over_n():
    n1, n2 = 1.585, 1.0
    critical = math.degrees(math.asin(n2 / n1))
    normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE).repeat(2000, 1)
    theta_i = torch.tensor(np.linspace(0.01, 89.9, 2000), dtype=DTYPE)
    ti = torch.deg2rad(theta_i)
    direction = torch.stack([torch.sin(ti), torch.zeros_like(ti), -torch.cos(ti)], dim=-1)
    _, tir = rt.refract_or_reflect(direction, normal, n1, n2)
    below = theta_i < critical - 1e-6
    above = theta_i > critical + 1e-6
    assert not tir[below].any()
    assert tir[above].all()


def test_reflection_law_equal_angles():
    normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE).repeat(30, 1)
    theta_i = torch.tensor(np.random.uniform(1.0, 89.0, 30), dtype=DTYPE)
    ti = torch.deg2rad(theta_i)
    direction = torch.stack([torch.sin(ti), torch.zeros_like(ti), -torch.cos(ti)], dim=-1)
    out = rt.mirror_reflect(direction, normal)
    theta_r = torch.rad2deg(torch.acos(out[:, 2].clamp(-1, 1)))
    assert torch.allclose(theta_r, theta_i, atol=1e-9)


def test_flat_right_angle_prism_known_fold():
    """A 45 deg hypotenuse face TIRs a normal-incidence beam by exactly 90 deg,
    the textbook right-angle-prism periscope fold. Flat surfaces (coeff = 0)."""
    flat = lambda origin, theta: rt.Surface(
        origin=torch.tensor(origin, dtype=DTYPE), theta=torch.tensor(theta, dtype=DTYPE),
        coeff=torch.zeros(rt.N_COEFF, dtype=DTYPE))
    entry = flat([0.0, 0.0, 0.0], math.pi / 2)          # normal along -Y (faces the source)
    hyp = flat([0.0, 10.0, 0.0], math.pi / 2 + math.pi / 4)  # 45 deg to the entry face

    origin = torch.tensor([[0.0, -20.0, 0.0]], dtype=DTYPE)
    direction = torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE)
    n = 1.5
    t1, p1, n1 = rt.intersect(entry, origin, direction)
    d1, tir1 = rt.refract_or_reflect(direction, n1, 1.0, n)
    assert not tir1.any()
    assert torch.allclose(d1, direction, atol=1e-9)  # normal incidence: no bend

    t2, p2, n2 = rt.intersect(hyp, p1, d1)
    d2, tir2 = rt.refract_or_reflect(d1, n2, n, 1.0)
    assert tir2.item(), "45 deg incidence on glass n=1.5 (critical 41.8 deg) must TIR"
    # Textbook right-angle-prism fold: exactly 90 deg turn, confined to the
    # Y-Z plane (x unchanged at 0), sign of the turn is a convention only.
    cos_turn = (d1 * d2).sum(-1).clamp(-1, 1)
    assert torch.allclose(torch.rad2deg(torch.acos(cos_turn)), torch.tensor([90.0], dtype=DTYPE), atol=1e-7)
    assert torch.allclose(d2[:, 0], torch.zeros(1, dtype=DTYPE), atol=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

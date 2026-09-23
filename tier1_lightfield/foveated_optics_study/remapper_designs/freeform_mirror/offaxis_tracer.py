"""Batched, differentiable real-ray tracer for plane-symmetric off-axis
remappers: tilted, decentred XY-polynomial surfaces, refracting or reflecting
(torch, float64).

Traces B designs x F fields x P pupil points in one call, eye -> panel, exactly
like the optiland model in offaxis_optiland.py (tests/test_offaxis_tracer.py
checks it). Frame and prescription conventions are the ones documented there.
A ray is lost (alive = False) when it misses a surface's sag domain, does not
converge, would travel backwards to reach a surface, or is totally internally
reflected. Lost rays keep their last finite state, detached, so no NaN is ever
produced, not even in the backward pass.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

import offaxis_optiland as oo

NEWTON_STEPS = 30
GRAD_STEPS = 2   # from a converged intersection one step gives the exact implicit derivative


class Batch(NamedTuple):
    y: torch.Tensor       # (B, S) surface origins (x = 0: plane symmetry)
    z: torch.Tensor       # (B, S)
    rx: torch.Tensor      # (B, S) tilt about x, radians
    c: torch.Tensor       # (B, S) base curvature
    k: torch.Tensor       # (B, S) conic
    xy: torch.Tensor      # (B, S, I, J) coefficient of x^i y^j
    n: torch.Tensor       # (B, S) index after each surface
    mirror: tuple         # (S,) static: which surfaces reflect
    image_sag: tuple = () # optional (r_grid, sag) mm: a radial tabulated image surface
                          # (shared by the batch), replacing the last surface's shape


def pack(prescriptions, device):
    S = len(prescriptions[0]["surfaces"])
    I = max(len(s["xy"]) for rx in prescriptions for s in rx["surfaces"])
    J = max(len(row) for rx in prescriptions for s in rx["surfaces"] for row in s["xy"])
    mirror = tuple(bool(s["mirror"]) for s in prescriptions[0]["surfaces"])
    if any(tuple(bool(s["mirror"]) for s in rx["surfaces"]) != mirror for rx in prescriptions):
        raise ValueError("every design in a batch must have the same mirror layout")
    rows = {key: [] for key in ("y", "z", "rx", "c", "k", "xy", "n")}
    for rx in prescriptions:
        if len(rx["surfaces"]) != S:
            raise ValueError("every design in a batch must have the same number of surfaces")
        n_medium = 1.0
        for key in rows:
            rows[key].append([])
        for s in rx["surfaces"]:
            rows["y"][-1].append(s["y_mm"])
            rows["z"][-1].append(s["z_mm"])
            rows["rx"][-1].append(math.radians(s["rx_deg"]))
            rows["c"][-1].append(0.0 if math.isinf(s["radius_mm"]) else 1.0 / s["radius_mm"])
            rows["k"][-1].append(s["conic"])
            grid = [[0.0] * J for _ in range(I)]
            for i, row in enumerate(s["xy"]):
                for j, value in enumerate(row):
                    grid[i][j] = value
            rows["xy"][-1].append(grid)
            n_medium = n_medium if s["mirror"] else s["index_after"]
            rows["n"][-1].append(n_medium)
    t = {key: torch.tensor(v, dtype=torch.float64, device=device) for key, v in rows.items()}
    return Batch(mirror=mirror, **t)


def _poly(x, y, C):
    """sum C_ij x^i y^j and its x and y derivatives; C (..., I, J) broadcasts."""
    I, J = C.shape[-2], C.shape[-1]
    # Horner in y for each power of x (value and y-derivative), then in x:
    # elementwise only, never a (..., I, J) intermediate.
    q, dq = [], []
    for i in range(I):
        v, dv = torch.zeros_like(y), torch.zeros_like(y)
        for j in reversed(range(J)):
            dv = dv * y + v
            v = v * y + C[..., i, j]
        q.append(v)
        dq.append(dv)
    val, ddx, ddy = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
    for i in reversed(range(I)):
        ddx = ddx * x + val
        val = val * x + q[i]
        ddy = ddy * x + dq[i]
    return val, ddx, ddy


def sag(x, y, c, k, C):
    """Sag, its gradient and the sqrt domain mask."""
    r2 = x**2 + y**2
    arg = 1.0 - (1.0 + k) * c * c * r2
    ok = arg > 0.0
    root = torch.sqrt(torch.where(ok, arg, torch.ones_like(arg)))
    g = c / (2.0 * root)                         # d/d(r^2) of c r^2 / (1 + root)
    val, ddx, ddy = _poly(x, y, C)
    return c * r2 / (1.0 + root) + val, 2.0 * g * x + ddx, 2.0 * g * y + ddy, ok


def _rot_x(v, a):
    """optiland's rotate_x by angle a (broadcast over the leading dims)."""
    ca, sa = torch.cos(a), torch.sin(a)
    return torch.stack([v[..., 0], v[..., 1] * ca - v[..., 2] * sa, v[..., 1] * sa + v[..., 2] * ca], -1)


def _newton(o, d, c, k, C, t):
    p = o + t[..., None] * d
    s, sx, sy, _ = sag(p[..., 0], p[..., 1], c, k, C)
    return t - (p[..., 2] - s) / (d[..., 2] - sx * d[..., 0] - sy * d[..., 1])


# Fused into one kernel: the eager step is memory-bound (about 18x slower).
_newton_fused = torch.compile(_newton, dynamic=True)


def table_sag(x, y, grid, table):
    """Radial tabulated sag, linear in r between grid points: sag, gradient and
    the domain mask (r inside the table)."""
    r = torch.sqrt(x**2 + y**2)
    ok = r <= grid[-1]
    i = torch.clamp(torch.searchsorted(grid, r.detach().contiguous()) - 1, 0, len(grid) - 2)
    g0, g1, s0, s1 = grid[i], grid[i + 1], table[i], table[i + 1]
    slope = (s1 - s0) / (g1 - g0)
    s = s0 + (r - g0) * slope
    rr = torch.clamp(r, min=1e-12)
    return s, slope * x / rr, slope * y / rr, ok


def _newton_table(o, d, grid, table, t):
    p = o + t[..., None] * d
    s, sx, sy, _ = table_sag(p[..., 0], p[..., 1], grid, table)
    return t - (p[..., 2] - s) / (d[..., 2] - sx * d[..., 0] - sy * d[..., 1])


def trace(batch: Batch, fields_deg, pupil, diagnostics=False):
    """fields_deg (F, 2) as (theta_x, theta_y); pupil (P, 2) normalised.
    Returns the landing (B, F, P, 2) in the image surface's local (x, y), the
    direction (B, F, P, 3) there in its local frame, and alive (B, F, P).
    With diagnostics also the global hit points per surface (B, S, F, P, 3), the
    sag-domain margin per surface (lost at 0) and sin^2 of the refracted angle
    per refracting surface (lost at 1)."""
    dev = batch.z.device
    B, S = batch.z.shape
    F, P = fields_deg.shape[0], pupil.shape[0]
    tx, ty = torch.tan(torch.deg2rad(fields_deg[:, 0])), torch.tan(torch.deg2rad(fields_deg[:, 1]))
    d = torch.stack([tx, ty, torch.ones_like(tx)], -1)
    d = (d / torch.linalg.norm(d, dim=-1, keepdim=True))[None, :, None, :].expand(B, F, P, 3)
    o = torch.cat([pupil * oo.PUPIL_RADIUS_MM, torch.zeros(P, 1, dtype=torch.float64, device=dev)], -1)
    o = o[None, None].expand(B, F, P, 3)
    alive = torch.ones(B, F, P, dtype=torch.bool, device=dev)
    n_before = torch.ones(B, F, P, dtype=torch.float64, device=dev)
    points, sin2, domain = [], [], []
    for s in range(S):
        origin = torch.stack([torch.zeros_like(batch.y[:, s]), batch.y[:, s], batch.z[:, s]], -1)[:, None, None]
        a = batch.rx[:, s, None, None]
        c, k, C = batch.c[:, s, None, None], batch.k[:, s, None, None], batch.xy[:, s, None, None]
        ol, dl = _rot_x(o - origin, -a), _rot_x(d, -a)                    # local frame
        tabulated = s == S - 1 and len(batch.image_sag) == 2
        step = (lambda o_, d_, t_: _newton_table(o_, d_, *batch.image_sag, t_)) if tabulated else \
            (lambda o_, d_, t_: _newton_fused(o_, d_, c, k, C, t_))  # noqa: E731
        with torch.no_grad():
            t = -ol[..., 2] / dl[..., 2]
            for _ in range(NEWTON_STEPS - GRAD_STEPS):
                t = step(ol.detach(), dl.detach(), t) if tabulated else \
                    _newton_fused(ol.detach(), dl.detach(), c.detach(), k.detach(), C.detach(), t)
            # every forward value stays finite, so the backward pass is finite too
            finite = torch.isfinite(t)
            t = torch.where(finite, t, torch.zeros_like(t))
        alive = alive & finite
        for _ in range(GRAD_STEPS):
            t = step(ol, dl, t)
        p = ol + t[..., None] * dl
        f, sx, sy, in_domain = (table_sag(p[..., 0], p[..., 1], *batch.image_sag) if tabulated
                                else sag(p[..., 0], p[..., 1], c, k, C))
        if diagnostics:
            domain.append(1.0 - (1.0 + k) * c * c * (p[..., 0] ** 2 + p[..., 1] ** 2))
        converged = (p[..., 2] - f).abs() < 1e-9
        alive = alive & in_domain & converged & (t > 0)
        normal = torch.stack([-sx, -sy, torch.ones_like(sx)], -1)
        normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True)
        cosi = (normal * dl).sum(-1)
        normal = torch.where((cosi < 0)[..., None], -normal, normal)       # along the ray
        cosi = cosi.abs()
        if batch.mirror[s]:
            d_new = dl - 2.0 * cosi[..., None] * normal
            n_after = n_before
        else:
            n_after = batch.n[:, s, None, None].expand(B, F, P)
            mu = n_before / n_after
            sin2t = mu**2 * (1.0 - cosi**2)
            if diagnostics:
                sin2.append(sin2t)
            alive = alive & (sin2t < 1.0)
            cost = torch.sqrt(torch.clamp(1.0 - sin2t, min=1e-30))
            d_new = mu[..., None] * dl + (cost - mu * cosi)[..., None] * normal
        if s == S - 1:
            p_img, d_img = p, dl
            break
        pg, dg = _rot_x(p, a) + origin, _rot_x(d_new / torch.linalg.norm(d_new, dim=-1, keepdim=True), a)
        if diagnostics:
            points.append(pg.detach())
        o = torch.where(alive[..., None], pg, o.detach())
        d = torch.where(alive[..., None], dg, d.detach())
        n_before = n_after
    if diagnostics:
        points.append((_rot_x(p_img, a) + origin).detach())
        return p_img[..., :2], d_img, alive, {"points": torch.stack(points, 1), "domain": torch.stack(domain, 1),
                                              "sin2t": torch.stack(sin2, 1)}
    return p_img[..., :2], d_img, alive

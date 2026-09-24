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
    image_sag: tuple = () # optional (x grid mm, y grid mm, sag (Nx, Ny) mm): a tabulated image surface
                          # (shared by the batch), replacing the last surface's shape
    spline: tuple = ()    # optional (surfaces (tuple), (x0, y0, h) mm, controls (B, Nx, Ny) mm): a bicubic
                          # B-spline added to the sag of each listed surface (one physical surface
                          # met several times, e.g. a pancake's half-mirror) (bspline_sag)


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


def _cubic(t):
    """Centred cubic B-spline (support |t| < 2) and its derivative."""
    a = t.abs()
    inner = 2.0 / 3.0 - a**2 + a**3 / 2.0
    outer = (2.0 - a) ** 3 / 6.0
    val = torch.where(a < 1.0, inner, torch.where(a < 2.0, outer, torch.zeros_like(a)))
    der = torch.where(a < 1.0, -2.0 * a + 1.5 * a**2, torch.where(a < 2.0, -0.5 * (2.0 - a) ** 2, torch.zeros_like(a)))
    return val, der * torch.sign(t)


def bspline_sag(x, y, grid, ctrl):
    """sum_jk ctrl[b, j, k] N((x - x0) / h - j) N((y - y0) / h - k) and its x, y
    slopes, N the centred cubic B-spline: control (j, k) sits at (x0 + j h,
    y0 + k h); the surface falls smoothly to zero within 2 h outside the grid.
    x, y (B, ...) and ctrl (B, Nx, Ny)."""
    x0, y0, h = grid
    B, Nx, Ny = ctrl.shape
    u, v = (x - x0) / h, (y - y0) / h
    iu, iv = torch.floor(u.detach()).long(), torch.floor(v.detach()).long()
    b = torch.arange(B, device=x.device).view((B,) + (1,) * (x.dim() - 1))
    flat = ctrl.reshape(B * Nx * Ny)
    val, dx, dy = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
    for dj in (-1, 0, 1, 2):
        j = iu + dj
        nx, dnx = _cubic(u - j)
        for dk in (-1, 0, 1, 2):
            k = iv + dk
            ny, dny = _cubic(v - k)
            inside = (j >= 0) & (j < Nx) & (k >= 0) & (k < Ny)
            c = flat[(b * Nx + j.clamp(0, Nx - 1)) * Ny + k.clamp(0, Ny - 1)] * inside
            val = val + c * nx * ny
            dx = dx + c * dnx * ny
            dy = dy + c * nx * dny
    return val, dx / h, dy / h


def _newton_spline(o, d, c, k, C, grid, ctrl, t):
    p = o + t[..., None] * d
    s, sx, sy, _ = sag(p[..., 0], p[..., 1], c, k, C)
    e, ex, ey = bspline_sag(p[..., 0], p[..., 1], grid, ctrl)
    return t - (p[..., 2] - s - e) / (d[..., 2] - (sx + ex) * d[..., 0] - (sy + ey) * d[..., 1])


_newton_spline_fused = torch.compile(_newton_spline, dynamic=True)


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


def table_sag(x, y, grid_x, grid_y, table):
    """Tabulated image surface on a uniform (x, y) grid, bilinear: sag, its x and
    y slopes, and the domain mask (always true: beyond the grid the surface keeps
    its edge values, flat across the edge, so a ray landing off the panel is
    penalised by the merit, not lost)."""
    dx, dy = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]
    fx = torch.clamp((x - grid_x[0]) / dx, 0.0, len(grid_x) - 1.0)
    fy = torch.clamp((y - grid_y[0]) / dy, 0.0, len(grid_y) - 1.0)
    inside_x = ((x - grid_x[0]) / dx >= 0.0) & ((x - grid_x[0]) / dx <= len(grid_x) - 1.0)
    inside_y = ((y - grid_y[0]) / dy >= 0.0) & ((y - grid_y[0]) / dy <= len(grid_y) - 1.0)
    i = torch.clamp(torch.floor(fx.detach()).long(), 0, len(grid_x) - 2)
    j = torch.clamp(torch.floor(fy.detach()).long(), 0, len(grid_y) - 2)
    wx, wy = fx - i, fy - j
    t00, t10, t01, t11 = table[i, j], table[i + 1, j], table[i, j + 1], table[i + 1, j + 1]
    s = (1 - wx) * (1 - wy) * t00 + wx * (1 - wy) * t10 + (1 - wx) * wy * t01 + wx * wy * t11
    sx = ((1 - wy) * (t10 - t00) + wy * (t11 - t01)) / dx * inside_x
    sy = ((1 - wx) * (t01 - t00) + wx * (t11 - t10)) / dy * inside_y
    return s, sx, sy, torch.ones_like(s, dtype=torch.bool)


def _newton_table(o, d, grid_x, grid_y, table, t):
    p = o + t[..., None] * d
    s, sx, sy, _ = table_sag(p[..., 0], p[..., 1], grid_x, grid_y, table)
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
        tabulated = s == S - 1 and len(batch.image_sag) == 3
        splined = len(batch.spline) == 3 and s in batch.spline[0]
        if tabulated and splined:
            raise ValueError("the tabulated image surface cannot also carry a spline")
        if tabulated:
            step = lambda o_, d_, t_: _newton_table(o_, d_, *batch.image_sag, t_)  # noqa: E731
        elif splined:
            step = lambda o_, d_, t_: _newton_spline_fused(o_, d_, c, k, C, *batch.spline[1:], t_)  # noqa: E731
        else:
            step = lambda o_, d_, t_: _newton_fused(o_, d_, c, k, C, t_)  # noqa: E731
        with torch.no_grad():
            t = -ol[..., 2] / dl[..., 2]
            for _ in range(NEWTON_STEPS - GRAD_STEPS):
                if tabulated:
                    t = step(ol.detach(), dl.detach(), t)
                elif splined:
                    t = _newton_spline_fused(ol.detach(), dl.detach(), c.detach(), k.detach(), C.detach(),
                                             batch.spline[1], batch.spline[2].detach(), t)
                else:
                    t = _newton_fused(ol.detach(), dl.detach(), c.detach(), k.detach(), C.detach(), t)
            # every forward value stays finite, so the backward pass is finite too
            finite = torch.isfinite(t)
            t = torch.where(finite, t, torch.zeros_like(t))
        alive = alive & finite
        for _ in range(GRAD_STEPS):
            t = step(ol, dl, t)
        p = ol + t[..., None] * dl
        f, sx, sy, in_domain = (table_sag(p[..., 0], p[..., 1], *batch.image_sag) if tabulated
                                else sag(p[..., 0], p[..., 1], c, k, C))
        if splined:
            e, ex, ey = bspline_sag(p[..., 0], p[..., 1], *batch.spline[1:])
            f, sx, sy = f + e, sx + ex, sy + ey
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

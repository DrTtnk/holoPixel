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
NEWTON_TOL_MM = 1e-12   # the free steps stop when no finite ray moves more than this


class Batch(NamedTuple):
    y: torch.Tensor       # (B, S) surface origins (x = 0: plane symmetry)
    z: torch.Tensor       # (B, S)
    rx: torch.Tensor      # (B, S) tilt about x, radians
    c: torch.Tensor       # (B, S) base curvature
    k: torch.Tensor       # (B, S) conic
    xy: torch.Tensor      # (B, S, T) coefficient of x^i y^j for each (i, j) of terms
    n: torch.Tensor       # (B, S) index after each surface
    mirror: tuple         # (S,) static: which surfaces reflect
    terms: tuple          # (T,) static: the (i, j) of the polynomial terms, shared by every surface
    image_sag: tuple = () # optional (x grid mm, y grid mm, sag (Nx, Ny) mm): a tabulated image surface
                          # (shared by the batch), replacing the last surface's shape
    spline: tuple = ()    # optional (surfaces (tuple), (x0, y0, h) mm, controls (B, Nx, Ny) mm): a bicubic
                          # B-spline added to the sag of each listed surface (one physical surface
                          # met several times, e.g. a pancake's half-mirror) (bspline_sag)


def pack(prescriptions, device):
    """Prescriptions -> Batch; the terms are every (i, j) with a nonzero
    coefficient in any surface of any prescription."""
    S = len(prescriptions[0]["surfaces"])
    mirror = tuple(bool(s["mirror"]) for s in prescriptions[0]["surfaces"])
    if any(tuple(bool(s["mirror"]) for s in rx["surfaces"]) != mirror for rx in prescriptions):
        raise ValueError("every design in a batch must have the same mirror layout")
    terms = tuple(sorted({(i, j) for rx in prescriptions for s in rx["surfaces"]
                          for i, row in enumerate(s["xy"]) for j, value in enumerate(row) if value != 0.0}))
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
            rows["xy"][-1].append([s["xy"][i][j] if i < len(s["xy"]) and j < len(s["xy"][i]) else 0.0
                                   for i, j in terms])
            n_medium = n_medium if s["mirror"] else s["index_after"]
            rows["n"][-1].append(n_medium)
    t = {key: torch.tensor(v, dtype=torch.float64, device=device).reshape(len(prescriptions), S, -1)
         if key == "xy" else torch.tensor(v, dtype=torch.float64, device=device) for key, v in rows.items()}
    return Batch(mirror=mirror, terms=terms, **t)


def _poly(x, y, C, terms):
    """sum_n C[..., n] x^i y^j over terms (i, j), and its x and y derivatives.
    Horner's rule in y along each row of equal i, then in x across the rows (in
    X = x^2 when every i is even): only the listed terms cost work. C (..., T)
    broadcasts against x and y; terms is static."""
    rows = {}
    for n, (i, j) in enumerate(terms):
        rows.setdefault(i, {})[j] = n
    zero = torch.zeros_like(x)
    if not rows:
        return zero, zero, zero
    even = all(i % 2 == 0 for i in rows)
    step = 2 if even else 1
    X = x * x if even else x
    val, dval, ddy = zero, zero, zero                  # dval: derivative in X
    for m in reversed(range(max(rows) // step + 1)):
        row = rows.get(m * step, {})
        v, dv = zero, zero
        for j in reversed(range(max(row) + 1 if row else 0)):
            dv = dv * y + v
            v = v * y + C[..., row[j]] if j in row else v * y
        dval = dval * X + val
        val = val * X + v
        ddy = ddy * X + dv
    return val, (2.0 * x * dval if even else dval), ddy


def sag(x, y, c, k, C, terms):
    """Sag, its gradient and the sqrt domain mask."""
    r2 = x**2 + y**2
    arg = 1.0 - (1.0 + k) * c * c * r2
    ok = arg > 0.0
    root = torch.sqrt(torch.where(ok, arg, torch.ones_like(arg)))
    g = c / (2.0 * root)                         # d/d(r^2) of c r^2 / (1 + root)
    val, ddx, ddy = _poly(x, y, C, terms)
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


def _newton_spline(o, d, c, k, C, terms, grid, ctrl, t):
    p = o + t[..., None] * d
    s, sx, sy, _ = sag(p[..., 0], p[..., 1], c, k, C, terms)
    e, ex, ey = bspline_sag(p[..., 0], p[..., 1], grid, ctrl)
    return t - (p[..., 2] - s - e) / (d[..., 2] - (sx + ex) * d[..., 0] - (sy + ey) * d[..., 1])


def _rot_x(v, a):
    """optiland's rotate_x by angle a (broadcast over the leading dims)."""
    return _rot(v, torch.cos(a), torch.sin(a))


def _rot(v, ca, sa):
    """rotate_x by the angle of cosine ca and sine sa: in the fused kernels an fp64
    cos and sin per ray element would cost more than the rest of the surface."""
    return torch.stack([v[..., 0], v[..., 1] * ca - v[..., 2] * sa, v[..., 1] * sa + v[..., 2] * ca], -1)


def _newton(o, d, c, k, C, terms, t):
    p = o + t[..., None] * d
    s, sx, sy, _ = sag(p[..., 0], p[..., 1], c, k, C, terms)
    return t - (p[..., 2] - s) / (d[..., 2] - sx * d[..., 0] - sy * d[..., 1])




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


def _stepped(fused):
    """The free Newton steps as a loop of one fused step each, at most
    NEWTON_STEPS - GRAD_STEPS: from the fourth on, every second step checks
    whether any ray still moves more than NEWTON_TOL_MM (a lost ray, NaN or
    infinite, does not count: it stays lost), and stops the loop when none does."""
    def loop(o, d, *args):
        *shape, t = args
        for n in range(NEWTON_STEPS - GRAD_STEPS):
            new = fused(o, d, *shape, t)
            if n >= 3 and n % 2 == 1 and not bool(((new - t).abs() > NEWTON_TOL_MM).any()):
                return new
            t = new
        return t
    return loop


# Fused into one kernel each: the eager step is memory-bound (about 18x slower).
_newton_fused = torch.compile(_newton, dynamic=True)
_newton_spline_fused = torch.compile(_newton_spline, dynamic=True)
_newton_table_fused = torch.compile(_newton_table, dynamic=True)
_newton_loop = _stepped(_newton_fused)
_newton_spline_loop = _stepped(_newton_spline_fused)
_newton_table_loop = _stepped(_newton_table_fused)


def _local(o, d, origin, ca, sa):
    return _rot(o - origin, ca, -sa), _rot(d, ca, -sa)


def _hit(kind, mirror, ol, dl, t, ca, sa, origin, c, k, n_before, n_after, shape):
    """One surface after its Newton solve (local ray ol + t dl): the hit point p
    (local), the ray leaving it in the global frame (pg, dg), whether the ray
    survives (in the sag domain, converged, forwards, no TIR), the sag-domain
    margin and, on a refracting surface, sin^2 of the refracted angle. kind
    ("poly", "spline", "table"), shape and mirror are static."""
    p = ol + t[..., None] * dl
    if kind == "table":
        f, sx, sy, ok = table_sag(p[..., 0], p[..., 1], *shape)
    else:
        f, sx, sy, ok = sag(p[..., 0], p[..., 1], c, k, *shape[:2])
    if kind == "spline":
        e, ex, ey = bspline_sag(p[..., 0], p[..., 1], *shape[2:])
        f, sx, sy = f + e, sx + ex, sy + ey
    domain = 1.0 - (1.0 + k) * c * c * (p[..., 0] ** 2 + p[..., 1] ** 2)
    ok = ok & ((p[..., 2] - f).abs() < 1e-9) & (t > 0)
    normal = torch.stack([-sx, -sy, torch.ones_like(sx)], -1)
    normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True)
    cosi = (normal * dl).sum(-1)
    normal = torch.where((cosi < 0)[..., None], -normal, normal)           # along the ray
    cosi = cosi.abs()
    if mirror:
        d_new = dl - 2.0 * cosi[..., None] * normal
        extra = ()
    else:
        mu = n_before / n_after
        sin2t = mu**2 * (1.0 - cosi**2)
        ok = ok & (sin2t < 1.0)
        cost = torch.sqrt(torch.clamp(1.0 - sin2t, min=1e-30))
        d_new = mu[..., None] * dl + (cost - mu * cosi)[..., None] * normal
        extra = (sin2t,)
    pg, dg = _rot(p, ca, sa) + origin, _rot(d_new / torch.linalg.norm(d_new, dim=-1, keepdim=True), ca, sa)
    return (p, pg, dg, ok, domain) + extra


# one function (one compile cache) per surface kind: each has a mirror and a lens variant
def _hit_poly(*args):
    return _hit("poly", *args)


def _hit_spline(*args):
    return _hit("spline", *args)


def _hit_table(*args):
    return _hit("table", *args)


_local_fused = torch.compile(_local, dynamic=True)
_hit_fused = {"poly": torch.compile(_hit_poly, dynamic=True), "spline": torch.compile(_hit_spline, dynamic=True),
              "table": torch.compile(_hit_table, dynamic=True)}


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
    n_before = torch.ones(B, 1, 1, dtype=torch.float64, device=dev)
    points, sin2, domain = [], [], []
    for s in range(S):
        origin = torch.stack([torch.zeros_like(batch.y[:, s]), batch.y[:, s], batch.z[:, s]], -1)[:, None, None]
        a = batch.rx[:, s, None, None]
        ca, sa = torch.cos(a), torch.sin(a)
        c, k, C = batch.c[:, s, None, None], batch.k[:, s, None, None], batch.xy[:, s, None, None]
        ol, dl = _local_fused(o, d, origin, ca, sa)
        tabulated = s == S - 1 and len(batch.image_sag) == 3
        splined = len(batch.spline) == 3 and s in batch.spline[0]
        if tabulated and splined:
            raise ValueError("the tabulated image surface cannot also carry a spline")
        if tabulated:
            kind, shape, step, free = "table", batch.image_sag, _newton_table_fused, _newton_table_loop
        elif splined:
            kind, shape = "spline", (C, batch.terms, *batch.spline[1:])
            step, free = _newton_spline_fused, _newton_spline_loop
        else:
            kind, shape, step, free = "poly", (C, batch.terms), _newton_fused, _newton_loop
        newton_shape = shape if tabulated else (c, k, *shape)
        with torch.no_grad():
            t = free(ol, dl, *newton_shape, -ol[..., 2] / dl[..., 2])
            # every forward value stays finite, so the backward pass is finite too
            finite = torch.isfinite(t)
            t = torch.where(finite, t, torch.zeros_like(t))
        for _ in range(GRAD_STEPS):
            t = step(ol, dl, *newton_shape, t)
        n_after = n_before if batch.mirror[s] else batch.n[:, s, None, None]
        p, pg, dg, ok, margin, *bent = _hit_fused[kind](batch.mirror[s], ol, dl, t, ca, sa, origin, c, k, n_before,
                                                        n_after, shape)
        alive = alive & finite & ok
        if diagnostics:
            domain.append(margin)
            sin2.extend(bent)
            points.append(pg.detach())
        if s == S - 1:
            p_img, d_img = p, dl
            break
        o = torch.where(alive[..., None], pg, o.detach())
        d = torch.where(alive[..., None], dg, d.detach())
        n_before = n_after
    if diagnostics:
        return p_img[..., :2], d_img, alive, {"points": torch.stack(points, 1), "domain": torch.stack(domain, 1),
                                              "sin2t": torch.stack(sin2, 1)}
    return p_img[..., :2], d_img, alive

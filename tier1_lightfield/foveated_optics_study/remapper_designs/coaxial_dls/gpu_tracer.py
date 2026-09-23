"""Batched, differentiable real-ray tracer for coaxial remappers (torch, float64).

Traces B designs x F field angles x P pupil points in one call, eye -> panel,
exactly like the optiland model in coaxial_optiland.py (tests/test_gpu_tracer.py
checks every landing against optiland to 1e-6 mm). Coordinates: z along the
axis from the stop (the eye pupil) towards the panel, y in the meridian of the
field, x across it; the pupil is a disc of radius co.PUPIL_RADIUS_MM at z = 0.

Surfaces are even aspheres, sag(r) = c r^2 / (1 + sqrt(1 - (1+k) c^2 r^2))
+ a4 r^4 + a6 r^6 + a8 r^8, vertex z_v. The last surface is the image
surface (flat when c = k = 0, curved for a fibre-optic faceplate).
A ray is lost (alive = False) when it misses a surface's domain, turns back,
or is totally internally reflected.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

import coaxial_optiland as co

NEWTON_STEPS = 30
GRAD_STEPS = 2   # only the last Newton steps carry gradients: from a converged
                 # intersection one step already gives the exact implicit derivative


class Batch(NamedTuple):
    z: torch.Tensor      # (B, S) vertex positions along the axis, stop at 0
    c: torch.Tensor      # (B, S) curvatures
    k: torch.Tensor      # (B, S) conics
    a: torch.Tensor      # (B, S, 3) r^4, r^6, r^8 coefficients
    n: torch.Tensor      # (B, S) index AFTER each surface (last = medium at the image)
    image_sag: tuple = ()  # optional (r^2 grid mm^2, sag mm): a radial tabulated image surface


def pack(prescriptions, device, image=None):
    """JSON prescriptions (coaxial_optiland format) -> Batch. `image` optionally
    gives (radius_mm, conic) for a curved image surface, else flat."""
    rows = []
    for rx in prescriptions:
        z, zc, cs, ks, aa, ns = [], rx["eye_relief_mm"], [], [], [], []
        for el in rx["elements"]:
            for side, after, n_after in (("front", el["thickness_mm"], el["index"]),
                                         ("back", el["gap_after_mm"], 1.0)):
                s = el[side]
                z.append(zc)
                cs.append(0.0 if math.isinf(s["radius_mm"]) else 1.0 / s["radius_mm"])
                ks.append(s["conic"])
                coef = list(s["coefficients"]) + [0.0] * 4
                aa.append(coef[1:4])
                ns.append(n_after)
                zc += after
        z.append(zc)
        cs.append(0.0 if image is None else 1.0 / image[0])
        ks.append(0.0 if image is None else image[1])
        aa.append([0.0, 0.0, 0.0])
        ns.append(1.0)
        rows.append((z, cs, ks, aa, ns))
    t = lambda i: torch.tensor([r[i] for r in rows], dtype=torch.float64, device=device)  # noqa: E731
    return Batch(z=t(0), c=t(1), k=t(2), a=t(3), n=t(4))


def sag(r2, c, k, a):
    """Even-asphere sag from r^2 (broadcasting); also returns the sqrt domain mask."""
    arg = 1.0 - (1.0 + k) * c * c * r2
    ok = arg > 0.0
    root = torch.sqrt(torch.where(ok, arg, torch.ones_like(arg)))
    s = c * r2 / (1.0 + root) + a[..., 0] * r2**2 + a[..., 1] * r2**3 + a[..., 2] * r2**4
    return s, ok


def dsag_dr2(r2, c, k, a):
    arg = 1.0 - (1.0 + k) * c * c * r2
    root = torch.sqrt(torch.clamp(arg, min=1e-30))
    # d/d(r2) of c r2 / (1 + root): c / (2 root), using d(root)/d(r2) = -(1+k) c^2 / (2 root)
    return c / (2.0 * root) + 2 * a[..., 0] * r2 + 3 * a[..., 1] * r2**2 + 4 * a[..., 2] * r2**3


def table_sag(r2, grid_r2, table):
    """Radial tabulated sag, linear in r^2 between grid points (no square root:
    finite derivatives on the axis): sag, d(sag)/d(r^2), and the domain mask."""
    ok = r2 <= grid_r2[-1]
    i = torch.clamp(torch.searchsorted(grid_r2, r2.detach().contiguous()) - 1, 0, len(grid_r2) - 2)
    slope = (table[i + 1] - table[i]) / (grid_r2[i + 1] - grid_r2[i])
    return table[i] + (r2 - grid_r2[i]) * slope, slope, ok


def _newton_table(o, d, zv, grid, table, t):
    p = o + t[..., None] * d
    r2 = p[..., 0] ** 2 + p[..., 1] ** 2
    f, g, _ = table_sag(r2, grid, table)
    return t - (p[..., 2] - zv - f) / (d[..., 2] - g * 2.0 * (p[..., 0] * d[..., 0] + p[..., 1] * d[..., 1]))


def _newton(o, d, zv, c, k, a, t):
    p = o + t[..., None] * d
    r2 = p[..., 0] ** 2 + p[..., 1] ** 2
    f, _ = sag(r2, c, k, a)
    g = dsag_dr2(r2, c, k, a)
    F_t = p[..., 2] - zv - f
    dF = d[..., 2] - g * 2.0 * (p[..., 0] * d[..., 0] + p[..., 1] * d[..., 1])
    return t - F_t / dF


def trace(batch: Batch, fields_deg, pupil, diagnostics=False):
    """fields_deg: (F,), pupil: (P, 2) normalised (x, y). Returns landing (B, F, P, 2)
    as (x, y) on the image surface, direction (B, F, P, 3) there, alive (B, F, P).
    With diagnostics, also a dict of per-surface hit radius, TIR margin (sin^2 of
    the refracted angle, lost at 1) and sag-domain margin (lost at 0)."""
    radius, sin2, domain, points = [], [], [], []
    B, S = batch.z.shape
    F, P = fields_deg.shape[0], pupil.shape[0]
    th = torch.deg2rad(fields_deg)
    d = torch.stack([torch.zeros_like(th), torch.sin(th), torch.cos(th)], dim=-1)  # (F, 3)
    d = d[None, :, None, :].expand(B, F, P, 3)
    o = torch.zeros(B, F, P, 3, dtype=torch.float64, device=batch.z.device)
    o = o + torch.cat([pupil * co.PUPIL_RADIUS_MM, torch.zeros(P, 1, dtype=torch.float64,
                                                              device=batch.z.device)], dim=-1)
    alive = torch.ones(B, F, P, dtype=torch.bool, device=batch.z.device)
    n_before = torch.ones(B, F, P, dtype=torch.float64, device=batch.z.device)
    for s in range(S):
        zv = batch.z[:, s, None, None]
        c, k = batch.c[:, s, None, None], batch.k[:, s, None, None]
        a = batch.a[:, s, None, None, :]
        t = (zv - o[..., 2]) / d[..., 2]
        tabulated = s == S - 1 and len(batch.image_sag) == 2
        with torch.no_grad():
            t_free = t.detach()
            for _ in range(NEWTON_STEPS - GRAD_STEPS):
                t_free = (_newton_table(o.detach(), d.detach(), zv.detach(), *batch.image_sag, t_free) if tabulated
                          else _newton(o.detach(), d.detach(), zv.detach(), c.detach(), k.detach(), a.detach(),
                                       t_free))
        t = t_free
        for _ in range(GRAD_STEPS):
            t = _newton_table(o, d, zv, *batch.image_sag, t) if tabulated else _newton(o, d, zv, c, k, a, t)
        p = o + t[..., None] * d
        r2 = p[..., 0] ** 2 + p[..., 1] ** 2
        if tabulated:
            f, g_tab, in_domain = table_sag(r2, *batch.image_sag)
        else:
            f, in_domain = sag(r2, c, k, a)
        if diagnostics:
            radius.append(torch.sqrt(r2))
            domain.append(1.0 - (1.0 + k) * c * c * r2)
            points.append(p.detach())
        g = g_tab if tabulated else dsag_dr2(r2, c, k, a)
        normal = torch.stack([-2.0 * g * p[..., 0], -2.0 * g * p[..., 1], torch.ones_like(g)], dim=-1)
        normal = normal / torch.linalg.norm(normal, dim=-1, keepdim=True)
        converged = (p[..., 2] - zv - f).abs() < 1e-9
        alive = alive & in_domain & converged & (t > 0)
        # A lost ray keeps its last finite state (detached), so no NaN or inf is
        # ever produced downstream, not even in the backward pass.
        o = torch.where(alive[..., None], p, o.detach())
        normal = torch.where(alive[..., None], normal, torch.tensor([0.0, 0.0, 1.0], dtype=normal.dtype,
                                                                     device=normal.device))
        if s == S - 1:
            break
        n_after = batch.n[:, s, None, None].expand(B, F, P)
        cosi = (normal * d).sum(-1)                   # normal points +z, rays go +z
        mu = n_before / n_after
        sin2t = mu**2 * (1.0 - cosi**2)
        if diagnostics:
            sin2.append(sin2t)
        alive = alive & (sin2t < 1.0)
        cost = torch.sqrt(torch.clamp(1.0 - sin2t, min=1e-30))
        d_new = mu[..., None] * d + (cost - mu * cosi)[..., None] * normal
        d_new = d_new / torch.linalg.norm(d_new, dim=-1, keepdim=True)
        alive = alive & (d_new[..., 2] > 0)
        d = torch.where(alive[..., None], d_new, d.detach())
        n_before = n_after
    if diagnostics:
        return o[..., :2], d, alive, {"radius": torch.stack(radius, 1), "sin2t": torch.stack(sin2, 1),
                                      "domain": torch.stack(domain, 1), "points": torch.stack(points, 1)}
    return o[..., :2], d, alive

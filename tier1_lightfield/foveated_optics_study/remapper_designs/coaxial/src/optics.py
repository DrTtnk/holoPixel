"""Differentiable sequential ray tracer for a coaxial (rotationally symmetric)
refractive system, axis = world +Y, torch float64.

Surfaces are even aspheres about the Y axis: for radial distance r in the
local x-z plane,

    sag(r) = c r^2 / (1 + sqrt(1 - (1+k) c^2 r^2)) + a4 r^4 + a6 r^6 + a8 r^8
    y(r)   = vertex_y + sag(r)

d(sag)/dr = c r / sqrt(1 - (1+k) c^2 r^2) + 4 a4 r^3 + 6 a6 r^5 + 8 a8 r^7,
confirmed against sympy.diff in tests/test_optics_math.py (schoolbook chain
rule, but checked per project rule anyway).

Ray/surface intersection is Newton's method on F(t) = O_y + t D_y - vertex_y
- sag(r(t)); the surface normal is the normalized implicit-function gradient.
Refraction is the standard vector form of Snell's law (e.g. PBRT 8.2), with
total internal reflection reflecting instead of refracting -- matching the
Cycles material `glass()` in lf_blender.py exactly (see its docstring).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

DTYPE = torch.float64


@dataclass
class Surface:
    """One even-asphere (or flat, c=0) surface, vertex on the Y axis."""
    vertex_y: torch.Tensor   # scalar
    c: torch.Tensor          # scalar, curvature = 1/R
    k: torch.Tensor          # scalar, conic
    a4: torch.Tensor
    a6: torch.Tensor
    a8: torch.Tensor
    aperture_mm: float       # clear (max) radius, for meshing/validity only

    @staticmethod
    def flat(vertex_y, aperture_mm):
        z = torch.zeros((), dtype=DTYPE)
        return Surface(as_t(vertex_y), z.clone(), z.clone(), z.clone(), z.clone(), z.clone(), aperture_mm)


def as_t(x):
    return x if torch.is_tensor(x) else torch.tensor(float(x), dtype=DTYPE)


def _safe_sphere_r(r, s: Surface):
    """Clamp r (not the sag) so the sphere term's sqrt domain never goes
    invalid, even for a transient Newton overshoot far outside the physical
    aperture: r stays exact wherever (1+k)c^2 r^2 < 0.98 already held."""
    u = (1.0 + s.k) * s.c**2
    r2_max = 0.98 / torch.clamp(u, min=1e-12)
    r2 = torch.minimum(r**2, r2_max)
    return torch.sqrt(r2)


def sag(r, s: Surface):
    rs = _safe_sphere_r(r, s)
    denom = 1.0 - (1.0 + s.k) * s.c**2 * rs**2
    denom = torch.clamp(denom, min=0.02)
    sphere = s.c * rs**2 / (1.0 + torch.sqrt(denom))
    return sphere + s.a4 * r**4 + s.a6 * r**6 + s.a8 * r**8


def dsag_dr(r, s: Surface):
    """d(sag)/dr, consistent with sag()'s r-clamp: the sphere term is frozen
    (zero slope) once r passes the clamp, matching sag() being frozen there."""
    rs = _safe_sphere_r(r, s)
    denom = 1.0 - (1.0 + s.k) * s.c**2 * rs**2
    denom = torch.clamp(denom, min=0.02)
    in_domain = (r <= rs).to(r.dtype)
    return in_domain * s.c * rs / torch.sqrt(denom) + 4 * s.a4 * r**3 + 6 * s.a6 * r**5 + 8 * s.a8 * r**7


def intersect(origin, direction, s: Surface, n_iter=12):
    """origin, direction: (..., 3) float64, direction unit. Returns t (..., )."""
    ox, oy, oz = origin[..., 0], origin[..., 1], origin[..., 2]
    dx, dy, dz = direction[..., 0], direction[..., 1], direction[..., 2]
    t = (s.vertex_y - oy) / dy
    for _ in range(n_iter):
        px, pz = ox + t * dx, oz + t * dz
        r = torch.sqrt(px**2 + pz**2 + 1e-24)
        F = oy + t * dy - s.vertex_y - sag(r, s)
        drdt = (px * dx + pz * dz) / r
        dF = dy - dsag_dr(r, s) * drdt
        dF = torch.where(dF.abs() < 1e-12, torch.full_like(dF, 1e-12), dF)
        t = t - F / dF
    return t


def surface_point_normal(origin, direction, s: Surface):
    t = intersect(origin, direction, s)
    p = origin + t[..., None] * direction
    r = torch.sqrt(p[..., 0]**2 + p[..., 2]**2 + 1e-24)
    dsdr = dsag_dr(r, s)
    gx = -dsdr * p[..., 0] / r
    gz = -dsdr * p[..., 2] / r
    gy = torch.ones_like(gx)
    n = torch.stack([gx, gy, gz], dim=-1)
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
    return t, p, n


def refract(direction, normal, n1, n2):
    """Vector Snell's law. `normal` is re-oriented to face the incident ray.
    Returns (new_direction, is_tir, sin2t): sin2t > 1 <=> TIR, and how far
    below 1 it sits is a smooth (differentiable) margin against TIR, useful
    as an anticipatory regulariser during optimisation."""
    facing = torch.where((torch.sum(normal * direction, dim=-1, keepdim=True) > 0), -1.0, 1.0)
    n = normal * facing
    cosi = -torch.sum(n * direction, dim=-1)
    mu = n1 / n2
    sin2t = mu**2 * torch.clamp(1.0 - cosi**2, min=0.0)
    tir = sin2t > 1.0
    cost = torch.sqrt(torch.clamp(1.0 - sin2t, min=0.0))
    refr = mu * direction + (mu * cosi - cost)[..., None] * n
    refl = direction + 2.0 * cosi[..., None] * n
    out = torch.where(tir[..., None], refl, refr)
    out = out / torch.linalg.norm(out, dim=-1, keepdim=True)
    return out, tir, sin2t


@dataclass
class Element:
    """One glass element: front and back even-asphere surfaces, index n."""
    front: Surface
    back: Surface
    index: float


def trace_system(origin, direction, elements: list[Element], image_y):
    """Trace through a sequence of glass elements (air on both sides of each)
    to a flat image plane at y = image_y. Returns (hit_xyz, hit_dir, alive,
    clean, sin2t_margins, forward_y, t_margins): `clean` is False for any ray
    that underwent total internal reflection anywhere on its path (its
    landing point is still physically computed, but is an unbounded,
    unhelpful training signal for a position-matching loss -- see
    design_opt.py). `sin2t_margins` is a list of per-surface sin2t tensors,
    a smooth (pre-TIR) proxy usable as an anticipatory regulariser.
    `t_margins` is a list of the per-surface Newton solution t (mm): going
    non-positive is what actually flips `alive` false, so it too is exposed
    for use as a smooth, anticipatory regulariser (see design_opt.py)."""
    o, d = origin, direction
    alive = torch.ones(direction.shape[:-1], dtype=torch.bool)
    clean = torch.ones(direction.shape[:-1], dtype=torch.bool)  # no TIR anywhere on the path
    margins = []
    forward_y = []  # d_y after each refraction: a ray that turns backward goes dead with no gradient
    t_margins = []
    for el in elements:
        for surf, n_out in ((el.front, el.index), (el.back, 1.0)):
            n_in = 1.0 if surf is el.front else el.index
            t, p, n = surface_point_normal(o, d, surf)
            alive = alive & (t > 0)
            t_margins.append(t)
            d, tir, sin2t = refract(d, n, n_in, n_out)
            clean = clean & (~tir)
            margins.append(sin2t)
            forward_y.append(d[..., 1])
            o = p
    img = Surface.flat(as_t(image_y), 1e9)
    t = intersect(o, d, img)
    hit = o + t[..., None] * d
    alive = alive & (t > 0)
    t_margins.append(t)
    return hit, d, alive, clean, margins, forward_y, t_margins

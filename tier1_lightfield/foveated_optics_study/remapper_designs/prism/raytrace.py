"""Differentiable real ray tracer for a freeform TIR-prism remapper.

One glass solid, three optically active freeform patches (S1 eye-facing, used
twice; S2 mirror-coated back face; S3 entry face towards the MLA/panel), each
a polynomial sag on its own local plane, tilted about the world X axis only
(so the whole solid is plane-symmetric about x = 0, matching the symmetric
70 deg horizontal field). All geometry is torch float64, CPU (the ray counts
here are far too small to need the GPU, which two other agents are sharing).

Local frame of a surface (origin O, tilt theta, all rotation about world X):
    e_x = (1, 0, 0)                      # world X, shared by every surface
    e_u = (0, cos theta, sin theta)      # in-plane fold coordinate
    e_n = (0, -sin theta, cos theta)     # nominal surface normal direction
World point = O + x_l * e_x + u_l * e_u + sag(x_l, u_l) * e_n.
sag is a polynomial in x_l (even powers only) and u_l (see TERMS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

DTYPE = torch.float64

# (power of x_l, power of u_l), x powers even only (plane symmetry about x=0).
TERMS = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
    (4, 0), (4, 1), (4, 2),
    (6, 0),
]
N_COEFF = len(TERMS)


APERTURE_MM = 15.0  # normalises x_l, u_l before the polynomial so a coeff of
                    # order 1 means "order-1-mm sag at one aperture radius",
                    # keeping gradients (which carry factors of u_l**6) sane.


@dataclass
class Surface:
    origin: torch.Tensor   # (3,)
    theta: torch.Tensor    # scalar, rotation about world X
    coeff: torch.Tensor    # (N_COEFF,)

    def frame(self):
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        zero, one = torch.zeros((), dtype=DTYPE), torch.ones((), dtype=DTYPE)
        e_x = torch.stack([one, zero, zero])
        e_u = torch.stack([zero, c, s])
        e_n = torch.stack([zero, -s, c])
        return e_x, e_u, e_n

    def local(self, p):
        e_x, e_u, e_n = self.frame()
        rel = p - self.origin
        return (rel * e_x).sum(-1), (rel * e_u).sum(-1), (rel * e_n).sum(-1)

    def sag_and_grad(self, x_l, u_l):
        """sag = sum coeff_i * (x_l/S)^px * (u_l/S)^qu, S = APERTURE_MM, so a
        unit coefficient gives an order-1-mm sag at one aperture radius."""
        S = APERTURE_MM
        xh, uh = x_l / S, u_l / S
        s = torch.zeros_like(x_l)
        dsdx = torch.zeros_like(x_l)
        dsdu = torch.zeros_like(x_l)
        for a, (px, qu) in zip(self.coeff, TERMS):
            xp = xh**px if px else torch.ones_like(xh)
            uq = uh**qu if qu else torch.ones_like(uh)
            s = s + a * xp * uq
            if px:
                dsdx = dsdx + a * px * xh ** (px - 1) * uq / S
            if qu:
                dsdu = dsdu + a * qu * xp * uh ** (qu - 1) / S
        return s, dsdx, dsdu

    def normal(self, x_l, u_l):
        _, dsdx, dsdu = self.sag_and_grad(x_l, u_l)
        e_x, e_u, e_n = self.frame()
        n = -dsdx[..., None] * e_x - dsdu[..., None] * e_u + e_n
        return n / n.norm(dim=-1, keepdim=True)

    def point(self, x_l, u_l):
        s, _, _ = self.sag_and_grad(x_l, u_l)
        e_x, e_u, e_n = self.frame()
        return self.origin + x_l[..., None] * e_x + u_l[..., None] * e_u + s[..., None] * e_n


def intersect(surface, origin, direction, iters=10):
    """Newton solve for ray/surface hit parameter t. origin, direction: (N, 3)."""
    e_x, e_u, e_n = surface.frame()
    rel0 = origin - surface.origin
    n0 = (rel0 * e_n).sum(-1)
    dn = (direction * e_n).sum(-1)
    t = -n0 / dn
    dx = (direction * e_x).sum(-1)
    du = (direction * e_u).sum(-1)
    for _ in range(iters):
        p = origin + t[..., None] * direction
        x_l, u_l, n_l = surface.local(p)
        s, dsdx, dsdu = surface.sag_and_grad(x_l, u_l)
        g = n_l - s
        dgdt = dn - dsdx * dx - dsdu * du
        t = t - g / dgdt
    p = origin + t[..., None] * direction
    x_l, u_l, _ = surface.local(p)
    n = surface.normal(x_l, u_l)
    return t, p, n


def refract_or_reflect(direction, normal, n1, n2):
    """Vector Snell's law air<->glass; returns (out_direction, tir_mask).

    normal is re-oriented to face against `direction`. TIR reflects instead
    of refracting, exactly matching the evaluator's Fresnel-mix shader."""
    cosi_signed = -(direction * normal).sum(-1)
    flip = cosi_signed < 0
    normal = torch.where(flip[..., None], -normal, normal)
    cosi = cosi_signed.abs()
    eta = n1 / n2
    sin2t = eta**2 * (1.0 - cosi**2)
    tir = sin2t > 1.0
    # sqrt(1 - sin2t) has an infinite gradient exactly at sin2t = 1 (grazing
    # exit / the TIR boundary we are deliberately optimising close to), and
    # torch.where still backprops NaNs from the unused branch. Clamp the
    # sqrt's argument away from zero; the refracted output is discarded by
    # `tir` there anyway.
    cost = torch.sqrt(torch.clamp(1.0 - sin2t, min=1e-12))
    refr = eta * direction + (eta * cosi - cost)[..., None] * normal
    refl = direction + 2.0 * cosi[..., None] * normal
    out = torch.where(tir[..., None], refl, refr)
    return out / out.norm(dim=-1, keepdim=True), tir


def mirror_reflect(direction, normal):
    d = direction - 2.0 * (direction * normal).sum(-1, keepdim=True) * normal
    return d / d.norm(dim=-1, keepdim=True)


def incidence_deg(direction, normal):
    c = (-(direction * normal).sum(-1)).abs().clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(c))


@dataclass
class Prism:
    """S1 (eye-facing, used twice), S2 (mirror), S3 (entry towards MLA)."""
    s1: Surface
    s2: Surface
    s3: Surface
    index: float

    def trace(self, origin, direction, iters=10):
        """Pupil -> S1 (refract in) -> S2 (mirror) -> S1 (TIR) -> S3 (refract
        out). Returns dict of intermediate points/normals/angles plus the
        final (point, direction) in air past S3, and a validity mask."""
        n = self.index
        t1, p1, n1 = intersect(self.s1, origin, direction, iters)
        d1, tir1 = refract_or_reflect(direction, n1, 1.0, n)
        inc1 = incidence_deg(direction, n1)

        t2, p2, n2 = intersect(self.s2, p1, d1, iters)
        d2 = mirror_reflect(d1, n2)

        t3, p3, n3 = intersect(self.s1, p2, d2, iters)
        d3, tir3 = refract_or_reflect(d2, n3, n, 1.0)
        inc3 = incidence_deg(d2, n3)

        t4, p4, n4 = intersect(self.s3, p3, d3, iters)
        d4, tir4 = refract_or_reflect(d3, n4, n, 1.0)
        inc4 = incidence_deg(d3, n4)

        valid = (t1 > 0) & (t2 > 0) & (t3 > 0) & (t4 > 0) & (~tir1) & tir3 & (~tir4)
        return {
            "p1": p1, "p2": p2, "p3": p3, "p4": p4,
            "n1": n1, "n2": n2, "n3": n3, "n4": n4,
            "d_out": d4, "p_out": p4,
            "inc1_deg": inc1, "inc3_deg": inc3, "inc4_deg": inc4,
            "tir1": tir1, "tir3": tir3, "tir4": tir4,
            "valid": valid,
        }

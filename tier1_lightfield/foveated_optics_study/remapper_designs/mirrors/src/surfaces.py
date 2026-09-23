"""Off-axis freeform mirror surfaces and a differentiable ray tracer, in torch
float64. Everything is world-frame millimetres, +Y the optical axis away from
the eye, +Z up, +X sideways.

A surface lives in its own local orthonormal frame (V, u, v, w): u is always
the world +X axis (so every surface stays symmetric about the world x = 0
plane when its own polynomial is even in the local x), and (v, w) are the
world (y, z) axes rotated by a single tilt angle `a` about x. The sag is a
base conic plus an XY polynomial, even in local x:

    s(x, y) = c*r^2 / (1 + sqrt(1 - (1+k)*c^2*r^2)) + sum_{i even, j} A[i,j] x^i y^j

World point:  P(x, y) = V + x*u + y*v + s(x, y)*w
World normal: R @ normalize(-ds/dx, -ds/dy, 1),  R = [u v w] (columns)

Ray/surface intersection is Newton's method on the implicit equation
z(t) - s(x(t), y(t)) = 0 along the ray O + t*D, started from the intersection
with the tangent plane at the vertex. Reflection uses the ordinary vector law
d' = d - 2 (d.n) n, which does not depend on the sign of n.
"""
from __future__ import annotations

import torch

DTYPE = torch.float64


def rot_x_frame(tilt):
    """u, v, w for a frame whose x-axis is the world x-axis and whose (v, w)
    are (y, z) rotated by `tilt` (radians, tensor) about x. Right-handed:
    u x v = w."""
    zero, one = torch.zeros_like(tilt), torch.ones_like(tilt)
    u = torch.stack([one, zero, zero])
    v = torch.stack([zero, torch.cos(tilt), torch.sin(tilt)])
    w = torch.stack([zero, -torch.sin(tilt), torch.cos(tilt)])
    return u, v, w


class Freeform:
    """A conic + XY-polynomial freeform surface, even in local x.

    terms: list of (i, j) exponent pairs, i even, i + j >= 2. coeffs: tensor
    matching terms, one per term, each a sag contribution in mm AT THE
    NORMALISED APERTURE EDGE (u = x/u_scale, v = y/v_scale both up to about
    1), i.e. s_poly = sum a_ij u^i v^j. Normalising by the aperture half-size
    keeps every coefficient the same order of magnitude (a few mm) rather
    than spanning mm^{1-i-j} for i+j up to 6, which is what made a single
    Adam learning rate blow up the high-order terms in early tuning.
    """

    def __init__(self, V, tilt, c, k, terms, coeffs, u_scale=1.0, v_scale=1.0):
        self.V, self.tilt, self.c, self.k = V, tilt, c, k
        self.terms = terms
        self.coeffs = coeffs
        self.u_scale, self.v_scale = u_scale, v_scale

    def frame(self):
        u, v, w = rot_x_frame(self.tilt)
        return u, v, w

    def sag(self, x, y):
        r2 = x * x + y * y
        disc = 1.0 - (1.0 + self.k) * self.c * self.c * r2
        disc = torch.clamp(disc, min=1e-6)
        s = self.c * r2 / (1.0 + torch.sqrt(disc))
        u, v = x / self.u_scale, y / self.v_scale
        for (i, j), a in zip(self.terms, self.coeffs):
            s = s + a * u.pow(i) * v.pow(j)
        return s

    def sag_grad(self, x, y):
        """Analytic (ds/dx, ds/dy); matches autograd, checked in tests."""
        r2 = x * x + y * y
        disc = torch.clamp(1.0 - (1.0 + self.k) * self.c * self.c * r2, min=1e-6)
        sq = torch.sqrt(disc)
        denom = 1.0 + sq
        # d/dr2 [c r2/(1+sqrt(1-(1+k)c^2 r2))]
        ddisc_dr2 = -(1.0 + self.k) * self.c * self.c
        dsq_dr2 = 0.5 * ddisc_dr2 / sq
        dsdr2 = self.c / denom - self.c * r2 / denom**2 * dsq_dr2
        dx = 2.0 * x * dsdr2
        dy = 2.0 * y * dsdr2
        u, v = x / self.u_scale, y / self.v_scale
        for (i, j), a in zip(self.terms, self.coeffs):
            if i > 0:
                dx = dx + a * i * u.pow(i - 1) * v.pow(j) / self.u_scale
            if j > 0:
                dy = dy + a * j * u.pow(i) * v.pow(j - 1) / self.v_scale
        return dx, dy

    def local_coords(self, P):
        u, v, w = self.frame()
        d = P - self.V
        return d @ u, d @ v, d @ w

    def to_world(self, x, y, z):
        u, v, w = self.frame()
        return self.V + x[:, None] * u + y[:, None] * v + z[:, None] * w

    def normal_world(self, x, y):
        dx, dy = self.sag_grad(x, y)
        n_local = torch.stack([-dx, -dy, torch.ones_like(dx)], dim=1)
        n_local = n_local / torch.linalg.norm(n_local, dim=1, keepdim=True)
        u, v, w = self.frame()
        R = torch.stack([u, v, w], dim=1)  # (3,3) columns u,v,w
        return n_local @ R.T

    def intersect(self, O, D, n_iter=20, tol=1e-10):
        """Newton solve for t such that O+tD lies on the surface. Returns
        (t, P_world, hit_mask): hit_mask is False where Newton failed to
        converge to `tol` mm (used to exclude/penalise bad rays, never to
        silently accept a wrong point)."""
        u, v, w = self.frame()
        Ou, Ov, Ow = (O - self.V) @ u, (O - self.V) @ v, (O - self.V) @ w
        Du, Dv, Dw = D @ u, D @ v, D @ w
        t = (0.0 - Ow) / Dw  # tangent-plane start
        for _ in range(n_iter):
            x, y, z = Ou + t * Du, Ov + t * Dv, Ow + t * Dw
            s = self.sag(x, y)
            f = z - s
            dx, dy = self.sag_grad(x, y)
            df = Dw - (dx * Du + dy * Dv)
            df = torch.where(df.abs() < 1e-12, torch.full_like(df, 1e-12), df)
            t = t - f / df
        x, y, z = Ou + t * Du, Ov + t * Dv, Ow + t * Dw
        resid = (z - self.sag(x, y)).abs()
        hit = (resid < tol) & (t > 1e-6)
        P = O + t[:, None] * D
        return t, P, hit


def reflect(D, n):
    n = n / torch.linalg.norm(n, dim=1, keepdim=True)
    return D - 2.0 * (D * n).sum(dim=1, keepdim=True) * n

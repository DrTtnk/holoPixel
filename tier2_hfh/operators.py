"""
The forward operator of hogel-free holography, its adjoint, and the normal
operator they generate.

This module exists because three separate pieces of work -- a local/global
solver, temporal subframes, and hardware-state quantisation -- all need the
same operator and would otherwise each rederive it. The adjoint in particular
is load-bearing far beyond its size: a wrong one still produces plausible
images while silently invalidating every gradient and every claim below.

THE OPERATOR

For pupil position k, what the eye sees is the transform of the panel region
the pupil admits:

    P_k = S . F . A . R_k

    R_k   restrict the panel to the window at offset k
    A     multiply by the pupil aperture (real, and currently binary)
    F     unnormalised 2D DFT
    S     fftshift, putting the optical axis at the centre

This is exactly what `optimise.render_views` computes before taking the
squared magnitude, and a test pins the two together: the target light fields
were built against that renderer, so a solver using a different P would be
solving a different problem from the one the targets describe.

THE NORMAL OPERATOR IS DIAGONAL, EXACTLY

    P_k^H P_k = R_k^H A^H F^H S^H S F A R_k

S is a permutation and F is scaled-unitary with F^H F = W^2 I, so the middle
collapses and

    sum_k rho_k P_k^H P_k = W^2 . diag( sum_k rho_k A^2 at offset k )

which is diagonal, not approximately but exactly. The consequence is the whole
reason this module is written: the least-squares step of a local/global solver

    (sum_k rho_k P_k^H P_k) u = sum_k rho_k P_k^H z_k

needs no conjugate gradient, no factorisation and no iteration. It is an
elementwise division.

A WARNING ABOUT WEIGHTS

That result holds for PER-VIEW scalar weights rho_k. It does NOT survive a
per-output-sample weight W_k, because F^H W_k F is circulant rather than
diagonal. Measured on this geometry with a real foveal weight map, half the
operator norm moves off the diagonal. Anything that wants to weight individual
retinal samples has to solve a genuinely non-diagonal system, and `diagonal`
below stops being the answer.

THE NULL SPACE

Some panel pixels are reached by no pupil in the set, either because no window
covers them or because the aperture zeroes them. Their diagonal entry is zero
and they cannot affect any measurement, so `active` marks the rest. Note the
qualifier: dead RELATIVE TO THIS PUPIL SET. A pixel invisible to the current
batch may be visible once the eye moves, so the mask must be recomputed, never
baked in.
"""

import torch


class ViewOperators:
    """`P_k` and `P_k^H` for every pupil position of a `Geometry`."""

    def __init__(self, geom):
        self.geom = geom
        self.offsets = geom.offsets
        self.window = geom.window
        self.panel = geom.panel
        self.aperture = geom.aperture

        # W^2 sum_k A^2, scattered to panel coordinates. Built once: it depends
        # only on the geometry, never on the field.
        self._diagonal = self._scatter_aperture(
            torch.ones(len(self.offsets), dtype=geom.dtype, device=geom.device))

    def _scatter_aperture(self, weights):
        w = self.window
        out = torch.zeros(self.panel, self.panel,
                          dtype=self.geom.dtype, device=self.geom.device)
        square = (w * w) * self.aperture ** 2
        for weight, (oy, ox) in zip(weights.tolist(), self.offsets.tolist()):
            out[oy:oy + w, ox:ox + w] += weight * square
        return out

    def _selected(self, view_idx):
        if view_idx is None:
            return self.offsets
        return self.offsets[view_idx.to(self.offsets.device)]

    def forward(self, u, view_idx=None):
        """Panel field -> (views, W, W) complex fields at the comparison plane."""
        w = self.window
        tiles = torch.stack([u[oy:oy + w, ox:ox + w]
                             for oy, ox in self._selected(view_idx).tolist()])
        return torch.fft.fftshift(torch.fft.fft2(tiles * self.aperture), dim=(-2, -1))

    def adjoint(self, y, view_idx=None):
        """
        (views, W, W) -> panel field, scatter-added over views.

        The inverse of an unnormalised DFT is not its adjoint: torch's `ifft2`
        carries a 1/W^2, so F^H = W^2 . ifft2. Dropping that factor leaves an
        operator that still looks like a back-projection and still produces
        images, and the dot-product test is what catches it.
        """
        w = self.window
        back = (w * w) * torch.fft.ifft2(torch.fft.ifftshift(y, dim=(-2, -1)))
        back = back * self.aperture.conj()
        out = torch.zeros(self.panel, self.panel, dtype=back.dtype, device=back.device)
        for tile, (oy, ox) in zip(back, self._selected(view_idx).tolist()):
            out[oy:oy + w, ox:ox + w] += tile
        return out

    @property
    def diagonal(self):
        """diag(sum_k P_k^H P_k), as a panel-shaped real tensor."""
        return self._diagonal

    def diagonal_for(self, weights):
        """diag(sum_k rho_k P_k^H P_k) for per-view scalar weights."""
        return self._scatter_aperture(weights)

    @property
    def active(self):
        """Panel pixels that reach at least one view. See the null-space note."""
        return self._diagonal > 0

    def solve_normal(self, rhs, weights=None):
        """
        The global least-squares step: u = (sum_k rho_k P_k^H P_k)^-1 rhs.

        An elementwise division, because the normal operator is diagonal. Dead
        pixels have no equation to satisfy and are returned as zero rather than
        as a division by zero -- they cannot affect any measurement either way,
        so the choice is free, and zero is the minimum-norm one.
        """
        d = self.diagonal if weights is None else self.diagonal_for(weights)
        return torch.where(d > 0, rhs / d.to(rhs.dtype), torch.zeros_like(rhs))


def dense_matrix(op, normal=False, weights=None):
    """
    `P` stacked over views, or `sum_k rho_k P_k^H P_k`, as an explicit matrix.

    For validation only, and quadratic in the panel size: this is the reference
    the matrix-free paths are checked against on small geometries, exactly as
    the brief asks. Never call it at a real panel size.
    """
    n = op.panel * op.panel
    eye = torch.eye(n, dtype=torch.complex128, device=op.geom.device)
    columns = []
    for j in range(n):
        e = eye[j].reshape(op.panel, op.panel)
        if not normal:
            columns.append(op.forward(e).reshape(-1))
        else:
            y = op.forward(e)
            if weights is not None:
                y = y * weights.to(y.dtype)[:, None, None]
            columns.append(op.adjoint(y).reshape(-1))
    return torch.stack(columns, dim=1)

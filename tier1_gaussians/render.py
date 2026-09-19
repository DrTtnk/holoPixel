"""
A differentiable Gaussian rasteriser in pure torch.

Built rather than installed. gsplat resolves, but installing it would pull a
torch without the +cu130 tag this machine's sm_120 GPU needs, and it compiles
CUDA kernels at first use. More importantly, every convention it assumes would
be one we could not check, and a silent convention mismatch is the failure
mode that has cost this project the most: it produces plausible numbers rather
than errors.

The maths was verified symbolically before this was written -- the EWA
projection Jacobian, the covariance change of variables, quaternion
orthogonality, and that the front-to-back compositing recursion equals
sum_i c_i a_i prod_(j<i) (1 - a_j). `tests/test_gaussian_render.py` re-pins
each of those numerically, and pins the CONVENTIONS against known geometry:
a point right of the axis must land right of centre, a point above it must
land at a SMALLER row index, and swapping two depths must swap which colour
wins.

Conventions, stated once:
  * the camera looks along +Z in camera space, so depth is positive in front
  * world Y is up; image row indices grow DOWNWARD, matching the path tracer
    in tier1_lightfield/cornell_lightfield.py so the two can be compared
  * quaternions are (w, x, y, z) and are normalised on use
  * Gaussians composite front to back, nearest first

Performance. An earlier version looped over primitives in Python, evaluating
each over its own bounding box. Measured, that cost about 100 microseconds per
Gaussian on the CPU and 435 on the GPU -- and, tellingly, the SAME whether the
splat covered 4 pixels or 21. The cost was per-primitive launch overhead, not
arithmetic, which is why the GPU was four times SLOWER than the CPU: thousands
of tiny kernels, each mostly latency.

So the loop now runs over chunks of primitives instead, compositing a whole
chunk at once with an exclusive cumulative product of (1 - alpha). Each
Gaussian is evaluated over the full frame and masked back to its bounding box,
which does more arithmetic than the per-box version but in a few hundred
kernels rather than a few thousand. Arithmetic is the cheap resource here.

Measured on 4000 Gaussians into a 256 square frame: 1764 ms before, 41 ms
after, a factor of 43, and the per-primitive cost fell from 433 to 10
microseconds.

The same change makes the CPU path 25 times SLOWER -- 399 ms to 10042 ms --
because a CPU has no spare arithmetic to trade for launches, and the dense
evaluation does roughly three thousand times more of it for a four-pixel
splat. That is accepted rather than fixed: light field generation runs on the
GPU, the tests render frames of 32 to 64 pixels where it does not matter, and
carrying a second compositing implementation to serve a case nobody uses would
risk the two drifting apart. If a large CPU render is ever needed, the answer
is tiling, not a second code path.

What this does NOT fix: work grows as the number of Gaussians times the FRAME
area, not times the projected area, so a hundred thousand primitives at 512
squared is out of reach. The backward pass binds sooner still, because autograd
keeps every chunk's intermediates alive, so peak training memory scales the
same way. Tile binning answers all three -- CPU cost, large N, and backward
memory -- and belongs here rather than in a caller.
"""

from dataclasses import dataclass

import numpy as np
import torch

# A Gaussian is evaluated out to this many standard deviations and no further.
# Beyond three sigma it contributes under 1.1% of its peak, and the bound is
# what keeps the cost proportional to projected area rather than to the image.
CUTOFF_SIGMA = 3.0

# Screen-space low-pass dilation, in pixels squared. Kerbl et al. 2023.
DILATION_PX2 = 0.3

# How many elements a single chunk's working tensors may reach. The chunk size
# follows from this and the frame area, so a big frame automatically takes
# fewer primitives per pass and peak memory stays put instead of tracking
# resolution. Nine or so tensors of this size are live at once.
CHUNK_ELEMENTS = 1 << 22


@dataclass
class GaussianScene:
    position: torch.Tensor      # (N, 3) world
    quaternion: torch.Tensor    # (N, 4) as (w, x, y, z)
    scale: torch.Tensor         # (N, 3) standard deviations along local axes
    opacity: torch.Tensor       # (N,) in [0, 1]
    colour: torch.Tensor        # (N, 3)

    def __len__(self):
        return self.position.shape[0]


@dataclass
class Camera:
    width: int
    height: int
    fov_deg: float
    position: torch.Tensor      # (3,)
    look_at: torch.Tensor       # (3,)
    up_hint: tuple = (0.0, 1.0, 0.0)

    @property
    def focal_px(self):
        return 0.5 * self.width / np.tan(np.radians(self.fov_deg) / 2)

    def basis(self):
        """Rows (right, up, forward) mapping world offsets into camera space."""
        fwd = self.look_at - self.position
        fwd = fwd / fwd.norm()
        up_hint = torch.tensor(self.up_hint, dtype=fwd.dtype, device=fwd.device)
        right = torch.linalg.cross(fwd, up_hint)
        right = right / right.norm()
        up = torch.linalg.cross(right, fwd)
        return torch.stack([right, up, fwd])


def quaternion_to_rotation(q):
    """(N, 4) as (w, x, y, z) -> (N, 3, 3). Orthogonal with determinant +1."""
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=1)


def covariance_3d(quaternion, scale):
    """Sigma = R S S^T R^T, symmetric and positive semi-definite by form."""
    R = quaternion_to_rotation(quaternion)
    M = R * scale[:, None, :]
    return M @ M.transpose(1, 2)


def projection_jacobian(p_cam, focal):
    """
    d(u, v)/d(x, y, z) for u = f x / z, v = f y / z, at each camera-space point.
    Derived with sympy before use; the tests check it against finite
    differences of the projection itself, not against this expression.
    """
    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    zero = torch.zeros_like(z)
    return torch.stack([
        torch.stack([focal / z, zero, -focal * x / z ** 2], -1),
        torch.stack([zero, focal / z, -focal * y / z ** 2], -1),
    ], dim=1)


def project_gaussians(scene, camera):
    """
    Returns (mean_px, cov2d, depth, visible).

    `mean_px` is (N, 2) in (column, row); `cov2d` is (N, 2, 2) in pixels;
    `visible` marks Gaussians in front of the camera.
    """
    W = camera.basis().to(scene.position.dtype)
    p_cam = (scene.position - camera.position) @ W.T

    depth = p_cam[:, 2]
    visible = depth > 1e-6
    safe_z = torch.where(visible, depth, torch.ones_like(depth))
    p_safe = torch.stack([p_cam[:, 0], p_cam[:, 1], safe_z], dim=1)

    f = camera.focal_px
    u = f * p_safe[:, 0] / safe_z + (camera.width - 1) / 2
    # World up maps to a smaller row index, matching the path tracer.
    v = -f * p_safe[:, 1] / safe_z + (camera.height - 1) / 2
    mean_px = torch.stack([u, v], dim=1)

    J = projection_jacobian(p_safe, f)
    flip = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=J.dtype, device=J.device)
    JW = (flip @ J) @ W
    cov2d = JW @ covariance_3d(scene.quaternion, scene.scale) @ JW.transpose(1, 2)
    return mean_px, cov2d, depth, visible


def render(scene, camera, background=None):
    """
    Rasterise to an (H, W, 3) image by front-to-back alpha compositing.

    Differentiable in every scene parameter. The depth SORT is not
    differentiable, which is standard and correct: ordering is discrete, and
    gradients flow through the blend given the order. Nor is the bounding box,
    whose edges are floors and ceilings of the projected mean -- also as
    before, when those edges were Python integers.
    """
    H, W = camera.height, camera.width
    dtype, device = scene.position.dtype, scene.position.device
    image = torch.zeros(H, W, 3, dtype=dtype, device=device)
    transmittance = torch.ones(H, W, dtype=dtype, device=device)

    mean_px, cov2d, depth, visible = project_gaussians(scene, camera)
    if not bool(visible.any()):
        return image if background is None else image + background

    # Low-pass dilation, from Kerbl et al. 2023 section 5.1, for the same
    # reason they give: a primitive projecting to less than a pixel cannot be
    # represented on the sampling grid and would invert to an unbounded conic.
    # It guarantees a minimum eigenvalue of 0.3 px^2, so the determinant is
    # always positive. The VALUE is inherited from that paper's datasets and
    # has not been re-derived for this project's focal lengths and depth range.
    cov2d = cov2d + torch.eye(2, dtype=dtype, device=device) * DILATION_PX2

    order = torch.argsort(torch.where(visible, depth, torch.full_like(depth, np.inf)))
    order = order[visible[order]]                      # nearest first, visible only

    S = cov2d[order]
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    inv_xx, inv_xy, inv_yy = S[:, 1, 1] / det, -S[:, 0, 1] / det, S[:, 0, 0] / det

    # The ellipse is bounded by the LARGEST EIGENVALUE, not by the largest
    # diagonal entry. For [[a, b], [b, d]] they coincide only when b = 0, i.e.
    # when the projected ellipse happens to be axis-aligned on screen. A tilted
    # anisotropic Gaussian is the common case in a fitted scene, and using the
    # diagonal clipped it by up to 29%, visible as a straight cut across the
    # blob and as zero gradient outside the box.
    trace = S[:, 0, 0] + S[:, 1, 1]
    lam_max = 0.5 * (trace + torch.sqrt(torch.clamp(trace ** 2 - 4 * det, min=0.0)))
    radius = CUTOFF_SIGMA * torch.sqrt(lam_max)

    mu = mean_px[order]
    colour, opacity = scene.colour[order], scene.opacity[order]
    lo_c, hi_c = torch.floor(mu[:, 0] - radius), torch.ceil(mu[:, 0] + radius)
    lo_r, hi_r = torch.floor(mu[:, 1] - radius), torch.ceil(mu[:, 1] + radius)

    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                            torch.arange(W, device=device, dtype=dtype),
                            indexing="ij")
    col = xx[None]
    row = yy[None]

    n = order.shape[0]
    step = max(1, CHUNK_ELEMENTS // (H * W))
    for start in range(0, n, step):
        s = slice(start, start + step)
        dx = col - mu[s, 0, None, None]
        dy = row - mu[s, 1, None, None]
        power = -0.5 * (inv_xx[s, None, None] * dx * dx
                        + 2 * inv_xy[s, None, None] * dx * dy
                        + inv_yy[s, None, None] * dy * dy)

        # Exactly the old per-primitive bounding box, expressed as a mask: a
        # pixel was included iff its integer coordinate fell within the floored
        # and ceiled extent of the projected mean. Keeping the rectangle rather
        # than switching to a clean elliptical cutoff is deliberate -- it makes
        # this a pure speed change, provable against the old behaviour.
        inside = ((col >= lo_c[s, None, None]) & (col <= hi_c[s, None, None])
                  & (row >= lo_r[s, None, None]) & (row <= hi_r[s, None, None]))
        alpha = opacity[s, None, None] * torch.exp(torch.clamp(power, max=0.0))
        alpha = torch.where(inside, alpha, torch.zeros_like(alpha))

        # Front-to-back compositing, in closed form rather than by recursion:
        # the light reaching primitive k is the product of (1 - alpha) over
        # every nearer one, which is an EXCLUSIVE cumulative product. Built by
        # shifting the inclusive one rather than dividing by alpha, because the
        # division is unbounded wherever a primitive is opaque.
        one_minus = 1 - alpha
        inclusive = torch.cumprod(one_minus, dim=0)
        exclusive = torch.cat([torch.ones_like(inclusive[:1]), inclusive[:-1]], dim=0)

        weight = alpha * exclusive * transmittance[None]
        image = image + torch.einsum("khw,kc->hwc", weight, colour[s])
        transmittance = transmittance * inclusive[-1]

    if background is not None:
        image = image + transmittance[:, :, None] * background
    return image

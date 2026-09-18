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

Performance: this evaluates each Gaussian over a bounding box rather than
binning into tiles, so cost grows as the number of Gaussians times their
projected area. That is ample for scenes of a few thousand primitives and is
NOT the architecture for a hundred thousand; tiling is the fix when it is
needed, and it belongs here rather than in a caller.
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
    gradients flow through the blend given the order.
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
    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                            torch.arange(W, device=device, dtype=dtype),
                            indexing="ij")

    for i in order.tolist():
        if not bool(visible[i]):
            continue
        S = cov2d[i]
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        inv = torch.stack([torch.stack([S[1, 1], -S[0, 1]]),
                           torch.stack([-S[1, 0], S[0, 0]])]) / det

        # The ellipse is bounded by the LARGEST EIGENVALUE, not by the largest
        # diagonal entry. For [[a, b], [b, d]] they coincide only when b = 0,
        # i.e. when the projected ellipse happens to be axis-aligned on screen.
        # A tilted anisotropic Gaussian is the common case in a fitted scene,
        # and using the diagonal clipped it by up to 29%, visible as a straight
        # cut across the blob and as zero gradient outside the box.
        trace = S[0, 0] + S[1, 1]
        lam_max = 0.5 * (trace + torch.sqrt(torch.clamp(trace ** 2 - 4 * det, min=0.0)))
        radius = CUTOFF_SIGMA * torch.sqrt(lam_max)
        c0 = int(max(0, torch.floor(mean_px[i, 0] - radius).item()))
        c1 = int(min(W, torch.ceil(mean_px[i, 0] + radius).item() + 1))
        r0 = int(max(0, torch.floor(mean_px[i, 1] - radius).item()))
        r1 = int(min(H, torch.ceil(mean_px[i, 1] + radius).item() + 1))
        if c0 >= c1 or r0 >= r1:
            continue

        dx = xx[r0:r1, c0:c1] - mean_px[i, 0]
        dy = yy[r0:r1, c0:c1] - mean_px[i, 1]
        power = -0.5 * (inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy
                        + inv[1, 1] * dy * dy)
        alpha = scene.opacity[i] * torch.exp(torch.clamp(power, max=0.0))

        # Slice assignment, NOT Tensor.index_put: the out-of-place form clones
        # the whole image for every primitive, so cost scaled with frame area
        # rather than with the Gaussian's footprint -- measured at 6.07 ms per
        # call against 0.008 ms for this, a factor of 755.
        # The patches are CLONED before the write. Autograd saves the views it
        # needs for the backward pass, and overwriting a slice in place would
        # invalidate the very view the blend depends on. Cloning a bounding box
        # is cheap; cloning the frame, which index_put did, was not.
        patch_image = image[r0:r1, c0:c1].clone()
        patch_T = transmittance[r0:r1, c0:c1].clone()
        image[r0:r1, c0:c1] = patch_image + \
            (patch_T * alpha)[:, :, None] * scene.colour[i]
        transmittance[r0:r1, c0:c1] = patch_T * (1 - alpha)

    if background is not None:
        image = image + transmittance[:, :, None] * background
    return image

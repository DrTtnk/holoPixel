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

Performance, in three stages, each measured.

The first version looped over primitives in Python, each over its own bounding
box. That cost 435 microseconds per Gaussian on the GPU and -- the diagnostic
detail -- the SAME whether the splat covered 4 pixels or 21. A cost that
ignores the work being done is launch overhead, not arithmetic, which is also
why the GPU was 4.3 times SLOWER than the CPU: thousands of tiny kernels, each
mostly latency.

The second compositted a chunk of primitives at once over the whole frame.
That fixed the launches (41 ms for 4000 primitives at 256 squared, a factor of
43) and broke the scaling, because work became the primitive count times the
FRAME area instead of the projected area. On the CPU, which has no spare
arithmetic to trade for launches, it was 25 times slower than the loop.

This version buckets primitives into the tiles their bounding box touches, so
work follows projected area again while still running a few kernels per pass.
For 4000 primitives at 256 squared:

    per-primitive loop     1764 ms      435 us per primitive
    chunked dense            41 ms       10 us
    tiled                   3.9 ms     0.98 us       450x over the loop

and it scales into territory neither earlier version could reach:

    100000 primitives, 512 squared      37.7 ms per view    10.9 s per 289
    500000 primitives, 512 squared     179.0 ms per view    51.7 s per 289

Padding is what tiling trades away: a tile's primitive list is padded to the
longest list in the frame, measured at 1.4 to 1.9 times the true number of
tile-primitive pairs. `tile_assignment` is public so that ratio can be checked
rather than assumed.

The CPU is no longer the casualty it was under the dense version -- 848 ms for
that same 4000-primitive frame against 399 for the original loop -- but the
GPU is the target and the CPU is not optimised for.

Still outstanding: autograd keeps every chunk's intermediates alive, so peak
memory during a backward pass grows with the padded pair count rather than
with the frame. That binds long before forward rendering does.
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

# Side of a square tile, in pixels. Primitives are bucketed into the tiles
# their bounding box touches, so work follows projected area rather than frame
# area. Smaller tiles waste less on primitives that barely overlap them and
# cost more bookkeeping; 16 is the usual choice and the tests prove the value
# cannot change the image.
TILE = 16

# How many elements a single chunk's working tensors may reach. The chunk is
# taken along the per-tile depth-slot axis, so peak memory stays put whatever
# the frame size or the primitive count. Around nine tensors of this size are
# live at once.
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


def _exclusive_cumsum(counts):
    """[0, c0, c0+c1, ...], the start offset of each run."""
    out = torch.zeros_like(counts)
    out[1:] = torch.cumsum(counts, 0)[:-1]
    return out


def tile_assignment(scene, camera):
    """
    Which primitives each tile must composite, and in what order.

    Returns `(slots, n_tiles, k_max)`, where `slots` is an
    `(n_tiles, k_max)` table of indices into the depth-sorted primitive list,
    padded with -1. Column k of a row holds that tile's k-th nearest
    primitive, so a cumulative product along the columns is exactly the
    front-to-back transmittance for that tile.

    Exposed rather than hidden because the padding ratio is the whole
    performance story: `n_tiles * k_max` against the true number of
    tile-primitive pairs says how much the bucketing is wasting.
    """
    H, W = camera.height, camera.width
    n_tx, n_ty = -(-W // TILE), -(-H // TILE)
    n_tiles = n_tx * n_ty
    device = scene.position.device

    mean_px, cov2d, depth, visible = project_gaussians(scene, camera)
    cov2d = cov2d + torch.eye(2, dtype=cov2d.dtype, device=device) * DILATION_PX2
    order = torch.argsort(torch.where(visible, depth, torch.full_like(depth, np.inf)))
    order = order[visible[order]]
    if order.numel() == 0:
        return torch.full((n_tiles, 0), -1, dtype=torch.long, device=device), n_tiles, 0

    S, mu = cov2d[order], mean_px[order]
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    trace = S[:, 0, 0] + S[:, 1, 1]
    lam_max = 0.5 * (trace + torch.sqrt(torch.clamp(trace ** 2 - 4 * det, min=0.0)))
    radius = CUTOFF_SIGMA * torch.sqrt(lam_max)

    # The same pixel box as before, now reduced to the tiles it touches.
    lo_c = torch.clamp(torch.floor(mu[:, 0] - radius), 0, W - 1).long() // TILE
    hi_c = torch.clamp(torch.ceil(mu[:, 0] + radius), 0, W - 1).long() // TILE
    lo_r = torch.clamp(torch.floor(mu[:, 1] - radius), 0, H - 1).long() // TILE
    hi_r = torch.clamp(torch.ceil(mu[:, 1] + radius), 0, H - 1).long() // TILE

    # A primitive whose box misses the frame entirely still clamps to an edge
    # tile; the per-pixel mask in `render` discards it there, so it costs a
    # slot and never a wrong pixel.
    span_x, span_y = hi_c - lo_c + 1, hi_r - lo_r + 1
    counts = span_x * span_y
    total = int(counts.sum())

    # Expand each primitive into one pair per tile it touches, without a loop:
    # repeat its index `counts` times, then turn the position within that run
    # into a 2D offset in its own tile box.
    primitive = torch.repeat_interleave(torch.arange(counts.shape[0], device=device), counts)
    within = torch.arange(total, device=device) - _exclusive_cumsum(counts)[primitive]
    tile_x = lo_c[primitive] + within % span_x[primitive]
    tile_y = lo_r[primitive] + within // span_x[primitive]
    tile = tile_y * n_tx + tile_x

    # Primitives are already in depth order, so a STABLE sort on the tile alone
    # leaves each tile's own list sorted by depth. Sorting on a combined key
    # would need the primitive count to fit inside it; this does not.
    tile, perm = torch.sort(tile, stable=True)
    primitive = primitive[perm]

    per_tile = torch.bincount(tile, minlength=n_tiles)
    k_max = int(per_tile.max())
    slot = torch.arange(total, device=device) - _exclusive_cumsum(per_tile)[tile]

    slots = torch.full((n_tiles, k_max), -1, dtype=torch.long, device=device)
    slots[tile, slot] = primitive
    return slots, n_tiles, k_max


def render(scene, camera, background=None):
    """
    Rasterise to an (H, W, 3) image by front-to-back alpha compositing.

    Differentiable in every scene parameter. The depth SORT is not
    differentiable, which is standard and correct: ordering is discrete, and
    gradients flow through the blend given the order. Neither is the tile
    assignment, nor the bounding box, whose edges are floors and ceilings of
    the projected mean -- as before, when those edges were Python integers.
    """
    H, W = camera.height, camera.width
    dtype, device = scene.position.dtype, scene.position.device
    image = torch.zeros(H, W, 3, dtype=dtype, device=device)

    slots, n_tiles, k_max = tile_assignment(scene, camera)
    if k_max == 0:
        return image if background is None else image + background

    mean_px, cov2d, depth, visible = project_gaussians(scene, camera)
    cov2d = cov2d + torch.eye(2, dtype=dtype, device=device) * DILATION_PX2
    order = torch.argsort(torch.where(visible, depth, torch.full_like(depth, np.inf)))
    order = order[visible[order]]

    S, mu = cov2d[order], mean_px[order]
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    inv_xx, inv_xy, inv_yy = S[:, 1, 1] / det, -S[:, 0, 1] / det, S[:, 0, 0] / det
    trace = S[:, 0, 0] + S[:, 1, 1]
    lam_max = 0.5 * (trace + torch.sqrt(torch.clamp(trace ** 2 - 4 * det, min=0.0)))
    radius = CUTOFF_SIGMA * torch.sqrt(lam_max)
    lo_c, hi_c = torch.floor(mu[:, 0] - radius), torch.ceil(mu[:, 0] + radius)
    lo_r, hi_r = torch.floor(mu[:, 1] - radius), torch.ceil(mu[:, 1] + radius)
    colour, opacity = scene.colour[order], scene.opacity[order]

    # Absolute pixel coordinate of every sample of every tile.
    n_tx = -(-W // TILE)
    t_index = torch.arange(n_tiles, device=device)
    local = torch.arange(TILE, device=device, dtype=dtype)
    col = ((t_index % n_tx) * TILE)[:, None, None] + local[None, None, :]
    row = ((t_index // n_tx) * TILE)[:, None, None] + local[None, :, None]

    tiles = torch.zeros(n_tiles, TILE, TILE, 3, dtype=dtype, device=device)
    transmittance = torch.ones(n_tiles, TILE, TILE, dtype=dtype, device=device)

    step = max(1, CHUNK_ELEMENTS // (n_tiles * TILE * TILE))
    for start in range(0, k_max, step):
        g = slots[:, start:start + step]                  # (n_tiles, C)
        live = g >= 0
        safe = torch.where(live, g, torch.zeros_like(g))

        c, r = col[:, None], row[:, None]                 # (n_tiles, 1, T, T)
        dx = c - mu[safe, 0][:, :, None, None]
        dy = r - mu[safe, 1][:, :, None, None]
        power = -0.5 * (inv_xx[safe][:, :, None, None] * dx * dx
                        + 2 * inv_xy[safe][:, :, None, None] * dx * dy
                        + inv_yy[safe][:, :, None, None] * dy * dy)

        # Exactly the old per-primitive rectangle, and additionally the frame
        # edge, since the last tile of a row or column runs past it.
        inside = ((c >= lo_c[safe][:, :, None, None]) & (c <= hi_c[safe][:, :, None, None])
                  & (r >= lo_r[safe][:, :, None, None]) & (r <= hi_r[safe][:, :, None, None])
                  & live[:, :, None, None] & (c <= W - 1) & (r <= H - 1))
        alpha = opacity[safe][:, :, None, None] * torch.exp(torch.clamp(power, max=0.0))
        alpha = torch.where(inside, alpha, torch.zeros_like(alpha))

        one_minus = 1 - alpha
        inclusive = torch.cumprod(one_minus, dim=1)
        exclusive = torch.cat([torch.ones_like(inclusive[:, :1]), inclusive[:, :-1]], dim=1)

        weight = alpha * exclusive * transmittance[:, None]
        tiles = tiles + torch.einsum("nkij,nkc->nijc", weight, colour[safe])
        transmittance = transmittance * inclusive[:, -1]

    # Lay the tiles back out, then crop away the padding the last row and
    # column of tiles carry when the frame is not a whole number of tiles.
    n_ty = -(-H // TILE)
    grid = tiles.reshape(n_ty, n_tx, TILE, TILE, 3).permute(0, 2, 1, 3, 4)
    image = grid.reshape(n_ty * TILE, n_tx * TILE, 3)[:H, :W]

    if background is not None:
        trans = transmittance.reshape(n_ty, n_tx, TILE, TILE).permute(0, 2, 1, 3)
        trans = trans.reshape(n_ty * TILE, n_tx * TILE)[:H, :W]
        image = image + trans[:, :, None] * background
    return image

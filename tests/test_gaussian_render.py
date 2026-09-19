"""
A differentiable Gaussian rasteriser, to replace the path tracer as a source
of light fields.

Why build rather than install: gsplat would reinstall torch without the +cu130
tag this machine's sm_120 GPU needs, and every convention it assumes would be
one we could not check. The conventions are the danger here -- camera looking
down +Z or -Z, row or column vectors, depth sorted front-to-back or back-to-
front -- so each is pinned by a test against geometry whose answer is known in
advance rather than by comparison with a reference implementation.

The maths itself was verified symbolically before any of this was written
(scratchpad, and restated in the tests below): the EWA projection Jacobian,
the covariance change of variables, quaternion orthogonality, and that the
compositing recursion equals the product form.
"""

import numpy as np
import pytest
import torch

from tier1_gaussians.render import (
    Camera,
    GaussianScene,
    covariance_3d,
    project_gaussians,
    projection_jacobian,
    quaternion_to_rotation,
    render,
)

IDENTITY_Q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)


def a_camera(width=64, height=64, fov_deg=30.0, position=(0.0, 0.0, 0.0),
             look_at=(0.0, 0.0, -1.0)):
    """
    Looks along -Z, matching the path tracer's camera. That convention is not
    cosmetic: with right = fwd x up, a camera looking along +Z puts world +X on
    the LEFT, because in a right-handed frame with +Y up and +Z forward, +X
    points left. Matching the path tracer keeps the two renderers comparable.
    """
    return Camera(width=width, height=height, fov_deg=fov_deg,
                  position=torch.tensor(position, dtype=torch.float64),
                  look_at=torch.tensor(look_at, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Rotation. Verified symbolically; pinned numerically here.
# ---------------------------------------------------------------------------

def test_identity_quaternion_gives_the_identity_rotation():
    R = quaternion_to_rotation(IDENTITY_Q[None])[0]
    assert torch.abs(R - torch.eye(3, dtype=torch.float64)).max() < 1e-15


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotations_are_orthogonal_with_unit_determinant(seed):
    rng = np.random.default_rng(seed)
    q = torch.from_numpy(rng.standard_normal((64, 4)))
    q = q / q.norm(dim=1, keepdim=True)
    R = quaternion_to_rotation(q)

    eye = torch.eye(3, dtype=torch.float64).expand(64, 3, 3)
    assert torch.abs(R.transpose(1, 2) @ R - eye).max() < 1e-12
    assert torch.abs(torch.linalg.det(R) - 1.0).max() < 1e-12


def test_a_quarter_turn_about_z_sends_x_to_y():
    s = np.sqrt(0.5)
    q = torch.tensor([[s, 0.0, 0.0, s]], dtype=torch.float64)
    out = quaternion_to_rotation(q)[0] @ torch.tensor([1.0, 0, 0], dtype=torch.float64)
    assert torch.abs(out - torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)).max() < 1e-12


# ---------------------------------------------------------------------------
# Covariance.
# ---------------------------------------------------------------------------

def test_covariance_is_symmetric_and_positive_semidefinite():
    rng = np.random.default_rng(3)
    q = torch.from_numpy(rng.standard_normal((32, 4)))
    q = q / q.norm(dim=1, keepdim=True)
    scale = torch.from_numpy(rng.uniform(0.01, 0.5, (32, 3)))

    S = covariance_3d(q, scale)

    assert torch.abs(S - S.transpose(1, 2)).max() < 1e-14
    assert torch.linalg.eigvalsh(S).min() > -1e-14


def test_an_isotropic_gaussian_is_unchanged_by_rotation():
    """Rotation cannot matter when all three scales are equal."""
    rng = np.random.default_rng(4)
    q = torch.from_numpy(rng.standard_normal((16, 4)))
    q = q / q.norm(dim=1, keepdim=True)
    scale = torch.full((16, 3), 0.3, dtype=torch.float64)

    S = covariance_3d(q, scale)
    expected = 0.09 * torch.eye(3, dtype=torch.float64).expand(16, 3, 3)

    assert torch.abs(S - expected).max() < 1e-13


# ---------------------------------------------------------------------------
# The Jacobian. This is the derivation CLAUDE.md singles out, so it is checked
# against finite differences of the actual projection, not against itself.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [5, 6])
def test_the_projection_jacobian_matches_finite_differences(seed):
    rng = np.random.default_rng(seed)
    cam = a_camera()
    fx = cam.focal_px

    for _ in range(40):
        p = torch.tensor(rng.uniform(-0.4, 0.4, 3), dtype=torch.float64)
        p[2] = rng.uniform(1.0, 4.0)      # camera-space depth, positive in front

        J = projection_jacobian(p[None], fx)[0]

        h = 1e-6
        numeric = torch.zeros(2, 3, dtype=torch.float64)
        for k in range(3):
            step = torch.zeros(3, dtype=torch.float64)
            step[k] = h
            plus = fx * (p + step)[:2] / (p + step)[2]
            minus = fx * (p - step)[:2] / (p - step)[2]
            numeric[:, k] = (plus - minus) / (2 * h)

        assert torch.abs(J - numeric).max() < 1e-5


# ---------------------------------------------------------------------------
# Projection geometry: known input, known pixel.
# ---------------------------------------------------------------------------

def test_a_gaussian_on_the_optical_axis_lands_at_the_image_centre():
    cam = a_camera(width=65, height=65)
    scene = GaussianScene(
        position=torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float64),
        quaternion=IDENTITY_Q[None], scale=torch.full((1, 3), 0.05, dtype=torch.float64),
        opacity=torch.ones(1, dtype=torch.float64),
        colour=torch.ones(1, 3, dtype=torch.float64))

    mean, _, depth, _ = project_gaussians(scene, cam)

    assert torch.abs(mean[0] - torch.tensor([32.0, 32.0], dtype=torch.float64)).max() < 1e-9
    assert depth[0].item() == pytest.approx(2.0)


def test_moving_a_gaussian_right_moves_it_right_in_the_image():
    """Pins the handedness. A sign error here mirrors every view."""
    cam = a_camera(width=65, height=65)

    def column_of(x):
        scene = GaussianScene(
            position=torch.tensor([[x, 0.0, -2.0]], dtype=torch.float64),
            quaternion=IDENTITY_Q[None], scale=torch.full((1, 3), 0.05, dtype=torch.float64),
            opacity=torch.ones(1, dtype=torch.float64),
            colour=torch.ones(1, 3, dtype=torch.float64))
        return project_gaussians(scene, cam)[0][0, 0].item()

    assert column_of(+0.2) > column_of(0.0) > column_of(-0.2)


def test_moving_a_gaussian_up_moves_it_up_in_the_image():
    """Row indices grow downwards, so up in the world is a SMALLER row."""
    cam = a_camera(width=65, height=65)

    def row_of(y):
        scene = GaussianScene(
            position=torch.tensor([[0.0, y, -2.0]], dtype=torch.float64),
            quaternion=IDENTITY_Q[None], scale=torch.full((1, 3), 0.05, dtype=torch.float64),
            opacity=torch.ones(1, dtype=torch.float64),
            colour=torch.ones(1, 3, dtype=torch.float64))
        return project_gaussians(scene, cam)[0][0, 1].item()

    assert row_of(+0.2) < row_of(0.0) < row_of(-0.2)


def test_an_isotropic_gaussian_on_axis_projects_to_a_circle():
    cam = a_camera()
    scene = GaussianScene(
        position=torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float64),
        quaternion=IDENTITY_Q[None], scale=torch.full((1, 3), 0.1, dtype=torch.float64),
        opacity=torch.ones(1, dtype=torch.float64),
        colour=torch.ones(1, 3, dtype=torch.float64))

    _, cov2d, _, _ = project_gaussians(scene, cam)

    assert cov2d[0, 0, 1].abs().item() < 1e-12
    assert cov2d[0, 0, 0].item() == pytest.approx(cov2d[0, 1, 1].item(), rel=1e-12)


def test_a_gaussian_twice_as_far_projects_half_as_wide():
    cam = a_camera()

    def width(z):   # z is camera-space depth; the camera looks along -Z
        scene = GaussianScene(
            position=torch.tensor([[0.0, 0.0, -z]], dtype=torch.float64),
            quaternion=IDENTITY_Q[None], scale=torch.full((1, 3), 0.1, dtype=torch.float64),
            opacity=torch.ones(1, dtype=torch.float64),
            colour=torch.ones(1, 3, dtype=torch.float64))
        return project_gaussians(scene, cam)[1][0, 0, 0].sqrt().item()

    assert width(2.0) / width(4.0) == pytest.approx(2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Rasterising and compositing.
# ---------------------------------------------------------------------------

def test_one_gaussian_renders_a_blob_at_its_projected_position():
    cam = a_camera(width=65, height=65)
    scene = GaussianScene(
        position=torch.tensor([[0.15, 0.0, -2.0]], dtype=torch.float64),
        quaternion=IDENTITY_Q[None], scale=torch.full((1, 3), 0.06, dtype=torch.float64),
        opacity=torch.ones(1, dtype=torch.float64),
        colour=torch.ones(1, 3, dtype=torch.float64))

    img = render(scene, cam)
    mean = project_gaussians(scene, cam)[0][0]
    peak = torch.argmax(img.sum(dim=2))

    assert abs((peak % 65).item() - mean[0].item()) <= 1
    assert abs((peak // 65).item() - mean[1].item()) <= 1


def test_an_opaque_gaussian_hides_what_is_behind_it():
    cam = a_camera(width=33, height=33)
    scene = GaussianScene(
        position=torch.tensor([[0.0, 0.0, -1.5], [0.0, 0.0, -3.0]], dtype=torch.float64),
        quaternion=IDENTITY_Q.expand(2, 4).clone(),
        scale=torch.full((2, 3), 0.25, dtype=torch.float64),
        opacity=torch.tensor([1.0, 1.0], dtype=torch.float64),
        colour=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64))

    centre = render(scene, cam)[16, 16]

    assert centre[0] > 0.9            # the near, red one
    assert centre[1] < 0.05           # the far, green one is hidden


def test_swapping_depths_swaps_which_colour_wins():
    """Pins the sort direction. Back-to-front would invert this."""
    cam = a_camera(width=33, height=33)

    def centre_colour(z_red, z_green):
        scene = GaussianScene(
            position=torch.tensor([[0.0, 0.0, -z_red], [0.0, 0.0, -z_green]],
                                  dtype=torch.float64),
            quaternion=IDENTITY_Q.expand(2, 4).clone(),
            scale=torch.full((2, 3), 0.25, dtype=torch.float64),
            opacity=torch.tensor([1.0, 1.0], dtype=torch.float64),
            colour=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64))
        return render(scene, cam)[16, 16]

    near_red = centre_colour(1.5, 3.0)
    near_green = centre_colour(3.0, 1.5)

    assert near_red[0] > near_red[1]
    assert near_green[1] > near_green[0]


def test_a_half_transparent_gaussian_lets_half_the_background_through():
    cam = a_camera(width=33, height=33)
    scene = GaussianScene(
        position=torch.tensor([[0.0, 0.0, -1.5], [0.0, 0.0, -3.0]], dtype=torch.float64),
        quaternion=IDENTITY_Q.expand(2, 4).clone(),
        scale=torch.full((2, 3), 0.3, dtype=torch.float64),
        opacity=torch.tensor([0.5, 1.0], dtype=torch.float64),
        colour=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64))

    centre = render(scene, cam)[16, 16]

    assert centre[0] == pytest.approx(0.5, abs=0.02)
    assert centre[1] == pytest.approx(0.5, abs=0.05)


def test_gaussians_behind_the_camera_are_dropped():
    cam = a_camera(width=33, height=33)
    scene = GaussianScene(
        position=torch.tensor([[0.0, 0.0, +2.0]], dtype=torch.float64),
        quaternion=IDENTITY_Q[None], scale=torch.full((1, 3), 0.2, dtype=torch.float64),
        opacity=torch.ones(1, dtype=torch.float64),
        colour=torch.ones(1, 3, dtype=torch.float64))

    assert render(scene, cam).abs().max().item() == 0.0


def test_render_is_differentiable_in_every_parameter():
    """It has to be, for the analytic warm-start route and for fitting."""
    cam = a_camera(width=33, height=33)
    rng = np.random.default_rng(7)
    scene = GaussianScene(
        position=torch.tensor([[0.05, -0.05, -2.0]], dtype=torch.float64, requires_grad=True),
        quaternion=torch.tensor([[0.92, 0.13, -0.29, 0.22]], dtype=torch.float64,
                                requires_grad=True),
        # Anisotropic on purpose: an isotropic Gaussian is rotation-invariant,
        # as test_an_isotropic_gaussian_is_unchanged_by_rotation asserts, so its
        # quaternion gradient is correctly zero and would prove nothing here.
        scale=torch.tensor([[0.22, 0.09, 0.14]], dtype=torch.float64, requires_grad=True),
        opacity=torch.tensor([0.7], dtype=torch.float64, requires_grad=True),
        colour=torch.rand(1, 3, dtype=torch.float64, requires_grad=True))

    render(scene, cam).sum().backward()

    for name in ("position", "quaternion", "scale", "opacity", "colour"):
        g = getattr(scene, name).grad
        assert g is not None, name
        assert torch.isfinite(g).all(), name
        assert g.abs().max() > 0, name


# ---------------------------------------------------------------------------
# Tilted, anisotropic Gaussians. Every test above uses either an isotropic
# Gaussian or the identity quaternion, which is exactly why two bugs survived
# the first suite: the screen-space ellipse is axis-aligned in both cases, and
# both bugs only appear when it is not.
# ---------------------------------------------------------------------------

def _tilted_anisotropic(angle_rad, position=(0.0, 0.0, -2.0)):
    """A cigar-shaped Gaussian rotated about the camera's viewing axis."""
    c, s = np.cos(angle_rad / 2), np.sin(angle_rad / 2)
    return GaussianScene(
        position=torch.tensor([position], dtype=torch.float64),
        quaternion=torch.tensor([[c, 0.0, 0.0, s]], dtype=torch.float64),
        scale=torch.tensor([[0.15, 0.015, 0.015]], dtype=torch.float64),
        opacity=torch.ones(1, dtype=torch.float64),
        colour=torch.ones(1, 3, dtype=torch.float64))


@pytest.mark.parametrize("degrees,sign", [(45.0, -1), (-45.0, +1)])
def test_a_tilted_gaussian_tilts_the_right_way(degrees, sign):
    """
    Pins the row-flip convention, which nothing else does. Negating one row of
    the Jacobian leaves both DIAGONAL covariance entries unchanged -- they are
    squared quantities -- and flips only the cross term. So deleting the flip
    passed all twenty original tests while silently mirroring every tilted
    ellipse.
    """
    cam = a_camera(width=129, height=129)
    _, cov2d, _, _ = project_gaussians(_tilted_anisotropic(np.radians(degrees)), cam)

    assert np.sign(cov2d[0, 0, 1].item()) == sign
    assert abs(cov2d[0, 0, 1].item()) > 0.3 * cov2d[0, 0, 0].item()


def test_the_bounding_box_covers_a_tilted_gaussian_to_three_sigma():
    """
    The box must be sized by the largest EIGENVALUE of the 2D covariance, not
    the largest diagonal entry. They agree only for an axis-aligned ellipse;
    at 45 degrees the diagonal under-covers by 29%, which clips the blob with
    a straight edge and zeroes the gradient outside it.
    """
    cam = a_camera(width=257, height=257)
    _, cov2d, _, _ = project_gaussians(_tilted_anisotropic(np.radians(45.0)), cam)
    S = (cov2d[0] + torch.eye(2, dtype=torch.float64) * 0.3).numpy()

    lam_max = np.linalg.eigvalsh(S).max()
    naive = max(S[0, 0], S[1, 1])

    assert lam_max > naive * 1.5
    assert 3 * np.sqrt(lam_max) > 3 * np.sqrt(naive) * 1.25


def test_a_tilted_gaussian_fades_smoothly_instead_of_being_cut_off():
    """
    The visible symptom of an undersized box: intensity along the major axis
    should decay smoothly, not drop to exactly zero at a straight edge. Sample
    along the ellipse's long direction and require no cliff.
    """
    cam = a_camera(width=257, height=257)
    scene = _tilted_anisotropic(np.radians(45.0))
    img = render(scene, cam).sum(dim=2).numpy()

    centre = 128
    walk = np.array([img[centre - k, centre + k] for k in range(1, centre)])

    # Walk out along the major axis until the render returns exact zero. The
    # property is that the last non-zero sample is already faint: a clipped
    # box leaves a bright sample immediately followed by nothing.
    nonzero = np.nonzero(walk > 0)[0]
    assert len(nonzero) > 10, "the blob should extend well beyond a few pixels"
    last = nonzero.max()

    assert walk[last] < walk.max() * 0.02, (
        f"cliff at the box edge: last lit sample is {walk[last]/walk.max():.3f} "
        "of the peak, so the bounding box is cutting the Gaussian short")

    # and the decay must be monotone, with no step jumping by more than half
    ratios = walk[1:last + 1] / np.maximum(walk[:last], 1e-300)
    assert ratios.max() < 1.001, "intensity should not rise walking outwards"
    assert ratios.min() > 0.5, "no single step should halve abruptly"


def test_rendering_stays_differentiable_after_the_in_place_write():
    """Slice assignment replaced index_put for speed; gradients must survive."""
    cam = a_camera(width=65, height=65)
    scene = _tilted_anisotropic(np.radians(30.0))
    scene.position.requires_grad_(True)
    scene.scale.requires_grad_(True)

    render(scene, cam).sum().backward()

    for t in (scene.position, scene.scale):
        assert t.grad is not None and torch.isfinite(t.grad).all()
        assert t.grad.abs().max() > 0


# ---------------------------------------------------------------------------
# The rasteriser composites a chunk at a time with a cumulative product rather
# than one primitive at a time. That is a pure speed change, so it has to agree
# with the compositing definition written out longhand.
# ---------------------------------------------------------------------------

def _composite_longhand(scene, camera):
    """
    sum_i c_i a_i prod_(j<i) (1 - a_j), evaluated with an explicit Python loop
    over primitives in depth order. Deliberately naive and deliberately not
    sharing code with `render`, so agreement means something.
    """
    from tier1_gaussians.render import (CUTOFF_SIGMA, DILATION_PX2,
                                        project_gaussians)
    H, W = camera.height, camera.width
    mean_px, cov2d, depth, visible = project_gaussians(scene, camera)
    cov2d = cov2d + torch.eye(2, dtype=cov2d.dtype) * DILATION_PX2

    image = np.zeros((H, W, 3))
    trans = np.ones((H, W))
    order = np.argsort(np.where(visible.numpy(), depth.detach().numpy(), np.inf))

    for i in order:
        if not bool(visible[i]):
            continue
        S = cov2d[i].detach().numpy()
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        lam_max = np.linalg.eigvalsh(S).max()
        radius = CUTOFF_SIGMA * np.sqrt(lam_max)
        u, v = mean_px[i].detach().numpy()

        cols = np.arange(W)[None, :]
        rows = np.arange(H)[:, None]
        dx, dy = cols - u, rows - v
        power = -0.5 * (inv[0, 0] * dx ** 2 + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy ** 2)
        inside = ((cols >= np.floor(u - radius)) & (cols <= np.ceil(u + radius))
                  & (rows >= np.floor(v - radius)) & (rows <= np.ceil(v + radius)))
        alpha = np.where(inside, float(scene.opacity[i]) * np.exp(np.minimum(power, 0.0)), 0.0)

        image += (trans * alpha)[:, :, None] * scene.colour[i].detach().numpy()
        trans = trans * (1 - alpha)
    return image, trans


@pytest.mark.parametrize("n", [1, 5, 40])
def test_chunked_compositing_equals_the_longhand_sum(n):
    g = torch.Generator().manual_seed(7 + n)
    r = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    scene = GaussianScene(
        position=(r(n, 3) - 0.5) * 3 - torch.tensor([0.0, 0.0, 6.0], dtype=torch.float64),
        quaternion=r(n, 4) - 0.5,
        scale=0.05 * (0.5 + r(n, 3)),
        opacity=0.2 + 0.6 * r(n),
        colour=r(n, 3))
    cam = a_camera(width=48, height=48)

    got = render(scene, cam).detach().numpy()
    want, _ = _composite_longhand(scene, cam)
    # Guard against the test passing by rendering nothing: this camera looks
    # along -Z, and an earlier version of this scene sat behind it, so the
    # comparison was between two blank images and could not fail.
    assert got.max() > 0.05, "the scene must actually be in front of the camera"
    assert np.allclose(got, want, atol=1e-12), f"max diff {np.abs(got - want).max():.3e}"


def test_chunking_does_not_change_the_result():
    """The chunk size is a memory knob; it must not be a correctness knob."""
    import tier1_gaussians.render as R
    g = torch.Generator().manual_seed(3)
    r = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)
    n = 30
    scene = GaussianScene(
        position=(r(n, 3) - 0.5) * 3 - torch.tensor([0.0, 0.0, 6.0], dtype=torch.float64),
        quaternion=r(n, 4) - 0.5, scale=0.06 * (0.5 + r(n, 3)),
        opacity=0.2 + 0.6 * r(n), colour=r(n, 3))
    cam = a_camera(width=40, height=40)

    original = R.CHUNK_ELEMENTS
    try:
        R.CHUNK_ELEMENTS = 1 << 22
        big = render(scene, cam)
        R.CHUNK_ELEMENTS = 40 * 40            # exactly one primitive per chunk
        one = render(scene, cam)
        R.CHUNK_ELEMENTS = 40 * 40 * 7
        seven = render(scene, cam)
    finally:
        R.CHUNK_ELEMENTS = original

    assert float(big.max()) > 0.05, "the scene must actually be in front of the camera"
    assert torch.allclose(big, one, atol=1e-12)
    assert torch.allclose(big, seven, atol=1e-12)


def test_an_opaque_near_primitive_still_blocks_across_a_chunk_boundary():
    """
    The running transmittance has to survive between chunks. If it did not,
    a blocker in chunk 0 would stop hiding things in chunk 1.
    """
    import tier1_gaussians.render as R
    cam = a_camera(width=32, height=32)
    scene = GaussianScene(
        position=torch.tensor([[0.0, 0.0, -3.0], [0.0, 0.0, -6.0]], dtype=torch.float64),
        quaternion=torch.tensor([[1.0, 0, 0, 0]] * 2, dtype=torch.float64),
        scale=torch.full((2, 3), 0.4, dtype=torch.float64),
        opacity=torch.tensor([1.0, 1.0], dtype=torch.float64),
        colour=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64))

    original = R.CHUNK_ELEMENTS
    try:
        R.CHUNK_ELEMENTS = 32 * 32            # one primitive per chunk: they split
        split = render(scene, cam)
        R.CHUNK_ELEMENTS = 1 << 22            # both in one chunk
        together = render(scene, cam)
    finally:
        R.CHUNK_ELEMENTS = original

    assert torch.allclose(split, together, atol=1e-12)
    centre = split[16, 16]
    assert float(centre[0]) > 0.9 and float(centre[1]) < 0.05, "the near red one must win"

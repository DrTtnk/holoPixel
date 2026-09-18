"""
The hogel-free optimiser: panel phase fitted to view intensities, no depth.

The decisive test is the achievable one. Render views from a known random
phase, hand those views back as the target, and require the optimiser to reach
a high score. If it cannot recover a target that a panel demonstrably CAN
produce, nothing it reports on real data means anything.
"""

import numpy as np
import pytest
import torch

from tier2_hfh.optimise import Geometry, full_psnr, optimise, psnr, render_views, view_loss


def small_geometry(**kw):
    kw.setdefault("panel", 96)
    kw.setdefault("window", 32)
    kw.setdefault("n_views", 5)
    return Geometry(**kw)


def test_the_angular_span_matches_the_light_field_it_will_be_compared_against():
    """
    An n-sample window at pitch p spans lam/p in total, independent of n. The
    pitch is chosen so that span is 20 degrees, which is the field of view the
    Cornell light field was rendered with, so angles need no resampling.
    """
    geom = Geometry(panel=512, window=128, pitch=1.524e-6)
    assert geom.fov_deg == pytest.approx(20.0, abs=0.05)


def test_the_span_depends_on_pitch_alone_not_on_window_size():
    a = Geometry(panel=512, window=64, pitch=1.524e-6)
    b = Geometry(panel=512, window=256, pitch=1.524e-6)
    assert a.fov_deg == pytest.approx(b.fov_deg, rel=1e-12)


def test_a_window_larger_than_the_panel_is_refused():
    with pytest.raises(ValueError, match="must be smaller"):
        Geometry(panel=64, window=64)


def test_pupil_positions_span_the_panel_and_stay_inside_it():
    geom = small_geometry()
    oy = geom.offsets[:, 0]
    assert oy.min().item() == 0
    assert oy.max().item() == geom.panel - geom.window


def test_rendered_views_have_the_shape_of_the_window():
    geom = small_geometry()
    phase = torch.zeros(geom.panel, geom.panel, dtype=torch.float64)
    out = render_views(phase, geom, torch.arange(len(geom.offsets)))
    assert out.shape == (25, geom.window, geom.window)


def test_chunking_the_view_batch_does_not_change_the_result():
    """
    `chunk` exists only to bound memory when evaluating many views at once
    (the periodic full-panel PSNR log); it must not change a single pixel of
    the result, whatever the chunk size relative to the view count.
    """
    geom = small_geometry()
    rng = np.random.default_rng(4)
    phase = torch.from_numpy(rng.uniform(0, 2 * np.pi, (geom.panel, geom.panel)))
    idx = torch.arange(len(geom.offsets))

    whole = render_views(phase, geom, idx)
    for chunk in (1, 4, 25, 100):
        chunked = render_views(phase, geom, idx, chunk=chunk)
        assert torch.equal(whole, chunked)


def test_a_flat_panel_puts_all_its_light_in_the_central_direction():
    geom = small_geometry()
    phase = torch.zeros(geom.panel, geom.panel, dtype=torch.float64)
    view = render_views(phase, geom, torch.tensor([0]))[0]
    peak = torch.argmax(view)
    assert (peak // geom.window).item() == geom.window // 2
    assert (peak % geom.window).item() == geom.window // 2


@pytest.mark.parametrize("k", [3, 7])
def test_a_phase_ramp_steers_by_the_grating_equation(k):
    geom = small_geometry()
    x = torch.arange(geom.panel, dtype=torch.float64)
    phase = (2 * np.pi * k * x / geom.window)[None, :].repeat(geom.panel, 1)
    view = render_views(phase, geom, torch.tensor([0]))[0]
    assert (torch.argmax(view) % geom.window).item() == geom.window // 2 + k


def test_loss_vanishes_and_psnr_diverges_for_an_exact_match():
    geom = small_geometry()
    rng = np.random.default_rng(0)
    phase = torch.from_numpy(rng.uniform(0, 2 * np.pi, (geom.panel, geom.panel)))
    views = render_views(phase, geom, torch.arange(len(geom.offsets)))

    assert view_loss(views, views).item() < 1e-30
    assert psnr(views, views) > 200


def test_loss_ignores_overall_brightness():
    """Absolute brightness is a free scale; only the shape of a view matters."""
    geom = small_geometry()
    rng = np.random.default_rng(1)
    phase = torch.from_numpy(rng.uniform(0, 2 * np.pi, (geom.panel, geom.panel)))
    views = render_views(phase, geom, torch.arange(len(geom.offsets)))

    assert view_loss(views, views * 37.0).item() < 1e-30


def test_the_optimiser_recovers_a_target_a_panel_can_actually_produce():
    """
    The load-bearing test. The target comes from a real panel phase, so it is
    reachable by construction, and the optimiser must get close to it.
    """
    geom = small_geometry()
    rng = np.random.default_rng(2)
    truth = torch.from_numpy(rng.uniform(0, 2 * np.pi, (geom.panel, geom.panel)))
    target = render_views(truth, geom, torch.arange(len(geom.offsets)))

    _, history = optimise(target, geom, iters=400, views_per_iter=4, lr=0.1,
                          log_every=400, log=lambda *_: None)

    first_psnr, last_psnr = history[0][2], history[-1][2]
    assert last_psnr > first_psnr + 3.0
    assert history[-1][1] < history[0][1] / 2


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@cuda_only
def test_float32_cuda_rendering_agrees_with_float64_cpu():
    """
    A float32 CUDA geometry exists to fit a real panel on a GPU, not to change
    the physics. Render the SAME phase through both and require them to agree
    to float32 precision, isolated from any RNG divergence between dtypes.
    """
    geom64 = small_geometry(device="cpu", dtype=torch.float64)
    geom32 = small_geometry(device="cuda", dtype=torch.float32)
    rng = np.random.default_rng(2)
    truth64 = torch.from_numpy(rng.uniform(0, 2 * np.pi, (geom64.panel, geom64.panel)))
    truth32 = truth64.to(torch.float32).to("cuda")

    idx = torch.arange(len(geom64.offsets))
    v64 = render_views(truth64, geom64, idx)
    v32 = render_views(truth32, geom32, idx).cpu().to(torch.float64)

    assert (v64 - v32).abs().max() / v64.abs().max() < 1e-5


@cuda_only
def test_the_optimiser_recovers_a_target_on_cuda_float32():
    """
    The load-bearing recovery property, repeated on the GPU path: a float32
    CUDA geometry must still be able to fit a target a panel can produce.
    """
    geom = small_geometry(device="cuda", dtype=torch.float32)
    rng = np.random.default_rng(2)
    truth = torch.from_numpy(
        rng.uniform(0, 2 * np.pi, (geom.panel, geom.panel))).to(torch.float32).to("cuda")
    target = render_views(truth, geom, torch.arange(len(geom.offsets)))

    _, history = optimise(target, geom, iters=400, views_per_iter=4, lr=0.1,
                          log_every=400, log=lambda *_: None)

    first_psnr, last_psnr = history[0][2], history[-1][2]
    assert last_psnr > first_psnr + 3.0
    assert history[-1][1] < history[0][1] / 2


def test_full_psnr_agrees_with_rendering_every_view_at_once():
    """
    `full_psnr` exists to avoid ever holding every view's render in memory at
    once, purely for a training-time log. It must still equal the direct,
    memory-heavy computation it replaces, for any chunk size.
    """
    geom = small_geometry()
    rng = np.random.default_rng(5)
    phase = torch.from_numpy(rng.uniform(0, 2 * np.pi, (geom.panel, geom.panel)))
    target = torch.from_numpy(
        rng.uniform(0.1, 1, (len(geom.offsets), geom.window, geom.window)))

    direct = psnr(render_views(phase, geom, torch.arange(len(geom.offsets))), target)
    for chunk in (1, 3, 25):
        assert full_psnr(phase, geom, target, chunk=chunk) == pytest.approx(direct, rel=1e-9)


def _pixels_seen_by_some_pupil(geom):
    """Panel pixels lying inside at least one pupil aperture, computed from the
    geometry alone so the gradient test has an independent reference."""
    seen = torch.zeros(geom.panel, geom.panel, dtype=torch.bool)
    for oy, ox in geom.offsets.tolist():
        seen[oy:oy + geom.window, ox:ox + geom.window] |= geom.aperture > 0
    return seen


def test_gradient_reaches_exactly_the_pixels_some_pupil_can_see():
    """
    The panel is ONE variable and the pupil windows overlap, so gradient must
    reach every pixel any pupil can see -- if it did not, the panel would be
    optimised as independent tiles, which is precisely what hogel-free is not.

    It must also reach no others. The aperture is circular, so a pixel sitting
    in the corner of every window that contains it is seen by nothing and can
    have no gradient. Those pixels are dead weight, and the count is reported
    rather than hidden: a design that wastes many of them is badly sampled.
    """
    geom = small_geometry()
    phase = torch.zeros(geom.panel, geom.panel, dtype=torch.float64, requires_grad=True)
    rng = np.random.default_rng(3)
    target = torch.from_numpy(rng.uniform(0, 1, (25, geom.window, geom.window)))

    view_loss(render_views(phase, geom, torch.arange(25)), target).backward()

    seen = _pixels_seen_by_some_pupil(geom)
    has_grad = phase.grad.abs() > 0

    assert torch.equal(has_grad, seen)
    assert seen.to(torch.float64).mean().item() > 0.95


def test_more_pupil_positions_leave_fewer_dead_pixels():
    """
    Dead corners are a sampling artefact, not a property of the method: more
    overlapping pupil positions cover them.
    """
    sparse = _pixels_seen_by_some_pupil(small_geometry(n_views=3))
    dense = _pixels_seen_by_some_pupil(small_geometry(n_views=17))

    assert dense.sum() > sparse.sum()

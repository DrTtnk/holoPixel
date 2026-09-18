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

from tier2_hfh.optimise import Geometry, full_psnr, optimise, psnr, quantised_field, render_views, view_loss
from tier2_slfh.quantization_aware import hard_quantize, realistic_levels, uniform_levels


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


# ======================================================================
# Quantisation: the real device is 8 levels, not a continuum, and each
# level's amplitude differs. `quantised_field` folds that in; `levels=None`
# must leave every test above untouched.
# ======================================================================

def test_quantised_field_defaults_to_the_continuous_panel():
    """`levels=None` is the default precisely so nothing above changes."""
    phase = torch.from_numpy(np.random.default_rng(0).uniform(0, 2 * np.pi, (16, 16)))
    assert torch.equal(quantised_field(phase), torch.exp(1j * phase))


def test_quantised_field_forward_matches_hard_quantize():
    """
    The STE forward value must be the exact quantised field -- the same
    value `tier2_slfh.quantization_aware.hard_quantize` computes, reused
    here rather than re-derived, since only the gradient path differs.
    """
    levels = uniform_levels(8)
    phase = torch.tensor(
        np.random.default_rng(1).uniform(0, 2 * np.pi, 2000),
        dtype=torch.float32, device=levels.phase.device)

    got = quantised_field(phase, levels)
    want = hard_quantize(phase, levels)

    assert torch.allclose(got, want, atol=1e-6)


def test_quantised_field_uses_the_non_uniform_realistic_amplitude():
    """
    The whole point of folding in the real device: amplitude must vary
    across levels for the realistic level set (measured reflectance
    0.583-0.946), unlike the idealised uniform level set (amplitude 1 always).
    """
    levels = realistic_levels(8)
    phase = torch.linspace(0, 2 * np.pi, 4000, dtype=torch.float32)

    field = quantised_field(phase, levels)
    assert field.abs().std().item() > 0.01

    uniform = quantised_field(phase, uniform_levels(8))
    assert torch.allclose(uniform.abs(), torch.ones_like(uniform.abs()), atol=1e-6)


def test_quantised_field_gradient_is_nonzero_and_finite():
    """
    The straight-through estimator must let a genuine, non-degenerate
    gradient flow through the free phase.

    A pointwise loss like sum_i |field_i|^2 is the WRONG probe here: for the
    STE, each pixel's amplitude is a hard (zero-autograd-gradient) function
    of phase, so per-pixel |field_i|^2 is exactly phase-invariant by
    construction (verified: probing tier2_slfh.quantization_aware's own
    ste_quantize the same way gives gradients of order 1e-8, pure
    floating-point round-off, not signal). The gradient that actually
    matters, and that `optimise` relies on, is against an FFT-interference
    loss that depends on RELATIVE phase between pixels -- exactly what a
    real hologram's target intensity does. Summing power in one non-DC
    frequency bin (not every bin: Parseval's theorem makes the full-bin sum
    collapse back to the same phase-invariant total power) is such a loss.
    """
    levels = uniform_levels(8)
    phase = torch.nn.Parameter(torch.linspace(0, 2 * np.pi, 64, dtype=torch.float32))

    field = quantised_field(phase, levels)
    loss = (torch.fft.fft(field).abs() ** 2)[1:8].sum()
    loss.backward()

    assert phase.grad is not None
    assert torch.all(torch.isfinite(phase.grad))
    assert phase.grad.abs().max().item() > 1e-3


def test_render_views_with_levels_emits_only_achievable_amplitudes():
    """
    Rendering through a quantised panel must only ever pass on light through
    one of the level set's real amplitudes -- checked by reconstructing the
    field directly and comparing pixel-for-pixel with what render_views used,
    not just by shape.
    """
    geom = small_geometry(dtype=torch.float32)
    levels = uniform_levels(4)
    rng = np.random.default_rng(6)
    phase = torch.from_numpy(
        rng.uniform(0, 2 * np.pi, (geom.panel, geom.panel))).to(torch.float32)

    idx = torch.arange(len(geom.offsets))
    got = render_views(phase, geom, idx, levels=levels)

    field = quantised_field(phase, levels)
    tiles = torch.stack([
        field[oy:oy + geom.window, ox:ox + geom.window]
        for oy, ox in geom.offsets[idx].tolist()
    ])
    spectra = torch.fft.fftshift(torch.fft.fft2(tiles * geom.aperture), dim=(-2, -1))
    want = torch.abs(spectra) ** 2

    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)


def test_optimise_with_levels_still_reduces_loss_and_reaches_a_reachable_target():
    """
    Quantisation-aware training's load-bearing property, mirrored from
    test_the_optimiser_recovers_a_target_a_panel_can_actually_produce: the
    target is rendered through the SAME quantised (levels=...) forward model,
    so it is reachable by an 8-level panel by construction, and training with
    `levels` set must recover it -- not merely match the unquantised case.
    """
    geom = small_geometry()
    levels = uniform_levels(8)
    rng = np.random.default_rng(7)
    truth = torch.from_numpy(rng.uniform(0, 2 * np.pi, (geom.panel, geom.panel)))
    target = render_views(truth, geom, torch.arange(len(geom.offsets)), levels=levels)

    _, history = optimise(target, geom, iters=400, views_per_iter=4, lr=0.1,
                          log_every=400, log=lambda *_: None, levels=levels)

    first_psnr, last_psnr = history[0][2], history[-1][2]
    assert last_psnr > first_psnr + 3.0
    assert history[-1][1] < history[0][1] / 2


def test_naive_quantisation_after_continuous_optimisation_costs_amplitude_uniformity():
    """
    Naive (optimise continuously, quantise once at the end) is only exposed
    to the realistic level set's non-uniform amplitude at readout time, never
    during optimisation, so it cannot route around it. Quantising the same
    continuous solution onto the uniform vs. realistic level set must
    therefore not do better with the realistic (lower-amplitude, non-uniform)
    levels than with the idealised uniform one.
    """
    geom = small_geometry()
    rng = np.random.default_rng(8)
    truth = torch.from_numpy(rng.uniform(0, 2 * np.pi, (geom.panel, geom.panel)))
    target = render_views(truth, geom, torch.arange(len(geom.offsets)))

    cont_phase, _ = optimise(target, geom, iters=200, views_per_iter=4, lr=0.1,
                             log_every=200, log=lambda *_: None)

    uniform_psnr = full_psnr(cont_phase, geom, target, chunk=8, levels=uniform_levels(8))
    realistic_psnr = full_psnr(cont_phase, geom, target, chunk=8, levels=realistic_levels(8))

    assert uniform_psnr >= realistic_psnr - 1e-6

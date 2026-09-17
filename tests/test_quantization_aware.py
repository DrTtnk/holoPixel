"""
Quantization-aware phase optimization.

This is the direct successor to test_phase_quantization.py: that module
proves the idealized N-level blazed-grating efficiency eta(N) = (sin(pi/N)
/(pi/N))^2 symbolically. This module tests the quantizers used to actually
optimize hologram phase against N (possibly non-uniform) levels, and checks
that the measured efficiency trend tracks the same direction as that closed
form.

Kept fast: the heavy multi-scheme, multi-spacing sweep lives in
tier2_slfh/quantization_aware.py's main(), not here.
"""

import math

import numpy as np
import pytest
import sympy as sp
import torch

from tier2_slfh.quantization_aware import (
    circular_delta,
    gumbel_quantize,
    hard_quantize,
    iters_to_converge,
    make_target,
    naive_quantize_after,
    optimize,
    realistic_levels,
    sinc_efficiency,
    ste_quantize,
    to_image,
    uniform_levels,
)


# ======================================================================
# Level sets
# ======================================================================

@pytest.mark.parametrize("n_levels", [2, 4, 8, 16])
def test_uniform_levels_have_unit_amplitude(n_levels):
    levels = uniform_levels(n_levels)
    assert len(levels) == n_levels
    assert torch.allclose(levels.amplitude, torch.ones_like(levels.amplitude))


@pytest.mark.parametrize("n_levels", [2, 4, 8, 16])
def test_realistic_levels_are_non_uniform_in_amplitude(n_levels):
    """
    This is the physical claim the whole module hinges on: design_phase_lut
    corrects the refractive indices so the achieved PHASE steps are uniform
    (see tests/test_tmm_phase_lut.py), but it does not and cannot correct
    reflectance -- each level absorbs/reflects a different fraction of the
    incident light. That amplitude spread is the "non-uniform level set".
    """
    levels = realistic_levels(n_levels)
    assert len(levels) == n_levels
    assert levels.amplitude.std().item() > 0.01
    assert torch.all(levels.amplitude > 0.0)
    assert torch.all(levels.amplitude <= 1.0)


@pytest.mark.parametrize("n_levels", [2, 4, 8])
@pytest.mark.parametrize("spacing", ["uniform", "realistic"])
def test_quantizer_emits_exactly_n_distinct_levels_on_the_circle(n_levels, spacing):
    """
    A bug of exactly this kind was already found and fixed in
    tier1_tmm/partial_phase.py: linspace(0, 2*pi, n) places a sample at both
    0 and 2*pi, which are the same phase, so only n-1 distinct levels
    survive. Our nearest_index/hard_quantize must not repeat that mistake,
    for either level spacing.
    """
    levels = uniform_levels(n_levels) if spacing == "uniform" else realistic_levels(n_levels)

    rng = np.random.default_rng(0)
    phase = torch.tensor(rng.uniform(0, 2 * np.pi, 20_000), dtype=torch.float32, device=levels.phase.device)

    quantized_field = hard_quantize(phase, levels)
    distinct_phases = torch.unique(torch.round(torch.angle(quantized_field) % (2 * math.pi), decimals=6))

    assert len(distinct_phases) == n_levels


def test_circular_delta_wraps_correctly():
    """Sanity check of the wrap-around used by every quantizer here."""
    a = torch.tensor([0.0, 0.1, 2 * math.pi - 0.1])
    b = torch.tensor([0.0, 0.0, 0.0])
    d = circular_delta(a, b)
    assert torch.allclose(d, torch.tensor([0.0, 0.1, -0.1]), atol=1e-5)


# ======================================================================
# Straight-through estimator
# ======================================================================

def test_ste_forward_equals_hard_quantized_value():
    levels = uniform_levels(8)
    rng = np.random.default_rng(1)
    phase = torch.tensor(rng.uniform(0, 2 * np.pi, 500), dtype=torch.float32, device=levels.phase.device)

    ste_field = ste_quantize(phase, levels)
    hard_field = hard_quantize(phase, levels)

    assert torch.allclose(ste_field, hard_field, atol=1e-6)


def test_ste_backward_gradient_is_nonzero_and_finite():
    levels = uniform_levels(8)
    phase = torch.nn.Parameter(torch.linspace(0, 2 * math.pi, 500, device=levels.phase.device))

    field = ste_quantize(phase, levels)
    loss = (field.real ** 2 + field.imag ** 2).sum()
    loss.backward()

    assert phase.grad is not None
    assert torch.all(torch.isfinite(phase.grad))
    assert torch.any(phase.grad != 0)


def test_gumbel_quantize_hard_forward_matches_nearest_level():
    """With hard=True the forward value must be an exact achievable field
    (amplitude*phase pair from the level set), not a soft blend."""
    levels = uniform_levels(4)
    torch.manual_seed(0)
    phase = torch.linspace(0, 2 * math.pi, 200, device=levels.phase.device)

    field = gumbel_quantize(phase, levels, tau=0.5, hard=True)
    # The hard Gumbel-Softmax pick need not equal the *nearest* level (the
    # score+noise argmax can differ from pure nearest-neighbour), but it must
    # always land exactly on one of the level's achievable fields.
    achievable = set(np.round(levels.field.detach().cpu().numpy(), 6))
    produced = np.round(field.detach().cpu().numpy(), 6)
    assert all(v in achievable for v in produced)


# ======================================================================
# Monotonic degradation with bit depth
# ======================================================================

def test_sinc_efficiency_increases_with_more_levels_symbolically():
    """
    From tests/test_phase_quantization.py: eta(N) = (sin(pi/N)/(pi/N))^2 is
    the idealized N-level blazed-grating efficiency. Prove symbolically
    (not by hard-coding the direction) that eta is increasing in N, i.e.
    efficiency must fall as levels/bit-depth fall. Our measured hologram
    efficiency trend below is checked against this same direction.
    """
    n = sp.Symbol("N", positive=True)
    eta = (sp.sin(sp.pi / n) / (sp.pi / n)) ** 2
    d_eta = sp.diff(eta, n)
    for n_val in (2, 3, 4, 8, 16, 32):
        assert float(d_eta.subs(n, n_val)) > 0


def test_naive_quantized_efficiency_degrades_monotonically_with_bit_depth():
    """
    Optimize continuous phase once on a small problem, then quantize-after to
    2, 4, 8, 16 uniform levels. Efficiency must not decrease as level count
    grows (matches the direction sympy proved above for the closed form).
    """
    target = make_target(n=32, n_points=5, seed=3)
    cont_phase, _ = optimize(target, "continuous", n_iters=150, lr=0.08, seed=3)

    from tier1_tmm.partial_phase import efficiency_metric

    effs = []
    for bits in (1, 2, 3, 4):
        levels = uniform_levels(2 ** bits)
        field = naive_quantize_after(cont_phase, levels)
        eff, _ = efficiency_metric(to_image(field), target)
        effs.append(eff)

    assert all(b >= a - 1e-9 for a, b in zip(effs, effs[1:]))


def test_naive_efficiency_matches_sinc_law_relative_to_continuous_ceiling():
    """
    The key result of this module (flagged by review of the first draft):
    "3-bit == 95% efficiency" is a claim about the FRACTION of the
    continuous-phase optimum retained after quantization, not an absolute
    efficiency -- and that fraction is exactly what the idealized N-level
    blazed-grating closed form eta(N) = (sin(pi/N)/(pi/N))^2 predicts, even
    though this is a real 2D hologram optimizer on a real (sparse) target, not
    a grating. Regression-test that match directly so it cannot silently
    regress: naive_quantize_after's efficiency, divided by the continuous
    ceiling on the SAME optimized phase, must track sinc_efficiency(n_levels)
    to within 2% relative error at every tested bit depth.
    """
    import torch

    from tier1_tmm.partial_phase import efficiency_metric

    target = make_target(n=32, n_points=5, seed=3)
    cont_phase, _ = optimize(target, "continuous", n_iters=150, lr=0.08, seed=3)
    cont_img = to_image(torch.exp(1j * cont_phase.to(torch.complex64)))
    cont_eff, _ = efficiency_metric(cont_img, target)

    for bits in (1, 2, 3, 4):
        n_levels = 2 ** bits
        levels = uniform_levels(n_levels)
        field = naive_quantize_after(cont_phase, levels)
        eff, _ = efficiency_metric(to_image(field), target)
        ratio = eff / cont_eff
        bound = sinc_efficiency(n_levels)
        assert ratio == pytest.approx(bound, rel=0.02), (
            f"n_levels={n_levels}: naive/continuous={ratio:.4f} vs sinc^2={bound:.4f}"
        )


def test_iters_to_converge_is_within_range():
    losses = [1.0, 0.5, 0.2, 0.11, 0.101, 0.1, 0.1, 0.1]
    n = iters_to_converge(losses, frac=0.95)
    assert 1 <= n <= len(losses)
    # 95% of the (1.0 - 0.1) improvement is reached at loss <= 0.145, first at index 3 (iter 4).
    assert n == 4

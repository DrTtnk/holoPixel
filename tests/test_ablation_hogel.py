"""
Machinery tests for tier2_slfh/ablation_hogel_vs_slfh.py.

These test the plumbing of the hogel-vs-direct-SLFH ablation, not the
science: that the PSNR/SSIM implementations agree with an independent
reference, that hogel tiling+reassembly is a lossless round trip when no
optimisation happens in between, and that the angle<->spatial-frequency
convention the hogel path encodes as an FFT bin index agrees, in the
paraxial limit, with the pupil-shift convention the direct SLFH path uses.
No GPU optimisation loop runs here — everything is seconds, not minutes.
"""

import math

import numpy as np
import pytest
import sympy as sp
import torch
from skimage.metrics import structural_similarity as sk_ssim

from tier2_slfh import ablation_hogel_vs_slfh as ablation
from tier2_slfh import hogel_optimizer
from tier2_slfh import hogel_grid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PSNR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_psnr_matches_closed_form_on_synthetic_noise():
    rng = np.random.default_rng(0)
    a = rng.random((16, 16, 3))
    b = np.clip(a + rng.normal(0.0, 0.05, a.shape), 0.0, 1.0)

    mse = np.mean((a - b) ** 2)
    expected = 10.0 * math.log10(1.0 ** 2 / mse)

    assert ablation.psnr(a, b, data_range=1.0) == pytest.approx(expected, rel=1e-9)


def test_psnr_identical_images_is_infinite():
    a = np.random.default_rng(1).random((8, 8, 3))
    assert ablation.psnr(a, a) == float("inf")


def test_psnr_matches_closed_form_for_known_constant_offset():
    """A known constant error e on every pixel gives mse=e^2 exactly."""
    a = np.full((10, 10, 3), 0.5)
    e = 0.1
    b = np.clip(a + e, 0.0, 1.0)
    expected = 10.0 * math.log10(1.0 / (e ** 2))
    assert ablation.psnr(a, b, data_range=1.0) == pytest.approx(expected, rel=1e-9)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SSIM — cross-checked against the independent skimage implementation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _skimage_reference_ssim(a, b):
    return sk_ssim(a, b, data_range=1.0, channel_axis=2,
                    gaussian_weights=True, sigma=1.5,
                    use_sample_covariance=False)


def test_ssim_identical_images_is_one():
    a = np.random.default_rng(3).random((32, 32, 3)).astype(np.float32)
    assert ablation.ssim(a, a) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("noise_std", [0.0, 0.02, 0.1, 0.3])
def test_ssim_matches_independent_skimage_reference(noise_std):
    rng = np.random.default_rng(4)
    a = rng.random((64, 64, 3)).astype(np.float32)
    b = np.clip(a + rng.normal(0.0, noise_std, a.shape), 0.0, 1.0).astype(np.float32)

    mine = ablation.ssim(a, b)
    reference = _skimage_reference_ssim(a, b)

    assert mine == pytest.approx(reference, abs=0.01)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hogel tiling / reassembly round trip (no optimisation in between)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.mark.parametrize("H,grid_size", [(160, 63), (160, 31), (160, 16), (160, 8),
                                          (37, 9), (100, 100), (5, 3)])
def test_hogel_anchor_indices_round_trip_through_upsample(H, grid_size):
    """
    If a hogel's reconstruction were simply "whatever value it was given"
    (identity instead of an optimiser), tiling into hogels and reassembling
    via nearest-neighbour upsampling must recover that exact value at every
    hogel's own anchor pixel — this is the lossless round trip the ablation
    relies on to make "upsample the coarse hogel grid to compare against the
    full-resolution ground truth" a fair operation, not one that silently
    shifts pixels between hogels.

    np.linspace(0, H-1, grid_size, dtype=int) (the convention hogel_grid.py's
    own main() uses) does NOT round-trip through upsample_nearate's floor-
    division binning when grid_size does not evenly divide H — verified to
    fail below. hogel_anchor_indices() (bin-centre convention) is used
    instead specifically because it round-trips for every grid_size.
    """
    W = H
    rng = np.random.default_rng(0)
    image = rng.random((H, W, 3))

    gy = ablation.hogel_anchor_indices(H, grid_size)
    gx = ablation.hogel_anchor_indices(W, grid_size)
    assert len(set(gy.tolist())) == grid_size  # every hogel gets a distinct pixel
    assert len(set(gx.tolist())) == grid_size

    hogel_values = np.array([[image[py, px] for px in gx] for py in gy])
    reassembled = ablation.upsample_nearest(hogel_values, H, W)

    for py in gy:
        for px in gx:
            assert np.array_equal(reassembled[py, px], image[py, px])


def test_linspace_endpoint_convention_does_not_round_trip_in_general():
    """
    Documents *why* hogel_anchor_indices exists: the naive endpoint-inclusive
    linspace sampling used by hogel_grid.py's own main() breaks the round
    trip for a grid size that does not evenly divide the image size (63 and
    31 hogels across our 160px ablation light field, from N_SUB=64 and 128).
    """
    H, grid_size = 160, 63
    gy = np.linspace(0, H - 1, grid_size, dtype=int)
    row_of_pixel = np.arange(H) * grid_size // H

    mismatches = sum(1 for hy, py in enumerate(gy) if row_of_pixel[py] != hy)
    assert mismatches > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Angle <-> FFT-bin mapping: hogel path self-consistency and agreement with
# the direct SLFH path's own angle/frequency convention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.mark.parametrize("N,pitch,wavelength", [
    (40, 0.706e-6, 532e-9),
    (64, 0.706e-6, 632e-9),
    (128, 0.706e-6, 450e-9),
])
def test_hogel_angle_to_fft_bin_inverts_fftfreq(N, pitch, wavelength):
    """
    hogel_optimizer.angle_to_fft_bin maps a viewing angle to an index into
    fftshift(fft2(...)). Round-tripping bin -> frequency (via the actual
    torch.fft.fftfreq the optimiser's FFT uses) -> angle -> bin must recover
    the original bin exactly, for every bin whose frequency is opticaly
    reachable (|sin(theta)| <= 1).
    """
    freqs = torch.fft.fftshift(torch.fft.fftfreq(N, d=pitch)).numpy()

    checked = 0
    for k in range(N):
        sin_theta = freqs[k] * wavelength
        if abs(sin_theta) > 1.0:
            continue
        theta = math.asin(sin_theta)
        bin_back = hogel_optimizer.angle_to_fft_bin(theta, wavelength, pitch, N)
        assert round(bin_back) == k
        checked += 1

    assert checked > 0  # sanity: the parametrization actually exercised bins


def test_hogel_grid_and_hogel_optimizer_bin_formulas_agree():
    """
    hogel_grid.py's angle_to_fft_bin (module-global N_SUB/PITCH) and
    hogel_optimizer.py's angle_to_fft_bin (explicit N/pitch arguments) must
    be the same formula. The ablation script sweeps hogel size by setting
    hogel_grid's globals, so this equivalence is what makes that safe.
    """
    original_n_sub, original_pitch = hogel_grid.N_SUB, hogel_grid.PITCH
    try:
        hogel_grid.N_SUB = 96
        hogel_grid.PITCH = 0.706e-6
        for theta in (-0.2, -0.01, 0.0, 0.05, 0.18):
            for wavelength in (632e-9, 532e-9, 450e-9):
                b1 = hogel_optimizer.angle_to_fft_bin(theta, wavelength, hogel_grid.PITCH, hogel_grid.N_SUB)
                b2 = hogel_grid.angle_to_fft_bin(theta, wavelength)
                assert b1 == pytest.approx(b2, rel=1e-12)
    finally:
        hogel_grid.N_SUB, hogel_grid.PITCH = original_n_sub, original_pitch


def test_hogel_and_direct_angle_conventions_agree_in_paraxial_limit():
    """
    Symbolic check (sympy), not a hand-derived formula taken on faith.

    Hogel path (hogel_grid.angle_to_fft_bin / hogel_optimizer.angle_to_fft_bin):
    spatial frequency from a viewing angle is the grating equation
        freq = sin(theta) / wavelength

    Direct SLFH path (slfh.circular_aperture): a pupil shifted to physical
    position shift_x at the eyebox plane, seen through a relay lens of focal
    length f, corresponds to a spatial-frequency shift
        sx = shift_x / (wavelength * focal_length)

    These are the same physics iff shift_x/focal_length is the paraxial
    (small-angle) approximation of theta, i.e. iff first-order-in-theta
    sin(theta)/wavelength, with theta substituted by shift_x/focal_length,
    is *exactly* shift_x/(wavelength*focal_length) — not just close.
    """
    theta, wavelength, shift_x, focal_length = sp.symbols(
        "theta wavelength shift_x focal_length", positive=True)

    freq_hogel = sp.sin(theta) / wavelength
    freq_hogel_paraxial = sp.series(freq_hogel, theta, 0, 2).removeO()
    freq_hogel_paraxial = freq_hogel_paraxial.subs(theta, shift_x / focal_length)

    freq_direct = (shift_x / focal_length) / wavelength  # slfh.py's sx, literally

    assert sp.simplify(freq_hogel_paraxial - freq_direct) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-hogel energy correction (undoes batch_optimize's per-hogel target
# normalization, which otherwise erases spatial brightness — see
# docs/notes_hogel_ablation.md)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_per_hogel_energy_correction_sets_each_hogel_sum_to_the_target():
    """The whole point of the correction: regardless of what total energy the
    optimiser's raw output happens to carry, the corrected output's sum over
    viewing angles must equal the true (pre-normalization) target sum for
    every hogel independently."""
    rng = np.random.default_rng(5)
    nv, nu, B = 3, 3, 4
    raw_views = rng.random((nv, nu, B)) + 0.1  # keep strictly positive
    pre_norm_sum = np.array([1.0, 2.0, 0.5, 10.0])

    corrected = ablation.apply_per_hogel_energy_correction(raw_views, pre_norm_sum)

    for b in range(B):
        assert corrected[:, :, b].sum() == pytest.approx(pre_norm_sum[b], rel=1e-9)


def test_per_hogel_energy_correction_is_invariant_to_the_raw_output_scale():
    """This is the property the earlier (multiply-only) version of the
    correction lacked: it must not matter what overall scale the optimiser's
    raw output happens to have converged to (achieved_sum), only its
    relative shape across viewing angles — since the earlier version
    implicitly assumed that scale was always close to n_samples, which
    docs/notes_hogel_ablation.md's convergence-ratio table shows is false
    precisely at the N_SUB values this correction matters most for."""
    rng = np.random.default_rng(6)
    nv, nu, B = 4, 4, 3
    raw_views = rng.random((nv, nu, B)) + 0.1
    pre_norm_sum = np.array([3.0, 7.0, 0.2])
    arbitrary_per_hogel_scale = np.array([0.001, 1000.0, 5.0])

    corrected_a = ablation.apply_per_hogel_energy_correction(raw_views, pre_norm_sum)
    corrected_b = ablation.apply_per_hogel_energy_correction(
        raw_views * arbitrary_per_hogel_scale[None, None, :], pre_norm_sum)

    assert np.allclose(corrected_a, corrected_b, rtol=1e-9)


def test_per_hogel_energy_correction_handles_an_all_zero_hogel():
    """A hogel whose raw output is all zero (achieved_sum=0) must not
    produce NaN/inf — the minimal clamp exists for this, not to hide a real
    error elsewhere."""
    raw_views = np.zeros((2, 2, 1))
    pre_norm_sum = np.array([5.0])
    corrected = ablation.apply_per_hogel_energy_correction(raw_views, pre_norm_sum)
    assert np.all(np.isfinite(corrected))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# grid_size_for(): the conserved-resolution-budget rule the sweep relies on
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_grid_size_matches_baseline_at_baseline_n_sub():
    assert ablation.grid_size_for(ablation.BASELINE_N_SUB) == ablation.GRID_BASE


def test_grid_size_is_monotonically_decreasing_in_n_sub():
    sizes = [ablation.grid_size_for(n) for n in sorted(ablation.N_SUB_SWEEP)]
    assert sizes == sorted(sizes, reverse=True)


def test_grid_size_times_n_sub_is_approximately_conserved():
    """
    HOGEL_SIZE = N_SUB * PITCH (fixed pitch) and GRID_SIZE * HOGEL_SIZE =
    panel width (fixed) together imply GRID_SIZE * N_SUB = const. Rounding
    grid_size to an integer breaks this only slightly.
    """
    products = [ablation.grid_size_for(n) * n for n in ablation.N_SUB_SWEEP]
    baseline = ablation.GRID_BASE * ablation.BASELINE_N_SUB
    for p in products:
        assert p == pytest.approx(baseline, rel=0.15)


def test_grid_size_is_capped_at_max_grid():
    """
    Found by running the angular-matched sweep: at N_SUB=9 the uncapped
    formula wants grid=444 against a 160px light field, which is not a finer
    grid, it is >1 hogel silently sharing each source pixel through
    hogel_anchor_indices. max_grid must cap this.
    """
    uncapped = ablation.grid_size_for(9)
    assert uncapped > 160  # the actual failure this guards against

    capped = ablation.grid_size_for(9, max_grid=160)
    assert capped == 160


def test_grid_size_cap_is_a_noop_when_not_binding():
    for n in ablation.N_SUB_SWEEP:
        assert ablation.grid_size_for(n, max_grid=1_000_000) == ablation.grid_size_for(n)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# representative_view_indices(): the eval-cost fix for the angular-matched
# sweep (found necessary when a full 64x64=4096-view evaluation measured
# 747s for one configuration)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_representative_view_indices_includes_both_extremes_and_centre():
    idx = ablation.representative_view_indices(64, k=5)
    assert idx[0] == 0
    assert idx[-1] == 63
    assert len(idx) == 5
    assert len(set(idx.tolist())) == 5  # distinct


def test_representative_view_indices_caps_at_n_angular():
    idx = ablation.representative_view_indices(3, k=5)
    assert len(idx) == 3
    assert list(idx) == [0, 1, 2]

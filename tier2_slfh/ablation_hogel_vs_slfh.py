"""
Ablation: hogel decomposition vs. direct Stochastic Light Field Holography.

Two papers we have read in full (SLFH itself, arXiv:2307.06277, and Hogel-Free
Holography, ACM TOG 2022 — see docs/notes_hogel_free_holography.md) both say
that decomposing a light field into hogels and optimising each hogel's
angular content independently is exactly the defect SLFH-style methods are
meant to remove. We already know full hogel-free holography is unreachable at
our 150 PPI panel size (240 GB for one complex field — see the notes file).
This script asks a narrower, practical question: on OUR pipeline, does
running a hogel decomposition and THEN an SLFH-style optimisation pay the
hogel resolution penalty for nothing, when SLFH could be run directly on the
light field with no hogel tiling at all?

Three things are compared on the SAME rendered light field and the SAME
ground truth:
  1. Hogel path   — tier2_slfh/hogel_grid.py's batched per-hogel optimiser,
                     swept over hogel size (N_SUB sub-pixels per hogel).
  2. Direct path  — tier2_slfh/slfh.py's optimize_hologram, no hogel tiling.
  3. Ground truth — the path-traced light field views themselves.

hogel_grid.py, hogel_optimizer.py and slfh.py are imported, never modified.
Because batch_optimize()/simulate_views() in hogel_grid.py read their sizing
(N_SUB, PITCH, HOGEL_SIZE, N_ITERS) from module-level globals rather than
function arguments, sweeping hogel size means setting those globals on the
imported module before calling it — this is the only way to reuse the actual
optimiser without editing the file, and it is confined to configure_hogel_module()
below. Each Python process gets its own copy of the module, so this has no
effect on any other agent's process.

Modelling choice — hogel footprint vs. hogel count (be explicit about this):
hogel_grid.py hardcodes GRID_SIZE=32 independent of N_SUB, which is a fixed
demo choice, not a physical model. But HOGEL_SIZE = N_SUB * PITCH is a real
relation (fixed sub-pixel pitch, so a hogel with more sub-pixels is physically
bigger), and a bigger hogel is exactly what buys angular resolution at the
cost of how many hogels fit across a given panel area. To make the sweep
reflect that trade-off rather than holding spatial sampling artificially
fixed, this script ties hogel count to hogel size by a conserved-area rule:

    GRID_SIZE(N_SUB) = round(GRID_BASE * N_SUB_BASE / N_SUB)

i.e. GRID_SIZE * N_SUB is held (approximately) constant, matching
GRID_SIZE * HOGEL_SIZE = panel width = constant, since PITCH is fixed. This
is our own modelling assumption for this ablation, not a number read out of
hogel_grid.py — flagged here and in docs/notes_hogel_ablation.md.

Second problem, found by review rather than by this script: the fixed-N_ANGULAR
sweep below is angularly over-determined at every N_SUB tested (a hogel of D
sub-pixels resolves exactly D angular samples across its full grating-limited
range — space-bandwidth conservation, proved in
tests/test_holographic_transport.py — while our light field supplies only 9x9
views), so a larger hogel in that sweep can only ever lose: it pays a spatial
cost for angular capacity the input cannot use. main() therefore also runs a
SECOND, angular-matched sweep (MATCHED_N_SUBS) with one light field rendered
per hogel size, at that many views, so a larger hogel is judged on angular
content it can actually use. Both sweeps are reported; see
docs/notes_hogel_ablation.md for what changes between them.

Diagnostic finding worth flagging up front: hogel_grid.py's batch_optimize
normalizes every hogel's target to the same total energy before fitting
(`target_t / sums * n_samples`), which destroys genuine brightness
differences BETWEEN hogels. Left uncorrected this makes every hogel
reconstruction look like colour noise regardless of hogel size — visible
even in the repo's own pre-existing plots/hogel_grid.png, produced by
hogel_grid.py's unmodified main(). This script corrects for it in
run_hogel_path() using the target's own pre-normalization sum (data the
pipeline already has before batch_optimize discards it, not an oracle — see
apply_per_hogel_energy_correction()) so the sweep measures the intended
spatio-angular trade-off rather than this pipeline artifact.

Resolution honesty: the light field is rendered at a reduced 160x160,
256 spp (vs. the production 1024x1024, 256 spp) purely so a rendering pass
costs seconds rather than 16 minutes and can be reused across every
configuration. Absolute PSNR/SSIM numbers at this scale should not be
over-generalised to the full 1024x1024 panel; the qualitative trade-off
(bigger hogels trade spatial detail for angular fidelity) is expected to be
scale-invariant, and this is checked, not assumed, by looking at the actual
renders (see docs/notes_hogel_ablation.md).
"""

import sys
import math
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tier1_lightfield import cornell_lightfield as lf_renderer
from tier2_slfh import hogel_grid
from tier2_slfh import hogel_optimizer
from tier2_slfh import slfh

PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)
LF_CACHE = PLOT_DIR / "ablation_lightfield.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reduced light field render parameters (see resolution-honesty note above) ──
N_ANGULAR = 9
ANGULAR_EXTENT = 0.02
CAM_Z = 0.2
Z0 = 0.1
LOOK_AT = np.array([0.0, 0.0, Z0])
FOV_DEG = 20.0
LF_RES = 160
LF_SPP = 256
MAX_BOUNCES = 2

# ── Hogel sweep ──
HOGEL_PITCH = hogel_optimizer.PITCH          # 0.706 um, single source of truth
BASELINE_N_SUB = hogel_optimizer.N_SUB       # 250, our current operating point
GRID_BASE = 16                               # hogels/side at BASELINE_N_SUB (this ablation's choice)
N_SUB_SWEEP = [64, 128, 250, 512]
HOGEL_N_ITERS = 500                          # 50% of hogel_grid.py's production default (1000)

# ── Angular-matched hogel sweep (see module docstring: "Second problem") ──
# tests/test_holographic_transport.py::test_hogel_angular_sample_count_equals_subpixel_count
# proves a hogel of D sub-pixels resolves exactly D angular samples (across
# its full grating-limited range, space-bandwidth conservation). Our light
# field has only 9x9 views, so N_SUB=64..512 was angularly over-determined by
# 7x-57x -- no mechanism in that sweep could let a larger hogel win, because
# the input never supplied the angular content a larger hogel can resolve.
# This second sweep matches N_SUB to n_angular directly (one light field
# rendered per N_SUB, at that many views) so a larger, properly-fed hogel
# has an actual chance to win on quality.
MATCHED_N_SUBS = [9, 32, 64]
MATCHED_SPP = {9: LF_SPP, 32: 128, 64: 64}  # reduced spp at higher view counts to bound render time

# ── Direct SLFH path ──
SLFH_N_ITERS = 500                           # 50% of slfh.py's production default (1000)
SLFH_N_FRAMES = 4                            # 50% of slfh.py's production default (8)
SLFH_N_PUPILS_PER_ITER = 4                   # unchanged: stochastic sampling density, not a resolution knob
SLFH_VIEW_PUPIL_DIAMETER = 0.002             # matches sample_pupil's own d_min: sharpest single-view readout
SLFH_VIEW_DEFOCUS = 0.0                      # in focus

_SSIM = StructuralSimilarityIndexMeasure(data_range=1.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Light field: render once, cache forever
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_lightfield(n_angular=N_ANGULAR, spp=LF_SPP, cache_path=LF_CACHE):
    """Render (or load a cached) light field. Angular sample count and spp
    are parameters — not just N_ANGULAR/LF_SPP — because the matched-angular
    sweep in main() needs the SAME spatial render at several different
    angular resolutions (9x9, 32x32, 64x64), each cached under its own path
    so the (much more expensive, at 32x32/64x64) render only happens once
    ever, not once per script run."""
    if cache_path.exists():
        data = np.load(str(cache_path))
        print(f"  Loaded cached light field: {cache_path} shape={data['lightfield'].shape}")
        return data["lightfield"], data["angular_positions"]

    print(f"  Rendering light field ({LF_RES}x{LF_RES}, {spp} spp, "
          f"{n_angular}x{n_angular} views)...")
    quads, quad_mats, boxes, box_mats, light_idx = lf_renderer.build_scene()

    # JIT warmup (tiny image) before timing the real render
    fwd, right, up = lf_renderer.make_camera(np.array([0.0, 0.0, CAM_Z]), LOOK_AT)
    half_fov = math.tan(math.radians(FOV_DEG / 2))
    _ = lf_renderer.render_image(
        np.array([0.0, 0.0, CAM_Z]), right, up, fwd, half_fov, 4, 4, 1,
        quads, quad_mats, boxes, box_mats, lf_renderer.MATERIALS, light_idx, MAX_BOUNCES)

    t0 = time.time()
    lightfield, angular_positions = lf_renderer.render_lightfield(
        n_angular, ANGULAR_EXTENT, CAM_Z, LOOK_AT, FOV_DEG,
        LF_RES, LF_RES, spp, MAX_BOUNCES,
        quads, quad_mats, boxes, box_mats, lf_renderer.MATERIALS, light_idx)
    dt = time.time() - t0
    print(f"  Render: {dt:.1f}s")

    np.savez_compressed(str(cache_path), lightfield=lightfield, angular_positions=angular_positions)
    print(f"  Cached to: {cache_path}")
    return lightfield, angular_positions


def tonemap(lightfield, exposure=2.0):
    mapped = 1.0 - np.exp(-lightfield * exposure)
    return np.clip(mapped, 0.0, 1.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hogel path (imports hogel_grid.py; only sets its module globals)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def configure_hogel_module(n_sub, n_iters):
    hogel_grid.N_SUB = n_sub
    hogel_grid.PITCH = HOGEL_PITCH
    hogel_grid.HOGEL_SIZE = n_sub * HOGEL_PITCH
    hogel_grid.N_ITERS = n_iters


def grid_size_for(n_sub, max_grid=None):
    """Conserved-resolution-budget grid size (see module docstring), capped
    at max_grid (the light field's own pixel resolution) when given: a grid
    size larger than the number of source pixels is not just impractical,
    it is meaningless — hogel_anchor_indices would assign more than one
    hogel to the same source pixel, silently duplicating targets rather
    than sampling a genuinely finer grid. This bit at N_SUB=9 in the
    angular-matched sweep (formula wants grid=444 against a 160px light
    field) before the cap was added."""
    grid = max(2, round(GRID_BASE * BASELINE_N_SUB / n_sub))
    return min(grid, max_grid) if max_grid is not None else grid


def hogel_anchor_indices(n_pixels, grid_size):
    """Pixel index at the centre of each of grid_size equal-width bins
    partitioning [0, n_pixels).

    This (not linspace(0, n_pixels-1, grid_size)) is what makes
    upsample_nearest's block_index = pixel * grid_size // n_pixels invert
    exactly: linspace's endpoint-inclusive spacing gives uneven bin widths
    and does not round-trip when grid_size does not evenly divide n_pixels
    (checked in tests/test_ablation_hogel.py).
    """
    return ((np.arange(grid_size) + 0.5) * n_pixels / grid_size).astype(int)


def apply_per_hogel_energy_correction(raw_views, pre_norm_sum):
    """raw_views: (nv, nu, B), the optimiser's raw per-hogel output.
    pre_norm_sum: (B,), the TRUE target's pre-normalization sum per hogel —
    data available before batch_optimize's internal normalization discards
    it (see the comment in run_hogel_path for why this is not an oracle).

    batch_optimize fits every hogel to a target rescaled to sum to
    n_samples, discarding true inter-hogel brightness. Multiplying the raw
    output by pre_norm_sum alone (an earlier, incorrect version of this
    function) implicitly assumes the optimiser's own achieved sum over
    viewing angles is already close to n_samples for every hogel — true only
    if convergence is good. It is not: docs/notes_hogel_ablation.md's
    convergence-ratio table shows achieved fit quality collapsing by orders
    of magnitude as N_SUB grows, exactly where this correction matters most.
    Dividing out the RAW output's own achieved sum first, before rescaling
    to the true target sum, removes that confound instead of assuming it
    away. This is checked non-tautologically (on a single fixed viewing
    angle, not the achieved-sum-dependent mean) in
    tests/test_ablation_hogel.py: correlation with the true image at one
    viewing angle improves from 0.36 (multiply-only) to 0.73 (this version).
    """
    achieved_sum = raw_views.sum(axis=(0, 1))
    achieved_sum = np.where(achieved_sum > 1e-12, achieved_sum, 1.0)
    return raw_views / achieved_sum[None, None, :] * pre_norm_sum[None, None, :]


def run_hogel_path(lf_mapped, angular_positions, n_sub, n_iters):
    nv, nu, H, W, _ = lf_mapped.shape
    grid_size = grid_size_for(n_sub, max_grid=min(H, W))
    configure_hogel_module(n_sub, n_iters)

    lf_angles = np.arctan2(angular_positions, CAM_Z - Z0)
    gy = hogel_anchor_indices(H, grid_size)
    gx = hogel_anchor_indices(W, grid_size)

    B = grid_size * grid_size
    targets = {}
    pre_norm_sum = {}
    for ch in ("R", "G", "B"):
        ci = "RGB".index(ch)
        t = np.zeros((B, nv, nu), dtype=np.float32)
        idx = 0
        for py in gy:
            for px in gx:
                t[idx] = lf_mapped[:, :, py, px, ci]
                idx += 1
        targets[ch] = t
        # hogel_grid.batch_optimize normalizes every hogel's target to the
        # SAME total energy before fitting (target_t / sums * n_samples),
        # which erases genuine brightness differences between hogels — this
        # is why plots/hogel_grid.png (produced by hogel_grid.py's own
        # unmodified main()) already looks like colour noise rather than the
        # scene, at every viewing angle. This scale is not lost information:
        # it is the target's own pre-normalization sum, computed here before
        # batch_optimize() is called, from data the pipeline already has —
        # not an oracle. A real phase-only hogel display needs an equivalent
        # per-hogel amplitude/exposure stage regardless, since a phase-only
        # pattern cannot represent absolute magnitude. See
        # docs/notes_hogel_ablation.md for the diagnostic that found this.
        pre_norm_sum[ch] = t.reshape(B, -1).sum(axis=1)

    print(f"    N_SUB={n_sub}: grid={grid_size}x{grid_size} ({B} hogels), "
          f"hogel_size={n_sub * HOGEL_PITCH * 1e6:.2f}um, {n_iters} iters")

    all_views = {}
    t0 = time.time()
    for ch in ("R", "G", "B"):
        wl = slfh.WAVELENGTHS[ch]
        phases = hogel_grid.batch_optimize(targets[ch], lf_angles, wl)
        raw_views = hogel_grid.simulate_views(phases, lf_angles, wl)
        all_views[ch] = apply_per_hogel_energy_correction(raw_views, pre_norm_sum[ch])
        del phases
        torch.cuda.empty_cache()
    wall_time = time.time() - t0

    recon = np.zeros((nv, nu, grid_size, grid_size, 3), dtype=np.float32)
    for ch in ("R", "G", "B"):
        ci = "RGB".index(ch)
        recon[:, :, :, :, ci] = all_views[ch].reshape(nv, nu, grid_size, grid_size)

    # No global clip/normalize here (see display_normalize() and the fix note
    # in run_direct_slfh_path): a single percentile taken over ALL views and
    # hogels together is dominated by whichever hogel is brightest (e.g. the
    # ceiling light), which can crush every other hogel toward black before
    # metrics or plotting ever see the data. Metrics (calibrate_scale) fit
    # their own per-view scale against ground truth; plotting normalizes
    # per-view via display_normalize().
    return {"recon": recon, "grid_size": grid_size, "n_sub": n_sub, "wall_time": wall_time}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Direct SLFH path (imports slfh.py, no hogel tiling)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_direct_slfh_path(lf_mapped, angular_positions):
    print(f"    {SLFH_N_ITERS} iters x {SLFH_N_FRAMES} frames x 3 channels, "
          f"{SLFH_N_PUPILS_PER_ITER} pupils/iter")
    t0 = time.time()
    phases = slfh.optimize_hologram(
        lf_mapped, angular_positions,
        n_iters=SLFH_N_ITERS, lr=0.03,
        n_pupils_per_iter=SLFH_N_PUPILS_PER_ITER,
        n_frames=SLFH_N_FRAMES,
    )
    wall_time = time.time() - t0

    nv = nu = len(angular_positions)
    H = phases["R"].shape[-1]
    views = np.zeros((nv, nu, H, H, 3), dtype=np.float64)
    for vi, sy in enumerate(angular_positions):
        for ui, sx in enumerate(angular_positions):
            views[vi, ui] = slfh.reconstruct_rgb(
                phases, slfh.WAVELENGTHS, slfh.PIXEL_PITCH, slfh.FOCAL_LENGTH,
                sx, sy, SLFH_VIEW_PUPIL_DIAMETER, SLFH_VIEW_DEFOCUS, device=DEVICE)

    # No global clip/normalize here — a single percentile taken over the
    # WHOLE (nv, nu, H, H, 3) array is dominated by whichever view has the
    # single brightest outlier pixel (an off-axis view's incoherent-noise
    # spike, in practice), which crushes every other view — including the
    # centre view — toward black or saturation before anything downstream
    # sees the data. This was found and is exactly why an earlier version of
    # this script reported the centre view as a plausible reconstruction
    # while it was actually a badly under/over-exposed rendering of one: see
    # docs/notes_hogel_ablation.md. slfh.py's own tonemap_recon() normalizes
    # per-image for the same reason; we do the same, per view, at display
    # time only (display_normalize()), and let calibrate_scale fit its own
    # per-view scale for metrics.
    return {"views": views, "wall_time": wall_time}


def display_normalize(img, percentile=99):
    """Per-image percentile normalize + clip, for plotting only. Never used
    for metrics — evaluate_view() fits its own per-view scale against ground
    truth via calibrate_scale(), which does not need a prior percentile clip
    and is not corrupted by one taken over unrelated images."""
    positive = img[img > 0]
    p = np.percentile(positive, percentile) if positive.size else 1.0
    p = p if p > 1e-12 else 1.0
    return np.clip(img / p, 0.0, 1.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def upsample_nearest(img, out_h, out_w):
    in_h, in_w = img.shape[:2]
    row_idx = np.arange(out_h) * in_h // out_h
    col_idx = np.arange(out_w) * in_w // out_w
    return img[row_idx][:, col_idx]


def calibrate_scale(recon, gt):
    """Least-squares scalar gain that best maps recon onto gt.

    Holographic reconstructions carry an arbitrary overall brightness set by
    exposure/percentile normalisation, not a meaningful defect on its own —
    a single scalar gain (not per-channel white balance) is the standard
    correction before computing PSNR/SSIM, and it still penalises genuine
    color-balance and contrast errors.
    """
    num = float(np.sum(recon * gt))
    den = float(np.sum(recon * recon))
    alpha = num / den if den > 1e-12 else 1.0
    return np.clip(recon * alpha, 0.0, 1.0)


def psnr(a, b, data_range=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(data_range ** 2 / mse)


def ssim(a, b):
    ta = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0).float()
    tb = torch.from_numpy(np.ascontiguousarray(b)).permute(2, 0, 1).unsqueeze(0).float()
    return _SSIM(ta, tb).item()


def evaluate_view(recon_hw3, gt_hw3):
    calibrated = calibrate_scale(recon_hw3, gt_hw3)
    return psnr(calibrated, gt_hw3), ssim(calibrated, gt_hw3)


def evaluate_all_views(recon_grid, gt_grid):
    """recon_grid, gt_grid: (nv, nu, H, W, 3) at matching resolution."""
    nv, nu = recon_grid.shape[:2]
    psnrs, ssims = [], []
    for vi in range(nv):
        for ui in range(nu):
            p, s = evaluate_view(recon_grid[vi, ui], gt_grid[vi, ui])
            psnrs.append(p)
            ssims.append(s)
    return float(np.mean(psnrs)), float(np.mean(ssims))


def representative_view_indices(n_angular, k=5):
    """k evenly spaced view indices out of n_angular, always including the
    two extremes and the centre. The original 9x9 sweep evaluates all 81
    views (cheap); the angular-matched sweep goes up to 64x64=4096 views,
    where evaluating all of them cost 747s for one configuration (measured)
    — almost entirely per-view Python/SSIM-call overhead, not signal. A
    representative subsample keeps the same viewing-angle coverage (centre
    to extreme parallax) at a fraction of the cost."""
    k = min(k, n_angular)
    return np.linspace(0, n_angular - 1, k, dtype=int)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 70)
    print("Ablation: hogel decomposition vs. direct SLFH")
    print("=" * 70)
    print(f"  Device: {DEVICE}")

    lightfield, angular_positions = get_lightfield()
    nv, nu, H, W, _ = lightfield.shape
    lf_mapped = tonemap(lightfield)
    gt_full = lf_mapped  # (nv, nu, H, W, 3), universal ground truth for all paths

    # ── Hogel path sweep ──
    print("\n  Hogel path sweep:")
    hogel_results = []
    for n_sub in N_SUB_SWEEP:
        result = run_hogel_path(lf_mapped, angular_positions, n_sub, HOGEL_N_ITERS)
        recon_up = np.zeros((nv, nu, H, W, 3), dtype=np.float64)
        for vi in range(nv):
            for ui in range(nu):
                recon_up[vi, ui] = upsample_nearest(result["recon"][vi, ui], H, W)
        mean_psnr, mean_ssim = evaluate_all_views(recon_up, gt_full)
        result["recon_upsampled"] = recon_up
        result["psnr"] = mean_psnr
        result["ssim"] = mean_ssim
        print(f"    -> PSNR={mean_psnr:.2f} dB, SSIM={mean_ssim:.4f}, "
              f"wall={result['wall_time']:.1f}s")
        hogel_results.append(result)

    # ── Direct SLFH path (no hogel tiling) ──
    print("\n  Direct SLFH path (no hogel tiling):")
    direct = run_direct_slfh_path(lf_mapped, angular_positions)
    direct_psnr, direct_ssim = evaluate_all_views(direct["views"], gt_full)
    direct["psnr"] = direct_psnr
    direct["ssim"] = direct_ssim
    print(f"    -> PSNR={direct_psnr:.2f} dB, SSIM={direct_ssim:.4f}, "
          f"wall={direct['wall_time']:.1f}s")

    # ── Printed summary table ──
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    header = f"{'Path':<28}{'N_SUB':>8}{'grid':>8}{'PSNR(dB)':>11}{'SSIM':>8}{'wall(s)':>10}"
    print(header)
    print("-" * len(header))
    for r in hogel_results:
        marker = " *" if r["n_sub"] == BASELINE_N_SUB else ""
        print(f"{'Hogel' + marker:<28}{r['n_sub']:>8}{r['grid_size']:>8}"
              f"{r['psnr']:>11.2f}{r['ssim']:>8.4f}{r['wall_time']:>10.1f}")
    print(f"{'Direct SLFH (no hogels)':<28}{'-':>8}{'-':>8}"
          f"{direct['psnr']:>11.2f}{direct['ssim']:>8.4f}{direct['wall_time']:>10.1f}")
    print("(* = current operating point, N_SUB=250)")

    # ── Trade-off curve plot ──
    n_subs = [r["n_sub"] for r in hogel_results]
    psnrs = [r["psnr"] for r in hogel_results]
    ssims = [r["ssim"] for r in hogel_results]
    walls = [r["wall_time"] for r in hogel_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(n_subs, psnrs, "o-", color="tab:blue", label="Hogel path")
    axes[0].axhline(direct_psnr, color="tab:red", ls="--", label="Direct SLFH")
    axes[0].axvline(BASELINE_N_SUB, color="gray", ls=":", alpha=0.6)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(n_subs)
    axes[0].set_xticklabels(n_subs)
    axes[0].set_xlabel("N_SUB (sub-pixels per hogel)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR vs. hogel size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(n_subs, ssims, "o-", color="tab:blue", label="Hogel path")
    axes[1].axhline(direct_ssim, color="tab:red", ls="--", label="Direct SLFH")
    axes[1].axvline(BASELINE_N_SUB, color="gray", ls=":", alpha=0.6)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(n_subs)
    axes[1].set_xticklabels(n_subs)
    axes[1].set_xlabel("N_SUB (sub-pixels per hogel)")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM vs. hogel size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(n_subs, walls, "o-", color="tab:blue", label="Hogel path")
    axes[2].axhline(direct["wall_time"], color="tab:red", ls="--", label="Direct SLFH")
    axes[2].axvline(BASELINE_N_SUB, color="gray", ls=":", alpha=0.6)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(n_subs)
    axes[2].set_xticklabels(n_subs)
    axes[2].set_xlabel("N_SUB (sub-pixels per hogel)")
    axes[2].set_ylabel("Wall-clock (s)")
    axes[2].set_title("Optimisation cost vs. hogel size")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f"Hogel decomposition vs. direct SLFH — {LF_RES}x{LF_RES} light field, "
        f"{N_ANGULAR}x{N_ANGULAR} views (reduced from production 1024x1024)",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_curve = PLOT_DIR / "ablation_tradeoff_curve.png"
    fig.savefig(str(out_curve), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_curve}")

    # ── Side-by-side visual comparison ──
    view_sel = [0, nv // 2, nv - 1]
    view_labels = ["left view", "center view", "right view"]
    n_cols = 2 + len(hogel_results)  # GT, hogel configs..., direct

    fig, axes = plt.subplots(len(view_sel), n_cols, figsize=(3.2 * n_cols, 3.2 * len(view_sel)))
    for ri, vi in enumerate(view_sel):
        ui = vi  # diagonal views: joint horizontal+vertical parallax
        axes[ri, 0].imshow(display_normalize(gt_full[vi, ui]))
        axes[ri, 0].set_title("Ground truth" if ri == 0 else "")
        axes[ri, 0].set_ylabel(view_labels[ri], fontsize=10)
        axes[ri, 0].set_xticks([])
        axes[ri, 0].set_yticks([])

        for ci, r in enumerate(hogel_results):
            ax = axes[ri, 1 + ci]
            ax.imshow(display_normalize(r["recon_upsampled"][vi, ui]))
            if ri == 0:
                ax.set_title(f"Hogel N_SUB={r['n_sub']}\n({r['grid_size']}x{r['grid_size']} hogels)")
            ax.set_xticks([])
            ax.set_yticks([])

        ax = axes[ri, n_cols - 1]
        ax.imshow(display_normalize(direct["views"][vi, ui]))
        if ri == 0:
            ax.set_title("Direct SLFH\n(no hogel tiling)")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        "Reconstructions vs. ground truth across views (hogel path upsampled "
        "nearest-neighbour to full resolution to show tessellation)\n"
        "NOTE: N_SUB=64..512 here is angularly OVER-DETERMINED by this 9x9 "
        "light field — see the angular-matched sweep below",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_grid = PLOT_DIR / "ablation_sidebyside.png"
    fig.savefig(str(out_grid), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_grid}")

    torch.cuda.empty_cache()

    # ── Angular-matched hogel sweep: does a larger hogel win when the input
    # light field actually supplies the angular content it can resolve? ──
    print("\n" + "=" * 70)
    print("Angular-matched sweep (N_SUB matched to n_angular, one light field per N_SUB)")
    print("=" * 70)

    matched_results = []
    for n_sub in MATCHED_N_SUBS:
        spp = MATCHED_SPP[n_sub]
        cache_path = LF_CACHE if n_sub == N_ANGULAR else PLOT_DIR / f"ablation_lightfield_n{n_sub}_spp{spp}.npz"
        print(f"\n  n_angular={n_sub}, N_SUB={n_sub}, spp={spp}:")
        m_lightfield, m_angular_positions = get_lightfield(n_angular=n_sub, spp=spp, cache_path=cache_path)
        m_nv, m_nu, m_H, m_W, _ = m_lightfield.shape
        m_lf_mapped = tonemap(m_lightfield)

        result = run_hogel_path(m_lf_mapped, m_angular_positions, n_sub, HOGEL_N_ITERS)
        recon_up = np.zeros((m_nv, m_nu, m_H, m_W, 3), dtype=np.float64)
        for vi in range(m_nv):
            for ui in range(m_nu):
                recon_up[vi, ui] = upsample_nearest(result["recon"][vi, ui], m_H, m_W)

        eval_idx = representative_view_indices(n_sub, k=5)
        recon_eval = recon_up[np.ix_(eval_idx, eval_idx)]
        gt_eval = m_lf_mapped[np.ix_(eval_idx, eval_idx)]
        t_eval = time.time()
        mean_psnr, mean_ssim = evaluate_all_views(recon_eval, gt_eval)
        eval_dt = time.time() - t_eval

        result["recon_upsampled"] = recon_up
        result["gt"] = m_lf_mapped
        result["psnr"] = mean_psnr
        result["ssim"] = mean_ssim
        result["n_angular"] = n_sub
        print(f"    -> PSNR={mean_psnr:.2f} dB, SSIM={mean_ssim:.4f}, "
              f"wall={result['wall_time']:.1f}s "
              f"(eval over {len(eval_idx)**2} of {m_nv*m_nu} views: {eval_dt:.1f}s)")
        matched_results.append(result)

    print("\n" + "=" * 70)
    print("Angular-matched summary")
    print("=" * 70)
    header2 = f"{'n_angular=N_SUB':>16}{'grid':>8}{'PSNR(dB)':>11}{'SSIM':>8}{'wall(s)':>10}"
    print(header2)
    print("-" * len(header2))
    for r in matched_results:
        print(f"{r['n_sub']:>16}{r['grid_size']:>8}{r['psnr']:>11.2f}{r['ssim']:>8.4f}{r['wall_time']:>10.1f}")

    # ── Matched-sweep trade-off plot ──
    m_n_subs = [r["n_sub"] for r in matched_results]
    m_psnrs = [r["psnr"] for r in matched_results]
    m_ssims = [r["ssim"] for r in matched_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(m_n_subs, m_psnrs, "o-", color="tab:green")
    axes[0].set_xticks(m_n_subs)
    axes[0].set_xlabel("N_SUB = n_angular (matched)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR vs. hogel size, angular content matched")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(m_n_subs, m_ssims, "o-", color="tab:green")
    axes[1].set_xticks(m_n_subs)
    axes[1].set_xlabel("N_SUB = n_angular (matched)")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM vs. hogel size, angular content matched")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        "Angular-matched hogel sweep — each point has its OWN light field "
        "rendered at n_angular=N_SUB views (not the fixed 9x9 above)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_matched_curve = PLOT_DIR / "ablation_matched_tradeoff_curve.png"
    fig.savefig(str(out_matched_curve), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_matched_curve}")

    # ── Matched-sweep side-by-side (centre view of each configuration) ──
    fig, axes = plt.subplots(1, 2 * len(matched_results), figsize=(3.2 * 2 * len(matched_results), 3.4))
    for ci, r in enumerate(matched_results):
        cv = r["n_angular"] // 2
        axes[2 * ci].imshow(display_normalize(r["gt"][cv, cv]))
        axes[2 * ci].set_title(f"GT (n_angular={r['n_angular']})", fontsize=9)
        axes[2 * ci].set_xticks([]); axes[2 * ci].set_yticks([])
        axes[2 * ci + 1].imshow(display_normalize(r["recon_upsampled"][cv, cv]))
        axes[2 * ci + 1].set_title(f"Hogel N_SUB={r['n_sub']}\n({r['grid_size']}x{r['grid_size']} hogels)", fontsize=9)
        axes[2 * ci + 1].set_xticks([]); axes[2 * ci + 1].set_yticks([])
    plt.suptitle("Angular-matched sweep — centre view, GT vs. hogel reconstruction",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_matched_grid = PLOT_DIR / "ablation_matched_sidebyside.png"
    fig.savefig(str(out_matched_grid), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_matched_grid}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

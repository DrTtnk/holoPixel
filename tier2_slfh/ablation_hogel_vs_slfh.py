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

def get_lightfield():
    if LF_CACHE.exists():
        data = np.load(str(LF_CACHE))
        print(f"  Loaded cached light field: {LF_CACHE} shape={data['lightfield'].shape}")
        return data["lightfield"], data["angular_positions"]

    print(f"  Rendering light field ({LF_RES}x{LF_RES}, {LF_SPP} spp, "
          f"{N_ANGULAR}x{N_ANGULAR} views)...")
    quads, quad_mats, boxes, box_mats, light_idx = lf_renderer.build_scene()

    # JIT warmup (tiny image) before timing the real render
    fwd, right, up = lf_renderer.make_camera(np.array([0.0, 0.0, CAM_Z]), LOOK_AT)
    half_fov = math.tan(math.radians(FOV_DEG / 2))
    _ = lf_renderer.render_image(
        np.array([0.0, 0.0, CAM_Z]), right, up, fwd, half_fov, 4, 4, 1,
        quads, quad_mats, boxes, box_mats, lf_renderer.MATERIALS, light_idx, MAX_BOUNCES)

    t0 = time.time()
    lightfield, angular_positions = lf_renderer.render_lightfield(
        N_ANGULAR, ANGULAR_EXTENT, CAM_Z, LOOK_AT, FOV_DEG,
        LF_RES, LF_RES, LF_SPP, MAX_BOUNCES,
        quads, quad_mats, boxes, box_mats, lf_renderer.MATERIALS, light_idx)
    dt = time.time() - t0
    print(f"  Render: {dt:.1f}s")

    np.savez_compressed(str(LF_CACHE), lightfield=lightfield, angular_positions=angular_positions)
    print(f"  Cached to: {LF_CACHE}")
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


def grid_size_for(n_sub):
    return max(2, round(GRID_BASE * BASELINE_N_SUB / n_sub))


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
    """raw_views: (nv, nu, B). pre_norm_sum: (B,). Broadcasts a per-hogel
    scalar gain over the (nv, nu) viewing-angle axes — see the comment in
    run_hogel_path for why this correction is applied and why it is not an
    oracle."""
    return raw_views * pre_norm_sum[None, None, :]


def run_hogel_path(lf_mapped, angular_positions, n_sub, n_iters):
    nv, nu, H, W, _ = lf_mapped.shape
    grid_size = grid_size_for(n_sub)
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

    p99 = np.percentile(recon[recon > 0], 99) if (recon > 0).any() else 1.0
    recon = np.clip(recon / p99, 0.0, 1.0)

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

    p99 = np.percentile(views[views > 0], 99) if (views > 0).any() else 1.0
    views = np.clip(views / p99, 0.0, 1.0)

    return {"views": views, "wall_time": wall_time}


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
        axes[ri, 0].imshow(gt_full[vi, ui])
        axes[ri, 0].set_title("Ground truth" if ri == 0 else "")
        axes[ri, 0].set_ylabel(view_labels[ri], fontsize=10)
        axes[ri, 0].set_xticks([])
        axes[ri, 0].set_yticks([])

        for ci, r in enumerate(hogel_results):
            ax = axes[ri, 1 + ci]
            ax.imshow(r["recon_upsampled"][vi, ui])
            if ri == 0:
                ax.set_title(f"Hogel N_SUB={r['n_sub']}\n({r['grid_size']}x{r['grid_size']} hogels)")
            ax.set_xticks([])
            ax.set_yticks([])

        ax = axes[ri, n_cols - 1]
        ax.imshow(direct["views"][vi, ui])
        if ri == 0:
            ax.set_title("Direct SLFH\n(no hogel tiling)")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        "Reconstructions vs. ground truth across views (hogel path upsampled "
        "nearest-neighbour to full resolution to show tessellation)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_grid = PLOT_DIR / "ablation_sidebyside.png"
    fig.savefig(str(out_grid), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_grid}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""
Fold the real 8-level device into the hogel-free optimiser, and measure the
cost -- on the real Cornell light field, not a toy target.

`tier2_hfh/optimise.py`'s panel used to be a continuous, unit-amplitude
phase modulator: `exp(i*phase)`. The real device is neither: it has 8 phase
levels (not a continuum), and, per `tier2_slfh/quantization_aware.py`'s
measurement of the actual Sb2Se3/DBR stack, each level reflects a different
fraction of the incident light (power reflectance 0.583-0.946 across the 8
levels). `quantised_field` in `optimise.py` folds both defects in via the
same `LevelSet` / straight-through-estimator machinery `tier2_slfh` already
built and tested -- reused here, not reimplemented.

Three strategies, crossed with two level sets and two mode counts:

  * continuous  -- the existing ceiling: free continuous phase, never quantised.
  * naive       -- optimise continuously, then quantise once at the end. The
                   free phase never sees the device during optimisation.
  * aware       -- quantise (via STE) on every forward pass during
                   optimisation, so the free phase can route around both the
                   phase steps AND the per-level amplitude non-uniformity.

  * ideal level set     -- unit amplitude, phase spaced by 2*pi/8 (what the
                            sinc closed form and phase-only-SLM literature
                            assume; isolates the PHASE-quantisation cost,
                            already proven exactly as sinc^2(pi/8) = 94.96%
                            in tests/test_phase_quantization.py).
  * realistic level set -- the actual TMM-corrected device levels from
                            tier2_rcwa.rcwa_optimization.design_phase_lut:
                            phase steps near-uniform, amplitude NOT
                            corrected (0.583-0.946 power reflectance). Diffing
                            realistic against ideal at matched strategy
                            isolates the AMPLITUDE cost the phase-only
                            literature has no counterpart for.

  * n_modes = 1, 4 -- time-multiplexed subframes. Free phase differs per
                      subframe and their intensities average, which is how
                      speckle falls as 1/sqrt(M); whether that same freedom
                      also absorbs some of the quantisation error is an open
                      question this script answers empirically, not
                      assumed.

PSNR is reported but is not the primary judgement call here (see the module
docstring notes on why -- the Cornell ceiling light is ~50x the walls, so a
noise-plus-bright-bar reconstruction can score high PSNR while looking
wrong). The figure is the primary evidence; read it.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tier2_hfh.optimise import Geometry, full_psnr, optimise
from tier2_slfh.quantization_aware import realistic_levels, uniform_levels

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64

ITERS = 5000
VIEWS_PER_ITER = 8
LR = 0.05
SEED = 0
N_LEVELS = 8
CENTER_VIEW = 8 * 17 + 8   # (u, v) = (8, 8): the middle of the 17x17 grid


def load_target():
    """
    Green channel of the real Cornell light field, downsampled from its
    native 512x512 render resolution to the optimiser's 128x128 window
    resolution by exact 4x4 block averaging (512 / 128 = 4 exactly, so this
    is an exact area-average, not an approximation) -- both resolutions
    sample the SAME 20-degree field of view (Geometry's default pitch is
    chosen to match it, see optimise.py's docstring and
    test_optimise.py::test_the_angular_span_matches_the_light_field_it_will_be_compared_against),
    just at different pixel counts, and render_views cannot emit more than
    `window` samples across that span.
    """
    lf = np.load(PLOT_DIR / "cornell_lightfield.npz")["lightfield"][..., 1]  # (17,17,512,512)
    lf = lf.reshape(289, 512, 512)
    lf = lf.reshape(289, 128, 4, 128, 4).mean(axis=(2, 4))
    return torch.from_numpy(lf.astype(np.float32))


def run_condition(target, geom, n_modes, levels, phase_cont=None):
    """
    `levels=None` -> continuous (the ceiling). `phase_cont` given and
    `levels` given -> naive (quantise the continuous solution once, no
    optimisation of its own). `phase_cont=None` and `levels` given ->
    quantisation-aware (STE quantised on every forward pass of training).
    """
    if phase_cont is not None:
        psnr = full_psnr(phase_cont, geom, target, chunk=VIEWS_PER_ITER, levels=levels)
        return phase_cont, psnr
    phase, history = optimise(target, geom, iters=ITERS, views_per_iter=VIEWS_PER_ITER,
                              lr=LR, seed=SEED, log_every=ITERS, log=lambda *_: None,
                              n_modes=n_modes, levels=levels)
    return phase, history[-1][2]


def render_view_image(phase, geom, levels, view_idx=CENTER_VIEW):
    from tier2_hfh.optimise import render_views
    with torch.no_grad():
        img = render_views(phase, geom, torch.tensor([view_idx]), levels=levels)[0]
    return img.cpu().numpy()


def display_normalise(img, pct=99.5):
    """
    Per-image percentile normalisation, NOT by the max: the reconstruction's
    brightest pixel sits several times above its own 99.9th percentile (the
    loss is scale-invariant and normalises by total power, not peak), so
    dividing by the max crushes everything else to near-black. Percentile
    normalisation plus a mild gamma makes the room visible alongside the
    bright ceiling light, in both target and reconstruction.
    """
    p = np.percentile(img, pct)
    return np.clip(img / max(p, 1e-30), 0, 1) ** 0.45


def main():
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    target = load_target()
    geom = Geometry(device=DEVICE, dtype=DTYPE)

    level_sets = {"ideal": uniform_levels(N_LEVELS), "realistic": realistic_levels(N_LEVELS)}
    results = {}       # (level_name, strategy, n_modes) -> psnr
    images = {}         # (level_name, strategy, n_modes) -> view image, for every n_modes:
                         # the effect is only sometimes visible, and which is exactly the
                         # question this script has to answer honestly, not assume.

    for n_modes in (1, 4):
        t0 = time.time()
        phase_cont, ceiling_psnr = run_condition(target, geom, n_modes, levels=None)
        results[("-", "continuous", n_modes)] = ceiling_psnr
        print(f"n_modes={n_modes} continuous: {ceiling_psnr:.2f} dB "
              f"({time.time() - t0:.1f}s)")
        images[("-", "continuous", n_modes)] = render_view_image(phase_cont, geom, levels=None)

        for level_name, levels in level_sets.items():
            t0 = time.time()
            _, naive_psnr = run_condition(target, geom, n_modes, levels, phase_cont=phase_cont)
            results[(level_name, "naive", n_modes)] = naive_psnr
            print(f"n_modes={n_modes} {level_name} naive:    {naive_psnr:.2f} dB "
                  f"(gap {naive_psnr - ceiling_psnr:+.2f} dB, {time.time() - t0:.1f}s)")
            images[(level_name, "naive", n_modes)] = render_view_image(phase_cont, geom, levels)

            t0 = time.time()
            phase_aware, aware_psnr = run_condition(target, geom, n_modes, levels)
            results[(level_name, "aware", n_modes)] = aware_psnr
            print(f"n_modes={n_modes} {level_name} aware:    {aware_psnr:.2f} dB "
                  f"(gap {aware_psnr - ceiling_psnr:+.2f} dB, {time.time() - t0:.1f}s)")
            images[(level_name, "aware", n_modes)] = render_view_image(phase_aware, geom, levels)

    print("\n" + "=" * 78)
    header = f"{'level set':<10} {'strategy':<12} {'n_modes':>7} {'PSNR dB':>9} {'gap dB':>8}"
    print(header)
    print("-" * len(header))
    for n_modes in (1, 4):
        ceiling = results[("-", "continuous", n_modes)]
        print(f"{'-':<10} {'continuous':<12} {n_modes:>7} {ceiling:>9.2f} {0.0:>8.2f}")
        for level_name in level_sets:
            for strategy in ("naive", "aware"):
                psnr = results[(level_name, strategy, n_modes)]
                print(f"{level_name:<10} {strategy:<12} {n_modes:>7} {psnr:>9.2f} "
                      f"{psnr - ceiling:>8.2f}")

    target_img = target[CENTER_VIEW].numpy()
    rows = [(n_modes, level_name) for n_modes in (1, 4) for level_name in ("ideal", "realistic")]
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 7.5 * len(rows) / 2))
    for row, (n_modes, level_name) in enumerate(rows):
        ceiling = results[("-", "continuous", n_modes)]
        cols = [
            ("target", target_img, None),
            ("continuous", images[("-", "continuous", n_modes)], ceiling),
            ("naive", images[(level_name, "naive", n_modes)], results[(level_name, "naive", n_modes)]),
            ("aware (STE)", images[(level_name, "aware", n_modes)], results[(level_name, "aware", n_modes)]),
        ]
        for col, (label, img, psnr) in enumerate(cols):
            ax = axes[row, col]
            ax.imshow(display_normalise(img), cmap="inferno")
            title = f"{level_name}, n_modes={n_modes}: {label}" if col == 0 else label
            if psnr is not None:
                title += f"\n{psnr:.1f} dB"
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.suptitle("Hogel-free reconstruction under the real 8-level device model (center view)",
                 fontsize=13)
    fig.tight_layout()
    out = PLOT_DIR / "hfh_quantisation_aware.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")

    return results


if __name__ == "__main__":
    main()

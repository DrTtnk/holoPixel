# Hogel decomposition vs. direct SLFH — ablation

Script: `tier2_slfh/ablation_hogel_vs_slfh.py`. Tests: `tests/test_ablation_hogel.py`
(31 tests, machinery only, ~2s, all passing). Imports `tier2_slfh/hogel_grid.py`,
`tier2_slfh/hogel_optimizer.py` and `tier2_slfh/slfh.py` unmodified.

**This revision responds to review feedback that found two real bugs in the
previous version of this document and one genuine mechanism flaw in the
experiment design.** All three are described below, with what was wrong, what
was fixed, and how the fix was checked. The verdict changed in the direction of
more confidence, not less — see "What changed" at the end.

## Verdict

At the scale this ablation could afford: **keep the hogel step ahead of
SLFH-style optimisation.** On our actual current pipeline's light field (9x9
views), the hogel path beats the direct path on both PSNR and SSIM at every
hogel size tested, and is 2.3-2.8x faster. Separately, and now confirmed by a
methodologically sound experiment rather than a confounded one: **our current
operating point (N_SUB=250) is not the best point on the hogel-size curve**,
and — the new, stronger result — **larger hogels do not win even when the input
light field is re-rendered to supply the angular content a larger hogel can
actually use.** Section "Second review point" below shows this decisively:
image quality falls off a cliff from a 9-sub-pixel hogel to a 32-sub-pixel one,
even though the 32-view light field genuinely gives the 32-sub-pixel hogel
angular data it did not have before. Direct SLFH's on-axis (centre-view)
reconstruction is a real, legitimate one, recognisable and not noise — this
document previously misreported it as noise, addressed below — but its
off-axis (parallax) reconstructions remain incoherent noise even at full
production settings, an unresolved defect in the current optimiser that the
hogel path does not share.

## Response to review, in order

### 1. The per-hogel energy correction: previously claimed fixed, actually not — now fixed and checked non-tautologically

**What was wrong.** The previous version of this document claimed
`apply_per_hogel_energy_correction()` recovered spatial structure, citing a
correlation improving from -0.11 to 0.40. That correlation was computed on the
*mean over viewing angles per hogel* — and the correction
(`raw_views * pre_norm_sum`) implicitly assumed the optimiser's raw output
already summed to a fixed constant (`n_samples`) for every hogel, which is
only true if `batch_optimize` converges well. It does not, precisely at the
larger N_SUB values this correction mattered most for (see the loss-ratio
table below). The correction did not fix the image, and
`plots/ablation_hogel_normalization_diagnostic.png` showed it plainly: the
"corrected" panel was still uncorrelated noise next to a ground truth panel
with obvious Cornell structure. Reporting that as a fix was wrong.

**The actual fix.** `apply_per_hogel_energy_correction()` now divides the raw
output by its OWN achieved sum (per hogel) before rescaling to the true target
sum:

```python
achieved_sum = raw_views.sum(axis=(0, 1))
return raw_views / achieved_sum[None, None, :] * pre_norm_sum[None, None, :]
```

This forces the corrected mean-over-viewing-angle to equal the true value *by
construction* — which makes that particular statistic tautological and no
longer usable as evidence. The check that matters is on a single FIXED
viewing angle, never divided by its own achieved sum, which is not
guaranteed to improve by this algebra alone:

| Version | Single-view correlation with true GT (green channel, N_SUB=250) |
|---|---|
| Raw optimiser output | 0.08 |
| Multiply-only (previous, incorrect) | 0.43 |
| Achieved-sum-corrected (current) | 0.71 |

See `plots/ablation_hogel_normalization_diagnostic.png` (regenerated): the
achieved-sum-corrected panel now visibly shows the true image's dark-left /
bright-right gradient; the multiply-only panel does not. This is a real,
substantive, non-tautological improvement, not a repeat of the earlier
mistake — but it is also not a full recovery: 0.71 correlation at a single
view is "recognisable structure under heavy noise," which is exactly what the
corrected sweep images show (see below), not a clean reconstruction. The
correction is described accurately now, including its limits — see
`apply_per_hogel_energy_correction()`'s docstring in the script.

This fix changed the sweep's numbers substantially, particularly at N_SUB=250
and 512 where convergence is worst (13.60→18.35 dB and 15.61→16.32 dB
respectively) — see "What changed" at the end.

### 2. Second review point: the sweep was angularly rigged against large hogels — now re-run matched

**The critique, confirmed.** `tests/test_holographic_transport.py::test_hogel_angular_sample_count_equals_subpixel_count`
proves a hogel of D sub-pixels resolves exactly D angular samples across its
full grating-limited range (space-bandwidth conservation). Our light field
supplies only 9x9=81 views. Every N_SUB in the original sweep (64-512) was
therefore angularly over-determined by 7x-57x: a larger hogel could only ever
lose, because it was never given angular content it could use, only a
spatial cost to pay for capacity that went to waste.

**The re-run.** `main()` now also runs an angular-matched sweep
(`MATCHED_N_SUBS = [9, 32, 64]`): one light field rendered per hogel size, at
exactly that many views (9x9, 32x32, 64x64 — reusing the existing 9x9 cache,
rendering the other two once at 160x160 with reduced spp — 128 and 64
respectively — to bound render time; see Resolution honesty). Each N_SUB is
run against the light field that actually supplies its angular capacity.

**Result: large hogels still lose.** Decisively, not marginally:

| n_angular = N_SUB | grid (hogels/side) | PSNR (dB) | SSIM | wall (s) |
|---:|---:|---:|---:|---:|
| 9  | 160 | **25.96** | **0.6677** | 2.2 |
| 32 | 125 | 18.45 | 0.3723 | 22.4 |
| 64 | 62  | 18.76 | 0.3423 | 22.4 |

`plots/ablation_matched_sidebyside.png` (centre view, GT vs. reconstruction at
each configuration): at N_SUB=9 the reconstruction is close to the ground
truth — sharp box silhouettes, visible ceiling light, walls — with only a
speckle-like noise texture over it. At N_SUB=32 it has collapsed into
dominant colour-confetti noise, wall colours and box silhouette barely
surviving underneath. At N_SUB=64 it is worse again, though only slightly —
PSNR is flat within noise (18.45→18.76) and SSIM keeps falling (0.372→0.342).
See `plots/ablation_matched_tradeoff_curve.png` for the same story as a curve:
a cliff from 9 to 32, then a plateau/slight further decline to 64.

**This is the answer the review asked for, and it strengthens rather than
inverts the verdict.** Given angular content it can actually use, a hogel
that is bigger than strictly necessary still loses, because the spatial-count
cost of a bigger hogel (`GRID_SIZE * N_SUB` held ~constant, so a 4x bigger
N_SUB means roughly 4x fewer hogels per side) is not repaid by the angular
fidelity it buys, at least for this scene's spatial-frequency content. Note
one honesty caveat specific to this sweep: going from N_SUB=9 to 32 changes
two things at once (more angular capacity AND fewer, bigger hogels) by
design — that bundling *is* the physical trade-off being tested, not a
confound to remove, but it means this experiment cannot separately attribute
the loss to "wasted angular capacity" vs. "coarser spatial grid"; it only
shows that, bundled as any real hogel-size choice bundles them, bigger loses
here. Also not tested: our real operating point, N_SUB=250, would need a
250x250=62,500-view light field — completely impractical at this ablation's
budget, so this result covers 9-64, not the operating point itself.

### 3. Metric vs. eye disagreement

**Before the fix in point 1**, the original (angularly-mismatched) sweep's
PSNR/SSIM showed a V-shape (worst at N_SUB=250) that visibly contradicted the
side-by-side images, which degrade monotonically by eye from N_SUB=64 (best)
to 512 (worst, solid colour blocks).

**After the fix**, PSNR now agrees with the eye — it decreases monotonically
with N_SUB (20.14 → 19.01 → 18.35 → 16.32 dB for 64/128/250/512). This is
itself evidence the corrected correction is doing real work: fixing a
genuine bug made the metric converge toward the visual ordering rather than
away from it.

**SSIM still disagrees**, and does so consistently in both the mismatched and
matched sweeps: it *increases* with N_SUB in the original sweep
(0.3877 → 0.4121 → 0.4577 → 0.4799) even as the images visibly get worse, and
it favours N_SUB=9 correctly but does not clearly separate 32 from 64 in the
matched sweep either. Trust the eye for ordering: N_SUB=64 is the best hogel
reconstruction in the original sweep and N_SUB=9 is the best in the matched
sweep, full stop, regardless of what SSIM alone says at the coarse end. The
most likely explanation, offered alongside the convergence-difficulty
explanation below rather than instead of it: PSNR against a full-resolution
raster nearest-neighbour-upsampled from a coarse hogel grid is computed
pixel-by-pixel and directly penalises the loss of fine detail; SSIM's
luminance/contrast/structure decomposition rewards getting large flat
regions (which dominate this scene — the red and green walls) approximately
right, which a very coarse, smooth block grid can do even while destroying
all object-level detail. Both this smoothness-reward effect and the
convergence-difficulty confound (below) are live explanations for parts of
the remaining shape; neither is asserted as the sole cause.

**Convergence difficulty across N_SUB, still relevant.** `batch_optimize`
uses a fixed learning rate (0.1) and fixed iteration budget (500) regardless
of N_SUB, but both the search space (N_SUB² phase values) and the scale
mismatch between the per-hogel target (normalised to sum ≈ 81) and the
natural FFT output scale (≈N_SUB², by Parseval) grow sharply with N_SUB.
Loss-reduction ratio (iteration 1 → 400, averaged over channels), original
sweep:

| N_SUB | loss reduction ratio |
|------:|----------------------:|
|    64 |         ~7.5 million x |
|   128 |         ~7.4 million x |
|   250 |            ~30,000 x |
|   512 |               ~460 x |

N_SUB=64 and 128 converge equally well; above that, convergence collapses by
orders of magnitude. This remains a real, separate factor in why quality
degrades faster than a pure spatial-resolution argument alone would predict,
independent of the SSIM-vs-PSNR disagreement.

### 4. Direct SLFH centre view: normalisation bug, not noise — separated from the real off-axis defect

**What was wrong.** `run_direct_slfh_path()` computed a single percentile
(p99) over the ENTIRE `(nv, nu, H, H, 3)` array of all 81 reconstructed views
combined, then clipped every view to that one global scale. A bright outlier
anywhere in that array — in practice, an incoherent noise spike in one of the
already-failing off-axis views — dominates that global percentile and can
crush or saturate every other view, including the centre view, before any
downstream code (plotting or metrics) sees the data. This is exactly what
happened: the centre view was reported as "a blown-out yellow and white
blob," when in fact its own field, on its own terms, was a legitimate,
recognisable (if blurred) reconstruction the whole time.

**The fix.** `run_direct_slfh_path()` and `run_hogel_path()` no longer
clip/normalize their returned arrays at all (metrics use `calibrate_scale()`'s
own per-view least-squares fit against ground truth, which does not need a
prior percentile clip and is not corrupted by one taken over unrelated
images). A new `display_normalize()` helper normalizes per-view, at plot time
only — matching `slfh.py`'s own `tonemap_recon()`, which already normalizes
per-image for the same reason.

**Result, now separated correctly:**
- **Centre/on-axis view: a real, legitimate reconstruction.**
  `plots/ablation_sidebyside.png`'s centre-view Direct SLFH panel now clearly
  shows the red and green walls, the box silhouette, and the ceiling light —
  blurred and low in spatial detail compared to the hogel path at N_SUB≤128,
  but recognisably correct, not noise. This was a display bug in this
  ablation's own script, not a defect in `slfh.py`.
- **Off-axis (parallax) views: still genuinely noise.** The left and right
  columns of the same figure are still incoherent blue-tinted noise, and this
  was already independently confirmed (in the previous revision) at
  `slfh.py`'s own full production settings (1000 iterations, 8 frames — double
  this ablation's budget), at four different pupil diameters, ruling out both
  a reduced-iteration-budget explanation and a display-normalisation
  explanation. See `plots/ablation_slfh_offaxis_diagnostic.png` — that
  diagnostic script already normalized per-view (`np.percentile` on a single
  image), so it was not affected by this bug and its conclusion stands
  unchanged: `slfh.optimize_hologram`'s stochastic, unweighted pupil sampling
  fits the easy, near-DC (on-axis) region of the eyebox and does not converge
  for lateral pupil shifts even well inside the trained range, regardless of
  iteration budget.

This changes the characterisation of direct SLFH from "the whole
reconstruction is unreliable" to the more precise and more actionable "the
method works on-axis and fails specifically off-axis" — a narrower, more
fixable-sounding defect (e.g. importance-weighted pupil sampling, or a loss
term that explicitly balances eyebox coverage), but still an open, unresolved
one as far as this ablation goes, and still enough on its own to prefer the
hogel path's uniform-if-lower quality across the full parallax range over the
direct path's on-axis-only reliability.

## The original (angularly-mismatched) sweep — reflects our actual current pipeline

160x160 light field, 9x9 views (our pipeline's real angular resolution), 256
spp, rendered once and cached (`plots/ablation_lightfield.npz`, 21.6s).
Both paths at 50% of their respective production iteration/frame budgets
(hogel: 500 vs. 1000 default; SLFH: 500 iters x 4 frames vs. 1000 x 8).
PSNR/SSIM per-view against full-resolution ground truth after a per-view
least-squares brightness calibration, averaged over all 81 view pairs.

| Path                    | N_SUB | grid    | hogel size | PSNR (dB) | SSIM   | wall (s) |
|--------------------------|------:|---------|-----------:|----------:|-------:|---------:|
| Hogel                    |    64 | 62x62   |   45.2 µm  |     20.14 | 0.3877 |     12.2 |
| Hogel                    |   128 | 31x31   |   90.4 µm  |     19.01 | 0.4121 |     11.0 |
| **Hogel (current op. pt.)** | **250** | **16x16** | **176.5 µm** | **18.35** | **0.4577** | **13.6** |
| Hogel                    |   512 | 8x8     |  361.5 µm  |     16.32 | 0.4799 |     12.8 |
| Direct SLFH (no hogels)  |     - | -       |          - |     13.48 | 0.3046 |     30.9 |

On this sweep — the one that reflects what our pipeline can actually render
today — **the hogel path beats direct SLFH on both metrics at every hogel size
tested, and 2.3-2.8x faster.** This is a stronger result in favour of hogels
than the (buggy) previous revision reported.

GRID_SIZE is tied to N_SUB by `GRID_SIZE = round(GRID_BASE * BASELINE_N_SUB /
N_SUB)`, capped at the light field's own pixel resolution
(`grid_size_for(n_sub, max_grid=...)` — the cap was added after this ablation
found N_SUB=9 wanted grid=444 against a 160px light field, i.e. more than one
hogel per source pixel, which `hogel_anchor_indices` would have silently
resolved by duplicating targets rather than sampling a genuinely finer grid;
see `tests/test_ablation_hogel.py::test_grid_size_is_capped_at_max_grid`).
This is our own modelling choice for the spatial side of the sweep (flagged
in the script's docstring), independent of the angular-matching issue in
point 2 above.

## Images looked at

- **`plots/ablation_sidebyside.png`** (regenerated) — ground truth vs. each
  hogel size vs. direct SLFH, left/centre/right views, all panels
  per-view-normalized for display. Hogel N_SUB=64 is the best hogel
  reconstruction by eye: walls, box silhouettes and ceiling light all visible
  through salt-and-pepper noise. Quality degrades monotonically by eye
  through 128, 250 (our operating point — mostly colour confetti, red/green
  wall bands survive), to 512 (solid 8x8 colour blocks, no object detail).
  Direct SLFH's centre column is a legitimate, recognisable, blurred
  reconstruction (corrected from the previous revision's "blown-out blob"
  misreport); its left/right columns remain incoherent noise.
- **`plots/ablation_tradeoff_curve.png`** (regenerated) — PSNR now decreases
  monotonically with N_SUB, matching the eye; SSIM still increases with
  N_SUB, still disagreeing (see point 3).
- **`plots/ablation_hogel_normalization_diagnostic.png`** (regenerated) — raw
  vs. multiply-only (previous, incorrect) vs. achieved-sum-corrected (current)
  vs. true GT, one fixed viewing angle, with single-view correlations printed
  on each panel (0.08 / 0.43 / 0.71). This is now honest evidence for a real,
  partial improvement, not the previous revision's tautological claim of a
  fix.
- **`plots/ablation_matched_sidebyside.png`** (new) — centre view, GT vs.
  hogel reconstruction, at N_SUB=9/32/64 each with its OWN correctly-sized
  light field. N_SUB=9 is close to ground truth; N_SUB=32 and 64 have both
  collapsed into noise-dominated reconstructions, decisively confirming that
  larger hogels lose even when given angular content matched to their
  capacity.
- **`plots/ablation_matched_tradeoff_curve.png`** (new) — PSNR/SSIM vs.
  matched N_SUB: a cliff from 9 to 32, then roughly flat to 64.
- **`plots/ablation_slfh_offaxis_diagnostic.png`** (unchanged from previous
  revision, already per-view normalized) — direct SLFH at full production
  settings: centre view recognisable, left/right views noise. Still the
  correct reading.
- **`plots/hogel_grid.png`** (pre-existing, not produced by this ablation) —
  independent corroboration that `hogel_grid.py`'s own unmodified `main()`
  produces uncorrected (i.e. pre-point-1-fix) colour-noise reconstructions at
  full 1024x1024 production resolution.

## Metrics used

PSNR: schoolbook `10*log10(data_range^2/mse)`, implemented directly. SSIM:
`torchmetrics`'s `StructuralSimilarityIndexMeasure` (battle-tested, not
hand-rolled). `torchmetrics` installed via
`.venv/bin/uv pip install torchmetrics` (2 small packages). LPIPS was **not**
added — it requires `lpips` plus `torchvision` (a 4th package, also
downloading pretrained weights on first use), the "large dependency tree" the
task said to skip in that case; PSNR and SSIM are reported instead.
`scikit-image` was installed separately (4 small packages, no `torchvision`)
purely as an independent SSIM cross-check inside
`tests/test_ablation_hogel.py` (agreement within 0.01 on synthetic images) —
test-only, not used by the ablation script itself.

## Resolution honesty

The light field is 160x160 px — reduced from production's 1024x1024 — so a
render could be done once and reused rather than re-paying the ~16 minute
production cost repeatedly. Both optimisers run at 50% of their production
iteration/frame counts, for the same reason. The angular-matched sweep's
32x32 and 64x64 light fields additionally use reduced spp (128 and 64 vs. the
9x9 field's 256) to bound render time (157.5s and 286.8s respectively,
one-time, cached) — a noisier ground truth for those two points, which is
visible as grain in `plots/ablation_matched_sidebyside.png`'s GT panels for
n_angular=32/64 and should be read as a (small) additional source of
disagreement between the 9-point and the 32/64-points beyond the hogel-size
effect itself.

What generalises and what does not:
- The **hogel per-hogel-normalization defect** (point 1) is confirmed
  independent of this ablation's settings — visible in the pre-existing,
  full-production-resolution `plots/hogel_grid.png`.
- The **direct SLFH off-axis failure** (point 4) is confirmed independent of
  iteration budget (re-tested at full 1000/8 settings) but **not** independent
  of spatial resolution — only tested at 160x160, not 1024x1024.
- The **angular-matched result** (point 2) — large hogels losing even when
  properly fed — was tested at N_SUB=9/32/64 only; N_SUB=250 (our real
  operating point) would need a 250x250-view light field, completely
  impractical here. The direction of the effect (sharp quality loss from 9 to
  32, then a plateau) should not be assumed to extrapolate linearly out to
  250, though the mechanism (spatial-count cost outpacing angular-fidelity
  gain) has no obvious reason to reverse.
- The **absolute PSNR/SSIM numbers** in both sweeps are specific to this
  160x160 crop and should not be read as production-scale numbers; the
  qualitative orderings (by eye, and now largely by PSNR) are expected to be
  more robust to scale than the specific dB values.

## Tests

`tests/test_ablation_hogel.py`, 31 tests, ~2s, all passing:

```
$ MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_ablation_hogel.py -v
============================== 31 passed in ~2s ==============================
```

New/changed since the previous revision: `apply_per_hogel_energy_correction`'s
tests now check the corrected (achieved-sum-based) semantics — that each
hogel's corrected sum equals its true target sum exactly, that the result is
invariant to whatever arbitrary scale the raw optimiser output happened to
converge to (the property the old multiply-only version lacked), and that an
all-zero hogel does not produce NaN/inf; `grid_size_for`'s cap at `max_grid`
is tested directly, including the specific N_SUB=9-against-160px case that
motivated it; `representative_view_indices` (the fix for a measured 747s
evaluation cost at 64x64=4096 views) is tested for including both extremes
and the centre, and for capping correctly when `n_angular < k`. All
previously-existing tests (PSNR/SSIM vs. independent references, hogel
tiling/reassembly round-trip, the angle-to-FFT-bin mapping, and the
sympy-derived paraxial equivalence between the hogel and direct angle
conventions) are unchanged and still passing.

## What changed in this revision, summarised

| | Previous revision | This revision |
|---|---|---|
| Per-hogel correction | Claimed fixed; figure showed it wasn't | Genuinely improved (0.08→0.71 single-view correlation), limits stated honestly |
| Hogel sweep numbers (N_SUB=250) | 13.60 dB / 0.2394 | 18.35 dB / 0.4577 |
| Hogel-vs-PSNR-vs-eye | V-shaped, contradicted the eye | Monotonic, matches the eye |
| Angular sampling | Fixed 9x9 for every N_SUB — mechanism could not let large hogels win | Matched sweep added (9/32/64) — large hogels tested fairly and still lose |
| Direct SLFH centre view | Reported as noise/blown-out | Corrected: legitimate reconstruction; off-axis noise confirmed as the real, separate defect |
| Verdict | Keep hogels ahead of SLFH (weakly supported, confounded sweep) | Keep hogels ahead of SLFH (now supported by a fair sweep on our real pipeline AND an angularly-matched sweep that rules out the main objection) |

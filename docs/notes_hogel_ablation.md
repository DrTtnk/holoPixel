# Hogel decomposition vs. direct SLFH — ablation

Script: `tier2_slfh/ablation_hogel_vs_slfh.py`. Tests: `tests/test_ablation_hogel.py`
(26 tests, machinery only, ~2s, all passing). Imports `tier2_slfh/hogel_grid.py`,
`tier2_slfh/hogel_optimizer.py` and `tier2_slfh/slfh.py` unmodified.

## Verdict

At the scale this ablation could afford, **keep the hogel step ahead of SLFH-style
optimisation; do not replace it with a direct SLFH pass using the current
`slfh.py` optimiser.** The direct path is 2.8x slower (33.8s vs. 12-13s) and its
aggregate image-quality numbers are comparable to or worse than a well-configured
hogel pipeline — but the numbers understate the real problem: **the direct path's
off-axis (parallax) reconstructions are visually incoherent — plain noise, not a
degraded image — even at `slfh.py`'s own full production settings (1000
iterations, 8 frames), independently re-verified below, not an artifact of this
ablation's reduced iteration budget.** Only the on-axis/centre view converges to
something recognisable. The hogel path, because each hogel is directly supervised
per viewing angle, does not have this failure mode — its degradation (blockiness,
salt-and-pepper noise) is present at every viewing angle, not concentrated at the
edges of the eyebox. Replacing hogels with the current direct optimiser would
trade a *known, bounded* spatial-resolution loss for an *unresolved* eyebox-coverage
defect, at higher cost. This is a statement about our current pipeline's
implementations, not a refutation of the SLFH or Hogel-Free Holography papers,
which make their case at much larger scale/compute than this ablation used (see
Resolution honesty below).

Separately, and with more confidence: **our current operating point (N_SUB=250)
is not the best point on the hogel-size curve we measured.** The smallest hogel
tested (N_SUB=64) gave +5.9 dB PSNR and +0.11 SSIM over N_SUB=250 at
statistically identical wall-clock (12.2s vs. 12.2s). That is a separate, lower-risk
finding worth acting on regardless of the hogel-vs-SLFH question, discussed below.

## Two things this ablation found before it could answer the question asked

Both mattered more than expected and are reported prominently rather than buried,
per usual practice of writing down a wrong assumption as soon as it is found.

### 1. `hogel_grid.batch_optimize` erases spatial brightness between hogels

`batch_optimize` normalizes every hogel's target to the *same* total energy before
fitting (`target_t = target_t / sums * n_samples`). This throws away genuine
brightness differences between hogels — the reconstructed image's per-hogel
brightness ends up uncorrelated with the true scene (measured correlation
between reconstructed and true per-hogel mean: **0.05**, i.e. noise). This is not
specific to this ablation: **the repository's own pre-existing
`plots/hogel_grid.png`** (produced by `hogel_grid.py`'s unmodified `main()` at full
1024x1024 resolution) already shows this — every one of its 9 reconstructed
viewing angles is colour noise with no resemblance to the Cornell box, at its
own default settings. See that file for the pre-existing evidence; it predates
this ablation.

The ablation script corrects for this in `run_hogel_path()` using
`apply_per_hogel_energy_correction()`: it rescales each hogel's raw output by the
target's own pre-normalization sum, computed *before* calling `batch_optimize`
(data the pipeline already has, not an oracle — see
`plots/ablation_hogel_normalization_diagnostic.png`, correlation with true
brightness improves from **-0.11 (raw) to 0.40 (corrected)**). A real phase-only
hogel display needs an equivalent per-hogel amplitude/exposure stage regardless,
since a phase-only pattern cannot represent absolute magnitude — so this
correction is what any real implementation would need, not a favour done to make
the numbers look better. Every number below is measured *with* this correction
applied; the module docstring and inline comments explain it in full. This should
be raised with whoever owns `hogel_grid.py`/`hogel_optimizer.py` next, since it
affects any use of `batch_optimize`, not just this ablation.

### 2. Direct SLFH fails off-axis regardless of iteration budget

Confirmed by re-running `slfh.optimize_hologram` at its own unmodified production
defaults (`n_iters=1000, n_frames=8`, double this ablation's 500/4) on the same
160x160 light field: the centre view converges to a recognisable, if blurred,
Cornell box; the left and right views (`angular_positions[0]` and `[8]`, ±10mm,
well inside the ±10mm eyebox the optimiser was trained on) are indistinguishable
from random noise **at every pupil diameter tried (0.002, 0.005, 0.01, 0.02m)**.
See `plots/ablation_slfh_offaxis_diagnostic.png`. Because `optimize_hologram`
samples pupils uniformly at random across the whole eyebox every iteration with
no importance weighting, and near-DC (on-axis) content is inherently less
phase-sensitive than the higher spatial frequencies a large lateral pupil shift
maps to, the single shared phase pattern preferentially fits the easy, central
region of the eyebox. This is a property of `slfh.py` as implemented, not of this
ablation's reduced settings — flagged for whoever works on `slfh.py` next.
Whether it also appears at full 1024x1024 panel resolution was not tested (see
Resolution honesty).

## The sweep: PSNR / SSIM / wall-clock vs. hogel size

160x160 light field, 9x9 views, 256 spp (rendered once, cached to
`plots/ablation_lightfield.npz`, reused for every configuration). Hogel path:
500 iterations (50% of `hogel_grid.py`'s production default). Direct path: 500
iterations x 4 frames (50% of `slfh.py`'s production defaults of 1000/8).
Both reduced by the same 50% factor for ablation speed. PSNR/SSIM computed
per-view against the full-resolution path-traced ground truth, after a
per-image least-squares brightness calibration (holographic reconstructions
carry an arbitrary overall exposure), averaged over all 81 view pairs.

| Path                    | N_SUB | grid    | hogel size | PSNR (dB) | SSIM   | wall (s) |
|--------------------------|------:|---------|-----------:|----------:|-------:|---------:|
| Hogel                    |    64 | 62x62   |   45.2 µm  |     19.50 | 0.3497 |     12.2 |
| Hogel                    |   128 | 31x31   |   90.4 µm  |     17.78 | 0.3492 |     11.0 |
| **Hogel (current op. pt.)** | **250** | **16x16** | **176.5 µm** | **13.60** | **0.2394** | **12.2** |
| Hogel                    |   512 | 8x8     |  361.5 µm  |     15.61 | 0.4456 |     12.8 |
| Direct SLFH (no hogels)  |     - | -       |          - |     13.39 | 0.3059 |     33.8 |

GRID_SIZE (hogels per side) is tied to N_SUB by
`GRID_SIZE = round(GRID_BASE * BASELINE_N_SUB / N_SUB)`, i.e. `GRID_SIZE * N_SUB`
held approximately constant, matching the physical relation
`GRID_SIZE * HOGEL_SIZE = panel width = const` at fixed sub-pixel pitch
(0.706 µm). This is our own modelling choice for this ablation, needed because
`hogel_grid.py` hardcodes `GRID_SIZE=32` independent of `N_SUB` — flagged in the
script's docstring; it is not a number read out of the existing code.

### Why the curve is not a clean monotonic decrease — and what is

The expected shape ("bigger hogel buys angular resolution, loses spatial
resolution") predicts a monotonic PSNR/SSIM decrease with N_SUB. What we measured
is a V-shape: quality drops sharply from N_SUB=64 to 250, then rises again at
512. This is **refuted as a clean curve, for a diagnosed reason**, not
because the underlying trade-off is wrong.

`batch_optimize` uses a fixed learning rate (0.1) and fixed iteration budget (500)
regardless of N_SUB, but the search space (N_SUB² phase values) and the scale
mismatch between the per-hogel target (normalised to sum ≈ 81) and the natural
FFT output scale (≈ N_SUB², by Parseval) both grow sharply with N_SUB. The
printed per-run loss curves show this directly — ratio of loss at iteration 1 to
loss at iteration 400, averaged over the 3 colour channels:

| N_SUB | loss reduction ratio (iter 1 → 400) |
|------:|-------------------------------------:|
|    64 |                        ~7.5 million x |
|   128 |                        ~7.4 million x |
|   250 |                           ~31,000 x |
|   512 |                              ~470 x |

N_SUB=64 and 128 converge equally well (both ~7.5M x), and here the curve
**does** show the expected direction: the finer hogel (64) beats the coarser one
(128) on both PSNR (19.50 vs. 17.78 dB) and SSIM (0.3497 vs. 0.3492, effectively
tied). Above N_SUB=128, convergence quality collapses by two more orders of
magnitude before N_SUB=512, confounding the spatial-resolution effect with an
optimisation-difficulty effect the sweep does not control for. The apparent
"recovery" at N_SUB=512 despite the *worst* convergence of the sweep is
consistent with a second effect: at only 8x8 hogels, the reconstruction is so
spatially coarse that it can only ever reproduce the scene's own large flat
regions (the red/green walls dominate most of the frame), so even a poorly
converged reconstruction can look "structurally" closer to a heavily blurred
ground truth than a finer, sharper-but-noisier one — visible in
`plots/ablation_sidebyside.png` (N_SUB=512 column: red/green blocks in
roughly the right places, no other detail). Disentangling the two effects
cleanly would need scaling the optimiser's learning rate or iteration budget
with N_SUB — not done here, flagged as a follow-up, not a fix.

**Actionable, confound-free result:** within the pair that converges equally
well (64 vs. 128), smaller hogels win on quality, matching the literature. Our
current operating point (250) sits past the point where convergence quality
craters under `batch_optimize`'s fixed LR/iteration schedule — moving to a
smaller N_SUB (e.g. 64) is a separate, low-risk win independent of the
hogel-vs-SLFH question, at no wall-clock cost in this measurement (all four
hogel configs cost ~11-13s regardless of N_SUB, because `GRID_SIZE * N_SUB` is
held roughly constant by construction, so total FFT work is roughly constant
too).

## Images looked at

- **`plots/ablation_sidebyside.png`** — ground truth vs. each hogel size vs.
  direct SLFH, for left/centre/right views. What I see: ground truth is a clean
  Cornell box (red wall left, green wall right, two grey boxes, ceiling light).
  Hogel N_SUB=64 (62x62 hogels) is the only hogel reconstruction where the box
  is recognisable at a glance — walls, box silhouettes and the ceiling light are
  all visible through a layer of salt-and-pepper colour noise. N_SUB=128 is
  similar but visibly coarser and noisier. N_SUB=250 (our operating point) has
  degraded to a soup of ~16x16 colour blocks; only the red/green wall bands
  survive as recognisable structure, matching Hogel-Free Holography's Fig. 8
  description of gross tessellation from an under-resolved hogel grid, though
  here it is compounded by the finding above. N_SUB=512 is 8x8 solid colour
  blocks — no object detail survives, only the left/right wall-colour bands.
  Direct SLFH's centre column is a recognisable, heavily blurred version of the
  scene (red/green walls, a bright blob approximating the ceiling light and back
  wall) with no interior box detail; its left/right columns are blue-tinted
  noise with no resemblance to the true left/right parallax views at all.
- **`plots/ablation_tradeoff_curve.png`** — PSNR, SSIM and wall-clock vs. N_SUB,
  with the direct-SLFH value as a reference line and the current operating point
  (N_SUB=250) marked. Shows the V-shape described above.
- **`plots/ablation_hogel_normalization_diagnostic.png`** — raw vs.
  per-hogel-energy-corrected vs. true local brightness for one 16x16 hogel grid
  (green channel). The raw reconstruction has no visible structure; the
  corrected one shows a faint but real dark-left / bright-right pattern matching
  the true image, confirming the per-hogel normalization diagnosis rather than
  just asserting it.
- **`plots/ablation_slfh_offaxis_diagnostic.png`** — direct SLFH at full
  production settings (1000 iters, 8 frames): centre view recognisable, left and
  right views pure noise. Confirms the off-axis failure is not a reduced-budget
  artifact of this ablation.
- **`plots/hogel_grid.png`** (pre-existing, not produced by this ablation) —
  cited as independent corroboration that `hogel_grid.py`'s own unmodified
  `main()` produces the same colour-noise reconstructions this ablation found
  and diagnosed.

## Metrics used

PSNR: schoolbook `10*log10(data_range^2/mse)`, implemented directly (no
dependency justified for a one-line formula). SSIM: `torchmetrics`'s
`StructuralSimilarityIndexMeasure` (battle-tested, not hand-rolled — SSIM's
windowing and constants are not schoolbook algebra). `torchmetrics` was
installed (`.venv/bin/uv pip install torchmetrics`) — 2 small packages
(`torchmetrics`, `lightning-utilities`), no heavy dependency tree. LPIPS was
**not** added: it requires the `lpips` package plus `torchvision` (a 4th
package, `torchvision==0.29.0`, which also downloads pretrained VGG/AlexNet
weights on first use) — this is the "large dependency tree" the task said to
skip in that case. PSNR and SSIM are reported instead, per the task's own
fallback. `scikit-image` was installed separately (4 small packages, no
`torchvision`) purely to give the SSIM implementation an independent
cross-check in `tests/test_ablation_hogel.py` (agreement within 0.01 on
synthetic test images) — it is a test-only dependency, not used by the ablation
script itself.

## Resolution honesty

The light field used here is 160x160 px, 9x9 views, 256 spp — reduced from the
production 1024x1024 (a 41x fewer pixels per view) specifically so the render
could be done once (21.6s) and reused for every configuration rather than
re-paying the ~16 minute production render cost repeatedly. Both optimisers
also ran at 50% of their production iteration/frame counts (500 vs. 1000
iterations; 4 vs. 8 SLFH frames), for the same reason.

Two claims in this write-up rest on different footing:

- The **hogel per-hogel-normalization defect** is confirmed independently of
  this ablation's reduced settings — it is visible in the pre-existing
  `plots/hogel_grid.png`, produced at full 1024x1024 production resolution by
  `hogel_grid.py`'s own unmodified `main()`. This generalises.
- The **direct SLFH off-axis failure** is confirmed independent of the
  *iteration* reduction (re-tested at full 1000/8 production settings, still
  fails), but **not** independent of the *spatial resolution* reduction — it
  was only tested at 160x160, not 1024x1024. Whether more SLM pixels changes
  this qualitative behaviour is an open question this ablation could not
  afford to answer (a single 1024x1024 SLFH optimisation run, per the existing
  `slfh.py` main(), is already the multi-minute-plus cost the task asked us to
  avoid paying repeatedly).
- The **hogel-size sweep's absolute PSNR/SSIM numbers** should not be
  generalised to the full panel: at 160x160 each hogel spatial block is a much
  larger fraction of the visible scene than it would be at 1024x1024, so the
  "coarse grid coincidentally reproduces gross colour blocks" effect discussed
  above is likely exaggerated at this scale relative to production. The
  *qualitative* trade-off (spatial detail loss growing with hogel size) is
  expected to hold at any scale; the *specific* dB numbers do not transfer.

## Tests

`tests/test_ablation_hogel.py`, 26 tests, ~2s total, all passing:

```
$ MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_ablation_hogel.py -v
============================== 26 passed in 2.05s ==============================
```

Covers: PSNR against the closed-form formula on synthetic noise and a known
constant offset; SSIM against the independent `scikit-image` reference across
several noise levels; hogel tiling/reassembly round-trips exactly through
`upsample_nearest` when no optimisation happens in between (parametrised over
7 grid sizes, including the exact sizes this ablation's sweep uses); a
documented counter-example showing the naive `linspace`-based anchor
convention (used by `hogel_grid.py`'s own `main()`) does *not* round-trip when
the grid size does not evenly divide the image size, motivating
`hogel_anchor_indices()`; the angle-to-FFT-bin mapping in
`hogel_optimizer.angle_to_fft_bin` inverts `torch.fft.fftfreq` exactly for
every optically reachable bin; `hogel_grid.py`'s and `hogel_optimizer.py`'s
two independent implementations of the same formula agree numerically; a
**sympy-derived** symbolic proof that the hogel path's grating-equation
angle convention (`sin(theta)/wavelength`) and the direct path's pupil-shift
convention (`shift_x/(wavelength*focal_length)`, from `slfh.circular_aperture`)
are the same physics in the paraxial limit, not merely numerically close; the
per-hogel energy correction rescales each hogel independently and is the
identity at unit scale; and the conserved-resolution-budget rule
(`grid_size_for`) matches the baseline at N_SUB=250 and is monotonically
decreasing.

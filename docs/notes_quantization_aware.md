# Quantization-Aware Phase Optimization — Findings

Module: `tier2_slfh/quantization_aware.py`. Tests: `tests/test_quantization_aware.py`
(22 tests, all passing, ~5s). Sweep reproduced with
`MPLBACKEND=Agg .venv/bin/python tier2_slfh/quantization_aware.py` (~17s wall-clock
total on the RTX 5090 Laptop, 64×64 single hogel, two target types).

**Revision note:** an earlier draft of this document treated "3-bit == 95%
efficiency" as an absolute-efficiency claim, concluded it was false, and read
naive quantize-after beating STE/Gumbel-Softmax as a ranking reversal versus
Choi et al. Both of those were category errors, caught on review by dividing
the naive rows by the module's own continuous-phase ceiling. This revision
corrects them, adds a dense natural-image target (the regime the original
draft was missing), and replaces the Gumbel-Softmax score gain with a derived
constant instead of a fitted one. Section-by-section changes are noted below.

## The gap this closes

README.md and DRAFT.md §6.4 claim "3-bit phase (8 levels) is sufficient: 95%
diffraction efficiency", from the idealized N-level blazed-grating closed form
`eta = (sin(pi/N)/(pi/N))^2` (proven symbolically in `tests/test_phase_quantization.py`;
`eta(8) = 0.9496`, not the `0.9505` this document previously quoted — corrected
below and in the shared test docstring). None of `slfh.py`, `hogel_grid.py`,
`hogel_optimizer.py` ever quantize phase — they optimize continuous phase and
the 95% number has never been checked against the code that actually makes
our holograms. This module runs that check on a real optimizer, at 1–4 bits,
with three optimization schemes (naive quantize-after, straight-through
estimator, Gumbel-Softmax, following Choi et al. arXiv:2205.02367 §3.2) and
against two level sets (idealized uniform, and the realistic non-uniform
level set of our actual Sb2Se3/DBR device) and two target classes (a sparse
point target, and a dense natural-image target).

## 1. The sinc law is confirmed, not refuted — on the sparse target

**`eta(N) = (sin(pi/N)/(pi/N))^2` is exactly what "3-bit == 95%" measures,
and naive quantize-after-optimize reproduces it to within 0.1% at every bit
depth**, once it is correctly read as a ratio to the continuous-phase
optimum, not as an absolute efficiency:

| bits | N | naive/continuous | sinc²(π/N) | error |
|---:|---:|---:|---:|---:|
| 1 | 2 | 0.4056 | 0.4053 | +0.08% |
| 2 | 4 | 0.8101 | 0.8106 | −0.06% |
| 3 | 8 | 0.9495 | 0.9496 | −0.01% |
| 4 | 16 | 0.9872 | 0.9872 | −0.01% |

(uniform level set, sparse 6-point target, 64×64, continuous ceiling 0.8018;
reproduced in `tests/test_quantization_aware.py::test_naive_efficiency_matches_sinc_law_relative_to_continuous_ceiling`
at a 2% tolerance so this cannot silently regress). This is, as far as we can
tell, the first confirmation of the idealized blazed-grating closed form on a
*real 2D hologram optimizer with a real target*, rather than on an idealized
grating. The absolute number people should quote is: **naive quantize-after
retains 94.95% of whatever continuous phase achieves, at 3 bits** — not "95%
efficiency" as a standalone figure. Our measured absolute efficiency at 3-bit
was 0.7613 (continuous ceiling 0.8018) — a property of this target, not of
the quantization law, and not comparable across different targets or hogels.

## 2. Finding 1, corrected: this is the wrong regime to see Choi et al.'s effect, not a reversed ranking

The previous draft read naive beating STE and Gumbel-Softmax at every bit
depth ≥ 2 as "the opposite ranking from Choi et al." That framing is wrong.
**Naive is not winning on merit — it is saturating a theoretical bound to two
decimal places, for free, because it starts from a fully-converged continuous
solution and rounds it once.** STE and Gumbel-Softmax are stochastic,
iterative optimizers that converge *imperfectly* against a non-differentiable
target; on the sparse point target there is essentially nothing left in the
bound for them to recover, so their only visible effect is the noise and
convergence slack of online optimization. They are not losing to naive, they
are failing to reach a ceiling naive reaches without even needing gradients.

Choi et al.'s reported advantage for surrogate-gradient and Gumbel-Softmax
methods appears specifically where naive quantize-after **falls short** of
what continuous phase can do — dense, high-spatial-frequency natural-image
content, with time multiplexing. A 6-point sparse target is close to the
opposite extreme: continuous phase already has very little to lose to
quantization (the sinc law describes exactly how little), so there is no gap
for a quantization-aware optimizer to close. Section 3 below reruns the
comparison on dense content, which is the regime Choi et al. actually test
and the regime our display has to render.

## 3. Dense natural-image target: this is where the effect shows up, and it is large

Re-ran the full sweep against `skimage.data.camera()` downsampled to 64×64
(dense, broadband spatial content, unlike the 6 isolated points above).

**`efficiency_metric` (signal-power-in-target-support / total-power)
degenerates to ≈1.0 for every scheme at every bit depth on this target**,
because a natural photograph has almost no exact-zero pixels: the "signal
support" mask covers ~100% of the frame, so the ratio is vacuous by
construction. This is a property of using a full-frame support mask as the
efficiency definition, not a claim about the optimizer, and the module now
detects and flags this automatically (`print_sinc_law_check` checks the
target's bright-pixel support fraction and skips the sinc-law comparison when
it exceeds 50%). PSNR/SSIM against the continuous-phase reconstruction remain
meaningful and are the operative metrics here.

Caveat on absolute quality: the continuous-phase ceiling itself is a poor
match to the source photograph (PSNR-vs-target only 7.21 dB) — 600 iterations
of plain per-pixel MSE gradient descent on a phase-only Fourier hologram is a
weak recipe for dense image content compared to e.g. Gerchberg-Saxton or a
perceptual loss, and that was not the point of this experiment. What *is*
meaningful, because every scheme is compared against the same continuous
solution, is the **relative** gap each quantization scheme opens up against
it:

| spacing | bits | naive PSNR (dB) | STE PSNR (dB) | Gumbel PSNR (dB) | naive SSIM | STE SSIM |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 1 | 12.93 | 12.56 | 13.21 | 0.263 | 0.245 |
| uniform | 2 | 16.09 | 16.03 | 17.08 | 0.643 | 0.658 |
| uniform | 3 | 21.04 | **21.82** | 18.89 | 0.888 | **0.913** |
| uniform | 4 | **27.37** | 21.75 | 18.83 | **0.973** | 0.923 |
| realistic | 1 | 12.93 | 12.32 | 13.25 | 0.263 | 0.215 |
| realistic | 2 | 13.26 | **15.73** | 16.71 | 0.354 | 0.634 |
| realistic | 3 | 12.82 | **19.78** | 18.36 | 0.305 | **0.862** |
| realistic | 4 | 13.38 | **23.12** | 18.19 | 0.380 | **0.942** |

Two distinct results here:

- **Uniform level set:** naive is competitive and wins outright at 4 bits
  (27.37 dB), roughly ties at 1–2 bits, and STE edges it out at 3 bits
  (21.82 vs 21.04 dB). Much closer contest than the sparse case, but naive is
  still not clearly beaten.
- **Realistic (non-uniform amplitude) level set: naive collapses.** At 2, 3,
  and 4 bits, naive's PSNR stays pinned near 13 dB (SSIM 0.30–0.38) — barely
  better than the 1-bit case, effectively failing to use the extra levels at
  all — while STE reaches 15.7–23.1 dB (SSIM 0.63–0.94) and Gumbel-Softmax
  reaches 16.7–18.4 dB. **This is the gap Choi et al.'s method closes, in our
  units, and it is large: 6–10 dB of PSNR, and roughly triple the SSIM.**

The mechanism is exactly the coupling the rest of this document is about:
naive quantize-after picks the nearest *phase* to an already-optimized
continuous solution and ignores that each level also carries a different,
fixed *amplitude* (Section 4). For a handful of isolated points, amplitude
mismatch barely matters — the points are either bright or they are not. For
dense image content, every pixel's gray level matters, so throwing away the
amplitude information at quantization time is much more costly, and an
optimizer that sees the true complex-valued levels *during* optimization
(STE, Gumbel-Softmax) can route around it by choosing phases that also land
on favorable amplitudes. **Conclusion: naive quantize-after is not
universally sufficient. It saturates the theoretical bound on sparse content
regardless of level spacing, but on dense content with our actual
non-uniform-amplitude device levels, it fails badly and quantization-aware
optimization is necessary.**

(Single run; STE and naive are deterministic given the fixed seed and
reproduced exactly across two independent runs of this sweep. Gumbel-Softmax
draws unseeded per-iteration noise, so its numbers have some run-to-run
variation — but the naive-collapse/STE-recovery gap is large enough, and
present in both runs, that it is not noise-driven.)

## 4. Non-uniform level spacing: its own section, as requested

`design_phase_lut` TMM-corrects the refractive indices of the 300 nm Sb2Se3 /
6-pair TiO2-SiO2 DBR stack (532 nm) so that the *achieved phase steps* are
uniform by construction (confirmed in
`tests/test_tmm_phase_lut.py::test_corrected_lut_phase_steps_are_uniform`,
std/mean < 2%). It does **not** and cannot correct reflectance. For this
stack the 8 corrected levels have power reflectance ranging from **0.583 to
0.946** (field amplitude 0.76–0.97) — a real, device-specific ~20% amplitude
spread with no counterpart in the phase-only SLMs the literature quantizes.

On the sparse target, holding the scheme fixed and comparing `realistic` vs.
`uniform` level sets, naive's efficiency-relative-to-the-sinc-bound drops by a
**flat and remarkably consistent −0.76%** at 2, 3, and 4 bits (0.7% at 2 bits
is close enough to be the same effect within our measurement noise):

| bits | naive/continuous, uniform | naive/continuous, realistic | Δ |
|---:|---:|---:|---:|
| 2 | 0.8101 | 0.8043 | −0.72% |
| 3 | 0.9495 | 0.9424 | −0.76% |
| 4 | 0.9872 | 0.9797 | −0.76% |

That flat penalty comes entirely from the reflectance spread, not from phase
spacing (which is uniform by construction) — it is a small, second-order
amplitude tax, roughly what you would expect from the ~20% amplitude range
producing a modest average light loss, independent of bit depth. **On sparse
content, and with naive quantize-after specifically, level count dominates
this effect by roughly an order of magnitude** (going from 8 levels to 2
costs 40+ points of relative efficiency, not 0.76).

That conclusion does **not** carry over to dense content or to
quantization-aware schemes (Section 3): there, the same non-uniform-amplitude
level set costs naive 6–10 dB of PSNR relative to STE, because dense content
cannot tolerate throwing amplitude information away at quantization time the
way sparse content can. **The right statement is conditional: for sparse,
point-like content optimized naively, non-uniform spacing is a minor,
flat tax dominated by bit depth. For dense content, it is the dominant
source of naive's failure, and it disappears almost entirely once the
optimizer is quantization-aware.**

## 5. 1-bit STE collapse (unchanged — verified against finite differences)

STE collapses to efficiency ≈ 0.0013 — chance level for a 6-point target in a
64×64 field (6/4096 ≈ 0.0015) — at every learning rate tested (0.02–0.15) and
every random seed, on the sparse target. Warm-starting STE from the
already-converged continuous phase does not help (still ≈0.004): the
per-pixel gradient at 2 levels appears to actively destroy a good
binary-quantized starting point rather than refine it, and this was confirmed
with a direct finite-difference check against autograd (both agree, so it is
a genuine vanishing/uninformative-gradient property of binary STE on this
loss, not a backward-pass bug). Gumbel-Softmax does not share this failure
(≈0.31, essentially matching naive) because its noise-driven exploration
keeps per-pixel assignments from freezing at initialization. **Practical
implication unchanged: never wire up a bare STE optimizer for 1-bit phase
without a schedule or noise injection.**

## 6. The Gumbel-Softmax score gain is now derived, not fitted

The first implementation used Choi et al.'s Eq. 6–8 literally:
`score_l = sigmoid(w·delta)·(1-sigmoid(w·delta))`, added directly to standard
Gumbel(0,1) noise, softmax at temperature `tau`. It produced pure noise at
every bit depth (efficiency pinned at the ~1/N² chance level, confirmed by a
direct finite-difference check, not a backward-pass bug). Cause: `x(1-x)` for
`x = sigmoid(...)` is bounded to `[0, 0.25]` for *any* `w` (`w` only narrows
the peak, it never raises it), so the deterministic score can never compete
with unit-variance Gumbel noise, at any temperature, since dividing both
terms by the same `tau` never changes their relative scale.

Per review feedback, the fix is now a **derived** constant rather than a
fitted magic number:

```
noise_std   = pi / sqrt(6)                      # std of standard Gumbel(0,1)
max_score   = 0.25                              # analytic max of x(1-x)
confidence  = 2.0                                # tunable: noise-std multiples wanted
                                                  # at the START of the anneal
GUMBEL_SCORE_SCALE = confidence * noise_std * TAU_START / max_score   # = 20.52
```

The reasoning: at the *start* of the temperature anneal (`tau = TAU_START`,
the most exploration-heavy point), we want the deterministic score signal to
be a mild, learnable bias — comparable to a couple of noise standard
deviations, not so large that it freezes assignments at their random initial
values before any gradient has acted (which is what happens with a much
larger scale, and which reproduces the same chance-level failure by a
different route: an immediately-saturated softmax has no useful gradient
either). At the *end* of the anneal, the same fixed scale is
`TAU_START/TAU_END = 40×` more decisive by construction, which is what makes
the final pick close to deterministic. `confidence = 2.0` is still a choice,
not derived from the paper (which defers this exact tuning to "the
supplement" we do not have) — so it is documented as a choice, with the
scale's *order of magnitude* now justified by the noise distribution's known
variance rather than picked by grid search. **The Gumbel-Softmax numbers in
this document should be read as a demonstrated lower bound on that method's
achievable performance with a principled but not paper-verified scale
constant, not as a refutation of the technique** — Section 3 shows
Gumbel-Softmax comfortably beating naive on the dense+realistic combination,
so the lower bound is already informative.

## What this changes for the device

1. **"3-bit == 95%" is correct, but it is a ratio, not an absolute number,
   and it is now confirmed against a real optimizer, not just an idealized
   grating.** Naive quantize-after-optimize retains 94.95% of whatever
   continuous phase achieves at 3 bits, on sparse content, to within 0.1%.
   Keep citing it, but state it as a retention ratio.
2. **Do not add quantization-aware optimization by default — check the
   content class first.** On sparse, point-like content, naive already
   saturates the theoretical bound; STE and Gumbel-Softmax only add
   optimization noise there. On dense, image-like content — the regime our
   display actually has to render — naive can fail badly (6–10 dB PSNR loss
   in our test) specifically when combined with the real device's
   non-uniform-amplitude levels, and a quantization-aware optimizer (STE or
   Gumbel-Softmax) recovers most of that gap. **This means the answer depends
   on what tier2_slfh actually renders in practice** — if hogel content is
   closer to natural imagery than to sparse point clouds (which is the more
   realistic assumption for this display), quantization-aware optimization is
   not optional.
3. **The realistic level set's amplitude non-uniformity is not a flat, minor
   tax — it is conditional on the optimizer.** With naive quantize-after it
   costs a consistent ~0.76% on sparse content but a severe, optimizer-visible
   6-10 dB PSNR loss on dense content. With a quantization-aware optimizer,
   most of that dense-content penalty disappears (STE loses only ~0-3 dB going
   from uniform to realistic levels at matched bit depth, vs. naive's ~8 dB
   loss). Effort spent flattening per-level reflectance (e.g. an AR coating
   tuned per level) is still a legitimate minor refinement, but pairing the
   real device's levels with a quantization-aware optimizer is a larger and
   cheaper lever.
4. **If 1-bit (binary phase) is ever considered** for cost reasons, treat it
   as fundamentally different from higher bit depths: quantize-after-optimize
   or Gumbel-Softmax, never a bare straight-through estimator.

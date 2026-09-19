# HoloPixel — state of the research

Written for someone arriving cold, including a language model being asked to
help. It records what is measured, what is proved, what is merely believed, and
what is known to be wrong. Numbers here were produced in this repository; where
a figure is an estimate rather than a measurement it says so.

Last updated at the commit that introduced this file. Test suite: **628 passing**.

---

## 1. The goal, and the honest distance from it

Build a near-eye holographic display. A 3D scene goes in; a phase pattern for a
physical phase-only panel comes out; the eye sees a hologram with real focus
cues, not a stereo pair.

**Current cost of one frame, extrapolated from measurements: about 31 minutes.
The budget is 11.1 ms. The gap is roughly 167000x.**

    one phase solve (1 colour, 1 subframe, 1 eye)    51 s
    solves per frame: 3 colours x 6 subframes x 2 eyes   36
    light field generation, both eyes                22 s
    total                                          1856 s

Applying the two levers that have actually been measured -- gaze-contingent
window solving (16x) and warm starting from the previous frame (10x) -- leaves
about **1000x**. Foveating the loss contributes **nothing** to compute, for a
structural reason given in section 6.

That remaining factor is not an engineering problem. It will not come from
better kernels: the FFTs were measured at 50% of the memory roofline, so
implementation tuning is worth perhaps 2x. It has to come from a different
algorithm class -- a feed-forward network in place of iteration is the only
route with a published real-time demonstration.

---

## 2. Architecture, and one decision that shapes everything

    3D scene -> light field -> phase solve -> panel -> propagation -> pupil -> retina

**No depth map anywhere.** The optimiser is driven by a light field alone. This
is the central architectural choice and it is what lets glass, refraction and
view-dependent effects survive: a depth map cannot represent a surface that is
in two places at once. The cost is that the target phase is unknown and must be
optimised rather than assigned.

A consequence proved the hard way (section 7): a light field is spatially
INCOHERENT -- 231 to 256 significant modes out of 289 -- so coherent-mode
decomposition cannot supply the missing phase. Optimisation must.

---

## 3. What is in the repository

| module | what it is |
|---|---|
| `tier1_lightfield/cornell_lightfield.py` | Numba path tracer, 17x17 views, 512^2, 256 spp, 8 bounces, dielectric sphere |
| `tier1_gaussians/render.py` | tiled differentiable Gaussian splat rasteriser |
| `tier2_hfh/operators.py` | the forward operator `P_k`, its adjoint, the diagonal normal operator |
| `tier2_hfh/optimise.py` | the Adam baseline: stochastic pupil sampling, quantisation-aware, multi-subframe |
| `tier2_hfh/local_global.py` | Gerchberg-Saxton and a local/global (PD-style) solver |
| `tier2_hfh/subframes.py` | the multi-subframe amplitude projection |
| `tier2_hfh/hardware_states.py` | projection onto the finite complex states of the real device |
| `tier2_hfh/acuity.py` | retinal geometry, Watson mRGC model, foveation |
| `tier2_hfh/propagate.py`, `display.py`, `modes.py`, `wft.py` | angular spectrum, pupil and defocus, coherent modes, windowed transform |
| `lean/holopixel/` | Lean 4 proofs of the operator algebra, Theorems A-F, no `sorry` |

Literature notes are in `docs/notes_*.md`. **`useful_knowledge.md` is the list of
assumptions that turned out to be wrong** and is the single most useful file for
avoiding repeated mistakes.

---

## 4. The forward operator, and the one structural result worth knowing

For pupil position k:

    P_k = S . F . A . R_k

restrict the panel to the pupil window, apodise, unnormalised DFT, fftshift.

**The normal operator is exactly diagonal:**

    sum_k rho_k P_k^H P_k  =  W^2 . diag( sum_k rho_k A^2 at offset k )

Verified against dense matrices on six geometries, overlapping and disjoint,
weighted and unweighted; proved in Lean (Theorems A-C). So the least-squares
step of a local/global solver is an **elementwise division** -- no conjugate
gradient, no factorisation.

Three riders, each of which has already caused or nearly caused an error:

- It holds for **per-view scalar** weights. It does NOT hold for per-output-sample
  weights, because `F^H W F` is circulant. Measured with a real foveal weight
  map, **51% of the operator norm leaves the diagonal**, and the condition number
  goes from 13 to 70 (10.5 after diagonal preconditioning).
- Diagonality is a property of the **complete** set of DFT rows. A single retinal
  sample's Gram is rank 1 with 96% of its norm off-diagonal, and random subsets
  give 0.29 to 0.66 off-diagonal fraction. **You cannot subsample H.** You do not
  need to: H is closed-form and free, so sample only the right-hand side `b`,
  and the estimator stays unbiased (verified by Monte Carlo, bias falling as
  1/sqrt(trials)).
- The null-space mask is relative to the **current pupil set**. A pixel dead for
  one batch may be alive once the eye moves. Never bake it in.

Lean also surfaced a precondition nobody had stated: **Theorem B needs the
restriction to be injective** -- distinct window rows must read distinct panel
pixels. True by construction here, but untested, and the theorem is false
without it.

---

## 5. Solvers: what is known

Two solvers exist. Both alternate between an exact target-amplitude projection
and an exact panel constraint.

**A correction to an intuition that seemed obvious and was wrong.** With `H`
diagonal and `|u_j| = 1`, the term `u^H H u = sum_j d_j` is CONSTANT on the
feasible set. So the exact constrained global minimiser is `u_j = b_j/|b_j|`,
and "unconstrained solve then normalise" coincides with it -- verified to
2.5e-16. **Both block steps are therefore exact minimisers and the amplitude
loss provably cannot increase** (measured: 0 increases in 400 iterations).

So when the solver stalls it is not damping, not a bug, and not incompatible
steps. It is an exact descent method sitting at a **non-global fixed point**,
which is ordinary behaviour for a nonconvex problem.

Measured on the toy problem of section 9, guaranteed-feasible targets:

| configuration | reaches the global solution |
|---|---|
| local/global, random start | 1 / 12 |
| Gerchberg-Saxton, random start | 11 / 12 |
| local/global, spectral start | **7 / 12** |
| Gerchberg-Saxton, spectral start | **12 / 12** |
| either solver handed the other's stuck result | **16 / 16**, both orders |

Three findings that follow:

- **They never fail on the same instance.** 0 of 16. The failure modes differ:
  Gerchberg-Saxton commits to each view in turn and can lock in a bad early
  choice; local/global averages everything and can settle into a compromise.
- **Spectral initialisation is cheap and large.** The leading eigenvector of
  `P^H diag(a^2) P` by power iteration, using only the existing forward and
  adjoint. Alignment with the answer rises from 0.05 (random is essentially
  orthogonal) to 0.19-0.50.
- **Block Gauss-Seidel beats both ends of the scale.** Splitting the views into
  B groups, exact diagonal solve within a group, updated panel between groups:

      B = 1 (pure Jacobi, the current solver)    1/12
      B = 4                                      8/12
      B = 12                                    12/12
      B = 36 (pure Gauss-Seidel)                11/12

  Note `B = 6` collapses to 3/12, out of trend. The suspected cause is that
  stride-6 grouping on a 6x6 pupil grid puts an entire COLUMN of pupils in one
  block. A random-grouping control was running when this file was written and
  is not yet reported.

**No early predictor of success was found.** F at iterations 10, 25, 50 and 100
separates solved from stuck runs on average (934 against 2509 by iteration 100)
but the distributions overlap at every checkpoint, so no threshold classifies.
Step size points the wrong way. Alignment with the true answer predicts well,
but requires the answer.

---

## 6. Foveation: measured, and mostly negative

**A panel pixel does not correspond to a retinal position.** The pupil's
transform has global support, so every pixel of the window contributes to every
retinal sample. There is no region of the panel that is "the periphery".
Foveation can therefore change the objective but never the cost of an iteration.

That retires two of five gaps an earlier survey listed as opportunities:
varying hogel density with eccentricity (we have no hogels) and varying phase
bit depth with eccentricity (a pixel is neither foveal nor peripheral).

Measured, panel 4096 / window 1024 / 4 subframes / 3 seeds:

| arm | fovea dB | as the eye sees it |
|---|---|---|
| none | 28.21 +- 0.07 | 46.66 +- 0.60 |
| per-sample weight | 29.11 +- 0.48 (+0.90) | 45.52 +- 0.60 (-1.14) |
| blur, Watson curve | 28.40 +- 0.08 (+0.19) | 48.63 +- 0.54 (+1.97) |
| weight + blur | 27.92 +- 0.12 (-0.29) | 46.96 +- 0.58 (+0.30) |
| blur, behavioural MAR curve | 23.37 +- 0.09 (**-4.84**) | 44.18 +- 1.51 (-2.48) |

Four things to take from it. The effects are sub-decibel against **+8.29 dB** for
going from one subframe to four. The two routes trade in opposite directions.
**They do not compose** -- together they are worse than either alone. And the
largest foveation number in the table is **negative**: using the classical
behavioural acuity curve instead of Watson's anatomical one costs 4.84 dB.

What *does* map to panel position is **pupil position in the eyebox**. One pupil
reads 6.25% of the panel, so gaze-contingent window solving is a real 16x. It is
not foveation.

---

## 7. Assumptions that turned out to be wrong

The full list with reasoning is in `useful_knowledge.md`. The load-bearing ones:

1. **Foveate by low-passing the target.** Wrong; weight the loss instead. Blurring
   cost 2.4 to 5.7 dB at the fovea. Each view is blurred about its own centre, so
   the target set becomes mutually inconsistent and no physical field satisfies it.
2. **`MAR(e) = 1 + e/2.3` is Watson's.** It is not -- it is the classical
   Levi/Klein/Aitsebaomo and Rovamo/Virsu M-scaling form with a letter-acuity
   constant. Watson's own model gives 65.4 c/deg on axis against its 30. The
   misattribution was worth 5 dB.
3. **The periphery is colourblind.** A stimulus-size artefact of studies from 1919
   and 1959. With large targets, hue at 45 degrees is fovea-like and colour
   survives to 75 degrees. Nothing inside our 46-degree field can drop a wavelength.
4. **The periphery is the forgiving place, so cut subframes there.** Backwards on
   the temporal axis: critical flicker fusion RISES from fovea outwards and
   survives cortical-magnification scaling. Residual speckle redrawn each frame
   modulates at about 90 Hz, exactly the peripheral ceiling.
5. **A light field has few coherent modes.** An artefact of zero-padding. The real
   answer is 231-256 of 289; the field is incoherent and there is nothing to
   decompose.
6. **The Gaussian rasteriser was arithmetic-bound.** It was launch-bound. The tell
   was a per-primitive cost that did not change between a 4-pixel and a 21-pixel
   splat.
7. **A comparison test can pass by rendering nothing.** One did, and would have
   survived any rewrite of the code it claimed to test.

---

## 8. Hardware

Sb2Se3 phase-change material, 8 levels. **The states do not lie on the unit
circle.** Measured amplitudes, in phase order 0 to 315 degrees in 45-degree steps:

    0.883  0.805  0.764  0.777  0.839  0.921  0.973  0.947

Non-monotonic, dipping at 90-135 degrees. Minimum pairwise complex distance
0.590, so no degeneracy. Two consequences, both measured:

- Quantising by **phase distance** and by **complex distance** disagree on **0.68%**
  of inputs, and when they disagree the worst case is a full 45-degree
  adjacent-state jump, not a rounding difference. On an equal-amplitude level
  set they never disagree.
- In a local/global solver the exact coordinate update is the nearest state to
  **`b_j / d_j`**, not to `b_j/|b_j|`. Those pick different states on **10.46%** of
  coordinates on the real device, and coincide exactly on an equal-modulus set.

Quantisation-aware optimisation is worth 1.74 dB against quantising after the
fact (2.63 dB penalty naive, 0.89 dB aware).

---

## 9. How the solver experiments are set up, and why they are small

The toy problem used for every solver comparison:

| | toy | real target |
|---|---|---|
| panel | 16 x 16 = 256 pixels, 24.4 um across | 8192^2 = 67 million, 12.5 mm |
| pupil window | 6 x 6 = 36 px, 9.1 um | 2048^2, 3.1 mm |
| pupil positions | 6 x 6 = 36 views | 17 x 17 = 289 |
| per image pixel | 3.35 deg | 0.01 deg |
| measurements per unknown | 5.14 | 18.35 |

A run: invent a random phase `u_true`, compute what the eye sees from each pupil,
**keep only the brightness**, hide `u_true`, and ask the solver to recover a
pattern producing those brightnesses. Because the target was built from a real
pattern, a perfect answer is guaranteed to exist -- which is what makes "stuck"
meaningful.

**Three caveats that limit every solver number above.** The toy is 262000x
smaller. Real scenes have **no exact answer**, so "stuck" will mean "best
compromise", and success/failure counting stops being the right metric. And
spectral initialisation is justified by theory that assumes a true signal
exists, which a Cornell box does not provide.

---

## 10. Where the numbers came from

Hardware: RTX 5090 Laptop (20.8 TFLOPS fp32, 601 GB/s), Ultra 9 275HX, 24 cores,
2917 MHz sustained all-core. A `llama-server` holds about 18 GB of the GPU
during much of this work, so GPU figures are from a contended device.

Reproduce: `MPLBACKEND=Agg .venv/bin/python -m pytest` for the suite;
`cd lean/holopixel && lake build` for the proofs, about 10 s with the Mathlib
cache present.

---

## 11. Open, and what I would do next

**Genuinely open:**

- Does any of section 5 survive at realistic scale? Everything is 16x16.
- Real scenes are not exactly realisable. Nobody has measured how the solvers
  behave when the target is out of reach.
- The random-grouping control for the `B = 6` anomaly.
- Whether Kurdyka-Lojasiewicz theory upgrades "the loss converges" to "the
  iterate converges" -- work in flight, and expected to be a rigour exercise
  rather than a practical gain.
- `T = 0.1`, the speckle detection threshold, is an unsourced free parameter and
  is the dominant uncertainty in the display spec. At `T = 0.05` the subframe
  requirement quadruples and the timing margin disappears. It is a literature
  question, not a simulation one.

**What I would do next, in order.** Stop trying to make the iteration converge
more reliably -- reliability is not the binding constraint, three orders of
magnitude are. Measure instead whether a feed-forward network can hit the
quality bar, since that is the only demonstrated route to real-time. Keep the
iterative solver as the ground truth that generates its training data and scores
its output.

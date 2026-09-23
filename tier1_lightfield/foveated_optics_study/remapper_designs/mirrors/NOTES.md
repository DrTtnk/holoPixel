# Off-axis freeform mirror remapper — design notes

Status: FINAL for this session. Evaluator verdict: **REJECTED**. Best design
delivered with honest numbers and a diagnosis below.

## Approach

Two off-axis freeform mirrors (M1, M2), plane-symmetric about world x = 0,
fold the eye's forward view up and back down onto the flat MLA plane. Each
mirror is a base conic (curvature `c`, conic constant `k`) plus an XY
polynomial sag, even in local x (to preserve the x = 0 symmetry), with terms
`(i, j)`, `i` even, `2 <= i + j <= 6` (14 terms). Local coordinates are
normalised by the mirror's own aperture half-size so every polynomial
coefficient is an O(1 mm) quantity (see `src/surfaces.py` docstring) — this
was necessary for a single Adam learning rate to be stable at all; the first
attempt, with raw mm^(1-i-j) coefficients, diverged within 15 iterations.

The whole system (both mirrors' positions/tilts/conics/polynomials, plus the
flat MLA panel's pose) is optimised together in `src/optimize.py` by
differentiable reverse ray tracing in torch float64: for a grid of field
directions over the 70x45 deg FOV, a hexagonal set of pupil points all
travel in that one direction (an infinity conjugate), reflect off M1 then
M2, and are scored against the target panel position from
`scripts/foveation_target.py` (chief-ray position match), their own spread
at the panel (spot size, a blur proxy), and the incidence angle at the panel
(telecentricity, for lenslet acceptance). `src/export.py` tessellates the
final surfaces on a regular local (x, y) grid, clipped to an ellipse, with
exact analytic loop normals; `src/run_design.py` drives export + PNG
diagnostics from a trained checkpoint; `src/verify_geometry.py` checks eye
relief / clearance against the evaluator's own `geometry()` before spending
a Blender run.

## Initial layout

Solved in closed form in `src/layout.py` from one central (on-axis) ray
using the ordinary law of reflection in the y-z (meridional) plane, then
handed to the optimiser as a starting guess, not fixed:
- M1 vertex (0, 21, 0) mm, tilt 48 deg about x (a ~96 deg fold)
- M2 vertex ~(0, 19.1, 17.9) mm, tilt ~ -2.4 deg
- panel origin (0, 14, -9) mm, tilt ~ -10.8 deg

This keeps M2 and the panel outside the pupil's own 70x45 deg forward field
of view (so they do not block the eye's sight of M1): at this starting
layout M2 sits at about 38 deg from the optical axis (pupil-to-M2
direction), the panel at about -27 deg, both outside the +-22.5 deg
vertical half-angle. `layout.py` also derives a paraxial two-mirror
starting curvature (`_paraxial_curvatures`, code retained and tested) from
the thin-mirror equation; a from-scratch optimisation run seeded with it
(`f1=150` -> R1=300, R2=45.4mm) was tried (see "What I tried and rejected"
below) but was far more expensive per iteration (steep sag -> more Newton
steps) without a clear quality win inside the time available, so the
DELIVERED design starts from flat mirrors (c = k = 0) as originally coded.

## Final prescription (opt_state_v3.pt, 1500 Adam iterations)

- M1: vertex (0, 19.74, 1.51) mm, tilt 47.4 deg, R = -297 mm (nearly flat),
  k = -0.19, aperture (measured/exported) half-extents 11.55 x 9.36 mm.
- M2: vertex (0, 18.71, 15.48) mm, tilt -2.46 deg, R = -30.8 mm, k = -0.40,
  aperture half-extents 17.58 x 11.68 mm.
- Both surfaces carry the full 14-term freeform correction on top (max
  |coefficient| 1.32 mm on M1, 2.23 mm on M2 -- i.e. up to a couple of mm of
  extra sag at the aperture edge beyond the base conic).
- Panel pose: origin (0, 15.43, -6.35) mm, basis u = (1,0,0),
  v = (0, 0.99905, 0.04358), w = (0, -0.04358, 0.99905) (a 2.5 deg tilt about
  x). `focal_um` = 45 (unoptimised -- see caveats).
- Mesh: 20,081 vertices / 39,520 triangles per mirror (a 161x161 regular
  grid clipped to an ellipse), `remapper.npz` is 8.4 MB.
- Total remapper volume envelope: M1 spans x in [-11.6, 11.6], y in
  [13.6, 26.3], z in [-5.6, 8.2] mm; M2 spans x in [-17.6, 17.6],
  y in [6.9, 30.3], z in [10.0, 15.5] mm.

## What I verified

- `tests/test_surfaces.py`: reflection law on a hand case (45 deg fold) and
  sign-invariance; a paraxial spherical mirror's focus lands at R/2 to
  micron accuracy; Newton's ray/surface intersection matches an independent
  bisection solve (bracketed around Newton's own root, since a 6th-order
  polynomial surface can have more than one root along a ray -- picking a
  disjoint global bracket is not well-defined for that surface, not a
  weakening of the check); the analytic sag gradient (used for the surface
  normal) matches torch autograd; the local frame is a proper right-handed
  rotation; an even-in-x polynomial gives a plane-symmetric surface.
- `tests/test_export.py`: exported mesh shapes match the evaluator's
  contract, loop normals are unit vectors and match the analytic surface
  normal at the same point, no degenerate triangles.
- `tests/test_design_contract.py`: the full export path (untrained layout)
  passes the shared evaluator's own `validate_surfaces`.
- `src/verify_geometry.py`: eye relief and clearance recomputed with the
  evaluator's own `geometry()` before each Blender run, to catch a
  geometry problem in seconds rather than after a 1-3 minute render.

All 12 tests pass (`cd remapper_designs/mirrors && python -m pytest tests/ -q`).

## Evaluator results (report.json, `python lf_evaluate.py ../remapper_designs/mirrors`)

| check | value | threshold | pass? |
|---|---|---|---|
| coverage | 0.106 | >= 0.98 | **no** |
| ratio median | 0.686 | [0.8, 1.25] | **no** |
| ratio p5 / p95 | 0.252 / 3.02 | [0.67, 1.5] | **no** |
| blur p90 (pitch) | 1.90 | <= 0.75 | **no** |
| fill p10 | 0.361 | >= 0.8 | **no** |
| ghost mean | 0.961 | <= 0.05 | **no** |
| throughput | 0.052 | >= 0.9 | **no** |
| eye relief | 21.97 mm | >= 20 mm | yes |
| clearance | 16.30 mm | >= 15 mm | yes |

**REJECTED.** Only the two purely-geometric packaging checks pass.

## What limited performance / diagnosis

Three compounding problems, in order of how directly I could see each one
in the diagnostics:

1. **Focus (blur).** The torch merit function's own spot-size term
   (RMS position spread over the pupil at fixed field angle) plateaued at
   about 0.3-0.4 mm across most of the field after 1500-3200 total Adam
   iterations across several runs (`figures/sampling_map.png`,
   `figures/spot_diagrams.png`), against a target of ~5-11 um -- 30-80x too
   large. That converts almost directly to the evaluator's blur = 1.9-2.2
   pitches at the field points that get any lens at all (vs. 0.75 allowed,
   0.373 ideal). I do not believe this is a fully-converged optimum: the
   loss was still slowly drifting/oscillating at the end of every run
   rather than flattening cleanly, and raising the spot-loss weight (from
   ~7x to ~21x the position weight, ramped in) did not visibly change the
   floor. My reading is that a 2-mirror, 14-term-per-surface freeform system
   is close to its correction budget for this combination: R = 6 foveation
   (a 6x non-uniform magnification across the field) on top of a 70x45 deg
   FOV from a 4 mm pupil, off-axis, in a package small enough to clear a
   20 mm eye relief. More polynomial terms, more mirrors, or substantially
   more optimisation compute (a proper L-BFGS polish after the Adam warm
   start, which I did not have time to add) are the likely next levers.

2. **Telecentricity -> ghosting.** The incidence angle at the panel
   plateaued around 30-35 deg (1 - cos ~0.17-0.19) against a lenslet
   acceptance half-angle of about 19 deg at focal_um = 45. That alone would
   push most rays onto a neighbour lens's pixels, which is exactly what the
   evaluator's ghost metric (mean 0.96, i.e. 96% of rays land more than 3
   pitches from their lens's own chief direction) shows. `focal_um` was
   left as a free optimiser leaf but I found only after the fact that it
   never entered the loss (see caveats) -- fixing that wiring (or simply
   hand-choosing a smaller focal_um to widen acceptance) is a cheap thing
   to try next, though it would not fix the underlying 30 deg incidence
   angle by itself.

3. **Field coverage is asymmetric, not just blurry.** `figures/per_lens_maps.png`
   (four panels: blur, fill, ghost, sampling ratio, all plotted at each
   lens's own field direction) shows every lens that exists at all sits in
   a band tz in about [-22.5, -8] deg -- i.e. the achieved mapping only
   ever lands panel positions for the LOWER half of the intended 45 deg
   vertical field; tz > -8 deg produces no lenses whatsoever. That matches
   coverage = 0.106 far better than "blur is uniformly bad" would. The x = 0
   mirror symmetry is enforced structurally (even powers of local x only),
   but nothing enforces a symmetric response in elevation (tz), and the
   position-matching loss evidently converged to a solution that folds
   most of the upper field onto panel positions already claimed by other
   directions (or off the 8.176 mm panel entirely) rather than spreading
   evenly. I noticed this only at the very end, from the evaluator's own
   per_lens.npz, and did not have budget left to re-target the optimisation
   at it (a first thing to try: an explicit coverage/uniformity term, or
   just checking whether the panel_target seed biases the fold this way).

4. **Aperture had to be shrunk below what the (poor) solution actually
   uses, which further hurts coverage/throughput.** Sizing the exported
   mesh from the 97th percentile of the ray footprint left one mesh vertex
   9.3 mm from the pupil centre -- comfortably failing the 15 mm clearance
   check outright, because a handful of extreme-corner field points send
   the freeform surface into a badly-behaved region (large coefficients
   evaluated well past the u, v ~ 1 range they were normalised for) that
   folds part of the mesh back toward the eye. Tightening to the 80th
   percentile (`src/run_design.py`) was the loosest cut that passed
   clearance (16.3 mm) and eye relief (22.0 mm) both, checked directly with
   `verify_geometry.py` before spending the Blender run -- but it also
   discards some of the (already small) usable aperture, which is part of
   why throughput (0.052) is even lower than coverage.

## What I tried and rejected

- **A paraxially-curved starting guess** (`layout._paraxial_curvatures`,
  `run(..., f1=...)`): thin-mirror equation gives R1, R2 that image a
  collimated on-axis bundle exactly onto the panel. Evaluating the merit
  function AT that starting point (before any training) gave spot sizes of
  hundreds of mm, not less than the flat start's ~1 mm -- off-axis
  spherical/conic mirrors at this aperture-to-focal-length ratio (roughly
  F/0.6-F/1) have enormous inherent coma/astigmatism tens of degrees off
  axis, so "paraxially focused on-axis" is a poor proxy for "well-behaved
  over the whole 70x45 deg field." A short training run from this start
  reached a similar pos/spot RMS to the flat start within a few hundred
  iterations, but each iteration cost ~10x longer (many more Newton steps
  near the steep initial sag), so I reverted to the flat start given the
  time budget. The code and its motivation are kept (tested indirectly via
  the same surface tests) in case a future session has more compute to
  spend on it.
- **A hard aperture cutoff during optimisation** (masking rays that left a
  fixed local (x, y) box): this makes the masked rays' loss gradient
  exactly zero (a boolean comparison has no gradient in torch), so nothing
  in the loss ever pulled a drifting ray back in -- the valid fraction was
  observed to decay steadily over iterations with no corrective signal.
  Replaced with a smooth `relu(|x| - half_extent)^2` penalty
  (`aperture_excess` in `src/system.py`), which does have a gradient; the
  hard aperture is now used only to size the exported mesh, and even then
  only as a percentile-based, capped estimate (see point 4 above), not a
  training-time mask.
- **A raw-mm polynomial coefficient parametrisation**: diverged within 15
  iterations under a single Adam learning rate, because coefficients for
  different total orders (i + j = 2 vs 6) live in wildly different units
  (mm^-1 vs mm^-5) and a single step size cannot suit both. Fixed by
  normalising local (x, y) by the aperture half-size before evaluating the
  polynomial (see `src/surfaces.py`).

## Honest caveats (not independently re-verified)

- `focal_um` is a leaf handed to the Adam optimiser but is never referenced
  by `loss_and_metrics` or `trace()`, so it never received a gradient and
  stayed at its initial value (45) throughout every run. This was only
  noticed while writing these notes, too late to re-run; it does not affect
  the exported prescription's correctness (45 um is a reasonable, in-range
  value per the prompt's own R = 6 table) but it means focal_um was not
  actually tuned against the telecentricity/blur trade-off it could help
  with.
- The 80th-percentile aperture cut (point 4 above) was chosen from a single
  percentile sweep (50/60/70/80/90/97th) checked only against clearance,
  not re-optimised jointly with the mirror shapes; a smaller, purpose-fit
  aperture chosen before export (or during training) would likely do
  better than post-hoc percentile clipping.
- I did not attempt an L-BFGS or other second-order refinement after the
  Adam runs; given the loss was still oscillating rather than flat at the
  end of every run, it is likely under-converged rather than at a true
  local optimum, and both the spot size and the coverage asymmetry might
  improve substantially with a longer or better-conditioned optimisation
  that I did not have time to run.

## Files

- `design.json`, `remapper.npz` -- the exported design (mirror form only).
- `report.json` -- evaluator output for this design (copied from the
  Blender work dir; the heavy per-view renders themselves are not
  copied here, only into the scratch work directory per instructions).
- `src/` -- `surfaces.py` (freeform mirror + ray tracer), `layout.py`
  (closed-form starting guess), `system.py` (system assembly, loss,
  footprint measurement), `optimize.py` (Adam training loop),
  `export.py` (mesh tessellation), `run_design.py` (export + figures
  driver), `verify_geometry.py` (pre-Blender geometry check),
  `opt_state_v3.pt` (the trained checkpoint that produced this design).
- `tests/` -- `test_surfaces.py`, `test_export.py`, `test_design_contract.py`.
- `figures/` -- `layout_section.png` (y-z section with a ray fan),
  `spot_diagrams.png` (per-field spot diagrams from the torch model),
  `sampling_map.png` (achieved vs. target panel position, position error
  and spot size over the field, from the torch model), `per_lens_maps.png`
  (blur / fill / ghost / sampling-ratio maps over field direction, from the
  real evaluator's `per_lens.npz` -- the one figure built from Cycles
  ground truth rather than the torch proxy model).

# Freeform TIR prism remapper -- design notes

## Approach

Single glass wedge, three optically active freeform patches on ONE watertight
solid (`remapper.npz` has `n_surfaces = 1`, kind `"glass"`, `surf0_mirror_faces`
flagging S2's triangles):

- **S1** (eye-facing): entry refraction on the first hit, total internal
  reflection on the second hit.
- **S2** (back face, mirror-coated): single reflection.
- **S3** (towards the MLA/panel): exit refraction.

Ray order (matches the evaluator's camera rays, pupil -> panel):
`S1 (refract in) -> S2 (mirror) -> S1 (TIR) -> S3 (refract out) -> air -> MLA
flat plane`.

Each surface is a polynomial sag over its own local (x, u) plane, tilted only
about the world X axis (so the whole solid is exactly plane-symmetric about
x = 0, matching the +/-35 deg horizontal field): 15 monomial terms up to
x^6/u^6 (even powers of x only), normalized by a 15 mm aperture scale so
coefficients stay O(1) and gradients don't blow up at the edge of the
aperture (`raytrace.py:APERTURE_MM`). Index 1.9 (near the allowed ceiling,
chosen for TIR/refraction margin headroom given how aggressive the field is).
`focal_um = 43` (mla.radius_for_focal_length_um(43, 1.5) = 21.5 um radius),
the value the task brief itself derives from etendue for R = 6.

### Method (files, in the order used)

1. `raytrace.py`: differentiable torch/float64 ray tracer -- Newton
   ray/surface intersection, vector Snell's law + TIR, mirror reflection.
   Proven in `tests/test_raytrace.py` (6 tests) against an independent
   bisection solve, the scalar Snell law, `asin(1/n)`, autograd surface
   normals, and a textbook flat right-angle-prism TIR fold.
2. First-order layout: random search (`raytrace.Prism.trace` over the full
   70x45 deg field and the 4 mm pupil, flat coeff = 0 surfaces) for tilts and
   apex positions satisfying eye relief and TIR/exit margins, refined with
   Adam to an 8 deg margin. `_phase1_test.py`-style scripts (curriculum:
   conic-only x^2/u^2 terms) then Adam-refine this into a good PARAXIAL
   solution: `_phase1_test.py` -> `/tmp/phase1_adam.npz` (map-error RMS down
   from ~108 deg to ~1.9 deg with conic power alone).
3. Full freeform unlock: `_phase2c_test.py`, warm-started from step 2, all 15
   terms free, mapping + TIR-margin loss only (see "wrong assumptions" below
   for why the spot term was dropped here) -> map-error RMS ~1.15 deg.
4. Eye-relief fix: `_phase5_eyerelief.py`, warm-started from step 3, adds a
   soft constraint on the on-axis eye-relief distance (was 19.7 mm, just
   under the 20 mm minimum) and continues optimizing the mapping at the same
   time -- ended at 20.56 mm eye relief AND a slightly BETTER map-error RMS
   (~1.05 deg) than step 3. This is `params.npz`, the delivered design.
5. `mesh_export.py`: builds the one watertight solid -- three curved
   rectangular patches, ruled-strip side walls connecting matched edges BY
   VERTEX INDEX (not by re-matching positions, so shared edges are exact,
   not merely close), and two fan-triangulated end caps. Winding is fixed
   deterministically against the analytic surface normal (patches) or the
   solid's centroid (walls/caps). Proven in `tests/test_mesh_export.py` (7
   tests): closed (every edge used exactly twice), positive signed volume,
   every individual triangle's winding agrees with its own loop normal (not
   just the mesh-wide aggregate lf_evaluate.py checks), unit-length normals,
   S2's flagged faces actually lie on S2, and the panel_pose basis is a
   proper rotation.
6. `export.py`: writes `design.json` + `remapper.npz` (11165 verts, 22326
   faces, 7200 of them mirror-coated S2 faces).
7. `figures.py`: the own-ray-tracer diagnostics (section, mapping, margin,
   spot-size maps). A 5th figure, `fig_evaluator_maps.png`, is built
   separately from the REAL evaluator's `per_lens.npz` (see Results).

All 13 tests pass (`pytest tests/ -q`, ~55 s).

## Final prescription

Index 1.9, focal_um 43, gap 3 mm (S3 exit to MLA flat face). Surface apex
positions / tilts (world mm, world X = 0 for every origin by construction)
and freeform coefficients are in `params.npz` (`pose`: y1,z1,th1,y2,z2,
th2,y3,z3,th3; `coeff`: (3, 15) per S1/S2/S3, see `raytrace.TERMS` for the
term order) and, fully baked into world-space geometry, in `design.json` +
`remapper.npz`. Solid bounding box: x in [-23.8, 23.8], y in [13.4, 66.4],
z in [-22.1, 34.1] mm.

## Evaluator metrics (report.json, run_2, GPU Cycles, REJECTED)

| check | value | threshold | pass |
|---|---|---|---|
| coverage | 0.301 | >= 0.98 | **FAIL** |
| ratio median | 1.86 | [0.8, 1.25] | **FAIL** |
| ratio p5/p95 | 0.18 / 110 | [0.67, 1.5] | **FAIL** |
| blur p90 (pitches) | 2.13 (median 0.089, ideal 0.373) | <= 0.75 | **FAIL** |
| fill p10 | 0.262 | >= 0.8 | **FAIL** |
| ghost mean | 0.974 | <= 0.05 | **FAIL** |
| throughput | 0.341 | >= 0.9 | **FAIL** |
| eye relief | 20.56 mm | >= 20 mm | pass |
| clearance | 20.41 mm | >= 15 mm | pass |

An earlier run (`run_1`, before the eye-relief fix) failed eye relief too
(19.69 mm) with otherwise similar (slightly worse) numbers; both logs and
`per_lens.npz` are kept under the evaluator work dirs referenced in the
final report, `report.json` here is `run_2`.

## What limited performance -- binding constraint

**The blur is the root cause, and it is not subtle: 1.0-1.3 mm RMS spot on
the MLA plane against a 30 um lens pitch, i.e. roughly 30-40x too large,
essentially everywhere in the field** (`fig_spots.png`, own tracer, matches
the evaluator's ghost = 0.97: 97% of the light landing near a lens's pixels
actually belongs to a neighbour). The median blur reported by the REAL
evaluator (0.089 pitch) looks contradictorily good, but `fig_evaluator_maps.png`
shows why it is not: the set of lenses that get a well-defined "chief
direction" at all collapses from the full 70x45 deg rectangle onto a thin
near-1-D arc hugging the field boundary, plus one thin horizontal band --
coverage is only 30%. With a ~1 mm blur spread over a 30 um pitch, a lens's
pupil-centre "chief direction" is set almost entirely by which of ~1000
overlapping neighbours happens to dominate a 1-pixel bin, so the handful of
lenses that DO get a clean, locally-consistent direction are a biased,
low-dimensional subset, not evidence of good imaging. Every other failing
check (ratio spread, fill, throughput, ghost) is a direct consequence of the
same root cause, not an independent problem.

The MAPPING (foveation density, i.e. dr/dtheta vs `foveation_target`'s R=6
curve) is, by contrast, good: `fig_mapping.png` shows the achieved
landing-radius-vs-eccentricity curve from the own ray tracer sitting almost
exactly on the target curve out to 35 deg, and TIR/exit margins stay >= 5 deg
everywhere in the field (`fig_margin.png`). So the freeform surfaces DID
learn the right paraxial-power-to-eccentricity profile (long "focal length"
on-axis collapsing 6x within about 7 deg, then a near-uniform wide-angle
mapping to the edge) -- optimizing the mapping alone, or the mapping plus an
eye-relief constraint, is a well-behaved, convergent problem with this
15-term-per-surface, 3-surface basis.

Getting the mapping AND a diffraction/aberration-limited spot simultaneously
is not. Every attempt to add a spot-size term to the loss (`_phase2b_test.py`,
`_phase3_test.py` with a slow ramp, `_phase4_test.py` with the pose frozen
and coefficients only) reduced spot RMS by at most ~15% before either the
optimizer diverged (loss and gradients exploding, `valid` fraction collapsing
to ~20-30%) or the mapping error grew far faster than the spot improved
(`_phase4_test.py`: map error grew >5x for spot RMS still stuck near
1.1 mm). This is a genuine capacity limit, not a tuning failure: an R=6
foveation profile needs strong, RAPIDLY ECCENTRICITY-VARYING local power
(the effective focal length falls 6x within the first ~7 deg and then stays
flat), and a low-order (up to 6th) polynomial basis on 3 surfaces does not
have enough independent local curvature control to also correct the
resulting field-dependent aberrations (coma/astigmatism-like terms) across a
70x45 deg field. This matches the brief's own citation: Lyu & Hua's
two-reflection freeform prism reaches only ~1.5x centre-to-edge with a
dedicated (E1 + E2) two-element system; asking one single-solid, one-element
TIR prism for R=6 -- 4x more aggressive still -- and good imaging is, on this
evidence, past what the form can do. A relay or a second powered element
(explicitly flagged as likely necessary in the brief) is the fix; it is
outside this agent's assigned form (one glass solid).

## What was NOT independently verified

- The evaluator's own Cycles render is the acceptance authority and WAS run
  (twice); its numbers above are real, not estimated.
- The own ray tracer's "spot on the MLA plane" is a simplified proxy (chief
  + 6 pupil-ring rays straight to the flat w=0 plane, ignoring the
  microlens's own few-micron sag) -- good enough to catch the >1 mm blur
  problem, but not a substitute for the evaluator's full through-lenslet
  trace.
- No attempt was made to tune `focal_um` away from the brief's suggested 43
  um value; a different lenslet acceptance cone was not explored given the
  spot size is ~30x over budget regardless.

## Wrong assumptions found along the way

- Flat (unpowered) facets preserve ray parallelism, so a bundle of parallel
  pupil rays exits a purely flat-faceted wedge still parallel: the "landing
  spot" on the panel then scales as (total propagation length) * tan(field
  angle), not as the intended few-mm focal length. With ~50-70 mm of fold
  length this put the first flat-facet layout's edge landing radius at
  31 mm against a 4.088 mm target -- ALL of the system's optical power has
  to come from the freeform curvature terms, not the fold geometry.
- `torch.sqrt(torch.clamp(1 - sin2t, min=0.0))` inside a `torch.where`
  TIR/refraction branch has an infinite gradient exactly at `sin2t = 1`,
  and PyTorch still backpropagates that NaN through the unused branch. The
  optimizer is deliberately pushed close to the TIR boundary (that's the
  whole point of a TIR prism), so this triggered immediately with LBFGS's
  aggressive line search. Fixed by clamping the sqrt argument to `min=1e-12`
  instead of `0.0`.
- Un-normalized polynomial coefficients (raw `x_l**6`, `u_l**6` at a ~15-90 mm
  local coordinate scale) gave gradients of order 1e12 for a coefficient
  that was exactly zero -- the monomial's OWN magnitude at that aperture
  size dominates, regardless of the coefficient. Fixed by normalizing
  `x_l, u_l` by a fixed 15 mm aperture scale before the polynomial
  (`raytrace.APERTURE_MM`), so coefficients represent an order-1-mm sag at
  one aperture radius and stay well-conditioned.
- Adding the spot-size (imaging-quality) loss term at the same time as
  unlocking the full freeform basis reliably diverged (three separate
  attempts, described above), even with small learning rates and gradient
  clipping. Optimizing the mapping alone first, unlocking freeform terms in
  a strict curriculum (conic -> full basis), and only then trying (and
  ultimately abandoning) a spot term, was necessary to get a stable,
  reproducible result at all. Chasing spot size and mapping accuracy
  together, for this system, is not a matter of hyperparameters -- the
  design genuinely does not have the freedom to do both.

## Files

- `raytrace.py` -- differentiable ray tracer (Surface, intersect, Snell/TIR,
  Prism.trace).
- `design.py` -- Design class, loss function, mapping_grid/pupil_ring,
  `python design.py` entry point (the actual final run used the `_phaseN`
  scripts below directly for curriculum control; `design.py`'s own
  `--unlock_at` CLI flag implements the same idea in one script).
- `_phase1_test.py` .. `_phase5_eyerelief.py` -- the actual optimization
  runs used to reach the delivered design, in order (see Method above);
  kept as the real record of what was tried, including the ones that
  diverged (`_phase2_test.py`, `_phase2b_test.py`, `_phase3_test.py`,
  `_phase4_test.py`).
- `mesh_export.py` -- watertight solid construction (apertures, edge
  labelling, patch/wall/cap meshing, winding fix).
- `export.py` -- writes `design.json` + `remapper.npz` from `params.npz`.
- `figures.py` -- own-tracer diagnostic PNGs.
- `params.npz` -- final optimized (pose, coeff, index).
- `design.json`, `remapper.npz` -- the export contract deliverables.
- `report.json` -- the evaluator's report for the delivered design (run_2).
- `fig_section.png` -- y-z section with pupil-ring ray fans for 6 fields.
- `fig_mapping.png` -- achieved landing radius vs the R=6 target curve.
- `fig_margin.png` -- TIR margin at S1 and exit-refraction margin at S3
  over the field.
- `fig_spots.png` -- own-tracer spot RMS and per-field validity maps.
- `fig_evaluator_maps.png` -- REAL evaluator per-lens blur/fill/ghost over
  the field (`per_lens.npz` from the Cycles run) -- this is the figure that
  shows the coverage collapse described above.
- `tests/test_raytrace.py`, `tests/test_mesh_export.py` -- 13 tests, all
  passing.

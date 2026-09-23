# Coaxial foveated remapper — design notes

## Form

Rotationally symmetric refractive system, axis = world +Y. Three glass
elements (6 aspheric surfaces) between the eye pupil and the MLA:

- Element 1 (PMMA, n = 1.49): low index -> wide critical angle (42.2 deg),
  chosen deliberately for the front element because it carries the whole
  41.6 deg field and is the one most at risk of total internal reflection.
- Element 2 (N-BK7, n = 1.517)
- Element 3 (polycarbonate, n = 1.585): highest index saved for the element
  closest to the image, where ray angles are small and a higher index buys
  correction power cheaply without TIR risk.

Surfaces are even aspheres (conic + r^4 + r^6) about the Y axis. The MLA's
flat face is the optimisation target plane: the task states the remapper
must image the MLA plane to infinity (all pupil rays at one field angle
converge onto ~one lens), so the design target is the FLAT plane holding the
lenslet apertures, not the pixel plane behind them (that focusing is the
MLA's own job, already handled by mla_design.py/mla_mesh.py).

## Key simplification found in the target itself

`foveation_target.local_focal_mm(theta)` is 33.9 mm on axis but already
flat at 5.6488 mm beyond about 8 deg eccentricity, and stays exactly flat to
the 41.6 deg diagonal corner. That means panel_radius_mm(theta) is a plain
CONSTANT-power f-theta mapping (r = 5.6488 * theta + const) for roughly 80%
of the field, and only the central 8 degrees need the full 6x power ramp.
This reframes the problem: most of the field is an ordinary wide-angle
f-theta lens; the hard part is a smooth, well-corrected 6x magnification
bump concentrated in the fovea, done with the SAME continuous aspheric
surfaces (no separate central lenslet).

## Loss-in-mm already matches the evaluator's blur metric

The evaluator's blur is `spot_rms_rad / target_pitch_rad(theta)`, and
`target_pitch_rad(theta) = LENS_NEIGHBOUR_MM / local_focal_mm(theta)`
while `local_focal_mm(theta) = dr/dtheta`. So a spot's angular blur in
pitches works out to `spot_rms_mm / LENS_NEIGHBOUR_MM` -- independent of
theta (LENS_NEIGHBOUR_MM = sqrt(3) * 17.37 um = 30.09 um is constant). This
means a plain, theta-unweighted mean-squared (or robust) position error in
mm, summed over the field, is already the right merit function: no
per-theta reweighting by local pitch is needed.

## Symmetry: one meridian, full pupil disc

The system is rotationally symmetric about Y, and so is the pupil (centred
on the axis). Reflecting the whole ray path through the plane containing the
Y axis and the field direction is a symmetry of the combined system+field,
so sweeping the field along ONE meridian (az = 90 deg, direction
(0, cos theta, sin theta)) while sampling the FULL 2-D pupil disc already
exercises meridional and sagittal/skew rays for every field azimuth. This
cut the ray-tracing cost by not needing a 2-D (tx, tz) field grid.

## What went wrong first, and the fixes (see also useful_knowledge.md)

1. **Squared position error is not robust to a stray ray.** A ray near a
   critical angle or grazing a steep surface can land tens of mm off target;
   squared error against that dominates the whole batch and the Adam step
   becomes chaotic (observed: loss spikes to 10^5, spot RMS to hundreds of
   mm, within a few hundred iterations). Fixed by using `sqrt(err^2 + eps)`
   (a robust, Euclidean, not squared, distance) as the per-ray loss.

2. **A ray that goes total internal reflection lands at an arbitrary,
   unbounded position; including it in the position loss is noise, not
   signal.** `optics.trace_system` now returns a `clean` mask (no TIR
   anywhere on the path) and the position loss is masked to `alive & clean`
   rays only. Avoiding TIR itself is driven by a SEPARATE, smooth,
   anticipatory regulariser on `sin2t` (the vector Snell's law already
   computes this continuously; it only becomes a discontinuous switch at
   sin2t = 1), penalised once it passes 0.85 (about 20 deg of margin before
   the actual critical angle).

3. **A ray that turns backward (d_y <= 0) after a strong refraction
   produces a "dead" ray (Newton correctly finds t <= 0 downstream) with
   ZERO loss gradient** -- alive/dead is a hard boolean, non-differentiable
   switch, so nothing pulls the optimiser back out once curvature grows
   enough to turn some marginal rays around. Same fix as TIR: an
   anticipatory, smooth penalty on d_y once it drops below 0.5 (60 deg from
   the axis), well before it actually goes non-forward.

4. **Exposing the full 41.6 deg field from a fresh, largely-arbitrary
   surface guess is unstable.** A field curriculum -- optimise at a small
   max eccentricity first (10 deg), then grow it in steps (16, 22, 28, 34,
   41.6 deg), re-optimising fully at each step -- keeps every new field
   increment a small correction to an already-reasonable design instead of
   a large jump from a bad joint optimum.

5. The even-asphere sag derivative was checked against `sympy.diff` (exact,
   `sp.simplify` gives 0) before being hand-coded; Newton intersection was
   checked against independent bisection; the surface normal was checked
   against `torch.autograd` of the same sag function at the exact hit point
   (an earlier version of this test compared normals at MISMATCHED points --
   the ray target used the wrong y-height -- and appeared to fail for a
   real reason that was actually a test bug); vector Snell's law was checked
   against the scalar law and against the law of reflection under TIR; a
   thin biconvex singlet was checked against the paraxial ABCD image
   distance. All in `tests/test_optics.py`. The mesh normals are checked
   against autograd again, at the mesh's own corner coordinates, in
   `tests/test_mesh_export.py`, and the closed/outward-wound check is the
   exact formula `lf_evaluate.validate_surfaces` uses, run locally before
   any Blender call.

## Training method actually used (two more lessons the field-growth curriculum taught)

`design_opt.optimise()` (field-growth curriculum: 10 -> 16 -> 22 -> 28 -> 34
-> 41.6 deg, re-optimising fully at each step) is in the repository but was
NOT the one that produced the delivered design. It reliably degraded once
the field passed about 20 deg: dead+TIR fraction climbed past 50% and never
recovered, because the position loss kept demanding curvature strong enough
to fit the fovea, and that curvature is exactly what causes TIR and
backward-turning rays at a 41.6 deg field edge. Growing the field slowly
did not help escape this -- each new increment just re-discovered the same
trap.

What worked, `design_opt.optimise_feasibility_first()`: fix the FULL field
range from the very start, begin from a deliberately WEAK (gentle
curvature) seed, and first optimise almost purely for FEASIBILITY (no TIR,
no backward/dead rays, everywhere in the field, via the sin2t/d_y/t
anticipatory regularisers below) with the position loss weighted at only
2-5%. Only once every ray in the field is forward-going and TIR-free does
the position weight get raised -- and critically, the regulariser weights
are then held CONSTANT (not decayed) while doing so, because an earlier
version that decayed them while raising the position weight let the design
backslide into the same infeasibility every time.

A third, separate packaging problem only showed up once the shrink-to-fit
mesh exporter was made honest about it: `domain_penalty` (a NEW smooth,
anticipatory regulariser, symmetric in spirit with the TIR/dead ones, but
this time comparing each surface's own even-asphere sqrt-domain limit
against the aperture the FIELD requires it to span, not against any radius
a traced ray actually reached). Warm-starting from the feasible checkpoint
and fine-tuning with this on (`continue_with_domain_safety`) grew the
built, exported clear apertures from about 14/18/14 mm (front to back
element, badly vignetting) to 24/26/20 mm against a 26/49/70 mm
requirement -- still short, particularly for element 3, but the difference
between "coverage 0.0002" and "coverage 0.94" in the real evaluator (below).

## A second mesh-winding bug, found only by the real evaluator's own check

`mesh_export.py`'s cap winding convention ("front cap CCW in (x, z)
projection, back cap CW", copied from `mla_mesh.py`'s own top/bottom
convention) passed every unit test built against gentle, near-flat test
surfaces. It silently FAILED for the strongly curved, strongly conic
surfaces this design actually produced: at small aperture the heuristic
gave the correct (positive-volume, outward) winding, and past some
aperture -- different per element, 15-20 mm here -- it flipped to negative
(inward) winding for the SAME surface, with no change in convention, only
in how far out it was meshed.

The mesh is still a single, closed, consistently-oriented triangulation
throughout (the topological argument -- any polar-grid triangulation of a
single-valued height graph, built with one fixed convention, is
combinatorially consistent -- still holds), so the honest fix is not a
different fixed convention but to stop trusting any fixed convention:
`mesh_export.orient_outward` computes the actual signed volume and reverses
every face (and the matching per-corner loop normals) if it is negative.
This is always valid for a mesh that is already known to be closed, and it
is exactly the check `lf_evaluate.validate_surfaces` itself performs, run
locally before ever calling Blender.

Lesson: a convention borrowed from a working piece of code, verified with
gentle test cases, is not the same as a convention proven for the actual
surfaces in hand. The fix that stuck was to stop asserting an orientation
and start checking it.

## Final prescription

3 elements, 6 even-asphere surfaces, axis = world +Y. Vertex positions,
radii (R = 1/c) and conics from `checkpoint.pt` / `design.json`:

| element | material (n)     | front y (mm) | front R (mm) | front k | back y (mm) | back R (mm) | back k | centre t (mm) | built aperture (mm), FOV wants |
|--------:|------------------|-------------:|--------------:|--------:|------------:|-------------:|-------:|---------------:|---------------------------------|
| 1       | PMMA (1.490)     | 18.00        | -78.19        | 4.27    | 33.13       | -32.03       | -0.71  | 15.13          | 23.85, wants 26.36               |
| 2       | N-BK7 (1.517)    | 41.13        | 226.42        | -2.47   | 54.58       | -35.48       | -1.42  | 13.45          | 26.29, wants 49.24               |
| 3       | polycarbonate (1.585) | 62.58   | 22.02         | -0.95   | 76.81       | 252.55       | -0.86  | 14.22          | 19.95, wants 70.46               |

Track length (pupil to MLA flat face): 91.81 - (-3.6) = 95.4 mm. Panel pose:
`origin_mm = [0, 91.808, 0]`, `basis = [[1,0,0],[0,0,1],[0,-1,0]]` (fixed
coaxial form). `focal_um = 43` (the task's own R = 6 recommendation; not
separately re-tuned -- see "What would fix it" below).

Shape, by eye (`figures/section.png`): element 1 is a strongly negative
meniscus curved toward the eye -- the classic front element of a reversed
fisheye, doing the barrel-style angular compression. Element 2 is weak.
Element 3 is a strong positive field lens close to the image, concentrating
the focusing power near the MLA -- consistent with the design intent of
putting most of the convergent power close to the target plane.

## Evaluator results (`report.json`, real Cycles run, `lf_evaluate.py`)

| check | value | threshold | pass? |
|---|---|---|---|
| coverage | 0.939 | >= 0.98 | **no** |
| ratio median | 0.817 | 0.8 - 1.25 | yes |
| ratio p5 / p95 | 0.485 / 1.769 | 0.67 - 1.5 | **no** |
| blur p90 (pitches) | 2.23 (median 0.98, ideal 0.373) | <= 0.75 | **no** |
| fill p10 | 0.590 | >= 0.8 | **no** |
| ghost mean | 0.981 | <= 0.05 | **no** |
| throughput | 0.858 | >= 0.9 | **no** |
| eye relief | 21.6 mm | >= 20 mm | yes |
| clearance | 21.6 mm | >= 15 mm | yes |

**REJECTED** (2 of 9 checks pass). 51,762 lenses register in the field.

## Diagnosis: what limited performance, read from `figures/per_lens_maps.png`

- **Ghost fraction is ~1.0 almost everywhere**, even where blur (the
  INLIER-only spread) is small near mid-field. This is the key finding: a
  minority of each lens's rays are tightly focused (giving a low blur
  number, since blur only measures the inliers within 3 pitches), but the
  MAJORITY of rays landing in a given lens's footprint come from elsewhere
  -- residual aberration (spherical aberration / coma from a 3-element
  system asked to correct a 4 mm pupil at close to f/1.4 near the field
  edge, per the task's own note) spreads a large fraction of each lens's
  own pupil bundle out past the 3-pitch ghost radius, onto neighbouring
  lenses' pixels.
- **Coverage (0.939) and vignetting past ~37 deg** (`figures/mapping_and_spots.png`,
  ray-survival panel) trace directly to the aperture shortfall above: element
  3 was only built to 19.95 mm against a 70.46 mm requirement, so rays for
  the outer ~15% of the field simply miss it.
- **Sampling ratio has a strong azimuthal (hex-lattice-aligned) pattern**
  near the centre (`figures/per_lens_maps.png`, bottom right) -- likely an
  interaction between residual astigmatism/coma (not azimuthally uniform
  once traced through the discrete hex MLA, even though the underlying
  glass is rotationally symmetric) and the hex grid's 6-fold symmetry, not
  investigated further given time.

## What would fix it, if continued

1. **More correction power**: a 4th element (or splitting element 3 into a
   doublet) specifically to control spherical aberration/coma at the fast,
   wide-pupil edge of the field -- the binding constraint per the task's own
   framing (R = 6 is "far more aggressive" than the 1.5x built literature
   example).
2. **A stronger, earlier domain_penalty**: run `continue_with_domain_safety`
   longer, or fold `domain_penalty` into `optimise_feasibility_first` from
   the start rather than as a late warm-started addition, so the surfaces
   never need shrinking at export time at all.
3. **A genuine 2-D (not meridian + rotational-symmetry-only) merit check**
   against the hex MLA's actual lattice, to see whether the azimuthal ratio
   pattern is a real aberration or a sampling artefact of `per_lens.npz`'s
   pairing logic.
4. **Re-examine `focal_um`**: left at the task's table value (43 um)
   throughout; smaller values widen each lenslet's angular acceptance
   (helps ghost/fill directly) at the cost of coarser pupil sampling per
   lens, and were not explored given time.

## What was NOT verified

- The azimuthal sampling-ratio pattern's exact cause (aberration vs. hex-grid
  sampling artefact).
- Whether a 4th element would in fact fix the ghost fraction, or whether
  the pupil/field/aperture combination is fundamentally too fast for any
  small number of spherical-ish elements (not tested).
- `focal_um` was not swept; 43 um is the task's own table recommendation,
  used as-is.

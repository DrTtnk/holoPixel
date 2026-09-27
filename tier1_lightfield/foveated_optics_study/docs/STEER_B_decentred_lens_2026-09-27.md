# Route B, checked: decentring lens 1 to steer the fovea — 2026-09-27

> **Reviewer note (main session, 2026-09-27).** The coverage drop at 1 mm
> (0.979 -> 0.920) is NOT the eyebox cost of a ~1.5 deg gaze: these runs use
> --optics-follow lateral, which fully compensates the pupil shift (at 20 deg
> gaze it gave coverage 0.995), and at 1.5 deg that shift is only 0.26 mm. The
> loss comes from the decentre itself: it shifts the whole displayed field, so
> one edge of the ellipse leaves the panel. Steering by decentring therefore
> trades field on one side for the moved fovea. On merging, the code was made
> strict: the pancake layout always carries decentre_lens1_mm (0.0 by default)
> and to_batch applies it without a fallback.

## Summary

Decentring lens 1 (and its half-mirror) works and traces, exports, and
renders with the numbers the chief-ray map predicted. The fixed lenslet
array still serves the moved fovea well at 1 mm decentre (gaze steered
-1.54 deg): coverage, blur and ghost stay close to the straight-ahead
numbers, and the per-lens blur and lens-spacing ratio in a 3 deg cone
around the new fovea barely move.

The exporter itself, not the optics, stops at 2 mm: its mesh-building step
(the back surface's Newton solve) leaves the back surface's own valid sag
domain by a small margin (about -0.02 against a 0.05 floor) once the
footprint has moved this far off-axis. The full pupil-sampled ray trace
(no meshing) still keeps every ray alive at 2 mm and starts losing rays
only at 3 mm decentre (95.2 % alive). So the usable range for THIS stored
design and THIS exporter is 0-1 mm (about 0 to -1.5 deg of steering); 2 mm
(-4.4 deg) is an exporter limitation, not a ray-loss failure, and could
likely be recovered with a more robust initial guess in the mesh solver
(see Next steps).

Route B verdict: promising over the small range checked (0-1 mm, 0 to
-1.5 deg), but the exporter needs a fix before the larger, more useful
steering angles (2-4 deg and beyond) can be checked with Cycles.

## Method

1. Added `decentre_lens1_mm` as an explicit key of the pancake layout
   (`pancake_search.layout`/`to_batch`), and `--decentre-lens1-mm` to
   `export_pancake.py`. `to_batch` adds the requested mm to `batch.y` at
   tracer surfaces 0, 2, 3 (the half-mirror, met twice, and lens 1's back);
   surface 1 (the polariser) is untouched. The exporter builds its meshes
   from the very same (decentred) `batch` it traces, so a single value
   drives the tracer, the mesh and `design.json`'s own
   `decentre_lens1_mm` record.
2. Found the steering angle: for `results_pancake/best_pancake_el2_glass_ell100x80.json`
   (rank 0, the field-100x80 ellipse design), traced the chief ray over
   gaze `tz` in [-30, 30] deg and located where it lands on the panel
   centre (`scratchpad/steer_B/find_steer_angles.py`, the same method as
   `scratchpad/gaze/decentre.py`).
3. For decentre 0 and 1 mm: exported the design (`export_pancake.py
   --decentre-lens1-mm <dy>`) and ran the Cycles evaluator
   (`scripts/lf_evaluate.py --pixels 2560 --gaze-deg 0 <tz> --optics-follow
   lateral`) at each design's own steering angle. 2 and 3 mm could not be
   exported or fully traced (see Summary); the report captures why.
4. Compared the per-lens blur and lens-spacing ratio (`per_lens.npz`) in a
   3 deg cone around the panel centre (where the foveal lenslets sit)
   between the 0 mm and 1 mm runs.

All commands used `HOLOPIXEL_FIELD_DEG=100x80 HOLOPIXEL_FIELD_SHAPE=ellipse`
and, for the freeform_mirror scripts, `PYTHONPATH=.:../../scripts`. Results
are under `scratchpad/steer_B/` in this worktree.

## Results

### Steering angle (tracer, chief ray only)

| Decentre | Gaze tz for the fovea | dv/dtz there | Chief-ray alive fraction |
|---|---|---|---|
| 0 mm | +1.06 deg | 25.2 mm/rad | 1.00 |
| 1 mm | -1.54 deg | 24.3 mm/rad | 1.00 |
| 2 mm | -4.38 deg | 22.8 mm/rad | 1.00 |
| 3 mm | -7.73 deg | 19.7 mm/rad | 1.00 |

These match the report's earlier, independent chief-ray numbers for a
similar design (0.5 mm -> -0.2 deg, 1 mm -> -1.5 deg, 2 mm -> -4.4 deg).

### Full pupil-sampled trace (61 pupil points, every field point)

| Decentre | Alive fraction | Exports? |
|---|---|---|
| 0 mm | 1.000 | yes |
| 1 mm | 1.000 | yes |
| 2 mm | 1.000 | no — mesh solver leaves the back surface's sag domain (margin -0.02 vs floor 0.05) |
| 3 mm | 0.952 | no — 4.8 % of rays lost even before meshing |

### Cycles evaluation (2560 px panel, gaze at the steered fovea, optics-follow lateral)

| Decentre | Gaze tz | Coverage | Blur median/p90 (x tolerance) | Ghost | Stray | Throughput | Fill p10 |
|---|---|---|---|---|---|---|---|
| 0 mm (= straight-ahead reference) | +1.06 | 0.979 | 5.59 / 8.90 | 0.0062 | 0.48 % | 0.900 | 1.00 |
| 1 mm | -1.54 | 0.920 | 5.98 / 9.90 | 0.0069 | 0.39 % | 0.853 | 1.00 |

The 0 mm row (gaze at the design's own natural fovea direction,
optics-follow lateral, no decentre) reproduces the stored straight-ahead
reference (coverage 0.979, blur 5.58/8.88, ghost 0.006) essentially
exactly — a useful cross-check that the gaze/optics-follow machinery and
the decentre plumbing agree when the decentre is zero.

At 1 mm the coverage and throughput numbers, which are pupil-turned-eyebox
effects already characterised in `REPORT_gaze_pupil_2026-09-27.md` (a
~1.5-2 deg gaze there costs about as much), account for nearly all of the
change; blur only drifts by 7 %.

### Per-lens metrics in a 3 deg cone around the fovea

The fovea is the group of lenses near the panel centre (tx, tz within 3
deg of 0), the same lenses for every decentre (the panel and the lenslet
array do not move; only the gaze and, physically, lens 1 do).

| Decentre | Lenses in cone | Blur median/p90 (cone) | Blur median/p90 (whole field) | Ratio median/p5/p95 (cone) |
|---|---|---|---|---|
| 0 mm | 4123 | 5.23 / 6.50 | 5.59 / 8.90 | 2.23 / 1.83 / 3.56 |
| 1 mm | 4045 | 5.92 / 7.23 | 5.98 / 9.90 | 2.26 / 1.84 / 3.57 |

The cone's blur and spacing ratio move by 13 % and 1 % respectively
between 0 and 1 mm decentre — far smaller than the field-wide change, and
small in absolute terms. **The fixed lenslet array serves the 1 mm-steered
fovea about as well as it serves the design's own, undecentred fovea.**
(The ratio itself sits above 1 in the fovea for both configs — a
pre-existing characteristic of this variable-focal array's centre, not
something the decentre introduces, since it is nearly identical at 0 and
1 mm.)

## Code changes

- `tier1_lightfield/foveated_optics_study/remapper_designs/freeform_mirror/pancake_search.py`:
  added `DECENTRE_LENS1_SURFACES = [0, 2, 3]` and `to_batch` now reads
  `lay.get("decentre_lens1_mm", 0.0)` and adds it to `batch.y` at those
  surfaces (default 0.0, no behaviour change for existing callers).
- `tier1_lightfield/foveated_optics_study/remapper_designs/freeform_mirror/export_pancake.py`:
  `export()` takes `decentre_lens1_mm=0.0`, folds it into the layout before
  calling `fs.to_batch` (so the mesh and the trace share one batch),
  records it in `design.json`, and a new `--decentre-lens1-mm` CLI flag.

## Tests

`tests/test_decentre_lens1.py` (5 tests, all pass on the CUDA tracer):

- the decentre offsets `batch.y` at tracer surfaces 0, 2, 3 by exactly the
  requested mm, and leaves surface 1 (the polariser) untouched;
- `decentre_lens1_mm=0.0` reproduces the undecentred batch exactly;
- `design.json` records the requested decentre;
- every exported mesh's grid-centre vertex, taken back into its own
  surface's local frame (which subtracts that surface's `batch.y`), sits
  exactly on that surface's own sag equation — for the polariser (surface
  1) and for lens 1's front and back (surfaces 2, 3) — at both 0 mm and
  0.5 mm decentre. This is the strong check the task asked for: it proves
  the mesh, the tracer's batch and the design's own local frame all agree,
  because a bug that decentred one but not the other would make the grid
  centre miss its own analytic surface;
- the polariser's plane (its z in the tracer frame) does not move with the
  decentre, unlike its mesh's footprint (which follows the upstream,
  decentred half-mirror bounce and so is not itself a useful invariant —
  an earlier version of this test wrongly expected the footprint's mean y
  to be exactly unchanged, and had to be corrected).

Ran the full existing pancake test suite (`tests/test_export_pancake.py`,
`tests/test_pancake_search.py`, 45 tests) after the change: all still pass,
no regression.

## Next steps

1. **Fix the exporter's mesh solver so 2-4 mm can be checked.** The
   Newton solve in `export_fold._back_along_front_axis` starts from a
   flat-surface guess along the front surface's local axis; at 2 mm
   decentre that guess alone already falls (barely) outside the back
   surface's own sag domain, before Newton has a chance to correct it
   toward the true (smaller-radius) intersection. A more robust initial
   guess (e.g. project along the surface normal instead of the front's
   axis, or start from the front-surface point itself) would likely
   recover 2 mm and probably more. This is a shared routine
   (`export_fold.py`), so a fix should be tested against the existing
   `test_export_fold.py`/`test_export_pancake.py` suites, not just this
   design.
2. Once fixed, repeat the Cycles evaluation at 2, 3 (and maybe 4) mm to
   see where the per-lens cone metrics actually degrade — the tracer
   alone says full-pupil rays survive to 2 mm and mostly to 3 mm, so the
   optical range may reach further than the exporter currently allows.
3. If the per-lens mismatch grows too large by ~3-4 mm, the task's
   suggestion — searching the design jointly for two decentres (e.g. 0
   and 2 mm) as two configurations sharing one lenslet array — is worth a
   short, cheap prototype before any long search: add a second merit term
   in `pancake_search.py`'s loss that re-traces the same design at a
   second, fixed decentre and penalises the spacing-ratio mismatch there,
   then see whether a short local re-optimisation from the current best
   design already narrows the gap.

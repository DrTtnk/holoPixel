# The 100 x 80 ellipse pancake with a turning eye — 2026-09-27

## Question

When the eye looks away from straight ahead, its pupil turns about the eye's
centre of rotation (13.5 mm behind the cornea, 9.9 mm behind the pupil). The
pupil then moves sideways and tilts. The stored designs assume a fixed pupil
on the axis. How much does the stored 100 x 80 ellipse design
(`results_pancake/best_pancake_el2_glass_ell100x80.json`, rank 0) degrade when
the optics stay fixed and only the pupil moves with the gaze?

## Method

`lf_evaluate.py --gaze-deg TX TZ` turns the evaluator's 61 pupil points about
the centre of rotation so that the eye's axis points along the gaze
(`lf_evaluate.gaze_views_mm`, Listing's minimal rotation). The cameras keep
their straight-ahead orientation (each pupil point is a pinhole that sees
every direction), so all metrics are computed as before, against the
straight-ahead retina. At gaze (0, 0) the renders are bitwise the same as the
stored evaluation.

These numbers therefore measure the **eyebox** only: what the moved pupil
sees. They do not yet judge the new fovea's resolution. The static map
samples the gaze direction 8-19x more coarsely than the fovea needs at
10-30 deg gaze (`docs/research_eye_tracked_foveal_lightfield_2026-09-27.md`),
and that loss comes on top.

## Results (Cycles, 2560 px panel)

| Gaze | Pupil shift | Coverage | Blur median / p90 (x tolerance) | Ghost | Stray | Throughput | Fill p10 |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.979 | 5.58 / 8.88 | 0.006 | 0.48 % | 0.900 | 1.00 |
| 10 deg right | 1.7 mm | 0.925 | 6.62 / 13.67 | 0.012 | 0.41 % | 0.864 | 1.00 |
| 20 deg right | 3.4 mm | 0.862 | 7.69 / 27.43 | 0.074 | 0.59 % | 0.790 | 0.98 |
| 30 deg right | 5.0 mm | 0.798 | 9.42 / 40.04 | 0.208 | 1.87 % | 0.714 | 0.84 |
| 20 deg down | 3.4 mm | 0.900 | 7.24 / 17.03 | 0.013 | 0.25 % | 0.851 | 1.00 |

Straight ahead:

![gaze 0](img_gaze/gaze_0.png)

20 deg to the right:

![gaze 20 right](img_gaze/gaze_20_right.png)

At 20 deg to the right:

- The field beyond about -35 deg on the far side is lost: the moved pupil
  no longer sees it (eyebox vignetting).
- The worst blur (30-40x tolerance) sits at +10 to +25 deg, **where the eye
  now looks**. Straight ahead, the same region is below 10x.
- A ghost area appears at +20 to +30 deg, also at the new gaze.

## Conclusion

From about 10-20 deg of gaze the static design fails in the direction the eye
turns to, from the pupil shift alone. A steered or a larger eyebox is needed
as well as a steered fovea. Most natural gaze shifts stay within about
15-20 deg before the head turns too (see the research note), so the eyebox
target is roughly +-20 deg of gaze, a pupil travel of +-3.4 mm.

## Steered eyebox: the optics follow the pupil

`--optics-follow lateral|centre` moves the whole optics (panel, lenslets,
pancake) with the pupil centre, sideways or in 3D; in the evaluator that is
the same as moving the pupil points back by that shift.

| Gaze | Optics | Coverage | Blur median / p90 | Ghost | Stray | Throughput |
|---|---|---|---|---|---|---|
| 20 deg right | fixed | 0.862 | 7.69 / 27.43 | 0.074 | 0.59 % | 0.790 |
| 20 deg right | follow sideways | 0.995 | 5.37 / 9.69 | 0.015 | 0.51 % | 0.945 |
| 20 deg right | follow in 3D | 0.979 | 5.19 / 9.38 | 0.010 | 0.51 % | 0.901 |
| 30 deg right | fixed | 0.798 | 9.42 / 40.04 | 0.208 | 1.87 % | 0.714 |
| 30 deg right | follow sideways | 0.998 | 5.18 / 10.41 | 0.026 | 0.58 % | 0.976 |
| 30 deg right | follow in 3D | 0.979 | 4.82 / 9.49 | 0.014 | 0.54 % | 0.903 |

- Following the pupil centre gives back the straight-ahead numbers at 20
  and 30 deg: the pupil's tilt does not matter. Sideways following alone is
  nearly as good (ghosts a little higher).
- **Actuator specification for the eyebox:** a sideways travel of
  9.9 mm x sin(gaze): 1.7 / 3.4 / 5.0 mm at 10 / 20 / 30 deg, settled inside
  the post-saccadic window. For 3.4 mm in ~60 ms that is ~60 mm/s on average.
- This does not steer the fovea: the dense sampling stays straight ahead.
  Rotating the optics about the eye's centre of rotation would move both the
  eyebox and the fovea (by symmetry it is the same as gaze 0 for the eye),
  but it moves a much larger mass along a larger path.

## Steered fovea: first numbers

**A. Rotate the whole module about the eye's centre of rotation.** For the
eye this is the same as gaze 0 (the numbers above, straight ahead). The glass
of the stored design is ~23 g (two lenses, 43 and 26 mm across, n = 1.9,
density assumed 5.0 g/cm^3) at 29-37 mm from the centre of rotation; with
~2 g of panel and ~10 g of housing (assumed) the inertia is ~4e-5 kg m^2.
Bang-bang moves: 10 deg in 50 ms needs 11.5 mN m and 0.08 W peak; 20 deg in
60 ms needs 16 mN m, 0.19 W and a peak of 670 deg/s; 30 deg in 80 ms needs
13.5 mN m. A slower move over the post-saccadic window (~150 ms) needs ~6x
less torque. The hard parts are mechanical: a remote centre of motion 13.5
mm inside the eye, ~10 mm of sweep at the front lens edge next to the face,
the reaction torque on the head, and two of everything.

**B. Decentre one pancake lens** (tracer, chief rays; the panel centre, where
the foveal lenslets are, is seen at +1.06 deg straight ahead):

| Element moved up | 0.5 mm | 1 mm | 2 mm | 4 mm |
|---|---|---|---|---|
| lens 2: panel centre seen at | +1.2 | +1.4 | +1.7 | +2.1 deg |
| lens 1 + half-mirror: panel centre seen at | -0.2 | -1.5 | -4.4 | -12.6 deg |
| lens 1: magnification there | 0.99 | 0.97 | 0.91 | 0.58 (3 % of rays lost) |

Decentring the half-mirror lens (~12 g) steers the fovea by ~2.5-3 deg per mm
up to about +-5 deg, with 9 % less magnification; beyond that the steered
fovea soon gets coarse. Not yet checked: the variable-focal lenslets are
matched to the undecentred map (a Cycles check needs exporter support for a
decentred lens). The eyebox translation is still needed on top.

**C. A steered foveal inset** (a second display and a combiner, as in Kim et
al., Foveated AR, 2019) reaches any gaze but changes the architecture.

## Next

- Choose between A, B (small gaze angles only) and C.
- A gaze-aware search: include shifted pupils in the merit, to see how much
  eyebox the fixed optics can give without an actuator.

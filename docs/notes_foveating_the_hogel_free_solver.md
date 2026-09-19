# Foveating a hogel-free holographic solver: measured, and mostly negative

What this answers: our own survey
(`docs/research_foveated_holography.md`) records that every published
foveated-CGH paper reports image quality and none reports a compute or
wall-clock saving attributable to foveation. This note measures what
foveation is actually worth in *our* architecture, and finds it is small
for a structural reason.

All numbers: panel 4096, window 1024, pitch 1.524 um, 4 subframes, 2000
Adam iterations, 2 views per iteration, three seeds, green only. "Fovea"
is PSNR over the central 2.5 degrees, averaged over three pupil
positions. "As the eye sees it" is PSNR after both images pass through
the Watson acuity blur.

## 1. The structural reason foveation is weak here

A panel pixel does not correspond to a retinal position.

The eye's lens Fourier-transforms whatever the pupil admits, so every
pixel of the 1024-sample window contributes to every retinal sample. The
support is global. There is no region of the panel that is "the
periphery", so there is nothing to coarsen, thin or quantise more
roughly.

This retires two of the five gaps our survey listed as opportunities:

- *vary hogel density with eccentricity* -- we have no hogels, and the
  panel is uniform by construction;
- *vary phase bit depth with eccentricity* -- a panel pixel is neither
  foveal nor peripheral, so there is nothing to assign a depth to.

What foveation can still change is the OBJECTIVE: where the residual
error lands, and how fast the solver gets there. It cannot change the
cost of an iteration.

What panel position does map to is pupil position in the eyebox. One
pupil reads 6.25% of a 8192-sample panel at a time, so gaze-contingent
solving of the correct window is a 16x lever -- spatial, real, and much
larger than anything below. It is not foveation.

## 2. Two routes, and they are not the same thing

**Weighting** keeps the full-detail target and multiplies the per-sample
squared error by the midget ganglion cell sampling rate, so the fovea
counts for more. This is Chakravarthula et al. (arXiv:2108.06192,
Equations 8 and 9). `acuity.foveal_weight`.

**Blurring** replaces the target with what the eye can resolve, through a
space-variant Gaussian pyramid. `acuity.foveate`.

They are not two routes to the same place. Weighting reallocates the
optimiser's effort; blurring changes what it aims at.

## 3. Results

| arm | fovea dB | as the eye sees it |
|---|---|---|
| none | 28.21 +- 0.07 | 46.66 +- 0.60 |
| weight | 29.11 +- 0.48 (**+0.90**) | 45.52 +- 0.60 (-1.14) |
| blur, Watson curve | 28.40 +- 0.08 (+0.19) | 48.63 +- 0.54 (**+1.97**) |
| weight + blur | 27.92 +- 0.12 (**-0.29**) | 46.96 +- 0.58 (+0.30) |
| blur, behavioural curve | 23.37 +- 0.09 (**-4.84**) | 44.18 +- 1.51 (-2.48) |

Four things worth stating plainly.

**The effects are sub-decibel to two decibels.** For scale, going from
one subframe to four is worth 8.29 dB at the fovea in the same setup,
and one to sixteen was worth 18.3 dB earlier. Foveation is not in the
same league, and section 1 says why.

**The two routes trade in opposite directions.** Weighting buys the
fovea and spends the periphery; blurring does the reverse. Which one
"wins" therefore depends entirely on the metric, and a metric weighted
by the same map as the loss would be circular. The foveal-region figure
is a region selection rather than a weighting, so it is the honest
headline for the weighting arm.

**They do not compose.** Weight plus blur is worse than either alone on
both metrics, and worse than doing nothing at the fovea. The two
mechanisms discount the periphery twice over, and the shared panel phase
is what pays for it.

**The largest foveation-related number in the table is negative.** Using
the classical behavioural acuity curve instead of Watson's anatomical one
costs 4.84 dB at the fovea -- eight times larger than the benefit of
using the right curve. The two curves differ by only about half a
pyramid level from 1 degree outward, and the fovea's target is identical
under both. The damage travels entirely through the shared panel phase.

## 4. Why over-blurring damages the fovea

The fovea's target is byte-identical under both curves: at window 1024
the grid is coarser than the retina on axis, so the level clamps to zero
either way. Yet the fovea comes out 4.84 dB worse. The only channel
between the periphery and the fovea is the one panel phase they share.

The likely mechanism is that a foveated target set is not physically
realisable. Each view is blurred about its OWN centre, so the same scene
point is asked to be sharp in one view and blurred in another. No single
field satisfies that. The unblurred light field is realisable by
construction, because a real scene produced it. Blurring makes the
target easier to score and harder to achieve, and the residual
inconsistency lands wherever the solver has least slack.

This also predicts the interaction in section 3: weighting concentrates
effort exactly where the inconsistency is worst.

## 5. What we did not model

Aberrations. The hard circular pupil in `Geometry.aperture` already
produces the correct diffraction-limited Airy PSF for our 3121 um pupil,
whose Gaussian equivalent is sigma = 0.4254 lambda / D, or 0.249
arcminutes -- fitted numerically to the Airy core, RMS 0.019.
Chakravarthula et al. use 0.6 samples, fitted to 100 real eyes, which is
larger because it includes aberrations.

Adding them is cheap: convolving the retinal field by a Gaussian is the
same operation as apodising the pupil by a Gaussian, so it is a change
to the aperture, not an extra convolution. It is deliberately not done
here, because a blurrier eye model makes the display's job easier and
that is a change worth making on purpose rather than by accident.

## 6. Honest limits

- One scene, one wavelength, one panel scale, three seeds.
- The weighting arm's spread is large (+-0.48 dB against +-0.07 for the
  others) and one of its three seeds is an outlier, so +0.90 dB rests on
  about three standard errors, not on a clean separation.
- Iterations-to-quality, which is the number our survey says nobody has
  published, is not measured here. Quality at a fixed iteration count is
  measured, which is exactly what everyone else already reports.
- The weight exponent is a free choice. Chakravarthula use the sampling
  rate itself; the density, its square, would be more aggressive and is
  untested.

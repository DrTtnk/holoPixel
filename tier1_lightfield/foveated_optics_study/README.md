# Retina-matched foveated light-field optics study

This directory records the September 2026 exploratory optics study for a tracked, foveated near-eye light-field architecture in HoloPixel.

The core idea is to **stop spending light-field samples uniformly**. A tracked eye gives two independent opportunities to compress the 4-D light field:

1. **Retinal foveation:** allocate angular/spatial field samples according to human retinal sampling density rather than holding foveal resolution over the entire FOV.
2. **Tracked pupil steering:** steer the useful bundle onto the known pupil instead of illuminating a large static eyebox.

The reference parameterization at the pupil plane is

\[
L(x_p,y_p,\theta_x,\theta_y),
\]

where `(x_p, y_p)` are pupil coordinates and `(theta_x, theta_y)` are visual-field directions. The retinal model constrains the latter pair. Pupil/view sampling is a separate light-field dimension.

## Current design point

The working thought experiment uses:

- field of view: **70 deg x 45 deg**;
- fixed, tracked pupil: **4 mm diameter**;
- eye relief: **20 mm**;
- nominal display pixel pitch: **4 um**;
- retinal sampling based on the Watson (2014) midget-RGC model;
- pupil views that become denser with eccentricity rather than staying uniform;
- global steering assumed perfect for now;
- the eye is intentionally treated as fixed in space so that pupil swim, eye rotation and torsion do not contaminate the first optical study.

The first-order retina-matched field budget over 70 x 45 deg is about **338k field samples**, compared with roughly **53.8M** samples if foveal sampling were maintained uniformly over the same FOV. A naive 7 x 7 pupil grid therefore landed close to a 4096 x 4096 display budget, but diffraction showed that a fixed 7 x 7 view count is not physically sensible at the fovea.

The more useful result is a **4-D foveation rule**: high retinal spatial resolution should use a large fraction of the pupil per view; farther from the fovea, where retinal spatial bandwidth falls, the pupil can be partitioned into more independent views. This is the phase-space/etendue trade-off appearing directly in the design.

## Main findings

### 1. Retina-matched angular sampling is extremely valuable

Using the Watson model as a fast retinal ROM reduces the 2-D field-sample count by roughly two orders of magnitude compared with uniform foveal sampling. The exact number depends on the ON/OFF mosaic interpretation and on how aggressively the optical cutoff is enforced.

### 2. Steering is not only an eyebox convenience

With a fixed static 8 x 8 mm support and a 4 mm pupil, a crude geometric capture ratio is only about 20%. Perfect steering of a 4 mm support onto the pupil removes that structural geometric loss in the idealized model and also reduces the number of pupil samples that must exist simultaneously.

This does **not** imply 100% total optical efficiency: Fresnel loss, fill factor, polarization loss, absorption, clipping, aberrations and real coatings remain.

### 3. A fixed 7 x 7 pupil lattice fails the strict foveal wave-optics test

For incoherent light, splitting the pupil into many independent sub-apertures reduces the spatial bandwidth of each view. The simulations and the analytic circular-aperture MTF model both show that the fovea cannot simultaneously demand the finest Watson spatial pitch and seven independent pupil samples per axis through a 4 mm pupil.

The physically cleaner design is therefore

\[
K = K(\theta_x,\theta_y),
\]

where the number of pupil views increases with eccentricity.

At 550 nm, under an MTF50-at-local-Nyquist criterion, seven samples per axis become supportable only several degrees away from the fovea, with the threshold depending on retinal meridian.

### 4. Circular pupil geometry favors hexagonal pupil packing

A square 7 x 7 grid has 49 nominal points, but only 29 lie inside a 4 mm circular pupil when the endpoints span +/-2 mm. Edge-centered views also lose a large fraction of their energy outside the pupil.

Hexagonal packing is a better fit. The exploratory 28 um encoder model compared 7, 19, 37 and 61-view hex packings; 19-37 views looked like a more useful engineering region than 61 for that simple encoder because crosstalk rises quickly as spacing shrinks.

### 5. The 2K x 2K Display is a useful prototype, not the full target

With 7 x 7 physical subpixels per lenslet, a 2048 x 2048 screen provides 292 x 292 lenslets (2044 active pixels per side), i.e. about **85k field samples**. That is only about one quarter of the 338k retina-matched 70 x 45 deg target and gives a substantially coarser foveal pitch.

It is still useful for optical prototyping, validation of steering, remapping, crosstalk, calibration and Blender/mechanical work.

### 6. Two optical architectures remain worth exploring

**A. Identical hex microlens array + global remapper**

- simpler local micro-optics;
- global freeform/reflective element carries the retina-matched angular warp;
- likely the friendlier path to a first prototype.

**B. Tiled freeform hex array**

- each hex tile carries local chief-ray tilt and potentially local power/aperture behavior;
- distributes the remapping function across the array;
- attractive end-state, but fabrication, metrology and tolerance control are substantially harder.

The preview renders in `renders/` show both interpretations.

## Directory layout

- `docs/ARCHITECTURE.md` - physical interpretation and current assumptions.
- `docs/RESULTS.md` - consolidated numerical results and caveats.
- `docs/NEXT_STEPS.md` - recommended local simulation/fine-tuning plan.
- `docs/REFERENCES.md` - papers used as anchors for the ROMs.
- `scripts/retina_model.py` - Watson retinal-sampling implementation.
- `scripts/phase_space_budget.py` - field/pupil budget and diffraction-aware view-count ROM.
- `scripts/waveoptics_encoder.py` - scalar wave-optics stress test for a simple pupil encoder.
- `scripts/blender_hex_project.py` - Blender scene generator starter for both hex-array variants.
- `results/` - small CSV snapshots from the exploratory runs.
- `renders/` - compressed preview renders; regenerate full-resolution diagnostics locally.

## Important modeling boundary

The rendered "optics-aware" pinhole images are **diagnostic ROMs**, not a finished optical prescription. Crosstalk and diffraction are represented using first-order scalar models and local blur/mixing proxies. A real relay/freeform prescription must be propagated end-to-end before treating the image quality predictions as design numbers.

## Local handoff

The next execution/fine-tuning pass is intentionally expected to run on a local workstation. Start from the scripts in this folder, increase sampling only after profiling, and keep the geometric and wave-optics models separate so that performance and physical assumptions remain inspectable.

The current study strongly suggests that the right target is not "a uniform light field with foveated rendering" but a genuinely **foveated 4-D phase-space allocation**.

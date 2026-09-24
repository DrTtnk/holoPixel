> Written by a Sonnet subagent on request (brainstorm), 2026-09-24; kept verbatim. Numbers it quotes from the repo were recomputed by it; the literature list was not re-checked.

# Compact, cheap, eye-matched HMD architectures — brainstorm

Panel: 1.03" micro-OLED, 2560x2560, 7.2 um pixels, 18.432 mm square, 1800 cd/m2, $399, one per eye.
Fovea target 0.46 arcmin, 35 deg target 9.98 arcmin (Watson), Airy floor 0.58 arcmin @ 4 mm pupil.
Recomputed from the repo's own scripts (`retina_model.py`, `foveation_target.py`): F0 = 117.8 mm,
F(35deg) = 5.41 mm (ratio 21.8), lenslet focal 1.06 mm (fovea, f/29) -> 48.7 um (edge, f/1.35),
~237,700 lenses fit the panel against 337,928 retina-matched samples needed over 70x45 deg (panel
undersamples periphery by ~1.4x even at "2x retina pitch" everywhere). A central 10-deg-radius
(20-deg-diameter) circle — 10% of the 70x45 deg field's area — already contains 185,515 of the
337,928 samples, 55% of the total. That number drives several ideas below.

## 1-15: architecture ideas

**1. Variable-focal lenslet array (the planned next step).** Same hex pitch (36 um), locally varying
sag so f follows F(theta): ~1.06 mm at center down to 48.7 um at the edge. One printed part, no
separate remapper. *Numbers:* fovea ~2x retina pitch (0.92 arcmin), edge ~2x retina pitch (~20
arcmin), FOV 70x45 deg, thickness ~1-2 mm, weight negligible (<1 g optics), cost = one resin print.
*Risk:* edge lenses at f/1.35 sag approach a hemisphere — near the practical floor for UV-resin
vat printing (surface tension, minimum radius vs. pixel size of the printer); a 22x focal-length
range on one flat substrate is a big ask for uniform cure/shrinkage. *Prototypable:* yes, this is
the current plan; it just isn't built yet.

**2. Two-tier panel: foveal inset (this panel) + cheap wide periphery.** Use the whole 2560x2560
panel undistorted, through a simple ~52 mm singlet magnifier (half-panel/tan(10deg)), for the
central ~20 deg only — no lenslets, no remapper. A second, cheap low-res panel (e.g. a $20-50
generic 720p microdisplay) with a simple wide-angle lens covers the periphery, combined via a
birdbath/pellicle or canted side-by-side like other wide-FOV HMDs. *Numbers:* fovea 2560/20deg =
128 px/deg = 0.47 arcmin/px (matches target almost exactly, no lenslet losses); periphery only
needs 152,413 samples (45% of budget) over 90% of the field area, so a coarse panel suffices; FOV
70x45 deg; weight +10-20 g; cost ~$450-500/eye. *Risk:* visible seam/luminance mismatch at the tile
boundary, combiner throughput (pellicle ~50%, birdbath ~25-40%). *Prototypable:* yes, easiest of
the radical options — the fovea half needs no new optics, just a single printed or COTS eyepiece.

**3. Eye-tracked steering of a small high-res patch.** Keep one panel, but only render/illuminate a
~20 deg window around gaze, and steer that window's optical axis (liquid lens / MEMS mirror) to
track the eye instead of covering the whole FOV in hardware as idea 2 does. *Numbers:* same fovea
numbers as idea 2, but only ~half the panel is ever "hot"; steering must cover +/-35 deg eye
rotation and beat saccade speed (~4-500 deg/s peak). *Risk:* steering latency vs. the ~50-100 ms
saccadic-suppression window is unforgiving; adds a moving part. *Prototypable:* optically yes with a
printed cam/linkage; the control loop, not the resin optics, is the hard part.

**4. Varifocal fovea + light-field periphery (uses your own K(theta) result).**
`hex_pupil_packing_waveoptics.csv` already shows K=1 (a single, near-full-aperture view) is the only
option near the fovea under the fine Watson pitch, while K=7..29 becomes supportable only past
~1.5-8 deg. The fovea gets no useful parallax from this panel anyway — so render it as sharp 2D with
a fast liquid-lens varifocal singlet (Half Dome/Butterscotch style) for accommodation, and spend the
lenslet budget on periphery-only light field, where K=19-29 packing is wave-optically legitimate.
*Numbers:* fovea ~60 PPD (Meta's Butterscotch figure), periphery 19-29 views/lenslet at the same
5.4 mm edge focal length. *Risk:* varifocal adds a moving/liquid element (~10-20 ms response).
*Prototypable:* varifocal singlet yes (COTS liquid lens, ~$30-80); periphery reuses idea 1's MLA.

**5. Digital/firmware foveation (no new optics).** Keep a uniform lenslet array, but bin neighboring
subpixels together in the periphery (2x2, 3x3) so fewer independent samples are actually rendered,
even though the optics sample uniformly. *Numbers:* cuts render/driver bandwidth roughly in
proportion to the retina/uniform sample ratio (~160x for the full budget) without touching the
optics. *Risk:* saves compute/power only — doesn't fix the core problem (coaxial is blurry, fold
doesn't foveate). *Prototypable:* trivially; a rendering change layered on any other idea.

**6. Curved/tilted lenslet substrate.** Mold or print the hex array on a shallow spherical/freeform
(not flat) surface centered near the eye's rotation point. Targets finding (a) directly: the coaxial
5-lens stack's off-axis blur (2.8x center, 8x edge) is largely field curvature/coma from a flat
lenslet plane. *Numbers:* same pitch/FOV as idea 1, potential 2-4x edge-blur reduction if curvature
removes most off-axis aberration (a hypothesis to render-test, not measured). *Risk:* lenslets can
no longer sit on the flat OLED cover glass; needs an air gap + curved carrier, adding an alignment
stack. *Prototypable:* yes over a printed curved mandrel, but harder than idea 1's flat part.

**7. Foveation folded into the freeform mirror itself.** Instead of a uniform-power fold mirror
(finding (b), ratio ~1.5) plus a separate lenslet remap, give the mirror a strong radial power
gradient (higher-order aspheric/Zernike terms) so R approaches the needed ~22 on the mirror alone,
keeping a plain microlens array on the panel. *Numbers:* would remove the panel-side variable-focal
fabrication risk entirely if it worked. *Risk:* a 22x curvature gradient on one smooth surface is a
lot to ask (the existing optimizer only reached ~1.5x); a partial factor (3-5x) before ripple or
manufacturability breaks down is more likely. *Prototypable:* yes, same resin/mirror process
already in use — re-run the freeform optimizer with a foveation penalty term added.

**8. Diffractive/kinoform lenslets instead of refractive sag.** Replace the steep f/1.35 edge
lenslets' deep refractive sag with a blazed diffractive (Fresnel/kinoform) profile: same power in
much shallower relief, easing the resin sag/aspect-ratio limit flagged in idea 1. *Numbers:*
single-layer kinoform efficiency is typically ~85-95% in-band (order of magnitude only) against a
micro-OLED's ~20-30 nm FWHM, worst at the edge where power is highest. *Risk:* chromatic dispersion
competes directly with the retina-matched blur budget; needs per-color kinoform design or
achromatized doublets, adding a layer back. *Prototypable:* resin prints coarse kinoform rings
(tens of um), but the sub-um blaze steps needed for high efficiency exceed typical resin-printer
resolution — likely needs grayscale lithography, i.e. no longer "cheap."

**9. Pancake-fold the foveal relay only.** Use a polarization pancake fold just for the small,
high-power central relay (idea 2/4's inset), leaving the periphery a simple single-pass lens.
*Numbers:* pancake folding typically loses 65-75% of light per pass and wants 5,000+ nit source
brightness — this 1800 cd/m2 panel is 3-8x too dim for a full-eye pancake system, maybe adequate
for a small-FOV foveal-only fold if losses are contained. *Risk:* efficiency, plus a second distinct
optical technology stacked on an already two-tier design. Discard-or-defer given the brightness
shortfall. *Prototypable:* partially — polarizer/QWP films are COTS, but 1800 cd/m2 is a real ceiling.

**10. Time-multiplexed multifocal sweep.** Trade angular (light-field) multiplexing for temporal:
sweep a fast liquid lens through N focal planes per frame in sync with the panel, so one moderate-res
panel builds a multifocal volume the retina integrates instead of lenslet-split parallax.
*Numbers:* N=3 needs panel+lens at 3x60=180 Hz to avoid flicker; periphery's high flicker
sensitivity (vs. low spatial acuity) sets the refresh floor, a real risk if underestimated. *Risk:*
liquid-lens settling (~10-20 ms) vs. the ~5.5 ms/plane budget at 180 Hz is tight. *Prototypable:*
COTS liquid lenses, not resin-printable; software/driver-heavy.

**11. Exploit color/chroma subsampling in the periphery.** Peripheral cone density (especially
S-cones) falls off faster than luminance sensitivity, so share color samples across 2-3 neighboring
lenslets off-axis (gaze-fixed chroma subsampling, like JPEG 4:2:0). *Numbers:* could cut periphery
subpixel/routing count ~2-3x at the same lenslet pitch — optics unchanged, only what's rendered.
*Risk:* colored fringing if not gaze-locked precisely; needs eye tracking to stay valid off-axis.
*Prototypable:* rendering-only, layers onto any optical architecture.

**12. Exploit Stiles-Crawford apodization to relax edge-of-pupil tolerances.** Photoreceptor
directionality makes rays entering near the pupil rim roughly 0.5x as effective as central rays
(approximate, from memory — verify before relying on it), so lenslet/remapper aberrations
concentrated at the marginal pupil zone are perceptually discounted for free. *Numbers:* could
justify relaxing edge-of-aperture MTF specs ~0.3-0.5 log units in the optimizer, easing the f/1.35
edge-lenslet design. *Risk:* a soft perceptual credit, not a hard limit — treat as a 20-30% margin,
not a free pass. *Prototypable:* yes, just an optimizer weighting change.

**13. Non-uniform panel pixel use (hardware-free spatial foveation).** Use more OLED pixels per lens
near the fovea (finer sub-lens anti-aliasing) and fewer per lens in the periphery, instead of
changing lenslet pitch. *Numbers:* pixels-per-lenslet ratio is fixed by geometry (5 px / 36 um), so
this only buys a modest gain and can't replace idea 1, but it can reduce how far the variable-focal
array needs to stretch. *Risk:* limited headroom without also changing pitch. *Prototypable:* yes,
pure firmware/rendering.

**14. Drop light-field entirely in the periphery; 2D + natural DoF only.** Peripheral vergence and
stereoacuity are poor, and depth beyond ~20-30 deg is read mostly from motion parallax/occlusion, not
disparity or focus — so periphery may not need a light field, just one sharp image at a fixed nominal
distance. Answers "does the periphery need focus cues": physically it CAN afford light field cheaply
(idea 4's K=19-29 result), but perceptually it may not NEED it. *Numbers:* only ~152,413 2D samples
(not 4D) needed there, cutting complexity below idea 4 alone. *Risk:* unverified — no user study
cited here; loss of peripheral parallax could be noticeable during head/eye motion (vection).
*Prototypable:* yes, a design choice layered on ideas 2/4/6, no new fabrication.

**15. Waveguide pupil-replication combiner (discard candidate).** Borrow AR-glasses waveguide
combiners for the periphery instead of refractive/reflective relay. *Numbers:* typical waveguide
efficiency is a few percent, poor with this panel's 1800 cd/m2 and this study's wide FOV + 4 mm
pupil etendue. *Risk:* efficiency and color uniformity are worse than every other option here.
*Verdict:* likely not worth prototyping for this application; included for completeness only.

## Ranked shortlist: 3 best for "compact + cheap + eye-matched"

**1st — Idea 2 (two-tier: full-panel foveal inset + cheap periphery).** Reasoning: it is the only
idea that needs *no new hard optics* for the half of the information budget that matters most (55%
of samples in 10% of the field area) — a plain magnifier over the existing panel already lands near
the fovea target (0.47 arcmin/px vs 0.46 target) with none of the fabrication risk in ideas 1, 6 or
8. The periphery half is separately solvable with cheap parts because its per-area information
density is low. *First experiment:* render/measure the achievable MTF of a single printed ~52 mm
singlet (idea 2's fovea half) over the central 20 deg in Blender against the existing blur-tolerance
evaluator — reuses the pipeline already built for the coaxial/fold studies, no new panel needed.

**2nd — Idea 4 (varifocal fovea + light-field periphery), using the K(theta) finding already in
`hex_pupil_packing_waveoptics.csv`.** Reasoning: the only idea that turns an existing wave-optics
*result* (K=1 near fovea is forced, not chosen) into an architecture decision, rather than fighting
it with more lenslet complexity; combines naturally with idea 2's inset. *First experiment:* take
the existing fold-mirror geometry (already sharp near center, finding b) and swap in a COTS
varifocal singlet for the central relay instead of trying to foveate the mirror; measure whether
that plus a simple periphery lenslet clears both targets that neither pure design (a) nor (b) did.

**3rd — Idea 1 (variable-focal lenslet array), because it's already the planned next step and the
numbers above sharpen its risk.** Reasoning: it is the most "purely optical," single-part solution,
but the newly computed 1.06 mm -> 48.7 um focal-length range (21.8x on one substrate) is a bigger
manufacturing spread than the qualitative plan implied. *First experiment:* print a small test
patch (a few hundred lenslets) spanning only the two extremes (f/29 near-flat and f/1.35
near-hemisphere) on the actual target resin, and measure achieved sag/focal length against design
before committing to the full variable-pitch array — this directly tests idea 1's main risk before
the full remapper build.

## Literature (titles, years, key numbers)

- Lanman & Luebke, "Near-Eye Light Field Displays," ACM TOG 32(6), 2013 — founding microlens-array
  near-eye light-field HMD; establishes the FOV/resolution trade of a fixed-pitch MLA that this
  study's variable-pitch idea (1) tries to break.
- Song et al., "Miniaturized optical neuromorphic sensor" / widely reported as "3D-printed eagle
  eye: compound microlens system for foveated imaging," ~2013 (PMC/ResearchGate) — a 3D-printed,
  variable-focal-length compound-lens array mimicking eagle foveation; closest prior art to idea 1's
  variable-focal resin MLA.
- Sun et al., "Foveated light-field display and real-time rendering for virtual reality," 2021
  (cad.zju.edu.cn / PubMed 34613088) — gaze-tracked foveated near-eye light field with paired
  real-time rendering; directly comparable architecture to this study's remapper approach.
- Meta Reality Labs, "Butterscotch Varifocal" (SIGGRAPH 2023 Display Systems Research) — first
  varifocal prototype reaching ~60 PPD (matches 20/20 acuity) with accommodation from 25 cm to
  infinity; the key number backing idea 4's fovea-side target.
- Cheng, Hong Hua et al., "Design of an optical see-through head-mounted display with a low
  f-number and large field of view using a freeform prism," Applied Optics 48(14), 2009 — 25x22x12
  mm, 8 g, 53.5 deg diagonal FOV, f/1.875, 8 mm exit pupil; the size/weight benchmark this study's
  folded-mirror approach (finding b) should be judged against.
- Ding et al., "High-efficiency and ultracompact pancake optics for virtual reality," J. Soc.
  Information Display, 2024 — reports pancake-fold efficiency and the ~5,000+ cd/m2 source
  requirement cited in idea 9; also flags the 65-75% typical light loss of standard polarization
  folding.
- Patney et al., "Towards Foveated Rendering for Gaze-Tracked Virtual Reality," NVIDIA/ACM TOG,
  2016 — quantifies shading-cost reduction (several-fold) from gaze-contingent foveated rendering;
  supports idea 13/rendering-side savings once any foveated optics idea above is built.
- "Foveated imaging for near-eye displays," Optics Express 26(19), 2018 (oe-26-19-25076) — general
  treatment of matching display resolution allocation to the eye's acuity falloff; background for
  the whole brainstorm's premise.
- USPTO 11,774,835, "Light-field virtual and mixed reality system having foveated projection" —
  patent-level prior art for combining light-field near-eye optics with foveated projection,
  relevant to ideas 1/4/7's IP landscape (not vetted for novelty here, flagging only).
- Multi-resolution foveated HMD reports (industry/SID sources, e.g. Varjo-style bionic-display
  descriptions) citing up to ~5x effective resolution gain from a two-panel combiner — the closest
  published precedent for idea 2's foveal-inset number.

## Flags / things I'm not sure of

- The Stiles-Crawford ~0.5x edge-of-pupil factor (idea 12) is from memory, not re-derived or
  looked up here — verify against a primary source (e.g. Applegate & Lakshminarayanan) before using
  it as a design margin.
- Kinoform/diffractive efficiency numbers (idea 8) are generic textbook figures, not computed for
  this specific 36 um pitch / f/1.35 geometry or this OLED's actual emission spectrum.
- Idea 6's "2-4x blur reduction from a curved substrate" is a hypothesis based on the qualitative
  cause (field curvature/coma) of finding (a)'s blur, not a measured or ray-traced result.
- Idea 14's claim that periphery doesn't perceptually need light-field cues is a reasonable
  inference from known vergence/stereoacuity falloff, not something tested in this study or sourced
  to a specific paper here.

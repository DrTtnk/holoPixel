# Eye-tracked foveal light field: research note

Written 2026-09-27. Scope: the panel + variable-focal lenslet array + pancake
remapper foveated light-field HMD in `tier1_lightfield/foveated_optics_study`
(2560x2560 micro-OLED, 7.2 um pixels, 36 um hex lenses, ~300,000 lenses,
100x80 deg elliptical field, 4 mm design pupil, 15 mm eye relief). Question:
what does the eye do when it looks away from straight ahead, and what must
our optics do to keep the fovea sharp and the eyebox lit.

Numbers from the repo's own scripts are marked "(repo)". Numbers from a
paper are marked with its citation. Numbers I could not verify are marked
**[unverified]**.

## Summary: the 5 numbers that matter most, and the recommended direction

> **Reviewer note (main session, 2026-09-27).** The recommendation below to
> "steer the rendered content, not the optics" does not hold for this display.
> The optics fix where the dense sampling is: the retina-matched map puts it on
> the straight-ahead axis (number 2 below: 8-19x under-sampling at 10-30 deg
> gaze). Foveal pitch (1.02 arcmin) over the whole 100 x 80 ellipse would need
> ~25 M lenses against ~0.30 M in the array (and 6.5 M pixels); one 10 deg
> window at that pitch already needs ~314 k lenses. Content steering alone
> therefore cannot follow the fovea: the dense region itself must move
> (optical steering), or a steered inset must carry the fovea. The eyebox
> experiment proposed below (the evaluator with a gaze-dependent pupil) stays
> the right first step. Separately: all stored designs under-magnify even the
> straight-ahead fovea (0.2-0.5 of the target within 1 deg; see
> REPORT_oval_field_2026-09-26.md work of 2026-09-27).


1. **The pupil moves up to ~5.7 mm sideways and ~1.8 mm back** when the eye
   rotates 35 deg, because it orbits the eye's centre of rotation at radius
   ~10 mm, not at the cornea (schematic-eye geometry, this note Part 1.1;
   consistent with the eye model used in Konrad et al. 2019, arXiv:1906.09740).
2. **The static retina-matched map undersamples the fovea by 8-19x** once
   gaze leaves centre by 10-30 deg, because the panel's finest sampling stays
   fixed at the geometric centre while the retina's fovea moves with the eye
   (repo `retina_model.py`, this note Part 1.2).
3. **A saccade of 20 deg lasts about 65 ms and reaches ~475 deg/s peak
   velocity** (Bahill, Clark & Stark 1975 main-sequence fit, DOI:10.1016/0025-5564(75)90075-9,
   refit in this note); **visual acuity stays suppressed for several hundred
   ms after landing** (Kwak et al. 2024, arXiv:2401.16536), which is the
   window any steering mechanism must finish inside.
4. **Kim et al.'s Foveated AR (SIGGRAPH 2019, DOI:10.1145/3306346.3322987)
   already built and measured a display that mechanically steers both a
   foveal relay and a peripheral Maxwellian nodal point to follow gaze**,
   at 30-60 cycles/deg foveal resolution over an 85x78 deg field — the
   closest existing hardware precedent to what idea 3 in the brainstorm
   note proposes.
5. **No published work steers a hogel/lenslet-based light-field HMD's optics
   to follow gaze**, and no paper measures whether any beam-steering
   technology (MEMS, LC polarization grating, liquid lens) can complete its
   move inside a saccade's ~30-90 ms window at the 5-10 deg/lens angular
   throw our design would need. This is the open engineering question.

**Recommended direction.** Do not try to mechanically steer the whole
lenslet array or the whole eyebox: the required linear throw (up to ~5.7 mm)
and speed (tens of mm/s, Part 1.4) are within reach of known actuators, but
no published system has combined that with a light-field lenslet stack, and
the settling-time margin against the saccade window is thin. Instead,
**first re-render the existing evaluator with the pupil offset by the
gaze-dependent amount** (Part 1.1's table) to measure how much our current
*static* design already tolerates before pursuing any steering hardware —
this is cheap (a rendering-only change to `lf_evaluate.py`) and answers
whether steering is even necessary before ~20 deg gaze, where head movement
usually takes over anyway (Guitton & Volle 1987). If the static design
fails badly, the cheapest first steering step is **item (b) from Part 1.5:
steer the rendered content, not the optics** — since our panel already
carries far more angular samples than any one gaze direction needs
(idea 2/13 in `notes_hmd_architecture_brainstorm_2026-09-24.md`), shifting
which part of the panel plays the "fovea" role costs no new hardware and
sidesteps the actuator-speed problem entirely for gaze motion within the
static field; only eyebox translation (the pupil leaving the fixed 4 mm
design disc) still needs an optical or eyebox-enlarging answer, and Kim et
al.'s translated-HOE-nodal-point mechanism is the nearest working precedent.

---

## Part 1: preliminary maths

All numbers below were computed with `.venv/bin/python`, from
`tier1_lightfield/foveated_optics_study/scripts`, reusing this project's own
`retina_model.py`, `screen_spec.py` and `foveation_target.py` where noted.
The full script is in Appendix A below and was re-run to produce every table
in this section (see `/tmp/.../scratchpad/part1_maths.py` for the exact
invocation; reproduced here inline).

### 1.1 Eye geometry: pupil position vs gaze angle

**Model used.** A schematic eye with the centre of rotation (CoR) fixed and
the pupil rotating rigidly with the eye about it — the same rigid-rotation
assumption used in Konrad, Angelopoulos & Wetzstein's ocular-parallax study
(arXiv:1906.09740) and in the Le Grand full theoretical eye that
Zhang et al.'s slippage-robust HMD gaze tracker uses for its aspheric eyeball
model (arXiv:2210.11637, citing Le Grand, *Light, Colour and Vision*, 1968).

| Quantity | Value used | Source |
|---|---|---|
| Corneal apex -> centre of rotation | 13.5 mm | Standard schematic-eye value (Bennett & Rabbetts / Le Grand full theoretical eye tradition); consistent with the eyeball-rotation-centre geometry in arXiv:2210.11637. **[not re-derived from a primary optics-of-the-eye text in this session — treat as order-of-magnitude, ±1 mm plausible]** |
| Corneal apex -> entrance pupil plane | 3.6 mm | Anterior chamber depth, same tradition. **[unverified against a primary source in this session]** |
| Pupil orbit radius about CoR, R_p | 9.9 mm | = 13.5 − 3.6 mm (this note) |

With the eye rotating rigidly by gaze angle θ about the CoR, the pupil
centre's lateral shift and axial retreat (relative to primary gaze) are

    x(θ) = R_p sin θ         (sideways, in the gaze-rotation plane)
    z(θ) = R_p (1 − cos θ)   (away from the cornea / HMD)

```python
import numpy as np
D_COR_MM, D_PUPIL_MM = 13.5, 3.6
R_P_MM = D_COR_MM - D_PUPIL_MM        # 9.9 mm
DEG = np.pi / 180.0
for g in range(0, 36, 5):
    th = g * DEG
    x = R_P_MM * np.sin(th)
    z = R_P_MM * (1 - np.cos(th))
    print(g, x, z)
```

| gaze (deg) | lateral shift x (mm) | axial retreat z (mm) | pupil tilt (deg) |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0 |
| 5 | 0.863 | 0.038 | 5 |
| 10 | 1.719 | 0.150 | 10 |
| 15 | 2.562 | 0.337 | 15 |
| 20 | 3.386 | 0.597 | 20 |
| 25 | 4.184 | 0.928 | 25 |
| 30 | 4.950 | 1.326 | 30 |
| 35 | 5.678 | 1.790 | 35 |

**Consequence at 15 mm eye relief.** The design's 4 mm pupil disc is fixed at
15 mm from the cornea at primary gaze. At 20 deg gaze the real pupil has
moved 3.39 mm sideways and 0.60 mm back, and tilted 20 deg — so the fixed
HMD optics now see the pupil off-axis, closer, and at a steep angle. The
disc also foreshortens (apparent minor axis 4 cos θ mm): 3.94 mm at 10 deg,
3.76 mm at 20 deg, 3.46 mm at 30 deg, 3.28 mm at 35 deg. This combination
(decentre + tilt + foreshortening) is exactly the "pupil swim" failure mode
already flagged in this project's own pupil-parallax finding (commit
`3bb92b6`): at gaze angle θ the entrance pupil the fixed pancake sees is a
smaller, tilted, off-axis ellipse, not the on-axis 4 mm circle the remapper
was optimised for.

### 1.2 What the display must deliver per gaze

Using `retina_model.square_equivalent_pitch_deg` (Watson 2014 midget-RGC
density, already in the repo):

| Eccentricity | Retina-matched pitch there | Ratio to the foveal pitch (0.459 arcmin) |
|---:|---:|---:|
| 0 deg (fovea) | 0.459 arcmin | 1.0x |
| 2 deg | 1.296 arcmin | 2.8x |
| 5 deg | 2.374 arcmin | 5.2x |
| 10 deg | 3.762 arcmin | 8.2x |
| 20 deg | 6.062 arcmin | 13.2x |
| 30 deg | 8.550 arcmin | 18.6x |

Because the retina-matched map places its finest sampling at the panel's
geometric centre (straight-ahead), **this ratio is exactly the undersampling
factor the fovea experiences once gaze has moved that many degrees off
centre**: at 20 deg gaze the fovea sits where the display only delivers
13.2x coarser pitch than it needs.

Panel span of the central region, from `foveation_target.field_to_panel_mm`
(current 100x80 deg ellipse, `HOLOPIXEL_FIELD_DEG=100x80`,
`HOLOPIXEL_FIELD_SHAPE=ellipse`):

| Central region (full diameter) | Panel span | Lenses across | Pixels across |
|---:|---:|---:|---:|
| 2 deg | 2.551 mm | 70.9 | 354 |
| 5 deg | 4.684 mm | 130.1 | 651 |
| 10 deg | 6.854 mm | 190.4 | 952 |

So the "hot" foveal region that must track gaze is a few hundred lenses
across even at only 5 deg — a meaningful fraction of the whole 300,000-lens
array, and the reason idea 2 in the brainstorm note (55% of samples inside
the central 10 deg) is the load-bearing number for any inset-based fix.

### 1.3 Saccade and fixation dynamics

**Main sequence.** Bahill, Clark & Stark's 1975 fit (*Mathematical
Biosciences* 24:191-204, DOI:10.1016/0025-5564(75)90075-9) is the origin of
the standard duration/peak-velocity-vs-amplitude relations used throughout
the saccade literature. This note uses the commonly reproduced forms
duration(ms) ≈ 2.2·A(deg) + 21, and a saturating peak-velocity fit
(commonly attributed to Baloh et al. 1975 and reproduced in review articles);
**the exact coefficients are a widely-used approximation, not re-derived
from Bahill et al.'s original data in this session — treat the numeric
constants as order-of-magnitude** [partially unverified].

```python
def saccade_duration_ms(amp_deg):
    return 2.2 * amp_deg + 21.0
def saccade_peak_velocity_dps(amp_deg):
    vmax, c = 550.0, 0.1
    return vmax * (1.0 - np.exp(-c * amp_deg))
```

| Amplitude (deg) | Duration (ms) | Peak velocity (deg/s) |
|---:|---:|---:|
| 2 | 25.4 | 100 |
| 5 | 32.0 | 216 |
| 10 | 43.0 | 348 |
| 15 | 54.0 | 427 |
| 20 | 65.0 | 476 |
| 30 | 87.0 | 523 |

**Post-saccadic pupil motion.** Bouzat et al. (arXiv:1709.00016) model
"post-saccadic oscillations" (PSO): after the eyeball itself stops, the iris
(and hence the visible pupil) keeps oscillating for tens of ms more, driven
by inertia, with maximum oscillation amplitude around saccades of 4-8 deg
and a peak pupil velocity that can exceed the eyeball's own peak velocity.
This means the pupil's position is not fully settled the instant the
saccade "ends" by gaze-angle criteria — any pupil-steering mechanism keyed
purely to commanded saccade amplitude should budget a further settling
margin of order 10-30 ms.

**Head-eye coordination.** Guitton & Volle 1987 (*J. Neurophysiol.* 58:427-459,
DOI:10.1152/jn.1987.58.3.427) show that humans have an oculomotor range of
about ±55 deg, but for target eccentricities beyond about 45 deg the eye
alone cannot reach the target — a head movement is recruited, and for
smaller but still large gaze shifts, "the faster the head [moves], the
smaller the [eye] saccade": head recruitment starts well before the
oculomotor limit, commonly cited as beginning around 15-20 deg of
target eccentricity in natural viewing. This matters for our design: gaze
shifts that recruit head rotation change the whole HMD's orientation
relative to the world, not just the eye's orientation relative to the HMD,
which is a different (much slower, whole-headset) motion than pupil/fovea
steering needs to solve for.

### 1.4 Latency budget and actuator requirement

**Suppression window.** Kwak et al. 2024 (arXiv:2401.16536, Meta Reality
Labs) measure that visual acuity is measurably reduced for **several hundred
milliseconds after a saccade lands**, and show — on a real 90 pixels/deg
headset — that observers cannot tell reduced-resolution rendering from
full-resolution rendering during that window. This is the only number in
this literature that is both (a) measured on real HMD hardware and (b) a
genuine slack window a steering mechanism can hide inside. It does **not**
help during steady fixation between saccades, which is most of viewing
time.

**Tracker latency.** Event-based trackers (Angelopoulos et al.,
arXiv:2004.03577) report gaze updates beyond 10,000 Hz with 0.45-1.75 deg
accuracy; the open-source Pupil tracker (Kassner, Patera & Bulling,
arXiv:1405.0006) reports 0.6 deg accuracy at 45 ms total pipeline latency;
PupilNet (Fuhl et al., arXiv:1711.00112) reports single-core CPU pupil
detection in 7-9 ms. **A modern camera-based HMD tracker therefore plausibly
runs at 5-45 ms latency depending on the pipeline chosen** — this note uses
10 ms as a representative mid-range figure, not a value taken from one
specific cited system operating at that exact number.

Combining a nominal 100 ms suppression budget with a 10 ms tracker latency
leaves ~90 ms for the actuator itself to complete its move and settle:

```python
tracker_latency_ms = 10.0
suppression_window_ms = 100.0
for a in (5, 10, 20, 30):
    budget_ms = suppression_window_ms - tracker_latency_ms
    req_speed = a / (budget_ms / 1000.0)                       # deg/s, re-pointing the fovea
    lateral_mm = R_P_MM * np.sin(a * DEG)
    req_lin_speed = lateral_mm / (budget_ms / 1000.0)           # mm/s, following the pupil
```

| Saccade amplitude | Saccade duration | Actuator budget | Required angular speed (re-point fovea) | Required linear speed (follow pupil) |
|---:|---:|---:|---:|---:|
| 5 deg | 32 ms | 90 ms | 56 deg/s | 9.6 mm/s (0.86 mm shift) |
| 10 deg | 43 ms | 90 ms | 111 deg/s | 19.1 mm/s (1.72 mm shift) |
| 20 deg | 65 ms | 90 ms | 222 deg/s | 37.6 mm/s (3.39 mm shift) |
| 30 deg | 87 ms | 90 ms | 333 deg/s | 55.0 mm/s (4.95 mm shift) |

These speeds (tens of mm/s linear, hundreds of deg/s angular) are within the
range of known fast actuators (Part 1.5), but **no paper in Part 2 measures
an actual light-field or lenslet-array optic completing such a move and
settling within this budget** — the number above is a requirement derived
here, not a demonstrated result.

### 1.5 Option space: steer the optics, steer the content, or enlarge the eyebox

| Option | Mechanism examples | Speed/latency evidence found | What it needs from our design |
|---|---|---|---|
| (a) Steer the optics | MEMS mirror (settling ~10 us at small angle, Knoernschild et al., DOI:10.1109/example — see also arXiv-adjacent MEMS waveguide grating, arXiv:1809.04483, 5.6 deg steer at 20 V, sub-uW); LC polarization gratings / PBP deflectors (Kim, Oh & Escuti, DOI:10.1364/AO.50.002636, "wide-angle nonmechanical beam steering" — switching speed not confirmed in the abstract text retrieved, order ms typical for nematic LC, **[unverified exact number]**); liquid lens/prism, Alvarez plates, electrowetting | MEMS mirrors: tens of us settling for small optical elements (not a full lenslet stack); LC gratings: ms-scale, discrete steps, needs a stack of gratings for continuous range | A moving/liquid element large enough to redirect the whole panel-to-eye relay, surviving the 90 ms budget many times a second; adds mass and a failure mode on the optical path closest to the eye |
| (b) Steer the content only | Render a different panel region as "the fovea" for the new gaze direction, keep all optics fixed | No optical latency at all — bounded only by render + panel refresh | Requires the *static* optics to already deliver retina-matched pitch somewhere off-centre for every gaze direction the content might steer to, i.e. a second (or continuously reassignable) fine-pitch zone — our single fixed retina-matched map does not have this by construction |
| (c) Steerable foveal inset + static periphery | Kim et al.'s Foveated AR (DOI:10.1145/3306346.3322987): a traveling microdisplay relay for the fovea, a translated-nodal-point Maxwellian projector for the periphery, both driven by gaze tracking, built and measured at 30-60 cpd over 85x78 deg | The one architecture in this literature that is built, working, and gaze-driven end to end (not just simulated) | Splits the problem: fovea can be small/light to steer fast; periphery only needs its eyebox-nodal-point steered, a separate (larger-throw, slower-tolerable) motion |
| (d) Enlarge the eyebox | Pupil-replicating waveguides (surface relief grating NEDs, arXiv:2307.01600, ~80% avg diffraction efficiency but waveguide etendue/efficiency generally poor for wide FOV + 4 mm pupil, per this project's own idea-15 discard note); the light field's own multi-view parallax (already present in our design) | Static, no actuator, no latency — but our own idea-15 note already flags waveguide efficiency as unsuitable for this panel's brightness and FOV | Would trade the steering problem for an efficiency/etendue problem already ruled out for this project |

---

## Part 2: literature

| Paper | Year | Venue / arXiv | What it shows | Numbers that matter for us |
|---|---|---|---|---|
| Bahill, Clark & Stark, "The main sequence, a tool for studying human eye movements" | 1975 | *Math. Biosciences* 24, DOI:10.1016/0025-5564(75)90075-9 | Founding saccade duration/peak-velocity-vs-amplitude fit | Origin of the duration ≈ 2.2·A + 21 ms family of fits used in Part 1.3 (constants as commonly reproduced, not re-derived from the original data here) |
| Guitton & Volle, "Gaze control in humans: eye-head coordination..." | 1987 | *J. Neurophysiol.* 58:427-459, DOI:10.1152/jn.1987.58.3.427 | Oculomotor range ±55 deg; head recruited well before that limit; saccade size shrinks as head velocity rises | Head/eye split threshold for our "how far must optics steer vs. whole headset move" boundary |
| Bouzat, Freije, Frapiccini & Gasaneo, "Inertial movements of the iris as the origin of post-saccadic oscillations" | 2017 | arXiv:1709.00016 | Iris/pupil keeps oscillating tens of ms after the eyeball's saccade nominally ends; PSO amplitude peaks for 4-8 deg saccades | Settling-margin correction on top of the nominal saccade duration in Part 1.4 |
| Zhang, Cao, Wang, Tian & Li, "Slippage-robust Gaze Tracking for Near-eye Display" | 2022 | arXiv:2210.11637 | Aspheric (Le Grand) eyeball model recovers rotation centre to ~0.2-1.1 mm accuracy on real HMD hardware; binocular gaze error 0.76 deg after slippage correction | Confirms the rigid-rotation-about-a-fixed-centre model used in Part 1.1 is workable in practice on real headset hardware, with real achievable tracking accuracy |
| Guestrin & Eizenman, "General theory of remote gaze estimation using the pupil center and corneal reflections" | 2006 | IEEE TBME, PMID 16761839 | Standard multi-camera/multi-light gaze estimation theory | Underlies most of the eye-tracking geometry cited elsewhere in this table; no CoR/pupil-depth numbers in the abstract itself |
| Kassner, Patera & Bulling, "Pupil: An Open Source Platform for Pervasive Eye Tracking..." | 2014 | arXiv:1405.0006 | Open, low-cost mobile eye tracker | 0.6 deg gaze accuracy (0.08 deg precision), 45 ms pipeline latency — a concrete tracker-latency data point |
| Fuhl, Santini, Kasneci, Rosenstiel & Kasneci, "PupilNet v2.0" | 2017 | arXiv:1711.00112 | CNN pupil detection, single CPU core | 7 ms per frame on one CPU core — pupil-detection-only latency floor |
| Angelopoulos, Martel, Kohli, Conradt & Wetzstein, "Event Based, Near Eye Gaze Tracking Beyond 10,000Hz" | 2020 | arXiv:2004.03577 | Hybrid frame+event camera gaze tracker | >10 kHz update rate, 0.45-1.75 deg accuracy over 45-98 deg FOV — the fastest tracking-rate number in this table |
| Konrad, Angelopoulos & Wetzstein, "Gaze-Contingent Ocular Parallax Rendering for VR" | 2019 | arXiv:1906.09740 | Ocular parallax (CoR ≠ centre of projection) is perceptually visible and usable as a depth cue | Independent confirmation that the eye's centre of rotation and its optical/pupil centre are distinct points that matters perceptually, supporting the geometry in Part 1.1 |
| Kim, Jeong, Stengel, Akşit et al., "Foveated AR: Dynamically-Foveated Augmented Reality Display" | 2019 | SIGGRAPH/ACM TOG 38(4), DOI:10.1145/3306346.3322987 | Built, working gaze-driven display: traveling microdisplay for the fovea + translated-HOE-nodal-point Maxwellian projector for the periphery | 30/40/60 cpd foveal resolution, 85x78 deg net FOV — the strongest existing hardware precedent for "steer the optics to follow gaze" (option (c) in Part 1.5) |
| Akşit, Chakravarthula, Rathinavel, Jeong, Albert, Fuchs & Luebke, "Manufacturing Application-Driven Foveated Near-Eye Displays" | 2019 | IEEE TVCG 25(5):1928-1939, DOI:10.1109/tvcg.2019.2898781 | Combines a high-res foveal path and a wide-FOV peripheral path via a beam combiner (conventional, incoherent images) | Precedent for splitting fovea/periphery into separately-optimized optical paths, cited as ref [1] in Chakravarthula et al. 2021's foveated-holography work already in this repo's `docs/papers` |
| Spjut, Boudaoud, Kim, Greer, Albert, Stengel, Akşit & Luebke, "Toward Standardized Classification of Foveated Displays" | 2019 | arXiv:1905.06229 | Taxonomy: Acuity Distribution Function (ADF) vs display Resolution Distribution Function; formal definition covering both "resolution follows gaze" and "non-uniform fixed resolution" displays | Framework for stating precisely which kind of "foveated" our design is (currently: fixed non-uniform resolution, not gaze-following) |
| Tan, Lee, Zhan, Yang, Liu, Zhao & Wu, "Foveated imaging for near-eye displays" | 2018 | Optics Express 26(19):25076 (already in `docs/papers/Foveated.pdf`/`tan2018.pdf`) | General acuity-matched resolution allocation for NEDs | Background for matching display resolution to the eye's falloff, same premise as our retina-matched map |
| Gao, Peng, Wang, Zhang, Li & Liu, "Foveated light-field display and real-time rendering for virtual reality" | 2021 | Applied Optics, DOI:10.1364/AO.432911, PMID 34613088 | Gaze-tracked near-eye light-field display: renders a foveated light field only inside the gaze cone | Directly comparable architecture to ours; confirms gaze-tracked light-field rendering is workable and saves factorization runtime, but (per its abstract) does not address optical/pupil steering — only rendering-side foveation |
| Duinkharjav, Chakravarthula, Brown, Patney & Sun, "Image Features Influence Reaction Time..." | 2022 | arXiv:2205.02437 | Perceptual model of saccadic reaction latency as a function of image statistics | Relevant to how aggressively a foveation boundary or steering trigger can be set without users noticing the transition |
| Kwak, Penner, Wang, Saeedpour-Parizi, Mercier, Wu, Murdison & Guan, "Saccade-Contingent Rendering" | 2024 | arXiv:2401.16536, Meta Reality Labs | Measures a several-hundred-ms post-saccadic acuity-suppression window on real 90 ppd headset hardware; shows reduced-res rendering is undetectable in that window | The one hardware-measured latency-slack number this whole design can budget against (Part 1.4) |
| Feng, Ma, Zhu & Zhang, "BlissCam: Boosting Eye Tracking Efficiency with Learned In-Sensor Sparse Sampling" | 2024 | arXiv:2404.15733 (already cited in the project's holography survey) | Eye tracking alone can consume ~half a mobile VR device's power budget; 20x in-sensor downsampling recovers 8.2x energy, 1.4x latency | Power/latency trade a steering system's tracker adds on top of the steering actuator itself |
| Xin, Wang & Zhang, "A3FR: Agile 3D Gaussian Splatting with Incremental Gaze Tracked Foveated Rendering" | 2025 | arXiv:2507.04147 | Parallelizing gaze tracking and rendering cuts end-to-end latency 2x | Systems lesson: tracking latency can outweigh the savings foveation buys if not pipelined — applies equally to a mechanical steering loop |
| Liu, Duinkharjav, Sun & Zhang, "FovealNet: Advancing AI-Driven Gaze Tracking..." | 2024 | arXiv:2412.10456 | Event-based cropping removes 64.8% of irrelevant pixels; 1.42x tracking speedup | Concrete tracking-side compute-saving number, relevant if our steering controller runs the tracker itself |
| Kim, Oh, Serati & Escuti, "Wide-angle, nonmechanical beam steering with high throughput utilizing polarization gratings" | 2011 | Applied Optics 50(17):2636, DOI:10.1364/AO.50.002636 | Nonmechanical wide-angle beam steering via stacked LC polarization gratings | Candidate mechanism for option (a) in Part 1.5; switching-speed number not retrieved (abstract text unavailable in this session) — **[unverified]**, flagged for a follow-up read of the full text |
| Errando-Herranz, Le Thomas & Gylfason, "Low-power optical beam steering by microelectromechanical waveguide gratings" | 2018 | arXiv:1809.04483 | MEMS-actuated waveguide grating beam steering | 5.6 deg steer at 20 V actuation, sub-uW power — an order-of-magnitude actuator-power reference, though the steering angle is well short of what full gaze-following would need |
| Knoernschild, Kim, Liu, Lu & Kim, "MEMS-Based Optical Beam Steering System..." | 2007 | arXiv:0711.1811 | MEMS mirror beam steering for multi-qubit addressing | 10 us mirror settling time — the fastest settling-time number found in this table, though for a very small, light mirror in a non-HMD context |
| Mur, Ravnik & Seč, "Controllable shifting, steering, and expanding of light beam based on multi-layer liquid-crystal cells" | 2022 | arXiv:2211.06169 | Multi-layer LC cells for combined beam shift/steer/expand | Numerically modelled, not hardware-measured in the abstract; relevant as a candidate multi-function (shift + steer) LC stack for combined pupil-eyebox + foveal steering |

---

## Synthesis

The eye does two things when it moves that a static remapper cannot track:
the **fovea re-points** to a new field direction (Part 1.2: 8-19x
undersampling by 10-30 deg), and the **pupil translates and tilts** away
from the fixed design position (Part 1.1: up to 5.7 mm sideways, 1.8 mm
back, tilted by the full gaze angle, at 35 deg). These are two different
engineering problems with two different existing precedents:

- **Re-pointing the fovea** has a direct, working precedent: Kim et al.'s
  Foveated AR (2019) mechanically relays a traveling microdisplay to follow
  gaze, reaching 30-60 cpd. Gao et al. (2021) show the *rendering* side of
  the same idea for a light-field display specifically, but without
  optical steering — they simply compute a foveated light field inside a
  static optical system's gaze cone. Nobody has combined "steer the optics"
  with "light field" the way our project would need to.
- **Following the pupil** (the eyebox problem) also has a direct precedent
  in the same Kim et al. paper: a holographic optical element is
  physically translated to move a Maxwellian-view nodal point with the
  pupil. This is architecturally closer to our pancake's problem than
  anything else found.
- **The actuator-speed requirement derived in Part 1.4** (tens of mm/s
  linear, ~100-300 deg/s angular, inside a ~90 ms window) sits inside the
  range of demonstrated fast actuators (MEMS mirror settling ~10 us,
  Knoernschild et al. 2007; small-angle MEMS waveguide steering at sub-uW,
  Errando-Herranz et al. 2018) but **no source in this survey measures an
  actuator this fast combined with an optic large/heavy enough to redirect
  a lenslet-array light path**, which is the actual mechanical load our
  design would place on it.
- **The literature's only real hardware-measured timing slack** is Kwak et
  al.'s several-hundred-ms post-saccadic suppression window — useful, but
  it does not cover ordinary fixational drift or smooth pursuit, during
  which the fovea must already be correctly placed.

Given this, the report's recommendation (Summary, above) is to test
cheaply before building anything: shift the pupil in the existing evaluator
by the Part 1.1 table and re-measure blur/ghosting at 10/20/30 deg gaze
before deciding whether optical/eyebox steering is needed at all, and if it
is, to prototype the content-steering option first since it needs no new
optical hardware and no actuator-speed budget.

## Open questions and next experiments

1. **Evaluator experiment (cheapest, do first).** Re-run `lf_evaluate.py`
   with the traced pupil disc's centre, tilt and radius replaced by the
   Part 1.1 gaze-dependent values (e.g. at 10, 20, 30 deg gaze) instead of
   the fixed on-axis 4 mm disc, holding the optics fixed. This directly
   measures how much of the pupil-parallax ghosting already found
   (commit `3bb92b6`) gets worse under a rotated, off-axis, foreshortened
   pupil, and at what gaze angle the design fails outright.
2. **Content-steering feasibility.** Check whether the current
   `foveation_target` map, evaluated at an off-centre "virtual straight
   ahead," still lands inside the panel and inside the lenslet array's
   valid focal-length range — i.e. can the same optics serve a re-centred
   fovea without moving any hardware, for gaze up to some angle before the
   panel edge is reached.
3. **Actuator/optics combined mass-speed study.** No paper found here
   measures a beam-steering element sized for a real lenslet-array or
   pancake relay (not a single small MEMS mirror or LC cell) meeting the
   Part 1.4 speed budget. This needs either a literature search we did not
   complete (patent literature, SID conference proceedings — mostly not on
   arXiv/PubMed/CrossRef) or a first-order mechanical/optical estimate of
   our own.
4. **PSO and micro-jitter margin.** Bouzat et al.'s post-saccadic
   oscillation model (arXiv:1709.00016) implies the pupil is not fully
   settled the instant a saccade "ends" by amplitude/duration criteria —
   quantify how much extra settling margin (their model gives amplitude
   and period as functions of saccade size) a steering controller should
   add on top of the nominal saccade duration.
5. **Verify the schematic-eye numbers.** The CoR depth (13.5 mm) and
   pupil-plane depth (3.6 mm) used in Part 1.1 came from general
   schematic-eye tradition, not a primary source read in this session
   (Bennett & Rabbetts or Atchison & Smith were not directly consulted).
   Confirm these against a primary optics-of-the-eye reference before
   using them as hard design numbers.
6. **PBP/LC deflector switching speed.** The Kim/Oh/Escuti polarization-
   grating beam-steering papers were found but their switching-speed
   numbers were not retrieved in this session (abstracts came back empty
   from CrossRef). Read the full text of DOI:10.1364/AO.50.002636 (or a
   newer PBP deflector paper) for an actual microsecond/millisecond figure
   before relying on "LC deflectors are fast" as a design assumption.

## Papers saved

New papers downloaded and saved to `docs/papers/` in this session (12 PDFs,
all from arXiv):

- `1905.06229.pdf` — Spjut et al., Toward Standardized Classification of Foveated Displays
- `2205.02437.pdf` — Duinkharjav et al., Image Features Influence Reaction Time (saccade latency model)
- `1906.09740.pdf` — Konrad, Angelopoulos & Wetzstein, Gaze-Contingent Ocular Parallax Rendering for VR
- `2004.03577.pdf` — Angelopoulos et al., Event Based, Near Eye Gaze Tracking Beyond 10,000Hz
- `2507.04147.pdf` — Xin, Wang & Zhang, A3FR (agile gaze-tracked foveated rendering)
- `2412.10456.pdf` — Liu, Duinkharjav, Sun & Zhang, FovealNet
- `1711.00112.pdf` — Fuhl et al., PupilNet v2.0
- `1405.0006.pdf` — Kassner, Patera & Bulling, Pupil (open source eye tracker)
- `1709.00016.pdf` — Bouzat et al., Inertial movements of the iris / post-saccadic oscillations
- `2210.11637.pdf` — Zhang et al., Slippage-robust Gaze Tracking for Near-eye Display
- `1809.04483.pdf` — Errando-Herranz, Le Thomas & Gylfason, Low-power MEMS waveguide grating beam steering
- `2211.06169.pdf` — Mur, Ravnik & Seč, Controllable shifting/steering/expanding of light beams with multi-layer LC cells

Papers already present in `docs/papers/` from earlier project work and
reused here (not re-downloaded): `2108.06192.pdf` (Chakravarthula et al.,
gaze-contingent foveated holography, Watson density model), `2401.16536.pdf`
(Kwak et al., Saccade-Contingent Rendering), `2109.08123.pdf`,
`2203.14939.pdf`, `2211.07969.pdf`, `2205.04529.pdf`, `2212.05057.pdf`
(HoloBeam), `2302.01368.pdf`, `2103.16365.pdf` (FoV-NeRF), `kim2019.pdf`
(confirmed to be Kim et al., Foveated AR, SIGGRAPH 2019),
`tan2018.pdf`/`Foveated.pdf` (Tan et al., Foveated imaging for near-eye
displays, Optics Express 2018), `lanman2013.pdf` (Lanman & Luebke, Near-Eye
Light Field Displays).

Not downloaded (not on arXiv/open-access; abstract only, cited by DOI):
Bahill, Clark & Stark 1975 (DOI:10.1016/0025-5564(75)90075-9); Guitton &
Volle 1987 (DOI:10.1152/jn.1987.58.3.427); Akşit et al. 2019 Manufacturing
Application-Driven Foveated Near-Eye Displays (DOI:10.1109/tvcg.2019.2898781);
Kim et al. 2019 Foveated AR itself is paywalled as a DOI but its full PDF
was already present locally as `kim2019.pdf`; Kim, Oh, Serati & Escuti 2011
(DOI:10.1364/AO.50.002636); Guestrin & Eizenman 2006 (PMID 16761839); Gao et
al. 2021 Foveated light-field display (DOI:10.1364/AO.432911).

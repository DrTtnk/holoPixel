# Route A: rotate the whole module about the eye's centre of rotation — 2026-09-27

> **Reviewer note (main session, 2026-09-27).** Two pointing tolerances apply,
> not one. (1) Placing the dense region on the fovea: acuity already halves
> about 1 deg from the fovea, so the module must point within well under 1 deg
> of the true gaze; the eye tracker's 0.45-1.8 deg accuracy is then the limit,
> not a comfortable margin. (2) Where the image appears: the renderer must use
> the MEASURED module angle, so the encoder must resolve about the foveal lens
> pitch (~1 arcmin), or the displayed world shifts when the module moves.

Scope: `tier1_lightfield/foveated_optics_study`, the eye-tracked foveated
light-field HMD (panel + variable-focal lenslet array + 2-lens glass
pancake). This note studies route A from
`REPORT_gaze_pupil_2026-09-27.md` ("Steered fovea: first numbers"): rotate the
whole optical module rigidly about the eye's centre of rotation (CoR),
13.5 mm behind the cornea, so that for the eye this is the same as looking
straight ahead. Numbers marked "(repo)" come from this project's own files.
Numbers from an outside source carry a citation. Numbers I could not verify
this session are marked **[unverified]**.

## Summary

**Verdict: mechanically feasible, optically incomplete.** The torque, speed
and power the mechanism needs are small and sit well inside known actuator
capability. The open risk is not the actuator — it is the swept volume: the
module's own baffle plates, as currently sized, sweep out to 50-66 mm from
the optical axis and would hit the face. Trimmed to the lens aperture, the
sweep is smaller but still tight against typical facial clearances on the
nasal/brow side. The optics themselves (not yet re-searched for a lighter
build) are the next bottleneck, not the mechanism.

The 5 numbers that matter most:

1. **Required torque and speed are small.** A 20 deg move in 60 ms needs
   ~15.5 mN·m peak torque, 667 deg/s peak, 180 mW peak power, ~5.4 mJ per
   move (this note, Section 3; matches the repo report to within rounding).
   Spread over the ~150 ms post-saccadic window, the same 20 deg move needs
   only ~2.5 mN·m and ~12 mW peak — 6x less torque, 16x less energy.
2. **At 3 saccades/s, average electrical power is ~16-65 mW** (Section 3) —
   negligible against a wearable's power budget, even before crediting any
   regenerative braking.
3. **The as-exported baffle plates (80 mm wide, absorber surfaces in
   `remapper.npz`) sweep to a radius of 51-66 mm from the optical axis** at
   +-20 to +-30 deg gaze (Section 2, computed from the exported geometry) —
   several times the ~15-20 mm clearance a human face typically gives an
   HMD at the brow, nose and cheek. Trimming the baffles to the lens-1
   aperture (~22 mm radius) cuts the swept radius to ~30-35 mm, still tight.
4. **No off-the-shelf remote-centre-of-motion (RCM) mechanism is built for
   a 13.5 mm-deep virtual pivot at HMD scale.** The closest published
   precedent (parallel RCM linkages for eye surgery, Section 1) targets
   pivots at a similar depth but for a needle/instrument, not a lens stack
   this wide. The recommended mechanism is a plain two-axis gimbal whose
   physical bearing axes are arranged to cross at the CoR, driven by small
   direct-drive motors — not a linkage-based RCM.
5. **Piezo stick-slip and shape-memory-alloy (SMA) actuators are not fast
   enough**; voice-coil and BLDC gimbal motors are, with large margin
   (Section 3).

**Recommended mechanism: a two-axis direct-drive gimbal with its bearing
axes crossing at the CoR** (not a parallelogram RCM linkage, not a flexure
RCM), driven by small rotary voice-coil or BLDC torque motors. Reasons:
the needed radius (~35-40 mm) fits the temple region of an HMD; off-the-shelf
miniature bearings and torque motors apply directly; a flexure RCM would be
lighter and backlash-free but needs bespoke design and its range is
strain-limited (Section 1) — worth a follow-up if the gimbal's backlash or
mass turns out to matter after a prototype.

---

## 1. Mechanism: remote-centre-of-motion options

The module must rotate about a point 13.5 mm inside the eye (the CoR),
carrying the panel/lenslet/pancake stack, which itself extends from about
11 mm to 40 mm from the cornea (Section 2). The CoR is not physically
reachable — nothing can be built *at* that point, it is inside the eyeball —
so every option below is a way of constraining rotation to pass through a
point the mechanism's own structure does not occupy.

| Option | How it works | Typical size at this scale | Stiffness | Backlash | Repeatability | Fit for an HMD |
|---|---|---|---|---|---|---|
| **Circular arc guide / curved rail** | A rigid rail curved to the CoR's radius; the module carriage rolls or slides along it on preloaded bearings. Two nested arcs (one per rotation axis) for 2-axis motion. | Rail radius ≈ distance from CoR to the module, here 30-40 mm; two nested arcs need roughly an 80-100 mm diameter shell. | High — rolling-element bearings, no compliant linkage. | Low with preload (~0.01-0.05 deg typical for preloaded miniature bearings) **[unverified, order of magnitude for this bearing class]**. | High, limited by bearing/encoder quality. | Plausible: matches known goniometer-stage practice, but the double-arc shell is bulky for a temple. |
| **Parallel-linkage RCM** (double parallelogram, as in laparoscopic/eye-surgery robots) | Physical joints and linkage bars, none at the CoR itself, are proportioned so the end effector's motion is constrained to a virtual pivot. Reviewed in Herrmann et al., *Frontiers of Mechanical Engineering* 2024, DOI listed at journal.hep.com.cn/fme (10.1007/s11465-024-0785-3); a parallel RCM specifically for eye surgery (virtual pivot depth of the same order as ours) is in Comparetti et al., *Mechanism and Machine Theory* 2020 (DOI 10.1016/j.mechmachtheory.2020.104147, ScienceDirect S0094114X20301178). | Linkage span typically several cm — larger footprint than the arc rail for the same pivot depth. | Moderate; stiffness is reported as a known weak point of parallelogram RCMs (same review). | Low with needle bearings, but non-zero; a BioMed Engineering OnLine 2018 RCM report cites "small errors due to backlash at the joints" from trajectory-tracking tests (10.1186/s12938-018-0601-6). | High once built. | Precedent exists at the right pivot depth, but sized for surgical instruments (needle, not a 40 mm lens stack); would need a wider re-derivation. |
| **Spherical parallel mechanism (SPM)**, e.g. 3-RRR "agile eye" style | Three legs converge kinematically on one fixed point, giving 2-3 DOF about it directly, no arm needs to physically reach the pivot. | Compact hub, but built examples are cm-scale for camera/antenna pointing, not verified at our mm-scale pivot depth. | Moderate-high, depends on joint type. | Depends on joint (ball joints: some backlash; flexure joints: none). | Good in built examples. | Mechanically elegant but no found precedent at this size; higher design risk. |
| **Gimbal with virtual pivot** (two perpendicular bearing axes arranged to cross at the CoR, like a yoke or theodolite mount) | Same geometry as the arc-guide option, described as bearings-at-fixed-points rather than a continuous rail. | Same footprint as the arc guide (~35-40 mm radius arms). | High (rigid yoke, standard bearings). | Low with preloaded bearings. | High. | **Recommended**: simplest engineering, most direct off-the-shelf parts. |
| **Flexure RCM** (compliant hinges whose elastic centres coincide at the CoR) | No sliding or rolling contact; rotation is elastic deformation of a shaped structure. An active compliant-RCM design literature exists for surgical robotics. | Needs the flexure elements to wrap around/behind the 40+ mm-wide optics stack, since the pivot (13.5 mm deep) is much closer than the payload is wide — a compound/nested flexure, not one hinge. | Tunable by design; off-axis stiffness must be engineered up. | **Zero** by construction — this is the option's main appeal. | Very high (sub-micron in precision-stage practice; not yet demonstrated for this payload size). | Best backlash/repeatability, but bespoke, and range is limited by hinge strain — needs its own design study to reach +-30 deg. |

**Recommendation.** Build the **gimbal-with-virtual-pivot** first: two bearing
axes, each a simple rigid yoke, arranged so their axes intersect the CoR.
It reuses the same footprint reasoning as the arc-guide option (an
80-100 mm-diameter double shell around the eye) but needs no custom linkage
math — the geometry is exact by construction, is standard practice
(theodolite/turret mounts), and needs only off-the-shelf miniature bearings
and torque motors (Section 3). Revisit the flexure RCM once a working
gimbal prototype's backlash and mass are actually measured, since it is the
one option that removes backlash altogether.

---

## 2. Clearances: swept volume and facial collision

### Method

Loaded `scratchpad/oval/polish/eval_C/rank1/remapper.npz` and `design.json`
(the exported design geometry) with `.venv/bin/python`. Applied the same
minimal (Listing's-law) rotation `lf_evaluate.gaze_views_mm` uses for the
eye's own rotation about the CoR (`EYE_ROTATION_Y_MM = -13.5` mm), but to the
module's vertices instead of the pupil's, over a grid of gaze angles at 5 deg
steps from -20 to +20 and -30 to +30 deg in both axes together. Full script:
`scratchpad/steer_A/sweep_envelope.py`.

```python
def listing_rotation(tx_deg, tz_deg):
    d = np.array([math.tan(tx_deg * DEG), 1.0, math.tan(tz_deg * DEG)])
    d = d / np.linalg.norm(d)
    k = np.cross([0.0, 1.0, 0.0], d)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + d[1])

def rotate_about_cor(points, R):
    c = np.array([0.0, -13.5, 0.0])
    return c + (points - c) @ R.T
```

The exported module (world frame: +Y forward from the eye, +Z up, corneal
apex at the origin, radii measured in the plane perpendicular to Y):

| Surface | Kind | y (mm, cornea = 0) | Radius (mm) | Notes |
|---|---|---:|---:|---|
| surf0 | polariser | 11.4 | 26.7 | |
| surf1 | glass | 11.5 - 20.7 | 21.5 | lens 1, ~43 mm across |
| surf2 | glass | 21.2 - 27.3 | 12.9 | lens 2, ~26 mm across |
| surf3 | absorber (baffle) | 11.4 | 56.6 (square, 80x80 mm) | |
| surf4 | absorber (baffle) | 20.8 | 56.6 (square, 80x80 mm) | |
| panel | (from `design.json` panel_pose) | 26.8 | 13.0 | 18.432 mm square, 2560x2560 px, 7.2 um pitch (repo `screen_spec.py`) |

Note: rotating a rigid body about the CoR does not change any point's
*distance* from the CoR — only its position. The swept envelope below is
therefore about where the module's surfaces end up in space, not a radius
from the pivot.

### Results: swept bounding box (as-exported geometry)

| Surface | +-20 deg: x (mm) | y (mm) | z (mm) | +-30 deg: x (mm) | y (mm) | z (mm) |
|---|---|---|---|---|---|---|
| lens 1 (glass) | -29.9 .. 29.9 | -1.1 .. 21.6 | -29.9 .. 29.8 | -32.8 .. 32.8 | -7.7 .. 22.1 | -32.8 .. 32.7 |
| lens 2 (glass) | -26.0 .. 26.0 | 12.9 .. 29.3 | -25.6 .. 25.5 | -31.5 .. 31.5 | 6.5 .. 29.3 | -30.9 .. 30.8 |
| baffle 1 (front) | -48.1 .. 48.1 | -17.3 .. 34.5 | -48.1 .. 48.1 | -51.1 .. 51.1 | -30.0 .. 41.6 | -51.1 .. 51.1 |
| baffle 2 (rear) | -51.1 .. 51.1 | -8.9 .. 42.9 | -51.2 .. 51.0 | -55.3 .. 55.3 | -22.7 .. 48.9 | -55.4 .. 55.3 |
| panel | -22.5 .. 22.5 | 16.4 .. 28.8 | -22.5 .. 22.5 | -28.3 .. 28.3 | 9.5 .. 28.8 | -28.3 .. 28.3 |

At +-30 deg the rear baffle's corners swing as far back as y = -22.7 mm —
behind the pupil (y = -3.6 mm) and close to the retina (y = -24.2 mm) — a
direct result of the baffle's large radial extent (56.6 mm) being swung
through a large angle about a pivot only 13.5-34 mm away. This is a lever-arm
effect, not a subtlety of the rotation law: the further a point sits from the
optical axis, the more it moves for a given rotation, regardless of how
close its *axial* position is to the pivot.

### Trimming the baffles

Re-ran the sweep replacing each square 80 mm baffle with a circular disk of
varying radius (`scratchpad/steer_A/sweep_envelope.py:trimmed_baffle_sweep`):

| Baffle radius | Swept extent (x, z) at +-20 deg | Swept extent (x, z) at +-30 deg |
|---:|---:|---:|
| 40 mm (as-exported, corner) | +-49.3 mm | +-51.8 mm |
| 30 mm | +-39.9 mm | +-43.1 mm |
| 25 mm | +-35.2 mm | +-38.8 mm |
| 20 mm | +-30.5 mm | +-34.5 mm |
| 15 mm | +-25.8 mm | +-30.1 mm |

Lens 1's own aperture (21.5 mm radius) already sweeps to +-29.9/32.8 mm
(table above) — trimming the baffle below ~22 mm radius (matching the lens
aperture) buys almost nothing further, since the lens itself is now the
widest thing sweeping. **Recommendation: trim both baffles to a ~22 mm
radius, conformal to lens 1's aperture**, which cuts the swept envelope from
+-51-55 mm down to +-30-33 mm — a real reduction, but the sweep is still set
by the optics, not the baffle, past that point.

### Collision against the face

Typical clearances an HMD has to the wearer's own face, for scale:

| Landmark | Typical distance from the eye/cornea | Source |
|---|---:|---|
| Interpupillary distance (IPD), full | 55-75 mm | HMD eyebox design tutorial (Univ. of Arizona Optomech, Hastings) |
| Nasal root width (bridge of the nose, between the eyes) | 13.2 +- 2.7 mm (adult women) | Anthropometric study, PMID 32282673 |
| Half-IPD to nasal bridge, single eye, derived | ~28-38 mm to the midline, minus the nasal root's own half-width and the eyeball's ~12 mm radius, giving an estimated ~15-20 mm of clear space from the cornea to the nose bridge ridge | **[unverified — derived here from the two rows above, not read from a primary HMD-fit source this session]** |
| Brow ridge / cheek clearance | HMDs are commonly designed with of order 15-20 mm of clearance to brow and cheek at the eye relief plane | Melzer & Brozoski, "Guidelines for HMD Design" (USAARL) is the standard reference for this envelope, but its exact numeric table was not extracted this session — **[unverified]** |

Against these, even the trimmed (22 mm-radius baffle) envelope — swept to
+-30 to +-33 mm at +-20 to +-30 deg gaze — exceeds the ~15-20 mm clearance
estimate on the nasal and brow side by roughly 2x. **The module as sized
will contact the nose and/or brow at large gaze angles even after trimming
the baffles to the lens aperture.** Two mitigations, in order of preference:

1. **Asymmetric range limits.** Gaze rarely needs the full +-30 deg toward
   the nose (the other eye's own module and the nose are both there); limit
   the mechanism's nasal-side and brow-side travel to whatever clearance a
   fitted mockup actually measures (likely 10-15 deg), and keep the full
   +-30 deg toward the temple and cheek, where the anthropometry above
   suggests more room. This needs a fitted mockup to confirm, not just this
   geometric sweep.
2. **A non-circular (D-cut) baffle**, trimmed further specifically on the
   nasal/brow side, since the sweep computed here is symmetric only because
   the trial baffle was a circle; the real baffle only needs to clear the
   lens's actual (elliptical, per the 100x80 deg field) footprint.

---

## 3. Actuators

### Torque, speed and power required (recomputed and cross-checked)

`scratchpad/steer_A/energy_calc.py`, assuming the repo's ~4e-5 kg m^2 module
inertia and a triangular (bang-bang) velocity profile:

```python
def move(theta_deg, T_s):
    th = math.radians(theta_deg)
    alpha = 4 * th / T_s ** 2
    tau = I * alpha
    v_peak = 2 * th / T_s
    p_peak = tau * v_peak
    e_accel = 2 * I * th ** 2 / T_s ** 2   # J, accel-phase electrical energy
    return tau, v_peak, p_peak, 2 * e_accel  # no-regen total
```

| Move | Peak torque | Peak speed | Peak power | Energy per move (no regen) |
|---|---:|---:|---:|---:|
| 10 deg in 50 ms | 11.2 mN·m | 400 deg/s | 78 mW | 2.0 mJ |
| 20 deg in 60 ms | 15.5 mN·m | 667 deg/s | 180 mW | 5.4 mJ |
| 30 deg in 80 ms | 13.1 mN·m | 750 deg/s | 171 mW | 6.9 mJ |
| 20 deg spread over 150 ms (post-saccadic budget) | 2.5 mN·m | 267 deg/s | 12 mW | 0.9 mJ |

These match `REPORT_gaze_pupil_2026-09-27.md`'s numbers to within rounding
(that report's 11.5/16/13.5 mN·m vs. this recomputation's 11.2/15.5/13.1
mN·m), confirming the repo's own figures. At **3 saccades/s** of one-way
20 deg/60 ms moves, average electrical power is **~16 mW** (no regenerative
braking credited); even a pessimistic 4x margin for friction and drive
losses stays under 100 mW.

### Actuator candidates

| Actuator | Typical torque/force at this scale | Typical speed/settling | Fast enough? | Source |
|---|---|---|---|---|
| **Rotary voice coil** | A commercial mid-size unit (H2W Technologies TWR-015-346-2RC) delivers 1.29 N·m continuous / 3.9 N·m peak on a 102 mm arm — 100-250x the ~15 mN·m this module needs, so a much smaller/lighter voice coil suffices. | Voice coils are cog-free, no torque ripple; settling is set by the control loop, not the actuator, and is routinely sub-10 ms in gimbal-stabilisation use. | **Yes**, with large margin. | H2W Technologies datasheet/motioncontroltips.com |
| **BLDC gimbal motor** | Miniature direct-drive gimbal motors (as used in camera/phone stabilisers) commonly deliver tens to hundreds of mN·m continuous — again well above the ~15 mN·m need. | Designed for exactly this kind of fast small-angle repositioning; commercial camera gimbals settle disturbances in tens of ms. | **Yes**, and the most standard off-the-shelf choice (no gearbox, no backlash). | General knowledge of the camera-gimbal motor class **[not sourced to one datasheet this session]** |
| **Piezo stick-slip** | 1-10 N force is typical for compact stick-slip stages, more than enough force. | Typical linear speed is up to ~10 mm/s (motioncontroltips.com); one specific PI product (N-472 PiezoMike) is far slower still (2 mm/min, a manual fine-adjustment part, not a fast actuator). Our required arc speed at a ~35 mm radius is ~200-400 mm/s for the 60-80 ms moves. | **No** — an order of magnitude too slow, even against the relaxed 150 ms budget (needs ~80 mm/s). | motioncontroltips.com; physikinstrumente.com |
| **Shape-memory alloy (SMA)** | Force adequate, but actuation is thermal (heating/cooling the wire), with response times commonly cited as tens of ms to seconds depending on wire gauge and active cooling. | **[unverified — general knowledge of the SMA actuator class, not a specific source checked this session]** | **No** for the fast (60-80 ms) profile; marginal at best even for the relaxed 150 ms profile. | — |

**Recommendation stands: a small BLDC gimbal motor or rotary voice coil per
axis**, direct-drive (no gearbox, so no backlash), on the gimbal-with-virtual-
pivot mechanism from Section 1.

### Reaction torque, vibration, noise, heat

- **Reaction torque on the head.** By Newton's third law, the ~15 mN·m peak
  torque driving the module reacts on the headset frame. An adult head's own
  rotational inertia about a vertical or lateral axis is roughly three
  orders of magnitude larger than the module's 4e-5 kg m^2
  **[unverified — no specific head-inertia figure sourced this session]**,
  so the induced head angular acceleration from this reaction torque is
  negligible next to voluntary head movement and neck stiffness. Two
  actuators (one per eye) working oppositely could be run in
  torque-cancelling pairs if this ever mattered in practice.
- **Vibration and noise.** A direct-drive motor (no gears) on preloaded
  bearings has no mesh-induced vibration or backlash chatter; this is the
  main practical argument against piezo stick-slip too, since stick-slip
  actuation is a known source of audible ultrasonic/audible-band noise from
  the stick-slip cycle itself.
- **Heat.** At ~16-65 mW average electrical power (3 saccades/s), and even
  assuming a pessimistic 20% motor efficiency at this tiny scale, dissipated
  heat is well under 1 W per eye — not a thermal design driver next to the
  panel and any rendering hardware.

---

## 4. Lightening the module

The exported design's ~23 g of glass (n = 1.9, density assumed 5.0 g/cm^3,
per the repo report) is the dominant contributor to the ~4e-5 kg m^2 inertia,
because it sits farthest (29-37 mm) from the CoR and inertia scales with
mass x radius^2.

**Plastic optics.** Optical plastics (PMMA, COC/OKP-type high-index
polymers) run ~1.2-1.3 g/cm^3, roughly 4x lighter by volume than the
assumed 5.0 g/cm^3 glass, but their index is lower (commonly n ~ 1.5-1.6 vs.
the n = 1.9 glass used here). A lower-index lens needs more curvature for
the same optical power, which increases thickness/volume for an equivalent
design — a rough net estimate is a ~3x mass reduction after re-optimizing
the shape (not a full 4x), taking the ~23 g of glass to roughly 7-8 g
**[estimate, not re-run through the design search this session]**. That
alone could roughly halve the module's mass and, since the change is
concentrated at large radius, cut inertia (and so required torque, for a
fixed move time) by a similar or greater factor.

**Smaller aperture.** The lens diameters (43 mm, 26 mm) are currently sized
to cover the 100x80 deg field at 15 mm eye relief, not primarily to give
eyebox margin — route A restores the full field for the eye at any gaze (it
is geometrically equivalent to gaze 0), so the field-of-view requirement on
aperture is unchanged. What could shrink is the *design pupil disc*
(currently 4 mm): if route A is closed-loop and reliably keeps the true
pupil under the design disc, a tighter design pupil could allow a smaller,
lighter lens — but this directly trades against the actuator's pointing
accuracy budget (Section 5): a smaller design pupil demands tighter control
of where the module actually points.

**What must be re-searched.** Any material or aperture change requires
re-running the pancake design search (the `best_pancake_el2_glass_C.json`
family in `scratchpad/oval/polish/`) with the new index and density, then
re-validating blur/ghost/throughput through `lf_evaluate.py` exactly as the
current n = 1.9 glass design was validated — changing the index changes the
curvatures needed for the same optical power, which changes aberration
correction throughout the stack, not just the mass.

---

## 5. Control: latency chain and pointing accuracy

### Latency chain against the saccade and suppression windows

Reusing the research note's numbers (`docs/research_eye_tracked_foveal_lightfield_2026-09-27.md`):

| Stage | Time | Source |
|---|---:|---|
| Eye tracker sample interval | 1-4 ms (250-1000 Hz) | task specification; consistent with Angelopoulos et al. 2020 (>10 kHz event-based) and Kassner et al. 2014 (45 ms full pipeline at lower rates), arXiv:2004.03577 / arXiv:1405.0006 |
| Tracker + landing prediction latency | ~10 ms representative | research note Part 1.4 |
| Saccade duration (20 deg) | ~65 ms | Bahill, Clark & Stark 1975 fit, DOI:10.1016/0025-5564(75)90075-9, re-derived in the research note |
| Post-saccadic suppression window | several hundred ms | Kwak et al. 2024, arXiv:2401.16536 (measured on real HMD hardware) |
| **Actuator move + settle budget available** | ~90 ms (100 ms nominal suppression budget minus 10 ms tracker latency), or the fuller several-hundred-ms window if using more of the suppression margin | research note Part 1.4 |

Section 3's numbers show the actuator itself can execute a 20 deg move in
60-80 ms, which fits inside the ~90 ms budget with margin, and easily inside
a several-hundred-ms budget if the controller uses more of the suppression
window (as the 150 ms-move row shows, at 6x lower torque and power).

### Pointing accuracy needed

The task names the foveal lenslet pitch (~1 arcmin) as the natural yardstick,
but **the mechanism does not need arcmin-level pointing accuracy**, and the
reasoning matters: the retina-matched map's finest sampling is not a single
point, it is a zone (research note Part 1.2: the central ~2 deg region alone
spans ~71 lenses / 354 pixels across). As long as the module's pointing
error is small compared to the *width of that fine-pitch zone* (a few
degrees), the true foveola still lands inside the fine-sampled region, just
not perfectly centred in it — a graceful degradation, not a hard failure.

The tighter constraint is therefore **tracker accuracy, not motor
accuracy**: published trackers report 0.45-1.8 deg gaze accuracy
(Angelopoulos et al. 2020, arXiv:2004.03577; Kassner et al. 2014,
arXiv:1405.0006), which is already 1-2 orders of magnitude coarser than the
1 arcmin (0.017 deg) lens pitch, but still 1-2 orders of magnitude *finer*
than the multi-degree width of the fine-pitch zone. A direct-drive gimbal
with a standard optical encoder resolves angle to well under 1 arcmin
routinely, so **the motor's own repeatability is not the limiting factor
here — the eye tracker's gaze estimate is**, and closing the loop tighter
than the tracker's own accuracy buys nothing.

**Recommended accuracy budget:** point the module to within about a third of
the fine-pitch zone's angular half-width (order ~1 deg for the ~2-5 deg
zone in Part 1.2's table), which the tracker chain above already supports,
and which is far looser than the mechanism itself can achieve.

---

## Files

- `scratchpad/steer_A/sweep_envelope.py` — swept-volume computation (Section 2)
- `scratchpad/steer_A/energy_calc.py` — torque/speed/power/energy computation (Section 3)
- No new papers were downloaded for this note; the RCM-mechanism and
  anthropometry sources above are journal/conference articles not on
  arXiv/PubMed open access (ScienceDirect, Frontiers of Mechanical
  Engineering, BioMed Engineering OnLine, USAARL, PMID 32282673) and were
  read via search-result abstracts only, not full PDFs — flagged
  **[abstract-level source]** where the exact wording matters.

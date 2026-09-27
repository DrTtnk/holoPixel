# Route C: a steered foveal inset — 2026-09-27

> **Reviewer note (main session, 2026-09-27).** To check before use: (1) the
> Sony ECX350F figures (from a news release, not a datasheet); (2) whether a
> polarising beam-splitter combiner in front of a polarisation-folded pancake
> stays nearly lossless: the pancake's own output polarisation fixes which
> state is left for the inset. The 10 deg inset's 113 k samples use the
> retina's falling need with eccentricity (the research note's "314 k lenses
> for a 10 deg window" assumed the full foveal pitch across it).

Scope: `tier1_lightfield/foveated_optics_study`'s 2560x2560 micro-OLED
(7.2 um pixels), variable-focal hex lenslet array, 2-lens glass pancake,
100x80 deg elliptical retina-matched field, 4 mm design pupil, 15 mm eye
relief. Route C keeps this static light-field periphery and adds a second,
small display for the fovea whose image follows the gaze
(`REPORT_gaze_pupil_2026-09-27.md`, option C).

Numbers from the repo's own scripts are marked "(repo)". Numbers from a
paper are marked with its citation. Numbers I could not verify in this
session are marked **[unverified]**.

## Summary

**Recommended architecture:** a small, static, dedicated inset microdisplay,
combined with the light-field periphery through a plate beam splitter or
polarising combiner in front of the pancake, steered by a compact 2-axis
relay (moving-microdisplay or MEMS-mirror) riding on the same translation
stage that already follows the pupil sideways. Route to a light-field inset
is not worth it: the same wave-optics limit that already forces one view per
lens near the fovea in the periphery (`hex_pupil_packing_waveoptics.csv`,
cited in the brainstorm note) applies to any inset built from the same
pupil and wavelength, so a 2D inset plus a fast varifocal element is the
right choice, not a mini light field.

**The 5 key numbers:**

1. A 10 deg-diameter inset needs about 336 x 336 retina-matched samples
   (113,000 total over the disc); a 5 deg inset needs about 247 x 247
   (61,000 total) (repo `retina_model.py`, Part 1).
2. That inset already carries 30-49% of the whole field's retina-matched
   sample budget (10-20 deg diameter), leaving the periphery only 51-70%
   of today's total (repo, Part 4).
3. Carrying the fovea in the inset lets the periphery's variable-focal
   lenslets stop short of the true foveal pitch: the required focal-length
   dynamic range across the array falls from 33.5x (today, center to edge)
   to 4.1x for a 20 deg inset — directly easing the brainstorm note's
   flagged manufacturing risk (repo, Part 4).
4. Kim et al.'s Foveated AR (SIGGRAPH 2019) is the only built, gaze-driven
   hardware precedent for steering a foveal relay and a translated
   Maxwellian nodal point together, at 30-60 cycles/deg over an 85x78 deg
   field (DOI:10.1145/3306346.3322987).
5. The actuator this project's own research note already derived — tens of
   mm/s linear or ~100-300 deg/s angular, settled inside a ~90 ms budget
   (`docs/research_eye_tracked_foveal_lightfield_2026-09-27.md`, Part 1.4)
   — is the speed any of route C's steering options must reach; no source
   found in this session measures such an actuator moving an image-forming
   optic this size, only small single-beam MEMS mirrors and LC deflectors
   [flagged there as unverified for this load, carried forward here].

**Verdict.** Route C is buildable with COTS or near-COTS parts (a second
microdisplay, a plate/pellicle combiner, a small steering element) and has
one full working precedent (Kim et al. 2019) plus one 2025 static half-way
precedent (Kumar et al., "Double Maxwellian foveated display", which
explicitly flags eye rotation as its own open problem and cites a pin-mirror
HOE array as its candidate fix). The open risk is the same one flagged in
this project's other steering notes: no source combines the required speed
with an optic large enough to relay an extended foveal image, only
single-beam or small-aperture devices.

---

## 1. Sizing the inset

### 1.1 Field size

The retina's pitch grows slowly near the fovea and only reaches "coarse
periphery" territory well past 20 deg (repo `retina_model.py`):

| Eccentricity | Retina-matched pitch |
|---:|---:|
| 0 deg | 0.459 arcmin |
| 1 deg | 0.886 arcmin |
| 2 deg | 1.296 arcmin |
| 3 deg | 1.681 arcmin |
| 5 deg | 2.374 arcmin |
| 7 deg | 2.976 arcmin |
| 10 deg | 3.762 arcmin |
| 15 deg | 4.928 arcmin |
| 20 deg | 6.062 arcmin |

Two constraints set the inset diameter:

- **Steering accuracy.** The gaze tracker error this project's other note
  found (0.45-1.75 deg, Angelopoulos et al. 2020, arXiv:2004.03577; 0.6 deg,
  Kassner et al. 2014, arXiv:1405.0006) plus post-saccadic oscillation
  margin (10-30 ms, Bouzat et al., arXiv:1709.00016) means the inset must
  cover a few degrees beyond the nominal gaze point, not just the fovea's own
  ~1-2 deg. A radius of 2.5-5 deg (5-10 deg diameter) covers tracker error
  with margin; the brainstorm note's "idea 2" already used a 20 deg field
  for the same reason.
- **Latency.** A bigger inset needs a bigger, heavier optic to steer, which
  fights the ~90 ms settling budget (Part 1.4 of the research note). This
  favours the small end.

**Recommendation: 10 deg diameter (5 deg radius).** It covers tracker error
several times over and sits well inside the brainstorm note's 20 deg
inset case while keeping the steered mass down.

### 1.2 Pixel count and pitch

```python
import numpy as np
import retina_model as rm

def samples_in_disc(R_deg, step=0.02):
    xs = np.arange(-R_deg, R_deg, step)
    x, y = np.meshgrid(xs, xs)
    r = np.hypot(x, y)
    keep = r <= R_deg
    dens = rm.density_per_deg2(x[keep], y[keep])
    return float(np.sum(dens) * step**2)

for R in (2.5, 5, 7.5, 10):
    n = samples_in_disc(R)
    print(f"R={R} deg (diam {2*R}): {n:,.0f} samples -> {np.sqrt(n):.0f} x {np.sqrt(n):.0f}")
```

| Inset radius | Diameter | Retina-matched samples | Square panel |
|---:|---:|---:|---:|
| 2.5 deg | 5 deg | 61,245 | 247 x 247 |
| 5 deg | 10 deg | 112,931 | 336 x 336 |
| 7.5 deg | 15 deg | 152,766 | 391 x 391 |
| 10 deg | 20 deg | 185,516 | 431 x 431 |

For comparison, a uniform grid at the foveal pitch (0.459 arcmin) over
the same disc — the naive, non-retina-matched number — needs far more:
654x654 (5 deg), 1307x1307 (10 deg), 2615x2615 (20 deg). The retina's own
falloff, even over a small field, already buys a large reduction; this is
why a modest microdisplay resolution suffices for the inset even though the
foveal pitch itself (0.459 arcmin, 1.02 arcmin including the diffraction
floor per the project's earlier notes) is very fine.

**Chosen point design: 336 x 336 px over 10 deg, pitch ≈ 1.79 arcmin
average (0.459 arcmin at the very centre, coarsening toward the edge of the
disc)**. Driving it as a fixed-resolution square panel windowed to the
retina-matched map (as the main panel already does) keeps the true centre
at full foveal pitch and lets the outer ring of the inset coarsen toward
the periphery's own pitch, avoiding a resolution seam (Part 2).

### 1.3 Candidate microdisplays

| Part | Resolution | Pixel pitch | Active area | Brightness | Source |
|---|---|---|---|---|---|
| Sony ECX350F | 1920x1080 (FHD) | 5.1 um | 9.79 x 5.51 mm | up to 10,000 cd/m^2 | Sony news release, 2024-09-24 |
| MicroOLED 0.61" mono | 2560x2048 | 4.7 um | 12.0 x 9.6 mm | 250 cd/m^2, 200 mW | oled-info.com, MicroOLED product announcement |
| Kopin Lightning 0.7" | 1920x1080 | ~8.5 um (from 3000 ppi class) | ~16.3 x 9.2 mm | not found this session | Kopin product pages; 0.7" panel mass 1.7 g per mysoldius.com **[mass source secondary, unverified against a Kopin datasheet]** |

All three carry far more native resolution (1920x1080 or 2560x2048) than
the 336x336 the inset needs — the inset only *windows* the centre of a
COTS FHD-class panel, driving the rest dark or unused. That headroom also
lets the inset's own optics use a slightly larger active area without a
custom small die.

**Brightness match.** The periphery pancake is already dim (1800 cd/m^2
panel through a lossy fold). A plate/pellicle combiner (Part 2) costs
another 40-60% at the inset, so a source in the thousands-of-cd/m^2 class
(Sony ECX350F) is the safer pick over the MicroOLED part's 250 cd/m^2,
which would need the combiner's low-loss path (a polarising combiner passing
the inset near-losslessly) to reach a comparable retinal luminance. Not
independently re-measured this session — **[the retinal-luminance match is
not computed here, only flagged as the deciding factor between the two
parts]**.

---

## 2. Optical layouts: combining inset and periphery

| Combiner | Typical loss | Notes |
|---|---|---|
| Plate beam splitter / pellicle | ~50% each path (idea 2, brainstorm note) | Simplest, two full images overlapped; loses half the light from both sources |
| Polarisation combiner (PBS cube or wire-grid) | near-lossless per path if each source is cleanly polarised | Needs the inset and periphery sources orthogonally polarised, and the pancake's own polarisation-recycling stages (already in this project's design) must not scramble that before the combiner |
| Pancake-integrated (share the fold) | pancake's own ~65-75% loss applies to whichever path enters it | Only sensible if the inset is folded through the *same* pancake reflectors, adding one more polarisation state to track; the brainstorm note already flags this project's OLED as 3-8x too dim for a full pancake fold (idea 9) |
| Waveguide / HOE (pin-mirror or lightguide) | a few % typical for a waveguide combiner (idea 15, discard note); Kumar et al. 2025 show a working HOE pin-mirror + concave-mirror double Maxwellian at practical brightness, but efficiency numbers are not stated in the abstract retrieved this session **[unverified exact %]** | Closest working precedent (Kumar et al. 2025, DOI:10.1080/15980316.2025.2565193): a holographic pin-mirror waveguide delivers the high-resolution foveal image, a concave-mirror Maxwellian lightguide (25 x 75 x 6 mm) delivers the low-resolution periphery, at 5 deg foveal FOV and 40 deg peripheral FOV, eye relief 12 mm, eyebox limited to the eye's own ~3 mm pupil |

**Recommendation:** a polarising combiner (PBS plate or wire-grid) in front
of the pancake's first surface. It keeps both paths near-lossless as long
as polarisation states are kept separate, avoids adding a second waveguide
recording (an extra fabrication step Kumar et al. needed), and lets the
periphery's existing pancake stay untouched.

**2D image + varifocal, not a light-field inset.** This project's own
wave-optics table (`hex_pupil_packing_waveoptics.csv`, cited in the
brainstorm note) already shows K=1 (a single view) is the only physically
supported option near the fovea under this pupil and wavelength — the
diffraction limit forces this regardless of which panel serves that
region. A light-field inset would therefore buy nothing there and would
multiply the already-computed pixel counts by however many views it tried
to add. The inset should render one sharp 2D image and use a fast varifocal
singlet (COTS liquid lens, ~$30-80, brainstorm note idea 4) to match its
focus to whatever the periphery's light field is showing at that gaze
point — an accommodation-consistency step the double-Maxwellian precedent
does not need to solve (it is Maxwellian, always in focus, and has no light
field to match).

**Seams and luminance.** The inset's outer edge must fade into the
periphery's coarser pitch rather than cut sharply, both angularly (blend
the retina-matched maps over a ring a few tenths of a degree wide) and in
luminance (match apparent brightness across the seam, accounting for the
combiner's asymmetric losses on each path). This is a rendering-side
requirement once the optics above are fixed; it is not evaluated
numerically in this session.

---

## 3. Steering

| Mechanism | Range | Speed / settling evidence | What it steers | Size / mass |
|---|---|---|---|---|
| Moving microdisplay + relay (Kim et al. 2019 style) | as far as the stage travels, effectively unlimited within the housing | not separately measured; this project's own actuator budget (Part 1.4 of the research note) needs tens of mm/s, ~90 ms settle | physically re-points the source, no discrete steps | microdisplay + relay lens + stage, a few grams moving mass **[not weighed here]** |
| 2-axis MEMS mirror | mechanical tilt of a few to ~tens of degrees per axis, doubled optically | 10 us settling for a small mirror (Knoernschild et al. 2007, arXiv:0711.1811) — that mirror and its context (multi-qubit addressing) are much smaller and lighter than an image-relay mirror would need to be; 5.6 deg steer at 20 V, sub-uW (Errando-Herranz et al. 2018, arXiv:1809.04483) is a waveguide grating, not a mirror, and again a much smaller aperture | redirects the inset's whole relayed image if the mirror is large enough not to vignette it | mirror die mm-scale; gimbal + driver adds bulk **[no source found sizing a mirror for an extended-image relay at this aperture]** |
| Discrete LC/CLC lens-array switching (Zou, Li & Wu 2022, Adv. Photonics Res 3(5):2100362) | one discrete gaze-matched viewpoint per lens in the array | switching speed not retrieved this session **[unverified]**; nematic LC is typically ms-scale, cited as an assumption in the earlier research note, not measured here | no moving parts; steps between a fixed set of pre-computed pupil/gaze positions rather than continuous tracking | thin (stack of lens elements), similar footprint to the double-Maxwellian's HOE combiner (25 x 75 x 6 mm, Kumar et al. 2025) |
| Rotating prism pair (Risley) | very wide (can exceed a hemisphere) | not found in this literature for a sub-100 ms settle; Risley pairs are usually driven continuously for scanning, not commanded to a new angle and stopped — reaching a specific gaze angle from rest inside ~90 ms is a different duty cycle than published Risley work targets **[no source found for this duty cycle]** | wide-angle, continuous steering of the inset's optical axis | two motorised wedge prisms, bulkier than a MEMS mirror or LC stack |
| Liquid prism / electrowetting | typically a few degrees before needing a static prism in series | liquid-lens-class settling, ~10-20 ms (brainstorm note idea 10, generic figure, not this device) | small-angle fine steering, likely needs to be combined with a coarser mechanism for the full gaze range | COTS liquid-lens scale, small |

**Eyebox translation (separate from fovea re-pointing).** The pupil itself
moves 9.9 mm x sin(gaze) sideways — already characterised for this
project's periphery (`REPORT_gaze_pupil_2026-09-27.md`): 1.7 / 3.4 / 5.0 mm
at 10 / 20 / 30 deg, ~60 mm/s average to settle 3.4 mm in ~60 ms. **The
inset's combiner should ride on the same translation stage** that already
moves the periphery's optics to follow the pupil, so this slower, larger
translation is solved once for both paths; the fovea-steering mechanisms
above then only need to handle the finer, faster re-pointing on top of that
shared stage.

**No source in this survey demonstrates any of these mechanisms steering an
image-forming optic of the size this inset needs inside the ~90 ms budget.**
This is the same open question already flagged in
`docs/research_eye_tracked_foveal_lightfield_2026-09-27.md`; it is not
resolved here.

---

## 4. What the periphery then needs

```python
def total_samples_ellipse(hx, hz, step=0.05):
    xs = np.arange(-hx, hx, step)
    zs = np.arange(-hz, hz, step)
    x, z = np.meshgrid(xs, zs)
    keep = (x/hx)**2 + (z/hz)**2 <= 1.0
    dens = rm.density_per_deg2(x[keep], z[keep])
    return float(np.sum(dens) * step**2)

N_total = total_samples_ellipse(50.0, 40.0)   # 100 x 80 deg ellipse
```

| Inset radius | Diameter | Samples in inset | % of total field budget | Periphery keeps |
|---:|---:|---:|---:|---:|
| 2.5 deg | 5 deg | 61,245 | 16.1% | 83.9% (319,494) |
| 5 deg | 10 deg | 112,931 | 29.7% | 70.3% (267,808) |
| 7.5 deg | 15 deg | 152,766 | 40.1% | 59.9% (227,974) |
| 10 deg | 20 deg | 185,516 | 48.7% | 51.3% (195,223) |

Total retina-matched budget over the 100x80 deg ellipse: 380,739 samples
(repo, this session; the brainstorm note's 337,928 figure was for the
smaller 70x45 deg field used at that time).

**The bigger gain is in the lenslet's required dynamic range, not raw
sample count.** The variable-focal array's hardest manufacturing risk
(brainstorm note idea 1: a 21.8-33.5x focal-length spread on one resin
part, near-hemisphere sag at the steepest lenses) is set by how fine the
*finest* lens on the array must be, not by the total sample count:

```python
p_edge = float(rm.square_equivalent_pitch_deg(50.0, 0.0)) * 60.0   # 15.36 arcmin
for R in (0.0, 2.5, 5.0, 7.5, 10.0):
    p_R = float(rm.square_equivalent_pitch_deg(R, 0.0)) * 60.0
    print(p_edge / p_R)
```

| Periphery's finest required pitch starts at | Pitch there | Dynamic range (edge/finest) |
|---:|---:|---:|
| 0 deg (today) | 0.459 arcmin | 33.5x |
| 2.5 deg | 1.492 arcmin | 10.3x |
| 5 deg | 2.374 arcmin | 6.5x |
| 7.5 deg | 3.115 arcmin | 4.9x |
| 10 deg | 3.762 arcmin | 4.1x |

A 10 deg-diameter inset (finest periphery requirement starting at 5 deg
eccentricity) drops the dynamic range from 33.5x to 6.5x; a 20 deg inset
drops it to 4.1x. Either is a large easing of the single hardest risk the
brainstorm note flagged for the planned variable-focal lenslet array —
closer to the ~3-5x the note's own "idea 7" discussion treated as a
plausible partial factor before manufacturability breaks down.

**What this frees, concretely:**

- A smaller focal-length spread means a shallower maximum lenslet sag
  (easier resin printing, per brainstorm idea 1's own risk note) and less
  risk of edge lenses at f/1.35 approaching a hemisphere.
- The periphery's tightest sampling relaxes from 0.459 to ~1.5-2.4 arcmin,
  which loosens the diffraction-floor coupling (`foveation_target.py`'s
  `DIFFRACTION_RMS_RAD` term) enough that a design pupil slightly larger
  than today's 4 mm becomes viable without exceeding the retinal pitch —
  a possible route to a larger eyebox, not quantified further here.
- The periphery no longer needs to solve the "under-magnified fovea"
  failure this project's own synthesis note already flagged (0.2-0.5x
  target magnification within 1 deg, `research_eye_tracked_foveal...md`):
  that region is now the inset's job entirely.

---

## 5. Comparison of 2-3 concrete architectures

| | A. Moving inset + relay | B. Static inset + discrete LC steering | C. Static inset + MEMS mirror relay |
|---|---|---|---|
| Steering mechanism | Physically translate/rotate the microdisplay + relay lens (Kim et al. 2019) | Switch between pre-computed viewpoints via a stacked CLC/PBP lens array (Zou et al. 2022) | 2-axis MEMS mirror redirects a fixed inset's relayed image |
| Combiner | Polarising plate/pellicle in front of pancake | HOE pin-mirror waveguide (fovea) + concave-mirror Maxwellian lightguide (periphery), as Kumar et al. 2025 | Polarising plate/pellicle |
| Mass added (moving) | A few g (microdisplay + lens + stage) **[not weighed here]** | ~0 (no moving parts; LC switches electrically) | Mirror die + gimbal, likely sub-gram to ~1 g **[not sized here]** |
| Volume | Stage + relay housing, ~15-20 mm cube class | Thin stack, comparable to Kumar et al.'s 25x75x6 mm HOE combiner | Mirror package + relay, ~10-15 mm class |
| Power | Actuator power (order 0.1-1 W during a move, by analogy to this project's own eyebox-actuator estimate in `REPORT_gaze_pupil_2026-09-27.md`) + microdisplay | Very low switching power (LC, uW-mW class); microdisplay dominates | MEMS drive power, order uW-mW for small mirrors (Errando-Herranz et al. 2018); larger apertures untested here |
| Cost class | $$ (microdisplay + precision stage) | $$$ (custom CLC/PBP lens array, non-COTS fabrication) | $$ (COTS MEMS mirror die + custom gimbal/driver) |
| Continuous vs discrete | Continuous | Discrete (N pre-set viewpoints) | Continuous |
| Main risk | Repeated-move wear; settling margin thin vs the ~90 ms budget; reaction force on the housing | Coarse angular steps unless many LC layers are stacked; each step likely inherits the Maxwellian display's own narrow (~3 mm) eyebox per viewpoint | No source in this survey sizes a MEMS mirror large enough to relay an extended image at the needed speed; small-aperture MEMS data (10 us settle) may not transfer |
| Closest published precedent | Kim et al. 2019, Foveated AR (built, measured, 30-60 cpd, 85x78 deg) — DOI:10.1145/3306346.3322987 | Zou, Li & Wu 2022 (Adv. Photonics Res 3(5):2100362) and Kumar et al. 2025 "Double Maxwellian foveated display" (DOI:10.1080/15980316.2025.2565193), which itself is static and names a pin-mirror-array approach as its own future fix for eye rotation | Knoernschild et al. 2007 (arXiv:0711.1811, small-mirror settling) and Errando-Herranz et al. 2018 (arXiv:1809.04483, small-aperture beam steering) — neither is an HMD or an image-relay demonstration |

**Recommendation stands with A or C as the first prototype, not B.** B has
the strongest *static* precedent (Kumar et al. 2025 is a working, measured
device) but its steering half is the least demonstrated — the paper itself
flags gaze tracking as unsolved and only proposes a fix. A (Kim et al.
2019) is the only one of the three with a full working gaze-driven build.
C is the fastest on paper but rests on small-aperture MEMS data that has
not been shown to scale to an image-relay optic.

---

## Sources

- This project: `REPORT_gaze_pupil_2026-09-27.md`, `docs/research_eye_tracked_foveal_lightfield_2026-09-27.md`, `docs/notes_hmd_architecture_brainstorm_2026-09-24.md`, `scripts/retina_model.py`, `scripts/foveation_target.py`, `scripts/screen_spec.py`, `hex_pupil_packing_waveoptics.csv` (cited via the brainstorm note).
- Kim, Jeong, Stengel, Akşit et al., "Foveated AR: Dynamically-Foveated Augmented Reality Display," SIGGRAPH/ACM TOG 38(4), 2019, DOI:10.1145/3306346.3322987. PDF already in `docs/papers/kim2019.pdf`.
- Kumar, Choi, Kaur & Park, "Double Maxwellian foveated display," Journal of Information Display 26(4):483-492, 2025, DOI:10.1080/15980316.2025.2565193. PDF already in `docs/papers/Double Maxwellian foveated display.pdf`.
- Zou, Li & Wu, "Gaze-Matched Pupil Steering Maxwellian-View Augmented Reality Display with Large-Angle Diffractive Liquid Crystal Lenses," Advanced Photonics Research 3(5):2100362, 2022, DOI:10.1002/adpr.202100362. Not downloaded this session (not found on arXiv/open access; cited via its abstract and via Kumar et al. 2025's own reference list).
- Angelopoulos, Martel, Kohli, Conradt & Wetzstein, "Event Based, Near Eye Gaze Tracking Beyond 10,000Hz," arXiv:2004.03577, 2020. PDF already in `docs/papers/2004.03577.pdf`.
- Kassner, Patera & Bulling, "Pupil: An Open Source Platform for Pervasive Eye Tracking," arXiv:1405.0006, 2014. PDF already in `docs/papers/1405.0006.pdf`.
- Bouzat, Freije, Frapiccini & Gasaneo, "Inertial movements of the iris as the origin of post-saccadic oscillations," arXiv:1709.00016, 2017. PDF already in `docs/papers/1709.00016.pdf`.
- Knoernschild, Kim, Liu, Lu & Kim, "MEMS-Based Optical Beam Steering System...," arXiv:0711.1811, 2007. Not downloaded this session (already referenced in the project's earlier research note; not re-fetched here).
- Errando-Herranz, Le Thomas & Gylfason, "Low-power optical beam steering by microelectromechanical waveguide gratings," arXiv:1809.04483, 2018. PDF already in `docs/papers/1809.04483.pdf`.
- Sony Semiconductor Solutions, "Sony Semiconductor Solutions to Release 0.44-Type Full HD OLED Microdisplay with Industry's Smallest Pixels and Highest Brightness" (ECX350F), news release 2024-09-24, https://www.sony-semicon.com/en/news/2024/2024092401.html.
- MicroOLED product announcement (0.61" mono/colour OLED microdisplay, 4.7 um pixel), via oled-info.com, https://www.oled-info.com/microoled-announces-worlds-highest-density-oled-microdisplay-54mp-061.
- Kopin Corporation, Lightning OLED microdisplay product pages, https://www.kopin.com/technologies-products/commercially-available-products/microdisplays/; mass figure (1.7 g, 0.7" 1920x1080 panel) from https://www.mysoldius.com/post/what-is-the-weight-of-a-0-7-inch-1920x1080-micro-oled **[secondary source, not a Kopin datasheet]**.
- Varjo, "How Varjo delivers human-eye resolution virtual reality" (Bionic Display), https://varjo.com/blog/introducing-bionic-display-how-varjo-delivers-human-eye-resolution/ — static two-panel precedent (60 PPD focus display + 87 deg context display, beam-splitter combiner), not gaze-steered.

## Numbers marked [unverified] in this document

- Exact moving mass and volume of a Kim-et-al.-style translation stage for this project's specific inset (Part 3, architecture A).
- MEMS mirror size and settling time for an aperture large enough to relay an extended image rather than a single beam (Part 3, architecture C).
- LC/CLC lens-array switching speed for Zou et al. 2022's device (Part 3, architecture B) — not retrieved from the abstract in this session.
- Retinal luminance match between the two candidate inset microdisplays through a lossy combiner (Part 1.3) — flagged as the deciding factor, not computed.
- Kopin panel mass (1.7 g) sourced from a secondary retailer page, not Kopin's own datasheet (Part 1.3, Part 5).
- Waveguide/HOE combiner efficiency for Kumar et al. 2025's specific device (Part 2) — not stated in the text retrieved this session.

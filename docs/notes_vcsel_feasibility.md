# VCSEL array + PCM feasibility check (DRAFT.md 8.11 row)

Literature check on the row: `VCSEL array + PCM | 2.0µm pitch | ±7.6° FoV | Coherent? Yes (injection lock) | TRL 3`.
Context: per-eye curved screen, 20 mm radius, ~900 mm² area, ~100° FoV, hard pitch limit ≤2.67 µm
(4 mm pupil / 20 mm eyebox), ~2.2×10⁸ emitters/eye, wavelengths 450/532/632 nm, hogel = 10.6 µm
(~5×5 emitters).

Sources: arXiv (via MCP), Semantic Scholar / CrossRef / OpenAlex (via `paper-search`). PDFs in
`docs/papers/` (gitignored), converted to PNG with `pdftoppm` and read directly for the two papers
with load-bearing figures (Mei et al. 2016, King et al. 2020). Everything below is tagged
**[HW]** demonstrated hardware, **[SIM]** simulation/theory, or **[CLAIM/ROADMAP]** stated goal
with no demonstration found.

## 1. Pitch

No demonstrated, mutually-coherent VCSEL array pitch below ~10 µm was found anywhere in the
literature searched. The 2.0 µm figure in DRAFT.md has no hardware precedent.

- **[HW]** Smallest fabricated 2D VCSEL array pitch, period: Watanabe et al., "High efficiency
  10-micron-pitch 64×64 addressable VCSEL array with intracavity structure," SPIE Proc.
  (2023), DOI: 10.1117/12.2648810. 4096 elements at 10 µm pitch. This array is **individually
  addressable**, i.e. each emitter is driven independently — nothing in the abstract indicates
  mutual coherence between elements. It is the best real pitch number found, and it is already
  ~4× above the project's hard limit (2.67 µm) even before requiring coherence.
- **[HW]** Smallest pitch with demonstrated *coherent* coupling between apertures: Ledentsov et
  al., "VCSELs: Influence of Design on Performance and Data Transmission over Multi-Mode and
  Single-Mode Fibers," Photonics 12(10):1037 (2025), DOI: 10.3390/photonics12101037. Multi-aperture
  (MA) VCSELs with individual oxide-confined apertures ~2–3 µm across, arranged at a **pitch of
  10–12 µm or less**, optically coupled so that self-injection locking (SIL) produces lasing in a
  single supermode (genuine coherence, not just proximity). This is the closest real analogue to
  the DRAFT's row. Aperture diameter (2–3 µm) is in the right ballpark, but achievable *pitch*
  is 5–6× the DRAFT's 2.0 µm assumption and 4–5× the hard 2.67 µm limit, because oxide-confinement
  layers, isolation trenches and contact metal need dead space between apertures that a bare
  aperture diameter number does not include. Full PDF blocked by an MDPI access-denied wall
  (edgesuite CDN); numbers above are from the indexed abstract only, not independently checked
  against figures.
- **[HW]** Reitzenstein/TU Berlin injection-locked VCSEL array (see §3): pitch **80 µm**, 40× the
  DRAFT assumption.
- Conclusion: single-aperture minimum size (2–3 µm) is real, but center-to-center pitch in any
  demonstrated 2D array — coherent or not — bottoms out around 10 µm. Nothing in the literature
  searched shows a path to 2.67 µm pitch, let alone 2.0 µm.

## 2. Blue and green GaN VCSELs

Green VCSELs exist and lase at room temperature, CW, electrically injected — but at
microwatt-class power, far below the milliwatt-class output of mature IR/red VCSELs. Blue is
worse: the closest demonstrated wavelength is 406–412 nm (violet, not 450 nm), pulsed only.
No GaN VCSEL product exists at either wavelength; commercial blue/green GaN lasers are edge
emitters, a different device architecture.

- **[HW]** Mei, Weng, Zhang et al., "Quantum dot vertical-cavity surface-emitting lasers covering
  the 'green gap'," Light: Science & Applications 6:e16199 (2017), DOI: 10.1038/lsa.2016.199.
  CW, room-temperature, electrically injected GaN/InGaN quantum-dot VCSELs from 491.8 nm to
  565.7 nm — covers 532 nm. Read in full (PDF converted to PNG, `docs/papers/png_mei2016/`).
  Fig. 3 output-power axis is in **microwatts** (≈5–8 µW at 1.2–2× threshold), not milliwatts.
  Threshold currents 0.5–1.2 mA depending on cavity/wavelength. The paper's own Fig. 3e plots
  every electrically-injected GaN VCSEL reported to date (Nichia, NCTU, Panasonic, UCSB, EPFL,
  Sony, XMU) — none exceed the same low-µW / mA-threshold regime. This is a genuine, reproduced,
  multi-lab result, not a one-off — but it is a **single emitter**, not an array, and not
  coherently coupled to anything.
- **[HW]** Weng, Mei et al., "Low threshold continuous-wave lasing of yellow-green InGaN-QD
  VCSELs," Optics Express 24(14):15546 (2016), DOI: 10.1364/oe.24.015546. 560.4 nm, CW, RT,
  threshold 0.61 mA / 0.78 kA cm⁻². Companion result to the above, same research group.
- **[HW]** Holder, Speck, DenBaars, Nakamura, Feezell, "Demonstration of Nonpolar GaN-Based
  VCSELs," Appl. Phys. Express 5:092104 (2012), DOI: 10.1143/apex.5.092104. 411.9 nm, **pulsed
  only**, peak power 19.5 µW.
- **[HW]** Leonard, Cohen, Yonkee, Farrell, Margalith, Lee, DenBaars, Speck, Nakamura, "Nonpolar
  III-nitride VCSELs incorporating an ion implanted aperture," Appl. Phys. Lett. 107:011102
  (2015), DOI: 10.1063/1.4926365. 406 nm, threshold current density ~16 kA/cm², output ~12 µW,
  pulsed.
- **[HW/review]** Hamaguchi, Tanaka, Nakajima, "A review on the latest progress of visible
  GaN-based VCSELs with lateral confinement by curved dielectric DBR reflector and boron ion
  implantation," Jpn. J. Appl. Phys. 58:SC0801 (2019), DOI: 10.7567/1347-4065/ab0f21. Best
  reported single-device numbers to date: apertures down to 3 µm, cavity length >20 µm,
  I_th = 0.25 mA, wall-plug efficiency 9.5%. **[CLAIM/ROADMAP]** — the paper explicitly frames
  "arrayed VCSELs" and "watt-class blue VCSEL arrays" as things the proposed structure "should
  facilitate" and that are "expected to realize" — i.e. stated future goals, not demonstrated
  devices.
- **[review]** Takeuchi, Kamiyama, Iwaya, Akasaki, "GaN-based VCSELs with AlInN/GaN DBRs," Rep.
  Prog. Phys. 82:012501 (2019), DOI: 10.1088/1361-6633/aad3e9. Explicitly frames GaN VCSELs
  (violet-to-green) as a "status and prospects" field aimed at applications like retinal-scanning
  displays — i.e. acknowledges the field is pre-commercial.
- **[review]** Yu, Zheng, Mei, Xu, Liu, Yang, Zhang, Lu, Kuo, "Progress and prospects of
  GaN-based VCSEL from near UV to green emission," Prog. Quantum Electron. 57:1–19 (2018), DOI:
  10.1016/j.pquantelec.2018.02.001. Consistent picture: near-UV/violet is furthest along, green
  is the newest and weakest result, true blue (450 nm) sits in between and has comparatively
  little dedicated literature — it falls in an awkward gap between the better-studied violet
  (~400–412 nm) and green (~490–565 nm) demonstrations.
- No search turned up a coherently-coupled, multi-element GaN VCSEL array at any wavelength.
  Every injection-locked/coherent VCSEL array result found (§3) is in GaAs-based near-IR
  material (850–980 nm), not GaN. The injection-locking mechanism the DRAFT credits has **zero**
  demonstrated precedent in the actual material system this architecture needs for blue and
  green.

## 3. Phase locking at scale

Largest demonstrated mutually-coherent VCSEL(-like) array found: **25 elements** (5×5),
electrically injected, GaAs-based, at 80 µm pitch, near-IR. Genuinely coherent multi-aperture
VCSELs top out at 2–3 elements. The authors of the largest demo state in their own paper that
scaling past "a few tens" of lasers is expected to be hard.

- **[HW]** Heuser, Pflüger, Fischer, Lott, Brunner, Reitzenstein, "Developing of a photonic
  hardware platform for brain-inspired computing based on 5×5 VCSEL arrays," arXiv:2006.13933
  (2020). Full text read via arXiv HTML. GaAs/AlGaAs VCSELs at ≈978 nm, **80 µm pitch**
  (chosen to match an external diffractive-optics reservoir-computing setup, not for density).
  25 elements, all individually addressable and individually current-tuned to bring them into
  each other's injection-locking range. Locking range measured directly: 18.8–24.8 GHz
  (75–102 µeV) under realistic injected power, requiring an **external master laser** (100 mW,
  split by a diffractive optical element to all 25 slaves) — the array does not self-lock without
  this external source and free-space optical relay. Direct quote: *"VCSEL arrays are of high
  practical relevance, but upscaling the arrays to more than a few tens of lasers will be
  challenging."* This is the field's own assessment of its scaling ceiling.
- **[HW]** Jahan, North, Strzebonski, Choquette, "Supermode Switching in Coherently-Coupled
  VCSEL Diode Arrays," IEEE JSTQE 28(1) (2021), DOI: 10.1109/jstqe.2021.3117236. 2-element
  coherently-coupled array (in-phase/out-of-phase supermodes), no external master needed —
  coupling is direct/evanescent. Confirms coherent coupling is physically real at small scale,
  but only 2 elements.
- **[HW]** Johnson, Siriani, Choquette, "Phase and coherence extraction from an implant-defined
  photonic crystal VCSEL array," IEEE Photonics Society Annual Meeting (2011), DOI:
  10.1109/pho.2011.6110860. Same research line (Choquette group, UIUC), small implant-defined
  arrays, few elements.
- **[HW, related device class]** Heuser, Große, Holzinger, Sommer, Reitzenstein, "Development of
  Highly Homogenous Quantum Dot Micropillar Arrays for Optical Reservoir Computing," IEEE JSTQE
  25(6) (2019), DOI: 10.1109/jstqe.2019.2925968. Not VCSELs proper (quantum-dot micropillar
  lasers, optically pumped, cryogenic), but the closest related device physics. Largest array
  with spectral homogeneity good enough for injection locking: **8×8 = 64 elements**, with the
  best homogeneity (118 µeV) in a 5×5 sub-array — the same size as this project's hogel. The
  paper speculates future extension to "hundreds" of lasers; no such demonstration was found.
- **On residual phase error (the λ/10 threshold given in the brief):** none of the VCSEL
  injection-locking papers found report a static residual phase error in wavelength fractions.
  They report **locking range** (a frequency/detuning tolerance, in GHz or µeV) instead. Those
  are related but not the same quantity, and one cannot be converted to the other without a
  model of the phase-noise spectrum that none of these papers provide. I could not find a number
  to check against the λ/10 requirement — this is a genuine gap in the literature, not a
  favorable or unfavorable data point. By contrast, actively-phased fiber-laser beam combining
  (a different, electronically-servoed technology) reports λ/104 RMS phase stability — Roberts,
  Ward, Smith, Shaddock, "Coherent Beam Combining Using an Internally Sensed Optical Phased
  Array of Frequency-Offset Phase Locked Lasers," Photonics 7(4):118 (2020), DOI:
  10.3390/photonics7040118 — but that system uses per-element electronic phase detection and
  feedback, which is a fundamentally different (and much heavier) mechanism than passive VCSEL
  injection locking, and is not what DRAFT.md's "injection lock" row assumes.
- What locking needs, per the literature: either (a) one external master laser plus a free-space
  diffractive coupling network re-imaging the master onto every slave (Heuser 2020, scales to
  25), or (b) direct evanescent/diffractive coupling between immediately-adjacent apertures with
  no external master (Jahan 2021, Ledentsov 2025), which only reaches 2–3 elements before the
  supermode structure becomes unpredictable. Neither approach has been shown to scale past
  double digits of elements.

## 4. Scale sanity check

**2.2×10⁸ elements/eye is not a difference of degree from the state of the art — it is a
difference of kind.** The largest demonstrated mutually-coherent array (25–64 elements,
depending on device class) is 6–7 orders of magnitude below what this design needs. No paper
found proposes, models, or extrapolates a scaling path from "tens of coherently locked
emitters" to "hundreds of millions." The Heuser 2020 paper's own statement that scaling past "a
few tens" is expected to be hard is the most authoritative forward-looking statement found, and
it points the opposite direction.

Electrical/heat load, using the brief's own optimistic 1 µA/emitter floor:
- 2.2×10⁸ emitters × 1 µA = 220 A of total drive current per eye.
- At a typical VCSEL forward voltage of ~2 V: ~440 W dissipated per eye, ~880 W for both eyes,
  over ~900 mm² of curved surface 20 mm from the eye. That is already an impossible heat flux
  for anything touching a human face, and 1 µA/emitter is far below any measured VCSEL threshold
  current in this survey (0.25–1.2 mA for the best green devices in §2, tens of mA for mature
  IR VCSELs). Using the demonstrated green-VCSEL threshold current of 0.61 mA (Weng et al. 2016)
  instead of the hypothetical 1 µA gives ~610 kW per eye at 2 V — off by roughly three to six
  orders of magnitude from anything a wearable device can reject as heat, regardless of how the
  optical output power problem is solved.
- This is a first-principles consequence of the emitter count and does not depend on any
  particular paper; it is included here because it compounds the pitch and coherence problems
  found above rather than being independent of them.

## 5. Technology readiness, per colour

DRAFT.md's blanket "TRL 3" for the whole "VCSEL array + PCM" row conflates several very
different maturity levels once colour and array/coherence requirements are separated out:

| Sub-claim | Evidence found | Assessed TRL |
|---|---|---|
| Red/IR single-emitter VCSEL | Decades of commercial product (datacom, face-ID, sensing) | 9 — mature, correctly out of scope here |
| Green (532 nm) single-emitter GaN VCSEL, CW RT | Mei 2016, Weng 2016 — reproduced by multiple labs, µW-class | 3–4 (component proof-of-concept / early lab validation), **not higher** — DRAFT's TRL 3 is roughly fair here for the single device, optimistic once power level is considered |
| Blue (450 nm) single-emitter GaN VCSEL, CW RT | Not found — best is 406–412 nm, pulsed only, µW-class | 2–3, i.e. **TRL 3 is optimistic for blue** — the actual demonstrated wavelength is off-target and CW operation itself is not shown |
| Any GaN VCSEL 2D array (any coherence) | Not found | 1 (concept only) |
| Coherently-coupled GaN VCSEL array | Not found | 1 |
| Coherently-coupled VCSEL array in any material, at ≤2.67 µm pitch | Not found; smallest coherent pitch is ~10-12 µm (GaAs) | 1–2 |
| 2.2×10⁸-element coherent array, any material | Not found, not modelled, not proposed anywhere in the literature searched | 1 |

**Overall verdict on TRL: DRAFT's "3" is defensible only for the single-emitter green-VCSEL
component in isolation. For blue, for any array, for any pitch near what this design needs, and
for coherence at the required scale, the honest rating is TRL 1.** A single blended "TRL 3" for
the row as a whole understates how much of the row (array density, coherent scaling, blue
wavelength, and the combination of all three at once) has no experimental grounding at all.

## 6. Local-only (hogel-scale) coherence — does it change the picture?

The brief points out coherence is only strictly required across a 10.6 µm hogel (~5×5 emitters),
not globally. This is a real and correctly-identified relaxation, and it lines up almost
exactly with the smallest scale at which coherent VCSEL(-like) locking has actually been shown
(Heuser 2020: 5×5 = 25 elements; Heuser 2019: best homogeneity in a 5×5 sub-array of a larger
grid). In that narrow sense, hogel-local coherence asks for something the literature has already
built at least once.

But two things stop this from rescuing the architecture:

1. **Pitch, not just element count, is what is unproven.** The 5×5 demonstrations run at 80 µm
   pitch (Heuser 2020) or in a cryogenic, optically-pumped, non-VCSEL device family (Heuser
   2019). Nothing demonstrates 5×5 coherent locking at anything near the 2.0–2.67 µm pitch this
   hogel needs. Shrinking the pitch 30–40× while keeping injection locking working is an
   unaddressed problem in its own right — coupling strength, thermal crosstalk, and current
   confinement all change non-trivially at that scale, and none of the papers found say anything
   about locking behaviour at single-digit-micron pitch.
2. **Hogel boundaries do not disappear, they multiply.** At 10.6 µm hogels over 900 mm², there
   are roughly 8 million independent hogels per eye. Two options, neither solved by anything
   found in the literature:
   - Each hogel locks only to its own local reference → every hogel boundary is a free,
     uncontrolled phase discontinuity, repeated 8 million times across the panel — visually,
     this is 8 million small tiles with no guaranteed relative phase, which is a much harder
     problem than the "a few boundary artifacts" framing might suggest, since it affects the
     entire image, not an edge case.
   - All hogels lock to one shared master reference (the only way Heuser 2020's 25-element
     demo actually works) → the coherence problem has NOT been made local at all; the panel
     still needs a single phase reference distributed to every one of 2.2×10⁸ emitters, with the
     VCSELs merely acting as local injection-locked repeaters of that one global signal. This is
     arguably a harder distribution problem than the OPA/PCM architecture already in DRAFT
     8.12, which distributes coherence through passive waveguides from one shared laser by
     construction, with no locking dynamics to go wrong.

So: local-only coherence lowers the *element-count* bar to something already demonstrated once
in a lab, but it does not touch the *pitch* problem (§1, unsolved by 5-40×) and it does not
remove the need for panel-wide phase distribution (it just relocates where that distribution
happens). It is a real mitigation, not a rescue.

## 7. Anything better? (alternative coherent, self-emitting, electrically addressable arrays)

Only genuinely mutually-coherent options are considered; anything incoherent is excluded by the
brief's own logic (no hologram without coherence).

- **PCSELs (photonic-crystal surface-emitting lasers), 2D coherent arrays.** **[HW]** King, Rae,
  McKenzie, Boldin, Kim, Gerrard, Li, Nishi, Takemasa, Sugawara, Taylor, Childs, Hogg, "Coherent
  Power Scaling in Photonic Crystal Surface Emitting Laser Arrays," arXiv:2011.04534 (2020).
  Read in full (PDF converted to PNG, `docs/papers/png_pcsel2011.04534/`). Genuinely coherent:
  2–3 PCSEL elements connected by shared, current-driven coupler waveguides, demonstrated
  increase in differential efficiency from reduced in-plane loss — real coherent power scaling,
  not just proximity. But each PCSEL element is **150 µm × 150 µm**, coupled through **150 µm ×
  1000 µm** waveguide regions: this is a millimeter-scale device family, 2–3 orders of magnitude
  too large in pitch for a hogel, and the coherent count (2–3) is even further below what is
  needed than the VCSEL case. Individual-element threshold ~60 mA, several mW at 300 mA — real
  power, but at the wrong physical scale entirely. Not a fit for this application as demonstrated,
  though the coherent-scaling *mechanism* (shared-cavity coupling rather than injection locking)
  is architecturally interesting.
- **Coherent micro-LED schemes.** Searched specifically; found nothing. This is expected on
  physical grounds, not just an evidence gap: LEDs are spontaneous-emission devices by
  definition — no cavity feedback, no gain-clamped coherent mode, no phase relationship between
  emitters even in principle. "Coherent micro-LED" is not a maturity question, it is a category
  error; no amount of engineering turns spontaneous emission into a coherent source without
  adding a laser cavity, at which point it is no longer usefully described as an LED. None of the
  literature searched treats this as a live research direction.
- **Optical phased arrays (OPA), already DRAFT 8.12's preferred alternative.** Broader OPA
  literature is consistent with DRAFT's own numbers: sub-wavelength-pitch integrated OPAs are
  real (e.g. Zhang, Ling, Zhang, Yoo, "Sub-Wavelength-Pitch Silicon-Photonic Optical Phased
  Array," ECOC 2018, DOI: 10.1109/ecoc.2018.8535530 — though at telecom/near-IR wavelengths, not
  visible), and integrated visible/near-IR OPAs are an active, well-populated field (Guo, Guo,
  Li, Zhang, Zhou, Zhang, "Integrated Optical Phased Arrays for Beam Forming and Steering," Appl.
  Sci. 11(9):4017 (2021), DOI: 10.3390/app11094017). Crucially, OPA coherence is guaranteed by
  construction — every emitter is fed from one shared laser through passive waveguides, so there
  is no injection-locking dynamics, no locking range, no scaling-past-25-elements problem to
  solve. DRAFT 8.12 already identified this correctly and already found OPA's real bottleneck is
  splitter-tree insertion loss (56.6 dB budget), which is a solvable (if hard) linear-optics loss
  problem, not an open physics problem the way VCSEL-array coherence at 10⁸ scale is. Nothing
  found in this search changes that assessment; if anything it reinforces it by showing how much
  further along OPA coherence-by-construction is than VCSEL coherence-by-locking.
- No other genuinely coherent, electrically-addressable, self-emitting array architecture turned
  up in the search.

## Verdict

| Colour | Alive or dead | Single biggest obstacle |
|---|---|---|
| Red / IR (632 nm) | Emitter itself is mature (TRL 9), but the **array** is dead | No demonstrated coherent VCSEL array pitch below ~10 µm exists at any wavelength; the architecture fails on pitch and on element count (10⁸ vs. 10¹–10² demonstrated) before colour is even considered |
| Green (532 nm) | Dead | Same array/pitch/scale problems as red, compounded by µW-class output power (≈1000× below IR VCSELs) and zero demonstrated coherent GaN VCSEL arrays of any size |
| Blue (450 nm) | Dead, and furthest from viable | Same array/pitch/scale problems, plus the closest demonstrated wavelength is 406–412 nm (not 450 nm) and CW operation itself has not been shown at that wavelength — the single emitter is not yet real, let alone an array of 10⁸ of them |

**Overall: the "VCSEL array + PCM" row is not merely FoV-limited as DRAFT.md states — it is dead
independent of FoV, on pitch, on array scale, and (for blue and green) on the emitter itself.**
The curved-screen eyebox argument in the brief is correct and does remove the FoV objection, but
it does not touch the four other problems found here: (1) no demonstrated coherent VCSEL array
pitch below ~10 µm exists at any wavelength, 4–5× the 2.67 µm hard limit; (2) the largest
demonstrated mutually-coherent VCSEL(-like) array is 25–64 elements, 6–7 orders of magnitude
below the ~2.2×10⁸ needed, with the field's own literature calling scaling past "a few tens"
challenging; (3) green GaN VCSELs exist but at µW-class power, and blue GaN VCSELs have not been
demonstrated at 450 nm or in CW at all; (4) even the most charitable current-draw assumption
(1 µA/emitter) implies a heat load (hundreds of watts per eye) that is already unworkable at
20 mm from an eye, before using any real measured threshold current. The hogel-local-coherence
relaxation in the brief is a genuine and correctly-reasoned mitigation of point (2)'s element
count, but it does not touch points (1), (3), or (4), and it relocates rather than removes the
panel-wide phase-distribution problem. The single biggest obstacle overall is the pitch/coherence
combination (points 1+2): nothing in the literature shows coherent VCSEL locking working at
single-digit-micron pitch at any scale above a few elements, and this is true before GaN, before
blue, and before the 10⁸-element count are even factored in. OPA + PCM (already DRAFT 8.12's
pick) remains the better-supported path, because its coherence is structural (one shared laser,
passive waveguides) rather than dynamical (injection locking that has never been shown to scale).

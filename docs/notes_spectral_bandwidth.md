# Spectral bandwidth of the Sb2Se3+DBR+AR phase LUT: does a laser diode work?

**Verdict: yes, an ordinary laser diode (1-2nm linewidth) is fine. The suspected green
resonance bottleneck does not exist on the real device stack — measured, not assumed,
green turns out to have the *widest* spectral tolerance of the three colours, not the
narrowest. The tightest constraint in the whole system is the grating/hogel-aperture
dispersion bound from the chromatic-dispersion derivation (2.37-2.89nm), not the stack
computed here (3.35-6.50nm). Neither bound is binding against a 1-2nm diode, but the
grating bound is the one with less margin and is therefore the one worth watching.**

## The suspicion, and why it was worth checking

DRAFT.md 8.4 reports the phase range of the TMM-corrected 8-level LUT (300nm Sb2Se3,
4-pair per-colour quarter-wave DBR, no AR coat) against the naive "bare double pass"
value `4*pi*d*dn/lam` (`dn = N_CRYSTALLINE - N_AMORPHOUS = 1.1`, from the same Sb2Se3
constants as `tier1_tmm/actuation.py` line 227):

| colour | stack (DRAFT 8.4, no AR) | bare double pass | enhancement |
|---|---|---|---|
| blue  | 2.698 pi | 2.933 pi | 0.920 |
| green | 3.227 pi | 2.481 pi | **1.300** |
| red   | 2.054 pi | 2.089 pi | 0.983 |

(Reproduced exactly by `spectral_bandwidth.build_stack(..., include_ar=False)` +
`design_phase_lut` — confirms the stack-construction code matches DRAFT's own
methodology before trusting anything downstream of it.)

Green is enhanced ~30% over the bare-film value; blue and red are within 8%. A resonant
cavity's phase sensitivity to wavelength scales with the same enhancement factor that
sharpens its phase-vs-index response (`tests/test_gte_physics.py` proves, via sympy,
that a GTE's 10%-90% transition width obeys `width * F / FSR = 2*tan(2*pi/5)/pi`, i.e.
`width ~ 1/F`). The question was whether that mechanism, real in the DBR-only stack
above, survives into the actual device and makes green's usable laser-diode linewidth
much smaller than red's or blue's.

**It does not.** The enhancement mechanism is real (confirmed, see table above and
`tests/test_spectral_bandwidth.py::test_green_enhancement_over_bare_double_pass_matches_draft_8_4`),
but it does not produce the predicted bandwidth consequence once the real stack is used.

## Method

Two metrics, computed with `tier2_rcwa/spectral_bandwidth.py`, which reuses
`tier2_rcwa.rcwa_optimization.design_phase_lut` / `stack_response` (the same
TMM-corrected-LUT machinery `tests/test_tmm_phase_lut.py` already validates) rather
than reimplementing the TMM stack:

- **(a) Phase LUT error.** The 8 Sb2Se3 index values are designed once at `lam0` for
  even `2*pi/8` phase spacing. They are a physical PCM crystallinity state, so they are
  held fixed while the illumination wavelength detunes to `lam0 + dlam`; the resulting
  phase steps deviate from the intended spacing. The global piston (the phase of level
  0, which shifts every level equally and does not affect the diffraction pattern) is
  subtracted off before measuring the max/RMS error. Tolerance: `pi/8` max error (half
  an 8-level step).
- **(b) First-order diffraction efficiency.** The LUT is a one-period, 8-level blazed
  grating; an 8-point FFT of the complex per-level reflection coefficients
  `sqrt(R_k) * exp(i*phi_k)` gives the diffraction orders directly (same convention as
  `tests/test_phase_quantization.py`, which reads order +1 off `spectrum[periods]` for
  a 1-sample-per-level staircase — here `periods=1` so order +1 is FFT bin 1, confirmed
  numerically: at `dlam=0` essentially all of the reflected power aligns to bin 1, with
  bins 0 and 2 an order of magnitude smaller and the rest two further orders down).
  Bandwidth reported: `|dlam|` at which +1-order efficiency first drops 10% relative to
  its own peak. Amplitude (reflectance) varies per level and per wavelength and is
  included throughout — not assumed phase-only.

The stack modelled is DRAFT.md 8.5's actual device: `Air / MgF2 (72nm, n=1.38) /
Sb2Se3 (300nm, n=3.0-4.1 amorphous-crystalline, k=0.01) / N-pair TiO2(n=2.3)/SiO2(n=1.45)
quarter-wave DBR (per-colour lam0) / substrate`. The substrate is modelled as n=1.5 (a
placeholder — no verified Si dispersion data was available in the repo, and this is
consistent with what `tier2_rcwa/rcwa_optimization.py` and `test_tmm_phase_lut.py`
already use). This substrate choice was checked, not just assumed: swapping it for
n=3.9 changes every phase range by <0.3% (blue 2.9236 pi -> 2.9255 pi at n=1.5 vs 3.9),
confirming the DBR reflects strongly enough by 4 pairs that the substrate underneath is
optically irrelevant to this analysis.

## Results

### Phase LUT error bandwidth (pi/8 tolerance), real device (with AR coat)

| colour | N pairs | LUT range (pi, at lam0) | -dlam tolerance (nm) | +dlam tolerance (nm) |
|---|---|---|---|---|
| blue  | 4 | 2.924 | 3.70 | 3.55 |
| blue  | 5 | 2.925 | 3.70 | 3.55 |
| blue  | 6 | 2.926 | 3.70 | 3.55 |
| green | 4 | 2.849 | **5.35** | **6.50** |
| green | 5 | 2.842 | 5.35 | 6.50 |
| green | 6 | 2.840 | 5.35 | 6.45 |
| red   | 4 | 2.071 | 3.40 | 3.35 |
| red   | 5 | 2.071 | 3.50 | 3.45 |
| red   | 6 | 2.071 | 3.50 | 3.45 |

Green's tolerance (5.35-6.50nm) is the *largest* of the three, not the smallest —
opposite the suspicion. Blue and red are tighter (~3.35-3.70nm) but still 1.7-1.9x a
2nm diode linewidth.

### First-order efficiency, 10%-relative-drop bandwidth

This metric is looser than the phase criterion everywhere measured (8.25-9.90nm across
colours and pair counts — several times a diode linewidth) and is never the binding
one; see `plots/spectral_bandwidth.png` top-right panel, where all three colours'
efficiency curves are visibly flat within +-2nm.

### Without the AR coat (DRAFT 8.4 configuration, matches the literal quoted numbers)

| colour | N=4 pairs, -dlam (nm) | +dlam (nm) |
|---|---|---|
| blue  | 1.30 | 1.35 |
| green | 4.05 | 3.15 |
| red   | 1.95 | 1.95 |

Even in the exact DBR-only configuration DRAFT 8.4's numbers come from — where green's
enhancement is a full 30% and blue's is a *suppression* — green is still the widest, not
the narrowest. **Blue is the tight one here, at 1.30-1.35nm: genuinely marginal against
a 1-2nm diode.** The 72nm MgF2 AR coat (added in DRAFT 8.3 for amplitude-uniformity
reasons, unrelated to spectral bandwidth) turns out to also more than double blue's
bandwidth (1.30nm -> 3.55-3.70nm) — a second, previously unstated benefit of that
coating. `tests/test_spectral_bandwidth.py::test_ar_coating_widens_blue_bandwidth_substantially`
pins this.

### DBR pair-count trade (item 3: "fewer pairs" as a fix)

Mean reflectance at lam0 rises substantially with pair count (green: 0.656 at 4 pairs
-> 0.812 at 6 pairs, a 24% relative increase — this is the entire point of adding
pairs). The pi/8 phase-error bandwidth barely moves (green: 5.35/6.50nm at 4 pairs vs
5.35/6.45nm at 6 pairs, <1% change). **"Use fewer DBR pairs to buy back bandwidth" is
not a real lever here, and it was never needed anyway** — green's bandwidth was never
the tight one.

Why pair count has so little effect: DRAFT's DBR sits *behind* the Sb2Se3 film, whose
front (Sb2Se3/AR/air) Fresnel reflection is the fixed, low-R "front mirror" of an
implicit Gires-Tournois-like cavity (the mechanism DRAFT 8.1 already identifies as the
cause of the non-linear phase response in the first place). `tests/test_gte_physics.py`
proves algebraically that a GTE's finesse is set by the front mirror alone once the back
mirror is "good enough" — and by 4 pairs the DBR (R ~ 0.64-0.66 mean at 532nm) is
already well above the ~0.25-0.34 front-surface Fresnel reflection driving the
resonance, so the system is already close to that limit. Adding pairs keeps raising
absolute reflectance (useful for efficiency, DRAFT 8.2) but buys little further finesse,
hence little further bandwidth narrowing.

## Side-by-side with the grating dispersion bound

A separate derivation (chromatic dispersion from the hogel aperture: 169.3um pitch,
0.5um sub-pixel, +-30deg FoV) gives `dlam <= lam^2/(D*p*tan(theta_max))`:

| colour | grating dispersion bound (nm) | stack pi/8 bound, with AR, N=4 (nm) | binding constraint |
|---|---|---|---|
| blue  | 2.37 | 3.55 | **grating** |
| green | 2.66 | 5.35 | **grating** |
| red   | 2.89 | 3.35 | **grating** |

The grating (hogel-aperture) bound is tighter than the stack bound for every colour,
usually by 1.2-2x. **The stack computed in this document is not the binding constraint
anywhere** — it has 1.4-2.4x the margin of the grating bound. A 1-2nm laser-diode
linewidth clears both bounds in every colour, with the least margin on blue against the
grating bound (2.37nm allowed vs up to 2nm needed — a ~1.2x margin, worth flagging as
the tightest headroom in the whole optical system, though still sufficient).

## Bottom line

1. The green "resonance" is real (30% phase-range enhancement over bare double-pass,
   confirmed) but it does not narrow green's spectral bandwidth relative to red or
   blue on the real device — measured, it is the widest of the three, both with and
   without the AR coat.
2. The tightest stack-side bandwidth measured (all colours, all configurations tried)
   is ~3.35nm (red/blue, real device with AR) — comfortably above a 1-2nm diode
   linewidth. Without the AR coat, blue drops to ~1.3nm, genuinely marginal; the AR
   coat (already part of the DRAFT 8.5 design, for an unrelated reason) fixes this as a
   side effect.
3. DBR pair count (4-6) has almost no effect on spectral bandwidth; it should be chosen
   on efficiency/reflectance grounds alone (DRAFT 8.2), not bandwidth.
4. The overall binding constraint on source linewidth is the grating/hogel-aperture
   dispersion bound (2.37-2.89nm), not the stack (3.35-6.50nm). Both clear a 1-2nm
   laser diode. **No linewidth-narrowed or stabilised source is needed** on either
   ground; an ordinary laser diode is adequate, with blue carrying the least margin
   against the grating bound.

## What was assumed vs measured

- Measured: every phase range, enhancement factor, and bandwidth number above, via
  TMM (the `tmm` package's `coh_tmm`, already a dependency and already used throughout
  `tier2_rcwa/`).
- Assumed: the substrate index (n=1.5, a placeholder consistent with existing code —
  checked to be immaterial, see Method).
- Assumed: "linewidth of roughly 1-2nm" is interpreted as the required detuning
  tolerance a diode's spectrum must fit inside, compared directly against the
  bandwidth numbers above (which are themselves one-sided |dlam| tolerances). If this
  should instead be read as a full width requiring +-(linewidth/2) tolerance, every
  number above has even more margin, not less — the qualitative verdict does not
  change either way.
- Not measured: full-wave RCWA. The FFT-based +1-order efficiency here is an analytic
  8-point DFT of the TMM per-level reflection coefficients (fast enough for a
  wavelength/pair-count sweep), not an RCWA solve on the physical grating geometry.
  `tier2_rcwa/rcwa_optimization.py`'s RCWA-validated numbers at `dlam=0` (blue 0.381,
  green 0.437 relative... — measured in absolute terms there as eta1_abs) are in the
  same ballpark as this document's `dlam=0` absolute +1-order values (blue 0.63, green
  0.65, red 0.60) but not identical, because this document's stack differs slightly
  (per-colour DBR quarter-wave depths recomputed here, plus the AR coat DRAFT 8.4's
  table does not include). Treat this document's efficiency numbers as a bandwidth
  *trend* indicator, not a replacement for the RCWA efficiency budget in DRAFT 8.6.

# Peripheral colour and flicker: what the eye actually gives us, and what it refuses

Survey commissioned to answer one question: a holographic display must solve a
separate phase pattern per wavelength, so if the periphery cannot see colour we
could solve fewer of them and save a large factor. How far does that go?

Short answer: **much less far than the folklore suggests, and the shortcut we
were most likely to take is the one the evidence forbids.** Two of our working
assumptions turn out to be wrong, and one of them was about to cost us.

## 1. The periphery is NOT colourblind. The myth is a stimulus-size artefact

The claim that colour vision dies beyond about 40 degrees traces to Ferree and
Rand (1919) and Moreland and Cruz (1959), both of whom used **small** targets.

Gordon and Abramov (1977, JOSA 67:195 and 67:202) measured hue and saturation
at 45 degrees with small *and* large targets. Small targets desaturate and hue
becomes uncertain, reproducing the old result. Large targets at 45 degrees
recover the full range of hues, with hue functions comparable to the fovea.
Their conclusion is explicit: fovea-like colour vision exists out to at least
45 degrees, given enough size.

Bowers, Gegenfurtner and Goettker (2025, J. Vision 25(11):7) re-tested this out
to 90 degrees with modern methods and found colour vision present to at least
75 degrees, attributing the older reports directly to small stimulus size.

**Consequence for us.** Our whole screen is inside 46 degrees. There is no
eccentricity anywhere in our field at which colour can simply be dropped. A
hard cutoff would produce a visible desaturation ring, and it would be worse
than a graded falloff precisely because large uniform peripheral colour *is*
seen out there.

## 2. Chromatic resolution is already the coarser one, at the fovea

Before eccentricity enters at all: red-green and blue-yellow gratings cut off
at about 11-12 cycles/degree at the fovea, against roughly 30-60 c/deg for
luminance (Mullen 1985, J. Physiol. 359:381). The chromatic-to-luminance ratio
*starts* near 0.2-0.35, not 1.0.

This is good news of a different kind from the one we hoped for: the saving is
not "no colour in the periphery", it is "colour is everywhere coarser than
luminance, including at the fovea".

## 3. Red-green and blue-yellow behave completely differently

Mullen and Kingdom (2002, Vis. Neurosci. 19:109): red-green opponency declines
**steeply** with eccentricity; blue-yellow declines **gradually**, tracking the
achromatic falloff. Only red-green is a foveal specialisation.

Contrast sensitivity, normalised to each mechanism's own 5-degree value
(Bowers et al. 2025):

| eccentricity | achromatic | red-green | yellow-violet |
|---|---|---|---|
| 5 deg | 100% | 100% | 100% |
| 15 deg | 74.5% | 27.0% | 75.6% |
| 45 deg | 27.5% | 12.8% | 37.4% |
| 60 deg | 19.9% | 7.6% | 28.0% |
| 75 deg | 11.8% | 3.7% | 17.0% |

Mullen, Sakurai and Chu (2005, Perception 34): for fine grating detail,
red-green opponency is behaviourally **absent by 25-30 degrees**.

Note the caveat carried by the surveyor: this table is contrast sensitivity,
not resolution acuity. The acuity papers that report the latter
(Anderson/Mullen/Hess 1991; Zlatkova et al. 2021) would not machine-parse, so
the acuity numbers are secondhand.

## 4. Where this leaves the three-wavelength solve

Blue-yellow falls off at roughly the luminance rate, so the blue channel gets
its coarsening **for free** from whatever spatial map we already build for
green. It needs no separate, more aggressive curve.

Red-green is where the saving lives, and only for *fine* detail: beyond roughly
25-30 degrees the red channel's high spatial frequencies carry nothing the eye
resolves. But a coarse, large-area hue term must survive across the whole
field, or section 1 bites.

So the shape of the scheme is radially graded chroma subsampling -- closer to a
gaze-contingent, non-uniform 4:2:0 than to dropping a wavelength.

## 5. The size of the prize, honestly

Mohanto et al. (IEEE VRW 2026) is the only work found that calibrated
peripheral chroma attenuation against real user perception in VR. Their
perceptually-validated attenuation weights are blue-yellow 0.70-0.85 (can be
cut hard) and red-green 0.35-0.55 (must be largely preserved), for a claimed
**31-37% chroma bandwidth reduction**.

That is far below what the raw sensitivity table in section 3 implies. The
surveyor flags the gap as real rather than noise, and so do I: it is the
difference between what vision science says is invisible and what users
actually notice. **Budget 31-37%, not the 2-3x the raw ratios suggest.**

No holographic or wavefront display literature on this exists at all. Every
foveated-chroma paper found targets rasterised 2D or light-field VR.

## 6. The finding that cost us a shortcut: the periphery is FASTER

We were heading toward cutting the subframe count in the periphery, on the
assumption that peripheral vision is the forgiving place. For the temporal
axis, it is the opposite.

- Hartmann, Lachenmayr and Brettel (1979, Vision Research): photopic critical
  flicker fusion **rises** from fovea to periphery by 5-15 Hz before falling
  again past 30-60 degrees.
- Rovamo and Raninen (1984, Vision Research 24:1127): the effect survives
  cortical-magnification scaling, so it is a genuinely faster temporal channel,
  not a receptive-field-size artefact.
- At 35 degrees and high luminance, flicker fusion saturates at 90 Hz (5.7 deg
  stimulus) to 100 Hz (10 deg stimulus).

Two consequences, pointing opposite ways:

1. **Our subframe rate is safe.** Every measured ceiling is 90-110 Hz. The mode
   sweep put the worst pupil position at M = 5.3, i.e. 477 Hz, which is four to
   five times above any flicker fusion frequency in the literature.
2. **Reducing M in the periphery is not free, and the reason is indirect.**
   Cutting M does not change the subframe rate, so it creates no flicker by
   itself. What it leaves is speckle that has not averaged away; if that
   residual is redrawn from a fresh random phase every frame it modulates at
   roughly 90 Hz, which is precisely the peripheral flicker ceiling and
   precisely where the periphery outperforms the fovea. The failure mode is
   peripheral temporal noise, and no purely spatial acuity budget can see it
   coming. Holding the peripheral phase across frames avoids it, at the price
   of a fixed speckle pattern that a moving eye then sweeps across the retina.
   An earlier scratchpad estimate valued peripheral mode foveation at 20.4x;
   that number is spatial-only and is an untested upper bound.

Krajancich et al. (2021, ACM TOG 40:4) additionally show spatial and temporal
foveation are non-separable: a joint model finds about 7x more saving than
spatial-only, which also means a spatial budget cannot be assumed to cover the
temporal axis.

## 7. What would break if we push too far

1. A hard eccentricity cutoff for colour: visible desaturation ring (section 1).
2. Treating red-green and blue-yellow symmetrically. Blue-yellow tolerates more,
   despite the raw table appearing to say red-green can be cut hardest -- the
   one applied study found red-green error the *more* salient of the two
   (colour fringing at moving edges).
3. Stacking peripheral chroma reduction and peripheral subframe reduction in the
   same region (section 6).
4. Stale peripheral chroma after a saccade. A saccade brings a coarsened region
   under foveal scrutiny in about 200 ms; gaze-contingent update budgets are
   50-70 ms. All the psychophysics above assumes steady fixation.
5. Driving a falloff curve from castleCSF beyond 45 degrees. Bowers et al. show
   it is wrong there in three different ways for three different mechanisms,
   because its log-linear eccentricity term cannot capture the observed slowing
   of the decay. Our field's outer rim sits exactly in that zone.

## 8. Models worth using

- Watson's Pyramid of Visibility (2018) -- the standard eccentricity x spatial
  frequency x luminance CSF used in foveated rendering budgets.
- stelaCSF (Mantiuk lab, ACM TOG 2022) -- adds the temporal axis.
- castleCSF (Ashraf, Mantiuk et al., J. Vision 2024) -- adds colour. Fit to 18
  datasets at 3.59 dB RMS. **Wrong beyond 45 degrees**, see above.
- Bowers et al. 2025 raw data, OSF osf.io/d3hw9 -- more trustworthy than any
  unified model past 45 degrees.

## 9. Open, and deliberately not resolved here

- The L:M cone ratio versus eccentricity is genuinely contradictory across
  sources (invariant to 4 deg then rising 70% through 12 deg; no change at
  17 deg; a 2:3 to 1:3 shift by 40 deg). Individual variation runs 1.1:1 to
  16.5:1 (Hofer et al. 2005), which almost certainly dwarfs any eccentricity
  trend. Do not build a wavelength crossover on it.
- Several load-bearing citations were only reachable as abstracts. If an exact
  number from Anderson/Mullen/Hess 1991, Mullen and Kingdom 2002, Duchowski et
  al. 2009, the ASPLOS 2024 paper or Mohanto et al. becomes load-bearing for
  implementation, the PDF needs reading directly.

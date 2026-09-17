# Hogel-Free Holography — full-text notes

Chakravarthula, Tseng, Fuchs, Heide. *Hogel-free Holography.* ACM TOG 2022,
DOI:10.1145/3516428. PDF: `docs/papers/HFH_sig2022.pdf` (16 pages).

Read in full. This paper was cited second-hand in
`docs/research_holo_sota.md` and `docs/research_foveated_holography.md` as the
strongest published argument against a hogel architecture. The full text both
confirms that argument and, on the numbers, disqualifies the proposed
alternative at our panel size.

## Verdict

Hogel-free holography wins on image quality by a wide margin and wins the
argument about hogels, but it cannot be applied to a 150 PPI panel of our
dimensions. Its cost and memory scale with the SLM pixel count of a single
global optimisation that cannot be decomposed — that indecomposability is the
method, not an implementation detail. At 1080p it needs 300 s and most of an
8 GB GPU. At our panel resolution a single complex field is 240 GB. The
architecture we should take from this paper is not the method but the
benchmark: HFH is the quality ceiling our hogel pipeline is being measured
against, and Table 1 quantifies the gap.

## Verified facts

| Quantity | Value | Source |
|---|---|---|
| PSNR, hogel-free holography | 40.2 dB | Table 1, p. 10 |
| PSNR, overlap-add stereogram (Padmanaban 2019) | 30.0 dB | Table 1 |
| PSNR, phase-added stereogram (Yamaguchi 1993) | 12.6 dB | Table 1 |
| PSNR, holographic stereogram (Yatagai 1976) | 13.0 dB | Table 1 |
| SSIM / LPIPS, HFH | 0.964 / 0.140 | Table 1 |
| SSIM / LPIPS, OLAS | 0.810 / 0.417 | Table 1 |
| Hologram resolution optimised | 1080 × 1920 | §4.1 |
| Optimiser | Adam, 800 iterations, lr 0.1 | §4.1 |
| Time per hologram | ~300 s | §4.1, and stated again in Overview of Limitations, p. 2 |
| Hardware | single Nvidia GTX 1080, 8 GB | §4.1 |
| Light field input | 9 × 9 views, RGB-D, rendered at 1080p in Unity | §4.1 |
| WFT window | 9 × 9 Hamming, matched to the view count | §4.1 |
| Scene depth volume | 13 mm (stated as not fundamental) | §4.1 |
| Display prototype | HOLOEYE LETO 1080p LCoS, 6.4 µm pitch | §4.2 |
| Illumination | 630 / 520 / 450 nm laser diodes, single-mode fibre | §4.2 |

Loss (Eq. 12): weighted sum of ℓ2, SSIM, VGG-19 perceptual (relu1_2 and
relu2_2), and Watson FFT.

## Why the method cannot be decomposed

Eq. 1 poses the SLM phase optimisation as a sum over the M×N light field
angular views — a consensus problem. Eq. 2 is the entire contribution: it
recasts that sum as a **single term in the wave domain**, using an operator
σ⁻¹ that maps discrete angular views to one complex wavefront via a Windowed
Fourier Transform, then relaxes the phase constraint into a continuous volume
amplitude constraint solved by first-order SGD.

Collapsing M×N terms into one is exactly what removes the spatio-angular
trade-off, and it is exactly what removes any per-region or per-view structure
to exploit. There is no hogel to thin out, no hemisphere to shrink, no angular
view to skip. The optimisation variable is the whole SLM phase at once.

## Why it does not reach our panel

Our full specification is 512 × 512 hogels at 169.3 µm pitch (150 PPI) with a
0.5 µm sub-pixel pitch, i.e. 338 sub-pixels per hogel side:

```text
panel side          512 × 338.6          = 173,363 sub-pixels
SLM pixel count     173,363²             = 3.0e10  (30.0 Gpixel)
HFH's SLM           1920 × 1080          = 2.07e6  (2.07 Mpixel)
ratio                                     = 14,500×
```

One complex64 field at our resolution is 30.0e9 × 8 B = **240 GB**. Adam needs
the phase plus two moment buffers plus gradients, so roughly a terabyte of
working set. The GPU has 24 GB. This is a memory wall, not a runtime estimate,
so it does not depend on any FLOP extrapolation. Runtime, for what it is worth,
extrapolates to O(100 hours) per frame even granting the RTX 5090 a 5–20×
advantage over the GTX 1080.

Our own pipeline survives at 17.2 Gpixel in 12.6 s precisely because the hogel
decomposition lets it stream: the simulator's own history records "Streaming
scatter architecture: remove large hemi_dev" as the change that made large
grids fit. Hogels are what make the problem tractable at this panel size. That
is the trade the paper is asking us to make, stated in its own terms.

## What the paper gives us anyway

- **A quality ceiling and a scoreboard.** 40.2 dB versus 30.0 dB for OLAS and
  ~13 dB for classical stereograms, averaged over depth planes. Fig. 8 shows
  the failure modes visually: classical stereograms show gross tessellation
  from the hogel grid, OLAS is smooth but rings from double-phase amplitude
  coding. If our hogel reconstructions show tessellation of that kind, the
  cause is diagnosed and named.
- **Robustness to a non-linear phase LUT (§5.3).** Simulating a gamma of 0.8 on
  the SLM phase degrades OLAS substantially but HFH only mildly. This matters
  directly to us: our Sb₂Se₃ levels are not uniformly spaced, and this is the
  only quantified statement in the literature about how CGH methods behave
  under exactly that defect.
- **A verdict on tensor holography (§5.2, Fig. 9–10).** Tensor holography takes
  a single RGB-D image, so it cannot model wavefronts from occluded sources:
  light leaks from background to foreground and rings at depth discontinuities.
  Its depth map also fails on non-Lambertian surfaces (a mirror is rendered as
  a plane). Anyone claiming a fast feed-forward CGH network should be checked
  against this.
- **Input format compatibility.** HFH consumes a 9 × 9 RGB-D light field.
  `tier1_lightfield/cornell_lightfield.py` already produces a 9 × 9 light
  field. Running HFH at 1080p on our own Cornell scene is a cheap, directly
  comparable experiment; it just cannot be scaled to the real panel.

## Consequences for the foveation plan

The paper strengthens rather than weakens the case for foveating hogels.
HFH's cost is governed by the SLM pixel count of one indivisible optimisation,
which foveation cannot reduce without sub-sampling the SLM plane itself. A
hogel pipeline's cost is hogel count × hemisphere pixels, and **both factors
are functions of retinal eccentricity**. The decomposition that costs us image
quality is the same decomposition that makes foveation possible at all.

The honest framing for any future write-up: hogels are a quality compromise
that buys tractability, hogel-free holography is the quality reference that is
intractable at panel scale, and foveation is the argument that the compromise
can be spent where the eye cannot see it.

## Corrections to the earlier surveys

- `research_holo_sota.md` Verdict 1 says hogel decomposition is "a known-inferior
  baseline, not SOTA". Confirmed on quality (40.2 vs 30.0 vs 13 dB) but the
  survey omits that the alternative costs 300 s for a 2 Mpixel hologram and
  does not decompose. Read the verdict as a quality statement only.
- Both surveys list this paper as verified only to abstract level. It is now
  verified in full; the claims above carry page and table references.

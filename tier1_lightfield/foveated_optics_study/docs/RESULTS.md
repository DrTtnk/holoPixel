# Consolidated exploratory results

These numbers are reduced-order-model outputs, not final hardware specifications.

## Retinal sampling

The Watson-derived square-grid-equivalent target pitch is about 0.46 arcmin at the foveal center in the initial neural model. The pitch grows rapidly with eccentricity and is anisotropic across temporal, nasal, superior, and inferior meridians.

For a 70 deg x 45 deg field, the exploratory integration gave about 338k retina-matched 2-D field samples. Uniform foveal sampling over the same field would require about 53.8M field samples.

## Static versus tracked pupil support

Initial bookkeeping with 0.667 mm pupil-plane spacing used 13 x 13 samples across an 8 mm static support versus 7 x 7 nominal samples across a tracked 4 mm support. Combined foveation and steering reduced the idealized simultaneous sample count by hundreds of times relative to uniform foveal field sampling plus a static eyebox.

That 7 x 7 result was useful as a budget estimate, but the later wave-optics work showed that a uniform 7 x 7 view count is not physically appropriate near the fovea.

## Diffraction-aware pupil-view rule

For incoherent light and a circular sub-aperture, a useful ROM is

    nu = lambda K / (2 D DeltaTheta)

where D is pupil diameter, K is approximately the pupil samples per axis, and DeltaTheta is local retinal/angular sample pitch. Combining this with the circular-aperture MTF50 criterion gives a local admissible view count.

At 550 nm, the raw Watson target is slightly finer than the MTF50 bandwidth of an ideal 4 mm pupil in the central fraction of a degree. The optics-aware model therefore caps the neural target where the full pupil cannot support it and increases view count with eccentricity.

See results/green_K_thresholds_by_meridian.csv and results/optics_capped_adaptive_pupil_budget.csv.

## Simple 28 um encoder stress test

A scalar wave-optics model intentionally stressed the naive architecture in which one lenslet directly defines one pupil-channel footprint after 20 mm of free-space propagation. The robust conclusion is that small pupil footprints and very fine retinal PSFs cannot be obtained simultaneously from one incoherent channel. This is the expected etendue / phase-space trade-off.

See results/simple_encoder_waveoptics_sweep.csv and results/simple_encoder_7x7_crosstalk.csv.

## Hexagonal pupil packing

The physical pupil is circular, so a square pupil lattice wastes nominal samples and creates poor edge placement. The preliminary 28 um encoder study compared 7, 19, 37, and 61-view hex packings. Under that simplified model, 19-37 views were a more useful region than 61 because crosstalk rose quickly as spacing shrank.

See results/hex_pupil_packing_waveoptics.csv.

## 2K x 2K prototype

A 2048 x 2048 screen with 7 x 7 physical subpixels per lenslet gives a 2044 x 2044 divisible active region, 292 x 292 lenslets, and 85,264 field samples. This is about 25% of the nominal 338k retina-matched 70 x 45 deg target. The 2K panel is therefore a prototype platform, not the full retinal target.

See results/screen_2k2k_support_summary.csv.

## Diagnostic pinhole render

An aggressive test chart was used so that optical differences were not hidden by natural imagery. In the stored 2K run, crosstalk-only mean absolute luminance error was about 0.0011, while the optics-aware mean absolute error was about 0.0066; the optics-aware 95th percentile was about 0.023 and local peaks were much larger around high-frequency features.

The first-order degradation looked relatively clean rather than catastrophic, but this should not be over-interpreted until the relay/remapper is propagated explicitly.

See results/diagnostic_2k_diff_stats.csv.

## Hex panel variants

The 2K-scale hex tiling exercise produced about 84.7k hex cells across an 8.176 mm square active region in the illustrative 4 um-pixel geometry. The visualization used about 17.4 um hex side length and equivalent local normal tilts up to about 20 deg at the most extreme field positions.

See results/hex_microlens_variants_summary.csv.
# Useful Knowledge

## GTE Phase Physics
- The GTE phase sweeps through exactly 2π per FSR (Free Spectral Range), regardless of finesse/reflectivity
- FSR in terms of Δn is always λ/(2d) — this is a fundamental limit
- Resonance enhancement (finesse F) concentrates the 2π transition into a narrow window of width ~λ/(2dF)
- The enhancement factor F = (1+√R₁)/(1-√R₁) is the small-signal phase sensitivity multiplier
- You MUST be operating at a cavity resonance (δ = 2mπ) for the enhancement to work

## Resonance enhancement in phase-vs-index does not predict phase-vs-wavelength bandwidth ordering
- DRAFT 8.4's DBR-only stack shows green's phase-vs-Sb2Se3-index range enhanced ~30%
  over the bare double-pass value, while red/blue roughly match it. The reasonable
  hypothesis (by analogy with a GTE, where finesse F narrows the phase transition
  width as 1/F) was that green's spectral bandwidth against a 1-2nm laser diode would
  therefore be the narrowest of the three colours. Measured (`tier2_rcwa/spectral_bandwidth.py`,
  `tests/test_spectral_bandwidth.py`), it is the opposite: green has the *widest*
  pi/8 phase-LUT-error bandwidth, both with and without the AR coat. The GTE's 1/F
  law describes the width of the phase-vs-*index* (or phase-vs-detuning-from-resonance)
  transition at fixed wavelength; it does not directly transfer to the width of
  phase-vs-*wavelength* for a LUT whose index values are re-evaluated at a different
  wavelength, because changing wavelength moves the whole cavity's resonance
  condition (both the round-trip phase and each layer's optical thickness), not just
  a single detuning parameter the way index does at fixed wavelength. Do not assume a
  phase-vs-index enhancement factor is also the relevant phase-vs-wavelength
  enhancement factor without checking — they are different derivatives of the same
  phase function and can rank colours in opposite order.
- Corollary, also measured rather than assumed: DBR pair count (4-6) barely moves the
  phase-vs-wavelength bandwidth even though it substantially raises reflectance
  (green: R_mean 0.656 at 4 pairs -> 0.812 at 6 pairs, bandwidth changes <1%). The
  likely reason is that the Sb2Se3/AR front-surface Fresnel reflection (fixed, ~0.25-0.34)
  is already the finesse-limiting "front mirror" of the implicit GTE-like cavity once
  the DBR "back mirror" clears it by 4 pairs — consistent with `tests/test_gte_physics.py`
  proving a GTE's finesse is set by the front mirror alone once the back is reflective
  enough. "Reduce DBR pairs to fix a bandwidth problem" is not a real lever here.

## Phase Unwrapping Pitfalls
- np.angle returns [-π, π]; at the GTE resonance the phase IS at ±π (a discontinuity)
- Starting a phase sweep AT the resonance causes np.unwrap to fail or give misleading results
- Always start the sweep well BEFORE the resonance (at anti-resonance if possible)
- The GTE phase is monotonically DECREASING with increasing δ (increasing n)

## TMM vs Analytical
- Real metal mirrors (Ag) introduce absorption losses that break the ideal GTE assumption (|r|=1)
- The effective R₁ from an Ag thin film depends strongly on wavelength
- Ag absorbs significantly at cavity resonance — reflectance can drop below 0.5
- The analytical GTE model overestimates performance compared to realistic TMM stacks

## DRAFT Corrections
- DRAFT claims F=14 (R₁=0.75) is sufficient — actually need F≥30 (R₁≥0.877) for blue/green, F≥60 (R₁≥0.936) for red
- DRAFT's d=300nm only resonant for blue at n=1.5; not a universal cavity depth
- DRAFT's claim of "Δn=0.05 for 2π" is approximately correct for blue at high finesse, but not for red

## Partial Phase Holography
- 1.84π is indistinguishable from full 2π: η=0.781 vs 0.785, SNR=3.6 vs 3.7
- Even 1.5π gives usable quality: η=0.681, SNR=2.1
- Below π, quality degrades rapidly (η=0.291, SNR=0.4)
- 8 quantization levels with 1.84π gives η=0.742 — still good
- This means R₁=0.90 design is MORE than sufficient, no need to push for full 2π

## Architecture Comparison Key Findings
- SAW (Architecture C) has Δn ~833× too small for per-pixel modulation (1.2e-4 vs 0.1 needed)
- SAW is a BEAM STEERER, not a per-pixel modulator — could work as hybrid
- Si Photonics MZI (Architecture B) fits easily: 6.3µm arm for 2π at Δn=0.1
- GMR (Architecture G "Guitar String") merges phase modulation + beam steering in one layer
- "Didgeridoo" multi-order: d=1053nm with m_R=5, m_G=6, m_B=7 is the magic RGB depth
- Dielectric Bragg mirrors: 4 pairs TiO₂/SiO₂ gives R≈0.935, near-zero absorption

## Actuation Feasibility (CRITICAL)
- LiNbO₃ is TOO WEAK: Δn=0.002 at 3.3V, 300nm cavity → useless without extreme finesse
- BaTiO₃ thin film: Δn=0.02 at 3.3V (d=300nm) → needs F≥30, 10V gives Δn=0.06
- KNbO₃: Δn=0.023 at 3.3V → similar to BaTiO₃ but less studied
- LC nematic: Δn=0.2 → needs F≥2 only! But fringing at 0.5µm pitch is fatal (AR=0.60)
- LC blue phase: Δn=0.05, 0.1ms speed → elegant but low contrast
- **Sb₂Se₃ (PCM): Δn=1.1, k=0.01, non-volatile → NO RESONANCE NEEDED AT ALL**
- MEMS: feasible displacement (7nm) but pull-in voltage is 53V at 0.5µm → CMOS incompatible
- Sb₂Se₃ + simple thin film is the simplest path: 300nm layer gives >2π, no cavity needed
- PCM is quantized (not continuous) but 8 levels give 81% diffraction efficiency
- Sb₂Se₃ switching: 100ns, non-volatile (holds state without power)

## RCWA Tier 2 Lessons

### Phase-to-Index LUT Must Be TMM-Computed
- Naive linear model φ = 4πnd/λ gives η₁ = 44.5% — HALF the theoretical limit
- Cause: air–Sb₂Se₃ interface (R ≈ 25–34%) creates implicit Fabry-Perot with mirror
- Fix: compute actual phase(n) via TMM, then invert to get n(phase) for uniform steps
- TMM-corrected LUT recovers η₁ = 84–91%
- This LUT must be recomputed for each stack variant (different mirror, AR coating, thickness)

### DBR vs Metal Mirror
- Al mirror absorbs 20–50% at resonance — RCWA confirms Al gives only η₁_abs = 0.462
- DBR 6 pairs (TiO₂/SiO₂) gives η₁_abs = 0.635 — 37% improvement
- DBR 2 pairs insufficient (only 0.06π phase range — acts as transparent, not reflective)
- DBR 4 pairs is minimum viable (83.8% relative efficiency)
- Quarter-wave DBR thickness = λ/(4n): TiO₂ = 57.8nm, SiO₂ = 91.7nm at 532nm

### AR Coating
- 72nm MgF₂ (n=1.38) is the sweet spot for the Sb₂Se₃ stack
- Reduces reflectance variation ratio from 3.49 to 1.87
- Improves relative η₁ from 83.8% to 89.8%
- Ideal AR index for n_mid=3.5 would be 1.87 (MgF₂ at 1.38 is a compromise)

### RGB Validation
- d=300nm is the only film thickness where ALL three colors get ≥2π phase range
- d=200nm: red only gets 1.66π (too thin for long wavelength)
- d=250nm: red only gets 1.72π (still insufficient)
- d=300nm: blue=2.70π, green=3.23π, red=2.05π — all sufficient

### Multiprocessing Shared Memory
- `str(np.complex128)` gives `"<class 'numpy.complex128'>"` — NOT a valid dtype string
- Use `np.dtype(np.complex128).str` (gives `"<c16"`) when passing dtype across processes
- This bug causes infinite worker respawning since every worker crashes on init
- Absolute efficiencies: 35–44% depending on color and stack configuration

### grcwa Practical Notes
- nG=51 is sufficient for 8-level 1D blazed grating (results within ~2% of nG=101)
- nG=101 with 8+ DBR layers is EXTREMELY slow (minutes per solve)
- Units: grcwa uses µm for lengths, freq = 1/λ_µm
- Grid: Nx = n_levels × 20, Ny = 20; indexing='ij' in meshgrid
- normalize=1 for energy conservation (R+T=1)
- Layers added in order: first = incidence side

### OPA + PCM Architecture
- Silicon is OPAQUE at visible wavelengths (bandgap 1.1eV = 1127nm). Must use Si₃N₄/TiO₂/LNOI for visible OPA.
- PCM waveguide phase shifter loss for 2π shift is a FUNDAMENTAL constant: loss_dB = 4.343 × 4π × k_avg / Δn. For Sb₂Se₃: always 0.99dB regardless of geometry. Only improvable by finding a material with higher Δn/k ratio.
- OPA splitting loss is 10·log₁₀(N) dB — unavoidable with guided-wave splitting. Free-space illumination has no splitting loss (plane wave hits all pixels). This makes OPA impractical for large arrays (>~1000 emitters).
- Marcatili method for waveguide n_eff: MUST include penetration depth correction: kx = π/(w + 2/γ) where γ = k0·sqrt(n_core² - n_clad²). Without it, n_eff can be below n_clad (unphysical).
- Crosstalk between Si₃N₄ waveguides at 500nm gap: coupling length ~276µm. Need ≥636nm gap for -20dB over 85µm parallel runs.

### Holographic Parallax Formula
- Phase pattern exp(+ikR) produces a DIVERGING (real-image) wavefront, not converging.
- Parallax slope = z_point / (z_observer + z_point), NOT z_point / (z_observer - z_point).
- Derivation: brightest hogel at hx where (hx - px)/pz = (ox - hx)/oz → hx = (pz·ox + oz·px)/(oz + pz) → d(hx)/d(ox) = pz/(oz + pz).
- This means points FARTHER from display (closer to observer) show MORE parallax — they appear more "3D".
- Verified: 256×144 hogels, 32×32 sub-pixels. NEAR 7%, MID 7%, FAR 1% error vs theory.


## Session 2026-09-17 — Environment Revival and First Test Suite

### Wrong assumptions found by writing tests

- **`quantize_phase` produced one level fewer than requested at a full 2π range.**
  `np.linspace(0, max_phase, n_levels)` places a sample at both 0 and 2π, which are
  the same physical phase, so `n_levels=8` delivered 7 distinct phases. The fix
  branches on whether the range wraps: at 2π the levels are `arange(n)·(2π/n)` and
  the nearest level must be found on the circle, not on the line, so that 2π−ε maps
  to level 0. Below 2π the range does not wrap and `linspace` with both endpoints is
  correct. `rcwa_optimization.design_phase_lut` never had this bug — it already used
  `linspace(0, usable·(1−1/n_levels), n_levels)`.
  Only the last sample of the partial-phase sweep moved; the 1.84π conclusions stand.

- **A sweep that starts on a resonance gives a misleading transition width.**
  This file already warned about `np.unwrap`, but the trap is wider than unwrapping:
  with d=355nm, λ=532nm the natural starting point n=1.5 sits at δ/2π = 2.002, i.e.
  on resonance. The 2π drop then straddles both ends of the sweep and a 10%–90%
  width measurement reports the whole FSR regardless of finesse. Start every sweep
  at the anti-resonance (half-integer δ/2π) so the resonance lands in the middle.

- **`effective_index_slab` uses n_core = 2.0 for Si₃N₄, not the textbook 2.05.**
  The DRAFT's quoted n_eff = 1.750 only reproduces with the value in `PLATFORMS`.
  Import the constant, never retype it.

### Newly proven results (in tests/, derived with sympy, not asserted)

- dφ/dδ = (r²−1)/(r²−2r·cos δ+1) for the GTE, and on resonance this is exactly −F
  with F = (1+r)/(1−r). "F is the small-signal phase sensitivity multiplier" is now
  a theorem in the repository, not a note.
- The 10%–90% transition width obeys `width · F / FSR → 2·tan(2π/5)/π = 1.95932`.
  Near resonance φ = −(1+r)/√r · arctan(√r·u/(1−r)), so the constant is closed form.
  The old note "width ~ λ/(2dF)" is right to within that factor of 1.96.
- N-level blazed grating efficiency η = (sin(π/N)/(π/N))². N=8 gives 0.9505, which is
  where DRAFT §6.4's "95%" comes from. N=4 gives 0.8106 — note that
  useful_knowledge's earlier line "8 levels give 81% diffraction efficiency" quotes
  the 4-level number. Treat 81% as a typo unless a source says otherwise.
- An FFT of a staircase sampled with M points per level gives
  η = [sin(π/N)/(M·sin(π/(N·M)))]², not the sinc form. At M=8 the difference is 0.08%,
  which is enough to break a 1e-6 tolerance. Test against the discrete form.
- The parallax slope pz/(oz+pz) survives symbolic re-derivation and matches a real
  96×4 hogel render to within 25%.

### Toolchain

- Python 3.14 has no numba wheel. The project pins 3.12 via `uv venv --python 3.12`.
- RTX 5090 Laptop is sm_120 (Blackwell). torch 2.14.0+cu130 detects it; a default
  CPU-index wheel will not. Pass the cu128 extra index with
  `--index-strategy unsafe-best-match`.
- `uv pip install` defaults to a 30s HTTP timeout, which is not enough for the
  ~500MB cuDNN wheel. Set `UV_HTTP_TIMEOUT=300`.

## Session 2026-09-18 (continued)

### Wrong: a light field's coherent mode count is small

Measured on 1D strips of the Cornell light field, the mode count looked like
9 to 16 for 90% of the energy, and appeared to track the number of views. That
was an artefact of my own preprocessing: I embedded 9 or 17 angular samples
into a 127- or 255-wide frequency grid by ZERO PADDING. Padding asserts the
light field is exactly zero at every other angle, which band-limits it by
construction and forces low rank. I noted the padding as a caveat and then
reported the number anyway.

On the proper square grid (17 spatial by 17 angular samples, no padding) the
same scene needs 231 to 256 modes out of 289, and the eigenvalue spectrum is
nearly flat.

The reason is physics, not numerics. A path-traced diffuse scene under an area
light is spatially INCOHERENT, so its mutual coherence matrix is essentially
diagonal, and a diagonal matrix has full rank with equal eigenvalues. Measured
off-diagonal energy: 0.0073 at an occlusion edge, 0.0109 on a plain wall,
0.0395 through the glass ball, against 0.0000 for a synthetic fully incoherent
field and 0.9925 for a single coherent one.

Consequence: a coherent-mode decomposition cannot be used to build a target
wavefront from a light field. There is no coherence to decompose. Depth in
hogel-free holography is not recovering the scene's coherence either -- it is
a heuristic phase assignment that happens to also produce correct focus cues.
Without depth the phase must come from optimisation instead.

The glass ball carrying 4-5x more off-diagonal energy than the diffuse regions
is real and worth remembering: refraction correlates neighbouring points, so a
dielectric is the most spatially coherent thing in the scene.

### Wrong: the periphery is the forgiving place, so cut subframes there

I was heading toward spending fewer time-multiplexed subframes in the periphery,
on the reasoning that if peripheral vision resolves less detail it must be the
cheap place to cut everything.

That is right for the spatial axis and backwards for the temporal one. Photopic
critical flicker fusion RISES from the fovea outwards, by 5 to 15 Hz, before
falling again past 30 to 60 degrees (Hartmann, Lachenmayr and Brettel 1979), and
the rise survives cortical-magnification scaling (Rovamo and Raninen 1984), so it
is a genuinely faster temporal channel rather than a receptive-field artefact.
Peripheral vision is coarser in space and FASTER in time.

Krajancich et al. (2021) additionally show the two axes are non-separable, so a
spatial foveation budget cannot be assumed to cover the temporal one.

The absolute rate is still fine: every measured flicker ceiling is 90 to
110 Hz and the mode sweep's worst case is 477 Hz.

The precise risk, stated carefully rather than as a slogan. Cutting M in the
periphery does not by itself create flicker -- the subframe rate is unchanged.
What it leaves behind is speckle that has not averaged away, and if that
residual pattern is redrawn from a new random phase every FRAME it modulates at
the frame rate, roughly 90 Hz, which is exactly where the peripheral flicker
ceiling sits and exactly where the periphery beats the fovea. So the failure
mode is peripheral temporal noise, not peripheral blur, and it is invisible to
any purely spatial acuity budget. Holding the peripheral phase across frames
instead of resampling it would avoid this, at the cost of a fixed speckle
pattern that a moving eye would then sweep across the retina.

An earlier scratchpad estimate put foveating the mode count at 20.4x. That
figure is spatial-only and does not account for any of the above, so treat it
as an upper bound that has not been tested against the temporal axis.

### Wrong: colour vision stops in the periphery, so we can drop wavelengths there

The "colourblind periphery" claim comes from Ferree and Rand (1919) and Moreland
and Cruz (1959), both using SMALL targets. Gordon and Abramov (1977) showed that
at 45 degrees a large target recovers the full range of hues with fovea-like hue
functions; Bowers, Gegenfurtner and Goettker (2025) found colour vision present
to at least 75 degrees and attributed the older result directly to stimulus size.

Our entire screen lies inside 46 degrees, so there is nowhere in our field where
a wavelength can be dropped. A hard cutoff would show as a desaturation ring, and
would be worse than a graded falloff exactly because large uniform peripheral
colour is genuinely seen.

What is true instead is narrower and less exciting: chromatic resolution is
coarser than luminance resolution EVERYWHERE including the fovea (11-12 c/deg
against 30-60), and red-green fine detail specifically becomes behaviourally
absent by 25 to 30 degrees while blue-yellow tracks the luminance falloff. The
only work that validated peripheral chroma reduction against real observers
(Mohanto et al. 2026) got 31-37%, not the 2-3x the raw sensitivity ratios imply.

### Wrong: foveate by low-passing the target

I built foveation as a space-variant blur of the TARGET: a Gaussian pyramid, with
the level at each retinal sample set by the eye's resolution limit there. The
reasoning in the docstring was that merely down-weighting the periphery still
asks the optimiser for detail the eye cannot resolve and then forgives it for
failing, whereas low-passing the target stops asking and frees the panel's
degrees of freedom for the fovea.

That reasoning is wrong and the experiment says so. At panel 8192 / window 2048,
3000 iterations, same seed, the foveated solve was WORSE than the uniform one at
the fovea by 2.4 to 5.7 dB across three pupil positions, and it won its own
perceptual metric at only one of the three.

Two reasons, and the second is the deeper one.

1. At one subframe the error is SPECKLE, not a shortage of capacity, and
   foveation cannot touch speckle. Étendue says the panel is 1.8x over-provisioned
   for a single pupil, and 16 subframes reach 47.5 dB at the same scale, so
   capacity was never the binding constraint. Repeating the comparison at M = 4
   did move foveation into positive territory, which supports this.
2. Each view is blurred about ITS OWN centre, so the foveated target set is
   mutually inconsistent across views: no single physical field can produce it.
   The unblurred light field is realisable by construction, because a real scene
   produced it. Blurring makes the target easier to score and harder to achieve.

What the literature actually does (Chakravarthula et al., arXiv:2108.06192,
Eq. 8-9) is the opposite of what I did. They keep the full-detail target and
apply a per-pixel WEIGHT proportional to midget ganglion cell density, so the
fovea simply counts for more. Separately they convolve the RECONSTRUCTION with
the eye's optical point spread function (a Gaussian of 0.6 pixels) before
comparing it with the sharp target, so the panel must produce a field that is
correct after the eye blurs it. Their own ablation finds the PSF term is worth
more than the foveation term.

Blurring the target and weighting the loss are not two routes to the same place.
Weighting reallocates effort; blurring destroys information.

### Process: read the equation the survey points at, before writing the code

`docs/research_foveated_holography.md` already named Chakravarthula et al.
(arXiv:2108.06192) as THE foveated-CGH reference, with the relevant equation
numbers, "Eq. 8-9", written out. I designed and built a foveation scheme
without opening it, got a negative result, and only then read the paper --
which says plainly to weight the loss and keep the full-detail target, the
opposite of what I had built.

The survey was in the repository and correct. The cost was a wrong
implementation and roughly two hours of GPU time, both avoidable by reading one
equation first.

Rule: when a survey in this repository points at a specific equation in a
specific paper, read that equation before writing the code that replaces it.

### Measured: the Gaussian rasteriser was launch-bound, not arithmetic-bound

The per-primitive Python loop cost about 100 microseconds per Gaussian on the
CPU and 435 on the GPU, and -- the diagnostic detail -- the SAME whether the
splat covered 4 pixels or 21. A cost independent of the work done is not
arithmetic, it is overhead. That is also why the GPU was four times slower than
the CPU: thousands of tiny kernels, each mostly launch latency.

Compositing a chunk of primitives at once, with an exclusive cumulative product
of (1 - alpha), took 4000 Gaussians at 256 squared from 1764 ms to 41 ms on the
GPU, a factor of 43.

The same change is 25x SLOWER on the CPU (399 ms to 10042 ms), because dense
evaluation does about three thousand times more arithmetic for a four-pixel
splat and a CPU has none to spare. Accepted deliberately: the real workload is
on the GPU, and a second compositing path would be a correctness risk for a
case nobody uses.

Lesson worth keeping: when a cost per item does not move with the size of the
item, stop optimising the arithmetic and count the launches.

### A test that passes because it renders nothing

The equivalence test for the vectorised rasteriser put its Gaussians at z = +6
and used the test-suite camera helper, which looks along -Z. Every primitive
was behind the camera, so both the new and the reference implementation
returned a blank image, and the test compared zeros with zeros and passed.

It would have kept passing through any rewrite of the compositing maths.

The fix is not just the sign: every image-comparison test now asserts that the
render is non-blank BEFORE comparing. A comparison test needs a liveness check,
or it silently becomes a tautology.

### Tiling beat both of its predecessors, and the middle version was a detour

The Gaussian rasteriser went through three forms, and the measured numbers for
4000 primitives at 256 squared on the GPU tell the whole story:

    per-primitive Python loop   1764 ms    435 us per primitive
    chunked dense over frame      41 ms     10 us
    tiled                        3.9 ms   0.98 us

The middle version traded the right thing (kernel launches) for the wrong one
(work proportional to FRAME area rather than projected area). It was a 43x win
on the GPU and a 25x loss on the CPU, and the CPU loss was the signal that the
scaling was wrong -- a change that helps one device that much and hurts the
other that much is not a clean win, it is a trade.

Tiling gets both: work follows projected area again AND the launches stay
batched. It also made the CPU regression mostly go away (848 ms against the
original loop's 399) without a second code path.

The lesson is about the intermediate step rather than the destination. Shipping
the dense version was still right -- it was small, it was provably equivalent,
and it made the real bottleneck legible. But the 25x CPU regression should have
been read immediately as "the scaling is wrong" rather than as "the CPU does
not matter". The second reading is the comfortable one and it was nearly the
one I kept.

### Block grouping matters more than block count

Splitting the pupil views into B groups, with an exact diagonal solve inside a
group and the updated panel between groups, interpolates between the Jacobi
solver (B = 1) and Gerchberg-Saxton (B = number of views). More groups is
broadly better, and B = 12 beat both ends at 12/12 against 1/12 and 11/12.

But B = 6 collapsed to 3/12, badly out of trend. The cause was not the count.
The views are indexed row-major on a 6x6 pupil grid, so grouping by stride 6
put an ENTIRE COLUMN of pupil positions in each block. Those views share one
x-offset, so their windows overlap heavily in x and tile in y: the block
constrains a narrow vertical strip of the panel and leaves the rest untouched,
and information propagates poorly between groups.

Regrouping the same B = 6 at random: 3/12 becomes 11/12.

The lesson generalises beyond this solver. When an experiment sweeps a count and
one value falls out of trend, suspect that the count interacted with a hidden
periodicity in the indexing before concluding anything about the count itself.
Here the stride and the grid width were both 6, which is exactly the kind of
coincidence that produces a clean-looking but meaningless data point.

### The test suite was ten times slower for using every core

The suite ran 313 s at 1773% CPU. Single-threaded it runs 30 s at 99% CPU: a
factor of 10.4 gained by using LESS of the machine.

These tests are hundreds of operations on tiny tensors -- 16x16 panels, 6x6
windows. Torch fans each operation across all 18 cores and then spends longer
synchronising than computing. The arithmetic was never the cost.

Fixed in `tests/conftest.py`, which sets OMP_NUM_THREADS and MKL_NUM_THREADS
before torch imports and calls `torch.set_num_threads(1)`.

This is the same lesson as the Gaussian rasteriser, in a different costume:
there, cost per primitive did not change with primitive size, and the answer was
that launches rather than arithmetic dominated. Here, cost did not fall when
cores were added. Both times the tell was a cost that ignored the quantity it
should have depended on. When work is small, parallelism is a tax.

### Mathlib v4.30.0-rc1 has no `Complex.abs`

It was removed. Use `Complex.normSq` and real coordinates instead, which also
turns out to make the phase-retrieval proofs far more tractable: the projection
arguments are about squared distances anyway, so avoiding the square root
avoids the side conditions it drags in.

Also: `lake build Holopixel.<NewFile>` works without registering the file
anywhere, because Lake globs the whole `Holopixel/` directory for the library
target. New theorem files need no edit to `Holopixel.lean` or `lakefile.toml`.

## Session 2026-09-23 — Foveated hex MLA Blender scene

### Wrong: the lenslet focal length is the eye relief

I set the MLA focal length to f0 = 20 mm because `waveoptics_encoder.py` uses
a 20 mm throw to the pupil. That throw is the distance from the lenslet to the
pupil, not the focal length. In an integral-imaging screen the panel sits at
the lenslet's focal plane, so f equals the panel-to-lens GAP, and each pixel
under a lenslet becomes one collimated direction. The gap follows from how far
the views must spread: the ~28 um of pixels under one lenslet must cover the
4 mm pupil at 20 mm, i.e. about 0.2 rad, so f is about 28 um / 0.2 = 0.14 mm,
not 20 mm. With n = 1.5 that gives a radius of curvature of about 70 um and a
real sag of about 1.6 um, which is visible without any exaggeration. The x15
"visual exaggeration" I added only existed to hide the wrong focal length.

Rule: before deriving a lens from a system distance, write down which two
planes that lens must image onto each other.

### Blender: one object per lenslet is quadratic

Creating 169k objects one at a time (`bpy.data.objects.new` plus a collection
link each) ran for more than 80 minutes and never finished. One merged mesh per
variant built in 2.5 s. For large arrays use one mesh, or Geometry Nodes
instancing, never one object per element.

### Blender 5.2 / Cycles traps met while building the physical model

- Setting `colorspace_settings.name` on a GENERATED image rebuilds its buffer
  and silently zeroes pixels written earlier with `foreach_set`. Set the
  colorspace first, then write, then read back and compare.
- A new World renders from its node tree; `world.color` is ignored. Set the
  Background node's colour. The default is grey 0.05, which quietly broke a
  threshold measurement.
- Depth of field has only `aperture_fstop`. The aperture radius in scene units
  is `lens_mm * 1e-3 / (2 * fstop)` and does NOT follow `unit_settings.scale_length`:
  with 1 unit = 1 mm, the obvious `lens / (2 * fstop)` is 1000x too large.
  Measured with a defocused point: 1.91 mm for a 2 mm target, 0.49 for 0.5.
- In a world shader, `Texture Coordinate > Generated` is the unit world-space
  ray direction. Render row 0 is the bottom of the frame.
- The Refraction BSDF DROPS the path under total internal reflection (black),
  and the Glass BSDF picks Fresnel reflection at random. For a deterministic,
  lossless dielectric: mix Refraction with a sharp Glossy, factor
  `Fresnel(IOR) > 0.9999`. Verified on a 45 degree prism: the pupil points it
  loses are exactly the ones Snell's law puts below the critical angle.
- Russian roulette ends 1-sample paths at random after a few bounces, even at
  unit throughput. Set `min_light_bounces` to the bounce limit.
- The Glossy BSDF's default colour is 0.8, not 1.0, and its default
  distribution is multiscatter GGX.
- A hand-wound glass solid came out wound inwards, and Cycles then refracts with
  the index inverted, silently. The evaluator now rejects open or inward glass.

### Wrong: a pinhole view proves a light-field screen

With the panel at the lenslet focal plane every pixel is one collimated beam,
so for content at infinity the direct-view screen resolves one pixel BIN,
pixel / f = 4 / 150 = 26.7 mrad, not the lens pitch. A pinhole camera still
shows a crisp lens mosaic, because it samples one ray per lens. Measured: a
linear light field comes back with mean error exactly quantum / 4 = 0.0222,
the uniform-quantisation value. Only a remapper with the lenses at its focal
surface turns lens = field sample and pixel = pupil view.

### Wrong: estimate a lens's direction as the robust mean over the pupil

The pupil patch that sees a lens's pixels is its hexagon scaled by L / f, and
it shifts with the lens position, so the all-ray mean is biased, and
anisotropically (the hexagon is wider across corners than across flats). An
inlier window around that biased mean locked onto it: vertical scale came out
26 mm instead of 20. The physical definition is simpler and unbiased: a lens's
field direction is where it is seen from the pupil centre.

### Hex grid spacing: orientation and formula must match

Vertices at 30 + 60*i degrees give a POINTY-TOP hexagon. Its tiling is
dx = sqrt(3)*R within a row and dy = 1.5*R between rows, with odd rows offset
by dx/2. The original script used the flat-top formulas (dx = 1.5*R,
dy = sqrt(3)*R), which overlapped neighbours horizontally.

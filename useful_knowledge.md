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

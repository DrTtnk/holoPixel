# Useful Knowledge

## GTE Phase Physics
- The GTE phase sweeps through exactly 2π per FSR (Free Spectral Range), regardless of finesse/reflectivity
- FSR in terms of Δn is always λ/(2d) — this is a fundamental limit
- Resonance enhancement (finesse F) concentrates the 2π transition into a narrow window of width ~λ/(2dF)
- The enhancement factor F = (1+√R₁)/(1-√R₁) is the small-signal phase sensitivity multiplier
- You MUST be operating at a cavity resonance (δ = 2mπ) for the enhancement to work

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


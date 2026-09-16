# Tier 1 Findings: GTE Cavity Feasibility Analysis

## Executive Summary

The Gires-Tournois Etalon (GTE) architecture **is feasible** for achieving 2π phase
modulation with Δn ≤ 0.1, but the DRAFT's parameters need significant revision.
The required mirror reflectivity is **much higher than F=14** (R₁=0.75), and the
DRAFT's cavity depth of 300 nm only works for blue light.

## Key Findings

### 1. Resonance Condition is Critical

The GTE only provides rapid phase transitions when the round-trip phase
δ = 4πnd/λ is near a resonance (δ = 2mπ). At n=1.5 and d=300nm:

| Color | λ (nm) | δ/2π | On resonance? |
|-------|--------|------|---------------|
| Blue  | 450    | 2.000 | **YES (m=2)** |
| Green | 532    | 1.692 | No |
| Red   | 632    | 1.424 | No |

**Implication**: d=300nm is only resonant for blue. Each color needs its own
resonant depth, or a common deeper cavity must be used.

### 2. Required Mirror Reflectivity (at nearest resonant depth)

| Color | d_res (nm) | m | R₁ for Δn≤0.1 | F needed |
|-------|-----------|---|---------------|----------|
| Blue  | 300       | 2 | ≥ 0.877       | ≥ 30x    |
| Green | 355       | 2 | ≥ 0.877       | ≥ 30x    |
| Red   | 211       | 1 | ≥ 0.936       | ≥ 60x    |

The DRAFT's assumption of F=14 (R₁=0.75) is **insufficient**. Actual requirement
is F ≥ 30-60x depending on wavelength.

### 3. DRAFT Error: Enhancement vs FSR

The DRAFT conflates resonance enhancement with FSR reduction. The GTE phase
sweeps through exactly 2π per Free Spectral Range (FSR), regardless of finesse.
The FSR in terms of Δn is always λ/(2d). What the resonance enhancement does
is **concentrate** the 2π transition into a narrow window near the resonance peak.

The formula Δn ≈ λ/(2dF) correctly describes the width of this transition zone,
not a reduction in the fundamental FSR.

### 4. Silver Mirror Absorption (TMM Finding)

Real silver (Ag) mirrors introduce absorption losses at the cavity resonance.
At d=300nm with 25-50nm Ag top mirror:
- Reflectance drops to 0.5-0.8 at resonance (~530nm)
- This means **significant light is absorbed**, not just phase-modulated
- The ideal GTE (|r|=1) assumption breaks down

**Impact**: Lower optical efficiency, amplitude modulation mixed with phase
modulation, reduced effective finesse.

### 5. Depth Optimization Results (R₁=0.85, F=25x)

At moderate reflectivity, the minimum cavity depth for 2π within Δn≤0.1:
- Red: d ≥ 510 nm
- Green: d ≥ 430 nm  
- Blue: d ≥ 365 nm

A **common depth of ~510-600 nm** could potentially work for all three colors,
but each color would need to be tuned to a different resonance order.

## Design Recommendations

1. **Increase cavity depth** from 300nm to ~500-600nm for RGB compatibility
2. **Target R₁ ≥ 0.90** (F ≥ 38x) for comfortable margin
3. **Consider dielectric mirrors** instead of Ag to avoid absorption losses
4. **Red is the bottleneck** — consider using a separate cavity depth for red sub-pixels
5. **Validate with TMM** using actual material stack (dielectric Bragg reflector data)

## Open Questions for Tier 2

- How does the nonlinear phase response affect hologram quality?
- Can we use partial phase coverage (< 2π) with quantized phase levels?
- What is the trade-off between cavity depth and fringing field crosstalk?
- Do dielectric mirrors (DBR) provide sufficient bandwidth for all three colors?

## Generated Plots

| Plot | Description |
|------|-------------|
| `gte_feasibility.png` | Required Δn vs R₁ for each wavelength |
| `gte_operating_window.png` | Phase curves with Δn=0.1 window marked |
| `gte_depth_optimization.png` | Cavity depth sweep showing feasible ranges |
| `gte_resonant_phase.png` | Full sigmoid phase response at resonant depths |
| `resonance_map.png` | Which depths are resonant for which colors |
| `tmm_reflectance_spectrum.png` | Ag mirror absorption at cavity resonance |
| `tmm_vs_analytical.png` | TMM realistic vs analytical ideal comparison |

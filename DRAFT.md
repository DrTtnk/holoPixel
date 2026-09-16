
Here is a comprehensive markdown document designed to be fed directly into your GitHub agent (like Copilot Workspace, AutoGPT, or Aider). It contains all the necessary context, mathematical constraints, theoretical frameworks, and a structured simulation roadmap using strictly Free and Open-Source Software (FOSS).

***

# Project Bootstrap: 150 PPI True Holographic Pixel (Hogel) R&D

## 1. Project Objective & Core Specifications

The objective is to simulate and design a single, self-contained holographic pixel ("hogel") capable of generating a true 3D interference pattern using active nanostructures. This hogel will serve as the building block for a flat-panel holographic display.

**Target Hardware Specifications:**

* **Macro-Resolution:** 150 PPI (Pixels Per Inch) for the overall screen.
* **Hogel Physical Size:** 169.3 µm x 169.3 µm (derived from 150 PPI).
* **Target Field of View (FoV):** $\pm 30^\circ$ (60° total).
* **Sub-Pixel / Fringe Pitch:** $\sim 0.5$ µm (calculated via grating equation to achieve 30° FoV for visible light).
* **Sub-Pixel Density:** $\sim 314,000$ active sub-pixels per RGB hogel.
* **Refresh Rate:** 1 FPS (Low refresh rate chosen to eliminate computational/bandwidth bottlenecks and allow non-traditional slow-switching materials).
* **Data Bandwidth:** $\sim 2.5$ Mbps per hogel at 1 FPS.

## 2. Theoretical Pitfalls at the 0.5 µm Scale

Standard spatial light modulator (SLM) architectures fail at the required $0.5$ µm sub-pixel pitch. The agent must account for these failure modes in simulations:

1. **Mechanical MEMS (Piston-Mode Mirrors):**
    * *High-Voltage/Transistor Paradox:* Electrostatic force at $0.5$ µm is incredibly weak. Overcoming this requires high voltages (10V-15V), but high-voltage CMOS transistors cannot physically fit under a $0.5$ µm footprint.
    * *Mechanical Snap-Down:* Pulling a parallel-plate actuator down past 1/3 of its gap causes the electrostatic force to overwhelm the spring, crashing the mirror permanently.
    * *Fill-Factor Loss:* Gaps required between $0.5$ µm moving mirrors cause massive zeroth-order light leakage (blinding the viewer).
2. **LCoS (Liquid Crystal on Silicon):**
    * *Fringing Fields (Electrical Crosstalk):* To achieve a full $2\pi$ optical phase delay, standard LCs require a depth of $\sim 2$ µm. A $0.5$ µm wide by $2.0$ µm deep pixel acts as a narrow column. Electric fields bow outwards, causing adjacent sub-pixels to blur together.
    * *Birefringence Limits:* Shrinking the depth to avoid crosstalk results in insufficient optical path delay to achieve a $2\pi$ shift.

## 3. The "Acoustic Analogy" Solution Framework

To bypass the pitfalls above, the R&D phase will test three architectures inspired by analog acoustic/musical instruments mapped to nanoscale silicon lithography.

### Architecture A: The "Pipe Organ" (Resonant Micro-Cavity Metasurface)

* **Audio Analogy:** Sound folding back on itself inside a tube to resonate, amplifying the wave in a small physical space.
* **Optical Counterpart:** The **Gires-Tournois Etalon (GTE)**.
* **Mechanism:** Trap an electro-optic material (or LC) inside a $300$ nm cavity bounded by a high-reflector at the bottom and a partial-reflector at the top.
* **Hypothesis:** The light bounces ~10-15 times. A required $\Delta n$ of $0.05$ (easily achievable with low voltage) will result in a full $2\pi$ phase shift inside an ultra-shallow $300$ nm layer, completely eliminating electrical crosstalk. **(Primary Candidate)**

### Architecture B: The "Trumpet Valve" (Silicon Photonics)

* **Audio Analogy:** Valves redirecting waves to different paths away from the bell.
* **Optical Counterpart:** Mach-Zehnder Interferometers & Grating Couplers.
* **Mechanism:** Use standard CMOS to build horizontal photonic waveguides. Modulate the phase horizontally (where space is infinite), then route the light to a passive $0.5$ µm grating coupler that shoots it vertically out of the screen.

### Architecture C: The "Chladni Plate" (Acousto-Optics)

* **Audio Analogy:** Vibrating a 2D metal plate to create complex standing wave patterns.
* **Optical Counterpart:** Surface Acoustic Waves (SAW).
* **Mechanism:** Print Interdigitated Transducers (IDTs) on piezoelectric glass. Fire GHz RF signals to create high-resolution 2D physical acoustic standing waves on the screen surface. Light diffracts off these invisible sound ripples.

## 4. FOSS Simulation Roadmap & Agent Instructions

The GH agent should scaffold a Python-based repository divided into three simulation tiers, using strictly open-source tools.

### Tier 1: Analytical 1D Profiling

* **Goal:** Rapidly sweep parameter space (thickness, refractive index, voltage) to validate the "Pipe Organ" (GTE) architecture.
* **Tools:** Python, `numpy`, `matplotlib`, `tmm` (Transfer Matrix Method package).
* **Agent Task 1:** Write a Python script to simulate a GTE cavity at $632$nm (Red), $532$nm (Green), and $450$nm (Blue). Find the exact cavity depth ($d$) and top-mirror reflectivity ($R_1$) that yields a $2\pi$ phase shift for an index change ($\Delta n$) of $\le 0.1$.

### Tier 2: 2D/3D Rigorous Coupled-Wave Analysis (RCWA)

* **Goal:** Simulate periodic arrays of these pixels to measure diffraction efficiency, Field of View (FoV), and fill-factor penalties.
* **Tools:** **S4** (Stanford Stratified Structure Solver) via Python bindings, or **Reticolo** (via GNU Octave).
* **Agent Task 2:** Create an RCWA script mapping a $10 \times 10$ array of $0.5$ µm sub-pixels. Apply a blazed grating phase profile (staircase pattern). Simulate light entering the array and plot the far-field diffraction angles. Quantify how much light is lost to the zero-order (undiffracted) beam.

### Tier 3: Finite-Difference Time-Domain (FDTD)

* **Goal:** Full 3D electromagnetic simulation of the pixel to hunt for fatal physical flaws (fringing fields, parasitic capacitance, edge scattering).
* **Tools:** **MEEP** (MIT FDTD tool, via Python `meep` library).
* **Agent Task 3:** Define a 3D simulation cell containing two adjacent $0.5$ µm GTE cavities. Assign a voltage/index state to Pixel A and a different state to Pixel B. Fire a plane wave source from the bottom. Output an animation (`.h5` to `.gif`) of the Electric Field ($E_z$) to visually inspect optical crosstalk at the boundary between the two pixels.

## 5. Repository Bootstrap Commands

*Agent: Please initialize the repository with the following structure and dependencies.*

```bash
mkdir hogel_sim
cd hogel_sim
mkdir -p tier1_tmm tier2_rcwa tier3_fdtd docs

# Create requirements.txt
cat <<EOT >> requirements.txt
numpy
scipy
matplotlib
tmm
h5py
# Note: S4 and MEEP usually require conda/mamba installations.
# Instructions for MEEP: conda install -c conda-forge pymeep
EOT

# Create README.md based on this document.
```

**Initial Prompt for the GH Agent (Once initialized):**
*"Agent, using the parameters in this document, please implement the Tier 1 Python simulation using the TMM (Transfer Matrix Method). Generate a plot showing the phase shift of a 300nm cavity at 632nm wavelength, sweeping the refractive index from 1.5 to 1.7. Let me know what top-mirror reflectivity is required to hit the 14x resonance enhancement mentioned in the theory."*

---

## 6. R&D Findings (Tier 1 Complete)

### 6.1 GTE Cavity — Corrections to Original Assumptions

The original hypothesis (F=14, R₁=0.75, d=300nm, Δn=0.05) contained three errors discovered during simulation:

| Parameter | DRAFT Value | Corrected Value | Why |
|-----------|-------------|-----------------|-----|
| Enhancement F | 14 (R₁=0.75) | **≥30–60** | F=14 only concentrates 2π into Δn≈0.23 window, not 0.05 |
| Cavity depth d | 300nm (all colors) | **Per-color resonant depths** | d=300nm is only resonant for blue (λ=450nm). Red and green need different depths |
| Required Δn | 0.05 | **0.1 (with F≥30)** | Lower Δn needs higher F, which needs R₁≥0.93 |

**Corrected per-color resonant depths** (at R₁=0.90, m=2):

* Red (632nm): d = 421.3nm
* Green (532nm): d = 354.7nm
* Blue (450nm): d = 300.0nm

**Mirror reality**: Real Ag mirrors lose 20–50% reflectance at resonance due to absorption. **DBR mirrors** (4 pairs TiO₂/SiO₂) achieve R≈0.935 without absorption. Total DBR thickness ~500–710nm.

### 6.2 Architecture Survey (8 Architectures Explored)

| ID | Acoustic Analogy | Optical Mechanism | Result |
|----|-----------------|-------------------|--------|
| A | Pipe Organ | GTE micro-cavity | ✅ Works but needs high-R mirrors and per-color depth |
| B | Trumpet Valve | Si Photonics MZI | 6.3µm arm for 2π. Too large for 0.5µm pitch |
| C | Chladni Plate | SAW acousto-optics | Δn=1.2×10⁻⁴, 833× too weak for phase modulation |
| D | Whispering Gallery | Microring resonator | 0.67µm radius feasible but requires waveguide coupling |
| E | Helmholtz | Slot waveguide | 21µm for 2π. Way too large |
| F | Tuning Fork | Coupled resonators | Complex to fabricate, limited tuning range |
| G | Guitar String | GMR | 1 layer, lowest fab complexity. Narrow resonance |
| H | Didgeridoo | Multi-order GTE | d=1053nm with m_R=5, m_G=6, m_B=7 best RGB alignment |

### 6.3 Actuation Feasibility — Sb₂Se₃ PCM Emerges as Winner

| Mechanism | Δn achievable | Voltage | Verdict |
|-----------|--------------|---------|---------|
| LiNbO₃ (Pockels) | 0.002 | 3.3V | Useless — needs F≥500 |
| BaTiO₃ (Kerr) | 0.02 | 3.3V | Needs F≥30, possible with DBR |
| LC nematic | 0.2 | <3V | Fringing kills it at 0.5µm (AR=0.60) |
| **Sb₂Se₃ PCM** | **1.1** | **~1V pulse** | **NO CAVITY NEEDED. Simple 300nm film gives >2π** |
| MEMS | — | 53V pull-in | Way too high for 0.5µm |

**Sb₂Se₃ is a paradigm shift**: Δn=1.1 (amorphous→crystalline), k=0.01 (low loss), non-volatile (no power to hold state), 100ns electrical switching, CMOS-compatible deposition. Eliminates the entire GTE mirror stack.

### 6.4 Phase Quantization — 3 Bits (8 Levels) is the Sweet Spot

Diffraction efficiency follows $\eta = \operatorname{sinc}^2(1/N_\text{levels})$:

| Bits | Levels | η | Quality |
|------|--------|---|---------|
| 1 | 2 | 40.5% | Binary, bad ghost orders |
| 2 | 4 | 81.1% | Usable, visible ghosts |
| **3** | **8** | **95.0%** | **Good, minimal ghosts** |
| 4 | 16 | 98.7% | Excellent |
| 5+ | 32+ | ~100% | Indistinguishable from continuous |

This matches Sb₂Se₃'s multi-level crystallization capability. 8 distinguishable levels is demonstrated in literature for GST-family PCMs.

### 6.5 Sub-pixel Sizing — Blue Light Constraint

At 0.5µm sub-pixel pitch, the maximum diffraction angle is wavelength-dependent:

* Red (632nm): ±39° ✅
* Green (532nm): ±32° ✅ (barely exceeds ±30° target)
* **Blue (450nm): ±26.7° ❌** (fails ±30° FoV)

**Options**: (a) accept ±27° for blue, (b) shrink pitch to 0.45µm (376×376 sub-pixels), (c) reduce FoV target to ±25°.

Angular resolution is fixed by hogel size: $\Delta\theta = \lambda / D_\text{hogel} = 0.18°$ regardless of pitch.

### 6.6 Hogel Display Simulation — Confirmed 3D Parallax

Light-field simulation (200×200 hogels) confirms:

* **Parallax works**: near objects shift more than far objects when observer moves
* **Stereo works**: left/right eye views produce correct binocular disparity
* **Sub-pixel count (not phase bits) dominates image quality** — going from 32 to 128 angular bins has far more visual impact than 2-bit→3-bit phase

### 6.7 Data Bandwidth — Local Compute Wins

| Strategy | Data/frame | Compression vs raw |
|----------|-----------|-------------------|
| Raw phase patterns | 267 GB | 1× |
| Angular subsampling (100mm eye box @ 0.5m) | 8.5 GB | 31× |
| Subsampling + predictive coding | 0.5 GB | 500× |
| **Scene description (broadcast, compute locally)** | **0.03 GB** | **8,000×** |

The winning architecture: **broadcast 3D scene data (~32 MB) to all hogel tiles, each computes its own hologram locally**. Per-hogel compute = 40 MFLOPS — trivial for an embedded processor. Total compute = 80 PFLOPS (embarrassingly parallel).

### 6.8 Revised Display Specification

```
Hogel pitch:          169.3 µm (150 PPI)
Sub-pixel pitch:      0.5 µm (or 0.45 µm for blue FoV)
Sub-pixels per hogel: 338×338 = 114,244 (or 376² = 141,376)
Phase levels:         8 (3-bit)
Phase range:          >2π (Sb₂Se₃ PCM, no cavity)
Film stack:           ~300nm Sb₂Se₃ on CMOS backplane
FoV:                  ±32° green, ±39° red, ±27° blue
Angular resolution:   0.18°
Refresh:              1 FPS (PCM switching ≪ 1s)
Data architecture:    Broadcast scene + local hologram compute
Compute per hogel:    ~40 MFLOPS (tiny embedded DSP)
```

## 7. Open Questions for Tier 2

1. **Sb₂Se₃ multi-level reliability**: Can 8 crystallization levels be maintained over 10⁶+ cycles?
2. **Thermal crosstalk**: At 0.5µm pitch, does heating one PCM cell affect neighbors?
3. **Sub-pixel fill factor**: What's the gap between Sb₂Se₃ cells, and how much zero-order leakage?
4. **RGB integration**: Single broadband film or per-color stacked layers?
5. **CMOS backplane**: Can standard 28nm CMOS address 338² sub-pixels at 1 FPS with 3-bit data?

## 8. Tier 2 — RCWA Validation Results

Full-wave RCWA (Rigorous Coupled-Wave Analysis) validated the Tier 1 analytical predictions using `grcwa`. All simulations model an 8-level blazed grating (one period = 8 sub-pixels × 0.5µm = 4µm) in reflection geometry.

### 8.1 Critical Discovery: Non-Linear Phase Response

**Naive linear model fails.** Assuming φ = 4πnd/λ (no reflections) gives only η₁ = 44.5% — half the theoretical 95%. The cause: the air–Sb₂Se₃ interface has R ≈ 25–34% (due to n = 3.0–4.1), creating an implicit Fabry-Perot cavity with the mirror underneath. The phase-vs-index response is highly non-linear.

**Fix: TMM-designed lookup table.** Computing the actual phase response via transfer-matrix method and choosing n values for uniform phase steps recovers η₁ = 84–91%. This TMM-corrected LUT is mandatory for any practical device.

### 8.2 DBR Mirror Eliminates Absorption

Replacing the Al mirror (Johnson & Christy data: n=0.92, k=6.28 at 532nm) with a TiO₂/SiO₂ Bragg reflector:

| Mirror | η₁ (relative) | η₀ | R_total | η₁ (absolute) |
|--------|---------------|-----|---------|----------------|
| Al (bulk) | 90.9% | 1.8% | 0.508 | 0.462 |
| DBR 3 pairs | 54.7% | 27.0% | 0.220 | 0.120 |
| DBR 4 pairs | 83.8% | 3.8% | 0.415 | 0.348 |
| **DBR 6 pairs** | **91.3%** | **0.2%** | **0.696** | **0.635** |

DBR 6 pairs gives 37% higher absolute efficiency than Al (0.635 vs 0.462), lossless, and eliminates resonant absorption dips. DBR 2 pairs insufficient (only 0.06π phase range).

### 8.3 AR Coating Reduces Amplitude Modulation

The phase-dependent reflectance variation corrupts the pure phase grating. A single-layer MgF₂ (n=1.38) AR coating on top:

| MgF₂ thickness | η₁ (relative) | R_total | R variation ratio |
|----------------|---------------|---------|-------------------|
| None | 83.8% | 0.416 | 3.49 |
| 48nm | 88.7% | 0.482 | 2.40 |
| **72nm** | **89.8%** | **0.533** | **1.87** |
| 96nm | 88.1% | 0.556 | 1.69 |

72nm MgF₂ is the sweet spot: maximizes relative efficiency while substantially reducing amplitude modulation.

### 8.4 RGB Wavelength Validation

Each color uses its own quarter-wave DBR (4 pairs TiO₂/SiO₂), TMM-corrected 8-level LUT:

| Film d | Color | η₁ (relative) | η₁ (absolute) | R_total | Phase range |
|--------|-------|---------------|----------------|---------|-------------|
| 200nm | Blue | 77.8% | 0.323 | 0.416 | 2.04π |
| 200nm | Green | 86.4% | 0.399 | 0.462 | 1.83π |
| 200nm | Red | 67.0% | 0.268 | 0.400 | 1.66π |
| 250nm | Blue | 88.6% | 0.447 | 0.504 | 3.11π |
| 250nm | Green | 84.7% | 0.354 | 0.418 | 2.06π |
| 250nm | Red | 82.3% | 0.397 | 0.482 | 1.72π |
| **300nm** | **Blue** | **86.7%** | **0.381** | **0.439** | **2.70π** |
| **300nm** | **Green** | **85.7%** | **0.437** | **0.510** | **3.23π** |
| **300nm** | **Red** | **85.2%** | **0.355** | **0.416** | **2.05π** |

**d=300nm is the design target** — the only thickness where all three RGB wavelengths achieve ≥2π phase range with >85% relative efficiency. The 300nm Sb₂Se₃ film confirms the Tier 1 analytical estimate.

### 8.5 Revised Stack Design

```
         Air (incidence)
    ┌─────────────────────┐
    │  MgF₂ AR: 72nm      │  n = 1.38
    ├─────────────────────┤
    │  Sb₂Se₃ PCM: 300nm  │  n = 3.0–4.1 (8 levels)
    ├─────────────────────┤
    │  DBR: 4–6 pairs      │  TiO₂ (57.8nm) / SiO₂ (91.7nm)
    │  (per-color λ/4)     │  Total: 600–900nm
    ├─────────────────────┤
    │  CMOS backplane       │  Si substrate
    └─────────────────────┘
    Total film stack: ~1.0–1.3 µm
```

### 8.6 Efficiency Budget

For green (532nm) with 6-pair DBR + 72nm MgF₂ AR + 300nm Sb₂Se₃:

* Mirror reflectance: ~96% (DBR)
* First-order diffraction efficiency: ~86% (of reflected light)
* AR coating gain: reduces amplitude modulation penalty
* **Net first-order absolute efficiency: ~44%**
* Remaining 56%: distributed among zero-order, higher orders, and Fresnel losses

### 8.7 Tier 2 Conclusions

1. **TMM-corrected phase LUT is mandatory** — naive linear model loses half the light
2. **DBR mirror essential** — Al absorption costs 37% absolute efficiency
3. **300nm Sb₂Se₃ film works for all RGB** — phase range ≥2π everywhere
4. **AR coating worthwhile** — halves reflectance variation, improves uniformity
5. **Total stack ~1µm** — compatible with CMOS backplane integration
6. **Absolute efficiency ~35–44%** — acceptable for display (comparable to LCD backlight losses)

### 8.8 2D Hologram Validation — 'H' Letter

Reconstructed an 'H' letter using a 16×16 sub-pixel super-cell (8µm × 8µm), Gerchberg-Saxton phase retrieval, 8-level TMM-corrected quantization, and full RCWA:

| Stage | η into 'H' |
|-------|------------|
| GS continuous (ideal phase-only) | 87.9% |
| 8-level quantized (analytical) | 83.6% |
| **RCWA full EM (4-pair DBR)** | **68.8% relative, 36.0% absolute** |

The 15% gap between analytical and RCWA comes from amplitude modulation (reflectance varies 0.20–0.90 across phase levels). Zero-order leakage: 7.2%. The 'H' shape is clearly recognizable in the RCWA far-field. This validates that the Sb₂Se₃+DBR stack can encode arbitrary 2D holographic patterns, not just simple blazed gratings.

### 8.9 E-Beam Architecture — Alternative to CMOS Backplane

Can a CRT-style electron beam directly write PCM states, eliminating the CMOS backplane?

**Approach**: Field Emission Array (FEA) with one Spindt-type tip per hogel. Each tip scans its 338×338 sub-pixel area via electrostatic deflection, writing 8-level crystallization states into Sb₂Se₃.

| Parameter | Value | Feasible? |
|-----------|-------|-----------|
| Beam spot size | ≤0.5µm (need Schottky FEG or better) | ✅ (SEM-proven) |
| Dwell time per sub-pixel | 8.8 µs (88× margin over 100ns PCM) | ✅ |
| Thermal diffusion at 100ns | 18nm (0.04× pitch) | ✅ (no crosstalk) |
| Deflection voltage | 160V (500eV deflect, 10keV post-accel) | ⚠️ High but possible |
| FEA tip density | 3,500/cm² (need 0.35M/cm² demonstrated) | ✅ (Canon FLAT: 1.6M) |
| Vacuum packaging | ~10⁻⁴ Pa (FED-grade) | ⚠️ Cost/reliability |
| 8-level crystallization by e-beam | Undemonstrated | ❓ Key risk |

**Verdict**: Worth investigating if CMOS backplane proves infeasible at 0.5µm pitch. Eliminates 114,244 transistors per hogel and achieves 100% fill factor. Trades electronic complexity for vacuum + beam control complexity.

### 8.10 CMOS Backplane Feasibility

Can standard 28nm CMOS address 338² sub-pixels at 0.5µm pitch?

**Key insight**: Sb₂Se₃ is non-volatile → only need 2T1R per sub-pixel (select + heater driver + resistive heater), identical to Intel Optane PCM memory.

| Parameter | Value | Feasible? |
|-----------|-------|-----------|
| Transistors per sub-pixel | 2 (at 28nm: 0.06µm² of 0.25µm² available) | ✅ |
| Addressing | Row-at-a-time (DRAM-style), 2.96ms/row | ✅ |
| Wire routing | 2 wires (row+col), 5 tracks available at 28nm | ✅ |
| PCM write timing | 100ns pulse, 29,586× margin per row | ✅ |
| Crystallization control | Multi-pulse (Optane-proven) | ✅ |
| Display power (with sharing) | ~100W (dominated by CMOS leakage) | ⚠️ |
| PCM-on-CMOS integration | Sb₂Se₃ at 200°C, BEOL-compatible | ✅ |

**Comparison**: Our 0.5µm pitch is 25× larger than DRAM/Optane cells, 12× denser than LCoS SLMs, 2× denser than smallest image sensors.

**Process flow**: Standard 28nm CMOS → DBR deposition → TiN heater patterning → Sb₂Se₃ film → MgF₂ AR coating. All steps are BEOL-compatible.

**Verdict**: ✅ FEASIBLE. The 2T1R architecture is identical to production PCM memory. 28nm CMOS provides 4× margin on transistor area. Key risk is optical film stack integration (DBR + PCM + AR) on top of CMOS, not the electronics.

### 8.11 Micro-Emitter & Dual E-Beam Alternatives

**Question**: Can we create self-emitting holographic sub-pixels instead of relying on external laser illumination? Dual e-beam (one for emission, one for phase)? Micro laser emitters at 0.5µm pitch?

**Critical constraint**: Holography requires spatial coherence across the entire 169µm hogel. Independent incoherent emitters = flat display, not hologram.

**Architectures evaluated** (script: `tier2_rcwa/micro_emitter_feasibility.py`):

| Architecture | Pitch | FoV | Coherent? | TRL | Verdict |
|---|---|---|---|---|---|
| Current (PCM + ext. laser) | 0.5µm | ±30° | Yes (shared source) | 4 | ✅ Baseline |
| Dual e-beam (CL + PCM) | 0.5µm | ±30° | No (CL incoherent) | 2 | ❌ |
| E-beam pumped microlaser | 0.5µm | ±30° | Possible | 1 | ⚠ Unfabricable |
| VCSEL array + PCM | 2.0µm | ±7.6° | Yes (injection lock) | 3 | ⚠ FoV limited |
| OPA (SOI + PCM) | 0.85µm | ±18° | Yes (single source) | 3 | ⭐ Best alternative |
| OPA (LiNbO₃) | 1.1µm | ±14° | Yes (electro-optic) | 3 | ⭐ Best near-eye |
| PhC nanolaser array | 0.5µm | ±30° | Yes (injection lock) | 1 | 🔬 Long-term |

**Key findings**:

* Raw cathodoluminescence is spontaneous emission → incoherent → cannot form holographic wavefronts
* E-beam pumped microlasers are real physics (InGaN MQW in DBR cavity) but need 338² individual cavities per hogel
* Thermal crosstalk: 426nm diffusion in 1µs exceeds 300nm layer separation → pump beam corrupts PCM
* OPA + PCM hybrid is the most promising alternative: waveguide-distributed single source, PCM phase shifters (L_PCM = 0.48µm for 2π shift, zero standby power)
* VCSEL array at 2µm pitch gives only ±7.6° FoV — suitable for near-eye AR but not wide-angle

**Bottom line**: Self-emitting phase-locked pixels are the holy grail. The OPA+PCM hybrid on SOI (0.85µm pitch, ±18° FoV) is the bridge technology (5-10 year horizon). For ±30° FoV, external laser + CMOS + PCM remains the only path.

### 8.12 OPA + PCM Deep Dive (Si₃N₄ Photonics)

**Critical correction**: Silicon is opaque at visible wavelengths. OPA for holographic display must use Si₃N₄, TiO₂, or LNOI platforms. (Script: `tier2_rcwa/opa_pcm_architecture.py`)

**Si₃N₄ platform** (best overall):

* Waveguide: 400×200nm, n_eff = 1.750 at 532nm, Γ = 0.47
* Min pitch: 0.90µm → ±17° FoV, 188×188 = 35,344 emitters/hogel
* Propagation loss: 0.1 dB/cm (state of art)

**Splitter tree**: 16-level binary tree (65,535 splitters)

* Fundamental splitting loss: 48dB (= 10·log₁₀(35,344))
* Total with excess + propagation: 51.2dB

**PCM phase tuner** (Sb₂Se₃ cladding on waveguide):

* Best config: 100nm × 400nm PCM, L_2π = 0.8µm
* Insertion loss: 0.99dB — this is a FUNDAMENTAL material constant: loss = 4.343 × 4π × k_avg / Δn
* Switching energy: 13.3fJ per tuner, zero standby

**Crosstalk**: At 500nm gap, coupling length = 276µm. Need ≥636nm gap for -20dB over 85µm parallel runs.

**Power budget** (show-stopper):

| Component | Loss (dB) |
|---|---|
| Fiber → chip | 3.0 |
| Splitter tree | 51.2 |
| PCM tuner | 0.5 |
| Grating coupler | 1.9 |
| **Total** | **56.6** |

Required: 23mW per hogel → **905W for 200×200 display** (impractical).

**Root cause**: In the baseline, a plane wave illuminates all sub-pixels — zero splitting loss. The OPA pays the full 1/N penalty through guided-wave splitting. This is the fundamental disadvantage of integrated photonics vs free-space for this application.

**Verdict**: OPA + PCM viable only for small arrays (≤32×32, ~30dB loss, ~0.5mW/hogel) or near-eye AR with modest hogel counts. For full holographic display, baseline (CMOS + free-space laser) remains strictly superior.

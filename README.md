# HoloPixel — 150 PPI True Holographic Pixel R&D

Two work tracks in one repository:

- **`tier*/`** — Python physics research. Analytical, TMM and RCWA simulation of a
  Sb₂Se₃ phase-change-material hogel, plus light-field and hologram generation.
- **`holosim/`** — HoloSim, the interactive Rust/CUDA/Electron hogel light-field
  simulator. It has its own README with its own build steps.

## Getting Started (Python research)

Requires Python 3.12 and, for the CUDA paths, an NVIDIA GPU with compute
capability ≥ 8.0.

```bash
uv venv --python 3.12 .venv
UV_HTTP_TIMEOUT=300 VIRTUAL_ENV=.venv uv pip install \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.txt
source .venv/bin/activate
```

Plain pip works too: `pip install -r requirements.txt`, adding the PyTorch CUDA
index if the default wheel does not match your GPU.

## Run Tests

```bash
source .venv/bin/activate
MPLBACKEND=Agg pytest
```

The suite checks the physics claims recorded in `DRAFT.md` and
`useful_knowledge.md` against the code that produced them. Every hand derivation
is re-derived with sympy inside the test rather than hard-coded.

## Run Simulations

```bash
# === Tier 1: Light Field Generation ===
python tier1_lightfield/cornell_lightfield.py  # Numba path tracer → 9×9 LF (512×512)

# === Tier 2: SLFH Hologram Optimization ===
python tier2_slfh/slfh.py                      # Stochastic Light Field Holography (PyTorch+CUDA)

# === Tier 1: GTE / TMM Analysis ===
python tier1_tmm/gte_definitive.py       # Definitive GTE feasibility (corrected)
python tier1_tmm/gte_tmm.py              # TMM with realistic Ag mirrors
python tier1_tmm/gte_corrected.py        # Corrected design + DBR mirrors

# === Architecture & Materials ===
python tier1_tmm/acoustic_analogies.py   # Architectures D-H survey
python tier1_tmm/arch_bc.py              # Si Photonics + SAW analysis
python tier1_tmm/actuation.py            # Actuation mechanisms (Sb₂Se₃ winner)

# === Hologram Quality ===
python tier1_tmm/partial_phase.py        # Partial phase holography viability
python tier1_tmm/hogel_projection.py     # Phase quantization + single hogel far-field

# === Display Simulation ===
python tier1_tmm/hogel_display_v3.py     # Full light-field display with parallax
python tier1_tmm/compression.py          # Data bandwidth & compression strategies
python tier3_display/holo_cornell.py     # Holographic display simulation (Numba)

# === Tier 2: RCWA Validation ===
python tier2_rcwa/rcwa_subpixel.py       # Baseline blazed grating + 2D hologram
python tier2_rcwa/rcwa_diagnosis.py      # TMM phase correction study
python tier2_rcwa/rcwa_optimization.py   # DBR mirror + AR coating optimization
python tier2_rcwa/rcwa_rgb_fast.py       # RGB validation (fast, table only)
python tier2_rcwa/opa_pcm_architecture.py  # OPA + PCM power budget (verdict: dead)
```

All plots are written to `plots/`.

## Build / Deploy (HoloSim)

See [holosim/README.md](holosim/README.md).

## Project Structure

```text
tier1_lightfield/  — Numba path tracer for Cornell box light field generation
tier1_tmm/         — Tier 1: Analytical & TMM thin-film simulations
tier2_slfh/        — SLFH hologram optimizer (Schiffers et al.)
tier2_rcwa/        — Tier 2: RCWA periodic array simulations
tier3_display/     — Holographic display simulation
tier3_fdtd/        — Tier 3: FDTD full 3D electromagnetic sims (not started)
holosim/           — Interactive Rust/CUDA/Electron hogel simulator
tests/             — Physics regression tests for the tier scripts
plots/             — Generated simulation outputs
docs/              — Findings, technical notes and literature surveys
```

## Key Findings (Light Field + SLFH Pipeline)

- **Path tracer**: Numba CPU, 9×9 angular views × 512×512 px, 256 spp, 2 bounces → 70s
- **SLFH optimizer**: PyTorch CUDA (RTX 5090), 1000 iters × 3 channels → 2 min
- **Physics**: Angular Spectrum propagation (Eq. 16-19), stochastic pupil sampling (Eq. 15)
- **Center view** reproduces Cornell box (red/green walls, boxes, area light)
- **Speckle** in off-axis views matches expected coherent display behavior for 512×512 SBP

## Key Findings (Tier 1)

- **Sb₂Se₃ PCM** eliminates the GTE cavity entirely (Δn=1.1, no mirrors needed)
- **3-bit phase (8 levels)** is sufficient: 95% diffraction efficiency
- **0.5µm sub-pixel pitch** achieves ±32° FoV for green, but blue limited to ±27°
- **Local hologram computation** reduces data from 267 GB/frame to 32 MB scene broadcast

## Key Findings (Tier 2 — RCWA)

- **TMM-corrected phase LUT is mandatory**: naive linear model gives only 44% efficiency; TMM-corrected recovers 84–91%
- **DBR 6 pairs** (TiO₂/SiO₂) gives 63.5% absolute efficiency vs 46.2% for Al mirror
- **300nm Sb₂Se₃** is the design target: only thickness where all RGB get ≥2π phase range
- **72nm MgF₂ AR coating** halves reflectance variation, improves uniformity
- **Absolute efficiency 35–44%** per color — acceptable for display application

See [DRAFT.md](DRAFT.md) §6–8 for full findings.

## Literature Surveys

- [docs/research_foveated_holography.md](docs/research_foveated_holography.md) — foveated holographic rendering for head-mounted displays
- [docs/research_holo_sota.md](docs/research_holo_sota.md) — CGH, speckle, quantization and PCM SLM state of the art

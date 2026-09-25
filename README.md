# HoloPixel — 150 PPI True Holographic Pixel R&D

> **Work in progress. Public so it can be read, not so it can be used.**
>
> This is an active research notebook, not a library and not a product. Results
> change week to week, several conclusions here have already reversed at least
> once, and `useful_knowledge.md` exists precisely to record the assumptions
> that turned out to be wrong. Anything in here may be wrong too.
>
> Interfaces are not stable, nothing is versioned, and no part of it is
> supported. Please do not build on it.
>
> No licence is granted. All rights reserved.

Two work tracks in one repository:

- **`tier*/`** — Python physics research. Analytical, TMM and RCWA simulation of a
  Sb₂Se₃ phase-change-material hogel, plus light-field, Gaussian-splat and
  hologram generation, and the hogel-free optimiser.
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

Tests marked `blender` render through headless Blender 5.2 (Cycles, OptiX GPU)
and need `blender` on the `PATH`; they take a few minutes. Skip them with
`pytest -m "not blender"`.

## Physical light-field screen model

`tier1_lightfield/foveated_optics_study/scripts/` holds a Cycles model of the
light-field screen: a real hex microlens array over the panel, optional
remapper optics, and pinhole or 4 mm-aperture eye cameras at the schematic
eye's pupil. `lf_evaluate.py <design_dir>` is the shared acceptance evaluator
for remapper designs (contract in its docstring); its blur is the per-pixel
beam width at the eye.

Remapper design searches (GPU, float64), under `remapper_designs/`. The target field of view is
`HOLOPIXEL_FIELD_DEG` (`<width>x<height>` degrees, default `70x45`); results and designs record it, and a
design is refused under another field (the older families are fixed at 70 x 45):

```bash
cd tier1_lightfield/foveated_optics_study/remapper_designs
# coaxial lenses: search, export a candidate, score it in Cycles
python coaxial_dls/gpu_search.py <out> --elements 5
python coaxial_dls/export_candidate.py <out>/best_el5_flat.json <design_dir>
# folded: panel above the eye, freeform mirror, 1-2 freeform correctors
python freeform_mirror/fold_search.py <out> --elements 1 --material resin   # or glass; --ratio-weight 0 frees the mapping
# seeded from earlier designs, and with a B-spline of K cells on the (half-)mirror (fold and pancake alike)
python freeform_mirror/fold_search.py <out> --elements 1 --material glass --seed-from freeform_mirror/results_fold/best_fold_el1_glass.json --spline-cells 4
# a seed that already carries a spline keeps its own grid: give no --spline-cells
python freeform_mirror/pancake_search.py <out> --elements 1 --material glass --seed-from freeform_mirror/results_pancake/best_pancake_el1_glass_spline4.json
python freeform_mirror/pancake_search.py <out> --elements 1 --material resin   # pancake: polarisation fold, round lens
python freeform_mirror/export_fold.py <out>/best_fold_el1_resin.json <design_dir>
python freeform_mirror/export_pancake.py <out>/best_pancake_el1_resin.json <design_dir>
blender -b --factory-startup --python freeform_mirror/view_fold_blender.py -- <design_dir> view.blend
python ../scripts/lf_evaluate.py <design_dir> --pixels 2560
# simulated headset from the pupil: raw chart on the panel, or pre-warped content
# (the latter needs the lf_evaluate run above, it reuses <design_dir>/evaluation)
python ../scripts/hmd_view.py <design_dir> <out_dir> --work <scratch> [--aperture-mm 4]
python ../scripts/hmd_encoded.py <design_dir> <out_dir> --work <scratch>   # --fovea: true-size +/-5 deg crop, bar chart
# one display scene per design, in colour: dispersive glass (N-LASF46B lens, silica lenslets,
# narrow-band R G B), panel content through per-channel ST-maps; then all scenes in one file
python ../scripts/design_scenes.py scene <name> pancake freeform_mirror/results_pancake/best_pancake_el1_glass_spline4.json <out>
python ../scripts/design_scenes.py targets <out>             # once: the depth scene (sphere 0.35 m, cube 0.7 m, backdrop 6 m) per pupil view
python ../scripts/design_scenes.py blend <name> <out>        # save the scene again from its calibration
python ../scripts/design_scenes.py scene-panel <name> <out>  # encode the depth scene for an already calibrated design
blender -b --factory-startup --python ../scripts/combine_scenes.py -- <out>/designs.blend <name>=<out>/<name>/<name>.blend ...
# in the file: EYE (fisheye ~160 deg) and FOVEA (12 deg) cameras with a 4 mm pupil; the panel's CONTENT_SWITCH: 0 charts, 1 depth scene
# foveal inset: the whole panel over +/- 10 deg through a moulded singlet or achromat (CPU)
python foveal_inset/inset_study.py <out>
```

Fold designs use the variable-focal lenslet array (`"lenslets": "variable_retina"` in
design.json, `scripts/variable_lenslets.py`): each lens's focal length is set so its number
of views follows the retina and diffraction (about one at the fovea, up to 5 x 5 outside),
so the lens vertices form a 2D surface that is the fold search's image surface. The search
also keeps the map one-to-one on a dense chief-ray grid and sends directions beyond the
field of view off the panel: a folded map shows in Cycles as ghosts. Coaxial
designs are round and keep the round map (`foveation_target_radial.py`) with a uniform array.

## Run Simulations

```bash
# === Tier 1: Scene Generation ===
python tier1_lightfield/cornell_lightfield.py  # Numba path tracer → 17×17 LF (512², 256 spp,
                                               #   8 bounces, glass sphere, Russian roulette)
# tier1_gaussians/render.py is a differentiable EWA splat rasteriser (library, not a script)

# === Tier 2: Hologram Optimization ===
python tier2_slfh/slfh.py                      # Stochastic Light Field Holography (PyTorch+CUDA)
# tier2_hfh/ is the hogel-free optimiser: a light field in, one panel phase out, NO depth
#   optimise.py   stochastic-pupil Adam, quantisation-aware, multi-subframe, foveation-aware
#   acuity.py     retinal geometry, Watson acuity, foveation pyramid
#   propagate.py  angular spectrum;  display.py  pupil + defocus;  modes.py  coherent modes

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
tier1_gaussians/   — differentiable EWA Gaussian splat rasteriser
tier2_hfh/         — hogel-free optimiser: light field in, panel phase out, no depth
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

## Key Findings (Light Field + Hologram Pipeline)

- **Path tracer**: Numba CPU, 17×17 views × 512², 256 spp, 8 bounces, dielectric sphere
- **Hogel-free, no depth**: a light field alone drives the solve, so glass and refraction
  survive — a depth map cannot represent them
- **Scale reached**: panel 8192², window 2048 (12.48 mm panel, 3121 µm pupil) on an
  RTX 5090, float32, 3000 iterations in 196 s
- **Subframes**: 16 modes reach 47.5 dB. A pupil-position sweep measures the speckle
  exponent at −0.46 to −0.99 (against −0.5 for independent averaging), so the worst
  position needs M = 5.3, i.e. 477 Hz against 2690 Hz available
- **Quantisation**: the real 8-level Sb₂Se₃ device costs 2.63 dB if applied after the
  fact, 0.89 dB if the optimiser is aware of it
- **A light field is spatially incoherent** (231–256 modes of 289), so a coherent-mode
  decomposition cannot supply the missing phase — optimisation must

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

- [docs/holographic_rendering_equation.md](docs/holographic_rendering_equation.md) — the derived rendering equation, Wigner form, and where foveation enters
- [docs/research_foveated_holography.md](docs/research_foveated_holography.md) — foveated holographic rendering for head-mounted displays
- [docs/research_holo_sota.md](docs/research_holo_sota.md) — CGH, speckle, quantization and PCM SLM state of the art
- [docs/notes_peripheral_colour_and_flicker.md](docs/notes_peripheral_colour_and_flicker.md) — what peripheral vision does and does not give us
- [docs/notes_gaussian_splatting_holography.md](docs/notes_gaussian_splatting_holography.md) — Gaussian splats as a light-field source
- [useful_knowledge.md](useful_knowledge.md) — assumptions that turned out wrong, and why

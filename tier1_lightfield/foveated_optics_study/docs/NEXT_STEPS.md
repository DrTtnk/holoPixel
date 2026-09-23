# Next steps for local execution and fine tuning

The next pass should focus on **optics first**. Display-driver electronics and product rendering are secondary until the optical relay survives a stricter simulation.

## P0 - Reproduce and profile the ROMs

1. Run the retinal model and verify the Watson meridian curves.
2. Reproduce the 70 x 45 deg field-sample integration.
3. Reproduce the diffraction-aware local view-count thresholds.
4. Profile each stage before increasing spatial resolution.
5. Keep deterministic seeds/configuration files for every stored render.

## P1 - Replace heuristic crosstalk/blur with explicit propagation

Build a minimal physical chain:

`display -> microlens/pupil encoder -> relay -> remapper -> steering mirror -> 4 mm pupil`.

Propagate RGB fields at least at 450/550/650 nm. Report:

- pupil-plane complex field or irradiance;
- PSF at representative field points;
- MTF versus spatial frequency;
- view-to-view crosstalk matrix;
- throughput / clipping;
- chromatic chief-ray shift;
- field-dependent aberration.

Start with 1-D or separable 2-D if needed, but keep the coordinate transforms explicit.

## P2 - Optimize pupil allocation continuously

Replace the coarse `1 -> 7 -> 19 -> 29` view rule with an optimizer that chooses local pupil support and view count from:

- retinal target bandwidth;
- diffraction/MTF requirement;
- crosstalk cap;
- minimum useful depth/parallax support;
- light-efficiency constraint.

The design variable should be a local phase-space allocation, not merely an integer K.

## P3 - Compare the two hex architectures

### Global remapper

Optimize a regular hex MLA plus one or two freeform surfaces. A reflective remapper is especially interesting because it avoids first-order chromatic dispersion in the steering/remapping element.

### Tiled freeform array

Fit locally varying tile normal and power. Add manufacturability penalties:

- slope limit;
- sag limit;
- minimum radius of curvature;
- tile-to-tile discontinuity;
- dead-zone / fill-factor penalty.

Compare both architectures under the **same** pupil-plane and retinal MTF metrics.

## P4 - Blender/mechanical visualization

Use `scripts/blender_hex_project.py` to create two collections in one `.blend`:

- `GLOBAL_REMAPPER_VARIANT`;
- `TILED_FREEFORM_VARIANT`.

Add a simplified display, steering mirror, pupil plane, sparse chief rays and diagnostic material colors. Keep Blender as a geometry/packaging/communication tool; do not treat Cycles as the diffraction solver.

## P5 - Calibration and eye motion, later

Only after the fixed-eye model is stable:

- pupil swim with eye rotation;
- torsion;
- gaze-dependent mapping;
- finite tracking latency;
- steering bandwidth;
- calibration errors and compensation prewarp.

## Suggested local outputs

For every candidate prescription, store:

- `config.json`;
- field/pupil sampling maps;
- PSF/MTF plots for center, 2 deg, 5 deg, 10 deg, 20 deg and edge;
- RGB pupil irradiance maps;
- crosstalk matrix;
- final pinhole render;
- signed luminance difference versus ideal;
- 10x absolute difference diagnostic;
- runtime and memory profile.

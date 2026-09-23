# Local execution handoff

This folder is the workstation handoff for the next optics pass. The goal is to replace the exploratory reduced-order optics with an explicit end-to-end prescription while preserving the same retinal and pupil-plane coordinate model.

## Scope lock

- fixed eye position and orientation
- 4 mm pupil
- 20 mm eye relief
- perfect pupil tracking and steering
- 70 deg x 45 deg reference FOV
- RGB wavelengths at 450 / 550 / 650 nm
- compare both hex architectures under the same metrics
- optics first; panel driving and product electronics are deliberately deferred

## Reproduce first

Run:

    python scripts/retina_model.py
    python scripts/phase_space_budget.py
    python scripts/waveoptics_encoder.py

Record exact environment, runtime, and peak memory. The CSVs are snapshots from the exploratory session; large discrepancies should be understood before increasing resolution.

## Main task

Replace the heuristic pinhole diagnostic with explicit propagation through:

    display -> hex pupil encoder / MLA -> pupil relay -> foveated remapper -> steering mirror -> 4 mm pupil

For representative field points (center, 2, 5, 10, 20 deg, and edge), save chief-ray pupil landing, pupil irradiance/complex field, captured power, PSF, MTF, crosstalk matrix, RGB chromatic shift, field aberration, and the retinal sample pitch actually supported by the optics.

## Architecture A: identical hex MLA + global remapper

Start here first. Keep the MLA regular and identical, use it primarily as the pupil encoder, then fit one or two downstream freeform surfaces to the retina-matched angular transport. Try a reflective remapper early because it removes first-order chromatic dispersion from that stage. Penalize wavefront error, field-dependent PSF, clipping, and excessive local angular magnification, not merely sag.

## Architecture B: tiled freeform hex array

Move part or all of the remapping into per-tile normal and power. Optimize x, y, normal, local power, pupil support, and view count. Add manufacturability penalties for slope, sag, minimum radius, tile discontinuity, and dead area. Compare against A with the same optical metrics.

## Pupil sampling

Do not restore a fixed 7 x 7 pupil grid as the design target. The wave-optics work in this folder exists specifically because that abstraction failed at the fovea. Treat pupil allocation as field-dependent. The coarse 1 / 7 / 19 / 29 regions are only a starting quantization; optimize pupil support from retinal bandwidth, MTF, crosstalk, depth/parallax needs, and throughput. Prefer hex packing when discrete samples are required because the pupil is circular.

## Diagnostic output contract

For every candidate save a deterministic config plus retinal target map, pupil support/view map, RGB pupil irradiance, PSF/MTF, crosstalk, throughput/clipping, ideal pinhole render, propagated optics-aware pinhole render, signed luminance difference, 10x absolute difference, runtime, and peak memory.

Keep the aggressive test chart: natural imagery hid small differences too easily.

## Blender

scripts/blender_hex_project.py is a geometry/communication starter, not an optical solver. Replace its placeholder foveated mapping with the optimized table. A useful export schema is:

    x_mm, y_mm, theta_x_deg, theta_y_deg, normal_x, normal_y, normal_z,
    local_power_or_focal_length, pupil_support_mm, view_count

Use Blender for packaging and visualization, not diffraction validation.

## Milestone

The existing optics-aware renders are encouraging but still use reduced-order crosstalk and blur proxies. The next milestone is to make an explicit relay/freeform prescription survive the same diagnostics without those phenomenological shortcuts.
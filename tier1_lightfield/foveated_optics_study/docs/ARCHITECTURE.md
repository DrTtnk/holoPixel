# Architecture notes

## Reference light-field coordinates

Use the pupil-plane parameterization

\[
L(x_p,y_p,\theta_x,\theta_y).
\]

- `(x_p, y_p)`: where a ray crosses the pupil plane; controls viewpoint/parallax/depth support.
- `(theta_x, theta_y)`: visual-field direction; with the pinhole-eye diagnostic, this maps directly to retinal field position.

Do not mix these dimensions. The retinal acuity model constrains angular field sampling, not pupil-plane sampling.

## Baseline optical chain

The working architecture is

`display -> pupil encoder / MLA -> foveated remapper -> global steering -> fixed pupil -> eye`.

Global tracking/steering is assumed. For the present study the eye is fixed; eye rotation, pupil swim and torsion are deferred.

## Global-remapper variant

Use a regular hex MLA as a local pupil encoder. Keep the display/MLA registration regular and perform the retina-matched angular density warp in a downstream freeform element, preferably reflective if chromatic performance becomes dominant.

Advantages:

- identical or near-identical local micro-optics;
- clean separation between pupil encoding and field remapping;
- easier calibration and fabrication path;
- global remapper can be optimized independently.

Risks:

- the remapper must support a large local change in angular magnification;
- field aberrations and wavefront quality become the central difficulty;
- packaging a reflective remapper and MEMS steering stage may be nontrivial.

## Tiled-freeform variant

Each hex cell carries its own local normal and potentially its own optical power. The array itself therefore performs some or all of the field remapping.

Advantages:

- naturally distributes the mapping across the aperture;
- potentially fewer macroscopic freeform elements;
- directly exposes local phase-space allocation at each tile.

Risks:

- approximately 10^5 locally different tiles at the 2K prototype scale;
- fabrication and metrology are harder;
- stitching, dead zones and tile-to-tile discontinuities may create scatter/crosstalk;
- local tolerances may dominate image quality.

## Steering

For a 4 mm useful pupil support that must move inside an 8 mm nominal static region at 20 mm eye relief, the first-order bundle-center translation is +/-2 mm, corresponding to about +/-5.7 deg optical deflection per axis. A flat mirror would require roughly half that mechanical tilt in the ideal reflection geometry.

## Brightness interpretation

The architecture is attractive because steering can direct the useful bundle onto the pupil rather than discarding most light with a pinhole mask. Do not equate this with unity system efficiency. The correct metric is useful flux at the pupil divided by emitted optical flux, with all real losses included.

## Etendue / phase-space constraint

The key design rule is qualitative but unavoidable:

- finer retinal/angular spatial resolution requires a larger effective pupil support per view;
- more independent pupil views require smaller effective sub-apertures and therefore lower spatial bandwidth per view.

This is why a constant view count is the wrong abstraction. The architecture should make both retinal pitch and pupil-view density field-dependent.

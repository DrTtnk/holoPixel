# References

## Retinal sampling

A. B. Watson, "A formula for human retinal ganglion cell receptive field density as a function of visual field location," Journal of Vision 14(7):15, 2014. DOI: 10.1167/14.7.15.

Used as the fast retinal-sampling ROM in this study.

## Foveated light fields

Q. Sun et al., "Perceptually-Guided Foveation for Light Field Displays," SIGGRAPH Asia, 2017.

Relevant precedent for applying retinal/perceptual bandwidth to light-field ray allocation. This study pushes the foveation requirement into the physical optical sampling/remapping architecture rather than treating it only as a rendering-budget problem.

## Pinhole near-eye baseline

K. Aksit, J. Kautz, D. Luebke, "Slim near-eye display using pinhole aperture arrays," Applied Optics 54(11), 3422-3427, 2015. DOI: 10.1364/AO.54.003422.

Useful as a baseline for aperture-spacing geometry and the diffraction/geometric-blur trade-off. The paper PDF is intentionally not copied into this public repository; obtain it from the publisher or another authorized source.

The paper derives the paraxial optimum pinhole scale approximately as

    d_a ~= sqrt(2 lambda d_ai)

from the intersection of diffraction and geometric blur. Our intended architecture differs because it redirects light with refractive/reflective optics instead of deliberately rejecting most light at a pinhole mask.

## General optics topics to cross-check

- conservation of etendue / optical invariant
- freeform beam shaping via prescribed irradiance / optimal transport
- scalar Fresnel and angular-spectrum propagation
- circular-aperture incoherent MTF
- microlens-array light-field displays and pupil-replicating near-eye displays
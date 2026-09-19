"""
Where each sample of a rendered view lands on the retina, and how much the eye
can actually resolve there.

`optimise.render_views` returns an n x n intensity image that is the squared
transform of one windowed panel region. That image is the eye's retinal image,
and its samples are uniform in DIRECTION COSINE, not in angle: bin k of an
n-point transform at pitch p carries spatial frequency (k - n/2)/(n p), and a
spatial frequency f leaves the panel at the direction whose cosine is lam f.
The full span is therefore lam/p in cosine, and the angle it subtends is
arcsin of that, not the cosine itself. At the optimiser's 0.349 the two differ
by 0.45% -- small, but the small-angle form is simply the wrong function and
there is no reason to carry it.

Coordinates are those of the VISUAL FIELD, matching the light field the target
views were rendered from: column index increases to the right, row index
increases downwards, so row 0 is the superior field. The eye's optics invert
the retinal image, but that inversion applies to the rendered and the target
view alike and so cancels everywhere in this codebase.

The meridian matters because the retina is not radially symmetric. Ganglion
cell density falls fastest in the superior field and slowest in the nasal one,
and the optic disc -- which resolves nothing at all -- sits in the temporal
field of each eye. Both need to know which eye this is, hence `eye`.
"""

import numpy as np
import torch


def direction_cosines(n, span, device=None, dtype=torch.float64):
    """
    The (horizontal, vertical) direction cosines of every sample of an n x n
    view whose full span is `span` = wavelength / pitch.

    The grid matches `torch.fft.fftshift(torch.fft.fft2(...))`, which puts the
    zero frequency at index n // 2, so that sample is exactly on axis.
    """
    if span > 2:
        raise ValueError(f"span {span} exceeds 2: beyond that the orders are evanescent")
    k = (torch.arange(n, device=device, dtype=dtype) - n // 2) * (span / n)
    alpha = k[None, :].expand(n, n).contiguous()      # horizontal, along columns
    beta = -k[:, None].expand(n, n).contiguous()      # vertical, row 0 is superior
    return alpha, beta


def eccentricity_map(n, span, eye="right", device=None, dtype=torch.float64):
    """
    Eccentricity in degrees from fixation, and meridian in degrees, for every
    sample of an n x n view.

    Meridian is measured from the temporal field: 0 temporal, 90 superior,
    180 nasal, 270 inferior. Temporal is +x for a right eye and -x for a left
    one, which is the only thing `eye` changes here.
    """
    if eye not in ("left", "right"):
        raise ValueError(f"eye must be 'left' or 'right', not {eye!r}")
    alpha, beta = direction_cosines(n, span, device=device, dtype=dtype)
    radial = torch.sqrt(alpha ** 2 + beta ** 2).clamp(max=1.0)
    eccentricity = torch.rad2deg(torch.asin(radial))
    temporal = alpha if eye == "right" else -alpha
    meridian = torch.rad2deg(torch.atan2(beta, temporal)) % 360
    return eccentricity, meridian


_BINOMIAL = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _blur(image):
    """One separable binomial low-pass, edges replicated."""
    kernel = _BINOMIAL.to(device=image.device, dtype=image.dtype)
    flat = image.reshape(-1, 1, *image.shape[-2:])
    pad = torch.nn.functional.pad
    conv = torch.nn.functional.conv2d
    flat = conv(pad(flat, (2, 2, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    flat = conv(pad(flat, (0, 0, 2, 2), mode="replicate"), kernel.view(1, 1, -1, 1))
    return flat.reshape(*image.shape)


def gaussian_pyramid(image, levels):
    """
    `levels` progressively coarser copies of `image`, the first being the
    image itself. Each step low-passes with the separable binomial kernel and
    decimates by two -- Burt and Adelson's construction, unchanged.

    The kernel sums to one, so a flat field survives every level intact. That
    is the property the whole foveation rests on: blurring the periphery must
    not change its brightness, or the eye sees a ring where the blur changes.
    """
    if levels < 1:
        raise ValueError(f"levels must be at least 1, not {levels}")
    smallest = image.shape[-1] >> (levels - 1)
    if smallest < 4:
        raise ValueError(
            f"{levels} levels of a {image.shape[-1]}-wide image ends at {smallest} "
            "samples, narrower than the 5-tap kernel")

    lead = image.shape[:-2]
    out = [image]
    current = image
    for _ in range(levels - 1):
        decimated = _blur(current)[..., ::2, ::2]
        current = decimated.reshape(*lead, *decimated.shape[-2:])
        out.append(current)
    return out


def foveate(image, level):
    """
    Blur `image` by a different amount at every sample: `level` gives the
    pyramid level to read there, and a fractional level blends the two that
    bracket it.

    This is the honest way to spend less on the periphery. Merely weighting
    the loss down out there still asks the optimiser for detail the eye cannot
    resolve and then forgives it for failing; low-passing the target instead
    stops asking, which frees the panel's degrees of freedom for the fovea.

    Differentiable in `image`, which is what lets it sit inside the loss.
    """
    if float(level.min()) < 0:
        raise ValueError("level must not be negative: it is a pyramid depth")
    n_levels = int(np.ceil(float(level.max()))) + 1
    size = image.shape[-2:]

    out = torch.zeros_like(image)
    for i, coarse in enumerate(gaussian_pyramid(image, n_levels)):
        if coarse.shape[-2:] != size:
            coarse = torch.nn.functional.interpolate(
                coarse.reshape(-1, 1, *coarse.shape[-2:]), size=size,
                mode="bilinear", align_corners=False).reshape(*image.shape)
        weight = (1 - (level - i).abs()).clamp(min=0)
        out = out + weight * coarse
    return out


# Measured, not assumed. `scratchpad/mtf.py` drives a cosine of known frequency
# through `foveate` at each level and reads off where the modulation halves;
# `plots/foveation_pyramid_mtf.png` is the curve. Levels 1 to 4 give
# 0.140, 0.0670, 0.0330 and 0.0164 cycles per sample, which is 0.528 / 2^L to
# within 3%. The nominal brick wall would be 0.5 / 2^L, so a pyramid level is
# very nearly its own Nyquist frequency -- the binomial kernel is a good enough
# anti-alias filter that no correction factor is needed, only this constant.
PYRAMID_MTF50 = 0.528


def level_for_cutoff(cutoff):
    """
    The pyramid level whose 50% modulation point sits at `cutoff`, in cycles
    per view sample.

    Clamped at zero: the view grid cannot carry detail finer than its own
    samples, so an eye sharper than the grid gets the grid, and the shortfall
    is a property of the simulation resolution rather than of the eye.
    """
    if float(cutoff.min()) <= 0:
        raise ValueError("cutoff must be positive: a zero cutoff resolves nothing")
    return torch.log2(PYRAMID_MTF50 / cutoff).clamp(min=0)


def mar_arcmin(eccentricity, meridian):
    """
    Behavioural minimum angle of resolution, in arcminutes: the angular size
    of the finest feature the eye resolves at that point in the visual field.

        MAR(e) = 1 + e / 2.3

    One arcminute on axis is the textbook 20/20 figure, and the slope says
    acuity halves every 2.3 degrees out.

    Provenance, stated carefully because it is commonly got wrong, including
    earlier in this repository. This is the classical cortical-magnification
    form MAR(e) = 1 + e/E2 from Levi, Klein and Aitsebaomo (1985) and Rovamo
    and Virsu (1979), with E2 = 2.3 degrees, which is a LETTER-ACUITY value.
    It is frequently attributed to Watson (2014); a search of the accessible
    literature found no primary-source statement that Watson's paper contains
    it, and Watson's actual model is the anisotropic midget density formula
    implemented below, which gives a substantially finer 65.4 cycles per
    degree on axis against this form's 30. E2 for other tasks differs sharply
    -- Vernier acuity averages about 1.5 degrees across studies -- so this
    constant is task-specific and not a universal property of the eye.

    `meridian` is accepted and unused: this form is isotropic, unlike the
    midget lattice. The argument keeps the two interchangeable at call sites.
    """
    if float(eccentricity.min()) < 0:
        raise ValueError("eccentricity must not be negative")
    return 1.0 + eccentricity / 2.3


def luminance_cutoff_cpd(eccentricity, meridian):
    """
    The highest luminance grating the eye resolves there, in cycles per degree.

    A resolvable cycle is two bars, each one MAR wide, so the cutoff is
    1 / (2 * MAR) per degree, which is 30 / MAR for a MAR in arcminutes. On
    axis that gives the familiar 30 cycles per degree.

    Luminance only. Chromatic resolution is lower everywhere, including at the
    fovea (11-12 cycles per degree against 30-60), and red-green and
    blue-yellow fall off at quite different rates --
    `docs/notes_peripheral_colour_and_flicker.md` has the measured curves.
    """
    return 30.0 / mar_arcmin(eccentricity, meridian)


# ---------------------------------------------------------------------------
# Watson (2014), "A formula for human retinal ganglion cell receptive field
# density as a function of visual field location", J. Vis. 14(7):15.
#
# The correct limit outside the fovea is not cone density. At the very centre
# each cone drives its own ON and OFF midget ganglion cell, so cone and midget
# spacing coincide; further out many cones converge on one midget, and the
# midget is the last stage that can carry a spatially resolved signal off the
# retina. Whichever stage is sparser sets the Nyquist limit, and past a few
# degrees that is the midget.
#
# Table 1 below is transcribed from the paper. The journal is behind a
# Cloudflare challenge that no automated fetch here could pass, so the numbers
# come from ISET/isetcam's `human/watsonRGCSpacing.m`, which reprints Table 1
# verbatim in its header comment. That is a secondary source, but it is
# checkable: the paper also prints four derived constants, and the tests
# reproduce all of them from these parameters alone. A transcription error
# would have to be a conspiracy to survive that.
# ---------------------------------------------------------------------------

#: (a, r_2 in degrees, r_e in degrees) per meridian, Watson Table 1.
MERIDIANS = {
    "temporal": (0.9851, 1.058, 22.14),
    "superior": (0.9935, 1.035, 16.35),
    "nasal":    (0.9729, 1.084, 7.633),
    "inferior": (0.996, 0.9932, 12.13),
}

PEAK_CONE_DENSITY = 14804.6      # deg^-2, Curcio's value as Watson uses it
PEAK_RGCF_DENSITY = 2 * 1.12 * PEAK_CONE_DENSITY     # 33162.3 deg^-2
MIDGET_FRACTION_AT_FOVEA = 1 / 1.12
MIDGET_FRACTION_SCALE_DEG = 41.03


def _meridian_weights(meridian):
    """
    Watson combines the two meridians that bracket a field point, weighted by
    the squared components along each. `meridian` is degrees from temporal.

    Returns the horizontal and vertical meridian names paired with cos^2 and
    sin^2 of the polar angle. Using the trigonometric components rather than
    x/r and y/r keeps the origin finite, where x, y and r all vanish together
    and every meridian agrees anyway.
    """
    rad = torch.deg2rad(meridian)
    c, s = torch.cos(rad), torch.sin(rad)
    return c ** 2, s ** 2, c >= 0, s >= 0


def _blend(eccentricity, meridian, per_meridian):
    """Apply `per_meridian(a, r2, re)` and mix the two bracketing meridians."""
    c2, s2, east, north = _meridian_weights(meridian)
    horizontal = torch.where(east, per_meridian(*MERIDIANS["temporal"], eccentricity),
                             per_meridian(*MERIDIANS["nasal"], eccentricity))
    vertical = torch.where(north, per_meridian(*MERIDIANS["superior"], eccentricity),
                           per_meridian(*MERIDIANS["inferior"], eccentricity))
    return c2, s2, horizontal, vertical


def _rgcf_one(a, r2, re, r):
    return PEAK_RGCF_DENSITY * (a * (1 + r / r2) ** -2 + (1 - a) * torch.exp(-r / re))


def rgcf_density(eccentricity, meridian):
    """
    All retinal ganglion cell receptive fields per square degree, Watson's
    Equation 4. Densities, not spacings, so the two bracketing meridians mix
    linearly in the squared components.
    """
    if float(eccentricity.min()) < 0:
        raise ValueError("eccentricity must not be negative")
    c2, s2, horizontal, vertical = _blend(eccentricity, meridian, _rgcf_one)
    return c2 * horizontal + s2 * vertical


def midget_fraction(eccentricity):
    """Watson's Equation 7: the share of ganglion cells that are midgets."""
    return MIDGET_FRACTION_AT_FOVEA * (1 + eccentricity / MIDGET_FRACTION_SCALE_DEG) ** -1


def mrgcf_density(eccentricity, meridian):
    """Midget receptive fields per square degree: 29609.2 on axis."""
    return midget_fraction(eccentricity) * rgcf_density(eccentricity, meridian)


def mrgcf_spacing_arcmin(eccentricity, meridian):
    """
    Spacing of the ON-centre midget lattice, in arcminutes.

    Two details decide whether this comes out right. The lattice is hexagonal,
    so spacing is sqrt(2 / (sqrt(3) * density)), and only ONE polarity forms a
    sampling lattice, so the density is half of all midgets. Together they
    give 0.5299 arcmin on axis -- which is the cone spacing, as it must be,
    since at the fovea each cone has exactly one ON midget. Forgetting the
    factor of two gives 0.375 arcmin and a retina that resolves 41% too fine.

    Spacings combine as Watson's Equation 9: the two bracketing meridians are
    mixed in quadrature, weighted by the squared direction components.
    """
    if float(eccentricity.min()) < 0:
        raise ValueError("eccentricity must not be negative")

    def spacing_one(a, r2, re, r):
        on_centre = 0.5 * midget_fraction(r) * _rgcf_one(a, r2, re, r)
        return 2.0 / (np.sqrt(3.0) * on_centre)          # squared spacing

    c2, s2, horizontal, vertical = _blend(eccentricity, meridian, spacing_one)
    return torch.sqrt(c2 * horizontal + s2 * vertical) * 60.0


def mrgcf_nyquist_cpd(eccentricity, meridian):
    """
    The finest grating the midget lattice can sample, in cycles per degree:
    1 / (sqrt(3) * spacing) for a hexagonal lattice. 65.37 on axis, the value
    Watson prints.

    This is an ANATOMICAL limit and it is roughly twice the behavioural one
    (`luminance_cutoff_cpd`, 30 cycles per degree on axis). The retina samples
    finer than the eye's optics deliver, so the two are different quantities
    and a display is bounded by the smaller of them. Which to use is a design
    choice: the anatomical bound renders detail the optics will destroy, the
    behavioural bound risks discarding detail a sharp young eye could use.
    """
    spacing_deg = mrgcf_spacing_arcmin(eccentricity, meridian) / 60.0
    return 1.0 / (np.sqrt(3.0) * spacing_deg)


def foveal_weight(eccentricity, meridian, normalise=True):
    """
    A per-sample loss weight proportional to how densely the midget lattice
    samples there: the reciprocal of its spacing, in samples per degree.

    This is the route Chakravarthula et al. (arXiv:2108.06192, Equations 8 and
    9) take, and it is NOT the same as blurring the target with `foveate`.
    Weighting keeps the full-detail target and simply makes the fovea count
    for more, so the optimiser reallocates effort. Blurring replaces the
    target, which destroys information and, worse, makes the target set
    inconsistent: each view is blurred about its own centre, so no single
    physical field can satisfy all of them at once. The unblurred light field
    is realisable by construction, because a real scene produced it.

    Measured on this repository's own geometry, blurring lost 2.4 to 5.7 dB at
    the fovea against no foveation at all. `useful_knowledge.md` records it.

    `normalise` scales the weights to unit mean, which keeps the loss
    magnitude comparable with the unweighted one so a learning rate tuned
    without foveation still applies.
    """
    weight = 60.0 / mrgcf_spacing_arcmin(eccentricity, meridian)
    return weight / weight.mean() if normalise else weight

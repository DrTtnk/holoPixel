"""Per-channel refractive indices for narrow-band display primaries.

Each optical element was designed at one index (n_base, taken at the d line).
At another wavelength it keeps its shape and takes its material's dispersion:
its power scales with the material's (n - 1), so

    n'(lambda) = 1 + (n_base - 1) (n(lambda) - 1) / (n(d) - 1).

Material data: refractiveindex.info through optiland.
"""
from __future__ import annotations

import numpy as np
from optiland.materials import Material

D_LINE_UM = 0.5876
PRIMARIES_UM = (0.620, 0.530, 0.460)          # narrow-band R, G, B (directly patterned micro-OLED)
MATERIALS = {"nlasf46b": ("N-LASF46B", "schott"), "silica": ("SiO2", "malitson"), "pmma": ("PMMA", None)}


def _index(material, um):
    name, ref = MATERIALS[material]
    m = Material(name, reference=ref) if ref else Material(name)
    return float(np.asarray(m.n(um)).ravel()[0])


def channel_indices(n_base, material, wavelengths_um=PRIMARIES_UM):
    """The element's index at each wavelength (a tuple, R G B by default)."""
    nd = _index(material, D_LINE_UM)
    return tuple(1.0 + (n_base - 1.0) * (_index(material, um) - 1.0) / (nd - 1.0) for um in wavelengths_um)

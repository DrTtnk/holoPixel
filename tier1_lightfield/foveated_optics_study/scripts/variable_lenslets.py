"""The variable-focal lenslet array of the retina-matched target, oriented to a
remapper's image.

The lens at panel (u, v) serves the field direction foveation_target maps
there, so its focal length is foveation_target.lenslet_focal_um(u, v). A
remapper may form its image upside down relative to that map (a real image
through a lens or a concave mirror): `flip_v` = -1 builds the array for such an
image, focal_um(u, v) = lenslet_focal_um(u, -v). The map is left-right
symmetric, so a horizontal flip needs nothing.

vertex_height_um gives the smooth surface the lens vertices lie on (the
remapper must focus the field there); bowl_table tabulates it for the tracers.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

import foveation_target as ft
import mla_mesh
import screen_spec as spec

PANEL_UM = spec.PANEL_MM * 1e3
TABLE_STEP_MM = 0.02
TABLE_HALF_MM = spec.PANEL_MM / 2.0 + 0.2


def focal_of_position(flip_v):
    if flip_v not in (1, -1):
        raise ValueError(f"flip_v must be +1 or -1, not {flip_v!r}")
    return lambda u_um, v_um: ft.lenslet_focal_um(np.asarray(u_um), flip_v * np.asarray(v_um))


def _profile(flip_v):
    return dict(panel_um=PANEL_UM, side_um=spec.LENS_SIDE_UM, focal_of_position=focal_of_position(flip_v),
                index=spec.LENS_INDEX, min_thickness_um=spec.LENS_MIN_THICKNESS_UM)


def build(flip_v, subdivisions=2):
    return mla_mesh.build_variable(subdivisions=subdivisions, **_profile(flip_v))


@lru_cache(maxsize=4)
def _gap_um(flip_v):
    centres = mla_mesh.mla.hex_centres_um(PANEL_UM + 4.0 * spec.LENS_SIDE_UM, spec.LENS_SIDE_UM)
    return mla_mesh.variable_lens_heights(centres, focal_of_position(flip_v), spec.LENS_INDEX,
                                          spec.LENS_SIDE_UM, spec.LENS_MIN_THICKNESS_UM)[3]


def vertex_height_um(u_um, v_um, flip_v):
    """Height of the lens vertices above the array's flat face: n (f - gap)."""
    return spec.LENS_INDEX * (focal_of_position(flip_v)(u_um, v_um) - _gap_um(flip_v))


@lru_cache(maxsize=4)
def bowl_table(flip_v):
    """(grid_u mm, grid_v mm, height mm relative to the panel centre's vertex) on a
    uniform grid over the panel plus a 0.2 mm margin."""
    g = np.arange(-TABLE_HALF_MM, TABLE_HALF_MM + 1e-9, TABLE_STEP_MM)
    U, V = np.meshgrid(g, g, indexing="ij")
    h = vertex_height_um(U * 1e3, V * 1e3, flip_v) - vertex_height_um(0.0, 0.0, flip_v)
    return g, g, 1e-3 * h

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

import dataclasses
from functools import lru_cache

import numpy as np
from matplotlib.path import Path

import foveation_target as ft
import mla_mesh
import screen_spec as spec

PANEL_UM = spec.PANEL_MM * 1e3
TABLE_STEP_MM = 0.02
TABLE_HALF_MM = spec.PANEL_MM / 2.0 + 0.2


def focal_of_position(flip_v, pitch_um=spec.LENS_PITCH_UM):
    if flip_v not in (1, -1):
        raise ValueError(f"flip_v must be +1 or -1, not {flip_v!r}")
    return lambda u_um, v_um: ft.lenslet_focal_um(np.asarray(u_um), flip_v * np.asarray(v_um), pitch_um)


def _side_um(pitch_um):
    return pitch_um / np.sqrt(3.0)


def _profile(flip_v, pitch_um):
    return dict(panel_um=PANEL_UM, side_um=_side_um(pitch_um), focal_of_position=focal_of_position(flip_v, pitch_um),
                index=spec.LENS_INDEX, min_thickness_um=spec.LENS_MIN_THICKNESS_UM)


MASK_STEP_MM = 0.02


def mask_table(polygon_mm):
    """A black mask's opening as a table: (grid_u mm, grid_v mm, open) over the
    panel plus 1 mm, open where the grid point is inside the closed polygon (M, 2)
    in array coordinates."""
    g = np.arange(-TABLE_HALF_MM - 0.8, TABLE_HALF_MM + 0.8 + 1e-9, MASK_STEP_MM)
    U, V = np.meshgrid(g, g, indexing="ij")
    return g, g, Path(polygon_mm).contains_points(np.column_stack([U.ravel(), V.ravel()])).reshape(U.shape)


def masked(centres_um, table):
    """True for the lenses behind the mask: their centre is outside its opening."""
    gu, gv, is_open = table
    c = np.asarray(centres_um, float) * 1e-3
    i = np.rint((c[:, 0] - gu[0]) / (gu[1] - gu[0])).astype(int)
    j = np.rint((c[:, 1] - gv[0]) / (gv[1] - gv[0])).astype(int)
    if i.min() < 0 or j.min() < 0 or i.max() >= len(gu) or j.max() >= len(gv):
        raise ValueError("a lens lies beyond the mask table")
    return ~is_open[i, j]


def with_mask(arr, table):
    """The array with the top faces of the lenses behind the mask black, like the walls."""
    hidden = np.nonzero(masked(arr.mesh.centres, table))[0]
    face_lens = np.where(np.isin(arr.mesh.face_lens, hidden), mla_mesh.WALL, arr.mesh.face_lens)
    return dataclasses.replace(arr, mesh=dataclasses.replace(arr.mesh, face_lens=face_lens))


def build(flip_v, subdivisions=2, pitch_um=spec.LENS_PITCH_UM):
    """subdivisions per spec.LENS_PITCH_UM of pitch: the mesh spacing stays the
    same in um, so the border triangles overshoot the hex corners by the same
    share of the side (the floor lens covers 1.15 side) whatever the pitch."""
    scaled = subdivisions * pitch_um / spec.LENS_PITCH_UM
    if abs(scaled - round(scaled)) > 1e-9:
        raise ValueError(f"pitch {pitch_um} um is not a whole multiple of the mesh spacing")
    return mla_mesh.build_variable(subdivisions=int(round(scaled)), **_profile(flip_v, pitch_um))


@lru_cache(maxsize=8)
def _gap_um(flip_v, pitch_um=spec.LENS_PITCH_UM):
    side = _side_um(pitch_um)
    centres = mla_mesh.mla.hex_centres_um(PANEL_UM + 4.0 * side, side)
    return mla_mesh.variable_lens_heights(centres, focal_of_position(flip_v, pitch_um), spec.LENS_INDEX,
                                          side, spec.LENS_MIN_THICKNESS_UM)[3]


def vertex_height_um(u_um, v_um, flip_v, pitch_um=spec.LENS_PITCH_UM):
    """Height of the lens vertices above the array's flat face: n (f - gap)."""
    return spec.LENS_INDEX * (focal_of_position(flip_v, pitch_um)(u_um, v_um) - _gap_um(flip_v, pitch_um))


@lru_cache(maxsize=8)
def bowl_table(flip_v, pitch_um=spec.LENS_PITCH_UM):
    """(grid_u mm, grid_v mm, height mm relative to the panel centre's vertex) on a
    uniform grid over the panel plus a 0.2 mm margin."""
    g = np.arange(-TABLE_HALF_MM, TABLE_HALF_MM + 1e-9, TABLE_STEP_MM)
    U, V = np.meshgrid(g, g, indexing="ij")
    h = vertex_height_um(U * 1e3, V * 1e3, flip_v, pitch_um) - vertex_height_um(0.0, 0.0, flip_v, pitch_um)
    return g, g, 1e-3 * h

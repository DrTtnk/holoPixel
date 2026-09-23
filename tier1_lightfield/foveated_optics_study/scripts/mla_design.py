"""Plano-convex hexagonal microlens array: lattice and paraxial prescription.

Geometry, along the light path: pixel, air gap g, flat face of the substrate,
glass of centre thickness t, convex face of radius R, air. The convex face is
the only powered surface, so f = R / (n - 1), and the pixel sits at the front
focal point when g = f - t / n (tests/test_mla_design.py derives this with
ray-transfer matrices). A pixel offset y from the lens axis then leaves as a
collimated beam at angle -y / f.

The lattice is flat-top hexagons of circumradius `side`: columns 1.5*side
apart, rows sqrt(3)*side apart, odd columns shifted by half a row. Every lens
fills its own hexagon, so the convex faces of neighbours meet on the shared
edge with no gap.
"""
from __future__ import annotations

import numpy as np


def focal_length_um(radius_um, index):
    return radius_um / (index - 1.0)


def radius_for_focal_length_um(focal_um, index):
    return focal_um * (index - 1.0)


def back_focal_gap_um(radius_um, index, centre_thickness_um):
    return focal_length_um(radius_um, index) - centre_thickness_um / index


def sag_um(r_um, radius_um):
    r = np.asarray(r_um, dtype=float)
    if np.any(r >= radius_um):
        raise ValueError(f"aperture radius {float(np.max(r))} um reaches the sphere radius {radius_um} um")
    return radius_um - np.sqrt(radius_um * radius_um - r * r)


def column_pitch_um(side_um):
    return 1.5 * side_um


def row_pitch_um(side_um):
    return np.sqrt(3.0) * side_um


def hex_centres_um(panel_um, side_um):
    """Lens centres, symmetric about the panel centre, fully inside the panel."""
    limit = panel_um / 2.0 - side_um
    dx, dy = column_pitch_um(side_um), row_pitch_um(side_um)
    n_col = int(np.floor(limit / dx))
    n_row = int(np.floor(limit / dy)) + 1
    cols = np.arange(-n_col, n_col + 1)
    rows = np.arange(-n_row, n_row + 1)
    i, j = np.meshgrid(cols, rows, indexing="ij")
    x = dx * i
    y = dy * (j + 0.5 * (i % 2))
    keep = np.abs(y) <= limit
    return np.stack([x[keep], y[keep]], axis=1)

"""Watertight triangle mesh of a plano-convex hex MLA, in the panel frame (um).

u, v span the panel and w points from the panel towards the eye. The flat face
is w = 0; each lens is a sphere cap clipped to its hexagon, peaking at the
centre thickness.

Flat-top hex centres and corners all lie on one triangular lattice of spacing
side / N, so meshing that lattice puts a mesh edge exactly on every hex edge:
the crease between neighbouring lenses is represented exactly, not sampled.
Loop normals are the exact sphere normals of the owning lens, so the renderer
refracts through the true surface normal, not the facet normal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

import mla_design as mla


@dataclass(frozen=True)
class MlaMesh:
    verts: np.ndarray         # (V, 3) um
    faces: np.ndarray         # (F, 3) int, wound outwards
    loop_normals: np.ndarray  # (3F, 3) unit, in face-corner order
    face_lens: np.ndarray     # (F,) owning lens index, -1 for wall and bottom faces
    centres: np.ndarray       # (L, 2) um
    centre_thickness: float   # um


def surface_height_um(uv, centres, radius_um, centre_thickness_um):
    _, owner = cKDTree(centres).query(uv)
    r = np.linalg.norm(uv - centres[owner], axis=1)
    return centre_thickness_um - mla.sag_um(r, radius_um)


def _rows(panel_um, spacing):
    half = panel_um / 2.0
    h = spacing * np.sqrt(3.0) / 2.0
    n_max = int(np.floor((half - 0.3 * h) / h))
    rows = [(-half, np.arange(-half, half + 0.5 * spacing, spacing))]
    for n in range(-n_max, n_max + 1):
        shift = 0.5 * spacing * (n % 2)
        k_max = int(np.ceil(half / spacing)) + 1
        x = spacing * np.arange(-k_max, k_max + 1) + shift
        x = x[np.abs(x) < half - 0.3 * spacing]
        rows.append((n * h, np.concatenate([[-half], x, [half]])))
    rows.append((half, np.arange(-half, half + 0.5 * spacing, spacing)))
    fixed = []
    for y, x in rows:
        x = np.unique(np.clip(x, -half, half))
        fixed.append((y, x))
    return fixed


def _zip_strip(a0, na, b0, nb, xa, xb):
    """Triangulate the strip between two sorted rows sharing both end x values."""
    xs = np.concatenate([xa[1:], xb[1:]])
    tag = np.concatenate([np.zeros(na - 1, int), np.ones(nb - 1, int)])
    order = np.lexsort((tag, xs))
    tag = tag[order]
    ia = np.concatenate([[0], np.cumsum(tag == 0)])
    ib = np.concatenate([[0], np.cumsum(tag == 1)])
    prev_a, prev_b = ia[:-1], ib[:-1]
    tri = np.where(
        (tag == 0)[:, None],
        np.column_stack([a0 + prev_a, a0 + prev_a + 1, b0 + prev_b]),
        np.column_stack([a0 + prev_a, b0 + prev_b + 1, b0 + prev_b]),
    )
    return tri


def build(panel_um, side_um, radius_um, min_thickness_um, subdivisions):
    # Include the partial lenses the panel edge cuts, so every point of the panel
    # is within one circumradius of a centre (a short-focus sphere ends there).
    centres = mla.hex_centres_um(panel_um + 4.0 * side_um, side_um)
    t_c = min_thickness_um + float(mla.sag_um(side_um, radius_um))
    spacing = side_um / subdivisions
    half = panel_um / 2.0

    rows = _rows(panel_um, spacing)
    starts, uv = [], []
    for y, x in rows:
        starts.append(sum(len(r) for r in uv))
        uv.append(np.column_stack([x, np.full(len(x), y)]))
    lengths = [len(r) for r in uv]
    uv = np.concatenate(uv)
    n_top = len(uv)

    top_faces = np.concatenate([
        _zip_strip(starts[k], lengths[k], starts[k + 1], lengths[k + 1], rows[k][1], rows[k + 1][1])
        for k in range(len(rows) - 1)
    ])
    p = uv[top_faces]
    area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - \
           (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    if np.any(area <= 0):
        raise AssertionError(f"{int(np.sum(area <= 0))} top faces are degenerate or wound inwards")

    top = np.column_stack([uv, surface_height_um(uv, centres, radius_um, t_c)])

    bottom_row = np.arange(starts[0], starts[0] + lengths[0])
    top_row = np.arange(starts[-1], starts[-1] + lengths[-1])
    right = np.array([s + l - 1 for s, l in zip(starts, lengths)])
    left = np.array(starts)
    loop = np.concatenate([bottom_row[:-1], right[:-1], top_row[::-1][:-1], left[::-1][:-1]])

    bottom = np.column_stack([uv[loop], np.zeros(len(loop))])
    b = n_top + np.arange(len(loop))
    b_next = np.roll(b, -1)
    t, t_next = loop, np.roll(loop, -1)
    walls = np.concatenate([np.column_stack([b, b_next, t_next]), np.column_stack([b, t_next, t])])
    centre_index = n_top + len(loop)
    floor = np.column_stack([np.full(len(loop), centre_index), b_next, b])

    verts = np.concatenate([top, bottom, [[0.0, 0.0, 0.0]]])
    faces = np.concatenate([top_faces, walls, floor]).astype(np.int64)

    _, owner = cKDTree(centres).query(uv[top_faces].mean(axis=1))
    face_lens = np.concatenate([owner, np.full(len(walls) + len(floor), -1)])

    corners = verts[faces]
    flat = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)
    loop_normals = np.repeat(flat[:, None, :], 3, axis=1)
    n_tf = len(top_faces)
    duv = corners[:n_tf, :, :2] - centres[owner][:, None, :]
    rise = np.sqrt(radius_um**2 - np.sum(duv**2, axis=2))
    loop_normals[:n_tf] = np.concatenate([duv, rise[:, :, None]], axis=2) / radius_um

    return MlaMesh(verts=verts, faces=faces, loop_normals=loop_normals.reshape(-1, 3),
                   face_lens=face_lens, centres=centres, centre_thickness=t_c)

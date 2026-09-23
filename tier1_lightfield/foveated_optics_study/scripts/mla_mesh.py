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


@dataclass(frozen=True)
class VariableMla:
    mesh: MlaMesh
    focal_um: np.ndarray          # (L,) per lens
    radius_um: np.ndarray         # (L,) per lens
    centre_height_um: np.ndarray  # (L,) glass thickness on each lens axis
    gap_um: float                 # the one air gap between the flat face and the panel


MERGE_UM = 1e-6


def _zip_chains(p, q, hp, hq):
    """Triangulate the planar strip bounded by chain p (p[0] -> p[-1]) and chain q
    (q[0] -> q[-1]) whose cyclic order is p[0] .. p[-1], q[-1] .. q[0]. Both chains
    run monotonically in height; they are zipped by normalised height."""
    def param(h):
        span = h[-1] - h[0]
        return np.zeros(len(h)) if abs(span) < MERGE_UM else (h - h[0]) / span
    sp, sq = param(np.asarray(hp)), param(np.asarray(hq))
    tris, i, j = [], 0, 0
    while i < len(p) - 1 or j < len(q) - 1:
        if j == len(q) - 1 or (i < len(p) - 1 and sp[i + 1] <= sq[j + 1]):
            tris.append((p[i], p[i + 1], q[j]))
            i += 1
        else:
            tris.append((p[i], q[j + 1], q[j]))
            j += 1
    return tris


def variable_lens_heights(centres, focal_of_radius, index, side_um, min_thickness_um):
    """Per lens: focal length, sphere radius, axial glass height, and the one air
    gap, for lenses that all focus on a flat panel (f - h / n = gap)."""
    focal = np.asarray(focal_of_radius(np.linalg.norm(centres, axis=1)), dtype=float)
    radius = mla.radius_for_focal_length_um(focal, index)
    gap = float(np.min(focal - (min_thickness_um + mla.sag_um(side_um, radius)) / index))
    return focal, radius, index * (focal - gap), gap


def variable_vertex_profile_um(r_um, panel_um, side_um, focal_of_radius, index, min_thickness_um):
    """Height of the lens vertices above the flat face, as a smooth function of
    panel radius: n (f(r) - gap), the gap being the array's own."""
    centres = mla.hex_centres_um(panel_um + 4.0 * side_um, side_um)
    _, _, _, gap = variable_lens_heights(centres, focal_of_radius, index, side_um, min_thickness_um)
    return index * (np.asarray(focal_of_radius(np.asarray(r_um)), dtype=float) - gap)


def build_variable(panel_um, side_um, focal_of_radius, index, min_thickness_um, subdivisions):
    """Hex MLA whose lens focal lengths follow focal_of_radius(|centre|) (um), all
    focused on one flat panel: lens i stands on a glass column of height
    h_i = n (f_i - g), with the air gap g chosen so the thinnest lens is
    min_thickness_um at its hex corners. Neighbouring lenses of different height
    meet at vertical walls; where three lenses meet, the wall edges stack."""
    centres = mla.hex_centres_um(panel_um + 4.0 * side_um, side_um)
    focal, radius, height, gap = variable_lens_heights(centres, focal_of_radius, index, side_um, min_thickness_um)

    spacing = side_um / subdivisions
    rows = _rows(panel_um, spacing)
    starts, uv = [], []
    for y, x in rows:
        starts.append(sum(len(r) for r in uv))
        uv.append(np.column_stack([x, np.full(len(x), y)]))
    lengths = [len(r) for r in uv]
    uv = np.concatenate(uv)
    lattice_faces = np.concatenate([
        _zip_strip(starts[k], lengths[k], starts[k + 1], lengths[k + 1], rows[k][1], rows[k + 1][1])
        for k in range(len(rows) - 1)
    ])
    _, owner = cKDTree(centres).query(uv[lattice_faces].mean(axis=1))

    # one vertex per (lattice vertex, height): lenses of equal height share it
    corner_v = lattice_faces.ravel()
    corner_o = np.repeat(owner, 3)
    r = np.linalg.norm(uv[corner_v] - centres[corner_o], axis=1)
    corner_h = height[corner_o] - mla.sag_um(r, radius[corner_o])
    # cluster: per lattice vertex, sort the heights and start a new copy only
    # where the next height is more than MERGE_UM above the previous one
    order = np.lexsort((corner_h, corner_v))
    v_sorted, h_sorted = corner_v[order], corner_h[order]
    new = np.ones(len(order), dtype=bool)
    new[1:] = (v_sorted[1:] != v_sorted[:-1]) | (np.diff(h_sorted) > MERGE_UM)
    copy_sorted = np.cumsum(new) - 1
    copy_of_corner = np.empty(len(order), dtype=np.int64)
    copy_of_corner[order] = copy_sorted
    n_copies = int(copy_sorted[-1]) + 1
    copy_v = v_sorted[new]
    copy_h = h_sorted[new]
    top_faces = copy_of_corner.reshape(-1, 3)
    verts = [np.column_stack([uv[copy_v], copy_h])]

    stacks = {}
    for c, v in enumerate(copy_v):
        stacks.setdefault(int(v), []).append(c)                           # already sorted by height

    def chain(v, h_from, h_to):
        lo, hi = min(h_from, h_to), max(h_from, h_to)
        cs = [c for c in stacks[v] if lo - MERGE_UM <= copy_h[c] <= hi + MERGE_UM]
        return cs if h_from <= h_to else cs[::-1]

    # every directed edge of the lattice faces, with its face and corner slots
    n_f = len(lattice_faces)
    e_face = np.tile(np.arange(n_f), 3)
    e_k0 = np.repeat(np.arange(3), n_f)
    e_k1 = (e_k0 + 1) % 3
    e_a, e_b = lattice_faces[e_face, e_k0], lattice_faces[e_face, e_k1]
    n_v = len(uv)
    undirected = np.minimum(e_a, e_b) * n_v + np.maximum(e_a, e_b)
    order = np.argsort(undirected, kind="stable")
    same = undirected[order][1:] == undirected[order][:-1]
    first, second = order[:-1][same], order[1:][same]                  # the two sides of an interior edge
    swap = owner[e_face[first]] > owner[e_face[second]]
    e1 = np.where(swap, second, first)
    e2 = np.where(swap, first, second)
    step = owner[e_face[e1]] != owner[e_face[e2]]
    walls = []
    for i, j in zip(e1[step], e2[step]):
        f, g_ = e_face[i], e_face[j]
        a, b = int(e_a[i]), int(e_b[i])
        ha1, hb1 = copy_h[top_faces[f, e_k0[i]]], copy_h[top_faces[f, e_k1[i]]]
        hb2, ha2 = copy_h[top_faces[g_, e_k0[j]]], copy_h[top_faces[g_, e_k1[j]]]     # face g runs b -> a
        p, q = chain(a, ha1, ha2), chain(b, hb1, hb2)
        if len(p) == 1 and len(q) == 1:
            continue
        walls += _zip_chains(p, q, copy_h[p], copy_h[q])
    lone = np.ones(len(undirected), dtype=bool)
    lone[first], lone[second] = False, False
    directed = {(int(e_a[i]), int(e_b[i])): int(i) for i in np.nonzero(lone)[0]}  # the outer border only

    bottom_row = np.arange(starts[0], starts[0] + lengths[0])
    top_row = np.arange(starts[-1], starts[-1] + lengths[-1])
    right = np.array([s + l - 1 for s, l in zip(starts, lengths)])
    left = np.array(starts)
    loop = np.concatenate([bottom_row[:-1], right[:-1], top_row[::-1][:-1], left[::-1][:-1]])
    bottom_id = {int(v): n_copies + k for k, v in enumerate(loop)}
    verts.append(np.column_stack([uv[loop], np.zeros(len(loop))]))
    border = []
    for t, t_next in zip(loop, np.roll(loop, -1)):
        i = directed[(int(t), int(t_next))]
        f = e_face[i]
        ht, hn = copy_h[top_faces[f, e_k0[i]]], copy_h[top_faces[f, e_k1[i]]]
        p = chain(int(t), ht, 0.0) + [bottom_id[int(t)]]
        q = chain(int(t_next), hn, 0.0) + [bottom_id[int(t_next)]]
        hs = np.concatenate([copy_h, np.zeros(len(loop))])
        border += _zip_chains(p, q, hs[p], hs[q])
    centre_index = n_copies + len(loop)
    b = np.array([bottom_id[int(v)] for v in loop])
    floor = np.column_stack([np.full(len(loop), centre_index), np.roll(b, -1), b])
    verts.append([[0.0, 0.0, 0.0]])
    verts = np.concatenate(verts)
    faces = np.concatenate([top_faces, np.array(walls, dtype=np.int64).reshape(-1, 3),
                            np.array(border, dtype=np.int64), floor]).astype(np.int64)
    face_lens = np.concatenate([owner, np.full(len(faces) - len(top_faces), -1)])

    corners = verts[faces]
    flat = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)
    loop_normals = np.repeat(flat[:, None, :], 3, axis=1)
    n_tf = len(top_faces)
    duv = corners[:n_tf, :, :2] - centres[owner][:, None, :]
    rr = radius[owner][:, None]
    rise = np.sqrt(rr**2 - np.sum(duv**2, axis=2))
    loop_normals[:n_tf] = np.concatenate([duv, rise[:, :, None]], axis=2) / rr[:, :, None]
    m = MlaMesh(verts=verts, faces=faces, loop_normals=loop_normals.reshape(-1, 3), face_lens=face_lens,
                centres=centres, centre_thickness=float(height.max()))
    return VariableMla(mesh=m, focal_um=focal, radius_um=radius, centre_height_um=height, gap_um=gap)

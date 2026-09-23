"""Tessellate the optimised coaxial prescription into watertight glass solids
and write design.json + remapper.npz, per lf_evaluate.py's export contract.

Each element is a solid of revolution about the Y axis: a front cap (convex
toward the eye, outward normal roughly -Y) and a back cap (outward roughly
+Y), joined by a cylindrical rim at the clear aperture. Both caps are built
on the SAME polar grid (apex + concentric rings, n_phi points per ring) so
the rim can zip the two outer rings together edge-for-edge, exactly as
mla_mesh.py zips its top surface to its flat bottom.

Winding convention (matches mla_mesh.py's own top/bottom convention, which
the shared evaluator already accepts): a cap whose outward normal points
toward -Y (front, "top"-like) is wound CCW as seen in the (x, z) projection;
a cap facing +Y (back, "bottom"-like) is wound CW in that same projection.
The two curved caps get exact analytic per-corner normals (evaluated with
the same torch sag/dsag_dr as the optimiser, so mesh position error is the
only thing tessellation can still get wrong); the flat rim gets flat
per-face normals, as mla_mesh.py does for its own walls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import optics as op  # noqa: E402

DTYPE = op.DTYPE


def _polar_rings(n_r, n_phi, aperture_mm):
    """Ring index (i, j) -> (x, z), i in [0, n_r-1] (radius increasing,
    excludes r=0 which is the separate apex vertex 0), j in [0, n_phi-1]."""
    r = np.linspace(0.0, aperture_mm, n_r + 1)[1:]
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    rr, pp = np.meshgrid(r, phi, indexing="ij")  # (n_r, n_phi)
    x, z = rr * np.cos(pp), rr * np.sin(pp)
    return rr, x, z


def _cap_mesh(surface, aperture_mm, n_r, n_phi, facing_front):
    """One curved cap. facing_front=True: outward ~ -Y, CCW-in-(x,z) winding.
    facing_front=False: outward ~ +Y, CW-in-(x,z) winding (reverse of CCW)."""
    rr, x, z = _polar_rings(n_r, n_phi, aperture_mm)
    r_t = torch.tensor(np.concatenate([[0.0], rr.ravel()]), dtype=DTYPE)
    y_t = surface.vertex_y + op.sag(r_t, surface)
    x_all = np.concatenate([[0.0], x.ravel()])
    z_all = np.concatenate([[0.0], z.ravel()])
    verts = np.stack([x_all, y_t.detach().numpy(), z_all], axis=1)

    def ring_idx(i, j):
        return 1 + i * n_phi + (j % n_phi)

    tris = []
    for j in range(n_phi):
        tris.append([0, ring_idx(0, j), ring_idx(0, j + 1)])
    for i in range(1, n_r):
        for j in range(n_phi):
            a, b, c, d = ring_idx(i - 1, j), ring_idx(i - 1, j + 1), ring_idx(i, j + 1), ring_idx(i, j)
            tris.append([a, b, c])
            tris.append([a, c, d])
    faces = np.array(tris, dtype=np.int64)
    if not facing_front:
        faces = faces[:, ::-1]

    # Exact analytic per-CORNER normal: outward = -grad(F) for the front cap
    # (glass occupies the +F side), +grad(F) for the back cap (see optics.py
    # and NOTES.md). grad(F) = (-dsag_dr * x/r, 1, -dsag_dr * z/r).
    corner_xz = verts[faces][:, :, [0, 2]]
    r_corner = np.linalg.norm(corner_xz, axis=2)
    r_corner_t = torch.tensor(r_corner, dtype=DTYPE)
    dsdr = op.dsag_dr(r_corner_t, surface).detach().numpy()
    safe_r = np.where(r_corner > 1e-12, r_corner, 1.0)
    gx = -dsdr * corner_xz[:, :, 0] / safe_r
    gz = -dsdr * corner_xz[:, :, 1] / safe_r
    gy = np.ones_like(gx)
    grad = np.stack([gx, gy, gz], axis=2)
    grad /= np.linalg.norm(grad, axis=2, keepdims=True)
    sign = -1.0 if facing_front else 1.0
    normals = (sign * grad).reshape(-1, 3)

    outer_ring = np.array([ring_idx(n_r - 1, j) for j in range(n_phi)])
    return verts, faces, normals, outer_ring


def _rim(front_outer, back_outer, n_phi, verts):
    """Zip the front cap's outer ring ('top', CCW) to the back cap's outer
    ring ('bottom'), same pattern as mla_mesh.py's wall construction."""
    t, b = front_outer, back_outer
    t_next, b_next = np.roll(t, -1), np.roll(b, -1)
    walls = np.concatenate([np.column_stack([b, b_next, t_next]), np.column_stack([b, t_next, t])])
    corners = verts[walls]
    flat = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)
    normals = np.repeat(flat[:, None, :], 3, axis=1).reshape(-1, 3)
    return walls, normals


def build_element_solid(front, back, aperture_mm, n_r=48, n_phi=96):
    """One closed, outward-wound glass solid: front cap + back cap + rim."""
    fv, ff, fn, f_outer = _cap_mesh(front, aperture_mm, n_r, n_phi, facing_front=True)
    bv, bf, bn, b_outer = _cap_mesh(back, aperture_mm, n_r, n_phi, facing_front=False)
    off = len(fv)
    verts = np.concatenate([fv, bv])
    faces = np.concatenate([ff, bf + off])
    normals = np.concatenate([fn, bn])
    rim_faces, rim_normals = _rim(f_outer, b_outer + off, n_phi, verts)
    faces = np.concatenate([faces, rim_faces]).astype(np.int64)
    normals = np.concatenate([normals, rim_normals])
    oriented = orient_outward(verts, faces)
    if not np.array_equal(oriented, faces):
        # Reversing a face's vertex order must reverse its 3 per-corner
        # (loop) normals the same way, or the loop-normal-to-corner
        # correspondence Blender's normals_split_custom_set expects breaks.
        normals = normals.reshape(-1, 3, 3)[:, ::-1, :].reshape(-1, 3)
        faces = oriented
    return verts, faces, normals


def check_closed_and_outward(verts, faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if not np.all(counts == 2):
        raise AssertionError(f"not closed: {int(np.sum(counts != 2))} open edges")
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    vol = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    if vol <= 0:
        raise AssertionError(f"wound inwards: signed volume {vol}")
    return vol


def orient_outward(verts, faces):
    """A mesh built as a topological sphere (as build_element_solid's cap +
    cap + rim always is, by construction: every quad split with the SAME
    convention) is consistently wound one way or the other -- there is no
    such thing as a partially-inward mesh once it passes the closure check.
    Which of the two global orientations is "outward" was found (see
    NOTES.md) to flip with aperture for a strongly curved front/back pair:
    the fixed "front cap is CCW in (x, z)" convention used to build each cap
    is only a heuristic, and stops matching reality once the surfaces curve
    enough that their (x, z)-projection is no longer the dominant term in
    the true 3-D orientation. Rather than trust the heuristic, this checks
    the sign directly and reverses every face if needed -- always valid,
    because reversing every face of a closed, consistently-oriented mesh
    exactly flips the sign of its enclosed volume, and does nothing else."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    vol = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    return faces if vol > 0 else faces[:, ::-1]

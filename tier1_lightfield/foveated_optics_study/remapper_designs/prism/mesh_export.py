"""Builds the ONE watertight glass solid (S1 used twice, S2 mirror-coated,
S3 entry) from the optimized freeform surfaces, and writes design.json +
remapper.npz in the export contract lf_evaluate.py expects.

Topology: three curved rectangular patches (S1, S2, S3), all sharing the
same world-X sampling (every surface's local x IS world X, since surfaces
only tilt about the X axis -- see raytrace.py). Each patch has two "u-edges"
(near its two neighbours); a ruled wall strip connects matched edges
DIRECTLY BY VERTEX INDEX (no new vertices), so every wall edge is shared
with its patch's own boundary edge automatically -- closed by construction,
not by best-effort stitching. The two ends of the X-extrusion are closed by
fan-triangulated caps from a single new centroid vertex, reusing the same
patch/wall rim vertices.

Winding: each patch's (i, j) grid triangulation is tested once against its
own analytic surface normal and flipped if backwards; walls and caps are
tested once against the vector from the solid's centroid to the face
centroid (the solid is star-shaped enough for this) and flipped the same
way. This is a deterministic per-block decision, not a per-face heuristic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raytrace as rt  # noqa: E402
from design import Design, mapping_grid, pupil_ring, field_direction, PUPIL_Y_MM  # noqa: E402

DTYPE = rt.DTYPE


def _grid_points(surface, x_vals, u_vals):
    """world (Nx, Nu, 3) points and (Nx, Nu, 3) analytic unit normals."""
    xg, ug = np.meshgrid(x_vals, u_vals, indexing="ij")
    x_l = torch.tensor(xg.ravel(), dtype=DTYPE)
    u_l = torch.tensor(ug.ravel(), dtype=DTYPE)
    pts = surface.point(x_l, u_l).detach().numpy().reshape(len(x_vals), len(u_vals), 3)
    nrm = surface.normal(x_l, u_l).detach().numpy().reshape(len(x_vals), len(u_vals), 3)
    return pts, nrm


def apertures(design, margin_frac=0.35, n_field=25, n_pupil=13):
    """(x,u) half-extents per surface from the footprint of a fine field x
    pupil grid, padded by margin_frac so the working rays stay well inside
    the optically active area (away from the wall seams)."""
    eccs = np.linspace(0.0, 35.0, n_field)
    azs = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    tx, tz = [], []
    for e in eccs:
        for a in azs:
            x, z = e * np.cos(a), e * np.sin(a)
            if abs(x) <= 35.0 and abs(z) <= 22.5:
                tx.append(x)
                tz.append(z)
    tx, tz = np.array(tx), np.array(tz)
    pupil = pupil_ring(n=n_pupil, radius=2.0)
    d = field_direction(torch.tensor(tx, dtype=DTYPE), torch.tensor(tz, dtype=DTYPE))
    d = d[:, None, :].expand(len(tx), len(pupil), 3).reshape(-1, 3)
    px = torch.tensor(pupil[:, 0], dtype=DTYPE)
    pz = torch.tensor(pupil[:, 1], dtype=DTYPE)
    origin = torch.stack([px, torch.full_like(px, PUPIL_Y_MM), pz], dim=-1)
    origin = origin[None, :, :].expand(len(tx), len(pupil), 3).reshape(-1, 3)
    prism = design.prism()
    out = prism.trace(origin, d, iters=12)
    if out["valid"].float().mean() < 0.9:
        raise RuntimeError(f"aperture probe: only {float(out['valid'].float().mean()):.3f} of rays valid")

    def bounds(surface, p):
        x_l, u_l, _ = surface.local(p)
        xh = float(x_l.abs().max()) * (1 + margin_frac)
        u0, u1 = float(u_l.min()), float(u_l.max())
        pad = (u1 - u0) * margin_frac
        return xh, u0 - pad, u1 + pad

    s1, s2, s3 = design.surfaces()
    b1a = bounds(s1, out["p1"])
    b1b = bounds(s1, out["p3"])
    b1 = (max(b1a[0], b1b[0]), min(b1a[1], b1b[1]), max(b1a[2], b1b[2]))
    b2 = bounds(s2, out["p2"])
    b3 = bounds(s3, out["p_out"])
    return b1, b2, b3


def label_edges(design, bounds):
    """For each surface, decide which u-edge (lo or hi) faces which neighbour,
    by comparing the edge midpoint to the two OTHER surfaces' apex positions."""
    s1, s2, s3 = design.surfaces()
    surfs = {"S1": s1, "S2": s2, "S3": s3}
    apex = {k: s.origin.detach().numpy() for k, s in surfs.items()}
    b = {"S1": bounds[0], "S2": bounds[1], "S3": bounds[2]}
    edge_pt = {}
    for k, s in surfs.items():
        xh, u0, u1 = b[k]
        for tag, u in [("lo", u0), ("hi", u1)]:
            p = s.point(torch.tensor(0.0, dtype=DTYPE), torch.tensor(u, dtype=DTYPE))
            edge_pt[(k, tag)] = p.detach().numpy()

    others = {"S1": ("S2", "S3"), "S2": ("S1", "S3"), "S3": ("S1", "S2")}
    label = {}
    for k in surfs:
        o1, o2 = others[k]
        d_lo = {o1: np.linalg.norm(edge_pt[(k, "lo")] - apex[o1]),
               o2: np.linalg.norm(edge_pt[(k, "lo")] - apex[o2])}
        d_hi = {o1: np.linalg.norm(edge_pt[(k, "hi")] - apex[o1]),
               o2: np.linalg.norm(edge_pt[(k, "hi")] - apex[o2])}
        # assign lo/hi to whichever neighbour it's closer to; must be a
        # consistent bijection onto {o1, o2}.
        if d_lo[o1] - d_hi[o1] < d_lo[o2] - d_hi[o2]:
            label[(k, "lo")], label[(k, "hi")] = o1, o2
        else:
            label[(k, "lo")], label[(k, "hi")] = o2, o1
    for a, b_ in [("S1", "S2"), ("S2", "S3"), ("S3", "S1")]:
        tag_a = "lo" if label[(a, "lo")] == b_ else "hi"
        tag_b = "lo" if label[(b_, "lo")] == a else "hi"
        if label[(a, tag_a)] != b_ or label[(b_, tag_b)] != a:
            raise RuntimeError(f"edge labelling inconsistent for {a}-{b_}")
    return label


def build_patch(surface, xh, u0, u1, nx, nu):
    x_vals = np.linspace(-xh, xh, nx)
    u_vals = np.linspace(u0, u1, nu)
    pts, nrm = _grid_points(surface, x_vals, u_vals)
    verts = pts.reshape(-1, 3)
    normals = nrm.reshape(-1, 3)

    def vid(i, j):
        return i * nu + j

    faces = []
    for i in range(nx - 1):
        for j in range(nu - 1):
            a, b, c, e = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            faces.append((a, b, c))
            faces.append((a, c, e))
    faces = np.array(faces, dtype=np.int64)

    # orient so the winding matches the analytic outward normal; loop_normals
    # are the exact per-vertex analytic normal at every face corner (smooth).
    p = verts[faces]
    flat = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    ref = normals[faces[:, 0]]
    if (flat * ref).sum(-1).mean() < 0:
        faces = faces[:, [0, 2, 1]]
    loop_normals = normals[faces].reshape(-1, 3)
    return verts, faces, loop_normals, nx, nu


def stitch_wall(base_a, nx, nu_a, edge_a, base_b, nu_b, edge_b, verts, centroid):
    """Ruled strip directly connecting patch A's edge column (j=edge_a) to
    patch B's edge column (j=edge_b), reusing existing vertex indices. Flat
    (faceted) loop_normals: these seam walls sit outside the working
    aperture by construction, so smoothness there is not required."""
    ja = 0 if edge_a == "lo" else nu_a - 1
    jb = 0 if edge_b == "lo" else nu_b - 1
    faces = []
    for i in range(nx - 1):
        a0, a1 = base_a + i * nu_a + ja, base_a + (i + 1) * nu_a + ja
        b0, b1 = base_b + i * nu_b + jb, base_b + (i + 1) * nu_b + jb
        faces.append((a0, a1, b1))
        faces.append((a0, b1, b0))
    faces = np.array(faces, dtype=np.int64)
    p = verts[faces]
    flat = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    ref = p.mean(axis=1) - centroid
    if (flat * ref).sum(-1).mean() < 0:
        faces = faces[:, [0, 2, 1]]
    p = verts[faces]
    flat = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    loop_normals = np.repeat(flat[:, None, :], 3, axis=1).reshape(-1, 3)
    return faces, loop_normals


def cap(loop_indices, verts, centroid, sign):
    """Fan-triangulate a rim loop (existing vertex indices, in order) from a
    NEW centroid vertex appended to verts. Returns (new_vertex, faces, loop_normals)."""
    rim = verts[loop_indices]
    c = rim.mean(axis=0)
    n = len(loop_indices)
    new_idx = len(verts)
    faces = [(new_idx, loop_indices[k], loop_indices[(k + 1) % n]) for k in range(n)]
    faces = np.array(faces, dtype=np.int64)
    all_verts = np.concatenate([verts, c[None, :]], axis=0)
    p = all_verts[faces]
    flat = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    ref = (p.mean(axis=1) - centroid) * sign
    if (flat * ref).sum(-1).mean() < 0:
        faces = faces[:, [0, 2, 1]]
    p = all_verts[faces]
    flat = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    loop_normals = np.repeat(flat[:, None, :], 3, axis=1).reshape(-1, 3)
    return c, faces, loop_normals


def build_solid(design, nx=41, nu=41):
    b1, b2, b3 = apertures(design)
    xh = max(b1[0], b2[0], b3[0])
    b1 = (xh, b1[1], b1[2])
    b2 = (xh, b2[1], b2[2])
    b3 = (xh, b3[1], b3[2])
    label = label_edges(design, (b1, b2, b3))
    s1, s2, s3 = design.surfaces()

    v1, f1, ln1, nx1, nu1 = build_patch(s1, xh, b1[1], b1[2], nx, nu)
    v2, f2, ln2, nx2, nu2 = build_patch(s2, xh, b2[1], b2[2], nx, nu)
    v3, f3, ln3, nx3, nu3 = build_patch(s3, xh, b3[1], b3[2], nx, nu)

    base1, base2, base3 = 0, len(v1), len(v1) + len(v2)
    verts = np.concatenate([v1, v2, v3], axis=0)
    faces = [f1 + base1, f2 + base2, f3 + base3]
    loop_normals = [ln1, ln2, ln3]
    n_mirror_faces = len(f2)  # S2's own faces, for the mirror_faces flag
    centroid = verts.mean(axis=0)

    bases = {"S1": (base1, nu1), "S2": (base2, nu2), "S3": (base3, nu3)}
    for a, b in [("S1", "S2"), ("S2", "S3"), ("S3", "S1")]:
        edge_a = "lo" if label[(a, "lo")] == b else "hi"
        edge_b = "lo" if label[(b, "lo")] == a else "hi"
        base_a, nu_a = bases[a]
        base_b, nu_b = bases[b]
        wf, wln = stitch_wall(base_a, nx, nu_a, edge_a, base_b, nu_b, edge_b, verts, centroid)
        faces.append(wf)
        loop_normals.append(wln)

    # Loop order S1 -> S2 -> S3 -> (back to S1). For each surface, walk its
    # FULL rim (all nu points, no vertex dropped) starting at the edge shared
    # with the PREVIOUS neighbour and ending at the edge shared with the
    # NEXT one, so consecutive loop entries are always real mesh edges: the
    # nu-1 edges inside a surface's own rim (second use of its boundary
    # edge), and the single edge at each transition (second use of that
    # wall's own boundary edge). No point is ever skipped or duplicated.
    order = ["S1", "S2", "S3"]
    next_of = {"S1": "S2", "S2": "S3", "S3": "S1"}
    prev_of = {"S1": "S3", "S2": "S1", "S3": "S2"}
    for i, sign in [(0, -1.0), (nx - 1, +1.0)]:
        loop = []
        for a in order:
            base_a, nu_a = bases[a]
            edge_from_prev = "lo" if label[(a, "lo")] == prev_of[a] else "hi"
            edge_to_next = "lo" if label[(a, "lo")] == next_of[a] else "hi"
            if edge_from_prev == edge_to_next:
                raise RuntimeError(f"{a}: prev/next neighbours share one edge label")
            j_start = 0 if edge_from_prev == "lo" else nu_a - 1
            j_end = 0 if edge_to_next == "lo" else nu_a - 1
            step = 1 if j_end > j_start else -1
            loop.extend(base_a + i * nu_a + j for j in range(j_start, j_end + step, step))
        c_pt, cf, cln = cap(np.array(loop, dtype=np.int64), verts, centroid, sign)
        verts = np.concatenate([verts, c_pt[None, :]], axis=0)
        faces.append(cf)
        loop_normals.append(cln)

    faces = np.concatenate(faces, axis=0)
    loop_normals = np.concatenate(loop_normals, axis=0)
    mirror_faces = np.zeros(len(faces), dtype=bool)
    mirror_faces[len(f1):len(f1) + n_mirror_faces] = True
    return verts, faces, loop_normals, mirror_faces


if __name__ == "__main__":
    z = np.load(sys.argv[1] if len(sys.argv) > 1 else "params.npz")
    design = Design(index=float(z["index"]), requires_grad=False)
    design.pose = torch.tensor(z["pose"], dtype=DTYPE)
    design.coeff = torch.tensor(z["coeff"], dtype=DTYPE)
    verts, faces, loop_normals, mirror_faces = build_solid(design)
    print("verts", verts.shape, "faces", faces.shape, "mirror", mirror_faces.sum())

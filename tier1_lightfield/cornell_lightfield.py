"""
Light Field Renderer — Numba Path Tracer

Renders a 4D Light Field L(u, v, y, x, c) of a Cornell box scene
using path tracing with next event estimation (NEE).

Physics: 1-2 bounce path tracing with cosine-weighted hemisphere
sampling and explicit light sampling. Produces the input light field
for the SLFH (Stochastic Light Field Holography) optimizer.
"""

import numpy as np
import numba
from numba import njit, prange
import math
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scene Definition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HS = 0.015      # box half-size (15mm)
Z0 = 0.1        # depth center (100mm)

# Material indices
MAT_WHITE = 0
MAT_RED = 1
MAT_GREEN = 2
MAT_LIGHT = 3

# Materials: [albedo_r, g, b, emission_r, g, b]
MATERIALS = np.array([
    [0.73, 0.73, 0.73, 0.0, 0.0, 0.0],      # white
    [0.65, 0.05, 0.05, 0.0, 0.0, 0.0],      # red
    [0.12, 0.45, 0.15, 0.0, 0.0, 0.0],      # green
    [0.78, 0.78, 0.78, 15.0, 15.0, 15.0],   # area light
], dtype=np.float64)


def build_scene():
    """Build Cornell box: axis-aligned quads (walls, light) + AABBs (inner boxes).

    Quad format: [axis, pos, a_min, a_max, b_min, b_max, normal_sign]
      axis=0(x): a=y, b=z  |  axis=1(y): a=x, b=z  |  axis=2(z): a=x, b=y
    """
    quads = []
    quad_mats = []

    # Floor: y = -HS, normal +y
    quads.append([1, -HS, -HS, HS, Z0 - HS, Z0 + HS, 1.0])
    quad_mats.append(MAT_WHITE)

    # Ceiling: y = +HS, normal -y
    quads.append([1, HS, -HS, HS, Z0 - HS, Z0 + HS, -1.0])
    quad_mats.append(MAT_WHITE)

    # Left wall: x = -HS, normal +x
    quads.append([0, -HS, -HS, HS, Z0 - HS, Z0 + HS, 1.0])
    quad_mats.append(MAT_RED)

    # Right wall: x = +HS, normal -x
    quads.append([0, HS, -HS, HS, Z0 - HS, Z0 + HS, -1.0])
    quad_mats.append(MAT_GREEN)

    # Back wall: z = Z0-HS (far from camera), normal +z (facing room interior)
    quads.append([2, Z0 - HS, -HS, HS, -HS, HS, 1.0])
    quad_mats.append(MAT_WHITE)

    # Area light on ceiling (slightly below to avoid z-fighting)
    light_hs = HS * 0.25
    quads.append([1, HS - 1e-4, -light_hs, light_hs,
                  Z0 - light_hs, Z0 + light_hs, -1.0])
    quad_mats.append(MAT_LIGHT)

    light_idx = len(quads) - 1

    # Inner boxes (AABBs) — matching holo_cornell.py
    tb_cx, tb_cy, tb_cz = -HS * 0.35, -HS + HS * 0.6, Z0 + HS * 0.2
    tb_sx, tb_sy, tb_sz = HS * 0.3, HS * 0.6, HS * 0.3

    sb_cx, sb_cy, sb_cz = HS * 0.35, -HS + HS * 0.25, Z0 - HS * 0.1
    sb_sx, sb_sy, sb_sz = HS * 0.3, HS * 0.25, HS * 0.3

    boxes = np.array([
        [tb_cx - tb_sx, tb_cy - tb_sy, tb_cz - tb_sz,
         tb_cx + tb_sx, tb_cy + tb_sy, tb_cz + tb_sz],
        [sb_cx - sb_sx, sb_cy - sb_sy, sb_cz - sb_sz,
         sb_cx + sb_sx, sb_cy + sb_sy, sb_cz + sb_sz],
    ], dtype=np.float64)
    box_mats = np.array([MAT_WHITE, MAT_WHITE], dtype=np.int32)

    return (np.array(quads, dtype=np.float64),
            np.array(quad_mats, dtype=np.int32),
            boxes, box_mats, light_idx)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Numba Ray Tracing Kernels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@njit(cache=True)
def _quad_perp(axis):
    if axis == 0:
        return 1, 2
    elif axis == 1:
        return 0, 2
    return 0, 1


@njit(cache=True)
def intersect_quad(ox, oy, oz, dx, dy, dz, quad):
    axis = numba.int32(quad[0])
    pos = quad[1]

    if axis == 0:
        d_ax, o_ax = dx, ox
    elif axis == 1:
        d_ax, o_ax = dy, oy
    else:
        d_ax, o_ax = dz, oz

    if abs(d_ax) < 1e-15:
        return -1.0

    t = (pos - o_ax) / d_ax
    if t < 1e-6:
        return -1.0

    a, b = _quad_perp(axis)
    o_vals = (ox, oy, oz)
    d_vals = (dx, dy, dz)

    ha = o_vals[a] + t * d_vals[a]
    hb = o_vals[b] + t * d_vals[b]

    if ha < quad[2] or ha > quad[3] or hb < quad[4] or hb > quad[5]:
        return -1.0

    return t


@njit(cache=True)
def intersect_aabb(ox, oy, oz, dx, dy, dz,
                   bx0, by0, bz0, bx1, by1, bz1):
    o = (ox, oy, oz)
    d = (dx, dy, dz)
    bmin = (bx0, by0, bz0)
    bmax = (bx1, by1, bz1)

    t_near = -1e30
    t_far = 1e30
    near_axis = 0
    near_sign = 0.0

    for i in range(3):
        if abs(d[i]) < 1e-15:
            if o[i] < bmin[i] or o[i] > bmax[i]:
                return -1.0, 0, 0.0
            continue

        inv_d = 1.0 / d[i]
        t1 = (bmin[i] - o[i]) * inv_d
        t2 = (bmax[i] - o[i]) * inv_d

        if t1 < t2:
            if t1 > t_near:
                t_near = t1
                near_axis = i
                near_sign = -1.0
            if t2 < t_far:
                t_far = t2
        else:
            if t2 > t_near:
                t_near = t2
                near_axis = i
                near_sign = 1.0
            if t1 < t_far:
                t_far = t1

        if t_near > t_far:
            return -1.0, 0, 0.0

    if t_near < 1e-6:
        return -1.0, 0, 0.0

    return t_near, near_axis, near_sign


@njit(cache=True)
def intersect_scene(ox, oy, oz, dx, dy, dz,
                    quads, quad_mats, boxes, box_mats):
    t_min = 1e30
    nx, ny, nz = 0.0, 0.0, 0.0
    mat = numba.int32(-1)

    for qi in range(quads.shape[0]):
        t = intersect_quad(ox, oy, oz, dx, dy, dz, quads[qi])
        if 0.0 < t < t_min:
            t_min = t
            axis = numba.int32(quads[qi][0])
            nx, ny, nz = 0.0, 0.0, 0.0
            if axis == 0:
                nx = quads[qi][6]
            elif axis == 1:
                ny = quads[qi][6]
            else:
                nz = quads[qi][6]
            mat = quad_mats[qi]

    for bi in range(boxes.shape[0]):
        b = boxes[bi]
        t, near_axis, near_sign = intersect_aabb(
            ox, oy, oz, dx, dy, dz,
            b[0], b[1], b[2], b[3], b[4], b[5])
        if 0.0 < t < t_min:
            t_min = t
            nx, ny, nz = 0.0, 0.0, 0.0
            if near_axis == 0:
                nx = near_sign
            elif near_axis == 1:
                ny = near_sign
            else:
                nz = near_sign
            mat = box_mats[bi]

    return t_min, nx, ny, nz, mat


@njit(cache=True)
def cosine_hemisphere(nx, ny, nz):
    r = math.sqrt(np.random.random())
    phi = 2.0 * math.pi * np.random.random()
    lx = r * math.cos(phi)
    ly = r * math.sin(phi)
    lz = math.sqrt(max(0.0, 1.0 - lx * lx - ly * ly))

    # Tangent frame
    if abs(nx) > 0.9:
        ux, uy, uz = 0.0, 1.0, 0.0
    else:
        ux, uy, uz = 1.0, 0.0, 0.0

    tx = uy * nz - uz * ny
    ty = uz * nx - ux * nz
    tz = ux * ny - uy * nx
    inv_len = 1.0 / math.sqrt(tx * tx + ty * ty + tz * tz)
    tx *= inv_len
    ty *= inv_len
    tz *= inv_len

    bx = ny * tz - nz * ty
    by = nz * tx - nx * tz
    bz = nx * ty - ny * tx

    return (tx * lx + bx * ly + nx * lz,
            ty * lx + by * ly + ny * lz,
            tz * lx + bz * ly + nz * lz)


@njit(cache=True)
def sample_light_point(quads, light_idx):
    q = quads[light_idx]
    axis = numba.int32(q[0])
    pos = q[1]
    a, b = _quad_perp(axis)

    a_val = q[2] + np.random.random() * (q[3] - q[2])
    b_val = q[4] + np.random.random() * (q[5] - q[4])

    p = [0.0, 0.0, 0.0]
    p[axis] = pos
    p[a] = a_val
    p[b] = b_val

    n = [0.0, 0.0, 0.0]
    n[axis] = q[6]

    area = (q[3] - q[2]) * (q[5] - q[4])
    return p[0], p[1], p[2], n[0], n[1], n[2], area


@njit(cache=True)
def trace_path(ox, oy, oz, dx, dy, dz,
               quads, quad_mats, boxes, box_mats,
               materials, light_idx, max_bounces):
    AMBIENT = 0.05
    rad_r, rad_g, rad_b = 0.0, 0.0, 0.0
    thr_r, thr_g, thr_b = 1.0, 1.0, 1.0
    EPS = 1e-5

    for bounce in range(max_bounces + 1):
        t, nx, ny, nz, mat_idx = intersect_scene(
            ox, oy, oz, dx, dy, dz,
            quads, quad_mats, boxes, box_mats)

        if mat_idx < 0:
            break

        hx = ox + t * dx
        hy = oy + t * dy
        hz = oz + t * dz

        albr, albg, albb = materials[mat_idx, 0], materials[mat_idx, 1], materials[mat_idx, 2]
        emr, emg, emb = materials[mat_idx, 3], materials[mat_idx, 4], materials[mat_idx, 5]

        # Emission only on camera ray (avoid double-counting with NEE)
        if bounce == 0:
            rad_r += thr_r * emr
            rad_g += thr_g * emg
            rad_b += thr_b * emb

        # Ambient fill
        rad_r += thr_r * albr * AMBIENT
        rad_g += thr_g * albg * AMBIENT
        rad_b += thr_b * albb * AMBIENT

        if bounce >= max_bounces:
            break

        # Ensure normal faces incoming ray
        if nx * dx + ny * dy + nz * dz > 0:
            nx, ny, nz = -nx, -ny, -nz

        # NEE: direct light sampling
        lpx, lpy, lpz, lnx, lny, lnz, larea = sample_light_point(quads, light_idx)

        tlx = lpx - hx
        tly = lpy - hy
        tlz = lpz - hz
        dist_l = math.sqrt(tlx * tlx + tly * tly + tlz * tlz)

        if dist_l > 1e-10:
            inv_d = 1.0 / dist_l
            tlx *= inv_d
            tly *= inv_d
            tlz *= inv_d

            cos_i = tlx * nx + tly * ny + tlz * nz
            cos_l = -(tlx * lnx + tly * lny + tlz * lnz)

            if cos_i > 0 and cos_l > 0:
                st, _, _, _, smat = intersect_scene(
                    hx + nx * EPS, hy + ny * EPS, hz + nz * EPS,
                    tlx, tly, tlz,
                    quads, quad_mats, boxes, box_mats)

                if st >= dist_l - 2 * EPS:
                    lmat = quad_mats[light_idx]
                    lem_r = materials[lmat, 3]
                    lem_g = materials[lmat, 4]
                    lem_b = materials[lmat, 5]

                    geom = cos_i * cos_l / (dist_l * dist_l)
                    inv_pi = 1.0 / math.pi

                    rad_r += thr_r * albr * inv_pi * lem_r * geom * larea
                    rad_g += thr_g * albg * inv_pi * lem_g * geom * larea
                    rad_b += thr_b * albb * inv_pi * lem_b * geom * larea

        # Continue path: cosine-weighted hemisphere
        ndx, ndy, ndz = cosine_hemisphere(nx, ny, nz)

        # Throughput update: BRDF * cos / pdf = albedo (for cosine-weighted)
        thr_r *= albr
        thr_g *= albg
        thr_b *= albb

        ox = hx + nx * EPS
        oy = hy + ny * EPS
        oz = hz + nz * EPS
        dx, dy, dz = ndx, ndy, ndz

    return rad_r, rad_g, rad_b


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Image / Light Field Rendering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@njit(parallel=True, cache=True)
def render_image(cam_pos, cam_right, cam_up, cam_fwd, half_fov,
                 width, height, spp,
                 quads, quad_mats, boxes, box_mats, materials,
                 light_idx, max_bounces):
    image = np.zeros((height, width, 3))

    for pixel_idx in prange(height * width):
        i = pixel_idx // width
        j = pixel_idx % width

        acc_r, acc_g, acc_b = 0.0, 0.0, 0.0
        for s in range(spp):
            px = (2.0 * (j + np.random.random()) / width - 1.0) * half_fov
            py = (1.0 - 2.0 * (i + np.random.random()) / height) * half_fov

            dx = cam_fwd[0] + px * cam_right[0] + py * cam_up[0]
            dy = cam_fwd[1] + px * cam_right[1] + py * cam_up[1]
            dz = cam_fwd[2] + px * cam_right[2] + py * cam_up[2]
            inv_len = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
            dx *= inv_len
            dy *= inv_len
            dz *= inv_len

            r, g, b = trace_path(
                cam_pos[0], cam_pos[1], cam_pos[2],
                dx, dy, dz,
                quads, quad_mats, boxes, box_mats,
                materials, light_idx, max_bounces)
            acc_r += r
            acc_g += g
            acc_b += b

        inv_spp = 1.0 / spp
        image[i, j, 0] = acc_r * inv_spp
        image[i, j, 1] = acc_g * inv_spp
        image[i, j, 2] = acc_b * inv_spp

    return image


def make_camera(position, target):
    fwd = target - position
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up


def render_lightfield(n_angular, angular_extent, cam_z, look_at, fov_deg,
                      width, height, spp, max_bounces,
                      quads, quad_mats, boxes, box_mats, materials,
                      light_idx):
    half_fov = math.tan(math.radians(fov_deg / 2))
    angular_pos = np.linspace(-angular_extent / 2, angular_extent / 2, n_angular)
    lightfield = np.zeros((n_angular, n_angular, height, width, 3))

    total = n_angular * n_angular
    with tqdm(total=total, desc="  Views", unit="img") as pbar:
        for vi, cy in enumerate(angular_pos):
            for ui, cx in enumerate(angular_pos):
                cam_pos = np.array([cx, cy, cam_z])
                fwd, right, up = make_camera(cam_pos, look_at)

                img = render_image(
                    cam_pos, right, up, fwd, half_fov,
                    width, height, spp,
                    quads, quad_mats, boxes, box_mats, materials,
                    light_idx, max_bounces)
                lightfield[vi, ui] = img
                pbar.update(1)

    return lightfield, angular_pos


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("Cornell Box Light Field Renderer")
    print("=" * 60)

    quads, quad_mats, boxes, box_mats, light_idx = build_scene()
    print(f"  Scene: {quads.shape[0]} quads, {boxes.shape[0]} AABBs")

    # Parameters
    n_angular = 9
    angular_extent = 0.02   # 20mm eyebox
    cam_z = 0.2             # 200mm
    look_at = np.array([0.0, 0.0, Z0])
    fov_deg = 20.0
    width = height = 1024
    spp = 256
    max_bounces = 2

    print(f"  Light field: {n_angular}×{n_angular} views, {width}×{height} px, "
          f"{spp} spp, {max_bounces} bounces")

    # JIT warmup
    print("  JIT compiling...", end="", flush=True)
    fwd, right, up = make_camera(np.array([0.0, 0.0, cam_z]), look_at)
    half_fov = math.tan(math.radians(fov_deg / 2))
    _ = render_image(
        np.array([0.0, 0.0, cam_z]), right, up, fwd, half_fov,
        4, 4, 1,
        quads, quad_mats, boxes, box_mats, MATERIALS,
        light_idx, max_bounces)
    print(" done")

    # Render
    t0 = time.time()
    lightfield, angular_pos = render_lightfield(
        n_angular, angular_extent, cam_z, look_at, fov_deg,
        width, height, spp, max_bounces,
        quads, quad_mats, boxes, box_mats, MATERIALS, light_idx)
    dt = time.time() - t0
    print(f"\n  Total: {dt:.1f}s ({dt / 60:.1f} min)")

    # ── Save light field ──
    out_npz = PLOT_DIR / "cornell_lightfield.npz"
    np.savez_compressed(str(out_npz),
                        lightfield=lightfield,
                        angular_positions=angular_pos,
                        cam_z=cam_z,
                        fov_deg=fov_deg)
    print(f"✅ Saved: {out_npz}")

    # ── Tone mapping helper ──
    def tonemap(img):
        img = np.clip(img, 0, None)
        # Exposure + gamma
        img = 1.0 - np.exp(-img * 2.0)  # simple filmic
        return np.clip(img, 0, 1)

    # ── Plot 1: Center view ──
    center = n_angular // 2
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(tonemap(lightfield[center, center]))
    ax.set_title(f"Cornell Box — Center View ({spp} spp, {max_bounces} bounces)")
    ax.axis('off')
    out1 = PLOT_DIR / "cornell_lf_center.png"
    fig.savefig(str(out1), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out1}")

    # ── Plot 2: Grid of views ──
    fig, axes = plt.subplots(n_angular, n_angular,
                             figsize=(2 * n_angular, 2 * n_angular))
    for vi in range(n_angular):
        for ui in range(n_angular):
            ax = axes[vi, ui]
            ax.imshow(tonemap(lightfield[vi, ui]))
            ax.axis('off')
    plt.suptitle(f"Cornell Box Light Field ({n_angular}×{n_angular})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out2 = PLOT_DIR / "cornell_lf_grid.png"
    fig.savefig(str(out2), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out2}")

    # ── Plot 3: Horizontal parallax strip ──
    fig, axes = plt.subplots(1, n_angular, figsize=(2.5 * n_angular, 3))
    for ui in range(n_angular):
        ax = axes[ui]
        ax.imshow(tonemap(lightfield[center, ui]))
        ax.set_title(f"x={angular_pos[ui]*1e3:+.1f}mm", fontsize=9)
        ax.axis('off')
    plt.suptitle("Horizontal Parallax Strip (center row)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out3 = PLOT_DIR / "cornell_lf_parallax.png"
    fig.savefig(str(out3), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out3}")


if __name__ == "__main__":
    main()

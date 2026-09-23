"""Look at the design: meridional section with ray fans, spot diagrams, and
achieved r(theta) vs the R = 6 target. Saved as PNGs and read back with the
Read tool before anything is reported done (project rule: never report
visual work from code alone)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import optics as op  # noqa: E402
import design_opt as d  # noqa: E402
import build_design as bd  # noqa: E402
import foveation_target as ft  # noqa: E402

DTYPE = op.DTYPE


def load(ckpt, n_elements=d.N_ELEMENTS):
    params = d.Params(n_elements)
    params.load_state_dict(torch.load(ckpt))
    return params


def section_figure(params, aperture_mm, out_path, n_theta_rays=7, n_pupil_rays=5):
    with torch.no_grad():
        elements, image_y = params.build(aperture_mm)
    fig, ax = plt.subplots(figsize=(11, 7))

    # Surfaces: sag profile z in [-aperture, aperture] at each element, using
    # EACH element's own physical clear aperture (not the training-time
    # domain-safety aperture, which is shared and larger, and would plot the
    # frozen extrapolated sag well past where the design actually uses it).
    z = torch.linspace(-1.0, 1.0, 400, dtype=DTYPE)
    for el in elements:
        elem_ap = bd.element_aperture_mm(float(el.front.vertex_y))
        for surf in (el.front, el.back):
            r_local = z.abs() * elem_ap
            y = surf.vertex_y + op.sag(r_local, surf)
            ax.plot((y).numpy(), (z * elem_ap).numpy(), "b-", lw=1.2)

    # Image plane
    ax.axhline(0, color="gray", lw=0.3)
    ax.plot([float(image_y), float(image_y)], [-5, 5], "g-", lw=2, label="MLA plane")

    # Ray fans: theta grid, pupil grid (meridional plane x=0 slice: use pupil z-offset only)
    thetas = np.radians(np.linspace(0.0, d.EDGE_ECC_DEG, n_theta_rays))
    pupil_z = np.linspace(-d.PUPIL_R, d.PUPIL_R, n_pupil_rays)
    cmap = plt.cm.viridis(np.linspace(0, 1, n_theta_rays))
    for ti, th in enumerate(thetas):
        origins = np.stack([np.zeros_like(pupil_z), np.full_like(pupil_z, d.PUPIL_Y), pupil_z], axis=1)
        dirs = np.tile([0.0, np.cos(th), np.sin(th)], (n_pupil_rays, 1))
        o = torch.tensor(origins, dtype=DTYPE)
        di = torch.tensor(dirs, dtype=DTYPE)
        pts = [o.numpy()]
        cur_o, cur_d = o, di
        for el in elements:
            for surf, n_out in ((el.front, el.index), (el.back, 1.0)):
                n_in = 1.0 if surf is el.front else el.index
                t, p, n = op.surface_point_normal(cur_o, cur_d, surf)
                cur_d, tir, _ = op.refract(cur_d, n, n_in, n_out)
                cur_o = p
                pts.append(cur_o.numpy())
        img = op.Surface.flat(op.as_t(image_y), 1e9)
        t = op.intersect(cur_o, cur_d, img)
        hit = (cur_o + t[:, None] * cur_d).numpy()
        pts.append(hit)
        pts = np.stack(pts, axis=1)  # (n_pupil_rays, n_surf+2, 3)
        for r in range(n_pupil_rays):
            ax.plot(pts[r, :, 1], pts[r, :, 2], "-", color=cmap[ti], lw=0.6,
                   label=f"{np.degrees(th):.1f} deg" if r == 0 else None)

    ax.set_xlabel("world Y (mm, optical axis)")
    ax.set_ylabel("world Z (mm)")
    ax.set_title("Coaxial remapper: meridional section with pupil ray fans")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")
    # A ray that goes TIR/dead at the field edge can fly off to a huge,
    # physically meaningless (x, z) at the (infinite) image plane; clip the
    # view to the system's real extent so it doesn't swamp the autoscale.
    ax.set_xlim(d.PUPIL_Y - 5, float(image_y) + 15)
    ax.set_ylim(-1.3 * aperture_mm, 1.3 * aperture_mm)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def mapping_and_spots_figure(params, aperture_mm, out_path, n_theta=30):
    theta, target_r = d.r_theta_grid(n_theta, d.EDGE_ECC_DEG)
    pupil = d.pupil_samples()
    (hit, out_dir, alive, clean, margins, forward_y, t_margins,
     elements, image_y) = d.trace_all(params, theta, pupil, aperture_mm)
    x, z = hit[..., 0].detach().numpy(), hit[..., 2].detach().numpy()
    alive_np, clean_np = alive.numpy(), clean.numpy()
    chief_x, chief_z = x[:, 0], z[:, 0]
    theta_np, target_np = theta.numpy(), target_r.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(np.degrees(theta_np), target_np, "k-", label="target r(theta)")
    ax.plot(np.degrees(theta_np), np.hypot(chief_x, chief_z), "r.", label="achieved (chief ray)")
    ax.set_xlabel("field eccentricity (deg)")
    ax.set_ylabel("panel radius (mm)")
    ax.set_title("Achieved vs target mapping")
    ax.legend()

    ax = axes[1]
    usable = alive_np & clean_np
    frac_usable = usable.mean(axis=1)
    ax.plot(np.degrees(theta_np), frac_usable, "b.-")
    ax.set_xlabel("field eccentricity (deg)")
    ax.set_ylabel("fraction of pupil rays alive & TIR-free")
    ax.set_title("Ray survival across the pupil")
    ax.set_ylim(-0.05, 1.05)

    ax = axes[2]
    pick = np.linspace(0, n_theta - 1, 6).astype(int)
    for i in pick:
        m = usable[i]
        dx = (x[i, m] - 0.0) * 1000
        dz = (z[i, m] - target_np[i]) * 1000
        ax.scatter(dx, dz, s=6, label=f"{np.degrees(theta_np[i]):.1f} deg")
    ax.set_xlabel("x offset from target (um)")
    ax.set_ylabel("z offset from target (um)")
    ax.set_title("Spot diagrams (relative to target point)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return theta_np, target_np, chief_x, chief_z, usable


if __name__ == "__main__":
    ckpt = sys.argv[1]
    fig_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[0] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    params = load(ckpt)
    aperture_mm = bd.element_aperture_mm(d.Y0) + 20.0
    section_figure(params, aperture_mm, fig_dir / "section.png")
    mapping_and_spots_figure(params, aperture_mm, fig_dir / "mapping_and_spots.png")
    print("wrote", fig_dir / "section.png", fig_dir / "mapping_and_spots.png")

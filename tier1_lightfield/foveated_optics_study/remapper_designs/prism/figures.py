"""Diagnostic PNGs for the freeform TIR prism: a y-z section with ray fans,
spot diagrams on the MLA plane, achieved-vs-target sampling density, and a
TIR/exit margin map over the field. Run after design.py has produced
params.npz. All figures are saved next to this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design import Design, mapping_grid, pupil_ring, field_direction, PUPIL_Y_MM  # noqa: E402
import raytrace as rt  # noqa: E402
import foveation_target as ft  # noqa: E402

HERE = Path(__file__).resolve().parent
DTYPE = rt.DTYPE


def load(params_path):
    z = np.load(params_path)
    d = Design(index=float(z["index"]), requires_grad=False)
    d.pose = torch.tensor(z["pose"], dtype=DTYPE)
    d.coeff = torch.tensor(z["coeff"], dtype=DTYPE)
    return d


def section_figure(design, out=HERE / "fig_section.png"):
    s1, s2, s3 = design.surfaces()
    fig, ax = plt.subplots(figsize=(8, 8))
    for s, name, color in [(s1, "S1", "tab:blue"), (s2, "S2 (mirror)", "tab:red"), (s3, "S3", "tab:green")]:
        u = torch.linspace(-30.0, 30.0, 400, dtype=DTYPE)
        x0 = torch.zeros_like(u)
        p = s.point(x0, u).detach().numpy()
        ax.plot(p[:, 1], p[:, 2], color=color, label=name, lw=2)

    pupil = torch.tensor(pupil_ring(n=7), dtype=DTYPE)
    fields = [(0, 0), (35, 0), (-35, 0), (0, 22.5), (0, -22.5), (20, 15)]
    cmap = plt.get_cmap("plasma")
    prism = design.prism()
    for k, (tx, tz) in enumerate(fields):
        d = field_direction(torch.tensor([float(tx)]), torch.tensor([float(tz)]))
        d = d.expand(len(pupil), 3)
        px, pz = pupil[:, 0], pupil[:, 1]
        origin = torch.stack([px, torch.full_like(px, PUPIL_Y_MM), pz], dim=-1)
        out_ = prism.trace(origin, d, iters=10)
        color = cmap(k / max(1, len(fields) - 1))
        for i in range(len(pupil)):
            path = np.stack([origin[i].numpy(),
                             out_["p1"][i].numpy(), out_["p2"][i].numpy(),
                             out_["p3"][i].numpy(), out_["p4"][i].numpy()])
            valid = bool(out_["valid"][i])
            ax.plot(path[:, 1], path[:, 2], color=color, lw=0.6, alpha=0.9 if valid else 0.15,
                   linestyle="-" if valid else ":")
        ax.plot([], [], color=color, label=f"field ({tx},{tz})")

    ax.scatter([PUPIL_Y_MM], [0.0], color="k", marker="o", s=30, zorder=5, label="pupil centre")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("Prism section (x=0) with pupil-ring ray fans")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def mapping_and_margin_figures(design, out_map=HERE / "fig_mapping.png",
                               out_margin=HERE / "fig_margin.png", out_spot=HERE / "fig_spots.png"):
    tx, tz = mapping_grid()
    ecc_deg = np.hypot(tx, tz)
    ecc = np.radians(ecc_deg)
    target_r = ft.panel_radius_mm(ecc)

    pupil = torch.tensor(pupil_ring(n=7), dtype=DTYPE)
    uv, out = design.trace_to_panel(torch.tensor(tx, dtype=DTYPE), torch.tensor(tz, dtype=DTYPE), pupil, iters=12)
    r_ach = uv[:, 0, :].norm(dim=-1).detach().numpy()
    centroid = uv.mean(dim=1, keepdim=True)
    spot_rms_um = ((uv - centroid).pow(2).sum(-1).mean(dim=1).sqrt()).detach().numpy() * 1000.0
    valid = out["valid"].reshape(len(tx), len(pupil)).detach().numpy()

    # mapping: achieved landing radius vs target, against eccentricity
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ecc_deg, r_ach, s=6, c="tab:blue", label="achieved (this design)", alpha=0.6)
    ecc_line = np.linspace(0, 35, 200)
    ax.plot(ecc_line, ft.panel_radius_mm(np.radians(ecc_line)), "k--", label="R=6 target")
    ax.set_xlabel("field eccentricity (deg)")
    ax.set_ylabel("landing radius on MLA plane (mm)")
    ax.set_title("Achieved panel mapping vs the R=6 foveation target")
    ax.axhline(4.088, color="gray", lw=0.5, ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_map, dpi=150)
    plt.close(fig)
    print(f"wrote {out_map}")

    # margin map
    n_g = design.index
    import math
    crit = math.degrees(math.asin(1.0 / n_g))
    inc3 = out["inc3_deg"].reshape(len(tx), len(pupil)).detach().numpy()
    inc4 = out["inc4_deg"].reshape(len(tx), len(pupil)).detach().numpy()
    margin_tir = (inc3 - crit).min(axis=1)
    margin_exit = (crit - inc4).min(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, m, title in [(axes[0], margin_tir, "TIR margin at S1 (2nd hit), deg"),
                         (axes[1], margin_exit, "Exit refraction margin at S3, deg")]:
        sc = ax.scatter(tx, tz, c=m, cmap="RdYlGn", vmin=-2, vmax=10, s=40)
        ax.set_xlabel("theta_x (deg)")
        ax.set_ylabel("theta_z (deg)")
        ax.set_title(title)
        fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(out_margin, dpi=150)
    plt.close(fig)
    print(f"wrote {out_margin}")

    # spot size and validity map
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sc = axes[0].scatter(tx, tz, c=np.clip(spot_rms_um, 0, 200), cmap="viridis", s=40)
    axes[0].set_title("Spot RMS on MLA plane (um, clipped at 200)")
    fig.colorbar(sc, ax=axes[0])
    frac_valid = valid.mean(axis=1)
    sc2 = axes[1].scatter(tx, tz, c=frac_valid, cmap="RdYlGn", vmin=0, vmax=1, s=40)
    axes[1].set_title("Fraction of pupil rays that trace validly")
    fig.colorbar(sc2, ax=axes[1])
    for ax in axes:
        ax.set_xlabel("theta_x (deg)")
        ax.set_ylabel("theta_z (deg)")
    fig.tight_layout()
    fig.savefig(out_spot, dpi=150)
    plt.close(fig)
    print(f"wrote {out_spot}")


if __name__ == "__main__":
    params = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "params.npz")
    d = load(params)
    section_figure(d)
    mapping_and_margin_figures(d)

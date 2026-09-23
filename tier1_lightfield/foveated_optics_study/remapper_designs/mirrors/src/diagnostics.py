"""PNG diagnostics for the mirror-remapper design: a y-z layout section with
ray fans, spot diagrams at the panel, and the achieved vs target sampling
density. Uses only the torch ray tracer (no Blender)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from system import field_grid, hex_pupil_samples, loss_and_metrics, make_mirror, panel_frame, trace  # noqa: E402
import foveation_target as ft  # noqa: E402

PUPIL_Y_MM = -3.6


def layout_section(params, apertures, path, n_fields=9):
    """y-z section (x=0 plane): mirror profiles + ray fans for fields along
    tz at tx=0, plus the full-aperture x=0 fan."""
    fig, ax = plt.subplots(figsize=(9, 8))
    (ax1, ay1), (ax2, ay2) = apertures
    m1, m2 = make_mirror(params["m1"]), make_mirror(params["m2"])
    for mirror, ay, color, label in [(m1, ay1, "tab:blue", "M1"), (m2, ay2, "tab:orange", "M2")]:
        y_local = torch.linspace(-ay, ay, 200, dtype=torch.float64)
        x_local = torch.zeros_like(y_local)
        with torch.no_grad():
            s = mirror.sag(x_local, y_local)
            P = mirror.to_world(x_local, y_local, s)
        ax.plot(P[:, 1].numpy(), P[:, 2].numpy(), color=color, lw=2, label=f"{label} (x=0 profile)")

    u_p, v_p, w_p = panel_frame(params["panel"])
    origin = params["panel"]["origin"].detach().numpy()
    half = ft.HALF_PANEL_MM
    seg = np.stack([origin - half * v_p.detach().numpy(), origin + half * v_p.detach().numpy()])
    ax.plot(seg[:, 1], seg[:, 2], color="tab:green", lw=3, label="panel (x=0 edge)")

    tz = np.radians(np.linspace(-22.5, 22.5, n_fields))
    tx = np.zeros_like(tz)
    pupil = np.array([[0.0, 0.0]])
    with torch.no_grad():
        out = trace(params, tx, tz, pupil, aperture=apertures)
    P0 = np.array([-3.6, 0.0])
    ax.scatter([PUPIL_Y_MM], [0.0], color="black", zorder=5, label="pupil centre")
    for f in range(n_fields):
        if not out["valid"][f, 0]:
            continue
        pts = np.stack([[PUPIL_Y_MM, 0.0], out["P1"][f, 0, 1:].numpy(), out["P2"][f, 0, 1:].numpy(),
                        out["P3"][f, 0, 1:].numpy()])
        ax.plot(pts[:, 0], pts[:, 1], color="crimson", lw=0.6, alpha=0.8)
    ax.set_xlabel("world Y (mm)")
    ax.set_ylabel("world Z (mm)")
    ax.set_title("y-z section (x=0): mirror profiles, panel, and a tz ray fan (tx=0)")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def spot_diagrams(params, apertures, path, nx=6, nz=5, pupil_spacing=0.4):
    tx, tz = field_grid(nx, nz)
    pupil = hex_pupil_samples(pupil_spacing)
    with torch.no_grad():
        out = trace(params, tx, tz, pupil, aperture=apertures)
    fig, axes = plt.subplots(nz, nx, figsize=(2.0 * nx, 2.0 * nz), squeeze=False)
    for f in range(len(tx)):
        j, i = f // nx, f % nx
        a = axes[nz - 1 - j][i]
        v = out["valid"][f].numpy()
        u, w = out["up"][f].numpy(), out["vp"][f].numpy()
        a.scatter(u[v] * 1000, w[v] * 1000, s=4, color="tab:blue")
        a.scatter(u[~v] * 1000, w[~v] * 1000, s=4, color="lightgray")
        a.set_title(f"{np.degrees(tx[f]):.0f},{np.degrees(tz[f]):.0f}", fontsize=7)
        a.tick_params(labelsize=5)
        a.axhline(0, color="gray", lw=0.3)
        a.axvline(0, color="gray", lw=0.3)
    fig.suptitle("Spot diagrams at the panel (um), one per field (tx, tz) deg")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sampling_map(params, apertures, path, nx=29, nz=21, pupil_spacing=0.6):
    tx, tz = field_grid(nx, nz)
    pupil = hex_pupil_samples(pupil_spacing)
    with torch.no_grad():
        loss, metrics, out = loss_and_metrics(params, tx, tz, pupil, aperture=apertures)
    valid = metrics["frac_valid"].numpy() > 0.5
    cu, cv = metrics["cu"].numpy(), metrics["cv"].numpy()
    tu, tv = metrics["tu"].numpy(), metrics["tv"].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    sc = ax.scatter(cu[valid], cv[valid], c=np.degrees(np.hypot(tx, tz))[valid], s=10, cmap="viridis")
    ax.scatter(tu[valid], tv[valid], marker="x", s=6, color="red", alpha=0.4)
    plt.colorbar(sc, ax=ax, label="eccentricity (deg, approx)")
    ax.set_title("Achieved (dots) vs target (x) panel positions (mm)")
    ax.axis("equal")

    ax = axes[1]
    pos_err = metrics["pos_err_mm"].numpy()
    sc = ax.scatter(np.degrees(tx), np.degrees(tz), c=np.where(valid, pos_err, np.nan), s=25, cmap="magma")
    plt.colorbar(sc, ax=ax, label="position error (mm)")
    ax.set_title("Chief-ray position error vs target")
    ax.set_xlabel("tx (deg)")
    ax.set_ylabel("tz (deg)")

    ax = axes[2]
    spot = metrics["spot_rms_mm"].numpy()
    sc = ax.scatter(np.degrees(tx), np.degrees(tz), c=np.where(valid, spot, np.nan), s=25, cmap="magma")
    plt.colorbar(sc, ax=ax, label="spot RMS (mm)")
    ax.set_title("Spot size over the field")
    ax.set_xlabel("tx (deg)")
    ax.set_ylabel("tz (deg)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

"""Initial two-mirror off-axis fold layout, plane-symmetric about world x = 0.

The whole system is built from the y-z (meridional) plane only: M1 folds the
eye's forward axis up and out of its own field of view, M2 catches that beam
and bends it down onto the flat MLA plane. The 2-D (y, z) geometry below is
solved with the ordinary law of reflection to get everything (M2's position
and tilt, the panel tilt) consistent with one chosen central-ray path; it is
only a STARTING GUESS for the gradient-based optimiser in optimize.py, which
owns every one of these numbers afterwards (positions, tilts, curvatures, and
the freeform polynomial terms).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DTYPE = torch.float64
PUPIL_Y_MM = -3.6

# XY-polynomial terms, even in local x (plane symmetry), total order 2..6.
TERMS = [(i, j) for i in (0, 2, 4, 6) for j in range(0, 7 - i) if i + j >= 2]


def _reflect2d(d, n):
    return d - 2.0 * np.dot(d, n) * n


def _normal_from_fold(d_in, d_out):
    n = d_in - d_out
    return n / np.linalg.norm(n)


def _tilt_from_normal(n_yz):
    """n = (-sin a, cos a) in (y, z) -> a."""
    return float(np.arctan2(-n_yz[0], n_yz[1]))


@dataclass
class InitLayout:
    V1: np.ndarray
    a1: float
    V2: np.ndarray
    a2: float
    panel_origin: np.ndarray
    panel_tilt: float
    c1: float = 0.0
    c2: float = 0.0


def _paraxial_curvatures(d12, d2p, f1):
    """Two concave mirrors, thin-mirror equation applied along the folded
    axis: M1 focuses the collimated (infinity) pupil bundle at f1 behind
    itself; M2 relays that (real or virtual) intermediate image onto the
    panel at distance d2p. Returns (c1, c2) = (1/R1, 1/R2) = (1/2f1, 1/2f2),
    a paraxial STARTING GUESS only -- the optimiser owns both afterwards."""
    so2 = f1 - d12
    f2 = 1.0 / (1.0 / so2 + 1.0 / d2p)
    return 1.0 / (2.0 * f1), 1.0 / (2.0 * f2)


def solve_initial_layout(a1_deg=35.0, y1=22.0, d12=16.0, panel_target=(14.0, -9.0), f1=35.0):
    """a1_deg: M1's tilt. y1: M1 vertex y (x=z=0). d12: M1-to-M2 vertex
    distance along the folded central ray. panel_target: (y, z) the central
    (on-axis) ray must reach -- this becomes the panel origin (on-axis lens,
    panel radius 0)."""
    a1 = np.radians(a1_deg)
    d0 = np.array([1.0, 0.0])  # (y, z) of world (0, 1, 0)
    n1 = np.array([-np.sin(a1), np.cos(a1)])
    d1 = _reflect2d(d0, n1)
    V1 = np.array([y1, 0.0])
    V2 = V1 + d12 * d1

    target = np.array(panel_target)
    d2 = (target - V2)
    d2 = d2 / np.linalg.norm(d2)
    n2 = _normal_from_fold(d1, d2)
    a2 = _tilt_from_normal(n2)

    w_p = -d2
    a_p = _tilt_from_normal(w_p)

    d2p = np.linalg.norm(target - V2)
    c1, c2 = _paraxial_curvatures(d12, d2p, f1)

    return InitLayout(V1=V1, a1=a1, V2=V2, a2=a2, panel_origin=target, panel_tilt=a_p, c1=c1, c2=c2)


def build_params(layout: InitLayout, aperture, seed=0):
    """Free torch parameters for the two mirrors and the panel pose, all
    requires_grad. Freeform coefficients start at 0 (pure conic fold).
    `aperture`: ((half_x1, half_y1), (half_x2, half_y2)) mm, used only to
    normalise the polynomial terms (see surfaces.Freeform) -- fixed, not
    learned."""
    g = torch.Generator().manual_seed(seed)
    (ax1, ay1), (ax2, ay2) = aperture

    def mirror_params(V_yz, a, c0, u_scale, v_scale):
        V = torch.tensor([0.0, V_yz[0], V_yz[1]], dtype=DTYPE, requires_grad=True)
        tilt = torch.tensor(a, dtype=DTYPE, requires_grad=True)
        c = torch.tensor(c0, dtype=DTYPE, requires_grad=True)
        k = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
        coeffs = torch.zeros(len(TERMS), dtype=DTYPE, requires_grad=True)
        return dict(V=V, tilt=tilt, c=c, k=k, coeffs=coeffs, u_scale=u_scale, v_scale=v_scale)

    # Start from the paraxial curvatures that already focus a collimated
    # on-axis bundle onto the panel (see _paraxial_curvatures), so the
    # optimiser begins near a focused system instead of having to discover
    # optical power from scratch under competing distortion-matching
    # pressure -- a flat (c=0) start was observed to plateau with a ~0.3-0.4
    # mm spot (60x the ~5 um target) after thousands of Adam iterations.
    m1 = mirror_params(layout.V1, layout.a1, layout.c1, ax1, ay1)
    m2 = mirror_params(layout.V2, layout.a2, layout.c2, ax2, ay2)
    panel = dict(
        origin=torch.tensor([0.0, layout.panel_origin[0], layout.panel_origin[1]],
                            dtype=DTYPE, requires_grad=True),
        tilt=torch.tensor(layout.panel_tilt, dtype=DTYPE, requires_grad=True),
    )
    focal_um = torch.tensor(45.0, dtype=DTYPE, requires_grad=True)
    return dict(m1=m1, m2=m2, panel=panel, focal_um=focal_um)


def all_leaves(params):
    leaves = []
    for grp in ("m1", "m2"):
        leaves += [params[grp][k] for k in ("V", "tilt", "c", "k", "coeffs")]
    leaves += [params["panel"]["origin"], params["panel"]["tilt"], params["focal_um"]]
    return leaves

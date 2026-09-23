"""Optimise a coaxial (rotationally symmetric) remapper against the R = 6
foveation target, by differentiable ray tracing in torch float64.

Because the system is rotationally symmetric, the field can be swept along a
single meridian (az = 90 deg, i.e. direction (0, cos theta, sin theta)): by
symmetry every other azimuth behaves identically for the same eccentricity,
so the full 2-D pupil disc at that one meridian already exercises meridional
AND skew (sagittal) rays for every field azimuth. See NOTES.md.

Per (theta, pupil sample) the loss is a robust (sqrt, not squared) distance
in mm from the landing point on the image plane (which will hold the MLA's
flat face) to the target point (0, panel_radius_mm(theta)). Squared error was
tried first and is unstable: a ray that lands far off (from a near-critical-
angle refraction early in training) then dominates the Adam step. sqrt-error
keeps every ray's gradient contribution bounded. NOTES.md records this.

sin2t (how close each refraction sits to total internal reflection) is a
smooth, differentiable quantity even though the refract/reflect *choice* is
not; penalising it directly gives the optimiser an anticipatory gradient
away from TIR, instead of only finding out via a discontinuous jump once a
ray actually goes total.

The field target itself is far gentler than it first looks: local_focal_mm
is flat at 5.6488 mm beyond about 8 deg eccentricity (see NOTES.md), so 80%
of the field is a plain constant-power f-theta mapping and only the central
fovea needs the full 6x power ramp -- this is the shape the optimiser has to
find, not an arbitrary one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import optics as op  # noqa: E402
import foveation_target as ft  # noqa: E402

DTYPE = op.DTYPE
torch.set_default_dtype(DTYPE)

PUPIL_Y = -3.6
PUPIL_R = 2.0
EDGE_ECC_DEG = 41.6852  # diagonal corner: atan(hypot(tan 35, tan 22.5))
N_ELEMENTS = 3
# PMMA first (low index -> wide critical angle, safe for the big-field front
# element), then N-BK7, then polycarbonate close to the image where ray
# angles are small and a higher index buys correction power cheaply.
INDICES = [1.49, 1.517, 1.585]
Y0 = 18.0  # first surface vertex, mm (eye relief margin above the 16.4 mm floor)
C_SCALE, K_SCALE, A4_SCALE, A6_SCALE = 0.008, 1.5, 2e-5, 2e-7


def r_theta_grid(n, max_ecc_deg=EDGE_ECC_DEG):
    r_max = float(ft.panel_radius_mm(np.radians(max_ecc_deg)))
    r = np.linspace(0.0, r_max, n)
    r[0] = 1e-4
    theta = ft.eccentricity_rad(r)
    return torch.tensor(theta, dtype=DTYPE), torch.tensor(ft.panel_radius_mm(theta), dtype=DTYPE)


def pupil_samples(rings=((0.7, 6), (1.3, 8), (2.0, 10))):
    pts = [(0.0, 0.0)]
    for rp, n in rings:
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
            pts.append((rp * np.cos(a), rp * np.sin(a)))
    return torch.tensor(pts, dtype=DTYPE)  # (P, 2)


class Params(torch.nn.Module):
    """Raw, unconstrained parameters -> physical surface prescriptions.
    Scaled so a unit change in each raw parameter is a comparable physical
    change, which keeps a single Adam learning rate sane across all of them.
    """

    def __init__(self, n_elements):
        super().__init__()
        self.n = n_elements
        z = lambda: torch.nn.Parameter(torch.zeros(n_elements, dtype=DTYPE))  # noqa: E731
        self.c_f, self.c_b = z(), z()
        self.k_f, self.k_b = z(), z()
        self.a4_f, self.a4_b = z(), z()
        self.a6_f, self.a6_b = z(), z()
        self.th_raw = torch.nn.Parameter(torch.zeros(n_elements, dtype=DTYPE))
        self.gap_raw = torch.nn.Parameter(torch.zeros(n_elements, dtype=DTYPE))  # gap AFTER element i

    def build(self, aperture_mm):
        sp = torch.nn.functional.softplus
        th = 3.0 + sp(self.th_raw)
        # Generous floors: a curved surface's sag at the full clear aperture
        # can be several mm (see NOTES.md), so a thin nominal air gap can be
        # physically eaten by sag at large radius even though the on-axis
        # gap looks fine -- that showed up as marginal rays going "dead"
        # (Newton finding no forward intersection) with no smooth gradient
        # pointing back out, until the floor was raised here.
        gaps = list(8.0 + sp(self.gap_raw[:-1])) if self.n > 1 else []
        gaps.append(15.0 + sp(self.gap_raw[-1]))  # back focal distance to the MLA
        y = torch.tensor(Y0, dtype=DTYPE)
        elements = []
        for i in range(self.n):
            front = op.Surface(y, self.c_f[i] * C_SCALE, self.k_f[i] * K_SCALE,
                               self.a4_f[i] * A4_SCALE, self.a6_f[i] * A6_SCALE,
                               torch.zeros((), dtype=DTYPE), aperture_mm)
            y_back = y + th[i]
            back = op.Surface(y_back, self.c_b[i] * C_SCALE, self.k_b[i] * K_SCALE,
                              self.a4_b[i] * A4_SCALE, self.a6_b[i] * A6_SCALE,
                              torch.zeros((), dtype=DTYPE), aperture_mm)
            elements.append(op.Element(front, back, INDICES[i]))
            y = y_back + gaps[i]
        return elements, y  # y is now the image (MLA flat-face) plane


def trace_all(params: Params, theta, pupil, aperture_mm):
    elements, image_y = params.build(aperture_mm)
    n_t, n_p = theta.shape[0], pupil.shape[0]
    dirs = torch.stack([torch.zeros_like(theta), torch.cos(theta), torch.sin(theta)], dim=-1)
    dirs = dirs[:, None, :].expand(n_t, n_p, 3).reshape(-1, 3)
    origins = torch.zeros(n_t, n_p, 3, dtype=DTYPE)
    origins[..., 0] = pupil[None, :, 0]
    origins[..., 1] = PUPIL_Y
    origins[..., 2] = pupil[None, :, 1]
    origins = origins.reshape(-1, 3)
    hit, out_dir, alive, clean, margins, forward_y, t_margins = op.trace_system(origins, dirs, elements, image_y)
    hit = hit.reshape(n_t, n_p, 3)
    out_dir = out_dir.reshape(n_t, n_p, 3)
    alive = alive.reshape(n_t, n_p)
    clean = clean.reshape(n_t, n_p)
    margins = [m.reshape(n_t, n_p) for m in margins]
    forward_y = [f.reshape(n_t, n_p) for f in forward_y]
    t_margins = [tm.reshape(n_t, n_p) for tm in t_margins]
    return hit, out_dir, alive, clean, margins, forward_y, t_margins, elements, image_y


def required_aperture_mm(vertex_y, max_ecc_deg=EDGE_ECC_DEG, margin_mm=3.0):
    """The clear aperture a surface at this vertex_y actually needs to cover
    the full field from the pupil (same formula build_design.py uses to
    mesh the final solid -- kept in sync deliberately, see domain_pen)."""
    return (vertex_y - PUPIL_Y) * np.tan(np.radians(max_ecc_deg + 3.0)) + PUPIL_R + margin_mm


def domain_penalty(elements, domain_margin=0.6):
    """A strong conic/curvature can make an even asphere's sag formula hit
    its own sqrt domain edge well inside the aperture the field actually
    needs (see optics.py's _safe_sphere_r and NOTES.md): the ray tracer
    safely freezes the sag there, but a MESHED, EXPORTED solid must not --
    it would either flatten unphysically or (observed) wind inconsistently
    once the cap stops being a simple graph. This never showed up in the
    ray-traced loss, because no traced ray needs the full aperture: it is a
    packaging constraint, checked here directly against each surface's own
    domain limit, sqrt(domain_margin / ((1+k) c^2)), compared with the
    aperture the surface must physically span."""
    pen = 0.0
    for el in elements:
        for surf in (el.front, el.back):
            need = required_aperture_mm(surf.vertex_y)
            u = (1.0 + surf.k) * surf.c ** 2
            # domain_limit^2 = domain_margin / u; want domain_limit >= need,
            # i.e. u <= domain_margin / need^2. Penalise the excess of u.
            pen = pen + torch.relu(u - domain_margin / need ** 2) * need ** 2
    return pen / (2 * len(elements))


def loss_fn(params, theta, target_r, pupil, aperture_mm, tele_weight=0.05, tir_weight=0.3, fwd_weight=0.3,
           tmargin_weight=0.1, spot_weight=1.0, domain_weight=0.0):
    (hit, out_dir, alive, clean, margins, forward_y, t_margins,
     elements, image_y) = trace_all(params, theta, pupil, aperture_mm)
    x, z = hit[..., 0], hit[..., 2]
    err = torch.sqrt((x - 0.0) ** 2 + (z - target_r[:, None]) ** 2 + 1e-10)  # mm, robust (not squared)
    # A ray that went through total internal reflection lands at an arbitrary,
    # unbounded position (see optics.trace_system docstring): including it in
    # the position loss is pure numerical noise, not a useful gradient, so it
    # is masked out here. TIR avoidance itself comes from tir_pen below, which
    # is smooth and anticipates TIR before it happens.
    usable = (clean & alive).float()
    n_usable = usable.sum().clamp(min=1.0)
    spot_loss = (err * usable).sum() / n_usable

    chief = 0  # index of the (0,0) pupil sample
    chief_angle = torch.atan2(torch.sqrt(out_dir[:, chief, 0] ** 2 + out_dir[:, chief, 2] ** 2),
                              out_dir[:, chief, 1])
    tele_loss = (chief_angle ** 2).mean()

    # Anticipatory, smooth TIR-avoidance: penalise sin2t once it passes 0.85
    # (about a 20 deg margin from the critical angle), well before it hits 1.
    tir_pen = sum(torch.relu(m - 0.85).pow(2).mean() for m in margins) / len(margins)
    # Same idea for a ray turning backward (d_y <= 0, which silently produces
    # a dead ray with zero loss gradient): penalise d_y once it drops below
    # 0.5 (60 deg from the axis), well before it actually goes non-forward.
    fwd_pen = sum(torch.relu(0.5 - f).pow(2).mean() for f in forward_y) / len(forward_y)
    # Same idea again for the actual quantity that flips `alive` false: the
    # per-surface Newton solution t (mm) going non-positive. A ray can go
    # dead at an intermediate surface (e.g. sag eating an air gap at large
    # radius) while still looking "forward enough" in direction, so this is
    # a materially different signal from fwd_pen, not a duplicate of it.
    # Linear (not squared): t can be very negative (tens of mm) once a ray
    # is badly lost, and a squared penalty there was observed to explode
    # (loss > 1e4, Adam's second-moment EMA gets poisoned by one huge
    # gradient and the optimiser stops responding for many iterations
    # afterward -- see NOTES.md). Linear keeps the gradient magnitude
    # bounded regardless of how far t has gone negative.
    tmargin_pen = sum(torch.relu(1.0 - tm).mean() for tm in t_margins) / len(t_margins)

    reg = sum((p ** 2).mean() for p in (params.a4_f, params.a4_b, params.a6_f, params.a6_b))
    dead_loss = (~alive).float().mean()
    tir_loss = (~clean).float().mean()
    dom_pen = domain_penalty(elements) if domain_weight > 0 else torch.zeros((), dtype=DTYPE)
    total = (spot_weight * spot_loss + tele_weight * tele_loss + tir_weight * tir_pen + fwd_weight * fwd_pen
            + tmargin_weight * tmargin_pen + domain_weight * dom_pen
            + 1e-3 * reg + 2.0 * dead_loss + 0.5 * tir_loss)
    return total, {
        "spot_rms_um": float(((err.detach() * usable).sum() / n_usable)) * 1e3,
        "tele_deg": float(chief_angle.detach().abs().mean()) * 180 / np.pi,
        "dead": float(dead_loss.detach()), "tir": float(tir_loss.detach()), "tir_pen": float(tir_pen.detach()),
        "fwd_pen": float(fwd_pen.detach()), "tmargin_pen": float(tmargin_pen.detach()),
        "dom_pen": float(dom_pen.detach()),
    }


def _freeze(params, names, requires_grad):
    for name in names:
        getattr(params, name).requires_grad_(requires_grad)


ALL_SHAPE = ["c_f", "c_b", "k_f", "k_b", "a4_f", "a4_b", "a6_f", "a6_b"]


def run_stage(params, theta, target_r, pupil, aperture_mm, n_iters, lr, free_params, log_every, tag,
             loss_kwargs=None):
    _freeze(params, ALL_SHAPE + ["th_raw", "gap_raw"], False)
    _freeze(params, free_params, True)
    opt = torch.optim.Adam([p for p in params.parameters() if p.requires_grad], lr=lr)
    warm = 30
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / warm))
    loss_kwargs = loss_kwargs or {}
    for it in range(n_iters):
        opt.zero_grad()
        loss, info = loss_fn(params, theta, target_r, pupil, aperture_mm, **loss_kwargs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in params.parameters() if p.requires_grad], 0.5)
        opt.step()
        sched.step()
        if it % log_every == 0 or it == n_iters - 1:
            print(f"[{tag}] it {it:5d}  loss {float(loss.detach()):.5f}  spot_rms_um {info['spot_rms_um']:8.2f}  "
                  f"tele_deg {info['tele_deg']:6.3f}  dead {info['dead']:.4f}  tir {info['tir']:.4f}  "
                  f"tir_pen {info['tir_pen']:.5f}  fwd_pen {info['fwd_pen']:.5f}  "
                  f"tmargin_pen {info['tmargin_pen']:.5f}  dom_pen {info['dom_pen']:.5f}")
    return params


def optimise(n_theta=20, seed=0, n_elements=N_ELEMENTS, log_every=100,
            field_curriculum=(10.0, 16.0, 22.0, 28.0, 34.0, EDGE_ECC_DEG)):
    """Grow the field range gradually: a bad early curvature exposed straight
    away to 41.6 deg rays goes TIR immediately and the optimiser gets no
    useful gradient there (see NOTES.md); starting narrow and widening lets
    each new field increment only need a small correction to an already-sane
    design, instead of a wide jump from a bad joint optimum."""
    torch.manual_seed(seed)
    chief_pupil = pupil_samples(rings=())  # just (0, 0)
    full_pupil = pupil_samples()
    aperture_mm = (Y0 - PUPIL_Y) * np.tan(np.radians(EDGE_ECC_DEG + 3)) + PUPIL_R + 3.0

    params = Params(n_elements)
    with torch.no_grad():
        params.c_f[0], params.c_b[0] = -0.4, -0.3
        if n_elements > 1:
            params.c_f[1:], params.c_b[1:] = 0.15, 0.1

    theta0, target_r0 = r_theta_grid(n_theta, field_curriculum[0])
    run_stage(params, theta0, target_r0, chief_pupil, aperture_mm, 400, 0.02,
             ["c_f", "c_b", "th_raw", "gap_raw"], log_every, f"seed-chief-{field_curriculum[0]:.0f}deg")

    for stage_i, max_ecc in enumerate(field_curriculum):
        theta, target_r = r_theta_grid(n_theta, max_ecc)
        run_stage(params, theta, target_r, full_pupil, aperture_mm, 300, 0.012,
                 ["c_f", "c_b", "th_raw", "gap_raw"], log_every, f"grow{stage_i}-sph-{max_ecc:.0f}deg")
        run_stage(params, theta, target_r, full_pupil, aperture_mm, 300, 0.008,
                 ["c_f", "c_b", "k_f", "k_b", "th_raw", "gap_raw"], log_every,
                 f"grow{stage_i}-conic-{max_ecc:.0f}deg")
        run_stage(params, theta, target_r, full_pupil, aperture_mm, 300, 0.006,
                 ALL_SHAPE + ["th_raw", "gap_raw"], log_every, f"grow{stage_i}-asph-{max_ecc:.0f}deg")

    theta_full, target_full = r_theta_grid(n_theta, EDGE_ECC_DEG)
    return params, aperture_mm, theta_full, target_full, full_pupil


def optimise_feasibility_first(n_theta=20, seed=0, n_elements=N_ELEMENTS, log_every=200, ckpt_dir=None):
    """The field-growth curriculum (optimise()) still degrades badly once the
    field passes about 20 deg: dead+TIR fraction climbs past 50% and does
    not recover, because the spot-position loss keeps demanding curvature
    strong enough to fit the fovea, and that curvature is what causes TIR
    and backward-turning rays at the edge of a 41.6 deg field (see NOTES.md).

    This is the opposite curriculum: fix the FULL field range from the
    start, start from a deliberately gentle (weak curvature) seed, and first
    optimise almost PURELY for feasibility (no TIR, no dead/backward rays,
    everywhere in the field) with the position loss barely weighted in, then
    anneal toward the normal (position-dominant) loss once every ray is
    forward-going and TIR-free -- so the position fit only ever has to work
    with curvatures that are already known to be geometrically safe.
    """
    torch.manual_seed(seed)
    full_pupil = pupil_samples()
    aperture_mm = (Y0 - PUPIL_Y) * np.tan(np.radians(EDGE_ECC_DEG + 3)) + PUPIL_R + 3.0
    theta, target_r = r_theta_grid(n_theta, EDGE_ECC_DEG)

    params = Params(n_elements)
    with torch.no_grad():
        params.c_f[0], params.c_b[0] = -0.15, -0.1
        if n_elements > 1:
            params.c_f[1:], params.c_b[1:] = 0.05, 0.03

    # The regulariser weights stay HIGH throughout (this differs from an
    # earlier version of this curriculum, which decayed them while ramping
    # spot_weight up -- that let the design backslide into TIR/dead-ray
    # infeasibility every time, because the position loss alone keeps
    # pulling curvature toward the same regime that causes TIR at the field
    # edge; see NOTES.md). Only spot_weight is ramped.
    def checkpoint(tag):
        if ckpt_dir is not None:
            torch.save(params.state_dict(), Path(ckpt_dir) / f"{tag}.pt")

    feas = dict(tele_weight=0.0, tir_weight=3.0, fwd_weight=3.0, tmargin_weight=1.5)
    run_stage(params, theta, target_r, full_pupil, aperture_mm, 500, 0.02,
             ["c_f", "c_b", "th_raw", "gap_raw"], log_every, "feas-sph",
             loss_kwargs={**feas, "spot_weight": 0.02})
    checkpoint("feas-sph")
    run_stage(params, theta, target_r, full_pupil, aperture_mm, 500, 0.015,
             ["c_f", "c_b", "k_f", "k_b", "th_raw", "gap_raw"], log_every, "feas-conic",
             loss_kwargs={**feas, "spot_weight": 0.05})
    checkpoint("feas-conic")

    hold = dict(tele_weight=0.02, tir_weight=3.0, fwd_weight=3.0, tmargin_weight=1.5)
    run_stage(params, theta, target_r, full_pupil, aperture_mm, 500, 0.01,
             ALL_SHAPE + ["th_raw", "gap_raw"], log_every, "hold-asph-0.15",
             loss_kwargs={**hold, "spot_weight": 0.15})
    checkpoint("hold-asph-0.15")
    run_stage(params, theta, target_r, full_pupil, aperture_mm, 500, 0.008,
             ALL_SHAPE + ["th_raw", "gap_raw"], log_every, "hold-asph-0.4",
             loss_kwargs={**hold, "spot_weight": 0.4})
    checkpoint("hold-asph-0.4")
    run_stage(params, theta, target_r, full_pupil, aperture_mm, 800, 0.006,
             ALL_SHAPE + ["th_raw", "gap_raw"], log_every, "hold-asph-0.8",
             loss_kwargs={**hold, "spot_weight": 0.8})
    checkpoint("hold-asph-0.8")

    return params, aperture_mm, theta, target_r, full_pupil


def continue_with_domain_safety(ckpt_path, n_theta=20, n_elements=N_ELEMENTS, log_every=100, ckpt_dir=None):
    """Warm-start from an existing checkpoint and add domain_penalty: the
    ray-traced losses alone left one surface's conic so strong that its own
    valid domain fell well inside the aperture the field actually needs (see
    domain_penalty's docstring and NOTES.md) -- the exported solid then had
    to be meshed at a much smaller aperture than the design intended, and
    the real Cycles evaluator showed the result: coverage 0.0002, throughput
    6e-6, almost total vignetting. This does NOT re-run the whole curriculum;
    it fine-tunes the already-feasible, already-reasonably-focused design to
    also respect the aperture it will actually be built at.
    """
    params = Params(n_elements)
    params.load_state_dict(torch.load(ckpt_path))
    full_pupil = pupil_samples()
    aperture_mm = (Y0 - PUPIL_Y) * np.tan(np.radians(EDGE_ECC_DEG + 3)) + PUPIL_R + 3.0
    theta, target_r = r_theta_grid(n_theta, EDGE_ECC_DEG)

    def checkpoint(tag):
        if ckpt_dir is not None:
            torch.save(params.state_dict(), Path(ckpt_dir) / f"{tag}.pt")

    kw = dict(tele_weight=0.05, tir_weight=1.0, fwd_weight=1.0, tmargin_weight=0.5, spot_weight=0.6)
    run_stage(params, theta, target_r, full_pupil, aperture_mm, 400, 0.006,
             ALL_SHAPE + ["th_raw", "gap_raw"], log_every, "domsafe-0.3",
             loss_kwargs={**kw, "domain_weight": 0.3})
    checkpoint("domsafe-0.3")
    run_stage(params, theta, target_r, full_pupil, aperture_mm, 400, 0.005,
             ALL_SHAPE + ["th_raw", "gap_raw"], log_every, "domsafe-0.6",
             loss_kwargs={**kw, "domain_weight": 0.6, "spot_weight": 0.8})
    checkpoint("domsafe-0.6")
    run_stage(params, theta, target_r, full_pupil, aperture_mm, 500, 0.004,
             ALL_SHAPE + ["th_raw", "gap_raw"], log_every, "domsafe-1.0",
             loss_kwargs={**kw, "domain_weight": 1.0, "spot_weight": 1.0})
    checkpoint("domsafe-1.0")
    return params, aperture_mm, theta, target_r, full_pupil


if __name__ == "__main__":
    optimise_feasibility_first()

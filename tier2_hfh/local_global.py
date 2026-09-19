"""
Local/global phase retrieval for hogel-free holography: a Gerchberg-Saxton /
Projective-Dynamics-style solver built on the exactly-diagonal normal
operator in `tier2_hfh/operators.py`.

THE ITERATION

For a single frame, continuous phase, one wavelength, each round is four
steps:

  1. `y_k = P_k u`                             -- render every pupil view.
  2. local:  `z_k = a_k * y_k / |y_k|`          -- project onto the target
             amplitude circle, per view, independently (`project_onto_amplitude`).
  3. global: `u* = solve_normal(sum_k rho_k P_k^H z_k)`
             -- reconcile the (generally incompatible) per-view projections
             by least squares. Because `ViewOperators`'s normal operator is
             exactly diagonal, this is an elementwise division, not an
             iterative solve (`operators.py` proves the diagonal claim).
  4. panel:  `u = u* / |u*|`                    -- re-impose the phase-only,
             unit-amplitude modulator constraint (`panel_projection`, itself
             `project_onto_amplitude` with amplitude 1).

Steps 2 and 4 are the same operation -- Euclidean projection onto a circle
of fixed radius, centred at the origin -- applied to different targets
(measured views vs. the panel's own modulator), so both are implemented by
one function.

WHY THIS SPLIT IS EXACT, NOT APPROXIMATE

The projection in step 2/4 is the exact minimiser of `|z - y|^2` subject to
`|z| = a`, for `y != 0` (proved symbolically in
`tests/test_local_global.py` via a Lagrange-multiplier argument checked with
sympy, not merely asserted). The division in step 3 is exact because the
normal operator has no off-diagonal terms at all -- not because it is being
approximated by one. Neither step introduces a tolerance; the only genuine
indeterminacy in the whole iteration is the `y = 0` / `u* = 0` corner
handled below.

THE ZERO CASE IS NOT A NUMERICAL COARSENING

When `|y_k| = 0` (or `|u*| = 0`), every point on the target circle is
exactly `a` away from the origin, so the minimiser of step 2/4 is genuinely
non-unique -- not "numerically ill-conditioned", but tied on the nose. Two
explicit, mutually exclusive policies are offered through `zero_policy`
(see `project_onto_amplitude` for the full argument):

  "zero"            (default) -- leave the value at exactly 0. Matches the
                    minimum-norm convention `ViewOperators.solve_normal`
                    already uses for its own dead-pixel case, injects no
                    phase the data did not determine, and is a fixed point.

  "reference_phase" -- place the value at `a * exp(i * reference_phase)`,
                    a fixed, caller-chosen phase (0 radians by default).
                    Actually lies on the constraint circle, at the cost of
                    injecting a phase with no support in the measurement.

Silently substituting `y / (|y| + eps)` is deliberately not offered: it
would pick an arbitrary phase too (whatever direction rounding noise left
in `y`) while pretending, via the epsilon, that no choice had been made.

ENERGY MONITORING, AND WHAT WAS ACTUALLY OBSERVED

`solve` records, once per iteration, the coupling objective

    L(u) = sum_k rho_k * sum_{pixels} ( |P_k u| - a_k )^2

(`energy`) at the three points in the cycle where `u` exists as a panel
field: "before" the iteration, "after_global" (post least-squares, pre
panel projection), and "after_panel" (post phase-only projection). The
local step (2) does not produce a panel field to evaluate `L` at -- it
lives entirely in view space -- so it has no corresponding entry; its own
effect is that it drives each view's residual to exactly zero individually
(subject to the zero policy), which is not the same thing as driving `L`
of any achievable `u` to zero, since the per-view targets `z_k` are
generally mutually incompatible on overlapping windows.

Measured on the test geometries here: `before -> after_global` is
monotone non-increasing every time observed (the global step is a
least-squares projection onto the achievable subspace of the very set of
targets whose distance IS `L`, so it cannot move `L` uphill measured
against those same targets). `after_global -> after_panel` is NOT
monotone -- the panel projection changes `u`'s amplitude back to 1
independently at every pixel, which is unrelated to what makes `L` small,
and it was observed to increase `L` on some iterations of the realisable-
target test in `tests/test_local_global.py`. Both facts are pinned by
tests rather than asserted only here.

A PLAIN GERCHBERG-SAXTON BASELINE

`gerchberg_saxton` is the "obvious multi-view generalisation" the brief
asks for: instead of reconciling all views by least squares, it cycles
through the views one at a time and, for each, directly inverts that one
view's own FFT relation to overwrite its window of the panel (the window
transform is invertible view-by-view; only the presence of several
*overlapping* windows makes a joint solve necessary, and plain GS declines
to do one). Where two windows overlap, whichever view was processed last
in a sweep wins that overlap and partially undoes what the earlier view in
the same sweep achieved -- unlike `solve`, which reconciles all views
simultaneously. This is why it exists purely as a weaker point of
comparison, per the brief, and not as a competing solver.
"""

import cmath

import torch

from tier2_hfh.operators import ViewOperators


def _complex_dtype(real_dtype):
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def project_onto_amplitude(y, amplitude, zero_policy="zero", reference_phase=0.0):
    """
    Elementwise `argmin_{|z| = amplitude} |z - y|^2`.

    For `y != 0`: `z = amplitude * y / |y|`. Proved (not merely stated) in
    `tests/test_local_global.py` via sympy: minimising `|z-y|^2` subject to
    `z1^2+z2^2=amplitude^2` by Lagrange multipliers has exactly two
    stationary points, `z = +-amplitude*y/|y|`, and the `+` one is the
    strictly smaller of the two whenever `amplitude>0` and `y!=0` -- see the
    module docstring above for the corner where that stops holding.

    For `y == 0`: see `zero_policy` in the module docstring. `"zero"` (the
    default) returns 0; `"reference_phase"` returns
    `amplitude * exp(i*reference_phase)`. The two agree everywhere except at
    this measure-zero locus.

    `amplitude` broadcasts against `y`; it is typically per-view-and-pixel
    (`a_k`, shape `(views, window, window)`) for the local step, or the
    scalar `1.0` for the panel step.
    """
    if zero_policy not in ("zero", "reference_phase"):
        raise ValueError(f"unknown zero_policy: {zero_policy!r}")

    real_dtype = y.real.dtype
    amplitude = torch.as_tensor(amplitude, dtype=real_dtype, device=y.device)

    mag = y.abs()
    is_zero = mag == 0
    safe_mag = torch.where(is_zero, torch.ones_like(mag), mag)
    projected = y * (amplitude / safe_mag).to(y.dtype)

    if zero_policy == "zero":
        fallback = torch.zeros((), dtype=y.dtype, device=y.device)
    else:
        phase_factor = cmath.exp(1j * reference_phase)
        fallback = amplitude.to(y.dtype) * phase_factor

    return torch.where(is_zero, fallback, projected)


def panel_projection(u_star, zero_policy="zero", reference_phase=0.0):
    """Phase-only, unit-amplitude panel constraint: `u = u* / |u*|`."""
    return project_onto_amplitude(u_star, 1.0, zero_policy=zero_policy,
                                   reference_phase=reference_phase)


def energy(op, u, amplitude, weights=None):
    """
    `L(u) = sum_k rho_k * sum_pixels ( |P_k u| - a_k )^2`.

    Returned as a python float since this is a diagnostic, never a value
    fed back into the optimisation.
    """
    y = op.forward(u)
    residual = y.abs() - amplitude.to(y.real.dtype)
    per_view = (residual ** 2).sum(dim=(-2, -1))
    if weights is not None:
        per_view = per_view * weights.to(per_view.dtype)
    return float(per_view.sum())


def local_step(op, u, amplitude, zero_policy="zero"):
    """`y_k = P_k u`, then project each view onto its target amplitude."""
    y = op.forward(u)
    return project_onto_amplitude(y, amplitude, zero_policy=zero_policy)


def global_step(op, z, weights=None):
    """`u* = solve_normal(sum_k rho_k P_k^H z_k)`."""
    zw = z if weights is None else z * weights.to(z.dtype)[:, None, None]
    rhs = op.adjoint(zw)
    return op.solve_normal(rhs, weights)


def solve(op, amplitude, iters=50, weights=None, u0=None,
          zero_policy="zero", reference_phase=0.0):
    """
    Run the local/global/panel cycle `iters` times.

    `u0` defaults to a uniform, zero-phase panel (`exp(i*0) = 1` everywhere)
    -- deliberately not a random default, so a run is reproducible without
    the caller having to thread a generator through this function only to
    immediately discard it.

    Returns `(u, history)`. `history` is a list of length `iters`, one dict
    per iteration with keys `"before"`, `"after_global"`, `"after_panel"`
    -- the objective `energy(op, ., amplitude, weights)` at the three points
    in the cycle described in the module docstring.
    """
    real_dtype = op.geom.dtype
    if u0 is None:
        u = torch.ones(op.panel, op.panel, dtype=_complex_dtype(real_dtype),
                        device=op.geom.device)
    else:
        u = u0.clone()

    history = []
    for _ in range(iters):
        before = energy(op, u, amplitude, weights)
        z = local_step(op, u, amplitude, zero_policy=zero_policy)
        u_star = global_step(op, z, weights)
        after_global = energy(op, u_star, amplitude, weights)
        u = panel_projection(u_star, zero_policy=zero_policy,
                              reference_phase=reference_phase)
        after_panel = energy(op, u, amplitude, weights)
        history.append({"before": before, "after_global": after_global,
                         "after_panel": after_panel})
    return u, history


def gerchberg_saxton(op, amplitude, iters=50, u0=None, zero_policy="zero",
                      reference_phase=0.0):
    """
    Plain, single-view-at-a-time Gerchberg-Saxton (see the module docstring
    for how this differs from `solve`).

    Each sweep visits every view in offset order and, for each, inverts its
    own windowed-FFT relation directly:

        u[window_k] <- ifft(ishift(z_k)) / aperture   where aperture != 0
                        (left unchanged where aperture == 0, since a zero
                         aperture pixel is invisible to this view and this
                         view's measurement carries no information about it)

    then, once every view in the sweep has been visited, re-imposes the
    panel's own phase-only constraint (`panel_projection`) exactly as
    `solve` does. `amplitude` has shape `(views, window, window)`, one
    target amplitude map per pupil, matching `op.offsets`'s order.
    """
    w = op.window
    real_dtype = op.geom.dtype
    amplitude = torch.as_tensor(amplitude, dtype=real_dtype, device=op.geom.device)

    if u0 is None:
        u = torch.ones(op.panel, op.panel, dtype=_complex_dtype(real_dtype),
                        device=op.geom.device)
    else:
        u = u0.clone()

    aperture = op.aperture
    aperture_nonzero = aperture != 0
    safe_aperture = torch.where(aperture_nonzero, aperture, torch.ones_like(aperture))

    history = []
    for _ in range(iters):
        for k, (oy, ox) in enumerate(op.offsets.tolist()):
            tile = u[oy:oy + w, ox:ox + w]
            y = torch.fft.fftshift(torch.fft.fft2(tile * aperture))
            z = project_onto_amplitude(y, amplitude[k], zero_policy=zero_policy,
                                        reference_phase=reference_phase)
            back = torch.fft.ifft2(torch.fft.ifftshift(z)) / safe_aperture.to(z.dtype)
            u[oy:oy + w, ox:ox + w] = torch.where(aperture_nonzero, back, tile)
        u = panel_projection(u, zero_policy=zero_policy, reference_phase=reference_phase)
        history.append(energy(op, u, amplitude, weights=None))
    return u, history

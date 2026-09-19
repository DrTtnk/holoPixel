"""
Proximal variants of the local/global solver's two blocks.

See `docs/notes_kl_convergence.md` for the theorem this module targets --
Attouch, Bolte, Redont, Soubeyran (2010), "Proximal alternating minimization
and projection methods for nonconvex problems: an approach based on the
Kurdyka-Lojasiewicz inequality" (arXiv:0801.1780) -- the per-hypothesis check
against this algorithm, and the numerical evidence gathered with the
functions below. This module only supplies the two closed forms and the
objective bookkeeping the theorem's hypotheses need; it does not itself
prove anything.

THE TWO BLOCKS

Write `x=(u,z)`, `L(u,z) = f(u) + Q(u,z) + g(z)` with

    Q(u,z) = sum_k rho_k ||P_k u - z_k||^2            (`operators.py`'s coupling)
    f(u)   = indicator of {u : |u_j|=1 for every panel pixel j}
    g(z)   = indicator of {z : |z_{k,s}|=a_{k,s} for every (view,pixel) (k,s)}

`local_global.solve`'s "global step, then panel step" pair, composed, is
already exactly `argmin_{|u|=1} L(u,z)` -- because `H=sum_k rho_k P_k^H P_k`
is diagonal, `d_j|u_j|^2` is constant on `|u_j|=1`, so normalising the
unconstrained least-squares solution IS the constrained minimiser (see
`operators.py` and the notes' "Established fact 2"). So the existing solver
is already a (non-proximal) two-block alternating minimisation of this same
`L`; this module adds a genuine proximal term to each block, matching the
scheme the theorem analyses:

    z_{t+1} in argmin_v { g(v)   + Q(u_t, v)     + (1/(2*lam)) ||v-z_t||^2 }
    u_{t+1} in argmin_w { f(w)   + Q(w, z_{t+1}) + (1/(2*mu))  ||w-u_t||^2 }

Iteration order is z-then-u, matching `local_global.solve`'s own order, not
the u-then-z order the paper's (x,y) happens to be written in -- see the
notes for the relabelling table (`x_theirs = z_ours`, `y_theirs = u_ours`).

THE TWO CLOSED FORMS, RE-DERIVED IN sympy, NOT ASSUMED

Both are proved to be the exact, strict constrained minimiser (not merely a
stationary point) in `tests/test_proximal.py`, by the same Lagrange-
multiplier-plus-branch-comparison method `test_local_global.py` already
uses for the non-proximal projection:

    u_j = (b_j + eta*u_old_j) / |b_j + eta*u_old_j|,    eta   = 1/(2*mu)
    z   = a*(y + gamma*z_old) / |y + gamma*z_old|,      gamma = 1/(2*lam*rho_k)

where `b = sum_k rho_k P_k^H z_k` (the same right-hand side `global_step`
already computes) and `y = P_k u` (the same forward map `local_step` already
computes). Both match the candidate forms given in the task description
exactly -- the sympy re-derivation found no missing factor and no sign
error.

`eta` does not depend on the (diagonal) normal-operator entry `d_j`: the
`d_j|u_j|^2` term is a constant on `|u_j|=1`, so the constrained minimiser
of the proximal-regularised objective is governed only by the linear and
proximal terms. This is Established Fact 2 surviving the addition of a
proximal term, re-derived rather than assumed to survive it (the sympy
script subtracts the two candidate branches' objective values and confirms
the difference is independent of `d`).

`gamma` carries a `1/rho_k` because the per-view weight scales `Q`'s term
but not the proximal term: minimising `rho_k*A + C*B` over the same variable
has the same argmin as minimising `A + (C/rho_k)*B` for `rho_k>0`. With
uniform weights (`rho_k=1`, `local_global.solve`'s default) this reduces to
the single shared constant `gamma=1/(2*lam)`.

BOTH FORMULAS ARE `project_onto_amplitude` OF A SHIFTED TARGET

`(b+eta*u_old)/|b+eta*u_old|` is exactly `project_onto_amplitude(b+eta*u_old,
1.0)`, and `a*(y+gamma*z_old)/|y+gamma*z_old|` is exactly
`project_onto_amplitude(y+gamma*z_old, a)`. Both proximal steps below are
therefore implemented by reusing that already-proved function (including its
explicit, tested zero-locus policy) on a shifted argument, rather than by a
second, parallel projection implementation.
"""

import torch

from tier2_hfh.local_global import project_onto_amplitude


def _complex_dtype(real_dtype):
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def proximal_global_step(op, z, u_old, eta, weights=None, zero_policy="zero",
                          reference_phase=0.0):
    """
    The u-block proximal step.

    Exact `argmin_{|u_j|=1 for every j}` of
    `sum_k rho_k ||P_k u - z_k||^2 + eta * ||u - u_old||^2`, closed form
    re-derived (not merely stated) in `tests/test_proximal.py`. `eta>0` is a
    single scalar shared by every pixel -- see the module docstring for why
    it needs no per-pixel scaling by the (diagonal) normal-operator entries.
    """
    zw = z if weights is None else z * weights.to(z.dtype)[:, None, None]
    b = op.adjoint(zw)
    target = b + eta * u_old
    return project_onto_amplitude(target, 1.0, zero_policy=zero_policy,
                                   reference_phase=reference_phase)


def proximal_local_step(op, u, amplitude, z_old, gamma, weights=None,
                         zero_policy="zero", reference_phase=0.0):
    """
    The z-block proximal step.

    Exact `argmin_{|z_{k,s}|=a_{k,s}}` of, per view `k`,
    `rho_k * ||P_k u - z_k||^2 + (1/(2*lam)) * ||z_k - z_old_k||^2`, which
    (see the module docstring) is the same as minimising
    `||P_k u - z_k||^2 + (gamma/rho_k) * ||z_k - z_old_k||^2` for
    `gamma = 1/(2*lam)`. `weights=None` means `rho_k=1` for every view, the
    same convention `global_step`/`local_step` already use.
    """
    y = op.forward(u)
    if weights is None:
        shift = gamma * z_old
    else:
        rho = weights.to(y.real.dtype)[:, None, None]
        shift = (gamma / rho).to(z_old.dtype) * z_old
    target = y + shift
    return project_onto_amplitude(target, amplitude, zero_policy=zero_policy,
                                   reference_phase=reference_phase)


def coupling_energy(op, u, z, weights=None):
    """
    `Q(u,z) = sum_k rho_k ||P_k u - z_k||^2`, i.e. `L(u,z)` restricted to the
    feasible set (`f(u)=g(z)=0` there, since the proximal closed forms above
    always land exactly on the constraint by construction).

    This is deliberately NOT the same quantity as `local_global.energy`,
    which measures `|P_k u| - a_k` (amplitude-only, phase-blind) rather than
    `P_k u - z_k` (the actual coupling residual the KL theorem's sufficient-
    decrease inequality is about). Returned as a python float, a diagnostic
    only, matching `local_global.energy`'s own convention.
    """
    y = op.forward(u)
    residual = (y - z).abs() ** 2
    per_view = residual.sum(dim=(-2, -1))
    if weights is not None:
        per_view = per_view * weights.to(per_view.dtype)
    return float(per_view.sum())


def proximal_solve(op, amplitude, iters=50, eta=0.1, gamma=0.1, weights=None,
                    u0=None, z0=None, zero_policy="zero", reference_phase=0.0):
    """
    Run the proximal alternating scheme (z-block, then u-block) `iters`
    times, matching the order `local_global.solve` already uses.

    `u0` defaults to the same uniform, zero-phase panel `solve` defaults to.
    `z0` defaults to the (non-proximal) projection of `P_k u0` onto the
    target amplitudes, i.e. what the first local step of `local_global.solve`
    itself would produce -- there is no proximal history before the first
    round, so there is nothing for a proximal term to regularise towards yet.

    Returns `(u, z, history)`. `history[t]` records, using `L := coupling_energy`:
      `"L"`      -- `L(u_t, z_t)` before this round's two updates,
      `"L_next"` -- `L(u_{t+1}, z_{t+1})` after them,
      `"step"`   -- `||z_{t+1}-z_t||^2 + ||u_{t+1}-u_t||^2`,
    the three quantities the sufficient-decrease inequality
    `L(x_t) - L(x_{t+1}) >= c * ||x_{t+1}-x_t||^2` relates -- see
    `tests/test_proximal.py` for the measured constant `c` and
    `docs/notes_kl_convergence.md` for where that constant comes from
    (`c = min(eta, gamma)` with uniform weights, a direct consequence of each
    step being an exact minimiser of its own proximal-regularised
    subproblem, needing no KL property to hold).
    """
    real_dtype = op.geom.dtype
    cdtype = _complex_dtype(real_dtype)
    if u0 is None:
        u = torch.ones(op.panel, op.panel, dtype=cdtype, device=op.geom.device)
    else:
        u = u0.clone()
    if z0 is None:
        z = project_onto_amplitude(op.forward(u), amplitude, zero_policy=zero_policy,
                                    reference_phase=reference_phase)
    else:
        z = z0.clone()

    history = []
    for _ in range(iters):
        l_before = coupling_energy(op, u, z, weights)
        z_new = proximal_local_step(op, u, amplitude, z, gamma, weights=weights,
                                     zero_policy=zero_policy,
                                     reference_phase=reference_phase)
        u_new = proximal_global_step(op, z_new, u, eta, weights=weights,
                                      zero_policy=zero_policy,
                                      reference_phase=reference_phase)
        l_after = coupling_energy(op, u_new, z_new, weights)
        step = float((z_new - z).abs().pow(2).sum()) + float((u_new - u).abs().pow(2).sum())
        history.append({"L": l_before, "L_next": l_after, "step": step})
        u, z = u_new, z_new
    return u, z, history

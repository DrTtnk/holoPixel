"""
Hardware complex-state projection for a phase-change-material panel.

tier2_slfh.quantization_aware.nearest_index/hard_quantize treat the display
constraint as PHASE quantisation: argmin over the angular distance to one of
L phase levels, amplitude never enters the comparison. That is the correct
model for an idealised phase-only SLM, but it is not what the Sb2Se3 device
actually is. realistic_levels' own per-level reflectance runs 0.583-0.946 (it
is the "non-uniform level set" that module's docstring already names), so
each achievable hardware state is a point off the unit circle,

    h_q = A_q * exp(i*theta_q),   A_q != const,

and the physically correct display constraint is projection onto that finite
COMPLEX constellation,

    Proj(v) = argmin_q | v - h_q |^2,

not argmin_q |arg(v) - theta_q|. The two coincide only when every A_q is
equal (proved in test_hardware_states.py, both symbolically and numerically);
on the realistic level set they measurably do not, since the nearest-in-angle
state need not be the nearest-in-distance one once amplitudes differ.

This module adds that missing projection. It deliberately does not touch
tier2_slfh/quantization_aware.py or tier2_hfh/optimise.py: it imports
`LevelSet` and `nearest_index` from the former and leaves the straight-
through-estimator training path in the latter exactly as it is. Whether
complex-state projection trains better than that STE is a separate, deferred
question -- this module only establishes that the two quantisers are
different operations and measures how often and by how much.

Every function here runs on the CPU in float64/complex128, regardless of the
device `uniform_levels`/`realistic_levels` happened to build their tensors
on (tier2_slfh.quantization_aware.DEVICE follows torch.cuda.is_available(),
which this module must not).
"""

import math

import torch

from tier2_slfh.quantization_aware import LevelSet, nearest_index

# Two states that are mathematically at the EXACT same complex distance from
# a point are not necessarily bit-identical once that distance has to go
# through a transcendental evaluation: state_field builds h_q = A_q *
# exp(i*theta_q) via cos/sin, and cos(theta)^2 + sin(theta)^2 is only equal
# to 1 to within a couple of double-precision ULPs (~1e-16 relative), not
# exactly. Comparing squared distances with `==` therefore breaks a genuine
# tie in whichever direction that rounding noise happens to fall -- an
# accident of the trig library, not a documented, reproducible policy. This
# tolerance is what makes "tied" mean "tied up to floating-point precision"
# instead, so the explicit lowest-index rule below actually fires. It sits
# many orders of magnitude below the smallest genuine amplitude/phase
# difference any level set in this project has (>= 1e-2), so no two
# distinguishable states are ever merged by it -- see
# test_hardware_states.py's brute-force cross-checks, which use exact `<`
# comparison and still agree with this tolerance-based version everywhere.
_TIE_RTOL = 1e-9
_TIE_ATOL = 1e-12


def _to_cpu_f64_levels(levels: LevelSet) -> LevelSet:
    """`levels` with its two tensors moved to the CPU and widened to
    float64, whatever device/dtype they arrived in."""
    return LevelSet(
        name=levels.name,
        phase=levels.phase.detach().to("cpu", dtype=torch.float64),
        amplitude=levels.amplitude.detach().to("cpu", dtype=torch.float64),
    )


def state_field(levels: LevelSet) -> torch.Tensor:
    """
    The L achievable hardware states h_q = A_q * exp(i*theta_q), as a
    complex128 CPU tensor of shape (L,), in the order `levels` stores them.

    This is a COMPLEX field -- amplitude and phase both included -- never a
    bare phase, matching the convention `tier2_hfh.optimise.quantised_field`
    already uses for what the panel actually emits.
    """
    lv = _to_cpu_f64_levels(levels)
    return lv.amplitude.to(torch.complex128) * torch.exp(1j * lv.phase.to(torch.complex128))


def nearest_state_index(field: torch.Tensor, levels: LevelSet) -> torch.Tensor:
    """
    argmin_q | field - h_q |^2 : the index of the nearest hardware state, by
    COMPLEX distance, for every element of `field`.

    This is NOT tier2_slfh.quantization_aware.nearest_index, which argmins
    over PHASE distance alone and never looks at A_q. The two agree only
    when every A_q in `levels` is equal (see
    test_unit_amplitude_reduces_to_phase_only_quantiser below); on a level
    set with unequal amplitudes, such as realistic_levels, they can and do
    pick different states for the same input.

    Vectorised over any shape of `field` by broadcasting against the L
    states on a trailing axis -- there is no Python loop over `field`'s
    elements.

    Ties (two or more states at the same complex distance, up to floating-
    point precision -- see `_TIE_RTOL`/`_TIE_ATOL` above for why exact `==`
    is not the right test) are broken by taking the LOWEST state index among
    the tied candidates. This is implemented explicitly below with an index
    mask, not left to rely on torch.argmin's own tie-break convention, and
    is pinned by test_hardware_states.py's tie tests.

    Returns an int64 CPU tensor with the same shape as `field`. Raises
    TypeError if `field` is not a complex tensor -- a state with no phase
    cannot be projected onto anything.
    """
    if not torch.is_complex(field):
        raise TypeError(f"field must be a complex tensor, got dtype {field.dtype}")

    states = state_field(levels)                       # (L,) complex128, cpu
    v = field.detach().to("cpu", dtype=torch.complex128)
    diff = v.unsqueeze(-1) - states                     # (..., L)
    dist2 = diff.real ** 2 + diff.imag ** 2             # squared distance; sqrt is not needed for argmin

    min_dist2 = dist2.min(dim=-1, keepdim=True).values
    is_nearest = torch.isclose(dist2, min_dist2, rtol=_TIE_RTOL, atol=_TIE_ATOL)  # (..., L) bool

    n_states = states.shape[-1]
    state_ids = torch.arange(n_states, dtype=torch.int64)
    sentinel = torch.full_like(state_ids, n_states)     # one past the last valid index
    candidate_ids = torch.where(is_nearest, state_ids, sentinel)
    return candidate_ids.min(dim=-1).values


def project_to_states(field: torch.Tensor, levels: LevelSet) -> torch.Tensor:
    """
    The nearest achievable hardware state h_q = A_q * exp(i*theta_q) for
    every element of `field` -- i.e.
    `state_field(levels)[nearest_state_index(field, levels)]`.

    Complex128, CPU, same shape as `field`. A COMPLEX FIELD, amplitude
    included -- never a bare phase.
    """
    idx = nearest_state_index(field, levels)
    return state_field(levels)[idx]


def phase_vs_complex_disagreement(phase: torch.Tensor, levels: LevelSet) -> dict:
    """
    Compares, element by element, the two display-constraint quantisers on
    the SAME unit-amplitude continuous field v = exp(i*phase) -- the field a
    continuous-phase optimiser (e.g. tier2_hfh.optimise's `phase` variable
    with `levels=None`) would hand to a quantiser at read-out time:

      * phase-only   : tier2_slfh.quantization_aware.nearest_index(phase, levels.phase)
      * complex-dist. : nearest_state_index(exp(i*phase), levels)

    Returns a dict:
      "phase_index"                 int64 CPU tensor, shape of `phase`
      "complex_index"               int64 CPU tensor, shape of `phase`
      "disagree"                    bool CPU tensor, shape of `phase`
      "disagreement_rate"           float, fraction of elements where the two differ
      "max_angular_disagreement_rad"  float, the largest wrapped phase gap between the
                                       two picks' OWN theta_q, over disagreeing elements
                                       only; None if there was no disagreement
      "max_squared_distance_gap"    float, the largest amount by which the phase-only
                                     pick's squared complex distance exceeds the
                                     complex-distance pick's, over disagreeing elements
                                     only -- how much the physically correct quantiser
                                     wins by in the worst observed case; None if there
                                     was no disagreement

    On a level set where every A_q is equal, this rate is provably exactly
    zero (see test_unit_amplitude_reduces_to_phase_only_quantiser): a
    nonzero rate there would mean this function, not the physics, is wrong.
    """
    level_phase = levels.phase.detach().to("cpu", dtype=torch.float64)
    phase64 = phase.detach().to("cpu", dtype=torch.float64)

    phase_idx = nearest_index(phase64, level_phase)
    field = torch.exp(1j * phase64.to(torch.complex128))
    complex_idx = nearest_state_index(field, levels)

    disagree = phase_idx != complex_idx
    result = {
        "phase_index": phase_idx,
        "complex_index": complex_idx,
        "disagree": disagree,
        "disagreement_rate": disagree.to(torch.float64).mean().item(),
        "max_angular_disagreement_rad": None,
        "max_squared_distance_gap": None,
    }

    if disagree.any():
        chosen_phase_a = level_phase[phase_idx[disagree]]
        chosen_phase_b = level_phase[complex_idx[disagree]]
        angular_gap = (chosen_phase_a - chosen_phase_b + math.pi) % (2 * math.pi) - math.pi
        result["max_angular_disagreement_rad"] = angular_gap.abs().max().item()

        states = state_field(levels)
        v_dis = field[disagree]
        dist2_a = (v_dis - states[phase_idx[disagree]]).abs() ** 2
        dist2_b = (v_dis - states[complex_idx[disagree]]).abs() ** 2
        result["max_squared_distance_gap"] = (dist2_a - dist2_b).max().item()

    return result


def constellation_geometry(levels: LevelSet) -> dict:
    """
    The geometry of the L hardware states {h_q}, for inspecting whether a
    level set is well- or ill-conditioned for nearest-state projection.

    Returns a dict:
      "amplitude"                  (L,) float64 CPU, A_q, in `levels`' own order
      "phase"                      (L,) float64 CPU, theta_q, in `levels`' own order
      "phase_gap"                  (L,) float64 CPU: for each state, the angular gap
                                    forward to whichever state is NEXT in ascending-phase
                                    order (wrapping the last state's gap back to the
                                    first), in `levels`' own order. Sums to 2*pi.
      "pairwise_distance"          (L, L) float64 CPU, |h_i - h_j|, zero on the diagonal
      "min_pairwise_distance"      float, the smallest OFF-DIAGONAL entry above -- two
                                    states this close are barely distinguishable to any
                                    quantiser (their Voronoi cells meet at a knife edge,
                                    and the smallest input perturbation flips the choice
                                    between them); a value near 0 means two states have
                                    effectively COLLAPSED into one, so the level set has
                                    fewer than L usable levels in practice
      "min_pairwise_distance_pair" (int, int), the state indices achieving that minimum

    Deliberately makes no judgement call about what counts as "too close":
    read "min_pairwise_distance" against "amplitude"/"phase_gap" for the
    level set at hand -- a separation that is fine at 8 levels can be
    alarming at 64.
    """
    lv = _to_cpu_f64_levels(levels)
    states = state_field(lv)
    n_states = len(lv)

    order = torch.argsort(lv.phase)
    sorted_phase = lv.phase[order]
    next_phase = torch.roll(sorted_phase, -1)
    gap_sorted = (next_phase - sorted_phase) % (2 * math.pi)
    gap = torch.empty_like(gap_sorted)
    gap[order] = gap_sorted

    pairwise = (states.unsqueeze(0) - states.unsqueeze(1)).abs()
    off_diagonal = pairwise + torch.diag(torch.full((n_states,), float("inf"), dtype=pairwise.dtype))
    flat_argmin = torch.argmin(off_diagonal).item()
    i, j = divmod(flat_argmin, n_states)

    return {
        "amplitude": lv.amplitude,
        "phase": lv.phase,
        "phase_gap": gap,
        "pairwise_distance": pairwise,
        "min_pairwise_distance": pairwise[i, j].item(),
        "min_pairwise_distance_pair": (i, j),
    }

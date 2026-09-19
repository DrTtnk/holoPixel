"""
Hardware complex-state projection: Proj(v) = argmin_q |v - h_q|^2, where
h_q = A_q*exp(i*theta_q) is one of the L states a real Sb2Se3 panel can
actually emit.

The central question this file answers is whether that projection is a
different operation from tier2_slfh.quantization_aware.nearest_index's
phase-only argmin, and if so, by how much -- see
test_phase_only_and_complex_distance_genuinely_disagree_on_realistic_levels
and test_unit_amplitude_reduces_to_phase_only_quantiser for the two sides
of that question (a measured positive result on the realistic level set,
a proved negative result on the uniform one).

All CPU, float64/complex128, no CUDA -- see hardware_states.py's own
docstring for why realistic_levels/uniform_levels' own device is not
trusted here.
"""

import math

import numpy as np
import pytest
import sympy as sp
import torch

from tier2_hfh.hardware_states import (
    constellation_geometry,
    nearest_state_index,
    phase_vs_complex_disagreement,
    project_to_states,
    state_field,
)
from tier2_slfh.quantization_aware import LevelSet, nearest_index, realistic_levels, uniform_levels


# ======================================================================
# Helpers
# ======================================================================

def _brute_force_index(field: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Reference argmin_q |v-h_q|^2 by an explicit double loop -- deliberately
    a different code path from nearest_state_index's broadcast, so it can
    catch a bug the vectorised version would share with itself."""
    flat = field.reshape(-1).tolist()
    states_list = states.tolist()
    out = []
    for v in flat:
        best_q, best_d2 = None, None
        for q, h in enumerate(states_list):
            d2 = (v.real - h.real) ** 2 + (v.imag - h.imag) ** 2
            if best_d2 is None or d2 < best_d2:
                best_q, best_d2 = q, d2
        out.append(best_q)
    return torch.tensor(out, dtype=torch.int64).reshape(field.shape)


def _random_complex(n, seed, max_amplitude=2.0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.0, max_amplitude, n)
    theta = rng.uniform(0.0, 2 * np.pi, n)
    return torch.tensor(r * np.cos(theta) + 1j * r * np.sin(theta), dtype=torch.complex128)


# ======================================================================
# Correctness of the projection itself
# ======================================================================

@pytest.mark.parametrize("n_levels", [2, 4, 8])
@pytest.mark.parametrize("spacing", ["uniform", "realistic"])
def test_nearest_state_index_matches_brute_force(n_levels, spacing):
    levels = uniform_levels(n_levels) if spacing == "uniform" else realistic_levels(n_levels)
    field = _random_complex(2000, seed=hash((n_levels, spacing)) % (2 ** 31))

    got = nearest_state_index(field, levels)
    want = _brute_force_index(field, state_field(levels))

    assert torch.equal(got, want)


@pytest.mark.parametrize("n_levels", [2, 4, 8])
@pytest.mark.parametrize("spacing", ["uniform", "realistic"])
def test_project_to_states_returns_the_indexed_state(n_levels, spacing):
    levels = uniform_levels(n_levels) if spacing == "uniform" else realistic_levels(n_levels)
    field = _random_complex(500, seed=1)

    idx = nearest_state_index(field, levels)
    projected = project_to_states(field, levels)

    assert torch.allclose(projected, state_field(levels)[idx])


@pytest.mark.parametrize("n_levels", [2, 4, 8])
@pytest.mark.parametrize("spacing", ["uniform", "realistic"])
def test_projection_is_idempotent(n_levels, spacing):
    """A state that is already achievable must project to itself: it is its
    own nearest neighbour with distance exactly 0."""
    levels = uniform_levels(n_levels) if spacing == "uniform" else realistic_levels(n_levels)
    states = state_field(levels)

    idx = nearest_state_index(states, levels)

    assert torch.equal(idx, torch.arange(n_levels, dtype=torch.int64))


def test_nearest_state_index_rejects_non_complex_input():
    levels = uniform_levels(4)
    with pytest.raises(TypeError):
        nearest_state_index(torch.tensor([0.0, 1.0]), levels)


@pytest.mark.parametrize("n_levels", [2, 4, 8])
@pytest.mark.parametrize("spacing", ["uniform", "realistic"])
def test_outputs_are_cpu_float64_regardless_of_input_device(n_levels, spacing):
    """The hard CPU-only constraint: even though uniform_levels/realistic_levels
    build their tensors on tier2_slfh.quantization_aware.DEVICE (cuda when
    available), everything computed here must stay on the CPU."""
    levels = uniform_levels(n_levels) if spacing == "uniform" else realistic_levels(n_levels)
    field = _random_complex(50, seed=2)

    states = state_field(levels)
    idx = nearest_state_index(field, levels)
    projected = project_to_states(field, levels)
    geometry = constellation_geometry(levels)

    assert states.device.type == "cpu" and states.dtype == torch.complex128
    assert idx.device.type == "cpu" and idx.dtype == torch.int64
    assert projected.device.type == "cpu" and projected.dtype == torch.complex128
    assert geometry["amplitude"].device.type == "cpu" and geometry["amplitude"].dtype == torch.float64
    assert geometry["phase"].device.type == "cpu" and geometry["phase"].dtype == torch.float64


# ======================================================================
# Ties
# ======================================================================

def test_exact_ties_break_to_the_lowest_index_symmetric_uniform_case():
    """Two adjacent unit-amplitude states are exactly equidistant from the
    point that bisects their angle -- construct that point directly and
    confirm the documented tie-break (lowest index) fires."""
    levels = uniform_levels(4)   # phases 0, pi/2, pi, 3*pi/2, all amplitude 1
    phase_cpu = levels.phase.detach().cpu().to(torch.float64)
    bisector_angle = (phase_cpu[0] + phase_cpu[1]) / 2  # exactly pi/4
    v = torch.exp(1j * bisector_angle.to(torch.complex128)).reshape(1)

    states = state_field(levels)
    d2 = (v.unsqueeze(-1) - states).abs() ** 2   # (1, L)
    tied = torch.isclose(d2[0, 0], d2[0, 1])
    assert tied, "test setup did not actually produce an exact tie"

    idx = nearest_state_index(v, levels)
    assert idx.item() == 0


def test_exact_ties_break_to_the_lowest_index_unequal_amplitude_case():
    """A tie constructed with genuinely unequal amplitudes (not just the
    unit-circle symmetry above), so the tie-break is checked independently
    of the unit-amplitude special case."""
    # h0 = 1+0j, h1 = 0+1j: any point on the line x=y is equidistant from
    # both. A third, distant state with a different amplitude is present so
    # the test also exercises L > 2.
    levels = LevelSet(
        name="hand-built",
        phase=torch.tensor([0.0, math.pi / 2, math.pi], dtype=torch.float32),
        amplitude=torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32),
    )
    v = torch.tensor([0.5 + 0.5j], dtype=torch.complex128)

    states = state_field(levels)
    d2 = (v.unsqueeze(-1) - states).abs() ** 2   # (1, L)
    assert torch.isclose(d2[0, 0], d2[0, 1])
    assert d2[0, 2] > d2[0, 0]   # the third state is not part of the tie

    idx = nearest_state_index(v, levels)
    assert idx.item() == 0


def test_three_way_exact_tie_breaks_to_the_lowest_index():
    """Origin is equidistant from every state on a common circle -- forces
    an L-way tie, not just a pairwise one."""
    levels = uniform_levels(6)
    v = torch.zeros(1, dtype=torch.complex128)

    idx = nearest_state_index(v, levels)
    assert idx.item() == 0


# ======================================================================
# The unit-amplitude reduction: complex-distance projection MUST collapse
# to nearest-phase projection when every A_q is equal.
# ======================================================================

def test_unit_amplitude_reduction_is_algebraically_forced():
    """
    Symbolic proof that |v - A*exp(i*theta)|^2 = A^2 + r^2 - 2*A*r*cos(phi-theta)
    for v = r*exp(i*phi). When A is the SAME constant for every state q (the
    unit/uniform-amplitude level set), A and r do not depend on q, so
    argmin_q of the left-hand side is exactly argmax_q cos(phi-theta_q) --
    i.e. argmin over angular distance to theta_q. That is precisely
    tier2_slfh.quantization_aware.nearest_index's definition, so the two
    quantisers are forced to agree whenever amplitude is uniform, for every
    v and every A > 0. This is checked once symbolically here and then
    numerically for the actual code below.
    """
    r, phi, A, theta = sp.symbols("r phi A theta", real=True)
    v = r * sp.exp(sp.I * phi)
    h = A * sp.exp(sp.I * theta)
    dist2 = sp.expand_complex(sp.re((v - h) * sp.conjugate(v - h)))
    expected = A ** 2 + r ** 2 - 2 * A * r * sp.cos(phi - theta)
    assert sp.simplify(dist2 - expected) == 0


@pytest.mark.parametrize("n_levels", [2, 3, 4, 8, 16])
def test_unit_amplitude_reduces_to_phase_only_quantiser(n_levels):
    levels = uniform_levels(n_levels)
    rng = np.random.default_rng(n_levels)
    n = 20_000
    r = rng.uniform(0.01, 3.0, n)          # any positive amplitude, per the symbolic proof above
    phi = rng.uniform(0.0, 2 * np.pi, n)
    field = torch.tensor(r * np.cos(phi) + 1j * r * np.sin(phi), dtype=torch.complex128)
    phase = torch.tensor(phi, dtype=torch.float64)

    complex_idx = nearest_state_index(field, levels)
    phase_idx = nearest_index(phase, levels.phase.detach().cpu().to(torch.float64))

    assert torch.equal(complex_idx, phase_idx)


# ======================================================================
# The realistic level set: does complex-distance projection genuinely
# differ from phase-only projection? Quantify it.
# ======================================================================

@pytest.mark.parametrize("n_levels", [4, 8, 16])
def test_phase_only_and_complex_distance_genuinely_disagree_on_realistic_levels(n_levels):
    levels = realistic_levels(n_levels)
    assert levels.amplitude.std().item() > 0.01, (
        "precondition failed: realistic_levels is not actually non-uniform in amplitude "
        "here, so a disagreement test against it would be meaningless"
    )

    phase = torch.tensor(
        np.random.default_rng(n_levels).uniform(0.0, 2 * np.pi, 200_000), dtype=torch.float64
    )
    result = phase_vs_complex_disagreement(phase, levels)

    assert result["disagreement_rate"] > 0.0, (
        "NEGATIVE RESULT: phase-only and complex-distance quantisers never disagreed "
        f"on the realistic {n_levels}-level set over 200k random inputs -- see this "
        "test's failure as a finding, not a bug, if it ever fires"
    )
    assert result["max_angular_disagreement_rad"] is not None
    assert result["max_squared_distance_gap"] > 0.0


def test_disagreement_is_exactly_zero_on_the_uniform_level_set():
    """The negative-result counterpart of the test above: on the idealised
    unit-amplitude level set the two quantisers must never disagree, over
    the same random-input procedure used for the realistic one."""
    levels = uniform_levels(8)
    phase = torch.tensor(np.random.default_rng(0).uniform(0.0, 2 * np.pi, 200_000), dtype=torch.float64)

    result = phase_vs_complex_disagreement(phase, levels)

    assert result["disagreement_rate"] == 0.0
    assert result["max_angular_disagreement_rad"] is None
    assert result["max_squared_distance_gap"] is None


def test_realistic_8level_disagreement_rate_is_in_the_measured_range():
    """
    Pins the actual measured numbers for the device this project targets (8
    levels, 300nm Sb2Se3 / 6-pair TiO2-SiO2 DBR at 532nm) so a future change
    to design_phase_lut's TMM correction that silently made the level set
    amplitude-uniform (defeating the whole point of this module) would be
    caught here, not just in a report.
    """
    levels = realistic_levels(8)
    phase = torch.tensor(np.random.default_rng(42).uniform(0.0, 2 * np.pi, 500_000), dtype=torch.float64)

    result = phase_vs_complex_disagreement(phase, levels)

    # Measured ~0.7% disagreement, worst-case a full adjacent-state (45-degree)
    # jump, at this level set's specific amplitude spread. Loose bounds: this
    # pins "genuinely disagrees, and not by a rounding-error amount", not the
    # exact figure.
    assert 1e-4 < result["disagreement_rate"] < 0.1
    assert result["max_angular_disagreement_rad"] > math.radians(1.0)


# ======================================================================
# Constellation geometry
# ======================================================================

@pytest.mark.parametrize("n_levels", [2, 4, 8, 16])
@pytest.mark.parametrize("spacing", ["uniform", "realistic"])
def test_constellation_geometry_shapes_and_invariants(n_levels, spacing):
    levels = uniform_levels(n_levels) if spacing == "uniform" else realistic_levels(n_levels)
    g = constellation_geometry(levels)

    assert g["amplitude"].shape == (n_levels,)
    assert g["phase"].shape == (n_levels,)
    assert g["phase_gap"].shape == (n_levels,)
    assert g["pairwise_distance"].shape == (n_levels, n_levels)

    # phase_gap tiles the whole circle exactly once
    assert math.isclose(g["phase_gap"].sum().item(), 2 * math.pi, rel_tol=1e-9)
    assert torch.all(g["phase_gap"] > 0)

    # pairwise_distance is symmetric with an exact zero diagonal
    assert torch.allclose(g["pairwise_distance"], g["pairwise_distance"].T)
    assert torch.allclose(torch.diag(g["pairwise_distance"]), torch.zeros(n_levels, dtype=torch.float64))

    # no two distinct states have collapsed onto each other
    assert g["min_pairwise_distance"] > 1e-6
    i, j = g["min_pairwise_distance_pair"]
    assert i != j
    assert math.isclose(g["pairwise_distance"][i, j].item(), g["min_pairwise_distance"], rel_tol=1e-9)


def test_uniform_level_set_phase_gaps_are_all_equal():
    """Sanity check of constellation_geometry against a level set whose
    spacing is known exactly: 2*pi/N for every gap."""
    n_levels = 8
    levels = uniform_levels(n_levels)
    g = constellation_geometry(levels)

    expected = 2 * math.pi / n_levels
    assert torch.allclose(g["phase_gap"], torch.full((n_levels,), expected, dtype=torch.float64), atol=1e-5)


def test_realistic_8level_geometry_matches_measured_hardware_values():
    """
    Pins the actual Sb2Se3 device geometry this project targets: TMM-corrected
    phase steps are (by construction) uniform at 45 degrees, and per-level
    amplitude runs roughly 0.76-0.97 (reflectance 0.58-0.95) -- the numbers
    quoted in this module's docstring and in the task's own brief.
    """
    levels = realistic_levels(8)
    g = constellation_geometry(levels)

    assert torch.allclose(g["phase_gap"], torch.full((8,), math.pi / 4, dtype=torch.float64), atol=1e-3)
    assert 0.75 < g["amplitude"].min().item() < 0.80
    assert 0.95 < g["amplitude"].max().item() < 1.00
    assert g["amplitude"].std().item() > 0.01

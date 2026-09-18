"""
Recovering coherent modes from a light field, with no depth channel.

This is what lets hogel-free holography run on a plain light field. Their
method takes colour-plus-depth: the colour gives the angular intensities and
the depth supplies the phase, because a ray from a known distance arrives with
a phase geometry determines. Drop the depth and that phase is missing.

Wolf's theorem supplies it instead. A light field determines the mutual
coherence J, which is Hermitian positive semi-definite, so its eigenvectors
are coherent modes carrying definite phase -- derived from the measurement
rather than from an assumed surface. Crucially this places no limit of one
surface per pixel, so glass, smoke and mirrors are representable, which is
exactly what a depth map cannot do.

These tests come before the implementation. Each one synthesises a field whose
mode content is known by construction and checks that it comes back.
"""

import numpy as np
import pytest

from tier2_hfh.modes import (
    coherence_from_wigner,
    decompose,
    mode_count_for,
    wigner_2d,
)

N = 9          # odd: the (c+s, c-s) index map is a bijection only for odd n


def _random_field(rng, n=N):
    return (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))).astype(np.complex128)


# ---------------------------------------------------------------------------
# The property everything rests on: a coherent field is rank one.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_single_coherent_field_decomposes_to_exactly_one_mode(seed):
    rng = np.random.default_rng(seed)
    u = _random_field(rng)

    weights, modes = decompose(wigner_2d(u))

    assert weights[0] / weights.sum() > 1 - 1e-9
    assert weights[1] / weights.sum() < 1e-9


@pytest.mark.parametrize("seed", [3, 4])
def test_the_recovered_mode_is_the_original_field_up_to_a_global_phase(seed):
    """
    Rank one is necessary but not sufficient: the mode must be the field
    itself. A global phase is unobservable, so it is divided out before
    comparing.
    """
    rng = np.random.default_rng(seed)
    u = _random_field(rng)

    weights, modes = decompose(wigner_2d(u))
    recovered = modes[0] * np.sqrt(weights[0])

    phase = np.vdot(recovered.ravel(), u.ravel())
    phase /= abs(phase)
    assert np.abs(recovered * phase - u).max() / np.abs(u).max() < 1e-9


@pytest.mark.parametrize("n_modes", [2, 3, 5])
def test_incoherently_summed_fields_give_exactly_that_many_modes(n_modes):
    """Wigner distributions of mutually incoherent fields add."""
    rng = np.random.default_rng(10 + n_modes)
    fields = [_random_field(rng) for _ in range(n_modes)]

    weights, _ = decompose(sum(wigner_2d(f) for f in fields))
    frac = weights / weights.sum()

    assert frac[:n_modes].sum() > 1 - 1e-9
    assert frac[n_modes] < 1e-9


def test_the_modes_reproduce_the_light_field_they_came_from():
    """Round trip: decompose, then re-sum the modes' Wigner distributions."""
    rng = np.random.default_rng(20)
    target = sum(wigner_2d(_random_field(rng)) for _ in range(3))

    weights, modes = decompose(target)
    rebuilt = sum(w * wigner_2d(m) for w, m in zip(weights[:3], modes[:3]))

    assert np.abs(rebuilt - target).max() / np.abs(target).max() < 1e-9


# ---------------------------------------------------------------------------
# Structure of the coherence matrix.
# ---------------------------------------------------------------------------

def test_coherence_is_hermitian_and_positive_semidefinite():
    rng = np.random.default_rng(30)
    J = coherence_from_wigner(sum(wigner_2d(_random_field(rng)) for _ in range(4)))

    assert np.abs(J - J.conj().T).max() < 1e-10
    assert np.linalg.eigvalsh(J).min() > -1e-10


def test_the_diagonal_of_the_coherence_is_the_intensity():
    rng = np.random.default_rng(31)
    u = _random_field(rng)

    J = coherence_from_wigner(wigner_2d(u))

    assert np.allclose(np.diag(J).real.reshape(N, N), np.abs(u) ** 2, atol=1e-10)


def test_an_even_grid_is_refused_rather_than_silently_wrong():
    """
    The (c+s, c-s) map is a bijection only when 2 is invertible modulo n, i.e.
    for odd n. On an even grid it covers half the matrix and doubles up on the
    rest, which produced a plausible-looking 52% first eigenvalue instead of
    100% before this was caught.
    """
    with pytest.raises(ValueError, match="odd"):
        coherence_from_wigner(np.zeros((8, 8, 8, 8)))


# ---------------------------------------------------------------------------
# Batching, which is what makes a whole panel affordable.
# ---------------------------------------------------------------------------

def test_a_batch_gives_the_same_answer_as_decomposing_one_at_a_time():
    rng = np.random.default_rng(40)
    batch = np.stack([sum(wigner_2d(_random_field(rng)) for _ in range(2))
                      for _ in range(5)])

    w_batch, m_batch = decompose(batch)

    for i in range(5):
        w_one, m_one = decompose(batch[i])
        assert np.allclose(w_batch[i], w_one, rtol=1e-9, atol=1e-12)


def test_mode_count_for_reports_the_modes_needed_for_a_given_energy():
    rng = np.random.default_rng(50)
    weights, _ = decompose(sum(wigner_2d(_random_field(rng)) for _ in range(4)))

    assert mode_count_for(weights, 0.50) == 2
    assert mode_count_for(weights, 0.99) == 4
    assert mode_count_for(weights, 0.999999) == 4


def test_mode_count_never_exceeds_the_number_of_modes_that_exist():
    """
    Asking for every last fraction of the power runs into the 1e-13 tail left
    by rounding, where the cumulative sum need not reach 1.0 and searchsorted
    runs off the end. The answer must still be a count of real modes.
    """
    rng = np.random.default_rng(51)
    weights, modes = decompose(sum(wigner_2d(_random_field(rng)) for _ in range(4)))

    assert mode_count_for(weights, 1.0) == len(weights) == len(modes)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_mode_count_refuses_a_fraction_outside_the_unit_interval(bad):
    rng = np.random.default_rng(52)
    weights, _ = decompose(wigner_2d(_random_field(rng)))

    with pytest.raises(ValueError, match="fraction"):
        mode_count_for(weights, bad)

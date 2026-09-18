"""
Coherent modes from a light field, with no depth channel.

Hogel-free holography (Chakravarthula et al., ACM TOG 2022) takes
colour-plus-depth views: colour supplies the angular intensities, depth
supplies the phase, since a ray from a known distance arrives with a phase
that geometry fixes. A depth map holds one surface per pixel, so glass, smoke
and mirrors cannot be represented -- a limitation that paper itself levels at
tensor holography in its section 5.2, and which our Cornell scene now exhibits
deliberately (the glass ball's interior image parallaxes as though it were
66 mm outside the box, while its surface is inside).

This module supplies the phase from the measurement instead. A light field is
the Wigner distribution; transforming it back gives the mutual coherence J,
which is Hermitian positive semi-definite, so

    J = sum_m  alpha_m  phi_m phi_m^*

with alpha_m >= 0 and phi_m mutually orthogonal coherent fields. That is
Wolf's coherent-mode representation (JOSA 72(3), 1982). The eigenvectors carry
definite phase and assume nothing about surfaces.

Two facts govern how this is used, both measured rather than assumed:

  * The mode count tracks the ANGULAR sample count, not the spatial one.
    Our 17-view light field needs about 16 modes for 90% of its energy and
    our 9-view one about 9, so neither has yet reached the scene's own
    complexity -- see docs/notes_coherent_modes.md.

  * M mutually incoherent modes give speckle contrast 1/sqrt(M)
    (docs/holographic_rendering_equation.md section 9, verified in
    tests/test_holographic_transport.py). The modes needed for fidelity are
    the same ones that suppress speckle.

The decomposition is LOCAL. A global coherence matrix for a 512x512 panel
would hold 512^4 entries, about 550 GB. The windowed Fourier transform in
`wft.py` already localises the problem: within a window of n samples per axis
J is only n^2 by n^2, so 1.3 MB at n = 17, and the windows batch. The panel is
still optimised as a whole; the windowing is analysis only, exactly as in the
original method.
"""

import numpy as np


def wigner_2d(field):
    """
    Discrete Wigner distribution of a 2D complex field, on the same grid.

    W[y, x, ky, kx] = sum_{sy,sx} U[y+sy, x+sx] conj(U[y-sy, x-sx])
                                  exp(-2i pi (ky sy + kx sx) / n)

    This is the light field a renderer would produce if the scene were exactly
    this coherent field, which is what makes it the right synthetic test case.
    """
    n = field.shape[0]
    if field.shape != (n, n):
        raise ValueError(f"field must be square, got {field.shape}")
    idx = np.arange(n)
    plus_y = (idx[:, None] + idx[None, :]) % n
    minus_y = (idx[:, None] - idx[None, :]) % n
    # [y, x, sy, sx]
    plus = field[plus_y[:, None, :, None], plus_y[None, :, None, :]]
    minus = np.conj(field[minus_y[:, None, :, None], minus_y[None, :, None, :]])
    return np.fft.fft2(plus * minus, axes=(2, 3))


def coherence_from_wigner(wigner):
    """
    Mutual coherence J from a Wigner distribution, as an (n^2, n^2) matrix
    indexed by flattened (y, x). Accepts a leading batch axis.

    The inverse transform gives a[cy, cx, sy, sx] = J[(cy+sy, cx+sx),
    (cy-sy, cx-sx)], so the grid must be ODD: the map (c, s) -> (c+s, c-s)
    modulo n is a bijection only when 2 is invertible, which fails for even n
    and silently fills half the matrix twice while leaving the rest at zero.
    """
    batched = wigner.ndim == 5
    w = wigner if batched else wigner[None]
    b, n = w.shape[0], w.shape[1]
    if n % 2 == 0:
        raise ValueError(f"grid must be odd for the Wigner index map to invert, got n={n}")

    a = np.fft.ifft2(w, axes=(3, 4))
    c = np.arange(n)
    plus = (c[:, None] + c[None, :]) % n        # [c, s]
    minus = (c[:, None] - c[None, :]) % n

    J = np.zeros((b, n, n, n, n), dtype=complex)
    # rows (cy+sy, cx+sx), columns (cy-sy, cx-sx)
    J[:,
      plus[:, None, :, None], plus[None, :, None, :],
      minus[:, None, :, None], minus[None, :, None, :]] = a

    J = J.reshape(b, n * n, n * n)
    J = 0.5 * (J + np.conj(np.transpose(J, (0, 2, 1))))     # Hermitian by construction
    return J if batched else J[0]


def decompose(wigner):
    """
    Coherent modes of a light field, strongest first.

    Returns (weights, modes). `weights` are the eigenvalues of J, so the power
    in each mode; `modes` are unit-norm complex fields shaped like the input
    grid. Negative eigenvalues, which arise only from rounding, are clipped:
    J is positive semi-definite by construction.
    """
    batched = wigner.ndim == 5
    J = coherence_from_wigner(wigner)
    Jb = J if batched else J[None]
    n = int(round(Jb.shape[-1] ** 0.5))

    vals, vecs = np.linalg.eigh(Jb)
    vals = np.clip(vals[:, ::-1].real, 0.0, None)
    vecs = vecs[:, :, ::-1]

    modes = np.transpose(vecs, (0, 2, 1)).reshape(Jb.shape[0], n * n, n, n)
    return (vals, modes) if batched else (vals[0], modes[0])


def mode_count_for(weights, fraction):
    """
    How many modes hold at least `fraction` of the total power.

    Capped at the number of modes that exist. Rounding leaves a tail of
    eigenvalues around 1e-13, so a cumulative sum need not reach exactly 1.0
    and searchsorted then runs off the end; the cap is a structural fact about
    the decomposition, not a tolerance.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    total = weights.sum()
    if total <= 0:
        raise ValueError("light field carries no power")
    idx = int(np.searchsorted(np.cumsum(weights) / total, fraction) + 1)
    return min(idx, len(weights))

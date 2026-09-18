"""
Windowed Fourier transform: the operator at the core of hogel-free holography.

Chakravarthula et al., "Hogel-free Holography", ACM TOG 2022. Their Eq. 2
collapses a sum over M x N angular light-field views into a single wave-domain
term, using an operator that maps the discrete angular views onto one complex
wavefront. That operator is a windowed Fourier transform (WFT).

The same transform defines a hogel: `docs/holographic_rendering_equation.md`
section 7 proves the per-hogel angular intensity is a spectrogram, verified to
3e-16 in `tests/test_holographic_transport.py`. The difference between the two
methods is what they do with it. A hogel pipeline uses the window to PARTITION
the panel into independently solvable tiles. Hogel-free applies it once to
convert the target, then optimises the whole panel jointly. Same operator,
used as a partition in one case and as a change of coordinates in the other.

Two properties matter and both are measured in `tests/test_wft.py`:

1. Analysis followed by synthesis must be EXACT. It is, to ~1e-16, for every
   window whose overlap-squared sum is nowhere zero.

2. The operator must be WELL CONDITIONED. Exactness on clean input says
   nothing about what happens to error in the light field. Non-overlapping
   frames with a tapered window amplify error at the frame seams; see
   `overlap_noise_gain`.

Property 2 is not discussed in the paper and it is the reason this module
exposes the hop size as a parameter rather than fixing it at the window
length.
"""

import numpy as np

# A window used at hop == len(window) must be non-zero at both endpoints,
# otherwise the samples landing there are covered by no frame at all and are
# unrecoverable. Of the common tapered windows only Hamming satisfies this,
# which is very likely why the paper uses it.
WINDOWS = {
    "rect": lambda w: np.ones(w),
    "hann": lambda w: np.hanning(w + 1)[:-1],
    "hamming": lambda w: np.hamming(w + 1)[:-1],
    "blackman": lambda w: np.blackman(w + 1)[:-1],
}


def window_1d(name, width):
    return WINDOWS[name](width)


def window_2d(name, width):
    w = window_1d(name, width)
    return np.outer(w, w)


def _starts(n, hop):
    if n % hop:
        raise ValueError(f"hop {hop} must divide the signal length {n}")
    return np.arange(0, n, hop)


def overlap_weight(window, hop, n):
    """
    Sum of window^2 over every frame that covers each sample. Synthesis
    divides by this, so it is the conditioning of the whole operator: a
    sample where it is small is a sample where light-field error is amplified.
    """
    acc = np.zeros(n)
    w = len(window)
    for start in _starts(n, hop):
        np.add.at(acc, (start + np.arange(w)) % n, window ** 2)
    return acc


def _reject_uncovered(acc, hop):
    """
    A sample whose weight is below the peak weight times the double-precision
    epsilon carries no recoverable information: dividing by it returns rounding
    noise, not signal. The threshold is the precision of the arithmetic, not a
    tolerance chosen to make a case pass. Windows that are analytically zero at
    an endpoint land around 1e-34 here rather than at exactly zero, so testing
    against zero is not enough.
    """
    floor = acc.max() * np.finfo(float).eps
    if acc.min() <= floor:
        raise ValueError(
            f"window leaves {int((acc <= floor).sum())} samples uncovered at hop "
            f"{hop}: they are covered by no frame and cannot be recovered"
        )


def overlap_noise_gain(window, hop, n):
    """
    Worst-case amplification of an error in the light field, as an amplitude
    ratio. 1.0 is ideal. Raises if any sample is uncovered, because that is
    not a large gain but an unrecoverable sample.
    """
    acc = overlap_weight(window, hop, n)
    _reject_uncovered(acc, hop)
    return float(np.sqrt(acc.max() / acc.min()))


def analyse(field, window, hop):
    """
    Light field from a wavefront. Returns (n_frames_y, n_frames_x, w, w):
    for each window position, the angular spectrum seen through that window.
    `field` is 2D, `window` is the separable 2D window.
    """
    ny, nx = field.shape
    w = window.shape[0]
    sy, sx = _starts(ny, hop), _starts(nx, hop)
    iy = (sy[:, None] + np.arange(w)[None, :]) % ny
    ix = (sx[:, None] + np.arange(w)[None, :]) % nx
    tiles = field[iy[:, None, :, None], ix[None, :, None, :]]
    return np.fft.fft2(tiles * window[None, None, :, :], axes=(2, 3))


def synthesise(spectra, window, hop, shape):
    """
    Wavefront from a light field. Weighted overlap-add, exact inverse of
    `analyse` whenever `overlap_weight` is nowhere zero.
    """
    ny, nx = shape
    w = window.shape[0]
    tiles = np.fft.ifft2(spectra, axes=(2, 3))
    out = np.zeros((ny, nx), dtype=complex)
    acc = np.zeros((ny, nx))
    for a, start_y in enumerate(_starts(ny, hop)):
        iy = (start_y + np.arange(w)) % ny
        for b, start_x in enumerate(_starts(nx, hop)):
            ix = (start_x + np.arange(w)) % nx
            np.add.at(out, (iy[:, None], ix[None, :]), tiles[a, b] * window)
            np.add.at(acc, (iy[:, None], ix[None, :]), window ** 2)
    _reject_uncovered(acc, hop)
    return out / acc

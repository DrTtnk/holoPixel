"""
Verification spike for the holographic rendering equation
(docs/holographic_rendering_equation.md).

CLAUDE.md requires that any derivation past schoolbook algebra be checked with
a tool before it reaches code, on random inputs rather than a convenient
special case, and that the check stay in the repository. Every identity the
document relies on is therefore re-derived here, either symbolically with
sympy or numerically to machine precision on random data.

The five load-bearing claims:

  1. Free-space propagation acts on the Wigner distribution as a pure SHEAR,
     W_z(x, u) = W_0(x - lambda z u, u).
  2. That shear is the same first-order transport as the classical rendering
     equation's free-space term. This is the bridge to radiance.
  3. A hogel is a short-time Fourier transform, so per-hogel intensity is a
     SPECTROGRAM, whose transform factorises into ambiguity functions. The
     hogel spatio-angular trade-off is therefore the time-frequency
     uncertainty principle and nothing more exotic.
  4. The Gabor bound sigma_x * sigma_nu >= 1/(4 pi), with equality for a
     Gaussian window, fixes the exchange rate of that trade-off.
  5. Averaging M independent speckle realisations gives contrast 1/sqrt(M).

Conventions, fixed once and used throughout:
    omega          = exp(-2i pi / N)                  (numpy's forward DFT)
    Wigner         W(x, u)   = int U(x+s/2) U*(x-s/2) exp(-2i pi u s) ds
    ambiguity      A(tau, a) = sum_t V(t) V*(t-tau) omega^{a t}
    spatial freq   u in cycles/metre; paraxial angle theta with sin(theta) = lambda u
"""

import numpy as np
import pytest
import sympy as sp


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def fresnel(field, lam, z, dx):
    """Angular-spectrum propagation under the Fresnel (paraxial) kernel."""
    f = np.fft.fftfreq(len(field), d=dx)
    return np.fft.ifft(np.fft.fft(field) * np.exp(-1j * np.pi * lam * z * f ** 2))


def band_limited(n, rng, keep_fraction=0.25):
    """Random complex field with its spectrum confined to the central band."""
    field = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    spec = np.fft.fft(field)
    keep = int(n * keep_fraction)
    spec[keep: n - keep + 1] = 0
    return np.fft.ifft(spec)


def wigner(field, dx):
    """
    Discrete Wigner distribution. The field is band-limit-upsampled 2x so the
    half-lag s/2 lands on a sample and no interpolation is needed.
    Returns W[x, u] on the fine grid, with its (x, u) axis spacings.
    """
    n = len(field)
    spec = np.fft.fft(field)
    padded = np.zeros(2 * n, dtype=complex)
    padded[: n // 2] = spec[: n // 2]
    padded[-(n // 2):] = spec[n // 2:]
    fine = np.fft.ifft(padded) * 2

    m = 2 * n
    idx = np.arange(m)[:, None]
    half_lag = np.arange(m)[None, :]
    correlation = (fine[(idx + half_lag) % m]
                   * np.conj(fine[(idx - half_lag) % m]))

    w = np.fft.fft(correlation, axis=1)
    nu_axis = np.fft.fftfreq(m, d=dx / 2)   # conjugate to the fine x grid
    u_axis = np.fft.fftfreq(m, d=dx)        # conjugate to the full lag s
    return w, nu_axis, u_axis


def ambiguity(field):
    """A[tau, a] = DFT over t of  V(t) V*(t - tau)."""
    n = len(field)
    t = np.arange(n)[None, :]
    tau = np.arange(n)[:, None]
    return np.fft.fft(field[None, :] * np.conj(field[(t - tau) % n]), axis=1)


def spectrogram(field, window):
    """S[x, k] = |sum_t field(t) window*(t - x) omega^{k t}|^2 — one hogel per x."""
    return np.stack([
        np.abs(np.fft.fft(field * np.conj(np.roll(window, x)))) ** 2
        for x in range(len(field))
    ])


# --------------------------------------------------------------------------
# 1. free-space propagation is a shear of the Wigner distribution
# --------------------------------------------------------------------------

def test_shear_phase_algebra_is_exact():
    """The whole shear rests on (u + v/2)^2 - (u - v/2)^2 = 2 u v."""
    u, v = sp.symbols("u v", real=True)
    assert sp.expand((u + v / 2) ** 2 - (u - v / 2) ** 2) == 2 * u * v


@pytest.mark.parametrize("seed,z", [(0, 3.7), (1, -2.0), (2, 11.25)])
def test_wigner_transport_is_a_pure_shear(seed, z):
    """
    Checked in the ambiguity domain, where the x-shear becomes an exact
    pointwise phase and no interpolation is involved. Random band-limited
    input, forward and backward propagation.
    """
    rng = np.random.default_rng(seed)
    n, dx, lam = 128, 1.0, 1.0

    start = band_limited(n, rng)
    moved = fresnel(start, lam, z, dx)

    w_start, nu_axis, u_axis = wigner(start, dx)
    w_moved, _, _ = wigner(moved, dx)

    amb_start = np.fft.fft(w_start, axis=0)
    amb_moved = np.fft.fft(w_moved, axis=0)

    shear = np.exp(-1j * 2 * np.pi * lam * z * u_axis[None, :] * nu_axis[:, None])
    residual = np.abs(amb_moved - amb_start * shear).max() / np.abs(amb_start).max()
    assert residual < 1e-12


def test_wigner_is_real_and_conserves_energy():
    rng = np.random.default_rng(5)
    start = band_limited(128, rng)
    moved = fresnel(start, 1.0, 4.0, 1.0)

    w_start, _, _ = wigner(start, 1.0)
    w_moved, _, _ = wigner(moved, 1.0)

    assert np.abs(w_start.imag).max() < 1e-9 * np.abs(w_start).max()
    assert w_start.sum().real == pytest.approx(w_moved.sum().real, rel=1e-10)


# --------------------------------------------------------------------------
# 2. the bridge: the shear IS free-space radiance transport
# --------------------------------------------------------------------------

def test_shear_satisfies_the_radiance_transport_equation():
    """
    The classical rendering equation's free-space term transports radiance
    along straight rays:  dL/dz + tan(theta) dL/dx = 0.
    Show the Wigner shear obeys the identical PDE with tan(theta) -> lambda u,
    which is the paraxial identification. This is the incoherent limit.
    """
    x, u, z, lam = sp.symbols("x u z lambda", real=True)
    w0 = sp.Function("W0")

    sheared = w0(x - lam * z * u, u)
    transport = sp.diff(sheared, z) + lam * u * sp.diff(sheared, x)
    assert sp.simplify(transport) == 0

    # and the classical form, for the same reason
    theta = sp.Symbol("theta", real=True)
    radiance = sp.Function("L0")(x - z * sp.tan(theta), theta)
    classical = sp.diff(radiance, z) + sp.tan(theta) * sp.diff(radiance, x)
    assert sp.simplify(classical) == 0


def test_paraxial_identification_is_first_order_consistent():
    """sin(theta) = lambda u, so tan(theta) = lambda u + O((lambda u)^3)."""
    lu = sp.Symbol("lu", real=True)
    theta = sp.asin(lu)
    series = sp.series(sp.tan(theta), lu, 0, 4).removeO()
    assert sp.simplify(series - lu - lu ** 3 / 2) == 0


# --------------------------------------------------------------------------
# 3. a hogel is an STFT; its intensity is a spectrogram
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed", [(48, 1), (64, 7), (37, 3)])
def test_spectrogram_factorises_into_ambiguity_functions(n, seed):
    """
    Derived, not guessed. Expanding |STFT|^2 and substituting t' = t - tau,
    then transforming over the window position x, gives

        F[a, k] = DFT_tau->k { A_field[tau, a] * conj(A_window[tau, a]) }

    Random field AND random window, so this is not a special case.
    """
    rng = np.random.default_rng(seed)
    field = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    window = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    transformed = np.fft.fft(spectrogram(field, window), axis=0)
    product = ambiguity(field) * np.conj(ambiguity(window))
    predicted = np.fft.fft(product, axis=0).T

    residual = np.abs(transformed - predicted).max() / np.abs(transformed).max()
    assert residual < 1e-12


def test_hogel_angular_sample_count_equals_subpixel_count():
    """
    Space-bandwidth conservation. A hogel of D sub-pixels at pitch p spans
    D*p, so its angular resolution is lambda/(D*p) while the grating limit is
    +/- lambda/(2p). The number of resolvable angles is therefore exactly D,
    independent of p and lambda. This is why, in a plain hogel grid, spatial
    and angular resolution are a single knob and not two.
    """
    lam = 532e-9
    for d in (32, 64, 128, 250, 512):
        for pitch in (0.5e-6, 0.706e-6):
            angular_span = 2 * lam / (2 * pitch)     # sin(theta) range
            angular_resolution = lam / (d * pitch)
            assert angular_span / angular_resolution == pytest.approx(d, rel=1e-12)


# --------------------------------------------------------------------------
# 4. the Gabor bound sets the exchange rate
# --------------------------------------------------------------------------

def test_gaussian_window_attains_the_gabor_bound_symbolically():
    """sigma_x * sigma_nu = 1/(4 pi) exactly, for any Gaussian width."""
    x, nu, sigma = sp.symbols("x nu sigma", positive=True)

    intensity = sp.exp(-x ** 2 / sigma ** 2)
    norm_x = sp.integrate(intensity, (x, -sp.oo, sp.oo))
    var_x = sp.integrate(x ** 2 * intensity, (x, -sp.oo, sp.oo)) / norm_x

    spectral = sp.exp(-4 * sp.pi ** 2 * sigma ** 2 * nu ** 2)
    norm_nu = sp.integrate(spectral, (nu, -sp.oo, sp.oo))
    var_nu = sp.integrate(nu ** 2 * spectral, (nu, -sp.oo, sp.oo)) / norm_nu

    product = sp.simplify(sp.sqrt(var_x * var_nu))
    assert sp.simplify(product - 1 / (4 * sp.pi)) == 0


def test_numeric_gaussian_window_matches_the_bound():
    n = 256
    index = np.fft.fftfreq(n, d=1.0 / n)
    for width in (3.0, 6.0, 12.0):
        window = np.exp(-(index ** 2) / (2 * width ** 2))

        p_x = window ** 2
        var_x = (p_x * index ** 2).sum() / p_x.sum()

        p_nu = np.abs(np.fft.fft(window)) ** 2
        freq = np.fft.fftfreq(n, d=1.0)
        var_nu = (p_nu * freq ** 2).sum() / p_nu.sum()

        assert np.sqrt(var_x * var_nu) == pytest.approx(1 / (4 * np.pi), rel=2e-2)


# --------------------------------------------------------------------------
# 5. coherent-mode decomposition and speckle contrast
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_coherent_mode_decomposition_reconstructs_the_coherence_function(seed):
    """
    Wolf's decomposition J = sum_n lambda_n phi_n phi_n^*. This is what makes
    the partially coherent formalism tractable: M coherent solves, not a
    squared-dimension object.
    """
    rng = np.random.default_rng(seed)
    n, n_modes = 24, 6
    modes = rng.standard_normal((n, n_modes)) + 1j * rng.standard_normal((n, n_modes))
    weights = rng.uniform(0.1, 1.0, n_modes)

    coherence = (modes * weights) @ modes.conj().T
    assert np.abs(coherence - coherence.conj().T).max() < 1e-10

    eigenvalues, eigenvectors = np.linalg.eigh(coherence)
    rebuilt = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
    assert np.abs(rebuilt - coherence).max() < 1e-9

    assert (eigenvalues > -1e-9).all()
    assert (eigenvalues > 1e-9).sum() == n_modes            # rank is the mode count

    intensity = np.real(np.diag(coherence))
    from_modes = (np.abs(eigenvectors) ** 2 * eigenvalues).sum(axis=1)
    assert np.abs(intensity - from_modes).max() < 1e-9


def test_speckle_contrast_is_one_over_root_m_symbolically():
    """
    A fully developed speckle intensity is negative-exponential. Averaging M
    independent realisations gives a Gamma(M) intensity, whose contrast
    sd/mean is 1/sqrt(M). This is why time multiplexing works, and it is the
    quantity our switching budget has to buy.
    """
    from sympy.stats import Gamma, E, variance

    m, scale = sp.symbols("m theta", positive=True)
    total = Gamma("I", m, scale)
    contrast = sp.simplify(sp.sqrt(variance(total)) / E(total))
    assert sp.simplify(contrast - 1 / sp.sqrt(m)) == 0

    single = Gamma("I1", 1, scale)
    assert sp.simplify(sp.sqrt(variance(single)) / E(single) - 1) == 0


@pytest.mark.parametrize("n_modes", [1, 4, 16, 64])
def test_speckle_contrast_monte_carlo(n_modes):
    rng = np.random.default_rng(11)
    trials = 200_000
    intensity = rng.exponential(1.0, size=(trials, n_modes)).mean(axis=1)
    contrast = intensity.std() / intensity.mean()
    assert contrast == pytest.approx(1 / np.sqrt(n_modes), rel=0.02)


# --------------------------------------------------------------------------
# 6. the foveation design rule that follows from 3 and 4
# --------------------------------------------------------------------------

def test_hogel_feasibility_condition():
    """
    A hogel grid of D sub-pixels at pitch p must satisfy two requirements at
    retinal eccentricity e:

        angular:  lambda / (D p) <= dtheta(e)    =>  D >= lambda / (p dtheta)
        spatial:  D p            <= dx(e)        =>  D <= dx / p

    A feasible D exists exactly when those bounds do not cross, i.e. when

        dx(e) * dtheta(e) >= lambda

    Acuity falls with eccentricity, so both dx and dtheta grow and the product
    grows: the constraint is tightest at the fovea and slackens outward. That
    is the formal reason foveating a hogel grid is possible at all.
    """
    lam, p = sp.symbols("lambda p", positive=True)
    dx, dtheta = sp.symbols("Delta_x Delta_theta", positive=True)

    lower = lam / (p * dtheta)
    upper = dx / p
    # the interval is non-empty iff lower <= upper
    assert sp.simplify(sp.solve(sp.Ge(upper, lower), dx)[0].rhs
                       if False else sp.simplify(upper - lower)) == \
        sp.simplify((dx * dtheta - lam) / (p * dtheta))

    # so feasibility is exactly dx * dtheta >= lambda
    feasible = sp.simplify((upper - lower) * p * dtheta)
    assert sp.simplify(feasible - (dx * dtheta - lam)) == 0


@pytest.mark.parametrize("lam,arcmin,ecc_deg", [
    (450e-9, 9.14, 18.7),
    (532e-9, 10.80, 22.5),
    (632e-9, 12.83, 27.2),
])
def test_our_hogel_grid_is_a_peripheral_grade_design(lam, arcmin, ecc_deg):
    """
    The uncertainty bound, evaluated on our actual specification.

    At 150 PPI the hogel pitch is 169.33 um, which at a 0.5 um sub-pixel pitch
    is D = 338.7 sub-pixels. Angular resolution is therefore lambda / 169.33 um,
    i.e. 9-13 arcmin depending on colour. Foveal acuity is about 1 arcmin, so
    the grid is roughly an order of magnitude short of it.

    Under Watson's 1/(1 + e/2.3) acuity falloff, 10.8 arcmin is exactly what the
    eye needs at about 22 degrees of eccentricity. So the uniform 150 PPI design
    is not an under-foveated foveal display; it is a correctly specified
    PERIPHERAL display that has no fovea. That inverts the foveation strategy:
    the gain is not in thinning the periphery, it is in adding a foveal inset.
    """
    pitch = 0.5e-6
    hogel_pitch = 25.4e-3 / 150

    assert hogel_pitch / pitch == pytest.approx(338.67, rel=1e-3)

    resolution = lam / hogel_pitch
    assert np.degrees(resolution) * 60 == pytest.approx(arcmin, rel=1e-3)

    foveal = np.radians(1 / 60)
    matched_eccentricity = 2.3 * (resolution / foveal - 1)
    assert matched_eccentricity == pytest.approx(ecc_deg, rel=2e-2)

    # Reaching 1 arcmin costs exactly that same factor in spatial resolution:
    # the hogel must grow by the arcmin ratio, so the PPI falls by it.
    required_hogel = lam / foveal
    assert required_hogel / hogel_pitch == pytest.approx(arcmin, rel=1e-3)
    assert 25.4e-3 / required_hogel == pytest.approx(150 / arcmin, rel=1e-3)
    assert 25.4e-3 / required_hogel < 17          # PPI, versus our 150


def test_spatial_and_angular_resolution_are_one_knob():
    """
    In a plain hogel grid the hogel pitch IS D * p, so choosing D fixes both
    the spatial sampling and the angular sampling. Their product is pinned at
    lambda regardless of D, which is the uncertainty bound restated.
    """
    lam, pitch = 532e-9, 0.5e-6
    for d in (64, 128, 338, 1024, 3658):
        spatial = d * pitch
        angular = lam / (d * pitch)
        assert spatial * angular == pytest.approx(lam, rel=1e-12)

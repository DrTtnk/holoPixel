"""
Hogel-free optimisation of a panel phase against a light field, no depth.

The target is a set of view INTENSITIES. No target phase is ever constructed,
so nothing has to supply one -- which is the whole point, since a depth map is
what supplies it in the original method and a depth map cannot hold glass,
smoke or a mirror.

The phase is therefore free, and that freedom is the resource. Measured in
scratchpad experiments and reported in docs: discarding it naively costs a
factor pi/4, choosing it at random recovers part, and optimising it recovers
more. The same freedom also pays for speckle (different phase per subframe
averages as 1/sqrt(M)) and for quantisation (levels can be approached rather
than straddled).

Geometry. The eye's pupil admits a sub-aperture of the panel and its lens
Fourier transforms what it admits, so a view is the squared magnitude of the
transform of one windowed region. Moving the pupil moves the window, which is
why views differ. This is the same windowed transform that defines a hogel
(`wft.py`); the difference is that the panel is optimised as ONE variable and
the windows only read it. That is what hogel-free means here.

The window width sets the angular span exactly: an n-sample window at pitch p
spans lam/p radians in total. At 1.524 um and 128 samples that is 20.0
degrees, chosen to match the field of view the Cornell light field was
rendered with, so no resampling of angles is needed anywhere.
"""

import numpy as np
import torch

from tier2_slfh.quantization_aware import nearest_index


def quantised_field(phase, levels=None):
    """
    The field the panel actually emits, given its free phase.

    `levels=None` is the continuous, unit-amplitude modulator every other
    function here assumes -- `exp(i*phase)` -- and is the default precisely
    so every existing caller and test is untouched.

    Given a `levels` (a `tier2_slfh.quantization_aware.LevelSet`, reused
    rather than reimplemented: that module already built and tested it
    against the real 8-level Sb2Se3/DBR device, both the idealised uniform
    case and the realistic one whose per-level reflectance runs 0.583-0.946),
    the panel instead emits `amplitude[level] * exp(i*phase[level])` for the
    nearest level to each pixel's continuous phase, through a straight-
    through estimator: the forward value is the true quantised field, the
    backward pass treats quantisation as the identity on phase, so gradients
    keep flowing through the one free phase variable exactly as if the panel
    were continuous. `nearest_index` is `tier2_slfh`'s own quantiser,
    reused as-is rather than re-derived here.
    """
    if levels is None:
        return torch.exp(1j * phase)
    level_phase = levels.phase.to(device=phase.device, dtype=phase.dtype)
    level_amplitude = levels.amplitude.to(device=phase.device, dtype=phase.dtype)
    idx = nearest_index(phase, level_phase)
    q_phase = level_phase[idx]
    q_amplitude = level_amplitude[idx]
    phase_ste = phase + (q_phase - phase).detach()
    complex_dtype = torch.complex128 if phase.dtype == torch.float64 else torch.complex64
    return q_amplitude.detach().to(complex_dtype) * torch.exp(1j * phase_ste.to(complex_dtype))


class Geometry:
    """
    Panel and pupil sampling, with the angular span pinned to the target.

    `dtype` is the one real floating dtype for everything derived here (the
    aperture, and the phase and Adam state that `optimise` builds from it):
    float64 for the CPU path already covered by tests, float32 to fit a GPU.
    It is carried on the geometry, not passed again at each call site, so a
    float32 geometry cannot silently pick up a float64 tensor from elsewhere
    and fall back to float64 by promotion.
    """

    def __init__(self, panel=512, window=128, pitch=1.524e-6, wavelength=532e-9,
                 n_views=17, device="cpu", dtype=torch.float64):
        self.panel = panel
        self.window = window
        self.pitch = pitch
        self.wavelength = wavelength
        self.n_views = n_views
        self.device = device
        self.dtype = dtype

        self.fov_rad = wavelength / pitch          # total angular span of a view
        travel = panel - window
        if travel <= 0:
            raise ValueError(f"window {window} must be smaller than panel {panel}")
        centres = torch.linspace(0, travel, n_views, device=device).round().long()
        self.offsets = torch.stack(
            torch.meshgrid(centres, centres, indexing="ij"), dim=-1).reshape(-1, 2)

        y = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2
        r2 = y[:, None] ** 2 + y[None, :] ** 2
        self.aperture = (r2 <= (window / 2) ** 2).to(dtype)

    @property
    def fov_deg(self):
        return float(np.degrees(self.fov_rad))


def render_views(phase, geom, view_idx, chunk=None, levels=None):
    """
    Intensities seen from the given pupil positions. `phase` is the whole panel
    and is a single optimisation variable; the windows only read from it.

    `phase` may carry a leading mode axis. Modes are separate subframes shown
    faster than the eye integrates, so they are mutually incoherent and their
    INTENSITIES add. That is the whole mechanism by which speckle falls as
    1/sqrt(M): each mode is free to choose a different phase for the same
    light field, and the realisations average.

    `levels` is forwarded to `quantised_field`: `None` (the default) is the
    continuous unit-amplitude panel every caller already assumes; a
    `LevelSet` makes the panel emit its real 8-level quantised field instead,
    with the same phase-to-window plumbing below untouched, since
    quantisation is a per-pixel nonlinearity and commutes with windowing.

    `chunk` splits `view_idx` into groups of at most that many views, each
    written into its slice of a pre-allocated output tensor. It changes only
    memory, never the result. Collecting a list of per-group results and
    concatenating them at the end would keep every earlier group's tensor
    alive until the last one finishes, so peak memory would still scale with
    the total view count -- exactly the cost chunking exists to avoid, which
    is what turns an occasional all-views evaluation (used for logging) into
    the binding memory constraint at a large window, well before training
    itself is.
    """
    single = phase.ndim == 2
    ph = phase[None] if single else phase
    view_idx = view_idx.to(geom.offsets.device)
    n = view_idx.shape[0]
    step = n if chunk is None else min(chunk, n)

    field = quantised_field(ph, levels)
    out = torch.empty(n, geom.window, geom.window, dtype=ph.dtype, device=field.device)
    for start in range(0, n, step):
        g = view_idx[start:start + step]
        tiles = torch.stack([
            field[:, oy:oy + geom.window, ox:ox + geom.window]
            for oy, ox in geom.offsets[g].tolist()
        ], dim=1)                                          # (modes, views, w, w)
        spectra = torch.fft.fftshift(torch.fft.fft2(tiles * geom.aperture), dim=(-2, -1))
        out[start:start + g.shape[0]] = (torch.abs(spectra) ** 2).mean(dim=0)
    return out


def _normalise(x):
    """Absolute brightness is a free scale, so compare shapes, not levels."""
    return x / x.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-30)


def view_loss(rendered, target, compress=1.0):
    """
    Squared error between views, optionally on a compressed intensity scale.

    `compress=1.0` compares raw intensity, which a bright highlight dominates:
    in the Cornell scene the ceiling light is about 50x the walls, so it
    outweighs the rest of the image by around 2500x and the optimiser
    reproduces the lamp and abandons the room. `compress=0.5` compares
    amplitudes, which is the usual choice in computer-generated holography and
    is also closer to how brightness is perceived.
    """
    a, b = _normalise(rendered), _normalise(target)
    if compress != 1.0:
        a = a.clamp_min(0) ** compress
        b = b.clamp_min(0) ** compress
    return ((a - b) ** 2).sum(dim=(-2, -1)).mean()


def psnr(rendered, target):
    a, b = _normalise(rendered), _normalise(target)
    mse = ((a - b) ** 2).mean(dim=(-2, -1))
    peak = b.amax(dim=(-2, -1))
    return (10 * torch.log10(peak ** 2 / mse)).mean().item()


def full_psnr(phase, geom, target_views, chunk, levels=None):
    """
    PSNR averaged over every view in `target_views`, one chunk of views at a
    time. Exactly equal to `psnr(render_views(phase, geom, idx), target_views)`
    for `idx` covering every view, because PSNR is already an unweighted
    average of a per-view quantity -- accumulating that average one chunk at
    a time changes nothing but how it is computed.

    This exists because that direct call would render and hold every view at
    once just to reduce them to one number afterwards: for a periodic
    training-time log, that peak -- proportional to the view count times the
    window area -- is the real memory ceiling at a large window, well before
    the training step itself is. `render_views(..., chunk=...)` alone does
    not fix this, because its output buffer is still sized for every view;
    only reducing each chunk before moving to the next one bounds memory by
    `chunk` rather than by the view count. The same streaming applies to
    `target_views` itself: it is read one chunk-sized slice at a time from
    wherever it lives, never moved to `geom.device` as a whole (see
    `optimise`'s docstring for why that whole-array move does not fit at a
    large window).
    """
    n = target_views.shape[0]
    idx_all = torch.arange(n)
    total = 0.0
    for start in range(0, n, chunk):
        idx = idx_all[start:start + chunk]
        rendered = render_views(phase, geom, idx, levels=levels)
        batch = target_views[idx].to(device=geom.device, dtype=geom.dtype)
        a, b = _normalise(rendered), _normalise(batch)
        mse = ((a - b) ** 2).mean(dim=(-2, -1))
        peak = b.amax(dim=(-2, -1))
        total += (10 * torch.log10(peak ** 2 / mse)).sum().item()
    return total / n


def optimise(target_views, geom, iters=1000, views_per_iter=8, lr=0.05, seed=0,
             log_every=100, log=print, compress=0.5, n_modes=1, levels=None):
    """
    Adam on the whole panel phase, with stochastic pupil sampling: each step
    sees a handful of the views rather than all of them, which is what makes
    the cost independent of how many views the target has.

    `levels=None` (the default) optimises the continuous, unit-amplitude
    panel every existing call site assumes -- so nothing here changes for
    them. Passing a `LevelSet` makes this quantisation-AWARE: every forward
    pass during training, not just the final readout, goes through the real
    8-level quantised field (straight-through estimator, see
    `quantised_field`), so the free phase can route around the device's
    quantisation -- including its non-uniform per-level amplitude -- rather
    than discovering it only after optimisation has already committed to a
    continuous solution (that discover-it-only-at-the-end path is "naive":
    call `optimise` with `levels=None` and quantise the returned phase
    separately).

    The phase dtype and device come from `geom`, not from a second parameter
    here, so there is one place that decides float64-CPU versus float32-GPU.

    `target_views` is left on whatever device the caller put it on, and only
    the dtype is matched to `geom`; each step moves just the small slice of
    it that step needs. Casting the WHOLE target to `geom.device` up front
    would be simpler, but the target is n_views * window^2 real numbers with
    no way to shrink it, and at a large window that alone can exceed a GPU's
    free memory before the panel is even involved -- 289 views at a
    2048-pixel window is 4.8 GB in float32. Keeping the target on the CPU and
    streaming per-iteration slices is what lets a window that large train at
    all on a memory-constrained GPU; a target that already fits on the GPU
    (the common, smaller-scale case) pays only a same-device no-op here.
    """
    target_views = target_views.to(dtype=geom.dtype)
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (n_modes, geom.panel, geom.panel) if n_modes > 1 else (geom.panel, geom.panel)
    phase = (torch.rand(*shape, generator=g, dtype=geom.dtype)
             * 2 * np.pi).to(geom.device).requires_grad_(True)
    opt = torch.optim.Adam([phase], lr=lr)
    n_views = len(geom.offsets)
    history = []

    for it in range(iters + 1):
        idx = torch.randperm(n_views, generator=g)[:views_per_iter]
        rendered = render_views(phase, geom, idx, levels=levels)
        batch = target_views[idx.to(target_views.device)].to(geom.device)
        loss = view_loss(rendered, batch, compress)

        if it % log_every == 0:
            with torch.no_grad():
                p = full_psnr(phase, geom, target_views, chunk=views_per_iter, levels=levels)
            history.append((it, loss.item(), p))
            log(f"  iter {it:5d}  loss {loss.item():.6f}  PSNR {p:6.2f} dB")

        if it == iters:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()

    return phase.detach(), history

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


class Geometry:
    """Panel and pupil sampling, with the angular span pinned to the target."""

    def __init__(self, panel=512, window=128, pitch=1.524e-6, wavelength=532e-9,
                 n_views=17, device="cpu"):
        self.panel = panel
        self.window = window
        self.pitch = pitch
        self.wavelength = wavelength
        self.n_views = n_views
        self.device = device

        self.fov_rad = wavelength / pitch          # total angular span of a view
        travel = panel - window
        if travel <= 0:
            raise ValueError(f"window {window} must be smaller than panel {panel}")
        centres = torch.linspace(0, travel, n_views, device=device).round().long()
        self.offsets = torch.stack(
            torch.meshgrid(centres, centres, indexing="ij"), dim=-1).reshape(-1, 2)

        y = torch.arange(window, device=device, dtype=torch.float64) - (window - 1) / 2
        r2 = y[:, None] ** 2 + y[None, :] ** 2
        self.aperture = (r2 <= (window / 2) ** 2).to(torch.float64)

    @property
    def fov_deg(self):
        return float(np.degrees(self.fov_rad))


def render_views(phase, geom, view_idx):
    """
    Intensities seen from the given pupil positions. `phase` is the whole panel
    and is a single optimisation variable; the windows only read from it.

    `phase` may carry a leading mode axis. Modes are separate subframes shown
    faster than the eye integrates, so they are mutually incoherent and their
    INTENSITIES add. That is the whole mechanism by which speckle falls as
    1/sqrt(M): each mode is free to choose a different phase for the same
    light field, and the realisations average.
    """
    single = phase.ndim == 2
    ph = phase[None] if single else phase
    field = torch.exp(1j * ph)
    tiles = torch.stack([
        field[:, oy:oy + geom.window, ox:ox + geom.window]
        for oy, ox in geom.offsets[view_idx].tolist()
    ], dim=1)                                          # (modes, views, w, w)
    spectra = torch.fft.fftshift(torch.fft.fft2(tiles * geom.aperture), dim=(-2, -1))
    return (torch.abs(spectra) ** 2).mean(dim=0)


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


def optimise(target_views, geom, iters=1000, views_per_iter=8, lr=0.05, seed=0,
             log_every=100, log=print, compress=0.5, n_modes=1):
    """
    Adam on the whole panel phase, with stochastic pupil sampling: each step
    sees a handful of the views rather than all of them, which is what makes
    the cost independent of how many views the target has.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (n_modes, geom.panel, geom.panel) if n_modes > 1 else (geom.panel, geom.panel)
    phase = (torch.rand(*shape, generator=g, dtype=torch.float64)
             * 2 * np.pi).to(geom.device).requires_grad_(True)
    opt = torch.optim.Adam([phase], lr=lr)
    n_views = len(geom.offsets)
    history = []

    for it in range(iters + 1):
        idx = torch.randperm(n_views, generator=g)[:views_per_iter]
        rendered = render_views(phase, geom, idx)
        loss = view_loss(rendered, target_views[idx], compress)

        if it % log_every == 0:
            with torch.no_grad():
                full = render_views(phase, geom, torch.arange(n_views))
                p = psnr(full, target_views)
            history.append((it, loss.item(), p))
            log(f"  iter {it:5d}  loss {loss.item():.6f}  PSNR {p:6.2f} dB")

        if it == iters:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()

    return phase.detach(), history

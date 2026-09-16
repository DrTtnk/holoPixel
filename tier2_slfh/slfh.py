"""
Stochastic Light Field Holography (SLFH) Optimizer

Implements the SLFH algorithm from Schiffers et al.:
  - Angular Spectrum propagation forward model (Eq. 16-19)
  - Light Field photographic projection (Eq. 3)
  - Stochastic pupil sampling optimization (Eq. 14-15)

Input:  4D Light Field L(v, u, y, x, c) from path tracer
Output: Phase-only SLM pattern φ(y, x) per wavelength
"""

import torch
import torch.fft as fft
import numpy as np
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Physical parameters matching our Cornell box setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WAVELENGTHS = {
    "R": 632e-9,
    "G": 532e-9,
    "B": 450e-9,
}

PIXEL_PITCH = 8e-6          # SLM pixel pitch (8 µm)
FOCAL_LENGTH = 0.4          # relay lens focal length (400mm)
EYEBOX_SIZE = 0.02          # 20mm eyebox (matches our LF angular_extent)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Forward Model: Angular Spectrum Propagation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def angular_spectrum_kernel(N, pixel_pitch, wavelength, z, device=DEVICE):
    """Compute the Angular Spectrum propagation kernel H(q; z).

    H(q; z) = exp(j * 2π/λ * sqrt(1 - ||λq||²) * z)  if ||q|| < 1/λ
              0                                          otherwise

    Args:
        N: grid size (assumes square)
        pixel_pitch: SLM pixel pitch [m]
        wavelength: illumination wavelength [m]
        z: propagation distance [m]
        device: torch device

    Returns:
        H: (N, N) complex tensor — frequency-domain propagation kernel
    """
    freq = torch.fft.fftfreq(N, d=pixel_pitch, device=device)
    qx, qy = torch.meshgrid(freq, freq, indexing="ij")
    q_sq = qx ** 2 + qy ** 2

    # Evanescent wave cutoff
    cutoff = 1.0 / wavelength
    propagating = q_sq < cutoff ** 2

    # Phase: 2π/λ * sqrt(1 - (λ²)(qx²+qy²)) * z
    arg = torch.zeros_like(q_sq)
    arg[propagating] = (2 * math.pi / wavelength) * torch.sqrt(
        1.0 - (wavelength ** 2) * q_sq[propagating]
    ) * z

    H = torch.zeros(N, N, dtype=torch.complex64, device=device)
    H[propagating] = torch.exp(1j * arg[propagating])

    return H


def circular_aperture(N, pixel_pitch, wavelength, focal_length,
                      shift_x, shift_y, diameter, device=DEVICE):
    """Circular aperture in frequency domain: A(q - s/(λf), d/(λf)).

    Args:
        N: grid size
        pixel_pitch: SLM pixel pitch [m]
        wavelength: wavelength [m]
        focal_length: relay lens focal length [m]
        shift_x, shift_y: pupil shift in spatial coords [m]
        diameter: pupil diameter [m]
        device: torch device

    Returns:
        A: (N, N) float tensor — binary aperture mask in freq domain
    """
    freq = torch.fft.fftfreq(N, d=pixel_pitch, device=device)
    qx, qy = torch.meshgrid(freq, freq, indexing="ij")

    # Shift in frequency domain: s / (λf)
    sx = shift_x / (wavelength * focal_length)
    sy = shift_y / (wavelength * focal_length)

    # Aperture radius in frequency domain: d / (2λf)
    radius = diameter / (2.0 * wavelength * focal_length)

    dist_sq = (qx - sx) ** 2 + (qy - sy) ** 2
    A = (dist_sq < radius ** 2).float()

    return A


def wigner_projection(phase, wavelength, pixel_pitch, focal_length,
                      pupil_shift_x, pupil_shift_y, pupil_diameter,
                      defocus_z, device=DEVICE):
    """Wigner projection via wave optics (Eq. 16).

    P_wd[u(r), p](r) = |IFFT{ U(q) · K(q,p) }|²

    where:
        U(q) = FFT{exp(jφ(r))}
        K(q,p) = A(q - s/(λf), d/(λf)) · H(q; z)

    Args:
        phase: (N, N) tensor — SLM phase pattern φ
        wavelength: scalar [m]
        pixel_pitch: scalar [m]
        focal_length: scalar [m]
        pupil_shift_x, pupil_shift_y: pupil shift [m]
        pupil_diameter: pupil diameter [m]
        defocus_z: defocus distance [m]
        device: torch device

    Returns:
        intensity: (N, N) tensor — projected intensity image
    """
    N = phase.shape[0]

    # SLM field: u(r) = exp(jφ)
    u = torch.exp(1j * phase.to(torch.float32))

    # U(q) = FFT{u(r)}
    U = fft.fft2(u)

    # Kernel K(q,p) = A(...) * H(q; z)
    A = circular_aperture(N, pixel_pitch, wavelength, focal_length,
                          pupil_shift_x, pupil_shift_y, pupil_diameter,
                          device=device)
    H = angular_spectrum_kernel(N, pixel_pitch, wavelength, defocus_z,
                                device=device)

    K = A * H

    # IFFT{ U(q) · K(q,p) }
    field = fft.ifft2(U * K)

    # Intensity = |field|²
    intensity = torch.abs(field) ** 2

    return intensity


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Light Field Projection Operator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def lightfield_projection(lightfield, angular_positions,
                          pupil_shift_x, pupil_shift_y,
                          pupil_diameter, defocus_z,
                          focal_length, device=DEVICE):
    """Light Field photographic projection (Eq. 3).

    P_lf[L; p](r) = ∫ A(q;p) · L(q - (q-r)/α, q) dq

    Implemented as: shear the LF by α, mask with aperture, sum over angular dims.

    Args:
        lightfield: (nv, nu, H, W) tensor — 4D light field (monochrome)
        angular_positions: (n_angular,) array — angular sample positions [m]
        pupil_shift_x, pupil_shift_y: pupil center [m]
        pupil_diameter: pupil diameter [m]
        defocus_z: defocus distance [m]
        focal_length: relay lens focal length [m]
        device: torch device

    Returns:
        image: (H, W) tensor — projected photograph
    """
    nv, nu, H, W = lightfield.shape
    alpha = focal_length / (defocus_z + focal_length)

    image = torch.zeros(H, W, device=device)
    weight = torch.zeros(H, W, device=device)

    pupil_radius = pupil_diameter / 2.0

    for vi in range(nv):
        for ui in range(nu):
            qx = angular_positions[ui]
            qy = angular_positions[vi]

            # Aperture test: is this angular sample inside the pupil?
            dist_sq = (qx - pupil_shift_x) ** 2 + (qy - pupil_shift_y) ** 2
            if dist_sq > pupil_radius ** 2:
                continue

            # Shear: shift spatial coords by (q-r)/α → implemented as grid warp
            # The LF view at (qx, qy) is shifted by (1 - 1/α) * (qx, qy)
            # in pixel coordinates
            shift_x_px = (1.0 - 1.0 / alpha) * qx / (PIXEL_PITCH * W) * W
            shift_y_px = (1.0 - 1.0 / alpha) * qy / (PIXEL_PITCH * H) * H

            view = lightfield[vi, ui]  # (H, W)

            # Sub-pixel shift via grid_sample
            shift_grid_x = torch.linspace(-1, 1, W, device=device)
            shift_grid_y = torch.linspace(-1, 1, H, device=device)
            gy, gx = torch.meshgrid(shift_grid_y, shift_grid_x, indexing="ij")

            # Convert pixel shift to normalized grid coords [-1, 1]
            gx = gx - 2.0 * shift_x_px / W
            gy = gy - 2.0 * shift_y_px / H

            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            view_4d = view.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            shifted = torch.nn.functional.grid_sample(
                view_4d, grid, mode="bilinear", padding_mode="zeros",
                align_corners=True
            ).squeeze()

            image += shifted
            weight += 1.0

    # Normalize by number of contributing views
    image = torch.where(weight > 0, image / weight, image)
    return image


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pupil Sampling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sample_pupil(eyebox_size, z_min, z_max, d_min, d_max):
    """Stochastic pupil sampling (Eq. 15).

    Returns dict with pupil parameters.
    """
    half_eb = eyebox_size / 2.0
    return {
        "shift_x": np.random.uniform(-half_eb, half_eb),
        "shift_y": np.random.uniform(-half_eb, half_eb),
        "defocus_z": np.random.uniform(z_min, z_max),
        "diameter": np.random.uniform(d_min, d_max),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLFH Optimizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def optimize_hologram(lightfield_rgb, angular_positions,
                      n_iters=1000, lr=0.03,
                      n_pupils_per_iter=4,
                      n_frames=8,
                      z_min=0.0, z_max=0.015,
                      d_min=0.002, d_max=0.020):
    """SLFH optimization loop (Eq. 14) with temporal averaging.

    Optimizes n_frames independent phase patterns per channel.
    Each frame uses a different random initialization to produce
    uncorrelated speckle, which averages out during display.

    Returns:
        phases: dict of {channel: (n_frames, H, W) numpy array}
    """
    nv, nu, H, W, _ = lightfield_rgb.shape

    channel_names = ["R", "G", "B"]
    phases = {}

    for ci, ch in enumerate(channel_names):
        wavelength = WAVELENGTHS[ch]
        print(f"\n  Optimizing channel {ch} (λ={wavelength*1e9:.0f}nm), "
              f"{n_frames} frames...")

        lf_ch = torch.tensor(
            lightfield_rgb[:, :, :, :, ci],
            dtype=torch.float32, device=DEVICE
        )

        ch_phases = np.zeros((n_frames, H, W))

        for frame in range(n_frames):
            phase = torch.randn(H, W, device=DEVICE) * 0.1
            phase.requires_grad_(True)

            optimizer = torch.optim.Adam([phase], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_iters
            )

            for it in range(n_iters):
                optimizer.zero_grad()
                loss_total = 0.0

                for _ in range(n_pupils_per_iter):
                    pupil = sample_pupil(EYEBOX_SIZE, z_min, z_max, d_min, d_max)

                    with torch.no_grad():
                        target = lightfield_projection(
                            lf_ch, angular_positions,
                            pupil["shift_x"], pupil["shift_y"],
                            pupil["diameter"], pupil["defocus_z"],
                            FOCAL_LENGTH, device=DEVICE
                        )

                    output = wigner_projection(
                        phase, wavelength, PIXEL_PITCH, FOCAL_LENGTH,
                        pupil["shift_x"], pupil["shift_y"],
                        pupil["diameter"], pupil["defocus_z"],
                        device=DEVICE
                    )

                    loss = torch.mean((output - target) ** 2)
                    loss_total += loss

                loss_total /= n_pupils_per_iter
                loss_total.backward()
                optimizer.step()
                scheduler.step()

            cur_loss = loss_total.item()
            print(f"    frame {frame+1}/{n_frames}: final loss={cur_loss:.6f}")
            ch_phases[frame] = phase.detach().cpu().numpy()

        phases[ch] = ch_phases

    return phases


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def reconstruct_rgb(phases, wavelength_map, pixel_pitch, focal_length,
                    shift_x, shift_y, diameter, defocus_z,
                    spatial_avg=2, device=DEVICE):
    """Reconstruct RGB with temporal averaging (over frames) + spatial averaging.

    phases: dict of {channel: (n_frames, H, W)} or {channel: (H, W)}
    spatial_avg: kernel size for spatial averaging (2 = 2×2 binning)
    """
    channels = ["R", "G", "B"]
    first = phases[channels[0]]
    if first.ndim == 2:
        n_frames, N = 1, first.shape[0]
    else:
        n_frames, N = first.shape[0], first.shape[1]

    rgb = np.zeros((N, N, 3))

    for ci, ch in enumerate(channels):
        ch_data = phases[ch]
        if ch_data.ndim == 2:
            ch_data = ch_data[np.newaxis, :, :]

        acc = torch.zeros(N, N, device=device)

        for fi in range(n_frames):
            phase_t = torch.tensor(ch_data[fi], dtype=torch.float32, device=device)
            with torch.no_grad():
                img = wigner_projection(
                    phase_t, wavelength_map[ch], pixel_pitch, focal_length,
                    shift_x, shift_y, diameter, defocus_z, device=device
                )
            acc += img

        # Temporal average
        acc /= n_frames

        # Spatial averaging (2×2 block mean, replicated back to full res)
        if spatial_avg > 1:
            k = spatial_avg
            acc_4d = acc.unsqueeze(0).unsqueeze(0)
            acc = torch.nn.functional.avg_pool2d(acc_4d, k, stride=k)
            acc = torch.nn.functional.interpolate(
                acc, size=(N, N), mode="nearest"
            ).squeeze()

        rgb[:, :, ci] = acc.cpu().numpy()

    return rgb


def tonemap_recon(rgb):
    """Percentile-based normalization to avoid speckle hot-pixel domination."""
    p99 = np.percentile(rgb[rgb > 0], 99) if np.any(rgb > 0) else 1.0
    return np.clip(rgb / (p99 + 1e-10), 0, 1)


def main():
    print("=" * 60)
    print("Stochastic Light Field Holography (SLFH) Optimizer")
    print("=" * 60)

    # Load light field from path tracer
    lf_path = PLOT_DIR / "cornell_lightfield.npz"
    data = np.load(str(lf_path))
    lightfield = data["lightfield"]
    angular_positions = data["angular_positions"]

    print(f"  Light field: {lightfield.shape}")
    print(f"  Angular positions: {angular_positions}")
    print(f"  Device: {DEVICE}")

    # Tone-map the light field to [0, 1] for optimization targets
    exposure = 2.0
    lf_mapped = 1.0 - np.exp(-lightfield * exposure)
    lf_mapped = np.clip(lf_mapped, 0, 1)

    n_frames = 8
    t0 = time.time()
    phases = optimize_hologram(
        lf_mapped, angular_positions,
        n_iters=1000, lr=0.03,
        n_pupils_per_iter=4,
        n_frames=n_frames,
    )
    dt = time.time() - t0
    print(f"\n  Optimization: {dt:.1f}s ({dt/60:.1f} min) "
          f"({n_frames} frames × 3 channels)")

    # Save phase patterns
    out_path = PLOT_DIR / "slfh_phases.npz"
    np.savez_compressed(str(out_path), **phases)
    print(f"✅ Saved: {out_path}")

    N = lightfield.shape[2]

    # ── Plot 1: Phase patterns (frame 0) + center reconstruction ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ci, ch in enumerate(["R", "G", "B"]):
        ax = axes[ci]
        ax.imshow(phases[ch][0], cmap="twilight", vmin=-math.pi, vmax=math.pi)
        ax.set_title(f"Phase φ ({ch}, λ={WAVELENGTHS[ch]*1e9:.0f}nm) frame 0")
        ax.axis("off")

    recon_center = reconstruct_rgb(phases, WAVELENGTHS, PIXEL_PITCH, FOCAL_LENGTH,
                                   0.0, 0.0, EYEBOX_SIZE * 0.4, 0.005)
    axes[3].imshow(tonemap_recon(recon_center))
    axes[3].set_title(f"Center view ({n_frames}-frame avg + 2×2)")
    axes[3].axis("off")
    plt.suptitle("SLFH Phase Patterns", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out1 = PLOT_DIR / "slfh_phases_vis.png"
    fig.savefig(str(out1), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out1}")

    # ── Plot 2: Multi-view reconstruction grid ──
    n_views = 5
    shifts = np.linspace(-EYEBOX_SIZE * 0.4, EYEBOX_SIZE * 0.4, n_views)
    fig, axes = plt.subplots(n_views, n_views, figsize=(2.5 * n_views, 2.5 * n_views))

    for vi, sy in enumerate(shifts):
        for ui, sx in enumerate(shifts):
            rgb = reconstruct_rgb(
                phases, WAVELENGTHS, PIXEL_PITCH, FOCAL_LENGTH,
                sx, sy, EYEBOX_SIZE * 0.3, 0.005
            )
            ax = axes[vi, ui]
            ax.imshow(tonemap_recon(rgb))
            ax.axis("off")
            if vi == 0:
                ax.set_title(f"x={sx*1e3:.1f}mm", fontsize=8)
            if ui == 0:
                ax.set_ylabel(f"y={sy*1e3:.1f}mm", fontsize=8)

    plt.suptitle(f"Multi-View Holographic Reconstruction "
                 f"({n_frames}-frame temporal + 2×2 spatial avg)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = PLOT_DIR / "slfh_multiview.png"
    fig.savefig(str(out2), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out2}")

    # ── Plot 3: Physical display simulation ──
    print("\n  Simulating physical display (RGB laser front-lit)...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    focus_distances = [0.0, 0.005, 0.010]
    focus_labels = ["In focus (0mm)", "Mid defocus (5mm)", "Far defocus (10mm)"]

    for fi, (z, label) in enumerate(zip(focus_distances, focus_labels)):
        rgb = reconstruct_rgb(
            phases, WAVELENGTHS, PIXEL_PITCH, FOCAL_LENGTH,
            0.0, 0.0, EYEBOX_SIZE * 0.4, z
        )
        axes[0, fi].imshow(tonemap_recon(rgb))
        axes[0, fi].set_title(f"Center view — {label}")
        axes[0, fi].axis("off")

    parallax_shifts = [(-0.006, 0.0), (0.0, 0.0), (0.006, 0.0)]
    parallax_labels = ["Left eye (−6mm)", "Center", "Right eye (+6mm)"]

    for pi, ((sx, sy), label) in enumerate(zip(parallax_shifts, parallax_labels)):
        rgb = reconstruct_rgb(
            phases, WAVELENGTHS, PIXEL_PITCH, FOCAL_LENGTH,
            sx, sy, EYEBOX_SIZE * 0.3, 0.005
        )
        axes[1, pi].imshow(tonemap_recon(rgb))
        axes[1, pi].set_title(f"Parallax — {label}")
        axes[1, pi].axis("off")

    active_mm = N * PIXEL_PITCH * 1e3
    plt.suptitle(
        f"Physical Display: {active_mm:.1f}mm × {active_mm:.1f}mm SLM, "
        f"8µm pitch, RGB lasers (632/532/450nm)\n"
        f"Eyebox: {EYEBOX_SIZE*1e3:.0f}mm, f={FOCAL_LENGTH*1e3:.0f}mm, "
        f"{n_frames}-frame temporal avg + 2×2 spatial avg",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out3 = PLOT_DIR / "slfh_display_sim.png"
    fig.savefig(str(out3), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out3}")


if __name__ == "__main__":
    main()

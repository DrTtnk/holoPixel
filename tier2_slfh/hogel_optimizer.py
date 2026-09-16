"""
Single Hogel Hologram Optimizer

Simulates one holographic pixel (hogel) from the holoPixel architecture:
  - 250×250 active phase modulators (waveguides)
  - 0.706 µm sub-pixel pitch (144 DPI, 176.4 µm hogel)
  - Far-field (Fraunhofer) diffraction → angular intensity
  - Phase retrieval via gradient descent (ADAM)

The hogel encodes the angular distribution of light at one spatial
position, extracted from the path-traced light field.
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hogel physical parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

N_SUB = 250               # sub-pixels per axis
PITCH = 0.706e-6          # sub-pixel pitch [m] (176.4µm hogel / 250)
HOGEL_SIZE = N_SUB * PITCH  # 176.4 µm
DPI = 144

WAVELENGTHS = {
    "R": 632e-9,
    "G": 532e-9,
    "B": 450e-9,
}

# Max diffraction angles: sin(θ) = λ/(2p)
for ch, wl in WAVELENGTHS.items():
    theta_max = math.degrees(math.asin(min(1.0, wl / (2 * PITCH))))
    print(f"  {ch} (λ={wl*1e9:.0f}nm): ±{theta_max:.1f}° FoV")

# Angular resolution: Δθ = λ/D_hogel
for ch, wl in WAVELENGTHS.items():
    d_theta = math.degrees(wl / HOGEL_SIZE)
    print(f"  {ch} angular resolution: {d_theta:.3f}°")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Far-field forward model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hogel_farfield(phase, wavelength):
    """Compute far-field angular intensity from hogel phase pattern.

    I(θ_x, θ_y) = |FFT{exp(jφ(x,y))}|²

    The FFT maps spatial coordinates to angular frequencies.
    Frequency q maps to angle θ via sin(θ) = λq.

    Args:
        phase: (N, N) tensor — phase pattern [radians]
        wavelength: scalar [m]

    Returns:
        intensity: (N, N) tensor — angular intensity distribution
        angles: (N,) tensor — angle axis in degrees
    """
    u = torch.exp(1j * phase.to(torch.float32))
    U = fft.fftshift(fft.fft2(u))
    intensity = torch.abs(U) ** 2

    # Map frequency bins to angles
    freq = torch.fft.fftshift(torch.fft.fftfreq(phase.shape[0], d=PITCH))
    # sin(θ) = λ * freq → θ = arcsin(λ * freq)
    sin_theta = wavelength * freq.cpu().numpy()
    # Clip to valid range
    sin_theta = np.clip(sin_theta, -1, 1)
    angles_deg = np.degrees(np.arcsin(sin_theta))

    return intensity, angles_deg


def hogel_farfield_padded(phase, wavelength, pad_factor=4):
    """Far-field with zero-padding for angular super-resolution.

    Padding interpolates the far-field pattern for smoother visualization.
    """
    N = phase.shape[0]
    N_pad = N * pad_factor
    u = torch.exp(1j * phase.to(torch.float32))
    u_pad = torch.zeros(N_pad, N_pad, dtype=torch.complex64, device=phase.device)
    offset = (N_pad - N) // 2
    u_pad[offset:offset + N, offset:offset + N] = u
    U = fft.fftshift(fft.fft2(u_pad))
    intensity = torch.abs(U) ** 2

    freq = torch.fft.fftshift(torch.fft.fftfreq(N_pad, d=PITCH))
    sin_theta = wavelength * freq.cpu().numpy()
    sin_theta = np.clip(sin_theta, -1, 1)
    angles_deg = np.degrees(np.arcsin(sin_theta))

    return intensity, angles_deg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Angular target from light field
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_hogel_angular_samples(lightfield, angular_positions, pixel_y, pixel_x):
    """Extract angular intensity samples for one hogel from the light field.

    Returns per-channel angular samples and their corresponding angles.
    """
    nv, nu = lightfield.shape[:2]
    cam_z = 0.2
    Z0 = 0.1

    # LF angular positions → viewing angles [radians]
    lf_angles = np.arctan2(angular_positions, cam_z - Z0)

    # Angular intensities at this pixel for each channel
    samples = {}
    for ci, ch in enumerate(["R", "G", "B"]):
        samples[ch] = lightfield[:, :, pixel_y, pixel_x, ci]  # (nv, nu)

    return samples, lf_angles


def angle_to_fft_bin(angle_rad, wavelength, pitch, N):
    """Convert a viewing angle to an FFT frequency bin index.

    sin(θ) = λ·freq → freq = sin(θ)/λ
    bin = freq · N · pitch  (in fftshift coordinates)
    """
    freq = math.sin(angle_rad) / wavelength
    bin_idx = freq * N * pitch + N / 2  # fftshift center = N/2
    return bin_idx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hogel phase optimizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def optimize_hogel(angular_target, lf_angles, wavelength,
                   n_iters=3000, lr=0.1):
    """Optimize hogel phase to match angular intensity at specific angles.

    Computes loss only at the FFT bins corresponding to the 9×9 LF angles.
    This avoids fighting energy conservation over the full FFT.

    Args:
        angular_target: (nv, nu) numpy array — target intensities at LF angles
        lf_angles: (n_angular,) array — viewing angles [radians]
        wavelength: scalar [m]
        n_iters: iterations
        lr: learning rate

    Returns:
        phase: (N_SUB, N_SUB) numpy array
    """
    nv, nu = angular_target.shape

    # Pre-compute FFT bin indices for each angular sample
    bin_y = []
    bin_x = []
    target_vals = []

    for vi in range(nv):
        for ui in range(nu):
            by = angle_to_fft_bin(lf_angles[vi], wavelength, PITCH, N_SUB)
            bx = angle_to_fft_bin(lf_angles[ui], wavelength, PITCH, N_SUB)
            # Round to nearest integer bin
            byi = int(round(by))
            bxi = int(round(bx))
            if 0 <= byi < N_SUB and 0 <= bxi < N_SUB:
                bin_y.append(byi)
                bin_x.append(bxi)
                target_vals.append(angular_target[vi, ui])

    bin_y = torch.tensor(bin_y, dtype=torch.long, device=DEVICE)
    bin_x = torch.tensor(bin_x, dtype=torch.long, device=DEVICE)
    target_t = torch.tensor(target_vals, dtype=torch.float32, device=DEVICE)

    # Normalize target: phase-only FFT has total energy N²
    # Scale target so sum matches expected energy at sample points
    if target_t.sum() > 0:
        target_t = target_t / target_t.sum() * (N_SUB ** 2) * len(target_vals) / (N_SUB ** 2)

    phase = torch.randn(N_SUB, N_SUB, device=DEVICE) * math.pi
    phase.requires_grad_(True)

    optimizer = torch.optim.Adam([phase], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

    for it in range(n_iters):
        optimizer.zero_grad()

        u = torch.exp(1j * phase)
        U = fft.fftshift(fft.fft2(u))
        intensity = torch.abs(U) ** 2

        # Loss only at the angular sample bins
        sampled = intensity[bin_y, bin_x]
        loss = torch.mean((sampled - target_t) ** 2)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (it + 1) % 500 == 0 or it == 0:
            print(f"    iter {it+1:5d}/{n_iters}: loss={loss.item():.6f}")

    return phase.detach().cpu().numpy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print(f"Single Hogel Simulator — {N_SUB}×{N_SUB} @ {PITCH*1e6:.3f}µm pitch")
    print(f"  Hogel size: {HOGEL_SIZE*1e6:.1f}µm ({DPI} DPI)")
    print("=" * 60)

    # Load light field
    lf_path = PLOT_DIR / "cornell_lightfield.npz"
    data = np.load(str(lf_path))
    lightfield = data["lightfield"]
    angular_positions = data["angular_positions"]
    H, W = lightfield.shape[2], lightfield.shape[3]

    # Tone-map
    exposure = 2.0
    lf_mapped = 1.0 - np.exp(-lightfield * exposure)
    lf_mapped = np.clip(lf_mapped, 0, 1)

    # Pick hogels at various positions to show different parts of the scene
    hogel_positions = [
        (H // 2, W // 2, "center"),
        (H // 4, W // 4, "top-left"),
        (H // 4, 3 * W // 4, "top-right"),
        (3 * H // 4, W // 2, "bottom-center"),
    ]

    print(f"  Device: {DEVICE}")
    print(f"  Optimizing {len(hogel_positions)} hogels × 3 channels...")

    angular_samples_all, lf_angles = extract_hogel_angular_samples(
        lf_mapped, angular_positions, H // 2, W // 2
    )

    all_results = {}
    t0 = time.time()

    for py, px, name in hogel_positions:
        print(f"\n  Hogel '{name}' at ({py}, {px}):")
        samples, lf_angles = extract_hogel_angular_samples(
            lf_mapped, angular_positions, py, px
        )

        hogel_phases = {}
        for ch in ["R", "G", "B"]:
            wl = WAVELENGTHS[ch]
            target = samples[ch]
            print(f"    {ch}: target range [{target.min():.3f}, {target.max():.3f}]")
            phase = optimize_hogel(target, lf_angles, wl, n_iters=3000, lr=0.1)
            hogel_phases[ch] = phase

        all_results[name] = hogel_phases

    dt = time.time() - t0
    print(f"\n  Total: {dt:.1f}s ({dt/60:.1f} min)")

    # ── Visualization ──
    n_hogels = len(hogel_positions)
    fig, axes = plt.subplots(n_hogels, 4, figsize=(20, 5 * n_hogels))

    for hi, (py, px, name) in enumerate(hogel_positions):
        phases = all_results[name]
        samples, _ = extract_hogel_angular_samples(
            lf_mapped, angular_positions, py, px
        )

        # Col 0: Phase pattern (green channel)
        axes[hi, 0].imshow(phases["G"], cmap="twilight", vmin=-math.pi, vmax=math.pi)
        axes[hi, 0].set_title(f"'{name}' phase (G)")
        axes[hi, 0].axis("off")

        # Col 1: Target angular samples (as 9×9 grid)
        rgb_target = np.stack([samples["R"], samples["G"], samples["B"]], axis=-1)
        rgb_target = rgb_target / (rgb_target.max() + 1e-10)
        axes[hi, 1].imshow(rgb_target, interpolation="nearest")
        axes[hi, 1].set_title(f"Target 9×9 angular (RGB)")
        axes[hi, 1].axis("off")

        # Col 2: Far-field intensity (green)
        phase_t = torch.tensor(phases["G"], dtype=torch.float32, device=DEVICE)
        intensity, angles = hogel_farfield(phase_t, WAVELENGTHS["G"])
        ff = intensity.cpu().numpy()
        axes[hi, 2].imshow(np.log10(ff + 1), cmap="hot")
        axes[hi, 2].set_title(f"Far-field |FFT|² (G, log)")
        axes[hi, 2].axis("off")

        # Col 3: Angular cross-section comparison
        center = N_SUB // 2
        ff_slice = ff[center, :]
        target_slice = samples["G"][samples["G"].shape[0] // 2, :]

        # Map target angular positions to FFT bins for overlay
        target_bins = []
        for ai in range(len(lf_angles)):
            b = angle_to_fft_bin(lf_angles[ai], WAVELENGTHS["G"], PITCH, N_SUB)
            target_bins.append(int(round(b)))

        axes[hi, 3].plot(angles, ff_slice / (ff_slice.max() + 1e-10),
                         'b-', alpha=0.7, label='Recon (G)')
        axes[hi, 3].scatter(
            [angles[b] for b in target_bins if 0 <= b < N_SUB],
            target_slice / (target_slice.max() + 1e-10),
            c='red', s=50, zorder=5, label='Target (9 pts)'
        )
        theta_max = math.degrees(math.asin(min(1.0, WAVELENGTHS["G"] / (2 * PITCH))))
        axes[hi, 3].axvline(theta_max, color='gray', ls=':', alpha=0.5)
        axes[hi, 3].axvline(-theta_max, color='gray', ls=':', alpha=0.5)
        axes[hi, 3].set_xlim(-30, 30)
        axes[hi, 3].set_xlabel("Angle (°)")
        axes[hi, 3].set_title(f"Angular profile: '{name}'")
        axes[hi, 3].legend(fontsize=8)
        axes[hi, 3].grid(True, alpha=0.3)

    plt.suptitle(
        f"HoloPixel Hogel: {N_SUB}×{N_SUB} waveguides, "
        f"{PITCH*1e6:.3f}µm pitch, {HOGEL_SIZE*1e6:.1f}µm hogel ({DPI} DPI)\n"
        f"Phase retrieval from Cornell box light field",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out_fig = PLOT_DIR / "hogel_single.png"
    fig.savefig(str(out_fig), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out_fig}")


if __name__ == "__main__":
    main()

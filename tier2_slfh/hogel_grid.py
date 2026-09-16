"""
Hogel Grid Display Simulator

Optimizes a grid of hogels covering the Cornell box light field,
then simulates what an observer sees from each viewing angle.

Each hogel: 250×250 waveguides at 0.706µm pitch (144 DPI).
All hogels for one color channel are optimized as a single GPU batch.
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

N_SUB = 250
PITCH = 0.706e-6
HOGEL_SIZE = N_SUB * PITCH
DPI = 144

WAVELENGTHS = {"R": 632e-9, "G": 532e-9, "B": 450e-9}

GRID_SIZE = 32
N_ITERS = 1000
LR = 0.1


def angle_to_fft_bin(angle_rad, wavelength):
    freq = math.sin(angle_rad) / wavelength
    return freq * N_SUB * PITCH + N_SUB / 2


def batch_optimize(targets, lf_angles, wavelength):
    """Optimize phase for all hogels in one GPU batch.

    Args:
        targets: (B, nv, nu) numpy — angular intensities per hogel
        lf_angles: (n_ang,) numpy — viewing angles [rad]
        wavelength: float [m]

    Returns:
        phases: (B, N_SUB, N_SUB) tensor on DEVICE
    """
    B, nv, nu = targets.shape

    # Map LF angles to FFT bins — same for all hogels
    bin_y, bin_x, valid = [], [], []
    for vi in range(nv):
        for ui in range(nu):
            by = int(round(angle_to_fft_bin(lf_angles[vi], wavelength)))
            bx = int(round(angle_to_fft_bin(lf_angles[ui], wavelength)))
            if 0 <= by < N_SUB and 0 <= bx < N_SUB:
                bin_y.append(by)
                bin_x.append(bx)
                valid.append((vi, ui))

    n_samples = len(valid)
    bin_y_t = torch.tensor(bin_y, dtype=torch.long, device=DEVICE)
    bin_x_t = torch.tensor(bin_x, dtype=torch.long, device=DEVICE)

    # Target: (B, n_samples)
    target_np = np.array(
        [[targets[b, vi, ui] for vi, ui in valid] for b in range(B)],
        dtype=np.float32,
    )
    target_t = torch.tensor(target_np, device=DEVICE)
    # Normalize so each hogel's target sums to n_samples
    sums = target_t.sum(dim=1, keepdim=True).clamp(min=1e-10)
    target_t = target_t / sums * n_samples

    phases = torch.randn(B, N_SUB, N_SUB, device=DEVICE) * math.pi
    phases.requires_grad_(True)

    optimizer = torch.optim.Adam([phases], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_ITERS)

    for it in range(N_ITERS):
        optimizer.zero_grad()

        u = torch.exp(1j * phases)
        U = fft.fftshift(fft.fft2(u), dim=(-2, -1))
        intensity = torch.abs(U) ** 2

        sampled = intensity[:, bin_y_t, bin_x_t]  # (B, n_samples)
        loss = ((sampled - target_t) ** 2).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (it + 1) % 200 == 0 or it == 0:
            print(f"      iter {it+1:5d}/{N_ITERS}: loss={loss.item():.2f}")

    return phases.detach()


def simulate_views(phases, lf_angles, wavelength):
    """For each viewing angle, get each hogel's output intensity.

    Returns: (nv, nu, B) numpy array
    """
    nv = nu = len(lf_angles)
    B = phases.shape[0]

    with torch.no_grad():
        u = torch.exp(1j * phases)
        U = fft.fftshift(fft.fft2(u), dim=(-2, -1))
        intensity = torch.abs(U) ** 2

    views = np.zeros((nv, nu, B), dtype=np.float32)
    for vi in range(nv):
        for ui in range(nu):
            by = int(round(angle_to_fft_bin(lf_angles[vi], wavelength)))
            bx = int(round(angle_to_fft_bin(lf_angles[ui], wavelength)))
            if 0 <= by < N_SUB and 0 <= bx < N_SUB:
                views[vi, ui] = intensity[:, by, bx].cpu().numpy()

    return views


def main():
    print("=" * 60)
    print(f"Hogel Grid Display — {GRID_SIZE}×{GRID_SIZE} hogels")
    print(f"  Each: {N_SUB}×{N_SUB} @ {PITCH*1e6:.3f}µm ({DPI} DPI)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    data = np.load(str(PLOT_DIR / "cornell_lightfield.npz"))
    lightfield = data["lightfield"]
    angular_positions = data["angular_positions"]
    nv, nu, H, W, C = lightfield.shape

    # Tone-map
    lf = np.clip(1.0 - np.exp(-lightfield * 2.0), 0, 1)

    # Spatial grid of hogel positions
    gy = np.linspace(0, H - 1, GRID_SIZE, dtype=int)
    gx = np.linspace(0, W - 1, GRID_SIZE, dtype=int)

    # LF camera positions → viewing angles
    cam_z, Z0 = 0.2, 0.1
    lf_angles = np.arctan2(angular_positions, cam_z - Z0)

    # Extract targets: (B, nv, nu) per channel
    B = GRID_SIZE * GRID_SIZE
    print(f"  {B} hogels, {nv}×{nu} angular samples each")

    targets = {}
    for ci, ch in enumerate(["R", "G", "B"]):
        t = np.zeros((B, nv, nu), dtype=np.float32)
        idx = 0
        for py in gy:
            for px in gx:
                t[idx] = lf[:, :, py, px, ci]
                idx += 1
        targets[ch] = t

    # Optimize + simulate per channel
    all_views = {}
    t0 = time.time()

    for ch in ["R", "G", "B"]:
        wl = WAVELENGTHS[ch]
        print(f"\n  {ch} (λ={wl*1e9:.0f}nm):")
        print(f"    Optimizing {B} hogels...")
        phases = batch_optimize(targets[ch], lf_angles, wl)
        print(f"    Simulating views...")
        all_views[ch] = simulate_views(phases, lf_angles, wl)
        del phases
        torch.cuda.empty_cache()

    dt = time.time() - t0
    print(f"\n  Total: {dt:.1f}s ({dt/60:.1f} min)")

    # Assemble RGB images per viewing angle
    recon = np.zeros((nv, nu, GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
    for ci, ch in enumerate(["R", "G", "B"]):
        recon[:, :, :, :, ci] = all_views[ch].reshape(nv, nu, GRID_SIZE, GRID_SIZE)

    # Ground truth at same spatial positions
    gt = np.zeros_like(recon)
    idx = 0
    for py in gy:
        for px in gx:
            hy, hx = divmod(idx, GRID_SIZE)
            gt[:, :, hy, hx, :] = lf[:, :, py, px, :]
            idx += 1

    # Normalize recon to match GT brightness range
    p99 = np.percentile(recon[recon > 0], 99) if (recon > 0).any() else 1.0
    recon = np.clip(recon / p99, 0, 1)

    # ── Visualization: 3×3 viewing angles, recon vs GT ──
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    angle_sel = [0, nv // 2, nv - 1]

    for ri, vi in enumerate(angle_sel):
        for ci, ui in enumerate(angle_sel):
            deg_v = math.degrees(lf_angles[vi])
            deg_u = math.degrees(lf_angles[ui])

            axes[ri, ci * 2].imshow(recon[vi, ui])
            axes[ri, ci * 2].set_title(f"Recon ({deg_v:.1f}°, {deg_u:.1f}°)", fontsize=9)
            axes[ri, ci * 2].axis("off")

            axes[ri, ci * 2 + 1].imshow(gt[vi, ui])
            axes[ri, ci * 2 + 1].set_title(f"GT ({deg_v:.1f}°, {deg_u:.1f}°)", fontsize=9)
            axes[ri, ci * 2 + 1].axis("off")

    plt.suptitle(
        f"Hogel Grid: {GRID_SIZE}×{GRID_SIZE} hogels, "
        f"{N_SUB}×{N_SUB} waveguides each ({DPI} DPI)\n"
        f"Reconstructed views vs Ground Truth at 9 viewing angles",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = PLOT_DIR / "hogel_grid.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


if __name__ == "__main__":
    main()

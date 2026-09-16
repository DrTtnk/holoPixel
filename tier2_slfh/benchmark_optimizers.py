"""
Benchmark: Adam vs Lion optimizer for SLFH hologram optimization.
Single channel (Green), 512×512, 500 iterations, same random seed.
"""

import torch
import numpy as np
import math
import time
import sys
from pathlib import Path
from lion_pytorch import Lion

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slfh import (
    wigner_projection, lightfield_projection, sample_pupil,
    WAVELENGTHS, PIXEL_PITCH, FOCAL_LENGTH, EYEBOX_SIZE,
    DEVICE, PLOT_DIR,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_benchmark(lf_ch, angular_positions, optimizer_cls, opt_kwargs,
                  name, n_iters=500, n_pupils=4, seed=42):
    wavelength = WAVELENGTHS["G"]
    N = lf_ch.shape[2]

    torch.manual_seed(seed)
    np.random.seed(seed)

    phase = torch.randn(N, N, device=DEVICE) * 0.1
    phase.requires_grad_(True)

    optimizer = optimizer_cls([phase], **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

    losses = []
    t0 = time.time()

    for it in range(n_iters):
        np.random.seed(seed + it)  # deterministic pupil sampling per iter
        optimizer.zero_grad()
        loss_total = 0.0

        for _ in range(n_pupils):
            pupil = sample_pupil(EYEBOX_SIZE, 0.0, 0.015, 0.002, 0.020)

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

        loss_total /= n_pupils
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss_total.item())

    dt = time.time() - t0
    print(f"  {name:8s}: {dt:.1f}s, final_loss={losses[-1]:.6f}, "
          f"min_loss={min(losses):.6f}, iter/s={n_iters/dt:.1f}")

    return losses, dt, phase.detach().cpu().numpy()


def main():
    print("=" * 60)
    print("SLFH Optimizer Benchmark: Adam vs Lion")
    print("=" * 60)

    # Load light field
    data = np.load(str(PLOT_DIR / "cornell_lightfield.npz"))
    lf = data["lightfield"]
    ang_pos = data["angular_positions"]

    # Tone-map
    lf_mapped = 1.0 - np.exp(-lf * 2.0)
    lf_mapped = np.clip(lf_mapped, 0, 1)

    # Green channel only
    lf_g = torch.tensor(lf_mapped[:, :, :, :, 1], dtype=torch.float32, device=DEVICE)

    n_iters = 500

    # Run benchmarks with matched LR schedules
    results = {}

    # Adam with different LRs
    for lr in [0.01, 0.03, 0.05]:
        name = f"Adam-{lr}"
        losses, dt, _ = run_benchmark(
            lf_g, ang_pos, torch.optim.Adam, {"lr": lr},
            name, n_iters=n_iters
        )
        results[name] = (losses, dt)

    # Lion with different LRs (Lion typically needs 3-10× lower LR than Adam)
    for lr in [0.001, 0.003, 0.01]:
        name = f"Lion-{lr}"
        losses, dt, _ = run_benchmark(
            lf_g, ang_pos, Lion, {"lr": lr},
            name, n_iters=n_iters
        )
        results[name] = (losses, dt)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, (losses, dt) in results.items():
        color = "tab:blue" if "Adam" in name else "tab:red"
        alpha = 0.4 if losses[-1] > min(l[-1] for l, _ in results.values()) * 1.5 else 1.0
        # Smoothed loss (rolling average)
        window = 20
        smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax1.plot(smoothed, label=f"{name} (final={losses[-1]:.4f})",
                 color=color, alpha=alpha)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("L2 Loss (smoothed)")
    ax1.set_title("Loss Convergence")
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Bar chart: wall time
    names = list(results.keys())
    times = [results[n][1] for n in names]
    final_losses = [results[n][0][-1] for n in names]
    colors = ["tab:blue" if "Adam" in n else "tab:red" for n in names]

    ax2.bar(range(len(names)), times, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Wall Time (s)")
    ax2.set_title("Wall Time (500 iters)")
    ax2.grid(True, alpha=0.3, axis="y")

    for i, (t, l) in enumerate(zip(times, final_losses)):
        ax2.text(i, t + 0.5, f"loss={l:.4f}", ha="center", fontsize=7)

    plt.suptitle("SLFH Optimizer Benchmark — Green channel, 512×512",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PLOT_DIR / "slfh_optimizer_benchmark.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved: {out}")


if __name__ == "__main__":
    main()

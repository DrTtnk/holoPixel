"""
Partial Phase Holography Analysis

Key question: Do we NEED full 2π phase coverage?

If we can work with < 2π, the GTE requirements relax dramatically:
- Lower R₁ (cheaper, more robust mirrors)
- Shallower cavities (less crosstalk)
- Single depth for all colors

We simulate holographic reconstruction quality as a function of
available phase range, using a simple 1D hologram.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def make_target_1d(N, num_points=3):
    """Create a 1D target: a few point sources at different angles."""
    rng = np.random.RandomState(42)
    target = np.zeros(N)
    positions = rng.choice(N, num_points, replace=False)
    target[positions] = 1.0
    return target


def compute_hologram_phase(target):
    """Compute ideal hologram phase pattern (inverse Fourier)."""
    return np.angle(np.fft.ifft(target))


def quantize_phase(phase, max_phase, n_levels=0):
    """
    Restrict phase to [0, max_phase] range.
    If n_levels > 0, also quantize to discrete levels.

    Level placement depends on whether the range wraps:
      max_phase >= 2π — 0 and max_phase are the same physical phase, so the
                        levels are the n_levels points of arange(n)·Δ with
                        Δ = max_phase/n_levels. Using both endpoints here
                        would collapse two levels onto one and silently
                        deliver n_levels-1 distinct phases.
      max_phase <  2π — the range does not wrap, so both endpoints are usable
                        and the spacing is max_phase/(n_levels-1).
    """
    if max_phase > 2 * np.pi:
        raise ValueError(f"max_phase must not exceed 2π, got {max_phase}")

    # Wrap phase to [0, 2π]
    phase_wrapped = phase % (2 * np.pi)
    # Scale to [0, max_phase]
    if max_phase < 2 * np.pi:
        phase_clipped = np.clip(phase_wrapped, 0, max_phase)
    else:
        phase_clipped = phase_wrapped

    if n_levels > 0:
        if max_phase == 2 * np.pi:
            step = max_phase / n_levels
            # Nearest level on the circle: 2π-ε belongs to level 0, not to n-1.
            idx = (np.round(phase_clipped / step).astype(int)) % n_levels
            phase_clipped = idx * step
        else:
            levels = np.linspace(0, max_phase, n_levels)
            idx = np.argmin(np.abs(phase_clipped[:, None] - levels[None, :]), axis=1)
            phase_clipped = levels[idx]

    return phase_clipped


def reconstruct(phase):
    """Reconstruct far-field from phase-only hologram."""
    field = np.exp(1j * phase)
    return np.abs(np.fft.fft(field)) ** 2


def efficiency_metric(reconstruction, target):
    """
    Diffraction efficiency: fraction of light going to desired directions.
    Signal-to-noise: ratio of signal power to noise power.
    """
    target_mask = target > 0
    signal_power = reconstruction[target_mask].sum()
    total_power = reconstruction.sum()
    noise_power = reconstruction[~target_mask].sum()

    efficiency = signal_power / total_power
    snr = signal_power / noise_power if noise_power > 0 else np.inf
    return efficiency, snr


# =============================================================
# 1. Reconstruction quality vs phase range
# =============================================================
def sweep_phase_range():
    N = 256  # number of pixels
    target = make_target_1d(N, num_points=5)
    ideal_phase = compute_hologram_phase(target)

    phase_ranges = np.linspace(0.1 * np.pi, 2.0 * np.pi, 100)
    efficiencies = []
    snrs = []

    for max_phase in phase_ranges:
        clipped_phase = quantize_phase(ideal_phase, max_phase)
        recon = reconstruct(clipped_phase)
        eff, snr = efficiency_metric(recon, target)
        efficiencies.append(eff)
        snrs.append(snr)

    # Also test with quantized levels
    level_counts = [4, 8, 16, 32, 256]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].plot(phase_ranges / np.pi, efficiencies, "k-", linewidth=2, label="Continuous")
    axes[1].plot(phase_ranges / np.pi, snrs, "k-", linewidth=2, label="Continuous")

    for n_levels in [4, 8, 16]:
        eff_q = []
        snr_q = []
        for max_phase in phase_ranges:
            q_phase = quantize_phase(ideal_phase, max_phase, n_levels)
            recon = reconstruct(q_phase)
            eff, snr = efficiency_metric(recon, target)
            eff_q.append(eff)
            snr_q.append(snr)
        axes[0].plot(phase_ranges / np.pi, eff_q, "--", linewidth=1.5, label=f"{n_levels} levels")
        axes[1].plot(phase_ranges / np.pi, snr_q, "--", linewidth=1.5, label=f"{n_levels} levels")

    for ax in axes:
        ax.axvline(x=1.84, color="green", linestyle=":", alpha=0.5,
                   label="1.84π (R₁=0.90)")
        ax.axvline(x=1.5, color="orange", linestyle=":", alpha=0.5,
                   label="1.5π (R₁=0.80)")
        ax.set_xlabel("Available Phase Range (×π rad)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Diffraction Efficiency", fontsize=12)
    axes[0].set_title("Efficiency vs Phase Range", fontsize=13)
    axes[1].set_ylabel("Signal-to-Noise Ratio", fontsize=12)
    axes[1].set_title("SNR vs Phase Range", fontsize=13)

    fig.suptitle("Hologram Quality with Partial Phase Coverage (1D, 5 points)", fontsize=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "partial_phase_quality.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'partial_phase_quality.png'}")


# =============================================================
# 2. Visual comparison: full vs partial phase reconstructions
# =============================================================
def visual_comparison():
    N = 256
    target = make_target_1d(N, num_points=5)
    ideal_phase = compute_hologram_phase(target)

    phase_configs = [
        ("Full 2π", 2.0 * np.pi, 0),
        ("1.84π (R₁=0.90)", 1.84 * np.pi, 0),
        ("1.5π (R₁=0.80)", 1.50 * np.pi, 0),
        ("π (R₁=0.65)", 1.0 * np.pi, 0),
        ("1.84π, 8 levels", 1.84 * np.pi, 8),
        ("1.84π, 4 levels", 1.84 * np.pi, 4),
    ]

    ideal_recon = reconstruct(ideal_phase)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (label, max_phase, n_levels) in zip(axes.flat, phase_configs):
        clipped = quantize_phase(ideal_phase, max_phase, n_levels)
        recon = reconstruct(clipped)
        eff, snr = efficiency_metric(recon, target)

        ax.plot(recon / recon.max(), "b-", linewidth=1, alpha=0.7)
        ax.plot(ideal_recon / ideal_recon.max(), "r--", linewidth=1, alpha=0.3, label="Ideal")
        # Mark target positions
        target_pos = np.where(target > 0)[0]
        ax.scatter(target_pos, np.ones(len(target_pos)), color="red", s=50, zorder=5)

        ax.set_title(f"{label}\nη={eff:.3f}, SNR={snr:.1f}", fontsize=11)
        ax.set_ylim(-0.05, 1.2)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Reconstruction Quality: Full vs Partial Phase (normalized)", fontsize=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "partial_phase_visual.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'partial_phase_visual.png'}")


# =============================================================
# 3. 2D hologram test (more realistic)
# =============================================================
def test_2d_hologram():
    N = 64  # keep small for speed
    # Target: 3 point sources at different (kx, ky) angles
    target = np.zeros((N, N))
    target[N//4, N//4] = 1.0
    target[N//2, 3*N//4] = 1.0
    target[3*N//4, N//2] = 1.0

    ideal_phase = np.angle(np.fft.ifft2(target))

    configs = [
        ("Full 2π", 2.0 * np.pi),
        ("1.84π", 1.84 * np.pi),
        ("1.5π", 1.5 * np.pi),
        ("π", 1.0 * np.pi),
    ]

    fig, axes = plt.subplots(1, len(configs) + 1, figsize=(20, 4))

    # Target
    recon_ideal = np.abs(np.fft.fft2(np.exp(1j * ideal_phase))) ** 2
    axes[0].imshow(np.log1p(recon_ideal), cmap="hot")
    axes[0].set_title("Ideal (full 2π)")

    for ax, (label, max_phase) in zip(axes[1:], configs):
        clipped = quantize_phase(ideal_phase.ravel(), max_phase).reshape(N, N)
        recon = np.abs(np.fft.fft2(np.exp(1j * clipped))) ** 2
        ax.imshow(np.log1p(recon), cmap="hot")

        target_mask = target > 0
        sig = recon[target_mask].sum()
        tot = recon.sum()
        ax.set_title(f"{label}\nη = {sig/tot:.3f}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("2D Hologram Reconstruction with Partial Phase", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "partial_phase_2d.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'partial_phase_2d.png'}")


if __name__ == "__main__":
    print("Partial Phase Holography Analysis")
    print("=" * 50)
    sweep_phase_range()
    visual_comparison()
    test_2d_hologram()
    print("\nDone!")

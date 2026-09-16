"""
Hogel Projection Analysis

1. Phase quantization: how many bits/levels for acceptable hologram quality?
2. Far-field projection: what does an observer see from a single hogel?
3. Sub-pixel design: pitch, count, and angular resolution relationships

A hogel (holographic element) is a 2D array of phase-shifting sub-pixels.
The far-field pattern is the 2D Fourier transform of exp(i·φ(x,y)).
Each point in the far field corresponds to a viewing angle (θx, θy).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# =============================================================
# Physical parameters
# =============================================================
HOGEL_SIZE = 169.3e-6    # meters (150 PPI)
SUB_PIXEL_PITCH = 0.5e-6  # meters (from DRAFT)
FOV_DEG = 30              # half-angle in degrees
LAMBDA = 532e-9           # green light for single-color analysis


def angles_from_fft(N, pitch, lam):
    """Convert FFT pixel indices to physical angles (degrees)."""
    # Spatial frequency: f = n/(N·pitch) for n = -N/2 to N/2
    # Diffraction angle: sin(θ) = f·λ = n·λ/(N·pitch)
    n = np.arange(N) - N // 2
    sin_theta = n * lam / (N * pitch)
    # Clip to valid range
    sin_theta = np.clip(sin_theta, -1, 1)
    return np.degrees(np.arcsin(sin_theta))


# =============================================================
# 1. Phase Quantization Depth Study
# =============================================================
def quantization_study():
    """
    Sweep from 1-bit (binary) to 8-bit (256 levels) phase quantization.
    Measure: diffraction efficiency, SNR, RMSE, and ghost order strength.
    """
    print("=" * 70)
    print("PHASE QUANTIZATION STUDY")
    print("=" * 70)

    # Hogel geometry
    N = int(HOGEL_SIZE / SUB_PIXEL_PITCH)  # sub-pixels per row
    print(f"Hogel: {N}×{N} = {N**2:,} sub-pixels at {SUB_PIXEL_PITCH*1e6:.1f}µm pitch")

    # But N=338 is large for exhaustive study. Use a range.
    grid_sizes = [32, 64, 128, 256]
    bit_depths = [1, 2, 3, 4, 5, 6, 7, 8]

    # Create test targets: 3D scene projected into angular space
    # Target: 5 point sources at different angles within ±30°
    rng = np.random.RandomState(42)

    fig_summary, ax_summary = plt.subplots(1, 2, figsize=(14, 6))

    for N_grid in grid_sizes:
        target = np.zeros((N_grid, N_grid))
        # Place points at known angular positions
        n_points = 7
        for _ in range(n_points):
            ix = rng.randint(N_grid // 4, 3 * N_grid // 4)
            iy = rng.randint(N_grid // 4, 3 * N_grid // 4)
            target[ix, iy] = 1.0

        # Ideal hologram phase
        ideal_phase = np.angle(np.fft.ifft2(target))

        efficiencies = []
        snrs = []

        for bits in bit_depths:
            n_levels = 2 ** bits
            # Quantize phase to n_levels within [0, 2π]
            phase_wrapped = ideal_phase % (2 * np.pi)
            levels = np.linspace(0, 2 * np.pi * (1 - 1 / n_levels), n_levels)
            # Map to nearest level
            idx = np.round(phase_wrapped / (2 * np.pi) * n_levels).astype(int) % n_levels
            quantized_phase = levels[idx]

            # Reconstruct
            field = np.exp(1j * quantized_phase)
            recon = np.abs(np.fft.fft2(field)) ** 2

            # Metrics
            target_mask = target > 0
            signal = recon[target_mask].sum()
            total = recon.sum()
            noise = recon[~target_mask].sum()

            eff = signal / total
            snr = signal / noise if noise > 0 else np.inf
            efficiencies.append(eff)
            snrs.append(snr)

        ax_summary[0].plot(bit_depths, efficiencies, "o-", linewidth=2,
                          label=f"N={N_grid}")
        ax_summary[1].plot(bit_depths, snrs, "o-", linewidth=2,
                          label=f"N={N_grid}")

    # Theoretical efficiency for uniform quantization
    theoretical_eta = [np.sinc(1 / (2 ** b)) ** 2 for b in bit_depths]
    ax_summary[0].plot(bit_depths, theoretical_eta, "k--", linewidth=2,
                      label="Theory: sinc²(1/N)")

    for ax in ax_summary:
        ax.set_xlabel("Phase Quantization (bits)", fontsize=12)
        ax.set_xticks(bit_depths)
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax_summary[0].set_ylabel("Diffraction Efficiency η")
    ax_summary[0].set_title("Efficiency vs Phase Bits")
    ax_summary[1].set_ylabel("Signal-to-Noise Ratio")
    ax_summary[1].set_title("SNR vs Phase Bits")

    fig_summary.suptitle("Phase Quantization: How Many Bits?", fontsize=14)
    fig_summary.tight_layout()
    fig_summary.savefig(PLOT_DIR / "quantization_bits.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'quantization_bits.png'}")

    # Key numbers
    print("\nTheoretical efficiency by bit depth:")
    for bits, eta in zip(bit_depths, theoretical_eta):
        print(f"  {bits}-bit ({2**bits} levels): η = {eta:.1%}")

    # Visual comparison
    N_vis = 128
    target = np.zeros((N_vis, N_vis))
    points = [(40, 50), (60, 80), (90, 30), (70, 100), (110, 60)]
    for px, py in points:
        target[px, py] = 1.0

    ideal_phase = np.angle(np.fft.ifft2(target))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, bits in enumerate([1, 2, 3, 4]):
        n_levels = 2 ** bits
        levels = np.linspace(0, 2 * np.pi * (1 - 1 / n_levels), n_levels)
        phase_wrapped = ideal_phase % (2 * np.pi)
        idx = np.round(phase_wrapped / (2 * np.pi) * n_levels).astype(int) % n_levels
        q_phase = levels[idx]

        recon = np.abs(np.fft.fft2(np.exp(1j * q_phase))) ** 2
        recon_log = np.log1p(recon / recon.max() * 100)

        axes[0, col].imshow(q_phase, cmap="twilight")
        axes[0, col].set_title(f"{bits}-bit ({n_levels} levels)")

        axes[1, col].imshow(np.fft.fftshift(recon_log), cmap="hot")
        target_mask = target > 0
        eff = recon[target_mask].sum() / recon.sum()
        axes[1, col].set_title(f"η = {eff:.1%}")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel("Phase Pattern")
    axes[1, 0].set_ylabel("Far-Field (log)")
    fig.suptitle("Phase Quantization: Pattern & Reconstruction", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "quantization_visual.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'quantization_visual.png'}")


# =============================================================
# 2. Single Hogel Far-Field Projection
# =============================================================
def hogel_projection():
    """
    Simulate what an observer sees when looking at a single hogel
    from different angles. This IS the holographic pixel output.
    """
    print("\n" + "=" * 70)
    print("SINGLE HOGEL FAR-FIELD PROJECTION")
    print("=" * 70)

    # Use a manageable grid (full 338×338 is fine for FFT)
    N = 256  # close to actual 338, power of 2 for FFT efficiency

    # Physical setup
    pitch = SUB_PIXEL_PITCH
    angles = angles_from_fft(N, pitch, LAMBDA)
    max_angle = np.degrees(np.arcsin(LAMBDA / (2 * pitch)))
    print(f"Sub-pixel pitch: {pitch*1e6:.1f} µm")
    print(f"Maximum diffraction angle: ±{max_angle:.1f}°")
    print(f"Grid: {N}×{N} sub-pixels")

    # Scenario 1: Hogel encoding a single bright point at (15°, 10°)
    # The phase pattern is a linear phase ramp
    x = np.arange(N) * pitch
    y = np.arange(N) * pitch
    X, Y = np.meshgrid(x, y)

    target_theta_x = 15  # degrees
    target_theta_y = 10  # degrees

    # Phase ramp for off-axis point
    kx = 2 * np.pi * np.sin(np.radians(target_theta_x)) / LAMBDA
    ky = 2 * np.pi * np.sin(np.radians(target_theta_y)) / LAMBDA
    phase_single = kx * X + ky * Y

    # Scenario 2: Hogel encoding 3 bright points at different angles
    points_3 = [(15, 10), (-20, 5), (5, -25)]
    target_field = np.zeros((N, N), dtype=complex)
    for tx, ty in points_3:
        kx = 2 * np.pi * np.sin(np.radians(tx)) / LAMBDA
        ky = 2 * np.pi * np.sin(np.radians(ty)) / LAMBDA
        target_field += np.exp(1j * (kx * X + ky * Y))
    phase_multi = np.angle(target_field)

    # Compute far-field patterns
    scenarios = [
        ("Single point (15°, 10°)", phase_single),
        ("3 points", phase_multi),
    ]

    # Also show quantized versions
    for label, phase in scenarios:
        for bits in [0, 3, 4]:  # 0 = continuous
            if bits == 0:
                used_phase = phase % (2 * np.pi)
                tag = "continuous"
            else:
                n_levels = 2 ** bits
                phase_w = phase % (2 * np.pi)
                idx = np.round(phase_w / (2 * np.pi) * n_levels).astype(int) % n_levels
                used_phase = idx * (2 * np.pi / n_levels)
                tag = f"{bits}-bit"

            field = np.exp(1j * used_phase)
            far_field = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
            far_field_norm = far_field / far_field.max()

            if bits == 0 and "Single" in label:
                # Save the first one for detailed analysis
                ff_single = far_field_norm

    # Big comparison plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for row, (label, phase) in enumerate(scenarios):
        for col, bits in enumerate([0, 2, 3, 4]):
            if bits == 0:
                used_phase = phase % (2 * np.pi)
                tag = "Continuous"
            else:
                n_levels = 2 ** bits
                phase_w = phase % (2 * np.pi)
                idx = np.round(phase_w / (2 * np.pi) * n_levels).astype(int) % n_levels
                used_phase = idx * (2 * np.pi / n_levels)
                tag = f"{bits}-bit ({2**bits} levels)"

            field = np.exp(1j * used_phase)
            far_field = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
            far_field_db = 10 * np.log10(far_field / far_field.max() + 1e-10)

            extent = [angles[0], angles[-1], angles[0], angles[-1]]
            im = axes[row, col].imshow(far_field_db, cmap="hot",
                                       extent=extent, vmin=-30, vmax=0,
                                       aspect="equal")
            axes[row, col].set_xlim(-35, 35)
            axes[row, col].set_ylim(-35, 35)

            # Mark target points
            if "Single" in label:
                axes[row, col].plot(target_theta_x, target_theta_y, "c+",
                                   markersize=15, markeredgewidth=2)
            else:
                for tx, ty in points_3:
                    axes[row, col].plot(tx, ty, "c+", markersize=10,
                                       markeredgewidth=2)

            if row == 0:
                axes[row, col].set_title(tag, fontsize=12)
            if col == 0:
                axes[row, col].set_ylabel(f"{label}\nθ_y (°)")
            axes[row, col].set_xlabel("θ_x (°)")
            axes[row, col].grid(True, alpha=0.2, color="white")

    fig.suptitle(f"Hogel Far-Field Projection — λ={LAMBDA*1e9:.0f}nm, "
                 f"pitch={pitch*1e6:.1f}µm, {N}×{N} sub-pixels",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hogel_projection.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'hogel_projection.png'}")

    # Cross-section through the main lobe
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    phase = scenarios[0][1]  # single point

    for bits in [0, 2, 3, 4, 6]:
        if bits == 0:
            used_phase = phase % (2 * np.pi)
            tag = "Continuous"
        else:
            n_levels = 2 ** bits
            phase_w = phase % (2 * np.pi)
            idx = np.round(phase_w / (2 * np.pi) * n_levels).astype(int) % n_levels
            used_phase = idx * (2 * np.pi / n_levels)
            tag = f"{bits}-bit"

        field = np.exp(1j * used_phase)
        far_field = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
        far_field_norm = far_field / far_field.max()

        # Horizontal cross-section through peak
        peak_y = N // 2 + int(np.sin(np.radians(10)) * N * pitch / LAMBDA)
        peak_y = np.clip(peak_y, 0, N - 1)
        cross = far_field_norm[peak_y, :]
        cross_db = 10 * np.log10(cross + 1e-10)

        axes2[0].plot(angles, cross, linewidth=1.5, label=tag)
        axes2[1].plot(angles, cross_db, linewidth=1.5, label=tag)

    axes2[0].set_ylabel("Normalized Intensity")
    axes2[0].set_title("Linear Scale")
    axes2[1].set_ylabel("Intensity (dB)")
    axes2[1].set_title("Log Scale")

    for ax in axes2:
        ax.set_xlabel("θ_x (°)")
        ax.set_xlim(-35, 35)
        ax.axvline(x=15, color="gray", linestyle=":", alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes2[1].set_ylim(-40, 0)
    fig2.suptitle("Far-Field Cross Section — Single Point at 15°", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "hogel_cross_section.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'hogel_cross_section.png'}")


# =============================================================
# 3. Sub-pixel Sizing Analysis
# =============================================================
def subpixel_sizing():
    """
    Relationship between sub-pixel pitch, count, and angular resolution.
    
    Key equations:
    - Max angle: sin(θ_max) = λ/(2·pitch)  [Nyquist]
    - Angular resolution: Δθ = λ/(N·pitch) = λ/hogel_size [Rayleigh]
    - Number of resolvable angles: N_angles = 2·sin(θ_max)·N·pitch/λ = N
    """
    print("\n" + "=" * 70)
    print("SUB-PIXEL SIZING ANALYSIS")
    print("=" * 70)

    pitches = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]) * 1e-6
    wavelengths = {"Red (632nm)": 632e-9, "Green (532nm)": 532e-9, "Blue (450nm)": 450e-9}
    colors = {"Red (632nm)": "red", "Green (532nm)": "green", "Blue (450nm)": "blue"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for wl_name, lam in wavelengths.items():
        # Max diffraction angle
        sin_max = np.minimum(lam / (2 * pitches), 1.0)
        theta_max = np.degrees(np.arcsin(sin_max))

        # Sub-pixels per hogel
        N_sub = (HOGEL_SIZE / pitches).astype(int)

        # Angular resolution
        delta_theta = np.degrees(lam / (N_sub * pitches))

        # Total resolvable directions (in 2D: N_sub²)
        N_directions_1d = N_sub
        N_directions_2d = N_sub ** 2

        axes[0, 0].plot(pitches * 1e6, theta_max, "o-", color=colors[wl_name],
                       linewidth=2, label=wl_name)
        axes[0, 1].plot(pitches * 1e6, N_sub, "o-", color=colors[wl_name],
                       linewidth=2, label=wl_name)
        axes[1, 0].plot(pitches * 1e6, delta_theta, "o-", color=colors[wl_name],
                       linewidth=2, label=wl_name)
        axes[1, 1].plot(pitches * 1e6, N_directions_2d, "o-", color=colors[wl_name],
                       linewidth=2, label=wl_name)

    axes[0, 0].axhline(y=30, color="k", linestyle="--", alpha=0.5, label="30° target")
    axes[0, 0].set_ylabel("Max Diffraction Angle (°)")
    axes[0, 0].set_title("Field of View vs Pitch")
    axes[0, 0].legend()

    axes[0, 1].set_ylabel("Sub-pixels per Row")
    axes[0, 1].set_title("Sub-pixel Count (1D)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()

    axes[1, 0].set_ylabel("Angular Resolution Δθ (°)")
    axes[1, 0].set_title("Angular Resolution (Rayleigh)")
    axes[1, 0].legend()

    axes[1, 1].set_ylabel("Resolvable Directions (2D)")
    axes[1, 1].set_title("Total Directions per Hogel")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Sub-pixel Pitch (µm)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Sub-pixel Design Space — Hogel = {HOGEL_SIZE*1e6:.1f}µm (150 PPI)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "subpixel_design.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'subpixel_design.png'}")

    # Summary table
    print(f"\n{'Pitch (µm)':<12} {'Max θ (°)':<12} {'N (1D)':<10} {'N² (2D)':<12} "
          f"{'Δθ (°)':<10} {'FoV OK?':<8}")
    print("-" * 70)
    for p in pitches:
        sin_max = min(532e-9 / (2 * p), 1.0)
        theta = np.degrees(np.arcsin(sin_max))
        n = int(HOGEL_SIZE / p)
        dtheta = np.degrees(532e-9 / (n * p))
        fov_ok = "YES" if theta >= 30 else "no"
        print(f"{p*1e6:<12.1f} {theta:<12.1f} {n:<10} {n**2:<12,} "
              f"{dtheta:<10.3f} {fov_ok:<8}")

    # Key finding about pitch vs phase bits trade-off
    print(f"\nKEY INSIGHT: Larger pitch → fewer sub-pixels → lower angular resolution")
    print(f"But larger pitch also → easier fabrication + less fringing")
    print(f"At 1.0µm pitch: still 30° FoV for blue/green, 169 sub-pixels/row")
    print(f"At 2.0µm pitch: only 15° FoV but trivial to fabricate")


if __name__ == "__main__":
    print("Hogel Projection Analysis")
    print("=" * 50)
    quantization_study()
    hogel_projection()
    subpixel_sizing()
    print("\nDone!")

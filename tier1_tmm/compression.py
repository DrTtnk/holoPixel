"""
Data Compression Strategies for Holographic Display

Problem: 1920×1080 hogels × 338² sub-pixels × 3 bits = 89 GB/frame
Target: 1 FPS refresh → 89 GB/s raw bandwidth

Strategies explored:
  1. Angular subsampling — not all angular bins need data
  2. Holographic sparsity — most hogels have few active directions
  3. Hierarchical encoding — coarse angular field + residuals
  4. Zone-based addressing — only update changed regions
  5. Symmetry exploitation — use holographic symmetries
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# =============================================================
# Baseline calculation
# =============================================================
def baseline_budget():
    """Raw data budget for the display."""
    print("=" * 70)
    print("BASELINE DATA BUDGET")
    print("=" * 70)

    hogels_x, hogels_y = 1920, 1080
    sub_per_hogel = 338
    phase_bits = 3
    colors = 3  # RGB
    fps = 1

    total_subpix = hogels_x * hogels_y * sub_per_hogel ** 2
    bits_per_frame = total_subpix * phase_bits * colors
    bytes_per_frame = bits_per_frame / 8
    gb_per_frame = bytes_per_frame / 1e9

    print(f"Resolution:       {hogels_x}×{hogels_y} hogels (Full HD)")
    print(f"Sub-pixels/hogel: {sub_per_hogel}² = {sub_per_hogel**2:,}")
    print(f"Total sub-pixels: {total_subpix:,.0f} ({total_subpix/1e9:.1f}B)")
    print(f"Phase bits:       {phase_bits}")
    print(f"Colors:           {colors} (RGB)")
    print(f"")
    print(f"Raw data/frame:   {gb_per_frame:.1f} GB ({bits_per_frame/1e12:.2f} Tbit)")
    print(f"At {fps} FPS:        {gb_per_frame * fps:.1f} GB/s")

    return hogels_x, hogels_y, sub_per_hogel, phase_bits, colors


# =============================================================
# Strategy 1: Angular subsampling
# =============================================================
def strategy_angular_subsampling():
    """
    Insight: at 0.5µm pitch, max angle is ±32°.
    But the useful FoV might be smaller for a given viewing zone.
    
    Also: human angular acuity is ~1 arcmin (0.017°).
    At 0.5m viewing distance, 0.18° hogel angular resolution is already
    10× coarser than human acuity. So angular supersampling is wasteful.
    
    Key: only a fraction of the 338² angular bins are actually needed.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 1: ANGULAR SUBSAMPLING")
    print("=" * 70)

    sub_per_hogel = 338
    fov_max = 32  # degrees for green

    # For a viewing zone (eye box):
    # At distance D, eye box width W, the angular range is ±atan(W/2D)
    viewing_distances = [0.3, 0.5, 0.7, 1.0]  # meters
    eye_box_widths = [50e-3, 100e-3, 200e-3]  # 5cm, 10cm, 20cm

    print(f"\nUsed angular range (degrees) for viewing zone:")
    print(f"{'Eye box':<12}", end="")
    for d in viewing_distances:
        print(f"  D={d}m  ", end="")
    print()
    print("-" * 55)

    results = {}
    for w in eye_box_widths:
        print(f"{w*1e3:.0f}mm     ", end="")
        for d in viewing_distances:
            angle = np.degrees(np.arctan(w / (2 * d)))
            frac = (2 * angle) / (2 * fov_max)
            bins_needed = int(frac * sub_per_hogel)
            print(f" ±{angle:5.1f}°   ", end="")
            results[(w, d)] = (angle, frac, bins_needed)
        print()

    print(f"\nCompression ratio (angular bins actually needed / 338):")
    print(f"{'Eye box':<12}", end="")
    for d in viewing_distances:
        print(f"  D={d}m  ", end="")
    print()
    print("-" * 55)

    for w in eye_box_widths:
        print(f"{w*1e3:.0f}mm     ", end="")
        for d in viewing_distances:
            angle, frac, bins = results[(w, d)]
            # But display is 34mm, so hogels at edges see slightly different angle
            # For a single viewer, we need the union of all hogel FoVs
            # This is approximately frac per axis, frac² for 2D
            print(f" {frac*100:5.1f}%    ", end="")
        print()

    print(f"\nKey insight: a single viewer with 100mm eye box at 0.5m needs")
    angle, frac, bins = results[(100e-3, 0.5)]
    print(f"only ±{angle:.1f}° = {frac*100:.0f}% of each angular axis")
    print(f"= {frac**2*100:.0f}% of angular bins = {int(frac * 338)}×{int(frac * 338)} effective sub-pixels")
    print(f"Compression: {1/frac**2:.1f}×")


# =============================================================
# Strategy 2: Holographic sparsity
# =============================================================
def strategy_sparsity():
    """
    In a typical 3D scene, each hogel only needs to produce
    a few dozen directions (sparse angular pattern), not 338².
    
    A scene with 1000 point sources → each hogel needs ~1000 non-zero
    angular bins (likely fewer due to occlusion).
    """
    print("\n" + "=" * 70)
    print("STRATEGY 2: HOLOGRAPHIC SPARSITY")
    print("=" * 70)

    sub_per_hogel = 338
    total_angular_bins = sub_per_hogel ** 2  # 114,244

    # Simulate sparsity for different scene complexities
    scene_complexities = [100, 1000, 10000, 50000, 100000]

    print(f"Total angular bins per hogel: {total_angular_bins:,}")
    print(f"\n{'Scene points':<15} {'Active bins':<15} {'Sparsity':<12} {'Compression':<12}")
    print("-" * 55)

    for n_pts in scene_complexities:
        # Each point lights up ~1 angular bin per hogel
        # With some overlap (multiple points in same bin)
        active_bins = min(n_pts, total_angular_bins)
        # Account for overlap with Poisson approximation
        if n_pts < total_angular_bins:
            active_bins = int(total_angular_bins * (1 - np.exp(-n_pts / total_angular_bins)))
        sparsity = active_bins / total_angular_bins
        compression = 1 / sparsity if sparsity > 0 else float('inf')
        print(f"{n_pts:<15,} {active_bins:<15,} {sparsity:<12.1%} {compression:<12.1f}×")

    print(f"\nFor a typical scene (10k polygons → ~10k visible points per hogel):")
    n_pts = 10000
    active = int(total_angular_bins * (1 - np.exp(-n_pts / total_angular_bins)))
    print(f"Active bins: {active:,} / {total_angular_bins:,} = {active/total_angular_bins:.1%}")
    print(f"Sparse encoding: store only active (index, value) pairs")
    print(f"Per active bin: ~17 bits index + 3 bits value = 20 bits")
    print(f"Sparse data: {active * 20 / 8 / 1024:.1f} KB per hogel per color")
    print(f"vs dense: {total_angular_bins * 3 / 8 / 1024:.1f} KB per hogel per color")

    # But wait: the PHASE PATTERN (sub-pixel array) is NOT sparse even if
    # the far-field is sparse. The phase hologram is always dense.
    print(f"\n⚠ CRITICAL: The phase hologram itself is always DENSE")
    print(f"  Even for 1 point source, all {total_angular_bins:,} sub-pixels need values")
    print(f"  Sparsity helps with COMPUTATION (hologram generation),")
    print(f"  not with STORAGE/BANDWIDTH of the phase pattern")
    print(f"\n  The compression must happen at the PHASE PATTERN level!")


# =============================================================
# Strategy 3: Phase pattern compression
# =============================================================
def strategy_phase_compression():
    """
    Phase holograms have specific statistical properties:
    - For point sources: linear phase ramps (highly compressible)
    - For multiple sources: sum of complex exponentials
    - 3-bit quantization already limits entropy
    
    Key: with 3-bit (8 levels), max entropy = 3 bits/pixel.
    Real holograms have lower entropy due to correlations.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 3: PHASE PATTERN COMPRESSION")
    print("=" * 70)

    N = 338  # sub-pixels per row

    # Generate realistic holograms and measure compressibility
    rng = np.random.RandomState(42)
    n_levels = 8
    scenarios = []

    # Scenario A: single point source (linear ramp)
    x = np.arange(N)
    X, Y = np.meshgrid(x, x)
    phase = (0.3 * X + 0.2 * Y) % (2 * np.pi)
    quantized = (phase / (2 * np.pi) * n_levels).astype(int) % n_levels
    scenarios.append(("Single point (ramp)", quantized))

    # Scenario B: 10 point sources
    field = np.zeros((N, N), dtype=complex)
    for _ in range(10):
        kx = rng.uniform(-0.5, 0.5)
        ky = rng.uniform(-0.5, 0.5)
        field += np.exp(1j * 2 * np.pi * (kx * X + ky * Y))
    phase = np.angle(field) % (2 * np.pi)
    quantized = (phase / (2 * np.pi) * n_levels).astype(int) % n_levels
    scenarios.append(("10 points", quantized))

    # Scenario C: 100 point sources
    field = np.zeros((N, N), dtype=complex)
    for _ in range(100):
        kx = rng.uniform(-0.5, 0.5)
        ky = rng.uniform(-0.5, 0.5)
        field += np.exp(1j * 2 * np.pi * (kx * X + ky * Y))
    phase = np.angle(field) % (2 * np.pi)
    quantized = (phase / (2 * np.pi) * n_levels).astype(int) % n_levels
    scenarios.append(("100 points", quantized))

    # Scenario D: 1000 point sources (approaches random)
    field = np.zeros((N, N), dtype=complex)
    for _ in range(1000):
        kx = rng.uniform(-0.5, 0.5)
        ky = rng.uniform(-0.5, 0.5)
        field += np.exp(1j * 2 * np.pi * (kx * X + ky * Y))
    phase = np.angle(field) % (2 * np.pi)
    quantized = (phase / (2 * np.pi) * n_levels).astype(int) % n_levels
    scenarios.append(("1000 points", quantized))

    # Measure entropy and compression ratios
    print(f"\n{'Scenario':<20} {'Entropy':<10} {'Raw':<12} {'Predictive':<12} {'2D-DCT':<12}")
    print("-" * 66)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col, (label, data) in enumerate(scenarios):
        # Shannon entropy
        hist = np.histogram(data, bins=n_levels, range=(0, n_levels))[0]
        probs = hist / hist.sum()
        entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))

        # Prediction residuals (subtract from neighbor)
        residuals = np.diff(data.astype(int), axis=1) % n_levels
        hist_r = np.histogram(residuals, bins=n_levels, range=(0, n_levels))[0]
        probs_r = hist_r / hist_r.sum()
        entropy_pred = -np.sum(probs_r[probs_r > 0] * np.log2(probs_r[probs_r > 0]))

        # 2D DCT-like: measure energy concentration
        from scipy.fft import dctn
        coeffs = dctn(data.astype(float))
        sorted_coeffs = np.sort(np.abs(coeffs.ravel()))[::-1]
        cumulative_energy = np.cumsum(sorted_coeffs ** 2) / np.sum(sorted_coeffs ** 2)
        # Fraction of coefficients needed for 99% energy
        n_99 = np.searchsorted(cumulative_energy, 0.99) + 1
        dct_ratio = n_99 / data.size

        raw_bits = 3.0  # bits per pixel
        pred_bits = entropy_pred
        dct_bits = raw_bits * dct_ratio  # approximate

        print(f"{label:<20} {entropy:<10.2f} {raw_bits:<12.1f} "
              f"{pred_bits:<12.2f} {dct_bits:<12.2f}")

        # Visualize
        axes[0, col].imshow(data, cmap="twilight", vmin=0, vmax=n_levels - 1)
        axes[0, col].set_title(label, fontsize=10)

        axes[1, col].hist(residuals.ravel(), bins=n_levels, range=(0, n_levels),
                         density=True, alpha=0.7, color="steelblue")
        axes[1, col].set_title(f"Residual H={entropy_pred:.2f} bits", fontsize=10)
        axes[1, col].set_xlabel("Residual value")

    axes[0, 0].set_ylabel("Phase pattern")
    axes[1, 0].set_ylabel("Probability")

    fig.suptitle("Phase Hologram Compressibility Analysis\n"
                 "3-bit (8 level) quantized phase, 338×338 sub-pixels",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "compression_analysis.png", dpi=150)
    print(f"\nSaved: {PLOT_DIR / 'compression_analysis.png'}")

    # Final bandwidth estimates
    print(f"\n{'='*55}")
    print("BANDWIDTH ESTIMATES")
    print(f"{'='*55}")

    hogels = 1920 * 1080
    subpix = 338 ** 2
    colors = 3

    strategies = [
        ("Raw (no compression)", 3.0),
        ("Predictive coding (10 pts)", 1.5),
        ("Predictive coding (100 pts)", 2.5),
        ("Angular subsampling (100mm eye box, 0.5m)", 3.0 * 0.156 ** 2),
        ("Subsampling + predictive", 1.5 * 0.156 ** 2),
    ]

    # For angular subsampling, effective sub-pixels reduce
    frac = 0.156  # from earlier: 15.6% of angular range
    effective_subpix_angular = int(338 * frac) ** 2

    print(f"\n{'Strategy':<45} {'bits/px':<10} {'GB/frame':<10} {'GB/s@1FPS':<10}")
    print("-" * 75)
    for label, bpp in strategies:
        if "ubsampl" in label:
            total_bits = hogels * effective_subpix_angular * bpp * colors
        else:
            total_bits = hogels * subpix * bpp * colors
        gb = total_bits / 8 / 1e9
        print(f"{label:<45} {bpp:<10.2f} {gb:<10.1f} {gb:<10.1f}")

    print(f"\n💡 Combined strategy (subsampling + predictive + eye tracking):")
    best_bits = hogels * effective_subpix_angular * 1.5 * colors
    best_gb = best_bits / 8 / 1e9
    print(f"   {best_gb:.2f} GB/frame = {best_gb * 8:.2f} Gbit/s @ 1 FPS")
    print(f"   That's {89/best_gb:.0f}× compression vs raw 89 GB")
    print(f"   Comparable to 8K video data rate")


# =============================================================
# Strategy 4: Compute vs Transfer trade-off
# =============================================================
def strategy_compute_local():
    """
    Alternative: don't transfer phase patterns at all.
    Transfer the 3D SCENE and compute holograms locally.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 4: LOCAL HOLOGRAM COMPUTATION")
    print("=" * 70)

    print("""
Instead of transferring phase patterns, transfer the 3D scene
and compute holograms per-hogel on the display's local processor.

Scene data per frame:
  - 100k polygons × 32 bytes/polygon = 3.2 MB
  - 1M polygons × 32 bytes/polygon = 32 MB
  - Texture data: ~100 MB

Hologram computation per hogel:
  - Gerchberg-Saxton algorithm: ~10 FFT iterations
  - FFT of 338×338 = O(338² × log(338²)) ≈ 4M ops per iteration
  - Per hogel: ~40M ops
  - Total: 2M hogels × 40M ops = 80 Peta-ops per frame

At 1 FPS:
  - Need 80 PFLOPS
  - GPU: A100 = 300 TFLOPS → need ~270 GPUs
  - Wafer-scale: Cerebras CS-2 ≈ 1 EFLOP → possible!
  - Custom ASIC: feasible with massive parallelism
  
  But each hogel is INDEPENDENT → embarrassingly parallel!
  Per-hogel compute: 40M ops / 1s = 40 MFLOPS (trivial)
  
  ⇒ Embed a tiny processor per hogel (or per tile of hogels)
     Each computes its own hologram from the scene description
     
  Data transfer: 32 MB scene data via broadcast to all hogel tiles
  Much better than 89 GB phase patterns!
""")

    # Compare bandwidth requirements
    approaches = {
        "Raw phase patterns": 89e9 * 8,  # bits
        "Compressed (subsampling+pred)": 0.54e9 * 8,
        "Scene description (100k poly)": 3.2e6 * 8,
        "Scene description (1M poly)": 32e6 * 8,
        "Scene + textures": 132e6 * 8,
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(approaches.keys())
    values = [v / 8e9 for v in approaches.values()]  # GB
    colors = ["red", "orange", "green", "green", "yellowgreen"]

    bars = ax.barh(labels, values, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Data per frame (GB)")
    ax.set_title("Bandwidth Comparison: Transfer Strategy")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() * 1.2, bar.get_y() + bar.get_height() / 2,
               f"{val:.3g} GB", va="center", fontsize=10)

    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "compression_bandwidth.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'compression_bandwidth.png'}")


if __name__ == "__main__":
    print("Data Compression Strategies")
    print("=" * 50)
    hx, hy, sub, bits, colors = baseline_budget()
    strategy_angular_subsampling()
    strategy_sparsity()
    strategy_phase_compression()
    strategy_compute_local()
    print("\nDone!")

"""
Extended Acoustic Analogies for Holographic Pixel Design

Beyond the DRAFT's three architectures (Pipe Organ, Trumpet Valve, Chladni Plate),
we explore additional musical/acoustic systems mapped to nanophotonics.

Each analogy maps: Sound phenomenon → Optical mechanism → Hogel design
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# =============================================================
# Architecture D: "Whispering Gallery" — Microring Resonators
# =============================================================
# Sound: In St. Paul's Cathedral, a whisper travels around the dome
# Optics: Light circulates in a microring, accumulating massive phase
# Advantage: Phase accumulation scales with circumference, not depth
#
# Key physics: ring circumference = 2πR, modes at n·2πR = m·λ
# Phase sensitivity: dφ/dn = 2πR · (2π/λ) · F_ring
# where F_ring is the ring finesse

def whispering_gallery_analysis():
    """
    Microring resonator: light circulates around a ring waveguide.
    Phase shift per Δn scales with circumference × finesse.
    
    For a 5µm radius ring at λ=632nm:
    - Circumference = 31.4 µm → fits inside a 169.3 µm hogel
    - Single-pass phase: 2π·n·C/λ = 2π·1.5·31.4/0.632 = ~468 rad
    - For Δn=0.1: Δφ_single = 2π·0.1·31.4/0.632 = 31.2 rad ≈ 10π
    """
    radii = np.array([1, 2, 3, 5, 8, 10, 15, 20]) * 1e-6  # meters
    wavelengths = {"Red": 632e-9, "Green": 532e-9, "Blue": 450e-9}
    dn = 0.1
    n_base = 1.5
    hogel_size = 169.3e-6

    print("=" * 70)
    print("ARCHITECTURE D: Whispering Gallery (Microring Resonator)")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"Red": "red", "Green": "green", "Blue": "blue"}

    for wl_name, lam in wavelengths.items():
        circumferences = 2 * np.pi * radii
        # Single-pass phase change for Δn
        delta_phi = 2 * np.pi * dn * circumferences / lam
        axes[0].plot(radii * 1e6, delta_phi / np.pi, "o-",
                     color=colors[wl_name], linewidth=2, label=wl_name)

        # How many rings fit in one hogel?
        rings_per_hogel = (hogel_size / (2 * radii)) ** 2  # area packing
        axes[1].plot(radii * 1e6, rings_per_hogel, "o-",
                     color=colors[wl_name], linewidth=2, label=wl_name)

    axes[0].axhline(y=2.0, color="k", linestyle="--", alpha=0.5, label="2π target")
    axes[0].set_xlabel("Ring Radius (µm)")
    axes[0].set_ylabel("Phase Shift from Δn=0.1 (×π)")
    axes[0].set_title("Single-Pass Phase (no resonance)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Ring Radius (µm)")
    axes[1].set_ylabel("Rings per 169µm Hogel")
    axes[1].set_title("Packing Density")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Architecture D: Whispering Gallery Microring", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "arch_d_whispering_gallery.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'arch_d_whispering_gallery.png'}")

    # Key finding
    R_min = 632e-9 / (2 * np.pi * dn * n_base)  # for 2π with single pass
    print(f"\nMinimum radius for 2π (single pass, red): {R_min*1e6:.2f} µm")
    print(f"Diameter: {2*R_min*1e6:.2f} µm (vs hogel size 169.3 µm)")
    print(f"With resonance (F=10): radius = {R_min*1e6/10:.3f} µm")
    print("→ Microrings easily fit multiple per hogel with Δn=0.1")


# =============================================================
# Architecture E: "Helmholtz Resonator" — Slot Waveguide
# =============================================================
# Sound: Air oscillates in the neck of a bottle, cavity stores energy
# Optics: Light concentrated in a narrow low-index slot between high-index rails
# Advantage: Extreme field confinement → maximum sensitivity to index changes

def helmholtz_resonator_analysis():
    """
    Slot waveguide: a narrow gap (~50-100nm) between two high-index rails.
    Light is confined in the slot by the discontinuity of the E-field
    at the dielectric boundary. The slot material is the EO modulator.
    
    Γ_slot = fraction of optical mode in the slot ≈ 0.3-0.5
    Phase change: Δφ = Γ_slot · 2π · Δn · L / λ
    """
    slot_widths = np.array([30, 50, 80, 100, 150, 200]) * 1e-9
    L_values = np.arange(1, 50) * 1e-6  # waveguide length
    lam = 632e-9
    dn = 0.1

    # Approximate slot confinement factor (from literature)
    # Γ increases as slot narrows (better confinement)
    gamma_slot = 0.5 * np.exp(-slot_widths / 100e-9)  # rough model
    gamma_slot = np.clip(gamma_slot, 0.1, 0.6)

    print("\n" + "=" * 70)
    print("ARCHITECTURE E: Helmholtz Resonator (Slot Waveguide)")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))
    for w, gamma in zip(slot_widths, gamma_slot):
        phase_shift = gamma * 2 * np.pi * dn * L_values / lam
        ax.plot(L_values * 1e6, phase_shift / np.pi,
                linewidth=2, label=f"slot={w*1e9:.0f}nm (Γ={gamma:.2f})")

    ax.axhline(y=2.0, color="k", linestyle="--", alpha=0.5, label="2π target")
    ax.axvline(x=169.3, color="gray", linestyle=":", alpha=0.5, label="Hogel size")
    ax.set_xlabel("Waveguide Length L (µm)")
    ax.set_ylabel("Phase Shift (×π)")
    ax.set_title("Slot Waveguide Phase Sensitivity (Δn=0.1, λ=632nm)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "arch_e_slot_waveguide.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'arch_e_slot_waveguide.png'}")

    # Minimum length for 2π
    gamma_typical = 0.3
    L_min = lam / (gamma_typical * dn)
    print(f"Minimum length for 2π (Γ=0.3): {L_min*1e6:.1f} µm")
    print(f"This is {L_min/169.3e-6:.1f}× the hogel size")
    print("→ Needs to be folded (serpentine) or used with resonance")


# =============================================================
# Architecture F: "Tuning Fork" — Coupled Resonators
# =============================================================
# Sound: Two tuning fork prongs vibrate, slight detuning creates beating
# Optics: Two coupled micro-cavities that split into bonding/antibonding modes
# Advantage: Phase sensitivity DOUBLES compared to single resonator

def tuning_fork_analysis():
    """
    Coupled resonators: two identical cavities with evanescent coupling.
    The mode splits into symmetric (bonding) and antisymmetric (antibonding).
    Near the coupling-induced anti-crossing, phase sensitivity is enhanced.
    
    The coupling adds an additional degree of freedom:
    - Detune one cavity relative to the other
    - The through-port phase response has an ultra-steep transition
    """
    # Model: two coupled Lorentzian resonances
    delta = np.linspace(-5, 5, 2000)  # detuning (normalized to linewidth)
    kappa_values = [0.1, 0.5, 1.0, 2.0, 4.0]  # coupling strength

    print("\n" + "=" * 70)
    print("ARCHITECTURE F: Tuning Fork (Coupled Resonators)")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for kappa in kappa_values:
        # Transfer matrix for two coupled resonances
        # Transmission: t = 1 / ((1 + i·δ)(1 + i·δ) + κ²)
        # This is the coupled-mode theory result
        t = 1.0 / ((1 + 1j * delta) * (1 + 1j * delta) + kappa ** 2)
        phase = np.unwrap(np.angle(t))

        # Phase derivative (sensitivity)
        d_phase = np.gradient(phase, delta)

        axes[0].plot(delta, phase / np.pi, linewidth=1.5, label=f"κ={kappa}")
        axes[1].plot(delta, np.abs(d_phase), linewidth=1.5, label=f"κ={kappa}")

    axes[0].set_xlabel("Detuning (×linewidth)")
    axes[0].set_ylabel("Phase (×π)")
    axes[0].set_title("Coupled Resonator Phase Response")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Detuning (×linewidth)")
    axes[1].set_ylabel("|dφ/dδ| (sensitivity)")
    axes[1].set_title("Phase Sensitivity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Architecture F: Coupled Resonators (Tuning Fork)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "arch_f_coupled_resonators.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'arch_f_coupled_resonators.png'}")
    print("→ Coupling provides additional control: sharper transitions near anti-crossing")


# =============================================================
# Architecture G: "Guitar String" — Guided Mode Resonance (GMR)
# =============================================================
# Sound: Standing wave on a string — frequency set by length & tension
# Optics: Light couples into a waveguide mode in a grating, re-radiates
# Advantage: A SINGLE thin grating layer acts as both modulator AND diffractor

def guitar_string_analysis():
    """
    Guided Mode Resonance: a sub-wavelength grating on a waveguide slab.
    Light couples into a guided mode via the grating, propagates laterally,
    then re-radiates. This creates an ultra-sharp resonance in reflection.
    
    The resonance condition: n_eff · Λ = m · λ
    where Λ is the grating period, n_eff is the effective guided mode index.
    
    Tuning n_eff (via EO material) shifts the resonance.
    Near resonance, the reflected phase changes by 2π over a very narrow range.
    
    KEY ADVANTAGE: The grating that creates the resonance IS the pixel grating.
    No separate modulator needed — the structure both phase-shifts AND diffracts.
    """
    print("\n" + "=" * 70)
    print("ARCHITECTURE G: Guitar String (Guided Mode Resonance)")
    print("=" * 70)

    # Model: Fano resonance lineshape
    # r(ω) = (ω - ω₀ + q·γ) / (ω - ω₀ + i·γ) where q is the Fano parameter
    omega = np.linspace(-10, 10, 5000)  # normalized detuning
    q_values = [0, 0.5, 1.0, 2.0, 5.0]  # Fano asymmetry

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for q in q_values:
        gamma = 1.0
        r = (omega + q * gamma) / (omega + 1j * gamma)
        phase = np.unwrap(np.angle(r))
        reflectance = np.abs(r) ** 2

        axes[0, 0].plot(omega, reflectance, linewidth=1.5, label=f"q={q}")
        axes[0, 1].plot(omega, phase / np.pi, linewidth=1.5, label=f"q={q}")

    axes[0, 0].set_title("Reflectance (Fano Lineshape)")
    axes[0, 0].set_ylabel("|r|²")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Reflected Phase")
    axes[0, 1].set_ylabel("Phase (×π)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # GMR advantage: dual-function pixel
    # Compare: GTE = separate modulator + separate grating
    #          GMR = one structure does both

    # Pitch requirements for ±30° FoV
    lam_values = np.array([450, 532, 632]) * 1e-9
    theta_max = 30  # degrees
    pitch = lam_values / np.sin(np.radians(theta_max))

    for lam, p in zip(lam_values, pitch):
        print(f"λ={lam*1e9:.0f}nm: grating pitch for ±30° = {p*1e9:.0f} nm")

    # Sub-pixel count comparison
    hogel_size = 169.3e-6
    for lam, p in zip(lam_values, pitch):
        n_subpixels = (hogel_size / p) ** 2
        print(f"  Sub-pixels per hogel: {n_subpixels:.0f}")

    # Plot: How GMR merges phase modulation and beam steering
    architectures = {
        "A: GTE (Pipe Organ)": {"layers": 4, "functions": "Separate modulator + grating"},
        "B: Si Photonics (Trumpet)": {"layers": 3, "functions": "Waveguide + grating coupler"},
        "G: GMR (Guitar String)": {"layers": 1, "functions": "Single resonant grating"},
    }

    labels = list(architectures.keys())
    layer_counts = [v["layers"] for v in architectures.values()]

    axes[1, 0].barh(labels, layer_counts, color=["steelblue", "coral", "gold"])
    axes[1, 0].set_xlabel("Number of Functional Layers")
    axes[1, 0].set_title("Fabrication Complexity")

    # Phase accumulation: GMR vs GTE
    # GMR quality factor Q determines the phase sensitivity
    Q_values = np.logspace(1, 4, 100)
    dn = 0.1
    n_eff = 1.8  # typical for a slab waveguide
    lam = 532e-9

    # GMR phase shift ≈ 2π for Q > λ/(n_eff · Δn · Λ) roughly
    # Simpler: phase sensitivity scales with Q
    delta_phi_gmr = 2 * np.arctan(Q_values * dn / n_eff)
    axes[1, 1].plot(Q_values, delta_phi_gmr / np.pi, "k-", linewidth=2)
    axes[1, 1].axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="π (usable)")
    axes[1, 1].axhline(y=1.84, color="g", linestyle="--", alpha=0.5, label="1.84π (ideal)")
    axes[1, 1].set_xlabel("GMR Quality Factor Q")
    axes[1, 1].set_ylabel("Phase Range (×π)")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("GMR Phase Range vs Quality Factor")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Architecture G: Guided Mode Resonance (Guitar String)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "arch_g_gmr.png", dpi=150)
    print(f"\nSaved: {PLOT_DIR / 'arch_g_gmr.png'}")
    print("→ GMR merges phase modulation + beam steering in ONE layer")
    print("→ Potentially the simplest fabrication path")


# =============================================================
# Architecture H: "Didgeridoo" — Multi-Order Cavity
# =============================================================
# Sound: The didgeridoo produces multiple harmonics simultaneously
# Optics: A single thick cavity where R, G, B each use different orders
# Advantage: ONE cavity depth serves all three colors

def didgeridoo_analysis():
    """
    Multi-order cavity: find a depth d where all three wavelengths
    have a resonance order m such that n_base·d = m·λ/2.
    
    This is a number theory problem: find d such that
    2·n·d/λ_R, 2·n·d/λ_G, 2·n·d/λ_B are all near integers.
    """
    print("\n" + "=" * 70)
    print("ARCHITECTURE H: Didgeridoo (Multi-Order Cavity)")
    print("=" * 70)

    n = 1.5
    lam_r, lam_g, lam_b = 632e-9, 532e-9, 450e-9

    depths = np.arange(100, 5001, 1) * 1e-9

    # For each depth, compute how close each wavelength is to a resonance
    def resonance_error(d, lam):
        m = 2 * n * d / lam
        return abs(m - round(m))

    err_r = np.array([resonance_error(d, lam_r) for d in depths])
    err_g = np.array([resonance_error(d, lam_g) for d in depths])
    err_b = np.array([resonance_error(d, lam_b) for d in depths])

    # Combined error (all three must be near resonance)
    combined_err = err_r + err_g + err_b

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(depths * 1e9, err_r, "r-", alpha=0.7, linewidth=0.5, label="Red")
    axes[0].plot(depths * 1e9, err_g, "g-", alpha=0.7, linewidth=0.5, label="Green")
    axes[0].plot(depths * 1e9, err_b, "b-", alpha=0.7, linewidth=0.5, label="Blue")
    axes[0].set_ylabel("Distance to nearest resonance (×FSR)")
    axes[0].set_title("Individual Resonance Errors")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(depths * 1e9, combined_err, "k-", linewidth=0.5)
    axes[1].set_xlabel("Cavity Depth d (nm)")
    axes[1].set_ylabel("Combined Error")
    axes[1].set_title("RGB Combined Resonance Error")
    axes[1].grid(True, alpha=0.3)

    # Find best depths
    threshold = 0.15  # each color within 15% of resonance
    good = (err_r < threshold) & (err_g < threshold) & (err_b < threshold)
    if np.any(good):
        good_depths = depths[good]
        axes[1].scatter(good_depths * 1e9, combined_err[good], color="gold", s=10, zorder=5)

        # Sort by combined error and show top 10
        sorted_idx = np.argsort(combined_err[good])
        print("\nBest common depths (all colors within 15% of resonance):")
        seen_ranges = set()
        count = 0
        for idx in sorted_idx:
            d = good_depths[idx]
            d_nm = round(d * 1e9)
            range_key = d_nm // 10
            if range_key in seen_ranges:
                continue
            seen_ranges.add(range_key)

            m_r = round(2 * n * d / lam_r)
            m_g = round(2 * n * d / lam_g)
            m_b = round(2 * n * d / lam_b)
            print(f"  d = {d_nm} nm: m_R={m_r}, m_G={m_g}, m_B={m_b}, "
                  f"err = {combined_err[np.where(depths == d)[0][0]]:.4f}")
            count += 1
            if count >= 10:
                break
    else:
        print("No depths with all colors within 15% of resonance.")

    fig.suptitle("Architecture H: Multi-Order Cavity (Didgeridoo)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "arch_h_multiorder.png", dpi=150)
    print(f"\nSaved: {PLOT_DIR / 'arch_h_multiorder.png'}")


# =============================================================
# Summary: Compare all architectures
# =============================================================
def architecture_comparison():
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 70)

    archs = [
        ("A: Pipe Organ (GTE)",
         "Resonant cavity with EO fill",
         "Best understood, proven physics",
         "Needs per-color depth OR high finesse",
         "Moderate"),
        ("B: Trumpet Valve (Si Photonics)",
         "Horizontal MZI + grating coupler",
         "Decouples modulation from emission",
         "Complex routing, large footprint",
         "High"),
        ("C: Chladni Plate (SAW)",
         "Surface acoustic waves diffract light",
         "No per-pixel electronics needed!",
         "GHz RF drive, coupling efficiency",
         "Low-Med"),
        ("D: Whispering Gallery (Microring)",
         "Light circulates in ring waveguide",
         "Huge effective path length",
         "Narrow bandwidth, thermal sensitivity",
         "High"),
        ("E: Helmholtz (Slot Waveguide)",
         "Field concentrated in narrow gap",
         "Maximum index sensitivity",
         "Propagation loss, serpentine routing",
         "High"),
        ("F: Tuning Fork (Coupled Cavities)",
         "Two coupled resonators",
         "Enhanced sensitivity at anti-crossing",
         "Alignment precision, 2× complexity",
         "Very High"),
        ("G: Guitar String (GMR)",
         "Resonant grating = modulator + diffractor",
         "Simplest structure, dual function!",
         "Q vs bandwidth trade-off",
         "LOW"),
        ("H: Didgeridoo (Multi-Order)",
         "One cavity depth for RGB",
         "Manufacturing simplicity",
         "Tolerance to depth errors tighter",
         "Moderate"),
    ]

    for name, mechanism, pro, con, complexity in archs:
        print(f"\n{name}")
        print(f"  Mechanism:  {mechanism}")
        print(f"  Pro:        {pro}")
        print(f"  Con:        {con}")
        print(f"  Fab. Complexity: {complexity}")


if __name__ == "__main__":
    print("Extended Acoustic Analogies Analysis")
    print("=" * 50)
    whispering_gallery_analysis()
    helmholtz_resonator_analysis()
    tuning_fork_analysis()
    guitar_string_analysis()
    didgeridoo_analysis()
    architecture_comparison()
    print("\nDone!")

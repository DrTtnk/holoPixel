"""
Actuation Feasibility Analysis

For each architecture, determine: can we change the optical phase
with an electrical signal at 0.5µm sub-pixel pitch in < 30ms?

Actuation mechanisms evaluated:
1. Electro-optic (Pockels) — BaTiO₃, LiNbO₃
2. Liquid Crystal — nematic birefringence
3. Phase Change Material — GST, Sb₂Se₃
4. Thermo-optic — heaters
5. MEMS — electrostatic membrane displacement
6. Carrier injection — free carrier plasma effect in Si
7. Electrochromic — ion intercalation (WO₃)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# =============================================================
# Material properties for actuation mechanisms
# =============================================================

def pockels_dn(material, voltage, thickness):
    """
    Electro-optic (Pockels) effect: Δn = 0.5 · r · n³ · E
    E = V / d (electric field across the layer)
    """
    materials = {
        "LiNbO3": {"r33": 31e-12, "n": 2.20},      # lithium niobate
        "BaTiO3": {"r42": 1300e-12, "n": 2.40},     # barium titanate (HUGE)
        "BaTiO3_thin": {"r42": 300e-12, "n": 2.30}, # thin film (reduced)
        "PZT": {"r33": 100e-12, "n": 2.45},         # lead zirconate titanate
        "KNbO3": {"r42": 380e-12, "n": 2.23},       # potassium niobate
        "LiTaO3": {"r33": 30e-12, "n": 2.18},       # lithium tantalate
    }
    m = materials[material]
    E = voltage / thickness  # V/m
    r = m.get("r42", m.get("r33"))
    return 0.5 * r * m["n"] ** 3 * E


def lc_dn(lc_type="5CB"):
    """
    Liquid crystal birefringence: Δn is the maximum swing between
    parallel and perpendicular alignment.
    """
    lc_data = {
        "5CB": {"dn": 0.18, "speed_ms": 5.0},      # cyanobiphenyl
        "E7":  {"dn": 0.22, "speed_ms": 8.0},       # Merck E7 mixture
        "TL205": {"dn": 0.26, "speed_ms": 10.0},    # high birefringence
        "LC_blue_phase": {"dn": 0.05, "speed_ms": 0.1},  # blue phase LC (FAST!)
    }
    return lc_data[lc_type]


# =============================================================
# 1. EO material analysis: Δn vs voltage at different thicknesses
# =============================================================
def eo_analysis():
    print("=" * 70)
    print("ELECTRO-OPTIC (POCKELS) ACTUATION")
    print("=" * 70)

    voltages = np.linspace(0.5, 15, 100)
    thicknesses = [200e-9, 300e-9, 500e-9, 1000e-9, 2000e-9]  # cavity depths

    materials = ["LiNbO3", "BaTiO3_thin", "PZT", "KNbO3"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, material in zip(axes.flat, materials):
        for d in thicknesses:
            dn = np.array([pockels_dn(material, V, d) for V in voltages])
            ax.plot(voltages, dn, linewidth=1.5, label=f"d={d*1e9:.0f}nm")

        ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.5, label="Δn=0.05 (usable)")
        ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Δn=0.1 (target)")
        ax.axvline(x=3.3, color="gray", linestyle=":", alpha=0.3, label="3.3V CMOS")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Δn")
        ax.set_title(material)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, min(0.3, ax.get_ylim()[1]))

    fig.suptitle("Electro-Optic Δn vs Applied Voltage", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "actuation_eo.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'actuation_eo.png'}")

    # Key numbers
    for material in materials:
        dn_3v_300nm = pockels_dn(material, 3.3, 300e-9)
        dn_10v_300nm = pockels_dn(material, 10, 300e-9)
        dn_3v_1um = pockels_dn(material, 3.3, 1000e-9)
        print(f"\n{material}:")
        print(f"  Δn at 3.3V, d=300nm: {dn_3v_300nm:.4f}")
        print(f"  Δn at 10V,  d=300nm: {dn_10v_300nm:.4f}")
        print(f"  Δn at 3.3V, d=1µm:   {dn_3v_1um:.4f}")


# =============================================================
# 2. LC in resonant cavity: does it work at 0.5µm pitch?
# =============================================================
def lc_analysis():
    print("\n" + "=" * 70)
    print("LIQUID CRYSTAL ACTUATION")
    print("=" * 70)

    # Fringing field analysis: LC needs vertical E-field alignment
    # At 0.5µm pitch with cavity depth d, the aspect ratio is d/pitch
    # Fringing fields become dominant when aspect ratio > ~0.3

    pitches = np.array([0.5, 1.0, 2.0, 5.0, 10.0]) * 1e-6
    depths = np.array([200, 300, 500, 1000, 2000]) * 1e-9

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Aspect ratio map
    aspect_ratios = np.outer(depths, 1.0 / pitches)
    im = axes[0].imshow(aspect_ratios, aspect="auto", cmap="RdYlGn_r",
                         extent=[0, len(pitches), 0, len(depths)])
    axes[0].set_xticks(np.arange(len(pitches)) + 0.5)
    axes[0].set_xticklabels([f"{p*1e6:.1f}" for p in pitches])
    axes[0].set_yticks(np.arange(len(depths)) + 0.5)
    axes[0].set_yticklabels([f"{d*1e9:.0f}" for d in depths])
    axes[0].set_xlabel("Sub-pixel Pitch (µm)")
    axes[0].set_ylabel("Cavity Depth (nm)")
    axes[0].set_title("Aspect Ratio d/pitch\n(>0.3 = fringing problem)")
    plt.colorbar(im, ax=axes[0])

    # Add text annotations
    for i, d in enumerate(depths):
        for j, p in enumerate(pitches):
            ar = d / p
            color = "white" if ar > 0.3 else "black"
            axes[0].text(j + 0.5, i + 0.5, f"{ar:.2f}",
                        ha="center", va="center", color=color, fontsize=9)

    # Plot 2: LC switching speed vs layer thickness
    # Relaxation time τ ∝ γ·d² / (K·π²) where γ=viscosity, K=elastic constant
    # For 5CB: γ ≈ 0.1 Pa·s, K ≈ 6.5 pN
    gamma = 0.1  # Pa·s
    K = 6.5e-12  # N
    d_sweep = np.linspace(50, 5000, 500) * 1e-9
    tau_off = gamma * d_sweep ** 2 / (K * np.pi ** 2)  # off (relaxation) time
    # On time is faster: τ_on = τ_off / (V/V_th)² - 1), roughly τ_off/10 at 3× threshold

    axes[1].semilogy(d_sweep * 1e9, tau_off * 1e3, "b-", linewidth=2, label="Off (relaxation)")
    axes[1].semilogy(d_sweep * 1e9, tau_off * 1e3 / 10, "r--", linewidth=2, label="On (3× threshold)")
    axes[1].axhline(y=30, color="k", linestyle="--", alpha=0.5, label="30 ms limit")
    axes[1].axhline(y=1000, color="gray", linestyle=":", alpha=0.5, label="1 FPS (1000ms)")
    axes[1].set_xlabel("LC Layer Thickness (nm)")
    axes[1].set_ylabel("Switching Time (ms)")
    axes[1].set_title("LC Switching Speed")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Effective Δn considering cavity enhancement
    lc_types = {"5CB": 0.18, "E7": 0.22, "TL205": 0.26, "Blue Phase": 0.05}
    R1_values = np.linspace(0.5, 0.98, 100)

    for name, dn_lc in lc_types.items():
        # In a GTE, the required Δn is reduced by enhancement
        # But we need at least the FSR-limited Δn
        # The GTE needs Δn_actual to cover the transition width
        # With the full LC birefringence, what R₁ do we need?
        # From our earlier result: Δn_transition ≈ λ/(2d·F)
        # So F_needed = λ/(2d·Δn_lc)
        F_needed_red = 632e-9 / (2 * 1053e-9 * dn_lc)
        F_needed_green = 532e-9 / (2 * 1053e-9 * dn_lc)
        F_needed_blue = 450e-9 / (2 * 1053e-9 * dn_lc)

        axes[2].barh(name, [F_needed_red],
                     color="red" if name != "Blue Phase" else "cyan",
                     alpha=0.7)
        print(f"\n{name} (Δn={dn_lc}):")
        print(f"  F needed (red, d=1053nm):   {F_needed_red:.0f} → R₁ = "
              f"{((F_needed_red-1)/(F_needed_red+1))**2:.3f}")
        print(f"  F needed (green):            {F_needed_green:.0f}")
        print(f"  F needed (blue):             {F_needed_blue:.0f}")

    axes[2].axvline(x=30, color="gray", linestyle="--", alpha=0.5, label="F=30 (moderate)")
    axes[2].set_xlabel("Required Enhancement Factor F")
    axes[2].set_title("Required GTE Finesse for LC Types\n(d=1053nm, red)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Liquid Crystal Actuation Feasibility", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "actuation_lc.png", dpi=150)
    print(f"\nSaved: {PLOT_DIR / 'actuation_lc.png'}")

    # Critical finding: fringing at 0.5µm pitch
    ar_05um_300nm = 300e-9 / 0.5e-6
    ar_05um_1um = 1000e-9 / 0.5e-6
    print(f"\nFringing field analysis:")
    print(f"  0.5µm pitch, d=300nm: AR = {ar_05um_300nm:.2f} "
          f"{'→ SAFE' if ar_05um_300nm < 0.3 else '→ FRINGING!'}")
    print(f"  0.5µm pitch, d=1µm:   AR = {ar_05um_1um:.2f} "
          f"{'→ SAFE' if ar_05um_1um < 0.3 else '→ FRINGING!'}")
    print(f"  0.5µm pitch, d=300nm in GTE: light bounces F times,")
    print(f"    but the LC is only 300nm thick → AR=0.60 is borderline")


# =============================================================
# 3. Phase Change Materials — the dark horse
# =============================================================
def pcm_analysis():
    print("\n" + "=" * 70)
    print("PHASE CHANGE MATERIAL (PCM) ACTUATION")
    print("=" * 70)

    # GST (Ge₂Sb₂Te₅): the most studied PCM
    # Sb₂Se₃: newer, lower loss in visible
    # GSST (Ge₂Sb₂Se₂Te₃): designed for photonics
    pcm_data = {
        "GST": {"n_amorphous": 4.0, "n_crystal": 6.5, "k_a": 1.0, "k_c": 2.5,
                 "switch_time_ns": 50, "notes": "Very lossy in visible"},
        "Sb2Se3": {"n_amorphous": 3.0, "n_crystal": 4.1, "k_a": 0.01, "k_c": 0.01,
                    "switch_time_ns": 100, "notes": "Low loss, good for visible!"},
        "GSST": {"n_amorphous": 3.2, "n_crystal": 4.8, "k_a": 0.1, "k_c": 0.3,
                  "switch_time_ns": 100, "notes": "Moderate loss"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Δn for each PCM
    names = list(pcm_data.keys())
    dn_values = [pcm_data[n]["n_crystal"] - pcm_data[n]["n_amorphous"] for n in names]
    k_values = [(pcm_data[n]["k_a"] + pcm_data[n]["k_c"]) / 2 for n in names]

    bars = axes[0].bar(names, dn_values, color=["gray", "gold", "steelblue"])
    axes[0].axhline(y=0.1, color="r", linestyle="--", label="Δn=0.1 (GTE target)")
    axes[0].set_ylabel("Δn (amorphous → crystalline)")
    axes[0].set_title("PCM Index Change\n(10-100× more than EO!)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Absorption (k) — the killer for visible holography
    x = np.arange(len(names))
    width = 0.35
    axes[1].bar(x - width / 2, [pcm_data[n]["k_a"] for n in names], width,
                label="Amorphous", color="lightblue")
    axes[1].bar(x + width / 2, [pcm_data[n]["k_c"] for n in names], width,
                label="Crystalline", color="salmon")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("Extinction coefficient k")
    axes[1].set_title("Absorption Loss\n(must be < 0.1 for phase-only)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Multi-level PCM for phase quantization
    # Sb₂Se₃ can be partially crystallized for intermediate n values
    n_levels = np.arange(2, 17)
    # Phase levels achievable per 2π range
    phase_resolution = 2 * np.pi / n_levels
    # Diffraction efficiency with quantized phase
    eta = np.sinc(1 / n_levels) ** 2  # sinc² for uniform quantization

    axes[2].plot(n_levels, eta * 100, "bo-", linewidth=2)
    axes[2].set_xlabel("Number of Phase Levels")
    axes[2].set_ylabel("Diffraction Efficiency (%)")
    axes[2].set_title("PCM Multi-Level Phase Quantization")
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=81, color="gray", linestyle="--", alpha=0.5, label="8-level (81%)")
    axes[2].legend()

    fig.suptitle("Phase Change Material Actuation", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "actuation_pcm.png", dpi=150)
    print(f"Saved: {PLOT_DIR / 'actuation_pcm.png'}")

    for name, data in pcm_data.items():
        dn = data["n_crystal"] - data["n_amorphous"]
        print(f"\n{name}: Δn = {dn:.1f}, k_avg = {(data['k_a']+data['k_c'])/2:.2f}")
        print(f"  Speed: {data['switch_time_ns']}ns, Non-volatile: YES")
        print(f"  Notes: {data['notes']}")

    print(f"\n*** Sb₂Se₃ is the standout: Δn=1.1, k≈0.01, non-volatile ***")
    print(f"*** With GTE: need F = λ/(2d·Δn) = 632/(2·300·1.1) = {632/(2*300*1.1):.1f} ***")
    print(f"*** That's F < 1! No resonance needed at all!! ***")
    print(f"*** A SIMPLE THIN FILM of Sb₂Se₃ gives > 2π at d ≥ {632/(2*1.1):.0f}nm ***")


# =============================================================
# 4. MEMS actuation at 0.5µm pitch
# =============================================================
def mems_analysis():
    print("\n" + "=" * 70)
    print("MEMS ACTUATION")
    print("=" * 70)

    # For a GTE with resonance: need to change d by Δd_eff ≈ λ/(2F)
    # Alternatively: move a membrane by Δd where Δφ = 4π·n·Δd/λ · F
    # For 2π: Δd = λ/(2nF)

    F_values = np.array([10, 20, 30, 50, 100])
    wavelengths = {"Red": 632e-9, "Green": 532e-9, "Blue": 450e-9}

    print("\nRequired membrane displacement for 2π (with resonance):")
    for wl_name, lam in wavelengths.items():
        for F in F_values:
            dd = lam / (2 * 1.5 * F)
            print(f"  {wl_name}, F={F}: Δd = {dd*1e9:.1f} nm")

    # Electrostatic MEMS: pull-in voltage for parallel plate
    # V_pull-in = sqrt(8·k·g³ / (27·ε₀·A))
    # For 0.5µm pixel: A = (0.5µm)², gap g ≈ 100nm
    pixel_size = 0.5e-6
    A = pixel_size ** 2
    epsilon_0 = 8.854e-12
    g = 100e-9  # gap
    # Spring constant: k ≈ E·t³·w / (4·L³) for a cantilever
    # For a 0.5µm × 0.5µm × 50nm Si membrane:
    E_si = 170e9  # Pa
    t = 50e-9     # membrane thickness
    w = pixel_size
    L = pixel_size
    k = E_si * t ** 3 * w / (4 * L ** 3)

    V_pullin = np.sqrt(8 * k * g ** 3 / (27 * epsilon_0 * A))
    displacement_max = g / 3  # snap-down at 1/3 of gap

    print(f"\nMEMS at 0.5µm pitch:")
    print(f"  Spring constant: {k:.4e} N/m")
    print(f"  Pull-in voltage: {V_pullin:.1f} V")
    print(f"  Max displacement (before snap-down): {displacement_max*1e9:.1f} nm")
    print(f"  Required for 2π at F=30: {632e-9/(2*1.5*30)*1e9:.1f} nm")
    print(f"  Feasible: {'YES' if displacement_max > 632e-9/(2*1.5*30) else 'MARGINAL'}")
    print(f"  Resonant frequency: {np.sqrt(k/(2.5e-17))/(2*np.pi)/1e6:.1f} MHz")
    print(f"  → Way faster than 30ms requirement")


# =============================================================
# 5. COMPREHENSIVE COMPARISON
# =============================================================
def comprehensive_comparison():
    print("\n" + "=" * 70)
    print("ACTUATION MECHANISM × ARCHITECTURE FEASIBILITY MATRIX")
    print("=" * 70)

    # Score: 0 = impossible, 1 = marginal, 2 = feasible, 3 = excellent
    mechanisms = {
        "EO (BaTiO₃)":    {"dn": 0.05, "speed_ms": 0.001, "voltage_V": 10, "continuous": True},
        "LC (nematic)":    {"dn": 0.20, "speed_ms": 5.0, "voltage_V": 3, "continuous": True},
        "LC (blue phase)": {"dn": 0.05, "speed_ms": 0.1, "voltage_V": 10, "continuous": True},
        "PCM (Sb₂Se₃)":   {"dn": 1.10, "speed_ms": 0.0001, "voltage_V": 3, "continuous": False},
        "MEMS":            {"dn": "N/A", "speed_ms": 0.01, "voltage_V": 5, "continuous": True},
        "Thermo-optic":    {"dn": 0.01, "speed_ms": 1.0, "voltage_V": 1, "continuous": True},
    }

    architectures = ["A:GTE", "B:SiPhot", "D:Ring", "G:GMR"]

    # Compatibility matrix (qualitative)
    compat = {
        ("EO (BaTiO₃)", "A:GTE"):    "Δn=0.05 needs F≥30. Thin film quality matters.",
        ("EO (BaTiO₃)", "B:SiPhot"):  "Standard in LNOI. Needs 12µm arm for 2π.",
        ("EO (BaTiO₃)", "D:Ring"):    "BaTiO₃-on-Si rings demonstrated. Compact.",
        ("EO (BaTiO₃)", "G:GMR"):     "BaTiO₃ slab as waveguide layer. Novel.",
        ("LC (nematic)", "A:GTE"):     "Δn=0.2 needs F≥6 only! But fringing at 0.5µm.",
        ("LC (nematic)", "B:SiPhot"):  "LC cladding on waveguide. Well studied.",
        ("LC (nematic)", "D:Ring"):    "LC-clad microrings. Speed limited.",
        ("LC (nematic)", "G:GMR"):     "LC on GMR grating. Most practical GMR approach!",
        ("PCM (Sb₂Se₃)", "A:GTE"):    "Δn=1.1, NO resonance needed! Simple thin film.",
        ("PCM (Sb₂Se₃)", "B:SiPhot"):  "PCM patch on waveguide. Ultra-compact.",
        ("PCM (Sb₂Se₃)", "D:Ring"):    "PCM-loaded ring. Non-volatile states.",
        ("PCM (Sb₂Se₃)", "G:GMR"):     "PCM grating fill. BEST COMBO?",
        ("MEMS", "A:GTE"):             "Move cavity mirror ~7nm. Snap-down risk.",
        ("MEMS", "G:GMR"):             "Tunable air gap above grating. Feasible.",
    }

    # Print matrix
    print(f"\n{'Mechanism':<20}", end="")
    for arch in architectures:
        print(f"{arch:>12}", end="")
    print()
    print("-" * 70)

    for mech_name, mech in mechanisms.items():
        print(f"{mech_name:<20}", end="")
        for arch in architectures:
            key = (mech_name, arch)
            if key in compat:
                verdict = compat[key][:30]
                print(f"{verdict:>12}", end="")
            else:
                print(f"{'—':>12}", end="")
        print()

    # THE KEY FINDING
    print("\n" + "=" * 70)
    print("TOP 3 MOST PROMISING COMBINATIONS")
    print("=" * 70)

    combos = [
        ("PCM (Sb₂Se₃) + Simple Thin Film",
         "Δn=1.1 means NO resonant cavity needed. A 300nm Sb₂Se₃ layer gives >2π.",
         "Simple 2-layer stack: Sb₂Se₃ on Ag mirror. Each pixel heated individually.",
         "Binary/multi-level only (not continuous). But 8 levels give 81% efficiency."),
        ("LC (nematic) + GMR Grating",
         "LC on top of a GMR grating. Changing LC alignment tunes the resonance.",
         "Single grating layer does phase modulation + beam steering.",
         "Fringing at 0.5µm pitch with ~1µm LC layer. May need wider pitch (~1µm)."),
        ("BaTiO₃ + Microring Resonator",
         "BaTiO₃-on-Si microrings with EO tuning. Proven in literature.",
         "Each ring is a compact phase shifter. Ring radius ~1-5µm fits in hogel.",
         "Thin film BaTiO₃ quality and integration with CMOS."),
    ]

    for name, description, advantage, challenge in combos:
        print(f"\n★ {name}")
        print(f"  What: {description}")
        print(f"  Why:  {advantage}")
        print(f"  Risk: {challenge}")


if __name__ == "__main__":
    print("Actuation Feasibility Analysis")
    print("=" * 50)
    eo_analysis()
    lc_analysis()
    pcm_analysis()
    mems_analysis()
    comprehensive_comparison()
    print("\nDone!")

"""
OPA + PCM Holographic Hogel — Deep Dive Simulation

Architecture: Optical Phased Array with PCM phase shifters for visible-light
holographic sub-pixel control.

CRITICAL: Silicon is opaque at visible wavelengths (bandgap 1.1eV = 1127nm).
Must use visible-transparent photonics platforms:
  - Si₃N₄ (silicon nitride): mature, n=2.0, transparent 400-2000nm
  - TiO₂: higher index n=2.3, transparent 400-8000nm
  - LiNbO₃ thin film (LNOI): EO active, n=2.2, transparent 400-5000nm

Simulation covers:
  1. Waveguide mode analysis (effective index, confinement)
  2. Splitter tree network (splitting ratio, insertion loss, propagation loss)
  3. PCM phase tuner (Sb₂Se₃ cladding on waveguide, interaction length)
  4. Grating coupler (waveguide → vertical emission)
  5. Crosstalk at minimum pitch
  6. Full power budget (laser → emitter)
  7. FoV and holographic quality metrics
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────
c = 3e8
h_planck = 6.626e-34
q = 1.6e-19


@dataclass
class WaveguidePlatform:
    name: str
    n_core: float         # core refractive index at 532nm
    n_clad: float         # cladding (SiO₂) index
    wg_width_nm: float    # single-mode width
    wg_height_nm: float   # slab thickness
    prop_loss_dB_cm: float  # propagation loss
    bend_radius_um: float   # minimum bend radius for <0.01dB loss
    splitter_loss_dB: float  # excess loss per Y-splitter (beyond 3dB split)
    min_gap_nm: float     # minimum gap for <-20dB crosstalk


# ── Platform definitions ───────────────────────────────────────
PLATFORMS = {
    'Si₃N₄': WaveguidePlatform(
        name='Si₃N₄', n_core=2.0, n_clad=1.45,
        wg_width_nm=400, wg_height_nm=200,
        prop_loss_dB_cm=0.1,    # state of art: 0.01 dB/cm, typical: 0.1
        bend_radius_um=20,
        splitter_loss_dB=0.2,   # MMI splitter excess loss
        min_gap_nm=500,
    ),
    'TiO₂': WaveguidePlatform(
        name='TiO₂', n_core=2.3, n_clad=1.45,
        wg_width_nm=350, wg_height_nm=200,
        prop_loss_dB_cm=1.0,    # higher due to scattering
        bend_radius_um=10,
        splitter_loss_dB=0.3,
        min_gap_nm=400,
    ),
    'LNOI': WaveguidePlatform(
        name='LNOI (LiNbO₃)', n_core=2.2, n_clad=1.45,
        wg_width_nm=380, wg_height_nm=300,
        prop_loss_dB_cm=0.3,
        bend_radius_um=30,
        splitter_loss_dB=0.15,  # low-loss EO-compatible
        min_gap_nm=450,
    ),
}


def effective_index_slab(n_core, n_clad, width_nm, height_nm, lam_nm=532):
    """
    Effective index of a rectangular waveguide using the corrected
    Marcatili method with penetration depth correction.

    The simple kx = π/w overestimates transverse wavevectors.
    Correction: kx ≈ π / (w + 2/γ) where γ = k0·sqrt(n_core²-n_clad²).
    """
    lam = lam_nm * 1e-9
    w = width_nm * 1e-9
    h_wg = height_nm * 1e-9

    k0 = 2 * np.pi / lam
    # Penetration depth into cladding
    gamma_clad = k0 * np.sqrt(n_core ** 2 - n_clad ** 2)
    pen_depth = 1 / gamma_clad

    # Corrected transverse wavevectors (account for mode penetration)
    kx = np.pi / (w + 2 * pen_depth)
    ky = np.pi / (h_wg + 2 * pen_depth)

    beta_sq = (k0 * n_core) ** 2 - kx ** 2 - ky ** 2

    if beta_sq <= (k0 * n_clad) ** 2:
        # Below cutoff: return cladding index
        return n_clad, 0.0

    n_eff = np.sqrt(beta_sq) / k0

    # Confinement factor: fraction of power in core
    # Γ ≈ (w/(w+2/γ)) × (h/(h+2/γ))
    conf_x = w / (w + 2 * pen_depth)
    conf_y = h_wg / (h_wg + 2 * pen_depth)
    confinement = conf_x * conf_y

    return n_eff, confinement


def waveguide_analysis():
    """Analyze waveguide modes for each platform at RGB wavelengths."""
    print("=" * 70)
    print("1. WAVEGUIDE MODE ANALYSIS (Visible Light Platforms)")
    print("=" * 70)

    wavelengths = {'Blue (450nm)': 450, 'Green (532nm)': 532, 'Red (633nm)': 633}

    results = {}
    for pname, plat in PLATFORMS.items():
        print(f"\n--- {plat.name} ---")
        print(f"  Core: n={plat.n_core}, WG: {plat.wg_width_nm}×{plat.wg_height_nm}nm")
        print(f"  Prop loss: {plat.prop_loss_dB_cm} dB/cm, Min bend: {plat.bend_radius_um}µm")

        pitch_nm = plat.wg_width_nm + plat.min_gap_nm
        print(f"  Min pitch: {pitch_nm}nm = {pitch_nm/1000:.2f}µm")

        results[pname] = {'pitch_um': pitch_nm / 1000}
        for cname, lam in wavelengths.items():
            n_eff, conf = effective_index_slab(
                plat.n_core, plat.n_clad,
                plat.wg_width_nm, plat.wg_height_nm, lam
            )
            # FoV at this pitch
            sin_th = (lam * 1e-9) / (2 * pitch_nm * 1e-9)
            fov = np.degrees(np.arcsin(min(sin_th, 1.0))) if sin_th <= 1 else 90
            print(f"  {cname}: n_eff={n_eff:.3f}, Γ={conf:.2f}, FoV=±{fov:.1f}°")

        results[pname]['fov_green'] = np.degrees(np.arcsin(
            min(532e-9 / (2 * pitch_nm * 1e-9), 1.0)))

    return results


def splitter_tree(platform, n_outputs):
    """
    Model a binary tree of 1×2 splitters to distribute light from
    one input to n_outputs.

    Tree structure:
      Level 0: 1 → 2 (1 splitter)
      Level 1: 2 → 4 (2 splitters)
      ...
      Level k: 2^k → 2^(k+1) (2^k splitters)

    Total levels: ceil(log₂(n_outputs))
    Total splitters: n_outputs - 1

    Loss per level:
      - 3 dB fundamental (energy conservation)
      - excess loss (splitter imperfection)
      - propagation loss through routing waveguides
    """
    n_levels = int(np.ceil(np.log2(n_outputs)))
    n_splitters = 2 ** n_levels - 1

    # Average waveguide length per level (routing in a hogel)
    # At level k, 2^k splitters route to 2^(k+1) outputs
    # Average routing length per level scales with hogel_size / 2^(level+1)
    hogel_size_cm = 169.3e-4  # cm

    total_splitting_dB = 0
    total_excess_dB = 0
    total_prop_dB = 0

    level_data = []
    for k in range(n_levels):
        splitting_dB = 3.0  # fundamental
        excess_dB = platform.splitter_loss_dB

        # Routing length: at level k, average distance is hogel/(2^(k+1))
        # Plus bends: ~2 bends per level, each ~π×R_bend long
        route_straight_cm = hogel_size_cm / (2 ** (k + 1))
        bend_length_cm = 2 * np.pi * platform.bend_radius_um * 1e-4  # cm
        total_route_cm = route_straight_cm + bend_length_cm

        prop_dB = platform.prop_loss_dB_cm * total_route_cm

        total_splitting_dB += splitting_dB
        total_excess_dB += excess_dB
        total_prop_dB += prop_dB

        level_data.append({
            'level': k,
            'outputs': 2 ** (k + 1),
            'split_dB': splitting_dB,
            'excess_dB': excess_dB,
            'route_cm': total_route_cm,
            'prop_dB': prop_dB,
        })

    total_loss_dB = total_splitting_dB + total_excess_dB + total_prop_dB

    return {
        'n_levels': n_levels,
        'n_splitters': n_splitters,
        'total_splitting_dB': total_splitting_dB,
        'total_excess_dB': total_excess_dB,
        'total_prop_dB': total_prop_dB,
        'total_loss_dB': total_loss_dB,
        'transmission': 10 ** (-total_loss_dB / 10),
        'level_data': level_data,
    }


def splitter_tree_analysis():
    """Analyze splitter tree for each platform."""
    print("\n\n" + "=" * 70)
    print("2. SPLITTER TREE NETWORK")
    print("=" * 70)

    hogel_um = 169.3
    results = {}

    for pname, plat in PLATFORMS.items():
        pitch_um = (plat.wg_width_nm + plat.min_gap_nm) / 1000
        n_sub_side = int(hogel_um / pitch_um)
        n_outputs = n_sub_side ** 2

        print(f"\n--- {plat.name} (pitch={pitch_um:.2f}µm, {n_sub_side}×{n_sub_side}) ---")

        tree = splitter_tree(plat, n_outputs)

        print(f"  Tree levels: {tree['n_levels']}")
        print(f"  Total splitters: {tree['n_splitters']:,}")
        print(f"  Splitting loss: {tree['total_splitting_dB']:.1f} dB (fundamental)")
        print(f"  Excess loss: {tree['total_excess_dB']:.1f} dB (splitter imperfection)")
        print(f"  Propagation loss: {tree['total_prop_dB']:.1f} dB (waveguide attenuation)")
        print(f"  TOTAL LOSS: {tree['total_loss_dB']:.1f} dB")
        print(f"  Transmission: {tree['transmission']:.2e} "
              f"({tree['transmission']*100:.4f}%)")

        # Required input laser power
        # Target: 0.05 µW optical per hogel (from power budget analysis)
        p_target_uW = 0.05
        p_laser_uW = p_target_uW / tree['transmission']
        p_laser_mW = p_laser_uW / 1000

        print(f"  Required laser input: {p_laser_mW:.2f} mW per hogel")
        print(f"  (for {p_target_uW} µW output at 300 cd/m²)")

        results[pname] = {
            'tree': tree,
            'n_sub_side': n_sub_side,
            'pitch_um': pitch_um,
            'p_laser_mW': p_laser_mW,
        }

    return results


def pcm_phase_tuner():
    """
    PCM (Sb₂Se₃) cladding on waveguide as phase shifter.

    The PCM sits on top of the waveguide. Changing its crystallization
    changes the effective index of the guided mode.

    Key parameters:
      - Δn_eff: change in effective index when PCM switches
      - L_π: length needed for π phase shift
      - L_2π: length needed for 2π (full) phase shift
      - Insertion loss: absorption in PCM (k values)
    """
    print("\n\n" + "=" * 70)
    print("3. PCM PHASE TUNER ON WAVEGUIDE")
    print("=" * 70)

    # Sb₂Se₃ properties
    n_amor = 3.0    # amorphous
    n_cryst = 4.1   # crystalline
    k_amor = 0.01   # very low loss
    k_cryst = 0.03  # slightly higher in crystalline

    # PCM geometry on waveguide
    pcm_widths = [200, 300, 400]   # nm, width of PCM strip on waveguide
    pcm_heights = [20, 50, 100]    # nm, thickness of PCM layer

    wavelengths_nm = [450, 532, 633]

    print(f"\nSb₂Se₃: n_amor={n_amor}, n_cryst={n_cryst}, "
          f"k_amor={k_amor}, k_cryst={k_cryst}")
    print(f"Δn_material = {n_cryst - n_amor}")

    # For each platform, calculate the effective index change
    # when PCM switches from amorphous to crystalline.
    # Perturbation theory: Δn_eff ≈ Γ_PCM × Δn_material
    # where Γ_PCM = fraction of mode energy in PCM region

    print("\n--- Effective index change vs PCM geometry ---")

    # Use perturbation theory
    # Γ_PCM depends on overlap of mode with PCM cladding
    # For thin PCM on top of waveguide:
    # Γ_PCM ≈ (2·h_pcm/h_wg) × exp(-2·gap/δ) where δ is evanescent decay

    results = {}
    for pname, plat in PLATFORMS.items():
        print(f"\n  {plat.name} (WG: {plat.wg_width_nm}×{plat.wg_height_nm}nm):")

        n_eff_base, conf_base = effective_index_slab(
            plat.n_core, plat.n_clad,
            plat.wg_width_nm, plat.wg_height_nm, 532
        )

        # Evanescent decay into cladding above waveguide
        lam = 532e-9
        k0 = 2 * np.pi / lam
        beta = k0 * n_eff_base
        kappa_clad = np.sqrt(beta ** 2 - (k0 * plat.n_clad) ** 2)
        decay_length_nm = 1 / kappa_clad * 1e9

        print(f"    n_eff = {n_eff_base:.3f}, evanescent decay = {decay_length_nm:.0f}nm")

        best_config = None
        print(f"\n    {'h_PCM':>6} {'w_PCM':>6} {'Γ_PCM':>8} {'Δn_eff':>8} "
              f"{'L_2π (µm)':>10} {'Loss (dB)':>10}")
        print("    " + "-" * 52)

        for h_pcm in pcm_heights:
            for w_pcm in pcm_widths:
                if w_pcm > plat.wg_width_nm + 100:
                    continue

                # Overlap integral approximation
                # Vertical: exponential decay from waveguide top
                # Assume 5nm SiO₂ spacer between WG and PCM
                spacer_nm = 5
                gamma_vert = (1 - np.exp(-2 * h_pcm / decay_length_nm)) * \
                             np.exp(-2 * spacer_nm / decay_length_nm)

                # Lateral: fraction of mode width covered by PCM
                gamma_lat = min(w_pcm / plat.wg_width_nm, 1.0) * 0.8

                gamma_pcm = gamma_vert * gamma_lat

                dn_eff = gamma_pcm * (n_cryst - n_amor)
                L_2pi = lam / dn_eff if dn_eff > 0 else float('inf')
                L_2pi_um = L_2pi * 1e6

                # Loss from PCM absorption (averaged over amorphous/crystalline)
                dk_eff = gamma_pcm * (k_cryst + k_amor) / 2
                alpha_pcm = 4 * np.pi * dk_eff / (532e-9)  # 1/m
                loss_dB = 10 * np.log10(np.e) * alpha_pcm * L_2pi  # dB over L_2π (L_2pi in meters)

                print(f"    {h_pcm:>4}nm {w_pcm:>4}nm {gamma_pcm:>8.4f} "
                      f"{dn_eff:>8.4f} {L_2pi_um:>9.1f} {loss_dB:>9.2f}")

                if best_config is None or L_2pi_um < best_config['L_2pi_um']:
                    best_config = {
                        'h_pcm': h_pcm, 'w_pcm': w_pcm,
                        'gamma_pcm': gamma_pcm, 'dn_eff': dn_eff,
                        'L_2pi_um': L_2pi_um, 'loss_dB': loss_dB,
                    }

        results[pname] = best_config
        print(f"\n    → Best: {best_config['h_pcm']}nm × {best_config['w_pcm']}nm, "
              f"L_2π = {best_config['L_2pi_um']:.1f}µm, loss = {best_config['loss_dB']:.2f}dB")

        # Switching energy
        # E = ρ·c_p·ΔT·V_pcm (to heat PCM above crystallization temp ~200°C)
        rho_cp = 2.2e6  # J/(m³·K) for Sb₂Se₃
        dT = 200  # K above ambient
        V_pcm = best_config['w_pcm'] * 1e-9 * best_config['h_pcm'] * 1e-9 * best_config['L_2pi_um'] * 1e-6
        E_switch = rho_cp * dT * V_pcm
        print(f"    Switching energy: {E_switch*1e15:.1f} fJ per phase tuner")
        print(f"    At 1 FPS: {E_switch*1e9:.4f} nW standby (zero — non-volatile!)")

    return results


def grating_coupler():
    """
    Grating coupler: converts in-plane waveguide mode to vertical emission.

    Each sub-pixel needs a grating coupler to emit light upward.
    Standard SOI grating couplers: ~30-50% efficiency.
    Si₃N₄ grating couplers: ~50-70% with bottom reflector.
    """
    print("\n\n" + "=" * 70)
    print("4. GRATING COUPLER (WAVEGUIDE → VERTICAL EMISSION)")
    print("=" * 70)

    # Grating period for vertical emission: Λ = λ / n_eff
    for pname, plat in PLATFORMS.items():
        n_eff, _ = effective_index_slab(
            plat.n_core, plat.n_clad,
            plat.wg_width_nm, plat.wg_height_nm, 532
        )
        period = 532 / n_eff  # nm
        print(f"\n  {plat.name}: grating period = {period:.0f}nm (for vertical emission at 532nm)")

    print("\n--- Grating coupler efficiency ---")
    gc_efficiencies = {
        'Si₃N₄ (simple)':          0.35,
        'Si₃N₄ (with Al mirror)':  0.65,
        'Si₃N₄ (optimized, DBR)':  0.75,
        'TiO₂ (simple)':           0.30,
        'TiO₂ (with mirror)':      0.55,
        'LNOI (simple)':           0.40,
        'LNOI (with mirror)':      0.60,
    }

    print(f"\n{'Coupler type':<28} {'Efficiency':>10}")
    print("-" * 40)
    for name, eff in gc_efficiencies.items():
        print(f"{name:<28} {eff:>9.0%}")

    print("\n  Note: With a metal mirror below the grating, most downward")
    print("  light is redirected upward → 60-75% efficiency achievable.")
    print("  Our DBR stack already provides this mirror!")

    return gc_efficiencies


def crosstalk_analysis():
    """
    Crosstalk between adjacent waveguides at minimum pitch.

    Uses empirical coupling model calibrated to Si₃N₄ literature data:
    - 800×400nm Si₃N₄ at 1550nm: L_π = 8.5µm (200nm gap), 46µm (400nm gap)
    - Ref: "Ultra-compact Si₃N₄ directional coupler" (Opt. Lett., 2017)

    Scaled to our wavelength/geometry using γ ratio.
    """
    print("\n\n" + "=" * 70)
    print("5. CROSSTALK ANALYSIS")
    print("=" * 70)

    # Reference data: Si₃N₄ 800×400nm at 1550nm
    # Fit: κ = A × exp(-γ_ref × gap), where γ_ref ≈ 8.5e6 /m, A ≈ 1e6 /m
    A_ref = 1.0e6     # /m (pre-exponential, fitted to data)
    gamma_ref = 8.5e6  # /m (fitted from gap=200nm and gap=400nm data points)

    for pname, plat in PLATFORMS.items():
        n_eff, _ = effective_index_slab(
            plat.n_core, plat.n_clad,
            plat.wg_width_nm, plat.wg_height_nm, 532
        )

        lam = 532e-9
        k0 = 2 * np.pi / lam
        beta = k0 * n_eff
        kappa_clad = np.sqrt(beta ** 2 - (k0 * plat.n_clad) ** 2)
        decay_nm = 1 / kappa_clad * 1e9

        # Scale coupling to our platform
        # Higher γ → weaker coupling (mode more confined)
        # A scales as γ² (overlap integral), exp decays faster
        gamma_ratio = kappa_clad / gamma_ref
        A_scaled = A_ref * gamma_ratio ** 2

        gap = plat.min_gap_nm * 1e-9
        kappa_coupling = A_scaled * np.exp(-kappa_clad * gap)

        L_coupling = np.pi / (2 * kappa_coupling)
        L_coupling_um = L_coupling * 1e6

        # Max parallel run for -20dB crosstalk: sin²(κL) = 0.01 → κL = 0.1
        L_max_20dB_um = (0.1 / kappa_coupling) * 1e6

        pitch_um = (plat.wg_width_nm + plat.min_gap_nm) / 1000

        print(f"\n  {plat.name} (gap={plat.min_gap_nm}nm, pitch={pitch_um:.2f}µm):")
        print(f"    Evanescent decay: 1/e at {decay_nm:.0f}nm")
        print(f"    Coupling coefficient: κ = {kappa_coupling:.0f} /m")
        print(f"    Coupling length (full transfer): {L_coupling_um:.0f}µm")
        print(f"    Max parallel run for -20dB crosstalk: {L_max_20dB_um:.0f}µm")

        # Worst case: waveguides parallel for hogel_size/2
        L_parallel_um = 169.3 / 2
        L_parallel = L_parallel_um * 1e-6
        P_coupled = np.sin(kappa_coupling * L_parallel) ** 2
        crosstalk_dB = 10 * np.log10(P_coupled + 1e-30)

        if crosstalk_dB < -20:
            print(f"    At {L_parallel_um:.0f}µm parallel run: {crosstalk_dB:.1f}dB ✅")
        else:
            print(f"    At {L_parallel_um:.0f}µm parallel run: {crosstalk_dB:.1f}dB ⚠")
            print(f"    → Solutions: wider gap, staggered routing, or shorter parallel runs")
            # What gap is needed for -20dB over 85µm?
            # sin²(κ×85µm) < 0.01 → κ < 1176 /m
            # A_scaled × exp(-γ×gap_needed) < 1176
            gap_needed = -np.log(1176 / A_scaled) / kappa_clad
            if gap_needed > 0:
                print(f"    → Need gap ≥ {gap_needed*1e9:.0f}nm for -20dB at 85µm parallel")


def full_power_budget():
    """
    End-to-end power budget from laser input to emitted holographic field.

    Chain: Laser → Fiber coupling → Chip coupling → Splitter tree →
           PCM phase tuner → Grating coupler → Free space
    """
    print("\n\n" + "=" * 70)
    print("6. FULL POWER BUDGET")
    print("=" * 70)

    # Target: 300 cd/m² display luminance
    luminance = 300  # cd/m²
    hogel_area = (169.3e-6) ** 2
    # Luminous flux per hogel
    lum_flux = luminance * hogel_area * np.pi  # lumens (Lambertian)
    # At 532nm: 545 lm/W (peak photopic)
    p_optical_hogel = lum_flux / 545  # W

    print(f"Target: {luminance} cd/m² → {p_optical_hogel*1e6:.3f} µW per hogel")

    losses = {}
    for pname, plat in PLATFORMS.items():
        pitch_um = (plat.wg_width_nm + plat.min_gap_nm) / 1000
        n_sub_side = int(169.3 / pitch_um)
        n_outputs = n_sub_side ** 2

        tree = splitter_tree(plat, n_outputs)

        # Loss chain (all in dB)
        fiber_to_chip_dB = 3.0  # fiber-to-chip grating coupler
        tree_dB = tree['total_loss_dB']

        # PCM loss (use worst case from our analysis)
        pcm_dB = 0.5  # typical for optimized PCM tuner

        # Grating coupler (with mirror)
        gc_eff = 0.65
        gc_dB = -10 * np.log10(gc_eff)

        total_dB = fiber_to_chip_dB + tree_dB + pcm_dB + gc_dB
        total_transmission = 10 ** (-total_dB / 10)

        p_laser = p_optical_hogel / total_transmission
        p_laser_mW = p_laser * 1e3

        losses[pname] = {
            'pitch_um': pitch_um,
            'n_sub': n_sub_side,
            'fiber_dB': fiber_to_chip_dB,
            'tree_dB': tree_dB,
            'pcm_dB': pcm_dB,
            'gc_dB': gc_dB,
            'total_dB': total_dB,
            'transmission': total_transmission,
            'p_laser_mW': p_laser_mW,
        }

        print(f"\n--- {plat.name} ({n_sub_side}×{n_sub_side} at {pitch_um:.2f}µm) ---")
        print(f"  Fiber → chip:     {fiber_to_chip_dB:>6.1f} dB")
        print(f"  Splitter tree:    {tree_dB:>6.1f} dB")
        print(f"  PCM phase tuner:  {pcm_dB:>6.1f} dB")
        print(f"  Grating coupler:  {gc_dB:>6.1f} dB")
        print(f"  ─────────────────────────")
        print(f"  TOTAL LOSS:       {total_dB:>6.1f} dB")
        print(f"  Transmission:     {total_transmission:.2e}")
        print(f"  Required laser:   {p_laser_mW:.2f} mW per hogel")

    # Display-level
    print("\n--- Display-level power (200×200 hogels) ---")
    n_hogels = 200 * 200
    for pname, L in losses.items():
        p_display = L['p_laser_mW'] * n_hogels / 1000  # W
        print(f"  {pname}: {p_display:.1f} W total laser power")
        if p_display > 100:
            print(f"    ⚠ This exceeds practical limits!")
        elif p_display > 10:
            print(f"    ⚠ Feasible but requires powerful laser array")
        else:
            print(f"    ✅ Practical with commercial laser diodes")

    return losses


def plot_opa_deep_dive(tree_results, pcm_results, power_results):
    """Generate comprehensive plots for OPA architecture."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ── Plot 1: Splitter tree loss breakdown ──
    ax = axes[0, 0]
    platforms = list(tree_results.keys())
    split_loss = [tree_results[p]['tree']['total_splitting_dB'] for p in platforms]
    excess_loss = [tree_results[p]['tree']['total_excess_dB'] for p in platforms]
    prop_loss = [tree_results[p]['tree']['total_prop_dB'] for p in platforms]

    x = np.arange(len(platforms))
    w = 0.6
    ax.bar(x, split_loss, w, label='Splitting (fundamental)', color='steelblue')
    ax.bar(x, excess_loss, w, bottom=split_loss, label='Excess (imperfection)', color='orange')
    ax.bar(x, prop_loss, w, bottom=[s + e for s, e in zip(split_loss, excess_loss)],
           label='Propagation', color='green')
    ax.set_ylabel('Loss (dB)')
    ax.set_title('Splitter Tree Loss Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(platforms)
    ax.legend(fontsize=8)
    for i, p in enumerate(platforms):
        total = split_loss[i] + excess_loss[i] + prop_loss[i]
        ax.text(i, total + 0.5, f'{total:.1f}dB', ha='center', fontsize=9, fontweight='bold')

    # ── Plot 2: Power budget waterfall ──
    ax = axes[0, 1]
    # Show Si₃N₄ as example
    pname = 'Si₃N₄'
    L = power_results[pname]
    categories = ['Input\nlaser', 'Fiber→chip\n(-3dB)', 'Splitter\ntree',
                  'PCM tuner', 'Grating\ncoupler', 'Output\nper pixel']
    powers_dBm = [10 * np.log10(L['p_laser_mW'])]
    powers_dBm.append(powers_dBm[-1] - L['fiber_dB'])
    powers_dBm.append(powers_dBm[-1] - L['tree_dB'])
    powers_dBm.append(powers_dBm[-1] - L['pcm_dB'])
    powers_dBm.append(powers_dBm[-1] - L['gc_dB'])
    p_per_pixel_dBm = 10 * np.log10(
        L['p_laser_mW'] * L['transmission'] / (L['n_sub'] ** 2) + 1e-30)
    powers_dBm.append(p_per_pixel_dBm)

    colors = ['green'] + ['red'] * 4 + ['blue']
    ax.bar(range(len(categories)), powers_dBm, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel('Power (dBm)')
    ax.set_title(f'Power Budget Waterfall ({pname})')
    for i, p in enumerate(powers_dBm):
        ax.text(i, p + 0.5, f'{p:.1f}', ha='center', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Plot 3: FoV vs pitch (corrected for visible platforms) ──
    ax = axes[0, 2]
    pitches = np.linspace(0.3, 5, 200)
    wavelengths = {'Blue (450nm)': 450e-9, 'Green (532nm)': 532e-9, 'Red (633nm)': 633e-9}
    c_map = {'Blue (450nm)': 'blue', 'Green (532nm)': 'green', 'Red (633nm)': 'red'}

    for cname, lam in wavelengths.items():
        fovs = []
        for p in pitches:
            sin_th = lam / (2 * p * 1e-6)
            fovs.append(np.degrees(np.arcsin(min(sin_th, 1.0))) if sin_th <= 1 else 90)
        ax.plot(pitches, fovs, color=c_map[cname], label=cname, linewidth=2)

    # Mark platforms
    for pname, plat in PLATFORMS.items():
        pitch = (plat.wg_width_nm + plat.min_gap_nm) / 1000
        fov = np.degrees(np.arcsin(min(532e-9 / (2 * pitch * 1e-6), 1.0)))
        ax.plot(pitch, fov, 'ko', markersize=8)
        ax.annotate(pname, (pitch, fov - 4), fontsize=8, ha='center', fontweight='bold')

    # Mark baseline
    ax.axvline(0.5, color='black', linestyle=':', alpha=0.5)
    ax.annotate('Baseline\n(PCM+CMOS)', (0.5, 35), fontsize=7, ha='center')
    ax.set_xlabel('Waveguide pitch (µm)')
    ax.set_ylabel('Max FoV (±degrees)')
    ax.set_title('FoV vs Platform Pitch')
    ax.legend(fontsize=8)
    ax.set_xlim(0.3, 5)
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.3)

    # ── Plot 4: PCM interaction length vs thickness ──
    ax = axes[1, 0]
    thicknesses = np.linspace(5, 150, 100)

    for pname, plat in PLATFORMS.items():
        n_eff, _ = effective_index_slab(
            plat.n_core, plat.n_clad,
            plat.wg_width_nm, plat.wg_height_nm, 532
        )
        lam = 532e-9
        k0 = 2 * np.pi / lam
        beta = k0 * n_eff
        kappa_clad = np.sqrt(beta ** 2 - (k0 * plat.n_clad) ** 2)
        decay_nm = 1 / kappa_clad * 1e9

        L_2pi = []
        for h in thicknesses:
            gamma_vert = (1 - np.exp(-2 * h / decay_nm)) * np.exp(-2 * 5 / decay_nm)
            gamma_lat = 0.8
            gamma = gamma_vert * gamma_lat
            dn = gamma * 1.1
            L = 532e-3 / dn if dn > 0 else 1e6  # µm
            L_2pi.append(L)

        ax.plot(thicknesses, L_2pi, linewidth=2, label=pname)

    ax.set_xlabel('PCM thickness (nm)')
    ax.set_ylabel('L_2π (µm)')
    ax.set_title('PCM Phase Tuner Length for 2π Shift')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(2, color='gray', linestyle='--', alpha=0.5, label='2µm target')
    ax.grid(True, alpha=0.3)

    # ── Plot 5: Required laser power vs number of emitters ──
    ax = axes[1, 1]
    n_range = np.logspace(1, 5, 100)

    for pname, plat in PLATFORMS.items():
        p_lasers = []
        for n in n_range:
            tree = splitter_tree(plat, int(n))
            total_dB = 3.0 + tree['total_loss_dB'] + 0.5 + 1.87  # fiber + tree + pcm + gc
            total_T = 10 ** (-total_dB / 10)
            p_target = 0.05e-3  # 0.05 µW → mW
            p_lasers.append(p_target / total_T)

        ax.plot(n_range, p_lasers, linewidth=2, label=pname)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of emitters')
    ax.set_ylabel('Required laser power (mW)')
    ax.set_title('Laser Power vs Array Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Practical laser limits
    ax.axhline(100, color='red', linestyle='--', alpha=0.5)
    ax.text(20, 120, '100mW (single-mode LD limit)', fontsize=7, color='red')
    ax.axhline(1000, color='red', linestyle='--', alpha=0.3)
    ax.text(20, 1200, '1W (fiber laser)', fontsize=7, color='red')

    # ── Plot 6: Architecture roadmap ──
    ax = axes[1, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('OPA + PCM Hogel Architecture', fontsize=12, fontweight='bold')

    # Draw schematic
    # Master laser
    ax.add_patch(plt.Rectangle((0.5, 4), 2, 1.5, fc='red', ec='black', alpha=0.3))
    ax.text(1.5, 4.75, 'Master\nLaser\n(532nm)', ha='center', va='center', fontsize=7)

    # Waveguide tree
    ax.annotate('', xy=(3.5, 4.75), xytext=(2.5, 4.75),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Tree fan-out
    for y_off in np.linspace(-2, 2, 5):
        ax.plot([3.5, 5.5], [4.75, 4.75 + y_off], 'g-', linewidth=1.5, alpha=0.5)
        # PCM phase shifter
        ax.add_patch(plt.Rectangle((5.5, 4.55 + y_off, ), 1, 0.4,
                                   fc='purple', ec='black', alpha=0.3))

    ax.text(6, 8, 'PCM\nΔφ', ha='center', fontsize=7, color='purple')

    # Grating couplers
    for y_off in np.linspace(-2, 2, 5):
        ax.plot(7, 4.75 + y_off, 'b^', markersize=8)

    ax.text(7, 8, 'Grating\ncouplers\n↑', ha='center', fontsize=7, color='blue')

    # Labels
    ax.text(4.5, 8.5, 'Waveguide distribution tree', ha='center', fontsize=8, color='green')
    ax.text(5, 1.5, '1 laser → N² phase-controlled emitters\n'
                     'PCM: non-volatile, zero standby power\n'
                     'All emitters coherent (single source)',
            ha='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out = PLOT_DIR / "opa_pcm_deep_dive.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"\n✅ Saved: {out}")


def print_final_verdict():
    print("\n\n" + "=" * 70)
    print("FINAL VERDICT: OPA + PCM HOLOGRAPHIC HOGEL")
    print("=" * 70)
    print("""
PLATFORM RECOMMENDATION: Si₃N₄ (silicon nitride)

  Why:
  - Lowest propagation loss (0.1 dB/cm) → least tree loss
  - Mature foundry support (LioniX, IMEC, Ligentec)
  - Compatible with PCM (Sb₂Se₃) deposition
  - Visible-light transparent (400-2000nm)

  Specs:
  - Waveguide: 400×200nm, pitch 0.90µm
  - Sub-pixels: 188×188 = 35,344 per hogel
  - FoV: ±17° (green) — good for near-eye AR/VR
  - PCM tuner: 100nm Sb₂Se₃, L_2π = 0.8µm, loss = 1.0dB
  - Total loss: 57dB (48dB fundamental splitting + 9dB system)
  - Required laser: ~23mW per hogel
  - Display power (40k hogels): ~900W ← SHOW-STOPPER

  WHY THE POWER IS SO HIGH:
  The 48dB splitting loss is FUNDAMENTAL (1/N for N=35,344 outputs).
  In the baseline architecture, a plane wave illuminates all sub-pixels
  simultaneously — no splitting loss! The OPA pays the full penalty
  because it distributes light through guided-wave splitting.

  Mitigation strategies:
  1. Semiconductor Optical Amplifiers (SOAs) on each branch → complex
  2. Fewer emitters: 32×32 = 1,024 → 30dB loss → 0.5mW/hogel → 20W display
  3. Star coupler instead of binary tree → same loss but fewer components
  4. Free-space illumination + waveguide pickup → hybrid approach

  Comparison to baseline:
  ┌────────────────┬──────────────────┬──────────────────┐
  │                │ Baseline (CMOS)  │ OPA + PCM (Si₃N₄)│
  ├────────────────┼──────────────────┼──────────────────┤
  │ Pitch          │ 0.5µm            │ 0.9µm            │
  │ FoV            │ ±30°             │ ±17°             │
  │ Sub-pixels     │ 338²=114,244     │ 188²=35,344      │
  │ Phase control  │ PCM (resistance) │ PCM (waveguide)  │
  │ Light source   │ External laser   │ On-chip laser    │
  │ Standby power  │ 0 (non-volatile) │ 0 (non-volatile) │
  │ Splitting loss │ 0 (plane wave)   │ 48dB (tree)      │
  │ Laser/hogel    │ shared (cheap)   │ 23mW (expensive) │
  │ Crosstalk      │ N/A (free space) │ Need 636nm gap   │
  │ PCM tuner loss │ N/A              │ 1.0dB (FoM-limited)│
  │ Fabrication    │ 28nm CMOS + films│ Si₃N₄ foundry    │
  │ TRL            │ 4                │ 2-3              │
  └────────────────┴──────────────────┴──────────────────┘

  KEY INSIGHT: The PCM insertion loss of ~1dB per 2π shift is a
  FUNDAMENTAL material constant: loss ≈ 4.343 × 4π × k_avg / Δn.
  For Sb₂Se₃ (Δn=1.1, k=0.02): exactly 0.99dB. This is the material
  figure of merit and cannot be improved by geometry — only by finding
  a PCM with higher Δn/k ratio.

  VERDICT: OPA + PCM is a viable path for SMALL arrays (≤32×32) or
  for near-eye displays where total power is modest. For full display
  (188² per hogel × 40k hogels), the splitting loss makes it impractical
  without on-chip amplification. The baseline (CMOS + free-space laser)
  remains superior for the full holographic display.
""")


if __name__ == "__main__":
    wg_results = waveguide_analysis()
    tree_results = splitter_tree_analysis()
    pcm_results = pcm_phase_tuner()
    gc_results = grating_coupler()
    crosstalk_analysis()
    power_results = full_power_budget()
    plot_opa_deep_dive(tree_results, pcm_results, power_results)
    print_final_verdict()

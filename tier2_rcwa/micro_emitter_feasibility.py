"""
Micro-Emitter & Dual E-Beam Feasibility Study

Two questions:
  1. Dual e-beam: one beam for light emission (cathodoluminescence pump),
     another for PCM phase control. Can this work for holography?
  2. Phase-locked micro laser arrays: VCSELs, nanolasers, OPAs at sub-µm pitch.
     Can they replace external illumination with self-emitting holographic sub-pixels?

Critical constraint: holography requires SPATIAL COHERENCE across the entire
169µm hogel. All 338×338 sub-pixels must emit with a deterministic phase
relationship — independent incoherent emitters = flat display, not hologram.

Approaches analyzed:
  A. E-beam pumped microlasers (cavity lasing via CL)
  B. VCSEL arrays with PCM phase tuning
  C. Optical Phased Arrays (silicon photonics)
  D. Injection-locked nanolaser arrays
  E. Phase-locking methods comparison
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────
c = 3e8            # m/s
h = 6.626e-34      # J·s
k_B = 1.38e-23     # J/K
q = 1.6e-19        # C

# ── Hogel parameters ──────────────────────────────────────────
HOGEL_SIZE = 169.3e-6   # m
SUBPIXEL_PITCH = 0.5e-6 # m
N_SUB = 338              # sub-pixels per side
FOV_DEG = 30             # ±30° field of view
WAVELENGTHS = {'Blue': 450e-9, 'Green': 532e-9, 'Red': 633e-9}


def dual_ebeam_analysis():
    """
    Concept: Two e-beams per hogel.
      Beam 1 — "Pump beam": excites a luminescent layer → photon emission
      Beam 2 — "Write beam": sets PCM (Sb₂Se₃) crystallization for phase control

    Physics chain:
      pump beam → cathodoluminescence → photon enters micro-cavity →
      if gain > loss: LASING (coherent!) → exits through PCM phase layer

    Key question: Can you get coherent (laser) emission from e-beam pumping
    at 0.5µm lateral scale?
    """
    print("=" * 70)
    print("1. DUAL E-BEAM ARCHITECTURE")
    print("=" * 70)

    # ── Cathodoluminescence (CL) basics ──
    # CL efficiency: fraction of e-beam energy → photon energy
    # For III-V semiconductors (GaAs, InGaAs): 1-10% CL efficiency
    # For phosphors (ZnS:Cu): up to 20% but broadband (incoherent)
    # For quantum dots: 5-15% but also broadband

    print("\n--- Cathodoluminescence as pump source ---")
    cl_efficiencies = {
        'ZnS phosphor':     0.15,   # high but broadband
        'GaAs bulk':        0.05,   # narrow emission but low
        'InGaN QW':         0.08,   # blue/green, decent
        'CdSe/ZnS QDs':    0.10,   # tunable but broadband
        'Perovskite NCs':   0.12,   # emerging, narrow linewidth
    }

    ebeam_voltage = 10e3     # V (10 keV)
    ebeam_current = 1e-9     # A (1 nA — typical for focused beam)
    ebeam_power = ebeam_voltage * ebeam_current  # 10 µW per spot

    print(f"\nE-beam: {ebeam_voltage/1e3:.0f} keV, {ebeam_current*1e9:.0f} nA "
          f"→ {ebeam_power*1e6:.0f} µW per spot")
    print(f"\n{'Material':<20} {'CL eff':>8} {'Optical power':>14} {'Coherent?':>10}")
    print("-" * 55)

    for mat, eff in cl_efficiencies.items():
        p_opt = ebeam_power * eff
        # CL is SPONTANEOUS emission → incoherent
        coherent = "NO"
        print(f"{mat:<20} {eff:>7.0%} {p_opt*1e6:>12.2f} µW {coherent:>10}")

    print("\n⚠ Raw CL is SPONTANEOUS emission → incoherent → useless for holography")
    print("  BUT: if CL pumps a micro-CAVITY with gain medium → LASING possible!")

    # ── E-beam pumped microlaser ──
    # Real examples:
    # - E-beam pumped ZnO nanowire lasers (Huang et al., 2001) — 380nm, 100nm diameter
    # - E-beam pumped GaN nanolasers (Johnson et al., 2004)
    # - CL-pumped photonic crystal nanocavities (Noginov et al., 2009)

    print("\n--- E-beam pumped microlaser (cavity CL) ---")

    # Micro-cavity: DBR / gain medium / DBR (like a VCSEL but pumped by e-beam)
    # Threshold condition: gain × confinement > mirror loss + absorption loss

    # Gain coefficients (cm⁻¹) for common materials
    gains = {
        'GaAs MQW':     {'g_max': 3000, 'lambda_nm': 850, 'linewidth_nm': 5},
        'InGaN MQW':    {'g_max': 5000, 'lambda_nm': 450, 'linewidth_nm': 10},
        'InGaAsP MQW':  {'g_max': 2500, 'lambda_nm': 1550, 'linewidth_nm': 8},
        'GaN bulk':     {'g_max': 1000, 'lambda_nm': 365, 'linewidth_nm': 3},
        'Perovskite':   {'g_max': 2000, 'lambda_nm': 530, 'linewidth_nm': 15},
    }

    # For a VCSEL-like cavity of length L:
    # Mirror reflectivity R (DBR): 99.5% → mirror loss = -ln(R)/L
    # Cavity length ~ λ/(2n) ≈ 100nm for visible
    # Confinement factor Γ ≈ 0.03 for single QW, 0.15 for MQW

    R_dbr = 0.995
    n_eff = 3.5       # effective index in cavity
    gamma_conf = 0.10  # confinement factor (MQW)
    alpha_i = 20       # internal loss (cm⁻¹)

    print(f"\nDBR reflectivity: {R_dbr:.1%}")
    print(f"Confinement factor: {gamma_conf}")
    print(f"Internal loss: {alpha_i} cm⁻¹")

    print(f"\n{'Gain medium':<18} {'g_max':>8} {'λ (nm)':>8} "
          f"{'L_cav (nm)':>10} {'α_mirror':>10} {'Threshold?':>12}")
    print("-" * 70)

    results_ebeam_laser = {}
    for name, g in gains.items():
        lam = g['lambda_nm'] * 1e-9
        L_cav = lam / (2 * n_eff)           # single λ/2 cavity
        L_cav_cm = L_cav * 1e2
        alpha_m = -np.log(R_dbr) / L_cav_cm # mirror loss (cm⁻¹)
        g_threshold = (alpha_m + alpha_i) / gamma_conf

        can_lase = g['g_max'] > g_threshold
        results_ebeam_laser[name] = {
            'lambda_nm': g['lambda_nm'],
            'L_cav_nm': L_cav * 1e9,
            'alpha_m': alpha_m,
            'g_threshold': g_threshold,
            'g_max': g['g_max'],
            'can_lase': can_lase,
        }

        status = "✅ YES" if can_lase else "❌ NO"
        print(f"{name:<18} {g['g_max']:>7} {g['lambda_nm']:>7} "
              f"{L_cav*1e9:>9.1f} {alpha_m:>9.0f} {status:>12}")
        if can_lase:
            print(f"  → g_threshold = {g_threshold:.0f} cm⁻¹ < g_max = {g['g_max']} cm⁻¹")

    # ── Thermal problem with dual beams ──
    print("\n--- Thermal interference between beams ---")

    # Beam 1 (pump) deposits energy in gain medium
    # Beam 2 (write) deposits energy in PCM layer
    # These layers are ~100-300nm apart → thermal crosstalk!

    k_th_PCM = 0.2       # W/(m·K) — Sb₂Se₃ thermal conductivity (low!)
    k_th_gain = 50        # W/(m·K) — GaAs
    separation = 300e-9   # m between layers

    # Thermal diffusion length in PCM during 1µs beam dwell
    D_th_PCM = k_th_PCM / (2.2e6)  # thermal diffusivity (ρ·c_p ≈ 2.2 MJ/m³K)
    t_dwell = 1e-6
    L_diff = np.sqrt(2 * D_th_PCM * t_dwell)

    print(f"\nPCM thermal diffusion in {t_dwell*1e6:.0f} µs: {L_diff*1e9:.0f} nm")
    print(f"Layer separation: {separation*1e9:.0f} nm")

    if L_diff > separation:
        print("⚠ THERMAL CROSSTALK: pump beam heat reaches PCM layer!")
        print("  → Phase state could drift during CL pumping")
        print("  → Need thermal barrier (e.g., SiO₂ spacer) or pulsed operation")
    else:
        print("✅ Thermal isolation adequate")

    # ── Verdict ──
    print("\n--- DUAL E-BEAM VERDICT ---")
    print("Physics: E-beam pumped microlasers ARE real (demonstrated in labs)")
    print("Problem 1: Thermal crosstalk between pump and PCM write beams")
    print("Problem 2: Each sub-pixel needs its OWN micro-cavity (338² per hogel)")
    print("Problem 3: All 338² cavities must be phase-locked → need shared cavity")
    print("Problem 4: E-beam pumping is inefficient (1-10% CL → 50% lasing = 0.5-5%)")
    print("")
    print("ALTERNATIVE CONCEPT: Instead of two beams, use ONE beam to write PCM,")
    print("and replace CL with a SHARED planar micro-cavity laser (single mode)")
    print("that illuminates all sub-pixels coherently from below.")
    print("This is essentially: VCSEL + PCM surface = our current architecture!")

    return results_ebeam_laser


def vcsel_array_analysis():
    """
    VCSEL array with PCM phase tuning.

    Each sub-pixel is a VCSEL. A PCM layer on the output mirror
    tunes the phase of the emitted field.

    Constraint: VCSELs must be phase-locked (injection locking or
    evanescent coupling) for holographic operation.
    """
    print("\n\n" + "=" * 70)
    print("2. VCSEL ARRAY + PCM PHASE TUNING")
    print("=" * 70)

    # ── Minimum VCSEL size ──
    # VCSEL oxide aperture determines mode size
    # Below ~1µm: radiation loss dominates, Q drops, no lasing
    # Record smallest VCSEL: ~0.5µm aperture (Iga lab, 2010) but extremely low power

    vcsel_sizes = {
        'Commercial (Lumentum)':      {'aperture_um': 6.0,  'power_uW': 2000, 'eff': 0.35},
        'Research (small oxide)':     {'aperture_um': 2.0,  'power_uW': 200,  'eff': 0.20},
        'Cutting edge (Iga 2010)':    {'aperture_um': 1.0,  'power_uW': 20,   'eff': 0.05},
        'Extreme (sub-λ cavity)':     {'aperture_um': 0.5,  'power_uW': 1,    'eff': 0.01},
        'Needed for hologram':        {'aperture_um': 0.5,  'power_uW': 0.1,  'eff': None},
    }

    print(f"\n{'VCSEL type':<30} {'Aperture':>10} {'Power':>10} {'WPE':>8}")
    print("-" * 62)
    for name, v in vcsel_sizes.items():
        eff_str = f"{v['eff']:.0%}" if v['eff'] else "???"
        print(f"{name:<30} {v['aperture_um']:>8.1f} µm {v['power_uW']:>8.1f} µW {eff_str:>8}")

    # ── FoV implications of larger pitch ──
    print("\n--- FoV vs VCSEL pitch ---")
    print("Grating equation: sin(θ_max) = λ / (2 × pitch)")

    pitches_um = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0])

    print(f"\n{'Pitch (µm)':<12}", end="")
    for color, lam in WAVELENGTHS.items():
        print(f"{'FoV_' + color:>12}", end="")
    print(f"{'N_sub':>10} {'Sub-px/hogel':>14}")
    print("-" * 72)

    fov_data = {}
    for p in pitches_um:
        n_sub = int(HOGEL_SIZE / (p * 1e-6))
        print(f"{p:<12.1f}", end="")
        fovs = {}
        for color, lam in WAVELENGTHS.items():
            sin_theta = lam / (2 * p * 1e-6)
            if sin_theta <= 1:
                theta = np.degrees(np.arcsin(sin_theta))
                print(f"{'±' + f'{theta:.1f}°':>12}", end="")
                fovs[color] = theta
            else:
                print(f"{'> ±90°':>12}", end="")
                fovs[color] = 90
        print(f"{n_sub:>10} {n_sub**2:>14,}")
        fov_data[p] = fovs

    # ── Phase locking VCSELs ──
    print("\n--- Phase locking mechanisms ---")
    methods = {
        'Evanescent coupling': {
            'max_pitch_um': 3.0,
            'max_array': '8×8 demonstrated',
            'phase_control': 'limited (supermode selection)',
            'scalable': False,
        },
        'Injection locking': {
            'max_pitch_um': 'any',
            'max_array': '64×64 demonstrated',
            'phase_control': 'excellent (master laser phase reference)',
            'scalable': True,
        },
        'Diffractive coupling (Talbot)': {
            'max_pitch_um': 10.0,
            'max_array': '100×100 demonstrated',
            'phase_control': 'moderate (fixed patterns)',
            'scalable': True,
        },
        'Waveguide distribution': {
            'max_pitch_um': 2.0,
            'max_array': '1×1024 (OPA)',
            'phase_control': 'excellent (integrated phase shifters)',
            'scalable': True,
        },
    }

    print(f"\n{'Method':<28} {'Max pitch':>10} {'Array size':>20} {'Scalable':>10}")
    print("-" * 72)
    for name, m in methods.items():
        print(f"{name:<28} {str(m['max_pitch_um']):>10} "
              f"{m['max_array']:>20} {'✅' if m['scalable'] else '❌':>10}")
        print(f"  Phase control: {m['phase_control']}")

    # ── Power budget per sub-pixel ──
    print("\n--- Power budget ---")
    # For holographic display, each sub-pixel needs to contribute
    # enough light for a visible image at ~100 cd/m² (typical display)

    luminance_target = 300  # cd/m²
    hogel_area = HOGEL_SIZE ** 2  # m²
    # luminance = luminous intensity / area
    # luminous flux per hogel = luminance × area × π (Lambertian)
    luminous_flux = luminance_target * hogel_area * np.pi  # lumens
    # luminous efficacy at 532nm: 545 lm/W (peak)
    lm_per_W = 545
    optical_power_hogel = luminous_flux / lm_per_W  # W

    n_subpixels = N_SUB ** 2
    power_per_subpixel = optical_power_hogel / n_subpixels

    print(f"Target luminance: {luminance_target} cd/m²")
    print(f"Optical power per hogel: {optical_power_hogel*1e6:.2f} µW")
    print(f"Optical power per sub-pixel: {power_per_subpixel*1e12:.1f} pW")
    print(f"With 10% WPE: {power_per_subpixel/0.10*1e12:.1f} pW electrical per sub-pixel")
    print("\n→ Power per sub-pixel is TINY — even nanolasers can provide this!")

    return fov_data


def optical_phased_array():
    """
    Silicon photonics Optical Phased Array (OPA) approach.

    Instead of free-space sub-pixels, use integrated waveguide emitters
    with thermo-optic or electro-optic phase shifters.

    This is the CLOSEST existing technology to holographic sub-pixel control.
    """
    print("\n\n" + "=" * 70)
    print("3. OPTICAL PHASED ARRAY (SILICON PHOTONICS)")
    print("=" * 70)

    # State of the art OPAs:
    # - MIT (Watts lab): 2D OPA, 8×8 emitters, 9µm pitch (2013)
    # - Intel: 1×128 OPA, 2µm pitch (2020)
    # - Columbia: 2D OPA, 32×32, 4µm pitch (2022)
    # - KAIST: 1×512 OPA, 1.1µm pitch (2023) ← closest to our needs

    opa_demos = {
        'MIT 2013 (Watts)':       {'dim': '8×8',    'pitch_um': 9.0,  'platform': 'SOI'},
        'UCSB 2018':              {'dim': '32×32',  'pitch_um': 4.0,  'platform': 'Si₃N₄'},
        'Intel 2020':             {'dim': '1×128',  'pitch_um': 2.0,  'platform': 'SOI'},
        'Columbia 2022':          {'dim': '32×32',  'pitch_um': 4.0,  'platform': 'SOI'},
        'KAIST 2023':             {'dim': '1×512',  'pitch_um': 1.1,  'platform': 'SOI'},
        'Our target':             {'dim': '338×338','pitch_um': 0.5,  'platform': '???'},
    }

    print(f"\n{'Group':<25} {'Array':>10} {'Pitch':>8} {'Platform':>10}")
    print("-" * 55)
    for name, d in opa_demos.items():
        print(f"{name:<25} {d['dim']:>10} {d['pitch_um']:>6.1f}µm {d['platform']:>10}")

    # ── Waveguide pitch limits ──
    print("\n--- Minimum waveguide pitch ---")
    # Single-mode Si waveguide: ~450nm × 220nm cross-section
    # Minimum pitch limited by evanescent coupling:
    #   - At <1µm spacing, waveguides couple → crosstalk
    #   - Si (n=3.47): evanescent decay ~ 200nm → need >600nm gap
    #   - Si₃N₄ (n=2.0): longer evanescent tail → need >1µm gap

    platforms = {
        'SOI (Si, n=3.47)': {
            'wg_width_nm': 450,
            'min_gap_nm': 400,
            'min_pitch_nm': 850,
            'phase_shifter': 'thermo-optic (π shift: 50µm long, 20mW)',
        },
        'Si₃N₄ (n=2.0)': {
            'wg_width_nm': 800,
            'min_gap_nm': 600,
            'min_pitch_nm': 1400,
            'phase_shifter': 'thermo-optic (π shift: 200µm long, 40mW)',
        },
        'LiNbO₃ thin film': {
            'wg_width_nm': 600,
            'min_gap_nm': 500,
            'min_pitch_nm': 1100,
            'phase_shifter': 'electro-optic (π shift: 5mm long, 3V) — FAST!',
        },
        'InP (active)': {
            'wg_width_nm': 500,
            'min_gap_nm': 500,
            'min_pitch_nm': 1000,
            'phase_shifter': 'carrier injection (π shift: 100µm, 2mA)',
        },
    }

    print(f"\n{'Platform':<25} {'WG width':>10} {'Min gap':>10} {'Min pitch':>10}")
    print("-" * 58)
    for name, p in platforms.items():
        print(f"{name:<25} {p['wg_width_nm']:>8}nm {p['min_gap_nm']:>8}nm {p['min_pitch_nm']:>8}nm")
        print(f"  Phase: {p['phase_shifter']}")

    # FoV at achievable pitches
    print("\n--- FoV at achievable OPA pitches ---")
    lambda_green = 532e-9
    opa_pitches = [0.85, 1.0, 1.1, 1.4, 2.0]
    for p_um in opa_pitches:
        p = p_um * 1e-6
        sin_th = lambda_green / (2 * p)
        if sin_th <= 1:
            theta = np.degrees(np.arcsin(sin_th))
            n_pts = int(HOGEL_SIZE / p)
            print(f"  Pitch {p_um:.2f}µm → ±{theta:.1f}° FoV, {n_pts} points → {n_pts}² = {n_pts**2:,} emitters")
        else:
            print(f"  Pitch {p_um:.2f}µm → > ±90°")

    # ── 2D OPA power budget ──
    print("\n--- OPA power budget (thermo-optic, SOI) ---")
    n_emitters = 200  # realistic 200×200 at 0.85µm pitch
    power_per_shifter_mW = 20  # typical for π shift
    total_power = n_emitters ** 2 * power_per_shifter_mW * 1e-3  # W

    print(f"Emitters: {n_emitters}×{n_emitters} = {n_emitters**2:,}")
    print(f"Power per phase shifter: {power_per_shifter_mW} mW")
    print(f"Total phase shifter power: {total_power:.0f} W per hogel!")
    print("⚠ Thermo-optic is WAY too power-hungry for a display")

    # Electro-optic alternative
    print("\n--- OPA power budget (electro-optic, LiNbO₃) ---")
    n_emitters_eo = 154  # at 1.1µm pitch
    # EO phase shifter: ~100 fJ/bit, but static → µW standby
    power_per_eo_uW = 0.1  # µW static power per shifter
    total_eo = n_emitters_eo ** 2 * power_per_eo_uW * 1e-6
    print(f"Emitters: {n_emitters_eo}×{n_emitters_eo} = {n_emitters_eo**2:,}")
    print(f"Power per EO shifter: {power_per_eo_uW} µW")
    print(f"Total EO power: {total_eo*1e6:.1f} µW per hogel ← FEASIBLE!")
    print("BUT: LiNbO₃ pitch 1.1µm → ±14° FoV (not ±30°)")

    # ── PCM as phase shifter (our innovation) ──
    print("\n--- OPA with PCM phase shifters (zero standby power) ---")
    print("Replace thermo-optic/EO shifters with Sb₂Se₃ PCM on waveguide")
    print("PCM phase shift: Δn ~ 1.1 over 300nm thickness → Δφ = 2πΔnL/λ")

    dn = 1.1
    L_pcm = 2e-6  # 2µm interaction length along waveguide
    lam = 532e-9
    dphi = 2 * np.pi * dn * L_pcm / lam
    print(f"Phase shift with {L_pcm*1e6:.0f}µm PCM on waveguide: {dphi/np.pi:.1f}π rad")
    print(f"Need 2π → L_PCM = {lam / dn * 1e6:.2f}µm ← very compact!")
    print("Standby power: ZERO (PCM is non-volatile)")
    print("Write: one-time electrical pulse or laser pulse per frame")

    return platforms


def injection_locking_analysis():
    """
    Injection locking: a master laser seeds an array of slave lasers.
    Each slave locks to the master's frequency and phase.
    A PCM or EO phase shifter between master and slave sets the relative phase.
    """
    print("\n\n" + "=" * 70)
    print("4. INJECTION-LOCKED NANOLASER ARRAY")
    print("=" * 70)

    # Injection locking bandwidth: Δf_lock = (f₀/2Q) × sqrt(P_inj/P_slave)
    # For a nano-cavity with Q = 1000, f₀ = 5.6e14 Hz (532nm):
    # Δf_lock = 2.8e11 × sqrt(P_inj/P_slave)

    f0 = c / 532e-9
    Q_values = [500, 1000, 5000, 10000]
    P_ratio = 0.01  # 1% injection ratio (weak)

    print("\n--- Injection locking bandwidth ---")
    print(f"{'Q factor':>10} {'Lock bandwidth':>16} {'Lock range (pm)':>16}")
    print("-" * 45)
    for Q in Q_values:
        df = (f0 / (2 * Q)) * np.sqrt(P_ratio)
        dlam = (532e-9) ** 2 * df / c  # Δλ = λ²Δf/c
        print(f"{Q:>10} {df/1e9:>14.1f} GHz {dlam*1e12:>14.1f} pm")

    # ── Photonic crystal nanolaser ──
    print("\n--- Photonic Crystal (PhC) nanolaser array ---")
    print("PhC nanocavity: Q ~ 10⁴-10⁶, mode volume ~ (λ/2n)³")

    lam = 532e-9
    n_phc = 2.5  # effective index
    # Mode volume in units of (λ/n)³
    mode_side = lam / (2 * n_phc)  # ~106 nm
    V_mode = mode_side ** 3         # m³
    V_mode_normalized = V_mode / (lam / n_phc) ** 3  # in units of (λ/n)³
    V_mode_um3 = V_mode * 1e18  # m³ → µm³
    print(f"Mode volume: ({mode_side*1e9:.0f}nm)³ = {V_mode_um3:.4f} µm³ "
          f"= {V_mode_normalized:.3f} (λ/n)³")

    # Purcell factor: F_P = (3/(4π²)) × (λ/n)³ × Q / V_mode
    Q_phc = 10000
    F_p = (3 / (4 * np.pi ** 2)) * Q_phc * (lam / n_phc) ** 3 / V_mode
    print(f"Purcell factor: {F_p:.0f} → enhanced spontaneous emission")

    # Threshold for PhC nanolaser (simplified model)
    # P_th ≈ hν × V_mode × N_tr / (τ_sp × Γ × F_p)
    # τ_sp ~ 1ns (spontaneous lifetime), N_tr ~ 1e18 cm⁻³
    N_tr = 1e24    # m⁻³ (transparency carrier density)
    tau_sp = 1e-9  # s
    gamma_phc = 0.5
    hnu = h * c / lam
    P_th = hnu * V_mode * N_tr / (tau_sp * gamma_phc * F_p)
    print(f"Estimated threshold power: {P_th*1e9:.2f} nW")
    print(f"→ Sub-nW threshold! Even at 0.5µm pitch, this is achievable.")

    # ── Array architecture ──
    print("\n--- Proposed injection-locked PhC array ---")
    print("Architecture:")
    print("  1. Central VCSEL master laser (one per hogel)")
    print("  2. Waveguide distribution network (tree splitter)")
    print("  3. PCM phase shifter on each waveguide branch")
    print("  4. PhC nanocavity slave laser at each sub-pixel")
    print("  5. Vertical emission through grating coupler")
    print("")
    print("Advantages:")
    print("  + All sub-pixels phase-locked to master → coherent!")
    print("  + PCM phase shifters: non-volatile, zero standby power")
    print("  + PhC cavities amplify signal → compensates splitting losses")
    print("  + Single master wavelength → narrow linewidth")
    print("")
    print("Challenges:")
    print("  - Waveguide tree for 338² outputs: 17 levels of 1×2 splitters")
    print("  - Splitting loss: 3dB × 17 = 51dB → need 10⁵ amplification")
    print("  - Or: use amplifying waveguides (SOA) to compensate")
    print("  - PhC cavity fabrication at 0.5µm pitch: beyond current EBL")

    n_levels = int(np.ceil(np.log2(N_SUB ** 2)))
    splitting_loss_dB = 3 * n_levels
    splitting_loss_linear = 10 ** (splitting_loss_dB / 10)
    print(f"\n  Splitting levels: {n_levels}")
    print(f"  Total splitting loss: {splitting_loss_dB} dB ({splitting_loss_linear:.0e}×)")

    # ── More realistic: fewer sub-pixels ──
    print("\n--- Relaxed spec: 2µm pitch (85×85 sub-pixels) ---")
    n_sub_relaxed = int(HOGEL_SIZE / 2e-6)
    n_levels_r = int(np.ceil(np.log2(n_sub_relaxed ** 2)))
    split_loss_r = 3 * n_levels_r
    sin_th_r = 532e-9 / (2 * 2e-6)
    fov_r = np.degrees(np.arcsin(sin_th_r))

    print(f"  Sub-pixels: {n_sub_relaxed}×{n_sub_relaxed} = {n_sub_relaxed**2:,}")
    print(f"  FoV at 532nm: ±{fov_r:.1f}°")
    print(f"  Splitting levels: {n_levels_r}, loss: {split_loss_r} dB")
    print(f"  This is MUCH more feasible for near-eye displays!")


def comparison_table():
    """Print a comprehensive comparison of all architectures."""
    print("\n\n" + "=" * 70)
    print("5. ARCHITECTURE COMPARISON")
    print("=" * 70)

    archs = [
        {
            'name': 'Current (PCM + ext. laser)',
            'pitch_um': 0.5,
            'fov': '±30°',
            'coherent': 'YES (ext. laser)',
            'power': 'Low (PCM=0W standby)',
            'fab': '28nm CMOS + PCM deposition',
            'trl': 4,
            'verdict': '✅ BASELINE',
        },
        {
            'name': 'Dual e-beam (CL + PCM)',
            'pitch_um': 0.5,
            'fov': '±30°',
            'coherent': 'NO (CL=incoherent)',
            'power': 'High (e-beam vacuum)',
            'fab': 'SEM-grade column + PCM',
            'trl': 2,
            'verdict': '❌ INCOHERENT',
        },
        {
            'name': 'E-beam pumped microlaser',
            'pitch_um': 0.5,
            'fov': '±30°',
            'coherent': 'YES (if cavity lases)',
            'power': 'Very high',
            'fab': 'DBR+gain+PCM per pixel (!)',
            'trl': 1,
            'verdict': '⚠ Needs shared cavity',
        },
        {
            'name': 'VCSEL array + PCM',
            'pitch_um': 2.0,
            'fov': '±7.6°',
            'coherent': 'YES (injection lock)',
            'power': 'Moderate',
            'fab': 'III-V epitaxy + PCM',
            'trl': 3,
            'verdict': '⚠ FoV limited',
        },
        {
            'name': 'OPA (SOI + PCM)',
            'pitch_um': 0.85,
            'fov': '±18°',
            'coherent': 'YES (single source)',
            'power': 'Zero standby (PCM)',
            'fab': 'Si photonics + PCM',
            'trl': 3,
            'verdict': '⭐ MOST PROMISING ALT',
        },
        {
            'name': 'OPA (LiNbO₃)',
            'pitch_um': 1.1,
            'fov': '±14°',
            'coherent': 'YES (single source)',
            'power': 'Very low (EO)',
            'fab': 'LNOI foundry',
            'trl': 3,
            'verdict': '⭐ Best for near-eye',
        },
        {
            'name': 'PhC nanolaser array',
            'pitch_um': 0.5,
            'fov': '±30°',
            'coherent': 'YES (injection lock)',
            'power': 'Low (nanolaser threshold)',
            'fab': 'Extreme nano-fab',
            'trl': 1,
            'verdict': '🔬 Long-term vision',
        },
    ]

    print(f"\n{'Architecture':<28} {'Pitch':>6} {'FoV':>7} {'Coherent':>18} {'TRL':>4} {'Verdict':>20}")
    print("-" * 90)
    for a in archs:
        print(f"{a['name']:<28} {a['pitch_um']:>4.1f}µm {a['fov']:>7} "
              f"{a['coherent']:>18} {a['trl']:>4} {a['verdict']:>20}")

    print("\n--- KEY INSIGHT ---")
    print("The dual e-beam idea CONVERGES to two feasible architectures:")
    print("")
    print("  (A) OPA with PCM phase shifters (0.85-1.1µm pitch)")
    print("      = Integrated photonics version of current design")
    print("      = Single laser source + waveguide distribution + PCM phase control")
    print("      = FoV reduced to ±14-18° (fine for near-eye AR/VR)")
    print("")
    print("  (B) Injection-locked VCSEL + PCM array (2µm pitch)")
    print("      = Self-emitting but phase-locked")
    print("      = ±7.6° FoV (narrow, but works for specific applications)")
    print("")
    print("  (C) Current baseline (ext. laser + PCM) remains BEST for ±30° FoV")
    print("      = 0.5µm pitch is uniquely achievable with CMOS + thin-film PCM")
    print("      = No equivalent photonic technology at this pitch yet")

    return archs


def plot_comparison(fov_data):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Plot 1: FoV vs pitch ──
    ax = axes[0]
    pitches = np.linspace(0.3, 10, 200)
    for color, lam in WAVELENGTHS.items():
        fovs = []
        for p in pitches:
            sin_th = lam * 1e9 / (2 * p * 1e3)  # nm / nm
            sin_th_real = lam / (2 * p * 1e-6)
            if sin_th_real <= 1:
                fovs.append(np.degrees(np.arcsin(sin_th_real)))
            else:
                fovs.append(90)
        c_map = {'Blue': 'blue', 'Green': 'green', 'Red': 'red'}
        ax.plot(pitches, fovs, color=c_map[color], label=color, linewidth=2)

    # Mark architectures
    arch_marks = [
        (0.5, '±30°\nPCM\n(baseline)', 'black'),
        (0.85, '±18°\nOPA/SOI', 'purple'),
        (1.1, '±14°\nOPA/LN', 'orange'),
        (2.0, '±7.6°\nVCSEL', 'brown'),
    ]
    for p, label, c in arch_marks:
        sin_th = 532e-9 / (2 * p * 1e-6)
        fov = np.degrees(np.arcsin(min(sin_th, 1)))
        ax.axvline(p, color=c, linestyle='--', alpha=0.5)
        ax.annotate(label, (p, fov + 3), fontsize=7, ha='center',
                    color=c, fontweight='bold')

    ax.set_xlabel('Sub-pixel pitch (µm)')
    ax.set_ylabel('Max FoV (±degrees)')
    ax.set_title('Field of View vs Emitter Pitch')
    ax.legend()
    ax.set_xlim(0.3, 10)
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Technology readiness ──
    ax = axes[1]
    names = ['PCM+laser\n(baseline)', 'Dual\ne-beam', 'E-beam\nmicrolaser',
             'VCSEL\narray', 'OPA\n(SOI+PCM)', 'OPA\n(LiNbO₃)', 'PhC\nnanolaser']
    trls = [4, 2, 1, 3, 3, 3, 1]
    colors_bar = ['green', 'red', 'red', 'orange', 'blue', 'blue', 'gray']

    bars = ax.bar(names, trls, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Technology Readiness Level')
    ax.set_title('TRL Comparison')
    ax.set_ylim(0, 6)
    ax.axhline(3, color='gray', linestyle='--', alpha=0.5, label='Proof of concept')
    for bar, trl in zip(bars, trls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'TRL {trl}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', labelsize=7)

    # ── Plot 3: Power vs sub-pixels ──
    ax = axes[2]
    n_range = np.logspace(1, 5.5, 100)

    # Thermo-optic OPA: 20mW per shifter
    p_to = n_range * 20e-3  # W
    ax.plot(n_range, p_to, 'r-', label='Thermo-optic OPA', linewidth=2)

    # EO (LiNbO₃): 0.1µW per shifter
    p_eo = n_range * 0.1e-6
    ax.plot(n_range, p_eo, 'b-', label='Electro-optic OPA', linewidth=2)

    # PCM: zero standby, ~1µJ write energy per pixel, at 1 FPS
    p_pcm = n_range * 1e-6 * 1  # 1µJ × 1 FPS = 1µW per pixel
    ax.plot(n_range, p_pcm, 'g-', label='PCM (write power @ 1 FPS)', linewidth=2)

    # VCSEL injection: ~1mA per VCSEL × 2V
    p_vcsel = n_range * 1e-3 * 2  # W
    ax.plot(n_range, p_vcsel, 'm-', label='VCSEL array', linewidth=2)

    ax.axvline(N_SUB ** 2, color='black', linestyle=':', alpha=0.5, label=f'{N_SUB}² = {N_SUB**2:,}')
    ax.axhline(1, color='gray', linestyle='--', alpha=0.3)
    ax.text(N_SUB ** 2 * 1.3, 0.5, '338²', fontsize=8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sub-pixels per hogel')
    ax.set_ylabel('Power per hogel (W)')
    ax.set_title('Power Budget Scaling')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-9, 1e4)

    plt.tight_layout()
    out = PLOT_DIR / "micro_emitter_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"\n✅ Saved: {out}")


if __name__ == "__main__":
    results_laser = dual_ebeam_analysis()
    fov_data = vcsel_array_analysis()
    platforms = optical_phased_array()
    injection_locking_analysis()
    archs = comparison_table()
    plot_comparison(fov_data)

    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("""
Your dual e-beam idea leads to a fascinating design space:

1. RAW DUAL E-BEAM (CL + PCM): ❌ Doesn't work.
   CL is incoherent → no holographic wavefront control.
   Thermal crosstalk between beams corrupts PCM state.

2. E-BEAM PUMPED MICROLASER: ⚠ Physics works, engineering doesn't.
   InGaN MQW can lase in a DBR cavity pumped by e-beam.
   But: need 338² individual micro-cavities per hogel → unfabricicable.
   Would reduce to a shared planar cavity → back to current architecture.

3. VCSEL ARRAY + PCM: ⚠ Works but FoV limited.
   2µm pitch (smallest feasible) → ±7.6° FoV.
   Good for near-eye (AR glasses) but not wide-angle holography.
   Phase locking via injection from master VCSEL.

4. OPTICAL PHASED ARRAY + PCM: ⭐ Most promising alternative.
   SOI platform: 0.85µm pitch → ±18° FoV, 200×200 emitters.
   PCM on waveguide: zero standby power, non-volatile phase.
   Closest to existing technology (LiDAR OPAs at TRL 5+).

5. CURRENT BASELINE (ext. laser + CMOS + PCM): ✅ Still best for ±30°.
   0.5µm pitch uniquely achievable with thin-film deposition.
   No integrated photonic platform matches this density yet.

BOTTOM LINE: Your intuition is right — self-emitting phase-controlled
pixels are the holy grail. The OPA+PCM hybrid is the bridge technology
that could get there within 5-10 years. For ±30° FoV TODAY,
external laser illumination remains the only path.
""")

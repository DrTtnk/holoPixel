"""
Quantization-Aware Phase Optimization

The headline claim in README.md and DRAFT.md section 6.4 — "3-bit phase
(8 levels) is sufficient: 95% diffraction efficiency" — comes from the
idealized closed form for an N-level blazed grating,

    eta = ( sin(pi/N) / (pi/N) )^2

which is proven symbolically in tests/test_phase_quantization.py. A blazed
grating is not a hologram: none of slfh.py, hogel_grid.py or hogel_optimizer.py
in this package ever quantizes phase. They optimize continuous phase and the
95% number has never been checked against the code that actually produces our
holograms.

Choi et al., "Time-Multiplexed Neural Holography" (arXiv:2205.02367,
docs/papers/2205.02367.pdf, Sec. 3.2 "Optimizing Phase Patterns for Quantized
SLMs") show that for a real quantized SLM, image quality is recovered by a
QUANTIZATION-AWARE optimizer — not by adding phase levels. They compare:

  * Naive: optimize continuous phase, quantize once at the end (ignores q).
  * Surrogate-gradient (SG): forward pass uses the true quantizer q, backward
    pass substitutes a differentiable proxy — the straight-through estimator
    (STE) is the simplest member of this family.
  * Gumbel-Softmax (GS, their Eq. 6-8): a temperature-annealed relaxed
    categorical over the quantized levels, hard in the forward pass.

This module implements and measures all three, at 1-4 bits and continuous
phase as the ceiling, against BOTH an idealized uniform level set (amplitude
1, phase spaced by 2*pi/N — what the sinc closed form and the literature both
assume) and a physically realistic non-uniform level set built from
tier2_rcwa.rcwa_optimization.design_phase_lut for the 300 nm Sb2Se3 / 6-pair
TiO2-SiO2 DBR stack. design_phase_lut TMM-corrects the refractive indices so
that the resulting PHASE steps are (very nearly) uniform — that correction is
the whole point of the LUT (see tests/test_tmm_phase_lut.py). What it does
NOT correct is amplitude: each phase level sits at a different point on the
material's dispersion curve and therefore reflects a different fraction of
the incident light. That per-level reflectance (amplitude) spread is the real
"non-uniform level set" of this device, and it has no counterpart in the
phase-only SLMs the literature quantizes.
"""

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from tier1_tmm.partial_phase import efficiency_metric
from tier2_rcwa.rcwa_optimization import N_SIO2, N_TIO2, design_phase_lut, stack_response

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 64                       # hologram side length (single hogel — kept small, GPU is shared)
BIT_DEPTHS = (1, 2, 3, 4)    # -> 2, 4, 8, 16 levels
WAVELENGTH_NM = 532
FILM_NM = 300
N_DBR_PAIRS = 6

CONT_ITERS = 600
ITERS = 400
LR = 0.08
GUMBEL_W = 2.0                 # score sharpness in Choi et al. Eq. 8
TAU_START, TAU_END = 2.0, 0.05

# score_l = sigmoid(w*delta)*(1-sigmoid(w*delta)) is bounded to [0, 0.25] for
# ANY w (that is the max of x*(1-x)); w only narrows the peak, never raises
# it. Eq. 6 adds this score directly to standard Gumbel(0,1) noise (std =
# pi/sqrt(6)), and dividing both terms by the same tau never changes their
# relative scale -- so the raw score can never out-compete the noise, at any
# temperature, and the optimizer measurably degenerates to chance level (see
# gumbel_quantize docstring). GUMBEL_SCORE_SCALE is the explicit gain the
# paper's main text omits (it defers to "the supplement"). Its order of
# magnitude is derived, not fitted: pick it so that at the START of the
# anneal (tau=TAU_START, the most exploration-heavy point) the peak score is
# GUMBEL_CONFIDENCE noise-standard-deviations tall -- enough for a mild,
# learnable bias without collapsing exploration immediately. At the end of
# the anneal (tau=TAU_END) the same fixed scale is TAU_START/TAU_END times
# more decisive, which is what makes the final pick deterministic.
_GUMBEL_NOISE_STD = math.pi / math.sqrt(6)   # std of a standard Gumbel(0,1)
_GUMBEL_MAX_SCORE = 0.25                      # max of sigmoid(x)(1-sigmoid(x))
GUMBEL_CONFIDENCE = 2.0                       # noise-std multiples at anneal start
GUMBEL_SCORE_SCALE = GUMBEL_CONFIDENCE * _GUMBEL_NOISE_STD * TAU_START / _GUMBEL_MAX_SCORE


# ======================================================================
# Level sets
# ======================================================================

@dataclass
class LevelSet:
    name: str
    phase: torch.Tensor        # (L,) float32 radians, in [0, 2*pi)
    amplitude: torch.Tensor    # (L,) float32, in (0, 1]

    @property
    def field(self):
        return self.amplitude.to(torch.complex64) * torch.exp(
            1j * self.phase.to(torch.complex64)
        )

    def __len__(self):
        return self.phase.shape[0]


def uniform_levels(n_levels):
    """The idealized level set: unit amplitude, spaced by 2*pi/N. This is what
    the sinc closed form and the Choi et al. phase-only SLM both assume."""
    phase = torch.arange(n_levels, device=DEVICE, dtype=torch.float32) * (2 * math.pi / n_levels)
    amplitude = torch.ones(n_levels, device=DEVICE)
    return LevelSet("uniform", phase, amplitude)


def _dbr_stack(film_nm=FILM_NM, n_pairs=N_DBR_PAIRS, wavelength_nm=WAVELENGTH_NM):
    d_tio2 = wavelength_nm / (4 * N_TIO2)
    d_sio2 = wavelength_nm / (4 * N_SIO2)
    n_list, d_list = [1.0, None], [np.inf, film_nm]
    for _ in range(n_pairs):
        n_list.extend([N_TIO2, N_SIO2])
        d_list.extend([d_tio2, d_sio2])
    n_list.append(1.5)
    d_list.append(np.inf)
    return n_list, d_list


def realistic_levels(n_levels):
    """The real device's level set for 300 nm Sb2Se3 on a 6-pair TiO2/SiO2 DBR
    at 532 nm. design_phase_lut TMM-corrects the indices to hit evenly spaced
    phase targets; feeding those indices back through the same TMM stack
    recovers the achieved phase (near-uniform, by construction) and the
    achieved reflectance (NOT uniform — this is the non-uniform part)."""
    n_list, d_list = _dbr_stack()
    corrected_n, corrected_refl, _ = design_phase_lut(
        n_list, d_list, WAVELENGTH_NM, n_levels=n_levels
    )
    phase, _ = stack_response(n_list, d_list, WAVELENGTH_NM, corrected_n)
    rel_phase = np.mod(phase - phase[0], 2 * np.pi)
    order = np.argsort(rel_phase)

    phase_t = torch.tensor(rel_phase[order], dtype=torch.float32, device=DEVICE)
    amplitude_t = torch.tensor(np.sqrt(corrected_refl[order]), dtype=torch.float32, device=DEVICE)
    return LevelSet("realistic", phase_t, amplitude_t)


LEVEL_BUILDERS = {"uniform": uniform_levels, "realistic": realistic_levels}


# ======================================================================
# Quantizers
# ======================================================================

def circular_delta(a, b):
    """Signed smallest angle from b to a, wrapped to (-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def nearest_index(phase, level_phase):
    d = circular_delta(phase.unsqueeze(-1), level_phase)
    return torch.argmin(d.abs(), dim=-1)


def hard_quantize(phase, levels: LevelSet):
    """Quantize-after-optimize: the naive scheme, and the eval-time readout
    for every other scheme."""
    idx = nearest_index(phase, levels.phase)
    return levels.field[idx]


def ste_quantize(phase, levels: LevelSet):
    """Straight-through estimator: forward pass is the exact quantized field,
    backward pass treats quantization as the identity on phase."""
    idx = nearest_index(phase, levels.phase)
    q_phase = levels.phase[idx]
    q_amplitude = levels.amplitude[idx]
    phase_ste = phase + (q_phase - phase).detach()
    return q_amplitude.detach() * torch.exp(1j * phase_ste.to(torch.complex64))


def gumbel_quantize(phase, levels: LevelSet, tau, hard, w=GUMBEL_W, scale=GUMBEL_SCORE_SCALE):
    """Gumbel-Softmax relaxation over the L levels (Choi et al. Eq. 6-8): a
    peaked score per level from the signed angular distance, Gumbel noise,
    softmax at temperature tau, optionally hardened with a straight-through
    argmax. The level values are complex (amplitude * phase) so the mixture
    is a physically meaningful weighted sum of achievable fields.

    `scale` is required and derived in GUMBEL_SCORE_SCALE's comment above: raw
    score (Eq. 6 as literally written, scale=1) is bounded to [0, 0.25] and
    can never out-compete standard Gumbel(0,1) noise (std ~1.28) at any
    temperature, since tau divides both terms equally. Verified empirically:
    with scale=1 the optimizer never beats a random phase pattern (efficiency
    pinned at the ~1/N^2 chance level for a point target, confirmed against
    finite differences -- not a backward-pass bug, a genuine dominated-signal
    problem). This is an engineering necessity the paper's main text does not
    spell out (it defers to "the supplement"); treat the Gumbel-Softmax
    numbers in this module as a demonstrated LOWER BOUND on that method's
    achievable performance with a principled but not paper-verified scale,
    not as a refutation of the technique itself.
    """
    delta = circular_delta(phase.unsqueeze(-1), levels.phase)
    score = torch.sigmoid(w * delta) * (1 - torch.sigmoid(w * delta))
    u = torch.rand_like(score).clamp_(1e-20, 1 - 1e-20)
    gumbel_noise = -torch.log(-torch.log(u))
    weights = torch.softmax((scale * score + gumbel_noise) / tau, dim=-1)
    if hard:
        idx = weights.argmax(dim=-1, keepdim=True)
        hard_weights = torch.zeros_like(weights).scatter_(-1, idx, 1.0)
        weights = hard_weights - weights.detach() + weights
    return (weights.to(torch.complex64) * levels.field).sum(dim=-1)


# ======================================================================
# Forward model and target
# ======================================================================

def far_field_intensity(field):
    """|FFT{field}|^2 — the same Fraunhofer far-field forward model as
    hogel_optimizer.hogel_farfield, copied minimally so this module has no
    dependency on files owned by another agent."""
    U = torch.fft.fftshift(torch.fft.fft2(field))
    return torch.abs(U) ** 2


def make_target(n=N, n_points=6, seed=0):
    """Sparse target: a handful of point sources. Cheap, but the regime where
    continuous phase already has little to lose to quantization -- see
    make_dense_target for the regime Choi et al. actually test."""
    rng = np.random.default_rng(seed)
    target = np.zeros((n, n), dtype=np.float32)
    ys = rng.integers(n // 8, 7 * n // 8, n_points)
    xs = rng.integers(n // 8, 7 * n // 8, n_points)
    target[ys, xs] = 1.0
    return target


def make_dense_target(n=N):
    """Dense natural-image target: skimage's standard 'camera' test photograph,
    downsampled to n x n. This is the content class (dense, structured,
    broadband spatial frequency content) that Choi et al. optimize for, unlike
    the sparse point target above."""
    from skimage.data import camera
    from skimage.transform import resize

    img = resize(camera().astype(np.float32) / 255.0, (n, n), anti_aliasing=True)
    return img.astype(np.float32)


# ======================================================================
# Optimization
# ======================================================================

def _init_phase(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=gen) * 2 * math.pi).to(DEVICE).requires_grad_(True)


def iters_to_converge(losses, frac=0.95):
    """First iteration at which `frac` of the total loss improvement (from
    the first to the best loss seen) has been achieved."""
    start, best = losses[0], min(losses)
    target = start - frac * (start - best)
    for i, loss in enumerate(losses):
        if loss <= target:
            return i + 1
    return len(losses)


def optimize(target, scheme, levels=None, n_iters=ITERS, lr=LR, seed=0):
    """scheme: 'continuous' | 'ste' | 'gumbel'. ('naive' needs no
    optimization loop of its own — see naive_quantize_after.)"""
    phase = _init_phase(target.shape, seed)
    target_t = torch.tensor(target, device=DEVICE)
    target_norm = target_t / target_t.sum()

    optimizer = torch.optim.Adam([phase], lr=lr)
    losses = []
    for it in range(n_iters):
        optimizer.zero_grad()

        if scheme == "continuous":
            field = torch.exp(1j * phase.to(torch.complex64))
        elif scheme == "ste":
            field = ste_quantize(phase, levels)
        elif scheme == "gumbel":
            tau = TAU_START * (TAU_END / TAU_START) ** (it / max(n_iters - 1, 1))
            field = gumbel_quantize(phase, levels, tau, hard=True)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")

        recon = far_field_intensity(field)
        recon_norm = recon / recon.sum()
        loss = torch.mean((recon_norm - target_norm) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return phase.detach(), losses


def naive_quantize_after(continuous_phase, levels):
    return hard_quantize(continuous_phase, levels)


# ======================================================================
# Metrics
# ======================================================================

def to_image(field):
    return far_field_intensity(field).detach().cpu().numpy()


def _normalize01(img):
    return img / (img.max() + 1e-12)


def compute_metrics(recon_img, continuous_img, target_img):
    recon_n = _normalize01(recon_img)
    cont_n = _normalize01(continuous_img)
    targ_n = _normalize01(target_img)

    eff, snr = efficiency_metric(recon_img, target_img)

    return {
        "psnr_vs_continuous": peak_signal_noise_ratio(cont_n, recon_n, data_range=1.0),
        "ssim_vs_continuous": structural_similarity(cont_n, recon_n, data_range=1.0),
        "psnr_vs_target": peak_signal_noise_ratio(targ_n, recon_n, data_range=1.0),
        "ssim_vs_target": structural_similarity(targ_n, recon_n, data_range=1.0),
        "efficiency": eff,
        "snr": snr,
    }


def sinc_efficiency(n_levels):
    """Closed-form efficiency of an idealized N-level blazed grating relative
    to its own continuous-phase limit: eta = (sin(pi/N)/(pi/N))^2. Proven
    symbolically in tests/test_phase_quantization.py; reused here as the
    theoretical bound that naive_quantize_after should saturate whenever the
    continuous-phase solution is locally well-approximated by a linear phase
    ramp per quantization cell (true for a grating; empirically also true for
    the sparse-point hologram target here, see docs/notes_quantization_aware.md)."""
    return (math.sin(math.pi / n_levels) / (math.pi / n_levels)) ** 2


# ======================================================================
# Sweep
# ======================================================================

def run_sweep(target):
    results = []

    t0 = time.time()
    cont_phase, cont_losses = optimize(target, "continuous", n_iters=CONT_ITERS, lr=LR)
    cont_time = time.time() - t0
    cont_field = torch.exp(1j * cont_phase.to(torch.complex64))
    cont_img = to_image(cont_field)

    results.append({
        "scheme": "continuous", "spacing": "-", "bits": float("inf"), "n_levels": float("inf"),
        "time_s": cont_time, "iters": CONT_ITERS, "iters_to_converge": iters_to_converge(cont_losses),
        "mean_level_power": 1.0, "_image": cont_img,
        **compute_metrics(cont_img, cont_img, target),
    })

    for spacing_name, build_levels in LEVEL_BUILDERS.items():
        for bits in BIT_DEPTHS:
            n_levels = 2 ** bits
            levels = build_levels(n_levels)
            mean_level_power = (levels.amplitude ** 2).mean().item()

            t0 = time.time()
            naive_field = naive_quantize_after(cont_phase, levels)
            naive_time = time.time() - t0
            naive_img = to_image(naive_field)
            results.append({
                "scheme": "naive", "spacing": spacing_name, "bits": bits, "n_levels": n_levels,
                "time_s": naive_time, "iters": 0, "iters_to_converge": 0,
                "mean_level_power": mean_level_power, "_image": naive_img,
                **compute_metrics(naive_img, cont_img, target),
            })

            t0 = time.time()
            ste_phase, ste_losses = optimize(target, "ste", levels=levels, n_iters=ITERS, lr=LR)
            ste_time = time.time() - t0
            ste_img = to_image(hard_quantize(ste_phase, levels))
            results.append({
                "scheme": "ste", "spacing": spacing_name, "bits": bits, "n_levels": n_levels,
                "time_s": ste_time, "iters": ITERS, "iters_to_converge": iters_to_converge(ste_losses),
                "mean_level_power": mean_level_power, "_image": ste_img,
                **compute_metrics(ste_img, cont_img, target),
            })

            t0 = time.time()
            gumbel_phase, gumbel_losses = optimize(target, "gumbel", levels=levels, n_iters=ITERS, lr=LR)
            gumbel_time = time.time() - t0
            with torch.no_grad():
                gumbel_img = to_image(hard_quantize(gumbel_phase, levels))
            results.append({
                "scheme": "gumbel", "spacing": spacing_name, "bits": bits, "n_levels": n_levels,
                "time_s": gumbel_time, "iters": ITERS,
                "iters_to_converge": iters_to_converge(gumbel_losses),
                "mean_level_power": mean_level_power, "_image": gumbel_img,
                **compute_metrics(gumbel_img, cont_img, target),
            })

            del levels

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return results, cont_img, target


def print_table(results):
    header = (f"{'scheme':<10} {'spacing':<10} {'bits':>5} {'levels':>7} "
              f"{'eff':>7} {'snr':>8} {'psnr_cont':>10} {'ssim_cont':>10} "
              f"{'psnr_tgt':>9} {'ssim_tgt':>9} {'time_s':>7} {'iters':>6} {'conv@':>6}")
    print(header)
    print("-" * len(header))
    for r in results:
        bits = "cont" if r["bits"] == float("inf") else str(r["bits"])
        levels = "-" if r["n_levels"] == float("inf") else str(r["n_levels"])
        print(f"{r['scheme']:<10} {r['spacing']:<10} {bits:>5} {levels:>7} "
              f"{r['efficiency']:>7.4f} {r['snr']:>8.2f} "
              f"{r['psnr_vs_continuous']:>10.2f} {r['ssim_vs_continuous']:>10.4f} "
              f"{r['psnr_vs_target']:>9.2f} {r['ssim_vs_target']:>9.4f} "
              f"{r['time_s']:>7.2f} {r['iters']:>6} {r['iters_to_converge']:>6}")


def print_sinc_law_check(results, cont_eff, target, label):
    """The key validation: does naive_quantize_after's efficiency, relative to
    the continuous-phase ceiling, match the idealized blazed-grating closed
    form (sin(pi/N)/(pi/N))^2? This is the number the README/DRAFT's "3-bit ==
    95%" claim is actually about (a RATIO to the continuous optimum, not an
    absolute efficiency).

    efficiency_metric's signal_power/total_power definition is only
    discriminative when the target's bright support is a small fraction of
    the frame (true for a sparse point target). For a dense, full-frame
    image, nearly every pixel is nonzero, so target_mask covers almost the
    whole frame and the ratio degenerates to ~1.0 for every scheme -- that is
    a property of the metric on full-frame content, not a claim about the
    optimizer, so this check is skipped and flagged when it would be
    uninformative."""
    support_fraction = float((target > 0).mean())
    print(f"\nSinc-law check ({label}): naive_eff / continuous_eff vs (sin(pi/N)/(pi/N))^2")
    print(f"  target bright-pixel support: {support_fraction:.1%} of the frame")
    if support_fraction > 0.5:
        print("  SKIPPED: efficiency_metric is not discriminative on a full-frame target "
              "(signal_power/total_power trivially -> ~1.0). See psnr/ssim ranking below instead.")
        return
    print(f"{'bits':>5} {'levels':>7} {'spacing':>10} {'naive/cont':>11} {'sinc^2':>9} {'error':>8}")
    for r in results:
        if r["scheme"] != "naive":
            continue
        ratio = r["efficiency"] / cont_eff
        bound = sinc_efficiency(r["n_levels"])
        err = (ratio - bound) / bound
        print(f"{r['bits']:>5} {r['n_levels']:>7} {r['spacing']:>10} "
              f"{ratio:>11.4f} {bound:>9.4f} {err:>+7.2%}")


def print_scheme_ranking(results, label):
    """Which scheme has the best PSNR-vs-continuous at each (spacing, bits)?
    This is the metric that stays meaningful on dense, full-frame content
    where efficiency_metric saturates (see print_sinc_law_check)."""
    print(f"\nScheme ranking by PSNR-vs-continuous ({label}):")
    print(f"{'spacing':>10} {'bits':>5} {'best':>8} {'naive':>8} {'ste':>8} {'gumbel':>8}")
    for spacing in ("uniform", "realistic"):
        for bits in BIT_DEPTHS:
            row = {r["scheme"]: r["psnr_vs_continuous"]
                   for r in results
                   if r["spacing"] == spacing and r["bits"] == bits}
            best = max(row, key=row.get)
            print(f"{spacing:>10} {bits:>5} {best:>8} "
                  f"{row['naive']:>8.2f} {row['ste']:>8.2f} {row['gumbel']:>8.2f}")


def plot_results(results, cont_img, target, suffix=""):
    schemes = ["naive", "ste", "gumbel"]
    spacings = ["uniform", "realistic"]
    colors = {"naive": "tab:red", "ste": "tab:blue", "gumbel": "tab:green"}
    styles = {"uniform": "-", "realistic": "--"}

    cont_eff = next(r["efficiency"] for r in results if r["scheme"] == "continuous")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for scheme in schemes:
        for spacing in spacings:
            rows = [r for r in results if r["scheme"] == scheme and r["spacing"] == spacing]
            rows.sort(key=lambda r: r["bits"])
            bits = [r["bits"] for r in rows]
            eff = [r["efficiency"] for r in rows]
            psnr = [r["psnr_vs_continuous"] for r in rows]
            axes[0].plot(bits, eff, styles[spacing], color=colors[scheme], marker="o",
                         label=f"{scheme} / {spacing}")
            axes[1].plot(bits, psnr, styles[spacing], color=colors[scheme], marker="o",
                         label=f"{scheme} / {spacing}")

    axes[0].axhline(cont_eff, color="k", ls=":", alpha=0.6, label="continuous ceiling")
    axes[0].set_xlabel("Bit depth")
    axes[0].set_ylabel("Diffraction efficiency (signal / total power)")
    axes[0].set_title("Efficiency vs bit depth")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Bit depth")
    axes[1].set_ylabel("PSNR vs continuous-phase optimum (dB)")
    axes[1].set_title("Reconstruction fidelity vs bit depth")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Quantization-aware optimization: naive vs STE vs Gumbel-Softmax{suffix and f' ({suffix})'}",
                 fontsize=13)
    fig.tight_layout()
    out = PLOT_DIR / f"quantization_aware_sweep{suffix and f'_{suffix}'}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # Reconstruction gallery at 3-bit (8 levels), both spacings, all schemes, vs target/continuum.
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for row, spacing in enumerate(spacings):
        images = [("target", target), ("continuous", cont_img)]
        for scheme in schemes:
            r = next(r for r in results if r["scheme"] == scheme and r["spacing"] == spacing and r["bits"] == 3)
            images.append((f"{scheme} (3-bit)", r["_image"]))
        for col, (label, img) in enumerate(images):
            ax = axes[row, col]
            if img is not None:
                ax.imshow(np.log1p(img), cmap="hot")
            ax.set_title(f"{spacing}: {label}" if col == 0 else label, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    out2 = PLOT_DIR / f"quantization_aware_recon_gallery{suffix and f'_{suffix}'}.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")


def run_and_report(target, label, suffix):
    print("\n" + "=" * 70)
    print(f"Sweep: {label}")
    print("=" * 70)

    t0 = time.time()
    results, cont_img, target = run_sweep(target)
    print_table(results)
    print(f"Total sweep wall-clock: {time.time() - t0:.1f}s")

    cont_eff = next(r["efficiency"] for r in results if r["scheme"] == "continuous")
    print(f"\nContinuous-phase ceiling efficiency: {cont_eff:.4f}")
    print_sinc_law_check(results, cont_eff, target, label)
    print_scheme_ranking(results, label)

    for spacing in ("uniform", "realistic"):
        r3 = next(r for r in results if r["scheme"] == "gumbel" and r["spacing"] == spacing and r["bits"] == 3)
        print(f"3-bit Gumbel-Softmax ({spacing}): efficiency={r3['efficiency']:.4f} "
              f"({r3['efficiency'] / cont_eff:.1%} of continuous), "
              f"PSNR-vs-continuous={r3['psnr_vs_continuous']:.2f} dB")

    plot_results(results, cont_img, target, suffix=suffix)
    return results, cont_eff


def main():
    print("=" * 70)
    print("Quantization-Aware Phase Optimization")
    print("=" * 70)
    print(f"Device: {DEVICE}, hologram size: {N}x{N}, bit depths: {BIT_DEPTHS}")
    print(f"GUMBEL_W={GUMBEL_W}, GUMBEL_SCORE_SCALE={GUMBEL_SCORE_SCALE:.2f} "
          f"(derived: {GUMBEL_CONFIDENCE} x noise_std x TAU_START / max_score)")

    run_and_report(make_target(), "sparse point target (6 points, 64x64)", "sparse")
    run_and_report(make_dense_target(), "dense natural-image target (camera, 64x64)", "dense")


if __name__ == "__main__":
    main()

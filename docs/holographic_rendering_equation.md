# A Rendering Equation for Holographic Displays

Every equation in this document that is not a definition or a citation is
re-derived in `tests/test_holographic_transport.py`, either symbolically with
sympy or numerically to machine precision on random inputs. Section 12 maps
each claim to its test. Nothing here is asserted on authority.

Scope: scalar diffraction theory for propagation, with the per-pixel modulator
response supplied by rigorous coupled-wave analysis. Section 11 states where
that seam sits and why it is there.

---

## 1. The classical reference

Kajiya's rendering equation describes radiance `L(x, ω)` leaving a surface
point in a direction:

```
L(x, ω_o) = L_e(x, ω_o) + ∫_Ω f_r(x, ω_i, ω_o) L(x, ω_i) (ω_i · n) dω_i
```

Two properties of this equation matter for what follows.

First, it is a **forward description**. It is a Fredholm equation of the second
kind whose unknown *is* the answer, and a renderer evaluates it — by a Neumann
series, by Monte Carlo path tracing, by whatever means. Nothing is being
designed.

Second, between surfaces radiance is **transported along straight rays**:

```
∂L/∂z + tan θ · ∂L/∂x = 0
```

That free-space term is the part a holographic display has to replace, because
light in a holographic display does not travel along rays. Everything else in
Kajiya's equation — emission, the BRDF, the visibility structure — survives
intact and is still the right way to describe the scene.

## 2. The three objects

**Field.** For a fully coherent source, `U(x)`, complex scalar. Intensity is
`I = |U|²`. This is what our code manipulates today.

**Mutual coherence.** Real sources have finite bandwidth and finite emitting
area, so `U` is a random process and only its correlations are stable:

```
J(x₁, x₂) = ⟨U(x₁) U*(x₂)⟩        I(x) = J(x, x)
```

**Wigner distribution.** The same information, re-coordinated into position and
spatial frequency:

```
W(x, u) = ∫ J(x + s/2, x − s/2) e^{−2πi u s} ds
```

with `u` in cycles per metre and the paraxial angle given by `sin θ = λu`.

`W` is the object this document is built on, for one reason: it is the only one
of the three that lives on the same domain as radiance. `L` is a function of
position and direction; so is `W`.

## 3. Transport

Under Fresnel propagation over a distance `z`, the Wigner distribution
undergoes a **pure shear**:

```
W_z(x, u) = W_0(x − λ z u, u)
```

No diffusion, no mixing, no loss of information. The proof is three lines in
the frequency domain: propagation multiplies `Û` by `exp(−iπλz u²)`, the Wigner
integrand picks up `exp(−iπλz[(u+ν/2)² − (u−ν/2)²])`, and that bracket is
exactly `2uν`, which is a linear phase in `ν` and therefore a shift in `x`.

Verified to a relative error of 5×10⁻¹⁶ on random band-limited fields, forward
and backward, checked in the ambiguity domain where the shear is an exact
pointwise phase and no interpolation is involved.

## 4. The bridge

Differentiate the shear:

```
∂W/∂z + λu · ∂W/∂x = 0
```

This is the free-space radiance transport equation of Section 1, with
`tan θ` replaced by `λu`. Under `sin θ = λu` the two agree to first order,
`tan θ = λu + (λu)³/2 + …`.

**So free-space transport is identical for radiance and for the Wigner
distribution.** This is worth stating precisely, because it is easy to
overclaim: coherence does not change how light propagates through empty space.
Both objects shear.

What coherence changes is **positivity**. `L` is non-negative by construction.
`W` may be negative, and its negative lobes are exactly where interference
lives. When a field is incoherent enough that `W ≥ 0` everywhere, `W` *is* a
radiance and Kajiya's equation applies verbatim. When it is not, the transport
equation still holds but the ray interpretation does not.

That is the whole relationship between the two theories, and it is why the
Wigner route is the one that produces a rendering equation rather than an
unrelated second formalism.

## 5. What is not closed-form

Free space is easy. Occlusion is not.

In the incoherent theory, occlusion is a visibility function — a binary factor
inside the integral. In the coherent theory, an occluding edge diffracts, and
the field beyond it is not the unoccluded field times a mask. There is no
closed-form transport operator for a scene with hard occluders, which is
precisely why Hogel-Free Holography spends its length on the problem and why
tensor holography, which takes a single RGB-D image and cannot see behind
edges, leaks light across depth discontinuities.

This document does not solve that. It takes the scene description as given, in
the form of an RGB-D light field produced by an incoherent path tracer, and
converts it to a target `W`. The conversion is exact only where the scene is
locally smooth in depth. **This is the largest approximation in the pipeline
and it should be named as such in any write-up.**

## 6. The display equation

Our panel is a phase-only, quantised, non-volatile modulator under coherent or
partially coherent illumination.

Let the illumination have coherent modes `{ψ_m}` with weights `{α_m}`,
`Σ α_m = 1` (Section 9 explains where these come from). Let `n(x) ∈ {0, …, N−1}`
be the level index written to the pixel at `x`, and let `R_λ(n)` be the complex
reflectance of level `n` — amplitude *and* phase, from RCWA, not assumed
uniform. Then the emitted field in mode `m` is

```
U_m(x) = ψ_m(x) · R_λ(n(x))
```

For an eye with pupil centred at `q` and accommodation `f`, write `E_{q,f}` for
the eye's imaging operator and `P_z` for propagation to the pupil plane. The
retinal intensity is an incoherent sum over modes:

```
I(r; q, f) = Σ_m α_m | E_{q,f} P_z U_m (r) |²
```

and the **holographic rendering problem** is

```
n* = argmin_{n(·) ∈ {0..N−1}^Ω}   E_{q,f} [ D( I(r; q, f), I_target(r; q, f) ) ]
```

where `I_target` comes from the scene's radiance by Kajiya, `D` is a perceptual
distance, and the expectation runs over pupil positions and accommodation
states — which is exactly the stochastic pupil sampling of SLFH.

**This is structurally different from Kajiya's equation and the difference
matters.** Kajiya's equation is solved by evaluation: you compute the radiance.
Ours is solved by *design*: you search a discrete space of modulator states for
one whose transported field resembles the target. Rendering is a forward
problem; holographic rendering is a constrained inverse problem with a forward
model inside it. Any intuition imported from a rasteriser about "cost per pixel"
does not survive that change.

## 7. The hogel operator

A hogel is not a primitive of the physics. It is a choice of representation,
and the choice has a name in signal processing.

Tiling the panel into hogels of `D` sub-pixels at pitch `p`, and computing each
hogel's angular content by a Fourier transform over its own aperture, is a
**short-time Fourier transform** of the field with a rectangular window of
width `Dp`. The per-hogel angular intensity is therefore a **spectrogram**.

The consequence is exact. Writing `A_V(τ, a)` for the ambiguity function of a
signal `V`, the spectrogram `S` of a field through a window satisfies

```
DFT_x { S }(a, k) = DFT_{τ→k} { A_field(τ, a) · A_window*(τ, a) }
```

Verified to 3×10⁻¹⁶ on random fields and random windows at three grid sizes.
In words: a hogel decomposition *smooths* the true Wigner distribution by the
window's own Wigner distribution. The hogel grid does not sample the light
field, it blurs it, and the blur kernel is set by the hogel aperture.

That immediately gives the trade-off, because the window's spatial and spectral
widths obey the Gabor bound

```
σ_x · σ_u ≥ 1/(4π)
```

with equality for a Gaussian window (verified symbolically; a Gaussian window
hits 0.07958 = 1/(4π) numerically).

**The hogel spatio-angular resolution trade-off, which SLFH and Hogel-Free
Holography both name as a fundamental defect, is the time-frequency uncertainty
principle.** It is not an artefact of a particular algorithm and no amount of
optimisation removes it. What can be changed is the window — its width, its
shape, and whether it is the same everywhere.

Concretely, for a hogel of `D` sub-pixels at pitch `p`:

```
spatial sampling    Δx = D p
angular resolution  Δθ = λ / (D p)
                    Δx · Δθ = λ            for every D
```

The product is pinned at `λ`. `D` is a single knob that sets both, in opposite
directions.

## 8. Our numbers

Applying that to the actual specification — 150 PPI, 0.5 µm sub-pixel pitch —
gives a hogel pitch of 169.33 µm and `D = 338.7` sub-pixels.

| λ | Δθ = λ / 169.33 µm | matches foveal acuity at | hogel needed for 1′ | PPI at 1′ |
|---|---|---|---|---|
| 450 nm | 9.14′ | 18.7° | 1.55 mm | 16.4 |
| 532 nm | 10.80′ | 22.5° | 1.83 mm | 13.9 |
| 632 nm | 12.83′ | 27.2° | 2.17 mm | 11.7 |

Foveal acuity is about 1 arcminute. Our grid delivers 9–13 arcminutes. Under
Watson's `1/(1 + e/2.3)` falloff, 10.8 arcminutes is precisely what the eye
requires at about **22 degrees of eccentricity**.

The conclusion follows without any further argument:

> **The uniform 150 PPI hogel design is not an under-foveated foveal display.
> It is a correctly specified peripheral display with no fovea.**

And the exchange rate is brutal and symmetric: reaching 1 arcminute costs the
same factor in spatial resolution, dropping the panel from 150 PPI to about
14 PPI. You may have 150 PPI at 11 arcminutes, or 14 PPI at 1 arcminute. The
uncertainty bound forbids both.

## 9. Partial coherence, modes, and speckle

Carrying `J(x₁, x₂)` directly would square the dimension. Wolf's **coherent-mode
decomposition** avoids it: `J` is Hermitian and positive semi-definite, so

```
J(x₁, x₂) = Σ_m α_m φ_m(x₁) φ_m*(x₂)
```

with `{φ_m}` its eigenfunctions and `α_m ≥ 0`. Intensity is `Σ_m α_m |φ_m|²`,
and each mode propagates by the ordinary coherent machinery we already have.
The cost of partial coherence is therefore a factor `M`, the number of
significant modes — not a squared dimension. Verified numerically: random
positive semi-definite `J`, eigendecomposition, exact reconstruction, rank
equal to the mode count.

This is also where speckle enters the theory rather than being observed in the
output. A fully developed speckle intensity is negative-exponential. Averaging
`M` independent realisations gives a Gamma-distributed intensity with contrast

```
C = σ_I / ⟨I⟩ = 1 / √M
```

Verified symbolically via sympy's Gamma distribution and by Monte Carlo at
M = 1, 4, 16, 64.

`M` is a single number that three separate things all contribute to: the
number of time-multiplexed subframes, the number of illumination modes from a
reduced-coherence source, and the number of wavelengths in a multiplexed
scheme. They are interchangeable in the formalism, and the achievable subframe
count is exactly what the phase-change-material switching budget determines.
**The coherence formalism and the hardware timing budget are the same
variable.**

## 10. Where foveation enters

Section 7 established that the hogel window is the free parameter and Section 8
established that a uniform window is a peripheral-grade choice. Foveation is
therefore not a pruning heuristic bolted onto the pipeline. It is the statement
that the STFT window should be a function of retinal eccentricity:

```
D = D(e)
```

which in signal-processing terms is an **adaptive-window STFT**, i.e. a
multiresolution analysis, and in optical terms is a hogel grid whose aperture
varies across the panel.

The feasibility condition follows from the two requirements at eccentricity `e`:

```
angular:  λ / (D p) ≤ Δθ(e)   ⟹   D ≥ λ / (p Δθ(e))
spatial:  D p       ≤ Δx(e)   ⟹   D ≤ Δx(e) / p

feasible ⟺  Δx(e) · Δθ(e) ≥ λ
```

Both `Δx` and `Δθ` grow with eccentricity as acuity falls, so their product
grows: **the constraint is tightest at the fovea and slackens outward.** That is
the formal reason foveating a hogel grid is possible at all, and it is also the
reason the fovea is where our current design fails.

Two honest gaps in this section. First, the mapping from retinal eccentricity
to a requirement `Δx(e)` at the panel depends on the eyepiece, which we have
not specified; without it, only the angular column of Section 8's table is
firm. Second, `Δθ(e)` is taken from an acuity model fitted to incoherent
natural imagery, and no published model describes speckle detectability versus
eccentricity. Section 9 gives us the contrast `1/√M` but not how visible that
contrast is off-axis.

## 11. Limits of this formalism

**The scalar/RCWA seam.** Propagation here is scalar. At a 0.5 µm pitch with
450–632 nm light the pitch is comparable to the wavelength, which is where
scalar diffraction theory degrades — and our own tier-2 work already measured
the consequence: a naive scalar phase model gives 44.5% first-order efficiency
where the TMM-corrected model gives 84–91%. The formalism handles this by
refusing to model the pixel scalar-ly: `R_λ(n)` in Section 6 is a complex
reflectance obtained from RCWA, and the scalar theory is used only for
propagation *between* the modulator plane and the eye, where feature sizes are
large. The seam is explicit and it is at the modulator surface.

**Paraxial.** The shear in Section 3 is the Fresnel result. Our ±32° field of
view at 532 nm is not comfortably paraxial; `tan θ` versus `sin θ` differ by
about 18% at 32°. The transport structure survives — the exact angular spectrum
is still a one-to-one remapping — but the clean shear does not, and anything
quantitative at the edge of the field needs the non-paraxial kernel.

**No polarisation.** Scalar throughout. Sb₂Se₃ on a DBR at sub-wavelength pitch
will have a polarisation-dependent response that `R_λ(n)` currently hides.

**Occlusion.** Section 5.

## 12. Verification index

| Claim | Section | Test |
|---|---|---|
| Shear phase algebra `(u+ν/2)² − (u−ν/2)² = 2uν` | 3 | `test_shear_phase_algebra_is_exact` |
| Free-space propagation shears the Wigner distribution | 3 | `test_wigner_transport_is_a_pure_shear` |
| Wigner is real; energy conserved under propagation | 3 | `test_wigner_is_real_and_conserves_energy` |
| Shear satisfies the radiance transport PDE | 4 | `test_shear_satisfies_the_radiance_transport_equation` |
| `tan θ = λu + O((λu)³)` | 4 | `test_paraxial_identification_is_first_order_consistent` |
| Spectrogram factorises into ambiguity functions | 7 | `test_spectrogram_factorises_into_ambiguity_functions` |
| D sub-pixels give exactly D angular samples | 7 | `test_hogel_angular_sample_count_equals_subpixel_count` |
| Gabor bound, equality for a Gaussian | 7 | `test_gaussian_window_attains_the_gabor_bound_symbolically` |
| Gabor bound numerically | 7 | `test_numeric_gaussian_window_matches_the_bound` |
| `Δx · Δθ = λ` for every D | 7 | `test_spatial_and_angular_resolution_are_one_knob` |
| 150 PPI is a peripheral-grade design | 8 | `test_our_hogel_grid_is_a_peripheral_grade_design` |
| Coherent-mode decomposition reconstructs J | 9 | `test_coherent_mode_decomposition_reconstructs_the_coherence_function` |
| Speckle contrast `1/√M`, symbolically | 9 | `test_speckle_contrast_is_one_over_root_m_symbolically` |
| Speckle contrast, Monte Carlo | 9 | `test_speckle_contrast_monte_carlo` |
| Hogel feasibility `Δx·Δθ ≥ λ` | 10 | `test_hogel_feasibility_condition` |

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_holographic_transport.py -v
```

## References

Goodman, *Statistical Optics* and *Introduction to Fourier Optics* — Wigner
transport, speckle statistics. Wolf, *Coherent-mode representation of partially
coherent sources*, JOSA 72(3), 1982. Kajiya, *The rendering equation*,
SIGGRAPH 1986. Bastiaans, *Wigner distribution function applied to optical
signals and systems*, Opt. Commun. 25, 1978 — the shear. Watson, *A formula for
human retinal ganglion cell receptive field density*, J. Vision 14(7), 2014 —
the acuity falloff. Schiffers et al., *Stochastic Light Field Holography*,
arXiv:2307.06277. Chakravarthula et al., *Hogel-free Holography*, ACM TOG 2022
— see `docs/notes_hogel_free_holography.md`.

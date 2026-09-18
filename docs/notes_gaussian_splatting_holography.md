# Gaussian Splatting for Holography — can it skip the light field?

Primary source, read in full: Choi, Chao, Yang, Gopakumar, Wetzstein. *Gaussian
Wave Splatting for Computer-Generated Holography.* ACM TOG 44(4), SIGGRAPH
2025. arXiv:2505.06582, DOI:10.1145/3731163. PDF: `docs/papers/2505.06582.pdf`.
This is the only paper found that goes **3D Gaussians → hologram** directly,
and it answers the question the boss asked almost exactly. Everything else
below is context, adjacent evidence, or papers that looked like a direct hit
by title and turned out not to be.

**Correction, second pass:** the first version of this file said refraction
"is not solved by anyone" for Gaussian splatting. That was wrong, or at
least too strong — it was true of Gaussian Wave Splatting specifically, but
the *ray-traced* branch of 3DGS (a paper I had already downloaded,
arXiv:2407.07090, and had not actually read) does real, index-of-refraction,
Snell's-law refraction with secondary rays, demonstrated in simulation. See
the "Round 2" section near the end of this file for the corrected, properly-
researched answer to the refraction question, including 3DGUT, hair, and
full path-traced Gaussian global illumination. The verdict below on the
*Fourier/coherence/hologram* side (§1–3, §6) is unaffected and still stands;
only the refraction verdict (§4) needed revising.

## Verdict, up front

**The analytic-Fourier idea is real, not a trap — but it is not a free lunch,
and it does not currently reach our scene.** A Gaussian's Fourier transform
being a Gaussian survives the full projection from a 3D scene primitive to a
2D wavefront at a fixed plane: this is exact, published, peer-reviewed math
(GWS Eq. 9–10), and it has been built into a benchtop hologram that was
captured on real hardware. That part is not speculation.

What breaks the "skip everything" story:

1. **Alpha blending's order-dependence is real and it does bite.** The exact,
   order-correct wave-domain compositing needs a per-Gaussian convolution in
   the Fourier domain (their Eq. 11) that cannot be parallelized per-frequency.
   The authors' own "exact" GWS takes 180–470 s for just 15,000 Gaussians at
   1024×1280–2000×2000 resolution (Table 3) — far from real time, at a scene
   scale two to three orders of magnitude smaller than ours. Their fast,
   parallel variant only gets a 30× speedup by **throwing the ordering away**
   (order-invariant transparency, Eq. 12–13) and then patching the resulting
   loss of occlusion with a heuristic (view-dependent opacity via spherical
   harmonics). That patch is not free — it is exactly the kind of workaround
   CLAUDE.md tells us to be suspicious of, and the authors are honest about it
   producing "background color leakage" artifacts.
2. **Coherent-vs-incoherent is bridged, not dissolved.** GWS is explicit that
   "the degree of freedom to represent arbitrary spatio-angular light field is
   inherently limited by the rank-1 constraint of coherent wavefronts" — the
   same Wolf coherent-mode argument this repo already leans on in
   `tier2_hfh`. Their fix for view-dependent effects is to **average multiple
   partially-coherent, randomly-phased renders** (24 frames in their demo).
   That is the same time-multiplexing idea our own pipeline already uses; it
   is not a shortcut around the physics, and it costs frames.
3. **No refraction, at all.** GWS's alpha wave blending is the wave-optics
   twin of ordinary 3D/2D Gaussian-splatting volume rendering — one opacity
   and one wavefront per Gaussian, composited front-to-back along a straight
   line to the SLM. It is exactly as "surface-ish" as a depth map, just with
   a soft falloff instead of a hard edge. It does not bend rays. Nobody has
   published Gaussian-splatting CGH with refraction (see below).
4. **Scale.** GWS's own benchmark tops out at 2000×2000 (4 Mpixel) exact
   holograms from 15k Gaussians. Our panel needs ~0.26 Gpixel per eye — 65×
   more pixels — from a Cornell box scene that will need many more than 15k
   Gaussians to resolve sharp specular and refractive detail. This is the
   same wall `docs/notes_hogel_free_holography.md` already found for
   hogel-free holography: a single global, non-decomposable optimization or
   closed form does not survive to hogel-array panel sizes.

So: real math, real hardware demo, real speedup over classical primitive-based
CGH — but for a small, fully-diffuse-to-mildly-occluded scene at sub-4-Mpixel
scale. It has not been shown to survive contact with either our panel size or
our glass sphere.

## 1. Does the analytic Fourier transform survive 3D Gaussian → 2D wavefront?

Yes, exactly, for a single flat (2D) Gaussian at a known pose. GWS's
derivation (§4.1, Eq. 7–10) goes: a zero-mean unit-variance Gaussian's Fourier
transform is a Gaussian (Eq. 7); an affine transform (scale + rotation +
translation) taking the Gaussian from canonical space to its position/pose in
"hologram space" maps to a **remapping of the angular-spectrum coordinates
plus a phase ramp**, by the affine theorem for 2D Fourier transforms
(Bracewell et al. 1993) and the tilted-plane angular-spectrum rotation
identity (Matsushima et al. 2003). The result (Eq. 10) is closed form:

```
û_i(k) = 2π · det(J) · det(S_i) · exp(−2π² kᵀ Σ_i k) · exp(j k·μ_i)
```

— i.e. still a Gaussian in k-space, now anisotropic via the 3×3 covariance
Σ_i = R_i S_i S_iᵀ R_iᵀ, times a phase ramp for the 3D position. No FFT per
primitive is needed; this is why it is efficient. Note the paper deliberately
uses **2D** Gaussian disks (2DGS, Huang et al. 2024) rather than the usual
3DGS ellipsoids, specifically because a flat primitive has a well-defined
angular spectrum on a plane — the authors state this reasoning explicitly.
They also note the derivation generalizes to any primitive with a closed-form
canonical-space angular spectrum, not just Gaussians (footnote after Eq. 10).

This closed form is for a **single** Gaussian's contribution at the SLM
plane, independent of all the others — the multi-primitive difficulty is
entirely in the compositing step (§2 below), not in the per-Gaussian Fourier
transform.

## 2. Does order-dependent alpha blending break the analytic path?

Yes — this is the real cost center, and the paper is candid about it.

- **Exact treatment ("alpha wave blending", Eq. 5–6):** the wave-domain twin
  of the Kajiya–Von Herzen volume rendering equation. Each Gaussian's
  wavefront is attenuated by the accumulated transmittance
  `T_i(x) = Π_{j<i}(1 − o_j|u_j(x;z_j)|)` before being propagated to the SLM
  — a genuinely nonlinear, order-dependent, nested product. In the Fourier
  domain (Eq. 11) this becomes a **convolution per Gaussian**
  `û_i(k) ∗ F(T_i(x))`, which cannot be evaluated per-frequency-bin in
  parallel — it is exactly the kind of non-local coupling our own CGH work
  already knows to be expensive (compare: the WFT/hogel argument in
  `docs/notes_hogel_free_holography.md`).
- **Fast, approximate treatment (§4.2, "order-invariant transparency",
  Eq. 12–13):** drops `T_i(x) ≈ 1`, i.e. discards ordering entirely and sums
  all Gaussians' contributions unconditionally weighted by opacity. This
  *is* parallelizable per-frequency (their 30× CUDA speedup). But without
  ordering, two Gaussians on the same line of sight from opposite sides give
  identical renders regardless of which one is "in front" (their Fig. 3
  demonstrates this failure directly). Their fix is to make opacity
  **view-dependent** via spherical-harmonic coefficients — a learned
  heuristic that recovers plausible-looking occlusion cues, not an ordering
  guarantee.
- **Runtime (Table 3):** exact GWS: 180 s (1024×1280) to 470 s (2000×2000)
  for 15k Gaussians. Fast GWS: 6.84 s to 18.06 s for the same. Classical
  point-based CGH (Chen & Wilkinson 2009): 2.47 s at 1024×1280/15k points.
  None of these are real-time, and all are far below our target resolution.

**Conclusion: yes, it breaks the fully-analytic, fully-parallel story.** You
get to choose between (a) exact and slow, with a per-primitive Fourier
convolution, or (b) fast and order-blind, patched with a learned opacity
heuristic that only partially restores correct occlusion.

## 3. Is fitting Gaussians to incoherent radiance and calling it a coherent field a category error?

Not an error, but not free either — and it directly touches the 99%
incoherence result already measured in this repo (`tests/test_coherent_modes.py`,
`docs/notes_hogel_free_holography.md`).

GWS states the limit plainly (§5.2, "Partially Coherent GWS"): representing
an arbitrary spatio-angular light field with a **single coherent** wavefront
per Gaussian is "inherently limited by the rank-1 constraint of coherent
wavefronts" (citing Zhang, *Analysis and synthesis of 3D illumination using
partial coherence*, Stanford PhD thesis 2011 — the same rank-1/coherent-mode
argument Wolf's theorem gives us). Their bridge is to attach an **angular
kernel** with a randomly sampled phase `φ(k) ~ U[−π, π]` to each Gaussian
(Eq. 14) and average multiple such renders — 24 frames in their toy
parallax demo — so that the ensemble reproduces the desired (incoherent-like)
angular radiance pattern in expectation. This is the same idea as
time-multiplexed / partially-coherent holography already cited in this
project's other notes (`choi2022time` — Time-multiplexed Neural Holography,
arXiv:2205.02367, already in `docs/papers/`), not a new trick specific to
Gaussians.

Practical upshot for us: GWS's default (non-partially-coherent) mode "only
ensures correct in-focus imagery, exhibiting unnatural defocus blur and a
limited eyebox" (their own Limitations). Getting believable view-dependent,
incoherent-looking behavior costs extra frames of averaging, same as our own
approach — Gaussians do not remove this cost, they just give you a clean
analytic single-frame building block to average over.

## 4. Refraction and the glass sphere — the load-bearing question

**Not solved, by GWS or by anyone connecting refraction to CGH.** GWS's
alpha wave blending is geometrically a straight-line, front-to-back
composite exactly like ordinary 3DGS/2DGS rasterization — it has no notion
of a bent secondary ray. It would treat the glass sphere exactly like a
depth-map method does: one primitive, one straight propagation path, at the
sphere's outer surface. The refracted interior image would not appear
correctly, for the same structural reason a depth map fails.

Searching specifically for Gaussian-splatting + refraction turns up a small,
recent (all still simulation-only) cluster, **none of it connected to
holography**:

- **RefracGS** (Shao et al., arXiv:2603.21695, SIGGRAPH-track 2026,
  `docs/papers/2603.21695.pdf`) — refraction-aware 3D Gaussian ray tracing
  through **one** non-planar interface (a water surface), using Snell's law
  with a neural height field for the interface and a Gaussian field for the
  scene beneath it. Real-time at 200 FPS, 15× faster training than prior
  refractive baselines. This is a single refracting boundary between two
  known media, not a solid two-surface lens.
- **TransparentGS** (Huang et al., arXiv:2504.18768,
  `docs/papers/2504.18768.pdf`) — the closest match to "refracted image of
  the surrounding scene appears inside a solid transparent object." Uses
  "transparent Gaussian primitives" with a deferred-refraction shading
  strategy, plus baked Gaussian light-field probes (GaussProbe) for nearby
  content, and an **iterative probe query (IterQuery)** whose whole purpose
  is to reduce — not eliminate — the parallax error the probe
  approximation introduces. That residual-error-correction step is itself
  evidence that parallax-correct refraction through a solid Gaussian object
  is still an open, only-approximately-solved problem in this line of work.
- **RT-Splatting** (Shi et al., arXiv:2605.18263,
  `docs/papers/2605.18263.pdf`) — joint reflection/transmission for
  semi-transparent *surfaces* (e.g. windows), factoring geometric occupancy
  from optical opacity. Still surface-like, not a volumetric lens with two
  curved interfaces.
- **3D Gaussian Ray Tracing** (Moenne-Loccoz et al., NVIDIA, arXiv:2407.07090,
  `docs/papers/2407.07090.pdf`) — the enabling substrate: ray-traces
  Gaussians via a BVH of bounding proxies instead of rasterizing them,
  which is what makes secondary rays (shadows, reflections, and in
  follow-on work, refraction) tractable at all. The base paper's own
  abstract only claims shadows and reflections as the payoff, not
  refraction.

None of these four has been combined with a hologram/CGH pipeline, coherent
or otherwise. Given that even *radiance-only* refraction through a solid
object needs specialized machinery and still has acknowledged residual
parallax error, and that GWS's own occlusion model already needs a
heuristic patch for ordinary opaque occlusion, a physically correct
glass-sphere hologram from Gaussian splatting is not a small extension —
it would be original, unpublished work stacking two open problems.

## 5. Can splatting at least render our 289 views faster, even if the analytic route fails?

Yes, for the parts of the scene 3DGS already renders correctly (i.e., not
the glass sphere) — and the speedup is large and directly measured on real
GPU hardware, though not on an optical bench.

**CoherentRaster** (Sim, Shin, Jeon, Lee, Choo, Cho — POSTECH/ETRI, SIGGRAPH
2026, arXiv:2605.04509, `docs/papers/2605.04509.pdf`) targets exactly our
adjacent question: rendering many-view light fields from 3D Gaussians fast
enough for a real display, by rasterizing directly at the interlaced
subpixel level instead of rendering N full frames and resampling. Measured
on an RTX 5090 (Table 1):

| Scene | Views / resolution | 3DGS baseline (full multi-view render) | CoherentRaster |
|---|---|---|---|
| Synthetic Blender | 63 views, 2K (1440×2560) | 5.8 FPS | 88 FPS (cluster size 8) |
| Mip-NeRF 360 (real) | 63 views, 2K | 3.9 FPS | 56 FPS |
| Mip-NeRF 360 (real) | 71 views, 4K (3840×2160) | 2.1 FPS | 23 FPS |

That is a 10–15× speedup over rendering the views with 3DGS one-by-one, and
3DGS itself is already >100× faster than a 256-spp path tracer for ordinary
diffuse/glossy content. Caveats that matter for us: (1) their view counts
(63–71) are fewer than our 289 (17×17), though the method's clustering
scheme should scale further; (2) quality (PSNR/SSIM in Table 1) is measured
against **full 3DGS rendering as pseudo-ground-truth**, not against a
path-traced reference — so this tells us nothing about whether 3DGS itself
gets the glass sphere right (it does not, by the same surface/opacity
argument as §4); and (3) this produces ordinary incoherent RGB views, feeding
the same downstream WFT/hogel pipeline we already have — it is a drop-in
replacement for the renderer, not a replacement for the architecture.

Similarly, **Light Field Display Point Rendering** (Gavane & Watson,
arXiv:2601.19901) reports a 2–8× speedup over multi-view rendering using
texture-splatted point primitives for LFDs — an older, non-Gaussian sibling
of the same idea, useful mainly as confirmation that "stop rendering N full
frames, splat directly to the interlaced/sub-hologram target" is a general
and repeatedly-validated speedup, not a fluke of one paper.

## 6. Gaussians, Wigner distributions, and the Gabor bound — real connection, or wishful thinking?

**No paper was found making this connection explicit.** Targeted search for
Gaussian/minimum-uncertainty/Gabor-bound arguments applied to CGH or to
Gaussian splatting turned up nothing relevant (mostly unrelated pure-math
Gabor-transform papers). GWS's own derivation exploits the Gaussian's
self-Fourier property (§4.1, Eq. 7) as a matter of calculational convenience,
and the paper never invokes the uncertainty principle or Wigner/ambiguity
functions to justify it.

The observation that a Gaussian uniquely attains equality in the
space-bandwidth (Gabor) product — already verified elsewhere in this repo —
is true and plausibly *is* part of why Gaussian primitives are a good match
for a windowed-Fourier-transform-based hologram pipeline (the hogel/WFT
architecture already forces a joint space-frequency tiling, and a Gaussian
window is the matched, minimum-uncertainty choice for exactly that kind of
tiling). But this is our own inference, not a claim anyone in the literature
has made or tested. Flag it as a hypothesis worth a small feasibility check
(e.g., does a Gaussian-windowed WFT hogel outperform the Hamming window HFH
uses?) rather than as a validated result.

## 7. Does Gaussian splatting add anything over classic point-based CGH?

Yes, on the evidence in GWS's own Table 2 (image quality) and Table 3
(runtime): compared with the classical point-based method (Chen & Wilkinson
2009, GPU point-cloud CGH) and polygon-based methods (Matsushima & Nakahara
2009), GWS gets **5–11 dB higher PSNR** at a runtime within the same order
of magnitude (Table 3: 2.47 s point-based vs. 6.84 s fast-GWS at matched
resolution/primitive count). The reason given is structural, not just a
better fit: point clouds are inherently sparse and need an "excessive number
of primitives" or heuristic point-size tuning to avoid visible holes, and
polygon meshes with per-face color (required for polygon CGH, since texture
mapping has no known polygon-CGH treatment) are "overly smooth" unless an
impractically large triangle count is used. A Gaussian's smooth,
anisotropic, differentiably-optimized falloff fills gaps and captures
anisotropic detail neither primitive does well. So: real, demonstrated
value over 1990s–2000s point/polygon CGH — but that comparison is entirely
about *primitive quality per unit of scene*, and does not touch the panel-
scale or refraction problems above.

## What is demonstrated in hardware vs. simulated vs. proposed

- **Demonstrated in hardware:** GWS benchtop hologram (FISBA READYBeam fiber
  laser + Holoeye PLUTO phase-only LCoS SLM), 3-wavelength captured focal
  stacks, real occlusion and refocus effects from 2DGS scenes optimized from
  a handful of photographs. Scale: 15k Gaussians, ≤2000×2000 hologram,
  ~1 cm depth range in hologram space (~0.5 m–∞ in view space). No
  refractive/glass object attempted.
- **Simulated only (GPU benchmark, no optical bench):** CoherentRaster
  (light-field rasterization speed), RefracGS and TransparentGS and
  RT-Splatting (refraction/transmission in radiance-only Gaussian
  splatting), CVQPG (below).
- **Proposed / adjacent, not answering this question despite matching
  titles:**
  - *Hologram Representation via Quadratic Phase Gaussian Splatting* (Wang,
    Zhan, Akşit, Qiu — arXiv:2609.11434, `docs/papers/2609.11434.pdf`) uses
    Gaussian-like 2D quadratic-phase primitives to **compress an
    already-computed** hologram's wavefront (+0.19–0.33 dB over prior
    hologram-representation compression at equal parameter count). This is
    about representing a hologram after it exists, not generating one from
    a scene.
  - *Gaussian splatting holography* (Zhang & Cao, Tsinghua, arXiv:2509.20774,
    `docs/papers/2509.20774.pdf`) uses Gaussian splatting as a compact,
    twin-image-suppressing parameterization for **in-line hologram phase
    retrieval** — an inverse problem on a captured lensless hologram,
    unrelated to synthesizing a hologram from a 3D scene. Worth ruling out
    explicitly since the title is an exact match for the search term.

## Recommendation

Do not expect a shortcut that skips path-traced light-field rendering for
this scene. GWS's analytic Gaussian-to-wavefront transform is genuine and
well-demonstrated, but every one of its costs lands exactly where our
scene is hardest: order-correct occlusion (slow), incoherent-looking
view-dependence (needs multi-frame averaging, same as we already do),
scale (4 Mpixel demonstrated vs. our ~0.26 Gpixel/eye target, with a scene
that will need far more than their 15k Gaussians), and refraction (not
modeled at all, and the nearest published refraction-capable Gaussian
methods handle only a single interface or leave acknowledged residual
parallax error, with no connection to holography).

The lower-risk, actually-useful borrow from this literature is narrower:
**use a 3DGS-family rasterizer (à la CoherentRaster) as a drop-in
replacement for the path tracer that generates our 289-view light field**,
keeping the rest of our hogel/WFT pipeline unchanged. That is a measured
10–15× rendering speedup on real hardware, for the diffuse/glossy majority
of the Cornell box. It does not fix the glass sphere — 3DGS is exactly as
surface-ish as everything else here, so the sphere would still need the
path tracer (or a not-yet-existing refraction-aware Gaussian renderer,
unvalidated for this geometry) until someone publishes the combination we
were hoping already existed.

---

## Round 2 — ray-traced Gaussian splatting, properly read

The coordinator caught a real gap: `docs/papers/2407.07090.pdf` (3D Gaussian
Ray Tracing) was downloaded in round 1 and never opened. It should have
been — it changes the refraction answer. This section reads it in full,
plus its direct successor (3DGUT) and the surrounding reflection/refraction/
hair/path-tracing literature, and corrects §4 above accordingly. §1–3 and
§6 (the Fourier/coherence/hologram math) are untouched by this correction —
none of the papers below touch holography at all.

## Revised refraction verdict

**Ray-traced Gaussian splatting does support real, Snell's-law refraction
with secondary rays — read in full from arXiv:2407.07090 (3D Gaussian Ray
Tracing, "3DGRT," Moenne-Loccoz et al., NVIDIA, ACM TOG 43(6) / SIGGRAPH Asia
2024, DOI:10.1145/3687934, `docs/papers/2407.07090.pdf`), demonstrated in
simulation, not yet in hologram or optical hardware, and not yet as an
inverse-rendering fit of an unknown glass object.** The important nuance the
boss should have: it is not that a solid glass object gets represented *as
Gaussians* and Gaussians alone bend light — it is that 3DGRT ray-traces a
**mixed scene** of Gaussians (fit from photos, for diffuse/glossy content)
and ordinary **triangle meshes with real materials** (for known refractive
or mirror objects), tracing both together in one BVH. Quoting the method
(§6.1, "Reflections, Refractions and Inserted Meshes"):

> "Optical ray effects are supported by interleaved tracing of triangular
> faces and Gaussian particles... When casting each ray... we first cast
> rays against inserted meshes; if a mesh is hit, we render all particles
> only up to the hit, and then compute a response based on the material.
> For refractions and reflections, this means continuing tracing along a
> new redirected ray according to the laws of optics."

This is demonstrated qualitatively in the paper's Figure 2 and Figure 13
("reflections... refractions... hard shadows... myriad combinations") —
real secondary-ray optics, not a baked approximation. For a synthetic
Cornell-box scene where the glass sphere's geometry and index of refraction
are already known exactly (as ours is), this is a good structural fit: put
the box/diffuse content in as Gaussians, insert the sphere as an ordinary
ray-traced glass mesh with the correct IOR, and ray-trace the combination.
Because this is literal Snell's-law ray bending through a known interface —
the same physics the reference path tracer already uses for that object —
the internal image's parallax would come out correct for the same reason
the path tracer's does, not through any Gaussian-specific trick.

**What is not demonstrated:** fitting an *unknown* refractive object's shape
and index purely from photographs and getting correct bent rays out the
other side. The paper is explicit that this section is "manually-specified
forward rendering, although inverse rendering in concert with these effects
is indeed supported by our approach, and is a promising area for ongoing
work" (§6.1) — i.e., an open problem, not a demonstrated one. Our case sits
on the easy side of that line (we already know the sphere's geometry and
IOR from the scene definition), but it is worth being precise that the hard
version — reconstructing an unknown glass object's shape/IOR as Gaussians
from images and getting it to refract correctly — remains unsolved.

**Total internal reflection:** not explicitly discussed, tested, or
benchmarked in 3DGRT, 3DGUT, or any of the other papers below. "Continuing
tracing... according to the laws of optics" should, if implemented
correctly, include the standard Snell's-law check for TIR past the critical
angle — but no paper claims this explicitly or shows a TIR test case. This
is an inference, not a confirmed result; flag it as such rather than
assuming it works.

**Speed:** 3DGRT's ray tracer alone (no secondary rays) runs 52–190 FPS on
an RTX 6000 Ada depending on dataset (Table 2), about 2–4× slower than pure
3DGS rasterization (238–476 FPS) but still real-time. No isolated FPS number
is given for the reflection/refraction demo specifically — those figures are
qualitative. Either way, this is orders of magnitude faster than a 256-spp
path tracer, so even at a "with secondary rays" penalty on top of the base
ray-tracing cost, it stays in a completely different performance class than
our current bottleneck.

## 3DGUT — does it supersede 3DGRT?

**Yes, for the primary-ray/main-scene cost, while keeping 3DGRT's tracer for
the secondary rays that need it.** 3D Gaussian Unscented Transform ("3DGUT,"
Wu, Martinez Esturo, Mirzaei, Moenne-Loccoz, Gojcic, NVIDIA, arXiv:2412.12507,
`docs/papers/2412.12507.pdf`, from the same group) replaces 3DGS's EWA
splatting Jacobian with an Unscented-Transform approximation of each
Gaussian by sigma points that can be projected exactly under any nonlinear
camera model — this is what gives it distorted-camera and rolling-shutter
support "for free." Separately, it aligns its particle-ordering and
response-evaluation formulas with 3DGRT's, so the **same trained
representation can be rendered by either rasterization or ray tracing**.

The payoff for us is the hybrid mode (§6.2 of the paper): "we first compute
all the primary rays intersections with the scene, then render these
primary rays using rasterization... Next, we compute and trace the
secondary rays using 3DGRT. This hybrid rendering method allows us to
achieve complex visual effects, such as reflections and refractions, that
would otherwise only be possible with ray tracing." Measured speed
(Table 1/2, MipNeRF360, RTX 6000 Ada): 3DGUT's sorted/tracing-compatible
variant renders at 200–272 FPS, a full order of magnitude faster than plain
3DGRT's 52 FPS on the same benchmark, and within a small factor of 3DGS's
238–347 FPS — because the expensive ray tracer is only invoked for the
(comparatively rare) secondary rays that actually hit a reflective/
refractive surface, not for every pixel of the whole frame. So 3DGUT is
the more practical vehicle: fast splatting for the diffuse Cornell-box
walls, 3DGRT-compatible ray tracing only for the rays that enter the glass
sphere.

## Reflection literature: two genuinely different categories

Searching turned up more than a dozen "reflective Gaussian splatting"
papers. They split cleanly into two families, and conflating them is exactly
the trap the coordinator was worried about:

- **True secondary-ray tracing** (can, in principle, also refract): 3DGRT
  (arXiv:2407.07090), 3DGUT (arXiv:2412.12507), **RT-GS** (Zeng, Peng, Xie,
  Tomizuka, Yuksel, arXiv:2604.00509, `docs/papers/2604.00509.pdf`) — a
  microfacet material model combined with differentiable ray tracing, using
  *separate* Gaussian primitives for reflection and for transmittance so it
  can "jointly model specular reflection and transmittance" and reconstruct
  "objects behind transparent surfaces" — and **Reflective Gaussian
  Splatting / "Ref-Gaussian"** (Yao, Zeng, Gu, Zhu, Zhang, arXiv:2412.19282,
  `docs/papers/2412.19282.pdf`), which implements Gaussian-grounded
  inter-reflection ("for the first time," per its abstract) inside a
  physically-based deferred-rendering framework. These are the ones doing
  real light transport, not a shading trick.
- **Baked / deferred shading approximations** (visually convincing on
  glossy or mirror-like surfaces, but not physically bending rays through a
  volume): GaussianShader (Jiang et al., arXiv:2311.17977,
  `docs/papers/2311.17977.pdf` — a per-Gaussian shading function driven by
  an estimated normal, environment-map style, +1.57 dB over vanilla 3DGS,
  0.58 h training vs Ref-NeRF's 23 h), 3D Gaussian Splatting with Deferred
  Reflection (arXiv:2404.18454), Ref-GS directional factorization
  (arXiv:2412.00905), Ref-DGS (arXiv:2603.07664, explicitly markets itself
  as avoiding "explicit ray tracing at substantial computational cost"),
  SSR-GS (arXiv:2603.05152), Ref-Unlock (arXiv:2507.06103), RGS
  (arXiv:2609.19421), HybridSplat (arXiv:2512.08334). These bake a
  view-dependent appearance model (environment maps, spherical Gaussians,
  split-sum BRDF terms) onto surface-aligned Gaussians. They are fast
  (near-3DGS speed) and look convincing for shiny floors, metal, and glass
  *windows* viewed near head-on — this is almost certainly what the user
  has seen and correctly remembered as "reflections rendered convincingly."
  But none of them trace a bent secondary ray through a volume, so none of
  them would get the glass sphere's internal parallax right; they would
  paint a plausible-looking specular highlight on its surface and nothing
  correct on the inside.

This distinction is the answer to "which handle true specular reflection
versus baked view-dependent colour": the ray-traced family (3DGRT, 3DGUT,
RT-GS, Ref-Gaussian) does the former; everything in the second list does the
latter.

## Full path tracing on Gaussian primitives

Yes, this now exists, simulation-only: **PTIR-GS** (Zhu, Zhang, Zhu, Li, Hu,
Gai, Zhu, Huang, Li, arXiv:2606.09606, `docs/papers/2606.09606.pdf`,
2026) is a "splatting-free path-traced inverse-rendering framework for 3D
Gaussian fields that unifies forward rendering and backward optimization
within the same recursive multi-bounce light-transport pipeline," doing
genuine Monte Carlo path tracing in path space over Gaussian primitives,
with ray-traced visibility and global illumination jointly optimized against
materials and environment lighting under the full rendering equation. Its
stated target is more plausible shadows, reflections, and relighting under
global illumination — the abstract does not claim refraction or dielectric
transmission specifically, so treat it as evidence that full GI on Gaussian
scenes is now a going concern, not as a second refraction result on top of
3DGRT/3DGUT.

## Hair and thin structure

Directly relevant to the panel's fine-structure selling point, and
encouraging: **GaussianHair** (Luo et al., arXiv:2402.10483,
`docs/papers/2402.10483.pdf`) represents each hair strand as a sequence of
connected cylindrical 3D Gaussian primitives with a dedicated "GaussianHair
Scattering Model" for strand-level light interaction, reconstructed from
images and rendered via ordinary differentiable rasterization — i.e.,
Gaussians degrade gracefully down to sub-millimeter elongated structures,
not just soft blobs. **Gaussian Haircut** (Zakharov, Sklyarova, Black, Nam,
Thies, Hilliges, arXiv:2409.14778, `docs/papers/2409.14778.pdf`) pairs
classical 3D polyline hair strands with "strand-aligned 3D Gaussians" so the
strand prior and 3DGS's differentiable rendering reinforce each other.
HairGS (arXiv:2509.07774, already in round 1) does the same class of thing
with a merge-into-strands post-process. All three are reconstruction/
rendering-quality papers with no ray-tracing or hologram angle — they answer
"can Gaussians resolve strand-scale geometry" (yes) but say nothing about
"can that survive a CGH pipeline," which is still an open combination like
everything else in this file.

## Revised bottom line on refraction

For a **known-geometry** refractive object (our case — the glass sphere's
position, radius and IOR are part of the scene definition, not something to
be reconstructed), 3DGUT's hybrid splat-primary/trace-secondary pipeline is
a credible, demonstrated (in simulation, on real GPU hardware, not on an
optical bench) way to get physically correct Snell's-law refraction and
therefore correct internal parallax, at a speed (order 200+ FPS for the
Gaussian majority of the scene, ray-tracing only the rays that actually
enter the sphere) that would meaningfully help the "replace the path
tracer" half of the original question. It does **not** help the "skip the
light field, go straight to a coherent hologram" half — nobody has combined
ray-traced Gaussians with Gaussian Wave Splatting's Fourier-optics wavefront
pipeline, and GWS's own compositing (§2, §4 above) still has no ray-bending
at all. The two problems — fast correct-refraction rendering, and analytic
scene-to-hologram — remain solved separately, by different papers, with no
published bridge between them.

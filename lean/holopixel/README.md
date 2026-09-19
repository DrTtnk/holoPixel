# HoloPixel — Lean formalisation of the diagonal normal-operator algebra

This is a small Lean 4 / Mathlib project that machine-checks the linear
algebra behind one claim in the Python research code: for the multi-pupil
holographic forward operator `P_k = F ∘ D_k ∘ R_k`, the accumulated
least-squares normal operator `H = Σ_k ρ_k P_kᴴ P_k` is *exactly* diagonal,
which is why the global step of the solver can be an elementwise division
instead of a linear solve.

It is a proof artefact, not a library: everything the Python code needs is
proved once, concretely, over `Matrix (Fin _) (Fin _) ℂ`. The DFT itself is
**not** formalised — `F` is an arbitrary matrix satisfying `Fᴴ F = c • I`,
which is the only property of the DFT the argument uses.

## What is proved

All six theorems below are fully proved (no `sorry`, no `axiom`, no missing
cases). Each file's proof is self-contained; `Holopixel.lean` imports all of
them so `lake build` type-checks everything in one shot.

| File | Theorem | Statement |
|---|---|---|
| `Holopixel/TheoremA.lean` | `forward_normal_op` | `(F D R)ᴴ (F D R) = c • (Rᴴ Dᴴ D R)`, given `Fᴴ F = c • I`. |
| `Holopixel/TheoremB.lean` | `restriction_of_diag_is_diag` | For diagonal `D` and a coordinate-selection matrix `R`, `Rᴴ D R` is diagonal in the parent (panel) coordinates. |
| `Holopixel/TheoremC.lean` | `sum_diag_is_diag` | If every `H_k` is diagonal, `Σ_k ρ_k H_k` is diagonal. |
| `Holopixel/TheoremD.lean` | `gram_diag_zero_implies_column_zero` | If `H = Σ_k ρ_k P_kᴴ P_k` (`ρ_k ≥ 0`) has `H_jj = 0`, then `P_k *ᵥ e_j = 0` for every `k` with `ρ_k > 0`. |
| `Holopixel/TheoremE.lean` | `normal_equations_minimize` | A solution `u₀` of the normal equations `H u₀ = b` is a global minimiser of `E(u) = Σ_k ρ_k ‖P_k u − z_k‖²`. |
| `Holopixel/TheoremF.lean` | `diag_coordinate_unique_minimizer` | If `H = diag(d)` with `d_j > 0`, the coordinate-`j` objective is minimised uniquely at `u_j = b_j / d_j`. |

Chaining A → B → C is the actual structural argument for the numerical
result: A rewrites each pupil's normal operator as a scaled Gram matrix in
window coordinates, B shows that Gram matrix is diagonal once windowed back
into panel coordinates, and C shows the weighted sum over pupils stays
diagonal. D and E/F are the surrounding least-squares facts (a coordinate
that no active pupil ever illuminates is exactly the one where the "elementwise
division" trivially has nothing to divide, and the diagonal solve is honestly
the least-squares answer).

### Hypotheses that turned out to matter (read this before reusing these lemmas)

- **Theorem A needs only `Fᴴ F = c • I`.** It does *not* need `D` diagonal or
  `R` a selection matrix — those hypotheses are unused in this theorem and
  only become necessary for Theorem B. If you see Theorem A's conclusion
  quoted as if it depended on `D`/`R`'s special structure, that's not what is
  proved here.
- **Theorem B needs `σ` injective**, not just any coordinate map. This is
  the formal counterpart of "each window row reads a distinct panel pixel."
  Without injectivity the claim is false: two window rows selecting the same
  panel pixel produce a nonzero off-diagonal entry equal to `D` at that
  shared index. This is a real, non-obvious precondition on `R_k` that the
  Python code must also satisfy (it does, since `R_k` is built from a
  non-overlapping index gather), but it is worth stating explicitly since
  the original task description did not spell it out.
- **Theorem D does *not* need an extra positive-semidefiniteness hypothesis.**
  The task description flagged this as a possible extra requirement, but
  `(P_kᴴ P_k)_jj = ‖(P_k)_{·j}‖²` is automatically nonnegative for *any*
  matrix `P_k` — it is a Gram-matrix diagonal entry, not a property that
  needs assuming. The one hypothesis that *is* load-bearing (and is already
  in the task statement) is `ρ_k ≥ 0` for every `k`, not just the one whose
  conclusion you want: without it, a negative term elsewhere in the sum
  could cancel a nonzero `ρ_k ‖P_k e_j‖²` term and the implication would fail.
- **Theorem E is proved as a one-directional (sufficiency) statement**:
  solving the normal equations gives a global minimiser, for free real
  weights `ρ_k ≥ 0`, with *no* positive-definiteness assumption on `H`. The
  converse ("every minimiser solves the normal equations") is not proved
  here; it is not needed to justify why solving `Hu = b` is a correct
  least-squares step, which is the direction the solver actually uses. The
  proof is algebraic (a "complete the square" identity via `dotProduct` /
  `mulVec` / `star` algebra), not calculus — no differentiability or
  convexity machinery from Mathlib's analysis library is invoked.
- **Theorem F** is proved as the genuinely scalar statement it is: for
  `H = diag(d)` the objective decouples coordinate-by-coordinate, and each
  coordinate's 1-D quadratic `d|x|² − 2 Re(conj(b) x)` has the unique
  minimiser `x = b/d` for `d > 0`. It is proved directly (its own "complete
  the square" identity, `d · f(x) = ‖dx − b‖² − ‖b‖²`), not derived as a
  special case of Theorem E's machinery.

## Build

Requires a Lean 4 toolchain via `elan` (already pinned by `lean-toolchain`
to `leanprover/lean4:v4.30.0-rc1`, matching the Mathlib revision this project
depends on).

```bash
cd lean/holopixel
lake build
```

The first build fetches the Mathlib dependency and its prebuilt `.olean`
cache (`lake exe cache get`, run automatically by `lake new .. math` when
this project was created, and by `lake build` if the cache is missing) —
that step needs network access and takes on the order of two minutes.
After the cache is in place, `lake build` type-checks all six theorem files
in about 10–15 seconds on a modern CPU (measured: 11.6s wall clock for a
clean rebuild of just this project's own files, with the Mathlib cache
already present; no GPU is used anywhere in this project).

To rebuild from scratch, including a fresh Mathlib cache fetch:

```bash
cd lean/holopixel
lake exe cache get   # fetch prebuilt Mathlib .olean files (~2 min, needs network)
lake build            # type-check this project (~10-15s)
```

There is no test suite beyond the proofs themselves: a successful `lake
build` *is* the check. `grep -rn sorry Holopixel/` finds nothing, confirming
no theorem is stubbed out.

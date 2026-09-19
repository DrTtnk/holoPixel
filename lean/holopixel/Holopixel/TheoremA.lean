import Mathlib

/-!
# Theorem A — the normal operator of a windowed forward operator

The forward optical operator for one pupil position is `P = F * D * R`, where
`R` restricts a panel-sized vector to a window, `D` is a diagonal aperture on
that window, and `F` is an (unnormalised) DFT on the window satisfying
`Fᴴ * F = c • I`.

This theorem shows `Pᴴ * P = c • (Rᴴ * Dᴴ * D * R)`.

Note: the proof uses only `Fᴴ * F = c • I` together with associativity of
matrix multiplication and how `conjTranspose` distributes over products. It
does **not** need `D` to be diagonal or `R` to be a selection matrix — those
hypotheses only start to matter in Theorem B. -/

open Matrix

variable {N M K : ℕ}

/-- `(F * D * R)ᴴ * (F * D * R) = c • (Rᴴ * Dᴴ * D * R)`, given `Fᴴ * F = c • I`. -/
theorem forward_normal_op
    (F : Matrix (Fin K) (Fin M) ℂ) (D : Matrix (Fin M) (Fin M) ℂ)
    (R : Matrix (Fin M) (Fin N) ℂ) (c : ℂ)
    (hF : Fᴴ * F = c • (1 : Matrix (Fin M) (Fin M) ℂ)) :
    (F * D * R)ᴴ * (F * D * R) = c • (Rᴴ * Dᴴ * D * R) := by
  calc
    (F * D * R)ᴴ * (F * D * R)
        = Rᴴ * Dᴴ * Fᴴ * (F * D * R) := by
          simp [conjTranspose_mul, Matrix.mul_assoc]
    _ = Rᴴ * Dᴴ * (Fᴴ * F) * D * R := by
          simp [Matrix.mul_assoc]
    _ = Rᴴ * Dᴴ * (c • (1 : Matrix (Fin M) (Fin M) ℂ)) * D * R := by rw [hF]
    _ = c • (Rᴴ * Dᴴ * D * R) := by
          simp [Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_assoc]

import Mathlib

/-- Sanity check: the toolchain and the Mathlib dependency both work. -/
theorem sanity_check (n : ℕ) : n + 0 = n := by simp

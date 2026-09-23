"""Diffraction-aware phase-space budget for a tracked foveated light field."""
from __future__ import annotations
import numpy as np
from retina_model import square_equivalent_pitch_deg, density_per_deg2

def circular_incoherent_mtf(nu):
    nu = np.asarray(nu, dtype=float)
    out = np.zeros_like(nu)
    m = (nu >= 0.0) & (nu <= 1.0)
    z = nu[m]
    out[m] = (2.0 / np.pi) * (np.arccos(z) - z * np.sqrt(1.0 - z*z))
    return out

def mtf50_normalized_frequency():
    nu = np.linspace(0.0, 1.0, 500001)
    return float(nu[np.argmin(np.abs(circular_incoherent_mtf(nu) - 0.5))])

def local_k_max(pitch_deg, wavelength_m=550e-9, pupil_diameter_m=4e-3, k_cap=7):
    nu50 = mtf50_normalized_frequency()
    pitch_rad = np.deg2rad(pitch_deg)
    k = np.floor(2.0 * nu50 * pupil_diameter_m * pitch_rad / wavelength_m).astype(int)
    return np.clip(k, 0, k_cap)

def run(fov_x=70.0, fov_y=45.0, step=0.025):
    xs = np.arange(-fov_x/2 + step/2, fov_x/2, step)
    ys = np.arange(-fov_y/2 + step/2, fov_y/2, step)
    x, y = np.meshgrid(xs, ys)
    pitch = square_equivalent_pitch_deg(x, y)
    rho = density_per_deg2(x, y)
    n_field = float(np.sum(rho) * step**2)
    print(f"Retina-matched field samples: {n_field:,.0f}")
    for nm, lam in [(450,450e-9),(550,550e-9),(650,650e-9)]:
        k = local_k_max(pitch, wavelength_m=lam)
        k_use = np.clip(k, 1, 7)
        n4 = float(np.sum(rho * k_use**2) * step**2)
        print(f"{nm} nm: adaptive square-grid 4-D budget = {n4/1e6:.2f} M samples")

if __name__ == "__main__":
    run()
"""Watson-2014 retinal sampling ROM for the foveated light-field study."""
from __future__ import annotations
import numpy as np

PARAMS = {
    "temporal": (0.9851, 1.0580, 22.140),
    "superior": (0.9935, 1.0350, 16.350),
    "nasal": (0.9729, 1.0840, 7.633),
    "inferior": (0.9960, 0.9932, 12.130),
}
DC0 = 14804.6
RM = 41.03

def total_mrgc_density(ecc_deg, meridian):
    e = np.asarray(ecc_deg, dtype=float)
    a, r2, re = PARAMS[meridian]
    return 2.0 * DC0 * (1.0 + e / RM) ** -1.0 * (
        a * (1.0 + e / r2) ** -2.0 + (1.0 - a) * np.exp(-e / re)
    )

def one_mosaic_spacing_deg(ecc_deg, meridian):
    d = total_mrgc_density(ecc_deg, meridian) / 2.0
    return np.sqrt(2.0 / (np.sqrt(3.0) * d))

def spacing_xy_deg(x_deg, y_deg):
    x = np.asarray(x_deg, dtype=float)
    y = np.asarray(y_deg, dtype=float)
    r = np.hypot(x, y)
    rr = np.where(r > 0.0, r, 1.0)
    sh = np.where(x >= 0, one_mosaic_spacing_deg(r, "temporal"), one_mosaic_spacing_deg(r, "nasal"))
    sv = np.where(y >= 0, one_mosaic_spacing_deg(r, "superior"), one_mosaic_spacing_deg(r, "inferior"))
    s = np.sqrt((x / rr) ** 2 * sh**2 + (y / rr) ** 2 * sv**2)
    s0 = one_mosaic_spacing_deg(0.0, "temporal")
    return np.where(r > 0.0, s, s0)

def square_equivalent_pitch_deg(x_deg, y_deg):
    return np.sqrt(3.0) / 2.0 * spacing_xy_deg(x_deg, y_deg)

def density_per_deg2(x_deg, y_deg):
    p = square_equivalent_pitch_deg(x_deg, y_deg)
    return 1.0 / p**2

def integrate_field_samples(fov_x_deg=70.0, fov_y_deg=45.0, step_deg=0.025):
    xs = np.arange(-fov_x_deg/2 + step_deg/2, fov_x_deg/2, step_deg)
    ys = np.arange(-fov_y_deg/2 + step_deg/2, fov_y_deg/2, step_deg)
    x, y = np.meshgrid(xs, ys)
    return float(np.sum(density_per_deg2(x, y)) * step_deg**2)

if __name__ == "__main__":
    p0 = float(square_equivalent_pitch_deg(0.0, 0.0))
    n = integrate_field_samples()
    print(f"Foveal square-equivalent pitch: {p0*60:.3f} arcmin")
    print(f"Retina-matched samples over 70 x 45 deg: {n:,.0f}")
"""Scalar wave-optics stress test for a minimal microlens pupil encoder."""
from __future__ import annotations
import numpy as np

def pupil_field_square_aperture(x, y, clear_aperture_m, wavelength_m, z_m, x0=0.0, y0=0.0):
    return np.sinc(clear_aperture_m*(x-x0)/(wavelength_m*z_m)) * np.sinc(clear_aperture_m*(y-y0)/(wavelength_m*z_m))

def retinal_psf(clear_aperture_um=28.0, wavelength_nm=550.0, z_mm=20.0, pupil_diameter_mm=4.0, n=1024, window_mm=8.0):
    wavelength = wavelength_nm * 1e-9
    aperture = clear_aperture_um * 1e-6
    z = z_mm * 1e-3
    d = pupil_diameter_mm * 1e-3
    window = window_mm * 1e-3
    dx = window / n
    coord = (np.arange(n) - n/2) * dx
    x, y = np.meshgrid(coord, coord)
    pupil = (x*x + y*y) <= (d/2)**2
    field = pupil_field_square_aperture(x, y, aperture, wavelength, z) * pupil
    ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    psf = np.abs(ft)**2
    psf /= psf.sum()
    return coord, psf

if __name__ == "__main__":
    _, psf = retinal_psf()
    print("PSF generated:", psf.shape, "sum=", float(psf.sum()))
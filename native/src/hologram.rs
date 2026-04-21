use rustfft::{FftPlanner, num_complex::Complex64};
use rayon::prelude::*;
use napi_derive::napi;
use napi::bindgen_prelude::*;

use crate::scene::{self, Vec3};
use crate::renderer;

// ── Physical constants ────────────────────────────────────
const WAVELENGTH: f64 = 532e-9; // 532 nm green laser
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

// ── Display panel geometry ────────────────────────────────
// The holographic display sits at the opening of the Cornell Box (z=0),
// covering the full aperture. Each hogel is a small patch of the display.

const BOX_W: f32 = 552.8; // Cornell Box width (x)
const BOX_H: f32 = 548.8; // Cornell Box height (y)

// ── 2D FFT (row-column decomposition) ─────────────────────

fn fft2(data: &mut [Complex64], width: usize, height: usize) {
    let mut planner = FftPlanner::new();

    let fft_row = planner.plan_fft_forward(width);
    for row in data.chunks_exact_mut(width) {
        fft_row.process(row);
    }

    let fft_col = planner.plan_fft_forward(height);
    let mut col = vec![Complex64::new(0.0, 0.0); height];
    for x in 0..width {
        for y in 0..height {
            col[y] = data[y * width + x];
        }
        fft_col.process(&mut col);
        for y in 0..height {
            data[y * width + x] = col[y];
        }
    }
}

fn ifft2(data: &mut [Complex64], width: usize, height: usize) {
    let mut planner = FftPlanner::new();
    let norm = 1.0 / (width * height) as f64;

    let ifft_row = planner.plan_fft_inverse(width);
    for row in data.chunks_exact_mut(width) {
        ifft_row.process(row);
    }

    let ifft_col = planner.plan_fft_inverse(height);
    let mut col = vec![Complex64::new(0.0, 0.0); height];
    for x in 0..width {
        for y in 0..height {
            col[y] = data[y * width + x];
        }
        ifft_col.process(&mut col);
        for y in 0..height {
            data[y * width + x] = col[y];
        }
    }

    for v in data.iter_mut() {
        *v *= norm;
    }
}

// ── Hogel computation ─────────────────────────────────────

/// Compute a single hogel's hemisphere rendering and fringe pattern.
///
/// The hogel sits at position (hogel_x, hogel_y) on the display grid.
/// We render a 180° fisheye hemisphere looking INTO the Cornell Box (+z),
/// then IFFT the angular spectrum to get the diffraction fringe pattern.
pub fn compute_hogel(
    hogel_x: u32,
    hogel_y: u32,
    grid_w: u32,
    grid_h: u32,
    hemi_res: u32,
    spp: u32,
) -> HogelData {
    let scene = scene::build_cornell_box();

    // Hogel center position on the display panel (z=0 plane)
    let cell_w = BOX_W / grid_w as f32;
    let cell_h = BOX_H / grid_h as f32;
    let origin = Vec3::new(
        (hogel_x as f32 + 0.5) * cell_w,
        (hogel_y as f32 + 0.5) * cell_h,
        0.0, // display panel at z=0
    );

    // Hemisphere faces INTO the box (+z direction)
    let forward = Vec3::new(0.0, 0.0, 1.0);
    let up = Vec3::new(0.0, 1.0, 0.0);

    // Render the hemisphere (fisheye lightfield)
    let hemisphere_rgba = renderer::render_hemisphere(
        &scene, origin, forward, up, hemi_res, spp,
    );

    // Convert hemisphere to complex angular spectrum
    // Each pixel → luminance → amplitude, zero initial phase
    let n = hemi_res as usize;
    let mut angular_spectrum: Vec<Complex64> = hemisphere_rgba
        .chunks_exact(4)
        .map(|px| {
            let r = px[0] as f64 / 255.0;
            let g = px[1] as f64 / 255.0;
            let b = px[2] as f64 / 255.0;
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;
            Complex64::new(lum.sqrt(), 0.0)
        })
        .collect();

    // IFFT to get the fringe pattern (hologram sub-aperture)
    ifft2(&mut angular_spectrum, n, n);

    // Extract fringe phase for visualization
    let fringe_rgba = phase_to_rgba(&angular_spectrum);

    HogelData {
        hemi_res,
        hemisphere_rgba,
        fringe_rgba,
        complex_field: angular_spectrum,
    }
}

/// Compute the full hologram by assembling all hogels' fringe patterns.
/// Returns the assembled hologram phase + lightfield reconstruction.
pub fn compute_full_hologram(
    grid_w: u32,
    grid_h: u32,
    hemi_res: u32,
    spp: u32,
) -> FullHologramResult {
    let scene = scene::build_cornell_box();
    let n = hemi_res as usize;

    let full_w = grid_w as usize * n;
    let full_h = grid_h as usize * n;
    let mut full_hologram = vec![Complex64::new(0.0, 0.0); full_w * full_h];

    // Store all hemispheres for lightfield reconstruction
    let mut hemispheres: Vec<Vec<u8>> = Vec::with_capacity((grid_w * grid_h) as usize);
    let mut last_hemi: Vec<u8> = Vec::new();
    let mut last_fringe: Vec<u8> = Vec::new();

    let cell_w = BOX_W / grid_w as f32;
    let cell_h = BOX_H / grid_h as f32;
    let forward = Vec3::new(0.0, 0.0, 1.0);
    let up = Vec3::new(0.0, 1.0, 0.0);

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let origin = Vec3::new(
                (gx as f32 + 0.5) * cell_w,
                (gy as f32 + 0.5) * cell_h,
                0.0,
            );

            let hemisphere_rgba = renderer::render_hemisphere(
                &scene, origin, forward, up, hemi_res, spp,
            );

            // Angular spectrum → IFFT → fringe
            let mut angular_spectrum: Vec<Complex64> = hemisphere_rgba
                .chunks_exact(4)
                .map(|px| {
                    let r = px[0] as f64 / 255.0;
                    let g = px[1] as f64 / 255.0;
                    let b = px[2] as f64 / 255.0;
                    let lum = 0.299 * r + 0.587 * g + 0.114 * b;
                    Complex64::new(lum.sqrt(), 0.0)
                })
                .collect();

            ifft2(&mut angular_spectrum, n, n);

            // Place sub-hologram into the full hologram mosaic
            let ox = gx as usize * n;
            let oy = gy as usize * n;
            for ly in 0..n {
                for lx in 0..n {
                    full_hologram[(oy + ly) * full_w + (ox + lx)] = angular_spectrum[ly * n + lx];
                }
            }

            // Save hemisphere for reconstruction and last for diagnostics
            last_hemi = hemisphere_rgba.clone();
            last_fringe = phase_to_rgba(&angular_spectrum);
            hemispheres.push(hemisphere_rgba);
        }
    }

    let hologram_rgba = phase_to_rgba(&full_hologram);

    // Lightfield reconstruction: simulate what a viewer sees looking at the display
    let recon_rgba = reconstruct_lightfield(
        &hemispheres, grid_w, grid_h, hemi_res,
        full_w as u32, full_h as u32,
    );

    FullHologramResult {
        grid_w,
        grid_h,
        hemi_res,
        full_width: full_w as u32,
        full_height: full_h as u32,
        hologram_rgba,
        recon_rgba,
        last_hogel_hemi: last_hemi,
        last_hogel_fringe: last_fringe,
    }
}

// ── Lightfield Reconstruction ─────────────────────────────

/// Simulate what an observer sees looking at the holographic display.
/// For each output pixel, find which hogel is hit, sample its hemisphere
/// at the viewing angle determined by the observer-to-hogel direction.
fn reconstruct_lightfield(
    hemispheres: &[Vec<u8>],
    grid_w: u32,
    grid_h: u32,
    hemi_res: u32,
    output_w: u32,
    output_h: u32,
) -> Vec<u8> {
    // Observer position (same as path tracer camera)
    let eye = Vec3::new(278.0, 273.0, -800.0);
    let cell_w = BOX_W / grid_w as f32;
    let cell_h = BOX_H / grid_h as f32;
    let half_pi = std::f32::consts::FRAC_PI_2;
    let hr = hemi_res as f32;

    let pixels: Vec<[u8; 4]> = (0..output_w * output_h)
        .into_par_iter()
        .map(|idx| {
            let px = idx % output_w;
            let py = idx / output_w;

            // Map output pixel to position on the display panel [0, BOX_W] × [0, BOX_H]
            let panel_x = (px as f32 + 0.5) / output_w as f32 * BOX_W;
            let panel_y = (1.0 - (py as f32 + 0.5) / output_h as f32) * BOX_H; // flip Y

            // Which hogel does this hit?
            let gx = ((panel_x / cell_w) as u32).min(grid_w - 1);
            let gy = ((panel_y / cell_h) as u32).min(grid_h - 1);

            // Direction from observer through this panel point INTO the scene
            // observer → panel_point → continues into scene at +z
            let dx = panel_x - eye.x;
            let dy = panel_y - eye.y;
            let dz = 0.0 - eye.z; // = 800 (positive, into the scene)
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            // In the hogel's frame (forward=+z, right=+x, up=+y):
            let d_fwd = dz / dist; // positive
            let d_right = dx / dist;
            let d_up = dy / dist;

            // Equidistant fisheye: polar angle from forward axis
            let theta = d_fwd.clamp(-1.0, 1.0).acos();
            let r_fish = theta / half_pi;

            if r_fish > 1.0 {
                return [0, 0, 0, 255];
            }

            // Azimuthal angle
            let sin_theta = theta.sin();
            let (phi_cos, phi_sin) = if sin_theta > 1e-6 {
                (d_right / sin_theta, d_up / sin_theta)
            } else {
                (0.0, 0.0)
            };

            // Map to hemisphere pixel coordinates
            let fu = (phi_cos * r_fish * 0.5 + 0.5) * hr;
            let fv = (phi_sin * r_fish * 0.5 + 0.5) * hr;
            let hu = (fu as u32).min(hemi_res - 1);
            let hv = (fv as u32).min(hemi_res - 1);

            // Look up the hogel's hemisphere
            let hogel_idx = (gy * grid_w + gx) as usize;
            let hemi = &hemispheres[hogel_idx];
            let pidx = ((hv * hemi_res + hu) * 4) as usize;
            [hemi[pidx], hemi[pidx + 1], hemi[pidx + 2], 255]
        })
        .collect();

    pixels.into_iter().flat_map(|p| p).collect()
}

// ── Helpers ───────────────────────────────────────────────

fn phase_to_rgba(field: &[Complex64]) -> Vec<u8> {
    field
        .iter()
        .flat_map(|c| {
            let phase = c.arg(); // -π to π
            let normalized = (phase + std::f64::consts::PI) / TWO_PI;
            let v = (normalized * 255.0) as u8;
            [v, v, v, 255]
        })
        .collect()
}

// ── Data structures ───────────────────────────────────────

pub struct HogelData {
    pub hemi_res: u32,
    pub hemisphere_rgba: Vec<u8>,
    pub fringe_rgba: Vec<u8>,
    pub complex_field: Vec<Complex64>,
}

pub struct FullHologramResult {
    pub grid_w: u32,
    pub grid_h: u32,
    pub hemi_res: u32,
    pub full_width: u32,
    pub full_height: u32,
    pub hologram_rgba: Vec<u8>,
    pub recon_rgba: Vec<u8>,
    pub last_hogel_hemi: Vec<u8>,
    pub last_hogel_fringe: Vec<u8>,
}

// ── napi exports ──────────────────────────────────────────

#[napi(object)]
pub struct HogelResult {
    pub hemi_res: u32,
    pub hemisphere_data: Buffer,
    pub fringe_data: Buffer,
}

/// Compute a single hogel's hemisphere + fringe pattern.
#[napi]
pub fn compute_single_hogel(
    hogel_x: u32,
    hogel_y: u32,
    grid_w: u32,
    grid_h: u32,
    hemi_res: u32,
    spp: u32,
) -> HogelResult {
    let data = compute_hogel(hogel_x, hogel_y, grid_w, grid_h, hemi_res, spp);
    HogelResult {
        hemi_res: data.hemi_res,
        hemisphere_data: data.hemisphere_rgba.into(),
        fringe_data: data.fringe_rgba.into(),
    }
}

#[napi(object)]
pub struct FullHologramNapiResult {
    pub grid_w: u32,
    pub grid_h: u32,
    pub hemi_res: u32,
    pub full_width: u32,
    pub full_height: u32,
    pub hologram_data: Buffer,
    pub recon_data: Buffer,
    pub last_hogel_hemi: Buffer,
    pub last_hogel_fringe: Buffer,
}

/// Compute the full assembled hologram from all hogels.
#[napi]
pub fn compute_hologram_full(
    grid_w: u32,
    grid_h: u32,
    hemi_res: u32,
    spp: u32,
) -> FullHologramNapiResult {
    let result = compute_full_hologram(grid_w, grid_h, hemi_res, spp);
    FullHologramNapiResult {
        grid_w: result.grid_w,
        grid_h: result.grid_h,
        hemi_res: result.hemi_res,
        full_width: result.full_width,
        full_height: result.full_height,
        hologram_data: result.hologram_rgba.into(),
        recon_data: result.recon_rgba.into(),
        last_hogel_hemi: result.last_hogel_hemi.into(),
        last_hogel_fringe: result.last_hogel_fringe.into(),
    }
}

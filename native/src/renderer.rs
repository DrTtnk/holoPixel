use rayon::prelude::*;
use napi_derive::napi;
use napi::bindgen_prelude::*;

use crate::scene::{self, Vec3, Ray, Scene, MaterialKind};

// ── Light geometry for NEE (matches build_cornell_box) ────
const LIGHT_X0: f32 = 213.0;
const LIGHT_X1: f32 = 343.0;
const LIGHT_Z0: f32 = 227.0;
const LIGHT_Z1: f32 = 332.0;
const LIGHT_Y: f32 = 548.8 - 0.01;
const LIGHT_AREA: f32 = (LIGHT_X1 - LIGHT_X0) * (LIGHT_Z1 - LIGHT_Z0);
const LIGHT_EMISSION: Vec3 = Vec3::new(40.0, 40.0, 40.0);

// ── Random number generator (xorshift64) ──────────────────
// Lightweight per-thread RNG — no allocation, deterministic per seed.

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1)) // avoid zero seed
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }

    /// Cosine-weighted hemisphere sample (Malley's method)
    fn hemisphere_cosine(&mut self, normal: Vec3) -> Vec3 {
        let r1 = self.next_f32();
        let r2 = self.next_f32();
        let sin_theta = r2.sqrt();
        let cos_theta = (1.0 - r2).sqrt();
        let phi = 2.0 * std::f32::consts::PI * r1;

        // Build orthonormal basis from normal
        let w = normal;
        let a = if w.x.abs() > 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
        let u = w.cross(a).normalize();
        let v = w.cross(u);

        // Transform sample to world space
        u.scale(phi.cos() * sin_theta) + v.scale(phi.sin() * sin_theta) + w.scale(cos_theta)
    }
}

// ── Path Tracer with Next-Event Estimation ────────────────

fn trace_path(scene: &Scene, ray: &Ray, rng: &mut Rng, max_bounces: u32) -> Vec3 {
    let mut throughput = Vec3::new(1.0, 1.0, 1.0);
    let mut radiance = Vec3::new(0.0, 0.0, 0.0);
    let mut current_ray = Ray { origin: ray.origin, dir: ray.dir };

    for _bounce in 0..max_bounces {
        let hit = match scene.trace(&current_ray) {
            Some(h) => h,
            None => break, // escaped scene
        };

        let mat = &scene.materials[hit.material_idx];

        // Add emission (only on first bounce, or specular; for diffuse NEE handles it)
        if _bounce == 0 {
            radiance = radiance + Vec3::new(
                throughput.x * mat.emission.x,
                throughput.y * mat.emission.y,
                throughput.z * mat.emission.z,
            );
        }

        match mat.kind {
            MaterialKind::Emissive => break, // light source — stop bouncing
            MaterialKind::Diffuse => {
                // NEE: sample a point on the area light
                let light_x = LIGHT_X0 + rng.next_f32() * (LIGHT_X1 - LIGHT_X0);
                let light_z = LIGHT_Z0 + rng.next_f32() * (LIGHT_Z1 - LIGHT_Z0);
                let light_pos = Vec3::new(light_x, LIGHT_Y, light_z);

                let to_light = light_pos.sub(hit.point);
                let dist2 = to_light.dot(to_light);
                let dist = dist2.sqrt();
                let light_dir = to_light.scale(1.0 / dist);

                let cos_hit = hit.normal.dot(light_dir).max(0.0);
                let cos_light = light_dir.y.max(0.0); // LIGHT_NORMAL is (0,-1,0), dot(-light_dir) = light_dir.y

                if cos_hit > 0.0 && cos_light > 0.0 {
                    // Shadow ray
                    let shadow_ray = Ray {
                        origin: hit.point + hit.normal.scale(1e-3),
                        dir: light_dir,
                    };
                    let occluded = if let Some(shadow_hit) = scene.trace(&shadow_ray) {
                        shadow_hit.t < dist - 2e-3
                    } else {
                        true // shouldn't happen inside box
                    };

                    if !occluded {
                        // Direct illumination: Le * brdf * cos_hit * cos_light * A / dist²
                        // BRDF for Lambertian = albedo / π
                        let geom = cos_hit * cos_light * LIGHT_AREA / dist2;
                        let inv_pi = 1.0 / std::f32::consts::PI;
                        radiance = radiance + Vec3::new(
                            throughput.x * mat.albedo.x * inv_pi * LIGHT_EMISSION.x * geom,
                            throughput.y * mat.albedo.y * inv_pi * LIGHT_EMISSION.y * geom,
                            throughput.z * mat.albedo.z * inv_pi * LIGHT_EMISSION.z * geom,
                        );
                    }
                }

                // Attenuate throughput by albedo
                throughput = Vec3::new(
                    throughput.x * mat.albedo.x,
                    throughput.y * mat.albedo.y,
                    throughput.z * mat.albedo.z,
                );

                // Russian roulette (after 3 bounces)
                if _bounce > 2 {
                    let p = throughput.x.max(throughput.y).max(throughput.z);
                    if rng.next_f32() > p { break; }
                    throughput = throughput.scale(1.0 / p);
                }

                // Cosine-weighted bounce
                let new_dir = rng.hemisphere_cosine(hit.normal);
                current_ray = Ray {
                    origin: hit.point + hit.normal.scale(1e-3),
                    dir: new_dir.normalize(),
                };
            }
        }
    }

    radiance
}

// ── Render Function ───────────────────────────────────────

pub fn render_cornell_box(width: u32, height: u32, samples_per_pixel: u32) -> Vec<u8> {
    let scene = scene::build_cornell_box();

    // Camera setup: standard Cornell Box camera
    let eye = Vec3::new(278.0, 273.0, -800.0);
    let look_at = Vec3::new(278.0, 273.0, 0.0);
    let up = Vec3::new(0.0, 1.0, 0.0);
    let fov_deg = 39.3_f32; // standard Cornell Box FOV

    let forward = look_at.sub(eye).normalize();
    let right = up.cross(forward).normalize();
    let cam_up = forward.cross(right);

    let aspect = width as f32 / height as f32;
    let half_h = (fov_deg.to_radians() / 2.0).tan();
    let half_w = half_h * aspect;

    let max_bounces = 8;

    // Parallel render: each pixel is independent
    let pixels: Vec<[u8; 4]> = (0..width * height)
        .into_par_iter()
        .map(|idx| {
            let px = idx % width;
            let py = idx / width;
            let mut rng = Rng::new((idx as u64) * 6364136223846793005 + 1442695040888963407);

            let mut color = Vec3::new(0.0, 0.0, 0.0);

            for _s in 0..samples_per_pixel {
                // Jittered sample within pixel
                let u = (px as f32 + rng.next_f32()) / width as f32;
                let v = (py as f32 + rng.next_f32()) / height as f32;

                // Map to [-1, 1] with Y flipped (screen Y=0 is top)
                let sx = (2.0 * u - 1.0) * half_w;
                let sy = (1.0 - 2.0 * v) * half_h;

                let dir = (forward + right.scale(sx) + cam_up.scale(sy)).normalize();
                let ray = Ray { origin: eye, dir };

                color = color + trace_path(&scene, &ray, &mut rng, max_bounces);
            }

            // Average and tonemap (exposure + gamma 2.0)
            let inv_spp = 1.0 / samples_per_pixel as f32;
            let exposure = 2.0_f32; // exposure boost
            let r = (color.x * inv_spp * exposure).sqrt().clamp(0.0, 1.0);
            let g = (color.y * inv_spp * exposure).sqrt().clamp(0.0, 1.0);
            let b = (color.z * inv_spp * exposure).sqrt().clamp(0.0, 1.0);

            [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]
        })
        .collect();

    // Flatten to RGBA bytes
    pixels.into_iter().flat_map(|p| p).collect()
}

// ── Hemisphere (Fisheye) Renderer ─────────────────────────

/// Render a 180° equidistant fisheye hemisphere from a given position/direction.
/// The resulting image is a circle inscribed in a square: pixels outside the circle are black.
///
/// - `origin`: position of the hogel on the display panel
/// - `forward`: direction into the scene (hemisphere optical axis)
/// - `up`: up vector for orientation
/// - `resolution`: width = height of the square output image
/// - `spp`: samples per pixel for path tracing
pub fn render_hemisphere(
    scene: &Scene,
    origin: Vec3,
    forward: Vec3,
    up: Vec3,
    resolution: u32,
    spp: u32,
) -> Vec<u8> {
    let forward = forward.normalize();
    let right = up.cross(forward).normalize();
    let cam_up = forward.cross(right);

    let max_bounces = 6;
    let half_pi = std::f32::consts::FRAC_PI_2;
    let res = resolution as f32;

    let pixels: Vec<[u8; 4]> = (0..resolution * resolution)
        .into_par_iter()
        .map(|idx| {
            let px = idx % resolution;
            let py = idx / resolution;
            let mut rng = Rng::new((idx as u64).wrapping_mul(6364136223846793005).wrapping_add(1));

            // Normalized coordinates in [-1, 1]
            let u_center = (2.0 * (px as f32 + 0.5) / res) - 1.0;
            let v_center = (2.0 * (py as f32 + 0.5) / res) - 1.0;
            let r = (u_center * u_center + v_center * v_center).sqrt();

            // Outside the inscribed circle → black
            if r > 1.0 {
                return [0, 0, 0, 255];
            }

            let mut color = Vec3::new(0.0, 0.0, 0.0);

            for _s in 0..spp {
                // Jitter within pixel
                let u = (2.0 * (px as f32 + rng.next_f32()) / res) - 1.0;
                let v = (2.0 * (py as f32 + rng.next_f32()) / res) - 1.0;
                let r = (u * u + v * v).sqrt();
                if r > 1.0 {
                    continue;
                }

                // Equidistant fisheye: r maps linearly to polar angle [0, π/2]
                let theta = r * half_pi;
                let (sin_t, cos_t) = theta.sin_cos();

                let dir = if r > 1e-6 {
                    let phi_cos = u / r;
                    let phi_sin = v / r;
                    // Direction in local frame → world
                    let d = right.scale(phi_cos * sin_t)
                        + cam_up.scale(phi_sin * sin_t)
                        + forward.scale(cos_t);
                    d.normalize()
                } else {
                    forward
                };

                let ray = Ray { origin, dir };
                color = color + trace_path(scene, &ray, &mut rng, max_bounces);
            }

            // Tonemap (same as main renderer: exposure + sqrt gamma)
            let inv_spp = 1.0 / spp as f32;
            let exposure = 2.0_f32;
            let r = (color.x * inv_spp * exposure).sqrt().clamp(0.0, 1.0);
            let g = (color.y * inv_spp * exposure).sqrt().clamp(0.0, 1.0);
            let b = (color.z * inv_spp * exposure).sqrt().clamp(0.0, 1.0);

            [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]
        })
        .collect();

    pixels.into_iter().flat_map(|p| p).collect()
}

// ── napi export ───────────────────────────────────────────

#[napi(object)]
pub struct RenderResult {
    pub width: u32,
    pub height: u32,
    pub data: Buffer,
    pub samples_per_pixel: u32,
}

#[napi]
pub fn render_scene(width: u32, height: u32, spp: u32) -> RenderResult {
    let data = render_cornell_box(width, height, spp);
    RenderResult {
        width,
        height,
        data: data.into(),
        samples_per_pixel: spp,
    }
}

// CUDA hemisphere ray tracing kernel for holographic display simulator
// Compiled via nvrtc at runtime with --gpu-architecture=compute_80 --use_fast_math
// Scene geometry is injected as __constant__ memory by the host at compile time:
//   __constant__ float c_tris[NUM_TRIS * 13];
//   __constant__ float c_mats[NUM_MATS * 7];
//   #define NUM_TRIS <n>
//   #define NUM_MATS <n>

extern "C" {

// ── Data structures (match Rust scene.rs) ─────────────────

struct Vec3 {
    float x, y, z;
};

__device__ Vec3 v3(float x, float y, float z) {
    Vec3 v; v.x = x; v.y = y; v.z = z; return v;
}

__device__ Vec3 v3_add(Vec3 a, Vec3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ Vec3 v3_sub(Vec3 a, Vec3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ Vec3 v3_scale(Vec3 a, float s) { return v3(a.x*s, a.y*s, a.z*s); }
__device__ float v3_dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ Vec3 v3_cross(Vec3 a, Vec3 b) {
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__device__ float v3_len(Vec3 a) { return sqrtf(v3_dot(a, a)); }
__device__ Vec3 v3_norm(Vec3 a) {
    float l = v3_len(a);
    return l > 1e-12f ? v3_scale(a, 1.0f/l) : a;
}
__device__ Vec3 v3_mul(Vec3 a, Vec3 b) { return v3(a.x*b.x, a.y*b.y, a.z*b.z); }

struct Triangle {
    Vec3 v0, v1, v2, normal;
    int material_idx;
};

// material_kind: 0=Diffuse, 1=Emissive
struct Material {
    Vec3 albedo;
    Vec3 emission;
    int kind;
};

struct Ray {
    Vec3 origin, dir;
};

struct Hit {
    float t;
    Vec3 point, normal;
    int material_idx;
};

// ── Scene data (uploaded as kernel params) ────────────────
// Triangle data: v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z, nx,ny,nz, mat_idx (13 floats packed)
// Material data: albedo.xyz, emission.xyz, kind (7 floats packed)

// ── Light constants (Cornell Box ceiling light) ───────────
#define LIGHT_X0 213.0f
#define LIGHT_X1 343.0f
#define LIGHT_Z0 227.0f
#define LIGHT_Z1 332.0f
#define LIGHT_Y  548.79f
#define LIGHT_AREA ((LIGHT_X1-LIGHT_X0)*(LIGHT_Z1-LIGHT_Z0))
#define LIGHT_EMISSION_R 40.0f
#define LIGHT_EMISSION_G 40.0f
#define LIGHT_EMISSION_B 40.0f

// ── RNG (xorshift64) ─────────────────────────────────────

__device__ unsigned long long xorshift64(unsigned long long* state) {
    unsigned long long x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

__device__ float rng_f32(unsigned long long* state) {
    return (float)(xorshift64(state) & 0xFFFFFF) / (float)0xFFFFFF;
}

// ── Cosine-weighted hemisphere sample ─────────────────────

__device__ Vec3 hemisphere_cosine(unsigned long long* rng, Vec3 normal) {
    float r1 = rng_f32(rng);
    float r2 = rng_f32(rng);
    float sin_theta = sqrtf(r2);
    float cos_theta = sqrtf(1.0f - r2);
    float phi = 2.0f * 3.14159265f * r1;

    // Build orthonormal basis from normal
    Vec3 w = normal;
    Vec3 a = fabsf(w.x) > 0.9f ? v3(0,1,0) : v3(1,0,0);
    Vec3 u = v3_norm(v3_cross(w, a));
    Vec3 vv = v3_cross(w, u);

    return v3_add(v3_add(v3_scale(u, cosf(phi)*sin_theta),
                         v3_scale(vv, sinf(phi)*sin_theta)),
                  v3_scale(w, cos_theta));
}

// ── Ray-Triangle Intersection (Möller-Trumbore, precomputed edges) ──

__device__ float intersect_tri_e(Ray ray, Vec3 v0, Vec3 edge1, Vec3 edge2) {
    Vec3 h = v3_cross(ray.dir, edge2);
    float a = v3_dot(edge1, h);
    if (fabsf(a) < 1e-7f) return -1.0f;
    float f = 1.0f / a;
    Vec3 s = v3_sub(ray.origin, v0);
    float u = f * v3_dot(s, h);
    if (u < 0.0f || u > 1.0f) return -1.0f;
    Vec3 q = v3_cross(s, edge1);
    float v = f * v3_dot(ray.dir, q);
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    float t = f * v3_dot(edge2, q);
    return t > 1e-4f ? t : -1.0f;
}

// ── Scene Trace (brute force over constant-memory triangles) ─

__device__ bool trace_scene(Ray ray, Hit* hit) {
    float best_t = 1e30f;
    bool found = false;

    #pragma unroll 1
    for (int i = 0; i < NUM_TRIS; i++) {
        const float* tp = c_tris + i * 12;
        Vec3 v0 = v3(tp[0], tp[1], tp[2]);
        Vec3 e1 = v3(tp[3], tp[4], tp[5]);
        Vec3 e2 = v3(tp[6], tp[7], tp[8]);

        float t = intersect_tri_e(ray, v0, e1, e2);
        if (t > 0.0f && t < best_t) {
            best_t = t;
            Vec3 n = v3(tp[9], tp[10], tp[11]);
            int mat_idx = c_tri_mat[i];

            Vec3 point = v3_add(ray.origin, v3_scale(ray.dir, t));
            if (v3_dot(n, ray.dir) > 0.0f) n = v3_scale(n, -1.0f);

            hit->t = t;
            hit->point = point;
            hit->normal = n;
            hit->material_idx = mat_idx;
            found = true;
        }
    }
    return found;
}

// Shadow ray: any-hit early-out
__device__ bool shadow_ray(Ray ray, float max_dist) {
    #pragma unroll 1
    for (int i = 0; i < NUM_TRIS; i++) {
        const float* tp = c_tris + i * 12;
        Vec3 v0 = v3(tp[0], tp[1], tp[2]);
        Vec3 e1 = v3(tp[3], tp[4], tp[5]);
        Vec3 e2 = v3(tp[6], tp[7], tp[8]);
        float t = intersect_tri_e(ray, v0, e1, e2);
        if (t > 0.0f && t < max_dist - 2e-3f) return true;
    }
    return false;
}

// ── Path Trace with NEE ───────────────────────────────────

__device__ Vec3 trace_path(Ray ray, unsigned long long* rng, int max_bounces, float ambient) {
    Vec3 throughput = v3(1,1,1);
    Vec3 radiance = v3(0,0,0);
    Ray current = ray;

    for (int bounce = 0; bounce <= max_bounces; bounce++) {
        Hit hit;
        if (!trace_scene(current, &hit)) break;

        const float* mp = c_mats + hit.material_idx * 6;
        Vec3 albedo = v3(mp[0], mp[1], mp[2]);
        Vec3 emission = v3(mp[3], mp[4], mp[5]);
        int kind = c_mat_kind[hit.material_idx];

        // Emission on first bounce only (NEE handles subsequent)
        if (bounce == 0) {
            radiance = v3_add(radiance, v3_mul(throughput, emission));
        }

        if (kind == 1) break; // emissive — stop

        // Ambient term: uniform hemisphere illumination (cheap indirect proxy)
        // radiance += throughput * albedo * ambient
        radiance = v3_add(radiance, v3(
            throughput.x * albedo.x * ambient,
            throughput.y * albedo.y * ambient,
            throughput.z * albedo.z * ambient
        ));

        // NEE: sample area light
        float lx = LIGHT_X0 + rng_f32(rng) * (LIGHT_X1 - LIGHT_X0);
        float lz = LIGHT_Z0 + rng_f32(rng) * (LIGHT_Z1 - LIGHT_Z0);
        Vec3 light_pos = v3(lx, LIGHT_Y, lz);
        Vec3 to_light = v3_sub(light_pos, hit.point);
        float dist2 = v3_dot(to_light, to_light);
        float inv_dist = rsqrtf(dist2);
        float dist = dist2 * inv_dist;
        Vec3 light_dir = v3_scale(to_light, inv_dist);

        float cos_hit = fmaxf(v3_dot(hit.normal, light_dir), 0.0f);
        float cos_light = fmaxf(light_dir.y, 0.0f);

        if (cos_hit > 0.0f && cos_light > 0.0f) {
            Ray shadow;
            shadow.origin = v3_add(hit.point, v3_scale(hit.normal, 1e-3f));
            shadow.dir = light_dir;
            if (!shadow_ray(shadow, dist)) {
                float geom = cos_hit * cos_light * LIGHT_AREA / dist2;
                float inv_pi = 0.31830988618f;
                radiance = v3_add(radiance, v3(
                    throughput.x * albedo.x * inv_pi * LIGHT_EMISSION_R * geom,
                    throughput.y * albedo.y * inv_pi * LIGHT_EMISSION_G * geom,
                    throughput.z * albedo.z * inv_pi * LIGHT_EMISSION_B * geom
                ));
            }
        }

        // Stop after direct lighting if no indirect bounces requested
        if (bounce >= max_bounces) break;

        // Attenuate throughput for next bounce
        throughput = v3_mul(throughput, albedo);

        // Russian roulette
        if (bounce >= 1) {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (p < 1e-4f) break;
            if (p < 1.0f) {
                if (rng_f32(rng) > p) break;
                throughput = v3_scale(throughput, 1.0f / p);
            }
        }

        // Cosine-weighted bounce
        Vec3 new_dir = hemisphere_cosine(rng, hit.normal);
        current.origin = v3_add(hit.point, v3_scale(hit.normal, 1e-3f));
        current.dir = v3_norm(new_dir);
    }

    return radiance;
}

// ── Main Kernel: Render All Hogels ────────────────────────
// One thread per (hogel, pixel, sample).
// Grid: (num_hogels, ceil(res*res/BLOCK_SIZE), 1)
// Each thread computes one sample for one pixel of one hogel.

__global__ void render_hemispheres(
    float* __restrict__ output,       // [num_hogels * res * res * 3] RGB float
    int grid_w, int grid_h,
    int res,  // hemisphere resolution
    int spp,  // samples per pixel
    float box_w, float box_h,
    int hogel_offset,  // global hogel index of blockIdx.x==0
    int max_bounces,   // 0 = direct light only + ambient, 1+ = add indirect bounces
    float ambient     // uniform ambient term (pre-multiplied by 1/π is NOT needed here)
) {
    int hogel_idx = hogel_offset + blockIdx.x;
    int pixel_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int total_pixels = res * res;
    if (pixel_idx >= total_pixels) return;
    if (hogel_idx >= grid_w * grid_h) return;

    int gx = hogel_idx % grid_w;
    int gy = hogel_idx / grid_w;
    int px = pixel_idx % res;
    int py = pixel_idx / res;

    // Hogel center position on the display panel
    float cell_w = box_w / (float)grid_w;
    float cell_h = box_h / (float)grid_h;
    Vec3 origin = v3(((float)gx + 0.5f) * cell_w, ((float)gy + 0.5f) * cell_h, 0.0f);

    // Camera frame: forward=+z, up=+y → right = up×forward = +x
    // (matches render_hemisphere in renderer.rs)
    Vec3 forward = v3(0, 0, 1);
    Vec3 right = v3(1, 0, 0);   // up.cross(forward) = (0,1,0)×(0,0,1) = (1,0,0)
    Vec3 cam_up = v3(0, 1, 0);  // forward.cross(right) = (0,0,1)×(1,0,0) = (0,1,0)

    float half_pi = 1.5707963f;
    float fres = (float)res;

    // RNG seed: unique per hogel + pixel
    unsigned long long rng_state = (unsigned long long)hogel_idx * 6364136223846793005ULL
                                 + (unsigned long long)pixel_idx * 1442695040888963407ULL + 1ULL;

    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
    int valid_samples = 0;

    for (int s = 0; s < spp; s++) {
        float u = (2.0f * ((float)px + rng_f32(&rng_state)) / fres) - 1.0f;
        float v = (2.0f * ((float)py + rng_f32(&rng_state)) / fres) - 1.0f;
        float r = sqrtf(u*u + v*v);
        if (r > 1.0f) continue;

        // Equidistant fisheye
        float theta = r * half_pi;
        float sin_t, cos_t;
        sincosf(theta, &sin_t, &cos_t);

        Vec3 dir;
        if (r > 1e-6f) {
            float phi_cos = u / r;
            float phi_sin = v / r;
            dir = v3_norm(v3_add(v3_add(
                v3_scale(right, phi_cos * sin_t),
                v3_scale(cam_up, phi_sin * sin_t)),
                v3_scale(forward, cos_t)));
        } else {
            dir = forward;
        }

        Ray ray;
        ray.origin = origin;
        ray.dir = dir;
        Vec3 color = trace_path(ray, &rng_state, max_bounces, ambient);
        r_sum += color.x;
        g_sum += color.y;
        b_sum += color.z;
        valid_samples++;
    }

    // Average and store (no tonemapping — store linear float)
    // Write to LOCAL position within the batch buffer (blockIdx.x), not global hogel_idx,
    // so that hemi_batch_dev only needs to be batch-sized rather than all-hogels-sized.
    float inv = valid_samples > 0 ? 1.0f / (float)valid_samples : 0.0f;
    long long out_idx = ((long long)blockIdx.x * total_pixels + pixel_idx) * 3;
    output[out_idx + 0] = r_sum * inv;
    output[out_idx + 1] = g_sum * inv;
    output[out_idx + 2] = b_sum * inv;
}

// ── Streaming Scatter Architecture ──────────────────────────────
// Instead of storing all hemisphere data and gathering at reconstruction
// time, we process hogels in batches:
//   render batch → extract preview thumbnails → extract target_amp
//   GS iterations → finalize → scatter intensity to output accumulator
//   normalize + tonemap accumulator → final RGBA image
// Memory: O(batch) + O(hogels × preview_res²) instead of O(hogels × hemi_res² × 3)

// Downsample a batch of full-resolution RGB hemispheres to small preview thumbnails.
// Output is stored at hogel_offset in the persistent preview buffer.
__global__ void downsample_to_preview(
    const float* __restrict__ hemi_batch,  // [batch × hemi_res × hemi_res × 3] RGB
    float* __restrict__ preview,           // [num_hogels × preview_res × preview_res × 3] RGB
    int batch,
    int hemi_res,
    int preview_res,
    int hogel_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * preview_res * preview_res;
    if (idx >= total) return;

    int hogel_in_batch = idx / (preview_res * preview_res);
    int local = idx % (preview_res * preview_res);
    int px = local % preview_res;
    int py = local / preview_res;

    float scale = (float)hemi_res / (float)preview_res;
    int hx0 = (int)((float)px * scale);
    int hy0 = (int)((float)py * scale);
    int hx1 = min((int)((float)(px + 1) * scale), hemi_res);
    int hy1 = min((int)((float)(py + 1) * scale), hemi_res);

    float r = 0.0f, g = 0.0f, b = 0.0f;
    int count = 0;
    for (int hy = hy0; hy < hy1; hy++) {
        for (int hx = hx0; hx < hx1; hx++) {
            long long hidx = ((long long)hogel_in_batch * hemi_res * hemi_res + (long long)hy * hemi_res + hx) * 3;
            r += hemi_batch[hidx];
            g += hemi_batch[hidx + 1];
            b += hemi_batch[hidx + 2];
            count++;
        }
    }
    if (count > 0) { r /= count; g /= count; b /= count; }

    int hogel_global = hogel_offset + hogel_in_batch;
    long long pidx = ((long long)hogel_global * preview_res * preview_res + (long long)py * preview_res + px) * 3;
    preview[pidx]     = r;
    preview[pidx + 1] = g;
    preview[pidx + 2] = b;
}

// Scatter the GS-reconstructed intensity from one batch of hogels into the
// output accumulator with Gaussian-aperture weighting.
// Each hogel pushes its contribution to the ~(2*half_w+1)^2 output pixels
// it covers. Uses atomicAdd.
__global__ void scatter_hogel_contributions(
    const float* __restrict__ intensity_batch, // [batch × hemi_res × hemi_res] float
    float* __restrict__ output_accum,          // [out_w × out_h] float (atomic)
    float* __restrict__ weight_accum,          // [out_w × out_h] float (atomic)
    int hogel_offset,
    int batch,
    int grid_w, int grid_h,
    int hemi_res,
    int out_w, int out_h,
    float box_w, float box_h,
    float eye_x, float eye_y, float eye_z,
    float sigma_panel,   // Gaussian σ in panel coordinates (mm)
    int half_w           // ceil(2.5 * sigma_panel/box_w * out_w) + 1
) {
    // blockIdx.y = hogel index within batch
    // blockIdx.x * blockDim.x + threadIdx.x = index within scatter window
    int hogel_in_batch = blockIdx.y;
    if (hogel_in_batch >= batch) return;

    int hogel_global = hogel_offset + hogel_in_batch;
    int gx = hogel_global % grid_w;
    int gy = hogel_global / grid_w;
    float cell_w = box_w / (float)grid_w;
    float cell_h = box_h / (float)grid_h;
    float hcx = (gx + 0.5f) * cell_w;
    float hcy = (gy + 0.5f) * cell_h;
    float inv2s2 = 1.0f / (2.0f * sigma_panel * sigma_panel);

    // Hogel centre in output-pixel space (Y-flipped: panel y=0 → output bottom)
    float cx_out = hcx / box_w * (float)out_w;
    float cy_out = (1.0f - hcy / box_h) * (float)out_h;
    int cx_pix = (int)cx_out;
    int cy_pix = (int)cy_out;

    int window_dim = 2 * half_w + 1;
    int window_size = window_dim * window_dim;
    int pixel_in_window = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_in_window >= window_size) return;

    int wx = pixel_in_window % window_dim - half_w;
    int wy = pixel_in_window / window_dim - half_w;
    int px = cx_pix + wx;
    int py = cy_pix + wy;
    if (px < 0 || px >= out_w || py < 0 || py >= out_h) return;

    // Panel position for this output pixel
    float panel_x = ((float)px + 0.5f) / (float)out_w * box_w;
    float panel_y = (1.0f - ((float)py + 0.5f) / (float)out_h) * box_h;

    // Gaussian weight (panel-space distance from hogel centre)
    float ddx = panel_x - hcx;
    float ddy = panel_y - hcy;
    float w = expf(-(ddx*ddx + ddy*ddy) * inv2s2);
    if (w < 1e-4f) return;

    // Direction from observer through panel point
    float dx = panel_x - eye_x;
    float dy = panel_y - eye_y;
    float dz = -eye_z;  // eye_z is negative (observer is behind panel)
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    float d_fwd   = dz / dist;
    float d_right = dx / dist;
    float d_up    = dy / dist;

    // Fisheye lookup (equidistant)
    float half_pi = 1.5707963f;
    float theta = acosf(fminf(fmaxf(d_fwd, -1.0f), 1.0f));
    if (theta >= half_pi) return;  // behind the hogel's hemisphere

    float r_fish = theta / half_pi;
    float sin_theta = sinf(theta);
    float phi_cos = sin_theta > 1e-6f ? d_right / sin_theta : 0.0f;
    float phi_sin = sin_theta > 1e-6f ? d_up    / sin_theta : 0.0f;

    float hr = (float)hemi_res;
    int hu = min((int)((phi_cos * r_fish * 0.5f + 0.5f) * hr), hemi_res - 1);
    int hv = min((int)((phi_sin * r_fish * 0.5f + 0.5f) * hr), hemi_res - 1);

    long long hpx = (long long)hogel_in_batch * hemi_res * hemi_res + (long long)hv * hemi_res + hu;
    float intensity = intensity_batch[hpx];

    int out_idx = py * out_w + px;
    atomicAdd(&output_accum[out_idx], w * intensity);
    atomicAdd(&weight_accum[out_idx], w);
}

// Normalize the scatter accumulators and apply filmic tonemapping → RGBA.
__global__ void normalize_and_tonemap(
    const float* __restrict__ output_accum,   // [out_w × out_h] float
    const float* __restrict__ weight_accum,   // [out_w × out_h] float
    unsigned char* __restrict__ output_rgba,  // [out_w × out_h × 4] u8
    int out_w, int out_h,
    float exposure
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_w * out_h) return;

    float wt = weight_accum[idx];
    float v  = (wt > 1e-7f) ? output_accum[idx] / wt : 0.0f;

    // Filmic tonemap: sqrt(1 - exp(-v * exposure))
    float tv = sqrtf(1.0f - expf(-v * exposure));
    unsigned char c = (unsigned char)(fminf(tv, 1.0f) * 255.0f);

    output_rgba[idx * 4 + 0] = c;
    output_rgba[idx * 4 + 1] = c;
    output_rgba[idx * 4 + 2] = c;
    output_rgba[idx * 4 + 3] = 255;
}

// ════════════════════════════════════════════════════════════
// ── Phase-only Gerchberg-Saxton holography pipeline ─────────
// ════════════════════════════════════════════════════════════
// Input:  hemispheres (RGB float, [num_hogels × N × N × 3])
// Output: phase pattern φ(x,y) per hogel such that |FFT(exp(iφ))|² ≈ I_target
// Physics: simulates a phase-only SLM (e.g. LCoS). Each hogel has its own
// independent fringe pattern φ.
//
// Algorithm (per hogel, N iterations):
//   target_amp = sqrt( luminance(hemisphere) )
//   E_slm = exp(i·φ)                         (phase-only constraint)
//   E_far = FFT(E_slm)                        (far-field propagation)
//   E_far = target_amp · E_far / |E_far|      (far-field magnitude constraint)
//   E_slm = IFFT(E_far)                       (back-propagation)
//   φ     = arg(E_slm)                        (extract phase, drop amplitude)
//
// FFTs are done externally via cuFFT batched plan_many. These kernels only
// do the per-element point-wise steps.

// Extract target amplitude from hemisphere: √luminance, stored as float [N²·B]
__global__ void hemi_to_target_amp(
    const float* __restrict__ hemispheres,  // [B × N² × 3]
    float* __restrict__ target_amp,         // [B × N²]
    int num_hogels,
    int res
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)num_hogels * res * res;
    if ((long long)idx >= total) return;

    const float* px = hemispheres + (long long)idx * 3;
    float r = px[0], g = px[1], b = px[2];
    float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;  // Rec.709 luminance
    target_amp[idx] = sqrtf(fmaxf(lum, 0.0f));
}

// Seed initial phase: random in [-π, π). Uses per-pixel hash of index + seed.
__global__ void init_phase_random(
    float* __restrict__ phase,
    int num_elems,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elems) return;
    unsigned long long s = seed ^ ((unsigned long long)idx * 6364136223846793005ULL + 1442695040888963407ULL);
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    float u = (float)(s & 0xFFFFFF) / (float)0xFFFFFF;
    phase[idx] = u * 6.283185307f - 3.141592653f;
}

// Build complex field E = exp(i·φ) from real phase array.
__global__ void build_complex_from_phase(
    const float* __restrict__ phase,  // [N]
    float2* __restrict__ out,         // [N] (complex)
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float s, c;
    sincosf(phase[idx], &s, &c);
    out[idx].x = c;
    out[idx].y = s;
}

// Far-field magnitude constraint: E = target_amp · E / |E|
// Preserves phase of E, replaces magnitude with target_amp.
__global__ void enforce_far_magnitude(
    float2* __restrict__ e_far,         // [B × N²]
    const float* __restrict__ target_amp, // [B × N²]
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float2 e = e_far[idx];
    float mag = sqrtf(e.x*e.x + e.y*e.y);
    float scale = mag > 1e-12f ? target_amp[idx] / mag : 0.0f;
    e_far[idx].x = e.x * scale;
    e_far[idx].y = e.y * scale;
}

// Extract phase arg(E) and store as real float.
// After IFFT cuFFT does not normalize (gives N·x output) — we could normalize
// but since we only extract arg(·) the scale drops out.
__global__ void extract_phase(
    const float2* __restrict__ e_slm,  // [N]
    float* __restrict__ phase,         // [N]
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float2 e = e_slm[idx];
    phase[idx] = atan2f(e.y, e.x);
}

// Panel model: physical order = panel quantizes the commanded phase first,
// then the device adds physical phase noise (device jitter, thermal, etc).
// phase_bits in [1..16]; 0 means "no quantization". noise_sigma in radians.
__global__ void apply_panel_model(
    float* __restrict__ phase,  // [N] — modified in-place
    int n,
    int phase_bits,
    float noise_sigma,
    unsigned long long noise_seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float p = phase[idx];

    // 1) Quantize commanded phase to 2^bits levels across [-π, π)
    if (phase_bits > 0) {
        float levels = (float)(1 << phase_bits);
        float step = 6.283185307f / levels;
        float wrapped = p + 3.141592653f;
        wrapped -= 6.283185307f * floorf(wrapped / 6.283185307f);
        float q = floorf(wrapped / step + 0.5f) * step;
        p = q - 3.141592653f;
    }

    // 2) Add physical Gaussian phase noise (Box-Muller)
    if (noise_sigma > 0.0f) {
        unsigned long long s = noise_seed ^ ((unsigned long long)idx * 6364136223846793005ULL + 1442695040888963407ULL);
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        float u1 = fmaxf((float)(s & 0xFFFFFF) / (float)0xFFFFFF, 1e-7f);
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        float u2 = (float)(s & 0xFFFFFF) / (float)0xFFFFFF;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(6.283185307f * u2);
        p += noise_sigma * z;
    }

    phase[idx] = p;
}

// Compute intensity |E|² of a complex field and store as single-channel float.
// Also normalizes by N² (to compensate for cuFFT's unnormalized forward FFT).
__global__ void intensity_from_complex(
    const float2* __restrict__ e,  // [B × N²]
    float* __restrict__ intensity, // [B × N²]
    int n,
    float norm_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float2 z = e[idx];
    intensity[idx] = (z.x*z.x + z.y*z.y) * norm_factor;
}

} // extern "C"

use cudarc::driver::{CudaContext, CudaStream, CudaModule, CudaFunction, LaunchConfig, PushKernelArg, UnifiedSlice};
use cudarc::nvrtc::Ptx;
use cudarc::cufft::{CudaFft, FftDirection, sys as cufft_sys};
use std::sync::Arc;

use crate::scene::{self, Vec3, Triangle, Material, MaterialKind};

const CUDA_SRC: &str = include_str!("kernels.cu");
const BOX_W: f32 = 552.8;
const BOX_H: f32 = 548.8;

/// Generate __constant__ scene header for the kernel. Bakes Cornell Box geometry
/// into constant memory at PTX compile time: ~10-50× faster than global reads
/// for brute-force ray-triangle loops.
fn generate_scene_header(scene: &scene::Scene) -> String {
    let n_tris = scene.triangles.len();
    let n_mats = scene.materials.len();

    let mut out = String::with_capacity(n_tris * 140 + n_mats * 80 + 256);
    out.push_str(&format!("#define NUM_TRIS {}\n#define NUM_MATS {}\n", n_tris, n_mats));

    // Triangle geometry — precomputed Möller-Trumbore form:
    // 12 floats/tri: v0, edge1 = v1-v0, edge2 = v2-v0, normal
    // Saves 6 subs per ray-triangle test in the kernel's inner loop.
    out.push_str(&format!("__constant__ float c_tris[{} * 12] = {{\n", n_tris));
    for tri in &scene.triangles {
        let e1x = tri.v1.x - tri.v0.x; let e1y = tri.v1.y - tri.v0.y; let e1z = tri.v1.z - tri.v0.z;
        let e2x = tri.v2.x - tri.v0.x; let e2y = tri.v2.y - tri.v0.y; let e2z = tri.v2.z - tri.v0.z;
        out.push_str(&format!(
            "  {:.6}f,{:.6}f,{:.6}f, {:.6}f,{:.6}f,{:.6}f, {:.6}f,{:.6}f,{:.6}f, {:.6}f,{:.6}f,{:.6}f,\n",
            tri.v0.x, tri.v0.y, tri.v0.z,
            e1x, e1y, e1z,
            e2x, e2y, e2z,
            tri.normal.x, tri.normal.y, tri.normal.z,
        ));
    }
    out.push_str("};\n");

    // Material index per triangle (separate int array — avoids __int_as_float in const init)
    out.push_str(&format!("__constant__ int c_tri_mat[{}] = {{", n_tris));
    for tri in &scene.triangles {
        out.push_str(&format!("{},", tri.material_idx));
    }
    out.push_str("};\n");

    // Material data (6 floats/mat: albedo.rgb, emission.rgb)
    out.push_str(&format!("__constant__ float c_mats[{} * 6] = {{\n", n_mats));
    for mat in &scene.materials {
        out.push_str(&format!(
            "  {:.6}f,{:.6}f,{:.6}f, {:.6}f,{:.6}f,{:.6}f,\n",
            mat.albedo.x, mat.albedo.y, mat.albedo.z,
            mat.emission.x, mat.emission.y, mat.emission.z,
        ));
    }
    out.push_str("};\n");

    // Material kind per-material
    out.push_str(&format!("__constant__ int c_mat_kind[{}] = {{", n_mats));
    for mat in &scene.materials {
        let k: i32 = match mat.kind { MaterialKind::Diffuse => 0, MaterialKind::Emissive => 1 };
        out.push_str(&format!("{},", k));
    }
    out.push_str("};\n");

    out
}

/// Compile CUDA source (with injected scene header) via nvrtc.
fn compile_cuda() -> Ptx {
    let scene = scene::build_cornell_box();
    let header = generate_scene_header(&scene);
    let full_src = format!("{}{}", header, CUDA_SRC);

    let ptx = cudarc::nvrtc::compile_ptx_with_opts(full_src, cudarc::nvrtc::CompileOptions {
        arch: Some("compute_80"),
        use_fast_math: Some(true),
        ..Default::default()
    }).expect("nvrtc compile failed");

    // Patch PTX version: nvrtc 12.9 emits 8.8, driver 570.x only accepts ≤8.0
    Ptx::from_src(ptx.to_src().replace(".version 8.8", ".version 8.0"))
}

/// Pack scene triangles into flat f32 array for GPU.
/// Layout per triangle: v0.xyz, v1.xyz, v2.xyz, normal.xyz, material_idx (13 floats)
fn pack_triangles(scene: &scene::Scene) -> Vec<f32> {
    let mut data = Vec::with_capacity(scene.triangles.len() * 13);
    for tri in &scene.triangles {
        data.extend_from_slice(&[tri.v0.x, tri.v0.y, tri.v0.z]);
        data.extend_from_slice(&[tri.v1.x, tri.v1.y, tri.v1.z]);
        data.extend_from_slice(&[tri.v2.x, tri.v2.y, tri.v2.z]);
        data.extend_from_slice(&[tri.normal.x, tri.normal.y, tri.normal.z]);
        data.push(f32::from_bits(tri.material_idx as u32));
    }
    data
}

/// Pack materials into flat f32 array for GPU.
/// Layout per material: albedo.xyz, emission.xyz, kind (7 floats)
fn pack_materials(scene: &scene::Scene) -> Vec<f32> {
    let mut data = Vec::with_capacity(scene.materials.len() * 7);
    for mat in &scene.materials {
        data.extend_from_slice(&[mat.albedo.x, mat.albedo.y, mat.albedo.z]);
        data.extend_from_slice(&[mat.emission.x, mat.emission.y, mat.emission.z]);
        let kind_int: i32 = match mat.kind { MaterialKind::Diffuse => 0, MaterialKind::Emissive => 1 };
        data.push(f32::from_bits(kind_int as u32));
    }
    data
}

pub struct GpuHologramResult {
    pub grid_w: u32,
    pub grid_h: u32,
    pub hemi_res: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub recon_rgba: Vec<u8>,
}

/// Run the full GPU hologram pipeline:
/// 1. Render all hogel hemispheres on GPU (path tracing with NEE)
/// 2. Reconstruct lightfield on GPU at given output resolution
pub fn compute_hologram_gpu(
    grid_w: u32,
    grid_h: u32,
    hemi_res: u32,
    spp: u32,
    out_w: u32,
    out_h: u32,
    max_bounces: u32,
    ambient: f32,
) -> GpuHologramResult {
    let num_hogels = (grid_w * grid_h) as usize;
    let total_pixels = (hemi_res * hemi_res) as usize;

    // Init CUDA
    let ctx = CudaContext::new(0).expect("CUDA context init failed");
    let stream = ctx.default_stream();

    // Compile and load CUDA module (scene baked into PTX as __constant__)
    let ptx = compile_cuda();
    let module = ctx.load_module(ptx).expect("Module load failed");
    let render_fn = module.load_function("render_hemispheres").expect("render_hemispheres not found");
    let recon_fn = module.load_function("reconstruct_lightfield").expect("reconstruct_lightfield not found");

    // Allocate hemisphere output as Unified Memory (may spill to system RAM)
    let hemi_count = num_hogels * total_pixels * 3;
    let mut hemi_dev = unsafe {
        ctx.alloc_unified::<f32>(hemi_count, true).expect("Alloc hemi (unified) failed")
    };
    stream.memset_zeros(&mut hemi_dev).expect("hemi memset failed");

    // Launch hemisphere rendering kernel
    let threads_per_block = 256u32;
    let pixel_blocks = (total_pixels as u32 + threads_per_block - 1) / threads_per_block;
    let render_cfg = LaunchConfig {
        grid_dim: (num_hogels as u32, pixel_blocks, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream.launch_builder(&render_fn)
            .arg(&mut hemi_dev)
            .arg(&(grid_w as i32))
            .arg(&(grid_h as i32))
            .arg(&(hemi_res as i32))
            .arg(&(spp as i32))
            .arg(&BOX_W)
            .arg(&BOX_H)
            .arg(&0i32)  // hogel_offset = 0 (render all)
            .arg(&(max_bounces as i32))
            .arg(&ambient)
            .launch(render_cfg)
            .expect("Hemisphere render launch failed");
    }

    // Lightfield reconstruction on GPU at specified output resolution
    let out_count = (out_w * out_h) as usize;
    let mut recon_dev = stream.alloc_zeros::<u8>(out_count * 4).expect("Alloc recon failed");

    let recon_threads = 256u32;
    let recon_blocks = (out_count as u32 + recon_threads - 1) / recon_threads;
    let recon_cfg = LaunchConfig {
        grid_dim: (recon_blocks, 1, 1),
        block_dim: (recon_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let eye_x = 278.0f32;
    let eye_y = 273.0f32;
    let eye_z = -800.0f32;
    let exposure = 0.5f32;

    unsafe {
        stream.launch_builder(&recon_fn)
            .arg(&mut recon_dev)
            .arg(&hemi_dev)
            .arg(&(grid_w as i32))
            .arg(&(grid_h as i32))
            .arg(&(hemi_res as i32))
            .arg(&(out_w as i32))
            .arg(&(out_h as i32))
            .arg(&BOX_W)
            .arg(&BOX_H)
            .arg(&eye_x)
            .arg(&eye_y)
            .arg(&eye_z)
            .arg(&exposure)
            .launch(recon_cfg)
            .expect("Reconstruction launch failed");
    }

    // Download results
    let recon_rgba: Vec<u8> = stream.clone_dtoh(&recon_dev).expect("D2H recon failed");

    GpuHologramResult {
        grid_w,
        grid_h,
        hemi_res,
        out_w,
        out_h,
        recon_rgba,
    }
}

// ── Session-based batched GPU pipeline ────────────────────

use cudarc::driver::CudaSlice;

/// Persistent state for stepwise Gerchberg-Saxton hologram computation.
/// Buffers and kernel handles are allocated once (at `gs_setup`) and reused
/// across `gs_iterate` / `gs_preview` / `gs_finalize` calls.
pub struct GsState {
    plan: CudaFft,
    batch: usize,
    pixels_per_hogel: usize,
    num_hogels: usize,
    target_amp: CudaSlice<f32>,
    phase: CudaSlice<f32>,
    e_a: CudaSlice<cufft_sys::float2>,
    e_b: CudaSlice<cufft_sys::float2>,
    /// Scratch buffer for projection intensity (per-batch). Kept separate from
    /// target_amp so that preview/finalize can forward-project without destroying
    /// the GS target, which would otherwise break subsequent iterations.
    intensity_scratch: CudaSlice<f32>,
    build_complex_fn: CudaFunction,
    enforce_mag_fn: CudaFunction,
    extract_phase_fn: CudaFunction,
    apply_panel_fn: CudaFunction,
    intensity_fn: CudaFunction,
    gray_to_rgb_fn: CudaFunction,
    iters_done: u32,
}

pub struct GpuHologramSession {
    pub grid_w: u32,
    pub grid_h: u32,
    pub hemi_res: u32,
    pub spp: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub max_bounces: u32,
    pub ambient: f32,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    render_fn: CudaFunction,
    recon_fn: CudaFunction,
    /// Hemisphere radiance buffer. Backed by CUDA Unified Memory so allocations
    /// can exceed VRAM capacity and overflow transparently into system RAM.
    hemi_dev: UnifiedSlice<f32>,
    pub rows_rendered: u32,
    gs: Option<GsState>,
}

impl GpuHologramSession {
    pub fn new(
        grid_w: u32, grid_h: u32,
        hemi_res: u32, spp: u32,
        out_w: u32, out_h: u32,
        max_bounces: u32, ambient: f32,
    ) -> Self {
        let num_hogels = (grid_w * grid_h) as usize;
        let total_pixels = (hemi_res * hemi_res) as usize;

        let ctx = CudaContext::new(0).expect("CUDA init failed");
        let stream = ctx.default_stream();
        let ptx = compile_cuda();
        let module = ctx.load_module(ptx).expect("Module load failed");
        let render_fn = module.load_function("render_hemispheres").expect("render_hemispheres missing");
        let recon_fn = module.load_function("reconstruct_lightfield").expect("reconstruct_lightfield missing");

        let hemi_count = num_hogels * total_pixels * 3;
        // Unified Memory: overflow to system RAM when VRAM is insufficient.
        let mut hemi_dev = unsafe {
            ctx.alloc_unified::<f32>(hemi_count, true).expect("Alloc hemi (unified) failed")
        };
        stream.memset_zeros(&mut hemi_dev).expect("hemi memset failed");

        Self {
            grid_w, grid_h, hemi_res, spp, out_w, out_h,
            max_bounces, ambient,
            stream, module, render_fn, recon_fn,
            hemi_dev, rows_rendered: 0,
            gs: None,
        }
    }

    /// Render `num_rows` rows of hogels starting at `start_row`. Returns rows rendered so far.
    pub fn render_rows(&mut self, start_row: u32, num_rows: u32) -> u32 {
        let _t = std::time::Instant::now();
        let end_row = (start_row + num_rows).min(self.grid_h);
        let actual_rows = end_row.saturating_sub(start_row);
        if actual_rows == 0 { return self.rows_rendered; }

        let row_hogels = (actual_rows * self.grid_w) as usize;
        let total_pixels = (self.hemi_res * self.hemi_res) as usize;
        let hogel_offset = (start_row * self.grid_w) as i32;

        let threads = 256u32;
        let pixel_blocks = (total_pixels as u32 + threads - 1) / threads;
        let render_cfg = LaunchConfig {
            grid_dim: (row_hogels as u32, pixel_blocks, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let grid_w = self.grid_w as i32;
        let grid_h = self.grid_h as i32;
        let hemi_res = self.hemi_res as i32;
        let spp = self.spp as i32;
        let max_bounces = self.max_bounces as i32;
        let ambient = self.ambient;

        unsafe {
            self.stream.launch_builder(&self.render_fn)
                .arg(&mut self.hemi_dev)
                .arg(&grid_w)
                .arg(&grid_h)
                .arg(&hemi_res)
                .arg(&spp)
                .arg(&BOX_W)
                .arg(&BOX_H)
                .arg(&hogel_offset)
                .arg(&max_bounces)
                .arg(&ambient)
                .launch(render_cfg)
                .expect("Render launch failed");
        }

        self.stream.synchronize().expect("Sync failed");
        eprintln!("[holosim] render_rows rows={actual_rows} grid={}x{} hemi={} spp={} -> {}ms",
            self.grid_w, self.grid_h, self.hemi_res, self.spp, _t.elapsed().as_millis());

        self.rows_rendered = end_row;
        self.rows_rendered
    }

    /// Run phase-only Gerchberg-Saxton holography on the rendered hemispheres,
    /// then simulate the panel model (phase quantization + noise), then
    /// forward-propagate the degraded fringe to get the reconstructed intensity.
    /// Replaces hemi_dev with the reconstructed intensity (broadcast to RGB).
    ///
    /// Processes hogels in batches of up to BATCH_HOGELS to stay within VRAM.
    ///
    /// - `iterations`: number of GS iterations (typical 10-50)
    /// - `phase_bits`: 0 = no quantization, N = 2^N phase levels
    /// - `noise_sigma_rad`: standard deviation of Gaussian phase noise (radians)
    /// - `noise_seed`: RNG seed for phase noise (0 = deterministic default)
    pub fn run_gs(
        &mut self,
        iterations: u32,
        phase_bits: u32,
        noise_sigma_rad: f32,
        noise_seed: u64,
    ) {
        use cudarc::driver::CudaSlice;
        const MAX_BATCH_HOGELS: usize = 1024;
        let n = self.hemi_res as usize;
        let pixels_per_hogel = n * n;
        let num_hogels = (self.grid_w * self.grid_h) as usize;
        // Pick largest batch size ≤ MAX that divides num_hogels so no partial final batch.
        // Grid is always a multiple of 16 → num_hogels is multiple of 256, so this loop
        // always terminates at ≥ 256.
        let batch = {
            let mut b = MAX_BATCH_HOGELS.min(num_hogels);
            while num_hogels % b != 0 {
                b -= 1;
            }
            b
        };

        // Load GS kernels
        let hemi_to_amp_fn   = self.module.load_function("hemi_to_target_amp").expect("hemi_to_target_amp missing");
        let init_phase_fn    = self.module.load_function("init_phase_random").expect("init_phase_random missing");
        let build_complex_fn = self.module.load_function("build_complex_from_phase").expect("build_complex_from_phase missing");
        let enforce_mag_fn   = self.module.load_function("enforce_far_magnitude").expect("enforce_far_magnitude missing");
        let extract_phase_fn = self.module.load_function("extract_phase").expect("extract_phase missing");
        let apply_panel_fn   = self.module.load_function("apply_panel_model").expect("apply_panel_model missing");
        let intensity_fn     = self.module.load_function("intensity_from_complex").expect("intensity_from_complex missing");
        let gray_to_rgb_fn   = self.module.load_function("gray_to_rgb_hemi").expect("gray_to_rgb_hemi missing");

        // Per-batch buffers (reused across batches)
        let target_amp_dev: CudaSlice<f32> =
            self.stream.alloc_zeros(batch * pixels_per_hogel).expect("alloc target_amp");
        let mut phase_dev: CudaSlice<f32> =
            self.stream.alloc_zeros(batch * pixels_per_hogel).expect("alloc phase");
        // Complex buffers as float2
        let mut e_a: CudaSlice<cufft_sys::float2> =
            self.stream.alloc_zeros(batch * pixels_per_hogel).expect("alloc e_a");
        let mut e_b: CudaSlice<cufft_sys::float2> =
            self.stream.alloc_zeros(batch * pixels_per_hogel).expect("alloc e_b");

        // cuFFT batched 2D C2C plan (batch = BATCH_HOGELS; partial last batch handled by separate plan)
        let n_dims = [n as i32, n as i32];
        let plan_full = CudaFft::plan_many(
            &n_dims, None, 1, pixels_per_hogel as i32,
            None, 1, pixels_per_hogel as i32,
            cufft_sys::cufftType_t::CUFFT_C2C,
            batch as i32,
            self.stream.clone(),
        ).expect("cufft plan_many");

        let norm_factor = 1.0f32 / (n as f32 * n as f32);

        let threads = 256u32;
        let launch_cfg = |count: i32| LaunchConfig {
            grid_dim: (((count as u32) + threads - 1) / threads, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_batches = (num_hogels + batch - 1) / batch;
        for b_idx in 0..num_batches {
            let hogel_off = b_idx * batch;
            let count = batch.min(num_hogels - hogel_off);
            let elems = (count * pixels_per_hogel) as i32;
            let count_i32 = count as i32;
            let res_i32 = n as i32;

            // By construction batch divides num_hogels — last batch is always full.
            debug_assert_eq!(count, batch);

            // 1) target_amp = sqrt(luminance(hemi)) for this batch
            {
                let hemi_slice = self.hemi_dev.slice(hogel_off * pixels_per_hogel * 3 .. (hogel_off + count) * pixels_per_hogel * 3);
                unsafe {
                    self.stream.launch_builder(&hemi_to_amp_fn)
                        .arg(&hemi_slice)
                        .arg(&target_amp_dev)
                        .arg(&count_i32)
                        .arg(&res_i32)
                        .launch(launch_cfg(elems))
                        .expect("hemi_to_target_amp launch");
                }
            }

            // 2) init phase random
            let phase_seed = noise_seed ^ ((b_idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
            unsafe {
                self.stream.launch_builder(&init_phase_fn)
                    .arg(&phase_dev)
                    .arg(&elems)
                    .arg(&phase_seed)
                    .launch(launch_cfg(elems))
                    .expect("init_phase launch");
            }

            // 3) Gerchberg-Saxton iterations
            for _iter in 0..iterations {
                // E_A = exp(i·φ)
                unsafe {
                    self.stream.launch_builder(&build_complex_fn)
                        .arg(&phase_dev)
                        .arg(&mut e_a)
                        .arg(&elems)
                        .launch(launch_cfg(elems))
                        .expect("build_complex launch");
                }

                // E_B = FFT(E_A) — forward
                plan_full.exec_c2c(&mut e_a, &mut e_b, FftDirection::Forward)
                    .expect("FFT forward");

                // E_B = target_amp · E_B / |E_B|
                unsafe {
                    self.stream.launch_builder(&enforce_mag_fn)
                        .arg(&mut e_b)
                        .arg(&target_amp_dev)
                        .arg(&elems)
                        .launch(launch_cfg(elems))
                        .expect("enforce_mag launch");
                }

                // E_A = IFFT(E_B)
                plan_full.exec_c2c(&mut e_b, &mut e_a, FftDirection::Inverse)
                    .expect("FFT inverse");

                // φ = arg(E_A)
                unsafe {
                    self.stream.launch_builder(&extract_phase_fn)
                        .arg(&e_a)
                        .arg(&mut phase_dev)
                        .arg(&elems)
                        .launch(launch_cfg(elems))
                        .expect("extract_phase launch");
                }
            }

            // 4) Apply panel model (quantize phase + add noise)
            let panel_seed = noise_seed.wrapping_add(0xDEADBEEF).wrapping_mul((b_idx as u64).wrapping_add(1));
            unsafe {
                self.stream.launch_builder(&apply_panel_fn)
                    .arg(&mut phase_dev)
                    .arg(&elems)
                    .arg(&(phase_bits as i32))
                    .arg(&noise_sigma_rad)
                    .arg(&panel_seed)
                    .launch(launch_cfg(elems))
                    .expect("apply_panel launch");
            }

            // 5) Forward-project degraded phase: E_A = exp(iφ), E_B = FFT(E_A)
            unsafe {
                self.stream.launch_builder(&build_complex_fn)
                    .arg(&phase_dev)
                    .arg(&mut e_a)
                    .arg(&elems)
                    .launch(launch_cfg(elems))
                    .expect("build_complex final");
            }
            plan_full.exec_c2c(&mut e_a, &mut e_b, FftDirection::Forward)
                .expect("FFT final forward");

            // 6) intensity = |E_B|² · norm — use target_amp buffer as scratch
            // (needs to be mutable since we're writing into it now)
            {
                // Re-alias target_amp_dev as mutable by using a fresh slice; we need a mut binding.
                // Workaround: overwrite via a separate binding — we pass &target_amp_dev
                // but kernel declares it as float*, and napi/cudarc passes the underlying ptr,
                // so this is safe as long as we don't hold an outstanding &mut ref elsewhere.
                // In cudarc 0.19, `.arg(&T)` for CudaSlice works even if the kernel writes it
                // (kernel pointers are pointers; borrow-checking is pointer-level).
                unsafe {
                    self.stream.launch_builder(&intensity_fn)
                        .arg(&e_b)
                        .arg(&target_amp_dev)
                        .arg(&elems)
                        .arg(&norm_factor)
                        .launch(launch_cfg(elems))
                        .expect("intensity launch");
                }
            }

            // 7) broadcast intensity → RGB hemi_dev
            {
                let mut hemi_slice_out = self.hemi_dev.slice_mut(hogel_off * pixels_per_hogel * 3 .. (hogel_off + count) * pixels_per_hogel * 3);
                unsafe {
                    self.stream.launch_builder(&gray_to_rgb_fn)
                        .arg(&target_amp_dev)
                        .arg(&mut hemi_slice_out)
                        .arg(&elems)
                        .launch(launch_cfg(elems))
                        .expect("gray_to_rgb launch");
                }
            }
        }

        self.stream.synchronize().expect("sync after GS");
    }

    // ───── Step-wise GS API (for UI progress + live preview) ─────

    /// Initialise GS: alloc buffers, extract target amplitude from hemispheres,
    /// seed random phase. Must be called before `gs_iterate` / `gs_preview` /
    /// `gs_finalize`. Subsequent calls replace the previous state.
    pub fn gs_setup(&mut self, noise_seed: u64) {
        let _t = std::time::Instant::now();
        const MAX_BATCH_HOGELS: usize = 1024;
        let n = self.hemi_res as usize;
        let pixels_per_hogel = n * n;
        let num_hogels = (self.grid_w * self.grid_h) as usize;
        let batch = {
            let mut b = MAX_BATCH_HOGELS.min(num_hogels);
            while num_hogels % b != 0 { b -= 1; }
            b
        };

        let hemi_to_amp_fn   = self.module.load_function("hemi_to_target_amp").expect("hemi_to_target_amp missing");
        let init_phase_fn    = self.module.load_function("init_phase_random").expect("init_phase_random missing");
        let build_complex_fn = self.module.load_function("build_complex_from_phase").expect("build_complex_from_phase missing");
        let enforce_mag_fn   = self.module.load_function("enforce_far_magnitude").expect("enforce_far_magnitude missing");
        let extract_phase_fn = self.module.load_function("extract_phase").expect("extract_phase missing");
        let apply_panel_fn   = self.module.load_function("apply_panel_model").expect("apply_panel_model missing");
        let intensity_fn     = self.module.load_function("intensity_from_complex").expect("intensity_from_complex missing");
        let gray_to_rgb_fn   = self.module.load_function("gray_to_rgb_hemi").expect("gray_to_rgb_hemi missing");

        let target_amp: CudaSlice<f32> = self.stream.alloc_zeros(num_hogels * pixels_per_hogel).expect("alloc target_amp");
        let mut phase: CudaSlice<f32> = self.stream.alloc_zeros(num_hogels * pixels_per_hogel).expect("alloc phase");
        let e_a: CudaSlice<cufft_sys::float2> = self.stream.alloc_zeros(batch * pixels_per_hogel).expect("alloc e_a");
        let e_b: CudaSlice<cufft_sys::float2> = self.stream.alloc_zeros(batch * pixels_per_hogel).expect("alloc e_b");
        let intensity_scratch: CudaSlice<f32> = self.stream.alloc_zeros(batch * pixels_per_hogel).expect("alloc intensity_scratch");

        let n_dims = [n as i32, n as i32];
        let plan = CudaFft::plan_many(
            &n_dims, None, 1, pixels_per_hogel as i32,
            None, 1, pixels_per_hogel as i32,
            cufft_sys::cufftType_t::CUFFT_C2C,
            batch as i32,
            self.stream.clone(),
        ).expect("cufft plan_many");

        let threads = 256u32;
        let launch_cfg = |count: i32| LaunchConfig {
            grid_dim: (((count as u32) + threads - 1) / threads, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // target_amp = sqrt(luminance(hemi_dev)) — one launch over ALL hogels
        let total = (num_hogels * pixels_per_hogel) as i32;
        unsafe {
            self.stream.launch_builder(&hemi_to_amp_fn)
                .arg(&self.hemi_dev)
                .arg(&target_amp)
                .arg(&(num_hogels as i32))
                .arg(&(n as i32))
                .launch(launch_cfg(total))
                .expect("hemi_to_target_amp launch");
        }

        // init random phase across ALL hogels
        unsafe {
            self.stream.launch_builder(&init_phase_fn)
                .arg(&mut phase)
                .arg(&total)
                .arg(&noise_seed)
                .launch(launch_cfg(total))
                .expect("init_phase launch");
        }

        self.stream.synchronize().expect("sync gs_setup");
        eprintln!("[holosim] gs_setup hogels={} hemi={} -> {}ms",
            (self.grid_w * self.grid_h), self.hemi_res, _t.elapsed().as_millis());

        self.gs = Some(GsState {
            plan, batch, pixels_per_hogel, num_hogels,
            target_amp, phase, e_a, e_b, intensity_scratch,
            build_complex_fn, enforce_mag_fn, extract_phase_fn,
            apply_panel_fn, intensity_fn, gray_to_rgb_fn,
            iters_done: 0,
        });
    }

    /// Run `n_iters` more Gerchberg-Saxton iterations (in-place on phase buffer).
    pub fn gs_iterate(&mut self, n_iters: u32) {
        let _t = std::time::Instant::now();
        let gs = self.gs.as_mut().expect("gs_iterate called without gs_setup");
        let n = self.hemi_res as usize;
        let threads = 256u32;
        let launch_cfg = |count: i32| LaunchConfig {
            grid_dim: (((count as u32) + threads - 1) / threads, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let num_batches = gs.num_hogels / gs.batch;
        let batch_elems = (gs.batch * gs.pixels_per_hogel) as i32;

        for _iter in 0..n_iters {
            for b_idx in 0..num_batches {
                let hogel_off = b_idx * gs.batch;
                let elem_off = hogel_off * gs.pixels_per_hogel;
                let elem_end = elem_off + gs.batch * gs.pixels_per_hogel;

                // Slice views into per-hogel buffers for this batch
                let phase_slice = gs.phase.slice(elem_off..elem_end);
                let amp_slice = gs.target_amp.slice(elem_off..elem_end);

                // E_A = exp(i·φ)
                unsafe {
                    self.stream.launch_builder(&gs.build_complex_fn)
                        .arg(&phase_slice)
                        .arg(&mut gs.e_a)
                        .arg(&batch_elems)
                        .launch(launch_cfg(batch_elems))
                        .expect("build_complex");
                }
                // E_B = FFT(E_A)
                gs.plan.exec_c2c(&mut gs.e_a, &mut gs.e_b, FftDirection::Forward).expect("FFT fwd");
                // E_B = target·E_B/|E_B|
                unsafe {
                    self.stream.launch_builder(&gs.enforce_mag_fn)
                        .arg(&mut gs.e_b)
                        .arg(&amp_slice)
                        .arg(&batch_elems)
                        .launch(launch_cfg(batch_elems))
                        .expect("enforce_mag");
                }
                // E_A = IFFT(E_B)
                gs.plan.exec_c2c(&mut gs.e_b, &mut gs.e_a, FftDirection::Inverse).expect("FFT inv");
                // φ = arg(E_A) — write into phase slice
                {
                    let mut phase_mut = gs.phase.slice_mut(elem_off..elem_end);
                    unsafe {
                        self.stream.launch_builder(&gs.extract_phase_fn)
                            .arg(&gs.e_a)
                            .arg(&mut phase_mut)
                            .arg(&batch_elems)
                            .launch(launch_cfg(batch_elems))
                            .expect("extract_phase");
                    }
                }
            }
            gs.iters_done += 1;
        }
        let _ = n;
        self.stream.synchronize().expect("sync gs_iterate");
        let gs = self.gs.as_ref().expect("gs_iterate: gs lost after iterate");
        eprintln!("[holosim] gs_iterate n={n_iters} iters_done={} batch={} -> {}ms ({:.1}ms/iter)",
            gs.iters_done, gs.batch, _t.elapsed().as_millis(),
            _t.elapsed().as_secs_f64() * 1000.0 / n_iters as f64);
    }

    /// Forward-propagate current phase (WITHOUT panel model) to produce a
    /// preview intensity written into `hemi_dev` (broadcast to RGB). Safe to
    /// call between iterations for live convergence visualization.
    pub fn gs_preview(&mut self) {
        self.gs_project_phase_to_hemi(false, 0, 0.0, 0);
    }

    /// Apply panel model to the phase (quantization + physical noise), then
    /// forward-propagate to reconstructed intensity in `hemi_dev`. Final step.
    pub fn gs_finalize(&mut self, phase_bits: u32, noise_sigma_rad: f32, noise_seed: u64) {
        let _t = std::time::Instant::now();
        self.gs_project_phase_to_hemi(true, phase_bits, noise_sigma_rad, noise_seed);
        eprintln!("[holosim] gs_finalize -> {}ms", _t.elapsed().as_millis());
    }

    fn gs_project_phase_to_hemi(
        &mut self,
        apply_panel: bool,
        phase_bits: u32,
        noise_sigma_rad: f32,
        noise_seed: u64,
    ) {
        let gs = self.gs.as_mut().expect("gs_project called without gs_setup");
        let n = self.hemi_res as usize;
        let norm_factor = 1.0f32 / (n as f32 * n as f32);
        let threads = 256u32;
        let launch_cfg = |count: i32| LaunchConfig {
            grid_dim: (((count as u32) + threads - 1) / threads, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let num_batches = gs.num_hogels / gs.batch;
        let batch_elems = (gs.batch * gs.pixels_per_hogel) as i32;

        if apply_panel {
            // Apply panel model once over the full phase buffer (single launch, all hogels)
            let total = (gs.num_hogels * gs.pixels_per_hogel) as i32;
            unsafe {
                self.stream.launch_builder(&gs.apply_panel_fn)
                    .arg(&mut gs.phase)
                    .arg(&total)
                    .arg(&(phase_bits as i32))
                    .arg(&noise_sigma_rad)
                    .arg(&noise_seed)
                    .launch(launch_cfg(total))
                    .expect("apply_panel");
            }
        }

        // Forward-project per batch: build complex from phase → FFT → |E|² → broadcast RGB
        for b_idx in 0..num_batches {
            let hogel_off = b_idx * gs.batch;
            let elem_off = hogel_off * gs.pixels_per_hogel;
            let elem_end = elem_off + gs.batch * gs.pixels_per_hogel;
            let rgb_off = elem_off * 3;
            let rgb_end = elem_end * 3;

            let phase_slice = gs.phase.slice(elem_off..elem_end);
            unsafe {
                self.stream.launch_builder(&gs.build_complex_fn)
                    .arg(&phase_slice)
                    .arg(&mut gs.e_a)
                    .arg(&batch_elems)
                    .launch(launch_cfg(batch_elems))
                    .expect("build_complex proj");
            }
            gs.plan.exec_c2c(&mut gs.e_a, &mut gs.e_b, FftDirection::Forward).expect("FFT proj");

            // |E|² into dedicated scratch buffer (NOT target_amp — that would
            // corrupt the GS target and break subsequent iterations)
            {
                let amp_slice = gs.intensity_scratch.slice(0..batch_elems as usize);
                unsafe {
                    self.stream.launch_builder(&gs.intensity_fn)
                        .arg(&gs.e_b)
                        .arg(&amp_slice)
                        .arg(&batch_elems)
                        .arg(&norm_factor)
                        .launch(launch_cfg(batch_elems))
                        .expect("intensity");
                }
            }

            // Broadcast to RGB into hemi_dev
            {
                let amp_slice = gs.intensity_scratch.slice(0..batch_elems as usize);
                let mut hemi_out = self.hemi_dev.slice_mut(rgb_off..rgb_end);
                unsafe {
                    self.stream.launch_builder(&gs.gray_to_rgb_fn)
                        .arg(&amp_slice)
                        .arg(&mut hemi_out)
                        .arg(&batch_elems)
                        .launch(launch_cfg(batch_elems))
                        .expect("gray_to_rgb proj");
                }
            }
        }

        self.stream.synchronize().expect("sync gs_project");
    }

    /// Number of GS iterations completed in the current step-wise session.
    pub fn gs_iters_done(&self) -> u32 {
        self.gs.as_ref().map(|g| g.iters_done).unwrap_or(0)
    }

    /// Drop GS state and free its GPU buffers.
    pub fn gs_reset(&mut self) {
        self.gs = None;
    }

    /// Run lightfield reconstruction, return RGBA pixels.
    pub fn reconstruct(&mut self, eye_x: f32, eye_y: f32, eye_z: f32) -> Vec<u8> {
        let _t = std::time::Instant::now();
        let out_count = (self.out_w * self.out_h) as usize;
        let mut recon_dev = self.stream.alloc_zeros::<u8>(out_count * 4).expect("Alloc recon failed");

        let recon_threads = 256u32;
        let recon_blocks = (out_count as u32 + recon_threads - 1) / recon_threads;
        let recon_cfg = LaunchConfig {
            grid_dim: (recon_blocks, 1, 1),
            block_dim: (recon_threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let exposure = 0.5f32;
        let grid_w = self.grid_w as i32;
        let grid_h = self.grid_h as i32;
        let hemi_res = self.hemi_res as i32;
        let out_w = self.out_w as i32;
        let out_h = self.out_h as i32;

        unsafe {
            self.stream.launch_builder(&self.recon_fn)
                .arg(&mut recon_dev)
                .arg(&self.hemi_dev)
                .arg(&grid_w)
                .arg(&grid_h)
                .arg(&hemi_res)
                .arg(&out_w)
                .arg(&out_h)
                .arg(&BOX_W)
                .arg(&BOX_H)
                .arg(&eye_x)
                .arg(&eye_y)
                .arg(&eye_z)
                .arg(&exposure)
                .launch(recon_cfg)
                .expect("Reconstruction launch failed");
        }

        let result = self.stream.clone_dtoh(&recon_dev).expect("D2H recon failed");
        eprintln!("[holosim] reconstruct {}x{} grid={}x{} hemi={} -> {}ms",
            self.out_w, self.out_h, self.grid_w, self.grid_h, self.hemi_res, _t.elapsed().as_millis());
        result
    }

    /// Extract a single hogel's hemisphere + phase for UI diagnostics.
    /// Returns (hemisphere_rgb_u8, phase_gray_u8_opt, res). phase is None if GS hasn't run.
    /// Hemisphere is tonemapped via exposure + gamma 2.2; phase is mapped (φ+π)/(2π) → [0,255].
    pub fn get_hogel_preview(&mut self, hx: u32, hy: u32) -> Option<(Vec<u8>, Option<Vec<u8>>, u32)> {
        if hx >= self.grid_w || hy >= self.grid_h {
            return None;
        }
        let res = self.hemi_res;
        let pixels = (res * res) as usize;
        let hogel_idx = (hy * self.grid_w + hx) as usize;

        // Hemisphere: 3 floats per pixel, linear RGB
        let hemi_start = hogel_idx * pixels * 3;
        let hemi_end = hemi_start + pixels * 3;
        let hemi_slice = self.hemi_dev.slice(hemi_start..hemi_end);
        let hemi_f32: Vec<f32> = self.stream.clone_dtoh(&hemi_slice).ok()?;
        let exposure = 0.5f32;
        let mut hemi_u8 = Vec::with_capacity(pixels * 3);
        for v in hemi_f32 {
            let tm = (1.0 - (-(v * exposure)).exp()).sqrt();
            let g = tm.clamp(0.0, 1.0);
            hemi_u8.push((g * 255.0) as u8);
        }

        // Phase (optional — only if GS has been set up)
        let phase_u8 = if let Some(gs) = &self.gs {
            let phase_start = hogel_idx * pixels;
            let phase_end = phase_start + pixels;
            let phase_slice = gs.phase.slice(phase_start..phase_end);
            let phase_f32: Vec<f32> = self.stream.clone_dtoh(&phase_slice).ok()?;
            let two_pi = std::f32::consts::TAU;
            let pi = std::f32::consts::PI;
            let mut out = Vec::with_capacity(pixels);
            for p in phase_f32 {
                // wrap to [-π, π] then normalize to [0, 1]
                let wrapped = ((p + pi).rem_euclid(two_pi)) - pi;
                let norm = (wrapped + pi) / two_pi;
                out.push((norm.clamp(0.0, 1.0) * 255.0) as u8);
            }
            Some(out)
        } else {
            None
        };

        Some((hemi_u8, phase_u8, res))
    }
}

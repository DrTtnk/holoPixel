use cudarc::driver::{CudaContext, CudaStream, CudaModule, CudaFunction, LaunchConfig, PushKernelArg, UnifiedSlice, ValidAsZeroBits};
use cudarc::nvrtc::Ptx;
use cudarc::cufft::{CudaFft, FftDirection, sys as cufft_sys};
use std::sync::Arc;

use crate::scene::{self, MaterialKind};

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

// ── Session-based batched GPU pipeline ────────────────────

use cudarc::driver::CudaSlice;

const PREVIEW_RES: u32 = 64;  // hogel thumbnail resolution (rows × cols)
const MAX_BATCH_HOGELS: usize = 4096;

/// Evict a UnifiedSlice from VRAM back to CPU RAM (advisory prefetch).
/// Call before operations that need VRAM budget (e.g. cuFFT plan creation).
/// Caller must `stream.synchronize()` afterwards to wait for migration.
fn prefetch_to_cpu<T: ValidAsZeroBits>(slice: &UnifiedSlice<T>) {
    if let Ok(s) = slice.as_slice() {
        let ptr = s.as_ptr() as *const ::std::ffi::c_void;
        let bytes = slice.len() * std::mem::size_of::<T>();
        // Use runtime API cudaMemPrefetchAsync(devPtr, count, dstDevice=-1 CPU, stream=0)
        // cuMemPrefetchAsync_v2 (driver API) doesn't reliably evict on all platforms.
        let result = unsafe {
            cudarc::runtime::sys::cudaMemPrefetchAsync(ptr, bytes, -1i32, std::ptr::null_mut())
        };
        if result != cudarc::runtime::sys::cudaError::cudaSuccess {
            // Prefetch is advisory; if it fails, CUDA will still handle migration via page faults
            let _ = result;
        }
    }
}

/// Gerchberg-Saxton working state. Allocated once at `gs_setup`, reused per iteration.
/// Does NOT own target_amp (lives in GpuHologramSession), only owns per-batch working buffers.
pub struct GsState {
    plan: CudaFft,
    batch: usize,
    pub pixels_per_hogel: usize,
    num_hogels: usize,
    phase: UnifiedSlice<f32>,          // [num_hogels × hemi_res²] – all phases, unified mem
    e_a: CudaSlice<cufft_sys::float2>,
    e_b: CudaSlice<cufft_sys::float2>,
    intensity_scratch: CudaSlice<f32>,  // [batch × hemi_res²]
    build_complex_fn: CudaFunction,
    enforce_mag_fn: CudaFunction,
    extract_phase_fn: CudaFunction,
    apply_panel_fn: CudaFunction,
    intensity_fn: CudaFunction,
    pub iters_done: u32,
}

/// Streaming pipeline state: batch-local GS buffers + cuFFT plan (reused across batches).
pub struct StreamState {
    plan: CudaFft,
    gs_batch: usize,              // cuFFT batch size (e.g. 64)
    max_render_hogels: usize,     // max hogels per render batch
    pixels_per_hogel: usize,
    target_amp: CudaSlice<f32>,   // [max_render_hogels × pixels_per_hogel]
    phase: CudaSlice<f32>,        // [max_render_hogels × pixels_per_hogel]
    e_a: CudaSlice<cufft_sys::float2>,
    e_b: CudaSlice<cufft_sys::float2>,
    intensity_scratch: CudaSlice<f32>,
    build_complex_fn: CudaFunction,
    enforce_mag_fn: CudaFunction,
    extract_phase_fn: CudaFunction,
    apply_panel_fn: CudaFunction,
    intensity_fn: CudaFunction,
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
    pub streaming: bool,
    // Derived geometry constants (computed once from grid / output dimensions)
    sigma_panel: f32,  // Gaussian σ in panel coords (mm)
    half_w: i32,       // scatter window half-size in output pixels
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    render_fn: CudaFunction,
    downsample_fn: CudaFunction,
    hemi_to_amp_fn: CudaFunction,
    scatter_fn: CudaFunction,
    normalize_fn: CudaFunction,
    init_phase_fn: CudaFunction,
    // Streaming buffers:
    /// Small batch RGB buffer (batch × hemi_res² × 3). Overwritten each render batch.
    hemi_batch_dev: CudaSlice<f32>,
    /// Persistent preview thumbnails (num_hogels × PREVIEW_RES² × 3). Written during render.
    /// None in streaming mode (too large to hold all hogels).
    preview_dev: Option<UnifiedSlice<f32>>,
    /// Per-hogel GS target amplitude (num_hogels × hemi_res²). Written during render.
    /// None in streaming mode (too large).
    target_amp_dev: Option<UnifiedSlice<f32>>,
    /// Float accumulator for scatter output (out_w × out_h).
    output_accum_dev: CudaSlice<f32>,
    /// Gaussian weight accumulator (out_w × out_h).
    weight_accum_dev: CudaSlice<f32>,
    /// Final RGBA output (out_w × out_h × 4 u8).
    recon_dev: CudaSlice<u8>,
    pub rows_rendered: u32,
    pub gs: Option<GsState>,
    /// CPU-side backup of target_amp (for PCA k-sweep without re-rendering).
    target_amp_backup: Option<Vec<f32>>,
    pub ss: Option<StreamState>,
    /// PCA sampling: which hogel indices to capture during streaming.
    /// Sorted for efficient lookup.
    pca_sample_indices: Option<Vec<usize>>,
    /// PCA sampling: accumulated target_amp data [num_samples × pixels_per_hogel].
    pca_sample_data: Option<Vec<f32>>,
}

impl GpuHologramSession {
    pub fn new(
        grid_w: u32, grid_h: u32,
        hemi_res: u32, spp: u32,
        out_w: u32, out_h: u32,
        max_bounces: u32, ambient: f32,
    ) -> Self {
        let num_hogels = (grid_w * grid_h) as usize;
        let pixels_per_hogel = (hemi_res * hemi_res) as usize;
        let out_pixels = (out_w * out_h) as usize;

        // Auto-detect streaming mode: if global target_amp would exceed ~4 GB
        let target_amp_bytes = (num_hogels as u64) * (pixels_per_hogel as u64) * 4;
        let streaming = target_amp_bytes > 4_000_000_000;
        if streaming {
            eprintln!("[holosim] STREAMING mode: {}×{} grid ({} hogels), target_amp would be {:.1} GB",
                grid_w, grid_h, num_hogels, target_amp_bytes as f64 / 1e9);
        }

        let ctx = CudaContext::new(0).expect("CUDA init failed");
        let stream = ctx.default_stream();
        let ptx = compile_cuda();
        let module = ctx.load_module(ptx).expect("Module load failed");

        let render_fn       = module.load_function("render_hemispheres").expect("render_hemispheres missing");
        let downsample_fn   = module.load_function("downsample_to_preview").expect("downsample_to_preview missing");
        let hemi_to_amp_fn  = module.load_function("hemi_to_target_amp").expect("hemi_to_target_amp missing");
        let scatter_fn      = module.load_function("scatter_hogel_contributions").expect("scatter_hogel_contributions missing");
        let normalize_fn    = module.load_function("normalize_and_tonemap").expect("normalize_and_tonemap missing");
        let init_phase_fn   = module.load_function("init_phase_random").expect("init_phase_random missing");

        // Streaming buffers
        let batch = MAX_BATCH_HOGELS.min(num_hogels);
        let hemi_batch_dev  = stream.alloc_zeros::<f32>(batch * pixels_per_hogel * 3).expect("alloc hemi_batch");

        // Large global buffers: only allocate in standard (non-streaming) mode
        let (preview_dev, target_amp_dev) = if streaming {
            (None, None)
        } else {
            let mut p = unsafe { ctx.alloc_unified::<f32>(num_hogels * (PREVIEW_RES * PREVIEW_RES) as usize * 3, true) }.expect("alloc preview");
            stream.memset_zeros(&mut p).expect("zero preview");
            let mut t = unsafe { ctx.alloc_unified::<f32>(num_hogels * pixels_per_hogel, true) }.expect("alloc target_amp");
            stream.memset_zeros(&mut t).expect("zero target_amp");
            (Some(p), Some(t))
        };

        let output_accum_dev = stream.alloc_zeros::<f32>(out_pixels).expect("alloc output_accum");
        let weight_accum_dev = stream.alloc_zeros::<f32>(out_pixels).expect("alloc weight_accum");
        let recon_dev       = stream.alloc_zeros::<u8>(out_pixels * 4).expect("alloc recon");

        // Precompute scatter geometry (box fill: each hogel fills its tile, flat weight)
        let cell_w = BOX_W / grid_w as f32;
        let cell_h = BOX_H / grid_h as f32;
        let sigma_panel = 0.5f32 * cell_w.min(cell_h); // kept for API compat, not used
        let half_w = ((0.5f32 * cell_w / BOX_W * out_w as f32).ceil() as i32).max(1);

        Self {
            grid_w, grid_h, hemi_res, spp, out_w, out_h, max_bounces, ambient,
            streaming,
            sigma_panel, half_w,
            stream, ctx, module,
            render_fn, downsample_fn, hemi_to_amp_fn, scatter_fn, normalize_fn, init_phase_fn,
            hemi_batch_dev, preview_dev, target_amp_dev,
            output_accum_dev, weight_accum_dev, recon_dev,
            rows_rendered: 0,
            gs: None,
            target_amp_backup: None,
            ss: None,
            pca_sample_indices: None,
            pca_sample_data: None,
        }
    }

    /// Render `num_rows` rows of hogels. Splits into sub-batches to fit `hemi_batch_dev`.
    /// Populates `preview_dev` thumbnails and `target_amp_dev` for GS.
    /// Returns total rows rendered so far.
    pub fn render_rows(&mut self, start_row: u32, num_rows: u32) -> u32 {
        let _t = std::time::Instant::now();
        let end_row = (start_row + num_rows).min(self.grid_h);
        if end_row <= start_row { return self.rows_rendered; }

        // hemi_batch_dev holds MAX_BATCH_HOGELS hogels; clamp rows per sub-batch
        let max_rows_per_sub = ((MAX_BATCH_HOGELS as u32) / self.grid_w).max(1);
        let mut row = start_row;
        while row < end_row {
            let sub_rows = (end_row - row).min(max_rows_per_sub);
            self.render_sub_batch(row, sub_rows);
            row += sub_rows;
        }

        let total_rows = end_row - start_row;
        eprintln!("[holosim] render_rows rows={total_rows} grid={}x{} hemi={} spp={} -> {}ms",
            self.grid_w, self.grid_h, self.hemi_res, self.spp, _t.elapsed().as_millis());
        self.rows_rendered = end_row;
        self.rows_rendered
    }

    fn render_sub_batch(&mut self, start_row: u32, num_rows: u32) {
        let row_hogels = (num_rows * self.grid_w) as usize;
        let pixels_per_hogel = (self.hemi_res * self.hemi_res) as usize;
        let hogel_offset = (start_row * self.grid_w) as i32;

        let threads = 256u32;
        let pixel_blocks = (pixels_per_hogel as u32 + threads - 1) / threads;
        let render_cfg = LaunchConfig {
            grid_dim: (row_hogels as u32, pixel_blocks, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let (grid_w, grid_h, hemi_res, spp, max_bounces, ambient) = (
            self.grid_w as i32, self.grid_h as i32, self.hemi_res as i32,
            self.spp as i32, self.max_bounces as i32, self.ambient,
        );
        unsafe {
            self.stream.launch_builder(&self.render_fn)
                .arg(&mut self.hemi_batch_dev)
                .arg(&grid_w).arg(&grid_h).arg(&hemi_res).arg(&spp)
                .arg(&BOX_W).arg(&BOX_H)
                .arg(&hogel_offset)
                .arg(&max_bounces).arg(&ambient)
                .launch(render_cfg)
                .expect("Render launch failed");
        }

        // Downsample batch → preview thumbnails (standard mode only)
        if let Some(ref mut preview_dev) = self.preview_dev {
            let batch_i32 = row_hogels as i32;
            let preview_res_i32 = PREVIEW_RES as i32;
            let total_preview = row_hogels * (PREVIEW_RES * PREVIEW_RES) as usize;
            let downsample_cfg = LaunchConfig {
                grid_dim: ((total_preview as u32 + threads - 1) / threads, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream.launch_builder(&self.downsample_fn)
                    .arg(&self.hemi_batch_dev)
                    .arg(preview_dev)
                    .arg(&batch_i32)
                    .arg(&hemi_res)
                    .arg(&preview_res_i32)
                    .arg(&hogel_offset)
                    .launch(downsample_cfg)
                    .expect("Downsample launch failed");
            }
        }

        // Extract grayscale target_amp (standard mode only)
        if let Some(ref mut target_amp_dev) = self.target_amp_dev {
            let batch_i32 = row_hogels as i32;
            let total_amp = (row_hogels * pixels_per_hogel) as i32;
            let amp_cfg = LaunchConfig {
                grid_dim: ((total_amp as u32 + threads - 1) / threads, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let amp_offset: i64 = hogel_offset as i64 * pixels_per_hogel as i64;
            unsafe {
                self.stream.launch_builder(&self.hemi_to_amp_fn)
                    .arg(&self.hemi_batch_dev)
                    .arg(target_amp_dev)
                    .arg(&batch_i32)
                    .arg(&hemi_res)
                    .arg(&amp_offset)
                    .launch(amp_cfg)
                    .expect("hemi_to_amp launch failed");
            }
        }

        self.stream.synchronize().expect("Sync failed");
    }

    /// Initialise GS: alloc phase + FFT plan, seed random phase. `target_amp_dev`
    /// was already populated during `render_rows`; no re-extraction needed.
    pub fn gs_setup(&mut self, noise_seed: u64) {
        let _t = std::time::Instant::now();
        let n = self.hemi_res as usize;
        let pixels_per_hogel = n * n;
        let num_hogels = (self.grid_w * self.grid_h) as usize;

        // Evict large unified buffers from VRAM so cuFFT plan creation has enough room.
        if let Some(ref t) = self.target_amp_dev { prefetch_to_cpu(t); }
        if let Some(ref p) = self.preview_dev { prefetch_to_cpu(p); }
        // Device-wide synchronize to ensure all prefetch migrations complete
        unsafe { cudarc::runtime::sys::cudaDeviceSynchronize() };
        self.stream.synchronize().expect("sync pre-gs_setup (prefetch)");
        let batch = {
            let mut b = MAX_BATCH_HOGELS.min(num_hogels);
            // Cap batch to keep e_a + e_b ≤ 64 MB VRAM.
            // Larger batches degrade cuFFT cache efficiency (batch=64 is optimal for hemi=256).
            const MAX_EA_EB_BYTES: usize = 64 * 1024 * 1024;
            let max_by_vram = (MAX_EA_EB_BYTES / (pixels_per_hogel * 16)).max(1);
            b = b.min(max_by_vram);
            while num_hogels % b != 0 { b -= 1; }
            b
        };

        let build_complex_fn = self.module.load_function("build_complex_from_phase").expect("build_complex missing");
        let enforce_mag_fn   = self.module.load_function("enforce_far_magnitude").expect("enforce_far_magnitude missing");
        let extract_phase_fn = self.module.load_function("extract_phase").expect("extract_phase missing");
        let apply_panel_fn   = self.module.load_function("apply_panel_model").expect("apply_panel_model missing");
        let intensity_fn     = self.module.load_function("intensity_from_complex").expect("intensity_from_complex missing");

        let mut phase            = unsafe { self.ctx.alloc_unified::<f32>(num_hogels * pixels_per_hogel, true) }.expect("alloc phase: unified mem");
        self.stream.memset_zeros(&mut phase).expect("zero phase");
        let e_a                  = self.stream.alloc_zeros::<cufft_sys::float2>(batch * pixels_per_hogel).expect("alloc e_a");
        let e_b                  = self.stream.alloc_zeros::<cufft_sys::float2>(batch * pixels_per_hogel).expect("alloc e_b");
        let intensity_scratch    = self.stream.alloc_zeros::<f32>(batch * pixels_per_hogel).expect("alloc scratch");

        let n_dims = [n as i32, n as i32];
        let plan = CudaFft::plan_many(
            &n_dims, None, 1, pixels_per_hogel as i32,
            None, 1, pixels_per_hogel as i32,
            cufft_sys::cufftType_t::CUFFT_C2C,
            batch as i32,
            self.stream.clone(),
        ).expect("cufft plan");

        // Seed random phase across all hogels (total may exceed i32 for large grids)
        let total: i64 = (num_hogels * pixels_per_hogel) as i64;
        let threads = 256u32;
        let init_cfg = LaunchConfig {
            grid_dim: (((total as u64 + threads as u64 - 1) / threads as u64) as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.init_phase_fn)
                .arg(&mut phase)
                .arg(&total)
                .arg(&noise_seed)
                .launch(init_cfg)
                .expect("init_phase launch");
        }

        self.stream.synchronize().expect("sync gs_setup");
        eprintln!("[holosim] gs_setup hogels={} hemi={} -> {}ms",
            num_hogels, self.hemi_res, _t.elapsed().as_millis());

        self.gs = Some(GsState {
            plan, batch, pixels_per_hogel, num_hogels,
            phase, e_a, e_b, intensity_scratch,
            build_complex_fn, enforce_mag_fn, extract_phase_fn,
            apply_panel_fn, intensity_fn,
            iters_done: 0,
        });
    }

    /// Run `n_iters` Gerchberg-Saxton iterations (in-place on phase buffer).
    pub fn gs_iterate(&mut self, n_iters: u32) {
        let _t = std::time::Instant::now();
        let gs = self.gs.as_mut().expect("gs_iterate called without gs_setup");
        let threads = 256u32;
        let launch_cfg = |count: i32| LaunchConfig {
            grid_dim: ((count as u32 + threads - 1) / threads, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let num_batches = gs.num_hogels / gs.batch;
        let batch_elems = (gs.batch * gs.pixels_per_hogel) as i32;

        for _iter in 0..n_iters {
            for b_idx in 0..num_batches {
                let elem_off_i64 = (b_idx * gs.batch * gs.pixels_per_hogel) as i64;

                unsafe {
                    self.stream.launch_builder(&gs.build_complex_fn)
                        .arg(&gs.phase).arg(&mut gs.e_a).arg(&batch_elems).arg(&elem_off_i64)
                        .launch(launch_cfg(batch_elems)).expect("build_complex");
                }
                gs.plan.exec_c2c(&mut gs.e_a, &mut gs.e_b, FftDirection::Forward).expect("FFT fwd");
                unsafe {
                    self.stream.launch_builder(&gs.enforce_mag_fn)
                        .arg(&mut gs.e_b).arg(self.target_amp_dev.as_ref().expect("standard mode only")).arg(&batch_elems).arg(&elem_off_i64)
                        .launch(launch_cfg(batch_elems)).expect("enforce_mag");
                }
                gs.plan.exec_c2c(&mut gs.e_b, &mut gs.e_a, FftDirection::Inverse).expect("FFT inv");
                unsafe {
                    self.stream.launch_builder(&gs.extract_phase_fn)
                        .arg(&gs.e_a).arg(&mut gs.phase).arg(&batch_elems).arg(&elem_off_i64)
                        .launch(launch_cfg(batch_elems)).expect("extract_phase");
                }
            }
            gs.iters_done += 1;
        }
        self.stream.synchronize().expect("sync gs_iterate");
        let gs = self.gs.as_ref().unwrap();
        eprintln!("[holosim] gs_iterate n={n_iters} iters_done={} batch={} -> {}ms ({:.1}ms/iter)",
            gs.iters_done, gs.batch, _t.elapsed().as_millis(),
            _t.elapsed().as_secs_f64() * 1000.0 / n_iters as f64);
    }

    /// Forward-project current phase (no panel model) → scatter to output_accum.
    /// For live convergence preview between iterations.
    pub fn gs_preview(&mut self) {
        self.project_and_scatter(false, 0, 0.0, 0, 278.0, 273.0, -800.0);
    }

    /// Combined: iterate + preview + reconstruct in one call (saves 2 IPC round-trips).
    pub fn gs_iterate_preview_reconstruct(
        &mut self, n_iters: u32, eye_x: f32, eye_y: f32, eye_z: f32,
    ) -> (u32, Vec<u8>) {
        self.gs_iterate(n_iters);
        let iters = self.gs_iters_done();
        self.project_and_scatter(false, 0, 0.0, 0, eye_x, eye_y, eye_z);
        let recon = self.reconstruct_tonemap_only();
        (iters, recon)
    }

    /// Apply panel model to phase, forward-project → scatter to output_accum. Final step.
    pub fn gs_finalize(&mut self, phase_bits: u32, noise_sigma_rad: f32, noise_seed: u64) {
        let _t = std::time::Instant::now();
        self.project_and_scatter(true, phase_bits, noise_sigma_rad, noise_seed, 278.0, 273.0, -800.0);
        eprintln!("[holosim] gs_finalize -> {}ms", _t.elapsed().as_millis());
    }

    fn project_and_scatter(
        &mut self,
        apply_panel: bool,
        phase_bits: u32,
        noise_sigma_rad: f32,
        noise_seed: u64,
        eye_x: f32, eye_y: f32, eye_z: f32,
    ) {
        let gs = self.gs.as_mut().expect("project_and_scatter called without gs_setup");
        let n = gs.pixels_per_hogel;
        let norm_factor = 1.0f32 / (n as f32);  // hemi_res² already embedded in n
        let threads = 256u32;
        let launch_cfg = |count: i32| LaunchConfig {
            grid_dim: ((count as u32 + threads - 1) / threads, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let num_batches = gs.num_hogels / gs.batch;
        let batch_elems = (gs.batch * n) as i32;
        let hemi_res = (n as f32).sqrt() as i32;  // n = hemi_res²

        // Panel model: single launch over all hogels (total may exceed i32 for large grids)
        if apply_panel {
            let total: i64 = (gs.num_hogels * n) as i64;
            let panel_cfg = LaunchConfig {
                grid_dim: (((total as u64 + threads as u64 - 1) / threads as u64) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream.launch_builder(&gs.apply_panel_fn)
                    .arg(&mut gs.phase)
                    .arg(&total)
                    .arg(&(phase_bits as i32))
                    .arg(&noise_sigma_rad)
                    .arg(&noise_seed)
                    .launch(panel_cfg)
                    .expect("apply_panel");
            }
        }

        // Zero the output accumulators before scattering
        self.stream.memset_zeros(&mut self.output_accum_dev).expect("zero output_accum");
        self.stream.memset_zeros(&mut self.weight_accum_dev).expect("zero weight_accum");

        let (grid_w, grid_h, out_w, out_h) = (
            self.grid_w as i32, self.grid_h as i32, self.out_w as i32, self.out_h as i32,
        );
        let sigma_panel = self.sigma_panel;
        let half_w      = self.half_w;
        let window_dim  = (2 * half_w + 1) as u32;
        let window_size = window_dim * window_dim;

        let scatter_cfg = |b_size: u32| LaunchConfig {
            // blockIdx.y = hogel_in_batch, blockIdx.x * blockDim.x + threadIdx.x = pixel_in_window
            grid_dim: ((window_size + threads - 1) / threads, b_size, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        for b_idx in 0..num_batches {
            let hogel_off    = (b_idx * gs.batch) as i32;
            let elem_off_i64 = (b_idx * gs.batch * n) as i64;

            // forward project: E_A = exp(iφ)
            unsafe {
                self.stream.launch_builder(&gs.build_complex_fn)
                    .arg(&gs.phase).arg(&mut gs.e_a).arg(&batch_elems).arg(&elem_off_i64)
                    .launch(launch_cfg(batch_elems)).expect("build_complex scatter");
            }
            gs.plan.exec_c2c(&mut gs.e_a, &mut gs.e_b, FftDirection::Forward).expect("FFT scatter");
            // |E|² → intensity_scratch
            {
                let scratch = gs.intensity_scratch.slice(0..gs.batch * n);
                unsafe {
                    self.stream.launch_builder(&gs.intensity_fn)
                        .arg(&gs.e_b).arg(&scratch)
                        .arg(&batch_elems).arg(&norm_factor)
                        .launch(launch_cfg(batch_elems)).expect("intensity scatter");
                }
            }
            // scatter intensity_scratch → output_accum / weight_accum
            {
                let intensity_view = gs.intensity_scratch.slice(0..gs.batch * n);
                let batch_u32 = gs.batch as u32;
                unsafe {
                    self.stream.launch_builder(&self.scatter_fn)
                        .arg(&intensity_view)
                        .arg(&mut self.output_accum_dev)
                        .arg(&mut self.weight_accum_dev)
                        .arg(&hogel_off)
                        .arg(&(batch_u32 as i32))
                        .arg(&grid_w).arg(&grid_h)
                        .arg(&hemi_res)
                        .arg(&out_w).arg(&out_h)
                        .arg(&BOX_W).arg(&BOX_H)
                        .arg(&eye_x).arg(&eye_y).arg(&eye_z)
                        .arg(&sigma_panel)
                        .arg(&half_w)
                        .launch(scatter_cfg(batch_u32))
                        .expect("scatter launch");
                }
            }
        }
        self.stream.synchronize().expect("sync project_and_scatter");
    }

    pub fn gs_iters_done(&self) -> u32 {
        self.gs.as_ref().map(|g| g.iters_done).unwrap_or(0)
    }

    pub fn gs_reset(&mut self) { self.gs = None; }

    /// Reconstruct hologram from a given eye position.
    /// Re-runs the scatter with the provided eye coordinates (supporting parallax).
    pub fn reconstruct(&mut self, eye_x: f32, eye_y: f32, eye_z: f32) -> Vec<u8> {
        let _t = std::time::Instant::now();
        if self.gs.is_some() {
            self.project_and_scatter(false, 0, 0.0, 0, eye_x, eye_y, eye_z);
        }
        let result = self.reconstruct_tonemap_only();
        eprintln!("[holosim] reconstruct {}x{} grid={}x{} hemi={} -> {}ms",
            self.out_w, self.out_h, self.grid_w, self.grid_h, self.hemi_res, _t.elapsed().as_millis());
        result
    }

    /// Tonemap the current output_accum → recon_dev and D2H copy.
    /// Assumes output_accum/weight_accum are already populated.
    fn reconstruct_tonemap_only(&mut self) -> Vec<u8> {
        let out_pixels = (self.out_w * self.out_h) as usize;
        let threads = 256u32;
        let exposure = 0.5f32;
        let norm_cfg = LaunchConfig {
            grid_dim: ((out_pixels as u32 + threads - 1) / threads, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.normalize_fn)
                .arg(&self.output_accum_dev)
                .arg(&self.weight_accum_dev)
                .arg(&mut self.recon_dev)
                .arg(&(self.out_w as i32)).arg(&(self.out_h as i32))
                .arg(&exposure)
                .launch(norm_cfg)
                .expect("normalize_and_tonemap launch");
        }
        self.stream.clone_dtoh(&self.recon_dev).expect("D2H recon failed")
    }

    /// Return RGB thumbnail + phase for the hogel at (hx, hy) from stored preview_dev.
    pub fn get_hogel_preview(&mut self, hx: u32, hy: u32) -> Option<(Vec<u8>, Option<Vec<u8>>, u32)> {
        if hx >= self.grid_w || hy >= self.grid_h { return None; }
        let preview_dev = self.preview_dev.as_ref()?;  // None in streaming mode
        let res = PREVIEW_RES;
        let pixels = (res * res) as usize;
        let hogel_idx = (hy * self.grid_w + hx) as usize;

        // Read thumbnail from preview_dev
        let start = hogel_idx * pixels * 3;
        let end   = start + pixels * 3;
        let rgb_f32: Vec<f32> = self.stream.clone_dtoh(&preview_dev.slice(start..end)).ok()?;
        let exposure = 0.5f32;
        let mut hemi_u8 = Vec::with_capacity(pixels * 3);
        for v in rgb_f32 {
            let tm = (1.0f32 - (-(v * exposure)).exp()).sqrt().clamp(0.0, 1.0);
            hemi_u8.push((tm * 255.0) as u8);
        }

        // Phase from GsState (if GS ran)
        let phase_u8 = if let Some(gs) = &self.gs {
            let pstart = hogel_idx * gs.pixels_per_hogel;
            let pend   = pstart + gs.pixels_per_hogel;
            let pf32: Vec<f32> = self.stream.clone_dtoh(&gs.phase.slice(pstart..pend)).ok()?;
            let pi = std::f32::consts::PI;
            let tau = std::f32::consts::TAU;
            let out: Vec<u8> = pf32.iter().map(|&p| {
                let w = ((p + pi).rem_euclid(tau)) - pi;
                let n = ((w + pi) / tau).clamp(0.0, 1.0);
                (n * 255.0) as u8
            }).collect();
            Some(out)
        } else { None };

        Some((hemi_u8, phase_u8, res))
    }

    // ── Streaming pipeline: fused render + GS + scatter per row batch ──

    /// Lazily create/reuse streaming state (cuFFT plan + batch-local buffers).
    fn ensure_stream_state(&mut self) {
        if self.ss.is_some() { return; }
        let _t = std::time::Instant::now();
        let n = self.hemi_res as usize;
        let pixels_per_hogel = n * n;
        let max_render_hogels = MAX_BATCH_HOGELS;

        // cuFFT batch size: cap to keep e_a + e_b ≤ 64 MB
        let gs_batch = {
            let mut b = max_render_hogels;
            const MAX_EA_EB_BYTES: usize = 64 * 1024 * 1024;
            let max_by_vram = (MAX_EA_EB_BYTES / (pixels_per_hogel * 16)).max(1);
            b = b.min(max_by_vram);
            // gs_batch must divide max_render_hogels evenly
            while max_render_hogels % b != 0 { b -= 1; }
            b
        };

        let build_complex_fn = self.module.load_function("build_complex_from_phase").expect("build_complex missing");
        let enforce_mag_fn   = self.module.load_function("enforce_far_magnitude").expect("enforce_far_magnitude missing");
        let extract_phase_fn = self.module.load_function("extract_phase").expect("extract_phase missing");
        let apply_panel_fn   = self.module.load_function("apply_panel_model").expect("apply_panel_model missing");
        let intensity_fn     = self.module.load_function("intensity_from_complex").expect("intensity_from_complex missing");

        let target_amp = self.stream.alloc_zeros::<f32>(max_render_hogels * pixels_per_hogel)
            .expect("alloc stream target_amp");
        let phase = self.stream.alloc_zeros::<f32>(max_render_hogels * pixels_per_hogel)
            .expect("alloc stream phase");
        let e_a = self.stream.alloc_zeros::<cufft_sys::float2>(gs_batch * pixels_per_hogel)
            .expect("alloc stream e_a");
        let e_b = self.stream.alloc_zeros::<cufft_sys::float2>(gs_batch * pixels_per_hogel)
            .expect("alloc stream e_b");
        let intensity_scratch = self.stream.alloc_zeros::<f32>(gs_batch * pixels_per_hogel)
            .expect("alloc stream intensity_scratch");

        let n_dims = [n as i32, n as i32];
        let plan = CudaFft::plan_many(
            &n_dims, None, 1, pixels_per_hogel as i32,
            None, 1, pixels_per_hogel as i32,
            cufft_sys::cufftType_t::CUFFT_C2C,
            gs_batch as i32,
            self.stream.clone(),
        ).expect("cufft plan (streaming)");

        eprintln!("[holosim] stream_state created: gs_batch={}, max_render={} -> {}ms",
            gs_batch, max_render_hogels, _t.elapsed().as_millis());

        self.ss = Some(StreamState {
            plan, gs_batch, max_render_hogels, pixels_per_hogel,
            target_amp, phase, e_a, e_b, intensity_scratch,
            build_complex_fn, enforce_mag_fn, extract_phase_fn, apply_panel_fn, intensity_fn,
        });
    }

    /// Streaming pipeline: render rows → GS → scatter, using only batch-local memory.
    /// Returns total rows rendered so far.
    pub fn stream_batch(
        &mut self,
        start_row: u32,
        num_rows: u32,
        gs_iterations: u32,
        phase_bits: u32,
        noise_sigma: f32,
        noise_seed: u64,
    ) -> u32 {
        let _t = std::time::Instant::now();
        self.ensure_stream_state();
        let end_row = (start_row + num_rows).min(self.grid_h);
        if end_row <= start_row { return self.rows_rendered; }

        let max_rows_per_sub = ((MAX_BATCH_HOGELS as u32) / self.grid_w).max(1);
        let mut row = start_row;
        while row < end_row {
            let sub_rows = (end_row - row).min(max_rows_per_sub);
            self.stream_sub_batch(row, sub_rows, gs_iterations, phase_bits, noise_sigma, noise_seed);
            row += sub_rows;
        }

        let total_rows = end_row - start_row;
        eprintln!("[holosim] stream_batch rows={} grid={}x{} hemi={} spp={} gs={} -> {}ms",
            total_rows, self.grid_w, self.grid_h, self.hemi_res, self.spp, gs_iterations,
            _t.elapsed().as_millis());
        self.rows_rendered = end_row;
        self.rows_rendered
    }

    fn stream_sub_batch(
        &mut self,
        start_row: u32,
        num_rows: u32,
        gs_iterations: u32,
        phase_bits: u32,
        noise_sigma: f32,
        noise_seed: u64,
    ) {
        let row_hogels = (num_rows * self.grid_w) as usize;
        let pixels_per_hogel = (self.hemi_res * self.hemi_res) as usize;
        let hogel_offset = (start_row * self.grid_w) as i32;
        let threads = 256u32;
        let launch_cfg = |count: i32| LaunchConfig {
            grid_dim: ((count as u32 + threads - 1) / threads, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // ── 1. Render hemispheres ──
        let pixel_blocks = (pixels_per_hogel as u32 + threads - 1) / threads;
        let render_cfg = LaunchConfig {
            grid_dim: (row_hogels as u32, pixel_blocks, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let (grid_w, grid_h, hemi_res, spp, max_bounces, ambient) = (
            self.grid_w as i32, self.grid_h as i32, self.hemi_res as i32,
            self.spp as i32, self.max_bounces as i32, self.ambient,
        );
        unsafe {
            self.stream.launch_builder(&self.render_fn)
                .arg(&mut self.hemi_batch_dev)
                .arg(&grid_w).arg(&grid_h).arg(&hemi_res).arg(&spp)
                .arg(&BOX_W).arg(&BOX_H)
                .arg(&hogel_offset)
                .arg(&max_bounces).arg(&ambient)
                .launch(render_cfg)
                .expect("stream render failed");
        }

        // ── 2. Extract target_amp from hemi_batch → ss.target_amp ──
        let batch_i32 = row_hogels as i32;
        let total_amp = (row_hogels * pixels_per_hogel) as i32;
        let amp_offset: i64 = 0;
        {
            let ss = self.ss.as_mut().unwrap();
            unsafe {
                self.stream.launch_builder(&self.hemi_to_amp_fn)
                    .arg(&self.hemi_batch_dev)
                    .arg(&mut ss.target_amp)
                    .arg(&batch_i32)
                    .arg(&hemi_res)
                    .arg(&amp_offset)
                    .launch(launch_cfg(total_amp))
                    .expect("stream hemi_to_amp failed");
            }
        }

        // ── 2b. PCA sampling: collect target_amp for selected hogels ──
        if let Some(ref indices) = self.pca_sample_indices {
            let hogel_off_usize = hogel_offset as usize;
            let hogel_end = hogel_off_usize + row_hogels;
            let start_idx = indices.partition_point(|&i| i < hogel_off_usize);
            let end_idx = indices.partition_point(|&i| i < hogel_end);
            if start_idx < end_idx {
                self.stream.synchronize().expect("sync before PCA sample D2H");
                let ss_ref = self.ss.as_ref().unwrap();
                for sample_pos in start_idx..end_idx {
                    let global_hogel = indices[sample_pos];
                    let local_hogel = global_hogel - hogel_off_usize;
                    let src_offset = local_hogel * pixels_per_hogel;
                    let src_slice = ss_ref.target_amp.slice(src_offset..src_offset + pixels_per_hogel);
                    let mut tmp = vec![0.0f32; pixels_per_hogel];
                    self.stream.memcpy_dtoh(&src_slice, &mut tmp[..]).expect("D2H PCA sample");
                    let data = self.pca_sample_data.as_mut().unwrap();
                    let dst_offset = sample_pos * pixels_per_hogel;
                    data[dst_offset..dst_offset + pixels_per_hogel].copy_from_slice(&tmp);
                }
            }
        }

        // ── 3. Init random phase ──
        let ss = self.ss.as_mut().unwrap();
        let total_phase: i64 = (row_hogels * pixels_per_hogel) as i64;
        let row_seed = noise_seed.wrapping_add(start_row as u64 * 0x9E3779B97F4A7C15);
        {
            let init_cfg = LaunchConfig {
                grid_dim: (((total_phase as u64 + threads as u64 - 1) / threads as u64) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream.launch_builder(&self.init_phase_fn)
                    .arg(&mut ss.phase)
                    .arg(&total_phase)
                    .arg(&row_seed)
                    .launch(init_cfg)
                    .expect("stream init_phase failed");
            }
        }

        // ── 4. GS iterations (batch-local) ──
        let num_gs_batches = row_hogels / ss.gs_batch;
        let gs_batch_elems = (ss.gs_batch * pixels_per_hogel) as i32;

        for _iter in 0..gs_iterations {
            for b in 0..num_gs_batches {
                let off = (b * ss.gs_batch * pixels_per_hogel) as i64;
                unsafe {
                    self.stream.launch_builder(&ss.build_complex_fn)
                        .arg(&ss.phase).arg(&mut ss.e_a).arg(&gs_batch_elems).arg(&off)
                        .launch(launch_cfg(gs_batch_elems)).expect("stream build_complex");
                }
                ss.plan.exec_c2c(&mut ss.e_a, &mut ss.e_b, FftDirection::Forward).expect("stream FFT fwd");
                unsafe {
                    self.stream.launch_builder(&ss.enforce_mag_fn)
                        .arg(&mut ss.e_b).arg(&ss.target_amp).arg(&gs_batch_elems).arg(&off)
                        .launch(launch_cfg(gs_batch_elems)).expect("stream enforce_mag");
                }
                ss.plan.exec_c2c(&mut ss.e_b, &mut ss.e_a, FftDirection::Inverse).expect("stream FFT inv");
                unsafe {
                    self.stream.launch_builder(&ss.extract_phase_fn)
                        .arg(&ss.e_a).arg(&mut ss.phase).arg(&gs_batch_elems).arg(&off)
                        .launch(launch_cfg(gs_batch_elems)).expect("stream extract_phase");
                }
            }
        }

        // ── 5. Apply panel model ──
        if phase_bits > 0 || noise_sigma > 0.0 {
            let panel_cfg = LaunchConfig {
                grid_dim: (((total_phase as u64 + threads as u64 - 1) / threads as u64) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream.launch_builder(&ss.apply_panel_fn)
                    .arg(&mut ss.phase)
                    .arg(&total_phase)
                    .arg(&(phase_bits as i32))
                    .arg(&noise_sigma)
                    .arg(&noise_seed)
                    .launch(panel_cfg)
                    .expect("stream apply_panel");
            }
        }

        // ── 6. Forward project + scatter ──
        let norm_factor = 1.0f32 / (pixels_per_hogel as f32);
        let (out_w, out_h) = (self.out_w as i32, self.out_h as i32);
        let sigma_panel = self.sigma_panel;
        let half_w = self.half_w;
        let window_dim = (2 * half_w + 1) as u32;
        let window_size = window_dim * window_dim;
        let scatter_cfg = |b_size: u32| LaunchConfig {
            grid_dim: ((window_size + threads - 1) / threads, b_size, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        // Use default eye position for scatter (parallax reconstruct can re-do later)
        let (eye_x, eye_y, eye_z) = (278.0f32, 273.0f32, -800.0f32);

        for b in 0..num_gs_batches {
            let off = (b * ss.gs_batch * pixels_per_hogel) as i64;
            let hogel_off = hogel_offset + (b * ss.gs_batch) as i32;

            unsafe {
                self.stream.launch_builder(&ss.build_complex_fn)
                    .arg(&ss.phase).arg(&mut ss.e_a).arg(&gs_batch_elems).arg(&off)
                    .launch(launch_cfg(gs_batch_elems)).expect("stream build_complex scatter");
            }
            ss.plan.exec_c2c(&mut ss.e_a, &mut ss.e_b, FftDirection::Forward).expect("stream FFT scatter");
            {
                let scratch = ss.intensity_scratch.slice(0..ss.gs_batch * pixels_per_hogel);
                unsafe {
                    self.stream.launch_builder(&ss.intensity_fn)
                        .arg(&ss.e_b).arg(&scratch)
                        .arg(&gs_batch_elems).arg(&norm_factor)
                        .launch(launch_cfg(gs_batch_elems)).expect("stream intensity");
                }
            }
            {
                let intensity_view = ss.intensity_scratch.slice(0..ss.gs_batch * pixels_per_hogel);
                let batch_u32 = ss.gs_batch as u32;
                unsafe {
                    self.stream.launch_builder(&self.scatter_fn)
                        .arg(&intensity_view)
                        .arg(&mut self.output_accum_dev)
                        .arg(&mut self.weight_accum_dev)
                        .arg(&hogel_off)
                        .arg(&(batch_u32 as i32))
                        .arg(&(self.grid_w as i32)).arg(&(self.grid_h as i32))
                        .arg(&hemi_res)
                        .arg(&out_w).arg(&out_h)
                        .arg(&BOX_W).arg(&BOX_H)
                        .arg(&eye_x).arg(&eye_y).arg(&eye_z)
                        .arg(&sigma_panel)
                        .arg(&half_w)
                        .launch(scatter_cfg(batch_u32))
                        .expect("stream scatter");
                }
            }
        }

        self.stream.synchronize().expect("stream sync");
    }

    /// Zero the scatter accumulators (call before starting streaming pipeline).
    pub fn stream_reset_accum(&mut self) {
        self.stream.memset_zeros(&mut self.output_accum_dev).expect("zero output_accum");
        self.stream.memset_zeros(&mut self.weight_accum_dev).expect("zero weight_accum");
    }

    /// Profile one streaming batch with per-stage GPU synchronization for timing.
    /// Returns (total_rows_done, profile_map) where profile_map has stage timings in ms.
    pub fn stream_batch_profile(
        &mut self,
        start_row: u32,
        num_rows: u32,
        gs_iterations: u32,
        phase_bits: u32,
        noise_sigma: f32,
        noise_seed: u64,
    ) -> (u32, Vec<(String, f64)>) {
        let _t0 = std::time::Instant::now();
        self.ensure_stream_state();
        let end_row = (start_row + num_rows).min(self.grid_h);
        if end_row <= start_row { return (self.rows_rendered, vec![]); }

        // Only profile a single sub-batch (the first one)
        let max_rows_per_sub = ((MAX_BATCH_HOGELS as u32) / self.grid_w).max(1);
        let sub_rows = (end_row - start_row).min(max_rows_per_sub);
        let timings = self.stream_sub_batch_profiled(
            start_row, sub_rows, gs_iterations, phase_bits, noise_sigma, noise_seed,
        );

        // If there are more sub-batches, run them normally
        let mut row = start_row + sub_rows;
        while row < end_row {
            let sr = (end_row - row).min(max_rows_per_sub);
            self.stream_sub_batch(row, sr, gs_iterations, phase_bits, noise_sigma, noise_seed);
            row += sr;
        }

        self.rows_rendered = end_row;
        (self.rows_rendered, timings)
    }

    fn stream_sub_batch_profiled(
        &mut self,
        start_row: u32,
        num_rows: u32,
        gs_iterations: u32,
        phase_bits: u32,
        noise_sigma: f32,
        noise_seed: u64,
    ) -> Vec<(String, f64)> {
        let mut timings: Vec<(String, f64)> = Vec::new();
        let mut t = std::time::Instant::now();

        let row_hogels = (num_rows * self.grid_w) as usize;
        let pixels_per_hogel = (self.hemi_res * self.hemi_res) as usize;
        let hogel_offset = (start_row * self.grid_w) as i32;
        let threads = 256u32;
        let launch_cfg = |count: i32| LaunchConfig {
            grid_dim: ((count as u32 + threads - 1) / threads, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // ── 1. Render hemispheres ──
        let pixel_blocks = (pixels_per_hogel as u32 + threads - 1) / threads;
        let render_cfg = LaunchConfig {
            grid_dim: (row_hogels as u32, pixel_blocks, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let (grid_w, grid_h, hemi_res, spp, max_bounces, ambient) = (
            self.grid_w as i32, self.grid_h as i32, self.hemi_res as i32,
            self.spp as i32, self.max_bounces as i32, self.ambient,
        );
        unsafe {
            self.stream.launch_builder(&self.render_fn)
                .arg(&mut self.hemi_batch_dev)
                .arg(&grid_w).arg(&grid_h).arg(&hemi_res).arg(&spp)
                .arg(&BOX_W).arg(&BOX_H)
                .arg(&hogel_offset)
                .arg(&max_bounces).arg(&ambient)
                .launch(render_cfg)
                .expect("stream render failed");
        }
        self.stream.synchronize().expect("sync after render");
        timings.push(("render".into(), t.elapsed().as_secs_f64() * 1000.0));
        t = std::time::Instant::now();

        // ── 2. Extract target_amp ──
        let batch_i32 = row_hogels as i32;
        let total_amp = (row_hogels * pixels_per_hogel) as i32;
        let amp_offset: i64 = 0;
        {
            let ss = self.ss.as_mut().unwrap();
            unsafe {
                self.stream.launch_builder(&self.hemi_to_amp_fn)
                    .arg(&self.hemi_batch_dev)
                    .arg(&mut ss.target_amp)
                    .arg(&batch_i32)
                    .arg(&hemi_res)
                    .arg(&amp_offset)
                    .launch(launch_cfg(total_amp))
                    .expect("stream hemi_to_amp failed");
            }
        }
        self.stream.synchronize().expect("sync after hemi_to_amp");
        timings.push(("hemi_to_amp".into(), t.elapsed().as_secs_f64() * 1000.0));
        t = std::time::Instant::now();

        // ── 3. Init random phase ──
        let ss = self.ss.as_mut().unwrap();
        let total_phase: i64 = (row_hogels * pixels_per_hogel) as i64;
        let row_seed = noise_seed.wrapping_add(start_row as u64 * 0x9E3779B97F4A7C15);
        {
            let init_cfg = LaunchConfig {
                grid_dim: (((total_phase as u64 + threads as u64 - 1) / threads as u64) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream.launch_builder(&self.init_phase_fn)
                    .arg(&mut ss.phase)
                    .arg(&total_phase)
                    .arg(&row_seed)
                    .launch(init_cfg)
                    .expect("stream init_phase failed");
            }
        }
        self.stream.synchronize().expect("sync after init_phase");
        timings.push(("init_phase".into(), t.elapsed().as_secs_f64() * 1000.0));
        t = std::time::Instant::now();

        // ── 4. GS iterations ──
        let num_gs_batches = row_hogels / ss.gs_batch;
        let gs_batch_elems = (ss.gs_batch * pixels_per_hogel) as i32;

        for iter in 0..gs_iterations {
            let iter_t = std::time::Instant::now();
            for b in 0..num_gs_batches {
                let off = (b * ss.gs_batch * pixels_per_hogel) as i64;
                unsafe {
                    self.stream.launch_builder(&ss.build_complex_fn)
                        .arg(&ss.phase).arg(&mut ss.e_a).arg(&gs_batch_elems).arg(&off)
                        .launch(launch_cfg(gs_batch_elems)).expect("stream build_complex");
                }
                ss.plan.exec_c2c(&mut ss.e_a, &mut ss.e_b, FftDirection::Forward).expect("stream FFT fwd");
                unsafe {
                    self.stream.launch_builder(&ss.enforce_mag_fn)
                        .arg(&mut ss.e_b).arg(&ss.target_amp).arg(&gs_batch_elems).arg(&off)
                        .launch(launch_cfg(gs_batch_elems)).expect("stream enforce_mag");
                }
                ss.plan.exec_c2c(&mut ss.e_b, &mut ss.e_a, FftDirection::Inverse).expect("stream FFT inv");
                unsafe {
                    self.stream.launch_builder(&ss.extract_phase_fn)
                        .arg(&ss.e_a).arg(&mut ss.phase).arg(&gs_batch_elems).arg(&off)
                        .launch(launch_cfg(gs_batch_elems)).expect("stream extract_phase");
                }
            }
            self.stream.synchronize().expect("sync after GS iter");
            timings.push((format!("gs_iter_{}", iter), iter_t.elapsed().as_secs_f64() * 1000.0));
        }
        let gs_total: f64 = timings.iter()
            .filter(|(n, _)| n.starts_with("gs_iter_"))
            .map(|(_, v)| v).sum();
        timings.push(("gs_total".into(), gs_total));
        t = std::time::Instant::now();

        // ── 5. Apply panel model ──
        if phase_bits > 0 || noise_sigma > 0.0 {
            let panel_cfg = LaunchConfig {
                grid_dim: (((total_phase as u64 + threads as u64 - 1) / threads as u64) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream.launch_builder(&ss.apply_panel_fn)
                    .arg(&mut ss.phase)
                    .arg(&total_phase)
                    .arg(&(phase_bits as i32))
                    .arg(&noise_sigma)
                    .arg(&noise_seed)
                    .launch(panel_cfg)
                    .expect("stream apply_panel");
            }
            self.stream.synchronize().expect("sync after panel");
        }
        timings.push(("panel_model".into(), t.elapsed().as_secs_f64() * 1000.0));
        t = std::time::Instant::now();

        // ── 6. Forward project + scatter ──
        let norm_factor = 1.0f32 / (pixels_per_hogel as f32);
        let (out_w, out_h) = (self.out_w as i32, self.out_h as i32);
        let sigma_panel = self.sigma_panel;
        let half_w = self.half_w;
        let window_dim = (2 * half_w + 1) as u32;
        let window_size = window_dim * window_dim;
        let scatter_cfg = |b_size: u32| LaunchConfig {
            grid_dim: ((window_size + threads - 1) / threads, b_size, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let (eye_x, eye_y, eye_z) = (278.0f32, 273.0f32, -800.0f32);

        for b in 0..num_gs_batches {
            let off = (b * ss.gs_batch * pixels_per_hogel) as i64;
            let hogel_off = hogel_offset + (b * ss.gs_batch) as i32;

            unsafe {
                self.stream.launch_builder(&ss.build_complex_fn)
                    .arg(&ss.phase).arg(&mut ss.e_a).arg(&gs_batch_elems).arg(&off)
                    .launch(launch_cfg(gs_batch_elems)).expect("stream build_complex scatter");
            }
            ss.plan.exec_c2c(&mut ss.e_a, &mut ss.e_b, FftDirection::Forward).expect("stream FFT scatter");
            {
                let scratch = ss.intensity_scratch.slice(0..ss.gs_batch * pixels_per_hogel);
                unsafe {
                    self.stream.launch_builder(&ss.intensity_fn)
                        .arg(&ss.e_b).arg(&scratch)
                        .arg(&gs_batch_elems).arg(&norm_factor)
                        .launch(launch_cfg(gs_batch_elems)).expect("stream intensity");
                }
            }
            {
                let intensity_view = ss.intensity_scratch.slice(0..ss.gs_batch * pixels_per_hogel);
                let batch_u32 = ss.gs_batch as u32;
                unsafe {
                    self.stream.launch_builder(&self.scatter_fn)
                        .arg(&intensity_view)
                        .arg(&mut self.output_accum_dev)
                        .arg(&mut self.weight_accum_dev)
                        .arg(&hogel_off)
                        .arg(&(batch_u32 as i32))
                        .arg(&(self.grid_w as i32)).arg(&(self.grid_h as i32))
                        .arg(&hemi_res)
                        .arg(&out_w).arg(&out_h)
                        .arg(&BOX_W).arg(&BOX_H)
                        .arg(&eye_x).arg(&eye_y).arg(&eye_z)
                        .arg(&sigma_panel)
                        .arg(&half_w)
                        .launch(scatter_cfg(batch_u32))
                        .expect("stream scatter");
                }
            }
        }
        self.stream.synchronize().expect("sync after scatter");
        timings.push(("scatter".into(), t.elapsed().as_secs_f64() * 1000.0));

        let total: f64 = timings.iter()
            .filter(|(n, _)| !n.starts_with("gs_iter_") && n != "gs_total")
            .map(|(_, v)| v).sum::<f64>() + gs_total;
        timings.push(("total".into(), total));
        timings.push(("hogels".into(), row_hogels as f64));
        timings.push(("pixels_per_hogel".into(), pixels_per_hogel as f64));
        timings.push(("gs_batches".into(), num_gs_batches as f64));
        timings.push(("gs_batch_size".into(), ss.gs_batch as f64));

        eprintln!("[holosim] stream_profile: {} hogels, {} px/hog, {} gs_batches×{}, {} GS iters",
            row_hogels, pixels_per_hogel, num_gs_batches, ss.gs_batch, gs_iterations);
        for (name, ms) in &timings {
            if !name.starts_with("gs_iter_") {
                eprintln!("[holosim]   {}: {:.1}ms", name, ms);
            }
        }

        timings
    }

    // ── PCA eigenspectrum analysis ──────────────────────────

    /// PCA eigenspectrum (no logging). Same algorithm as pca_on_data but silent.
    fn pca_quiet(data: &[f32], n: usize, d_full: usize, max_components: usize) -> (Vec<f64>, Vec<f64>) {
        let max_hogels = 2048usize;
        let hogel_stride = if n > max_hogels { (n + max_hogels - 1) / max_hogels } else { 1 };
        let n_sub = (n + hogel_stride - 1) / hogel_stride;
        let k = max_components.min(n_sub);

        let max_pixels = 4096usize;
        let pixel_stride = if d_full > max_pixels { d_full / max_pixels } else { 1 };
        let d = (d_full + pixel_stride - 1) / pixel_stride;

        use nalgebra::{DMatrix, SymmetricEigen};
        let mut gram = DMatrix::<f64>::zeros(n_sub, n_sub);

        const BLOCK: usize = 256;
        for col_start in (0..d).step_by(BLOCK) {
            let col_end = (col_start + BLOCK).min(d);
            let bcols = col_end - col_start;
            let mut block = DMatrix::<f64>::zeros(n_sub, bcols);
            for (si, i) in (0..n).step_by(hogel_stride).enumerate() {
                for (bc, c) in (col_start..col_end).enumerate() {
                    let src_idx = i * d_full + c * pixel_stride;
                    block[(si, bc)] = data[src_idx] as f64;
                }
            }
            gram += &block * block.transpose();
        }
        gram /= n_sub as f64;

        let eig = SymmetricEigen::new(gram);
        let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let total_var: f64 = eigenvalues.iter().sum();
        let mut cumvar = Vec::with_capacity(k);
        let mut running = 0.0;
        for &ev in eigenvalues.iter().take(k) {
            running += ev;
            cumvar.push(if total_var > 0.0 { running / total_var } else { 0.0 });
        }
        let eigenvalues_top: Vec<f64> = eigenvalues.into_iter().take(k).collect();

        (eigenvalues_top, cumvar)
    }

    /// Generic PCA eigenspectrum on a flat n×d_full f32 buffer.
    /// Subsamples both hogels (rows) and pixels (columns) to keep computation feasible.
    fn pca_on_data(label: &str, data: &[f32], n: usize, d_full: usize, max_components: usize) -> (Vec<f64>, Vec<f64>) {
        let _t = std::time::Instant::now();

        // Subsample hogels if n is large (eigendecomp is O(n³))
        let max_hogels = 2048usize;
        let hogel_stride = if n > max_hogels { (n + max_hogels - 1) / max_hogels } else { 1 };
        let n_sub = (n + hogel_stride - 1) / hogel_stride;
        let k = max_components.min(n_sub);

        // Subsample pixels if d is large
        let max_pixels = 4096usize;
        let pixel_stride = if d_full > max_pixels { d_full / max_pixels } else { 1 };
        let d = (d_full + pixel_stride - 1) / pixel_stride;

        eprintln!("[holosim] PCA-{}: n={} (sub={}), d_full={} (sub={}), computing top {} eigenvalues",
            label, n, n_sub, d_full, d, k);

        use nalgebra::{DMatrix, SymmetricEigen};
        let mut gram = DMatrix::<f64>::zeros(n_sub, n_sub);

        const BLOCK: usize = 256;
        for col_start in (0..d).step_by(BLOCK) {
            let col_end = (col_start + BLOCK).min(d);
            let bcols = col_end - col_start;
            let mut block = DMatrix::<f64>::zeros(n_sub, bcols);
            for (si, i) in (0..n).step_by(hogel_stride).enumerate() {
                for (bc, c) in (col_start..col_end).enumerate() {
                    let src_idx = i * d_full + c * pixel_stride;
                    block[(si, bc)] = data[src_idx] as f64;
                }
            }
            gram += &block * block.transpose();
        }
        gram /= n_sub as f64;

        eprintln!("[holosim] PCA-{}: Gram {}×{} computed -> {}ms", label, n_sub, n_sub, _t.elapsed().as_millis());
        let _t2 = std::time::Instant::now();

        let eig = SymmetricEigen::new(gram);
        let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let total_var: f64 = eigenvalues.iter().sum();
        let mut cumvar = Vec::with_capacity(k);
        let mut running = 0.0;
        for &ev in eigenvalues.iter().take(k) {
            running += ev;
            cumvar.push(if total_var > 0.0 { running / total_var } else { 0.0 });
        }
        let eigenvalues_top: Vec<f64> = eigenvalues.into_iter().take(k).collect();

        eprintln!("[holosim] PCA-{}: eigendecomp done -> {}ms", label, _t2.elapsed().as_millis());
        eprintln!("[holosim] PCA-{}: top-10: {:?}", label, &eigenvalues_top[..10.min(k)]);
        eprintln!("[holosim] PCA-{}: cumVar k=10: {:.4}, k=50: {:.4}, k=100: {:.4}",
            label,
            cumvar.get(9).unwrap_or(&0.0),
            cumvar.get(49).unwrap_or(&0.0),
            cumvar.get(99).unwrap_or(&0.0));
        eprintln!("[holosim] PCA-{}: total -> {}ms", label, _t.elapsed().as_millis());

        (eigenvalues_top, cumvar)
    }

    /// PCA on GS phase patterns.
    pub fn pca_eigenspectrum(&mut self, max_components: usize) -> (Vec<f64>, Vec<f64>) {
        let gs = self.gs.as_ref().expect("PCA requires GS state");
        let n = (self.grid_w * self.grid_h) as usize;
        let d = gs.pixels_per_hogel;
        let data: Vec<f32> = self.stream.clone_dtoh(&gs.phase).expect("D2H phase");
        Self::pca_on_data("phase", &data, n, d, max_components)
    }

    /// PCA on target amplitude patterns (FFT magnitude of rendered hemispheres).
    pub fn pca_target_amp(&mut self, max_components: usize) -> (Vec<f64>, Vec<f64>) {
        let target = self.target_amp_dev.as_ref().expect("target_amp requires standard mode");
        let n = (self.grid_w * self.grid_h) as usize;
        let d = (self.hemi_res * self.hemi_res) as usize;
        let data: Vec<f32> = self.stream.clone_dtoh(target).expect("D2H target_amp");
        Self::pca_on_data("target_amp", &data, n, d, max_components)
    }

    /// PCA on raw hemisphere images (rendered ray-traced views per hogel).
    /// Analyzes the RGB average per pixel.
    pub fn pca_hemisphere(&mut self, max_components: usize) -> (Vec<f64>, Vec<f64>) {
        let n = (self.grid_w * self.grid_h) as usize;
        let d = (self.hemi_res * self.hemi_res) as usize;
        let hemi_f32: Vec<f32> = self.stream.clone_dtoh(&self.hemi_batch_dev).expect("D2H hemi");
        // hemi is RGB (3 channels per pixel), average to single channel for PCA
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                let base = (i * d + j) * 3;
                data[i * d + j] = (hemi_f32[base] + hemi_f32[base + 1] + hemi_f32[base + 2]) / 3.0;
            }
        }
        Self::pca_on_data("hemisphere", &data, n, d, max_components)
    }

    /// Save target_amp from GPU to CPU backup (for restore after PCA compress overwrites it).
    pub fn save_target_amp(&mut self) {
        let target = self.target_amp_dev.as_ref().expect("standard mode only");
        let data: Vec<f32> = self.stream.clone_dtoh(target).expect("D2H target_amp save");
        eprintln!("[holosim] save_target_amp: {} floats ({:.1} MB)", data.len(), data.len() as f64 * 4.0 / 1e6);
        self.target_amp_backup = Some(data);
    }

    /// Restore target_amp from CPU backup to GPU.
    pub fn restore_target_amp(&mut self) {
        let backup = self.target_amp_backup.as_ref().expect("no target_amp backup saved");
        let target = self.target_amp_dev.as_mut().expect("standard mode only");
        self.stream.memcpy_htod(backup.as_slice(), target).expect("H2D target_amp restore");
        self.stream.synchronize().expect("sync");
    }

    /// PCA compress/decompress target_amp in place with top-k components.
    /// Returns (k_used, compression_ratio, rmse).
    /// After this call, target_amp_dev contains the PCA-reconstructed approximation.
    pub fn pca_compress_target_amp(&mut self, k: usize) -> (usize, f64, f64) {
        use rayon::prelude::*;
        let _t = std::time::Instant::now();

        let target = self.target_amp_dev.as_ref().expect("standard mode only");
        let n = (self.grid_w * self.grid_h) as usize;
        let d = (self.hemi_res * self.hemi_res) as usize;
        let k = k.min(n).min(d);

        eprintln!("[holosim] PCA-compress: n={}, d={}, k={}", n, d, k);

        // D2H
        let data: Vec<f32> = self.stream.clone_dtoh(target).expect("D2H target_amp");
        assert_eq!(data.len(), n * d);
        eprintln!("[holosim] PCA-compress: D2H done -> {}ms", _t.elapsed().as_millis());

        use nalgebra::{DMatrix, SymmetricEigen};

        // Compute mean (per-pixel across all hogels) — parallelized over pixels
        let mean: Vec<f64> = (0..d).into_par_iter().map(|j| {
            let mut s = 0.0f64;
            for i in 0..n { s += data[i * d + j] as f64; }
            s / n as f64
        }).collect();

        // Build centered Gram matrix G = X_c X_c^T / N in blocks
        let mut gram = DMatrix::<f64>::zeros(n, n);
        const BLOCK: usize = 512;
        for col_start in (0..d).step_by(BLOCK) {
            let col_end = (col_start + BLOCK).min(d);
            let bcols = col_end - col_start;
            let mut block = DMatrix::<f64>::zeros(n, bcols);
            for i in 0..n {
                for (bc, c) in (col_start..col_end).enumerate() {
                    block[(i, bc)] = data[i * d + c] as f64 - mean[c];
                }
            }
            gram += &block * block.transpose();
        }
        gram /= n as f64;
        eprintln!("[holosim] PCA-compress: Gram done -> {}ms", _t.elapsed().as_millis());

        // Eigendecompose
        let eig = SymmetricEigen::new(gram);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());

        // Extract top-k eigenvectors of Gram (n×k)
        let mut q = DMatrix::<f64>::zeros(n, k);
        for (ki, &idx) in indices.iter().take(k).enumerate() {
            for i in 0..n {
                q[(i, ki)] = eig.eigenvectors[(i, idx)];
            }
        }

        // Precompute Λ^{-1/2} for scaling
        let inv_sqrt_lam: Vec<f64> = indices.iter().take(k).map(|&idx| {
            let lam = eig.eigenvalues[idx];
            if lam > 1e-10 { 1.0 / (n as f64 * lam).sqrt() } else { 0.0 }
        }).collect();
        eprintln!("[holosim] PCA-compress: eigendecomp done -> {}ms", _t.elapsed().as_millis());

        // Compute full-resolution eigenvectors: V = X_c^T Q Λ^{-1/2}  (d × k)
        // v_flat[c * k + ki] = eigenvector component
        let mut v_flat = vec![0.0f64; d * k];
        v_flat.par_chunks_mut(k).enumerate().for_each(|(c, row)| {
            let mean_c = mean[c];
            for ki in 0..k {
                let mut sum = 0.0f64;
                for i in 0..n {
                    sum += (data[i * d + c] as f64 - mean_c) * q[(i, ki)];
                }
                row[ki] = sum * inv_sqrt_lam[ki];
            }
        });
        // v_flat is d×k in row-major (pixel, component)
        eprintln!("[holosim] PCA-compress: eigenvectors done -> {}ms", _t.elapsed().as_millis());

        // Project: coefficients = X_c × V  (n × k) — parallelized over hogels
        let mut coeffs_flat = vec![0.0f64; n * k];
        coeffs_flat.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
            for ki in 0..k {
                let mut sum = 0.0f64;
                for j in 0..d {
                    sum += (data[i * d + j] as f64 - mean[j]) * v_flat[j * k + ki];
                }
                row[ki] = sum;
            }
        });
        eprintln!("[holosim] PCA-compress: projection done -> {}ms", _t.elapsed().as_millis());

        // Reconstruct: X_hat = coeffs × V^T + mean — parallelized over hogels
        let mut recon = vec![0.0f32; n * d];
        let sse: f64 = recon.par_chunks_mut(d).enumerate().map(|(i, row)| {
            let mut sse_local = 0.0f64;
            for j in 0..d {
                let mut val = mean[j];
                for ki in 0..k {
                    val += coeffs_flat[i * k + ki] * v_flat[j * k + ki];
                }
                row[j] = val as f32;
                let diff = data[i * d + j] as f64 - val;
                sse_local += diff * diff;
            }
            sse_local
        }).sum();
        let rmse = (sse / (n * d) as f64).sqrt();

        // Compression ratio
        let original_bytes = (n * d * 4) as f64;
        let compressed_bytes = ((k * d + n * k) * 4 + d * 8) as f64; // basis + coefficients + mean
        let ratio = original_bytes / compressed_bytes;

        eprintln!("[holosim] PCA-compress: reconstruct done -> {}ms", _t.elapsed().as_millis());
        eprintln!("[holosim] PCA-compress: k={}, RMSE={:.6}, ratio={:.1}x, original={:.1}MB, compressed={:.1}MB",
            k, rmse, ratio, original_bytes / 1e6, compressed_bytes / 1e6);

        // H2D the reconstructed data back to GPU
        let target_mut = self.target_amp_dev.as_mut().expect("standard mode");
        self.stream.memcpy_htod(&recon, target_mut).expect("H2D recon target_amp");
        self.stream.synchronize().expect("sync");

        eprintln!("[holosim] PCA-compress: total -> {}ms", _t.elapsed().as_millis());

        (k, ratio, rmse)
    }

    /// Enable PCA sampling during streaming. Randomly selects `num_samples` hogel indices.
    /// Must be called before streaming begins.
    pub fn enable_pca_sampling(&mut self, num_samples: usize) {
        let total_hogels = (self.grid_w * self.grid_h) as usize;
        let num_samples = num_samples.min(total_hogels);
        let pixels_per_hogel = (self.hemi_res * self.hemi_res) as usize;

        // Deterministic sampling: evenly spaced + small jitter from hash
        let mut indices: Vec<usize> = if num_samples >= total_hogels {
            (0..total_hogels).collect()
        } else {
            let step = total_hogels as f64 / num_samples as f64;
            (0..num_samples).map(|i| ((i as f64 * step) as usize).min(total_hogels - 1)).collect()
        };
        indices.sort();
        indices.dedup();

        let actual_samples = indices.len();
        let data = vec![0.0f32; actual_samples * pixels_per_hogel];

        eprintln!("[holosim] PCA-sampling enabled: {} hogels sampled from {}, buffer={:.1} MB",
            actual_samples, total_hogels, (actual_samples * pixels_per_hogel * 4) as f64 / 1e6);

        self.pca_sample_indices = Some(indices);
        self.pca_sample_data = Some(data);
    }

    /// Called during stream_sub_batch after target_amp is computed.
    /// Copies sampled hogels' target_amp from GPU to CPU buffer.
    fn collect_pca_samples(&mut self, hogel_offset: usize, num_hogels: usize) {
        let indices = match &self.pca_sample_indices {
            Some(v) => v,
            None => return,
        };
        let pixels_per_hogel = (self.hemi_res * self.hemi_res) as usize;
        let hogel_end = hogel_offset + num_hogels;

        // Find which sample indices fall in [hogel_offset, hogel_end)
        let start_idx = indices.partition_point(|&i| i < hogel_offset);
        let end_idx = indices.partition_point(|&i| i < hogel_end);

        if start_idx >= end_idx { return; }

        let ss = self.ss.as_ref().unwrap();

        // D2H the target_amp for these hogels
        for sample_pos in start_idx..end_idx {
            let global_hogel = indices[sample_pos];
            let local_hogel = global_hogel - hogel_offset;
            // Read from ss.target_amp at local offset
            let src_offset = local_hogel * pixels_per_hogel;
            let src_slice = ss.target_amp.slice(src_offset..src_offset + pixels_per_hogel);
            let mut tmp = vec![0.0f32; pixels_per_hogel];
            self.stream.memcpy_dtoh(&src_slice, &mut tmp[..]).expect("D2H PCA sample");
            // Store in pca_sample_data
            let data = self.pca_sample_data.as_mut().unwrap();
            let dst_offset = sample_pos * pixels_per_hogel;
            data[dst_offset..dst_offset + pixels_per_hogel].copy_from_slice(&tmp);
        }
    }

    /// Run PCA on the accumulated streaming samples.
    /// Returns (eigenvalues, cum_var) like pca_on_data.
    pub fn pca_streaming_eigenspectrum(&self, max_components: usize) -> (Vec<f64>, Vec<f64>) {
        let data = self.pca_sample_data.as_ref().expect("PCA sampling not enabled");
        let indices = self.pca_sample_indices.as_ref().unwrap();
        let n = indices.len();
        let d = (self.hemi_res * self.hemi_res) as usize;
        eprintln!("[holosim] PCA-streaming: n={} samples, d={} pixels, max_components={}",
            n, d, max_components);
        Self::pca_on_data("streaming_target_amp", data, n, d, max_components)
    }

    /// Number of PCA samples collected.
    pub fn pca_num_samples(&self) -> usize {
        self.pca_sample_indices.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    /// Tiled PCA analysis on target_amp. Partitions grid into spatial tiles,
    /// runs PCA per tile in parallel, and compares storage with flat PCA.
    /// Returns (TiledPcaInfo) with aggregate statistics.
    pub fn pca_tiled_target_amp(&self, tile_size: usize, max_components: usize) -> TiledPcaInfo {
        use rayon::prelude::*;
        let n_h = self.grid_h as usize;
        let n_w = self.grid_w as usize;
        let d = (self.hemi_res * self.hemi_res) as usize;
        let total_hogels = n_h * n_w;

        let t0 = std::time::Instant::now();

        // D2H full target_amp
        let target_amp_dev = self.target_amp_dev.as_ref().expect("target_amp requires standard mode");
        let data: Vec<f32> = self.stream.clone_dtoh(target_amp_dev).expect("D2H target_amp");
        eprintln!("[holosim] PCA-tiled: D2H {}×{} grid, hemi={}, tile={}, {:.1} MB -> {}ms",
            n_w, n_h, self.hemi_res, tile_size,
            data.len() as f64 * 4.0 / 1e6, t0.elapsed().as_millis());

        let tiles_h = (n_h + tile_size - 1) / tile_size;
        let tiles_w = (n_w + tile_size - 1) / tile_size;
        let num_tiles = tiles_h * tiles_w;

        // Build tile specs: (ty, tx)
        let tile_specs: Vec<(usize, usize)> = (0..tiles_h).flat_map(|ty| {
            (0..tiles_w).map(move |tx| (ty, tx))
        }).collect();

        // Parallel PCA per tile
        let tile_results: Vec<(usize, Vec<f64>, Vec<f64>)> = tile_specs.par_iter().map(|&(ty, tx)| {
            let mut tile_data = Vec::new();
            let mut tile_n = 0usize;
            for row in (ty * tile_size)..((ty + 1) * tile_size).min(n_h) {
                for col in (tx * tile_size)..((tx + 1) * tile_size).min(n_w) {
                    let hogel_idx = row * n_w + col;
                    let offset = hogel_idx * d;
                    tile_data.extend_from_slice(&data[offset..offset + d]);
                    tile_n += 1;
                }
            }
            let mc = max_components.min(tile_n);
            let (eig, cv) = Self::pca_quiet(&tile_data, tile_n, d, mc);
            (tile_n, eig, cv)
        }).collect();

        eprintln!("[holosim] PCA-tiled: {} tiles done -> {}ms", num_tiles, t0.elapsed().as_millis());

        // Flat PCA for comparison
        let (flat_eig, flat_cv) = Self::pca_on_data("flat_target_amp", &data, total_hogels, d, max_components);
        eprintln!("[holosim] PCA-tiled: flat PCA done -> {}ms", t0.elapsed().as_millis());

        // Compute storage comparison for accuracy targets
        let targets = vec![0.9, 0.95, 0.99, 0.999];
        let original_bytes = (total_hogels * d * 4) as f64;

        let mut tile_median_k = Vec::new();
        let mut tile_max_k = Vec::new();
        let mut tile_min_k = Vec::new();
        let mut tile_mean_k = Vec::new();
        let mut tiled_total_bytes = Vec::new();
        let mut flat_k_vec = Vec::new();
        let mut flat_total_bytes = Vec::new();
        let mut per_tile_k_flat: Vec<u32> = Vec::new();

        for &target in &targets {
            let mut ks: Vec<usize> = Vec::new();
            for (tile_n, _, ref cv) in &tile_results {
                let k = cv.iter().position(|&v| v >= target).map(|i| i + 1).unwrap_or(*tile_n);
                ks.push(k);
            }

            let mut sorted_ks = ks.clone();
            sorted_ks.sort();
            let median = sorted_ks[sorted_ks.len() / 2];
            let max_val = *sorted_ks.last().unwrap();
            let min_val = sorted_ks[0];
            let mean = ks.iter().sum::<usize>() as f64 / ks.len() as f64;

            // Tiled storage: per tile, basis = k × d × 4, coeffs = n_tile × k × 4
            let t_basis: usize = ks.iter().map(|&k| k * d * 4).sum();
            let t_coeffs: usize = tile_results.iter().zip(ks.iter())
                .map(|((n, _, _), &k)| n * k * 4).sum();
            let t_total = (t_basis + t_coeffs) as f64;

            // Flat storage
            let fk = flat_cv.iter().position(|&v| v >= target).map(|i| i + 1).unwrap_or(flat_cv.len());
            let f_basis = fk * d * 4;
            let f_coeffs = total_hogels * fk * 4;
            let f_total = (f_basis + f_coeffs) as f64;

            tile_median_k.push(median as u32);
            tile_max_k.push(max_val as u32);
            tile_min_k.push(min_val as u32);
            tile_mean_k.push(mean);
            tiled_total_bytes.push(t_total);
            flat_k_vec.push(fk as u32);
            flat_total_bytes.push(f_total);

            // Store per-tile k values for this target
            for &k in &ks {
                per_tile_k_flat.push(k as u32);
            }

            eprintln!("[holosim] PCA-tiled: target={:.1}%: tile k median={}, max={}, min={}, mean={:.1} | flat k={}",
                target * 100.0, median, max_val, min_val, mean, fk);
            eprintln!("[holosim]   tiled: {:.1} MB ({:.1}× compression) | flat: {:.1} MB ({:.1}× compression)",
                t_total / 1e6, original_bytes / t_total,
                f_total / 1e6, original_bytes / f_total);
        }

        eprintln!("[holosim] PCA-tiled: total -> {}ms", t0.elapsed().as_millis());

        TiledPcaInfo {
            num_tiles: num_tiles as u32,
            tile_size: tile_size as u32,
            hogels_per_tile: (tile_size.min(n_h) * tile_size.min(n_w)) as u32,
            pixels_per_hogel: d as u32,
            total_hogels: total_hogels as u32,
            targets,
            tile_median_k,
            tile_max_k,
            tile_min_k,
            tile_mean_k,
            tiled_total_bytes,
            flat_k: flat_k_vec,
            flat_total_bytes,
            original_bytes,
            per_tile_k: per_tile_k_flat,
            flat_eigenvalues: flat_eig,
            flat_cum_var: flat_cv,
        }
    }

    /// PCA compress with optional coefficient quantization.
    /// bits=0 means float32 (no quantization). bits=8 means per-component min/max int8.
    /// Returns (k_used, compression_ratio, rmse, quantized_bytes).
    pub fn pca_compress_quantized_target_amp(&mut self, k: usize, bits: u32) -> (usize, f64, f64, f64) {
        use rayon::prelude::*;
        let _t = std::time::Instant::now();

        let target = self.target_amp_dev.as_ref().expect("standard mode only");
        let n = (self.grid_w * self.grid_h) as usize;
        let d = (self.hemi_res * self.hemi_res) as usize;
        let k = k.min(n).min(d);

        eprintln!("[holosim] PCA-compress-q{}: n={}, d={}, k={}", bits, n, d, k);

        // D2H
        let data: Vec<f32> = self.stream.clone_dtoh(target).expect("D2H target_amp");
        assert_eq!(data.len(), n * d);

        use nalgebra::{DMatrix, SymmetricEigen};

        // Compute mean (per-pixel across all hogels)
        let mean: Vec<f64> = (0..d).into_par_iter().map(|j| {
            let mut s = 0.0f64;
            for i in 0..n { s += data[i * d + j] as f64; }
            s / n as f64
        }).collect();

        // Build centered Gram matrix
        let mut gram = DMatrix::<f64>::zeros(n, n);
        const BLOCK: usize = 512;
        for col_start in (0..d).step_by(BLOCK) {
            let col_end = (col_start + BLOCK).min(d);
            let bcols = col_end - col_start;
            let mut block = DMatrix::<f64>::zeros(n, bcols);
            for i in 0..n {
                for (bc, c) in (col_start..col_end).enumerate() {
                    block[(i, bc)] = data[i * d + c] as f64 - mean[c];
                }
            }
            gram += &block * block.transpose();
        }
        gram /= n as f64;
        eprintln!("[holosim] PCA-compress-q{}: Gram done -> {}ms", bits, _t.elapsed().as_millis());

        // Eigendecompose
        let eig = SymmetricEigen::new(gram);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());

        // Extract top-k eigenvectors
        let mut q = DMatrix::<f64>::zeros(n, k);
        for (ki, &idx) in indices.iter().take(k).enumerate() {
            for i in 0..n {
                q[(i, ki)] = eig.eigenvectors[(i, idx)];
            }
        }

        let inv_sqrt_lam: Vec<f64> = indices.iter().take(k).map(|&idx| {
            let lam = eig.eigenvalues[idx];
            if lam > 1e-10 { 1.0 / (n as f64 * lam).sqrt() } else { 0.0 }
        }).collect();

        // Full-resolution eigenvectors: V = X_c^T Q Λ^{-1/2}  (d × k)
        let mut v_flat = vec![0.0f64; d * k];
        v_flat.par_chunks_mut(k).enumerate().for_each(|(c, row)| {
            let mean_c = mean[c];
            for ki in 0..k {
                let mut sum = 0.0f64;
                for i in 0..n {
                    sum += (data[i * d + c] as f64 - mean_c) * q[(i, ki)];
                }
                row[ki] = sum * inv_sqrt_lam[ki];
            }
        });
        eprintln!("[holosim] PCA-compress-q{}: eigenvectors done -> {}ms", bits, _t.elapsed().as_millis());

        // Project: coefficients = X_c × V  (n × k)
        let mut coeffs_flat = vec![0.0f64; n * k];
        coeffs_flat.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
            for ki in 0..k {
                let mut sum = 0.0f64;
                for j in 0..d {
                    sum += (data[i * d + j] as f64 - mean[j]) * v_flat[j * k + ki];
                }
                row[ki] = sum;
            }
        });
        eprintln!("[holosim] PCA-compress-q{}: projection done -> {}ms", bits, _t.elapsed().as_millis());

        // Quantize coefficients if bits > 0
        if bits > 0 && bits <= 16 {
            let levels = (1u64 << bits) as f64;
            // Per-component min/max quantization
            for ki in 0..k {
                let mut min_val = f64::MAX;
                let mut max_val = f64::MIN;
                for i in 0..n {
                    let v = coeffs_flat[i * k + ki];
                    if v < min_val { min_val = v; }
                    if v > max_val { max_val = v; }
                }
                let range = max_val - min_val;
                if range > 1e-15 {
                    let scale = (levels - 1.0) / range;
                    for i in 0..n {
                        let v = coeffs_flat[i * k + ki];
                        // Quantize: round to nearest integer level, then dequantize
                        let q_val = ((v - min_val) * scale).round();
                        coeffs_flat[i * k + ki] = q_val / scale + min_val;
                    }
                }
            }
            eprintln!("[holosim] PCA-compress-q{}: quantized {} coefficients -> {}ms",
                bits, n * k, _t.elapsed().as_millis());
        }

        // Reconstruct: X_hat = coeffs × V^T + mean
        let mut recon = vec![0.0f32; n * d];
        let sse: f64 = recon.par_chunks_mut(d).enumerate().map(|(i, row)| {
            let mut sse_local = 0.0f64;
            for j in 0..d {
                let mut val = mean[j];
                for ki in 0..k {
                    val += coeffs_flat[i * k + ki] * v_flat[j * k + ki];
                }
                row[j] = val as f32;
                let diff = data[i * d + j] as f64 - val;
                sse_local += diff * diff;
            }
            sse_local
        }).sum();
        let rmse = (sse / (n * d) as f64).sqrt();

        // Storage calculation
        let original_bytes = (n * d * 4) as f64;
        let mean_bytes = d * 8; // f64
        let basis_bytes = k * d * 4; // f32 eigenvectors
        let coeff_bytes = if bits > 0 && bits <= 16 {
            // quantized: bits per coefficient + 2 f64 (min, max) per component for dequant
            (n * k * bits as usize + 7) / 8 + k * 16
        } else {
            n * k * 4
        };
        let compressed_bytes = (mean_bytes + basis_bytes + coeff_bytes) as f64;
        let ratio = original_bytes / compressed_bytes;

        eprintln!("[holosim] PCA-compress-q{}: k={}, RMSE={:.6}, ratio={:.1}x, orig={:.1}MB, comp={:.1}MB",
            bits, k, rmse, ratio, original_bytes / 1e6, compressed_bytes / 1e6);
        eprintln!("[holosim] PCA-compress-q{}: total -> {}ms", bits, _t.elapsed().as_millis());

        // H2D reconstructed data
        let target_mut = self.target_amp_dev.as_mut().expect("standard mode");
        self.stream.memcpy_htod(&recon, target_mut).expect("H2D recon");
        self.stream.synchronize().expect("sync");

        (k, ratio, rmse, compressed_bytes)
    }

    /// Batch quantization sweep: compute PCA once, then test multiple (k, bits) combos.
    /// Returns Vec of (k, bits, compression_ratio, rmse, compressed_bytes).
    pub fn pca_quantization_sweep(
        &self,
        k_values: &[usize],
        bit_values: &[u32],
    ) -> Vec<(usize, u32, f64, f64, f64)> {
        use rayon::prelude::*;
        let _t = std::time::Instant::now();

        let target = self.target_amp_dev.as_ref().expect("standard mode only");
        let n = (self.grid_w * self.grid_h) as usize;
        let d = (self.hemi_res * self.hemi_res) as usize;
        let max_k = *k_values.iter().max().unwrap_or(&1);
        let max_k = max_k.min(n).min(d);

        eprintln!("[holosim] PCA-sweep: n={}, d={}, max_k={}, {} k-vals × {} bit-vals",
            n, d, max_k, k_values.len(), bit_values.len());

        // D2H
        let data: Vec<f32> = self.stream.clone_dtoh(target).expect("D2H target_amp");
        assert_eq!(data.len(), n * d);

        use nalgebra::{DMatrix, SymmetricEigen};

        // Compute mean
        let mean: Vec<f64> = (0..d).into_par_iter().map(|j| {
            let mut s = 0.0f64;
            for i in 0..n { s += data[i * d + j] as f64; }
            s / n as f64
        }).collect();

        // Build Gram matrix
        let mut gram = DMatrix::<f64>::zeros(n, n);
        const BLOCK: usize = 512;
        for col_start in (0..d).step_by(BLOCK) {
            let col_end = (col_start + BLOCK).min(d);
            let bcols = col_end - col_start;
            let mut block = DMatrix::<f64>::zeros(n, bcols);
            for i in 0..n {
                for (bc, c) in (col_start..col_end).enumerate() {
                    block[(i, bc)] = data[i * d + c] as f64 - mean[c];
                }
            }
            gram += &block * block.transpose();
        }
        gram /= n as f64;
        eprintln!("[holosim] PCA-sweep: Gram done -> {}ms", _t.elapsed().as_millis());

        // Eigendecompose
        let eig = SymmetricEigen::new(gram);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());
        eprintln!("[holosim] PCA-sweep: eigendecomp done -> {}ms", _t.elapsed().as_millis());

        // Extract top-max_k eigenvectors of Gram
        let mut q = DMatrix::<f64>::zeros(n, max_k);
        for (ki, &idx) in indices.iter().take(max_k).enumerate() {
            for i in 0..n {
                q[(i, ki)] = eig.eigenvectors[(i, idx)];
            }
        }
        let inv_sqrt_lam: Vec<f64> = indices.iter().take(max_k).map(|&idx| {
            let lam = eig.eigenvalues[idx];
            if lam > 1e-10 { 1.0 / (n as f64 * lam).sqrt() } else { 0.0 }
        }).collect();

        // Full-resolution eigenvectors V (d × max_k)
        let mut v_flat = vec![0.0f64; d * max_k];
        v_flat.par_chunks_mut(max_k).enumerate().for_each(|(c, row)| {
            let mean_c = mean[c];
            for ki in 0..max_k {
                let mut sum = 0.0f64;
                for i in 0..n { sum += (data[i * d + c] as f64 - mean_c) * q[(i, ki)]; }
                row[ki] = sum * inv_sqrt_lam[ki];
            }
        });
        eprintln!("[holosim] PCA-sweep: eigenvectors done -> {}ms", _t.elapsed().as_millis());

        // Full coefficients (n × max_k)
        let mut coeffs_flat = vec![0.0f64; n * max_k];
        coeffs_flat.par_chunks_mut(max_k).enumerate().for_each(|(i, row)| {
            for ki in 0..max_k {
                let mut sum = 0.0f64;
                for j in 0..d { sum += (data[i * d + j] as f64 - mean[j]) * v_flat[j * max_k + ki]; }
                row[ki] = sum;
            }
        });
        eprintln!("[holosim] PCA-sweep: projection done -> {}ms", _t.elapsed().as_millis());

        // Now sweep (k, bits) combinations
        let original_bytes = (n * d * 4) as f64;
        let mut results = Vec::new();

        for &k in k_values {
            let k = k.min(max_k);
            for &bits in bit_values {
                // Quantize coefficients for this k and bits
                let mut q_coeffs = vec![0.0f64; n * k];
                for i in 0..n {
                    for ki in 0..k {
                        q_coeffs[i * k + ki] = coeffs_flat[i * max_k + ki];
                    }
                }

                if bits > 0 && bits <= 16 {
                    let levels = (1u64 << bits) as f64;
                    for ki in 0..k {
                        let mut min_val = f64::MAX;
                        let mut max_val = f64::MIN;
                        for i in 0..n {
                            let v = q_coeffs[i * k + ki];
                            if v < min_val { min_val = v; }
                            if v > max_val { max_val = v; }
                        }
                        let range = max_val - min_val;
                        if range > 1e-15 {
                            let scale = (levels - 1.0) / range;
                            for i in 0..n {
                                let v = q_coeffs[i * k + ki];
                                let q_val = ((v - min_val) * scale).round();
                                q_coeffs[i * k + ki] = q_val / scale + min_val;
                            }
                        }
                    }
                }

                // Compute RMSE (parallelized over hogels)
                let sse: f64 = (0..n).into_par_iter().map(|i| {
                    let mut sse_local = 0.0f64;
                    for j in 0..d {
                        let mut val = mean[j];
                        for ki in 0..k {
                            val += q_coeffs[i * k + ki] * v_flat[j * max_k + ki];
                        }
                        let diff = data[i * d + j] as f64 - val;
                        sse_local += diff * diff;
                    }
                    sse_local
                }).sum();
                let rmse = (sse / (n * d) as f64).sqrt();

                // Storage
                let mean_bytes = d * 8;
                let basis_bytes = k * d * 4;
                let coeff_bytes = if bits > 0 && bits <= 16 {
                    (n * k * bits as usize + 7) / 8 + k * 16
                } else {
                    n * k * 4
                };
                let compressed_bytes = (mean_bytes + basis_bytes + coeff_bytes) as f64;
                let ratio = original_bytes / compressed_bytes;

                let label = if bits > 0 { format!("{}b", bits) } else { "f32".to_string() };
                eprintln!("[holosim] PCA-sweep: k={}, {}: RMSE={:.6}, {:.1}x, {:.2}MB -> {}ms",
                    k, label, rmse, ratio, compressed_bytes / 1e6, _t.elapsed().as_millis());

                results.push((k, bits, ratio, rmse, compressed_bytes));
            }
        }

        eprintln!("[holosim] PCA-sweep: {} combos done -> {}ms", results.len(), _t.elapsed().as_millis());
        results
    }
}

/// Result struct for tiled PCA analysis.
pub struct TiledPcaInfo {
    pub num_tiles: u32,
    pub tile_size: u32,
    pub hogels_per_tile: u32,
    pub pixels_per_hogel: u32,
    pub total_hogels: u32,
    pub targets: Vec<f64>,
    pub tile_median_k: Vec<u32>,
    pub tile_max_k: Vec<u32>,
    pub tile_min_k: Vec<u32>,
    pub tile_mean_k: Vec<f64>,
    pub tiled_total_bytes: Vec<f64>,
    pub flat_k: Vec<u32>,
    pub flat_total_bytes: Vec<f64>,
    pub original_bytes: f64,
    pub per_tile_k: Vec<u32>,
    pub flat_eigenvalues: Vec<f64>,
    pub flat_cum_var: Vec<f64>,
}

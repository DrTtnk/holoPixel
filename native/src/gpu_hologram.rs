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
// ── Session-based batched GPU pipeline ────────────────────

use cudarc::driver::CudaSlice;

const PREVIEW_RES: u32 = 64;  // hogel thumbnail resolution (rows × cols)
const MAX_BATCH_HOGELS: usize = 1024;

/// Gerchberg-Saxton working state. Allocated once at `gs_setup`, reused per iteration.
/// Does NOT own target_amp (lives in GpuHologramSession), only owns per-batch working buffers.
pub struct GsState {
    plan: CudaFft,
    batch: usize,
    pixels_per_hogel: usize,
    num_hogels: usize,
    phase: CudaSlice<f32>,          // [num_hogels × hemi_res²] – all phases, VRAM
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

pub struct GpuHologramSession {
    pub grid_w: u32,
    pub grid_h: u32,
    pub hemi_res: u32,
    pub spp: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub max_bounces: u32,
    pub ambient: f32,
    // Derived geometry constants (computed once from grid / output dimensions)
    sigma_panel: f32,  // Gaussian σ in panel coords (mm)
    half_w: i32,       // scatter window half-size in output pixels
    stream: Arc<CudaStream>,
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
    preview_dev: CudaSlice<f32>,
    /// Per-hogel GS target amplitude (num_hogels × hemi_res²). Written during render.
    target_amp_dev: CudaSlice<f32>,
    /// Float accumulator for scatter output (out_w × out_h).
    output_accum_dev: CudaSlice<f32>,
    /// Gaussian weight accumulator (out_w × out_h).
    weight_accum_dev: CudaSlice<f32>,
    /// Final RGBA output (out_w × out_h × 4 u8).
    recon_dev: CudaSlice<u8>,
    pub rows_rendered: u32,
    pub gs: Option<GsState>,
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
        let preview_dev     = stream.alloc_zeros::<f32>(num_hogels * (PREVIEW_RES * PREVIEW_RES) as usize * 3).expect("alloc preview");
        let target_amp_dev  = stream.alloc_zeros::<f32>(num_hogels * pixels_per_hogel).expect("alloc target_amp");
        let output_accum_dev = stream.alloc_zeros::<f32>(out_pixels).expect("alloc output_accum");
        let weight_accum_dev = stream.alloc_zeros::<f32>(out_pixels).expect("alloc weight_accum");
        let recon_dev       = stream.alloc_zeros::<u8>(out_pixels * 4).expect("alloc recon");

        // Precompute scatter geometry
        let cell_w = BOX_W / grid_w as f32;
        let cell_h = BOX_H / grid_h as f32;
        let sigma_panel = 0.7f32 * cell_w.min(cell_h);
        let sigma_out_x = sigma_panel / BOX_W * out_w as f32;
        let half_w = (2.5f32 * sigma_out_x).ceil() as i32 + 1;

        Self {
            grid_w, grid_h, hemi_res, spp, out_w, out_h, max_bounces, ambient,
            sigma_panel, half_w,
            stream, module,
            render_fn, downsample_fn, hemi_to_amp_fn, scatter_fn, normalize_fn, init_phase_fn,
            hemi_batch_dev, preview_dev, target_amp_dev,
            output_accum_dev, weight_accum_dev, recon_dev,
            rows_rendered: 0,
            gs: None,
        }
    }

    /// Render `num_rows` rows of hogels. Populates `preview_dev` thumbnails and
    /// `target_amp_dev` for GS. Returns total rows rendered so far.
    pub fn render_rows(&mut self, start_row: u32, num_rows: u32) -> u32 {
        let _t = std::time::Instant::now();
        let end_row = (start_row + num_rows).min(self.grid_h);
        let actual_rows = end_row.saturating_sub(start_row);
        if actual_rows == 0 { return self.rows_rendered; }

        let row_hogels = (actual_rows * self.grid_w) as usize;
        let pixels_per_hogel = (self.hemi_res * self.hemi_res) as usize;
        let hogel_offset = (start_row * self.grid_w) as i32;

        let threads = 256u32;
        let pixel_blocks = (pixels_per_hogel as u32 + threads - 1) / threads;
        let render_cfg = LaunchConfig {
            grid_dim: (row_hogels as u32, pixel_blocks, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // 1. Render batch of hogels into hemi_batch_dev (local offset 0)
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

        // 2. Downsample batch → preview thumbnails at correct offset
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
                .arg(&mut self.preview_dev)
                .arg(&batch_i32)
                .arg(&hemi_res)
                .arg(&preview_res_i32)
                .arg(&hogel_offset)
                .launch(downsample_cfg)
                .expect("Downsample launch failed");
        }

        // 3. Extract grayscale target_amp for this batch, write at correct offset in target_amp_dev
        let total_amp = (row_hogels * pixels_per_hogel) as i32;
        let amp_cfg = LaunchConfig {
            grid_dim: ((total_amp as u32 + threads - 1) / threads, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let amp_off = hogel_offset as usize * pixels_per_hogel;
            let amp_end = amp_off + row_hogels * pixels_per_hogel;
            let mut amp_slice = self.target_amp_dev.slice_mut(amp_off..amp_end);
            unsafe {
                self.stream.launch_builder(&self.hemi_to_amp_fn)
                    .arg(&self.hemi_batch_dev)
                    .arg(&mut amp_slice)
                    .arg(&batch_i32)
                    .arg(&hemi_res)
                    .launch(amp_cfg)
                    .expect("hemi_to_amp launch failed");
            }
        }

        self.stream.synchronize().expect("Sync failed");
        eprintln!("[holosim] render_rows rows={actual_rows} grid={}x{} hemi={} spp={} -> {}ms",
            self.grid_w, self.grid_h, self.hemi_res, self.spp, _t.elapsed().as_millis());

        self.rows_rendered = end_row;
        self.rows_rendered
    }

    /// Initialise GS: alloc phase + FFT plan, seed random phase. `target_amp_dev`
    /// was already populated during `render_rows`; no re-extraction needed.
    pub fn gs_setup(&mut self, noise_seed: u64) {
        let _t = std::time::Instant::now();
        let n = self.hemi_res as usize;
        let pixels_per_hogel = n * n;
        let num_hogels = (self.grid_w * self.grid_h) as usize;
        let batch = {
            let mut b = MAX_BATCH_HOGELS.min(num_hogels);
            while num_hogels % b != 0 { b -= 1; }
            b
        };

        let build_complex_fn = self.module.load_function("build_complex_from_phase").expect("build_complex missing");
        let enforce_mag_fn   = self.module.load_function("enforce_far_magnitude").expect("enforce_far_magnitude missing");
        let extract_phase_fn = self.module.load_function("extract_phase").expect("extract_phase missing");
        let apply_panel_fn   = self.module.load_function("apply_panel_model").expect("apply_panel_model missing");
        let intensity_fn     = self.module.load_function("intensity_from_complex").expect("intensity_from_complex missing");

        let mut phase            = self.stream.alloc_zeros::<f32>(num_hogels * pixels_per_hogel).expect("alloc phase");
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

        // Seed random phase across all hogels
        let total = (num_hogels * pixels_per_hogel) as i32;
        let threads = 256u32;
        let init_cfg = LaunchConfig {
            grid_dim: ((total as u32 + threads - 1) / threads, 1, 1),
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
                let elem_off = b_idx * gs.batch * gs.pixels_per_hogel;
                let elem_end = elem_off + gs.batch * gs.pixels_per_hogel;

                let phase_slice = gs.phase.slice(elem_off..elem_end);
                let amp_slice   = self.target_amp_dev.slice(elem_off..elem_end);

                unsafe {
                    self.stream.launch_builder(&gs.build_complex_fn)
                        .arg(&phase_slice).arg(&mut gs.e_a).arg(&batch_elems)
                        .launch(launch_cfg(batch_elems)).expect("build_complex");
                }
                gs.plan.exec_c2c(&mut gs.e_a, &mut gs.e_b, FftDirection::Forward).expect("FFT fwd");
                unsafe {
                    self.stream.launch_builder(&gs.enforce_mag_fn)
                        .arg(&mut gs.e_b).arg(&amp_slice).arg(&batch_elems)
                        .launch(launch_cfg(batch_elems)).expect("enforce_mag");
                }
                gs.plan.exec_c2c(&mut gs.e_b, &mut gs.e_a, FftDirection::Inverse).expect("FFT inv");
                {
                    let mut phase_mut = gs.phase.slice_mut(elem_off..elem_end);
                    unsafe {
                        self.stream.launch_builder(&gs.extract_phase_fn)
                            .arg(&gs.e_a).arg(&mut phase_mut).arg(&batch_elems)
                            .launch(launch_cfg(batch_elems)).expect("extract_phase");
                    }
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
        self.project_and_scatter(false, 0, 0.0, 0);
    }

    /// Apply panel model to phase, forward-project → scatter to output_accum. Final step.
    pub fn gs_finalize(&mut self, phase_bits: u32, noise_sigma_rad: f32, noise_seed: u64) {
        let _t = std::time::Instant::now();
        self.project_and_scatter(true, phase_bits, noise_sigma_rad, noise_seed);
        eprintln!("[holosim] gs_finalize -> {}ms", _t.elapsed().as_millis());
    }

    fn project_and_scatter(
        &mut self,
        apply_panel: bool,
        phase_bits: u32,
        noise_sigma_rad: f32,
        noise_seed: u64,
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

        // Panel model: single launch over all hogels
        if apply_panel {
            let total = (gs.num_hogels * n) as i32;
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
        let eye_x = 278.0f32; let eye_y = 273.0f32; let eye_z = -800.0f32;

        let scatter_cfg = |b_size: u32| LaunchConfig {
            // blockIdx.y = hogel_in_batch, blockIdx.x * blockDim.x + threadIdx.x = pixel_in_window
            grid_dim: ((window_size + threads - 1) / threads, b_size, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        for b_idx in 0..num_batches {
            let hogel_off = (b_idx * gs.batch) as i32;
            let elem_off  = b_idx * gs.batch * n;
            let elem_end  = elem_off + gs.batch * n;

            let phase_slice = gs.phase.slice(elem_off..elem_end);
            // forward project: E_A = exp(iφ)
            unsafe {
                self.stream.launch_builder(&gs.build_complex_fn)
                    .arg(&phase_slice).arg(&mut gs.e_a).arg(&batch_elems)
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

    /// Normalize the scatter accumulators and apply filmic tonemapping → RGBA.
    /// Returns the final reconstruction as a Vec<u8>.
    pub fn reconstruct(&mut self, _eye_x: f32, _eye_y: f32, _eye_z: f32) -> Vec<u8> {
        let _t = std::time::Instant::now();
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
        let result = self.stream.clone_dtoh(&self.recon_dev).expect("D2H recon failed");
        eprintln!("[holosim] reconstruct {}x{} grid={}x{} hemi={} -> {}ms",
            self.out_w, self.out_h, self.grid_w, self.grid_h, self.hemi_res, _t.elapsed().as_millis());
        result
    }

    /// Return RGB thumbnail + phase for the hogel at (hx, hy) from stored preview_dev.
    pub fn get_hogel_preview(&mut self, hx: u32, hy: u32) -> Option<(Vec<u8>, Option<Vec<u8>>, u32)> {
        if hx >= self.grid_w || hy >= self.grid_h { return None; }
        let res = PREVIEW_RES;
        let pixels = (res * res) as usize;
        let hogel_idx = (hy * self.grid_w + hx) as usize;

        // Read thumbnail from preview_dev
        let start = hogel_idx * pixels * 3;
        let end   = start + pixels * 3;
        let rgb_f32: Vec<f32> = self.stream.clone_dtoh(&self.preview_dev.slice(start..end)).ok()?;
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
}


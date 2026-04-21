use napi_derive::napi;
use napi::bindgen_prelude::*;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Mutex;

pub mod scene;
pub mod renderer;
pub mod hologram;
pub mod gpu_hologram;

// Global GPU session (one at a time)
static GPU_SESSION: Mutex<Option<gpu_hologram::GpuHologramSession>> = Mutex::new(None);

// Hand-written PTX for vector_add — version 8.0 is compatible with any CUDA 12.x driver.
// This avoids the nvrtc 12.9 → driver 12.8 PTX version mismatch.
const VECTOR_ADD_PTX: &str = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [param_a];
    ld.param.u64 %rd2, [param_b];
    ld.param.u64 %rd3, [param_c];
    ld.param.u32 %r1, [param_n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.s32 %p1, %r2, %r1;
    @%p1 bra DONE;

    cvt.s64.s32 %rd4, %r2;
    shl.b64 %rd4, %rd4, 2;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

DONE:
    ret;
}
"#;

#[napi(object)]
pub struct GpuInfo {
    pub name: String,
    pub total_memory_mb: u32,
    pub compute_capability: String,
}

#[napi(object)]
pub struct CudaHelloResult {
    pub sum: f64,
    pub expected_sum: f64,
    pub r#match: bool,
}

#[napi]
pub fn ping() -> String {
    "holosim-native v0.1.0 — Rust + CUDA backend ready".to_string()
}

#[napi]
pub fn gpu_info() -> Result<GpuInfo> {
    let ctx = CudaContext::new(0)
        .map_err(|e| Error::from_reason(format!("CUDA context init failed: {e}")))?;

    let name = ctx.name()
        .map_err(|e| Error::from_reason(format!("Failed to get device name: {e}")))?;

    let total_mem = ctx.total_mem()
        .map_err(|e| Error::from_reason(format!("Failed to get device memory: {e}")))?;

    let (major, minor) = ctx.compute_capability()
        .map_err(|e| Error::from_reason(format!("Failed to get compute capability: {e}")))?;

    Ok(GpuInfo {
        name,
        total_memory_mb: (total_mem / (1024 * 1024)) as u32,
        compute_capability: format!("{major}.{minor}"),
    })
}

#[napi]
pub fn cuda_hello(size: u32) -> Result<CudaHelloResult> {
    let n = size as usize;

    let ctx = CudaContext::new(0)
        .map_err(|e| Error::from_reason(format!("CUDA context init failed: {e}")))?;
    let stream = ctx.default_stream();

    // Load hand-written PTX (version 8.0, compatible with CUDA 12.8 driver)
    let ptx = Ptx::from_src(VECTOR_ADD_PTX);
    let module = ctx.load_module(ptx)
        .map_err(|e| Error::from_reason(format!("Module load failed: {e}")))?;
    let func = module.load_function("vector_add")
        .map_err(|e| Error::from_reason(format!("Function load failed: {e}")))?;

    // Create input vectors: a[i] = i, b[i] = i*2
    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();

    let a_dev = stream.clone_htod(&a_host)
        .map_err(|e| Error::from_reason(format!("H2D copy failed: {e}")))?;
    let b_dev = stream.clone_htod(&b_host)
        .map_err(|e| Error::from_reason(format!("H2D copy failed: {e}")))?;
    let mut c_dev = stream.alloc_zeros::<f32>(n)
        .map_err(|e| Error::from_reason(format!("Alloc failed: {e}")))?;

    let threads_per_block = 256u32;
    let blocks = (n as u32 + threads_per_block - 1) / threads_per_block;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream.launch_builder(&func)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&(n as i32))
            .launch(cfg)
            .map_err(|e| Error::from_reason(format!("Kernel launch failed: {e}")))?;
    }

    let c_host: Vec<f32> = stream.clone_dtoh(&c_dev)
        .map_err(|e| Error::from_reason(format!("D2H copy failed: {e}")))?;

    // Sum result: should be sum(i + 2*i) = sum(3*i) = 3 * n*(n-1)/2
    let sum: f64 = c_host.iter().map(|&x| x as f64).sum();
    let expected = 3.0 * (n as f64) * ((n as f64) - 1.0) / 2.0;

    Ok(CudaHelloResult {
        sum,
        expected_sum: expected,
        r#match: (sum - expected).abs() < 1e-3,
    })
}

#[napi]
pub fn test_nvrtc() -> Result<String> {
    let cuda_src = r#"
extern "C" __global__ void test_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx * 3.14159f;
    }
}
"#;

    let ptx_compiled = cudarc::nvrtc::compile_ptx_with_opts(cuda_src, cudarc::nvrtc::CompileOptions {
        arch: Some("compute_80"),
        ..Default::default()
    }).map_err(|e| Error::from_reason(format!("nvrtc compile failed: {e}")))?;

    // Extract PTX as string, downgrade version from 8.8 to 8.0 (driver compat)
    let ptx_src = ptx_compiled.to_src().replace(".version 8.8", ".version 8.0");
    let ptx = Ptx::from_src(ptx_src);

    let ctx = CudaContext::new(0)
        .map_err(|e| Error::from_reason(format!("CUDA init: {e}")))?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(ptx)
        .map_err(|e| Error::from_reason(format!("Module load: {e}")))?;
    let func = module.load_function("test_kernel")
        .map_err(|e| Error::from_reason(format!("Func load: {e}")))?;

    let n = 256usize;
    let mut out_dev = stream.alloc_zeros::<f32>(n)
        .map_err(|e| Error::from_reason(format!("Alloc: {e}")))?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream.launch_builder(&func)
            .arg(&mut out_dev)
            .arg(&(n as i32))
            .launch(cfg)
            .map_err(|e| Error::from_reason(format!("Launch: {e}")))?;
    }

    let out_host: Vec<f32> = stream.clone_dtoh(&out_dev)
        .map_err(|e| Error::from_reason(format!("D2H: {e}")))?;

    let sum: f64 = out_host.iter().map(|&x| x as f64).sum();
    let expected: f64 = (0..256).map(|i| i as f64 * 3.14159).sum();

    Ok(format!("nvrtc OK: sum={sum:.2} expected={expected:.2} match={}", (sum - expected).abs() < 1.0))
}

#[napi(object)]
pub struct GpuHologramNapiResult {
    pub grid_w: u32,
    pub grid_h: u32,
    pub hemi_res: u32,
    pub full_width: u32,
    pub full_height: u32,
    pub recon_data: Buffer,
}

#[napi]
pub fn compute_hologram_gpu(
    grid_w: u32,
    grid_h: u32,
    hemi_res: u32,
    spp: u32,
    out_w: u32,
    out_h: u32,
    max_bounces: u32,
    ambient: f64,
) -> Result<GpuHologramNapiResult> {
    let result = gpu_hologram::compute_hologram_gpu(grid_w, grid_h, hemi_res, spp, out_w, out_h, max_bounces, ambient as f32);
    Ok(GpuHologramNapiResult {
        grid_w: result.grid_w,
        grid_h: result.grid_h,
        hemi_res: result.hemi_res,
        full_width: result.out_w,
        full_height: result.out_h,
        recon_data: result.recon_rgba.into(),
    })
}

// ── Batched GPU session API for progress reporting ────────

#[napi(object)]
pub struct GpuSessionInfo {
    pub grid_w: u32,
    pub grid_h: u32,
    pub hemi_res: u32,
    pub spp: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub total_rows: u32,
}

#[napi]
pub fn gpu_session_begin(
    grid_w: u32, grid_h: u32,
    hemi_res: u32, spp: u32,
    out_w: u32, out_h: u32,
    max_bounces: u32, ambient: f64,
) -> Result<GpuSessionInfo> {
    let session = gpu_hologram::GpuHologramSession::new(
        grid_w, grid_h, hemi_res, spp, out_w, out_h,
        max_bounces, ambient as f32,
    );
    let info = GpuSessionInfo {
        grid_w: session.grid_w,
        grid_h: session.grid_h,
        hemi_res: session.hemi_res,
        spp: session.spp,
        out_w: session.out_w,
        out_h: session.out_h,
        total_rows: session.grid_h,
    };
    *GPU_SESSION.lock().unwrap() = Some(session);
    Ok(info)
}

/// Render the next `num_rows` rows. Returns rows rendered so far (== progress numerator, denom = grid_h).
#[napi]
pub fn gpu_session_render_rows(start_row: u32, num_rows: u32) -> Result<u32> {
    let mut guard = GPU_SESSION.lock().unwrap();
    let session = guard.as_mut().ok_or_else(|| Error::from_reason("No active GPU session"))?;
    Ok(session.render_rows(start_row, num_rows))
}

/// Run Gerchberg-Saxton phase-only holography + panel model + forward projection.
/// Replaces the rendered hemispheres in-place with the reconstructed intensity.
///
/// - `iterations`: GS iterations (typical 10-50; 0 = skip GS, use random phase)
/// - `phase_bits`: phase quantization levels = 2^phase_bits (0 = no quantization)
/// - `noise_sigma_rad`: Gaussian phase noise std-dev (radians)
/// - `noise_seed`: RNG seed (use 0 for deterministic default)
#[napi]
pub fn gpu_session_run_gs(
    iterations: u32,
    phase_bits: u32,
    noise_sigma_rad: f64,
    noise_seed: BigInt,
) -> Result<()> {
    let mut guard = GPU_SESSION.lock().unwrap();
    let session = guard.as_mut().ok_or_else(|| Error::from_reason("No active GPU session"))?;
    let (_signed, seed_u64, _loss) = noise_seed.get_u64();
    session.run_gs(iterations, phase_bits, noise_sigma_rad as f32, seed_u64);
    Ok(())
}

/// Step-wise GS: setup (alloc buffers, extract target amp, seed random phase).
#[napi]
pub fn gpu_session_gs_setup(noise_seed: BigInt) -> Result<()> {
    let mut guard = GPU_SESSION.lock().unwrap();
    let session = guard.as_mut().ok_or_else(|| Error::from_reason("No active GPU session"))?;
    let (_s, seed_u64, _l) = noise_seed.get_u64();
    session.gs_setup(seed_u64);
    Ok(())
}

/// Step-wise GS: run `n_iters` more GS iterations. Returns total iters done so far.
#[napi]
pub fn gpu_session_gs_iterate(n_iters: u32) -> Result<u32> {
    let mut guard = GPU_SESSION.lock().unwrap();
    let session = guard.as_mut().ok_or_else(|| Error::from_reason("No active GPU session"))?;
    session.gs_iterate(n_iters);
    Ok(session.gs_iters_done())
}

/// Step-wise GS: forward-project current phase (without panel model) into hemi_dev
/// for live preview between iterations.
#[napi]
pub fn gpu_session_gs_preview() -> Result<()> {
    let mut guard = GPU_SESSION.lock().unwrap();
    let session = guard.as_mut().ok_or_else(|| Error::from_reason("No active GPU session"))?;
    session.gs_preview();
    Ok(())
}

/// Step-wise GS: apply panel model + final forward projection. Writes reconstructed
/// intensity into hemi_dev (overwriting the ray-traced hemispheres).
#[napi]
pub fn gpu_session_gs_finalize(phase_bits: u32, noise_sigma_rad: f64, noise_seed: BigInt) -> Result<()> {
    let mut guard = GPU_SESSION.lock().unwrap();
    let session = guard.as_mut().ok_or_else(|| Error::from_reason("No active GPU session"))?;
    let (_s, seed_u64, _l) = noise_seed.get_u64();
    session.gs_finalize(phase_bits, noise_sigma_rad as f32, seed_u64);
    Ok(())
}

/// Reconstruct lightfield WITHOUT consuming the session. Used for live preview.
/// Returns RGBA pixels.
#[napi]
pub fn gpu_session_reconstruct(eye_x: f64, eye_y: f64, eye_z: f64) -> Result<Buffer> {
    let mut guard = GPU_SESSION.lock().unwrap();
    let session = guard.as_mut().ok_or_else(|| Error::from_reason("No active GPU session"))?;
    let recon = session.reconstruct(eye_x as f32, eye_y as f32, eye_z as f32);
    Ok(recon.into())
}

#[napi(object)]
pub struct GpuSessionResult {
    pub grid_w: u32,
    pub grid_h: u32,
    pub hemi_res: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub recon_data: Buffer,
}

/// Reconstruct lightfield and finalize session.
#[napi]
pub fn gpu_session_finish(eye_x: f64, eye_y: f64, eye_z: f64) -> Result<GpuSessionResult> {
    let mut guard = GPU_SESSION.lock().unwrap();
    let session = guard.as_mut().ok_or_else(|| Error::from_reason("No active GPU session"))?;
    let recon = session.reconstruct(eye_x as f32, eye_y as f32, eye_z as f32);
    let result = GpuSessionResult {
        grid_w: session.grid_w,
        grid_h: session.grid_h,
        hemi_res: session.hemi_res,
        out_w: session.out_w,
        out_h: session.out_h,
        recon_data: recon.into(),
    };
    *guard = None; // release GPU memory
    Ok(result)
}

/// Explicitly release the GPU session (frees hemi_dev + GS buffers).
/// Must be called before starting a new session with different dimensions.
#[napi]
pub fn gpu_session_close() -> Result<()> {
    let mut guard = GPU_SESSION.lock().unwrap();
    *guard = None;
    Ok(())
}

#[napi(object)]
pub struct HogelPreview {
    pub hx: u32,
    pub hy: u32,
    pub res: u32,
    pub hemisphere: Buffer, // RGB u8, res*res*3 bytes
    pub phase: Option<Buffer>, // grayscale u8, res*res bytes (None if GS hasn't run)
}

/// Extract a single hogel's hemisphere + (optionally) GS phase for UI diagnostics.
#[napi]
pub fn gpu_session_get_hogel_preview(hx: u32, hy: u32) -> Result<HogelPreview> {
    let mut guard = GPU_SESSION.lock().unwrap();
    let session = guard.as_mut().ok_or_else(|| Error::from_reason("No active GPU session"))?;
    let (hemi, phase_opt, res) = session
        .get_hogel_preview(hx, hy)
        .ok_or_else(|| Error::from_reason(format!("Hogel ({}, {}) out of range", hx, hy)))?;
    Ok(HogelPreview {
        hx,
        hy,
        res,
        hemisphere: hemi.into(),
        phase: phase_opt.map(|p| p.into()),
    })
}

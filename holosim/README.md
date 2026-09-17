# HoloSim — Holographic Display Simulator

Real-time hogel-based lightfield simulator with CUDA path tracing. Renders a Cornell Box scene through a 32×32 hogel grid at 512px per-hogel hemisphere resolution, with GPU-accelerated ray tracing (NEE) and lightfield reconstruction.

## Getting Started

**Prerequisites:**
- Node.js 18+
- Rust 1.70+ with `cargo`
- CUDA Toolkit 12.x (tested with 12.9) at `/usr/local/cuda-12.9`
- NVIDIA GPU with compute capability ≥ 8.0

```bash
npm install
cd native && cargo build --release && cp target/release/libholosim_native.so holosim_native.node && cd ..
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH"
npm run dev
```

## Tests

```bash
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH"
npx playwright test
```

## Build / Deploy

```bash
npx vite build
# Outputs: dist/ (renderer) + dist-electron/ (main + preload)
# Run: electron dist-electron/main.js
```

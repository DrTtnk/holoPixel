// Test standard (non-streaming) path at 128×128 hemi=256
const NATIVE = require('./native/holosim_native.node');

const GRID = 128;
const HEMI = 256;
const SPP = 1;
const GS_ITERS = 5;
const OUT = GRID * 9;

console.log(`=== Standard Path Test: ${GRID}×${GRID} hemi=${HEMI} ===`);

// 1. Begin session
let t0 = Date.now();
const begin = NATIVE.gpuSessionBegin(GRID, GRID, HEMI, SPP, OUT, OUT, 2, 0.05);
console.log(`Session begin: streaming=${begin.streaming} -> ${Date.now()-t0}ms`);

if (begin.streaming) {
  console.log('Still in streaming mode! Threshold not high enough.');
  NATIVE.gpuSessionClose();
  process.exit(1);
}

// 2. Render all rows (standard path)
t0 = Date.now();
const BATCH = 32;
for (let row = 0; row < GRID; row += BATCH) {
  const n = Math.min(BATCH, GRID - row);
  const res = NATIVE.gpuSessionRenderRows(row, n);
  if (row === 0) console.log(`  First batch: ${JSON.stringify(res)}`);
}
console.log(`Render all rows -> ${Date.now()-t0}ms`);

// 3. GS setup + iterations
t0 = Date.now();
NATIVE.gpuSessionGsSetup(BigInt(42));
console.log(`GS setup -> ${Date.now()-t0}ms`);

t0 = Date.now();
NATIVE.gpuSessionGsIterate(GS_ITERS);
console.log(`GS ${GS_ITERS} iterations -> ${Date.now()-t0}ms`);

// 4. Reconstruct (center eye) — first call includes scatter overhead
t0 = Date.now();
const recon0 = NATIVE.gpuSessionReconstruct(278, 273, -800);
console.log(`Reconstruct center -> ${Date.now()-t0}ms, ${recon0.length} bytes`);

// 5. Reconstruct (shifted eye)
t0 = Date.now();
const recon1 = NATIVE.gpuSessionReconstruct(478, 273, -800);
console.log(`Reconstruct shifted -> ${Date.now()-t0}ms`);

// 6. Pixel diff (center vs shifted)
let diffCount = 0;
for (let i = 0; i < recon0.length; i++) {
  if (Math.abs(recon0[i] - recon1[i]) > 2) diffCount++;
}
console.log(`Center vs shifted diff (>2): ${diffCount}/${recon0.length} (${(diffCount/recon0.length*100).toFixed(1)}%)`);

// 7. Hogel preview
const preview = NATIVE.gpuSessionGetHogelPreview(GRID/2, GRID/2);
if (preview) {
  console.log(`Hogel preview: hemisphere=${preview.hemisphere.length}B, phase=${preview.phase ? preview.phase.length + 'B' : 'null'}, res=${preview.res}`);
} else {
  console.log('Hogel preview: null');
}

// 8. Parallax sweep timing
t0 = Date.now();
const NUM_FRAMES = 24;
for (let f = 0; f < NUM_FRAMES; f++) {
  const angle = (f / NUM_FRAMES) * 2 * Math.PI;
  const ex = 278 + 100 * Math.cos(angle);
  const ey = 273;
  const ez = -800 + 100 * Math.sin(angle);
  NATIVE.gpuSessionReconstruct(ex, ey, ez);
}
const parallaxTime = Date.now() - t0;
console.log(`24-frame parallax sweep -> ${parallaxTime}ms (${(parallaxTime/NUM_FRAMES).toFixed(1)}ms/frame)`);

NATIVE.gpuSessionClose();
console.log('=== DONE ===');

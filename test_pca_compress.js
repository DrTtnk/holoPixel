// Test PCA intensity compression for streaming-mode parallax
const NATIVE = require('./native/holosim_native.node');

const GRID = 128;  // 128×128 = 16384 hogels → streaming mode with hemi=256
const HEMI = 256;
const SPP = 1;
const GS_ITERS = 5;
const OUT = GRID * 9;  // match App.tsx formula

console.log(`=== PCA Compression Test: ${GRID}×${GRID} hemi=${HEMI} ===`);

// 1. Begin session
let t0 = Date.now();
const begin = NATIVE.gpuSessionBegin(GRID, GRID, HEMI, SPP, OUT, OUT, 2, 0.05);
console.log(`Session begin: streaming=${begin.streaming} -> ${Date.now()-t0}ms`);

if (!begin.streaming) {
  console.log('Not in streaming mode at this grid/hemi. Increase GRID or HEMI.');
  NATIVE.gpuSessionClose();
  process.exit(0);
}

// 2. Stream all rows
t0 = Date.now();
const BATCH = Math.max(1, Math.min(GRID, Math.floor(4096 / GRID)));
const seed = BigInt(42);
for (let row = 0; row < GRID; row += BATCH) {
  const n = Math.min(BATCH, GRID - row);
  NATIVE.gpuSessionStreamBatch(row, n, GS_ITERS, 0, 0, seed);
}
console.log(`Streaming done -> ${Date.now()-t0}ms`);

// 3. Initial reconstruct (from saved_intensity)
t0 = Date.now();
const recon0 = NATIVE.gpuSessionReconstruct(278, 273, -800);
console.log(`Reconstruct (saved_intensity) -> ${Date.now()-t0}ms, ${recon0.length} bytes`);

// 4. Compress intensity
t0 = Date.now();
NATIVE.gpuSessionCompressIntensity();
console.log(`PCA compression -> ${Date.now()-t0}ms`);

// 5. Reconstruct from PCA (center eye)
t0 = Date.now();
const recon1 = NATIVE.gpuSessionReconstruct(278, 273, -800);
console.log(`Reconstruct (PCA, center) -> ${Date.now()-t0}ms`);

// 6. Reconstruct from PCA (shifted eye)
t0 = Date.now();
const recon2 = NATIVE.gpuSessionReconstruct(478, 273, -800);
console.log(`Reconstruct (PCA, shifted) -> ${Date.now()-t0}ms`);

// 7. Compare center vs center (PCA should be similar to raw)
let diffCount = 0;
for (let i = 0; i < recon0.length; i++) {
  if (Math.abs(recon0[i] - recon1[i]) > 2) diffCount++;
}
console.log(`PCA vs raw diff (>2): ${diffCount}/${recon0.length} pixels (${(diffCount/recon0.length*100).toFixed(1)}%)`);

// 8. Compare center vs shifted (should show parallax)
let shiftDiff = 0;
for (let i = 0; i < recon1.length; i++) {
  if (Math.abs(recon1[i] - recon2[i]) > 2) shiftDiff++;
}
console.log(`PCA center vs shifted diff (>2): ${shiftDiff}/${recon1.length} pixels (${(shiftDiff/recon1.length*100).toFixed(1)}%)`);

// 9. Test hogel preview from PCA
const preview = NATIVE.gpuSessionGetHogelPreview(GRID/2, GRID/2);
if (preview) {
  console.log(`Hogel preview: hemisphere=${preview.hemisphere.length}B, phase=${preview.phase ? preview.phase.length + 'B' : 'null'}, res=${preview.res}`);
} else {
  console.log('Hogel preview: null (unexpected!)');
}

// 10. Parallax sweep timing
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

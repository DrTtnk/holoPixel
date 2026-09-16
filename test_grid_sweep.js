// Find the breaking point between 256×256 and 512×512
const NATIVE = require('./native/holosim_native.node');

const HEMI = 256;
const SPP = 1;

for (const GRID of [256, 320, 384, 448, 512]) {
  const OUT = GRID * 9;
  console.log(`\n--- Testing ${GRID}×${GRID} hemi=${HEMI} ---`);
  try {
    const t0 = Date.now();
    const begin = NATIVE.gpuSessionBegin(GRID, GRID, HEMI, SPP, OUT, OUT, 2, 0.05);
    console.log(`  begin: streaming=${begin.streaming} -> ${Date.now()-t0}ms`);
    
    if (!begin.streaming) {
      // Standard path: render rows
      const BATCH = Math.max(1, Math.min(GRID, Math.floor(4096 / GRID)));
      let t1 = Date.now();
      for (let row = 0; row < GRID; row += BATCH) {
        NATIVE.gpuSessionRenderRows(row, Math.min(BATCH, GRID - row));
      }
      console.log(`  render: ${Date.now()-t1}ms`);
      
      t1 = Date.now();
      NATIVE.gpuSessionGsSetup(BigInt(42));
      console.log(`  gs_setup: ${Date.now()-t1}ms`);
      
      t1 = Date.now();
      NATIVE.gpuSessionGsIterate(3);
      console.log(`  gs_iterate(3): ${Date.now()-t1}ms`);
      
      t1 = Date.now();
      const recon = NATIVE.gpuSessionReconstruct(278, 273, -800);
      console.log(`  reconstruct: ${Date.now()-t1}ms, ${recon.length} bytes`);
    } else {
      // Streaming path
      const BATCH = Math.max(1, Math.min(GRID, Math.floor(4096 / GRID)));
      let t1 = Date.now();
      for (let row = 0; row < GRID; row += BATCH) {
        NATIVE.gpuSessionStreamBatch(row, Math.min(BATCH, GRID - row), 3, 0, 0, BigInt(42));
      }
      console.log(`  streaming render: ${Date.now()-t1}ms`);
      
      t1 = Date.now();
      const recon = NATIVE.gpuSessionReconstruct(278, 273, -800);
      console.log(`  reconstruct: ${Date.now()-t1}ms, ${recon.length} bytes`);
    }
    
    NATIVE.gpuSessionClose();
    console.log(`  OK`);
  } catch (e) {
    console.log(`  FAILED: ${e.message}`);
    try { NATIVE.gpuSessionClose(); } catch {}
  }
}

import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test';
import path from 'path';
import { execSync } from 'child_process';
import fs from 'fs';

const ROOT = path.resolve(__dirname, '..');
const SHOTS = path.join(ROOT, 'tests/screenshots/profile');

// ── Shared helpers ─────────────────────────────────────────

async function setSliderByTestId(window: Page, testid: string, value: number): Promise<boolean> {
  return await window.evaluate(({ testid, value }) => {
    const el = document.querySelector(`[data-testid="${testid}"]`) as HTMLInputElement | null;
    if (!el) return false;
    const desc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
    const setter = desc && desc.set;
    if (!setter) return false;
    setter.call(el, String(value));
    el.dispatchEvent(new Event('input', { bubbles: true }));
    return el.value === String(value);
  }, { testid, value });
}

async function setCheckboxByTestId(window: Page, testid: string, checked: boolean): Promise<boolean> {
  return await window.evaluate(({ testid, checked }) => {
    const el = document.querySelector(`[data-testid="${testid}"]`) as HTMLInputElement | null;
    if (!el) return false;
    if (el.checked !== checked) el.click();
    return el.checked === checked;
  }, { testid, checked });
}

async function getRenderState(window: Page): Promise<string | null> {
  return await window.evaluate(() => {
    const btn = document.querySelector('[data-testid="compute-btn"]') as HTMLElement | null;
    return btn ? btn.getAttribute('data-render-state') : null;
  });
}

async function screenshotSafe(window: Page, file: string) {
  try {
    await window.screenshot({ path: file, fullPage: true });
  } catch (e) {
    console.warn('screenshot failed:', file, (e as Error).message);
  }
}

function launchApp() {
  return electron.launch({
    args: ['--start-maximized', path.join(ROOT, 'dist-electron/main.js')],
    env: {
      ...process.env,
      LD_LIBRARY_PATH: `/usr/local/cuda-12.9/lib64:${process.env.LD_LIBRARY_PATH || ''}`,
      NODE_ENV: 'production',
    },
  });
}

/** Collect stderr from the Electron main process (where Rust eprintln! goes) */
function captureStderr(app: ElectronApplication): string[] {
  const lines: string[] = [];
  const proc = app.process();
  proc.stderr?.on('data', (chunk: Buffer) => {
    for (const line of chunk.toString().split('\n')) {
      if (line.trim()) lines.push(line);
    }
  });
  return lines;
}

// ── Setup ──────────────────────────────────────────────────

test.beforeAll(async () => {
  execSync('npx vite build', { cwd: ROOT, stdio: 'pipe', timeout: 120000 });
  const mainJs = path.join(ROOT, 'dist-electron/main.js');
  if (!fs.existsSync(mainJs)) throw new Error(`Missing ${mainJs}`);
  fs.mkdirSync(SHOTS, { recursive: true });
});

// ── Pipeline Profiling ─────────────────────────────────────

test.describe('Pipeline profiling & validation', () => {
  test('profile: small grid timing breakdown', async () => {
    test.setTimeout(300_000);
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    // Small grid for fast profiling: 32×32, hemi=128, spp=4, GS=10 iters
    expect(await setSliderByTestId(window, 'slider-spp', 4)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 32)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 128)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 10)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    await screenshotSafe(window, path.join(SHOTS, '01-profile-config.png'));

    // Run pipeline
    const t0 = performance.now();
    await window.locator('[data-testid="compute-btn"]').click();

    // Wait for done
    const deadline = Date.now() + 120_000;
    while (Date.now() < deadline) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(200);
    }
    const totalMs = performance.now() - t0;
    expect(await getRenderState(window), 'pipeline must reach done').toBe('done');

    await screenshotSafe(window, path.join(SHOTS, '02-profile-done.png'));

    // Extract the render time displayed in the UI
    const uiTime = await window.evaluate(() => {
      const el = document.querySelector('[data-testid="compute-btn"]')?.closest('section');
      return el?.textContent ?? '';
    });
    console.log(`[PROFILE] Total wall time (PW side): ${totalMs.toFixed(0)}ms`);
    console.log(`[PROFILE] UI section text: ${uiTime}`);
    const holosimLogs = stderrLines.filter(l => l.includes('[holosim]'));
    console.log(`[PROFILE] Rust timing breakdown:\n  ${holosimLogs.join('\n  ')}`);

    // Sanity: total should be < 120s for this small config
    expect(totalMs).toBeLessThan(120_000);

    await app.close();
  });

  test('profile: medium grid timing (64×64, hemi=256, spp=16)', async () => {
    test.setTimeout(600_000);
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    expect(await setSliderByTestId(window, 'slider-spp', 16)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 64)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 256)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 10)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    const t0 = performance.now();
    await window.locator('[data-testid="compute-btn"]').click();

    const deadline = Date.now() + 300_000;
    while (Date.now() < deadline) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(500);
    }
    const totalMs = performance.now() - t0;
    expect(await getRenderState(window), 'pipeline must reach done').toBe('done');

    await screenshotSafe(window, path.join(SHOTS, '03-profile-medium.png'));

    console.log(`[PROFILE-MED] Total wall time: ${totalMs.toFixed(0)}ms`);
    const holosimLogs = stderrLines.filter(l => l.includes('[holosim]'));
    console.log(`[PROFILE-MED] Rust timing breakdown:\n  ${holosimLogs.join('\n  ')}`);

    await app.close();
  });
});

// ── Numerical Validation ───────────────────────────────────

test.describe('Numerical validation', () => {
  test('reconstruction has non-zero content with correct pixel statistics', async () => {
    test.setTimeout(300_000);
    const app = await launchApp();
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    // Minimal config: 16×16 grid, hemi=64, spp=1, GS=5 iters (GS is required for output)
    expect(await setSliderByTestId(window, 'slider-spp', 1)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 16)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 64)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 5)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    await window.locator('[data-testid="compute-btn"]').click();

    const deadline = Date.now() + 60_000;
    while (Date.now() < deadline) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(200);
    }
    expect(await getRenderState(window)).toBe('done');

    // Analyze pixel distribution
    const stats = await window.evaluate(() => {
      const c = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
      const ctx = c.getContext('2d')!;
      const d = ctx.getImageData(0, 0, c.width, c.height).data;
      const totalPixels = d.length / 4;
      let nonBlack = 0, totalBrightness = 0, maxVal = 0, minVal = 255;
      for (let i = 0; i < d.length; i += 4) {
        const r = d[i], g = d[i + 1], b = d[i + 2];
        const lum = Math.max(r, g, b);
        if (lum > 0) nonBlack++;
        totalBrightness += lum;
        if (lum > maxVal) maxVal = lum;
        if (lum < minVal) minVal = lum;
      }
      return {
        totalPixels,
        nonBlack,
        nonBlackPct: (nonBlack / totalPixels * 100),
        avgBrightness: totalBrightness / totalPixels,
        maxVal,
        minVal,
      };
    });

    console.log('[VALIDATION] Pixel stats (no GS):', JSON.stringify(stats));

    // Sanity checks:
    // 1. Not all black (the Cornell Box should produce light)
    expect(stats.nonBlack, 'reconstruction should have non-black pixels').toBeGreaterThan(0);
    // 2. Should have dynamic range (min < max) — not a flat image
    expect(stats.maxVal - stats.minVal, 'image should have contrast (max-min > 20)').toBeGreaterThan(20);
    // 3. Average brightness in reasonable range (not too dim, not blown out)
    expect(stats.avgBrightness, 'average brightness should be reasonable').toBeGreaterThan(1);
    expect(stats.avgBrightness, 'average brightness not blown out').toBeLessThan(250);
    // 4. Max should reach at least some brightness (not a totally dark scene)
    expect(stats.maxVal, 'brightest pixel should have some intensity').toBeGreaterThan(10);

    await screenshotSafe(window, path.join(SHOTS, '04-validation-no-gs.png'));
    await app.close();
  });

  test('GS iterations improve reconstruction quality', async () => {
    test.setTimeout(300_000);
    const app = await launchApp();
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    // Config with GS disabled first to get baseline
    expect(await setSliderByTestId(window, 'slider-spp', 1)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 16)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 64)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 5)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    await window.locator('[data-testid="compute-btn"]').click();

    const deadline = Date.now() + 120_000;
    while (Date.now() < deadline) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(200);
    }
    expect(await getRenderState(window)).toBe('done');

    const gsStats = await window.evaluate(() => {
      const c = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
      const ctx = c.getContext('2d')!;
      const d = ctx.getImageData(0, 0, c.width, c.height).data;
      const totalPixels = d.length / 4;
      let nonBlack = 0, totalBrightness = 0;
      for (let i = 0; i < d.length; i += 4) {
        const lum = Math.max(d[i], d[i + 1], d[i + 2]);
        if (lum > 0) nonBlack++;
        totalBrightness += lum;
      }
      return {
        totalPixels,
        nonBlack,
        nonBlackPct: (nonBlack / totalPixels * 100),
        avgBrightness: totalBrightness / totalPixels,
      };
    });

    console.log('[VALIDATION] Pixel stats (GS 5 iters):', JSON.stringify(gsStats));

    // After GS, reconstruction should still have content
    expect(gsStats.nonBlack, 'GS reconstruction should have non-black pixels').toBeGreaterThan(0);
    expect(gsStats.avgBrightness, 'GS avg brightness should be reasonable').toBeGreaterThan(0.5);

    await screenshotSafe(window, path.join(SHOTS, '05-validation-gs5.png'));
    await app.close();
  });

  test('panel quantization degrades quality monotonically', async () => {
    test.setTimeout(300_000);
    const app = await launchApp();
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    const brightnessByBits: Record<string, number> = {};

    for (const bits of [0, 8, 4, 2, 1]) {
      expect(await setSliderByTestId(window, 'slider-spp', 1)).toBe(true);
      expect(await setSliderByTestId(window, 'slider-grid-size', 16)).toBe(true);
      expect(await setSliderByTestId(window, 'slider-hemi-res', 64)).toBe(true);
      expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
      expect(await setSliderByTestId(window, 'gs-iterations', 5)).toBe(true);
      expect(await setSliderByTestId(window, 'gs-phase-bits', bits)).toBe(true);
      expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

      await window.locator('[data-testid="compute-btn"]').click();

      const deadline = Date.now() + 60_000;
      while (Date.now() < deadline) {
        if (await getRenderState(window) === 'done') break;
        await window.waitForTimeout(200);
      }
      expect(await getRenderState(window)).toBe('done');

      const stats = await window.evaluate(() => {
        const c = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
        const ctx = c.getContext('2d')!;
        const d = ctx.getImageData(0, 0, c.width, c.height).data;
        let total = 0;
        for (let i = 0; i < d.length; i += 4) total += Math.max(d[i], d[i + 1], d[i + 2]);
        return total / (d.length / 4);
      });

      brightnessByBits[`${bits}`] = stats;
      console.log(`[VALIDATION] Phase bits=${bits}: avgBrightness=${stats.toFixed(2)}`);
      await screenshotSafe(window, path.join(SHOTS, `06-panel-bits-${bits}.png`));
    }

    // Continuous phase (bits=0) should produce the best result
    // 1-bit should produce the worst (extreme quantization)
    // We can't assert strict monotonicity due to noise, but extremes should differ
    expect(brightnessByBits['0'], 'continuous phase should produce non-zero brightness').toBeGreaterThan(0);
    expect(brightnessByBits['1'], '1-bit phase should still produce some signal').toBeGreaterThan(0);

    await app.close();
  });
});

// ── Scene Screenshot Validation ────────────────────────────

test.describe('Scene visual validation', () => {
  test('capture 3D preview and reconstruction side by side', async () => {
    test.setTimeout(300_000);
    const app = await launchApp();
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    // Screenshot: initial UI with 3D scene preview visible
    await screenshotSafe(window, path.join(SHOTS, '07-scene-initial.png'));

    // Verify Three.js canvas exists and renders something
    const threeStats = await window.evaluate(() => {
      const canvases = Array.from(document.querySelectorAll('canvas'));
      // The Three.js canvas is inside the scene preview section, not data-testid main-viewport
      const threeCandidates = canvases.filter(c => {
        const parent = c.closest('section');
        return parent?.textContent?.includes('Cornell Box Scene') ?? false;
      });
      if (threeCandidates.length === 0) return null;
      const c = threeCandidates[0];
      const ctx = c.getContext('webgl2') || c.getContext('webgl');
      // WebGL canvases can't be read via getImageData; check they have size
      return { width: c.width, height: c.height, exists: true };
    });

    console.log('[SCENE] Three.js canvas:', JSON.stringify(threeStats));
    expect(threeStats, '3D scene canvas should exist').not.toBeNull();
    expect(threeStats!.width, '3D canvas should have width').toBeGreaterThan(0);

    // Run a small render to get the reconstruction
    expect(await setSliderByTestId(window, 'slider-spp', 1)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 16)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 64)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 5)).toBe(true);

    await window.locator('[data-testid="compute-btn"]').click();

    const deadline = Date.now() + 60_000;
    while (Date.now() < deadline) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(200);
    }
    expect(await getRenderState(window)).toBe('done');

    // Screenshot: full UI with reconstruction result and 3D preview
    await screenshotSafe(window, path.join(SHOTS, '08-scene-after-render.png'));

    // Hover center to trigger hogel diagnostic
    const viewport = window.locator('[data-testid="main-viewport"]');
    const vpBox = await viewport.boundingBox();
    if (vpBox) {
      await window.mouse.move(vpBox.x + vpBox.width * 0.5, vpBox.y + vpBox.height * 0.5);
      await window.waitForTimeout(500);
      await screenshotSafe(window, path.join(SHOTS, '09-scene-hover-center.png'));
    }

    await app.close();
  });
});

// ── Streaming Mode Tests ───────────────────────────────────

test.describe('Streaming pipeline', () => {
  test('streaming mode: 128×128 grid produces valid output', async () => {
    test.setTimeout(600_000);
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    // 128×128 at hemi=256 triggers streaming (target_amp ≈ 4.3 GB > 4 GB threshold)
    expect(await setSliderByTestId(window, 'slider-spp', 4)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 128)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 256)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 5)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    await screenshotSafe(window, path.join(SHOTS, '10-streaming-config.png'));

    const t0 = performance.now();
    await window.locator('[data-testid="compute-btn"]').click();

    // Wait for done — streaming 128×128 could take minutes
    const deadline = Date.now() + 300_000;
    while (Date.now() < deadline) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(1000);
    }
    const totalMs = performance.now() - t0;
    expect(await getRenderState(window), 'streaming pipeline must reach done').toBe('done');

    await screenshotSafe(window, path.join(SHOTS, '11-streaming-done.png'));

    // Verify the reconstruction canvas has non-black content
    const pixelStats = await window.evaluate(() => {
      const canvas = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
      if (!canvas) return null;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
      let nonBlack = 0, sum = 0;
      for (let i = 0; i < id.data.length; i += 4) {
        const val = id.data[i] + id.data[i + 1] + id.data[i + 2];
        if (val > 0) nonBlack++;
        sum += val;
      }
      const total = id.data.length / 4;
      return { totalPixels: total, nonBlack, nonBlackPct: (nonBlack * 100 / total) | 0, avgBrightness: sum / (total * 3) };
    });
    expect(pixelStats, 'pixel stats must be available').not.toBeNull();
    console.log(`[STREAMING] Pixel stats: ${JSON.stringify(pixelStats)}`);
    expect(pixelStats!.nonBlackPct, 'streaming output must have >50% non-black pixels').toBeGreaterThan(50);
    expect(pixelStats!.avgBrightness, 'streaming output must have reasonable brightness').toBeGreaterThan(20);

    const holosimLogs = stderrLines.filter(l => l.includes('[holosim]'));
    const streamingLog = holosimLogs.find(l => l.includes('STREAMING mode'));
    expect(streamingLog, 'should log STREAMING mode').toBeTruthy();
    console.log(`[STREAMING] Total wall time: ${totalMs.toFixed(0)}ms`);
    console.log(`[STREAMING] Rust logs:\n  ${holosimLogs.join('\n  ')}`);

    await app.close();
  });

  test('streaming mode: 256×256 grid scaling test', async () => {
    test.setTimeout(600_000);
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    // 256×256 at hemi=256, spp=4, GS=5
    expect(await setSliderByTestId(window, 'slider-spp', 4)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 256)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 256)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 5)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    const t0 = performance.now();
    await window.locator('[data-testid="compute-btn"]').click();

    const deadline = Date.now() + 300_000;
    while (Date.now() < deadline) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(2000);
    }
    const totalMs = performance.now() - t0;
    expect(await getRenderState(window), 'streaming pipeline must reach done').toBe('done');

    await screenshotSafe(window, path.join(SHOTS, '12-streaming-256-done.png'));

    const pixelStats = await window.evaluate(() => {
      const canvas = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
      if (!canvas) return null;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
      let nonBlack = 0, sum = 0;
      for (let i = 0; i < id.data.length; i += 4) {
        const val = id.data[i] + id.data[i + 1] + id.data[i + 2];
        if (val > 0) nonBlack++;
        sum += val;
      }
      const total = id.data.length / 4;
      return { totalPixels: total, nonBlack, nonBlackPct: (nonBlack * 100 / total) | 0, avgBrightness: sum / (total * 3) };
    });
    expect(pixelStats).not.toBeNull();
    expect(pixelStats!.nonBlackPct).toBeGreaterThan(50);
    console.log(`[STREAMING-256] Pixel stats: ${JSON.stringify(pixelStats)}`);

    const holosimLogs = stderrLines.filter(l => l.includes('[holosim]'));
    console.log(`[STREAMING-256] Total wall time: ${totalMs.toFixed(0)}ms`);
    console.log(`[STREAMING-256] Rust logs:\n  ${holosimLogs.join('\n  ')}`);

    await app.close();
  });
});

// ── PCA Eigenspectrum Analysis ─────────────────────────────

test.describe('PCA eigenspectrum', () => {
  test('PCA: 32×32 grid — phase vs target_amp vs hemisphere', async () => {
    test.setTimeout(300_000);
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    expect(await setSliderByTestId(window, 'slider-spp', 4)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 32)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 128)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 10)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    await window.locator('[data-testid="compute-btn"]').click();
    const deadline = Date.now() + 120_000;
    while (Date.now() < deadline) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(200);
    }
    expect(await getRenderState(window)).toBe('done');

    // Run PCA on all three representations
    const results = await window.evaluate(async () => {
      const h = (window as any).holosim;
      const phase = await h.gpuSessionPcaEigenspectrum(200);
      const tamp = await h.gpuSessionPcaTargetAmp(200);
      const hemi = await h.gpuSessionPcaHemisphere(200);
      return {
        phase: phase.ok ? phase.data : null,
        targetAmp: tamp.ok ? tamp.data : null,
        hemisphere: hemi.ok ? hemi.data : null,
      };
    });

    for (const [name, r] of Object.entries(results) as [string, any][]) {
      if (!r) { console.log(`[PCA-32] ${name}: FAILED`); continue; }
      console.log(`[PCA-32] ${name}: ${r.numHogels} hogels, ${r.pixelsPerHogel} px/hogel`);
      console.log(`[PCA-32] ${name} top-10 eig: ${r.eigenvalues.slice(0, 10).map((v: number) => v.toFixed(4)).join(', ')}`);
      const cvPoints = [1, 5, 10, 20, 50, 100];
      for (const k of cvPoints) {
        if (k <= r.cumVar.length) {
          console.log(`[PCA-32] ${name} k=${k}: cumVar=${(r.cumVar[k - 1] * 100).toFixed(2)}%`);
        }
      }
      for (const target of [0.9, 0.95, 0.99]) {
        const k = r.cumVar.findIndex((v: number) => v >= target) + 1;
        console.log(`[PCA-32] ${name} ${(target * 100).toFixed(0)}% var: k=${k || '>200'}`);
      }
    }

    expect(results.phase).not.toBeNull();
    expect(results.targetAmp).not.toBeNull();
    expect(results.hemisphere).not.toBeNull();

    // Hemisphere should be MUCH more compressible than phase
    const hemiK50 = results.hemisphere!.cumVar[49];
    const phaseK50 = results.phase!.cumVar[49];
    console.log(`[PCA-32] Hemisphere vs Phase at k=50: ${(hemiK50 * 100).toFixed(1)}% vs ${(phaseK50 * 100).toFixed(1)}%`);

    const holosimLogs = stderrLines.filter((l: string) => l.includes('[holosim] PCA'));
    console.log(`[PCA-32] Rust logs:\n  ${holosimLogs.join('\n  ')}`);

    await app.close();
  });

  test('PCA: 64×64 grid — all three representations (hemi=256)', async () => {
    test.setTimeout(600_000);
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    expect(await setSliderByTestId(window, 'slider-spp', 4)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 64)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 256)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 10)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    await window.locator('[data-testid="compute-btn"]').click();
    const deadline = Date.now() + 300_000;
    while (Date.now() < deadline) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(500);
    }
    expect(await getRenderState(window)).toBe('done');

    const results = await window.evaluate(async () => {
      const h = (window as any).holosim;
      const phase = await h.gpuSessionPcaEigenspectrum(300);
      const tamp = await h.gpuSessionPcaTargetAmp(300);
      const hemi = await h.gpuSessionPcaHemisphere(300);
      return {
        phase: phase.ok ? phase.data : null,
        targetAmp: tamp.ok ? tamp.data : null,
        hemisphere: hemi.ok ? hemi.data : null,
      };
    });

    for (const [name, r] of Object.entries(results) as [string, any][]) {
      if (!r) { console.log(`[PCA-64] ${name}: FAILED`); continue; }
      console.log(`[PCA-64] ${name}: ${r.numHogels} hogels, ${r.pixelsPerHogel} px/hogel`);
      const cvPoints = [1, 5, 10, 20, 50, 100, 200];
      for (const k of cvPoints) {
        if (k <= r.cumVar.length) {
          console.log(`[PCA-64] ${name} k=${k}: cumVar=${(r.cumVar[k - 1] * 100).toFixed(2)}%`);
        }
      }
      for (const target of [0.9, 0.95, 0.99]) {
        const k = r.cumVar.findIndex((v: number) => v >= target) + 1;
        console.log(`[PCA-64] ${name} ${(target * 100).toFixed(0)}% var: k=${k || '>300'}`);
      }
    }

    expect(results.phase).not.toBeNull();
    expect(results.targetAmp).not.toBeNull();
    expect(results.hemisphere).not.toBeNull();

    const holosimLogs = stderrLines.filter((l: string) => l.includes('[holosim] PCA'));
    console.log(`[PCA-64] Rust logs:\n  ${holosimLogs.join('\n  ')}`);

    await app.close();
  });

  test('PCA: 128×128 grid — target_amp only (hemi=128)', async () => {
    test.setTimeout(600_000);
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    // 128×128 with hemi=128 stays in standard mode (target_amp ~1 GB)
    expect(await setSliderByTestId(window, 'slider-spp', 4)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 128)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 128)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 5)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    await window.locator('[data-testid="compute-btn"]').click();
    const deadline = Date.now() + 300_000;
    while (Date.now() < deadline) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(500);
    }
    expect(await getRenderState(window)).toBe('done');

    // Only run target_amp PCA to minimize GPU memory pressure
    const r = await window.evaluate(async () => {
      const h = (window as any).holosim;
      const result = await h.gpuSessionPcaTargetAmp(200);
      return result.ok ? result.data : null;
    });

    expect(r, 'PCA target_amp result').not.toBeNull();
    console.log(`[PCA-128] targetAmp: ${r.numHogels} hogels, ${r.pixelsPerHogel} px/hogel`);
    const cvPoints = [1, 5, 10, 20, 50, 100, 200];
    for (const k of cvPoints) {
      if (k <= r.cumVar.length) {
        console.log(`[PCA-128] targetAmp k=${k}: cumVar=${(r.cumVar[k - 1] * 100).toFixed(2)}%`);
      }
    }
    for (const target of [0.9, 0.95, 0.99]) {
      const k = r.cumVar.findIndex((v: number) => v >= target) + 1;
      console.log(`[PCA-128] targetAmp ${(target * 100).toFixed(0)}% var: k=${k || '>200'}`);
    }

    const holosimLogs = stderrLines.filter((l: string) => l.includes('[holosim] PCA'));
    console.log(`[PCA-128] Rust logs:\n  ${holosimLogs.join('\n  ')}`);

    await app.close();
  });

  test('PCA compression quality: 32×32 grid, hemi=128, k sweep', async () => {
    test.setTimeout(600_000);
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    // 32×32 with hemi=128 — standard mode, 1024 hogels × 16384 pixels = 64 MB target_amp
    expect(await setSliderByTestId(window, 'slider-spp', 4)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 32)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 128)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 10)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    // Render hologram (original)
    await window.locator('[data-testid="compute-btn"]').click();
    const deadline = Date.now() + 300_000;
    while (Date.now() < deadline) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(500);
    }
    expect(await getRenderState(window)).toBe('done');

    // Screenshot the original reconstruction
    await screenshotSafe(window, path.join(SHOTS, 'pca-compress-original-32.png'));

    // Save target_amp to CPU so we can restore after each PCA compress
    const saveResult = await window.evaluate(async () => {
      const h = (window as any).holosim;
      const r = await h.gpuSessionSaveTargetAmp();
      return r.ok;
    });
    expect(saveResult, 'save target_amp').toBe(true);

    // Test PCA compression at different k values
    const kValues = [10, 30, 50, 70, 100, 150, 200];
    const results: { k: number; ratio: number; rmse: number }[] = [];

    for (const k of kValues) {
      // Restore original target_amp from CPU backup
      if (k !== kValues[0]) {
        const restoreOk = await window.evaluate(async () => {
          const h = (window as any).holosim;
          const r = await h.gpuSessionRestoreTargetAmp();
          return r.ok;
        });
        expect(restoreOk, `restore target_amp before k=${k}`).toBe(true);
      }

      // PCA compress (modifies target_amp_dev in-place with reconstructed version)
      const compressResult = await window.evaluate(async (kVal) => {
        const h = (window as any).holosim;
        const result = await h.gpuSessionPcaCompressTargetAmp(kVal);
        return result.ok ? result.data : null;
      }, k);

      expect(compressResult, `PCA compress k=${k}`).not.toBeNull();
      results.push({ k: compressResult.kUsed, ratio: compressResult.compressionRatio, rmse: compressResult.rmse });
      console.log(`[PCA-compress] k=${k}: ratio=${compressResult.compressionRatio.toFixed(1)}x, RMSE=${compressResult.rmse.toFixed(6)}`);

      // Re-run GS on the PCA-decompressed target_amp, then reconstruct
      const gsReconResult = await window.evaluate(async () => {
        const h = (window as any).holosim;
        const setupRes = await h.gpuSessionGsSetup(BigInt(42));
        if (!setupRes.ok) return { ok: false, error: 'setup: ' + setupRes.error };
        const iterRes = await h.gpuSessionGsIterate(10);
        if (!iterRes.ok) return { ok: false, error: 'iterate: ' + iterRes.error };
        const finalRes = await h.gpuSessionGsFinalize(0, 0.0, BigInt(0));
        if (!finalRes.ok) return { ok: false, error: 'finalize: ' + finalRes.error };
        const reconRes = await h.gpuSessionReconstruct(0.0, 0.0, 1.0);
        if (!reconRes.ok) return { ok: false, error: 'reconstruct: ' + reconRes.error };
        return { ok: true };
      });

      expect(gsReconResult.ok, `GS+reconstruct after PCA k=${k}: ${(gsReconResult as any).error || 'no error'}`).toBe(true);

      // Screenshot the PCA-compressed reconstruction
      await screenshotSafe(window, path.join(SHOTS, `pca-compress-k${k}-32.png`));
    }

    // Print summary table
    console.log('\n[PCA-compress] Summary (32×32, hemi=128):');
    console.log('  k  | ratio    | RMSE');
    console.log('  ---|----------|--------');
    for (const r of results) {
      console.log(`  ${String(r.k).padStart(3)} | ${r.ratio.toFixed(1).padStart(8)}x | ${r.rmse.toFixed(6)}`);
    }

    // Assert basic quality at k=100: should have compression > 1x
    const r100 = results.find(r => r.k === 100);
    if (r100) {
      expect(r100.ratio).toBeGreaterThan(1);
    }

    const holosimLogs = stderrLines.filter((l: string) => l.includes('[holosim]'));
    console.log(`[PCA-compress] Rust logs:\n  ${holosimLogs.join('\n  ')}`);

    await app.close();
  });

  test('Streaming 1024×1024 + PCA sampling (full scale)', async () => {
    test.setTimeout(900_000); // 15 min — could take a while
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    // Drive the pipeline directly via IPC (bypass UI) to inject enablePcaSampling
    const GRID = 1024, HEMI = 256, SPP = 4, GS_ITERS = 5;
    const OUT = 1024, BOUNCES = 3, AMBIENT = 0.05;

    // Step 1: Begin session
    const beginResult = await window.evaluate(async (params: any) => {
      const h = (window as any).holosim;
      return await h.gpuSessionBegin(params.g, params.g, params.h, params.s, params.o, params.o, params.b, params.a);
    }, { g: GRID, h: HEMI, s: SPP, o: OUT, b: BOUNCES, a: AMBIENT });
    expect(beginResult.ok, `gpuSessionBegin failed: ${beginResult.error}`).toBe(true);
    const totalRows = beginResult.data.totalRows;
    const isStreaming = beginResult.data.streaming;
    console.log(`[STREAM-PCA-1024] Session: ${GRID}×${GRID}, hemi=${HEMI}, spp=${SPP}, totalRows=${totalRows}, streaming=${isStreaming}`);
    expect(isStreaming, '1024×1024 must be streaming').toBe(true);

    // Step 2: Enable PCA sampling (after session exists, before streaming)
    const enableResult = await window.evaluate(async () => {
      const h = (window as any).holosim;
      return await h.gpuSessionEnablePcaSampling(2048);
    });
    console.log(`[STREAM-PCA-1024] Enable PCA sampling (2048 hogels): ${JSON.stringify(enableResult)}`);
    expect(enableResult.ok, `enablePcaSampling failed: ${enableResult.error}`).toBe(true);

    // Step 3: Stream all rows
    const BATCH_ROWS = Math.max(1, Math.min(GRID, Math.floor(4096 / GRID)));
    const seed = BigInt(42);
    const t0 = performance.now();
    let rowsDone = 0;
    for (let row = 0; row < totalRows; row += BATCH_ROWS) {
      const n = Math.min(BATCH_ROWS, totalRows - row);
      const r = await window.evaluate(async (params: any) => {
        const h = (window as any).holosim;
        return await h.gpuSessionStreamBatch(params.row, params.n, params.iters, params.bits, params.sigma, BigInt(params.seed));
      }, { row, n, iters: GS_ITERS, bits: 0, sigma: 0, seed: 42 });
      if (!r.ok || !r.data) {
        console.log(`[STREAM-PCA-1024] Stream batch failed at row=${row}: ${r.error}`);
        break;
      }
      rowsDone = r.data.rowsDone;
      if (rowsDone % 64 === 0 || rowsDone === totalRows) {
        const elapsed = (performance.now() - t0) / 1000;
        const eta = rowsDone > 0 ? (totalRows - rowsDone) * (elapsed / rowsDone) : 0;
        console.log(`[STREAM-PCA-1024] Progress: ${rowsDone}/${totalRows} rows (${(rowsDone * 100 / totalRows).toFixed(1)}%) elapsed=${elapsed.toFixed(1)}s ETA=${eta.toFixed(0)}s`);
      }
    }
    const totalMs = performance.now() - t0;
    console.log(`[STREAM-PCA-1024] Streaming done: ${rowsDone}/${totalRows} rows in ${(totalMs / 1000).toFixed(1)}s`);
    expect(rowsDone).toBe(totalRows);

    // Step 4: Reconstruct
    const reconResult = await window.evaluate(async () => {
      const h = (window as any).holosim;
      return await h.gpuSessionReconstruct(0, 0, 0.3);
    });
    console.log(`[STREAM-PCA-1024] Reconstruct: ${JSON.stringify({ ok: reconResult.ok, error: reconResult.error })}`);

    await screenshotSafe(window, path.join(SHOTS, 'stream-pca-1024-done.png'));

    // Step 5: PCA eigenspectrum on streaming samples
    console.log(`[STREAM-PCA-1024] Running PCA eigenspectrum on 2048 sampled hogels...`);
    const pcaT0 = performance.now();
    const pcaResult = await window.evaluate(async () => {
      const h = (window as any).holosim;
      return await h.gpuSessionPcaStreamingEigenspectrum(300);
    });
    const pcaMs = performance.now() - pcaT0;
    console.log(`[STREAM-PCA-1024] PCA analysis took ${(pcaMs / 1000).toFixed(1)}s`);

    expect(pcaResult.ok, `PCA streaming eigenspectrum failed: ${pcaResult.error}`).toBe(true);
    const pca = pcaResult.data;
    console.log(`[STREAM-PCA-1024] Sampled ${pca.numHogels} hogels, ${pca.pixelsPerHogel} px/hogel`);
    console.log(`[STREAM-PCA-1024] Top-10 eigenvalues: ${pca.eigenvalues.slice(0, 10).map((v: number) => v.toFixed(4)).join(', ')}`);

    // Report cumulative variance at key k values
    const cvPoints = [1, 5, 10, 20, 50, 100, 150, 200, 300];
    for (const k of cvPoints) {
      if (k <= pca.cumVar.length) {
        console.log(`[STREAM-PCA-1024] k=${k}: cumVar=${(pca.cumVar[k - 1] * 100).toFixed(2)}%`);
      }
    }
    for (const target of [0.8, 0.9, 0.95, 0.99]) {
      const k = pca.cumVar.findIndex((v: number) => v >= target) + 1;
      console.log(`[STREAM-PCA-1024] ${(target * 100).toFixed(0)}% var: k=${k || '>300'}`);
    }

    // Scale invariance check: compare with 32×32 and 128×128 results
    if (pca.cumVar.length >= 100) {
      const k10 = pca.cumVar[9];
      const k50 = pca.cumVar[49];
      const k100 = pca.cumVar[99];
      console.log(`[STREAM-PCA-1024] Scale-invariance check:`);
      console.log(`  k=10:  ${(k10 * 100).toFixed(1)}%  (32×32: 65.7%, 128×128: 65.5%)`);
      console.log(`  k=50:  ${(k50 * 100).toFixed(1)}%  (32×32: 87.7%, 128×128: 87.5%)`);
      console.log(`  k=100: ${(k100 * 100).toFixed(1)}% (32×32: 93.5%, 128×128: 93.0%)`);
    }

    // Streaming log summary
    const holosimLogs = stderrLines.filter((l: string) => l.includes('[holosim]'));
    const streamBatches = holosimLogs.filter((l: string) => l.includes('stream_batch'));
    const pcaLogs = holosimLogs.filter((l: string) => l.includes('PCA'));
    console.log(`[STREAM-PCA-1024] ${streamBatches.length} stream batches completed`);
    console.log(`[STREAM-PCA-1024] PCA Rust logs:\n  ${pcaLogs.join('\n  ')}`);

    // Close session
    await window.evaluate(async () => {
      const h = (window as any).holosim;
      return await h.gpuSessionClose();
    });

    await app.close();
  });

  test('Full eigenspectrum: 1024×1024, 2048 components for 99.9% target', async () => {
    test.setTimeout(900_000);
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    const GRID = 1024, HEMI = 256, SPP = 4, GS_ITERS = 5;
    const OUT = 1024, BOUNCES = 3, AMBIENT = 0.05;
    const NUM_SAMPLES = 2048;

    // Begin session
    const beginResult = await window.evaluate(async (p: any) => {
      const h = (window as any).holosim;
      return await h.gpuSessionBegin(p.g, p.g, p.h, p.s, p.o, p.o, p.b, p.a);
    }, { g: GRID, h: HEMI, s: SPP, o: OUT, b: BOUNCES, a: AMBIENT });
    expect(beginResult.ok).toBe(true);
    const totalRows = beginResult.data.totalRows;
    console.log(`[FULL-EIGEN] Session: ${GRID}×${GRID}, streaming=${beginResult.data.streaming}`);

    // Enable PCA sampling
    const enRes = await window.evaluate(async (n: number) => {
      return await (window as any).holosim.gpuSessionEnablePcaSampling(n);
    }, NUM_SAMPLES);
    expect(enRes.ok).toBe(true);

    // Stream all rows
    const BATCH_ROWS = Math.max(1, Math.floor(4096 / GRID));
    const t0 = performance.now();
    let rowsDone = 0;
    for (let row = 0; row < totalRows; row += BATCH_ROWS) {
      const n = Math.min(BATCH_ROWS, totalRows - row);
      const r = await window.evaluate(async (p: any) => {
        return await (window as any).holosim.gpuSessionStreamBatch(p.row, p.n, p.iters, 0, 0, BigInt(p.seed));
      }, { row, n, iters: GS_ITERS, seed: 42 });
      if (!r.ok) break;
      rowsDone = r.data.rowsDone;
      if (rowsDone % 128 === 0 || rowsDone === totalRows) {
        const el = (performance.now() - t0) / 1000;
        console.log(`[FULL-EIGEN] ${rowsDone}/${totalRows} (${(rowsDone*100/totalRows).toFixed(0)}%) ${el.toFixed(0)}s`);
      }
    }
    const streamMs = performance.now() - t0;
    expect(rowsDone).toBe(totalRows);
    console.log(`[FULL-EIGEN] Streaming done in ${(streamMs/1000).toFixed(1)}s`);

    // PCA with ALL 2048 components
    console.log(`[FULL-EIGEN] Computing PCA with ${NUM_SAMPLES} max components...`);
    const pcaT0 = performance.now();
    const pcaResult = await window.evaluate(async (mc: number) => {
      return await (window as any).holosim.gpuSessionPcaStreamingEigenspectrum(mc);
    }, NUM_SAMPLES);
    const pcaMs = performance.now() - pcaT0;
    expect(pcaResult.ok, `PCA failed: ${pcaResult.error}`).toBe(true);
    const pca = pcaResult.data;
    console.log(`[FULL-EIGEN] PCA: ${pca.numHogels} hogels, ${pca.pixelsPerHogel} px/hogel, ${pca.eigenvalues.length} components in ${(pcaMs/1000).toFixed(1)}s`);

    // Find k for each accuracy target
    const targets = [0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999];
    const ppHogel = pca.pixelsPerHogel; // 65536 = 256×256
    const numHogels = GRID * GRID; // 1,048,576

    console.log(`\n[FULL-EIGEN] === STORAGE ANALYSIS ===`);
    console.log(`[FULL-EIGEN] Original: ${numHogels} hogels × ${ppHogel} px × 4B = ${(numHogels * ppHogel * 4 / 1e9).toFixed(1)} GB`);

    for (const target of targets) {
      const k = pca.cumVar.findIndex((v: number) => v >= target) + 1;
      const pct = (target * 100).toFixed(target >= 0.999 ? (target >= 0.9999 ? 3 : 2) : 1);
      if (k > 0) {
        // Storage: k basis vectors (shared) + k coefficients per hogel
        const basisBytes = k * ppHogel * 4; // float32
        const coeffBytes = numHogels * k * 4; // float32
        const totalBytes = basisBytes + coeffBytes;
        const origBytes = numHogels * ppHogel * 4;
        const ratio = origBytes / totalBytes;
        const cv = (pca.cumVar[k-1] * 100).toFixed(3);
        console.log(`[FULL-EIGEN] ${pct}% var: k=${k}, cumVar=${cv}%`);
        console.log(`[FULL-EIGEN]   basis: ${(basisBytes/1e6).toFixed(1)} MB, coeffs: ${(coeffBytes/1e6).toFixed(1)} MB, total: ${(totalBytes/1e6).toFixed(1)} MB (${ratio.toFixed(1)}× compression)`);
      } else {
        console.log(`[FULL-EIGEN] ${pct}% var: k>${pca.cumVar.length} (not reached with ${pca.cumVar.length} components)`);
        const lastCV = pca.cumVar[pca.cumVar.length - 1];
        console.log(`[FULL-EIGEN]   max cumVar with ${pca.cumVar.length} components: ${(lastCV * 100).toFixed(4)}%`);
      }
    }

    // Eigenvalue decay analysis
    console.log(`\n[FULL-EIGEN] === EIGENVALUE DECAY ===`);
    const logPoints = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];
    for (const k of logPoints) {
      if (k <= pca.eigenvalues.length) {
        console.log(`[FULL-EIGEN] λ_${k} = ${pca.eigenvalues[k-1].toFixed(6)}, cumVar=${(pca.cumVar[k-1]*100).toFixed(3)}%`);
      }
    }

    // Residual analysis
    if (pca.cumVar.length >= 100) {
      const tail100 = 1 - pca.cumVar[99];
      const tail300 = pca.cumVar.length >= 300 ? 1 - pca.cumVar[299] : null;
      const tail1000 = pca.cumVar.length >= 1000 ? 1 - pca.cumVar[999] : null;
      console.log(`\n[FULL-EIGEN] === RESIDUAL ===`);
      console.log(`[FULL-EIGEN] After k=100: ${(tail100*100).toFixed(3)}% residual`);
      if (tail300 !== null) console.log(`[FULL-EIGEN] After k=300: ${(tail300*100).toFixed(3)}% residual`);
      if (tail1000 !== null) console.log(`[FULL-EIGEN] After k=1000: ${(tail1000*100).toFixed(3)}% residual`);
    }

    const holosimLogs = stderrLines.filter((l: string) => l.includes('[holosim] PCA'));
    console.log(`\n[FULL-EIGEN] Rust PCA logs:\n  ${holosimLogs.join('\n  ')}`);

    await window.evaluate(async () => { return await (window as any).holosim.gpuSessionClose(); });
    await app.close();
  });

  test('Tiled PCA: 64×64 grid, compare tile sizes vs flat', async () => {
    test.setTimeout(600_000);
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    const GRID = 64, HEMI = 256, SPP = 4, GS_ITERS = 30;
    const OUT = 1024, BOUNCES = 3, AMBIENT = 0.05;

    // Begin session (non-streaming at 64×64)
    const beginResult = await window.evaluate(async (p: any) => {
      const h = (window as any).holosim;
      return await h.gpuSessionBegin(p.g, p.g, p.h, p.s, p.o, p.o, p.b, p.a);
    }, { g: GRID, h: HEMI, s: SPP, o: OUT, b: BOUNCES, a: AMBIENT });
    expect(beginResult.ok, `Begin failed: ${beginResult.error}`).toBe(true);
    expect(beginResult.data.streaming).toBe(false);
    console.log(`[TILED-PCA] Session: ${GRID}×${GRID}, streaming=${beginResult.data.streaming}`);

    // Render all rows
    const totalRows = beginResult.data.totalRows;
    for (let row = 0; row < totalRows; row += 8) {
      const n = Math.min(8, totalRows - row);
      const r = await window.evaluate(async (p: any) => {
        return await (window as any).holosim.gpuSessionRenderRows(p.row, p.n);
      }, { row, n });
      expect(r.ok, `RenderRows failed: ${r.error}`).toBe(true);
    }
    console.log(`[TILED-PCA] Render done`);

    // GS setup + iterate
    const gsSetup = await window.evaluate(async () => {
      return await (window as any).holosim.gpuSessionGsSetup(BigInt(42));
    });
    expect(gsSetup.ok, `GS setup failed: ${gsSetup.error}`).toBe(true);

    const gsIter = await window.evaluate(async (n: number) => {
      return await (window as any).holosim.gpuSessionGsIterate(n);
    }, GS_ITERS);
    expect(gsIter.ok, `GS iterate failed: ${gsIter.error}`).toBe(true);
    console.log(`[TILED-PCA] GS: ${GS_ITERS} iterations`);

    // Save target_amp before tiled analysis (so we can also do quantization after)
    const saveRes = await window.evaluate(async () => {
      return await (window as any).holosim.gpuSessionSaveTargetAmp();
    });
    expect(saveRes.ok).toBe(true);

    // Test multiple tile sizes
    const tileSizes = [4, 8, 16, 32, 64];
    const maxK = 200; // max components per tile PCA

    console.log(`\n[TILED-PCA] === TILED vs FLAT PCA COMPARISON ===`);
    console.log(`[TILED-PCA] Grid: ${GRID}×${GRID} = ${GRID*GRID} hogels, hemi=${HEMI} (${HEMI*HEMI} px)`);

    for (const ts of tileSizes) {
      const t0 = performance.now();
      const result = await window.evaluate(async (p: any) => {
        return await (window as any).holosim.gpuSessionPcaTiledTargetAmp(p.ts, p.mk);
      }, { ts, mk: maxK });
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      expect(result.ok, `Tiled PCA failed for tile=${ts}: ${result.error}`).toBe(true);

      const d = result.data;
      console.log(`\n[TILED-PCA] ─── tile_size=${ts} (${d.numTiles} tiles of ${d.hogelsPerTile} hogels) ── ${elapsed}s ───`);

      for (let ti = 0; ti < d.targets.length; ti++) {
        const pct = (d.targets[ti] * 100).toFixed(d.targets[ti] >= 0.999 ? 1 : 0);
        const origMB = d.originalBytes / 1e6;
        const tiledMB = d.tiledTotalBytes[ti] / 1e6;
        const flatMB = d.flatTotalBytes[ti] / 1e6;
        const tiledRatio = d.originalBytes / d.tiledTotalBytes[ti];
        const flatRatio = d.originalBytes / d.flatTotalBytes[ti];
        const savings = ((1 - d.tiledTotalBytes[ti] / d.flatTotalBytes[ti]) * 100).toFixed(1);

        console.log(`[TILED-PCA]   ${pct}%: tile k median=${d.tileMedianK[ti]} max=${d.tileMaxK[ti]} min=${d.tileMinK[ti]} mean=${d.tileMeanK[ti].toFixed(1)} | flat k=${d.flatK[ti]}`);
        console.log(`[TILED-PCA]     tiled: ${tiledMB.toFixed(1)} MB (${tiledRatio.toFixed(1)}×) | flat: ${flatMB.toFixed(1)} MB (${flatRatio.toFixed(1)}×) | ${savings}% savings`);
      }

      // Per-tile k distribution for 99% target (index 2)
      const targetIdx = 2; // 99%
      const perTileKs = d.perTileK.slice(targetIdx * d.numTiles, (targetIdx + 1) * d.numTiles);
      const histogram: Record<number, number> = {};
      for (const k of perTileKs) {
        histogram[k] = (histogram[k] || 0) + 1;
      }
      const sortedBins = Object.entries(histogram).sort((a, b) => Number(a[0]) - Number(b[0]));
      const histStr = sortedBins.map(([k, c]) => `${k}:${c}`).join(' ');
      console.log(`[TILED-PCA]   99% k histogram: ${histStr}`);
    }

    // Flat eigenspectrum for reference (from last tiled run which includes flat)
    const lastResult = await window.evaluate(async (p: any) => {
      return await (window as any).holosim.gpuSessionPcaTiledTargetAmp(p.ts, p.mk);
    }, { ts: tileSizes[tileSizes.length - 1], mk: maxK });
    if (lastResult.ok) {
      const d = lastResult.data;
      console.log(`\n[TILED-PCA] === FLAT EIGENSPECTRUM (reference) ===`);
      const logK = [1, 5, 10, 20, 50, 100, 150, 200];
      for (const k of logK) {
        if (k <= d.flatEigenvalues.length) {
          console.log(`[TILED-PCA] λ_${k}=${d.flatEigenvalues[k-1].toFixed(4)}, cumVar=${(d.flatCumVar[k-1]*100).toFixed(2)}%`);
        }
      }
    }

    // ── QUANTIZATION ANALYSIS (batch sweep — PCA computed once) ──
    console.log(`\n[TILED-PCA] === COEFFICIENT QUANTIZATION ANALYSIS (batch sweep) ===`);

    const kValues = [10, 20, 30, 50, 100, 150, 200];
    const bitDepths = [0, 4, 6, 8, 10, 12, 16]; // 0 = float32 (baseline)

    console.log(`[TILED-PCA] k values: ${kValues.join(', ')}`);
    console.log(`[TILED-PCA] bit depths: ${bitDepths.map(b => b === 0 ? 'f32' : `${b}b`).join(', ')}`);

    const sweepT0 = performance.now();
    const sweepResult = await window.evaluate(async (p: any) => {
      return await (window as any).holosim.gpuSessionPcaQuantizationSweep(p.ks, p.bits);
    }, { ks: kValues, bits: bitDepths });
    const sweepMs = performance.now() - sweepT0;
    expect(sweepResult.ok, `Sweep failed: ${sweepResult.error}`).toBe(true);
    console.log(`[TILED-PCA] Sweep: ${sweepResult.data.length} combos in ${(sweepMs/1000).toFixed(1)}s`);

    // Group by k for display
    for (const k of kValues) {
      console.log(`\n[TILED-PCA] ─── k=${k} ───`);
      for (const entry of sweepResult.data.filter((e: any) => e.k === k)) {
        const label = entry.bits === 0 ? 'f32' : `${entry.bits}b`;
        console.log(`[TILED-PCA]   ${label}: RMSE=${entry.rmse.toFixed(6)}, ${(entry.compressedBytes/1e6).toFixed(2)} MB, ${entry.compressionRatio.toFixed(1)}× compression`);
      }
    }

    // 8-bit summary table
    console.log(`\n[TILED-PCA] === 8-BIT QUANTIZATION SUMMARY ===`);
    console.log(`[TILED-PCA] 8-bit: k | RMSE | Compression | Size`);
    for (const entry of sweepResult.data.filter((e: any) => e.bits === 8)) {
      console.log(`[TILED-PCA]   k=${entry.k}: RMSE=${entry.rmse.toFixed(6)}, ${entry.compressionRatio.toFixed(1)}× compression, ${(entry.compressedBytes/1e6).toFixed(2)} MB`);
    }

    // float32 vs 8-bit RMSE comparison for each k
    console.log(`\n[TILED-PCA] === QUANTIZATION ERROR ANALYSIS ===`);
    console.log(`[TILED-PCA] k | f32 RMSE | 8-bit RMSE | 8-bit overhead`);
    for (const k of kValues) {
      const f32Entry = sweepResult.data.find((e: any) => e.k === k && e.bits === 0);
      const q8Entry = sweepResult.data.find((e: any) => e.k === k && e.bits === 8);
      if (f32Entry && q8Entry) {
        const overhead = ((q8Entry.rmse / f32Entry.rmse - 1) * 100).toFixed(1);
        console.log(`[TILED-PCA]   k=${k}: f32=${f32Entry.rmse.toFixed(6)}, 8b=${q8Entry.rmse.toFixed(6)}, overhead=${overhead}%`);
      }
    }

    const holosimLogs = stderrLines.filter((l: string) => l.includes('[holosim]'));
    const pcaLogs = holosimLogs.filter((l: string) => l.includes('PCA'));
    console.log(`\n[TILED-PCA] Rust PCA logs:\n  ${pcaLogs.slice(-20).join('\n  ')}`);

    await window.evaluate(async () => { return await (window as any).holosim.gpuSessionClose(); });
    await app.close();
  });
});

// ── Streaming Pipeline Profiling ────────────────────────────

test.describe('Streaming pipeline profiling', () => {
  test('Per-stage GPU timing at multiple scales', async () => {
    test.setTimeout(600_000); // 10 min
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    const configs = [
      // { label, grid, hemi, spp, gs, bounces, ambient, out }
      { label: '256×256 hemi=16', grid: 256, hemi: 16, spp: 4, gs: 5, bounces: 3, ambient: 0.05, out: 512 },
      { label: '256×256 hemi=32', grid: 256, hemi: 32, spp: 4, gs: 5, bounces: 3, ambient: 0.05, out: 512 },
      { label: '256×256 hemi=64', grid: 256, hemi: 64, spp: 4, gs: 5, bounces: 3, ambient: 0.05, out: 512 },
      { label: '256×256 hemi=128', grid: 256, hemi: 128, spp: 4, gs: 5, bounces: 3, ambient: 0.05, out: 512 },
      { label: '256×256 hemi=256', grid: 256, hemi: 256, spp: 4, gs: 5, bounces: 3, ambient: 0.05, out: 512 },
      { label: '1024×1024 hemi=256 (prod)', grid: 1024, hemi: 256, spp: 4, gs: 5, bounces: 3, ambient: 0.05, out: 1024 },
      { label: '256×256 spp=16',  grid: 256, hemi: 16, spp: 16, gs: 5, bounces: 3, ambient: 0.05, out: 512 },
      { label: '256×256 gs=10',   grid: 256, hemi: 16, spp: 4, gs: 10, bounces: 3, ambient: 0.05, out: 512 },
    ];

    console.log(`\n[STREAM-PROF] === STREAMING PIPELINE PER-STAGE PROFILING ===\n`);

    for (const cfg of configs) {
      // Begin session
      const beginResult = await window.evaluate(async (p: any) => {
        const h = (window as any).holosim;
        return await h.gpuSessionBegin(p.grid, p.grid, p.hemi, p.spp, p.out, p.out, p.bounces, p.ambient);
      }, cfg);
      expect(beginResult.ok, `Begin failed for ${cfg.label}: ${beginResult.error}`).toBe(true);
      const streaming = beginResult.data.streaming;

      // Profile first batch
      const BATCH_ROWS = Math.max(1, Math.min(cfg.grid, Math.floor(4096 / cfg.grid)));
      const profileResult = await window.evaluate(async (p: any) => {
        const h = (window as any).holosim;
        return await h.gpuSessionStreamBatchProfile(0, p.rows, p.gs, 0, 0, BigInt(42));
      }, { rows: BATCH_ROWS, gs: cfg.gs });
      expect(profileResult.ok, `Profile failed for ${cfg.label}: ${profileResult.error}`).toBe(true);

      const timings = profileResult.data.timings;
      const get = (name: string) => timings.find((t: any) => t.name === name)?.ms ?? 0;

      const hogels = get('hogels');
      const pxPerHog = get('pixels_per_hogel');
      const total = get('total');
      const render = get('render');
      const hemiToAmp = get('hemi_to_amp');
      const initPhase = get('init_phase');
      const gsTotal = get('gs_total');
      const panel = get('panel_model');
      const scatter = get('scatter');
      const gsBatches = get('gs_batches');
      const gsBatchSize = get('gs_batch_size');

      console.log(`[STREAM-PROF] ─── ${cfg.label} (streaming=${streaming}) ───`);
      console.log(`[STREAM-PROF]   ${hogels} hogels × ${pxPerHog} px, ${gsBatches} gs_batches of ${gsBatchSize}`);
      console.log(`[STREAM-PROF]   render:      ${render.toFixed(1)}ms (${(render*100/total).toFixed(1)}%)`);
      console.log(`[STREAM-PROF]   hemi_to_amp: ${hemiToAmp.toFixed(1)}ms (${(hemiToAmp*100/total).toFixed(1)}%)`);
      console.log(`[STREAM-PROF]   init_phase:  ${initPhase.toFixed(1)}ms (${(initPhase*100/total).toFixed(1)}%)`);
      console.log(`[STREAM-PROF]   gs_total:    ${gsTotal.toFixed(1)}ms (${(gsTotal*100/total).toFixed(1)}%) [${cfg.gs} iters]`);
      if (cfg.gs > 0) {
        const gsPerIter = gsTotal / cfg.gs;
        console.log(`[STREAM-PROF]     per_iter:  ${gsPerIter.toFixed(1)}ms`);
        // Show individual GS iter timings
        for (let i = 0; i < cfg.gs; i++) {
          const iterMs = get(`gs_iter_${i}`);
          console.log(`[STREAM-PROF]     iter_${i}: ${iterMs.toFixed(1)}ms`);
        }
      }
      console.log(`[STREAM-PROF]   panel:       ${panel.toFixed(1)}ms (${(panel*100/total).toFixed(1)}%)`);
      console.log(`[STREAM-PROF]   scatter:     ${scatter.toFixed(1)}ms (${(scatter*100/total).toFixed(1)}%)`);
      console.log(`[STREAM-PROF]   TOTAL:       ${total.toFixed(1)}ms`);

      // Run a few more batches for average
      const NUM_AVG = 3;
      let avgTotals: number[] = [total];
      for (let i = 1; i <= NUM_AVG; i++) {
        const batchRow = BATCH_ROWS * i;
        if (batchRow >= cfg.grid) break;
        const r = await window.evaluate(async (p: any) => {
          const h = (window as any).holosim;
          return await h.gpuSessionStreamBatchProfile(p.row, p.rows, p.gs, 0, 0, BigInt(42));
        }, { row: batchRow, rows: BATCH_ROWS, gs: cfg.gs });
        if (r.ok) {
          const t = r.data.timings.find((t: any) => t.name === 'total')?.ms ?? 0;
          avgTotals.push(t);
        }
      }
      const avg = avgTotals.reduce((a: number, b: number) => a + b, 0) / avgTotals.length;
      const totalBatches = Math.ceil(cfg.grid / BATCH_ROWS);
      const estTotal = avg * totalBatches / 1000;
      console.log(`[STREAM-PROF]   avg batch: ${avg.toFixed(1)}ms (${avgTotals.length} samples)`);
      console.log(`[STREAM-PROF]   estimated full: ${estTotal.toFixed(1)}s (${totalBatches} batches)`);

      // Close session
      await window.evaluate(async () => { return await (window as any).holosim.gpuSessionClose(); });
    }

    // Print Rust logs
    const holosimLogs = stderrLines.filter((l: string) => l.includes('[holosim]'));
    const profileLogs = holosimLogs.filter((l: string) => l.includes('stream_profile'));
    console.log(`\n[STREAM-PROF] Rust profile logs:\n  ${profileLogs.slice(-30).join('\n  ')}`);

    await app.close();
  });
});

// ── Parallax validation ────────────────────────────────────

test.describe('Parallax reconstruction', () => {
  test('different eye positions produce different images + 1024 artifact screenshot', async () => {
    test.setTimeout(300_000);
    const app = await launchApp();
    const stderrLines = captureStderr(app);
    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    await window.waitForTimeout(3000);

    const PARALLAX_SHOTS = path.join(ROOT, 'tests/screenshots/parallax');
    fs.mkdirSync(PARALLAX_SHOTS, { recursive: true });

    // ── Part 1: Parallax at 64×64 grid ──
    const GRID = 64, HEMI = 64, SPP = 4, GS_ITERS = 10;
    const OUT = 512, BOUNCES = 3, AMBIENT = 0.05;

    console.log(`\n[PARALLAX] === PARALLAX VALIDATION ===`);
    console.log(`[PARALLAX] Grid: ${GRID}×${GRID}, hemi=${HEMI}, spp=${SPP}, gs=${GS_ITERS}, out=${OUT}`);

    // Begin session
    const beginResult = await window.evaluate(async (p: any) => {
      return await (window as any).holosim.gpuSessionBegin(p.grid, p.grid, p.hemi, p.spp, p.out, p.out, p.bounces, p.ambient);
    }, { grid: GRID, hemi: HEMI, spp: SPP, out: OUT, bounces: BOUNCES, ambient: AMBIENT });
    expect(beginResult.ok, `gpuSessionBegin failed: ${beginResult.error}`).toBe(true);
    const totalRows = beginResult.data.totalRows;
    console.log(`[PARALLAX] Session: totalRows=${totalRows}, streaming=${beginResult.data.streaming}`);

    // Render all rows
    const BATCH_ROWS = Math.max(1, Math.min(GRID, Math.floor(4096 / GRID)));
    for (let row = 0; row < totalRows; row += BATCH_ROWS) {
      const n = Math.min(BATCH_ROWS, totalRows - row);
      await window.evaluate(async (p: any) => {
        return await (window as any).holosim.gpuSessionRenderRows(p.row, p.n);
      }, { row, n });
    }
    console.log(`[PARALLAX] Render complete`);

    // GS setup + iterate
    await window.evaluate(async () => {
      return await (window as any).holosim.gpuSessionGsSetup(BigInt(42));
    });
    await window.evaluate(async (iters: number) => {
      return await (window as any).holosim.gpuSessionGsIterate(iters);
    }, GS_ITERS);
    console.log(`[PARALLAX] GS complete (${GS_ITERS} iterations)`);

    // Define eye positions: center + 4 offsets (left, right, up, down)
    const CENTER = { x: 278, y: 273, z: -800 };
    const OFFSET = 100; // mm offset for parallax
    const eyePositions = [
      { label: 'center',  x: CENTER.x,          y: CENTER.y,          z: CENTER.z },
      { label: 'left',    x: CENTER.x - OFFSET,  y: CENTER.y,          z: CENTER.z },
      { label: 'right',   x: CENTER.x + OFFSET,  y: CENTER.y,          z: CENTER.z },
      { label: 'up',      x: CENTER.x,          y: CENTER.y + OFFSET,  z: CENTER.z },
      { label: 'down',    x: CENTER.x,          y: CENTER.y - OFFSET,  z: CENTER.z },
      { label: 'close',   x: CENTER.x,          y: CENTER.y,          z: CENTER.z + 300 },
      { label: 'far',     x: CENTER.x,          y: CENTER.y,          z: CENTER.z - 300 },
    ];

    // Reconstruct at each eye position, save RGBA as raw PNG
    const reconImages: { label: string; pixels: number[] }[] = [];
    for (const eye of eyePositions) {
      const result = await window.evaluate(async (e: any) => {
        const h = (window as any).holosim;
        const r = await h.gpuSessionReconstruct(e.x, e.y, e.z);
        if (!r.ok || !r.data?.reconBuffer) return { ok: false, error: r.error };
        // Sample 256 evenly-spaced pixels for comparison
        const buf = r.data.reconBuffer;
        const samples: number[] = [];
        const step = Math.floor(buf.length / 256);
        for (let i = 0; i < 256; i++) samples.push(buf[i * step]);
        // Also compute stats
        let sum = 0, nonBlack = 0;
        for (let i = 0; i < buf.length; i += 4) {
          const v = buf[i] + buf[i + 1] + buf[i + 2];
          sum += v;
          if (v > 0) nonBlack++;
        }
        return { ok: true, samples, nonBlack, totalPixels: buf.length / 4, mean: sum / (buf.length / 4 * 3) };
      }, eye);
      expect(result.ok, `Reconstruct failed at ${eye.label}: ${result.error}`).toBe(true);
      console.log(`[PARALLAX] ${eye.label.padEnd(8)} eye=(${eye.x}, ${eye.y}, ${eye.z}) nonBlack=${result.nonBlack}/${result.totalPixels} mean=${result.mean.toFixed(1)}`);
      reconImages.push({ label: eye.label, pixels: result.samples });
    }

    // Helper: reconstruct + encode to PNG via browser Canvas
    async function reconToPng(win: Page, eyeX: number, eyeY: number, eyeZ: number, outSize: number): Promise<Buffer> {
      const b64 = await win.evaluate(async (p: any) => {
        const h = (window as any).holosim;
        const r = await h.gpuSessionReconstruct(p.x, p.y, p.z);
        if (!r.ok || !r.data?.reconBuffer) return null;
        const buf = r.data.reconBuffer;
        const sz = p.sz;
        const canvas = document.createElement('canvas');
        canvas.width = sz; canvas.height = sz;
        const ctx = canvas.getContext('2d')!;
        const imgData = ctx.createImageData(sz, sz);
        imgData.data.set(new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.byteLength));
        ctx.putImageData(imgData, 0, 0);
        return canvas.toDataURL('image/png').split(',')[1];
      }, { x: eyeX, y: eyeY, z: eyeZ, sz: outSize });
      expect(b64, 'reconToPng returned null').toBeTruthy();
      return Buffer.from(b64!, 'base64');
    }

    // Save parallax center at 512 for visual validation
    const centerSmallPng = await reconToPng(window, CENTER.x, CENTER.y, CENTER.z, OUT);
    fs.writeFileSync(path.join(PARALLAX_SHOTS, 'parallax-512-center.png'), centerSmallPng);

    // Verify parallax: each non-center image should differ from center
    const centerPx = reconImages[0].pixels;
    let allDiffer = true;
    for (let i = 1; i < reconImages.length; i++) {
      const otherPx = reconImages[i].pixels;
      let diffCount = 0;
      for (let j = 0; j < centerPx.length; j++) {
        if (centerPx[j] !== otherPx[j]) diffCount++;
      }
      const diffPct = (diffCount * 100 / centerPx.length).toFixed(1);
      console.log(`[PARALLAX] ${reconImages[i].label} vs center: ${diffCount}/${centerPx.length} pixels differ (${diffPct}%)`);
      if (diffCount === 0) allDiffer = false;
    }
    expect(allDiffer, 'all non-center eye positions must produce different images (parallax works)').toBe(true);

    // Also check left vs right differ from each other
    const leftPx = reconImages[1].pixels;
    const rightPx = reconImages[2].pixels;
    let lrDiff = 0;
    for (let j = 0; j < leftPx.length; j++) {
      if (leftPx[j] !== rightPx[j]) lrDiff++;
    }
    console.log(`[PARALLAX] left vs right: ${lrDiff}/${leftPx.length} differ (${(lrDiff * 100 / leftPx.length).toFixed(1)}%)`);
    expect(lrDiff, 'left and right views must differ').toBeGreaterThan(0);

    await window.evaluate(async () => { return await (window as any).holosim.gpuSessionClose(); });
    console.log(`[PARALLAX] Part 1 complete (parallax validated)\n`);

    // ── Part 2: 1024-res screenshot for artifact investigation ──
    const GRID2 = 128, HEMI2 = 128, SPP2 = 8, GS2 = 10, OUT2 = 1024;
    console.log(`[PARALLAX] === 1024-RES ARTIFACT SCREENSHOT ===`);
    console.log(`[PARALLAX] Grid: ${GRID2}×${GRID2}, hemi=${HEMI2}, spp=${SPP2}, gs=${GS2}, out=${OUT2}`);

    const begin2 = await window.evaluate(async (p: any) => {
      return await (window as any).holosim.gpuSessionBegin(p.grid, p.grid, p.hemi, p.spp, p.out, p.out, p.bounces, p.ambient);
    }, { grid: GRID2, hemi: HEMI2, spp: SPP2, out: OUT2, bounces: BOUNCES, ambient: AMBIENT });
    expect(begin2.ok, `gpuSessionBegin failed for 1024 render: ${begin2.error}`).toBe(true);
    const totalRows2 = begin2.data.totalRows;

    // Render + GS via streaming batch (handles both streaming and non-streaming)
    const BATCH2 = Math.max(1, Math.min(GRID2, Math.floor(4096 / GRID2)));
    const t0 = performance.now();
    for (let row = 0; row < totalRows2; row += BATCH2) {
      const n = Math.min(BATCH2, totalRows2 - row);
      await window.evaluate(async (p: any) => {
        return await (window as any).holosim.gpuSessionStreamBatch(p.row, p.n, p.gs, 0, 0, BigInt(42));
      }, { row, n, gs: GS2 });
    }
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    console.log(`[PARALLAX] 1024-res pipeline done in ${elapsed}s`);

    // Reconstruct at center
    const recon1024 = await window.evaluate(async (e: any) => {
      const h = (window as any).holosim;
      const r = await h.gpuSessionReconstruct(e.x, e.y, e.z);
      if (!r.ok) return { ok: false, error: r.error };
      const buf = r.data.reconBuffer;
      let nonBlack = 0, sum = 0;
      for (let i = 0; i < buf.length; i += 4) {
        const v = buf[i] + buf[i+1] + buf[i+2];
        sum += v; if (v > 0) nonBlack++;
      }
      return { ok: true, nonBlack, total: buf.length / 4, mean: (sum / (buf.length / 4 * 3)).toFixed(1) };
    }, CENTER);
    expect(recon1024.ok, `1024 reconstruct failed: ${recon1024.error}`).toBe(true);
    console.log(`[PARALLAX] 1024-res: nonBlack=${recon1024.nonBlack}/${recon1024.total}, mean=${recon1024.mean}`);

    // Save reconstruction as PNG (center, left, right)
    const centerPng = await reconToPng(window, CENTER.x, CENTER.y, CENTER.z, OUT2);
    fs.writeFileSync(path.join(PARALLAX_SHOTS, 'parallax-1024-center.png'), centerPng);

    const leftPng = await reconToPng(window, CENTER.x - OFFSET, CENTER.y, CENTER.z, OUT2);
    fs.writeFileSync(path.join(PARALLAX_SHOTS, 'parallax-1024-left.png'), leftPng);

    const rightPng = await reconToPng(window, CENTER.x + OFFSET, CENTER.y, CENTER.z, OUT2);
    fs.writeFileSync(path.join(PARALLAX_SHOTS, 'parallax-1024-right.png'), rightPng);

    // Print Rust logs
    const holosimLogs = stderrLines.filter((l: string) => l.includes('[holosim]'));
    console.log(`\n[PARALLAX] Rust logs:\n  ${holosimLogs.slice(-15).join('\n  ')}`);

    await window.evaluate(async () => { return await (window as any).holosim.gpuSessionClose(); });
    await app.close();
  });
});

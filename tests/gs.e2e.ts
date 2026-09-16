import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test';
import path from 'path';
import { execSync } from 'child_process';
import fs from 'fs';

const ROOT = path.resolve(__dirname, '..');
const SHOTS = path.join(ROOT, 'tests/screenshots/gs');

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
  try {
    return await window.evaluate(() => {
      const btn = document.querySelector('[data-testid="compute-btn"]') as HTMLElement | null;
      return btn ? btn.getAttribute('data-render-state') : null;
    });
  } catch {
    return null;
  }
}

async function getGsItersDone(window: Page): Promise<number> {
  try {
    return await window.evaluate(() => {
      const el = document.querySelector('[data-testid="gs-progress-text"]');
      if (!el || !el.textContent) return -1;
      const m = el.textContent.match(/(\d+)\/(\d+)/);
      return m ? parseInt(m[1], 10) : -1;
    });
  } catch {
    return -1;
  }
}

async function screenshotSafe(window: Page, file: string) {
  try {
    await window.screenshot({ path: file, fullPage: true });
  } catch (e) {
    console.warn('screenshot failed:', file, (e as Error).message);
  }
}

test.beforeAll(async () => {
  execSync('npx vite build', { cwd: ROOT, stdio: 'pipe', timeout: 120000 });
  const mainJs = path.join(ROOT, 'dist-electron/main.js');
  if (!fs.existsSync(mainJs)) throw new Error(`Missing ${mainJs}`);
  fs.mkdirSync(SHOTS, { recursive: true });
});

test.describe('Phase-only Holography — full pipeline', () => {
  test('GS live preview + panel degradation + hogel hover + parallax (single window)', async () => {
    test.setTimeout(600_000);

    const app: ElectronApplication = await electron.launch({
      args: ['--start-maximized', path.join(ROOT, 'dist-electron/main.js')],
      env: {
        ...process.env,
        LD_LIBRARY_PATH: `/usr/local/cuda-12.9/lib64:${process.env.LD_LIBRARY_PATH || ''}`,
        NODE_ENV: 'production',
      },
    });

    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');

    await app.evaluate(({ BrowserWindow }) => {
      const win = BrowserWindow.getAllWindows()[0];
      win.maximize();
      win.setMenuBarVisibility(false);
    });

    const pageErrors: string[] = [];
    window.on('pageerror', (err) => pageErrors.push(err.message));
    window.on('crash', () => pageErrors.push('RENDERER CRASHED'));

    await window.waitForTimeout(3000);
    await screenshotSafe(window, path.join(SHOTS, '00-launch.png'));

    // ────────────────────────────────────────────────────────
    // Phase 1: Clean GS with live iteration preview
    // ────────────────────────────────────────────────────────
    console.log('\n═══ Phase 1: Clean GS (32×32, hemi=128, spp=1, 20 iters, live preview) ═══');

    expect(await setSliderByTestId(window, 'slider-spp', 1)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 32)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 128)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 20)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-live-preview', true)).toBe(true);

    await window.waitForTimeout(300);
    await screenshotSafe(window, path.join(SHOTS, '01-configured-clean.png'));

    const computeBtn = window.locator('[data-testid="compute-btn"]');
    await expect(computeBtn).toBeVisible();
    await computeBtn.click();

    // Poll for GS iteration progress — screenshot each unique iter count
    const seenIters = new Set<number>();
    const startTime = Date.now();
    const MAX_WAIT = 120_000;
    let lastState: string | null = null;
    let hitDone = false;
    while (Date.now() - startTime < MAX_WAIT) {
      const state = await getRenderState(window);
      if (state !== lastState) {
        console.log(`[${Date.now() - startTime}ms] renderState:`, state);
        lastState = state;
      }
      const iters = await getGsItersDone(window);
      if (iters > 0 && !seenIters.has(iters)) {
        seenIters.add(iters);
        await screenshotSafe(window, path.join(SHOTS, `02-iter-${String(iters).padStart(3, '0')}.png`));
      }
      if (state === 'done') { hitDone = true; break; }
      await window.waitForTimeout(100);
    }
    expect(hitDone, 'renderState should reach "done"').toBe(true);

    await window.waitForTimeout(500);
    await screenshotSafe(window, path.join(SHOTS, '03-final-clean.png'));

    const cleanStats = await window.evaluate(() => {
      const canvas = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement | null;
      if (!canvas) return null;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      const d = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let nonBlack = 0;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i] > 0 || d[i + 1] > 0 || d[i + 2] > 0) nonBlack++;
      }
      return { total: d.length / 4, nonBlack };
    });
    expect(cleanStats).not.toBeNull();
    expect(cleanStats!.nonBlack).toBeGreaterThan(0);
    console.log('clean GS non-black pixels:', cleanStats!.nonBlack, '/', cleanStats!.total);
    console.log('iterations captured:', Array.from(seenIters).sort((a, b) => a - b));

    // ────────────────────────────────────────────────────────
    // Phase 2: Hogel hover diagnostic
    // ────────────────────────────────────────────────────────
    console.log('\n═══ Phase 2: Hogel hover diagnostic ═══');

    const viewport = window.locator('[data-testid="main-viewport"]');
    const vpBox = await viewport.boundingBox();
    expect(vpBox).not.toBeNull();

    // Hover near top-left, center, bottom-right to trigger multiple hogel lookups
    const hoverPoints = [
      { x: vpBox!.x + vpBox!.width * 0.15, y: vpBox!.y + vpBox!.height * 0.15 },
      { x: vpBox!.x + vpBox!.width * 0.5,  y: vpBox!.y + vpBox!.height * 0.5 },
      { x: vpBox!.x + vpBox!.width * 0.85, y: vpBox!.y + vpBox!.height * 0.85 },
    ];
    for (const p of hoverPoints) {
      await window.mouse.move(p.x, p.y);
      await window.waitForTimeout(400);
    }
    await window.waitForTimeout(500);
    await screenshotSafe(window, path.join(SHOTS, '04-hover-diagnostic.png'));

    // Inspect the diagnostic canvases — hemisphere + phase should have drawn pixels
    const hogelDiag = await window.evaluate(() => {
      const canvases = Array.from(document.querySelectorAll('canvas'));
      // Find small diagnostic canvases (not the main 1024 viewport)
      const hd = canvases.filter(c => c.width > 0 && c.width <= 512 && c.height > 0 && c.height <= 512);
      return hd.map(c => {
        const ctx = c.getContext('2d');
        if (!ctx) return { w: c.width, h: c.height, nonBlack: 0 };
        const d = ctx.getImageData(0, 0, c.width, c.height).data;
        let nb = 0;
        for (let i = 0; i < d.length; i += 4) if (d[i] || d[i + 1] || d[i + 2]) nb++;
        return { w: c.width, h: c.height, nonBlack: nb, total: d.length / 4 };
      });
    });
    console.log('hogel diagnostic canvases:', hogelDiag);
    const anyNonBlack = hogelDiag.some(s => s.nonBlack > 0);
    expect(anyNonBlack, 'hogel-hover should populate at least one diagnostic tile').toBe(true);

    // ────────────────────────────────────────────────────────
    // Phase 3: Degraded GS (heavy quantization + noise)
    // ────────────────────────────────────────────────────────
    console.log('\n═══ Phase 3: Degraded GS (3-bit quant, σ=0.3) ═══');

    expect(await setSliderByTestId(window, 'gs-phase-bits', 3)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0.3)).toBe(true);
    await window.waitForTimeout(200);
    await screenshotSafe(window, path.join(SHOTS, '05-configured-degraded.png'));

    await computeBtn.click();
    const t2 = Date.now();
    let hitDone2 = false;
    while (Date.now() - t2 < MAX_WAIT) {
      const s = await getRenderState(window);
      if (s === 'done') { hitDone2 = true; break; }
      await window.waitForTimeout(200);
    }
    expect(hitDone2, 'degraded run should reach "done"').toBe(true);

    await window.waitForTimeout(500);
    await screenshotSafe(window, path.join(SHOTS, '06-final-degraded.png'));

    const degradedStats = await window.evaluate(() => {
      const canvas = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement | null;
      if (!canvas) return null;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      const d = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let nonBlack = 0;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i] > 0 || d[i + 1] > 0 || d[i + 2] > 0) nonBlack++;
      }
      return { total: d.length / 4, nonBlack };
    });
    expect(degradedStats).not.toBeNull();
    expect(degradedStats!.nonBlack).toBeGreaterThan(0);
    console.log('degraded GS non-black pixels:', degradedStats!.nonBlack, '/', degradedStats!.total);

    // ────────────────────────────────────────────────────────
    // Phase 4: Parallax generation + playback
    // ────────────────────────────────────────────────────────
    console.log('\n═══ Phase 4: Parallax generation + playback ═══');

    // Reconfigure for fast parallax: smaller grid, fewer GS iters, clean panel
    expect(await setSliderByTestId(window, 'slider-grid-size', 16)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 64)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 5)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    await computeBtn.click();
    const t3 = Date.now();
    while (Date.now() - t3 < MAX_WAIT) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(200);
    }
    expect(await getRenderState(window), 'pipeline must reach done for parallax').toBe('done');
    await window.waitForTimeout(300);
    await screenshotSafe(window, path.join(SHOTS, '07-parallax-pre.png'));

    // Generate parallax frames
    const genBtn = window.locator('[data-testid="parallax-generate-btn"]');
    await expect(genBtn).toBeVisible();
    await expect(genBtn).toBeEnabled();
    await genBtn.click();

    // Wait for parallax-state=done
    const t4 = Date.now();
    while (Date.now() - t4 < 180_000) {
      const state = await window.evaluate(() =>
        document.querySelector('[data-testid="parallax-generate-btn"]')?.getAttribute('data-parallax-state')
      );
      if (state === 'done') break;
      await window.waitForTimeout(500);
    }
    const finalState = await window.evaluate(() =>
      document.querySelector('[data-testid="parallax-generate-btn"]')?.getAttribute('data-parallax-state')
    );
    expect(finalState, 'parallax must reach done').toBe('done');

    await screenshotSafe(window, path.join(SHOTS, '08-parallax-done.png'));
    await window.waitForTimeout(500);

    // Verify frame counter shows 24 frames
    const frameCounter = window.locator('[data-testid="parallax-frame-counter"]');
    await expect(frameCounter).toBeVisible();
    const counterText = await frameCounter.textContent();
    expect(counterText).toMatch(/\/ 24$/);
    console.log('parallax frame counter:', counterText);

    // Viewport should show a parallax frame (non-black)
    const nonBlackAfter = await window.evaluate(() => {
      const c = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
      let nb = 0;
      for (let i = 0; i < d.length; i += 4) if (d[i] || d[i + 1] || d[i + 2]) nb++;
      return nb;
    });
    console.log('parallax: nonBlackAfter=', nonBlackAfter);
    expect(nonBlackAfter, 'viewport should have non-black pixels after parallax').toBeGreaterThan(0);

    // Pause playback if auto-playing
    const playBtn = window.locator('[data-testid="parallax-play-btn"]');
    const playText = await playBtn.textContent();
    if (playText?.includes('Pause')) {
      await playBtn.click();
      await window.waitForTimeout(200);
    }

    // Sample frame 0 pixels
    const pixelsFrame0 = await window.evaluate(() => {
      const c = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
      const s: number[] = [];
      for (let i = 0; i < 64; i++) s.push(d[Math.floor(i * d.length / 64)]);
      return s;
    });

    // Scrub to frame 12 and verify canvas changed
    const scrubber = window.locator('[data-testid="parallax-scrubber"]');
    await expect(scrubber).toBeVisible();
    expect(await setSliderByTestId(window, 'parallax-scrubber', 12)).toBe(true);
    await window.waitForTimeout(400);

    const pixelsFrame12 = await window.evaluate(() => {
      const c = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
      const s: number[] = [];
      for (let i = 0; i < 64; i++) s.push(d[Math.floor(i * d.length / 64)]);
      return s;
    });

    const scrubChanged = pixelsFrame0.some((v, i) => v !== pixelsFrame12[i]);
    console.log('parallax scrub changed:', scrubChanged);
    expect(scrubChanged, 'scrubbing to frame 12 should change viewport pixels').toBe(true);

    await screenshotSafe(window, path.join(SHOTS, '09-parallax-frame12.png'));

    // ────────────────────────────────────────────────────────
    // Phase 5: High-res parallax (128×128 grid, hemi=256 — uses unified memory)
    // ────────────────────────────────────────────────────────
    console.log('\n═══ Phase 5: High-res parallax (128×128, hemi=256) ═══');

    expect(await setSliderByTestId(window, 'slider-spp', 1)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 128)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 256)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 5)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);

    await computeBtn.click();
    const t5 = Date.now();
    while (Date.now() - t5 < 300_000) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(1000);
    }
    expect(await getRenderState(window), 'high-res pipeline must reach done').toBe('done');
    console.log(`high-res render+GS done in ${Date.now() - t5}ms`);
    await window.waitForTimeout(300);

    // Generate parallax frames — uses project_and_scatter from stored phase
    const genBtn2 = window.locator('[data-testid="parallax-generate-btn"]');
    await expect(genBtn2).toBeEnabled();
    await genBtn2.click();

    const t6 = Date.now();
    while (Date.now() - t6 < 300_000) {
      const state = await window.evaluate(() =>
        document.querySelector('[data-testid="parallax-generate-btn"]')?.getAttribute('data-parallax-state')
      );
      if (state === 'done') break;
      await window.waitForTimeout(1000);
    }
    const streamParallaxState = await window.evaluate(() =>
      document.querySelector('[data-testid="parallax-generate-btn"]')?.getAttribute('data-parallax-state')
    );
    expect(streamParallaxState, 'high-res parallax must reach done').toBe('done');
    console.log(`high-res parallax done in ${Date.now() - t6}ms`);

    // Pause + verify frames differ
    const playBtn2 = window.locator('[data-testid="parallax-play-btn"]');
    const pt2 = await playBtn2.textContent();
    if (pt2?.includes('Pause')) {
      await playBtn2.click();
      await window.waitForTimeout(200);
    }

    const streamPx0 = await window.evaluate(() => {
      const c = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
      const s: number[] = [];
      for (let i = 0; i < 64; i++) s.push(d[Math.floor(i * d.length / 64)]);
      return s;
    });

    expect(await setSliderByTestId(window, 'parallax-scrubber', 12)).toBe(true);
    await window.waitForTimeout(400);

    const streamPx12 = await window.evaluate(() => {
      const c = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
      const s: number[] = [];
      for (let i = 0; i < 64; i++) s.push(d[Math.floor(i * d.length / 64)]);
      return s;
    });

    const streamScrubChanged = streamPx0.some((v, i) => v !== streamPx12[i]);
    console.log('high-res parallax scrub changed:', streamScrubChanged);
    expect(streamScrubChanged, 'high-res parallax: scrubbing should change viewport pixels').toBe(true);

    await screenshotSafe(window, path.join(SHOTS, '10-highres-parallax.png'));

    // ────────────────────────────────────────────────────────
    // Phase 6: 512×512 grid (streaming mode)
    // Tests the streaming render+GS pipeline at high resolution.
    // Parallax uses PCA-compressed intensity for subsequent frames.
    // ────────────────────────────────────────────────────────
    console.log('\n═══ Phase 6: 512×512 grid (streaming mode, hemi=256) ═══');

    expect(await setSliderByTestId(window, 'slider-spp', 1)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 512)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 256)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-enabled', true)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-iterations', 3)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-phase-bits', 0)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0)).toBe(true);
    expect(await setCheckboxByTestId(window, 'gs-live-preview', false)).toBe(true);

    await window.waitForTimeout(300);
    await screenshotSafe(window, path.join(SHOTS, '11-512-configured.png'));

    await computeBtn.click();
    const t7 = Date.now();
    while (Date.now() - t7 < 600_000) {
      if (await getRenderState(window) === 'done') break;
      await window.waitForTimeout(2000);
    }
    expect(await getRenderState(window), '512×512 pipeline must reach done').toBe('done');
    console.log(`512×512 render+GS done in ${Date.now() - t7}ms`);

    await window.waitForTimeout(500);
    await screenshotSafe(window, path.join(SHOTS, '12-512-render-done.png'));

    // Verify viewport has rendered content (non-black)
    const stats512 = await window.evaluate(() => {
      const canvas = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement | null;
      if (!canvas) return null;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      const d = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let nonBlack = 0;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i] > 0 || d[i + 1] > 0 || d[i + 2] > 0) nonBlack++;
      }
      return { total: d.length / 4, nonBlack };
    });
    expect(stats512).not.toBeNull();
    expect(stats512!.nonBlack, '512×512 viewport should have non-black pixels').toBeGreaterThan(0);
    console.log('512×512 non-black pixels:', stats512!.nonBlack, '/', stats512!.total);

    // Generate parallax frames — uses PCA-compressed intensity for different eye positions
    const genBtn3 = window.locator('[data-testid="parallax-generate-btn"]');
    await expect(genBtn3).toBeEnabled();
    await genBtn3.click();

    const t8 = Date.now();
    while (Date.now() - t8 < 300_000) {
      const state = await window.evaluate(() =>
        document.querySelector('[data-testid="parallax-generate-btn"]')?.getAttribute('data-parallax-state')
      );
      if (state === 'done') break;
      await window.waitForTimeout(1000);
    }
    const parallax512State = await window.evaluate(() =>
      document.querySelector('[data-testid="parallax-generate-btn"]')?.getAttribute('data-parallax-state')
    );
    expect(parallax512State, '512×512 parallax must reach done').toBe('done');
    console.log(`512×512 parallax done in ${Date.now() - t8}ms`);

    // Pause playback + take screenshots at multiple frames
    const playBtn3 = window.locator('[data-testid="parallax-play-btn"]');
    const pt3 = await playBtn3.textContent();
    if (pt3?.includes('Pause')) {
      await playBtn3.click();
      await window.waitForTimeout(200);
    }

    // Scrub to frame 0 for a clean screenshot
    expect(await setSliderByTestId(window, 'parallax-scrubber', 0)).toBe(true);
    await window.waitForTimeout(400);
    await screenshotSafe(window, path.join(SHOTS, '13-512-parallax-frame0.png'));

    const px512_0 = await window.evaluate(() => {
      const c = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
      const s: number[] = [];
      for (let i = 0; i < 64; i++) s.push(d[Math.floor(i * d.length / 64)]);
      return s;
    });

    // Scrub to frame 12 to verify parallax disparity
    expect(await setSliderByTestId(window, 'parallax-scrubber', 12)).toBe(true);
    await window.waitForTimeout(400);
    await screenshotSafe(window, path.join(SHOTS, '14-512-parallax-frame12.png'));

    const px512_12 = await window.evaluate(() => {
      const c = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
      const s: number[] = [];
      for (let i = 0; i < 64; i++) s.push(d[Math.floor(i * d.length / 64)]);
      return s;
    });

    const scrub512Changed = px512_0.some((v, i) => v !== px512_12[i]);
    console.log('512×512 parallax scrub changed:', scrub512Changed);
    expect(scrub512Changed, '512×512 parallax: scrubbing should change viewport pixels').toBe(true);

    // Scrub to frame 23 (last frame) for a third angle
    expect(await setSliderByTestId(window, 'parallax-scrubber', 23)).toBe(true);
    await window.waitForTimeout(400);
    await screenshotSafe(window, path.join(SHOTS, '15-512-parallax-frame23.png'));

    // ────────────────────────────────────────────────────────
    // Final checks
    // ────────────────────────────────────────────────────────
    expect(pageErrors, `Page errors:\n${pageErrors.join('\n')}`).toHaveLength(0);

    await app.close();
  });
});

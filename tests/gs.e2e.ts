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

test.describe('Phase-only Holography (Gerchberg-Saxton + panel model)', () => {
  test('live iteration preview + panel degradation sweep', async () => {
    test.setTimeout(300_000);

    const app: ElectronApplication = await electron.launch({
      args: [path.join(ROOT, 'dist-electron/main.js')],
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

    // Configure a fast preset
    expect(await setSliderByTestId(window, 'slider-spp', 1)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-grid-size', 32)).toBe(true);
    expect(await setSliderByTestId(window, 'slider-hemi-res', 128)).toBe(true);

    // Enable GS, 20 iters, clean (no quant, no noise), live preview on
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

    // Degraded pass: heavy quantization + noise
    expect(await setSliderByTestId(window, 'gs-phase-bits', 3)).toBe(true);
    expect(await setSliderByTestId(window, 'gs-noise-sigma', 0.3)).toBe(true);
    await window.waitForTimeout(200);
    await screenshotSafe(window, path.join(SHOTS, '04-configured-degraded.png'));

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
    await screenshotSafe(window, path.join(SHOTS, '05-final-degraded.png'));

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

    // Hogel-hover diagnostic: move cursor across the viewport, verify the
    // Hogel Diagnostics tiles receive a non-empty hemisphere + phase preview.
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
      await window.waitForTimeout(250); // let the async fetch + state update
    }
    await window.waitForTimeout(300);
    await screenshotSafe(window, path.join(SHOTS, '06-hover-diagnostic.png'));

    // Inspect the two diagnostic canvases — they should now have drawn pixels
    const hogelDiag = await window.evaluate(() => {
      const canvases = Array.from(document.querySelectorAll('canvas'));
      // Find canvases inside the Hogel Diagnostics section (by label text nearby)
      const hd = canvases.filter(c => {
        const w = c.width, h = c.height;
        // Only small diagnostic canvases (not the main 1024x768 viewport)
        return w > 0 && w <= 512 && h > 0 && h <= 512;
      });
      const stats = hd.map(c => {
        const ctx = c.getContext('2d');
        if (!ctx) return { w: c.width, h: c.height, nonBlack: 0 };
        const d = ctx.getImageData(0, 0, c.width, c.height).data;
        let nb = 0;
        for (let i = 0; i < d.length; i += 4) if (d[i] || d[i+1] || d[i+2]) nb++;
        return { w: c.width, h: c.height, nonBlack: nb, total: d.length / 4 };
      });
      return stats;
    });
    console.log('hogel diagnostic canvases:', hogelDiag);
    // At least ONE small canvas should have drawn content (hemisphere or phase)
    const anyNonBlack = hogelDiag.some(s => s.nonBlack > 0);
    expect(anyNonBlack, 'hogel-hover should populate at least one diagnostic tile').toBe(true);

    expect(pageErrors, `Page errors:\n${pageErrors.join('\n')}`).toHaveLength(0);

    await app.close();
  });
});

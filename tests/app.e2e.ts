import { test, expect, _electron as electron } from '@playwright/test';
import path from 'path';
import { execSync } from 'child_process';
import fs from 'fs';

const ROOT = path.resolve(__dirname, '..');

// Build everything before tests run
test.beforeAll(async () => {
  // vite build produces dist/ (renderer) and dist-electron/ (main + preload)
  execSync('npx vite build', { cwd: ROOT, stdio: 'pipe', timeout: 60000 });

  // Sanity check build outputs exist
  const mainJs = path.join(ROOT, 'dist-electron/main.js');
  const indexHtml = path.join(ROOT, 'dist/index.html');
  if (!fs.existsSync(mainJs)) throw new Error(`Missing ${mainJs}`);
  if (!fs.existsSync(indexHtml)) throw new Error(`Missing ${indexHtml}`);
});

test.describe('HoloSim Electron', () => {
  test('app launches and shows UI', async () => {
    const app = await electron.launch({
      args: [path.join(ROOT, 'dist-electron/main.js')],
      env: {
        ...process.env,
        LD_LIBRARY_PATH: `/usr/local/cuda-12.9/lib64:${process.env.LD_LIBRARY_PATH || ''}`,
        NODE_ENV: 'production',
      },
    });

    const window = await app.firstWindow();
    await window.waitForLoadState('domcontentloaded');
    // Give React a moment to mount and probe native backend
    await window.waitForTimeout(3000);

    // Screenshot: initial state with native backend detected
    await window.screenshot({ path: 'tests/screenshots/01-app-launch.png', fullPage: true });

    // Header exists with HoloSim branding
    const title = await window.locator('h1').first().textContent();
    expect(title).toContain('HoloSim');

    // Capture console errors for debugging
    const pageErrors: string[] = [];
    window.on('pageerror', (err) => pageErrors.push(err.message));

    // Click the render button
    const renderBtn = window.locator('button', { hasText: /Compute Hologram|Start Global Pipeline/ });
    if (await renderBtn.isVisible()) {
      await renderBtn.click();

      // Wait for GPU render to complete (128×128 hogels × 256px ≈ 12s + IPC overhead)
      await window.waitForTimeout(60000);

      // No uncaught errors during render
      expect(pageErrors).toHaveLength(0);

      // Screenshot: after render + hologram
      await window.screenshot({ path: 'tests/screenshots/02-after-render.png', fullPage: true });

      // Verify canvas has rendered content (not all-black)
      const canvasStats = await window.evaluate(() => {
        const canvas = document.querySelector('[data-testid="main-viewport"]') as HTMLCanvasElement;
        if (!canvas) return null;
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        let nonBlack = 0;
        for (let i = 0; i < data.length; i += 4) {
          if (data[i] > 0 || data[i + 1] > 0 || data[i + 2] > 0) nonBlack++;
        }
        return { total: data.length / 4, nonBlack };
      });
      expect(canvasStats).not.toBeNull();
      expect(canvasStats!.nonBlack).toBeGreaterThan(0);
    }

    await app.close();
  });
});

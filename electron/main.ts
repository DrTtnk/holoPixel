import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'path';

// Chromium GPU sandbox fails on Linux with custom NVIDIA/CUDA drivers
app.commandLine.appendSwitch('no-sandbox');
app.commandLine.appendSwitch('disable-gpu-sandbox');

let mainWindow: BrowserWindow | null = null;
let nativeAddon: any = null;

function loadNativeAddon() {
  try {
    // napi-rs compiled addon — try multiple resolution paths
    const paths = [
      path.join(__dirname, '../native/holosim_native.node'),
      path.join(process.cwd(), 'native/holosim_native.node'),
    ];
    for (const p of paths) {
      try {
        nativeAddon = require(p);
        console.log(`[holosim] Native addon loaded from ${p}`);
        return true;
      } catch { /* try next path */ }
    }
    console.warn('[holosim] Native addon not found at any expected path');
    return false;
  } catch (e) {
    console.warn('[holosim] Native addon not available:', (e as Error).message);
    return false;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1200,
    minHeight: 800,
    title: 'HoloSim — Holographic Display Simulator',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // In dev mode, load from Vite dev server; in production, load the built index.html
  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ── IPC Handlers ──────────────────────────────────────────

ipcMain.handle('native:ping', async () => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const result = nativeAddon.ping();
    return { ok: true, data: result };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuInfo', async () => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const result = nativeAddon.gpuInfo();
    return { ok: true, data: result };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:cudaHello', async (_event, size: number) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const result = nativeAddon.cudaHello(size);
    return { ok: true, data: result };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:sceneInfo', async () => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const result = nativeAddon.getCornellBoxInfo();
    return { ok: true, data: result };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:traceTestRay', async () => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const result = nativeAddon.traceTestRay();
    return { ok: true, data: result };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:renderScene', async (_event, width: number, height: number, spp: number) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const result = nativeAddon.renderScene(width, height, spp);
    // Convert Buffer to base64 for IPC transfer (ArrayBuffer serialization is unreliable)
    return {
      ok: true,
      data: {
        width: result.width,
        height: result.height,
        samplesPerPixel: result.samplesPerPixel,
        dataBase64: result.data.toString('base64'),
      },
    };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:computeHologram', async (_event, gridW: number, gridH: number, hemiRes: number, spp: number) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const result = nativeAddon.computeHologramFull(gridW, gridH, hemiRes, spp);
    return {
      ok: true,
      data: {
        gridW: result.gridW,
        gridH: result.gridH,
        hemiRes: result.hemiRes,
        fullWidth: result.fullWidth,
        fullHeight: result.fullHeight,
        hologramBase64: result.hologramData.toString('base64'),
        reconBase64: result.reconData.toString('base64'),
        lastHogelHemiBase64: result.lastHogelHemi.toString('base64'),
        lastHogelFringeBase64: result.lastHogelFringe.toString('base64'),
      },
    };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:computeHologramGpu', async (_event, gridW: number, gridH: number, hemiRes: number, spp: number, outW: number, outH: number, maxBounces: number, ambient: number) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const result = nativeAddon.computeHologramGpu(gridW, gridH, hemiRes, spp, outW, outH, maxBounces, ambient);
    return {
      ok: true,
      data: {
        gridW: result.gridW,
        gridH: result.gridH,
        hemiRes: result.hemiRes,
        fullWidth: result.fullWidth,
        fullHeight: result.fullHeight,
        reconBase64: result.reconData.toString('base64'),
      },
    };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuSessionBegin', async (_event, gridW: number, gridH: number, hemiRes: number, spp: number, outW: number, outH: number, maxBounces: number, ambient: number) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const info = nativeAddon.gpuSessionBegin(gridW, gridH, hemiRes, spp, outW, outH, maxBounces, ambient);
    return { ok: true, data: info };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuSessionRenderRows', async (_event, startRow: number, numRows: number) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const rowsDone = nativeAddon.gpuSessionRenderRows(startRow, numRows);
    return { ok: true, data: { rowsDone } };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuSessionRunGs', async (_event, iterations: number, phaseBits: number, noiseSigma: number, noiseSeed: bigint) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    nativeAddon.gpuSessionRunGs(iterations, phaseBits, noiseSigma, noiseSeed);
    return { ok: true };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuSessionGsSetup', async (_event, noiseSeed: bigint) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    nativeAddon.gpuSessionGsSetup(noiseSeed);
    return { ok: true };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuSessionGsIterate', async (_event, nIters: number) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const itersDone = nativeAddon.gpuSessionGsIterate(nIters);
    return { ok: true, data: { itersDone } };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuSessionGsPreview', async () => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    nativeAddon.gpuSessionGsPreview();
    return { ok: true };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuSessionGsFinalize', async (_event, phaseBits: number, noiseSigma: number, noiseSeed: bigint) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    nativeAddon.gpuSessionGsFinalize(phaseBits, noiseSigma, noiseSeed);
    return { ok: true };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuSessionReconstruct', async (_event, eyeX: number, eyeY: number, eyeZ: number) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const buf = nativeAddon.gpuSessionReconstruct(eyeX, eyeY, eyeZ);
    return { ok: true, data: { reconBase64: buf.toString('base64') } };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuSessionFinish', async (_event, eyeX: number, eyeY: number, eyeZ: number) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const result = nativeAddon.gpuSessionFinish(eyeX, eyeY, eyeZ);
    return {
      ok: true,
      data: {
        gridW: result.gridW,
        gridH: result.gridH,
        hemiRes: result.hemiRes,
        fullWidth: result.outW,
        fullHeight: result.outH,
        reconBase64: result.reconData.toString('base64'),
      },
    };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuSessionClose', async () => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    nativeAddon.gpuSessionClose();
    return { ok: true };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:gpuSessionGetHogelPreview', async (_event, hx: number, hy: number) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const result = nativeAddon.gpuSessionGetHogelPreview(hx, hy);
    return {
      ok: true,
      data: {
        hx: result.hx,
        hy: result.hy,
        res: result.res,
        hemisphereBase64: result.hemisphere.toString('base64'),
        phaseBase64: result.phase ? result.phase.toString('base64') : null,
      },
    };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

ipcMain.handle('native:computeSingleHogel', async (_event, hogelX: number, hogelY: number, gridW: number, gridH: number, hemiRes: number, spp: number) => {
  if (!nativeAddon) return { ok: false, error: 'Native addon not loaded' };
  try {
    const result = nativeAddon.computeSingleHogel(hogelX, hogelY, gridW, gridH, hemiRes, spp);
    return {
      ok: true,
      data: {
        hemiRes: result.hemiRes,
        hemisphereBase64: result.hemisphereData.toString('base64'),
        fringeBase64: result.fringeData.toString('base64'),
      },
    };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
});

// ── App Lifecycle ─────────────────────────────────────────

app.whenReady().then(() => {
  loadNativeAddon();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

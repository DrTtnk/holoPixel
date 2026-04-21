import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('holosim', {
  ping: () => ipcRenderer.invoke('native:ping'),
  gpuInfo: () => ipcRenderer.invoke('native:gpuInfo'),
  cudaHello: (size: number) => ipcRenderer.invoke('native:cudaHello', size),
  sceneInfo: () => ipcRenderer.invoke('native:sceneInfo'),
  traceTestRay: () => ipcRenderer.invoke('native:traceTestRay'),
  renderScene: (width: number, height: number, spp: number) =>
    ipcRenderer.invoke('native:renderScene', width, height, spp),
  computeHologram: (gridW: number, gridH: number, hemiRes: number, spp: number) =>
    ipcRenderer.invoke('native:computeHologram', gridW, gridH, hemiRes, spp),
  computeHologramGpu: (gridW: number, gridH: number, hemiRes: number, spp: number, outW: number, outH: number, maxBounces: number, ambient: number) =>
    ipcRenderer.invoke('native:computeHologramGpu', gridW, gridH, hemiRes, spp, outW, outH, maxBounces, ambient),
  computeSingleHogel: (hogelX: number, hogelY: number, gridW: number, gridH: number, hemiRes: number, spp: number) =>
    ipcRenderer.invoke('native:computeSingleHogel', hogelX, hogelY, gridW, gridH, hemiRes, spp),
  gpuSessionBegin: (gridW: number, gridH: number, hemiRes: number, spp: number, outW: number, outH: number, maxBounces: number, ambient: number) =>
    ipcRenderer.invoke('native:gpuSessionBegin', gridW, gridH, hemiRes, spp, outW, outH, maxBounces, ambient),
  gpuSessionRenderRows: (startRow: number, numRows: number) =>
    ipcRenderer.invoke('native:gpuSessionRenderRows', startRow, numRows),
  gpuSessionRunGs: (iterations: number, phaseBits: number, noiseSigma: number, noiseSeed: bigint) =>
    ipcRenderer.invoke('native:gpuSessionRunGs', iterations, phaseBits, noiseSigma, noiseSeed),
  gpuSessionGsSetup: (noiseSeed: bigint) =>
    ipcRenderer.invoke('native:gpuSessionGsSetup', noiseSeed),
  gpuSessionGsIterate: (nIters: number) =>
    ipcRenderer.invoke('native:gpuSessionGsIterate', nIters),
  gpuSessionGsPreview: () =>
    ipcRenderer.invoke('native:gpuSessionGsPreview'),
  gpuSessionGsFinalize: (phaseBits: number, noiseSigma: number, noiseSeed: bigint) =>
    ipcRenderer.invoke('native:gpuSessionGsFinalize', phaseBits, noiseSigma, noiseSeed),
  gpuSessionReconstruct: (eyeX: number, eyeY: number, eyeZ: number) =>
    ipcRenderer.invoke('native:gpuSessionReconstruct', eyeX, eyeY, eyeZ),
  gpuSessionFinish: (eyeX: number, eyeY: number, eyeZ: number) =>
    ipcRenderer.invoke('native:gpuSessionFinish', eyeX, eyeY, eyeZ),
  gpuSessionClose: () =>
    ipcRenderer.invoke('native:gpuSessionClose'),
  gpuSessionGetHogelPreview: (hx: number, hy: number) =>
    ipcRenderer.invoke('native:gpuSessionGetHogelPreview', hx, hy),
});

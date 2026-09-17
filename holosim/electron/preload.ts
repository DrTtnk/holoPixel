import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('holosim', {
  ping: () => ipcRenderer.invoke('native:ping'),
  gpuInfo: () => ipcRenderer.invoke('native:gpuInfo'),
  cudaHello: (size: number) => ipcRenderer.invoke('native:cudaHello', size),
  sceneInfo: () => ipcRenderer.invoke('native:sceneInfo'),
  traceTestRay: () => ipcRenderer.invoke('native:traceTestRay'),
  gpuSessionBegin: (gridW: number, gridH: number, hemiRes: number, spp: number, outW: number, outH: number, maxBounces: number, ambient: number) =>
    ipcRenderer.invoke('native:gpuSessionBegin', gridW, gridH, hemiRes, spp, outW, outH, maxBounces, ambient),
  gpuSessionRenderRows: (startRow: number, numRows: number) =>
    ipcRenderer.invoke('native:gpuSessionRenderRows', startRow, numRows),
  gpuSessionStreamBatch: (startRow: number, numRows: number, gsIterations: number, phaseBits: number, noiseSigma: number, noiseSeed: bigint) =>
    ipcRenderer.invoke('native:gpuSessionStreamBatch', startRow, numRows, gsIterations, phaseBits, noiseSigma, noiseSeed),
  gpuSessionStreamBatchProfile: (startRow: number, numRows: number, gsIterations: number, phaseBits: number, noiseSigma: number, noiseSeed: bigint) =>
    ipcRenderer.invoke('native:gpuSessionStreamBatchProfile', startRow, numRows, gsIterations, phaseBits, noiseSigma, noiseSeed),
  gpuSessionPcaEigenspectrum: (maxComponents: number) =>
    ipcRenderer.invoke('native:gpuSessionPcaEigenspectrum', maxComponents),
  gpuSessionPcaTargetAmp: (maxComponents: number) =>
    ipcRenderer.invoke('native:gpuSessionPcaTargetAmp', maxComponents),
  gpuSessionPcaHemisphere: (maxComponents: number) =>
    ipcRenderer.invoke('native:gpuSessionPcaHemisphere', maxComponents),
  gpuSessionPcaCompressTargetAmp: (k: number) =>
    ipcRenderer.invoke('native:gpuSessionPcaCompressTargetAmp', k),
  gpuSessionSaveTargetAmp: () =>
    ipcRenderer.invoke('native:gpuSessionSaveTargetAmp'),
  gpuSessionRestoreTargetAmp: () =>
    ipcRenderer.invoke('native:gpuSessionRestoreTargetAmp'),
  gpuSessionCompressIntensity: () =>
    ipcRenderer.invoke('native:gpuSessionCompressIntensity'),
  gpuSessionEnablePcaSampling: (numSamples: number) =>
    ipcRenderer.invoke('native:gpuSessionEnablePcaSampling', numSamples),
  gpuSessionPcaStreamingEigenspectrum: (maxComponents: number) =>
    ipcRenderer.invoke('native:gpuSessionPcaStreamingEigenspectrum', maxComponents),
  gpuSessionPcaTiledTargetAmp: (tileSize: number, maxComponents: number) =>
    ipcRenderer.invoke('native:gpuSessionPcaTiledTargetAmp', tileSize, maxComponents),
  gpuSessionPcaCompressQuantized: (k: number, bits: number) =>
    ipcRenderer.invoke('native:gpuSessionPcaCompressQuantized', k, bits),
  gpuSessionPcaQuantizationSweep: (kValues: number[], bitValues: number[]) =>
    ipcRenderer.invoke('native:gpuSessionPcaQuantizationSweep', kValues, bitValues),
  gpuSessionGsSetup: (noiseSeed: bigint) =>
    ipcRenderer.invoke('native:gpuSessionGsSetup', noiseSeed),
  gpuSessionGsIterate: (nIters: number) =>
    ipcRenderer.invoke('native:gpuSessionGsIterate', nIters),
  gpuSessionGsPreview: () =>
    ipcRenderer.invoke('native:gpuSessionGsPreview'),
  gpuSessionGsIterateAndReconstruct: (nIters: number, eyeX: number, eyeY: number, eyeZ: number) =>
    ipcRenderer.invoke('native:gpuSessionGsIterateAndReconstruct', nIters, eyeX, eyeY, eyeZ),
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

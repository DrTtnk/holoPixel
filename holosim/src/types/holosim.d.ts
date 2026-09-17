// Type declarations for the holosim native bridge exposed via preload
interface NativeResult<T = unknown> {
  ok: boolean;
  data?: T;
  error?: string;
}

interface HoloSimBridge {
  ping(): Promise<NativeResult<string>>;
  gpuInfo(): Promise<NativeResult<{ name: string; totalMemoryMb: number; computeCapability: string }>>;
  cudaHello(size: number): Promise<NativeResult<{ sum: number; expectedSum: number; match: boolean }>>;
  sceneInfo(): Promise<NativeResult<{ triangleCount: number; materialCount: number; sceneName: string }>>;
  traceTestRay(): Promise<NativeResult<{
    hit: boolean;
    distance: number;
    normal: number[];
    materialIdx: number;
    materialAlbedo: number[];
    isEmissive: boolean;
  }>>;
  gpuSessionBegin(gridW: number, gridH: number, hemiRes: number, spp: number, outW: number, outH: number, maxBounces: number, ambient: number): Promise<NativeResult<{
    gridW: number; gridH: number; hemiRes: number; spp: number; outW: number; outH: number; totalRows: number; streaming: boolean;
  }>>;
  gpuSessionRenderRows(startRow: number, numRows: number): Promise<NativeResult<{ rowsDone: number }>>;
  gpuSessionStreamBatch(startRow: number, numRows: number, gsIterations: number, phaseBits: number, noiseSigma: number, noiseSeed: bigint): Promise<NativeResult<{ rowsDone: number }>>;
  gpuSessionStreamBatchProfile(startRow: number, numRows: number, gsIterations: number, phaseBits: number, noiseSigma: number, noiseSeed: bigint): Promise<NativeResult<{
    rowsDone: number;
    timings: Array<{ name: string; ms: number }>;
  }>>;
  gpuSessionPcaEigenspectrum(maxComponents: number): Promise<NativeResult<{
    eigenvalues: number[]; cumVar: number[]; numHogels: number; pixelsPerHogel: number;
  }>>;
  gpuSessionPcaTargetAmp(maxComponents: number): Promise<NativeResult<{
    eigenvalues: number[]; cumVar: number[]; numHogels: number; pixelsPerHogel: number;
  }>>;
  gpuSessionPcaHemisphere(maxComponents: number): Promise<NativeResult<{
    eigenvalues: number[]; cumVar: number[]; numHogels: number; pixelsPerHogel: number;
  }>>;
  gpuSessionPcaCompressTargetAmp(k: number): Promise<NativeResult<{
    kUsed: number; compressionRatio: number; rmse: number;
  }>>;
  gpuSessionSaveTargetAmp(): Promise<NativeResult<void>>;
  gpuSessionRestoreTargetAmp(): Promise<NativeResult<void>>;
  gpuSessionCompressIntensity(): Promise<NativeResult<void>>;
  gpuSessionEnablePcaSampling(numSamples: number): Promise<NativeResult<void>>;
  gpuSessionPcaStreamingEigenspectrum(maxComponents: number): Promise<NativeResult<{
    eigenvalues: number[]; cumVar: number[]; numHogels: number; pixelsPerHogel: number;
  }>>;
  gpuSessionPcaTiledTargetAmp(tileSize: number, maxComponents: number): Promise<NativeResult<{
    numTiles: number; tileSize: number; hogelsPerTile: number; pixelsPerHogel: number; totalHogels: number;
    targets: number[];
    tileMedianK: number[]; tileMaxK: number[]; tileMinK: number[]; tileMeanK: number[];
    tiledTotalBytes: number[];
    flatK: number[]; flatTotalBytes: number[];
    originalBytes: number;
    perTileK: number[];
    flatEigenvalues: number[]; flatCumVar: number[];
  }>>;
  gpuSessionPcaCompressQuantized(k: number, bits: number): Promise<NativeResult<{
    kUsed: number; compressionRatio: number; rmse: number; compressedBytes: number;
  }>>;
  gpuSessionPcaQuantizationSweep(kValues: number[], bitValues: number[]): Promise<NativeResult<Array<{
    k: number; bits: number; compressionRatio: number; rmse: number; compressedBytes: number;
  }>>>;
  gpuSessionGsSetup(noiseSeed: bigint): Promise<NativeResult<void>>;
  gpuSessionGsIterate(nIters: number): Promise<NativeResult<{ itersDone: number }>>;
  gpuSessionGsPreview(): Promise<NativeResult<void>>;
  gpuSessionGsIterateAndReconstruct(nIters: number, eyeX: number, eyeY: number, eyeZ: number): Promise<NativeResult<{ itersDone: number; reconBuffer: Buffer }>>;
  gpuSessionGsFinalize(phaseBits: number, noiseSigma: number, noiseSeed: bigint): Promise<NativeResult<void>>;
  gpuSessionReconstruct(eyeX: number, eyeY: number, eyeZ: number): Promise<NativeResult<{ reconBuffer: Buffer }>>;
  gpuSessionFinish(eyeX: number, eyeY: number, eyeZ: number): Promise<NativeResult<{
    gridW: number; gridH: number; hemiRes: number; fullWidth: number; fullHeight: number; reconBase64: string;
  }>>;
  gpuSessionClose(): Promise<NativeResult<void>>;
  gpuSessionGetHogelPreview(hx: number, hy: number): Promise<NativeResult<{
    hx: number; hy: number; res: number;
    hemisphereBase64: string; // RGB u8, res*res*3 bytes
    phaseBase64: string | null; // grayscale u8, res*res bytes, or null if GS hasn't run
  }>>;
}

declare global {
  interface Window {
    holosim?: HoloSimBridge;
  }
}

export {};

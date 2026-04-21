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
  renderScene(width: number, height: number, spp: number): Promise<NativeResult<{
    width: number;
    height: number;
    samplesPerPixel: number;
    dataBase64: string;
  }>>;
  computeHologram(gridW: number, gridH: number, hemiRes: number, spp: number): Promise<NativeResult<{
    gridW: number;
    gridH: number;
    hemiRes: number;
    fullWidth: number;
    fullHeight: number;
    hologramBase64: string;
    reconBase64: string;
    lastHogelHemiBase64: string;
    lastHogelFringeBase64: string;
  }>>;
  computeHologramGpu(gridW: number, gridH: number, hemiRes: number, spp: number, outW: number, outH: number, maxBounces: number, ambient: number): Promise<NativeResult<{
    gridW: number;
    gridH: number;
    hemiRes: number;
    fullWidth: number;
    fullHeight: number;
    reconBase64: string;
  }>>;
  computeSingleHogel(hogelX: number, hogelY: number, gridW: number, gridH: number, hemiRes: number, spp: number): Promise<NativeResult<{
    hemiRes: number;
    hemisphereBase64: string;
    fringeBase64: string;
  }>>;
  gpuSessionBegin(gridW: number, gridH: number, hemiRes: number, spp: number, outW: number, outH: number, maxBounces: number, ambient: number): Promise<NativeResult<{
    gridW: number; gridH: number; hemiRes: number; spp: number; outW: number; outH: number; totalRows: number;
  }>>;
  gpuSessionRenderRows(startRow: number, numRows: number): Promise<NativeResult<{ rowsDone: number }>>;
  gpuSessionRunGs(iterations: number, phaseBits: number, noiseSigma: number, noiseSeed: bigint): Promise<NativeResult<void>>;
  gpuSessionGsSetup(noiseSeed: bigint): Promise<NativeResult<void>>;
  gpuSessionGsIterate(nIters: number): Promise<NativeResult<{ itersDone: number }>>;
  gpuSessionGsPreview(): Promise<NativeResult<void>>;
  gpuSessionGsFinalize(phaseBits: number, noiseSigma: number, noiseSeed: bigint): Promise<NativeResult<void>>;
  gpuSessionReconstruct(eyeX: number, eyeY: number, eyeZ: number): Promise<NativeResult<{ reconBase64: string }>>;
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

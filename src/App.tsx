/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Aperture, Layers, Cpu, Play, Pause, Zap } from "lucide-react";
import { useEffect, useRef, useState, useCallback } from "react";
import { SimulationEngine } from "./simulator/Engine";

import { DiagnosticsVisualizer } from "./components/DiagnosticsVisualizer";

const PARALLAX_FRAMES = 24;
const PARALLAX_EYE_CENTER = { x: 278, y: 273, z: -800 } as const;

function PreviewCanvas({ data, width, height, label }: { data?: Uint8ClampedArray, width: number, height: number, label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (data && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.putImageData(new ImageData(data, width, height), 0, 0);
      }
    }
  }, [data, width, height]);

  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-neutral-500 font-mono">{label}</span>
      <canvas 
        ref={canvasRef} 
        width={width} 
        height={height} 
        className="w-full aspect-square bg-neutral-950 border border-neutral-800 rounded-lg" 
        style={{ imageRendering: 'pixelated' }}
      />
    </div>
  );
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<SimulationEngine | null>(null);
  const [running, setRunning] = useState(false);
  const [nativeProbed, setNativeProbed] = useState(false);
  const [nativeStatus, setNativeStatus] = useState<{
    available: boolean;
    ping?: string;
    gpu?: { name: string; totalMemoryMb: number; computeCapability: string };
    cudaOk?: boolean;
    scene?: { triangleCount: number; materialCount: number; sceneName: string };
    traceHit?: boolean;
  }>({ available: false });
  const [renderState, setRenderState] = useState<'idle' | 'rendering' | 'done'>('idle');
  const [renderTime, setRenderTime] = useState(0);
  const [renderProgress, setRenderProgress] = useState(0); // 0-1
  const [renderEta, setRenderEta] = useState<number | null>(null); // ms
  const renderStartRef = useRef<number>(0);
  const renderTotalRowsRef = useRef<number>(128);
  // Light transport params (research knobs)
  const [maxBounces] = useState(0);
  const [ambient] = useState(0.08);
  const [spp, setSpp] = useState(64);
  const [gridSize, setGridSize] = useState(128);
  const [hemiRes, setHemiRes] = useState(256);
  // Phase-only holography (Gerchberg-Saxton + panel model)
  const [gsEnabled, setGsEnabled] = useState(true);
  const [gsIterations, setGsIterations] = useState(10);
  const [gsPhaseBits, setGsPhaseBits] = useState(0); // 0 = no quantization
  const [gsNoiseSigma, setGsNoiseSigma] = useState(0.0); // radians
  const [gsLivePreview, setGsLivePreview] = useState(true);
  const [gsPreviewDelayMs] = useState(0); // unused, kept for API compat
  const [gsProgress, setGsProgress] = useState(0);
  const [gsItersDone, setGsItersDone] = useState(0);
  const [gsPhase, setGsPhase] = useState<'idle' | 'running'>('idle');

  // Parallax test
  const [parallaxState, setParallaxState] = useState<'idle' | 'generating' | 'done'>('idle');
  const [parallaxProgress, setParallaxProgress] = useState(0);
  const [parallaxFrames, setParallaxFrames] = useState<Uint8ClampedArray[]>([]);
  const [parallaxFrameIdx, setParallaxFrameIdx] = useState(0);
  const [parallaxPlaying, setParallaxPlaying] = useState(false);
  const [parallaxOrbitRadius, setParallaxOrbitRadius] = useState(200); // mm
  const [parallaxOrbitAxis, setParallaxOrbitAxis] = useState<'horizontal' | 'vertical'>('horizontal');
  const parallaxRafRef = useRef<number | null>(null);
  const parallaxCanvasesRef = useRef<HTMLCanvasElement[]>([]);
  const parallaxFrameIdxRef = useRef(0); // mutable, used by rAF loop

  // Hover-on-hogel diagnostic preview
  const [hogelPreview, setHogelPreview] = useState<{
    hx: number;
    hy: number;
    res: number;
    hemisphereRgba: Uint8ClampedArray;
    phaseRgba: Uint8ClampedArray | null;
  } | null>(null);
  const hogelFetchBusyRef = useRef(false);
  const hogelLastKeyRef = useRef<string>('');
  const hemispheresDoneRef = useRef(false);
  const [holoState, setHoloState] = useState<{
    running: boolean;
    hemisphere?: Uint8ClampedArray;
    fringe?: Uint8ClampedArray;
    hologram?: Uint8ClampedArray;
    hemiRes?: number;
    gridW?: number;
    gridH?: number;
    fullWidth?: number;
    fullHeight?: number;
    time?: number;
  }>({ running: false });
  const [stats, setStats] = useState({
    computed: 0,
    total: 480000,
    subpixels: 0,
    currentX: 0,
    currentY: 0,
    hogelsPerSecond: 0,
    previews: null as any
  });

  // Initialize engine (skip when native backend handles rendering)
  useEffect(() => {
    if (!nativeProbed || !canvasRef.current || nativeStatus.available) return;
    
    try {
      const engine = new SimulationEngine(canvasRef.current);
      engine.onProgressUpdate = setStats;
      engineRef.current = engine;
      // Do a single draw to show the black sensor canvas
      engine.computeNextPatch(0); 
    } catch (e) {
      console.error(e);
    }
    
    return () => {
      engineRef.current = null;
    };
  }, [nativeProbed, nativeStatus.available]);

  // Probe native backend (Electron + Rust + CUDA)
  useEffect(() => {
    if (!window.holosim) {
      setNativeProbed(true);
      return;
    }
    (async () => {
      const pingResult = await window.holosim!.ping();
      if (!pingResult.ok) {
        setNativeProbed(true);
        return;
      }

      const gpuResult = await window.holosim!.gpuInfo();
      const cudaResult = await window.holosim!.cudaHello(1024);
      const sceneResult = await window.holosim!.sceneInfo();
      const traceResult = await window.holosim!.traceTestRay();

      setNativeStatus({
        available: true,
        ping: pingResult.data,
        gpu: gpuResult.ok ? gpuResult.data : undefined,
        cudaOk: cudaResult.ok && cudaResult.data?.match,
        scene: sceneResult.ok ? sceneResult.data : undefined,
        traceHit: traceResult.ok ? traceResult.data?.hit : undefined,
      });
      setNativeProbed(true);
    })();
  }, []);

  // Main compute loop
  useEffect(() => {
    let handle: number;
    const loop = () => {
      if (running && engineRef.current && engineRef.current.currentHogelY < 600) {
        // Compute chunks. Number of patches per frame determines interaction speed.
        // Higher = faster accumulation, lower = slower but browser stays responsive.
        engineRef.current.computeNextPatch(16);
        handle = requestAnimationFrame(loop);
      }
    };
    if (running) handle = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(handle);
  }, [running]);

  const decode = useCallback((b64: string) => {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Uint8ClampedArray(arr.buffer);
  }, []);

  const drawRecon = useCallback((reconData: Uint8ClampedArray, fw: number, fh: number) => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    const offscreen = document.createElement('canvas');
    offscreen.width = fw;
    offscreen.height = fh;
    const offCtx = offscreen.getContext('2d')!;
    offCtx.putImageData(new ImageData(reconData, fw, fh), 0, 0);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offscreen, 0, 0, canvasRef.current.width, canvasRef.current.height);
  }, []);

  // Reconstruct with current eye position (fast ~100ms) without re-rendering hemispheres
  // Fetch hogel preview (hemisphere + phase) for a given hogel index.
  // Throttled: drops requests if a previous fetch is still in flight.
  const fetchHogelPreview = useCallback(async (hx: number, hy: number) => {
    if (!window.holosim) return;
    const key = `${hx},${hy}`;
    if (key === hogelLastKeyRef.current) return;
    if (hogelFetchBusyRef.current) return;
    hogelLastKeyRef.current = key;
    hogelFetchBusyRef.current = true;
    try {
      const r = await window.holosim.gpuSessionGetHogelPreview(hx, hy);
      if (!r.ok || !r.data) return;
      const { res, hemisphereBase64, phaseBase64 } = r.data;
      // Decode hemisphere (RGB u8) → RGBA
      const hemiBin = atob(hemisphereBase64);
      const hemiRgba = new Uint8ClampedArray(res * res * 4);
      for (let i = 0, j = 0; i < res * res; i++, j += 3) {
        hemiRgba[i * 4] = hemiBin.charCodeAt(j);
        hemiRgba[i * 4 + 1] = hemiBin.charCodeAt(j + 1);
        hemiRgba[i * 4 + 2] = hemiBin.charCodeAt(j + 2);
        hemiRgba[i * 4 + 3] = 255;
      }
      let phaseRgba: Uint8ClampedArray | null = null;
      if (phaseBase64) {
        const phBin = atob(phaseBase64);
        phaseRgba = new Uint8ClampedArray(res * res * 4);
        for (let i = 0; i < res * res; i++) {
          const v = phBin.charCodeAt(i);
          phaseRgba[i * 4] = v;
          phaseRgba[i * 4 + 1] = v;
          phaseRgba[i * 4 + 2] = v;
          phaseRgba[i * 4 + 3] = 255;
        }
      }
      setHogelPreview({ hx, hy, res, hemisphereRgba: hemiRgba, phaseRgba });
    } finally {
      hogelFetchBusyRef.current = false;
    }
  }, []);

  // Auto-play parallax frames via rAF loop (bypasses React render cycle for smooth playback)
  useEffect(() => {
    if (!parallaxPlaying || parallaxCanvasesRef.current.length === 0 || !canvasRef.current) {
      if (parallaxRafRef.current) { cancelAnimationFrame(parallaxRafRef.current); parallaxRafRef.current = null; }
      return;
    }
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    const total = parallaxCanvasesRef.current.length;
    let lastTime = 0;
    const MS_PER_FRAME = 80;

    const tick = (now: number) => {
      if (now - lastTime >= MS_PER_FRAME) {
        lastTime = now;
        parallaxFrameIdxRef.current = (parallaxFrameIdxRef.current + 1) % total;
        const src = parallaxCanvasesRef.current[parallaxFrameIdxRef.current];
        ctx.drawImage(src, 0, 0, canvas.width, canvas.height);
      }
      parallaxRafRef.current = requestAnimationFrame(tick);
    };
    parallaxRafRef.current = requestAnimationFrame(tick);
    return () => {
      if (parallaxRafRef.current) { cancelAnimationFrame(parallaxRafRef.current); parallaxRafRef.current = null; }
      setParallaxFrameIdx(parallaxFrameIdxRef.current); // sync scrub slider once on stop
    };
  }, [parallaxPlaying]);

  // Pre-render parallax frames to offscreen canvases for zero-cost playback
  useEffect(() => {
    if (parallaxFrames.length === 0) { parallaxCanvasesRef.current = []; parallaxFrameIdxRef.current = 0; return; }
    const OUT = holoState.fullWidth ?? 1024;
    parallaxCanvasesRef.current = parallaxFrames.map(frame => {
      const c = document.createElement('canvas');
      c.width = OUT; c.height = OUT;
      c.getContext('2d')!.putImageData(new ImageData(frame, OUT, OUT), 0, 0);
      return c;
    });
  }, [parallaxFrames, holoState.fullWidth]);

  // Draw current parallax frame to main viewport canvas (manual scrub only)
  useEffect(() => {
    if (parallaxPlaying) return; // rAF loop handles this during playback
    const prebuilt = parallaxCanvasesRef.current[parallaxFrameIdx];
    if (!prebuilt || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d')!;
    ctx.drawImage(prebuilt, 0, 0, canvasRef.current.width, canvasRef.current.height);
  }, [parallaxFrameIdx, parallaxPlaying]);

  // Generate parallax sequence: reconstruct from 24 positions on a circle
  const generateParallax = useCallback(async () => {
    if (!window.holosim || renderState !== 'done') return;
    setParallaxState('generating');
    setParallaxProgress(0);
    setParallaxFrames([]);
    setParallaxFrameIdx(0);
    setParallaxPlaying(false);

    const frames: Uint8ClampedArray[] = [];
    const OUT = holoState.fullWidth ?? 1024;
    const { x: ex, y: ey, z: ez } = PARALLAX_EYE_CENTER;
    const r = parallaxOrbitRadius;

    for (let i = 0; i < PARALLAX_FRAMES; i++) {
      const t = (i / PARALLAX_FRAMES) * Math.PI * 2;
      const eyeX = parallaxOrbitAxis === 'horizontal' ? ex + Math.cos(t) * r : ex;
      const eyeY = parallaxOrbitAxis === 'vertical'   ? ey + Math.sin(t) * r : ey;
      const eyeZ = parallaxOrbitAxis === 'horizontal' ? ez + Math.sin(t) * r : ez + Math.cos(t) * r;

      const r2 = await window.holosim!.gpuSessionReconstruct(eyeX, eyeY, eyeZ);
      if (!r2.ok || !r2.data) break;
      frames.push(decode(r2.data.reconBase64));
      setParallaxProgress((i + 1) / PARALLAX_FRAMES);
    }

    setParallaxFrames(frames);
    setParallaxState('done');
    setParallaxPlaying(true);
  }, [renderState, parallaxOrbitRadius, parallaxOrbitAxis, holoState.fullWidth, decode]);

  // Mouse handler on main viewport → pick hogel under cursor, fetch preview
  const handleReconMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (renderState !== 'done' || !holoState.gridW || !holoState.gridH) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1) return;
    const hx = Math.min(holoState.gridW - 1, Math.max(0, Math.floor(fx * holoState.gridW)));
    const hy = Math.min(holoState.gridH - 1, Math.max(0, Math.floor(fy * holoState.gridH)));
    fetchHogelPreview(hx, hy);
  }, [renderState, holoState.gridW, holoState.gridH, fetchHogelPreview]);


  const toggleCompute = useCallback(() => {
    if (nativeStatus.available && window.holosim && renderState !== 'rendering') {
      setRenderState('rendering');
      setHoloState(s => ({ ...s, running: true }));
      setRenderProgress(0);
      setRenderEta(null);
      renderStartRef.current = performance.now();
      hemispheresDoneRef.current = false;

      const BATCH_ROWS = 4;
      const GRID = gridSize;
      const SPP = spp;
      const HEMI = hemiRes;
      const OUT = 1024;
      const BOUNCES = maxBounces;
      const AMB = ambient;
      const EX = 278, EY = 273, EZ = -800;

      (async () => {
        // Close any previous session (frees GPU memory) — idempotent if none
        try { await window.holosim!.gpuSessionClose(); } catch {}
        setHogelPreview(null);
        const beginResult = await window.holosim!.gpuSessionBegin(GRID, GRID, HEMI, SPP, OUT, OUT, BOUNCES, AMB);
        if (!beginResult.ok || !beginResult.data) {
          setRenderState('idle');
          setHoloState(s => ({ ...s, running: false }));
          return;
        }
        const totalRows = beginResult.data.totalRows;
        renderTotalRowsRef.current = totalRows;

        // Render row batches sequentially, reporting progress
        for (let row = 0; row < totalRows; row += BATCH_ROWS) {
          const batchResult = await window.holosim!.gpuSessionRenderRows(row, Math.min(BATCH_ROWS, totalRows - row));
          if (!batchResult.ok || !batchResult.data) break;
          const rowsDone = batchResult.data.rowsDone;
          const frac = rowsDone / totalRows;
          setRenderProgress(frac);
          const elapsed = performance.now() - renderStartRef.current;
          if (rowsDone > 0) {
            const msPerRow = elapsed / rowsDone;
            const remaining = (totalRows - rowsDone) * msPerRow;
            setRenderEta(Math.round(remaining));
          }
        }

        // Phase-only holography step (Gerchberg-Saxton + panel model)
        if (gsEnabled) {
          setGsPhase('running');
          setGsProgress(0);
          setGsItersDone(0);
          const seed = BigInt(Math.floor(Math.random() * 0xffff_ffff));
          const setupR = await window.holosim!.gpuSessionGsSetup(seed);
          if (!setupR.ok) {
            console.error('GS setup failed:', setupR.error);
          } else if (gsIterations > 0) {
            // 1 iter per chunk: every step shows in the UI, convergence is smoothly animated.
            const chunk = 1;
            for (let done = 0; done < gsIterations; done += chunk) {
              const thisChunk = Math.min(chunk, gsIterations - done);
              const r = await window.holosim!.gpuSessionGsIterate(thisChunk);
              if (!r.ok || !r.data) break;
              setGsItersDone(r.data.itersDone);
              setGsProgress(r.data.itersDone / gsIterations);
              // Live preview between chunks (not on the last chunk — that's finalize)
              if (gsLivePreview && done + thisChunk < gsIterations) {
                await window.holosim!.gpuSessionGsPreview();
                const prev = await window.holosim!.gpuSessionReconstruct(EX, EY, EZ);
                if (prev.ok && prev.data) {
                  drawRecon(decode(prev.data.reconBase64), OUT, OUT);
                }
              }
              // Yield to event loop between iterations so UI stays responsive.
              await new Promise(resolve => setTimeout(resolve, gsLivePreview ? gsPreviewDelayMs : 0));
            }
          }
          // Finalize: apply panel model + project to hemi_dev
          const finSeed = BigInt(Math.floor(Math.random() * 0xffff_ffff));
          const finR = await window.holosim!.gpuSessionGsFinalize(gsPhaseBits, gsNoiseSigma, finSeed);
          if (!finR.ok) {
            console.error('GS finalize failed:', finR.error);
          }
          setGsProgress(1);
          setGsPhase('idle');
        }

        // Reconstruct (non-consuming so hogel-preview hover works)
        const finishResult = await window.holosim!.gpuSessionReconstruct(EX, EY, EZ);
        if (finishResult.ok && finishResult.data) {
          drawRecon(decode(finishResult.data.reconBase64), OUT, OUT);
          const elapsed = Math.round(performance.now() - renderStartRef.current);
          setHoloState({
            running: false,
            gridW: GRID,
            gridH: GRID,
            hemiRes: HEMI,
            fullWidth: OUT,
            fullHeight: OUT,
            time: elapsed,
          });
          setRenderTime(elapsed);
          setRenderProgress(1);
          setRenderEta(0);
          setRenderState('done');
          // Session stays alive — hogel-preview hover will query it until next render
        } else {
          setRenderState('idle');
          setHoloState(s => ({ ...s, running: false }));
        }
      })();
      return;
    }
    setRunning(r => !r);
  }, [nativeStatus.available, renderState, spp, gridSize, hemiRes, decode, drawRecon, gsEnabled, gsIterations, gsPhaseBits, gsNoiseSigma, gsLivePreview, maxBounces, ambient]);

  const progressPct = nativeStatus.available
    ? (renderProgress * 100).toFixed(1)
    : ((stats.computed / stats.total) * 100).toFixed(2);
  const billionsSubpixels = (stats.subpixels / 1000000000).toFixed(2);

  return (
    <div className="h-screen overflow-hidden bg-neutral-950 text-neutral-200 font-sans flex flex-col">
      {/* Header */}
      <header className="border-b border-neutral-800 bg-neutral-900 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-500/10 rounded-lg">
            <Aperture className={`w-6 h-6 text-indigo-400 ${running ? 'animate-spin-slow' : ''}`} />
          </div>
          <div>
            <h1 className="font-semibold text-lg tracking-tight text-white">HoloSim</h1>
            <p className="text-xs text-neutral-400 font-mono">Phase & Intensity Optical Reconstructor</p>
          </div>
        </div>
        
        <div className="flex gap-4">
          {nativeStatus.available && (
            <div className="flex flex-col items-end">
              <span className="text-xs text-neutral-500 font-mono tracking-wider uppercase flex items-center gap-1">
                <Zap className="w-3 h-3 text-amber-400" />
                Native Backend
              </span>
              <span className="text-sm font-medium text-amber-400">
                {nativeStatus.gpu?.name?.replace('NVIDIA ', '') || 'CUDA'} — {nativeStatus.cudaOk ? 'OK' : 'Error'}
              </span>
            </div>
          )}
          {nativeStatus.available && <div className="w-px h-8 bg-neutral-800"></div>}
          <div className="flex flex-col items-end">
            <span className="text-xs text-neutral-500 font-mono tracking-wider uppercase">Panel Size</span>
            <span className="text-sm font-medium">800 × 600</span>
          </div>
          <div className="w-px h-8 bg-neutral-800"></div>
          <div className="flex flex-col items-end">
            <span className="text-xs text-neutral-500 font-mono tracking-wider uppercase">Subpixel Matrix</span>
            <span className="text-sm font-medium text-emerald-400">512 × 512</span>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 flex overflow-hidden min-h-0">
        {/* Left Sidebar - Controls Setup */}
        <aside className="w-80 border-r border-neutral-800 bg-neutral-900/50 p-6 flex flex-col gap-6 overflow-y-auto">
          
          <section className="space-y-4">
            <h2 className="text-sm font-medium text-neutral-400 uppercase tracking-wider flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              Compute State
            </h2>
            
            <button
              onClick={toggleCompute}
              disabled={renderState === 'rendering'}
              data-testid="compute-btn"
              data-render-state={renderState}
              className={`w-full flex items-center justify-center gap-2 py-3 rounded-lg font-medium transition-colors ${
                running || renderState === 'rendering'
                  ? 'bg-neutral-800 hover:bg-neutral-700 text-white border border-neutral-700'
                  : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-900/20'
              }`}
            >
              {renderState === 'rendering' ? (
                <><Aperture className="w-4 h-4 animate-spin" /> Rendering...</>
              ) : running ? (
                <><Pause className="w-4 h-4" /> Pause FFT Accumulator</>
              ) : (
                <><Play className="w-4 h-4" /> {nativeStatus.available ? 'Compute Hologram' : 'Start Global Pipeline'}</>
              )}
            </button>

            <div className="space-y-3 pt-2">
              <div>
                <div className="flex justify-between text-xs text-neutral-400 mb-1">
                  <span>Progress</span>
                  <span className="font-mono text-indigo-300">{progressPct}%</span>
                </div>
                <div className="h-1.5 w-full bg-neutral-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-indigo-500 transition-all duration-300"
                    style={{ width: `${progressPct}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-xs">
                {nativeStatus.available ? (
                  <>
                    {renderEta !== null && renderEta > 0 && (
                      <div className="p-3 bg-neutral-900 border border-neutral-800 rounded-lg flex flex-col col-span-2">
                        <span className="text-neutral-500 mb-1">ETA</span>
                        <span className="font-mono text-amber-400">{renderEta < 60000 ? `${(renderEta/1000).toFixed(0)}s` : `${(renderEta/60000).toFixed(1)}m`}</span>
                      </div>
                    )}
                    {renderTime > 0 && (
                      <div className="p-3 bg-neutral-900 border border-neutral-800 rounded-lg flex flex-col col-span-2">
                        <span className="text-neutral-500 mb-1">Render Time</span>
                        <span className="font-mono text-emerald-400">{renderTime < 60000 ? `${(renderTime/1000).toFixed(1)}s` : `${(renderTime/60000).toFixed(1)}m`}</span>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <div className="p-3 bg-neutral-900 border border-neutral-800 rounded-lg flex flex-col">
                      <span className="text-neutral-500 mb-1">Speed</span>
                      <span className="font-mono text-emerald-400">{stats.hogelsPerSecond.toLocaleString()} hz</span>
                    </div>
                    <div className="p-3 bg-neutral-900 border border-neutral-800 rounded-lg flex flex-col">
                      <span className="text-neutral-500 mb-1">Subpixels</span>
                      <span className="font-mono text-neutral-200 text-indigo-400">{billionsSubpixels}B</span>
                    </div>
                    {renderTime > 0 && (
                      <div className="p-3 bg-neutral-900 border border-neutral-800 rounded-lg flex flex-col col-span-2">
                        <span className="text-neutral-500 mb-1">Render Time</span>
                        <span className="font-mono text-amber-400">{renderTime}ms</span>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </section>

          {/* Light Transport / Panel Research Controls */}
          {nativeStatus.available && (
            <section className="space-y-4">
              <h2 className="text-sm font-medium text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Light Transport
              </h2>
              <div className="space-y-3 text-xs">
                {[
                  { label: 'Samples per pixel', value: spp, setter: setSpp, min: 1, max: 1024, step: 1, testid: 'slider-spp' },
                  { label: 'Hogel grid (N×N)', value: gridSize, setter: setGridSize, min: 16, max: 256, step: 16, testid: 'slider-grid-size' },
                  { label: 'Hemisphere resolution', value: hemiRes, setter: setHemiRes, min: 32, max: 512, step: 32, testid: 'slider-hemi-res' },
                ].map(({ label, value, setter, min, max, step, testid }) => (
                  <div key={label}>
                    <div className="flex justify-between text-neutral-400 mb-1">
                      <span>{label}</span>
                      <span className="font-mono text-amber-300">{value}</span>
                    </div>
                    <input
                      type="range"
                      min={min} max={max} step={step}
                      value={value}
                      onChange={e => setter(Number(e.target.value))}
                      className="w-full accent-amber-500"
                      data-testid={testid}
                    />
                  </div>
                ))}
                <p className="text-neutral-600 text-[10px]">
                  {gridSize}×{gridSize} hogels · {hemiRes}×{hemiRes} each · {(gridSize * gridSize * hemiRes * hemiRes / 1e6).toFixed(1)}M pixels total
                </p>
              </div>
            </section>
          )}

          {/* Phase-only Holography (Gerchberg-Saxton + Panel Model) */}
          {nativeStatus.available && (
            <section className="space-y-4">
              <h2 className="text-sm font-medium text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Holography (GS + Panel)
                {gsPhase === 'running' && (
                  <span className="text-xs text-emerald-400 animate-pulse ml-auto" data-testid="gs-progress-text">
                    {gsItersDone}/{gsIterations} iter
                  </span>
                )}
              </h2>
              <label className="flex items-center gap-2 text-xs text-neutral-300">
                <input
                  type="checkbox"
                  checked={gsEnabled}
                  onChange={e => setGsEnabled(e.target.checked)}
                  className="accent-emerald-500"
                  data-testid="gs-enabled"
                />
                <span>Enable phase-only hologram (Gerchberg-Saxton)</span>
              </label>
              {gsEnabled && (
                <div className="space-y-3 text-xs">
                  {[
                    { label: 'GS iterations', value: gsIterations, setter: setGsIterations, min: 0, max: 100, step: 1, testid: 'gs-iterations' },
                    { label: 'Phase bits (0 = continuous)', value: gsPhaseBits, setter: setGsPhaseBits, min: 0, max: 12, step: 1, testid: 'gs-phase-bits' },
                    { label: 'Phase noise σ (radians)', value: gsNoiseSigma, setter: setGsNoiseSigma, min: 0, max: 1.5, step: 0.01, fixed: 2, testid: 'gs-noise-sigma' },
                  ].map(({ label, value, setter, min, max, step, fixed, testid }) => (
                    <div key={label}>
                      <div className="flex justify-between text-neutral-400 mb-1">
                        <span>{label}</span>
                        <span className="font-mono text-emerald-300">{fixed != null ? (value as number).toFixed(fixed) : value}</span>
                      </div>
                      <input
                        type="range"
                        min={min} max={max} step={step}
                        value={value}
                        onChange={e => setter(Number(e.target.value))}
                        className="w-full accent-emerald-500"
                        data-testid={testid}
                      />
                    </div>
                  ))}
                  <label className="flex items-center gap-2 text-xs text-neutral-300">
                    <input
                      type="checkbox"
                      checked={gsLivePreview}
                      onChange={e => setGsLivePreview(e.target.checked)}
                      className="accent-emerald-500"
                      data-testid="gs-live-preview"
                    />
                    <span>Show live convergence preview</span>
                  </label>
                  {gsPhase === 'running' && (
                    <div className="w-full bg-neutral-800 h-1 rounded overflow-hidden">
                      <div
                        className="h-full bg-emerald-500 transition-all"
                        style={{ width: `${gsProgress * 100}%` }}
                        data-testid="gs-progress-bar"
                      />
                    </div>
                  )}
                  <p className="text-neutral-600 text-[10px]">
                    Runs Gerchberg-Saxton to synthesize a phase-only fringe, applies panel
                    quantization + physical noise, then forward-propagates (FFT) to get the
                    reconstructed intensity. Monochrome (Rec.709 luminance).
                  </p>
                </div>
              )}
            </section>
          )}

          <section className="space-y-4">
            <h2 className="text-sm font-medium text-neutral-400 uppercase tracking-wider flex items-center gap-2">
              Hogel Diagnostics
              {holoState.running && <span className="text-xs text-amber-400 animate-pulse">Computing hologram…</span>}
            </h2>
            {renderState === 'done' && (
              <p className="text-xs text-neutral-500">
                {hogelPreview
                  ? <>Hovering hogel <span className="font-mono text-emerald-400">({hogelPreview.hx}, {hogelPreview.hy})</span> · {hogelPreview.res}px</>
                  : <>Move cursor over the viewport to inspect a hogel.</>}
              </p>
            )}
            <div className="grid grid-cols-2 gap-3">
               <PreviewCanvas 
                 data={hogelPreview?.hemisphereRgba || holoState.hemisphere || stats.previews?.lightfield} 
                 width={hogelPreview?.res || holoState.hemiRes || stats.previews?.width || 128} 
                 height={hogelPreview?.res || holoState.hemiRes || stats.previews?.height || 128} 
                 label={hogelPreview ? `Hemisphere @ (${hogelPreview.hx},${hogelPreview.hy})` : (holoState.hemisphere ? "Hemisphere (180° Fisheye)" : "Ray-Traced Lightfield")} 
               />
               <PreviewCanvas 
                 data={hogelPreview?.phaseRgba || holoState.fringe || stats.previews?.fringe} 
                 width={hogelPreview?.res || holoState.hemiRes || stats.previews?.width || 128} 
                 height={hogelPreview?.res || holoState.hemiRes || stats.previews?.height || 128} 
                 label={hogelPreview?.phaseRgba ? `Phase φ ∈ [−π,π]` : (gsEnabled ? "Phase (enable GS to view)" : "Enable GS to compute phase")} 
               />
            </div>
            {holoState.gridW && (
              <div className="text-xs font-mono text-neutral-500 space-y-0.5">
                <div>{holoState.gridW}×{holoState.gridH} hogels ({holoState.gridW! * holoState.gridH!} total) · {holoState.hemiRes}px hemisphere</div>
                <div>{((holoState.gridW! * holoState.gridH! * holoState.hemiRes! * holoState.hemiRes!) / 1e6).toFixed(1)}M pixels · λ=532nm</div>
                <div>Reconstruction: {holoState.fullWidth}×{holoState.fullHeight}px</div>
                {holoState.time != null && <div className="text-emerald-400">Total time: {holoState.time}ms</div>}
              </div>
            )}
          </section>
          
          <section className="space-y-4">
            <h2 className="text-sm font-medium text-neutral-400 uppercase tracking-wider flex items-center gap-2">
              <Layers className="w-4 h-4" />
              Cornell Box Scene
            </h2>
            <div className="h-64 w-full bg-neutral-950 border border-neutral-800 rounded-xl overflow-hidden relative">
               <DiagnosticsVisualizer
                 hoveredHogel={hogelPreview
                   ? { hx: hogelPreview.hx, hy: hogelPreview.hy, gridW: holoState.gridW ?? gridSize, gridH: holoState.gridH ?? gridSize }
                   : undefined}
                 parallaxOrbit={parallaxState !== 'idle' ? {
                   centerX: PARALLAX_EYE_CENTER.x,
                   centerY: PARALLAX_EYE_CENTER.y,
                   centerZ: PARALLAX_EYE_CENTER.z,
                   radius: parallaxOrbitRadius,
                   axis: parallaxOrbitAxis,
                   frameIdx: parallaxFrameIdx,
                   totalFrames: PARALLAX_FRAMES,
                 } : undefined}
               />
               <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 backdrop-blur rounded text-[10px] text-neutral-400 font-mono">
                 Cornell Box · {gridSize}×{gridSize} hogels (drag to rotate)
               </div>
            </div>
          </section>

          {/* Parallax test */}
          {nativeStatus.available && (
            <section className="space-y-4">
              <h2 className="text-sm font-medium text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <Layers className="w-4 h-4" />
                Parallax Test
              </h2>
              <div className="space-y-3 text-xs">
                <div>
                  <div className="flex justify-between text-neutral-400 mb-1">
                    <span>Orbit radius (mm)</span>
                    <span className="font-mono text-amber-300">{parallaxOrbitRadius}</span>
                  </div>
                  <input type="range" min={50} max={600} step={50} value={parallaxOrbitRadius}
                    onChange={e => setParallaxOrbitRadius(Number(e.target.value))}
                    className="w-full accent-amber-500" />
                </div>
                <div className="flex gap-2">
                  {(['horizontal', 'vertical'] as const).map(ax => (
                    <button key={ax}
                      onClick={() => setParallaxOrbitAxis(ax)}
                      className={`flex-1 py-1.5 rounded text-xs font-medium border transition-colors ${parallaxOrbitAxis === ax ? 'bg-amber-600 border-amber-500 text-white' : 'bg-neutral-900 border-neutral-700 text-neutral-400 hover:border-neutral-600'}`}>
                      {ax}
                    </button>
                  ))}
                </div>
                <button
                  onClick={generateParallax}
                  disabled={renderState !== 'done' || parallaxState === 'generating'}
                  data-testid="parallax-generate-btn"
                  data-parallax-state={parallaxState}
                  className="w-full py-2 rounded-lg font-medium text-xs bg-amber-600 hover:bg-amber-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {parallaxState === 'generating' ? `Generating… ${(parallaxProgress * 100).toFixed(0)}%` : `Generate ${PARALLAX_FRAMES} frames`}
                </button>
                {parallaxState === 'generating' && (
                  <div className="w-full bg-neutral-800 h-1 rounded overflow-hidden">
                    <div data-testid="parallax-progress-bar" className="h-full bg-amber-500 transition-all" style={{ width: `${parallaxProgress * 100}%` }} />
                  </div>
                )}
                {parallaxState === 'done' && parallaxFrames.length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <button onClick={() => setParallaxPlaying(p => !p)}
                        data-testid="parallax-play-btn"
                        className="px-3 py-1.5 rounded bg-neutral-800 border border-neutral-700 text-neutral-300 hover:bg-neutral-700 transition-colors text-xs">
                        {parallaxPlaying ? '⏸ Pause' : '▶ Play'}
                      </button>
                      <span data-testid="parallax-frame-counter" className="font-mono text-neutral-500">{parallaxFrameIdx + 1} / {parallaxFrames.length}</span>
                    </div>
                    <input type="range" min={0} max={parallaxFrames.length - 1} step={1}
                      value={parallaxFrameIdx}
                      onChange={e => { setParallaxPlaying(false); setParallaxFrameIdx(Number(e.target.value)); }}
                      data-testid="parallax-scrubber"
                      className="w-full accent-amber-500" />
                    <PreviewCanvas
                      data={parallaxFrames[parallaxFrameIdx]}
                      width={holoState.fullWidth ?? 1024}
                      height={holoState.fullHeight ?? 1024}
                      label={`Frame ${parallaxFrameIdx + 1} · ${parallaxOrbitAxis} orbit`}
                    />
                  </div>
                )}
              </div>
            </section>
          )}

        </aside>

        {/* Viewport View */}
        <section className="flex-1 bg-black flex items-center justify-center overflow-hidden relative">
          <div className="w-full h-full flex items-center justify-center dashboard-grid p-6">
            <div
              className="relative rounded-lg border border-neutral-800 bg-neutral-900 shadow-2xl overflow-hidden"
              style={{ height: '100%', aspectRatio: '1/1', maxWidth: '100%' }}
            >
              {/* Virtual Camera Sensor Accumulation View */}
              <canvas 
                ref={canvasRef}
                data-testid="main-viewport"
                width={1024} 
                height={1024} 
                className="w-full h-full block" 
                style={{ imageRendering: 'pixelated', cursor: renderState === 'done' ? 'crosshair' : 'default' }}
                onMouseMove={handleReconMouseMove}
                onMouseLeave={() => { hogelLastKeyRef.current = ''; }}
              />
              
              {!running && renderState === 'idle' && (
                <div className="absolute inset-0 bg-neutral-950/80 backdrop-blur-sm flex flex-col items-center justify-center text-center z-10 space-y-4">
                  <Play className="w-12 h-12 text-indigo-500 opacity-50" />
                  <h3 className="text-xl font-medium text-white">
                    {nativeStatus.available ? 'Click "Compute Hologram" to start' : 'Press Start'}
                  </h3>
                  <p className="text-neutral-400 max-w-sm text-sm">
                    {nativeStatus.available
                      ? 'Computes per-hogel hemispheres, interference fringes, and lightfield reconstruction.'
                      : 'Because this browser process physically cannot allocate 120GB of RAM for the full array, we use an expanding-window reconstruction technique.'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
      <style>{`
        .dashboard-grid {
          background-image: 
            linear-gradient(to right, rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255,255,255,0.02) 1px, transparent 1px);
          background-size: 40px 40px;
        }
        .animate-spin-slow {
          animation: spin 4s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

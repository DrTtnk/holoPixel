/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Aperture, Layers, Play, Zap } from "lucide-react";
import { useEffect, useRef, useState, useCallback } from "react";

import { DiagnosticsVisualizer } from "./components/DiagnosticsVisualizer";

// ── Constants ──────────────────────────────────────────────

const PARALLAX_FRAMES = 24;
const PARALLAX_EYE_CENTER = { x: 278, y: 273, z: -800 } as const;
const OUTPUT_SIZE = 1024;
const MAX_BOUNCES = 0;
const AMBIENT = 0.08;

// ── Shared Types ───────────────────────────────────────────

interface NativeStatus {
  available: boolean;
  ping?: string;
  gpu?: { name: string; totalMemoryMb: number; computeCapability: string };
  cudaOk?: boolean;
  scene?: { triangleCount: number; materialCount: number; sceneName: string };
  traceHit?: boolean;
}

interface HoloState {
  running: boolean;
  gridW?: number;
  gridH?: number;
  hemiRes?: number;
  fullWidth?: number;
  fullHeight?: number;
  time?: number;
}

interface HogelPreviewData {
  hx: number;
  hy: number;
  res: number;
  hemisphereRgba: Uint8ClampedArray;
  phaseRgba: Uint8ClampedArray | null;
}

type RenderState = 'idle' | 'rendering' | 'done';
type ParallaxState = 'idle' | 'generating' | 'done';
type GsPhase = 'idle' | 'running';

// ── Utility ────────────────────────────────────────────────

function decodeBase64(b64: string): Uint8ClampedArray {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Uint8ClampedArray(arr.buffer);
}

function formatTime(ms: number): string {
  return ms < 60000 ? `${(ms / 1000).toFixed(1)}s` : `${(ms / 60000).toFixed(1)}m`;
}

// ── Preview Canvas ─────────────────────────────────────────

function PreviewCanvas({ data, width, height, label }: {
  data?: Uint8ClampedArray;
  width: number;
  height: number;
  label: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (data && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.putImageData(new ImageData(data as unknown as Uint8ClampedArray<ArrayBuffer>, width, height), 0, 0);
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

// ── Slider Row ─────────────────────────────────────────────

function SliderRow({ label, value, setter, min, max, step, testid, accentClass = 'accent-amber-500', valueClass = 'text-amber-300', fixed }: {
  label: string;
  value: number;
  setter: (v: number) => void;
  min: number;
  max: number;
  step: number;
  testid: string;
  accentClass?: string;
  valueClass?: string;
  fixed?: number;
}) {
  return (
    <div>
      <div className="flex justify-between text-neutral-400 mb-1">
        <span>{label}</span>
        <span className={`font-mono ${valueClass}`}>{fixed != null ? value.toFixed(fixed) : value}</span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => setter(Number(e.target.value))}
        className={`w-full ${accentClass}`}
        data-testid={testid}
      />
    </div>
  );
}

// ── Header ─────────────────────────────────────────────────

function Header({ nativeStatus, gridSize, hemiRes }: {
  nativeStatus: NativeStatus;
  gridSize: number;
  hemiRes: number;
}) {
  return (
    <header className="border-b border-neutral-800 bg-neutral-900 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-indigo-500/10 rounded-lg">
          <Aperture className="w-6 h-6 text-indigo-400" />
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
        {nativeStatus.available && <div className="w-px h-8 bg-neutral-800" />}
        <div className="flex flex-col items-end">
          <span className="text-xs text-neutral-500 font-mono tracking-wider uppercase">Output</span>
          <span className="text-sm font-medium">{OUTPUT_SIZE} × {OUTPUT_SIZE}</span>
        </div>
        <div className="w-px h-8 bg-neutral-800" />
        <div className="flex flex-col items-end">
          <span className="text-xs text-neutral-500 font-mono tracking-wider uppercase">Hogel Grid</span>
          <span className="text-sm font-medium text-emerald-400">{gridSize}×{gridSize} @ {hemiRes}px</span>
        </div>
      </div>
    </header>
  );
}

// ── Compute Controls ───────────────────────────────────────

function ComputeControls({ renderState, renderProgress, renderEta, renderTime, onCompute }: {
  renderState: RenderState;
  renderProgress: number;
  renderEta: number | null;
  renderTime: number;
  onCompute: () => void;
}) {
  const progressPct = (renderProgress * 100).toFixed(1);

  return (
    <section className="space-y-4">
      <h2 className="text-sm font-medium text-neutral-400 uppercase tracking-wider flex items-center gap-2">
        <Zap className="w-4 h-4" />
        Compute State
      </h2>

      <button
        onClick={onCompute}
        disabled={renderState === 'rendering'}
        data-testid="compute-btn"
        data-render-state={renderState}
        className={`w-full flex items-center justify-center gap-2 py-3 rounded-lg font-medium transition-colors ${
          renderState === 'rendering'
            ? 'bg-neutral-800 hover:bg-neutral-700 text-white border border-neutral-700'
            : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-900/20'
        }`}
      >
        {renderState === 'rendering' ? (
          <><Aperture className="w-4 h-4 animate-spin" /> Rendering...</>
        ) : (
          <><Play className="w-4 h-4" /> Compute Hologram</>
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
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          {renderEta !== null && renderEta > 0 && (
            <div className="p-3 bg-neutral-900 border border-neutral-800 rounded-lg flex flex-col col-span-2">
              <span className="text-neutral-500 mb-1">ETA</span>
              <span className="font-mono text-amber-400">{formatTime(renderEta)}</span>
            </div>
          )}
          {renderTime > 0 && (
            <div className="p-3 bg-neutral-900 border border-neutral-800 rounded-lg flex flex-col col-span-2">
              <span className="text-neutral-500 mb-1">Render Time</span>
              <span className="font-mono text-emerald-400">{formatTime(renderTime)}</span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

// ── Light Transport Controls ───────────────────────────────

function LightTransportControls({ spp, setSpp, gridSize, setGridSize, hemiRes, setHemiRes }: {
  spp: number;
  setSpp: (v: number) => void;
  gridSize: number;
  setGridSize: (v: number) => void;
  hemiRes: number;
  setHemiRes: (v: number) => void;
}) {
  return (
    <section className="space-y-4">
      <h2 className="text-sm font-medium text-neutral-400 uppercase tracking-wider flex items-center gap-2">
        <Zap className="w-4 h-4" />
        Light Transport
      </h2>
      <div className="space-y-3 text-xs">
        <SliderRow label="Samples per pixel" value={spp} setter={setSpp} min={1} max={16} step={1} testid="slider-spp" />
        <SliderRow label="Hogel grid (N×N)" value={gridSize} setter={setGridSize} min={16} max={1024} step={16} testid="slider-grid-size" />
        <SliderRow label="Hemisphere resolution" value={hemiRes} setter={setHemiRes} min={32} max={512} step={32} testid="slider-hemi-res" />
        <p className="text-neutral-600 text-[10px]">
          {gridSize}×{gridSize} hogels · {hemiRes}×{hemiRes} each · {(gridSize * gridSize * hemiRes * hemiRes / 1e6).toFixed(1)}M pixels total
        </p>
      </div>
    </section>
  );
}

// ── GS Controls ────────────────────────────────────────────

function GsControls({ gsEnabled, setGsEnabled, gsIterations, setGsIterations, gsPhaseBits, setGsPhaseBits, gsNoiseSigma, setGsNoiseSigma, gsLivePreview, setGsLivePreview, gsPhase, gsItersDone, gsProgress }: {
  gsEnabled: boolean;
  setGsEnabled: (v: boolean) => void;
  gsIterations: number;
  setGsIterations: (v: number) => void;
  gsPhaseBits: number;
  setGsPhaseBits: (v: number) => void;
  gsNoiseSigma: number;
  setGsNoiseSigma: (v: number) => void;
  gsLivePreview: boolean;
  setGsLivePreview: (v: boolean) => void;
  gsPhase: GsPhase;
  gsItersDone: number;
  gsProgress: number;
}) {
  return (
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
          <SliderRow label="GS iterations" value={gsIterations} setter={setGsIterations} min={0} max={100} step={1} testid="gs-iterations" accentClass="accent-emerald-500" valueClass="text-emerald-300" />
          <SliderRow label="Phase bits (0 = continuous)" value={gsPhaseBits} setter={setGsPhaseBits} min={0} max={12} step={1} testid="gs-phase-bits" accentClass="accent-emerald-500" valueClass="text-emerald-300" />
          <SliderRow label="Phase noise σ (radians)" value={gsNoiseSigma} setter={setGsNoiseSigma} min={0} max={1.5} step={0.01} testid="gs-noise-sigma" accentClass="accent-emerald-500" valueClass="text-emerald-300" fixed={2} />
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
  );
}

// ── Hogel Diagnostics ──────────────────────────────────────

function HogelDiagnostics({ holoState, hogelPreview, gsEnabled, renderState }: {
  holoState: HoloState;
  hogelPreview: HogelPreviewData | null;
  gsEnabled: boolean;
  renderState: RenderState;
}) {
  return (
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
          data={hogelPreview?.hemisphereRgba}
          width={hogelPreview?.res || holoState.hemiRes || 64}
          height={hogelPreview?.res || holoState.hemiRes || 64}
          label={hogelPreview ? `Hemisphere @ (${hogelPreview.hx},${hogelPreview.hy})` : "Hemisphere (180° Fisheye)"}
        />
        <PreviewCanvas
          data={hogelPreview?.phaseRgba ?? undefined}
          width={hogelPreview?.res || holoState.hemiRes || 64}
          height={hogelPreview?.res || holoState.hemiRes || 64}
          label={hogelPreview?.phaseRgba ? "Phase φ ∈ [−π,π]" : (gsEnabled ? "Phase (run GS to view)" : "Enable GS for phase")}
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
  );
}

// ── Scene Preview ──────────────────────────────────────────

function ScenePreview({ hogelPreview, holoState, gridSize, parallaxState, parallaxOrbitRadius, parallaxOrbitAxis, parallaxFrameIdx }: {
  hogelPreview: HogelPreviewData | null;
  holoState: HoloState;
  gridSize: number;
  parallaxState: ParallaxState;
  parallaxOrbitRadius: number;
  parallaxOrbitAxis: 'horizontal' | 'vertical';
  parallaxFrameIdx: number;
}) {
  return (
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
  );
}

// ── Parallax Controls ──────────────────────────────────────

function ParallaxControls({
  renderState, parallaxState, parallaxProgress, parallaxFrames,
  parallaxFrameIdx, setParallaxFrameIdx, parallaxPlaying, setParallaxPlaying,
  parallaxOrbitRadius, setParallaxOrbitRadius, parallaxOrbitAxis, setParallaxOrbitAxis,
  holoState, onGenerate, onScrub,
}: {
  renderState: RenderState;
  parallaxState: ParallaxState;
  parallaxProgress: number;
  parallaxFrames: Uint8ClampedArray[];
  parallaxFrameIdx: number;
  setParallaxFrameIdx: (v: number) => void;
  parallaxPlaying: boolean;
  setParallaxPlaying: (v: boolean | ((p: boolean) => boolean)) => void;
  parallaxOrbitRadius: number;
  setParallaxOrbitRadius: (v: number) => void;
  parallaxOrbitAxis: 'horizontal' | 'vertical';
  setParallaxOrbitAxis: (v: 'horizontal' | 'vertical') => void;
  holoState: HoloState;
  onGenerate: () => void;
  onScrub: (idx: number) => void;
}) {
  // Local slider state for scrubbing — avoids full App re-render on every tick
  const [localIdx, setLocalIdx] = useState(parallaxFrameIdx);
  useEffect(() => { setLocalIdx(parallaxFrameIdx); }, [parallaxFrameIdx]);

  return (
    <section className="space-y-4">
      <h2 className="text-sm font-medium text-neutral-400 uppercase tracking-wider flex items-center gap-2">
        <Layers className="w-4 h-4" />
        Parallax Test
      </h2>
      <div className="space-y-3 text-xs">
        <SliderRow
          label="Orbit radius (mm)" value={parallaxOrbitRadius} setter={setParallaxOrbitRadius}
          min={50} max={600} step={50} testid="slider-orbit-radius"
        />
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
          onClick={onGenerate}
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
              <button onClick={() => setParallaxPlaying((p: boolean) => !p)}
                data-testid="parallax-play-btn"
                className="px-3 py-1.5 rounded bg-neutral-800 border border-neutral-700 text-neutral-300 hover:bg-neutral-700 transition-colors text-xs">
                {parallaxPlaying ? '⏸ Pause' : '▶ Play'}
              </button>
              <span data-testid="parallax-frame-counter" className="font-mono text-neutral-500">{localIdx + 1} / {parallaxFrames.length}</span>
            </div>
            <input type="range" min={0} max={parallaxFrames.length - 1} step={1}
              value={localIdx}
              onChange={e => {
                const idx = Number(e.target.value);
                setLocalIdx(idx);
                onScrub(idx);
                setParallaxPlaying(false);
              }}
              onPointerUp={() => setParallaxFrameIdx(localIdx)}
              data-testid="parallax-scrubber"
              className="w-full accent-amber-500" />
            <PreviewCanvas
              data={parallaxFrames[localIdx]}
              width={holoState.fullWidth ?? OUTPUT_SIZE}
              height={holoState.fullHeight ?? OUTPUT_SIZE}
              label={`Frame ${localIdx + 1} · ${parallaxOrbitAxis} orbit`}
            />
          </div>
        )}
      </div>
    </section>
  );
}

// ── Main App ───────────────────────────────────────────────

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Native backend state
  const [nativeStatus, setNativeStatus] = useState<NativeStatus>({ available: false });
  const [renderState, setRenderState] = useState<RenderState>('idle');
  const [renderTime, setRenderTime] = useState(0);
  const [renderProgress, setRenderProgress] = useState(0);
  const [renderEta, setRenderEta] = useState<number | null>(null);
  const renderStartRef = useRef(0);

  // Light transport params
  const [spp, setSpp] = useState(8);
  const [gridSize, setGridSize] = useState(128);
  const [hemiRes, setHemiRes] = useState(256);

  // Phase-only holography (Gerchberg-Saxton + panel model)
  const [gsEnabled, setGsEnabled] = useState(true);
  const [gsIterations, setGsIterations] = useState(10);
  const [gsPhaseBits, setGsPhaseBits] = useState(0);
  const [gsNoiseSigma, setGsNoiseSigma] = useState(0.0);
  const [gsLivePreview, setGsLivePreview] = useState(true);
  const [gsProgress, setGsProgress] = useState(0);
  const [gsItersDone, setGsItersDone] = useState(0);
  const [gsPhase, setGsPhase] = useState<GsPhase>('idle');

  // Parallax
  const [parallaxState, setParallaxState] = useState<ParallaxState>('idle');
  const [parallaxProgress, setParallaxProgress] = useState(0);
  const [parallaxFrames, setParallaxFrames] = useState<Uint8ClampedArray[]>([]);
  const [parallaxFrameIdx, setParallaxFrameIdx] = useState(0);
  const [parallaxPlaying, setParallaxPlaying] = useState(false);
  const [parallaxOrbitRadius, setParallaxOrbitRadius] = useState(200);
  const [parallaxOrbitAxis, setParallaxOrbitAxis] = useState<'horizontal' | 'vertical'>('horizontal');
  const parallaxRafRef = useRef<number | null>(null);
  const parallaxCanvasesRef = useRef<HTMLCanvasElement[]>([]);
  const parallaxFrameIdxRef = useRef(0);

  // Hogel preview on hover
  const [hogelPreview, setHogelPreview] = useState<HogelPreviewData | null>(null);
  const hogelFetchBusyRef = useRef(false);
  const hogelLastKeyRef = useRef('');

  const [holoState, setHoloState] = useState<HoloState>({ running: false });

  // ── Probe native backend ─────────────────────────────────

  useEffect(() => {
    if (!window.holosim) return;
    (async () => {
      const pingResult = await window.holosim!.ping();
      if (!pingResult.ok) return;

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
    })();
  }, []);

  // ── Draw reconstruction to canvas ────────────────────────

  const drawRecon = useCallback((reconData: Uint8ClampedArray, fw: number, fh: number) => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    const offscreen = document.createElement('canvas');
    offscreen.width = fw;
    offscreen.height = fh;
    offscreen.getContext('2d')!.putImageData(new ImageData(reconData as unknown as Uint8ClampedArray<ArrayBuffer>, fw, fh), 0, 0);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offscreen, 0, 0, canvasRef.current.width, canvasRef.current.height);
  }, []);

  // ── Hogel preview fetch (throttled) ──────────────────────

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

  // Pre-render parallax frames to offscreen canvases
  // (must be declared BEFORE playback effect so React commits canvasesRef first)
  useEffect(() => {
    if (parallaxFrames.length === 0) { parallaxCanvasesRef.current = []; parallaxFrameIdxRef.current = 0; return; }
    const out = holoState.fullWidth ?? OUTPUT_SIZE;
    parallaxCanvasesRef.current = parallaxFrames.map(frame => {
      const c = document.createElement('canvas');
      c.width = out; c.height = out;
      c.getContext('2d')!.putImageData(new ImageData(frame as unknown as Uint8ClampedArray<ArrayBuffer>, out, out), 0, 0);
      return c;
    });
  }, [parallaxFrames, holoState.fullWidth]);

  // ── Parallax playback (rAF loop, bypasses React) ─────────

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
      setParallaxFrameIdx(parallaxFrameIdxRef.current);
    };
  }, [parallaxPlaying, parallaxFrames]);

  // Draw current parallax frame (manual scrub only)
  useEffect(() => {
    if (parallaxPlaying) return;
    const prebuilt = parallaxCanvasesRef.current[parallaxFrameIdx];
    if (!prebuilt || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d')!;
    ctx.drawImage(prebuilt, 0, 0, canvasRef.current.width, canvasRef.current.height);
  }, [parallaxFrameIdx, parallaxPlaying]);

  // Direct scrub handler — draws to main canvas without triggering App re-render
  const handleParallaxScrub = useCallback((idx: number) => {
    parallaxFrameIdxRef.current = idx;
    const prebuilt = parallaxCanvasesRef.current[idx];
    if (prebuilt && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')!;
      ctx.drawImage(prebuilt, 0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, []);

  // ── Generate parallax sequence ───────────────────────────

  const generateParallax = useCallback(async () => {
    if (!window.holosim || renderState !== 'done') return;
    setParallaxState('generating');
    setParallaxProgress(0);
    setParallaxFrames([]);
    setParallaxFrameIdx(0);
    setParallaxPlaying(false);

    const frames: Uint8ClampedArray[] = [];
    const { x: ex, y: ey, z: ez } = PARALLAX_EYE_CENTER;
    const r = parallaxOrbitRadius;

    for (let i = 0; i < PARALLAX_FRAMES; i++) {
      const t = (i / PARALLAX_FRAMES) * Math.PI * 2;
      const eyeX = parallaxOrbitAxis === 'horizontal' ? ex + Math.cos(t) * r : ex;
      const eyeY = parallaxOrbitAxis === 'vertical'   ? ey + Math.sin(t) * r : ey;
      const eyeZ = parallaxOrbitAxis === 'horizontal' ? ez + Math.sin(t) * r : ez + Math.cos(t) * r;

      const r2 = await window.holosim!.gpuSessionReconstruct(eyeX, eyeY, eyeZ);
      if (!r2.ok || !r2.data) break;
      frames.push(new Uint8ClampedArray(r2.data.reconBuffer));
      setParallaxProgress((i + 1) / PARALLAX_FRAMES);
      // Yield to event loop so UI stays responsive
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    setParallaxFrames(frames);
    setParallaxState('done');
    setParallaxPlaying(true);
  }, [renderState, parallaxOrbitRadius, parallaxOrbitAxis]);

  // ── Mouse handler: pick hogel under cursor ───────────────

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

  // ── Main compute ─────────────────────────────────────────

  const toggleCompute = useCallback(() => {
    if (!nativeStatus.available || !window.holosim || renderState === 'rendering') return;

    setRenderState('rendering');
    setHoloState(s => ({ ...s, running: true }));
    setRenderProgress(0);
    setRenderEta(null);
    renderStartRef.current = performance.now();

    // Maximize rows per IPC call (Rust sub-batches internally to fit GPU memory).
    // MAX_BATCH_HOGELS=4096 on the Rust side; send as many rows as that allows.
    const BATCH_ROWS = Math.max(1, Math.min(gridSize, Math.floor(4096 / gridSize)));
    const GRID = gridSize;
    const SPP = spp;
    const HEMI = hemiRes;
    const EX = PARALLAX_EYE_CENTER.x, EY = PARALLAX_EYE_CENTER.y, EZ = PARALLAX_EYE_CENTER.z;

    (async () => {
      try { await window.holosim!.gpuSessionClose(); } catch { /* idempotent */ }
      setHogelPreview(null);
      const beginResult = await window.holosim!.gpuSessionBegin(GRID, GRID, HEMI, SPP, OUTPUT_SIZE, OUTPUT_SIZE, MAX_BOUNCES, AMBIENT);
      if (!beginResult.ok || !beginResult.data) {
        setRenderState('idle');
        setHoloState(s => ({ ...s, running: false }));
        return;
      }
      const totalRows = beginResult.data.totalRows;
      const isStreaming = beginResult.data.streaming;

      if (isStreaming) {
        // ── Streaming mode: fused render+GS+scatter per batch ──
        const STREAM_BATCH_ROWS = Math.max(1, Math.min(GRID, Math.floor(4096 / GRID)));
        const seed = BigInt(Math.floor(Math.random() * 0xffff_ffff));
        const iters = gsEnabled ? gsIterations : 0;
        const bits = gsEnabled ? gsPhaseBits : 0;
        const sigma = gsEnabled ? gsNoiseSigma : 0;

        for (let row = 0; row < totalRows; row += STREAM_BATCH_ROWS) {
          const n = Math.min(STREAM_BATCH_ROWS, totalRows - row);
          const r = await window.holosim!.gpuSessionStreamBatch(row, n, iters, bits, sigma, seed);
          if (!r.ok || !r.data) break;
          const rowsDone = r.data.rowsDone;
          const frac = rowsDone / totalRows;
          setRenderProgress(frac);
          const elapsed = performance.now() - renderStartRef.current;
          if (rowsDone > 0) {
            setRenderEta(Math.round((totalRows - rowsDone) * (elapsed / rowsDone)));
          }
        }
        // PCA-compress the raw intensity cache for efficient parallax
        await window.holosim!.gpuSessionCompressIntensity();
      } else {
        // ── Standard mode: separate render → GS → reconstruct ──
        for (let row = 0; row < totalRows; row += BATCH_ROWS) {
          const batchResult = await window.holosim!.gpuSessionRenderRows(row, Math.min(BATCH_ROWS, totalRows - row));
          if (!batchResult.ok || !batchResult.data) break;
          const rowsDone = batchResult.data.rowsDone;
          const frac = rowsDone / totalRows;
          setRenderProgress(frac);
          const elapsed = performance.now() - renderStartRef.current;
          if (rowsDone > 0) {
            setRenderEta(Math.round((totalRows - rowsDone) * (elapsed / rowsDone)));
          }
        }

        if (gsEnabled) {
          setGsPhase('running');
          setGsProgress(0);
          setGsItersDone(0);
          const seed = BigInt(Math.floor(Math.random() * 0xffff_ffff));
          const setupR = await window.holosim!.gpuSessionGsSetup(seed);
          if (setupR.ok && gsIterations > 0) {
            if (!gsLivePreview) {
              const r = await window.holosim!.gpuSessionGsIterate(gsIterations);
              if (r.ok && r.data) {
                setGsItersDone(r.data.itersDone);
                setGsProgress(1);
              }
            } else {
              const MAX_PREVIEWS = 5;
              const PREVIEW_INTERVAL = Math.max(2, Math.ceil(gsIterations / MAX_PREVIEWS));
              for (let done = 0; done < gsIterations; ) {
                const chunk = Math.min(PREVIEW_INTERVAL, gsIterations - done);
                const r = await window.holosim!.gpuSessionGsIterateAndReconstruct(chunk, EX, EY, EZ);
                if (!r.ok || !r.data) break;
                done = r.data.itersDone;
                setGsItersDone(done);
                setGsProgress(done / gsIterations);
                if (done < gsIterations) {
                  drawRecon(new Uint8ClampedArray(r.data.reconBuffer), OUTPUT_SIZE, OUTPUT_SIZE);
                }
              }
            }
          }
          const finSeed = BigInt(Math.floor(Math.random() * 0xffff_ffff));
          await window.holosim!.gpuSessionGsFinalize(gsPhaseBits, gsNoiseSigma, finSeed);
          setGsProgress(1);
          setGsPhase('idle');
        }
      }

      const finishResult = await window.holosim!.gpuSessionReconstruct(EX, EY, EZ);
      if (finishResult.ok && finishResult.data) {
        drawRecon(new Uint8ClampedArray(finishResult.data.reconBuffer), OUTPUT_SIZE, OUTPUT_SIZE);
        const elapsed = Math.round(performance.now() - renderStartRef.current);
        setHoloState({
          running: false,
          gridW: GRID,
          gridH: GRID,
          hemiRes: HEMI,
          fullWidth: OUTPUT_SIZE,
          fullHeight: OUTPUT_SIZE,
          time: elapsed,
        });
        setRenderTime(elapsed);
        setRenderProgress(1);
        setRenderEta(0);
        setRenderState('done');
      } else {
        setRenderState('idle');
        setHoloState(s => ({ ...s, running: false }));
      }
    })();
  }, [nativeStatus.available, renderState, spp, gridSize, hemiRes, drawRecon, gsEnabled, gsIterations, gsPhaseBits, gsNoiseSigma, gsLivePreview]);

  // ── Render ───────────────────────────────────────────────

  return (
    <div className="h-screen overflow-hidden bg-neutral-950 text-neutral-200 font-sans flex flex-col">
      <Header nativeStatus={nativeStatus} gridSize={gridSize} hemiRes={hemiRes} />

      <main className="flex-1 flex overflow-hidden min-h-0">
        {/* Left Sidebar */}
        <aside className="w-80 border-r border-neutral-800 bg-neutral-900/50 p-6 flex flex-col gap-6 overflow-y-auto">
          <ComputeControls
            renderState={renderState}
            renderProgress={renderProgress}
            renderEta={renderEta}
            renderTime={renderTime}
            onCompute={toggleCompute}
          />

          {nativeStatus.available && (
            <LightTransportControls
              spp={spp} setSpp={setSpp}
              gridSize={gridSize} setGridSize={setGridSize}
              hemiRes={hemiRes} setHemiRes={setHemiRes}
            />
          )}

          {nativeStatus.available && (
            <GsControls
              gsEnabled={gsEnabled} setGsEnabled={setGsEnabled}
              gsIterations={gsIterations} setGsIterations={setGsIterations}
              gsPhaseBits={gsPhaseBits} setGsPhaseBits={setGsPhaseBits}
              gsNoiseSigma={gsNoiseSigma} setGsNoiseSigma={setGsNoiseSigma}
              gsLivePreview={gsLivePreview} setGsLivePreview={setGsLivePreview}
              gsPhase={gsPhase} gsItersDone={gsItersDone} gsProgress={gsProgress}
            />
          )}

          <HogelDiagnostics
            holoState={holoState}
            hogelPreview={hogelPreview}
            gsEnabled={gsEnabled}
            renderState={renderState}
          />

          <ScenePreview
            hogelPreview={hogelPreview}
            holoState={holoState}
            gridSize={gridSize}
            parallaxState={parallaxState}
            parallaxOrbitRadius={parallaxOrbitRadius}
            parallaxOrbitAxis={parallaxOrbitAxis}
            parallaxFrameIdx={parallaxFrameIdx}
          />

          {nativeStatus.available && (
            <ParallaxControls
              renderState={renderState}
              parallaxState={parallaxState}
              parallaxProgress={parallaxProgress}
              parallaxFrames={parallaxFrames}
              parallaxFrameIdx={parallaxFrameIdx}
              setParallaxFrameIdx={setParallaxFrameIdx}
              parallaxPlaying={parallaxPlaying}
              setParallaxPlaying={setParallaxPlaying}
              parallaxOrbitRadius={parallaxOrbitRadius}
              setParallaxOrbitRadius={setParallaxOrbitRadius}
              parallaxOrbitAxis={parallaxOrbitAxis}
              setParallaxOrbitAxis={setParallaxOrbitAxis}
              holoState={holoState}
              onGenerate={generateParallax}
              onScrub={handleParallaxScrub}
            />
          )}
        </aside>

        {/* Viewport */}
        <section className="flex-1 bg-black flex items-center justify-center overflow-hidden relative">
          <div className="w-full h-full flex items-center justify-center dashboard-grid p-6">
            <div
              className="relative rounded-lg border border-neutral-800 bg-neutral-900 shadow-2xl overflow-hidden"
              style={{ height: '100%', aspectRatio: '1/1', maxWidth: '100%' }}
            >
              <canvas
                ref={canvasRef}
                data-testid="main-viewport"
                width={OUTPUT_SIZE}
                height={OUTPUT_SIZE}
                className="w-full h-full block"
                style={{ imageRendering: 'pixelated', cursor: renderState === 'done' ? 'crosshair' : 'default' }}
                onMouseMove={handleReconMouseMove}
                onMouseLeave={() => { hogelLastKeyRef.current = ''; }}
              />

              {renderState === 'idle' && (
                <div className="absolute inset-0 bg-neutral-950/80 backdrop-blur-sm flex flex-col items-center justify-center text-center z-10 space-y-4">
                  <Play className="w-12 h-12 text-indigo-500 opacity-50" />
                  <h3 className="text-xl font-medium text-white">
                    {nativeStatus.available ? 'Click "Compute Hologram" to start' : 'Native backend not available'}
                  </h3>
                  <p className="text-neutral-400 max-w-sm text-sm">
                    Computes per-hogel hemispheres, interference fringes, and lightfield reconstruction.
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
      `}</style>
    </div>
  );
}

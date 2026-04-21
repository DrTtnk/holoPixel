import { Canvas } from '@react-three/fiber';
import { OrbitControls, Box, Plane } from '@react-three/drei';
import * as THREE from 'three';

// Cornell Box real-world dimensions (mm / 100 → Three.js units)
const W = 5.528; // width (X: 0..W)
const H = 5.488; // height (Y: 0..H)
const D = 5.592; // depth (Z: 0..D)
const cx = W / 2, cy = H / 2, cz = D / 2; // scene center

// Area light footprint (mm / 100)
const LX = [2.13, 3.43], LZ = [2.27, 3.32];
const LCX = (LX[0] + LX[1]) / 2, LCZ = (LZ[0] + LZ[1]) / 2;
const LW = LX[1] - LX[0], LD = LZ[1] - LZ[0];

interface Props {
  hoveredHogel?: { hx: number; hy: number; gridW: number; gridH: number };
  parallaxOrbit?: {
    // All coords in Cornell Box space (mm)
    centerX: number; centerY: number; centerZ: number;
    radius: number;
    axis: 'horizontal' | 'vertical';
    frameIdx?: number;   // current active frame (0-based)
    totalFrames?: number;
  };
}

// Build a polyline ring in Three.js space (Cornell Box mm / 100 → Three.js units)
function orbitPoints(
  cx_mm: number, cy_mm: number, cz_mm: number,
  radius_mm: number,
  axis: 'horizontal' | 'vertical',
  segments = 64
): Float32Array {
  const pts: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = (i / segments) * Math.PI * 2;
    if (axis === 'horizontal') {
      pts.push(
        (cx_mm + Math.cos(t) * radius_mm) / 100,
        cy_mm / 100,
        (cz_mm + Math.sin(t) * radius_mm) / 100,
      );
    } else {
      pts.push(
        cx_mm / 100,
        (cy_mm + Math.sin(t) * radius_mm) / 100,
        (cz_mm + Math.cos(t) * radius_mm) / 100,
      );
    }
  }
  return new Float32Array(pts);
}

export function DiagnosticsVisualizer({ hoveredHogel, parallaxOrbit }: Props) {
  // Hogel indicator position on the panel (z=0)
  const hogelX = hoveredHogel
    ? (hoveredHogel.hx + 0.5) / hoveredHogel.gridW * W
    : cx;
  const hogelY = hoveredHogel
    ? (1 - (hoveredHogel.hy + 0.5) / hoveredHogel.gridH) * H
    : cy;

  return (
    <Canvas shadows camera={{ position: [cx + 6, H * 1.3, -D * 0.5], fov: 42 }} frameloop="demand">
      <ambientLight intensity={0.3} />
      <pointLight position={[LCX, H - 0.15, LCZ]} intensity={80} castShadow shadow-mapSize={[512, 512]} />

      <OrbitControls target={[cx, cy, cz]} />

      {/* ── Cornell Box walls ── */}
      {/* Floor (y=0) */}
      <Plane args={[W, D]} position={[cx, 0, cz]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <meshLambertMaterial color="#b8b8b8" />
      </Plane>
      {/* Ceiling (y=H) */}
      <Plane args={[W, D]} position={[cx, H, cz]} rotation={[Math.PI / 2, 0, 0]}>
        <meshLambertMaterial color="#b8b8b8" />
      </Plane>
      {/* Back wall (z=D) */}
      <Plane args={[W, H]} position={[cx, cy, D]} rotation={[0, Math.PI, 0]} receiveShadow>
        <meshLambertMaterial color="#b8b8b8" />
      </Plane>
      {/* Left wall — red (x=0) */}
      <Plane args={[D, H]} position={[0, cy, cz]} rotation={[0, Math.PI / 2, 0]} receiveShadow>
        <meshLambertMaterial color="#a03020" />
      </Plane>
      {/* Right wall — green (x=W) */}
      <Plane args={[D, H]} position={[W, cy, cz]} rotation={[0, -Math.PI / 2, 0]} receiveShadow>
        <meshLambertMaterial color="#208030" />
      </Plane>

      {/* ── Area light on ceiling ── */}
      <Plane args={[LW, LD]} position={[LCX, H - 0.02, LCZ]} rotation={[Math.PI / 2, 0, 0]}>
        <meshBasicMaterial color="#ffffff" />
      </Plane>

      {/* ── Tall box (left-back, ~17° CCW around Y) ── */}
      {/* Center of base footprint: avg of 4 vertices / 100 */}
      <Box
        args={[1.65, 3.30, 2.10]}
        position={[1.855, 1.65, 1.69]}
        rotation={[0, 0.30, 0]}
        castShadow receiveShadow
      >
        <meshLambertMaterial color="#b8b8b8" />
      </Box>

      {/* ── Short box (right-front, ~-17° CW around Y) ── */}
      <Box
        args={[2.10, 1.65, 2.10]}
        position={[3.685, 0.825, 3.51]}
        rotation={[0, -0.30, 0]}
        castShadow receiveShadow
      >
        <meshLambertMaterial color="#b8b8b8" />
      </Box>

      {/* ── Holographic display panel (z=0) ── */}
      <group position={[cx, cy, 0]}>
        <Plane args={[W, H]}>
          <meshBasicMaterial color="#0a0a1a" transparent opacity={0.35} side={THREE.DoubleSide} />
        </Plane>
        <lineSegments>
          <edgesGeometry args={[new THREE.PlaneGeometry(W, H)] as any} />
          <lineBasicMaterial color="#4f46e5" />
        </lineSegments>
      </group>

      {/* ── Active hogel indicator on panel ── */}
      <mesh position={[hogelX, hogelY, 0.02]}>
        <circleGeometry args={[0.12, 16]} />
        <meshBasicMaterial color="#10b981" />
      </mesh>

      {/* ── Observer camera (in front of panel, z=-800mm = -8.0 units) ── */}
      <group position={[cx, cy, -8.0]}>
        <Box args={[0.7, 0.5, 0.9]}>
          <meshLambertMaterial color="#444" />
        </Box>
        {/* Lens cone pointing toward panel (+Z) */}
        <mesh position={[0, 0, 0.65]} rotation={[Math.PI / 2, 0, 0]}>
          <coneGeometry args={[0.22, 0.5, 12]} />
          <meshBasicMaterial color="#222" />
        </mesh>
      </group>

      {/* ── Ray from observer through active hogel ── */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array([cx, cy, -8.0, hogelX, hogelY, 0.0]), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#10b981" transparent opacity={0.5} />
      </line>

      {/* ── Parallax orbit circle ── */}
      {parallaxOrbit && (() => {
        const { centerX, centerY, centerZ, radius, axis, frameIdx, totalFrames = 24 } = parallaxOrbit;
        const ringPts = orbitPoints(centerX, centerY, centerZ, radius, axis);
        // Active camera position on the orbit
        const t = frameIdx !== undefined ? (frameIdx / totalFrames) * Math.PI * 2 : 0;
        const camPos: [number, number, number] = axis === 'horizontal'
          ? [(centerX + Math.cos(t) * radius) / 100, centerY / 100, (centerZ + Math.sin(t) * radius) / 100]
          : [centerX / 100, (centerY + Math.sin(t) * radius) / 100, (centerZ + Math.cos(t) * radius) / 100];
        return (
          <>
            <line>
              <bufferGeometry>
                <bufferAttribute attach="attributes-position" args={[ringPts, 3]} />
              </bufferGeometry>
              <lineBasicMaterial color="#f59e0b" transparent opacity={0.7} />
            </line>
            <mesh position={camPos}>
              <sphereGeometry args={[0.15, 12, 12]} />
              <meshBasicMaterial color="#f59e0b" />
            </mesh>
          </>
        );
      })()}
    </Canvas>
  );
}

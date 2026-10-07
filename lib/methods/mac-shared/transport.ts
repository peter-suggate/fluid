import type { SceneDescription } from "../../core/model";
import type { GPUTimestampPhase } from "../../core/performance-trace";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import type { MAC_PHASES } from "./pipeline";
import type { macOptions } from "./parameters";
import type { MethodParamValues } from "../../core/method-contract";
import type { GPUFluidParticleSource } from "../../core/webgpu-particle-overlay";

/** Transport owns its material state; the host owns projection and publication.
 * All hooks encode GPU work. No step waits for a host field readback.
 */
export interface MacTransportContext {
  device: GPUDevice;
  scene: SceneDescription;
  /** Runtime method values; absent for standalone transport diagnostics. */
  values?: MethodParamValues;
  dimensions: readonly [number, number, number];
  h: readonly [number, number, number];
  initialPhi: Float32Array;
  solids: Uint32Array;
  parameters: GPUBuffer;
  scalars: GPUBuffer;
  solidBuffer: GPUBuffer;
  velocity: GPUBuffer;
  transferredVelocity: GPUBuffer;
  /** Particle volume gathered independently at each staggered face. */
  faceVolumes?: GPUBuffer;
  phi: GPUBuffer;
  signal?: AbortSignal;
}
export interface MacTransport {
  readonly allocatedBytes: number;
  readonly diagnostics: Readonly<Record<string, number>>;
  readonly debug?: Record<string, unknown>;
  /** The material particles, for the particle layer. */
  readonly particleSource?: GPUFluidParticleSource;
  /** Update transport controls without replacing the material state. */
  applyRuntimeValues?(values: MethodParamValues): void;
  encodeInitial(encoder: GPUCommandEncoder): void;
  /** Write transferredVelocity and phi before grid forces and projection. */
  encodeTransfer(encoder: GPUCommandEncoder): void;
  /** Read projected velocity, advance material and rebuild phi. */
  encodeMove(encoder: GPUCommandEncoder): void;
  encodeStats(encoder: GPUCommandEncoder): void;
  updateStats(receipt: Float32Array, info: GPUEulerianInfo): void;
  destroy(): void;
}
export interface MacMethodConfiguration {
  id: string;
  label: string;
  phases: Record<keyof typeof MAC_PHASES, GPUTimestampPhase>;
  validateScene(scene: SceneDescription): void;
  createTransport?(context: MacTransportContext): Promise<MacTransport>;
  sceneWallMode?: boolean;
  resolveOptions?(scene: SceneDescription, values: MethodParamValues): ReturnType<typeof macOptions>;
  liveSceneUniforms?: boolean;
  /** Benchmarkable dispatch specialization; the pressure math is unchanged. */
  directPressure?: boolean;
  /** Bound one submission in estimated stable steps; GPU still chooses actual dt. */
  advanceStepCapacity?: number;
  /** Optional cell × estimated-step budget for browser submission latency. */
  advanceCellStepBudget?: number;
  /** Extra reserved slots for speed growth inside a submission; does not
   * change the physical CFL timestep selected by the GPU. */
  substepCapacityFactor?: number;
  /** Split long encoded advances without changing substeps or publication. */
  commandBufferSubsteps?: number;
  multigridPressure?: boolean;
  /** Encode bounded batches and continue from GPU receipts only as needed. */
  continuationSubsteps?: number;
}

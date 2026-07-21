import type { SceneDescription } from "../model";
import type { GPUQuality } from "../tall-cell-grid";
import type { GPUEulerianInfo, GPURigidLoad } from "../webgpu-eulerian";
import type { RigidBodyState } from "../rigid-body";
import type { GPURigidBodyPick } from "../webgpu-rigid-body";
import type { Vec3 } from "../model";
import type { GPUSecondaryParticleSource } from "../webgpu-secondary-particles";
import type { SparseVoxelRenderSource, SparseVoxelSceneRenderSource } from "../webgpu-voxel-debug";
import type { SparseSurfaceBandGPUSource } from "../webgpu-sparse-surface-band";
import type { OctreeFaceMirrorSource } from "../webgpu-octree-face-mirror";
import type { OctreeFaceVelocitySource } from "../webgpu-octree-face-transport";
import type { OctreeSurfacePageSource } from "../webgpu-octree-surface-pages";
import type { WebGPUFineLevelSetBrickSource } from "../webgpu-octree-fine-levelset-bricks";
import type { GPUInitializationPhase } from "../gpu-initialization";
import type { OctreeFaceBandGPUPlan } from "../webgpu-octree-face-fast-march";
import type { OctreeTechniqueDebugSource } from "../octree-technique-debug";

/**
 * Method plugin contract.
 *
 * A simulation method owns three things:
 *  1. its identity and descriptive copy (picker labels, viewport badge, the
 *     technical summary shown in diagnostics),
 *  2. a declarative parameter schema — common numerics the method consumes
 *     plus any method-specific knobs — which the UI renders generically, and
 *  3. a solver factory that receives the scene and the resolved parameter
 *     values and returns a GPU solver honouring the shared solver interface.
 *
 * Adding a new method means adding one module under lib/methods/ and listing
 * it in the registry; no UI or renderer changes are required.
 */

export type MethodParamValue = number | string | boolean;
export type MethodParamValues = Record<string, MethodParamValue>;

/** An authored, reproducible solver configuration attached to validation scenes. */
export interface MethodProfile {
  methodId: string;
  quality: GPUQuality;
  overrides: MethodParamValues;
}

interface ParamBase {
  key: string;
  label: string;
  /** One-line explanation shown under the control. */
  hint?: string;
  /** Coarse controls are always visible; fine controls sit behind "Advanced". */
  tier: "coarse" | "fine";
  /** Runtime parameters can be applied to a live solver. All others are
   * structural and start a transactional solver rebuild. */
  update?: "runtime" | "solver";
}

export interface NumberParamSpec extends ParamBase {
  kind: "number";
  unit: string;
  min: number;
  max: number;
  step: number;
  digits?: number;
  default: number;
}

export interface SelectParamSpec extends ParamBase {
  kind: "select";
  options: ReadonlyArray<{ value: string; label: string }>;
  default: string;
}

export type MethodParamSpec = NumberParamSpec | SelectParamSpec;

/** Minimal interface the renderer needs from a GPU solver. */
export interface GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  /** Field the renderer contours; a smooth level set when the solver keeps one separate from volumeTexture. */
  readonly surfaceFieldTexture?: GPUTexture;
  readonly columnBaseTexture: GPUTexture;
  /** Adaptive pressure-cell ownership for scientific grid slices. */
  readonly gridCellTexture?: GPUTexture;
  /** Live velocity field for scientific slice modes (CFL/speed heatmaps). */
  readonly velocityTexture?: GPUTexture;
  /** Optional one-way escaped spray droplets rendered above the liquid surface. */
  readonly secondaryParticles?: GPUSecondaryParticleSource;
  /** Always-resident structural sparse scene used by production SVO rendering. */
  readonly sparseVoxelSceneSource?: SparseVoxelSceneRenderSource;
  /** Lazily allocated expanded records used by raw/grid inspection. */
  readonly sparseVoxelRenderSource?: SparseVoxelRenderSource;
  /** Dynamically paged fine phi/velocity band for sparse extraction and inspection. */
  readonly sparseSurfaceBand?: SparseSurfaceBandGPUSource;
  /** Default-off adaptive MAC-face migration mirror and parity counters. */
  readonly adaptiveFaceMirrorSource?: OctreeFaceMirrorSource;
  /** Authoritative compact adaptive MAC velocity and its four-word reduction. */
  readonly adaptiveFaceVelocitySource?: OctreeFaceVelocitySource;
  /** Leaf-attached authoritative narrow-band phi pages for octree-native consumers. */
  readonly adaptiveSurfacePageSource?: OctreeSurfacePageSource;
  /** Exact compact topology/geometry buffers for paper-technique overlays. */
  readonly octreeTechniqueDebugSource?: OctreeTechniqueDebugSource;
  /** QA-only active compact pressure potential, indexed by power-leaf row. */
  readonly powerPressureBuffer?: GPUBuffer;
  /** QA-only compact leaf headers; 48 bytes per pressure row. */
  readonly powerLeafHeaders?: GPUBuffer;
  /** QA-only compact pressure CSR entries; `(neighborRow:u32, coefficient:f32)`. */
  readonly powerLeafEntries?: GPUBuffer;
  /** True only after the complete t=0 sparse authority has passed its queue fence. */
  readonly initialSparseAuthorityReady?: boolean;
  /** Row-independent, globally indexed sparse fine level-set bricks. */
  readonly globalFineLevelSetSource?: WebGPUFineLevelSetBrickSource;
  /** Diagnostic-only; never participates in authority selection. */
  readonly globalFineTransportControl?: GPUBuffer;
  readonly globalFineRedistanceControl?: GPUBuffer;
  readonly globalFineVolumeControl?: GPUBuffer;
  readonly globalFinePowerVelocityControl?: GPUBuffer;
  readonly globalFinePowerProjectionControl?: GPUBuffer;
  /** QA-only exact power-face endpoint queries; never affects publication. */
  readonly powerBoundaryPhiQueries?: GPUBuffer;
  /** QA-only packed owner lattice readback for topology forensics. */
  readonly ownerLatticeDebug?: {
    buffer: GPUBuffer;
    paged: boolean;
    maximumLeafSize: number;
    dimensions: readonly [number, number, number];
  };
  /** QA-only generation/slot sampled by the last power boundary build. */
  readonly powerBoundaryFineSource?: { generation: number; generationSlot: 0 | 1 };
  /** Diagnostic-only Stage-B point-sampler transaction used by fine transport. */
  readonly globalFinePowerVelocitySampleControl?: GPUBuffer;
  readonly globalFineCoarseLevelSetControl?: GPUBuffer;
  readonly globalFineRestrictionControl?: GPUBuffer;
  /** Diagnostic-only Section 5 regular-face extrapolation header and exact capacity plan. */
  readonly globalFineFaceBandControl?: GPUBuffer;
  /** Diagnostic-only catalog-Delaunay transition gate preceding regular-face emission. */
  readonly globalFineFaceBandTransitionControl?: GPUBuffer;
  /** Diagnostic-only transactional cell-vector field reconstructed after the face march. */
  readonly globalFineFaceBandPointFieldControl?: GPUBuffer;
  /** Diagnostic-only all-band transient physical generalized-face graph transaction. */
  readonly globalFineFaceBandTransientPowerControl?: GPUBuffer;
  /** Diagnostic-only regular-face to power-face publication transaction. */
  readonly globalFineFaceBandPowerPublicationControl?: GPUBuffer;
  readonly globalFineFaceBandPlan?: OctreeFaceBandGPUPlan;
  /** Failure-only bounded readback for a disconnected Section-5 face. */
  readGlobalFineDisconnectedFaceFailure?(index: number): Promise<unknown> | undefined;
  /** GPU-authored rigid records matching the renderer's four-vec4 body ABI. */
  readonly rigidRenderBuffer?: GPUBuffer;
  /** GPU-authored 128-byte primitive-motion sidecars, including conservative swept bounds. */
  readonly rigidMotionBuffer?: GPUBuffer;
  /** Updates selection metadata without mirroring dynamic poses through CPU memory. */
  setSelectedRigidBody?(index: number): void;
  /** User-triggered ray query against authoritative GPU rigid poses. */
  pickRigidBody?(origin: Vec3, direction: Vec3): Promise<GPURigidBodyPick | undefined>;
  /** Adaptive pressure-DOF ownership used by the representation alarm. */
  readonly gridPressureSamplesTexture?: GPUTexture;
  /** Fine MLS pressure materialized by the latest adaptive solve. */
  readonly gridPressureTexture?: GPUTexture;
  /** Post-projection fine-cell divergence diagnostic. */
  readonly gridDivergenceTexture?: GPUTexture;
  /** Lazily allocate dense adaptive fields when a scientific grid slice needs them. */
  ensureGridDiagnosticTextures?(): void;
  /** Apply configuration explicitly classified as runtime-safe by the method. */
  applyRuntimeValues?(values: MethodParamValues): void;
  advanceTo(time_s: number, bodies: RigidBodyState[]): boolean;
  readStats(): Promise<GPUEulerianInfo>;
  destroy(): void;
}

export interface SimulationMethod {
  id: string;
  /** Full name shown in the method picker. */
  label: string;
  /** Short name for segmented controls and the top bar. */
  shortLabel: string;
  /** Upper-case badge shown over the viewport. */
  badge: string;
  /** One-sentence summary for the picker. */
  description: string;
  /** Technical summary for the diagnostics panel. */
  detail: string;
  /** Where the authoritative fluid state lives. */
  backend: "webgpu" | "cpu";
  /** Per-quality flavour text for the quality selector. */
  qualityLabels: Record<GPUQuality, string>;
  /**
   * Method-specific parameters. Common parameters (resolution, time step,
   * pressure solve effort) live in the scene numerics and are declared once
   * in the common schema, not here.
   */
  params: ReadonlyArray<MethodParamSpec>;
  /**
   * How this method interprets the common "pressure solve effort" iteration
   * budget (scene.numerics.pressureMaxIterations). Purely descriptive; the
   * mapping itself happens in createSolver.
   */
  pressureMapping: string;
  /**
   * Effective parameter values implied by a quality preset. The UI shows
   * these as the baseline; user overrides are stored sparsely on top and
   * merged via resolveMethodValues before reaching createSolver.
   */
  presetFor(quality: GPUQuality): MethodParamValues;
  /** Keys omitted from the structural solver fingerprint and applied directly
   * to the active/candidate solver instead. */
  runtimeParamKeys?: readonly string[];
  /** WebGPU methods create a solver; the CPU reference method does not. */
  createSolver?(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    values: MethodParamValues,
    onRigidLoads?: (loads: GPURigidLoad[]) => void
  ): GPUSolverInstance;
  /** Browser-safe construction path. Long shader compilation must use the
   * asynchronous WebGPU pipeline APIs so the main thread can keep painting. */
  createSolverAsync?(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    values: MethodParamValues,
    onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    onProgress: GPUInitializationReporter,
    signal?: AbortSignal,
  ): Promise<GPUSolverInstance>;
}

export interface GPUInitializationProgress {
  phase: GPUInitializationPhase;
  taskId?: string;
  label: string;
  completed: number;
  total: number;
}

export type GPUInitializationReporter = (progress: GPUInitializationProgress) => void;

export function resolveMethodValues(method: SimulationMethod, quality: GPUQuality, overrides: MethodParamValues): MethodParamValues {
  const defaults = Object.fromEntries(method.params.map((spec) => [spec.key, spec.default]));
  return { ...defaults, ...method.presetFor(quality), ...overrides };
}

export function numberValue(values: MethodParamValues, spec: ReadonlyArray<MethodParamSpec>, key: string): number {
  const declared = spec.find((candidate) => candidate.key === key);
  const raw = values[key];
  if (typeof raw === "number" && Number.isFinite(raw)) {
    if (declared?.kind === "number") return Math.min(declared.max, Math.max(declared.min, raw));
    return raw;
  }
  return declared?.kind === "number" ? declared.default : 0;
}

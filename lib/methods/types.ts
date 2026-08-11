import type { SceneDescription } from "../model";
import type { GPUQuality } from "../tall-cell-grid";
import type { GPUEulerianInfo, GPURigidLoad } from "../webgpu-eulerian";
import type { RigidBodyState } from "../rigid-body";
import type { GPURigidBodyPick, GPURigidBodyPose } from "../webgpu-rigid-body";
import type { Vec3 } from "../model";
import type { GPUSecondaryParticleSource } from "../webgpu-secondary-particles";
import type { SparseVoxelSceneRenderSource } from "../webgpu-voxel-debug";
import type { WebGPUFineLevelSetBrickSource } from "../webgpu-octree-fine-levelset-bricks";
import type { GPUInitializationPhase } from "../gpu-initialization";
import type { OctreeTechniqueDebugSource } from "../octree-technique-debug";
import type { CoarseLevelSetConsumerSource } from "../octree-consumer-sampling";
import type { ResourcePluginDefinition } from "../resource-readiness";
import type { SparseScenePrimitiveUpdate } from "../webgpu-sparse-scene-proxies";

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
  /** Dense tall-column metadata; compact octree methods have no such field. */
  readonly columnBaseTexture?: GPUTexture;
  /** Adaptive pressure-cell ownership for scientific grid slices. */
  readonly gridCellTexture?: GPUTexture;
  /** Live velocity field for scientific slice modes (CFL/speed heatmaps). */
  readonly velocityTexture?: GPUTexture;
  /** Optional one-way escaped spray droplets rendered above the liquid surface. */
  readonly secondaryParticles?: GPUSecondaryParticleSource;
  /** Always-resident structural sparse scene used by production SVO rendering. */
  readonly sparseVoxelSceneSource?: SparseVoxelSceneRenderSource;
  /** Exact compact topology/geometry buffers for paper-technique overlays. */
  readonly octreeTechniqueDebugSource?: OctreeTechniqueDebugSource;
  /** QA-only active compact pressure potential, indexed by power-leaf row. */
  readonly powerPressureBuffer?: GPUBuffer;
  /** QA-only compact leaf headers; 48 bytes per pressure row. */
  readonly powerLeafHeaders?: GPUBuffer;
  /** True only after the complete t=0 sparse authority has passed its queue fence. */
  readonly initialSparseAuthorityReady?: boolean;
  /** Row-independent, globally indexed sparse fine level-set bricks. */
  readonly globalFineLevelSetSource?: WebGPUFineLevelSetBrickSource;
  /** Compact moving surface used when coarse-1 deliberately has no fine band. */
  readonly coarseLevelSetSource?: CoarseLevelSetConsumerSource;
  /** Diagnostic-only; never participates in authority selection. */
  readonly globalFineTransportControl?: GPUBuffer;
  readonly globalFineRedistanceControl?: GPUBuffer;
  readonly globalFineVolumeControl?: GPUBuffer;
  readonly structuredProjectionEnergyStats?: GPUBuffer;
  /** QA-only native Losasso axis-face authority for Dawn field reconstruction. */
  readonly losassoVelocityDebug?: {
    readonly control: GPUBuffer;
    readonly faceGeometry: GPUBuffer;
    readonly extendedVelocity: GPUBuffer;
    readonly dimensions: readonly [number, number, number];
    readonly maximumLeafSize: number;
  };
  /** QA-only accepted Losasso row operator, with no Power structured ABI. */
  readonly losassoPressureDebug?: {
    readonly control: GPUBuffer;
    readonly rightHandSide: GPUBuffer;
    readonly diagonal: GPUBuffer;
  };
  /** QA-only compact topology-delta streams for rejection forensics. */
  readonly globalFinePageDeltaDebug?: {
    readonly buffer: GPUBuffer;
    readonly params: GPUBuffer;
    readonly sparseCandidates: GPUBuffer;
    readonly sparseCandidateCapacity: number;
    readonly pageCapacity: number;
    readonly changedKeysOffsetWords: number;
    readonly dirtyPagesOffsetWords: number;
    readonly supportPagesOffsetWords: number;
    readonly promotionCountsOffsetWords: number;
  };
  /** QA-only sparse owner-page arena readback for topology forensics. */
  readonly ownerLatticeDebug?: {
    buffer: GPUBuffer;
    maximumLeafSize: number;
    dimensions: readonly [number, number, number];
  };
  /** QA-only generation/slot sampled by the last power boundary build. */
  readonly powerBoundaryFineSource?: { generation: number; generationSlot: 0 | 1 };
  /** QA-only exact sparse source sampled by the last power boundary build. */
  readonly powerBoundaryFineLevelSetSource?: WebGPUFineLevelSetBrickSource;
  readonly globalFineCoarseLevelSetControl?: GPUBuffer;
  readonly globalFineRestrictionControl?: GPUBuffer;
  /** QA-only first-failure receipt for Section-5 air-support publication. */
  readonly airSupportScratch?: GPUBuffer;
  /** GPU-authored rigid records matching the renderer's four-vec4 body ABI. */
  readonly rigidRenderBuffer?: GPUBuffer;
  /** GPU-authored 128-byte primitive-motion sidecars, including conservative swept bounds. */
  readonly rigidMotionBuffer?: GPUBuffer;
  /** Updates selection metadata without mirroring dynamic poses through CPU memory. */
  setSelectedRigidBody?(index: number): void;
  /**
   * Stage the latest authoritative scene revision for GPU consumers.
   *
   * Scene geometry, materials, lighting, and eventually fluid-domain inputs
   * all travel through this one live update seam. Implementations retain no
   * separate baked/fluid-free scene identity; they may reuse generation-matched
   * acceleration data, but the supplied scene is always the authority.
   */
  stageSceneUpdate?(scene: SceneDescription): void;
  /** Stage allocation-free keyed primitive motion into the shared sparse scene. */
  stageLivePrimitiveUpdates?(updates: readonly SparseScenePrimitiveUpdate[]): boolean;
  /**
   * Encode bounded maintenance for the scene revision staged above.
   * Presentation calls this even while simulation is paused so sparse
   * accelerators can converge without a solver step or a queue fence.
   */
  encodeSceneMaintenance?(encoder: GPUCommandEncoder): void;
  /** User-triggered ray query against authoritative GPU rigid poses. */
  pickRigidBody?(origin: Vec3, direction: Vec3): Promise<GPURigidBodyPick | undefined>;
  /**
   * Authoritative poses for the bodies the solver owns, in roster order.
   *
   * The host roster is a command channel — it is never written back from the
   * run — so anything that must agree with the drawn frame asks here.
   */
  readRigidBodyPoses?(): Promise<GPURigidBodyPose[] | undefined>;
  /** Adaptive pressure-DOF ownership used by the representation alarm. */
  readonly gridPressureSamplesTexture?: GPUTexture;
  /** Fine MLS pressure materialized by the latest adaptive solve. */
  readonly gridPressureTexture?: GPUTexture;
  /** Post-projection fine-cell divergence diagnostic. */
  readonly gridDivergenceTexture?: GPUTexture;
  /** Lazily allocate dense adaptive fields when a scientific grid slice needs them. */
  ensureGridDiagnosticTextures?(): void;
  /**
   * Adopt scene scalars that no lattice or seed depends on — density,
   * viscosity, surface tension, gravity. Solvers read these from the retained
   * scene when they write per-step params, so adopting a new scene is a
   * uniform write rather than a rebuild. Implementing this is what lets
   * `gpuSceneUniformKey` stay out of the rebuild trigger.
   */
  applySceneUniforms?(scene: SceneDescription): void;
  /**
   * Re-seed t=0 in place for a scene that differs only in the seed tier,
   * reusing every allocation, arena, and compiled pipeline. Resolves false
   * when the seed cannot be honoured, in which case the caller must take the
   * full rebuild — the solver is left usable but not re-seeded.
   */
  reseed?(scene: SceneDescription): Promise<boolean>;
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
  /** Resource lifecycle declaration, colocated with the method that owns its initialization. */
  resource?: ResourcePluginDefinition;
  /** Per-quality flavour text for the quality selector. */
  qualityLabels: Record<GPUQuality, string>;
  /** Hide the generic quality selector when the method exposes its relevant
   * scientific trade-offs directly. */
  showQualityControl?: boolean;
  /**
   * Field-visualization modes (`GridOverlayMode` strings) whose data this
   * method's solver actually publishes. Pickers narrow the visualization
   * catalog to these, so a method never offers a view of a publication it
   * does not produce — the uniform reference has no power topology, and the
   * CPU reference has no GPU fields at all. Declared as strings because the
   * modes belong to the catalog entries, not to the method module.
   */
  supportedFieldModes?: readonly string[];
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
  /**
   * Resolve method-owned invariants after defaults, the quality preset, and
   * sparse user overrides have been merged. This is the right seam for a
   * compound scientific choice whose dependent controls must survive URL
   * hydration and scene-profile application as one valid configuration.
   */
  normalizeValues?(values: MethodParamValues): MethodParamValues;
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
  const merged = { ...defaults, ...method.presetFor(quality), ...overrides };
  return method.normalizeValues?.(merged) ?? merged;
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

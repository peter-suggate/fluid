import { combineInitialBrickWet, damBreakFractions, initialFluidBrickContainsCell } from "./initial-fluid";
import {
  DynamicGPUPerformanceTraceRecorder,
  GPUPerformanceTraceRecorder,
  GPUQueueWallPerformanceTraceRecorder,
  GPUSegmentedQueueWallPerformanceTraceRecorder,
  type GPUTimestampPhase,
} from "./performance-trace";
import { usePerformanceInstrumentationStore } from "./stores/performance-instrumentation-store";
import { initializeRigidBodies, type RigidBodyState } from "./rigid-body";
import {
  GPU_RIGID_EXCHANGE_BYTES,
  legacyUniformComputeShader,
  type GPUEulerianInfo,
  type GPURigidLoad,
  type GPUVelocityTransport,
  type GPUQuality
} from "./webgpu-eulerian";
import type { SceneDescription } from "./model";
import { createTallCellLayout } from "./tall-cell-grid";
import { planGPUAdvance } from "./tall-cell-diagnostics";
import { averageInflowStrength, createInflowGridBoundary, type InflowGridBoundary } from "./inflow-boundary";
import { quadtreeChebyshevSpectrum, WebGPUQuadtreeTallCellProjection, type QuadtreeTallCellProjectionOptions } from "./webgpu-quadtree-tall-cell";
import { OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES, WebGPUOctreeProjection,
  type OctreeSemanticPhase, type OctreeProjectionOptions } from "./webgpu-octree";
import {
  OCTREE_POWER_COARSE_LEVELSET_ERROR,
  OCTREE_POWER_COARSE_LEVELSET_VALID,
  unpackOctreePowerCoarseLevelSetControl,
} from "./webgpu-octree-power-coarse-levelset";
import {
  FINE_LEVELSET_TOPOLOGY_ERROR,
  FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON,
  unpackFineLevelSetGPUTopologyControl,
} from "./webgpu-octree-fine-levelset-topology";
import { readFineLevelSetWorksetHeader } from "./octree-fine-levelset-bricks";
import { unpackFineLevelSetGPURedistanceControl } from "./webgpu-octree-fine-levelset-redistance";
import { unpackFineLevelSetGPUTransportControl } from "./webgpu-octree-fine-levelset-transport";
import { FINE_TO_COARSE_LEVELSET_ERROR, unpackFineToCoarseGPUControl } from "./webgpu-octree-fine-to-coarse-levelset";
import { planUniformHostAllocation, type UniformHostAllocationPlan } from "./octree-host-allocation";
import { sceneHasTerrain, terrainColumnHeights } from "./terrain";
import { WebGPURigidBodySystem } from "./webgpu-rigid-body";
import { GPUInitializationTaskRunner, type GPUInitializationTask } from "./gpu-initialization";
import {
  FINE_LEVELSET_VOLUME_VALID,
  unpackFineLevelSetGPUVolumeControl,
} from "./webgpu-octree-fine-levelset-volume";
import { supportsFluidM1MaxReduction } from "./webgpu-device-limits";
import {
  decodeStructuredProjectionEnergy,
  STRUCTURED_PROJECTION_ENERGY_WORDS,
} from "./webgpu-octree-structured-dynamics";
import { decodeOctreeStructuredRejectCarry } from "./octree-structured-reject-carry";

export type UniformVelocityTransport = GPUVelocityTransport;
export interface WebGPUUniformEulerianOptions { pressureIterations?: number; velocityTransport?: UniformVelocityTransport; densitySharpening?: boolean; tallCellSettings?: Partial<import("./tall-cell-grid").TallCellSettings>; quadtreeTallCells?: Partial<QuadtreeTallCellProjectionOptions>; octree?: Partial<OctreeProjectionOptions>; /** Allocate escaped spray droplets and set their initial live state. */ secondaryParticles?: boolean; secondaryParticleCapacity?: number; quadtreeRebuildTopology?: boolean; quadtreeRebuildIntervalSteps?: number; quadtreeTopologyStaleSteps?: number; /** Fully GPU-resident every-step topology regeneration (Algorithm 1); default on for uncoupled parallel preconditioners. */ quadtreeInlineRebuild?: boolean; deferPipelineCompilation?: boolean }

// Pipeline objects are immutable and device-scoped. Rebuilding buffers or
// textures for a settings change must not ask the browser/driver to compile
// identical programs again; that compilation can block input for seconds even
// through createComputePipelineAsync.
const uniformPipelineCache = new WeakMap<GPUDevice, Map<UniformVelocityTransport, GPUComputePipeline[]>>();

const OCTREE_SEMANTIC_TRACE_PHASE: Readonly<Record<OctreeSemanticPhase, GPUTimestampPhase>> = {
  structureEpoch: { id: "coarse-grid", label: "[engine:structure-epoch] Structure epoch" },
  rowEngineA: { id: "velocity-advection", label: "[engine:row-a] Coarse values + face completion" },
  solveEngine: { id: "pressure-solve", label: "[engine:solve] Pipelined MGPCG pressure solve" },
  rowEngineB: { id: "velocity-extrapolation", label: "[engine:row-b] Projection + velocity extension" },
  brickEngineA: { id: "fine-sdf-advection", label: "[engine:brick-a] Fine transport + topology" },
  closestPointWaves: { id: "fine-sdf-redistance", label: "[engine:cpt-waves] Closest-point waves" },
  brickEngineB: { id: "adaptive-publication", label: "[engine:brick-b] Fine harvest + epoch gate" },
  powerDescriptorTopology: { id: "power-topology", label: "Power topology publication" },
  structuredAdvectionBoundaryRhs: { id: "velocity-advection", label: "Structured advection + boundary RHS" },
  structuredVolumeCapture: { id: "pressure-system", label: "Structured physical-volume capture" },
  finalPressureRowAssembly: { id: "pressure-system", label: "Final pressure row assembly" },
  mgpcgSolve: { id: "pressure-solve", label: "Selected power pressure solve" },
  structuredProjection: { id: "velocity-projection", label: "Structured pressure projection + CPT seeds" },
  structuredProjectionTail: { id: "velocity-extrapolation", label: "Air support + projection publication" },
  finePreparation: { id: "fine-sdf-advection", label: "Fine SDF seed + transport preparation" },
  fineTransport: { id: "fine-sdf-advection", label: "Factor-m fine SDF advection" },
  fineTopology: { id: "coarse-grid", label: "Fine narrow-band topology" },
  fineRedistance: { id: "fine-sdf-redistance", label: "Fine SDF redistance" },
  fineRestriction: { id: "adaptive-publication", label: "Fine-to-coarse restriction" },
};

/** Serial queue probes are deliberately sparse because they trade throughput
 * for portable, per-phase measurements when timestamp queries fail. */
const PHYSICS_TRACE_CADENCE_MS = 100;
const SEGMENTED_QUEUE_TRACE_CADENCE_MS = 1_000;

/** Explicit capillary-wave stability bound for a finest cell. */
export function capillaryStableDt_s(
  density_kg_m3: number,
  surfaceTension_N_m: number,
  minimumCellSize_m: number,
  safety = 0.5
) {
  const density = Number.isFinite(density_kg_m3) ? Math.max(1e-9, density_kg_m3) : 1e-9;
  const sigma = Number.isFinite(surfaceTension_N_m) ? Math.max(0, surfaceTension_N_m) : 0;
  const h = Number.isFinite(minimumCellSize_m) ? Math.max(1e-9, minimumCellSize_m) : 1e-9;
  const boundedSafety = Number.isFinite(safety) ? Math.max(0.05, Math.min(1, safety)) : 0.5;
  return sigma > 0 ? boundedSafety * Math.sqrt(density * h * h * h / (Math.PI * sigma)) : Number.POSITIVE_INFINITY;
}

/** Readback-free CFL and capillary subdivision for the next adaptive frame. */
export function proactiveQuadtreeSubsteps(
  previousMaxSpeed_m_s: number,
  inflowSpeed_m_s: number,
  gravityMagnitude_m_s2: number,
  frameDt_s: number,
  minimumCellSize_m: number,
  maximumSubsteps = 64,
  density_kg_m3 = 1_000,
  surfaceTension_N_m = 0
) {
  const safeDt = Math.max(0, Number.isFinite(frameDt_s) ? frameDt_s : 0);
  const safeCell = Math.max(1e-9, Number.isFinite(minimumCellSize_m) ? minimumCellSize_m : 0);
  const residentBound = Math.max(0, previousMaxSpeed_m_s, inflowSpeed_m_s);
  const velocityBound = residentBound + Math.max(0, gravityMagnitude_m_s2) * safeDt;
  const cflRequired = Math.ceil(velocityBound * safeDt / safeCell);
  const capillaryDt = capillaryStableDt_s(density_kg_m3, surfaceTension_N_m, safeCell);
  const capillaryRequired = Number.isFinite(capillaryDt) && capillaryDt > 0 ? Math.ceil(safeDt / capillaryDt) : 1;
  const required = Math.max(cflRequired, capillaryRequired);
  return Math.max(1, Math.min(Math.max(1, Math.floor(maximumSubsteps)), required));
}

/** Convert a stale-limit wait into actual missed 60 Hz presentation frames. */
export function quadtreeMissedFrames(wait_ms: number, frameBudget_ms = 1000 / 60) {
  if (!(wait_ms > 0) || !(frameBudget_ms > 0)) return 0;
  return Math.max(0, Math.ceil(wait_ms / frameBudget_ms) - 1);
}

/** Bounded exponential backoff while a valid previous topology stays live. */
export function quadtreeRebuildRetryDelay(failureCount: number) {
  if (!(failureCount > 0)) return 0;
  return Math.min(60, 2 ** Math.min(6, Math.floor(failureCount)));
}

export interface GlobalFineVolumePublicationDiagnostics {
  readonly published?: boolean;
  readonly rolledBack?: boolean;
  readonly downstreamFinalizeReason?: number;
  readonly generation?: number;
  readonly volumeControl?: readonly number[];
}

export interface InitialGlobalFineAuthorityDiagnostics extends GlobalFineVolumePublicationDiagnostics {
  readonly seedControl: readonly number[];
  readonly topologyControl: readonly number[];
  readonly fineVolumeControl: readonly number[];
  readonly worklistHeader: readonly number[];
  readonly coarseControl: readonly number[];
  readonly fineRestrictionControl: readonly number[];
  readonly structuredVelocityControl: readonly number[];
  readonly structuredBoundaryControl: readonly number[];
  readonly configuredFineGeneration: number;
  readonly fineGenerationSlot: 0 | 1;
  readonly scheduledFineGeneration: number;
  readonly currentFineIsA: boolean;
}

/** Publish the bounded fine-transport control into the shared UI snapshot.
 * The diagnostic position is solver-local metres and is present only when
 * the GPU captured a real invalid-status sample rather than the sentinel. */
export function applyGlobalFineTransportDiagnostics(
  info: GPUEulerianInfo,
  words: readonly number[] | undefined,
): void {
  if (!words || words.length < 8) return;
  const transport = unpackFineLevelSetGPUTransportControl(words);
  info.globalFineTransportDepartureOutsideBand = transport.departureOutsideBand;
  info.globalFineTransportNonfiniteVelocity = transport.nonfiniteVelocity;
  info.globalFineTransportCommitted = transport.committed;
  info.globalFineTransportStructuredAuthorityUnavailable = transport.structuredAuthorityUnavailable;
  info.globalFineTransportVelocityUnavailable = transport.velocityUnavailable;
  info.globalFineTransportInvalidVelocityStatus = transport.invalidVelocityStatus;
  info.globalFineTransportNonpositiveVelocityResult = transport.nonpositiveVelocityResult;
  info.globalFineTransportVelocityStatusReasonOr = transport.velocityStatusReasonOr;
  const first = transport.firstInvalidVelocityLocalIndex;
  const position = transport.firstInvalidVelocityPosition;
  const captured = first !== undefined && first !== 0xffff_ffff && position !== undefined
    && position.every(Number.isFinite);
  info.globalFineTransportFirstInvalidVelocityStatus = captured
    ? transport.firstInvalidVelocityStatus : undefined;
  info.globalFineTransportFirstInvalidVelocityLocalIndex = captured ? first : undefined;
  info.globalFineTransportFirstInvalidVelocityPosition_m = captured
    ? { x: position[0], y: position[1], z: position[2] } : undefined;
}

export interface InitialSparseAuthorityReadiness {
  readonly ready: boolean;
  readonly label: string;
}

function namedControlBits(bits: number, values: Readonly<Record<string, number>>): string[] {
  return Object.entries(values).filter(([, mask]) => (bits & mask) !== 0).map(([name]) => name);
}

/** Durable, JSON-safe decoding of the bounded t=0 readback. Startup errors
 * include this object verbatim, so evidence remains available after the
 * renderer releases the failed GPU device. */
export function initialGlobalFineAuthorityEvidence(value: InitialGlobalFineAuthorityDiagnostics) {
  const topology=unpackFineLevelSetGPUTopologyControl(value.topologyControl);
  const coarse=unpackOctreePowerCoarseLevelSetControl(value.coarseControl);
  const restriction=unpackFineToCoarseGPUControl(value.fineRestrictionControl);
  return {
    generation:{configured:value.configuredFineGeneration,
      scheduled:value.scheduledFineGeneration},
    seeds:{count:value.seedControl[0] ?? 0,flags:value.seedControl[1] ?? 0,raw:value.seedControl},
    topology:{...topology,
      errors:namedControlBits(topology.flags,FINE_LEVELSET_TOPOLOGY_ERROR),
      downstream:namedControlBits(topology.downstreamFinalizeReason,FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON)},
    worklist:value.worklistHeader,
    coarse:{...coarse,errors:namedControlBits(coarse.flags,OCTREE_POWER_COARSE_LEVELSET_ERROR)},
    restriction:{...restriction,errors:namedControlBits(restriction.flags,FINE_TO_COARSE_LEVELSET_ERROR)},
    structuredVelocity:value.structuredVelocityControl,
    structuredBoundary:value.structuredBoundaryControl,
  };
}

/** One-time CPU acceptance mirror for the fenced t=0 publication. It proves
 * Section 5's paired fine/coarse level set and complete velocity round trip;
 * it never selects an alternative simulation path. */
export function initialGlobalFineAuthorityReadiness(
  value: InitialGlobalFineAuthorityDiagnostics | undefined,
  options: { readonly externallySeededColdBootstrap?: boolean } = {},
): InitialSparseAuthorityReadiness {
  if (!value) return { ready: false, label: "global-fine diagnostics are unavailable" };
  const generation = value.configuredFineGeneration & 0x3fff_ffff;
  const rejected = (reason: string): InitialSparseAuthorityReadiness => ({ ready: false,
    label: `${reason}: ${JSON.stringify(initialGlobalFineAuthorityEvidence(value))}` });
  const topology=unpackFineLevelSetGPUTopologyControl(value.topologyControl);
  const coarse=unpackOctreePowerCoarseLevelSetControl(value.coarseControl);
  const restriction=unpackFineToCoarseGPUControl(value.fineRestrictionControl);
  if ((value.seedControl[0] ?? 0) === 0 || (value.seedControl[1] ?? 0) !== 0) {
    return rejected("global-fine interface seeds are invalid");
  }
  if (!topology.published || topology.rolledBack || topology.flags !== 0
    || topology.downstreamFinalizeReason !== 0 || topology.desiredBricks === 0
    || topology.activatedBricks === 0
    || (readFineLevelSetWorksetHeader(value.worklistHeader)?.generation ?? 0) === 0
    || (!options.externallySeededColdBootstrap && topology.interfaceBricks === 0)) {
    return rejected("global-fine topology rejected");
  }
  if (generation === 0 || (value.scheduledFineGeneration & 0x3fff_ffff) !== generation) {
    return rejected("global-fine topology generation is stale");
  }
  if (coarse.valid !== OCTREE_POWER_COARSE_LEVELSET_VALID || coarse.flags !== 0
    || (coarse.generation & 0x3fff_ffff) !== generation) {
    return rejected("compact coarse level set is not paired with the fine generation");
  }
  // Fine-band samples need not own a liquid pressure row: after an advective
  // step the authoritative fine interface can lead the row topology by a
  // subcell distance. Restriction counts these misses observationally while
  // consumers sample fine before coarse fallback. Therefore validity/flags,
  // not the raw miss count, are the authority predicate.
  if (restriction.flags !== 0 || restriction.rowCount === 0 || !restriction.valid) {
    return rejected("fine-to-coarse level-set restriction did not publish");
  }
  const velocity=value.structuredVelocityControl,boundary=value.structuredBoundaryControl;
  if (velocity.length<6||velocity[0]!==0||velocity[2]===0||velocity[3]!==generation||velocity[4]>1
    ||boundary.length<7||boundary[0]!==0||boundary[2]!==velocity[2]
    ||boundary[4]!==generation||boundary[5]!==velocity[4]||boundary[6]!==generation) {
    return rejected("structured velocity/boundary authority did not publish");
  }
  return { ready: true, label: `global fine/coarse and structured generation ${generation} published` };
}

export interface InitialPowerPressureDiagnostics {
  readonly authoritative: boolean;
  readonly solverLabel: string;
  readonly pressureRows: number;
  readonly capacityOverflow: boolean;
  readonly mgpcgControl?: Uint32Array;
}

/** The sole production pressure solver uses this fail-closed control ABI and
 * 1e-4 native-L2 residual gate. */
export function initialPowerPressureReadiness(
  value: InitialPowerPressureDiagnostics,
): InitialSparseAuthorityReadiness {
  const section43 = value.solverLabel.includes("Section 4.3 hybrid");
  if (!value.authoritative || !section43) {
    return { ready: false, label: "power MGPCG authority is unavailable" };
  }
  if (value.capacityOverflow || value.pressureRows === 0) {
    return { ready: false, label: "resolved power rows did not publish" };
  }
  const words = value.mgpcgControl;
  if (!words || words.length < 16) return { ready: false, label: "selected pressure control is unavailable" };
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const residualSquared = floats[10] + floats[11];
  const rhsSquared = floats[8] + floats[9];
  const relativeSquared = residualSquared / Math.max(rhsSquared, 1e-30);
  const residualAccepted = relativeSquared <= 1e-8;
  if (words[0] !== 0 || words[1] === 0 || words[4] !== value.pressureRows
    || !Number.isFinite(residualSquared) || residualSquared < 0
    || !Number.isFinite(rhsSquared) || rhsSquared < 0
    || !Number.isFinite(relativeSquared) || !residualAccepted) {
    return { ready: false, label: "selected pressure solver did not converge through its residual gate" };
  }
  return { ready: true, label: `Section 4.3 power pressure published (${value.pressureRows} rows)` };
}

/**
 * Decode volume telemetry only when it belongs to the accepted compact-fine
 * publication. The A/B fine fields share one controller, so after rollback
 * that controller describes the rejected candidate and must not be reported.
 */
export function publishedGlobalFineVolumeCells(
  diagnostics: GlobalFineVolumePublicationDiagnostics,
  baseCellVolume_m3: number,
) {
  if (!diagnostics.published || diagnostics.rolledBack || diagnostics.downstreamFinalizeReason !== 0
    || !diagnostics.volumeControl || diagnostics.volumeControl.length < 16 || !(baseCellVolume_m3 > 0)
    || !Number.isFinite(baseCellVolume_m3)) return undefined;
  const bytes = new ArrayBuffer(64);
  new Uint32Array(bytes).set(diagnostics.volumeControl.slice(0, 16));
  const control = unpackFineLevelSetGPUVolumeControl(bytes);
  if (control.flags !== FINE_LEVELSET_VOLUME_VALID || !control.initialized
    || control.generation !== diagnostics.generation || control.coarseRows === 0
    || control.lookupFailureSamples !== 0 || control.staleOwnerSamples !== 0
    || !(control.referenceVolume > 0) || !Number.isFinite(control.referenceVolume)
    || !(control.currentVolume > 0) || !Number.isFinite(control.currentVolume)) return undefined;
  const referenceVolumeCells = control.referenceVolume / baseCellVolume_m3;
  const volumeCells = control.currentVolume / baseCellVolume_m3;
  return { referenceVolumeCells, volumeCells,
    drift: (volumeCells - referenceVolumeCells) / referenceVolumeCells };
}

export interface SparseSurfaceVolumeDiagnostics {
  readonly referenceVolumeCells: number;
  readonly volumeCells: number;
}

/**
 * Compact analytic startup intentionally gives the retired dense level-set
 * owner a one-texel placeholder. Sparse pages report their transported volume
 * as a delta from that owner's reference, so restore the physical t=0 volume
 * before publishing drift telemetry instead of treating zero as an empty tank.
 */
export function sparseSurfaceVolumeCells(
  diagnostics: SparseSurfaceVolumeDiagnostics,
  initialVolumeCells: number,
): SparseSurfaceVolumeDiagnostics {
  if (Number.isFinite(diagnostics.referenceVolumeCells) && diagnostics.referenceVolumeCells > 0) {
    return diagnostics;
  }
  const referenceVolumeCells = Number.isFinite(initialVolumeCells) && initialVolumeCells > 0
    ? initialVolumeCells : diagnostics.referenceVolumeCells;
  return {
    referenceVolumeCells,
    volumeCells: referenceVolumeCells + diagnostics.volumeCells - diagnostics.referenceVolumeCells,
  };
}

const quadtreePressureLabel = (projection: WebGPUQuadtreeTallCellProjection) => projection.solver === "chebyshev"
  ? "Chebyshev-Jacobi · row parallel"
  : ({ ic0: "ICCG(0)", blockic: "CG + block ICCG(0)", jacobi: "CG + diagonal Jacobi", line: "CG + vertical line Jacobi", poly: "CG + polynomial Jacobi", mg: "CG + geometric multigrid" })[projection.preconditioner];
const quadtreePressureDescription = (projection: WebGPUQuadtreeTallCellProjection, pressureIterations: number, tolerance: number) => projection.solver === "chebyshev"
  ? `${quadtreePressureLabel(projection)} · ${projection.info.pressureIterationBudget ?? pressureIterations} fixed passes · spectrum [${quadtreeChebyshevSpectrum.lower}, ${quadtreeChebyshevSpectrum.upper}] · experimental`
  : `${quadtreePressureLabel(projection)} · ${projection.info.pressureIterationBudget ?? pressureIterations} encoded / ${projection.info.pressureIterationHardBudget ?? pressureIterations} hard · relative ${tolerance}`;

/** Shared dense Eulerian host used by the uniform and adaptive product methods. */
export class WebGPUUniformEulerianSolver {
  readonly info: GPUEulerianInfo;
  private velocityA!: GPUTexture; private velocityB!: GPUTexture;
  private velocityC!: GPUTexture; private velocityD!: GPUTexture;
  private pressureA!: GPUTexture; private pressureB!: GPUTexture;
  private volumeA!: GPUTexture; private volumeB!: GPUTexture;
  private heightA!: GPUTexture; private heightB!: GPUTexture; private terrainTexture: GPUTexture;
  private transportA!: GPUTexture; private transportB!: GPUTexture; private fluxScales!: GPUTexture;
  private transportSampler!: GPUSampler;
  private params?: GPUBuffer; private reductionBuffer?: GPUBuffer; private sharpenBuffer?: GPUBuffer;
  private rigidBuffer: GPUBuffer; private rigidExchangeBuffer: GPUBuffer;
  private rigidSystem: WebGPURigidBodySystem;
  private statsReadbackBuffer?: GPUBuffer;
  private bindGroupLayout!: GPUBindGroupLayout;
  private advectPipeline!: GPUComputePipeline; private reversePipeline!: GPUComputePipeline;
  private correctPipeline!: GPUComputePipeline; private jacobiPipeline!: GPUComputePipeline;
  private projectPipeline!: GPUComputePipeline; private rigidPipeline!: GPUComputePipeline; private relaxSolidPhiPipeline!: GPUComputePipeline;
  private reductionPipeline!: GPUComputePipeline;
  private buildTransportPipeline!: GPUComputePipeline; private buildFluxScalesPipeline!: GPUComputePipeline;
  private buildOccupancyPipeline!: GPUComputePipeline;
  private sharpenComputePipeline!: GPUComputePipeline; private sharpenScatterPipeline!: GPUComputePipeline; private sharpenResolvePipeline!: GPUComputePipeline;
  private shaderModule!:GPUShaderModule;private pipelineLayout!:GPUPipelineLayout;private prepPipelineLayout!:GPUPipelineLayout;
  private advectGroup!: GPUBindGroup; private reverseGroup!: GPUBindGroup; private correctGroup!: GPUBindGroup;
  private jacobiABGroup!: GPUBindGroup;
  private jacobiBAGroup!: GPUBindGroup; private projectGroup!: GPUBindGroup;
  private rigidGroup!: GPUBindGroup; private reductionGroup!: GPUBindGroup; private solidPhiGroup?: GPUBindGroup;
  private occupancyGroup!: GPUBindGroup; private transportFromCurrentGroup!: GPUBindGroup; private fluxScaleGroup!: GPUBindGroup;
  private sharpenComputeGroup!: GPUBindGroup; private sharpenScatterGroup!: GPUBindGroup; private sharpenResolveGroup!: GPUBindGroup;
  private transportFromPredictedGroup?: GPUBindGroup;
  private lastTime = 0;
  private readbackPending = false;
  private physicsTraceSampleId = 0;
  private physicsTracePending = false;
  private lastPhysicsTraceAt_ms = -Infinity;
  private lastSegmentedPhysicsTraceAt_ms = -Infinity;
  private hardwarePhysicsTraceInvalid = false;
  private profiledAdvanceCompletion?: Promise<void>;
  private validationChecked = false;
  private validationPromise?: Promise<void>;
  // Not readonly: a warm re-seed rebuilds the boundary for the new nozzle.
  private inflowBoundary?: InflowGridBoundary;
  private readonly velocityTransport: UniformVelocityTransport;
  private readonly densitySharpening: boolean;
  private readonly hostAllocation?: UniformHostAllocationPlan;
  private readonly transportConservativeVolume: boolean;
  private quadtreeProjection?: WebGPUQuadtreeTallCellProjection;
  private octreeProjection?: WebGPUOctreeProjection;
  private readonly retiredQuadtreeProjections = new Set<WebGPUQuadtreeTallCellProjection>();
  private quadtreeRebuildPending = false;
  private quadtreeReadyProjection?: WebGPUQuadtreeTallCellProjection;
  private quadtreeRebuildBlockedFrames = 0;
  private quadtreeBlockedSince_ms?: number;
  private quadtreeRebuildFallbackWarned = false;
  private quadtreeRebuildCompletedCount = 0;
  private quadtreeRebuildFailureCount = 0;
  private quadtreeRebuildRetrySteps = 0;
  private readonly rebuildQuadtreeEachStep: boolean;
  private quadtreeStepsSinceTopology = 0;
  private quadtreeStepsSinceKick = 0;
  private quadtreeLastBodies: RigidBodyState[] = [];
  private readonly quadtreeRebuildInterval: number;
  private readonly quadtreeTopologyStaleLimit: number;
  private readonly quadtreeInlineRebuild: boolean;
  private disposed = false;
  private initialSparseAuthorityPublished = false;
  private baseAllocatedBytes = 0;

  constructor(
    private device: GPUDevice,
    // Not readonly: `applySceneUniforms` swaps in scalar-only scene revisions.
    public scene: SceneDescription,
    quality: GPUQuality,
    private onRigidLoads?: (loads: GPURigidLoad[]) => void,
    options: WebGPUUniformEulerianOptions = {}
  ) {
    // The octree has one measured executable profile. Reject it before the
    // first texture or buffer allocation so unsupported adapters cannot leave
    // a partially constructed graph or trigger compilation of another lane.
    if (options.octree && (!device.features.has("subgroups")
      || !supportsFluidM1MaxReduction(device.limits))) {
      throw new Error("Power octree requires the M1 Max 128-lane subgroup profile");
    }
    const c = scene.container, matched = createTallCellLayout(scene, quality, device.limits.maxTextureDimension3D, options.tallCellSettings);
    const nx = matched.nx, ny = matched.fineNy, nz = matched.nz;
    this.velocityTransport = options.velocityTransport ?? "maccormack";
    this.densitySharpening = options.densitySharpening ?? true;
    this.hostAllocation = options.octree
      ? undefined
      : planUniformHostAllocation(nx, ny, nz, this.velocityTransport);
    this.rebuildQuadtreeEachStep = options.quadtreeRebuildTopology ?? true;
    // Narita et al. Algorithm 1 evaluates and subdivides the quadtree on every
    // Advance_Step. A caller may still request a slower experimental cadence,
    // but the paper-faithful default is one rebuild per simulation step.
    this.quadtreeRebuildInterval = Math.max(1, Math.round(options.quadtreeRebuildIntervalSteps ?? 1));
    // W6 acceptance pipelines one cadence-1 rebuild across at most two steps;
    // zero staleness remains Algorithm 1's stretch goal once the complete pack
    // stays resident and no readback/upload handshake remains.
    this.quadtreeTopologyStaleLimit = Math.max(0, Math.round(options.quadtreeTopologyStaleSteps ?? 2));
    this.quadtreeInlineRebuild = options.quadtreeInlineRebuild ?? true;
    this.inflowBoundary=scene.fluid.inflow?createInflowGridBoundary(scene.fluid.inflow,scene.container,[nx,ny,nz]):undefined;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const scalarTexture = (format: GPUTextureFormat, extent: readonly [number, number, number]) => device.createTexture({ size: extent, dimension: "3d", format, usage });
    if (this.hostAllocation) {
      const velocityTexture = () => device.createTexture({
        size: this.hostAllocation!.velocityExtent,
        dimension: "3d",
        format: "rgba32float",
        usage,
      });
      this.velocityA = velocityTexture();
      this.velocityB = velocityTexture();
      this.velocityC = this.velocityTransport === "maccormack" ? velocityTexture() : this.velocityA;
      this.velocityD = this.velocityTransport === "maccormack" ? velocityTexture() : this.velocityB;
      this.pressureA = scalarTexture("r32float", this.hostAllocation.pressureExtent);
      this.pressureB = scalarTexture("r32float", this.hostAllocation.pressureExtent);
      this.volumeA = scalarTexture("r32float", this.hostAllocation.volumeExtent);
      this.volumeB = scalarTexture("r32float", this.hostAllocation.volumeExtent);
    }
    // x retains the historical highest-occupied-cell index used for culling.
    // y carries the octree-only, bottom-connected sub-cell surface eta. Keeping
    // them distinct prevents detached spray from becoming a hydrostatic column.
    if (this.hostAllocation) {
      this.heightA = device.createTexture({
        label: "Uniform column fallback A",
        size: [nx, nz],
        format: "rg32float",
        usage,
      });
      this.heightB = device.createTexture({
        label: "Uniform column occupancy and hydrostatic reference",
        size: [nx, nz],
        format: "rg32float",
        usage,
      });
    }
    this.terrainTexture = device.createTexture({ label: "Uniform terrain heights", size: [nx, nz], format: "r32float", usage });
    // Filterable fp16 transport fields, padded with a zero shell so hardware
    // clamp-to-edge sampling still reads zero at solid wall faces.
    if (this.hostAllocation) {
      const transportTexture = (label: string) => device.createTexture({ label, size: this.hostAllocation!.transportExtent, dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
      this.transportA = transportTexture("Uniform transport velocity A");
      this.transportB = this.velocityTransport === "maccormack" ? transportTexture("Uniform transport velocity B") : this.transportA;
      this.fluxScales = device.createTexture({ label: "Uniform volume flux scales", size: this.hostAllocation.fluxExtent, dimension: "3d", format: "rg32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
      this.transportSampler = device.createSampler({ minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    }
    if (this.hostAllocation) this.params = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    if (this.hostAllocation) {
      this.reductionBuffer = device.createBuffer({
        label: "Uniform diagnostics reduction",
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.sharpenBuffer = device.createBuffer({ label: "Uniform sharpening deposits", size: this.hostAllocation.conditioningBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    this.rigidExchangeBuffer = device.createBuffer({ size: GPU_RIGID_EXCHANGE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.rigidSystem = new WebGPURigidBodySystem(device, scene, this.rigidExchangeBuffer, this.terrainTexture);
    this.rigidBuffer = this.rigidSystem.stateBuffer;
    this.rigidSystem.syncBodies(initializeRigidBodies(scene.rigidBodies));
    let prepLayout: GPUBindGroupLayout | undefined;
    if (this.hostAllocation) {
    this.shaderModule = device.createShaderModule({ label: "Fluid Lab uniform reference kernels", code: legacyUniformComputeShader });
    this.bindGroupLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "2d" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
      { binding: 15, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 17, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 19, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 20, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 21, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } }
    ] });
    // The main layout already carries four storage textures (the per-stage
    // limit), so the transport/flux-scale writers get their own layout.
    prepLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
      { binding: 18, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "3d" } },
      { binding: 20, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    ] });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    this.prepPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [prepLayout] });
    if(!options.deferPipelineCompilation)this.createPipelinesSync();
    }
    const pressureIterations = options.octree
      ? 0
      : Math.max(8, Math.min(400, Math.round(options.pressureIterations ?? (quality === "balanced" ? 64 : quality === "high" ? 80 : 96))));
    const count = nx * ny * nz;
    this.info = {
      nx, ny, nz, storedNy: ny, cellCount: count, equivalentUniformCells: count,
      compressionRatio: 1, activeCompressionRatio: 1, activeSampleCount: count, regularLayers: ny, maximumNeighborDelta: 0,
      gridKind: "uniform", cellSize_m: Math.max(c.width_m / nx, c.height_m / ny, c.depth_m / nz),
      pressureIterations, allocatedBytes: this.hostAllocation?.allocatedBytes ?? 0, quality, encodedSteps: 0, maximumTallCellHeight: 0,
      submittedTime_s: 0, simulatedTime_s: 0, completedTime_s: 0, simulationLag_s: 0
    };
    this.baseAllocatedBytes = this.info.allocatedBytes;
    this.initializeVolume();
    if (options.quadtreeTallCells) {
      // Dynamic bodies are consumed from the resident storage buffer by the
      // dense immersed-boundary pass. Only immutable bodies enter the CPU-built
      // variational topology; otherwise a moving GPU body would leave behind a
      // stale host-authored K matrix and force a pose readback to rebuild it.
      const initialCouplingBodies = scene.rigidBodies.filter((body) => body.motion === "static").map((body) => initializeRigidBodies([body])[0]);
      this.quadtreeProjection = new WebGPUQuadtreeTallCellProjection(device, scene, { nx, ny, nz }, { velocityIn: this.velocityB, velocityOut: this.velocityA, velocityScratch: this.velocityD, volume: this.volumeA }, {
        pressureIterations,
        relativeTolerance: scene.numerics.pressureRelativeTolerance,
        adaptivityStrength: options.quadtreeTallCells.adaptivityStrength ?? 1,
        maximumLeafSize: options.quadtreeTallCells.maximumLeafSize ?? 16,
        opticalDepthFraction: options.quadtreeTallCells.opticalDepthFraction ?? 0.25,
        ...options.quadtreeTallCells
      }, undefined, initialCouplingBodies.length > 0 ? { bodies: initialCouplingBodies, dynamic: false } : undefined,options.deferPipelineCompilation);
      this.applyQuadtreeInfo(this.quadtreeProjection, pressureIterations);
    } else if (options.octree) {
      this.octreeProjection = new WebGPUOctreeProjection(device, scene, { nx, ny, nz }, {
        rigidBodies: this.rigidBuffer, rigidExchange: this.rigidExchangeBuffer, terrain: this.terrainTexture,
      }, {
        maximumLeafSize: options.octree.maximumLeafSize ?? 16,
        adaptivity: options.octree.adaptivity ?? 1,
        interfaceRefinementBandCells: options.octree.interfaceRefinementBandCells ?? 4,
        globalFineLevelSetFactor: options.octree.globalFineLevelSetFactor ?? 4,
        globalFineLevelSetMaximumBricks: options.octree.globalFineLevelSetMaximumBricks,
        pressureRowCapacity: options.octree.pressureRowCapacity,
      }, options.deferPipelineCompilation);
      this.applyOctreeInfo(this.octreeProjection);
    }
    // The octree's resident level set is the complete liquid state. Keep VOF
    // transport only for the uniform solver and quadtree catastrophe recovery.
    this.transportConservativeVolume = !this.octreeProjection;
    // The compact octree owns a separate pipeline graph. Shared dense bind
    // groups exist only for the uniform and quadtree methods.
    if (this.hostAllocation) {
    const surfaceAuthority = this.adaptiveProjection?.levelSetTexture ?? this.volumeA;
    const prepGroup = (velocity: GPUTexture, transport: GPUTexture) => device.createBindGroup({ layout: prepLayout!, entries: [
      { binding: 0, resource: velocity.createView() }, { binding: 4, resource: this.volumeA.createView() },
      { binding: 6, resource: { buffer: this.params! } }, { binding: 16, resource: transport.createView() },
      { binding: 18, resource: this.fluxScales.createView() }, { binding: 20, resource: surfaceAuthority.createView() },
    ] });
    this.transportFromCurrentGroup = prepGroup(this.velocityA, this.transportA);
    if (this.velocityTransport === "maccormack") this.transportFromPredictedGroup = prepGroup(this.velocityC, this.transportB);
    this.fluxScaleGroup = prepGroup(this.velocityA, this.transportA);
    // Advection groups read the column occupancy from heightB; heightA stays
    // zero for the renderer's uniform column-base fallback.
    this.occupancyGroup = this.group(this.velocityA, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightA, this.heightB, this.velocityA, this.velocityA, this.transportA, surfaceAuthority);
    this.advectGroup = this.velocityTransport === "maccormack"
      ? this.group(this.velocityA, this.velocityC, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityB, this.velocityD, this.transportA, surfaceAuthority)
      : this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityA, this.transportA, surfaceAuthority);
    this.reverseGroup = this.velocityTransport === "maccormack" ? this.group(this.velocityC, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityB, this.transportB, surfaceAuthority) : this.advectGroup;
    // The uniform path samples its current VOF field as the liquid authority.
    // Its correction output is volumeA, so sampling volumeA in the same
    // dispatch would alias one texture as both sampled and writable.
    const correctionSurfaceAuthority = this.adaptiveProjection?.levelSetTexture ?? this.volumeB;
    this.correctGroup = this.velocityTransport === "maccormack" ? this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA, this.velocityC, this.velocityD, this.transportA, correctionSurfaceAuthority) : this.advectGroup;
    this.jacobiABGroup = this.group(this.velocityB, this.velocityA, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.jacobiBAGroup = this.group(this.velocityB, this.velocityA, this.pressureB, this.pressureA, this.volumeB, this.volumeA, this.heightB, this.heightA);
    const finalPressure = pressureIterations % 2 === 0 ? this.pressureA : this.pressureB;
    const sparePressure = pressureIterations % 2 === 0 ? this.pressureB : this.pressureA;
    this.projectGroup = this.group(this.velocityB, this.velocityA, finalPressure, sparePressure, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.rigidGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.sharpenComputeGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.sharpenScatterGroup = this.group(this.velocityA, this.velocityB, this.pressureB, this.pressureA, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.sharpenResolveGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.reductionGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    // Paper Sec 3.9.1 phi-s over the quadtree resident level set: the pass
    // aliases the idle uniform pressure slots (pressureIn = pre-pass copy in
    // pressureA, pressureOut = the level-set texture itself). The velocity and
    // volume outputs are bound but never written by relaxSolidPhi.
    if (this.quadtreeProjection) this.solidPhiGroup = this.group(this.velocityA, this.velocityD, this.pressureA, this.quadtreeProjection.levelSetTexture, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityA, this.transportA, this.volumeA);
    }
    if (this.octreeProjection && !options.deferPipelineCompilation) this.publishInitialSparseScene();
  }

  private pipelineDescriptor(entryPoint:string,prep=false):GPUComputePipelineDescriptor{return{layout:prep?this.prepPipelineLayout:this.pipelineLayout,compute:{module:this.shaderModule,entryPoint}};}
  private assignPipelines(compiled:GPUComputePipeline[]){this.advectPipeline=compiled[0];this.reversePipeline=compiled[1];this.correctPipeline=compiled[2];this.jacobiPipeline=compiled[3];this.projectPipeline=compiled[4];this.rigidPipeline=compiled[5];this.relaxSolidPhiPipeline=compiled[6];this.reductionPipeline=compiled[7];this.buildOccupancyPipeline=compiled[8];this.buildTransportPipeline=compiled[9];this.buildFluxScalesPipeline=compiled[10];this.sharpenComputePipeline=compiled[11];this.sharpenScatterPipeline=compiled[12];this.sharpenResolvePipeline=compiled[13];}
  private createPipelinesSync(){const pipeline=(entryPoint:string,prep=false)=>this.device.createComputePipeline(this.pipelineDescriptor(entryPoint,prep));const compiled=[pipeline(this.velocityTransport==="maccormack"?"advect":"semiLagrangianAdvection"),pipeline("reverseAdvection"),pipeline("correctAdvection"),pipeline("jacobi"),pipeline("project"),pipeline("coupleRigid"),pipeline("relaxSolidPhi"),pipeline("reduceDiagnostics"),pipeline("buildOccupancy"),pipeline("buildTransport",true),pipeline("buildFluxScales",true),pipeline("sharpenCompute"),pipeline("sharpenScatter"),pipeline("sharpenResolve")];this.assignPipelines(compiled);let cache=uniformPipelineCache.get(this.device);if(!cache){cache=new Map();uniformPipelineCache.set(this.device,cache);}cache.set(this.velocityTransport,compiled);}
  static async createAsync(device:GPUDevice,scene:SceneDescription,quality:GPUQuality,onRigidLoads:((loads:GPURigidLoad[])=>void)|undefined,options:WebGPUUniformEulerianOptions,onProgress:(label:string,completed:number,total:number,phase?:string,taskId?:string)=>void,signal:AbortSignal=new AbortController().signal){
    const runner=new GPUInitializationTaskRunner((snapshot)=>onProgress(snapshot.label,snapshot.completed,snapshot.total,snapshot.phase,snapshot.taskId),signal);
    let solver:WebGPUUniformEulerianSolver|undefined;
    try{
      await runner.run([{id:"solver.allocate",phase:"allocation",label:options.octree||options.quadtreeTallCells?"Allocate adaptive solver resources":"Allocate uniform solver resources",run:()=>{solver=new WebGPUUniformEulerianSolver(device,scene,quality,onRigidLoads,{...options,deferPipelineCompilation:true});}}]);
      await runner.run(solver!.initializationTasks());
      return solver!;
    }catch(error){solver?.destroy();throw error;}
  }
  private initializationTasks():GPUInitializationTask[]{
    const tasks:GPUInitializationTask[]=[];
    if (this.hostAllocation) {
      const cached=uniformPipelineCache.get(this.device)?.get(this.velocityTransport);
      if(cached)tasks.push({id:"uniform.pipeline-cache",phase:"solver-pipelines",label:"Reuse compiled simulation programs",run:()=>this.assignPipelines(cached)});
      const definitions=[
        ["Advect velocity",this.velocityTransport==="maccormack"?"advect":"semiLagrangianAdvection",false],["Reverse advection","reverseAdvection",false],
        ["Correct advection","correctAdvection",false],["Relax pressure","jacobi",false],["Project velocity","project",false],
        ["Couple rigid bodies","coupleRigid",false],["Relax solid level set","relaxSolidPhi",false],["Reduce diagnostics","reduceDiagnostics",false],["Build occupancy","buildOccupancy",false],
        ["Build transport field","buildTransport",true],["Build flux scales","buildFluxScales",true],
        ["Sharpen density","sharpenCompute",false],["Scatter sharpened mass","sharpenScatter",false],["Resolve sharpened mass","sharpenResolve",false]
      ] as const,compiled:GPUComputePipeline[]=new Array(definitions.length);
      if(!cached)definitions.forEach(([label,entryPoint,prep],index)=>tasks.push({id:`uniform.pipeline.${entryPoint}`,phase:"solver-pipelines",label,run:async()=>{compiled[index]=await this.device.createComputePipelineAsync(this.pipelineDescriptor(entryPoint,prep));if(index===definitions.length-1){this.assignPipelines(compiled);let cache=uniformPipelineCache.get(this.device);if(!cache){cache=new Map();uniformPipelineCache.set(this.device,cache);}cache.set(this.velocityTransport,compiled);}}}));
    }
    if(this.quadtreeProjection)tasks.push({id:"quadtree.pipeline-set",phase:"adaptive-topology",label:"Compile adaptive pressure pipeline set",run:()=>this.quadtreeProjection!.initializePipelines(()=>{})});
    else if(this.octreeProjection)tasks.push(...this.octreeProjection.initializationTasks());
    if (this.octreeProjection) {
      let previousTaskId: string | undefined;
      OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES.forEach((authorityPhase, index) => {
        const id = index === 0 ? "solver.warmup" : `solver.warmup.${authorityPhase.id}`;
        tasks.push({ id, phase: "warmup",
          label: index === 0 ? `Publish and warm initial sparse scene: ${authorityPhase.label}` : authorityPhase.label,
          ...(previousTaskId ? { dependencies: [previousTaskId] } : {}),
          run: () => this.publishInitialSparseScenePhase(authorityPhase.id) });
        previousTaskId = id;
      });
    } else {
      tasks.push({id:"solver.warmup",phase:"warmup",label:"Finish initial GPU uploads",run:()=>this.publishInitialSparseScene()});
    }
    return tasks;
  }

  private async publishInitialSparseScenePhase(
    phase: typeof OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES[number]["id"],
  ) {
    if (!this.octreeProjection) throw new Error("Sparse authority phase requires an octree projection");
    if (phase === "cold-topology") this.initialSparseAuthorityPublished = false;
    const descriptor = OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES.find((candidate) => candidate.id === phase)!;
    this.device.pushErrorScope("validation");
    let validationScopeOpen = true;
    const initialSparseScene = this.device.createCommandEncoder({
      label: `Initial sparse authority: ${descriptor.label}`,
    });
    try {
      this.octreeProjection.encodeInitialSparseAuthorityPhase(initialSparseScene, phase);
      this.device.queue.submit([initialSparseScene.finish()]);
      this.octreeProjection.retireSubmittedEncoder(initialSparseScene);
      await this.device.queue.onSubmittedWorkDone();
      const validationError = await this.device.popErrorScope();
      validationScopeOpen = false;
      if (validationError) {
        throw new Error(`Initial sparse authority ${phase} validation failed: ${validationError.message}`);
      }
    } catch (error) {
      if (validationScopeOpen) {
        const validationError = await this.device.popErrorScope().catch(() => null);
        validationScopeOpen = false;
        if (validationError) {
          throw new Error(`Initial sparse authority ${phase} validation failed: ${validationError.message}`);
        }
      }
      throw error;
    }
    if (phase === "cold-topology") {
      const failure = await this.octreeProjection.readPowerFrontierFailure();
      const frontier = failure.frontier;
      const selector = frontier[2] ?? 0xffff_ffff;
      const count = selector < 2 ? frontier[selector] ?? 0 : 0;
      if (selector >= 2 || (frontier[3] ?? 0) === 0 || count === 0) {
        throw new Error(`Initial sparse authority cold-topology published no liquid-row frontier: ${JSON.stringify(failure)}`);
      }
      this.octreeProjection.finishTopologyCandidate();
    }
    if (phase === "sparse-render-world") {
      await this.validateInitialSparseAuthority();
      // This assignment is deliberately inside the final phase task, after
      // its fence and the bounded authority readbacks: an encoded, submitted,
      // or merely host-tagged render world is not ready.
      this.initialSparseAuthorityPublished = true;
      this.applyOctreeInfo(this.octreeProjection);
    }
  }

  private applyGlobalFineDiagnostics(value: InitialGlobalFineAuthorityDiagnostics) {
    const topology = unpackFineLevelSetGPUTopologyControl(value.topologyControl);
    const coarse = unpackOctreePowerCoarseLevelSetControl(value.coarseControl);
    this.info.globalFineSeedCount = value.seedControl[0] ?? 0;
    this.info.globalFineSeedError = value.seedControl[1] ?? 0;
    this.info.globalFineTopologyFlags = topology.flags;
    this.info.globalFineDownstreamFinalizeReason = topology.downstreamFinalizeReason;
    this.info.globalFineInterfaceBricks = topology.interfaceBricks;
    this.info.globalFineDesiredBricks = topology.desiredBricks;
    this.info.globalFineActivatedBricks = topology.activatedBricks;
    this.info.globalFinePublished = topology.published;
    this.info.globalFineRolledBack = topology.rolledBack;
    // Word one is the active count; word zero is the generation. Reading word
    // zero here reported the generation as a brick count, which is why fine
    // band occupancy never appeared in a benchmark.
    this.info.globalFineActiveBricks =
      readFineLevelSetWorksetHeader(value.worklistHeader)?.activeCount ?? 0;
    this.info.globalFineGeneration = value.configuredFineGeneration;
    this.info.globalFineCoarseLevelSetFlags = coarse.flags;
    this.info.globalFineCoarseLevelSetFirstErrorRow = coarse.firstErrorRow;
    const velocity = value.structuredVelocityControl;
    const boundary = value.structuredBoundaryControl;
    this.info.structuredVelocityValid = velocity.length >= 6 && velocity[0] === 0
      && (velocity[2] ?? 0) > 0 && (velocity[3] ?? 0) > 0 && (velocity[4] ?? 2) <= 1;
    this.info.structuredVelocityRows = velocity[2] ?? 0;
    this.info.structuredVelocityGeneration = velocity[3] ?? 0;
    this.info.structuredVelocitySlots = velocity[5] ?? 0;
    // Word 1 is the GPU reject carry and was previously read back and dropped.
    // Naming the stage here is what separates "a face was rejected in the
    // divergence gather" from the silent freeze it otherwise becomes.
    const reject = decodeOctreeStructuredRejectCarry(velocity);
    this.info.structuredRejectStage = reject.stage;
    this.info.structuredRejectIndex = reject.index;
    this.info.structuredRejectSummary = reject.clean ? undefined : reject.summary;
    this.info.structuredBoundaryGeneration = boundary[4] ?? 0;
    this.info.structuredBoundaryValid = boundary.length >= 7 && boundary[0] === 0
      && boundary[2] === velocity[2] && boundary[4] === velocity[3]
      && boundary[5] === velocity[4] && boundary[6] === velocity[3];
    // Accepted structured control is the GPU-owned coupled-epoch receipt.
    // Never relabel the host's newer attempt stamp as a published generation.
    this.info.powerDiagramGeneration = this.info.structuredVelocityValid
      && this.info.structuredBoundaryValid ? velocity[3] : undefined;
  }

  /** The paper path must be complete before the first trajectory can be
   * requested. These are one-time post-fence readbacks for UI readiness and
   * diagnostics; recurring frame scheduling remains GPU-resident. */
  private async validateInitialSparseAuthority() {
    const projection = this.octreeProjection;
    if (!projection) throw new Error("Initial sparse authority requires an octree projection");
    const [, fine, mgpcg] = await Promise.all([
      projection.readSolveDiagnostics(), projection.readGlobalFineLevelSetDiagnostics(),
      projection.readMGPCGDiagnostics(),
    ]);
    this.applyOctreeInfo(projection);
    const velocity = fine?.structuredVelocityControl ?? [];
    const boundary = fine?.structuredBoundaryControl ?? [];
    const structuredReady = velocity.length >= 6 && velocity[0] === 0 && velocity[2] > 0
      && velocity[3] !== 0 && velocity[4] <= 1;
    const boundaryReady = boundary.length >= 7 && boundary[0] === 0
      && boundary[2] === velocity[2] && boundary[4] === velocity[3]
      && boundary[5] === velocity[4] && boundary[6] === velocity[3];
    if (!structuredReady || !boundaryReady) {
      const packedStructuredFailure = Number(velocity[1] ?? 0xffff_ffff) >>> 0;
      const structuredFailureStage = packedStructuredFailure >>> 24;
      const structuredFailureIndex = packedStructuredFailure & 0x00ff_ffff;
      const structuredFailureRow = (structuredFailureStage >= 20 && structuredFailureStage < 24)
        || (structuredFailureStage >= 40 && structuredFailureStage < 60)
        ? await projection.readPowerCoarseFailureRow(structuredFailureIndex) : undefined;
      const frontier = await projection.readPowerFrontierFailure();
      const owner = await projection.readOwnerPageControl();
      throw new Error("Paused t=0 structured authority rejected: velocity=" + JSON.stringify(velocity)
        + "; boundary=" + JSON.stringify(boundary) + "; coarse=" + JSON.stringify(fine?.coarseControl ?? [])
        + "; restriction=" + JSON.stringify(fine?.fineRestrictionControl ?? [])
        + "; seedAdapter=" + JSON.stringify(fine?.fineSeedAdapterControl ?? [])
        + "; firstSeed=" + JSON.stringify(fine?.firstFineSeedLeaf ?? [])
        + "; firstVelocityA=" + JSON.stringify(fine?.firstStructuredVelocityA ?? [])
        + "; firstVelocityB=" + JSON.stringify(fine?.firstStructuredVelocityB ?? [])
        + "; firstCoarse=" + JSON.stringify(fine?.firstCoarsePhi ?? [])
        + "; compactPrefix=" + JSON.stringify(fine?.compactRowPrefix ?? [])
        + "; firstApertureAB=" + JSON.stringify(fine?.firstStructuredApertureAB ?? [])
        + "; firstSolidNormalVelocityAB=" + JSON.stringify(fine?.firstStructuredSolidNormalVelocityAB ?? [])
        + "; structuredFailureRow=" + JSON.stringify(structuredFailureRow)
        + "; mgpcg=" + JSON.stringify(mgpcg ? Array.from(mgpcg) : [])
        + "; frontier=" + JSON.stringify(frontier)
        + "; owner=" + JSON.stringify(owner));
    }
    // The same queue-fenced controls that admit t=0 are also the authoritative
    // diagnostics receipt. Dynamic readStats() applies this record after each
    // encoded step, but it deliberately does not run at encodedSteps === 0.
    // Publish it here so a reset cannot retain the previous step's rejection
    // (or leave the fields undefined) after this preflight has succeeded.
    this.applyGlobalFineDiagnostics(fine!);
    const pressure = initialPowerPressureReadiness({ authoritative: projection.info.powerDiagramAuthoritative,
      solverLabel: projection.pressureSolverLabel, pressureRows: projection.info.pressureRequiredRows ?? 0,
      capacityOverflow: projection.info.pressureCapacityOverflow ?? false, mgpcgControl: mgpcg });
    if (!pressure.ready) {
      const frontier = await projection.readPowerFrontierFailure();
      throw new Error("Paused t=0 authority rejected: " + pressure.label
        + "; mgpcg=" + JSON.stringify(mgpcg ? Array.from(mgpcg) : [])
        + "; frontier=" + JSON.stringify(frontier));
    }
    const floats = new Float32Array(mgpcg!.buffer, mgpcg!.byteOffset, mgpcg!.length);
    this.info.quadtreePressureConverged = true; this.info.quadtreePressureIterationsUsed = mgpcg![2];
    const residualSquared = floats[10] + floats[11], rhsSquared = floats[8] + floats[9];
    this.info.pressureResidual = Math.sqrt(Math.max(0, residualSquared));
    this.info.pressureRelativeResidual = Math.sqrt(residualSquared / Math.max(rhsSquared, 1e-30));
  }

  /** Publish a complete t=0 scene after rigid-solid raster pipelines exist. */
  private async publishInitialSparseScene() {
    this.initialSparseAuthorityPublished = false;
    if (this.octreeProjection) {
      for (const phase of OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES) {
        await this.publishInitialSparseScenePhase(phase.id);
      }
    } else {
      // This fence covers constructor-time texture uploads for non-octree
      // solvers. Octree startup already fenced every ordered phase above.
      await this.device.queue.onSubmittedWorkDone();
      this.initialSparseAuthorityPublished = true;
    }
  }

  get volumeTexture() { return this.octreeProjection?.levelSetTexture ?? this.volumeA; }
  get pendingAdvanceCompletion() { return this.profiledAdvanceCompletion; }
  get rigidRenderBuffer() { return this.rigidSystem.renderBuffer; }
  get rigidMotionBuffer() { return this.rigidSystem.motionBuffer; }
  setSelectedRigidBody(index: number) { this.rigidSystem.setSelectedIndex(index); }
  pickRigidBody(origin: RigidBodyState["position_m"], direction: RigidBodyState["position_m"]) { return this.rigidSystem.pick(origin,direction); }
  // Rendering contours the smooth resident level set when the quadtree
  // projection maintains one; the flux-form VOF field is near-binary and its
  // 0.5 contour is quantized to cell scale. Diagnostics keep reading the VOF
  // field through volumeTexture.
  private get adaptiveProjection() { return this.quadtreeProjection ?? this.octreeProjection; }
  get surfaceFieldTexture() { return this.adaptiveProjection?.levelSetTexture ?? this.volumeA; }
  /** False once global-fine publication has retired the dense bootstrap phi. */
  get hasDenseSurfaceField() { return this.octreeProjection?.hasDenseLevelSetPublication ?? true; }
  get sparseVoxelSceneSource() { return this.octreeProjection?.sparseVoxelSceneSource; }
  get sparseVoxelRenderSource() {
    const source = this.octreeProjection?.sparseVoxelRenderSource;
    if (this.octreeProjection) this.applyOctreeInfo(this.octreeProjection);
    return source;
  }
  get structuredVelocityControl() { return this.octreeProjection?.structuredVelocityControl; }
  get structuredBoundaryControl() { return this.octreeProjection?.structuredBoundaryControl; }
  get structuredRowVelocities() { return this.octreeProjection?.structuredRowVelocities; }
  get structuredAuthority() { return this.octreeProjection?.structuredAuthority; }
  get structuredWorksets() { return this.octreeProjection?.structuredWorksets; }
  /** Post-submit diagnostics only; never consumed by the simulation schedule. */
  get workAccounting() { return this.octreeProjection?.workAccounting; }
  get workAccountingBuffers() { return this.octreeProjection?.workAccountingBuffers; }
  get workAccountingPlan() { return this.octreeProjection?.workAccountingPlan; }
  captureWorkAccounting() { return this.octreeProjection?.captureWorkAccounting(); }
  /** QA-only passthrough for the authoritative Section 4.3 solver status. */
  get mgpcgControl() { return this.octreeProjection?.mgpcgControl; }
  get ownerLatticeDebug() { return this.octreeProjection?.ownerLatticeDebug; }
  get powerBoundaryFineSource() { return this.octreeProjection?.powerBoundaryFineSource; }
  get powerBoundaryFineLevelSetSource() { return this.octreeProjection?.powerBoundaryFineLevelSetSource; }
  get powerDescriptorControl() { return this.octreeProjection?.powerDescriptorControl; }
  get powerTopologyControl() { return this.octreeProjection?.powerTopologyControl; }
  get powerDescriptorRows() { return this.octreeProjection?.powerDescriptorRows; }
  get powerTopologyMetrics() { return this.octreeProjection?.powerTopologyMetrics; }
  get powerCatalogEntryHeaders() { return this.octreeProjection?.powerCatalogEntryHeaders; }
  get powerCatalogFaces() { return this.octreeProjection?.powerCatalogFaces; }
  get powerLeafHeaders() { return this.octreeProjection?.powerLeafHeaders; }
  get powerPressureBuffer() { return this.octreeProjection?.powerPressureBuffer; }
  get powerLeafFrontier() { return this.octreeProjection?.powerLeafFrontier; }
  get powerCompactionControl() { return this.octreeProjection?.powerCompactionControl; }
  get powerTopologyTileChangeFlags() { return this.octreeProjection?.powerTopologyTileChangeFlags; }
  get powerTopologyTileStates() { return this.octreeProjection?.powerTopologyTileStates; }
  get powerFrontierCarryFlags() { return this.octreeProjection?.powerFrontierCarryFlags; }
  get powerRowDelta() { return this.octreeProjection?.powerRowDelta; }
  get topologyTileWorklist() { return this.octreeProjection?.topologyTileWorklist; }
  get fineSeedCandidateControl() { return this.octreeProjection?.fineSeedCandidateControl; }
  get fineSeedLeaves() { return this.octreeProjection?.fineSeedLeaves; }
  get powerOwnerArena() { return this.octreeProjection?.powerOwnerArena; }
  get octreeTechniqueDebugSource() { return this.octreeProjection?.techniqueDebugSource; }
  get initialSparseAuthorityReady() { return !this.octreeProjection || this.initialSparseAuthorityPublished; }
  get globalFineLevelSetSource() { return this.octreeProjection?.globalFineLevelSetSource; }
  /** QA-only passthrough for reproducing recurring frontier phase decisions. */
  get globalFineSummaryDirectory() { return this.octreeProjection?.globalFineSummaryDirectory; }
  get globalFineSummaryDebug() { return this.octreeProjection?.globalFineSummaryDebug; }
  get globalFineTransportControl() { return this.octreeProjection?.globalFineTransportControl; }
  get globalFineTransportDeltaDebugPair() {
    return this.octreeProjection?.globalFineTransportDeltaDebugPair;
  }
  get globalFineSourceDebugPair() { return this.octreeProjection?.globalFineSourceDebugPair; }
  get globalFineRedistanceControl() { return this.octreeProjection?.globalFineRedistanceControl; }
  get globalFinePageDeltaDebug() { return this.octreeProjection?.globalFinePageDeltaDebug; }
  get globalFinePageDeltaDebugPair() { return this.octreeProjection?.globalFinePageDeltaDebugPair; }
  get globalFineVolumeControl() { return this.octreeProjection?.globalFineVolumeControl; }
  get structuredProjectionEnergyStats() { return this.octreeProjection?.structuredProjectionEnergyStats; }
  get globalFineCoarseLevelSetControl() { return this.octreeProjection?.globalFineCoarseLevelSetControl; }
  readPowerCoarseFailureRow(row: number) { return this.octreeProjection?.readPowerCoarseFailureRow(row); }
  get globalFineRestrictionControl() { return this.octreeProjection?.globalFineRestrictionControl; }
  get columnBaseTexture() { return this.hostAllocation ? this.heightA : undefined; }
  get gridCellTexture() { return this.adaptiveProjection?.topologyTexture; }
  get velocityTexture() { return this.octreeProjection ? undefined : this.velocityA; }
  get secondaryParticles() { return undefined; }
  applyRuntimeValues(_values: Record<string, string | number | boolean>) {}
  /**
   * Adopt scene scalars that no lattice, arena, or seed depends on. This
   * solver reads them from `this.scene` when it writes per-step params, so the
   * swap alone is enough; the octree projection keeps its own params buffer
   * and is refreshed explicitly.
   */
  applySceneUniforms(scene: SceneDescription) {
    this.scene = scene;
    this.octreeProjection?.applySceneUniforms(scene);
  }

  /**
   * Bake `this.scene`'s terrain into the r32float column texture, in cell
   * units. Mirrors the constructor's upload so a warm re-seed refreshes the
   * ground without reallocating the texture every consumer already binds.
   */
  private uploadTerrainColumns() {
    const { nx, ny, nz } = this.info;
    const cellHeight = this.scene.container.height_m / ny;
    const heights = terrainColumnHeights(this.scene, nx, nz);
    const cells = new Float32Array(nx * nz);
    for (let index = 0; index < cells.length; index += 1) cells[index] = heights[index]! / cellHeight;
    const rowBytes = nx * 4, padded = Math.ceil(rowBytes / 256) * 256;
    const packed = new Uint8Array(padded * nz), source = new Uint8Array(cells.buffer);
    for (let k = 0; k < nz; k += 1) packed.set(source.subarray(rowBytes * k, rowBytes * (k + 1)), padded * k);
    this.device.queue.writeTexture({ texture: this.terrainTexture }, packed, { bytesPerRow: padded, rowsPerImage: nz }, { width: nx, height: nz });
  }

  /**
   * Warm re-seed: adopt a scene that differs only in the seed tier (terrain,
   * bodies, initial condition, brick seeds, fill fraction, inflow) and re-run
   * the same four fenced t=0 phases against the existing allocations, arenas,
   * and compiled pipelines. This is the plan's "re-run the cold branch with a
   * new phi seed", not "build a new solver".
   *
   * Returns false — leaving the solver untouched and usable — whenever the new
   * seed cannot be honoured in place. The caller must then take the full
   * rebuild. Failing closed matters more than the speedup: a half-applied seed
   * would run the solver on state that matches no scene.
   */
  async reseed(scene: SceneDescription): Promise<boolean> {
    if (!this.octreeProjection) return false;
    const [nx, ny, nz] = [this.info.nx, this.info.ny, this.info.nz];
    // The seed tier cannot change the lattice; if it somehow has, this is a
    // structural change wearing a seed change's clothes.
    if (Math.max(8, Math.round(scene.container.width_m / scene.voxelDomain.finestCellSize_m)) !== nx) return false;
    try {
      this.scene = scene;
      this.inflowBoundary = scene.fluid.inflow
        ? createInflowGridBoundary(scene.fluid.inflow, scene.container, [nx, ny, nz])
        : undefined;
      // Rigid poses are pushed per step by syncBodies, so a re-seed only has
      // to refresh the ground the solver and contacts read.
      this.uploadTerrainColumns();
      if (!this.octreeProjection.reseed(scene)) return false;
      await this.publishInitialSparseScene();
      this.lastTime = 0;
      this.info.submittedTime_s = 0;
      this.info.simulatedTime_s = 0;
      this.info.simulationLag_s = 0;
      return true;
    } catch {
      // A throw here has already disturbed GPU state, so the caller must not
      // keep using this solver; report failure and let it rebuild.
      return false;
    }
  }
  get gridPressureSamplesTexture() { return this.adaptiveProjection?.pressureSamplesTexture; }
  get gridPressureTexture() { return this.adaptiveProjection?.pressureTexture; }
  get gridDivergenceTexture() { return this.octreeProjection ? undefined : this.quadtreeProjection?.divergenceTexture; }
  ensureGridDiagnosticTextures() {
    if (!this.octreeProjection?.ensureDiagnosticTextures()) return;
    const encoder = this.device.createCommandEncoder({ label: "Initialize lazy octree diagnostic fields" });
    this.octreeProjection.encodeOverlayMaterialization(encoder);
    this.device.queue.submit([encoder.finish()]);
    this.applyOctreeInfo(this.octreeProjection);
  }
  /** Instrumentation view: velocity after advection/forces and before quadtree projection. */
  get preProjectionVelocityTexture() { return this.octreeProjection ? undefined : this.velocityB; }

  private initializeVolume() {
    const { nx, ny, nz } = this.info, c = this.scene.container;
    const data = this.hostAllocation ? new Float32Array(nx * ny * nz) : undefined, dam = damBreakFractions(c.fillFraction);
    const terrainHeights = terrainColumnHeights(this.scene, nx, nz), cellHeight = c.height_m / ny;
    let initialSum = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const aboveGround = (j + 0.5) * cellHeight > terrainHeights[i + nx * k];
      const brickWet = initialFluidBrickContainsCell(this.scene, i, j, k, [nx, ny, nz]);
      const fill = aboveGround && combineInitialBrickWet(this.scene, brickWet, this.scene.fluid.initialCondition === "dam-break"
        ? (i + .5) / nx <= dam.width && (j + .5) / ny <= dam.height && (k + .5) / nz <= dam.depth
        : (j + .5) / ny <= c.fillFraction);
      if (data) data[i + nx * (j + ny * k)] = fill ? 1 : 0; if (fill) initialSum += 1;
    }
    const terrainCells = new Float32Array(nx * nz);
    for (let index = 0; index < terrainCells.length; index++) terrainCells[index] = terrainHeights[index] / cellHeight;
    const terrainRowBytes = nx * 4, terrainPadded = Math.ceil(terrainRowBytes / 256) * 256;
    const terrainPacked = new Uint8Array(terrainPadded * nz), terrainSource = new Uint8Array(terrainCells.buffer);
    for (let k = 0; k < nz; k++) terrainPacked.set(terrainSource.subarray(terrainRowBytes * k, terrainRowBytes * (k + 1)), terrainPadded * k);
    this.device.queue.writeTexture({ texture: this.terrainTexture }, terrainPacked, { bytesPerRow: terrainPadded, rowsPerImage: nz }, { width: nx, height: nz });
    Object.assign(this.info, { initialVolumeCellSum: initialSum,
      volumeCellSum: data ? initialSum : undefined,
      representedVolumeCellSum: data ? initialSum : undefined,
      representedVolumeDrift: data ? 0 : undefined,
      volumeDrift: data ? 0 : undefined,
      rawVolumeDrift: data ? 0 : undefined,
      volumeTelemetrySource: data ? "initial-condition" : "unavailable",
      maxSpeed_m_s: 0,
      front_m: this.scene.fluid.initialCondition === "dam-break" ? -c.width_m / 2 + dam.width * c.width_m : c.width_m / 2,
      frontTelemetrySource: "initial-condition" });
    if (data) {
      const rowBytes = nx * 4, padded = Math.ceil(rowBytes / 256) * 256;
      const packed = new Uint8Array(padded * ny * nz), source = new Uint8Array(data.buffer);
      for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) packed.set(source.subarray(rowBytes * (j + ny * k), rowBytes * (j + ny * k + 1)), padded * (j + ny * k));
      for (const texture of [this.volumeA, this.volumeB]) this.device.queue.writeTexture({ texture }, packed, { bytesPerRow: padded, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
    }
  }

  private group(velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture, pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture, heightIn: GPUTexture, heightOut: GPUTexture, predictedVelocity: GPUTexture = velocityIn, reversedVelocity: GPUTexture = velocityIn, transport: GPUTexture = this.transportA, surfaceIn: GPUTexture = this.adaptiveProjection?.levelSetTexture ?? volumeIn) {
    return this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [
      { binding: 0, resource: velocityIn.createView() }, { binding: 1, resource: velocityOut.createView() },
      { binding: 2, resource: pressureIn.createView() }, { binding: 3, resource: pressureOut.createView() },
      { binding: 4, resource: volumeIn.createView() }, { binding: 5, resource: volumeOut.createView() },
      { binding: 6, resource: { buffer: this.params! } }, { binding: 7, resource: heightIn.createView() },
      { binding: 8, resource: heightOut.createView() }, { binding: 9, resource: { buffer: this.reductionBuffer! } },
      { binding: 10, resource: { buffer: this.rigidBuffer } }, { binding: 11, resource: { buffer: this.rigidExchangeBuffer } },
      { binding: 12, resource: predictedVelocity.createView() }, { binding: 13, resource: reversedVelocity.createView() },
      { binding: 14, resource: transport.createView() }, { binding: 15, resource: this.transportSampler },
      { binding: 17, resource: this.fluxScales.createView() },
      { binding: 19, resource: { buffer: this.sharpenBuffer! } },
      { binding: 20, resource: surfaceIn.createView() },
      { binding: 21, resource: this.terrainTexture.createView() },
    ] });
  }

  private dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup) {
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
  }
  private dispatchTransport(pass: GPUComputePassEncoder, group: GPUBindGroup, paddedWorkgroups: [number, number, number]) {
    pass.setPipeline(this.buildTransportPipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...paddedWorkgroups);
  }
  private applyQuadtreeInfo(projection: WebGPUQuadtreeTallCellProjection, pressureIterations = this.info.pressureIterations) {
    const quadtree = projection.info;
    Object.assign(this.info, {
      gridKind: "quadtree-tall-cell",
      surfaceField: "levelset",
      volumeControl: true,
      referenceLiquidVolume_cells: projection.surfaceDiagnostics.referenceVolumeCells,
      phiInterfaceCellCount: projection.surfaceDiagnostics.interfaceCells,
      volumeCorrectionNormalSpeed_cells_s: projection.surfaceDiagnostics.correctionSpeed,
      volumeControlAgreeWeight: projection.surfaceDiagnostics.volumeControlAgreeWeight,
      pressureSolver: quadtreePressureDescription(projection, pressureIterations, Math.max(this.scene.numerics.pressureRelativeTolerance, 1e-4)),
      compressionRatio: quadtree.compressionRatio, activeCompressionRatio: quadtree.compressionRatio,
      activeSampleCount: quadtree.liquidDofCount,
      allocatedBytes: this.baseAllocatedBytes + quadtree.allocatedBytes,
      quadtreeLeafCount: quadtree.leafCount, quadtreePressureSampleCount: quadtree.pressureSampleCount,
      quadtreeLiquidDofCount: quadtree.liquidDofCount, quadtreeFaceCount: quadtree.faceCount, quadtreeMLSProjectionRowCount: quadtree.mlsProjectionRowCount,
      quadtreeOpticalLayerMode: quadtree.opticalLayerMode, quadtreeOpticalAlpha: quadtree.opticalAlpha,
      quadtreeOpticalMinimumCells: quadtree.opticalMinimumCells, quadtreeOpticalMaximumCells: quadtree.opticalMaximumCells,
      quadtreeTallSegmentCount: quadtree.tallSegmentCount, quadtreeGhostFaceCount: quadtree.ghostFaceCount,
      quadtreeMaximumNeighborRatio: quadtree.maximumNeighborRatio, quadtreeMaximumFluidScale: quadtree.maximumFluidScale,
      quadtreeLevelSetMismatchFraction: projection.levelSetMismatchFraction ?? 0,
      quadtreeCulledDebrisCells: projection.surfaceDiagnostics.culledDebrisCells,
      quadtreeVofReconciliationActive: projection.surfaceDiagnostics.reconciliationActive,
      quadtreeTopologyReused: quadtree.topologyReused,
      quadtreeTopologyReuseCount: quadtree.topologyReuseCount,
      quadtreePressureIterationsUsed: quadtree.pressureIterationsUsed,
      quadtreePressureIterationBudget: quadtree.pressureIterationBudget,
      quadtreePressureIterationHardBudget: quadtree.pressureIterationHardBudget,
      quadtreePressureConverged: quadtree.pressureConverged,
      quadtreeVelocityClampCount: quadtree.velocityClampCount,
      quadtreeFactorLevelCount: quadtree.factorLevelCount,
      quadtreeMultigridLevelCount: quadtree.multigridLevelCount,
      quadtreeMultigridCoarsestDofs: quadtree.multigridCoarsestDofs,
      quadtreeRebuildCadenceSteps: this.quadtreeRebuildInterval,
      // Report the effective path, not merely the preference: coupled and
      // host-factorized pressure variants cannot consume the resident pack.
      quadtreeInlineRebuild: this.quadtreeInlineRebuild && projection.canEncodeInlineRebuild,
      quadtreeTopologyStaleLimit: this.quadtreeTopologyStaleLimit,
      quadtreeTopologyStaleSteps: this.quadtreeStepsSinceTopology,
      quadtreeRebuildCompletedCount: this.quadtreeRebuildCompletedCount,
      quadtreeTopologyReadbackBytes: quadtree.topologyReadbackBytes
    });
  }
  private applyOctreeInfo(projection: WebGPUOctreeProjection) {
    const octree = projection.info;
    Object.assign(this.info, {
      gridKind: "octree",
      initialSparseAuthorityReady: this.initialSparseAuthorityPublished,
      surfaceField: "levelset",
      volumeControl: true,
      pressureSolver: projection.pressureSolverLabel,
      compressionRatio: octree.compressionRatio,
      activeCompressionRatio: octree.compressionRatio,
      activeSampleCount: octree.liquidDofCount,
      allocatedBytes: this.baseAllocatedBytes + octree.allocatedBytes,
      secondaryParticleCapacity: undefined,
      fluidBrickCapacity: projection.fluidBrickCapacity,
      quadtreeLeafCount: octree.leafCount,
      quadtreePressureSampleCount: octree.pressureSampleCount,
      quadtreeLiquidDofCount: octree.liquidDofCount,
      quadtreeMaximumNeighborRatio: octree.maximumNeighborRatio,
      quadtreeMaximumFluidScale: octree.maximumFluidScale,
      quadtreePressureIterationsUsed: octree.pressureIterationsUsed,
      quadtreePressureIterationBudget: octree.pressureIterationBudget,
      quadtreePressureIterationHardBudget: octree.pressureIterationHardBudget,
      quadtreePressureConverged: octree.pressureConverged,
      pressureRowCapacity: octree.pressureRowCapacity,
      pressureRequiredRows: octree.pressureRequiredRows,
      pressureCapacityOverflow: octree.pressureCapacityOverflow,
      powerDiagramReady: octree.powerDiagramReady,
      powerDiagramAuthoritative: octree.powerDiagramAuthoritative,
      ...(projection.powerPublicationGeneration === undefined ? {}
        : { powerDiagramGeneration: projection.powerPublicationGeneration }),
      powerDiagramAllocatedBytes: octree.powerDiagramAllocatedBytes,
      globalFineLevelSetAllocatedBytes: octree.globalFineLevelSetAllocatedBytes,
      globalFineLevelSetResidentBrickCapacity: octree.globalFineLevelSetResidentBrickCapacity,
      globalFineLevelSetLogicalBrickCount: octree.globalFineLevelSetLogicalBrickCount,
      globalFineLevelSetEnabled: projection.globalFineLevelSetSource !== undefined,
      globalFineLevelSetFactor: projection.globalFineLevelSetSource?.plan.fineFactor,
      globalFineTransportQueryCapacity: octree.globalFineTransportQueryCapacity,
      globalFineTransportChunkCapacity: octree.globalFineTransportChunkCapacity,
      globalFineTransportChunkCount: octree.globalFineTransportChunkCount,
      globalFineTransportSegmentCount: octree.globalFineTransportSegmentCount,
      globalFineTransportEncodedPasses: octree.globalFineTransportEncodedPasses,
      globalFineTransportPrepassScratchBytes: octree.globalFineTransportPrepassScratchBytes,
      globalFineTransportVertexScratchBytes: octree.globalFineTransportVertexScratchBytes,
      frontierListCapacity: octree.frontierListCapacity,
      frontierRequiredLeaves: octree.frontierRequiredLeaves,
      frontierCapacityOverflow: octree.frontierCapacityOverflow,
      quadtreeInlineRebuild: true,
      quadtreeRebuildCadenceSteps: 1,
      quadtreeTopologyStaleLimit: 0,
      quadtreeTopologyStaleSteps: 0,
      quadtreeTopologyReadbackBytes: 0,
      hostFluidAuthority: "gpu-resident",
      hostSimulationSizedWorkItems: 0,
      // The octree outer graph is encoded exactly once per fixed controller
      // step.  Its fine characteristic kernel owns the fixed maximum segment
      // schedule and derives the active segment count on the GPU from the
      // accepted structured velocity publication.
      hostSchedulingUsesReadback: false,
    });
  }
  private statsReadback() {
    // readbackPending guarantees that this buffer is never copied while mapped.
    return this.statsReadbackBuffer ??= this.device.createBuffer({
      label: "Uniform pooled statistics readback",
      size: 16 + STRUCTURED_PROJECTION_ENERGY_WORDS * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  }

  /**
   * A rebuild may resolve between frame encoding and queue submission. Waiting
   * on the queue immediately is therefore insufficient: onSubmittedWorkDone
   * only covers work submitted before it was called. Retire on the following
   * animation frame, after the frame loop has submitted every command buffer
   * that could still reference the old projection, and then wait for the GPU.
   */
  private retireQuadtreeProjection(projection: WebGPUQuadtreeTallCellProjection) {
    this.retiredQuadtreeProjections.add(projection);
    const waitForSubmittedFrame = () => {
      void this.device.queue.onSubmittedWorkDone().catch(() => { /* Device loss invalidates resources first. */ }).finally(() => {
        if (this.retiredQuadtreeProjections.delete(projection)) projection.destroy();
      });
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(waitForSubmittedFrame);
    else setTimeout(waitForSubmittedFrame, 0);
  }

  private shouldKickQuadtreeRebuild() {
    return this.quadtreeStepsSinceTopology >= this.quadtreeRebuildInterval;
  }

  /**
   * Launch the next topology rebuild from the current resident GPU level set.
   * Surface transport is per-step, so topology construction never integrates
   * a multi-step dt with an end-of-interval velocity.
   */
  private kickQuadtreeRebuild() {
    const previous = this.quadtreeProjection;
    if (!previous || this.quadtreeRebuildPending) return;
    this.quadtreeRebuildPending = true;
    this.info.quadtreeRebuildPending = true;
    const bodiesAtKick = this.quadtreeLastBodies.map((body) => structuredClone(body));
    this.quadtreeStepsSinceKick = 0;
    void previous.rebuildFromState(bodiesAtKick).then((next) => {
      if (this.quadtreeBlockedSince_ms !== undefined) {
        const missedFrames = quadtreeMissedFrames(performance.now() - this.quadtreeBlockedSince_ms);
        this.quadtreeRebuildBlockedFrames += missedFrames;
        this.quadtreeBlockedSince_ms = undefined;
      }
      this.quadtreeRebuildFallbackWarned = false;
      this.quadtreeRebuildPending = false;
      this.quadtreeRebuildFailureCount = 0;
      this.quadtreeRebuildRetrySteps = 0;
      this.info.quadtreeRebuildPending = false;
      this.info.quadtreeRebuildBlockedFrames = this.quadtreeRebuildBlockedFrames;
      if (this.disposed) { if (next !== previous) next.destroy(); return; }
      // Stage the finished projection; advanceTo applies it at the fixed
      // step boundary so the swap schedule depends only on step counts,
      // never on rebuild wall time (keeps stepping deterministic).
      this.quadtreeReadyProjection = next;
    }).catch((error) => {
      this.quadtreeBlockedSince_ms = undefined;
      this.quadtreeRebuildFallbackWarned = false;
      this.quadtreeRebuildPending = false;
      this.info.quadtreeRebuildPending = false;
      this.quadtreeRebuildFailureCount += 1;
      this.quadtreeRebuildRetrySteps = quadtreeRebuildRetryDelay(this.quadtreeRebuildFailureCount);
      console.error(`Quadtree tall-cell rebuild failed; reusing the previous topology and retrying in ${this.quadtreeRebuildRetrySteps} steps`, error);
    });
  }

  private applyReadyQuadtreeProjection() {
    const next = this.quadtreeReadyProjection, previous = this.quadtreeProjection;
    if (!next || !previous) return;
    this.quadtreeReadyProjection = undefined;
    this.quadtreeRebuildCompletedCount += 1;
    this.quadtreeProjection = next; this.applyQuadtreeInfo(next);
    // Steps advanced on the previous topology while the rebuild was in
    // flight; the swapped topology is stepsSinceKick steps behind the
    // surface, which the swap boundary keeps bounded.
    this.quadtreeStepsSinceTopology = this.quadtreeStepsSinceKick;
    // The replaced projection's buffers may still be referenced by queued
    // steps; only release them once the queue drains.
    if (next !== previous) this.retireQuadtreeProjection(previous);
  }

  advanceTo(time_s: number, bodies: RigidBodyState[] = []) {
    if (this.disposed || this.profiledAdvanceCompletion) return false;
    // Deterministic bounded-staleness rebuild pipeline. Algorithm 1 wants the
    // quadtree constructed before advection and pressure, but a synchronous
    // handshake costs one full GPU-readback + worker-pack round trip per
    // step. Instead, up to quadtreeTopologyStaleLimit steps run ahead on the
    // previous topology while its replacement is assembled, and the finished
    // projection is applied exactly at that step boundary — blocking there if
    // the rebuild is still in flight — so the swap schedule depends only on
    // step counts, never on rebuild wall time. refreshFaces re-derives the
    // free-surface fractions from the live level set every solve, so only
    // the DOF layout is stale in between.
    if (this.quadtreeProjection && this.rebuildQuadtreeEachStep && (this.quadtreeRebuildPending || this.quadtreeReadyProjection) && this.quadtreeStepsSinceKick >= this.quadtreeTopologyStaleLimit) {
      if (!this.quadtreeReadyProjection) {
        this.quadtreeBlockedSince_ms ??= performance.now();
        this.info.quadtreeRebuildBlockedFrames = this.quadtreeRebuildBlockedFrames;
        const blocked_ms = performance.now() - this.quadtreeBlockedSince_ms;
        // A rebuild is an optimization of the pressure layout, not a reason
        // to starve surface transport indefinitely. After three presentation
        // budgets, keep advancing on the previous valid topology until the
        // asynchronous replacement arrives.
        // A failed replacement already proved that waiting cannot make this
        // attempt usable. Advance immediately on the previous projection so
        // retry backoff can count down; only an in-flight first attempt gets
        // the short presentation-budget grace period.
        if (this.quadtreeRebuildFailureCount === 0 && blocked_ms < 3 * 1000 / 60) return false;
        if (!this.quadtreeRebuildFallbackWarned) {
          console.warn(`Quadtree topology rebuild blocked for ${blocked_ms.toFixed(1)} ms; reusing the previous topology until it completes`);
          this.quadtreeRebuildFallbackWarned = true;
        }
      }
      if (this.quadtreeReadyProjection) this.applyReadyQuadtreeProjection();
    }
    const advance = planGPUAdvance(time_s, this.lastTime, this.scene.numerics.maxDt_s); if (!advance) return false;
    const delta = advance.dt_s; if (delta < 1e-6) { this.info.simulatedTime_s = this.lastTime; this.info.simulationLag_s = advance.lag_s; return true; }
    this.lastTime = advance.nextTime_s; this.info.submittedTime_s = this.lastTime; this.info.simulatedTime_s = this.lastTime; this.info.simulationLag_s = advance.lag_s; const c = this.scene.container, rho = this.scene.fluid.density_kg_m3, sigma = this.scene.fluid.surfaceTension_N_m;
    // The quadtree backend still uses its host-side outer subdivision.  The
    // octree does not: its outer command graph is one fixed controller step,
    // while direct fine transport encodes a fixed maximum characteristic
    // schedule and selects active segments from GPU-resident structured
    // velocity state.  This keeps scheduling decisions off the host and
    // preserves the exact displacement used by support-closure validation.
    const coarseHMin = Math.min(c.width_m / this.info.nx, c.height_m / this.info.ny, c.depth_m / this.info.nz);
    // Sparse phi advection is semi-Lagrangian and does not force the global
    // pressure solve onto the fine geometric timestep. Preserve the coarse
    // Chebyshev cadence unless explicit fine dynamics is enabled.
    const hMin = coarseHMin;
    const inflowSpeed = this.scene.fluid.inflow ? Math.hypot(this.scene.fluid.inflow.velocity_m_s.x, this.scene.fluid.inflow.velocity_m_s.y, this.scene.fluid.inflow.velocity_m_s.z) : 0;
    const substeps = this.quadtreeProjection ? proactiveQuadtreeSubsteps(
      this.info.maxSpeed_m_s ?? 0,
      inflowSpeed,
      Math.hypot(this.scene.fluid.gravity_m_s2.x, this.scene.fluid.gravity_m_s2.y, this.scene.fluid.gravity_m_s2.z),
      delta,
      hMin,
      64,
      rho,
      sigma
    ) : 1;
    const dt = delta / substeps;
    this.info.lastDt_s = this.octreeProjection ? undefined : dt;
    this.info.lastSubsteps = this.octreeProjection ? undefined : substeps;
    this.octreeProjection?.setTimestep(dt);
    const activeBodies = bodies.slice(0, 12);
    this.rigidSystem.syncBodies(activeBodies);
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + (this.octreeProjection ? 1 : substeps);
    const measurementInstrumentationEnabled = usePerformanceInstrumentationStore.getState().enabled;
    const traceRequestedAt_ms = measurementInstrumentationEnabled ? performance.now() : 0;
    const timestampTraceUnavailable = !GPUPerformanceTraceRecorder.supported(this.device)
      || this.hardwarePhysicsTraceInvalid;
    const shouldSegmentPhysics = measurementInstrumentationEnabled
      && Boolean(this.octreeProjection)
      && !this.physicsTracePending
      && timestampTraceUnavailable
      && traceRequestedAt_ms - this.lastSegmentedPhysicsTraceAt_ms >= SEGMENTED_QUEUE_TRACE_CADENCE_MS;
    const shouldTracePhysics = measurementInstrumentationEnabled
      && !this.physicsTracePending
      && (shouldSegmentPhysics
        || traceRequestedAt_ms - this.lastPhysicsTraceAt_ms >= PHYSICS_TRACE_CADENCE_MS);
    this.octreeProjection?.setCouplingBodies(activeBodies.length, activeBodies.some((body) => body.inverseMass_kg > 0));
    const inflow=this.scene.fluid.inflow,outlet=this.inflowBoundary?.outletCenter_m,inflowStepStrength=inflow?averageInflowStrength(inflow,this.lastTime-delta,this.lastTime):0;
    if(this.adaptiveProjection&&this.inflowBoundary){const cellVolume=c.width_m*c.height_m*c.depth_m/(this.info.nx*this.info.ny*this.info.nz);this.adaptiveProjection.addSurfaceReferenceVolumeCells(this.inflowBoundary.flowRate_m3_s*inflowStepStrength*delta/cellVolume);}
    if (this.params) this.device.queue.writeBuffer(this.params, 0, new Float32Array([this.info.nx, this.info.ny, this.info.nz, dt, c.width_m / this.info.nx, c.height_m / this.info.ny, c.depth_m / this.info.nz, this.scene.fluid.gravity_m_s2.y, c.width_m, c.height_m, c.depth_m, sceneHasTerrain(this.scene) ? 1 : 0, rho, this.scene.fluid.dynamicViscosity_Pa_s, this.transportConservativeVolume ? 1 : 0, this.adaptiveProjection ? 1 : 0, sigma, c.fluidWallMode === "no-slip" ? 1 : 0, activeBodies.length, c.top === "open" ? 1 : 0,outlet?.x??0,outlet?.y??0,outlet?.z??0,inflow?.radius_m??0,inflow?.velocity_m_s.x??0,inflow?.velocity_m_s.y??0,inflow?.velocity_m_s.z??0,this.inflowBoundary?.apertureScale??0,inflowStepStrength,0,c.fillFraction*this.info.ny,0]));
    if (!this.validationChecked) this.device.pushErrorScope("validation");
    let encoder = this.device.createCommandEncoder({ label: "Uniform GPU fluid step" });
    const physicsTraceSampleId = shouldTracePhysics ? ++this.physicsTraceSampleId : 0;
    const physicsTrace = shouldTracePhysics
      && !shouldSegmentPhysics
      && GPUPerformanceTraceRecorder.supported(this.device)
      ? new DynamicGPUPerformanceTraceRecorder(
        this.device,
        physicsTraceSampleId,
        "physics",
        `${this.info.gridKind}:sim-${this.lastTime.toFixed(6)}`,
        2048,
      )
      : undefined;
    const segmentedPhysicsTrace = shouldSegmentPhysics
      ? new GPUSegmentedQueueWallPerformanceTraceRecorder(
        this.device,
        physicsTraceSampleId,
        "physics",
        `${this.info.gridKind}:sim-${this.lastTime.toFixed(6)}`,
      )
      : undefined;
    const physicsQueueTrace = shouldTracePhysics && !segmentedPhysicsTrace
      ? new GPUQueueWallPerformanceTraceRecorder(
        physicsTraceSampleId,
        "physics",
        `${this.info.gridKind}:sim-${this.lastTime.toFixed(6)}`,
      )
      : undefined;
    physicsTrace?.begin(encoder);
    const completePhysicsPhase = (
      completedEncoder: GPUCommandEncoder,
      phase: GPUTimestampPhase,
    ): GPUCommandEncoder => {
      physicsTrace?.completePhase(completedEncoder, phase);
      return segmentedPhysicsTrace?.completePhase(completedEncoder, phase) ?? completedEncoder;
    };
    encoder.clearBuffer(this.rigidExchangeBuffer);
    // Narita's quadtree path regenerates at the top of the step. The octree
    // path publishes only a previously validated candidate here; candidate
    // construction itself runs at the tail of each substep.
    let inlineRebuildEncoded = false;
    if (this.quadtreeProjection && this.rebuildQuadtreeEachStep && this.quadtreeInlineRebuild && !this.quadtreeRebuildPending && !this.quadtreeReadyProjection && this.quadtreeProjection.canEncodeInlineRebuild) {
      inlineRebuildEncoded = this.quadtreeProjection.encodeInlineRebuild(encoder);
    }
    encoder = completePhysicsPhase(encoder, inlineRebuildEncoded
      ? { id: "coarse-grid", label: "Adaptive coarse-grid topology" }
      : { id: "other", label: "Advance setup" });
    for (let substep = 0; substep < substeps; substep += 1) {
      // The active epoch is immutable for the entire substep. A ready
      // candidate from the prior tail may flip only at this boundary.
      this.octreeProjection?.encodeReadyTopologyFlip(encoder);
      if (this.octreeProjection) {
        encoder = completePhysicsPhase(
          encoder,
          { id: "power-topology", label: "Accepted topology epoch + Section 5 air support" },
        );
      }
      // U3 compact-face authority owns advection, forces, divergence, and
      // projection inside WebGPUOctreeProjection. The shared dense kernels are
      // not merely redundant here: dispatching them would address the 1x1
      // format-only compatibility textures as though they covered the box.
      if (this.hostAllocation) {
      {
        const paddedWorkgroups: [number, number, number] = [Math.ceil((this.info.nx + 2) / 4), Math.ceil((this.info.ny + 2) / 4), Math.ceil((this.info.nz + 2) / 4)];
        // Occupancy, transport extrapolation, and flux scales only read the
        // projected state, so they share one pass ahead of the predictor.
        const prep = encoder.beginComputePass({ label: "Uniform occupancy and transport preparation" });
        prep.setBindGroup(0, this.occupancyGroup);
        prep.setPipeline(this.buildOccupancyPipeline);
        prep.dispatchWorkgroups(Math.ceil(this.info.nx / 8), Math.ceil(this.info.nz / 8), 1);
        this.dispatchTransport(prep, this.transportFromCurrentGroup, paddedWorkgroups);
        if (this.transportConservativeVolume) {
          prep.setPipeline(this.buildFluxScalesPipeline);
          prep.setBindGroup(0, this.fluxScaleGroup);
          prep.dispatchWorkgroups(Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
        }
        prep.end();
        const predict = encoder.beginComputePass({ label: "Uniform velocity prediction" });
        this.dispatch(predict, this.advectPipeline, this.advectGroup); predict.end();
        if (this.velocityTransport === "maccormack" && this.transportFromPredictedGroup) {
          const predictedTransport = encoder.beginComputePass({ label: "Uniform predicted transport preparation" });
          this.dispatchTransport(predictedTransport, this.transportFromPredictedGroup, paddedWorkgroups); predictedTransport.end();
          const reverse = encoder.beginComputePass({ label: "Uniform reverse advection" }); this.dispatch(reverse, this.reversePipeline, this.reverseGroup); reverse.end();
          const correct = encoder.beginComputePass({ label: "Uniform MacCormack correction" }); this.dispatch(correct, this.correctPipeline, this.correctGroup); correct.end();
        }
      }
      if (this.densitySharpening && this.transportConservativeVolume) {
        // Mass-Conserving Eulerian Liquid Simulation Sec 3.5: sharpen the
        // advected density before the pressure solve. volumeB -> volumeA
        // (sharpened, deltas in pressureB) -> volumeB (resolved deposits).
        encoder.clearBuffer(this.sharpenBuffer!);
        const computePass = encoder.beginComputePass({ label: "Uniform density sharpening" }); this.dispatch(computePass, this.sharpenComputePipeline, this.sharpenComputeGroup); computePass.end();
        const scatterPass = encoder.beginComputePass({ label: "Uniform sharpened-mass scatter" }); this.dispatch(scatterPass, this.sharpenScatterPipeline, this.sharpenScatterGroup); scatterPass.end();
        const resolvePass = encoder.beginComputePass({ label: "Uniform sharpened-mass resolve" }); this.dispatch(resolvePass, this.sharpenResolvePipeline, this.sharpenResolveGroup); resolvePass.end();
      }
      encoder = completePhysicsPhase(encoder, { id: "velocity-advection", label: "Velocity advection + conditioning" });
      }
      if (this.adaptiveProjection) {
        if (this.transportConservativeVolume) encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
        const surfaceInflow = inflow && this.inflowBoundary ? {
          outletCenter_m: this.inflowBoundary.outletCenter_m, radius_m: inflow.radius_m,
          velocity_m_s: inflow.velocity_m_s, apertureScale: this.inflowBoundary.apertureScale,
          strength: inflowStepStrength
        } : undefined;
        if (this.octreeProjection) {
          // Advect both fine and coarse phi with the previous substep's
          // projected + closest-point-extended velocity. Current-substep
          // gravity/forces enter below, after surface transport; using that
          // unprojected predictor here creates systematic boundary volume
          // error.
          encoder = this.octreeProjection.encodeSurface(encoder, dt, surfaceInflow, this.scene.numerics.maxDt_s,
            physicsTrace || segmentedPhysicsTrace ? (phase, completedEncoder) => {
              return completePhysicsPhase(completedEncoder, OCTREE_SEMANTIC_TRACE_PHASE[phase]);
            } : undefined);
          encoder = this.octreeProjection.encode(
            encoder,
            this.info.nx,
            this.info.ny,
            this.info.nz,
            {
              productionBoundary: physicsTrace || segmentedPhysicsTrace ? (phase, completedEncoder) => {
                return completePhysicsPhase(completedEncoder, OCTREE_SEMANTIC_TRACE_PHASE[phase]);
              } : undefined,
            }
          );
          // The current surface and pressure solve consumed one immutable
          // active epoch. Build the next epoch only after both are complete;
          // its validated selector remains pending until the next substep.
          inlineRebuildEncoded =
            this.octreeProjection.encodeInactiveTopologyCandidate(encoder);
          encoder = completePhysicsPhase(
            encoder,
            { id: "coarse-grid", label: "Inactive next-substep topology candidate" },
          );
        } else if (this.quadtreeProjection) {
          this.quadtreeProjection.encode(encoder, this.info.nx, this.info.ny, this.info.nz);
          encoder = completePhysicsPhase(encoder, { id: "pressure-solve", label: "Adaptive pressure + projection" });
          this.adaptiveProjection.encodeSurface(encoder, dt, surfaceInflow, this.scene.numerics.maxDt_s);
        }
        if (!this.octreeProjection) {
          encoder = completePhysicsPhase(
            encoder,
            { id: "fine-sdf-redistance", label: "Surface transport + redistance" },
          );
        }
      } else {
        for (let iteration = 0; iteration < this.info.pressureIterations; iteration += 1) {
          const pass = encoder.beginComputePass({ label: "Uniform Jacobi pressure iteration" });
          this.dispatch(pass, this.jacobiPipeline, iteration % 2 === 0 ? this.jacobiABGroup : this.jacobiBAGroup);
          pass.end();
        }
        encoder = completePhysicsPhase(encoder, { id: "pressure-solve", label: "Uniform pressure solve" });
        { const pass = encoder.beginComputePass({ label: "Uniform pressure projection" }); this.dispatch(pass, this.projectPipeline, this.projectGroup); pass.end(); }
        encoder = completePhysicsPhase(encoder, { id: "velocity-projection", label: "Uniform velocity projection" });
      }
      // Chentanez & Müller Sec. 3.9.1 runs for every pressure backend. Both
      // adaptive projections constrain the normal face flux variationally;
      // this per-substep Brinkman blend supplies tangential drag and interior
      // momentum for moving bodies. Octree additionally folds the dynamic
      // body's pressure response into the next presentation batch, avoiding a
      // global K^T p reduction at every pressure iterate. Phi-s relaxation
      // keeps the resident level set sane inside either backend's solids so
      // they displace water instead of carrying sealed liquid plugs.
      if (activeBodies.length > 0 && !this.octreeProjection) {
        if (this.adaptiveProjection && this.solidPhiGroup) {
          encoder.copyTextureToTexture({ texture: this.adaptiveProjection.levelSetTexture }, { texture: this.pressureA }, [this.info.nx, this.info.ny, this.info.nz]);
          const phiPass = encoder.beginComputePass({ label: "Uniform solid level-set relaxation" }); this.dispatch(phiPass, this.relaxSolidPhiPipeline, this.solidPhiGroup); phiPass.end();
        }
        const pass = encoder.beginComputePass({ label: "Uniform rigid-body coupling" });
        this.dispatch(pass, this.rigidPipeline, this.rigidGroup); pass.end();
        if (this.transportConservativeVolume) encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
        encoder.copyTextureToTexture({ texture: this.velocityB }, { texture: this.velocityA }, [this.info.nx, this.info.ny, this.info.nz]);
      }
    }
    if (activeBodies.length > 0) {
      this.quadtreeProjection?.encodeBodyImpulseExchange(encoder, this.rigidExchangeBuffer);
      const cellVolume = c.width_m * c.height_m * c.depth_m / (this.info.nx * this.info.ny * this.info.nz);
      this.rigidSystem.encode(encoder, delta, cellVolume, substeps, c.height_m / this.info.ny);
      encoder = completePhysicsPhase(
        encoder,
        { id: "other", label: "Rigid-body impulse exchange + integration" },
      );
    }
    // Publish the final substep's resident fields into the shared sparse-brick
    // world. The topology and payload stay GPU-resident; rendering consumes
    // compact debug records and subsequent voxel kernels consume the same ABI.
    if (this.octreeProjection) {
      this.octreeProjection.encodeSparseBrickWorld(encoder, dt);
      encoder = completePhysicsPhase(
        encoder,
        { id: "adaptive-publication", label: "Sparse-brick residency + publication" },
      );
    }
    if (this.hostAllocation) {
      encoder.clearBuffer(this.reductionBuffer!);
      const pass = encoder.beginComputePass({ label: "Uniform diagnostics reduction" });
      this.dispatch(pass, this.reductionPipeline, this.reductionGroup); pass.end();
      encoder = completePhysicsPhase(
        encoder,
        { id: "other", label: "Diagnostics reduction" },
      );
    }
    if (segmentedPhysicsTrace) {
      this.lastPhysicsTraceAt_ms = traceRequestedAt_ms;
      this.lastSegmentedPhysicsTraceAt_ms = traceRequestedAt_ms;
      this.physicsTracePending = true;
      const completion = segmentedPhysicsTrace.read(
        this.device.queue,
        (submittedEncoder) => this.octreeProjection?.retireSubmittedEncoder(submittedEncoder),
      ).then((trace) => {
        const instrumentation = usePerformanceInstrumentationStore.getState();
        if (trace && !this.disposed && instrumentation.enabled
          && instrumentation.enabledAt_ms <= traceRequestedAt_ms) this.info.physicsTrace = trace;
      }).finally(() => {
        this.octreeProjection?.releaseDenseBootstrapPhi();
        this.physicsTracePending = false;
        this.profiledAdvanceCompletion = undefined;
      });
      this.profiledAdvanceCompletion = completion;
    } else {
      physicsTrace?.resolve(encoder);
      const submittedEncoder = encoder;
      physicsQueueTrace?.begin();
      this.device.queue.submit([submittedEncoder.finish()]);
      this.octreeProjection?.retireSubmittedEncoder(submittedEncoder);
      const physicsQueueTraceRead = physicsQueueTrace?.read(this.device.queue);
      const physicsTraceRead = physicsTrace
        ? physicsTrace.read()
          .then((trace) => {
            this.hardwarePhysicsTraceInvalid = !trace;
            return trace ?? physicsQueueTraceRead;
          })
          .catch(() => {
            this.hardwarePhysicsTraceInvalid = true;
            return physicsQueueTraceRead;
          })
        : physicsQueueTraceRead;
      if (physicsTraceRead) {
        this.lastPhysicsTraceAt_ms = traceRequestedAt_ms;
        this.physicsTracePending = true;
        void physicsTraceRead.then((trace) => {
          const instrumentation = usePerformanceInstrumentationStore.getState();
          if (trace && !this.disposed && instrumentation.enabled
            && instrumentation.enabledAt_ms <= traceRequestedAt_ms) this.info.physicsTrace = trace;
        }).catch(() => {
          physicsTrace?.destroy();
        }).finally(() => {
          this.physicsTracePending = false;
        });
      }
      // The submitted bootstrap command retains its own reference. Later
      // submissions use global-fine phi, so the final box-sized level set can die now.
      this.octreeProjection?.releaseDenseBootstrapPhi();
    }
    // Solver residuals are sampled only through the opt-in telemetry path.
    // Physics submission itself must not initiate a GPU-to-CPU map.
    if (this.octreeProjection && inlineRebuildEncoded) {
      for (let rebuild = 0; rebuild < substeps; rebuild += 1) {
        this.octreeProjection.finishTopologyCandidate();
      }
      this.applyOctreeInfo(this.octreeProjection);
      this.info.quadtreeRebuildCompletedCount = (this.info.quadtreeRebuildCompletedCount ?? 0) + substeps;
    }
    if (this.quadtreeProjection && this.rebuildQuadtreeEachStep) {
      this.quadtreeStepsSinceTopology += 1; this.quadtreeStepsSinceKick += 1;
      if (inlineRebuildEncoded) {
        // The step just submitted carries its own freshly regenerated
        // topology: staleness is zero by construction and every step counts
        // as a completed rebuild (paper cadence).
        this.quadtreeProjection.finishInlineRebuild();
        this.quadtreeStepsSinceTopology = 0;
        this.quadtreeRebuildCompletedCount += 1;
        // Republish projection telemetry: the non-blocking packControl
        // monitor refreshes leaf/DOF/face counts without any swap.
        this.applyQuadtreeInfo(this.quadtreeProjection);
      }
      this.info.quadtreeTopologyStaleSteps = this.quadtreeStepsSinceTopology;
      this.info.quadtreeTopologyStaleLimit = inlineRebuildEncoded ? 0 : this.quadtreeTopologyStaleLimit;
      this.quadtreeLastBodies = activeBodies.filter((body) => body.description.motion === "static");
      if (this.quadtreeRebuildRetrySteps > 0) this.quadtreeRebuildRetrySteps -= 1;
      if (!inlineRebuildEncoded && this.quadtreeRebuildRetrySteps === 0 && !this.quadtreeRebuildPending && !this.quadtreeReadyProjection && this.shouldKickQuadtreeRebuild()) this.kickQuadtreeRebuild();
    }
    if (!this.validationChecked) {
      this.validationChecked = true;
      this.validationPromise = this.device.popErrorScope().then((error) => {
        if (!error) return;
        this.info.gpuValidationError = error.message;
        console.error(`Uniform GPU validation: ${error.message}`);
      }).catch(() => { /* Device loss is handled by the renderer. */ });
    }
    return true;
  }

  async readStats() {
    if ((this.info.encodedSteps ?? 0) === 0 || this.readbackPending) return this.info;
    this.readbackPending = true;
    const buffer = this.statsReadback(), encoder = this.device.createCommandEncoder();
    if (this.reductionBuffer) encoder.copyBufferToBuffer(this.reductionBuffer, 0, buffer, 0, 16);
    const structuredProjectionEnergy = this.octreeProjection?.structuredProjectionEnergyStats;
    if (structuredProjectionEnergy) encoder.copyBufferToBuffer(
      structuredProjectionEnergy, 0, buffer, 16, STRUCTURED_PROJECTION_ENERGY_WORDS * 4,
    );
    this.device.queue.submit([encoder.finish()]);
    const mapPromise = buffer.mapAsync(GPUMapMode.READ);
    const compactFineExpected = Boolean(this.octreeProjection?.globalFineLevelSetSource);
    const quadtreeDiagnostics = this.adaptiveProjection?.readSolveDiagnostics();
    // Once compact global-fine volume is authoritative, the adaptive surface
    // diagnostic is both obsolete and ignored below. Avoid a separate queue
    // submission/map every 250 ms for data that cannot be selected.
    const surfaceDiagnosticsPromise = compactFineExpected
      ? undefined
      : this.adaptiveProjection?.readSurfaceDiagnostics();
    const globalFineDiagnosticsPromise = this.octreeProjection?.readGlobalFineLevelSetDiagnostics();
    try {
      await mapPromise;
      const [, , surfaceDiagnostics, globalFineDiagnostics, fluidBrickStats, fluidBulkBrickStats] = await Promise.all([
        this.validationPromise, quadtreeDiagnostics, surfaceDiagnosticsPromise, globalFineDiagnosticsPromise, this.octreeProjection?.readFluidBrickResidencyStats(), this.octreeProjection?.readFluidBulkBrickResidencyStats(),
      ]);
    if(globalFineDiagnostics)this.applyGlobalFineDiagnostics(globalFineDiagnostics);
    if(fluidBrickStats){this.info.fluidBrickCapacity=fluidBrickStats.capacity;this.info.fluidBrickResidentCount=fluidBrickStats.resident;this.info.fluidBrickCoreCount=fluidBrickStats.core;this.info.fluidBrickHaloCount=fluidBrickStats.halo;this.info.fluidBrickActivatedCount=fluidBrickStats.activated;this.info.fluidBrickRetiredCount=fluidBrickStats.retired;this.info.fluidBrickGeneration=fluidBrickStats.generation;}
    if(fluidBulkBrickStats){this.info.fluidBulkBrickResidentCount=fluidBulkBrickStats.resident;this.info.fluidBulkBrickHaloCount=fluidBulkBrickStats.halo;this.info.fluidBulkBrickActivatedCount=fluidBulkBrickStats.activated;this.info.fluidBulkBrickRetiredCount=fluidBulkBrickStats.retired;}
    if (this.quadtreeProjection) this.info.quadtreeVelocityClampCount = this.quadtreeProjection.info.velocityClampCount ?? 0;
    const words = this.reductionBuffer
      ? new Uint32Array(buffer.getMappedRange(0, 16))
      : new Uint32Array(4);
    const structuredEnergy = structuredProjectionEnergy
      ? decodeStructuredProjectionEnergy(new Uint32Array(
        buffer.getMappedRange(16, STRUCTURED_PROJECTION_ENERGY_WORDS * 4),
      ))
      : undefined;
    if (structuredEnergy?.sample) {
      this.info.structuredPreProjectionKineticEnergyProxy =
        structuredEnergy.sample.preProjectionKineticEnergyProxy;
      this.info.structuredPostProjectionKineticEnergyProxy =
        structuredEnergy.sample.postProjectionKineticEnergyProxy;
      this.info.structuredProjectionEnergyRatio = structuredEnergy.sample.projectionEnergyRatio;
      this.info.structuredProjectionEnergySampleCount = 1;
    } else {
      this.info.structuredPreProjectionKineticEnergyProxy = undefined;
      this.info.structuredPostProjectionKineticEnergyProxy = undefined;
      this.info.structuredProjectionEnergyRatio = undefined;
      this.info.structuredProjectionEnergySampleCount = undefined;
    }
    const initial = Math.max(1, this.info.initialVolumeCellSum ?? 1);
    const conservativeVolumeCells=words[3]/2048;this.info.rawVolumeDrift=this.transportConservativeVolume?(conservativeVolumeCells-initial)/initial:undefined;
    const c=this.scene.container,baseCellVolume_m3=c.width_m*c.height_m*c.depth_m/(this.info.nx*this.info.ny*this.info.nz);
    const compactVolumeTopology=compactFineExpected&&globalFineDiagnostics
      ? unpackFineLevelSetGPUTopologyControl(globalFineDiagnostics.topologyControl) : undefined;
    const compactVolume=compactFineExpected&&globalFineDiagnostics&&compactVolumeTopology
      ? publishedGlobalFineVolumeCells({
        published:compactVolumeTopology.published,
        rolledBack:compactVolumeTopology.rolledBack,
        downstreamFinalizeReason:compactVolumeTopology.downstreamFinalizeReason,
        generation:globalFineDiagnostics.configuredFineGeneration,
        volumeControl:globalFineDiagnostics.fineVolumeControl,
      },baseCellVolume_m3) : undefined;
    if(compactVolume){
      this.info.referenceLiquidVolume_cells=compactVolume?.referenceVolumeCells;
      this.info.volumeCellSum=compactVolume?.volumeCells;
      this.info.representedVolumeCellSum=compactVolume?.volumeCells;
      this.info.volumeDrift=compactVolume?.drift;
      this.info.representedVolumeDrift=compactVolume?.drift;
      this.info.volumeTelemetrySource="global-fine";
    }
    else if(surfaceDiagnostics&&!compactFineExpected){const resolved=sparseSurfaceVolumeCells(surfaceDiagnostics,this.info.initialVolumeCellSum??0),reference=Math.max(1,resolved.referenceVolumeCells);this.info.referenceLiquidVolume_cells=resolved.referenceVolumeCells;this.info.volumeCellSum=resolved.volumeCells;this.info.representedVolumeCellSum=resolved.volumeCells;this.info.volumeDrift=(resolved.volumeCells-reference)/reference;this.info.representedVolumeDrift=this.info.volumeDrift;this.info.volumeTelemetrySource="adaptive-pages";this.info.phiInterfaceCellCount=surfaceDiagnostics.interfaceCells;this.info.volumeCorrectionNormalSpeed_cells_s=surfaceDiagnostics.correctionSpeed;this.info.volumeControlAgreeWeight=surfaceDiagnostics.volumeControlAgreeWeight;this.info.quadtreeCulledDebrisCells=surfaceDiagnostics.culledDebrisCells;this.info.quadtreeLevelSetMismatchFraction=surfaceDiagnostics.mismatchFraction;this.info.quadtreeVofReconciliationActive=surfaceDiagnostics.reconciliationActive;}
    else if(this.octreeProjection){this.info.referenceLiquidVolume_cells=undefined;this.info.volumeCellSum=undefined;this.info.representedVolumeCellSum=undefined;this.info.volumeDrift=undefined;this.info.representedVolumeDrift=undefined;this.info.volumeTelemetrySource="unavailable";}
    else{this.info.representedVolumeCellSum=words[0]/2048;this.info.representedVolumeDrift=(this.info.representedVolumeCellSum-initial)/initial;this.info.volumeCellSum=conservativeVolumeCells;this.info.volumeDrift=this.info.rawVolumeDrift;this.info.volumeTelemetrySource="dense-volume";}
    // Compact transport never runs the dense reduction which owns words[1].
    // Do not relabel its cleared zero as a measured front at the tank wall.
    if(!this.octreeProjection){this.info.front_m = -this.scene.container.width_m / 2 + words[1] * this.scene.container.width_m / this.info.nx;this.info.frontTelemetrySource="dense-volume";}
    else{this.info.front_m=undefined;this.info.frontTelemetrySource="unavailable";}
    this.info.maxSpeed_m_s = this.octreeProjection
      ? undefined
      : new Float32Array(new Uint32Array([words[2]]).buffer)[0];
    if (this.quadtreeProjection?.relativeResidual !== undefined) this.info.pressureRelativeResidual = this.quadtreeProjection.relativeResidual;
    if (this.quadtreeProjection?.residualRms !== undefined) this.info.pressureResidual = this.quadtreeProjection.residualRms;
    if (this.octreeProjection?.relativeResidual !== undefined) this.info.pressureRelativeResidual = this.octreeProjection.relativeResidual;
    if (this.octreeProjection?.residualRms !== undefined) this.info.pressureResidual = this.octreeProjection.residualRms;
    if (this.quadtreeProjection) {
      this.info.quadtreePressureIterationsUsed = this.quadtreeProjection.info.pressureIterationsUsed;
      this.info.quadtreePressureIterationBudget = this.quadtreeProjection.info.pressureIterationBudget;
      this.info.quadtreePressureIterationHardBudget = this.quadtreeProjection.info.pressureIterationHardBudget;
      this.info.quadtreePressureConverged = this.quadtreeProjection.info.pressureConverged;
      this.info.quadtreeFactorLevelCount = this.quadtreeProjection.info.factorLevelCount;
      this.info.quadtreeMultigridLevelCount = this.quadtreeProjection.info.multigridLevelCount;
      this.info.quadtreeMultigridCoarsestDofs = this.quadtreeProjection.info.multigridCoarsestDofs;
      this.info.pressureSolver = quadtreePressureDescription(this.quadtreeProjection, this.info.pressureIterations, Math.max(this.scene.numerics.pressureRelativeTolerance, 1e-4));
    } else if (this.octreeProjection) {
      this.info.activeSampleCount = this.octreeProjection.info.liquidDofCount;
      this.info.activeCompressionRatio = this.octreeProjection.info.compressionRatio;
      this.info.compressionRatio = this.octreeProjection.info.compressionRatio;
      this.info.quadtreePressureSampleCount = this.octreeProjection.info.pressureSampleCount;
      this.info.quadtreeLiquidDofCount = this.octreeProjection.info.liquidDofCount;
      this.info.quadtreePressureIterationsUsed = this.octreeProjection.info.pressureIterationsUsed;
      this.info.quadtreePressureIterationBudget = this.octreeProjection.info.pressureIterationBudget;
          this.info.quadtreePressureIterationHardBudget = this.octreeProjection.info.pressureIterationHardBudget;
          this.info.pressureRowCapacity = this.octreeProjection.info.pressureRowCapacity;
          this.info.pressureRequiredRows = this.octreeProjection.info.pressureRequiredRows;
          this.info.pressureCapacityOverflow = this.octreeProjection.info.pressureCapacityOverflow;
          this.info.frontierListCapacity = this.octreeProjection.info.frontierListCapacity;
          this.info.frontierRequiredLeaves = this.octreeProjection.info.frontierRequiredLeaves;
          this.info.frontierCapacityOverflow = this.octreeProjection.info.frontierCapacityOverflow;
          this.info.pressureSolver = this.octreeProjection.pressureSolverLabel;
    }
      return this.info;
    } finally {
      // A diagnostic promise can reject before mapAsync settles. The pooled
      // staging buffer cannot be copied again while it is pending or mapped.
      await mapPromise.catch(() => { /* Device loss is handled by the renderer. */ });
      if (buffer.mapState === "mapped") buffer.unmap();
      this.readbackPending = false;
    }
  }

  destroy() {
    this.disposed = true;
    if (this.quadtreeReadyProjection && this.quadtreeReadyProjection !== this.quadtreeProjection) this.quadtreeReadyProjection.destroy();
    this.quadtreeReadyProjection = undefined;
    this.quadtreeProjection?.destroySharedSurface();
    this.quadtreeProjection?.destroy();
    this.octreeProjection?.destroy();
    for (const projection of this.retiredQuadtreeProjections) projection.destroy();
    this.retiredQuadtreeProjections.clear();
    const textures = this.hostAllocation
      ? [this.velocityA, this.velocityB, this.velocityC, this.velocityD,
        this.pressureA, this.pressureB, this.volumeA, this.volumeB,
        this.heightA, this.heightB, this.terrainTexture,
        this.transportA, this.transportB, this.fluxScales]
      : [this.terrainTexture];
    for (const texture of new Set(textures)) texture.destroy();
    this.params?.destroy(); this.reductionBuffer?.destroy(); this.sharpenBuffer?.destroy(); this.rigidSystem.destroy(); this.rigidExchangeBuffer.destroy(); this.statsReadbackBuffer?.destroy();
  }
}

import { combineInitialBrickWet, damBreakBoxContains, initialFluidBrickContainsCell, sceneDamBreakBox, sceneDamBreakFractions } from "./initial-fluid";
import {
  CPUPerformanceTrace,
  GPUQueueWallPerformanceTraceRecorder,
  GPUStageTimestampRecorder,
  type GPUTimestampPhase,
  type PerformanceTrace,
} from "./performance-trace";
import {
  performanceShaderVariant,
  usePerformanceInstrumentationStore,
} from "./stores/performance-instrumentation-store";
import { usePerformanceActivityStore } from "./stores/performance-activity-store";
import {
  createGPULogicalActivityAdoptionContext,
  gpuLogicalActivityTaskDescriptions,
  stableGPULogicalActivityId,
  type GPULogicalActivityAdoptionContext,
  type GPULogicalActivityBindingSession,
  type GPULogicalActivityBindingSessionDiagnostics,
  type GPULogicalActivityTaskDescription,
} from "./gpu-logical-activity-adoption";
import {
  GPU_LOGICAL_ACTIVITY_HEADER_WORDS,
  GPU_LOGICAL_ACTIVITY_MAX_CAPACITY,
  GPU_LOGICAL_ACTIVITY_RECORD_BYTES,
  GPU_LOGICAL_ACTIVITY_UNKNOWN_U32,
  type GPULogicalActivityCapture,
  type GPULogicalActivityRecorder,
  type GPULogicalActivityTimeLocator,
} from "./gpu-logical-activity";
import {
  gpuPhysicsPerformanceActivityFrameId,
  type ActivityWorkIdentity,
} from "./performance-activity";
import { publishDecodedGPULogicalActivity } from "./gpu-performance-activity";
import { initializeRigidBodies, type RigidBodyState } from "./rigid-body";
import {
  GPU_RIGID_EXCHANGE_BYTES,
  type GPUEulerianInfo,
  type GPURigidLoad,
  type GPUVelocityTransport,
  type GPUQuality
} from "./webgpu-eulerian";
import type { SceneDescription } from "./model";
import type { SparseScenePrimitiveUpdate } from "./webgpu-sparse-scene-proxies";
import type { OctreeCoarseDynamicsConfiguration } from "./octree-coarse-backend";
import { createTallCellLayout } from "./tall-cell-grid";
import { sceneLatticeDimensions } from "./scene-lattice";
import { planGPUAdvance } from "./tall-cell-diagnostics";
import { averageInflowStrength, createInflowGridBoundary, type InflowGridBoundary } from "./inflow-boundary";
import { OCTREE_ALLOCATION_STAGES, OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES, WebGPUOctreeProjection,
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
import { planGPUShaderCapabilities } from "./gpu-shader-plan";
import {
  FINE_LEVELSET_VOLUME_VALID,
  unpackFineLevelSetGPUVolumeControl,
} from "./webgpu-octree-fine-levelset-volume";
import { supportsFluidM1MaxReduction } from "./webgpu-device-limits";
import { isOctreePersistentMGPCGSolverLabel } from "./webgpu-octree-section43-contract";
import {
  decodeStructuredProjectionEnergy,
  STRUCTURED_PROJECTION_ENERGY_WORDS,
} from "./webgpu-octree-structured-dynamics";
import { decodeOctreeStructuredRejectCarry } from "./octree-structured-reject-carry";
import {
  MAXIMUM_PENDING_PHYSICS_ADVANCES,
  StructuredStepSnapshotRing,
  structuredAuthorityStepHealth,
  structuredStepSnapshotSlotCount,
  structuredStepWorkObservation,
  type StructuredStepSnapshotRecord,
} from "./structured-step-snapshot";
import { losassoStepSnapshotFailures, WebGPUOctreeLosassoStepSnapshotRing }
  from "./webgpu-octree-losasso-step-snapshot";
import {
  OCTREE_STEP_PROGRAM,
  PhysicsStepPredictionLedger,
  StepSequenceRecorder,
  physicsStepPredictionFailures,
} from "./physics-step-program";

// Dense uniform simulation has been removed. This declaration keeps the
// unreachable bootstrap branch type-safe until the octree host is renamed.
const legacyUniformComputeShader = "";

/**
 * Step-snapshot record sources that `WebGPUOctreeProjection` does not expose
 * yet (P0.4 item 1). Adding `topologyEpochState`, `spgridLevelDelta`, and
 * `airSupportScratch` getters to `lib/webgpu-octree.ts` — each a one-line
 * delegation to `topologyEpoch.state`, `firstOrderVCycle` and
 * `airVelocitySupport.scratch` — completes the widened record; until then the
 * ring copies the segments it can reach and reports the rest absent.
 */
interface PendingStepSnapshotSources {
  readonly topologyEpochState?: GPUBuffer;
  readonly spgridLevelDelta?: GPUBuffer;
  readonly airSupportScratch?: GPUBuffer;
  readonly losassoAuthorityControl?: GPUBuffer;
  readonly losassoCoarsePhiControl?: GPUBuffer;
  readonly losassoExtensionControl?: GPUBuffer;
  readonly globalFineCurrentWorklist?: GPUBuffer;
}

export type UniformVelocityTransport = GPUVelocityTransport;
export interface WebGPUUniformEulerianOptions { pressureIterations?: number; velocityTransport?: UniformVelocityTransport; densitySharpening?: boolean; tallCellSettings?: Partial<import("./tall-cell-grid").TallCellSettings>; octree?: Partial<OctreeProjectionOptions>; /** Construction-time coarse authority selection. Never uploaded as a shader flag. */ coarseDynamics?: OctreeCoarseDynamicsConfiguration; /** Allocate escaped spray droplets and set their initial live state. */ secondaryParticles?: boolean; secondaryParticleCapacity?: number; deferPipelineCompilation?: boolean; /** Internal lifecycle channel for the worker-owned allocation graph. */ allocationProgress?: (label: string, completed: number, total: number) => void }

/** One shared-host unit followed by the octree-owned allocation stages. */
export const OCTREE_SOLVER_ALLOCATION_WORK_UNITS = 1 + OCTREE_ALLOCATION_STAGES.length;

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

/** Stage boundaries are free, but the trace's query set, resolve and map are
 * not. Sample at a debugging cadence rather than every advance. */
const PHYSICS_TRACE_CADENCE_MS = 100;
const PHYSICS_ACTIVITY_PHASE_CAPACITY = 256;
const PHYSICS_ACTIVITY_RECORD_CAPACITY = 4_096;
const PHYSICS_ACTIVITY_TARGET_BYTES = GPU_LOGICAL_ACTIVITY_HEADER_WORDS * 4
  + PHYSICS_ACTIVITY_RECORD_CAPACITY * GPU_LOGICAL_ACTIVITY_RECORD_BYTES;

/** Bounded sampled ledger (plus one equal staging buffer), adapter-limited. */
export function physicsLogicalActivityCaptureCapacity(
  limits: Pick<GPUSupportedLimits, "maxStorageBufferBindingSize" | "maxBufferSize">,
): number {
  const headerBytes = GPU_LOGICAL_ACTIVITY_HEADER_WORDS * 4;
  const bytes = Math.min(
    PHYSICS_ACTIVITY_TARGET_BYTES,
    Number(limits.maxStorageBufferBindingSize),
    Number(limits.maxBufferSize),
  );
  return Math.max(1, Math.min(
    GPU_LOGICAL_ACTIVITY_MAX_CAPACITY,
    Math.floor((bytes - headerBytes) / GPU_LOGICAL_ACTIVITY_RECORD_BYTES),
  ));
}

/** Maximum one-shot ledger permitted by both the activity ABI and adapter.
 * Captures start at the smaller sampled budget above and grow toward this only
 * after an observed overflow. */
export function maximumPhysicsLogicalActivityCaptureCapacity(
  limits: Pick<GPUSupportedLimits, "maxStorageBufferBindingSize" | "maxBufferSize">,
): number {
  const headerBytes = GPU_LOGICAL_ACTIVITY_HEADER_WORDS * 4;
  const bytes = Math.min(
    Number(limits.maxStorageBufferBindingSize),
    Number(limits.maxBufferSize),
  );
  return Math.max(1, Math.min(
    GPU_LOGICAL_ACTIVITY_MAX_CAPACITY,
    Math.floor((bytes - headerBytes) / GPU_LOGICAL_ACTIVITY_RECORD_BYTES),
  ));
}

const activityTaskId = (moduleId: string, task: string) =>
  stableGPULogicalActivityId(`task\0${moduleId}\0${task}`);
const activityCheckpointId = (moduleId: string, task: string, checkpoint: string) =>
  stableGPULogicalActivityId(`checkpoint\0${moduleId}\0${task}\0${checkpoint}`);

export const PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID =
  activityTaskId("physics/phase-boundaries", "physics-frame");
export const PHYSICS_ACTIVITY_FRAME_BEGIN_CHECKPOINT_ID =
  activityCheckpointId("physics/phase-boundaries", "physics-frame", "frame-begin");
export const PHYSICS_ACTIVITY_PHASE_BOUNDARY_CHECKPOINT_ID =
  activityCheckpointId("physics/phase-boundaries", "physics-frame", "phase-boundary");
export const PHYSICS_ACTIVITY_FRAME_END_CHECKPOINT_ID =
  activityCheckpointId("physics/phase-boundaries", "physics-frame", "frame-end");
export const PHYSICS_ACTIVITY_POWER_VOLUME_TASK_ID =
  activityTaskId("octree/power-volume", "publish-power-cell-volumes");

export interface GPUPhysicsLogicalActivityTaskDescriptor extends GPULogicalActivityTaskDescription {
  /** Measured timestamp phase which encloses this task; placement inside it is reconstructed. */
  readonly phaseId?: GPUTimestampPhase["id"];
}

const PHYSICS_ACTIVITY_TASKS: Readonly<Record<number, GPUPhysicsLogicalActivityTaskDescriptor>> = {
  [PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID]:
    {
      id: "gpu.physics.frame-contract",
      label: "Physics frame contract",
    },
  [activityTaskId("octree/fine-redistance-jfa", "jump-flood-a-to-b")]:
    {
      id: "gpu.physics.fine-redistance-jfa-a-b",
      label: "Fine redistance · jump flood A→B",
      phaseId: "fine-sdf-redistance",
    },
  [activityTaskId("octree/fine-redistance-jfa", "jump-flood-b-to-a")]:
    {
      id: "gpu.physics.fine-redistance-jfa-b-a",
      label: "Fine redistance · jump flood B→A",
      phaseId: "fine-sdf-redistance",
    },
  [activityTaskId("octree/fine-volume-correction", "apply-fine-volume-correction")]:
    {
      id: "gpu.physics.fine-volume-correction",
      label: "Fine volume correction",
      phaseId: "adaptive-publication",
    },
  [PHYSICS_ACTIVITY_POWER_VOLUME_TASK_ID]: {
    id: "gpu.physics.power-cell-volumes",
    label: "Power-cell volume publication",
    checkpoints: {
      enter: activityCheckpointId("octree/power-volume", "publish-power-cell-volumes", "enter"),
      exit: activityCheckpointId("octree/power-volume", "publish-power-cell-volumes", "exit"),
    },
  },
  [activityTaskId("octree/structured-publication", "classify-structured-catalog-slots")]:
    {
      id: "gpu.physics.structured-catalog-classification",
      label: "Structured catalog classification",
      phaseId: "coarse-grid",
    },
};

export interface GPUPhysicsLogicalActivitySample {
  readonly identity: ActivityWorkIdentity;
  readonly shaderGeneration: number;
  readonly capture: GPULogicalActivityCapture;
  readonly trace: PerformanceTrace;
  readonly lane: "gpu-physics";
  readonly clockDomain: "gpu-physics-timestamp";
  readonly windowStart_ms: 0;
  readonly windowEnd_ms: number;
  /** Cumulative measured phase durations. Tick n denotes the end of phase n. */
  readonly phaseBoundaries_ms: readonly number[];
  readonly locateTime: GPULogicalActivityTimeLocator;
  readonly tasks: Readonly<Record<number, GPUPhysicsLogicalActivityTaskDescriptor>>;
  /** Whole-frame acceptance result. Partial evidence remains useful, but no
   * consumer may interpret it as a complete utilization ledger when false. */
  readonly captureDiagnostics: GPUPhysicsLogicalActivityCaptureDiagnostics;
}

export type GPUPhysicsLogicalActivityIncompleteReason =
  | "recorder-overflow"
  | "missing-frame-begin"
  | "missing-frame-end"
  | "phase-marker-mismatch"
  | "unprofiled-dispatch"
  | "timestamp-fallback";

export interface GPUPhysicsLogicalActivityCaptureDiagnostics {
  readonly complete: boolean;
  readonly reasons: readonly GPUPhysicsLogicalActivityIncompleteReason[];
  readonly captureId: number;
  readonly retainedEventCount: number;
  readonly attemptedEventCount: number;
  readonly overflowed: boolean;
  readonly droppedEventCount: number;
  readonly computeDispatchCount: number;
  readonly instrumentedComputeDispatchCount: number;
  readonly unregisteredComputeDispatchCount: number;
  readonly unregisteredComputePipelineCount: number;
  readonly unregisteredComputePipelineLabels: readonly string[];
  readonly frameBeginObserved: boolean;
  readonly frameEndObserved: boolean;
  readonly expectedPhaseMarkerCount: number;
  readonly observedPhaseMarkerCount: number;
  readonly measurementSource: PerformanceTrace["measurementSource"];
}

const EMPTY_BINDING_SESSION_DIAGNOSTICS: GPULogicalActivityBindingSessionDiagnostics = {
  computeDispatchCount: 0,
  instrumentedComputeDispatchCount: 0,
  unregisteredComputeDispatchCount: 0,
  unregisteredComputePipelineCount: 0,
  unregisteredComputePipelineLabels: [],
};

/** Validate that retained evidence brackets exactly the owned physics command
 * graph and that every compute dispatch had an instrumented pipeline. */
export function validateGPUPhysicsLogicalActivityCapture(
  capture: GPULogicalActivityCapture,
  trace: Pick<PerformanceTrace, "measurementSource" | "phases">,
  dispatches: GPULogicalActivityBindingSessionDiagnostics = EMPTY_BINDING_SESSION_DIAGNOSTICS,
  expectedPhaseMarkerCount = trace.phases.length,
): GPUPhysicsLogicalActivityCaptureDiagnostics {
  const first = capture.events[0];
  const last = capture.events.at(-1);
  const frameBeginCount = capture.events.filter((candidate) =>
    candidate.taskId === PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID
    && candidate.checkpointId === PHYSICS_ACTIVITY_FRAME_BEGIN_CHECKPOINT_ID).length;
  const frameEndCount = capture.events.filter((candidate) =>
    candidate.taskId === PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID
    && candidate.checkpointId === PHYSICS_ACTIVITY_FRAME_END_CHECKPOINT_ID).length;
  const frameBeginObserved = first?.taskId === PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID
    && first.checkpointId === PHYSICS_ACTIVITY_FRAME_BEGIN_CHECKPOINT_ID
    && frameBeginCount === 1;
  const frameEndObserved = last?.taskId === PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID
    && last.checkpointId === PHYSICS_ACTIVITY_FRAME_END_CHECKPOINT_ID
    && frameEndCount === 1;
  const phaseMarkers = capture.events.filter((candidate) =>
    candidate.taskId === PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID
    && candidate.checkpointId === PHYSICS_ACTIVITY_PHASE_BOUNDARY_CHECKPOINT_ID);
  const observedPhaseMarkerCount = phaseMarkers.length;
  const phaseMarkerChainMatches = observedPhaseMarkerCount === expectedPhaseMarkerCount
    && phaseMarkers.every((marker, index) => marker.tick === index);
  const reasons: GPUPhysicsLogicalActivityIncompleteReason[] = [];
  if (capture.overflowed || capture.droppedEventCount > 0) reasons.push("recorder-overflow");
  if (!frameBeginObserved) reasons.push("missing-frame-begin");
  if (!frameEndObserved) reasons.push("missing-frame-end");
  if (!phaseMarkerChainMatches) {
    reasons.push("phase-marker-mismatch");
  }
  if (dispatches.unregisteredComputeDispatchCount > 0) {
    reasons.push("unprofiled-dispatch");
  }
  if (trace.measurementSource === "gpu-queue-wall") reasons.push("timestamp-fallback");
  return Object.freeze({
    complete: reasons.length === 0,
    reasons: Object.freeze(reasons),
    captureId: capture.captureId,
    retainedEventCount: capture.events.length,
    attemptedEventCount: capture.events.length + capture.droppedEventCount,
    overflowed: capture.overflowed,
    droppedEventCount: capture.droppedEventCount,
    computeDispatchCount: dispatches.computeDispatchCount,
    instrumentedComputeDispatchCount: dispatches.instrumentedComputeDispatchCount,
    unregisteredComputeDispatchCount: dispatches.unregisteredComputeDispatchCount,
    unregisteredComputePipelineCount: dispatches.unregisteredComputePipelineCount,
    unregisteredComputePipelineLabels: Object.freeze([
      ...dispatches.unregisteredComputePipelineLabels,
    ]),
    frameBeginObserved,
    frameEndObserved,
    expectedPhaseMarkerCount,
    observedPhaseMarkerCount,
    measurementSource: trace.measurementSource,
  });
}

/**
 * Project explicit shader phase ticks into the timestamp trace's local clock.
 * The timestamp durations are measured; placement within that partition is
 * reconstructed. Append sequence is used only to bracket a heartbeat between
 * explicit command-ordered marker passes, never as a duration or shader clock.
 */
export function physicsPhaseBoundaryTimeProjection(
  trace: PerformanceTrace,
  capture?: GPULogicalActivityCapture,
  tasks: Readonly<Record<number, GPUPhysicsLogicalActivityTaskDescriptor>> = PHYSICS_ACTIVITY_TASKS,
): {
  phaseBoundaries_ms: readonly number[];
  locateTime: GPULogicalActivityTimeLocator;
} {
  let cursor_ms = 0;
  const phaseRanges = trace.phases.map((phase) => {
    const start_ms = cursor_ms;
    cursor_ms = Math.min(trace.total_ms, cursor_ms + Math.max(0, phase.duration_ms));
    return { id: phase.id, start_ms, end_ms: cursor_ms };
  });
  const phaseBoundaries_ms = phaseRanges.map((phase) => phase.end_ms);
  const markers = (capture?.events ?? [])
    .filter((event) => event.taskId === PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID
      && event.checkpointId === PHYSICS_ACTIVITY_PHASE_BOUNDARY_CHECKPOINT_ID
      && event.tick !== undefined)
    .sort((left, right) => left.sequence - right.sequence);
  const maximumMarkerTick = markers.reduce(
    (maximum, marker) => Math.max(maximum, marker.tick ?? -1),
    -1,
  );
  const markerBoundary_ms = (tick: number): number | undefined => {
    if (trace.measurementSource !== "gpu-queue-wall") return phaseBoundaries_ms[tick];
    // Queue completion has no semantic subphase timestamps. Preserve only the
    // command-ordered marker partition and label it reconstructed; equal
    // spacing is an explicit visualization fallback, never measured duration.
    return maximumMarkerTick >= 0
      ? trace.total_ms * (tick + 1) / (maximumMarkerTick + 1)
      : undefined;
  };
  return {
    phaseBoundaries_ms,
    locateTime: (event) => {
      if (event.taskId === PHYSICS_ACTIVITY_PHASE_MARKER_TASK_ID) {
        if (event.checkpointId === PHYSICS_ACTIVITY_FRAME_BEGIN_CHECKPOINT_ID) {
          return { time_ms: 0, evidence: "reconstructed" };
        }
        if (event.checkpointId === PHYSICS_ACTIVITY_FRAME_END_CHECKPOINT_ID) {
          return { time_ms: trace.total_ms, evidence: "reconstructed" };
        }
        if (event.checkpointId !== PHYSICS_ACTIVITY_PHASE_BOUNDARY_CHECKPOINT_ID) {
          return undefined;
        }
        const boundary_ms = event.tick === undefined ? undefined : markerBoundary_ms(event.tick);
        return boundary_ms === undefined
          ? undefined
          : { time_ms: boundary_ms, evidence: "reconstructed" };
      }
      if (event.taskId === PHYSICS_ACTIVITY_POWER_VOLUME_TASK_ID && event.tick !== undefined) {
        const boundary_ms = markerBoundary_ms(event.tick);
        if (boundary_ms !== undefined) {
          return { time_ms: boundary_ms, evidence: "reconstructed" };
        }
      }
      const nextMarker = markers.find((marker) => marker.sequence > event.sequence);
      const previousMarker = markers.findLast((marker) => marker.sequence < event.sequence);
      const enclosingTick = nextMarker?.tick
        ?? (previousMarker?.tick === undefined ? undefined : previousMarker.tick + 1);
      if (enclosingTick !== undefined) {
        const end_ms = markerBoundary_ms(enclosingTick);
        const start_ms = enclosingTick === 0 ? 0 : markerBoundary_ms(enclosingTick - 1);
        if (start_ms !== undefined && end_ms !== undefined && end_ms >= start_ms) {
          return { time_ms: (start_ms + end_ms) / 2, evidence: "reconstructed" };
        }
      }
      const descriptor = tasks[event.taskId];
      if (!descriptor?.phaseId) return undefined;
      // A shader heartbeat proves progress somewhere inside its measured
      // enclosing phase, but WGSL supplies no clock. Put the point at the
      // phase midpoint and keep reconstructed evidence; do not turn an
      // enter/exit pair into a full-phase occupancy claim.
      const enclosing = phaseRanges
        .filter((phase) => phase.id === descriptor.phaseId && phase.end_ms > phase.start_ms)
        .sort((left, right) => (right.end_ms - right.start_ms) - (left.end_ms - left.start_ms))[0];
      return enclosing
        ? { time_ms: (enclosing.start_ms + enclosing.end_ms) / 2, evidence: "reconstructed" }
        : undefined;
    },
  };
}

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
  /** Full 11-word reject carry (control words 0..10) including the stage-1/2
   * detail vec4 and workset class; absent on older readback layouts. */
  readonly structuredRejectCarry?: readonly number[];
  /** Section 5 air-support publication control (16 words, arena layout). */
  readonly airSupportControl?: readonly number[];
  /** recordArena[13..14]: identity/detail of the first stage-8 support
   * reconstruction failure, when one was recorded. */
  readonly firstAirSupportFailure?: readonly number[];
  /** recordArena[13..15] captured one-shot by the NEXT begin: the preceding
   * failed transaction's terminal flags/detail/layout. */
  readonly precedingAirSupportTerminal?: readonly number[];
  /** Failure-only CPU mirror of the rejected stage-6 leaf and its complete
   * paper face/edge neighborhood. It is absent on clean publications. */
  readonly airSupportFailureTopology?: Readonly<Record<string, unknown>>;
  /** scratch[51..59]: immutable identity/reason captured by the failure's
   * finalize pass before a later transaction may reuse its record slot. */
  readonly airSupportTopologyFailureLatch?: readonly number[];
  /** scratch[41..42]: stationary-air fallback patch count and first
   * (cell<<3)|axis identity from the most recent march. */
  readonly airSupportFallbacks?: readonly number[];
  /** Fine transport governor state[0..7] per bank: schedule-invalid word,
   * active substeps, substep dt bits, measured displacement. */
  readonly fineTransportScheduleA?: readonly number[];
  readonly fineTransportScheduleB?: readonly number[];
  /** Fine transport governor state[46..56] per bank: first-schedule latch,
   * repairs, rows, support, sleeping bit, delta count, why-not-sleeping
   * bitmask, displacement-cells bits, rows/support/pages census. */
  readonly fineTransportSleepA?: readonly number[];
  readonly fineTransportSleepB?: readonly number[];
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
  const section43MGPCG = isOctreePersistentMGPCGSolverLabel(value.solverLabel);
  if (!value.authoritative || !section43MGPCG) {
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
  return { ready: true, label: `Power pressure published (${value.pressureRows} rows)` };
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

/** Octree simulation host. Dense fields remain only as bootstrap resources. */
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
  private structuredFreezeDumpedEpoch = -1;
  private airSupportFailureLoggedWord: number | undefined;
  private airSupportFallbackLoggedCount: number | undefined;
  private structuredProbeCounter = 0;
  private structuredRowCounterRaces = 0;
  private structuredLagLoggedFine = -1;
  private structuredPublicationFailureDumped = false;
  private physicsTraceSampleId = 0;
  private physicsTracePending = false;
  private lastPhysicsTraceAt_ms = -Infinity;
  private hardwarePhysicsTraceInvalid = false;
  /** Step-coherent structured diagnostics; written by the step's own encoder. */
  private stepSnapshotRing?: StructuredStepSnapshotRing;
  private losassoStepSnapshotRing?: WebGPUOctreeLosassoStepSnapshotRing;
  private readonly stepSequenceRecorder = new StepSequenceRecorder();
  private stepSequenceFaulted = false;
  /**
   * P0.5 lag-k audit: what each step's encode predicted, resolved when that
   * step's own snapshot maps. Sized past the maximum pipeline depth so a
   * prediction is still held when its record arrives.
   */
  private readonly stepPredictions =
    new PhysicsStepPredictionLedger(2 * MAXIMUM_PENDING_PHYSICS_ADVANCES);
  private stepPredictionFaulted = false;
  private stepSnapshotFaulted = false;
  private readonly logicalActivity: GPULogicalActivityAdoptionContext;
  private logicalActivityMarkerPipeline?: GPUComputePipeline;
  private logicalActivityMarkerTickSource?: GPUBuffer;
  private logicalActivityMarkerTick?: GPUBuffer;
  private logicalActivityMarkerGroup?: GPUBindGroup;
  /** Permanent overflow-safe sink that keeps instrumented pipelines valid between retained samples. */
  private logicalActivityDiscardRecorder?: GPULogicalActivityRecorder;
  private logicalActivityCaptureId = 0;
  private logicalActivityCaptureCapacity = PHYSICS_ACTIVITY_RECORD_CAPACITY;
  private latestLogicalActivity?: GPUPhysicsLogicalActivitySample;
  private latestLogicalActivityCaptureDiagnostics?: GPUPhysicsLogicalActivityCaptureDiagnostics;
  private validationChecked = false;
  private validationPromise?: Promise<void>;
  // Not readonly: a warm re-seed rebuilds the boundary for the new nozzle.
  private inflowBoundary?: InflowGridBoundary;
  private readonly velocityTransport: UniformVelocityTransport;
  private readonly densitySharpening: boolean;
  private readonly hostAllocation?: UniformHostAllocationPlan;
  private readonly transportConservativeVolume: boolean;
  private octreeProjection?: WebGPUOctreeProjection;
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
    options.allocationProgress?.(
      "Allocate shared solver and rigid-body resources", 0, OCTREE_SOLVER_ALLOCATION_WORK_UNITS,
    );
    // The octree has one measured executable profile. Reject it before the
    // first texture or buffer allocation so unsupported adapters cannot leave
    // a partially constructed graph or trigger compilation of another lane.
    if (options.octree && (!device.features.has("subgroups")
      || !supportsFluidM1MaxReduction(device.limits))) {
      throw new Error("Power octree requires the M1 Max 128-lane subgroup profile");
    }
    this.logicalActivity = createGPULogicalActivityAdoptionContext({
      moduleId: "physics/phase-boundaries",
      profile: performanceShaderVariant(),
      identity: "workgroup",
    });
    const c = scene.container, matched = createTallCellLayout(scene, quality, device.limits.maxTextureDimension3D, options.tallCellSettings);
    const nx = matched.nx, ny = matched.fineNy, nz = matched.nz;
    this.velocityTransport = options.velocityTransport ?? "maccormack";
    this.densitySharpening = options.densitySharpening ?? true;
    this.hostAllocation = options.octree
      ? undefined
      : planUniformHostAllocation(nx, ny, nz, this.velocityTransport);
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
    this.rigidSystem = new WebGPURigidBodySystem(device, scene, this.rigidExchangeBuffer,
      this.terrainTexture, options.deferPipelineCompilation);
    this.rigidBuffer = this.rigidSystem.stateBuffer;
    this.rigidSystem.syncBodies(initializeRigidBodies(scene.rigidBodies));
    let prepLayout: GPUBindGroupLayout | undefined;
    if (this.hostAllocation) {
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
    if (options.octree) {
      this.octreeProjection = new WebGPUOctreeProjection(device, scene, { nx, ny, nz }, {
        rigidBodies: this.rigidBuffer, rigidExchange: this.rigidExchangeBuffer, terrain: this.terrainTexture,
      }, {
        maximumLeafSize: options.octree.maximumLeafSize ?? 16,
        adaptivity: options.octree.adaptivity ?? 1,
        interfaceRefinementBandCells: options.octree.interfaceRefinementBandCells ?? 4,
        surfaceRefinementGradingLayers: options.octree.surfaceRefinementGradingLayers ?? 1,
        // Left undefined the projection falls back to the pressure band, which
        // is the width every lane was measured at before the two separated.
        fineLevelSetBandCells: options.octree.fineLevelSetBandCells,
        globalFineLevelSetFactor: options.octree.globalFineLevelSetFactor ?? 4,
        globalFineLevelSetMaximumBricks: options.octree.globalFineLevelSetMaximumBricks,
        pressureRowCapacity: options.octree.pressureRowCapacity,
        coarseDynamics: options.coarseDynamics,
      }, options.deferPipelineCompilation, (label, completed) => options.allocationProgress?.(
        label, completed + 1, OCTREE_SOLVER_ALLOCATION_WORK_UNITS,
      ));
      this.applyOctreeInfo(this.octreeProjection);
    }
    // The octree's resident level set is the complete liquid state. Keep VOF
    // transport only for the uniform solver and quadtree catastrophe recovery.
    this.transportConservativeVolume = !this.octreeProjection;
    // The compact octree owns a separate pipeline graph. Shared dense bind
    // groups exist only for the uniform and quadtree methods.
    if (this.hostAllocation) {
    const surfaceAuthority = this.volumeA;
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
    const correctionSurfaceAuthority = this.volumeB;
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
    }
    if (this.octreeProjection && !options.deferPipelineCompilation) this.publishInitialSparseScene();
  }

  private async createLogicalActivityMarker(): Promise<void> {
    if (this.logicalActivityMarkerPipeline) return;
    const frameBeginHeartbeat = this.logicalActivity.workgroup("physics-frame", "frame-begin", {
      workgroupLaneCount: 1,
    });
    const phaseBoundaryHeartbeat = this.logicalActivity.workgroup("physics-frame", "phase-boundary", {
      tick: "marker.tick",
      workgroupLaneCount: 1,
    });
    const frameEndHeartbeat = this.logicalActivity.workgroup("physics-frame", "frame-end", {
      workgroupLaneCount: 1,
    });
    const markerShader = this.logicalActivity.module(/* wgsl */ `
struct PhysicsActivityMarker { tick: u32, kind: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<uniform> marker: PhysicsActivityMarker;

@compute @workgroup_size(1)
fn recordPhysicsPhaseBoundary(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32,
) {
  if (marker.kind == 0u) {
    ${frameBeginHeartbeat}
  } else if (marker.kind == 1u) {
    ${phaseBoundaryHeartbeat}
  } else {
    ${frameEndHeartbeat}
  }
}
`, "physics-phase-boundary-marker");
    const activityModule = this.device.createShaderModule({
      label: "Physics logical activity phase marker",
      code: markerShader.code,
    });
    this.logicalActivityMarkerPipeline = this.logicalActivity.registerPipeline(
      await this.device.createComputePipelineAsync({
        label: "Physics logical activity phase marker",
        layout: "auto",
        compute: { module: activityModule, entryPoint: "recordPhysicsPhaseBoundary" },
      }),
    );
    this.logicalActivityMarkerTickSource = this.device.createBuffer({
      label: "Physics logical activity phase ticks",
      size: (PHYSICS_ACTIVITY_PHASE_CAPACITY + 2) * 8,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const markerRecords = new Uint32Array((PHYSICS_ACTIVITY_PHASE_CAPACITY + 2) * 2);
    markerRecords.set([GPU_LOGICAL_ACTIVITY_UNKNOWN_U32, 0], 0);
    for (let index = 0; index < PHYSICS_ACTIVITY_PHASE_CAPACITY; index += 1) {
      markerRecords.set([index, 1], (index + 1) * 2);
    }
    markerRecords.set(
      [GPU_LOGICAL_ACTIVITY_UNKNOWN_U32, 2],
      (PHYSICS_ACTIVITY_PHASE_CAPACITY + 1) * 2,
    );
    this.device.queue.writeBuffer(
      this.logicalActivityMarkerTickSource,
      0,
      markerRecords,
    );
    this.logicalActivityMarkerTick = this.device.createBuffer({
      label: "Physics logical activity current phase tick",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.logicalActivityMarkerGroup = this.device.createBindGroup({
      label: "Physics logical activity phase marker",
      layout: this.logicalActivityMarkerPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.logicalActivityMarkerTick } }],
    });
  }

  private encodeLogicalActivityMarker(
    encoder: GPUCommandEncoder,
    recordIndex: number,
    label: string,
  ): void {
    if (!this.logicalActivityMarkerPipeline || !this.logicalActivityMarkerTickSource
      || !this.logicalActivityMarkerTick || !this.logicalActivityMarkerGroup) return;
    encoder.copyBufferToBuffer(
      this.logicalActivityMarkerTickSource,
      recordIndex * 8,
      this.logicalActivityMarkerTick,
      0,
      8,
    );
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(this.logicalActivityMarkerPipeline);
    pass.setBindGroup(0, this.logicalActivityMarkerGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  private encodeLogicalActivityFrameBegin(encoder: GPUCommandEncoder): void {
    this.encodeLogicalActivityMarker(encoder, 0, "Physics activity frame begin");
  }

  private encodeLogicalActivityPhaseBoundary(encoder: GPUCommandEncoder, phaseIndex: number): void {
    if (phaseIndex < 0 || phaseIndex >= PHYSICS_ACTIVITY_PHASE_CAPACITY) return;
    this.encodeLogicalActivityMarker(
      encoder,
      phaseIndex + 1,
      `Physics activity boundary ${phaseIndex}`,
    );
  }

  private encodeLogicalActivityFrameEnd(encoder: GPUCommandEncoder): void {
    this.encodeLogicalActivityMarker(
      encoder,
      PHYSICS_ACTIVITY_PHASE_CAPACITY + 1,
      "Physics activity frame end",
    );
  }

  /** Latest completed, generation-checked capture; retained until superseded. */
  get physicsLogicalActivity(): GPUPhysicsLogicalActivitySample | undefined {
    return this.latestLogicalActivity;
  }

  /** Latest whole-frame acceptance verdict, including incomplete captures. */
  get physicsLogicalActivityCaptureDiagnostics(): GPUPhysicsLogicalActivityCaptureDiagnostics | undefined {
    return this.latestLogicalActivityCaptureDiagnostics;
  }

  private nextLogicalActivityCaptureId(): number {
    this.logicalActivityCaptureId = (this.logicalActivityCaptureId + 1) >>> 0;
    if (this.logicalActivityCaptureId === 0) this.logicalActivityCaptureId = 1;
    return this.logicalActivityCaptureId;
  }

  private pipelineDescriptor(entryPoint:string,prep=false):GPUComputePipelineDescriptor{
    this.shaderModule??=this.device.createShaderModule({label:"Fluid Lab uniform reference kernels",code:legacyUniformComputeShader});
    return{layout:prep?this.prepPipelineLayout:this.pipelineLayout,compute:{module:this.shaderModule,entryPoint}};
  }
  private assignPipelines(compiled:GPUComputePipeline[]){this.advectPipeline=compiled[0];this.reversePipeline=compiled[1];this.correctPipeline=compiled[2];this.jacobiPipeline=compiled[3];this.projectPipeline=compiled[4];this.rigidPipeline=compiled[5];this.relaxSolidPhiPipeline=compiled[6];this.reductionPipeline=compiled[7];this.buildOccupancyPipeline=compiled[8];this.buildTransportPipeline=compiled[9];this.buildFluxScalesPipeline=compiled[10];this.sharpenComputePipeline=compiled[11];this.sharpenScatterPipeline=compiled[12];this.sharpenResolvePipeline=compiled[13];}
  static async createAsync(device:GPUDevice,scene:SceneDescription,quality:GPUQuality,onRigidLoads:((loads:GPURigidLoad[])=>void)|undefined,options:WebGPUUniformEulerianOptions,onProgress:(label:string,completed:number,total:number,phase?:string,taskId?:string)=>void,signal:AbortSignal=new AbortController().signal){
    const runner=new GPUInitializationTaskRunner((snapshot)=>onProgress(snapshot.label,snapshot.completed,snapshot.total,snapshot.phase,snapshot.taskId),signal);
    let solver:WebGPUUniformEulerianSolver|undefined;
    try{
      const capabilityPlan=planGPUShaderCapabilities(scene,{
        solver:"octree",
        fineInterface:Boolean(options.octree),
        logicalActivity:performanceShaderVariant().enabled,
      });
      await runner.run([
        {id:"solver.capabilities",phase:"planning",label:`Resolve ${capabilityPlan.values.size} scene-required GPU capabilities`,run:()=>{}},
        {id:"solver.allocate",phase:"allocation",label:"Allocate octree solver resources",dependencies:["solver.capabilities"],
          workUnits:OCTREE_SOLVER_ALLOCATION_WORK_UNITS,
          run:(_signal,report)=>{solver=new WebGPUUniformEulerianSolver(device,scene,quality,onRigidLoads,{...options,deferPipelineCompilation:true,
            allocationProgress:(label,completed)=>report?.(label,completed)});}},
      ]);
      await runner.run(solver!.initializationTasks());
      return solver!;
    }catch(error){solver?.destroy();throw error;}
  }
  private initializationTasks():GPUInitializationTask[]{
    const tasks:GPUInitializationTask[]=[...this.rigidSystem.initializationTasks()];
    if (this.logicalActivity.enabled) tasks.push({
      id: "uniform.pipeline.logical-activity-marker", phase: "solver-pipelines",
      label: "Compile physics logical-activity marker",
      run: () => this.createLogicalActivityMarker(),
    });
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
    if(this.octreeProjection)tasks.push(...this.octreeProjection.initializationTasks());
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
    const rawInitialSparseScene = this.device.createCommandEncoder({
      label: `Initial sparse authority: ${descriptor.label}`,
    });
    const discardActivity = this.logicalActivity.enabled
      ? this.logicalActivity.bindingSession(this.device, rawInitialSparseScene, {
        capacity: 1,
        captureId: this.nextLogicalActivityCaptureId(),
        label: `Discarded startup activity: ${descriptor.label}`,
      })
      : undefined;
    const initialSparseScene = discardActivity?.encoder ?? rawInitialSparseScene;
    try {
      this.octreeProjection.encodeInitialSparseAuthorityPhase(initialSparseScene, phase);
      discardActivity?.finish();
      this.device.queue.submit([initialSparseScene.finish()]);
      this.octreeProjection.retireSubmittedEncoder(initialSparseScene);
      const discardRead = discardActivity?.read();
      await this.device.queue.onSubmittedWorkDone();
      await discardRead?.catch(() => undefined);
      discardActivity?.destroy();
      const validationError = await this.device.popErrorScope();
      validationScopeOpen = false;
      if (validationError) {
        throw new Error(`Initial sparse authority ${phase} validation failed: ${validationError.message}`);
      }
    } catch (error) {
      discardActivity?.destroy();
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
    const reject = decodeOctreeStructuredRejectCarry(
      (value.structuredRejectCarry?.length ?? 0) >= 2 ? value.structuredRejectCarry! : velocity);
    this.info.structuredRejectStage = reject.stage;
    this.info.structuredRejectIndex = reject.index;
    this.info.structuredRejectSummary = reject.clean ? undefined : reject.summary;
    // A failed Section 5 air-support publication rejects EVERY advect lane
    // next step (supportPublicationValid gate) and freezes the epoch; name
    // the producer's own error record once per distinct failure word.
    const support = value.airSupportControl;
    this.info.structuredAirSupportRows = support?.[5] ?? 0;
    this.info.structuredAirSupportCells = support?.[6] ?? 0;
    this.info.structuredAirSupportCapacity = support?.[7] ?? 0;
    this.info.structuredAirSupportFaceItems = support?.[10] ?? 0;
    this.info.structuredAirSupportSeedFaces = support?.[11] ?? 0;
    this.info.structuredAirSupportMarchDepth = support?.[12] ?? 0;
    const latched = value.firstAirSupportFailure ?? [];
    const liveFailure = !!support && support.length >= 16
      && (support[0] !== 0 || (support[1] ?? 0xffff_ffff) !== 0xffff_ffff);
    const latchedFailure = (latched[0] ?? 0) !== 0;
    const failureWord = liveFailure ? support![1] : latchedFailure ? latched[1] : undefined;
    this.info.structuredAirSupportFailureFlags = liveFailure
      ? support![0] : latchedFailure ? latched[0] : undefined;
    this.info.structuredAirSupportFailureItem = failureWord;
    if ((liveFailure || latchedFailure)
      && this.airSupportFailureLoggedWord !== failureWord) {
      this.airSupportFailureLoggedWord = failureWord;
      console.error("[air-support-failure]", JSON.stringify({
        live: liveFailure, errors: support?.[0], firstError: support?.[1],
        directRows: support?.[5], supportCount: support?.[6],
        faceItems: support?.[10], seeds: support?.[11], maxWaves: support?.[12],
        latchedFlags: latched[0], latchedFirstError: latched[1],
        precedingTerminal: value.precedingAirSupportTerminal,
        topologyFailure: value.airSupportFailureTopology,
        // The transport governor's schedule + sleep forensics per bank: an
        // uncommitted transport delta is the recurring-band clause-512 root,
        // and sleep word 6 is the why-not-sleeping bitmask that names the
        // blocking term (1 never-scheduled, 2 repairs, 4 governor-changed,
        // 8 displacement, 64 schedule-invalid).
        transportScheduleA: value.fineTransportScheduleA,
        transportSleepA: value.fineTransportSleepA,
        transportScheduleB: value.fineTransportScheduleB,
        transportSleepB: value.fineTransportSleepB }));
    }
    const fallbacks = value.airSupportFallbacks;
    if (fallbacks && (fallbacks[0] ?? 0) > 0
      && this.airSupportFallbackLoggedCount !== fallbacks[0]) {
      this.airSupportFallbackLoggedCount = fallbacks[0];
      // Word 42 is the lowest carrier-free owned-face ITEM index. It used to
      // be a (flags<<16)|(cell<<3)|axis packing whose flags byte aliased the
      // cell index on every domain wider than 8192 cells, so neither the cell
      // nor the demand flags decoded here were trustworthy. The producer now
      // expands the item into its own forensic block; the item's row/patch
      // split is the only thing derivable without that block.
      const item = fallbacks[1] ?? 0xffff_ffff;
      const local = item === 0xffff_ffff ? undefined : item % 12;
      console.error("[air-support-fallback]", JSON.stringify({
        patches: fallbacks[0], firstItem: item === 0xffff_ffff ? undefined : item,
        firstFaceRow: item === 0xffff_ffff ? undefined : Math.floor(item / 12),
        firstAxis: local === undefined ? undefined : Math.floor(local / 4),
        firstQuadrant: local === undefined ? undefined : local % 4 }));
    }
    this.info.structuredBoundaryGeneration = boundary[4] ?? 0;
    this.info.structuredBoundaryValid = boundary.length >= 7 && boundary[0] === 0
      && boundary[2] === velocity[2] && boundary[4] === velocity[3]
      && boundary[5] === velocity[4] && boundary[6] === velocity[3];
    // Accepted structured control is the GPU-owned coupled-epoch receipt.
    // Never relabel the host's newer attempt stamp as a published generation.
    this.info.powerDiagramGeneration = this.info.structuredVelocityValid
      && this.info.structuredBoundaryValid ? velocity[3] : undefined;
  }

  /**
   * P0.5 lag-k check: step N's own snapshot has mapped, so compare the GPU's
   * counters against the prediction the driver recorded when it shaped step
   * N's encode. The comparison is pure audit — the step is long submitted —
   * but a mismatch means a launch-shape predicate deleted live work, which is
   * the one thing the carve-out in the driver contract forbids.
   *
   * Latched per run like the sequence check: one dishonest step invalidates
   * the predicate, so later conforming steps must not clear it.
   */
  private resolveStepPrediction(record: StructuredStepSnapshotRecord) {
    const prediction = this.stepPredictions.take(record.stamp.step);
    if (!prediction) return;
    const observation = structuredStepWorkObservation(record);
    const failures = physicsStepPredictionFailures(prediction, {
      step: observation.step,
      executedSolveIterations: observation.executedSolveIterations,
      solveConverged: observation.solveConverged,
      topologyFlipReady: observation.topologyFlipReady,
      topologyEpochError: observation.topologyEpochError,
      spgridLevelDirty: observation.spgridLevelDirty,
      fineActiveBricks: observation.fineActiveBricks,
      airSupportErrorFlags: observation.airSupportErrorFlags,
    }, OCTREE_STEP_PROGRAM);
    this.info.structuredSnapshotExecutedSolveIterations = observation.executedSolveIterations;
    this.info.structuredSnapshotSolveConverged = observation.solveConverged;
    if (failures.length === 0 || this.stepPredictionFaulted) return;
    this.stepPredictionFaulted = true;
    this.info.stepPredictionFailures = [...failures];
    console.error("[step-prediction]", JSON.stringify({
      step: prediction.step, encodedSolveBudget: prediction.encodedSolveBudget,
      skipped: prediction.skippedStageIds, conditions: prediction.conditions,
      failures,
    }));
  }

  /** The paper path must be complete before the first trajectory can be
   * requested. These are one-time post-fence readbacks for UI readiness and
   * diagnostics; recurring frame scheduling remains GPU-resident. */
  private async validateInitialSparseAuthority() {
    const projection = this.octreeProjection;
    if (!projection) throw new Error("Initial sparse authority requires an octree projection");
    if (projection.coarseBackend === "losasso") {
      const reduced = await projection.readLosassoAuthorityDiagnostics();
      this.applyOctreeInfo(projection);
      const authority = reduced?.authority ?? [];
      const solver = reduced?.solver ?? [];
      const coarsePhi = reduced?.coarsePhi ?? [];
      const ready = authority.length >= 5 && authority[0] !== 0
        && authority[1] > 0 && authority[2] > 0
        && authority[3] === 1 && authority[4] === 0;
      const solverReady = solver.length >= 3 && solver[0] === 0 && solver[1] !== 0;
      if (!ready || !solverReady) {
        const owner = await projection.readOwnerPageControl();
        throw new Error("Paused t=0 Losasso authority rejected: authority="
          + JSON.stringify(authority) + "; solver=" + JSON.stringify(solver)
          + "; coarsePhi=" + JSON.stringify(coarsePhi)
          + "; owner=" + JSON.stringify(owner));
      }
      this.info.quadtreePressureConverged = solverReady;
      this.info.quadtreePressureIterationsUsed = solver[2] ?? 0;
      return;
    }
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
  get rigidRenderBuffer() { return this.rigidSystem.renderBuffer; }
  get rigidMotionBuffer() { return this.rigidSystem.motionBuffer; }
  setSelectedRigidBody(index: number) { this.rigidSystem.setSelectedIndex(index); }
  pickRigidBody(origin: RigidBodyState["position_m"], direction: RigidBodyState["position_m"]) { return this.rigidSystem.pick(origin,direction); }
  // Rendering contours the smooth resident level set when the quadtree
  // projection maintains one; the flux-form VOF field is near-binary and its
  // 0.5 contour is quantized to cell scale. Diagnostics keep reading the VOF
  // field through volumeTexture.
  get surfaceFieldTexture() { return this.octreeProjection?.levelSetTexture ?? this.volumeA; }
  /** False once global-fine publication has retired the dense bootstrap phi. */
  get hasDenseSurfaceField() { return this.octreeProjection?.hasDenseLevelSetPublication ?? true; }
  get sparseVoxelSceneSource() { return this.octreeProjection?.sparseVoxelSceneSource; }
  stageSceneUpdate(scene: SceneDescription) { this.octreeProjection?.stageSceneUpdate(scene); }
  stageLivePrimitiveUpdates(updates: readonly SparseScenePrimitiveUpdate[]) {
    return this.octreeProjection?.stageLivePrimitiveUpdates(updates) ?? false;
  }
  encodeSceneMaintenance(encoder: GPUCommandEncoder) { this.octreeProjection?.encodeSceneMaintenance(encoder); }
  get sparseVoxelRenderSource() {
    const source = this.octreeProjection?.sparseVoxelRenderSource;
    if (this.octreeProjection) this.applyOctreeInfo(this.octreeProjection);
    return source;
  }
  get structuredVelocityControl() { return this.octreeProjection?.structuredVelocityControl; }
  get structuredBoundaryControl() { return this.octreeProjection?.structuredBoundaryControl; }
  get structuredRowVelocities() { return this.octreeProjection?.structuredRowVelocities; }
  get losassoVelocityDebug() { return this.octreeProjection?.losassoVelocityDebug; }
  get losassoPressureDebug() { return this.octreeProjection?.losassoPressureDebug; }
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
  get coarseLevelSetSource() { return this.octreeProjection?.coarseLevelSetSource; }
  /** QA-only passthrough for reproducing recurring frontier phase decisions. */
  get globalFineSummaryDirectory() { return this.octreeProjection?.globalFineSummaryDirectory; }
  get globalFineSummaryDebug() { return this.octreeProjection?.globalFineSummaryDebug; }
  get globalFineTransportControl() { return this.octreeProjection?.globalFineTransportControl; }
  get globalFineTransportDeltaDebugPair() {
    return this.octreeProjection?.globalFineTransportDeltaDebugPair;
  }
  get globalFineSourceDebugPair() { return this.octreeProjection?.globalFineSourceDebugPair; }
  get structuredBoundarySymmetryDebug() { return this.octreeProjection?.structuredBoundarySymmetryDebug; }
  get globalFineRedistanceControl() { return this.octreeProjection?.globalFineRedistanceControl; }
  get globalFinePageDeltaDebug() { return this.octreeProjection?.globalFinePageDeltaDebug; }
  get globalFinePageDeltaDebugPair() { return this.octreeProjection?.globalFinePageDeltaDebugPair; }
  get globalFineVolumeControl() { return this.octreeProjection?.globalFineVolumeControl; }
  get structuredProjectionEnergyStats() { return this.octreeProjection?.structuredProjectionEnergyStats; }
  get globalFineCoarseLevelSetControl() { return this.octreeProjection?.globalFineCoarseLevelSetControl; }
  readPowerCoarseFailureRow(row: number) { return this.octreeProjection?.readPowerCoarseFailureRow(row); }
  get globalFineRestrictionControl() { return this.octreeProjection?.globalFineRestrictionControl; }
  get airSupportScratch() { return this.octreeProjection?.airSupportScratch; }
  get columnBaseTexture() { return this.hostAllocation ? this.heightA : undefined; }
  get gridCellTexture() { return this.octreeProjection?.topologyTexture; }
  get velocityTexture() { return this.octreeProjection ? undefined : this.velocityA; }
  get secondaryParticles() { return undefined; }
  applyRuntimeValues(_values: Record<string, string | number | boolean>) {}
  /**
   * Adopt scene inputs that no lattice, arena, or seed depends on. This solver
   * reads most of them from `this.scene` when it writes per-step params, so the
   * swap alone is enough; the octree projection keeps its own params buffer and
   * is refreshed explicitly.
   *
   * The inflow boundary is the exception and has to be rederived. Where the
   * nozzle is and which way it points are a uniform-tier input — that is what
   * makes aiming the hose a params write rather than a restart — but the outlet
   * centre and aperture scale the step params carry are *derived* from them
   * here, on the host, and derived once. Swapping the scene without redoing
   * that derivation would move the arrow the user is dragging and leave the
   * water coming out of where it used to be.
   */
  applySceneUniforms(scene: SceneDescription) {
    this.scene = scene;
    const { nx, ny, nz } = this.info;
    this.inflowBoundary = scene.fluid.inflow
      ? createInflowGridBoundary(scene.fluid.inflow, scene.container, [nx, ny, nz])
      : undefined;
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
    // The seed tier may change the container extents and the finest cell size —
    // that is what scaling the world is — but only together, so the lattice
    // itself holds. A lattice that did move is a structural change wearing a
    // seed change's clothes, and rebuilding is the only honest answer.
    const dimensions = sceneLatticeDimensions(scene);
    if (dimensions[0] !== nx || dimensions[1] !== ny || dimensions[2] !== nz) return false;
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
  get gridPressureSamplesTexture() { return this.octreeProjection?.pressureSamplesTexture; }
  get gridPressureTexture() { return this.octreeProjection?.pressureTexture; }
  get gridDivergenceTexture() { return undefined; }
  async ensureGridDiagnosticTextures() {
    const octreeProjection = this.octreeProjection;
    if (!octreeProjection || !await octreeProjection.ensureDiagnosticTextures()) return;
    const rawEncoder = this.device.createCommandEncoder({ label: "Initialize lazy octree diagnostic fields" });
    const discardActivity = this.logicalActivity.enabled
      ? this.logicalActivity.bindingSession(this.device, rawEncoder, {
        capacity: 1,
        captureId: this.nextLogicalActivityCaptureId(),
        label: "Discarded diagnostic materialization activity",
      })
      : undefined;
    const encoder = discardActivity?.encoder ?? rawEncoder;
    octreeProjection.encodeOverlayMaterialization(encoder);
    discardActivity?.finish();
    this.device.queue.submit([encoder.finish()]);
    if (discardActivity) void discardActivity.read().finally(() => discardActivity.destroy());
    this.applyOctreeInfo(octreeProjection);
  }
  /** Instrumentation view: velocity after advection/forces and before quadtree projection. */
  get preProjectionVelocityTexture() { return this.octreeProjection ? undefined : this.velocityB; }

  private initializeVolume() {
    const { nx, ny, nz } = this.info, c = this.scene.container;
    const data = this.hostAllocation ? new Float32Array(nx * ny * nz) : undefined, dam = sceneDamBreakBox(this.scene);
    const terrainHeights = terrainColumnHeights(this.scene, nx, nz), cellHeight = c.height_m / ny;
    let initialSum = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const aboveGround = (j + 0.5) * cellHeight > terrainHeights[i + nx * k];
      const brickWet = initialFluidBrickContainsCell(this.scene, i, j, k, [nx, ny, nz]);
      const fill = aboveGround && combineInitialBrickWet(this.scene, brickWet, this.scene.fluid.initialCondition === "dam-break"
        ? damBreakBoxContains(dam, (i + .5) / nx, (j + .5) / ny, (k + .5) / nz)
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
      front_m: this.scene.fluid.initialCondition === "dam-break" ? -c.width_m / 2 + dam.max.x * c.width_m : c.width_m / 2,
      frontTelemetrySource: "initial-condition" });
    if (data) {
      const rowBytes = nx * 4, padded = Math.ceil(rowBytes / 256) * 256;
      const packed = new Uint8Array(padded * ny * nz), source = new Uint8Array(data.buffer);
      for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) packed.set(source.subarray(rowBytes * (j + ny * k), rowBytes * (j + ny * k + 1)), padded * (j + ny * k));
      for (const texture of [this.volumeA, this.volumeB]) this.device.queue.writeTexture({ texture }, packed, { bytesPerRow: padded, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
    }
  }

  private group(velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture, pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture, heightIn: GPUTexture, heightOut: GPUTexture, predictedVelocity: GPUTexture = velocityIn, reversedVelocity: GPUTexture = velocityIn, transport: GPUTexture = this.transportA, surfaceIn: GPUTexture = volumeIn) {
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
  private applyOctreeInfo(projection: WebGPUOctreeProjection) {
    const octree = projection.info;
    Object.assign(this.info, {
      gridKind: "octree",
      initialSparseAuthorityReady: this.initialSparseAuthorityPublished,
      surfaceField: "levelset",
      volumeControl: true,
      pressureSolver: projection.pressureSolverLabel,
      coarseDynamicsBackend: projection.coarseBackend,
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
      globalFineLevelSetFactor: projection.surfaceTrackingFactor,
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

  advanceTo(time_s: number, bodies: RigidBodyState[] = []) {
    if (this.disposed) return false;
    const advance = planGPUAdvance(time_s, this.lastTime, this.scene.numerics.maxDt_s); if (!advance) return false;
    const delta = advance.dt_s; if (delta < 1e-6) { this.info.simulatedTime_s = this.lastTime; this.info.simulationLag_s = advance.lag_s; return true; }
    this.lastTime = advance.nextTime_s; this.info.submittedTime_s = this.lastTime; this.info.simulatedTime_s = this.lastTime; this.info.simulationLag_s = advance.lag_s;
    const instrumentationSnapshot = usePerformanceInstrumentationStore.getState();
    const measurementInstrumentationEnabled = instrumentationSnapshot.enabled;
    const activityStoreSnapshot = usePerformanceActivityStore.getState();
    const traceRequestedAt_ms = measurementInstrumentationEnabled ? performance.now() : 0;
    const shouldTracePhysics = measurementInstrumentationEnabled
      && !this.physicsTracePending
      && traceRequestedAt_ms - this.lastPhysicsTraceAt_ms >= PHYSICS_TRACE_CADENCE_MS;
    const physicsTraceSampleId = shouldTracePhysics ? ++this.physicsTraceSampleId : 0;
    const physicsTraceContext = `${this.info.gridKind}:sim-${this.lastTime.toFixed(6)}`;
    const physicsCPUTrace = shouldTracePhysics
      ? new CPUPerformanceTrace(
        physicsTraceSampleId,
        physicsTraceContext,
        { id: "command-encoding", label: "Physics advance planning + command encoding" },
      )
      : undefined;
    const c = this.scene.container, rho = this.scene.fluid.density_kg_m3, sigma = this.scene.fluid.surfaceTension_N_m;
    const substeps = 1;
    const dt = delta / substeps;
    this.info.lastDt_s = this.octreeProjection ? undefined : dt;
    this.info.lastSubsteps = this.octreeProjection ? undefined : substeps;
    this.octreeProjection?.setTimestep(dt);
    const activeBodies = bodies.slice(0, 12);
    this.rigidSystem.syncBodies(activeBodies);
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + (this.octreeProjection ? 1 : substeps);
    this.octreeProjection?.setCouplingBodies(activeBodies.length, activeBodies.some((body) => body.inverseMass_kg > 0));
    const inflow=this.scene.fluid.inflow,outlet=this.inflowBoundary?.outletCenter_m,inflowStepStrength=inflow?averageInflowStrength(inflow,this.lastTime-delta,this.lastTime):0;
    if(this.inflowBoundary){const cellVolume=c.width_m*c.height_m*c.depth_m/(this.info.nx*this.info.ny*this.info.nz);const cells=this.inflowBoundary.flowRate_m3_s*inflowStepStrength*delta/cellVolume;this.octreeProjection?.addSurfaceReferenceVolumeCells(cells);}
    if (this.params) this.device.queue.writeBuffer(this.params, 0, new Float32Array([this.info.nx, this.info.ny, this.info.nz, dt, c.width_m / this.info.nx, c.height_m / this.info.ny, c.depth_m / this.info.nz, this.scene.fluid.gravity_m_s2.y, c.width_m, c.height_m, c.depth_m, sceneHasTerrain(this.scene) ? 1 : 0, rho, this.scene.fluid.dynamicViscosity_Pa_s, this.transportConservativeVolume ? 1 : 0, 1, sigma, c.fluidWallMode === "no-slip" ? 1 : 0, activeBodies.length, c.top === "open" ? 1 : 0,outlet?.x??0,outlet?.y??0,outlet?.z??0,inflow?.radius_m??0,inflow?.velocity_m_s.x??0,inflow?.velocity_m_s.y??0,inflow?.velocity_m_s.z??0,this.inflowBoundary?.apertureScale??0,inflowStepStrength,0,c.fillFraction*this.info.ny,0]));
    if (!this.validationChecked) this.device.pushErrorScope("validation");
    let encoder = this.device.createCommandEncoder({ label: "Uniform GPU fluid step" });
    const physicsTrace = shouldTracePhysics
      && !this.hardwarePhysicsTraceInvalid
      && GPUStageTimestampRecorder.supported(this.device)
      ? new GPUStageTimestampRecorder(
        this.device,
        physicsTraceSampleId,
        "physics",
        physicsTraceContext,
      )
      : undefined;
    const physicsQueueTrace = shouldTracePhysics
      ? new GPUQueueWallPerformanceTraceRecorder(
        physicsTraceSampleId,
        "physics",
        physicsTraceContext,
      )
      : undefined;
    // Every stage boundary rides the next pass this encoder already encodes,
    // so the traced step submits the same command graph as the untraced one.
    if (physicsTrace) encoder = physicsTrace.instrument(encoder);
    const shouldCaptureLogicalActivity = shouldTracePhysics
      && this.logicalActivity.enabled
      && instrumentationSnapshot.shaderActivityEnabled
      && instrumentationSnapshot.shaderGeneration === this.logicalActivity.generation
      && activityStoreSnapshot.enabled;
    let logicalActivitySession: GPULogicalActivityBindingSession | undefined;
    let logicalActivityCaptureId = 0;
    if (this.logicalActivity.enabled) {
      const sharedRecorder = shouldCaptureLogicalActivity
        ? undefined
        : this.logicalActivityDiscardRecorder ??= this.logicalActivity.recorder(this.device, {
          capacity: 1,
          captureId: this.nextLogicalActivityCaptureId(),
          label: "Discarded unsampled physics activity",
        });
      logicalActivityCaptureId = shouldCaptureLogicalActivity ? this.nextLogicalActivityCaptureId() : 0;
      logicalActivitySession = this.logicalActivity.bindingSession(this.device, encoder, {
        capacity: shouldCaptureLogicalActivity
          ? Math.min(
            this.logicalActivityCaptureCapacity,
            maximumPhysicsLogicalActivityCaptureCapacity(this.device.limits),
          )
          : 1,
        captureId: logicalActivityCaptureId,
        label: shouldCaptureLogicalActivity
          ? `Physics activity ${physicsTraceSampleId}`
          : "Discarded unsampled physics activity",
        sharedRecorder,
      });
      encoder = logicalActivitySession.encoder;
    }
    physicsTrace?.begin();
    if (shouldCaptureLogicalActivity) this.encodeLogicalActivityFrameBegin(encoder);
    let logicalPhaseIndex = 0;
    const completePhysicsPhase = (
      completedEncoder: GPUCommandEncoder,
      phase: GPUTimestampPhase,
    ): GPUCommandEncoder => {
      physicsCPUTrace?.completePhase(phase);
      if (shouldCaptureLogicalActivity) {
        this.encodeLogicalActivityPhaseBoundary(completedEncoder, logicalPhaseIndex);
        logicalPhaseIndex += 1;
      }
      physicsTrace?.completePhase(completedEncoder, phase);
      return completedEncoder;
    };
    encoder.clearBuffer(this.rigidExchangeBuffer);
    let inlineRebuildEncoded = false;
    encoder = completePhysicsPhase(encoder, { id: "other", label: "Advance setup" });
    for (let substep = 0; substep < substeps; substep += 1) {
      // The active epoch is immutable for the entire substep. A ready
      // candidate from the prior tail may flip only at this boundary.
      this.octreeProjection?.encodeReadyTopologyFlip(encoder);
      if (this.octreeProjection) {
        this.stepSequenceRecorder.record("ready-topology-flip");
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
      if (this.octreeProjection) {
        if (this.transportConservativeVolume) encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
        const surfaceInflow = inflow && this.inflowBoundary ? {
          outletCenter_m: this.inflowBoundary.outletCenter_m, radius_m: inflow.radius_m,
          velocity_m_s: inflow.velocity_m_s, apertureScale: this.inflowBoundary.apertureScale,
          strength: inflowStepStrength
        } : undefined;
        // Advect both fine and coarse phi with the previous substep's
        // projected + closest-point-extended velocity.
        encoder = this.octreeProjection.encodeSurface(encoder, dt, surfaceInflow, this.scene.numerics.maxDt_s,
          physicsTrace || shouldCaptureLogicalActivity ? (phase, completedEncoder) => {
            return completePhysicsPhase(completedEncoder, OCTREE_SEMANTIC_TRACE_PHASE[phase]);
          } : undefined);
        this.stepSequenceRecorder.record("surface-transport");
        encoder = this.octreeProjection.encode(
          encoder,
          this.info.nx,
          this.info.ny,
          this.info.nz,
          {
            step: this.info.encodedSteps ?? 0,
            productionBoundary: physicsTrace || shouldCaptureLogicalActivity ? (phase, completedEncoder) => {
              return completePhysicsPhase(completedEncoder, OCTREE_SEMANTIC_TRACE_PHASE[phase]);
            } : undefined,
          }
        );
        this.stepSequenceRecorder.record("pressure-projection");
        inlineRebuildEncoded = this.octreeProjection.encodeInactiveTopologyCandidateIfDue(encoder);
        this.stepSequenceRecorder.record("inactive-topology-candidate");
        encoder = completePhysicsPhase(
          encoder,
          { id: "coarse-grid", label: "Inactive next-substep topology candidate" },
        );
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
        const pass = encoder.beginComputePass({ label: "Uniform rigid-body coupling" });
        this.dispatch(pass, this.rigidPipeline, this.rigidGroup); pass.end();
        if (this.transportConservativeVolume) encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
        encoder.copyTextureToTexture({ texture: this.velocityB }, { texture: this.velocityA }, [this.info.nx, this.info.ny, this.info.nz]);
      }
    }
    if (activeBodies.length > 0) {
      const cellVolume = c.width_m * c.height_m * c.depth_m / (this.info.nx * this.info.ny * this.info.nz);
      this.rigidSystem.encode(encoder, delta, cellVolume, substeps, c.height_m / this.info.ny);
      if (this.octreeProjection) this.stepSequenceRecorder.record("rigid-exchange");
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
      this.stepSequenceRecorder.record("sparse-brick-world");
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
    // Step-coherent diagnostics: the step's last commands copy the accepted
    // structured receipt + fine worklist header into the snapshot ring, so a
    // mapped record always describes exactly one completed step. Shared ABI
    // with the Dawn harness audit (structured-authority-audit); readStats
    // consumes the record instead of racing the live control buffers.
    if (this.octreeProjection) {
      const pending = this.octreeProjection as unknown as PendingStepSnapshotSources;
      const losasso = this.octreeProjection.coarseBackend === "losasso";
      const fine = this.globalFineLevelSetSource;
      const coarse = this.coarseLevelSetSource;
      const surfaceHeader = fine?.worklist ?? coarse?.directory.buffer;
      if (losasso) {
        const sources = pending.losassoAuthorityControl && pending.losassoCoarsePhiControl
          && pending.losassoExtensionControl && this.mgpcgControl
          && pending.globalFineCurrentWorklist
          && this.globalFineTransportControl ? {
            authority: pending.losassoAuthorityControl,
            solver: this.mgpcgControl,
            fineWorklist: pending.globalFineCurrentWorklist,
            coarsePhi: pending.losassoCoarsePhiControl,
            extension: pending.losassoExtensionControl,
            fineTransport: this.globalFineTransportControl,
          } : undefined;
        this.losassoStepSnapshotRing ??= new WebGPUOctreeLosassoStepSnapshotRing(
          this.device, structuredStepSnapshotSlotCount());
        if (sources && this.losassoStepSnapshotRing.encode(
          encoder, sources, this.info.encodedSteps ?? 0)) {
          this.stepSequenceRecorder.record("step-snapshot");
        } else if (!this.stepSnapshotFaulted) {
          this.stepSnapshotFaulted = true;
          console.error("[step-snapshot]", JSON.stringify({
            backend: "losasso", step: this.info.encodedSteps,
            slots: this.losassoStepSnapshotRing.slotCount,
            skipped: this.losassoStepSnapshotRing.skippedRecords,
            reason: sources ? "every ring slot was mapping" : "native receipt source absent",
          }));
        }
      } else {
      const velocityControl = this.structuredVelocityControl;
      const boundaryControl = this.structuredBoundaryControl;
      if (velocityControl && boundaryControl && surfaceHeader) {
        // Slot count derives from the in-flight ceiling: an under-sized ring
        // makes `encode` skip the record when every slot is mapping, and a
        // missing `step-snapshot` stage latches a permanent sequence fault
        // (docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A4.2).
        this.stepSnapshotRing ??= new StructuredStepSnapshotRing(
          this.device, structuredStepSnapshotSlotCount());
        // P0.4: the topology-epoch state, the SPGrid per-level setup delta,
        // and the air-support failure scratch are the GPU's own verdict words
        // for the subsystems Part B intends to gate. They are private fields
        // of WebGPUOctreeProjection today (`lib/webgpu-octree.ts`); adding one
        // getter each is the remaining half of A4.1. Reading them structurally
        // lights the segments up the moment those accessors land, and until
        // then the ring reports the segments absent rather than copying zeros
        // that a skip predicate could mistake for evidence.
        if (this.stepSnapshotRing.encode(encoder, {
          structuredVelocityControl: velocityControl,
          structuredBoundaryControl: boundaryControl,
          fineWorklist: surfaceHeader,
          mgpcgControl: this.mgpcgControl,
          fineVolumeControl: this.globalFineVolumeControl,
          projectionEnergyStats: this.structuredProjectionEnergyStats,
          topologyEpochState: pending.topologyEpochState,
          spgridLevelDelta: pending.spgridLevelDelta,
          airSupportScratch: pending.airSupportScratch,
          fineTransportGovernor: this.workAccountingBuffers?.fineTransportGovernor?.buffer,
        }, {
          step: this.info.encodedSteps ?? 0,
          dt_s: dt,
          submittedTime_s: this.lastTime,
          hostFineGeneration: fine?.generation ?? coarse!.generation,
          surfaceKind: fine ? "fine" : "coarse",
        })) this.stepSequenceRecorder.record("step-snapshot");
        else if (!this.stepSnapshotFaulted) {
          // The producer must never skip. It is not a lost diagnostic: the
          // step encodes no `step-snapshot` stage, which is a step-sequence
          // deviation, and the lag-k prediction for this step can never be
          // resolved.
          this.stepSnapshotFaulted = true;
          console.error("[step-snapshot]", JSON.stringify({
            step: this.info.encodedSteps, slots: this.stepSnapshotRing.slotCount,
            skipped: this.stepSnapshotRing.skippedRecords,
            reason: "every ring slot was mapping; raise the slot count",
          }));
        }
      }
      }
    }
    if (shouldCaptureLogicalActivity) this.encodeLogicalActivityFrameEnd(encoder);
    // Freeze application dispatch coverage before timestamp-recorder closure
    // can append profiler infrastructure to the same proxied encoder.
    const logicalActivityDispatchDiagnostics = shouldCaptureLogicalActivity
      ? logicalActivitySession?.diagnostics
      : undefined;
    physicsTrace?.resolve(encoder);
    logicalActivitySession?.finish();
    const submittedEncoder = encoder;
    physicsQueueTrace?.begin();
    this.device.queue.submit([submittedEncoder.finish()]);
    if (physicsCPUTrace) {
      this.info.physicsCPUTrace = physicsCPUTrace.finish({
        id: "other",
        label: "Capture closure + command submission",
      });
      this.info.physicsCaptureIdentity = {
        sampleId: physicsTraceSampleId,
        context: physicsTraceContext,
        frameId: gpuPhysicsPerformanceActivityFrameId({
          sampleId: physicsTraceSampleId,
          context: physicsTraceContext,
        }),
      };
    }
    this.octreeProjection?.retireSubmittedEncoder(submittedEncoder);
    // Driver-independent sequence conformance: every advance, traced or not,
    // must have encoded exactly the declared step program. A deviation is a
    // first-class error published on the info record, not a silent drift.
    if (this.octreeProjection) {
      // P0.5: record what this encode predicted BEFORE the recorder resets.
      // Nothing in the step reads this back; it is resolved against step N's
      // own snapshot when that record maps (see `resolveStepPrediction`).
      this.stepPredictions.record({
        step: this.info.encodedSteps ?? 0,
        encodedSolveBudget:
          this.workAccountingPlan?.pressure.maximumOuterIterations ?? 0,
        conditions: [...this.stepSequenceRecorder.recordedConditions],
        skippedStageIds: [...this.stepSequenceRecorder.skippedStageIds(OCTREE_STEP_PROGRAM)],
      });
      const sequenceDeviations = this.stepSequenceRecorder.finishStep(OCTREE_STEP_PROGRAM);
      // Latched per run: one bad step is a broken driver contract even if
      // later steps conform, so gates always observe it.
      if (sequenceDeviations.length > 0 && !this.stepSequenceFaulted) {
        this.stepSequenceFaulted = true;
        this.info.stepSequenceDeviations = [...sequenceDeviations];
        console.error("[step-sequence]", JSON.stringify({
          step: this.info.encodedSteps, deviations: sequenceDeviations,
        }));
      }
    }
    const physicsQueueTraceRead = physicsQueueTrace?.read(this.device.queue);
    const hardwarePhysicsTraceRead = physicsTrace?.read();
    const physicsTraceRead = hardwarePhysicsTraceRead
      ? hardwarePhysicsTraceRead
        .then((trace) => {
          // One unusable hardware sample retires the stage recorder for this
          // solver; the non-invasive queue-wall observation takes over rather
          // than paying for boundaries that resolve to nothing.
          this.hardwarePhysicsTraceInvalid = !trace;
          return trace ?? physicsQueueTraceRead;
        })
        .catch(() => {
          this.hardwarePhysicsTraceInvalid = true;
          return physicsQueueTraceRead;
        })
      : physicsQueueTraceRead;
    const physicsPublicationId = `physics-publication:${this.info.encodedSteps ?? 0}`;
    const physicsActivityIdentity = (trace: Pick<PerformanceTrace, "sampleId" | "context">): ActivityWorkIdentity => ({
      frameId: gpuPhysicsPerformanceActivityFrameId(trace),
      generation: activityStoreSnapshot.generation,
      submissionId: `gpu-physics:${encodeURIComponent(physicsTraceContext)}:${physicsTraceSampleId}`,
      publicationId: physicsPublicationId,
    });
    const publishCaptureDiagnosticsOnly = (
      trace: Pick<PerformanceTrace, "sampleId" | "context">,
      reasons: readonly string[],
    ) => usePerformanceActivityStore.getState().ingestEvidence(
      physicsActivityIdentity(trace),
      { captureDiagnostics: { reasons } },
    );
    if (shouldCaptureLogicalActivity && logicalActivitySession && physicsTraceRead) {
      const session = logicalActivitySession;
      const activityGeneration = activityStoreSnapshot.generation;
      const shaderGeneration = this.logicalActivity.generation;
      void Promise.all([session.read(), physicsTraceRead]).then(([decoded, trace]) => {
        const instrumentation = usePerformanceInstrumentationStore.getState();
        const activityStore = usePerformanceActivityStore.getState();
        if (!trace || this.disposed
          || !instrumentation.shaderActivityEnabled
          || instrumentation.shaderGeneration !== shaderGeneration
          || activityStore.generation !== activityGeneration) return;
        if (!decoded?.ok) {
          this.latestLogicalActivity = undefined;
          this.latestLogicalActivityCaptureDiagnostics = undefined;
          publishCaptureDiagnosticsOnly(trace, ["logical-readback-failed"]);
          return;
        }
        if (decoded.capture.captureId !== logicalActivityCaptureId) {
          this.latestLogicalActivity = undefined;
          this.latestLogicalActivityCaptureDiagnostics = undefined;
          publishCaptureDiagnosticsOnly(trace, ["logical-capture-id-mismatch"]);
          return;
        }
        const identity = physicsActivityIdentity(trace);
        const tasks = Object.freeze({
          ...gpuLogicalActivityTaskDescriptions(shaderGeneration),
          ...PHYSICS_ACTIVITY_TASKS,
        }) as Readonly<Record<number, GPUPhysicsLogicalActivityTaskDescriptor>>;
        const captureDiagnostics = validateGPUPhysicsLogicalActivityCapture(
          decoded.capture,
          trace,
          logicalActivityDispatchDiagnostics,
          logicalPhaseIndex,
        );
        this.latestLogicalActivityCaptureDiagnostics = captureDiagnostics;
        if (decoded.capture.overflowed) {
          const maximumCapacity = maximumPhysicsLogicalActivityCaptureCapacity(this.device.limits);
          const attemptedEventCount = decoded.capture.events.length
            + decoded.capture.droppedEventCount;
          this.logicalActivityCaptureCapacity = Math.min(
            maximumCapacity,
            Math.max(
              this.logicalActivityCaptureCapacity,
              Math.ceil(attemptedEventCount * 1.25),
            ),
          );
        }
        const projection = physicsPhaseBoundaryTimeProjection(trace, decoded.capture, tasks);
        const sample: GPUPhysicsLogicalActivitySample = {
          identity,
          shaderGeneration,
          capture: decoded.capture,
          trace,
          lane: "gpu-physics",
          clockDomain: "gpu-physics-timestamp",
          windowStart_ms: 0,
          windowEnd_ms: trace.total_ms,
          ...projection,
          tasks,
          captureDiagnostics,
        };
        this.latestLogicalActivity = sample;
        publishDecodedGPULogicalActivity({
          sink: usePerformanceActivityStore.getState(),
          capture: sample.capture,
          identity: sample.identity,
          lane: sample.lane,
          clockDomain: sample.clockDomain,
          windowStart_ms: sample.windowStart_ms,
          windowEnd_ms: sample.windowEnd_ms,
          locateTime: sample.locateTime,
          granularity: "workgroup",
          tasks: sample.tasks,
          maximumRows: 128,
          captureDiagnostics: {
            reasons: sample.captureDiagnostics.reasons,
            recorderOverflowed: sample.captureDiagnostics.overflowed,
            droppedEventCount: sample.captureDiagnostics.droppedEventCount,
            unprofiledDispatchCount:
              sample.captureDiagnostics.unregisteredComputeDispatchCount,
            unprofiledPipelineLabels:
              sample.captureDiagnostics.unregisteredComputePipelineLabels,
          },
        });
      }).catch(() => {
        // Device loss and stale captures fail closed; aggregate timing remains.
      }).finally(() => session.destroy());
    } else if (shouldTracePhysics && physicsTraceRead && activityStoreSnapshot.enabled) {
      const activityGeneration = activityStoreSnapshot.generation;
      void physicsTraceRead.then((trace) => {
        const activityStore = usePerformanceActivityStore.getState();
        if (!trace || this.disposed || !activityStore.enabled
          || activityStore.generation !== activityGeneration) return;
        this.latestLogicalActivity = undefined;
        this.latestLogicalActivityCaptureDiagnostics = undefined;
        publishCaptureDiagnosticsOnly(trace, ["logical-capture-disabled"]);
      }).catch(() => {
        // Device loss is reported by the aggregate trace path.
      });
    }
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
    // Solver residuals are sampled only through the opt-in telemetry path.
    // Physics submission itself must not initiate a GPU-to-CPU map.
    if (this.octreeProjection && inlineRebuildEncoded) {
      for (let rebuild = 0; rebuild < substeps; rebuild += 1) {
        this.octreeProjection.finishTopologyCandidate();
      }
      this.applyOctreeInfo(this.octreeProjection);
      this.info.quadtreeRebuildCompletedCount = (this.info.quadtreeRebuildCompletedCount ?? 0) + substeps;
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
    // With the step-coherent ring active, projection energy comes from the
    // step's own record; the racing mid-pipeline copy exists only as the
    // pre-first-advance fallback.
    const structuredProjectionEnergy = this.stepSnapshotRing
      ? undefined
      : this.octreeProjection?.structuredProjectionEnergyStats;
    if (structuredProjectionEnergy) encoder.copyBufferToBuffer(
      structuredProjectionEnergy, 0, buffer, 16, STRUCTURED_PROJECTION_ENERGY_WORDS * 4,
    );
    this.device.queue.submit([encoder.finish()]);
    const mapPromise = buffer.mapAsync(GPUMapMode.READ);
    const compactFineExpected = Boolean(this.octreeProjection?.globalFineLevelSetSource);
    const solveDiagnostics = this.octreeProjection?.readSolveDiagnostics();
    // Once compact global-fine volume is authoritative, the adaptive surface
    // diagnostic is both obsolete and ignored below. Avoid a separate queue
    // submission/map every 250 ms for data that cannot be selected.
    const surfaceDiagnosticsPromise = compactFineExpected
      ? undefined
      : this.octreeProjection?.readSurfaceDiagnostics();
    const globalFineDiagnosticsPromise = this.octreeProjection?.coarseBackend === "power2017"
      ? this.octreeProjection.readGlobalFineLevelSetDiagnostics() : undefined;
    // The step-coherent record supersedes the racing live-buffer sample for
    // authority health: its words were copied by the step's own encoder.
      const stepSnapshotPromise = this.stepSnapshotRing?.readLatest();
      const losassoStepSnapshotPromise = this.losassoStepSnapshotRing?.readLatest();
    try {
      await mapPromise;
      const [, , surfaceDiagnostics, globalFineDiagnostics, fluidBrickStats, fluidBulkBrickStats,
        stepRecord, losassoStepRecord] = await Promise.all([
        this.validationPromise, solveDiagnostics, surfaceDiagnosticsPromise, globalFineDiagnosticsPromise,
        this.octreeProjection?.readFluidBrickResidencyStats(),
        this.octreeProjection?.readFluidBulkBrickResidencyStats(), stepSnapshotPromise,
        losassoStepSnapshotPromise,
      ]);
    if (losassoStepRecord) {
      const failures = losassoStepSnapshotFailures(losassoStepRecord);
      if (failures.length > 0 && !this.stepSequenceFaulted) {
        this.stepSequenceFaulted = true;
        this.info.stepSequenceDeviations = failures.map((failure) =>
          `Losasso step ${losassoStepRecord.step} receipt: ${failure}`);
        console.error("[losasso-step-receipt]", JSON.stringify({
          step: losassoStepRecord.step,
          authority: Array.from(losassoStepRecord.authority),
          solver: Array.from(losassoStepRecord.solver),
          fine: Array.from(losassoStepRecord.fine),
          coarsePhi: Array.from(losassoStepRecord.coarsePhi),
          extension: Array.from(losassoStepRecord.extension),
          fineTransport: Array.from(losassoStepRecord.fineTransport),
          failures,
        }));
      }
    }
    if(globalFineDiagnostics)this.applyGlobalFineDiagnostics(globalFineDiagnostics);
    // One-shot forensic dump: a structured epoch that stalls while the fine
    // generation keeps advancing is the browser-only candidate freeze
    // (accepted epoch retained "valid" while every new candidate poisons).
    // Capture the coupled candidate controls the first time it is observable.
    if (this.octreeProjection?.coarseBackend === "power2017") {
      // Prefer the step-coherent record: acceptedEpoch and the published fine
      // generation were copied at the SAME step boundary, so the lag is exact
      // whole-step staleness and cannot be inflated by pipeline depth or
      // diagnostics cadence (the legacy pair mixes a fenced GPU word with the
      // live host counter and legally over-reads by the in-flight depth).
      const powerStepRecord = stepRecord;
      const stepHealth = powerStepRecord
        ? structuredAuthorityStepHealth(powerStepRecord) : undefined;
      if (stepHealth) {
        this.info.structuredSnapshotStep = stepHealth.step;
        this.info.structuredAuthorityLagSteps = stepHealth.authorityLagSteps;
        // Active-brick count is worklist header word ONE (A3/A4.3). Word zero
        // is the generation; decoding it as a count reported a rising
        // generation as band occupancy and hid the capacity-overflow
        // sentinel, which silently no-ops the solver.
        this.info.structuredSnapshotActiveFineBricks = stepHealth.activeFineBricks;
        this.info.structuredSnapshotFineBandOverflow = stepHealth.fineBandCapacityOverflow;
      }
      if (powerStepRecord) this.resolveStepPrediction(powerStepRecord);
      const frozenEpoch = stepHealth?.acceptedEpoch ?? this.info.structuredVelocityGeneration ?? 0;
      const liveFine = stepHealth?.publishedFineGeneration ?? this.info.globalFineGeneration ?? 0;
      const lag = stepHealth ? stepHealth.authorityLagSteps : liveFine - frozenEpoch;
      this.structuredProbeCounter = (this.structuredProbeCounter ?? 0) + 1;
      if (this.structuredProbeCounter % 20 === 1) {
        console.debug("[structured-probe]", JSON.stringify({ frozenEpoch, liveFine, lag,
          source: stepHealth ? "step-snapshot" : "live-race",
          step: stepHealth?.step,
          rows: stepHealth?.acceptedRows ?? this.info.structuredVelocityRows ?? -1,
          valid: this.info.structuredVelocityValid,
          boundaryValid: this.info.structuredBoundaryValid,
          reject: this.info.structuredRejectSummary,
          pressureRows: this.info.pressureRequiredRows,
          counterRaces: this.structuredRowCounterRaces }));
      }
      // Episodic lag telemetry: any authority lag beyond normal pipeline depth
      // is logged (throttled to once per 8 fine generations) so the transient
      // stall episodes that visually damp the browser run are measurable.
      if (frozenEpoch > 0 && (stepHealth ? lag >= 2 : lag >= 4) && liveFine >= this.structuredLagLoggedFine + 8) {
        this.structuredLagLoggedFine = liveFine;
        console.warn("[structured-lag]", JSON.stringify({ frozenEpoch, liveFine, lag,
          source: stepHealth ? "step-snapshot" : "live-race",
          rows: stepHealth?.acceptedRows ?? this.info.structuredVelocityRows ?? 0,
          valid: this.info.structuredVelocityValid ?? false,
          reject: this.info.structuredRejectSummary }));
      }
      // Deep-freeze forensics, re-armed per frozen epoch so every distinct
      // stall episode gets one frontier dump, not just the first.
      if (this.octreeProjection && frozenEpoch > 0 && lag > 8
        && this.structuredFreezeDumpedEpoch !== frozenEpoch) {
        this.structuredFreezeDumpedEpoch = frozenEpoch;
        void this.octreeProjection.readPowerFrontierFailure().then((failure) => {
          console.error("[structured-epoch-freeze]",
            JSON.stringify({ frozenEpoch, liveFine, failure }));
        }).catch(() => { /* diagnostic only */ });
      }
      // Publication collapse ("0 live pressure rows"): the browser-only
      // failure is the POWER ROW publication intermittently resolving zero
      // rows while velocity/boundary controls stay valid and gen-current.
      // Dump the frontier decode once per episode, re-arm on recovery.
      const rows = this.info.structuredVelocityRows ?? 0;
      const valid = this.info.structuredVelocityValid ?? false;
      const pressureRows = this.info.pressureRequiredRows ?? -1;
      // The live row counter races in-flight steps (mid-step samples read the
      // cleared phase as 0). Count those separately; only the queue-fenced
      // receipt going invalid is a real publication collapse. The step-coherent
      // record is authoritative when available.
      const receiptHealthy = stepHealth?.receiptValid
        ?? (valid && (this.info.structuredBoundaryValid ?? false) && rows > 0);
      if (pressureRows === 0 && receiptHealthy) {
        this.structuredRowCounterRaces += 1;
      }
      const publicationCollapsed = !receiptHealthy;
      if (this.octreeProjection && publicationCollapsed
        && (this.info.encodedSteps ?? 0) > 0
        && !this.structuredPublicationFailureDumped) {
        this.structuredPublicationFailureDumped = true;
        void this.octreeProjection.readPowerFrontierFailure().then((failure) => {
          console.error("[structured-publication-failure]",
            JSON.stringify({ frozenEpoch, liveFine, rows, valid, pressureRows,
              interfaceBricks: this.info.globalFineInterfaceBricks,
              activeBricks: this.info.globalFineActiveBricks,
              rejectStage: this.info.structuredRejectStage,
              rejectSummary: this.info.structuredRejectSummary, failure }));
        }).catch(() => { /* diagnostic only */ });
      } else if (!publicationCollapsed && this.structuredPublicationFailureDumped) {
        console.warn("[structured-publication-recovered]",
          JSON.stringify({ frozenEpoch, liveFine, pressureRows }));
        this.structuredPublicationFailureDumped = false;
      }
      // Generation desync and publication collapse are first-class errors:
      // publish them on the stability card instead of only the console.
      if (this.octreeProjection) {
        const flags: string[] = [];
        // Exact snapshot lag flags real whole-step staleness; the legacy pair
        // keeps the old pipeline-depth allowance to avoid phantom alerts.
        if (stepHealth ? lag >= 2 : lag > 4) flags.push(`structured-authority-lag ${lag} gen`);
        if (publicationCollapsed && (this.info.encodedSteps ?? 0) > 0) {
          flags.push("structured-publication-invalid");
        }
        if (this.info.stepSequenceDeviations?.length) flags.push("step-sequence-deviation");
        // P0.5 ALERT: the browser surfaces a broken launch-shape prediction on
        // the stability card; harness lanes read the same info field and fail
        // the run. A skip may only ever delete work the GPU reports as zero.
        if (this.info.stepPredictionFailures?.length) flags.push("step-prediction-mismatch");
        if (stepHealth?.fineBandCapacityOverflow) flags.push("fine-band-capacity-overflow");
        if ((this.info.structuredAirSupportFailureFlags ?? 0) !== 0) {
          flags.push("air-support-publication-failure");
        }
        this.info.stabilityFlags = flags;
      }
    }
    if(fluidBrickStats){this.info.fluidBrickCapacity=fluidBrickStats.capacity;this.info.fluidBrickResidentCount=fluidBrickStats.resident;this.info.fluidBrickCoreCount=fluidBrickStats.core;this.info.fluidBrickHaloCount=fluidBrickStats.halo;this.info.fluidBrickActivatedCount=fluidBrickStats.activated;this.info.fluidBrickRetiredCount=fluidBrickStats.retired;this.info.fluidBrickGeneration=fluidBrickStats.generation;}
    if(fluidBulkBrickStats){this.info.fluidBulkBrickResidentCount=fluidBulkBrickStats.resident;this.info.fluidBulkBrickHaloCount=fluidBulkBrickStats.halo;this.info.fluidBulkBrickActivatedCount=fluidBulkBrickStats.activated;this.info.fluidBulkBrickRetiredCount=fluidBulkBrickStats.retired;}
    const words = this.reductionBuffer
      ? new Uint32Array(buffer.getMappedRange(0, 16))
      : new Uint32Array(4);
    const structuredEnergy = this.octreeProjection?.coarseBackend === "power2017" && stepRecord
      ? decodeStructuredProjectionEnergy(stepRecord.snapshot.projectionEnergyControl)
      : structuredProjectionEnergy
        ? decodeStructuredProjectionEnergy(new Uint32Array(
          buffer.getMappedRange(16, STRUCTURED_PROJECTION_ENERGY_WORDS * 4),
        ))
        : undefined;
    if (structuredEnergy?.sample) {
      this.info.structuredStartKineticEnergyProxy =
        structuredEnergy.sample.startKineticEnergyProxy;
      this.info.structuredPostAdvectionKineticEnergyProxy =
        structuredEnergy.sample.postAdvectionKineticEnergyProxy;
      this.info.structuredPreProjectionKineticEnergyProxy =
        structuredEnergy.sample.preProjectionKineticEnergyProxy;
      this.info.structuredPostProjectionKineticEnergyProxy =
        structuredEnergy.sample.postProjectionKineticEnergyProxy;
      this.info.structuredWetStartKineticEnergyProxy =
        structuredEnergy.sample.wetStartKineticEnergyProxy;
      this.info.structuredWetPostAdvectionKineticEnergyProxy =
        structuredEnergy.sample.wetPostAdvectionKineticEnergyProxy;
      this.info.structuredWetPreProjectionKineticEnergyProxy =
        structuredEnergy.sample.wetPreProjectionKineticEnergyProxy;
      this.info.structuredWetPostProjectionKineticEnergyProxy =
        structuredEnergy.sample.wetPostProjectionKineticEnergyProxy;
      this.info.structuredWetFaceCount = structuredEnergy.sample.wetFaceCount;
      this.info.structuredWetStartThetaEnergyProxy =
        structuredEnergy.sample.wetStartThetaEnergyProxy;
      this.info.structuredWetPostAdvectionThetaEnergyProxy =
        structuredEnergy.sample.wetPostAdvectionThetaEnergyProxy;
      this.info.structuredWetPreProjectionThetaEnergyProxy =
        structuredEnergy.sample.wetPreProjectionThetaEnergyProxy;
      this.info.structuredWetPostProjectionThetaEnergyProxy =
        structuredEnergy.sample.wetPostProjectionThetaEnergyProxy;
      this.info.structuredStaggeredPathCount = structuredEnergy.sample.staggeredPathCount;
      this.info.structuredProjectionEnergyRatio = structuredEnergy.sample.projectionEnergyRatio;
      this.info.structuredProjectionEnergySampleCount = 1;
    } else {
      this.info.structuredStartKineticEnergyProxy = undefined;
      this.info.structuredPostAdvectionKineticEnergyProxy = undefined;
      this.info.structuredWetStartKineticEnergyProxy = undefined;
      this.info.structuredWetPostAdvectionKineticEnergyProxy = undefined;
      this.info.structuredWetPreProjectionKineticEnergyProxy = undefined;
      this.info.structuredWetPostProjectionKineticEnergyProxy = undefined;
      this.info.structuredWetStartThetaEnergyProxy = undefined;
      this.info.structuredWetPostAdvectionThetaEnergyProxy = undefined;
      this.info.structuredWetPreProjectionThetaEnergyProxy = undefined;
      this.info.structuredWetPostProjectionThetaEnergyProxy = undefined;
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
    if (this.octreeProjection?.relativeResidual !== undefined) this.info.pressureRelativeResidual = this.octreeProjection.relativeResidual;
    if (this.octreeProjection?.residualRms !== undefined) this.info.pressureResidual = this.octreeProjection.residualRms;
    if (this.octreeProjection) {
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
    this.stepSnapshotRing?.destroy();
    this.stepSnapshotRing = undefined;
    this.losassoStepSnapshotRing?.destroy();
    this.losassoStepSnapshotRing = undefined;
    this.octreeProjection?.destroy();
    const textures = this.hostAllocation
      ? [this.velocityA, this.velocityB, this.velocityC, this.velocityD,
        this.pressureA, this.pressureB, this.volumeA, this.volumeB,
        this.heightA, this.heightB, this.terrainTexture,
        this.transportA, this.transportB, this.fluxScales]
      : [this.terrainTexture];
    for (const texture of new Set(textures)) texture.destroy();
    this.params?.destroy(); this.reductionBuffer?.destroy(); this.sharpenBuffer?.destroy(); this.rigidSystem.destroy(); this.rigidExchangeBuffer.destroy(); this.statsReadbackBuffer?.destroy();
    this.logicalActivityDiscardRecorder?.destroy();
    this.logicalActivityDiscardRecorder = undefined;
    this.logicalActivityMarkerTickSource?.destroy();
    this.logicalActivityMarkerTick?.destroy();
  }
}

import { damBreakBoxContains, initialLiquidContainsCell, sceneDamBreakBox, sceneDamBreakFractions } from "../../core/initial-fluid";
import { resolveOctreeRuntimeDials } from "./octree-runtime-dials";
import type { OctreeDebugSources } from "./octree-debug-sources";
import {
  CPUPerformanceTrace,
  GPUQueueWallPerformanceTraceRecorder,
  GPUStageTimestampRecorder,
  type GPUTimestampPhase,
  type PerformanceTrace,
} from "../../core/performance-trace";
import {
  performanceShaderVariant,
  usePerformanceInstrumentationStore,
} from "../../core/stores/performance-instrumentation-store";
import { usePerformanceActivityStore } from "../../core/stores/performance-activity-store";
import {
  createGPULogicalActivityAdoptionContext,
  gpuLogicalActivityTaskDescriptions,
  stableGPULogicalActivityId,
  type GPULogicalActivityAdoptionContext,
  type GPULogicalActivityBindingSession,
  type GPULogicalActivityBindingSessionDiagnostics,
  type GPULogicalActivityTaskDescription,
} from "../../core/gpu-logical-activity-adoption";
import {
  GPU_LOGICAL_ACTIVITY_HEADER_WORDS,
  GPU_LOGICAL_ACTIVITY_MAX_CAPACITY,
  GPU_LOGICAL_ACTIVITY_RECORD_BYTES,
  GPU_LOGICAL_ACTIVITY_UNKNOWN_U32,
  type GPULogicalActivityCapture,
  type GPULogicalActivityRecorder,
  type GPULogicalActivityTimeLocator,
} from "../../core/gpu-logical-activity";
import {
  gpuPhysicsPerformanceActivityFrameId,
  type ActivityWorkIdentity,
} from "../../core/performance-activity";
import { publishDecodedGPULogicalActivity } from "../../core/gpu-performance-activity";
import { initializeRigidBodies, type RigidBodyState } from "../../core/rigid-body";
import {
  GPU_RIGID_EXCHANGE_BYTES,
  type GPUEulerianInfo,
  type GPURigidLoad
} from "../../core/webgpu-eulerian";
import type { GPUQuality } from "../../core/gpu-quality";
import type { SceneDescription } from "../../core/model";
import type { SparseScenePrimitiveUpdate } from "../../core/webgpu-sparse-scene-proxies";
import type { OctreeCoarseDynamicsConfiguration } from "./octree-coarse-backend";
import { createTallCellLayout } from "../../core/tall-cell-grid";
import { sceneLatticeDimensions } from "../../core/scene-lattice";
import { planGPUAdvance } from "../../core/tall-cell-diagnostics";
import { averageInflowStrength, createInflowGridBoundary, type InflowGridBoundary } from "../../core/inflow-boundary";
import { WebGPUOctreeProjection } from "./webgpu-octree";
import { OCTREE_ALLOCATION_STAGES, OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES,
  type OctreeSemanticPhase, type OctreeProjectionOptions } from "./octree-projection-contract";
import {
  OCTREE_POWER_COARSE_LEVELSET_ERROR,
  OCTREE_POWER_COARSE_LEVELSET_VALID,
  unpackOctreePowerCoarseLevelSetControl,
} from "./octree-power-coarse-levelset-control-abi";
import {
  FINE_LEVELSET_TOPOLOGY_ERROR,
  FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON,
  unpackFineLevelSetGPUTopologyControl,
} from "./webgpu-octree-fine-levelset-topology";
import { readFineLevelSetWorksetHeader } from "./octree-fine-levelset-bricks";
import { unpackFineLevelSetGPURedistanceControl } from "./webgpu-octree-fine-levelset-redistance";
import { unpackFineLevelSetGPUTransportControl } from "./webgpu-octree-fine-levelset-transport";
import { FINE_TO_COARSE_LEVELSET_ERROR, unpackFineToCoarseGPUControl } from "./webgpu-octree-fine-to-coarse-levelset";
import { terrainColumnHeights } from "../../core/terrain";
import { WebGPURigidBodySystem } from "../../core/webgpu-rigid-body";
import { GPUInitializationTaskRunner, type GPUInitializationTask } from "../../core/gpu-initialization";
import { planGPUShaderCapabilities } from "../../core/gpu-shader-plan";
import {
  FINE_LEVELSET_VOLUME_VALID,
  unpackFineLevelSetGPUVolumeControl,
} from "./webgpu-octree-fine-levelset-volume";
import { supportsFluidM1MaxReduction } from "../../core/webgpu-device-limits";
import { isOctreePersistentMGPCGSolverLabel } from "../../core/pressure-solver-label-abi";
import {
  decodeStructuredProjectionEnergy,
  STRUCTURED_PROJECTION_ENERGY_WORDS,
} from "./octree-structured-receipt-abi";
import { decodeOctreeStructuredRejectCarry } from "./octree-structured-reject-carry";
import {
  MAXIMUM_PENDING_PHYSICS_ADVANCES,
  StructuredStepSnapshotRing,
  structuredAuthorityStepHealth,
  structuredStepSnapshotSlotCount,
  structuredStepWorkObservation,
  type StructuredStepSnapshotRecord,
} from "./structured-step-snapshot";
import { losassoStepSnapshotDiagnosticSummary, losassoStepSnapshotFailures,
  WebGPUOctreeLosassoStepSnapshotRing }
  from "./webgpu-octree-losasso-step-snapshot";
import { OCTREE_LOSASSO_ADAPTIVE_MASS_MAGIC, unpackAdaptiveMassReceipt }
  from "./octree-losasso-receipt-abi";
import {
  OCTREE_STEP_PROGRAM,
  PhysicsStepPredictionLedger,
  StepSequenceRecorder,
  physicsStepPredictionFailures,
} from "../../core/physics-step-program";

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
  readonly losassoAdaptiveAcceptedGraphControl?: GPUBuffer;
  readonly losassoAdaptiveCandidateGraphControl?: GPUBuffer;
  readonly losassoAdaptivePhiControl?: GPUBuffer;
  readonly losassoAdaptivePhiReceipts?: GPUBuffer;
  readonly losassoAdaptiveVelocityReceipts?: GPUBuffer;
  readonly losassoAdaptiveRendererDirectory?: GPUBuffer;
  readonly losassoCandidateAuthorityControl?: GPUBuffer;
  readonly losassoAdaptiveMassControl?: GPUBuffer;
  readonly losassoAdaptiveMassReceipts?: GPUBuffer;
  readonly losassoAdaptiveCandidateMassControl?: GPUBuffer;
  readonly losassoAdaptiveCandidateMassReceipts?: GPUBuffer;
  readonly losassoCandidateVelocityMigrationReceipt?: GPUBuffer;
  readonly globalFineCurrentWorklist?: GPUBuffer;
  readonly globalFineCurrentTopologyControl?: GPUBuffer;
  readonly globalFineCurrentRedistanceControl?: GPUBuffer;
  readonly globalFineCurrentVolumeControl?: GPUBuffer;
  readonly fluidBrickResidencyWorklist?: GPUBuffer;
  readonly fluidBulkBrickResidencyWorklist?: GPUBuffer;
}

/**
 * Losasso's live solve and residency controls are recurring GPU scratch.  A
 * running solve clears them before publishing its final verdict, so reading
 * them between submissions can turn an in-progress generation into a false
 * rejection.  Once the step-tail snapshot ring exists, only its coherent
 * records may update host diagnostics; a poll with no unread record retains
 * the last admitted verdict.
 */
export function mayReadLiveOctreeDiagnostics(
  coarseBackend: "losasso" | "power2017" | undefined,
  snapshotRingActive: boolean,
): boolean {
  return coarseBackend !== "losasso" || !snapshotRingActive;
}

export interface WebGPUOctreeEulerianOptions {
  octree: Partial<OctreeProjectionOptions>;
  tallCellSettings?: Partial<import("../../core/tall-cell-grid").TallCellSettings>;
  /** Construction-time coarse authority selection. Never uploaded as a shader flag. */
  coarseDynamics?: OctreeCoarseDynamicsConfiguration;
  deferPipelineCompilation?: boolean;
  /** Internal lifecycle channel for the worker-owned allocation graph. */
  allocationProgress?: (label: string, completed: number, total: number) => void;
}

/** One shared-host unit followed by the octree-owned allocation stages. */
export const OCTREE_SOLVER_ALLOCATION_WORK_UNITS = 1 + OCTREE_ALLOCATION_STAGES.length;

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

/** GPU-resident adaptive octree simulation host. */
export class WebGPUOctreeEulerianSolver {
  readonly info: GPUEulerianInfo;
  private terrainTexture: GPUTexture;
  private rigidBuffer: GPUBuffer; private rigidExchangeBuffer: GPUBuffer;
  private rigidSystem: WebGPURigidBodySystem;
  private statsReadbackBuffer?: GPUBuffer;
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
  /** Latched from a step-coherent GPU receipt. The next submission throws. */
  private fatalPhysicsError?: string;
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
  private octreeProjection: WebGPUOctreeProjection;
  private disposed = false;
  private initialSparseAuthorityPublished = false;
  private baseAllocatedBytes = 0;

  constructor(
    private device: GPUDevice,
    // Not readonly: `applySceneUniforms` swaps in scalar-only scene revisions.
    public scene: SceneDescription,
    quality: GPUQuality,
    private onRigidLoads: ((loads: GPURigidLoad[]) => void) | undefined,
    options: WebGPUOctreeEulerianOptions
  ) {
    options.allocationProgress?.(
      "Allocate shared solver and rigid-body resources", 0, OCTREE_SOLVER_ALLOCATION_WORK_UNITS,
    );
    // The octree has one measured executable profile. Reject it before the
    // first texture or buffer allocation.
    if (!device.features.has("subgroups")
      || !supportsFluidM1MaxReduction(device.limits)) {
      throw new Error("Power octree requires the M1 Max 128-lane subgroup profile");
    }
    this.logicalActivity = createGPULogicalActivityAdoptionContext({
      moduleId: "physics/phase-boundaries",
      profile: performanceShaderVariant(),
      identity: "workgroup",
    });
    const c = scene.container;
    const matched = createTallCellLayout(
      scene, quality, device.limits.maxTextureDimension3D, options.tallCellSettings,
    );
    const nx = matched.nx, ny = matched.fineNy, nz = matched.nz;
    this.inflowBoundary = scene.fluid.inflow
      ? createInflowGridBoundary(scene.fluid.inflow, scene.container, [nx, ny, nz])
      : undefined;
    const terrainUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    this.terrainTexture = device.createTexture({
      label: "Octree terrain heights", size: [nx, nz], format: "r32float", usage: terrainUsage,
    });
    this.rigidExchangeBuffer = device.createBuffer({
      size: GPU_RIGID_EXCHANGE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.rigidSystem = new WebGPURigidBodySystem(
      device, scene, this.rigidExchangeBuffer, this.terrainTexture, options.deferPipelineCompilation,
    );
    this.rigidBuffer = this.rigidSystem.stateBuffer;
    this.rigidSystem.syncBodies(initializeRigidBodies(scene.rigidBodies));
    const count = nx * ny * nz;
    this.info = {
      nx, ny, nz, storedNy: ny, cellCount: count, equivalentUniformCells: count,
      compressionRatio: 1, activeCompressionRatio: 1, activeSampleCount: count,
      regularLayers: ny, maximumNeighborDelta: 0, gridKind: "octree",
      cellSize_m: Math.max(c.width_m / nx, c.height_m / ny, c.depth_m / nz),
      pressureIterations: 0, allocatedBytes: 0, quality, encodedSteps: 0,
      maximumTallCellHeight: 0, submittedTime_s: 0, simulatedTime_s: 0,
      completedTime_s: 0, simulationLag_s: 0,
    };
    this.initializeVolume();
    this.octreeProjection = new WebGPUOctreeProjection(device, scene, { nx, ny, nz }, {
      rigidBodies: this.rigidBuffer, rigidExchange: this.rigidExchangeBuffer,
      rigidImmersedVolumes: this.rigidSystem.immersedVolumeBuffer,
      terrain: this.terrainTexture,
    }, {
      maximumLeafSize: options.octree.maximumLeafSize ?? 16,
      adaptivity: options.octree.adaptivity ?? 1,
      interfaceRefinementBandCells: options.octree.interfaceRefinementBandCells ?? 4,
      surfaceRefinementGradingLayers: options.octree.surfaceRefinementGradingLayers ?? 1,
      initialRuntimeDials: options.octree.initialRuntimeDials,
      fineLevelSetBandCells: options.octree.fineLevelSetBandCells,
      globalFineLevelSetFactor: options.octree.globalFineLevelSetFactor ?? 1,
      globalFineLevelSetMaximumBricks: options.octree.globalFineLevelSetMaximumBricks,
      pressureRowCapacity: options.octree.pressureRowCapacity,
      coarseDynamics: options.coarseDynamics,
    }, options.deferPipelineCompilation, (label, completed) => options.allocationProgress?.(
      label, completed + 1, OCTREE_SOLVER_ALLOCATION_WORK_UNITS,
    ));
    this.applyOctreeInfo(this.octreeProjection);
    if (!options.deferPipelineCompilation) void this.publishInitialSparseScene();
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

  static async createAsync(device:GPUDevice,scene:SceneDescription,quality:GPUQuality,onRigidLoads:((loads:GPURigidLoad[])=>void)|undefined,options:WebGPUOctreeEulerianOptions,onProgress:(label:string,completed:number,total:number,phase?:string,taskId?:string)=>void,signal:AbortSignal=new AbortController().signal){
    const runner=new GPUInitializationTaskRunner((snapshot)=>onProgress(snapshot.label,snapshot.completed,snapshot.total,snapshot.phase,snapshot.taskId),signal);
    let solver:WebGPUOctreeEulerianSolver|undefined;
    try{
      const capabilityPlan=planGPUShaderCapabilities(scene,{
        solver:"octree",
        fineInterface:true,
        logicalActivity:performanceShaderVariant().enabled,
      });
      await runner.run([
        {id:"solver.capabilities",phase:"planning",label:`Resolve ${capabilityPlan.values.size} scene-required GPU capabilities`,run:()=>{}},
        {id:"solver.allocate",phase:"allocation",label:"Allocate octree solver resources",dependencies:["solver.capabilities"],
          workUnits:OCTREE_SOLVER_ALLOCATION_WORK_UNITS,
          run:(_signal,report)=>{solver=new WebGPUOctreeEulerianSolver(device,scene,quality,onRigidLoads,{...options,deferPipelineCompilation:true,
            allocationProgress:(label,completed)=>report?.(label,completed)});}},
      ]);
      await runner.run(solver!.initializationTasks());
      return solver!;
    }catch(error){solver?.destroy();throw error;}
  }
  private initializationTasks():GPUInitializationTask[]{
    const tasks:GPUInitializationTask[]=[...this.rigidSystem.initializationTasks()];
    if (this.logicalActivity.enabled) tasks.push({
      id: "octree.pipeline.logical-activity-marker", phase: "solver-pipelines",
      label: "Compile physics logical-activity marker",
      run: () => this.createLogicalActivityMarker(),
    });
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
    const powerStructuredAuthority = this.octreeProjection?.coarseBackend === "power2017";
    this.info.structuredVelocityValid = powerStructuredAuthority && velocity.length >= 6 && velocity[0] === 0
      && (velocity[2] ?? 0) > 0 && (velocity[3] ?? 0) > 0 && (velocity[4] ?? 2) <= 1;
    this.info.structuredVelocityRows = velocity[2] ?? 0;
    this.info.structuredVelocityGeneration = velocity[3] ?? 0;
    this.info.structuredVelocitySlots = velocity[5] ?? 0;
    // Word 1 is the GPU reject carry and was previously read back and dropped.
    // Naming the stage here is what separates "a face was rejected in the
    // divergence gather" from the silent freeze it otherwise becomes.
    if (powerStructuredAuthority) {
      const reject = decodeOctreeStructuredRejectCarry(
        (value.structuredRejectCarry?.length ?? 0) >= 2 ? value.structuredRejectCarry! : velocity);
      this.info.structuredRejectStage = reject.stage;
      this.info.structuredRejectIndex = reject.index;
      this.info.structuredRejectSummary = reject.clean ? undefined : reject.summary;
    } else {
      this.info.structuredRejectStage = undefined;
      this.info.structuredRejectIndex = undefined;
      this.info.structuredRejectSummary = undefined;
    }
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
    const liveFailure = powerStructuredAuthority && !!support && support.length >= 16
      && (support[0] !== 0 || (support[1] ?? 0xffff_ffff) !== 0xffff_ffff);
    const latchedFailure = powerStructuredAuthority && (latched[0] ?? 0) !== 0;
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
    this.info.structuredBoundaryValid = powerStructuredAuthority && boundary.length >= 7 && boundary[0] === 0
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
    // A lane whose t=0 receipt is its own private ABI validates it itself:
    // this shell cannot decode those words without naming the lane's receipt
    // format, which is exactly the coupling the hook removes. `undefined`
    // means the lane has no dedicated gate and the shared structured and
    // boundary validation below applies instead.
    const laneReceipt = await projection.validateInitialLaneAuthority({
      dimensions: [this.info.nx, this.info.ny, this.info.nz],
      refreshInfo: () => this.applyOctreeInfo(projection),
    });
    if (laneReceipt) {
      this.info.quadtreePressureConverged = laneReceipt.converged;
      this.info.quadtreePressureIterationsUsed = laneReceipt.iterationsUsed;
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
    for (const phase of OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES) {
      await this.publishInitialSparseScenePhase(phase.id);
    }
  }

  get volumeTexture() { return this.octreeProjection?.levelSetTexture; }
  get rigidRenderBuffer() { return this.rigidSystem.renderBuffer; }
  get rigidMotionBuffer() { return this.rigidSystem.motionBuffer; }
  get rigidCouplingDebug() { return {
    state: this.rigidSystem.stateBuffer,
    exchange: this.rigidExchangeBuffer,
    immersedVolumes: this.rigidSystem.immersedVolumeBuffer,
    sealedPlugDiagnostics: this.octreeProjection?.rigidCouplingDiagnosticBuffer,
    rigidBoundaryRefreshDiagnostics: this.octreeProjection?.rigidBoundaryRefreshDiagnosticBuffer,
    bodyCount: this.scene.rigidBodies.length,
  }; }
  setSelectedRigidBody(index: number) { this.rigidSystem.setSelectedIndex(index); }
  pickRigidBody(origin: RigidBodyState["position_m"], direction: RigidBodyState["position_m"]) { return this.rigidSystem.pick(origin,direction); }
  readRigidBodyPoses() { return this.rigidSystem.readPoses(); }
  // Rendering contours the smooth resident level set when the quadtree
  // projection maintains one; the flux-form VOF field is near-binary and its
  // 0.5 contour is quantized to cell scale. Diagnostics keep reading the VOF
  // field through volumeTexture.
  get surfaceFieldTexture() { return this.octreeProjection?.levelSetTexture; }
  /** False once global-fine publication has retired the dense bootstrap phi. */
  get hasDenseSurfaceField() { return this.octreeProjection?.hasDenseLevelSetPublication ?? false; }
  get sparseVoxelSceneSource() { return this.octreeProjection?.sparseVoxelSceneSource; }
  stageSceneUpdate(scene: SceneDescription) { this.octreeProjection?.stageSceneUpdate(scene); }
  stageLivePrimitiveUpdates(updates: readonly SparseScenePrimitiveUpdate[]) {
    return this.octreeProjection?.stageLivePrimitiveUpdates(updates) ?? false;
  }
  encodeSceneMaintenance(encoder: GPUCommandEncoder) { this.octreeProjection?.encodeSceneMaintenance(encoder); }
  get structuredVelocityControl() { return this.octreeProjection?.structuredVelocityControl; }
  get structuredBoundaryControl() { return this.octreeProjection?.structuredBoundaryControl; }
  get structuredRowVelocities() { return this.octreeProjection?.structuredRowVelocities; }
  get losassoFrontierDebug() { return this.octreeProjection?.losassoFrontierDebug; }
  get losassoCoarsePhiDebug() { return this.octreeProjection?.losassoCoarsePhiDebug; }
  get structuredAuthority() { return this.octreeProjection?.structuredAuthority; }
  get structuredWorksets() { return this.octreeProjection?.structuredWorksets; }
  /** Post-submit diagnostics only; never consumed by the simulation schedule. */
  get workAccounting() { return this.octreeProjection?.workAccounting; }
  get workAccountingBuffers() { return this.octreeProjection?.workAccountingBuffers; }
  get workAccountingPlan() { return this.octreeProjection?.workAccountingPlan; }
  captureWorkAccounting() { return this.octreeProjection?.captureWorkAccounting(); }
  readCoarseSurfaceTrackerReceipt() {
    return this.octreeProjection?.readCoarseSurfaceTrackerReceipt();
  }
  /** QA-only passthrough for the authoritative Section 4.3 solver status. */
  get mgpcgControl() { return this.octreeProjection?.mgpcgControl; }
  get powerDescriptorControl() { return this.octreeProjection?.powerDescriptorControl; }
  get powerTopologyControl() { return this.octreeProjection?.powerTopologyControl; }
  get powerDescriptorRows() { return this.octreeProjection?.powerDescriptorRows; }
  get powerTopologyMetrics() { return this.octreeProjection?.powerTopologyMetrics; }
  get powerCatalogEntryHeaders() { return this.octreeProjection?.powerCatalogEntryHeaders; }
  get powerCatalogFaces() { return this.octreeProjection?.powerCatalogFaces; }
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
  get initialSparseAuthorityReady() { return this.initialSparseAuthorityPublished; }
  get globalFineLevelSetSource() { return this.octreeProjection?.globalFineLevelSetSource; }
  get coarseLevelSetSource() { return this.octreeProjection?.coarseLevelSetSource; }
  /** QA-only passthrough for reproducing recurring frontier phase decisions. */
  get globalFineSummaryDirectory() { return this.octreeProjection?.globalFineSummaryDirectory; }
  get globalFineSummaryDebug() { return this.octreeProjection?.globalFineSummaryDebug; }
  get globalFineTransportDeltaDebugPair() {
    return this.octreeProjection?.globalFineTransportDeltaDebugPair;
  }
  get globalFineSourceDebugPair() { return this.octreeProjection?.globalFineSourceDebugPair; }
  get structuredBoundarySymmetryDebug() { return this.octreeProjection?.structuredBoundarySymmetryDebug; }
  get globalFinePageDeltaDebugPair() { return this.octreeProjection?.globalFinePageDeltaDebugPair; }
  readPowerCoarseFailureRow(row: number) { return this.octreeProjection?.readPowerCoarseFailureRow(row); }
  /**
   * This lane's quality-assurance publications, opaque to the solver contract.
   *
   * The renderer's contract carries a method's diagnostics without naming
   * them; `octreeDebugSources` in `lib/octree-debug-sources.ts` is the typed
   * reader on the other side, and the declaration it reads against is the one
   * this bag is built to.
   *
   * Built once and never rebuilt. Harnesses poll these buffers every step, so
   * a fresh object per access would allocate in the hot readback loop; every
   * member is an accessor instead, which also keeps the slot selection live —
   * the projection swaps published/candidate slots between steps, and a bag of
   * captured handles would go on serving the retired one.
   */
  get debug(): Record<string, unknown> {
    return this.octreeDebug ??= this.createDebugSources();
  }
  private octreeDebug?: OctreeDebugSources;
  private createDebugSources(): OctreeDebugSources {
    // Read through a closure rather than capturing the projection: an accessor
    // must observe the projection the solver holds when it is called.
    const projection = () => this.octreeProjection;
    return {
      get powerPressureBuffer() { return projection()?.powerPressureBuffer; },
      get powerLeafHeaders() { return projection()?.powerLeafHeaders; },
      get globalFineTransportControl() { return projection()?.globalFineTransportControl; },
      get globalFineRedistanceControl() { return projection()?.globalFineRedistanceControl; },
      get globalFineVolumeControl() { return projection()?.globalFineVolumeControl; },
      get structuredProjectionEnergyStats() { return projection()?.structuredProjectionEnergyStats; },
      get losassoVelocityDebug() { return projection()?.losassoVelocityDebug; },
      get losassoPressureDebug() { return projection()?.losassoPressureDebug; },
      get globalFinePageDeltaDebug() { return projection()?.globalFinePageDeltaDebug; },
      get ownerLatticeDebug() { return projection()?.ownerLatticeDebug; },
      get globalFineCoarseLevelSetControl() { return projection()?.globalFineCoarseLevelSetControl; },
      get globalFineRestrictionControl() { return projection()?.globalFineRestrictionControl; },
      get airSupportScratch() { return projection()?.airSupportScratch; },
    };
  }
  get columnBaseTexture() { return undefined; }
  get gridCellTexture() { return this.octreeProjection?.topologyTexture; }
  get velocityTexture() { return undefined; }
  get secondaryParticles() { return undefined; }
  /**
   * Adopt the live coarse-band accuracy/frame-time dials.
   *
   * The renderer calls this on every resolved frame with the method's current
   * parameter bag, so it must stay cheap and idempotent: the octree resolves
   * the bag to clamped dial values and every consumer below compares before it
   * writes. Nothing here restarts, re-seeds, or re-allocates — that is the
   * whole point of the keys being declared `update: "runtime"`.
   */
  applyRuntimeValues(values: Record<string, string | number | boolean>) {
    this.octreeProjection?.applyRuntimeDials(resolveOctreeRuntimeDials(values));
  }
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
      this.octreeProjection.rescaleSparsePresentation(scene);
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
  /** Compact velocity authority has no dense pre-projection texture. */
  get preProjectionVelocityTexture() { return undefined; }

  private initializeVolume() {
    const { nx, ny, nz } = this.info, c = this.scene.container;
    const dam = sceneDamBreakBox(this.scene);
    const terrainHeights = terrainColumnHeights(this.scene, nx, nz), cellHeight = c.height_m / ny;
    let initialSum = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const aboveGround = (j + 0.5) * cellHeight > terrainHeights[i + nx * k];
      const fill = aboveGround && initialLiquidContainsCell(this.scene, i, j, k, [nx, ny, nz],
        this.scene.fluid.initialCondition === "dam-break"
          ? damBreakBoxContains(dam, (i + .5) / nx, (j + .5) / ny, (k + .5) / nz)
          : (j + .5) / ny <= c.fillFraction);
      if (fill) initialSum += 1;
    }
    const terrainCells = new Float32Array(nx * nz);
    for (let index = 0; index < terrainCells.length; index++) terrainCells[index] = terrainHeights[index] / cellHeight;
    const terrainRowBytes = nx * 4, terrainPadded = Math.ceil(terrainRowBytes / 256) * 256;
    const terrainPacked = new Uint8Array(terrainPadded * nz), terrainSource = new Uint8Array(terrainCells.buffer);
    for (let k = 0; k < nz; k++) terrainPacked.set(terrainSource.subarray(terrainRowBytes * k, terrainRowBytes * (k + 1)), terrainPadded * k);
    this.device.queue.writeTexture({ texture: this.terrainTexture }, terrainPacked, { bytesPerRow: terrainPadded, rowsPerImage: nz }, { width: nx, height: nz });
    Object.assign(this.info, { initialVolumeCellSum: initialSum,
      volumeCellSum: undefined,
      representedVolumeCellSum: undefined,
      representedVolumeDrift: undefined,
      volumeDrift: undefined,
      rawVolumeDrift: undefined,
      volumeTelemetrySource: "unavailable",
      maxSpeed_m_s: 0,
      front_m: this.scene.fluid.initialCondition === "dam-break" ? -c.width_m / 2 + dam.max.x * c.width_m : c.width_m / 2,
      frontTelemetrySource: "initial-condition" });
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
      label: "Octree pooled statistics readback",
      size: 16 + STRUCTURED_PROJECTION_ENERGY_WORDS * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  }

  advanceTo(time_s: number, bodies: RigidBodyState[] = []) {
    if (this.disposed) return false;
    if (this.fatalPhysicsError) throw new Error(this.fatalPhysicsError);
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
    const c = this.scene.container;
    const substeps = 1;
    const dt = delta / substeps;
    this.info.lastDt_s = undefined;
    this.info.lastSubsteps = undefined;
    this.octreeProjection?.setTimestep(dt);
    const activeBodies = bodies.slice(0, 12);
    this.rigidSystem.syncBodies(activeBodies);
    this.info.encodedSteps = (this.info.encodedSteps ?? 0) + 1;
    this.octreeProjection?.setCouplingBodies(activeBodies.length, activeBodies.some((body) => body.inverseMass_kg > 0));
    const inflow=this.scene.fluid.inflow,inflowStepStrength=inflow?averageInflowStrength(inflow,this.lastTime-delta,this.lastTime):0;
    if(this.inflowBoundary){const cellVolume=c.width_m*c.height_m*c.depth_m/(this.info.nx*this.info.ny*this.info.nz);const cells=this.inflowBoundary.flowRate_m3_s*inflowStepStrength*delta/cellVolume;this.octreeProjection?.addSurfaceReferenceVolumeCells(cells);}
    if (!this.validationChecked) this.device.pushErrorScope("validation");
    let encoder = this.device.createCommandEncoder({ label: "Octree GPU fluid step" });
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
      if (this.octreeProjection) {
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
      }
      // Chentanez & Müller Sec. 3.9.1 runs for every pressure backend. Both
      // adaptive projections constrain the normal face flux variationally;
      // this per-substep Brinkman blend supplies tangential drag and interior
      // momentum for moving bodies. Octree additionally folds the dynamic
      // body's pressure response into the next presentation batch, avoiding a
      // global K^T p reduction at every pressure iterate. Phi-s relaxation
      // keeps the resident level set sane inside either backend's solids so
      // they displace water instead of carrying sealed liquid plugs.
      if (activeBodies.length > 0) {
        // Keep the analytic rigid state on the same temporal grid as the
        // pressure/face boundary condition. Each substep consumes only its
        // own exchange and publishes the updated pose before the next
        // Losasso accepted-face refresh.
        const cellVolume = c.width_m * c.height_m * c.depth_m
          / (this.info.nx * this.info.ny * this.info.nz);
        this.rigidSystem.encode(encoder, dt, cellVolume, 1, c.height_m / this.info.ny);
        if (substep + 1 < substeps) encoder.clearBuffer(this.rigidExchangeBuffer);
      }
    }
    if (activeBodies.length > 0) {
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
        // Factor-one may intentionally withhold `coarseLevelSetSource` until
        // this very snapshot admits the first coherent GPU generation. Detect
        // the authority from the configured tracking factor, not from the
        // renderer view, or construction rejection creates a receipt deadlock.
        const coarseOnly = !fine && this.octreeProjection.surfaceTrackingFactor === 1;
        const adaptive = coarseOnly
          && pending.losassoAdaptiveAcceptedGraphControl !== undefined
          && pending.losassoAdaptiveCandidateGraphControl !== undefined
          && pending.losassoAdaptivePhiControl !== undefined
          && pending.losassoAdaptivePhiReceipts !== undefined
          && pending.losassoAdaptiveVelocityReceipts !== undefined
          && pending.losassoAdaptiveRendererDirectory !== undefined
          && pending.losassoCandidateAuthorityControl !== undefined
          && pending.losassoAdaptiveMassControl !== undefined
          && pending.losassoAdaptiveMassReceipts !== undefined
          && pending.losassoAdaptiveCandidateMassControl !== undefined
          && pending.losassoAdaptiveCandidateMassReceipts !== undefined
          && pending.losassoCandidateVelocityMigrationReceipt !== undefined;
        const nativeSurfaceSources = coarseOnly
          || (pending.globalFineCurrentWorklist !== undefined
            && pending.globalFineCurrentTopologyControl !== undefined
            && pending.globalFineCurrentRedistanceControl !== undefined
            && pending.globalFineCurrentVolumeControl !== undefined
            && this.octreeProjection?.globalFineTransportControl !== undefined);
        const sources = pending.losassoAuthorityControl && pending.losassoCoarsePhiControl
          && pending.losassoExtensionControl && this.mgpcgControl && nativeSurfaceSources ? {
            authority: pending.losassoAuthorityControl,
            solver: this.mgpcgControl,
            ...(pending.globalFineCurrentWorklist
              ? { fineWorklist: pending.globalFineCurrentWorklist } : {}),
            coarsePhi: pending.losassoCoarsePhiControl,
            extension: pending.losassoExtensionControl,
            ...(adaptive ? { adaptive: {
              candidateAuthority: pending.losassoCandidateAuthorityControl!,
              acceptedGraph: pending.losassoAdaptiveAcceptedGraphControl!,
              candidateGraph: pending.losassoAdaptiveCandidateGraphControl!,
              phiControl: pending.losassoAdaptivePhiControl!,
              phiReceipts: pending.losassoAdaptivePhiReceipts!,
              velocityReceipts: pending.losassoAdaptiveVelocityReceipts!,
              rendererDirectory: pending.losassoAdaptiveRendererDirectory!,
              massControl: pending.losassoAdaptiveMassControl!,
              massReceipts: pending.losassoAdaptiveMassReceipts!,
              candidateMassControl: pending.losassoAdaptiveCandidateMassControl!,
              candidateMassReceipts: pending.losassoAdaptiveCandidateMassReceipts!,
              velocityMigration: pending.losassoCandidateVelocityMigrationReceipt!,
            } } : {}),
            ...(pending.fluidBrickResidencyWorklist
              ? { fluidResidency: pending.fluidBrickResidencyWorklist } : {}),
            ...(pending.fluidBulkBrickResidencyWorklist
              ? { fluidBulkResidency: pending.fluidBulkBrickResidencyWorklist } : {}),
            ...(this.octreeProjection?.globalFineTransportControl
              ? { fineTransport: this.octreeProjection.globalFineTransportControl } : {}),
            ...(pending.globalFineCurrentTopologyControl
              ? { fineTopology: pending.globalFineCurrentTopologyControl } : {}),
            ...(pending.globalFineCurrentRedistanceControl
              ? { fineRedistance: pending.globalFineCurrentRedistanceControl } : {}),
            ...(pending.globalFineCurrentVolumeControl
              ? { fineVolume: pending.globalFineCurrentVolumeControl } : {}),
          } : undefined;
        this.losassoStepSnapshotRing ??= new WebGPUOctreeLosassoStepSnapshotRing(
          this.device, structuredStepSnapshotSlotCount());
        if (!this.losassoStepSnapshotRing.recordDue) {
          // The previous record is still the freshest unread one. Encoding
          // another would stage ~8 copies into host-visible memory that no
          // consumer will ever look at. The stage is declared optional, so its
          // absence is not a step-sequence deviation.
        } else if (sources && this.losassoStepSnapshotRing.encode(
          encoder, sources, this.info.encodedSteps ?? 0,
          adaptive ? "adaptive" : coarseOnly ? "coarse" : "fine")) {
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
        if (!this.stepSnapshotRing.recordDue) {
          // See the Losasso branch above: the record is retained, not lost.
        } else if (this.stepSnapshotRing.encode(encoder, {
          structuredVelocityControl: velocityControl,
          structuredBoundaryControl: boundaryControl,
          fineWorklist: surfaceHeader,
          mgpcgControl: this.mgpcgControl,
          fineVolumeControl: this.octreeProjection?.globalFineVolumeControl,
          projectionEnergyStats: this.octreeProjection?.structuredProjectionEnergyStats,
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
        console.error(`Octree GPU validation: ${error.message}`);
      }).catch(() => { /* Device loss is handled by the renderer. */ });
    }
    return true;
  }

  async readStats() {
    if ((this.info.encodedSteps ?? 0) === 0 || this.readbackPending) return this.info;
    this.readbackPending = true;
    // With the step-coherent ring active, projection energy comes from the
    // step's own record; the racing mid-pipeline copy exists only as the
    // pre-first-advance fallback.
    const structuredProjectionEnergy = this.stepSnapshotRing
      ? undefined
      : this.octreeProjection?.structuredProjectionEnergyStats;
    const buffer = structuredProjectionEnergy !== undefined ? this.statsReadback() : undefined;
    if (buffer) {
      const encoder = this.device.createCommandEncoder({ label: "Octree statistics readback" });
      if (structuredProjectionEnergy) encoder.copyBufferToBuffer(
        structuredProjectionEnergy, 0, buffer, 16, STRUCTURED_PROJECTION_ENERGY_WORDS * 4,
      );
      this.device.queue.submit([encoder.finish()]);
    }
    const mapPromise = buffer?.mapAsync(GPUMapMode.READ) ?? Promise.resolve();
    const compactFineExpected = Boolean(this.octreeProjection?.globalFineLevelSetSource);
    const losassoSnapshotRingActive = this.octreeProjection?.coarseBackend === "losasso"
      && this.losassoStepSnapshotRing !== undefined;
    const mayReadLiveDiagnostics = mayReadLiveOctreeDiagnostics(
      this.octreeProjection?.coarseBackend,
      losassoSnapshotRingActive,
    );
    const solveDiagnostics = mayReadLiveDiagnostics
      ? this.octreeProjection?.readSolveDiagnostics() : undefined;
    // Once compact global-fine volume is authoritative, the adaptive surface
    // diagnostic is both obsolete and ignored below. Avoid a separate queue
    // submission/map every 250 ms for data that cannot be selected.
    const surfaceDiagnosticsPromise = compactFineExpected
      ? undefined
      : this.octreeProjection?.readSurfaceDiagnostics();
    const globalFineDiagnosticsPromise = compactFineExpected
      ? this.octreeProjection?.readGlobalFineLevelSetDiagnostics() : undefined;
    // The step-coherent record supersedes the racing live-buffer sample for
    // authority health: its words were copied by the step's own encoder.
    const stepSnapshotPromise = this.stepSnapshotRing?.readLatest();
    const losassoStepSnapshotPromise = this.losassoStepSnapshotRing?.readLatest();
    try {
      await mapPromise;
      const [, , surfaceDiagnostics, globalFineDiagnostics, liveFluidBrickStats, liveFluidBulkBrickStats,
        stepRecord, losassoStepRecord] = await Promise.all([
        this.validationPromise, solveDiagnostics, surfaceDiagnosticsPromise, globalFineDiagnosticsPromise,
        mayReadLiveDiagnostics ? this.octreeProjection?.readFluidBrickResidencyStats() : undefined,
        mayReadLiveDiagnostics ? this.octreeProjection?.readFluidBulkBrickResidencyStats() : undefined,
        stepSnapshotPromise,
        losassoStepSnapshotPromise,
      ]);
    const residency = losassoStepRecord?.fluidResidency;
    const fluidBrickStats = residency && residency.length >= 16
      ? { resident: residency[0] ?? 0, retired: residency[11] ?? 0,
        core: residency[8] ?? 0, halo: residency[9] ?? 0,
        activated: residency[10] ?? 0, generation: residency[15] ?? 0,
        capacity: this.octreeProjection?.fluidBrickCapacity ?? 0 }
      : liveFluidBrickStats;
    const bulkResidency = losassoStepRecord?.fluidBulkResidency;
    const fluidBulkBrickStats = bulkResidency && bulkResidency.length >= 16
      && (this.octreeProjection?.fluidBulkBrickCapacity ?? 0) > 0
      ? { resident: bulkResidency[0] ?? 0, retired: bulkResidency[11] ?? 0,
        core: bulkResidency[8] ?? 0, halo: bulkResidency[9] ?? 0,
        activated: bulkResidency[10] ?? 0, generation: bulkResidency[15] ?? 0,
        capacity: this.octreeProjection?.fluidBulkBrickCapacity ?? 0 }
      : liveFluidBulkBrickStats;
    if (losassoStepRecord) {
      this.octreeProjection?.applyLosassoStepDiagnostics(
        losassoStepRecord.authority,
        losassoStepRecord.solver,
      );
      const failures = losassoStepSnapshotFailures(losassoStepRecord);
      if (failures.length > 0) {
        this.stepSequenceFaulted = true;
        const summary = losassoStepSnapshotDiagnosticSummary(losassoStepRecord, failures);
        this.info.quadtreePressureRejectionSummary = summary;
        this.fatalPhysicsError ??= `Losasso physics receipt rejected; simulation stopped fail-closed: ${summary}`;
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
          fineTopology: Array.from(losassoStepRecord.fineTopology),
          fineRedistance: Array.from(losassoStepRecord.fineRedistance),
          fineVolume: Array.from(losassoStepRecord.fineVolume),
          adaptive: losassoStepRecord.adaptive ? {
            acceptedGraph: Array.from(losassoStepRecord.adaptive.acceptedGraph),
            candidateGraph: Array.from(losassoStepRecord.adaptive.candidateGraph),
            phiControl: Array.from(losassoStepRecord.adaptive.phiControl),
            phiReceipts: Array.from(losassoStepRecord.adaptive.phiReceipts),
            velocityReceipts: Array.from(losassoStepRecord.adaptive.velocityReceipts),
            renderer: Array.from(losassoStepRecord.adaptive.renderer),
            velocityMigration: Array.from(losassoStepRecord.adaptive.velocityMigration),
          } : undefined,
          failures,
        }));
      } else {
        if (losassoStepRecord.surfaceKind === "adaptive" && losassoStepRecord.adaptive) {
          this.octreeProjection?.applyAdaptiveSurfaceGenerationReceipt(
            losassoStepRecord.adaptive.acceptedGraph[5]!,
          );
        }
        this.info.quadtreePressureRejectionSummary = undefined;
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
    const structuredEnergy = this.octreeProjection?.coarseBackend === "power2017" && stepRecord
      ? decodeStructuredProjectionEnergy(stepRecord.snapshot.projectionEnergyControl)
      : structuredProjectionEnergy && buffer
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
    this.info.rawVolumeDrift=undefined;
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
    const adaptiveMassControl=losassoStepRecord?.surfaceKind==="adaptive"
      ? losassoStepRecord.adaptive?.massControl:undefined;
    const adaptiveMassReceipt=adaptiveMassControl
      && adaptiveMassControl[0]===OCTREE_LOSASSO_ADAPTIVE_MASS_MAGIC
      && adaptiveMassControl[7]===1&&adaptiveMassControl[12]===0
      && losassoStepRecord?.adaptive?.massReceipts
      ? unpackAdaptiveMassReceipt(losassoStepRecord.adaptive.massReceipts):undefined;
    const adaptiveMassCells=adaptiveMassReceipt&&adaptiveMassReceipt.errors===0
      && Number.isFinite(adaptiveMassReceipt.acceptedMass_m3)
      ? adaptiveMassReceipt.acceptedMass_m3/baseCellVolume_m3:undefined;
    this.info.adaptiveCompressedExcessVolume_cells=adaptiveMassReceipt
      ? adaptiveMassReceipt.compressedExcessMass_m3/baseCellVolume_m3:undefined;
    this.info.adaptiveSubIsoVolume_cells=adaptiveMassReceipt
      ? adaptiveMassReceipt.subIsoMass_m3/baseCellVolume_m3:undefined;
    this.info.adaptiveOverfullLeafCount=adaptiveMassReceipt?.overfullLeafCount;
    this.info.adaptiveSubIsoLeafCount=adaptiveMassReceipt?.subIsoLeafCount;
    const authoredInitialMassCells=this.info.initialVolumeCellSum;
    // Exact authored volume fractions make the adaptive mass receipt the
    // conservative authority. Retain the legacy phi-page estimator for scenes
    // whose cold mass deliberately came from a different reconstruction.
    const adaptiveMassMatchesAuthored=adaptiveMassCells!==undefined
      && authoredInitialMassCells!==undefined&&authoredInitialMassCells>0
      && Math.abs(adaptiveMassCells-authoredInitialMassCells)
        <=Math.max(1e-4,1e-6*authoredInitialMassCells);
    if(compactVolume){
      this.info.referenceLiquidVolume_cells=compactVolume?.referenceVolumeCells;
      this.info.volumeCellSum=compactVolume?.volumeCells;
      this.info.representedVolumeCellSum=compactVolume?.volumeCells;
      this.info.volumeDrift=compactVolume?.drift;
      this.info.representedVolumeDrift=compactVolume?.drift;
      this.info.volumeTelemetrySource="global-fine";
    }
    else if(adaptiveMassMatchesAuthored){const reference=authoredInitialMassCells!;this.info.referenceLiquidVolume_cells=reference;this.info.volumeCellSum=adaptiveMassCells;this.info.representedVolumeCellSum=adaptiveMassCells;this.info.volumeDrift=(adaptiveMassCells-reference)/reference;this.info.representedVolumeDrift=this.info.volumeDrift;this.info.volumeTelemetrySource="adaptive-conservative-mass";if(surfaceDiagnostics){this.info.phiInterfaceCellCount=surfaceDiagnostics.interfaceCells;this.info.volumeCorrectionNormalSpeed_cells_s=surfaceDiagnostics.correctionSpeed;this.info.volumeControlAgreeWeight=surfaceDiagnostics.volumeControlAgreeWeight;this.info.quadtreeCulledDebrisCells=surfaceDiagnostics.culledDebrisCells;this.info.quadtreeLevelSetMismatchFraction=surfaceDiagnostics.mismatchFraction;this.info.quadtreeVofReconciliationActive=surfaceDiagnostics.reconciliationActive;}}
    else if(surfaceDiagnostics&&!compactFineExpected){const resolved=sparseSurfaceVolumeCells(surfaceDiagnostics,this.info.initialVolumeCellSum??0),reference=Math.max(1,resolved.referenceVolumeCells);this.info.referenceLiquidVolume_cells=resolved.referenceVolumeCells;this.info.volumeCellSum=resolved.volumeCells;this.info.representedVolumeCellSum=resolved.volumeCells;this.info.volumeDrift=(resolved.volumeCells-reference)/reference;this.info.representedVolumeDrift=this.info.volumeDrift;this.info.volumeTelemetrySource="adaptive-pages";this.info.phiInterfaceCellCount=surfaceDiagnostics.interfaceCells;this.info.volumeCorrectionNormalSpeed_cells_s=surfaceDiagnostics.correctionSpeed;this.info.volumeControlAgreeWeight=surfaceDiagnostics.volumeControlAgreeWeight;this.info.quadtreeCulledDebrisCells=surfaceDiagnostics.culledDebrisCells;this.info.quadtreeLevelSetMismatchFraction=surfaceDiagnostics.mismatchFraction;this.info.quadtreeVofReconciliationActive=surfaceDiagnostics.reconciliationActive;}
    else{this.info.referenceLiquidVolume_cells=undefined;this.info.volumeCellSum=undefined;this.info.representedVolumeCellSum=undefined;this.info.volumeDrift=undefined;this.info.representedVolumeDrift=undefined;this.info.volumeTelemetrySource="unavailable";}
    this.info.front_m=undefined;
    this.info.frontTelemetrySource="unavailable";
    this.info.maxSpeed_m_s = undefined;
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
      this.info.quadtreePressureConverged = this.octreeProjection.info.pressureConverged;
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
      if (buffer?.mapState === "mapped") buffer.unmap();
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
    this.terrainTexture.destroy();
    this.rigidSystem.destroy(); this.rigidExchangeBuffer.destroy(); this.statsReadbackBuffer?.destroy();
    this.logicalActivityDiscardRecorder?.destroy();
    this.logicalActivityDiscardRecorder = undefined;
    this.logicalActivityMarkerTickSource?.destroy();
    this.logicalActivityMarkerTick?.destroy();
  }
}

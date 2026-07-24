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
import { OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES, octreeFineEngineSplitsEnabled, WebGPUOctreeProjection,
  type OctreeSemanticPhase, type OctreeProjectionOptions } from "./webgpu-octree";
import { buildFixedAdaptiveOctreePowerGalerkinHierarchy } from "./octree-power-galerkin";
import {
  OCTREE_FACE_BAND_ACUTE_TETRA_FAILURE,
  OCTREE_FACE_BAND_OWNER_FAILURE_STAGE,
  unpackOctreeFaceBandControl,
  unpackOctreeFaceBandPointFieldControl,
  unpackOctreeFaceBandPowerPublication,
  unpackOctreeFaceBandTransientPowerControl,
  unpackOctreeFaceBandTransitionControl,
} from "./webgpu-octree-face-closest-point";
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

export type UniformVelocityTransport = GPUVelocityTransport;
export interface WebGPUUniformEulerianOptions { pressureIterations?: number; velocityTransport?: UniformVelocityTransport; densitySharpening?: boolean; tallCellSettings?: Partial<import("./tall-cell-grid").TallCellSettings>; quadtreeTallCells?: Partial<QuadtreeTallCellProjectionOptions>; octree?: Partial<OctreeProjectionOptions>; /** Allocate escaped spray droplets and set their initial live state. */ secondaryParticles?: boolean; secondaryParticleCapacity?: number; /** Bounded near-interface particle-to-level-set correction; zero keeps spray strictly one-way. */ secondaryParticleSurfaceCorrection?: number; quadtreeRebuildTopology?: boolean; quadtreeRebuildIntervalSteps?: number; quadtreeTopologyStaleSteps?: number; /** Fully GPU-resident every-step topology regeneration (Algorithm 1); default on for uncoupled parallel preconditioners. */ quadtreeInlineRebuild?: boolean; deferPipelineCompilation?: boolean }

// Pipeline objects are immutable and device-scoped. Rebuilding buffers or
// textures for a settings change must not ask the browser/driver to compile
// identical programs again; that compilation can block input for seconds even
// through createComputePipelineAsync.
const uniformPipelineCache = new WeakMap<GPUDevice, Map<UniformVelocityTransport, GPUComputePipeline[]>>();

const OCTREE_SEMANTIC_TRACE_PHASE: Readonly<Record<OctreeSemanticPhase, GPUTimestampPhase>> = {
  structureEpoch: { id: "coarse-grid", label: "[engine:structure-epoch] Structure epoch" },
  rowEngineA: { id: "velocity-advection", label: "[engine:row-a] Coarse values + face completion" },
  solveEngine: { id: "pressure-solve", label: "[engine:solve] Galerkin pressure solve" },
  rowEngineB: { id: "velocity-extrapolation", label: "[engine:row-b] Projection + velocity extension" },
  brickEngineA: { id: "fine-sdf-advection", label: "[engine:brick-a] Fine transport + topology" },
  closestPointWaves: { id: "fine-sdf-redistance", label: "[engine:cpt-waves] Closest-point waves" },
  brickEngineB: { id: "adaptive-publication", label: "[engine:brick-b] Fine harvest + epoch gate" },
  pressureLeafCompactionL1Capture: { id: "pressure-system", label: "Liquid rows + L1 capture" },
  powerDescriptorTopologyFaces: { id: "power-topology", label: "Power topology + physical faces" },
  powerFaceRegularCompletion: { id: "velocity-advection", label: "Generalized-face velocity completion" },
  powerOperatorRhsAssembly: { id: "pressure-system", label: "Power operator + divergence RHS" },
  finalPressureRowAssembly: { id: "pressure-system", label: "Final pressure row assembly" },
  mgpcgSolve: { id: "pressure-solve", label: "Selected power pressure solve" },
  powerProjectionPublication: { id: "velocity-projection", label: "Power-face pressure projection" },
  faceBandTopologyBuild: { id: "velocity-extrapolation", label: "Extrapolation face-band topology" },
  faceBandTransitionAdjacency: { id: "velocity-extrapolation", label: "Transition Delaunay adjacency" },
  faceBandClosestPointExtension: { id: "velocity-extrapolation", label: "Closest-point velocity extension" },
  faceBandPowerPublicationCapture: { id: "velocity-extrapolation", label: "Extended power-face publication" },
  powerProjectionTail: { id: "velocity-projection", label: "Solid impulse + projection publication" },
  finePreparation: { id: "fine-sdf-advection", label: "Fine SDF seed + transport preparation" },
  fineTransport: { id: "fine-sdf-advection", label: "Factor-m fine SDF advection" },
  fineTopology: { id: "coarse-grid", label: "Fine narrow-band topology" },
  fineRedistance: { id: "fine-sdf-redistance", label: "Fine SDF redistance" },
  fineRestriction: { id: "adaptive-publication", label: "Fine-to-coarse restriction" },
};

/** Serial queue probes are deliberately sparse because they trade throughput
 * for portable, per-phase measurements when timestamp queries fail. */
const SEGMENTED_QUEUE_TRACE_CADENCE_MS = 3_000;

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
  readonly published: boolean;
  readonly rolledBack: boolean;
  readonly downstreamFinalizeReason: number;
  readonly generation: number;
  readonly volumeControl: readonly number[];
}

export interface InitialGlobalFineAuthorityDiagnostics extends GlobalFineVolumePublicationDiagnostics {
  readonly seedControl?: readonly number[];
  readonly topologyControl?: readonly number[];
  readonly worklistHeader?: readonly number[];
  readonly coarseDirectoryHeader?: readonly number[];
  readonly coarseControl?: readonly number[];
  readonly fineRestrictionControl?: readonly number[];
  readonly seedCount: number;
  readonly seedError: number;
  readonly topologyFlags: number;
  readonly interfaceBricks: number;
  readonly interfaceSeedBricks: number;
  readonly desiredBricks: number;
  readonly activatedBricks: number;
  readonly activeBricks: number;
  readonly configuredFineGeneration: number;
  readonly scheduledFineGeneration: number;
  readonly coarseDirectoryState: number;
  readonly coarseDirectoryGeneration: number;
  readonly coarseControlFlags: number;
  readonly coarseControlGeneration: number;
  readonly coarseControlValid: number;
  readonly fineRestrictionFlags: number;
  readonly fineRestrictionUnowned: number;
  readonly fineRestrictionRows: number;
  readonly fineRestrictionValid: number;
  readonly transportControl: readonly number[];
  readonly redistanceControl: readonly number[];
  readonly redistanceControlDetailed?: readonly number[];
  readonly powerVelocityControl?: readonly number[];
  readonly powerFaceControl?: readonly number[];
  readonly powerFaceCandidateControl?: readonly number[];
  readonly powerRowDeltaControl?: readonly number[];
  readonly powerDescriptorControl?: readonly number[];
  readonly powerTopologyControl?: readonly number[];
  readonly faceBandControl: readonly number[];
  readonly faceBandCandidateControl?: readonly number[];
  readonly faceBandTransitionControl: readonly number[];
  readonly faceBandCandidateTransitionControl?: readonly number[];
  readonly faceBandTransitionOwnerFailure?: readonly number[];
  readonly faceBandPointFieldControl: readonly number[];
  readonly faceBandTransientPowerControl: readonly number[];
  readonly faceBandPowerPublicationControl: readonly number[];
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
  info.globalFineTransportFaceBandUnavailable = transport.faceBandUnavailable;
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
  const topologyWords=value.topologyControl??[
    value.topologyFlags,value.interfaceBricks,value.desiredBricks,value.activatedBricks,
    value.published?1:0,value.rolledBack?1:0,0,value.downstreamFinalizeReason,
    value.interfaceSeedBricks,
  ];
  const redistanceWords=value.redistanceControlDetailed??value.redistanceControl;
  const volumeBytes=new ArrayBuffer(64);new Uint32Array(volumeBytes).set(value.volumeControl.slice(0,16));
  const coarseWords=value.coarseControl??[
    value.coarseControlFlags,0,0,0,0,0,0,0,0,0,value.coarseControlGeneration,
    value.coarseControlValid,0,0,0,0,
  ];
  const restrictionWords=value.fineRestrictionControl??[
    0,0,value.fineRestrictionFlags,value.fineRestrictionUnowned,
    value.fineRestrictionRows,value.fineRestrictionValid,0,0,
  ];
  const topology=unpackFineLevelSetGPUTopologyControl(topologyWords);
  const redistance=unpackFineLevelSetGPURedistanceControl(redistanceWords);
  const volume=unpackFineLevelSetGPUVolumeControl(volumeBytes);
  const transitionWords = [...value.faceBandTransitionControl,
    ...(value.faceBandTransitionOwnerFailure ?? [])];
  const coarse=unpackOctreePowerCoarseLevelSetControl(coarseWords);
  const restriction=unpackFineToCoarseGPUControl(restrictionWords);
  return {
    generation:{current:value.generation,configured:value.configuredFineGeneration,
      scheduled:value.scheduledFineGeneration},
    seeds:{count:value.seedCount,flags:value.seedError,raw:value.seedControl},
    topology:{...topology,
      errors:namedControlBits(topology.flags,FINE_LEVELSET_TOPOLOGY_ERROR),
      downstream:namedControlBits(topology.downstreamFinalizeReason,FINE_LEVELSET_TOPOLOGY_FINALIZE_REASON)},
    worklist:value.worklistHeader,
    redistance:{...redistance,errors:namedControlBits(redistance.flags,
      {capacity:1,hashProbe:2,staleGeneration:4,nonfinite:8,conflictingRequest:16})},
    volume,
    coarseDirectory:value.coarseDirectoryHeader??{
      state:value.coarseDirectoryState,generation:value.coarseDirectoryGeneration,
    },
    coarse:{...coarse,errors:namedControlBits(coarse.flags,OCTREE_POWER_COARSE_LEVELSET_ERROR)},
    restriction:{...restriction,errors:namedControlBits(restriction.flags,FINE_TO_COARSE_LEVELSET_ERROR)},
    powerFaces:value.powerFaceControl,
    powerFaceCandidate:value.powerFaceCandidateControl,
    powerRowDelta:value.powerRowDeltaControl,
    powerDescriptor:value.powerDescriptorControl,
    powerTopology:value.powerTopologyControl,
    powerVelocity:value.powerVelocityControl,
    section5:{
      faceBand:unpackOctreeFaceBandControl(value.faceBandControl),
      candidateFaceBand:value.faceBandCandidateControl
        ? unpackOctreeFaceBandControl(value.faceBandCandidateControl) : undefined,
      transition:unpackOctreeFaceBandTransitionControl(transitionWords),
      candidateTransition:value.faceBandCandidateTransitionControl
        ? unpackOctreeFaceBandTransitionControl(value.faceBandCandidateTransitionControl) : undefined,
      pointField:unpackOctreeFaceBandPointFieldControl(value.faceBandPointFieldControl),
      transientPower:unpackOctreeFaceBandTransientPowerControl(value.faceBandTransientPowerControl),
      powerPublication:unpackOctreeFaceBandPowerPublication(value.faceBandPowerPublicationControl),
    },
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
  if (value.seedCount === 0 || value.seedError !== 0) {
    return rejected(`global-fine interface seeds are invalid (${value.seedCount}, fault ${value.seedError})`);
  }
  // Aanjaneya et al. Section 5 constructs a fresh SPGrid from copied
  // interface values. The cold predecessor is deliberately empty, so its
  // resident-page discovery count is zero; external seeds are the explicit
  // cold-only interface proof. Recurring generations must still discover an
  // interface from their transported predecessor.
  if (!value.published || value.rolledBack || value.topologyFlags !== 0
    || value.downstreamFinalizeReason !== 0 || value.desiredBricks === 0
    || value.activatedBricks === 0 || value.activeBricks === 0
    || (!options.externallySeededColdBootstrap && value.interfaceBricks === 0)) {
    return rejected("global-fine topology rejected");
  }
  if (generation === 0 || (value.generation & 0x3fff_ffff) !== generation
    || (value.scheduledFineGeneration & 0x3fff_ffff) !== generation) {
    return rejected("global-fine topology generation is stale");
  }
  if (value.coarseDirectoryState !== OCTREE_POWER_COARSE_LEVELSET_VALID
    || value.coarseControlValid !== OCTREE_POWER_COARSE_LEVELSET_VALID
    || value.coarseControlFlags !== 0
    || (value.coarseDirectoryGeneration & 0x3fff_ffff) !== generation
    || (value.coarseControlGeneration & 0x3fff_ffff) !== generation) {
    return rejected("compact coarse level set is not paired with the fine generation");
  }
  // Fine-band samples need not own a liquid pressure row: after an advective
  // step the authoritative fine interface can lead the row topology by a
  // subcell distance. Restriction counts these misses observationally while
  // consumers sample fine before coarse fallback. Therefore validity/flags,
  // not the raw miss count, are the authority predicate.
  if (value.fineRestrictionFlags !== 0 || value.fineRestrictionRows === 0
    || value.fineRestrictionValid !== OCTREE_POWER_COARSE_LEVELSET_VALID) {
    return rejected("fine-to-coarse level-set restriction did not publish");
  }
  const faceBand = unpackOctreeFaceBandControl(value.faceBandControl);
  const transition = unpackOctreeFaceBandTransitionControl(value.faceBandTransitionControl);
  const pointField = unpackOctreeFaceBandPointFieldControl(value.faceBandPointFieldControl);
  const transientPower = unpackOctreeFaceBandTransientPowerControl(value.faceBandTransientPowerControl);
  const powerPublication = unpackOctreeFaceBandPowerPublication(value.faceBandPowerPublicationControl);
  if (!faceBand.valid || faceBand.rowCount === 0 || faceBand.generation !== generation
    || !transition.ready || !transition.transferReady || transition.rowCount === 0
    || !pointField.valid || pointField.rowCount === 0 || pointField.generation !== generation
    || !transientPower.valid || transientPower.rowCount === 0 || transientPower.generation !== generation
    || !powerPublication.valid || powerPublication.fineGeneration !== generation) {
    return rejected("Section 5 velocity-band round trip did not publish");
  }
  return { ready: true, label: `global fine/coarse and Section 5 generation ${generation} published` };
}

export interface InitialPowerPressureDiagnostics {
  readonly authoritative: boolean;
  readonly solverLabel: string;
  readonly pressureRows: number;
  readonly pressureEntries: number;
  readonly capacityOverflow: boolean;
  readonly mgpcgControl?: Uint32Array;
}

/** Both explicitly selectable power solvers use the same fail-closed control
 * ABI and 1e-4 native-L2 residual gate. A zero-RHS t=0 solve is valid when the
 * selected GPU authority marks it converged and publishes finite data. */
export function initialPowerPressureReadiness(
  value: InitialPowerPressureDiagnostics,
): InitialSparseAuthorityReadiness {
  const section43 = value.solverLabel.includes("Section 4.3 hybrid");
  const fixedGalerkin = value.solverLabel.includes("fixed native-L2 Galerkin");
  if (!value.authoritative || (!section43 && !fixedGalerkin)) {
    return { ready: false, label: "selected power pressure authority is unavailable" };
  }
  if (value.capacityOverflow || value.pressureRows === 0 || value.pressureEntries === 0) {
    return { ready: false, label: "power pressure CSR did not publish" };
  }
  const words = value.mgpcgControl;
  if (!words || words.length < 16) return { ready: false, label: "selected pressure control is unavailable" };
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const residualSquared = floats[4], rhsSquared = floats[5];
  const relativeSquared = residualSquared / Math.max(rhsSquared, 1e-30);
  const residualAccepted = relativeSquared <= 1e-8
    || (fixedGalerkin && residualSquared <= value.pressureRows * 1e-14);
  if (words[0] !== 0 || words[1] === 0 || words[3] !== value.pressureRows
    || !Number.isFinite(residualSquared) || residualSquared < 0
    || !Number.isFinite(rhsSquared) || rhsSquared < 0
    || !Number.isFinite(relativeSquared) || !residualAccepted) {
    return { ready: false, label: "selected pressure solver did not converge through its residual gate" };
  }
  return { ready: true, label: `${fixedGalerkin ? "fixed Galerkin" : "Section 4.3"} power pressure published (${value.pressureRows} rows)` };
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
    || diagnostics.volumeControl.length < 16 || !(baseCellVolume_m3 > 0)
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
  private readonly inflowBoundary?: InflowGridBoundary;
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
    readonly scene: SceneDescription,
    quality: GPUQuality,
    private onRigidLoads?: (loads: GPURigidLoad[]) => void,
    options: WebGPUUniformEulerianOptions = {}
  ) {
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
        powerPressureSolver: options.octree.powerPressureSolver,
        powerGalerkinHierarchy: options.octree.powerPressureSolver === "galerkin"
          ? buildFixedAdaptiveOctreePowerGalerkinHierarchy(
            [nx, ny, nz],
            options.octree.maximumLeafSize ?? 16,
            {
            transfer: "trilinear",
            coarsestNodeLimit: 64,
            },
          )
          : undefined,
        powerBoundarySmoothingIterations: options.octree.powerBoundarySmoothingIterations,
        maximumLeafSize: options.octree.maximumLeafSize ?? 16,
        adaptivity: options.octree.adaptivity ?? 1,
        interfaceRefinementBandCells: options.octree.interfaceRefinementBandCells ?? 4,
        globalFineLevelSetFactor: options.octree.globalFineLevelSetFactor ?? 4,
        globalFineLevelSetMaximumBricks: options.octree.globalFineLevelSetMaximumBricks,
        pressureRowCapacity: options.octree.pressureRowCapacity,
        energyLedger: options.octree.energyLedger,
        energyLedgerStepCapacity: options.octree.energyLedgerStepCapacity,
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
      this.octreeProjection.finishInlineRebuild();
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
    const coarse=unpackOctreePowerCoarseLevelSetControl(value.coarseControl ?? [
      value.coarseControlFlags,0,0,0,0,0,0,0,0,0,value.coarseControlGeneration,
      value.coarseControlValid,0,0,0,0,
    ]);
    const faceBand=unpackOctreeFaceBandControl(value.faceBandControl);
    const transition=unpackOctreeFaceBandTransitionControl([...value.faceBandTransitionControl,
      ...(value.faceBandTransitionOwnerFailure ?? [])]);
    const pointField=unpackOctreeFaceBandPointFieldControl(value.faceBandPointFieldControl);
    const transientPower=unpackOctreeFaceBandTransientPowerControl(value.faceBandTransientPowerControl);
    const powerPublication=unpackOctreeFaceBandPowerPublication(value.faceBandPowerPublicationControl);
    this.info.globalFineSeedCount=value.seedCount;this.info.globalFineSeedError=value.seedError;
    this.info.globalFineTopologyFlags=value.topologyFlags;
    this.info.globalFineDownstreamFinalizeReason=value.downstreamFinalizeReason;
    this.info.globalFineInterfaceBricks=value.interfaceBricks;this.info.globalFineDesiredBricks=value.desiredBricks;
    this.info.globalFineActivatedBricks=value.activatedBricks;this.info.globalFinePublished=value.published;
    this.info.globalFineRolledBack=value.rolledBack;this.info.globalFineActiveBricks=value.activeBricks;
    this.info.globalFineGeneration=value.generation;
    this.info.globalFineRedistanceUnresolvedCells=value.redistanceControl[0];
    this.info.globalFineRedistanceSeeds=value.redistanceControl[2];
    this.info.globalFineRedistanceCommitted=value.redistanceControl[3]!==0;
    this.info.globalFineVolumeFlags=value.volumeControl[0];
    applyGlobalFineTransportDiagnostics(this.info,value.transportControl);
    this.info.globalFineFaceBandFlags=value.faceBandControl[0];
    this.info.globalFineFaceBandTransitionFlags=value.faceBandTransitionControl[0];
    this.info.globalFineFaceBandPowerPublicationFlags=value.faceBandPowerPublicationControl[0];
    this.info.globalFineFaceBandTransientPowerFlags=value.faceBandTransientPowerControl[0];
    this.info.globalFineFaceBandPointFieldFlags=value.faceBandPointFieldControl[0];
    this.info.globalFineCoarseLevelSetFlags=coarse.flags;
    this.info.globalFineCoarseLevelSetFirstErrorRow=coarse.firstErrorRow;
    this.info.globalFineFaceBandFirstError=faceBand.firstError;
    this.info.globalFineFaceBandRowCount=faceBand.rowCount;
    this.info.globalFineFaceBandFaceCount=faceBand.faceCount;
    this.info.globalFineFaceBandIncidenceCount=faceBand.incidenceCount;
    this.info.globalFineFaceBandSeedCount=faceBand.seedCount;
    this.info.globalFineFaceBandAcceptedCount=faceBand.acceptedCount;
    this.info.globalFineFaceBandUnresolvedCount=faceBand.unresolvedCount;
    this.info.globalFineFaceBandSampleFailures=faceBand.sampleFailures;
    this.info.globalFineFaceBandCoarsePhiSamples=faceBand.coarsePhiSamples;
    this.info.globalFineFaceBandCoarsePhiFailures=faceBand.coarsePhiFailures;
    this.info.globalFineFaceBandPhiExtensions=faceBand.bandPhiExtensions;
    this.info.globalFineFaceBandClosestPointFaces=faceBand.closestPointFaces;
    this.info.globalFineFaceBandClosestPointFailures=faceBand.closestPointFailures;
    this.info.globalFineFaceBandLiquidInterpolationFailures=faceBand.liquidInterpolationFailures;
    this.info.globalFineFaceBandCptNoOwnerFailures=faceBand.cptNoOwnerFailures;
    this.info.globalFineFaceBandCptSupportOwnerFailures=faceBand.cptSupportOwnerFailures;
    this.info.globalFineFaceBandCptNoContainingSimplexFailures=faceBand.cptNoContainingSimplexFailures;
    this.info.globalFineFaceBandCptMissingLiquidVertexFailures=faceBand.cptMissingLiquidVertexFailures;
    this.info.globalFineFaceBandTransitionFirstError=transition.firstError;
    this.info.globalFineFaceBandTransitionRowCount=transition.rowCount;
    this.info.globalFineFaceBandTransitionRows=transition.transitionRows;
    this.info.globalFineFaceBandTransitionAdjacencyCount=transition.adjacencyCount;
    this.info.globalFineFaceBandTransitionCoreRows=transition.coreRowCount;
    this.info.globalFineFaceBandTransitionSupport1Rows=transition.support1RowCount;
    this.info.globalFineFaceBandTransitionSupport2Rows=transition.support2RowCount;
    this.info.globalFineFaceBandTransitionSupport3Rows=transition.support3NodeRowCount;
    this.info.globalFineFaceBandTransitionEndpointRows=transition.endpointRowCount;
    this.info.globalFineFaceBandBoundaryGhostRequests=transition.boundaryGhostRequests;
    this.info.globalFineFaceBandPhiFailureCounts=transition.phiFailureCounts;
    this.info.globalFineFaceBandPhiFailure=transition.phiFailure;
    this.info.globalFineFaceBandTransientPowerFirstError=transientPower.firstError;
    this.info.globalFineFaceBandTransientPowerRows=transientPower.rowCount;
    this.info.globalFineFaceBandTransientPowerEmitted=transientPower.emittedCount;
    this.info.globalFineFaceBandTransientPowerSampled=transientPower.sampledCount;
    this.info.globalFineFaceBandTransientPowerValidated=transientPower.validatedCount;
    this.info.globalFineFaceBandPointFieldFirstError=pointField.firstError;
    this.info.globalFineFaceBandPointFieldRows=pointField.rowCount;
    this.info.globalFineFaceBandPointFieldSolved=pointField.solvedCount;
    this.info.globalFineFaceBandPointFieldWallContributions=pointField.wallContributions;
    this.info.globalFineFaceBandPowerPublicationFirstError=powerPublication.firstError;
    this.info.globalFineFaceBandPowerPublicationFaces=powerPublication.faceCount;
    this.info.globalFineFaceBandPowerPublicationTargets=powerPublication.targetCount;
    this.info.globalFineFaceBandPowerPublicationInterpolated=powerPublication.interpolatedCount;
    this.info.globalFineFaceBandPowerPublicationCommitted=powerPublication.committedCount;
    this.info.globalFineFaceBandGeneration=faceBand.generation;
    this.info.globalFineFaceBandValid=faceBand.valid;
    this.info.globalFineFaceBandTransitionValid=transition.ready&&transition.transferReady&&transition.flags===0;
    this.info.globalFineFaceBandPointFieldValid=pointField.valid;
    this.info.globalFineFaceBandTransientPowerValid=transientPower.valid;
    this.info.globalFineFaceBandPowerPublicationValid=powerPublication.valid;
    this.info.globalFineFaceBandPowerFineGeneration=powerPublication.fineGeneration;
    this.info.globalFineFaceBandPowerGeneration=powerPublication.powerGeneration;
  }

  /** The paper path must be complete before the first trajectory can be
   * requested. These are one-time post-fence readbacks for UI readiness and
   * diagnostics; recurring frame scheduling remains GPU-resident. */
  private async validateInitialSparseAuthority() {
    const projection=this.octreeProjection;
    if(!projection)throw new Error("Initial sparse authority requires an octree projection");
    const [,fine,mgpcg]=await Promise.all([
      projection.readSolveDiagnostics(),
      projection.readGlobalFineLevelSetDiagnostics(),
      projection.readMGPCGDiagnostics(),
    ]);
    this.applyOctreeInfo(projection);
    if(projection.globalFineLevelSetSource){
      const readiness=initialGlobalFineAuthorityReadiness(fine,
        { externallySeededColdBootstrap: true });
      if(!readiness.ready){
        const failureRow=await projection.readPowerCoarseFailureRow(fine?.coarseControl?.[1]??0xffff_ffff);
        const candidate=fine?.powerFaceCandidateControl;
        const reciprocalPair=candidate&&((candidate[3]??0)&16)!==0
          ? await projection.readPowerFaceCandidateFailurePair(
            candidate[4]??0xffff_ffff,candidate[13]??0xffff_ffff)
          : undefined;
        const powerFailure=await projection.readGlobalFinePowerPublicationFailure(
          fine?.faceBandPowerPublicationControl?.[1]??0xffff_ffff);
        const powerFrontierFailure=await projection.readPowerFrontierFailure();
        const ownerPageControl=await projection.readOwnerPageControl();
        const faceBandWords=fine?.faceBandCandidateControl??fine?.faceBandControl;
        const faceBand=faceBandWords
          ? unpackOctreeFaceBandControl(faceBandWords) : undefined;
        const candidateTransition=fine?.faceBandCandidateTransitionControl
          ? unpackOctreeFaceBandTransitionControl(fine.faceBandCandidateTransitionControl)
          : undefined;
        const transientPower=fine?.faceBandTransientPowerControl
          ? unpackOctreeFaceBandTransientPowerControl(fine.faceBandTransientPowerControl)
          : undefined;
        const transientPowerFailure=transientPower
          &&transientPower.firstError!==0xffff_ffff
          ? transientPower.failureDomain==="face"
            ? {
                domain:"face",
                face:await projection.readGlobalFineTransientPowerFaceFailure(transientPower.firstError),
              }
            : transientPower.firstError<transientPower.rowCount
              ? {
                  domain:"row",
                  row:await projection.readGlobalFineBandRowFailure(transientPower.firstError),
                  diagnostic:transientPower.diagnostic,
                }
              : {domain:"invalid",firstError:transientPower.firstError}
          : undefined;
        const ownerPageFailure=await projection.readOwnerPageForPowerRow(faceBand?.firstError??0);
        const faceBandCoarseRow=await projection.readPowerCoarseFailureRow(faceBand?.firstError??0);
        const powerSeedChain=await projection.readPowerSeedChainControls();
        const descriptorFailure=await projection.readPowerDescriptorCandidateFailure(
          fine?.powerDescriptorControl?.[3]??0xffff_ffff);
        const faceBandFailures: { cause: string; detail: unknown }[]=[];
        if(faceBand){
          const faceEmission=faceBand.stageFirstFailures.faceEmission;
          if(faceEmission!==0xffff_ffff){
            const producerReason=(faceBandWords?.[7]??0)&0xfff;
            const detail=producerReason===34
              ? await projection.readGlobalFineCandidateBandIncidenceFailure(faceBand.rowCount)
              : await projection.readGlobalFineCandidateBandRowFailure(faceEmission);
            if(detail)faceBandFailures.push({cause:"faceEmission",detail});
          }
          if(faceBand.firstPhiFailureSlot!==0xffff_ffff){
            const detail=await projection.readGlobalFineCandidateBandFaceFailure(faceBand.firstPhiFailureSlot);
            if(detail)faceBandFailures.push({cause:"phiClosestPoint",detail});
          }
          const phiStage=faceBand.stageFirstFailures.phi;
          if((phiStage&OCTREE_FACE_BAND_ACUTE_TETRA_FAILURE)!==0){
            const detail=await projection.readGlobalFineCandidateBandAcuteTetraFailure(phiStage);
            if(detail)faceBandFailures.push({cause:"acuteTetraRealization",detail});
          }else if(faceBand.firstPhiFailureSlot===0xffff_ffff
            &&candidateTransition?.ownerFailure?.stage===OCTREE_FACE_BAND_OWNER_FAILURE_STAGE.bandPhi){
            const detail=await projection.readGlobalFineCandidateBandRowFailure(
              candidateTransition.ownerFailure.band);
            if(detail)faceBandFailures.push({cause:"phiRow",detail});
          }
          const vectorRow=faceBand.stageFirstFailures.vectorReconstruction;
          if(vectorRow!==0xffff_ffff&&vectorRow<faceBand.rowCount){
            const detail=await projection.readGlobalFineCandidateBandRowFailure(vectorRow);
            if(detail)faceBandFailures.push({cause:"vectorReconstruction",detail});
          }
          for(const [cause,slot] of Object.entries(faceBand.firstClosestPointFailureSlotByCause)){
            if(slot!==0xffff_ffff){
              const detail=await projection.readGlobalFineCandidateBandFaceFailure(slot);
              if(detail)faceBandFailures.push({cause,detail});
            }
          }
        }
        throw new Error(`Paused t=0 authority rejected: ${readiness.label}${failureRow
          ? `; coarseFailureRow=${JSON.stringify(failureRow)}`:""}${powerFailure
          ? `; powerPublicationFailure=${JSON.stringify(powerFailure)}`:""}${reciprocalPair
          ? `; powerFaceReciprocalPair=${JSON.stringify(reciprocalPair)}`:""}${faceBandFailures.length
          ? `; faceBandFailures=${JSON.stringify(faceBandFailures)}`:""}; ownerPageControl=${JSON.stringify(ownerPageControl)}`
          + `${transientPowerFailure
            ? `; transientPowerFailure=${JSON.stringify(transientPowerFailure)}`:""}`
          + `${ownerPageFailure?`; ownerPageFailure=${JSON.stringify(ownerPageFailure)}`:""}`
          + `${faceBandCoarseRow?`; faceBandCoarseRow=${JSON.stringify(faceBandCoarseRow)}`:""}`
          + `; faceBandProducerFailure=${faceBandWords?.[7]??0xffff_ffff}`
          + `; powerProjectionControl=${JSON.stringify(fine?.powerProjectionControl??[])}`
          + `; powerFrontierFailure=${JSON.stringify(powerFrontierFailure)}`
          + `${descriptorFailure?`; powerDescriptorFailure=${JSON.stringify(descriptorFailure)}`:""}`
          + `; mgpcg=${JSON.stringify(mgpcg)}`
          + `${powerSeedChain?`; powerSeedChain=${JSON.stringify(powerSeedChain)}`:""}`);
      }
      this.applyGlobalFineDiagnostics(fine!);
      const c=this.scene.container;
      const baseCellVolume_m3=c.width_m*c.height_m*c.depth_m/(this.info.nx*this.info.ny*this.info.nz);
      const volume=publishedGlobalFineVolumeCells(fine!,baseCellVolume_m3);
      if(!volume)throw new Error("Paused t=0 authority rejected: global-fine volume publication is invalid");
      this.info.referenceLiquidVolume_cells=volume.referenceVolumeCells;
      this.info.volumeCellSum=volume.volumeCells;this.info.representedVolumeCellSum=volume.volumeCells;
      this.info.volumeDrift=volume.drift;this.info.representedVolumeDrift=volume.drift;
      this.info.volumeTelemetrySource="global-fine";
    }
    const pressure=initialPowerPressureReadiness({
        authoritative:projection.info.powerDiagramAuthoritative,
        solverLabel:projection.pressureSolverLabel,
        pressureRows:projection.info.pressureRequiredRows??0,
        pressureEntries:projection.info.pressureRequiredEntries??0,
        capacityOverflow:projection.info.pressureCapacityOverflow??false,
        mgpcgControl:mgpcg,
      });
      if(!pressure.ready)throw new Error(`Paused t=0 authority rejected: ${pressure.label}`);
      const floats=new Float32Array(mgpcg!.buffer,mgpcg!.byteOffset,mgpcg!.length);
      this.info.quadtreePressureConverged=true;this.info.quadtreePressureIterationsUsed=mgpcg![2];
      this.info.pressureResidual=Math.sqrt(Math.max(0,floats[4]));
    this.info.pressureRelativeResidual=Math.sqrt(floats[4]/Math.max(floats[5],1e-30));
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
  get powerFaceSeedControl() { return this.octreeProjection?.powerFaceSeedControl; }
  get powerFaceAdvectionControl() { return this.octreeProjection?.powerFaceAdvectionControl; }
  get powerSolidFaceControl() { return this.octreeProjection?.powerSolidFaceControl; }
  get powerOperatorControl() { return this.octreeProjection?.powerOperatorControl; }
  /** QA-only passthrough for the authoritative Section 4.3 solver status. */
  get mgpcgControl() { return this.octreeProjection?.mgpcgControl; }
  get powerFaceControl() { return this.octreeProjection?.powerFaceControl; }
  get powerFaceCandidateControl() { return this.octreeProjection?.powerFaceCandidateControl; }
  get powerFaceSource() { return this.octreeProjection?.powerFaceSource; }
  readPowerEnergyLedger() { return this.octreeProjection?.readEnergyLedger(); }
  get powerBoundaryPhiQueries() { return this.octreeProjection?.powerBoundaryPhiQueries; }
  get ownerLatticeDebug() { return this.octreeProjection?.ownerLatticeDebug; }
  get powerBoundaryFineSource() { return this.octreeProjection?.powerBoundaryFineSource; }
  get powerBoundaryFineLevelSetSource() { return this.octreeProjection?.powerBoundaryFineLevelSetSource; }
  get powerFaceRowDirectory() { return this.octreeProjection?.powerFaceRowDirectory; }
  get powerDescriptorControl() { return this.octreeProjection?.powerDescriptorControl; }
  get powerTopologyControl() { return this.octreeProjection?.powerTopologyControl; }
  get powerDescriptorRows() { return this.octreeProjection?.powerDescriptorRows; }
  get powerTopologyMetrics() { return this.octreeProjection?.powerTopologyMetrics; }
  get powerCatalogEntryHeaders() { return this.octreeProjection?.powerCatalogEntryHeaders; }
  get powerCatalogFaces() { return this.octreeProjection?.powerCatalogFaces; }
  get powerLeafHeaders() { return this.octreeProjection?.powerLeafHeaders; }
  /** QA-only passthrough for compact reconstructed-velocity readback. */
  get powerCellVelocityBuffer() { return this.octreeProjection?.powerCellVelocityBuffer; }
  get powerLeafEntries() { return this.octreeProjection?.powerLeafEntries; }
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
  get globalFinePowerVelocityControl() { return this.octreeProjection?.globalFinePowerVelocityControl; }
  get globalFinePowerProjectionControl() { return this.octreeProjection?.globalFinePowerProjectionControl; }
  get globalFinePowerVelocitySampleControl() { return this.octreeProjection?.globalFinePowerVelocitySampleControl; }
  get globalFineCoarseLevelSetControl() { return this.octreeProjection?.globalFineCoarseLevelSetControl; }
  readPowerCoarseFailureRow(row: number) { return this.octreeProjection?.readPowerCoarseFailureRow(row); }
  readPowerFaceCandidateFailurePair(row: number, neighbor: number) {
    return this.octreeProjection?.readPowerFaceCandidateFailurePair(row, neighbor);
  }
  readPowerFaceCandidateFailure(faceIndex: number) {
    return this.octreeProjection?.readPowerFaceCandidateFailure(faceIndex);
  }
  get globalFineRestrictionControl() { return this.octreeProjection?.globalFineRestrictionControl; }
  get globalFineFaceBandControl() { return this.octreeProjection?.globalFineFaceBandControl; }
  get globalFineFaceBandCandidateControl() {
    return this.octreeProjection?.globalFineFaceBandCandidateControl;
  }
  get globalFineFaceBandTransitionControl() {
    return this.octreeProjection?.globalFineFaceBandTransitionControl;
  }
  get globalFineFaceBandCandidateTransitionControl() {
    return this.octreeProjection?.globalFineFaceBandCandidateTransitionControl;
  }
  get globalFineFaceBandPointFieldControl() {
    return this.octreeProjection?.globalFineFaceBandPointFieldControl;
  }
  get globalFineFaceBandTransientPowerControl() {
    return this.octreeProjection?.globalFineFaceBandTransientPowerControl;
  }
  get globalFineFaceBandPowerPublicationControl() {
    return this.octreeProjection?.globalFineFaceBandPowerPublicationControl;
  }
  get globalFineFaceBandPlan() { return this.octreeProjection?.globalFineFaceBandPlan; }
  readGlobalFineBandRowFailure(index: number) {
    return this.octreeProjection?.readGlobalFineBandRowFailure(index);
  }
  readGlobalFineCandidateBandRowFailure(index: number) {
    return this.octreeProjection?.readGlobalFineCandidateBandRowFailure(index);
  }
  readGlobalFineBandFaceFailure(slot: number) {
    return this.octreeProjection?.readGlobalFineBandFaceFailure(slot);
  }
  readGlobalFineCandidateBandFaceFailure(slot: number) {
    return this.octreeProjection?.readGlobalFineCandidateBandFaceFailure(slot);
  }
  readGlobalFineCandidateBandIncidenceFailure(rowCount: number) {
    return this.octreeProjection?.readGlobalFineCandidateBandIncidenceFailure(rowCount);
  }
  readGlobalFineTransientPowerFaceFailure(slot: number) {
    return this.octreeProjection?.readGlobalFineTransientPowerFaceFailure(slot);
  }
  readGlobalFineBandAcuteTetraFailure(tagged: number) {
    return this.octreeProjection?.readGlobalFineBandAcuteTetraFailure(tagged);
  }
  readGlobalFineCandidateBandAcuteTetraFailure(tagged: number) {
    return this.octreeProjection?.readGlobalFineCandidateBandAcuteTetraFailure(tagged);
  }
  get columnBaseTexture() { return this.hostAllocation ? this.heightA : undefined; }
  get gridCellTexture() { return this.adaptiveProjection?.topologyTexture; }
  get velocityTexture() { return this.octreeProjection ? undefined : this.velocityA; }
  get secondaryParticles() { return undefined; }
  applyRuntimeValues(_values: Record<string, string | number | boolean>) {}
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
      pressureEntryCapacity: octree.pressureEntryCapacity,
      pressureRequiredRows: octree.pressureRequiredRows,
      pressureRequiredEntries: octree.pressureRequiredEntries,
      pressureCapacityOverflow: octree.pressureCapacityOverflow,
      powerDiagramReady: octree.powerDiagramReady,
      powerDiagramAuthoritative: octree.powerDiagramAuthoritative,
      powerDiagramGeneration: projection.powerPublicationGeneration,
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
      hostSchedulingUsesReadback: true,
    });
  }
  private statsReadback() {
    // readbackPending guarantees that this buffer is never copied while mapped.
    return this.statsReadbackBuffer ??= this.device.createBuffer({
      label: "Uniform pooled statistics readback",
      size: 16,
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
    // Proactive CFL control. The latest completed reduction is the previous
    // projected maximum; gravity is the only unbounded explicit acceleration
    // before the next solve, so prevMax + |g| dt is a conservative readback-
    // free bound for choosing this frame's subdivisions.
    const coarseHMin = Math.min(c.width_m / this.info.nx, c.height_m / this.info.ny, c.depth_m / this.info.nz);
    // Sparse phi advection is semi-Lagrangian and does not force the global
    // pressure solve onto the fine geometric timestep. Preserve the coarse
    // Chebyshev cadence unless explicit fine dynamics is enabled.
    const hMin = coarseHMin;
    const inflowSpeed = this.scene.fluid.inflow ? Math.hypot(this.scene.fluid.inflow.velocity_m_s.x, this.scene.fluid.inflow.velocity_m_s.y, this.scene.fluid.inflow.velocity_m_s.z) : 0;
    // Remaining residency exception: the latest asynchronous speed reduction
    // still selects the bounded host substep count. Section 5 requires the
    // full backtrace, so this must not be replaced by trajectory clamping.
    // Migrate it to a fixed maximum encoded schedule with GPU no-op stages.
    const substeps = this.adaptiveProjection ? proactiveQuadtreeSubsteps(
      this.info.maxSpeed_m_s ?? 0,
      inflowSpeed,
      Math.hypot(this.scene.fluid.gravity_m_s2.x, this.scene.fluid.gravity_m_s2.y, this.scene.fluid.gravity_m_s2.z),
      delta,
      hMin,
      64,
      rho,
      sigma
    ) : 1;
    const dt = delta / substeps; this.info.lastDt_s = dt; this.info.lastSubsteps = substeps;
    this.octreeProjection?.setTimestep(dt);
    const activeBodies = bodies.slice(0, 12);
    this.rigidSystem.syncBodies(activeBodies); this.info.encodedSteps = (this.info.encodedSteps ?? 0) + substeps;
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
      && (shouldSegmentPhysics || traceRequestedAt_ms - this.lastPhysicsTraceAt_ms >= 250);
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
    const fineEngineSplits = octreeFineEngineSplitsEnabled();
    encoder.clearBuffer(this.rigidExchangeBuffer);
    // Narita Algorithm 1: regenerate the quadtree at the top of every step.
    // The fully GPU-resident rebuild encodes ahead of advection in the same
    // command stream (queue order = algorithm order) with zero staleness;
    // the asynchronous pipeline below remains the warmup/regrow/rigid path.
    let inlineRebuildEncoded = false;
    if (this.quadtreeProjection && this.rebuildQuadtreeEachStep && this.quadtreeInlineRebuild && !this.quadtreeRebuildPending && !this.quadtreeReadyProjection && this.quadtreeProjection.canEncodeInlineRebuild) {
      inlineRebuildEncoded = this.quadtreeProjection.encodeInlineRebuild(encoder);
    } else if (this.octreeProjection) {
      inlineRebuildEncoded = this.octreeProjection.encodeInlineRebuild(encoder);
    }
    // The collapsed octree structure engine closes after frontier,
    // descriptor, topology, and physical-face publication below. Keep the
    // historical early topology checkpoint only for attribution mode (and for
    // solvers which do not own that engine boundary).
    if (!this.octreeProjection || fineEngineSplits) {
      encoder = completePhysicsPhase(encoder, this.adaptiveProjection
        ? { id: "coarse-grid", label: "Adaptive coarse-grid topology" }
        : { id: "other", label: "Advance setup" });
    }
    for (let substep = 0; substep < substeps; substep += 1) {
      // The first rebuild was encoded above so topology is ready before any
      // dynamics. If CFL control subdivides this advance, phi moves after each
      // projection; rebuild again before the next substep so a newly exposed
      // interface can never remain inside a coarse pressure leaf.
      if (substep > 0 && this.octreeProjection) {
        this.octreeProjection.encodeInlineRebuild(encoder);
        if (fineEngineSplits) {
          encoder = completePhysicsPhase(
            encoder, { id: "coarse-grid", label: "CFL substep topology refresh" },
          );
        }
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
        } else if (this.quadtreeProjection) {
          this.quadtreeProjection.encode(encoder, this.info.nx, this.info.ny, this.info.nz);
          encoder = completePhysicsPhase(encoder, { id: "pressure-solve", label: "Adaptive pressure + projection" });
        }
        // Transport phi from the freshly projected, narrow-band-extrapolated
        // velocity. Sampling the previous frame here was the one-frame lag
        // that froze crests and newly exposed interface cells.
        if (this.octreeProjection) {
          // Segmented boundary callbacks finish the encoder they are handed;
          // continuing on the stale reference would finish it a second time.
          encoder = this.octreeProjection.encodeSurface(encoder, dt, surfaceInflow, this.scene.numerics.maxDt_s,
            physicsTrace || segmentedPhysicsTrace ? (phase, completedEncoder) => {
              return completePhysicsPhase(completedEncoder, OCTREE_SEMANTIC_TRACE_PHASE[phase]);
            } : undefined);
        } else {
          this.adaptiveProjection.encodeSurface(encoder, dt, surfaceInflow, this.scene.numerics.maxDt_s);
        }
        if (!this.octreeProjection) {
          encoder = completePhysicsPhase(
            encoder,
            { id: "fine-sdf-redistance", label: "Surface transport + redistance" },
          );
        } else if (!fineEngineSplits && substep + 1 < substeps) {
          // The final Brick-B boundary is deferred through sparse-world
          // publication below. Intermediate CFL substeps have no such tail,
          // so close their harvest before the next structure epoch begins.
          encoder = completePhysicsPhase(
            encoder, OCTREE_SEMANTIC_TRACE_PHASE.brickEngineB,
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
    }
    // Publish the final substep's resident fields into the shared sparse-brick
    // world. The topology and payload stay GPU-resident; rendering consumes
    // compact debug records and subsequent voxel kernels consume the same ABI.
    if (this.octreeProjection) {
      this.octreeProjection.encodeSparseBrickWorld(encoder, dt);
    }
    if (this.hostAllocation) {
      encoder.clearBuffer(this.reductionBuffer!);
      const pass = encoder.beginComputePass({ label: "Uniform diagnostics reduction" });
      this.dispatch(pass, this.reductionPipeline, this.reductionGroup); pass.end();
    }
    encoder = completePhysicsPhase(
      encoder,
      this.octreeProjection && !fineEngineSplits
        ? OCTREE_SEMANTIC_TRACE_PHASE.brickEngineB
        : { id: "adaptive-publication", label: "Residency + sparse publication + diagnostics" },
    );
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
      for (let rebuild = 0; rebuild < substeps; rebuild += 1) this.octreeProjection.finishInlineRebuild();
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
    this.device.queue.submit([encoder.finish()]);
    const mapPromise = buffer.mapAsync(GPUMapMode.READ);
    const quadtreeDiagnostics = this.adaptiveProjection?.readSolveDiagnostics(); const surfaceDiagnosticsPromise = this.adaptiveProjection?.readSurfaceDiagnostics(); const globalFineDiagnosticsPromise = this.octreeProjection?.readGlobalFineLevelSetDiagnostics();
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
    const initial = Math.max(1, this.info.initialVolumeCellSum ?? 1);
    const conservativeVolumeCells=words[3]/2048;this.info.rawVolumeDrift=this.transportConservativeVolume?(conservativeVolumeCells-initial)/initial:undefined;
    const compactFineExpected=Boolean(this.octreeProjection?.globalFineLevelSetSource);
    const c=this.scene.container,baseCellVolume_m3=c.width_m*c.height_m*c.depth_m/(this.info.nx*this.info.ny*this.info.nz);
    const compactVolume=compactFineExpected&&globalFineDiagnostics?publishedGlobalFineVolumeCells(globalFineDiagnostics,baseCellVolume_m3):undefined;
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
          this.info.pressureEntryCapacity = this.octreeProjection.info.pressureEntryCapacity;
          this.info.pressureRequiredRows = this.octreeProjection.info.pressureRequiredRows;
          this.info.pressureRequiredEntries = this.octreeProjection.info.pressureRequiredEntries;
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

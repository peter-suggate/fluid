import {
  buildSparseAtlasCompositeGrid,
  createSparseAtlasCompositeGridBuildWorkspace,
  type SparseAtlasCompositeGrid,
  type SparseAtlasCompositeCell,
  type SparseAtlasGradientRow,
} from "./sparse-atlas-composite-projection";
import { sparseAtlasScalarsHaveHorizontalD4Symmetry } from
  "./sparse-atlas-surface-conditioning";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickContainingCoordinate,
  sparseBrickSpan,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "./sparse-brick-atlas";
import { CM12_SHARPENING_TRACE_STEPS } from "../../core/cm12-numerics";
import type { SphericalContainerFineGeometry } from "../../core/spherical-container";
import type { GPUFluidFaceVelocitySource } from "../../core/webgpu-face-velocity-overlay";
import type { GPUFluidTracerSource } from "../../core/webgpu-tracer-overlay";
import type { GPUPressureJournalSource } from "../../core/webgpu-pressure-journal-overlay";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import {
  FINE_LEVELSET_COMPACT_LOOKUP_FLAG,
  FINE_LEVELSET_METADATA_WORDS,
  FINE_LEVELSET_WORKSET_HEADER_WORDS,
  type FineLevelSetBrickPlan,
} from "../../core/fine-levelset-brick-abi";
import type {
  SparseAdaptiveGridConsumerSource,
  WebGPUFineLevelSetBrickSource,
} from "../../core/levelset-consumer-abi";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import {
  SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS,
  SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS,
  decodeSparseCM12PressureJournal,
  sparseCM12PressureJournalLayout,
  sparseCM12PressureJournalSchedule,
  type SparseCM12PressureJournal,
  type SparseCM12PressureJournalLayout,
} from "./sparse-cm12-pressure-journal";
import {
  createWebgpuSparseCM12ResidentWGSL,
  type SparseCM12PressureRepairLayout,
  type SparseCM12TemporalWorklistLayout,
} from "./webgpu-sparse-cm12-resident.wgsl";
import {
  createSparseCM12IncrementalActivityInitialWords,
  createSparseCM12IncrementalActivityLayout,
  type SparseCM12IncrementalActivityLayout,
} from "./sparse-cm12-incremental-activity";
import {
  SPARSE_CM12_PRESSURE_MEMBERSHIP_INDIRECT_BYTES,
  SPARSE_CM12_PRESSURE_REPAIR_HEADER,
  SPARSE_CM12_PRESSURE_REPAIR_HEADER_WORDS,
} from "./sparse-cm12-pressure-membership";
import {
  SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER,
  createSparseCM12CanonicalMembershipLayout,
  initializeSparseCM12CanonicalMembershipWords,
  sparseCM12CanonicalMembershipRepairIndirectByteOffset,
  type SparseCM12CanonicalMembershipLayout,
} from "./sparse-cm12-canonical-membership";
import {
  WebGPUSparseCM12RigidCoupling,
  type SparseCM12RigidResources,
} from "./webgpu-sparse-cm12-rigid-coupling";
import {
  createSparseCM12FramePlanInitialWords,
  createSparseCM12FramePlanLayout,
  sparseCM12FramePlanSource,
  type SparseCM12FramePlanLayout,
} from "../../core/sparse-cm12-frame-plan";
import {
  createSparseCM12FramePlanPresentationInitialWords,
  createSparseCM12FramePlanPresentationLayout,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS,
  type SparseCM12FramePlanPresentationLayout,
} from "./sparse-cm12-frame-plan-presentation";
import {
  SPARSE_CM12_FRAME_CONTROL_COVERAGE,
  SPARSE_CM12_FRAME_CONTROL_FAMILY,
  SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT,
  SPARSE_CM12_FRAME_CONTROL_HEADER,
  SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS,
  SPARSE_CM12_FRAME_CONTROL_INVALID,
  SPARSE_CM12_FRAME_CONTROL_MAGIC,
  SPARSE_CM12_FRAME_CONTROL_PHASE,
  createSparseCM12FrameControl,
  type SparseCM12FrameControlLayout,
} from "./sparse-cm12-frame-control";
import {
  SPARSE_CM12_SRR1_INGRESS_HEADER,
  SPARSE_CM12_SRR1_INGRESS_HEADER_WORDS,
  SPARSE_CM12_SRR1_INDIRECT_FAMILY,
  WebGPUSparseCM12SRR1RuntimeAdapter,
  createSparseCM12SRR1IngressLayout,
  createSparseCM12SRR1RuntimePlan,
  initializeSparseCM12SRR1IngressWords,
  type SparseCM12SRR1IngressLayout,
} from "./sparse-cm12-srr1-runtime-adapter";
import {
  SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT,
  createSparseCM12VexActivityBatchIndirectCopies,
  createSparseCM12VexActivityBatchInitialWords,
  createSparseCM12VexActivityBatchLayout,
  createSparseCM12VexActivityBatchPipelineDescriptors,
  type SparseCM12VexActivityBatchLayout,
  type SparseCM12VexActivityBatchPacket,
} from "./sparse-cm12-vex-activity-batch";
import {
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS,
  SPARSE_CM12_VELOCITY_EXTENSION_MAGIC,
  SPARSE_CM12_VELOCITY_EXTENSION_VERSION,
} from "./sparse-cm12-velocity-extension";
import {
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS,
  createSparseCM12PressureTopologyRepairInitialWords,
  createSparseCM12PressureTopologyRepairLayout,
  sparseCM12PressureTopologyRepairEntryPoints,
  sparseCM12PressureTopologyRepairHeaderIndirectByteOffset,
  sparseCM12PressureTopologyRepairIndirectByteOffset,
  type SparseCM12PressureTopologyRepairLayout,
} from "./sparse-cm12-pressure-topology-repair";
import {
  SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER,
  SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER,
  SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_CACHE_HEADER,
  SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS,
  createSparseCM12ResidentPersistentPressureCacheLayout,
  initializeSparseCM12PersistentPressureCacheWords,
  sparseCM12PersistentPressureCacheAggregateIndirectByteOffset,
  type SparseCM12PersistentPressureCacheLayout,
} from "./sparse-cm12-persistent-pressure-cache";
import {
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER,
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE,
  createSparseCM12PressureAddressingABPipelineDescriptors,
  createSparseCM12PressureAddressingABInitialWords,
  createSparseCM12PressureAddressingABLayout,
  createSparseCM12ProductionPressureAddressingLayout,
  inspectSparseCM12PressureAddressingABReceipt,
  sparseCM12PressureAddressingABPipelineConstants,
  type SparseCM12PressureAddressingABLayout,
  type SparseCM12PressureAddressingABModeName,
  type SparseCM12PressureAddressingABReceipt,
} from "./sparse-cm12-pressure-addressing-ab";
import {
  SPARSE_CM12_FACE_PROJECTION_HEADER,
  SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS,
  SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER,
  SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS,
  createSparseCM12FaceProjectionAuthorityInitialWords,
  createSparseCM12FaceProjectionAuthorityLayout,
  sparseCM12FaceProjectionBootstrapIndirectByteOffset,
  sparseCM12FaceProjectionIndirectByteOffset,
  type SparseCM12FaceProjectionAuthorityLayout,
} from "./sparse-cm12-face-projection-authority";
import {
  SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS,
  createSparseCM12FacePreparationTileCensusInitialWords,
  createSparseCM12FacePreparationTileCensusLayout,
  inspectSparseCM12FacePreparationTileCensusQA,
  type SparseCM12FacePreparationTileCensusLayout,
  type SparseCM12FacePreparationTileCensusQA,
} from "./sparse-cm12-face-preparation-tile-census";
import {
  createSparseCM12FpaVexReadCensusInitialWords,
  createSparseCM12FpaVexReadCensusLayout,
  inspectSparseCM12FpaVexReadCensusSummaryQA,
  type SparseCM12FpaVexReadCensusLayout,
  type SparseCM12FpaVexReadCensusSummaryQA,
} from "./sparse-cm12-fpa-vex-read-census";

/** CM12 Sec. 3.5 Algorithm 2's live trace bounds, in finest cells and substeps. */
export interface SharpeningTrace {
  readonly distanceCells?: number;
  readonly traceSteps?: number;
}

/** Live GPU-authored resolution policy. Accepted topology publication is a
 * separate transaction, so these controls tune candidate requests/history. */
export interface SparseCM12ActivityPolicy {
  readonly activitySignals: boolean;
  readonly finestTravelCells: number;
  readonly fourTravelCells: number;
  readonly twoTravelCells: number;
  readonly thinFeatureCells: number;
  readonly thinFeatureDensity: number;
  readonly residencyDensity: number;
  readonly residencyMassFineCells: number;
  readonly surfaceDensityMinimum: number;
  readonly surfaceDensityMaximum: number;
  readonly detailTolerance: number;
  readonly frontLookaheadSteps: number;
  readonly topologyCadenceSteps: number;
  readonly prepareBricksPerFrame: number;
  readonly promoteEpochs: number;
  readonly demoteEpochs: number;
  readonly promoteScore: number;
  readonly demoteScore: number;
  readonly emergencyScore: number;
}

export const SPARSE_CM12_ACTIVITY_POLICY = Object.freeze({
  activitySignals: true,
  finestTravelCells: 1,
  fourTravelCells: 0.5,
  twoTravelCells: 0.25,
  thinFeatureCells: 2,
  thinFeatureDensity: 0,
  residencyDensity: 0.005,
  residencyMassFineCells: 1,
  surfaceDensityMinimum: 0.05,
  surfaceDensityMaximum: 0.95,
  detailTolerance: 0.08,
  frontLookaheadSteps: 4,
  topologyCadenceSteps: 1,
  prepareBricksPerFrame: 64,
  promoteEpochs: 2,
  demoteEpochs: 1,
  promoteScore: 160 / 255,
  demoteScore: 96 / 255,
  emergencyScore: 224 / 255,
} satisfies SparseCM12ActivityPolicy);

const finiteClamp = (value: unknown, fallback: number, minimum: number, maximum: number) =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(maximum, Math.max(minimum, value)) : fallback;

const integerClamp = (value: unknown, fallback: number, minimum: number, maximum: number) =>
  Math.round(finiteClamp(value, fallback, minimum, maximum));

export function sparseCM12ActivityPolicy(
  values: Partial<Record<keyof SparseCM12ActivityPolicy, unknown>>,
): SparseCM12ActivityPolicy {
  const defaults = SPARSE_CM12_ACTIVITY_POLICY;
  const finestTravelCells = finiteClamp(
    values.finestTravelCells, defaults.finestTravelCells, 0.05, 8,
  );
  const fourTravelCells = Math.min(finestTravelCells, finiteClamp(
    values.fourTravelCells, defaults.fourTravelCells, 0, 8,
  ));
  const twoTravelCells = Math.min(fourTravelCells, finiteClamp(
    values.twoTravelCells, defaults.twoTravelCells, 0, 8,
  ));
  const promoteScore = finiteClamp(values.promoteScore, defaults.promoteScore, 0, 1);
  return {
    activitySignals: values.activitySignals === true,
    finestTravelCells,
    fourTravelCells,
    twoTravelCells,
    thinFeatureCells: finiteClamp(
      values.thinFeatureCells, defaults.thinFeatureCells, 0.25, 8,
    ),
    thinFeatureDensity: finiteClamp(
      values.thinFeatureDensity, defaults.thinFeatureDensity, 0, 0.5,
    ),
    residencyDensity: finiteClamp(
      values.residencyDensity, defaults.residencyDensity, 0.000_01, 0.5,
    ),
    residencyMassFineCells: finiteClamp(
      values.residencyMassFineCells, defaults.residencyMassFineCells, 0, 8,
    ),
    surfaceDensityMinimum: finiteClamp(
      values.surfaceDensityMinimum, defaults.surfaceDensityMinimum, 0, 0.49,
    ),
    surfaceDensityMaximum: finiteClamp(
      values.surfaceDensityMaximum, defaults.surfaceDensityMaximum, 0.51, 1,
    ),
    detailTolerance: finiteClamp(
      values.detailTolerance, defaults.detailTolerance, 0.005, 0.5,
    ),
    frontLookaheadSteps: integerClamp(
      values.frontLookaheadSteps, defaults.frontLookaheadSteps, 1, 32,
    ),
    topologyCadenceSteps: integerClamp(
      values.topologyCadenceSteps, defaults.topologyCadenceSteps, 1, 32,
    ),
    prepareBricksPerFrame: integerClamp(
      values.prepareBricksPerFrame, defaults.prepareBricksPerFrame, 1, 256,
    ),
    promoteEpochs: integerClamp(values.promoteEpochs, defaults.promoteEpochs, 1, 16),
    demoteEpochs: integerClamp(values.demoteEpochs, defaults.demoteEpochs, 1, 32),
    promoteScore,
    demoteScore: Math.min(promoteScore, finiteClamp(
      values.demoteScore, defaults.demoteScore, 0, 1,
    )),
    emergencyScore: Math.max(promoteScore, finiteClamp(
      values.emergencyScore, defaults.emergencyScore, 0, 1,
    )),
  };
}

/**
 * The sparse lane's D, at the top of the paper's 1.1-to-3.1 range rather than
 * at Uniform CM12's Fig. 5 value of 2.1. A shorter trace leaves the abandoned
 * splash's removed mass on the tall side walls, where the residue regression
 * finds it; matched A/B lanes against the uniform reference therefore have to
 * move one slider or the other.
 */
export const SPARSE_CM12_SHARPENING_DISTANCE_CELLS = 3.1;
export const SPARSE_CM12_SHARPENING_TRACE_STEPS = CM12_SHARPENING_TRACE_STEPS;

/** Kept inside the paper's own D range; the panel spec declares the same bounds. */
export const sparseCM12SharpeningDistance = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(3.1, Math.max(0.1, value))
    : SPARSE_CM12_SHARPENING_DISTANCE_CELLS;

export const sparseCM12SharpeningTraceSteps = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(16, Math.max(1, Math.round(value)))
    : SPARSE_CM12_SHARPENING_TRACE_STEPS;

const INVALID = 0xffff_ffff;
/**
 * The advance's stage partition, in encode order.
 *
 * Every dispatch `encode` issues belongs to exactly one of these, so a caller
 * that closes each stage in turn holds an exhaustive partition of the frame's
 * GPU work rather than a sampling of it. These ids are the ABI the method's
 * trace phases and its advance-pipeline diagram are keyed by; adding a
 * dispatch means placing it in a stage here.
 */
export type SparseCM12ResidentStageId =
  | "transport-velocity-extension"
  | "face-preparation"
  | "conservative-transport"
  | "tracer-advection"
  | "gamma-diffusion"
  | "surface-sharpening"
  | "symmetry-authority"
  | "body-forces"
  | "pressure-topology"
  | "pressure-rhs"
  | "pressure-solve"
  | "velocity-projection"
  | "projection-diagnostics"
  | "activity-measurement"
  | "resolution-planning"
  | "candidate-transfer"
  | "brick-retirement"
  | "presentation-publication";

/** Encode order. The seam chain closes these left to right, exactly once each. */
export const SPARSE_CM12_RESIDENT_STAGES: readonly SparseCM12ResidentStageId[] =
  Object.freeze([
    "transport-velocity-extension",
    "face-preparation",
    "conservative-transport",
    "tracer-advection",
    "gamma-diffusion",
    "surface-sharpening",
    "symmetry-authority",
    "body-forces",
    "pressure-topology",
    "pressure-rhs",
    "pressure-solve",
    "velocity-projection",
    "projection-diagnostics",
    "activity-measurement",
    "resolution-planning",
    "candidate-transfer",
    "brick-retirement",
    "presentation-publication",
  ] as const);

/** Where an observer is told about each stage of the advance. */
export interface SparseCM12ResidentStageSeams {
  /**
   * Close the named stage, immediately after its last dispatch. A stage that
   * encodes nothing this advance still reports: it closes on its successor's
   * boundary and costs exactly zero.
   */
  readonly close: (stage: SparseCM12ResidentStageId) => void;
  /** Bind the final timestamp to a word published at the true command tail. */
  readonly anchorFinalBoundary?: (source: GPUBuffer, offset?: number) => void;
}

const WORKGROUP_SIZE = 64;
/** Params in the resident WGSL, in bytes. Grow this when a field is added. */
const SPARSE_CM12_PARAMETER_BYTES = 448;
/** Twenty f32 convergence/diagnostic scalars; see the WGSL initialization. */
const SPARSE_CM12_PRESSURE_SCALAR_BYTES = 80;
const SPARSE_CM12_PCM_DIAGNOSTIC_DOMAIN_WORDS = 12;
const SPARSE_CM12_PCM_DIAGNOSTIC_BYTES =
  2 * 4 * SPARSE_CM12_PCM_DIAGNOSTIC_DOMAIN_WORDS;
const SPARSE_CM12_PRESSURE_CUTOVER_DIAGNOSTIC_WORDS =
  2 * SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS
  + SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS
  + SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS
  + 4 * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS;

interface SparseCM12PressureAddressingABQAResources {
  readonly mode: SparseCM12PressureAddressingABModeName;
  readonly layout: SparseCM12PressureAddressingABLayout;
  readonly indirectArguments: GPUBuffer;
  readonly receiptReadback?: GPUBuffer;
  readonly querySet?: GPUQuerySet;
  readonly queryResolve?: GPUBuffer;
  readonly timestampReadback?: GPUBuffer;
}
export const SPARSE_CM12_PRESSURE_ITERATIONS = 64;
/** Immediate production cutover: guarded by a fresh b-Ap check every batch. */
export const SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE = 5e-7;
export const SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE = 32;
const SPARSE_CM12_PRESSURE_ITERATIONS_MINIMUM = 8;
const SPARSE_CM12_PRESSURE_ITERATIONS_MAXIMUM = 256;
const SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE_MAXIMUM = 0.1;

export interface SparseCM12PressureControl {
  readonly iterations?: number;
  readonly relativeTolerance?: number;
}

/**
 * Capacity of the optional pressure journal, chosen at construction.
 *
 * Sized here rather than armed at runtime because the journal is a tail range
 * of the resident state buffer, which exists once. Defaulting it on would be
 * the "capacity is not inert" mistake: on a large scene the snapshot region is
 * tens of megabytes, and a lane that never opens the pressure lab would pay it
 * at t=0. So the default is zero floats and zero dispatches, and a caller that
 * wants the film asks for it.
 */
export interface SparseCM12PressureJournalCapacityRequest {
  /**
   * Encoded iterations to reserve records for. Pass the solve's iteration
   * ceiling; the reservation adds one for the seed record.
   */
  readonly iterationCapacity?: number;
  /** Whole-field snapshots to reserve. Twelve covers a 128-iteration solve. */
  readonly snapshotCapacity?: number;
}

/** Snapshots reserved when a caller asks for a journal without saying how many. */
export const SPARSE_CM12_PRESSURE_JOURNAL_SNAPSHOTS = 12;

export const sparseCM12PressureIterations = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(SPARSE_CM12_PRESSURE_ITERATIONS_MAXIMUM,
      Math.max(SPARSE_CM12_PRESSURE_ITERATIONS_MINIMUM, Math.round(value)))
    : SPARSE_CM12_PRESSURE_ITERATIONS;

export const sparseCM12PressureRelativeTolerance = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE_MAXIMUM,
      Math.max(0, value))
    : SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE;
const ACTIVITY_HEADER_WORDS = 28;
const ACTIVITY_RECORD_WORDS = 39;
const ACCEPTED_COARSE_ROW_COUNT_WORD = 22;
const ACCEPTED_MIXED_ROW_COUNT_WORD = 23;
const PRESSURE_ACTIVE_ROW_COUNT_WORD = 24;
const CANDIDATE_CELL_CHANNELS = 6;
const CANDIDATE_FACE_CHANNELS = 6;
const candidateFloatsPerBrick = (brickFineResolution: number) =>
  CANDIDATE_CELL_CHANNELS * brickFineResolution ** 3
  + CANDIDATE_FACE_CHANNELS * brickFineResolution ** 2;
const GPU_TOPOLOGY_PAGE_POOL_MINIMUM = 32;
const GPU_TOPOLOGY_PAGE_POOL_MAXIMUM = 512;
const GPU_TOPOLOGY_CELL_PAGE_HEADER_WORDS = 4;
const GPU_TOPOLOGY_CELL_RECORD_WORDS = 8;
const gpuTopologyCellPageWords = (brickFineResolution: number) =>
  GPU_TOPOLOGY_CELL_PAGE_HEADER_WORDS
  + brickFineResolution ** 3 * GPU_TOPOLOGY_CELL_RECORD_WORDS;

export interface SparseCM12TopologyPagePoolPlan {
  readonly pageCapacity: number;
  readonly freeListWords: number;
  readonly pageWords: number;
  readonly descriptorWords: number;
}

export interface SparseCM12OwnershipTablePlan {
  readonly hashCapacity: number;
  readonly hashWords: number;
  readonly brickRecordWords: number;
  readonly allocatedWords: number;
}

/** O(resident leaves), independent of the logical dimensions macro leaves cover. */
export function sparseCM12OwnershipTablePlan(
  residentBrickCount: number,
): SparseCM12OwnershipTablePlan {
  if (!Number.isSafeInteger(residentBrickCount) || residentBrickCount < 0
    || residentBrickCount >= 2 ** 28) {
    throw new RangeError(`Sparse CM12 resident brick count ${residentBrickCount} is invalid`);
  }
  const hashCapacity = 2 ** Math.ceil(Math.log2(Math.max(2, 2 * residentBrickCount)));
  const hashWords = hashCapacity;
  const brickRecordWords = 2 * residentBrickCount;
  return { hashCapacity, hashWords, brickRecordWords,
    allocatedWords: hashWords + brickRecordWords + 2 };
}

// The 192x96x32 long-dam scene has a 24x12x4 span-one receiver domain. Keeping
// the bound below its 1152 leaves routed its moving front through the cell-only
// dynamic page prototype, whose publication is deliberately deferred until it
// also owns rows and incidence. The result was worse than merely delayed adaptation:
// newly activated swept receivers remained at their construction-time coarse
// rung. A 2048-leaf compatibility frontier remains bounded while giving that
// benchmark the complete cell/row/incidence templates needed for urgent 8^3
// surface publication. Leave headroom for the same bounded frontier with a
// modest apron; the accepted cell/row limits above remain independent guards.
export const SPARSE_CM12_HOST_TEMPLATE_MUTABLE_BRICK_MAXIMUM = 2048;

/**
 * Host variant packing is a compatibility path for bounded live frontiers.
 * Its true cost is four cell rungs plus mixed-seam variants per mutable brick,
 * not the accepted cell count alone. Keeping the mutable bound explicit stops
 * a large low-resolution scene from passing the old cell threshold and then
 * constructing millions of persistent JavaScript template objects.
 */
export function sparseCM12HostTemplateVariantsEnabled(
  acceptedCellCount: number,
  acceptedRowCount: number,
  mutableBrickCount: number,
  brickFineResolution = 16,
): boolean {
  const mutableBrickMaximum = SPARSE_CM12_HOST_TEMPLATE_MUTABLE_BRICK_MAXIMUM
    * (8 / brickFineResolution) ** 3;
  return acceptedCellCount <= 250_000
    && acceptedRowCount <= 750_000
    && mutableBrickCount <= mutableBrickMaximum;
}

/** Bounded by the mutable frontier, never by logical-domain brick count. */
export function sparseCM12TopologyPagePoolPlan(
  mutableFrontierBricks: number,
  enabled = true,
  brickFineResolution = 16,
): SparseCM12TopologyPagePoolPlan {
  const pageWords = gpuTopologyCellPageWords(brickFineResolution);
  if (!enabled || mutableFrontierBricks <= 0) return {
    pageCapacity: 0, freeListWords: 0,
    pageWords, descriptorWords: 0,
  };
  const pageCapacity = Math.min(GPU_TOPOLOGY_PAGE_POOL_MAXIMUM,
    Math.max(GPU_TOPOLOGY_PAGE_POOL_MINIMUM, Math.ceil(mutableFrontierBricks / 2)));
  return {
    pageCapacity, freeListWords: pageCapacity,
    pageWords,
    descriptorWords: pageCapacity * pageWords,
  };
}

interface PackedResidentTopology {
  readonly words: Uint32Array;
  readonly cellOffset: number;
  readonly rowOffset: number;
  readonly termOffset: number;
  readonly incidenceOffset: number;
  readonly incidenceRecordOffset: number;
  readonly brickLookupOffset: number;
  readonly brickOffset: number;
  readonly backgroundOwnerOffset: number;
  readonly brickCount: number;
  readonly candidateBrickCount: number;
  readonly incidenceCount: number;
}

const sparseCM12TemplateLevels = (brickFineResolution: number): SparseBrickResolution[] =>
  Array.from({ length: Math.log2(brickFineResolution) + 1 }, (_, level) =>
    2 ** level as SparseBrickResolution);
const TEMPLATE_MAGIC = 0x5343_4d54; // "SCMT"
const TEMPLATE_HEADER_WORDS = 16;
// The resident backend rejects separating spherical boundaries before packing,
// so openFraction=1, openVolume=volume and separatingMinimum=false are
// invariants rather than per-cell data. Keep exact geometry plus one packed
// [brick:27 | resolution:5] word: a 32-byte record, half the former footprint.
const TEMPLATE_CELL_RECORD_WORDS = 8;
const TEMPLATE_CELL_RESOLUTION_BITS = 5;
const TEMPLATE_CELL_RESOLUTION_MASK = 0x1f;

function packedTemplateCellMetadata(brick: number, resolution: number): number {
  if (brick < 0 || brick >= 2 ** (32 - TEMPLATE_CELL_RESOLUTION_BITS)
    || ![1, 2, 4, 8, 16].includes(resolution)) {
    throw new Error(`Sparse CM12 cell metadata cannot be packed: ${brick}/${resolution}`);
  }
  return ((brick << TEMPLATE_CELL_RESOLUTION_BITS) | resolution) >>> 0;
}

const templateCellBrick = (words: Uint32Array, base: number) =>
  words[base + 7]! >>> TEMPLATE_CELL_RESOLUTION_BITS;
const templateCellResolution = (words: Uint32Array, base: number) =>
  words[base + 7]! & TEMPLATE_CELL_RESOLUTION_MASK;
// Immutable rows are read field-wise by wide shader invocations. Nine SoA
// planes keep those reads contiguous while two packed planes preserve every
// integer field: [term offset:28 | count:4] and
// [requirement offset:28 | kind:2 | axis:2]. Geometry remains exact f32 bits.
const TEMPLATE_ROW_PLANE_COUNT = 9;
const TEMPLATE_ROW_OFFSET_MASK = 0x0fff_ffff;

const templateRowWord = (
  rowOffset: number, rowCount: number, plane: number, row: number,
) => rowOffset + plane * rowCount + row;

function packedTemplateRowTerms(offset: number, count: number): number {
  if (offset > TEMPLATE_ROW_OFFSET_MASK || count > 0xf) {
    throw new Error(`Sparse CM12 row term range cannot be packed: ${offset}+${count}`);
  }
  return (offset | (count << 28)) >>> 0;
}

function packedTemplateRowMetadata(
  requirementOffset: number, kind: SparseAtlasGradientRow["kind"], axis: number,
): number {
  if (requirementOffset > TEMPLATE_ROW_OFFSET_MASK || axis > 3) {
    throw new Error(`Sparse CM12 row metadata cannot be packed: ${requirementOffset},${axis}`);
  }
  const kindCode = kind === "intra-brick" ? 0 : kind === "brick-face" ? 1
    : kind === "mixed-seam" ? 2 : 3;
  return (requirementOffset | (kindCode << 28) | (axis << 30)) >>> 0;
}

interface PackedResidentTopologyTemplates {
  readonly words: Uint32Array;
  readonly cellCount: number;
  readonly rowCount: number;
  readonly initialCellWorklist: Uint32Array;
  readonly initialRowWorklist: Uint32Array;
  readonly initialDensity: Float32Array;
  readonly initialGamma: Float32Array;
}

/** Serialize one accepted generation without duplicating its object graph. */
function packAcceptedTopologyTemplates(
  atlas: SparseAdaptiveMassAtlas,
  grid: SparseAtlasCompositeGrid,
): PackedResidentTopologyTemplates {
  const templateLevels = sparseCM12TemplateLevels(atlas.brickFineResolution);
  const cells = grid.cells, rows = grid.gradientRows;
  const brickIndex = new Map(atlas.bricks.map((brick, index) => [brick.key, index]));
  const cellCountByBrick = new Uint32Array(atlas.bricks.length);
  let termCount = 0, requirementWords = 0;
  const incidenceCounts = new Uint32Array(cells.length);
  for (const row of rows) {
    termCount += row.terms.length;
    const seen: number[] = [];
    for (const term of row.terms) {
      incidenceCounts[term.cellId] += 1;
      const owner = brickIndex.get(cells[term.cellId]!.brickKey)!;
      if (!seen.includes(owner)) seen.push(owner);
    }
    requirementWords += 1 + seen.length;
  }
  for (const cell of cells) cellCountByBrick[brickIndex.get(cell.brickKey)!] += 1;
  const incidenceOffsets = new Uint32Array(cells.length + 1);
  for (let cell = 0; cell < cells.length; cell += 1) {
    incidenceOffsets[cell + 1] = incidenceOffsets[cell]! + incidenceCounts[cell]!;
  }
  const incidenceCount = incidenceOffsets[cells.length]!;
  const pressureEdgeCounts = new Uint32Array(cells.length);
  for (const row of rows) for (const own of row.terms) {
    pressureEdgeCounts[own.cellId] += row.terms.length - 1;
  }
  const pressureEdgeOffsets = new Uint32Array(cells.length + 1);
  for (let cell = 0; cell < cells.length; cell += 1) {
    pressureEdgeOffsets[cell + 1] = pressureEdgeOffsets[cell]! + pressureEdgeCounts[cell]!;
  }
  const pressureEdgeCount = pressureEdgeOffsets[cells.length]!;
  let at = TEMPLATE_HEADER_WORDS;
  const cellOffset = at; at += TEMPLATE_CELL_RECORD_WORDS * cells.length;
  const rowOffset = at; at += TEMPLATE_ROW_PLANE_COUNT * rows.length;
  const termOffset = at; at += 2 * termCount;
  const incidenceOffset = at; at += cells.length + 1;
  const incidenceRecordOffset = at; at += 2 * incidenceCount;
  const cellRangeOffset = at; at += 2 * templateLevels.length * atlas.bricks.length;
  const rowRequirementOffset = at; at += requirementWords;
  const pressureEdgeOffset = at; at += cells.length + 1;
  const pressureEdgeRecordOffset = at; at += 3 * pressureEdgeCount;
  const words = new Uint32Array(at);
  words.set([TEMPLATE_MAGIC, 1, cells.length, rows.length, termCount, incidenceCount,
    cellOffset, rowOffset, termOffset, incidenceOffset, incidenceRecordOffset,
    cellRangeOffset, rowRequirementOffset, atlas.bricks.length], 0);
  words[15] = pressureEdgeOffset;
  for (const cell of cells) {
    const base = cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell.id;
    if (cell.openFraction !== 1 || cell.openVolume !== cell.volume
      || cell.separatingPressureMinimum) {
      throw new Error(`Sparse CM12 resident cell ${cell.id} has unsupported static boundary data`);
    }
    setF32(words, base, cell.centerFine[0]); setF32(words, base + 1, cell.centerFine[1]);
    setF32(words, base + 2, cell.centerFine[2]); setF32(words, base + 3, cell.volume);
    setF32(words, base + 4, cell.widthsFine[0]); setF32(words, base + 5, cell.widthsFine[1]);
    setF32(words, base + 6, cell.widthsFine[2]);
    words[base + 7] = packedTemplateCellMetadata(
      brickIndex.get(cell.brickKey)!, cell.brickResolution,
    );
  }
  for (const brick of atlas.bricks) {
    const index = brickIndex.get(brick.key)!;
    const first = grid.cellBaseByBrick.get(brick.key) ?? 0;
    for (let level = 0; level < templateLevels.length; level += 1) {
      words[cellRangeOffset + 2 * (templateLevels.length * index + level)] = first;
      words[cellRangeOffset + 2 * (templateLevels.length * index + level) + 1]
        = cellCountByBrick[index]!;
    }
  }
  words.set(incidenceOffsets, incidenceOffset);
  words.set(pressureEdgeOffsets, pressureEdgeOffset);
  const incidenceCursor = incidenceOffsets.slice(0, cells.length);
  const pressureEdgeCursor = pressureEdgeOffsets.slice(0, cells.length);
  let nextTerm = 0, requirementAt = rowRequirementOffset;
  for (const row of rows) {
    words[templateRowWord(rowOffset, rows.length, 0, row.id)]
      = packedTemplateRowTerms(nextTerm, row.terms.length);
    words[templateRowWord(rowOffset, rows.length, 1, row.id)]
      = packedTemplateRowMetadata(requirementAt, row.kind, row.axis);
    setF32(words, templateRowWord(rowOffset, rows.length, 2, row.id), row.dualWeight);
    setF32(words, templateRowWord(rowOffset, rows.length, 3, row.id), row.area);
    setF32(words, templateRowWord(rowOffset, rows.length, 4, row.id), row.distance);
    setF32(words, templateRowWord(rowOffset, rows.length, 5, row.id), row.exteriorPhi ?? 0.5);
    setF32(words, templateRowWord(rowOffset, rows.length, 6, row.id), row.centerFine[0]);
    setF32(words, templateRowWord(rowOffset, rows.length, 7, row.id), row.centerFine[1]);
    setF32(words, templateRowWord(rowOffset, rows.length, 8, row.id), row.centerFine[2]);
    const owners: { brick: number; resolution: number }[] = [];
    for (const term of row.terms) {
      const cell = cells[term.cellId]!;
      const owner = brickIndex.get(cell.brickKey)!;
      if (!owners.some((entry) => entry.brick === owner)) {
        owners.push({ brick: owner, resolution: cell.brickResolution });
      }
      words[termOffset + 2 * nextTerm] = term.cellId;
      setF32(words, termOffset + 2 * nextTerm + 1, term.coefficient);
      const incidence = incidenceCursor[term.cellId]++;
      words[incidenceRecordOffset + 2 * incidence] = row.id;
      words[incidenceRecordOffset + 2 * incidence + 1] = nextTerm;
      nextTerm += 1;
    }
    for (const own of row.terms) for (const other of row.terms) {
      if (other.cellId === own.cellId) continue;
      const edge = pressureEdgeCursor[own.cellId]++;
      const record = pressureEdgeRecordOffset + 3 * edge;
      words[record] = row.id;
      words[record + 1] = other.cellId;
      setF32(words, record + 2,
        own.coefficient * row.dualWeight * other.coefficient);
    }
    words[requirementAt++] = owners.length;
    for (const owner of owners) {
      words[requirementAt++] = packedTemplateCellMetadata(owner.brick, owner.resolution);
    }
  }
  return {
    words, cellCount: cells.length, rowCount: rows.length,
    initialCellWorklist: Uint32Array.from({ length: cells.length }, (_, id) => id),
    initialRowWorklist: Uint32Array.from({ length: rows.length }, (_, id) => id),
    initialDensity: Float32Array.from(cells, (cell) => cell.density),
    initialGamma: Float32Array.from(cells, (cell) => cell.gamma),
  };
}

const templateCellKey = (brickKey: number, resolution: number, local: number) =>
  `${brickKey}/${resolution}/${local}`;

// The composite-grid workspace recycles cell/row objects and their tuple
// storage on every build. Template variants outlive those builds, so retain a
// value snapshot rather than shallow-copying references that the next variant
// will rewrite.
function snapshotTemplateCell(source: SparseAtlasCompositeCell,
  id: number): SparseAtlasCompositeCell {
  return { ...source, id,
    brickCoordinate: [...source.brickCoordinate],
    local: [...source.local],
    minimumFine: [...source.minimumFine],
    maximumFine: [...source.maximumFine],
    centerFine: [...source.centerFine],
    widthsFine: [...source.widthsFine],
  };
}

function resampleBrick(brick: SparseAdaptiveMassBrick,
  resolution: SparseBrickResolution): SparseAdaptiveMassBrick {
  const sample = (values: Float64Array, local: number): number => {
    const z = Math.floor(local / (resolution * resolution));
    const yz = local - z * resolution * resolution;
    const y = Math.floor(yz / resolution), x = yz - y * resolution;
    if (resolution >= brick.resolution) {
      const factor = resolution / brick.resolution;
      const sx = Math.floor(x / factor), sy = Math.floor(y / factor), sz = Math.floor(z / factor);
      return values[sx + brick.resolution * (sy + brick.resolution * sz)]!;
    }
    const factor = brick.resolution / resolution;
    let sum = 0;
    for (let dz = 0; dz < factor; dz += 1) for (let dy = 0; dy < factor; dy += 1) {
      for (let dx = 0; dx < factor; dx += 1) {
        const sx = factor * x + dx, sy = factor * y + dy, sz = factor * z + dz;
        sum += values[sx + brick.resolution * (sy + brick.resolution * sz)]!;
      }
    }
    return sum / (factor ** 3);
  };
  const count = resolution ** 3;
  return { ...brick, resolution,
    density: Float64Array.from({ length: count }, (_, local) => sample(brick.density, local)),
    gamma: Float64Array.from({ length: count }, (_, local) => sample(brick.gamma, local)) };
}

/** Number of compact cells emitted for a possibly clipped domain-edge brick. */
function sparseCM12BrickLiveCellCount(atlas: SparseAdaptiveMassAtlas,
  brick: SparseAdaptiveMassBrick, resolution: SparseBrickResolution): number {
  const width = atlas.brickFineResolution * sparseBrickSpan(brick);
  const scale = width / resolution;
  return ([0, 1, 2] as const).reduce<number>((count, axis) => {
    const origin = atlas.brickFineResolution * brick.coordinate[axis];
    const extent = Math.max(0, Math.min(width, atlas.dimensions[axis] - origin));
    return count * Math.min(resolution, Math.ceil(extent / scale));
  }, 1);
}

/**
 * Construction-time physical template library. Four uniform builds provide
 * every cell, intra-brick row, same-level face, and sparse-air face. Eighteen
 * alternating builds provide both orientations of every valid 2:1 face pair.
 * Runtime publication can therefore switch cells and pressure rows by only
 * rebuilding compact worklists; no host topology build is needed after create.
 */
function packResidentTopologyTemplates(atlas: SparseAdaptiveMassAtlas,
  acceptedGrid: SparseAtlasCompositeGrid):
PackedResidentTopologyTemplates {
  const templateLevels = sparseCM12TemplateLevels(atlas.brickFineResolution);
  const mutableBrickKeys = new Set(atlas.bricks.filter((brick) =>
    sparseBrickSpan(brick) === 1).map((brick) => brick.key));
  // Even a bounded compatibility frontier is large enough that building one
  // full object-graph variant at 8^3 creates a costly transient heap. Partition
  // the mutable set and include a one-face-neighbour halo around each partition.
  // Rows touching a core brick then see exactly the same physical neighbours
  // as a whole-atlas build; halo-only boundary rows are discarded.
  const TEMPLATE_BUILD_CHUNK_BRICKS = Math.max(8,
    128 * (8 / atlas.brickFineResolution) ** 3);
  const mutableBricks = atlas.bricks.filter((brick) => mutableBrickKeys.has(brick.key));
  const mutableChunks = Array.from(
    { length: Math.ceil(mutableBricks.length / TEMPLATE_BUILD_CHUNK_BRICKS) },
    (_, index) => mutableBricks.slice(
      index * TEMPLATE_BUILD_CHUNK_BRICKS,
      (index + 1) * TEMPLATE_BUILD_CHUNK_BRICKS,
    ),
  );
  const localBricksFor = (core: readonly SparseAdaptiveMassBrick[]) => {
    const keys = new Set(core.map((brick) => brick.key));
    for (const brick of core) {
      for (let axis = 0; axis < 3; axis += 1) for (const direction of [-1, 1]) {
        const coordinate = [...brick.coordinate] as [number, number, number];
        coordinate[axis] += direction;
        const neighbor = sparseBrickContainingCoordinate(atlas, coordinate);
        if (neighbor) keys.add(neighbor.key);
      }
    }
    return atlas.bricks.filter((brick) => keys.has(brick.key));
  };
  const variantAtlasAtLevels = (
    choose: (brick: SparseAdaptiveMassBrick) => SparseBrickResolution,
    sourceBricks: readonly SparseAdaptiveMassBrick[],
  ) => createSparseAdaptiveMassAtlas(atlas.dimensions, sourceBricks.map((brick) =>
    resampleBrick(brick, choose(brick))), atlas.generation, atlas.boundary,
    atlas.brickFineResolution);
  // One reusable full-grid scratch bounds host construction at accepted plus
  // one transient variant. Only cells/rows touching the mutable frontier are
  // copied into the persistent template library.
  const variantWorkspace = createSparseAtlasCompositeGridBuildWorkspace();
  const cells: SparseAtlasCompositeCell[] = [];
  const cellId = new Map<string, number>();
  const cellRanges = new Uint32Array(atlas.bricks.length * templateLevels.length * 2);
  const brickIndex = new Map(atlas.bricks.map((brick, index) => [brick.key, index]));
  const compactBrickCellRange = (grid: SparseAtlasCompositeGrid,
    brick: SparseAdaptiveMassBrick,
    resolution: SparseBrickResolution, context: string) => {
    const key = brick.key;
    const first = grid.cellBaseByBrick.get(key);
    if (first === undefined) {
      throw new Error(`Sparse CM12 ${context} brick ${key}/${resolution} has no compact base`);
    }
    let end = first;
    while (end < grid.cells.length && grid.cells[end]!.brickKey === key) end += 1;
    const count = end - first;
    const expected = sparseCM12BrickLiveCellCount(atlas, brick, resolution);
    if (count !== expected || count === 0) {
      throw new Error(`Sparse CM12 ${context} brick ${key}/${resolution} has ${count} `
        + `compact cells; expected clipped count ${expected}`);
    }
    for (let cell = first; cell < end; cell += 1) {
      const source = grid.cells[cell];
      if (!source || source.brickKey !== key || source.brickResolution !== resolution) {
        throw new Error(`Sparse CM12 ${context} brick ${key}/${resolution} compact range `
          + `[${first},${end}) is not contiguous at ${cell}`);
      }
    }
    return { first, count } as const;
  };
  // Preserve generation-zero IDs so the old direct dispatch and the new
  // accepted worklist address the same state while physical cutover lands.
  for (const source of acceptedGrid.cells) {
    cells.push(snapshotTemplateCell(source, cells.length));
    cellId.set(templateCellKey(source.brickKey, source.brickResolution,
      source.localIndex), source.id);
  }
  for (const brick of atlas.bricks) for (let levelIndex = 0;
    levelIndex < templateLevels.length; levelIndex += 1) {
    const range = 2 * (templateLevels.length * brickIndex.get(brick.key)! + levelIndex);
    const accepted = compactBrickCellRange(acceptedGrid, brick,
      brick.resolution, "accepted template");
    cellRanges[range] = accepted.first;
    cellRanges[range + 1] = accepted.count;
  }

  const rows: SparseAtlasGradientRow[] = [];
  const rowRequirements: number[][] = [];
  const rowKeys = new Set<string>();
  const appendRows = (grid: SparseAtlasCompositeGrid,
    accept: (row: SparseAtlasGradientRow) => boolean): void => {
    for (const source of grid.gradientRows) {
      if (!accept(source)) continue;
      const requirements = new Map<number, number>();
      const terms = source.terms.map((term) => {
        const sourceCell = grid.cells[term.cellId]!;
        if (!sourceCell) {
          throw new Error(`Sparse CM12 template row ${source.id} references cell ${
            term.cellId} outside ${grid.cells.length}`);
        }
        requirements.set(sourceCell.brickKey, sourceCell.brickResolution);
        const id = cellId.get(templateCellKey(sourceCell.brickKey,
          sourceCell.brickResolution, sourceCell.localIndex));
        if (id === undefined) throw new Error("Sparse CM12 template cell remap failed");
        return { cellId: id, coefficient: term.coefficient };
      });
      const rowKey = `${source.axis}/${source.centerFine.join("/")}/${terms.map((term) =>
        `${term.cellId}:${term.coefficient}`).join(",")}`;
      if (rowKeys.has(rowKey)) continue;
      rowKeys.add(rowKey);
      rows.push({ ...source, id: rows.length, centerFine: [...source.centerFine], terms });
      rowRequirements.push([...requirements].map(([key, level]) =>
        packedTemplateCellMetadata(brickIndex.get(key)!, level)));
    }
  };
  appendRows(acceptedGrid, () => true);
  for (let levelIndex = 0;
    levelIndex < templateLevels.length; levelIndex += 1) {
    const level = templateLevels[levelIndex]!;
    for (const core of mutableChunks) {
      const coreKeys = new Set(core.map((brick) => brick.key));
      const localBricks = localBricksFor(core);
      const grid = buildSparseAtlasCompositeGrid(
        variantAtlasAtLevels((brick) => mutableBrickKeys.has(brick.key)
          ? level : brick.resolution, localBricks), 0.5, variantWorkspace,
      );
      for (const brick of localBricks.filter((candidate) =>
        mutableBrickKeys.has(candidate.key))) {
        const range = 2 * (templateLevels.length * brickIndex.get(brick.key)! + levelIndex);
        const sourceRange = compactBrickCellRange(grid, brick, level,
          "variant template");
        let first = INVALID;
        for (let offset = 0; offset < sourceRange.count; offset += 1) {
          const source = grid.cells[sourceRange.first + offset]!;
          const key = templateCellKey(brick.key, level, source.localIndex);
          let id = cellId.get(key);
          if (id === undefined) {
            id = cells.length; cells.push(snapshotTemplateCell(source, id)); cellId.set(key, id);
          }
          first = Math.min(first, id);
        }
        cellRanges[range] = first;
        cellRanges[range + 1] = sourceRange.count;
      }
      appendRows(grid, (row) => row.terms.some((term) =>
        coreKeys.has(grid.cells[term.cellId]!.brickKey)));
    }
  }
  const rungPairs = templateLevels.slice(1).map((high, index) =>
    [templateLevels[index]!, high] as const);
  for (let axis = 0; axis < 3; axis += 1) for (const [low, high] of rungPairs) {
    // Both parity phases are required: a physical face needs templates for
    // low→high and high→low accepted generations.
    for (let phase = 0; phase < 2; phase += 1) {
      for (const core of mutableChunks) {
        const coreKeys = new Set(core.map((brick) => brick.key));
        const localBricks = localBricksFor(core);
        const variant = variantAtlasAtLevels((brick) => mutableBrickKeys.has(brick.key)
          ? ((brick.coordinate[axis]! & 1) ^ phase) === 0 ? low : high
          : brick.resolution, localBricks);
        const variantGrid = buildSparseAtlasCompositeGrid(
          variant, 0.5, variantWorkspace,
        );
        appendRows(variantGrid, (row) =>
          row.axis === axis && row.kind === "mixed-seam"
            && row.terms.some((term) => coreKeys.has(
              variantGrid.cells[term.cellId]!.brickKey)));
      }
    }
  }

  const incidences: { row: number; term: number }[][] = Array.from(
    { length: cells.length }, () => []);
  let termCount = 0;
  for (const row of rows) for (let term = 0; term < row.terms.length; term += 1) {
    incidences[row.terms[term]!.cellId]!.push({ row: row.id, term: termCount++ });
  }
  const incidenceCount = incidences.reduce((sum, list) => sum + list.length, 0);
  const pressureEdgeCounts = new Uint32Array(cells.length);
  for (const row of rows) for (const own of row.terms) {
    pressureEdgeCounts[own.cellId] += row.terms.length - 1;
  }
  const pressureEdgeOffsets = new Uint32Array(cells.length + 1);
  for (let cell = 0; cell < cells.length; cell += 1) {
    pressureEdgeOffsets[cell + 1] = pressureEdgeOffsets[cell]! + pressureEdgeCounts[cell]!;
  }
  const pressureEdgeCount = pressureEdgeOffsets[cells.length]!;
  let at = TEMPLATE_HEADER_WORDS;
  const cellOffset = at; at += TEMPLATE_CELL_RECORD_WORDS * cells.length;
  const rowOffset = at; at += TEMPLATE_ROW_PLANE_COUNT * rows.length;
  const termOffset = at; at += 2 * termCount;
  const incidenceOffset = at; at += cells.length + 1;
  const incidenceRecordOffset = at; at += 2 * incidenceCount;
  const cellRangeOffset = at; at += cellRanges.length;
  const rowRequirementOffset = at;
  const rowRequirementOffsets = rowRequirements.map((requirements) => {
    const result = at; at += 1 + requirements.length; return result;
  });
  const pressureEdgeOffset = at; at += cells.length + 1;
  const pressureEdgeRecordOffset = at; at += 3 * pressureEdgeCount;
  const words = new Uint32Array(at);
  words.set([TEMPLATE_MAGIC, 1, cells.length, rows.length, termCount, incidenceCount,
    cellOffset, rowOffset, termOffset, incidenceOffset, incidenceRecordOffset,
    cellRangeOffset, rowRequirementOffset, atlas.bricks.length], 0);
  words[15] = pressureEdgeOffset;
  for (const cell of cells) {
    const base = cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell.id;
    if (cell.openFraction !== 1 || cell.openVolume !== cell.volume
      || cell.separatingPressureMinimum) {
      throw new Error(`Sparse CM12 resident cell ${cell.id} has unsupported static boundary data`);
    }
    setF32(words, base, cell.centerFine[0]); setF32(words, base + 1, cell.centerFine[1]);
    setF32(words, base + 2, cell.centerFine[2]); setF32(words, base + 3, cell.volume);
    setF32(words, base + 4, cell.widthsFine[0]); setF32(words, base + 5, cell.widthsFine[1]);
    setF32(words, base + 6, cell.widthsFine[2]);
    words[base + 7] = packedTemplateCellMetadata(
      brickIndex.get(cell.brickKey)!, cell.brickResolution,
    );
  }
  let nextTerm = 0;
  for (const row of rows) {
    words[templateRowWord(rowOffset, rows.length, 0, row.id)]
      = packedTemplateRowTerms(nextTerm, row.terms.length);
    words[templateRowWord(rowOffset, rows.length, 1, row.id)]
      = packedTemplateRowMetadata(rowRequirementOffsets[row.id]!, row.kind, row.axis);
    setF32(words, templateRowWord(rowOffset, rows.length, 2, row.id), row.dualWeight);
    setF32(words, templateRowWord(rowOffset, rows.length, 3, row.id), row.area);
    setF32(words, templateRowWord(rowOffset, rows.length, 4, row.id), row.distance);
    setF32(words, templateRowWord(rowOffset, rows.length, 5, row.id), row.exteriorPhi ?? 0.5);
    setF32(words, templateRowWord(rowOffset, rows.length, 6, row.id), row.centerFine[0]);
    setF32(words, templateRowWord(rowOffset, rows.length, 7, row.id), row.centerFine[1]);
    setF32(words, templateRowWord(rowOffset, rows.length, 8, row.id), row.centerFine[2]);
    for (const term of row.terms) {
      words[termOffset + 2 * nextTerm] = term.cellId;
      setF32(words, termOffset + 2 * nextTerm + 1, term.coefficient); nextTerm += 1;
    }
  }
  let nextIncidence = 0;
  for (let cell = 0; cell < incidences.length; cell += 1) {
    words[incidenceOffset + cell] = nextIncidence;
    for (const incidence of incidences[cell]!) {
      words[incidenceRecordOffset + 2 * nextIncidence] = incidence.row;
      words[incidenceRecordOffset + 2 * nextIncidence + 1] = incidence.term;
      nextIncidence += 1;
    }
  }
  words[incidenceOffset + cells.length] = nextIncidence;
  words.set(cellRanges, cellRangeOffset);
  let requirementAt = rowRequirementOffset;
  for (const requirements of rowRequirements) {
    words[requirementAt++] = requirements.length;
    words.set(requirements, requirementAt); requirementAt += requirements.length;
  }
  words.set(pressureEdgeOffsets, pressureEdgeOffset);
  const pressureEdgeCursor = pressureEdgeOffsets.slice(0, cells.length);
  for (const row of rows) for (const own of row.terms) for (const other of row.terms) {
    if (other.cellId === own.cellId) continue;
    const edge = pressureEdgeCursor[own.cellId]++;
    const record = pressureEdgeRecordOffset + 3 * edge;
    words[record] = row.id;
    words[record + 1] = other.cellId;
    setF32(words, record + 2,
      own.coefficient * row.dualWeight * other.coefficient);
  }
  const initialCellWorklist = Uint32Array.from({ length: acceptedGrid.cells.length },
    (_, id) => id);
  const initialRowWorklist = Uint32Array.from({ length: acceptedGrid.gradientRows.length },
    (_, id) => id);
  return { words, cellCount: cells.length, rowCount: rows.length,
    initialCellWorklist, initialRowWorklist,
    initialDensity: Float32Array.from(cells, (cell) => cell.density),
    initialGamma: Float32Array.from(cells, (cell) => cell.gamma) };
}

interface ResidentStateLayout {
  readonly floatCount: number;
  readonly densityA: number; readonly densityB: number;
  readonly gammaA: number; readonly gammaB: number;
  readonly cellVelocityA: number; readonly cellVelocityB: number;
  readonly faceA: number; readonly faceB: number;
  readonly pressure: number; readonly rhs: number; readonly diagonal: number;
  readonly liquid: number; readonly theta: number; readonly residual: number;
  readonly preconditioned: number; readonly direction: number;
  readonly applied: number; readonly divergence: number;
  readonly sharpeningDelta: number; readonly symmetryGamma: number;
  readonly solidCellOpen: number; readonly solidRowData: number;
  /** Four floats per tracer: fine-lattice position, then a live flag. */
  readonly tracers: number;
  /**
   * Base of the pressure journal, or zero when the solver was built without
   * the capability. Past the tracers for the same reason they sit past the
   * physics fields, and additionally because it is the only region whose size
   * is a debug choice rather than a property of the scene.
   */
  readonly journal: number;
  readonly journalLayout: SparseCM12PressureJournalLayout;
  /** VEX1 accepted air-band cache plus prior liquid-interface seed receipts. */
  readonly velocityExtensionAcceptedVelocity: number;
}

export interface SparseCM12FinePresentationPlan {
  readonly plan: FineLevelSetBrickPlan;
  readonly metadata: Uint32Array;
  readonly worklist: Uint32Array;
}

export type SparseCM12PresentationPageResolution = 4 | 8 | 16;
export const SPARSE_CM12_PRESENTATION_PAGE_SHIFT = 21;

// Compact presentation source word. B/P <= 2 retains the legacy three-bit
// local-page address, a 24-bit sparse-atlas leaf, and five span bits. B/P == 4
// uses six local-page bits for ordinary leaves and bit 31 to discriminate the
// one-page macro form. A macro leaf therefore costs one native-scale page
// instead of either disappearing or expanding into O(span^3) fine pages.
export const SPARSE_CM12_PRESENTATION_SOURCE_BRICK_BITS = 24;
export const SPARSE_CM12_PRESENTATION_SOURCE_BRICK_MASK =
  (2 ** SPARSE_CM12_PRESENTATION_SOURCE_BRICK_BITS) - 1;
export const SPARSE_CM12_PRESENTATION_SOURCE_SPAN_SHIFT = 27;
// Log 31 would make the shader's 2*span page extent wrap u32.
export const SPARSE_CM12_PRESENTATION_MAX_SPAN_LOG = 30;

export interface SparseCM12FinePresentationSource {
  readonly brick: number;
  readonly octant: number;
  readonly spanBricks: number;
}

export function encodeSparseCM12FinePresentationSource(
  brick: number,
  octant: number,
  spanBricks: number,
  brickFineResolution = 16,
  presentationPageResolution = 16,
): number {
  const spanLog = Math.log2(spanBricks);
  if (!Number.isInteger(brick) || brick < 0
    || brick > SPARSE_CM12_PRESENTATION_SOURCE_BRICK_MASK) {
    throw new RangeError(`Sparse CM12 presentation brick ${brick} exceeds the 24-bit ABI`);
  }
  const pagesPerAxis = brickFineResolution / presentationPageResolution;
  const pageCount = pagesPerAxis ** 3;
  if (!Number.isInteger(octant) || octant < 0 || octant >= pageCount) {
    throw new RangeError(`Sparse CM12 presentation octant ${octant} is invalid`);
  }
  if (!Number.isInteger(spanLog) || spanLog < 0
    || spanLog > SPARSE_CM12_PRESENTATION_MAX_SPAN_LOG) {
    throw new RangeError(`Sparse CM12 presentation span ${spanBricks} is not representable`);
  }
  if (pagesPerAxis < 4) {
    return ((spanLog << SPARSE_CM12_PRESENTATION_SOURCE_SPAN_SHIFT)
      | (brick << 3) | octant) >>> 0;
  }
  // 16-cell bricks need six local-page bits. Ordinary pages use the low six
  // bits and leave bit 31 clear; macro pages use bit 31 as a discriminator,
  // retaining the original 24-bit brick and five-bit span range.
  return spanLog === 0
    ? ((brick << 6) | octant) >>> 0
    : (0x8000_0000 | (spanLog << 24) | brick) >>> 0;
}

export function decodeSparseCM12FinePresentationSource(
  source: number,
  brickFineResolution = 16,
  presentationPageResolution = 16,
): SparseCM12FinePresentationSource {
  if (brickFineResolution / presentationPageResolution === 4) {
    const macro = (source & 0x8000_0000) !== 0;
    return macro ? {
      brick: source & SPARSE_CM12_PRESENTATION_SOURCE_BRICK_MASK,
      octant: 0,
      spanBricks: 2 ** ((source >>> 24) & 31),
    } : {
      brick: (source >>> 6) & SPARSE_CM12_PRESENTATION_SOURCE_BRICK_MASK,
      octant: source & 63,
      spanBricks: 1,
    };
  }
  const spanLog = source >>> SPARSE_CM12_PRESENTATION_SOURCE_SPAN_SHIFT;
  return {
    brick: (source >>> 3) & SPARSE_CM12_PRESENTATION_SOURCE_BRICK_MASK,
    octant: source & 7,
    spanBricks: 2 ** spanLog,
  };
}

export function sparseCM12FinePresentationPlan(
  atlas: SparseAdaptiveMassAtlas,
  presentationPageResolution: SparseCM12PresentationPageResolution = 16,
): SparseCM12FinePresentationPlan {
  const brickFineResolution = atlas.brickFineResolution;
  if (presentationPageResolution > brickFineResolution
    || brickFineResolution % presentationPageResolution !== 0) {
    throw new RangeError(`Sparse CM12 presentation page ${presentationPageResolution} does not divide brick ladder ${brickFineResolution}`);
  }
  const pagesPerAxis = brickFineResolution / presentationPageResolution;
  const sampleDimensions = atlas.dimensions;
  const brickDimensions = sampleDimensions.map((value) =>
    Math.ceil(value / presentationPageResolution)) as
    [number, number, number];
  const pages: { key: number; brick: number; octant: number; spanBricks: number }[] = [];
  let maximumSpanLog = 0;
  for (let brick = 0; brick < atlas.bricks.length; brick += 1) {
    const source = atlas.bricks[brick]!;
    const spanBricks = sparseBrickSpan(source);
    const spanLog = Math.log2(spanBricks);
    maximumSpanLog = Math.max(maximumSpanLog, spanLog);
    // Macro leaves publish one page across their complete physical extent.
    // Ordinary leaves retain (B/P)^3 independently addressable pages.
    // This makes cost proportional to leaves while preserving deep liquid and
    // closed walls in the global presentation.
    if (spanBricks > 1) {
      const coordinate = source.coordinate.map((value) => pagesPerAxis * value) as
        [number, number, number];
      pages.push({
        key: coordinate[0] + brickDimensions[0]
          * (coordinate[1] + brickDimensions[1] * coordinate[2]),
        brick,
        octant: 0,
        spanBricks,
      });
      continue;
    }
    for (let oz = 0; oz < pagesPerAxis; oz += 1)
      for (let oy = 0; oy < pagesPerAxis; oy += 1)
        for (let ox = 0; ox < pagesPerAxis; ox += 1) {
          const coordinate = [pagesPerAxis * source.coordinate[0] + ox,
            pagesPerAxis * source.coordinate[1] + oy,
            pagesPerAxis * source.coordinate[2] + oz] as const;
          if (coordinate.some((value, axis) => value >= brickDimensions[axis])) continue;
          pages.push({
            key: coordinate[0] + brickDimensions[0]
              * (coordinate[1] + brickDimensions[1] * coordinate[2]),
            brick,
            octant: ox + pagesPerAxis * (oy + pagesPerAxis * oz),
            spanBricks,
          });
        }
  }
  pages.sort((left, right) => left.key - right.key);
  for (let page = 1; page < pages.length; page += 1) {
    if (pages[page - 1]!.key === pages[page]!.key) {
      throw new Error(`Sparse CM12 presentation page ${pages[page]!.key} has two owners`);
    }
  }
  const pageCount = pages.length;
  const metadata = new Uint32Array(FINE_LEVELSET_METADATA_WORDS * pageCount);
  for (let page = 0; page < pageCount; page += 1) {
    metadata[FINE_LEVELSET_METADATA_WORDS * page] = page;
    metadata[FINE_LEVELSET_METADATA_WORDS * page + 1] = pages[page]!.key;
    metadata[FINE_LEVELSET_METADATA_WORDS * page + 2] = 1;
    // Compact Sparse CM12 owns this metadata word: source brick plus its local
    // page. Publication can therefore address the packed cell directly
    // instead of binary-searching the retained directory once per sample.
    metadata[FINE_LEVELSET_METADATA_WORDS * page + 3] =
      encodeSparseCM12FinePresentationSource(
        pages[page]!.brick, pages[page]!.octant, pages[page]!.spanBricks,
        brickFineResolution, presentationPageResolution,
      );
  }
  // Compact mode deliberately stops after the active physical-page list.  It
  // has no `logicalBrickCount`-sized direct directory; renderer lookup binary
  // searches the key-sorted metadata instead.
  const worklist = new Uint32Array(FINE_LEVELSET_WORKSET_HEADER_WORDS + pageCount);
  worklist.set([1, pageCount, pageCount,
    (FINE_LEVELSET_COMPACT_LOOKUP_FLAG | 3 | (maximumSpanLog << 8)
      | (brickFineResolution << 16)
      | (presentationPageResolution << SPARSE_CM12_PRESENTATION_PAGE_SHIFT)) >>> 0,
    Math.ceil(pageCount / WORKGROUP_SIZE), 1, 1]);
  for (let page = 0; page < pageCount; page += 1) {
    worklist[FINE_LEVELSET_WORKSET_HEADER_WORDS + page] = page;
  }
  const samplesPerBrick = presentationPageResolution ** 3;
  const payloadCapacityBytes = pageCount * samplesPerBrick * 4;
  const metadataCapacityBytes = metadata.byteLength;
  const worklistBytes = worklist.byteLength;
  return {
    metadata,
    worklist,
    plan: {
      domainOrigin: [0, 0, 0],
      finestCellDimensions: sampleDimensions,
      finestCellWidth: 1,
      fineFactor: 1,
      fineCellWidth: 1,
      brickResolution: presentationPageResolution,
      sampleDimensions,
      brickDimensions,
      logicalBrickCount: brickDimensions[0] * brickDimensions[1] * brickDimensions[2],
      maximumResidentBricks: pageCount,
      samplesPerBrick,
      payloadBytesPerBrick: samplesPerBrick * 4,
      payloadCapacityBytes,
      metadataCapacityBytes,
      worklistBytes,
      allocatedBytes: payloadCapacityBytes + metadataCapacityBytes + worklistBytes,
    },
  };
}

export interface SparseCM12GPUActivityRecord {
  readonly scoreByte: number;
  readonly reasons: number;
  /** True when reason bit 8 identifies represented fluid under two fine cells thick. */
  readonly thinFluid: boolean;
  readonly hotEpochs: number;
  readonly quietEpochs: number;
  /** Mean intensive density retained in the brick at the last activity census. */
  readonly meanDensity: number;
  /** Density-weighted local brick moments used by the temporal activity score. */
  readonly densityMoments: readonly [number, number, number];
  /** GPU-authored request for the next candidate topology epoch. */
  readonly plannedResolution: SparseBrickResolution;
  readonly planReasons: number;
  readonly active: boolean;
  readonly activatedStep: number;
  /** Accepted logical level. It remains equal to packed topology until publication. */
  readonly acceptedResolution: SparseBrickResolution;
  readonly candidateResolution: SparseBrickResolution;
  /** 0 retained, 1 transfer pending, 2 invalid/rejected. */
  readonly candidateStatus: 0 | 1 | 2;
  readonly candidateEpoch: number;
  readonly transferMassBeforeFineCells: number;
  readonly transferMassAfterFineCells: number;
  readonly transferMassErrorFineCells: number;
  readonly transferGammaErrorFineCells: number;
  readonly transferMomentumErrorFineCells: readonly [number, number, number];
  /** 0 not requested, 1 conservative cell transfer passed, 2 rejected. */
  readonly transferStatus: 0 | 1 | 2;
  readonly transferExteriorFluxErrorFineAreas: readonly [number, number, number,
    number, number, number];
  readonly maximumAbsoluteTransferFluxErrorFineAreas: number;
  /** 0 not requested, 1 exterior flux transfer passed, 2 rejected. */
  readonly faceTransferStatus: 0 | 1 | 2;
  /** Directional 3x3x3 free-surface/swept support mask; bit 13 is unused. */
  readonly supportMask: number;
  /** Characteristic-swept subset whose destination must be physically fine. */
  readonly sweptSupportMask: number;
  /** Maximum accepted-fluid displacement during one step, in finest cells. */
  readonly maximumVelocityTravelFineCells: number;
  /** Sub-residency-threshold mass discarded by the latest retirement transaction. */
  readonly retiredResidueMassFineCells: number;
  /** True when this candidate participates in the current shadow transaction. */
  readonly topologyPreparationScheduled: boolean;
  readonly topologyPreparationEpoch: number;
  /** QA receipt for a device-claimed dynamic topology page. Undefined for
   * packed host-template candidates and bricks that have not claimed a page. */
  readonly topologyPage: number | undefined;
}

export interface SparseCM12GPUActivitySnapshot {
  readonly acceptedSteps: number;
  readonly records: readonly SparseCM12GPUActivityRecord[];
}

/** Explicit QA materialization. Production rendering consumes sparse buffers
 * directly and never constructs these finest-domain arrays. */
export interface SparseCM12DiagnosticFields {
  readonly density: Float32Array;
  readonly gamma: Float32Array;
  readonly velocity: Float32Array;
  readonly pressure: Float32Array;
  readonly divergence: Float32Array;
}

export type SparseCM12TemporalSeedQAMode = "current" | "change";

/** Construction-only census for the immutable temporal-seed A/B. */
export interface SparseCM12TemporalSeedQA {
  readonly mode: SparseCM12TemporalSeedQAMode;
  readonly seedCells: number;
  readonly firstDilationCells: number;
  readonly finalCells: number;
  readonly finalRows: number;
}

/** QA-only materialization of the GPU-owned FCA1 accepted-frame header. */
export interface SparseCM12FrameControlQA {
  readonly phase: number;
  readonly fault: number;
  readonly firstFaultOwner: number;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly sealedGeneration: number;
  readonly scalarParity: number;
  readonly faceParity: number;
  readonly coverage: number;
  readonly committedFrames: number;
}

/** Header-only VEX1 receipt. Capacity-sized comparison arrays remain opt-in. */
export interface SparseCM12VelocityExtensionHeaderQA {
  readonly flags: number;
  readonly phase: number;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly topologyGeneration: number;
  readonly capacity: number;
  readonly rootCount: number;
  readonly blastCount: number;
  readonly maximumDepth: number;
  readonly executedCellCount: number;
  readonly reusedCellCount: number;
  readonly faultCount: number;
  readonly uncoveredWriteCount: number;
  readonly firstFault?: { readonly cell: number; readonly depth: number };
  readonly framePlanProvenance?: {
    readonly ownerBrick: number; readonly ownerTile: number; readonly slot: number;
    readonly tileGeneration: number; readonly packedStageMasks: number;
    readonly stage0MaskLow: number;
  };
}

const align4 = (value: number): number => (value + 3) & ~3;

/**
 * Markers the tracer view seeds at most: 262,144 x 16 B = 4 MiB of state.
 *
 * The budget is a count, not a fraction of the grid, because that is the whole
 * economy of the view — cost is decoupled from resolution, so refining the
 * scene does not refine the marker cloud and the picture stays comparable
 * across lanes.
 */
export const SPARSE_CM12_TRACER_BUDGET = 1_048_576;

/**
 * Markers per finest cell at the densest seeding.
 *
 * Not one-per-cell: two markers born inside the same cell do separate, because
 * each traces the characteristic from its own position and the two cross into
 * neighbouring cells at different times. Four is where the cloud reads as a
 * continuous body instead of a visible lattice, and it is still only ~1.6
 * markers per cell along each axis, so no marker is tracking detail the grid
 * does not carry. The drawn dot is sized from the spacing, so raising this
 * shrinks the dots by the same factor and the cloud does not turn into a wall.
 */
export const SPARSE_CM12_TRACER_DENSITY = 4;

export interface SparseCM12TracerLattice {
  readonly dimensions: readonly [number, number, number];
  readonly count: number;
  /** Lattice origin in fine cells; the lattice is centred in the domain. */
  readonly originFine: readonly [number, number, number];
  /** Isotropic seed spacing in fine cells. */
  readonly spacingFine: number;
}

/**
 * The seed lattice, as a pure function of the domain.
 *
 * Purity is the point: the compute pass and the vertex stage both reconstruct a
 * marker's seed position from its index through this same arithmetic, so the
 * colour a marker is drawn with never has to be stored, uploaded, or advected.
 * That is also why the spacing is isotropic — an axis-dependent spacing would
 * make the initial cloud read as already deformed.
 *
 * Markers seeded into air are retired rather than packed out, so the live
 * fraction is the scene's fill fraction. Compaction would have to be redone
 * every topology epoch to buy back memory that is already only single-digit
 * MiB, which is not a trade worth making.
 */
export function sparseCM12TracerLattice(
  dimensions: readonly [number, number, number],
  budget: number = SPARSE_CM12_TRACER_BUDGET,
): SparseCM12TracerLattice {
  const domain = dimensions.map((value) => Math.max(1, Math.floor(value)));
  const empty: SparseCM12TracerLattice = {
    dimensions: [0, 0, 0], count: 0, originFine: [0, 0, 0], spacingFine: 1,
  };
  if (!(budget >= 1)) return empty;
  const extent = (spacing: number) => domain.map(
    (value) => Math.max(1, Math.floor(value / spacing))) as [number, number, number];
  const finest = Math.cbrt(1 / SPARSE_CM12_TRACER_DENSITY);
  let spacing = Math.max(finest,
    Math.cbrt(domain[0]! * domain[1]! * domain[2]! / budget));
  let lattice = extent(spacing);
  // The cube-root estimate is exact only for a domain that divides evenly, so
  // walk the spacing up until the floor()ed lattice actually fits the budget.
  for (let attempt = 0; attempt < 64
    && lattice[0] * lattice[1] * lattice[2] > budget; attempt += 1) {
    spacing *= 1.05;
    lattice = extent(spacing);
  }
  const count = lattice[0] * lattice[1] * lattice[2];
  if (count > budget) return empty;
  return {
    dimensions: lattice,
    count,
    originFine: domain.map((value, axis) =>
      0.5 * (value - lattice[axis]! * spacing)) as [number, number, number],
    spacingFine: spacing,
  };
}

function residentStateLayout(
  cellCount: number,
  rowCount: number,
  rigidCoupling: boolean,
  tracerCount: number,
  journal: SparseCM12PressureJournalCapacityRequest = {},
): ResidentStateLayout {
  let at = 0;
  const cells = () => { const result = at; at += align4(cellCount); return result; };
  const rows = () => { const result = at; at += align4(rowCount); return result; };
  const cellVectors = () => { const result = at; at += align4(4 * cellCount); return result; };
  const solidRows = () => { const result = at; at += align4(3 * rowCount); return result; };
  return {
    densityA: cells(), densityB: cells(), gammaA: cells(), gammaB: cells(),
    cellVelocityA: cellVectors(), cellVelocityB: cellVectors(),
    faceA: rows(), faceB: rows(), pressure: cells(), rhs: cells(), diagonal: cells(),
    liquid: cells(), theta: rows(), residual: cells(), preconditioned: cells(),
    direction: cells(), applied: cells(), divergence: cells(),
    sharpeningDelta: cells(), symmetryGamma: cells(),
    solidCellOpen: rigidCoupling ? cells() : 0,
    solidRowData: rigidCoupling ? solidRows() : 0,
    // Tracers are presentation-only and are sized by the domain rather than by
    // the topology, so they sit past every physics field: a re-meshed frame
    // changes cell counts, and a marker must not silently land on a pressure
    // row because the arena in front of it grew.
    tracers: (() => { const result = at; at += align4(4 * tracerCount); return result; })(),
    ...(() => {
      const journalLayout = sparseCM12PressureJournalLayout({
        iterationCapacity: journal.iterationCapacity ?? 0,
        snapshotCapacity: journal.snapshotCapacity ?? 0,
        cellStride: journal.iterationCapacity ? cellCount : 0,
      });
      const base = journalLayout.floatCount > 0 ? at : 0;
      at += align4(journalLayout.floatCount);
      return { journal: base, journalLayout };
    })(),
    velocityExtensionAcceptedVelocity: cellVectors(),
    floatCount: at,
  };
}

function setF32(words: Uint32Array, index: number, value: number): void {
  new DataView(words.buffer).setFloat32(index * 4, value, true);
}

function packResidentTopology(
  atlas: SparseAdaptiveMassAtlas,
  grid: SparseAtlasCompositeGrid,
  mutableBrickKeys: ReadonlySet<number>,
): PackedResidentTopology {
  // Physical cells, rows, terms and incidence live in topologyArena. Older
  // versions serialized the complete graph a second time into `topology`,
  // although its consumers only read leaf ownership and brick records. On a
  // large ocean that dead duplicate was hundreds of MiB plus millions of JS
  // incidence objects. Keep this buffer as the compact ownership index it is.
  let incidenceCount = 0;
  for (const row of grid.gradientRows) incidenceCount += row.terms.length;
  const maximumSpanLog = Math.log2(atlas.maximumSpanBricks);
  if (!Number.isInteger(maximumSpanLog) || maximumSpanLog < 0
    || maximumSpanLog > SPARSE_CM12_PRESENTATION_MAX_SPAN_LOG) {
    throw new RangeError(`Sparse CM12 maximum span ${atlas.maximumSpanBricks} is invalid`);
  }
  const ownership = sparseCM12OwnershipTablePlan(atlas.bricks.length);
  const hashCapacity = ownership.hashCapacity;
  let at = 0;
  const brickLookupOffset = at; at += hashCapacity;
  const brickOffset = at; at += 2 * atlas.bricks.length;
  const backgroundOwnerOffset = at; at += 2;
  const words = new Uint32Array(at);
  words.fill(INVALID, brickLookupOffset, brickOffset);

  const candidateSlotByBrick = new Uint32Array(atlas.bricks.length).fill(INVALID);
  let candidateBrickCount = 0;
  for (let brick = 0; brick < atlas.bricks.length; brick += 1) {
    if (sparseBrickSpan(atlas.bricks[brick]!) === 1
      && mutableBrickKeys.has(atlas.bricks[brick]!.key)) {
      candidateSlotByBrick[brick] = candidateBrickCount++;
    }
  }
  words[backgroundOwnerOffset] = candidateBrickCount;
  words[backgroundOwnerOffset + 1] = (0x8000_0000
    | maximumSpanLog) >>> 0;
  for (let brick = 0; brick < atlas.bricks.length; brick += 1) {
    const source = atlas.bricks[brick]!;
    let slot = (Math.imul(source.key, 0x9e37_79b1) >>> 0) & (hashCapacity - 1);
    while (words[brickLookupOffset + slot] !== INVALID) {
      slot = (slot + 1) & (hashCapacity - 1);
    }
    words[brickLookupOffset + slot] = brick;
    const record = brickOffset + 2 * brick;
    // Log2(span) and the optional mutation-page slot share one word. Cell
    // ranges live only in the immutable template table, so the compact owner
    // directory needs no duplicate first/count pair per brick.
    words[record] = Math.log2(sparseBrickSpan(atlas.bricks[brick]!))
      | (candidateSlotByBrick[brick] === INVALID
        ? 0 : ((candidateSlotByBrick[brick]! + 1) << 5));
    words[record + 1] = source.key;
  }
  return { words, cellOffset: 0, rowOffset: 0, termOffset: 0, incidenceOffset: 0,
    incidenceRecordOffset: 0, brickLookupOffset, brickOffset, backgroundOwnerOffset,
    brickCount: atlas.bricks.length, candidateBrickCount, incidenceCount };
}

function uploadBuffer(
  device: GPUDevice,
  label: string,
  source: ArrayBufferView,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({ label, size: Math.max(4, source.byteLength), usage });
  device.queue.writeBuffer(buffer, 0, source.buffer as ArrayBuffer,
    source.byteOffset, source.byteLength);
  return buffer;
}

/** Pressure SpMV reads only the CSR edge tail of the physical template ABI.
 * Keep its alias-breaking read-only binding compact instead of cloning every
 * cell, row, term and incidence record a second time. */
function compactPressureTopology(
  templates: PackedResidentTopologyTemplates,
  atlas: SparseAdaptiveMassAtlas,
): Uint32Array {
  const brickCount = atlas.bricks.length;
  const sourceOffset = templates.words[15]!;
  if (sourceOffset < TEMPLATE_HEADER_WORDS || sourceOffset >= templates.words.length) {
    throw new Error(`Sparse CM12 pressure edge offset ${sourceOffset} is invalid`);
  }
  const cellOffset = templates.words[6]!;
  const edgeCount = templates.words[sourceOffset + templates.cellCount]!;
  const edgeRecords = sourceOffset + templates.cellCount + 1;
  const contributions = new Map<number, number[]>();
  for (let cell = 0; cell < templates.cellCount; cell += 1) {
    const brick = templateCellBrick(
      templates.words, cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell,
    );
    const begin = templates.words[sourceOffset + cell]!;
    const end = templates.words[sourceOffset + cell + 1]!;
    for (let edge = begin; edge < end; edge += 1) {
      const other = templates.words[edgeRecords + 3 * edge + 1]!;
      const otherBrick = templateCellBrick(
        templates.words, cellOffset + TEMPLATE_CELL_RECORD_WORDS * other,
      );
      if (otherBrick === brick) continue;
      const key = brick * brickCount + otherBrick;
      const list = contributions.get(key);
      if (list) list.push(edge); else contributions.set(key, [edge]);
    }
  }
  const ordered = [...contributions.entries()].sort(([left], [right]) => left - right);
  const coarseOffsets = new Uint32Array(brickCount + 1);
  for (const [key] of ordered) coarseOffsets[Math.floor(key / brickCount) + 1] += 1;
  for (let brick = 0; brick < brickCount; brick += 1) {
    coarseOffsets[brick + 1] += coarseOffsets[brick]!;
  }
  const contributionCount = ordered.reduce((sum, [, list]) => sum + list.length, 0);
  // Pressure consumers already keep neighbor IDs in their dense hot cache.
  // Preserve only CSR offsets plus contiguous row and coefficient planes here;
  // the source template's neighbor word is construction-only after `ordered`.
  const compactEdgeWords = templates.cellCount + 1 + 2 * edgeCount;
  const coarseBase = TEMPLATE_HEADER_WORDS + compactEdgeWords;
  const coarseRecordBase = coarseBase + 4 + coarseOffsets.length;
  const contributionBase = coarseRecordBase + 3 * ordered.length;
  const hierarchyBase = contributionBase + contributionCount;
  const brickDimensions = atlas.dimensions.map((value) =>
    Math.ceil(value / atlas.brickFineResolution));
  const hierarchyScales: number[] = [];
  for (let scale = 2; ; scale *= 2) {
    hierarchyScales.push(scale);
    if (brickDimensions.every((value) => value <= scale)) break;
    if (!Number.isSafeInteger(scale * 2)) {
      throw new RangeError("Sparse CM12 pressure hierarchy scale overflow");
    }
  }
  const hierarchy = hierarchyScales.map((scale) => {
    const dimensions = brickDimensions.map((value) => Math.ceil(value / scale));
    const groupCount = dimensions[0]! * dimensions[1]! * dimensions[2]!;
    const parents = Uint32Array.from(atlas.bricks, (brick) => {
      const coordinate = brick.coordinate.map((value) => Math.floor(value / scale));
      return coordinate[0]! + dimensions[0]!
        * (coordinate[1]! + dimensions[1]! * coordinate[2]!);
    });
    const childCounts = new Uint32Array(groupCount);
    for (const parent of parents) childCounts[parent] += 1;
    const childOffsets = new Uint32Array(groupCount + 1);
    for (let group = 0; group < groupCount; group += 1) {
      childOffsets[group + 1] = childOffsets[group]! + childCounts[group]!;
    }
    const childCursor = childOffsets.slice(0, groupCount);
    const children = new Uint32Array(brickCount);
    parents.forEach((parent, brick) => { children[childCursor[parent]++] = brick; });
    const internalCounts = new Uint32Array(groupCount);
    ordered.forEach(([key]) => {
      const source = Math.floor(key / brickCount), target = key % brickCount;
      if (parents[source] === parents[target]) internalCounts[parents[source]!] += 1;
    });
    const internalOffsets = new Uint32Array(groupCount + 1);
    for (let group = 0; group < groupCount; group += 1) {
      internalOffsets[group + 1] = internalOffsets[group]! + internalCounts[group]!;
    }
    const internalCursor = internalOffsets.slice(0, groupCount);
    const internalEdges = new Uint32Array(internalOffsets[groupCount]!);
    ordered.forEach(([key], edge) => {
      const source = Math.floor(key / brickCount), target = key % brickCount;
      if (parents[source] === parents[target]) {
        internalEdges[internalCursor[parents[source]!]++] = edge;
      }
    });
    const crossContributionsByPair = new Map<number, number[]>();
    ordered.forEach(([key], edge) => {
      const source = Math.floor(key / brickCount), target = key % brickCount;
      const sourceGroup = parents[source]!, targetGroup = parents[target]!;
      if (sourceGroup === targetGroup) return;
      const pair = sourceGroup * groupCount + targetGroup;
      const list = crossContributionsByPair.get(pair);
      if (list) list.push(edge); else crossContributionsByPair.set(pair, [edge]);
    });
    const cross = [...crossContributionsByPair.entries()]
      .sort(([left], [right]) => left - right);
    const crossOffsets = new Uint32Array(groupCount + 1);
    for (const [pair] of cross) crossOffsets[Math.floor(pair / groupCount) + 1] += 1;
    for (let group = 0; group < groupCount; group += 1) {
      crossOffsets[group + 1] += crossOffsets[group]!;
    }
    const crossContributionCount = cross.reduce((sum, [, list]) => sum + list.length, 0);
    return { groupCount, parents, childOffsets, children, internalOffsets, internalEdges,
      cross, crossOffsets, crossContributionCount };
  });
  const hierarchyDescriptorWords = 10;
  let hierarchyWords = 1 + hierarchyDescriptorWords * hierarchy.length;
  for (const level of hierarchy) hierarchyWords += level.parents.length
    + level.childOffsets.length + level.children.length
    + level.internalOffsets.length + level.internalEdges.length
    + level.crossOffsets.length + 3 * level.cross.length
    + level.crossContributionCount;
  const result = new Uint32Array(hierarchyBase + hierarchyWords);
  result.set(templates.words.subarray(0, TEMPLATE_HEADER_WORDS));
  result[15] = TEMPLATE_HEADER_WORDS;
  const compactEdgeRows = TEMPLATE_HEADER_WORDS + templates.cellCount + 1;
  const compactEdgeWeights = compactEdgeRows + edgeCount;
  result.set(templates.words.subarray(sourceOffset,
    sourceOffset + templates.cellCount + 1), TEMPLATE_HEADER_WORDS);
  for (let edge = 0; edge < edgeCount; edge += 1) {
    result[compactEdgeRows + edge] = templates.words[edgeRecords + 3 * edge]!;
    result[compactEdgeWeights + edge] = templates.words[edgeRecords + 3 * edge + 2]!;
  }
  result[13] = hierarchyBase;
  result[14] = coarseBase;
  result.set([brickCount, ordered.length, contributionCount, edgeCount], coarseBase);
  result.set(coarseOffsets, coarseBase + 4);
  let contributionAt = contributionBase;
  ordered.forEach(([key, list], coarseEdge) => {
    const record = coarseRecordBase + 3 * coarseEdge;
    result[record] = key % brickCount;
    result[record + 1] = contributionAt;
    result[record + 2] = list.length;
    result.set(list, contributionAt);
    contributionAt += list.length;
  });
  result[hierarchyBase] = hierarchy.length;
  let hierarchyAt = hierarchyBase + 1 + hierarchyDescriptorWords * hierarchy.length;
  let hierarchyDynamicAt = 0;
  hierarchy.forEach((level, index) => {
    const descriptor = hierarchyBase + 1 + hierarchyDescriptorWords * index;
    result[descriptor] = level.groupCount;
    result[descriptor + 9] = hierarchyDynamicAt;
    const append = (slot: number, values: Uint32Array): void => {
      result[descriptor + slot] = hierarchyAt;
      result.set(values, hierarchyAt);
      hierarchyAt += values.length;
    };
    append(1, level.parents);
    append(2, level.childOffsets);
    append(3, level.children);
    append(4, level.internalOffsets);
    append(5, level.internalEdges);
    append(6, level.crossOffsets);
    const crossRecordBase = hierarchyAt;
    result[descriptor + 7] = crossRecordBase;
    hierarchyAt += 3 * level.cross.length;
    const crossContributionBase = hierarchyAt;
    result[descriptor + 8] = crossContributionBase;
    let crossContributionAt = crossContributionBase;
    level.cross.forEach(([pair, list], edge) => {
      const record = crossRecordBase + 3 * edge;
      result[record] = pair % level.groupCount;
      result[record + 1] = crossContributionAt;
      result[record + 2] = list.length;
      result.set(list, crossContributionAt);
      crossContributionAt += list.length;
    });
    hierarchyAt += level.crossContributionCount;
    hierarchyDynamicAt += level.cross.length + 4 * level.groupCount;
  });
  return result;
}

/** Writable pressure dispatch headers/worklists followed by immutable SoA
 * edge payloads shared by transport extrapolation and every pressure SpMV.
 * Neighbor IDs and extrapolation weights are construction-time topology
 * products: frame kernels only apply the current row-acceptance predicate. */
function pressureWorklistAndNeighbors(
  templates: PackedResidentTopologyTemplates,
  pressureTopology: Uint32Array,
  templateLevelCount: number,
  brickCount: number,
): { readonly words: Uint32Array; readonly layout: SparseCM12PressureRepairLayout } {
  const edgeOffsets = templates.words[15]!;
  const edgeCount = templates.words[edgeOffsets + templates.cellCount]!;
  const edgeRecords = edgeOffsets + templates.cellCount + 1;
  const neighborOffset = 8 + templates.cellCount + templates.rowCount;
  const extrapolationWeightOffset = neighborOffset + edgeCount;
  const cellSlotBaseWords = extrapolationWeightOffset + edgeCount;
  const rowSlotBaseWords = cellSlotBaseWords + templates.cellCount;
  const cellChangeBaseWords = rowSlotBaseWords + templates.rowCount;
  const rowChangeBaseWords = cellChangeBaseWords + templates.cellCount;
  const brickStateBaseWords = rowChangeBaseWords + templates.rowCount;
  const rowTopologyStampBaseWords = brickStateBaseWords + brickCount;
  const aggregateEdgeForFineEdgeBaseWords = rowTopologyStampBaseWords + templates.rowCount;
  const coarseBase = pressureTopology[14]!;
  const coarseEdgeCount = pressureTopology[coarseBase + 1]!;
  const aggregateEdgeSourceBaseWords = aggregateEdgeForFineEdgeBaseWords + edgeCount;
  const hierarchyBase = pressureTopology[13]!;
  const hierarchyLevelCount = pressureTopology[hierarchyBase]!;
  const hierarchyEdgeForAggregateBaseWords = Array.from(
    { length: hierarchyLevelCount },
    (_, level) => aggregateEdgeSourceBaseWords + coarseEdgeCount * (level + 1),
  );
  const headerBaseWords = aggregateEdgeSourceBaseWords
    + coarseEdgeCount * (hierarchyLevelCount + 1);
  const layout: SparseCM12PressureRepairLayout = Object.freeze({
    cellSlotBaseWords,
    rowSlotBaseWords,
    cellChangeBaseWords,
    rowChangeBaseWords,
    brickStateBaseWords,
    rowTopologyStampBaseWords,
    aggregateEdgeForFineEdgeBaseWords,
    aggregateEdgeSourceBaseWords,
    hierarchyEdgeForAggregateBaseWords: Object.freeze(hierarchyEdgeForAggregateBaseWords),
    headerBaseWords,
    totalWords: headerBaseWords + SPARSE_CM12_PRESSURE_REPAIR_HEADER_WORDS,
  });
  const result = new Uint32Array(layout.totalWords);
  result.fill(INVALID, cellSlotBaseWords, rowSlotBaseWords + templates.rowCount);
  result.fill(INVALID, brickStateBaseWords, brickStateBaseWords + brickCount);
  result.fill(INVALID, aggregateEdgeForFineEdgeBaseWords,
    aggregateEdgeForFineEdgeBaseWords + edgeCount);
  const cellOffset = templates.words[6]!;
  const cellRangeOffset = templates.words[11]!;
  const rowOffset = templates.words[7]!;
  const termOffset = templates.words[8]!;
  for (let cell = 0; cell < templates.cellCount; cell += 1) {
    const base = cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell;
    const brick = templateCellBrick(templates.words, base);
    const resolution = templateCellResolution(templates.words, base);
    const range = cellRangeOffset + 2 * (templateLevelCount * brick
      + Math.log2(resolution));
    const first = templates.words[range]!, count = templates.words[range + 1]!;
    if (count === 0 || cell < first || cell >= first + count) {
      throw new Error(`Sparse CM12 pressure brick range ${brick}/${resolution} is not dense`);
    }
  }
  for (let edge = 0; edge < edgeCount; edge += 1) {
    const record = edgeRecords + 3 * edge;
    const row = templates.words[record]!;
    const neighbor = templates.words[record + 1]!;
    result[neighborOffset + edge] = neighbor;
    const packedTerms = templates.words[rowOffset + row]!;
    const begin = packedTerms & TEMPLATE_ROW_OFFSET_MASK;
    const end = begin + (packedTerms >>> 28);
    let found = false;
    for (let term = begin; term < end; term += 1) {
      if (templates.words[termOffset + 2 * term] !== neighbor) continue;
      result[extrapolationWeightOffset + edge]
        = templates.words[termOffset + 2 * term + 1]! & 0x7fff_ffff;
      found = true;
      break;
    }
    if (!found) {
      throw new Error(`Sparse CM12 pressure edge ${edge} has no neighbor term`);
    }
  }
  const coarseRecordBase = coarseBase + 4 + brickCount + 1;
  const coarseOffsets = coarseBase + 4;
  for (let brick = 0; brick < brickCount; brick += 1) {
    for (let edge = pressureTopology[coarseOffsets + brick]!;
      edge < pressureTopology[coarseOffsets + brick + 1]!; edge += 1) {
      result[aggregateEdgeSourceBaseWords + edge] = brick;
    }
  }
  for (let level = 0; level < hierarchyLevelCount; level += 1) {
    const destination = hierarchyEdgeForAggregateBaseWords[level]!;
    result.fill(INVALID, destination, destination + coarseEdgeCount);
    const descriptor = hierarchyBase + 1 + 10 * level;
    const groupCount = pressureTopology[descriptor]!;
    const edgeOffsets = pressureTopology[descriptor + 6]!;
    const records = pressureTopology[descriptor + 7]!;
    const hierarchyEdgeCount = pressureTopology[edgeOffsets + groupCount]!;
    for (let edge = 0; edge < hierarchyEdgeCount; edge += 1) {
      const record = records + 3 * edge;
      const first = pressureTopology[record + 1]!;
      const count = pressureTopology[record + 2]!;
      for (let local = 0; local < count; local += 1) {
        const aggregateEdge = pressureTopology[first + local]!;
        if (aggregateEdge >= coarseEdgeCount
          || result[destination + aggregateEdge] !== INVALID) {
          throw new Error(`Sparse CM12 aggregate edge ${aggregateEdge} has ambiguous hierarchy owner`);
        }
        result[destination + aggregateEdge] = edge;
      }
    }
  }
  for (let coarseEdge = 0; coarseEdge < coarseEdgeCount; coarseEdge += 1) {
    const record = coarseRecordBase + 3 * coarseEdge;
    const first = pressureTopology[record + 1]!;
    const count = pressureTopology[record + 2]!;
    for (let local = 0; local < count; local += 1) {
      const fineEdge = pressureTopology[first + local]!;
      if (fineEdge >= edgeCount
        || result[aggregateEdgeForFineEdgeBaseWords + fineEdge] !== INVALID) {
        throw new Error(`Sparse CM12 fine edge ${fineEdge} has ambiguous aggregate owner`);
      }
      result[aggregateEdgeForFineEdgeBaseWords + fineEdge] = coarseEdge;
    }
  }
  return { words: result, layout };
}

/** Static compact topology plus fully device-resident evolving frame state. */
export class WebGPUSparseCM12Resident {
  readonly cellCount: number;
  readonly rowCount: number;
  readonly allocatedBytes: number;
  private readonly parameters: GPUBuffer;
  private readonly topology: GPUBuffer;
  private readonly state: GPUBuffer;
  private readonly partials: GPUBuffer;
  private readonly scalars: GPUBuffer;
  private readonly conditioning: GPUBuffer;
  private readonly activity: GPUBuffer;
  private readonly temporalWorklistLayout: SparseCM12TemporalWorklistLayout;
  private readonly pressureRepairLayout: SparseCM12PressureRepairLayout;
  /** Persistent compact activity census and per-physical-brick 4^3 tile mask. */
  readonly incrementalActivityLayout: SparseCM12IncrementalActivityLayout;
  /** Deterministic GPU pressure membership and rank-select arena. */
  private readonly canonicalMembershipLayout: SparseCM12CanonicalMembershipLayout;
  private readonly framePlanLayout: SparseCM12FramePlanLayout;
  private readonly framePlanPresentationLayout: SparseCM12FramePlanPresentationLayout;
  /** FCA1 GPU-owned frame generation, parity, predicates, and indirect ABI. */
  private readonly frameControlLayout: SparseCM12FrameControlLayout;
  /** Producer-authored exact scalar receipts use a compact activity ingress;
   * the dedicated authority cannot bind or mutate physical state. */
  private readonly scalarResultIngressLayout: SparseCM12SRR1IngressLayout;
  private readonly vexActivityBatchLayout: SparseCM12VexActivityBatchLayout;
  private readonly vexActivityIndirectCopies: ReadonlyMap<string,
    ReturnType<typeof createSparseCM12VexActivityBatchIndirectCopies>[number]>;
  private readonly scalarResultAuthority: WebGPUSparseCM12SRR1RuntimeAdapter;
  /** Immutable construction-time QA mode; never selected by frame state. */
  private readonly pressureRefreshOracleForQA: boolean;
  /** Immutable construction-only HEAD presentation publisher oracle. */
  private readonly presentationPublisherOracleForQA: boolean;
  /** Immutable paired QA mode for proving FCA1 against the retired host
   * parity/D4 schedule. It is not reachable from ordinary solver options. */
  private readonly legacyHostAuthorityOracleForQA: boolean;
  private legacyHostAuthorityParityForQA = 0;
  private legacyHostD4AuthorityForQA: boolean;
  private readonly temporalCellIndirectArguments: GPUBuffer;
  private readonly temporalRowIndirectArguments: GPUBuffer;
  private readonly activityIndirectArguments: GPUBuffer;
  private readonly vexActivityIndirectArguments: GPUBuffer;
  private readonly framePlanIndirectArguments: GPUBuffer;
  private readonly presentationIndirectArguments: GPUBuffer;
  private readonly frameControlIndirectArguments: GPUBuffer;
  private readonly candidateState: GPUBuffer;
  /** Immutable physical templates followed by device-owned double-buffered
   * accepted/shadow cell+row IDs and GPU indirect arguments. */
  private readonly topologyArena: GPUBuffer;
  /** Read-only dispatch snapshot copied from the accepted arena header after
   * each commit. WebGPU forbids one buffer being writable storage and an
   * indirect source in the same compute-pass synchronization scope. */
  private readonly acceptedIndirectArguments: GPUBuffer;
  /** Copy-isolated indirect dispatch for the frame's compact liquid cells. */
  private readonly pressureCellIndirectArguments: GPUBuffer;
  /** Copy-isolated indirect dispatch for active pressure rows. */
  private readonly pressureRowIndirectArguments: GPUBuffer;
  /** GPU-authored bootstrap cell/row dispatches. The inactive epoch branch
   * publishes x=0, so the host encodes one fixed pressure schedule. */
  private readonly pressureMembershipIndirectArguments: GPUBuffer;
  /** Copy-isolated PTR1 indirect families: brick seed/repair/work, row
   * seed/repair/work, and changed-brick accepted-state commit. */
  private readonly pressureTopologyRepairIndirectArguments: GPUBuffer;
  /** Copy-isolated PCF1/PCA1 fine, seed, repair, and work dispatches. */
  private readonly persistentPressureCacheIndirectArguments: GPUBuffer;
  /** Copy-isolated FPA1 preparation/projection bootstrap/repair/work/verify. */
  private readonly faceProjectionAuthorityIndirectArguments: GPUBuffer;
  /** Construction-only observational FPA tile-mask census. */
  private readonly facePreparationTileCensusLayout:
    SparseCM12FacePreparationTileCensusLayout | undefined;
  /** Construction-only actual FPA VEX xyz-read dependency census. */
  private readonly fpaVexReadCensusLayout:
    SparseCM12FpaVexReadCensusLayout | undefined;
  /** Ordinary compact pressure IDs and dense immutable SpMV neighbors. */
  private readonly pressureWorklists: GPUBuffer;
  private readonly fineMetadata: GPUBuffer;
  private readonly fineWorklist: GPUBuffer;
  private readonly fineSamples: GPUBuffer;
  private readonly fineParams: GPUBuffer;
  private readonly fineWorkA: GPUBuffer;
  private readonly fineWorkB: GPUBuffer;
  private readonly fineRollback: GPUBuffer;
  /** Immutable topology duplicate used by pressure SpMVs through a
   * read-only binding instead of the mutable atomic arena. */
  private readonly pressureTemplates: GPUBuffer;
  readonly globalFineLevelSetSource: WebGPUFineLevelSetBrickSource;
  readonly sparseAdaptiveGridSource: SparseAdaptiveGridConsumerSource;
  private readonly diagnosticsReadback: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly pressureBindGroup: GPUBindGroup;
  private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;
  private readonly parameterWords = new ArrayBuffer(SPARSE_CM12_PARAMETER_BYTES);
  private readonly parameterU32 = new Uint32Array(this.parameterWords);
  private readonly parameterF32 = new Float32Array(this.parameterWords);
  private destroyed = false;
  /**
   * Derived here rather than threaded in from `create`, so the region
   * `residentStateLayout` reserved and the region the kernels address are the
   * same arithmetic over the same domain by construction.
   */
  private readonly tracerLattice: SparseCM12TracerLattice;
  /** Built once: a fresh object per read would defeat the consumer's identity
   * check and rebuild a bind group every frame. */
  readonly tracerSource: GPUFluidTracerSource;
  private tracersEnabled = false;
  /**
   * Whether the next encoded advance writes a pressure journal.
   *
   * Separate from having reserved the region: capture is a one-frame act and
   * the film is read on a paused sim. An armed frame splits no pass and adds
   * no barrier, so it computes exactly what an unarmed one computes — but it
   * does add dispatches, so a wall measured with the journal armed must never
   * be compared against one measured without it.
   */
  private journalArmed = false;
  /** Snapshot slots the last armed encode filled; the scrub's upper bound. */
  private journalSnapshotCount = 0;
  /** Set on every off-to-on transition, so enabling re-reads the live liquid. */
  private tracerSeedPending = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly dimensions: readonly [number, number, number],
    private readonly brickFineResolution: SparseAdaptiveMassAtlas["brickFineResolution"],
    private readonly layout: ResidentStateLayout,
    buffers: readonly [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer,
      GPUBuffer, GPUBuffer, GPUBuffer],
    acceptedIndirectArguments: GPUBuffer,
    pressureCellIndirectArguments: GPUBuffer,
    pressureRowIndirectArguments: GPUBuffer,
    pressureMembershipIndirectArguments: GPUBuffer,
    pressureTopologyRepairIndirectArguments: GPUBuffer,
    persistentPressureCacheIndirectArguments: GPUBuffer,
    faceProjectionAuthorityIndirectArguments: GPUBuffer,
    temporalCellIndirectArguments: GPUBuffer,
    temporalRowIndirectArguments: GPUBuffer,
    activityIndirectArguments: GPUBuffer,
    vexActivityIndirectArguments: GPUBuffer,
    framePlanIndirectArguments: GPUBuffer,
    presentationIndirectArguments: GPUBuffer,
    frameControlIndirectArguments: GPUBuffer,
    pressureWorklists: GPUBuffer,
    fineBuffers: readonly [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer,
      GPUBuffer],
    finePlan: FineLevelSetBrickPlan,
    diagnosticsReadback: GPUBuffer,
    bindGroup: GPUBindGroup,
    pressureBindGroup: GPUBindGroup,
    pressureTemplates: GPUBuffer,
    temporalWorklistLayout: SparseCM12TemporalWorklistLayout,
    pressureRepairLayout: SparseCM12PressureRepairLayout,
    incrementalActivityLayout: SparseCM12IncrementalActivityLayout,
    canonicalMembershipLayout: SparseCM12CanonicalMembershipLayout,
    framePlanLayout: SparseCM12FramePlanLayout,
    framePlanPresentationLayout: SparseCM12FramePlanPresentationLayout,
    frameControlLayout: SparseCM12FrameControlLayout,
    scalarResultIngressLayout: SparseCM12SRR1IngressLayout,
    vexActivityBatchLayout: SparseCM12VexActivityBatchLayout,
    private readonly pressureTopologyRepairLayout:
      SparseCM12PressureTopologyRepairLayout,
    private readonly persistentPressureCacheLayout:
      SparseCM12PersistentPressureCacheLayout,
    private readonly faceProjectionAuthorityLayout:
      SparseCM12FaceProjectionAuthorityLayout,
    facePreparationTileCensusLayout:
      SparseCM12FacePreparationTileCensusLayout | undefined,
    fpaVexReadCensusLayout: SparseCM12FpaVexReadCensusLayout | undefined,
    scalarResultAuthority: WebGPUSparseCM12SRR1RuntimeAdapter,
    pressureRefreshOracleForQA: boolean,
    presentationPublisherOracleForQA: boolean,
    legacyHostAuthorityOracleForQA: boolean,
    initialHorizontalD4AuthorityForQA: boolean,
    private readonly pressureAddressingABQA:
      SparseCM12PressureAddressingABQAResources,
    private readonly temporalSeedModeForQA:
      SparseCM12TemporalSeedQAMode | undefined,
    pipelines: Readonly<Record<string, GPUComputePipeline>>,
    cellCount: number,
    rowCount: number,
    private readonly templateCellCount: number,
    private readonly templateRowCount: number,
    private readonly pressureCoarseEdgeCount: number,
    private readonly pressureFineEdgeCount: number,
    private readonly pressureHierarchyGroupCount: number,
    private readonly pressureHierarchyEdgeCount: number,
    private readonly pressureScratchBytes: number,
    private readonly topologyWorklistBaseBytes: number,
    private readonly templateWords: Uint32Array,
    private readonly boundary?: SphericalContainerFineGeometry,
    private readonly rigidCoupling?: WebGPUSparseCM12RigidCoupling,
  ) {
    [this.parameters, this.topology, this.state, this.partials, this.scalars,
      this.conditioning, this.activity, this.candidateState, this.topologyArena] = buffers;
    this.temporalWorklistLayout = temporalWorklistLayout;
    this.pressureRepairLayout = pressureRepairLayout;
    this.incrementalActivityLayout = incrementalActivityLayout;
    this.canonicalMembershipLayout = canonicalMembershipLayout;
    this.framePlanLayout = framePlanLayout;
    this.framePlanPresentationLayout = framePlanPresentationLayout;
    this.frameControlLayout = frameControlLayout;
    this.scalarResultIngressLayout = scalarResultIngressLayout;
    this.vexActivityBatchLayout = vexActivityBatchLayout;
    this.vexActivityIndirectCopies = new Map(
      createSparseCM12VexActivityBatchIndirectCopies(vexActivityBatchLayout)
        .map((copy) => [copy.id, copy] as const),
    );
    this.scalarResultAuthority = scalarResultAuthority;
    this.pressureRefreshOracleForQA = pressureRefreshOracleForQA;
    this.presentationPublisherOracleForQA = presentationPublisherOracleForQA;
    this.legacyHostAuthorityOracleForQA = legacyHostAuthorityOracleForQA;
    this.legacyHostD4AuthorityForQA = initialHorizontalD4AuthorityForQA;
    this.tracerLattice = sparseCM12TracerLattice(dimensions);
    this.tracerSource = {
      buffer: this.state,
      floatOffset: this.layout.tracers,
      count: this.tracerLattice.count,
      latticeDimensions: this.tracerLattice.dimensions,
      originFine: this.tracerLattice.originFine,
      spacingFine: this.tracerLattice.spacingFine,
      domainFine: dimensions,
    };
    this.acceptedIndirectArguments = acceptedIndirectArguments;
    this.pressureCellIndirectArguments = pressureCellIndirectArguments;
    this.pressureRowIndirectArguments = pressureRowIndirectArguments;
    this.pressureMembershipIndirectArguments = pressureMembershipIndirectArguments;
    this.pressureTopologyRepairIndirectArguments =
      pressureTopologyRepairIndirectArguments;
    this.persistentPressureCacheIndirectArguments =
      persistentPressureCacheIndirectArguments;
    this.faceProjectionAuthorityIndirectArguments =
      faceProjectionAuthorityIndirectArguments;
    this.facePreparationTileCensusLayout = facePreparationTileCensusLayout;
    this.fpaVexReadCensusLayout = fpaVexReadCensusLayout;
    this.temporalCellIndirectArguments = temporalCellIndirectArguments;
    this.temporalRowIndirectArguments = temporalRowIndirectArguments;
    this.activityIndirectArguments = activityIndirectArguments;
    this.vexActivityIndirectArguments = vexActivityIndirectArguments;
    this.framePlanIndirectArguments = framePlanIndirectArguments;
    this.presentationIndirectArguments = presentationIndirectArguments;
    this.frameControlIndirectArguments = frameControlIndirectArguments;
    this.pressureWorklists = pressureWorklists;
    [this.fineParams, this.fineMetadata, this.fineWorklist, this.fineSamples,
      this.fineWorkA, this.fineWorkB, this.fineRollback] = fineBuffers;
    this.globalFineLevelSetSource = {
      plan: finePlan,
      generation: 1,
      generationSlot: 0,
      params: this.fineParams,
      metadata: this.fineMetadata,
      worklist: this.fineWorklist,
      samples: this.fineSamples,
      workA: this.fineWorkA,
      workB: this.fineWorkB,
      rollbackSamples: this.fineRollback,
      presentationControl: {
        buffer: this.activity,
        offset: 4 * this.framePlanPresentationLayout.baseWords,
        size: 4 * SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS,
      },
    };
    this.sparseAdaptiveGridSource = {
      kind: "sparse-adaptive-grid-sampling",
      params: { buffer: this.parameters },
      topology: { buffer: this.topology },
      topologyArena: { buffer: this.topologyArena },
      state: { buffer: this.state },
      activity: { buffer: this.activity },
      framePlan: sparseCM12FramePlanSource(
        this.framePlanLayout, this.activity, this.framePlanIndirectArguments),
      fineMetadata: { buffer: this.fineMetadata },
      fineWorklist: { buffer: this.fineWorklist },
      fineSamples: { buffer: this.fineSamples },
    };
    this.diagnosticsReadback = diagnosticsReadback;
    this.bindGroup = bindGroup;
    this.pressureBindGroup = pressureBindGroup;
    this.pressureTemplates = pressureTemplates;
    this.pipelines = pipelines;
    this.cellCount = cellCount;
    this.rowCount = rowCount;
    this.allocatedBytes = [acceptedIndirectArguments, pressureCellIndirectArguments,
      pressureRowIndirectArguments, pressureMembershipIndirectArguments,
      pressureTopologyRepairIndirectArguments,
      temporalCellIndirectArguments, temporalRowIndirectArguments,
      activityIndirectArguments, vexActivityIndirectArguments,
      framePlanIndirectArguments, presentationIndirectArguments,
      frameControlIndirectArguments,
      pressureTemplates, pressureWorklists,
      ...buffers, ...fineBuffers].reduce(
      (sum, buffer) => sum + buffer.size, 0,
    )
      + diagnosticsReadback.size + scalarResultAuthority.allocatedBytes
      + (pressureAddressingABQA ? pressureAddressingABQA.indirectArguments.size
        + (pressureAddressingABQA.receiptReadback?.size ?? 0)
        + (pressureAddressingABQA.queryResolve?.size ?? 0)
        + (pressureAddressingABQA.timestampReadback?.size ?? 0) : 0);
  }

  static create(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = 16,
  ): Promise<WebGPUSparseCM12Resident> {
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      false, false, false, false);
  }

  /** QA-only construction path. Production callers have no runtime option
   * that can select this full pressure-classification oracle. */
  static createPressureRefreshOracleForQA(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = 16,
  ): Promise<WebGPUSparseCM12Resident> {
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      true, false, false, false);
  }

  /** QA-only construction path for the immutable HEAD presentation publisher.
   * Production frame state cannot select it and FPP1 never falls back to it. */
  static createPresentationPublisherOracleForQA(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = 16,
  ): Promise<WebGPUSparseCM12Resident> {
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      false, true, false, false);
  }

  /** Construction-only paired oracle for the retired host authority schedule.
   * It has no runtime selector and intentionally rejects rigid/boundary cases
   * until their legacy direct-dispatch comparison is required. */
  static createLegacyHostAuthorityOracleForQA(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = 16,
  ): Promise<WebGPUSparseCM12Resident> {
    if (rigid || atlas.boundary) {
      throw new Error("legacy host-authority QA oracle currently requires no rigid/boundary source");
    }
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      false, false, true, true);
  }

  /** Construction-only full scalar oracle. It invalidates every clean SAW1
   * receipt while running the same compiled physical mass/gamma/surface path. */
  static createScalarFullPathOracleForQA(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = 16,
  ): Promise<WebGPUSparseCM12Resident> {
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      false, false, false, true);
  }

  /** Construction-only PCM rank-select pressure-address arm. */
  static createPressureAddressingRankSelectForQA(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = 16,
  ): Promise<WebGPUSparseCM12Resident> {
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      false, false, false, false, "canonicalRankSelect");
  }

  /** Construction-only materialized pressure-address arm. */
  static createPressureAddressingMaterializedListForQA(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = 16,
  ): Promise<WebGPUSparseCM12Resident> {
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      false, false, false, false, "materializedList");
  }

  /** Construction-only production-seed arm with QA census enabled. */
  static createTemporalCurrentSeedOracleForQA(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = 16,
  ): Promise<WebGPUSparseCM12Resident> {
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      false, false, false, false, undefined, "current");
  }

  /** Construction-only exact endpoint-change seed arm with QA census enabled. */
  static createTemporalChangeSeedOracleForQA(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = 16,
  ): Promise<WebGPUSparseCM12Resident> {
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      false, false, false, false, undefined, "change");
  }

  private static async createConfigured(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = 16,
    pressureRefreshOracleForQA = false,
    presentationPublisherOracleForQA = false,
    legacyHostAuthorityOracleForQA = false,
    scalarFullPathOracleForQA = false,
    pressureAddressingModeForQA?: SparseCM12PressureAddressingABModeName,
    temporalSeedModeForQA?: SparseCM12TemporalSeedQAMode,
  ): Promise<WebGPUSparseCM12Resident> {
    if (atlas.brickFineResolution !== 16 || presentationPageResolution !== 16) {
      throw new Error("Sparse CM12 production requires B16/P16 while pressure addressing uses its canonical materialized list");
    }
    if (atlas.boundary) {
      throw new Error("Sparse CM12 sparse MGPCG does not support separating boundaries; no fallback solver is installed");
    }
    // Macro-bricks are immutable, but their presence must not disable
    // transitions for every ordinary span-one surface/receiver brick in the
    // same hydrostatic scene. Prepack variants for those ordinary bricks and
    // retain each macro only at its accepted resolution.
    const mutableBrickKeysForBudget = atlas.bricks.filter((brick) =>
      sparseBrickSpan(brick) === 1)
      .map((brick) => brick.key);
    const hostTemplateVariants = sparseCM12HostTemplateVariantsEnabled(
      grid.cells.length, grid.gradientRows.length, mutableBrickKeysForBudget.length,
      atlas.brickFineResolution,
    );
    // The compatibility library must be topology-complete: forcing only a
    // subset through its low rungs can violate 2:1 against an immutable 4/8
    // neighbour. Bound the whole mutable set up front; larger scenes retain
    // their accepted physical rungs and use GPU topology pages for later work.
    const mutableBrickKeys: ReadonlySet<number> = hostTemplateVariants
      ? new Set(mutableBrickKeysForBudget)
      : new Set<number>();
    const packed = packResidentTopology(atlas, grid, mutableBrickKeys);
    const templates = hostTemplateVariants
      ? packResidentTopologyTemplates(atlas, grid)
      : packAcceptedTopologyTemplates(atlas, grid);
    if (hostTemplateVariants) {
      const cellOffset = templates.words[6]!, rangeOffset = templates.words[11]!;
      const templateFloats = new Float32Array(templates.words.buffer,
        templates.words.byteOffset, templates.words.length);
      for (let brick = 0; brick < atlas.bricks.length; brick += 1) {
        if (!mutableBrickKeys.has(atlas.bricks[brick]!.key)) continue;
        const templateLevels = sparseCM12TemplateLevels(atlas.brickFineResolution);
        for (let level = 0; level < templateLevels.length; level += 1) {
          const resolution = templateLevels[level]!;
          const range = rangeOffset + 2 * (templateLevels.length * brick + level);
          const first = templates.words[range]!, count = templates.words[range + 1]!;
          const expected = sparseCM12BrickLiveCellCount(atlas,
            atlas.bricks[brick]!, resolution);
          if (count !== expected || count === 0) {
            throw new Error(`Sparse CM12 template range ${brick}/${resolution} has ${count} `
              + `cells; expected clipped count ${expected}`);
          }
          for (let cell = first; cell < first + count; cell += 1) {
            const base = cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell;
            const coordinate = atlas.bricks[brick]!.coordinate;
            const lower = [0, 1, 2].map((axis) => Math.round(
              templateFloats[base + axis]! - 0.5 * templateFloats[base + 4 + axis]!,
            ));
            if (templateCellResolution(templates.words, base) !== resolution
              || templateCellBrick(templates.words, base) !== brick
              || lower[0]! < atlas.brickFineResolution * coordinate[0]
              || lower[0]! >= atlas.brickFineResolution * (coordinate[0] + 1)
              || lower[1]! < atlas.brickFineResolution * coordinate[1]
              || lower[1]! >= atlas.brickFineResolution * (coordinate[1] + 1)
              || lower[2]! < atlas.brickFineResolution * coordinate[2]
              || lower[2]! >= atlas.brickFineResolution * (coordinate[2] + 1)) {
              throw new Error(`Sparse CM12 template range ${brick}/${resolution} aliases cell ${cell}`);
            }
          }
        }
      }
    }
    const fine = sparseCM12FinePresentationPlan(atlas, presentationPageResolution);
    (fine.plan as { finestCellWidth: number; fineCellWidth: number }).finestCellWidth =
      finestCellSize_m;
    (fine.plan as { finestCellWidth: number; fineCellWidth: number }).fineCellWidth =
      finestCellSize_m;
    const tracerLattice = sparseCM12TracerLattice(atlas.dimensions);
    const layout = residentStateLayout(
      templates.cellCount, templates.rowCount, Boolean(rigid),
      tracerLattice.count,
      journal?.iterationCapacity
        ? {
          // One more than the ceiling: record 0 is the seed, before any
          // iteration has run, and the film starts there.
          iterationCapacity: journal.iterationCapacity + 1,
          snapshotCapacity: journal.snapshotCapacity
            ?? SPARSE_CM12_PRESSURE_JOURNAL_SNAPSHOTS,
        }
        : {},
    );
    const horizontalD4Authority = sparseAtlasScalarsHaveHorizontalD4Symmetry(
      grid,
      Float64Array.from(grid.cells, (cell) => cell.density),
      Float64Array.from(grid.cells, (cell) => cell.gamma),
    );
    const cellWorkgroups = Math.ceil(templates.cellCount / WORKGROUP_SIZE);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const parameters = device.createBuffer({ label: "Sparse CM12 resident parameters",
      size: SPARSE_CM12_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const topology = uploadBuffer(device, "Sparse CM12 resident topology", packed.words, storage);
    // WebGPU buffers start zeroed. Seed only the four nonzero scalar ranges;
    // PCM bootstrap authors persistent liquid membership for accepted cells.
    // instead of allocating and uploading the complete (mostly-zero) resident
    // state on the host. Rigid coupling retains its explicit initialization
    // until its two strided row flags move to the bootstrap kernel.
    let state: GPUBuffer;
    if (rigid) {
      const initialState = new Float32Array(layout.floatCount);
      for (let cell = 0; cell < templates.cellCount; cell += 1) {
        const density = templates.initialDensity[cell]!, gamma = templates.initialGamma[cell]!;
        initialState[layout.densityA + cell] = density;
        initialState[layout.densityB + cell] = density;
        initialState[layout.gammaA + cell] = gamma;
        initialState[layout.gammaB + cell] = gamma;
        initialState[layout.solidCellOpen + cell] = 1;
      }
      for (let row = 0; row < templates.rowCount; row += 1) {
        initialState[layout.solidRowData + 3 * row] = 1;
        initialState[layout.solidRowData + 3 * row + 2] = 1;
      }
      state = uploadBuffer(device, "Sparse CM12 resident state", initialState, storage);
    } else {
      state = device.createBuffer({ label: "Sparse CM12 resident state",
        size: Math.max(4, 4 * layout.floatCount), usage: storage });
      const seed = (floatOffset: number, values: Float32Array) => {
        if (values.byteLength === 0) return;
        device.queue.writeBuffer(state, 4 * floatOffset, values.buffer as ArrayBuffer,
          values.byteOffset, values.byteLength);
      };
      seed(layout.densityA, templates.initialDensity);
      seed(layout.densityB, templates.initialDensity);
      seed(layout.gammaA, templates.initialGamma);
      seed(layout.gammaB, templates.initialGamma);
    }
    const partials = device.createBuffer({ label: "Sparse CM12 resident reductions",
      size: Math.max(16, 16 * cellWorkgroups), usage: storage });
    const scalars = device.createBuffer({ label: "Sparse CM12 resident scalar reductions",
      size: SPARSE_CM12_PRESSURE_SCALAR_BYTES, usage: storage });
    const conditioning = device.createBuffer({
      label: "Sparse CM12 conservative transport and conditioning accumulators",
      // Transport needs four cell planes. Pressure reuses the dead arena for
      // cell and row headers plus their stable-ID compact worklists.
      size: 4 * Math.max(4 * templates.cellCount,
        templates.cellCount + templates.rowCount + 8),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const activityHistoryWords = ACTIVITY_HEADER_WORDS
      + ACTIVITY_RECORD_WORDS * packed.brickCount;
    const logicalBrickDimensions = atlas.dimensions.map((value) =>
      Math.ceil(value / atlas.brickFineResolution));
    const logicalBrickCount = logicalBrickDimensions[0]!
      * logicalBrickDimensions[1]! * logicalBrickDimensions[2]!;
    const tilesPerLogicalBrickAxis = atlas.brickFineResolution / 4;
    const stableDirtyTileCount = logicalBrickCount * tilesPerLogicalBrickAxis ** 3;
    const activityAlignmentWords = 64;
    const temporalHeaderBaseWords = Math.ceil(
      activityHistoryWords / activityAlignmentWords,
    ) * activityAlignmentWords;
    const temporalWorklistLayout: SparseCM12TemporalWorklistLayout = Object.freeze({
      headerBaseWords: temporalHeaderBaseWords,
      cellListBaseWords: temporalHeaderBaseWords + 10,
      rowListBaseWords: temporalHeaderBaseWords + 10 + templates.cellCount,
      cellFlagABaseWords: temporalHeaderBaseWords + 10
        + templates.cellCount + templates.rowCount,
      cellFlagBBaseWords: temporalHeaderBaseWords + 10
        + 2 * templates.cellCount + templates.rowCount,
      totalWords: temporalHeaderBaseWords + 10
        + 3 * templates.cellCount + templates.rowCount,
    });
    const incrementalActivityLayout = createSparseCM12IncrementalActivityLayout({
      baseWords: temporalWorklistLayout.totalWords,
      stableTileCount: stableDirtyTileCount,
      brickCount: packed.brickCount,
      alignmentWords: activityAlignmentWords,
    });
    const canonicalMembershipLayout = createSparseCM12CanonicalMembershipLayout({
      baseWords: incrementalActivityLayout.totalWords,
      cellCapacity: templates.cellCount,
      rowCapacity: templates.rowCount,
    });
    const framePlanLayout = createSparseCM12FramePlanLayout({
      baseWords: Math.ceil(canonicalMembershipLayout.totalWords
        / activityAlignmentWords)
        * activityAlignmentWords,
      brickCapacity: packed.brickCount,
      brickFineResolution: atlas.brickFineResolution,
      packetCount: 6,
    });
    if (presentationPageResolution !== atlas.brickFineResolution) {
      throw new Error("FPL1/FPP1 resident cutover requires one B-sized presentation page");
    }
    const pageCount = fine.metadata.length / FINE_LEVELSET_METADATA_WORDS;
    if (pageCount !== packed.brickCount) {
      throw new Error(`FPP1 requires one page per brick; found ${pageCount}/${packed.brickCount}`);
    }
    const brickPages = new Uint32Array(packed.brickCount).fill(INVALID);
    for (let page = 0; page < pageCount; page += 1) {
      const source = fine.metadata[FINE_LEVELSET_METADATA_WORDS * page + 3]!;
      const brick = (source >>> 3) & 0x00ff_ffff;
      if (brick >= packed.brickCount || brickPages[brick] !== INVALID) {
        throw new Error(`FPP1 invalid or duplicate brick/page mapping ${brick}/${page}`);
      }
      brickPages[brick] = page;
    }
    if (brickPages.some((page) => page === INVALID)) {
      throw new Error("FPP1 physical brick/page mapping is incomplete");
    }
    const framePlanPresentationLayout = createSparseCM12FramePlanPresentationLayout({
      baseWords: framePlanLayout.totalWords,
      pageCapacity: packed.brickCount,
      brickFineResolution: atlas.brickFineResolution,
      pageResolution: presentationPageResolution,
      packetIndex: 5,
    });
    const scalarResultIngressLayout = createSparseCM12SRR1IngressLayout({
      baseWords: framePlanPresentationLayout.totalWords,
      tileCapacity: stableDirtyTileCount,
    });
    const scalarResultPlan = createSparseCM12SRR1RuntimePlan({
      baseWords: framePlanPresentationLayout.totalWords,
      tileCapacity: stableDirtyTileCount,
      constructionMode: scalarFullPathOracleForQA
        ? "immutable-full-oracle" : "temporal",
    });
    if (scalarResultPlan.ingressLayout.totalWords !== scalarResultIngressLayout.totalWords) {
      throw new Error("SIR1 resident/runtime layout mismatch");
    }
    const vexActivityBatchLayout = createSparseCM12VexActivityBatchLayout({
      activityTailWords: scalarResultIngressLayout.totalWords,
      // residentStateLayout appends VEX1 immediately after the journal.
      stateTailFloats: layout.velocityExtensionAcceptedVelocity,
      cellCapacity: templates.cellCount,
    });
    if (vexActivityBatchLayout.velocityState.acceptedVelocityFloatBase
        !== layout.velocityExtensionAcceptedVelocity
      || vexActivityBatchLayout.totalStateFloats !== layout.floatCount) {
      throw new Error("VEX1 resident activity/state composition mismatch");
    }
    const pressureAddressingMode = pressureAddressingModeForQA ?? "materializedList";
    const pressureAddressingABLayout = pressureAddressingModeForQA === undefined
      ? createSparseCM12ProductionPressureAddressingLayout({
        baseWords: vexActivityBatchLayout.totalActivityWords,
        cellCapacity: templates.cellCount,
        brickFineResolution: 16,
        presentationPageResolution: 16,
      }) : createSparseCM12PressureAddressingABLayout({
        baseWords: vexActivityBatchLayout.totalActivityWords,
        cellCapacity: templates.cellCount,
        brickFineResolution: 16,
        presentationPageResolution: 16,
        constructionMode: "qa-pressure-addressing-ab",
    });
    const initialActivity = new Uint32Array(pressureAddressingABLayout.totalWords);
    initialActivity.set(createSparseCM12IncrementalActivityInitialWords(
      incrementalActivityLayout,
    ), incrementalActivityLayout.headerBaseWords);
    initializeSparseCM12CanonicalMembershipWords(initialActivity,
      canonicalMembershipLayout);
    initialActivity.set(createSparseCM12FramePlanInitialWords(framePlanLayout),
      framePlanLayout.baseWords);
    initialActivity.set(createSparseCM12FramePlanPresentationInitialWords(
      framePlanPresentationLayout, { brickPages: Array.from(brickPages) }),
    framePlanPresentationLayout.baseWords);
    initializeSparseCM12SRR1IngressWords(initialActivity,
      scalarResultIngressLayout, scalarResultPlan.constructionMode);
    initialActivity.set(createSparseCM12VexActivityBatchInitialWords(
      vexActivityBatchLayout,
    ), vexActivityBatchLayout.activityBaseWords);
    initialActivity.set(createSparseCM12PressureAddressingABInitialWords(
      pressureAddressingABLayout), pressureAddressingABLayout.baseWords);
    initialActivity[8] = initiallyActiveBrickKeys.size;
    initialActivity[12] = 1;
    for (let brick = 0; brick < packed.brickCount; brick += 1) {
      const at = ACTIVITY_HEADER_WORDS + ACTIVITY_RECORD_WORDS * brick;
      initialActivity[at + 8] = atlas.bricks[brick]!.resolution;
      initialActivity[at + 9] = 32; // retained until the first GPU topology epoch
      initialActivity[at + 10] = initiallyActiveBrickKeys.has(atlas.bricks[brick]!.key)
        ? 1 : 0;
      initialActivity[at + 12] = atlas.bricks[brick]!.resolution;
      initialActivity[at + 13] = atlas.bricks[brick]!.resolution;
      initialActivity[at + 37] = INVALID;
      // Low five bits retain the coarsest calm level accepted before this
      // brick's first promotion; bit 31 is latched by that promotion.
      initialActivity[at + 38] = atlas.bricks[brick]!.resolution;
      if (initialActivity[at + 10] !== 0) {
        if (atlas.bricks[brick]!.resolution === atlas.brickFineResolution) {
          initialActivity[19] += 1;
        }
        else initialActivity[20] += 1;
      }
      if (initialActivity[at + 10] !== 0) {
        const level = Math.log2(atlas.bricks[brick]!.resolution);
        const range = templates.words[11]! + 2
          * (sparseCM12TemplateLevels(atlas.brickFineResolution).length * brick + level);
        initialActivity[11] += templates.words[range + 1]!;
      }
    }
    const activity = uploadBuffer(device, "Sparse CM12 resident activity history",
      initialActivity, storage);
    const pressureEdgeOffset = templates.words[15]!;
    const pressureEdgeCount = templates.words[pressureEdgeOffset + templates.cellCount]!;
    const pressureTopology = compactPressureTopology(templates, atlas);
    const pressureWorklistData = pressureWorklistAndNeighbors(
      templates, pressureTopology,
      sparseCM12TemplateLevels(atlas.brickFineResolution).length,
      packed.brickCount,
    );
    const pressureCoarseBase = pressureTopology[14]!;
    const pressureCoarseEdgeCount = pressureTopology[pressureCoarseBase + 1]!;
    const pressureHierarchyBase = pressureTopology[13]!;
    const pressureHierarchyGroupCounts = Array.from(
      { length: pressureTopology[pressureHierarchyBase]! },
      (_, level) => pressureTopology[pressureHierarchyBase + 1 + 10 * level]!,
    );
    const pressureHierarchyEdgeCounts = pressureHierarchyGroupCounts.map((groupCount, level) => {
      const descriptor = pressureHierarchyBase + 1 + 10 * level;
      const crossOffsets = pressureTopology[descriptor + 6]!;
      return pressureTopology[crossOffsets + groupCount]!;
    });
    if (templates.cellCount >= 0x1fff_ffff) {
      throw new Error("Sparse CM12 pressure brick range cache exhausts its 29-bit cell base");
    }
    const pressureScratchBytes = 4 * (pressureEdgeCount + pressureCoarseEdgeCount
      + 5 * packed.brickCount
      + pressureHierarchyGroupCounts.reduce((sum, count, level) =>
        sum + 4 * count + pressureHierarchyEdgeCounts[level]!, 0)
      + templates.cellCount);
    const candidateState = device.createBuffer({
      label: "Sparse CM12 isolated candidate cell fields",
      // Candidate transfer begins after projection, so the same storage first
      // carries the pressure epoch's baked effective edge coefficients.
      size: Math.max(4, pressureScratchBytes,
        4 * candidateFloatsPerBrick(atlas.brickFineResolution) * packed.candidateBrickCount),
      usage: storage,
    });
    const topologyPagePool = sparseCM12TopologyPagePoolPlan(
      initiallyActiveBrickKeys.size, !hostTemplateVariants, atlas.brickFineResolution,
    );
    const worklistHeaderWords = 32;
    const cellList0 = worklistHeaderWords;
    const cellList1 = cellList0 + templates.cellCount;
    const rowList0 = cellList1 + templates.cellCount;
    const rowList1 = rowList0 + templates.rowCount;
    const pageFreeList = rowList1 + templates.rowCount;
    const pageDescriptors = pageFreeList + topologyPagePool.freeListWords;
    const initialWorklists = new Uint32Array(
      pageDescriptors + topologyPagePool.descriptorWords,
    );
    initialWorklists.set([1, 1, 0, 0, templates.initialCellWorklist.length,
      templates.initialRowWorklist.length, templates.cellCount, templates.rowCount,
      Math.ceil(templates.initialCellWorklist.length / WORKGROUP_SIZE), 1, 1,
      Math.ceil(templates.initialRowWorklist.length / WORKGROUP_SIZE), 1, 1,
      cellList0, cellList1, rowList0, rowList1], 0);
    initialWorklists.set([templates.initialCellWorklist.length,
      templates.initialRowWorklist.length,
      Math.ceil(templates.initialCellWorklist.length / WORKGROUP_SIZE), 1, 1,
      Math.ceil(templates.initialRowWorklist.length / WORKGROUP_SIZE), 1, 1], 18);
    // Device-side LIFO page allocator. Header indices 26..29 are deliberately
    // outside the accepted indirect-dispatch ABI at 8..25.
    initialWorklists[26] = topologyPagePool.pageCapacity;
    initialWorklists[27] = topologyPagePool.pageCapacity;
    initialWorklists[28] = pageFreeList;
    initialWorklists[29] = 0;
    initialWorklists[30] = pageDescriptors;
    initialWorklists[31] = topologyPagePool.pageWords;
    initialWorklists.set(templates.initialCellWorklist, cellList0);
    initialWorklists.set(templates.initialCellWorklist, cellList1);
    initialWorklists.set(templates.initialRowWorklist, rowList0);
    initialWorklists.set(templates.initialRowWorklist, rowList1);
    for (let page = 0; page < topologyPagePool.pageCapacity; page += 1) {
      initialWorklists[pageFreeList + page] = page;
    }
    // One binding keeps the resident shader within WebGPU's portable ten
    // storage-buffer limit. Upload its immutable head and mutable tail
    // separately: materializing their concatenation briefly doubled the
    // largest host allocation during scene loading.
    // The pressure CSR tail has already been transformed into the two compact
    // pressure buffers above. No ordinary/topology pass reads it, so do not
    // upload that construction-only duplicate into the mutable template arena.
    const physicalTemplateWordCount = pressureEdgeOffset;
    const physicalTemplateBytes = 4 * physicalTemplateWordCount;
    templates.words[14] = physicalTemplateWordCount;
    const frameControlBaseWords = Math.ceil(
      (physicalTemplateWordCount + initialWorklists.length) / 64,
    ) * 64;
    const topologyMutableEndWords = physicalTemplateWordCount
      + pageDescriptors + topologyPagePool.descriptorWords;
    if (topologyMutableEndWords !== physicalTemplateWordCount + initialWorklists.length
      || frameControlBaseWords < topologyMutableEndWords) {
      throw new Error("FCA1 overlaps the mutable topology worklist/page arena");
    }
    const frameControl = createSparseCM12FrameControl({
      baseWords: frameControlBaseWords,
      cellWorkgroups,
      rowWorkgroups: Math.ceil(templates.rowCount / WORKGROUP_SIZE),
      bodyCapacity: rigid ? 12 : 0,
      d4Capable: true,
      rigidCapable: Boolean(rigid),
      boundaryCapable: Boolean(atlas.boundary),
      initialScalarD4Authority: horizontalD4Authority,
      initialFaceD4Authority: horizontalD4Authority,
    });
    const pressureTopologyRepairLayout =
      createSparseCM12PressureTopologyRepairLayout({
        baseWords: frameControl.layout.totalWords,
        brickCapacity: packed.brickCount,
        rowCapacity: templates.rowCount,
        brickFineResolution: 16,
        presentationPageResolution: 16,
      });
    const pressureTopologyRepairWords =
      createSparseCM12PressureTopologyRepairInitialWords(
        pressureTopologyRepairLayout,
      );
    for (let brick = 0; brick < packed.brickCount; brick += 1) {
      pressureTopologyRepairWords[
        pressureTopologyRepairLayout.brickAcceptedStateBaseWords + brick
      ] = atlas.bricks[brick]!.resolution
        | (initiallyActiveBrickKeys.has(atlas.bricks[brick]!.key) ? 0x8000_0000 : 0);
    }
    const persistentPressureCacheLayout =
      createSparseCM12ResidentPersistentPressureCacheLayout({
        baseWords: pressureTopologyRepairLayout.totalWords,
        cellCount: templates.cellCount,
        rowCount: templates.rowCount,
        directedEdgeCount: pressureEdgeCount,
        brickCount: packed.brickCount,
        aggregateEdgeCount: pressureCoarseEdgeCount,
        hierarchyLevelCounts: pressureHierarchyGroupCounts,
        hierarchyEdgeLevelCounts: pressureHierarchyEdgeCounts,
        rigidScaleEnabled: Boolean(rigid),
        // Immutable QA arm rebuilds every fine coefficient through the same
        // stable-ID arithmetic. Production remains local-only; the paired arm
        // exposes a missed PCM/topology coefficient event byte-for-byte.
        qaFullOracle: false,
      });
    const persistentPressureCacheWords = new Uint32Array(
      persistentPressureCacheLayout.bufferSizeWords,
    );
    initializeSparseCM12PersistentPressureCacheWords(
      persistentPressureCacheWords, persistentPressureCacheLayout,
    );
    // Reproduce the legacy hierarchy bootstrap byte-for-byte for groups that
    // remain outside the first local repair. The old full bake sums the
    // max(0, 1e-12) diagonal of every child brick with the shader's 64-lane
    // reduction order; a flat 1e-12 node initializer loses (childCount-1)
    // floors and perturbs active coarse solves that reference an inactive
    // neighboring group. Internal aggregate edges are construction-zero.
    const pressureDiagonalFloor = Math.fround(1e-12);
    const pressureDiagonalBits = (value: number) =>
      new Uint32Array(new Float32Array([value]).buffer)[0]!;
    pressureHierarchyGroupCounts.forEach((groupCount, level) => {
      const descriptor = pressureHierarchyBase + 1 + 10 * level;
      const childOffsets = pressureTopology[descriptor + 2]!;
      for (let group = 0; group < groupCount; group += 1) {
        const begin = pressureTopology[childOffsets + group]!;
        const end = pressureTopology[childOffsets + group + 1]!;
        const lanes = new Float32Array(WORKGROUP_SIZE);
        for (let lane = 0; lane < WORKGROUP_SIZE; lane += 1) {
          let sum = Math.fround(0);
          for (let at = begin + lane; at < end; at += WORKGROUP_SIZE) {
            sum = Math.fround(sum + pressureDiagonalFloor);
          }
          lanes[lane] = sum;
        }
        for (let width = WORKGROUP_SIZE >>> 1; width >= 1; width >>>= 1) {
          for (let lane = 0; lane < width; lane += 1) {
            lanes[lane] = Math.fround(lanes[lane]! + lanes[lane + width]!);
          }
        }
        persistentPressureCacheWords[
          persistentPressureCacheLayout.hierarchyDiagonalBaseWords[level]! + group
        ] = pressureDiagonalBits(Math.max(lanes[0]!, pressureDiagonalFloor));
      }
    });
    const faceProjectionAuthorityLayout = createSparseCM12FaceProjectionAuthorityLayout({
      baseWords: persistentPressureCacheLayout.bufferSizeWords,
      rowCapacity: templates.rowCount, cellCapacity: templates.cellCount,
      // Construction-only specialization; production has no selector or
      // full-work fallback. The paired runner compares this arm byte-for-byte.
      qaFullOracle: pressureRefreshOracleForQA,
    });
    const faceProjectionAuthorityWords =
      createSparseCM12FaceProjectionAuthorityInitialWords(faceProjectionAuthorityLayout);
    // The census exists only in the two immutable temporal QA constructors.
    // Production retains the exact FPA arena extent and dispatch sequence.
    const facePreparationTileCensusLayout = temporalSeedModeForQA === undefined
      ? undefined : createSparseCM12FacePreparationTileCensusLayout({
        baseWords: faceProjectionAuthorityLayout.totalWords,
        rowCapacity: templates.rowCount,
      });
    const facePreparationTileCensusWords = facePreparationTileCensusLayout
      ? createSparseCM12FacePreparationTileCensusInitialWords(
        facePreparationTileCensusLayout) : undefined;
    const fpaVexReadCensusLayout = temporalSeedModeForQA === undefined
      ? undefined : createSparseCM12FpaVexReadCensusLayout({
        baseWords: facePreparationTileCensusLayout!.totalWords,
        cellCapacity: templates.cellCount, rowCapacity: templates.rowCount,
        tileCapacity: stableDirtyTileCount,
      });
    const fpaVexReadCensusWords = fpaVexReadCensusLayout
      ? createSparseCM12FpaVexReadCensusInitialWords(fpaVexReadCensusLayout)
      : undefined;
    const topologyArena = device.createBuffer({
      label: "Sparse CM12 physical topology templates and worklists",
      size: Math.max(4, fpaVexReadCensusLayout?.totalBytes
        ?? facePreparationTileCensusLayout?.totalBytes
        ?? faceProjectionAuthorityLayout.totalBytes),
      usage: storage,
    });
    device.queue.writeBuffer(topologyArena, 0, templates.words.buffer as ArrayBuffer,
      templates.words.byteOffset, physicalTemplateBytes);
    device.queue.writeBuffer(topologyArena, physicalTemplateBytes,
      initialWorklists.buffer as ArrayBuffer, initialWorklists.byteOffset,
      initialWorklists.byteLength);
    const frameControlWords = frameControl.words.subarray(frameControl.layout.baseWords);
    device.queue.writeBuffer(topologyArena, 4 * frameControl.layout.baseWords,
      frameControlWords.buffer as ArrayBuffer, frameControlWords.byteOffset,
      frameControlWords.byteLength);
    const pressureTopologyRepairRegion = pressureTopologyRepairWords.subarray(
      pressureTopologyRepairLayout.baseWords,
    );
    device.queue.writeBuffer(topologyArena,
      4 * pressureTopologyRepairLayout.baseWords,
      pressureTopologyRepairRegion.buffer as ArrayBuffer,
      pressureTopologyRepairRegion.byteOffset,
      pressureTopologyRepairRegion.byteLength);
    const persistentPressureCacheRegion = persistentPressureCacheWords.subarray(
      persistentPressureCacheLayout.thetaBaseWords,
    );
    device.queue.writeBuffer(topologyArena,
      4 * persistentPressureCacheLayout.thetaBaseWords,
      persistentPressureCacheRegion.buffer as ArrayBuffer,
      persistentPressureCacheRegion.byteOffset,
      persistentPressureCacheRegion.byteLength);
    const faceProjectionAuthorityRegion = faceProjectionAuthorityWords.subarray(
      faceProjectionAuthorityLayout.baseWords,
    );
    device.queue.writeBuffer(topologyArena,
      4 * faceProjectionAuthorityLayout.baseWords,
      faceProjectionAuthorityRegion.buffer as ArrayBuffer,
      faceProjectionAuthorityRegion.byteOffset,
      faceProjectionAuthorityRegion.byteLength);
    if (facePreparationTileCensusLayout && facePreparationTileCensusWords) {
      device.queue.writeBuffer(topologyArena,
        4 * facePreparationTileCensusLayout.baseWords,
        facePreparationTileCensusWords.buffer as ArrayBuffer,
        facePreparationTileCensusWords.byteOffset,
        facePreparationTileCensusWords.byteLength);
    }
    if (fpaVexReadCensusLayout && fpaVexReadCensusWords) {
      device.queue.writeBuffer(topologyArena, 4 * fpaVexReadCensusLayout.baseWords,
        fpaVexReadCensusWords.buffer as ArrayBuffer,
        fpaVexReadCensusWords.byteOffset, fpaVexReadCensusWords.byteLength);
    }
    const pressureTemplates = uploadBuffer(device,
      "Sparse CM12 read-only pressure topology", pressureTopology,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const pressureWorklists = uploadBuffer(device,
      "Sparse CM12 ordinary pressure worklists and dense neighbors",
      pressureWorklistData.words,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const acceptedIndirectArguments = uploadBuffer(device,
      "Sparse CM12 accepted indirect dispatch snapshot",
      initialWorklists.subarray(8, 14),
      GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const pressureCellIndirectArguments = device.createBuffer({
      label: "Sparse CM12 pressure-cell indirect dispatch",
      size: 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const pressureRowIndirectArguments = device.createBuffer({
      label: "Sparse CM12 pressure-row indirect dispatch",
      size: 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const pressureMembershipIndirectArguments = device.createBuffer({
      label: "Sparse CM12 pressure-membership bootstrap indirect dispatches",
      size: SPARSE_CM12_PRESSURE_MEMBERSHIP_INDIRECT_BYTES,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const pressureTopologyRepairIndirectArguments = device.createBuffer({
      label: "Sparse CM12 bounded pressure-topology repair indirect dispatches",
      size: 7 * 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const persistentPressureCacheIndirectArguments = device.createBuffer({
      label: "Sparse CM12 persistent pressure-cache indirect dispatches",
      // Fine repair, then seed/repair/work for four aggregate families.
      size: 13 * 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const faceProjectionAuthorityIndirectArguments = device.createBuffer({
      label: "Sparse CM12 face/projection authority indirect dispatches",
      // Preparation/projection bootstrap, repair, work, and verify triplets.
      size: 10 * 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const temporalCellIndirectArguments = device.createBuffer({
      label: "Sparse CM12 temporal scalar-cell indirect dispatch",
      size: 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const temporalRowIndirectArguments = device.createBuffer({
      label: "Sparse CM12 temporal scalar-row indirect dispatch",
      size: 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const activityIndirectArguments = device.createBuffer({
      label: "Sparse CM12 incremental activity indirect dispatch",
      size: 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const vexActivityIndirectArguments = device.createBuffer({
      label: "Sparse CM12 VEX1 indirect dispatches",
      size: 4 * vexActivityBatchLayout.indirectWords,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const pressureAddressingABQA = {
        mode: pressureAddressingMode,
        layout: pressureAddressingABLayout,
        indirectArguments: device.createBuffer({
          label: "Sparse CM12 pressure-address materialization indirect dispatch",
          size: 12, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        }),
        ...(pressureAddressingModeForQA !== undefined ? {
        receiptReadback: device.createBuffer({
          label: "Sparse CM12 PAB1 receipt readback",
          size: 4 * SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        querySet: device.createQuerySet({ type: "timestamp", count: 4 }),
        queryResolve: device.createBuffer({
          label: "Sparse CM12 PAB1 timestamp resolve", size: 32,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        }),
        timestampReadback: device.createBuffer({
          label: "Sparse CM12 PAB1 timestamp readback", size: 32,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        } : {}),
      } satisfies SparseCM12PressureAddressingABQAResources;
    const framePlanIndirectArguments = device.createBuffer({
      label: "Sparse CM12 FPL1 fixed packet indirect dispatches",
      size: 4 * framePlanLayout.indirectSnapshotWords,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const presentationIndirectArguments = device.createBuffer({
      label: "Sparse CM12 FPP1 compact page indirect dispatch",
      size: 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const frameControlIndirectArguments = device.createBuffer({
      label: "Sparse CM12 FCA1 fixed authority indirect dispatches",
      size: 12 * SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
        | GPUBufferUsage.COPY_SRC,
    });
    const fineParams = device.createBuffer({
      label: "Sparse CM12 compact fine presentation parameters",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const fineMetadata = uploadBuffer(device,
      "Sparse CM12 compact fine presentation metadata", fine.metadata, storage);
    const fineWorklist = uploadBuffer(device,
      "Sparse CM12 compact fine presentation worklist", fine.worklist, storage);
    const fineSamples = device.createBuffer({
      label: "Sparse CM12 accepted/candidate fine presentation samples",
      size: Math.max(4, 2 * fine.plan.payloadCapacityBytes),
      usage: storage,
    });
    const fineWorkA = device.createBuffer({ label: "Sparse CM12 fine presentation work A",
      size: 4, usage: storage });
    const fineWorkB = device.createBuffer({ label: "Sparse CM12 fine presentation work B",
      size: 4, usage: storage });
    const fineRollback = device.createBuffer({ label: "Sparse CM12 fine presentation rollback",
      size: 4, usage: storage });
    const diagnosticsReadback = device.createBuffer({
      label: "Sparse CM12 resident diagnostic readback",
      // Reduction scalars, activity header, then authoritative accepted
      // cell/row worklist counts and compact pressure-cell count. These are QA
      // receipts, never schedule input.
      size: SPARSE_CM12_PRESSURE_SCALAR_BYTES + 4 * ACTIVITY_HEADER_WORDS + 12
        + SPARSE_CM12_PCM_DIAGNOSTIC_BYTES
        + 4 * SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS
        + 4 * (SPARSE_CM12_PRESSURE_CUTOVER_DIAGNOSTIC_WORDS
          + SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const bindGroupLayout = device.createBindGroupLayout({
      label: "Sparse CM12 resident layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        ...[2, 3, 4].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" as const } })),
        { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 14, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" } },
        { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const bindGroup = device.createBindGroup({ label: "Sparse CM12 resident bindings",
      layout: bindGroupLayout, entries: [
        { binding: 0, resource: { buffer: parameters } },
        { binding: 1, resource: { buffer: topology } },
        { binding: 2, resource: { buffer: state } },
        { binding: 3, resource: { buffer: partials } },
        { binding: 4, resource: { buffer: scalars } },
        { binding: 11, resource: { buffer: conditioning } },
        { binding: 12, resource: { buffer: activity } },
        { binding: 13, resource: { buffer: candidateState } },
        { binding: 14, resource: { buffer: fineMetadata } },
        { binding: 15, resource: { buffer: fineSamples } },
        { binding: 16, resource: { buffer: topologyArena } },
      ] });
    const pressureBindGroup = device.createBindGroup({
      label: "Sparse CM12 read-only pressure bindings",
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: parameters } },
        { binding: 1, resource: { buffer: topology } },
        { binding: 2, resource: { buffer: state } },
        { binding: 3, resource: { buffer: partials } },
        { binding: 4, resource: { buffer: scalars } },
        { binding: 11, resource: { buffer: conditioning } },
        { binding: 12, resource: { buffer: activity } },
        { binding: 13, resource: { buffer: candidateState } },
        { binding: 14, resource: { buffer: pressureTemplates } },
        { binding: 15, resource: { buffer: pressureWorklists } },
        { binding: 16, resource: { buffer: topologyArena } },
      ],
    });
    const compiler = gpuCompilationManagerFor(device);
    const shaderModule = compiler.createShaderModule({ label: "Sparse CM12 resident shader",
      code: createWebgpuSparseCM12ResidentWGSL(
        atlas.brickFineResolution, presentationPageResolution,
        temporalWorklistLayout, pressureWorklistData.layout,
        incrementalActivityLayout, canonicalMembershipLayout,
        framePlanLayout, framePlanPresentationLayout,
        frameControl.layout, scalarResultIngressLayout, pressureTopologyRepairLayout,
        persistentPressureCacheLayout,
        faceProjectionAuthorityLayout,
        vexActivityBatchLayout,
        pressureAddressingABLayout,
        pressureAddressingModeForQA === undefined ? "materializedList" : undefined,
        facePreparationTileCensusLayout,
        fpaVexReadCensusLayout,
        pressureRefreshOracleForQA,
      ) });
    const pipelineLayout = device.createPipelineLayout({ label: "Sparse CM12 resident pipeline layout",
      bindGroupLayouts: [bindGroupLayout] });
    const names = ["injectLiquid",
      "beginSparseCM12FrameControl", "publishSparseCM12FrameBodyAuthority",
      "publishSparseCM12FrameBoundaryAuthority", "sealSparseCM12FrameControl",
      "publishSparseCM12FrameScalarOutput", "publishSparseCM12FrameFaceOutput",
      "commitSparseCM12FrameControl", "invalidateSparseCM12FrameD4ForInjection",
      "sparseCM12FrameControlNoop",
      "publishSparseCM12MovingSolidVelocityRoots",
      "beginSparseCM12SRR1Ingress",
      "resetSparseCM12SRR1CopiedEvents", "beginSparseCM12SRR1ReceiptBatch",
      "resetTemporalScalarWorklists", "classifyTemporalPressureCells",
      "dilateTemporalScalarAtoB", "dilateTemporalScalarBtoA",
      "compactTemporalScalarCells", "compactTemporalScalarRows",
      "finalizeTemporalScalarWorklists",
      "traceGammaAndBeta", "scatterDensityDeficit",
      "gatherConservativeDensity", "compareSparseCM12MassResult",
      "seedTracers", "advanceTracers",
      "scatterGammaSnapshot", "finalizeGammaSnapshot", "scatterGammaRefinement",
      "finalizeGammaRefinement", "prepareSharpeningField",
      "scatterSharpeningMass", "finalizeSharpening",
      "clearSolidExcess", "scatterSolidExcess",
      "finalizeSolidExcess", "preserveHorizontalD4",
      "commitHorizontalD4", "preserveVelocityHorizontalD4",
      "commitVelocityHorizontalD4",
      "forceFaces", "classifyPressureCells", "classifyRows",
      "beginCanonicalPressureCells", "beginCanonicalPressureRows",
      "planPressureMembershipEpoch",
      "finalizeCanonicalPressureCellFrontier",
      "repairCanonicalPressureCellLeaves", "finalizeCanonicalPressureCells",
      "finalizeCanonicalPressureRowFrontier",
      "repairCanonicalPressureRowLeaves", "finalizeCanonicalPressureRows",
      "classifyDirtyPressureCells",
      "classifyDirtyPressureRows",
      "preparePressure",
      "restrictBrickAggregateResidual",
      "restrictPressureHierarchyResidual", "refinePressureHierarchyCorrection",
      "combinePressureHierarchyCorrection",
      "refineBrickAggregateAtoB1",
      "refineBrickAggregateBtoA2", "refineBrickAggregateAtoB3",
      "refineBrickAggregateBtoA4", "refineBrickAggregateAtoB5",
      "refineBrickAggregateBtoA6", "refineBrickAggregateAtoB7",
      "applyBrickAggregatePreconditioner",
      "initializeBrickAggregateDirection",
      "initializePCG", "measureTrueResidual", "measureGuardedTrueResidual",
      "reduceInitialTrueResidual", "reduceGuardedTrueResidual",
      "restartPCGAfterCurvatureLoss", "initializeBrickAggregateRecoveryDirection",
      "reduceCurvatureRecovery",
      "reduceFinalTrueResidual",
      "initializePipelinedImage", "reducePipelinedInitialize",
      "updatePipelinedState", "applyPipelinedImage", "reducePipelinedIteration",
      "applyPipelinedRecovery", "reducePipelinedRecovery",
      "reduceInitialize", "projectFaces",
      "collocateAndDiagnose", "measureDivergenceDiagnostics",
      "reduceDivergenceDiagnostics",
      "advanceActivityClock", "beginIncrementalActivity",
      "markIncrementalActivityTemporalCells", "markIncrementalActivityTopology",
      "markIncrementalActivityPostTopology",
      "finalizeIncrementalActivityWorklist", "measureBrickActivity",
      "ageIncrementalActivityHistory", "finalizeIncrementalActivityCensus",
      "planBrickResolution", "activateSweptReceivers", "closePlannedResolution",
      "validateCandidateResolution", "scheduleTopologyPreparation",
      "allocateCandidateTopologyPages", "synthesizeCandidateCellPages",
      "deferDynamicTopologyPublication",
      "beginShadowTopology", "buildShadowCellWorklist", "buildShadowRowWorklist",
      "finalizeShadowWorklists", "transferCandidateCells",
      "prepareCandidateFaceReceipts", "transferCandidateFaces",
      "writeCandidateCellsToShadow", "reconstructShadowFaces",
      "validateAndCommitShadowTopology",
      "publishSparseCM12TopologyVelocityRoots",
      "retireUnsupportedEmptyBricks", "sealSparseCM12PressureTopologyJournal",
      "classifyPresentationBricks",
      "publishSparseLevelSet",
      "beginSparseCM12FramePlanNext", "initializeSparseCM12FramePlanNext",
      "populateSparseCM12PresentationFramePlan",
      "importVelocityExtensionBlastToFramePlanNext",
      "resolveSparseCM12FramePlanNextClosure", "sealSparseCM12FramePlanNextBricks",
      "finalizeSparseCM12FramePlanNext", "markSparseCM12GlobalFramePlanReceipts",
      "beginSparseCM12FramePlanPresentationPacket",
      "buildSparseCM12FramePlanPresentationPacket",
      "finalizeSparseCM12FramePlanPresentationPacket",
      "executeSparseCM12FramePlanPresentationPacket",
      "commitSparseCM12FramePlanPresentationPacket",
      "finalizeSparseCM12FramePlanPresentationExecution",
      "rejectSparseCM12FramePlanPresentationFaults",
      "beginPersistentPressureCache", "finalizePersistentPressureCacheFrontier",
      "repairPersistentPressureCache", "finalizePersistentPressureFineCache",
      "seedPreviousPCFBrickLeaves", "seedPreviousPCFAggregateEdgeLeaves",
      "seedPreviousPCFHierarchyNodeLeaves", "seedPreviousPCFHierarchyEdgeLeaves",
      "repairPersistentPressureBrickWorkset",
      "repairPersistentPressureAggregateEdgeWorkset",
      "repairPersistentPressureHierarchyNodeWorkset",
      "repairPersistentPressureHierarchyEdgeWorkset",
      "finalizePersistentPressureAggregatePlan",
      "repairPersistentPressureAggregateEdges",
      "repairPersistentPressureBrickDiagonals",
      "finalizePersistentPressureAggregateExecution",
      "finalizePersistentPressureHierarchyPlan",
      "repairPersistentPressureHierarchyEdges",
      "repairPersistentPressureHierarchyDiagonals",
      "finalizePersistentPressureCache",
      "beginSparseCM12FacePreparationAuthority",
      "beginSparseCM12FaceProjectionAuthority",
      "seedSparseCM12FacePreparationBootstrap",
      "seedSparseCM12FaceProjectionBootstrap",
      "seedSparseCM12PreviousFacePreparationLeaves",
      "seedSparseCM12PreviousFaceProjectionLeaves",
      "seedSparseCM12ProjectionFromPreparation",
      "finalizeSparseCM12FacePreparationFrontier",
      "finalizeSparseCM12FaceProjectionFrontier",
      "repairSparseCM12FacePreparationLeaves",
      "repairSparseCM12FaceProjectionLeaves",
      "finalizeSparseCM12FacePreparationPlan",
      "finalizeSparseCM12FaceProjectionPlan",
      "executeSparseCM12FacePreparation", "executeSparseCM12FaceProjection",
      "verifySparseCM12FacePreparationLeaves",
      "verifySparseCM12FaceProjectionLeaves",
      "finalizeSparseCM12FacePreparationExecution",
      "finalizeSparseCM12FaceProjectionExecution",
      "markSparseCM12FacePreparationFromActivity",
      "markSparseCM12FaceProjectionFromPressure",
      ...sparseCM12PressureTopologyRepairEntryPoints(
        pressureTopologyRepairLayout,
      ),
      // These immutable oracle kernels are not production fallbacks. Keeping
      // them out of the ordinary eager set shortens first-visible startup and
      // makes their construction-only ownership mechanically obvious.
      ...(legacyHostAuthorityOracleForQA ? [
        "initializeTransportVelocity", "captureLegacyVelocityExtensionForQA",
        "finalizeLegacyVelocityExtensionClockForQA",
        "extrapolateTransportVelocityToSource",
        "extrapolateTransportVelocityToDestination",
      ] as const : []),
      ...(pressureRefreshOracleForQA ? [
        "captureLegacyFacePreparationAuthority", "preparePressureFullOracle",
        "bakeBrickAggregateDiagonal", "bakePressureHierarchyDiagonal",
        "bakePressureHierarchyEdges", "bakeBrickAggregateEdges",
      ] as const : []),
      ...(pressureRefreshOracleForQA || fpaVexReadCensusLayout
        ? ["prepareTransportFaces"] as const : []),
      ...(layout.journal !== 0
        ? ["journalIteration", "journalSnapshot"] as const : []),
    ] as const;
    const pressureAddressingConstants = pressureAddressingModeForQA === undefined
      ? undefined : sparseCM12PressureAddressingABPipelineConstants(
        pressureAddressingMode);
    const temporalSeedConstants = temporalSeedModeForQA === undefined ? undefined : {
      TEMPORAL_CHANGE_SEED_QA: temporalSeedModeForQA === "change" ? 1 : 0,
      TEMPORAL_SEED_COUNT_QA: 1,
    } as const;
    const residentConstants = pressureAddressingConstants || temporalSeedConstants
      ? { ...pressureAddressingConstants, ...temporalSeedConstants }
      : undefined;
    const entries = await Promise.all(names.map(async (name) => [name,
      await compiler.compileComputePipeline({ label: `Sparse CM12 ${name}`,
        layout: pipelineLayout, compute: { module: shaderModule, entryPoint: name,
          ...(residentConstants ? { constants: residentConstants } : {}) } },
      { priority: "visible" })] as const));
    const facePreparationTileCensusEntries = facePreparationTileCensusLayout
      ? await Promise.all([
        "beginSparseCM12FacePreparationTileCensus",
        "clearSparseCM12FacePreparationTileCensus",
        "markSparseCM12FacePreparationTileCensus",
        "finalizeSparseCM12FacePreparationTileCensus",
        "finalizeSparseCM12FacePreparationTileCensusWitness",
      ].map(async (name) => [name, await compiler.compileComputePipeline({
        label: `Sparse CM12 ${name}`,
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: name,
          ...(residentConstants ? { constants: residentConstants } : {}) },
      }, { priority: "visible" })] as const)) : [];
    const fpaVexReadCensusEntries = fpaVexReadCensusLayout
      ? await Promise.all([
        "beginSparseCM12FpaVexReadCensus", "clearSparseCM12FpaVexReadCensus",
        "captureSparseCM12ChangedEffectiveTransport",
        "captureSparseCM12PriorFaceForOracle",
        "scheduleSparseCM12FpaFromAcceptedVexReads",
        "beginSparseCM12FpaVexReadRecording", "endSparseCM12FpaVexReadRecording",
        "captureSparseCM12FpaOracleAndRestore",
        "finalizeSparseCM12FpaVexReadSummary", "verifySparseCM12FpaOracle",
        "commitSparseCM12FpaVexReadCensus",
      ].map(async (name) => [name, await compiler.compileComputePipeline({
        label: `Sparse CM12 ${name}`, layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: name,
          ...(residentConstants ? { constants: residentConstants } : {}) },
      }, { priority: "visible" })] as const)) : [];
    const pressureAddressingDescriptors =
      createSparseCM12PressureAddressingABPipelineDescriptors(
        pressureAddressingMode).slice(0, 4).filter((descriptor) =>
          pressureAddressingModeForQA !== undefined
            || descriptor.key !== "verifyPressureCellAddresses");
    const pressureAddressingEntries = await Promise.all(
      pressureAddressingDescriptors.map(async (descriptor) => [
        descriptor.key,
        await compiler.compileComputePipeline({
          label: `Sparse CM12 ${descriptor.key}`,
          layout: pipelineLayout,
          compute: { module: shaderModule, entryPoint: descriptor.entryPoint,
            ...(pressureAddressingConstants
              ? { constants: pressureAddressingConstants } : {}) },
        }, { priority: "visible" }),
      ] as const));
    const vexActivityEntries = await Promise.all(
      createSparseCM12VexActivityBatchPipelineDescriptors().map(async (descriptor) => [
        descriptor.key,
        await compiler.compileComputePipeline({
          label: `Sparse CM12 ${descriptor.key}`,
          layout: pipelineLayout,
          compute: {
            module: shaderModule,
            entryPoint: descriptor.entryPoint,
            constants: descriptor.constants,
          },
        }, { priority: "visible" }),
      ] as const),
    );
    const framePlanVerifyEntry = ["verifySparseCM12FramePlanPresentation",
      await compiler.compileComputePipeline({
        label: "Sparse CM12 verify FPL1 presentation coverage",
        layout: pipelineLayout,
        compute: { module: shaderModule,
          entryPoint: "verifySparseCM12FramePlanCurrentStage",
          constants: { CM12_FRAME_PLAN_VERIFY_STAGE: 5 } },
      }, { priority: "visible" })] as const;
    // Two pipelines from one entry point: the snapshot variant additionally
    // advances the device-side snapshot cursor and stamps its slot into the
    // record it writes. Compiled only when a journal was actually reserved.
    const journalEntries = layout.journal === 0 ? [] : [
      ["journalIterationSnapshot", await compiler.compileComputePipeline({
        label: "Sparse CM12 journalIteration with field snapshot",
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: "journalIteration",
          constants: { JOURNAL_SNAPSHOT: 1 },
        },
      }, { priority: "visible" })] as const,
    ];
    const rigidCoupling = rigid ? await WebGPUSparseCM12RigidCoupling.create(device, {
      parameters,
      state,
      topologyArena,
      frameControlIndirectArguments,
      rigidBodies: rigid.bodies,
      exchange: rigid.exchange,
    }) : undefined;
    const scalarResultAuthority = await WebGPUSparseCM12SRR1RuntimeAdapter.create(
      device, scalarResultPlan,
    );
    const result = new WebGPUSparseCM12Resident(
      device, atlas.dimensions, atlas.brickFineResolution, layout,
      [parameters, topology, state, partials, scalars, conditioning, activity,
        candidateState, topologyArena],
      acceptedIndirectArguments,
      pressureCellIndirectArguments,
      pressureRowIndirectArguments,
      pressureMembershipIndirectArguments,
      pressureTopologyRepairIndirectArguments,
      persistentPressureCacheIndirectArguments,
      faceProjectionAuthorityIndirectArguments,
      temporalCellIndirectArguments,
      temporalRowIndirectArguments,
      activityIndirectArguments,
      vexActivityIndirectArguments,
      framePlanIndirectArguments,
      presentationIndirectArguments,
      frameControlIndirectArguments,
      pressureWorklists,
      [fineParams, fineMetadata, fineWorklist, fineSamples, fineWorkA, fineWorkB,
        fineRollback],
      fine.plan,
      diagnosticsReadback,
      bindGroup,
      pressureBindGroup,
      pressureTemplates,
      temporalWorklistLayout,
      pressureWorklistData.layout,
      incrementalActivityLayout,
      canonicalMembershipLayout,
      framePlanLayout,
      framePlanPresentationLayout,
      frameControl.layout,
      scalarResultIngressLayout,
      vexActivityBatchLayout,
      pressureTopologyRepairLayout,
      persistentPressureCacheLayout,
      faceProjectionAuthorityLayout,
      facePreparationTileCensusLayout,
      fpaVexReadCensusLayout,
      scalarResultAuthority,
      pressureRefreshOracleForQA,
      presentationPublisherOracleForQA,
      legacyHostAuthorityOracleForQA,
      horizontalD4Authority,
      pressureAddressingABQA,
      temporalSeedModeForQA,
      Object.fromEntries([...entries, ...facePreparationTileCensusEntries,
        ...fpaVexReadCensusEntries,
        ...vexActivityEntries,
        ...pressureAddressingEntries, ...journalEntries, framePlanVerifyEntry]),
      templates.cellCount, templates.rowCount,
      templates.cellCount, templates.rowCount,
      pressureCoarseEdgeCount,
      pressureEdgeCount,
      pressureHierarchyGroupCounts.reduce((sum, count) => sum + count, 0),
      pressureHierarchyEdgeCounts.reduce((sum, count) => sum + count, 0),
      pressureScratchBytes,
      physicalTemplateBytes,
      templates.words,
      atlas.boundary, rigidCoupling);
    result.writeParameters(packed, 0.004, 1, 1, [0, 0, 0], undefined, undefined,
      undefined, 0, rigid?.worldDimensions_m);
    return result;
  }

  encode(
    encoder: GPUCommandEncoder,
    dt_s: number,
    finestCellSize_m: number,
    pressureScale: number,
    accelerationFinePerSecond2: readonly [number, number, number],
    sharpening?: SharpeningTrace,
    activityPolicy?: SparseCM12ActivityPolicy,
    pressureControl?: SparseCM12PressureControl,
    seams?: SparseCM12ResidentStageSeams,
    bodyCount = 0,
    worldDimensions_m?: readonly [number, number, number],
  ): void {
    this.assertLive();
    const packed = this.lastPacked!;
    this.writeParameters(packed, dt_s, finestCellSize_m, pressureScale,
      accelerationFinePerSecond2, sharpening, activityPolicy, pressureControl, bodyCount,
      worldDimensions_m);
    const pressureIterations = sparseCM12PressureIterations(pressureControl?.iterations);
    // The header carries the two device-side cursors, so it starts each
    // captured frame at zero. The records and snapshots behind it are
    // overwritten in place and never read past their cursor.
    const journaling = this.journalArmed && this.layout.journal !== 0;
    if (journaling) {
      encoder.clearBuffer(this.state, 4 * this.layout.journal,
        4 * SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS);
    }
    // Which encoded iterations carry a whole-field snapshot. A pure function of
    // the ceiling and the reserved capacity, so the host and the device-side
    // cursor agree without either telling the other.
    const journalSnapshots = journaling
      ? new Set(sparseCM12PressureJournalSchedule(pressureIterations,
        this.layout.journalLayout.snapshotCapacity))
      : undefined;
    // Published for the film's scrub, which otherwise has no way to know where
    // the capture ends: the reserved capacity is an upper bound, and a short
    // solve fills fewer slots than it. Scrubbing across the unfilled tail would
    // show the previous capture's frames as though they belonged to this one.
    if (journalSnapshots) this.journalSnapshotCount = journalSnapshots.size;
    encoder.clearBuffer(this.conditioning, 0,
      Math.max(4, 12 * this.templateCellCount));
    // Liquid membership and ghost-fluid theta are PCM-owned persistent caches.
    // Bootstrap initializes the complete accepted domain; later epochs repair
    // only dirty/topology closure and explicitly zero retired entries.
    // Live pressure-row census. classifyRows increments these counters from
    // the accepted row worklist after topology publication and liquid/ghost-
    // fluid classification; they are diagnostics only and never schedule work.
    encoder.clearBuffer(this.activity, 4 * ACCEPTED_COARSE_ROW_COUNT_WORD,
      4 * (PRESSURE_ACTIVE_ROW_COUNT_WORD - ACCEPTED_COARSE_ROW_COUNT_WORD + 1));
    // The pass opens on first dispatch rather than up front. A stage that
    // encodes nothing this advance — the D4 authority on an asymmetric scene —
    // then leaves no empty pass behind, which matters because Metal writes no
    // timestamp for a pass that does no work and one unsampled boundary
    // rejects the whole chain.
    let pass: GPUComputePassEncoder | undefined;
    let activeBindGroup = this.bindGroup;
    let passLabel = "Sparse CM12 resident frame";
    const openPass = () => {
      if (!pass) {
        pass = encoder.beginComputePass({ label: passLabel });
        pass.setBindGroup(0, activeBindGroup);
      }
      return pass;
    };
    const dispatch = (name: string, count: number, y = 1, z = 1) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroups(count, y, z);
    };
    const dispatchAccepted = (name: string, kind: "cell" | "row") => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      const argumentWord = kind === "cell" ? 8 : 11;
      activePass.dispatchWorkgroupsIndirect(this.acceptedIndirectArguments,
        4 * (argumentWord - 8));
    };
    const dispatchScalarResult = (name: string,
      family: keyof typeof SPARSE_CM12_SRR1_INDIRECT_FAMILY) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.scalarResultAuthority.indirectBuffer,
        12 * SPARSE_CM12_SRR1_INDIRECT_FAMILY[family]);
    };
    const dispatchPressureCell = (name: string) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.pressureCellIndirectArguments, 0);
    };
    const dispatchPressureRow = (name: string) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.pressureRowIndirectArguments, 0);
    };
    const dispatchPressureBootstrap = (name: string, kind: "cell" | "row") => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.pressureMembershipIndirectArguments,
        kind === "cell" ? 0 : 12);
    };
    const dispatchPressureTopologyRepair = (name: string, byteOffset: number) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(
        this.pressureTopologyRepairIndirectArguments, byteOffset,
      );
    };
    const dispatchPersistentPressureCache = (name: string, slot: number) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(
        this.persistentPressureCacheIndirectArguments, 12 * slot,
      );
    };
    const dispatchPressureBrickSolve = (name: string) => {
      dispatch(name, packed.brickCount);
    };
    const dispatchPressureHierarchySolve = (name: string) => {
      dispatch(name, this.pressureHierarchyGroupCount);
    };
    const dispatchFaceProjectionAuthority = (name: string, slot: number) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(
        this.faceProjectionAuthorityIndirectArguments, 12 * slot,
      );
    };
    const dispatchTemporalCell = (name: string) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.temporalCellIndirectArguments, 0);
    };
    const dispatchTemporalRow = (name: string) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.temporalRowIndirectArguments, 0);
    };
    const dispatchActivity = (name: string) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.activityIndirectArguments, 0);
    };
    const dispatchVexActivity = (name: string, packet: SparseCM12VexActivityBatchPacket) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.vexActivityIndirectArguments,
        4 * SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT[packet].offsetWords);
    };
    const copyVexActivity = (id: string) => {
      const copy = this.vexActivityIndirectCopies.get(id);
      if (!copy) throw new Error(`unknown VEX1 indirect copy ${id}`);
      closePass();
      encoder.copyBufferToBuffer(this.activity, 4 * copy.sourceWord,
        this.vexActivityIndirectArguments, 4 * copy.destinationWord, 12);
    };
    const dispatchFrameControl = (
      name: string,
      family: typeof SPARSE_CM12_FRAME_CONTROL_FAMILY[
        keyof typeof SPARSE_CM12_FRAME_CONTROL_FAMILY],
    ) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.frameControlIndirectArguments, 12 * family);
    };
    const closePass = () => {
      pass?.end();
      pass = undefined;
    };
    const useBindGroup = (bindGroup: GPUBindGroup) => {
      activeBindGroup = bindGroup;
      pass?.setBindGroup(0, bindGroup);
    };
    // Without seams this is the single frame pass it has always been. With
    // them, each stage becomes its own pass so a boundary chain can land a
    // hardware timestamp on the pass that opens the next stage. Dispatch order
    // and the implicit barriers between dispatches are identical either way,
    // so a traced advance computes exactly what an untraced one computes.
    const stage = (id: SparseCM12ResidentStageId, encodeStage: () => void) => {
      if (seams) {
        passLabel = `Sparse CM12 resident ${id}`;
      }
      encodeStage();
      if (!seams) return;
      pass?.end();
      pass = undefined;
      seams.close(id);
    };
    /**
     * Record one encoded iteration, and snapshot the fields when this is a
     * scheduled snapshot iteration.
     *
     * Encodes nothing at all when the journal is disarmed — the whole feature
     * costs a branch on the host, which is the same bargain the tracer stage
     * strikes. Inside a captured frame it is two dispatches, in the open pass,
     * with no copy and therefore no pass boundary.
     */
    const journalRecord = (iteration: number) => {
      if (!journalSnapshots) return;
      const snapshot = journalSnapshots.has(iteration);
      dispatch(snapshot ? "journalIterationSnapshot" : "journalIteration", 1);
      if (snapshot) dispatchAccepted("journalSnapshot", "cell");
    };
    const bricks = Math.ceil(packed.brickCount / WORKGROUP_SIZE);
    stage("transport-velocity-extension", () => {
      // FCA1 translates external inputs and persistent D4 receipts into a
      // sealed set of fixed indirect families. The host always encodes both
      // work and singleton bypass packets; it never inspects evolving state.
      // The immutable legacy-physics oracle still advances the passive FCA1
      // clock. Downstream SRR/FPL receipts share this generation/parity even
      // though the oracle deliberately does not consume FCA indirect packets.
      dispatch("beginSparseCM12FrameControl", 1);
      dispatch("publishSparseCM12FrameBodyAuthority", 1);
      dispatch("publishSparseCM12FrameBoundaryAuthority", 1);
      dispatch("sealSparseCM12FrameControl", 1);
      if (!this.legacyHostAuthorityOracleForQA) {
        closePass();
        encoder.copyBufferToBuffer(this.topologyArena,
          4 * this.frameControlLayout.indirectBaseWords,
          this.frameControlIndirectArguments, 0,
          12 * SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT);
        this.rigidCoupling?.encodeVoxelization(encoder);
        useBindGroup(this.bindGroup);
        dispatchFrameControl("publishSparseCM12MovingSolidVelocityRoots",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.solidCellWork);
        dispatchFrameControl("sparseCM12FrameControlNoop",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyBypass);
        dispatchFrameControl("sparseCM12FrameControlNoop",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyRowBypass);
      }
      // Transport extrapolation consumes the same construction-time CSR edge
      // cache as pressure. Keep that immutable topology bound through the hot
      // physics stages; presentation restores its own metadata binding later.
      useBindGroup(this.pressureBindGroup);
      if (this.legacyHostAuthorityOracleForQA) {
        dispatchAccepted("initializeTransportVelocity", "cell");
        for (let sweep = 0; sweep < 8; sweep += 1) {
          dispatchAccepted(sweep % 2 === 0
            ? "extrapolateTransportVelocityToSource"
            : "extrapolateTransportVelocityToDestination", "cell");
        }
        dispatchAccepted("captureLegacyVelocityExtensionForQA", "cell");
      } else {
        // The prior frame sealed and copied this exact blast after its final
        // producer. Frame start consumes it without rebuilding topology or
        // mutating FPL Current.
        dispatch("beginVelocityExtensionExecution", 1);
        dispatchVexActivity("initializeVelocityExtensionCandidates", "vexSerial");
        for (let depth = 1; depth <= 8; depth += 1) {
          dispatchVexActivity(`advanceVelocityExtensionCandidates${depth}`, "vexSerial");
        }
        dispatchVexActivity("commitVelocityExtensionCandidates", "vexSerial");
        dispatch("finalizeVelocityExtensionCandidate", 1);
      }
      // Producer-authored activity/topology changes append bounded SIR1 tile
      // events. The dedicated authority plans only those events plus its
      // persistent work tree; no accepted-cell scan or host count participates.
      dispatch("beginSparseCM12SRR1Ingress", 1);
      closePass();
      this.scalarResultAuthority.encodePlan(encoder, this.activity);
      useBindGroup(this.pressureBindGroup);
      dispatch("resetSparseCM12SRR1CopiedEvents",
        Math.ceil(this.scalarResultIngressLayout.tileCapacity / WORKGROUP_SIZE));
      dispatch("beginSparseCM12SRR1ReceiptBatch", 1);
    });
    stage("face-preparation", () => {
      // Immutable construction QA arm: preserve the original accepted-row
      // invocation stream and arithmetic, then snapshot its destination bank
      // for byte-for-byte comparison with production FPA. This is not exposed
      // as a runtime fallback or selected from GPU/scene state.
      if (this.pressureRefreshOracleForQA) {
        dispatchAccepted("prepareTransportFaces", "row");
        dispatchAccepted("captureLegacyFacePreparationAuthority", "row");
        return;
      }
      // Construction-only actual-read census. Snapshot current dependencies,
      // run the unchanged full FPA arithmetic while recording only VEX xyz
      // accessor reads, restore the physical bank, then leave the production
      // FPA schedule below untouched. The singleton bootstrap publishes the
      // first complete graph; later candidates promote only after C⊆S verify.
      if (this.fpaVexReadCensusLayout) {
        const census = this.fpaVexReadCensusLayout;
        dispatch("beginSparseCM12FpaVexReadCensus", 1);
        dispatch("clearSparseCM12FpaVexReadCensus", Math.max(1, Math.ceil(Math.max(
          census.rowCapacity, census.cellBitWords, census.tileBitWords,
          census.rowBitWords) / WORKGROUP_SIZE)));
        dispatchAccepted("captureSparseCM12ChangedEffectiveTransport", "cell");
        dispatchAccepted("captureSparseCM12PriorFaceForOracle", "row");
        dispatch("beginSparseCM12FpaVexReadRecording", 1);
        dispatchAccepted("prepareTransportFaces", "row");
        dispatch("endSparseCM12FpaVexReadRecording", 1);
        dispatchAccepted("captureSparseCM12FpaOracleAndRestore", "row");
        dispatch("finalizeSparseCM12FpaVexReadSummary", Math.max(1,
          Math.ceil(census.rowCapacity / WORKGROUP_SIZE)));
        dispatch("scheduleSparseCM12FpaFromAcceptedVexReads", Math.max(1,
          Math.ceil(census.rowCapacity / WORKGROUP_SIZE)));
      }
      // Construction-only observation: snapshot the prior accepted authority
      // and author full-brick/tile-selected shadow row sets. The production
      // producer and execution stream immediately below remain unchanged.
      if (this.facePreparationTileCensusLayout) {
        dispatch("beginSparseCM12FacePreparationTileCensus", 1);
        dispatch("clearSparseCM12FacePreparationTileCensus", Math.max(1,
          Math.ceil(Math.max(this.facePreparationTileCensusLayout.rowCapacity,
            this.facePreparationTileCensusLayout.rowBitWordCount) / WORKGROUP_SIZE)));
        dispatchActivity("markSparseCM12FacePreparationTileCensus");
      }
      dispatch("beginSparseCM12FacePreparationAuthority", 1);
      closePass();
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12FaceProjectionBootstrapIndirectByteOffset(
          this.faceProjectionAuthorityLayout, "preparation"),
        this.faceProjectionAuthorityIndirectArguments, 0, 12);
      dispatchFaceProjectionAuthority("seedSparseCM12FacePreparationBootstrap", 0);
      dispatch("seedSparseCM12PreviousFacePreparationLeaves",
        Math.max(1, Math.ceil(this.faceProjectionAuthorityLayout.preparation.leafCount
          / WORKGROUP_SIZE)));
      dispatchActivity("markSparseCM12FacePreparationFromActivity");
      dispatch("finalizeSparseCM12FacePreparationFrontier", 1);
      closePass();
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12FaceProjectionIndirectByteOffset(
          this.faceProjectionAuthorityLayout, "preparation", "repair"),
        this.faceProjectionAuthorityIndirectArguments, 24, 12);
      dispatchFaceProjectionAuthority("repairSparseCM12FacePreparationLeaves", 2);
      dispatch("finalizeSparseCM12FacePreparationPlan", 1);
      closePass();
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12FaceProjectionIndirectByteOffset(
          this.faceProjectionAuthorityLayout, "preparation", "work"),
        this.faceProjectionAuthorityIndirectArguments, 48, 12);
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12FaceProjectionBootstrapIndirectByteOffset(
          this.faceProjectionAuthorityLayout, "preparation"),
        this.faceProjectionAuthorityIndirectArguments, 72, 12);
      dispatchFaceProjectionAuthority("executeSparseCM12FacePreparation", 4);
      dispatchFaceProjectionAuthority("verifySparseCM12FacePreparationLeaves", 6);
      dispatch("finalizeSparseCM12FacePreparationExecution", 1);
      if (this.facePreparationTileCensusLayout) {
        dispatch("finalizeSparseCM12FacePreparationTileCensus", Math.max(1,
          Math.ceil(this.facePreparationTileCensusLayout.rowCapacity / WORKGROUP_SIZE)));
        dispatch("finalizeSparseCM12FacePreparationTileCensusWitness", 1);
      }
      if (this.fpaVexReadCensusLayout) {
        dispatchAccepted("verifySparseCM12FpaOracle", "row");
        dispatch("commitSparseCM12FpaVexReadCensus", 1);
      }
    });
    stage("conservative-transport", () => {
      dispatchScalarResult("traceGammaAndBeta", "traceGammaAndBeta");
      dispatchScalarResult("scatterDensityDeficit", "scatterDensityDeficit");
      dispatchScalarResult("gatherConservativeDensity", "gatherConservativeDensity");
    });
    // After the conservative transport, which leaves both fields the markers
    // need untouched: `gatherConservativeDensity` writes the *destination*
    // density, and nothing between the extrapolation sweeps and the projection
    // writes the collocated transport velocity. So a marker here integrates the
    // same characteristic, through the same velocity, as the mass did.
    //
    // Encoding nothing when the view is off is the point: a stage that dispatches
    // no work closes at zero and the whole feature costs a branch on the host.
    stage("tracer-advection", () => {
      if (!this.tracersEnabled || this.tracerLattice.count === 0) return;
      const groups = Math.ceil(this.tracerLattice.count / WORKGROUP_SIZE);
      if (this.tracerSeedPending) {
        dispatch("seedTracers", groups);
        this.tracerSeedPending = false;
      }
      dispatch("advanceTracers", groups);
    });
    // Conservative transport has consumed the first three accumulator banks.
    // Recycle them for the row-owned gamma transaction; a buffer clear cannot
    // be encoded inside an open compute pass, so this is the deliberate phase
    // boundary in the otherwise resident frame.
    closePass();
    encoder.clearBuffer(this.conditioning, 0,
      Math.max(4, 8 * this.templateCellCount));
    stage("gamma-diffusion", () => {
      dispatchAccepted("scatterGammaSnapshot", "row");
      dispatchAccepted("finalizeGammaSnapshot", "cell");
      closePass();
      encoder.clearBuffer(this.conditioning, 0,
        Math.max(4, 16 * this.templateCellCount));
      encoder.clearBuffer(this.candidateState, 0, this.pressureScratchBytes);
      dispatchAccepted("scatterGammaRefinement", "row");
      dispatchAccepted("finalizeGammaRefinement", "cell");
      closePass();
    });
    stage("surface-sharpening", () => {
      // Surface scatter can write receivers outside the source tile. Until a
      // producer-authored receiver closure is sealed, retain the exact full
      // accepted traversal; the attempted SRR-source-only cut lost mass when
      // symmetric expansion reached an unscheduled receiver at step twenty.
      dispatchAccepted("prepareSharpeningField", "cell");
      dispatchAccepted("scatterSharpeningMass", "cell");
      dispatchAccepted("finalizeSharpening", "cell");
      if (!this.legacyHostAuthorityOracleForQA) {
        dispatchFrameControl("clearSolidExcess",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.solidCellWork);
        dispatchFrameControl("scatterSolidExcess",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.solidCellWork);
        dispatchFrameControl("finalizeSolidExcess",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.solidCellWork);
        dispatchFrameControl("sparseCM12FrameControlNoop",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.solidCellBypass);
      }
      // Numerical scalar/gamma/surface work still uses the complete accepted
      // traversal. Build the downstream pressure/activity frontier only after
      // the fully conditioned destination banks exist; observing source banks
      // here misses interface crossings produced during this frame.
      dispatch("resetTemporalScalarWorklists", 1);
      dispatchAccepted("classifyTemporalPressureCells", "cell");
      dispatchAccepted("dilateTemporalScalarAtoB", "cell");
      dispatchAccepted("dilateTemporalScalarBtoA", "cell");
      dispatchAccepted("compactTemporalScalarCells", "cell");
      dispatchAccepted("compactTemporalScalarRows", "row");
      dispatch("finalizeTemporalScalarWorklists", 1);
      closePass();
      // The exact-result receipt observes the fully conditioned physical banks,
      // after gamma diffusion and surface sharpening have published the values
      // the next frame will consume. It never certifies an intermediate mass
      // result that a later scalar producer can invalidate.
      useBindGroup(this.pressureBindGroup);
      dispatchScalarResult("compareSparseCM12MassResult", "compareMassResult");
      closePass();
      this.scalarResultAuthority.encodePublish(encoder, this.activity);
      encoder.copyBufferToBuffer(this.activity,
        4 * (this.temporalWorklistLayout.headerBaseWords + 1),
        this.temporalCellIndirectArguments, 0, 12);
      encoder.copyBufferToBuffer(this.activity,
        4 * (this.temporalWorklistLayout.headerBaseWords + 5),
        this.temporalRowIndirectArguments, 0, 12);
    });
    stage("symmetry-authority", () => {
      if (this.legacyHostAuthorityOracleForQA) {
        if (this.legacyHostD4AuthorityForQA) {
          dispatchAccepted("preserveHorizontalD4", "cell");
          dispatchAccepted("commitHorizontalD4", "cell");
        }
      } else {
        dispatchFrameControl("preserveHorizontalD4",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Work);
        dispatchFrameControl("commitHorizontalD4",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Work);
        dispatchFrameControl("sparseCM12FrameControlNoop",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Bypass);
      }
      dispatch("publishSparseCM12FrameScalarOutput", 1);
    });
    stage("body-forces", () => {
      dispatchAccepted("forceFaces", "row");
    });
    stage("pressure-topology", () => {
      // Conservative conditioning is dead after sharpening. Reuse its large
      // cell-scaled arena for stable-ID liquid compaction, then copy only the
      // three indirect words outside the pass to satisfy WebGPU usage scopes.
      // Every header and per-workgroup count read by the finalizers is written
      // by this epoch, so clearing the complete template-sized arena here only
      // spends bandwidth and adds a pass boundary.
      closePass();
      useBindGroup(this.pressureBindGroup);
      dispatch("beginCanonicalPressureCells", 1);
      dispatch("beginCanonicalPressureRows", 1);
      dispatch("beginPersistentPressureCache", 1);
      dispatch("planPressureMembershipEpoch", 1);
      dispatch("beginSparseCM12PressureTopologyRepair", 1);
      dispatch("captureSparseCM12PressureTopologyConsumerGenerations", 1);
      closePass();
      const persistentFamilies = ["brick", "aggregateEdge", "hierarchyNode",
        "hierarchyEdge"] as const;
      persistentFamilies.forEach((family, index) => {
        encoder.copyBufferToBuffer(this.topologyArena,
          sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(
            this.persistentPressureCacheLayout, family, "seed"),
          this.persistentPressureCacheIndirectArguments, 12 * (1 + index), 12);
      });
      dispatchPersistentPressureCache("seedPreviousPCFBrickLeaves", 1);
      dispatchPersistentPressureCache("seedPreviousPCFAggregateEdgeLeaves", 2);
      dispatchPersistentPressureCache("seedPreviousPCFHierarchyNodeLeaves", 3);
      dispatchPersistentPressureCache("seedPreviousPCFHierarchyEdgeLeaves", 4);
      // Indirect arguments are copied out of GPU-authored headers. WebGPU
      // forbids transfer commands while the shared compute pass is open.
      closePass();
      encoder.copyBufferToBuffer(this.pressureWorklists,
        4 * (this.pressureRepairLayout.headerBaseWords
          + SPARSE_CM12_PRESSURE_REPAIR_HEADER.bootstrapCellIndirect),
        this.pressureMembershipIndirectArguments, 0,
        SPARSE_CM12_PRESSURE_MEMBERSHIP_INDIRECT_BYTES);
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12PressureTopologyRepairHeaderIndirectByteOffset(
          this.pressureTopologyRepairLayout, "brickSeed"),
        this.pressureTopologyRepairIndirectArguments, 0, 12);
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12PressureTopologyRepairHeaderIndirectByteOffset(
          this.pressureTopologyRepairLayout, "rowSeed"),
        this.pressureTopologyRepairIndirectArguments, 36, 12);
      dispatchPressureTopologyRepair(
        "seedPreviousSparseCM12PressureTopologyBrickLeaves", 0);
      dispatchPressureTopologyRepair(
        "seedPreviousSparseCM12PressureTopologyRowLeaves", 36);
      dispatch("finalizeSparseCM12PressureTopologyBrickFrontier", 1);
      closePass();
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12PressureTopologyRepairIndirectByteOffset(
          this.pressureTopologyRepairLayout, "brick", "repair"),
        this.pressureTopologyRepairIndirectArguments, 12, 12);
      dispatchPressureTopologyRepair(
        "repairSparseCM12PressureTopologyBrickLeaves", 12);
      for (let level = 1;
        level < this.pressureTopologyRepairLayout.brick.treeLevelCounts.length;
        level += 1) {
        dispatchPressureTopologyRepair(
          `reduceSparseCM12PressureTopologyBrickLevel${level}`, 12);
      }
      dispatch("finalizeSparseCM12PressureTopologyBrickPlan", 1);
      closePass();
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12PressureTopologyRepairIndirectByteOffset(
          this.pressureTopologyRepairLayout, "brick", "work"),
        this.pressureTopologyRepairIndirectArguments, 24, 12);
      dispatchPressureTopologyRepair(
        "repairSparseCM12PressureTopologyChangedBricks", 24);
      if (this.pressureRefreshOracleForQA) {
        dispatchAccepted("classifyPressureCells", "cell");
      } else {
        dispatchPressureBootstrap("classifyPressureCells", "cell");
      }
      if (!this.pressureRefreshOracleForQA) {
        dispatchTemporalCell("classifyDirtyPressureCells");
      }
      dispatch("finalizeCanonicalPressureCellFrontier", 1);
      closePass();
      encoder.copyBufferToBuffer(this.activity,
        sparseCM12CanonicalMembershipRepairIndirectByteOffset(
          this.canonicalMembershipLayout, "cell"),
        this.pressureMembershipIndirectArguments, 0, 12);
      dispatchPressureBootstrap("repairCanonicalPressureCellLeaves", "cell");
      dispatch("finalizeCanonicalPressureCells", 1);
      if (this.pressureAddressingABQA.mode === "materializedList") {
        const qa = this.pressureAddressingABQA;
        dispatch("beginPressureAddressMaterialization", 1);
        closePass();
        encoder.copyBufferToBuffer(this.activity,
          4 * (qa.layout.baseWords
            + SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER.materializeIndirectX),
          qa.indirectArguments, 0, 12);
        if (qa.querySet) {
          const materializePass = encoder.beginComputePass({
            label: "Sparse CM12 PAB1 materialize",
            timestampWrites: { querySet: qa.querySet,
              beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
          });
          materializePass.setBindGroup(0, this.pressureBindGroup);
          materializePass.setPipeline(this.pipelines.materializePressureCellAddresses!);
          materializePass.dispatchWorkgroupsIndirect(qa.indirectArguments, 0);
          materializePass.end();
          const verifyPass = encoder.beginComputePass({
            label: "Sparse CM12 PAB1 verify",
            timestampWrites: { querySet: qa.querySet,
              beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 },
          });
          verifyPass.setBindGroup(0, this.pressureBindGroup);
          verifyPass.setPipeline(this.pipelines.verifyPressureCellAddresses!);
          verifyPass.dispatchWorkgroupsIndirect(qa.indirectArguments, 0);
          verifyPass.end();
        } else {
          const materializePass = encoder.beginComputePass({
            label: "Sparse CM12 production pressure-address materialization",
          });
          materializePass.setBindGroup(0, this.pressureBindGroup);
          materializePass.setPipeline(this.pipelines.materializePressureCellAddresses!);
          materializePass.dispatchWorkgroupsIndirect(qa.indirectArguments, 0);
          materializePass.end();
        }
        dispatch("finalizePressureAddressMaterialization", 1);
        if (qa.querySet && qa.queryResolve && qa.timestampReadback
          && qa.receiptReadback) {
          closePass();
          encoder.resolveQuerySet(qa.querySet, 0, 4, qa.queryResolve, 0);
          encoder.copyBufferToBuffer(qa.queryResolve, 0, qa.timestampReadback, 0, 32);
          encoder.copyBufferToBuffer(this.activity, 4 * qa.layout.baseWords,
            qa.receiptReadback, 0,
            4 * SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS);
        }
      }
      dispatch("finalizeSparseCM12PressureTopologyCellExecution", 1);
      closePass();
      encoder.copyBufferToBuffer(this.activity,
        sparseCM12CanonicalMembershipRepairIndirectByteOffset(
          this.canonicalMembershipLayout, "cell"),
        this.pressureCellIndirectArguments, 0, 12);
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12PressureTopologyRepairIndirectByteOffset(
          this.pressureTopologyRepairLayout, "row", "repair"),
        this.pressureTopologyRepairIndirectArguments, 48, 12);
      dispatchPressureTopologyRepair(
        "repairSparseCM12PressureTopologyRowLeaves", 48);
      for (let level = 1;
        level < this.pressureTopologyRepairLayout.row.treeLevelCounts.length;
        level += 1) {
        dispatchPressureTopologyRepair(
          `reduceSparseCM12PressureTopologyRowLevel${level}`, 48);
      }
      dispatch("finalizeSparseCM12PressureTopologyRowPlan", 1);
      closePass();
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12PressureTopologyRepairIndirectByteOffset(
          this.pressureTopologyRepairLayout, "row", "work"),
        this.pressureTopologyRepairIndirectArguments, 60, 12);
      dispatchPressureTopologyRepair("repairSparseCM12PressureTopologyRows", 60);
      if (this.pressureRefreshOracleForQA) {
        dispatchAccepted("classifyRows", "row");
      } else {
        dispatchPressureBootstrap("classifyRows", "row");
      }
      if (!this.pressureRefreshOracleForQA) {
        dispatchTemporalRow("classifyDirtyPressureRows");
      }
      dispatch("finalizeCanonicalPressureRowFrontier", 1);
      closePass();
      encoder.copyBufferToBuffer(this.activity,
        sparseCM12CanonicalMembershipRepairIndirectByteOffset(
          this.canonicalMembershipLayout, "row"),
        this.pressureMembershipIndirectArguments, 12, 12);
      dispatchPressureBootstrap("repairCanonicalPressureRowLeaves", "row");
      dispatch("finalizeCanonicalPressureRows", 1);
      dispatch("finalizeSparseCM12PressureTopologyRowExecution", 1);
      dispatch("finalizePersistentPressureCacheFrontier", 1);
      closePass();
      encoder.copyBufferToBuffer(this.activity,
        sparseCM12CanonicalMembershipRepairIndirectByteOffset(
          this.canonicalMembershipLayout, "row"),
        this.pressureRowIndirectArguments, 0, 12);
      closePass();
      encoder.copyBufferToBuffer(this.topologyArena,
        4 * (this.persistentPressureCacheLayout.headerBaseWords
          + SPARSE_CM12_PRESSURE_CACHE_HEADER.repairIndirectX),
        this.persistentPressureCacheIndirectArguments, 0, 12);
      dispatchPersistentPressureCache("repairPersistentPressureCache", 0);
      dispatch("finalizePersistentPressureFineCache", 1);
      closePass();
      (persistentFamilies.slice(0, 2) as readonly (typeof persistentFamilies[number])[])
        .forEach((family, index) => {
          encoder.copyBufferToBuffer(this.topologyArena,
            sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(
              this.persistentPressureCacheLayout, family, "repair"),
            this.persistentPressureCacheIndirectArguments, 12 * (5 + index), 12);
        });
      dispatchPersistentPressureCache("repairPersistentPressureBrickWorkset", 5);
      dispatchPersistentPressureCache("repairPersistentPressureAggregateEdgeWorkset", 6);
      dispatch("finalizePersistentPressureAggregatePlan", 1);
      closePass();
      (persistentFamilies.slice(0, 2) as readonly (typeof persistentFamilies[number])[])
        .forEach((family, index) => {
          encoder.copyBufferToBuffer(this.topologyArena,
            sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(
              this.persistentPressureCacheLayout, family, "work"),
            this.persistentPressureCacheIndirectArguments, 12 * (9 + index), 12);
        });
      dispatchPersistentPressureCache("repairPersistentPressureAggregateEdges", 10);
      dispatchPersistentPressureCache("repairPersistentPressureBrickDiagonals", 9);
      dispatch("finalizePersistentPressureAggregateExecution", 1);
      closePass();
      (persistentFamilies.slice(2) as readonly (typeof persistentFamilies[number])[])
        .forEach((family, index) => {
          encoder.copyBufferToBuffer(this.topologyArena,
            sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(
              this.persistentPressureCacheLayout, family, "repair"),
            this.persistentPressureCacheIndirectArguments, 12 * (7 + index), 12);
        });
      dispatchPersistentPressureCache("repairPersistentPressureHierarchyNodeWorkset", 7);
      dispatchPersistentPressureCache("repairPersistentPressureHierarchyEdgeWorkset", 8);
      dispatch("finalizePersistentPressureHierarchyPlan", 1);
      closePass();
      (persistentFamilies.slice(2) as readonly (typeof persistentFamilies[number])[])
        .forEach((family, index) => {
          encoder.copyBufferToBuffer(this.topologyArena,
            sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(
              this.persistentPressureCacheLayout, family, "work"),
            this.persistentPressureCacheIndirectArguments, 12 * (11 + index), 12);
        });
      dispatchPersistentPressureCache("repairPersistentPressureHierarchyDiagonals", 11);
      dispatchPersistentPressureCache("repairPersistentPressureHierarchyEdges", 12);
      dispatch("finalizePersistentPressureCache", 1);
      // PTR and PCF form one GPU-authored transaction. PTR has already built
      // its commit indirect, but persistent brick states cannot become visible
      // until PCF accepts the exact candidate generation captured by PTR.
      closePass();
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12PressureTopologyRepairHeaderIndirectByteOffset(
          this.pressureTopologyRepairLayout, "commit"),
        this.pressureTopologyRepairIndirectArguments, 72, 12);
      dispatchPressureTopologyRepair(
        "commitSparseCM12PressureTopologyBrickStates", 72);
      dispatch("finalizeSparseCM12BoundedPressureTopologyRepair", 1);
      // Open the next topology journal immediately. Resolution/activation/
      // retirement producers later in this frame append without host state.
      dispatch("beginSparseCM12PressureTopologyRepair", 1);
      if (this.pressureRefreshOracleForQA) {
        // Construction-only legacy aggregate oracle. It uses the same stable
        // PCM rank/order for fine cells, then reproduces the pre-PCF full
        // aggregate/hierarchy bakes into their original scratch banks.
        dispatchPressureCell("preparePressureFullOracle");
        dispatch("bakeBrickAggregateEdges",
          Math.ceil(this.pressureCoarseEdgeCount / WORKGROUP_SIZE));
        dispatch("bakeBrickAggregateDiagonal", packed.brickCount);
        dispatch("bakePressureHierarchyEdges",
          Math.ceil(this.pressureHierarchyEdgeCount / WORKGROUP_SIZE));
        dispatch("bakePressureHierarchyDiagonal", this.pressureHierarchyGroupCount);
      }
      if (!this.pressureRefreshOracleForQA) {
        dispatchPressureCell("preparePressure");
      }
      closePass();
    });
    stage("pressure-rhs", () => {
      dispatchPressureCell("initializePCG");
      dispatchPressureBrickSolve("restrictBrickAggregateResidual");
      dispatchPressureBrickSolve("refineBrickAggregateAtoB1");
      dispatchPressureBrickSolve("refineBrickAggregateBtoA2");
      dispatchPressureBrickSolve("refineBrickAggregateAtoB3");
      dispatchPressureHierarchySolve("restrictPressureHierarchyResidual");
      dispatchPressureHierarchySolve("refinePressureHierarchyCorrection");
      dispatchPressureBrickSolve("combinePressureHierarchyCorrection");
      dispatchPressureCell("initializeBrickAggregateDirection");
      dispatch("reduceInitialize", 1);
      dispatchPressureCell("measureTrueResidual");
      dispatch("reduceInitialTrueResidual", 1);
      dispatchPressureCell("initializePipelinedImage");
      dispatch("reducePipelinedInitialize", 1);
      // Record 0 is the seed: the warm-started pressure and its true residual
      // before any iteration has touched them. The first correction is the
      // largest one in the solve, so a film that started at iteration 1 would
      // miss the only frame where the seed is visible.
      journalRecord(0);
    });
    stage("pressure-solve", () => {
      for (let iteration = 0;
        iteration < pressureIterations; iteration += 1) {
          dispatchPressureCell("updatePipelinedState");
          dispatchPressureBrickSolve("restrictBrickAggregateResidual");
          dispatchPressureBrickSolve("refineBrickAggregateAtoB1");
          dispatchPressureBrickSolve("refineBrickAggregateBtoA2");
          dispatchPressureBrickSolve("refineBrickAggregateAtoB3");
          dispatchPressureHierarchySolve("restrictPressureHierarchyResidual");
          dispatchPressureHierarchySolve("refinePressureHierarchyCorrection");
          dispatchPressureBrickSolve("combinePressureHierarchyCorrection");
          dispatchPressureCell("applyBrickAggregatePreconditioner");
          dispatchPressureCell("applyPipelinedImage");
          dispatch("reducePipelinedIteration", 1);
          if ((iteration + 1) % SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE === 0
            && iteration + 1 < pressureIterations) {
            dispatchPressureCell("measureGuardedTrueResidual");
            dispatch("reduceGuardedTrueResidual", 1);
            dispatchPressureCell("restartPCGAfterCurvatureLoss");
            dispatchPressureBrickSolve("restrictBrickAggregateResidual");
            dispatchPressureBrickSolve("refineBrickAggregateAtoB1");
            dispatchPressureBrickSolve("refineBrickAggregateBtoA2");
            dispatchPressureBrickSolve("refineBrickAggregateAtoB3");
            dispatchPressureBrickSolve("refineBrickAggregateBtoA4");
            dispatchPressureBrickSolve("refineBrickAggregateAtoB5");
            dispatchPressureBrickSolve("refineBrickAggregateBtoA6");
            dispatchPressureBrickSolve("refineBrickAggregateAtoB7");
            dispatchPressureHierarchySolve("restrictPressureHierarchyResidual");
            dispatchPressureHierarchySolve("refinePressureHierarchyCorrection");
            dispatchPressureBrickSolve("combinePressureHierarchyCorrection");
            dispatchPressureCell("initializeBrickAggregateRecoveryDirection");
            dispatch("reduceCurvatureRecovery", 1);
            dispatchPressureCell("applyPipelinedRecovery");
            dispatch("reducePipelinedRecovery", 1);
          }
          // After the cadence guard, so a record carries the true-residual
          // check and the gate closure that its own iteration earned.
          journalRecord(iteration + 1);
      }
      // Always close the solve with a fresh b-Ap application. No convergence
      // or performance receipt may rely on the recursive residual.
      dispatchPressureCell("measureTrueResidual");
      dispatch("reduceFinalTrueResidual", 1);
    });
    stage("velocity-projection", () => {
      // Open the generation before final projected velocity is authored: the
      // collocation producer can then append only its local owner/row closure.
      useBindGroup(this.bindGroup);
      dispatch("advanceActivityClock", 1);
      dispatch("beginIncrementalActivity", 1);
      useBindGroup(this.pressureBindGroup);
      if (this.pressureRefreshOracleForQA) {
        // Construction-only legacy projection oracle. It deliberately keeps
        // the original one-bank write semantics: any source-bank mirroring or
        // sparse-row omission in production FPA must therefore appear in the
        // paired physical/activity/topology receipts.
        dispatchPressureRow("projectFaces");
      } else {
        dispatch("beginSparseCM12FaceProjectionAuthority", 1);
        closePass();
        encoder.copyBufferToBuffer(this.topologyArena,
          sparseCM12FaceProjectionBootstrapIndirectByteOffset(
            this.faceProjectionAuthorityLayout, "projection"),
          this.faceProjectionAuthorityIndirectArguments, 12, 12);
        dispatchFaceProjectionAuthority("seedSparseCM12FaceProjectionBootstrap", 1);
        dispatch("seedSparseCM12PreviousFaceProjectionLeaves",
          Math.max(1, Math.ceil(this.faceProjectionAuthorityLayout.projection.leafCount
            / WORKGROUP_SIZE)));
        dispatchPressureCell("markSparseCM12FaceProjectionFromPressure");
        dispatchFaceProjectionAuthority("seedSparseCM12ProjectionFromPreparation", 4);
        dispatch("finalizeSparseCM12FaceProjectionFrontier", 1);
        closePass();
        encoder.copyBufferToBuffer(this.topologyArena,
          sparseCM12FaceProjectionIndirectByteOffset(
            this.faceProjectionAuthorityLayout, "projection", "repair"),
          this.faceProjectionAuthorityIndirectArguments, 36, 12);
        dispatchFaceProjectionAuthority("repairSparseCM12FaceProjectionLeaves", 3);
        dispatch("finalizeSparseCM12FaceProjectionPlan", 1);
        closePass();
        encoder.copyBufferToBuffer(this.topologyArena,
          sparseCM12FaceProjectionIndirectByteOffset(
            this.faceProjectionAuthorityLayout, "projection", "work"),
          this.faceProjectionAuthorityIndirectArguments, 60, 12);
        encoder.copyBufferToBuffer(this.topologyArena,
          sparseCM12FaceProjectionBootstrapIndirectByteOffset(
            this.faceProjectionAuthorityLayout, "projection"),
          this.faceProjectionAuthorityIndirectArguments, 84, 12);
        dispatchFaceProjectionAuthority("executeSparseCM12FaceProjection", 5);
        dispatchFaceProjectionAuthority("verifySparseCM12FaceProjectionLeaves", 7);
        dispatch("finalizeSparseCM12FaceProjectionExecution", 1);
      }
      useBindGroup(this.bindGroup);
      dispatchAccepted("collocateAndDiagnose", "cell");
      if (this.legacyHostAuthorityOracleForQA) {
        if (this.legacyHostD4AuthorityForQA) {
          dispatchAccepted("preserveVelocityHorizontalD4", "cell");
          dispatchAccepted("commitVelocityHorizontalD4", "cell");
        }
      } else {
        dispatchFrameControl("preserveVelocityHorizontalD4",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Work);
        dispatchFrameControl("commitVelocityHorizontalD4",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Work);
        dispatchFrameControl("sparseCM12FrameControlNoop",
          SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Bypass);
      }
      if (this.rigidCoupling) {
        pass?.end();
        pass = undefined;
        this.rigidCoupling.encodeReaction(encoder);
      }
      useBindGroup(this.bindGroup);
      dispatch("publishSparseCM12FrameFaceOutput", 1);
    });
    stage("projection-diagnostics", () => {
      dispatchAccepted("measureDivergenceDiagnostics", "cell");
      dispatch("reduceDivergenceDiagnostics", 1);
    });
    stage("activity-measurement", () => {
      useBindGroup(this.bindGroup);
      dispatchTemporalCell("markIncrementalActivityTemporalCells");
      dispatch("markIncrementalActivityTopology",
        Math.ceil(packed.brickCount / WORKGROUP_SIZE));
      dispatch("finalizeIncrementalActivityWorklist", 1);
      closePass();
      encoder.copyBufferToBuffer(this.activity,
        4 * (this.incrementalActivityLayout.headerBaseWords + 8),
        this.activityIndirectArguments, 0, 12);
      dispatchActivity("measureBrickActivity");
      dispatch("ageIncrementalActivityHistory",
        Math.ceil(packed.brickCount / WORKGROUP_SIZE));
      dispatch("finalizeIncrementalActivityCensus", 1);
    });
    stage("resolution-planning", () => {
      dispatch("planBrickResolution", bricks);
      dispatch("activateSweptReceivers", bricks);
      for (let gradingPass = 0;
        gradingPass < Math.log2(this.brickFineResolution); gradingPass += 1) {
        dispatch("closePlannedResolution", bricks);
      }
      dispatch("validateCandidateResolution", bricks);
      dispatch("scheduleTopologyPreparation", 1);
      dispatch("allocateCandidateTopologyPages", bricks);
      dispatch("synthesizeCandidateCellPages", packed.brickCount);
      dispatch("deferDynamicTopologyPublication", 1);
      dispatch("beginShadowTopology", 1);
      dispatch("buildShadowCellWorklist",
        Math.ceil(this.templateCellCount / WORKGROUP_SIZE));
      dispatch("buildShadowRowWorklist",
        Math.ceil(this.templateRowCount / WORKGROUP_SIZE));
      dispatch("finalizeShadowWorklists", 1);
    });
    stage("candidate-transfer", () => {
      dispatch("transferCandidateCells", packed.brickCount);
      dispatch("prepareCandidateFaceReceipts", bricks);
      dispatch("transferCandidateFaces", packed.brickCount, 6);
      dispatch("writeCandidateCellsToShadow", packed.brickCount);
      dispatch("reconstructShadowFaces",
        Math.ceil(this.templateRowCount / WORKGROUP_SIZE));
      dispatch("validateAndCommitShadowTopology", 1);
      dispatch("publishSparseCM12TopologyVelocityRoots", packed.brickCount);
    });
    stage("brick-retirement", () => {
      dispatch("retireUnsupportedEmptyBricks", bricks);
      dispatch("sealSparseCM12PressureTopologyJournal", 1);
      // Candidate commit/retirement occurs after the activity census. Re-open
      // only those changed leaf spans for presentation and next-frame reuse.
      dispatch("markIncrementalActivityPostTopology",
        Math.ceil(packed.brickCount / WORKGROUP_SIZE));
      // Post-census topology commits can append bricks to the same generation.
      // Republish the exact final count and snapshot its indirect triplet now;
      // otherwise next frame's FPA dispatch uses the pre-commit count while
      // its fail-closed expected receipt reads the larger post-commit count.
      dispatch("finalizeIncrementalActivityWorklist", 1);
      closePass();
      encoder.copyBufferToBuffer(this.activity,
        4 * (this.incrementalActivityLayout.headerBaseWords + 8),
        this.activityIndirectArguments, 0, 12);
    });
    stage("presentation-publication", () => {
      // The plan and compact page count are GPU publications. Split at the
      // storage-to-indirect copy seam; no host parity/count controls this path.
      closePass();
      if (this.legacyHostAuthorityOracleForQA) {
        useBindGroup(this.pressureBindGroup);
        dispatch("finalizeLegacyVelocityExtensionClockForQA", 1);
        closePass();
      } else {
        this.encodeVelocityExtensionPlan(encoder, "Sparse CM12 next-frame VEX plan");
      }
      this.encodeFramePlanPresentation(encoder, "Sparse CM12 frame presentation");
      dispatch("commitSparseCM12FrameControl", 1);
    });
    closePass();
    if (this.legacyHostAuthorityOracleForQA) this.legacyHostAuthorityParityForQA ^= 1;
    // The commit above authors next frame's accepted workgroup triplets. Keep
    // the writable arena out of indirect-dispatch synchronization scopes by
    // snapshotting those six words with a device-side copy between passes.
    encoder.copyBufferToBuffer(this.topologyArena,
      this.topologyWorklistBaseBytes + 4 * 8,
      this.acceptedIndirectArguments, 0, 6 * 4);
    seams?.anchorFinalBoundary?.(this.acceptedIndirectArguments, 0);
  }

  /**
   * Turn the marker view on or off.
   *
   * Off is free rather than cheap: the advance encodes no tracer dispatch at
   * all, so the feature costs one host branch when nobody is looking at it.
   *
   * Every off-to-on transition re-seeds. Advancing markers nobody can see would
   * be paying for the view while it is hidden, and showing markers that stood
   * still while it was hidden would be worse than either — so enabling instead
   * colours the liquid as it is at that moment. Re-seeding a running scene is a
   * feature, not a compromise: "where does this go from here" is the question
   * most of the time, and only a t=0 seed can answer "where did this come from".
   */
  setTracersEnabled(enabled: boolean): void {
    if (enabled === this.tracersEnabled) return;
    this.tracersEnabled = enabled;
    if (enabled) this.tracerSeedPending = true;
  }

  /** Re-read the mixing from this instant, without toggling the view. */
  reseedTracers(): void {
    if (this.tracersEnabled) this.tracerSeedPending = true;
  }

  /** The journal region this solver reserved. `floatCount` 0 means none. */
  get pressureJournalLayout(): SparseCM12PressureJournalLayout {
    return this.layout.journalLayout;
  }

  /**
   * Capture the next advance's pressure solve, or stop capturing.
   *
   * Returns whether the request could be honoured: a solver built without a
   * reserved journal cannot capture, and says so rather than arming a flag
   * that would encode nothing and read back zeros.
   */
  armPressureJournal(armed: boolean): boolean {
    if (armed && this.layout.journal === 0) return false;
    this.journalArmed = armed;
    return true;
  }

  get pressureJournalArmed(): boolean {
    return this.journalArmed;
  }

  /**
   * Where the captured fields are, for a view to read on the device.
   *
   * The snapshots deliberately never come back to the host. They are the large
   * part of the capture — four fields per snapshot over every accepted cell —
   * and a scrubber that mapped them would stall on every frame of the film for
   * data the draw could read where it already sits. Only the records below,
   * which are kilobytes, are read back.
   */
  get pressureJournalSource(): GPUPressureJournalSource | undefined {
    if (this.layout.journal === 0) return undefined;
    return {
      state: this.state,
      topologyArena: this.topologyArena,
      journalFloatOffset: this.layout.journal,
      layout: this.layout.journalLayout,
      snapshotCount: this.journalSnapshotCount,
      liquidFloatOffset: this.layout.liquid,
      cellCount: this.templateCellCount,
      domainFine: this.dimensions,
      // The width the last advance was parameterised with, as the face
      // velocity source reads it: it is the lattice-to-metres scale, and a
      // captured frame is always the last one advanced.
      finestCell_m: this.parameterF32[41] ?? 0,
    };
  }

  /**
   * Read back the header and iteration records of the last captured advance.
   *
   * QA and panel path only, never frame scheduling: it maps a buffer and
   * therefore stalls. Returns undefined when nothing has been captured, so a
   * panel opened before the first armed frame shows "no capture" rather than a
   * film of a hundred and twenty-eight zeroed iterations.
   */
  async readPressureJournal(): Promise<SparseCM12PressureJournal | undefined> {
    this.assertLive();
    const layout = this.layout.journalLayout;
    if (this.layout.journal === 0) return undefined;
    const floats = SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS
      + layout.iterationCapacity * SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 pressure journal readback",
      size: 4 * floats,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 pressure journal readback" });
      encoder.copyBufferToBuffer(this.state, 4 * this.layout.journal,
        readback, 0, 4 * floats);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange()).slice();
      const journal = decodeSparseCM12PressureJournal(values, layout);
      return journal.armed ? journal : undefined;
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /**
   * Where the finished frame's face velocities are, for a view to read.
   *
   * Rebuilt on every access rather than cached, because the two banks swap
   * with parity: `advanceTo` flips it on completion, so the bank the *next*
   * step will read as its source is the one the finished step accepted.
   *
   * There is no enable flag to pair with this. Everything it names already
   * exists because the solve needs it, so a reader costs the solver nothing —
   * unlike the markers, which are only advected when someone is looking.
   */
  get faceVelocitySource(): GPUFluidFaceVelocitySource {
    if (this.legacyHostAuthorityOracleForQA) return {
      state: this.state,
      topologyArena: this.topologyArena,
      faceFloatOffset: this.legacyHostAuthorityParityForQA !== 0
        ? this.layout.faceB : this.layout.faceA,
      rowCount: this.rowCount,
      domainFine: this.dimensions,
      finestCell_m: this.parameterF32[41] ?? 0,
    };
    return {
      state: this.state,
      topologyArena: this.topologyArena,
      faceFloatOffset: this.layout.faceA,
      parityWordOffset: this.frameControlLayout.baseWords
        + SPARSE_CM12_FRAME_CONTROL_HEADER.faceParity,
      rowCount: this.rowCount,
      domainFine: this.dimensions,
      // The width the last advance was actually parameterised with, which is
      // the unit its face velocities are stored in.
      finestCell_m: this.parameterF32[41] ?? 0,
    };
  }

  private encodeFramePlanPresentation(
    encoder: GPUCommandEncoder,
    label: string,
  ): void {
    if (this.presentationPublisherOracleForQA) {
      const oracle = encoder.beginComputePass({ label: `${label} QA publisher oracle` });
      oracle.setBindGroup(0, this.bindGroup);
      oracle.setPipeline(this.pipelines.publishSparseLevelSet!);
      oracle.dispatchWorkgroups(this.globalFineLevelSetSource.plan.maximumResidentBricks);
      oracle.end();
      return;
    }
    const bricks = this.lastPacked!.brickCount;
    const plan = encoder.beginComputePass({ label: `${label} FPL1/FPP1 plan` });
    plan.setBindGroup(0, this.bindGroup);
    const dispatch = (name: string, x: number) => {
      plan.setPipeline(this.pipelines[name]!);plan.dispatchWorkgroups(x);
    };
    dispatch("beginSparseCM12FramePlanNext", 1);
    dispatch("initializeSparseCM12FramePlanNext", bricks);
    dispatch("populateSparseCM12PresentationFramePlan", bricks);
    plan.setPipeline(this.pipelines.importVelocityExtensionBlastToFramePlanNext!);
    plan.dispatchWorkgroupsIndirect(this.vexActivityIndirectArguments,
      4 * SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT.vexSerial.offsetWords);
    dispatch("resolveSparseCM12FramePlanNextClosure", bricks);
    dispatch("sealSparseCM12FramePlanNextBricks", bricks);
    dispatch("finalizeSparseCM12FramePlanNext", 1);
    dispatch("markSparseCM12GlobalFramePlanReceipts", bricks);
    dispatch("beginSparseCM12FramePlanPresentationPacket", 1);
    dispatch("buildSparseCM12FramePlanPresentationPacket", bricks);
    dispatch("finalizeSparseCM12FramePlanPresentationPacket", 1);
    plan.end();
    encoder.copyBufferToBuffer(this.activity,
      this.framePlanLayout.fixedIndirectBinding.offset,
      this.framePlanIndirectArguments, 0,
      this.framePlanLayout.fixedIndirectBinding.size);
    encoder.copyBufferToBuffer(this.activity,
      this.framePlanPresentationLayout.indirectBinding.offset,
      this.presentationIndirectArguments, 0, 12);
    const execute = encoder.beginComputePass({ label: `${label} FPP1 execute` });
    execute.setBindGroup(0, this.bindGroup);
    execute.setPipeline(this.pipelines.executeSparseCM12FramePlanPresentationPacket!);
    execute.dispatchWorkgroupsIndirect(this.presentationIndirectArguments, 0);
    execute.setPipeline(this.pipelines.commitSparseCM12FramePlanPresentationPacket!);
    execute.dispatchWorkgroupsIndirect(this.presentationIndirectArguments, 0);
    execute.setPipeline(this.pipelines.verifySparseCM12FramePlanPresentation!);
    execute.dispatchWorkgroups(bricks);
    execute.setPipeline(this.pipelines.finalizeSparseCM12FramePlanPresentationExecution!);
    execute.dispatchWorkgroups(1);
    execute.setPipeline(this.pipelines.rejectSparseCM12FramePlanPresentationFaults!);
    execute.dispatchWorkgroups(bricks);
    execute.end();
  }

  /** Seal the producer-owned root queue and publish the exact next-frame blast. */
  private encodeVelocityExtensionPlan(
    encoder: GPUCommandEncoder,
    label: string,
  ): void {
    let pass: GPUComputePassEncoder | undefined;
    const openPass = () => {
      if (!pass) {
        pass = encoder.beginComputePass({ label });
        pass.setBindGroup(0, this.pressureBindGroup);
      }
      return pass;
    };
    const closePass = () => { pass?.end();pass = undefined; };
    const dispatch = (name: string, count = 1) => {
      const active = openPass();
      active.setPipeline(this.pipelines[name]!);
      active.dispatchWorkgroups(count);
    };
    const dispatchIndirect = (name: string) => {
      const active = openPass();
      active.setPipeline(this.pipelines[name]!);
      active.dispatchWorkgroupsIndirect(this.vexActivityIndirectArguments,
        4 * SPARSE_CM12_VEX_ACTIVITY_BATCH_INDIRECT.vexSerial.offsetWords);
    };
    const copy = (id: string) => {
      const receipt = this.vexActivityIndirectCopies.get(id);
      if (!receipt) throw new Error(`unknown VEX1 indirect copy ${id}`);
      closePass();
      encoder.copyBufferToBuffer(this.activity, 4 * receipt.sourceWord,
        this.vexActivityIndirectArguments, 4 * receipt.destinationWord, 12);
    };
    dispatch("beginVelocityExtensionCandidate");
    dispatch("sealVelocityExtensionRoots");
    copy("copy-vex-roots");
    dispatchIndirect("seedVelocityExtensionRoots");
    dispatch("sealVelocityExtensionSeedFrontier");
    for (let depth = 1; depth <= 8; depth += 1) {
      dispatch(`prepareVelocityExtensionFrontier${depth}`);
      copy(depth % 2 === 1 ? "copy-vex-frontier-a" : "copy-vex-frontier-b");
      dispatchIndirect(`expandVelocityExtensionFrontier${depth}`);
      dispatch(`sealVelocityExtensionFrontier${depth}`);
    }
    dispatch("finalizeVelocityExtensionBlast");
    copy("copy-vex-blast");
    closePass();
  }

  /** Publish generation zero without executing a physics step or mapping state. */
  encodeInitialPresentation(encoder: GPUCommandEncoder, finestCellSize_m: number): void {
    this.assertLive();
    this.writeParameters(this.lastPacked!, 0.004, finestCellSize_m, 1, [0, 0, 0]);
    const pass = encoder.beginComputePass({ label: "Sparse CM12 resident initial presentation" });
    pass.setPipeline(this.pipelines.classifyPresentationBricks!);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.lastPacked!.brickCount / WORKGROUP_SIZE));
    // Construction is the only full VEX root publication. The initial VEX
    // header is already collecting FCA generation 1, so the recurring frame
    // schedule contains no bootstrap branch or accepted-domain dispatch.
    pass.setPipeline(this.pipelines.bootstrapVelocityExtensionRoots!);
    pass.dispatchWorkgroupsIndirect(this.acceptedIndirectArguments, 0);
    pass.end();
    if (!this.legacyHostAuthorityOracleForQA) {
      this.encodeVelocityExtensionPlan(encoder, "Sparse CM12 construction VEX plan");
    }
    this.encodeFramePlanPresentation(encoder, "Sparse CM12 initial presentation");
  }

  encodeLiquidInjection(
    encoder: GPUCommandEncoder,
    finestCellSize_m: number,
    centerFine: readonly [number, number, number],
    radiusFine: readonly [number, number, number],
  ): void {
    this.assertLive();
    if (this.legacyHostAuthorityOracleForQA
      && (Math.abs(centerFine[0] - 0.5 * this.dimensions[0]) > 1e-6
        || Math.abs(centerFine[2] - 0.5 * this.dimensions[2]) > 1e-6
        || Math.abs(radiusFine[0] - radiusFine[2]) > 1e-6)) {
      this.legacyHostD4AuthorityForQA = false;
    }
    this.writeParameters(this.lastPacked!, 0.004, finestCellSize_m, 1, [0, 0, 0]);
    // The trailing one is the injection enable that every ordinary frame writes
    // as zero. It is what lets `activateSweptReceivers` and `injectLiquid` read
    // the drop out of the shared frame uniform without costing a quiescent
    // frame anything.
    this.parameterF32.set([...centerFine, 1], 52);
    this.parameterF32.set([...radiusFine, 0], 56);
    this.device.queue.writeBuffer(this.parameters, 0, this.parameterWords);
    const packed = this.lastPacked!;
    const bricks = Math.ceil(packed.brickCount / WORKGROUP_SIZE);
    const topologyPass = encoder.beginComputePass({
      label: "Sparse CM12 resident liquid injection topology",
    });
    topologyPass.setBindGroup(0, this.bindGroup);
    const dispatchTopology = (name: string, count: number, y = 1, z = 1) => {
      topologyPass.setPipeline(this.pipelines[name]!);
      topologyPass.dispatchWorkgroups(count, y, z);
    };
    if (!this.legacyHostAuthorityOracleForQA) {
      dispatchTopology("reopenVelocityExtensionPlanForInjection", 1);
    }
    // Promote every intersected brick before writing any density. The planner
    // treats the enabled injection as refine-only: untouched accepted bricks
    // are preserved, while closure may still grow the required 2:1 support.
    dispatchTopology("planBrickResolution", bricks);
    dispatchTopology("activateSweptReceivers", bricks);
    for (let gradingPass = 0;
      gradingPass < Math.log2(this.brickFineResolution); gradingPass += 1) {
      dispatchTopology("closePlannedResolution", bricks);
    }
    dispatchTopology("validateCandidateResolution", bricks);
    dispatchTopology("scheduleTopologyPreparation", 1);
    dispatchTopology("allocateCandidateTopologyPages", bricks);
    dispatchTopology("synthesizeCandidateCellPages", packed.brickCount);
    dispatchTopology("deferDynamicTopologyPublication", 1);
    dispatchTopology("beginShadowTopology", 1);
    dispatchTopology("buildShadowCellWorklist",
      Math.ceil(this.templateCellCount / WORKGROUP_SIZE));
    dispatchTopology("buildShadowRowWorklist",
      Math.ceil(this.templateRowCount / WORKGROUP_SIZE));
    dispatchTopology("finalizeShadowWorklists", 1);
    dispatchTopology("transferCandidateCells", packed.brickCount);
    dispatchTopology("prepareCandidateFaceReceipts", bricks);
    dispatchTopology("transferCandidateFaces", packed.brickCount, 6);
    dispatchTopology("writeCandidateCellsToShadow", packed.brickCount);
    dispatchTopology("reconstructShadowFaces",
      Math.ceil(this.templateRowCount / WORKGROUP_SIZE));
    dispatchTopology("validateAndCommitShadowTopology", 1);
    if (!this.legacyHostAuthorityOracleForQA) {
      dispatchTopology("publishSparseCM12TopologyVelocityRoots", packed.brickCount);
    }
    topologyPass.end();
    encoder.copyBufferToBuffer(this.topologyArena,
      this.topologyWorklistBaseBytes + 4 * 8,
      this.acceptedIndirectArguments, 0, 6 * 4);

    const injectionPass = encoder.beginComputePass({
      label: "Sparse CM12 resident liquid injection",
    });
    injectionPass.setBindGroup(0, this.bindGroup);
    // Brick-indexed rather than indirect over the accepted cell worklist: this
    // also covers a newly activated receiver in the same command buffer.
    injectionPass.setPipeline(this.pipelines.injectLiquid!);
    injectionPass.dispatchWorkgroups(bricks);
    injectionPass.setPipeline(this.pipelines.invalidateSparseCM12FrameD4ForInjection!);
    injectionPass.dispatchWorkgroups(1);
    injectionPass.setPipeline(this.pipelines.classifyPresentationBricks!);
    injectionPass.dispatchWorkgroups(bricks);
    injectionPass.end();
    if (!this.legacyHostAuthorityOracleForQA) {
      this.encodeVelocityExtensionPlan(encoder, "Sparse CM12 injection VEX replan");
    }
    this.encodeFramePlanPresentation(encoder, "Sparse CM12 injection presentation");
  }

  private lastPacked?: PackedResidentTopology;

  private writeParameters(
    packed: PackedResidentTopology,
    dt_s: number,
    finestCellSize_m: number,
    pressureScale: number,
    acceleration: readonly [number, number, number],
    sharpening?: SharpeningTrace,
    activityPolicy?: SparseCM12ActivityPolicy,
    pressureControl?: SparseCM12PressureControl,
    bodyCount = 0,
    worldDimensions_m?: readonly [number, number, number],
  ): void {
    this.lastPacked = packed;
    const u = this.parameterU32, f = this.parameterF32, l = this.layout;
    u.fill(0);
    u.set([this.cellCount, this.rowCount, packed.incidenceCount,
      this.dimensions[0] * this.dimensions[1] * this.dimensions[2]], 0);
    u.set([...this.dimensions,
      (this.boundary ? 1 : 0) | (this.brickFineResolution << 1)], 4);
    u.set([packed.cellOffset, packed.rowOffset, packed.termOffset, packed.incidenceOffset], 8);
    u.set([packed.incidenceRecordOffset, packed.brickLookupOffset,
      packed.brickOffset, packed.backgroundOwnerOffset], 12);
    u.set([l.densityA, l.densityB, l.gammaA, l.gammaB], 16);
    u.set([l.cellVelocityA, l.cellVelocityB, l.faceA, l.faceB], 20);
    u.set([l.pressure, l.rhs, l.diagonal, l.liquid], 24);
    u.set([l.theta, l.residual, l.preconditioned, l.direction], 28);
    // stateOffsets4.w is a static pointer to FCA1 in topologyArena. Consumers
    // read accepted parity there; the host never mirrors or predicts it.
    u.set([l.applied, l.divergence, 0x4643_4131,
      this.frameControlLayout.baseWords + SPARSE_CM12_FRAME_CONTROL_HEADER.scalarParity], 32);
    // The D4 pass needs two disjoint scalar scratch arrays. In particular the
    // gamma scratch must never alias densityA at offset zero: doing so corrupts
    // gamma after the first symmetric frame and makes transport create mass on
    // the next frame.
    u.set([l.sharpeningDelta, l.symmetryGamma, l.tracers, 0], 36);
    f.set([dt_s, finestCellSize_m, pressureScale, 0], 40);
    f.set([...acceleration, 0], 44);
    u.set([Math.ceil(this.cellCount / WORKGROUP_SIZE),
      Math.ceil(this.rowCount / WORKGROUP_SIZE),
      sparseCM12PressureIterations(pressureControl?.iterations),
      packed.brickCount], 48);
    f.set([0, 0, 0, 0, 1, 1, 1, 0], 52);
    const policy = sparseCM12ActivityPolicy(activityPolicy ?? {});
    // CM12 Algorithm 2's trace bounds plus the sparse residency density floor.
    // Direct diagnostic constructors pass no controls and get the production
    // defaults, so an unparameterized probe and a default panel run agree.
    f.set([
      sparseCM12SharpeningDistance(sharpening?.distanceCells),
      sparseCM12SharpeningTraceSteps(sharpening?.traceSteps),
      policy.residencyDensity, policy.residencyMassFineCells,
    ], 60);
    f.set([policy.finestTravelCells, policy.fourTravelCells,
      policy.twoTravelCells, policy.thinFeatureCells], 64);
    f.set([policy.thinFeatureDensity, policy.surfaceDensityMinimum,
      policy.surfaceDensityMaximum, policy.detailTolerance], 68);
    f.set([policy.frontLookaheadSteps, policy.promoteScore,
      policy.emergencyScore, policy.demoteScore], 72);
    u.set([policy.topologyCadenceSteps, policy.promoteEpochs,
      policy.demoteEpochs, policy.activitySignals ? 1 : 0], 76);
    f.set(this.boundary ? [...this.boundary.centerFine, 0] : [0, 0, 0, 0], 80);
    f.set(this.boundary ? [...this.boundary.radiiFine, 0] : [1, 1, 1, 0], 84);
    u.set([policy.prepareBricksPerFrame, 0, 0, 0], 88);
    f[89] = sparseCM12PressureRelativeTolerance(pressureControl?.relativeTolerance);
    u.set([l.solidCellOpen, l.solidRowData, 0, 0], 92);
    f.set([...(worldDimensions_m ?? [0, 0, 0]), bodyCount], 96);
    u.set([...this.tracerLattice.dimensions, this.tracerLattice.count], 100);
    f.set([...this.tracerLattice.originFine, this.tracerLattice.spacingFine], 104);
    const journal = this.layout.journalLayout;
    u.set([this.layout.journal, journal.iterationCapacity, journal.snapshotCapacity,
      journal.cellStride], 108);
    this.device.queue.writeBuffer(this.parameters, 0, this.parameterWords);
    if (this.legacyHostAuthorityOracleForQA) {
      this.device.queue.writeBuffer(this.topologyArena,
        4 * (this.frameControlLayout.baseWords
          + SPARSE_CM12_FRAME_CONTROL_HEADER.scalarParity),
        Uint32Array.of(this.legacyHostAuthorityParityForQA,
          this.legacyHostAuthorityParityForQA));
    }
  }

  async readDiagnostics(): Promise<{
    readonly pressureRelativeResidual: number;
    readonly pressureRecursiveRelativeResidual: number;
    readonly pressureTrueResidualMaximum: number;
    readonly pressureInitialTrueRelativeResidual: number;
    readonly pressureIterationsExecuted: number;
    readonly pressureIterationsEncoded: number;
    readonly pressureFirstToleranceCrossingIteration: number | undefined;
    readonly pressureSolveConverged: boolean;
    readonly pressureIterationCapReached: boolean;
    readonly pressureConvergenceReason: "tolerance" | "iteration-cap" | "fixed-budget";
    readonly pressureCurvatureBreakdown: boolean;
    readonly pressureCurvatureRecoveryCount: number;
    readonly pressureRecursiveToTrueResidualRatio: number;
    readonly pressureResidualDrift: boolean;
    readonly maximumDivergence_s: number;
    readonly maximumMixedSeamDivergence_s: number;
    readonly activityMaximumScore: number;
    readonly activitySurfaceBrickCount: number;
    readonly activityHotBrickCount: number;
    readonly activityQuietBrickCount: number;
    readonly activityTopologyEpoch: boolean;
    readonly activityMeasuredBrickCount: number;
    readonly activeBrickCount: number;
    readonly newlyActivatedBrickCount: number;
    readonly residencyGeneration: number;
    readonly activeCellCount: number;
    readonly acceptedTopologyGeneration: number;
    readonly topologyUrgentQueuedBrickCount: number;
    readonly topologyOrdinaryQueuedBrickCount: number;
    readonly topologyPreparedBrickCount: number;
    readonly topologyCommittedBrickCount: number;
    readonly topologyDeferredBrickCount: number;
    readonly acceptedFineBrickCount: number;
    readonly acceptedCoarseBrickCount: number;
    readonly acceptedCellCount: number;
    readonly acceptedRowCount: number;
    readonly acceptedSameLevelCoarseRowCount: number;
    readonly acceptedMixedSeamRowCount: number;
    readonly pressureActiveRowCount: number;
    readonly pressureCellCount: number;
    readonly pressureCanonicalMembership: {
      readonly cell: {
        readonly phase: number; readonly fault: number; readonly firstFault: number;
        readonly dirtyCount: number; readonly totalCount: number;
        readonly candidateGeneration: number; readonly acceptedGeneration: number;
      };
      readonly row: {
        readonly phase: number; readonly fault: number; readonly firstFault: number;
        readonly dirtyCount: number; readonly totalCount: number;
        readonly candidateGeneration: number; readonly acceptedGeneration: number;
      };
    };
    readonly pressureTopologyRepair: {
      readonly phase: number; readonly fault: number;
      readonly firstFaultFamily: number; readonly firstFaultId: number;
      readonly candidateGeneration: number; readonly acceptedGeneration: number;
      readonly topologyGeneration: number;
      readonly changedBrickCount: number; readonly changedRowCount: number;
      readonly cellExecutionCount: number; readonly rowExecutionCount: number;
      readonly brickDirtyLeafCount: number; readonly rowDirtyLeafCount: number;
      readonly expectedProducerReceipts: number;
      readonly coveredProducerReceipts: number;
    };
    readonly pressureCutoverAuthorities: NonNullable<NonNullable<
      GPUEulerianInfo["adaptivePressureTopologyAttribution"]>["authorities"]>;
    readonly temporalScalarCellCount: number;
    readonly temporalScalarRowCount: number;
    readonly temporalScalarRejectionMask: number;
  }> {
    this.assertLive();
    const encoder = this.device.createCommandEncoder({
      label: "Sparse CM12 diagnostic scalar readback",
    });
    encoder.copyBufferToBuffer(this.scalars, 0, this.diagnosticsReadback, 0,
      SPARSE_CM12_PRESSURE_SCALAR_BYTES);
    encoder.copyBufferToBuffer(this.activity, 0, this.diagnosticsReadback,
      SPARSE_CM12_PRESSURE_SCALAR_BYTES,
      4 * ACTIVITY_HEADER_WORDS);
    encoder.copyBufferToBuffer(this.topologyArena,
      this.topologyWorklistBaseBytes + 4 * 4,
      this.diagnosticsReadback,
      SPARSE_CM12_PRESSURE_SCALAR_BYTES + 4 * ACTIVITY_HEADER_WORDS, 8);
    const pcmDiagnosticOffset = SPARSE_CM12_PRESSURE_SCALAR_BYTES
      + 4 * ACTIVITY_HEADER_WORDS + 12;
    encoder.copyBufferToBuffer(this.activity,
      4 * this.canonicalMembershipLayout.cell.headerBaseWords,
      this.diagnosticsReadback, pcmDiagnosticOffset,
      4 * SPARSE_CM12_PCM_DIAGNOSTIC_DOMAIN_WORDS);
    encoder.copyBufferToBuffer(this.activity,
      4 * this.canonicalMembershipLayout.row.headerBaseWords,
      this.diagnosticsReadback,
      pcmDiagnosticOffset + 4 * SPARSE_CM12_PCM_DIAGNOSTIC_DOMAIN_WORDS,
      4 * SPARSE_CM12_PCM_DIAGNOSTIC_DOMAIN_WORDS);
    const ptrDiagnosticOffset = pcmDiagnosticOffset
      + SPARSE_CM12_PCM_DIAGNOSTIC_BYTES;
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.pressureTopologyRepairLayout.baseWords,
      this.diagnosticsReadback, ptrDiagnosticOffset,
      4 * SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS);
    const fpaDiagnosticOffset = ptrDiagnosticOffset
      + 4 * SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS;
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.faceProjectionAuthorityLayout.preparation.headerBaseWords,
      this.diagnosticsReadback, fpaDiagnosticOffset,
      4 * SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS);
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.faceProjectionAuthorityLayout.projection.headerBaseWords,
      this.diagnosticsReadback,
      fpaDiagnosticOffset + 4 * SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS,
      4 * SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS);
    const pcfDiagnosticOffset = fpaDiagnosticOffset
      + 8 * SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS;
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.persistentPressureCacheLayout.headerBaseWords,
      this.diagnosticsReadback, pcfDiagnosticOffset,
      4 * SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS);
    const pcaDiagnosticOffset = pcfDiagnosticOffset
      + 4 * SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS;
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.persistentPressureCacheLayout.aggregateHeaderBaseWords,
      this.diagnosticsReadback, pcaDiagnosticOffset,
      4 * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS);
    const persistentFamilies = ["brick", "aggregateEdge", "hierarchyNode",
      "hierarchyEdge"] as const;
    persistentFamilies.forEach((family, index) => {
      encoder.copyBufferToBuffer(this.topologyArena,
        4 * this.persistentPressureCacheLayout.aggregateFamilies[family].headerBaseWords,
        this.diagnosticsReadback,
        pcaDiagnosticOffset + 4 * (SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS
          + index * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS),
        4 * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS);
    });
    const pabDiagnosticOffset = pcaDiagnosticOffset + 4
      * (SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS
        + 4 * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS);
    encoder.copyBufferToBuffer(this.activity,
      4 * this.pressureAddressingABQA.layout.baseWords,
      this.diagnosticsReadback, pabDiagnosticOffset,
      4 * SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS);
    this.device.queue.submit([encoder.finish()]);
    await this.diagnosticsReadback.mapAsync(GPUMapMode.READ);
    const mapped = this.diagnosticsReadback.getMappedRange();
    const values = new Float32Array(mapped, 0,
      SPARSE_CM12_PRESSURE_SCALAR_BYTES / 4);
    const activity = new Uint32Array(mapped,
      SPARSE_CM12_PRESSURE_SCALAR_BYTES, ACTIVITY_HEADER_WORDS);
    const acceptedCounts = new Uint32Array(mapped,
      SPARSE_CM12_PRESSURE_SCALAR_BYTES + 4 * ACTIVITY_HEADER_WORDS, 2);
    const pcmCell = new Uint32Array(mapped, pcmDiagnosticOffset,
      SPARSE_CM12_PCM_DIAGNOSTIC_DOMAIN_WORDS);
    const pcmRow = new Uint32Array(mapped,
      pcmDiagnosticOffset + 4 * SPARSE_CM12_PCM_DIAGNOSTIC_DOMAIN_WORDS,
      SPARSE_CM12_PCM_DIAGNOSTIC_DOMAIN_WORDS);
    const ptrHeader = new Uint32Array(mapped, ptrDiagnosticOffset,
      SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS);
    const fpaPreparationHeader = new Uint32Array(mapped, fpaDiagnosticOffset,
      SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS);
    const fpaProjectionHeader = new Uint32Array(mapped,
      fpaDiagnosticOffset + 4 * SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS,
      SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS);
    const pcfHeader = new Uint32Array(mapped, pcfDiagnosticOffset,
      SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS);
    const pcaHeader = new Uint32Array(mapped, pcaDiagnosticOffset,
      SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS);
    const pcaFamilies = persistentFamilies.map((_, index) => new Uint32Array(mapped,
      pcaDiagnosticOffset + 4 * (SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS
        + index * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS),
      SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS));
    const pressureAddressingReceipt = inspectSparseCM12PressureAddressingABReceipt(
      new Uint32Array(mapped, pabDiagnosticOffset,
        SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS),
      this.pressureAddressingABQA.layout,
      this.pressureAddressingABQA.layout.baseWords,
    );
    const pcmDomainReceipt = (words: Uint32Array) => {
      const h = SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER;
      return {
        phase: words[h.phase]!, fault: words[h.fault]!,
        firstFault: words[h.firstFaultId]!, dirtyCount: words[h.dirtyCount]!,
        totalCount: words[h.totalCount]!,
        candidateGeneration: words[h.candidateGeneration]!,
        acceptedGeneration: words[h.acceptedGeneration]!,
      };
    };
    const fpaStageReceipt = (words: Uint32Array) => {
      const h = SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER;
      return {
        acceptedGeneration: words[h.acceptedGeneration]!,
        candidateGeneration: words[h.candidateGeneration]!,
        topologyGeneration: words[h.topologyGeneration]!,
        directCount: words[h.directWriteCount]!,
        closureCount: words[h.closureWriteCount]!,
        dirtyCount: words[h.dirtyLeafCount]!,
        workCount: words[h.workCount]!, executedCount: words[h.executedCount]!,
        skippedCount: words[h.reusedCount]!,
        expectedProducerReceipts: words[h.expectedProducerReceipts]!,
        coveredProducerReceipts: words[h.coveredProducerReceipts]!,
        causeMask: words[h.causeMask]!, fault: words[h.fault]!,
        firstFaultId: words[h.firstFaultRow]!,
      };
    };
    const cache = SPARSE_CM12_PRESSURE_CACHE_HEADER;
    const aggregate = SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER;
    const aggregateFamily = SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER;
    const pcaDirty = pcaFamilies.map((words) => words[aggregateFamily.dirtyLeafCount]!);
    const pcaWork = pcaFamilies.map((words) => words[aggregateFamily.workCount]!);
    const pcaExecuted = pcaFamilies.map((words) => words[aggregateFamily.executedCount]!);
    const pcaCauses = pcaFamilies.reduce((mask, words) =>
      mask | words[aggregateFamily.causeMask]!, 0);
    const sum = (values: readonly number[]) => values.reduce((total, value) => total + value, 0);
    const pressureCacheTopologyGeneration = pcaHeader[aggregate.topologyGeneration]!;
    const pressureCacheReceipt = {
      acceptedGeneration: pcfHeader[cache.acceptedGeneration]!,
      candidateGeneration: pcfHeader[cache.candidateGeneration]!,
      topologyGeneration: pressureCacheTopologyGeneration,
      directCount: pcfHeader[cache.directEventCount]!,
      closureCount: pcfHeader[cache.closureWriteCount]!,
      dirtyCount: pcfHeader[cache.dirtyLeafCount]!,
      workCount: pcfHeader[cache.dirtyLeafCount]!,
      executedCount: pcfHeader[cache.repairedLeafCount]!,
      skippedCount: Math.max(0, pcfHeader[cache.dirtyLeafCount]!
        - pcfHeader[cache.repairedLeafCount]!),
      expectedProducerReceipts: pcfHeader[cache.expectedEventCount]!,
      coveredProducerReceipts: pcfHeader[cache.coveredEventCount]!,
      causeMask: pcfHeader[cache.causeMask]!, fault: pcfHeader[cache.fault]!,
      firstFaultId: pcfHeader[cache.firstFaultId]!,
    };
    const pressureAggregateReceipt = {
      acceptedGeneration: pressureCacheReceipt.acceptedGeneration,
      candidateGeneration: pressureCacheReceipt.candidateGeneration,
      topologyGeneration: pressureCacheTopologyGeneration,
      directCount: 0, closureCount: 0,
      dirtyCount: sum(pcaDirty), workCount: sum(pcaWork),
      executedCount: sum(pcaExecuted),
      skippedCount: Math.max(0, sum(pcaWork) - sum(pcaExecuted)),
      expectedProducerReceipts: pressureCacheReceipt.expectedProducerReceipts,
      coveredProducerReceipts: pressureCacheReceipt.coveredProducerReceipts,
      causeMask: pcaCauses, fault: pressureCacheReceipt.fault,
      firstFaultId: pressureCacheReceipt.fault === 0
        ? 0xffff_ffff : pcaHeader[aggregate.firstFaultId]!,
      familyDirtyCount: pcaDirty as [number, number, number, number],
      familyExecutedCount: pcaExecuted as [number, number, number, number],
    };
    const preparationReceipt = fpaStageReceipt(fpaPreparationHeader);
    const projectionReceipt = fpaStageReceipt(fpaProjectionHeader);
    const pressureAddressingReady = this.pressureAddressingABQA.mode === "canonicalRankSelect"
      || (pressureAddressingReceipt.phase
          === SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE.accepted
        && pressureAddressingReceipt.fault === 0
        && pressureAddressingReceipt.expectedPCMGeneration
          === pressureAddressingReceipt.materializedPCMGeneration
        && pressureAddressingReceipt.expectedCount
          === pressureAddressingReceipt.materializedCount
        && pressureAddressingReceipt.materializedExecutions
          === pressureAddressingReceipt.expectedCount);
    const pressureCutoverFault = preparationReceipt.fault !== 0
      || projectionReceipt.fault !== 0 || pressureCacheReceipt.fault !== 0
      || !pressureAddressingReady;
    const pressureCutoverAuthorities = {
      status: pressureCutoverFault ? "fault" as const : "matched" as const,
      inputTopologyGeneration: preparationReceipt.topologyGeneration,
      fpa: { preparation: preparationReceipt, projection: projectionReceipt },
      pcf: pressureCacheReceipt, pca: pressureAggregateReceipt,
      pressureAddressing: {
        ready: pressureAddressingReady,
        phase: pressureAddressingReceipt.phase,
        fault: pressureAddressingReceipt.fault,
        firstFaultRank: pressureAddressingReceipt.firstFaultRank,
        expectedPCMGeneration: pressureAddressingReceipt.expectedPCMGeneration,
        materializedPCMGeneration: pressureAddressingReceipt.materializedPCMGeneration,
        expectedCount: pressureAddressingReceipt.expectedCount,
        materializedCount: pressureAddressingReceipt.materializedCount,
      },
    };
    const rhsSquared = values[1]!;
    const recursiveResidualSquared = values[4]!;
    const initialTrueResidualSquared = values[8]!;
    const trueResidualSquared = values[10]!;
    const relative = (squared: number) => Math.sqrt(Math.max(0, squared)
      / Math.max(rhsSquared, Number.MIN_VALUE));
    const firstCrossing = Math.round(values[13]!);
    const pressureIterationsExecuted = Math.max(0, Math.round(values[12]!));
    const pressureIterationsEncoded = this.parameterU32[50]!;
    const fixedBudget = this.parameterF32[89]! <= 0;
    const pressureSolveConverged = !fixedBudget && firstCrossing >= 0;
    const pressureConvergenceReason: "tolerance" | "iteration-cap" | "fixed-budget" =
      fixedBudget ? "fixed-budget" : pressureSolveConverged ? "tolerance" : "iteration-cap";
    const result = {
      pressureRelativeResidual: relative(trueResidualSquared),
      pressureRecursiveRelativeResidual: relative(recursiveResidualSquared),
      pressureTrueResidualMaximum: values[11]!,
      pressureInitialTrueRelativeResidual: relative(initialTrueResidualSquared),
      pressureIterationsExecuted,
      pressureIterationsEncoded,
      pressureFirstToleranceCrossingIteration: firstCrossing >= 0
        ? firstCrossing : undefined,
      pressureSolveConverged,
      pressureIterationCapReached: !fixedBudget && !pressureSolveConverged
        && pressureIterationsExecuted >= pressureIterationsEncoded,
      pressureConvergenceReason,
      pressureCurvatureBreakdown: values[18]! > 0.5,
      pressureCurvatureRecoveryCount: Math.max(0, Math.round(values[18]!)),
      pressureRecursiveToTrueResidualRatio: values[16]!,
      pressureResidualDrift: values[17]! > 0.5,
      maximumDivergence_s: values[6]!,
      maximumMixedSeamDivergence_s: values[7]!,
      activityMaximumScore: activity[1]!,
      activitySurfaceBrickCount: activity[2]!,
      activityHotBrickCount: activity[3]!,
      activityQuietBrickCount: activity[4]!,
      activityTopologyEpoch: activity[5] !== 0,
      activityMeasuredBrickCount: activity[6]!,
      activeBrickCount: activity[8]!,
      newlyActivatedBrickCount: activity[9]!,
      residencyGeneration: activity[10]!,
      activeCellCount: activity[11]!,
      acceptedTopologyGeneration: activity[12]!,
      topologyUrgentQueuedBrickCount: activity[14]!,
      topologyOrdinaryQueuedBrickCount: activity[15]!,
      topologyPreparedBrickCount: activity[16]!,
      topologyCommittedBrickCount: activity[17]!,
      topologyDeferredBrickCount: activity[18]!,
      acceptedFineBrickCount: activity[19]!,
      acceptedCoarseBrickCount: activity[20]!,
      acceptedCellCount: acceptedCounts[0]!,
      acceptedRowCount: acceptedCounts[1]!,
      acceptedSameLevelCoarseRowCount: activity[ACCEPTED_COARSE_ROW_COUNT_WORD]!,
      acceptedMixedSeamRowCount: activity[ACCEPTED_MIXED_ROW_COUNT_WORD]!,
      pressureActiveRowCount: activity[PRESSURE_ACTIVE_ROW_COUNT_WORD]!,
      pressureCellCount: pcmCell[
        SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER.totalCount]!,
      pressureCanonicalMembership: {
        cell: pcmDomainReceipt(pcmCell), row: pcmDomainReceipt(pcmRow),
      },
      pressureTopologyRepair: {
        phase: ptrHeader[SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.phase]!,
        fault: ptrHeader[SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.fault]!,
        firstFaultFamily: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.firstFaultFamily]!,
        firstFaultId: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.firstFaultId]!,
        candidateGeneration: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.candidateGeneration]!,
        acceptedGeneration: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.acceptedGeneration]!,
        topologyGeneration: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.topologyGeneration]!,
        changedBrickCount: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.acceptedChangedBrickCount]!,
        changedRowCount: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.acceptedChangedRowCount]!,
        cellExecutionCount: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.acceptedCellExecutionCount]!,
        rowExecutionCount: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.acceptedRowExecutionCount]!,
        brickDirtyLeafCount: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.acceptedBrickDirtyLeafCount]!,
        rowDirtyLeafCount: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.acceptedRowDirtyLeafCount]!,
        expectedProducerReceipts: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.expectedProducerReceipts]!,
        coveredProducerReceipts: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.coveredProducerReceipts]!,
      },
      pressureCutoverAuthorities,
      temporalScalarCellCount: activity[25]!,
      temporalScalarRowCount: activity[26]!,
      temporalScalarRejectionMask: activity[27]! & 0x7fff_ffff,
    };
    this.diagnosticsReadback.unmap();
    return result;
  }

  /** QA-only PCM1 header and active-bitset receipt. This readback is never
   * consulted by frame encoding or any physics/worklist decision. */
  async readPressureCanonicalMembershipQA() {
    this.assertLive();
    const headerWords = SPARSE_CM12_PCM_DIAGNOSTIC_DOMAIN_WORDS;
    const cell = this.canonicalMembershipLayout.cell;
    const row = this.canonicalMembershipLayout.row;
    const cellHeaderAt = 0;
    const cellBitsAt = cellHeaderAt + headerWords;
    const rowHeaderAt = cellBitsAt + cell.activeBitWordCount;
    const rowBitsAt = rowHeaderAt + headerWords;
    const cellClassificationAt = rowBitsAt + row.activeBitWordCount;
    const rowClassificationAt = cellClassificationAt + cell.capacity;
    const coefficientAt = rowClassificationAt + row.capacity;
    const rhsAt = coefficientAt + this.pressureFineEdgeCount;
    const preparedAt = rhsAt + cell.capacity;
    const faceAAt = preparedAt + row.capacity;
    const faceBAt = faceAAt + row.capacity;
    const fpaHeaderAt = faceBAt + row.capacity;
    const preparationHeaderAt = fpaHeaderAt + SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS;
    const projectionHeaderAt = preparationHeaderAt
      + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS;
    const aggregateEdgeAt = projectionHeaderAt
      + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS;
    const brickDiagonalAt = aggregateEdgeAt
      + this.persistentPressureCacheLayout.aggregateEdgeCount;
    const hierarchyEdgeAt = brickDiagonalAt
      + this.persistentPressureCacheLayout.brickCount;
    const hierarchyDiagonalAt = hierarchyEdgeAt
      + this.persistentPressureCacheLayout.hierarchyEdgeCount;
    const wordCount = hierarchyDiagonalAt
      + this.persistentPressureCacheLayout.hierarchyNodeCount;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 PCM1 QA receipt",
      size: Math.max(4, 4 * wordCount),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Sparse CM12 PCM1 QA copy",
    });
    const copy = (sourceWords: number, targetWords: number, words: number) => {
      encoder.copyBufferToBuffer(this.activity, 4 * sourceWords, readback,
        4 * targetWords, 4 * words);
    };
    copy(cell.headerBaseWords, cellHeaderAt, headerWords);
    copy(cell.activeBitsBaseWords, cellBitsAt, cell.activeBitWordCount);
    copy(row.headerBaseWords, rowHeaderAt, headerWords);
    copy(row.activeBitsBaseWords, rowBitsAt, row.activeBitWordCount);
    encoder.copyBufferToBuffer(this.state, 4 * this.layout.liquid, readback,
      4 * cellClassificationAt, 4 * cell.capacity);
    encoder.copyBufferToBuffer(this.state, 4 * this.layout.theta, readback,
      4 * rowClassificationAt, 4 * row.capacity);
    // Hash the authority the production operator actually consumes. The old
    // candidate-state coefficient bank is scratch after PCF1 cutover.
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.persistentPressureCacheLayout.effectiveEdgeBaseWords, readback,
      4 * coefficientAt, 4 * this.pressureFineEdgeCount);
    encoder.copyBufferToBuffer(this.state, 4 * this.layout.rhs, readback,
      4 * rhsAt, 4 * cell.capacity);
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.faceProjectionAuthorityLayout.preparedAuthorityBaseWords, readback,
      4 * preparedAt, 4 * row.capacity);
    encoder.copyBufferToBuffer(this.state, 4 * this.layout.faceA, readback,
      4 * faceAAt, 4 * row.capacity);
    encoder.copyBufferToBuffer(this.state, 4 * this.layout.faceB, readback,
      4 * faceBAt, 4 * row.capacity);
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.faceProjectionAuthorityLayout.baseWords, readback,
      4 * fpaHeaderAt, 4 * SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS);
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.faceProjectionAuthorityLayout.preparation.headerBaseWords, readback,
      4 * preparationHeaderAt, 4 * SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS);
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.faceProjectionAuthorityLayout.projection.headerBaseWords, readback,
      4 * projectionHeaderAt, 4 * SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS);
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.persistentPressureCacheLayout.brickAggregateEdgeBaseWords, readback,
      4 * aggregateEdgeAt,
      4 * this.persistentPressureCacheLayout.aggregateEdgeCount);
    encoder.copyBufferToBuffer(this.topologyArena,
      4 * this.persistentPressureCacheLayout.brickAggregateDiagonalBaseWords, readback,
      4 * brickDiagonalAt, 4 * this.persistentPressureCacheLayout.brickCount);
    let hierarchyTarget = hierarchyEdgeAt;
    for (let level = 0;
      level < this.persistentPressureCacheLayout.hierarchyEdgeLevelCounts.length;
      level += 1) {
      const count = this.persistentPressureCacheLayout.hierarchyEdgeLevelCounts[level]!;
      encoder.copyBufferToBuffer(this.topologyArena,
        4 * this.persistentPressureCacheLayout.hierarchyEdgeBaseWords[level]!, readback,
        4 * hierarchyTarget, 4 * count);
      hierarchyTarget += count;
    }
    hierarchyTarget = hierarchyDiagonalAt;
    for (let level = 0;
      level < this.persistentPressureCacheLayout.hierarchyLevelCounts.length;
      level += 1) {
      const count = this.persistentPressureCacheLayout.hierarchyLevelCounts[level]!;
      encoder.copyBufferToBuffer(this.topologyArena,
        4 * this.persistentPressureCacheLayout.hierarchyDiagonalBaseWords[level]!, readback,
        4 * hierarchyTarget, 4 * count);
      hierarchyTarget += count;
    }
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = new Uint32Array(readback.getMappedRange());
      const h = SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER;
      const popcount = (value: number) => {
        let x = value >>> 0;
        x -= (x >>> 1) & 0x5555_5555;
        x = (x & 0x3333_3333) + ((x >>> 2) & 0x3333_3333);
        return (((x + (x >>> 4)) & 0x0f0f_0f0f) * 0x0101_0101) >>> 24;
      };
      const sha256 = async (words: Uint32Array) => {
        const bytes = new Uint8Array(words.byteLength);
        bytes.set(new Uint8Array(words.buffer, words.byteOffset, words.byteLength));
        const digest = new Uint8Array(await globalThis.crypto.subtle.digest(
          "SHA-256", bytes.buffer,
        ));
        return Array.from(digest,
          (byte) => byte.toString(16).padStart(2, "0")).join("");
      };
      const contains = (bitsAt: number, id: number) =>
        (mapped[bitsAt + (id >>> 5)]! & (1 << (id & 31))) !== 0;
      // Authority receipts deliberately exclude inactive stable slots. Those
      // slots retain scratch from older pressure epochs and are outside the
      // PCM domain; hashing them would turn harmless storage history into a
      // false local-vs-oracle physics failure. Stable ids are interleaved with
      // values so the digest still commits to both membership position and
      // exact f32 bits.
      const sha256ActiveValues = (valuesAt: number, capacity: number,
        bitsAt: number) => {
        const authority: number[] = [];
        for (let id = 0; id < capacity; id += 1) {
          if (contains(bitsAt, id)) authority.push(id, mapped[valuesAt + id]!);
        }
        return sha256(Uint32Array.from(authority));
      };
      const sha256ActiveCoefficients = () => {
        const authority: number[] = [];
        const edgeOffsets = this.templateWords[15]!;
        for (let cellId = 0; cellId < cell.capacity; cellId += 1) {
          if (!contains(cellBitsAt, cellId)) continue;
          const first = this.templateWords[edgeOffsets + cellId]!;
          const end = this.templateWords[edgeOffsets + cellId + 1]!;
          for (let edge = first; edge < end; edge += 1) {
            authority.push(edge, mapped[coefficientAt + edge]!);
          }
        }
        return sha256(Uint32Array.from(authority));
      };
      const receipt = async (headerAt: number, bitsAt: number, bitsWords: number,
        classificationAt: number, capacity: number, threshold: number) => {
        const header = mapped.slice(headerAt, headerAt + headerWords);
        const bits = mapped.slice(bitsAt, bitsAt + bitsWords);
        const classifications = new Float32Array(mapped.buffer,
          mapped.byteOffset + 4 * classificationAt, capacity);
        const classificationBits = new Uint32Array(bitsWords);
        for (let id = 0; id < capacity; id += 1) {
          if (classifications[id]! > threshold) {
            classificationBits[id >>> 5]! |= 1 << (id & 31);
          }
        }
        const activeBitsSha256 = await sha256(bits);
        const classificationBitsSha256 = await sha256(classificationBits);
        return {
          phase: header[h.phase]!, fault: header[h.fault]!,
          firstFault: header[h.firstFaultId]!, dirtyCount: header[h.dirtyCount]!,
          totalCount: header[h.totalCount]!,
          candidateGeneration: header[h.candidateGeneration]!,
          acceptedGeneration: header[h.acceptedGeneration]!,
          activeBitCount: bits.reduce((sum, word) => sum + popcount(word), 0),
          activeBitsSha256,
          classificationBitCount: classificationBits.reduce(
            (sum, word) => sum + popcount(word), 0),
          classificationBitsSha256,
          matchesClassification: activeBitsSha256 === classificationBitsSha256,
        };
      };
      const [cellReceipt, rowReceipt, thetaSha256, coefficientSha256, rhsSha256,
        facePreparationSha256, aggregateEdgeSha256, brickDiagonalSha256,
        hierarchyEdgeSha256, hierarchyDiagonalSha256,
        rawThetaSha256, rawCoefficientSha256, rawRhsSha256] =
        await Promise.all([
          receipt(cellHeaderAt, cellBitsAt, cell.activeBitWordCount,
          cellClassificationAt, cell.capacity, 0.5),
          receipt(rowHeaderAt, rowBitsAt, row.activeBitWordCount,
          rowClassificationAt, row.capacity, 0),
          sha256ActiveValues(rowClassificationAt, row.capacity, rowBitsAt),
          sha256ActiveCoefficients(),
          sha256ActiveValues(rhsAt, cell.capacity, cellBitsAt),
          sha256(mapped.slice(preparedAt, preparedAt + row.capacity)),
          sha256(mapped.slice(aggregateEdgeAt, brickDiagonalAt)),
          sha256(mapped.slice(brickDiagonalAt, hierarchyEdgeAt)),
          sha256(mapped.slice(hierarchyEdgeAt, hierarchyDiagonalAt)),
          sha256(mapped.slice(hierarchyDiagonalAt, wordCount)),
          sha256(mapped.slice(rowClassificationAt, rowClassificationAt + row.capacity)),
          sha256(mapped.slice(coefficientAt, coefficientAt + this.pressureFineEdgeCount)),
          sha256(mapped.slice(rhsAt, rhsAt + cell.capacity)),
        ]);
      const fpaStage = (at: number) => {
        const d = SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER;
        return {
          phase: mapped[at + d.phase]!, fault: mapped[at + d.fault]!,
          firstFaultRow: mapped[at + d.firstFaultRow]!,
          acceptedGeneration: mapped[at + d.acceptedGeneration]!,
          candidateGeneration: mapped[at + d.candidateGeneration]!,
          frameGeneration: mapped[at + d.frameGeneration]!,
          topologyGeneration: mapped[at + d.topologyGeneration]!,
          pcmGeneration: mapped[at + d.pcmGeneration]!,
          sourceParity: mapped[at + d.sourceParity]!,
          workCount: mapped[at + d.workCount]!,
          executedCount: mapped[at + d.executedCount]!,
        };
      };
      const result = {
        mode: this.pressureRefreshOracleForQA ? "full-refresh-oracle" : "local",
        cell: cellReceipt,
        row: rowReceipt,
        thetaSha256,
        coefficientSha256,
        rhsSha256,
        facePreparationSha256,
        aggregateEdgeSha256,
        brickDiagonalSha256,
        hierarchyEdgeSha256,
        hierarchyDiagonalSha256,
        faceAuthority: {
          firstFaultStage: mapped[fpaHeaderAt
            + SPARSE_CM12_FACE_PROJECTION_HEADER.firstFaultStage]!,
          preparation: fpaStage(preparationHeaderAt),
          projection: fpaStage(projectionHeaderAt),
        },
        rawThetaSha256,
        rawCoefficientSha256,
        rawRhsSha256,
      };
      Object.defineProperty(result, "qaRaw", { enumerable: false, value: {
        coefficientBits: mapped.slice(coefficientAt,
          coefficientAt + this.pressureFineEdgeCount),
        preparedBits: mapped.slice(preparedAt, preparedAt + row.capacity),
        faceABits: mapped.slice(faceAAt, faceAAt + row.capacity),
        faceBBits: mapped.slice(faceBAt, faceBAt + row.capacity),
        aggregateEdgeBits: mapped.slice(aggregateEdgeAt, brickDiagonalAt),
        brickDiagonalBits: mapped.slice(brickDiagonalAt, hierarchyEdgeAt),
        hierarchyEdgeBits: mapped.slice(hierarchyEdgeAt, hierarchyDiagonalAt),
        hierarchyDiagonalBits: mapped.slice(hierarchyDiagonalAt, wordCount),
      } });
      return result as typeof result & { readonly qaRaw: {
        readonly coefficientBits: Uint32Array; readonly preparedBits: Uint32Array;
        readonly faceABits: Uint32Array; readonly faceBBits: Uint32Array;
        readonly aggregateEdgeBits: Uint32Array; readonly brickDiagonalBits: Uint32Array;
        readonly hierarchyEdgeBits: Uint32Array; readonly hierarchyDiagonalBits: Uint32Array;
      } };
    } finally {
      readback.unmap();
      readback.destroy();
    }
  }

  /** QA-only dense readback used by Dawn physics gates. Never called by frame
   * scheduling or visualization; callers explicitly pay for materialization. */
  /**
   * The marker range, as `[x, y, z, live]` per marker in fine-lattice units.
   *
   * A QA receipt, never a frame input: the overlay reads the same range on the
   * device without a copy, and a readback here would be a stall if the draw
   * needed one.
   */
  async readTracers(): Promise<Float32Array> {
    this.assertLive();
    const count = this.tracerLattice.count;
    if (count === 0) return new Float32Array(0);
    const bytes = 16 * count;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 marker readback",
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 marker copy",
      });
      encoder.copyBufferToBuffer(this.state, 4 * this.layout.tracers, readback, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return new Float32Array(readback.getMappedRange()).slice();
    } finally {
      readback.destroy();
    }
  }

  async readDiagnosticFields(): Promise<SparseCM12DiagnosticFields> {
    this.assertLive();
    const activitySnapshot = await this.readActivitySnapshot();
    const pressureCellBitsBytes = 4 * this.canonicalMembershipLayout.cell.activeBitWordCount;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 QA field readback",
      size: this.state.size + pressureCellBitsBytes + 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 QA field copy",
      });
      encoder.copyBufferToBuffer(this.state, 0, readback, 0, this.state.size);
      encoder.copyBufferToBuffer(this.activity,
        4 * this.canonicalMembershipLayout.cell.activeBitsBaseWords,
        readback, this.state.size, pressureCellBitsBytes);
      encoder.copyBufferToBuffer(this.topologyArena,
        4 * (this.frameControlLayout.baseWords
          + SPARSE_CM12_FRAME_CONTROL_HEADER.scalarParity),
        readback, this.state.size + pressureCellBitsBytes, 4);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const state = new Float32Array(readback.getMappedRange());
      const pressureCellBits = new Uint32Array(state.buffer,
        state.byteOffset + this.state.size,
        this.canonicalMembershipLayout.cell.activeBitWordCount);
      const acceptedParity = this.legacyHostAuthorityOracleForQA
        ? this.legacyHostAuthorityParityForQA
        : new Uint32Array(state.buffer,
          state.byteOffset + this.state.size + pressureCellBitsBytes, 1)[0]! & 1;
      const [nx, ny, nz] = this.dimensions;
      const count = nx * ny * nz;
      const density = new Float32Array(count);
      const gamma = new Float32Array(count);
      const velocity = new Float32Array(4 * count);
      const pressure = new Float32Array(count);
      const divergence = new Float32Array(count);
      const densityOffset = acceptedParity !== 0 ? this.layout.densityB : this.layout.densityA;
      const gammaOffset = acceptedParity !== 0 ? this.layout.gammaB : this.layout.gammaA;
      const velocityOffset = acceptedParity !== 0
        ? this.layout.cellVelocityB : this.layout.cellVelocityA;
      const cellWidth_m = this.parameterF32[41]!;
      const pressureScale = this.parameterF32[42]!;
      const topologyFloats = new Float32Array(this.templateWords.buffer,
        this.templateWords.byteOffset, this.templateWords.length);
      const cellOffset = this.templateWords[6]!, rangeOffset = this.templateWords[11]!;
      for (let brick = 0; brick < activitySnapshot.records.length; brick += 1) {
        const record = activitySnapshot.records[brick]!;
        if (!record.active || (record.reasons & 64) === 0) continue;
        const level = Math.log2(record.acceptedResolution);
        const templateLevelCount = Math.log2(this.brickFineResolution) + 1;
        const first = this.templateWords[
          rangeOffset + 2 * (templateLevelCount * brick + level)]!;
        const cellCount = this.templateWords[
          rangeOffset + 2 * (templateLevelCount * brick + level) + 1]!;
        const brickRecord = this.lastPacked!.brickOffset + 2 * brick;
        const key = this.lastPacked!.words[brickRecord + 1]!;
        const spanBricks = 1 << (this.lastPacked!.words[brickRecord]! & 31);
        const brickDimensions = this.dimensions.map((size) =>
          Math.ceil(size / this.brickFineResolution));
        const brickZ = Math.floor(key / (brickDimensions[0]! * brickDimensions[1]!));
        const keyXY = key - brickZ * brickDimensions[0]! * brickDimensions[1]!;
        const brickY = Math.floor(keyXY / brickDimensions[0]!);
        const brickX = keyXY - brickY * brickDimensions[0]!;
        for (let cell = first; cell < first + cellCount; cell += 1) {
        const base = cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell;
        const lower = [0, 1, 2].map((axis) => Math.round(
          topologyFloats[base + axis]! - 0.5 * topologyFloats[base + 4 + axis]!,
        ));
        if (lower[0] < this.brickFineResolution * brickX
          || lower[0] >= this.brickFineResolution * (brickX + spanBricks)
          || lower[1] < this.brickFineResolution * brickY
          || lower[1] >= this.brickFineResolution * (brickY + spanBricks)
          || lower[2] < this.brickFineResolution * brickZ
          || lower[2] >= this.brickFineResolution * (brickZ + spanBricks)) {
          throw new Error(`Sparse CM12 active brick ${brick} at ${brickX},${brickY},${brickZ}`
            + ` aliases cell ${cell} at ${lower.join(",")}`);
        }
        const span = [Math.round(topologyFloats[base + 4]!),
          Math.round(topologyFloats[base + 5]!),
          Math.round(topologyFloats[base + 6]!)] as const;
        const rho = state[densityOffset + cell]!;
        const gammaValue = state[gammaOffset + cell]!;
        const vx = state[velocityOffset + 4 * cell]! * cellWidth_m;
        const vy = state[velocityOffset + 4 * cell + 1]! * cellWidth_m;
        const vz = state[velocityOffset + 4 * cell + 2]! * cellWidth_m;
        const mappedPressure = state[this.layout.pressure + cell]! * pressureScale;
        const div = state[this.layout.divergence + cell]!;
        for (let dz = 0; dz < span[2] && lower[2] + dz < nz; dz += 1)
          for (let dy = 0; dy < span[1] && lower[1] + dy < ny; dy += 1)
            for (let dx = 0; dx < span[0] && lower[0] + dx < nx; dx += 1) {
              const at = lower[0] + dx + nx * (lower[1] + dy + ny * (lower[2] + dz));
              density[at] = rho;
              gamma[at] = gammaValue;
              velocity[4 * at] = vx; velocity[4 * at + 1] = vy;
              velocity[4 * at + 2] = vz;
              // Pressure is physical only on the current PCM domain. Inactive
              // stable slots intentionally retain solver scratch, so dense QA
              // publication canonicalizes them to zero without changing the
              // resident numerical state or frame scheduling.
              pressure[at] = (pressureCellBits[cell >>> 5]! & (1 << (cell & 31))) !== 0
                ? mappedPressure : 0;
              divergence[at] = div;
            }
        }
      }
      return { density, gamma, velocity, pressure, divergence };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Explicit QA receipt only; no production frame decision reads this back. */
  async readFrameControlQA(): Promise<SparseCM12FrameControlQA> {
    this.assertLive();
    const bytes = 4 * SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 FCA1 QA readback",
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({ label: "Sparse CM12 FCA1 QA copy" });
      encoder.copyBufferToBuffer(this.topologyArena, 4 * this.frameControlLayout.baseWords,
        readback, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      if (words[SPARSE_CM12_FRAME_CONTROL_HEADER.magic] !== SPARSE_CM12_FRAME_CONTROL_MAGIC) {
        throw new Error("Sparse CM12 FCA1 QA header is unavailable or incompatible");
      }
      const at = (name: keyof typeof SPARSE_CM12_FRAME_CONTROL_HEADER) =>
        words[SPARSE_CM12_FRAME_CONTROL_HEADER[name]]!;
      return {
        phase: at("phase"), fault: at("fault"), firstFaultOwner: at("firstFaultOwner"),
        acceptedGeneration: at("acceptedGeneration"),
        candidateGeneration: at("candidateGeneration"),
        sealedGeneration: at("sealedGeneration"), scalarParity: at("scalarParity"),
        faceParity: at("faceParity"), coverage: at("coverage"),
        committedFrames: at("committedFrames"),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Explicit SAW1 QA materialization; never consulted by frame encoding. */
  readScalarAuthorityQA() { return this.scalarResultAuthority.readQA(); }
  readScalarAuthorityIndirectQA() { return this.scalarResultAuthority.readIndirectQA(); }
  /** Header-only SRR1/SAW1 receipt for per-advance generation diagnostics. */
  readScalarAuthorityHeaderQA() { return this.scalarResultAuthority.readHeaderQA(); }
  /** Resident SIR1 ingress header; diagnostics only, never a scheduling input. */
  async readScalarIngressHeaderQA() {
    this.assertLive();
    const wordCount = SPARSE_CM12_SRR1_INGRESS_HEADER_WORDS;
    const readback = this.device.createBuffer({ label: "Sparse CM12 SIR1 header QA readback",
      size: 4 * wordCount, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder({ label: "Sparse CM12 SIR1 header QA copy" });
      encoder.copyBufferToBuffer(this.activity,
        4 * this.scalarResultIngressLayout.headerBaseWords,
        readback, 0, 4 * wordCount);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const h = SPARSE_CM12_SRR1_INGRESS_HEADER;
      return Object.freeze({
        magic: words[h.magic]!, version: words[h.version]!,
        tileCapacity: words[h.tileCapacity]!, eventCapacity: words[h.eventCapacity]!,
        candidateGeneration: words[h.candidateGeneration]!,
        topologyGeneration: words[h.topologyGeneration]!, sourceParity: words[h.sourceParity]!,
        eventCount: words[h.eventCount]!, receiptCount: words[h.receiptCount]!,
        fault: words[h.fault]!, firstFaultTile: words[h.firstFaultTile]!,
        committedGeneration: words[h.committedGeneration]!,
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Construction-only temporal seed census; never consulted by encoding. */
  async readTemporalSeedQA(): Promise<SparseCM12TemporalSeedQA> {
    this.assertLive();
    if (this.temporalSeedModeForQA === undefined) {
      throw new Error("temporal seed QA receipt requested from an ordinary resident");
    }
    const wordCount = 10;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 temporal seed QA readback",
      size: 4 * wordCount,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 temporal seed QA copy",
      });
      encoder.copyBufferToBuffer(this.activity,
        4 * this.temporalWorklistLayout.headerBaseWords,
        readback, 0, 4 * wordCount);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return {
        mode: this.temporalSeedModeForQA,
        seedCells: words[8]!,
        firstDilationCells: words[9]!,
        finalCells: words[0]!,
        finalRows: words[4]!,
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Construction-only FPA tile shadow census; never consulted by encoding. */
  async readFacePreparationTileCensusQA():
    Promise<SparseCM12FacePreparationTileCensusQA> {
    this.assertLive();
    const layout = this.facePreparationTileCensusLayout;
    if (!layout) {
      throw new Error("FPA tile census requested from an ordinary resident");
    }
    const readback = this.device.createBuffer({
      label: "Sparse CM12 FPA tile census QA readback",
      size: 4 * SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 FPA tile census QA copy",
      });
      encoder.copyBufferToBuffer(this.topologyArena, 4 * layout.headerBaseWords,
        readback, 0, 4 * SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return inspectSparseCM12FacePreparationTileCensusQA(
        new Uint32Array(readback.getMappedRange()));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Construction-only accepted FPA/VEX donor graph and C⊆S receipt. */
  async readFpaVexReadCensusQA(): Promise<SparseCM12FpaVexReadCensusSummaryQA> {
    this.assertLive();
    const layout = this.fpaVexReadCensusLayout;
    if (!layout) throw new Error("FVR1 census requested from an ordinary resident");
    const bytes = layout.compactSummaryBytes;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 FVR1 bounded QA readback", size: Math.max(4, bytes),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({ label: "Sparse CM12 FVR1 QA copy" });
      encoder.copyBufferToBuffer(this.topologyArena, 4 * layout.baseWords,
        readback, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return inspectSparseCM12FpaVexReadCensusSummaryQA(
        new Uint32Array(readback.getMappedRange()), layout);
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Construction-only PAB1 receipt and isolated timestamp intervals. */
  async readPressureAddressingABQA(): Promise<{
    readonly mode: SparseCM12PressureAddressingABModeName;
    readonly receipt?: SparseCM12PressureAddressingABReceipt;
    readonly materialization_ms: number;
    readonly verification_ms: number;
  }> {
    this.assertLive();
    const qa = this.pressureAddressingABQA;
    if (qa.mode === "canonicalRankSelect") {
      return { mode: qa.mode, materialization_ms: 0, verification_ms: 0 };
    }
    if (!qa.receiptReadback || !qa.timestampReadback) {
      throw new Error("PAB1 QA receipt requested from the production pressure-address authority");
    }
    try {
      await Promise.all([
        qa.receiptReadback.mapAsync(GPUMapMode.READ),
        qa.timestampReadback.mapAsync(GPUMapMode.READ),
      ]);
      const receipt = inspectSparseCM12PressureAddressingABReceipt(
        new Uint32Array(qa.receiptReadback.getMappedRange()), qa.layout,
        qa.layout.baseWords);
      const timestamps = new BigUint64Array(qa.timestampReadback.getMappedRange());
      return { mode: qa.mode, receipt,
        materialization_ms: Number(timestamps[1]! - timestamps[0]!) / 1e6,
        verification_ms: Number(timestamps[3]! - timestamps[2]!) / 1e6 };
    } finally {
      if (qa.receiptReadback.mapState === "mapped") qa.receiptReadback.unmap();
      if (qa.timestampReadback.mapState === "mapped") qa.timestampReadback.unmap();
    }
  }

  /** Header-only pressure-address receipt. This is diagnostic-only and works
   * for both the immutable production list and the two construction arms. */
  async readPressureAddressingHeaderQA(): Promise<SparseCM12PressureAddressingABReceipt> {
    this.assertLive();
    const qa = this.pressureAddressingABQA;
    const bytes = 4 * SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 PAB1 header QA readback", size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 PAB1 header QA copy",
      });
      encoder.copyBufferToBuffer(this.activity, 4 * qa.layout.baseWords,
        readback, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return inspectSparseCM12PressureAddressingABReceipt(
        new Uint32Array(readback.getMappedRange()), qa.layout, qa.layout.baseWords);
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** The two accepted cell/row dispatch triplets consumed by full-domain
   * kernels. Kept compact so a runaway packet can be diagnosed directly. */
  async readAcceptedIndirectQA(): Promise<readonly number[]> {
    this.assertLive();
    const bytes = 6 * 4;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 accepted-indirect QA readback", size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 accepted-indirect QA copy",
      });
      encoder.copyBufferToBuffer(this.acceptedIndirectArguments, 0, readback, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return Object.freeze(Array.from(new Uint32Array(readback.getMappedRange())));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  async readFrameControlIndirectQA(): Promise<readonly number[]> {
    this.assertLive();
    const bytes = 12 * SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 FCA1 indirect QA readback", size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 FCA1 indirect QA copy",
      });
      encoder.copyBufferToBuffer(this.frameControlIndirectArguments, 0,
        readback, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return Object.freeze(Array.from(new Uint32Array(readback.getMappedRange())));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Header-only VEX1 receipt; suitable for bounded per-frame fault diagnosis. */
  async readVelocityExtensionHeaderQA(): Promise<SparseCM12VelocityExtensionHeaderQA> {
    this.assertLive();
    const layout = this.vexActivityBatchLayout.velocityExtension;
    const bytes = 4 * SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 VEX1 header QA readback", size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 VEX1 header QA copy",
      });
      encoder.copyBufferToBuffer(this.activity, 4 * layout.headerBaseWords,
        readback, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const h = SPARSE_CM12_VELOCITY_EXTENSION_HEADER;
      if (words[h.magic] !== SPARSE_CM12_VELOCITY_EXTENSION_MAGIC
        || words[h.version] !== SPARSE_CM12_VELOCITY_EXTENSION_VERSION
        || words[h.headerWords] !== SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS
        || words[h.capacity] !== layout.cellCapacity) {
        throw new Error("Sparse CM12 VEX1 header QA receipt is unavailable or incompatible");
      }
      const firstFaultCell = words[h.firstFaultCell]!;
      const hasFirstFault = words[h.faultCount]! > 0
        && firstFaultCell !== SPARSE_CM12_FRAME_CONTROL_INVALID
        && firstFaultCell < layout.cellCapacity;
      const hasFramePlanProvenance = hasFirstFault && words[h.reserved] === 3;
      return {
        flags: words[h.flags]!, phase: words[h.reserved]!,
        acceptedGeneration: words[h.acceptedGeneration]!,
        candidateGeneration: words[h.candidateGeneration]!,
        topologyGeneration: words[h.topologyGeneration]!,
        capacity: words[h.capacity]!, rootCount: words[h.rootCount]!,
        blastCount: words[h.blastCount]!, maximumDepth: words[h.maximumDepth]!,
        executedCellCount: words[h.executedCellCount]!,
        reusedCellCount: words[h.reusedCellCount]!, faultCount: words[h.faultCount]!,
        uncoveredWriteCount: words[h.uncoveredWriteCount]!,
        ...(hasFirstFault ? {
          firstFault: { cell: firstFaultCell, depth: words[h.firstFaultDepth]! },
        } : {}),
        ...(hasFramePlanProvenance ? {
          framePlanProvenance: {
            ownerBrick: words[h.rootDispatchY]!, ownerTile: words[h.rootDispatchZ]!,
            slot: words[h.frontierADispatchY]!,
            tileGeneration: words[h.frontierADispatchZ]!,
            packedStageMasks: words[h.frontierBDispatchY]!,
            stage0MaskLow: words[h.frontierBDispatchZ]!,
          },
        } : {}),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Construction-only VEX comparison receipt. No frame decision consumes it. */
  async readVelocityExtensionQA() {
    this.assertLive();
    const layout = this.vexActivityBatchLayout.velocityExtension;
    const capacity = layout.cellCapacity;
    const headerWords = SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS;
    const blastDepthAt = headerWords;
    const candidateDepthAt = blastDepthAt + capacity;
    const rootCauseAt = candidateDepthAt + capacity;
    const acceptedDepthAt = rootCauseAt + capacity;
    const acceptedOwnerAt = acceptedDepthAt + capacity;
    const velocityAt = acceptedOwnerAt + capacity;
    const totalWords = velocityAt + 4 * capacity;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 VEX1 QA readback",
      size: 4 * totalWords,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({ label: "Sparse CM12 VEX1 QA copy" });
      encoder.copyBufferToBuffer(this.activity, 4 * layout.headerBaseWords,
        readback, 0, 4 * headerWords);
      for (const [source, destination] of [
        [layout.blastDepthBaseWords, blastDepthAt],
        [layout.candidateDepthBaseWords, candidateDepthAt],
        [layout.rootCauseBaseWords, rootCauseAt],
        [layout.acceptedDepthBaseWords, acceptedDepthAt],
        [layout.acceptedOwnerBaseWords, acceptedOwnerAt],
      ] as const) {
        encoder.copyBufferToBuffer(this.activity, 4 * source,
          readback, 4 * destination, 4 * capacity);
      }
      encoder.copyBufferToBuffer(this.state,
        4 * this.vexActivityBatchLayout.velocityState.acceptedVelocityFloatBase,
        readback, 4 * velocityAt, 16 * capacity);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const header = words.slice(0, headerWords);
      if (header[SPARSE_CM12_VELOCITY_EXTENSION_HEADER.magic]
          !== SPARSE_CM12_VELOCITY_EXTENSION_MAGIC) {
        throw new Error("Sparse CM12 VEX1 QA header is unavailable or incompatible");
      }
      return {
        header,
        blastDepth: words.slice(blastDepthAt, blastDepthAt + capacity),
        candidateDepth: words.slice(candidateDepthAt, candidateDepthAt + capacity),
        rootCause: words.slice(rootCauseAt, rootCauseAt + capacity),
        acceptedDepth: words.slice(acceptedDepthAt, acceptedDepthAt + capacity),
        acceptedOwner: words.slice(acceptedOwnerAt, acceptedOwnerAt + capacity),
        velocityBits: words.slice(velocityAt, velocityAt + 4 * capacity),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** QA-only policy readback. It is never called by frame scheduling. */
  async readActivitySnapshot(): Promise<SparseCM12GPUActivitySnapshot> {
    this.assertLive();
    const readback = this.device.createBuffer({
      label: "Sparse CM12 activity QA readback",
      size: this.activity.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 activity QA copy",
      });
      encoder.copyBufferToBuffer(this.activity, 0, readback, 0, this.activity.size);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const records = Array.from({ length: this.lastPacked!.brickCount }, (_, brick) => {
        const at = ACTIVITY_HEADER_WORDS + ACTIVITY_RECORD_WORDS * brick;
        return {
          scoreByte: words[at]!,
          reasons: words[at + 1]!,
          thinFluid: (words[at + 1]! & 256) !== 0,
          hotEpochs: words[at + 2]! & 0xff,
          quietEpochs: (words[at + 2]! >>> 8) & 0xff,
          meanDensity: new DataView(words.buffer).getFloat32(4 * (at + 4), true),
          densityMoments: [5, 6, 7].map((offset) =>
            new DataView(words.buffer).getFloat32(4 * (at + offset), true)) as
              [number, number, number],
          plannedResolution: words[at + 8] as SparseBrickResolution,
          planReasons: words[at + 9]!,
          active: words[at + 10] !== 0,
          activatedStep: words[at + 11]!,
          acceptedResolution: words[at + 12] as SparseBrickResolution,
          candidateResolution: words[at + 13] as SparseBrickResolution,
          candidateStatus: (words[at + 14] === 1 ? 1 : words[at + 14] === 2 ? 2 : 0) as
            SparseCM12GPUActivityRecord["candidateStatus"],
          candidateEpoch: words[at + 15]!,
          transferMassBeforeFineCells: new DataView(words.buffer).getFloat32(
            4 * (at + 16), true,
          ),
          transferMassAfterFineCells: new DataView(words.buffer).getFloat32(
            4 * (at + 17), true,
          ),
          transferMassErrorFineCells: new DataView(words.buffer).getFloat32(
            4 * (at + 18), true,
          ),
          transferGammaErrorFineCells: new DataView(words.buffer).getFloat32(
            4 * (at + 19), true,
          ),
          transferMomentumErrorFineCells: [20, 21, 22].map((offset) =>
            new DataView(words.buffer).getFloat32(4 * (at + offset), true)) as
              [number, number, number],
          transferStatus: (words[at + 23] === 1 ? 1 : words[at + 23] === 2 ? 2 : 0) as
            SparseCM12GPUActivityRecord["transferStatus"],
          transferExteriorFluxErrorFineAreas: [24, 25, 26, 27, 28, 29].map((offset) =>
            new DataView(words.buffer).getFloat32(4 * (at + offset), true)) as
              [number, number, number, number, number, number],
          maximumAbsoluteTransferFluxErrorFineAreas: new DataView(words.buffer).getFloat32(
            4 * (at + 30), true,
          ),
          faceTransferStatus: (words[at + 31] === 1 ? 1 : words[at + 31] === 2 ? 2 : 0) as
            SparseCM12GPUActivityRecord["faceTransferStatus"],
          supportMask: words[at + 32]!,
          sweptSupportMask: words[at + 3]!,
          maximumVelocityTravelFineCells: new DataView(words.buffer).getFloat32(
            4 * (at + 33), true,
          ),
          retiredResidueMassFineCells: new DataView(words.buffer).getFloat32(
            4 * (at + 34), true,
          ),
          topologyPreparationScheduled: words[at + 35] !== 0,
          topologyPreparationEpoch: words[at + 36]!,
          topologyPage: words[at + 37] === INVALID ? undefined : words[at + 37],
        };
      });
      return { acceptedSteps: words[0]!, records };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.rigidCoupling?.destroy();
    this.scalarResultAuthority.destroy();
    this.pressureAddressingABQA.indirectArguments.destroy();
    this.pressureAddressingABQA.receiptReadback?.destroy();
    this.pressureAddressingABQA.queryResolve?.destroy();
    this.pressureAddressingABQA.timestampReadback?.destroy();
    this.pressureAddressingABQA.querySet?.destroy();
    for (const buffer of [this.parameters, this.topology, this.state, this.partials,
      this.scalars, this.conditioning, this.activity, this.candidateState,
      this.topologyArena, this.acceptedIndirectArguments,
      this.pressureCellIndirectArguments, this.pressureRowIndirectArguments,
      this.pressureMembershipIndirectArguments,
      this.frameControlIndirectArguments,
      this.pressureTopologyRepairIndirectArguments,
      this.persistentPressureCacheIndirectArguments,
      this.faceProjectionAuthorityIndirectArguments,
      this.temporalCellIndirectArguments, this.temporalRowIndirectArguments,
      this.activityIndirectArguments, this.vexActivityIndirectArguments,
      this.framePlanIndirectArguments,
      this.presentationIndirectArguments,
      this.pressureTemplates, this.pressureWorklists,
      this.fineParams, this.fineMetadata, this.fineWorklist, this.fineSamples,
      this.fineWorkA, this.fineWorkB, this.fineRollback]) {
      buffer.destroy();
    }
    this.diagnosticsReadback.destroy();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Sparse CM12 resident pipeline is destroyed");
  }
}

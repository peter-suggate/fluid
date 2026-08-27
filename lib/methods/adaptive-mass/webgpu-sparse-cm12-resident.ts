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
import type { SolidWorld } from "../../core/solid-world";
import {
  createSparseCM12SolidOccupancyLayout,
  SPARSE_CM12_SOLID_OCCUPANCY_ENTRY_WORDS,
  SPARSE_CM12_SOLID_FRACTION_PAGE_WORDS,
  SPARSE_CM12_SOLID_OCCUPANCY_PAGE_WORDS,
  SPARSE_CM12_SOLID_REGION_WORDS,
  writeSparseCM12SolidOccupancy,
  type SparseCM12SolidOccupancyLayout,
} from "./sparse-cm12-solid-occupancy";
import type { GPUFluidFaceVelocitySource } from "../../core/webgpu-face-velocity-overlay";
import type { GPUFluidTracerSource } from "../../core/webgpu-tracer-overlay";
import type { GPUPressureJournalSource } from "../../core/webgpu-pressure-journal-overlay";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import type { GPURigidSolidWorldCollisionSource } from
  "../../core/webgpu-rigid-body";
import {
  SPARSE_CM12_REFINEMENT_REGION_BYTES,
  SPARSE_CM12_REFINEMENT_REGION_PARAMETER_OFFSET,
} from "./sparse-cm12-refinement-regions";
import {
  FINE_LEVELSET_COMPACT_LOOKUP_FLAG,
  FINE_LEVELSET_SIGNED_SPARSE_ADDRESS_FLAG,
  FINE_LEVELSET_METADATA_WORDS,
  FINE_LEVELSET_WORKSET_HEADER_WORDS,
  type FineLevelSetBrickPlan,
} from "../../core/fine-levelset-brick-abi";
import type {
  SparseAdaptiveGridConsumerSource,
  WebGPUFineLevelSetBrickSource,
} from "../../core/levelset-consumer-abi";
import {
  gpuCompilationManagerFor,
  type GPUCompilationSnapshot,
} from "../../core/gpu-compilation-manager";
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
} from "./webgpu-sparse-cm12-resident.wgsl";
import {
  createSparseCM12LogicalOwnerDirectory,
} from "./sparse-cm12-logical-owner-directory";
import { createSparseCM12TransportExecutionImage,
  createSparseCM12TransportExecutionImageLayout,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS,
  type SparseCM12TransportExecutionImageLayout } from
  "./sparse-cm12-transport-execution-image";
import { createSparseCM12TransportPacketAuthorityLayout,
  type SparseCM12TransportPacketAuthorityLayout } from
  "./sparse-cm12-transport-packet-authority";
import { createSparseCM12EffectiveTransportVelocityLayout } from
  "./sparse-cm12-effective-transport-velocity";
import { createSparseCM12FinalScalarPacketMaskInitialWords,
  createSparseCM12FinalScalarPacketMaskLayout,
  SPARSE_CM12_FINAL_SCALAR_MASK_HEADER,
  SPARSE_CM12_FINAL_SCALAR_MASK_HEADER_WORDS,
  type SparseCM12FinalScalarPacketMaskLayout } from
  "./sparse-cm12-final-scalar-packet-masks";
import { compileSparseCM12FactoredAEIPackedTemplateCatalog } from
  "./sparse-cm12-factored-aei-packed-template";
import { compileSparseCM12StableLeafFaceNeighbors } from
  "./sparse-cm12-factored-aei-topology";
import { compileSparseCM12InternedBoundaryOperators } from
  "./sparse-cm12-interned-boundary-operators";
import { createSparseCM12InternedBoundaryImage } from
  "./sparse-cm12-interned-boundary-image";
import { createSparseCM12InternedRefLookup } from
  "./sparse-cm12-interned-ref-lookup";
import { createSparseCM12IboTRASupplement } from
  "./sparse-cm12-ibo-tra-supplement";
import { compareSparseCM12IBOSemanticAuthority,
  compileSparseCM12GeometryFaceNeighbors,
} from
  "./sparse-cm12-ibo-semantic-authority";
import {
  SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER,
  SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS,
  createSparseCM12TopologyEffectsAuthorityInitialWords,
  createSparseCM12TopologyEffectsAuthorityLayout,
  sparseCM12TopologyEffectsIndirectByteOffset,
  type SparseCM12TopologyEffectsAuthorityLayout,
} from "./sparse-cm12-topology-effects-authority";
import {
  SPARSE_CM12_WORLD_DIRECTORY_HEADER,
  SPARSE_CM12_WORLD_DIRECTORY_HEADER_WORDS,
  SPARSE_CM12_WORLD_DIRECTORY_INVALID,
  SPARSE_CM12_WORLD_DIRECTORY_LEAF,
  SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS,
  createSparseCM12WorldDirectoryInitialWords,
  createSparseCM12WorldDirectoryLayout,
  createSparseCM12WorldDirectoryWGSL,
  type SparseCM12WorldDirectoryLayout,
} from "./sparse-cm12-world-directory";
import {
  SPARSE_CM12_PHASE1_TRANSPORT_QA_HEADER,
  SPARSE_CM12_PHASE1_TRANSPORT_QA_MAGIC,
  SPARSE_CM12_PHASE1_TRANSPORT_QA_VERSION,
  createSparseCM12Phase1TransportQALayout,
  sparseCM12Phase1Sha256,
  type SparseCM12Phase1TransportQALayout,
  type SparseCM12Phase1TransportReceipt,
} from "./sparse-cm12-phase1-transport-receipt";
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
  SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER_WORDS,
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
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS,
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
  SPARSE_CM12_VELOCITY_EXTENSION_DEPTH,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS,
  SPARSE_CM12_VELOCITY_EXTENSION_MAGIC,
  SPARSE_CM12_VELOCITY_EXTENSION_VERSION,
  createSparseCM12VelocityExtensionInitialWords,
  createSparseCM12VelocityExtensionResidentLayouts,
  sparseCM12VelocityExtensionMaskDensity,
  type SparseCM12VelocityExtensionLayout,
} from "./sparse-cm12-velocity-extension";
import {
  compileSparseCM12BrickTileFaceAddressProgram,
  type SparseCM12BrickTileFaceAddressLayout,
} from "./sparse-cm12-brick-tile-face-address-program";
import {
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS,
  createSparseCM12PressureTopologyRepairInitialWords,
  createSparseCM12PressureTopologyRepairLayout,
  sparseCM12PressureTopologyRepairEntryPoints,
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
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_ENTRY_POINTS,
  createSparseCM12PressureExecutionImageInitialWords,
  createSparseCM12PressureExecutionImageLayout,
  sparseCM12PressureExecutionImageCellIndirectByteOffset,
  sparseCM12PressureExecutionImageIndirectByteOffset,
  type SparseCM12PressureExecutionImageLayout,
} from "./sparse-cm12-pressure-execution-image";

/** CM12 Sec. 3.5 Algorithm 2's live trace bounds, in finest cells and substeps. */
export interface SharpeningTrace {
  readonly distanceCells?: number;
  readonly traceSteps?: number;
  /** Defaults on for direct diagnostic constructors and existing callers. */
  readonly gammaDiffusionEnabled?: boolean;
  /** Defaults on; the mandatory final-scalar publication is independent. */
  readonly surfaceSharpeningEnabled?: boolean;
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

export interface SparseCM12InternedBoundaryMemoryPlan {
  readonly immutableMaximumBytes: number;
  readonly slotMaximumBytes: number;
  readonly semanticAuthorityMaximumBytes: number;
}

/**
 * Construction budgets for the resident compiled-boundary image. B8/P8 is the
 * shipping lane and has 4.27x as many resident leaves as ocean B16/P16; using
 * the B16 census ceilings for it rejected a 2.1 MiB exact image before device
 * allocation. These remain hard representation gates, now sized for the lane
 * they govern rather than for one larger-brick census.
 */
export function sparseCM12InternedBoundaryMemoryPlan(
  brickFineResolution: SparseBrickResolution,
): SparseCM12InternedBoundaryMemoryPlan {
  return brickFineResolution === 16
    ? Object.freeze({
      immutableMaximumBytes: 512 * 1024,
      slotMaximumBytes: 256 * 1024,
      semanticAuthorityMaximumBytes: 16_864,
    })
    : Object.freeze({
      immutableMaximumBytes: 1024 * 1024,
      slotMaximumBytes: 640 * 1024,
      semanticAuthorityMaximumBytes: 32 * 1024,
    });
}

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
import type { StageLensSource, StageTapSink } from "../../core/stage-lens";
import {
  SPARSE_CM12_LENSES,
  SPARSE_CM12_STAGE_LENSES,
  sparseCM12StageTaps,
  type SparseCM12StageTapName,
} from "./sparse-cm12-stage-lenses";
import { SparseCM12StageLensSource } from "./sparse-cm12-stage-lens-source";
import type { SparseCM12AddressingSpec } from "./sparse-cm12-stage-contract";

/** Encode order. The seam chain closes these left to right, exactly once each. */
export const SPARSE_CM12_RESIDENT_STAGES = Object.freeze([
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
export type SparseCM12ResidentStageId = (typeof SPARSE_CM12_RESIDENT_STAGES)[number];

/**
 * The sub-seams each stage closes, in encode order.
 *
 * Exhaustive over the stage ids on purpose: a stage with no sub-seams says so
 * with an empty tuple. Each sub-seam owns one disjoint timestamp interval
 * inside its stage — the stage's own seam then times only what follows the
 * last of them — and a stage body can close exactly the sub-seams listed
 * under its own id: `encode`'s `stage()` helper hands the body a
 * `closeSubstage` typed to this row, so closing a neighbour's seam, or one
 * that has been renamed here, is a compile error rather than a mis-timed
 * interval. The method's stage registry (`sparse-cm12-stages.ts`) is typed
 * against this same table, so every sub-seam here must also have a phase
 * there, and no phase there may name a sub-seam that is not here.
 */
export const SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES = Object.freeze({
  "transport-velocity-extension": [
    "frame-control-authority",
    "velocity-extension-mask-initialization",
    "velocity-extension-sweeps",
    "transport-packet-authority",
  ],
  "face-preparation": [
    "face-support-publication",
    "dirty-face-row-preparation",
  ],
  "conservative-transport": [
    "transport-trace",
    "transport-scatter",
    "transport-gather",
  ],
  "tracer-advection": [],
  "gamma-diffusion": [],
  "surface-sharpening": [
    "sharpening-receipt-setup",
    "sharpening-transform",
    "sharpening-finalize",
    "density-capacity-repair",
    "final-scalar-mask-publication",
  ],
  "symmetry-authority": [],
  "body-forces": [],
  "pressure-topology": [
    "ptr-setup-brick-plan",
    "pcm-cell-publication",
    "pcm-row-publication",
    "pca-fine-publication",
    "pca-coarse-repair",
    "pca-hierarchy-and-freeze",
    "pei-publication",
    "ptr-commit-and-prepare-pressure",
  ],
  "pressure-rhs": [],
  "pressure-solve": [],
  "velocity-projection": [],
  "projection-diagnostics": [],
  "activity-measurement": [],
  "resolution-planning": [],
  "candidate-transfer": [
    "candidate-field-transfer",
    "candidate-face-reconstruction",
    "candidate-face-validation",
    "candidate-effects-preflight",
    "candidate-ibo-construction",
    "candidate-ibo-validation",
    "candidate-tei-compilation",
    "candidate-authorization",
    "candidate-ptr-publication",
    "candidate-effects-seal",
    "candidate-state-publication",
    "candidate-image-replay",
  ],
  "brick-retirement": [],
  "presentation-publication": [],
} as const satisfies Readonly<Record<SparseCM12ResidentStageId, readonly string[]>>);

/** The sub-seams one stage owns; `never` for a stage that closes none. */
export type SparseCM12ResidentSubstage<
  Stage extends SparseCM12ResidentStageId = SparseCM12ResidentStageId,
> = (typeof SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES)[Stage][number];

/** Where an observer is told about each stage of the advance. */
export interface SparseCM12ResidentStageSeams {
  /**
   * Close the named stage, immediately after its last dispatch. A stage that
   * encodes nothing this advance still reports: it closes on its successor's
   * boundary and costs exactly zero.
   */
  readonly close: (stage: SparseCM12ResidentStageId) => void;
  /**
   * Close one named sub-seam inside a stage. Unlike a diagram rollup, each
   * sub-seam owns one disjoint timestamp interval; the stage's own `close`
   * then covers only what followed its last sub-seam.
   */
  readonly closeSubstage?: <Stage extends SparseCM12ResidentStageId>(
    stage: Stage,
    substage: SparseCM12ResidentSubstage<Stage>,
  ) => void;
  /** Bind the final timestamp to a word published at the true command tail. */
  readonly anchorFinalBoundary?: (source: GPUBuffer, offset?: number) => void;
}

/**
 * What a stage body is handed by `encode`'s `stage()` helper: the closer for
 * its own sub-seams and the tap sink of its own lens, and nothing of any other
 * stage's. Both are typed to the stage id the body was opened under.
 */
export interface SparseCM12StageEncodeContext<Stage extends SparseCM12ResidentStageId> {
  readonly closeSubstage: (substage: SparseCM12ResidentSubstage<Stage>) => void;
  readonly lens: StageTapSink<SparseCM12StageTapName<Stage>>;
}

const WORKGROUP_SIZE = 64;
/** Small predefined call-graph families amortize shared helper compilation.
 * Chunks are submitted and released serially below. Four roots keep each Metal
 * compiler unit modest without returning to hundreds of one-use modules. */
const SPARSE_CM12_SIMULATION_SHADER_ENTRY_CHUNK_SIZE = 4;
const SPARSE_CM12_ISOLATED_SIMULATION_SHADER_ENTRIES = new Set([
  "traceGammaAndBeta", "scatterDensityDeficit", "gatherConservativeDensity",
  "measureBrickActivity",
]);

export function sparseCM12PresentationPageAllocatorWGSL(
  brickCount: number,
  initialBrickCount: number,
  brickRecordBaseWords: number,
  layout: SparseCM12FramePlanPresentationLayout,
  worldDirectoryLayout?: SparseCM12WorldDirectoryLayout,
  brickDimensions: readonly [number, number, number] = [1, 1, 1],
): string {
  // WDR1 is a signed-coordinate authority. Do not make its presentation-key
  // ABI a second caller-controlled switch: that allowed a resident SparseWorld
  // to be paired with the retired dense atlas key for authored leaves while
  // dynamic leaves used their WDR coordinates. The no-WDR form remains only
  // for the bounded static-atlas shader fixture.
  const signedSparseAddressing = worldDirectoryLayout !== undefined;
  return /* wgsl */ `
const INVALID:u32=0xffffffffu;
const BRICK_COUNT:u32=${brickCount}u;
const INITIAL_BRICK_COUNT:u32=${initialBrickCount}u;
const BRICK_RECORD_BASE:u32=${brickRecordBaseWords}u;
const ACTIVITY_HEADER:u32=${ACTIVITY_HEADER_WORDS}u;
const ACTIVITY_RECORD_WORDS:u32=${ACTIVITY_RECORD_WORDS}u;
const BRICK_PAGES:u32=${layout.brickPagesBaseWords}u;
const ALLOCATOR:u32=${layout.allocatorBaseWords}u;
const FREE_LIST:u32=${layout.allocatorFreeListBaseWords}u;
const PAGE_CAPACITY:u32=${layout.pageCapacity}u;
const WORKLIST_HEADER:u32=${FINE_LEVELSET_WORKSET_HEADER_WORDS}u;
@group(0)@binding(0)var<storage,read>topology:array<u32>;
@group(0)@binding(1)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(2)var<storage,read_write>metadata:array<atomic<u32>>;
@group(0)@binding(3)var<storage,read_write>worklist:array<atomic<u32>>;
@group(0)@binding(4)var<storage,read_write>topologyArena:array<atomic<u32>>;
${worldDirectoryLayout
    ? createSparseCM12WorldDirectoryWGSL(worldDirectoryLayout)
    : ""}

fn reserveSparseCM12PresentationPage()->u32{
  // Weak CAS is permitted to fail spuriously and has no forward-progress
  // guarantee. Dynamic expansion can make many leaves contend here at once,
  // so an unbounded retry is a GPU-watchdog loop. Fail closed and retry the
  // still-unmapped brick next frame instead.
  for(var attempt=0u;attempt<256u;attempt+=1u){
    let count=atomicLoad(&activity[ALLOCATOR+5u]);
    if(count==0u){return INVALID;}
    let reservation=atomicCompareExchangeWeak(&activity[ALLOCATOR+5u],count,count-1u);
    if(reservation.exchanged){
      let page=atomicLoad(&activity[FREE_LIST+count-1u]);
      return select(INVALID,page,page<PAGE_CAPACITY);
    }
  }
  return INVALID;
}

@compute @workgroup_size(64)
fn allocateSparseCM12PresentationPages(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=BRICK_COUNT){return;}
  let activityRecord=ACTIVITY_HEADER+ACTIVITY_RECORD_WORDS*brick;
  if(atomicLoad(&activity[activityRecord+10u])==0u
    ||atomicLoad(&activity[BRICK_PAGES+brick])!=INVALID){return;}
  var key=topology[BRICK_RECORD_BASE+2u*min(brick,INITIAL_BRICK_COUNT-1u)+1u];
  if(${worldDirectoryLayout ? "true" : "brick>=INITIAL_BRICK_COUNT"}){
    let leaf=${worldDirectoryLayout?.baseWords ?? 0}u
      +${worldDirectoryLayout?.leafBaseWords ?? 0}u+5u*brick;
    let coordinate=vec3i(bitcast<i32>(atomicLoad(&topologyArena[leaf])),
      bitcast<i32>(atomicLoad(&topologyArena[leaf+1u])),
      bitcast<i32>(atomicLoad(&topologyArena[leaf+2u])));
    ${signedSparseAddressing ? /* wgsl */ `if(coordinate.x< -1024||coordinate.x>1023
      ||coordinate.y< -512||coordinate.y>511||coordinate.z< -1024||coordinate.z>1022){
      atomicStore(&activity[ALLOCATOR+1u],3u);atomicOr(&activity[7],32u);return;
    }
    key=u32(coordinate.x+1024)|(u32(coordinate.y+512)<<11u)
      |(u32(coordinate.z+1024)<<21u);` : /* wgsl */ `if(any(coordinate<vec3i(0))
      ||any(coordinate>=vec3i(${brickDimensions[0]},${brickDimensions[1]},${brickDimensions[2]}))){
      atomicStore(&activity[ALLOCATOR+1u],3u);atomicOr(&activity[7],32u);return;
    }
    key=u32(coordinate.x)+${brickDimensions[0]}u
      *(u32(coordinate.y)+${brickDimensions[1]}u*u32(coordinate.z));`}
  }
  // Validate the signed address before claiming a physical page. Invalid
  // coordinates must not leak one allocator slot on every subsequent frame.
  let page=reserveSparseCM12PresentationPage();
  if(page==INVALID){
    atomicStore(&activity[ALLOCATOR+1u],1u);
    atomicOr(&activity[7],32u);return;
  }
  atomicStore(&metadata[4u*page],page);
  atomicStore(&metadata[4u*page+1u],key);
  atomicStore(&metadata[4u*page+2u],0u);
  atomicStore(&metadata[4u*page+3u],brick<<3u);
  atomicStore(&activity[BRICK_PAGES+brick],page);
  let work=atomicAdd(&worklist[1],1u);
  if(work>=PAGE_CAPACITY){
    atomicStore(&activity[ALLOCATOR+1u],2u);
    atomicOr(&activity[7],32u);return;
  }
  atomicStore(&worklist[WORKLIST_HEADER+work],page);
  atomicMax(&worklist[4],(work+64u)/64u);
  atomicAdd(&activity[ALLOCATOR],1u);
  atomicMax(&activity[ALLOCATOR+2u],page+1u);
  atomicAdd(&activity[ALLOCATOR+3u],1u);
}

fn sparseCM12PresentationPageLess(left:u32,right:u32)->bool{
  if(left==INVALID){return false;}
  if(right==INVALID){return true;}
  let leftKey=atomicLoad(&metadata[4u*left+1u]);
  let rightKey=atomicLoad(&metadata[4u*right+1u]);
  return leftKey<rightKey||(leftKey==rightKey&&left<right);
}

@compute @workgroup_size(1)
fn sortSparseCM12PresentationPageDirectory(){
  let count=min(atomicLoad(&worklist[1]),PAGE_CAPACITY);
  // The prefix was sorted at generation zero and allocations only append.
  // Insert just the newly allocated pages; cost is O(new pages × resident
  // pages), independent of empty logical-world capacity.
  let sorted=min(atomicLoad(&activity[ALLOCATOR+4u]),count);
  for(var index=sorted;index<count;index+=1u){
    let page=atomicLoad(&worklist[WORKLIST_HEADER+index]);var cursor=index;
    loop{
      if(cursor==0u){break;}
      let previous=atomicLoad(&worklist[WORKLIST_HEADER+cursor-1u]);
      if(sparseCM12PresentationPageLess(previous,page)){break;}
      atomicStore(&worklist[WORKLIST_HEADER+cursor],previous);cursor-=1u;
    }
    atomicStore(&worklist[WORKLIST_HEADER+cursor],page);
  }
  atomicStore(&activity[ALLOCATOR+4u],count);
}

// Presentation retirement is deliberately after FPP1 execution. The retiring
// generation first publishes its complete all-air page transaction; only then
// may the renderer directory stop naming the page and return its physical slot.
@compute @workgroup_size(64)
fn retireSparseCM12PresentationPages(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=BRICK_COUNT){return;}
  let activityRecord=ACTIVITY_HEADER+ACTIVITY_RECORD_WORDS*brick;
  // Candidate-active or scheduled inactive leaves are frontier reservations,
  // not garbage. Reclaim only a fully accepted, quiescent retirement.
  if(atomicLoad(&activity[activityRecord+10u])!=0u
    ||atomicLoad(&activity[activityRecord+35u])!=0u){return;}
  let page=atomicExchange(&activity[BRICK_PAGES+brick],INVALID);
  if(page!=INVALID){
    if(page>=PAGE_CAPACITY){
      atomicStore(&activity[ALLOCATOR+1u],4u);atomicOr(&activity[7],32u);return;
    }
    atomicStore(&metadata[4u*page],INVALID);
    atomicStore(&metadata[4u*page+1u],INVALID);
    atomicStore(&metadata[4u*page+2u],0u);
    atomicStore(&metadata[4u*page+3u],INVALID);
    let free=atomicAdd(&activity[ALLOCATOR+5u],1u);
    if(free<PAGE_CAPACITY){
      atomicStore(&activity[FREE_LIST+free],page);
      atomicSub(&activity[ALLOCATOR],1u);
    }else{
      atomicSub(&activity[ALLOCATOR+5u],1u);
      atomicStore(&activity[ALLOCATOR+1u],4u);atomicOr(&activity[7],32u);
    }
  }
  ${worldDirectoryLayout ? /* wgsl */ `if(brick>=CM12_WDR_INITIAL_LEAVES){
    let topologyPage=atomicLoad(&activity[activityRecord+37u]);
    if(topologyPage!=INVALID&&cm12WorldReleaseLeaf(brick)){
    let topologyBase=atomicLoad(&topologyArena[14u]);
    let descriptor=topologyBase+atomicLoad(&topologyArena[topologyBase+30u])
      +topologyPage*atomicLoad(&topologyArena[topologyBase+31u]);
    atomicStore(&topologyArena[descriptor],INVALID);
    atomicStore(&topologyArena[descriptor+1u],0u);
    atomicStore(&topologyArena[descriptor+2u],0u);
    atomicStore(&topologyArena[descriptor+3u],0u);
    atomicStore(&activity[activityRecord+37u],INVALID);
    }
  }` : ""}
}

fn sparseCM12PresentationRecomputeWorldBounds(){
  ${worldDirectoryLayout ? "cm12WorldRecomputeBounds();" : ""}
}

@compute @workgroup_size(1)
fn compactSparseCM12PresentationPageDirectory(){
  let count=min(atomicLoad(&worklist[1]),PAGE_CAPACITY);var retained=0u;
  sparseCM12PresentationRecomputeWorldBounds();
  for(var index=0u;index<count;index+=1u){
    let page=atomicLoad(&worklist[WORKLIST_HEADER+index]);
    if(page<PAGE_CAPACITY&&atomicLoad(&metadata[4u*page])==page){
      atomicStore(&worklist[WORKLIST_HEADER+retained],page);retained+=1u;
    }
  }
  for(var index=retained;index<count;index+=1u){
    atomicStore(&worklist[WORKLIST_HEADER+index],INVALID);
  }
  atomicStore(&worklist[1],retained);
  atomicStore(&worklist[4],(retained+63u)/64u);
  // Compaction preserves key order, so next frame's insertion sort can begin
  // at the retained prefix and touch only newly allocated pages.
  atomicStore(&activity[ALLOCATOR+4u],retained);
}
`;
}

interface SparseCM12DeviceCompilationCache {
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly pipelineLayout: GPUPipelineLayout;
  readonly shaderModules: Map<string, Promise<GPUShaderModule>>;
  readonly presentationPipelines: Map<string,
    Promise<Readonly<Record<string, GPUComputePipeline>>>>;
  readonly simulationPipelines: Map<string,
    Promise<Readonly<Record<string, GPUComputePipeline>>>>;
}

const sparseCM12CompilationCacheByDevice =
  new WeakMap<GPUDevice, SparseCM12DeviceCompilationCache>();

/** Immutable CM12 binding ABI and compiled programs belong to the device, not a scene. */
function sparseCM12DeviceCompilationCache(device: GPUDevice):
SparseCM12DeviceCompilationCache {
  const existing = sparseCM12CompilationCacheByDevice.get(device);
  if (existing) return existing;
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Sparse CM12 resident layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } },
      ...[2, 3, 4].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" as const } })),
      { binding: 5, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", minBindingSize: 4 } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const cache: SparseCM12DeviceCompilationCache = {
    bindGroupLayout,
    pipelineLayout: device.createPipelineLayout({
      label: "Sparse CM12 resident pipeline layout",
      bindGroupLayouts: [bindGroupLayout],
    }),
    shaderModules: new Map(),
    presentationPipelines: new Map(),
    simulationPipelines: new Map(),
  };
  sparseCM12CompilationCacheByDevice.set(device, cache);
  return cache;
}

/**
 * Retain the WGSL declarations and only the function call graph needed by a
 * small entry-point family. Metal otherwise compiles every function in the
 * monolithic CM12 module even when a pipeline names one presentation kernel.
 */
export function sparseCM12WGSLForEntryPoints(source: string, roots: readonly string[]): string {
  type FunctionSpan = { name: string; start: number; end: number; body: string };
  type GlobalSpan = { name: string; start: number; end: number; text: string };
  // Mask comments while preserving offsets/newlines. Generated WGSL is often
  // deliberately compact (several declarations per line), so line anchoring
  // misses real globals while an unmasked regex mistakes prose for syntax.
  const syntaxCharacters = source.split("");
  for (let index = 0; index < syntaxCharacters.length;) {
    if (source[index] === "/" && source[index + 1] === "/") {
      while (index < syntaxCharacters.length && source[index] !== "\n") {
        syntaxCharacters[index++] = " ";
      }
    } else if (source[index] === "/" && source[index + 1] === "*") {
      syntaxCharacters[index++] = " ";syntaxCharacters[index++] = " ";
      while (index < syntaxCharacters.length
        && !(source[index] === "*" && source[index + 1] === "/")) {
        if (source[index] !== "\n") syntaxCharacters[index] = " ";
        index += 1;
      }
      if (index < syntaxCharacters.length) {
        syntaxCharacters[index++] = " ";syntaxCharacters[index++] = " ";
      }
    } else index += 1;
  }
  const syntaxSource = syntaxCharacters.join("");
  const spans: FunctionSpan[] = [];
  const declaration = /(?:@\w+(?:\([^)]*\))?\s*)*fn\s+([A-Za-z_]\w*)\s*\(/g;
  for (let match = declaration.exec(syntaxSource); match;
    match = declaration.exec(syntaxSource)) {
    const open = syntaxSource.indexOf("{", declaration.lastIndex);
    if (open < 0) throw new Error(`WGSL function ${match[1]} has no body`);
    let depth = 0, end = open;
    for (; end < syntaxSource.length; end += 1) {
      const character = syntaxSource[end]!;
      if (character === "{") depth += 1;
      else if (character === "}" && --depth === 0) { end += 1; break; }
    }
    if (depth !== 0) throw new Error(`WGSL function ${match[1]} has an unclosed body`);
    spans.push({ name: match[1]!, start: match.index, end,
      body: syntaxSource.slice(open, end) });
    declaration.lastIndex = end;
  }
  const byName = new Map(spans.map((span) => [span.name, span]));
  const retained = new Set<string>();
  const pending = roots.filter((root) => byName.has(root));
  while (pending.length > 0) {
    const name = pending.pop()!;
    if (retained.has(name)) continue;
    retained.add(name);
    const body = byName.get(name)!.body;
    for (const call of body.matchAll(/\b([A-Za-z_]\w*)\s*\(/g)) {
      const dependency = call[1]!;
      if (byName.has(dependency) && !retained.has(dependency)) pending.push(dependency);
    }
  }
  const insideFunction = (offset: number) => spans.some((span) =>
    offset >= span.start && offset < span.end);
  const globals: GlobalSpan[] = [];
  const addSimpleGlobals = (pattern: RegExp) => {
    for (let match = pattern.exec(syntaxSource); match;
      match = pattern.exec(syntaxSource)) {
      if (insideFunction(match.index)) continue;
      globals.push({ name: match[1]!, start: match.index,
        end: pattern.lastIndex, text: match[0] });
    }
  };
  // WGSL has no executable global initializers. These declaration forms are
  // therefore sufficient to close the lexical dependency graph of a sliced
  // entry point. Keeping every declaration had left each tiny pipeline with
  // the monolith's complete binding and workgroup-memory topology.
  addSimpleGlobals(/(?:@\w+(?:\([^)]*\))?\s*)*\bvar(?:<[^>]+>)?\s*([A-Za-z_]\w*)[^;]*;/g);
  addSimpleGlobals(/\b(?:const|override|alias)\s+([A-Za-z_]\w*)[^;]*;/g);
  const structPattern = /\bstruct\s+([A-Za-z_]\w*)\s*\{/g;
  for (let match = structPattern.exec(syntaxSource); match;
    match = structPattern.exec(syntaxSource)) {
    if (insideFunction(match.index)) continue;
    const open = syntaxSource.indexOf("{", match.index);
    let depth = 0, end = open;
    for (; end < syntaxSource.length; end += 1) {
      if (syntaxSource[end] === "{") depth += 1;
      else if (syntaxSource[end] === "}" && --depth === 0) {
        end += 1;
        if (syntaxSource[end] === ";") end += 1;
        break;
      }
    }
    globals.push({ name: match[1]!, start: match.index, end,
      text: source.slice(match.index, end) });
    structPattern.lastIndex = end;
  }
  const globalByName = new Map(globals.map((span) => [span.name, span]));
  const requiredGlobals = new Set<string>();
  const globalPending: string[] = [];
  const enqueueIdentifiers = (text: string) => {
    for (const token of text.matchAll(/\b([A-Za-z_]\w*)\b/g)) {
      const name = token[1]!;
      if (globalByName.has(name) && !requiredGlobals.has(name)) globalPending.push(name);
    }
  };
  for (const name of retained) enqueueIdentifiers(
    source.slice(byName.get(name)!.start, byName.get(name)!.end));
  while (globalPending.length > 0) {
    const name = globalPending.pop()!;
    if (requiredGlobals.has(name)) continue;
    requiredGlobals.add(name);
    enqueueIdentifiers(globalByName.get(name)!.text);
  }
  const removable = [
    ...spans.filter((span) => !retained.has(span.name)),
    ...globals.filter((span) => !requiredGlobals.has(span.name)),
  ].sort((left, right) => left.start - right.start);
  let result = "", cursor = 0;
  for (const span of removable) {
    if (span.start < cursor) continue;
    result += source.slice(cursor, span.start);
    cursor = span.end;
  }
  return result + source.slice(cursor);
}
const SPARSE_CM12_PHASE1_TRANSPORT_PROFILE_WORDS = 64;
/** Params in the resident WGSL, including the fixed authored-region tail. */
const SPARSE_CM12_PARAMETER_BYTES = SPARSE_CM12_REFINEMENT_REGION_PARAMETER_OFFSET
  + SPARSE_CM12_REFINEMENT_REGION_BYTES;
/** Twenty f32 convergence/diagnostic scalars; see the WGSL initialization. */
const SPARSE_CM12_PRESSURE_SCALAR_BYTES = 80;
const SPARSE_CM12_PCM_DIAGNOSTIC_DOMAIN_WORDS =
  SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER_WORDS;
const SPARSE_CM12_PCM_DIAGNOSTIC_BYTES =
  2 * 4 * SPARSE_CM12_PCM_DIAGNOSTIC_DOMAIN_WORDS;
const SPARSE_CM12_PRESSURE_CUTOVER_DIAGNOSTIC_WORDS =
  SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS
  + SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS
  + 4 * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS;

export const SPARSE_CM12_PRESSURE_ITERATIONS = 128;
/** A conservative interactive default; zero still requests fixed-budget work. */
export const SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE = 1e-3;
/** Sixteen fixed eight-iteration blocks cover the default 128-iteration ceiling. */
export const SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE = 8;
const SPARSE_CM12_PRESSURE_ITERATIONS_MINIMUM = 8;
const SPARSE_CM12_PRESSURE_ITERATIONS_MAXIMUM = 256;
const SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE_MAXIMUM = 1;

export interface SparseCM12PressureControl {
  readonly iterations?: number;
  readonly relativeTolerance?: number;
}

export interface SparseCM12InflowControl {
  readonly outletFine: readonly [number, number, number];
  readonly radiusFine: number;
  readonly velocityFinePerSecond: readonly [number, number, number];
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
      Math.max(SPARSE_CM12_PRESSURE_ITERATIONS_MINIMUM,
        SPARSE_CM12_PRESSURE_ITERATIONS_MINIMUM
        * Math.round(value / SPARSE_CM12_PRESSURE_ITERATIONS_MINIMUM)))
    : SPARSE_CM12_PRESSURE_ITERATIONS;

export const sparseCM12PressureRelativeTolerance = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE_MAXIMUM,
      Math.max(0, value))
    : SPARSE_CM12_PRESSURE_RELATIVE_TOLERANCE;

/**
 * Choose the next encoded ceiling from the last queue-confirmed solve.
 *
 * A converged frame keeps one residual block in reserve. A frame that consumed
 * its entire smaller ceiling doubles that ceiling so a newly disturbed scene
 * returns to the hard budget quickly. No receipt, a disabled tolerance, and an
 * explicitly armed pressure journal retain the caller's complete ceiling.
 */
export function sparseCM12PressureIterationsFromReceipt(
  maximum: unknown,
  relativeTolerance: unknown,
  previous?: Readonly<{ executed: number; encoded: number }>,
): number {
  const hardMaximum = sparseCM12PressureIterations(maximum);
  if (!(sparseCM12PressureRelativeTolerance(relativeTolerance) > 0) || !previous
    || !Number.isFinite(previous.executed) || !Number.isFinite(previous.encoded)) {
    return hardMaximum;
  }
  const executed = Math.max(0, Math.round(previous.executed));
  const encoded = sparseCM12PressureIterations(previous.encoded);
  const next = executed >= encoded
    ? Math.max(encoded + SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE, 2 * encoded)
    : executed + SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE;
  return Math.min(hardMaximum, sparseCM12PressureIterations(next));
}
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
const GPU_TOPOLOGY_CELL_PAGE_HEADER_WORDS = 16;
const gpuTopologyCellPageWords = (brickFineResolution: number) =>
  GPU_TOPOLOGY_CELL_PAGE_HEADER_WORDS
  // Seven structure-of-array row planes (two uniform values are implicit),
  // with two words per term and one two-word incidence override per boundary
  // cell face. Uniform geometry and interior incidences are arithmetic.
  + 7 * (3 * (brickFineResolution + 1) * brickFineResolution ** 2)
  + 4 * (3 * (brickFineResolution + 1) * brickFineResolution ** 2)
  + 12 * brickFineResolution ** 2;

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

// CPU-only sizing oracle retained by topology reports; production uses only
// accepted topology plus GPU-published frontier pages.
export const SPARSE_CM12_HOST_TEMPLATE_MUTABLE_BRICK_MAXIMUM = 2048;

/** CPU-only sizing oracle for historical all-rung topology reports. */
export function sparseCM12HostTemplateVariantsEnabled(
  acceptedCellCount: number,
  acceptedRowCount: number,
  mutableBrickCount: number,
  brickFineResolution = 8,
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
  brickFineResolution = 8,
): SparseCM12TopologyPagePoolPlan {
  const pageWords = gpuTopologyCellPageWords(brickFineResolution);
  if (!enabled || mutableFrontierBricks <= 0) return {
    pageCapacity: 0, freeListWords: 0,
    pageWords, descriptorWords: 0,
  };
  const pageCapacity = Math.min(GPU_TOPOLOGY_PAGE_POOL_MAXIMUM,
    Math.max(GPU_TOPOLOGY_PAGE_POOL_MINIMUM, Math.ceil(mutableFrontierBricks)));
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
const TEMPLATE_HEADER_WORDS = 27;
// SolidWorld fractions live only in their derived GPU state arrays. The
// immutable topology record therefore carries geometry plus one packed
// [brick:27 | resolution:5] word: a 32-byte record.
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

/** Generation-zero directional frontier intent without allocating a world page. */
function sparseCM12AuthoredFluidFrontierMask(
  atlas: SparseAdaptiveMassAtlas,
  brick: SparseAdaptiveMassBrick,
): number {
  if (sparseBrickSpan(brick) !== 1) return 0;
  const resolution = brick.resolution;
  let mask = 0;
  for (let axis = 0; axis < 3; axis += 1) for (const direction of [-1, 1]) {
    const target = [...brick.coordinate] as [number, number, number];
    target[axis] += direction;
    if (target[axis] < 0 || target[axis] >= atlas.brickDimensions[axis]) continue;
    const fixed = direction < 0 ? 0 : resolution - 1;
    let wet = false;
    for (let v = 0; v < resolution && !wet; v += 1)
      for (let u = 0; u < resolution; u += 1) {
        const q = axis === 0 ? [fixed, u, v]
          : axis === 1 ? [u, fixed, v] : [u, v, fixed];
        const local = q[0] + resolution * (q[1] + resolution * q[2]);
        if (brick.density[local]! > 0) { wet = true; break; }
      }
    if (!wet) continue;
    const delta = [0, 0, 0]; delta[axis] = direction;
    const bit = 1 + delta[0]! + 3 * (1 + delta[1]!) + 9 * (1 + delta[2]!);
    mask |= 2 ** bit;
  }
  return mask >>> 0;
}
// Immutable rows are read field-wise by wide shader invocations. Nine SoA
// planes keep those reads contiguous while two packed planes preserve every
// integer field: [term offset:23 | count:9] and
// [requirement offset:28 | kind:2 | axis:2]. Geometry remains exact f32 bits.
// Nine count bits cover the widest supported B16 macro/fine face (256 fine
// endpoints plus its coarse endpoint); 23 offset bits still address 8,388,608
// two-word term records (64 MiB) before any other topology section is counted.
const TEMPLATE_ROW_PLANE_COUNT = 9;
const TEMPLATE_ROW_TERM_OFFSET_BITS = 23;
const TEMPLATE_ROW_TERM_OFFSET_MASK = 0x007f_ffff;
const TEMPLATE_ROW_TERM_COUNT_MASK = 0x1ff;
const TEMPLATE_ROW_METADATA_OFFSET_MASK = 0x0fff_ffff;

const templateRowWord = (
  rowOffset: number, rowCount: number, plane: number, row: number,
) => rowOffset + plane * rowCount + row;

function packedTemplateRowTerms(offset: number, count: number): number {
  if (offset > TEMPLATE_ROW_TERM_OFFSET_MASK || count > TEMPLATE_ROW_TERM_COUNT_MASK) {
    throw new Error(`Sparse CM12 row term range cannot be packed: ${offset}+${count}`);
  }
  return (offset | (count << TEMPLATE_ROW_TERM_OFFSET_BITS)) >>> 0;
}

function packedTemplateRowMetadata(
  requirementOffset: number, kind: SparseAtlasGradientRow["kind"], axis: number,
): number {
  if (requirementOffset > TEMPLATE_ROW_METADATA_OFFSET_MASK || axis > 3) {
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
  /** Maximum rows in one [owner brick, accepted resolution] interval. */
  readonly maximumOwnedRowCount: number;
}

interface SparseCM12CandidateFaceCatalog {
  /** Relative indices into boundaryOffsets, one per brick/accepted/side. */
  readonly configurations: Uint32Array;
  /** One row range per accepted boundary cell. */
  readonly patchOffsets: Uint32Array;
  readonly rows: Uint32Array;
}

function sparseCM12CandidateFaceCatalog(
  atlas: SparseAdaptiveMassAtlas,
  templateLevels: readonly SparseBrickResolution[],
  cells: readonly SparseAtlasCompositeCell[],
  rows: readonly SparseAtlasGradientRow[],
  rowRequirements: readonly (readonly number[])[],
  brickIndex: ReadonlyMap<number, number>,
): SparseCM12CandidateFaceCatalog {
  const levels = templateLevels.length;
  const configurations = new Uint32Array(atlas.bricks.length * levels * 6);
  let patchOffsetCount = 0;
  for (let brick = 0; brick < atlas.bricks.length; brick += 1) {
    for (let accepted = 0; accepted < levels; accepted += 1) {
      const boundaryCells = templateLevels[accepted]! ** 2;
      for (let side = 0; side < 6; side += 1) {
        const configuration = (brick * levels + accepted) * 6 + side;
        configurations[configuration] = patchOffsetCount;
        patchOffsetCount += boundaryCells + 1;
      }
    }
  }
  const patchCounts = new Uint32Array(patchOffsetCount);
  const visitMappings = (visit: (configuration: number, boundary: number, row: number) => void) => {
   for (const row of rows) {
    for (const requirement of rowRequirements[row.id]!) {
      const brick = requirement >>> TEMPLATE_CELL_RESOLUTION_BITS;
      const resolution = requirement & TEMPLATE_CELL_RESOLUTION_MASK;
      const acceptedLevel = templateLevels.indexOf(resolution as SparseBrickResolution);
      if (acceptedLevel < 0) continue;
      let claimant = Number.MAX_SAFE_INTEGER;
      for (const term of row.terms) {
        const cell = cells[term.cellId]!;
        if (brickIndex.get(cell.brickKey) === brick) claimant = Math.min(claimant, term.cellId);
      }
      if (!Number.isFinite(claimant) || claimant === Number.MAX_SAFE_INTEGER) continue;
      const source = cells[claimant]!;
      if (source.brickResolution !== resolution) continue;
      const sourceBrick = atlas.bricks[brick]!;
      const origin = sourceBrick.coordinate.map((coordinate) =>
        coordinate * atlas.brickFineResolution);
      const width = atlas.brickFineResolution * sparseBrickSpan(sourceBrick);
      const axis = row.axis;
      let side = -1;
      if (Math.abs(row.centerFine[axis]! - origin[axis]!) <= 1e-4) side = 2 * axis;
      else if (Math.abs(row.centerFine[axis]!
        - (origin[axis]! + width)) <= 1e-4) side = 2 * axis + 1;
      if (side < 0) continue;
      const z = Math.floor(source.localIndex / (resolution * resolution));
      const yz = source.localIndex - z * resolution * resolution;
      const y = Math.floor(yz / resolution), x = yz - y * resolution;
      const u = axis === 0 ? y : x;
      const v = axis === 2 ? y : z;
      const configuration = (brick * levels + acceptedLevel) * 6 + side;
      visit(configuration, u + resolution * v, row.id);
    }
   }
  };
  visitMappings((configuration, boundary) => {
    patchCounts[configurations[configuration]! + boundary + 1] += 1;
  });
  const patchOffsets = new Uint32Array(patchOffsetCount);
  let total = 0;
  for (let configuration = 0; configuration < configurations.length; configuration += 1) {
    const base = configurations[configuration]!;
    const next = configuration + 1 < configurations.length
      ? configurations[configuration + 1]! : patchOffsetCount;
    patchOffsets[base] = total;
    for (let at = base + 1; at < next; at += 1) {
      total += patchCounts[at]!;
      patchOffsets[at] = total;
    }
  }
  const outputRows = new Uint32Array(total);
  const cursors = patchOffsets.slice();
  visitMappings((configuration, boundary, row) => {
    const range = configurations[configuration]! + boundary;
    outputRows[cursors[range]++] = row;
  });
  return { configurations, patchOffsets, rows: outputRows };
}

interface SparseCM12ContiguousRowOwnership {
  readonly rows: readonly SparseAtlasGradientRow[];
  readonly requirements: readonly (readonly number[])[];
  readonly offsets: Uint32Array;
  readonly oldToNew: Uint32Array;
  readonly maximumOwnedRowCount: number;
}

/**
 * Give every row one owner and make each [brick, rung] ownership interval the
 * row ID interval itself. Consumers can then enumerate `first + local`
 * without loading a row list or a row-to-owner plane.
 */
function sparseCM12ContiguousRowOwnership(
  brickCount: number,
  templateLevels: readonly SparseBrickResolution[],
  rows: readonly SparseAtlasGradientRow[],
  rowRequirements: readonly (readonly number[])[],
): SparseCM12ContiguousRowOwnership {
  if (rows.length !== rowRequirements.length) {
    throw new Error("Sparse CM12 row ownership input lengths differ");
  }
  const buckets: SparseAtlasGradientRow[][] = Array.from(
    { length: brickCount * templateLevels.length }, () => [],
  );
  rowRequirements.forEach((requirements, oldRow) => {
    if (requirements.length === 0) {
      throw new Error(`Sparse CM12 row ${oldRow} has no owner`);
    }
    const owner = requirements.reduce((minimum, packed) =>
      (packed >>> TEMPLATE_CELL_RESOLUTION_BITS)
        < (minimum >>> TEMPLATE_CELL_RESOLUTION_BITS) ? packed : minimum);
    const brick = owner >>> TEMPLATE_CELL_RESOLUTION_BITS;
    const resolution = owner & TEMPLATE_CELL_RESOLUTION_MASK;
    const level = templateLevels.indexOf(resolution as SparseBrickResolution);
    if (brick >= brickCount || level < 0) {
      throw new Error(`Sparse CM12 row ${oldRow} has invalid owner ${brick}/${resolution}`);
    }
    buckets[templateLevels.length * brick + level]!.push(rows[oldRow]!);
  });
  const offsets = new Uint32Array(buckets.length + 1);
  for (let bucket = 0; bucket < buckets.length; bucket += 1) {
    offsets[bucket + 1] = offsets[bucket]! + buckets[bucket]!.length;
  }
  const orderedRows: SparseAtlasGradientRow[] = [];
  const orderedRequirements: (readonly number[])[] = [];
  const oldToNew = new Uint32Array(rows.length);
  let maximumOwnedRowCount = 0;
  for (let bucket = 0; bucket < buckets.length; bucket += 1) {
    maximumOwnedRowCount = Math.max(maximumOwnedRowCount,
      offsets[bucket + 1]! - offsets[bucket]!);
    for (const source of buckets[bucket]!) {
      const id = orderedRows.length;
      oldToNew[source.id] = id;
      orderedRows.push({ ...source, id });
      orderedRequirements.push(rowRequirements[source.id]!);
    }
  }
  if (orderedRows.length !== rowRequirements.length) {
    throw new Error("Sparse CM12 row-owner catalog does not partition template rows");
  }
  return { rows: orderedRows, requirements: orderedRequirements, offsets,
    oldToNew, maximumOwnedRowCount };
}

interface SparseCM12AdaptiveStructureCatalog {
  readonly candidateFaces: SparseCM12CandidateFaceCatalog;
}

/**
 * Build the immutable candidate-face view once. Frame transactions add only
 * compact slot membership and dynamic packets; gamma has no private topology.
 */
function sparseCM12AdaptiveStructureCatalog(
  atlas: SparseAdaptiveMassAtlas,
  templateLevels: readonly SparseBrickResolution[],
  cells: readonly SparseAtlasCompositeCell[],
  rows: readonly SparseAtlasGradientRow[],
  rowRequirements: readonly (readonly number[])[],
  brickIndex: ReadonlyMap<number, number>,
): SparseCM12AdaptiveStructureCatalog {
  return {
    candidateFaces: sparseCM12CandidateFaceCatalog(atlas, templateLevels,
      cells, rows, rowRequirements, brickIndex),
  };
}

/** Serialize one accepted generation without duplicating its object graph. */
function packAcceptedTopologyTemplates(
  atlas: SparseAdaptiveMassAtlas,
  grid: SparseAtlasCompositeGrid,
): PackedResidentTopologyTemplates {
  const templateLevels = sparseCM12TemplateLevels(atlas.brickFineResolution);
  const cells = grid.cells, sourceRows = grid.gradientRows;
  const brickIndex = new Map(atlas.bricks.map((brick, index) => [brick.key, index]));
  const sourceRowRequirements = sourceRows.map((row) => {
    const requirements = new Map<number, number>();
    for (const term of row.terms) {
      const cell = cells[term.cellId]!;
      requirements.set(brickIndex.get(cell.brickKey)!, cell.brickResolution);
    }
    return [...requirements].map(([brick, resolution]) =>
      packedTemplateCellMetadata(brick, resolution));
  });
  const ownership = sparseCM12ContiguousRowOwnership(atlas.bricks.length,
    templateLevels, sourceRows, sourceRowRequirements);
  const rows = ownership.rows;
  const rowRequirements = ownership.requirements;
  const structure = sparseCM12AdaptiveStructureCatalog(
    atlas, templateLevels, cells, rows, rowRequirements, brickIndex);
  const candidateFaces = structure.candidateFaces;
  const cellCountByBrick = new Uint32Array(atlas.bricks.length);
  let termCount = 0, requirementWords = 0;
  const incidenceCounts = new Uint32Array(cells.length);
  for (const row of rows) {
    termCount += row.terms.length;
    for (const term of row.terms) {
      incidenceCounts[term.cellId] += 1;
    }
    requirementWords += 1 + rowRequirements[row.id]!.length;
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
  const rowOwnerRangeOffset = at; at += ownership.offsets.length;
  const candidateFaceConfigurationOffset = at; at += candidateFaces.configurations.length;
  const candidateFacePatchOffset = at; at += candidateFaces.patchOffsets.length;
  const candidateFaceRowOffset = at; at += candidateFaces.rows.length;
  const pressureEdgeOffset = at; at += cells.length + 1;
  const pressureEdgeRecordOffset = at; at += 3 * pressureEdgeCount;
  const words = new Uint32Array(at);
  words.set([TEMPLATE_MAGIC, 1, cells.length, rows.length, termCount, incidenceCount,
    cellOffset, rowOffset, termOffset, incidenceOffset, incidenceRecordOffset,
    cellRangeOffset, rowRequirementOffset, atlas.bricks.length], 0);
  words[15] = pressureEdgeOffset;
  words[16] = rowOwnerRangeOffset;
  words[17] = 0;
  words[18] = rows.length;
  words[24] = candidateFaceConfigurationOffset;
  words[25] = candidateFacePatchOffset;
  words[26] = candidateFaceRowOffset;
  for (const cell of cells) {
    const base = cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell.id;
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
    for (const term of row.terms) {
      const cell = cells[term.cellId]!;
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
    const requirements = rowRequirements[row.id]!;
    words[requirementAt++] = requirements.length;
    words.set(requirements, requirementAt); requirementAt += requirements.length;
  }
  words.set(ownership.offsets, rowOwnerRangeOffset);
  for (let configuration = 0; configuration < candidateFaces.configurations.length;
    configuration += 1) {
    words[candidateFaceConfigurationOffset + configuration]
      = candidateFacePatchOffset + candidateFaces.configurations[configuration]!;
  }
  for (let patch = 0; patch < candidateFaces.patchOffsets.length; patch += 1) {
    words[candidateFacePatchOffset + patch]
      = candidateFaceRowOffset + candidateFaces.patchOffsets[patch]!;
  }
  words.set(candidateFaces.rows, candidateFaceRowOffset);
  return {
    words, cellCount: cells.length, rowCount: rows.length,
    initialCellWorklist: Uint32Array.from({ length: cells.length }, (_, id) => id),
    initialRowWorklist: Uint32Array.from({ length: rows.length }, (_, id) => id),
    initialDensity: Float32Array.from(cells, (cell) => cell.density),
    initialGamma: Float32Array.from(cells, (cell) => cell.gamma),
    maximumOwnedRowCount: ownership.maximumOwnedRowCount,
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
  // Even a bounded initial frontier is large enough that building one
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
    resampleBrick(brick, choose(brick))), atlas.generation,
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

  let rows: SparseAtlasGradientRow[] = [];
  let rowRequirements: number[][] = [];
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
  const initialAcceptedRowCount = rows.length;
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

  const ownership = sparseCM12ContiguousRowOwnership(atlas.bricks.length,
    templateLevels, rows, rowRequirements);
  rows = Array.from(ownership.rows);
  rowRequirements = ownership.requirements.map((requirements) => [...requirements]);

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
  const structure = sparseCM12AdaptiveStructureCatalog(
    atlas, templateLevels, cells, rows, rowRequirements, brickIndex);
  const candidateFaces = structure.candidateFaces;
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
  const rowOwnerRangeOffset = at; at += ownership.offsets.length;
  const candidateFaceConfigurationOffset = at; at += candidateFaces.configurations.length;
  const candidateFacePatchOffset = at; at += candidateFaces.patchOffsets.length;
  const candidateFaceRowOffset = at; at += candidateFaces.rows.length;
  const pressureEdgeOffset = at; at += cells.length + 1;
  const pressureEdgeRecordOffset = at; at += 3 * pressureEdgeCount;
  const words = new Uint32Array(at);
  words.set([TEMPLATE_MAGIC, 1, cells.length, rows.length, termCount, incidenceCount,
    cellOffset, rowOffset, termOffset, incidenceOffset, incidenceRecordOffset,
    cellRangeOffset, rowRequirementOffset, atlas.bricks.length], 0);
  words[15] = pressureEdgeOffset;
  words[16] = rowOwnerRangeOffset;
  words[17] = 0;
  words[18] = rows.length;
  words[24] = candidateFaceConfigurationOffset;
  words[25] = candidateFacePatchOffset;
  words[26] = candidateFaceRowOffset;
  for (const cell of cells) {
    const base = cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell.id;
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
  words.set(ownership.offsets, rowOwnerRangeOffset);
  for (let configuration = 0; configuration < candidateFaces.configurations.length;
    configuration += 1) {
    words[candidateFaceConfigurationOffset + configuration]
      = candidateFacePatchOffset + candidateFaces.configurations[configuration]!;
  }
  for (let patch = 0; patch < candidateFaces.patchOffsets.length; patch += 1) {
    words[candidateFacePatchOffset + patch]
      = candidateFaceRowOffset + candidateFaces.patchOffsets[patch]!;
  }
  words.set(candidateFaces.rows, candidateFaceRowOffset);
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
  const initialRowWorklist = Uint32Array.from({ length: initialAcceptedRowCount },
    (_, oldId) => ownership.oldToNew[oldId]!);
  return { words, cellCount: cells.length, rowCount: rows.length,
    initialCellWorklist, initialRowWorklist,
    initialDensity: Float32Array.from(cells, (cell) => cell.density),
    initialGamma: Float32Array.from(cells, (cell) => cell.gamma),
    maximumOwnedRowCount: ownership.maximumOwnedRowCount };
}

/** CPU-only QA seam for censusing the full all-rung production branch. */
export function packSparseCM12ResidentTopologyTemplatesForQA(
  atlas: SparseAdaptiveMassAtlas,
  acceptedGrid: SparseAtlasCompositeGrid,
): Readonly<{ words: Uint32Array; cellCount: number; rowCount: number }> {
  const packed = packResidentTopologyTemplates(atlas, acceptedGrid);
  return Object.freeze({ words: packed.words,
    cellCount: packed.cellCount, rowCount: packed.rowCount });
}

/** CPU-only QA seam for the accepted-only production template branch. */
export function packSparseCM12AcceptedTopologyTemplatesForQA(
  atlas: SparseAdaptiveMassAtlas,
  acceptedGrid: SparseAtlasCompositeGrid,
): Readonly<{ words: Uint32Array; cellCount: number; rowCount: number }> {
  const packed = packAcceptedTopologyTemplates(atlas, acceptedGrid);
  return Object.freeze({ words: packed.words,
    cellCount: packed.cellCount, rowCount: packed.rowCount });
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
  readonly solidCellOpen: number;
  readonly solidRowData: number;
  /** SolidWorld row openness, multiplied with dynamic-solid openness. */
  readonly solidVoxelRowOpen: number;
  /** SolidWorld cell openness, multiplied with dynamic-solid openness. */
  readonly solidVoxelCellOpen: number;
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
  /** Persistent swept-characteristic support consumed by FSM1 packet scheduling. */
  readonly transportCharacteristicSupport: number;
  /** Dense read cache derived from TEI/effective velocity for face tracing. */
  readonly faceVelocitySupport: number;
}

export interface SparseCM12FinePresentationPlan {
  readonly plan: FineLevelSetBrickPlan;
  readonly metadata: Uint32Array;
  readonly worklist: Uint32Array;
}

export type SparseCM12PresentationPageResolution = 4 | 8 | 16;
export const SPARSE_CM12_PRESENTATION_PAGE_SHIFT = 21;
// Signed sparse presentation keys use every metadata-key bit without creating
// a dense logical lattice: x/z are biased 11-bit coordinates and y is a
// biased signed 10-bit coordinate. This is a presentation-address limit, not a
// physical floor: WDR1 remains signed i32 and rejects a publication only if it
// reaches the finite compact-key envelope.
export const SPARSE_CM12_SIGNED_PAGE_XZ_MIN = -1024;
export const SPARSE_CM12_SIGNED_PAGE_X_MAX = 1023;
export const SPARSE_CM12_SIGNED_PAGE_Y_MIN = -512;
export const SPARSE_CM12_SIGNED_PAGE_Y_MAX = 511;
// 0xffffffff is the shared INVALID sentinel, so its sole coordinate is kept
// outside the valid key space. The reachable WDR1 bound is at most 767.
export const SPARSE_CM12_SIGNED_PAGE_Z_MAX = 1022;

export function encodeSparseCM12SignedPresentationKey(
  coordinate: readonly [number, number, number],
): number {
  const [x, y, z] = coordinate;
  if (![x, y, z].every(Number.isSafeInteger)
    || x < SPARSE_CM12_SIGNED_PAGE_XZ_MIN || x > SPARSE_CM12_SIGNED_PAGE_X_MAX
    || y < SPARSE_CM12_SIGNED_PAGE_Y_MIN || y > SPARSE_CM12_SIGNED_PAGE_Y_MAX
    || z < SPARSE_CM12_SIGNED_PAGE_XZ_MIN || z > SPARSE_CM12_SIGNED_PAGE_Z_MAX) {
    throw new RangeError(`Sparse CM12 signed presentation coordinate ${coordinate.join(",")} is not representable`);
  }
  return ((x + 1024) | ((y + 512) << 11) | ((z + 1024) << 21)) >>> 0;
}

export function decodeSparseCM12SignedPresentationKey(
  key: number,
): readonly [number, number, number] {
  if (!Number.isSafeInteger(key) || key < 0 || key > 0xffff_ffff || key === INVALID) {
    throw new RangeError(`Sparse CM12 signed presentation key ${key} is invalid`);
  }
  return [(key & 0x7ff) - 1024, ((key >>> 11) & 0x3ff) - 512,
    ((key >>> 21) & 0x7ff) - 1024];
}

/** The authored page box must fit; each later growth publication is checked
 * against the exact coordinate it actually consumes. The physical growth pool
 * cannot be projected independently onto all six directions here: the same
 * finite page slots are shared by those directions. */
export function sparseCM12SignedPresentationInitialWorldFits(
  pageDimensions: readonly [number, number, number],
): boolean {
  if (!pageDimensions.every((value) => Number.isSafeInteger(value) && value > 0)) {
    return false;
  }
  return 0 >= SPARSE_CM12_SIGNED_PAGE_XZ_MIN
    && 0 >= SPARSE_CM12_SIGNED_PAGE_Y_MIN
    && pageDimensions[0] - 1 <= SPARSE_CM12_SIGNED_PAGE_X_MAX
    && pageDimensions[1] - 1 <= SPARSE_CM12_SIGNED_PAGE_Y_MAX
    && pageDimensions[2] - 1 <= SPARSE_CM12_SIGNED_PAGE_Z_MAX;
}
export type SparseCM12ResidentInitializationReporter = (label: string) => void;

// Compact presentation source word. B/P <= 2 uses a three-bit
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
  brickFineResolution = 8,
  presentationPageResolution = brickFineResolution,
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
  brickFineResolution = 8,
  presentationPageResolution = brickFineResolution,
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
  presentationPageResolution: SparseCM12PresentationPageResolution = atlas.brickFineResolution,
  options: {
    readonly residentBrickKeys?: ReadonlySet<number>;
    readonly pageCapacity?: number;
    /** Sparse addressing offset only; it allocates no logical-domain table. */
    readonly coordinateOffsetPages?: readonly [number, number, number];
    readonly sampleDimensions?: readonly [number, number, number];
    /** Metadata keys are packed signed page coordinates, never dense extents. */
    readonly signedSparseAddressing?: boolean;
  } = {},
): SparseCM12FinePresentationPlan {
  const brickFineResolution = atlas.brickFineResolution;
  if (presentationPageResolution > brickFineResolution
    || brickFineResolution % presentationPageResolution !== 0) {
    throw new RangeError(`Sparse CM12 presentation page ${presentationPageResolution} does not divide brick ladder ${brickFineResolution}`);
  }
  const pagesPerAxis = brickFineResolution / presentationPageResolution;
  const coordinateOffsetPages = options.coordinateOffsetPages ?? [0, 0, 0];
  const sampleDimensions = options.sampleDimensions ?? atlas.dimensions;
  const signedSparseAddressing = options.signedSparseAddressing === true;
  if (signedSparseAddressing
    && (options.coordinateOffsetPages !== undefined || options.sampleDimensions !== undefined)) {
    throw new RangeError("Signed Sparse CM12 presentation cannot advertise a padded address lattice");
  }
  if (coordinateOffsetPages.some((value) => !Number.isSafeInteger(value) || value < 0)
    || sampleDimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Sparse CM12 presentation address lattice is invalid");
  }
  const brickDimensions = sampleDimensions.map((value) =>
    Math.ceil(value / presentationPageResolution)) as
    [number, number, number];
  const pages: { key: number; brick: number; octant: number; spanBricks: number }[] = [];
  let maximumSpanLog = 0;
  for (let brick = 0; brick < atlas.bricks.length; brick += 1) {
    const source = atlas.bricks[brick]!;
    if (options.residentBrickKeys && !options.residentBrickKeys.has(source.key)) continue;
    const spanBricks = sparseBrickSpan(source);
    const spanLog = Math.log2(spanBricks);
    maximumSpanLog = Math.max(maximumSpanLog, spanLog);
    // Macro leaves publish one page across their complete physical extent.
    // Ordinary leaves retain (B/P)^3 independently addressable pages.
    // This makes cost proportional to leaves while preserving deep liquid and
    // closed walls in the global presentation.
    if (spanBricks > 1) {
      const coordinate = source.coordinate.map((value, axis) =>
        coordinateOffsetPages[axis]! + pagesPerAxis * value) as
        [number, number, number];
      pages.push({
        key: signedSparseAddressing
          ? encodeSparseCM12SignedPresentationKey(coordinate)
          : coordinate[0] + brickDimensions[0]
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
          const coordinate = [coordinateOffsetPages[0] + pagesPerAxis * source.coordinate[0] + ox,
            coordinateOffsetPages[1] + pagesPerAxis * source.coordinate[1] + oy,
            coordinateOffsetPages[2] + pagesPerAxis * source.coordinate[2] + oz] as const;
          if (coordinate.some((value, axis) => value >= brickDimensions[axis])) continue;
          pages.push({
            key: signedSparseAddressing
              ? encodeSparseCM12SignedPresentationKey(coordinate)
              : coordinate[0] + brickDimensions[0]
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
  const pageCapacity = options.pageCapacity ?? pageCount;
  if (!Number.isSafeInteger(pageCapacity) || pageCapacity < pageCount) {
    throw new RangeError(`Sparse CM12 presentation capacity ${pageCapacity} cannot hold ${pageCount} resident pages`);
  }
  const metadata = new Uint32Array(FINE_LEVELSET_METADATA_WORDS * pageCapacity);
  metadata.fill(INVALID);
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
  const worklist = new Uint32Array(FINE_LEVELSET_WORKSET_HEADER_WORDS + pageCapacity);
  worklist.fill(INVALID, FINE_LEVELSET_WORKSET_HEADER_WORDS);
  worklist.set([1, pageCount, pageCapacity,
    (FINE_LEVELSET_COMPACT_LOOKUP_FLAG | 3 | (maximumSpanLog << 8)
      | (brickFineResolution << 16)
      | (presentationPageResolution << SPARSE_CM12_PRESENTATION_PAGE_SHIFT)
      | (signedSparseAddressing ? FINE_LEVELSET_SIGNED_SPARSE_ADDRESS_FLAG : 0)) >>> 0,
    Math.ceil(pageCount / WORKGROUP_SIZE), 1, 1]);
  for (let page = 0; page < pageCount; page += 1) {
    worklist[FINE_LEVELSET_WORKSET_HEADER_WORDS + page] = page;
  }
  const samplesPerBrick = presentationPageResolution ** 3;
  const payloadCapacityBytes = pageCapacity * samplesPerBrick * 4;
  const metadataCapacityBytes = metadata.byteLength;
  const worklistBytes = worklist.byteLength;
  return {
    metadata,
    worklist,
    plan: {
      domainOrigin: coordinateOffsetPages.map((value) =>
        -value * presentationPageResolution) as [number, number, number],
      finestCellDimensions: sampleDimensions,
      finestCellWidth: 1,
      fineFactor: 1,
      fineCellWidth: 1,
      brickResolution: presentationPageResolution,
      sampleDimensions,
      brickDimensions,
      logicalBrickCount: brickDimensions[0] * brickDimensions[1] * brickDimensions[2],
      maximumResidentBricks: pageCapacity,
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
  /** Stable WDR leaf identifier; equal to the packed brick index for authored leaves. */
  readonly leafId: number;
  /** Signed SparseWorld page coordinate when the complete-world QA lane is requested. */
  readonly coordinate?: readonly [number, number, number];
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
  readonly acceptedTopologyGeneration: number;
  /** Complete accepted census, including GPU-grown world leaves. */
  readonly residentBrickCount: number;
  readonly faultFlags: number;
  readonly newlyActivatedBrickCount: number;
  readonly preparedBrickCount: number;
  readonly committedBrickCount: number;
  readonly commitFailed: boolean;
  readonly records: readonly SparseCM12GPUActivityRecord[];
}

export interface SparseCM12PresentationPageAllocatorReceipt {
  readonly residentPages: number;
  readonly faultCode: number;
  readonly highWaterMark: number;
  readonly cloneCount: number;
  readonly capacity: number;
}

export interface SparseCM12WorldGrowthReceipt {
  readonly initialLeaves: number;
  readonly liveLeaves: number;
  readonly capacity: number;
  readonly insertionFaults: number;
  readonly capacityFaults: number;
  readonly boundsGeneration: number;
  readonly minimum: readonly [number, number, number];
  readonly maximumExclusive: readonly [number, number, number];
  readonly claimedTopologyPages: number;
  readonly synthesizedTopologyPages: number;
  readonly publishedTopologyPages: number;
  readonly activeTopologyPages: number;
  readonly activeTransportTopologyPages: number;
  readonly connectedHostIncidences: number;
  readonly failedHostIncidences: number;
  readonly publishedTopologyPageCoordinates: readonly (readonly [number, number, number])[];
  readonly dynamicLiquidMassFineCells: number;
  /** Finest-lattice bounds of represented liquid in GPU-grown pages. */
  readonly dynamicLiquidBoundsFine?: {
    readonly minimum: readonly [number, number, number];
    readonly maximumExclusive: readonly [number, number, number];
  };
  readonly dynamicMaximumAbsFaceVelocityFineCells_s: number;
  readonly furthestLiquidLeafCoordinate?: readonly [number, number, number];
}

/** Explicit QA materialization. Production rendering consumes sparse buffers
 * directly and never constructs these finest-domain arrays. */
export interface SparseCM12DiagnosticFields {
  readonly density: Float32Array;
  readonly gamma: Float32Array;
  /** CM12 Sec. 3.6 non-solid capacity V_i, expanded over accepted cells. */
  readonly solidOpenFraction: Float32Array;
  readonly velocity: Float32Array;
  readonly pressure: Float32Array;
  readonly divergence: Float32Array;
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

/** Header-only receipt for the direct VEX2 packet transform. */
export interface SparseCM12VelocityExtensionHeaderQA {
  readonly sourceFrameGeneration: number;
  readonly topologyGeneration: number;
  readonly cellCapacity: number;
  readonly packetCapacity: number;
  readonly dispatchPacketCount: number;
  readonly validCellCount: number;
  readonly emptyPacketCount: number;
  readonly faultCount: number;
  readonly firstFault?: { readonly cell: number; readonly depth: number };
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
  denseCellCount: number,
  cutCellState: boolean,
  staticSolidVoxels: boolean,
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
    solidCellOpen: cutCellState ? cells() : 0,
    solidRowData: cutCellState ? solidRows() : 0,
    solidVoxelRowOpen: staticSolidVoxels ? rows() : 0,
    solidVoxelCellOpen: staticSolidVoxels ? cells() : 0,
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
    faceVelocitySupport: (() => {
      const result = at; at += align4(4 * denseCellCount); return result;
    })(),
    transportCharacteristicSupport: cells(),
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

  // The high bits of an authored leaf record identify its topology-complete
  // candidate field slot. SparseWorld's signed-page cutover accidentally
  // cleared this catalogue for every authored leaf, which made
  // brickCandidatePlanningEnabled false and silently discarded both rerung
  // and active-to-inactive lifecycle requests. Dynamic leaves still use their
  // page-local same-rung path; only span-one authored leaves with prepacked
  // all-rung templates receive a slot here.
  const candidateSlotByBrick = new Uint32Array(atlas.bricks.length).fill(INVALID);
  let candidateBrickCount = 0;
  for (let brick = 0; brick < atlas.bricks.length; brick += 1) {
    const source = atlas.bricks[brick]!;
    if (sparseBrickSpan(source) === 1 && mutableBrickKeys.has(source.key)) {
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
    // Cell ranges live only in the immutable template table. Log2(span) and
    // the optional candidate slot therefore share this compact owner word.
    words[record] = Math.log2(sparseBrickSpan(source))
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
  // Keep the immutable directed-edge record together. A packed nibble per cell
  // certifies the exact canonical interior pattern once; recurring SpMVs read
  // this ordinary pressure image and never interpret the atomic topology arena.
  const compactEdgeWords = templates.cellCount + 1 + 3 * edgeCount;
  const strictInteriorBase = TEMPLATE_HEADER_WORDS + compactEdgeWords;
  const strictInteriorWords = Math.ceil(templates.cellCount / 8);
  const coarseBase = strictInteriorBase + strictInteriorWords;
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
  const compactEdgeNeighbors = compactEdgeRows + edgeCount;
  const compactEdgeWeights = compactEdgeNeighbors + edgeCount;
  result.set(templates.words.subarray(sourceOffset,
    sourceOffset + templates.cellCount + 1), TEMPLATE_HEADER_WORDS);
  for (let edge = 0; edge < edgeCount; edge += 1) {
    result[compactEdgeRows + edge] = templates.words[edgeRecords + 3 * edge]!;
    result[compactEdgeNeighbors + edge] = templates.words[edgeRecords + 3 * edge + 1]!;
    result[compactEdgeWeights + edge] = templates.words[edgeRecords + 3 * edge + 2]!;
  }
  const strictWeightBits = new Uint32Array(1);
  const strictWeight = new Float32Array(strictWeightBits.buffer);
  for (let cell = 0; cell < templates.cellCount; cell += 1) {
    const begin = templates.words[sourceOffset + cell]!;
    const end = templates.words[sourceOffset + cell + 1]!;
    if (end - begin !== 6) continue;
    const brick = templateCellBrick(
      templates.words, cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell,
    );
    for (let code = 1; code <= 4; code += 1) {
      const resolution = 1 << (code - 1);
      const square = resolution * resolution;
      const expected = [cell - 1, cell + 1, cell - resolution, cell + resolution,
        cell - square, cell + square];
      strictWeight[0] = -resolution;
      const canonical = expected.every((other, local) =>
        templates.words[edgeRecords + 3 * (begin + local) + 1] === other
        && templates.words[edgeRecords + 3 * (begin + local) + 2]
          === strictWeightBits[0]
        && templateCellBrick(
          templates.words, cellOffset + TEMPLATE_CELL_RECORD_WORDS * other,
        ) === brick);
      if (!canonical) continue;
      result[strictInteriorBase + (cell >>> 3)]! |= code << (4 * (cell & 7));
      break;
    }
  }
  // Header words 13-15 are pressure hierarchy/coarse/edge descriptors. Word
  // 12 is deliberately private to this compact pressure image.
  result[12] = strictInteriorBase;
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

/** Aggregate ownership maps followed by the compact pressure bootstrap/fault
 * receipt. Immutable directed-edge topology lives exclusively in binding 14;
 * canonical PCM owns cell/row order. */
function pressureAuxiliaryArena(
  templates: PackedResidentTopologyTemplates,
  pressureTopology: Uint32Array,
  brickCount: number,
): { readonly words: Uint32Array; readonly layout: SparseCM12PressureRepairLayout } {
  const edgeOffsets = templates.words[15]!;
  const edgeCount = templates.words[edgeOffsets + templates.cellCount]!;
  const aggregateEdgeForFineEdgeBaseWords = 0;
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
    aggregateEdgeForFineEdgeBaseWords,
    aggregateEdgeSourceBaseWords,
    hierarchyEdgeForAggregateBaseWords: Object.freeze(hierarchyEdgeForAggregateBaseWords),
    headerBaseWords,
    totalWords: headerBaseWords + SPARSE_CM12_PRESSURE_REPAIR_HEADER_WORDS,
  });
  const result = new Uint32Array(layout.totalWords);
  result.fill(INVALID, aggregateEdgeForFineEdgeBaseWords,
    aggregateEdgeForFineEdgeBaseWords + edgeCount);
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
  private readonly pressureRepairLayout: SparseCM12PressureRepairLayout;
  /** Persistent compact activity census and per-physical-brick 4^3 tile mask. */
  readonly incrementalActivityLayout: SparseCM12IncrementalActivityLayout;
  /** Deterministic GPU pressure membership and rank-select arena. */
  private readonly canonicalMembershipLayout: SparseCM12CanonicalMembershipLayout;
  private readonly framePlanLayout: SparseCM12FramePlanLayout;
  private readonly framePlanPresentationLayout: SparseCM12FramePlanPresentationLayout;
  /** FCA1 GPU-owned frame generation, parity, predicates, and indirect ABI. */
  private readonly frameControlLayout: SparseCM12FrameControlLayout;
  private readonly velocityExtensionLayout: SparseCM12VelocityExtensionLayout;
  /** Immutable construction-only HEAD presentation publisher oracle. */
  private readonly presentationPublisherOracleForQA: boolean;
  private readonly horizontalD4Authority: boolean;
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
  /** GPU-authored bootstrap cell/row dispatches. The inactive epoch branch
   * publishes x=0, so the host encodes one fixed pressure schedule. */
  private readonly pressureMembershipIndirectArguments: GPUBuffer;
  /** Copy-isolated PEI1 wet-brick and hierarchy lane/reduction dispatches. */
  private readonly pressureExecutionIndirectArguments: GPUBuffer;
  /** Copy-isolated PCA1 seed, repair, and work dispatches. */
  private readonly persistentPressureCacheIndirectArguments: GPUBuffer;
  private readonly transportPacketIndirectArguments?: GPUBuffer;
  /** Pressure aggregate maps, receipts, and compiled execution-image tail. */
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
  private readonly transportBindGroup: GPUBindGroup;
  private readonly transportDepthBindGroups: readonly GPUBindGroup[];
  private readonly effectiveVelocityPressureBindGroup: GPUBindGroup;
  private readonly presentationAllocatorBindGroup: GPUBindGroup;
  private readonly transportExecutionImage?: GPUBuffer;
  private readonly transportExecutionImageLayout?: SparseCM12TransportExecutionImageLayout;
  private readonly effectiveTransportVelocity?: GPUBuffer;
  private readonly pipelines: Record<string, GPUComputePipeline>;
  private simulationPipelinesReady = false;
  private simulationPipelineFailure?: string;
  private simulationPipelineCompilation?: Promise<void>;
  private startSimulationPipelineCompilation?: () =>
    Promise<Readonly<Record<string, GPUComputePipeline>>>;
  private readonly parameterWords = new ArrayBuffer(SPARSE_CM12_PARAMETER_BYTES);
  private readonly parameterU32 = new Uint32Array(this.parameterWords);
  private readonly parameterF32 = new Float32Array(this.parameterWords);
  private readonly refinementRegionParameters =
    new Uint8Array(SPARSE_CM12_REFINEMENT_REGION_BYTES);
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
  /** One-shot command-prefix diagnostic used only by isolated Dawn bring-up. */
  private stageLimitForQA?: SparseCM12ResidentStageId;
  private activityPhaseLimitForQA?: "scalar" | "topology" | "masks" | "measure"
    | "history" | "census" | "allocation" | "synthesis" | "connection";
  private transportPhaseLimitForQA?: "setup" | "trace" | "scatter" | "gather";
  private pressureTopologyPhaseLimitForQA?: "setup" | "cells" | "rows" | "fine"
    | "coarse-plan" | "coarse-indirect" | "coarse-edge" | "coarse-work"
    | "coarse" | "hierarchy";
  private constructor(
    private readonly device: GPUDevice,
    private readonly dimensions: readonly [number, number, number],
    private readonly brickFineResolution: SparseAdaptiveMassAtlas["brickFineResolution"],
    private readonly layout: ResidentStateLayout,
    buffers: readonly [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer,
      GPUBuffer, GPUBuffer, GPUBuffer],
    acceptedIndirectArguments: GPUBuffer,
    pressureCellIndirectArguments: GPUBuffer,
    pressureMembershipIndirectArguments: GPUBuffer,
    pressureExecutionIndirectArguments: GPUBuffer,
    persistentPressureCacheIndirectArguments: GPUBuffer,
    transportPacketIndirectArguments: GPUBuffer | undefined,
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
    transportBindGroup: GPUBindGroup,
    transportDepthBindGroups: readonly GPUBindGroup[],
    effectiveVelocityPressureBindGroup: GPUBindGroup,
    presentationAllocatorBindGroup: GPUBindGroup,
    transportExecutionImage: GPUBuffer | undefined,
    transportExecutionImageLayout: SparseCM12TransportExecutionImageLayout | undefined,
    private readonly topologyEffectsAuthorityLayout:
      SparseCM12TopologyEffectsAuthorityLayout | undefined,
    private readonly iboSemanticAuthorityBaseWords: number | undefined,
    private readonly iboSlotBaseWords: readonly [number, number] | undefined,
    effectiveTransportVelocity: GPUBuffer | undefined,
    private readonly velocityExtensionDepths: GPUBuffer,
    pressureTemplates: GPUBuffer,
    pressureRepairLayout: SparseCM12PressureRepairLayout,
    private readonly pressureExecutionImageLayout:
      SparseCM12PressureExecutionImageLayout,
    incrementalActivityLayout: SparseCM12IncrementalActivityLayout,
    canonicalMembershipLayout: SparseCM12CanonicalMembershipLayout,
    framePlanLayout: SparseCM12FramePlanLayout,
    framePlanPresentationLayout: SparseCM12FramePlanPresentationLayout,
    frameControlLayout: SparseCM12FrameControlLayout,
    velocityExtensionLayout: SparseCM12VelocityExtensionLayout,
    private readonly faceAddressLayout: SparseCM12BrickTileFaceAddressLayout,
    private readonly transportPacketAuthorityLayout:
      SparseCM12TransportPacketAuthorityLayout | undefined,
    private readonly finalScalarPacketMaskLayout:
      SparseCM12FinalScalarPacketMaskLayout,
    private readonly phase1TransportQALayout:
      SparseCM12Phase1TransportQALayout | undefined,
    private readonly phase1TransportProfileBaseWords: number | undefined,
    private readonly pressureTopologyRepairLayout:
      SparseCM12PressureTopologyRepairLayout,
    private readonly persistentPressureCacheLayout:
      SparseCM12PersistentPressureCacheLayout,
    presentationPublisherOracleForQA: boolean,
    initialHorizontalD4Authority: boolean,
    pipelines: Readonly<Record<string, GPUComputePipeline>>,
    startSimulationPipelineCompilation: () =>
      Promise<Readonly<Record<string, GPUComputePipeline>>>,
    private readonly simulationCompilationSnapshot: () => GPUCompilationSnapshot,
    cellCount: number,
    rowCount: number,
    private readonly templateCellCount: number,
    private readonly templateRowCount: number,
    private readonly maximumOwnedRowCount: number,
    private readonly pressureCoarseEdgeCount: number,
    private readonly pressureFineEdgeCount: number,
    private readonly pressureHierarchyGroupCount: number,
    private readonly pressureHierarchyEdgeCount: number,
    private readonly pressureScratchBytes: number,
    private readonly pressureFineEdgeImageBaseWords: number,
    private readonly topologyWorklistBaseBytes: number,
    private readonly acceptedLeafManifestBaseBytes: number,
    private readonly topologyPageCapacity: number,
    private readonly topologyPageDescriptorBaseBytes: number,
    private readonly topologyPageWords: number,
    private readonly worldDirectoryLayout: SparseCM12WorldDirectoryLayout,
    private readonly solidOccupancyLayout:
      SparseCM12SolidOccupancyLayout | undefined,
    private readonly initialWorldLeafCount: number,
    private readonly initialBrickCoordinates:
      readonly (readonly [number, number, number])[],
    private readonly templateWords: Uint32Array,
    private readonly rigidCoupling?: WebGPUSparseCM12RigidCoupling,
  ) {
    [this.parameters, this.topology, this.state, this.partials, this.scalars,
      this.conditioning, this.activity, this.candidateState, this.topologyArena] = buffers;
    this.pressureRepairLayout = pressureRepairLayout;
    this.incrementalActivityLayout = incrementalActivityLayout;
    this.canonicalMembershipLayout = canonicalMembershipLayout;
    this.framePlanLayout = framePlanLayout;
    this.framePlanPresentationLayout = framePlanPresentationLayout;
    this.frameControlLayout = frameControlLayout;
    this.velocityExtensionLayout = velocityExtensionLayout;
    this.presentationPublisherOracleForQA = presentationPublisherOracleForQA;
    this.horizontalD4Authority = initialHorizontalD4Authority;
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
    this.pressureMembershipIndirectArguments = pressureMembershipIndirectArguments;
    this.pressureExecutionIndirectArguments = pressureExecutionIndirectArguments;
    this.persistentPressureCacheIndirectArguments =
      persistentPressureCacheIndirectArguments;
    this.transportPacketIndirectArguments = transportPacketIndirectArguments;
    this.framePlanIndirectArguments = framePlanIndirectArguments;
    this.presentationIndirectArguments = presentationIndirectArguments;
    this.frameControlIndirectArguments = frameControlIndirectArguments;
    this.presentationAllocatorBindGroup = presentationAllocatorBindGroup;
    this.transportDepthBindGroups = transportDepthBindGroups;
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
      worldDirectoryBaseWords: this.worldDirectoryLayout.baseWords,
      worldDirectoryInitialLeaves: this.worldDirectoryLayout.initialLeaves,
    };
    this.diagnosticsReadback = diagnosticsReadback;
    this.bindGroup = bindGroup;
    this.pressureBindGroup = pressureBindGroup;
    this.transportBindGroup = transportBindGroup;
    this.effectiveVelocityPressureBindGroup = effectiveVelocityPressureBindGroup;
    this.transportExecutionImage = transportExecutionImage;
    this.transportExecutionImageLayout = transportExecutionImageLayout;
    this.effectiveTransportVelocity = effectiveTransportVelocity;
    this.pressureTemplates = pressureTemplates;
    this.pipelines = { ...pipelines };
    this.startSimulationPipelineCompilation = startSimulationPipelineCompilation;
    this.cellCount = cellCount;
    this.rowCount = rowCount;
    this.allocatedBytes = [acceptedIndirectArguments, pressureCellIndirectArguments,
      pressureMembershipIndirectArguments,
      pressureExecutionIndirectArguments,
      framePlanIndirectArguments, presentationIndirectArguments,
      frameControlIndirectArguments,
      pressureTemplates, pressureWorklists,
      ...buffers, ...fineBuffers].reduce(
      (sum, buffer) => sum + buffer.size, 0,
    )
      + diagnosticsReadback.size
      + (transportExecutionImage?.size ?? 0)
      + (effectiveTransportVelocity?.size ?? 0)
      + (transportPacketIndirectArguments?.size ?? 0);
  }

  /** Restrict the next encoded frame to a stage prefix for isolated QA only. */
  setStageLimitForQA(stage: SparseCM12ResidentStageId | undefined): void {
    this.stageLimitForQA = stage;
  }

  setActivityPhaseLimitForQA(
    phase: "scalar" | "topology" | "masks" | "measure" | "history" | "census"
      | "allocation" | "synthesis" | "connection" | undefined,
  ): void {
    this.activityPhaseLimitForQA = phase;
  }

  setTransportPhaseLimitForQA(
    phase: "setup" | "trace" | "scatter" | "gather" | undefined,
  ): void {
    this.transportPhaseLimitForQA = phase;
  }
  setPressureTopologyPhaseLimitForQA(
    phase: "setup" | "cells" | "rows" | "fine" | "coarse-plan"
      | "coarse-indirect" | "coarse-edge" | "coarse-work" | "coarse"
      | "hierarchy" | undefined,
  ): void {
    this.pressureTopologyPhaseLimitForQA = phase;
  }

  static create(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    solidWorld: SolidWorld,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = atlas.brickFineResolution,
    report?: SparseCM12ResidentInitializationReporter,
  ): Promise<WebGPUSparseCM12Resident> {
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      solidWorld, initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      false, false, report);
  }

  /** QA-only construction path for the immutable HEAD presentation publisher.
   * Production frame state cannot select it and FPP1 never falls back to it. */
  static createPresentationPublisherOracleForQA(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    solidWorld: SolidWorld,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = atlas.brickFineResolution,
    report?: SparseCM12ResidentInitializationReporter,
  ): Promise<WebGPUSparseCM12Resident> {
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      solidWorld, initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      true, false, report);
  }

  /** Construction-only resident with raw Phase-1 transport receipts enabled.
   * Production reserves no receipt arena, so it cannot select or pay for this
   * diagnostic path at runtime. */
  static createPhase1TransportReceiptOracleForQA(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    solidWorld: SolidWorld,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = atlas.brickFineResolution,
    report?: SparseCM12ResidentInitializationReporter,
  ): Promise<WebGPUSparseCM12Resident> {
    return this.createConfigured(device, atlas, grid, finestCellSize_m,
      solidWorld, initiallyActiveBrickKeys, rigid, journal, presentationPageResolution,
      false, true, report);
  }

  private static async createConfigured(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    solidWorld: SolidWorld,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
    presentationPageResolution: SparseCM12PresentationPageResolution = atlas.brickFineResolution,
    presentationPublisherOracleForQA = false,
    phase1TransportReceiptForQA = false,
    report: SparseCM12ResidentInitializationReporter = () => {},
  ): Promise<WebGPUSparseCM12Resident> {
    if (atlas.brickFineResolution !== 8 || presentationPageResolution !== 8) {
      throw new Error("Sparse CM12 PEI1 production is an aggressive B8/P8 cutover");
    }
    const initialSolidWorld = solidWorld;
    const dynamicWorldGrowth = true;
    const signedWorldGrowth = true;
    // GPU-grown SparseWorld pages own a complete fixed-B8 graph, but authored
    // span-one leaves still need the prepacked dyadic catalogue for physical
    // 1/2/4/8 refinement and coarsening. Keep the catalogue bounded by the
    // established resident-work threshold; macro leaves remain immutable.
    const mutableBrickKeysForBudget = atlas.bricks.filter((brick) =>
      sparseBrickSpan(brick) === 1).map((brick) => brick.key);
    const hostTemplateVariants = sparseCM12HostTemplateVariantsEnabled(
      grid.cells.length, grid.gradientRows.length, mutableBrickKeysForBudget.length,
      atlas.brickFineResolution,
    );
    const mutableBrickKeys: ReadonlySet<number> = hostTemplateVariants
      ? new Set(mutableBrickKeysForBudget) : new Set<number>();
    report("Pack resident ownership topology");
    const packed = packResidentTopology(atlas, grid, mutableBrickKeys);
    // Sparse residency has one policy in every scene. A demanded adjacent page
    // is admitted from face-adjacent non-solid voxels; there is no tank-bound,
    // opening, or scene opt-in classification.
    const topologyPagePool = sparseCM12TopologyPagePoolPlan(
      // Physical working-set headroom follows authored liquid occupancy, not
      // the logical scene extent. A moving three-dimensional surface needs a
      // wet page plus both its adjacent-air and swept-destination companions;
      // two-times headroom filled before the canonical long dam reached its
      // far voxel wall. Retired pages still recycle as the course advances.
      // The fixed floor supports later injection into an initially empty scene.
      // A narrow reservoir can expose many dry course pages before its wake
      // becomes old enough to retire. Twelve pages per initially wet brick
      // covers that simultaneous moving band; the fixed 512-page ceiling
      // still bounds large scenes and page recycling remains authoritative.
      Math.max(1, 12 * initiallyActiveBrickKeys.size),
      true,
      atlas.brickFineResolution,
    );
    const worldLeafCapacity = packed.brickCount + topologyPagePool.pageCapacity;
    report("Build logical owner directory");
    const logicalOwnerDirectory = createSparseCM12LogicalOwnerDirectory(atlas, {
      brickFineResolution: atlas.brickFineResolution,
      presentationPageResolution,
    });
    const transportExecutionImageLayout = createSparseCM12TransportExecutionImageLayout({
        brickFineResolution: atlas.brickFineResolution as 8 | 16,
        logicalBrickDimensions: logicalOwnerDirectory.layout.logicalBrickDimensions,
        leafCapacity: worldLeafCapacity,
        maximumSpanBricks: atlas.maximumSpanBricks,
        logicalSlotsPerLeaf: Math.max(1, ...atlas.bricks.map((brick) => {
          const span = sparseBrickSpan(brick);
          const extent = brick.coordinate.map((origin, axis) => Math.max(0,
            Math.min(span, atlas.brickDimensions[axis]! - origin)));
          return (extent[2]! - 1) * span * span
            + (extent[1]! - 1) * span + extent[0]!;
        })),
      });
    // Runtime ownership is now the signed-coordinate WDR1 hash in the mutable
    // arena. LOD1 remains a construction oracle for initial TEI compilation,
    // but its world-volume-sized direct plane is no longer uploaded.
    const logicalOwnerBaseWords = 0;
    const logicalOwnerPacked16BaseWords = 0;
    report(hostTemplateVariants
      ? "Build four-rung and 2:1 seam topology templates"
      : "Pack accepted topology templates");
    const templates = hostTemplateVariants
      ? packResidentTopologyTemplates(atlas, grid)
      : packAcceptedTopologyTemplates(atlas, grid);
    const dynamicCellsPerPage = atlas.brickFineResolution ** 3;
    const dynamicRowsPerPage = 3 * (atlas.brickFineResolution + 1)
      * atlas.brickFineResolution ** 2;
    const physicsCellCapacity = templates.cellCount
      + topologyPagePool.pageCapacity * dynamicCellsPerPage;
    const physicsRowCapacity = templates.rowCount
      + topologyPagePool.pageCapacity * dynamicRowsPerPage;
    report("Plan resident GPU arenas");
    // Presentation has exactly one physical page slot per possible live leaf
    // in WDR1's bounded physical slab.
    const presentationPageCapacity = worldLeafCapacity;
    if (signedWorldGrowth) {
      const initialPages = atlas.dimensions.map((value) =>
        Math.ceil(value / presentationPageResolution));
      if (!sparseCM12SignedPresentationInitialWorldFits(
        initialPages as [number, number, number],
      )) {
        throw new RangeError("Sparse CM12 dynamic world exceeds the signed presentation address ABI");
      }
    }
    const fine = sparseCM12FinePresentationPlan(atlas, presentationPageResolution, {
      residentBrickKeys: initiallyActiveBrickKeys,
      pageCapacity: presentationPageCapacity,
      signedSparseAddressing: signedWorldGrowth,
    });
    (fine.plan as { finestCellWidth: number; fineCellWidth: number }).finestCellWidth =
      finestCellSize_m;
    (fine.plan as { finestCellWidth: number; fineCellWidth: number }).fineCellWidth =
      finestCellSize_m;
    (fine.plan as unknown as { domainOrigin: [number, number, number] }).domainOrigin =
      fine.plan.domainOrigin.map((value) => value * finestCellSize_m) as
        [number, number, number];
    const tracerLattice = sparseCM12TracerLattice(atlas.dimensions);
    const layout = residentStateLayout(
      physicsCellCapacity, physicsRowCapacity,
      atlas.dimensions[0]! * atlas.dimensions[1]! * atlas.dimensions[2]!,
      Boolean(rigid) || Boolean(initialSolidWorld),
      Boolean(initialSolidWorld),
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
    const cellWorkgroups = Math.ceil(physicsCellCapacity / WORKGROUP_SIZE);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const parameters = device.createBuffer({ label: "Sparse CM12 resident parameters",
      size: SPARSE_CM12_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const topology = uploadBuffer(device, "Sparse CM12 resident topology", packed.words, storage);
    // WebGPU buffers start zeroed. Upload only the nonzero ranges instead of
    // materializing the complete (mostly-zero) resident state on the host.
    const state = device.createBuffer({ label: "Sparse CM12 resident state",
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
    if (layout.solidRowData !== 0) {
      // Inactive dynamic pages are initialized on-device when synthesized.
      // Upload only the immutable host-template prefix instead of mirroring
      // the complete page-pool capacity in temporary host arrays.
      seed(layout.solidCellOpen, new Float32Array(templates.cellCount).fill(1));
      const solidRows = new Float32Array(3 * templates.rowCount);
      for (let row = 0; row < templates.rowCount; row += 1) {
        solidRows[3 * row] = 1;
        solidRows[3 * row + 2] = 1;
      }
      seed(layout.solidRowData, solidRows);
    }
    if (layout.solidVoxelRowOpen !== 0) {
      seed(layout.solidVoxelRowOpen, new Float32Array(templates.rowCount).fill(1));
      seed(layout.solidVoxelCellOpen, new Float32Array(templates.cellCount).fill(1));
    }
    const partials = device.createBuffer({ label: "Sparse CM12 resident reductions",
      size: Math.max(16, 16 * cellWorkgroups), usage: storage });
    const effectiveTransportVelocityLayout =
      createSparseCM12EffectiveTransportVelocityLayout(physicsCellCapacity);
    const effectiveTransportVelocity = effectiveTransportVelocityLayout
      ? device.createBuffer({
        label: "Sparse CM12 accepted effective transport velocity",
        size: effectiveTransportVelocityLayout.byteLength,
        usage: storage,
      })
      : undefined;
    const scalars = device.createBuffer({ label: "Sparse CM12 resident scalar reductions",
      size: SPARSE_CM12_PRESSURE_SCALAR_BYTES, usage: storage });
    const conditioning = device.createBuffer({
      label: "Sparse CM12 conservative transport and conditioning accumulators",
      // Transport needs beta, rho/gamma deficit, three momentum-deficit, and
      // one sharpening plane. Pressure reuses the dead arena for
      // cell and row headers plus their stable-ID compact worklists.
      size: 4 * Math.max(7 * physicsCellCapacity,
        physicsCellCapacity + physicsRowCapacity + 8),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const activityHistoryWords = ACTIVITY_HEADER_WORDS
      + ACTIVITY_RECORD_WORDS * worldLeafCapacity;
    const activityAlignmentWords = 64;
    const activityHistoryTailWords = Math.ceil(
      activityHistoryWords / activityAlignmentWords,
    ) * activityAlignmentWords;
    const incrementalActivityLayout = createSparseCM12IncrementalActivityLayout({
      baseWords: activityHistoryTailWords,
      brickCount: worldLeafCapacity,
      alignmentWords: activityAlignmentWords,
    });
    const canonicalMembershipLayout = createSparseCM12CanonicalMembershipLayout({
      baseWords: incrementalActivityLayout.totalWords,
      cellCapacity: physicsCellCapacity,
      rowCapacity: physicsRowCapacity,
    });
    const framePlanLayout = createSparseCM12FramePlanLayout({
      baseWords: Math.ceil(canonicalMembershipLayout.totalWords
        / activityAlignmentWords)
        * activityAlignmentWords,
      brickCapacity: worldLeafCapacity,
      brickFineResolution: atlas.brickFineResolution,
      packetCount: 6,
    });
    if (presentationPageResolution !== atlas.brickFineResolution) {
      throw new Error("FPL1/FPP1 resident cutover requires one B-sized presentation page");
    }
    const pageCount = fine.worklist[1]!;
    if (pageCount !== initiallyActiveBrickKeys.size) {
      throw new Error(`FPP1 generation zero requires one page per accepted brick; found ${pageCount}/${initiallyActiveBrickKeys.size}`);
    }
    const brickPages = new Uint32Array(worldLeafCapacity).fill(INVALID);
    for (let page = 0; page < pageCount; page += 1) {
      const source = fine.metadata[FINE_LEVELSET_METADATA_WORDS * page + 3]!;
      const brick = (source >>> 3) & 0x00ff_ffff;
      if (brick >= packed.brickCount || brickPages[brick] !== INVALID) {
        throw new Error(`FPP1 invalid or duplicate brick/page mapping ${brick}/${page}`);
      }
      brickPages[brick] = page;
    }
    for (let brick = 0; brick < packed.brickCount; brick += 1) {
      const expected = initiallyActiveBrickKeys.has(atlas.bricks[brick]!.key);
      if ((brickPages[brick] !== INVALID) !== expected) {
        throw new Error(`FPP1 generation-zero residency mismatch for brick ${brick}`);
      }
    }
    const framePlanPresentationLayout = createSparseCM12FramePlanPresentationLayout({
      baseWords: framePlanLayout.totalWords,
      pageCapacity: presentationPageCapacity,
      brickCapacity: worldLeafCapacity,
      brickFineResolution: atlas.brickFineResolution,
      pageResolution: presentationPageResolution,
      packetIndex: 5,
    });
    const velocityExtensionLayouts = createSparseCM12VelocityExtensionResidentLayouts({
      activityTailWords: framePlanPresentationLayout.totalWords,
      stateTailFloats: layout.transportCharacteristicSupport,
      cellCapacity: physicsCellCapacity,
      packetCapacity: transportExecutionImageLayout.packetCapacity,
      brickFineResolution: atlas.brickFineResolution as 4 | 8 | 16,
    });
    const velocityExtensionLayout = velocityExtensionLayouts.activity;
    if (velocityExtensionLayouts.state.characteristicSupportFloatBase
        !== layout.transportCharacteristicSupport
      || velocityExtensionLayouts.state.floatCount !== layout.floatCount) {
      throw new Error("VEX2 mask/transport-state composition mismatch: "
        + `base ${velocityExtensionLayouts.state.characteristicSupportFloatBase}`
        + `/${layout.transportCharacteristicSupport}, tail `
        + `${velocityExtensionLayouts.state.floatCount}/${layout.floatCount}, `
        + `cells ${physicsCellCapacity}`);
    }
    const transportPacketAuthorityLayout = transportExecutionImageLayout
      ? createSparseCM12TransportPacketAuthorityLayout({
        baseWords: Math.ceil(velocityExtensionLayout.totalWords / 64) * 64,
        packetCapacity: transportExecutionImageLayout.packetCapacity,
        dispatchPacketsPerLeaf: velocityExtensionLayout.dispatchPacketsPerLeaf,
        dispatchPacketCount: velocityExtensionLayout.dispatchPacketCount,
      })
      : undefined;
    const activityTailWords = transportPacketAuthorityLayout?.totalWords
      ?? velocityExtensionLayout.totalWords;
    const phase1TransportQALayout = phase1TransportReceiptForQA
      ? createSparseCM12Phase1TransportQALayout({
        baseWords: Math.ceil(activityTailWords / 64) * 64,
        cellCapacity: physicsCellCapacity,
      })
      : undefined;
    const phase1TransportProfileBaseWords = undefined;
    const phase1ActivityTailWords = phase1TransportProfileBaseWords === undefined
      ? (phase1TransportQALayout?.totalWords ?? activityTailWords)
      : phase1TransportProfileBaseWords + SPARSE_CM12_PHASE1_TRANSPORT_PROFILE_WORDS;
    const initialActivity = new Uint32Array(phase1ActivityTailWords);
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
    initialActivity.set(createSparseCM12VelocityExtensionInitialWords(
      velocityExtensionLayout,
    ), velocityExtensionLayout.headerBaseWords);
    if (transportPacketAuthorityLayout) {
      initialActivity[transportPacketAuthorityLayout.indirectBaseWords + 1] = 1;
      initialActivity[transportPacketAuthorityLayout.indirectBaseWords + 2] = 1;
    }
    if (phase1TransportQALayout) {
      const h = phase1TransportQALayout.baseWords;
      initialActivity[h + SPARSE_CM12_PHASE1_TRANSPORT_QA_HEADER.magic]
        = SPARSE_CM12_PHASE1_TRANSPORT_QA_MAGIC;
      initialActivity[h + SPARSE_CM12_PHASE1_TRANSPORT_QA_HEADER.version]
        = SPARSE_CM12_PHASE1_TRANSPORT_QA_VERSION;
      initialActivity[h + SPARSE_CM12_PHASE1_TRANSPORT_QA_HEADER.cellCapacity]
        = physicsCellCapacity;
    }
    initialActivity[8] = initiallyActiveBrickKeys.size;
    initialActivity[12] = 1;
    for (let brick = 0; brick < packed.brickCount; brick += 1) {
      const at = ACTIVITY_HEADER_WORDS + ACTIVITY_RECORD_WORDS * brick;
      initialActivity[at + 8] = atlas.bricks[brick]!.resolution;
      initialActivity[at + 9] = 32; // retained until the first GPU topology epoch
      initialActivity[at + 10] = initiallyActiveBrickKeys.has(atlas.bricks[brick]!.key)
        ? 1 : 0;
      // Candidate membership occupies the high bit of candidate-only schedule
      // word 35. Its low bit remains the existing preparation-scheduled flag,
      // preserving the baseline 39-word record stride byte-for-byte.
      initialActivity[at + 35] = initialActivity[at + 10] !== 0 ? 0x8000_0000 : 0;
      initialActivity[at + 12] = atlas.bricks[brick]!.resolution;
      initialActivity[at + 13] = atlas.bricks[brick]!.resolution;
      initialActivity[at + 37] = INVALID;
      // Low five bits retain the coarsest calm level accepted before this
      // brick's first promotion; bit 31 is latched by that promotion.
      initialActivity[at + 38] = atlas.bricks[brick]!.resolution;
      if (initialActivity[at + 10] !== 0) {
        initialActivity[at + 32] = sparseCM12AuthoredFluidFrontierMask(
          atlas, atlas.bricks[brick]!,
        );
      }
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
    const transportExecutionImage = createSparseCM12TransportExecutionImage(
      atlas, logicalOwnerDirectory, {
        brickActive: (brick) => initialActivity[
          ACTIVITY_HEADER_WORDS + ACTIVITY_RECORD_WORDS * brick + 10
        ] !== 0,
        acceptedBrickResolution: (brick) => atlas.bricks[brick]!.resolution,
        templateBrickCellRange: (brick, resolution) => {
          const level = Math.log2(resolution);
          const at = templates.words[11]! + 2
            * (sparseCM12TemplateLevels(atlas.brickFineResolution).length * brick + level);
          return [templates.words[at]!, templates.words[at + 1]!] as const;
        },
      }, { generation: 1, layout: transportExecutionImageLayout });
    const activity = uploadBuffer(device, "Sparse CM12 resident activity history",
      initialActivity, storage);
    const pressureEdgeOffset = templates.words[15]!;
    const pressureEdgeCount = templates.words[pressureEdgeOffset + templates.cellCount]!;
    const pressureTopology = compactPressureTopology(templates, atlas);
    const pressureWorklistBase = pressureAuxiliaryArena(
      templates, pressureTopology, packed.brickCount,
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
    const pressureExecutionImageLayout =
      createSparseCM12PressureExecutionImageLayout({
        baseWords: pressureWorklistBase.words.length,
        // Mutable PAB/PCM publish once into PEI's ordinary cell-address and
        // B8 membership planes; iterative kernels never revisit their atomics.
        cellCapacity: physicsCellCapacity,
        brickCapacity: worldLeafCapacity,
        hierarchyCapacity: Math.max(1,
          pressureHierarchyGroupCounts.reduce((sum, count) => sum + count, 0)),
        brickFineResolution: 8,
        presentationPageResolution: 8,
      });
    const pressureExecutionImageWords =
      createSparseCM12PressureExecutionImageInitialWords(
        pressureExecutionImageLayout,
      );
    const pressureWorklistWords = new Uint32Array(
      pressureExecutionImageLayout.totalWords,
    );
    pressureWorklistWords.set(pressureWorklistBase.words);
    pressureWorklistWords.set(pressureExecutionImageWords,
      pressureExecutionImageLayout.baseWords);
    const pressureWorklistData = Object.freeze({
      words: pressureWorklistWords,
      layout: pressureWorklistBase.layout,
    });
    if (templates.cellCount >= 0x1fff_ffff) {
      throw new Error("Sparse CM12 pressure brick range cache exhausts its 29-bit cell base");
    }
    if (templates.cellCount >= 0x00ff_ffff) {
      throw new Error("Sparse CM12 mass stencil cache exhausts its packed 24-bit cell IDs");
    }
    // Preserve the established transient pressure scratch ABI. Fine-edge
    // coefficients need frame-to-frame persistence, so their sole authority
    // follows every transient candidate/scratch use in the same ordinary buffer.
    const pressureScratchBytes = 4 * (pressureEdgeCount + pressureCoarseEdgeCount
      + 5 * worldLeafCapacity
      + pressureHierarchyGroupCounts.reduce((sum, count, level) =>
        sum + 4 * count + pressureHierarchyEdgeCounts[level]!, 0)
      + physicsCellCapacity);
    const candidateTransientWords = Math.max(
      pressureScratchBytes / 4,
      14 * physicsCellCapacity,
      candidateFloatsPerBrick(atlas.brickFineResolution)
        * Math.max(packed.candidateBrickCount, worldLeafCapacity),
    );
    const pressureFineEdgeImageBaseWords = Math.ceil(candidateTransientWords / 64) * 64;
    const candidateState = device.createBuffer({
      label: "Sparse CM12 isolated candidate cell fields",
      size: 4 * Math.max(1, pressureFineEdgeImageBaseWords + pressureEdgeCount),
      usage: storage,
    });
    const worklistHeaderWords = 32;
    const cellList0 = worklistHeaderWords;
    const cellList1 = cellList0 + physicsCellCapacity;
    const rowList0 = cellList1 + physicsCellCapacity;
    const rowList1 = rowList0 + physicsRowCapacity;
    const pageFreeList = rowList1 + physicsRowCapacity;
    const pageDescriptors = pageFreeList + topologyPagePool.freeListWords;
    // A compact leaf-ID list is the brick/rung view of the same double-
    // buffered accepted topology as the cell and shared-row lists. The two
    // slots flip with topology header word 2; this is not a second authority.
    const acceptedLeafManifestBase = pageDescriptors + topologyPagePool.descriptorWords;
    const acceptedLeafHeaderWords = 20;
    const acceptedLeafList0 = acceptedLeafHeaderWords;
    const acceptedLeafList1 = acceptedLeafList0 + worldLeafCapacity;
    const acceptedLeafDeltaList = acceptedLeafList1 + worldLeafCapacity;
    const acceptedRowMembershipStamps = acceptedLeafDeltaList + worldLeafCapacity;
    const acceptedLeafManifestWords = acceptedLeafHeaderWords + 3 * worldLeafCapacity
      + physicsRowCapacity;
    const initialLeafIds = atlas.bricks.flatMap((_brick, index) =>
      initialActivity[ACTIVITY_HEADER_WORDS + ACTIVITY_RECORD_WORDS * index + 10] !== 0
        ? [index] : []);
    const templateLevels = sparseCM12TemplateLevels(atlas.brickFineResolution);
    const initialCellIds: number[] = [];
    for (const brick of initialLeafIds) {
      const resolution = atlas.bricks[brick]!.resolution;
      const level = templateLevels.indexOf(resolution);
      if (level < 0) throw new Error(
        `Sparse CM12 initial leaf ${brick} has unsupported rung ${resolution}`,
      );
      const range = templates.words[11]! + 2 * (templateLevels.length * brick + level);
      const first = templates.words[range]!;
      const count = templates.words[range + 1]!;
      for (let local = 0; local < count; local += 1) initialCellIds.push(first + local);
    }
    const rowMetadataPlane = templates.words[7]! + templates.rowCount;
    const initialRowIds = Array.from(templates.initialRowWorklist).filter((row) => {
      const requirements = templates.words[rowMetadataPlane + row]!
        & TEMPLATE_ROW_METADATA_OFFSET_MASK;
      const count = templates.words[requirements]!;
      for (let at = 0; at < count; at += 1) {
        const metadata = templates.words[requirements + 1 + at]!;
        const brick = metadata >>> TEMPLATE_CELL_RESOLUTION_BITS;
        const resolution = metadata & TEMPLATE_CELL_RESOLUTION_MASK;
        const active = initialActivity[
          ACTIVITY_HEADER_WORDS + ACTIVITY_RECORD_WORDS * brick + 10
        ] !== 0;
        if (!active || atlas.bricks[brick]?.resolution !== resolution) return false;
      }
      return true;
    });
    const initialWorklists = new Uint32Array(
      acceptedLeafManifestBase + acceptedLeafManifestWords,
    );
    initialWorklists.set([1, 1, 0, 0, initialCellIds.length,
      initialRowIds.length, physicsCellCapacity, physicsRowCapacity,
      Math.ceil(initialCellIds.length / WORKGROUP_SIZE), 1, 1,
      Math.ceil(initialRowIds.length / WORKGROUP_SIZE), 1, 1,
      cellList0, cellList1, rowList0, rowList1], 0);
    initialWorklists.set([initialCellIds.length, initialRowIds.length,
      Math.ceil(initialCellIds.length / WORKGROUP_SIZE), 1, 1,
      Math.ceil(initialRowIds.length / WORKGROUP_SIZE), 1, 1], 18);
    // Device-side LIFO page allocator. Header indices 26..29 are deliberately
    // outside the accepted indirect-dispatch ABI at 8..25.
    initialWorklists[26] = topologyPagePool.pageCapacity;
    initialWorklists[27] = topologyPagePool.pageCapacity;
    initialWorklists[28] = pageFreeList;
    initialWorklists[29] = 0;
    initialWorklists[30] = pageDescriptors;
    initialWorklists[31] = topologyPagePool.pageWords;
    const initialLeafGroups = Math.ceil(initialLeafIds.length / WORKGROUP_SIZE);
    initialWorklists.set([initialLeafIds.length, initialLeafIds.length,
      acceptedLeafList0, acceptedLeafList1,
      initialLeafGroups, 1, 1, initialLeafGroups, 1, 1,
      0, acceptedLeafDeltaList, 0, 1, 1,
      initialLeafIds.length, 1, 1,
      initialRowIds.length, initialRowIds.length],
    acceptedLeafManifestBase);
    initialWorklists.set(initialLeafIds, acceptedLeafManifestBase + acceptedLeafList0);
    initialWorklists.set(initialLeafIds, acceptedLeafManifestBase + acceptedLeafList1);
    for (const row of initialRowIds) {
      initialWorklists[acceptedLeafManifestBase + acceptedRowMembershipStamps + row] = 3;
    }
    initialWorklists.set(initialCellIds, cellList0);
    initialWorklists.set(initialCellIds, cellList1);
    initialWorklists.set(initialRowIds, rowList0);
    initialWorklists.set(initialRowIds, rowList1);
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
      + acceptedLeafManifestBase + acceptedLeafManifestWords;
    if (topologyMutableEndWords !== physicalTemplateWordCount + initialWorklists.length
      || frameControlBaseWords < topologyMutableEndWords) {
      throw new Error("FCA1 overlaps the mutable topology worklist/page arena");
    }
    const frameControl = createSparseCM12FrameControl({
      baseWords: frameControlBaseWords,
      brickFineResolution: atlas.brickFineResolution,
      presentationPageResolution,
      cellWorkgroups,
      rowWorkgroups: Math.ceil(physicsRowCapacity / WORKGROUP_SIZE),
      bodyCapacity: rigid ? 12 : 0,
      d4Capable: true,
      rigidCapable: Boolean(rigid),
      initialScalarD4Authority: horizontalD4Authority,
      initialFaceD4Authority: horizontalD4Authority,
    });
    const pressureTopologyRepairLayout =
      createSparseCM12PressureTopologyRepairLayout({
        baseWords: frameControl.layout.totalWords,
        brickCapacity: worldLeafCapacity,
        brickFineResolution: atlas.brickFineResolution,
        presentationPageResolution,
      });
    const pressureTopologyRepairWords =
      createSparseCM12PressureTopologyRepairInitialWords(
        pressureTopologyRepairLayout,
      );
    const persistentPressureCacheLayout =
      createSparseCM12ResidentPersistentPressureCacheLayout({
        baseWords: pressureTopologyRepairLayout.totalWords,
        cellCount: physicsCellCapacity,
        rowCount: physicsRowCapacity,
        directedEdgeCount: pressureEdgeCount,
        brickCount: worldLeafCapacity,
        aggregateEdgeCount: pressureCoarseEdgeCount,
        hierarchyLevelCounts: pressureHierarchyGroupCounts,
        hierarchyEdgeLevelCounts: pressureHierarchyEdgeCounts,
      });
    const persistentPressureCacheWords = new Uint32Array(
      persistentPressureCacheLayout.bufferSizeWords,
    );
    initializeSparseCM12PersistentPressureCacheWords(
      persistentPressureCacheWords, persistentPressureCacheLayout,
    );
    // Reproduce the construction hierarchy bootstrap exactly for groups that
    // remain outside the first local repair. The full bake sums the
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
    const internedBoundaryBaseWords = transportExecutionImageLayout ? Math.ceil(
      persistentPressureCacheLayout.bufferSizeWords / 64,
    ) * 64 : undefined;
    // The accepted arm owns one composed, relocatable IBO1 image. Mini64 and
    // symmetric-expansion use the already-selected all-rung template packing;
    // ocean keeps the accepted-only production packing. Baseline constructs
    // none of this object graph and allocates no image words.
    // Independent logical-occupancy construction defines the stable-domain
    // closure even when accepted-only SCMT has no rows for an inactive leaf.
    const stableLeafFaceNeighbors = transportExecutionImageLayout
      ? compileSparseCM12StableLeafFaceNeighbors({
        coordinates: atlas.bricks.map((brick) => brick.coordinate),
        spans: atlas.bricks.map((brick) => sparseBrickSpan(brick)),
      }) : undefined;
    const internedBoundaryCatalog = transportExecutionImageLayout
      ? compileSparseCM12FactoredAEIPackedTemplateCatalog({
        words: templates.words,
        brickFineResolution: atlas.brickFineResolution as 8 | 16,
        brickKeyByLeafId: atlas.bricks.map((brick) => brick.key),
        validDimensions: (leaf, resolution) => {
          const brick = atlas.bricks[leaf]!;
          const width = atlas.brickFineResolution * sparseBrickSpan(brick);
          const scale = width / resolution;
          return ([0, 1, 2] as const).map((axis) => {
            const origin = atlas.brickFineResolution * brick.coordinate[axis]!;
            return Math.min(resolution, Math.ceil(Math.max(0, Math.min(width,
              atlas.dimensions[axis]! - origin)) / scale));
          }) as [number, number, number];
        },
        scaleLog2: (leaf, resolution) => Math.log2(
          atlas.brickFineResolution * sparseBrickSpan(atlas.bricks[leaf]!) / resolution),
        selectedResolution: (leaf) => atlas.bricks[leaf]!.resolution,
        neighborLeavesByLeaf: stableLeafFaceNeighbors,
      }) : undefined;
    const internedBoundaryCompilation = internedBoundaryCatalog
      ? compileSparseCM12InternedBoundaryOperators({
        catalog: internedBoundaryCatalog, packedWords: templates.words,
        activeLeaves: initialLeafIds,
      }) : undefined;
    if (internedBoundaryCompilation && !internedBoundaryCompilation.exactStableRowSet) {
      throw new Error("Sparse CM12 IBO1 production image failed its exact stable-row proof: "
        + `expected=${internedBoundaryCompilation.expectedStableRows.length}, `
        + `represented=${internedBoundaryCompilation.representedStableRows.length}, `
        + `firstMissing=${internedBoundaryCompilation.firstMissing}, `
        + `firstExtra=${internedBoundaryCompilation.firstExtra}`);
    }
    const internedRefLookup = internedBoundaryCompilation
      ? createSparseCM12InternedRefLookup({ ibo: internedBoundaryCompilation,
        baseWords: internedBoundaryCompilation.layout.immutableWords })
      : undefined;
    const iboGeometryNeighbors = internedBoundaryCompilation && internedRefLookup
      ? compileSparseCM12GeometryFaceNeighbors({
        coordinates: atlas.bricks.map((brick) => brick.coordinate),
        spans: atlas.bricks.map((brick) => sparseBrickSpan(brick)),
      }) : undefined;
    const iboGeometryBaseWords = internedRefLookup
      ? Math.ceil(internedRefLookup.layout.totalWords / 64) * 64
      : undefined;
    if (iboGeometryNeighbors && internedBoundaryCatalog) {
      for (let leaf = 0; leaf < packed.brickCount; leaf += 1) {
        const actual = [...iboGeometryNeighbors.neighbors.subarray(
          iboGeometryNeighbors.offsets[leaf]!, iboGeometryNeighbors.offsets[leaf + 1]!,
        )];
        const expected = [...internedBoundaryCatalog.neighborLeavesByLeaf[leaf]!];
        if (actual.length !== expected.length
          || actual.some((neighbor, index) => neighbor !== expected[index])) {
          throw new Error("Sparse CM12 IGN1 geometry closure differs from the canonical "
            + `catalog at leaf ${leaf}: actual=[${actual}], expected=[${expected}]`);
        }
      }
      const expectedMaximumFanout = internedBoundaryCatalog.neighborLeavesByLeaf.reduce(
        (maximum, neighbors) => Math.max(maximum, neighbors.length), 0);
      if (iboGeometryNeighbors.maximumFanout !== expectedMaximumFanout) {
        throw new Error("Sparse CM12 IGN1 maximum fanout receipt differs from catalog: "
          + `actual=${iboGeometryNeighbors.maximumFanout}, `
          + `expected=${expectedMaximumFanout}`);
      }
    }
    const iboTRABaseWords = iboGeometryBaseWords !== undefined && iboGeometryNeighbors
      ? Math.ceil((iboGeometryBaseWords + iboGeometryNeighbors.words.length) / 64) * 64
      : undefined;
    const iboTRASupplement = internedBoundaryCompilation
      && iboTRABaseWords !== undefined
      ? createSparseCM12IboTRASupplement({
        ibo: internedBoundaryCompilation, baseWords: iboTRABaseWords,
      }) : undefined;
    const internedBoundaryImage = internedBoundaryCompilation && internedRefLookup
      && iboGeometryNeighbors && iboGeometryBaseWords !== undefined
      && iboTRASupplement
      ? createSparseCM12InternedBoundaryImage(internedBoundaryCompilation,
        initialLeafIds, 1, internedBoundaryCatalog!.descriptorIdByLeaf, [
          { label: "IRL1", baseWords: internedRefLookup.layout.baseWords,
            words: internedRefLookup.words },
          { label: "IGN1", baseWords: iboGeometryBaseWords,
            words: iboGeometryNeighbors.words },
          { label: "ITR1", baseWords: iboTRASupplement.layout.baseWords,
            words: iboTRASupplement.words },
        ]) : undefined;
    if (internedBoundaryImage) {
      const initialSemantics = compareSparseCM12IBOSemanticAuthority({
        image: internedBoundaryImage, packedWords: templates.words,
        activeLeaves: initialLeafIds,
        descriptorIdByLeaf: internedBoundaryCatalog!.descriptorIdByLeaf,
        leaves: Array.from({ length: packed.brickCount }, (_, leaf) => leaf), slot: 0,
      });
      if (!initialSemantics.exact || initialSemantics.duplicateCandidateRows !== 0) {
        throw new Error("Sparse CM12 ISA1 initial slot semantic receipt failed: "
          + `exact=${initialSemantics.exact}, `
          + `firstMismatch=${initialSemantics.firstMismatchLeaf}, `
          + `duplicateRows=${initialSemantics.duplicateCandidateRows}`);
      }
    }
    const iboMemoryPlan = sparseCM12InternedBoundaryMemoryPlan(
      atlas.brickFineResolution,
    );
    if (internedBoundaryImage
      && (internedBoundaryImage.layout.immutableBytes
          > iboMemoryPlan.immutableMaximumBytes
        || internedBoundaryImage.layout.bytesPerSlot
          > iboMemoryPlan.slotMaximumBytes)) {
      throw new Error("Sparse CM12 IBO1 production image exceeds its memory plan: "
        + `immutable=${internedBoundaryImage.layout.immutableBytes}`
        + `/${iboMemoryPlan.immutableMaximumBytes} B, `
        + `slot=${internedBoundaryImage.layout.bytesPerSlot}`
        + `/${iboMemoryPlan.slotMaximumBytes} B, `
        + `leaves=${internedBoundaryImage.layout.leafCapacity}, `
        + `canonicals=${internedBoundaryImage.layout.canonicalCapacity}, `
        + `templates=${internedBoundaryImage.layout.templateCount}`);
    }
    const iboSemanticAuthorityBaseWords = internedBoundaryImage
      && internedBoundaryBaseWords !== undefined
      ? Math.ceil((internedBoundaryBaseWords + internedBoundaryImage.layout.totalWords)
        / 64) * 64 : undefined;
    // ISA1 retains only the generation stamps and exact closure list. Semantic
    // receipts compare in-register/workgroup and do not need per-leaf journals.
    const iboSemanticAuthorityWords = internedBoundaryImage
      ? 16 + 2 * internedBoundaryImage.layout.leafCapacity : 0;
    if (4 * iboSemanticAuthorityWords
      > iboMemoryPlan.semanticAuthorityMaximumBytes) {
      throw new Error("Sparse CM12 ISA1 dynamic authority exceeds its memory plan: "
        + `dynamic=${4 * iboSemanticAuthorityWords}`
        + `/${iboMemoryPlan.semanticAuthorityMaximumBytes} B, `
        + `leaves=${internedBoundaryImage?.layout.leafCapacity ?? 0}, `
        + `fanout=${iboGeometryNeighbors?.maximumFanout ?? 0}`);
    }
    const transportAuthorityTotalWords = iboSemanticAuthorityBaseWords !== undefined
      ? iboSemanticAuthorityBaseWords + iboSemanticAuthorityWords
      : internedBoundaryBaseWords ?? persistentPressureCacheLayout.bufferSizeWords;
    const topologyEffectsAuthorityLayout = transportExecutionImageLayout
      ? createSparseCM12TopologyEffectsAuthorityLayout({
        baseWords: transportAuthorityTotalWords,
        ptrCapacity: worldLeafCapacity,
        ptrLeafCapacity: Math.ceil(worldLeafCapacity / 256),
      }) : undefined;
    if (topologyEffectsAuthorityLayout
      && topologyEffectsAuthorityLayout.totalBytes > 524_288) {
      throw new Error("Sparse CM12 TFX1 candidate-effects authority exceeds its memory plan: "
        + `dynamic=${topologyEffectsAuthorityLayout.totalBytes}/524288 B, `
        + `bricks=${topologyEffectsAuthorityLayout.ptrCapacity}, `
        + `leaves=${topologyEffectsAuthorityLayout.ptrLeafCapacity}`);
    }
    const topologyAuthorityTotalWords = topologyEffectsAuthorityLayout?.totalWords
      ?? transportAuthorityTotalWords;
    const finalScalarPacketMaskLayout = transportExecutionImageLayout
      ? createSparseCM12FinalScalarPacketMaskLayout({
        baseWords: topologyAuthorityTotalWords,
        packetCapacity: transportExecutionImageLayout.packetCapacity,
        brickFineResolution: transportExecutionImageLayout.brickFineResolution,
      }) : undefined;
    const faceAddressProgram = internedBoundaryCompilation
      ? compileSparseCM12BrickTileFaceAddressProgram({
        ibo: internedBoundaryCompilation,
        baseWords: finalScalarPacketMaskLayout?.totalWords
          ?? topologyAuthorityTotalWords,
      }) : undefined;
    if (!faceAddressProgram) {
      throw new Error("Sparse CM12 production requires the all-rung BFA1 face program");
    }
    const worldDirectoryLayout = createSparseCM12WorldDirectoryLayout({
      initialLeaves: packed.brickCount,
      growthLeaves: topologyPagePool.pageCapacity,
      maximumSpanLog: Math.log2(atlas.maximumSpanBricks),
      baseWords: faceAddressProgram.layout.totalWords,
    });
    const solidOccupancyLayout = dynamicWorldGrowth && initialSolidWorld
      ? createSparseCM12SolidOccupancyLayout({
        baseWords: worldDirectoryLayout.totalWords,
        authoredPageCount: initialSolidWorld.pages.length,
        authoredRegionCount: initialSolidWorld.regions?.length ?? 0,
      })
      : undefined;
    // Live host incidence records are the only mutable words in the physical
    // template prefix. Keep an immutable copy beside the dynamic arenas so a
    // recycled SparseWorld page can restore and repatch the exact host slot.
    const immutableHostIncidenceBaseWords = Math.ceil((solidOccupancyLayout?.totalWords
      ?? worldDirectoryLayout.totalWords) / 64) * 64;
    const hostIncidenceCount = templates.words[5]!;
    const immutableHostIncidenceWords = templates.words.subarray(
      templates.words[10]!, templates.words[10]! + 2 * hostIncidenceCount);
    const topologyArenaWords = immutableHostIncidenceBaseWords
      + immutableHostIncidenceWords.length;
    const topologyArena = device.createBuffer({
      label: "Sparse CM12 physical topology templates and worklists",
      size: Math.max(4, 4 * topologyArenaWords),
      usage: storage | (topologyEffectsAuthorityLayout ? GPUBufferUsage.INDIRECT : 0),
    });
    device.queue.writeBuffer(topologyArena, 0, templates.words.buffer as ArrayBuffer,
      templates.words.byteOffset, physicalTemplateBytes);
    device.queue.writeBuffer(topologyArena, 4 * immutableHostIncidenceBaseWords,
      immutableHostIncidenceWords.buffer as ArrayBuffer,
      immutableHostIncidenceWords.byteOffset, immutableHostIncidenceWords.byteLength);
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
      persistentPressureCacheLayout.baseWords,
    );
    device.queue.writeBuffer(topologyArena,
      4 * persistentPressureCacheLayout.baseWords,
      persistentPressureCacheRegion.buffer as ArrayBuffer,
      persistentPressureCacheRegion.byteOffset,
      persistentPressureCacheRegion.byteLength);
    if (internedBoundaryImage && internedBoundaryBaseWords !== undefined) {
      device.queue.writeBuffer(topologyArena, 4 * internedBoundaryBaseWords,
        internedBoundaryImage.words.buffer as ArrayBuffer,
        internedBoundaryImage.words.byteOffset, internedBoundaryImage.words.byteLength);
    }
    if (iboSemanticAuthorityBaseWords !== undefined && internedBoundaryImage) {
      const isaHeader = new Uint32Array(16);
      isaHeader.set([0x4953_4131, 1, internedBoundaryImage.layout.leafCapacity], 0);
      device.queue.writeBuffer(topologyArena, 4 * iboSemanticAuthorityBaseWords,
        isaHeader.buffer as ArrayBuffer, isaHeader.byteOffset, isaHeader.byteLength);
    }
    if (topologyEffectsAuthorityLayout) {
      const tfxWords = createSparseCM12TopologyEffectsAuthorityInitialWords(
        topologyEffectsAuthorityLayout);
      device.queue.writeBuffer(topologyArena, 4 * topologyEffectsAuthorityLayout.baseWords,
        tfxWords.buffer as ArrayBuffer, tfxWords.byteOffset, tfxWords.byteLength);
    }
    if (finalScalarPacketMaskLayout) {
      const fsmWords = createSparseCM12FinalScalarPacketMaskInitialWords(
        finalScalarPacketMaskLayout);
      device.queue.writeBuffer(topologyArena, 4 * finalScalarPacketMaskLayout.baseWords,
        fsmWords.buffer as ArrayBuffer, fsmWords.byteOffset, fsmWords.byteLength);
    }
    device.queue.writeBuffer(topologyArena, 4 * faceAddressProgram.layout.baseWords,
      faceAddressProgram.words.buffer as ArrayBuffer,
      faceAddressProgram.words.byteOffset, faceAddressProgram.words.byteLength);
    const worldDirectoryWords = createSparseCM12WorldDirectoryInitialWords(
      worldDirectoryLayout, atlas);
    device.queue.writeBuffer(topologyArena, 4 * worldDirectoryLayout.baseWords,
      worldDirectoryWords.buffer as ArrayBuffer, worldDirectoryWords.byteOffset,
      worldDirectoryWords.byteLength);
    if (solidOccupancyLayout && initialSolidWorld) {
      writeSparseCM12SolidOccupancy(device.queue, topologyArena,
        solidOccupancyLayout, initialSolidWorld, [0, 0, 0]);
    }
    const pressureTemplates = uploadBuffer(device,
      "Sparse CM12 read-only pressure topology", pressureTopology,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const pressureWorklists = uploadBuffer(device,
      "Sparse CM12 pressure aggregate and execution arena",
      pressureWorklistData.words,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        | GPUBufferUsage.INDIRECT);
    const transportExecutionImageBuffer = transportExecutionImage
      ? uploadBuffer(device, "Sparse CM12 Phase-1 transport execution image",
        transportExecutionImage.words, storage)
      : undefined;
    // Candidate TFX1 appends two exact delta-sized publication triplets at
    // words 24 and 27. Baseline construction keeps them zero and never
    // dispatches them.
    const acceptedAndShadowIndirect = new Uint32Array(30);
    acceptedAndShadowIndirect.set(initialWorklists.subarray(8, 14), 0);
    acceptedAndShadowIndirect.set(initialWorklists.subarray(20, 26), 6);
    acceptedAndShadowIndirect.set(initialWorklists.subarray(
      acceptedLeafManifestBase + 4, acceptedLeafManifestBase + 7), 12);
    acceptedAndShadowIndirect.set(initialWorklists.subarray(
      acceptedLeafManifestBase + 12, acceptedLeafManifestBase + 15), 18);
    acceptedAndShadowIndirect.set([0, 1, 1, 0, 1, 1], 24);
    acceptedAndShadowIndirect.set(initialWorklists.subarray(
      acceptedLeafManifestBase + 15, acceptedLeafManifestBase + 18), 21);
    const acceptedIndirectArguments = uploadBuffer(device,
      "Sparse CM12 accepted indirect dispatch snapshot",
      acceptedAndShadowIndirect,
      GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const pressureCellIndirectArguments = device.createBuffer({
      label: "Sparse CM12 pressure-cell indirect dispatch",
      size: 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const pressureMembershipIndirectArguments = device.createBuffer({
      label: "Sparse CM12 pressure-membership bootstrap indirect dispatches",
      size: SPARSE_CM12_PRESSURE_MEMBERSHIP_INDIRECT_BYTES,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const pressureExecutionIndirectArguments = device.createBuffer({
      label: "Sparse CM12 PEI1 pressure execution dispatches",
      size: 48,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const persistentPressureCacheIndirectArguments = device.createBuffer({
      label: "Sparse CM12 persistent pressure-cache indirect dispatches",
      // Seed/repair/work for four aggregate families.
      size: 12 * 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
        | GPUBufferUsage.COPY_SRC,
    });
    const transportPacketIndirectArguments = transportPacketAuthorityLayout
      ? device.createBuffer({
        label: "Sparse CM12 rung-major transport packet indirect dispatch",
        size: 12,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
          | GPUBufferUsage.COPY_SRC,
      })
      : undefined;
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
    const presentationAllocatorBindGroupLayout = device.createBindGroupLayout({
      label: "Sparse CM12 presentation page allocator layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" } },
      ],
    });
    const presentationAllocatorPipelineLayout = device.createPipelineLayout({
      label: "Sparse CM12 presentation page allocator pipeline layout",
      bindGroupLayouts: [presentationAllocatorBindGroupLayout],
    });
    const presentationAllocatorBindGroup = device.createBindGroup({
      label: "Sparse CM12 presentation page allocator bindings",
      layout: presentationAllocatorBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: topology } },
        { binding: 1, resource: { buffer: activity } },
        { binding: 2, resource: { buffer: fineMetadata } },
        { binding: 3, resource: { buffer: fineWorklist } },
        { binding: 4, resource: { buffer: topologyArena } },
      ],
    });
    const presentationAllocatorShaderSource = sparseCM12PresentationPageAllocatorWGSL(
      worldLeafCapacity, packed.brickCount, packed.brickOffset,
      framePlanPresentationLayout, worldDirectoryLayout,
      fine.plan.brickDimensions,
    );
    const diagnosticsReadback = device.createBuffer({
      label: "Sparse CM12 resident diagnostic readback",
      // Reduction scalars, activity header, then authoritative accepted
      // cell/row worklist counts and compact pressure-cell count. These are QA
      // receipts, never schedule input.
      size: SPARSE_CM12_PRESSURE_SCALAR_BYTES + 4 * ACTIVITY_HEADER_WORDS + 12
        + SPARSE_CM12_PCM_DIAGNOSTIC_BYTES
        + 4 * SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS
        + 4 * SPARSE_CM12_PRESSURE_CUTOVER_DIAGNOSTIC_WORDS,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const deviceCompilation = sparseCM12DeviceCompilationCache(device);
    const bindGroupLayout = deviceCompilation.bindGroupLayout;
    const velocityExtensionDepths = device.createBuffer({
      label: "Sparse CM12 velocity-extension depth records",
      size: 256 * SPARSE_CM12_VELOCITY_EXTENSION_DEPTH,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    });
    {
      const depths = new Uint32Array(velocityExtensionDepths.getMappedRange());
      for (let depth = 1; depth <= SPARSE_CM12_VELOCITY_EXTENSION_DEPTH; depth += 1) {
        depths[64 * (depth - 1)] = depth;
      }
      velocityExtensionDepths.unmap();
    }
    report("Bind resident GPU arenas");
    const bindGroup = device.createBindGroup({ label: "Sparse CM12 resident bindings",
      layout: bindGroupLayout, entries: [
        { binding: 0, resource: { buffer: parameters } },
        { binding: 1, resource: { buffer: topology } },
        { binding: 2, resource: { buffer: state } },
        { binding: 3, resource: { buffer: partials } },
        { binding: 4, resource: { buffer: scalars } },
        { binding: 5, resource: { buffer: velocityExtensionDepths, size: 4 } },
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
        { binding: 5, resource: { buffer: velocityExtensionDepths, size: 4 } },
        { binding: 11, resource: { buffer: conditioning } },
        { binding: 12, resource: { buffer: activity } },
        { binding: 13, resource: { buffer: candidateState } },
        { binding: 14, resource: { buffer: pressureTemplates } },
        { binding: 15, resource: { buffer: pressureWorklists } },
        { binding: 16, resource: { buffer: topologyArena } },
      ],
    });
    const transportDepthBindGroups = Array.from(
      { length: SPARSE_CM12_VELOCITY_EXTENSION_DEPTH }, (_, index) =>
        device.createBindGroup({
          label: `Sparse CM12 Phase-1 transport execution bindings depth ${index + 1}`,
          layout: bindGroupLayout,
          entries: [
          { binding: 0, resource: { buffer: parameters } },
          { binding: 1, resource: { buffer: topology } },
          { binding: 2, resource: { buffer: state } },
          { binding: 3, resource: { buffer: transportExecutionImageBuffer
            ? effectiveTransportVelocity! : partials } },
          { binding: 4, resource: { buffer: scalars } },
          { binding: 5, resource: { buffer: velocityExtensionDepths,
            offset: 256 * index, size: 4 } },
          { binding: 11, resource: { buffer: conditioning } },
          { binding: 12, resource: { buffer: activity } },
          { binding: 13, resource: { buffer: candidateState } },
          { binding: 14, resource: { buffer: transportExecutionImageBuffer
            ?? pressureTemplates } },
          { binding: 15, resource: { buffer: pressureWorklists } },
          { binding: 16, resource: { buffer: topologyArena } },
          ],
        }),
    );
    const transportBindGroup = transportDepthBindGroups[0]!;
    const effectiveVelocityPressureBindGroup = effectiveTransportVelocity
      ? device.createBindGroup({
        label: "Sparse CM12 VEX/collocation effective-velocity bindings",
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: parameters } },
          { binding: 1, resource: { buffer: topology } },
          { binding: 2, resource: { buffer: state } },
          { binding: 3, resource: { buffer: effectiveTransportVelocity } },
          { binding: 4, resource: { buffer: scalars } },
          { binding: 5, resource: { buffer: velocityExtensionDepths, size: 4 } },
          { binding: 11, resource: { buffer: conditioning } },
          { binding: 12, resource: { buffer: activity } },
          { binding: 13, resource: { buffer: candidateState } },
          { binding: 14, resource: { buffer: pressureTemplates } },
          { binding: 15, resource: { buffer: pressureWorklists } },
          { binding: 16, resource: { buffer: topologyArena } },
        ],
      }) : pressureBindGroup;
    report("Generate presentation shader");
    const compiler = gpuCompilationManagerFor(device);
    const createResidentShaderSource = (velocityExtensionFixedRecurrenceDepth?: number) =>
      createWebgpuSparseCM12ResidentWGSL(
        atlas.brickFineResolution, presentationPageResolution,
        pressureWorklistData.layout,
        incrementalActivityLayout, canonicalMembershipLayout,
        framePlanLayout, framePlanPresentationLayout,
        frameControl.layout, pressureTopologyRepairLayout,
        persistentPressureCacheLayout,
        velocityExtensionLayouts,
        pressureExecutionImageLayout,
        { layout: logicalOwnerDirectory.layout, baseWords: logicalOwnerBaseWords,
          packedOwner16BaseWords: logicalOwnerPacked16BaseWords },
        layout.faceVelocitySupport,
        effectiveTransportVelocityLayout,
        transportExecutionImage?.layout,
        transportPacketAuthorityLayout,
        phase1TransportQALayout,
        phase1TransportProfileBaseWords,
        internedBoundaryImage && internedRefLookup && iboGeometryNeighbors
          && iboTRASupplement
          && internedBoundaryBaseWords !== undefined
          && iboGeometryBaseWords !== undefined
          && iboSemanticAuthorityBaseWords !== undefined
          ? { layout: internedBoundaryImage.layout,
            refLookupLayout: internedRefLookup.layout,
            traSupplementLayout: iboTRASupplement.layout,
            baseWords: internedBoundaryBaseWords,
            semanticAuthority: {
              geometryBaseWords: internedBoundaryBaseWords + iboGeometryBaseWords,
              geometryOffsetBaseWords: iboGeometryNeighbors.offsetBaseWords,
              geometryNeighborBaseWords: iboGeometryNeighbors.neighborBaseWords,
              authorityBaseWords: iboSemanticAuthorityBaseWords,
              leafCapacity: internedBoundaryImage.layout.leafCapacity,
              immutableContentHash:
                internedBoundaryImage.immutableCertificate.contentHash,
              immutableCertificateHash:
                internedBoundaryImage.immutableCertificate.certificateHash,
            } }
          : undefined,
        topologyEffectsAuthorityLayout,
        finalScalarPacketMaskLayout,
        faceAddressProgram.layout,
        pressureFineEdgeImageBaseWords,
        velocityExtensionFixedRecurrenceDepth,
        worldDirectoryLayout,
        dynamicWorldGrowth,
        solidOccupancyLayout,
        immutableHostIncidenceBaseWords,
        signedWorldGrowth,
      );
    const shaderSource = createResidentShaderSource();
    const sourceByShaderModule = new WeakMap<GPUShaderModule, string>();
    const shaderModuleFor = (source: string, label: string) => {
      let promise = deviceCompilation.shaderModules.get(source);
      if (!promise) {
        // Pipeline creation reports the errors for the entry points it consumes.
        // A module-wide getCompilationInfo() would eagerly compile the full family.
        promise = Promise.resolve(compiler.createShaderModule({ label, code: source }))
          .then((module) => {
            sourceByShaderModule.set(module, source);
            return module;
          });
        deviceCompilation.shaderModules.set(source, promise);
      }
      return promise;
    };
    const presentationShaderRoots = presentationPublisherOracleForQA
      ? ["refreshSparseCM12SolidWorldCells", "refreshSparseCM12SolidWorldRows",
        "classifyPresentationBricks", "validateSparseCM12InternedBoundaryImmutable",
        "publishSparseLevelSet"]
      : ["refreshSparseCM12SolidWorldCells", "refreshSparseCM12SolidWorldRows",
        "classifyPresentationBricks", "validateSparseCM12InternedBoundaryImmutable",
        "beginSparseCM12FramePlanNext", "initializeSparseCM12FramePlanNext",
        "populateSparseCM12PresentationFramePlan",
        "resolveSparseCM12FramePlanNextClosure", "sealSparseCM12FramePlanNextBricks",
        "finalizeSparseCM12FramePlanNext", "markSparseCM12GlobalFramePlanReceipts",
        "beginSparseCM12FramePlanPresentationPacket",
        "buildSparseCM12FramePlanPresentationPacket",
        "finalizeSparseCM12FramePlanPresentationPacket",
        "executeSparseCM12FramePlanPresentationPacket",
        "commitSparseCM12FramePlanPresentationPacket",
        "verifySparseCM12FramePlanCurrentStage",
        "finalizeSparseCM12FramePlanPresentationExecution",
        "rejectSparseCM12FramePlanPresentationFaults"] as const;
    const presentationShaderSource = sparseCM12WGSLForEntryPoints(
      shaderSource, presentationShaderRoots,
    );
    report("Compile first-frame presentation pipelines");
    const shaderModule = await shaderModuleFor(
      presentationShaderSource, "Sparse CM12 presentation shader",
    );
    const pipelineLayout = deviceCompilation.pipelineLayout;
    const names = ["injectLiquid", "injectLiquidFaces",
      "clearLiquidJetOverflowReceipts", "scatterLiquidJetOverflow",
      "finalizeLiquidJetOverflow",
      "beginSparseCM12FrameControl", "publishSparseCM12FrameBodyAuthority",
      "sealSparseCM12FrameControl",
      "publishSparseCM12FrameScalarOutput", "publishSparseCM12FrameFaceOutput",
      "commitSparseCM12FrameControl", "sparseCM12FrameControlNoop",
      "publishSparseCM12MovingSolidActivity",
      "compileSparseCM12TransportPacketsFromFinalScalarMasks",
      "scatterGammaSnapshotRows", "scatterGammaRefinementRows",
      "beginSparseCM12FinalScalarMasks",
      "publishSparseCM12FinalScalarMasks",
      "sealSparseCM12FinalScalarMasks",
      "compileSparseCM12TransportExecutionImageShadow",
      "replaySparseCM12TransportExecutionImageRetired",
      ...(internedBoundaryImage ? [
        "validateSparseCM12InternedBoundaryImmutable",
        "beginSparseCM12InternedBoundaryDelta",
        "compileSparseCM12InternedBoundaryDelta",
        "finalizeSparseCM12ISAChangedSetReceipt",
        "validateSparseCM12InternedBoundaryDeltaPackets",
        "finalizeSparseCM12InternedBoundaryDelta",
        "replaySparseCM12InternedBoundaryDelta",
      ] as const : []),
      "clearSparseCM12TransportReceipts",
      "traceGammaAndBeta", "scatterDensityDeficit",
      "gatherConservativeDensity",
      "seedTracers", "advanceTracers",
      "clearGammaReceipts", "finalizeGammaSnapshot", "finalizeGammaRefinement",
      "prepareSharpeningField", "scatterSharpeningMass", "finalizeSharpening",
      "initializeDensityCapacityRepair", "scatterDensityCapacityRepair",
      "finalizeDensityCapacityRepair", "preserveHorizontalD4",
      "commitHorizontalD4", "preserveVelocityHorizontalD4",
      "commitVelocityHorizontalD4", "preserveActivityHorizontalD4",
      "commitActivityHorizontalD4",
      "initializeVelocityExtensionPackets", "advanceVelocityExtensionPackets",
      "prepareSparseCM12DynamicFaceRows", "projectSparseCM12DynamicFaceRows",
      "forceFaces", "enforceSparseCM12InflowFaces",
      "classifyPressureCells", "compileCanonicalPressureRows",
      "beginCanonicalPressureCells", "beginCanonicalPressureRows",
      "planPressureMembershipEpoch",
      "finalizeCanonicalPressureCellFrontier",
      "repairCanonicalPressureCellLeaves", "finalizeCanonicalPressureCells",
      "finalizeCanonicalPressureRows",
      "classifyDirtyPressureCells",
      "preparePressure",
      "beginPressureSolve",
      "publishPressureSolveDispatchGate", "restorePressureSolveDispatches",
      "applyJacobiPreconditioner",
      "initializeJacobiDirection",
      "initializePCG", "measureTrueResidual", "measureGuardedTrueResidual",
      "reduceInitialTrueResidual", "reduceGuardedTrueResidual",
      "restartPCGAfterCurvatureLoss", "initializeJacobiRecoveryDirection",
      "reduceCurvatureRecovery",
      "reduceFinalTrueResidual",
      "initializePipelinedImage", "reducePipelinedInitialize",
      "updatePipelinedState", "applyPipelinedImage", "reducePipelinedIteration",
      "applyPipelinedRecovery", "reducePipelinedRecovery",
      "reduceInitialize",
      "collocateAndDiagnose",
      "reduceDivergenceDiagnostics",
      "advanceActivityClock", "beginIncrementalActivity",
      "markIncrementalActivityScalarBricks", "markIncrementalActivityTopology",
      "markIncrementalActivityPostTopology",
      "finalizeIncrementalActivityMasks", "measureBrickActivity",
      "ageIncrementalActivityHistory", "finalizeIncrementalActivityCensus",
      "planBrickResolution", "activateSweptFrontierPages", "activateInjectionFrontierPages",
      "closePlannedResolution",
      "validateCandidateResolution", "scheduleTopologyPreparation",
      "allocateCandidateTopologyPages", "synthesizeCandidateCellPages",
      "allocateSparseWorldFrontier", "allocateSparseWorldInteractionPages",
      "synthesizeSparseWorldFrontierPages",
      "connectSparseWorldFrontierPages",
      "publishSparseWorldFrontierAcceptance",
      "compileSparseWorldFrontierExecutionImage",
      "clearShadowRowMembership", "beginShadowTopology", "buildShadowLeafWorklist",
      "buildShadowStructureWorklist",
      "buildShadowCellWorklist", "buildShadowRowWorklist",
      "finalizeShadowWorklists", "transferCandidateCells",
      "transferCandidateCellsFromTopologyDelta",
      "prepareCandidateFaceReceipts", "transferCandidateFaces",
      "transferCandidateFacesFromTopologyDelta",
      "publishCandidateTopologyDelta", "publishCandidateTopologyDeltaFromWorklist",
      "validateCandidateShadowFaces", "publishCandidateShadowFaces",
      "validateAndAuthorizeShadowTopology", "finalizeAuthorizedShadowTopology",
      "sealSparseCM12AuthorizedTopologyEffects",
      ...(topologyEffectsAuthorityLayout ? [
        "beginSparseCM12TopologyEffectsPreflight",
        "recordCandidateTopologyEffectsFromTopologyDelta",
        "finalizeSparseCM12TopologyEffectsPreflight",
        "publishSparseCM12TopologyPTREffects",
        "finishSparseCM12TopologyEffectsPublication",
      ] as const : []),
      "retireUnsupportedEmptyBricks",
      "refreshSparseCM12SolidWorldCells", "refreshSparseCM12SolidWorldRows",
      "refreshSparseCM12FrontierSolidWorld",
      "classifyPresentationBricks",
      ...(presentationPublisherOracleForQA ? ["publishSparseLevelSet"] as const : []),
      "beginSparseCM12FramePlanNext", "initializeSparseCM12FramePlanNext",
      "populateSparseCM12PresentationFramePlan",
      "resolveSparseCM12FramePlanNextClosure", "sealSparseCM12FramePlanNextBricks",
      "finalizeSparseCM12FramePlanNext", "markSparseCM12GlobalFramePlanReceipts",
      "beginSparseCM12FramePlanPresentationPacket",
      "buildSparseCM12FramePlanPresentationPacket",
      "finalizeSparseCM12FramePlanPresentationPacket",
      "executeSparseCM12FramePlanPresentationPacket",
      "commitSparseCM12FramePlanPresentationPacket",
      "finalizeSparseCM12FramePlanPresentationExecution",
      "rejectSparseCM12FramePlanPresentationFaults",
      "beginPersistentPressureCache", "sealPersistentPressureAggregateFrontier",
      ...SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_ENTRY_POINTS,
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
      "publishFrozenPressureCoarseCache",
      "publishFrozenPressureCellIds",
      "publishFrozenPressureMembership",
      "publishFrozenPressureCoefficients",
      "clearSparseCM12RetiredFaceVelocitySupport",
      "publishSparseCM12FaceVelocitySupport",
      "prepareSparseCM12InteriorFaceTiles",
      "prepareSparseCM12SeamFacePackets",
      "prepareSparseCM12SparseAirFacePackets",
      "projectSparseCM12InteriorFaceTiles",
      "projectSparseCM12SeamFacePackets",
      "projectSparseCM12SparseAirFacePackets",
      "measureDivergenceDiagnostics",
      ...sparseCM12PressureTopologyRepairEntryPoints(
        pressureTopologyRepairLayout,
      ),
      ...(phase1TransportQALayout
        ? ["captureSparseCM12Phase1TransportPackets"] as const : []),
      ...(layout.journal !== 0
        ? ["journalIteration", "journalSnapshot"] as const : []),
    ] as const;
    const presentationEntryNames = new Set<string>(presentationPublisherOracleForQA
      ? ["refreshSparseCM12SolidWorldCells", "refreshSparseCM12SolidWorldRows",
        "classifyPresentationBricks", "validateSparseCM12InternedBoundaryImmutable",
        "publishSparseLevelSet"]
      : ["refreshSparseCM12SolidWorldCells", "refreshSparseCM12SolidWorldRows",
        "classifyPresentationBricks", "validateSparseCM12InternedBoundaryImmutable",
        "beginSparseCM12FramePlanNext", "initializeSparseCM12FramePlanNext",
        "populateSparseCM12PresentationFramePlan",
        "resolveSparseCM12FramePlanNextClosure", "sealSparseCM12FramePlanNextBricks",
        "finalizeSparseCM12FramePlanNext", "markSparseCM12GlobalFramePlanReceipts",
        "beginSparseCM12FramePlanPresentationPacket",
        "buildSparseCM12FramePlanPresentationPacket",
        "finalizeSparseCM12FramePlanPresentationPacket",
        "executeSparseCM12FramePlanPresentationPacket",
        "commitSparseCM12FramePlanPresentationPacket",
        "finalizeSparseCM12FramePlanPresentationExecution",
        "rejectSparseCM12FramePlanPresentationFaults"]);
    const presentationNames = names.filter((name) => presentationEntryNames.has(name));
    const simulationNames = names.filter((name) => !presentationEntryNames.has(name));
    const compileNamedEntries = async (
      selectedNames: readonly string[],
      priority: "critical" | "background",
      module: GPUShaderModule = shaderModule,
    ) => Promise.all(selectedNames.map(async (name) => {
      try {
        const descriptor: GPUComputePipelineDescriptor = {
          label: `Sparse CM12 ${name}`, layout: pipelineLayout,
          compute: { module, entryPoint: name },
        };
        const pipeline = await compiler.compileComputePipeline(descriptor, { priority });
        return [name, pipeline] as const;
      } catch (error) {
        const compilation = await module.getCompilationInfo();
        const shaderErrors = compilation.messages
          .filter((message) => message.type === "error")
          .map((message) => {
            const line = sourceByShaderModule.get(module)?.split("\n")[message.lineNum - 1];
            return `${message.lineNum}:${message.linePos} ${message.message}${
              line ? `\n  ${line}` : ""}`;
          })
          .join("\n");
        const detail = error instanceof Error
          ? `${error.name}: ${error.message}${error.stack ? `\n${error.stack}` : ""}`
          : String(error);
        throw new Error(`Sparse CM12 pipeline ${name} failed: ${detail}${
          shaderErrors ? `\nShader errors:\n${shaderErrors}` : ""}`, { cause: error });
      }
    }));
    const compileFramePlanVerify = async () => ["verifySparseCM12FramePlanPresentation",
      await compiler.compileComputePipeline({
        label: "Sparse CM12 verify FPL1 presentation coverage",
        layout: pipelineLayout,
        compute: { module: shaderModule,
          entryPoint: "verifySparseCM12FramePlanCurrentStage",
          constants: { CM12_FRAME_PLAN_VERIFY_STAGE: 5 } },
      }, { priority: "critical" })] as const;
    const compilationKey = `${presentationPublisherOracleForQA ? "oracle" : "resident"}\0${shaderSource}`;
    let presentationPipelinePromise =
      deviceCompilation.presentationPipelines.get(compilationKey);
    if (!presentationPipelinePromise) {
      presentationPipelinePromise = (async () => Object.freeze(Object.fromEntries([
        ...await compileNamedEntries(presentationNames, "critical"),
        ...(presentationPublisherOracleForQA ? [] : [await compileFramePlanVerify()]),
      ])))();
      deviceCompilation.presentationPipelines.set(compilationKey,
        presentationPipelinePromise);
      void presentationPipelinePromise.catch(() => {
        if (deviceCompilation.presentationPipelines.get(compilationKey)
            === presentationPipelinePromise) {
          deviceCompilation.presentationPipelines.delete(compilationKey);
        }
      });
    }
    const presentationPipelines = await presentationPipelinePromise;
    // Pipelines retain their compiled programs; the one-use source module does
    // not need to remain in the device cache for the lifetime of the solver.
    // Keeping every slice alive made total compiler residency proportional to
    // entry-point count even though compilation itself was serialized.
    deviceCompilation.shaderModules.delete(presentationShaderSource);
    report("First-frame presentation pipelines ready");
    const startSimulationPipelineCompilation = () => {
      let fullPipelinePromise = deviceCompilation.simulationPipelines.get(compilationKey);
      if (!fullPipelinePromise) {
        fullPipelinePromise = (async () => {
          // A single resident module contains hundreds of entry points and a
          // very large helper graph. Passing that module to every pipeline made
          // the driver repeatedly process irrelevant physics. Compile bounded
          // call-graph slices instead; independent chunks can also occupy the
          // compilation manager's background lanes concurrently.
          const simulationNameChunks: string[][] = [];
          let ordinaryChunk: string[] = [];
          const flushOrdinaryChunk = () => {
            if (ordinaryChunk.length > 0) simulationNameChunks.push(ordinaryChunk);
            ordinaryChunk = [];
          };
          for (const name of simulationNames) {
            if (SPARSE_CM12_ISOLATED_SIMULATION_SHADER_ENTRIES.has(name)) {
              flushOrdinaryChunk();
              simulationNameChunks.push([name]);
            } else {
              ordinaryChunk.push(name);
              if (ordinaryChunk.length === SPARSE_CM12_SIMULATION_SHADER_ENTRY_CHUNK_SIZE) {
                flushOrdinaryChunk();
              }
            }
          }
          flushOrdinaryChunk();
          // Dawn's Metal backend retains native module/compiler state for each
          // outstanding request. Fan-out here previously constructed every
          // module and queued every pipeline before the first chunk settled;
          // interrupting that backlog also exposed a Dawn ProcessEvents crash.
          // Keep one chunk's native state live at a time. The compilation
          // manager still serializes the pipelines within the chunk.
          const entries: (readonly [string, GPUComputePipeline])[] = [];
          for (let chunkIndex = 0; chunkIndex < simulationNameChunks.length;
            chunkIndex += 1) {
            const chunkNames = simulationNameChunks[chunkIndex]!;
            const chunkSource = sparseCM12WGSLForEntryPoints(shaderSource, chunkNames);
            const chunkModule = await shaderModuleFor(chunkSource,
              `Sparse CM12 simulation shader ${chunkIndex + 1}/${simulationNameChunks.length}`);
            try {
              entries.push(...await compileNamedEntries(
                chunkNames, "background", chunkModule));
            } finally {
              deviceCompilation.shaderModules.delete(chunkSource);
            }
          }
          // Two pipelines from one entry point: the snapshot variant additionally
          // advances the device-side snapshot cursor and stamps its slot.
          const journalEntries = layout.journal === 0 ? [] : await (async () => {
            const journalShaderModule = await shaderModuleFor(
              sparseCM12WGSLForEntryPoints(shaderSource, ["journalIteration"]),
              "Sparse CM12 journal snapshot shader",
            );
            const journalSource = sparseCM12WGSLForEntryPoints(
              shaderSource, ["journalIteration"]);
            try {
              return [["journalIterationSnapshot", await compiler.compileComputePipeline({
                label: "Sparse CM12 journalIteration with field snapshot",
                layout: pipelineLayout,
                compute: { module: journalShaderModule, entryPoint: "journalIteration",
                  constants: { JOURNAL_SNAPSHOT: 1 } },
              }, { priority: "background" })] as const];
            } finally {
              deviceCompilation.shaderModules.delete(journalSource);
            }
          })();
          const presentationAllocatorEntries = await Promise.all([
            "allocateSparseCM12PresentationPages",
            "sortSparseCM12PresentationPageDirectory",
            "retireSparseCM12PresentationPages",
            "compactSparseCM12PresentationPageDirectory",
          ].map(async (entryPoint) => {
            const module = await shaderModuleFor(
              sparseCM12WGSLForEntryPoints(presentationAllocatorShaderSource, [entryPoint]),
              `Sparse CM12 ${entryPoint} shader`,
            );
            const allocatorSource = sparseCM12WGSLForEntryPoints(
              presentationAllocatorShaderSource, [entryPoint]);
            try {
              return [entryPoint, await compiler.compileComputePipeline({
                label: `Sparse CM12 ${entryPoint}`,
                layout: presentationAllocatorPipelineLayout,
                compute: { module, entryPoint },
              }, { priority: "background" })] as const;
            } finally {
              deviceCompilation.shaderModules.delete(allocatorSource);
            }
          }));
          return Object.freeze(Object.fromEntries([
            ...Object.entries(presentationPipelines),
            ...entries, ...journalEntries,
            ...presentationAllocatorEntries,
          ]));
        })();
        deviceCompilation.simulationPipelines.set(compilationKey, fullPipelinePromise);
        void fullPipelinePromise.catch(() => {
          if (deviceCompilation.simulationPipelines.get(compilationKey)
              === fullPipelinePromise) {
            deviceCompilation.simulationPipelines.delete(compilationKey);
          }
        });
      }
      return fullPipelinePromise;
    };
    const rigidCoupling = rigid
      ? await WebGPUSparseCM12RigidCoupling.create(device, {
        parameters,
        state,
        topologyArena,
        frameControlIndirectArguments,
        rigidBodies: rigid.bodies,
        exchange: rigid.exchange,
      })
      : undefined;
    const result = new WebGPUSparseCM12Resident(
      device, atlas.dimensions, atlas.brickFineResolution, layout,
      [parameters, topology, state, partials, scalars, conditioning, activity,
        candidateState, topologyArena],
      acceptedIndirectArguments,
      pressureCellIndirectArguments,
      pressureMembershipIndirectArguments,
      pressureExecutionIndirectArguments,
      persistentPressureCacheIndirectArguments,
      transportPacketIndirectArguments,
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
      transportBindGroup,
      transportDepthBindGroups,
      effectiveVelocityPressureBindGroup,
      presentationAllocatorBindGroup,
      transportExecutionImageBuffer,
      transportExecutionImageLayout,
      topologyEffectsAuthorityLayout,
      iboSemanticAuthorityBaseWords,
      internedBoundaryImage && internedBoundaryBaseWords !== undefined
        ? [internedBoundaryBaseWords + internedBoundaryImage.layout.slotBaseWords[0],
          internedBoundaryBaseWords + internedBoundaryImage.layout.slotBaseWords[1]]
        : undefined,
      effectiveTransportVelocity,
      velocityExtensionDepths,
      pressureTemplates,
      pressureWorklistData.layout,
      pressureExecutionImageLayout,
      incrementalActivityLayout,
      canonicalMembershipLayout,
      framePlanLayout,
      framePlanPresentationLayout,
      frameControl.layout,
      velocityExtensionLayout,
      faceAddressProgram.layout,
      transportPacketAuthorityLayout,
      finalScalarPacketMaskLayout!,
      phase1TransportQALayout,
      phase1TransportProfileBaseWords,
      pressureTopologyRepairLayout,
      persistentPressureCacheLayout,
      presentationPublisherOracleForQA,
      horizontalD4Authority,
      presentationPipelines,
      startSimulationPipelineCompilation,
      () => compiler.snapshot(),
      physicsCellCapacity, physicsRowCapacity,
      physicsCellCapacity, physicsRowCapacity,
      Math.max(templates.maximumOwnedRowCount, dynamicRowsPerPage),
      pressureCoarseEdgeCount,
      pressureEdgeCount,
      pressureHierarchyGroupCounts.reduce((sum, count) => sum + count, 0),
      pressureHierarchyEdgeCounts.reduce((sum, count) => sum + count, 0),
      pressureScratchBytes,
      pressureFineEdgeImageBaseWords,
      physicalTemplateBytes,
      physicalTemplateBytes + 4 * acceptedLeafManifestBase,
      topologyPagePool.pageCapacity,
      physicalTemplateBytes + 4 * pageDescriptors,
      topologyPagePool.pageWords,
      worldDirectoryLayout,
      solidOccupancyLayout,
      packed.brickCount,
      atlas.bricks.map((brick) => brick.coordinate),
      templates.words,
      rigidCoupling);
    result.writeParameters(packed, 0.004, 1, 1, [0, 0, 0], undefined, undefined,
      undefined, 0, undefined);
    if (solidOccupancyLayout) {
      const initialization = device.createCommandEncoder({
        label: "Sparse CM12 SolidWorld aperture initialization",
      });
      result.encodeSolidWorldApertureRefresh(initialization);
      device.queue.submit([initialization.finish()]);
    }
    if (rigidCoupling) {
      const initialization = device.createCommandEncoder({
        label: "Sparse CM12 solid topology initialization",
      });
      rigidCoupling.encodeInitialization(
        initialization, templates.cellCount, templates.rowCount,
      );
      device.queue.submit([initialization.finish()]);
    }
    return result;
  }

  /** True once every pipeline used by an advancing frame is device-resident. */
  get simulationReady(): boolean { return this.simulationPipelinesReady; }

  get simulationCompilationError(): string | undefined {
    return this.simulationPipelineFailure;
  }

  get simulationCompilationProgress(): GPUCompilationSnapshot {
    return this.simulationCompilationSnapshot();
  }

  /**
   * Start non-critical compilation after generation zero has reached the queue.
   * Presentation pipelines were awaited during create; physics may now warm in
   * the worker without delaying the first visible scene.
   */
  warmSimulationPipelines(): void {
    const start = this.startSimulationPipelineCompilation;
    if (!start) return;
    this.startSimulationPipelineCompilation = undefined;
    this.simulationPipelineCompilation = start().then((pipelines) => {
      if (this.destroyed) return;
      Object.assign(this.pipelines, pipelines);
      this.simulationPipelinesReady = true;
    }).catch((error) => {
      if (this.destroyed) return;
      this.simulationPipelineFailure = error instanceof Error
        ? error.message : String(error);
    });
  }

  /** Deterministic harness seam; the UI remains free to present generation zero. */
  async waitForSimulationPipelines(): Promise<void> {
    this.warmSimulationPipelines();
    await this.simulationPipelineCompilation;
    if (this.simulationPipelineFailure) {
      throw new Error(`Sparse CM12 simulation compilation failed: ${
        this.simulationPipelineFailure}`);
    }
    if (!this.destroyed && !this.simulationPipelinesReady) {
      throw new Error("Sparse CM12 simulation compilation did not publish pipelines");
    }
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
    inflow?: SparseCM12InflowControl,
  ): void {
    this.assertLive();
    this.lastInflow = inflow;
    const packed = this.lastPacked!;
    this.writeParameters(packed, dt_s, finestCellSize_m, pressureScale,
      accelerationFinePerSecond2, sharpening, activityPolicy, pressureControl, bodyCount,
      worldDimensions_m, inflow);
    const pressureIterations = sparseCM12PressureIterations(pressureControl?.iterations);
    const gammaDiffusionEnabled = sharpening?.gammaDiffusionEnabled !== false;
    const surfaceSharpeningEnabled = sharpening?.surfaceSharpeningEnabled !== false;
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
    if (this.phase1TransportQALayout) {
      const layout = this.phase1TransportQALayout;
      const first = layout.baseWords + 3;
      encoder.clearBuffer(this.activity, 4 * first, 4 * (layout.totalWords - first));
    }
    if (this.phase1TransportProfileBaseWords !== undefined) {
      encoder.clearBuffer(this.activity, 4 * this.phase1TransportProfileBaseWords,
        4 * SPARSE_CM12_PHASE1_TRANSPORT_PROFILE_WORDS);
    }
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
      if (count <= 0 || y <= 0 || z <= 0) return;
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroups(count, y, z);
    };
    const pressureFrozenCoarseWorkgroupCount = Math.ceil(Math.max(
      this.pressureCoarseEdgeCount,
      this.pressureHierarchyGroupCount,
      this.pressureHierarchyEdgeCount,
    ) / WORKGROUP_SIZE);
    const pressureFrozenCoarseWorkgroupsX = Math.ceil(
      Math.sqrt(pressureFrozenCoarseWorkgroupCount),
    );
    const pressureFrozenCoarseWorkgroupsY = Math.ceil(
      pressureFrozenCoarseWorkgroupCount / pressureFrozenCoarseWorkgroupsX,
    );
    const pressureMembershipWorkgroups = Math.ceil(
      this.pressureExecutionImageLayout.pressureMembershipWordCount / WORKGROUP_SIZE,
    );
    const dispatchAccepted = (name: string, kind: "cell" | "row") => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      const argumentWord = kind === "cell" ? 8 : 11;
      activePass.dispatchWorkgroupsIndirect(this.acceptedIndirectArguments,
        4 * (argumentWord - 8));
    };
    const dispatchShadow = (name: string, kind: "cell" | "row") => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      const argumentWord = kind === "cell" ? 6 : 9;
      activePass.dispatchWorkgroupsIndirect(this.acceptedIndirectArguments,
        4 * argumentWord);
    };
    const dispatchTopologyDelta = (name: string) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.acceptedIndirectArguments, 72);
    };
    const dispatchShadowLeaves = (name: string) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.acceptedIndirectArguments, 84);
    };
    const dispatchTransportPacket = (name: string) => {
      if (!this.transportPacketIndirectArguments) {
        throw new Error("Sparse CM12 transport packet authority is unavailable");
      }
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.transportPacketIndirectArguments, 0);
    };
    const dispatchPressureCell = (name: string) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.pressureCellIndirectArguments, 0);
    };
    const dispatchPressureBootstrap = (name: string) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(this.pressureMembershipIndirectArguments, 0);
    };
    const dispatchPersistentPressureCache = (name: string, slot: number) => {
      const activePass = openPass();
      activePass.setPipeline(this.pipelines[name]!);
      activePass.dispatchWorkgroupsIndirect(
        this.persistentPressureCacheIndirectArguments, 12 * slot,
      );
    };
    const copyPressureSolveDispatchGate = () => {
      closePass();
      encoder.copyBufferToBuffer(this.pressureWorklists,
        sparseCM12PressureExecutionImageCellIndirectByteOffset(
          this.pressureExecutionImageLayout,
        ),
        this.pressureCellIndirectArguments, 0, 12);
      encoder.copyBufferToBuffer(this.pressureWorklists,
        sparseCM12PressureExecutionImageIndirectByteOffset(
          this.pressureExecutionImageLayout,
        ),
        this.pressureExecutionIndirectArguments, 0, 48);
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
    // Inert unless a lens is armed, and typed so a stage can only name its own
    // lens's taps. Closing the pass is part of the sink because a copy cannot
    // happen inside an open compute pass — and because that close must not
    // happen on the frames nobody is looking, which would change the advance's
    // pass structure for every scene.
    const lensTaps = sparseCM12StageTaps(this.stageLenses, encoder, closePass);
    const useBindGroup = (bindGroup: GPUBindGroup) => {
      activeBindGroup = bindGroup;
      pass?.setBindGroup(0, bindGroup);
    };
    // Without seams this is the single frame pass it has always been. With
    // them, each stage becomes its own pass so a boundary chain can land a
    // hardware timestamp on the pass that opens the next stage. Dispatch order
    // and the implicit barriers between dispatches are identical either way,
    // so a traced advance computes exactly what an untraced one computes.
    const stageLimitForQA = this.stageLimitForQA;
    this.stageLimitForQA = undefined;
    const activityPhaseLimitForQA = this.activityPhaseLimitForQA;
    this.activityPhaseLimitForQA = undefined;
    const transportPhaseLimitForQA = this.transportPhaseLimitForQA;
    this.transportPhaseLimitForQA = undefined;
    const pressureTopologyPhaseLimitForQA = this.pressureTopologyPhaseLimitForQA;
    this.pressureTopologyPhaseLimitForQA = undefined;
    let stageLimitReached = false;
    const stage = <Id extends SparseCM12ResidentStageId>(
      id: Id,
      encodeStage: (own: SparseCM12StageEncodeContext<Id>) => void,
    ) => {
      if (stageLimitReached) return;
      if (seams) {
        passLabel = `Sparse CM12 resident ${id}`;
      }
      encodeStage({
        // A sub-seam is a pass boundary too, and only the stage that owns it
        // may close it: the type of `substage` is this stage's row of
        // SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES.
        closeSubstage: (substage) => {
          if (!seams?.closeSubstage) return;
          pass?.end();
          pass = undefined;
          seams.closeSubstage(id, substage);
        },
        lens: lensTaps.for(id),
      });
      if (id === stageLimitForQA) stageLimitReached = true;
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
    const leafCapacity = this.worldDirectoryLayout.leafCapacity;
    const bricks = Math.ceil(leafCapacity / WORKGROUP_SIZE);
    const directPacketWidth = this.transportPacketAuthorityLayout!.dispatchWidth;
    const directPacketRows = this.transportPacketAuthorityLayout!.dispatchRows;
    stage("transport-velocity-extension", ({ closeSubstage }) => {
      // FCA1 translates external inputs and persistent D4 receipts into a
      // sealed set of fixed indirect families. The host always encodes both
      // work and singleton bypass packets; it never inspects evolving state.
      dispatch("beginSparseCM12FrameControl", 1);
      dispatch("publishSparseCM12FrameBodyAuthority", 1);
      dispatch("sealSparseCM12FrameControl", 1);
      closePass();
      encoder.copyBufferToBuffer(this.topologyArena,
        4 * this.frameControlLayout.indirectBaseWords,
        this.frameControlIndirectArguments, 0,
        12 * SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT);
      this.rigidCoupling?.encodeVoxelization(encoder);
      useBindGroup(this.bindGroup);
      dispatchFrameControl("publishSparseCM12MovingSolidActivity",
        SPARSE_CM12_FRAME_CONTROL_FAMILY.solidCellWork);
      dispatchFrameControl("sparseCM12FrameControlNoop",
        SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyBypass);
      dispatchFrameControl("sparseCM12FrameControlNoop",
        SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyRowBypass);
      closeSubstage("frame-control-authority");
      // VEX2 owns no catalogue or generation protocol. Compact B-profile
      // ordinals map directly to the stable leaf*64 TEI packet address.
      useBindGroup(this.transportBindGroup);
      dispatch("initializeVelocityExtensionPackets", directPacketWidth, directPacketRows);
      closeSubstage("velocity-extension-mask-initialization");
      for (let depth = 1; depth <= 8; depth += 1) {
        useBindGroup(this.transportDepthBindGroups[depth - 1]!);
        dispatch("advanceVelocityExtensionPackets", directPacketWidth, directPacketRows);
      }
      closeSubstage("velocity-extension-sweeps");
      closePass();
      encoder.clearBuffer(this.activity,
        4 * this.transportPacketAuthorityLayout!.indirectBaseWords, 4);
      dispatch("compileSparseCM12TransportPacketsFromFinalScalarMasks",
        this.transportPacketAuthorityLayout!.compilerDispatchWidth,
        this.transportPacketAuthorityLayout!.compilerDispatchRows);
      closePass();
      encoder.copyBufferToBuffer(this.activity,
        4 * this.transportPacketAuthorityLayout!.indirectBaseWords,
        this.transportPacketIndirectArguments!, 0, 12);
      closeSubstage("transport-packet-authority");
      useBindGroup(this.pressureBindGroup);
    });
    stage("face-preparation", ({ closeSubstage }) => {
      useBindGroup(this.transportBindGroup);
      dispatch("clearSparseCM12RetiredFaceVelocitySupport",
        this.incrementalActivityLayout.brickCount);
      dispatch("publishSparseCM12FaceVelocitySupport",
        this.incrementalActivityLayout.brickCount);
      closeSubstage("face-support-publication");
      dispatch("prepareSparseCM12InteriorFaceTiles",
        Math.min(this.faceAddressLayout.dispatchWidth,
          this.faceAddressLayout.interiorTileCount),
        this.faceAddressLayout.interiorDispatchRows);
      dispatch("prepareSparseCM12SeamFacePackets",
        Math.min(this.faceAddressLayout.dispatchWidth,
          this.faceAddressLayout.seamPacketCount),
        this.faceAddressLayout.seamDispatchRows);
      dispatch("prepareSparseCM12SparseAirFacePackets",
        Math.min(this.faceAddressLayout.dispatchWidth,
          this.faceAddressLayout.seamPacketCount),
        this.faceAddressLayout.seamDispatchRows);
      dispatchAccepted("prepareSparseCM12DynamicFaceRows", "row");
      closeSubstage("dirty-face-row-preparation");
    });
    stage("conservative-transport", ({ closeSubstage }) => {
      useBindGroup(this.transportBindGroup);
      dispatchAccepted("clearSparseCM12TransportReceipts", "cell");
      if (this.phase1TransportQALayout) {
        dispatchAccepted("captureSparseCM12Phase1TransportPackets", "cell");
      }
      if (transportPhaseLimitForQA === "setup") return;
      dispatchTransportPacket("traceGammaAndBeta");
      closeSubstage("transport-trace");
      if (transportPhaseLimitForQA === "trace") return;
      dispatchTransportPacket("scatterDensityDeficit");
      closeSubstage("transport-scatter");
      if (transportPhaseLimitForQA === "scatter") return;
      dispatchTransportPacket("gatherConservativeDensity");
      closeSubstage("transport-gather");
      if (transportPhaseLimitForQA === "gather") return;
    });
    useBindGroup(this.pressureBindGroup);
    closePass();
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
      useBindGroup(this.transportBindGroup);
      const groups = Math.ceil(this.tracerLattice.count / WORKGROUP_SIZE);
      if (this.tracerSeedPending) {
        dispatch("seedTracers", groups);
        this.tracerSeedPending = false;
      }
      dispatch("advanceTracers", groups);
      useBindGroup(this.pressureBindGroup);
    });
    stage("gamma-diffusion", () => {
      if (!gammaDiffusionEnabled) return;
      // Gamma shares the accepted physical row topology with pressure and
      // SolidWorld. No host-only boundary image or packet mask is a second
      // authority for which rows exist.
      dispatchAccepted("clearGammaReceipts", "cell");
      dispatchAccepted("scatterGammaSnapshotRows", "row");
      dispatchAccepted("finalizeGammaSnapshot", "cell");
      dispatchAccepted("clearGammaReceipts", "cell");
      dispatchAccepted("scatterGammaRefinementRows", "row");
      dispatchAccepted("finalizeGammaRefinement", "cell");
    });
    stage("surface-sharpening", ({ closeSubstage }) => {
      // Start the stage with real, already-required GPU work. Besides resetting
      // the downstream counters before any producer can observe them, this pass
      // materializes the preceding stage's timestamp before the native receipt
      // clear and indirect copy below; copy-only prefixes cannot carry pass
      // timestamp writes of their own.
      // Conservative transport publishes physical air-side source packets.
      closePass();
      if (surfaceSharpeningEnabled) {
        // A native fill of the receipt segment is cheaper than a lock/stamp
        // protocol on every trilinear receiver.
        encoder.clearBuffer(this.conditioning, 24 * this.templateCellCount,
          4 * this.templateCellCount);
        encoder.clearBuffer(this.state, 4 * this.layout.sharpeningDelta,
          4 * this.templateCellCount);
      }
      closeSubstage("sharpening-receipt-setup");
      useBindGroup(this.transportBindGroup);
      if (surfaceSharpeningEnabled) {
        dispatchTransportPacket("prepareSharpeningField");
        dispatchTransportPacket("scatterSharpeningMass");
      }
      closeSubstage("sharpening-transform");
      useBindGroup(this.pressureBindGroup);
      // Diffusion finishes in immutable scratch banks. When sharpening is off,
      // its finalizer becomes the lightweight commit that publishes those
      // banks into the destination pair. With both transforms off, transport
      // already owns the destination pair and no scalar transform is needed.
      if (surfaceSharpeningEnabled || gammaDiffusionEnabled) {
        dispatchAccepted("finalizeSharpening", "cell");
      }
      closeSubstage("sharpening-finalize");
      // Relay over no more than one B8 page width (the accepted support reach).
      // One pass only moved an over-capacity packet into an already-full
      // neighbour; after floor impact that concentrated conserved mass into a
      // shrinking set of cells (rho > 6) and looked like volume loss. Eight
      // paired debit/credit passes reach nearby free-surface capacity without
      // turning the repair into an unbounded flood across SparseWorld.
      for (let capacityPass = 0; capacityPass < 8; capacityPass += 1) {
        dispatchAccepted("initializeDensityCapacityRepair", "cell");
        dispatchAccepted("scatterDensityCapacityRepair", "cell");
        dispatchAccepted("finalizeDensityCapacityRepair", "cell");
      }
      closeSubstage("density-capacity-repair");
      // Final scalar facts are authored once in the accepted TEI packet space.
      // Every remaining dirty carrier below is a mask consumer; finalization no
      // longer walks incidence or appends a tile event per changed cell.
      useBindGroup(this.transportBindGroup);
      dispatch("beginSparseCM12FinalScalarMasks", 1);
      dispatch("publishSparseCM12FinalScalarMasks", leafCapacity);
      dispatch("sealSparseCM12FinalScalarMasks", 1);
      closeSubstage("final-scalar-mask-publication");
      useBindGroup(this.pressureBindGroup);
      closePass();
    });
    stage("symmetry-authority", () => {
      dispatchFrameControl("preserveHorizontalD4",
        SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Work);
      dispatchFrameControl("commitHorizontalD4",
        SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Work);
      dispatchFrameControl("sparseCM12FrameControlNoop",
        SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Bypass);
      dispatch("publishSparseCM12FrameScalarOutput", 1);
    });
    stage("body-forces", () => {
      dispatchAccepted("forceFaces", "row");
    });
    stage("pressure-topology", ({ closeSubstage }) => {
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
      closePass();
      const persistentFamilies = ["brick", "aggregateEdge", "hierarchyNode",
        "hierarchyEdge"] as const;
      persistentFamilies.forEach((family, index) => {
        encoder.copyBufferToBuffer(this.topologyArena,
          sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(
            this.persistentPressureCacheLayout, family, "seed"),
          this.persistentPressureCacheIndirectArguments, 12 * index, 12);
      });
      dispatchPersistentPressureCache("seedPreviousPCFBrickLeaves", 0);
      dispatchPersistentPressureCache("seedPreviousPCFAggregateEdgeLeaves", 1);
      dispatchPersistentPressureCache("seedPreviousPCFHierarchyNodeLeaves", 2);
      dispatchPersistentPressureCache("seedPreviousPCFHierarchyEdgeLeaves", 3);
      dispatch("finalizeSparseCM12PressureTopologyBrickFrontier", 1);
      // Indirect arguments are copied out of GPU-authored headers. WebGPU
      // forbids transfer commands while the shared compute pass is open.
      closePass();
      encoder.copyBufferToBuffer(this.pressureWorklists,
        4 * (this.pressureRepairLayout.headerBaseWords
          + SPARSE_CM12_PRESSURE_REPAIR_HEADER.bootstrapCellIndirect),
        this.pressureMembershipIndirectArguments, 0,
        SPARSE_CM12_PRESSURE_MEMBERSHIP_INDIRECT_BYTES);
      closeSubstage("ptr-setup-brick-plan");
      if (pressureTopologyPhaseLimitForQA === "setup") return;
      dispatchPressureBootstrap("classifyPressureCells");
      dispatchAccepted("classifyDirtyPressureCells", "cell");
      dispatch("finalizeCanonicalPressureCellFrontier", 1);
      closePass();
      encoder.copyBufferToBuffer(this.activity,
        sparseCM12CanonicalMembershipRepairIndirectByteOffset(
          this.canonicalMembershipLayout),
        this.pressureMembershipIndirectArguments, 0, 12);
      dispatchPressureBootstrap("repairCanonicalPressureCellLeaves");
      dispatch("finalizeCanonicalPressureCells", 1);
      closePass();
      closeSubstage("pcm-cell-publication");
      if (pressureTopologyPhaseLimitForQA === "cells") return;
      dispatch("compileCanonicalPressureRows",
        this.canonicalMembershipLayout.row.dispatchWorkgroupCount);
      dispatch("finalizeCanonicalPressureRows", 1);
      closePass();
      closeSubstage("pcm-row-publication");
      if (pressureTopologyPhaseLimitForQA === "rows") return;
      // Fine coefficients are now a complete PEI publication, not a private
      // PCF dirty-leaf repair. PTR captured the already-open PCA candidate;
      // publish the canonical PEI image before any aggregate reduction.
      encoder.copyBufferToBuffer(this.pressureWorklists,
        sparseCM12PressureExecutionImageCellIndirectByteOffset(
          this.pressureExecutionImageLayout,
        ),
        this.pressureCellIndirectArguments, 0, 12);
      dispatchPressureCell("publishFrozenPressureCellIds");
      dispatch("publishFrozenPressureMembership", pressureMembershipWorkgroups);
      dispatchPressureCell("publishFrozenPressureCoefficients");
      dispatch("sealPersistentPressureAggregateFrontier", 1);
      closePass();
      closeSubstage("pca-fine-publication");
      if (pressureTopologyPhaseLimitForQA === "fine") return;
      (persistentFamilies.slice(0, 2) as readonly (typeof persistentFamilies[number])[])
        .forEach((family, index) => {
          encoder.copyBufferToBuffer(this.topologyArena,
            sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(
              this.persistentPressureCacheLayout, family, "repair"),
            this.persistentPressureCacheIndirectArguments, 12 * (4 + index), 12);
        });
      dispatchPersistentPressureCache("repairPersistentPressureBrickWorkset", 4);
      dispatchPersistentPressureCache("repairPersistentPressureAggregateEdgeWorkset", 5);
      dispatch("finalizePersistentPressureAggregatePlan", 1);
      closePass();
      if (pressureTopologyPhaseLimitForQA === "coarse-plan") return;
      (persistentFamilies.slice(0, 2) as readonly (typeof persistentFamilies[number])[])
        .forEach((family, index) => {
          encoder.copyBufferToBuffer(this.topologyArena,
            sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(
              this.persistentPressureCacheLayout, family, "work"),
            this.persistentPressureCacheIndirectArguments, 12 * (8 + index), 12);
        });
      if (pressureTopologyPhaseLimitForQA === "coarse-indirect") return;
      dispatchPersistentPressureCache("repairPersistentPressureAggregateEdges", 9);
      if (pressureTopologyPhaseLimitForQA === "coarse-edge") return;
      dispatchPersistentPressureCache("repairPersistentPressureBrickDiagonals", 8);
      if (pressureTopologyPhaseLimitForQA === "coarse-work") return;
      dispatch("finalizePersistentPressureAggregateExecution", 1);
      closePass();
      closeSubstage("pca-coarse-repair");
      if (pressureTopologyPhaseLimitForQA === "coarse") return;
      (persistentFamilies.slice(2) as readonly (typeof persistentFamilies[number])[])
        .forEach((family, index) => {
          encoder.copyBufferToBuffer(this.topologyArena,
            sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(
              this.persistentPressureCacheLayout, family, "repair"),
            this.persistentPressureCacheIndirectArguments, 12 * (6 + index), 12);
        });
      dispatchPersistentPressureCache("repairPersistentPressureHierarchyNodeWorkset", 6);
      dispatchPersistentPressureCache("repairPersistentPressureHierarchyEdgeWorkset", 7);
      dispatch("finalizePersistentPressureHierarchyPlan", 1);
      closePass();
      (persistentFamilies.slice(2) as readonly (typeof persistentFamilies[number])[])
        .forEach((family, index) => {
          encoder.copyBufferToBuffer(this.topologyArena,
            sparseCM12PersistentPressureCacheAggregateIndirectByteOffset(
              this.persistentPressureCacheLayout, family, "work"),
            this.persistentPressureCacheIndirectArguments, 12 * (10 + index), 12);
        });
      dispatchPersistentPressureCache("repairPersistentPressureHierarchyDiagonals", 10);
      dispatchPersistentPressureCache("repairPersistentPressureHierarchyEdges", 11);
      dispatch("finalizePersistentPressureCache", 1);
      dispatch("publishFrozenPressureCoarseCache", pressureFrozenCoarseWorkgroupsX,
        pressureFrozenCoarseWorkgroupsY);
      closeSubstage("pca-hierarchy-and-freeze");
      if (pressureTopologyPhaseLimitForQA === "hierarchy") return;
      dispatch("finalizeSparseCM12PressureExecutionImage", 1);
      // Republish after finalize so a generation/count fault overwrites the
      // temporary publication triplet and fail-closes every solve consumer.
      closePass();
      closeSubstage("pei-publication");
      encoder.copyBufferToBuffer(this.pressureWorklists,
        sparseCM12PressureExecutionImageCellIndirectByteOffset(
          this.pressureExecutionImageLayout,
        ),
        this.pressureCellIndirectArguments, 0, 12);
      encoder.copyBufferToBuffer(this.pressureWorklists,
        sparseCM12PressureExecutionImageIndirectByteOffset(
          this.pressureExecutionImageLayout,
        ),
        this.pressureExecutionIndirectArguments, 0, 48);
      // PTR and PCF form one GPU-authored transaction. Once PCF accepts the
      // captured coefficient generation, one scalar receipt closes PTR; the
      // per-brick states were already published by the topology transaction.
      dispatch("finalizeSparseCM12BoundedPressureTopologyRepair", 1);
      // Open the next topology journal immediately. Resolution/activation/
      // retirement producers later in this frame append without host state.
      dispatch("beginSparseCM12PressureTopologyRepair", 1);
      dispatchPressureCell("preparePressure");
      closePass();
      closeSubstage("ptr-commit-and-prepare-pressure");
    });
    stage("pressure-rhs", () => {
      dispatch("beginPressureSolve", 1);
      dispatchPressureCell("initializePCG");
      dispatchPressureCell("initializeJacobiDirection");
      dispatch("reduceInitialize", 1);
      dispatchPressureCell("measureTrueResidual");
      dispatch("reduceInitialTrueResidual", 1);
      dispatchPressureCell("initializePipelinedImage");
      dispatch("reducePipelinedInitialize", 1);
      // The seed can already satisfy the tolerance. Publish the device gate
      // before the first fixed eight-iteration block in that case.
      dispatch("publishPressureSolveDispatchGate", 1);
      copyPressureSolveDispatchGate();
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
          dispatchPressureCell("applyJacobiPreconditioner");
          dispatchPressureCell("applyPipelinedImage");
          dispatch("reducePipelinedIteration", 1);
          if ((iteration + 1) % SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE === 0
            && iteration + 1 < pressureIterations) {
            dispatchPressureCell("measureGuardedTrueResidual");
            dispatch("reduceGuardedTrueResidual", 1);
            dispatchPressureCell("restartPCGAfterCurvatureLoss");
            dispatchPressureCell("initializeJacobiRecoveryDirection");
            dispatch("reduceCurvatureRecovery", 1);
            dispatchPressureCell("applyPipelinedRecovery");
            dispatch("reducePipelinedRecovery", 1);
            // Keep the complete command schedule encoded, but snapshot the
            // GPU-authored x=0 gate before the next block when this block met
            // the tolerance. Curvature recovery republishes the live counts.
            dispatch("publishPressureSolveDispatchGate", 1);
            copyPressureSolveDispatchGate();
          }
          // After the cadence guard, so a record carries the true-residual
          // check and the gate closure that its own iteration earned.
          journalRecord(iteration + 1);
      }
      // Always close the solve with a fresh b-Ap application. No convergence
      // or performance receipt may rely on the recursive residual.
      dispatch("restorePressureSolveDispatches", 1);
      copyPressureSolveDispatchGate();
      dispatchPressureCell("measureTrueResidual");
      dispatch("reduceFinalTrueResidual", 1);
    });
    stage("velocity-projection", () => {
      // The brick-scalar arm opens this generation before scalar comparison.
      useBindGroup(this.bindGroup);
      dispatch("advanceActivityClock", 1);
      dispatch("beginIncrementalActivity", 1);
      useBindGroup(this.pressureBindGroup);
      dispatch("projectSparseCM12InteriorFaceTiles",
        Math.min(this.faceAddressLayout.dispatchWidth,
          this.faceAddressLayout.interiorTileCount),
        this.faceAddressLayout.interiorDispatchRows);
      dispatch("projectSparseCM12SeamFacePackets",
        Math.min(this.faceAddressLayout.dispatchWidth,
          this.faceAddressLayout.seamPacketCount),
        this.faceAddressLayout.seamDispatchRows);
      dispatch("projectSparseCM12SparseAirFacePackets",
        Math.min(this.faceAddressLayout.dispatchWidth,
          this.faceAddressLayout.seamPacketCount),
        this.faceAddressLayout.seamDispatchRows);
      dispatchAccepted("projectSparseCM12DynamicFaceRows", "row");
      dispatchAccepted("enforceSparseCM12InflowFaces", "row");
      useBindGroup(this.effectiveVelocityPressureBindGroup);
      dispatchAccepted("collocateAndDiagnose", "cell");
      dispatchAccepted("measureDivergenceDiagnostics", "cell");
      dispatch("reduceDivergenceDiagnostics", 1);
      // The divergence this stage produced. Four stages downstream
      // `candidate-transfer` zeroes it on every cell whose topology changed,
      // so a lens reading it at frame end would be right everywhere except
      // where the frame was interesting.
      dispatchFrameControl("preserveVelocityHorizontalD4",
        SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Work);
      dispatchFrameControl("commitVelocityHorizontalD4",
        SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Work);
      dispatchFrameControl("sparseCM12FrameControlNoop",
        SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Bypass);
      useBindGroup(this.bindGroup);
      if (this.rigidCoupling) {
        pass?.end();
        pass = undefined;
        this.rigidCoupling.encodeReaction(encoder);
      }
      useBindGroup(this.bindGroup);
      dispatch("publishSparseCM12FrameFaceOutput", 1);
    });
    stage("projection-diagnostics", () => {});
    stage("activity-measurement", () => {
      useBindGroup(this.bindGroup);
      dispatch("markIncrementalActivityScalarBricks", leafCapacity);
      if (activityPhaseLimitForQA === "scalar") return;
      dispatch("markIncrementalActivityTopology",
        Math.ceil(leafCapacity / WORKGROUP_SIZE));
      if (activityPhaseLimitForQA === "topology") return;
      dispatch("finalizeIncrementalActivityMasks", 1);
      if (activityPhaseLimitForQA === "masks") return;
      dispatch("measureBrickActivity", this.incrementalActivityLayout.brickCount);
      if (this.horizontalD4Authority) {
        dispatch("preserveActivityHorizontalD4", bricks);
        dispatch("commitActivityHorizontalD4", bricks);
      }
      if (activityPhaseLimitForQA === "measure") return;
      dispatch("ageIncrementalActivityHistory",
        Math.ceil(leafCapacity / WORKGROUP_SIZE));
      if (activityPhaseLimitForQA === "history") return;
      dispatch("finalizeIncrementalActivityCensus", 1);
      if (activityPhaseLimitForQA === "census") return;
      if (this.solidOccupancyLayout) {
        dispatch("allocateSparseWorldFrontier",
          Math.ceil(26 * leafCapacity / WORKGROUP_SIZE));
        if (activityPhaseLimitForQA === "allocation") return;
        dispatch("synthesizeSparseWorldFrontierPages", this.topologyPageCapacity);
        if (activityPhaseLimitForQA === "synthesis") return;
      }
    });
    stage("resolution-planning", () => {
      dispatch("planBrickResolution", bricks);
      dispatch("activateSweptFrontierPages", bricks);
      // Lifecycle membership is a topology candidate, not a post-publication
      // mutation.  Retirement marks same-rung delta work consumed by the
      // shadow worklists and the single selector flip below.
      dispatch("retireUnsupportedEmptyBricks", bricks);
      for (let gradingPass = 0;
        gradingPass < Math.log2(this.brickFineResolution); gradingPass += 1) {
        dispatch("closePlannedResolution", bricks);
      }
      dispatch("validateCandidateResolution", bricks);
      dispatch("scheduleTopologyPreparation", 1);
      dispatch("allocateCandidateTopologyPages", bricks);
      // Candidate rerung pages exist only for authored leaves. Dynamic WDR
      // leaves already own complete fixed-B8 pages, so dispatching the entire
      // growth slab merely launched hundreds of workgroups that returned at
      // the shader's CM12_WDR_INITIAL_LEAVES guard.
      dispatch("synthesizeCandidateCellPages", this.worldDirectoryLayout.initialLeaves);
      dispatchShadow("clearShadowRowMembership", "row");
      dispatch("beginShadowTopology", 1);
      dispatch("buildShadowLeafWorklist", 1);
      closePass();
      encoder.copyBufferToBuffer(this.topologyArena,
        this.acceptedLeafManifestBaseBytes + 4 * 15,
        this.acceptedIndirectArguments, 84, 12);
      dispatchShadowLeaves("buildShadowStructureWorklist");
      dispatch("finalizeShadowWorklists", 1);
      closePass();
      encoder.copyBufferToBuffer(this.topologyArena,
        this.topologyWorklistBaseBytes + 4 * 20,
        this.acceptedIndirectArguments, 24, 24);
      encoder.copyBufferToBuffer(this.topologyArena,
        this.acceptedLeafManifestBaseBytes + 4 * 7,
        this.acceptedIndirectArguments, 60, 12);
      encoder.copyBufferToBuffer(this.topologyArena,
        this.acceptedLeafManifestBaseBytes + 4 * 12,
        this.acceptedIndirectArguments, 72, 12);
    });
    stage("candidate-transfer", ({ closeSubstage }) => {
      dispatchTopologyDelta("transferCandidateCellsFromTopologyDelta");
      closeSubstage("candidate-field-transfer");
      dispatchTopologyDelta("transferCandidateFacesFromTopologyDelta");
      closeSubstage("candidate-face-reconstruction");
      dispatchShadow("validateCandidateShadowFaces", "row");
      closeSubstage("candidate-face-validation");
      dispatch("beginSparseCM12TopologyEffectsPreflight", 1);
      dispatchTopologyDelta("recordCandidateTopologyEffectsFromTopologyDelta");
      dispatch("finalizeSparseCM12TopologyEffectsPreflight", 1);
      closeSubstage("candidate-effects-preflight");
      useBindGroup(this.transportBindGroup);
        dispatch("beginSparseCM12InternedBoundaryDelta", 1);
        dispatchTopologyDelta("compileSparseCM12InternedBoundaryDelta");
        closeSubstage("candidate-ibo-construction");
        dispatch("finalizeSparseCM12ISAChangedSetReceipt", 1);
        dispatchTopologyDelta("validateSparseCM12InternedBoundaryDeltaPackets");
        dispatch("finalizeSparseCM12InternedBoundaryDelta", 1);
      closeSubstage("candidate-ibo-validation");
      dispatchTopologyDelta("compileSparseCM12TransportExecutionImageShadow");
      closeSubstage("candidate-tei-compilation");
      useBindGroup(this.bindGroup);
      dispatch("validateAndAuthorizeShadowTopology", 1);
      closeSubstage("candidate-authorization");
        closePass();
        encoder.copyBufferToBuffer(this.topologyArena,
          sparseCM12TopologyEffectsIndirectByteOffset(
            this.topologyEffectsAuthorityLayout!),
          this.frameControlIndirectArguments, 0, 12);
        const ptrPass = openPass();
        ptrPass.setPipeline(this.pipelines.publishSparseCM12TopologyPTREffects!);
        ptrPass.dispatchWorkgroupsIndirect(this.frameControlIndirectArguments, 0);
        closeSubstage("candidate-ptr-publication");
        dispatch("sealSparseCM12AuthorizedTopologyEffects", 1);
      dispatch("finishSparseCM12TopologyEffectsPublication", 1);
      closeSubstage("candidate-effects-seal");
      useBindGroup(this.transportBindGroup);
      dispatchTopologyDelta("publishCandidateTopologyDeltaFromWorklist");
      useBindGroup(this.bindGroup);
      if (this.solidOccupancyLayout) {
        // The authorized host fields/rungs are now stable while the accepted
        // selector still names the old worklists. Reconcile the canonical
        // voxel seam graph in that protected publication interval so face
        // publication below consumes the exact graph that will be accepted.
        dispatch("connectSparseWorldFrontierPages", this.topologyPageCapacity);
      }
      dispatchShadow("publishCandidateShadowFaces", "row");
      dispatch("finalizeAuthorizedShadowTopology", 1);
      if (this.solidOccupancyLayout) {
        dispatch("publishSparseWorldFrontierAcceptance",
          this.topologyPageCapacity);
        useBindGroup(this.transportBindGroup);
        dispatch("compileSparseWorldFrontierExecutionImage",
          this.topologyPageCapacity);
        useBindGroup(this.bindGroup);
      }
      closeSubstage("candidate-state-publication");
      useBindGroup(this.transportBindGroup);
      dispatchTopologyDelta("replaySparseCM12TransportExecutionImageRetired");
      dispatchTopologyDelta("replaySparseCM12InternedBoundaryDelta");
      closeSubstage("candidate-image-replay");
      useBindGroup(this.bindGroup);
    });
    stage("brick-retirement", () => {
      // Candidate transfer can publish a newly accepted cell after the normal
      // projection-side D4 authority pass.  Re-apply the same scene authority
      // to the final accepted worklist so pressure/velocity diagnostics cannot
      // expose whichever member of a horizontal orbit happened to transition
      // first.  A direct capacity dispatch is intentional: the accepted
      // indirect snapshot is not promoted until the frame commit below.
      if (this.horizontalD4Authority) {
        dispatch("preserveVelocityHorizontalD4",
          Math.ceil(this.templateCellCount / WORKGROUP_SIZE));
        dispatch("commitVelocityHorizontalD4",
          Math.ceil(this.templateCellCount / WORKGROUP_SIZE));
      }
      // Candidate commit/retirement occurs after the activity census. Re-open
      // only those changed leaf spans for presentation and next-frame reuse.
      dispatch("markIncrementalActivityPostTopology",
        Math.ceil(leafCapacity / WORKGROUP_SIZE));
      // Post-census topology commits update next frame's direct brick mask.
      dispatch("finalizeIncrementalActivityMasks", 1);
    });
    stage("presentation-publication", () => {
      // Newly accepted topology receives a physical presentation page here,
      // from the same working-set-shaped slab used at generation zero.
      // Inactive leaves retain INVALID and consume neither metadata nor payload.
      // FPP1 sees the mapping only after this dispatch completes.
      useBindGroup(this.presentationAllocatorBindGroup);
      dispatch("allocateSparseCM12PresentationPages",
        Math.ceil(leafCapacity / WORKGROUP_SIZE));
      dispatch("sortSparseCM12PresentationPageDirectory", 1);
      useBindGroup(this.bindGroup);
      // The plan and compact page count are GPU publications. Split at the
      // storage-to-indirect copy seam; no host parity/count controls this path.
      closePass();
      this.encodeFramePlanPresentation(encoder, "Sparse CM12 frame presentation");
      // FPP1 has now published the retiring generation's complete all-air
      // pages. Remove those pages from renderer lookup and return their slots
      // before accepting the next frame-plan generation.
      useBindGroup(this.presentationAllocatorBindGroup);
      dispatch("retireSparseCM12PresentationPages",
        Math.ceil(leafCapacity / WORKGROUP_SIZE));
      dispatch("compactSparseCM12PresentationPageDirectory", 1);
      useBindGroup(this.bindGroup);
      dispatch("commitSparseCM12FrameControl", 1);
    });
    closePass();
    // The commit above authors next frame's accepted workgroup triplets. Keep
    // the writable arena out of indirect-dispatch synchronization scopes by
    // snapshotting those six words with a device-side copy between passes.
    encoder.copyBufferToBuffer(this.topologyArena,
      this.topologyWorklistBaseBytes + 4 * 8,
      this.acceptedIndirectArguments, 0, 6 * 4);
    encoder.copyBufferToBuffer(this.topologyArena,
      this.acceptedLeafManifestBaseBytes + 4 * 4,
      this.acceptedIndirectArguments, 48, 12);
    // Promote this frame's captures and queue the header read. A tap in a
    // branch the frame did not take is simply absent from the new set, so its
    // phases paint magenta next frame rather than redrawing the last frame
    // that did take it.
    this.stageLenses?.endFrame(encoder);
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
   * Per-stage lenses and the buffers they draw.
   *
   * Everything a lens reads already exists because the solve needs it, so
   * offering them costs the solver nothing. What is *not* free is a tap — the
   * copy at a seam inside a stage — and that is why the source arms one lens at
   * a time and the encode body asks it before copying anything.
   *
   * The addressing block is rebuilt from the layout on every access rather than
   * cached, because row and cell counts move with the topology and a lens that
   * held last re-mesh's offsets would read the pressure field where the
   * divergence now is.
   */
  get stageLensSource(): StageLensSource {
    this.stageLenses ??= new SparseCM12StageLensSource({
      device: this.device,
      state: this.state,
      arena: this.topologyArena,
      lenses: SPARSE_CM12_LENSES,
      addressing: () => this.stageLensAddressing(),
      space: () => ({
        dimensions: this.dimensions,
        origin_m: [0, 0, 0],
        extent_m: [
          this.dimensions[0] * (this.parameterF32[41] ?? 0),
          this.dimensions[1] * (this.parameterF32[41] ?? 0),
          this.dimensions[2] * (this.parameterF32[41] ?? 0),
        ],
      }),
      headerRange: () => undefined,
    });
    return this.stageLenses;
  }

  /** This frame's field bases, in the frames the addressing block documents. */
  private stageLensAddressing(): SparseCM12AddressingSpec {
    const l = this.layout;
    return {
      rowCount: this.rowCount,
      cellCount: this.cellCount,
      brickCount: this.lastPacked?.brickCount ?? 0,
      faceParityWord: this.frameControlLayout.baseWords
        + SPARSE_CM12_FRAME_CONTROL_HEADER.faceParity,
      faceBase: l.faceA,
      faceBankStride: l.faceB - l.faceA,
      cellFieldStride: l.rhs - l.pressure,
      pressure: l.pressure, rhs: l.rhs, divergence: l.divergence,
      diagonal: l.diagonal, liquid: l.liquid, theta: l.theta,
      residual: l.residual, applied: l.applied,
      // No lens reads the extension banks yet, and publishing a base a lens
      // could not have been written against would be an invitation to read one.
      extensionRootCauseWords: 0,
      extensionRootStampWords: 0,
      extensionAcceptedDepthWords: 0,
      cellVelocityA: l.cellVelocityA,
      cellVelocityB: l.cellVelocityB,
      scalarParityWord: this.frameControlLayout.baseWords
        + SPARSE_CM12_FRAME_CONTROL_HEADER.scalarParity,
    };
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
    const bricks = this.worldDirectoryLayout.leafCapacity;
    const plan = encoder.beginComputePass({ label: `${label} FPL1/FPP1 plan` });
    plan.setBindGroup(0, this.bindGroup);
    const dispatch = (name: string, x: number) => {
      plan.setPipeline(this.pipelines[name]!);plan.dispatchWorkgroups(x);
    };
    dispatch("beginSparseCM12FramePlanNext", 1);
    dispatch("initializeSparseCM12FramePlanNext", bricks);
    dispatch("populateSparseCM12PresentationFramePlan", bricks);
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

  /**
   * Adopt authored cell-size boxes for the next topology plan.
   *
   * This is a uniform-tier edit: no topology or field is changed on the host.
   * The next encoded planner pass reads these bytes and publishes any required
   * resolution transaction through the ordinary candidate path.
   */
  setRefinementRegionParameters(parameters: ArrayBuffer): void {
    this.assertLive();
    if (parameters.byteLength !== SPARSE_CM12_REFINEMENT_REGION_BYTES) {
      throw new RangeError(`Sparse CM12 refinement-region parameters must occupy ${
        SPARSE_CM12_REFINEMENT_REGION_BYTES} bytes`);
    }
    this.refinementRegionParameters.set(new Uint8Array(parameters));
  }

  /** Adopt one accepted uniform solid generation without rebuilding fluid topology. */
  setSolidWorld(solidWorld: SolidWorld): void {
    this.assertLive();
    if (!this.solidOccupancyLayout) return;
    const clear = this.device.createCommandEncoder({
      label: "Sparse CM12 replace SolidWorld occupancy",
    });
    clear.clearBuffer(this.topologyArena, 4 * this.solidOccupancyLayout.baseWords,
      4 * (this.solidOccupancyLayout.totalWords - this.solidOccupancyLayout.baseWords));
    this.device.queue.submit([clear.finish()]);
    writeSparseCM12SolidOccupancy(this.device.queue, this.topologyArena,
      this.solidOccupancyLayout, solidWorld, [0, 0, 0]);
    const refresh = this.device.createCommandEncoder({
      label: "Sparse CM12 refresh SolidWorld apertures",
    });
    this.encodeSolidWorldApertureRefresh(refresh);
    this.device.queue.submit([refresh.finish()]);
  }

  /**
   * The same compact occupancy arena used to derive fluid apertures. Moving
   * rigid bodies bind this view directly, so a voxel edit cannot leave contact
   * and flow consulting different static geometry.
   */
  get solidWorldCollisionSource(): Omit<GPURigidSolidWorldCollisionSource,
    "origin_m" | "cellSize_m"> | undefined {
    const layout = this.solidOccupancyLayout;
    if (!layout) return undefined;
    return {
      buffer: this.topologyArena,
      baseWords: layout.baseWords,
      directoryCapacity: layout.directoryCapacity,
      directoryBaseWords: layout.directoryBaseWords,
      regionCapacity: layout.regionCapacity,
      regionBaseWords: layout.regionBaseWords,
      regionWords: SPARSE_CM12_SOLID_REGION_WORDS,
      entryWords: SPARSE_CM12_SOLID_OCCUPANCY_ENTRY_WORDS,
      pageBaseWords: layout.pageBaseWords,
      pageWords: SPARSE_CM12_SOLID_OCCUPANCY_PAGE_WORDS,
      fractionPageWords: SPARSE_CM12_SOLID_FRACTION_PAGE_WORDS,
    };
  }

  /** Derive finite-volume apertures from the canonical GPU voxel occupancy. */
  private encodeSolidWorldApertureRefresh(encoder: GPUCommandEncoder): void {
    if (!this.solidOccupancyLayout || this.layout.solidVoxelCellOpen === 0) return;
    const pass = encoder.beginComputePass({
      label: "Sparse CM12 derive SolidWorld apertures",
    });
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines.refreshSparseCM12SolidWorldCells!);
    pass.dispatchWorkgroups(Math.ceil(this.cellCount / WORKGROUP_SIZE));
    pass.setPipeline(this.pipelines.refreshSparseCM12SolidWorldRows!);
    pass.dispatchWorkgroups(Math.ceil(this.rowCount / WORKGROUP_SIZE));
    pass.end();
  }

  /** Publish generation zero without executing a physics step or mapping state. */
  encodeInitialPresentation(encoder: GPUCommandEncoder, finestCellSize_m: number): void {
    this.assertLive();
    this.writeParameters(this.lastPacked!, 0.004, finestCellSize_m, 1, [0, 0, 0]);
    const pass = encoder.beginComputePass({ label: "Sparse CM12 resident initial presentation" });
    pass.setBindGroup(0, this.bindGroup);
    const brickWorkgroups = Math.ceil(this.lastPacked!.brickCount / WORKGROUP_SIZE);
    if (brickWorkgroups > 0) {
      pass.setPipeline(this.pipelines.classifyPresentationBricks!);
      pass.dispatchWorkgroups(brickWorkgroups);
    }
    pass.setPipeline(this.pipelines.validateSparseCM12InternedBoundaryImmutable!);
    pass.dispatchWorkgroups(1);
    pass.end();
    this.encodeFramePlanPresentation(encoder, "Sparse CM12 initial presentation");
  }

  encodeLiquidInjection(
    encoder: GPUCommandEncoder,
    finestCellSize_m: number,
    centerFine: readonly [number, number, number],
    radiusFine: readonly [number, number, number],
  ): void {
    this.encodeLiquidInjectionTransaction(encoder, finestCellSize_m, centerFine,
      radiusFine, 1, 0, 0.004, true);
  }

  encodeLiquidJetInjection(
    encoder: GPUCommandEncoder,
    finestCellSize_m: number,
    outletFine: readonly [number, number, number],
    radiusFine: number,
    velocityFinePerSecond: readonly [number, number, number],
    dt_s: number,
  ): void {
    const speed = Math.hypot(...velocityFinePerSecond);
    if (!(radiusFine > 0) || !(speed > 0) || !(dt_s > 0)) return;
    const halfDisplacement = velocityFinePerSecond.map((value) => 0.5 * value * dt_s) as
      [number, number, number];
    const center = outletFine.map((value, axis) => value + halfDisplacement[axis]!) as
      [number, number, number];
    this.encodeLiquidInjectionTransaction(encoder, finestCellSize_m, center,
      halfDisplacement, 2, radiusFine, dt_s, false);
  }

  private encodeLiquidInjectionTransaction(
    encoder: GPUCommandEncoder,
    finestCellSize_m: number,
    centerFine: readonly [number, number, number],
    radiusFine: readonly [number, number, number],
    mode: 1 | 2,
    jetRadiusFine: number,
    injectionDt_s: number,
    publishPresentation: boolean,
  ): void {
    this.assertLive();
    this.writeParameters(this.lastPacked!, injectionDt_s, finestCellSize_m, 1,
      [0, 0, 0], undefined, undefined, undefined, 0, undefined, this.lastInflow);
    // The trailing word is the injection mode that every ordinary frame writes
    // as zero. It lets the demand planner and writer distinguish an editor
    // ellipsoid from a swept hose plug at no cost to a quiescent frame.
    this.parameterF32.set([...centerFine, mode], 52);
    this.parameterF32.set([...radiusFine, jetRadiusFine], 56);
    this.device.queue.writeBuffer(this.parameters, 0, this.parameterWords);
    const packed = this.lastPacked!;
    const leafCapacity = this.worldDirectoryLayout.leafCapacity;
    const bricks = Math.ceil(leafCapacity / WORKGROUP_SIZE);
    const interactionPageCount = [0, 1, 2].map((axis) => {
      const lower = Math.floor((centerFine[axis]! - radiusFine[axis]!)
        / this.brickFineResolution);
      const upper = Math.floor((centerFine[axis]! + radiusFine[axis]!)
        / this.brickFineResolution);
      return Math.max(1, upper - lower + 1);
    }) as [number, number, number];
    let topologyPass: GPUComputePassEncoder | undefined;
    let topologyBindGroup = this.bindGroup;
    const openTopologyPass = () => {
      if (!topologyPass) {
        topologyPass = encoder.beginComputePass({
          label: "Sparse CM12 resident liquid injection topology",
        });
        topologyPass.setBindGroup(0, topologyBindGroup);
      }
      return topologyPass;
    };
    const closeTopologyPass = () => {
      topologyPass?.end();
      topologyPass = undefined;
    };
    const useTopologyBindGroup = (bindGroup: GPUBindGroup) => {
      topologyBindGroup = bindGroup;
      topologyPass?.setBindGroup(0, bindGroup);
    };
    const dispatchTopology = (name: string, count: number, y = 1, z = 1) => {
      const pass = openTopologyPass();
      pass.setPipeline(this.pipelines[name]!);
      pass.dispatchWorkgroups(count, y, z);
    };
    const dispatchTopologyIndirect = (name: string, byteOffset: number) => {
      const pass = openTopologyPass();
      pass.setPipeline(this.pipelines[name]!);
      pass.dispatchWorkgroupsIndirect(this.acceptedIndirectArguments, byteOffset);
    };
    const dispatchTopologyDelta = (name: string) => {
      dispatchTopologyIndirect(name, 72);
    };
    // Injection can be the first physics transaction after generation-zero
    // presentation. Open PTR1 before any tile lifecycle effects are recorded:
    // its zero-filled stamps otherwise alias candidate generation zero and make
    // the first newly populated tile look like an incompatible pending write.
    // `beginSparseCM12PressureTopologyRepair` is deliberately idempotent while
    // a journal is collecting, so an injection between ordinary frames keeps
    // the frame's already-recorded topology effects intact.
    dispatchTopology("beginSparseCM12PressureTopologyRepair", 1);
    dispatchTopology("allocateSparseWorldInteractionPages",
      Math.ceil(interactionPageCount[0] / 4),
      Math.ceil(interactionPageCount[1] / 4),
      Math.ceil(interactionPageCount[2] / 4));
    dispatchTopology("synthesizeSparseWorldFrontierPages", this.topologyPageCapacity);
    // Promote every intersected brick before writing any density. The planner
    // treats the enabled injection as refine-only: untouched accepted bricks
    // are preserved, while closure may still grow the required 2:1 support.
    dispatchTopology("planBrickResolution", bricks);
    dispatchTopology("activateInjectionFrontierPages", bricks);
    for (let gradingPass = 0;
      gradingPass < Math.log2(this.brickFineResolution); gradingPass += 1) {
      dispatchTopology("closePlannedResolution", bricks);
    }
    dispatchTopology("validateCandidateResolution", bricks);
    dispatchTopology("scheduleTopologyPreparation", 1);
    dispatchTopology("allocateCandidateTopologyPages", bricks);
    dispatchTopology("synthesizeCandidateCellPages",
      this.worldDirectoryLayout.initialLeaves);
    dispatchTopologyIndirect("clearShadowRowMembership", 36);
    dispatchTopology("beginShadowTopology", 1);
    dispatchTopology("buildShadowLeafWorklist", 1);
    closeTopologyPass();
    encoder.copyBufferToBuffer(this.topologyArena,
      this.acceptedLeafManifestBaseBytes + 4 * 15,
      this.acceptedIndirectArguments, 84, 12);
    dispatchTopologyIndirect("buildShadowStructureWorklist", 84);
    dispatchTopology("finalizeShadowWorklists", 1);
    closeTopologyPass();
    encoder.copyBufferToBuffer(this.topologyArena,
      this.topologyWorklistBaseBytes + 4 * 20,
      this.acceptedIndirectArguments, 24, 24);
    encoder.copyBufferToBuffer(this.topologyArena,
      this.acceptedLeafManifestBaseBytes + 4 * 7,
      this.acceptedIndirectArguments, 60, 12);
    encoder.copyBufferToBuffer(this.topologyArena,
      this.acceptedLeafManifestBaseBytes + 4 * 12,
      this.acceptedIndirectArguments, 72, 12);
    dispatchTopologyIndirect("transferCandidateCellsFromTopologyDelta", 72);
    dispatchTopologyDelta("transferCandidateFacesFromTopologyDelta");
    dispatchTopologyIndirect("validateCandidateShadowFaces", 36);
    dispatchTopology("beginSparseCM12TopologyEffectsPreflight", 1);
    dispatchTopologyDelta("recordCandidateTopologyEffectsFromTopologyDelta");
    dispatchTopology("finalizeSparseCM12TopologyEffectsPreflight", 1);
    useTopologyBindGroup(this.transportBindGroup);
      dispatchTopology("beginSparseCM12InternedBoundaryDelta", 1);
      dispatchTopologyDelta("compileSparseCM12InternedBoundaryDelta");
      dispatchTopology("finalizeSparseCM12ISAChangedSetReceipt", 1);
      dispatchTopologyDelta("validateSparseCM12InternedBoundaryDeltaPackets");
      dispatchTopology("finalizeSparseCM12InternedBoundaryDelta", 1);
      dispatchTopologyDelta("compileSparseCM12TransportExecutionImageShadow");
    useTopologyBindGroup(this.bindGroup);
    dispatchTopology("validateAndAuthorizeShadowTopology", 1);
      closeTopologyPass();
      encoder.copyBufferToBuffer(this.topologyArena,
        sparseCM12TopologyEffectsIndirectByteOffset(
          this.topologyEffectsAuthorityLayout!),
        this.frameControlIndirectArguments, 0, 12);
      const ptrPass = openTopologyPass();
      ptrPass.setPipeline(this.pipelines.publishSparseCM12TopologyPTREffects!);
      ptrPass.dispatchWorkgroupsIndirect(this.frameControlIndirectArguments, 0);
      dispatchTopology("sealSparseCM12AuthorizedTopologyEffects", 1);
    dispatchTopology("finishSparseCM12TopologyEffectsPublication", 1);
    useTopologyBindGroup(this.transportBindGroup);
    dispatchTopologyDelta("publishCandidateTopologyDeltaFromWorklist");
    useTopologyBindGroup(this.bindGroup);
    dispatchTopology("connectSparseWorldFrontierPages", this.topologyPageCapacity);
    dispatchTopologyIndirect("publishCandidateShadowFaces", 36);
    dispatchTopology("finalizeAuthorizedShadowTopology", 1);
    dispatchTopology("publishSparseWorldFrontierAcceptance", this.topologyPageCapacity);
    useTopologyBindGroup(this.transportBindGroup);
    dispatchTopology("compileSparseWorldFrontierExecutionImage", this.topologyPageCapacity);
    dispatchTopologyDelta("replaySparseCM12TransportExecutionImageRetired");
    dispatchTopologyDelta("replaySparseCM12InternedBoundaryDelta");
    useTopologyBindGroup(this.bindGroup);
    dispatchTopology("refreshSparseCM12FrontierSolidWorld",
      this.worldDirectoryLayout.leafCapacity);
    closeTopologyPass();
    encoder.copyBufferToBuffer(this.topologyArena,
      this.topologyWorklistBaseBytes + 4 * 8,
      this.acceptedIndirectArguments, 0, 6 * 4);
    encoder.copyBufferToBuffer(this.topologyArena,
      this.acceptedLeafManifestBaseBytes + 4 * 4,
      this.acceptedIndirectArguments, 48, 12);

    const injectionPass = encoder.beginComputePass({
      label: "Sparse CM12 resident liquid injection",
    });
    // Conservative transport samples the persistent effective-velocity plane
    // bound at slot 3, not the diagnostics scratch carried by the ordinary
    // bind group. Hose wetting publishes that authority alongside both state
    // velocity banks so the next frame actually advects the source plug.
    injectionPass.setBindGroup(0, this.effectiveVelocityPressureBindGroup);
    // Brick-indexed rather than indirect over the accepted cell worklist: this
    // also covers a newly activated frontier page in the same command buffer.
    injectionPass.setPipeline(this.pipelines.injectLiquid!);
    injectionPass.dispatchWorkgroups(bricks);
    const overflowSweeps = mode === 2 ? Math.min(16, Math.ceil(
      2 * Math.hypot(radiusFine[0], radiusFine[1], radiusFine[2]))) : 0;
    for (let sweep = 0; sweep < overflowSweeps; sweep += 1) {
      injectionPass.setPipeline(this.pipelines.clearLiquidJetOverflowReceipts!);
      injectionPass.dispatchWorkgroupsIndirect(this.acceptedIndirectArguments, 0);
      injectionPass.setPipeline(this.pipelines.scatterLiquidJetOverflow!);
      injectionPass.dispatchWorkgroupsIndirect(this.acceptedIndirectArguments, 0);
      injectionPass.setPipeline(this.pipelines.finalizeLiquidJetOverflow!);
      injectionPass.dispatchWorkgroupsIndirect(this.acceptedIndirectArguments, 0);
    }
    injectionPass.setPipeline(this.pipelines.injectLiquidFaces!);
    injectionPass.dispatchWorkgroupsIndirect(this.acceptedIndirectArguments, 12);
    // This is accepted-state occupancy, not merely renderer bookkeeping. A
    // post-step hose plug must publish its wet reason before the next frame's
    // topology planner runs, otherwise that planner can retire the just-wet
    // frontier page as a dry brick before transport sees the new mass.
    injectionPass.setPipeline(this.pipelines.classifyPresentationBricks!);
    injectionPass.dispatchWorkgroups(bricks);
    injectionPass.end();
    if (publishPresentation) {
      this.encodeFramePlanPresentation(encoder, "Sparse CM12 injection presentation");
    }
  }

  private lastPacked?: PackedResidentTopology;
  private lastInflow?: SparseCM12InflowControl;
  /**
   * Built on first access and never before.
   *
   * A method with eighteen declared lenses must cost a scene that opens none of
   * them nothing at all, and the source is where that promise is kept: until
   * someone asks for it there is no addressing uniform, no snapshot and no
   * readback slot.
   */
  private stageLenses?: SparseCM12StageLensSource;

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
    inflow?: SparseCM12InflowControl,
  ): void {
    this.lastPacked = packed;
    const u = this.parameterU32, f = this.parameterF32, l = this.layout;
    u.fill(0);
    u.set([this.cellCount, this.rowCount, packed.incidenceCount,
      this.dimensions[0] * this.dimensions[1] * this.dimensions[2]], 0);
    // dimensions.w retains the public sparse-consumer ABI: bit zero used to
    // advertise the retired analytic boundary, while the upper bits publish
    // twice the brick's fine resolution. Removing the old boundary flag must
    // not erase the resolution; grid overlays and FPL1 validation decode B8
    // from this lane without importing the resident's compile-time profile.
    u.set([...this.dimensions, this.brickFineResolution << 1], 4);
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
    u.set([l.sharpeningDelta, l.symmetryGamma, l.tracers,
      l.faceVelocitySupport], 36);
    f.set([dt_s, finestCellSize_m, pressureScale, 0], 40);
    f.set([...acceleration, 0], 44);
    u.set([Math.ceil(this.cellCount / WORKGROUP_SIZE),
      Math.ceil(this.rowCount / WORKGROUP_SIZE),
      sparseCM12PressureIterations(pressureControl?.iterations),
      this.worldDirectoryLayout.leafCapacity], 48);
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
    u.set([policy.prepareBricksPerFrame, 0,
      sharpening?.gammaDiffusionEnabled === false ? 0 : 1,
      sharpening?.surfaceSharpeningEnabled === false ? 0 : 1], 80);
    f[81] = sparseCM12PressureRelativeTolerance(pressureControl?.relativeTolerance);
    // bit 0: a static or dynamic cut-cell source is live; bit 2: SolidWorld
    // row openness is live. Authored vessel shells, terrain and edits all use
    // this single voxel authority. solidOffsets.w is reserved.
    const staticSolidWorld = Boolean(this.solidOccupancyLayout);
    const dynamicSolidWorld = bodyCount > 0 && l.solidRowData !== 0;
    u.set([l.solidCellOpen, l.solidRowData,
      (staticSolidWorld || dynamicSolidWorld ? 1 : 0)
        | (staticSolidWorld ? 4 : 0),
      0], 84);
    f.set([...(worldDimensions_m ?? [0, 0, 0]), bodyCount], 88);
    u.set([...this.tracerLattice.dimensions, this.tracerLattice.count], 92);
    f.set([...this.tracerLattice.originFine, this.tracerLattice.spacingFine], 96);
    const journal = this.layout.journalLayout;
    u.set([this.layout.journal, journal.iterationCapacity, journal.snapshotCapacity,
      journal.cellStride], 100);
    f.set(inflow ? [...inflow.outletFine, inflow.radiusFine] : [0, 0, 0, 0], 104);
    f.set(inflow ? [...inflow.velocityFinePerSecond, 1] : [0, 0, 0, 0], 108);
    new Uint8Array(this.parameterWords).set(this.refinementRegionParameters,
      SPARSE_CM12_REFINEMENT_REGION_PARAMETER_OFFSET);
    this.device.queue.writeBuffer(this.parameters, 0, this.parameterWords);
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
        readonly dirtyCount: number; readonly directWriteCount: number;
        readonly totalCount: number;
        readonly candidateGeneration: number; readonly acceptedGeneration: number;
      };
      readonly row: {
        readonly phase: number; readonly fault: number; readonly firstFault: number;
        readonly dirtyCount: number; readonly directWriteCount: number;
        readonly totalCount: number;
        readonly candidateGeneration: number; readonly acceptedGeneration: number;
      };
    };
    readonly pressureTopologyRepair: {
      readonly phase: number; readonly fault: number;
      readonly firstFaultFamily: number; readonly firstFaultId: number;
      readonly candidateGeneration: number; readonly acceptedGeneration: number;
      readonly topologyGeneration: number;
      readonly changedBrickCount: number; readonly cellExecutionCount: number;
      readonly brickDirtyLeafCount: number;
      readonly expectedProducerReceipts: number;
      readonly coveredProducerReceipts: number;
    };
    readonly pressureCutoverAuthorities: NonNullable<NonNullable<
      GPUEulerianInfo["adaptivePressureTopologyAttribution"]>["authorities"]>;
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
    const pcfDiagnosticOffset = ptrDiagnosticOffset
      + 4 * SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS;
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
    const pcfHeader = new Uint32Array(mapped, pcfDiagnosticOffset,
      SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS);
    const pcaHeader = new Uint32Array(mapped, pcaDiagnosticOffset,
      SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS);
    const pcaFamilies = persistentFamilies.map((_, index) => new Uint32Array(mapped,
      pcaDiagnosticOffset + 4 * (SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS
        + index * SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS),
      SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS));
    const pcmDomainReceipt = (words: Uint32Array) => {
      const h = SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER;
      return {
        phase: words[h.phase]!, fault: words[h.fault]!,
        firstFault: words[h.firstFaultId]!, dirtyCount: words[h.dirtyCount]!,
        directWriteCount: words[h.directWriteCount]!,
        totalCount: words[h.totalCount]!,
        candidateGeneration: words[h.candidateGeneration]!,
        acceptedGeneration: words[h.acceptedGeneration]!,
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
    const pressureCacheFault = pcfHeader[cache.fault]!;
    const pressureAggregateReceipt = {
      acceptedGeneration: pcfHeader[cache.acceptedGeneration]!,
      candidateGeneration: pcfHeader[cache.candidateGeneration]!,
      topologyGeneration: pressureCacheTopologyGeneration,
      directCount: 0, closureCount: 0,
      dirtyCount: sum(pcaDirty), workCount: sum(pcaWork),
      executedCount: sum(pcaExecuted),
      skippedCount: Math.max(0, sum(pcaWork) - sum(pcaExecuted)),
      expectedProducerReceipts: 0, coveredProducerReceipts: 0,
      causeMask: pcaCauses, fault: pressureCacheFault,
      firstFaultId: pressureCacheFault === 0
        ? 0xffff_ffff : pcfHeader[cache.firstFaultId]!,
      familyDirtyCount: pcaDirty as [number, number, number, number],
      familyExecutedCount: pcaExecuted as [number, number, number, number],
    };
    const pressureCutoverFault = pressureCacheFault !== 0;
    const pressureCutoverAuthorities = {
      status: pressureCutoverFault ? "fault" as const : "matched" as const,
      inputTopologyGeneration: pressureCacheTopologyGeneration,
      pca: pressureAggregateReceipt,
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
    const fixedBudget = this.parameterF32[81]! <= 0;
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
        cellExecutionCount: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.acceptedCellExecutionCount]!,
        brickDirtyLeafCount: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.acceptedBrickDirtyLeafCount]!,
        expectedProducerReceipts: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.expectedProducerReceipts]!,
        coveredProducerReceipts: ptrHeader[
          SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER.coveredProducerReceipts]!,
      },
      pressureCutoverAuthorities,
    };
    this.diagnosticsReadback.unmap();
    return result;
  }

  /**
   * Copy only the executed-iteration scalar into a caller-owned staging buffer.
   *
   * The live SIM panel samples this beside its timestamp capture. It deliberately
   * does not use `readDiagnostics`: that larger paired receipt is a pause-time
   * instrument, while this four-byte copy neither schedules physics nor fences
   * the host during a running frame.
   */
  encodePressureIterationReceipt(
    encoder: GPUCommandEncoder,
    destination: GPUBuffer,
  ): void {
    this.assertLive();
    encoder.copyBufferToBuffer(this.scalars, 12 * 4, destination, 0, 4);
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
    const faceAAt = rhsAt + cell.capacity;
    const faceBAt = faceAAt + row.capacity;
    const aggregateEdgeAt = faceBAt + row.capacity;
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
    // Hash the same ordinary coefficient image consumed by production SpMV.
    encoder.copyBufferToBuffer(this.candidateState,
      4 * this.pressureFineEdgeImageBaseWords, readback,
      4 * coefficientAt, 4 * this.pressureFineEdgeCount);
    encoder.copyBufferToBuffer(this.state, 4 * this.layout.rhs, readback,
      4 * rhsAt, 4 * cell.capacity);
    encoder.copyBufferToBuffer(this.state, 4 * this.layout.faceA, readback,
      4 * faceAAt, 4 * row.capacity);
    encoder.copyBufferToBuffer(this.state, 4 * this.layout.faceB, readback,
      4 * faceBAt, 4 * row.capacity);
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
          directWriteCount: header[h.directWriteCount]!,
          directCauseMask: header[h.directCauseMask]!,
          conflictPacket: header[h.flags]!,
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
        aggregateEdgeSha256, brickDiagonalSha256,
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
          sha256(mapped.slice(aggregateEdgeAt, brickDiagonalAt)),
          sha256(mapped.slice(brickDiagonalAt, hierarchyEdgeAt)),
          sha256(mapped.slice(hierarchyEdgeAt, hierarchyDiagonalAt)),
          sha256(mapped.slice(hierarchyDiagonalAt, wordCount)),
          sha256(mapped.slice(rowClassificationAt, rowClassificationAt + row.capacity)),
          sha256(mapped.slice(coefficientAt, coefficientAt + this.pressureFineEdgeCount)),
          sha256(mapped.slice(rhsAt, rhsAt + cell.capacity)),
        ]);
      const result = {
        mode: "local" as const,
        cell: cellReceipt,
        row: rowReceipt,
        thetaSha256,
        coefficientSha256,
        rhsSha256,
        aggregateEdgeSha256,
        brickDiagonalSha256,
        hierarchyEdgeSha256,
        hierarchyDiagonalSha256,
        rawThetaSha256,
        rawCoefficientSha256,
        rawRhsSha256,
      };
      Object.defineProperty(result, "qaRaw", { enumerable: false, value: {
        coefficientBits: mapped.slice(coefficientAt,
          coefficientAt + this.pressureFineEdgeCount),
        faceABits: mapped.slice(faceAAt, faceAAt + row.capacity),
        faceBBits: mapped.slice(faceBAt, faceBAt + row.capacity),
        aggregateEdgeBits: mapped.slice(aggregateEdgeAt, brickDiagonalAt),
        brickDiagonalBits: mapped.slice(brickDiagonalAt, hierarchyEdgeAt),
        hierarchyEdgeBits: mapped.slice(hierarchyEdgeAt, hierarchyDiagonalAt),
        hierarchyDiagonalBits: mapped.slice(hierarchyDiagonalAt, wordCount),
      } });
      return result as typeof result & { readonly qaRaw: {
        readonly coefficientBits: Uint32Array;
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

  async readDiagnosticFields(
    includeWorldLeaves = false,
  ): Promise<SparseCM12DiagnosticFields> {
    this.assertLive();
    const activitySnapshot = await this.readActivitySnapshot(includeWorldLeaves);
    const readback = this.device.createBuffer({
      label: "Sparse CM12 QA field readback",
      size: this.state.size + 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 QA field copy",
      });
      encoder.copyBufferToBuffer(this.state, 0, readback, 0, this.state.size);
      encoder.copyBufferToBuffer(this.topologyArena,
        4 * (this.frameControlLayout.baseWords
          + SPARSE_CM12_FRAME_CONTROL_HEADER.scalarParity),
        readback, this.state.size, 4);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const state = new Float32Array(readback.getMappedRange());
      const acceptedParity = new Uint32Array(state.buffer,
        state.byteOffset + this.state.size, 1)[0]! & 1;
      const [nx, ny, nz] = this.dimensions;
      const count = nx * ny * nz;
      const density = new Float32Array(count);
      const gamma = new Float32Array(count);
      const solidOpenFraction = new Float32Array(count); solidOpenFraction.fill(1);
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
      for (const record of activitySnapshot.records) {
        const brick = record.leafId;
        if (brick >= this.lastPacked!.brickCount) continue;
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
        const openFraction = this.layout.solidCellOpen !== 0
          ? state[this.layout.solidCellOpen + cell]!
            * (this.layout.solidVoxelCellOpen !== 0
              ? state[this.layout.solidVoxelCellOpen + cell]!
              : 1)
          : 1;
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
              solidOpenFraction[at] = openFraction;
              velocity[4 * at] = vx; velocity[4 * at + 1] = vy;
              velocity[4 * at + 2] = vz;
              // Publish pressure over the accepted liquid phase. PCM is an
              // iteration-local solve worklist and may differ for one frame
              // across an otherwise identical topology transition; using it
              // as a display/QA mask exposed a false asymmetric zero. Dry
              // slots remain canonical zero, while the resident D4 authority
              // has already cleared/averaged every accepted active slot.
              pressure[at] = rho >= 0.5 ? mappedPressure : 0;
              divergence[at] = div;
            }
        }
      }
      // GPU-grown leaves live in the fixed B8 suffix of the same state planes,
      // not in the immutable host-template catalog above. Materialize them by
      // their signed WDR coordinates so diagnostic fields and correctness
      // oracles observe the complete accepted world rather than silently
      // clipping back to the authored seed atlas.
      if (includeWorldLeaves) {
        const cellsPerPage = this.brickFineResolution ** 3;
        const dynamicCellOffset = this.templateCellCount
          - cellsPerPage * this.topologyPageCapacity;
        for (const record of activitySnapshot.records) {
          if (record.leafId < this.initialWorldLeafCount || !record.active
            || record.topologyPage === undefined || !record.coordinate) continue;
          const first = dynamicCellOffset + record.topologyPage * cellsPerPage;
          const origin = record.coordinate.map((value) =>
            value * this.brickFineResolution) as [number, number, number];
          for (let local = 0; local < cellsPerPage; local += 1) {
            const z = Math.floor(local / (this.brickFineResolution ** 2));
            const yz = local - z * this.brickFineResolution ** 2;
            const y = Math.floor(yz / this.brickFineResolution);
            const x = yz - y * this.brickFineResolution;
            const q = [origin[0] + x, origin[1] + y, origin[2] + z] as const;
            if (q[0] < 0 || q[0] >= nx || q[1] < 0 || q[1] >= ny
              || q[2] < 0 || q[2] >= nz) continue;
            const cell = first + local;
            const at = q[0] + nx * (q[1] + ny * q[2]);
            const rho = state[densityOffset + cell]!;
            const gammaValue = state[gammaOffset + cell]!;
            const openFraction = this.layout.solidCellOpen !== 0
              ? state[this.layout.solidCellOpen + cell]!
                * (this.layout.solidVoxelCellOpen !== 0
                  ? state[this.layout.solidVoxelCellOpen + cell]! : 1)
              : 1;
            const velocityAt = velocityOffset + 4 * cell;
            density[at] = rho; gamma[at] = gammaValue;
            solidOpenFraction[at] = openFraction;
            velocity[4 * at] = state[velocityAt]! * cellWidth_m;
            velocity[4 * at + 1] = state[velocityAt + 1]! * cellWidth_m;
            velocity[4 * at + 2] = state[velocityAt + 2]! * cellWidth_m;
            pressure[at] = rho >= 0.5
              ? state[this.layout.pressure + cell]! * pressureScale : 0;
            divergence[at] = state[this.layout.divergence + cell]!;
          }
        }
      }
      return { density, gamma, solidOpenFraction, velocity, pressure, divergence };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /**
   * QA-only raw Phase-1 transport receipt. Every physical array below is
   * written by the producer/consumer pass that owns the value, indexed by the
   * stable accepted cell id. Production and timed experiment constructions do
   * not reserve this arena and reject the read.
   */
  async readPhase1TransportReceiptQA(): Promise<SparseCM12Phase1TransportReceipt> {
    this.assertLive();
    const layout = this.phase1TransportQALayout;
    if (!layout) throw new Error(
      "Phase-1 transport receipt requested from a non-QA resident",
    );
    const receiptWords = layout.totalWords - layout.baseWords;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 Phase-1 raw transport QA readback",
      size: 4 * receiptWords,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    let raw: Uint32Array;
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 Phase-1 raw transport QA copy",
      });
      encoder.copyBufferToBuffer(this.activity, 4 * layout.baseWords,
        readback, 0, 4 * receiptWords);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      raw = new Uint32Array(readback.getMappedRange()).slice();
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
    const h = SPARSE_CM12_PHASE1_TRANSPORT_QA_HEADER;
    if (raw[h.magic] !== SPARSE_CM12_PHASE1_TRANSPORT_QA_MAGIC
      || raw[h.version] !== SPARSE_CM12_PHASE1_TRANSPORT_QA_VERSION
      || raw[h.cellCapacity] !== layout.cellCapacity) {
      throw new Error("Sparse CM12 Phase-1 raw transport QA header is incompatible");
    }
    const relative = (absolute: number) => absolute - layout.baseWords;
    const plane = (base: number, count: number) => raw.slice(relative(base),
      relative(base) + count);
    const capacity = layout.cellCapacity;
    const departure = plane(layout.departureBaseWords, 3 * capacity);
    const stencilCells = plane(layout.stencilCellBaseWords, 8 * capacity);
    const stencilWeights = plane(layout.stencilWeightBaseWords, 8 * capacity);
    const beta = plane(layout.betaBaseWords, capacity);
    const deficitDensity = plane(layout.deficitDensityBaseWords, capacity);
    const deficitGamma = plane(layout.deficitGammaBaseWords, capacity);
    const massDensity = plane(layout.massDensityBaseWords, capacity);
    const massGamma = plane(layout.massGammaBaseWords, capacity);
    const packetIds = plane(layout.packetIdBaseWords, capacity);
    const packetLanes = plane(layout.packetLaneBaseWords, capacity);
    const mass = new Uint32Array(2 * capacity);
    mass.set(massDensity); mass.set(massGamma, capacity);
    const transportPackets = new Uint32Array(2 * capacity);
    transportPackets.set(packetIds); transportPackets.set(packetLanes, capacity);
    const betaSigned = new Int32Array(beta.buffer, beta.byteOffset, beta.length);
    const deficitDensitySigned = new Int32Array(deficitDensity.buffer,
      deficitDensity.byteOffset, deficitDensity.length);
    let minimumBetaFixed = 0;
    let maximumBetaFixed = 0;
    let minimumDeficitDensityFixed = 0;
    let maximumDeficitDensityFixed = 0;
    for (let cell = 0; cell < capacity; cell += 1) {
      minimumBetaFixed = Math.min(minimumBetaFixed, betaSigned[cell]!);
      maximumBetaFixed = Math.max(maximumBetaFixed, betaSigned[cell]!);
      minimumDeficitDensityFixed = Math.min(
        minimumDeficitDensityFixed, deficitDensitySigned[cell]!,
      );
      maximumDeficitDensityFixed = Math.max(
        maximumDeficitDensityFixed, deficitDensitySigned[cell]!,
      );
    }

    const [activity, scalarHeader, frameControl] = await Promise.all([
      this.readActivitySnapshot(), this.readFinalScalarMaskHeaderQA(),
      this.readFrameControlQA(),
    ]);
    const transportTopologyGeneration = scalarHeader.topologyGeneration;
    const topologyGeneration = activity.acceptedTopologyGeneration;
    const frameGeneration = frameControl.acceptedGeneration;
    const capturedFrame = raw[h.frameGeneration]!;
    const capturedTopology = raw[h.topologyGeneration]!;
    if (capturedFrame !== 0 && capturedFrame !== frameGeneration) {
      throw new Error(`Phase-1 trace frame ${capturedFrame} != FCA1 ${frameGeneration}`);
    }
    if (capturedTopology !== 0 && capturedTopology !== transportTopologyGeneration) {
      throw new Error(`Phase-1 trace topology ${capturedTopology} != FSM1 ${
        transportTopologyGeneration}`);
    }

    const transportPacketSet = new Set<string>();
    const transportPacketIds = new Set<number>();
    const transportPacketMasks = new Map<number, [number, number]>();
    let transportPacketCellCount = 0;
    let duplicateScatterCellCount = 0;
    for (let cell = 0; cell < capacity; cell += 1) {
      const laneMarker = packetLanes[cell]!;
      if (laneMarker === 0) continue;
      transportPacketCellCount += 1;
      const packetId = packetIds[cell]!;
      const lane = laneMarker - 1;
      const key = `${packetId}:${lane}`;
      if (transportPacketSet.has(key)) duplicateScatterCellCount += 1;
      transportPacketSet.add(key); transportPacketIds.add(packetId);
      const mask = transportPacketMasks.get(packetId) ?? [0, 0];
      if (lane < 32) mask[0] = (mask[0] | (1 << lane)) >>> 0;
      else mask[1] = (mask[1] | (1 << (lane - 32))) >>> 0;
      transportPacketMasks.set(packetId, mask);
    }
    const capturedPacketCellCount = raw[h.packetCellCount]!;
    const omittedAcceptedCellCount = Math.max(0,
      capturedPacketCellCount - transportPacketCellCount);
    if (raw[h.packetFault] !== 0) throw new Error(
      `Phase-1 transport packet snapshot has ${raw[h.packetFault]} TEI mismatches; `
      + `cell ${raw[13]} resolved ${raw[14]} through packed ${raw[15]}`,
    );

    // Canonical rung-major packet identity is built independently for both
    // QA arms from the accepted leaves. The AEI arm below must additionally
    // prove that its live TEI and frame authority encode these same packets.
    const packetRecords: number[] = [];
    const canonical = new Map<number, readonly [number, number, number, number, number]>();
    const seenCells = new Uint8Array(capacity);
    let publishedDuplicateCellCount = 0;
    let acceptedCellCount = 0;
    const levels = sparseCM12TemplateLevels(this.brickFineResolution);
    const rangeOffset = this.templateWords[11]!;
    const brickDimensions = this.dimensions.map((value) =>
      Math.ceil(value / this.brickFineResolution));
    for (let brick = 0; brick < activity.records.length; brick += 1) {
      const record = activity.records[brick]!;
      if (!record.active) continue;
      const level = Math.log2(record.acceptedResolution);
      const rangeAt = rangeOffset + 2 * (levels.length * brick + level);
      const firstRange = this.templateWords[rangeAt]!;
      const rangeCount = this.templateWords[rangeAt + 1]!;
      acceptedCellCount += rangeCount;
      const packedAt = this.lastPacked!.brickOffset + 2 * brick;
      const key = this.lastPacked!.words[packedAt + 1]!;
      const spanBricks = 1 << (this.lastPacked!.words[packedAt]! & 31);
      const z = Math.floor(key / (brickDimensions[0]! * brickDimensions[1]!));
      const rem = key - z * brickDimensions[0]! * brickDimensions[1]!;
      const y = Math.floor(rem / brickDimensions[0]!);
      const x = rem - y * brickDimensions[0]!;
      const origin = [x, y, z].map((value) => value * this.brickFineResolution);
      const spanFine = this.brickFineResolution * spanBricks;
      const scale = spanFine / record.acceptedResolution;
      const valid = origin.map((value, axis) => Math.ceil(Math.max(0, Math.min(
        spanFine, this.dimensions[axis]! - value,
      )) / scale));
      const packetAxis = Math.max(1, Math.ceil(record.acceptedResolution / 4));
      for (let localPacket = 0; localPacket < packetAxis ** 3; localPacket += 1) {
        const pz = Math.floor(localPacket / (packetAxis ** 2));
        const packetRem = localPacket - pz * packetAxis ** 2;
        const py = Math.floor(packetRem / packetAxis);
        const px = packetRem - py * packetAxis;
        const local = [4 * px, 4 * py, 4 * pz];
        const counts = local.map((value, axis) => Math.max(0,
          Math.min(4, valid[axis]! - value)));
        if (counts.some((value) => value === 0)) continue;
        const first = firstRange + local[0]! + valid[0]!
          * (local[1]! + valid[1]! * local[2]!);
        const packetId = 64 * brick + localPacket;
        const countsWord = (counts[0]! | (counts[1]! << 5)
          | (counts[2]! << 10) | 0x8000_0000) >>> 0;
        const strides = (valid[0]! | ((valid[0]! * valid[1]!) << 16)) >>> 0;
        let maskLow = 0, maskHigh = 0;
        for (let lz = 0; lz < counts[2]!; lz += 1) {
          for (let ly = 0; ly < counts[1]!; ly += 1) {
            for (let lx = 0; lx < counts[0]!; lx += 1) {
              const lane = lx + 4 * (ly + 4 * lz);
              if (lane < 32) maskLow = (maskLow | (1 << lane)) >>> 0;
              else maskHigh = (maskHigh | (1 << (lane - 32))) >>> 0;
              const cell = first + lx + valid[0]! * (ly + valid[1]! * lz);
              if (cell >= capacity || cell >= firstRange + rangeCount) {
                throw new Error(`Phase-1 packet ${packetId} contains invalid cell ${cell}`);
              }
              if (seenCells[cell] !== 0) publishedDuplicateCellCount += 1;
              seenCells[cell] = 1;
            }
          }
        }
        const descriptor = [first, countsWord, strides, maskLow, maskHigh] as const;
        canonical.set(packetId, descriptor);
        packetRecords.push(packetId, ...descriptor);
      }
    }
    let packetCellCount = 0;
    for (const present of seenCells) packetCellCount += present;
    const publishedOmittedCellCount = Math.max(0, acceptedCellCount - packetCellCount);
    if (publishedDuplicateCellCount !== 0 || publishedOmittedCellCount !== 0) {
      throw new Error(`Phase-1 published packets duplicate/omit ${
        publishedDuplicateCellCount}/${publishedOmittedCellCount} accepted cells`);
    }

    let publishedPacketGeneration = topologyGeneration;
    if (this.transportExecutionImage && this.transportExecutionImageLayout
      && this.transportPacketAuthorityLayout) {
      const imageLayout = this.transportExecutionImageLayout;
      const authorityLayout = this.transportPacketAuthorityLayout;
      const imageWords = imageLayout.totalWords;
      const authorityWords = authorityLayout.totalWords - authorityLayout.baseWords;
      const validationReadback = this.device.createBuffer({
        label: "Sparse CM12 Phase-1 TEI/packet-authority QA readback",
        size: 4 * (imageWords + authorityWords + 1),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = this.device.createCommandEncoder({
          label: "Sparse CM12 Phase-1 TEI/packet-authority QA copy",
        });
        encoder.copyBufferToBuffer(this.transportExecutionImage, 0,
          validationReadback, 0, 4 * imageWords);
        encoder.copyBufferToBuffer(this.activity, 4 * authorityLayout.baseWords,
          validationReadback, 4 * imageWords, 4 * authorityWords);
        encoder.copyBufferToBuffer(this.topologyArena,
          this.topologyWorklistBaseBytes + 8, validationReadback,
          4 * (imageWords + authorityWords), 4);
        this.device.queue.submit([encoder.finish()]);
        await validationReadback.mapAsync(GPUMapMode.READ);
        const validation = new Uint32Array(validationReadback.getMappedRange());
        const slot = validation[imageWords + authorityWords]! & 1;
        const slotBase = imageLayout.slotBaseWords[slot]!;
        const livePacketGeneration = validation[slotBase]!;
        if (topologyGeneration !== 0 && livePacketGeneration !== topologyGeneration) {
          throw new Error(
            `Phase-1 TEI generation ${livePacketGeneration} != FSM1 ${topologyGeneration}`,
          );
        }
        // Generation zero is the pre-physics checkpoint: the TEI is already
        // seeded for the first frame, while FSM1 has not published scalar
        // result yet. No transport pass consumes that construction receipt.
        publishedPacketGeneration = topologyGeneration === 0 ? 0 : livePacketGeneration;
        const packetBase = imageLayout.slotPacketBaseOffsets[slot]!;
        for (const [packetId, expected] of canonical) {
          const at = packetBase
            + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS * packetId;
          const actual = Array.from(validation.slice(at + 1, at + 4));
          if (actual[0] !== expected[0] || actual[1] !== expected[1]
            || actual[2] !== expected[2]) {
            throw new Error(`Phase-1 TEI packet ${packetId} differs from accepted rung`);
          }
        }
        const authorityBase = (absolute: number) => absolute - authorityLayout.baseWords;
        const authority = validation.subarray(imageWords, imageWords + authorityWords);
        const indirect = authorityBase(authorityLayout.indirectBaseWords);
        const authorityCount = Math.min(authority[indirect]!,
          authorityLayout.dispatchPacketCount);
        const listed = new Set<number>();
        for (let rank = 0; rank < authorityCount; rank += 1) {
          const ordinal = authority[authorityBase(
            authorityLayout.packetListBaseWords) + rank]!;
          const leaf = Math.floor(ordinal / authorityLayout.dispatchPacketsPerLeaf);
          const local = ordinal - leaf * authorityLayout.dispatchPacketsPerLeaf;
          const packetId = 64 * leaf + local;
          const expectedMask = transportPacketMasks.get(packetId);
          if (ordinal >= authorityLayout.dispatchPacketCount || !expectedMask
            || listed.has(packetId)) {
            throw new Error(
              `Phase-1 packet authority has invalid/duplicate packet ${packetId}`,
            );
          }
          listed.add(packetId);
          const low = authority[authorityBase(
            authorityLayout.transportMaskLowBaseWords) + ordinal]!;
          const high = authority[authorityBase(
            authorityLayout.transportMaskHighBaseWords) + ordinal]!;
          if ((low & ~expectedMask[0]) !== 0 || (high & ~expectedMask[1]) !== 0
            || (low === 0 && high === 0)) {
            throw new Error(`Phase-1 packet authority mask escapes packet ${packetId}`);
          }
        }
      } finally {
        if (validationReadback.mapState === "mapped") validationReadback.unmap();
        validationReadback.destroy();
      }
    }

    const effectiveFrame = frameGeneration === 0 ? 0
      : raw[h.effectiveVelocityFrameGeneration]! || frameGeneration;
    const effectiveTopology = transportTopologyGeneration === 0 ? 0
      : raw[h.effectiveVelocityTopologyGeneration]! || transportTopologyGeneration;
    if (effectiveFrame !== frameGeneration
      || effectiveTopology !== transportTopologyGeneration) {
      throw new Error(`Phase-1 effective velocity provenance ${effectiveFrame}/${
        effectiveTopology} != frame/transport-topology ${frameGeneration}/${
        transportTopologyGeneration}`);
    }
    const [stencilCellsSha256, stencilWeightBitsSha256, departureBitsSha256,
      betaFixedSha256, deficitDensityFixedSha256, deficitGammaFixedSha256,
      massReceiptSha256, packetSha256, publishedPacketSha256] = await Promise.all([
      sparseCM12Phase1Sha256(stencilCells), sparseCM12Phase1Sha256(stencilWeights),
      sparseCM12Phase1Sha256(departure), sparseCM12Phase1Sha256(beta),
      sparseCM12Phase1Sha256(deficitDensity), sparseCM12Phase1Sha256(deficitGamma),
      sparseCM12Phase1Sha256(mass),
      sparseCM12Phase1Sha256(transportPackets),
      sparseCM12Phase1Sha256(Uint32Array.from(packetRecords)),
    ]);
    return Object.freeze({ stencilCellsSha256, stencilWeightBitsSha256,
      departureBitsSha256, betaFixedSha256, deficitDensityFixedSha256,
      deficitGammaFixedSha256, massReceiptSha256, packetSha256,
      publishedPacketSha256,
      packetCount: transportPacketIds.size, packetCellCount: transportPacketCellCount,
      duplicateScatterCellCount, omittedAcceptedCellCount,
      minimumBetaFixed, maximumBetaFixed,
      minimumDeficitDensityFixed, maximumDeficitDensityFixed,
      acceptedTopologyGeneration: topologyGeneration, transportTopologyGeneration,
      frameGeneration, packetGeneration: transportTopologyGeneration,
      publishedPacketGeneration,
      effectiveVelocityTopologyGeneration: effectiveTopology,
      effectiveVelocityFrameGeneration: effectiveFrame });
  }

  async readFramePlanPresentationFaultRecordQA() {
    this.assertLive();
    const header = await this.readFramePlanPresentationHeaderQA() as Record<string, number>;
    const brick = header.firstFaultBrick;
    if (brick === undefined || brick >= this.framePlanPresentationLayout.brickCapacity) {
      return undefined;
    }
    const readback = this.device.createBuffer({
      label: "Sparse CM12 FPP1 fault-record QA readback",
      size: 4 * SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 FPP1 fault-record QA copy",
      });
      encoder.copyBufferToBuffer(this.activity,
        4 * (this.framePlanPresentationLayout.recordsBaseWords
          + SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS * brick), readback, 0,
        4 * SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return Object.freeze({ brick, ...Object.fromEntries(Object.entries(
        SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE).map(([name, word]) =>
        [name, words[word]!])) });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Explicit QA receipt only; no production frame decision reads this back. */
  async readPhase1TransportProfileQA() {
    this.assertLive();
    const base = this.phase1TransportProfileBaseWords;
    if (base === undefined) return undefined;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 Phase-1 transport profile readback",
      size: 4 * SPARSE_CM12_PHASE1_TRANSPORT_PROFILE_WORDS,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 Phase-1 transport profile copy",
      });
      encoder.copyBufferToBuffer(this.activity, 4 * base, readback, 0,
        4 * SPARSE_CM12_PHASE1_TRANSPORT_PROFILE_WORDS);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = Uint32Array.from(new Uint32Array(readback.getMappedRange()));
      const families = ["trace", "scatter", "gather"] as const;
      return Object.freeze({
        families: Object.fromEntries(families.map((family, index) => [family, {
          packets: words[index]!, selectedLanes: words[3 + index]!,
          stagedSites: words[6 + index]!, residentStagedSites: words[9 + index]!,
          stageCalls: words[45 + index]!,
          radiusPackets: Array.from(words.slice(12 + 4 * index, 16 + 4 * index)),
          rungPackets: Array.from(words.slice(24 + 5 * index, 29 + 5 * index)),
        }])),
        lookups: {
          total: words[39]!, cacheHits: words[40]!, globalFallbacks: words[41]!,
          seamFallbacks: words[42]!, outsideStagedRadius: words[43]!,
          invalidFallbacks: words[44]!,
        },
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Explicit QA receipt only; no production frame decision reads this readback. */
  readWorkShapeQA() {
    this.assertLive();
    const logicalBrickDimensions = this.dimensions.map((value) =>
      Math.ceil(value / this.brickFineResolution));
    const logicalBrickCount = logicalBrickDimensions[0]!
      * logicalBrickDimensions[1]! * logicalBrickDimensions[2]!;
    return {
      finestDomainCellCount: this.dimensions[0] * this.dimensions[1] * this.dimensions[2],
      logicalBrickDimensions,
      logicalBrickCount,
      packedBrickCount: this.lastPacked?.brickCount ?? 0,
      templateCellCount: this.templateCellCount,
      templateRowCount: this.templateRowCount,
      maximumOwnedRowCount: this.maximumOwnedRowCount,
      templateCellWorkgroups: Math.ceil(this.templateCellCount / WORKGROUP_SIZE),
      templateRowWorkgroups: Math.ceil(this.templateRowCount / WORKGROUP_SIZE),
      conditioningClearBytesPerFrame: 0,
      conditioningClearBytesPerAcceptedCell: 24,
      pressureScratchClearBytesPerFrame: 0,
      rowOwnershipCatalogBytes: 4 * (this.templateWords[24]! - this.templateWords[16]!),
      gammaPairCatalogBytes: 0,
      candidateFaceCatalogBytes: 4 * (this.templateWords[15]! - this.templateWords[24]!),
      acceptedRowMembershipBytes:
        4 * this.canonicalMembershipLayout.row.activeBitWordCount,
      massDepartureCacheCapacityBytes: 56 * this.templateCellCount,
      pressureHierarchyGroupCount: this.pressureHierarchyGroupCount,
      pressureHierarchyEdgeCount: this.pressureHierarchyEdgeCount,
      pressureFineEdgeCount: this.pressureFineEdgeCount,
      pressureCoarseEdgeCount: this.pressureCoarseEdgeCount,
      transportSpatialTileCapacity: this.transportExecutionImageLayout?.spatialTileCapacity ?? 0,
      facePreparationLeafCount: 0,
      facePreparationMode: "brick-owned" as const,
      faceAddressProgramBytes: this.faceAddressLayout.totalBytes,
      faceAddressInteriorTileCount: this.faceAddressLayout.interiorTileCount,
      faceAddressSeamCount: this.faceAddressLayout.seamAddressCount,
      faceAddressPackedSeamBytes: 4 * this.faceAddressLayout.seamAddressCount,
      faceAddressPackingSavedBytes: 4 * this.faceAddressLayout.seamAddressCount,
      presentationPageCount: this.lastPacked?.brickCount ?? 0,
      allocatedBytes: this.allocatedBytes,
    };
  }

  /** Proves which pieces can share one accepted adaptive-topology generation.
   * This is a terminal QA readback only; frame scheduling never consumes it. */
  async readAdaptiveRepresentationQA() {
    this.assertLive();
    const headerWords = 32;
    const headerReadback = this.device.createBuffer({
      label: "Sparse CM12 accepted-topology header QA",
      size: 4 * headerWords,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    let header: Uint32Array;
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 accepted-topology header QA copy",
      });
      encoder.copyBufferToBuffer(this.topologyArena, this.topologyWorklistBaseBytes,
        headerReadback, 0, 4 * headerWords);
      this.device.queue.submit([encoder.finish()]);
      await headerReadback.mapAsync(GPUMapMode.READ);
      header = new Uint32Array(headerReadback.getMappedRange()).slice();
    } finally {
      if (headerReadback.mapState === "mapped") headerReadback.unmap();
      headerReadback.destroy();
    }
    const slot = header[2]! & 1;
    const cellCount = header[4]!;
    const rowCount = header[5]!;
    const cellListOffset = header[14 + slot]!;
    const rowListOffset = header[16 + slot]!;
    const listReadback = this.device.createBuffer({
      label: "Sparse CM12 accepted-topology lists QA",
      size: Math.max(4, 4 * (cellCount + rowCount)),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    let acceptedCells: Uint32Array;
    let acceptedRows: Uint32Array;
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 accepted-topology lists QA copy",
      });
      if (cellCount > 0) encoder.copyBufferToBuffer(this.topologyArena,
        this.topologyWorklistBaseBytes + 4 * cellListOffset,
        listReadback, 0, 4 * cellCount);
      if (rowCount > 0) encoder.copyBufferToBuffer(this.topologyArena,
        this.topologyWorklistBaseBytes + 4 * rowListOffset,
        listReadback, 4 * cellCount, 4 * rowCount);
      this.device.queue.submit([encoder.finish()]);
      await listReadback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(listReadback.getMappedRange());
      acceptedCells = words.slice(0, cellCount);
      acceptedRows = words.slice(cellCount, cellCount + rowCount);
    } finally {
      if (listReadback.mapState === "mapped") listReadback.unmap();
      listReadback.destroy();
    }

    const activity = await this.readActivitySnapshot();
    const leafManifestWords = 20 + 3 * activity.records.length;
    const leafReadback = this.device.createBuffer({
      label: "Sparse CM12 accepted-leaf manifest QA",
      size: Math.max(4, 4 * leafManifestWords),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    let manifestLeafIds: Uint32Array;
    let topologyDeltaLeafIds: Uint32Array;
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 accepted-leaf manifest QA copy",
      });
      encoder.copyBufferToBuffer(this.topologyArena,
        this.acceptedLeafManifestBaseBytes, leafReadback, 0, 4 * leafManifestWords);
      this.device.queue.submit([encoder.finish()]);
      await leafReadback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(leafReadback.getMappedRange());
      const manifestCount = words[slot]!;
      const manifestOffset = words[2 + slot]!;
      manifestLeafIds = words.slice(manifestOffset, manifestOffset + manifestCount);
      const deltaCount = words[10]!;
      const deltaOffset = words[11]!;
      topologyDeltaLeafIds = words.slice(deltaOffset, deltaOffset + deltaCount);
    } finally {
      if (leafReadback.mapState === "mapped") leafReadback.unmap();
      leafReadback.destroy();
    }
    const expectedCell = new Uint8Array(this.templateCellCount);
    const acceptedCell = new Uint8Array(this.templateCellCount);
    const expectedRow = new Uint8Array(this.templateRowCount);
    const acceptedRow = new Uint8Array(this.templateRowCount);
    const levels = sparseCM12TemplateLevels(this.brickFineResolution);
    const cellRangeOffset = this.templateWords[11]!;
    let activeLeafCount = 0;
    const expectedLeafIds: number[] = [];
    const expectedTopologyDeltaLeafIds: number[] = [];
    let expectedCellCount = 0;
    let overlappingLeafCellCount = 0;
    for (let brick = 0; brick < activity.records.length; brick += 1) {
      const record = activity.records[brick]!;
      if (record.topologyPreparationScheduled) expectedTopologyDeltaLeafIds.push(brick);
      if (!record.active) continue;
      activeLeafCount += 1;
      expectedLeafIds.push(brick);
      const level = Math.log2(record.acceptedResolution);
      const at = cellRangeOffset + 2 * (levels.length * brick + level);
      const first = this.templateWords[at]!;
      const count = this.templateWords[at + 1]!;
      expectedCellCount += count;
      for (let cell = first; cell < first + count; cell += 1) {
        if (expectedCell[cell] !== 0) overlappingLeafCellCount += 1;
        expectedCell[cell] = 1;
      }
    }
    let duplicateAcceptedCells = 0;
    let invalidAcceptedCells = 0;
    for (const cell of acceptedCells) {
      if (cell >= this.templateCellCount) { invalidAcceptedCells += 1;continue; }
      if (acceptedCell[cell] !== 0) duplicateAcceptedCells += 1;
      acceptedCell[cell] = 1;
    }
    let missingCells = 0;
    let unexpectedCells = 0;
    for (let cell = 0; cell < this.templateCellCount; cell += 1) {
      if (expectedCell[cell] !== 0 && acceptedCell[cell] === 0) missingCells += 1;
      if (expectedCell[cell] === 0 && acceptedCell[cell] !== 0) unexpectedCells += 1;
    }
    let duplicateAcceptedRows = 0;
    let invalidAcceptedRows = 0;
    for (const row of acceptedRows) {
      if (row >= this.templateRowCount) { invalidAcceptedRows += 1;continue; }
      if (acceptedRow[row] !== 0) duplicateAcceptedRows += 1;
      acceptedRow[row] = 1;
    }
    const rowMetadataPlane = this.templateWords[7]! + this.templateRowCount;
    let expectedRowCount = 0;
    let sharedRowCount = 0;
    let maximumRowOwnerCount = 0;
    for (let row = 0; row < this.templateRowCount; row += 1) {
      const requirements = this.templateWords[rowMetadataPlane + row]!
        & TEMPLATE_ROW_METADATA_OFFSET_MASK;
      const ownerCount = this.templateWords[requirements]!;
      let enabled = true;
      for (let owner = 0; owner < ownerCount; owner += 1) {
        const metadata = this.templateWords[requirements + 1 + owner]!;
        const brick = metadata >>> TEMPLATE_CELL_RESOLUTION_BITS;
        const resolution = metadata & TEMPLATE_CELL_RESOLUTION_MASK;
        const record = activity.records[brick];
        enabled = enabled && record !== undefined && record.active
          && record.acceptedResolution === resolution;
      }
      if (!enabled) continue;
      expectedRow[row] = 1;
      expectedRowCount += 1;
      if (ownerCount > 1) sharedRowCount += 1;
      maximumRowOwnerCount = Math.max(maximumRowOwnerCount, ownerCount);
    }
    let missingRows = 0;
    let unexpectedRows = 0;
    for (let row = 0; row < this.templateRowCount; row += 1) {
      if (expectedRow[row] !== 0 && acceptedRow[row] === 0) missingRows += 1;
      if (expectedRow[row] === 0 && acceptedRow[row] !== 0) unexpectedRows += 1;
    }
    let manifestLeafMismatches = Math.abs(
      manifestLeafIds.length - expectedLeafIds.length);
    let nonAscendingManifestLeafIds = 0;
    for (let index = 0; index < manifestLeafIds.length; index += 1) {
      if (manifestLeafIds[index] !== expectedLeafIds[index]) manifestLeafMismatches += 1;
      if (index > 0 && manifestLeafIds[index]! <= manifestLeafIds[index - 1]!) {
        nonAscendingManifestLeafIds += 1;
      }
    }
    let topologyDeltaLeafMismatches = Math.abs(
      topologyDeltaLeafIds.length - expectedTopologyDeltaLeafIds.length);
    let nonAscendingTopologyDeltaLeafIds = 0;
    for (let index = 0; index < topologyDeltaLeafIds.length; index += 1) {
      if (topologyDeltaLeafIds[index] !== expectedTopologyDeltaLeafIds[index]) {
        topologyDeltaLeafMismatches += 1;
      }
      if (index > 0 && topologyDeltaLeafIds[index]!
        <= topologyDeltaLeafIds[index - 1]!) {
        nonAscendingTopologyDeltaLeafIds += 1;
      }
    }
    return {
      topologyGeneration: header[0]!, acceptedSlot: slot,
      activeLeafCount,
      manifestLeafCount: manifestLeafIds.length,
      manifestLeafMismatches,
      nonAscendingManifestLeafIds,
      topologyDeltaLeafCount: topologyDeltaLeafIds.length,
      expectedTopologyDeltaLeafCount: expectedTopologyDeltaLeafIds.length,
      topologyDeltaLeafMismatches,
      nonAscendingTopologyDeltaLeafIds,
      leafRecordEstimateWords: 8 * activeLeafCount,
      acceptedCellCount: cellCount, expectedCellCount,
      acceptedRowCount: rowCount, expectedRowCount,
      sharedRowCount, maximumRowOwnerCount,
      overlappingLeafCellCount,
      duplicateAcceptedCells, invalidAcceptedCells, missingCells, unexpectedCells,
      duplicateAcceptedRows, invalidAcceptedRows, missingRows, unexpectedRows,
      leafCellRangesExactlyPartitionAcceptedCells: overlappingLeafCellCount === 0
        && duplicateAcceptedCells === 0 && invalidAcceptedCells === 0
        && missingCells === 0 && unexpectedCells === 0,
      rowRequirementsExactlyDescribeAcceptedRows: duplicateAcceptedRows === 0
        && invalidAcceptedRows === 0 && missingRows === 0 && unexpectedRows === 0,
      manifestExactlyMatchesActiveLeaves: manifestLeafMismatches === 0
        && nonAscendingManifestLeafIds === 0,
      topologyDeltaExactlyMatchesScheduledLeaves: topologyDeltaLeafMismatches === 0
        && nonAscendingTopologyDeltaLeafIds === 0,
    };
  }

  /** Candidate-only transaction receipt for Dawn diagnostics. This method
   * performs an explicit readback and is never called by production scheduling. */
  async readCandidateEffectsTransactionQA() {
    this.assertLive();
    const tfx = this.topologyEffectsAuthorityLayout;
    if (!tfx) return undefined;
    const topologyWords = 32;
    const manifestWords = 20 + 3 * this.lastPacked!.brickCount;
    const isaWords = this.iboSemanticAuthorityBaseWords === undefined ? 0 : 16;
    const iboWords = this.iboSlotBaseWords === undefined ? 0 : 14;
    const totalWords = SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS
      + topologyWords + manifestWords
      + isaWords + iboWords;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 candidate-effects transaction QA readback",
      size: 4 * totalWords,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 candidate-effects transaction QA copy",
      });
      encoder.copyBufferToBuffer(this.topologyArena, 4 * tfx.baseWords, readback, 0,
        4 * SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS);
      encoder.copyBufferToBuffer(this.topologyArena, this.topologyWorklistBaseBytes,
        readback, 4 * SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS, 4 * topologyWords);
      encoder.copyBufferToBuffer(this.topologyArena, this.acceptedLeafManifestBaseBytes,
        readback, 4 * (SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS + topologyWords),
        4 * manifestWords);
      if (this.iboSemanticAuthorityBaseWords !== undefined) {
        encoder.copyBufferToBuffer(this.topologyArena,
          4 * this.iboSemanticAuthorityBaseWords, readback,
          4 * (SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS
            + topologyWords + manifestWords),
          4 * isaWords);
      }
      if (this.iboSlotBaseWords !== undefined) {
        const destination = SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS
          + topologyWords + manifestWords + isaWords;
        encoder.copyBufferToBuffer(this.topologyArena,
          4 * this.iboSlotBaseWords[0], readback, 4 * destination, 7 * 4);
        encoder.copyBufferToBuffer(this.topologyArena,
          4 * this.iboSlotBaseWords[1], readback, 4 * (destination + 7), 7 * 4);
      }
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const named = (header: Readonly<Record<string, number>>, base: number) =>
        Object.fromEntries(Object.entries(header).map(([name, word]) =>
          [name, words[base + word]!])) as Readonly<Record<string, number>>;
      return Object.freeze({
        tfx: named(SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER, 0),
        iboPreAuthorization: Array.from(words.slice(
          SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.reservedBase,
          SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.reservedBase + 7)),
        authorizationChecks: Array.from(words.slice(
          SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.reservedBase + 7,
          SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.reservedBase + 20)),
        topology: Array.from(words.slice(
          SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS,
          SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS + topologyWords)),
        manifest: Array.from(words.slice(
          SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS + topologyWords,
          SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS + topologyWords + manifestWords)),
        isa: isaWords === 0 ? undefined : Array.from(words.slice(
          SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS + topologyWords + manifestWords,
          SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER_WORDS + topologyWords
            + manifestWords + isaWords)),
        ibo: iboWords === 0 ? undefined : {
          slot0: Array.from(words.slice(totalWords - 14, totalWords - 7)),
          slot1: Array.from(words.slice(totalWords - 7, totalWords)),
        },
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Explicit QA materialization of the renderer-facing FPP1 publication header. */
  async readFramePlanPresentationHeaderQA() {
    this.assertLive();
    const readback = this.device.createBuffer({
      label: "Sparse CM12 FPP1 header QA readback",
      size: 4 * SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 FPP1 header QA copy",
      });
      encoder.copyBufferToBuffer(this.activity,
        4 * this.framePlanPresentationLayout.baseWords, readback, 0,
        4 * SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return Object.freeze(Object.fromEntries(Object.entries(
        SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER).map(([name, word]) =>
        [name, words[word]!])));
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

  /** FSM1 header materialization; never consulted by frame scheduling. */
  async readFinalScalarMaskHeaderQA() {
    this.assertLive();
    const readback = this.device.createBuffer({
      label: "Sparse CM12 FSM1 header QA readback",
      size: 4 * SPARSE_CM12_FINAL_SCALAR_MASK_HEADER_WORDS,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 FSM1 header QA copy",
      });
      encoder.copyBufferToBuffer(this.topologyArena,
        4 * this.finalScalarPacketMaskLayout.baseWords, readback, 0,
        4 * SPARSE_CM12_FINAL_SCALAR_MASK_HEADER_WORDS);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const h = SPARSE_CM12_FINAL_SCALAR_MASK_HEADER;
      return Object.freeze({
        phase: words[h.phase]!, fault: words[h.fault]!,
        firstFaultPacket: words[h.firstFaultPacket]!,
        generation: words[h.generation]!,
        topologyGeneration: words[h.topologyGeneration]!,
        changedCellCount: words[h.changedCellCount]!,
        nonexactCellCount: words[h.nonexactCellCount]!,
        bulkCellCount: words[h.bulkCellCount]!, flipCellCount: words[h.flipCellCount]!,
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  async readTransportPacketIndirectQA(): Promise<readonly number[]> {
    this.assertLive();
    if (!this.transportPacketIndirectArguments) return Object.freeze(Array(3).fill(0));
    const readback = this.device.createBuffer({
      label: "Sparse CM12 transport packet indirect QA readback", size: 12,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 transport packet indirect QA copy",
      });
      encoder.copyBufferToBuffer(this.transportPacketIndirectArguments, 0, readback, 0, 12);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return Object.freeze(Array.from(new Uint32Array(readback.getMappedRange())));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Compact validation of the accepted TEI records owned by signed frontier
   * pages. This is an isolated Dawn diagnostic: production never reads packet
   * metadata back to the host. */
  async readDynamicTransportPacketsQA() {
    this.assertLive();
    const layout = this.transportExecutionImageLayout;
    const image = this.transportExecutionImage;
    if (!layout || !image) return Object.freeze({ activePages: 0, packets: 0, cells: 0 });
    const leafWords = 8;
    const packetWordsPerPage = 64
      * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS;
    const slotWords = this.topologyPageCapacity * (leafWords + packetWordsPerPage);
    const readback = this.device.createBuffer({
      label: "Sparse CM12 dynamic TEI QA readback",
      size: 4 * (1 + 2 * slotWords),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 dynamic TEI QA copy",
      });
      encoder.copyBufferToBuffer(this.topologyArena,
        this.topologyWorklistBaseBytes + 8, readback, 0, 4);
      for (let slot = 0; slot < 2; slot += 1) {
        const output = 4 * (1 + slot * slotWords);
        encoder.copyBufferToBuffer(image,
          4 * (layout.slotLeafBaseOffsets[slot]! + leafWords * this.initialWorldLeafCount),
          readback, output, 4 * leafWords * this.topologyPageCapacity);
        encoder.copyBufferToBuffer(image,
          4 * (layout.slotPacketBaseOffsets[slot]!
            + packetWordsPerPage * this.initialWorldLeafCount),
          readback, output + 4 * leafWords * this.topologyPageCapacity,
          4 * packetWordsPerPage * this.topologyPageCapacity);
      }
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const slot = words[0]! & 1;
      const slotBase = 1 + slot * slotWords;
      const packetBase = slotBase + leafWords * this.topologyPageCapacity;
      const cellsPerPage = this.brickFineResolution ** 3;
      const dynamicCellBase = this.templateCellCount
        - cellsPerPage * this.topologyPageCapacity;
      let activePages = 0, packets = 0, cells = 0;
      for (let page = 0; page < this.topologyPageCapacity; page += 1) {
        const leaf = slotBase + leafWords * page;
        const flags = words[leaf + 1]!;
        if ((flags & 0x8000_0000) === 0) continue;
        activePages += 1;
        const first = dynamicCellBase + page * cellsPerPage;
        const expectedValid = this.brickFineResolution
          | (this.brickFineResolution << 5) | (this.brickFineResolution << 10);
        if ((flags & 31) !== this.brickFineResolution || words[leaf + 2] !== first
          || words[leaf + 3] !== cellsPerPage || words[leaf + 5] !== expectedValid
          || words[leaf + 6] !== 1) {
          throw new Error(`Dynamic TEI leaf ${page} has an invalid B${
            this.brickFineResolution} descriptor`);
        }
        const packetAxis = this.brickFineResolution / 4;
        const activePacketCount = packetAxis ** 3;
        for (let local = 0; local < 64; local += 1) {
          const at = packetBase + packetWordsPerPage * page
            + SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS * local;
          const packetFirst = words[at + 1]!;
          const counts = words[at + 2]!;
          const strides = words[at + 3]!;
          if (local >= activePacketCount) {
            if (packetFirst !== 0xffff_ffff || counts !== 0 || strides !== 0) {
              throw new Error(`Dynamic TEI leaf ${page} publishes spare packet ${local}`);
            }
            continue;
          }
          const pz = Math.floor(local / (packetAxis * packetAxis));
          const remainder = local - pz * packetAxis * packetAxis;
          const py = Math.floor(remainder / packetAxis);
          const px = remainder - py * packetAxis;
          const expectedFirst = first + 4 * px + this.brickFineResolution
            * (4 * py + this.brickFineResolution * 4 * pz);
          const expectedCounts = (0x8000_0000 | 4 | (4 << 5) | (4 << 10)) >>> 0;
          const expectedStrides = (this.brickFineResolution
            | ((this.brickFineResolution ** 2) << 16)) >>> 0;
          if (packetFirst !== expectedFirst || counts !== expectedCounts
            || strides !== expectedStrides || packetFirst + 3
              + 3 * this.brickFineResolution + 3 * this.brickFineResolution ** 2
              >= first + cellsPerPage) {
            throw new Error(`Dynamic TEI leaf ${page} packet ${local} is malformed`);
          }
          packets += 1;
          cells += 64;
        }
      }
      return Object.freeze({ activePages, packets, cells });
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

  async readPersistentPressureCacheIndirectQA(): Promise<readonly number[]> {
    this.assertLive();
    const bytes = 12 * 12;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 persistent pressure-cache indirect QA readback", size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 persistent pressure-cache indirect QA copy",
      });
      encoder.copyBufferToBuffer(this.persistentPressureCacheIndirectArguments,
        0, readback, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return Object.freeze(Array.from(new Uint32Array(readback.getMappedRange())));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Working-set presentation allocation census; no simulation decision reads it. */
  async readPresentationPageAllocatorReceiptQA():
  Promise<SparseCM12PresentationPageAllocatorReceipt> {
    this.assertLive();
    const readback = this.device.createBuffer({
      label: "Sparse CM12 presentation allocator QA readback", size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 presentation allocator QA copy",
      });
      encoder.copyBufferToBuffer(this.activity,
        4 * this.framePlanPresentationLayout.allocatorBaseWords,
        readback, 0, 16);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      return Object.freeze({
        residentPages: words[0]!, faultCode: words[1]!,
        highWaterMark: words[2]!, cloneCount: words[3]!,
        capacity: this.framePlanPresentationLayout.pageCapacity,
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Signed-coordinate allocation and complete dynamic-page publication receipt. */
  async readWorldGrowthReceiptQA(): Promise<SparseCM12WorldGrowthReceipt> {
    this.assertLive();
    const localDirectoryWords = this.worldDirectoryLayout.totalWords
      - this.worldDirectoryLayout.baseWords;
    const directoryBytes = 4 * localDirectoryWords;
    const pageHeaderWords = 16;
    const pageHeaderBytes = 4 * pageHeaderWords * this.topologyPageCapacity;
    const activityRecordBytes = 4 * ACTIVITY_RECORD_WORDS * this.topologyPageCapacity;
    const cellsPerPage = this.brickFineResolution ** 3;
    const rowsPerPage = 3 * (this.brickFineResolution + 1)
      * this.brickFineResolution * this.brickFineResolution;
    const dynamicFieldBytes = 4 * cellsPerPage * this.topologyPageCapacity;
    const dynamicFaceBytes = 4 * rowsPerPage * this.topologyPageCapacity;
    const transportLeafWords = 8 * this.topologyPageCapacity;
    const transportLeafBytes = 4 * transportLeafWords;
    const bytes = directoryBytes + pageHeaderBytes + activityRecordBytes
      + 2 * dynamicFieldBytes + 2 * dynamicFaceBytes + 2 * transportLeafBytes;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 world-growth QA readback", size: Math.max(4, bytes),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 world-growth QA copy",
      });
      encoder.copyBufferToBuffer(this.topologyArena,
        4 * this.worldDirectoryLayout.baseWords,
        readback, 0, directoryBytes);
      for (let page = 0; page < this.topologyPageCapacity; page += 1) {
        encoder.copyBufferToBuffer(this.topologyArena,
          this.topologyPageDescriptorBaseBytes + 4 * page * this.topologyPageWords,
          readback, directoryBytes + 4 * pageHeaderWords * page,
          4 * pageHeaderWords);
      }
      encoder.copyBufferToBuffer(this.activity,
        4 * (ACTIVITY_HEADER_WORDS
          + ACTIVITY_RECORD_WORDS * this.initialWorldLeafCount),
        readback, directoryBytes + pageHeaderBytes, activityRecordBytes);
      // `templateCellCount` is the complete fixed-capacity physics slab used by
      // the resident hot path (initial templates followed by one B8 block per
      // dynamic page). The dynamic suffix therefore begins after subtracting
      // that reserved page slab, not after the whole capacity.
      const dynamicCellOffset = this.templateCellCount
        - cellsPerPage * this.topologyPageCapacity;
      if (dynamicCellOffset < 0) {
        throw new Error("Sparse CM12 world-growth QA dynamic cell slab is invalid");
      }
      encoder.copyBufferToBuffer(this.state,
        4 * (this.layout.densityA + dynamicCellOffset), readback,
        directoryBytes + pageHeaderBytes + activityRecordBytes, dynamicFieldBytes);
      encoder.copyBufferToBuffer(this.state,
        4 * (this.layout.densityB + dynamicCellOffset), readback,
        directoryBytes + pageHeaderBytes + activityRecordBytes + dynamicFieldBytes,
        dynamicFieldBytes);
      const dynamicRowOffset = this.templateWords[3]!;
      encoder.copyBufferToBuffer(this.state,
        4 * (this.layout.faceA + dynamicRowOffset), readback,
        directoryBytes + pageHeaderBytes + activityRecordBytes
          + 2 * dynamicFieldBytes, dynamicFaceBytes);
      encoder.copyBufferToBuffer(this.state,
        4 * (this.layout.faceB + dynamicRowOffset), readback,
        directoryBytes + pageHeaderBytes + activityRecordBytes
          + 2 * dynamicFieldBytes + dynamicFaceBytes, dynamicFaceBytes);
      if (this.transportExecutionImage && this.transportExecutionImageLayout) {
        for (let slot = 0; slot < 2; slot += 1) {
          encoder.copyBufferToBuffer(this.transportExecutionImage,
            4 * (this.transportExecutionImageLayout.slotLeafBaseOffsets[slot]!
              + 8 * this.initialWorldLeafCount),
            readback, directoryBytes + pageHeaderBytes + activityRecordBytes
              + 2 * dynamicFieldBytes + 2 * dynamicFaceBytes
              + slot * transportLeafBytes,
            transportLeafBytes);
        }
      }
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const h = SPARSE_CM12_WORLD_DIRECTORY_HEADER;
      const signed = (value: number) => ((value ^ 0x8000_0000) | 0);
      let synthesizedTopologyPages = 0;
      let publishedTopologyPages = 0;
      let activeTopologyPages = 0;
      let activeTransportTopologyPages = 0;
      let connectedHostIncidences = 0;
      let failedHostIncidences = 0;
      const publishedTopologyPageCoordinates: (readonly [number, number, number])[] = [];
      let claimedTopologyPages = 0;
      let dynamicLiquidMassFineCells = 0;
      let dynamicLiquidMinimumFine: [number, number, number] | undefined;
      let dynamicLiquidMaximumExclusiveFine: [number, number, number] | undefined;
      let dynamicMaximumAbsFaceVelocityFineCells_s = 0;
      let furthestLiquidLeafCoordinate: readonly [number, number, number] | undefined;
      const pageBase = localDirectoryWords;
      const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
      const activityBase = (directoryBytes + pageHeaderBytes) / 4;
      const densityABase = (directoryBytes + pageHeaderBytes + activityRecordBytes) / 4;
      const densityBBase = densityABase + dynamicFieldBytes / 4;
      const faceABase = densityBBase + dynamicFieldBytes / 4;
      const faceBBase = faceABase + dynamicFaceBytes / 4;
      const transportLeafABase = faceBBase + dynamicFaceBytes / 4;
      const transportLeafBBase = transportLeafABase + transportLeafWords;
      for (let page = 0; page < this.topologyPageCapacity; page += 1) {
        connectedHostIncidences += words[pageBase + pageHeaderWords * page + 12]!;
        const pageFailedHostIncidences = words[
          pageBase + pageHeaderWords * page + 13]!;
        failedHostIncidences += pageFailedHostIncidences;
        if (words[pageBase + pageHeaderWords * page + 2] === cellsPerPage) {
          claimedTopologyPages += 1;
        }
        const pageReceipt = words[pageBase + pageHeaderWords * page + 3]!;
        if (((pageReceipt & 0x8000_001f) >>> 0)
          === 0x8000_001f) {
          synthesizedTopologyPages += 1;
        }
        if (pageReceipt === 0x8000_003f) {
          publishedTopologyPages += 1;
          activeTopologyPages += Number(words[
            activityBase + ACTIVITY_RECORD_WORDS * page + 10] !== 0);
          const transportActive = [transportLeafABase, transportLeafBBase].some((base) =>
            (words[base + 8 * page + 1]! & 0x8000_0000) !== 0);
          activeTransportTopologyPages += Number(transportActive);
          const leaf = words[pageBase + pageHeaderWords * page]!;
          if (leaf < this.worldDirectoryLayout.leafCapacity) {
            const leafAt = this.worldDirectoryLayout.leafBaseWords + 5 * leaf;
            publishedTopologyPageCoordinates.push(Object.freeze([
              words[leafAt]! | 0, words[leafAt + 1]! | 0, words[leafAt + 2]! | 0,
            ]) as readonly [number, number, number]);
          }
        }
        let pageMass = 0;
        for (let local = 0; local < cellsPerPage; local += 1) {
          const density = Math.max(0, floats[densityABase + page * cellsPerPage + local]!,
            floats[densityBBase + page * cellsPerPage + local]!);
          pageMass += density;
          if (density <= 0.05) continue;
          const leaf = words[pageBase + pageHeaderWords * page]!;
          if (leaf >= this.worldDirectoryLayout.leafCapacity) continue;
          const leafAt = this.worldDirectoryLayout.leafBaseWords + 5 * leaf;
          const coordinate = [words[leafAt]! | 0, words[leafAt + 1]! | 0,
            words[leafAt + 2]! | 0] as const;
          const z = Math.floor(local / (this.brickFineResolution ** 2));
          const yz = local - z * this.brickFineResolution ** 2;
          const y = Math.floor(yz / this.brickFineResolution);
          const x = yz - y * this.brickFineResolution;
          const fine = [coordinate[0] * this.brickFineResolution + x,
            coordinate[1] * this.brickFineResolution + y,
            coordinate[2] * this.brickFineResolution + z] as const;
          if (!dynamicLiquidMinimumFine) {
            dynamicLiquidMinimumFine = [...fine];
            dynamicLiquidMaximumExclusiveFine = fine.map((value) => value + 1) as
              [number, number, number];
          } else {
            for (let axis = 0; axis < 3; axis += 1) {
              dynamicLiquidMinimumFine[axis] = Math.min(dynamicLiquidMinimumFine[axis]!, fine[axis]!);
              dynamicLiquidMaximumExclusiveFine![axis] = Math.max(
                dynamicLiquidMaximumExclusiveFine![axis]!, fine[axis]! + 1);
            }
          }
        }
        dynamicLiquidMassFineCells += pageMass;
        if (pageMass > 1e-4) {
          const leaf = words[pageBase + pageHeaderWords * page]!;
          if (leaf < this.worldDirectoryLayout.leafCapacity) {
            const leafAt = this.worldDirectoryLayout.leafBaseWords + 5 * leaf;
            const coordinate = [words[leafAt]! | 0, words[leafAt + 1]! | 0,
              words[leafAt + 2]! | 0] as const;
            if (!furthestLiquidLeafCoordinate
              || coordinate[0] > furthestLiquidLeafCoordinate[0]) {
              furthestLiquidLeafCoordinate = coordinate;
            }
          }
        }
      }
      for (let row = 0; row < rowsPerPage * this.topologyPageCapacity; row += 1) {
        dynamicMaximumAbsFaceVelocityFineCells_s = Math.max(
          dynamicMaximumAbsFaceVelocityFineCells_s,
          Math.abs(floats[faceABase + row]!), Math.abs(floats[faceBBase + row]!),
        );
      }
      return Object.freeze({
        initialLeaves: this.initialWorldLeafCount,
        liveLeaves: words[h.liveCount]!,
        capacity: words[h.leafCapacity]!,
        insertionFaults: words[h.insertionFaults]!,
        capacityFaults: words[h.capacityFaults]!,
        boundsGeneration: words[h.boundsGeneration]!,
        minimum: Object.freeze([signed(words[h.minimumX]!),
          signed(words[h.minimumY]!), signed(words[h.minimumZ]!)]) as
          readonly [number, number, number],
        maximumExclusive: Object.freeze([signed(words[h.maximumX]!),
          signed(words[h.maximumY]!), signed(words[h.maximumZ]!)]) as
          readonly [number, number, number],
        claimedTopologyPages,
        synthesizedTopologyPages,
        publishedTopologyPages,
        activeTopologyPages,
        activeTransportTopologyPages,
        connectedHostIncidences,
        failedHostIncidences,
        publishedTopologyPageCoordinates: Object.freeze(publishedTopologyPageCoordinates),
        dynamicLiquidMassFineCells,
        ...(dynamicLiquidMinimumFine && dynamicLiquidMaximumExclusiveFine ? {
          dynamicLiquidBoundsFine: Object.freeze({
            minimum: Object.freeze(dynamicLiquidMinimumFine),
            maximumExclusive: Object.freeze(dynamicLiquidMaximumExclusiveFine),
          }),
        } : {}),
        dynamicMaximumAbsFaceVelocityFineCells_s,
        ...(furthestLiquidLeafCoordinate
          ? { furthestLiquidLeafCoordinate: Object.freeze(furthestLiquidLeafCoordinate) }
          : {}),
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** Compact VEX2 receipt plus a QA-only census of the final validity mask. */
  async readVelocityExtensionHeaderQA(): Promise<SparseCM12VelocityExtensionHeaderQA> {
    this.assertLive();
    const layout = this.velocityExtensionLayout;
    const headerWords = SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS;
    const maskWords = 2 * layout.packetCapacity;
    const bytes = 4 * (headerWords + maskWords);
    const readback = this.device.createBuffer({
      label: "Sparse CM12 VEX2 header QA readback", size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 VEX2 header QA copy",
      });
      encoder.copyBufferToBuffer(this.activity, 4 * layout.headerBaseWords,
        readback, 0, 4 * headerWords);
      encoder.copyBufferToBuffer(this.activity, 4 * layout.validityABaseWords,
        readback, 4 * headerWords, 4 * maskWords);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const validityA = words.slice(headerWords, headerWords + maskWords);
      const density = sparseCM12VelocityExtensionMaskDensity(
        validityA, layout.packetCapacity, layout.dispatchPacketCount);
      const h = SPARSE_CM12_VELOCITY_EXTENSION_HEADER;
      if (words[h.magic] !== SPARSE_CM12_VELOCITY_EXTENSION_MAGIC
        || words[h.version] !== SPARSE_CM12_VELOCITY_EXTENSION_VERSION
        || words[h.headerWords] !== SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS
        || words[h.capacity] !== layout.cellCapacity
        || words[h.packetCapacity] !== layout.packetCapacity) {
        throw new Error("Sparse CM12 VEX2 header QA receipt is unavailable or incompatible");
      }
      const firstFaultCell = words[h.firstFaultCell]!;
      const hasFirstFault = words[h.faultCount]! > 0
        && firstFaultCell !== SPARSE_CM12_FRAME_CONTROL_INVALID
        && firstFaultCell < layout.cellCapacity;
      return {
        sourceFrameGeneration: words[h.sourceFrameGeneration]!,
        topologyGeneration: words[h.topologyGeneration]!,
        cellCapacity: words[h.capacity]!,
        packetCapacity: words[h.packetCapacity]!,
        dispatchPacketCount: layout.dispatchPacketCount,
        validCellCount: density.validCellCount,
        emptyPacketCount: density.emptyPacketCount,
        faultCount: words[h.faultCount]!,
        ...(hasFirstFault ? {
          firstFault: { cell: firstFaultCell, depth: words[h.firstFaultDepth]! },
        } : {}),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** VEX2 mask/depth/value diagnostic. No frame decision consumes it. */
  async readVelocityExtensionQA() {
    this.assertLive();
    if (!this.effectiveTransportVelocity) {
      throw new Error("Sparse CM12 VEX2 effective-velocity plane is unavailable");
    }
    const layout = this.velocityExtensionLayout;
    const capacity = layout.cellCapacity;
    const headerWords = SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS;
    const maskWords = 2 * layout.packetCapacity;
    const validityAAt = headerWords;
    const validityBAt = validityAAt + maskWords;
    const acceptedDepthAt = validityBAt + maskWords;
    const velocityAt = acceptedDepthAt + capacity;
    const totalWords = velocityAt + 4 * capacity;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 VEX2 QA readback",
      size: 4 * totalWords,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({ label: "Sparse CM12 VEX2 QA copy" });
      encoder.copyBufferToBuffer(this.activity, 4 * layout.headerBaseWords,
        readback, 0, 4 * headerWords);
      for (const [source, destination, words] of [
        [layout.validityABaseWords, validityAAt, maskWords],
        [layout.validityBBaseWords, validityBAt, maskWords],
        [layout.acceptedDepthBaseWords, acceptedDepthAt],
      ] as const) {
        encoder.copyBufferToBuffer(this.activity, 4 * source,
          readback, 4 * destination, 4 * (words ?? capacity));
      }
      encoder.copyBufferToBuffer(this.effectiveTransportVelocity, 0,
        readback, 4 * velocityAt, 16 * capacity);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const header = words.slice(0, headerWords);
      if (header[SPARSE_CM12_VELOCITY_EXTENSION_HEADER.magic]
          !== SPARSE_CM12_VELOCITY_EXTENSION_MAGIC) {
        throw new Error("Sparse CM12 VEX2 QA header is unavailable or incompatible");
      }
      const validityA = words.slice(validityAAt, validityAAt + maskWords);
      const density = sparseCM12VelocityExtensionMaskDensity(
        validityA, layout.packetCapacity, layout.dispatchPacketCount);
      header[SPARSE_CM12_VELOCITY_EXTENSION_HEADER.validCellCount]
        = density.validCellCount;
      header[SPARSE_CM12_VELOCITY_EXTENSION_HEADER.emptyPacketCount]
        = density.emptyPacketCount;
      return {
        dispatchPacketCount: layout.dispatchPacketCount,
        header,
        validityA,
        validityB: words.slice(validityBAt, validityBAt + maskWords),
        acceptedDepth: words.slice(acceptedDepthAt, acceptedDepthAt + capacity),
        velocityBits: words.slice(velocityAt, velocityAt + 4 * capacity),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  /** QA-only policy readback. It is never called by frame scheduling. */
  async readActivitySnapshot(
    includeWorldLeaves = false,
  ): Promise<SparseCM12GPUActivitySnapshot> {
    this.assertLive();
    const recordCapacity = includeWorldLeaves
      ? this.worldDirectoryLayout.leafCapacity : this.lastPacked!.brickCount;
    const wordsToRead = ACTIVITY_HEADER_WORDS
      + ACTIVITY_RECORD_WORDS * recordCapacity;
    const worldHeaderAt = wordsToRead;
    const worldLeavesAt = worldHeaderAt + SPARSE_CM12_WORLD_DIRECTORY_HEADER_WORDS;
    const totalWords = includeWorldLeaves
      ? worldLeavesAt + SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS * recordCapacity
      : wordsToRead;
    const readback = this.device.createBuffer({
      label: "Sparse CM12 activity QA readback",
      size: 4 * totalWords,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 activity QA copy",
      });
      encoder.copyBufferToBuffer(this.activity, 0, readback, 0, 4 * wordsToRead);
      if (includeWorldLeaves) {
        encoder.copyBufferToBuffer(this.topologyArena,
          4 * this.worldDirectoryLayout.baseWords, readback, 4 * worldHeaderAt,
          4 * SPARSE_CM12_WORLD_DIRECTORY_HEADER_WORDS);
        encoder.copyBufferToBuffer(this.topologyArena,
          4 * (this.worldDirectoryLayout.baseWords
            + this.worldDirectoryLayout.leafBaseWords),
          readback, 4 * worldLeavesAt,
          4 * SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS * recordCapacity);
      }
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const leafLimit = includeWorldLeaves ? Math.min(recordCapacity,
        words[worldHeaderAt + SPARSE_CM12_WORLD_DIRECTORY_HEADER.nextLeaf]!)
        : recordCapacity;
      const leafIds = Array.from({ length: leafLimit }, (_, leaf) => leaf).filter((leaf) =>
        !includeWorldLeaves || words[worldLeavesAt
          + SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS * leaf
          + SPARSE_CM12_WORLD_DIRECTORY_LEAF.generation]
            !== SPARSE_CM12_WORLD_DIRECTORY_INVALID);
      const records = leafIds.map((brick) => {
        const at = ACTIVITY_HEADER_WORDS + ACTIVITY_RECORD_WORDS * brick;
        const leafAt = worldLeavesAt + SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS * brick;
        return {
          leafId: brick,
          ...(includeWorldLeaves ? { coordinate: [words[leafAt]! | 0,
            words[leafAt + 1]! | 0, words[leafAt + 2]! | 0] as const } : {}),
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
          topologyPreparationScheduled: (words[at + 35]! & 1) !== 0,
          topologyPreparationEpoch: words[at + 36]!,
          topologyPage: words[at + 37] === INVALID ? undefined : words[at + 37],
        };
      });
      return {
        acceptedSteps: words[0]!, acceptedTopologyGeneration: words[12]!,
        residentBrickCount: words[8]!,
        faultFlags: words[7]!, newlyActivatedBrickCount: words[9]!,
        preparedBrickCount: words[16]!, committedBrickCount: words[17]!,
        commitFailed: words[21] !== 0, records,
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.rigidCoupling?.destroy();
    for (const buffer of [this.parameters, this.topology, this.state, this.partials,
      this.scalars, this.conditioning, this.activity, this.candidateState,
      this.topologyArena, this.acceptedIndirectArguments,
      this.pressureCellIndirectArguments,
      this.pressureMembershipIndirectArguments,
      this.pressureExecutionIndirectArguments,
      this.frameControlIndirectArguments,
      this.persistentPressureCacheIndirectArguments,
      this.framePlanIndirectArguments,
      this.presentationIndirectArguments,
      this.pressureTemplates, this.pressureWorklists,
      this.velocityExtensionDepths,
      this.fineParams, this.fineMetadata, this.fineWorklist, this.fineSamples,
      this.fineWorkA, this.fineWorkB, this.fineRollback]) {
      buffer.destroy();
    }
    this.transportExecutionImage?.destroy();
    this.effectiveTransportVelocity?.destroy();
    this.transportPacketIndirectArguments?.destroy();
    this.diagnosticsReadback.destroy();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Sparse CM12 resident pipeline is destroyed");
  }
}

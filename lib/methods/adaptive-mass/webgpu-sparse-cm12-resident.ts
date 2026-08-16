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
import {
  SPARSE_CM12_PRESSURE_JOURNAL_HEADER_FLOATS,
  SPARSE_CM12_PRESSURE_JOURNAL_ITERATION_FLOATS,
  decodeSparseCM12PressureJournal,
  sparseCM12PressureJournalLayout,
  sparseCM12PressureJournalSchedule,
  type SparseCM12PressureJournal,
  type SparseCM12PressureJournalLayout,
} from "./sparse-cm12-pressure-journal";
import { webgpuSparseCM12ResidentWGSL } from "./webgpu-sparse-cm12-resident.wgsl";
import {
  WebGPUSparseCM12RigidCoupling,
  type SparseCM12RigidResources,
} from "./webgpu-sparse-cm12-rigid-coupling";

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

/** The last stage in encode order; its boundary needs the treatment below. */
export const SPARSE_CM12_RESIDENT_FINAL_STAGE: SparseCM12ResidentStageId =
  SPARSE_CM12_RESIDENT_STAGES[SPARSE_CM12_RESIDENT_STAGES.length - 1]!;

/** Where an observer is told about each stage of the advance. */
export interface SparseCM12ResidentStageSeams {
  /**
   * Close the named stage, immediately after its last dispatch. A stage that
   * encodes nothing this advance still reports: it closes on its successor's
   * boundary and costs exactly zero.
   */
  readonly close: (stage: SparseCM12ResidentStageId) => void;
  /**
   * Name the final stage before its first dispatch, so an observer can put the
   * closing boundary on that stage's own pass. Measured on Dawn/Metal: a
   * trailing marker pass touches nothing the frame touches, so the driver
   * scheduled it 10 ms *before* the boundary it was meant to close and the
   * whole sample decoded as non-monotonic.
   */
  readonly openFinal?: (stage: SparseCM12ResidentStageId) => void;
}

const WORKGROUP_SIZE = 64;
/** Params in the resident WGSL, in bytes. Grow this when a field is added. */
const SPARSE_CM12_PARAMETER_BYTES = 448;
/** Twenty f32 convergence/diagnostic scalars; see the WGSL initialization. */
const SPARSE_CM12_PRESSURE_SCALAR_BYTES = 80;
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
const CANDIDATE_CELLS_PER_BRICK = 8 ** 3;
const CANDIDATE_CELL_CHANNELS = 6;
const CANDIDATE_FACE_CHANNELS = 6;
const CANDIDATE_FACE_SAMPLES_PER_SIDE = 8 ** 2;
const CANDIDATE_FLOATS_PER_BRICK = CANDIDATE_CELL_CHANNELS * CANDIDATE_CELLS_PER_BRICK
  + CANDIDATE_FACE_CHANNELS * CANDIDATE_FACE_SAMPLES_PER_SIDE;
const GPU_TOPOLOGY_PAGE_POOL_MINIMUM = 32;
const GPU_TOPOLOGY_PAGE_POOL_MAXIMUM = 512;
const GPU_TOPOLOGY_CELL_PAGE_HEADER_WORDS = 4;
const GPU_TOPOLOGY_CELL_RECORD_WORDS = 8;
const GPU_TOPOLOGY_CELL_PAGE_WORDS = GPU_TOPOLOGY_CELL_PAGE_HEADER_WORDS
  + CANDIDATE_CELLS_PER_BRICK * GPU_TOPOLOGY_CELL_RECORD_WORDS;

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
): boolean {
  return acceptedCellCount <= 250_000
    && acceptedRowCount <= 750_000
    && mutableBrickCount <= SPARSE_CM12_HOST_TEMPLATE_MUTABLE_BRICK_MAXIMUM;
}

/** Bounded by the mutable frontier, never by logical-domain brick count. */
export function sparseCM12TopologyPagePoolPlan(
  mutableFrontierBricks: number,
  enabled = true,
): SparseCM12TopologyPagePoolPlan {
  if (!enabled || mutableFrontierBricks <= 0) return {
    pageCapacity: 0, freeListWords: 0,
    pageWords: GPU_TOPOLOGY_CELL_PAGE_WORDS, descriptorWords: 0,
  };
  const pageCapacity = Math.min(GPU_TOPOLOGY_PAGE_POOL_MAXIMUM,
    Math.max(GPU_TOPOLOGY_PAGE_POOL_MINIMUM, Math.ceil(mutableFrontierBricks / 2)));
  return {
    pageCapacity, freeListWords: pageCapacity,
    pageWords: GPU_TOPOLOGY_CELL_PAGE_WORDS,
    descriptorWords: pageCapacity * GPU_TOPOLOGY_CELL_PAGE_WORDS,
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

const TEMPLATE_LEVELS = [1, 2, 4, 8] as const;
const TEMPLATE_MAGIC = 0x5343_4d54; // "SCMT"
const TEMPLATE_HEADER_WORDS = 16;
// The resident backend rejects separating spherical boundaries before packing,
// so openFraction=1, openVolume=volume and separatingMinimum=false are
// invariants rather than per-cell data. Keep exact geometry plus one packed
// [brick:28 | resolution:4] word: a 32-byte record, half the former footprint.
const TEMPLATE_CELL_RECORD_WORDS = 8;
const TEMPLATE_CELL_RESOLUTION_MASK = 0xf;

function packedTemplateCellMetadata(brick: number, resolution: number): number {
  if (brick < 0 || brick >= 2 ** 28 || ![1, 2, 4, 8].includes(resolution)) {
    throw new Error(`Sparse CM12 cell metadata cannot be packed: ${brick}/${resolution}`);
  }
  return ((brick << 4) | resolution) >>> 0;
}

const templateCellBrick = (words: Uint32Array, base: number) => words[base + 7]! >>> 4;
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
  const cellRangeOffset = at; at += 8 * atlas.bricks.length;
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
    for (let level = 0; level < 4; level += 1) {
      words[cellRangeOffset + 2 * (4 * index + level)] = first;
      words[cellRangeOffset + 2 * (4 * index + level) + 1] = cellCountByBrick[index]!;
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
  const mutableBrickKeys = new Set(atlas.bricks.filter((brick) =>
    sparseBrickSpan(brick) === 1).map((brick) => brick.key));
  // Even a bounded compatibility frontier is large enough that building one
  // full object-graph variant at 8^3 creates a costly transient heap. Partition
  // the mutable set and include a one-face-neighbour halo around each partition.
  // Rows touching a core brick then see exactly the same physical neighbours
  // as a whole-atlas build; halo-only boundary rows are discarded.
  const TEMPLATE_BUILD_CHUNK_BRICKS = 128;
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
    resampleBrick(brick, choose(brick))), atlas.generation, atlas.boundary);
  // One reusable full-grid scratch bounds host construction at accepted plus
  // one transient variant. Only cells/rows touching the mutable frontier are
  // copied into the persistent template library.
  const variantWorkspace = createSparseAtlasCompositeGridBuildWorkspace();
  const cells: SparseAtlasCompositeCell[] = [];
  const cellId = new Map<string, number>();
  const cellRanges = new Uint32Array(atlas.bricks.length * TEMPLATE_LEVELS.length * 2);
  const brickIndex = new Map(atlas.bricks.map((brick, index) => [brick.key, index]));
  // Preserve generation-zero IDs so the old direct dispatch and the new
  // accepted worklist address the same state while physical cutover lands.
  for (const source of acceptedGrid.cells) {
    cells.push(snapshotTemplateCell(source, cells.length));
    cellId.set(templateCellKey(source.brickKey, source.brickResolution,
      source.localIndex), source.id);
  }
  for (const brick of atlas.bricks) for (let levelIndex = 0;
    levelIndex < TEMPLATE_LEVELS.length; levelIndex += 1) {
    const range = 2 * (TEMPLATE_LEVELS.length * brickIndex.get(brick.key)! + levelIndex);
    cellRanges[range] = acceptedGrid.cellBaseByBrick.get(brick.key) ?? 0;
    cellRanges[range + 1] = brick.resolution ** 3;
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
    levelIndex < TEMPLATE_LEVELS.length; levelIndex += 1) {
    const level = TEMPLATE_LEVELS[levelIndex]!;
    for (const core of mutableChunks) {
      const coreKeys = new Set(core.map((brick) => brick.key));
      const localBricks = localBricksFor(core);
      const grid = buildSparseAtlasCompositeGrid(
        variantAtlasAtLevels((brick) => mutableBrickKeys.has(brick.key)
          ? level : brick.resolution, localBricks), 0.5, variantWorkspace,
      );
      for (const brick of localBricks.filter((candidate) =>
        mutableBrickKeys.has(candidate.key))) {
        const range = 2 * (TEMPLATE_LEVELS.length * brickIndex.get(brick.key)! + levelIndex);
        const firstSource = grid.cellBaseByBrick.get(brick.key)!;
        const count = level ** 3;
        let first = INVALID;
        for (let offset = 0; offset < count; offset += 1) {
          const source = grid.cells[firstSource + offset]!;
          const key = templateCellKey(brick.key, level, source.localIndex);
          let id = cellId.get(key);
          if (id === undefined) {
            id = cells.length; cells.push(snapshotTemplateCell(source, id)); cellId.set(key, id);
          }
          first = Math.min(first, id);
        }
        cellRanges[range] = first;
        cellRanges[range + 1] = count;
      }
      appendRows(grid, (row) => row.terms.some((term) =>
        coreKeys.has(grid.cells[term.cellId]!.brickKey)));
    }
  }
  const rungPairs = [[1, 2], [2, 4], [4, 8]] as const;
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
}

export interface SparseCM12FinePresentationPlan {
  readonly plan: FineLevelSetBrickPlan;
  readonly metadata: Uint32Array;
  readonly worklist: Uint32Array;
}

// Compact presentation source word. The low three bits retain the legacy
// 4^3 octant address, the next 24 bits address the sparse atlas leaf, and the
// high five bits carry log2(spanBricks). A macro leaf therefore costs one
// native-scale presentation page instead of either disappearing or expanding
// into O(span^3) fine pages.
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
): number {
  const spanLog = Math.log2(spanBricks);
  if (!Number.isInteger(brick) || brick < 0
    || brick > SPARSE_CM12_PRESENTATION_SOURCE_BRICK_MASK) {
    throw new RangeError(`Sparse CM12 presentation brick ${brick} exceeds the 24-bit ABI`);
  }
  if (!Number.isInteger(octant) || octant < 0 || octant > 7) {
    throw new RangeError(`Sparse CM12 presentation octant ${octant} is invalid`);
  }
  if (!Number.isInteger(spanLog) || spanLog < 0
    || spanLog > SPARSE_CM12_PRESENTATION_MAX_SPAN_LOG) {
    throw new RangeError(`Sparse CM12 presentation span ${spanBricks} is not representable`);
  }
  return ((spanLog << SPARSE_CM12_PRESENTATION_SOURCE_SPAN_SHIFT)
    | (brick << 3) | octant) >>> 0;
}

export function decodeSparseCM12FinePresentationSource(
  source: number,
): SparseCM12FinePresentationSource {
  const spanLog = source >>> SPARSE_CM12_PRESENTATION_SOURCE_SPAN_SHIFT;
  return {
    brick: (source >>> 3) & SPARSE_CM12_PRESENTATION_SOURCE_BRICK_MASK,
    octant: source & 7,
    spanBricks: 2 ** spanLog,
  };
}

export function sparseCM12FinePresentationPlan(
  atlas: SparseAdaptiveMassAtlas,
): SparseCM12FinePresentationPlan {
  const sampleDimensions = atlas.dimensions;
  const brickDimensions = sampleDimensions.map((value) => Math.ceil(value / 4)) as
    [number, number, number];
  const pages: { key: number; brick: number; octant: number; spanBricks: number }[] = [];
  let maximumSpanLog = 0;
  for (let brick = 0; brick < atlas.bricks.length; brick += 1) {
    const source = atlas.bricks[brick]!;
    const spanBricks = sparseBrickSpan(source);
    const spanLog = Math.log2(spanBricks);
    maximumSpanLog = Math.max(maximumSpanLog, spanLog);
    // Macro leaves publish one 4^3 page on their own 2*span finest-cell
    // sampling lattice. Surface leaves retain their eight unit-span octants.
    // This makes cost proportional to leaves while preserving deep liquid and
    // closed walls in the global presentation.
    if (spanBricks > 1) {
      const coordinate = source.coordinate.map((value) => 2 * value) as
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
    for (let oz = 0; oz < 2; oz += 1)
      for (let oy = 0; oy < 2; oy += 1)
        for (let ox = 0; ox < 2; ox += 1) {
          const coordinate = [2 * source.coordinate[0] + ox,
            2 * source.coordinate[1] + oy, 2 * source.coordinate[2] + oz] as const;
          if (coordinate.some((value, axis) => value >= brickDimensions[axis])) continue;
          pages.push({
            key: coordinate[0] + brickDimensions[0]
              * (coordinate[1] + brickDimensions[1] * coordinate[2]),
            brick,
            octant: ox | (oy << 1) | (oz << 2),
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
    // Compact Sparse CM12 owns this metadata word: source brick plus its 4^3
    // octant. Publication can therefore address the packed cell directly
    // instead of binary-searching the retained directory once per sample.
    metadata[FINE_LEVELSET_METADATA_WORDS * page + 3] =
      encodeSparseCM12FinePresentationSource(
        pages[page]!.brick, pages[page]!.octant, pages[page]!.spanBricks,
      );
  }
  // Compact mode deliberately stops after the active physical-page list.  It
  // has no `logicalBrickCount`-sized direct directory; renderer lookup binary
  // searches the key-sorted metadata instead.
  const worklist = new Uint32Array(FINE_LEVELSET_WORKSET_HEADER_WORDS + pageCount);
  worklist.set([1, pageCount, pageCount,
    (FINE_LEVELSET_COMPACT_LOOKUP_FLAG | 3 | (maximumSpanLog << 8)) >>> 0,
    Math.ceil(pageCount / WORKGROUP_SIZE), 1, 1]);
  for (let page = 0; page < pageCount; page += 1) {
    worklist[FINE_LEVELSET_WORKSET_HEADER_WORDS + page] = page;
  }
  const samplesPerBrick = 4 ** 3;
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
      brickResolution: 4,
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
  readonly plannedResolution: 1 | 2 | 4 | 8;
  readonly planReasons: number;
  readonly active: boolean;
  readonly activatedStep: number;
  /** Accepted logical level. It remains equal to packed topology until publication. */
  readonly acceptedResolution: 1 | 2 | 4 | 8;
  readonly candidateResolution: 1 | 2 | 4 | 8;
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
  readonly velocity: Float32Array;
  readonly pressure: Float32Array;
  readonly divergence: Float32Array;
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
  const brickIndexByKey = new Map(atlas.bricks.map((brick, index) => [brick.key, index]));
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
  const brickDimensions = atlas.dimensions.map((value) => Math.ceil(value / 8));
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
): Uint32Array {
  const edgeOffsets = templates.words[15]!;
  const edgeCount = templates.words[edgeOffsets + templates.cellCount]!;
  const edgeRecords = edgeOffsets + templates.cellCount + 1;
  const neighborOffset = 8 + templates.cellCount + templates.rowCount;
  const extrapolationWeightOffset = neighborOffset + edgeCount;
  const result = new Uint32Array(extrapolationWeightOffset + edgeCount);
  const cellOffset = templates.words[6]!;
  const cellRangeOffset = templates.words[11]!;
  const rowOffset = templates.words[7]!;
  const termOffset = templates.words[8]!;
  for (let cell = 0; cell < templates.cellCount; cell += 1) {
    const base = cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell;
    const brick = templateCellBrick(templates.words, base);
    const resolution = templateCellResolution(templates.words, base);
    const range = cellRangeOffset + 2 * (4 * brick + Math.log2(resolution));
    const first = templates.words[range]!, count = templates.words[range + 1]!;
    if (count !== resolution ** 3 || cell < first || cell >= first + count) {
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
  return result;
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
  private parity = 0;
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
    private readonly layout: ResidentStateLayout,
    buffers: readonly [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer,
      GPUBuffer, GPUBuffer, GPUBuffer],
    acceptedIndirectArguments: GPUBuffer,
    pressureCellIndirectArguments: GPUBuffer,
    pressureRowIndirectArguments: GPUBuffer,
    pressureWorklists: GPUBuffer,
    fineBuffers: readonly [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer,
      GPUBuffer],
    finePlan: FineLevelSetBrickPlan,
    diagnosticsReadback: GPUBuffer,
    bindGroup: GPUBindGroup,
    pressureBindGroup: GPUBindGroup,
    pressureTemplates: GPUBuffer,
    pipelines: Readonly<Record<string, GPUComputePipeline>>,
    cellCount: number,
    rowCount: number,
    private readonly templateCellCount: number,
    private readonly templateRowCount: number,
    private readonly pressureCoarseEdgeCount: number,
    private readonly pressureHierarchyGroupCount: number,
    private readonly pressureHierarchyEdgeCount: number,
    private readonly pressureScratchBytes: number,
    private readonly topologyWorklistBaseBytes: number,
    private readonly templateWords: Uint32Array,
    private horizontalD4Authority: boolean,
    private readonly boundary?: SphericalContainerFineGeometry,
    private readonly rigidCoupling?: WebGPUSparseCM12RigidCoupling,
  ) {
    [this.parameters, this.topology, this.state, this.partials, this.scalars,
      this.conditioning, this.activity, this.candidateState, this.topologyArena] = buffers;
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
    };
    this.sparseAdaptiveGridSource = {
      kind: "sparse-adaptive-grid-sampling",
      params: { buffer: this.parameters },
      topology: { buffer: this.topology },
      topologyArena: { buffer: this.topologyArena },
      state: { buffer: this.state },
      activity: { buffer: this.activity },
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
      pressureRowIndirectArguments,
      pressureTemplates, pressureWorklists,
      ...buffers, ...fineBuffers].reduce(
      (sum, buffer) => sum + buffer.size, 0,
    )
      + diagnosticsReadback.size;
  }

  static async create(
    device: GPUDevice,
    atlas: SparseAdaptiveMassAtlas,
    grid: SparseAtlasCompositeGrid,
    finestCellSize_m: number,
    initiallyActiveBrickKeys: ReadonlySet<number> = new Set(atlas.bricks.map(
      (brick) => brick.key,
    )),
    rigid?: SparseCM12RigidResources,
    journal?: SparseCM12PressureJournalCapacityRequest,
  ): Promise<WebGPUSparseCM12Resident> {
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
        for (let level = 0; level < TEMPLATE_LEVELS.length; level += 1) {
          const resolution = TEMPLATE_LEVELS[level]!;
          const range = rangeOffset + 2 * (TEMPLATE_LEVELS.length * brick + level);
          const first = templates.words[range]!, count = templates.words[range + 1]!;
          if (count !== resolution ** 3) {
            throw new Error(`Sparse CM12 template range ${brick}/${resolution} has ${count} cells`);
          }
          for (let cell = first; cell < first + count; cell += 1) {
            const base = cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell;
            const coordinate = atlas.bricks[brick]!.coordinate;
            const lower = [0, 1, 2].map((axis) => Math.round(
              templateFloats[base + axis]! - 0.5 * templateFloats[base + 4 + axis]!,
            ));
            if (templateCellResolution(templates.words, base) !== resolution
              || templateCellBrick(templates.words, base) !== brick
              || lower[0]! < 8 * coordinate[0]
              || lower[0]! >= 8 * (coordinate[0] + 1)
              || lower[1]! < 8 * coordinate[1]
              || lower[1]! >= 8 * (coordinate[1] + 1)
              || lower[2]! < 8 * coordinate[2]
              || lower[2]! >= 8 * (coordinate[2] + 1)) {
              throw new Error(`Sparse CM12 template range ${brick}/${resolution} aliases cell ${cell}`);
            }
          }
        }
      }
    }
    const fine = sparseCM12FinePresentationPlan(atlas);
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
    // WebGPU buffers start zeroed. Seed only the five nonzero scalar ranges
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
        initialState[layout.liquid + cell] = density >= 0.5 ? 1 : 0;
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
      seed(layout.liquid, Float32Array.from(templates.initialDensity,
        (density) => density >= 0.5 ? 1 : 0));
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
    const initialActivity = new Uint32Array(ACTIVITY_HEADER_WORDS
      + ACTIVITY_RECORD_WORDS * packed.brickCount);
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
      // Low four bits retain the coarsest calm level accepted before this
      // brick's first promotion; bit 31 is latched by that promotion.
      initialActivity[at + 38] = atlas.bricks[brick]!.resolution;
      if (initialActivity[at + 10] !== 0) {
        if (atlas.bricks[brick]!.resolution === 8) initialActivity[19] += 1;
        else initialActivity[20] += 1;
      }
      if (initialActivity[at + 10] !== 0) {
        const level = Math.log2(atlas.bricks[brick]!.resolution);
        const range = templates.words[11]! + 2 * (4 * brick + level);
        initialActivity[11] += templates.words[range + 1]!;
      }
    }
    const activity = uploadBuffer(device, "Sparse CM12 resident activity history",
      initialActivity, storage);
    const pressureEdgeOffset = templates.words[15]!;
    const pressureEdgeCount = templates.words[pressureEdgeOffset + templates.cellCount]!;
    const pressureTopology = compactPressureTopology(templates, atlas);
    const pressureWorklistData = pressureWorklistAndNeighbors(templates);
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
        4 * CANDIDATE_FLOATS_PER_BRICK * packed.candidateBrickCount),
      usage: storage,
    });
    const topologyPagePool = sparseCM12TopologyPagePoolPlan(
      initiallyActiveBrickKeys.size, !hostTemplateVariants,
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
    const topologyArena = device.createBuffer({
      label: "Sparse CM12 physical topology templates and worklists",
      size: Math.max(4, physicalTemplateBytes + initialWorklists.byteLength),
      usage: storage,
    });
    device.queue.writeBuffer(topologyArena, 0, templates.words.buffer as ArrayBuffer,
      templates.words.byteOffset, physicalTemplateBytes);
    device.queue.writeBuffer(topologyArena, physicalTemplateBytes,
      initialWorklists.buffer as ArrayBuffer, initialWorklists.byteOffset,
      initialWorklists.byteLength);
    const pressureTemplates = uploadBuffer(device,
      "Sparse CM12 read-only pressure topology", pressureTopology,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const pressureWorklists = uploadBuffer(device,
      "Sparse CM12 ordinary pressure worklists and dense neighbors",
      pressureWorklistData,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const acceptedIndirectArguments = uploadBuffer(device,
      "Sparse CM12 accepted indirect dispatch snapshot",
      initialWorklists.subarray(8, 14),
      GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
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
      label: "Sparse CM12 compact fine presentation samples",
      size: Math.max(4, fine.plan.payloadCapacityBytes),
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
      size: SPARSE_CM12_PRESSURE_SCALAR_BYTES + 4 * ACTIVITY_HEADER_WORDS + 12,
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
          buffer: { type: "read-only-storage" } },
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
    const shaderModule = device.createShaderModule({ label: "Sparse CM12 resident shader",
      code: webgpuSparseCM12ResidentWGSL });
    const pipelineLayout = device.createPipelineLayout({ label: "Sparse CM12 resident pipeline layout",
      bindGroupLayouts: [bindGroupLayout] });
    const names = ["injectLiquid", "initializeTransportVelocity",
      "extrapolateTransportVelocityToSource", "extrapolateTransportVelocityToDestination",
      "prepareTransportFaces", "traceGammaAndBeta", "scatterDensityDeficit",
      "gatherConservativeDensity", "seedTracers", "advanceTracers",
      "scatterGammaSnapshot", "finalizeGammaSnapshot", "scatterGammaRefinement",
      "finalizeGammaRefinement", "prepareSharpeningField", "scatterSharpeningMass",
      "finalizeSharpening", "clearSolidExcess", "scatterSolidExcess",
      "finalizeSolidExcess", "preserveHorizontalD4",
      "commitHorizontalD4",
      "forceFaces", "classifyPressureCells", "countPressureCells",
      "finalizePressureCellWorklist", "compactPressureCells", "classifyRows",
      "countPressureRows", "finalizePressureRowWorklist", "compactPressureRows",
      "bakeEffectivePressureEdges", "preparePressure",
      "bakeBrickAggregateDiagonal", "restrictBrickAggregateResidual",
      "bakePressureHierarchyDiagonal", "bakePressureHierarchyEdges",
      "restrictPressureHierarchyResidual", "refinePressureHierarchyCorrection",
      "combinePressureHierarchyCorrection",
      "bakeBrickAggregateEdges", "refineBrickAggregateAtoB1",
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
      "advanceActivityClock", "measureBrickActivity",
      "planBrickResolution", "activateSweptReceivers", "closePlannedResolution",
      "validateCandidateResolution", "scheduleTopologyPreparation",
      "allocateCandidateTopologyPages", "synthesizeCandidateCellPages",
      "deferDynamicTopologyPublication",
      "beginShadowTopology", "buildShadowCellWorklist", "buildShadowRowWorklist",
      "finalizeShadowWorklists", "transferCandidateCells",
      "prepareCandidateFaceReceipts", "transferCandidateFaces",
      "writeCandidateCellsToShadow", "reconstructShadowFaces",
      "validateAndCommitShadowTopology",
      "retireUnsupportedEmptyBricks",
      "classifyPresentationBricks",
      "publishSparseLevelSet",
      "journalIteration", "journalSnapshot"] as const;
    const entries = await Promise.all(names.map(async (name) => [name,
      await device.createComputePipelineAsync({ label: `Sparse CM12 ${name}`,
        layout: pipelineLayout, compute: { module: shaderModule, entryPoint: name } })] as const));
    // Two pipelines from one entry point: the snapshot variant additionally
    // advances the device-side snapshot cursor and stamps its slot into the
    // record it writes. Compiled only when a journal was actually reserved.
    const journalEntries = layout.journal === 0 ? [] : [
      ["journalIterationSnapshot", await device.createComputePipelineAsync({
        label: "Sparse CM12 journalIteration with field snapshot",
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: "journalIteration",
          constants: { JOURNAL_SNAPSHOT: 1 },
        },
      })] as const,
    ];
    const rigidCoupling = rigid ? await WebGPUSparseCM12RigidCoupling.create(device, {
      parameters,
      state,
      topologyArena,
      acceptedIndirectArguments,
      rigidBodies: rigid.bodies,
      exchange: rigid.exchange,
    }) : undefined;
    const result = new WebGPUSparseCM12Resident(device, atlas.dimensions, layout,
      [parameters, topology, state, partials, scalars, conditioning, activity,
        candidateState, topologyArena],
      acceptedIndirectArguments,
      pressureCellIndirectArguments,
      pressureRowIndirectArguments,
      pressureWorklists,
      [fineParams, fineMetadata, fineWorklist, fineSamples, fineWorkA, fineWorkB,
        fineRollback],
      fine.plan,
      diagnosticsReadback,
      bindGroup,
      pressureBindGroup,
      pressureTemplates,
      Object.fromEntries([...entries, ...journalEntries]),
      templates.cellCount, templates.rowCount,
      templates.cellCount, templates.rowCount,
      pressureCoarseEdgeCount,
      pressureHierarchyGroupCounts.reduce((sum, count) => sum + count, 0),
      pressureHierarchyEdgeCounts.reduce((sum, count) => sum + count, 0),
      pressureScratchBytes,
      physicalTemplateBytes,
      templates.words,
      horizontalD4Authority, atlas.boundary, rigidCoupling);
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
    // classifyRows only visits accepted rows. Clear the cached ghost-fluid
    // theta field once so the pressure iteration can use theta==0 as its row
    // acceptance mask instead of revalidating topology requirements in every
    // cell incidence on every PCG iteration.
    encoder.clearBuffer(this.state, 4 * this.layout.theta,
      4 * this.templateRowCount);
    encoder.clearBuffer(this.state, 4 * this.layout.liquid,
      4 * this.templateCellCount);
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
        if (id === SPARSE_CM12_RESIDENT_FINAL_STAGE) seams.openFinal?.(id);
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
      this.rigidCoupling?.encodeVoxelization(encoder, bodyCount);
      // Transport extrapolation consumes the same construction-time CSR edge
      // cache as pressure. Keep that immutable topology bound through the hot
      // physics stages; presentation restores its own metadata binding later.
      useBindGroup(this.pressureBindGroup);
      dispatchAccepted("initializeTransportVelocity", "cell");
      for (let sweep = 0; sweep < 8; sweep += 1) {
        dispatchAccepted(sweep % 2 === 0
          ? "extrapolateTransportVelocityToSource"
          : "extrapolateTransportVelocityToDestination", "cell");
      }
    });
    stage("face-preparation", () => {
      dispatchAccepted("prepareTransportFaces", "row");
    });
    stage("conservative-transport", () => {
      dispatchAccepted("traceGammaAndBeta", "cell");
      dispatchAccepted("scatterDensityDeficit", "cell");
      dispatchAccepted("gatherConservativeDensity", "cell");
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
    });
    stage("surface-sharpening", () => {
      dispatchAccepted("prepareSharpeningField", "cell");
      dispatchAccepted("scatterSharpeningMass", "cell");
      dispatchAccepted("finalizeSharpening", "cell");
      if (this.boundary || bodyCount > 0) {
        dispatchAccepted("clearSolidExcess", "cell");
        dispatchAccepted("scatterSolidExcess", "cell");
        dispatchAccepted("finalizeSolidExcess", "cell");
      }
    });
    stage("symmetry-authority", () => {
      if (!this.horizontalD4Authority || bodyCount > 0) return;
      dispatchAccepted("preserveHorizontalD4", "cell");
      dispatchAccepted("commitHorizontalD4", "cell");
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
      useBindGroup(this.pressureBindGroup);
      dispatchAccepted("classifyPressureCells", "cell");
      dispatchAccepted("countPressureCells", "cell");
      dispatch("finalizePressureCellWorklist", 1);
      dispatchAccepted("compactPressureCells", "cell");
      dispatchAccepted("classifyRows", "row");
      dispatchAccepted("countPressureRows", "row");
      dispatch("finalizePressureRowWorklist", 1);
      dispatchAccepted("compactPressureRows", "row");
      dispatchAccepted("bakeEffectivePressureEdges", "cell");
      dispatchAccepted("preparePressure", "cell");
      dispatch("bakeBrickAggregateEdges",
        Math.max(1, Math.ceil(this.pressureCoarseEdgeCount / WORKGROUP_SIZE)));
      dispatch("bakePressureHierarchyEdges", Math.max(1,
        Math.ceil(this.pressureHierarchyEdgeCount / WORKGROUP_SIZE)));
      dispatch("bakeBrickAggregateDiagonal", packed.brickCount);
      dispatch("bakePressureHierarchyDiagonal", this.pressureHierarchyGroupCount);
      closePass();
      encoder.copyBufferToBuffer(this.conditioning, 4,
        this.pressureCellIndirectArguments, 0, 12);
      encoder.copyBufferToBuffer(this.conditioning,
        4 * (4 + this.templateCellCount + 1),
        this.pressureRowIndirectArguments, 0, 12);
    });
    stage("pressure-rhs", () => {
      dispatchPressureCell("initializePCG");
      dispatch("restrictBrickAggregateResidual", packed.brickCount);
      dispatch("refineBrickAggregateAtoB1", packed.brickCount);
      dispatch("refineBrickAggregateBtoA2", packed.brickCount);
      dispatch("refineBrickAggregateAtoB3", packed.brickCount);
      dispatch("restrictPressureHierarchyResidual", this.pressureHierarchyGroupCount);
      dispatch("refinePressureHierarchyCorrection", this.pressureHierarchyGroupCount);
      dispatch("combinePressureHierarchyCorrection",
        Math.max(1, Math.ceil(packed.brickCount / WORKGROUP_SIZE)));
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
          dispatch("restrictBrickAggregateResidual", packed.brickCount);
          dispatch("refineBrickAggregateAtoB1", packed.brickCount);
          dispatch("refineBrickAggregateBtoA2", packed.brickCount);
          dispatch("refineBrickAggregateAtoB3", packed.brickCount);
          dispatch("restrictPressureHierarchyResidual", this.pressureHierarchyGroupCount);
          dispatch("refinePressureHierarchyCorrection", this.pressureHierarchyGroupCount);
          dispatch("combinePressureHierarchyCorrection",
            Math.max(1, Math.ceil(packed.brickCount / WORKGROUP_SIZE)));
          dispatchPressureCell("applyBrickAggregatePreconditioner");
          dispatchPressureCell("applyPipelinedImage");
          dispatch("reducePipelinedIteration", 1);
          if ((iteration + 1) % SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE === 0
            && iteration + 1 < pressureIterations) {
            dispatchPressureCell("measureGuardedTrueResidual");
            dispatch("reduceGuardedTrueResidual", 1);
            dispatchPressureCell("restartPCGAfterCurvatureLoss");
            dispatch("restrictBrickAggregateResidual", packed.brickCount);
            dispatch("refineBrickAggregateAtoB1", packed.brickCount);
            dispatch("refineBrickAggregateBtoA2", packed.brickCount);
            dispatch("refineBrickAggregateAtoB3", packed.brickCount);
            dispatch("refineBrickAggregateBtoA4", packed.brickCount);
            dispatch("refineBrickAggregateAtoB5", packed.brickCount);
            dispatch("refineBrickAggregateBtoA6", packed.brickCount);
            dispatch("refineBrickAggregateAtoB7", packed.brickCount);
            dispatch("restrictPressureHierarchyResidual", this.pressureHierarchyGroupCount);
            dispatch("refinePressureHierarchyCorrection", this.pressureHierarchyGroupCount);
            dispatch("combinePressureHierarchyCorrection",
              Math.max(1, Math.ceil(packed.brickCount / WORKGROUP_SIZE)));
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
      dispatchPressureRow("projectFaces");
      dispatchAccepted("collocateAndDiagnose", "cell");
      if (bodyCount > 0 && this.rigidCoupling) {
        pass?.end();
        pass = undefined;
        this.rigidCoupling.encodeReaction(encoder);
      }
    });
    stage("projection-diagnostics", () => {
      dispatchAccepted("measureDivergenceDiagnostics", "cell");
      dispatch("reduceDivergenceDiagnostics", 1);
    });
    stage("activity-measurement", () => {
      useBindGroup(this.bindGroup);
      dispatch("advanceActivityClock", 1);
      dispatch("measureBrickActivity", packed.brickCount);
    });
    stage("resolution-planning", () => {
      dispatch("planBrickResolution", bricks);
      dispatch("activateSweptReceivers", bricks);
      for (let gradingPass = 0; gradingPass < 3; gradingPass += 1) {
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
    });
    stage("brick-retirement", () => {
      dispatch("retireUnsupportedEmptyBricks", bricks);
    });
    stage("presentation-publication", () => {
      // measureBrickActivity has already reduced the identical density/mass
      // wetness predicate and published its existing occupied bit per resident
      // brick. Do not sweep every brick's cells a second time here.
      dispatch("publishSparseLevelSet",
        this.globalFineLevelSetSource.plan.maximumResidentBricks);
    });
    closePass();
    // The commit above authors next frame's accepted workgroup triplets. Keep
    // the writable arena out of indirect-dispatch synchronization scopes by
    // snapshotting those six words with a device-side copy between passes.
    encoder.copyBufferToBuffer(this.topologyArena,
      this.topologyWorklistBaseBytes + 4 * 8,
      this.acceptedIndirectArguments, 0, 6 * 4);
    this.parity ^= 1;
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
    return {
      state: this.state,
      topologyArena: this.topologyArena,
      faceFloatOffset: this.parity !== 0 ? this.layout.faceB : this.layout.faceA,
      rowCount: this.rowCount,
      domainFine: this.dimensions,
      // The width the last advance was actually parameterised with, which is
      // the unit its face velocities are stored in.
      finestCell_m: this.parameterF32[41] ?? 0,
    };
  }

  /** Publish generation zero without executing a physics step or mapping state. */
  encodeInitialPresentation(encoder: GPUCommandEncoder, finestCellSize_m: number): void {
    this.assertLive();
    this.writeParameters(this.lastPacked!, 0.004, finestCellSize_m, 1, [0, 0, 0]);
    const pass = encoder.beginComputePass({ label: "Sparse CM12 resident initial presentation" });
    pass.setPipeline(this.pipelines.classifyPresentationBricks!);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.lastPacked!.brickCount / WORKGROUP_SIZE));
    pass.setPipeline(this.pipelines.publishSparseLevelSet!);
    pass.dispatchWorkgroups(this.globalFineLevelSetSource.plan.maximumResidentBricks);
    pass.end();
  }

  encodeLiquidInjection(
    encoder: GPUCommandEncoder,
    finestCellSize_m: number,
    centerFine: readonly [number, number, number],
    radiusFine: readonly [number, number, number],
  ): void {
    this.assertLive();
    if (Math.abs(centerFine[0] - 0.5 * this.dimensions[0]) > 1e-6
      || Math.abs(centerFine[2] - 0.5 * this.dimensions[2]) > 1e-6
      || Math.abs(radiusFine[0] - radiusFine[2]) > 1e-6) {
      this.horizontalD4Authority = false;
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
    // Promote every intersected brick before writing any density. The planner
    // treats the enabled injection as refine-only: untouched accepted bricks
    // are preserved, while closure may still grow the required 2:1 support.
    dispatchTopology("planBrickResolution", bricks);
    dispatchTopology("activateSweptReceivers", bricks);
    for (let gradingPass = 0; gradingPass < 3; gradingPass += 1) {
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
    injectionPass.setPipeline(this.pipelines.classifyPresentationBricks!);
    injectionPass.dispatchWorkgroups(bricks);
    injectionPass.setPipeline(this.pipelines.publishSparseLevelSet!);
    injectionPass.dispatchWorkgroups(
      this.globalFineLevelSetSource.plan.maximumResidentBricks);
    injectionPass.end();
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
    u.set([...this.dimensions, this.boundary ? 1 : 0], 4);
    u.set([packed.cellOffset, packed.rowOffset, packed.termOffset, packed.incidenceOffset], 8);
    u.set([packed.incidenceRecordOffset, packed.brickLookupOffset,
      packed.brickOffset, packed.backgroundOwnerOffset], 12);
    u.set([l.densityA, l.densityB, l.gammaA, l.gammaB], 16);
    u.set([l.cellVelocityA, l.cellVelocityB, l.faceA, l.faceB], 20);
    u.set([l.pressure, l.rhs, l.diagonal, l.liquid], 24);
    u.set([l.theta, l.residual, l.preconditioned, l.direction], 28);
    u.set([l.applied, l.divergence, 0, 0], 32);
    // The D4 pass needs two disjoint scalar scratch arrays. In particular the
    // gamma scratch must never alias densityA at offset zero: doing so corrupts
    // gamma after the first symmetric frame and makes transport create mass on
    // the next frame.
    u.set([l.sharpeningDelta, l.symmetryGamma, l.tracers, 0], 36);
    f.set([dt_s, finestCellSize_m, pressureScale, this.parity], 40);
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
    encoder.copyBufferToBuffer(this.conditioning, 0, this.diagnosticsReadback,
      SPARSE_CM12_PRESSURE_SCALAR_BYTES + 4 * ACTIVITY_HEADER_WORDS + 8, 4);
    this.device.queue.submit([encoder.finish()]);
    await this.diagnosticsReadback.mapAsync(GPUMapMode.READ);
    const mapped = this.diagnosticsReadback.getMappedRange();
    const values = new Float32Array(mapped, 0,
      SPARSE_CM12_PRESSURE_SCALAR_BYTES / 4);
    const activity = new Uint32Array(mapped,
      SPARSE_CM12_PRESSURE_SCALAR_BYTES, ACTIVITY_HEADER_WORDS);
    const acceptedCounts = new Uint32Array(mapped,
      SPARSE_CM12_PRESSURE_SCALAR_BYTES + 4 * ACTIVITY_HEADER_WORDS, 2);
    const pressureCounts = new Uint32Array(mapped,
      SPARSE_CM12_PRESSURE_SCALAR_BYTES + 4 * ACTIVITY_HEADER_WORDS + 8, 1);
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
      pressureCellCount: pressureCounts[0]!,
    };
    this.diagnosticsReadback.unmap();
    return result;
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
    const readback = this.device.createBuffer({
      label: "Sparse CM12 QA field readback",
      size: this.state.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 QA field copy",
      });
      encoder.copyBufferToBuffer(this.state, 0, readback, 0, this.state.size);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const state = new Float32Array(readback.getMappedRange());
      const [nx, ny, nz] = this.dimensions;
      const count = nx * ny * nz;
      const density = new Float32Array(count);
      const velocity = new Float32Array(4 * count);
      const pressure = new Float32Array(count);
      const divergence = new Float32Array(count);
      const densityOffset = this.parity !== 0 ? this.layout.densityB : this.layout.densityA;
      const velocityOffset = this.parity !== 0
        ? this.layout.cellVelocityB : this.layout.cellVelocityA;
      const cellWidth_m = this.parameterF32[41]!;
      const pressureScale = this.parameterF32[42]!;
      const topologyFloats = new Float32Array(this.templateWords.buffer,
        this.templateWords.byteOffset, this.templateWords.length);
      const cellOffset = this.templateWords[6]!, rangeOffset = this.templateWords[11]!;
      for (let brick = 0; brick < activitySnapshot.records.length; brick += 1) {
        const record = activitySnapshot.records[brick]!;
        if (!record.active || (record.reasons & 64) === 0) continue;
        const level = record.acceptedResolution === 8 ? 3
          : record.acceptedResolution === 4 ? 2 : record.acceptedResolution === 2 ? 1 : 0;
        const first = this.templateWords[rangeOffset + 2 * (4 * brick + level)]!;
        const cellCount = this.templateWords[rangeOffset + 2 * (4 * brick + level) + 1]!;
        const brickRecord = this.lastPacked!.brickOffset + 2 * brick;
        const key = this.lastPacked!.words[brickRecord + 1]!;
        const spanBricks = 1 << (this.lastPacked!.words[brickRecord]! & 31);
        const brickDimensions = this.dimensions.map((size) => Math.ceil(size / 8));
        const brickZ = Math.floor(key / (brickDimensions[0]! * brickDimensions[1]!));
        const keyXY = key - brickZ * brickDimensions[0]! * brickDimensions[1]!;
        const brickY = Math.floor(keyXY / brickDimensions[0]!);
        const brickX = keyXY - brickY * brickDimensions[0]!;
        for (let cell = first; cell < first + cellCount; cell += 1) {
        const base = cellOffset + TEMPLATE_CELL_RECORD_WORDS * cell;
        const lower = [0, 1, 2].map((axis) => Math.round(
          topologyFloats[base + axis]! - 0.5 * topologyFloats[base + 4 + axis]!,
        ));
        if (lower[0] < 8 * brickX || lower[0] >= 8 * (brickX + spanBricks)
          || lower[1] < 8 * brickY || lower[1] >= 8 * (brickY + spanBricks)
          || lower[2] < 8 * brickZ || lower[2] >= 8 * (brickZ + spanBricks)) {
          throw new Error(`Sparse CM12 active brick ${brick} at ${brickX},${brickY},${brickZ}`
            + ` aliases cell ${cell} at ${lower.join(",")}`);
        }
        const span = [Math.round(topologyFloats[base + 4]!),
          Math.round(topologyFloats[base + 5]!),
          Math.round(topologyFloats[base + 6]!)] as const;
        const rho = state[densityOffset + cell]!;
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
              velocity[4 * at] = vx; velocity[4 * at + 1] = vy;
              velocity[4 * at + 2] = vz;
              pressure[at] = mappedPressure; divergence[at] = div;
            }
        }
      }
      return { density, velocity, pressure, divergence };
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
          plannedResolution: (words[at + 8] === 8 ? 8
            : words[at + 8] === 4 ? 4 : words[at + 8] === 2 ? 2 : 1) as
              SparseCM12GPUActivityRecord["plannedResolution"],
          planReasons: words[at + 9]!,
          active: words[at + 10] !== 0,
          activatedStep: words[at + 11]!,
          acceptedResolution: (words[at + 12] === 8 ? 8
            : words[at + 12] === 4 ? 4 : words[at + 12] === 2 ? 2 : 1) as
              SparseCM12GPUActivityRecord["acceptedResolution"],
          candidateResolution: (words[at + 13] === 8 ? 8
            : words[at + 13] === 4 ? 4 : words[at + 13] === 2 ? 2 : 1) as
              SparseCM12GPUActivityRecord["candidateResolution"],
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
    for (const buffer of [this.parameters, this.topology, this.state, this.partials,
      this.scalars, this.conditioning, this.activity, this.candidateState,
      this.topologyArena, this.acceptedIndirectArguments,
      this.pressureCellIndirectArguments, this.pressureRowIndirectArguments,
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

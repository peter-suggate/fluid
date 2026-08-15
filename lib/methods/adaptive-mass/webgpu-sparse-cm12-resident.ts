import type { SparseAtlasCompositeGrid } from "./sparse-atlas-composite-projection";
import { sparseAtlasScalarsHaveHorizontalD4Symmetry } from
  "./sparse-atlas-surface-conditioning";
import type { SparseAdaptiveMassAtlas } from "./sparse-brick-atlas";
import { CM12_SHARPENING_TRACE_STEPS } from "../../core/cm12-numerics";
import type { SphericalContainerFineGeometry } from "../../core/spherical-container";
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
import { webgpuSparseCM12ResidentWGSL } from "./webgpu-sparse-cm12-resident.wgsl";

/** CM12 Sec. 3.5 Algorithm 2's live trace bounds, in finest cells and substeps. */
export interface SharpeningTrace {
  readonly distanceCells?: number;
  readonly traceSteps?: number;
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
const PCG_ITERATIONS = 128;
const ACTIVITY_HEADER_WORDS = 12;
const ACTIVITY_RECORD_WORDS = 33;
const CANDIDATE_CELLS_PER_BRICK = 8 ** 3;
const CANDIDATE_CHANNELS = 12;

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
  readonly incidenceCount: number;
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
  readonly presentationBrickWet: number;
  readonly sharpeningDelta: number; readonly symmetryGamma: number;
}

export interface SparseCM12FinePresentationPlan {
  readonly plan: FineLevelSetBrickPlan;
  readonly metadata: Uint32Array;
  readonly worklist: Uint32Array;
}

export function sparseCM12FinePresentationPlan(
  atlas: SparseAdaptiveMassAtlas,
): SparseCM12FinePresentationPlan {
  const sampleDimensions = atlas.dimensions;
  const brickDimensions = sampleDimensions.map((value) => Math.ceil(value / 4)) as
    [number, number, number];
  const pages: { key: number; brick: number; octant: number }[] = [];
  for (let brick = 0; brick < atlas.bricks.length; brick += 1) {
    const source = atlas.bricks[brick]!;
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
          });
        }
  }
  pages.sort((left, right) => left.key - right.key);
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
      (pages[page]!.brick << 3) | pages[page]!.octant;
  }
  // Compact mode deliberately stops after the active physical-page list.  It
  // has no `logicalBrickCount`-sized direct directory; renderer lookup binary
  // searches the key-sorted metadata instead.
  const worklist = new Uint32Array(FINE_LEVELSET_WORKSET_HEADER_WORDS + pageCount);
  worklist.set([1, pageCount, pageCount,
    (FINE_LEVELSET_COMPACT_LOOKUP_FLAG | 3) >>> 0,
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

function residentStateLayout(
  cellCount: number,
  rowCount: number,
  brickCount: number,
): ResidentStateLayout {
  let at = 0;
  const cells = () => { const result = at; at += align4(cellCount); return result; };
  const rows = () => { const result = at; at += align4(rowCount); return result; };
  const cellVectors = () => { const result = at; at += align4(4 * cellCount); return result; };
  return {
    densityA: cells(), densityB: cells(), gammaA: cells(), gammaB: cells(),
    cellVelocityA: cellVectors(), cellVelocityB: cellVectors(),
    faceA: rows(), faceB: rows(), pressure: cells(), rhs: cells(), diagonal: cells(),
    liquid: cells(), theta: rows(), residual: cells(), preconditioned: cells(),
    direction: cells(), applied: cells(), divergence: cells(),
    presentationBrickWet: (() => { const result = at; at += align4(brickCount); return result; })(),
    sharpeningDelta: cells(), symmetryGamma: cells(),
    floatCount: at,
  };
}

function setF32(words: Uint32Array, index: number, value: number): void {
  new DataView(words.buffer).setFloat32(index * 4, value, true);
}

function packResidentTopology(
  atlas: SparseAdaptiveMassAtlas,
  grid: SparseAtlasCompositeGrid,
): PackedResidentTopology {
  let termCount = 0;
  for (const row of grid.gradientRows) termCount += row.terms.length;
  const byCell: { row: number; term: number }[][] = Array.from(
    { length: grid.cells.length }, () => [],
  );
  let nextTerm = 0;
  for (const row of grid.gradientRows) {
    for (const term of row.terms) {
      byCell[term.cellId]!.push({ row: row.id, term: nextTerm++ });
    }
  }
  const incidenceCount = byCell.reduce((sum, values) => sum + values.length, 0);
  const brickIndexByKey = new Map(atlas.bricks.map((brick, index) => [brick.key, index]));
  let at = 0;
  const cellOffset = at; at += 16 * grid.cells.length;
  const rowOffset = at; at += 12 * grid.gradientRows.length;
  const termOffset = at; at += 2 * termCount;
  const incidenceOffset = at; at += grid.cells.length + 1;
  const incidenceRecordOffset = at; at += 2 * incidenceCount;
  // Compact sorted key/index pairs replace both the finest-cell owner image
  // and the logical-domain-sized direct brick directory.  Every query derives
  // its owner from the immutable brick record after an O(log resident) lookup.
  const brickLookupOffset = at; at += 2 * atlas.bricks.length;
  const brickOffset = at; at += 4 * atlas.bricks.length;
  const backgroundOwnerOffset = at; at += 2;
  const words = new Uint32Array(at);

  for (const cell of grid.cells) {
    const base = cellOffset + 16 * cell.id;
    setF32(words, base, cell.centerFine[0]);
    setF32(words, base + 1, cell.centerFine[1]);
    setF32(words, base + 2, cell.centerFine[2]);
    setF32(words, base + 3, cell.volume);
    setF32(words, base + 4, cell.widthsFine[0]);
    setF32(words, base + 5, cell.widthsFine[1]);
    setF32(words, base + 6, cell.widthsFine[2]);
    words[base + 7] = cell.minimumFine[0];
    words[base + 8] = cell.minimumFine[1];
    words[base + 9] = cell.minimumFine[2];
    words[base + 10] = cell.widthsFine[0];
    const brickIndex = brickIndexByKey.get(cell.brickKey);
    if (brickIndex === undefined) throw new Error(`Sparse CM12 cell ${cell.id} has no brick`);
    words[base + 11] = brickIndex;
    setF32(words, base + 12, cell.openFraction);
    setF32(words, base + 13, cell.openVolume);
    words[base + 14] = cell.separatingPressureMinimum ? 1 : 0;
  }

  nextTerm = 0;
  const rowKinds = { "intra-brick": 0, "brick-face": 1, "mixed-seam": 2, "sparse-air": 3 } as const;
  for (const row of grid.gradientRows) {
    const base = rowOffset + 12 * row.id;
    words[base] = nextTerm;
    words[base + 1] = row.terms.length;
    words[base + 2] = row.axis;
    words[base + 3] = rowKinds[row.kind];
    setF32(words, base + 4, row.dualWeight);
    setF32(words, base + 5, row.area);
    setF32(words, base + 6, row.distance);
    setF32(words, base + 7, row.exteriorPhi ?? 0.5);
    setF32(words, base + 8, row.centerFine[0]);
    setF32(words, base + 9, row.centerFine[1]);
    setF32(words, base + 10, row.centerFine[2]);
    for (const term of row.terms) {
      words[termOffset + 2 * nextTerm] = term.cellId;
      setF32(words, termOffset + 2 * nextTerm + 1, term.coefficient);
      nextTerm += 1;
    }
  }

  let nextIncidence = 0;
  for (let cell = 0; cell < byCell.length; cell += 1) {
    words[incidenceOffset + cell] = nextIncidence;
    for (const incidence of byCell[cell]!) {
      words[incidenceRecordOffset + 2 * nextIncidence] = incidence.row;
      words[incidenceRecordOffset + 2 * nextIncidence + 1] = incidence.term;
      nextIncidence += 1;
    }
  }
  words[incidenceOffset + grid.cells.length] = nextIncidence;

  words[backgroundOwnerOffset] = 1 << 20;
  words[backgroundOwnerOffset + 1] = 1 << 10;
  const firstCellByBrick = new Uint32Array(atlas.bricks.length).fill(INVALID);
  const cellCountByBrick = new Uint32Array(atlas.bricks.length);
  for (const cell of grid.cells) {
    const brickIndex = brickIndexByKey.get(cell.brickKey);
    if (brickIndex === undefined) {
      throw new Error(`Sparse CM12 cell ${cell.id} has no resident brick`);
    }
    if (firstCellByBrick[brickIndex] === INVALID) firstCellByBrick[brickIndex] = cell.id;
    if (cell.id !== firstCellByBrick[brickIndex] + cellCountByBrick[brickIndex]) {
      throw new Error(`Sparse CM12 brick ${cell.brickKey} cells are not contiguous`);
    }
    cellCountByBrick[brickIndex] += 1;
  }
  const sortedBrickIndices = atlas.bricks.map((_, index) => index).sort((left, right) =>
    atlas.bricks[left]!.key - atlas.bricks[right]!.key);
  for (let sorted = 0; sorted < sortedBrickIndices.length; sorted += 1) {
    const brick = sortedBrickIndices[sorted]!;
    words[brickLookupOffset + 2 * sorted] = atlas.bricks[brick]!.key;
    words[brickLookupOffset + 2 * sorted + 1] = brick;
    const record = brickOffset + 4 * brick;
    words[record] = firstCellByBrick[brick];
    words[record + 1] = cellCountByBrick[brick];
    words[record + 2] = atlas.bricks[brick]!.resolution;
    words[record + 3] = atlas.bricks[brick]!.key;
  }
  return { words, cellOffset, rowOffset, termOffset, incidenceOffset,
    incidenceRecordOffset, brickLookupOffset, brickOffset, backgroundOwnerOffset,
    brickCount: atlas.bricks.length, incidenceCount };
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
  private readonly fineMetadata: GPUBuffer;
  private readonly fineWorklist: GPUBuffer;
  private readonly fineSamples: GPUBuffer;
  private readonly fineParams: GPUBuffer;
  private readonly fineWorkA: GPUBuffer;
  private readonly fineWorkB: GPUBuffer;
  private readonly fineRollback: GPUBuffer;
  readonly globalFineLevelSetSource: WebGPUFineLevelSetBrickSource;
  readonly sparseAdaptiveGridSource: SparseAdaptiveGridConsumerSource;
  private readonly diagnosticsReadback: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;
  private readonly parameterWords = new ArrayBuffer(352);
  private readonly parameterU32 = new Uint32Array(this.parameterWords);
  private readonly parameterF32 = new Float32Array(this.parameterWords);
  private parity = 0;
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly dimensions: readonly [number, number, number],
    private readonly layout: ResidentStateLayout,
    buffers: readonly [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer,
      GPUBuffer, GPUBuffer],
    fineBuffers: readonly [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer,
      GPUBuffer],
    finePlan: FineLevelSetBrickPlan,
    diagnosticsReadback: GPUBuffer,
    bindGroup: GPUBindGroup,
    pipelines: Readonly<Record<string, GPUComputePipeline>>,
    cellCount: number,
    rowCount: number,
    private horizontalD4Authority: boolean,
    private readonly boundary?: SphericalContainerFineGeometry,
  ) {
    [this.parameters, this.topology, this.state, this.partials, this.scalars,
      this.conditioning, this.activity, this.candidateState] = buffers;
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
      state: { buffer: this.state },
      activity: { buffer: this.activity },
      fineMetadata: { buffer: this.fineMetadata },
      fineWorklist: { buffer: this.fineWorklist },
      fineSamples: { buffer: this.fineSamples },
    };
    this.diagnosticsReadback = diagnosticsReadback;
    this.bindGroup = bindGroup;
    this.pipelines = pipelines;
    this.cellCount = cellCount;
    this.rowCount = rowCount;
    this.allocatedBytes = [...buffers, ...fineBuffers].reduce(
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
  ): Promise<WebGPUSparseCM12Resident> {
    const packed = packResidentTopology(atlas, grid);
    const fine = sparseCM12FinePresentationPlan(atlas);
    (fine.plan as { finestCellWidth: number; fineCellWidth: number }).finestCellWidth =
      finestCellSize_m;
    (fine.plan as { finestCellWidth: number; fineCellWidth: number }).fineCellWidth =
      finestCellSize_m;
    const layout = residentStateLayout(
      grid.cells.length, grid.gradientRows.length, packed.brickCount,
    );
    const initialState = new Float32Array(layout.floatCount);
    for (const cell of grid.cells) {
      initialState[layout.densityA + cell.id] = cell.density;
      initialState[layout.densityB + cell.id] = cell.density;
      initialState[layout.gammaA + cell.id] = cell.gamma;
      initialState[layout.gammaB + cell.id] = cell.gamma;
      initialState[layout.liquid + cell.id] = cell.density >= 0.5 ? 1 : 0;
    }
    const horizontalD4Authority = sparseAtlasScalarsHaveHorizontalD4Symmetry(
      grid,
      Float64Array.from(grid.cells, (cell) => cell.density),
      Float64Array.from(grid.cells, (cell) => cell.gamma),
    );
    const cellWorkgroups = Math.ceil(grid.cells.length / WORKGROUP_SIZE);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const parameters = device.createBuffer({ label: "Sparse CM12 resident parameters",
      size: 352, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const topology = uploadBuffer(device, "Sparse CM12 resident topology", packed.words, storage);
    const state = uploadBuffer(device, "Sparse CM12 resident state", initialState, storage);
    const partials = device.createBuffer({ label: "Sparse CM12 resident reductions",
      size: Math.max(8, 8 * cellWorkgroups), usage: storage });
    const scalars = device.createBuffer({ label: "Sparse CM12 resident scalar reductions",
      size: 32, usage: storage });
    const conditioning = device.createBuffer({
      label: "Sparse CM12 conservative transport and conditioning accumulators",
      size: Math.max(4, 4 * grid.cells.length * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const initialActivity = new Uint32Array(ACTIVITY_HEADER_WORDS
      + ACTIVITY_RECORD_WORDS * packed.brickCount);
    initialActivity[8] = initiallyActiveBrickKeys.size;
    for (let brick = 0; brick < packed.brickCount; brick += 1) {
      const at = ACTIVITY_HEADER_WORDS + ACTIVITY_RECORD_WORDS * brick;
      initialActivity[at + 8] = atlas.bricks[brick]!.resolution;
      initialActivity[at + 9] = 32; // retained until the first GPU topology epoch
      initialActivity[at + 10] = initiallyActiveBrickKeys.has(atlas.bricks[brick]!.key)
        ? 1 : 0;
      initialActivity[at + 12] = atlas.bricks[brick]!.resolution;
      initialActivity[at + 13] = atlas.bricks[brick]!.resolution;
      if (initialActivity[at + 10] !== 0) {
        initialActivity[11] += packed.words[packed.brickOffset + 4 * brick + 1]!;
      }
    }
    const activity = uploadBuffer(device, "Sparse CM12 resident activity history",
      initialActivity, storage);
    const candidateState = device.createBuffer({
      label: "Sparse CM12 isolated candidate cell fields",
      size: Math.max(4, 4 * CANDIDATE_CHANNELS * CANDIDATE_CELLS_PER_BRICK
        * packed.brickCount),
      usage: storage,
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
      size: 80,
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
      ] });
    const shaderModule = device.createShaderModule({ label: "Sparse CM12 resident shader",
      code: webgpuSparseCM12ResidentWGSL });
    const pipelineLayout = device.createPipelineLayout({ label: "Sparse CM12 resident pipeline layout",
      bindGroupLayouts: [bindGroupLayout] });
    const names = ["injectLiquid", "initializeTransportVelocity",
      "extrapolateTransportVelocityToSource", "extrapolateTransportVelocityToDestination",
      "prepareTransportFaces", "traceGammaAndBeta", "scatterDensityDeficit",
      "gatherConservativeDensity", "diffuseGammaForwardX", "diffuseGammaForwardY",
      "diffuseGammaForwardZ", "diffuseGammaReverseZ", "diffuseGammaReverseY",
      "diffuseGammaReverseX", "averageGammaDiffusion", "scatterSharpeningMass",
      "finalizeSharpening", "clearSolidExcess", "scatterSolidExcess",
      "finalizeSolidExcess", "preserveHorizontalD4",
      "commitHorizontalD4",
      "forceFaces", "classifyRows", "preparePressure",
      "initializePCG", "reduceInitialize", "applyDirection", "reduceCurvature",
      "updateResidual", "reduceResidual", "updateDirection",
      "projectedJacobiToApplied", "projectedJacobiToPressure", "projectFaces",
      "collocateAndDiagnose", "measureDivergenceDiagnostics",
      "reduceDivergenceDiagnostics",
      "advanceActivityClock", "measureBrickActivity",
      "planBrickResolution", "activateSweptReceivers", "closePlannedResolution",
      "validateCandidateResolution", "transferCandidateCells",
      "prepareCandidateFaceReceipts", "transferCandidateFaces",
      "retireUnsupportedEmptyBricks",
      "classifyPresentationBricks",
      "publishSparseLevelSet"] as const;
    const entries = await Promise.all(names.map(async (name) => [name,
      await device.createComputePipelineAsync({ label: `Sparse CM12 ${name}`,
        layout: pipelineLayout, compute: { module: shaderModule, entryPoint: name } })] as const));
    const result = new WebGPUSparseCM12Resident(device, atlas.dimensions, layout,
      [parameters, topology, state, partials, scalars, conditioning, activity,
        candidateState],
      [fineParams, fineMetadata, fineWorklist, fineSamples, fineWorkA, fineWorkB,
        fineRollback],
      fine.plan,
      diagnosticsReadback,
      bindGroup,
      Object.fromEntries(entries), grid.cells.length, grid.gradientRows.length,
      horizontalD4Authority, atlas.boundary);
    result.writeParameters(packed, 0.004, 1, 1, [0, 0, 0]);
    return result;
  }

  encode(
    encoder: GPUCommandEncoder,
    dt_s: number,
    finestCellSize_m: number,
    pressureScale: number,
    accelerationFinePerSecond2: readonly [number, number, number],
    sharpening?: SharpeningTrace,
    seams?: SparseCM12ResidentStageSeams,
  ): void {
    this.assertLive();
    const packed = this.lastPacked!;
    this.writeParameters(packed, dt_s, finestCellSize_m, pressureScale,
      accelerationFinePerSecond2, sharpening);
    encoder.clearBuffer(this.conditioning);
    // The pass opens on first dispatch rather than up front. A stage that
    // encodes nothing this advance — the D4 authority on an asymmetric scene —
    // then leaves no empty pass behind, which matters because Metal writes no
    // timestamp for a pass that does no work and one unsampled boundary
    // rejects the whole chain.
    let pass: GPUComputePassEncoder | undefined;
    let passLabel = "Sparse CM12 resident frame";
    const dispatch = (name: string, count: number, y = 1, z = 1) => {
      pass ??= encoder.beginComputePass({ label: passLabel });
      pass.setPipeline(this.pipelines[name]!); pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(count, y, z);
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
    const cells = Math.ceil(this.cellCount / WORKGROUP_SIZE);
    const rows = Math.ceil(this.rowCount / WORKGROUP_SIZE);
    const bricks = Math.ceil(packed.brickCount / WORKGROUP_SIZE);
    stage("transport-velocity-extension", () => {
      dispatch("initializeTransportVelocity", cells);
      for (let sweep = 0; sweep < 8; sweep += 1) {
        dispatch(sweep % 2 === 0
          ? "extrapolateTransportVelocityToSource"
          : "extrapolateTransportVelocityToDestination", cells);
      }
    });
    stage("face-preparation", () => {
      dispatch("prepareTransportFaces", rows);
    });
    stage("conservative-transport", () => {
      dispatch("traceGammaAndBeta", cells);
      dispatch("scatterDensityDeficit", cells);
      dispatch("gatherConservativeDensity", cells);
    });
    stage("gamma-diffusion", () => {
      dispatch("diffuseGammaForwardX", cells);
      dispatch("diffuseGammaForwardY", cells);
      dispatch("diffuseGammaForwardZ", cells);
      dispatch("diffuseGammaReverseZ", cells);
      dispatch("diffuseGammaReverseY", cells);
      dispatch("diffuseGammaReverseX", cells);
      dispatch("averageGammaDiffusion", cells);
    });
    stage("surface-sharpening", () => {
      dispatch("scatterSharpeningMass", cells);
      dispatch("finalizeSharpening", cells);
      if (this.boundary) {
        dispatch("clearSolidExcess", cells);
        dispatch("scatterSolidExcess", cells);
        dispatch("finalizeSolidExcess", cells);
      }
    });
    stage("symmetry-authority", () => {
      if (!this.horizontalD4Authority) return;
      dispatch("preserveHorizontalD4", cells);
      dispatch("commitHorizontalD4", cells);
    });
    stage("body-forces", () => {
      dispatch("forceFaces", rows);
    });
    stage("pressure-topology", () => {
      dispatch("preparePressure", cells);
      dispatch("classifyRows", rows);
      dispatch("preparePressure", cells);
    });
    stage("pressure-rhs", () => {
      dispatch("initializePCG", cells);
      dispatch("reduceInitialize", 1);
    });
    stage("pressure-solve", () => {
      if (this.boundary) {
        // Projected Jacobi solves the cut-boundary LCP. Pressure samples whose
        // centres lie in solid enforce p >= 0; unconstrained cells retain the
        // ordinary free-surface equation. Two kernels provide the ping-pong
        // image without allocating another persistent pressure field.
        for (let iteration = 0; iteration < PCG_ITERATIONS / 2; iteration += 1) {
          dispatch("projectedJacobiToApplied", cells);
          dispatch("projectedJacobiToPressure", cells);
        }
      } else {
        for (let iteration = 0; iteration < PCG_ITERATIONS; iteration += 1) {
          dispatch("applyDirection", cells);
          dispatch("reduceCurvature", 1);
          dispatch("updateResidual", cells);
          dispatch("reduceResidual", 1);
          dispatch("updateDirection", cells);
        }
      }
    });
    stage("velocity-projection", () => {
      dispatch("projectFaces", rows);
      dispatch("collocateAndDiagnose", cells);
    });
    stage("projection-diagnostics", () => {
      dispatch("measureDivergenceDiagnostics", cells);
      dispatch("reduceDivergenceDiagnostics", 1);
    });
    stage("activity-measurement", () => {
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
    });
    stage("candidate-transfer", () => {
      dispatch("transferCandidateCells", packed.brickCount);
      dispatch("prepareCandidateFaceReceipts", bricks);
      dispatch("transferCandidateFaces", packed.brickCount, 6);
    });
    stage("brick-retirement", () => {
      dispatch("retireUnsupportedEmptyBricks", bricks);
    });
    stage("presentation-publication", () => {
      dispatch("classifyPresentationBricks", bricks);
      dispatch("publishSparseLevelSet",
        this.globalFineLevelSetSource.plan.maximumResidentBricks);
    });
    pass?.end();
    this.parity ^= 1;
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
    this.parameterF32.set([...centerFine, 0], 52);
    this.parameterF32.set([...radiusFine, 0], 56);
    this.device.queue.writeBuffer(this.parameters, 0, this.parameterWords);
    const pass = encoder.beginComputePass({ label: "Sparse CM12 resident liquid injection" });
    pass.setPipeline(this.pipelines.injectLiquid!);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.cellCount / WORKGROUP_SIZE));
    pass.setPipeline(this.pipelines.classifyPresentationBricks!);
    pass.dispatchWorkgroups(Math.ceil(this.lastPacked!.brickCount / WORKGROUP_SIZE));
    pass.setPipeline(this.pipelines.publishSparseLevelSet!);
    pass.dispatchWorkgroups(this.globalFineLevelSetSource.plan.maximumResidentBricks);
    pass.end();
  }

  private lastPacked?: PackedResidentTopology;

  private writeParameters(
    packed: PackedResidentTopology,
    dt_s: number,
    finestCellSize_m: number,
    pressureScale: number,
    acceleration: readonly [number, number, number],
    sharpening?: SharpeningTrace,
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
    u.set([l.applied, l.divergence, l.presentationBrickWet, 0], 32);
    // The D4 pass needs two disjoint scalar scratch arrays. In particular the
    // gamma scratch must never alias densityA at offset zero: doing so corrupts
    // gamma after the first symmetric frame and makes transport create mass on
    // the next frame.
    u.set([l.sharpeningDelta, l.symmetryGamma, 0, 0], 36);
    f.set([dt_s, finestCellSize_m, pressureScale, this.parity], 40);
    f.set([...acceleration, 0], 44);
    u.set([Math.ceil(this.cellCount / WORKGROUP_SIZE),
      Math.ceil(this.rowCount / WORKGROUP_SIZE), PCG_ITERATIONS, packed.brickCount], 48);
    f.set([0, 0, 0, 0, 1, 1, 1, 0], 52);
    // CM12 Algorithm 2's trace bounds. Direct diagnostic constructors pass no
    // controls and get the paper's own values, so an unparameterized probe and
    // a default panel run are the same simulation.
    f.set([
      sparseCM12SharpeningDistance(sharpening?.distanceCells),
      sparseCM12SharpeningTraceSteps(sharpening?.traceSteps),
      0, 0,
    ], 60);
    f.set(this.boundary ? [...this.boundary.centerFine, 0] : [0, 0, 0, 0], 64);
    f.set(this.boundary ? [...this.boundary.radiiFine, 0] : [1, 1, 1, 0], 68);
    this.device.queue.writeBuffer(this.parameters, 0, this.parameterWords);
  }

  async readDiagnostics(): Promise<{
    readonly pressureRelativeResidual: number;
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
  }> {
    this.assertLive();
    const encoder = this.device.createCommandEncoder({
      label: "Sparse CM12 diagnostic scalar readback",
    });
    encoder.copyBufferToBuffer(this.scalars, 0, this.diagnosticsReadback, 0, 32);
    encoder.copyBufferToBuffer(this.activity, 0, this.diagnosticsReadback, 32, 48);
    this.device.queue.submit([encoder.finish()]);
    await this.diagnosticsReadback.mapAsync(GPUMapMode.READ);
    const mapped = this.diagnosticsReadback.getMappedRange();
    const values = new Float32Array(mapped, 0, 8);
    const activity = new Uint32Array(mapped, 32, ACTIVITY_HEADER_WORDS);
    const rhsSquared = values[1]!;
    const residualSquared = values[4]!;
    const result = {
      pressureRelativeResidual: Math.sqrt(Math.max(0, residualSquared)
        / Math.max(rhsSquared, Number.MIN_VALUE)),
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
    };
    this.diagnosticsReadback.unmap();
    return result;
  }

  /** QA-only dense readback used by Dawn physics gates. Never called by frame
   * scheduling or visualization; callers explicitly pay for materialization. */
  async readDiagnosticFields(): Promise<SparseCM12DiagnosticFields> {
    this.assertLive();
    const packed = this.lastPacked!;
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
      const topologyFloats = new Float32Array(packed.words.buffer);
      for (let cell = 0; cell < this.cellCount; cell += 1) {
        const base = packed.cellOffset + 16 * cell;
        const brick = packed.words[base + 11]!;
        if (!(state[this.layout.presentationBrickWet + brick]! > 0.5)) continue;
        const lower = [packed.words[base + 7]!, packed.words[base + 8]!,
          packed.words[base + 9]!] as const;
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
          hotEpochs: words[at + 2]!,
          quietEpochs: words[at + 3]!,
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
    for (const buffer of [this.parameters, this.topology, this.state, this.partials,
      this.scalars, this.conditioning, this.activity, this.candidateState,
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

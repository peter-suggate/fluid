/**
 * Sparse CM12 temporal-coherence publications and field views.
 *
 * These modes deliberately use the generic grid overlay: the dirty records
 * are indexed by stable logical 4³ work tiles in the same finest-domain frame
 * as Grid structure, so the existing slice/volume controls are the honest way
 * to inspect them. The solver may omit the optional source while bringing the
 * producer up; the shader renders that state as UNKNOWN rather than as clean.
 */
import { fieldVisualization, type Visualization } from "./visualization-registry";

export const SPARSE_CM12_DIRTY_GPU_MAGIC = 0x434d_4431;
export const SPARSE_CM12_DIRTY_GPU_VERSION = 1;
export const SPARSE_CM12_DIRTY_GPU_HEADER_WORDS = 16;
export const SPARSE_CM12_DIRTY_GPU_RECORD_WORDS = 16;
export const SPARSE_CM12_DIRTY_TILE_EDGE = 4;

/** Optional CMD1 v1 tail describing which logical stages share physical passes. */
export const SPARSE_CM12_DIRTY_PACKING_GPU_MAGIC = 0x504b_5431; // "PKT1"
export const SPARSE_CM12_DIRTY_PACKING_GPU_VERSION = 1;
export const SPARSE_CM12_DIRTY_PACKING_GPU_HEADER_WORDS = 16;
export const SPARSE_CM12_DIRTY_PACKING_GPU_PACKET_WORDS = 4;
export const SPARSE_CM12_DIRTY_PACKING_GPU_TILE_WORDS = 4;
export const SPARSE_CM12_DIRTY_PACKING_MAX_PACKETS = 32;

export const SPARSE_CM12_DIRTY_GPU_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, recordWords: 3,
  tileEdge: 4, tilesPerLogicalBrickAxis: 5, logicalBrickCount: 6, stageCount: 7,
  acceptedGeneration: 8, candidateGeneration: 9, provenanceGeneration: 10,
  publicationFlags: 11, uncoveredWriteCount: 12,
  firstFaultLogicalBrick: 13, firstFaultLocalTile: 14, firstFaultStage: 15,
} as const);

export const SPARSE_CM12_DIRTY_GPU_RECORD = Object.freeze({
  directStageMask: 0,
  closureStageMask: 1,
  processedStageMask: 2,
  skippedStageMask: 3,
  unknownStageMask: 4,
  uncoveredWriteStageMask: 5,
  originCauseMask: 6,
  inheritedCauseMask: 7,
  packedClosureDepths: 8,
  packedOriginStages: 9,
  stageGenerationPairs: 10,
} as const);

export const SPARSE_CM12_DIRTY_PACKING_GPU_HEADER = Object.freeze({
  magic: 0, version: 1, headerWords: 2, packetWords: 3, tileWords: 4,
  packetCount: 5, tileCount: 6, epochCount: 7,
  acceptedGeneration: 8, candidateGeneration: 9, provenanceGeneration: 10,
  publicationFlags: 11, uncoveredPackingFaultCount: 12,
  firstFaultTile: 13, firstFaultPacket: 14, reserved: 15,
} as const);

export const SPARSE_CM12_DIRTY_PACKING_GPU_PACKET = Object.freeze({
  identity: 0, epoch: 1, logicalStageMask: 2, executionGeneration: 3,
} as const);

export const SPARSE_CM12_DIRTY_PACKING_GPU_TILE = Object.freeze({
  assignedPacketMask: 0,
  executedPacketMask: 1,
  /** Five 6-bit lanes for stages 0..4: zero skipped, 1..32 packet index + 1. */
  packedStagePacketIndicesLow: 2,
  /** Stage 5 uses the low 6 bits; the remaining bits are reserved and zero. */
  packedStagePacketIndexHigh: 3,
} as const);

export function sparseCM12DirtyTilesPerLogicalBrickAxis(brickFineResolution: number): number {
  if (!Number.isSafeInteger(brickFineResolution) || brickFineResolution < SPARSE_CM12_DIRTY_TILE_EDGE
    || brickFineResolution % SPARSE_CM12_DIRTY_TILE_EDGE !== 0) {
    throw new RangeError("Sparse CM12 dirty tiles require a brick resolution divisible by 4");
  }
  return brickFineResolution / SPARSE_CM12_DIRTY_TILE_EDGE;
}

export function sparseCM12DirtyGPUBufferWords(
  logicalBrickCount: number,
  brickFineResolution: number,
): number {
  if (!Number.isSafeInteger(logicalBrickCount) || logicalBrickCount < 0) {
    throw new RangeError("Sparse CM12 dirty logical-brick count must be non-negative");
  }
  const tilesPerAxis = sparseCM12DirtyTilesPerLogicalBrickAxis(brickFineResolution);
  return SPARSE_CM12_DIRTY_GPU_HEADER_WORDS
    + logicalBrickCount * tilesPerAxis ** 3 * SPARSE_CM12_DIRTY_GPU_RECORD_WORDS;
}

/** Total words for CMD1 records plus the optional fixed-layout PKT1 tail. */
export function sparseCM12DirtyPackingGPUBufferWords(
  logicalBrickCount: number,
  brickFineResolution: number,
  packetCount: number,
): number {
  if (!Number.isSafeInteger(packetCount) || packetCount < 0
    || packetCount > SPARSE_CM12_DIRTY_PACKING_MAX_PACKETS) {
    throw new RangeError("Sparse CM12 dirty pass packing supports 0..32 packets");
  }
  const tilesPerAxis = sparseCM12DirtyTilesPerLogicalBrickAxis(brickFineResolution);
  const tileCount = logicalBrickCount * tilesPerAxis ** 3;
  return sparseCM12DirtyGPUBufferWords(logicalBrickCount, brickFineResolution)
    + SPARSE_CM12_DIRTY_PACKING_GPU_HEADER_WORDS
    + packetCount * SPARSE_CM12_DIRTY_PACKING_GPU_PACKET_WORDS
    + tileCount * SPARSE_CM12_DIRTY_PACKING_GPU_TILE_WORDS;
}

export const SPARSE_CM12_DIRTY_STAGE = Object.freeze({
  facePreparation: 0,
  massTransport: 1,
  gammaTransport: 2,
  surfaceConditioning: 3,
  pressureCoefficients: 4,
  presentation: 5,
} as const);
export const SPARSE_CM12_DIRTY_STAGE_COUNT = 6;

export const SPARSE_CM12_DIRTY_CAUSE_BIT = Object.freeze({
  topologyCreated: 1 << 0,
  topologyRetired: 1 << 1,
  phaseCrossing: 1 << 2,
  densityChanged: 1 << 3,
  gammaChanged: 1 << 4,
  velocityCharacteristic: 1 << 5,
  cflGrowth: 1 << 6,
  movingSolidSweep: 1 << 7,
  boundarySource: 1 << 8,
  coefficientChanged: 1 << 9,
  dependencyClosure: 1 << 10,
  generationMismatch: 1 << 11,
  capacityOrProvenance: 1 << 12,
  pageActivated: 1 << 13,
  pageRetired: 1 << 14,
} as const);

export const SPARSE_CM12_DIRTY_PUBLICATION_FLAG = Object.freeze({
  complete: 1 << 0,
  accepted: 1 << 1,
  rejected: 1 << 2,
  capacityOrProvenanceFault: 1 << 3,
} as const);

/**
 * Header words followed by one 16-word record for every logical base-brick
 * 4³ tile. Logical identity, not a mutable physical slot, keeps the picture
 * stable through coarsening and refinement.
 *
 * Header:
 *  0 magic, 1 version, 2 header words, 3 record words,
 *  4 tile edge, 5 tiles/base-brick axis, 6 logical brick count, 7 stage count,
 *  8 accepted generation, 9 candidate generation, 10 provenance generation,
 * 11 publication flags, 12 uncovered-write count,
 * 13 first-fault logical brick, 14 first-fault local tile, 15 first-fault stage.
 *
 * Record:
 *  0 direct stages, 1 closure stages, 2 processed/executed stages,
 *  3 skipped/reused stages, 4 unknown stages, 5 uncovered-write stages,
 *  6 origin cause mask, 7 inherited cause mask,
 *  8 closure depths (six 4-bit lanes), 9 origin stages (six 3-bit lanes),
 * 10..15 producer generation in low 16 bits and consumer in high 16 bits.
 */
export interface SparseCM12DirtyGPUHeader {
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly provenanceGeneration: number;
  readonly publicationFlags: number;
  readonly uncoveredWriteCount: number;
}

/** Optional PKT1 header appended after all v1 CMD1 tile records. */
export interface SparseCM12DirtyPackingGPUHeader {
  readonly packetCount: number;
  readonly tileCount: number;
  readonly epochCount: number;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly provenanceGeneration: number;
  readonly publicationFlags: number;
  readonly uncoveredPackingFaultCount: number;
}

export const SPARSE_CM12_DIRTY_OVERLAY_MODES = [
  "dirty-face-preparation",
  "dirty-mass-transport",
  "dirty-gamma-transport",
  "dirty-surface-conditioning",
  "dirty-pressure-coefficients",
  "dirty-presentation",
  "dirty-causes",
  "dirty-closure-depth",
  "dirty-generations",
  "dirty-pass-packing",
] as const;
export type SparseCM12DirtyOverlayMode = typeof SPARSE_CM12_DIRTY_OVERLAY_MODES[number];

export function isSparseCM12DirtyOverlayMode(value: unknown): value is SparseCM12DirtyOverlayMode {
  return typeof value === "string"
    && (SPARSE_CM12_DIRTY_OVERLAY_MODES as readonly string[]).includes(value);
}

/** Generic grid-overlay mode codes; 1..10 are the pre-existing field modes. */
export const SPARSE_CM12_DIRTY_OVERLAY_CODES: Readonly<Record<SparseCM12DirtyOverlayMode, number>> =
  Object.freeze({
    "dirty-face-preparation": 11,
    "dirty-mass-transport": 12,
    "dirty-gamma-transport": 13,
    "dirty-surface-conditioning": 14,
    "dirty-pressure-coefficients": 15,
    "dirty-presentation": 16,
    "dirty-causes": 17,
    "dirty-closure-depth": 18,
    "dirty-generations": 19,
    "dirty-pass-packing": 20,
  });

export function sparseCM12DirtyOverlayCode(value: unknown): number {
  return isSparseCM12DirtyOverlayMode(value) ? SPARSE_CM12_DIRTY_OVERLAY_CODES[value] : 0;
}

const stageLegend = [
  { swatch: "#f5a524", label: "direct origin" },
  { swatch: "#2979ff", label: "dependency closure" },
  { swatch: "#174f3b", label: "proven reused / skipped" },
  { swatch: "#ff2fa4", label: "unknown, stale, or uncovered write" },
] as const;

function stageView(input: {
  readonly id: string;
  readonly label: string;
  readonly description: string;
  readonly mode: SparseCM12DirtyOverlayMode;
}): Visualization {
  return fieldVisualization({
    kind: "field", id: input.id, pass: "Sparse CM12 coherence",
    figure: "4³", label: input.label, description: input.description,
    source: "GPU-resident FPL1 FramePlan, with CMD1 compatibility fallback; no readback",
    mode: input.mode, axis: "volume", legend: stageLegend,
  });
}

export const sparseCM12DirtyVisualizations: readonly Visualization[] = Object.freeze([
  stageView({
    id: "sparse-cm12-dirty/face-preparation", label: "Face preparation dirt",
    description: "Direct and closure 4³ tiles feeding oriented faces, characteristic donors, solids, and boundary sources.",
    mode: "dirty-face-preparation",
  }),
  stageView({
    id: "sparse-cm12-dirty/mass-transport", label: "Mass transport dirt",
    description: "Density origins and conservative donor/receiver closure actually scheduled for transport.",
    mode: "dirty-mass-transport",
  }),
  stageView({
    id: "sparse-cm12-dirty/gamma-transport", label: "Gamma transport dirt",
    description: "Gamma support, inherited mass events, diffusion neighbours, and conservation partners.",
    mode: "dirty-gamma-transport",
  }),
  stageView({
    id: "sparse-cm12-dirty/surface-conditioning", label: "Surface conditioning dirt",
    description: "Transported phase changes and the sharpening, scatter, and local-return closure they induce.",
    mode: "dirty-surface-conditioning",
  }),
  stageView({
    id: "sparse-cm12-dirty/pressure-coefficients", label: "Pressure coefficient dirt",
    description: "Changed liquid rows, edges, aggregates, and hierarchy ancestors—not global pressure-solution influence.",
    mode: "dirty-pressure-coefficients",
  }),
  stageView({
    id: "sparse-cm12-dirty/presentation", label: "Presentation dirt",
    description: "Conditioned-density and topology changes through the exact 4³ subtile filter/restriction footprint.",
    mode: "dirty-presentation",
  }),
  fieldVisualization({
    kind: "field", id: "sparse-cm12-dirty/causes", pass: "Sparse CM12 coherence",
    figure: "WHY", label: "Dirty origins",
    description: "Physical/structural origin causes, with inherited dependency causes retained rather than relabelled downstream.",
    source: "FPL1 tile cause masks, with CMD1 compatibility fallback",
    mode: "dirty-causes", axis: "volume",
    legend: [
      { swatch: "#f5f5e6", label: "topology / coefficient" },
      { swatch: "#f5d442", label: "density / gamma / phase" },
      { swatch: "#14c7d9", label: "velocity / CFL / moving solid" },
      { swatch: "#f28b30", label: "boundary source" },
      { swatch: "#ff2fa4", label: "generation / capacity / unknown" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "sparse-cm12-dirty/closure-depth", pass: "Sparse CM12 coherence",
    figure: "DEPTH", label: "Dirty closure depth",
    description: "Maximum declared dilation/dependency depth from a direct origin across all six stage worksets.",
    source: "FPL1 packed per-stage closure depths, with CMD1 compatibility fallback",
    mode: "dirty-closure-depth", axis: "volume",
    legend: [
      { swatch: "#ffffff", label: "depth 0 — direct" },
      { swatch: "linear-gradient(90deg,#44b9ff,#3153d6,#7b32bd)", label: "closure depth 1 → 15" },
      { swatch: "#174f3b", label: "proven reused" },
      { swatch: "#ff2fa4", label: "unknown / overflow" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "sparse-cm12-dirty/generations", pass: "Sparse CM12 coherence",
    figure: "GEN", label: "Dirty generation coherence",
    description: "Producer/consumer stamps against accepted, candidate, and provenance generations; mixed state is an alarm.",
    source: "FPL1 frame/brick/tile generations, with CMD1 per-stage compatibility fallback",
    mode: "dirty-generations", axis: "volume",
    legend: [
      { swatch: "#32c982", label: "coherent producer → consumer chain" },
      { swatch: "#2aa7dc", label: "candidate-generation work" },
      { swatch: "#ff2fa4", label: "stale, incomplete, rejected, or unavailable" },
      { swatch: "#ff312e", label: "uncovered write" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "sparse-cm12-dirty/pass-packing", pass: "Sparse CM12 coherence",
    figure: "PACK", label: "Dirty pass packing",
    description: "Physical dispatch packets and epochs over the logical stage masks they coalesce; equal saturated colours share one pass.",
    source: "FPL1 physical packet assignments, with PKT1 compatibility fallback",
    mode: "dirty-pass-packing", axis: "volume",
    legend: [
      { swatch: "linear-gradient(90deg,#20c7b7,#4279e8,#b75be8,#ef776f)", label: "physical packet / epoch identity" },
      { swatch: "#36cdb7", label: "saturated — two or more logical stages share the packet" },
      { swatch: "#74879a", label: "muted — one logical stage in the packet" },
      { swatch: "#174f3b", label: "all logical stages proven skipped" },
      { swatch: "#ff2fa4", label: "missing, stale, invalid, or local brick fault" },
      { swatch: "#ff312e", label: "invalid physical packet mapping or packet capacity" },
    ],
  }),
]);

import type {
  SparseAdaptiveMassBrick,
  SparseBrickResolution,
  SparseBrickVec3,
} from "./sparse-brick-atlas";

/** TCP1: sparse coordinate directory and physical-page allocator. */
export const SPARSE_CM12_TILE_CLONE_POOL_MAGIC = 0x5443_5031;
export const SPARSE_CM12_TILE_CLONE_POOL_VERSION = 1;
export const SPARSE_CM12_TILE_CLONE_POOL_INVALID = 0xffff_ffff;
export const SPARSE_CM12_TILE_CLONE_POOL_HEADER_WORDS = 32;
export const SPARSE_CM12_TILE_CLONE_POOL_RECORD_WORDS = 32;
export const SPARSE_CM12_TILE_CLONE_POOL_HASH_WORDS = 1;
export const SPARSE_CM12_TILE_CLONE_POOL_NEIGHBOR_COUNT = 6;
export const SPARSE_CM12_TILE_CLONE_POOL_BOUNDARY_PROGRAM_COUNT = 6;
export const SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY = 8 ** 3;
/** Three oriented face lattices, including both exterior planes. */
export const SPARSE_CM12_TILE_CLONE_POOL_B8_FACE_CAPACITY = 3 * 9 * 8 * 8;
export const SPARSE_CM12_TILE_CLONE_POOL_CELL_FIELD_PLANES = 9;
export const SPARSE_CM12_TILE_CLONE_POOL_FACE_FIELD_PLANES = 2;
export const SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT_WORDS = 16;

export const SPARSE_CM12_TILE_CLONE_POOL_CELL_FIELD = Object.freeze({
  densityA: 0,
  densityB: 1,
  gammaA: 2,
  gammaB: 3,
  velocityX: 4,
  velocityY: 5,
  velocityZ: 6,
  pressure: 7,
  divergence: 8,
} as const);

export const SPARSE_CM12_TILE_CLONE_POOL_FACE_FIELD = Object.freeze({
  velocity: 0,
  openFraction: 1,
} as const);

export const SPARSE_CM12_TILE_CLONE_POOL_BOUNDARY_PROGRAM = Object.freeze({
  sparseAir: 1,
  sameRung: 2,
  coarseToFine: 3,
  fineToCoarse: 4,
  wall: 5,
} as const);

export const SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT = Object.freeze({
  generation: 0,
  flags: 1,
  resolution: 2,
  cellCount: 3,
  faceCount: 4,
  incidenceCount: 5,
  internalPressureEdgeCount: 6,
  boundaryProgramCount: 7,
  cellFieldPage: 8,
  faceFieldPage: 9,
  pressureAggregate: 10,
  presentationPage: 11,
} as const);

export const SPARSE_CM12_TILE_CLONE_POOL_PAGE_FLAG = Object.freeze({
  fieldsInitialized: 1 << 0,
  topologyComplete: 1 << 1,
  pressureComplete: 1 << 2,
  boundaryComplete: 1 << 3,
  presentationComplete: 1 << 4,
  accepted: 1 << 5,
} as const);

export const SPARSE_CM12_TILE_CLONE_POOL_PAGE_COMPLETE_FLAGS =
  SPARSE_CM12_TILE_CLONE_POOL_PAGE_FLAG.fieldsInitialized
  | SPARSE_CM12_TILE_CLONE_POOL_PAGE_FLAG.topologyComplete
  | SPARSE_CM12_TILE_CLONE_POOL_PAGE_FLAG.pressureComplete
  | SPARSE_CM12_TILE_CLONE_POOL_PAGE_FLAG.boundaryComplete
  | SPARSE_CM12_TILE_CLONE_POOL_PAGE_FLAG.presentationComplete
  | SPARSE_CM12_TILE_CLONE_POOL_PAGE_FLAG.accepted;

export const SPARSE_CM12_TILE_CLONE_POOL_FLAG = Object.freeze({
  complete: 1 << 0,
  validated: 1 << 1,
} as const);

export const SPARSE_CM12_TILE_CLONE_POOL_TILE_FLAG = Object.freeze({
  resident: 1 << 0,
  accepted: 1 << 1,
  authoredFluid: 1 << 2,
  candidate: 1 << 3,
} as const);

export const SPARSE_CM12_TILE_CLONE_POOL_HEADER = Object.freeze({
  magic: 0,
  version: 1,
  headerWords: 2,
  recordWords: 3,
  capacity: 4,
  residentCount: 5,
  highWaterMark: 6,
  freeCount: 7,
  acceptedGeneration: 8,
  candidateGeneration: 9,
  flags: 10,
  hashCapacity: 11,
  hashBase: 12,
  recordBase: 13,
  freeListBase: 14,
  totalWords: 15,
  cellPageStride: 16,
  facePageStride: 17,
  cloneCount: 18,
  retiredCount: 19,
  faultCount: 20,
  firstFaultSlot: 21,
} as const);

export const SPARSE_CM12_TILE_CLONE_POOL_RECORD = Object.freeze({
  generation: 0,
  flags: 1,
  coordinateX: 2,
  coordinateY: 3,
  coordinateZ: 4,
  spanLog2: 5,
  rung: 6,
  fieldPage: 7,
  presentationPage: 8,
  pressureAggregate: 9,
  neighborBase: 10,
  boundaryProgramBase: 16,
  parentAggregate: 22,
  directorySlot: 23,
  lifecycleGeneration: 24,
  sourceSlot: 25,
} as const);

export interface SparseCM12TileClonePoolLayout {
  readonly capacity: number;
  readonly hashCapacity: number;
  readonly hashBaseWords: number;
  readonly recordBaseWords: number;
  readonly freeListBaseWords: number;
  readonly totalWords: number;
  readonly metadataBytes: number;
  readonly cellFieldFloats: number;
  readonly faceFieldFloats: number;
  readonly pageReceiptWords: number;
  readonly totalBytes: number;
  readonly cellPageCapacity: number;
  readonly facePageCapacity: number;
}

export interface SparseCM12TileClonePool {
  readonly layout: SparseCM12TileClonePoolLayout;
  readonly words: Uint32Array;
  /** Capacity-shaped physical slabs. Only pages with a complete receipt are resident. */
  readonly cellFields: Float32Array;
  readonly faceFields: Float32Array;
  readonly pageReceipts: Uint32Array;
}

export interface SparseCM12TileCloneSeed {
  readonly coordinate: SparseBrickVec3;
  readonly resolution: SparseBrickResolution;
  readonly spanBricks?: number;
  readonly authoredFluid?: boolean;
  readonly density?: ArrayLike<number>;
  readonly gamma?: ArrayLike<number>;
}

const checkedCount = (value: number, label: string, positive = false): number => {
  if (!Number.isSafeInteger(value) || value < (positive ? 1 : 0)
    || value >= SPARSE_CM12_TILE_CLONE_POOL_INVALID) {
    throw new RangeError(`${label} is outside the addressable u32 range`);
  }
  return value;
};

const nextPowerOfTwo = (value: number): number => {
  let result = 1;
  while (result < value) result *= 2;
  return result;
};

/**
 * The first physical slab is working-set-shaped. WebGPU buffers cannot grow in
 * place, so later cutover phases may chain slabs; they must not replace this
 * calculation with logical-domain volume.
 */
export function sparseCM12TileClonePoolCapacity(
  authoredFluidTileCount: number,
  minimum = 1024,
  headroomFactor = 2,
): number {
  checkedCount(authoredFluidTileCount, "authoredFluidTileCount");
  checkedCount(minimum, "minimum", true);
  if (!Number.isFinite(headroomFactor) || headroomFactor < 1) {
    throw new RangeError("headroomFactor must be finite and at least one");
  }
  return checkedCount(nextPowerOfTwo(Math.max(minimum,
    Math.ceil(authoredFluidTileCount * headroomFactor))), "tile clone capacity", true);
}

export function createSparseCM12TileClonePoolLayout(
  capacity: number,
): SparseCM12TileClonePoolLayout {
  checkedCount(capacity, "capacity", true);
  const hashCapacity = checkedCount(nextPowerOfTwo(2 * capacity),
    "hashCapacity", true);
  const hashBaseWords = SPARSE_CM12_TILE_CLONE_POOL_HEADER_WORDS;
  const recordBaseWords = hashBaseWords
    + SPARSE_CM12_TILE_CLONE_POOL_HASH_WORDS * hashCapacity;
  const freeListBaseWords = recordBaseWords
    + SPARSE_CM12_TILE_CLONE_POOL_RECORD_WORDS * capacity;
  const totalWords = checkedCount(freeListBaseWords + capacity, "totalWords", true);
  const cellFieldFloats = checkedCount(capacity
    * SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY
    * SPARSE_CM12_TILE_CLONE_POOL_CELL_FIELD_PLANES, "cellFieldFloats");
  const faceFieldFloats = checkedCount(capacity
    * SPARSE_CM12_TILE_CLONE_POOL_B8_FACE_CAPACITY
    * SPARSE_CM12_TILE_CLONE_POOL_FACE_FIELD_PLANES, "faceFieldFloats");
  const pageReceiptWords = checkedCount(capacity
    * SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT_WORDS, "pageReceiptWords");
  const metadataBytes = 4 * totalWords;
  return Object.freeze({
    capacity,
    hashCapacity,
    hashBaseWords,
    recordBaseWords,
    freeListBaseWords,
    totalWords,
    metadataBytes,
    cellFieldFloats,
    faceFieldFloats,
    pageReceiptWords,
    totalBytes: metadataBytes + 4 * (cellFieldFloats + faceFieldFloats
      + pageReceiptWords),
    cellPageCapacity: checkedCount(
      capacity * SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY,
      "cellPageCapacity",
    ),
    facePageCapacity: checkedCount(
      capacity * SPARSE_CM12_TILE_CLONE_POOL_B8_FACE_CAPACITY,
      "facePageCapacity",
    ),
  });
}

const coordinateHash = (coordinate: readonly number[]): number => {
  let hash = 0x811c_9dc5;
  for (const value of coordinate) {
    hash ^= value | 0;
    hash = Math.imul(hash, 0x0100_0193);
    hash ^= hash >>> 16;
  }
  return hash >>> 0;
};

const recordAt = (layout: SparseCM12TileClonePoolLayout, slot: number) =>
  layout.recordBaseWords + SPARSE_CM12_TILE_CLONE_POOL_RECORD_WORDS * slot;

const coordinateAt = (
  words: Uint32Array,
  layout: SparseCM12TileClonePoolLayout,
  slot: number,
): SparseBrickVec3 => {
  const at = recordAt(layout, slot), r = SPARSE_CM12_TILE_CLONE_POOL_RECORD;
  const signed = new Int32Array(words.buffer, words.byteOffset, words.length);
  return [signed[at + r.coordinateX]!, signed[at + r.coordinateY]!,
    signed[at + r.coordinateZ]!];
};

function directorySlotFor(
  words: Uint32Array,
  layout: SparseCM12TileClonePoolLayout,
  coordinate: SparseBrickVec3,
): { readonly directorySlot: number; readonly tileSlot?: number } {
  const mask = layout.hashCapacity - 1;
  let directorySlot = coordinateHash(coordinate) & mask;
  for (let probe = 0; probe < layout.hashCapacity; probe += 1) {
    const tileSlot = words[layout.hashBaseWords + directorySlot]!;
    if (tileSlot === SPARSE_CM12_TILE_CLONE_POOL_INVALID) return { directorySlot };
    const existing = coordinateAt(words, layout, tileSlot);
    if (existing.every((value, axis) => value === coordinate[axis])) {
      return { directorySlot, tileSlot };
    }
    directorySlot = (directorySlot + 1) & mask;
  }
  throw new Error("TCP1 sparse coordinate directory is full");
}

const directionForNeighbor = (index: number): SparseBrickVec3 => {
  const axis = Math.floor(index / 2), sign = (index & 1) === 0 ? -1 : 1;
  return [axis === 0 ? sign : 0, axis === 1 ? sign : 0, axis === 2 ? sign : 0];
};

function writeTileRecord(
  words: Uint32Array,
  layout: SparseCM12TileClonePoolLayout,
  slot: number,
  seed: SparseCM12TileCloneSeed,
  generation: number,
  authoredFluid: boolean,
): void {
  const at = recordAt(layout, slot), r = SPARSE_CM12_TILE_CLONE_POOL_RECORD;
  const signed = new Int32Array(words.buffer, words.byteOffset, words.length);
  const span = seed.spanBricks ?? 1;
  const spanLog2 = Math.log2(span);
  if (!Number.isInteger(spanLog2) || spanLog2 < 0) {
    throw new RangeError(`TCP1 span ${span} is not a positive power of two`);
  }
  words.fill(SPARSE_CM12_TILE_CLONE_POOL_INVALID, at,
    at + SPARSE_CM12_TILE_CLONE_POOL_RECORD_WORDS);
  words[at + r.generation] = generation;
  words[at + r.flags] = SPARSE_CM12_TILE_CLONE_POOL_TILE_FLAG.resident
    | SPARSE_CM12_TILE_CLONE_POOL_TILE_FLAG.accepted
    | (authoredFluid ? SPARSE_CM12_TILE_CLONE_POOL_TILE_FLAG.authoredFluid : 0);
  signed[at + r.coordinateX] = seed.coordinate[0];
  signed[at + r.coordinateY] = seed.coordinate[1];
  signed[at + r.coordinateZ] = seed.coordinate[2];
  words[at + r.spanLog2] = spanLog2;
  words[at + r.rung] = seed.resolution;
  words[at + r.fieldPage] = slot;
  words[at + r.presentationPage] = slot;
  words[at + r.pressureAggregate] = slot;
  words[at + r.lifecycleGeneration] = generation;
}

const cellCountForResolution = (resolution: SparseBrickResolution) => resolution ** 3;
const faceCountForResolution = (resolution: SparseBrickResolution) =>
  3 * (resolution + 1) * resolution * resolution;
const incidenceCountForResolution = (resolution: SparseBrickResolution) =>
  6 * resolution ** 3;
const internalPressureEdgeCountForResolution = (resolution: SparseBrickResolution) =>
  3 * Math.max(0, resolution - 1) * resolution * resolution;

const cellFieldBase = (slot: number, plane: number) => slot
  * SPARSE_CM12_TILE_CLONE_POOL_CELL_FIELD_PLANES
  * SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY
  + plane * SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY;

const faceFieldBase = (slot: number, plane: number) => slot
  * SPARSE_CM12_TILE_CLONE_POOL_FACE_FIELD_PLANES
  * SPARSE_CM12_TILE_CLONE_POOL_B8_FACE_CAPACITY
  + plane * SPARSE_CM12_TILE_CLONE_POOL_B8_FACE_CAPACITY;

const pageReceiptAt = (slot: number) =>
  slot * SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT_WORDS;

function initializeTilePage(
  cellFields: Float32Array,
  faceFields: Float32Array,
  pageReceipts: Uint32Array,
  slot: number,
  seed: SparseCM12TileCloneSeed,
  generation: number,
): void {
  const c = SPARSE_CM12_TILE_CLONE_POOL_CELL_FIELD;
  const f = SPARSE_CM12_TILE_CLONE_POOL_FACE_FIELD;
  const cellPageBegin = cellFieldBase(slot, 0);
  const cellPageEnd = cellPageBegin
    + SPARSE_CM12_TILE_CLONE_POOL_CELL_FIELD_PLANES
    * SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY;
  const facePageBegin = faceFieldBase(slot, 0);
  const facePageEnd = facePageBegin
    + SPARSE_CM12_TILE_CLONE_POOL_FACE_FIELD_PLANES
    * SPARSE_CM12_TILE_CLONE_POOL_B8_FACE_CAPACITY;
  cellFields.fill(0, cellPageBegin, cellPageEnd);
  faceFields.fill(0, facePageBegin, facePageEnd);
  cellFields.fill(1, cellFieldBase(slot, c.gammaA),
    cellFieldBase(slot, c.gammaA) + SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY);
  cellFields.fill(1, cellFieldBase(slot, c.gammaB),
    cellFieldBase(slot, c.gammaB) + SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY);
  faceFields.fill(1, faceFieldBase(slot, f.openFraction),
    faceFieldBase(slot, f.openFraction) + SPARSE_CM12_TILE_CLONE_POOL_B8_FACE_CAPACITY);

  const count = cellCountForResolution(seed.resolution);
  if (seed.density && seed.density.length !== count) {
    throw new Error(`TCP1 page ${slot} density has ${seed.density.length}/${count} values`);
  }
  if (seed.gamma && seed.gamma.length !== count) {
    throw new Error(`TCP1 page ${slot} gamma has ${seed.gamma.length}/${count} values`);
  }
  for (let local = 0; local < count; local += 1) {
    const density = seed.density?.[local] ?? 0;
    const gamma = seed.gamma?.[local] ?? 1;
    cellFields[cellFieldBase(slot, c.densityA) + local] = density;
    cellFields[cellFieldBase(slot, c.densityB) + local] = density;
    cellFields[cellFieldBase(slot, c.gammaA) + local] = gamma;
    cellFields[cellFieldBase(slot, c.gammaB) + local] = gamma;
  }

  const receipt = pageReceiptAt(slot), r = SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT;
  pageReceipts.fill(0, receipt, receipt + SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT_WORDS);
  pageReceipts[receipt + r.generation] = generation;
  pageReceipts[receipt + r.flags] = SPARSE_CM12_TILE_CLONE_POOL_PAGE_COMPLETE_FLAGS
    & ~SPARSE_CM12_TILE_CLONE_POOL_PAGE_FLAG.boundaryComplete;
  pageReceipts[receipt + r.resolution] = seed.resolution;
  pageReceipts[receipt + r.cellCount] = count;
  pageReceipts[receipt + r.faceCount] = faceCountForResolution(seed.resolution);
  pageReceipts[receipt + r.incidenceCount] = incidenceCountForResolution(seed.resolution);
  pageReceipts[receipt + r.internalPressureEdgeCount]
    = internalPressureEdgeCountForResolution(seed.resolution);
  pageReceipts[receipt + r.boundaryProgramCount]
    = SPARSE_CM12_TILE_CLONE_POOL_BOUNDARY_PROGRAM_COUNT;
  pageReceipts[receipt + r.cellFieldPage] = slot;
  pageReceipts[receipt + r.faceFieldPage] = slot;
  pageReceipts[receipt + r.pressureAggregate] = slot;
  pageReceipts[receipt + r.presentationPage] = slot;
}

function insertTile(
  words: Uint32Array,
  layout: SparseCM12TileClonePoolLayout,
  slot: number,
): void {
  const coordinate = coordinateAt(words, layout, slot);
  const found = directorySlotFor(words, layout, coordinate);
  if (found.tileSlot !== undefined) {
    throw new Error(`TCP1 duplicate coordinate ${coordinate.join("/")}`);
  }
  words[layout.hashBaseWords + found.directorySlot] = slot;
  words[recordAt(layout, slot)
    + SPARSE_CM12_TILE_CLONE_POOL_RECORD.directorySlot] = found.directorySlot;
}

function rebuildNeighborLinks(
  words: Uint32Array,
  layout: SparseCM12TileClonePoolLayout,
  pageReceipts: Uint32Array,
): void {
  const r = SPARSE_CM12_TILE_CLONE_POOL_RECORD;
  for (let slot = 0; slot < layout.capacity; slot += 1) {
    const at = recordAt(layout, slot);
    if ((words[at + r.flags]!
      & SPARSE_CM12_TILE_CLONE_POOL_TILE_FLAG.resident) === 0) continue;
    const coordinate = coordinateAt(words, layout, slot);
    for (let neighbor = 0; neighbor < SPARSE_CM12_TILE_CLONE_POOL_NEIGHBOR_COUNT;
      neighbor += 1) {
      const direction = directionForNeighbor(neighbor);
      const target: SparseBrickVec3 = [coordinate[0] + direction[0],
        coordinate[1] + direction[1], coordinate[2] + direction[2]];
      const found = directorySlotFor(words, layout, target);
      words[at + SPARSE_CM12_TILE_CLONE_POOL_RECORD.neighborBase + neighbor]
        = found.tileSlot ?? SPARSE_CM12_TILE_CLONE_POOL_INVALID;
      let boundaryProgram: number = SPARSE_CM12_TILE_CLONE_POOL_BOUNDARY_PROGRAM.sparseAir;
      if (found.tileSlot !== undefined) {
        const neighborRung = words[recordAt(layout, found.tileSlot) + r.rung]!;
        const ownRung = words[at + r.rung]!;
        boundaryProgram = neighborRung === ownRung
          ? SPARSE_CM12_TILE_CLONE_POOL_BOUNDARY_PROGRAM.sameRung
          : ownRung < neighborRung
            ? SPARSE_CM12_TILE_CLONE_POOL_BOUNDARY_PROGRAM.coarseToFine
            : SPARSE_CM12_TILE_CLONE_POOL_BOUNDARY_PROGRAM.fineToCoarse;
      }
      words[at + r.boundaryProgramBase + neighbor] = boundaryProgram;
    }
    pageReceipts[pageReceiptAt(slot) + SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT.flags]
      |= SPARSE_CM12_TILE_CLONE_POOL_PAGE_FLAG.boundaryComplete;
  }
}

export function createSparseCM12TileClonePool(
  seeds: readonly SparseCM12TileCloneSeed[],
  options: { readonly capacity?: number; readonly generation?: number } = {},
): SparseCM12TileClonePool {
  const capacity = options.capacity ?? sparseCM12TileClonePoolCapacity(seeds.length);
  if (seeds.length > capacity) {
    throw new RangeError(`TCP1 seeds ${seeds.length} exceed capacity ${capacity}`);
  }
  const generation = checkedCount(options.generation ?? 1, "generation", true);
  const layout = createSparseCM12TileClonePoolLayout(capacity);
  const words = new Uint32Array(layout.totalWords);
  const cellFields = new Float32Array(layout.cellFieldFloats);
  const faceFields = new Float32Array(layout.faceFieldFloats);
  const pageReceipts = new Uint32Array(layout.pageReceiptWords);
  words.fill(SPARSE_CM12_TILE_CLONE_POOL_INVALID, layout.hashBaseWords,
    layout.hashBaseWords + layout.hashCapacity);
  const h = SPARSE_CM12_TILE_CLONE_POOL_HEADER;
  words.set([
    SPARSE_CM12_TILE_CLONE_POOL_MAGIC,
    SPARSE_CM12_TILE_CLONE_POOL_VERSION,
    SPARSE_CM12_TILE_CLONE_POOL_HEADER_WORDS,
    SPARSE_CM12_TILE_CLONE_POOL_RECORD_WORDS,
    capacity,
    seeds.length,
    seeds.length,
    capacity - seeds.length,
    generation,
    generation + 1,
    0,
    layout.hashCapacity,
    layout.hashBaseWords,
    layout.recordBaseWords,
    layout.freeListBaseWords,
    layout.totalWords,
    SPARSE_CM12_TILE_CLONE_POOL_B8_CELL_CAPACITY,
    SPARSE_CM12_TILE_CLONE_POOL_B8_FACE_CAPACITY,
    0,
    0,
    0,
    SPARSE_CM12_TILE_CLONE_POOL_INVALID,
  ], 0);
  seeds.forEach((seed, slot) => {
    writeTileRecord(words, layout, slot, seed, generation, seed.authoredFluid ?? true);
    initializeTilePage(cellFields, faceFields, pageReceipts, slot, seed, generation);
    insertTile(words, layout, slot);
  });
  for (let slot = seeds.length; slot < capacity; slot += 1) {
    words[layout.freeListBaseWords + slot - seeds.length] = slot;
  }
  rebuildNeighborLinks(words, layout, pageReceipts);
  words[h.flags] = SPARSE_CM12_TILE_CLONE_POOL_FLAG.complete
    | SPARSE_CM12_TILE_CLONE_POOL_FLAG.validated;
  const result = Object.freeze({ layout, words, cellFields, faceFields, pageReceipts });
  validateSparseCM12TileClonePool(result);
  return result;
}

/** Authored-fluid seeds are the only generation-zero residents. */
export function sparseCM12TileCloneSeedsFromBricks(
  bricks: readonly SparseAdaptiveMassBrick[],
): readonly SparseCM12TileCloneSeed[] {
  return Object.freeze(bricks.filter((brick) => brick.density.some((value) => value > 0))
    .map((brick) => Object.freeze({ coordinate: [...brick.coordinate] as SparseBrickVec3,
      resolution: brick.resolution, spanBricks: brick.spanBricks,
      authoredFluid: true, density: brick.density, gamma: brick.gamma })));
}

export function sparseCM12TileClonePoolLookup(
  pool: SparseCM12TileClonePool,
  coordinate: SparseBrickVec3,
): number | undefined {
  return directorySlotFor(pool.words, pool.layout, coordinate).tileSlot;
}

/** CPU oracle for the exact allocation/link mutation the C1 GPU transaction performs. */
export function cloneSparseCM12TilePoolReceiver(
  source: SparseCM12TileClonePool,
  coordinate: SparseBrickVec3,
  resolution: SparseBrickResolution = 8,
): SparseCM12TileClonePool {
  if (sparseCM12TileClonePoolLookup(source, coordinate) !== undefined) return source;
  const words = source.words.slice(), cellFields = source.cellFields.slice();
  const faceFields = source.faceFields.slice(), pageReceipts = source.pageReceipts.slice();
  const { layout } = source;
  const h = SPARSE_CM12_TILE_CLONE_POOL_HEADER;
  const freeCount = words[h.freeCount]!;
  if (freeCount === 0) throw new Error("TCP1 physical page pool is exhausted");
  const freeIndex = freeCount - 1;
  const slot = words[layout.freeListBaseWords + freeIndex]!;
  if (slot === SPARSE_CM12_TILE_CLONE_POOL_INVALID || slot >= layout.capacity) {
    throw new Error("TCP1 free list contains an invalid physical slot");
  }
  const generation = words[h.candidateGeneration]!;
  const seed = { coordinate, resolution } as const;
  writeTileRecord(words, layout, slot, seed, generation, false);
  initializeTilePage(cellFields, faceFields, pageReceipts, slot, seed, generation);
  insertTile(words, layout, slot);
  words[layout.freeListBaseWords + freeIndex] = SPARSE_CM12_TILE_CLONE_POOL_INVALID;
  words[h.freeCount] = freeIndex;
  words[h.residentCount] += 1;
  words[h.highWaterMark] = Math.max(words[h.highWaterMark]!, words[h.residentCount]!);
  words[h.cloneCount] += 1;
  words[h.acceptedGeneration] = generation;
  words[h.candidateGeneration] = generation + 1;
  rebuildNeighborLinks(words, layout, pageReceipts);
  const result = Object.freeze({ layout, words, cellFields, faceFields, pageReceipts });
  validateSparseCM12TileClonePool(result);
  return result;
}

export function validateSparseCM12TileClonePool(pool: SparseCM12TileClonePool): void {
  const { words, layout } = pool, h = SPARSE_CM12_TILE_CLONE_POOL_HEADER;
  if (pool.cellFields.length !== layout.cellFieldFloats
    || pool.faceFields.length !== layout.faceFieldFloats
    || pool.pageReceipts.length !== layout.pageReceiptWords) {
    throw new Error("TCP1 physical page slab/layout mismatch");
  }
  if (words[h.magic] !== SPARSE_CM12_TILE_CLONE_POOL_MAGIC
    || words[h.version] !== SPARSE_CM12_TILE_CLONE_POOL_VERSION
    || words[h.headerWords] !== SPARSE_CM12_TILE_CLONE_POOL_HEADER_WORDS
    || words[h.recordWords] !== SPARSE_CM12_TILE_CLONE_POOL_RECORD_WORDS
    || words[h.capacity] !== layout.capacity
    || words[h.hashCapacity] !== layout.hashCapacity
    || words[h.hashBase] !== layout.hashBaseWords
    || words[h.recordBase] !== layout.recordBaseWords
    || words[h.freeListBase] !== layout.freeListBaseWords
    || words[h.totalWords] !== layout.totalWords) {
    throw new Error("TCP1 header/layout mismatch");
  }
  const residentCount = words[h.residentCount]!, freeCount = words[h.freeCount]!;
  if (residentCount + freeCount !== layout.capacity
    || words[h.highWaterMark]! < residentCount) {
    throw new Error("TCP1 resident/free/high-water census is inconsistent");
  }
  const seen = new Set<string>();
  let observedResidents = 0;
  for (let slot = 0; slot < layout.capacity; slot += 1) {
    const at = recordAt(layout, slot), flags = words[at
      + SPARSE_CM12_TILE_CLONE_POOL_RECORD.flags]!;
    if ((flags & SPARSE_CM12_TILE_CLONE_POOL_TILE_FLAG.resident) === 0) continue;
    observedResidents += 1;
    const coordinate = coordinateAt(words, layout, slot), key = coordinate.join("/");
    if (seen.has(key)) throw new Error(`TCP1 duplicate resident coordinate ${key}`);
    seen.add(key);
    if (sparseCM12TileClonePoolLookup(pool, coordinate) !== slot) {
      throw new Error(`TCP1 directory does not resolve resident slot ${slot}`);
    }
    const page = pageReceiptAt(slot), p = SPARSE_CM12_TILE_CLONE_POOL_PAGE_RECEIPT;
    const resolution = words[at + SPARSE_CM12_TILE_CLONE_POOL_RECORD.rung] as SparseBrickResolution;
    if ((pool.pageReceipts[page + p.flags]!
        & SPARSE_CM12_TILE_CLONE_POOL_PAGE_COMPLETE_FLAGS)
      !== SPARSE_CM12_TILE_CLONE_POOL_PAGE_COMPLETE_FLAGS
      || pool.pageReceipts[page + p.generation] !== words[at
        + SPARSE_CM12_TILE_CLONE_POOL_RECORD.generation]
      || pool.pageReceipts[page + p.resolution] !== resolution
      || pool.pageReceipts[page + p.cellCount] !== cellCountForResolution(resolution)
      || pool.pageReceipts[page + p.faceCount] !== faceCountForResolution(resolution)
      || pool.pageReceipts[page + p.incidenceCount]
        !== incidenceCountForResolution(resolution)
      || pool.pageReceipts[page + p.internalPressureEdgeCount]
        !== internalPressureEdgeCountForResolution(resolution)
      || pool.pageReceipts[page + p.boundaryProgramCount]
        !== SPARSE_CM12_TILE_CLONE_POOL_BOUNDARY_PROGRAM_COUNT
      || pool.pageReceipts[page + p.cellFieldPage] !== slot
      || pool.pageReceipts[page + p.faceFieldPage] !== slot
      || pool.pageReceipts[page + p.pressureAggregate] !== slot
      || pool.pageReceipts[page + p.presentationPage] !== slot) {
      throw new Error(`TCP1 resident page ${slot} is incomplete`);
    }
    for (let boundary = 0; boundary < SPARSE_CM12_TILE_CLONE_POOL_BOUNDARY_PROGRAM_COUNT;
      boundary += 1) {
      const program = words[at
        + SPARSE_CM12_TILE_CLONE_POOL_RECORD.boundaryProgramBase + boundary]!;
      if (program < SPARSE_CM12_TILE_CLONE_POOL_BOUNDARY_PROGRAM.sparseAir
        || program > SPARSE_CM12_TILE_CLONE_POOL_BOUNDARY_PROGRAM.wall) {
        throw new Error(`TCP1 page ${slot} boundary ${boundary} is incomplete`);
      }
    }
  }
  if (observedResidents !== residentCount) {
    throw new Error(`TCP1 resident census ${observedResidents}/${residentCount} disagrees`);
  }
  const freeSlots = new Set<number>();
  for (let index = 0; index < freeCount; index += 1) {
    const slot = words[layout.freeListBaseWords + index]!;
    const at = slot < layout.capacity ? recordAt(layout, slot) : 0;
    if (slot >= layout.capacity || freeSlots.has(slot)
      || (words[at + SPARSE_CM12_TILE_CLONE_POOL_RECORD.flags]!
        & SPARSE_CM12_TILE_CLONE_POOL_TILE_FLAG.resident) !== 0) {
      throw new Error(`TCP1 invalid free slot ${slot}`);
    }
    freeSlots.add(slot);
  }
  if (freeSlots.size !== freeCount) throw new Error("TCP1 free-list cardinality mismatch");
}

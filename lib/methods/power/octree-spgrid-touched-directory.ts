export interface SPGridTouchedCell {
  readonly coordinate: readonly [number, number, number];
  readonly slot: number;
}

export interface SPGridTouchedBrick {
  readonly dense: number;
  readonly masks: readonly [number, number];
  readonly rankedBase: number;
}

export interface SPGridTouchedDirectory {
  /** First-touch order, matching the ordered GPU claim owner. */
  readonly touchedBricks: readonly number[];
  /** Dense-ID radix order consumed by mask/rank publication. */
  readonly bricks: readonly SPGridTouchedBrick[];
  readonly rankedSlots: readonly number[];
  /** Unique dense logical page IDs in stable physical-ID order. */
  readonly pages: readonly number[];
}

function positiveDimensions(dimensions: readonly [number, number, number]) {
  if (dimensions.some((value) => !Number.isSafeInteger(value) || value <= 0)) {
    throw new RangeError("SPGrid touched-directory dimensions must be positive integers");
  }
}

/** Stable LSD radix sort used by the candidate GPU design: four byte passes,
 * with no comparison or domain-sized histogram. */
export function stableRadixSortU32(values: readonly number[]): number[] {
  let source = values.map((value) => {
    if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
      throw new RangeError("SPGrid touched identity must be a u32");
    }
    return value >>> 0;
  });
  let destination = new Array<number>(source.length);
  for (let radixPass = 0; radixPass < 4; radixPass += 1) {
    const histogram = new Uint32Array(256), shift = radixPass * 8;
    for (const value of source) histogram[(value >>> shift) & 255] += 1;
    let prefix = 0;
    for (let bucket = 0; bucket < 256; bucket += 1) {
      const count = histogram[bucket]; histogram[bucket] = prefix; prefix += count;
    }
    for (const value of source) {
      const bucket = (value >>> shift) & 255;
      destination[histogram[bucket]++] = value;
    }
    [source, destination] = [destination, source];
  }
  return source;
}

/** Footprint-only construction oracle. Its work is O(live cells + touched
 * bricks log_radix + touched pages); no loop is shaped by dense brick/page
 * cardinality. */
export function buildSPGridTouchedDirectory(
  dimensions: readonly [number, number, number],
  cells: readonly SPGridTouchedCell[],
  queriedEmptyCoordinates: readonly (readonly [number, number, number])[] = [],
): SPGridTouchedDirectory {
  positiveDimensions(dimensions);
  const brickDims = dimensions.map((value) => Math.ceil(value / 4)) as [number, number, number];
  const pageDims = [Math.ceil(dimensions[0] / 8), Math.ceil(dimensions[1] / 8),
    Math.ceil(dimensions[2] / 4)] as const;
  const byBrick = new Map<number, { masks: [number, number]; slots: Map<number, number> }>();
  const touchedBricks: number[] = [];
  const touchBrick = (coordinate: readonly [number, number, number]) => {
    const [x, y, z] = coordinate;
    if (![x, y, z].every(Number.isSafeInteger)
      || x < 0 || y < 0 || z < 0 || x >= dimensions[0] || y >= dimensions[1]
      || z >= dimensions[2]) throw new RangeError("Malformed SPGrid queried-empty coordinate");
    const dense = Math.floor(x / 4) + brickDims[0]
      * (Math.floor(y / 4) + brickDims[1] * Math.floor(z / 4));
    let brick = byBrick.get(dense);
    if (!brick) {
      brick = { masks: [0, 0], slots: new Map() };
      byBrick.set(dense, brick); touchedBricks.push(dense);
    }
    return { brick, dense };
  };
  const seenSlots = new Set<number>();
  for (const cell of cells) {
    const [x, y, z] = cell.coordinate;
    if (![x, y, z, cell.slot].every(Number.isSafeInteger)
      || x < 0 || y < 0 || z < 0 || x >= dimensions[0] || y >= dimensions[1]
      || z >= dimensions[2] || cell.slot < 0 || cell.slot > 0xffff_ffff) {
      throw new RangeError("Malformed SPGrid touched cell");
    }
    if (seenSlots.has(cell.slot)) throw new RangeError("Duplicate SPGrid touched slot");
    seenSlots.add(cell.slot);
    const { brick } = touchBrick(cell.coordinate);
    const bit = (x & 3) + 4 * (y & 3) + 16 * (z & 3);
    if (brick.slots.has(bit)) throw new RangeError("Duplicate SPGrid touched coordinate");
    brick.slots.set(bit, cell.slot >>> 0);
    const word = bit >>> 5;
    brick.masks[word] = (brick.masks[word] | (1 << (bit & 31))) >>> 0;
  }
  for (const coordinate of queriedEmptyCoordinates) touchBrick(coordinate);

  const sortedBricks = stableRadixSortU32(touchedBricks);
  const rankedSlots: number[] = [], bricks: SPGridTouchedBrick[] = [];
  const pageCandidates: number[] = [];
  for (const dense of sortedBricks) {
    const brick = byBrick.get(dense)!;
    const rankedBase = rankedSlots.length;
    for (let bit = 0; bit < 64; bit += 1) {
      const slot = brick.slots.get(bit); if (slot !== undefined) rankedSlots.push(slot);
    }
    bricks.push(Object.freeze({ dense, masks: Object.freeze([...brick.masks]) as readonly [number, number], rankedBase }));
    if (brick.masks[0] !== 0 || brick.masks[1] !== 0) {
      const bx = dense % brickDims[0];
      const by = Math.floor(dense / brickDims[0]) % brickDims[1];
      const bz = Math.floor(dense / (brickDims[0] * brickDims[1]));
      const px = Math.floor(bx / 2), py = Math.floor(by / 2), pz = bz;
      pageCandidates.push(px + pageDims[0] * (py + pageDims[1] * pz));
    }
  }
  const pages = stableRadixSortU32(pageCandidates)
    .filter((value, index, sorted) => index === 0 || value !== sorted[index - 1]);
  return Object.freeze({
    touchedBricks: Object.freeze(touchedBricks), bricks: Object.freeze(bricks),
    rankedSlots: Object.freeze(rankedSlots), pages: Object.freeze(pages),
  });
}

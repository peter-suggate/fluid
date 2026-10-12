import type { PickedSolidVoxel } from "../editor-solid-voxel";
import { VOXEL_BLOCK_MAXIMUM_CELLS, voxelBlockPatch } from "../solid-patch/block";
import { solidFractionOccupies, type VoxelBlock } from "../solid-patch/kind";
import type { SolidPatchOperation, SolidWorldCoordinate, SolidWorldVoxelPatch } from "../solid-world";
import type { VoxelReader, VoxelRegion } from "./voxel-access";

/**
 * What selection verbs are made of: which cells belong together, and what a
 * set of cells becomes when it is flipped, turned, thickened or written back.
 *
 * Everything here is a function of a reader, a region or a block and returns a
 * new one. Nothing touches a scene; the patches a verb appends come out of
 * `regionPatches` and `blockPatches` and are the caller's to commit.
 *
 * Every function is bounded. A search stops at a window and throws past its
 * cell limit, and anything that allocates a mask or a block refuses a box of
 * more than `VOXEL_BLOCK_MAXIMUM_CELLS` cells, so no gesture can ask for more
 * memory than one block patch may hold.
 */

/** Cells a search looks either side of where it starts, unless told otherwise. */
export const DEFAULT_REGION_REACH = 128;
/** The widest a search window may be asked to reach: 511 cells across. */
export const MAXIMUM_REGION_REACH = 255;
/** Cells a found region may hold before the search throws: one stroke's worth. */
export const DEFAULT_REGION_CELL_LIMIT = 128 ** 3;
/** Boxes a masked region may be written as before it becomes one block patch. */
export const REGION_BOX_PATCH_LIMIT = 64;

const EMPTY_DISTANCE_Q8 = 0x7fff;
/** Half a cell inside: the distance a whole box fill writes. */
const WHOLE_DISTANCE_Q8 = -128;

export interface RegionSearchOptions {
  /** Keep to cells of the material the search starts on. */
  readonly byMaterial?: boolean;
  /** Most cells the region may hold; one more is an error, not a truncation. */
  readonly limit?: number;
  /** Cells the search looks either side of its start. */
  readonly reach?: number;
}

/** A region a search found: always masked, with what the search knows about it. */
export interface FoundVoxelRegion extends VoxelRegion {
  readonly mask: Uint8Array;
  readonly cellCount: number;
  /** True when the piece carries on past the search window, which is where this region stops. */
  readonly clipped: boolean;
}

/** The exposed faces a search found, as the layer of cells that carry them. */
export interface FoundVoxelFaceRegion extends FoundVoxelRegion {
  readonly faceAxis: 0 | 1 | 2;
  readonly faceSign: -1 | 1;
}

const boxSize = (minimum: SolidWorldCoordinate, maximumExclusive: SolidWorldCoordinate): SolidWorldCoordinate =>
  [maximumExclusive[0] - minimum[0], maximumExclusive[1] - minimum[1], maximumExclusive[2] - minimum[2]];

/** Size and cell count of a region's box, having refused a box or a mask that cannot be one. */
function measured(region: VoxelRegion): { readonly size: SolidWorldCoordinate; readonly cells: number } {
  const size = boxSize(region.minimum, region.maximumExclusive);
  if (![...region.minimum, ...region.maximumExclusive].every((value) => (value | 0) === value)
    || size.some((value) => value < 1)) {
    throw new RangeError("A voxel region needs a non-empty box of signed i32 cells");
  }
  const cells = size[0] * size[1] * size[2];
  if (region.mask && region.mask.length !== cells) {
    throw new RangeError(`A voxel region mask of ${region.mask.length} cells does not cover its ${size.join(" x ")} box`);
  }
  return { size, cells };
}

function bounded(cells: number, size: SolidWorldCoordinate, what: string): void {
  if (cells > VOXEL_BLOCK_MAXIMUM_CELLS) {
    throw new RangeError(`${what} of ${size.join(" x ")} cells exceeds the `
      + `${VOXEL_BLOCK_MAXIMUM_CELLS.toLocaleString("en-US")}-cell limit`);
  }
}

function measuredBlock(block: VoxelBlock): number {
  const cells = block.size[0] * block.size[1] * block.size[2];
  if (block.size.some((value) => !Number.isSafeInteger(value) || value < 1)
    || [block.solidFraction, block.signedDistanceQ8, block.materialId].some((channel) => channel.length !== cells)) {
    throw new RangeError(`Voxel block channels must each hold the cells of a ${block.size.join(" x ")} box`);
  }
  return cells;
}

function searchBounds(options: RegionSearchOptions): { readonly reach: number; readonly limit: number } {
  const reach = options.reach ?? DEFAULT_REGION_REACH, limit = options.limit ?? DEFAULT_REGION_CELL_LIMIT;
  if (!Number.isSafeInteger(reach) || reach < 0 || reach > MAXIMUM_REGION_REACH) {
    throw new RangeError(`A region search reaches 0 to ${MAXIMUM_REGION_REACH} cells, not ${reach}`);
  }
  if (!Number.isSafeInteger(limit) || limit < 1) throw new RangeError(`A region cell limit must be a positive integer, not ${limit}`);
  return { reach, limit };
}

/**
 * Flood a window of `side` cells along each of `axes` from its centre.
 *
 * `belongs` is asked about each cell at most once, in window coordinates, and
 * returns whether the cell is part of the piece; cells outside the window are
 * asked only to learn whether the piece was cut off there.
 */
function flood(side: readonly [number, number, number], start: readonly [number, number, number],
  belongs: (x: number, y: number, z: number) => boolean, limit: number, what: string,
): { readonly state: Uint8Array; readonly lo: number[]; readonly hi: number[]; readonly count: number; readonly clipped: boolean } {
  // 0 not looked at, 1 part of the piece, 2 looked at and not part of it.
  const state = new Uint8Array(side[0] * side[1] * side[2]);
  const strideY = side[0], strideZ = side[0] * side[1];
  const stack = [start[0] + strideY * start[1] + strideZ * start[2]];
  state[stack[0]!] = 1;
  const lo = [...start], hi = [...start];
  let count = 1, clipped = false;
  const visit = (x: number, y: number, z: number): void => {
    if (x < 0 || y < 0 || z < 0 || x >= side[0] || y >= side[1] || z >= side[2]) {
      if (!clipped && belongs(x, y, z)) clipped = true;
      return;
    }
    const cell = x + strideY * y + strideZ * z;
    if (state[cell] !== 0) return;
    if (!belongs(x, y, z)) { state[cell] = 2; return; }
    state[cell] = 1;
    if (++count > limit) {
      throw new RangeError(`${what} holds more than ${limit.toLocaleString("en-US")} cells; `
        + "select a smaller piece or raise the limit");
    }
    if (x < lo[0]!) lo[0] = x; else if (x > hi[0]!) hi[0] = x;
    if (y < lo[1]!) lo[1] = y; else if (y > hi[1]!) hi[1] = y;
    if (z < lo[2]!) lo[2] = z; else if (z > hi[2]!) hi[2] = z;
    stack.push(cell);
  };
  while (stack.length > 0) {
    const cell = stack.pop()!;
    const x = cell % strideY, y = Math.floor(cell / strideY) % side[1], z = Math.floor(cell / strideZ);
    // A window one cell thick on an axis is a search that does not move along it.
    if (side[0] > 1) { visit(x - 1, y, z); visit(x + 1, y, z); }
    if (side[1] > 1) { visit(x, y - 1, z); visit(x, y + 1, z); }
    if (side[2] > 1) { visit(x, y, z - 1); visit(x, y, z + 1); }
  }
  return { state, lo, hi, count, clipped };
}

/** Cut the piece a flood found out of its window, as a region in lattice coordinates. */
function found(windowMinimum: readonly number[], side: readonly [number, number, number],
  result: ReturnType<typeof flood>): FoundVoxelRegion {
  const { state, lo, hi } = result;
  const size = [hi[0]! - lo[0]! + 1, hi[1]! - lo[1]! + 1, hi[2]! - lo[2]! + 1] as const;
  const mask = new Uint8Array(size[0] * size[1] * size[2]);
  for (let z = 0, cell = 0; z < size[2]; z += 1) for (let y = 0; y < size[1]; y += 1) {
    const row = lo[0]! + side[0] * (lo[1]! + y + side[1] * (lo[2]! + z));
    for (let x = 0; x < size[0]; x += 1, cell += 1) if (state[row + x] === 1) mask[cell] = 1;
  }
  const minimum: SolidWorldCoordinate = [windowMinimum[0]! + lo[0]!, windowMinimum[1]! + lo[1]!, windowMinimum[2]! + lo[2]!];
  return { minimum, maximumExclusive: [minimum[0] + size[0], minimum[1] + size[1], minimum[2] + size[2]],
    mask, cellCount: result.count, clipped: result.clipped };
}

/**
 * The solid cells joined to `seed` face to face — one piece, however it bends.
 *
 * Looked for inside a window `reach` cells either side of the seed; a piece
 * that carries on past it is returned cut at the window and marked `clipped`.
 * Throws when the seed is not solid, and when the piece holds more than
 * `limit` cells.
 */
export function connectedRegion(reader: VoxelReader, seed: SolidWorldCoordinate,
  options: RegionSearchOptions = {}): FoundVoxelRegion {
  const { reach, limit } = searchBounds(options);
  const probe: [number, number, number] = [seed[0], seed[1], seed[2]];
  if (!reader.solid(probe)) throw new RangeError(`No solid voxel at ${seed.join(", ")} to select a piece from`);
  const material = options.byMaterial ? reader.material(probe) : undefined;
  const windowMinimum = [seed[0] - reach, seed[1] - reach, seed[2] - reach];
  const side = 2 * reach + 1;
  const result = flood([side, side, side], [reach, reach, reach], (x, y, z) => {
    probe[0] = windowMinimum[0]! + x; probe[1] = windowMinimum[1]! + y; probe[2] = windowMinimum[2]! + z;
    return reader.solid(probe) && (material === undefined || reader.material(probe) === material);
  }, limit, "The connected piece");
  return found(windowMinimum, [side, side, side], result);
}

/**
 * The exposed faces that continue the picked one: cells in the picked cell's
 * layer, joined edge to edge, each solid with nothing in front of it.
 *
 * Returned as that layer of cells, one deep along `faceAxis` — the footprint a
 * face extrude pushes or pulls, and what a wand on a face selects. Throws when
 * the picked cell has no exposed face on that side.
 */
export function faceRegion(reader: VoxelReader,
  picked: Pick<PickedSolidVoxel, "coordinate" | "faceAxis" | "faceSign">,
  options: RegionSearchOptions = {}): FoundVoxelFaceRegion {
  const { reach, limit } = searchBounds(options);
  const { coordinate, faceAxis, faceSign } = picked;
  const probe: [number, number, number] = [coordinate[0], coordinate[1], coordinate[2]];
  const material = options.byMaterial ? reader.material(probe) : undefined;
  const side: [number, number, number] = [2 * reach + 1, 2 * reach + 1, 2 * reach + 1];
  const start: [number, number, number] = [reach, reach, reach];
  const windowMinimum = [coordinate[0] - reach, coordinate[1] - reach, coordinate[2] - reach];
  side[faceAxis] = 1; start[faceAxis] = 0; windowMinimum[faceAxis] = coordinate[faceAxis];
  const exposed = (x: number, y: number, z: number): boolean => {
    probe[0] = windowMinimum[0]! + x; probe[1] = windowMinimum[1]! + y; probe[2] = windowMinimum[2]! + z;
    if (!reader.solid(probe) || (material !== undefined && reader.material(probe) !== material)) return false;
    probe[faceAxis] += faceSign;
    return !reader.solid(probe);
  };
  if (!exposed(start[0], start[1], start[2])) {
    throw new RangeError(`Voxel ${coordinate.join(", ")} has no exposed face on its ${faceSign > 0 ? "+" : "-"}${"xyz"[faceAxis]} side`);
  }
  return { ...found(windowMinimum, side, flood(side, start, exposed, limit, "The connected face")), faceAxis, faceSign };
}

/** Whether a cell belongs to a region. */
export function regionContains(region: VoxelRegion, coordinate: SolidWorldCoordinate): boolean {
  const { minimum, maximumExclusive, mask } = region;
  if (!coordinate.every((value, axis) => value >= minimum[axis]! && value < maximumExclusive[axis]!)) return false;
  if (!mask) return true;
  const sx = maximumExclusive[0] - minimum[0], sy = maximumExclusive[1] - minimum[1];
  return mask[coordinate[0] - minimum[0] + sx * (coordinate[1] - minimum[1] + sy * (coordinate[2] - minimum[2]))] !== 0;
}

/** How many cells a region holds. */
export function regionCellCount(region: VoxelRegion): number {
  const { cells } = measured(region);
  if (!region.mask) return cells;
  let count = 0;
  for (let cell = 0; cell < cells; cell += 1) if (region.mask[cell] !== 0) count += 1;
  return count;
}

/** Every cell of a region, x fastest, then y, then z. */
export function* regionCells(region: VoxelRegion): Generator<SolidWorldCoordinate, void, undefined> {
  const { size } = measured(region), { minimum, mask } = region;
  for (let z = 0, cell = 0; z < size[2]; z += 1)
    for (let y = 0; y < size[1]; y += 1)
      for (let x = 0; x < size[0]; x += 1, cell += 1) {
        if (!mask || mask[cell] !== 0) yield [minimum[0] + x, minimum[1] + y, minimum[2] + z];
      }
}

/** The solid cells of a block standing at `minimum`, as a region. */
export function regionFromBlock(block: VoxelBlock, minimum: SolidWorldCoordinate): VoxelRegion & { readonly mask: Uint8Array } {
  const cells = measuredBlock(block), mask = new Uint8Array(cells);
  for (let cell = 0; cell < cells; cell += 1) if (solidFractionOccupies(block.solidFraction[cell]!)) mask[cell] = 1;
  return { minimum: [minimum[0], minimum[1], minimum[2]],
    maximumExclusive: [minimum[0] + block.size[0], minimum[1] + block.size[1], minimum[2] + block.size[2]], mask };
}

/** The cells of a region as the reader holds them; a cell outside the mask comes back empty. */
export function blockOfRegion(reader: VoxelReader, region: VoxelRegion): VoxelBlock {
  const { cells } = measured(region);
  const block = reader.read(region.minimum, region.maximumExclusive);
  if (region.mask) for (let cell = 0; cell < cells; cell += 1) {
    if (region.mask[cell] !== 0) continue;
    block.solidFraction[cell] = 0;
    block.signedDistanceQ8[cell] = EMPTY_DISTANCE_Q8;
    block.materialId[cell] = 0;
  }
  return block;
}

/** Move every cell of a block to where `target` sends it, in a block of `size`. */
function rearranged(block: VoxelBlock, size: SolidWorldCoordinate,
  target: (x: number, y: number, z: number) => number): VoxelBlock {
  const cells = measuredBlock(block), [sx, sy, sz] = block.size;
  const solidFraction = new Uint8Array(cells), signedDistanceQ8 = new Int16Array(cells);
  const materialId = new Uint16Array(cells);
  for (let z = 0, cell = 0; z < sz; z += 1) for (let y = 0; y < sy; y += 1) for (let x = 0; x < sx; x += 1, cell += 1) {
    const to = target(x, y, z);
    solidFraction[to] = block.solidFraction[cell]!;
    signedDistanceQ8[to] = block.signedDistanceQ8[cell]!;
    materialId[to] = block.materialId[cell]!;
  }
  return { size, solidFraction, signedDistanceQ8, materialId };
}

/** The block mirrored along one axis. Its size is unchanged. */
export function flipBlock(block: VoxelBlock, axis: 0 | 1 | 2): VoxelBlock {
  const [sx, sy, sz] = block.size;
  return rearranged(block, [sx, sy, sz], (x, y, z) =>
    (axis === 0 ? sx - 1 - x : x) + sx * ((axis === 1 ? sy - 1 - y : y) + sy * (axis === 2 ? sz - 1 - z : z)));
}

/**
 * The block turned about one axis by `turns` quarter turns.
 *
 * A turn is right-handed: about +z it carries +x to +y, about +x it carries +y
 * to +z, about +y it carries +z to +x. Negative turns go the other way. The
 * two sizes across the axis swap on an odd number of turns.
 */
export function rotateBlock90(block: VoxelBlock, axis: 0 | 1 | 2, turns: number): VoxelBlock {
  if (!Number.isSafeInteger(turns)) throw new RangeError(`A block turns by whole quarter turns, not ${turns}`);
  const quarter = ((turns % 4) + 4) % 4;
  // The plane the turn happens in: a quarter turn carries +a to +b.
  const a = (axis + 1) % 3, b = (axis + 2) % 3;
  const from = block.size, size = [from[0], from[1], from[2]];
  if (quarter % 2 === 1) { size[a] = from[b]!; size[b] = from[a]!; }
  const at = [0, 0, 0];
  return rearranged(block, size as unknown as SolidWorldCoordinate, (x, y, z) => {
    at[0] = x; at[1] = y; at[2] = z;
    const u = at[a]!, v = at[b]!;
    if (quarter === 1) { at[a] = from[b]! - 1 - v; at[b] = u; }
    else if (quarter === 2) { at[a] = from[a]! - 1 - u; at[b] = from[b]! - 1 - v; }
    else if (quarter === 3) { at[a] = v; at[b] = from[a]! - 1 - u; }
    return at[0]! + size[0]! * (at[1]! + size[1]! * at[2]!);
  });
}

/**
 * A standing region with another added to it or taken out of it.
 *
 * The result is cropped to the cells it holds; `undefined` when a subtraction
 * leaves none.
 */
export function combinedRegion(standing: VoxelRegion, other: VoxelRegion, mode: "add" | "subtract"): VoxelRegion | undefined {
  measured(standing); measured(other);
  const minimum: SolidWorldCoordinate = mode === "subtract" ? standing.minimum
    : [Math.min(standing.minimum[0], other.minimum[0]), Math.min(standing.minimum[1], other.minimum[1]), Math.min(standing.minimum[2], other.minimum[2])];
  const maximumExclusive: SolidWorldCoordinate = mode === "subtract" ? standing.maximumExclusive
    : [Math.max(standing.maximumExclusive[0], other.maximumExclusive[0]), Math.max(standing.maximumExclusive[1], other.maximumExclusive[1]),
      Math.max(standing.maximumExclusive[2], other.maximumExclusive[2])];
  const size = boxSize(minimum, maximumExclusive), cells = size[0] * size[1] * size[2];
  bounded(cells, size, "A combined selection");
  const held = new Uint8Array(cells);
  const write = (region: VoxelRegion, value: 0 | 1) => {
    const from = boxSize(region.minimum, region.maximumExclusive), mask = region.mask;
    const x0 = Math.max(region.minimum[0], minimum[0]), x1 = Math.min(region.maximumExclusive[0], maximumExclusive[0]);
    if (x0 >= x1) return;
    for (let z = Math.max(region.minimum[2], minimum[2]); z < Math.min(region.maximumExclusive[2], maximumExclusive[2]); z += 1)
      for (let y = Math.max(region.minimum[1], minimum[1]); y < Math.min(region.maximumExclusive[1], maximumExclusive[1]); y += 1) {
        const source = x0 - region.minimum[0] + from[0] * (y - region.minimum[1] + from[1] * (z - region.minimum[2]));
        const target = x0 - minimum[0] + size[0] * (y - minimum[1] + size[1] * (z - minimum[2]));
        for (let x = 0; x < x1 - x0; x += 1) if (!mask || mask[source + x] !== 0) held[target + x] = value;
      }
  };
  write(standing, 1);
  write(other, mode === "add" ? 1 : 0);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  let count = 0;
  for (let z = 0, cell = 0; z < size[2]; z += 1) for (let y = 0; y < size[1]; y += 1) for (let x = 0; x < size[0]; x += 1, cell += 1) {
    if (held[cell] === 0) continue;
    count += 1;
    lo[0] = Math.min(lo[0]!, x); hi[0] = Math.max(hi[0]!, x);
    lo[1] = Math.min(lo[1]!, y); hi[1] = Math.max(hi[1]!, y);
    lo[2] = Math.min(lo[2]!, z); hi[2] = Math.max(hi[2]!, z);
  }
  if (count === 0) return undefined;
  const kept = [hi[0]! - lo[0]! + 1, hi[1]! - lo[1]! + 1, hi[2]! - lo[2]! + 1] as const;
  const from: SolidWorldCoordinate = [minimum[0] + lo[0]!, minimum[1] + lo[1]!, minimum[2] + lo[2]!];
  const box = { minimum: from, maximumExclusive: [from[0] + kept[0], from[1] + kept[1], from[2] + kept[2]] as SolidWorldCoordinate };
  // A full box is a box: it draws as one, and the verbs write it as one patch.
  if (count === kept[0] * kept[1] * kept[2]) return box;
  const mask = new Uint8Array(kept[0] * kept[1] * kept[2]);
  for (let z = 0, cell = 0; z < kept[2]; z += 1) for (let y = 0; y < kept[1]; y += 1) {
    const row = lo[0]! + size[0] * (lo[1]! + y + size[1] * (lo[2]! + z));
    for (let x = 0; x < kept[0]; x += 1, cell += 1) mask[cell] = held[row + x]!;
  }
  return { ...box, mask };
}

/**
 * The region and every cell that shares a face with it.
 *
 * Face neighbours only, so a box grows into a box with its edges and corners
 * left off — the same shape a flood would reach in one step.
 */
export function growRegion(region: VoxelRegion): VoxelRegion & { readonly mask: Uint8Array } {
  const { size } = measured(region), source = region.mask;
  const grown: SolidWorldCoordinate = [size[0] + 2, size[1] + 2, size[2] + 2];
  const cells = grown[0] * grown[1] * grown[2];
  bounded(cells, grown, "A grown region");
  const mask = new Uint8Array(cells), strideY = grown[0], strideZ = grown[0] * grown[1];
  for (let z = 0, cell = 0; z < size[2]; z += 1) for (let y = 0; y < size[1]; y += 1)
    for (let x = 0; x < size[0]; x += 1, cell += 1) {
      if (source && source[cell] === 0) continue;
      const to = x + 1 + strideY * (y + 1) + strideZ * (z + 1);
      mask[to] = 1;
      mask[to - 1] = 1; mask[to + 1] = 1;
      mask[to - strideY] = 1; mask[to + strideY] = 1;
      mask[to - strideZ] = 1; mask[to + strideZ] = 1;
    }
  return { minimum: [region.minimum[0] - 1, region.minimum[1] - 1, region.minimum[2] - 1],
    maximumExclusive: [region.maximumExclusive[0] + 1, region.maximumExclusive[1] + 1, region.maximumExclusive[2] + 1],
    mask };
}

/**
 * The region without its skin: the cells whose six face neighbours all belong.
 *
 * `undefined` when nothing is left, which is every region thinner than three
 * cells somewhere along each of its cells.
 */
export function shrinkRegion(region: VoxelRegion): VoxelRegion | undefined {
  const { size, cells } = measured(region), source = region.mask;
  if (!source) {
    if (size.some((value) => value < 3)) return undefined;
    return { minimum: [region.minimum[0] + 1, region.minimum[1] + 1, region.minimum[2] + 1],
      maximumExclusive: [region.maximumExclusive[0] - 1, region.maximumExclusive[1] - 1, region.maximumExclusive[2] - 1] };
  }
  const strideY = size[0], strideZ = size[0] * size[1];
  const kept = new Uint8Array(cells);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let z = 1; z < size[2] - 1; z += 1) for (let y = 1; y < size[1] - 1; y += 1) for (let x = 1; x < size[0] - 1; x += 1) {
    const cell = x + strideY * y + strideZ * z;
    if (source[cell] === 0 || source[cell - 1] === 0 || source[cell + 1] === 0 || source[cell - strideY] === 0
      || source[cell + strideY] === 0 || source[cell - strideZ] === 0 || source[cell + strideZ] === 0) continue;
    kept[cell] = 1;
    lo[0] = Math.min(lo[0]!, x); hi[0] = Math.max(hi[0]!, x);
    lo[1] = Math.min(lo[1]!, y); hi[1] = Math.max(hi[1]!, y);
    lo[2] = Math.min(lo[2]!, z); hi[2] = Math.max(hi[2]!, z);
  }
  if (lo[0] === Infinity) return undefined;
  const shrunk = [hi[0]! - lo[0]! + 1, hi[1]! - lo[1]! + 1, hi[2]! - lo[2]! + 1] as const;
  const mask = new Uint8Array(shrunk[0] * shrunk[1] * shrunk[2]);
  for (let z = 0, cell = 0; z < shrunk[2]; z += 1) for (let y = 0; y < shrunk[1]; y += 1) {
    const row = lo[0]! + strideY * (lo[1]! + y) + strideZ * (lo[2]! + z);
    for (let x = 0; x < shrunk[0]; x += 1, cell += 1) mask[cell] = kept[row + x]!;
  }
  const minimum: SolidWorldCoordinate = [region.minimum[0] + lo[0]!, region.minimum[1] + lo[1]!, region.minimum[2] + lo[2]!];
  return { minimum, maximumExclusive: [minimum[0] + shrunk[0], minimum[1] + shrunk[1], minimum[2] + shrunk[2]], mask };
}

/**
 * The cells a hollow takes out of a block: every solid cell more than `wall`
 * cells from anything empty, with the block's own edge counting as empty.
 *
 * Distance is counted across edges and corners as well as faces. A skin kept
 * by faces alone is connected only through its corners where the surface
 * steps, and the smooth renderer draws that as a shell full of holes.
 */
export function blockInterior(block: VoxelBlock, wall = 1): Uint8Array {
  const cells = measuredBlock(block), [sx, sy, sz] = block.size;
  if (!Number.isSafeInteger(wall) || wall < 1) throw new RangeError(`A hollow keeps a wall of one or more cells, not ${wall}`);
  let interior = new Uint8Array(cells), scratch = new Uint8Array(cells);
  for (let cell = 0; cell < cells; cell += 1) if (solidFractionOccupies(block.solidFraction[cell]!)) interior[cell] = 1;
  // One cell of skin per round, and the 3 x 3 x 3 neighbourhood one axis at a time.
  const strides = [1, sx, sx * sy], extents = [sx, sy, sz];
  for (let round = 0; round < wall; round += 1) for (let axis = 0; axis < 3; axis += 1) {
    const stride = strides[axis]!, extent = extents[axis]!;
    for (let cell = 0; cell < cells; cell += 1) {
      const along = Math.floor(cell / stride) % extent;
      scratch[cell] = interior[cell] !== 0 && along > 0 && along < extent - 1
        && interior[cell - stride] !== 0 && interior[cell + stride] !== 0 ? 1 : 0;
    }
    [interior, scratch] = [scratch, interior];
  }
  return interior;
}

/** The block with its inside removed, leaving a shell `wall` cells thick. */
export function hollowBlock(block: VoxelBlock, wall = 1): VoxelBlock {
  const interior = blockInterior(block, wall), cells = interior.length, [sx, sy] = block.size;
  const solidFraction = block.solidFraction.slice(), signedDistanceQ8 = block.signedDistanceQ8.slice();
  const materialId = block.materialId.slice();
  for (let cell = 0; cell < cells; cell += 1) {
    if (interior[cell] === 0) continue;
    solidFraction[cell] = 0; signedDistanceQ8[cell] = EMPTY_DISTANCE_Q8; materialId[cell] = 0;
    // The cells left facing the new cavity are at a surface now, not deep inside.
    for (const stride of [1, sx, sx * sy]) for (const neighbour of [cell - stride, cell + stride]) {
      if (interior[neighbour] === 0 && signedDistanceQ8[neighbour]! < WHOLE_DISTANCE_Q8) {
        signedDistanceQ8[neighbour] = WHOLE_DISTANCE_Q8;
      }
    }
  }
  return { size: block.size, solidFraction, signedDistanceQ8, materialId };
}

interface MergedBox {
  readonly minimum: SolidWorldCoordinate;
  readonly maximumExclusive: SolidWorldCoordinate;
  readonly group: number;
}

/**
 * Cover the cells of each non-zero group with boxes, greedily: a run along x,
 * widened along y, then deepened along z. `undefined` once that takes more
 * than `limit` boxes, which is how a round shape is found out early.
 */
function mergedBoxes(size: SolidWorldCoordinate, groups: Int32Array, limit: number): MergedBox[] | undefined {
  const [sx, sy, sz] = size, taken = new Uint8Array(groups.length), boxes: MergedBox[] = [];
  const free = (cell: number, group: number): boolean => groups[cell] === group && taken[cell] === 0;
  for (let z = 0, cell = 0; z < sz; z += 1) for (let y = 0; y < sy; y += 1) for (let x = 0; x < sx; x += 1, cell += 1) {
    const group = groups[cell]!;
    if (group === 0 || taken[cell] !== 0) continue;
    if (boxes.length === limit) return undefined;
    let x1 = x + 1, y1 = y + 1, z1 = z + 1;
    while (x1 < sx && free(cell + x1 - x, group)) x1 += 1;
    const rowFree = (rowY: number, rowZ: number): boolean => {
      for (let at = x; at < x1; at += 1) if (!free(at + sx * (rowY + sy * rowZ), group)) return false;
      return true;
    };
    while (y1 < sy && rowFree(y1, z)) y1 += 1;
    const layerFree = (layerZ: number): boolean => {
      for (let at = y; at < y1; at += 1) if (!rowFree(at, layerZ)) return false;
      return true;
    };
    while (z1 < sz && layerFree(z1)) z1 += 1;
    for (let tz = z; tz < z1; tz += 1) for (let ty = y; ty < y1; ty += 1) taken.fill(1, x + sx * (ty + sy * tz), x1 + sx * (ty + sy * tz));
    boxes.push({ minimum: [x, y, z], maximumExclusive: [x1, y1, z1], group });
  }
  return boxes;
}

const offset = (minimum: SolidWorldCoordinate, cell: SolidWorldCoordinate): SolidWorldCoordinate =>
  [minimum[0] + cell[0], minimum[1] + cell[1], minimum[2] + cell[2]];

/**
 * The patches that apply one operation to exactly the cells of a region.
 *
 * A box is one box patch. A masked region is boxes while a few of them cover
 * it, and otherwise a single block patch stamping the mask — a round selection
 * is thousands of rows as boxes and a few kilobytes as a block. `materialId` is
 * what a fill or a paint writes; a paint must name one.
 */
export function regionPatches(operation: SolidPatchOperation, region: VoxelRegion,
  materialId?: number): SolidWorldVoxelPatch[] {
  const { size, cells } = measured(region), mask = region.mask;
  if (operation === "paint" && materialId === undefined) throw new RangeError("A paint needs the material to paint with");
  const material = operation === "clear" || materialId === undefined ? {} : { materialId };
  if (!mask) {
    return [{ operation, minimum: [...region.minimum], maximumExclusive: [...region.maximumExclusive], ...material }];
  }
  bounded(cells, size, "A masked region");
  const groups = new Int32Array(cells);
  let members = 0;
  for (let cell = 0; cell < cells; cell += 1) if (mask[cell] !== 0) { groups[cell] = 1; members += 1; }
  if (members === 0) return [];
  const boxes = mergedBoxes(size, groups, REGION_BOX_PATCH_LIMIT);
  if (boxes) {
    return boxes.map((box) => ({ operation, minimum: offset(region.minimum, box.minimum),
      maximumExclusive: offset(region.minimum, box.maximumExclusive), ...material }));
  }
  const solidFraction = new Uint8Array(cells), signedDistanceQ8 = new Int16Array(cells).fill(EMPTY_DISTANCE_Q8);
  for (let cell = 0; cell < cells; cell += 1) if (mask[cell] !== 0) {
    solidFraction[cell] = 255; signedDistanceQ8[cell] = WHOLE_DISTANCE_Q8;
  }
  // Material 0 in a block cell means the patch's own, so one id serves every cell.
  return [voxelBlockPatch(operation, region.minimum,
    { size, solidFraction, signedDistanceQ8, materialId: new Uint16Array(cells) },
    operation === "clear" ? undefined : materialId)];
}

/**
 * The patches that apply one operation with the cells of a block standing at
 * `minimum`, each cell keeping its own material.
 *
 * Boxes when the block is whole cells that a few boxes cover — a pasted wall
 * stays a box in the document — and otherwise one block patch, which is the
 * only form that can carry partial coverage and the distances that go with it.
 */
export function blockPatches(operation: SolidPatchOperation, minimum: SolidWorldCoordinate,
  block: VoxelBlock): SolidWorldVoxelPatch[] {
  const cells = measuredBlock(block);
  bounded(cells, block.size, "A voxel block");
  // What a box patch would write differs by operation: a fill writes whole
  // cells half a cell deep, a clear takes whole cells away, a paint only names
  // a material. A block that is anything else has to travel as itself.
  const groups = new Int32Array(cells);
  let members = 0, boxable = true;
  for (let cell = 0; cell < cells && boxable; cell += 1) {
    const fraction = block.solidFraction[cell]!;
    if (!solidFractionOccupies(fraction)) continue;
    members += 1;
    groups[cell] = operation === "clear" ? 1 : block.materialId[cell]! + 1;
    if (operation === "paint") continue;
    boxable = fraction === 255 && (operation === "clear" || block.signedDistanceQ8[cell] === WHOLE_DISTANCE_Q8);
  }
  if (boxable && members === 0) return [];
  const boxes = boxable ? mergedBoxes(block.size, groups, REGION_BOX_PATCH_LIMIT) : undefined;
  if (!boxes) return [voxelBlockPatch(operation, minimum, block)];
  return boxes.map((box) => ({ operation, minimum: offset(minimum, box.minimum),
    maximumExclusive: offset(minimum, box.maximumExclusive),
    // Material 0 in a block cell is the patch's own, which a box says by naming none.
    ...(operation === "clear" || box.group === 1 ? {} : { materialId: box.group - 1 }) }));
}

/** A block with nothing in it: every cell as a whole clear leaves one. */
function emptyBlock(size: SolidWorldCoordinate): VoxelBlock {
  const cells = size[0] * size[1] * size[2];
  return { size: [size[0], size[1], size[2]], solidFraction: new Uint8Array(cells),
    signedDistanceQ8: new Int16Array(cells).fill(EMPTY_DISTANCE_Q8), materialId: new Uint16Array(cells) };
}

/** The box around the cells `member` accepts, in a box of `size`; `undefined` when it accepts none. */
function memberBounds(size: SolidWorldCoordinate, member: (cell: number) => boolean):
  { readonly lo: SolidWorldCoordinate; readonly size: SolidWorldCoordinate } | undefined {
  const lo = [size[0], size[1], size[2]], hi = [-1, -1, -1];
  for (let z = 0, cell = 0; z < size[2]; z += 1) for (let y = 0; y < size[1]; y += 1)
    for (let x = 0; x < size[0]; x += 1, cell += 1) {
      if (!member(cell)) continue;
      if (x < lo[0]!) lo[0] = x;
      if (x > hi[0]!) hi[0] = x;
      if (y < lo[1]!) lo[1] = y;
      if (y > hi[1]!) hi[1] = y;
      if (z < lo[2]!) lo[2] = z;
      if (z > hi[2]!) hi[2] = z;
    }
  if (hi[0]! < 0) return undefined;
  return { lo: [lo[0]!, lo[1]!, lo[2]!], size: [hi[0]! - lo[0]! + 1, hi[1]! - lo[1]! + 1, hi[2]! - lo[2]! + 1] };
}

/** Copy a sub-box of one channel out of a box of `from` cells. */
function cutChannel<Channel extends Uint8Array | Int16Array | Uint16Array>(channel: Channel, from: SolidWorldCoordinate,
  lo: SolidWorldCoordinate, size: SolidWorldCoordinate): Channel {
  const cut = new (channel.constructor as new (length: number) => Channel)(size[0] * size[1] * size[2]);
  for (let z = 0, to = 0; z < size[2]; z += 1) for (let y = 0; y < size[1]; y += 1, to += size[0]) {
    const row = lo[0] + from[0] * (lo[1] + y + from[1] * (lo[2] + z));
    cut.set(channel.subarray(row, row + size[0]), to);
  }
  return cut;
}

/**
 * The block cut down to the box around its solid cells, and where in the block
 * that box starts. `undefined` when no cell is solid.
 *
 * What a copy keeps and what a transform writes: the empty margin of a
 * selection is neither worth carrying on a clipboard nor the patch's to claim.
 */
export function croppedBlock(block: VoxelBlock):
  { readonly offset: SolidWorldCoordinate; readonly block: VoxelBlock } | undefined {
  measuredBlock(block);
  const bounds = memberBounds(block.size, (cell) => solidFractionOccupies(block.solidFraction[cell]!));
  if (!bounds) return undefined;
  return { offset: bounds.lo, block: { size: bounds.size,
    solidFraction: cutChannel(block.solidFraction, block.size, bounds.lo, bounds.size),
    signedDistanceQ8: cutChannel(block.signedDistanceQ8, block.size, bounds.lo, bounds.size),
    materialId: cutChannel(block.materialId, block.size, bounds.lo, bounds.size) } };
}

/** The solid cells of a block standing at `minimum`, as the tightest region that holds them. */
export function solidRegionOfBlock(block: VoxelBlock, minimum: SolidWorldCoordinate):
  (VoxelRegion & { readonly mask: Uint8Array }) | undefined {
  const cropped = croppedBlock(block);
  return cropped && regionFromBlock(cropped.block, offset(minimum, cropped.offset));
}

/** The same cells, `delta` cells away. The mask is shared, not copied: a region is never written to. */
export function translatedRegion(region: VoxelRegion, delta: SolidWorldCoordinate): VoxelRegion {
  return { minimum: offset(region.minimum, delta), maximumExclusive: offset(region.maximumExclusive, delta),
    ...(region.mask ? { mask: region.mask } : {}) };
}

/** A region's mask carried through the rearrangement a block of its contents went through. */
function rearrangedMask(region: VoxelRegion, rearrange: (block: VoxelBlock) => VoxelBlock): Uint8Array | undefined {
  if (!region.mask) return undefined;
  const { size, cells } = measured(region);
  // Carried as a block's coverage so one definition of a flip and a turn serves cells and masks alike.
  return rearrange({ size, solidFraction: region.mask, signedDistanceQ8: new Int16Array(cells),
    materialId: new Uint16Array(cells) }).solidFraction;
}

/** The region mirrored along one axis inside its own box. */
export function flippedRegion(region: VoxelRegion, axis: 0 | 1 | 2): VoxelRegion {
  const mask = rearrangedMask(region, (block) => flipBlock(block, axis));
  return { minimum: region.minimum, maximumExclusive: region.maximumExclusive, ...(mask ? { mask } : {}) };
}

const parity = (value: number): number => ((value % 2) + 2) % 2;

/**
 * Where a box stands after one right-handed quarter turn about `axis` through
 * its centre: its new first cell. Its two sizes across the axis swap.
 *
 * A turn about the exact centre lands on the lattice only when those two sizes
 * are both odd or both even. Otherwise the centre is half a cell off every
 * point a turn can be made about, and the turn is made about the nearest such
 * point instead: the lattice vertex half a cell from the centre whose two
 * coordinates across the axis sum to an even number. Exactly one vertex
 * qualifies, and every quarter turn of the same box picks the same one — so
 * four turns bring the box back to where it started rather than walking it.
 */
export function quarterTurnedMinimum(minimum: SolidWorldCoordinate, size: SolidWorldCoordinate,
  axis: 0 | 1 | 2): SolidWorldCoordinate {
  const a = (axis + 1) % 3, b = (axis + 2) % 3;
  // Twice the centre, so half cells stay whole numbers.
  let pivotA = 2 * minimum[a]! + size[a]!, pivotB = 2 * minimum[b]! + size[b]!;
  if (parity(pivotA) !== parity(pivotB)) {
    const even = (candidateA: number, candidateB: number) => parity((candidateA + candidateB) / 2) === 0;
    if (parity(pivotA) === 1) pivotA += even(pivotA - 1, pivotB) ? -1 : 1;
    else pivotB += even(pivotA, pivotB - 1) ? -1 : 1;
  }
  const turned = [minimum[0], minimum[1], minimum[2]];
  turned[a] = (pivotA + pivotB) / 2 - minimum[b]! - size[b]!;
  turned[b] = (pivotB - pivotA) / 2 + minimum[a]!;
  return [turned[0]!, turned[1]!, turned[2]!];
}

/** The region after one right-handed quarter turn about `axis`: see `quarterTurnedMinimum` for where it lands. */
export function quarterTurnedRegion(region: VoxelRegion, axis: 0 | 1 | 2): VoxelRegion {
  const { size } = measured(region);
  const minimum = quarterTurnedMinimum(region.minimum, size, axis);
  const a = (axis + 1) % 3, b = (axis + 2) % 3;
  const turned = [size[0], size[1], size[2]];
  turned[a] = size[b]!; turned[b] = size[a]!;
  const mask = rearrangedMask(region, (block) => rotateBlock90(block, axis, 1));
  return { minimum, maximumExclusive: [minimum[0] + turned[0]!, minimum[1] + turned[1]!, minimum[2] + turned[2]!],
    ...(mask ? { mask } : {}) };
}

/** Whether a cell of a block's `halo` is solid, addressed in the block's own coordinates (-1 to size). */
function haloSolid(block: VoxelBlock, halo: VoxelBlock): (x: number, y: number, z: number) => boolean {
  measuredBlock(halo);
  if (halo.size.some((value, axis) => value !== block.size[axis]! + 2)) {
    throw new RangeError(`A ${block.size.join(" x ")} block needs a halo of ${block.size.map((value) => value + 2).join(" x ")} `
      + `cells, not ${halo.size.join(" x ")}`);
  }
  const strideY = halo.size[0], strideZ = halo.size[0] * halo.size[1];
  return (x, y, z) => solidFractionOccupies(halo.solidFraction[x + 1 + strideY * (y + 1) + strideZ * (z + 1)]!);
}

/**
 * The block with one more layer of whole cells on every exposed side, in a
 * box one cell larger each way.
 *
 * `halo` is the world over that larger box. A side is exposed where the cell
 * beyond it is empty *there* — a piece standing on the ground does not grow
 * into the ground, and a neighbour that is already solid keeps its material.
 * Faces only: a layer added across edges and corners would round nothing and
 * thicken diagonals by more than the one cell asked for.
 */
export function grownBlock(block: VoxelBlock, halo: VoxelBlock, materialId: number): VoxelBlock {
  measuredBlock(block);
  const solid = haloSolid(block, halo), [sx, sy, sz] = block.size;
  const size: SolidWorldCoordinate = [sx + 2, sy + 2, sz + 2];
  bounded(size[0] * size[1] * size[2], size, "A grown block");
  const grown = emptyBlock(size), strideY = size[0], strideZ = size[0] * size[1];
  const add = (x: number, y: number, z: number): void => {
    if (solid(x, y, z)) return;
    const to = x + 1 + strideY * (y + 1) + strideZ * (z + 1);
    grown.solidFraction[to] = 255; grown.signedDistanceQ8[to] = WHOLE_DISTANCE_Q8; grown.materialId[to] = materialId;
  };
  for (let z = 0, cell = 0; z < sz; z += 1) for (let y = 0; y < sy; y += 1) for (let x = 0; x < sx; x += 1, cell += 1) {
    if (!solidFractionOccupies(block.solidFraction[cell]!)) continue;
    const to = x + 1 + strideY * (y + 1) + strideZ * (z + 1);
    grown.solidFraction[to] = block.solidFraction[cell]!;
    grown.signedDistanceQ8[to] = block.signedDistanceQ8[cell]!;
    grown.materialId[to] = block.materialId[cell]!;
    add(x - 1, y, z); add(x + 1, y, z); add(x, y - 1, z); add(x, y + 1, z); add(x, y, z - 1); add(x, y, z + 1);
  }
  return grown;
}

/**
 * The block without its exposed layer: every solid cell with an empty face
 * neighbour is taken away. `halo` is the world over the block's box and one
 * cell beyond, for the reason `grownBlock` reads one — a face against solid
 * ground is not exposed, so a piece is not peeled where it stands on something.
 */
export function peeledBlock(block: VoxelBlock, halo: VoxelBlock): VoxelBlock {
  const cells = measuredBlock(block), solid = haloSolid(block, halo), [sx, sy, sz] = block.size;
  const solidFraction = block.solidFraction.slice(), signedDistanceQ8 = block.signedDistanceQ8.slice();
  const materialId = block.materialId.slice();
  const peeled = new Uint8Array(cells);
  for (let z = 0, cell = 0; z < sz; z += 1) for (let y = 0; y < sy; y += 1) for (let x = 0; x < sx; x += 1, cell += 1) {
    if (!solidFractionOccupies(block.solidFraction[cell]!)) continue;
    if (solid(x - 1, y, z) && solid(x + 1, y, z) && solid(x, y - 1, z) && solid(x, y + 1, z)
      && solid(x, y, z - 1) && solid(x, y, z + 1)) continue;
    peeled[cell] = 1;
    solidFraction[cell] = 0; signedDistanceQ8[cell] = EMPTY_DISTANCE_Q8; materialId[cell] = 0;
  }
  // The cells left facing where the layer was are at a surface now, not deep inside.
  const strides = [1, sx, sx * sy], extents = [sx, sy, sz];
  for (let cell = 0; cell < cells; cell += 1) {
    if (peeled[cell] !== 0 || signedDistanceQ8[cell]! >= WHOLE_DISTANCE_Q8) continue;
    for (let axis = 0; axis < 3; axis += 1) {
      const stride = strides[axis]!, along = Math.floor(cell / stride) % extents[axis]!;
      if ((along > 0 && peeled[cell - stride] !== 0) || (along < extents[axis]! - 1 && peeled[cell + stride] !== 0)) {
        signedDistanceQ8[cell] = WHOLE_DISTANCE_Q8;
        break;
      }
    }
  }
  return { size: block.size, solidFraction, signedDistanceQ8, materialId };
}

/** A block set down on the lattice: where its first cell stands. */
export interface PlacedVoxelBlock {
  readonly minimum: SolidWorldCoordinate;
  readonly block: VoxelBlock;
}

/**
 * The patches that turn the cells of `before` into the cells of `after`, both
 * standing at `minimum` — only the cells that differ are touched.
 *
 * Clear the old cells, then write the new. A cell that changes and was not
 * already an empty cell is cleared whole first, because a fill only ever adds:
 * it keeps the larger coverage and the deeper distance, so a cell written
 * straight over another would be a blend of the two. Cleared, the fill that
 * follows writes each new cell exactly. Each half is cut to the box around its
 * own cells and travels as `regionPatches` and `blockPatches` send it — boxes
 * where a few cover it, one block otherwise — so sliding a slab one cell is
 * two thin boxes, not two copies of the slab.
 *
 * `after` must hold, in every cell that differs, either a solid cell or the
 * empty cell a whole clear leaves; anything else has no patch that writes it.
 */
export function replacementPatches(minimum: SolidWorldCoordinate, before: VoxelBlock,
  after: VoxelBlock): SolidWorldVoxelPatch[] {
  const cells = measuredBlock(before);
  if (measuredBlock(after) !== cells || after.size.some((value, axis) => value !== before.size[axis])) {
    throw new RangeError(`A replacement compares two blocks of one box, not ${before.size.join(" x ")} and ${after.size.join(" x ")}`);
  }
  const size = before.size;
  // 1 the cell must be cleared, 2 it must then be written, 3 both.
  const change = new Uint8Array(cells);
  for (let cell = 0; cell < cells; cell += 1) {
    const fraction = after.solidFraction[cell]!, distance = after.signedDistanceQ8[cell]!;
    if (before.solidFraction[cell] === fraction && before.signedDistanceQ8[cell] === distance
      && before.materialId[cell] === after.materialId[cell]) continue;
    const written = solidFractionOccupies(fraction);
    if (!written && (distance !== EMPTY_DISTANCE_Q8 || after.materialId[cell] !== 0)) {
      throw new RangeError("A replacement can leave a changed cell solid or empty, not empty with a distance or a material");
    }
    const wasEmpty = before.solidFraction[cell] === 0 && before.signedDistanceQ8[cell] === EMPTY_DISTANCE_Q8
      && before.materialId[cell] === 0;
    change[cell] = (wasEmpty ? 0 : 1) | (written ? 2 : 0);
  }
  const patches: SolidWorldVoxelPatch[] = [];
  const cleared = memberBounds(size, (cell) => (change[cell]! & 1) !== 0);
  if (cleared) {
    const mask = cutChannel(change, size, cleared.lo, cleared.size);
    let whole = true;
    for (let cell = 0; cell < mask.length; cell += 1) { mask[cell] = mask[cell]! & 1; whole &&= mask[cell] === 1; }
    const first = offset(minimum, cleared.lo);
    patches.push(...regionPatches("clear", { minimum: first, maximumExclusive: offset(first, cleared.size),
      ...(whole ? {} : { mask }) }));
  }
  const written = memberBounds(size, (cell) => (change[cell]! & 2) !== 0);
  if (written) {
    const marks = cutChannel(change, size, written.lo, written.size);
    const block: VoxelBlock = { size: written.size,
      solidFraction: cutChannel(after.solidFraction, size, written.lo, written.size),
      signedDistanceQ8: cutChannel(after.signedDistanceQ8, size, written.lo, written.size),
      materialId: cutChannel(after.materialId, size, written.lo, written.size) };
    // Cells of the cut that did not change are left out of the stamp: a block leaves a cell of no coverage alone.
    for (let cell = 0; cell < marks.length; cell += 1) if ((marks[cell]! & 2) === 0) {
      block.solidFraction[cell] = 0; block.signedDistanceQ8[cell] = EMPTY_DISTANCE_Q8; block.materialId[cell] = 0;
    }
    patches.push(...blockPatches("fill", offset(minimum, written.lo), block));
  }
  return patches;
}

/**
 * The patches that lift the contents of a region — its solid cells — and set
 * `placed` down instead, as one replacement.
 *
 * Every selection transform is this: what it takes away is the region's solid
 * cells, and what it puts back is those cells flipped, turned, moved, thinned
 * or thickened. Solid cells of `placed` overwrite whatever they land on; its
 * empty cells leave the world alone, as a block's always do. Read over the box
 * that holds both, so a move's vacated and newly covered cells are one edit.
 */
export function contentsReplacementPatches(reader: VoxelReader, region: VoxelRegion,
  placed: PlacedVoxelBlock): SolidWorldVoxelPatch[] {
  const { size: from } = measured(region);
  measuredBlock(placed.block);
  const to = placed.block.size;
  const minimum = region.minimum.map((value, axis) => Math.min(value, placed.minimum[axis]!)) as unknown as SolidWorldCoordinate;
  const maximumExclusive = region.maximumExclusive.map((value, axis) =>
    Math.max(value, placed.minimum[axis]! + to[axis]!)) as unknown as SolidWorldCoordinate;
  const size = boxSize(minimum, maximumExclusive), cells = size[0] * size[1] * size[2];
  bounded(cells, size, "A selection transform");
  const before = reader.read(minimum, maximumExclusive);
  const after: VoxelBlock = { size, solidFraction: before.solidFraction.slice(),
    signedDistanceQ8: before.signedDistanceQ8.slice(), materialId: before.materialId.slice() };
  const strideY = size[0], strideZ = size[0] * size[1];
  const lift = boxSize(minimum, region.minimum);
  for (let z = 0, cell = 0; z < from[2]; z += 1) for (let y = 0; y < from[1]; y += 1) {
    const row = lift[0] + strideY * (lift[1] + y) + strideZ * (lift[2] + z);
    for (let x = 0; x < from[0]; x += 1, cell += 1) {
      if ((region.mask && region.mask[cell] === 0) || !solidFractionOccupies(after.solidFraction[row + x]!)) continue;
      after.solidFraction[row + x] = 0; after.signedDistanceQ8[row + x] = EMPTY_DISTANCE_Q8; after.materialId[row + x] = 0;
    }
  }
  const drop = boxSize(minimum, placed.minimum);
  for (let z = 0, cell = 0; z < to[2]; z += 1) for (let y = 0; y < to[1]; y += 1) {
    const row = drop[0] + strideY * (drop[1] + y) + strideZ * (drop[2] + z);
    for (let x = 0; x < to[0]; x += 1, cell += 1) {
      if (!solidFractionOccupies(placed.block.solidFraction[cell]!)) continue;
      after.solidFraction[row + x] = placed.block.solidFraction[cell]!;
      after.signedDistanceQ8[row + x] = placed.block.signedDistanceQ8[cell]!;
      after.materialId[row + x] = placed.block.materialId[cell]!;
    }
  }
  return replacementPatches(minimum, before, after);
}

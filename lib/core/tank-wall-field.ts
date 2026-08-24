/**
 * Editable tank walls in solver-face space.
 *
 * A wall is not a volume.  It is the closed face between two neighbouring
 * fluid cells, which is also exactly the quad the raster renderer draws.  The
 * document therefore stores four side atlases as run-length encoded bitsets:
 * a set bit means "wall present" and a cleared bit means "opening".
 *
 * Runs are pairs `[start, count]` over row-major cells.  Full box walls cost
 * one run per face while arbitrary brush edits remain compact around their
 * changed cells.
 */

export const TANK_WALL_SIDES = ["left", "right", "front", "back"] as const;
export type TankWallSide = typeof TANK_WALL_SIDES[number];
export type TankWallRun = readonly [start: number, count: number];

/** GPU ABI: magic, xyz dimensions, then one absolute word offset per side. */
export const PACKED_TANK_WALL_HEADER_WORDS = 8;
export const PACKED_TANK_WALL_MAGIC = 0x31574b54; // "TKW1" little-endian

export interface TankWallFaceField {
  readonly uCells: number;
  readonly vCells: number;
  readonly solidRuns: readonly TankWallRun[];
}

export interface TankWallField {
  readonly version: 1;
  /** Tank-interior cell dimensions the four face atlases fit. */
  readonly dimensions: { readonly x: number; readonly y: number; readonly z: number };
  readonly faces: Readonly<Record<TankWallSide, TankWallFaceField>>;
}

const integer = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
};

export function tankWallFaceDimensions(
  dimensions: TankWallField["dimensions"],
  side: TankWallSide,
): readonly [uCells: number, vCells: number] {
  return side === "left" || side === "right"
    ? [dimensions.z, dimensions.y]
    : [dimensions.x, dimensions.y];
}

/** The complete four-sided shell for a tank lattice. */
export function createBoxTankWallField(
  dimensionsInput: TankWallField["dimensions"],
): TankWallField {
  const dimensions = {
    x: integer(dimensionsInput.x, "tank wall x dimension"),
    y: integer(dimensionsInput.y, "tank wall y dimension"),
    z: integer(dimensionsInput.z, "tank wall z dimension"),
  };
  const face = (side: TankWallSide): TankWallFaceField => {
    const [uCells, vCells] = tankWallFaceDimensions(dimensions, side);
    return { uCells, vCells, solidRuns: [[0, uCells * vCells]] };
  };
  return {
    version: 1,
    dimensions,
    faces: { left: face("left"), right: face("right"), front: face("front"), back: face("back") },
  };
}

function normalizedRuns(runs: readonly TankWallRun[], cellCount: number): TankWallRun[] {
  const intervals = runs
    .map(([start, count]) => [Math.max(0, Math.trunc(start)), Math.min(cellCount, Math.trunc(start + count))] as const)
    .filter(([start, end]) => end > start)
    .sort((a, b) => a[0] - b[0]);
  const result: Array<[number, number]> = [];
  for (const [start, end] of intervals) {
    const previous = result.at(-1);
    if (previous && start <= previous[0] + previous[1]) {
      previous[1] = Math.max(previous[0] + previous[1], end) - previous[0];
    } else result.push([start, end - start]);
  }
  return result;
}

export function tankWallCellIsSolid(
  field: TankWallField,
  side: TankWallSide,
  u: number,
  v: number,
): boolean {
  const face = field.faces[side];
  if (!Number.isInteger(u) || !Number.isInteger(v) || u < 0 || v < 0 || u >= face.uCells || v >= face.vCells) {
    return false;
  }
  const cell = u + face.uCells * v;
  return face.solidRuns.some(([start, count]) => cell >= start && cell < start + count);
}

/** Area fraction of a solver-face rectangle that has no authored wall. */
export function tankWallOpeningFraction(
  field: TankWallField,
  side: TankWallSide,
  uMinimum: number,
  uMaximum: number,
  vMinimum: number,
  vMaximum: number,
): number {
  const face = field.faces[side];
  const u0 = Math.max(0, uMinimum), u1 = Math.min(face.uCells, uMaximum);
  const v0 = Math.max(0, vMinimum), v1 = Math.min(face.vCells, vMaximum);
  const area = Math.max(0, u1 - u0) * Math.max(0, v1 - v0);
  if (!(area > 0)) return 0;
  let openArea = 0;
  for (let v = Math.floor(v0); v < Math.ceil(v1); v += 1) {
    const vOverlap = Math.max(0, Math.min(v1, v + 1) - Math.max(v0, v));
    for (let u = Math.floor(u0); u < Math.ceil(u1); u += 1) {
      if (tankWallCellIsSolid(field, side, u, v)) continue;
      const uOverlap = Math.max(0, Math.min(u1, u + 1) - Math.max(u0, u));
      openArea += uOverlap * vOverlap;
    }
  }
  return openArea / area;
}

/** Return a new field with an inclusive wall-face rectangle removed or restored. */
export function withTankWallRectangle(
  field: TankWallField,
  side: TankWallSide,
  uStart: number,
  vStart: number,
  uEnd: number,
  vEnd: number,
  solid: boolean,
): TankWallField {
  const face = field.faces[side];
  if (![uStart, vStart, uEnd, vEnd].every(Number.isFinite)) return field;
  const u0 = Math.max(0, Math.min(face.uCells - 1, Math.floor(Math.min(uStart, uEnd))));
  const u1 = Math.max(0, Math.min(face.uCells - 1, Math.floor(Math.max(uStart, uEnd))));
  const v0 = Math.max(0, Math.min(face.vCells - 1, Math.floor(Math.min(vStart, vEnd))));
  const v1 = Math.max(0, Math.min(face.vCells - 1, Math.floor(Math.max(vStart, vEnd))));
  const cells = new Uint8Array(face.uCells * face.vCells);
  for (const [start, count] of normalizedRuns(face.solidRuns, cells.length)) cells.fill(1, start, start + count);
  const value = solid ? 1 : 0;
  let changed = false;
  for (let v = v0; v <= v1; v += 1) {
    for (let u = u0; u <= u1; u += 1) {
      const cell = u + face.uCells * v;
      if (cells[cell] !== value) changed = true;
      cells[cell] = value;
    }
  }
  if (!changed) return field;
  const runs: Array<[number, number]> = [];
  for (let at = 0; at < cells.length;) {
    if (cells[at] === 0) { at += 1; continue; }
    const start = at;
    while (at < cells.length && cells[at] !== 0) at += 1;
    runs.push([start, at - start]);
  }
  return { ...field, faces: { ...field.faces, [side]: { ...face, solidRuns: runs } } };
}

/** Return a new field with one wall-face cell removed or restored. */
export function withTankWallCell(
  field: TankWallField,
  side: TankWallSide,
  u: number,
  v: number,
  solid: boolean,
): TankWallField {
  if (!Number.isInteger(u) || !Number.isInteger(v)) return field;
  return withTankWallRectangle(field, side, u, v, u, v, solid);
}

export function validateTankWallField(
  field: TankWallField | undefined,
  expectedDimensions?: TankWallField["dimensions"],
): string[] {
  if (!field || field.version !== 1) return ["Tank wall field version must be 1"];
  const errors: string[] = [];
  const d = field.dimensions;
  if (![d?.x, d?.y, d?.z].every((value) => Number.isSafeInteger(value) && value! > 0)) {
    errors.push("Tank wall dimensions must be positive integers");
    return errors;
  }
  if (expectedDimensions && (d.x !== expectedDimensions.x
    || d.y !== expectedDimensions.y || d.z !== expectedDimensions.z)) {
    errors.push("Tank wall dimensions must match the container lattice");
  }
  for (const side of TANK_WALL_SIDES) {
    const face = field.faces?.[side];
    const [uCells, vCells] = tankWallFaceDimensions(d, side);
    if (!face || face.uCells !== uCells || face.vCells !== vCells) {
      errors.push(`Tank wall ${side} dimensions do not match the tank lattice`);
      continue;
    }
    const normalized = normalizedRuns(face.solidRuns, uCells * vCells);
    if (normalized.length !== face.solidRuns.length
      || normalized.some((run, index) => run[0] !== face.solidRuns[index]?.[0] || run[1] !== face.solidRuns[index]?.[1])) {
      errors.push(`Tank wall ${side} runs must be sorted, disjoint and in bounds`);
    }
  }
  return errors;
}

/** Preserve the normalized shape of openings when the container is resized. */
export function resampleTankWallField(
  field: TankWallField,
  dimensions: TankWallField["dimensions"],
): TankWallField {
  const target = createBoxTankWallField(dimensions);
  const faces: Record<TankWallSide, TankWallFaceField> = { ...target.faces };
  for (const side of TANK_WALL_SIDES) {
    const sourceFace = field.faces[side];
    const targetFace = target.faces[side];
    const solidRuns: Array<[number, number]> = [];
    let runStart = -1;
    for (let v = 0; v < targetFace.vCells; v += 1) {
      for (let u = 0; u < targetFace.uCells; u += 1) {
        const sourceU = Math.min(sourceFace.uCells - 1,
          Math.floor((u + 0.5) / targetFace.uCells * sourceFace.uCells));
        const sourceV = Math.min(sourceFace.vCells - 1,
          Math.floor((v + 0.5) / targetFace.vCells * sourceFace.vCells));
        const cell = u + targetFace.uCells * v;
        const solid = tankWallCellIsSolid(field, side, sourceU, sourceV);
        if (solid && runStart < 0) runStart = cell;
        if (!solid && runStart >= 0) { solidRuns.push([runStart, cell - runStart]); runStart = -1; }
      }
    }
    const cellCount = targetFace.uCells * targetFace.vCells;
    if (runStart >= 0) solidRuns.push([runStart, cellCount - runStart]);
    faces[side] = { ...targetFace, solidRuns };
  }
  return { ...target, faces };
}

/** Number of removed face cells, used by UI readouts and tests. */
export function tankWallOpeningCellCount(field: TankWallField): number {
  return TANK_WALL_SIDES.reduce((sum, side) => {
    const face = field.faces[side];
    const solid = face.solidRuns.reduce((count, run) => count + run[1], 0);
    return sum + face.uCells * face.vCells - solid;
  }, 0);
}

/**
 * Dense GPU bitset compiled from the document's compact run representation.
 * Keeping this as a derived buffer lets the editor optimize for tiny scene
 * patches while every solver lookup remains one offset, load and bit test.
 */
export function packTankWallField(field: TankWallField): Uint32Array<ArrayBuffer> {
  const offsets: number[] = [];
  let wordCount = PACKED_TANK_WALL_HEADER_WORDS;
  for (const side of TANK_WALL_SIDES) {
    offsets.push(wordCount);
    const face = field.faces[side];
    wordCount += Math.ceil(face.uCells * face.vCells / 32);
  }
  const packed = new Uint32Array(wordCount);
  packed[0] = PACKED_TANK_WALL_MAGIC;
  packed[1] = field.dimensions.x;
  packed[2] = field.dimensions.y;
  packed[3] = field.dimensions.z;
  for (let sideIndex = 0; sideIndex < TANK_WALL_SIDES.length; sideIndex += 1) {
    const side = TANK_WALL_SIDES[sideIndex]!;
    const offset = offsets[sideIndex]!;
    packed[4 + sideIndex] = offset;
    for (const [start, count] of field.faces[side].solidRuns) {
      for (let cell = start; cell < start + count; cell += 1) {
        packed[offset + (cell >>> 5)]! |= 1 << (cell & 31);
      }
    }
  }
  return packed;
}

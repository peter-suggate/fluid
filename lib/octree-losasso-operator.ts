/**
 * Compact first-order Losasso pressure authority.
 *
 * The backend publishes one record per physical, axis-aligned face. A coarse
 * face meeting finer leaves is represented by its four fine subfaces; there
 * is no transition case, descriptor, catalogue, or edge-velocity channel.
 * The coefficient is the closed form
 *
 *     openFraction * faceArea / centreDistance.
 *
 * Keeping the unique-face authority separate from the row CSR makes the two
 * incident rows consume the same rounded coefficient and therefore preserves
 * symmetry at 2:1 and terrain-cut faces.
 */

export const OCTREE_LOSASSO_INVALID_ROW = 0xffff_ffff;
export const OCTREE_LOSASSO_FACE_WORDS = 8;
export const OCTREE_LOSASSO_FACE_BYTES = OCTREE_LOSASSO_FACE_WORDS * 4;
export const OCTREE_LOSASSO_CONTROL_WORDS = 8;

export type OctreeLosassoAxis = 0 | 1 | 2;

export interface OctreeLosassoCell {
  readonly key: string;
  readonly origin: readonly [number, number, number];
  readonly size: number;
  /** Fluid volume after any terrain cut. Defaults to size^3. */
  readonly volume?: number;
}

export interface OctreeLosassoAxisFaceInput {
  /** Row on the negative side of the stored positive-axis normal. */
  readonly negativeRow: number;
  /** Undefined denotes a physical/free-surface boundary face. */
  readonly positiveRow?: number;
  readonly axis: OctreeLosassoAxis;
  /** Geometric open face area before the cut-cell fraction. */
  readonly area: number;
  /** Shared solid aperture. Zero removes pressure coupling but keeps flux. */
  readonly openFraction: number;
  /**
   * Optional explicit centre distance. Internal faces default to half the sum
   * of incident cell sizes, the first-order Losasso distance. Boundary faces
   * with no distance are closed Neumann faces.
   */
  readonly centreDistance?: number;
  readonly normalVelocity?: number;
}

export interface OctreeLosassoFace {
  readonly id: number;
  readonly negativeRow: number;
  readonly positiveRow: number;
  readonly axis: OctreeLosassoAxis;
  readonly area: number;
  readonly inverseDistance: number;
  readonly openFraction: number;
  readonly normalVelocity: number;
  readonly coefficient: number;
}

export interface OctreeLosassoRow {
  readonly row: number;
  readonly key: string;
  readonly volume: number;
  readonly faceOffset: number;
  readonly faceCount: number;
  readonly diagonal: number;
  readonly integratedRhs: number;
}

export interface OctreeLosassoOperator {
  readonly cells: readonly OctreeLosassoCell[];
  readonly rows: readonly OctreeLosassoRow[];
  readonly faces: readonly OctreeLosassoFace[];
  /** CSR offsets followed by face ids. Face orientation comes from the face. */
  readonly rowFaceOffsets: Uint32Array;
  readonly rowFaces: Uint32Array;
  readonly maximumFacesPerRow: number;
}

export interface OctreeLosassoPackedAuthority {
  readonly control: Uint32Array;
  readonly rowFaceOffsets: Uint32Array;
  readonly rowFaces: Uint32Array;
  /** Eight words per face, interpreted by `octreeLosassoOperatorWGSL`. */
  readonly faces: Uint32Array;
}

function finitePositive(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${label} must be positive and finite`);
  return value;
}

function rowIndex(value: number, count: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value >= count) {
    throw new RangeError(`${label} must name an existing Losasso row`);
  }
  return value;
}

function validateCell(cell: OctreeLosassoCell, index: number): void {
  if (cell.key.length === 0 || cell.origin.length !== 3
    || cell.origin.some((coordinate) => !Number.isFinite(coordinate))) {
    throw new RangeError(`Losasso cell ${index} needs a key and finite origin`);
  }
  finitePositive(cell.size, `Losasso cell ${index} size`);
  finitePositive(cell.volume ?? cell.size ** 3, `Losasso cell ${index} volume`);
}

/** Build deterministic compact rows from an already unique physical-face list. */
export function buildOctreeLosassoOperator(
  inputCells: readonly OctreeLosassoCell[],
  inputFaces: readonly OctreeLosassoAxisFaceInput[],
): OctreeLosassoOperator {
  if (inputCells.length === 0) throw new RangeError("Losasso operator requires at least one cell");
  inputCells.forEach(validateCell);
  if (new Set(inputCells.map((cell) => cell.key)).size !== inputCells.length) {
    throw new RangeError("Losasso cell keys must be unique");
  }
  const cells = inputCells.map((cell) => {
    const origin = Object.freeze([cell.origin[0], cell.origin[1], cell.origin[2]] as const);
    return Object.freeze({ ...cell, origin });
  });
  const incidence = cells.map(() => [] as number[]);
  const pairAxes = new Set<string>();
  const faces = inputFaces.map((input, id): OctreeLosassoFace => {
    const negativeRow = rowIndex(input.negativeRow, cells.length, `Losasso face ${id} negative row`);
    const positiveRow = input.positiveRow === undefined ? OCTREE_LOSASSO_INVALID_ROW
      : rowIndex(input.positiveRow, cells.length, `Losasso face ${id} positive row`);
    if (positiveRow === negativeRow) throw new RangeError(`Losasso face ${id} is self-incident`);
    if (input.axis !== 0 && input.axis !== 1 && input.axis !== 2) {
      throw new RangeError(`Losasso face ${id} axis must be 0, 1, or 2`);
    }
    const area = finitePositive(input.area, `Losasso face ${id} area`);
    if (!Number.isFinite(input.openFraction) || input.openFraction < 0 || input.openFraction > 1) {
      throw new RangeError(`Losasso face ${id} open fraction must be in [0, 1]`);
    }
    const normalVelocity = input.normalVelocity ?? 0;
    if (!Number.isFinite(normalVelocity)) throw new RangeError(`Losasso face ${id} velocity must be finite`);
    const defaultDistance = positiveRow === OCTREE_LOSASSO_INVALID_ROW ? undefined
      : 0.5 * (cells[negativeRow]!.size + cells[positiveRow]!.size);
    const distance = input.centreDistance ?? defaultDistance;
    if (distance !== undefined) finitePositive(distance, `Losasso face ${id} centre distance`);
    if (positiveRow !== OCTREE_LOSASSO_INVALID_ROW) {
      const first = Math.min(negativeRow, positiveRow), second = Math.max(negativeRow, positiveRow);
      const key = `${first}:${second}:${input.axis}`;
      if (pairAxes.has(key)) throw new RangeError(`Losasso face ${id} duplicates an incident row pair`);
      pairAxes.add(key);
      const ratio = Math.max(cells[negativeRow]!.size, cells[positiveRow]!.size)
        / Math.min(cells[negativeRow]!.size, cells[positiveRow]!.size);
      if (ratio > 2 + 1e-6) throw new RangeError(`Losasso face ${id} violates 2:1 grading`);
      const negative = cells[negativeRow]!, positive = cells[positiveRow]!;
      const scale = Math.max(negative.size, positive.size);
      if (Math.abs(negative.origin[input.axis] + negative.size - positive.origin[input.axis])
        > 1e-6 * scale) {
        throw new RangeError(`Losasso face ${id} rows are not ordered across its positive axis`);
      }
      let geometricArea = 1;
      for (let axis = 0; axis < 3; axis += 1) {
        if (axis === input.axis) continue;
        const overlap = Math.min(negative.origin[axis] + negative.size,
          positive.origin[axis] + positive.size)
          - Math.max(negative.origin[axis], positive.origin[axis]);
        if (overlap <= 0) throw new RangeError(`Losasso face ${id} incident cells do not overlap`);
        geometricArea *= overlap;
      }
      if (Math.abs(area - geometricArea) > 1e-5 * Math.max(1, geometricArea)) {
        throw new RangeError(`Losasso face ${id} area does not match its axis-aligned overlap`);
      }
    }
    const inverseDistance = distance === undefined ? 0 : 1 / distance;
    const coefficient = Math.fround(Math.fround(input.openFraction * area) * inverseDistance);
    incidence[negativeRow]!.push(id);
    if (positiveRow !== OCTREE_LOSASSO_INVALID_ROW) incidence[positiveRow]!.push(id);
    return Object.freeze({ id, negativeRow, positiveRow, axis: input.axis, area,
      inverseDistance, openFraction: input.openFraction, normalVelocity, coefficient });
  });
  for (const row of incidence) row.sort((a, b) => a - b);
  const rowFaceOffsets = new Uint32Array(cells.length + 1);
  const flatFaces: number[] = [];
  const rows = cells.map((cell, row): OctreeLosassoRow => {
    rowFaceOffsets[row] = flatFaces.length;
    flatFaces.push(...incidence[row]!);
    let diagonal = 0, integratedRhs = 0;
    for (const faceId of incidence[row]!) {
      const face = faces[faceId]!;
      const sign = face.negativeRow === row ? 1 : -1;
      diagonal = Math.fround(diagonal + face.coefficient);
      integratedRhs = Math.fround(integratedRhs
        + Math.fround(sign * face.area * face.openFraction * face.normalVelocity));
    }
    return Object.freeze({ row, key: cell.key, volume: cell.volume ?? cell.size ** 3,
      faceOffset: rowFaceOffsets[row]!, faceCount: incidence[row]!.length,
      diagonal, integratedRhs });
  });
  rowFaceOffsets[cells.length] = flatFaces.length;
  return Object.freeze({ cells, rows, faces, rowFaceOffsets,
    rowFaces: Uint32Array.from(flatFaces),
    maximumFacesPerRow: Math.max(...incidence.map((row) => row.length)) });
}

/** Apply the exact f32 fold used by the compact GPU operator. */
export function applyOctreeLosassoMatrix(
  operator: OctreeLosassoOperator,
  input: ArrayLike<number>,
): Float32Array {
  if (input.length !== operator.rows.length) throw new RangeError("Losasso vector length must match rows");
  const result = new Float32Array(input.length);
  for (const row of operator.rows) {
    const centre = Math.fround(input[row.row]!);
    if (!Number.isFinite(centre)) throw new RangeError(`Losasso vector row ${row.row} is non-finite`);
    let value = 0;
    for (let cursor = row.faceOffset; cursor < row.faceOffset + row.faceCount; cursor += 1) {
      const face = operator.faces[operator.rowFaces[cursor]!]!;
      let difference = centre;
      const neighbor = face.negativeRow === row.row ? face.positiveRow : face.negativeRow;
      if (neighbor !== OCTREE_LOSASSO_INVALID_ROW) {
        const neighborValue = Math.fround(input[neighbor]!);
        if (!Number.isFinite(neighborValue)) throw new RangeError(`Losasso vector row ${neighbor} is non-finite`);
        difference = Math.fround(centre - neighborValue);
      }
      value = Math.fround(value + Math.fround(face.coefficient * difference));
    }
    result[row.row] = value;
  }
  return result;
}

/** Pack the reduced production ABI; no Power descriptor/catalog fields exist. */
export function packOctreeLosassoAuthority(
  operator: OctreeLosassoOperator,
  epoch: number,
): OctreeLosassoPackedAuthority {
  if (!Number.isSafeInteger(epoch) || epoch < 1 || epoch > 0xffff_ffff) {
    throw new RangeError("Losasso epoch must be a positive u32");
  }
  const words = new Uint32Array(operator.faces.length * OCTREE_LOSASSO_FACE_WORDS);
  const floats = new Float32Array(words.buffer);
  for (const face of operator.faces) {
    const base = face.id * OCTREE_LOSASSO_FACE_WORDS;
    words[base] = face.negativeRow;
    words[base + 1] = face.positiveRow;
    words[base + 2] = face.axis;
    words[base + 3] = 0;
    floats[base + 4] = face.area;
    floats[base + 5] = face.inverseDistance;
    floats[base + 6] = face.openFraction;
    floats[base + 7] = face.normalVelocity;
  }
  return Object.freeze({
    control: Uint32Array.of(epoch, operator.rows.length, operator.faces.length, 1, 0, 0, 0, 0),
    rowFaceOffsets: new Uint32Array(operator.rowFaceOffsets),
    rowFaces: new Uint32Array(operator.rowFaces),
    faces: words,
  });
}

export interface OctreeLosassoStoragePlan {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly incidenceCapacity: number;
  readonly allocatedBytes: number;
}

export function planOctreeLosassoStorage(
  rowCapacity: number,
  faceCapacity: number,
  incidenceCapacity = 2 * faceCapacity,
): OctreeLosassoStoragePlan {
  for (const [label, value] of [["row", rowCapacity], ["face", faceCapacity],
    ["incidence", incidenceCapacity]] as const) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`Losasso ${label} capacity must be positive`);
  }
  if (incidenceCapacity > 2 * faceCapacity) {
    throw new RangeError("Losasso unique faces cannot have more than two incidences each");
  }
  return Object.freeze({ rowCapacity, faceCapacity, incidenceCapacity,
    allocatedBytes: OCTREE_LOSASSO_CONTROL_WORDS * 4 + (rowCapacity + 1) * 4
      + incidenceCapacity * 4 + faceCapacity * OCTREE_LOSASSO_FACE_BYTES });
}

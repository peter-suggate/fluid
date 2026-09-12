/**
 * Two-dimensional projection of Sparse CM12's accepted topology authority.
 *
 * The Z axis is removed, but the remaining records follow the production
 * composite-grid conventions: B8 stable brick slots, dyadic brick spans and
 * rungs, compact accepted cells, one oriented G row per physical face port,
 * incidence records, and pairwise physical subfaces for geometric transport.
 */

export const SLICE_TOPOLOGY_BRICK_FINE_RESOLUTION = 8 as const;
export const SLICE_TOPOLOGY_BRICK_CELL_CAPACITY = 8 ** 2;
export const SLICE_TOPOLOGY_INVALID = 0xffff_ffff;

export type SliceTopologyAxis = 0 | 1;
export type SliceTopologyResolution = 1 | 2 | 4 | 8;
export type SliceTopologyVec2 = readonly [number, number];
export type SliceTopologyRowKind =
  | "intra-brick" | "brick-face" | "mixed-seam" | "sparse-air";
export type SliceTopologyBoundaryMode = "open" | "closed";
export interface SliceTopologyBoundaryModes {
  readonly negativeX: SliceTopologyBoundaryMode;
  readonly positiveX: SliceTopologyBoundaryMode;
  readonly negativeY: SliceTopologyBoundaryMode;
  readonly positiveY: SliceTopologyBoundaryMode;
}

export interface SliceTopologyBrick {
  /** Stable physical leaf id. It survives accepted/candidate generations. */
  readonly id: number;
  /** Stable logical key. Keys need only be unique within this authority. */
  readonly key: number;
  readonly coordinate: SliceTopologyVec2;
  readonly spanBricks?: number;
  readonly resolution: SliceTopologyResolution;
  /** Inactive allocated leaves remain in the directory but publish no cells. */
  readonly active?: boolean;
  readonly density?: ArrayLike<number>;
  readonly gamma?: ArrayLike<number>;
}

export interface SliceTopologyCell {
  readonly id: number;
  readonly stableLeafId: number;
  readonly brickId: number;
  readonly brickKey: number;
  readonly brickCoordinate: SliceTopologyVec2;
  readonly brickResolution: SliceTopologyResolution;
  readonly local: SliceTopologyVec2;
  readonly localIndex: number;
  readonly minimumFine: SliceTopologyVec2;
  readonly maximumFine: SliceTopologyVec2;
  readonly centerFine: SliceTopologyVec2;
  readonly widthsFine: SliceTopologyVec2;
  /** Unit-depth control volume, in finest-cell squared units. */
  readonly volumeFineCells: number;
  readonly density: number;
  readonly gamma: number;
}

export interface SliceTopologyTerm {
  readonly cellId: number;
  readonly coefficient: number;
}

export interface SliceTopologyRow {
  readonly id: number;
  readonly kind: SliceTopologyRowKind;
  readonly axis: SliceTopologyAxis;
  readonly centerFine: SliceTopologyVec2;
  /** Unit-depth face measure, in finest-cell units. */
  readonly areaFineCells: number;
  readonly centerDistanceFine: number;
  readonly dualWeight: number;
  readonly terms: readonly SliceTopologyTerm[];
  readonly negativeBrickKey?: number;
  readonly positiveBrickKey?: number;
  readonly exteriorPhi?: number;
  /** Only populated for a one-term row on the physical domain boundary. */
  readonly boundaryMode?: SliceTopologyBoundaryMode;
}

export interface SliceTopologyIncidence {
  readonly row: number;
  readonly termOrdinal: number;
}

/** One pairwise face piece consumed by geometric volume transport. */
export interface SliceTopologySubface {
  readonly id: number;
  readonly row: number;
  readonly axis: SliceTopologyAxis;
  readonly centerFine: SliceTopologyVec2;
  readonly areaFineCells: number;
  readonly negativeCell: number;
  readonly positiveCell: number;
}

export interface SliceTopology {
  readonly generation: number;
  readonly dimensions: SliceTopologyVec2;
  readonly brickFineResolution: 8;
  readonly brickCellCapacity: 64;
  readonly bricks: readonly SliceTopologyBrick[];
  readonly brickByKey: ReadonlyMap<number, SliceTopologyBrick>;
  readonly cellBaseByBrick: ReadonlyMap<number, number>;
  readonly cells: readonly SliceTopologyCell[];
  readonly rows: readonly SliceTopologyRow[];
  readonly incidenceOffsets: Uint32Array;
  readonly incidences: readonly SliceTopologyIncidence[];
  readonly subfaces: readonly SliceTopologySubface[];
  readonly mixedSeamRowCount: number;
  readonly sparseAirRowCount: number;
  readonly boundaryModes: SliceTopologyBoundaryModes;
}

export interface SliceTopologyAuthority {
  readonly accepted: SliceTopology;
  readonly candidate?: SliceTopology;
}

const rung = (value: number): value is SliceTopologyResolution =>
  value === 1 || value === 2 || value === 4 || value === 8;
const spanOf = (brick: SliceTopologyBrick): number => brick.spanBricks ?? 1;
const f32 = Math.fround;

function positiveDimensions(dimensions: SliceTopologyVec2): void {
  if (!dimensions.every(value => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("slice topology dimensions must be positive integers");
  }
}

function validateBrick(brick: SliceTopologyBrick): void {
  const span = spanOf(brick);
  if (!Number.isSafeInteger(brick.id) || brick.id < 0
    || !Number.isSafeInteger(brick.key) || brick.key < 0) {
    throw new RangeError("slice topology brick ids and keys must be nonnegative integers");
  }
  if (!Number.isSafeInteger(span) || span < 1 || (span & (span - 1)) !== 0) {
    throw new RangeError(`slice topology brick ${brick.key} span must be dyadic`);
  }
  if (!brick.coordinate.every(value => Number.isSafeInteger(value) && value % span === 0)) {
    throw new RangeError(`slice topology brick ${brick.key} origin must align to span ${span}`);
  }
  if (!rung(brick.resolution)) {
    throw new RangeError(`slice topology brick ${brick.key} resolution is off the 1/2/4/8 ladder`);
  }
  const count = brick.resolution ** 2;
  if (brick.density && brick.density.length !== count) {
    throw new RangeError(`slice topology brick ${brick.key} density does not match its rung`);
  }
  if (brick.gamma && brick.gamma.length !== count) {
    throw new RangeError(`slice topology brick ${brick.key} gamma does not match its rung`);
  }
}

function overlap(a0: number, a1: number, b0: number, b1: number): number {
  return Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
}

function brickBounds(brick: SliceTopologyBrick, dimensions: SliceTopologyVec2) {
  const span = spanOf(brick) * SLICE_TOPOLOGY_BRICK_FINE_RESOLUTION;
  const minimum = brick.coordinate.map(value =>
    value * SLICE_TOPOLOGY_BRICK_FINE_RESOLUTION) as [number, number];
  const maximum = minimum.map((value, axis) =>
    Math.min(dimensions[axis]!, value + span)) as [number, number];
  return { minimum, maximum };
}

function bricksShareFace(a: SliceTopologyBrick, b: SliceTopologyBrick,
  dimensions: SliceTopologyVec2): boolean {
  const ab = brickBounds(a, dimensions), bb = brickBounds(b, dimensions);
  return (ab.maximum[0] === bb.minimum[0] || bb.maximum[0] === ab.minimum[0])
      && overlap(ab.minimum[1], ab.maximum[1], bb.minimum[1], bb.maximum[1]) > 0
    || (ab.maximum[1] === bb.minimum[1] || bb.maximum[1] === ab.minimum[1])
      && overlap(ab.minimum[0], ab.maximum[0], bb.minimum[0], bb.maximum[0]) > 0;
}

interface AtomicSubface {
  axis: SliceTopologyAxis;
  coordinate: number;
  lower: number;
  upper: number;
  negativeCell: number;
  positiveCell: number;
}

interface FaceRecord {
  cell: number;
  side: -1 | 1;
  lower: number;
  upper: number;
}

/** Compile the immutable accepted or candidate topology image for one epoch. */
export function compileSliceTopology(
  seedBricks: readonly SliceTopologyBrick[],
  dimensions: SliceTopologyVec2,
  generation = 1,
  sparseAirPhi = 0.5,
  boundaryModes: SliceTopologyBoundaryModes = {
    negativeX: "open", positiveX: "open", negativeY: "open", positiveY: "open",
  },
): SliceTopology {
  positiveDimensions(dimensions);
  if (!Number.isSafeInteger(generation) || generation < 1) {
    throw new RangeError("slice topology generation must be a positive integer");
  }
  if (!(Number.isFinite(sparseAirPhi) && sparseAirPhi > 0)) {
    throw new RangeError("slice sparse-air phi must be finite and positive");
  }
  for (const brick of seedBricks) validateBrick(brick);
  const ids = new Set<number>(), keys = new Set<number>();
  for (const brick of seedBricks) {
    if (ids.has(brick.id) || keys.has(brick.key)) {
      throw new Error("slice topology brick ids and keys must be unique");
    }
    ids.add(brick.id); keys.add(brick.key);
  }
  const bricks = [...seedBricks].sort((a, b) => a.key - b.key);
  for (let i = 0; i < bricks.length; i++) for (let j = i + 1; j < bricks.length; j++) {
    const a = brickBounds(bricks[i]!, dimensions), b = brickBounds(bricks[j]!, dimensions);
    if (overlap(a.minimum[0], a.maximum[0], b.minimum[0], b.maximum[0]) > 0
      && overlap(a.minimum[1], a.maximum[1], b.minimum[1], b.maximum[1]) > 0) {
      throw new Error(`slice topology bricks ${bricks[i]!.key}/${bricks[j]!.key} overlap`);
    }
    if (bricksShareFace(bricks[i]!, bricks[j]!, dimensions)) {
      const aw = 8 * spanOf(bricks[i]!) / bricks[i]!.resolution;
      const bw = 8 * spanOf(bricks[j]!) / bricks[j]!.resolution;
      if (Math.max(aw, bw) / Math.min(aw, bw) > 2) {
        throw new Error(`slice topology face ${bricks[i]!.key}/${bricks[j]!.key} exceeds 2:1 grading`);
      }
    }
  }

  const cells: SliceTopologyCell[] = [];
  const cellBaseByBrick = new Map<number, number>();
  for (const brick of bricks) {
    cellBaseByBrick.set(brick.key, cells.length);
    if (brick.active === false) continue;
    const bounds = brickBounds(brick, dimensions);
    const scale = 8 * spanOf(brick) / brick.resolution;
    for (let y = 0; y < brick.resolution; y++) for (let x = 0; x < brick.resolution; x++) {
      const localIndex = x + brick.resolution * y;
      const minimum = [brick.coordinate[0] * 8 + x * scale,
        brick.coordinate[1] * 8 + y * scale] as [number, number];
      const maximum = [Math.min(minimum[0] + scale, bounds.maximum[0]),
        Math.min(minimum[1] + scale, bounds.maximum[1])] as [number, number];
      const widths = [maximum[0] - minimum[0], maximum[1] - minimum[1]] as const;
      if (widths[0] <= 0 || widths[1] <= 0) continue;
      const id = cells.length;
      cells.push(Object.freeze({
        id, stableLeafId: brick.key * 64 + localIndex,
        brickId: brick.id, brickKey: brick.key, brickCoordinate: brick.coordinate,
        brickResolution: brick.resolution, local: [x, y] as const, localIndex,
        minimumFine: minimum, maximumFine: maximum,
        centerFine: [f32(0.5 * (minimum[0] + maximum[0])),
          f32(0.5 * (minimum[1] + maximum[1]))] as const,
        widthsFine: widths, volumeFineCells: f32(widths[0] * widths[1]),
        density: f32(brick.density?.[localIndex] ?? 0),
        gamma: f32(brick.gamma?.[localIndex] ?? 1),
      }));
    }
  }

  const facePlanes = ([0, 1] as const).map(() => new Map<number, FaceRecord[]>());
  for (const cell of cells) for (const axis of [0, 1] as const) {
    const tangent = (1 - axis) as SliceTopologyAxis;
    for (const side of [-1, 1] as const) {
      const coordinate = side < 0 ? cell.minimumFine[axis] : cell.maximumFine[axis];
      let records = facePlanes[axis].get(coordinate);
      if (!records) facePlanes[axis].set(coordinate, records = []);
      records.push({ cell: cell.id, side,
        lower: cell.minimumFine[tangent], upper: cell.maximumFine[tangent] });
    }
  }

  const atomic: AtomicSubface[] = [];
  for (const axis of [0, 1] as const) for (const [coordinate, faces] of facePlanes[axis]) {
    const stops = [...new Set(faces.flatMap(face => [face.lower, face.upper]))]
      .sort((a, b) => a - b);
    for (let at = 0; at + 1 < stops.length; at++) {
      const lower = stops[at]!, upper = stops[at + 1]!;
      if (!(upper > lower)) continue;
      const midpoint = 0.5 * (lower + upper);
      const negative = faces.filter(face => face.side > 0
        && face.lower <= midpoint && face.upper >= midpoint);
      const positive = faces.filter(face => face.side < 0
        && face.lower <= midpoint && face.upper >= midpoint);
      if (negative.length > 1 || positive.length > 1) {
        throw new Error(`slice topology face plane ${axis}/${coordinate}/${midpoint} has overlapping ownership`);
      }
      if (negative.length === 0 && positive.length === 0) continue;
      atomic.push({ axis, coordinate, lower, upper,
        negativeCell: negative[0]?.cell ?? SLICE_TOPOLOGY_INVALID,
        positiveCell: positive[0]?.cell ?? SLICE_TOPOLOGY_INVALID });
    }
  }

  // Production groups the fine pieces of a mixed interface into the coarse
  // pressure port, while geometric transport retains pairwise subfaces.
  const groups = new Map<string, AtomicSubface[]>();
  for (const face of atomic) {
    const negative = face.negativeCell === SLICE_TOPOLOGY_INVALID
      ? undefined : cells[face.negativeCell];
    const positive = face.positiveCell === SLICE_TOPOLOGY_INVALID
      ? undefined : cells[face.positiveCell];
    let key: string;
    if (!negative || !positive) {
      const cell = negative ?? positive!;
      const side = negative ? 1 : -1;
      key = `air/${face.axis}/${face.coordinate}/${cell.id}/${side}/${face.lower}`;
    } else {
      const nw = negative.widthsFine[face.axis], pw = positive.widthsFine[face.axis];
      if (nw === pw) key = `pair/${face.axis}/${face.coordinate}/${negative.id}/${positive.id}`;
      else {
        const coarse = nw > pw ? negative : positive;
        key = `mixed/${face.axis}/${face.coordinate}/${coarse.id}`;
      }
    }
    let list = groups.get(key); if (!list) groups.set(key, list = []);
    list.push(face);
  }

  const rows: SliceTopologyRow[] = [];
  const subfaces: SliceTopologySubface[] = [];
  let mixedSeamRowCount = 0, sparseAirRowCount = 0;
  const brickOrder = new Map(bricks.map((brick, index) => [brick.key, index]));
  const groupOrder = (pieces: readonly AtomicSubface[]): readonly number[] => {
    const first = pieces[0]!;
    const negative = first.negativeCell === SLICE_TOPOLOGY_INVALID
      ? undefined : cells[first.negativeCell];
    const positive = first.positiveCell === SLICE_TOPOLOGY_INVALID
      ? undefined : cells[first.positiveCell];
    if (negative && positive && negative.brickKey === positive.brickKey) {
      // Production emits every brick's ordinary rows before any seam row.
      return [0, brickOrder.get(negative.brickKey)!, first.axis,
        positive.local[1], positive.local[0], first.lower];
    }
    if (negative && positive) {
      // Then, brick-major and axis-major: positive interfaces before air.
      return [1, brickOrder.get(negative.brickKey)!, first.axis, 0,
        brickOrder.get(positive.brickKey)!, first.lower];
    }
    const own = negative ?? positive!;
    // Positive sparse-air patches precede negative patches for each axis.
    return [1, brickOrder.get(own.brickKey)!, first.axis, negative ? 1 : 2,
      own.local[1], own.local[0], first.lower];
  };
  const compareOrder = (a: readonly number[], b: readonly number[]): number => {
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
      const delta = (a[i] ?? 0) - (b[i] ?? 0);
      if (delta) return delta;
    }
    return 0;
  };
  const orderedGroups = [...groups.values()].sort((a, b) =>
    compareOrder(groupOrder(a), groupOrder(b)));
  for (const pieces of orderedGroups) {
    pieces.sort((a, b) => a.lower - b.lower);
    const first = pieces[0]!;
    const negativeCells = [...new Set(pieces.map(piece => piece.negativeCell)
      .filter(id => id !== SLICE_TOPOLOGY_INVALID))];
    const positiveCells = [...new Set(pieces.map(piece => piece.positiveCell)
      .filter(id => id !== SLICE_TOPOLOGY_INVALID))];
    const lower = Math.min(...pieces.map(piece => piece.lower));
    const upper = Math.max(...pieces.map(piece => piece.upper));
    const area = f32(pieces.reduce((sum, piece) => sum + piece.upper - piece.lower, 0));
    const negativeCenter = negativeCells.length === 0 ? undefined
      : pieces.reduce((sum, piece) => piece.negativeCell === SLICE_TOPOLOGY_INVALID ? sum
        : sum + (piece.upper - piece.lower) * cells[piece.negativeCell]!.centerFine[first.axis], 0) / area;
    const positiveCenter = positiveCells.length === 0 ? undefined
      : pieces.reduce((sum, piece) => piece.positiveCell === SLICE_TOPOLOGY_INVALID ? sum
        : sum + (piece.upper - piece.lower) * cells[piece.positiveCell]!.centerFine[first.axis], 0) / area;
    const only = negativeCells.length ? cells[negativeCells[0]!]! : cells[positiveCells[0]!]!;
    const distance = f32(negativeCenter !== undefined && positiveCenter !== undefined
      ? positiveCenter - negativeCenter : only.widthsFine[first.axis]);
    const terms: SliceTopologyTerm[] = [];
    const appendTerms = (idsForSide: readonly number[], sign: -1 | 1): void => {
      for (const id of idsForSide) {
        const covered = pieces.reduce((sum, piece) => sum
          + ((sign < 0 ? piece.negativeCell : piece.positiveCell) === id
            ? piece.upper - piece.lower : 0), 0);
        terms.push(Object.freeze({ cellId: id,
          coefficient: f32(sign * covered / (area * distance)) }));
      }
    };
    appendTerms(negativeCells, -1); appendTerms(positiveCells, 1);
    let kind: SliceTopologyRowKind;
    if (!negativeCells.length || !positiveCells.length) kind = "sparse-air";
    else {
      const negative = cells[negativeCells[0]!]!, positive = cells[positiveCells[0]!]!;
      kind = negative.widthsFine[first.axis] !== positive.widthsFine[first.axis]
        ? "mixed-seam" : negative.brickKey === positive.brickKey
          ? "intra-brick" : "brick-face";
    }
    if (kind === "mixed-seam") mixedSeamRowCount++;
    if (kind === "sparse-air") sparseAirRowCount++;
    const id = rows.length;
    const negativeKey = negativeCells.length ? cells[negativeCells[0]!]!.brickKey : undefined;
    const positiveKey = positiveCells.length ? cells[positiveCells[0]!]!.brickKey : undefined;
    let boundaryMode: SliceTopologyBoundaryMode | undefined;
    if (kind === "sparse-air") {
      if (first.axis === 0 && first.coordinate === 0) boundaryMode = boundaryModes.negativeX;
      else if (first.axis === 0 && first.coordinate === dimensions[0]) boundaryMode = boundaryModes.positiveX;
      else if (first.axis === 1 && first.coordinate === 0) boundaryMode = boundaryModes.negativeY;
      else if (first.axis === 1 && first.coordinate === dimensions[1]) boundaryMode = boundaryModes.positiveY;
    }
    const centerFine: SliceTopologyVec2 = first.axis === 0
      ? [first.coordinate, f32(0.5 * (lower + upper))]
      : [f32(0.5 * (lower + upper)), first.coordinate];
    rows.push(Object.freeze({ id, kind, axis: first.axis,
      centerFine,
      areaFineCells: area, centerDistanceFine: distance,
      dualWeight: f32(area * distance), terms: Object.freeze(terms),
      negativeBrickKey: negativeKey, positiveBrickKey: positiveKey,
      ...(kind === "sparse-air" ? { exteriorPhi: sparseAirPhi } : {}),
      ...(boundaryMode ? { boundaryMode } : {}),
    }));
    for (const piece of pieces) {
      const subfaceCenter: SliceTopologyVec2 = piece.axis === 0
        ? [piece.coordinate, f32(0.5 * (piece.lower + piece.upper))]
        : [f32(0.5 * (piece.lower + piece.upper)), piece.coordinate];
      subfaces.push(Object.freeze({
      id: subfaces.length, row: id, axis: piece.axis,
      centerFine: subfaceCenter,
      areaFineCells: f32(piece.upper - piece.lower),
      negativeCell: piece.negativeCell, positiveCell: piece.positiveCell,
      }));
    }
  }

  const byCell: SliceTopologyIncidence[][] = Array.from({ length: cells.length }, () => []);
  for (const row of rows) row.terms.forEach((term, termOrdinal) =>
    byCell[term.cellId]!.push(Object.freeze({ row: row.id, termOrdinal })));
  const templateLevels = [1, 2, 4, 8] as const;
  // Resident incidence is packed after sparseCM12ContiguousRowOwnership. A
  // row belongs to the lowest physical brick slot in its requirement set and
  // then to that brick's resolution bucket; source-row order is retained
  // inside the bucket. This order is observable because GV_CELL_FACE walks the
  // incidence stream directly and performs f32 limiter budget reductions in
  // that order.
  const incidenceBucket = new Uint32Array(rows.length);
  for (const row of rows) {
    let owner = Number.MAX_SAFE_INTEGER;
    let ownerResolution: SliceTopologyResolution = 8;
    for (const term of row.terms) {
      const cell = cells[term.cellId]!;
      const brick = brickOrder.get(cell.brickKey)!;
      if (brick < owner) {
        owner = brick;
        ownerResolution = cell.brickResolution;
      }
    }
    const level = templateLevels.indexOf(ownerResolution);
    if (!Number.isSafeInteger(owner) || level < 0) {
      throw new Error(`slice topology row ${row.id} has no valid template owner`);
    }
    incidenceBucket[row.id] = owner * templateLevels.length + level;
  }
  const incidenceOffsets = new Uint32Array(cells.length + 1);
  const incidences: SliceTopologyIncidence[] = [];
  for (let cell = 0; cell < cells.length; cell++) {
    incidenceOffsets[cell] = incidences.length;
    byCell[cell]!.sort((a, b) => incidenceBucket[a.row]! - incidenceBucket[b.row]!
      || a.row - b.row || a.termOrdinal - b.termOrdinal);
    incidences.push(...byCell[cell]!);
  }
  incidenceOffsets[cells.length] = incidences.length;

  return Object.freeze({ generation, dimensions, brickFineResolution: 8,
    brickCellCapacity: 64, bricks: Object.freeze(bricks),
    brickByKey: new Map(bricks.map(brick => [brick.key, brick])), cellBaseByBrick,
    cells: Object.freeze(cells), rows: Object.freeze(rows), incidenceOffsets,
    incidences: Object.freeze(incidences), subfaces: Object.freeze(subfaces),
    mixedSeamRowCount, sparseAirRowCount, boundaryModes: Object.freeze({ ...boundaryModes }) });
}

export function createSliceTopologyAuthority(accepted: SliceTopology): SliceTopologyAuthority {
  return Object.freeze({ accepted });
}

export function stageSliceTopologyCandidate(authority: SliceTopologyAuthority,
  bricks: readonly SliceTopologyBrick[]): SliceTopologyAuthority {
  if (authority.candidate) throw new Error("slice topology already has a candidate generation");
  return Object.freeze({ accepted: authority.accepted,
    candidate: compileSliceTopology(bricks, authority.accepted.dimensions,
      authority.accepted.generation + 1, 0.5, authority.accepted.boundaryModes) });
}

export function cancelSliceTopologyCandidate(authority: SliceTopologyAuthority): SliceTopologyAuthority {
  if (!authority.candidate) throw new Error("slice topology has no candidate generation");
  return Object.freeze({ accepted: authority.accepted });
}

export function commitSliceTopologyCandidate(authority: SliceTopologyAuthority): SliceTopologyAuthority {
  if (!authority.candidate) throw new Error("slice topology has no candidate generation");
  if (authority.candidate.generation !== authority.accepted.generation + 1) {
    throw new Error("slice topology candidate generation is stale or discontinuous");
  }
  return Object.freeze({ accepted: authority.candidate });
}

export interface SliceTopologyTransferPlan {
  readonly cellOffsets: Uint32Array;
  readonly cellSources: Uint32Array;
  readonly cellAreas: Float32Array;
  readonly faceOffsets: Uint32Array;
  readonly faceSources: Uint32Array;
  readonly faceAreas: Float32Array;
}

export interface SliceTopologyNewAirCoverage {
  readonly minimumFine: SliceTopologyVec2;
  readonly maximumExclusiveFine: SliceTopologyVec2;
}

export const SLICE_TOPOLOGY_NEW_AIR = 0xffff_ffff;

/** Sparse overlap plan used by the production-style candidate transfer stage. */
export function transferSliceTopology(source: SliceTopology,
  target: SliceTopology,
  newAirCoverage: readonly SliceTopologyNewAirCoverage[] = [],
): SliceTopologyTransferPlan {
  if (source.dimensions[0] !== target.dimensions[0]
    || source.dimensions[1] !== target.dimensions[1]) {
    throw new Error("slice topology transfer requires the same physical domain");
  }
  const cellOffsets = new Uint32Array(target.cells.length + 1);
  const cellSources: number[] = [], cellAreas: number[] = [];
  for (const next of target.cells) {
    cellOffsets[next.id] = cellSources.length;
    let covered = 0;
    for (const old of source.cells) {
      const area = overlap(next.minimumFine[0], next.maximumFine[0],
        old.minimumFine[0], old.maximumFine[0])
        * overlap(next.minimumFine[1], next.maximumFine[1],
          old.minimumFine[1], old.maximumFine[1]);
      if (area <= 0) continue;
      cellSources.push(old.id); cellAreas.push(f32(area)); covered += area;
    }
    if (covered > next.volumeFineCells + 1e-6) {
      throw new Error(`slice topology target cell ${next.id} has overlapping accepted coverage`);
    }
    if (covered < next.volumeFineCells - 1e-6) {
      const newAir = newAirCoverage.some(box => next.minimumFine.every((q, axis) =>
        q >= box.minimumFine[axis]!
        && next.maximumFine[axis]! <= box.maximumExclusiveFine[axis]!));
      if (!newAir) {
        throw new Error(`slice topology target cell ${next.id} lacks complete accepted coverage`);
      }
      cellSources.push(SLICE_TOPOLOGY_NEW_AIR);
      cellAreas.push(f32(next.volumeFineCells - covered));
    }
  }
  cellOffsets[target.cells.length] = cellSources.length;

  const faceOffsets = new Uint32Array(target.rows.length + 1);
  const faceSources: number[] = [], faceAreas: number[] = [];
  for (const next of target.rows) {
    faceOffsets[next.id] = faceSources.length;
    const tangent = (1 - next.axis) as SliceTopologyAxis;
    const n0 = next.centerFine[tangent] - 0.5 * next.areaFineCells;
    const n1 = next.centerFine[tangent] + 0.5 * next.areaFineCells;
    for (const old of source.rows) {
      if (old.axis !== next.axis || old.centerFine[next.axis] !== next.centerFine[next.axis]) continue;
      const o0 = old.centerFine[tangent] - 0.5 * old.areaFineCells;
      const o1 = old.centerFine[tangent] + 0.5 * old.areaFineCells;
      const area = overlap(n0, n1, o0, o1);
      if (area <= 0) continue;
      faceSources.push(old.id); faceAreas.push(f32(area));
    }
  }
  faceOffsets[target.rows.length] = faceSources.length;
  return Object.freeze({ cellOffsets, cellSources: Uint32Array.from(cellSources),
    cellAreas: Float32Array.from(cellAreas), faceOffsets,
    faceSources: Uint32Array.from(faceSources), faceAreas: Float32Array.from(faceAreas) });
}

/** Sparse state planes transferred during accepted -> candidate publication. */
export interface SliceTopologyTransferFields {
  /** Accepted liquid amount divided by full cell area (not by open capacity). */
  readonly density: Float32Array;
  readonly gamma: Float32Array;
  readonly pressure: Float32Array;
  /** Interleaved x/y collocated velocity. */
  readonly cellVelocity: Float32Array;
  readonly faceVelocity: Float32Array;
  /** Open fraction of full cell area. */
  readonly capacity: Float32Array;
  /** Accepted raw PLIC gradient cache. Only x/y are retained in the slice. */
  readonly interfaceNormal?: Float32Array;
}

export interface SliceTopologyTransferResult extends SliceTopologyTransferFields {
  readonly interfaceNormal: Float32Array;
  readonly interfaceOffset: Float32Array;
  readonly plan: SliceTopologyTransferPlan;
  readonly sourceAmounts: Float32Array;
  readonly sourceCapacities: Float32Array;
  readonly targetAmounts: Float32Array;
  readonly targetCapacities: Float32Array;
}

/** Candidate remap cannot represent the accepted amount in its capacity. */
export class SliceTopologyGenerationCapacityDeferred extends Error {
  constructor(readonly fault: number, readonly owner: number,
    readonly amount: number, readonly capacity: number) {
    super(`Slice generation remap deferred: fault=${fault}, owner=${owner}, amount=${amount}, capacity=${capacity}`);
    this.name = "SliceTopologyGenerationCapacityDeferred";
  }
}

const transferTolerance = (capacity: number): number =>
  f32(f32(9.5367431640625e-7) * f32(capacity));

function transferAdd(total: readonly [number, number], value: number): [number, number] {
  const sum = f32(f32(total[0]) + f32(value));
  const error = Math.abs(total[0]) >= Math.abs(value)
    ? f32(f32(f32(total[0]) - sum) + f32(value))
    : f32(f32(f32(value) - sum) + f32(total[0]));
  const tail = f32(f32(total[1]) + error);
  const result = f32(sum + tail);
  return [result, f32(f32(sum - result) + tail)];
}

function transferValid(amount: number, capacity: number): boolean {
  const tolerance = transferTolerance(capacity);
  return Number.isFinite(capacity) && capacity >= 0
    && Number.isFinite(amount) && amount >= -tolerance && amount <= capacity + tolerance;
}

function planeBoxFraction2(normal: SliceTopologyVec2, offset: number,
  widths: SliceTopologyVec2): number {
  const projected = [f32(Math.abs(normal[0]) * widths[0]),
    f32(Math.abs(normal[1]) * widths[1])] as const;
  const dominant = Math.max(projected[0], projected[1]);
  if (dominant <= 1e-20) return offset >= 0 ? 1 : 0;
  const spans: number[] = [];
  for (const value of projected) {
    const span = f32(value / dominant);
    if (span >= 1e-6) spans.push(span);
  }
  const total = f32((spans[0] ?? 0) + (spans[1] ?? 0));
  const shifted = f32(f32(offset / dominant) + f32(0.5 * total));
  if (shifted <= 0) return 0;
  if (shifted >= total) return 1;
  const complement = shifted > f32(0.5 * total);
  const x = complement ? f32(total - shifted) : shifted;
  let fraction: number;
  if (spans.length === 1) fraction = f32(x / spans[0]!);
  else {
    const a = Math.min(spans[0]!, spans[1]!);
    const b = Math.max(spans[0]!, spans[1]!);
    fraction = x < a
      ? f32(f32(0.5 * f32(x / a)) * f32(x / b))
      : f32(f32(x - f32(0.5 * a)) / b);
  }
  fraction = Math.max(0, Math.min(1, fraction));
  return complement ? f32(1 - fraction) : fraction;
}

function interfacePlane2(fill: number, gradient: SliceTopologyVec2,
  widths: SliceTopologyVec2): { normal: SliceTopologyVec2; offset: number; valid: boolean } {
  const maximum = Math.max(Math.abs(gradient[0]), Math.abs(gradient[1]));
  if (!(maximum > 1e-20)) return { normal: [0, 0], offset: 0, valid: false };
  const gx = f32(gradient[0] / maximum), gy = f32(gradient[1] / maximum);
  const length = f32(Math.sqrt(f32(f32(gx * gx) + f32(gy * gy))));
  const normal = [f32(gx / length), f32(gy / length)] as const;
  const projected = [f32(Math.abs(normal[0]) * widths[0]),
    f32(Math.abs(normal[1]) * widths[1])] as const;
  const dominant = Math.max(projected[0], projected[1]);
  const radius = f32(0.5 * f32(projected[0] + projected[1]));
  let offset = 0;
  if (fill <= 0) offset = -radius;
  else if (fill >= 1) offset = radius;
  else if (fill !== 0.5 && dominant > 1e-20) {
    const spans = projected.map(value => f32(value / dominant)).filter(value => value >= 1e-6);
    if (spans.length === 1) offset = f32(f32(fill - 0.5) * f32(spans[0]! * dominant));
    else {
      const complement = fill > 0.5;
      const target = complement ? f32(1 - fill) : f32(fill);
      const a = Math.min(spans[0]!, spans[1]!);
      const b = Math.max(spans[0]!, spans[1]!);
      let shifted = f32(f32(target * b) + f32(0.5 * a));
      if (target < f32(f32(0.5 * a) / b)) {
        shifted = f32(Math.sqrt(f32(f32(f32(2 * target) * a) * b)));
      }
      const lower = f32(f32(shifted - f32(0.5 * f32(a + b))) * dominant);
      offset = complement ? -lower : lower;
    }
  }
  return { normal, offset: f32(offset), valid: f32(normal[0] * normal[0] + normal[1] * normal[1]) > 0.5 };
}

function validateTransferFields(topology: SliceTopology, fields: SliceTopologyTransferFields,
  label: string): void {
  const cells = topology.cells.length, rows = topology.rows.length;
  for (const [name, values, count] of [
    ["density", fields.density, cells], ["gamma", fields.gamma, cells],
    ["pressure", fields.pressure, cells], ["capacity", fields.capacity, cells],
    ["cellVelocity", fields.cellVelocity, 2 * cells], ["faceVelocity", fields.faceVelocity, rows],
    ["interfaceNormal", fields.interfaceNormal, fields.interfaceNormal ? 2 * cells : 0],
  ] as const) {
    if (values && values.length !== count) {
      throw new RangeError(`${label} ${name} length ${values.length} != ${count}`);
    }
  }
}

/**
 * CPU rank-reduction of production's source-owned generation-transfer shader.
 * It intentionally preserves its f32 compensated sums, capacity fallback,
 * PLIC admission test, two residual sweeps, and dry face/cell fill behavior.
 */
export function transferSliceTopologyFields(
  source: SliceTopology,
  target: SliceTopology,
  sourceFields: SliceTopologyTransferFields,
  targetCapacity: Float32Array,
  newAirCoverage: readonly SliceTopologyNewAirCoverage[] = [],
): SliceTopologyTransferResult {
  validateTransferFields(source, sourceFields, "source");
  if (targetCapacity.length !== target.cells.length) {
    throw new RangeError("target capacity length does not match candidate cells");
  }
  const plan = transferSliceTopology(source, target, newAirCoverage);
  const entryTarget = new Uint32Array(plan.cellSources.length);
  const overlapBounds: { center: SliceTopologyVec2; widths: SliceTopologyVec2 }[] = [];
  const groups: number[][] = Array.from({ length: source.cells.length }, () => []);
  for (const next of target.cells) {
    for (let entry = plan.cellOffsets[next.id]!; entry < plan.cellOffsets[next.id + 1]!; entry++) {
      entryTarget[entry] = next.id;
      const beforeId = plan.cellSources[entry]!;
      if (beforeId === SLICE_TOPOLOGY_NEW_AIR) {
        overlapBounds[entry] = { center: [0, 0], widths: [0, 0] };
        continue;
      }
      const before = source.cells[beforeId]!;
      const minimum = [Math.max(next.minimumFine[0], before.minimumFine[0]),
        Math.max(next.minimumFine[1], before.minimumFine[1])] as const;
      const maximum = [Math.min(next.maximumFine[0], before.maximumFine[0]),
        Math.min(next.maximumFine[1], before.maximumFine[1])] as const;
      const widths = [f32(maximum[0] - minimum[0]), f32(maximum[1] - minimum[1])] as const;
      overlapBounds[entry] = { center: [f32(minimum[0] + f32(0.5 * widths[0])),
        f32(minimum[1] + f32(0.5 * widths[1]))], widths };
      groups[beforeId]!.push(entry);
    }
  }
  for (const group of groups) group.sort((a, b) => {
    const ac = target.cells[entryTarget[a]!]!, bc = target.cells[entryTarget[b]!]!;
    return ac.minimumFine[0] - bc.minimumFine[0]
      || ac.minimumFine[1] - bc.minimumFine[1] || a - b;
  });

  const contributions = new Float32Array(plan.cellSources.length);
  const sourceAmounts = new Float32Array(source.cells.length);
  const sourceCapacities = new Float32Array(source.cells.length);
  for (const before of source.cells) {
    const amount = f32(sourceFields.density[before.id]! * before.volumeFineCells);
    const capacity = f32(sourceFields.capacity[before.id]! * before.volumeFineCells);
    sourceAmounts[before.id] = amount; sourceCapacities[before.id] = capacity;
    if (!transferValid(amount, capacity)) {
      throw new SliceTopologyGenerationCapacityDeferred(1, before.id, amount, capacity);
    }
    const group = groups[before.id]!;
    let covered: [number, number] = [0, 0], capacities: [number, number] = [0, 0];
    for (const entry of group) {
      covered = transferAdd(covered, plan.cellAreas[entry]!);
      const targetId = entryTarget[entry]!;
      const childCapacity = f32(targetCapacity[targetId]! * plan.cellAreas[entry]!);
      if (!(childCapacity >= 0 && Number.isFinite(childCapacity))) {
        throw new SliceTopologyGenerationCapacityDeferred(64, before.id, amount, childCapacity);
      }
      capacities = transferAdd(capacities, childCapacity);
    }
    const coverage = f32(covered[0] + covered[1]);
    if (Math.abs(f32(coverage - before.volumeFineCells)) > transferTolerance(before.volumeFineCells)
      && amount !== 0) {
      throw new SliceTopologyGenerationCapacityDeferred(128, before.id, amount, coverage);
    }
    if (group.length === 0) continue;
    if (group.length === 1) { contributions[group[0]!] = amount; continue; }
    const available = f32(capacities[0] + capacities[1]);
    if (!(Math.abs(amount) <= f32(available + transferTolerance(available)))) {
      throw new SliceTopologyGenerationCapacityDeferred(64, before.id, amount, available);
    }
    const gradient = [sourceFields.interfaceNormal?.[2 * before.id] ?? 0,
      sourceFields.interfaceNormal?.[2 * before.id + 1] ?? 0] as const;
    const fill = Math.max(0, Math.min(1, f32(amount / before.volumeFineCells)));
    const plane = interfacePlane2(fill, gradient, before.widthsFine);
    const usePlane = capacity === before.volumeFineCells && plane.valid && amount >= 0;
    let remaining: [number, number] = [amount, 0];
    for (const entry of group) {
      const targetId = entryTarget[entry]!;
      const childCapacity = f32(targetCapacity[targetId]! * plan.cellAreas[entry]!);
      let proposed = available > 0 ? f32(amount * f32(childCapacity / available)) : 0;
      if (usePlane && childCapacity === plan.cellAreas[entry]!) {
        const box = overlapBounds[entry]!;
        const relative = [f32(box.center[0] - before.centerFine[0]),
          f32(box.center[1] - before.centerFine[1])] as const;
        proposed = f32(plan.cellAreas[entry]! * planeBoxFraction2(plane.normal,
          f32(plane.offset - f32(f32(plane.normal[0] * relative[0])
            + f32(plane.normal[1] * relative[1]))), box.widths));
      }
      if (amount >= 0) {
        const excess = Math.max(0, f32(amount - available));
        if (excess > 0 && available > 0) {
          proposed = f32(childCapacity + f32(excess * f32(childCapacity / available)));
        }
        proposed = Math.max(0, Math.min(f32(childCapacity + transferTolerance(childCapacity)), proposed));
      }
      contributions[entry] = f32(proposed);
      remaining = transferAdd(remaining, -f32(proposed));
    }
    for (let sweep = 0; sweep < 2; sweep++) for (const entry of group) {
      const targetId = entryTarget[entry]!;
      const childCapacity = f32(targetCapacity[targetId]! * plan.cellAreas[entry]!);
      const previous = contributions[entry]!;
      const residual = f32(remaining[0] + remaining[1]);
      const lower = amount < 0 ? -transferTolerance(childCapacity) : 0;
      const upper = amount < 0 ? 0 : f32(childCapacity + transferTolerance(childCapacity));
      const proposed = Math.max(lower, Math.min(upper, f32(previous + residual)));
      contributions[entry] = f32(proposed);
      remaining = transferAdd(remaining, previous);
      remaining = transferAdd(remaining, -f32(proposed));
    }
    const remainder = f32(remaining[0] + remaining[1]);
    if (Math.abs(remainder) > transferTolerance(Math.abs(amount))) {
      throw new SliceTopologyGenerationCapacityDeferred(256, before.id, remainder, Math.abs(amount));
    }
  }

  const density = new Float32Array(target.cells.length);
  const gamma = new Float32Array(target.cells.length);
  const pressure = new Float32Array(target.cells.length);
  const cellVelocity = new Float32Array(2 * target.cells.length);
  const targetAmounts = new Float32Array(target.cells.length);
  const targetCapacities = new Float32Array(target.cells.length);
  for (const next of target.cells) {
    let mass: [number, number] = [0, 0], observed: [number, number] = [0, 0];
    let g = 0, p = 0, momentumX = 0, momentumY = 0, dryX = 0, dryY = 0;
    for (let entry = plan.cellOffsets[next.id]!; entry < plan.cellOffsets[next.id + 1]!; entry++) {
      const beforeId = plan.cellSources[entry]!, area = plan.cellAreas[entry]!;
      if (beforeId === SLICE_TOPOLOGY_NEW_AIR) { g = f32(g + area); continue; }
      const contribution = contributions[entry]!;
      mass = transferAdd(mass, contribution);
      g = f32(g + f32(sourceFields.gamma[beforeId]! * area));
      p = f32(p + f32(sourceFields.pressure[beforeId]! * area));
      const weight = Math.max(0, contribution);
      observed = transferAdd(observed, weight);
      momentumX = f32(momentumX + f32(weight * sourceFields.cellVelocity[2 * beforeId]!));
      momentumY = f32(momentumY + f32(weight * sourceFields.cellVelocity[2 * beforeId + 1]!));
      dryX = f32(dryX + f32(area * sourceFields.cellVelocity[2 * beforeId]!));
      dryY = f32(dryY + f32(area * sourceFields.cellVelocity[2 * beforeId + 1]!));
    }
    const area = next.volumeFineCells;
    const amount = f32(mass[0] + mass[1]);
    density[next.id] = f32(amount / area);
    gamma[next.id] = f32(g / area);
    pressure[next.id] = f32(p / area);
    const capacity = f32(targetCapacity[next.id]! * area);
    targetAmounts[next.id] = amount; targetCapacities[next.id] = capacity;
    if (!transferValid(amount, capacity)) {
      throw new SliceTopologyGenerationCapacityDeferred(64, next.id, amount, capacity);
    }
    const liquidWeight = f32(observed[0] + observed[1]);
    cellVelocity[2 * next.id] = liquidWeight > 0
      ? f32(momentumX / liquidWeight) : f32(dryX / area);
    cellVelocity[2 * next.id + 1] = liquidWeight > 0
      ? f32(momentumY / liquidWeight) : f32(dryY / area);
  }

  const faceVelocity = new Float32Array(target.rows.length);
  for (const row of target.rows) {
    let flux = 0, covered = 0;
    for (let entry = plan.faceOffsets[row.id]!; entry < plan.faceOffsets[row.id + 1]!; entry++) {
      const area = plan.faceAreas[entry]!, before = plan.faceSources[entry]!;
      flux = f32(flux + f32(area * sourceFields.faceVelocity[before]!));
      covered = f32(covered + area);
    }
    let velocity = 0, weight = 0;
    for (const term of row.terms) {
      const w = Math.abs(term.coefficient);
      velocity = f32(velocity + f32(w * cellVelocity[2 * term.cellId + row.axis]!));
      weight = f32(weight + w);
    }
    const fill = f32(Math.max(0, f32(row.areaFineCells - covered))
      * f32(velocity / Math.max(weight, 1e-20)));
    faceVelocity[row.id] = f32(f32(flux + fill) / row.areaFineCells);
  }
  return Object.freeze({ density, gamma, pressure, cellVelocity, faceVelocity,
    capacity: Float32Array.from(targetCapacity), interfaceNormal: new Float32Array(2 * target.cells.length),
    interfaceOffset: new Float32Array(target.cells.length), plan,
    sourceAmounts, sourceCapacities, targetAmounts, targetCapacities });
}

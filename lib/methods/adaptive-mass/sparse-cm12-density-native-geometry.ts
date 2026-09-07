import type { SparseAtlasCompositeCell, SparseAtlasGradientRow } from "./sparse-atlas-composite-projection";

export type DensityNativeVec3 = readonly [number, number, number];
/** Integrals in physical coordinates, about the native cell centre. These are
 * supplied by the boundary authority; scalar solid fractions alone do not
 * determine first or second open-domain moments. */
export interface DensityNativeOpenMoments {
  readonly boundaryGeneration: number;
  readonly volume: number;
  readonly first: DensityNativeVec3;
  /** xx, yy, zz, xy, xz, yz. */
  readonly second: readonly [number, number, number, number, number, number];
}
export interface DensityNativeCell {
  readonly id: number;
  readonly stableLeafId: number;
  readonly lower: DensityNativeVec3;
  readonly upper: DensityNativeVec3;
  readonly centre: DensityNativeVec3;
  readonly volume: number;
  /** Absent means unknown, never implicitly fully open. */
  readonly openMoments?: DensityNativeOpenMoments;
}
export interface DensityNativeGeometry {
  readonly topologyGeneration: number;
  readonly boundaryGeneration: number;
  readonly cells: readonly DensityNativeCell[];
  /** Native IDs are not necessarily dense. CSR addresses compact ordinals. */
  readonly ordinalById: ReadonlyMap<number, number>;
  readonly offsets: Uint32Array;
  readonly neighborOrdinals: Uint32Array;
  readonly receipt: Readonly<{
    nativeCells: number; directedNeighbors: number; candidatePairs: number;
    rejectedNonFacePairs: number; adjacencyBytes: number; knownOpenMomentCells: number;
  }>;
}
const vector = (values: number[]): DensityNativeVec3 => Object.freeze(values) as unknown as DensityNativeVec3;
function generation(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error("invalid native geometry generation");
}
/** Compile only once per accepted topology/boundary geometry. No finest-grid
 * expansion, donor search or field reconstruction is performed here. Rows
 * nominate candidates; native box geometry independently certifies each edge.
 */
export function compileDensityNativeGeometry(options: Readonly<{
  cells: readonly SparseAtlasCompositeCell[];
  rows: readonly SparseAtlasGradientRow[];
  topologyGeneration: number;
  boundaryGeneration: number;
  fineCellWidth: number;
  origin?: DensityNativeVec3;
  openMomentsByCellId?: ReadonlyMap<number, DensityNativeOpenMoments>;
}>): DensityNativeGeometry {
  generation(options.topologyGeneration); generation(options.boundaryGeneration);
  const h = options.fineCellWidth, origin = options.origin ?? [0, 0, 0];
  if (!(h > 0) || !Number.isFinite(h) || !origin.every(Number.isFinite)) throw new Error("invalid physical coordinate transform");
  const ordinalById = new Map<number, number>();
  let knownOpenMomentCells = 0;
  const cells = options.cells.map((cell, ordinal): DensityNativeCell => {
    if (!Number.isInteger(cell.id) || cell.id < 0 || cell.id > 0xffff_fffe || ordinalById.has(cell.id)) throw new Error("invalid or duplicate native cell id");
    ordinalById.set(cell.id, ordinal);
    if (cell.minimumFine.some((lo, a) => !Number.isFinite(lo) || !Number.isFinite(cell.maximumFine[a]) || cell.maximumFine[a]! <= lo)) throw new Error("invalid native cell bounds");
    const lower = vector(cell.minimumFine.map((v, a) => origin[a]! + h * v));
    const upper = vector(cell.maximumFine.map((v, a) => origin[a]! + h * v));
    const centre = vector(lower.map((lo, a) => (lo + upper[a]!) / 2));
    const volume = lower.reduce((v, lo, a) => v * (upper[a]! - lo), 1);
    if (!(volume > 0) || !Number.isFinite(volume)) throw new Error("invalid physical native volume");
    const source = options.openMomentsByCellId?.get(cell.id);
    let openMoments: DensityNativeOpenMoments | undefined;
    if (source) {
      if (source.boundaryGeneration !== options.boundaryGeneration || !Number.isFinite(source.volume) || source.volume < 0 || source.volume > volume * (1 + 1e-12)
        || source.first.length !== 3 || source.second.length !== 6 || !source.first.every(Number.isFinite) || !source.second.every(Number.isFinite)) throw new Error("invalid or stale open-domain moments");
      openMoments = Object.freeze({ ...source, first: vector([...source.first]), second: Object.freeze([...source.second]) as unknown as DensityNativeOpenMoments["second"] });
      knownOpenMomentCells++;
    }
    return Object.freeze({ id: cell.id, stableLeafId: cell.stableLeafId, lower, upper, centre, volume, ...(openMoments ? { openMoments } : {}) });
  });
  const neighbors = cells.map(() => new Set<number>());
  let candidatePairs = 0, rejectedNonFacePairs = 0;
  for (const row of options.rows) {
    if (![0, 1, 2].includes(row.axis)) throw new Error("invalid native row axis");
    const negative: number[] = [], positive: number[] = [];
    for (const term of row.terms) {
      if (!Number.isFinite(term.coefficient)) throw new Error("invalid native row coefficient");
      const ordinal = ordinalById.get(term.cellId);
      if (ordinal === undefined) throw new Error("native row references absent cell");
      if (term.coefficient < 0) negative.push(ordinal);
      else if (term.coefficient > 0) positive.push(ordinal);
    }
    for (const a of negative) for (const b of positive) {
      candidatePairs++;
      const x = options.cells[a]!, y = options.cells[b]!, axis = row.axis;
      // Use integer/fine coordinate geometry, avoiding physical-unit roundoff.
      const touches = x.maximumFine[axis] === y.minimumFine[axis] || y.maximumFine[axis] === x.minimumFine[axis];
      const overlaps = [0, 1, 2].every(k => k === axis || Math.min(x.maximumFine[k]!, y.maximumFine[k]!) > Math.max(x.minimumFine[k]!, y.minimumFine[k]!));
      if (a === b || !touches || !overlaps) { rejectedNonFacePairs++; continue; }
      neighbors[a]!.add(b); neighbors[b]!.add(a);
    }
  }
  const offsets = new Uint32Array(cells.length + 1);
  for (let i = 0; i < cells.length; i++) {
    const end = offsets[i]! + neighbors[i]!.size;
    if (end > 0xffff_ffff) throw new Error("native adjacency exceeds u32 address space");
    offsets[i + 1] = end;
  }
  const neighborOrdinals = new Uint32Array(offsets[cells.length]!);
  neighbors.forEach((set, i) => neighborOrdinals.set([...set].sort((a, b) => cells[a]!.id - cells[b]!.id), offsets[i]!));
  return Object.freeze({ topologyGeneration: options.topologyGeneration, boundaryGeneration: options.boundaryGeneration,
    cells: Object.freeze(cells), ordinalById, offsets, neighborOrdinals,
    receipt: Object.freeze({ nativeCells: cells.length, directedNeighbors: neighborOrdinals.length, candidatePairs, rejectedNonFacePairs,
      adjacencyBytes: offsets.byteLength + neighborOrdinals.byteLength, knownOpenMomentCells }) });
}
/** Explicit bounded support expansion. A face graph requires multiple rings to
 * reach corner donors. This is a topology query, not a rank certificate; fitting
 * must still reject insufficient support. Overflow fails instead of truncating.
 */
export function densityNativeSupportIds(geometry: DensityNativeGeometry, options: Readonly<{
  homeId: number; topologyGeneration: number; boundaryGeneration: number;
  rings: number; maximumCells: number;
}>): Uint32Array {
  if (geometry.topologyGeneration !== options.topologyGeneration || geometry.boundaryGeneration !== options.boundaryGeneration) throw new Error("stale native density geometry");
  if (!Number.isSafeInteger(options.rings) || options.rings < 0 || !Number.isSafeInteger(options.maximumCells) || options.maximumCells < 1) throw new Error("invalid native support budget");
  const home = geometry.ordinalById.get(options.homeId);
  if (home === undefined) throw new Error("native support home is absent");
  const visited = new Set([home]), queue = [home];
  let begin = 0;
  for (let ring = 0; ring < options.rings && begin < queue.length; ring++) {
    const end = queue.length;
    for (; begin < end; begin++) {
      const ordinal = queue[begin]!;
      for (let at = geometry.offsets[ordinal]!; at < geometry.offsets[ordinal + 1]!; at++) {
        const neighbor = geometry.neighborOrdinals[at]!;
        if (visited.has(neighbor)) continue;
        if (visited.size === options.maximumCells) throw new Error("native density support budget exceeded");
        visited.add(neighbor); queue.push(neighbor);
      }
    }
  }
  return Uint32Array.from(queue, ordinal => geometry.cells[ordinal]!.id);
}

/** Immutable staggered field used by the next momentum gather after remeshing.
 * Storage is independent of the mutable CNX/TEI images and their stable ids. */
export function momentumSnapshotLayout(cells: number, rows: number, incidences: number) {
  if (![cells, rows, incidences].every(n => Number.isSafeInteger(n) && n >= 0)) {
    throw new RangeError("Invalid momentum snapshot capacity");
  }
  const hashCapacity = 2 ** Math.ceil(Math.log2(Math.max(2, 2 * cells)));
  const cellBase = 16;
  const rowBase = cellBase + 8 * cells;
  const incidenceBase = rowBase + 8 * rows;
  const hashBase = incidenceBase + incidences;
  return { cells, rows, incidences, hashCapacity, cellBase, rowBase, incidenceBase, hashBase,
    byteLength: 4 * (hashBase + hashCapacity) };
}
export const MOMENTUM_SNAPSHOT_ENTRY_POINTS = [
  "momentumSnapshotBegin", "momentumSnapshotCells", "momentumSnapshotFaces", "momentumSnapshotSeal",
] as const;

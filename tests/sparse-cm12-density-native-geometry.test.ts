import assert from "node:assert/strict";
import test from "node:test";
import { compileDensityNativeGeometry, densityNativeSupportIds, type DensityNativeVec3 } from "../lib/methods/adaptive-mass/sparse-cm12-density-native-geometry";
import type { SparseAtlasCompositeCell, SparseAtlasGradientRow } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";

function cell(id: number, lo: DensityNativeVec3, hi: DensityNativeVec3): SparseAtlasCompositeCell {
  const widths = hi.map((v, a) => v - lo[a]!) as unknown as DensityNativeVec3;
  const volume = widths.reduce((a, b) => a * b, 1);
  return { id, stableLeafId: id, brickKey: id, brickCoordinate: [0, 0, 0], brickResolution: 1,
    local: [0, 0, 0], localIndex: 0, minimumFine: lo, maximumFine: hi,
    centerFine: lo.map((v, a) => (v + hi[a]!) / 2) as unknown as DensityNativeVec3,
    widthsFine: widths, volume, volumeFineCells: volume, density: 0, gamma: 0 };
}
function row(id: number, axis: 0 | 1 | 2, minus: number[], plus: number[]): SparseAtlasGradientRow {
  return { id, axis, kind: "mixed-seam", centerFine: [0, 0, 0], area: 1, distance: 1,
    areaFineCells2: 1, centerDistanceFine: 1, dualWeight: 1,
    terms: [...minus.map(cellId => ({ cellId, coefficient: -1 })), ...plus.map(cellId => ({ cellId, coefficient: 1 }))] };
}
const options = { topologyGeneration: 4, boundaryGeneration: 7, fineCellWidth: 0.5 };
test("native macro/fine adjacency certifies faces and keeps compact IDs without volume expansion", () => {
  const cells = [cell(901, [0, 0, 0], [1048576, 1048576, 1048576]),
    cell(37, [1048576, 0, 0], [1048577, 1, 1]),
    cell(1009, [1048576, 1, 0], [1048577, 2, 1]),
    cell(88, [1048576, 1048576, 0], [1048577, 1048577, 1])];
  const geometry = compileDensityNativeGeometry({ ...options, cells,
    rows: [row(0, 0, [901], [37, 1009, 88]), row(1, 1, [37], [1009]), row(2, 0, [901], [37])] });
  assert.equal(geometry.receipt.nativeCells, 4);
  assert.equal(geometry.receipt.directedNeighbors, 6);
  assert.equal(geometry.receipt.rejectedNonFacePairs, 1);
  assert.equal(geometry.receipt.adjacencyBytes, 44);
  assert.deepEqual([...densityNativeSupportIds(geometry, { ...options, homeId: 901, rings: 1, maximumCells: 4 })], [901, 37, 1009]);
  assert.equal(geometry.cells[0]!.volume, (1048576 * 0.5) ** 3);
  assert.equal(geometry.cells[0]!.openMoments, undefined);
});
test("multiple graph rings expose corner support, generation and capacity fail closed", () => {
  const cells = [cell(10, [0, 0, 0], [1, 1, 1]), cell(20, [1, 0, 0], [2, 1, 1]), cell(30, [1, 1, 0], [2, 2, 1])];
  const geometry = compileDensityNativeGeometry({ ...options, cells, rows: [row(0, 0, [10], [20]), row(1, 1, [20], [30])] });
  const query = { ...options, homeId: 10, rings: 2, maximumCells: 3 };
  assert.deepEqual([...densityNativeSupportIds(geometry, query)], [10, 20, 30]);
  assert.deepEqual([...densityNativeSupportIds(geometry, { ...query, rings: 0 })], [10]);
  assert.throws(() => densityNativeSupportIds(geometry, { ...query, maximumCells: 2 }), /budget exceeded/);
  assert.throws(() => densityNativeSupportIds(geometry, { ...query, topologyGeneration: 3 }), /stale/);
  assert.throws(() => densityNativeSupportIds(geometry, { ...query, boundaryGeneration: 6 }), /stale/);
});
test("physical clipped bounds and supplied open moments remain separate and detached from input", () => {
  const first: [number, number, number] = [0, 0.001, 0];
  const moments = { boundaryGeneration: 7, volume: 0.01, first, second: [0, 0, 0, 0, 0, 0] as const };
  const args = { ...options, origin: [-3, 2, 7] as const, cells: [cell(45, [0, 0, 0], [1, 0.5, 1])], rows: [], openMomentsByCellId: new Map([[45, moments]]) };
  const geometry = compileDensityNativeGeometry(args);
  assert.deepEqual(geometry.cells[0]!.lower, [-3, 2, 7]);
  assert.deepEqual(geometry.cells[0]!.upper, [-2.5, 2.25, 7.5]);
  assert.equal(geometry.cells[0]!.volume, 0.0625);
  assert.equal(geometry.cells[0]!.openMoments!.volume, 0.01);
  first[1] = 12;
  assert.equal(geometry.cells[0]!.openMoments!.first[1], 0.001);
  assert.throws(() => compileDensityNativeGeometry({ ...args, boundaryGeneration: 8 }), /stale/);
  assert.throws(() => compileDensityNativeGeometry({ ...args, openMomentsByCellId: new Map([[45, { ...moments, volume: 2 }]]) }), /open-domain/);
});
test("invalid native IDs, boxes and incidence cannot enter the compiled graph", () => {
  const a = cell(1, [0, 0, 0], [1, 1, 1]);
  assert.throws(() => compileDensityNativeGeometry({ ...options, cells: [a, a], rows: [] }), /duplicate/);
  assert.throws(() => compileDensityNativeGeometry({ ...options, cells: [a], rows: [row(0, 0, [1], [2])] }), /absent/);
  assert.throws(() => compileDensityNativeGeometry({ ...options, cells: [cell(1, [0, 0, 0], [0, 1, 1])], rows: [] }), /bounds/);
});

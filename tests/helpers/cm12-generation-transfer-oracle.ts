import assert from "node:assert/strict";
import { buildSparseAtlasCompositeGrid, type SparseAtlasCompositeGrid } from "../../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { createSparseAdaptiveMassAtlas, sparseAtlasBrickKey, type SparseBrickResolution } from "../../lib/methods/adaptive-mass/sparse-brick-atlas";
import { packSparseCM12AcceptedTopologyTemplatesForQA } from "../../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import type { CM12CapturedGeometryRecipe } from "../../lib/methods/adaptive-mass/sparse-cm12-captured-geometry";
import type { SparseCM12NewAirCoverage } from "../../lib/methods/adaptive-mass/sparse-cm12-generation-transfer";

type Point = readonly [number, number, number];
export function transferGrid(dimensions: Point, descriptions: readonly {
  q: Point; r: SparseBrickResolution; span?: number; unclipped?: boolean;
}[]) {
  const brickDimensions = dimensions.map(value => Math.ceil(value / 8)) as [number, number, number];
  return buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas(dimensions,
    descriptions.map(({ q, r, span = 1, unclipped }) => ({
      key: sparseAtlasBrickKey(q, { brickDimensions, signedCoordinates: true }),
      coordinate: q, resolution: r, spanBricks: span, unclipped,
      density: new Float64Array(r ** 3).fill(.5), gamma: new Float64Array(r ** 3).fill(1),
    })), 0, 8, true));
}

export function transferFixtures() {
  const identity = transferGrid([16, 8, 8], [{ q: [0, 0, 0], r: 4 }, { q: [1, 0, 0], r: 2 }]);
  const macro = transferGrid([13, 15, 11], [{ q: [0, 0, 0], r: 2, span: 2 }]);
  const children = transferGrid([13, 15, 11], Array.from({ length: 8 }, (_, i) => ({
    q: [i & 1, (i >>> 1) & 1, i >>> 2] as Point, r: 4 as const,
  })));
  const seed = transferGrid([8, 8, 8], [{ q: [0, 0, 0], r: 2 }]);
  const grown = transferGrid([8, 8, 8], [-1, 0, 1].map(x => ({ q: [x, 0, 0] as Point, r: 2 as const, unclipped: x !== 0 })));
  const dynamic = transferGrid([8, 8, 8], [{ q: [-1, 0, 0], r: 8, unclipped: true }]);
  return [
    { name: "identity mixed widths", source: identity, target: identity },
    { name: "clipped macro split", source: macro, target: children },
    { name: "clipped macro merge", source: children, target: macro },
    { name: "signed new air", source: seed, target: grown, air: [
      { minimumFine: [-8, 0, 0], maximumExclusiveFine: [0, 8, 8] },
      { minimumFine: [8, 0, 0], maximumExclusiveFine: [16, 8, 8] },
    ] },
    { name: "signed dynamic source page", source: dynamic, target: dynamic, dynamicPage: 3 },
  ] satisfies { name: string; source: SparseAtlasCompositeGrid; target: SparseAtlasCompositeGrid;
    air?: SparseCM12NewAirCoverage[]; dynamicPage?: number }[];
}

export interface TransferLayout {
  densityOffset: number; densityOtherOffset: number; gammaOffset: number; gammaOtherOffset: number;
  velocityOffset: number; velocityOtherOffset: number; pressureOffset: number;
  faceOffset: number; faceOtherOffset: number; length: number;
}
export function transferLayout(cells: number, rows: number): TransferLayout {
  const base = 16;
  return { densityOffset: base, densityOtherOffset: base + cells,
    gammaOffset: base + 2 * cells, gammaOtherOffset: base + 3 * cells,
    velocityOffset: base + 4 * cells, velocityOtherOffset: base + 8 * cells,
    pressureOffset: base + 12 * cells, faceOffset: base + 13 * cells,
    faceOtherOffset: base + 13 * cells + rows, length: base + 13 * cells + 2 * rows + 16 };
}

/** Match the production row reorder by geometry only, without borrowing the
 * transfer compiler's overlap plan or evaluating its shader on the CPU. */
export function packedTransferGrid(grid: SparseAtlasCompositeGrid) {
  const packed = packSparseCM12AcceptedTopologyTemplatesForQA(grid.atlas, grid);
  const words = packed.words, floats = new Float32Array(words.buffer);
  const rowBase = words[7]!, rowCount = words[3]!;
  const key = (axis: number, center: readonly number[], area: number) => `${axis}/${center.join("/")}/${area}`;
  const rows = new Map<string, number>();
  for (let id = 0; id < rowCount; id++) rows.set(key(words[rowBase + rowCount + id]! >>> 30,
    [6, 7, 8].map(plane => floats[rowBase + plane * rowCount + id]!), floats[rowBase + 3 * rowCount + id]!), id);
  const physicalRows = Uint32Array.from(grid.gradientRows, row => {
    const id = rows.get(key(row.axis, row.centerFine, row.areaFineCells2));
    assert.notEqual(id, undefined, "packed row geometry must identify an actual CPU face");
    return id!;
  });
  const topology = new Uint32Array(words.length + 4);
  topology.set(words);
  return { topology, physicalRows, scalarParityWord: words.length, faceParityWord: words.length + 1 };
}

export function transferSourceRecipe(grid: SparseAtlasCompositeGrid, dynamicPage?: number) {
  const packed = packedTransferGrid(grid), cellPadding = 5;
  const physicalCells = Uint32Array.from(grid.cells, cell => cellPadding + cell.id);
  let physicalRows = packed.physicalRows;
  const sourcePageCoordinates = new Map<number, Point>(), dynamicKeys = new Set<number>();
  if (dynamicPage !== undefined) {
    assert.equal(grid.atlas.bricks.length, 1);
    const brick = grid.atlas.bricks[0]!; assert.equal(brick.resolution, 8);
    sourcePageCoordinates.set(dynamicPage, brick.coordinate);
    dynamicKeys.add(brick.key);
    packed.topology[3] = 0; // This fixture's accepted faces live entirely in a dynamic page.
    physicalRows = Uint32Array.from(grid.gradientRows, row => {
      const axis = row.axis, a = (axis + 1) % 3, b = (axis + 2) % 3;
      const normal = row.centerFine[axis]! - 8 * brick.coordinate[axis]!;
      const u = row.centerFine[a]! - 8 * brick.coordinate[a]! - .5;
      const v = row.centerFine[b]! - 8 * brick.coordinate[b]! - .5;
      assert.ok([normal, u, v].every(Number.isInteger));
      return dynamicPage * 1728 + axis * 576 + normal + 9 * (u + 8 * v);
    });
  }
  const recipe: CM12CapturedGeometryRecipe = { atlas: grid.atlas,
    active: new Set(grid.atlas.bricks.map(brick => brick.key)),
    sourceFirst: new Map(grid.atlas.bricks.map(brick => [brick.key, cellPadding + grid.cellBaseByBrick.get(brick.key)!])),
    sourcePageCoordinates, dynamicKeys, rows: physicalRows.slice().reverse(), templateWords: packed.topology };
  return { ...packed, recipe, physicalCells, physicalRows,
    layout: transferLayout(cellPadding + grid.cells.length, Math.max(...physicalRows) + 1) };
}

export function transferSourceValues(grid: SparseAtlasCompositeGrid, layout: TransferLayout,
  cellIds: Uint32Array, rowIds: Uint32Array) {
  const values = new Float32Array(layout.length).fill(-77);
  for (let bank = 0; bank < 2; bank++) {
    const density = bank ? layout.densityOtherOffset : layout.densityOffset;
    const gamma = bank ? layout.gammaOtherOffset : layout.gammaOffset;
    const velocity = bank ? layout.velocityOtherOffset : layout.velocityOffset;
    const face = bank ? layout.faceOtherOffset : layout.faceOffset;
    for (const cell of grid.cells) {
      const id = cellIds[cell.id]!;
      values[density + id] = cell.id % 7 === 0 ? 0 : .125 + .125 * bank + (cell.id % 5) / 32;
      values[gamma + id] = .75 + bank / 4 + (cell.id % 3) / 16;
      values[layout.pressureOffset + id] = (cell.id % 9 - 4) / 8;
      for (let axis = 0; axis < 3; axis++) values[velocity + 4 * id + axis] = bank - axis / 2 + (cell.id % 11) / 32;
    }
    for (const row of grid.gradientRows) values[face + rowIds[row.id]!] = (row.id % 13 - 6) / 16 + bank + row.axis / 4;
  }
  return values;
}

function intersection(a: readonly number[], aw: readonly number[], b: readonly number[], bw: readonly number[]) {
  return a.reduce((volume, low, axis) => volume * Math.max(0,
    Math.min(low + aw[axis]!, b[axis]! + bw[axis]!) - Math.max(low, b[axis]!)), 1);
}
function faceBounds(grid: SparseAtlasCompositeGrid) {
  return grid.gradientRows.map(row => {
    const axes = [0, 1, 2].filter(axis => axis !== row.axis);
    for (let span = 1; span <= 128; span *= 2) {
      const lower = axes.map(axis => Math.floor(row.centerFine[axis]! / span) * span);
      const width = axes.map((axis, i) => 2 * (row.centerFine[axis]! - lower[i]!));
      if (width.every(value => value > 0 && value <= span) && Math.abs(width[0]! * width[1]! - row.areaFineCells2) < 1e-8)
        return { lower, width };
    }
    throw new Error("oracle fixture face is not a clipped dyadic rectangle");
  });
}

/** Brute-force physical box intersections: no sparse lookup, dyadic walk,
 * production transfer plan, or GPU-generated contribution list is reused. */
export function referenceGenerationTransfer(source: SparseAtlasCompositeGrid, target: SparseAtlasCompositeGrid,
  values: Float32Array, layout: TransferLayout, cellIds: Uint32Array, rowIds: Uint32Array,
  scalarParity: number, faceParity: number, air: readonly SparseCM12NewAirCoverage[] = []) {
  const density = scalarParity ? layout.densityOtherOffset : layout.densityOffset;
  const gamma = scalarParity ? layout.gammaOtherOffset : layout.gammaOffset;
  const velocity = scalarParity ? layout.velocityOtherOffset : layout.velocityOffset;
  const face = faceParity ? layout.faceOtherOffset : layout.faceOffset;
  const cells = target.cells.map(cell => {
    let covered = 0, mass = 0, g = 0, p = 0;
    const momentum = [0, 0, 0], dryVelocity = [0, 0, 0];
    for (const before of source.cells) {
      const weight = intersection(before.minimumFine, before.widthsFine, cell.minimumFine, cell.widthsFine);
      if (weight === 0) continue;
      const id = cellIds[before.id]!, rho = values[density + id]!;
      covered += weight; mass += weight * rho; g += weight * values[gamma + id]!;
      p += weight * values[layout.pressureOffset + id]!;
      for (let axis = 0; axis < 3; axis++) {
        momentum[axis]! += weight * rho * values[velocity + 4 * id + axis]!;
        dryVelocity[axis]! += weight * values[velocity + 4 * id + axis]!;
      }
    }
    assert.ok(covered <= cell.volume + 1e-9);
    if (covered < cell.volume - 1e-9) assert.ok(air.some(box => cell.minimumFine.every((low, axis) =>
      low >= box.minimumFine[axis]! && cell.maximumFine[axis]! <= box.maximumExclusiveFine[axis]!)), "missing explicit new-air coverage");
    g += cell.volume - covered;
    return { density: mass / cell.volume, gamma: g / cell.volume, pressure: p / cell.volume,
      velocity: momentum.map((value, axis) => mass > 0 ? value / mass : dryVelocity[axis]! / cell.volume) };
  });
  const oldBounds = faceBounds(source), nextBounds = faceBounds(target);
  const faces = target.gradientRows.map(row => {
    let covered = 0, flux = 0;
    for (const before of source.gradientRows) {
      if (before.axis !== row.axis || before.centerFine[row.axis] !== row.centerFine[row.axis]) continue;
      const a = oldBounds[before.id]!, b = nextBounds[row.id]!;
      const weight = intersection(a.lower, a.width, b.lower, b.width);
      if (weight === 0) continue;
      covered += weight; flux += weight * values[face + rowIds[before.id]!]!;
    }
    assert.ok(covered <= row.areaFineCells2 + 1e-9, "source flux authority overlaps");
    const weights = row.terms.reduce((sum, term) => sum + Math.abs(term.coefficient), 0);
    const fallback = row.terms.reduce((sum, term) => sum + Math.abs(term.coefficient) * cells[term.cellId]!.velocity[row.axis]!, 0) / Math.max(weights, 1e-20);
    return (flux + Math.max(0, row.areaFineCells2 - covered) * fallback) / row.areaFineCells2;
  });
  return { cells, faces };
}

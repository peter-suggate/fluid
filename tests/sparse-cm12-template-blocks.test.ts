import assert from "node:assert/strict";
import test from "node:test";
import { createSparseAdaptiveMassAtlas, sparseAtlasBrickKey, sparseBrickSpan,
  type SparseAdaptiveMassAtlas, type SparseAdaptiveMassBrick, type SparseBrickResolution } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid, type SparseAtlasCompositeGrid,
  type SparseAtlasGradientRow } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { SparseCM12TemplateBlockCache, type SparseCM12TemplateBlockPlacement } from "../lib/methods/adaptive-mass/sparse-cm12-template-blocks";
import { packSparseCM12ResidentTopologyBlocksForQA,
  packSparseCM12ResidentTopologyTemplatesForQA } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

type Vec3 = readonly [number, number, number];
function atlas(dimensions: Vec3, leaves: readonly { q: Vec3; span: number; r: number }[], shift: Vec3 = [0, 0, 0]) {
  const size = dimensions.map((n, axis) => n + 8 * shift[axis]!) as [number, number, number];
  const brickDimensions = size.map(n => Math.ceil(n / 8)) as [number, number, number];
  return createSparseAdaptiveMassAtlas(size, leaves.map(leaf => {
    const coordinate = leaf.q.map((q, axis) => q + shift[axis]!) as [number, number, number];
    return { coordinate, key: sparseAtlasBrickKey(coordinate, { brickDimensions, signedCoordinates: true }),
      spanBricks: leaf.span, resolution: leaf.r as SparseBrickResolution,
      density: new Float64Array(leaf.r ** 3), gamma: new Float64Array(leaf.r ** 3) };
  }), 0, 8, true);
}

function checkBlock(block: SparseCM12TemplateBlockPlacement, rows: readonly SparseAtlasGradientRow[],
  grid: SparseAtlasCompositeGrid, leaves: readonly SparseAdaptiveMassBrick[]) {
  const slotByKey = new Map(leaves.map((leaf, slot) => [leaf.key, slot]));
  assert.deepEqual(block.rows.map(row => ({ ...row,
    centerFine: row.centerFine.map((q, axis) => q + block.originFine[axis]!) })), rows.map(row => ({
    kind: row.kind, axis: row.axis, centerFine: row.centerFine,
    area: row.area, distance: row.distance, dualWeight: row.dualWeight, exteriorPhi: row.exteriorPhi,
    terms: row.terms.map(term => ({ slot: slotByKey.get(grid.cells[term.cellId]!.brickKey)!,
      local: grid.cells[term.cellId]!.localIndex, coefficient: term.coefficient })),
  })));
}

function check(cache: SparseCM12TemplateBlockCache, source: SparseAdaptiveMassAtlas) {
  const grid = buildSparseAtlasCompositeGrid(source);
  for (const brick of source.bricks) {
    checkBlock(cache.interior(source, brick), grid.gradientRows.filter(row => row.kind === "intra-brick"
      && row.negativeBrickKey === brick.key), grid, [brick]);
    for (const axis of [0, 1, 2] as const) for (const side of [-1, 1] as const) {
      const face = brick.coordinate[axis] + (side > 0 ? sparseBrickSpan(brick) : 0);
      const neighbors = source.bricks.filter(neighbor => neighbor.key !== brick.key
        && neighbor.coordinate[axis] + (side < 0 ? sparseBrickSpan(neighbor) : 0) === face
        && ([0, 1, 2] as const).every(tangent => tangent === axis
          || Math.min(brick.coordinate[tangent] + sparseBrickSpan(brick),
            neighbor.coordinate[tangent] + sparseBrickSpan(neighbor))
          > Math.max(brick.coordinate[tangent], neighbor.coordinate[tangent])));
      if (side > 0) for (const neighbor of neighbors) checkBlock(cache.interface(source, brick, neighbor, axis),
        grid.gradientRows.filter(row => row.axis === axis && (row.kind === "brick-face" || row.kind === "mixed-seam")
          && row.negativeBrickKey === brick.key && row.positiveBrickKey === neighbor.key), grid, [brick, neighbor]);
      checkBlock(cache.air(source, brick, axis, side, neighbors),
        grid.gradientRows.filter(row => row.kind === "sparse-air" && row.axis === axis
          && (side < 0 ? row.positiveBrickKey : row.negativeBrickKey) === brick.key), grid, [brick, ...neighbors]);
    }
  }
}

for (const fixture of [
  { name: "clipped adjacent", dims: [13, 7, 5] as const, leaves: [
    { q: [0, 0, 0] as const, span: 1, r: 4 }, { q: [1, 0, 0] as const, span: 1, r: 8 }] },
  { name: "partially covered macro face", dims: [24, 16, 16] as const, leaves: [
    { q: [0, 0, 0] as const, span: 2, r: 2 }, { q: [2, 0, 0] as const, span: 1, r: 4 },
    { q: [2, 1, 0] as const, span: 1, r: 8 }] },
]) test(`${fixture.name}: translated local blocks preserve every geometric coefficient and term order`, () => {
  const cache = new SparseCM12TemplateBlockCache();
  check(cache, atlas(fixture.dims, fixture.leaves));
  const compiled = cache.statistics;
  check(cache, atlas(fixture.dims, fixture.leaves, [32, 16, 8]));
  check(cache, atlas(fixture.dims, fixture.leaves, [992, 480, 992]));
  assert.deepEqual(cache.statistics, compiled, "translated instances reuse every already compiled shape");
});

test("block catalog preserves native and row IDs across a chunk halo with reversed leaf storage", () => {
  const source = atlas([129 * 8, 8, 8], Array.from({ length: 129 }, (_, x) => ({
    q: [x, 0, 0] as const, span: 1, r: 1,
  })).reverse());
  const grid = buildSparseAtlasCompositeGrid(source);
  const reference = packSparseCM12ResidentTopologyTemplatesForQA(source, grid);
  const blocks = packSparseCM12ResidentTopologyBlocksForQA(source, grid);
  assert.equal(blocks.words.length, reference.words.length);
  for (let word = 0; word < reference.words.length; word++)
    if (blocks.words[word] !== reference.words[word]) assert.fail(
      `cross-chunk SCMT word ${word}: ${blocks.words[word]} != ${reference.words[word]}`);
});

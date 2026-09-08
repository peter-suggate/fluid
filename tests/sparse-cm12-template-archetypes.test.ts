import assert from "node:assert/strict";
import test from "node:test";
import { createSparseAdaptiveMassAtlas, sparseAtlasBrickKey, type SparseBrickResolution } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { packSparseCM12ResidentTopologyArchetypesForQA, packSparseCM12ResidentTopologyBlocksForQA,
  packSparseCM12ResidentTopologyTemplatesForQA } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

export const archetypeFixtures = [
  { name: "single full B8", dimensions: [8, 8, 8], bricks: [{ q: [0, 0, 0], span: 1, r: 2 }] },
  { name: "mixed adjacent B8", dimensions: [16, 8, 8], bricks: [{ q: [0, 0, 0], span: 1, r: 2 }, { q: [1, 0, 0], span: 1, r: 4 }] },
  { name: "clipped adjacent B8", dimensions: [13, 7, 5], bricks: [{ q: [0, 0, 0], span: 1, r: 4 }, { q: [1, 0, 0], span: 1, r: 8 }] },
  { name: "signed full frontier", dimensions: [8, 8, 8], bricks: [{ q: [-1, 0, 0], span: 1, r: 2, unclipped: true }, { q: [0, 0, 0], span: 1, r: 4 }] },
  { name: "immutable macro guard", dimensions: [24, 16, 16], bricks: [{ q: [0, 0, 0], span: 2, r: 2 }, { q: [2, 0, 0], span: 1, r: 4 }] },
  { name: "three accepted rungs", dimensions: [24, 8, 8], bricks: [{ q: [0, 0, 0], span: 1, r: 1 }, { q: [1, 0, 0], span: 1, r: 2 }, { q: [2, 0, 0], span: 1, r: 4 }] },
] as const;

export function archetypeFixtureAtlas(fixture: typeof archetypeFixtures[number]) {
  const brickDimensions = fixture.dimensions.map(n => Math.ceil(n / 8)) as [number, number, number];
  return createSparseAdaptiveMassAtlas(fixture.dimensions, fixture.bricks.map(brick => ({
    key: sparseAtlasBrickKey(brick.q, { brickDimensions, signedCoordinates: true }),
    coordinate: brick.q, spanBricks: brick.span, resolution: brick.r as SparseBrickResolution,
    unclipped: "unclipped" in brick ? brick.unclipped : undefined,
    density: Float64Array.from({ length: brick.r ** 3 }, (_, local) => (local % 7) / 7),
    gamma: Float64Array.from({ length: brick.r ** 3 }, (_, local) => 1 + local / 100),
  })), 0, 8, true);
}

for (const fixture of archetypeFixtures) test(`${fixture.name}: interned topology shadow is word-exact`, () => {
  const atlas = archetypeFixtureAtlas(fixture), grid = buildSparseAtlasCompositeGrid(atlas);
  const reference = packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid);
  const compact = packSparseCM12ResidentTopologyArchetypesForQA(atlas, grid);
  assert.equal(compact.cellCount, reference.cellCount);
  assert.equal(compact.rowCount, reference.rowCount);
  let mismatch = -1;
  for (let word = 0; word < Math.max(compact.words.length, reference.words.length); word++)
    if (compact.words[word] !== reference.words[word]) { mismatch = word; break; }
  assert.equal(mismatch, -1, `SCMT word ${mismatch}: ${compact.words[mismatch]} != ${reference.words[mismatch]}`);
  assert.ok(compact.gpuExpansion);
  assert.ok(compact.gpuExpansion.archetypeCount <= compact.rowCount);
  const blocks = packSparseCM12ResidentTopologyBlocksForQA(atlas, grid);
  assert.equal(blocks.words.length, reference.words.length);
  for (let word = 0; word < reference.words.length; word++)
    if (blocks.words[word] !== reference.words[word]) assert.fail(
      `block SCMT word ${word}: ${blocks.words[word]} != ${reference.words[word]}`);
});

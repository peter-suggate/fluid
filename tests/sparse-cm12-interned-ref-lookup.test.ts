import assert from "node:assert/strict";
import test from "node:test";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { compileSparseCM12FactoredAEIPackedTemplateCatalog } from
  "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-packed-template";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { compileSparseCM12InternedBoundaryOperators } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";
import { createSparseCM12InternedRefLookup,
  sparseCM12InternedRefLookup } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-ref-lookup";
import { createSparseCM12InternedRefLookupWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-ref-lookup.wgsl";
import { packSparseCM12ResidentTopologyTemplatesForQA } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const fixture = () => {
  const logical = [2, 1, 1] as const;
  const brick = (x: number): SparseAdaptiveMassBrick => ({
    key: sparseBrickKey([x, 0, 0], logical), coordinate: [x, 0, 0], resolution: 8,
    density: new Float64Array(8 ** 3), gamma: new Float64Array(8 ** 3).fill(1),
  });
  const atlas = createSparseAdaptiveMassAtlas([16, 8, 8], [brick(0), brick(1)],
    1, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const packed = packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid);
  const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({
    words: packed.words, brickFineResolution: 8,
    brickKeyByLeafId: atlas.bricks.map((value) => value.key),
    validDimensions: (_leaf, resolution) => [resolution, resolution, resolution],
    scaleLog2: (_leaf, resolution) => Math.log2(8 / resolution),
    selectedResolution: () => 8,
  });
  return compileSparseCM12InternedBoundaryOperators({ catalog,
    packedWords: packed.words, activeLeaves: [0, 1] });
};

test("IRL1 exhaustively resolves every packed all-rung patch instantiation", () => {
  const ibo = fixture();
  const lookup = createSparseCM12InternedRefLookup({ ibo,
    baseWords: ibo.layout.immutableWords });
  for (const patch of ibo.catalog.patches) {
    assert.deepEqual(sparseCM12InternedRefLookup({ lookup,
      sourceCanonicalId: patch.sourceCanonicalId, side: patch.sourceSide,
      targetCanonicalId: patch.targetCanonicalId }),
    [ibo.templateIdByPatch[patch.id]!, patch.targetLeaf,
      ibo.rowBaseByPatch[patch.id]!]);
  }
  assert.equal(lookup.layout.entryCount, ibo.catalog.patches.length);
  assert.ok(lookup.layout.maximumEntriesPerSide <= 16);
  assert.equal(sparseCM12InternedRefLookup({ lookup,
    sourceCanonicalId: 0, side: 0, targetCanonicalId: 12345 }), undefined);
});

test("IRL1 WGSL exposes a relocated bounded scheduled-ref lookup", () => {
  const ibo = fixture();
  const lookup = createSparseCM12InternedRefLookup({ ibo,
    baseWords: ibo.layout.immutableWords });
  const baseWords = 65536;
  const source = createSparseCM12InternedRefLookupWGSL({ layout: lookup.layout,
    arenaName: "fixtureArena", iboPrefix: "fixture", baseWords });
  assert.match(source, /fn fixtureIBOInstantiationCount/);
  assert.match(source, /fn fixtureIBOInstantiationEntry/);
  assert.match(source, /fn fixtureIBOFindScheduledRef/);
  assert.match(source, new RegExp(
    `const IRL1_DIRECTORY:u32=${baseWords + lookup.layout.directoryBaseWords}u`,
  ));
  assert.doesNotMatch(source, /rowTerm|incidence|fallback|for\(var leaf/);
});

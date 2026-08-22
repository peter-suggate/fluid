import assert from "node:assert/strict";
import test from "node:test";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { compileSparseCM12FactoredAEIPackedTemplateCatalog } from
  "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-packed-template";
import { createSparseAdaptiveMassAtlas, sparseBrickKey,
  type SparseAdaptiveMassBrick } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { compileSparseCM12InternedBoundaryOperators,
  SPARSE_CM12_INTERNED_BOUNDARY_SLOT_REF_WORDS } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";
import { createSparseCM12InternedBoundaryImage,
  prepareSparseCM12InternedBoundaryShadow } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-image";
import { compareSparseCM12IBOSemanticAuthority,
  compileSparseCM12GeometryDeltaClosure,
  compileSparseCM12GeometryFaceNeighbors } from
  "../lib/methods/adaptive-mass/sparse-cm12-ibo-semantic-authority";
import { packSparseCM12AcceptedTopologyTemplatesForQA,
  packSparseCM12ResidentTopologyTemplatesForQA } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const fixture = (allRungs = false) => {
  const logical = [2, 1, 1] as const;
  const brick = (x: number): SparseAdaptiveMassBrick => ({
    key: sparseBrickKey([x, 0, 0], logical), coordinate: [x, 0, 0], resolution: 8,
    density: new Float64Array(512), gamma: new Float64Array(512).fill(1),
  });
  const atlas = createSparseAdaptiveMassAtlas([16, 8, 8], [brick(0), brick(1)],
    1, undefined, 8); const grid = buildSparseAtlasCompositeGrid(atlas);
  const packed = allRungs ? packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid)
    : packSparseCM12AcceptedTopologyTemplatesForQA(atlas, grid);
  const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({
    words: packed.words, brickFineResolution: 8,
    brickKeyByLeafId: atlas.bricks.map((value) => value.key),
    validDimensions: (_leaf, resolution) => [resolution, resolution, resolution],
    scaleLog2: (_leaf, resolution) => Math.log2(8 / resolution),
    selectedResolution: () => 8,
  });
  const compilation = compileSparseCM12InternedBoundaryOperators({ catalog,
    packedWords: packed.words, activeLeaves: [0, 1] });
  return { packed, catalog, image: createSparseCM12InternedBoundaryImage(
    compilation, [0, 1], 1) };
};

test("ISA1 SCMT semantics exactly equal expanded IBO without reading IRL", () => {
  const { packed, catalog, image } = fixture();
  const authority = compareSparseCM12IBOSemanticAuthority({ image,
    packedWords: packed.words, activeLeaves: [0, 1],
    descriptorIdByLeaf: catalog.descriptorIdByLeaf, leaves: [0, 1], slot: 0 });
  assert.equal(authority.exact, true);
  assert.equal(authority.duplicateCandidateRows, 0);
  assert.ok(authority.totalRows > 0);
  let ref = image.layout.slotRefBaseWords[0];
  while (ref < image.layout.slotBaseWords[1] && image.words[ref] === 0xffff_ffff) {
    ref += SPARSE_CM12_INTERNED_BOUNDARY_SLOT_REF_WORDS;
  }
  assert.ok(ref < image.layout.slotBaseWords[1]);
  image.words[ref + 1]! += 1;
  assert.equal(compareSparseCM12IBOSemanticAuthority({ image,
    packedWords: packed.words, activeLeaves: [0, 1],
    descriptorIdByLeaf: catalog.descriptorIdByLeaf, leaves: [0, 1], slot: 0 }).exact,
  false);
});

test("ISA1 geometry CSR produces exact deduped face closure", () => {
  const geometry = compileSparseCM12GeometryFaceNeighbors({
    coordinates: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0]],
    spans: [1, 1, 1, 1],
  });
  assert.deepEqual([...geometry.offsets], [0, 2, 3, 4, 4]);
  assert.deepEqual(compileSparseCM12GeometryDeltaClosure({ geometry,
    changedLeaves: [0, 1] }).leaves, [0, 1, 2]);
  assert.ok(geometry.bytes < 1024);
});

test("ISA1 remains exact for a same-active-set rerung shadow", () => {
  const { packed, catalog, image } = fixture(true);
  const target = [...catalog.descriptorIdByLeaf];
  const rung4 = catalog.canonical.find((value) => value.leafId === 0
    && value.resolution === 4 && value.certified); assert.ok(rung4); target[0] = rung4.id;
  const prepared = prepareSparseCM12InternedBoundaryShadow({ image,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [0], candidateGeneration: 2 });
  const authority = compareSparseCM12IBOSemanticAuthority({ image,
    packedWords: packed.words, activeLeaves: [0, 1], descriptorIdByLeaf: target,
    leaves: prepared.deltaClosure, slot: prepared.shadowSlot });
  assert.equal(authority.exact, true);
  assert.equal(authority.duplicateCandidateRows, 0);
});

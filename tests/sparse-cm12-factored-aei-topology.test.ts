import assert from "node:assert/strict";
import test from "node:test";
import { buildSparseAtlasCompositeGrid, type SparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  SPARSE_CM12_FACTORED_AEI_FAULT,
  SPARSE_CM12_FACTORED_AEI_PATCHES_PER_LEAF,
  SPARSE_CM12_FACTORED_AEI_RELATION,
  SPARSE_CM12_FACTORED_AEI_SLOT_HEADER_WORDS,
  SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS,
  SPARSE_CM12_FACTORED_AEI_SLOT_PATCH_REF_WORDS,
  commitSparseCM12FactoredAEIShadow,
  compileSparseCM12FactoredAEICatalog,
  createSparseCM12FactoredAEIImage,
  prepareSparseCM12FactoredAEIShadow,
  validateSparseCM12FactoredAEIPreflip,
} from "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-topology";

function fixture(left: SparseBrickResolution, right: SparseBrickResolution,
  fine: 8 | 16, reverse = false): SparseAtlasCompositeGrid {
  const dimensions = [2 * fine, fine, fine] as const;
  const logical = [2, 1, 1] as const;
  const brick = (x: number, resolution: SparseBrickResolution): SparseAdaptiveMassBrick => ({
    key: sparseBrickKey([x, 0, 0], logical), coordinate: [x, 0, 0], resolution,
    density: new Float64Array(resolution ** 3),
    gamma: new Float64Array(resolution ** 3).fill(1),
  });
  const bricks = [brick(0, left), brick(1, right)];
  return buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas(
    dimensions, reverse ? bricks.reverse() : bricks, 1, fine,
  ));
}

test("factored AEI preserves authoritative atlas leaf order", () => {
  const grid = fixture(8, 8, 8, true);
  const catalog = compileSparseCM12FactoredAEICatalog(grid);
  assert.deepEqual(catalog.brickKeyByLeafId,
    grid.atlas.bricks.map((brick) => brick.key));
  for (let leaf = 0; leaf < grid.atlas.bricks.length; leaf += 1) {
    const descriptor = catalog.canonical[catalog.descriptorIdByLeaf[leaf]!]!;
    assert.equal(grid.cells[descriptor.cellFirst]!.brickKey,
      grid.atlas.bricks[leaf]!.key);
  }
});

test("factored AEI certifies interiors and keeps the sparse-air exterior explicit", () => {
  const catalog = compileSparseCM12FactoredAEICatalog(fixture(8, 8, 8));
  const activeDescriptors = catalog.descriptorIdByLeaf.map((id) => catalog.canonical[id]!);
  assert.equal(activeDescriptors.every((descriptor) => descriptor.certified), true);
  assert.equal(activeDescriptors.reduce((sum, descriptor) =>
    sum + descriptor.canonicalRowCount, 0), 2 * 3 * 7 * 8 * 8);
  const equalRung = catalog.patches.filter((patch) =>
    patch.relation === SPARSE_CM12_FACTORED_AEI_RELATION.equalRungCanonical);
  const sparseAir = catalog.patches.filter((patch) =>
    patch.relation === SPARSE_CM12_FACTORED_AEI_RELATION.explicitSparseAir);
  assert.equal(equalRung.length, 2);
  assert.equal(sparseAir.length, 10);
  assert.equal(catalog.patches.length, equalRung.length + sparseAir.length);
  assert.equal(equalRung.every((patch) => patch.mappingCertified), true);
  for (const patch of equalRung) {
    assert.equal(patch.faceDimensions[0] * patch.faceDimensions[1], patch.rowCount);
  }
  assert.equal(sparseAir.every((patch) => !patch.mappingCertified
    && patch.exceptionCount === patch.rowCount), true);
  assert.equal(catalog.exceptionRows.length,
    sparseAir.reduce((sum, patch) => sum + patch.rowCount, 0));
  assert.ok(catalog.layout.bytesPerSlot < 64 * 1024);
  assert.equal(catalog.layout.canonicalCapacity,
    catalog.layout.leafCapacity * catalog.layout.levelCount);
  const selectedRungSlotWords = SPARSE_CM12_FACTORED_AEI_SLOT_HEADER_WORDS
    + catalog.layout.leafCapacity * SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS
    + catalog.layout.leafCapacity * SPARSE_CM12_FACTORED_AEI_PATCHES_PER_LEAF
      * SPARSE_CM12_FACTORED_AEI_SLOT_PATCH_REF_WORDS;
  assert.equal(catalog.layout.bytesPerSlot,
    Math.ceil(selectedRungSlotWords / 64) * 64 * 4);
  assert.equal(catalog.canonical.filter((descriptor) =>
    descriptor.cellFirst === 0xffff_ffff).length,
  catalog.layout.canonicalCapacity - catalog.layout.leafCapacity);
});

test("mixed seams remain explicit stable row IDs", () => {
  const grid = fixture(8, 16, 16);
  const catalog = compileSparseCM12FactoredAEICatalog(grid);
  const mixed = catalog.patches.filter((patch) =>
    patch.relation === SPARSE_CM12_FACTORED_AEI_RELATION.explicitMixed);
  assert.equal(mixed.length, 2);
  assert.ok(catalog.exceptionRows.length > 0);
  for (const patch of mixed) {
    const rows = [...catalog.exceptionRows.subarray(
      patch.exceptionFirst, patch.exceptionFirst + patch.exceptionCount)];
    assert.deepEqual(rows, grid.gradientRows.filter((row) => row.kind === "mixed-seam")
      .map((row) => row.id));
  }
});

test("shadow delta patches fixed leaf/face records, validates, flips once and replays", () => {
  const catalog = compileSparseCM12FactoredAEICatalog(fixture(8, 8, 8));
  const image = createSparseCM12FactoredAEIImage(catalog, [0, 1], 7);
  const acceptedBefore = image.words[3];
  const generationBefore = image.words[4];
  const prepared = prepareSparseCM12FactoredAEIShadow({ image,
    targetActiveLeaves: [0], changedLeaves: [1], candidateGeneration: 8 });
  assert.deepEqual(prepared.deltaClosure, [0, 1]);
  assert.equal(image.words[3], acceptedBefore);
  assert.equal(image.words[4], generationBefore);
  const receipt = validateSparseCM12FactoredAEIPreflip({ image,
    targetActiveLeaves: [0], changedLeaves: [1] });
  assert.equal(receipt.passed, true);
  commitSparseCM12FactoredAEIShadow({ image, receipt,
    targetActiveLeaves: [0], changedLeaves: [1] });
  assert.equal(image.words[3], prepared.shadowSlot);
  assert.equal(image.words[4], 8);
  // Fixed local leaf records agree after retired-slot replay.
  for (let leaf = 0; leaf < catalog.layout.leafCapacity; leaf += 1) {
    const left = catalog.layout.slotLeafBaseWords[0]
      + leaf * SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS;
    const right = catalog.layout.slotLeafBaseWords[1]
      + leaf * SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS;
    assert.deepEqual([...image.words.subarray(left, left + 8)],
      [...image.words.subarray(right, right + 8)]);
  }
});

test("preflip fails closed on a corrupted patch reference", () => {
  const catalog = compileSparseCM12FactoredAEICatalog(fixture(8, 8, 8));
  const image = createSparseCM12FactoredAEIImage(catalog, [0, 1], 3);
  const prepared = prepareSparseCM12FactoredAEIShadow({ image,
    targetActiveLeaves: [0], changedLeaves: [1], candidateGeneration: 4 });
  image.words[catalog.layout.slotPatchRefBaseWords[prepared.shadowSlot] + 1] = 12345;
  const receipt = validateSparseCM12FactoredAEIPreflip({ image,
    targetActiveLeaves: [0], changedLeaves: [1] });
  assert.equal(receipt.passed, false);
  assert.equal(receipt.fault, SPARSE_CM12_FACTORED_AEI_FAULT.patchReference);
  assert.equal(image.words[3], 0);
  assert.equal(image.words[4], 3);
  assert.throws(() => commitSparseCM12FactoredAEIShadow({ image, receipt,
    targetActiveLeaves: [0], changedLeaves: [1] }), /invalid or stale/);
});

test("canonical term-order drift is certified as a preflip fault", () => {
  const source = fixture(8, 8, 8);
  const first = source.gradientRows.findIndex((row) => row.kind === "intra-brick");
  const rows = [...source.gradientRows];
  rows[first] = { ...rows[first]!, terms: [...rows[first]!.terms].reverse() };
  const grid = { ...source, gradientRows: rows } as SparseAtlasCompositeGrid;
  const catalog = compileSparseCM12FactoredAEICatalog(grid);
  assert.equal(catalog.canonical[catalog.descriptorIdByLeaf[0]!]!.certified, false);
  const image = createSparseCM12FactoredAEIImage(catalog, [0, 1], 1);
  prepareSparseCM12FactoredAEIShadow({ image, targetActiveLeaves: [0],
    changedLeaves: [1], candidateGeneration: 2 });
  const receipt = validateSparseCM12FactoredAEIPreflip({ image,
    targetActiveLeaves: [0], changedLeaves: [1] });
  assert.equal(receipt.passed, false);
  assert.equal(receipt.fault, SPARSE_CM12_FACTORED_AEI_FAULT.canonicalCertificate);
});

test("changed leaf receipt must exactly cover the accepted/target delta", () => {
  const catalog = compileSparseCM12FactoredAEICatalog(fixture(8, 8, 8));
  const image = createSparseCM12FactoredAEIImage(catalog, [0, 1], 1);
  assert.throws(() => prepareSparseCM12FactoredAEIShadow({ image,
    targetActiveLeaves: [0], changedLeaves: [], candidateGeneration: 2 }),
  /does not equal/);
});

test("preflip receipt checks the exact closure hash and candidate stamps", () => {
  const catalog = compileSparseCM12FactoredAEICatalog(fixture(8, 8, 8));
  const image = createSparseCM12FactoredAEIImage(catalog, [0, 1], 1);
  const prepared = prepareSparseCM12FactoredAEIShadow({ image,
    targetActiveLeaves: [0], changedLeaves: [1], candidateGeneration: 2 });
  const slotAt = catalog.layout.slotBaseWords[prepared.shadowSlot];
  image.words[slotAt + 10] ^= 1;
  let receipt = validateSparseCM12FactoredAEIPreflip({ image,
    targetActiveLeaves: [0], changedLeaves: [1] });
  assert.equal(receipt.passed, false);
  assert.equal(receipt.fault, SPARSE_CM12_FACTORED_AEI_FAULT.deltaCoverage);

  prepareSparseCM12FactoredAEIShadow({ image,
    targetActiveLeaves: [0], changedLeaves: [1], candidateGeneration: 2 });
  const inactiveLeafAt = catalog.layout.slotLeafBaseWords[prepared.shadowSlot]
    + SPARSE_CM12_FACTORED_AEI_SLOT_LEAF_WORDS;
  image.words[inactiveLeafAt] = 1;
  receipt = validateSparseCM12FactoredAEIPreflip({ image,
    targetActiveLeaves: [0], changedLeaves: [1] });
  assert.equal(receipt.passed, false);
  assert.equal(receipt.fault, SPARSE_CM12_FACTORED_AEI_FAULT.generation);
});

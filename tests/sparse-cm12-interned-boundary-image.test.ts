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
import {
  SPARSE_CM12_INTERNED_BOUNDARY_FAULT,
  commitSparseCM12InternedBoundaryShadow,
  createSparseCM12InternedBoundaryImage,
  prepareSparseCM12InternedBoundaryShadow,
  validateSparseCM12InternedBoundaryPreflip,
} from "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-image";
import { packSparseCM12AcceptedTopologyTemplatesForQA,
  packSparseCM12ResidentTopologyTemplatesForQA,
  sparseCM12InternedBoundaryMemoryPlan } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

test("IBO1 construction budgets cover the B8/P8 shipping lane", () => {
  assert.deepEqual(sparseCM12InternedBoundaryMemoryPlan(8), {
    immutableMaximumBytes: 1024 * 1024,
    slotMaximumBytes: 640 * 1024,
    semanticAuthorityMaximumBytes: 32 * 1024,
  });
  assert.deepEqual(sparseCM12InternedBoundaryMemoryPlan(16), {
    immutableMaximumBytes: 512 * 1024,
    slotMaximumBytes: 256 * 1024,
    semanticAuthorityMaximumBytes: 16_864,
  });
});

const compilation = (allRungs = false) => {
  const logical = [2, 1, 1] as const;
  const brick = (x: number): SparseAdaptiveMassBrick => ({
    key: sparseBrickKey([x, 0, 0], logical), coordinate: [x, 0, 0], resolution: 8,
    density: new Float64Array(8 ** 3), gamma: new Float64Array(8 ** 3).fill(1),
  });
  const atlas = createSparseAdaptiveMassAtlas([16, 8, 8], [brick(0), brick(1)],
    1, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const packed = allRungs
    ? packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid)
    : packSparseCM12AcceptedTopologyTemplatesForQA(atlas, grid);
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

const acceptedSnapshot = (image: ReturnType<typeof createSparseCM12InternedBoundaryImage>) => {
  const slot = (image.words[2]! & 1) as 0 | 1;
  const begin = image.layout.slotBaseWords[slot];
  const end = slot === 0 ? image.layout.slotBaseWords[1] : image.layout.totalWords;
  return { selector: image.words[2], generation: image.words[3], slot,
    words: image.words.slice(begin, end) };
};

const assertAcceptedUnchanged = (image: ReturnType<
  typeof createSparseCM12InternedBoundaryImage>, before: ReturnType<
    typeof acceptedSnapshot>) => {
  assert.equal(image.words[2], before.selector);
  assert.equal(image.words[3], before.generation);
  const begin = image.layout.slotBaseWords[before.slot];
  assert.deepEqual(image.words.slice(begin, begin + before.words.length), before.words);
};

test("IBO1 serializes exact canonical/template bits into its measured image", () => {
  const compiled = compilation();
  const image = createSparseCM12InternedBoundaryImage(compiled, [0, 1], 7);
  assert.equal(image.words.length, compiled.layout.totalWords);
  assert.equal(image.words[2], 0);
  assert.equal(image.words[3], 7);
  for (const template of compiled.templates) {
    const directory = compiled.layout.templateDirectoryBaseWords + 4 * template.id;
    const payload = image.words[directory]!;
    assert.deepEqual([...image.words.subarray(payload, payload + template.words.length)],
      [...template.words]);
    assert.deepEqual([...image.words.subarray(directory, directory + 4)],
      [payload, template.words.length, template.rowCount, template.termCount]);
  }
});

test("IBO1 retirement validates, flips once, and replays fixed local records", () => {
  const compiled = compilation();
  const image = createSparseCM12InternedBoundaryImage(compiled, [0, 1], 4);
  const prepared = prepareSparseCM12InternedBoundaryShadow({ image,
    targetActiveLeaves: [0], targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [1], candidateGeneration: 5 });
  assert.deepEqual(prepared.deltaClosure, [0, 1]);
  assert.equal(image.words[2], 0);
  assert.equal(image.words[3], 4);
  const receipt = validateSparseCM12InternedBoundaryPreflip({ image,
    targetActiveLeaves: [0], targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [1] });
  assert.equal(receipt.passed, true);
  assert.equal(receipt.selectorUnchanged, true);
  commitSparseCM12InternedBoundaryShadow({ image, receipt,
    targetActiveLeaves: [0], targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [1] });
  assert.equal(image.words[2], prepared.shadowSlot);
  assert.equal(image.words[3], 5);
  for (let leaf = 0; leaf < compiled.layout.leafCapacity; leaf += 1) {
    for (let word = 0; word < 8; word += 1) {
      assert.equal(image.words[compiled.layout.slotLeafBaseWords[0] + 8 * leaf + word],
        image.words[compiled.layout.slotLeafBaseWords[1] + 8 * leaf + word]);
    }
    for (let word = 0; word < 24 * 3; word += 1) {
      assert.equal(image.words[compiled.layout.slotRefBaseWords[0] + 24 * 3 * leaf + word],
        image.words[compiled.layout.slotRefBaseWords[1] + 24 * 3 * leaf + word]);
    }
  }
  assert.throws(() => commitSparseCM12InternedBoundaryShadow({ image, receipt,
    targetActiveLeaves: [0], targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [1] }), /stale/);
});

test("IBO1 activation commits once while rejected activation preserves accepted state", () => {
  const compiled = compilation();
  const target = compiled.catalog.descriptorIdByLeaf;
  const image = createSparseCM12InternedBoundaryImage(compiled, [0], 3, target);
  const prepared = prepareSparseCM12InternedBoundaryShadow({ image,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [1], candidateGeneration: 4 });
  const receipt = validateSparseCM12InternedBoundaryPreflip({ image,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [1] });
  assert.equal(receipt.passed, true);
  commitSparseCM12InternedBoundaryShadow({ image, receipt,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [1] });
  assert.equal(image.words[2], prepared.shadowSlot);
  assert.equal(image.words[3], 4);
  for (const slot of [0, 1] as const) {
    const leaf = image.layout.slotLeafBaseWords[slot] + 8;
    assert.equal(image.words[leaf + 1]! & 1, 1,
      "activation must be mirrored into both fixed slots after the flip");
  }

  const rejected = createSparseCM12InternedBoundaryImage(compiled, [0], 7, target);
  const before = acceptedSnapshot(rejected);
  const bad = prepareSparseCM12InternedBoundaryShadow({ image: rejected,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [1], candidateGeneration: 8 });
  const badLeaf = rejected.layout.slotLeafBaseWords[bad.shadowSlot] + 8;
  rejected.words[badLeaf + 2] ^= 1;
  const rejectedReceipt = validateSparseCM12InternedBoundaryPreflip({ image: rejected,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [1] });
  assert.equal(rejectedReceipt.passed, false);
  assert.throws(() => commitSparseCM12InternedBoundaryShadow({ image: rejected,
    receipt: rejectedReceipt, targetActiveLeaves: [0, 1],
    targetDescriptorIdByLeaf: target, changedLeaves: [1] }), /invalid or stale/);
  assertAcceptedUnchanged(rejected, before);
});

test("IBO1 rejected retirement preserves accepted state on preflip corruption", () => {
  const compiled = compilation();
  let image = createSparseCM12InternedBoundaryImage(compiled, [0, 1], 1);
  prepareSparseCM12InternedBoundaryShadow({ image,
    targetActiveLeaves: [0], targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [1], candidateGeneration: 2 });
  const payload = image.words[compiled.layout.templateDirectoryBaseWords]!;
  image.words[payload + 8] ^= 1;
  let receipt = validateSparseCM12InternedBoundaryPreflip({ image,
    targetActiveLeaves: [0], targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [1] });
  assert.equal(receipt.passed, false);
  assert.equal(receipt.fault, SPARSE_CM12_INTERNED_BOUNDARY_FAULT.immutableImage);

  image = createSparseCM12InternedBoundaryImage(compiled, [0, 1], 1);
  const before = acceptedSnapshot(image);
  const prepared = prepareSparseCM12InternedBoundaryShadow({ image,
    targetActiveLeaves: [0], targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [1], candidateGeneration: 2 });
  const refs = compiled.layout.slotRefBaseWords[prepared.shadowSlot];
  image.words[refs] = 0;
  receipt = validateSparseCM12InternedBoundaryPreflip({ image,
    targetActiveLeaves: [0], targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [1] });
  assert.equal(receipt.passed, false);
  assert.equal(receipt.fault, SPARSE_CM12_INTERNED_BOUNDARY_FAULT.reference);
  assert.throws(() => commitSparseCM12InternedBoundaryShadow({ image, receipt,
    targetActiveLeaves: [0], targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [1] }), /invalid or stale/);
  assertAcceptedUnchanged(image, before);
});

test("IBO1 requires the exact accepted/target changed-leaf set", () => {
  const compiled = compilation();
  const image = createSparseCM12InternedBoundaryImage(compiled, [0, 1], 1);
  assert.throws(() => prepareSparseCM12InternedBoundaryShadow({ image,
    targetActiveLeaves: [0], targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [], candidateGeneration: 2 }),
  /do not equal/);
});

test("IBO1 treats a same-active-set rerung as a changed leaf and rebuilds neighbors", () => {
  const compiled = compilation(true);
  const initial = [...compiled.catalog.descriptorIdByLeaf];
  const target = [...initial];
  const rung4 = compiled.catalog.canonical.find((descriptor) =>
    descriptor.leafId === 0 && descriptor.resolution === 4 && descriptor.certified);
  assert.ok(rung4);
  target[0] = rung4.id;
  const rejected = createSparseCM12InternedBoundaryImage(compiled, [0, 1], 9, initial);
  const before = acceptedSnapshot(rejected);
  const rejectedPrepared = prepareSparseCM12InternedBoundaryShadow({ image: rejected,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [0], candidateGeneration: 10 });
  const rejectedLeaf = rejected.layout.slotLeafBaseWords[rejectedPrepared.shadowSlot];
  rejected.words[rejectedLeaf + 2] = initial[0]!;
  const rejectedReceipt = validateSparseCM12InternedBoundaryPreflip({ image: rejected,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [0] });
  assert.equal(rejectedReceipt.passed, false);
  assert.throws(() => commitSparseCM12InternedBoundaryShadow({ image: rejected,
    receipt: rejectedReceipt, targetActiveLeaves: [0, 1],
    targetDescriptorIdByLeaf: target, changedLeaves: [0] }), /invalid or stale/);
  assertAcceptedUnchanged(rejected, before);

  const image = createSparseCM12InternedBoundaryImage(compiled, [0, 1], 9, initial);
  assert.throws(() => prepareSparseCM12InternedBoundaryShadow({ image,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [], candidateGeneration: 10 }), /do not equal/);
  const prepared = prepareSparseCM12InternedBoundaryShadow({ image,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [0], candidateGeneration: 10 });
  assert.deepEqual(prepared.deltaClosure, [0, 1]);
  const receipt = validateSparseCM12InternedBoundaryPreflip({ image,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [0] });
  assert.equal(receipt.passed, true);
  commitSparseCM12InternedBoundaryShadow({ image, receipt,
    targetActiveLeaves: [0, 1], targetDescriptorIdByLeaf: target,
    changedLeaves: [0] });
  for (const slot of [0, 1] as const) {
    assert.equal(image.words[compiled.layout.slotLeafBaseWords[slot] + 2], rung4.id);
  }
});

test("IBO1 inserts immutable execution supplements before both fixed slots", () => {
  const compiled = compilation();
  const supplement = new Uint32Array(64);
  supplement.set([0x4954_5231, 1, 2, 3]);
  const baseWords = compiled.layout.immutableWords;
  const image = createSparseCM12InternedBoundaryImage(compiled, [0, 1], 1,
    compiled.catalog.descriptorIdByLeaf,
    [{ label: "fixture", baseWords, words: supplement }]);
  assert.equal(image.layout.immutableWords, baseWords + supplement.length);
  assert.equal(image.layout.slotBaseWords[0], baseWords + supplement.length);
  assert.equal(image.words[11], image.layout.slotBaseWords[0]);
  assert.deepEqual([...image.words.subarray(baseWords, baseWords + 4)],
    [...supplement.subarray(0, 4)]);
  prepareSparseCM12InternedBoundaryShadow({ image, targetActiveLeaves: [0],
    targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [1], candidateGeneration: 2 });
  image.words[baseWords + 2] ^= 1;
  const receipt = validateSparseCM12InternedBoundaryPreflip({ image,
    targetActiveLeaves: [0],
    targetDescriptorIdByLeaf: compiled.catalog.descriptorIdByLeaf,
    changedLeaves: [1] });
  assert.equal(receipt.passed, false);
  assert.equal(receipt.fault, SPARSE_CM12_INTERNED_BOUNDARY_FAULT.immutableImage);
});

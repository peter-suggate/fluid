import assert from "node:assert/strict";
import test from "node:test";
import {
  SPARSE_CM12_PACKED_TEMPLATE_MAGIC,
  compileSparseCM12FactoredAEIPackedTemplateCatalog,
} from "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-packed-template";

const bits = (value: number): number => {
  const buffer = new ArrayBuffer(4);
  new Float32Array(buffer)[0] = value;
  return new Uint32Array(buffer)[0]!;
};

function completePackedTemplate(): Uint32Array {
  const levels = [1, 2, 4, 8];
  const cellCounts = levels.map((resolution) => resolution ** 3);
  const rowCounts = levels.map((resolution) =>
    3 * (resolution - 1) * resolution * resolution);
  const cellCount = cellCounts.reduce((sum, value) => sum + value, 0);
  const rowCount = rowCounts.reduce((sum, value) => sum + value, 0);
  const termCount = 2 * rowCount;
  let at = 27;
  const cellBase = at; at += 8 * cellCount;
  const rowBase = at; at += 9 * rowCount;
  const termBase = at; at += 2 * termCount;
  const cellRangeBase = at; at += 2 * levels.length;
  const rowOwnerBase = at; at += levels.length + 1;
  const candidateConfigurationBase = at; at += 6 * levels.length;
  const patchOffsetCount = 6 * levels.reduce((sum, resolution) =>
    sum + resolution ** 2 + 1, 0);
  const candidatePatchBase = at; at += patchOffsetCount;
  const candidateRowBase = at; at += 1;
  const words = new Uint32Array(at);
  words.set([SPARSE_CM12_PACKED_TEMPLATE_MAGIC, 1, cellCount, rowCount, termCount,
    0, cellBase, rowBase, termBase, 0, 0, cellRangeBase, 0, 1], 0);
  words[16] = rowOwnerBase;
  words[24] = candidateConfigurationBase;
  words[25] = candidatePatchBase;
  words[26] = candidateRowBase;

  let cellFirst = 0, rowFirst = 0, termFirst = 0, patchCursor = 0;
  for (let level = 0; level < levels.length; level += 1) {
    const resolution = levels[level]!;
    words[cellRangeBase + 2 * level] = cellFirst;
    words[cellRangeBase + 2 * level + 1] = resolution ** 3;
    words[rowOwnerBase + level] = rowFirst;
    for (let local = 0; local < resolution ** 3; local += 1) {
      words[cellBase + 8 * (cellFirst + local) + 7] = resolution;
    }
    const addRow = (axis: number, negative: number, positive: number) => {
      const row = rowFirst++;
      words[rowBase + row] = termFirst | (2 << 23);
      words[rowBase + rowCount + row] = axis << 30;
      words[rowBase + 2 * rowCount + row] = bits(1);
      words[rowBase + 3 * rowCount + row] = bits(1);
      words[rowBase + 4 * rowCount + row] = bits(1);
      words[termBase + 2 * termFirst] = negative;
      words[termBase + 2 * termFirst + 1] = bits(-1); termFirst += 1;
      words[termBase + 2 * termFirst] = positive;
      words[termBase + 2 * termFirst + 1] = bits(1); termFirst += 1;
    };
    for (let z = 0; z < resolution; z += 1) for (let y = 0; y < resolution; y += 1) {
      for (let x = 1; x < resolution; x += 1) {
        const positive = cellFirst + x + resolution * (y + resolution * z);
        addRow(0, positive - 1, positive);
      }
    }
    for (let z = 0; z < resolution; z += 1) for (let y = 1; y < resolution; y += 1) {
      for (let x = 0; x < resolution; x += 1) {
        const positive = cellFirst + x + resolution * (y + resolution * z);
        addRow(1, positive - resolution, positive);
      }
    }
    for (let z = 1; z < resolution; z += 1) for (let y = 0; y < resolution; y += 1) {
      for (let x = 0; x < resolution; x += 1) {
        const positive = cellFirst + x + resolution * (y + resolution * z);
        addRow(2, positive - resolution * resolution, positive);
      }
    }
    for (let side = 0; side < 6; side += 1) {
      words[candidateConfigurationBase + 6 * level + side]
        = candidatePatchBase + patchCursor;
      for (let boundary = 0; boundary <= resolution ** 2; boundary += 1) {
        words[candidatePatchBase + patchCursor++] = candidateRowBase;
      }
    }
    cellFirst += resolution ** 3;
  }
  words[rowOwnerBase + levels.length] = rowFirst;
  assert.equal(cellFirst, cellCount);
  assert.equal(rowFirst, rowCount);
  assert.equal(termFirst, termCount);
  assert.equal(patchCursor, patchOffsetCount);
  return words;
}

test("packed-template adapter certifies every authoritative rung", () => {
  const words = completePackedTemplate();
  const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({ words,
    brickFineResolution: 8, brickKeyByLeafId: [17],
    validDimensions: (_leaf, resolution) => [resolution, resolution, resolution],
    scaleLog2: (_leaf, resolution) => Math.log2(8 / resolution),
    selectedResolution: () => 8,
    immutableMaximumBytes: 0.5 * 2 ** 20 });
  assert.equal(catalog.layout.canonicalCapacity, 4);
  assert.equal(catalog.canonical.every((descriptor) => descriptor.certified), true);
  assert.deepEqual(catalog.canonical.map((descriptor) => descriptor.resolution),
    [1, 2, 4, 8]);
  assert.equal(catalog.canonical.reduce((sum, descriptor) =>
    sum + descriptor.canonicalRowCount, 0), words[3]);
  assert.equal(catalog.patches.length, 0);
  assert.equal(catalog.descriptorIdByLeaf[0], 3);
});

test("packed-template adapter fails closed on authoritative term-order drift", () => {
  const words = completePackedTemplate();
  const termBase = words[8]!;
  const selectedFirstRow = words[words[16]! + 3]!;
  const packed = words[words[7]! + selectedFirstRow]!;
  const firstTerm = packed & 0x007f_ffff;
  const at = termBase + 2 * firstTerm;
  const firstCell = words[at]!, firstBits = words[at + 1]!;
  words[at] = words[at + 2]!;
  words[at + 1] = words[at + 3]!;
  words[at + 2] = firstCell;
  words[at + 3] = firstBits;
  assert.throws(() => compileSparseCM12FactoredAEIPackedTemplateCatalog({ words,
    brickFineResolution: 8, brickKeyByLeafId: [17],
    validDimensions: (_leaf, resolution) => [resolution, resolution, resolution],
    scaleLog2: (_leaf, resolution) => Math.log2(8 / resolution),
    selectedResolution: () => 8 }), /term identity\/order differs/);
});

test("packed-template adapter marks unavailable nonselected rung aliases invalid", () => {
  const words = completePackedTemplate();
  const ranges = words[11]!;
  words[ranges] = words[ranges + 6]!;
  words[ranges + 1] = words[ranges + 7]!;
  const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({ words,
    brickFineResolution: 8, brickKeyByLeafId: [17],
    validDimensions: (_leaf, resolution) => [resolution, resolution, resolution],
    scaleLog2: (_leaf, resolution) => Math.log2(8 / resolution),
    selectedResolution: () => 8 });
  assert.equal(catalog.canonical[0]!.certified, false);
  assert.equal(catalog.canonical[0]!.cellFirst, 0xffff_ffff);
  assert.equal(catalog.canonical[3]!.certified, true);
});

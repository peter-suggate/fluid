import assert from "node:assert/strict";
import test from "node:test";
import {
  SPGRID_CELL_FLAG,
  buildSPGridContactLevelOracle,
} from "../lib/webgpu-octree-spgrid-vcycle";

const DIMENSIONS = [4, 4, 4] as const;
const LEAVES = [
  { origin: [0, 0, 0], size: 1 },
  { origin: [1, 0, 0], size: 1 },
  { origin: [2, 0, 0], size: 1 },
  { origin: [3, 0, 0], size: 1 },
] as const;
const ROW_COUNT = LEAVES.length;
const ROW_CAPACITY = ROW_COUNT;

interface CompactSystem {
  readonly l1Headers: Uint32Array<ArrayBuffer>;
  readonly l1Entries: Uint32Array<ArrayBuffer>;
  readonly l2Headers: Uint32Array<ArrayBuffer>;
  readonly l2Entries: Uint32Array<ArrayBuffer>;
  readonly matrix: readonly (readonly number[])[];
  readonly rhs: readonly number[];
}

/** Captured compact L1/L2 topology shared by the Section 4.3 numerical tests. */
function capturedSection43System(): CompactSystem {
  const l1 = Array.from({ length: ROW_COUNT },
    () => [] as Array<{ row: number; coefficient: number }>);
  const l2 = Array.from({ length: ROW_COUNT },
    () => [] as Array<{ row: number; coefficient: number }>);
  const connect = (left: number, right: number, coefficient: number) => {
    const l1Coefficient = Math.fround(0.001 + 0.01 * coefficient);
    l1[left].push({ row: right, coefficient: l1Coefficient });
    l1[right].push({ row: left, coefficient: l1Coefficient });
    l2[left].push({ row: right, coefficient });
    l2[right].push({ row: left, coefficient });
  };
  for (const [left, right, coefficient] of [
    [0, 1, 1.10], [1, 2, 0.65], [2, 3, 1.35],
  ] as const) connect(left, right, Math.fround(coefficient));
  l1.forEach((row) => row.sort((a, b) => a.row - b.row));
  l2.forEach((row) => row.sort((a, b) => a.row - b.row));

  const pack = (
    rows: readonly (readonly { row: number; coefficient: number }[])[],
    rhs: readonly number[],
    anchor: (row: number) => number,
  ) => {
    const entryCount = rows.reduce((sum, row) => sum + row.length, 0);
    const headerWords = new Uint32Array(ROW_CAPACITY * 12);
    const headerFloats = new Float32Array(headerWords.buffer);
    const entryWords = new Uint32Array(entryCount * 2);
    const entryFloats = new Float32Array(entryWords.buffer);
    let cursor = 0;
    rows.forEach((entries, row) => {
      const base = row * 12;
      const leaf = LEAVES[row];
      headerWords[base] = leaf.origin[0] + DIMENSIONS[0]
        * (leaf.origin[1] + DIMENSIONS[1] * leaf.origin[2]);
      headerWords[base + 1] = cursor;
      headerWords[base + 2] = entries.length;
      headerWords[base + 3] = leaf.size;
      headerFloats[base + 4] = Math.fround(anchor(row)
        + entries.reduce((sum, entry) => sum + entry.coefficient, 0));
      headerFloats[base + 5] = Math.fround(-rhs[row]);
      headerWords[base + 6] = 1;
      for (const entry of entries) {
        entryWords[cursor * 2] = entry.row;
        entryFloats[cursor * 2 + 1] = entry.coefficient;
        cursor += 1;
      }
    });
    return { headerWords, entryWords };
  };

  const rhs = [1.1, -0.35, 0.72, -1.24].map(Math.fround);
  const packedL1 = pack(l1, rhs, (row) => Math.fround(2
    - l1[row].reduce((sum, entry) => sum + entry.coefficient, 0)));
  const l2Anchors = [0.012, 0.027, 0.018, 0.041] as const;
  const packedL2 = pack(l2, rhs, (row) => Math.fround(l2Anchors[row]));
  const matrix = Array.from({ length: ROW_COUNT }, (_, row) => {
    const result = new Array<number>(ROW_COUNT).fill(0);
    result[row] = new Float32Array(packedL2.headerWords.buffer)[row * 12 + 4];
    for (const entry of l2[row]) result[entry.row] = -entry.coefficient;
    return result;
  });
  return {
    l1Headers: packedL1.headerWords,
    l1Entries: packedL1.entryWords,
    l2Headers: packedL2.headerWords,
    l2Entries: packedL2.entryWords,
    matrix,
    rhs,
  };
}

test("Section 4.3 compact L1/L2 fixture is symmetric and strictly positive", () => {
  const system = capturedSection43System();
  assert.equal(system.l1Headers.byteLength, ROW_CAPACITY * 48);
  assert.equal(system.l2Headers.byteLength, ROW_CAPACITY * 48);
  assert.equal(system.l1Entries.byteLength, system.l2Entries.byteLength,
    "captured and live operators must share one exact entry topology");
  for (let row = 0; row < ROW_COUNT; row += 1) {
    let offDiagonal = 0;
    for (let column = 0; column < ROW_COUNT; column += 1) {
      assert.equal(system.matrix[row][column], system.matrix[column][row]);
      if (column !== row) offDiagonal += Math.abs(system.matrix[row][column]);
    }
    assert.ok(system.matrix[row][row] > offDiagonal,
      `row ${row} must retain a strict positive anchor`);
    const base = row * 12;
    const cell = system.l1Headers[base];
    const origin = [
      cell % DIMENSIONS[0],
      Math.floor(cell / DIMENSIONS[0]) % DIMENSIONS[1],
      Math.floor(cell / (DIMENSIONS[0] * DIMENSIONS[1])),
    ];
    const size = system.l1Headers[base + 3];
    assert.ok(origin.every((value, axis) => value + size <= DIMENSIONS[axis]));
    const entryBegin = system.l1Headers[base + 1];
    const entryCount = system.l1Headers[base + 2];
    assert.ok(entryBegin + entryCount <= system.l1Entries.length / 2);
    for (let entry = entryBegin; entry < entryBegin + entryCount; entry += 1) {
      assert.ok(system.l1Entries[entry * 2] < ROW_COUNT);
      assert.equal(system.l1Entries[entry * 2], system.l2Entries[entry * 2]);
    }
  }
  assert.equal(system.rhs.length, ROW_COUNT);
  assert.deepEqual(
    Array.from({ length: ROW_COUNT }, (_, row) => system.l1Headers[row * 12 + 3]),
    [1, 1, 1, 1],
  );
});

test("mixed-resolution SPGrid ghost propagation remains an exact E/E-transpose pair", () => {
  const contact = buildSPGridContactLevelOracle([
    { origin: [0, 0, 0], size: 2 },
    { origin: [2, 0, 0], size: 1 },
  ], [{ negative: 0, positive: 1, coefficient: 0.75 }], 0);
  const ghost = contact.flags.findIndex((flags) => (flags & SPGRID_CELL_FLAG.ghost) !== 0);
  assert.ok(ghost >= 0, "coarse/fine contact must create a level-0 ghost alias");
  for (let slot = 0; slot < contact.flags.length; slot += 1) {
    for (let leaf = 0; leaf < 2; leaf += 1) {
      assert.equal(
        contact.propagate[slot * 2 + leaf],
        contact.accumulate[leaf * contact.flags.length + slot],
      );
    }
  }
});

import assert from "node:assert/strict";
import test from "node:test";
import { quadtreePackScanIOShader, quadtreePackScanShader } from "../lib/webgpu-quadtree-pack-builder";

type Word4 = [number, number, number, number];
const add = (a: Word4, b: Word4): Word4 => a.map((value, i) => value + b[i]) as Word4;

// CPU model of the two GPU passes. This intentionally models block-local
// exclusives and the 256-lane chunked block-total scan separately.
function hierarchicalExclusiveScan(input: Word4[]): Word4[] {
  const blockSize = 256, output = input.map((): Word4 => [0, 0, 0, 0]), blockTotals: Word4[] = [];
  for (let base = 0; base < input.length; base += blockSize) {
    let cursor: Word4 = [0, 0, 0, 0];
    for (let i = base; i < Math.min(base + blockSize, input.length); i += 1) { output[i] = cursor; cursor = add(cursor, input[i]); }
    blockTotals.push(cursor);
  }
  const chunk = Math.ceil(blockTotals.length / blockSize), chunkTotals: Word4[] = [];
  for (let lane = 0; lane < blockSize; lane += 1) {
    let total: Word4 = [0, 0, 0, 0];
    for (let block = lane * chunk; block < Math.min((lane + 1) * chunk, blockTotals.length); block += 1) total = add(total, blockTotals[block]);
    chunkTotals.push(total);
  }
  let chunkPrefix: Word4 = [0, 0, 0, 0];
  for (let lane = 0; lane < blockSize; lane += 1) {
    let cursor = chunkPrefix;
    for (let block = lane * chunk; block < Math.min((lane + 1) * chunk, blockTotals.length); block += 1) { const count = blockTotals[block]; blockTotals[block] = cursor; cursor = add(cursor, count); }
    chunkPrefix = add(chunkPrefix, chunkTotals[lane]);
  }
  return output.map((local, i) => add(local, blockTotals[Math.floor(i / blockSize)]));
}

test("parallel pack scan is exactly exclusive across tile and block-summary boundaries", () => {
  for (const length of [1, 255, 256, 257, 65_535, 65_537]) {
    let state = 0x12345678;
    const input = Array.from({ length }, (): Word4 => {
      state = (1664525 * state + 1013904223) >>> 0;
      return [state & 3, (state >>> 2) & 7, (state >>> 5) & 1, (state >>> 6) & 3];
    });
    const expected: Word4[] = []; let cursor: Word4 = [0, 0, 0, 0];
    for (const value of input) { expected.push(cursor); cursor = add(cursor, value); }
    assert.deepEqual(hierarchicalExclusiveScan(input), expected, `length ${length}`);
  }
});

test("parallel pack scans retain face axis ordering and portable workgroup structure", () => {
  const counts = [[2, 1, 3], [0, 4, 1], [5, 0, 2]];
  const packed = [...counts.map((v) => v[0]), ...counts.map((v) => v[1]), ...counts.map((v) => v[2])].map((value): Word4 => [value, 0, 0, 0]);
  const offsets = hierarchicalExclusiveScan(packed).map((v) => v[0]);
  assert.deepEqual(counts.map((_, slot) => [offsets[slot], offsets[counts.length + slot], offsets[2 * counts.length + slot]]), [[0, 7, 12], [2, 8, 15], [2, 12, 16]]);
  assert.doesNotMatch(quadtreePackScanShader, /@workgroup_size\(1\)/);
  assert.match(quadtreePackScanShader, /var<workgroup> partials: array<vec4u, 256>/);
  assert.match(quadtreePackScanIOShader, /fn prepareSegmentScan/);
  assert.match(quadtreePackScanIOShader, /fn finishFaceScan/);
  assert.match(quadtreePackScanIOShader, /fn finishRowScan/);
});

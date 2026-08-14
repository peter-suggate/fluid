import assert from "node:assert/strict";
import test from "node:test";

import {
  OCTREE_WORKSET_FLAGS,
  OCTREE_WORKSET_HEADER_WORDS,
  OCTREE_WORKSET_INDIRECT_OFFSET_BYTES,
  OCTREE_WORKSET_INVALID_ID,
  buildMarkedOctreeWorkset,
  buildOctreeWorkset,
  compactOctreeMarked,
  isOctreeEpochNewer,
  nextOctreeEpoch,
  octreeDispatchForCount,
  rankOctreeMarks,
  scatterOctreeMarked,
  unpackOctreeWorksetHeader,
  validateOctreeWorkset,
} from "../lib/methods/octree-shared/webgpu-octree-worksets";

test("common header places the indirect record at words four through six", () => {
  const packet = buildOctreeWorkset([9, 3, 7], {
    epoch: 12,
    capacity: 5,
    workgroupSize: 2,
  });
  assert.equal(OCTREE_WORKSET_HEADER_WORDS, 7);
  assert.equal(OCTREE_WORKSET_INDIRECT_OFFSET_BYTES, 16);
  assert.deepEqual([...packet.slice(0, 7)], [
    12,
    3,
    5,
    OCTREE_WORKSET_FLAGS.ready | OCTREE_WORKSET_FLAGS.validated,
    2,
    1,
    1,
  ]);
  assert.deepEqual([...packet.slice(7, 10)], [9, 3, 7]);
  assert.deepEqual([...packet.slice(10)], [OCTREE_WORKSET_INVALID_ID, OCTREE_WORKSET_INVALID_ID]);
  assert.deepEqual(unpackOctreeWorksetHeader(packet).dispatch, [2, 1, 1]);
  assert.deepEqual(validateOctreeWorkset(packet, 12, 2), []);
});

test("empty worksets always dispatch zero by one by one and contain no element work", () => {
  for (const workgroupSize of [1, 32, 64, 128]) {
    assert.deepEqual(octreeDispatchForCount(0, workgroupSize), [0, 1, 1]);
    const packet = buildOctreeWorkset([], {
      epoch: 4,
      capacity: 8,
      workgroupSize,
    });
    const header = unpackOctreeWorksetHeader(packet);
    assert.equal(header.count, 0);
    assert.deepEqual(header.dispatch, [0, 1, 1]);
    assert.ok(packet.slice(OCTREE_WORKSET_HEADER_WORDS)
      .every((word) => word === OCTREE_WORKSET_INVALID_ID));
  }
});

test("mark, exclusive rank, and scatter are stable and reject malformed marks", () => {
  const marks = Uint32Array.from([0, 1, 1, 0, 1, 0]);
  const ranking = rankOctreeMarks(marks);
  assert.equal(ranking.count, 3);
  assert.deepEqual([...ranking.ranks], [
    OCTREE_WORKSET_INVALID_ID, 0, 1, OCTREE_WORKSET_INVALID_ID, 2, OCTREE_WORKSET_INVALID_ID,
  ]);
  assert.deepEqual(scatterOctreeMarked(["a", "b", "c", "d", "e", "f"], marks, ranking),
    ["b", "c", "e"]);
  assert.deepEqual(compactOctreeMarked([50, 40, 30, 20, 10, 0], marks).values, [40, 30, 10]);
  assert.throws(() => rankOctreeMarks([0, 2]), /zero or one/);
  const brokenRanks = { count: 2, ranks: Uint32Array.from([0, 0]) };
  assert.throws(() => scatterOctreeMarked([1, 2], [1, 1], brokenRanks), /valid exclusive prefix/);
});

test("marked workset construction keeps canonical source order", () => {
  const packet = buildMarkedOctreeWorkset(
    Uint32Array.from([41, 17, 8, 99, 3]),
    Uint32Array.from([1, 0, 1, 0, 1]),
    { epoch: 1, capacity: 4, workgroupSize: 64 },
  );
  assert.deepEqual([...packet.slice(7, 10)], [41, 8, 3]);
  assert.deepEqual(validateOctreeWorkset(packet, 1, 64), []);
});

test("workset construction rejects overflow, duplicate, invalid, and stale packets", () => {
  assert.throws(() => buildOctreeWorkset([1, 2], {
    epoch: 1, capacity: 1, workgroupSize: 64,
  }), /capacity/);
  assert.throws(() => buildOctreeWorkset([1, 1], {
    epoch: 1, capacity: 2, workgroupSize: 64,
  }), /duplicate/);
  assert.throws(() => buildOctreeWorkset([OCTREE_WORKSET_INVALID_ID], {
    epoch: 1, capacity: 1, workgroupSize: 64,
  }), /reserved invalid/);
  const packet = buildOctreeWorkset([4], { epoch: 7, capacity: 1, workgroupSize: 64 });
  assert.match(validateOctreeWorkset(packet, 8, 64).join("; "), /epoch 7/);
  packet[4] = 0;
  assert.match(validateOctreeWorkset(packet, 7, 64).join("; "), /dispatch/);
});

test("epoch helpers reserve zero and compare correctly through uint32 wrap", () => {
  assert.equal(nextOctreeEpoch(0), 1);
  assert.equal(nextOctreeEpoch(0xffff_ffff), 1);
  assert.ok(isOctreeEpochNewer(8, 7));
  assert.ok(isOctreeEpochNewer(1, 0xffff_ffff));
  assert.equal(isOctreeEpochNewer(7, 7), false);
  assert.equal(isOctreeEpochNewer(6, 7), false);
  assert.equal(isOctreeEpochNewer(0, 0xffff_ffff), false);
});

test("large worksets use bounded multidimensional indirect records", () => {
  assert.deepEqual(octreeDispatchForCount(23, 4, 3), [3, 2, 1]);
  assert.deepEqual(octreeDispatchForCount(100, 1, 5), [5, 5, 4]);
  assert.throws(() => octreeDispatchForCount(126, 1, 5), /3-D limit/);
});

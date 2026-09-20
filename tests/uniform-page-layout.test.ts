import assert from "node:assert/strict";
import test from "node:test";
import { signedSpatialCoordinateHash } from "../lib/core/signed-spatial-hash";
import { lookupUniformPage, planUniformPages, uniformPageCellAddress,
  UNIFORM_PAGE_MISSING as missing, type UniformPageCoordinate } from "../lib/methods/uniform/uniform-page-layout";

test("negative cells have floor-divided pages and positive local addresses", () => {
  assert.deepEqual(uniformPageCellAddress([-1, -16, -17], 16), {
    coordinate: [-1, -1, -2], local: [15, 0, 15], localIndex: 3855,
  });
});

test("signed page directory resolves colliding keys and caches reciprocal seams", () => {
  const coordinates: UniformPageCoordinate[] = [];
  for (let x = -1000; coordinates.length < 8; x++) {
    if ((signedSpatialCoordinateHash([x, -2, 3]) & 15) === 1) coordinates.push([x, -2, 3]);
  }
  const collision = planUniformPages(16, 8, coordinates);
  coordinates.forEach((q, slot) => assert.equal(lookupUniformPage(collision, q), slot));
  assert.equal(lookupUniformPage(collision, [1000000, -2, 3]), missing);
  const layout = planUniformPages(32, 4, [[-1, 0, 0], [0, 0, 0], [0, -1, 0], [0, 0, -1]]);
  assert.equal(layout.neighbors[1], 1);
  assert.equal(layout.neighbors[6], 0);
  assert.equal(layout.neighbors[8], 2);
  assert.equal(layout.neighbors[10], 3);
  assert.equal(layout.neighbors[7], missing);
});

test("growth preserves slots; retirement cannot reuse an address in the same transition", () => {
  const initial = planUniformPages(16, 2, [[0, 0, 0], [1, 0, 0]]);
  const before = initial.directory.slice();
  assert.throws(() => planUniformPages(16, 2, [[1, 0, 0], [2, 0, 0]], initial), /capacity exhausted/);
  assert.deepEqual(initial.directory, before);
  const grown = planUniformPages(16, 3, [[1, 0, 0], [2, 0, 0]], initial);
  assert.deepEqual([...grown.activeSlots], [1, 2]);
  assert.deepEqual([...grown.releasedSlots], [0]);
  assert.deepEqual([...grown.newSlots], [2]);
  // A consumer must fence users of initial before installing this next plan.
  const reused = planUniformPages(16, 3, [[-1, 0, 0], [2, 0, 0], [1, 0, 0]], grown);
  assert.deepEqual([...reused.activeSlots], [0, 2, 1]);
  assert.equal(reused.neighbors[6], missing);
  assert.equal(reused.neighbors[7], 2);
  assert.equal(reused.generation, 2);
});

test("directory memory follows residency rather than empty world extent", () => {
  const near = planUniformPages(32, 2, [[0, 0, 0], [1, 0, 0]]);
  const far = planUniformPages(32, 2, [[-1000000, 0, 0], [1000000, 0, 0]]);
  assert.equal(near.directory.byteLength, far.directory.byteLength);
  assert.equal(near.neighbors.byteLength, far.neighbors.byteLength);
  assert.ok(far.neighbors.every(v => v === missing));
  assert.throws(() => planUniformPages(16, 2, [[0, 0, 0], [0, 0, 0]]), /Duplicate/);
  assert.throws(() => planUniformPages(16, 1, [[2 ** 31, 0, 0]]), /ABI/);
  assert.throws(() => planUniformPages(32, 2 ** 17, []), /overflow/);
  const empty = planUniformPages(16, 1, []);
  assert.equal(lookupUniformPage(empty, [0, 0, 0]), missing);
});

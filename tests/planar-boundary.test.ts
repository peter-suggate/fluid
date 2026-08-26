import assert from "node:assert/strict";
import test from "node:test";
import {
  intersectPlanarBoundaryPatch,
  packPlanarBoundaryPatches,
  PLANAR_BOUNDARY_PATCH_WORDS,
  type PlanarBoundaryPatch,
} from "../lib/core/planar-boundary";

const floor: PlanarBoundaryPatch = {
  center_m: [0, 0, 0],
  normal: [0, 1, 0],
  tangentU: [1, 0, 0],
  tangentV: [0, 0, 1],
  halfExtentU_m: 2,
  halfExtentV_m: 3,
  halfThickness_m: 0.05,
  materialId: 37,
  ownerId: 19,
};

test("finite planar boundary retains exact slab thickness and face normal", () => {
  const hit = intersectPlanarBoundaryPatch(floor, [0, 1, 0], [0, -1, 0]);
  assert.ok(hit);
  assert.ok(Math.abs(hit.tHit_m - 0.95) < 1e-12);
  assert.ok(Math.abs(hit.tEnter_m - 0.95) < 1e-12);
  assert.ok(Math.abs(hit.tExit_m - 1.05) < 1e-12);
  assert.deepEqual(hit.normal, [0, 1, 0]);
  assert.equal(intersectPlanarBoundaryPatch(floor, [2.1, 1, 0], [0, -1, 0]), null);
});

test("a ray beginning inside the slab resolves the physical exit face", () => {
  const hit = intersectPlanarBoundaryPatch(floor, [0, 0, 0], [0, 1, 0]);
  assert.ok(hit);
  assert.ok(Math.abs(hit.tHit_m - 0.05) < 1e-12);
  assert.equal(hit.tEnter_m, 0);
  assert.ok(Math.abs(hit.tExit_m - 0.05) < 1e-12);
  assert.deepEqual(hit.normal, [0, 1, 0]);
  assert.equal(hit.featureAxis, 2);

  const edge = intersectPlanarBoundaryPatch(floor, [3, 0, 0], [-1, 0, 0]);
  assert.ok(edge);
  assert.equal(edge.tHit_m, 1);
  assert.deepEqual(edge.normal, [1, 0, 0]);
  assert.equal(edge.featureAxis, 0);
});

test("planar boundary packing preserves frame, thickness, and identity", () => {
  const words = packPlanarBoundaryPatches([floor]);
  const floats = new Float32Array(words.buffer);
  assert.equal(words.length, PLANAR_BOUNDARY_PATCH_WORDS);
  assert.ok(Math.abs(floats[3] - 0.05) < 1e-7);
  assert.deepEqual([...floats.slice(4, 7)], [0, 1, 0]);
  assert.equal(words[15], (19 << 16) | 37);
});

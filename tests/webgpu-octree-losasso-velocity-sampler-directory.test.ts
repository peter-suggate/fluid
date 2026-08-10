import assert from "node:assert/strict";
import test from "node:test";

import {
  buildOctreeLosassoAxisFaceDirectory,
  planOctreeLosassoAxisFaceDirectory,
} from "../lib/webgpu-octree-losasso-velocity-sampler";

test("bounded-probe face directory stays at or below one-quarter load", () => {
  // The dam candidate that exposed the old 32-probe cluster contained 3,845
  // faces. Its former 8,192-slot table was 46.9% full; the bounded-probe ABI
  // now plans 16,384 slots and rejects that undersized legacy shape.
  const faces = 3_845;
  const plan = planOctreeLosassoAxisFaceDirectory(faces);
  assert.equal(plan.directoryCapacity, 16_384);
  assert.ok(plan.loadFactor <= 0.25);

  const geometry = new Uint32Array(4 * faces);
  for (let face = 0; face < faces; face += 1) {
    geometry[4 * face] = 0;
    geometry[4 * face + 1] = face;
  }
  assert.throws(() => buildOctreeLosassoAxisFaceDirectory(
    geometry, [faces, 1, 1], 8_192), /load exceeds 0\.25/);
  const directory = buildOctreeLosassoAxisFaceDirectory(
    geometry, [faces, 1, 1], plan.directoryCapacity);
  assert.equal(directory.length, 2 * plan.directoryCapacity);
  assert.equal(directory.filter((_word, index) => index % 2 === 0 && directory[index] !== 0).length,
    faces);
});

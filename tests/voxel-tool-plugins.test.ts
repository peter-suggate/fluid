import assert from "node:assert/strict";
import test from "node:test";
import { interpolateCells, mirrorPatchX, shapePatches } from "../lib/core/voxel-editor/geometry";


test("fast strokes cover negative coordinates continuously and mirroring is involutive", () => {
  const points = interpolateCells([-9, 0, 3], [10, 0, 3]);
  assert.equal(points.length, 20);
  assert.deepEqual(points.map((p) => p[0]), Array.from({ length: 20 }, (_, i) => i - 9));
  assert.throws(() => interpolateCells([0, 0, 0], [10000, 0, 0]));
  const patch = shapePatches([-3, 0, 0], [-1, 1, 1], "fill", "box")[0]!;
  assert.deepEqual(mirrorPatchX(mirrorPatchX(patch, 32), 32), patch);
});


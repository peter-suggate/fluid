import assert from "node:assert/strict";
import test from "node:test";
import { planFineLevelSetDispatch2D } from "../lib/webgpu-fine-levelset-dispatch";

test("fine workloads tile across both WebGPU dispatch dimensions without truncation", () => {
  assert.deepEqual(planFineLevelSetDispatch2D(0, 65_535), { x: 0, y: 0, z: 1, workgroups: 0 });
  assert.deepEqual(planFineLevelSetDispatch2D(65_535, 65_535),
    { x: 65_535, y: 1, z: 1, workgroups: 65_535 });
  assert.deepEqual(planFineLevelSetDispatch2D(213_648, 65_535),
    { x: 65_535, y: 4, z: 1, workgroups: 213_648 });
  assert.throws(() => planFineLevelSetDispatch2D(65_535 ** 2 + 1, 65_535), /exceeds/);
});

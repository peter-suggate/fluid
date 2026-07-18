import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const uniformEulerianSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");

test("octree smooth presentation keeps analytic solids and glass", () => {
  assert.match(rendererSource, /scene\.nominalResolution\.length_m, Math\.min\(bodies\.length, 12\), 0,/,
    "smooth presentation must publish rigid-body count and leave glass enabled");
  assert.match(rendererSource, /if \(sceneHasTerrain\(scene\) && scene\.terrain\)/,
    "terrain must remain part of the analytic scene for octree simulations");
  assert.match(rendererSource, /bodies\.slice\(0, 12\)\.forEach/,
    "rigid bodies must remain in the analytic scene for octree simulations");
  assert.match(rendererSource, /if \(environmentIndex\(\) != 7\) \{/,
    "the tank glass path must be selected from scene semantics, not representation mode");
  assert.doesNotMatch(rendererSource, /voxelSceneActive|voxelScenePipeline|Compiling voxel scene materials/,
    "smooth presentation must not instantiate or encode sparse voxel cubes as production solids");
});

test("raw voxel and brick-grid inspection retain the GPU sparse source", () => {
  assert.match(rendererSource, /this\.voxelDebugPipeline\?\.setSource\(solver\.sparseVoxelRenderSource\)/);
  assert.match(rendererSource, /if \(voxelRenderMode !== "smooth" && this\.voxelDebugDepth\)/);
  assert.match(rendererSource, /mode: voxelRenderMode/);
  assert.match(rendererSource, /colorLoadOp: "clear"/);
  assert.match(uniformEulerianSource, /Publish initial sparse-brick scene/);
  assert.match(uniformEulerianSource, /this\.octreeProjection\.encodeInlineRebuild\(initialSparseScene\)/);
  assert.match(uniformEulerianSource, /this\.octreeProjection\.encodeSparseBrickWorld\(initialSparseScene\)/);
  assert.doesNotMatch(rendererSource, /mode: "smooth", colorTarget/,
    "the renderer must never send smooth mode through the cube inspection pipeline");
});

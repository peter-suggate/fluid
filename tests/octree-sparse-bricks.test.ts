import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  ENVIRONMENT_MAXIMUM_COARSENING_POWER,
  octreeSparseBrickDebugPublicationShader,
  planOctreeBrickCoordinates,
} from "../lib/webgpu-octree-sparse-bricks";

test("octree sparse-brick planning covers the real balanced dam-break lattice", () => {
  const plan = planOctreeBrickCoordinates([61, 46, 41], 8);
  assert.deepEqual(plan.brickDimensions, [8, 6, 6]);
  assert.equal(plan.coordinates.length, 288);
  assert.deepEqual(plan.coordinates[0], { x: 0, y: 0, z: 0 });
  assert.deepEqual(plan.coordinates.at(-1), { x: 7, y: 5, z: 5 });
});

test("octree sparse-brick render publication preserves owners, materials, and GPU-only counts", () => {
  assert.match(octreeSparseBrickDebugPublicationShader, /bodyMaterials\[owner\]/);
  assert.match(octreeSparseBrickDebugPublicationShader, /material != 0u && owner != 0xffffu/);
  assert.match(octreeSparseBrickDebugPublicationShader, /candidate != 0u && candidateOwner != 0xffffu/);
  assert.match(octreeSparseBrickDebugPublicationShader, /let payloadIndex = leaf\.topology\.y \+ localIndex;/);
  assert.match(octreeSparseBrickDebugPublicationShader, /payloadIndex < arrayLength\(&materialOwners\)/);
  assert.doesNotMatch(octreeSparseBrickDebugPublicationShader, /materialOwners\[index\]/);
  assert.match(octreeSparseBrickDebugPublicationShader, /control\[2\]/);
  assert.match(octreeSparseBrickDebugPublicationShader, /select\(0u, 1u, isActive\)/);
  assert.doesNotMatch(octreeSparseBrickDebugPublicationShader, /mapAsync|getMappedRange/);
});

test("raw inspection keeps room shells modelled without letting them hide interior props", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  assert.match(source, /primitive\.tags\.includes\("shell"\) \? 0 : 1/);
});

test("scene environment terminal leaves are two levels deeper than the original coarse representation", () => {
  assert.equal(ENVIRONMENT_MAXIMUM_COARSENING_POWER, 1);
});

test("debug publication retains anisotropic XYZ extents for exact world bounds", () => {
  assert.match(octreeSparseBrickDebugPublicationShader, /f32\(scale\) \* params\.cell\.xyz/);
  assert.match(octreeSparseBrickDebugPublicationShader, /f32\(brickSize \* scale\) \* params\.cell\.xyz/);
  assert.doesNotMatch(octreeSparseBrickDebugPublicationShader, /min\(params\.cell/);
});

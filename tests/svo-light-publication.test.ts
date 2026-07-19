import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { getScenePreset } from "../lib/scenes";
import { SVO_LIGHT_RECORD_STRIDE_BYTES } from "../lib/svo-light-abi";
import {
  buildOctreeSvoLightPublication,
  OCTREE_SVO_LIGHT_REVISION,
} from "../lib/webgpu-octree-sparse-bricks";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";

test("light publication is optional and preserves legacy producer compatibility", () => {
  const binding = { buffer: {} as GPUBuffer };
  const legacy = {
    voxelRecords: binding,
    voxelCount: binding,
    brickRecords: binding,
    brickCount: binding,
    materials: binding,
    voxelCapacity: 64,
    brickCapacity: 8,
    materialCount: 20,
    revision: 3,
  } satisfies SparseVoxelRenderSource;
  assert.equal(legacy.materials, binding);
  const modern = {
    ...legacy,
    lights: { binding, count: 4, strideBytes: SVO_LIGHT_RECORD_STRIDE_BYTES, revision: 7 },
  } satisfies SparseVoxelRenderSource;
  assert.equal(modern.lights.binding, binding);
  assert.equal(modern.materials, legacy.materials);
});

test("night-lab and conservatory publish their authored emissive fixtures", () => {
  for (const environmentId of ["night-lab", "conservatory"] as const) {
    const scene = getScenePreset("sphere-jet").create();
    scene.environment = environmentId;
    const publication = buildOctreeSvoLightPublication(scene, { revision: 11 });
    assert.equal(publication.records[0].sourceKey, "authored/directional");
    assert.ok(publication.records.length > 1, `${environmentId} must publish finite fixtures`);
    assert.ok(publication.records.slice(1).every(({ sourceKey }) => sourceKey.startsWith(`${environmentId}/`)));
    assert.ok(publication.records.slice(1).every(({ ownerId }) => ownerId >= scene.rigidBodies.length));
    assert.equal(publication.packedRecords.byteLength, publication.count * SVO_LIGHT_RECORD_STRIDE_BYTES);
    assert.equal(publication.strideBytes, SVO_LIGHT_RECORD_STRIDE_BYTES);
  }
});

test("scene light content and publication revisions remain deterministic and explicit", () => {
  const scene = getScenePreset("sphere-jet").create();
  scene.environment = "night-lab";
  const first = buildOctreeSvoLightPublication(scene, { revision: 19 });
  const second = buildOctreeSvoLightPublication(scene, { revision: 19 });
  assert.deepEqual(second.records, first.records);
  assert.deepEqual(second.packedRecords, first.packedRecords);
  assert.strictEqual(second, first);
  assert.strictEqual(second.packedRecords, first.packedRecords);
  assert.equal(second.cacheKey, first.cacheKey);
  assert.equal(first.revision, 19);
  assert.ok(first.records.every(({ revision }) => revision === first.revision));
  const advanced = buildOctreeSvoLightPublication(scene, { revision: 20 });
  assert.notDeepEqual(advanced.packedRecords, first.packedRecords);
  assert.notEqual(advanced.cacheKey, first.cacheKey);
  assert.equal(OCTREE_SVO_LIGHT_REVISION, 1);
});

test("bounded producer publication reports every omitted lower-priority fixture", () => {
  const scene = getScenePreset("sphere-jet").create();
  scene.environment = "night-lab";
  const full = buildOctreeSvoLightPublication(scene);
  const bounded = buildOctreeSvoLightPublication(scene, { maximumRecords: 2 });
  assert.equal(bounded.count, 2);
  assert.equal(bounded.records[0].kind, "directional");
  assert.ok(bounded.omittedFixtureKeys.length > 0);
  assert.equal(
    bounded.omittedFixtureKeys.length,
    full.records.length - bounded.records.length,
    "every unselected fixture remains visible to capacity diagnostics",
  );
  assert.deepEqual(bounded.omittedFixtureKeys, [...bounded.omittedFixtureKeys].sort());
});

test("octree world owns, publishes, accounts, and destroys its light buffer once", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  assert.match(source, /private readonly lightBuffer: GPUBuffer/);
  assert.match(source, /this\.lightBuffer = storageBuffer\(/);
  assert.match(source, /binding: \{ buffer: this\.lightBuffer, size: lights\.packedRecords\.byteLength \}/);
  assert.match(source, /count: lights\.count/);
  assert.match(source, /strideBytes: lights\.strideBytes/);
  assert.match(source, /revision: lights\.revision/);
  assert.match(source, /\+ this\.pbrMaterialBuffer\.size \+ this\.lightBuffer\.size/);
  const destroy = source.slice(source.indexOf("  destroy(): void {"));
  assert.match(destroy, /this\.pbrMaterialBuffer, this\.lightBuffer, this\.environmentLightingBuffer/);
  assert.match(destroy, /\.\.\.\(this\.inspection\?\.buffers \?\? \[\]\)/);
  assert.equal((destroy.match(/this\.lightBuffer/g) ?? []).length, 1);
});

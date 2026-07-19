import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { environmentIds } from "../lib/environments";
import { getScenePreset } from "../lib/scenes";
import { SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES } from "../lib/svo-environment-lighting";
import {
  buildOctreeSvoEnvironmentLightingPublication,
  OCTREE_SVO_ENVIRONMENT_LIGHTING_REVISION,
} from "../lib/webgpu-octree-sparse-bricks";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";

test("environment lighting is optional and preserves legacy producer compatibility", () => {
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
    environmentLighting: {
      binding,
      count: 1,
      strideBytes: SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
      revision: 2,
      cacheKey: "svo-environment-lighting-v1:default:test",
    },
  } satisfies SparseVoxelRenderSource;
  assert.equal(modern.environmentLighting.binding, binding);
  assert.equal(modern.materials, legacy.materials);
});

test("producer selects exactly one stable record for every authored environment", () => {
  const scene = getScenePreset("sphere-jet").create();
  const keys = new Set<string>();
  for (const environmentId of environmentIds) {
    scene.environment = environmentId;
    const publication = buildOctreeSvoEnvironmentLightingPublication(scene, 13);
    assert.equal(publication.record.environmentId, environmentId);
    assert.equal(publication.count, 1);
    assert.equal(publication.strideBytes, SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES);
    assert.equal(publication.packedRecords.byteLength, SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES);
    assert.equal(publication.revision, 13);
    assert.match(publication.cacheKey, new RegExp(`^svo-environment-lighting-v1:${environmentId}:`));
    assert.ok(!keys.has(publication.cacheKey), `${environmentId} must have distinct content identity`);
    keys.add(publication.cacheKey);
  }
});

test("default selection, revisions, and cache keys are deterministic", () => {
  const first = buildOctreeSvoEnvironmentLightingPublication({ environment: undefined });
  const second = buildOctreeSvoEnvironmentLightingPublication({ environment: undefined });
  assert.equal(first.record.environmentId, "default");
  assert.equal(first.revision, OCTREE_SVO_ENVIRONMENT_LIGHTING_REVISION);
  assert.equal(second.cacheKey, first.cacheKey);
  assert.deepEqual(second.packedRecords, first.packedRecords);
  const advanced = buildOctreeSvoEnvironmentLightingPublication({ environment: "default" }, first.revision + 1);
  assert.notEqual(advanced.cacheKey, first.cacheKey);
});

test("octree world owns, publishes, accounts, and destroys environment lighting once", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
  assert.match(source, /private readonly environmentLightingBuffer: GPUBuffer/);
  assert.match(source, /this\.environmentLightingBuffer = storageBuffer\(/);
  assert.match(source, /binding: \{ buffer: this\.environmentLightingBuffer, size: environmentLighting\.packedRecords\.byteLength \}/);
  assert.match(source, /count: environmentLighting\.count/);
  assert.match(source, /strideBytes: environmentLighting\.strideBytes/);
  assert.match(source, /revision: environmentLighting\.revision/);
  assert.match(source, /cacheKey: environmentLighting\.cacheKey/);
  assert.match(source, /\+ this\.lightBuffer\.size \+ this\.environmentLightingBuffer\.size/);
  const destroy = source.slice(source.indexOf("  destroy(): void {"));
  assert.match(destroy, /this\.lightBuffer, this\.environmentLightingBuffer, this\.structuralPublicationState/);
  assert.match(destroy, /\.\.\.\(this\.inspection\?\.buffers \?\? \[\]\)/);
  assert.equal((destroy.match(/this\.environmentLightingBuffer/g) ?? []).length, 1);
});

import assert from "node:assert/strict";
import test from "node:test";

import { getScenePreset } from "../lib/scenes";
import { buildSvoEnvironmentLighting, SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES } from "../lib/svo-environment-lighting";
import { buildSvoSceneLights, SVO_LIGHT_RECORD_STRIDE_BYTES } from "../lib/svo-light-abi";
import {
  buildSparseVoxelDrySceneLightingMirrors,
  canConsumeSparseVoxelLighting,
  packSparseVoxelDrySceneLightingArena,
  SVO_DRY_SCENE_AREA_LIGHT_SAMPLES,
  SVO_DRY_SCENE_BINDING_CONTRACT,
  SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT,
  SVO_DRY_SCENE_MAX_SHADED_LIGHTS,
  svoDrySceneShader,
  type SparseVoxelDrySceneData,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelSceneRenderSource } from "../lib/webgpu-voxel-debug";

function fixture() {
  const sceneDescription = getScenePreset("sphere-jet").create();
  sceneDescription.environment = "night-lab";
  const lights = buildSvoSceneLights(sceneDescription, { revision: 7 });
  const environment = buildSvoEnvironmentLighting("night-lab", 9);
  const buffer = {} as GPUBuffer;
  const source = {
    materialCount: 20,
    revision: 1,
    lights: { binding: { buffer, size: lights.packedRecords.byteLength }, count: lights.records.length, strideBytes: SVO_LIGHT_RECORD_STRIDE_BYTES, revision: 7 },
    environmentLighting: { binding: { buffer, size: environment.packedRecord.byteLength }, count: 1, strideBytes: SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES, revision: 9, cacheKey: environment.cacheKey },
  } satisfies SparseVoxelSceneRenderSource;
  const scene = {
    primitiveRecords: new Uint32Array(16), ownerBase: 0,
    lightRecords: lights.packedRecords, lightRevision: lights.revision,
    environmentLightingRecord: environment.packedRecord,
    environmentLightingCacheKey: environment.cacheKey,
  } satisfies SparseVoxelDrySceneData;
  return { source, scene, lights, environment };
}

test("published light/environment metadata gates the CPU mirror and malformed identity fails closed", () => {
  const { source, scene } = fixture();
  assert.equal(canConsumeSparseVoxelLighting(source, scene), true);
  assert.equal(canConsumeSparseVoxelLighting({ ...source, lights: { ...source.lights, strideBytes: 96 } }, scene), false);
  assert.equal(canConsumeSparseVoxelLighting({ ...source, lights: { ...source.lights, count: source.lights.count + 1 } }, scene), false);
  assert.equal(canConsumeSparseVoxelLighting({ ...source, environmentLighting: { ...source.environmentLighting, count: 2 } }, scene), false);
  assert.equal(canConsumeSparseVoxelLighting(source, { ...scene, lightRevision: 8 }), false);
  assert.equal(canConsumeSparseVoxelLighting(source, { ...scene, environmentLightingCacheKey: "stale" }), false);
  const wrongRecord = Uint32Array.from(scene.lightRecords);
  wrongRecord[27] = 99;
  assert.equal(canConsumeSparseVoxelLighting(source, { ...scene, lightRecords: wrongRecord }), false);
});

test("canonical scene mirrors follow authoritative publication revisions and malformed metadata never throws", () => {
  const { source } = fixture();
  const sceneDescription = getScenePreset("sphere-jet").create();
  sceneDescription.environment = "night-lab";
  const mirrors = buildSparseVoxelDrySceneLightingMirrors(sceneDescription, source);
  assert.ok(mirrors);
  assert.equal(mirrors.lightRevision, source.lights.revision);
  assert.equal(mirrors.environmentLightingCacheKey, source.environmentLighting.cacheKey);
  assert.equal(buildSparseVoxelDrySceneLightingMirrors(sceneDescription, { ...source, lights: { ...source.lights, count: 0 } }), undefined);
  assert.equal(buildSparseVoxelDrySceneLightingMirrors(sceneDescription, { ...source, environmentLighting: { ...source.environmentLighting, cacheKey: "wrong" } }), undefined);
});

test("one uniform arena preserves exact published records without adding a storage binding", () => {
  const { source, scene } = fixture();
  const arena = packSparseVoxelDrySceneLightingArena(source, scene)!;
  assert.equal(arena.byteLength, SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.sizeBytes);
  assert.deepEqual([...arena.slice(0, 4)], [source.lights.count, source.lights.revision, source.environmentLighting.revision, 1]);
  assert.deepEqual(
    arena.slice(SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.lightWordOffset, SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.lightWordOffset + scene.lightRecords.length),
    scene.lightRecords,
  );
  assert.deepEqual(arena.slice(SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.environmentWordOffset), scene.environmentLightingRecord);
  assert.match(svoDrySceneShader, /@group\(0\) @binding\(13\) var<uniform> dryLighting:DryLightingArena/);
  assert.deepEqual(SVO_DRY_SCENE_BINDING_CONTRACT.find(({ binding }) => binding === 13), { binding: 13, type: "uniform" });
  assert.equal((svoDrySceneShader.match(/var<storage,\s*read>/g) ?? []).length, 10);
});

test("directional, point, sphere, and rectangle lighting share bounded stable visibility work", () => {
  assert.equal(SVO_DRY_SCENE_MAX_SHADED_LIGHTS, 8);
  assert.equal(SVO_DRY_SCENE_AREA_LIGHT_SAMPLES, 2);
  assert.match(svoDrySceneShader, /SVO_LIGHT_DIRECTIONAL/);
  assert.match(svoDrySceneShader, /SVO_LIGHT_SPHERE_AREA/);
  assert.match(svoDrySceneShader, /SVO_LIGHT_RECTANGLE_AREA/);
  assert.match(svoDrySceneShader, /sampleBudget>=8u/);
  assert.match(svoDrySceneShader, /sampleIndex<2u/);
  assert.match(svoDrySceneShader, /let emitterFacing=max\(dot\(normalize\(light\.directionCone\.xyz\),-towardLight\),0\.0\)/,
    "one-sided rectangle emitters cannot leak light through their back face");
  assert.match(svoDrySceneShader, /svoEnvironmentDiffuseIrradiance\(dryLighting\.environment,hit\.normal\)/);
  assert.match(svoDrySceneShader, /dryEnvironment\(reflected,surface\.roughness\)\*fresnel/);
  assert.match(svoDrySceneShader, /dryEnvironment\(reflect\(rd,glass\.hit\.geometricNormal\),\.04\)/);
});

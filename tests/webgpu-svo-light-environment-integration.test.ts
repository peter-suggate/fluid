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
import { svoDrySceneFixture } from "./svo-dry-scene-test-fixture";

function fixture() {
  const sceneDescription = getScenePreset("sphere-jet").create();
  sceneDescription.environment = "night-lab";
  const lights = buildSvoSceneLights(sceneDescription, { revision: 7 });
  const environment = buildSvoEnvironmentLighting("night-lab", 7);
  const buffer = {} as GPUBuffer;
  const source = {
    materialCount: 20,
    revision: 1,
    lights: { binding: { buffer, size: lights.packedRecords.byteLength }, count: lights.records.length, strideBytes: SVO_LIGHT_RECORD_STRIDE_BYTES, revision: 7 },
    environmentLighting: { binding: { buffer, size: environment.packedRecord.byteLength }, count: 1, strideBytes: SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES, revision: 9, cacheKey: environment.cacheKey },
  } satisfies SparseVoxelSceneRenderSource;
  const scene = {
    ...svoDrySceneFixture,
    lightRecords: lights.packedRecords, lightRevision: lights.revision,
    environmentLightingRecord: environment.packedRecord,
    environmentLightingCacheKey: environment.cacheKey,
  } satisfies SparseVoxelDrySceneData;
  return { source, scene, lights, environment };
}

test("published light/environment metadata gates the CPU mirror and malformed identity fails closed", () => {
  const { scene } = fixture();
  assert.equal(canConsumeSparseVoxelLighting(scene), true);
  assert.equal(canConsumeSparseVoxelLighting({ ...scene, lightRevision: 8 }), false);
  assert.equal(canConsumeSparseVoxelLighting({ ...scene, environmentLightingRecord: new Uint32Array(1) }), false);
  const wrongRecord = Uint32Array.from(scene.lightRecords);
  wrongRecord[27] = 99;
  assert.equal(canConsumeSparseVoxelLighting({ ...scene, lightRecords: wrongRecord }), false);
});

test("canonical scene mirrors follow authoritative publication revisions and malformed metadata never throws", () => {
  const sceneDescription = getScenePreset("sphere-jet").create();
  sceneDescription.environment = "night-lab";
  const mirrors = buildSparseVoxelDrySceneLightingMirrors(sceneDescription, 7);
  assert.ok(mirrors);
  assert.equal(mirrors.lightRevision, 7);
  assert.equal(buildSparseVoxelDrySceneLightingMirrors(sceneDescription, 0), undefined);
});

test("a light-only scene edit republishes current radiance into the fixed arena", () => {
  const sceneDescription = getScenePreset("sphere-jet").create();
  const first = buildSparseVoxelDrySceneLightingMirrors(sceneDescription, 7)!;
  sceneDescription.lighting = { ...sceneDescription.lighting,
    directional: { ...sceneDescription.lighting?.directional, intensity: 2.5 } };
  const changed = buildSparseVoxelDrySceneLightingMirrors(sceneDescription, 8)!;
  const arena = packSparseVoxelDrySceneLightingArena({
    ...svoDrySceneFixture,
    ...changed,
  })!;
  const floats = new Float32Array(arena.buffer, arena.byteOffset, arena.length);
  const directional = SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.lightWordOffset;

  assert.notDeepEqual(changed.lightRecords, first.lightRecords);
  assert.equal(arena[SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.metadataWordOffset + 1], 8);
  assert.equal(arena[directional + 27], 8, "the light record and arena publish one complete revision");
  assert.equal(floats[directional + 11], 2.5, "the next frame consumes the edited analytic intensity");
});

test("one uniform arena preserves exact published records without adding a storage binding", () => {
  const { source, scene } = fixture();
  const arena = packSparseVoxelDrySceneLightingArena(scene)!;
  assert.equal(arena.byteLength, SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.sizeBytes);
  assert.deepEqual([...arena.slice(0, 4)], [source.lights.count, scene.lightRevision, scene.lightRevision, 1]);
  assert.deepEqual(
    arena.slice(SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.lightWordOffset, SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.lightWordOffset + scene.lightRecords.length),
    scene.lightRecords,
  );
  assert.deepEqual(arena.slice(SVO_DRY_SCENE_LIGHTING_ARENA_LAYOUT.environmentWordOffset), scene.environmentLightingRecord);
  assert.match(svoDrySceneShader, /@group\(0\) @binding\(13\) var<uniform> dryLighting:DryLightingArena/);
  assert.deepEqual(SVO_DRY_SCENE_BINDING_CONTRACT.find(({ binding }) => binding === 13), { binding: 13, type: "uniform" });
  // The dry pass carries ten read-only storage bindings, past the WebGPU
  // default of eight. That is deliberate -- `requiredFluidDeviceLimits`
  // requests the adapter's advertised value for exactly this layout -- so the
  // invariant worth holding is that the shader and the published binding
  // contract agree, not a bare literal that drifts the moment one of them
  // gains a binding the other does not.
  assert.equal((svoDrySceneShader.match(/var<storage,\s*read>/g) ?? []).length,
    SVO_DRY_SCENE_BINDING_CONTRACT.filter(({ type }) => type === "read-only-storage").length,
    "every dry-pass storage binding must be declared in the shader and the contract");
  // Back to four: the scene-geometry lane was bound so the primary could
  // central-difference it for a shading normal, and the normal is baked into the
  // voxel now. The comment above is why this literal is worth keeping anyway.
  assert.equal(SVO_DRY_SCENE_BINDING_CONTRACT
    .filter(({ type }) => type === "read-only-storage").length, 4);
});

test("directional, point, sphere, and rectangle lighting share bounded stable visibility work", () => {
  assert.equal(SVO_DRY_SCENE_MAX_SHADED_LIGHTS, 8);
  assert.equal(SVO_DRY_SCENE_AREA_LIGHT_SAMPLES, 2);
  assert.match(svoDrySceneShader, /SVO_LIGHT_DIRECTIONAL/);
  assert.match(svoDrySceneShader, /SVO_LIGHT_SPHERE_AREA/);
  assert.match(svoDrySceneShader, /SVO_LIGHT_RECTANGLE_AREA/);
  assert.match(svoDrySceneShader, /sampleBudget>=dry\.tuningCounts0\.z/,
    "the live bounded light budget gates both light and sample iteration");
  assert.match(svoDrySceneShader, /sampleIndex<2u/);
  assert.match(svoDrySceneShader, /let emitterFacing=max\(dot\(normalize\(light\.directionCone\.xyz\),-towardLight\),0\.0\)/,
    "one-sided rectangle emitters cannot leak light through their back face");
  assert.match(svoDrySceneShader, /let visibilityDistance=select\(distance,max\(0\.0,distance-light\.shape\.x\),light\.identity\.x==SVO_LIGHT_POINT\)/,
    "point attenuation stays center-based while the shadow endpoint stops at the emissive surface");
  assert.match(svoDrySceneShader, /svoEnvironmentDiffuseIrradiance\(dryLighting\.environment,hit\.normal\)/);
  assert.match(svoDrySceneShader, /dryEnvironment\(reflected,surface\.roughness\)\*fresnel/);
  assert.match(svoDrySceneShader, /dryEnvironment\(reflect\(rd,glass\.hit\.geometricNormal\),\.04\)/);
});

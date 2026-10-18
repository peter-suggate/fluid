import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, parseScene, serializeScene, validateScene } from "../lib/core/model";
import { svoRenderScenePublication } from "../lib/svo/features/scene-publication/svo-render-scene-publication";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo/pipeline/svo-render-options";
import { createHeroGardenBlenderScene } from "../lib/core/hero-garden-blender-scene";

test("scene raster AO contrast survives serialization and rejects invalid values", () => {
  const scene = cloneScene(defaultScene);
  scene.lighting = { rasterAoContrast: 2 };
  assert.deepEqual(validateScene(scene), []);
  assert.equal(parseScene(serializeScene(scene)).lighting?.rasterAoContrast, 2);
  for (const value of [0, 4.1, NaN, Infinity]) {
    scene.lighting.rasterAoContrast = value;
    assert.ok(validateScene(scene).some(error => error.includes("raster AO contrast")));
  }
});

test("raster AO contrast edits publish appearance without changing geometry or lights", () => {
  const scene = cloneScene(defaultScene);
  const first = svoRenderScenePublication(scene, 1, undefined)!;
  assert.ok(first);
  const changed = { ...scene, lighting: { ...scene.lighting, rasterAoContrast: 2 } };
  const next = svoRenderScenePublication(changed, 1, first)!;
  assert.equal(next.change, "appearance");
  assert.equal(next.data.rasterAoContrast, 2);
  assert.equal(next.data.voxelGeometryRevision, first.data.voxelGeometryRevision);
  assert.equal(next.data.primitiveRecords, first.data.primitiveRecords);
  assert.equal(next.data.lightRecords, first.data.lightRecords);
  assert.equal(next.data.environmentLightingRecord, first.data.environmentLightingRecord);
  const restored = svoRenderScenePublication(scene, 1, next)!;
  assert.equal(restored.change, "appearance");
  assert.equal(restored.data.rasterAoContrast, undefined);
  assert.equal(svoRenderScenePublication(scene, 1, restored)!.change, "none");
});

test("Blender garden strengthens AO while retaining raster primary and raster-AO lighting", () => {
  assert.equal(DEFAULT_SVO_LIGHTING_OPTIONS.primaryTraversal, "mesh");
  assert.equal(DEFAULT_SVO_LIGHTING_OPTIONS.coneTracingMode, "raster-ao");
  assert.equal(DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled, true);
  assert.equal(DEFAULT_SVO_LIGHTING_OPTIONS.globalIlluminationEnabled, false);
  assert.equal(createHeroGardenBlenderScene().lighting?.rasterAoContrast, 2);
});

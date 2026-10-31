import assert from "node:assert/strict";
import test from "node:test";
import { createAtlanticLighthouseScene } from "../lib/core/atlantic-lighthouse-scene";
import type { SparseVoxelSceneRenderSource } from "../lib/core/webgpu-voxel-debug";
import { buildSvoDrySceneAssembly } from "../tools/svo-dry-frame-harness";
import { registerSceneMeshAssets } from "../tools/blender/register-assets";
import { SVO_MATERIAL_FLAGS, SVO_MATERIAL_RECORD_WORDS } from "../lib/svo/contracts/svo-material-abi";
import { meshSurfaceTextureAtlas } from "../lib/core/imported-assets/surface-texture-atlas";

test("headless lighthouse captures publish the actual cuts, textures and interior palettes", async () => {
  const scene = createAtlanticLighthouseScene();
  await registerSceneMeshAssets(scene);
  scene.solidVoxels.push({ operation: "clear", kind: "ellipsoid", affectsScenery: true,
    minimum: [117, 31, 95], maximumExclusive: [127, 41, 105] });
  const { drySceneData, scenePrimitives } = buildSvoDrySceneAssembly(scene, {} as SparseVoxelSceneRenderSource);
  assert.equal(drySceneData.sculptCuts?.[0], 1);
  assert.ok(drySceneData.sculptedPrimitiveIndices!.length > 0);
  const expectedAtlas = meshSurfaceTextureAtlas(scenePrimitives.fieldPrograms.flatMap(p => p.sampledVolume ? [p.sampledVolume.asset] : []));
  assert.deepEqual(drySceneData.surfaceTextureAtlas, expectedAtlas);
  assert.ok(drySceneData.surfaceTextureAtlas!.layers > 0);
  const table = drySceneData.materialRecords;
  let palettes = 0;
  for (let material = 1; material < 8192; material++) {
    const at = material * SVO_MATERIAL_RECORD_WORDS;
    if (!(table[at + 23]! & SVO_MATERIAL_FLAGS.sampledVolume)) continue;
    const interior = (material + 8192) * SVO_MATERIAL_RECORD_WORDS;
    assert.deepEqual(table.subarray(at, at + 20), table.subarray(interior, interior + 20));
    assert.equal(table[interior + 23]! & SVO_MATERIAL_FLAGS.sampledVolume, 0);
    palettes++;
  }
  assert.ok(palettes > 0);
});

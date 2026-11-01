import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { createAtlanticLighthouseScene, atlanticLighthouseTerrainHeight } from "../lib/core/atlantic-lighthouse-scene";
import { validateScene } from "../lib/core/model";
import { ProceduralSolidField } from "../lib/core/procedural-solid";
import { meshAssetHeader, sceneMeshAssetIds } from "../lib/core/imported-assets/asset-library";
import { decodeMeshVolume, sampleMeshVolumeDistance } from "../lib/core/imported-assets/mesh-volume";

test("Atlantic lighthouse keeps Blender static scenery and live fluid separate", () => {
  const scene = createAtlanticLighthouseScene();
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(sceneMeshAssetIds(scene), ["atlantic-lighthouse-rock", "atlantic-lighthouse-tower"]);
  assert.equal(scene.systems?.fluid, true);
  assert.equal(scene.scenery?.terrainSurface, "scenery", "Imported basalt owns the visible cliff, not the forest material bands");
  assert.equal(scene.fluid.initialLiquidVolumes?.length, 2);
  assert.equal(scene.container.width_m, 48, "Keep the downstream wall outside the photographed impact region");
  assert.equal(scene.fluid.initialLiquidVolumes?.[1]?.shape, "cylinder");
  assert.equal(scene.voxelDomain?.finestCellSize_m, .25, "Dry detail must not raise the fluid resolution");
  assert.ok(meshAssetHeader("atlantic-lighthouse-rock").cellSize <= .08,
    "Rock fractures must survive as finer SVO geometry, independently of the fluid grid");
  assert.ok(meshAssetHeader("atlantic-lighthouse-tower").cellSize <= .025,
    "Window reveals and gallery supports require centimetre-scale sampled geometry");
  for (const id of sceneMeshAssetIds(scene)) {
    assert.ok(meshAssetHeader(id).surfaceMapping, `${id} must retain its portable surface textures`);
  }
  const tower = scene.scenery!.nodes.find(node => node.id === "atlantic/atlantic-lighthouse-tower")!;
  const header = meshAssetHeader("atlantic-lighthouse-tower");
  const offset = header.origin.map((v, axis) => v + header.dimensions[axis]! * header.cellSize / 2);
  assert.ok(Math.abs(tower.place!.position!.y - offset[1]! - 5.8) < 1e-12);
  assert.ok(header.origin[1] + header.dimensions[1] * header.cellSize + 5.8 < scene.container.height_m);
});

test("the exported angular rock surface agrees with the physical collision heightfield", () => {
  const volume = decodeMeshVolume(gunzipSync(readFileSync(new URL(
    "../public/assets/atlantic-lighthouse/atlantic-lighthouse-rock.fvx.gz", import.meta.url))));
  for (const [x, z] of [[2.5, -2], [4.5, 4], [7, 0], [9, 3], [11, -2]] as const) {
    const height = atlanticLighthouseTerrainHeight(x, z);
    assert.ok(height > 2);
    assert.ok(Math.abs(sampleMeshVolumeDistance(volume, [x - 7, height, z])) < .06,
      `Authored basalt must meet the real fluid collider at ${x}, ${z}`);
    assert.ok(sampleMeshVolumeDistance(volume, [x - 7, height - .25, z]) < 0);
  }
});

test("the island has a broad summit and descending tidal shoulders", () => {
  for (const [x, z] of [[5, 0], [7, 0], [8, 1]] as const) {
    assert.ok(atlanticLighthouseTerrainHeight(x, z) > 5.2, "The tower must sit on a broad rock platform");
  }
  assert.ok(atlanticLighthouseTerrainHeight(0, 0) < 3.2, "The seaward shoulder slopes down instead of forming a high perimeter wall");
  const outcrop = atlanticLighthouseTerrainHeight(16, 6);
  assert.ok(outcrop > 1.6 && outcrop < 3, "Low outlying slabs break the shoreline silhouette");
});

test("the visible tower wall is a real collision surface, with its cavity preserved", () => {
  const scene = createAtlanticLighthouseScene();
  const field = new ProceduralSolidField(scene.solidBoundary!, []);
  const sample = (x: number) => field.sample([(x + 24) / .25, 7 / .25, 24 / .25])![3];
  // Granite is a closed masonry shell: water cannot pass through its west/east
  // wall, but the bake must not silently replace the whole tower by a box.
  assert.ok(sample(8.5) < 0);
  assert.ok(sample(7) > 0);
  assert.ok(sample(9) > 0);
  assert.deepEqual(scene.solidBoundary!.dimensions, [192, 96, 192]);
});

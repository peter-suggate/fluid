import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { createForestWaterfallScene } from "../lib/core/forest-waterfall-scene";
import { meshAssetHeader, meshAssetUrl, sceneMeshAssetIds } from "../lib/core/imported-assets/asset-library";
import { decodeMeshVolume } from "../lib/core/imported-assets/mesh-volume";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/core/voxel-environments";
import instances from "../lib/core/imported-assets/forest-waterfall-instances.json";

test("live waterfall imports its complete static Blender set without decorative liquid", () => {
  const scene = createForestWaterfallScene();
  const assets = sceneMeshAssetIds(scene);
  const manifest = JSON.parse(readFileSync(new URL("../public/assets/forest-waterfall/manifest.json", import.meta.url), "utf8"));
  assert.deepEqual([...assets].sort(), manifest.assets.map((asset: { id: string }) => asset.id).sort());
  assert.equal(scene.scenery!.nodes.length, instances.length + 1);
  assert.ok(instances.length > 2000, "the full planted environment is used by the standard scene");
  assert.ok(scene.fluid.inflow && scene.fluid.initialLiquidVolumes?.length);
  for (const instance of instances) {
    assert.doesNotMatch(instance.sourceName, /^(Pool \||Cascade \||Spray \||Foam \||Splash \||Ravine air)/);
    assert.doesNotMatch(instance.asset, /water|foam|spray|mist/);
  }
  const primitives = environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, "garden"));
  assert.equal(primitives.filter(primitive => primitive.kind === "field-program").length, instances.length);
  for (const primitive of primitives) {
    assert.ok(Object.values(primitive.aabb_m.min).every(Number.isFinite));
    assert.ok(Object.values(primitive.aabb_m.max).every(Number.isFinite));
  }
});

test("every forest runtime asset has a valid published mesh and signed, consistent native volume", () => {
  for (const id of sceneMeshAssetIds(createForestWaterfallScene())) {
    const url = meshAssetUrl(id);
    const compressed = readFileSync(new URL(`../public${url}`, import.meta.url));
    const volume = decodeMeshVolume(gunzipSync(compressed));
    assert.deepEqual(volume.header, meshAssetHeader(id));
    assert.ok([...volume.bricks.values()].some(brick => brick.distance.some(value => value < 0)), `${id}: solid interior`);
    assert.ok([...volume.bricks.values()].some(brick => brick.distance.some(value => value > 0)), `${id}: air exterior`);
    const glb = readFileSync(new URL(`../public/assets/forest-waterfall/${id}.glb`, import.meta.url));
    assert.equal(glb.readUInt32LE(0), 0x46546c67);
    assert.equal(glb.readUInt32LE(8), glb.length);
  }
});

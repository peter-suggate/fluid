import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { validateScene } from "../lib/core/model";
import { validateSceneryGraph, walkSceneryNodes } from "../lib/core/scenery-graph";
import { meshAssetHeader, sceneMeshAssetIds } from "../lib/core/imported-assets/asset-library";
import { buildSvoSceneLights } from "../lib/svo/contracts/svo-light-abi";

for (const id of ["tokyo-rain", "alpine-lake-sunrise", "atlantic-lighthouse"]) {
  test(`${id} is a valid live NB-FLIP landscape with one terrain shell and registered Blender assets`, () => {
    const definition = getSceneDefinition(id), scene = sceneDocument(definition);
    assert.equal(definition.methodProfile?.methodId, "uniform-narrow-band-flip");
    assert.equal(definition.presentationMode, "full-scene");
    assert.deepEqual(validateScene(scene), []);
    assert.ok(scene.scenery, "The interactive preset needs its authored surroundings");
    assert.deepEqual(validateSceneryGraph(scene.scenery), []);
    const shells = [...walkSceneryNodes(scene.scenery.nodes)].filter(({ node }) => node.kind === "terrain-shell");
    assert.equal(shells.length, 1, "The general scenery renderer requires exactly one terrain shell");
    const assets = sceneMeshAssetIds(scene);
    assert.ok(assets.length > 0);
    for (const asset of assets) assert.ok(meshAssetHeader(asset).brickCount > 0, `${asset} has no baked volume`);
    assert.equal(scene.systems?.fluid, true);
    assert.ok(scene.fluid.initialLiquidVolumes?.length);
  });
}

test("Tokyo paper lanterns publish warm light independently of their opaque rib palette", () => {
  const scene = sceneDocument(getSceneDefinition("tokyo-rain"));
  const lights = buildSvoSceneLights(scene).records.filter(light => light.sourceKey.includes("tokyo/lantern/"));
  assert.equal(lights.length, 14);
  for (const light of lights) {
    assert.equal(light.kind, "point");
    assert.equal(light.intensity, 6.4);
    assert.ok(Math.abs(light.colorLinear[0] - 1) < 1e-6);
    assert.ok(Math.abs(light.colorLinear[1] - .27) < 1e-6);
    assert.ok(Math.abs(light.colorLinear[2] - .055) < 1e-6);
  }
});

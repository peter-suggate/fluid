import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { terrainHeightAt } from "../lib/core/terrain";
import { sceneSvoGroundPlane } from "../lib/svo/features/materials/svo-terrain-material";
import { buildSvoRenderTerrainFieldSteps } from "../lib/svo/features/scene-publication/svo-render-solid-field";

test("authored scenery replaces both terrain visuals while preserving the physical heightfield", () => {
  const scene = cloneScene(defaultScene);
  scene.environment = "garden";
  scene.terrain = { baseHeight_m: .3, features: [] };
  scene.scenery = { palettes: {}, terrainSurface: "scenery", nodes: [
    { kind: "terrain-shell", id: "required-shell", materialModel: "forest-terrain" },
  ] };
  const terrain = scene.terrain;
  const before = terrainHeightAt(terrain, 0, 0);
  const field = buildSvoRenderTerrainFieldSteps(scene, [.25, .25, .25], 1);
  assert.deepEqual(field.next(), { done: true, value: undefined }, "No duplicate forest surface is rasterized over authored stone");
  assert.equal(sceneSvoGroundPlane(scene), undefined, "The analytic legacy horizon must also disappear");
  assert.equal(scene.terrain, terrain);
  assert.equal(terrainHeightAt(scene.terrain, 0, 0), before);
  scene.scenery = { ...scene.scenery, terrainSurface: undefined };
  assert.ok(sceneSvoGroundPlane(scene), "Ordinary terrain retains its existing horizon behavior");
});

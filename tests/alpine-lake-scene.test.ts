import assert from "node:assert/strict";
import test from "node:test";
import { parseScene, serializeScene, validateScene } from "../lib/core/model";
import { terrainHeightAt } from "../lib/core/terrain";
import { ALPINE_LAKE_CELL_M, ALPINE_LAKE_WATERLINE_M, createAlpineLakeScene } from "../lib/core/alpine-lake-scene";

test("alpine tarn closes its shoreline inside the solver's container and resolves its deep water", () => {
  const scene = createAlpineLakeScene(), waterline = ALPINE_LAKE_WATERLINE_M;
  assert.ok(waterline - terrainHeightAt(scene.terrain, 0, 0) >= 4 * ALPINE_LAKE_CELL_M - 1e-9);
  for (let i = 0; i <= 128; i++) {
    const x = -32 + i * .5, z = -40 + i * .625;
    for (const [px, pz] of [[x, -40], [x, 40], [-32, z], [32, z]]) {
      assert.ok(terrainHeightAt(scene.terrain, px!, pz!) > waterline + ALPINE_LAKE_CELL_M);
    }
  }
  // Shore depth decreases naturally and no tank wall is used as visible shore.
  assert.ok(terrainHeightAt(scene.terrain, 25, 0) < waterline);
  assert.ok(terrainHeightAt(scene.terrain, 30, 0) > waterline);
});

test("alpine scene survives scene-file validation and serialization with real initial liquid", () => {
  const scene = createAlpineLakeScene();
  assert.deepEqual(validateScene(scene), []);
  const loaded = parseScene(serializeScene(scene));
  assert.deepEqual(loaded.fluid, scene.fluid);
  assert.deepEqual(loaded.terrain, scene.terrain);
  assert.equal(loaded.fluid.initialLiquidVolumes?.length, 1);
  assert.equal(loaded.systems?.fluid, true);
  assert.equal(loaded.fluid.optics?.indexOfRefraction, 1.333);
});

test("interactive alpine environment references baked Blender assets and leaves all water to the solver", () => {
  const scene = createAlpineLakeScene();
  const assets = scene.scenery!.nodes.flatMap(node =>
    node.kind === "field-program" && node.program.sampledVolume ? [node.program.sampledVolume.asset] : []);
  assert.ok(assets.includes("alpine-mountain-ridge"));
  assert.ok(assets.includes("alpine-fir-0"));
  assert.equal(assets.filter(asset => asset === "alpine-shore-scree").length, 1);
  assert.ok(assets.includes("forest-rock-0"));
  assert.ok(assets.every(asset => !/water|liquid|fluid/i.test(asset)));
  assert.ok(scene.scenery!.nodes.some(node => node.kind === "terrain-shell"));
});

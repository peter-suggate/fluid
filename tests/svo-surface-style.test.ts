import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { sceneUsesFlatVoxelNormals } from "../lib/core/model";
import { defaultScenePresetId } from "../lib/core/scenes";
import { parseQueryState, serializeQueryState } from "../lib/core/url-state";
import { createSvoDrySceneFragmentWGSL } from "../lib/svo/webgpu-svo-dry-scene";

test("SVO surface style keeps voxel faces as the default and exposes smooth reconstruction explicitly", () => {
  assert.equal(sceneUsesFlatVoxelNormals({}), true);
  assert.equal(sceneUsesFlatVoxelNormals({ surfaceStyle: "voxel-flat" }), true);
  assert.equal(sceneUsesFlatVoxelNormals({ surfaceStyle: "smooth" }), false);
});

test("smooth SVO surface reconstruction changes the primary hit depth", () => {
  const wgsl = createSvoDrySceneFragmentWGSL();
  assert.match(wgsl, /fn drySmoothVoxelSurfaceT\(/);
  assert.match(wgsl, /let surfaceT=drySmoothVoxelSurfaceT\(/);
  assert.match(wgsl, /return DryHit\(surfaceT,shaded\.normal/);
});

test("smooth SVO surface round-trips through a scene URL", () => {
  const parsed = parseQueryState(
    `?scene=${defaultScenePresetId}&scene.surfaceStyle=${encodeURIComponent(JSON.stringify("smooth"))}`,
  );
  assert.equal(parsed.scene.surfaceStyle, "smooth");

  const serialized = serializeQueryState(
    "",
    { presetId: parsed.presetId, scene: parsed.scene },
    { methodId: parsed.methodId, quality: parsed.quality, overrides: parsed.overrides },
    parsed.ui,
    { view: "studio" },
  );
  assert.equal(new URLSearchParams(serialized).get("scene.surfaceStyle"), JSON.stringify("smooth"));
  assert.equal(parseQueryState(`?${serialized}`).scene.surfaceStyle, "smooth");
});

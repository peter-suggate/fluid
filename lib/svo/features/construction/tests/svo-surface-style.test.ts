import assert from "node:assert/strict";
import test from "node:test";
import "../../../../methods";
import { sceneUsesFlatVoxelNormals } from "../../../../core/model";
import { defaultScenePresetId } from "../../../../core/scenes";
import { parseQueryState, serializeQueryState } from "../../../../core/url-state";
import { createSvoDrySceneFragmentWGSL } from "../../shading/program";

test("SVO surface style defaults to smooth reconstruction and keeps voxel faces selectable", () => {
  assert.equal(sceneUsesFlatVoxelNormals({}), false);
  assert.equal(sceneUsesFlatVoxelNormals({ surfaceStyle: "voxel-flat" }), true);
  assert.equal(sceneUsesFlatVoxelNormals({ surfaceStyle: "smooth" }), false);
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

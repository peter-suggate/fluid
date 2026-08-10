import assert from "node:assert/strict";
import test from "node:test";

import { cloneScene, defaultScene, sceneUsesFlatVoxelNormals, validateScene } from "../lib/model";
import { SVO_DRY_VISIBILITY_FLAGS, svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";
import { buildSvoDrySceneAssembly } from "../tools/svo-dry-frame-harness";

test("voxel-flat is a validated scene surface style", () => {
  const scene = cloneScene(defaultScene);
  assert.equal(sceneUsesFlatVoxelNormals(scene), true,
    "omitted style must resolve to the renderer-wide flat-face default");
  scene.surfaceStyle = "voxel-flat";
  assert.equal(sceneUsesFlatVoxelNormals(scene), true);
  assert.doesNotMatch(validateScene(scene).join("\n"), /surface style/);
  scene.surfaceStyle = "smooth";
  assert.equal(sceneUsesFlatVoxelNormals(scene), false,
    "smooth shading is an explicit opt-out, never the implicit default");
  (scene as unknown as { surfaceStyle: string }).surfaceStyle = "polished";
  assert.match(validateScene(scene).join("\n"), /Unknown scene surface style polished/);
});

test("scene-owned environment radiance validates as linear RGB", () => {
  const scene = cloneScene(defaultScene);
  scene.lighting = { environment: { upperRadianceLinear: [0.76, 0.59, 0.40] } };
  assert.deepEqual(validateScene(scene), []);
  scene.lighting.environment!.upperRadianceLinear = [0.76, -0.1, 0.40];
  assert.match(validateScene(scene).join("\n"), /upper radiance/);
});

test("voxel-flat classifies every final opaque hit to one of six face normals", () => {
  assert.equal(SVO_DRY_VISIBILITY_FLAGS.flatVoxelNormals, 1 << 8);
  assert.match(svoDrySceneShader,
    new RegExp(`materialPublication\\.w&${SVO_DRY_VISIBILITY_FLAGS.flatVoxelNormals}u`));
  assert.match(svoDrySceneShader,
    /return DryShadingNormal\(faceNormal,SVO_FEATURE_SMOOTH\)/);
  assert.match(svoDrySceneShader,
    /fn dryVoxelFaceAxis\(normalIn:vec3f\)->vec3f/);
  assert.match(svoDrySceneShader,
    /if\(magnitude\.x>=magnitude\.y&&magnitude\.x>=magnitude\.z\)/);
  assert.match(svoDrySceneShader,
    /return dryPresentationHit\(hit\)/);
  assert.match(svoDrySceneShader,
    /fn dryVoxelFaceEdgeFactor\(position:vec3f,faceNormal:vec3f,depth_m:f32\)->f32/);
  assert.match(svoDrySceneShader,
    /shaded\*=dryVoxelFaceEdgeFactor\(position,hit\.normal,hit\.t\)/);
});

test("the headless fidelity renderer publishes the same surface style as the product", () => {
  const scene = cloneScene(defaultScene);
  const source = {
    structural: { domain: { cellSize_m: [0.0125, 0.0125, 0.0125] } },
  } as unknown as Parameters<typeof buildSvoDrySceneAssembly>[1];
  delete scene.surfaceStyle;
  assert.equal(buildSvoDrySceneAssembly(scene, source).drySceneData.flatVoxelNormals, true);
  scene.surfaceStyle = "smooth";
  assert.equal(buildSvoDrySceneAssembly(scene, source).drySceneData.flatVoxelNormals, false);
});

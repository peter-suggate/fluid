import assert from "node:assert/strict";
import test from "node:test";
import { svoDiffuseMultiBounceVisibility } from "../lib/svo-contact-visibility";
import { svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

test("diffuse multi-bounce compensation recovers energy according to albedo", () => {
  const visibility = 0.2;
  const result = svoDiffuseMultiBounceVisibility(visibility, [0, 0.5, 1]);
  assert.equal(result[0], visibility, "black surfaces cannot return missing diffuse energy");
  assert.ok(result[1] > result[0]);
  assert.ok(result[2] > result[1]);
  assert.ok(result.every((channel) => channel >= visibility && channel <= 1));
});

test("dry SVO shading applies albedo-aware compensation only to broad GI visibility", () => {
  assert.match(svoDrySceneShader, /fn dryDiffuseMultiBounceVisibility\(visibilityIn:f32,albedoIn:vec3f\)->vec3f/);
  assert.match(svoDrySceneShader, /let diffuseVisibility=dryDiffuseMultiBounceVisibility\(gi\.visibility,diffuseColor\)/);
  assert.match(svoDrySceneShader, /\*contactVisibility\*diffuseVisibility\*diffuseEnvironmentScale\/UNIFIED_PI/);
});

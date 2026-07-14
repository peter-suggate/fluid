import assert from "node:assert/strict";
import test from "node:test";
import { legacyUniformComputeShader } from "../lib/webgpu-eulerian";
import { uniformMethod } from "../lib/methods/uniform";
import { resolveMethodValues } from "../lib/methods/types";

test("uniform exposes both velocity transport schemes", () => {
  const transport = uniformMethod.params.find((spec) => spec.key === "velocityTransport");
  assert.equal(transport?.kind, "select");
  if (transport?.kind !== "select") return;
  assert.deepEqual(transport.options.map((option) => option.value), ["maccormack", "semi-lagrangian"]);
  assert.equal(resolveMethodValues(uniformMethod, "balanced", {}).velocityTransport, "maccormack");
  assert.equal(resolveMethodValues(uniformMethod, "balanced", { velocityTransport: "semi-lagrangian" }).velocityTransport, "semi-lagrangian");
});

test("uniform retains the single-pass semi-Lagrangian kernel", () => {
  assert.match(legacyUniformComputeShader, /fn semiLagrangianAdvection/);
  assert.match(legacyUniformComputeShader, /v=applyVelocityForces\(id,v,dt,h\)/);
});

test("uniform velocity transport uses bounded MacCormack correction", () => {
  assert.match(legacyUniformComputeShader, /fn reverseAdvection/);
  assert.match(legacyUniformComputeShader, /fn boundedMacCormack/);
  assert.match(legacyUniformComputeShader, /fn correctAdvection/);
  assert.match(legacyUniformComputeShader, /predicted\+0\.5\*\(original-reversed\)/);
  assert.match(legacyUniformComputeShader, /return select\(corrected,predicted,corrected<lower\|\|corrected>upper\)/);
});

test("uniform traces extrapolate liquid velocity into air and preserve wall faces", () => {
  assert.match(legacyUniformComputeShader, /return transportVelocity\(clampCell\(p\)\)\[component\]/);
  assert.match(legacyUniformComputeShader, /p\[component\]<0\|\|p\[component\]>=d\[component\]/);
  assert.doesNotMatch(legacyUniformComputeShader, /let oldV=transportVelocity/);
});

test("uniform capillary normals do not classify solid walls as air", () => {
  assert.match(legacyUniformComputeShader, /fn normalVolume/);
  assert.match(legacyUniformComputeShader, /id\.y>=dims\(\)\.y&&params\.boundary\.w>0\.5/);
  assert.match(legacyUniformComputeShader, /return textureLoad\(volumeIn,clampCell\(id\),0\)\.x/);
  assert.match(legacyUniformComputeShader, /normalVolume\(id\+vec3i\(1,0,0\)\)-normalVolume\(id-vec3i\(1,0,0\)\)/);
});

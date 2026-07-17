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
  assert.match(legacyUniformComputeShader, /fn buildTransport/);
  assert.match(legacyUniformComputeShader, /textureStore\(transportOut,padded,vec4f\(transportVelocity\(id\),0\.0\)\)/);
  assert.match(legacyUniformComputeShader, /p\[component\]<0\|\|p\[component\]>=d\[component\]/);
  assert.match(legacyUniformComputeShader, /return textureLoad\(transportIn,clampCell\(p\)\+vec3i\(1\),0\)\[component\]/);
  assert.doesNotMatch(legacyUniformComputeShader, /let oldV=transportVelocity/);
});

test("uniform advection samples precomputed transport with hardware filtering", () => {
  assert.match(legacyUniformComputeShader, /textureSampleLevel\(transportIn,transportSampler,transportCoordinate\(q\),0\.0\)/);
  assert.match(legacyUniformComputeShader, /fn transportVectorEstimate/);
  assert.match(legacyUniformComputeShader, /texture_storage_3d<rgba16float, write>/);
});

test("uniform volume limiter uses per-cell precomputed donor and receiver scales", () => {
  assert.match(legacyUniformComputeShader, /fn buildFluxScales/);
  assert.match(legacyUniformComputeShader, /vec4f\(donorScale\(id,dt\),receiverScale\(id,dt\),0\.0,0\.0\)/);
  assert.match(legacyUniformComputeShader, /fn cellFluxScales/);
});

test("uniform advection skips air far above every nearby occupied column", () => {
  assert.match(legacyUniformComputeShader, /fn buildOccupancy/);
  assert.match(legacyUniformComputeShader, /fn aboveOccupancy/);
  assert.match(legacyUniformComputeShader, /occupancy\+4\.0&&!nearInflow\(id\)/);
  const earlyOuts = legacyUniformComputeShader.match(/if\(aboveOccupancy\(id\)\)/g) ?? [];
  assert.equal(earlyOuts.length, 4);
});

test("uniform capillary normals do not classify solid walls as air", () => {
  assert.match(legacyUniformComputeShader, /fn normalSurfaceOccupancy/);
  assert.match(legacyUniformComputeShader, /id\.y>=dims\(\)\.y&&params\.boundary\.w>0\.5/);
  assert.match(legacyUniformComputeShader, /return surfaceOccupancy\(clampCell\(id\)\)/);
  assert.match(legacyUniformComputeShader, /normalSurfaceOccupancy\(id\+vec3i\(1,0,0\)\)-normalSurfaceOccupancy\(id-vec3i\(1,0,0\)\)/);
});

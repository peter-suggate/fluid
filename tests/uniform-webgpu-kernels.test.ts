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
  const correction = legacyUniformComputeShader.slice(legacyUniformComputeShader.indexOf("fn correctAdvection"), legacyUniformComputeShader.indexOf("@compute @workgroup_size(8,8,1)\nfn buildHeight"));
  assert.match(correction, /let predicted=textureLoad\(predictedVelocityIn,id,0\)\.xyz/);
  assert.equal(correction.match(/textureLoad\(predictedVelocityIn,id,0\)/g)?.length, 1,
    "MacCormack correction should load each vector field once rather than once per component");
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

test("uniform capillary force skips zero-gradient bulk cells and reuses centre curvature", () => {
  const forces = legacyUniformComputeShader.slice(legacyUniformComputeShader.indexOf("fn applyVelocityForces"), legacyUniformComputeShader.indexOf("@compute @workgroup_size(4,4,4)\nfn buildTransport"));
  assert.match(forces, /if\(sigmaOverRho>0\.0\)/);
  assert.match(forces, /if\(dx!=0\.0\|\|dy!=0\.0\|\|dz!=0\.0\)/);
  assert.match(forces, /let centreCurvature=curvatureAt\(id\)/);
  assert.equal(forces.match(/curvatureAt\(id\)/g)?.length, 1,
    "the expensive centre curvature stencil should be shared by all three faces");
});

test("shared force kernel subtracts only a fixed vertical rest-surface reference", () => {
  const occupancy = legacyUniformComputeShader.slice(
    legacyUniformComputeShader.indexOf("fn buildOccupancy"),
    legacyUniformComputeShader.indexOf("fn buildSparseOccupancy"),
  );
  assert.match(occupancy, /if\(!hydrostaticSplit\(\)\)/);
  assert.match(occupancy, /for\(var y:i32=d\.y-1;y>=0;y-=1\)/,
    "the ordinary path should retain its cheap top-down early exit");
  const forces = legacyUniformComputeShader.slice(
    legacyUniformComputeShader.indexOf("fn applyVelocityForces"),
    legacyUniformComputeShader.indexOf("@compute @workgroup_size(4,4,4)\nfn buildTransport"),
  );
  assert.match(legacyUniformComputeShader, /fn fixedHydrostaticPotentialAtY\(yCells:f32\)/);
  assert.match(legacyUniformComputeShader, /params\.inflowTiming\.z-yCells/);
  assert.match(forces, /if\(hydrostaticSplit\(\)\)\{v\.y\+=fixedHydrostaticAcceleration\(id\)\*dt;\}/);
  assert.match(forces, /else\{v\.y\+=params\.cellGravity\.w\*dt;\}/,
    "the disabled path must not eagerly evaluate the connected-column reference");
  assert.doesNotMatch(forces, /v\.[xz]\+=.*fixedHydrostatic/,
    "the fixed datum must not inject a local free-surface-slope acceleration");
});

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const shader = readFileSync(new URL(
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");
const resident = readFileSync(new URL(
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");

test("continuous inflow unions its admitted swept plug into adaptive phi", () => {
  assert.match(shader,
    /liveUnionSample: position =>\s*`levelSetSourceUnionSampleAt\(\$\{position}\)`/);
  const union = shader.slice(shader.indexOf("fn levelSetSourceUnionSampleAt"),
    shader.indexOf("fn injectionCoverageAt"));
  assert.match(union, /gsEnabled\(\)/);
  assert.match(union, /GEOMETRIC_SOURCE_LEDGER\+15u\]>0\.0/,
    "phi must use the finalized admitted-rate authority");
  assert.match(union, /continuousInflowSignedDistanceAt/);
});

test("continuous source phi publication is after rate admission and before pressure", () => {
  const bodyForces = resident.slice(resident.indexOf('stage("body-forces"'),
    resident.indexOf('stage("pressure-topology"'));
  const finalized = bodyForces.indexOf('dispatch("finalizeContinuousGeometricSourceRates"');
  const phi = bodyForces.indexOf('dispatch("lsvUnionPhi"');
  assert.ok(finalized >= 0 && phi > finalized,
    "the admitted rate must be finalized before phi union");
  assert.match(bodyForces.slice(phi), /dispatch\("lsvBeginConstraintProjection"/);
  assert.match(bodyForces.slice(phi), /dispatch\("lsvApplyConstraints"/);
  assert.match(bodyForces.slice(phi), /dispatch\("lsvAdvanceConstraintProjection"/);
});

test("continuous phi union does not mutate volume or the source ledger", () => {
  const union = shader.slice(shader.indexOf("fn levelSetSourceUnionSampleAt"),
    shader.indexOf("fn injectionCoverageAt"));
  assert.doesNotMatch(union, /state\[[^\]]+\]\s*=/);
  assert.doesNotMatch(union, /atomic(?:Store|Add|Exchange)/);
});

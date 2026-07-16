import assert from "node:assert/strict";
import test from "node:test";
import { tallCellComputeShader } from "../lib/tall-cell-kernels";
import { tallCellMethod } from "../lib/methods/tall-cell";
import { resolveMethodValues } from "../lib/methods/types";

test("tall cells expose both velocity transport schemes", () => {
  const transport = tallCellMethod.params.find((spec) => spec.key === "velocityTransport");
  assert.equal(transport?.kind, "select");
  if (transport?.kind !== "select") return;
  assert.deepEqual(transport.options.map((option) => option.value), ["maccormack", "semi-lagrangian"]);
  assert.equal(resolveMethodValues(tallCellMethod, "balanced", {}).velocityTransport, "maccormack");
  assert.equal(resolveMethodValues(tallCellMethod, "balanced", { velocityTransport: "semi-lagrangian" }).velocityTransport, "semi-lagrangian");
});

test("tall-cell semi-Lagrangian finish consumes the shared predictor", () => {
  assert.match(tallCellComputeShader, /fn finishSemiLagrangianAdvection/);
  assert.match(tallCellComputeShader, /var v=textureLoad\(predictedVelocityIn,id,0\)\.xyz/);
  assert.match(tallCellComputeShader, /fn finishAdvection/);
  assert.match(tallCellComputeShader, /var v=boundedMacCormack\(id,p\)/);
});

test("tall phi transport is independent of the velocity transport selector", () => {
  assert.match(tallCellComputeShader, /fn transportPhi/);
  assert.match(tallCellComputeShader, /var phi=volumeCorrectedPhi\(samplePhi\(traceDeparture\(p,dt\)\),dt\)/);
  assert.doesNotMatch(tallCellComputeShader, /correctedPhi|MacCormackPhi/);
  assert.match(tallCellComputeShader, /fn leastSquaresPhi/);
  assert.match(tallCellComputeShader, /if\(maxBase>=2\)\{desired=max\(2,desired\);\}/);
  assert.match(tallCellComputeShader, /fineDims\(\)\.y-regularLayers\(\)>=2&&base<2u/);
});

test("phi subdivision is GPU-governed without multiplying pressure work", () => {
  assert.match(tallCellComputeShader, /fn planSubsteps/);
  assert.match(tallCellComputeShader, /clamp\(u32\(ceil\(previousCfl\/2\.0\)\),1u,8u\)/);
  assert.match(tallCellComputeShader, /atomicStore\(&governor\[2\],bitcast<u32>\(dtPhi\)\)/);
  assert.match(tallCellComputeShader, /phiDispatchArgs\[base\+2u\]/);
  assert.match(tallCellComputeShader, /atomicMax\(&governor\[0\],bitcast<u32>\(cfl\)\)/);
});

test("tall-cell pressure warm start remains explicitly switchable", () => {
  const warmStart = tallCellMethod.params.find((spec) => spec.key === "pressureWarmStart");
  assert.equal(warmStart?.kind, "select");
  assert.equal(resolveMethodValues(tallCellMethod, "balanced", {}).pressureWarmStart, "off");
  assert.equal(resolveMethodValues(tallCellMethod, "balanced", { pressureWarmStart: "on" }).pressureWarmStart, "on");
});

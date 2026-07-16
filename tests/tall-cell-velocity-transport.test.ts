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
  assert.match(tallCellComputeShader, /var phi=samplePhi\(traceDeparture\(p,dt\)\)/);
  assert.doesNotMatch(tallCellComputeShader, /correctedPhi|MacCormackPhi/);
  assert.match(tallCellComputeShader, /fn leastSquaresPhi/);
  assert.match(tallCellComputeShader, /if\(maxBase>=2\)\{desired=max\(2,desired\);\}/);
  assert.match(tallCellComputeShader, /fineDims\(\)\.y-regularLayers\(\)>=2&&base<2u/);
});

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

test("tall density keeps a conservative average and a separate top topology endpoint", () => {
  assert.match(tallCellComputeShader, /var amount=textureLoad\(volumeIn,vec3i\(x,0,z\),0\)\.x\*f32\(base\)/);
  assert.match(tallCellComputeShader, /fn advectedTallTopGuide[\s\S]*offset<regularLayers\(\)/);
  assert.match(tallCellComputeShader, /if\(id\.y<=1\)\{tallAlpha=advectedTallVolume[\s\S]*if\(id\.y==0\)\{alpha=tallAlpha[\s\S]*id\.y==1\)\{alpha=advectedTallTopGuide/);
  assert.match(tallCellComputeShader, /else if\(id\.y==1\)\{for\(var offset=0;offset<regularLayers\(\)/);
  assert.match(tallCellComputeShader, /advectedTallVolume[\s\S]*return max\(0\.0,amount\/f32\(base\)\)/);
  assert.doesNotMatch(tallCellComputeShader, /advectedTallVolume[\s\S]*return clamp\(amount\/f32\(base\),0\.0,1\.0\)/);
  assert.match(tallCellComputeShader, /rawVolumeFlux[\s\S]*!representedWorld\(q\)\|\|!representedWorld\(q\+offset\)/);
  assert.match(tallCellComputeShader, /if\(maxBase>=2\)\{desired=max\(2,desired\);\}/);
  assert.match(tallCellComputeShader, /fineDims\(\)\.y-regularLayers\(\)>=2&&base<2u/);
});

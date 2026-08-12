import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const host = readFileSync(new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");
const shader = readFileSync(new URL("../lib/webgpu-uniform-reference.wgsl.ts", import.meta.url), "utf8");
const extrapolator = readFileSync(new URL("../lib/webgpu-uniform-velocity-extrapolation.ts", import.meta.url), "utf8");
const extrapolationShader = readFileSync(
  new URL("../lib/webgpu-uniform-velocity-extrapolation.wgsl.ts", import.meta.url), "utf8");
const tallCells = readFileSync(new URL("../docs/papers/tallCells.txt", import.meta.url), "utf8");

test("uniform velocity evolution follows CM11b bounded MacCormack transport", () => {
  assert.match(tallCells, /To advect u we use the modified MacCormack scheme/);
  assert.match(tallCells, /revert to simple Semi-Lagrangian advection/);
  assert.match(host, /\["advect", "Bounded MacCormack velocity prediction"/);
  assert.match(host, /\["reverse", "Bounded MacCormack reverse advection"/);
  assert.match(host, /\["correct", "Bounded MacCormack correction and body forces"/);
  assert.match(host, /encodeVelocityExtrapolation\(encoder, true\)/);
  assert.match(host, /this\.pipelines\.advect, this\.advectGroup/);
  assert.match(host, /this\.pipelines\.reverse, this\.reverseGroup/);
  assert.match(host, /this\.pipelines\.correct, this\.correctGroup/);
  assert.match(extrapolator, /predicted \? this\.packPredictedGroup : this\.packCurrentGroup/);
});

test("bounded correction falls back outside donor bounds and applies forces once", () => {
  const boundedStart = shader.indexOf("fn boundedMacCormack");
  const bounded = shader.slice(boundedStart, shader.indexOf("@compute", boundedStart));
  assert.match(bounded, /corrected<lower\|\|corrected>upper/);
  assert.match(bounded, /select\(corrected,predicted,revert\)/);
  const correctStart = shader.indexOf("fn correctAdvection");
  const correct = shader.slice(correctStart, shader.indexOf("@compute", correctStart));
  assert.match(correct, /applyVelocityForces\(id,v,dt,h\)/);
  const predictStart = shader.indexOf("fn advect(@builtin");
  const predict = shader.slice(predictStart, shader.indexOf("@compute", predictStart));
  assert.doesNotMatch(predict, /applyVelocityForces/);
});

test("hierarchy restriction declares corresponding fine support for every velocity component", () => {
  const start = extrapolationShader.indexOf("fn restrictKnownVelocity");
  const body = extrapolationShader.slice(start, extrapolationShader.indexOf("@compute", start));
  assert.match(body, /for \(var component = 0u; component < 3u; component \+= 1u\)/);
  assert.match(body, /if \(result\.y <= 0\.0\) \{\s*result = hierarchyCorrespondingCellSample\(p, sourceDims, component\)/);
  assert.doesNotMatch(body, /component\s*==\s*1u/,
    "the corresponding-cell fallback must not be restricted to the Y component");
});

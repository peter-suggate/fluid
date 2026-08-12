import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const host = readFileSync(new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");
const shader = readFileSync(new URL("../lib/webgpu-uniform-reference.wgsl.ts", import.meta.url), "utf8");
const extrapolator = readFileSync(new URL("../lib/webgpu-uniform-velocity-extrapolation.ts", import.meta.url), "utf8");
const extrapolationShader = readFileSync(
  new URL("../lib/webgpu-uniform-velocity-extrapolation.wgsl.ts", import.meta.url), "utf8");
const tallCells = readFileSync(new URL("../docs/papers/tallCells.txt", import.meta.url), "utf8");
const separatingBoundaries = readFileSync(
  new URL("../docs/papers/A_Multigrid_Fluid_Pressure_Solver_Handling_Separat.txt", import.meta.url), "utf8");

test("uniform velocity evolution selects semi-Lagrangian or bounded MacCormack transport", () => {
  assert.match(tallCells, /To advect u we use the modified MacCormack scheme/);
  assert.match(tallCells, /revert to simple Semi-Lagrangian advection/);
  assert.match(host, /\["semiLagrangian", "Semi-Lagrangian velocity advection and body forces"/);
  assert.match(host, /\["advect", "Bounded MacCormack velocity prediction"/);
  assert.match(host, /\["reverse", "Bounded MacCormack reverse advection"/);
  assert.match(host, /\["correct", "Bounded MacCormack correction and body forces"/);
  assert.match(host, /encodeVelocityExtrapolation\(encoder, true\)/);
  assert.match(host, /this\.velocityTransport === "maccormack"/);
  assert.match(host, /this\.pipelines\.semiLagrangian, this\.semiLagrangianGroup/);
  assert.match(host, /this\.pipelines\.advect, this\.advectGroup/);
  assert.match(host, /this\.pipelines\.reverse, this\.reverseGroup/);
  assert.match(host, /this\.pipelines\.correct, this\.correctGroup/);
  assert.match(extrapolator, /predicted \? this\.packPredictedGroup : this\.packCurrentGroup/);
});

test("semi-Lagrangian velocity transport applies forces in its single pass", () => {
  const start = shader.indexOf("fn semiLagrangianAdvection");
  const body = shader.slice(start, shader.indexOf("@compute", start));
  assert.match(body, /advectVelocityComponent/);
  assert.match(body, /applyVelocityForces\(id,v,dt,h\)/);
});

test("bounded correction falls back outside donor bounds and applies forces once", () => {
  const boundedStart = shader.indexOf("fn boundedMacCormack");
  const bounded = shader.slice(boundedStart, shader.indexOf("@compute", boundedStart));
  assert.match(bounded, /corrected<lower\|\|corrected>upper/);
  assert.match(bounded, /select\(corrected,predicted,revert\)/);
  const correctStart = shader.indexOf("fn correctAdvection");
  const correct = shader.slice(correctStart, shader.indexOf("@compute", correctStart));
  assert.match(correct, /applyVelocityForces\(id,v,dt,h\)/);
  assert.doesNotMatch(correct, /v\.[xyz]=original\.[xyz]/,
    "a released wall face must retain body forces until the LCP projection");
  const predictStart = shader.indexOf("fn advect(@builtin");
  const predict = shader.slice(predictStart, shader.indexOf("@compute", predictStart));
  assert.doesNotMatch(predict, /applyVelocityForces/);
});

test("released CM11a wall faces let backward density traces vacate the wall", () => {
  assert.match(separatingBoundaries, /liquid artificially crawling[\s\S]*sticking to the ceiling/);
  assert.match(separatingBoundaries, /normal velocity to be exactly zero and greater than or[\s\S]*zero at the solid boundary/);
  assert.match(shader, /fn backwardTraceExitsReleasedFace/);
  assert.match(shader,
    /if\(direction<0\.0&&backwardTraceExitsReleasedFace\(candidate\)\)\{p=candidate;break;\}[\s\S]*let next=clampTraceToDomain\(candidate\)/,
    "the released departure must escape before ordinary contact-wall clamping");
});

test("hierarchy restriction keeps cell-centred fallback off horizontal MAC faces", () => {
  const start = extrapolationShader.indexOf("fn restrictKnownVelocity");
  const body = extrapolationShader.slice(start, extrapolationShader.indexOf("@compute", start));
  assert.match(body, /for \(var component = 0u; component < 3u; component \+= 1u\)/);
  assert.match(body, /if \(result\.y <= 0\.0 && component == 1u\) \{\s*result = hierarchyCorrespondingCellSample\(p, sourceDims, component\)/);
  assert.match(extrapolationShader, /not centred on horizontal MAC faces/,
    "the collocated CM11b fallback must not shift x\/z face support under reflection");
});

test("paper stencils use D4-canonical floating-point reduction trees", () => {
  assert.match(shader, /fn d4Sum6\(/);
  assert.match(shader, /fn d4Sum8\(/);
  assert.match(extrapolationShader, /fn d4Sum3\(/);
  assert.match(extrapolationShader, /fn d4Sum8Vec2\(/);
});

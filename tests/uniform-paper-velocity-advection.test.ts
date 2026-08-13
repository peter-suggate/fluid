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
  assert.match(body, /v\.y=min\(v\.y,faceVelocity\(id\)\.y\)/,
    "a positive closed wall must preserve velocity away from the solid before forces");
  assert.doesNotMatch(body, /v\.y=faceVelocity\(id\)\.y/,
    "restoring the old ceiling face after gravity pins released liquid to zero velocity");
});

test("liquid momentum never samples exterior transport while retaining SL characteristics", () => {
  assert.doesNotMatch(extrapolationShader, /localMask|sourceMask/);
  assert.doesNotMatch(extrapolator, /packedValues, this\.resolvedDistances/);
  assert.match(shader, /fn samplePhysicalVelocityComponent/);
  const physicalStart = shader.indexOf("fn samplePhysicalVelocityComponent");
  const physical = shader.slice(physicalStart, shader.indexOf("// Reconstruct every RK2", physicalStart));
  assert.match(physical, /physicalVelocityFaceValue\(donor,component\)/);
  assert.doesNotMatch(physical, /transportIn/,
    "liquid momentum interpolation must never read the exterior transport field");
  assert.match(shader, /if\(!liquidOnlyVelocityAdvection\(\)\|\|!velocityFaceLiquid\(position,component\)\)/,
    "the complete extension remains transport support outside liquid");
  assert.match(shader, /let supported=samplePhysicalVelocityComponent\(departure,component\)/,
    "an updated liquid face must gather only prior-liquid velocityIn donors");
  assert.match(shader, /return 0\.0;\s*\}/,
    "a failed trace must not fall back to exterior transport momentum");
});

test("liquid-only velocity gathering excludes density halos and retains wall conditions", () => {
  const phaseStart = shader.indexOf("fn velocityPhaseWeight");
  const phase = shader.slice(phaseStart, shader.indexOf("// Interpolate the authoritative", phaseStart));
  assert.match(phase, /let liquidFraction=clamp\(textureLoad\(velocityPhaseIn,p,0\)\.x,0\.0,1\.0\)/,
    "prior rho/V must identify authoritative liquid donors");
  assert.match(phase, /return select\(0\.0,1\.0,liquidFraction>0\.5&&cellOpenFraction\(p\)>1e-5\)/,
    "thin numerical halos and closed cells must not become full velocity donors");
  assert.match(phase, /if\(!valid\(p\)&&!valid\(neighbor\)\)\{return 0\.0;\}/,
    "only truly exterior stencil locations may be rejected");
  assert.match(phase, /return boundaryVelocity\(neighbor\)\[component\]/,
    "negative domain faces must use their authoritative boundary condition");
  assert.match(phase, /if\(valid\(p\)\)\{return textureLoad\(velocityIn,p,0\)\[component\];\}/,
    "stored positive and embedded wall faces must remain authoritative donors");
  assert.match(shader, /return volume\(face\)>1e-5\|\|volume\(face\+axis\)>1e-5/,
    "sub-isovalue liquid must remain on the liquid-only path");
  assert.match(shader, /fn clampVelocityTraceToDomain/);
  assert.match(shader, /midpoint=clampVelocityTraceToDomain/);
  assert.match(shader, /point=clampVelocityTraceToDomain/);
});

test("long characteristics remain semi-Lagrangian and substep within the accurate band", () => {
  const start = shader.indexOf("fn departurePoint");
  const body = shader.slice(start, shader.indexOf("// advectVelocityComponent", start));
  assert.match(body, /for\(var step=0;step<32;step\+=1\)/);
  assert.match(body, /stepSeconds=min\(remaining,1\.5\/max\(rate,1e-6\)\)/);
  assert.match(body, /let midpoint=clampVelocityTraceToDomain\(point-0\.5\*first\*signedStep\/h\)/);
  assert.match(body, /point=clampVelocityTraceToDomain\(point-sampleVelocity\(midpoint\)\*signedStep\/h\)/);
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

test("sub-isovalue liquid keeps separating-wall force and velocity support", () => {
  assert.match(shader, /let centerLiquid=occupancy>1e-5/);
  assert.match(shader, /let yLiquid=yOccupancy>1e-5/);
  const projectStart = shader.indexOf("fn project(@builtin");
  const project = shader.slice(projectStart, shader.indexOf("// Moving-solid bookkeeping", projectStart));
  assert.match(project,
    /v\[axis\]=select\(solidVelocity,min\(v\[axis\],solidVelocity\),volume\(id\)>1e-5\)/,
    "thin liquid must retain only the positive-wall velocity directed away from the solid");
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

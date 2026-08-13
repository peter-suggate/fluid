import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const host = readFileSync(new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");
const shader = readFileSync(new URL("../lib/webgpu-uniform-reference.wgsl.ts", import.meta.url), "utf8");

test("gamma diffusion executes the paper's ordered dimension and pair-parity passes", () => {
  for (const entry of ["diffuseGammaX0", "diffuseGammaX1", "diffuseGammaY0",
    "diffuseGammaY1", "diffuseGammaZ0", "diffuseGammaZ1"]) {
    assert.match(host, new RegExp(`\\["${entry}"`));
    assert.match(shader, new RegExp(`fn ${entry}\\(`));
  }
  assert.match(host, /six passes are\n\s*\/\/ one complete paper diffusion iteration/);
  assert.match(host, /UNIFORM_GAMMA_DIFFUSION_ITERATIONS = 7/);
  assert.match(host, /iteration < this\.gammaDiffusionIterations/);
  assert.match(host, /if \(this\.gammaDiffusionIterations > 0\) seam/,
    "a zero-iteration ablation must not publish a stage seam");
});

test("each pair half-equalizes gamma and transfers the corresponding donor density", () => {
  assert.match(shader, /let averageGamma=0\.5\*\(lowerGamma\+upperGamma\)/);
  assert.match(shader, /upperRho\*\(upperGamma-lowerGamma\)\/\(2\.0\*max\(upperGamma,1e-9\)\)/);
  assert.match(shader, /lowerRho\*\(lowerGamma-upperGamma\)\/\(2\.0\*max\(lowerGamma,1e-9\)\)/);
  assert.doesNotMatch(shader, /flux=\(\(xFlux\+zFlux\)\+yFlux\)\/6\.0/);
});

test("each parity pass writes every cell exactly once, including unpaired walls", () => {
  assert.match(shader, /if\(lowerCoordinate<0\|\|!valid\(upper\)\)\{/);
  assert.match(shader, /textureStore\(volumeOut,id,vec4f\(volume\(id\)\)\)/);
  assert.match(shader, /select\(upperRho,lowerRho,coordinate==lowerCoordinate\)/);
  assert.doesNotMatch(shader, /textureStore\(volumeOut,(?:q|upper)/,
    "one invocation must never race another invocation's output texel");
});

test("closed-wall scalar characteristics substep and stop at the wall", () => {
  assert.match(shader, /fn clampTraceToDomain\(p:vec3f\)->vec3f/);
  assert.match(shader, /let substeps=clamp\(i32\(ceil\(length\(sampleVelocity\(position\)\)\*dt\/hMin\)\),1,16\)/);
  assert.match(shader, /let next=clampTraceToDomain\(candidate\)/);
  assert.match(shader, /if\(cellInsideSolid\(vec3i\(floor\(next\)\)\)\)\{break;\}/);
  assert.match(shader, /return integrateTraceOffset\(id,dt,h,-1\.0\)/);
  assert.doesNotMatch(shader, /reflectScalarCoordinate/,
    "folding an overshot trace back into the tank duplicates wall-film samples");
});

test("released-wall density transport leaves no synthetic backward coefficient", () => {
  assert.match(shader,
    /let advectedGamma=select\(min\(sampledGamma,2\.5\),clamp\(sampledGamma,0\.5,2\.5\),total>=1\.0-1e-6\)/,
    "the interior gamma floor must not fill the missing part of a released-wall row");
  const traceStart = shader.indexOf("fn traceGammaAndBeta");
  const trace = shader.slice(traceStart, shader.indexOf("@compute", traceStart));
  assert.match(trace, /if\(total<=1e-9\)\{return;\}/);
  assert.match(trace, /let weight=transportStencilWeight\(base,f,corner\);/,
    "partially exterior backward rows must retain their missing weight");
  assert.doesNotMatch(trace, /transportStencilWeight\(base,f,corner\)\/total/);
  assert.doesNotMatch(trace,
    /atomicAdd\(&sharpenDeposits\[linearIndex\(id\)\]/,
    "an empty backward stencil must not suppress the donor's forward remainder");
});

test("CM12 gather publishes the conditioned operator row sum exactly once", () => {
  assert.match(shader,
    /rhoNext\+=scaled\*volume\(donor\);/);
  assert.match(shader,
    /gammaNext\+=scaled;/,
    "step 1 already transported cumulative gamma; the gather must publish the resulting row sum");
  assert.doesNotMatch(shader,
    /gammaNext\+=scaled\*textureLoad\([^,]+,donor,0\)\.x/,
    "multiplying by donor gamma applies the cumulative factor a second time");
  assert.doesNotMatch(host,
    /copyTextureToTexture\(\{ texture: this\.gammaA \}, \{ texture: this\.surfaceA \}/,
    "the gather must not snapshot gamma for a second scalar advection");
  assert.match(host,
    /this\.densityGatherGroup = group\(this\.velocityA, this\.velocityB, this\.pressureA/);
});

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

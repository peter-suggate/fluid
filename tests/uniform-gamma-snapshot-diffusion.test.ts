import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const host = readFileSync(new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");
const shader = readFileSync(new URL("../lib/webgpu-uniform-reference.wgsl.ts", import.meta.url), "utf8");

test("gamma diffusion executes one Jacobi snapshot pass per dimension", () => {
  for (const entry of ["diffuseGammaX", "diffuseGammaY", "diffuseGammaZ"]) {
    assert.match(host, new RegExp(`\\["${entry}"`));
    assert.match(shader, new RegExp(`fn ${entry}\\(`));
  }
  assert.doesNotMatch(host, /diffuseGamma[XYZ][01]|averageGammaDiffusion|gammaMirror/);
  assert.doesNotMatch(shader, /diffuseGamma[XYZ][01]|averageGammaDiffusion|parity/);
  assert.match(host, /UNIFORM_GAMMA_DIFFUSION_DEFAULT_ITERATIONS = 1/);
  assert.match(host, /UNIFORM_GAMMA_DIFFUSION_MAX_ITERATIONS = 7/);
  assert.match(host, /const diffusionPasses = \[\s*\["x"[\s\S]*\["y"[\s\S]*\["z"/);
  assert.match(host, /iteration < this\.gammaDiffusionIterations/);
  assert.match(host, /if \(this\.gammaDiffusionIterations > 0\) seam/,
    "a zero-iteration ablation must not publish a stage seam");
});

type LineState = Readonly<{ rho: readonly number[]; gamma: readonly number[] }>;

function jacobiLine(state: LineState, apertures: readonly number[]): LineState {
  const rho = state.rho.map((ownRho, index) => {
    const ownGamma = state.gamma[index]!;
    let result = ownRho;
    for (const direction of [-1, 1]) {
      const neighborIndex = index + direction;
      if (neighborIndex < 0 || neighborIndex >= state.rho.length) continue;
      const open = apertures[Math.min(index, neighborIndex)]!;
      const neighborGamma = state.gamma[neighborIndex]!;
      if (neighborGamma > ownGamma) {
        result += open * state.rho[neighborIndex]!
          * (neighborGamma - ownGamma) / (2 * neighborGamma);
      } else if (ownGamma > neighborGamma) {
        result -= open * ownRho * (ownGamma - neighborGamma) / (2 * ownGamma);
      }
    }
    return result;
  });
  const gamma = state.gamma.map((ownGamma, index) => {
    let result = ownGamma;
    for (const direction of [-1, 1]) {
      const neighborIndex = index + direction;
      if (neighborIndex < 0 || neighborIndex >= state.gamma.length) continue;
      const open = apertures[Math.min(index, neighborIndex)]!;
      result += 0.5 * open * (state.gamma[neighborIndex]! - ownGamma);
    }
    return result;
  });
  return { rho, gamma };
}

const sum = (values: readonly number[]) => values.reduce((total, value) => total + value, 0);
const closeArray = (actual: readonly number[], expected: readonly number[], tolerance = 1e-12) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]!) <= tolerance,
    `index ${index}: expected ${expected[index]}, received ${value}`));
};

test("the Jacobi face formula conserves rho and gamma with partial apertures", () => {
  const before = {
    rho: [0.13, 0.92, 0.37, 0.81, 0.26, 0.55],
    gamma: [0.6, 1.9, 0.8, 2.3, 1.2, 0.7],
  };
  const after = jacobiLine(before, [1, 0.35, 0, 0.8, 1]);
  assert.ok(Math.abs(sum(after.rho) - sum(before.rho)) < 1e-12);
  assert.ok(Math.abs(sum(after.gamma) - sum(before.gamma)) < 1e-12);
  assert.ok(after.rho.every((value) => value >= 0));
  assert.ok(after.gamma.every((value) => value >= 0));
});

test("one axis pass reads both neighbours from the same snapshot", () => {
  const after = jacobiLine({
    rho: [0.2, 0.4, 0.8, 0.6, 0.3],
    gamma: [0, 0, 2, 0, 0],
  }, [1, 1, 1, 1]);
  closeArray(after.gamma, [0, 1, 0, 1, 0]);
  closeArray(after.rho, [0.2, 0.8, 0, 1, 0.3]);
  assert.match(shader, /let ownRho=volume\(id\);[\s\S]*let ownGamma=textureLoad\(gammaIn,id,0\)\.x/);
  assert.match(shader, /volume\(lower\),textureLoad\(gammaIn,lower,0\)\.x/);
  assert.match(shader, /volume\(upper\),textureLoad\(gammaIn,upper,0\)\.x/);
  assert.doesNotMatch(shader, /textureStore\(volumeOut,(?:lower|upper)/,
    "each invocation must own exactly one output texel");
});

test("the snapshot update has no even/odd grid-phase preference", () => {
  const before = {
    rho: [0.11, 0.72, 0.45, 0.88, 0.31, 0.64],
    gamma: [0.5, 1.7, 0.9, 2.2, 1.1, 0.6],
  };
  const apertures = [1, 0.4, 0.75, 0.2, 0.9];
  const forward = jacobiLine(before, apertures);
  const reflected = jacobiLine({
    rho: [...before.rho].reverse(),
    gamma: [...before.gamma].reverse(),
  }, [...apertures].reverse());
  closeArray(forward.rho, [...reflected.rho].reverse());
  closeArray(forward.gamma, [...reflected.gamma].reverse());
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

test("released-wall gamma sampling keeps the exterior zero without an empty self coefficient", () => {
  assert.match(shader,
    /let advectedGamma=select\(min\(sampledGamma,2\.5\),clamp\(sampledGamma,0\.5,2\.5\),total>=1\.0-1e-6\)/,
    "the interior gamma floor must not fill the missing part of a released-wall row");
  const traceStart = shader.indexOf("fn traceGammaAndBeta");
  const trace = shader.slice(traceStart, shader.indexOf("@compute", traceStart));
  assert.match(trace, /if\(total<=1e-9\)\{return;\}/);
  assert.match(trace, /let weight=transportStencilWeight\(base,f,corner\)\/total;/,
    "beta must normalize the surviving interpolation stencil");
  assert.doesNotMatch(trace,
    /atomicAdd\(&sharpenDeposits\[linearIndex\(id\)\]/,
    "an empty backward stencil must not suppress the donor's forward remainder");
  const gatherStart = shader.indexOf("fn gatherConservativeDensity");
  const gather = shader.slice(gatherStart, shader.indexOf("fn gammaDiffusionFluxInto", gatherStart));
  assert.match(gather, /let weight=transportStencilWeight\(base,f,corner\)\/total;/,
    "density gathering must use the same normalized visible weights as beta");
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

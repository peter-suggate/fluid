import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { resolveMethodValues } from "../lib/core/method-contract";
import {
  ADAPTIVE_MASS_RUNTIME_PARAM_KEYS,
  adaptiveMassMethod,
  adaptiveMassSolverOptions,
} from "../lib/methods/adaptive-mass/method";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from
  "../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline";

const resident = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts", import.meta.url,
), "utf8");
const wgsl = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url,
), "utf8");

test("Sparse CM12 exposes live gamma-diffusion and sharpening controls", () => {
  for (const key of ["gammaDiffusion", "surfaceSharpening"] as const) {
    const spec = adaptiveMassMethod.params.find((candidate) => candidate.key === key);
    assert.equal(spec?.kind, "select");
    assert.equal(spec?.update, "runtime");
    if (spec?.kind === "select") {
      assert.equal(spec.default, "on");
      assert.deepEqual(spec.options.map(({ value }) => value), ["on", "off"]);
    }
    assert.ok(ADAPTIVE_MASS_RUNTIME_PARAM_KEYS.includes(key));
  }

  const defaults = adaptiveMassSolverOptions(
    resolveMethodValues(adaptiveMassMethod, "balanced", {}),
  );
  assert.equal(defaults.gammaDiffusionEnabled, true);
  assert.equal(defaults.surfaceSharpeningEnabled, true);
  assert.equal(defaults.sharpeningStrength, 1);

  const strengthSpec = adaptiveMassMethod.params.find(
    (candidate) => candidate.key === "sharpeningStrength",
  );
  assert.equal(strengthSpec?.kind, "number");
  assert.equal(strengthSpec?.update, "runtime");
  assert.ok(ADAPTIVE_MASS_RUNTIME_PARAM_KEYS.includes("sharpeningStrength"));

  const disabled = adaptiveMassSolverOptions(
    resolveMethodValues(adaptiveMassMethod, "balanced", {
      gammaDiffusion: "off",
      surfaceSharpening: "off",
      sharpeningStrength: 0.5,
    }),
  );
  assert.equal(disabled.gammaDiffusionEnabled, false);
  assert.equal(disabled.surfaceSharpeningEnabled, false);
  assert.equal(disabled.sharpeningStrength, 0.5);
});

test("conditioning controls preserve the mandatory sparse scalar-publication suffix", () => {
  const gammaStage = resident.slice(resident.indexOf('stage("gamma-diffusion"'),
    resident.indexOf('stage("surface-sharpening"'));
  assert.match(gammaStage, /if \(!gammaDiffusionEnabled\) return/);
  assert.equal(gammaStage.match(/dispatchAccepted\("scatterGammaSnapshotRows"/g)?.length, 1);
  assert.equal(gammaStage.match(/dispatchAccepted\("finalizeGammaSnapshot"/g)?.length, 1);
  assert.doesNotMatch(gammaStage, /GammaRefinement/);

  const sharpeningStage = resident.slice(resident.indexOf('stage("surface-sharpening"'),
    resident.indexOf('stage("symmetry-authority"'));
  assert.match(sharpeningStage,
    /if \(surfaceSharpeningEnabled \|\| gammaDiffusionEnabled\)[\s\S]*finalizeSharpening/);
  assert.match(sharpeningStage, /beginSparseCM12FinalScalarMasks/);
  assert.match(sharpeningStage, /publishSparseCM12FinalScalarMasks/);
  assert.match(sharpeningStage, /sealSparseCM12FinalScalarMasks/);

  assert.match(wgsl, /fn conditionedDensity[\s\S]*gammaDiffusionEnabled\(\)/);
  assert.match(wgsl,
    /fn conditionedDensity\(cell:u32\)->f32\{return state\[select\(destinationDensity\(\),\s*p\.stateOffsets2\.x,gammaDiffusionEnabled\(\)\)\+cell\];\}/);
  assert.match(wgsl,
    /fn conditionedGamma\(cell:u32\)->f32\{return state\[select\(destinationGamma\(\),\s*p\.stateOffsets2\.y,gammaDiffusionEnabled\(\)\)\+cell\];\}/);
  assert.doesNotMatch(wgsl, /fn scatterGammaRefinementRows|fn finalizeGammaRefinement/);
  assert.match(wgsl,
    /if\(!surfaceSharpeningEnabled\(\)\)[\s\S]*state\[destinationDensity\(\)\+cell\]=max\(0\.0,conditionedDensity\(cell\)\)/);
  assert.match(wgsl,
    /return min\(0\.0,delta\*surfaceSharpeningStrength\(\)\)/);
});

test("the SIM pipeline exposes both transforms as live stage switches", () => {
  for (const [id, param] of [
    ["gamma-diffusion", "gammaDiffusion"],
    ["surface-sharpening", "surfaceSharpening"],
  ] as const) {
    const stage = ADAPTIVE_MASS_FLUID_PIPELINE.stages.find(
      (candidate) => candidate.id === id,
    );
    assert.ok(stage, `${id} must remain visible in the SIM pipeline`);
    assert.deepEqual(stage.toggle, { param, on: "on", off: "off",
      hint: stage.toggle?.hint });
    const context = { values: { [param]: "off" } } as Parameters<typeof stage.state>[0];
    assert.equal(stage.state(context), "off");
    assert.match(stage.chip(context), /disabled/i);
  }
});

test("capacity early exit is an isolated destination-bit fixed-point gate", () => {
  assert.match(resident, /createDensityCapacityEarlyExitOracleForQA/);
  assert.match(resident,
    /for \(let capacityPass = 0; capacityPass < 2; capacityPass \+= 1\)/);
  assert.match(resident,
    /for \(let gate = 0; gate < 6; gate \+= 1\)/);
  assert.match(wgsl,
    /bitcast<u32>\(state\[destinationDensity\(\)\+cell\]\)!=bitcast<u32>\(before\)/);
  assert.match(wgsl,
    /fn densityCapacityRepairGateOpen[\s\S]*atomicLoad\([\s\S]*DENSITY_CAPACITY_GATE_BASE/);
  assert.match(wgsl, /fn finalizeDensityCapacityRepairSeedGate/);
});

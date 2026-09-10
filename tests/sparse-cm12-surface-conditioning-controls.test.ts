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
    resident.indexOf('stage("scalar-publication"'));
  assert.match(sharpeningStage,
    /if \(surfaceSharpeningEnabled \|\| gammaDiffusionEnabled\)[\s\S]*finalizeSharpening/);
  const publicationStage = resident.slice(resident.indexOf('stage("scalar-publication"'),
    resident.indexOf('stage("body-forces"'));
  assert.match(publicationStage, /beginSparseCM12FinalScalarMasks/);
  assert.match(publicationStage, /publishSparseCM12FinalScalarMasks/);
  assert.match(publicationStage, /sealSparseCM12FinalScalarMasks/);
  assert.doesNotMatch(publicationStage, /if \(!corrections\.densityCapacityRepairEnabled/);

  assert.match(wgsl, /fn conditionedDensity[\s\S]*gammaDiffusionEnabled\(\)/);
  assert.match(wgsl,
    /fn conditionedDensity\(cell:u32\)->f32\{return state\[select\(destinationDensity\(\),\s*p\.stateOffsets2\.x,gammaDiffusionEnabled\(\)\)\+cell\];\}/);
  assert.match(wgsl,
    /fn conditionedGamma\(cell:u32\)->f32\{return state\[select\(destinationGamma\(\),\s*p\.stateOffsets2\.y,gammaDiffusionEnabled\(\)\)\+cell\];\}/);
  assert.doesNotMatch(wgsl, /fn scatterGammaRefinementRows|fn finalizeGammaRefinement/);
  assert.match(wgsl,
    /if\(!surfaceSharpeningEnabled\(\)\)[\s\S]*state\[destinationDensity\(\)\+cell\]=max\(0\.0,conditionedDensity\(cell\)\)/);
  assert.match(wgsl,
    /return max\(-rho,min\(0\.0,delta\*surfaceSharpeningStrength\(\)\)\)/);
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


test("correction dials persist, normalize, and reach direct solver options", () => {
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    massConservation: "off", massConservationStrength: 0.35,
    gammaConditioning: "off", gammaConditioningStrength: 1.5,
    gammaDiffusionStrength: 0.25, gammaDiffusionIterations: 4,
    sharpeningStrength: 3, sharpeningTau: 0.7,
    densityCapacityRepair: "off", densityCapacityRepairStrength: 0.4,
    densityCapacityRepairIterations: 12, volumeCorrection: "off",
    volumeCorrectionStrength: 2, volumeCorrectionCap: 3,
  });
  const options = adaptiveMassSolverOptions(values);
  assert.equal(options.massConservationEnabled, false);
  assert.equal(options.massConservationStrength, 0.35);
  assert.equal(options.gammaConditioningEnabled, false);
  assert.equal(options.gammaConditioningStrength, 1.5);
  assert.equal(options.gammaDiffusionIterations, 4);
  assert.equal(options.gammaDiffusionStrength, 0.25);
  assert.equal(options.sharpeningStrength, 3);
  assert.equal(options.sharpeningTau, 0.7);
  assert.equal(options.densityCapacityRepairEnabled, false);
  assert.equal(options.densityCapacityRepairStrength, 0.4);
  assert.equal(options.densityCapacityRepairIterations, 12);
  assert.equal(options.volumeCorrectionEnabled, false);
  assert.equal(options.volumeCorrectionStrength, 2);
  assert.equal(options.volumeCorrectionCap, 3);
  for (const stage of ADAPTIVE_MASS_FLUID_PIPELINE.stages) {
    for (const control of stage.controls ?? []) {
      if (control.kind !== "param-range" && control.kind !== "param-choice") continue;
      const spec = adaptiveMassMethod.params.find(param => param.key === control.param);
      assert.ok(spec, control.param);
      if (spec.kind === "number" && control.kind === "param-range") {
        assert.equal(control.min, spec.min, control.param);
        assert.equal(control.max, spec.max, control.param);
      }
    }
  }
  const defaults = adaptiveMassSolverOptions(resolveMethodValues(adaptiveMassMethod, "balanced", {}));
  assert.equal(defaults.gammaDiffusionIterations, 1);
  assert.equal(defaults.densityCapacityRepairIterations, 8);
  assert.equal(defaults.volumeCorrectionStrength, 1);
  assert.equal(defaults.volumeCorrectionCap, 1);
});

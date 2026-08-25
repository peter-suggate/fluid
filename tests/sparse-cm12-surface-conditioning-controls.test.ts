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

  const disabled = adaptiveMassSolverOptions(
    resolveMethodValues(adaptiveMassMethod, "balanced", {
      gammaDiffusion: "off",
      surfaceSharpening: "off",
    }),
  );
  assert.equal(disabled.gammaDiffusionEnabled, false);
  assert.equal(disabled.surfaceSharpeningEnabled, false);
});

test("conditioning controls preserve the mandatory sparse scalar-publication suffix", () => {
  const gammaStage = resident.slice(resident.indexOf('stage("gamma-diffusion"'),
    resident.indexOf('stage("surface-sharpening"'));
  assert.match(gammaStage, /if \(!gammaDiffusionEnabled\) return/);

  const sharpeningStage = resident.slice(resident.indexOf('stage("surface-sharpening"'),
    resident.indexOf('stage("symmetry-authority"'));
  assert.match(sharpeningStage,
    /if \(surfaceSharpeningEnabled \|\| gammaDiffusionEnabled\)[\s\S]*finalizeSharpening/);
  assert.match(sharpeningStage, /beginSparseCM12FinalScalarMasks/);
  assert.match(sharpeningStage, /publishSparseCM12FinalScalarMasks/);
  assert.match(sharpeningStage, /sealSparseCM12FinalScalarMasks/);

  assert.match(wgsl, /fn conditionedDensity[\s\S]*gammaDiffusionEnabled\(\)/);
  assert.match(wgsl,
    /if\(!surfaceSharpeningEnabled\(\)\)[\s\S]*state\[destinationDensity\(\)\+cell\]=max\(0\.0,conditionedDensity\(cell\)\)/);
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

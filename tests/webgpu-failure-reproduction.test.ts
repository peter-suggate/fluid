import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  DAM_UI_RUNTIME_DAWN_REPRODUCTION,
  DAM_UI_T0_DAWN_REPRODUCTION,
  MINIMAL_POWER_DAM_32_RUNTIME_DAWN_REPRODUCTION,
  MINIMAL_POWER_DAM_32_T0_DAWN_REPRODUCTION,
  dawnReproductionForGPUFailure,
  dawnReproductionForSmokeEnvironment,
} from "../lib/webgpu-failure-reproduction";

const exactEnvironment = {
  FLUID_SCENE: "dam-break-ui",
  FLUID_METHOD: "octree",
  FLUID_TARGET_S: "0.016",
  FLUID_MAX_DT: "0.008",
  FLUID_ORACLE_STEPS: "2",
  FLUID_EXPECT_EXACT_STEPS: "2",
  FLUID_EXPECT_GRID: "24,18,16",
};

const mini32Configuration = {
  sceneId: "minimal-power-dam-break-32",
  methodId: "octree",
  quality: "balanced",
  methodOverrides: {
    coarseBackend: "losasso",
    maximumLeafSize: "32",
    interfaceRefinementBandCells: 3,
    surfaceRefinementGradingLayers: 3,
    globalFineLevelSetFactor: "1",
  },
  grid: [32, 32, 32] as const,
  fixedDt_s: 0.004,
  maxDt_s: 0.004,
};

test("browser failures map to the smallest serialized exact Dawn case", () => {
  for (const message of [
    "GPU initialization failed: invalid compute pipeline",
    "Initial sparse authority structured publication validation failed",
    "Paused t=0 authority rejected: structured boundary did not publish",
  ]) {
    assert.deepEqual(dawnReproductionForGPUFailure(message), DAM_UI_T0_DAWN_REPRODUCTION);
  }
  for (const message of [
    "GPU runtime stopped: command submission failed",
    "GPU device lost: internal error",
    "GPU device lost mid-simulation",
  ]) {
    assert.deepEqual(dawnReproductionForGPUFailure(message), DAM_UI_RUNTIME_DAWN_REPRODUCTION);
  }
  assert.equal(dawnReproductionForGPUFailure("WebGPU stopped during component cleanup"), undefined);
  assert.equal(dawnReproductionForGPUFailure("WebGPU stopped during page close"), undefined);
});

test("browser failures retain the exact mini32 scene profile and timestep", () => {
  assert.deepEqual(dawnReproductionForGPUFailure(
    "GPU initialization failed: invalid compute pipeline", mini32Configuration,
  ), MINIMAL_POWER_DAM_32_T0_DAWN_REPRODUCTION);
  assert.deepEqual(dawnReproductionForGPUFailure(
    "GPU runtime stopped: command submission failed", mini32Configuration,
  ), MINIMAL_POWER_DAM_32_RUNTIME_DAWN_REPRODUCTION);
  assert.equal(dawnReproductionForGPUFailure(
    "GPU runtime stopped: command submission failed",
    { ...mini32Configuration, methodOverrides: { ...mini32Configuration.methodOverrides, maximumLeafSize: "16" } },
  ), undefined, "a tuned browser scene must not advertise the authored-profile reproducer");
  assert.equal(dawnReproductionForGPUFailure(
    "GPU runtime stopped: command submission failed",
    { ...mini32Configuration, maxDt_s: 0.008 },
  ), undefined, "a different outer step is a different command graph");
});

test("the isolated smoke identifies the same case only from the exact UI contract", () => {
  assert.deepEqual(dawnReproductionForSmokeEnvironment(exactEnvironment),
    DAM_UI_T0_DAWN_REPRODUCTION);
  for (const patch of [
    { FLUID_SCENE: "minimal-power-dam-break" },
    { FLUID_EXPECT_GRID: "16,16,16" },
    { FLUID_EXPECT_EXACT_STEPS: "1" },
    { FLUID_WEBGPU_DAWN_FEATURES: "skip_validation" },
  ]) {
    assert.equal(dawnReproductionForSmokeEnvironment({ ...exactEnvironment, ...patch }), undefined);
  }
  assert.deepEqual(dawnReproductionForSmokeEnvironment({
    ...exactEnvironment,
    FLUID_TARGET_S: "1.52",
    FLUID_ORACLE_STEPS: "190",
    FLUID_EXPECT_EXACT_STEPS: "190",
  }), DAM_UI_RUNTIME_DAWN_REPRODUCTION);
  const mini32Environment = {
    FLUID_SCENE: "minimal-power-dam-break-32",
    FLUID_METHOD: "octree",
    FLUID_QUALITY: "balanced",
    FLUID_TARGET_S: "0.008",
    FLUID_MAX_DT: "0.004",
    FLUID_ORACLE_STEPS: "2",
    FLUID_EXPECT_EXACT_STEPS: "2",
    FLUID_EXPECT_GRID: "32,32,32",
    FLUID_COARSE_BACKEND: "losasso",
    FLUID_MAXIMUM_LEAF_SIZE: "32",
    FLUID_OCTREE_INTERFACE_BAND: "3",
    FLUID_OCTREE_SURFACE_GRADING: "3",
    FLUID_OCTREE_GLOBAL_FINE_FACTOR: "1",
  };
  assert.deepEqual(dawnReproductionForSmokeEnvironment(mini32Environment),
    MINIMAL_POWER_DAM_32_T0_DAWN_REPRODUCTION);
  assert.deepEqual(dawnReproductionForSmokeEnvironment({
    ...mini32Environment,
    FLUID_TARGET_S: "0.276",
    FLUID_ORACLE_STEPS: "69",
    FLUID_EXPECT_EXACT_STEPS: "69",
  }), MINIMAL_POWER_DAM_32_RUNTIME_DAWN_REPRODUCTION);
});

test("the UI and isolated launcher expose the shared case without a debug switch", () => {
  const lab = readFileSync(new URL("../components/FluidLab.tsx", import.meta.url), "utf8");
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const launcher = readFileSync(new URL("../tools/run-webgpu-smoke-isolated.ts", import.meta.url), "utf8");
  assert.match(viewport, /dawnReproductionForGPUFailure\(label, \{/);
  assert.match(lab, /data-testid="gpu-failure-reproduction"/);
  assert.match(lab, /gpuStatus\.reproduction\.caseId/);
  assert.match(lab, /gpuStatus\.reproduction\.command/);
  assert.match(launcher, /dawnReproductionForSmokeEnvironment\(process\.env\)/);
  assert.match(launcher, /\.\.\.\(reproduction \? \{ reproduction \} : \{\}\)/);
  const packageJson = readFileSync(new URL("../package.json", import.meta.url), "utf8");
  assert.match(packageJson,
    /"test:webgpu:dam-ui-runtime":\s*"[^"]*FLUID_TARGET_S=1\.52[^"]*FLUID_EXPECT_EXACT_STEPS=190/);
  assert.doesNotMatch(packageJson.match(/"test:webgpu:dam-ui-runtime":[^\n]*/)?.[0] ?? "",
    /skip_validation/,
    "the UI runtime reproducer must retain Dawn validation");
  for (const script of [
    "test:webgpu:minimal-power-dam-32-two-step",
    "test:webgpu:minimal-power-dam-32-runtime",
  ]) {
    const command = (JSON.parse(packageJson) as { scripts: Record<string, string> }).scripts[script];
    assert.match(command, /FLUID_SCENE=minimal-power-dam-break-32/);
    assert.match(command, /FLUID_MAX_DT=0\.004/);
    assert.match(command, /FLUID_EXPECT_GRID=32,32,32/);
    assert.match(command, /FLUID_COARSE_BACKEND=losasso/);
    assert.match(command, /FLUID_MAXIMUM_LEAF_SIZE=32/);
    assert.match(command, /FLUID_OCTREE_INTERFACE_BAND=3/);
    assert.match(command, /FLUID_OCTREE_SURFACE_GRADING=3/);
    assert.match(command, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=1/);
    assert.doesNotMatch(command, /skip_validation/);
  }
  assert.doesNotMatch(`${lab}\n${viewport}\n${launcher}`, /FLUID_UI_FAILURE_REPRO/);
});

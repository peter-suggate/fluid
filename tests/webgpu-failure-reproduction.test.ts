import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  DAM_UI_RUNTIME_DAWN_REPRODUCTION,
  DAM_UI_T0_DAWN_REPRODUCTION,
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

test("browser failures map to the smallest serialized exact Dawn case", () => {
  for (const message of [
    "GPU initialization failed: invalid compute pipeline",
    "Initial sparse authority section5-face-band-transitions validation failed",
    "Paused t=0 authority rejected: Section 5 did not publish",
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
    FLUID_TARGET_S: "2.232",
    FLUID_ORACLE_STEPS: "279",
    FLUID_EXPECT_EXACT_STEPS: "279",
  }), DAM_UI_RUNTIME_DAWN_REPRODUCTION);
});

test("the UI and isolated launcher expose the shared case without a debug switch", () => {
  const lab = readFileSync(new URL("../components/FluidLab.tsx", import.meta.url), "utf8");
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const launcher = readFileSync(new URL("../tools/run-webgpu-smoke-isolated.ts", import.meta.url), "utf8");
  assert.match(viewport, /dawnReproductionForGPUFailure\(label\)/);
  assert.match(lab, /data-testid="gpu-failure-reproduction"/);
  assert.match(lab, /gpuStatus\.reproduction\.caseId/);
  assert.match(lab, /gpuStatus\.reproduction\.command/);
  assert.match(launcher, /dawnReproductionForSmokeEnvironment\(process\.env\)/);
  assert.match(launcher, /\.\.\.\(reproduction \? \{ reproduction \} : \{\}\)/);
  const packageJson = readFileSync(new URL("../package.json", import.meta.url), "utf8");
  assert.match(packageJson,
    /"test:webgpu:dam-ui-runtime":\s*"[^"]*FLUID_TARGET_S=2\.232[^"]*FLUID_EXPECT_EXACT_STEPS=279/);
  assert.doesNotMatch(packageJson.match(/"test:webgpu:dam-ui-runtime":[^\n]*/)?.[0] ?? "",
    /skip_validation/,
    "the UI runtime reproducer must retain Dawn validation");
  assert.doesNotMatch(`${lab}\n${viewport}\n${launcher}`, /FLUID_UI_FAILURE_REPRO|legacy/i);
});

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { getScenePreset } from "../lib/scenes";
import { effectiveSimulationStep_s, uniformPaperStepActive } from "../lib/simulation-step";
import { UNIFORM_PAPER_DT_S, uniformPaperAdvanceReady } from "../lib/uniform-paper";

const scene = getScenePreset("minimal-power-dam-break-64").create();

test("uniform paper mode owns the browser target clock at exactly 1/30 s", () => {
  const method = { methodId: "uniform", quality: "balanced" as const, overrides: {} };
  assert.equal(scene.numerics.fixedDt_s, 0.004, "the regression scene retains its authored comparison step");
  assert.equal(effectiveSimulationStep_s(scene, method), UNIFORM_PAPER_DT_S);
  assert.equal(uniformPaperStepActive(method), true);
  assert.equal(uniformPaperAdvanceReady(0.004, 0), false, "a 4 ms browser target must not encode a fractional paper step");
  assert.equal(uniformPaperAdvanceReady(UNIFORM_PAPER_DT_S, 0), true);
});

test("uniform scene mode and other methods retain the scene-authored clock", () => {
  const sceneMode = {
    methodId: "uniform",
    quality: "balanced" as const,
    overrides: { uniform: { timeStep: "scene" } },
  };
  assert.equal(effectiveSimulationStep_s(scene, sceneMode), 0.004);
  assert.equal(uniformPaperStepActive(sceneMode), false);
  assert.equal(effectiveSimulationStep_s(scene, {
    methodId: "octree", quality: "balanced", overrides: {},
  }), 0.004);
});

test("the browser controller and transport UI consume the effective method clock", () => {
  const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  const transport = readFileSync(new URL("../components/TransportBar.tsx", import.meta.url), "utf8");
  assert.match(controller, /const dt = effectiveSimulationStep_s\(scene, useMethodStore\.getState\(\)\);/);
  assert.match(transport, /const fixedDt = effectiveSimulationStep_s\(scene, methodState\);/);
  assert.match(transport, /disabled=\{paperStep\}/);
});

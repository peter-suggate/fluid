import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { cloneScene } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { effectiveSimulationStep_s, uniformPaperStepActive } from "../lib/simulation-step";
import { simulation } from "../lib/simulation/controller";
import { resolvedMethodValues, useMethodStore } from "../lib/stores/method-store";
import { useSceneStore } from "../lib/stores/scene-store";
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
  assert.doesNotMatch(transport, /disabled=\{paperStep\}/,
    "the step control stays live in paper mode; editing it is how the paper step is released");
});

test("editing the shared step releases the uniform paper step", () => {
  const { methodId, quality, overrides } = useMethodStore.getState();
  const originalScene = cloneScene(useSceneStore.getState().scene);
  const paperMode = () => useMethodStore.setState({ methodId: "uniform", quality: "balanced", overrides: {} });
  try {
    paperMode();
    assert.equal(uniformPaperStepActive(useMethodStore.getState()), true);

    simulation.setStepSize(0.006);
    assert.equal(resolvedMethodValues(useMethodStore.getState()).timeStep, "scene");
    assert.equal(uniformPaperStepActive(useMethodStore.getState()), false);
    assert.equal(useSceneStore.getState().scene.numerics.fixedDt_s, 0.006);
    assert.equal(useSceneStore.getState().scene.numerics.maxDt_s, 0.006);
    assert.equal(effectiveSimulationStep_s(useSceneStore.getState().scene, useMethodStore.getState()), 0.006);

    // The displayed paper step is 1/30 s, not the scene's dt, so the first
    // edit often asks for the step the scene already carries. That request
    // still has to leave paper mode instead of short-circuiting as a no-op.
    paperMode();
    simulation.setStepSize(0.006);
    assert.equal(uniformPaperStepActive(useMethodStore.getState()), false);
    assert.equal(useSceneStore.getState().scene.numerics.fixedDt_s, 0.006);
  } finally {
    useMethodStore.setState({ methodId, quality, overrides });
    simulation.reset(originalScene);
  }
});

import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene } from "../lib/model";
import { simulation } from "../lib/simulation/controller";
import { useRuntimeStore } from "../lib/stores/runtime-store";
import { useSceneStore } from "../lib/stores/scene-store";

test("adding a rigid body does not pause a running simulation", () => {
  const originalScene = cloneScene(useSceneStore.getState().scene);
  const originalRunState = useRuntimeStore.getState().runState;

  try {
    useRuntimeStore.getState().setRunState("running");
    simulation.addBody("sphere");

    assert.equal(useRuntimeStore.getState().runState, "running");
  } finally {
    simulation.reset(originalScene);
    useRuntimeStore.getState().setRunState(originalRunState);
  }
});

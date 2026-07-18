import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene } from "../lib/model";
import { createBodyDescription } from "../lib/rigid-body";
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

test("editing rigid-body properties preserves its current position", () => {
  const originalScene = cloneScene(useSceneStore.getState().scene);
  const originalRunState = useRuntimeStore.getState().runState;

  try {
    const scene = cloneScene(originalScene);
    scene.rigidBodies = [createBodyDescription("sphere", 1, scene.container.height_m)];
    simulation.reset(scene);

    const bodyId = scene.rigidBodies[0].id;
    const position = { x: 0.17, y: 0.42, z: -0.11 };
    simulation.dragBody(bodyId, position, { x: 0, y: 0, z: 0 }, "end");
    useRuntimeStore.getState().setRunState("running");

    simulation.updateBody(bodyId, { density_kg_m3: 725 });
    assert.deepEqual(simulation.currentBodies()[0].position_m, position);
    assert.equal(useRuntimeStore.getState().runState, "running");

    simulation.updateBody(bodyId, { dimensions_m: { x: 0.2, y: 0.2, z: 0.2 } });
    assert.deepEqual(simulation.currentBodies()[0].position_m, position);
    assert.equal(useRuntimeStore.getState().runState, "running");
  } finally {
    simulation.reset(originalScene);
    useRuntimeStore.getState().setRunState(originalRunState);
  }
});

test("the running clock prepares all elapsed fixed steps without frame caps", () => {
  const originalScene = cloneScene(useSceneStore.getState().scene);
  const originalRunState = useRuntimeStore.getState().runState;

  try {
    const scene = cloneScene(originalScene);
    scene.rigidBodies = [];
    scene.numerics.fixedDt_s = 0.004;
    simulation.reset(scene);
    useRuntimeStore.getState().setRunState("running");

    simulation.tick(1_000);
    simulation.tick(1_100);

    assert.ok(Math.abs(simulation.time() - 0.1) < 1e-9, "100 ms of wall time should prepare 25 fixed steps");
  } finally {
    simulation.reset(originalScene);
    useRuntimeStore.getState().setRunState(originalRunState);
  }
});

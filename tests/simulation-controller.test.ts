import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene } from "../lib/model";
import { createBodyDescription } from "../lib/rigid-body";
import { simulation } from "../lib/simulation/controller";
import { useRuntimeStore } from "../lib/stores/runtime-store";
import { useSceneStore } from "../lib/stores/scene-store";
import { useDiagnosticsStore } from "../lib/stores/diagnostics-store";
import type { GPUEulerianInfo } from "../lib/webgpu-eulerian";
import { useMethodStore } from "../lib/stores/method-store";

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

test("pause discards unsubmitted GPU debt but retains admitted work", () => {
  const originalScene = cloneScene(useSceneStore.getState().scene);
  const originalRunState = useRuntimeStore.getState().runState;
  const originalGPUStatus = useDiagnosticsStore.getState().gpuStatus;
  const originalGPUInfo = useDiagnosticsStore.getState().gpuInfo;

  try {
    const scene = cloneScene(originalScene);
    scene.rigidBodies = [];
    scene.numerics.fixedDt_s = 0.004;
    simulation.reset(scene);
    useDiagnosticsStore.getState().set({
      gpuStatus: { state: "ready", label: "test GPU ready", adapter: "test" },
      gpuInfo: { initialSparseAuthorityReady: true, initialRasterSurfaceReady: true } as GPUEulerianInfo,
    });
    useRuntimeStore.getState().setRunState("running");

    simulation.tick(1_000);
    simulation.tick(1_100);

    assert.ok(Math.abs(simulation.time() - 0.1) < 1e-9, "the renderer should receive enough prepared work to fill each frame budget");
    simulation.gpuSchedulingPaused(0.012);
    assert.ok(Math.abs(simulation.time() - 0.012) < 1e-9, "only already-submitted GPU work should survive pause");
  } finally {
    simulation.reset(originalScene);
    useDiagnosticsStore.getState().set({ gpuStatus: originalGPUStatus, gpuInfo: originalGPUInfo });
    useRuntimeStore.getState().setRunState(originalRunState);
  }
});

test("startup cannot advance the WebGPU target clock before t=0 authority is ready", () => {
  const originalScene = cloneScene(useSceneStore.getState().scene);
  const originalRunState = useRuntimeStore.getState().runState;
  const originalGPUStatus = useDiagnosticsStore.getState().gpuStatus;
  try {
    simulation.reset(cloneScene(originalScene));
    useDiagnosticsStore.getState().set({ gpuStatus: { state: "initializing", label: "Warming t=0 authority", kind: "startup" } });
    useRuntimeStore.getState().setRunState("running");
    simulation.tick(1_000);
    simulation.tick(1_100);
    simulation.singleStep();
    assert.equal(simulation.time(), 0);
  } finally {
    simulation.reset(originalScene);
    useDiagnosticsStore.getState().set({ gpuStatus: originalGPUStatus });
    useRuntimeStore.getState().setRunState(originalRunState);
  }
});

test("reset publishes an atomic t=0 epoch before stale GPU completions can land", () => {
  const originalScene = cloneScene(useSceneStore.getState().scene);
  const originalRunState = useRuntimeStore.getState().runState;
  const beforeEpoch = useRuntimeStore.getState().simulationEpoch;

  try {
    useRuntimeStore.getState().setSimulationTime(12.5);
    simulation.reset(cloneScene(originalScene));
    const runtime = useRuntimeStore.getState();
    assert.equal(simulation.time(), 0);
    assert.equal(runtime.simulationTime, 0);
    assert.equal(runtime.simulationEpoch, beforeEpoch + 1);
    assert.equal(runtime.runState, "paused");
  } finally {
    simulation.reset(originalScene);
    useRuntimeStore.getState().setRunState(originalRunState);
  }
});

test("safe browser bring-up admits one step and rejects continuous running", (t) => {
  const originalScene = cloneScene(useSceneStore.getState().scene);
  const originalRunState = useRuntimeStore.getState().runState;
  const originalGPUStatus = useDiagnosticsStore.getState().gpuStatus;
  const originalGPUInfo = useDiagnosticsStore.getState().gpuInfo;
  const originalMethodId = useMethodStore.getState().methodId;
  const previousLocation = Object.getOwnPropertyDescriptor(globalThis, "location");
  Object.defineProperty(globalThis, "location", { configurable: true, value: { search: "?gpu=safe" } });
  t.after(() => {
    if (previousLocation) Object.defineProperty(globalThis, "location", previousLocation);
    else Reflect.deleteProperty(globalThis, "location");
    useMethodStore.getState().setMethodId(originalMethodId);
    simulation.reset(originalScene);
    useDiagnosticsStore.getState().set({ gpuStatus: originalGPUStatus, gpuInfo: originalGPUInfo });
    useRuntimeStore.getState().setRunState(originalRunState);
  });

  useMethodStore.getState().setMethodId("octree");
  simulation.reset(cloneScene(originalScene));
  useDiagnosticsStore.getState().set({
    gpuStatus: { state: "ready", label: "test GPU ready", adapter: "test" },
    gpuInfo: { initialSparseAuthorityReady: true, initialRasterSurfaceReady: true } as GPUEulerianInfo,
  });
  simulation.singleStep();
  assert.equal(simulation.time(), 0.004);
  simulation.gpuAdvanceCompleted(0.004);
  simulation.singleStep();
  assert.equal(simulation.time(), 0.004, "a second explicit request must not advance the safe session");

  useRuntimeStore.getState().setRunState("running");
  simulation.tick(1_000);
  simulation.tick(1_100);
  assert.equal(useRuntimeStore.getState().runState, "paused");
  assert.equal(simulation.time(), 0.004, "continuous scheduling must remain disabled in safe mode");
});

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { cloneScene } from "../lib/model";
import { createBodyDescription } from "../lib/rigid-body";
import {
  matchingPhysicsCPUTrace,
  performanceReportCPUTrace,
  MAX_SHARED_STEP_S,
  MIN_SHARED_STEP_S,
  clampSharedStepSize,
  sharedStepNumerics,
  simulation,
} from "../lib/simulation/controller";
import { useRuntimeStore } from "../lib/stores/runtime-store";
import { useSceneStore } from "../lib/stores/scene-store";
import { useDiagnosticsStore } from "../lib/stores/diagnostics-store";
import type { GPUEulerianInfo } from "../lib/webgpu-eulerian";
import { useMethodStore } from "../lib/stores/method-store";
import { resolvedMethodValues } from "../lib/stores/method-store";
import type { PerformanceTrace } from "../lib/performance-trace";
import { gpuPhysicsPerformanceActivityFrameId } from "../lib/performance-activity";

const exactTrace = (
  domain: PerformanceTrace["domain"],
  lane: PerformanceTrace["lane"],
  sampleId: number,
  context: string,
): PerformanceTrace => ({
  sampleId,
  domain,
  lane,
  context,
  capturedAt_ms: 10,
  total_ms: 1,
  phases: [{ id: "other", label: "test", duration_ms: 1 }],
});

test("CPU encoding joins only the exact sampled GPU physics advance", () => {
  const physics = exactTrace("gpu", "physics", 7, "octree:sim-0.004000");
  const matching = exactTrace("cpu", "main-thread", 7, "octree:sim-0.004000");
  const capture = {
    sampleId: physics.sampleId,
    context: physics.context,
    frameId: gpuPhysicsPerformanceActivityFrameId(physics),
  };
  assert.equal(matchingPhysicsCPUTrace(physics, matching, capture), matching);
  assert.equal(matchingPhysicsCPUTrace(physics, matching, undefined), undefined,
    "report cadence and coincident sample numbers are not a capture identity");
  assert.equal(matchingPhysicsCPUTrace(physics,
    exactTrace("cpu", "main-thread", 8, matching.context), capture), undefined);
  assert.equal(matchingPhysicsCPUTrace(physics,
    exactTrace("cpu", "main-thread", 7, "octree:sim-0.008000"), capture), undefined);
  assert.equal(matchingPhysicsCPUTrace(physics, matching,
    { ...capture, frameId: "gpu-physics:unrelated:7" }), undefined);

  const fallback = { ...physics, context: `${physics.context}:queue-wall-fallback` };
  assert.equal(matchingPhysicsCPUTrace(fallback, matching, capture), matching,
    "hardware fallback retains the sampled advance identity even though its GPU timing is inexact");
});

test("a GPU physics report never borrows latest controller or renderer CPU traces", () => {
  const physics = exactTrace("gpu", "physics", 11, "octree:sim-0.044000");
  const controllerCPU = exactTrace("cpu", "main-thread", 100, "controller:later-frame");
  const rendererCPU = exactTrace("cpu", "main-thread", 101, "renderer:later-frame");
  assert.equal(performanceReportCPUTrace({
    physics,
    controllerCPU,
    rendererCPU,
    context: "octree",
  }), undefined, "latest callbacks are not rebased into an asynchronously completed GPU frame");

  const physicsCPU = exactTrace("cpu", "main-thread", 11, physics.context);
  const physicsCaptureIdentity = {
    sampleId: physics.sampleId,
    context: physics.context,
    frameId: gpuPhysicsPerformanceActivityFrameId(physics),
  };
  assert.equal(performanceReportCPUTrace({
    physics,
    physicsCPU,
    physicsCaptureIdentity,
    controllerCPU,
    rendererCPU,
    context: "octree",
  }), physicsCPU);

  const cpuOnly = performanceReportCPUTrace({ controllerCPU, rendererCPU, context: "octree" });
  assert.ok(cpuOnly);
  assert.equal(cpuOnly.total_ms, controllerCPU.total_ms + rendererCPU.total_ms,
    "latest callbacks remain valid for a CPU-only report");
});

test("controller source has no reporting-cadence correlation fallback", () => {
  const source = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  const recordFrame = source.slice(source.indexOf("  recordFrame("),
    source.indexOf("  // ---- persistence", source.indexOf("  recordFrame(")));
  assert.match(recordFrame,
    /physicsCaptureIdentity: diagnostics\.gpuInfo\?\.physicsCaptureIdentity/);
  assert.match(recordFrame, /activityStore\.publish\(!report\.physics/,
    "detailed controller CPU spans may only be rebased into CPU-owned frames");
  assert.doesNotMatch(recordFrame, /combineMainThreadPerformanceTraces\(/,
    "recordFrame must route correlation through the fail-closed selector");
  const selector = source.slice(source.indexOf("export function performanceReportCPUTrace"),
    source.indexOf("function rebasePerformanceActivityAddition"));
  assert.match(selector,
    /if \(input\.physics\) \{[\s\S]*return matchingPhysicsCPUTrace\([\s\S]*\);[\s\S]*\}[\s\S]*return combineMainThreadPerformanceTraces\(/,
    "latest CPU callbacks are combined only after the GPU-physics branch has returned");
});

test("Fast Refresh retains the controller paired with the retained WebGPU viewport", () => {
  const source = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  assert.match(source, /__fluidLabSimulationController/,
    "transport and the retained render loop must keep sharing one controller after a module refresh");
  assert.match(source, /Object\.setPrototypeOf\(retainedSimulation, SimulationController\.prototype\)/,
    "the retained controller must adopt edited methods without losing its live simulation clock");
  assert.match(source, /retainedSimulation \?\? new SimulationController\(\)/);
});

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

test("one bounded step size drives both rigid and fluid numerics", () => {
  assert.equal(clampSharedStepSize(Number.NaN), 0.004);
  assert.equal(clampSharedStepSize(0), MIN_SHARED_STEP_S);
  assert.equal(clampSharedStepSize(99), MAX_SHARED_STEP_S);
  assert.equal(clampSharedStepSize(0.0064), 0.006);
  const numerics = sharedStepNumerics(useSceneStore.getState().scene.numerics, 0.006);
  assert.equal(numerics.fixedDt_s, 0.006);
  assert.equal(numerics.maxDt_s, 0.006);
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

test("loading the minimal power dam applies its Losasso scene profile", () => {
  const originalScene = cloneScene(useSceneStore.getState().scene);
  const originalRunState = useRuntimeStore.getState().runState;
  const originalMethod = useMethodStore.getState();
  try {
    originalMethod.setMethodId("octree");
    originalMethod.setQuality("ultra");
    originalMethod.setParam("octree", "maximumLeafSize", "16");

    simulation.loadPreset("minimal-power-dam-break");

    const method = useMethodStore.getState();
    const values = resolvedMethodValues(method);
    assert.equal(method.methodId, "octree");
    assert.equal(method.quality, "balanced");
    assert.equal(values.coarseBackend, "losasso");
    assert.equal(values.maximumLeafSize, "16");
    assert.equal(values.interfaceRefinementBandCells, 3);
    assert.equal(values.globalFineLevelSetFactor, "4");
  } finally {
    useMethodStore.setState({
      methodId: originalMethod.methodId,
      quality: originalMethod.quality,
      overrides: originalMethod.overrides,
    });
    simulation.reset(originalScene);
    useRuntimeStore.getState().setRunState(originalRunState);
  }
});

test("coarse-only surface tracking survives scene profile changes", () => {
  const originalScene = cloneScene(useSceneStore.getState().scene);
  const originalRunState = useRuntimeStore.getState().runState;
  const originalMethod = useMethodStore.getState();
  try {
    originalMethod.setMethodId("octree");
    originalMethod.setParam("octree", "globalFineLevelSetFactor", "1");

    simulation.loadPreset("symmetric-expansion");
    assert.equal(
      resolvedMethodValues(useMethodStore.getState()).globalFineLevelSetFactor,
      "1",
      "opening a factor-4 validation scene must not silently re-enable its fine band",
    );

    simulation.loadPreset("minimal-power-dam-break");
    assert.equal(
      resolvedMethodValues(useMethodStore.getState()).globalFineLevelSetFactor,
      "1",
      "the user's coarse-only execution shape must apply to every octree scene",
    );
  } finally {
    useMethodStore.setState({
      methodId: originalMethod.methodId,
      quality: originalMethod.quality,
      overrides: originalMethod.overrides,
    });
    simulation.reset(originalScene);
    useRuntimeStore.getState().setRunState(originalRunState);
  }
});

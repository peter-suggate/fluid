/** Instruments worker: app-default narrow-band FLIP, with an optional attach gate. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { createPassEncoderIsolationScratch, isolateComputePassEncoders } from "../lib/harness/webgpu-pass-encoder-isolation";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";

for (const signal of ["SIGTERM", "SIGINT"] as const) process.once(signal, () => process.exit(124));
const dawn = await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE ?? "node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const toggles = (process.env.FLUID_WEBGPU_DAWN_FEATURES ?? "").split(",").filter(Boolean);
const gpu = createProcessRetainedDawnGPU(dawn, [
  `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
  ...(toggles.length ? [`enable-dawn-features=${toggles.join(",")}`] : []),
]);
const adapter = await gpu.requestAdapter(); assert.ok(adapter);
const raw = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const errors: string[] = [];
raw.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
const isolate = process.env.FLUID_GPU_ISOLATE_PASS_ENCODERS === "1";
const scratch = isolate ? createPassEncoderIsolationScratch(raw) : undefined;
const wrapped = scratch ? new Proxy(raw, { get(target, key) {
  if (key === "createCommandEncoder") return (descriptor?: GPUCommandEncoderDescriptor) =>
    isolateComputePassEncoders(target.createCommandEncoder(descriptor), scratch);
  const value = Reflect.get(target, key, target);
  return typeof value === "function" ? value.bind(target) : value;
} }) : raw;
const device = managedGPUDevice(wrapped, { requireWorkerRealm: false });
const scene = structuredClone(sceneDocument(getSceneDefinition(process.env.FLUID_SCENE ?? "minimal-power-dam-break-64")));
const dt = Number(process.env.FLUID_MAX_DT ?? .017);
const steps = Number(process.env.FLUID_ORACLE_STEPS ?? 120);
assert.ok(Number.isInteger(steps) && steps > 0 && Number.isFinite(dt) && dt > 0);
scene.numerics = { ...scene.numerics, fixedDt_s: dt, maxDt_s: dt };
const started = performance.now();
const solver = await uniformNarrowBandMethod.createSolverAsync!(device, scene, "balanced",
  { ...uniformNarrowBandMethod.appDefaults, timeStep: "scene" }, undefined, () => {});
try {
  const construction_ms = performance.now() - started;
  const expected = process.env.FLUID_EXPECT_GRID?.split(",").map(Number);
  if (expected) assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], expected);
  console.log(JSON.stringify({ phase: "constructed", construction_ms }));
  if (process.env.FLUID_PROFILE_FIRST_ADVANCE_GATE === "1") {
    await new Promise<void>(resolveGate => {
      process.once("SIGUSR2", resolveGate);
      process.once("SIGUSR1", resolveGate);
      console.log(JSON.stringify({ phase: "before-first-advance", profileGate: "waiting-for-sigusr2" }));
    });
  }
  const advancing = performance.now();
  for (let frame = 1; frame <= steps; frame++) {
    solver.advanceTo(frame * dt, []);
    await solver.awaitFrameCompletion?.();
    assert.equal(solver.info.simulationPipelineError, undefined);
    assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - frame * dt) < 1e-8,
      "each profiling advance must complete the requested simulation clock");
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ phase: "result", construction_ms, steps,
    simulationWall_ms: performance.now() - advancing }));
} finally {
  solver.destroy(); await raw.queue.onSubmittedWorkDone(); scratch?.destroy(); raw.destroy();
}

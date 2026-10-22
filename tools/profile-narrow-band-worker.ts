/** Instruments worker: app-default narrow-band FLIP, with an optional attach gate. */
import { NarrowBandDispatchProbe } from "./narrow-band-dispatch-probe";
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { createPassEncoderIsolationScratch, isolateComputePassEncoders } from "../lib/harness/webgpu-pass-encoder-isolation";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";

// Keep the capture harness fixed while profiling historical solver sources.
// This avoids copying modern solver helpers into the pre-regression baseline.
const sourceRoot = resolve(process.env.FLUID_PROFILE_SOURCE_ROOT ?? ".");
// Optional native Metal capture bridge for compiled-pipeline inspection. This
// is confined to the diagnostic worker; ordinary profiling never loads it.
const captureFrame = Number(process.env.FLUID_METAL_CAPTURE_FRAME ?? 0);
const targetFrame = Number(process.env.FLUID_PROFILE_TARGET_FRAME ?? 128);
const captureBridge: { start(path: string): boolean; stop(): void } | undefined =
  process.env.FLUID_METAL_CAPTURE_BRIDGE ? createRequire(import.meta.url)(process.env.FLUID_METAL_CAPTURE_BRIDGE) : undefined;
const sourceModule = (path: string) => import(pathToFileURL(resolve(sourceRoot, path)).href);
const [{ managedGPUDevice }, { sceneDocument }, { getSceneDefinition }, { uniformNarrowBandMethod }] = await Promise.all([
  sourceModule("lib/core/gpu-compilation-manager.ts"),
  sourceModule("lib/core/scene-definition.ts"),
  sourceModule("lib/core/scenes.ts"),
  sourceModule("lib/methods/uniform/uniform-narrow-band-method.ts"),
]);

for (const signal of ["SIGTERM", "SIGINT"] as const) process.once(signal, () => process.exit(124));
const dawn = await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE ?? "node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const toggles = (process.env.FLUID_WEBGPU_DAWN_FEATURES ?? "").split(",").filter(Boolean);
const gpu = createProcessRetainedDawnGPU(dawn, [
  `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
  ...(toggles.length ? [`enable-dawn-features=${toggles.join(",")}`] : []),
]);
const adapter = await gpu.requestAdapter(); assert.ok(adapter);
const raw = await adapter.requestDevice({ requiredFeatures: adapter.features.has("subgroups") ? ["subgroups"] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const errors: string[] = [];
raw.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
const isolate = process.env.FLUID_GPU_ISOLATE_PASS_ENCODERS === "1";
const labelFrames = process.env.FLUID_PROFILE_FRAME_LABELS === "1";
let simulationFrame = 0;
const scratch = isolate ? createPassEncoderIsolationScratch(raw) : undefined;
const wrapped = scratch || labelFrames ? new Proxy(raw, { get(target, key) {
  if (labelFrames && (key === "createComputePipeline" || key === "createComputePipelineAsync")) {
    return (descriptor: GPUComputePipelineDescriptor) => target[key]({ ...descriptor,
      label: `${descriptor.label ?? "Compute"} / ${descriptor.compute.entryPoint ?? "main"}`,
    });
  }
  if (key === "createCommandEncoder") return (descriptor?: GPUCommandEncoderDescriptor) => {
    const encoder = target.createCommandEncoder(descriptor);
    const isolated = scratch ? isolateComputePassEncoders(encoder, scratch) : encoder;
    if (!labelFrames) return isolated;
    return new Proxy(isolated, { get(command, property) {
      if (property === "beginComputePass") return (pass?: GPUComputePassDescriptor) =>
        command.beginComputePass({ ...pass, label: `[frame ${simulationFrame}] ${pass?.label ?? "(unlabelled)"}` });
      const value = Reflect.get(command, property, command);
      return typeof value === "function" ? value.bind(command) : value;
    } });
  };
  const value = Reflect.get(target, key, target);
  return typeof value === "function" ? value.bind(target) : value;
} }) : raw;
// FLUID_PROFILE_SPLIT_DISPATCHES=1: one labelled pass per dispatch (the
// dispatch probe over every pass), so counters attribute to single kernels.
// Splitting serialises the passes: read occupancy and limiters, not frame time.
const probe = process.env.FLUID_PROFILE_SPLIT_DISPATCHES === "1" ? new NarrowBandDispatchProbe(wrapped) : undefined;
if (probe) { process.env.NB_SPLIT_ALL = "1"; probe.enabled = true; }
const device = managedGPUDevice(probe?.device ?? wrapped, { requireWorkerRealm: false });
const scene = structuredClone(sceneDocument(getSceneDefinition(process.env.FLUID_SCENE ?? "minimal-power-dam-break-64")));
const dt = Number(process.env.FLUID_MAX_DT ?? .017);
const steps = Number(process.env.FLUID_ORACLE_STEPS ?? 120);
const warmupSteps=Number(process.env.FLUID_PROFILE_WARMUP_STEPS??0);
const transfer=process.env.FLUID_PROFILE_TRANSFER;
assert.ok(Number.isInteger(warmupSteps)&&warmupSteps>=0);
assert.ok(Number.isInteger(targetFrame) && targetFrame > 0);
assert.ok(Number.isInteger(captureFrame) && captureFrame >= 0);
if (captureBridge) {
  assert.ok(captureFrame > warmupSteps && captureFrame <= warmupSteps + steps,
    "native capture frame must be in the measured advances");
  assert.ok(process.env.FLUID_METAL_CAPTURE_OUTPUT);
}
assert.ok(transfer===undefined||transfer==="flip"||transfer==="apic");
assert.ok(Number.isInteger(steps) && steps > 0 && Number.isFinite(dt) && dt > 0);
scene.numerics = { ...scene.numerics, fixedDt_s: dt, maxDt_s: dt };
const started = performance.now();
const solver = await uniformNarrowBandMethod.createSolverAsync!(device, scene, "balanced",
  { ...uniformNarrowBandMethod.appDefaults, timeStep: "scene",...(transfer?{particleTransfer:transfer}:{}) }, undefined, () => {});
// The sealed schedule shares one native pass across stages, past the probe.
if (probe) { const plan = (solver as unknown as { mixedFrame: { surfaceWorkControl?: { framePlan: { coalescePasses: boolean } } } }).mixedFrame.surfaceWorkControl?.framePlan; if (plan) plan.coalescePasses = false; }
try {
  const construction_ms = performance.now() - started;
  const expected = process.env.FLUID_EXPECT_GRID?.split(",").map(Number);
  if (expected) assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], expected);
  console.log(JSON.stringify({ phase: "constructed", construction_ms }));
  // Warm the requested physical state before Instruments attaches. These
  // advances are outside the capture and its reported stepping interval.
  for(let frame=1;frame<=warmupSteps;frame++){
    simulationFrame = frame;
    assert.ok(solver.advanceTo(frame*dt,[]));await solver.awaitFrameCompletion?.();
    assert.equal(solver.info.simulationPipelineError,undefined);
  }
  console.log(JSON.stringify({phase:"capture-state",sourceRoot,warmupSteps,time_s:warmupSteps*dt,transfer}));
  if (process.env.FLUID_PROFILE_FIRST_ADVANCE_GATE === "1") {
    await new Promise<void>(resolveGate => {
      process.once("SIGUSR2", resolveGate);
      process.once("SIGUSR1", resolveGate);
      console.log(JSON.stringify({ phase: "before-first-advance", profileGate: "waiting-for-sigusr2" }));
    });
  }
  const advancing = performance.now();
  for (let frame = 1; frame <= steps; frame++) {
    simulationFrame = warmupSteps + frame;
    const capture = simulationFrame === captureFrame ? captureBridge : undefined;
    if (capture) capture.start(process.env.FLUID_METAL_CAPTURE_OUTPUT!);
    try {
      solver.advanceTo((warmupSteps+frame) * dt, []);
      await solver.awaitFrameCompletion?.();
    } finally {
      capture?.stop();
    }
    assert.equal(solver.info.simulationPipelineError, undefined);
    assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - (warmupSteps+frame) * dt) < 1e-8,
      "each profiling advance must complete the requested simulation clock");
    if (simulationFrame === targetFrame) console.log(JSON.stringify({
      phase: "target-frame", frame: simulationFrame, time_s: solver.info.completedTime_s,
      ...solver.workDomainInfo, ...solver.narrowBandFlipInfo,
      fineTiles: solver.workDomainInfo?.fineTiles ?? solver.info.uniformMixedFineTiles,
      bandTiles: solver.info.uniformPressureBandTiles,
      pages: solver.pagePoolInfo ? { ...solver.pagePoolInfo } : undefined,
    }));
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ phase: "result", construction_ms, steps,warmupSteps,dt,transfer,
    simulationWall_ms: performance.now() - advancing }));
} finally {
  solver.destroy(); await raw.queue.onSubmittedWorkDone(); scratch?.destroy(); raw.destroy();
}

/** Solver-only loading census. Run methods in separate processes, without browser GPU work.
 * FLUID_GPU_INIT_CENSUS=1 WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js
 * node --import tsx tools/probe-scene-loading-dawn.ts --method=adaptive-volume
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { getScenePreset } from "../lib/core/scenes";
import { resolveMethodValues, type GPUSolverInstance } from "../lib/core/method-contract";
import { readInitializationCensus } from "../lib/core/gpu-initialization";
import { gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { fluidExecutionDeviceFeatures } from "../lib/core/gpu-startup";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import { uniformMethod } from "../lib/methods/uniform/method";

const arg = (name: string, fallback: string) => process.argv.find(v => v.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const methodId = arg("method", "adaptive-volume");
assert.ok(["adaptive-volume", "uniform"].includes(methodId));
const method = methodId === "adaptive-volume" ? adaptiveMassMethod : uniformMethod;
const sceneId = arg("scene", "coarse-first-pool-impact-half-slab");
const nextSceneId = arg("next-scene", sceneId);
const repeats = Number(arg("repeats", "2"));
assert.ok(Number.isInteger(repeats) && repeats > 0);
const output = resolve(arg("out", `artifacts/scene-loading/${methodId}.json`));
const report: Record<string, unknown> = { methodId, sceneId, nextSceneId, repeats, backend: process.env.FLUID_WEBGPU_BACKEND ?? "metal", runs: [] };
const runs = report.runs as Record<string, unknown>[];
const save = () => { mkdirSync(dirname(output), { recursive: true }); writeFileSync(output, JSON.stringify(report, null, 2) + "\n"); };
await acquireWebGPUExclusiveLock("dawn-probe", `scene loading ${methodId}`);
let device: GPUDevice | undefined;
let solver: GPUSolverInstance | undefined;
let heartbeat: ReturnType<typeof setInterval> | undefined;
try {
  const dawn = await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE ?? "node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${report.backend}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  report.adapter = adapter.info;
  device = await adapter.requestDevice({ requiredFeatures: fluidExecutionDeviceFeatures(adapter.features), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  const modules: { label: string; characters: number; elapsed_ms: number }[] = [];
  const pipelines: { label: string; entryPoint: string; elapsed_ms: number; kind: string }[] = [];
  const nativeModule = device.createShaderModule.bind(device);
  device.createShaderModule = descriptor => {
    const started = performance.now();
    try { return nativeModule(descriptor); }
    finally { modules.push({ label: descriptor.label ?? "", characters: descriptor.code.length, elapsed_ms: performance.now() - started }); }
  };
  const nativeAsync = device.createComputePipelineAsync.bind(device);
  device.createComputePipelineAsync = async descriptor => {
    const started = performance.now();
    try { return await nativeAsync(descriptor); }
    finally { pipelines.push({ label: descriptor.label ?? "", entryPoint: descriptor.compute.entryPoint ?? "", elapsed_ms: performance.now() - started, kind: "async" }); }
  };
  const nativeSync = device.createComputePipeline.bind(device);
  device.createComputePipeline = descriptor => {
    const started = performance.now();
    try { return nativeSync(descriptor); }
    finally { pipelines.push({ label: descriptor.label ?? "", entryPoint: descriptor.compute.entryPoint ?? "", elapsed_ms: performance.now() - started, kind: "sync" }); }
  };
  for (let repeat = 0; repeat < repeats; repeat++) {
    const currentSceneId = repeat === 0 ? sceneId : nextSceneId;
    const preset = getScenePreset(currentSceneId);
    const scene = preset.create();
    const values = resolveMethodValues(method, "balanced",
      preset.methodProfile?.methodId === methodId ? preset.methodProfile.overrides ?? {} : {});
    const run: Record<string, unknown> = { repeat, sceneId: currentSceneId, values, progress: [] };
    runs.push(run);
    const moduleStart = modules.length, pipelineStart = pipelines.length, censusStart = readInitializationCensus().length;
    const started = performance.now();
    heartbeat = setInterval(() => {
      run.elapsed_ms = performance.now() - started;
      run.compilation = gpuCompilationManagerFor(device!).snapshot();
      save(); console.log(JSON.stringify({ repeat, elapsed_ms: run.elapsed_ms, modules: modules.length - moduleStart, pipelines: pipelines.length - pipelineStart, compilation: run.compilation }));
    }, 15000);
    solver = await method.createSolverAsync!(device, scene, "balanced", values, undefined, progress => {
      (run.progress as unknown[]).push({ ...progress, at_ms: performance.now() - started });
    });
    run.presentationReady_ms = performance.now() - started;
    console.log(JSON.stringify({ repeat, presentationReady_ms: run.presentationReady_ms })); save();
    await solver.waitForSimulationReady?.();
    await device.queue.onSubmittedWorkDone();
    run.simulationReady_ms = performance.now() - started;
    await gpuCompilationManagerFor(device).whenIdle();
    clearInterval(heartbeat); heartbeat = undefined;
    run.allCompilationIdle_ms = performance.now() - started;
    run.modules = modules.slice(moduleStart);
    run.pipelines = pipelines.slice(pipelineStart);
    run.initializationCensus = readInitializationCensus().slice(censusStart);
    run.grid = { nx: solver.info.nx, ny: solver.info.ny, nz: solver.info.nz };
    run.memory = process.memoryUsage(); run.validationErrors = [...errors];
    save();
    console.log(JSON.stringify({ repeat, presentationReady_ms: run.presentationReady_ms, simulationReady_ms: run.simulationReady_ms, modules: modules.length - moduleStart, pipelines: pipelines.length - pipelineStart, validationErrors: errors.length }));
    solver.destroy(); solver = undefined;
    assert.deepEqual(errors, []);
  }
} catch (error) {
  report.error = error instanceof Error ? error.stack : String(error); process.exitCode = 1;
} finally {
  if (heartbeat) clearInterval(heartbeat);
  solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); save();
  console.log(JSON.stringify({ output, error: report.error }));
}

#!/usr/bin/env node
/** Minimal Dawn/Metal diagnostic for GPUStageTimestampRecorder's trailing marker. */
import { writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  GPUStageTimestampRecorder,
  type GPUTimestampPhase,
} from "../lib/core/performance-trace";
import {
  gpuCompilationManagerFor,
  invalidateGPUCompilationManager,
} from "../lib/core/gpu-compilation-manager";
import {
  createProcessRetainedDawnGPU,
  type NodeDawnProvider,
} from "../lib/harness/node-dawn-provider";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

const output = fileURLToPath(new URL(
  "../artifacts/sparse-cm12-stage-timestamp-trailing-marker-debug.json", import.meta.url));
const first: GPUTimestampPhase = { id: "frame-control", label: "minimal first pass" };
const trailing: GPUTimestampPhase = {
  id: "adaptive-publication", label: "minimal multipass tail",
};

await acquireWebGPUExclusiveLock("dawn-probe", "minimal trailing timestamp marker");
let device: GPUDevice | undefined;
let receipt: Record<string, unknown> = { passed: false, probe: "trailing-marker" };
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(modulePath).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const backend = process.env.FLUID_WEBGPU_BACKEND ?? "metal";
  const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${backend}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter?.features.has("timestamp-query")) throw new Error("timestamp-query unavailable");
  device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
  const validation: string[] = [];
  device.addEventListener("uncapturederror", (event) => validation.push(event.error.message));
  const manager = gpuCompilationManagerFor(device);
  await GPUStageTimestampRecorder.prepare(device);
  const module = manager.createShaderModule({ label: "minimal timestamp work", code: `
struct Word { value: atomic<u32> }
@group(0) @binding(0) var<storage, read_write> word: Word;
@compute @workgroup_size(1) fn work() { atomicAdd(&word.value, 1u); }
` });
  const pipeline = await manager.compileComputePipeline({
    label: "minimal timestamp work", layout: "auto",
    compute: { module, entryPoint: "work" },
  });
  await manager.whenIdle();
  const word = device.createBuffer({ size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const copy = device.createBuffer({ size: 4,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: word } }] });
  const recorder = new GPUStageTimestampRecorder(device, 1, "physics", "minimal-trailing");
  const raw = device.createCommandEncoder({ label: "minimal trailing timestamp" });
  const encoder = recorder.instrument(raw);
  recorder.begin();
  const pass1 = encoder.beginComputePass({ label: "minimal first" });
  pass1.setPipeline(pipeline);pass1.setBindGroup(0, bindGroup);pass1.dispatchWorkgroups(1);pass1.end();
  recorder.completePhase(encoder, first);
  encoder.copyBufferToBuffer(word, 0, copy, 0, 4);
  const pass2 = encoder.beginComputePass({ label: "minimal tail A" });
  pass2.setPipeline(pipeline);pass2.setBindGroup(0, bindGroup);pass2.dispatchWorkgroups(1);pass2.end();
  encoder.copyBufferToBuffer(copy, 0, word, 0, 4);
  const pass3 = encoder.beginComputePass({ label: "minimal tail B" });
  pass3.setPipeline(pipeline);pass3.setBindGroup(0, bindGroup);pass3.dispatchWorkgroups(1);pass3.end();
  recorder.completePhase(encoder, trailing);
  recorder.resolve(encoder);
  device.queue.submit([encoder.finish()]);
  const trace = await recorder.read();
  await device.queue.onSubmittedWorkDone();
  receipt = { passed: trace !== undefined && validation.length === 0,
    probe: "trailing-marker", backend, validation, trace: trace ?? null };
  word.destroy();copy.destroy();
} catch (error) {
  receipt = { ...receipt, error: error instanceof Error
    ? { name: error.name, message: error.message, stack: error.stack }
    : { message: String(error) } };
} finally {
  await writeFile(output, `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
  if (device) {
    const manager = gpuCompilationManagerFor(device);
    await manager.whenIdle();
    await device.queue.onSubmittedWorkDone();
    invalidateGPUCompilationManager(device, "minimal trailing marker complete");
    await manager.whenIdle();
    device.destroy();
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  await releaseWebGPUExclusiveLock();
}
process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`);
if (!receipt.passed) process.exitCode = 1;

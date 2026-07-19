import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { captureTargetForStage, encodeGPUStageTextureCapture, gpuStageCapture, type GPUStageCaptureRequest } from "../lib/gpu-stage-capture";

const baseline = { methodId: "uniform", stage_ms: 1.25, gpuTotal_ms: 4, sampleCount: 30 };
const request = (overrides: Partial<GPUStageCaptureRequest> = {}): GPUStageCaptureRequest => ({
  lane: "physics",
  stageKey: "pressure",
  resourceId: "pressure",
  label: "Pressure field",
  selector: 0,
  selectorLabel: "pressure",
  visualization: "diverging",
  units: "Pa",
  nearZero: 1e-6,
  baseline,
  ...overrides,
});

test("capture registry exposes typed resources only on supported stages and methods", () => {
  assert.equal(captureTargetForStage("uniform", "physics", "pressure")?.resourceId, "pressure");
  assert.equal(captureTargetForStage("uniform", "physics", "topology"), undefined);
  assert.equal(captureTargetForStage("octree", "physics", "topology")?.visualization, "categorical");
  assert.equal(captureTargetForStage("octree", "presentation", "dry-scene")?.resourceId, "dry-scene-hdr");
  assert.equal(captureTargetForStage("octree", "physics", "diagnostics"), undefined);
});

test("one-shot coordinator claims only the requested boundary and fences stale completion", () => {
  gpuStageCapture.cancel();
  const first = gpuStageCapture.arm(request());
  assert.equal(gpuStageCapture.matches("physics", "advection"), false);
  assert.equal(gpuStageCapture.claim("physics", "advection"), undefined);
  assert.equal(gpuStageCapture.claim("physics", "pressure")?.captureId, first);
  assert.equal(gpuStageCapture.getSnapshot().phase, "encoding");
  assert.equal(gpuStageCapture.submitted(first), true);
  assert.equal(gpuStageCapture.reading(first), true);
  const second = gpuStageCapture.arm(request({ stageKey: "projection", resourceId: "projected-velocity" }));
  gpuStageCapture.fail(first, new Error("stale"));
  assert.equal(gpuStageCapture.getSnapshot().captureId, second);
  assert.equal(gpuStageCapture.getSnapshot().phase, "armed");
  gpuStageCapture.cancel();
  const cancelledToken = gpuStageCapture.getSnapshot().captureId;
  gpuStageCapture.fail(second, new Error("cancelled capture completed late"));
  assert.equal(gpuStageCapture.getSnapshot().captureId, cancelledToken);
  assert.equal(gpuStageCapture.getSnapshot().phase, "idle");
});

const modulePath = process.env.WEBGPU_NODE_MODULE;

async function waitForCapture(captureId: number) {
  const current = gpuStageCapture.getSnapshot();
  if (current.captureId === captureId && (current.phase === "ready" || current.phase === "failed")) return current;
  return await new Promise<ReturnType<typeof gpuStageCapture.getSnapshot>>((resolve, reject) => {
    const timer = setTimeout(() => { unsubscribe(); reject(new Error("Timed out waiting for GPU stage capture")); }, 10_000);
    const unsubscribe = gpuStageCapture.subscribe(() => {
      const state = gpuStageCapture.getSnapshot();
      if (state.captureId !== captureId || (state.phase !== "ready" && state.phase !== "failed")) return;
      clearTimeout(timer); unsubscribe(); resolve(state);
    });
  });
}

test("real GPU capture summarizes a full float field and reads back one bounded slice", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU stage-capture checks",
}, async (t) => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice();
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => validationErrors.push((event as { error: { message: string } }).error.message));
  const controlSource = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const controlReadback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.queue.writeBuffer(controlSource, 0, new Uint32Array([7, 11, 13, 17]));
  const controlEncoder = device.createCommandEncoder(); controlEncoder.copyBufferToBuffer(controlSource, 0, controlReadback, 0, 16); device.queue.submit([controlEncoder.finish()]);
  await controlReadback.mapAsync(GPUMapMode.READ); assert.deepEqual(Array.from(new Uint32Array(controlReadback.getMappedRange())), [7, 11, 13, 17]); controlReadback.unmap(); controlSource.destroy(); controlReadback.destroy();
  const computeSource = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const computeReadback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const computeModule = device.createShaderModule({ code: "@group(0) @binding(0) var<storage, read_write> output: array<atomic<u32>>; @compute @workgroup_size(1) fn main() { atomicStore(&output[0], 23u); }" });
  const computePipeline = device.createComputePipeline({ layout: "auto", compute: { module: computeModule, entryPoint: "main" } });
  const computeGroup = device.createBindGroup({ layout: computePipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: computeSource } }] });
  const computeEncoder = device.createCommandEncoder(); const computePass = computeEncoder.beginComputePass(); computePass.setPipeline(computePipeline); computePass.setBindGroup(0, computeGroup); computePass.dispatchWorkgroups(1); computePass.end(); computeEncoder.copyBufferToBuffer(computeSource, 0, computeReadback, 0, 16); device.queue.submit([computeEncoder.finish()]);
  await computeReadback.mapAsync(GPUMapMode.READ); const computeProbe = new Uint32Array(computeReadback.getMappedRange())[0]; computeReadback.unmap(); computeSource.destroy(); computeReadback.destroy();
  if (computeProbe !== 23) { device.destroy(); t.skip("Dawn adapter accepted pipelines but did not execute a trivial compute dispatch"); return; }
  const texture = device.createTexture({ size: [4, 4, 2], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const packed = new Float32Array(256 / 4 * 4 * 2);
  for (let z = 0; z < 2; z += 1) for (let y = 0; y < 4; y += 1) for (let x = 0; x < 4; x += 1) packed[z * 256 + y * 64 + x] = 1 + x + y * 4 + z * 16;
  device.queue.writeTexture({ texture }, packed, { bytesPerRow: 256, rowsPerImage: 4 }, { width: 4, height: 4, depthOrArrayLayers: 2 });
  const captureId = gpuStageCapture.arm(request());
  device.pushErrorScope("validation");
  const encoder = device.createCommandEncoder();
  const pending = encodeGPUStageTextureCapture({ device, encoder, lane: "physics", stageKey: "pressure", texture, dimension: "3d", dimensions: [4, 4, 2], identity: { methodId: "uniform", sceneId: "fixture", simulationTime_s: 1 } });
  assert.ok(pending);
  device.queue.submit([encoder.finish()]); pending.afterSubmit();
  const state = await waitForCapture(captureId);
  await device.queue.onSubmittedWorkDone();
  const scopedError = await device.popErrorScope();
  assert.equal(scopedError, null, scopedError?.message);
  assert.deepEqual(validationErrors, []);
  assert.equal(state.phase, "ready", state.message);
  const artifact = state.artifact!;
  assert.equal(artifact.totalValues, 32);
  assert.equal(artifact.invalidValues, 0);
  assert.equal(artifact.minimum, 1);
  assert.equal(artifact.maximum, 32);
  assert.deepEqual([artifact.previewWidth, artifact.previewHeight], [4, 4]);
  assert.equal(artifact.histogram.reduce((sum, count) => sum + count, 0), 16);
  assert.equal(artifact.previewRgba.length, 4 * 4 * 4);
  texture.destroy(); device.destroy(); gpuStageCapture.cancel();
});

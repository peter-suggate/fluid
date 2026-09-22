import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { globalFineClassifiedIndirectScanShader } from "../lib/core/webgpu-water-global-fine-tetra";
import { parallelSurfaceScanShader, surfaceClassifyDispatchShader } from "../lib/core/webgpu-water-surface-scan";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("parallel surface offsets match the original GPU scan, including publication and capacity edges", { timeout: 180_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "water surface parallel scan");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredFeatures: adapter.features.has("timestamp-query") ? ["timestamp-query"] : [] });
    assert.ok(device);
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const buffer = (size: number, extra = 0) => device!.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra });
    const capacity = 32769;
    const output = buffer(300 * 32), args = buffer(32), cubes = buffer(capacity * 8), values = buffer(capacity * 32), offsets = buffer(capacity * 24), dispatch = buffer(24, GPUBufferUsage.INDIRECT), blocks = buffer(Math.ceil(capacity / 64) * 4);
    const params = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const p = new Uint32Array(28); p[9] = 7; p[11] = 42; device.queue.writeBuffer(params, 0, p);
    const cubeData = new Uint32Array(capacity * 2), valueData = new Float32Array(capacity * 8);
    // Every marching-cube sign case, ambiguous faces, transitions, direct
    // patches, wall faces, heightfields and prebuilt adaptive triangles.
    for (let i = 0; i < capacity; i++) {
      const descriptor = i % 13 === 0 ? 193 : i % 13 === 1 ? 192 : i % 13 === 2 ? 224 : i % 13 === 3 ? 225 : i % 13 === 4 ? (1 | 256) : i % 13 === 5 ? (128 | (63 << 8)) : 1;
      cubeData[i * 2 + 1] = descriptor << 16;
      for (let j = 0; j < 8; j++) valueData[i * 8 + j] = ((i & (1 << j)) ? .75 : .25);
    }
    device.queue.writeBuffer(cubes, 0, cubeData); device.queue.writeBuffer(values, 0, valueData);
    const entries = [{ binding: 3, resource: { buffer: output } }, { binding: 4, resource: { buffer: args } }, { binding: 5, resource: { buffer: cubes } }, { binding: 6, resource: { buffer: values } }, { binding: 7, resource: { buffer: offsets } }, { binding: 10, resource: { buffer: params } }, { binding: 11, resource: { buffer: dispatch } }, { binding: 18, resource: { buffer: blocks } }];
    const make = async (code: string, entryPoint: string, bindings: number[]) => {
      const module = device!.createShaderModule({ code });
      const pipeline = await device!.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint } });
      const group = device!.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries.filter(e => bindings.includes(e.binding)) });
      return { pipeline, group };
    };
    const reference = await make(globalFineClassifiedIndirectScanShader, "scanGlobalFineTriangles", [3,4,5,6,7,10,11]);
    const prepare = await make(parallelSurfaceScanShader, "prepareSurfaceScan", [4,5,6,7,11]);
    const count = await make(parallelSurfaceScanShader, "countSurfaceBlocks", [4,5,6,7,18]);
    const scan = await make(parallelSurfaceScanShader, "scanSurfaceBlocks", [3,4,5,6,7,10,11,18]);
    const add = await make(parallelSurfaceScanShader, "addSurfaceBlockOffsets", [4,5,6,7,18]);
    const read = async (source: GPUBuffer, bytes: number) => {
      const target = device!.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device!.createCommandEncoder(); encoder.copyBufferToBuffer(source, 0, target, 0, bytes); device!.queue.submit([encoder.finish()]);
      await target.mapAsync(GPUMapMode.READ); const result = new Uint32Array(target.getMappedRange()).slice(); target.unmap(); target.destroy(); return result;
    };
    const encode = (encoder: GPUCommandEncoder, parallel: boolean) => {
      for (const stage of parallel ? [prepare, count, scan, add] : [reference]) {
        const pass = encoder.beginComputePass(); pass.setPipeline(stage.pipeline); pass.setBindGroup(0, stage.group);
        if (stage === count || stage === add) pass.dispatchWorkgroupsIndirect(dispatch, 12); else pass.dispatchWorkgroups(1);
        pass.end();
      }
    };
    for (const n of [0, 1, 63, 64, 65, 257, 16385, capacity, capacity + 19]) {
      for (const valid of [true, false]) {
        const initial = new Uint32Array([27,1,0,0,n,valid ? 0 : 0xffffffff,1,17]);
        const run = async (parallel: boolean) => {
          device!.queue.writeBuffer(args, 0, initial);
          const encoder = device!.createCommandEncoder(); encoder.clearBuffer(offsets); encode(encoder, parallel); device!.queue.submit([encoder.finish()]);
          return { args: await read(args, 32), offsets: await read(offsets, Math.max(4, Math.min(n, capacity) * 24)), dispatch: await read(dispatch, 12) };
        };
        const expected = await run(false), actual = await run(true);
        assert.deepEqual(actual, expected, `count=${n}, valid=${valid}`);
      }
    }
    // Queue-fenced sustained A/B timing, not a CI timing ceiling.
    const timing: Record<string, number> = {};
    for (const parallel of [false, true]) {
      device.queue.writeBuffer(args, 0, new Uint32Array([0,1,0,0,capacity,0,1,42]));
      const warmup = device.createCommandEncoder(); encode(warmup, parallel); device.queue.submit([warmup.finish()]); await device.queue.onSubmittedWorkDone();
      const start = performance.now();
      for (let i = 0; i < 12; i++) { const encoder = device.createCommandEncoder(); encode(encoder, parallel); device.queue.submit([encoder.finish()]); }
      await device.queue.onSubmittedWorkDone(); timing[parallel ? "parallel_ms" : "original_ms"] = (performance.now() - start) / 12;
    }
    // Active-page launch, including zero-page retirement and the 2D boundary.
    const worklist = buffer(28), classifyDispatch = buffer(12);
    const prepareClassify = await device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code: surfaceClassifyDispatchShader }), entryPoint: "prepareClassify" } });
    const classifyGroup = device.createBindGroup({ layout: prepareClassify.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: worklist } }, { binding: 1, resource: { buffer: params } }, { binding: 2, resource: { buffer: classifyDispatch } }] });
    for (const pages of [0, 1, 7, 32768]) {
      p[7] = 512; p[8] = 65536; p[10] = 65536; device.queue.writeBuffer(params, 0, p); device.queue.writeBuffer(worklist, 4, new Uint32Array([pages]));
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); pass.setPipeline(prepareClassify); pass.setBindGroup(0, classifyGroup); pass.dispatchWorkgroups(1); pass.end(); device.queue.submit([encoder.finish()]);
      const groups = Math.max(1, pages * 2); assert.deepEqual([...await read(classifyDispatch, 12)], [Math.min(groups,65535),Math.ceil(groups/65535),1]);
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ cubes: capacity, ...timing }));
    device.destroy(); device = undefined;
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});

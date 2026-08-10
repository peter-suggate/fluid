import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { octreeLosassoAdaptivePhiScheduleWGSL }
  from "../lib/webgpu-octree-losasso-adaptive-phi.wgsl";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const MAGIC = 0x4150_4849;

test("absent warm candidate stays local while cold bootstrap fails closed", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for adaptive phi candidate scheduling",
}, async () => {
  const dawn = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=metal"]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    | GPUBufferUsage.COPY_DST;
  const make = (size: number, usage = storage) => device.createBuffer({ size, usage });
  const params = make(128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const acceptedGraph = make(128), candidateGraph = make(128);
  const state = make(80), schedule = make(120, storage | GPUBufferUsage.INDIRECT);
  const receipts = make(144);

  const parameterWords = new Uint32Array(32);
  parameterWords.set([8_192, 8_192, 32, 1], 8);
  device.queue.writeBuffer(params, 0, parameterWords);
  const acceptedWords = new Uint32Array(32);
  acceptedWords.set([2, 2_509, 3_394, 2, 0, 3, 3], 0);
  device.queue.writeBuffer(acceptedGraph, 0, acceptedWords);
  const stateWords = new Uint32Array(20);
  stateWords.set([MAGIC, 2, 3, 3, 3_394, 2_509, 0, 1, 0, 2, 0, 0, 0], 0);
  device.queue.writeBuffer(state, 0, stateWords);
  device.queue.writeBuffer(schedule, 0, new Uint32Array(30).fill(0xffff_ffff));
  device.queue.writeBuffer(receipts, 0, new Uint32Array(36).fill(0xdead_beef));

  const module = device.createShaderModule({ code: octreeLosassoAdaptivePhiScheduleWGSL });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto",
    compute: { module, entryPoint: "scheduleCandidateSource" } });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: params } },
    { binding: 1, resource: { buffer: acceptedGraph } },
    { binding: 2, resource: { buffer: candidateGraph } },
    { binding: 4, resource: { buffer: state } },
    { binding: 5, resource: { buffer: schedule } },
    { binding: 6, resource: { buffer: receipts } },
  ] });

  const run = async (): Promise<Uint32Array> => {
    const readback = device.createBuffer({ size: 344,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(schedule, 0, readback, 0, 120);
    encoder.copyBufferToBuffer(state, 0, readback, 120, 80);
    encoder.copyBufferToBuffer(receipts, 0, readback, 200, 144);
    device.pushErrorScope("validation"); device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    assert.equal((await device.popErrorScope())?.message, undefined);
    await readback.mapAsync(GPUMapMode.READ);
    const words = Uint32Array.from(new Uint32Array(readback.getMappedRange()));
    readback.unmap(); readback.destroy(); return words;
  };

  const zeroDispatches = Array.from({ length: 5 }, () => [0, 1, 1]).flat();
  const warm = await run();
  assert.deepEqual(Array.from(warm.subarray(0, 15)), zeroDispatches);
  const warmState = warm.subarray(30, 50);
  assert.deepEqual(Array.from(warmState.subarray(0, 8)),
    Array.from(stateWords.subarray(0, 8)));
  assert.equal(warmState[8], 0); assert.equal(warmState[10], 0);
  assert.equal(warmState[11], 0, "candidate must remain locally invalid");
  assert.equal(warmState[12], 0, "candidate rejection must not poison accepted errors");
  const warmReceipts = warm.subarray(50);
  for (const word of [16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31]) {
    assert.equal(warmReceipts[word], 0, `candidate receipt ${word} must be invalid`);
  }

  device.queue.writeBuffer(acceptedGraph, 0, new Uint32Array(32));
  const cold = await run();
  assert.deepEqual(Array.from(cold.subarray(0, 15)), zeroDispatches);
  assert.equal(cold[30 + 12], 1,
    "cold bootstrap without any candidate must retain fail-closed ERR_GRAPH");

  for (const buffer of [params, acceptedGraph, candidateGraph, state, schedule, receipts]) {
    buffer.destroy();
  }
  device.destroy();
});

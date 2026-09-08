import assert from "node:assert/strict";
import test from "node:test";
import { createCM12ResourceRecorder, type CM12ResourceReference } from
  "../lib/methods/adaptive-mass/sparse-cm12-resource-recipe";
import { GPUTensorCSL4, TENSOR_CSL4_GPU_WGSL } from "../tools/implicit-density/tensor-csl4-gpu";
import { initializeTensorCSL4 } from "../tools/implicit-density/tensor-csl4-oracle";

test("GPU tensor translation records three current-bank passes and admission before commit without field readback", async () => {
  const constants = { GPUBufferUsage: { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8,
    INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 },
  GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 } };
  const descriptors = new Map(Object.keys(constants).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  try {
    for (const [key, value] of Object.entries(constants)) Object.defineProperty(globalThis, key, { configurable: true, value });
    const recorder = createCM12ResourceRecorder({} as GPUSupportedLimits);
    const one = { value: () => 1, derivative: () => 0, integral: (a: number, b: number) => b - a };
    const initial = initializeTensorCSL4([4, 3, 2], [1, 2, 3], [{ scale: .5, factors: [one, one, one] }]);
    const field = await GPUTensorCSL4.create(recorder.device, initial);
    const initialRecipe = recorder.finish({});
    const start = initialRecipe.operations.length;
    await field.advance([.137, -.25, .0625]);
    const operations = recorder.finish({}).operations;
    const advances = operations.slice(start);
    const pipelines = new Map(operations.filter(op => op.method === "createComputePipelineAsync").map(op =>
      [op.result, (op.args[0] as GPUComputePipelineDescriptor).compute.entryPoint]));
    assert.deepEqual(advances.filter(op => op.method === "setPipeline").map(op =>
      pipelines.get((op.args[0] as CM12ResourceReference).cm12Resource)),
    ["beginOperation", "translateX", "translateY", "translateZ", "admitRange", "commitValues", "finalizeGeneration"]);
    const writes = advances.filter(op => op.method === "writeBuffer");
    assert.equal(writes.length, 1, "a step uploads only the 16-word map/grid parameters");
    assert.equal((writes[0]!.args[2] as ArrayBuffer).byteLength, 64);
    assert.equal(advances.filter(op => op.method === "copyBufferToBuffer").length, 0);
    assert.equal(advances.filter(op => op.method === "createShaderModule").length, 0);
    const createdBuffers = operations.filter(op => op.method === "createBuffer").map(op =>
      op.args[0] as GPUBufferDescriptor);
    const momentBanks = createdBuffers.filter(buffer => /(?:accepted27|scratchA27|scratchB27)/.test(buffer.label!));
    assert.equal(momentBanks.length, 3);
    assert.ok(momentBanks.every(buffer => buffer.size === initial.data.length * 4));
    assert.ok(createdBuffers.every(buffer => (buffer.usage & constants.GPUBufferUsage.MAP_READ) === 0),
      "ordinary initialization/advance allocate no readback buffer");
    const source = operations.find(op => op.method === "createShaderModule")!.args[0] as GPUShaderModuleDescriptor;
    assert.equal(source.code, TENSOR_CSL4_GPU_WGSL, "dimensions and generations never specialize the shader text");
    const bindings = new Map(operations.filter(op => op.method === "createBindGroup").map(op =>
      [op.result, (op.args[0] as GPUBindGroupDescriptor).entries]));
    const flow = advances.filter(op => op.method === "setBindGroup").map(op => {
      const entries = [...bindings.get((op.args[1] as CM12ResourceReference).cm12Resource)!];
      return entries.slice(0, 2).map(entry => ((entry.resource as GPUBufferBinding).buffer as unknown as CM12ResourceReference).cm12Resource);
    });
    const accepted = flow[1]![0], scratchA = flow[1]![1], scratchB = flow[2]![1];
    assert.deepEqual(flow.slice(1, 4), [[accepted, scratchA], [scratchA, scratchB], [scratchB, scratchA]]);
    assert.deepEqual(flow[5], [scratchA, accepted], "only post-admission commit can write the accepted buffer");
    field.destroy();
  } finally {
    for (const [key, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else Reflect.deleteProperty(globalThis, key);
    }
  }
});

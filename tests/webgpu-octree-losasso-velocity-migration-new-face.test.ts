import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { octreeLosassoVelocityMigrationWGSL }
  from "../lib/webgpu-octree-losasso-velocity-migration.wgsl";

const modulePath = process.env.WEBGPU_NODE_MODULE;

test("uncovered candidate faces overwrite garbage from complete nodal velocity", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for velocity migration validation",
}, async () => {
  const dawn = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=metal"]).requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBuffersPerShaderStage: Math.min(10,
      adapter.limits.maxStorageBuffersPerShaderStage),
  } });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const make = (size: number, usage = storage) => device.createBuffer({ size, usage });
  const params = make(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const control = make(32), geometry = make(16), velocity = make(16);
  const status = make(16), receipt = make(32), graph = make(32);
  const directory = make(8 * 4), nodal = make(16 * 8), readback = make(64,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const words = new Uint32Array(8); words.set([1, 1, 1, 1, 4, 8], 0);
  device.queue.writeBuffer(params, 0, words);
  device.queue.writeBuffer(control, 0, new Uint32Array([7, 1, 1, 1, 0, 0, 2, 0]));
  device.queue.writeBuffer(geometry, 0, new Uint32Array([0, 0, 0, 0]));
  device.queue.writeBuffer(graph, 0, new Uint32Array([7, 1, 4, 7, 0, 3, 3, 0]));
  device.queue.writeBuffer(directory, 0, new Uint32Array([0, 0, 2, 1, 4, 2, 6, 3]));
  const nodalWords = new Uint32Array(32); const nodalFloats = new Float32Array(nodalWords.buffer);
  for (let node = 0; node < 4; node += 1) {
    nodalFloats[8 * node] = 2 + 2 * node; nodalWords[8 * node + 3] = 7;
  }
  device.queue.writeBuffer(nodal, 0, nodalWords);
  const module = device.createShaderModule({ code: octreeLosassoVelocityMigrationWGSL });
  const prepare = await device.createComputePipelineAsync({ layout: "auto",
    compute: { module, entryPoint: "prepareLosassoLaggedVelocityMigration" } });
  const complete = await device.createComputePipelineAsync({ layout: "auto",
    compute: { module, entryPoint: "completeLosassoLaggedVelocityFromNodes" } });
  const coverage = await device.createComputePipelineAsync({ layout: "auto",
    compute: { module, entryPoint: "countLosassoVelocityMigrationCoverage" } });
  const finish = await device.createComputePipelineAsync({ layout: "auto",
    compute: { module, entryPoint: "finishLosassoLaggedVelocityMigration" } });
  const prepareGroup = device.createBindGroup({ layout: prepare.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: params } }, { binding: 6, resource: { buffer: control } },
    { binding: 9, resource: { buffer: velocity } }, { binding: 12, resource: { buffer: status } },
    { binding: 13, resource: { buffer: receipt } },
  ] });
  const completeGroup = device.createBindGroup({ layout: complete.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: params } }, { binding: 6, resource: { buffer: control } },
    { binding: 7, resource: { buffer: geometry } }, { binding: 9, resource: { buffer: velocity } },
    { binding: 12, resource: { buffer: status } }, { binding: 13, resource: { buffer: receipt } },
    { binding: 14, resource: { buffer: graph } }, { binding: 15, resource: { buffer: directory } },
    { binding: 16, resource: { buffer: nodal } },
  ] });
  const finishGroup = device.createBindGroup({ layout: finish.getBindGroupLayout(0), entries: [
    { binding: 6, resource: { buffer: control } }, { binding: 13, resource: { buffer: receipt } },
  ] });
  const coverageGroup = device.createBindGroup({ layout: coverage.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: params } }, { binding: 6, resource: { buffer: control } },
    { binding: 12, resource: { buffer: status } }, { binding: 13, resource: { buffer: receipt } },
  ] });
  const run = async (garbage: number, breakNode = false) => {
    device.queue.writeBuffer(velocity, 0, new Float32Array([garbage]));
    if (breakNode) {
      const broken = nodalWords.slice(); broken[3] = 0; device.queue.writeBuffer(nodal, 0, broken);
    } else device.queue.writeBuffer(nodal, 0, nodalWords);
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
    pass.setPipeline(prepare); pass.setBindGroup(0, prepareGroup); pass.dispatchWorkgroups(1);
    pass.setPipeline(complete); pass.setBindGroup(0, completeGroup); pass.dispatchWorkgroups(1);
    pass.setPipeline(coverage); pass.setBindGroup(0, coverageGroup); pass.dispatchWorkgroups(1);
    pass.setPipeline(finish); pass.setBindGroup(0, finishGroup); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(velocity, 0, readback, 0, 16);
    encoder.copyBufferToBuffer(status, 0, readback, 16, 16);
    encoder.copyBufferToBuffer(receipt, 0, readback, 32, 32);
    device.pushErrorScope("validation"); device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone(); assert.equal((await device.popErrorScope())?.message, undefined);
    await readback.mapAsync(GPUMapMode.READ); const out = readback.getMappedRange().slice(0);
    readback.unmap(); return { floats: new Float32Array(out), words: new Uint32Array(out) };
  };
  const first = await run(123_456); const second = await run(-987_654);
  assert.equal(first.floats[0], 5); assert.equal(second.floats[0], 5);
  assert.equal(first.words[4], 2); assert.equal(first.words[8 + 4], 0);
  assert.equal(first.words[8 + 5], 7);
  const rejected = await run(42, true);
  assert.notEqual(rejected.words[8 + 4], 0); assert.equal(rejected.words[8 + 5], 0);
  for (const buffer of [params, control, geometry, velocity, status, receipt,
    graph, directory, nodal, readback]) buffer.destroy();
  device.destroy();
});

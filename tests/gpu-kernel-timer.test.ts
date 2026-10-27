import assert from "node:assert/strict";
import test from "node:test";
import { gpuKernelTimer } from "../tools/gpu-kernel-timer";

// Fake the device boundary: these checks need no GPU or shader execution.
Object.assign(globalThis, {
  GPUBufferUsage: { COPY_SRC: 4, COPY_DST: 8, MAP_READ: 1, QUERY_RESOLVE: 512 },
  GPUMapMode: { READ: 1 },
});

function fakeDevice() {
  const passes: { descriptor?: GPUComputePassDescriptor; dispatches: number; ended: boolean }[] = [];
  const queries: { destroyed: boolean }[] = [];
  const buffers: { label?: string; destroyed: boolean }[] = [];
  let failMap = false;
  const device = {
    queue: { onSubmittedWorkDone: async () => {} },
    createBuffer(descriptor: GPUBufferDescriptor) {
      const buffer = { label: descriptor.label, destroyed: false,
        destroy() { this.destroyed = true; },
        async mapAsync() { if (failMap) throw new Error("mapping failed"); },
        getMappedRange() { return new BigUint64Array([1n, 1_000_001n]).buffer; },
        unmap() {},
      };
      buffers.push(buffer); return buffer;
    },
    createQuerySet() {
      const query = { destroyed: false, destroy() { this.destroyed = true; } };
      queries.push(query); return query;
    },
    createCommandEncoder() {
      return {
        beginComputePass(descriptor?: GPUComputePassDescriptor) {
          const pass = { descriptor, dispatches: 0, ended: false };
          passes.push(pass);
          return { setPipeline() {}, setBindGroup() {},
            dispatchWorkgroups() { pass.dispatches++; }, end() { pass.ended = true; } };
        },
        copyBufferToBuffer() {}, resolveQuerySet() {}, finish() { return {}; },
      };
    },
  } as unknown as GPUDevice;
  return { device, passes, queries, buffers, failMapping() { failMap = true; } };
}

test("kernel timing preserves a caller's timestamped pass across multiple dispatches", async () => {
  const raw = fakeDevice();
  const timer = gpuKernelTimer(raw.device); timer.enabled = true;
  const encoder = timer.device.createCommandEncoder();
  const descriptor = { timestampWrites: { querySet: {} as GPUQuerySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } };
  const pass = encoder.beginComputePass(descriptor);
  pass.dispatchWorkgroups(1); pass.dispatchWorkgroups(2); pass.end(); encoder.finish();
  assert.equal(raw.passes.length, 1);
  assert.equal(raw.passes[0]!.dispatches, 2);
  assert.equal(raw.passes[0]!.descriptor, descriptor);
  assert.equal(raw.queries.length, 0);
  assert.equal((await timer.collect()).untimed, 1);
});

test("copy-only encoders allocate no timestamp queries", () => {
  const raw = fakeDevice();
  const timer = gpuKernelTimer(raw.device); timer.enabled = true;
  timer.device.createCommandEncoder().finish();
  assert.equal(raw.queries.length, 0);
});

test("a failed readback releases every pending timing resource", async () => {
  const raw = fakeDevice();
  const timer = gpuKernelTimer(raw.device); timer.enabled = true;
  for (let i = 0; i < 2; i++) {
    const encoder = timer.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.dispatchWorkgroups(1); pass.end(); encoder.finish();
  }
  raw.failMapping();
  await assert.rejects(timer.collect(), /mapping failed/);
  assert.equal(raw.queries.length, 2);
  assert.ok(raw.queries.every(query => query.destroyed));
  const readbacks = raw.buffers.filter(buffer => buffer.label === "kernel timer read" || buffer.label === "kernel timer resolve");
  assert.equal(readbacks.length, 4);
  assert.ok(readbacks.every(buffer => buffer.destroyed));
});

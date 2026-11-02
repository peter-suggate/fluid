import assert from "node:assert/strict";
import test from "node:test";

import { GPUPassTimestampRecorder, GPUPerformanceTraceRecorder, GPUStageTimestampRecorder } from "../lib/core/performance-trace";

Object.assign(globalThis, {
  GPUBufferUsage: { QUERY_RESOLVE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8, STORAGE: 16 },
  GPUMapMode: { READ: 1 },
});

/** A device whose buffers count maps and destroys, and whose maps wait on `release`. */
function fakeDevice() {
  const counts = { maps: 0, destroyed: 0, destroyedWhileMapping: 0 };
  let release = () => {};
  const gate = () => new Promise<void>(resolve => { release = resolve; });
  const device = {
    features: new Set(["timestamp-query"]),
    createShaderModule: () => ({}),
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createQuerySet: () => ({ destroy() {} }),
    createBuffer: () => {
      let mapping = false;
      const buffer = {
        mapState: "unmapped",
        async mapAsync() { counts.maps += 1; mapping = true; await gate(); mapping = false; buffer.mapState = "mapped"; },
        getMappedRange: (_offset = 0, bytes = 64) => new BigUint64Array(bytes / 8).map((_, i) => 100n * BigInt(i + 1)).buffer,
        unmap() { buffer.mapState = "unmapped"; },
        destroy() { counts.destroyed += 1; if (mapping) counts.destroyedWhileMapping += 1; },
      };
      return buffer;
    },
  } as unknown as GPUDevice;
  return { device, counts, release: () => release() };
}

const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
const encoder = { copyBufferToBuffer() {}, beginComputePass: () => pass, resolveQuerySet() {} } as unknown as GPUCommandEncoder;
const phase = { id: "pressure-solve", label: "Pressure" } as const;

test("a fixed trace read before resolve answers nothing and releases the recorder", async () => {
  const { device, counts } = fakeDevice();
  const recorder = new GPUPerformanceTraceRecorder(device, 1, "physics", "lifecycle", [phase]);
  assert.equal(await recorder.read(), undefined);
  assert.equal(counts.maps, 0);
  assert.ok(counts.destroyed > 0);
  assert.throws(() => recorder.boundary(encoder, "late"), /disposed/);
});

test("a second read of a fixed trace never destroys the buffer the first is mapping", async () => {
  const { device, counts, release } = fakeDevice();
  const recorder = new GPUPerformanceTraceRecorder(device, 1, "physics", "lifecycle", [phase]);
  recorder.boundary(encoder, "open");
  recorder.boundary(encoder, "close");
  recorder.resolve(encoder);
  assert.throws(() => recorder.resolve(encoder), /resolved/);
  const first = recorder.read();
  assert.equal(await recorder.read(), undefined);
  assert.equal(counts.maps, 1);
  assert.equal(counts.destroyed, 0);
  release();
  await first;
  assert.equal(counts.destroyedWhileMapping, 0);
});

test("a stage trace refuses stages after resolve, and an overflowed one reads as nothing", async () => {
  const { device } = fakeDevice();
  await GPUStageTimestampRecorder.prepare(device);
  const recorder = new GPUStageTimestampRecorder(device, 1, "physics", "lifecycle", 16);
  const instrumented = recorder.instrument(encoder);
  recorder.begin();
  instrumented.beginComputePass().end();
  recorder.completePhase(instrumented, phase);
  recorder.resolve(instrumented);
  assert.throws(() => recorder.completePhase(instrumented, phase), /resolved/);
  assert.throws(() => recorder.completeFinalPhaseOnNextPass(phase), /resolved/);
  assert.throws(() => recorder.anchorFinalBoundary({} as GPUBuffer), /resolved/);

  const small = new GPUStageTimestampRecorder(device, 2, "physics", "lifecycle", 2);
  const tiny = small.instrument(encoder);
  small.begin();
  for (let stage = 0; stage < 3; stage += 1) { tiny.beginComputePass().end(); small.completePhase(tiny, phase); }
  small.resolve(tiny);
  assert.equal(await small.read(), undefined);
});

test("a pass trace claims no passes and records no stages once resolved", async () => {
  const { device, counts, release } = fakeDevice();
  const recorder = new GPUPassTimestampRecorder(device, 8, "lifecycle");
  const instrumented = recorder.instrument(encoder);
  instrumented.beginComputePass({ label: "work" }).end();
  assert.equal(recorder.resolve(instrumented), true);
  assert.equal(recorder.resolve(instrumented), false);
  assert.throws(() => recorder.completePhase(phase), /resolved/);
  const first = recorder.read();
  assert.equal(await recorder.read(), undefined);
  release();
  await first;
  assert.equal(counts.maps, 1);
  assert.equal(counts.destroyedWhileMapping, 0);
});

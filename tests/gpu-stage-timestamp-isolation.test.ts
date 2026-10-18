import assert from "node:assert/strict";
import test from "node:test";
import { GPUStageTimestampRecorder } from "../lib/core/performance-trace";

test("stage timing isolates end-only compute counters as well as stage boundaries", async () => {
  const globals = globalThis as unknown as Record<string, unknown>;
  const savedUsage = globals.GPUBufferUsage, savedMode = globals.GPUMapMode;
  globals.GPUBufferUsage ??= { QUERY_RESOLVE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8, STORAGE: 16 };
  globals.GPUMapMode ??= { READ: 1 };
  const timestamps = new BigUint64Array(16);
  const commands: string[] = [];
  let separated = false, previousEnd: number | undefined, clock = 100n;
  const resource = () => ({ destroy() {} });
  const device = {
    features: new Set(["timestamp-query"]),
    createShaderModule: () => ({}),
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createQuerySet: resource,
    createBuffer: () => ({ ...resource(), mapState: "unmapped", async mapAsync() {},
      getMappedRange: (_offset: number, bytes: number) => timestamps.buffer.slice(0, bytes), unmap() {} }),
  } as unknown as GPUDevice;
  const encoder = {
    copyBufferToBuffer() { commands.push("copy"); separated = true; },
    beginComputePass(descriptor?: GPUComputePassDescriptor) {
      commands.push("compute");
      const writes = descriptor?.timestampWrites;
      // Model Metal folding: without a native encoder boundary, the preceding
      // pass's end counter is not sampled. This must also cover passes inside
      // a stage, where no new beginning counter is armed.
      if (!separated && previousEnd !== undefined) timestamps[previousEnd] = 0n;
      separated = false;
      if (writes?.beginningOfPassWriteIndex !== undefined) timestamps[writes.beginningOfPassWriteIndex] = clock;
      return { end() {
        clock += 100n;
        previousEnd = writes?.endOfPassWriteIndex;
        if (previousEnd !== undefined) timestamps[previousEnd] = clock;
      } };
    },
    resolveQuerySet() {},
  } as unknown as GPUCommandEncoder;
  try {
    await GPUStageTimestampRecorder.prepare(device);
    const recorder = new GPUStageTimestampRecorder(device, 1, "physics", "regression", 16);
    const instrumented = recorder.instrument(encoder);
    recorder.begin();
    instrumented.beginComputePass().end();
    instrumented.beginComputePass().end();
    recorder.completePhase(instrumented, { id: "pressure-solve", label: "Two-pass stage" });
    instrumented.beginComputePass().end();
    recorder.completePhase(instrumented, { id: "velocity-projection", label: "Final stage" });
    assert.deepEqual(commands, ["copy", "compute", "copy", "compute", "copy", "compute"]);
    recorder.resolve(instrumented);
    const trace = await recorder.read();
    assert.ok(trace);
    assert.equal(trace.total_ms, 300 / 1e6);
    assert.equal(trace.phases.length, 2);
    for (const [i, expected] of [200 / 1e6, 100 / 1e6].entries()) {
      assert.ok(Math.abs(trace.phases[i]!.duration_ms - expected) < 1e-12);
    }
  } finally {
    if (savedUsage === undefined) delete globals.GPUBufferUsage; else globals.GPUBufferUsage = savedUsage;
    if (savedMode === undefined) delete globals.GPUMapMode; else globals.GPUMapMode = savedMode;
  }
});

import assert from "node:assert/strict";
import test from "node:test";
import { FencePartitionedFrameSampler } from "../lib/webgpu-frame-band-sampler";
import { performanceTraceIsExact } from "../lib/performance-trace";

/**
 * The sampler's contract is pure fence arithmetic: fences retire in submit
 * order, and each band's wall is the difference between consecutive
 * `performance.now()` readings. A fake queue resolves each fence by hand and a
 * stubbed clock hands out scripted times, so the differences are checked
 * exactly.
 */

interface FakeFence {
  resolve: () => void;
  promise: Promise<undefined>;
}

const fakeDevice = () => {
  const fences: FakeFence[] = [];
  const submitted: string[] = [];
  const device = {
    queue: {
      submit: (buffers: { label: string }[]) => { submitted.push(...buffers.map((buffer) => buffer.label)); },
      onSubmittedWorkDone: () => {
        let resolve!: () => void;
        const promise = new Promise<undefined>((r) => { resolve = () => r(undefined); });
        fences.push({ resolve, promise });
        return promise;
      },
    },
    createCommandEncoder: ({ label }: { label: string }) => ({
      label,
      finish: () => ({ label }),
    }),
  } as unknown as GPUDevice;
  return { device, fences, submitted };
};

const scriptedClock = (times: number[]) => {
  const original = performance.now;
  let index = 0;
  performance.now = () => times[Math.min(index++, times.length - 1)];
  return () => { performance.now = original; };
};

test("band walls are consecutive fence differences and the trace is exact", async () => {
  const { device, fences } = fakeDevice();
  // Baseline drains at 100; bands complete at 102.5, 110, 116; final at 121.
  const restore = scriptedClock([100, 102.5, 110, 116, 121]);
  try {
    const first = device.createCommandEncoder({ label: "frame" });
    const sampler = new FencePartitionedFrameSampler(device, first, 7, "ctx:band-wall");
    sampler.boundary("source");
    sampler.boundary("svo-primary");
    sampler.boundary("svo-shading");
    const finalCompletion = Promise.resolve();
    // Retire the baseline and the three band fences in queue order before the
    // frame's own completion closes the ledger.
    for (const fence of fences) fence.resolve();
    const trace = await sampler.finish("composite-present", finalCompletion);
    assert.ok(trace);
    assert.equal(trace.sampleId, 7);
    assert.equal(trace.context, "ctx:band-wall");
    assert.equal(trace.measurementSource, "gpu-queue-wall");
    assert.deepEqual(trace.phases.map((phase) => phase.label), [
      "Source: world maintenance + fluid coverage",
      "Primary visibility",
      "Deferred shading",
      "Interfaces + composite + present",
    ]);
    assert.deepEqual(trace.phases.map((phase) => phase.duration_ms), [2.5, 7.5, 6, 5]);
    assert.equal(trace.total_ms, 21);
    assert.equal(performanceTraceIsExact(trace), true);
  } finally {
    restore();
  }
});

test("each boundary hands back a fresh encoder and submits the one it closed", () => {
  const { device, submitted } = fakeDevice();
  const restore = scriptedClock([0]);
  try {
    const first = device.createCommandEncoder({ label: "frame" });
    const sampler = new FencePartitionedFrameSampler(device, first, 1, "ctx");
    assert.equal(sampler.current, first);
    const second = sampler.boundary("source");
    assert.notEqual(second, first, "encoding into a finished encoder is a validation error");
    assert.equal(sampler.current, second);
    assert.deepEqual(submitted, ["frame"]);
    const third = sampler.boundary("svo-primary");
    assert.notEqual(third, second);
    assert.equal(submitted.length, 2);
  } finally {
    restore();
  }
});

test("a lost device voids the sample instead of publishing a partial ledger", async () => {
  const { device, fences } = fakeDevice();
  const restore = scriptedClock([100, 105]);
  try {
    const sampler = new FencePartitionedFrameSampler(
      device, device.createCommandEncoder({ label: "frame" }), 1, "ctx");
    sampler.boundary("source");
    // The baseline retires but the band fence never does (device loss).
    fences[0].resolve();
    const trace = await sampler.finish("composite-present", Promise.resolve());
    assert.equal(trace, undefined);
  } finally {
    restore();
  }
});

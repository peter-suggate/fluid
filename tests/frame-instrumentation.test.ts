import assert from "node:assert/strict";
import test from "node:test";
import {
  FrameInstrumentationRecorder, paneFrameCapture, readFrameCaptures, setFrameDispatchCounting, takeFrameDispatches,
} from "../lib/core/frame-instrumentation";
import type { PerformanceTrace } from "../lib/core/performance-trace";

const trace = (sampleId: number, total_ms: number, span_ms?: number): PerformanceTrace => ({
  sampleId, domain: "gpu", lane: "physics", context: "test", capturedAt_ms: 0, total_ms, span_ms, phases: [],
} as PerformanceTrace);

test("dispatch counting patches the compute pass encoder only while enabled and restores its own methods", () => {
  const calls: string[] = [];
  class FakePass {
    dispatchWorkgroups(x: number, y = 1, z = 1) { calls.push(`d${x}x${y}x${z}`); }
    dispatchWorkgroupsIndirect(_buffer: unknown, offset: number) { calls.push(`i${offset}`); }
  }
  const direct = FakePass.prototype.dispatchWorkgroups, indirect = FakePass.prototype.dispatchWorkgroupsIndirect;
  const global = globalThis as { GPUComputePassEncoder?: unknown };
  const previous = global.GPUComputePassEncoder;
  global.GPUComputePassEncoder = FakePass;
  try {
    assert.equal(takeFrameDispatches(), undefined, "nothing is counted before enabling");
    assert.equal(setFrameDispatchCounting(true), true);
    assert.equal(setFrameDispatchCounting(true), true, "idempotent");
    const pass = new FakePass();
    pass.dispatchWorkgroups(4, 2); pass.dispatchWorkgroups(1); pass.dispatchWorkgroupsIndirect({}, 16);
    assert.deepEqual(calls, ["d4x2x1", "d1x1x1", "i16"], "the platform methods still run with their arguments");
    assert.deepEqual(takeFrameDispatches(), { direct: 2, indirect: 1 });
    assert.deepEqual(takeFrameDispatches(), { direct: 0, indirect: 0 }, "a take resets");
    setFrameDispatchCounting(false);
    assert.equal(FakePass.prototype.dispatchWorkgroups, direct, "off is the platform's own method: zero cost");
    assert.equal(FakePass.prototype.dispatchWorkgroupsIndirect, indirect);
    pass.dispatchWorkgroups(1);
    assert.equal(takeFrameDispatches(), undefined);
  } finally {
    setFrameDispatchCounting(false);
    global.GPUComputePassEncoder = previous;
  }
});

test("dispatch counting reports unavailable without WebGPU", () => {
  const global = globalThis as { GPUComputePassEncoder?: unknown };
  const previous = global.GPUComputePassEncoder;
  delete global.GPUComputePassEncoder;
  try {
    assert.equal(setFrameDispatchCounting(true), false);
    assert.equal(takeFrameDispatches(), undefined);
  } finally { global.GPUComputePassEncoder = previous; }
});

test("a frame sample carries rAF period, host encode and each GPU trace once, on the frame it arrives", () => {
  const recorder = new FrameInstrumentationRecorder(3);
  const solver = { physicsTrace: trace(7, 3.5), uniformMixedFineTiles: 40, uniformMixedCoarseTiles: 900, narrowBandFlipParticles: 12345 };
  const first = recorder.record({ raf_ms: 100, cpu: trace(1, 1.25), presentation: trace(3, 6, 4), presentationSubmitted: true,
    dispatches: { direct: 210, indirect: 30 }, solver });
  assert.deepEqual(first, { frame: 1, raf_ms: 100, hostEncode_ms: 1.25, render_ms: 4, renderSampleId: 3, solverGpu_ms: 3.5, solverSampleId: 7,
    fineTiles: 40, coarseTiles: 900, particles: 12345, dispatches: { direct: 210, indirect: 30 }, presentationSubmitted: true });
  const second = recorder.record({ raf_ms: 116.5, presentation: trace(3, 6, 4), presentationSubmitted: false, solver });
  assert.equal(second.rafPeriod_ms, 16.5);
  assert.equal(second.render_ms, undefined, "a repeated presentation trace is not a new measurement");
  assert.equal(second.solverGpu_ms, undefined, "a repeated solver trace is not a new measurement");
  assert.equal(second.hostEncode_ms, undefined);
  const third = recorder.record({ raf_ms: 133, presentation: trace(4, 5), presentationSubmitted: true, solver: { ...solver, physicsTrace: trace(8, 2) } });
  assert.equal(third.render_ms, 5, "total where the trace has no span");
  assert.equal(third.solverGpu_ms, 2);
  recorder.record({ presentationSubmitted: true, solver: null });
  assert.deepEqual(recorder.take().map(s => s.frame), [2, 3, 4], "keeps the last capacity samples");
  assert.throws(() => new FrameInstrumentationRecorder(0), /positive integer/);
});

test("a pane capture exists only while enabled, survives disabling, restarts when re-enabled", () => {
  assert.equal(paneFrameCapture("t", false), undefined);
  const a = paneFrameCapture("t", true)!;
  assert.equal(paneFrameCapture("t", true), a);
  a.record({ presentationSubmitted: true });
  assert.equal(paneFrameCapture("t", false), undefined);
  assert.equal(readFrameCaptures().t?.length, 1, "readable after disabling");
  assert.equal(typeof (globalThis as { fluidFrameCapture?: unknown }).fluidFrameCapture, "function");
  const b = paneFrameCapture("t", true)!;
  assert.notEqual(b, a);
  assert.equal(readFrameCaptures().t?.length, 0);
});

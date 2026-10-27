import assert from "node:assert/strict";
import test from "node:test";
import { GPUQueueOccupancy, LOAD_STALL_MS, ThreadStallMeter } from "../lib/core/system-load";

test("a thread's blocked share is the lateness of its heartbeat over the window", () => {
  const meter = new ThreadStallMeter(50, () => 0);
  for (let now = 50; now <= 500; now += 50) meter.beat(now);
  assert.deepEqual(meter.sample(500), { blocked: 0, stalls: 0, longestStall_ms: 0 });
  // One beat 300 ms late in a 1 s window: 30% blocked, one stall.
  meter.beat(850);
  for (let now = 900; now <= 1500; now += 50) meter.beat(now);
  const sample = meter.sample(1500);
  assert.equal(sample.blocked, 0.3);
  assert.equal(sample.stalls, 1);
  assert.equal(sample.longestStall_ms, 300);
});

test("a thread frozen mid-window reads as blocked now, and that time is not counted twice", () => {
  const meter = new ThreadStallMeter(50, () => 0);
  meter.beat(50);
  // Nothing has beaten since 50 ms: 450 − 50 = 400 ms overdue at the sample.
  const frozen = meter.sample(500);
  assert.equal(frozen.blocked, 0.8);
  assert.ok(frozen.longestStall_ms > LOAD_STALL_MS);
  // The beat finally lands at 600: 500 ms late in all, 400 of it already reported.
  meter.beat(600);
  for (let now = 650; now <= 1000; now += 50) meter.beat(now);
  assert.equal(meter.sample(1000).blocked, 0.2);
});

test("queue occupancy covers submit to drain, with one completion watch at a time", async () => {
  let now = 0;
  const pending: (() => void)[] = [];
  const queue = new GPUQueueOccupancy(() => new Promise<void>(resolve => pending.push(resolve)), () => now);
  now = 100; queue.submit();
  now = 150; queue.submit();
  assert.equal(pending.length, 1, "one watch outstanding, however many submits");
  now = 300; pending.shift()!(); await Promise.resolve(); await Promise.resolve();
  // The watch covered only the first submit; it re-arms for the second.
  assert.equal(pending.length, 1);
  now = 400; pending.shift()!(); await Promise.resolve(); await Promise.resolve();
  now = 1000;
  // Busy from 100 to 400 in a 1 s window.
  assert.deepEqual(queue.sample(), { busy: 0.3, submits: 2 });
  assert.deepEqual(queue.sample(), { busy: 0, submits: 0 });
});

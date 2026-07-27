import assert from "node:assert/strict";
import test from "node:test";
import {
  CPU_PHYSICS_ACTIVITY_TASKS,
  CPU_QUADTREE_WORKER_ACTIVITY_TASKS,
  createCPUPerformanceActivityProfiler,
  createCPUPerformanceActivityTransportContext,
  createWorkerCPUPerformanceActivityProfiler,
  type CPUPerformanceClock,
} from "../lib/cpu-performance-activity";

class TestClock implements CPUPerformanceClock {
  private index = 0;
  reads = 0;
  constructor(readonly timeOrigin: number, private readonly values: number[]) {}
  now() {
    this.reads += 1;
    return this.values[Math.min(this.index++, this.values.length - 1)] ?? 0;
  }
}

const identity = { frameId: "frame-7", generation: 4, submissionId: "cpu-step-12", publicationId: "fluid-8" };

test("disabled profiling is a clock-free pass-through for sync and async work", async () => {
  const clock = new TestClock(1_000, [1, 2, 3]);
  const profiler = createCPUPerformanceActivityProfiler({
    enabled: false, identity, resourceId: "cpu.main", resourceLabel: "main", resourceKind: "cpu-main", clock,
  });
  assert.equal(profiler.measure(CPU_PHYSICS_ACTIVITY_TASKS.velocityAdvection, () => 17), 17);
  assert.equal(await profiler.measureAsync(CPU_PHYSICS_ACTIVITY_TASKS.pressure, async () => 23), 23);
  profiler.event("publish", CPU_PHYSICS_ACTIVITY_TASKS.publication);
  assert.equal(clock.reads, 0);
  assert.deepEqual(profiler.output().spans, []);
  assert.deepEqual(profiler.output().events, []);
});

test("one-expression sync and async measurements emit canonical raw spans and events", async () => {
  const clock = new TestClock(2_000, [10, 12.5, 13, 17]);
  const profiler = createCPUPerformanceActivityProfiler({
    enabled: true, identity, resourceId: "cpu.main", resourceLabel: "CPU main", resourceKind: "cpu-main", clock,
  });
  profiler.measure(CPU_PHYSICS_ACTIVITY_TASKS.velocityAdvection, () => "done");
  await profiler.measureAsync(CPU_PHYSICS_ACTIVITY_TASKS.pressure, async () => "done");
  const output = profiler.output();
  assert.equal(output.clock?.epochOrigin_ms, 2_000);
  assert.equal(output.clock?.status.state, "reference");
  assert.deepEqual(output.spans.map((span) => [span.taskId, span.start_ms, span.end_ms, span.evidence]), [
    [CPU_PHYSICS_ACTIVITY_TASKS.velocityAdvection.id, 10, 12.5, "measured"],
    [CPU_PHYSICS_ACTIVITY_TASKS.pressure.id, 13, 17, "measured"],
  ]);
  assert.deepEqual(output.events.map((event) => event.kind), ["begin", "end", "begin", "end"]);
  assert.ok(output.spans.every((span) => span.identity.publicationId === "fluid-8"));
  assert.ok(output.tasks.every((task) => /^#[0-9a-f]{6}$/.test(task.color)));
});

test("failed work still closes its measured interval", async () => {
  const syncClock = new TestClock(0, [4, 6]);
  const sync = createCPUPerformanceActivityProfiler({
    enabled: true, identity, resourceId: "cpu.main", resourceLabel: "main", resourceKind: "cpu-main", clock: syncClock,
  });
  assert.throws(() => sync.measure(CPU_PHYSICS_ACTIVITY_TASKS.viscosity, () => { throw new Error("sync"); }), /sync/);
  assert.deepEqual(sync.output().spans.map((span) => [span.start_ms, span.end_ms]), [[4, 6]]);

  const asyncClock = new TestClock(0, [7, 9]);
  const asynchronous = createCPUPerformanceActivityProfiler({
    enabled: true, identity, resourceId: "cpu.main", resourceLabel: "main", resourceKind: "cpu-main", clock: asyncClock,
  });
  await assert.rejects(asynchronous.measureAsync(CPU_PHYSICS_ACTIVITY_TASKS.pressure,
    async () => { throw new Error("async"); }), /async/);
  assert.deepEqual(asynchronous.output().spans.map((span) => [span.start_ms, span.end_ms]), [[7, 9]]);
});

test("worker transport retains local raw time and supplies explicit time-origin alignment", () => {
  const sourceClock = new TestClock(1_000, [10]);
  const transport = createCPUPerformanceActivityTransportContext({ identity, clock: sourceClock });
  assert.equal(transport.sentAt_epoch_ms, 1_010);
  const workerClock = new TestClock(1_005, [7, 7.5, 8, 11]);
  const worker = createWorkerCPUPerformanceActivityProfiler({
    enabled: true, workerId: "quadtree", workerLabel: "Quadtree worker", transport, clock: workerClock,
  });
  assert.deepEqual(worker.alignment, {
    from: "cpu-worker:quadtree:performance", to: "cpu-performance", offset_ms: 5,
    uncertainty_ms: 0, source: "shared-time-origin",
  });
  assert.equal(worker.receivedAt_ms, 7);
  assert.equal(worker.transportLatency_ms, 2);
  worker.profiler.measure(CPU_QUADTREE_WORKER_ACTIVITY_TASKS.unpack, () => undefined);
  const output = worker.profiler.output();
  assert.equal(output.events[0].taskId, CPU_QUADTREE_WORKER_ACTIVITY_TASKS.receive.id);
  assert.deepEqual(output.spans.map((span) => [span.start_ms, span.end_ms]), [[8, 11]]);
  assert.equal(output.spans[0].clockDomain, "cpu-worker:quadtree:performance");
  assert.equal(output.clock?.status.state, "synchronized");
});

test("conflicting metadata for one stable task ID is rejected", () => {
  const clock = new TestClock(0, [1, 2, 3]);
  const profiler = createCPUPerformanceActivityProfiler({
    enabled: true, identity, resourceId: "cpu.main", resourceLabel: "main", resourceKind: "cpu-main", clock,
  });
  profiler.measure({ id: "cpu.test", label: "Original" }, () => undefined);
  assert.throws(() => profiler.event("instant", { id: "cpu.test", label: "Changed" }), /conflicting metadata/);
});

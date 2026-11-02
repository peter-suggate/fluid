import assert from "node:assert/strict";
import test from "node:test";

import { Compiled } from "../lib/core/compiled";
import { RunScope } from "../lib/core/run-scope";
import { RuntimeLifecycle, type RuntimeTerminalPhase } from "../lib/core/runtime-lifecycle";
import { StagedReadback } from "../lib/core/staged-readback";

test("RuntimeLifecycle: the first terminal transition wins and is reported once", () => {
  const reported: RuntimeTerminalPhase[] = [];
  const lifecycle = new RuntimeLifecycle(phase => reported.push(phase));
  assert.equal(lifecycle.open, true);
  assert.equal(lifecycle.live, false);
  assert.equal(lifecycle.ready(), true);
  assert.equal(lifecycle.live, true);
  assert.equal(lifecycle.fail("bloom compile"), true);
  assert.equal(lifecycle.lose("device lost"), false);
  assert.equal(lifecycle.ready(), false);
  assert.equal(lifecycle.open, false);
  assert.equal(lifecycle.failure, "bloom compile");
  assert.throws(() => lifecycle.assertOpen("draw"), /draw: runtime failed \(bloom compile\)/);
  // A failed owner still disposes, once, keeping its reason and reporting nothing new.
  assert.equal(lifecycle.dispose(), true);
  assert.equal(lifecycle.dispose(), false);
  assert.equal(lifecycle.disposed, true);
  assert.equal(lifecycle.failure, "bloom compile");
  assert.deepEqual(reported.map(phase => phase.kind), ["failed"]);
});

test("RuntimeLifecycle: dispose before ready refuses ready", () => {
  const lifecycle = new RuntimeLifecycle();
  assert.equal(lifecycle.dispose(), true);
  assert.equal(lifecycle.ready(), false);
  assert.equal(lifecycle.disposed, true);
  assert.equal(lifecycle.failure, undefined);
});

test("RunScope: a new run aborts the previous one; dispose refuses new runs", () => {
  const scope = new RunScope();
  const first = scope.start();
  assert.equal(first.current(), true);
  const second = scope.start();
  assert.equal(first.current(), false);
  assert.equal(first.signal.reason, "superseded");
  assert.equal(second.current(), true);
  assert.equal(scope.active?.serial, second.serial);
  scope.cancel();
  assert.equal(second.current(), false);
  assert.equal(scope.active, undefined);
  const third = scope.start();
  scope.dispose();
  assert.equal(third.current(), false);
  assert.equal(third.signal.reason, "disposed");
  assert.throws(() => scope.start(), /disposed/);
});

test("Compiled: joins a build in flight, remembers failure, retries on ensure", async () => {
  let builds = 0;
  let failNext = true;
  const compiled = new Compiled(async () => {
    builds += 1;
    if (failNext) { failNext = false; throw new Error("compile"); }
    return builds;
  });
  const a = compiled.ensure();
  const b = compiled.ensure();
  assert.equal(a, b);
  await assert.rejects(a, /compile/);
  assert.equal(compiled.state.kind, "failed");
  assert.equal((compiled.error as Error).message, "compile");
  assert.equal(await compiled.ensure(), 2);
  assert.equal(compiled.value, 2);
  assert.equal(await compiled.ensure(), 2);
  assert.equal(builds, 2);
});

test("Compiled: reset drops a build in flight", async () => {
  let release!: (value: number) => void;
  const compiled = new Compiled(() => new Promise<number>(resolve => { release = resolve; }));
  const pending = compiled.ensure();
  compiled.reset();
  release(7);
  assert.equal(await pending, 7);
  assert.equal(compiled.state.kind, "idle");
  assert.equal(compiled.value, undefined);
});

function fakeDevice() {
  Object.assign(globalThis, { GPUBufferUsage: { COPY_DST: 8, MAP_READ: 1 }, GPUMapMode: { READ: 1 } });
  const bytes = new Uint32Array([0]);
  let resolveMap: (() => void) | undefined;
  let rejectMap: ((error: Error) => void) | undefined;
  const events: string[] = [];
  const buffer = {
    mapAsync: () => new Promise<void>((resolve, reject) => { resolveMap = resolve; rejectMap = reject; }),
    getMappedRange: () => bytes.buffer,
    unmap: () => events.push("unmap"),
    destroy: () => { events.push("destroy"); rejectMap?.(new Error("destroyed")); },
  };
  const device = { createBuffer: () => buffer } as unknown as GPUDevice;
  return { device, bytes, events, finishMap: () => resolveMap?.() };
}

test("StagedReadback: idle → copied → mapping → idle, one copy per read", async () => {
  const gpu = fakeDevice();
  const readback = new StagedReadback(gpu.device, { label: "test", size: 4 });
  const encoder = {} as GPUCommandEncoder;
  let copies = 0;
  assert.equal(await readback.read(() => 1), undefined);
  assert.equal(readback.encode(encoder, () => { copies += 1; }), true);
  assert.equal(readback.encode(encoder, () => { copies += 1; }), false);
  assert.equal(copies, 1);
  gpu.bytes[0] = 42;
  const reading = readback.read(bytes => new Uint32Array(bytes)[0]);
  assert.equal(readback.state, "mapping");
  gpu.finishMap();
  assert.equal(await reading, 42);
  assert.equal(readback.state, "idle");
  assert.deepEqual(gpu.events, ["unmap"]);
});

test("StagedReadback: destroy while mapping yields undefined and frees the staging buffer", async () => {
  const gpu = fakeDevice();
  const readback = new StagedReadback(gpu.device, { label: "test", size: 4 });
  readback.encode({} as GPUCommandEncoder, () => {});
  const reading = readback.read(() => "stale");
  readback.destroy();
  assert.equal(await reading, undefined);
  assert.equal(readback.state, "destroyed");
  assert.equal(readback.encode({} as GPUCommandEncoder, () => {}), false);
  assert.deepEqual(gpu.events, ["destroy"]);
});

test("StagedReadback: discard forgets an unsubmitted copy", () => {
  const gpu = fakeDevice();
  const readback = new StagedReadback(gpu.device, { label: "test", size: 4 });
  readback.encode({} as GPUCommandEncoder, () => {});
  readback.discard();
  assert.equal(readback.idle, true);
});

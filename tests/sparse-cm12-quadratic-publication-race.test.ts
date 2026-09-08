import assert from "node:assert/strict";
import test from "node:test";
import { GPUQuadraticPullback, IDENTITY_MATRIX } from "../tools/implicit-density/sparse-quadratic-pullback";

const identity = { matrix: IDENTITY_MATRIX, translation: [0, 0, 0] as const };
const targets = Uint32Array.of(0);
interface PendingMap { resolve(): void; reject(error: Error): void }
interface FakeBuffer {
  size: number; label: string; data: ArrayBuffer;
  destroy(): void; unmap(): void; getMappedRange(): ArrayBuffer; mapAsync(): Promise<void>;
}
interface Binding { entries: { resource: { buffer: FakeBuffer } }[] }
type Copy = [FakeBuffer, number, FakeBuffer, number, number];
interface Command { binding: Binding; copies: Copy[] }

// Exercise the actual host transaction methods. The fake device supplies only
// buffers, command recording and independently controlled receipt completion;
// it does not execute or pretend to validate the quadratic WGSL mathematics.
function harness() {
  const constants = { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, MAP_READ: 8 }, GPUMapMode: { READ: 1 } };
  const descriptors = new Map(Object.keys(constants).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(constants)) Object.defineProperty(globalThis, key, { configurable: true, value });
  const pending: PendingMap[] = [];
  const submissions: { source: string; destination: string; expectedSourceEpoch: number }[] = [];
  let nextReceipt: { fault?: number; completedSupports?: number } = {};
  const makeBuffer = (size: number, label: string): FakeBuffer => ({
    size, label, data: new ArrayBuffer(size), destroy() {}, unmap() {},
    getMappedRange() { return this.data; },
    mapAsync() { return new Promise<void>((resolve, reject) => pending.push({ resolve, reject })); },
  });
  const device = {
    createBuffer: ({ size, label }: { size: number; label: string }) => makeBuffer(size, label),
    createBindGroup: (binding: Binding) => binding,
    createCommandEncoder() {
      let binding: Binding | undefined;
      const copies: Copy[] = [];
      return {
        clearBuffer() {},
        beginComputePass() {
          return { setPipeline() {}, setBindGroup(_index: number, value: Binding) { binding = value; },
            dispatchWorkgroups() {}, end() {} };
        },
        copyBufferToBuffer(...copy: Copy) { copies.push(copy); },
        finish(): Command { assert.ok(binding); return { binding, copies }; },
      };
    },
    queue: {
      writeBuffer(destination: FakeBuffer, offset: number, source: ArrayBuffer, start: number, size: number) {
        new Uint8Array(destination.data, offset, size).set(new Uint8Array(source, start, size));
      },
      submit(commands: Command[]) {
        for (const { binding, copies } of commands) {
          const buffers = binding.entries.map(entry => entry.resource.buffer);
          const parameters = new Uint32Array(buffers[2]!.data);
          submissions.push({ source: buffers[0]!.label, destination: buffers[1]!.label,
            expectedSourceEpoch: parameters[20]! });
          const receipt = new Uint32Array(buffers[4]!.data);
          receipt[0] = nextReceipt.fault ?? 0;
          receipt[3] = nextReceipt.completedSupports ?? parameters[21]!;
          nextReceipt = {};
          for (const [source, sourceOffset, destination, destinationOffset, size] of copies) {
            new Uint8Array(destination.data, destinationOffset, size)
              .set(new Uint8Array(source.data, sourceOffset, size));
          }
        }
      },
    },
  };
  // Bypass only shader compilation/initial upload. No method, bank, epoch or
  // transaction lock is replaced: run(), advance() and publication are real.
  const field = Reflect.construct(GPUQuadraticPullback, [device,
    { origin: [0, 0, 0], dimensions: [1, 1, 1], h: 1 }, 1,
    [makeBuffer(64, "A"), makeBuffer(64, "B")], {}, new Map(), 2e-6]) as GPUQuadraticPullback;
  return {
    field, submissions,
    receipt(value: typeof nextReceipt) { nextReceipt = value; },
    complete() { const map = pending.shift(); assert.ok(map, "a submitted receipt must be pending"); map.resolve(); },
    rejectMap() { const map = pending.shift(); assert.ok(map); map.reject(new Error("synthetic map failure")); },
    destroy() {
      field.destroy();
      for (const [key, descriptor] of descriptors) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor); else Reflect.deleteProperty(globalThis, key);
      }
    },
  };
}

test("quadratic receipt completion publishes bank and epoch before the next microtask transaction", async () => {
  const h = harness();
  try {
    const first = h.field.advanceConservative(identity, targets);
    let second: ReturnType<GPUQuadraticPullback["advance"]> | undefined;
    h.complete();
    // This runs after run() resumes from mapAsync, but before its awaiting
    // advanceTransaction continuation. Unlock-before-flip encoded A→B twice.
    queueMicrotask(() => { second = h.field.advance(identity, targets); });
    const firstReceipt = await first;
    assert.ok(second);
    h.complete();
    const secondReceipt = await second;
    assert.deepEqual(h.submissions, [
      { source: "A", destination: "B", expectedSourceEpoch: 1 },
      { source: "B", destination: "A", expectedSourceEpoch: 2 },
    ]);
    assert.equal(firstReceipt.generation, 2, "a receipt retains its own committed generation");
    assert.equal(secondReceipt.generation, 3);
    assert.equal(h.field.generation, 3);
  } finally { h.destroy(); }
});

test("quadratic operations reject overlap while the current receipt is pending", async () => {
  const h = harness();
  try {
    const first = h.field.advanceConservative(identity, targets);
    await assert.rejects(h.field.advance(identity, targets), /transaction already running/);
    await assert.rejects(h.field.advanceConservative(identity, targets), /transaction already running/);
    assert.equal(h.submissions.length, 1, "rejected overlaps must not submit commands or unlock the first operation");
    assert.equal(h.field.generation, 1);
    h.complete(); assert.equal((await first).generation, 2);
    const next = h.field.advanceConservative(identity, targets);
    h.complete(); assert.equal((await next).generation, 3);
    assert.deepEqual(h.submissions[1], { source: "B", destination: "A", expectedSourceEpoch: 2 });
  } finally { h.destroy(); }
});

test("quadratic receipt, completeness and mapping failures release the lock without publishing", async () => {
  const h = harness();
  try {
    const initial = h.field.advanceConservative(identity, targets);
    h.complete(); assert.equal((await initial).generation, 2);
    for (const failure of ["integral", "incomplete", "map"] as const) {
      if (failure === "integral") h.receipt({ fault: 64 });
      if (failure === "incomplete") h.receipt({ completedSupports: 0 });
      const attempt = h.field.advanceConservative(identity, targets);
      const rejected = assert.rejects(attempt, failure === "integral" ? /fault=64 / :
        failure === "incomplete" ? /Incomplete quadratic generation receipt/ : /synthetic map failure/);
      if (failure === "map") h.rejectMap(); else h.complete();
      await rejected;
      assert.equal(h.field.generation, 2, `${failure} must not advance the generation`);
    }
    const retried = h.field.advance(identity, targets);
    h.complete(); assert.equal((await retried).generation, 3);
    assert.deepEqual(h.submissions.slice(1), Array.from({ length: 4 }, () =>
      ({ source: "B", destination: "A", expectedSourceEpoch: 2 })),
    "all failed attempts and the successful retry must use the same accepted source");
  } finally { h.destroy(); }
});

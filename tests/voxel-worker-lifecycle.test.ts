import assert from "node:assert/strict";
import test from "node:test";
import { defaultScene } from "../lib/core/model";
import { WebGPURenderWorkerClient, type WebGPURenderWorkerRequest, type WebGPURenderWorkerResponse } from "../lib/core/webgpu-render-worker-client";
import { webGPUPlatformResourcePlugin } from "../lib/core/webgpu-platform-resource";

class FakeWorker extends EventTarget {
  static latest: FakeWorker;
  messages: WebGPURenderWorkerRequest[] = [];
  terminated = false;
  constructor() { super(); FakeWorker.latest = this; }
  postMessage(message: WebGPURenderWorkerRequest) { this.messages.push(message); }
  terminate() { this.terminated = true; }
  emit(message: WebGPURenderWorkerResponse) { this.dispatchEvent(new MessageEvent("message", { data: message })); }
  finishShutdown() {
    const request = this.messages.findLast((message) => message.type === "shutdown");
    assert.ok(request && "requestId" in request);
    this.emit({ type: "shutdown-complete", requestId: request.requestId });
  }
}
async function withClient(run: (client: WebGPURenderWorkerClient, worker: FakeWorker) => Promise<void>) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "Worker");
  Object.defineProperty(globalThis, "Worker", { configurable: true, value: FakeWorker });
  try {
    const canvas = { transferControlToOffscreen: () => ({}) } as HTMLCanvasElement;
    const client = new WebGPURenderWorkerClient(canvas, { onStatus() {} });
    await run(client, FakeWorker.latest);
  } finally {
    if (descriptor) Object.defineProperty(globalThis, "Worker", descriptor);
    else Reflect.deleteProperty(globalThis, "Worker");
  }
}

test("shutdown rejects outstanding validation and rejects any later edit immediately", async () => {
  await withClient(async (client, worker) => {
    const pending = client.validateLiveSolidEdit(defaultScene, defaultScene);
    const rejected = assert.rejects(pending, /runtime stopped/);
    const shutdown = client.shutdown();
    await rejected;
    const count = worker.messages.length;
    await assert.rejects(client.validateLiveSolidEdit(defaultScene, defaultScene), /no longer available/);
    assert.equal(worker.messages.length, count);
    assert.equal(worker.terminated, false);
    worker.finishShutdown();
    await shutdown;
    assert.equal(worker.terminated, true);
  });
});

test("terminal solver status rejects editing but lets GPU shutdown complete", async () => {
  for (const state of ["unavailable", "lost", "blocked"] as const) {
    await withClient(async (client, worker) => {
      const pending = client.validateLiveSolidEdit(defaultScene, defaultScene);
      const rejected = assert.rejects(pending, /Solver halted/);
      worker.emit({ type: "status", status: { state, label: "Solver halted", resource: webGPUPlatformResourcePlugin }, workerNow_ms: 0 });
      await rejected;
      await assert.rejects(client.validateLiveSolidEdit(defaultScene, defaultScene), /Solver halted/);
      const shutdown = client.shutdown();
      worker.emit({ type: "status", status: { state, label: "Solver halted", resource: webGPUPlatformResourcePlugin }, workerNow_ms: 0 });
      assert.equal(worker.terminated, false);
      worker.finishShutdown();
      await shutdown;
      assert.equal(worker.terminated, true);
    });
  }
});

test("a crashed worker rejects future requests and terminates without waiting for an acknowledgement", async () => {
  await withClient(async (client, worker) => {
    const pending = client.validateLiveSolidEdit(defaultScene, defaultScene);
    const rejected = assert.rejects(pending, /unserializable/);
    worker.dispatchEvent(new Event("messageerror"));
    await rejected;
    await assert.rejects(client.validateLiveSolidEdit(defaultScene, defaultScene), /no longer available/);
    await client.shutdown();
    assert.equal(worker.terminated, true);
  });
});

test("fluid action waits for its receipt and shutdown rejects pending fluid actions", async () => {
  await withClient(async (client, worker) => {
    const edit = { operation: "add", shape: "torus", center_m: { x: 0, y: 1, z: 0 }, radius_m: .2 } as const;
    const first = client.editFluid(edit);
    const message = worker.messages.at(-1)!;
    assert.equal(message.type, "edit-fluid");
    assert.ok("requestId" in message);
    worker.emit({ type: "fluid-edit-result", requestId: message.requestId, result: { accepted: false, reason: "Capacity" } });
    assert.deepEqual(await first, { accepted: false, reason: "Capacity" });
    const pending = client.editFluid(edit), rejected = assert.rejects(pending, /runtime stopped/);
    const shutdown = client.shutdown(); await rejected;
    await assert.rejects(client.editFluid(edit), /no longer available/);
    worker.finishShutdown(); await shutdown;
  });
});

import assert from "node:assert/strict";
import test from "node:test";
import type { WebGPURenderWorkerRequest, WebGPURenderWorkerResponse } from "../lib/core/webgpu-render-worker-client";

test("a draw that throws still answers its frame, so the window's frame pump cannot wedge", async () => {
  const posted: WebGPURenderWorkerResponse[] = [];
  const scope = Object.assign(new EventTarget(), { postMessage: (message: WebGPURenderWorkerResponse) => posted.push(message) });
  const setIntervalBefore = globalThis.setInterval;
  const selfBefore = Object.getOwnPropertyDescriptor(globalThis, "self");
  // The worker's load meters would otherwise keep the test process alive.
  globalThis.setInterval = ((...args: Parameters<typeof setInterval>) => setIntervalBefore(...args).unref()) as typeof setInterval;
  Object.defineProperty(globalThis, "self", { configurable: true, value: scope });
  try {
    await import("../lib/core/webgpu-render-worker");
  } finally {
    globalThis.setInterval = setIntervalBefore;
    if (selfBefore) Object.defineProperty(globalThis, "self", selfBefore);
    else Reflect.deleteProperty(globalThis, "self");
  }
  const send = (data: unknown) => scope.dispatchEvent(new MessageEvent("message", { data: data as WebGPURenderWorkerRequest }));
  send({ type: "attach", canvas: {} });
  // No scene was published under this revision: the draw throws before it reaches the GPU.
  send({ type: "draw", frameId: 7, sceneRevision: 3, args: [0, undefined, undefined, { methodId: "test-method" }],
    instrumentationMode: "off", viewport: { width: 1, height: 1, devicePixelRatio: 1 } });
  await new Promise(resolve => setTimeout(resolve, 0));
  const status = posted.find(message => message.type === "status");
  assert.equal(status?.type === "status" && status.status.state, "unavailable");
  const frame = posted.find(message => message.type === "frame");
  assert.ok(frame?.type === "frame");
  assert.equal(frame.frameId, 7);
  assert.equal(frame.metrics.methodId, "test-method");
  assert.equal(frame.metrics.presentationSubmitted, false);
});

import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { gpuStatusAcceptsRebuild, type GPUStatus } from "../lib/core/gpu-status";
import { webGPUPlatformResourcePlugin } from "../lib/core/webgpu-platform-resource";
import type { WebGPURenderWorkerClient } from "../lib/core/webgpu-render-worker-client";
import { GPU_MANUAL_START_EVENT, type PaneId } from "../lib/core/gpu-startup";

/**
 * The pane's GPU status leaves a stopped, lost or waiting state only through a
 * new session. These are the regressions the effect-local flags had: a reset
 * or a setting change relabelled a stopped pane as "Preparing GPU work plan",
 * and a terminal status that arrived before the start neither stopped the
 * session nor showed its reason.
 */
const listeners = new Map<string, Set<(event: Event) => void>>();
Object.assign(globalThis, {
  window: Object.assign(globalThis, {
    location: { search: "?gpu=manual" },
    addEventListener: (type: string, listener: (event: Event) => void) => {
      listeners.set(type, (listeners.get(type) ?? new Set()).add(listener));
    },
    removeEventListener: (type: string, listener: (event: Event) => void) => { listeners.get(type)?.delete(listener); },
  }),
  requestAnimationFrame: () => 1,
  cancelAnimationFrame: () => {},
});

const { simulation } = await import("../lib/core/simulation/controller");
const { GPUViewportSession } = await import("../components/gpu-viewport-session");

type RendererCallbacks = ConstructorParameters<typeof WebGPURenderWorkerClient>[1];

function fakeRenderer() {
  const calls: string[] = [];
  let callbacks!: RendererCallbacks;
  const renderer = {
    initialize: async () => { calls.push("initialize"); },
    shutdown: async () => { calls.push("shutdown"); },
    setSimulationRunning: async () => undefined,
    resetSimulationTimeline: () => {},
    advanceInteractionFluid: async () => undefined,
  } as unknown as WebGPURenderWorkerClient;
  return {
    calls,
    status: (status: GPUStatus) => callbacks.onStatus(status),
    create: (_canvas: HTMLCanvasElement, given: RendererCallbacks) => { callbacks = given; return renderer; },
  };
}

function startSession(renderer: ReturnType<typeof fakeRenderer>) {
  const session = simulation.session();
  const lifecycles = new Map<PaneId, never>();
  const gpu = GPUViewportSession.start({
    canvas: {} as HTMLCanvasElement,
    paneId: "a",
    session,
    binding: { publishFrameRate: () => {}, pixelTraceDrawConfig: () => undefined, publishPixelTrace: () => {}, publishBodyPoses: () => {} },
    lifecycles,
    onReleased: () => {},
    createRenderer: renderer.create,
  });
  assert.ok(gpu);
  return { session, gpu };
}

test("a rebuild may announce itself only over a live session's status", () => {
  const resource = webGPUPlatformResourcePlugin;
  assert.equal(gpuStatusAcceptsRebuild({ state: "ready", label: "", adapter: "", resource }), true);
  assert.equal(gpuStatusAcceptsRebuild({ state: "initializing", label: "", resource }), true);
  assert.equal(gpuStatusAcceptsRebuild({ state: "cancelled", label: "", resource }), true);
  for (const state of ["manual", "stopping", "unavailable", "lost", "blocked"] as const) {
    assert.equal(gpuStatusAcceptsRebuild({ state, label: "", resource }), false, state);
  }
});

test("a lost device before the start stops the session and shows why", async () => {
  const renderer = fakeRenderer();
  const { session, gpu } = startSession(renderer);
  assert.equal(gpu.phase, "waiting");
  assert.equal(session.diagnostics.getState().gpuStatus.state, "manual");
  renderer.status({ state: "lost", label: "Device lost", resource: webGPUPlatformResourcePlugin });
  await new Promise(resolve => setTimeout(resolve));
  assert.equal(gpu.phase, "stopped");
  assert.deepEqual(renderer.calls, ["shutdown"]);
  assert.match(session.diagnostics.getState().gpuStatus.label, /^Device lost; device released/);
  await gpu.cleanupImmediately();
});

test("a stopped pane is not relabelled by a setting change, and a late renderer status is dropped", async (context) => {
  const renderer = fakeRenderer();
  const { session, gpu } = startSession(renderer);
  await gpu.stop();
  const stopped = session.diagnostics.getState().gpuStatus;
  assert.equal(stopped.state, "unavailable");
  context.mock.method(simulation, "reset", () => {});
  simulation.setQuality("balanced");
  renderer.status({ state: "initializing", label: "Compiling", resource: webGPUPlatformResourcePlugin });
  assert.equal(session.diagnostics.getState().gpuStatus, stopped);
  await gpu.cleanupImmediately();
});

test("a stop during the start abandons it: no render loop, no later status", async () => {
  const renderer = fakeRenderer();
  const { session, gpu } = startSession(renderer);
  for (const listener of listeners.get(GPU_MANUAL_START_EVENT) ?? []) listener(new Event(GPU_MANUAL_START_EVENT));
  assert.equal(gpu.phase, "starting");
  const stopping = gpu.stop("Stopped by test");
  await stopping;
  await new Promise(resolve => setTimeout(resolve));
  assert.equal(gpu.phase, "stopped");
  assert.match(session.diagnostics.getState().gpuStatus.label, /^Stopped by test/);
  await gpu.cleanupImmediately();
});

test("a hidden studio's render loop draws nothing and restarts when the studio is shown", async () => {
  const { useShellStore } = await import("../lib/core/stores/shell-store");
  const scheduled: FrameRequestCallback[] = [];
  const requestBefore = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = (callback: FrameRequestCallback) => scheduled.push(callback);
  const { gpu } = startSession(fakeRenderer());
  let drawn = 0;
  Object.assign(gpu.renderer, { setSimulationScene: () => { drawn += 1; throw new Error("drawn"); } });
  try {
    for (const listener of listeners.get(GPU_MANUAL_START_EVENT) ?? []) listener(new Event(GPU_MANUAL_START_EVENT));
    await new Promise(resolve => setTimeout(resolve));
    assert.equal(gpu.phase, "rendering");
    assert.equal(scheduled.length, 1, "the loop is scheduled once rendering");
    useShellStore.getState().setStudioVisible(false);
    scheduled.shift()!(0);
    assert.equal(drawn, 0, "nothing is drawn while hidden");
    assert.equal(scheduled.length, 0, "and the loop does not reschedule itself");
    useShellStore.getState().setStudioVisible(false);
    assert.equal(scheduled.length, 0);
    useShellStore.getState().setStudioVisible(true);
    assert.equal(scheduled.length, 1, "showing the studio restarts the loop once");
    useShellStore.getState().setStudioVisible(true);
    assert.equal(scheduled.length, 1);
  } finally {
    useShellStore.getState().setStudioVisible(true);
    globalThis.requestAnimationFrame = requestBefore;
    await gpu.stop();
    await gpu.cleanupImmediately();
  }
});

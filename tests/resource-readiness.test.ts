import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  initialResourceReadiness,
  reduceGPUResourceEvidence,
  reduceGPUResourceStatus,
  resourceInteractionGates,
  duplicateResourcePluginIds,
  resourceCapabilityUsable,
  resourceActivities,
} from "../lib/resource-readiness";
import { RESOURCE_PLUGIN_CATALOG, resourcePluginsProviding } from "../lib/resource-plugin-catalog";
import { WebGPURenderWorkerClient, type WebGPURenderWorkerResponse } from "../lib/webgpu-render-worker-client";
import { liveSvoSceneResourcePlugin } from "../lib/webgpu-live-svo-scene";
import { svoPresentationResourcePlugin } from "../lib/webgpu-svo-dry-scene";

test("resource plugins compose statically and claim unique identities", () => {
  assert.deepEqual(duplicateResourcePluginIds(RESOURCE_PLUGIN_CATALOG), []);
  assert.ok(resourcePluginsProviding("renderer").length > 0);
  assert.ok(resourcePluginsProviding("fluid-authority").length > 0);
  assert.ok(resourcePluginsProviding("sparse-voxel-presentation").length > 0);
  for (const plugin of RESOURCE_PLUGIN_CATALOG) {
    assert.ok(plugin.id.includes("."));
    assert.ok(plugin.provides.length > 0);
  }
});

test("independent resource lanes do not overwrite one another", () => {
  let readiness = initialResourceReadiness();
  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "WebGPU renderer ready", adapter: "test",
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "WebGPU t=0 ready", adapter: "test",
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "initializing", label: "Compiling sparse dry-scene pipeline",
    phase: "presentation", completed: 0, total: 1, startedAt_ms: 1,
  });

  assert.equal(readiness.platform.usable, true);
  assert.equal(readiness.fluid.usable, true);
  assert.equal(readiness.svo.state, "preparing");
  assert.equal(resourceCapabilityUsable(readiness, "fluid-authority"), true);
  assert.equal(resourceCapabilityUsable(readiness, "sparse-voxel-presentation"), false);
  assert.deepEqual(resourceInteractionGates(readiness, true), {
    shellInteractive: true,
    // The device is up, so the user may still look at the scene from anywhere;
    // only the ray has to wait for something published to hit.
    cameraInteractive: true,
    pickingInteractive: false,
    transportInteractive: false,
  });
});

test("a presentation rebuild costs picking and nothing else", () => {
  let readiness = reduceGPUResourceStatus(initialResourceReadiness(), {
    state: "ready", label: "WebGPU renderer ready", adapter: "test",
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "Live SVO renderer ready", adapter: "test",
    resource: svoPresentationResourcePlugin,
  });
  assert.equal(resourceInteractionGates(readiness, false).pickingInteractive, true);

  // An ordinary pipeline recompile: the image is being replaced, the device is not.
  readiness = reduceGPUResourceEvidence(readiness, undefined, {
    state: "pending", failureReason: "pipeline-compiling",
  });
  assert.deepEqual(resourceInteractionGates(readiness, false), {
    shellInteractive: true,
    cameraInteractive: true,
    pickingInteractive: false,
    transportInteractive: false,
  });
});

test("without a device nothing but the shell is interactive", () => {
  // Sparse presentation is satisfied outright and the platform lane is still
  // preparing: there is no image to orbit around, so the camera waits too.
  const readiness = reduceGPUResourceEvidence(initialResourceReadiness(), undefined, {
    state: "not-required",
  });
  assert.equal(resourceCapabilityUsable(readiness, "sparse-voxel-presentation"), true);
  assert.equal(resourceCapabilityUsable(readiness, "renderer"), false);
  assert.deepEqual(resourceInteractionGates(readiness, false), {
    shellInteractive: true,
    cameraInteractive: false,
    pickingInteractive: false,
    transportInteractive: false,
  });
});

test("published resource evidence outranks a late progress message", () => {
  const readiness = reduceGPUResourceEvidence(initialResourceReadiness(), {
    initialSparseAuthorityReady: true,
    initialRasterSurfaceReady: true,
  } as never);
  assert.equal(readiness.fluid.state, "ready");
  assert.equal(readiness.fluid.usable, true);
});

test("renderer completion closes 4/4 before solver planning opens", () => {
  const platform = RESOURCE_PLUGIN_CATALOG.find(({ id }) => id === "platform.webgpu-renderer")!;
  const fluid = RESOURCE_PLUGIN_CATALOG.find(({ id }) => id === "fluid.power-octree")!;
  let readiness = reduceGPUResourceStatus(initialResourceReadiness(), {
    state: "initializing", label: "Renderer ready", phase: "renderer",
    completed: 4, total: 4, startedAt_ms: 1, resource: platform,
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "WebGPU renderer ready", adapter: "test", resource: platform,
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "initializing", label: "Preparing fenced t=0 solver authority", phase: "planning",
    completed: 0, total: 0, startedAt_ms: 2, resource: fluid,
  });

  assert.deepEqual(resourceActivities(readiness).map(({ pluginId, completed, total }) => ({ pluginId, completed, total })), [
    { pluginId: "fluid.power-octree", completed: 0, total: 0 },
  ]);
});

test("renderer-only live source and first sparse frame close independent activities", () => {
  let readiness = reduceGPUResourceStatus(initialResourceReadiness(), {
    state: "initializing", label: "Attach warmed solver", phase: "attach",
    completed: 1, total: 2, startedAt_ms: 1, resource: liveSvoSceneResourcePlugin,
  });
  assert.deepEqual(resourceActivities(readiness).map(({ pluginId, label }) => ({ pluginId, label })), [{
    pluginId: liveSvoSceneResourcePlugin.id,
    label: "Attach warmed solver",
  }]);

  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "Live sparse scene source ready", adapter: "test",
    resource: liveSvoSceneResourcePlugin,
  });
  assert.deepEqual(resourceActivities(readiness), [],
    "source attachment must retire the solver-handoff activity");
  assert.equal(resourceCapabilityUsable(readiness, "live-scene"), true);
  assert.equal(resourceCapabilityUsable(readiness, "sparse-voxel-presentation"), false);

  readiness = reduceGPUResourceStatus(readiness, {
    state: "initializing", label: "Submit first sparse frame", phase: "presentation",
    completed: 9, total: 10, startedAt_ms: 2, resource: svoPresentationResourcePlugin,
  });
  assert.deepEqual(resourceActivities(readiness).map(({ pluginId, label }) => ({ pluginId, label })), [{
    pluginId: svoPresentationResourcePlugin.id,
    label: "Submit first sparse frame",
  }]);

  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "Live SVO renderer ready", adapter: "test",
    resource: svoPresentationResourcePlugin,
  });
  assert.deepEqual(resourceActivities(readiness), [],
    "the first-frame completion fence must retire sparse presentation activity");
  assert.equal(resourceCapabilityUsable(readiness, "live-scene"), true);
  assert.equal(resourceCapabilityUsable(readiness, "sparse-voxel-presentation"), true);
});

test("a rejected frame revokes the previously usable scene generation", () => {
  let readiness = reduceGPUResourceStatus(initialResourceReadiness(), {
    state: "ready", label: "WebGPU renderer ready", adapter: "test",
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "Live SVO renderer ready", adapter: "test",
    resource: svoPresentationResourcePlugin,
  });
  assert.equal(resourceInteractionGates(readiness, false).pickingInteractive, true);

  readiness = reduceGPUResourceEvidence(readiness, undefined, {
    state: "pending", failureReason: "missing-source",
  });
  assert.equal(resourceCapabilityUsable(readiness, "sparse-voxel-presentation"), false);
  assert.deepEqual(resourceInteractionGates(readiness, false), {
    shellInteractive: true,
    cameraInteractive: true,
    pickingInteractive: false,
    transportInteractive: false,
  });
});

test("an inapplicable sparse renderer satisfies presentation without an activity", () => {
  const readiness = reduceGPUResourceEvidence(initialResourceReadiness(), undefined, {
    state: "not-required",
  });
  assert.equal(resourceCapabilityUsable(readiness, "sparse-voxel-presentation"), true);
  assert.deepEqual(resourceActivities(readiness), []);
});

test("replacement work retains interaction when a previous generation is attached", () => {
  let readiness = reduceGPUResourceStatus(initialResourceReadiness(), {
    state: "ready", label: "WebGPU renderer ready", adapter: "test",
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "WebGPU direct-field solver ready", adapter: "test",
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "Live SVO renderer ready", adapter: "test",
    resource: svoPresentationResourcePlugin,
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "initializing", label: "Allocating replacement solver", phase: "allocation",
    completed: 1, total: 4, startedAt_ms: 1, kind: "rebuild", retainingPrevious: true,
  });
  assert.equal(readiness.fluid.state, "preparing");
  assert.equal(readiness.fluid.usable, true);
  assert.equal(resourceInteractionGates(readiness, true).transportInteractive, true);
});

test("the UI presents concurrent plugin work without treating all progress as a global block", () => {
  const fluidLab = readFileSync(new URL("../components/FluidLab.tsx", import.meta.url), "utf8");
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const transport = readFileSync(new URL("../components/TransportBar.tsx", import.meta.url), "utf8");
  assert.match(fluidLab, /resourceActivities\(resourceReadiness\)/);
  assert.match(fluidLab, /resource-activity-tray/);
  // The tray shows what each plugin declares it blocks; transport-blocking work
  // is stated by the control it suspends, so it is absent from the tray.
  assert.match(fluidLab, /resourceActivitiesFor\(resourceReadiness, "card"\)/);
  assert.match(fluidLab, /resourceActivitiesFor\(resourceReadiness, "pill"\)/);
  assert.doesNotMatch(fluidLab, /resourceActivitiesFor\(resourceReadiness, "transport-inline"\)/);
  assert.match(transport, /resourceActivitiesFor\(resourceReadiness, "transport-inline"\)/);
  assert.match(fluidLab, /aria-label="Resource tasks"/);
  assert.doesNotMatch(fluidLab, /viewportBlocked|blockingActivity|backgroundActivities/);
  assert.doesNotMatch(fluidLab, /<section className="viewport-shell" aria-busy=/);
  assert.doesNotMatch(fluidLab, /gpuStatus\.state === "initializing" && <GPUInitializationPanel/);
  assert.match(fluidLab, /finalizing \? "Finalizing…"/);
  assert.match(fluidLab, /GPU driver exposes no intermediate counters/);
  assert.match(viewport, /if \(rendererOnlyReady\) useDiagnosticsStore\.getState\(\)\.set\(\{ gpuStatus: status \}\)/,
    "the completed platform plugin must close before fluid planning begins");
  assert.match(viewport, /phase: "planning", completed: 0, total: 0/,
    "the solver handoff must not reuse the renderer's completed task count");
  assert.match(transport, /resourceInteractionGates\(resourceReadiness, !rendererOnlyScene\)/);
});

test("a rebuild cannot detach the viewport's own input", () => {
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const catalog = readFileSync(new URL("../lib/editor-entity-catalog.ts", import.meta.url), "utf8");
  for (const handler of ["onPointerDown={pointerDown}", "onPointerMove={pointerMove}",
    "onPointerUp={pointerUp}", "onPointerCancel={pointerUp}"]) {
    assert.ok(viewport.includes(handler),
      `${handler} must be attached unconditionally: a compiling pipeline is not a reason to stop reading the pointer`);
  }
  assert.match(viewport, /onWheel=\{cameraInteractive \?/,
    "zoom answers to the device, not to whether a generation is published");
  assert.doesNotMatch(viewport, /scenePresentationAvailable/,
    "the over-broad gate is gone; camera and picking are separate questions");
  // Handles are document geometry, so only the click into the scene is gated.
  assert.match(catalog, /export function entityAtRay\([\s\S]{0,200}?context\.pickingAvailable === false/);
  assert.doesNotMatch(
    catalog.slice(catalog.indexOf("export function surfacedEntities"), catalog.indexOf("export function entityAtRay")),
    /pickingAvailable/);
});

test("WebGPU resource work is worker-owned and frame traffic is bounded", () => {
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const client = readFileSync(new URL("../lib/webgpu-render-worker-client.ts", import.meta.url), "utf8");
  const worker = readFileSync(new URL("../lib/webgpu-render-worker.ts", import.meta.url), "utf8");
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const initialization = readFileSync(new URL("../lib/gpu-initialization.ts", import.meta.url), "utf8");

  assert.match(viewport, /new WebGPURenderWorkerClient\(canvas/);
  assert.doesNotMatch(viewport, /new FluidLabRenderer\(/,
    "React must never construct the WebGPU runtime in the UI realm");
  assert.match(client, /transferControlToOffscreen\(\)/);
  assert.match(client, /new Worker\(new URL\("\.\/webgpu-render-worker\.ts", import\.meta\.url\)/);
  assert.match(client, /if \(this\.frameInFlight \|\| this\.frameDispatchSuspended\) this\.queuedFrame = message/,
    "busy initialization must retain only the latest UI snapshot");
  assert.match(client,
    /setSimulationRunning\(running: boolean\): Promise<number \| undefined>[\s\S]*type: "set-simulation-running"/,
    "pause must return the worker's authoritative submitted time instead of a cached diagnostics snapshot");
  assert.match(client, /if \(queued && !this\.frameDispatchSuspended\) this\.sendFrame\(queued\)/,
    "a pause/play edge must fence old-intent frame snapshots until the worker observes it");
  assert.match(worker,
    /runtime\.setSimulationRunning\(message\.running\)[\s\S]*type: "simulation-running-set"/,
    "the worker must acknowledge the control edge from the renderer-owned solver clock");
  assert.match(viewport,
    /renderer\.setSimulationRunning\(runState === "running"\)\.then\([\s\S]*simulation\.gpuSchedulingPaused\(submittedTime_s\)/,
    "the controller may discard host debt only after the authoritative pause acknowledgement");
  assert.match(client, /performance\.now\(\) - Math\.max\(0, message\.workerNow_ms - message\.status\.startedAt_ms\)/,
    "worker-relative task clocks must be translated before the UI renders elapsed time");
  assert.match(worker, /renderer = new FluidLabRenderer\(/);
  assert.match(worker, /runtime\.setViewportSize/);
  assert.match(worker, /document: markSceneRevision\(message\.scene\)/,
    "the retained worker document must recover identity-based scene memoization after cloning");
  assert.match(client, /type: "set-render-scene", revision, scene, terrainContentStamp: stamp/);
  assert.match(renderer, /HTMLCanvasElement \| OffscreenCanvas/);
  assert.match(initialization, /typeof document !== "undefined" && typeof requestAnimationFrame === "function"/,
    "worker startup must not wait on a presentation rAF that cannot fire yet");
});

test("pause drops queued running frames and returns the worker-owned submitted clock", async () => {
  class FakeWorker {
    static instance: FakeWorker;
    readonly messages: unknown[] = [];
    private readonly listeners = new Map<string, (event: MessageEvent<WebGPURenderWorkerResponse>) => void>();
    constructor() { FakeWorker.instance = this; }
    addEventListener(type: string, listener: (event: MessageEvent<WebGPURenderWorkerResponse>) => void) {
      this.listeners.set(type, listener);
    }
    postMessage(message: unknown) { this.messages.push(message); }
    terminate() {}
    emit(message: WebGPURenderWorkerResponse) {
      this.listeners.get("message")?.({ data: message } as MessageEvent<WebGPURenderWorkerResponse>);
    }
  }
  const previousWorker = Object.getOwnPropertyDescriptor(globalThis, "Worker");
  Object.defineProperty(globalThis, "Worker", { configurable: true, value: FakeWorker });
  try {
    const canvas = {
      transferControlToOffscreen: () => ({}),
      getBoundingClientRect: () => ({ width: 640, height: 360 }),
    } as unknown as HTMLCanvasElement;
    const client = new WebGPURenderWorkerClient(canvas, { onStatus() {} });
    const drawArgs = Array(16).fill(undefined);
    drawArgs[1] = {};
    drawArgs[7] = { methodId: "octree" };
    const draw = () => (client.draw as unknown as (...args: unknown[]) => unknown)(...drawArgs);

    draw();
    draw();
    const pause = client.setSimulationRunning(false);
    const worker = FakeWorker.instance;
    const pauseRequest = worker.messages.find((message) =>
      (message as { type?: string }).type === "set-simulation-running") as { requestId: number };
    assert.ok(pauseRequest);
    assert.equal(worker.messages.filter((message) => (message as { type?: string }).type === "draw").length, 1,
      "only the already-dispatched frame may remain ahead of the pause edge");

    worker.emit({
      type: "frame", frameId: 1,
      metrics: { context: "running", methodId: "octree", presentationSubmitted: true },
      snapshot: {
        presentationRevision: 0, presentationResolution: "640 × 360", completedPresentationCount: 1,
        latestPixelTrace: undefined, pixelTraceRevision: 0, pixelTraceAvailable: false,
        pixelTraceStatus: "path-inactive", pixelTraceAnswersRequest: false, pixelTraceStale: false,
        latestFluidCellTrace: undefined, fluidCellTraceRevision: 0, fluidCellTraceReady: false,
        fluidCellTraceFineBand: undefined,
      },
    });
    worker.emit({ type: "simulation-running-set", requestId: pauseRequest.requestId, submittedTime_s: 0.125 });
    assert.equal(await pause, 0.125);
    assert.equal(worker.messages.filter((message) => (message as { type?: string }).type === "draw").length, 1,
      "the queued running-intent frame must be discarded, not replayed after pause");

    draw();
    assert.equal(worker.messages.filter((message) => (message as { type?: string }).type === "draw").length, 2,
      "the next fresh paused snapshot may resume presentation after acknowledgement");
  } finally {
    if (previousWorker) Object.defineProperty(globalThis, "Worker", previousWorker);
    else Reflect.deleteProperty(globalThis, "Worker");
  }
});

test("worker frames carry a retained scene revision instead of cloning the document", () => {
  class FakeWorker {
    static instance: FakeWorker;
    readonly messages: unknown[] = [];
    constructor() { FakeWorker.instance = this; }
    addEventListener() {}
    postMessage(message: unknown) { this.messages.push(message); }
    terminate() {}
  }
  const previousWorker = Object.getOwnPropertyDescriptor(globalThis, "Worker");
  Object.defineProperty(globalThis, "Worker", { configurable: true, value: FakeWorker });
  try {
    const canvas = {
      transferControlToOffscreen: () => ({}),
      getBoundingClientRect: () => ({ width: 640, height: 360 }),
    } as unknown as HTMLCanvasElement;
    const client = new WebGPURenderWorkerClient(canvas, { onStatus() {} });
    const terrain = {
      baseHeight_m: 0.2, features: [],
      grid: {
        kind: "grid", origin_m: { x: -1, z: -1 }, spacing_m: 1,
        size: { nx: 2, nz: 2 }, heights_m: [0.2, 0.2, 0.2, 0.2],
      },
    };
    const firstScene = { terrain };
    const drawArgs = Array(16).fill(undefined);
    drawArgs[1] = firstScene;
    drawArgs[7] = { methodId: "octree" };
    const draw = () => (client.draw as unknown as (...args: unknown[]) => unknown)(...drawArgs);

    client.setSimulationScene(firstScene as never);
    client.setSimulationScene(firstScene as never);
    draw();
    draw();
    const worker = FakeWorker.instance;
    const renderScenes = worker.messages.filter((message) => (message as { type?: string }).type === "set-render-scene") as Array<{
      revision: number; scene: unknown; terrainContentStamp: string;
    }>;
    assert.equal(renderScenes.length, 1, "an unchanged immutable document is published only once");
    assert.equal(worker.messages.filter((message) => (message as { type?: string }).type === "set-simulation-scene").length, 1,
      "the solver override is also change-only");
    const frame = worker.messages.find((message) => (message as { type?: string }).type === "draw") as {
      sceneRevision: number; args: unknown[];
    };
    assert.equal(frame.sceneRevision, renderScenes[0].revision);
    assert.equal(frame.args.length, 15);
    assert.ok(!frame.args.includes(firstScene), "the scene document must not ride on the frame message");

    drawArgs[1] = {
      terrain: { ...terrain, grid: { ...terrain.grid, heights_m: [0.2, 0.2, 0.3, 0.2] } },
    };
    draw();
    const changed = worker.messages.filter((message) => (message as { type?: string }).type === "set-render-scene") as typeof renderScenes;
    assert.equal(changed.length, 2);
    assert.notEqual(changed[1].revision, changed[0].revision);
    assert.notEqual(changed[1].terrainContentStamp, changed[0].terrainContentStamp,
      "a same-shape brush edit must cross the seam with a new receiver stamp");
  } finally {
    if (previousWorker) Object.defineProperty(globalThis, "Worker", previousWorker);
    else Reflect.deleteProperty(globalThis, "Worker");
  }
});

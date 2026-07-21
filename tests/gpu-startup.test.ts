import assert from "node:assert/strict";
import test from "node:test";
import {
  acquireBrowserGPULease,
  automaticGPURecoveryEnabled,
  BROWSER_GPU_LOCK_NAME,
  optionalBrowserTimestampFeatures,
  resolveGPUStartupMode,
  safeBrowserGPUBringupEnabled,
  safeBrowserGPUBringupViolations,
  safeBrowserSimulationEpochChanged,
  shutdownBrowserGPUSession,
} from "../lib/gpu-startup";
import { FluidLabRenderer, type GPUStatus } from "../lib/webgpu-renderer";
import { getMethod, resolveMethodValues } from "../lib/methods";

const dam = { presetId: "water-box-dam-break", methodId: "octree" };

test("GPU startup query has explicit safe, manual, and automatic modes", () => {
  assert.equal(resolveGPUStartupMode("?gpu=off", dam), "off");
  assert.equal(resolveGPUStartupMode("?gpu=safe", dam), "safe");
  assert.equal(resolveGPUStartupMode("?gpu=manual", dam), "manual");
  assert.equal(resolveGPUStartupMode("?gpu=on", dam), "automatic");
  assert.equal(resolveGPUStartupMode("", dam), "manual", "the default octree dam break must wait for consent");
  assert.equal(resolveGPUStartupMode("", { ...dam, presetId: "water-box-tank-fill" }), "automatic");
  assert.equal(resolveGPUStartupMode("", { ...dam, methodId: "cpu" }), "automatic");
  assert.equal(resolveGPUStartupMode("?gpu=unexpected", dam), "manual", "unknown input must not bypass the safe default");
});

test("safe browser bring-up fails closed when the bounded workload drifts", () => {
  assert.equal(safeBrowserGPUBringupEnabled("?gpu=safe"), true);
  assert.equal(safeBrowserGPUBringupEnabled("?gpu=manual"), false);
  const canonicalMethodValues = resolveMethodValues(getMethod("octree"), "balanced", {});
  const valid = {
    presetId: "water-box-dam-break",
    methodId: "octree",
    quality: "balanced",
    methodValues: canonicalMethodValues,
    canonicalMethodValues,
    exactScene: true,
    voxelRenderMode: "smooth",
    svoRenderMode: "raster",
    diagnosticsOpen: false,
    rightPanel: null,
    gridOverlayAxis: "off",
    stageCapturePhase: "idle",
    search: "?gpu=safe",
  } as const;
  assert.deepEqual(safeBrowserGPUBringupViolations(valid), []);
  const violations = safeBrowserGPUBringupViolations({
    ...valid,
    quality: "ultra",
    methodValues: { ...valid.methodValues, unexpectedSpatialControl: 768, pressureIterations: 400, sparseSurfaceBandCells: 16, sparseSurfacePageFraction: 1 },
    diagnosticsOpen: true,
    rightPanel: "performance",
    stageCapturePhase: "reading",
    search: "?gpu=safe&gpuTimestamps=1&panel=performance",
  });
  for (const expected of [
    "quality must be balanced", "diagnostics panel must remain closed",
    "all right-side panels must remain closed", "GPU stage capture/readback must be idle", "GPU timestamps must be off",
  ]) assert.ok(violations.includes(expected), `missing violation: ${expected}`);
  assert.ok(violations.some((value) => value.includes("unexpectedSpatialControl") && value.includes("pressureIterations") && value.includes("sparseSurfaceBandCells") && value.includes("sparseSurfacePageFraction")));
  assert.ok(violations.some((value) => value.includes("gpuTimestamps") && value.includes("panel")));
});

test("safe browser session invalidates any reset or rebuild epoch", () => {
  assert.equal(safeBrowserSimulationEpochChanged(true, true, 7, 7), false);
  assert.equal(safeBrowserSimulationEpochChanged(true, true, 7, 8), true);
  assert.equal(safeBrowserSimulationEpochChanged(true, false, 7, 8), false, "pre-consent URL hydration is allowed");
  assert.equal(safeBrowserSimulationEpochChanged(false, true, 7, 8), false, "normal product modes preserve rebuild behavior");
});

test("browser WebGPU lease excludes a second tab and can be explicitly released", async () => {
  let occupied = false;
  const manager = {
    async request(
      name: string,
      options: { mode: "exclusive"; ifAvailable: true },
      callback: (lock: { readonly name: string } | null) => Promise<void>,
    ) {
      assert.equal(name, BROWSER_GPU_LOCK_NAME);
      assert.deepEqual(options, { mode: "exclusive", ifAvailable: true });
      if (occupied) return callback(null);
      occupied = true;
      try { await callback({ name }); }
      finally { occupied = false; }
    },
  };
  const first = await acquireBrowserGPULease(manager);
  assert.equal(first.status, "acquired");
  const second = await acquireBrowserGPULease(manager);
  assert.equal(second.status, "held");
  if (first.status === "acquired") first.release();
  await new Promise<void>((resolve) => setImmediate(resolve));
  const third = await acquireBrowserGPULease(manager);
  assert.equal(third.status, "acquired");
  if (third.status === "acquired") third.release();
  assert.equal((await acquireBrowserGPULease(undefined)).status, "unsupported");
});

test("session shutdown keeps the lease until renderer drain and pending acquisition settle", async () => {
  let resolveDrain!: () => void;
  let resolveLease!: (lease: { status: "acquired"; release: () => void }) => void;
  const drain = new Promise<void>((resolve) => { resolveDrain = resolve; });
  const acquisition = new Promise<{ status: "acquired"; release: () => void }>((resolve) => { resolveLease = resolve; });
  let released = false;
  let complete = false;
  const shutdown = shutdownBrowserGPUSession({ shutdown: () => drain }, acquisition).then(() => { complete = true; });
  resolveLease({ status: "acquired", release: () => { released = true; } });
  await Promise.resolve();
  assert.equal(released, false, "lease must remain held while renderer work drains");
  assert.equal(complete, false);
  resolveDrain();
  await shutdown;
  assert.equal(released, true);
  assert.equal(complete, true);

  let pendingLeaseReleased = false;
  let resolvePendingLease!: (lease: { status: "acquired"; release: () => void }) => void;
  const pendingLease = new Promise<{ status: "acquired"; release: () => void }>((resolve) => { resolvePendingLease = resolve; });
  complete = false;
  const stoppedDuringLease = shutdownBrowserGPUSession({ shutdown: async () => {} }, pendingLease).then(() => { complete = true; });
  await Promise.resolve();
  assert.equal(complete, false, "STOP must not finish while a lease request can still acquire");
  resolvePendingLease({ status: "acquired", release: () => { pendingLeaseReleased = true; } });
  await stoppedDuringLease;
  assert.equal(pendingLeaseReleased, true);
});

test("renderer shutdown waits for requestDevice and destroys its late device", async (t) => {
  const previousLocation = Object.getOwnPropertyDescriptor(globalThis, "location");
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  let resolveDevice!: (device: GPUDevice) => void;
  let deviceDestroyed = 0;
  let contextRequests = 0;
  const device = { destroy: () => { deviceDestroyed += 1; } } as GPUDevice;
  const adapter = {
    features: new Set<GPUFeatureName>(),
    limits: { maxStorageBuffersPerShaderStage: 10, maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024, maxTextureDimension3D: 2048 },
    requestDevice: () => new Promise<GPUDevice>((resolve) => { resolveDevice = resolve; }),
  } as unknown as GPUAdapter;
  Object.defineProperty(globalThis, "location", { configurable: true, value: { search: "?gpu=manual" } });
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { requestAdapter: async () => adapter } } });
  t.after(() => {
    for (const [name, descriptor] of [["location", previousLocation], ["navigator", previousNavigator]] as const) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
  });
  const renderer = new FluidLabRenderer({ getContext: () => { contextRequests += 1; return null; } } as unknown as HTMLCanvasElement, () => {});
  const initializing = renderer.initialize();
  await Promise.resolve();
  const shutdown = renderer.shutdown();
  let shutdownComplete = false;
  void shutdown.then(() => { shutdownComplete = true; });
  await Promise.resolve();
  assert.equal(shutdownComplete, false);
  resolveDevice(device);
  await Promise.all([initializing, shutdown]);
  assert.equal(deviceDestroyed, 1);
  assert.equal(contextRequests, 0, "a disposed renderer must not continue into canvas initialization");
});

test("renderer destroys a requested device when the WebGPU canvas context is absent", async (t) => {
  const previousLocation = Object.getOwnPropertyDescriptor(globalThis, "location");
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  let deviceDestroyed = 0;
  const device = { destroy: () => { deviceDestroyed += 1; } } as GPUDevice;
  const adapter = {
    features: new Set<GPUFeatureName>(),
    limits: { maxStorageBuffersPerShaderStage: 10, maxStorageBufferBindingSize: 128 * 1024 * 1024, maxBufferSize: 256 * 1024 * 1024, maxTextureDimension3D: 2048 },
    requestDevice: async () => device,
  } as unknown as GPUAdapter;
  Object.defineProperty(globalThis, "location", { configurable: true, value: { search: "?gpu=manual" } });
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { requestAdapter: async () => adapter } } });
  t.after(() => {
    for (const [name, descriptor] of [["location", previousLocation], ["navigator", previousNavigator]] as const) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
  });
  const statuses: GPUStatus[] = [];
  const renderer = new FluidLabRenderer({ getContext: () => null } as unknown as HTMLCanvasElement, (status) => statuses.push(status));
  await renderer.initialize();
  assert.equal(deviceDestroyed, 1);
  assert.equal(statuses.at(-1)?.state, "unavailable");
});

test("renderer shutdown waits for an in-flight solver initialization", async () => {
  let resolveSolver!: () => void;
  const solverPending = new Promise<void>((resolve) => { resolveSolver = resolve; });
  let deviceDestroyed = 0;
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
  const internals = renderer as unknown as { gpuFluidPending: Promise<void>; device: { destroy(): void } };
  internals.gpuFluidPending = solverPending;
  internals.device = { destroy: () => { deviceDestroyed += 1; } };
  const shutdown = renderer.shutdown();
  let complete = false;
  void shutdown.then(() => { complete = true; });
  await Promise.resolve();
  assert.equal(deviceDestroyed, 1, "device teardown starts immediately");
  assert.equal(complete, false, "lease-facing shutdown must wait for solver host tasks");
  resolveSolver();
  await shutdown;
  assert.equal(complete, true);
});

test("automatic device recovery is opt-in", () => {
  assert.equal(automaticGPURecoveryEnabled(""), false);
  assert.equal(automaticGPURecoveryEnabled("?gpuRecovery=0"), false);
  assert.equal(automaticGPURecoveryEnabled("?gpuRecovery=1"), true);
});

test("browser timestamp queries default on for profiling with explicit safe-mode and driver opt-outs", () => {
  const supported = new Set(["timestamp-query"]);
  assert.deepEqual(optionalBrowserTimestampFeatures("", supported), ["timestamp-query"]);
  assert.deepEqual(optionalBrowserTimestampFeatures("?gpuTimestamps=0", supported), []);
  assert.deepEqual(optionalBrowserTimestampFeatures("?gpuTimestamps=1", supported), ["timestamp-query"]);
  assert.deepEqual(optionalBrowserTimestampFeatures("?gpu=safe", supported), []);
  assert.deepEqual(optionalBrowserTimestampFeatures("?gpuTimestamps=1", new Set()), []);
});

test("gpu=off returns before requesting a WebGPU adapter", async (t) => {
  let adapterRequests = 0;
  const previousLocation = Object.getOwnPropertyDescriptor(globalThis, "location");
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "location", { configurable: true, value: { search: "?gpu=off" } });
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: {
    requestAdapter: async () => { adapterRequests += 1; throw new Error("requestAdapter must not run"); },
  } } });
  t.after(() => {
    for (const [name, descriptor] of [["location", previousLocation], ["navigator", previousNavigator]] as const) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
  });

  const statuses: GPUStatus[] = [];
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, (status) => statuses.push(status));
  await renderer.initialize();
  assert.equal(adapterRequests, 0);
  assert.deepEqual(statuses, [{ state: "unavailable", label: "WebGPU disabled by gpu=off (UI-only mode)" }]);
  renderer.destroy();
});

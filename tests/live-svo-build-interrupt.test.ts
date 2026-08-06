import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { getScenePreset } from "../lib/scenes";
import { WebGPULiveSvoScene } from "../lib/webgpu-live-svo-scene";
import { isCooperativeBuildAbort } from "../lib/cooperative-build";

/**
 * What a superseded scene build must actually do, on real hardware.
 *
 * The unit lane next door proves the driver stops a generator; only a device
 * can prove that stopping it releases the arenas. Both claims are needed: a
 * build that aborts promptly but leaks its tree turns every slider move at
 * environment refinement depth 3 into permanent device pressure, which is worse
 * than the freeze it replaced.
 */
const modulePath = process.env.WEBGPU_NODE_MODULE;

async function dawnDevice(): Promise<GPUDevice> {
  const dawn = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "no adapter — this lane did not execute on the GPU");
  return adapter.requestDevice();
}

test("a superseded live-SVO build stops promptly and leaves the device where it found it", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for Dawn validation",
}, async () => {
  process.env.FLUID_SVO_GROUND_SHELL ??= "1";
  process.env.FLUID_SVO_REFINEMENT_MODE ??= "surface";
  const device = await dawnDevice();
  const scene = getScenePreset("hero-garden-hose").create();

  // Refinement depth 1 rather than 3: this asserts the *mechanism*, and a depth
  // whose build is minutes long would make the lane's own runtime the thing
  // under test. Depth 1 is still far longer than the abort deadline below.
  const abort = new AbortController();
  const stages: string[] = [];
  const started_ms = performance.now();
  const superseded = WebGPULiveSvoScene.create(
    device, scene, "balanced", (progress) => { stages.push(progress.label); }, abort.signal,
    { environmentRefinementDepth: 1 },
  );
  // Abort well inside the build. The plan alone takes far longer than this.
  await new Promise((resolve) => setTimeout(resolve, 120));
  const abortedAt_ms = performance.now();
  abort.abort();
  await assert.rejects(superseded, (error: unknown) => isCooperativeBuildAbort(error));
  const timeToAbort_ms = performance.now() - abortedAt_ms;

  // The bound that matters: an abandoned build stops on the order of a slice,
  // not on the order of the work it had left. Generous against the driver's
  // 8 ms slice so the assertion survives a loaded machine, and still two orders
  // of magnitude below the whole-build time it replaces.
  assert.ok(timeToAbort_ms < 3000,
    `a superseded build took ${timeToAbort_ms.toFixed(0)} ms to stop`);
  assert.ok(performance.now() - started_ms < 60_000);
  // Progress must have streamed rather than arriving in one jump at the end,
  // or the label a user reads is still "allocating" for the whole build.
  assert.ok(stages.length >= 2, `the build reported only ${stages.length} stages before it was abandoned`);

  // The device must still serve a whole build afterwards. A leaked arena or a
  // half-published tree from the abandoned attempt shows up here, because this
  // one allocates against whatever the first one failed to release.
  const replacement = await WebGPULiveSvoScene.create(
    device, scene, "balanced", () => {}, undefined, { environmentRefinementDepth: 1 });
  assert.equal(replacement.builtRefinementDepth, 1);
  assert.ok(replacement.info.allocatedBytes > 0);
  replacement.destroy();
});

test("a build interleaves with the event loop instead of holding it", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for Dawn validation",
}, async () => {
  process.env.FLUID_SVO_GROUND_SHELL ??= "1";
  process.env.FLUID_SVO_REFINEMENT_MODE ??= "surface";
  const device = await dawnDevice();
  const scene = getScenePreset("hero-garden-hose").create();

  // A posted message is what a `draw` request is in the render worker, so the
  // longest one waits here is the worst-case frame latency there.
  let worstLatency_ms = 0;
  let delivered = 0;
  let running = true;
  let sentAt_ms = 0;
  const channel = new MessageChannel();
  const ping = () => { if (running) { sentAt_ms = performance.now(); channel.port2.postMessage(0); } };
  channel.port1.onmessage = () => {
    worstLatency_ms = Math.max(worstLatency_ms, performance.now() - sentAt_ms);
    delivered += 1;
    ping();
  };
  ping();
  const solver = await WebGPULiveSvoScene.create(
    device, scene, "balanced", () => {}, undefined, { environmentRefinementDepth: 1 });
  running = false;
  channel.port1.close(); channel.port2.close();
  solver.destroy();

  assert.ok(delivered > 10, `only ${delivered} messages were serviced during the whole build`);
  assert.ok(worstLatency_ms < 3000,
    `a message queued during the build waited ${worstLatency_ms.toFixed(0)} ms`);
});

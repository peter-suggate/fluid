/** Native renderer + solver cadence probe. The offscreen target replaces only
 * the browser swap chain; drawWithDeferredPresentation and the 60 Hz,
 * latest-wins worker request loop are the production path.
 * node --import tsx tools/probe-nbflip-presentation.ts /tmp/pour.json
 * This measures submissions/completions, not compositor/display timestamps.
 */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { Session } from "node:inspector";
import "../lib/methods";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { defaultCamera, markSceneRevision } from "../lib/core/model";
import { getSceneDefinition, getScenePreset } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";

const out = process.argv[2] ?? "/tmp/pour-presentation.json";
const steps = Number(process.env.POUR_STEPS ?? 180);
paceDawnEventPolling();
const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
const adapter = await gpu.requestAdapter(); assert.ok(adapter);
let device: GPUDevice | undefined, target: GPUTexture | undefined;
const errors: string[] = [];
Object.defineProperty(navigator, "gpu", { value: {
  requestAdapter: async () => ({ features: adapter.features, limits: adapter.limits, info: adapter.info,
    requestDevice: async (descriptor: GPUDeviceDescriptor) => {
      device = await adapter.requestDevice(descriptor);
      managedGPUDevice(device, { requireWorkerRealm: false });
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      return device;
    } }), getPreferredCanvasFormat: () => "bgra8unorm",
} });
const canvas = { width: 800, height: 600, getContext: () => ({
  configure: () => { target = device!.createTexture({ size: [800, 600], format: "bgra8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT }); },
  getCurrentTexture: () => target!, unconfigure: () => {},
}) } as unknown as OffscreenCanvas;
const definition = getSceneDefinition("nbflip-figure-7-pour"), preset = getScenePreset(definition.id);
const scene = markSceneRevision(structuredClone(sceneDocument(definition)));
const quality = definition.methodProfile!.quality;
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
const rows: Record<string, unknown>[] = [], completions: { at: number; time: number }[] = [];
let started = 0, completed = 0, failure: Error | undefined;
const renderer = new FluidLabRenderer(canvas, status => {
  if (status.state === "unavailable" || status.state === "lost") failure = new Error(status.label);
}, undefined, undefined, time => { completed = time; if (started) completions.push({ at: performance.now() - started, time }); });
// Inspection only: the probe records the actual solver's submission ledger.
const internals = renderer as unknown as { gpuFluid?: { info: { submittedTime_s?: number; completedTime_s?: number }; simulationReady?: boolean }; gpuFluidPending?: Promise<unknown> };
const args = (time: number): Parameters<FluidLabRenderer["draw"]> => [time, scene, { ...defaultCamera, ...preset.camera }, [], undefined,
  { methodId: method.id, quality, values, simulationEpoch: 0, inFlightDepth: 2 },
  { axis: "off", position: 0 }, preset.background, preset.presentationMode];
const inspector = new Session(); inspector.connect();
const inspect = <T>(name: string) => new Promise<T>((resolve, reject) => inspector.post(name, (error, result) => error ? reject(error) : resolve(result as T)));
let timer: ReturnType<typeof setInterval> | undefined;
try {
  await renderer.initialize();
  renderer.draw(...args(0));
  await internals.gpuFluidPending;
  assert.ok(internals.gpuFluid && internals.gpuFluid.simulationReady !== false, failure?.message ?? "solver must initialize");
  await device!.queue.onSubmittedWorkDone();
  console.log(JSON.stringify({ phase: "ready", presentationMode: preset.presentationMode, dt }));
  await inspect("Profiler.enable"); await inspect("Profiler.start");
  started = performance.now();
  let busy = false, queued = false, finished = false;
  await new Promise<void>((resolve, reject) => {
    const draw = async () => {
      busy = true;
      try {
        const at = performance.now() - started;
        const before = internals.gpuFluid!.info.submittedTime_s ?? 0;
        // PaneClockHost supplies whole fixed steps, never the raw wall clock.
        const pending = renderer.drawWithDeferredPresentation(args(Math.min(steps, Math.floor(at / (1000 * dt))) * dt));
        const encode = performance.now() - started - at;
        const metrics = await pending;
        rows.push({ at, encode, elapsed: performance.now() - started - at, before,
          submitted: internals.gpuFluid!.info.submittedTime_s, presented: metrics.presentationSubmitted,
          completed, presentations: renderer.completedPresentationCount });
        if (failure) throw failure;
        if (completed >= steps * dt - 1e-8) { finished = true; resolve(); }
        if (at > 120000) throw new Error("Timed out awaiting completed presentation");
      } catch (error) { finished = true; reject(error); }
      finally { busy = false; if (queued && !finished) { queued = false; void draw(); } }
    };
    timer = setInterval(() => { if (finished) return; if (busy) queued = true; else void draw(); }, 1000 / 60);
  });
  assert.deepEqual(errors, []);
} catch (error) {
  failure = error instanceof Error ? error : new Error(String(error));
  throw error;
} finally {
  clearInterval(timer);
  const { profile } = await inspect<{ profile: unknown }>("Profiler.stop").catch(() => ({ profile: undefined }));
  inspector.disconnect();
  writeFileSync(out, JSON.stringify({ dt, steps, rows, completions, errors, failure: failure?.message }, null, 2) + "\n");
  if (profile) writeFileSync(`${out}.cpuprofile`, JSON.stringify(profile));
  await device?.queue.onSubmittedWorkDone(); renderer.destroy(); target?.destroy(); device?.destroy();
}

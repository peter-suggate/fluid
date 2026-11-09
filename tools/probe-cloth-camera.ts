/** Camera-motion cost in a cloth scene on the production renderer (Dawn/Metal).
 * Same draw loop as the render worker (reply-gated drawWithDeferredPresentation
 * at 60 Hz, the product's method and UI stores), alternating a still camera
 * with an orbiting one. Frame wall time is draw start to presentation retired.
 * CLOTH_CAMERA_SCENE=bridson-2002-rotating-sphere CLOTH_CAMERA_FRAMES=60 \
 *   node --import tsx tools/probe-cloth-camera.ts /tmp/cloth-camera.json
 */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import "../lib/methods";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { markSceneRevision, type CameraState } from "../lib/core/model";
import { cameraForPreset, getSceneDefinition, getScenePreset } from "../lib/core/scenes";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { createMethodStore, resolvedMethodValues } from "../lib/core/stores/method-store";
import { createUIStore } from "../lib/core/stores/ui-store";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { GPUPassTimestampRecorder } from "../lib/core/performance-trace";
import { encodeRgbPng } from "../lib/core/png-codec";
import { SvoRasterAo } from "../lib/svo/features/lighting-visibility/svo-raster-ao";
import { rasterLocalShadowViews } from "../lib/svo/features/lighting-visibility/raster-local-shadows";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";

const out = process.argv[2] ?? "/tmp/cloth-camera.json";
const sceneId = process.env.CLOTH_CAMERA_SCENE ?? "bridson-2002-rotating-sphere";
const framesPerPhase = Number(process.env.CLOTH_CAMERA_FRAMES ?? 60);
const motion = process.env.CLOTH_CAMERA_MOTION ?? "orbit"; // orbit | pan
const width = Number(process.env.CLOTH_CAMERA_WIDTH ?? 1280), height = Number(process.env.CLOTH_CAMERA_HEIGHT ?? 800);
const dpr = Number(process.env.CLOTH_CAMERA_DPR ?? 2);
const instrumentation = process.env.CLOTH_CAMERA_TRACE === "1";
/** Freeze the clock after warmup (a paused or finished run): only the camera changes. */
const frozen = process.env.CLOTH_CAMERA_FROZEN === "1";
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
const canvas = { width: width * dpr, height: height * dpr, getContext: () => ({
  configure: () => { target = device!.createTexture({ size: [width * dpr, height * dpr], format: "bgra8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC }); },
  getCurrentTexture: () => target!, unconfigure: () => {},
}) } as unknown as OffscreenCanvas;

const definition = getSceneDefinition(sceneId), preset = getScenePreset(sceneId);
const scene = markSceneRevision(preset.create());
const methods = createMethodStore();
if (definition.methodProfile) methods.getState().applyProfile(definition.methodProfile);
const method = methods.getState();
const config = { methodId: method.methodId, quality: method.quality, values: resolvedMethodValues(method), simulationEpoch: 0, inFlightDepth: 1 };
const ui = createUIStore().getState();
const flag = (name: string, fallback: boolean) => process.env[name] === undefined ? fallback : process.env[name] === "1";
const lighting = { shadowsEnabled: flag("CLOTH_CAMERA_SHADOWS", ui.svoShadowsEnabled), ambientOcclusionEnabled: flag("CLOTH_CAMERA_AO", ui.svoAmbientOcclusionEnabled),
  silhouetteRefinementEnabled: flag("CLOTH_CAMERA_SILHOUETTE", ui.silhouetteRefinementEnabled), coneTracingMode: ui.svoConeTracingMode,
  globalIlluminationEnabled: ui.svoGlobalIlluminationEnabled, worldGiCacheEnabled: ui.svoWorldGiCacheEnabled,
  latticeVisibilityEnabled: ui.svoLatticeVisibilityEnabled, primaryTraversal: ui.svoPrimaryTraversal, disabledStages: ui.disabledRenderStages };
const diagnostics = { stageView: ui.svoStageView, lightSlot: ui.svoStageLightSlot,
  maximumTraversalDepth: ui.svoMaximumTraversalDepth, maximumNodeVisits: ui.svoMaximumNodeVisits };
const overlay = { axis: ui.gridOverlayAxis, position: ui.gridOverlaySlice, mode: ui.gridOverlayMode, layers: ui.visualLayers, particles: ui.fluidParticleDisplay };
const bodies = process.env.CLOTH_CAMERA_NOBODIES === "1" ? [] : initializeRigidBodies(scene.rigidBodies);
const base = cameraForPreset(preset);
const cameraAt = (frame: number, moving: boolean): CameraState => !moving ? base : motion === "pan"
  ? { ...base, target_m: { ...base.target_m, x: base.target_m.x + .3 * Math.sin(frame * .07) } }
  : { ...base, azimuth_rad: base.azimuth_rad + .01 * frame };
const dt = scene.numerics.fixedDt_s;
const args = (time: number, camera: CameraState) => [time, scene, camera, bodies, undefined, config, overlay,
  preset.background, preset.presentationMode, ui.fluidSurfaceRenderMode, lighting, diagnostics, ui.svoRenderTuning, undefined] as unknown as Parameters<FluidLabRenderer["draw"]>;

let failure: Error | undefined;
const statuses: string[] = [];
let effective: { state: string; surfaceMesh?: { state: string }; lightingVisibility?: { state: string } } | undefined;
const renderer = new FluidLabRenderer(canvas, status => {
  statuses.push(`gpu ${status.state}: ${status.label}`);
  if (status.state === "unavailable" || status.state === "lost") failure = new Error(status.label);
}, undefined, undefined, undefined, status => (effective = status, statuses.push(`effective ${status.state}${status.failureReason ? ` (${status.failureReason})` : ""}: ${status.detail ?? ""} surfaceMesh=${status.surfaceMesh?.state ?? "-"} lighting=${status.lightingVisibility?.state ?? "-"}`)));
const internals = renderer as unknown as { gpuFluid?: { simulationReady?: boolean }; gpuFluidPending?: Promise<unknown> };
type Row = { phase: string; frame: number; draw_ms: number; wall_ms: number; presented: boolean; hash?: string; gpu?: Record<string, number>; gpuSpan_ms?: number };
const lastTrace = new Map<string, number>();
/** Per-pass GPU time of every traced presentation, by pass label (diagnostic only). */
const passSamples = new Map<string, number[]>();
let passPhase = "";
// Report the local shadow views the renderer derives once per distinct domain.
{ const update = SvoRasterAo.prototype.update, seen = new Set<string>();
  SvoRasterAo.prototype.update = function (this: SvoRasterAo, origin, extent, cellSize, direction, lights) {
    const key = [...origin, ...extent].join(",");
    if (!seen.has(key)) { seen.add(key); console.log(JSON.stringify({ shadowDomain: { origin, extent },
      views: rasterLocalShadowViews(lights, { origin, extent }).map(v => ({ index: v.index, pointFace: v.pointFace, tan: +v.tanHalfAngle.toFixed(3), near: +v.near.toFixed(3), far: +v.far.toFixed(3) })) })); }
    return update.call(this, origin, extent, cellSize, direction, lights);
  }; }
// CLOTH_CAMERA_FORCE_CASTERS=1: treat paused cloth as moving, so each frame
// pays the moving-caster shadow cost without the solver's step in the timing.
if (process.env.CLOTH_CAMERA_FORCE_CASTERS === "1") {
  const set = SvoRasterAo.prototype.setTriangleCasters; let forced = 0;
  SvoRasterAo.prototype.setTriangleCasters = function (this: SvoRasterAo, sources) {
    forced++; return set.call(this, sources.map(source => ({ ...source, tick: source.tick * 1e6 + forced })));
  };
}
const read = GPUPassTimestampRecorder.prototype.read;
GPUPassTimestampRecorder.prototype.read = async function (this: GPUPassTimestampRecorder) {
  const reading = await read.call(this);
  if (reading && !passPhase.startsWith("warmup")) for (const p of reading.passes) if (p.sampled) {
    const key = `${passPhase.replace(/-\d$/, "")} | ${p.label}`; (passSamples.get(key) ?? passSamples.set(key, []).get(key)!).push(p.duration_ms);
  }
  return reading;
};
/** Hash of the presented target: proves a moving frame was actually redrawn. */
const pngPath = process.env.CLOTH_CAMERA_PNG; let pngWritten = false;
async function targetHash(): Promise<string> {
  const w = width * dpr, h = height * dpr, pitch = Math.ceil(w * 4 / 256) * 256;
  const read = device!.createBuffer({ size: pitch * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device!.createCommandEncoder(); encoder.copyTextureToBuffer({ texture: target! }, { buffer: read, bytesPerRow: pitch }, [w, h]);
  device!.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(read.getMappedRange()); let a = 0, nonzero = 0;
  for (let i = 0; i < words.length; i++) { a = (Math.imul(a, 31) + words[i]!) | 0; if (words[i]) nonzero++; }
  if (pngPath && !pngWritten) {
    // CLOTH_CAMERA_PNG: the first hashed frame (bgra8unorm), for visual A/B.
    pngWritten = true; const rgb = new Uint8Array(w * h * 3), bytes = new Uint8Array(words.buffer);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = y * pitch + x * 4, o = (y * w + x) * 3; rgb[o] = bytes[i + 2]!; rgb[o + 1] = bytes[i + 1]!; rgb[o + 2] = bytes[i]!; }
    writeFileSync(pngPath, encodeRgbPng({ width: w, height: h, rgb }));
  }
  read.unmap(); read.destroy(); return `${(a >>> 0).toString(16)}/${nonzero}`;
}
const rows: Row[] = [];
try {
  if (instrumentation) usePerformanceInstrumentationStore.getState().setMode("timeline");
  await renderer.initialize();
  renderer.setViewportSize(width, height, dpr);
  const ready = performance.now();
  while (!internals.gpuFluid) {
    renderer.draw(...args(0, base));
    await internals.gpuFluidPending;
    await new Promise(r => setTimeout(r, 50));
    if (failure) throw failure;
    assert.ok(performance.now() - ready < 300_000, "solver must initialize");
  }
  console.log(JSON.stringify({ phase: "ready", after_ms: Math.round(performance.now() - ready) }));
  await device!.queue.onSubmittedWorkDone();
  // The frame is a placeholder until the full-scene presentation is active.
  while (!(effective?.state === "active" && (ui.svoPrimaryTraversal !== "mesh" || effective.surfaceMesh?.state === "ready")
    && effective.lightingVisibility?.state === ui.svoConeTracingMode)) {
    await renderer.drawWithDeferredPresentation(args(0, base)); await device!.queue.onSubmittedWorkDone();
    await new Promise(r => setTimeout(r, 20));
    if (failure) throw failure;
    assert.ok(performance.now() - ready < 600_000, `presentation must become active: ${statuses.at(-1)}`);
  }
  { const dry = (renderer as unknown as { svoDryScenePipeline?: { scene?: { lightRecords?: ArrayLike<number> } } }).svoDryScenePipeline;
    console.log(JSON.stringify({ lightRecordWords: dry?.scene?.lightRecords?.length, silhouette: lighting.silhouetteRefinementEnabled, cone: lighting.coneTracingMode, primary: lighting.primaryTraversal })); }
  console.log(JSON.stringify({ phase: "presentation-active", after_ms: Math.round(performance.now() - ready) }));
  let frame = 0;
  const run = async (phase: string, count: number, moving: boolean) => {
    passPhase = phase;
    for (let k = 0; k < count; k++, frame++) {
      const start = performance.now();
      const time = frozen && !phase.startsWith("warmup") ? 40 * dt : frame * dt;
      const metrics = await renderer.drawWithDeferredPresentation(args(time, cameraAt(frame, moving)));
      const draw_ms = performance.now() - start;
      await device!.queue.onSubmittedWorkDone();
      const wall_ms = performance.now() - start;
      const trace = metrics.presentation;
      const fresh = trace && lastTrace.get("id") !== trace.sampleId;
      if (trace) lastTrace.set("id", trace.sampleId);
      rows.push({ phase, frame, draw_ms, wall_ms, presented: metrics.presentationSubmitted, ...(k >= count - 4 ? { hash: await targetHash() } : {}),
        ...(fresh ? { gpuSpan_ms: trace.span_ms ?? trace.total_ms, gpu: Object.fromEntries(trace.phases.map(p => [p.label, +p.duration_ms.toFixed(3)])) } : {}) });
      if (failure) throw failure;
    }
  };
  await run("warmup-still", 20, false);
  await run("warmup-moving", 20, true);
  if (frozen) await renderer.setSimulationRunning(false);
  for (let round = 0; round < 2; round++) {
    await run(`still-${round}`, framesPerPhase, false);
    await run(`moving-${round}`, framesPerPhase, true);
  }
  assert.deepEqual(errors, []);
} catch (error) {
  failure = error instanceof Error ? error : new Error(String(error));
  throw error;
} finally {
  const median = (xs: number[]) => xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]! : NaN;
  const p90 = (xs: number[]) => xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length * .9)]! : NaN;
  const summary: Record<string, unknown> = {};
  for (const phase of new Set(rows.map(r => r.phase))) {
    const r = rows.filter(x => x.phase === phase);
    summary[phase] = { frames: r.length, presented: r.filter(x => x.presented).length,
      draw_ms_median: +median(r.map(x => x.draw_ms)).toFixed(2), wall_ms_median: +median(r.map(x => x.wall_ms)).toFixed(2), wall_ms_p90: +p90(r.map(x => x.wall_ms)).toFixed(2), hashes: r.flatMap(x => x.hash ? [x.hash] : []),
      gpu: (() => { const t = r.filter(x => x.gpu); if (!t.length) return undefined; const out: Record<string, number> = { span_ms: +median(t.map(x => x.gpuSpan_ms!)).toFixed(2) };
        for (const label of new Set(t.flatMap(x => Object.keys(x.gpu!)))) { const v = median(t.map(x => x.gpu![label] ?? 0)); if (v > .05) out[label] = +v.toFixed(2); } return out; })() };
  }
  const passRows = [...passSamples].map(([k, v]) => [k, median(v), v.length] as const).filter(([, m]) => m > .5).sort((a, b) => b[1] - a[1]);
  console.log(passRows.map(([k, m, n]) => `${m.toFixed(2).padStart(8)} ms  x${n}  ${k}`).join("\n"));
  console.log([...new Set(statuses)].slice(-30).join("\n"));
  console.log(JSON.stringify({ sceneId, motion, frozen, dpr, width, height, summary, failure: failure?.message }, null, 1));
  writeFileSync(out, JSON.stringify({ sceneId, motion, summary, rows, errors, failure: failure?.message }, null, 2) + "\n");
  await device?.queue.onSubmittedWorkDone(); renderer.destroy(); target?.destroy(); device?.destroy();
}

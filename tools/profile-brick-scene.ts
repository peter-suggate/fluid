/**
 * A catalog scene on NB-FLIP's brick engine through the app's createBrickSolver (the pool rule, or the scene's
 * authored brickPools), for step timing and per-kernel GPU time. One JSON line `PROF {...}`:
 * setup and bytes, the fenced step median (steps 9.., the volume steps left out), the per-kernel split-pass profile
 * (tools/gpu-kernel-timer.ts) over --profSteps steps from --profAt, the pools' high water, FAIL, and the volume.
 *
 *   node --import tsx tools/profile-brick-scene.ts --scene=bulk-rest-tank [--steps=120] [--profAt=110 --profSteps=10]
 *     [--nopresent] [--burst=N]
 *
 * Run under tools/gpu/fluid-gpu-run.sh timing. The split-pass steps run slow (a pass a dispatch): score a change on
 * the median, never on the profile's sum. Under tools/profile-probe-xctrace.ts (--probe=tools/profile-brick-scene.ts)
 * every split pass reaches Metal System Trace labelled `<pass>|<pipeline>`.
 *
 * The volume is the presentation volume texture's sum (cells) every 5th step. With --nopresent the presentation is off
 * and that texture stays the last presented one (t = 0), so the run measures no volume: it reports
 * `"volume": "unmeasured (presentation off)"` and no V0, curve, V1 or drift, so no gate can read a 0 from it.
 */
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { getMethod } from "../lib/core/method-registry";
import "../lib/methods/index";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { BRICK_ENGINE_SCENES } from "../lib/methods/narrow-band/brick-scene";
import { createBrickSolver } from "../lib/methods/narrow-band/brick-method";
import { H } from "../lib/methods/narrow-band/brick-abi";
import { gpuKernelTimer } from "./gpu-kernel-timer";

const args = process.argv.slice(2), opt = (k: string) => args.find(a => a.startsWith(`--${k}=`))?.slice(k.length + 3);
const id = opt("scene"), steps = Number(opt("steps") ?? 60);
if (!id) throw new Error("--scene=<catalog id> is required");
const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD), definition = getSceneDefinition(id), scene = sceneDocument(definition);
const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
// A probe may run a scene the brick does not yet list (the still tank): the list is the app's enable switch.
if (!BRICK_ENGINE_SCENES.includes(scene.sceneId ?? id)) (BRICK_ENGINE_SCENES as string[]).push(scene.sceneId ?? id);
paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization",
  ...(process.env.FLUID_WEBGPU_DAWN_FEATURES ? [`enable-dawn-features=${process.env.FLUID_WEBGPU_DAWN_FEATURES}`] : [])]).requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
const raw = await adapter.requestDevice({ requiredFeatures: (["subgroups", "float32-filterable", "timestamp-query"] as const).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const timer = gpuKernelTimer(raw), device = timer.device;
let gpuError = "";
device.addEventListener("uncapturederror", (e: GPUUncapturedErrorEvent) => { e.preventDefault(); gpuError ||= e.error.message.slice(0, 300); });
const out: Record<string, unknown> = { id, steps };
const t0 = performance.now();
try {
  const solver = await createBrickSolver(device, scene, values as Record<string, unknown>, quality);
  // The live-edit pipelines compile in the background after creation: let them land before any step is timed.
  await solver.waitForSimulationReady();
  const e = solver.brickEngine, dt = e.options.dt_s;
  const presented = !args.includes("--nopresent");
  if (!presented) {
    solver.setPresentationEnabled(false); out.volume = "unmeasured (presentation off)";
    process.stderr.write("profile-brick-scene: --nopresent: no volume measured (the presentation texture stays at t = 0)\n");
  }
  out.create_ms = Math.round(performance.now() - t0); out.lattice = e.options.lattice.join("x"); out.pools = e.options.capacities;
  out.MB = Math.round((e.bytes + solver.brickPresentation.bytes) / 1e6);
  const volume = async () => {
    await e.fence();
    const tex = solver.brickPresentation.volume, [w, hgt, dep] = [tex.width, tex.height, tex.depthOrArrayLayers], bpr = Math.ceil(4 * w / 256) * 256;
    const buf = device.createBuffer({ size: bpr * hgt * dep, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder(); enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: hgt }, [w, hgt, dep]); device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ); const f = new Float32Array(buf.getMappedRange()); let v = 0;
    for (let z = 0; z < dep; z++) for (let y = 0; y < hgt; y++) { const o = (z * hgt + y) * bpr / 4; for (let x = 0; x < w; x++) v += f[o + x]!; }
    buf.unmap(); buf.destroy(); return v;
  };
  const v0 = presented ? await volume() : NaN; if (presented) out.V0 = Math.round(v0);
  const ms: number[] = [], curve: number[] = [], trace: number[][] = [];
  const profSteps = Number(opt("profSteps") ?? 2), profAt = Number(opt("profAt") ?? Math.max(1, steps - profSteps + 1));
  let timedSteps = 0;
  for (let k = 1; k <= steps; k++) {
    timer.enabled = k >= profAt && k < profAt + profSteps; if (timer.enabled) timedSteps++;
    const a = performance.now();
    if (!solver.advanceTo(k * dt + 1e-12, [])) { out.refused = k; break; }
    await solver.assertSimulationHealthy();
    // Every 5th step reads the volume, and that step is left out of the median; so are the split-pass steps.
    if (presented && k % 5 === 0) curve.push(+(100 * ((await volume()) / Math.max(v0, 1e-9) - 1)).toFixed(3));
    else if (!timer.enabled) ms.push(performance.now() - a);
    if (k % 10 === 0 || k === steps) { const h = await e.read("set", 0, 256); trace.push([k, h[H.LIVE_F]!, h[H.LIVE_C]!, h[H.LIST_COUNT + 11]!, h[H.LIST_COUNT + 18]!]); }
  }
  timer.enabled = false;
  const burst = Number(opt("burst") ?? 0);
  if (burst > 0 && out.refused === undefined) {
    await e.fence(); const b0 = performance.now();
    for (let j = 1; j <= burst; j++) solver.advanceTo((steps + j) * dt + 1e-12, []);
    await e.fence(); out.burst_ms = +((performance.now() - b0) / burst).toFixed(2);
  }
  const prof = await timer.collect(), agg = new Map<string, number>();
  for (const [l, v] of prof.byLabel) { const k = l.replace(/step\d+\./, ""); agg.set(k, (agg.get(k) ?? 0) + v); }
  out.prof = [...agg].sort((x, y) => y[1] - x[1]).slice(0, 40).map(([l, v]) => [l, +(v / Math.max(1, timedSteps)).toFixed(3)]);
  out.profTotal = +([...prof.byLabel.values()].reduce((a, b) => a + b, 0) / Math.max(1, timedSteps)).toFixed(2);
  out.trace = trace; out.traceColumns = "k LIVE_F LIVE_C C_ALL C_ACT";
  const h = await e.read("set", 0, 256);
  Object.assign(out, { fail: h[H.FAIL], highF: h[H.HIGH_F], highC: h[H.HIGH_C], peakF: h[H.PEAK_F], peakC: h[H.PEAK_C], peakP: h[H.PEAK_PARTICLES] });
  if (presented) {
    out.curve = curve;
    if (out.refused === undefined) { const v1 = await volume(); out.V1 = Math.round(v1); out.drift = +((v1 - v0) / Math.max(v0, 1e-9)).toExponential(3); }
  }
  const w = ms.slice(Math.min(8, ms.length - 1)).sort((x, y) => x - y); out.median_ms = w.length ? +w[w.length >> 1]!.toFixed(2) : undefined;
  solver.destroy();
} catch (err) { out.error = (err as Error).message.slice(0, 400); }
if (gpuError) out.gpuError = gpuError;
console.log("PROF " + JSON.stringify(out));
process.exit(out.error || out.fail || out.refused !== undefined || gpuError ? 3 : 0);

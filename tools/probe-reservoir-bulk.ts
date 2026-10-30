/** The 16h bulk tier's gates on reservoir-seep (W4, docs/plans/tiered-adaptivity-huge-grids-2026-10-24.md).
 * Seep discharge over time (liquid volume past the wall), the jet's landing
 * (the furthest liquid x on the catch floor), volume drift, max speed, region
 * tiers and the bulk owner counts.
 * node --import tsx tools/probe-reservoir-bulk.ts [--cell=0.005] [--steps=1200] [--every=60]
 *   [--values={"bulkTier":"16h"}] [--closed] [--tiers=FILE.json] [--out=PATH.json]
 * --max-gib=N caps construction and live requested GPU bytes (default and maximum 6).
 * --stats-only skips dense spatial readback (phi drift remains freshly measured).
 * --height=N scales the reservoir depth; --pool=N reserves a bounded page pool for rest probes.
 * --width=N and --depth=N scale the box (the scene's widthScale and depthScale: both chambers about
 * the middle wall, and the span), at the same h and head.
 * --tiers: a JSON file of fluid.tiers fields over the scene's (e.g. {"start":"4h","bricks":512}).
 * --water=inviscid zeroes the scene's viscosity and surface tension.
 * --closed fills the throat: the hydrostatic rest gate.
 * --pipelined keeps up to the solver's receipt ring of frames in flight, as the
 * app does: a step's ms is then the admission period, not advance + fence.
 * --kernels=<step> times every dispatch from that step on (tools/gpu-kernel-timer.ts;
 * distorted, for attribution only); the report keeps every step's fenced ms. */
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import "../lib/methods";
import { createUniformSolver, readUniformFields, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { auditUniformGPUAllocations, uniformGPUAllocationClass } from "./uniform-gpu-allocation-audit";
import { UNIFORM_PAGE_POOL_STATUS as S } from "../lib/methods/uniform/uniform-page-pool";
import { BULK16_STATUS } from "../lib/methods/uniform/uniform-bulk-pages";
import { gpuKernelTimer } from "./gpu-kernel-timer";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import type { MethodParamValues } from "../lib/core/method-contract";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";

const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const cell = Number(option("cell") ?? 0.005), steps = Number(option("steps") ?? 1200), every = Number(option("every") ?? 60);
const closed = process.argv.includes("--closed");
const pipelined = process.argv.includes("--pipelined");
const kernelsFrom = option("kernels") === undefined ? Infinity : Number(option("kernels"));
const stepMs: number[] = [], kernels = new Map<string, number[]>();
const values: MethodParamValues = JSON.parse(option("values") ?? "{}");
process.env.WEBGPU_NODE_MODULE ??= resolve("node_modules/webgpu/index.js");
paceDawnEventPolling();
const height=Number(option("height")??1),width=Number(option("width")??1),depth=Number(option("depth")??1);
assert.ok([height,width,depth].every(n=>Number.isFinite(n)&&n>0),"--height, --width and --depth must be positive");
const base = getSceneDefinition(RESERVOIR_SEEP_ID);
const definition = { ...base, buildAt: (l: {cellSize_m:number}) => createReservoirSeepScene(l.cellSize_m,{heightScale:height,widthScale:width,depthScale:depth}) };
const { scene } = sceneDocumentAtLattice(definition, { cellSize_m: cell });
if (closed) {
  // The rest gate: the throat and its mouths filled, the passage's wetted
  // start dropped (it would sit inside the wall).
  scene.solidVoxels = scene.solidVoxels!.filter(v => v.operation === "fill");
  scene.fluid.initialLiquidVolumes = scene.fluid.initialLiquidVolumes!.slice(0, 1);
}
if(option("pool")!==undefined){
 const blocks=Number(option("pool"));assert.ok(Number.isSafeInteger(blocks)&&blocks>=4,"--pool must be an integer >= 4");
 scene.fluid.pagePool={...scene.fluid.pagePool,blocks};
}
const tiersFile = option("tiers");
if (tiersFile !== undefined) scene.fluid.tiers = { ...scene.fluid.tiers, ...JSON.parse(readFileSync(resolve(tiersFile), "utf8")) };
// --water=inviscid: zero viscosity and surface tension (the realtime water model).
if (option("water") === "inviscid") { scene.fluid.dynamicViscosity_Pa_s = 0; scene.fluid.surfaceTension_N_m = 0; }
scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
const tag = Object.entries(values).map(([k, v]) => `${k}-${v}`).join("_") || "default";
const out = resolve(option("out") ?? `artifacts/reservoir-bulk/${closed ? "rest" : "seep"}-${cell * 1000}mm-${tag}.json`);
const rows: Record<string, unknown>[] = [];
/** Requested bytes by allocation class (uniformGPUAllocationClass), at construction and at the end. */
type Snapshot = ReturnType<ReturnType<typeof auditUniformGPUAllocations>["snapshot"]>;
const byClass = (s: Snapshot) => { const c: Record<string, number> = {}; for (const r of s.resources) c[uniformGPUAllocationClass(r.label)] = (c[uniformGPUAllocationClass(r.label)] ?? 0) + r.bytes; return c; };
let memory: Record<string, unknown> = {};
await withUniformDevice("reservoir bulk probe", async raw => {
  const maxGiB=Number(option("max-gib")??6);
  assert.ok(Number.isFinite(maxGiB)&&maxGiB>0&&maxGiB<=6,"--max-gib must be in (0, 6]");
  const allocation=auditUniformGPUAllocations(raw,maxGiB*2**30);
  const timer = gpuKernelTimer(allocation.device), device = timer.device;
  device.addEventListener("uncapturederror", event => console.error("GPU validation:", event.error.message));
  const solver = await createUniformSolver(device, scene, { ...definition.methodProfile!.overrides, ...values });
  const built = allocation.snapshot();
  memory = { construction: { bytes: built.liveBytes, byClass: byClass(built), resources: built.resources.map(r => [r.label, r.count, r.bytes]) } };
  const frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
  const { nx, ny, nz } = solver.info;
  const high = 16 * Math.floor(nx / 32);
  const read = device.createBuffer({ size: 4 * Math.max(8, S.words, BULK16_STATUS.words), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const words = async (buffer: GPUBuffer, offset: number) => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, offset, read, 0, 32);
    device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
    const w = [...new Uint32Array(read.getMappedRange()).slice(0, 8)]; read.unmap(); return w;
  };
  let wall = 0;
  try {
    for (let step = 1; step <= steps; step++) {
      if(step%every===0||step===steps)frame.requestNarrowBandVolumeProbe();
      const t = performance.now();
      timer.enabled = step >= kernelsFrom;
      assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s, []));
      // --pipelined: as the app, admit the next advance once the receipt ring has room; drain only on report steps.
      if (!pipelined || step % every === 0 || step === steps) await solver.awaitFrameCompletion();
      else while (solver.framePending) await new Promise(r => setImmediate(r));
      timer.enabled = false;
      const ms = performance.now() - t; wall += ms; stepMs.push(ms);
      if (step >= kernelsFrom) for (const [k, v] of (await timer.collect()).byLabel) { const list = kernels.get(k) ?? []; list.push(v); kernels.set(k, list); }
      assert.equal(solver.info.simulationPipelineError, undefined);
      if (step % every !== 0 && step !== steps) continue;
      const tiers = (await words(frame.heat.tiers, 4 * frame.heat.regions)).slice(0, 3);
      const bulk = frame.bulk ? await frame.bulk.readCounts() : undefined;
      const stats = await solver.readStats();
      // The page pool's receipt: resident and free blocks.
      let pool: { resident: number; free: number } | undefined;
      { const encoder = device.createCommandEncoder(); frame.pagePool.encodeReceipt(encoder, read, 0); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ); const w = new Uint32Array(read.getMappedRange().slice(0, 4 * S.words)); read.unmap(); pool = { resident: w[S.resident]!, free: w[S.free]! }; }
      // The BULK16 store's receipt: whole pages held in 16h slots (no pool block).
      let bulk16: { live: number; capacity: number; refused: number } | undefined;
      if (frame.bulkPages) { const encoder = device.createCommandEncoder(); frame.bulkPages.encodeReceipt(encoder, read, 0); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ); const w = new Uint32Array(read.getMappedRange().slice(0, 4 * BULK16_STATUS.words)); read.unmap();
        bulk16 = { live: w[BULK16_STATUS.live]!, capacity: w[BULK16_STATUS.capacity]!, refused: w[BULK16_STATUS.refused]! }; }
      const volume = process.argv.includes("--stats-only") ? undefined : (await readUniformFields(device, solver)).density;
      let escaped = 0, front = -1, total = 0, moment = 0;
      const floor = Math.max(2, Math.round(0.02 / cell));
      if(volume) for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const v = volume[x + nx * (y + ny * z)]!;
        total += v;
        if (x >= high) { escaped += v; moment += v * x; if (y < floor && v > 0.5 && x > front) front = x; }
      }
      const row = { step, t_s: step * scene.numerics.fixedDt_s, wall_ms: wall / step, tiers, bulk,
        escaped_L: escaped * cell ** 3 * 1000, front_m: front < 0 ? null : (front + 0.5) * cell, mean_x_m: escaped > 0 ? (moment / escaped + 0.5) * cell : null,
        volume: volume?total:undefined, drift: stats.volumeDrift, maxSpeed: stats.maxSpeed_m_s, fine: solver.info.uniformMixedFineTiles,
        residual: stats.uniformPressureAcceptedResidual, pool, bulk16, slots: stats.uniformMixedSlotsUsed, refused: stats.uniformMixedRefusedTiles,
        pressure: { cycles: stats.uniformPressureCyclesExecuted, band: stats.uniformPressureBandCycles, bandTiles: stats.uniformPressureBandTiles } };
      rows.push(row); console.log(JSON.stringify(row));
    }
    const live = allocation.snapshot();
    memory = { ...memory, live: { bytes: live.liveBytes, peak: live.peakBytes, byClass: byClass(live) } };
  } finally { read.destroy(); solver.destroy(); }
}, Number.isFinite(kernelsFrom) ? ["timestamp-query"] : []);
mkdirSync(dirname(out), { recursive: true });
const quantile = (list: number[], q: number) => { const s = [...list].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))]! : NaN; };
const kernelMedians_ms = Object.fromEntries([...kernels].map(([k, v]) => [k, quantile(v, 0.5)] as const).sort((a, b) => b[1] - a[1]));
// Means too: a frame's kernel time is their sum, which medians of skewed labels understate.
const kernelMeans_ms = Object.fromEntries([...kernels].map(([k, v]) => [k, v.reduce((s, x) => s + x, 0) / v.length] as const).sort((a, b) => b[1] - a[1]));
const late = stepMs.slice(Math.floor(stepMs.length / 2));
const timing = { median_ms: quantile(stepMs, 0.5), p95_ms: quantile(stepMs, 0.95), lateHalfMedian_ms: quantile(late, 0.5), lateHalfP95_ms: quantile(late, 0.95) };
console.log(JSON.stringify(timing));
writeFileSync(out, JSON.stringify({ date: new Date().toISOString(), cell, height, width, depth, steps, closed, memory, values, water: option("water") ?? "scene", timing, stepMs, kernelMedians_ms, kernelMeans_ms, rows }, null, 2) + "\n");
console.log(JSON.stringify({ out }));

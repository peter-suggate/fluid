/** The 16h bulk tier's gates on reservoir-seep (W4, docs/plans/tiered-adaptivity-huge-grids-2026-10-24.md).
 * Seep discharge over time (liquid volume past the wall), the jet's landing
 * (the furthest liquid x on the catch floor), volume drift, max speed, region
 * tiers and the bulk owner counts.
 * node --import tsx tools/probe-reservoir-bulk.ts [--cell=0.005] [--steps=1200] [--every=60]
 *   [--values={"bulkTier":"16h"}] [--closed] [--tiers=FILE.json] [--out=PATH.json]
 * --max-gib=N caps construction and live requested GPU bytes (default and maximum 6).
 * --stats-only skips dense spatial readback (phi drift remains freshly measured).
 * --height=N scales the reservoir depth; --pool=N reserves a bounded page pool for rest probes
 * (--pool=rule: the scene's pool by rule, uniformPagePoolRule).
 * --width=N and --depth=N scale the box (the scene's widthScale and depthScale: both chambers about
 * the middle wall, and the span), at the same h and head.
 * --box-width=K widens the box K times past its +x wall with the same water, walls and pool
 * (tools/scene-box-widen.ts): the scaling acceptance test's wider box.
 * --dispatches records each row's step dispatches (direct, indirect, direct workgroups;
 * tools/gpu-dispatch-counter.ts).
 * --tiers: a JSON file of fluid.tiers fields over the scene's (e.g. {"start":"4h","bricks":512}).
 * --water=inviscid zeroes the scene's viscosity and surface tension.
 * --closed fills the throat: the hydrostatic rest gate.
 * --scene=ID runs any catalog scene (its own lattice unless --cell is given); the seep's
 * geometry options do not apply, and --stats-only is implied.
 * --pure builds the pure BULK16 rest tank instead (lib/core/bulk-rest-tank-scene.ts):
 * --px/--py/--pz=N liquid pages a side (default 8/8/4); --height/--width/--depth are ignored.
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
import { gpuDispatchCounter } from "./gpu-dispatch-counter";
import { widenSceneBox } from "./scene-box-widen";
import { readInitializationCensus } from "../lib/core/gpu-initialization";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument, sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { createBulkRestTankScene, BULK_REST_TANK_ID } from "../lib/core/bulk-rest-tank-scene";
import type { MethodParamValues } from "../lib/core/method-contract";
import { initializeRigidBodies, quaternionRotate, type RigidBodyState } from "../lib/core/rigid-body";
import { rigidMotorAngle, rigidMotorPose, rigidMotorStrength } from "../lib/core/rigid-motor";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";

const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const sceneId = option("scene");
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
const pure = process.argv.includes("--pure");
const pages = { pagesX: Number(option("px") ?? 8), pagesY: Number(option("py") ?? 8), pagesZ: Number(option("pz") ?? 4) };
const base = getSceneDefinition(pure ? BULK_REST_TANK_ID : RESERVOIR_SEEP_ID);
const definition = { ...base, buildAt: (l: {cellSize_m:number}) => pure ? createBulkRestTankScene(l.cellSize_m, pages)
  : createReservoirSeepScene(l.cellSize_m,{heightScale:height,widthScale:width,depthScale:depth}) };
const scene = sceneId !== undefined
  ? (option("cell") === undefined ? sceneDocument(getSceneDefinition(sceneId)) : sceneDocumentAtLattice(getSceneDefinition(sceneId), { cellSize_m: cell }).scene)
  : sceneDocumentAtLattice(definition, { cellSize_m: cell }).scene;
if (sceneId !== undefined && !process.argv.includes("--stats-only")) process.argv.push("--stats-only");
assert.ok(!(pure && closed), "--pure has no throat to close");
if (closed) {
  // The rest gate: the throat and its mouths filled, the passage's wetted
  // start dropped (it would sit inside the wall).
  scene.solidVoxels = scene.solidVoxels!.filter(v => v.operation === "fill");
  scene.fluid.initialLiquidVolumes = scene.fluid.initialLiquidVolumes!.slice(0, 1);
}
if(option("pool")==="rule")delete scene.fluid.pagePool;
else if(option("pool")!==undefined){
 const blocks=Number(option("pool"));assert.ok(Number.isSafeInteger(blocks)&&blocks>=4,"--pool must be an integer >= 4");
 scene.fluid.pagePool={...scene.fluid.pagePool,blocks};
}
const tiersFile = option("tiers");
if (tiersFile !== undefined) scene.fluid.tiers = { ...scene.fluid.tiers, ...JSON.parse(readFileSync(resolve(tiersFile), "utf8")) };
// --water=inviscid: zero viscosity and surface tension (the realtime water model).
if (option("water") === "inviscid") { scene.fluid.dynamicViscosity_Pa_s = 0; scene.fluid.surfaceTension_N_m = 0; }
const boxWidth = Number(option("box-width") ?? 1);
Object.assign(scene, widenSceneBox(scene, boxWidth));
// --dt=S steps at another fixed interval (the letters gate runs at 1/60).
if (option("dt") !== undefined) { const dt = Number(option("dt")); assert.ok(dt > 0, "--dt must be positive"); scene.numerics = { ...scene.numerics, fixedDt_s: dt, maxDt_s: dt }; }
scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
const tag = Object.entries(values).map(([k, v]) => `${k}-${v}`).join("_") || "default";
const out = resolve(option("out") ?? `artifacts/reservoir-bulk/${sceneId ? sceneId : pure ? `pure-${pages.pagesX}x${pages.pagesY}x${pages.pagesZ}` : closed ? "rest" : "seep"}-${cell * 1000}mm-${tag}.json`);
const rows: Record<string, unknown>[] = [];
// The authored liquid boxes: their volume and their top faces' area (the t=0 free surface; overlaps not removed).
const boxes = (scene.fluid.initialLiquidVolumes ?? []).filter(v => v.shape === "box") as { min_m: { x: number; y: number; z: number }; max_m: { x: number; y: number; z: number } }[];
const liquid = { m3: boxes.reduce((n, b) => n + (b.max_m.x - b.min_m.x) * (b.max_m.y - b.min_m.y) * (b.max_m.z - b.min_m.z), 0),
  surface_m2: boxes.reduce((n, b) => n + (b.max_m.x - b.min_m.x) * (b.max_m.z - b.min_m.z), 0) };
/** Requested bytes by allocation class (uniformGPUAllocationClass), at construction and at the end. */
type Snapshot = ReturnType<ReturnType<typeof auditUniformGPUAllocations>["snapshot"]>;
const byClass = (s: Snapshot) => { const c: Record<string, number> = {}; for (const r of s.resources) c[uniformGPUAllocationClass(r.label)] = (c[uniformGPUAllocationClass(r.label)] ?? 0) + r.bytes; return c; };
let memory: Record<string, unknown> = {};
await withUniformDevice("reservoir bulk probe", async raw => {
  const maxGiB=Number(option("max-gib")??6);
  assert.ok(Number.isFinite(maxGiB)&&maxGiB>0&&maxGiB<=6,"--max-gib must be in (0, 6]");
  const allocation=auditUniformGPUAllocations(raw,maxGiB*2**30);
  const timer = gpuKernelTimer(allocation.device), counter = process.argv.includes("--dispatches") ? gpuDispatchCounter(timer.device) : undefined;
  const device = counter?.device ?? timer.device;
  device.addEventListener("uncapturederror", event => console.error("GPU validation:", event.error.message));
  const built_t0 = performance.now();
  const solver = await createUniformSolver(device, scene, { ...(sceneId !== undefined ? getSceneDefinition(sceneId) : definition).methodProfile?.overrides, ...values });
  const built = allocation.snapshot(), built_ms = performance.now() - built_t0;
  // A census start's BULK16 pages and the host's time for them (V2c).
  const bulkStart = (solver as unknown as { bulkStartInfo?: Record<string, number> }).bulkStartInfo;
  // A census or 4h start's t=0 h tiles and the pages that hold them (one h slot each), against the slots by rule.
  const seedTiles = (solver as unknown as { mixedSeed?: { fineTiles: Uint32Array } }).mixedSeed?.fineTiles;
  const seed = seedTiles ? { tiles: seedTiles.length, pages: new Set(Array.from(seedTiles, k => k >>> 9)).size, slotCapacity: solver.info.uniformMixedSlotCapacity } : undefined;
  // FLUID_GPU_INIT_CENSUS=1: construction's GPU tasks by phase and the slowest (lib/core/gpu-initialization); the rest of ms is the host's.
  const census = readInitializationCensus(), stages = census.length ? {
    tasks_ms: census.reduce((n, e) => n + e.elapsed_ms, 0),
    byPhase: Object.fromEntries(census.reduce((m, e) => m.set(e.phase, (m.get(e.phase) ?? 0) + e.elapsed_ms), new Map<string, number>())),
    slowest: [...census].sort((a, b) => b.elapsed_ms - a.elapsed_ms).slice(0, 15).map(e => [e.taskId, e.elapsed_ms]) } : undefined;
  memory = { construction: { ms: built_ms, stages, bulkStart, bytes: built.liveBytes, peak: built.peakBytes, seed, byClass: byClass(built), resources: built.resources.map(r => [r.label, r.count, r.bytes]) } };
  const frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
  const { nx, ny, nz } = solver.info;
  // The middle wall's +x face: on the scene's own cells, whatever --box-width.
  const high = 16 * Math.floor(nx / boxWidth / 32);
  const read = device.createBuffer({ size: 4 * Math.max(8, S.words, BULK16_STATUS.words), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const words = async (buffer: GPUBuffer, offset: number) => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, offset, read, 0, 32);
    device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
    const w = [...new Uint32Array(read.getMappedRange()).slice(0, 8)]; read.unmap(); return w;
  };
  let wall = 0;
  // The NB-FLIP records: the live count each receipt leaves (sampled every step) and its peak, against the store's capacity.
  const flip = frame.narrowBandFlip as unknown as { count: number; capacity: number; reseedClippedTotal: number; state: GPUBuffer; binSlots: number } | undefined;
  let particlePeak = 0, finePeak = 0, brickPeak = 0;
  // --bodies: the scene's bodies, each free body held at its t=0 pose and each
  // motor fixture (a paddle) turned on its parent at the parent's throttle by
  // its authored drive (rigidMotorPose), with the drive's velocity: a moored
  // boat's wheels. Without it the fluid sees no bodies.
  const bodies: RigidBodyState[] = process.argv.includes("--bodies") ? initializeRigidBodies(scene.rigidBodies) : [];
  const parents = new Map(bodies.map(b => [b.description.id, b] as const));
  const poseBodies = (time: number) => {
    for (const body of bodies) {
      const m = body.description.motorAttachment;
      if (!m) { body.held = true; continue; }
      const parent = parents.get(m.parentId)!, strength = rigidMotorStrength(parent.description, m.channel);
      const drive = { ...m, angularSpeed_rad_s: strength * m.angularSpeed_rad_s }, dt = scene.numerics.fixedDt_s;
      const now = rigidMotorPose(parent, drive, time), before = rigidMotorPose(parent, drive, time - dt);
      body.position_m = now.position_m; body.orientation = now.orientation;
      body.linearVelocity_m_s = { x: (now.position_m.x - before.position_m.x) / dt, y: (now.position_m.y - before.position_m.y) / dt, z: (now.position_m.z - before.position_m.z) / dt };
      const omega = (rigidMotorAngle(drive, time) - rigidMotorAngle(drive, time - dt)) / dt;
      const axis = quaternionRotate(parent.orientation, m.axis);
      body.angularVelocity_rad_s = { x: omega * axis.x, y: omega * axis.y, z: omega * axis.z };
    }
    return bodies;
  };
  try {
    for (let step = 1; step <= steps; step++) {
      if(step%every===0||step===steps)frame.requestNarrowBandVolumeProbe();
      const t = performance.now();
      timer.enabled = step >= kernelsFrom;
      if (counter) counter.enabled = true;
      assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s, poseBodies(step * scene.numerics.fixedDt_s)));
      // --pipelined: as the app, admit the next advance once the receipt ring has room; drain only on report steps.
      if (!pipelined || step % every === 0 || step === steps) await solver.awaitFrameCompletion();
      else while (solver.framePending) await new Promise(r => setImmediate(r));
      timer.enabled = false;
      // A pipelined step's passes are those encoded during it (its advance), not those it waited for.
      const counts = counter ? Object.values(counter.take()) : undefined;
      if (counter) counter.enabled = false;
      const dispatches = counts && counts.reduce((n, c) => ({ direct: n.direct + c.direct, indirect: n.indirect + c.indirect, workgroups: n.workgroups + c.workgroups }), { direct: 0, indirect: 0, workgroups: 0 });
      const ms = performance.now() - t; wall += ms; stepMs.push(ms);
      if (flip) particlePeak = Math.max(particlePeak, flip.count);
      finePeak = Math.max(finePeak, solver.info.uniformMixedFineTiles ?? 0);
      if (step >= kernelsFrom) for (const [k, v] of (await timer.collect()).byLabel) { const list = kernels.get(k) ?? []; list.push(v); kernels.set(k, list); }
      assert.equal(solver.info.simulationPipelineError, undefined);
      if (step % every !== 0 && step !== steps) continue;
      const tiers = (await words(frame.heat.tiers, 4 * frame.heat.regions)).slice(0, 3);
      const bulk = frame.bulk ? await frame.bulk.readCounts() : undefined;
      const stats = await solver.readStats();
      // The particle bricks the last sort held (the slab count, state word 20), against the cell bins' brick slots; sampled on report steps.
      // --criteria: the census's verdict a tile (the importance flags NB-FLIP
      // copies beside its heat) and its particle heat: the wet band tiles each
      // criterion triggers, those required at h, and the tiles holding heat.
      let criteria: Record<string, number> | undefined;
      if (flip && process.argv.includes("--criteria")) {
        const f = flip as unknown as { activityWord: number; activityHeatWord: number; tiles: number; bins: GPUBuffer };
        const big = device.createBuffer({ size: 12 * f.tiles, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(f.bins, 4 * f.activityWord, big, 0, 8 * f.tiles);
        encoder.copyBufferToBuffer(f.bins, 4 * f.activityHeatWord, big, 8 * f.tiles, 4 * f.tiles);
        device.queue.submit([encoder.finish()]); await big.mapAsync(GPUMapMode.READ);
        const range = big.getMappedRange(), scores = new Uint32Array(range, 0, 2 * f.tiles), heat = new Float32Array(range, 8 * f.tiles, f.tiles);
        criteria = { wet: 0, crossing: 0, required: 0, held: 0, dropped: 0, heated: 0, hot: 0, shape: 0, thin: 0, strain: 0, rotation: 0, impact: 0, approach: 0, only_shape: 0, only_thin: 0, only_strain: 0, only_rotation: 0, only_impact: 0, only_approach: 0 };
        const names = ["shape", "thin", "strain", "rotation", "impact", "approach"];
        for (let t = 0; t < f.tiles; t++) {
          const w = scores[2 * t + 1]!, bits = (w >>> 16) & 63;
          if (w & (1 << 29)) criteria.wet!++; if (w & (1 << 28)) criteria.crossing!++; if (w & (1 << 25)) criteria.required!++; if (w & (1 << 26)) criteria.held!++; if (w & (1 << 27)) criteria.dropped!++;
          for (let c = 0; c < 6; c++) if (bits & (1 << c)) { criteria[names[c]!]!++; if (bits === 1 << c) criteria[`only_${names[c]}`]!++; }
          if (heat[t]! > 0) criteria.heated!++; if (heat[t]! >= 2) criteria.hot!++;
        }
        big.unmap(); big.destroy();
      }
      // And the surface arena's windows at their greatest (state word 24), against the arena's windows.
      const slabWords = flip ? await words(flip.state, 80) : undefined;
      const bricks = slabWords?.[0], arenaPeak = slabWords?.[4];
      if (bricks !== undefined) brickPeak = Math.max(brickPeak, bricks);
      // The page pool's receipt: resident and free blocks.
      let pool: { resident: number; free: number } | undefined;
      { const encoder = device.createCommandEncoder(); frame.pagePool.encodeReceipt(encoder, read, 0); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ); const w = new Uint32Array(read.getMappedRange().slice(0, 4 * S.words)); read.unmap(); pool = { resident: w[S.resident]!, free: w[S.free]! }; }
      // The BULK16 store's receipt: whole pages held in 16h slots (no pool block).
      let bulk16: { live: number; capacity: number; refused: number; promoteRefused: number } | undefined;
      if (frame.bulkPages) { const encoder = device.createCommandEncoder(); frame.bulkPages.encodeReceipt(encoder, read, 0); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ); const w = new Uint32Array(read.getMappedRange().slice(0, 4 * BULK16_STATUS.words)); read.unmap();
        bulk16 = { live: w[BULK16_STATUS.live]!, capacity: w[BULK16_STATUS.capacity]!, refused: w[BULK16_STATUS.refused]!, promoteRefused: w[BULK16_STATUS.promoteRefused]! }; }
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
        finePeak, fineCeiling: frame.fineCeiling, residual: stats.uniformPressureAcceptedResidual, pool, bulk16, slots: stats.uniformMixedSlotsUsed, slotCapacity: stats.uniformMixedSlotCapacity, slotPeak: stats.uniformMixedSlotPeak, refused: stats.uniformMixedRefusedTiles,
        pressure: { cycles: stats.uniformPressureCyclesExecuted, band: stats.uniformPressureBandCycles, bandTiles: stats.uniformPressureBandTiles }, dispatches,
        criteria, particles: flip ? { live: flip.count, peak: particlePeak, capacity: flip.capacity, clipped: flip.reseedClippedTotal, bricks, brickPeak, binSlots: flip.binSlots, arenaPeak, arenaWindows: (flip as unknown as { arenaWindows: number }).arenaWindows } : undefined };
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
writeFileSync(out, JSON.stringify({ date: new Date().toISOString(), cell, pure: pure ? pages : undefined, height, width, depth, boxWidth, steps, closed, liquid, memory, values, water: option("water") ?? "scene", timing, stepMs, kernelMedians_ms, kernelMeans_ms, rows }, null, 2) + "\n");
console.log(JSON.stringify({ out }));

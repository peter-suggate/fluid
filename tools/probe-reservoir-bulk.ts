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
import { createUniformSolver, readUniformFields, readUniformLiveAddress, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { uniformTileKey } from "../lib/methods/uniform/uniform-lattice-address";
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
      // --particle-census: the live records by state (ballistic, coupled), heat (hot > 1, collar <= 1), depth and speed.
      let particleCensus: Record<string, number> | undefined;
      if (flip && process.argv.includes("--particle-census")) {
        const f = flip as unknown as { activeParticles: GPUBuffer; particleStride: number };
        const stride = f.particleStride, bytes = Math.min(f.activeParticles.size, 4 * stride * Math.ceil(1.5 * flip.count + 1024));
        const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        { const e = device.createCommandEncoder(); e.copyBufferToBuffer(f.activeParticles, 0, read, 0, bytes); device.queue.submit([e.finish()]); }
        await read.mapAsync(GPUMapMode.READ); const a = new Float32Array(read.getMappedRange().slice(0)), u = new Uint32Array(a.buffer); read.unmap(); read.destroy();
        const c: Record<string, number> = { found: 0, ballistic: 0, hot: 0, collar: 0, deep: 0, shallow: 0, air: 0, fast: 0, speedSum: 0, ballisticSpeedSum: 0 };
        const apic = stride === 28;
        for (let o = 0; o + stride <= a.length; o += stride) {
          if (!(a[o]! > 0 && a[o + 3]! > 0)) continue;
          c.found!++;
          const ballistic = apic ? a[o + 11] === 1 : (u[o + 11]! >>> 31) === 1;
          const speed = Math.hypot(a[o + 4]!, a[o + 5]!, a[o + 6]!), depth = a[o + 7]!;
          if (ballistic) { c.ballistic!++; c.ballisticSpeedSum! += speed; }
          if (a[o + 3]! > 1) c.hot!++; else c.collar!++;
          if (depth < -1) c.deep!++; else if (depth <= 0.5) c.shallow!++; else c.air!++;
          if (speed > 1) c.fast!++; c.speedSum! += speed;
        }
        c.meanSpeed = c.speedSum! / Math.max(1, c.found!); c.ballisticMeanSpeed = c.ballisticSpeedSum! / Math.max(1, c.ballistic!);
        delete c.speedSum; delete c.ballisticSpeedSum; particleCensus = c;
      }
      let criteria: Record<string, number> | undefined;
      if (flip && process.argv.includes("--criteria")) {
        const f = flip as unknown as { activityWord: number; activityHeatWord: number; tiles: number; bins: GPUBuffer };
        // Scores (2 words), heat, its spread target, new, theta, then the particles' surface heat: 7 words a tile.
        const big = device.createBuffer({ size: 28 * f.tiles, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(f.bins, 4 * f.activityWord, big, 0, 28 * f.tiles);
        device.queue.submit([encoder.finish()]); await big.mapAsync(GPUMapMode.READ);
        const range = big.getMappedRange(), scores = new Uint32Array(range, 0, 2 * f.tiles), heat = new Float32Array(range, 8 * f.tiles, f.tiles);
        const target = new Float32Array(range, 12 * f.tiles, f.tiles), carried = new Float32Array(range, 24 * f.tiles, f.tiles);
        criteria = { wet: 0, crossing: 0, required: 0, held: 0, dropped: 0, heated: 0, hot: 0, shape: 0, thin: 0, strain: 0, rotation: 0, impact: 0, approach: 0, only_shape: 0, only_thin: 0, only_strain: 0, only_rotation: 0, only_impact: 0, only_approach: 0 };
        const names = ["shape", "thin", "strain", "rotation", "impact", "approach"];
        for (let t = 0; t < f.tiles; t++) {
          const w = scores[2 * t + 1]!, bits = (w >>> 16) & 63;
          if (w & (1 << 29)) criteria.wet!++; if (w & (1 << 28)) criteria.crossing!++; if (w & (1 << 25)) criteria.required!++; if (w & (1 << 26)) criteria.held!++; if (w & (1 << 27)) criteria.dropped!++;
          for (let c = 0; c < 6; c++) if (bits & (1 << c)) { criteria[names[c]!]!++; if (bits === 1 << c) criteria[`only_${names[c]}`]!++; }
          if (heat[t]! > 0) criteria.heated!++; if (heat[t]! >= 2) criteria.hot!++;
          // Crossing tiles by particle state: core (heat), collar (spread heat only), carried (particle heat
          // in a cold tile), cold (none). Their shape scores (byte 0 of word 0, 64 = 1) and shape triggers.
          if (w & (1 << 28)) {
            const cls = heat[t]! > 0 ? "core" : target[t]! > 0 ? "collar" : carried[t]! > 0 ? "carried" : "cold";
            const score = (scores[2 * t]! & 255) / 64;
            criteria[`${cls}_n`] = (criteria[`${cls}_n`] ?? 0) + 1;
            criteria[`${cls}_shape`] = (criteria[`${cls}_shape`] ?? 0) + ((bits & 1) ? 1 : 0);
            criteria[`${cls}_score`] = (criteria[`${cls}_score`] ?? 0) + score;
            const bucket = score < 0.25 ? "lt025" : score < 0.5 ? "lt05" : score < 1 ? "lt1" : score < 2 ? "lt2" : "ge2";
            criteria[`${cls}_${bucket}`] = (criteria[`${cls}_${bucket}`] ?? 0) + 1;
          }
        }
        // --phi-profile (last row): phi along each crossing tile's centre column, in h, k vertices
        // from the lowest vertex below its first crossing, averaged by particle class. A signed
        // distance reads k - 0.5 on average; the h tiles' raw vertices, the 4h tiles' trilinear ones.
        if (process.argv.includes("--phi-profile") && step === steps) {
          const fields = await readUniformFields(device, solver), address = await readUniformLiveAddress(device, solver);
          const { nx, ny, nz } = solver.info, h = scene.voxelDomain.finestCellSize_m, K = 5;
          const raw = (x: number, y: number, z: number) => fields.phi[x + (nx + 1) * (y + (ny + 1) * z)]!;
          // PHI_SMOOTH=1: the same measures on the field after one [1 2 1]/4 binomial pass an axis.
          const plain = (x: number, y: number, z: number) => { x = Math.max(0, Math.min(nx, x)); y = Math.max(0, Math.min(ny, y)); z = Math.max(0, Math.min(nz, z)); return fields.widthAt(Math.min(x, nx - 1), Math.min(y, ny - 1), Math.min(z, nz - 1)) === 1 ? raw(x, y, z) : fields.vertex(x, y, z); };
          const W = [0.25, 0.5, 0.25], field = process.env.PHI_SMOOTH === "1" ? (x: number, y: number, z: number) => { let v = 0; for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let d = -1; d <= 1; d++) v += W[a + 1]! * W[b + 1]! * W[d + 1]! * plain(x + a, y + b, z + d); return v; } : plain;
          const sums: Record<string, { n: number; at: number[]; rough?: number; worst?: number; fit?: number; slope?: number; tn?: number; tmean?: number; tworst?: number; tplane?: number; cw?: number; cv?: number; cc?: number; ck?: number; cn?: number; multi?: number; blobs?: number; pockets?: number; terms?: number[][] }> = {};
          for (let tz = 0; 4 * tz < nz; tz++) for (let ty = 0; 4 * ty < ny; ty++) for (let tx = 0; 4 * tx < nx; tx++) {
            const t = uniformTileKey(address, tx, ty, tz);
            if (!(t >= 0 && t < f.tiles) || !(scores[2 * t + 1]! & (1 << 28))) continue;
            const x = 4 * tx + 2, z = 4 * tz + 2, fine = fields.widthAt(x, 4 * ty, z) === 1, phiAt = (y: number) => fine ? raw(x, y, z) : fields.vertex(x, y, z);
            let base = -1;
            for (let y = 4 * ty; y < 4 * ty + 4 && y < ny; y++) if ((phiAt(y) < 0) !== (phiAt(y + 1) < 0)) { base = y; break; }
            if (base < K || base + K + 1 > ny) continue;
            const cls = `${heat[t]! > 0 ? "core" : target[t]! > 0 ? "collar" : carried[t]! > 0 ? "carried" : "cold"}_${fine ? "h" : "4h"}`;
            const sum = sums[cls] ??= { n: 0, at: new Array(2 * K + 2).fill(0) };
            const sign = phiAt(base) < 0 ? 1 : -1; sum.n++;
            for (let k = -K; k <= K + 1; k++) sum.at[k + K]! += sign * phiAt(base + k) / h;
            // Roughness: the first crossing height of each of the tile's 5x5 vertex columns
            // (searched from a vertex below the tile to one above), a least-squares plane through
            // them, and its RMS and worst residual in h; and the tile-corner bilinear plane's worst.
            const heights: [number, number, number][] = [];
            for (let i = 0; i <= 4; i++) for (let j = 0; j <= 4; j++) {
              const cx = 4 * tx + i, cz = 4 * tz + j, at = (y: number) => field(cx, y, cz);
              for (let y = Math.max(0, 4 * ty - 2); y < Math.min(ny, 4 * ty + 6); y++) { const a = at(y), b = at(y + 1); if ((a < 0) !== (b < 0)) { heights.push([i, j, y + a / (a - b)]); break; } }
            }
            // The census's four terms (umSurfaceDisplacement, umVolumeError, umCurveError, umCornerCurveError), in h, on this field.
            if (fine) {
              const o = [4 * tx, 4 * ty, 4 * tz], F = (a: number, b: number, d: number) => field(o[0]! + a, o[1]! + b, o[2]! + d);
              const c8 = [0, 1].map(a => [0, 1].map(b => [0, 1].map(d => F(4 * a, 4 * b, 4 * d))));
              const tri = (x: number, y: number, z: number) => { let v = 0; for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let d = 0; d < 2; d++) v += (a ? x : 1 - x) * (b ? y : 1 - y) * (d ? z : 1 - z) * c8[a]![b]![d]!; return v; };
              let worstD = 0, along = 0, crossings = 0, curve = 0, curves = 0, corner = 0, kappa = 0;
              for (let a = 0; a <= 4; a++) for (let b = 0; b <= 4; b++) for (let d = 0; d <= 4; d++) {
                const v = [a, b, d], value = F(a, b, d);
                if (a >= 1 && a <= 3 && b >= 1 && b <= 3 && d >= 1 && d <= 3 && Math.abs(value) <= 1.5 * h) {
                  curve += (F(a - 1, b, d) + F(a + 1, b, d) + F(a, b - 1, d) + F(a, b + 1, d) + F(a, b, d - 1) + F(a, b, d + 1) - 6 * value) * (4 / 3) / h; curves++;
                  // The level set's own curvature, div(grad phi / |grad phi|) = (|g|^2 tr H - g.H.g) / |g|^3, in cells.
                  const P = (x: number, y: number, z: number) => F(a + x, b + y, d + z) / h;
                  const g = [(P(1, 0, 0) - P(-1, 0, 0)) / 2, (P(0, 1, 0) - P(0, -1, 0)) / 2, (P(0, 0, 1) - P(0, 0, -1)) / 2];
                  const e = (i: number) => [i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0];
                  const Hm = [0, 1, 2].map(i => [0, 1, 2].map(j => { if (i === j) { const u = e(i); return P(u[0]!, u[1]!, u[2]!) + P(-u[0]!, -u[1]!, -u[2]!) - 2 * P(0, 0, 0); } const u = e(i), w = e(j); return (P(u[0]! + w[0]!, u[1]! + w[1]!, u[2]! + w[2]!) - P(u[0]! - w[0]!, u[1]! - w[1]!, u[2]! - w[2]!) - P(-u[0]! + w[0]!, -u[1]! + w[1]!, -u[2]! + w[2]!) + P(-u[0]! - w[0]!, -u[1]! - w[1]!, -u[2]! - w[2]!)) / 4; }));
                  const gg = g[0]! ** 2 + g[1]! ** 2 + g[2]! ** 2, tr = Hm[0]![0]! + Hm[1]![1]! + Hm[2]![2]!;
                  let gHg = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) gHg += g[i]! * Hm[i]![j]! * g[j]!;
                  if (gg > 1e-6) kappa += (gg * tr - gHg) / gg ** 1.5 * (4 / 3);
                }
                if (a % 4 === 0 && b % 4 === 0 && d % 4 === 0 && Math.abs(value) <= 2 * h) {
                  const lap = F(a + 4, b, d) + F(a - 4, b, d) + F(a, b + 4, d) + F(a, b - 4, d) + F(a, b, d + 4) + F(a, b, d - 4) - 6 * value;
                  corner = Math.max(corner, Math.abs(lap) / (12 * h));
                }
                for (let ax = 0; ax < 3; ax++) {
                  if (v[ax]! >= 4) continue; const n = [...v]; n[ax]!++; const next = F(n[0]!, n[1]!, n[2]!);
                  if ((value < 0) === (next < 0)) continue;
                  const t = v.map(q => q / 4); t[ax]! += 0.25 * value / (value - next);
                  const e = 1e-3, g = [0, 1, 2].map(q => { const u = [...t], w = [...t]; u[q]! = Math.min(1, u[q]! + e); w[q]! = Math.max(0, w[q]! - e); return (tri(u[0]!, u[1]!, u[2]!) - tri(w[0]!, w[1]!, w[2]!)) / (u[q]! - w[q]!); });
                  const dd = 4 * tri(t[0]!, t[1]!, t[2]!) / Math.max(Math.hypot(g[0]!, g[1]!, g[2]!), 4e-3 * h);
                  worstD = Math.max(worstD, Math.abs(dd)); along += dd; crossings++;
                }
              }
              // Columns (vertical vertex lines of the tile, 5x5, y over the tile and two vertices either side) with
              // three or more crossings: a blob above or a pocket below the first surface. Their sign: a liquid blob in air or an air pocket.
              let multi = 0, blobs = 0, pockets = 0;
              for (let a = 0; a <= 4; a++) for (let d = 0; d <= 4; d++) {
                const signs: boolean[] = []; for (let b = -2; b <= 6; b++) signs.push(F(a, b, d) < 0);
                let flips = 0; for (let q = 1; q < signs.length; q++) if (signs[q] !== signs[q - 1]) flips++;
                if (flips >= 3) { multi++; if (signs[0]) blobs++; else pockets++; }
              }
              sum.multi = (sum.multi ?? 0) + (multi > 0 ? 1 : 0); sum.blobs = (sum.blobs ?? 0) + blobs; sum.pockets = (sum.pockets ?? 0) + pockets;
              const vol = crossings ? 4 * Math.abs(along / crossings) : 0, cur = curves ? 4 * Math.abs(curve / curves) : 0;
              (sum.terms ??= []).push([worstD, vol, cur, 4 * corner, multi > 0 ? 1 : 0, curves ? 4 * Math.abs(kappa / curves) : 0]); sum.cw = (sum.cw ?? 0) + worstD; sum.cv = (sum.cv ?? 0) + vol; sum.cc = (sum.cc ?? 0) + cur; sum.ck = (sum.ck ?? 0) + 4 * corner; sum.cn = (sum.cn ?? 0) + 1;
            }
            if (heights.length >= 20) {
              // Normal equations for z = p + q i + r j.
              let s = 0, si = 0, sj = 0, sii = 0, sjj = 0, sij = 0, sz = 0, siz = 0, sjz = 0;
              for (const [i, j, y] of heights) { s++; si += i; sj += j; sii += i * i; sjj += j * j; sij += i * j; sz += y; siz += i * y; sjz += j * y; }
              const M = [[s, si, sj], [si, sii, sij], [sj, sij, sjj]], v = [sz, siz, sjz];
              const det = (m: number[][]) => m[0]![0]! * (m[1]![1]! * m[2]![2]! - m[1]![2]! * m[2]![1]!) - m[0]![1]! * (m[1]![0]! * m[2]![2]! - m[1]![2]! * m[2]![0]!) + m[0]![2]! * (m[1]![0]! * m[2]![1]! - m[1]![1]! * m[2]![0]!);
              const D = det(M), c = [0, 1, 2].map(k => det(M.map((row, r) => row.map((e, q) => q === k ? v[r]! : e))) / D);
              let rms = 0, worst = 0;
              for (const [i, j, y] of heights) { const e = y - (c[0]! + c[1]! * i + c[2]! * j); rms += e * e; worst = Math.max(worst, Math.abs(e)); }
              sum.rough = (sum.rough ?? 0) + Math.sqrt(rms / heights.length); sum.worst = (sum.worst ?? 0) + worst; sum.fit = (sum.fit ?? 0) + 1;
              sum.slope = (sum.slope ?? 0) + Math.hypot(c[1]!, c[2]!);
              // The census's view: the tile's 4h corners (the field at its corner vertices) interpolated
              // trilinearly, its zero in each column against the h crossing; mean signed and worst, in h.
              const corner = (a: number, b: number, d: number) => field(4 * tx + 4 * a, 4 * ty + 4 * b, 4 * tz + 4 * d);
              const C = [0, 1].map(a => [0, 1].map(b => [0, 1].map(d => corner(a, b, d))));
              let dn = 0, dm = 0, dw = 0, pw = 0;
              for (const [i, j, y] of heights) {
                const u = i / 4, w = j / 4, bl = (b: number) => (1 - u) * (1 - w) * C[0]![b]![0]! + u * (1 - w) * C[1]![b]![0]! + (1 - u) * w * C[0]![b]![1]! + u * w * C[1]![b]![1]!;
                const A = bl(0), B = bl(1); if ((A < 0) === (B < 0) || y < 4 * ty || y > 4 * ty + 4) continue;
                const tri = 4 * ty + 4 * A / (A - B), d = tri - y, pd = tri - (c[0]! + c[1]! * i + c[2]! * j);
                dn++; dm += d; dw = Math.max(dw, Math.abs(d)); pw = Math.max(pw, Math.abs(pd));
              }
              if (dn > 0) { sum.tn = (sum.tn ?? 0) + 1; sum.tmean = (sum.tmean ?? 0) + Math.abs(dm / dn); sum.tworst = (sum.tworst ?? 0) + dw; sum.tplane = (sum.tplane ?? 0) + pw; }
            }
          }
          for (const [cls, { n, at, rough, worst, fit, slope, tn, tmean, tworst, tplane, cw, cv, cc, ck, cn, multi, blobs, pockets, terms }] of Object.entries(sums)) { if (terms?.length) { const med = (k: number) => terms.map(t => t[k]!).sort((a, b) => a - b)[terms.length >> 1]!.toFixed(2), over = (k: number, single?: number) => (100 * terms.filter(t => t[k]! > 1 && (single === undefined || t[4] === single)).length / terms.length).toFixed(0); console.log(`terms ${cls.padEnd(12)} median worst ${med(0)} vol ${med(1)} curve ${med(2)} corner ${med(3)} | >1h worst ${over(0)}% vol ${over(1)}% curve ${over(2)}% corner ${over(3)}% | >1h among single-surface: worst ${over(0, 0)}% vol ${over(1, 0)}% curve ${over(2, 0)}% | kappa median ${med(5)} >1h ${over(5)}% >0.5h ${(100 * terms.filter(t => t[5]! > 0.5).length / terms.length).toFixed(0)}% | any>0.5h ${(100 * terms.filter(t => Math.max(t[0]!, t[1]!, t[2]!, t[3]!) > 0.5).length / terms.length).toFixed(0)}% any>1h ${(100 * terms.filter(t => Math.max(t[0]!, t[1]!, t[2]!, t[3]!) > 1).length / terms.length).toFixed(0)}% any(kappa)>1h ${(100 * terms.filter(t => Math.max(t[0]!, t[1]!, t[5]!, t[3]!) > 1).length / terms.length).toFixed(0)}%`); } console.log(`phi ${cls.padEnd(12)} n ${String(n).padStart(5)} k=-${K}..${K + 1}: ${at.map(v => (v / n).toFixed(2)).join(" ")}  plane rms ${((rough ?? 0) / (fit || 1)).toFixed(3)}h worst ${((worst ?? 0) / (fit || 1)).toFixed(3)}h slope ${((slope ?? 0) / (fit || 1)).toFixed(2)} | tri |mean| ${((tmean ?? 0) / (tn || 1)).toFixed(3)}h worst ${((tworst ?? 0) / (tn || 1)).toFixed(3)}h vs plane ${((tplane ?? 0) / (tn || 1)).toFixed(3)}h | census worst ${((cw ?? 0) / (cn || 1)).toFixed(2)} vol ${((cv ?? 0) / (cn || 1)).toFixed(2)} curve ${((cc ?? 0) / (cn || 1)).toFixed(2)} corner ${((ck ?? 0) / (cn || 1)).toFixed(2)} | multi-crossing tiles ${(100 * (multi ?? 0) / (cn || 1)).toFixed(0)}% blob cols ${blobs ?? 0} pocket cols ${pockets ?? 0}`); }
          // --zb: the particle surfaces of the same tiles, from the live particles alone (cell units): the union
          // of 0.875-cell spheres (what reconstruct pins to) against Zhu & Bridson 2005's averaged surface
          // |x - xbar| - r, xbar the (1 - s^2)^3-weighted mean position within R. Plane-fit rms of each column's first
          // crossing (0.125-cell samples) for up to 300 tiles a class.
          if (process.argv.includes("--zb")) {
            // The sort's position copy (nbPosition): links (the next buffer) from word NB_LINKS = binCells, four words a particle.
            const fl = flip as unknown as { next: GPUBuffer; binCells: number };
            const live = flip!.count, stride = 4;
            const pread = device.createBuffer({ size: Math.max(16, 16 * live), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
            { const e = device.createCommandEncoder(); e.copyBufferToBuffer(fl.next, 4 * fl.binCells, pread, 0, Math.max(16, 16 * live)); device.queue.submit([e.finish()]); }
            await pread.mapAsync(GPUMapMode.READ); const P = new Float32Array(pread.getMappedRange().slice(0)); pread.unmap(); pread.destroy();
            // The window's offset (box cell 0 sits at lattice cell 32 * a frame page): the multiple of 32 an axis
            // that holds every particle in the box and puts the most of them in crossing tiles.
            const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], dims = [nx, ny, nz];
            for (let q = 0; q < live; q++) { P[stride * q + 3] = P[stride * q]! < 0 ? 0 : 1; if (!P[stride * q + 3]) continue; for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a]!, P[stride * q + a]!); hi[a] = Math.max(hi[a]!, P[stride * q + a]!); } }
            const choices = [0, 1, 2].map(a => { const c: number[] = []; for (let o = 32 * Math.floor((lo[a]! - dims[a]!) / 32); o <= hi[a]!; o += 32) c.push(o); return c; });
            let best = [0, 0, 0], bestCount = -1;
            for (const ox of choices[0]!) for (const oy of choices[1]!) for (const oz of choices[2]!) {
              let count = 0;
              for (let q = 0; q < live; q += 37) { if (!P[stride * q + 3]) continue; const t = uniformTileKey(address, Math.floor((P[stride * q]! - ox) / 4), Math.floor((P[stride * q + 1]! - oy) / 4), Math.floor((P[stride * q + 2]! - oz) / 4)); if (t >= 0 && t < f.tiles && (scores[2 * t + 1]! & (1 << 28))) count++; }
              if (count > bestCount) { bestCount = count; best = [ox, oy, oz]; }
            }
            console.log(`zb offset ${best.join(",")} (${bestCount} sampled particles in crossing tiles; choices ${choices.map(c => c.length).join("x")})`);
            for (let q = 0; q < live; q++) for (let a = 0; a < 3; a++) P[stride * q + a]! -= best[a]!;
            const grid = new Map<number, number[]>(), key = (x: number, y: number, z: number) => x + 4096 * (y + 4096 * z);
            for (let q = 0; q < live; q++) { if (!(P[stride * q + 3]! > 0)) continue; const k = key(Math.floor(P[stride * q]!), Math.floor(P[stride * q + 1]!), Math.floor(P[stride * q + 2]!)); let list = grid.get(k); if (!list) grid.set(k, list = []); list.push(q); }
            const radii = [1.5, 2], out: Record<string, { n: number; rough: number[] }> = {};
            const taken: Record<string, number> = {};
            for (let tz = 0; 4 * tz < nz; tz++) for (let ty = 0; 4 * ty < ny; ty++) for (let tx = 0; 4 * tx < nx; tx++) {
              const t = uniformTileKey(address, tx, ty, tz);
              if (!(t >= 0 && t < f.tiles) || !(scores[2 * t + 1]! & (1 << 28)) || fields.widthAt(4 * tx, 4 * ty, 4 * tz) !== 1) continue;
              const cls = heat[t]! > 0 ? "core" : target[t]! > 0 ? "collar" : null; if (!cls || (taken[cls] ?? 0) >= 300) continue;
              const per: [number, number, number][][] = [[], [], []];
              for (let i = 0; i <= 4; i++) for (let j = 0; j <= 4; j++) {
                const cx = 4 * tx + i, cz = 4 * tz + j, near: number[] = [];
                for (let a = cx - 3; a <= cx + 2; a++) for (let b = 4 * ty - 5; b <= 4 * ty + 8; b++) for (let d = cz - 3; d <= cz + 2; d++) { const list = grid.get(key(a, b, d)); if (list) near.push(...list); }
                const evals = (y: number) => {
                  let u = 3; const sw = [0, 0], sx = [[0, 0, 0], [0, 0, 0]];
                  for (const q of near) {
                    const dx = cx - P[stride * q]!, dy = y - P[stride * q + 1]!, dz = cz - P[stride * q + 2]!, d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 < 9) u = Math.min(u, Math.sqrt(d2) - 0.875);
                    for (let r = 0; r < 2; r++) { const s2 = d2 / (radii[r]! * radii[r]!); if (s2 < 1) { const w = (1 - s2) ** 3; sw[r]! += w; sx[r]![0]! += w * P[stride * q]!; sx[r]![1]! += w * P[stride * q + 1]!; sx[r]![2]! += w * P[stride * q + 2]!; } }
                  }
                  return [u, ...[0, 1].map(r => sw[r]! > 0 ? Math.hypot(cx - sx[r]![0]! / sw[r]!, y - sx[r]![1]! / sw[r]!, cz - sx[r]![2]! / sw[r]!) - 0.5 * radii[r]! : 3)];
                };
                let prev = evals(4 * ty - 2);
                const found = [false, false, false];
                for (let y = 4 * ty - 2 + 0.125; y <= 4 * ty + 6; y += 0.125) {
                  const now = evals(y);
                  for (let m = 0; m < 3; m++) if (!found[m] && prev[m]! < 0 && now[m]! >= 0) { found[m] = true; per[m]!.push([i, j, y - 0.125 + 0.125 * prev[m]! / (prev[m]! - now[m]!)]); }
                  prev = now; if (found.every(Boolean)) break;
                }
              }
              if (per.some(h => h.length < 20)) continue;
              taken[cls] = (taken[cls] ?? 0) + 1;
              const names = ["union", "zb1.5", "zb2"];
              for (let m = 0; m < 3; m++) {
                const hs = per[m]!; let s = 0, si = 0, sj = 0, sii = 0, sjj = 0, sij = 0, sz = 0, siz = 0, sjz = 0;
                for (const [a, b, y] of hs) { s++; si += a; sj += b; sii += a * a; sjj += b * b; sij += a * b; sz += y; siz += a * y; sjz += b * y; }
                const M = [[s, si, sj], [si, sii, sij], [sj, sij, sjj]], v = [sz, siz, sjz];
                const det = (q: number[][]) => q[0]![0]! * (q[1]![1]! * q[2]![2]! - q[1]![2]! * q[2]![1]!) - q[0]![1]! * (q[1]![0]! * q[2]![2]! - q[1]![2]! * q[2]![0]!) + q[0]![2]! * (q[1]![0]! * q[2]![1]! - q[1]![1]! * q[2]![0]!);
                const D = det(M), c = [0, 1, 2].map(k => det(M.map((row, r) => row.map((e, qq) => qq === k ? v[r]! : e))) / D);
                let rms = 0; for (const [a, b, y] of hs) rms += (y - (c[0]! + c[1]! * a + c[2]! * b)) ** 2;
                (out[`${cls} ${names[m]}`] ??= { n: 0, rough: [] }).rough.push(Math.sqrt(rms / hs.length));
              }
            }
            for (const [name, { rough }] of Object.entries(out)) { const sorted = [...rough].sort((a, b) => a - b); console.log(`zb ${name.padEnd(14)} tiles ${rough.length} rms mean ${(rough.reduce((a, b) => a + b, 0) / rough.length).toFixed(3)}h median ${sorted[sorted.length >> 1]!.toFixed(3)}h p90 ${sorted[Math.floor(0.9 * sorted.length)]!.toFixed(3)}h`); }
          }
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
        criteria, particleCensus, particles: flip ? { live: flip.count, peak: particlePeak, capacity: flip.capacity, clipped: flip.reseedClippedTotal, bricks, brickPeak, binSlots: flip.binSlots, arenaPeak, arenaWindows: (flip as unknown as { arenaWindows: number }).arenaWindows } : undefined };
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

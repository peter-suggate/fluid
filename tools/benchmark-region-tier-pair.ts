/** Alternate two production launch/setup policies on one GPU.
 * node --import tsx tools/benchmark-region-tier-pair.ts --scene=warm [--length=128]
 *   [--steps=120] [--from=40] [--out=PATH.json]
 * --batched-setup compares batched levels against the current held-input
 * baseline; add --retained-setup to include regional reuse in the candidate.
 * Both arms use identical production shaders and physics settings. */
import { profileRegionTierCommands } from "./region-tier-command-profile";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { cpus } from "node:os";
import { createUniformSolver, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { readMixedBuffer, readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import type { UniformMixedPressureCycles } from "../lib/methods/uniform/uniform-mixed-pressure-cycles";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { NARROW_BAND_PARTICLE_GRID, NARROW_BAND_QUIET_PARTICLE_GRID, UniformNarrowBandOrder } from "../lib/methods/uniform/uniform-narrow-band-order";
import { UNIFORM_PAGE_TABLE_WORDS } from "../lib/methods/uniform/uniform-lattice-address";
import { createRegionTierScene, REGION_TIER_SCENES, type RegionTierScene } from "./region-tier-scenes";
const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const mode = (option("scene") ?? "warm") as RegionTierScene, length = Number(option("length") ?? 128);
assert.ok(REGION_TIER_SCENES.includes(mode));
const steps = Number(option("steps") ?? 120), from = Number(option("from") ?? 40);
const orderOffset = process.argv.includes("--reverse-order") ? 1 : 0;
const timings = process.argv.includes("--timings");
const frameSchedule = process.argv.includes("--frame-schedule");
const regionalPipeline = process.argv.includes("--regional-pipeline");
const crossingPlan = frameSchedule || regionalPipeline || process.argv.includes("--crossing-plan");
const regionalGeometry = process.argv.includes("--regional-geometry");
const regionalWork = process.argv.includes("--regional-work");
const batchedSetup = process.argv.includes("--batched-setup");
const baselineBatched = process.argv.includes("--baseline-batched");
assert.ok(!baselineBatched || batchedSetup, "--baseline-batched requires --batched-setup");
const regionalSetup = process.argv.includes("--retained-setup");
const retainedSetup = batchedSetup || regionalSetup || process.argv.includes("--held-setup");
const stressSmallGrid = process.argv.includes("--stress-small-grid");
const candidateHotGrid = option("candidate-hot-grid") === undefined ? undefined : Number(option("candidate-hot-grid"));
assert.ok(candidateHotGrid === undefined || (Number.isInteger(candidateHotGrid) && candidateHotGrid > 0 && candidateHotGrid <= NARROW_BAND_PARTICLE_GRID));
assert.ok(Number.isInteger(steps) && Number.isInteger(from) && from > 0 && from <= steps);
const depth = Number(option("depth") ?? 32);
const scene = createRegionTierScene(mode, length, depth);
if(option("surface-tension")!==undefined)scene.fluid.surfaceTension_N_m=Number(option("surface-tension"));
const poolBlocks = option("pool-blocks") === undefined ? undefined : Number(option("pool-blocks"));
if (poolBlocks !== undefined) { assert.ok(Number.isInteger(poolBlocks) && poolBlocks >= 3); scene.fluid.pagePool = { blocks: poolBlocks }; }
const splashCount = Number(option("splash-count") ?? 1);
assert.ok(Number.isInteger(splashCount) && splashCount >= 1);
if (mode === "splash") {
  assert.ok(.6 + .8 * (splashCount - 1) + .1 < scene.container.width_m, "splashes must fit inside the pool");
  const drop = scene.fluid.scheduledDrops![0]!;
  assert.equal(drop.volume.shape, "sphere");
  // The source contract permits one drop per step. Keep successive impacts
  // two steps apart so floating-point time boundaries cannot merge them.
  scene.fluid.scheduledDrops = Array.from({ length: splashCount }, (_, i) => ({ ...structuredClone(drop), time_s: drop.time_s + 2 * i * scene.numerics.fixedDt_s,
    volume: { shape: "sphere" as const, radius_m: .1, center_m: { x: -scene.container.width_m / 2 + .6 + .8 * i, y: .6, z: 0 } } }));
}
scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
const suffix = `${depth === 32 ? "" : `-depth${depth}`}${baselineBatched ? "-baseline-batched" : ""}${orderOffset ? "-reverse" : ""}${splashCount > 1 ? `-drops${splashCount}` : ""}${poolBlocks === undefined ? "" : `-pool${poolBlocks}`}`;
const out = resolve(option("out") ?? `artifacts/region-tiers/${batchedSetup ? "batched-" : ""}${regionalSetup ? "retained-" : retainedSetup ? "held-" : ""}paired-${mode}-${length}${suffix}.json`);
process.env.WEBGPU_NODE_MODULE ??= resolve("node_modules/webgpu/index.js");
paceDawnEventPolling();
const hashSource = () => {
  const hash = createHash("sha256");
  for (const f of readdirSync("lib/methods/uniform").filter(f => f.endsWith(".ts")).sort()) hash.update(f).update(readFileSync(`lib/methods/uniform/${f}`));
  return hash.digest("hex");
};
const sourceHash = hashSource();
await withUniformDevice("paired region tiers", async device => {
  const profiler=process.argv.includes("--profile")||timings?profileRegionTierCommands(device,timings):undefined;
  const commandRows:unknown[]=[];
  const solvers: Awaited<ReturnType<typeof createUniformSolver>>[] = [];
  const rows: { step: number; first: number; wall_ms: number[]; encode_ms: number[]; state: unknown[]; setupWork: number[][] }[] = [];
  const read = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    for (let arm = 0; arm < 2; arm++) {
      UniformNarrowBandOrder.useLegacyScanForDevice(device,regionalPipeline&&arm===0);
      const start = performance.now(), solver = await createUniformSolver(device, structuredClone(scene)); solvers.push(solver);
      UniformNarrowBandOrder.useLegacyScanForDevice(device,false);
      const frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
      if (regionalPipeline) {
        frame.narrowBandFlip.regionalCrossingPlanEnabled = arm === 1;
        const order = Reflect.get(frame.narrowBandFlip, "order");
        const cells = Reflect.get(order, "cells") as number;
        assert.equal(order.sparse, arm === 1 || cells >= 4_194_304, "regional pipeline arm must use its requested ordering policy");
      }
      if (frameSchedule) { frame.surfaceWorkControl.framePlan.scheduleStages = arm === 1; frame.surfaceWorkControl.framePlan.fuseAirClear = !process.argv.includes("--schedule-only"); frame.surfaceWorkControl.framePlan.coalescePasses = !process.argv.includes("--air-only"); }
      else if (crossingPlan) { frame.narrowBandFlip.regionalCrossingPlanEnabled = arm === 1; }
      else if (regionalGeometry) { frame.surfaceWorkControl.regionalGeometrySetup = arm === 1; frame.surfaceWorkControl.framePlan.retainGeometryCache = arm === 1; }
      else if (regionalWork) { frame.surfaceWorkControl.framePlan.enabled = arm === 1; }
      else if (retainedSetup) {
        const cycles = (frame as unknown as { cycles: UniformMixedPressureCycles }).cycles;
        cycles.batchSetup = batchedSetup && (arm === 1 || baselineBatched);
        cycles.retainSetup = batchedSetup || arm === 1;
        cycles.retainRegionalSetup = arm === 1 && regionalSetup;
      } else if (arm === 0) {
        const band = (frame as unknown as { band: object }).band;
        Object.defineProperty(band, "empty", { get: () => frame.ownership.capacity.fineTiles === 0 ||
          (Reflect.get(band, "cold") !== undefined && frame.cold.quiet) });
        const particles = frame.narrowBandFlip, begin = particles.beginAdvance.bind(particles);
        particles.beginAdvance = cold => {
          begin(cold);
          Reflect.set(particles, "particleGroups", NARROW_BAND_PARTICLE_GRID);
          Reflect.get(particles, "order").workgroups = NARROW_BAND_PARTICLE_GRID;
        };
      } else if (stressSmallGrid) {
        // Correctness stress: force the small grid even through a HOT impact.
        // Production never does this; it tests that the launches really stride.
        const particles = frame.narrowBandFlip, begin = particles.beginAdvance.bind(particles);
        particles.beginAdvance = cold => {
          begin(cold);
          Reflect.set(particles, "particleGroups", NARROW_BAND_QUIET_PARTICLE_GRID);
          Reflect.get(particles, "order").workgroups = NARROW_BAND_QUIET_PARTICLE_GRID;
        };
      } else if (candidateHotGrid !== undefined) {
        const particles = frame.narrowBandFlip, begin = particles.beginAdvance.bind(particles);
        particles.beginAdvance = cold => {
          begin(cold);
          if (cold.ceiling === 2 && particles.count <= 65536) {
            Reflect.set(particles, "particleGroups", candidateHotGrid);
            Reflect.get(particles, "order").workgroups = candidateHotGrid;
          }
        };
      }
      console.log(JSON.stringify({ arm, scene: scene.sceneId, construction_ms: performance.now() - start }));
    }
    for (let step = 1; step <= steps; step++) {
      const wall_ms = [0, 0], encode_ms = [0, 0], state: unknown[] = [], setupWork: number[][] = [];
      const first = (step + orderOffset) % 2 ? 0 : 1;
      for (const arm of first === 0 ? [0, 1] : [1, 0]) {
        const solver = solvers[arm]!, frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
        profiler?.begin();
        const start = performance.now();
        assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s, [])); encode_ms[arm]=performance.now()-start; await solver.awaitFrameCompletion();
        wall_ms[arm] = performance.now() - start;
        const commands=profiler?.end();if(commands){commandRows.push({step,arm,commands,timings:timings?await profiler!.readTimings():undefined});assert.ok(Object.values(commands).every(c=>c.indirectDispatches===0),"frame used an indirect dispatch");}
        assert.equal(solver.info.simulationPipelineError, undefined);
        assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - step * scene.numerics.fixedDt_s) < 1e-8);
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(frame.heat.tiers, 4 * frame.heat.regions, read, 0, 32);
        device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
        const tiers = [...new Uint32Array(read.getMappedRange()).slice(0, 5)]; read.unmap();
        if (retainedSetup) {
          const work = (frame as unknown as { cycles: UniformMixedPressureCycles }).cycles.setupWork;
          const e = device.createCommandEncoder(); e.copyBufferToBuffer(work.buffer, work.offset ?? 0, read, 0, 12);
          device.queue.submit([e.finish()]); await read.mapAsync(GPUMapMode.READ);
          setupWork[arm] = [...new Uint32Array(read.getMappedRange()).slice(0, 3)]; read.unmap();
        }
        if(crossingPlan){
          const work=frame.narrowBandFlip.crossingPlanEvidence;
          const e=device.createCommandEncoder();e.copyBufferToBuffer(work.buffer,work.offset,read,0,16);
          device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
          setupWork[arm]=[...new Uint32Array(read.getMappedRange()).slice(0,4),work.regions];read.unmap();
        }
        state[arm] = { tiers, ceiling: frame.cold.ceiling, particles: frame.narrowBandFlip.count,
          fine: solver.info.uniformMixedFineTiles, band: solver.info.uniformPressureBandTiles, residual: solver.info.uniformPressureBandResidual,
          rootResidual: solver.info.uniformPressureAcceptedResidual, rootCycles: solver.info.uniformPressureCyclesExecuted };
      }
      rows.push({ step, first, wall_ms, encode_ms, state, setupWork });
      if (step % 20 === 0 || step === steps) console.log(JSON.stringify(rows.at(-1)));
    }
    const physics = [], fields: Record<string, string>[] = [], rootLayouts: { page: number; block: number; hash: string }[][] = [];
    for (const solver of solvers) {
      const stats = await solver.readStats();
      physics.push({ speed: stats.maxSpeed_m_s, volume: stats.volumeCellSum, drift: stats.volumeDrift, residual: stats.uniformPressureAcceptedResidual });
      const hashes: Record<string, string> = {};
      for (const [name, texture] of Object.entries({ velocity: solver.velocityTexture, volume: solver.volumeTexture,
        phi: solver.vertexPhiTexture!, pressure: solver.gridPressureTexture })) {
        const data = await readMixedTexture(device, texture);
        hashes[name] = createHash("sha256").update(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)).digest("hex");
      }
      // The display pressure texture can be a placeholder. Check the actual
      // pressure-root binding as well, outside the timed simulation window.
      const frame = (solver as unknown as { mixedFrame: { cycles: UniformMixedPressureCycles } }).mixedFrame;
      const root = frame.cycles.level.pressure, rootData = await readMixedBuffer(device, root.buffer);
      const ownership = frame.cycles.level.ownership, blocks = ownership.layout.address.blocks;
      const topology = await readMixedBuffer(device, ownership.presentation.buffer);
      const topologyWords = new Uint32Array(topology.buffer, ownership.presentation.offset ?? 0);
      const records = 512 * blocks * 5 + UNIFORM_PAGE_TABLE_WORDS;
      // Newly claimed pages may take different physical blocks. Compare the
      // pressure by page identity; unused pool slots are not simulation state.
      const layout = Array.from({ length: blocks - 2 }, (_, i) => i + 1).filter(b => topologyWords[records + b] !== 0xffffffff).map(block => {
        const hash = createHash("sha256"), base = root.offset ?? 0;
        hash.update(new Uint8Array(rootData.buffer, base + 4 * 512 * block, 4 * 512));
        hash.update(new Uint8Array(rootData.buffer, base + 4 * (512 * blocks + 384 * block), 4 * 384));
        return { page: topologyWords[records + block]!, block, hash: hash.digest("hex") };
      }).sort((a, b) => a.page - b.page);
      rootLayouts.push(layout);
      hashes.rootPressure = createHash("sha256").update(JSON.stringify(layout.map(({ page, hash }) => ({ page, hash })))).digest("hex");
      fields.push(hashes);
    }
    const measured = rows.filter(r => r.step >= from), mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;
    const median = (v: number[]) => { const a = [...v].sort((x, y) => x - y), i = a.length >> 1; return a.length % 2 ? a[i]! : (a[i - 1]! + a[i]!) / 2; };
    const distribution = (values: number[]) => {
      const sorted = [...values].sort((a, b) => a - b);
      return { mean_ms: mean(values), median_ms: median(values), p95_ms: sorted[Math.floor(.95 * (sorted.length - 1))], max_ms: sorted.at(-1) };
    };
    const windows = Object.fromEntries([["startup", 1, 5], ["preActivation", 10, 27], ["activation", 28, 39], ["later", 40, steps]].map(([name, start, end]) => {
      const selected = rows.filter(r => r.step >= Number(start) && r.step <= Number(end));
      return [name, selected.length ? { baseline: distribution(selected.map(r => r.wall_ms[0]!)), candidate: distribution(selected.map(r => r.wall_ms[1]!)),
        worstPairedPenalty_ms: Math.max(...selected.map(r => r.wall_ms[1]! - r.wall_ms[0]!)),
        setupWork: [0, 1].map(arm => [0, 1, 2].map(stage => mean(selected.map(r => r.setupWork[arm]?.[stage] ?? 0)))) } : null];
    }));
    const summary = { baseline_ms: mean(measured.map(r => r.wall_ms[0]!)), candidate_ms: mean(measured.map(r => r.wall_ms[1]!)),
      pairedMedianSaved_ms: median(measured.map(r => r.wall_ms[0]! - r.wall_ms[1]!)),
      baselineFirstSaved_ms: mean(measured.filter(r => r.first === 0).map(r => r.wall_ms[0]! - r.wall_ms[1]!)),
      candidateFirstSaved_ms: mean(measured.filter(r => r.first === 1).map(r => r.wall_ms[0]! - r.wall_ms[1]!)),
      identicalFields: JSON.stringify(fields[0]) === JSON.stringify(fields[1]),
      identicalWork: rows.every(r => JSON.stringify(r.state[0]) === JSON.stringify(r.state[1])), sourceUnchanged: sourceHash === hashSource() };
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, JSON.stringify({ date: new Date().toISOString(), cpu: cpus()[0]?.model, node: process.version, sourceHash,
      scope: "alternating fenced simulation steps on one device; diagnostics and construction excluded", scene, steps, from, stressSmallGrid, candidateHotGrid, timings, frameSchedule, scheduleOnly:process.argv.includes("--schedule-only"), airOnly:process.argv.includes("--air-only"), regionalPipeline, crossingPlan, regionalWork, regionalGeometry, retainedSetup, regionalSetup, batchedSetup, baselineBatched, orderOffset, summary, windows, physics, fields, rootLayouts, commandRows, rows }, null, 2) + "\n");
    console.log(JSON.stringify({ out, summary }));
    if (crossingPlan || retainedSetup || regionalWork || regionalGeometry) { assert.ok(summary.identicalFields, "retained setup changed final fields"); assert.ok(summary.identicalWork, "retained setup changed work/residual history"); }
    assert.ok(summary.sourceUnchanged, "sources changed during measurement; repeat before citing");
  } catch (error) {
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(`${out}.failed.json`, JSON.stringify({ sourceHash, scene, steps, from, stressSmallGrid, candidateHotGrid, retainedSetup, orderOffset, error: String(error), rows }, null, 2) + "\n");
    throw error;
  } finally { UniformNarrowBandOrder.useLegacyScanForDevice(device,false);profiler?.destroy();read.destroy(); for (const solver of solvers) solver.destroy(); }
},timings?["timestamp-query"]:[]);

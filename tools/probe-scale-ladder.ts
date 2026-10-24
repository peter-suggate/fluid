/** Scale ladder for reservoir-seep (tiered-adaptivity plan, W9): builds the
 * scene at a cell size (throat kept at h) with optional width/depth
 * multipliers, constructs the production NB-FLIP solver with an explicit page
 * pool, runs N fenced steps and reports the first failure, the bytes by
 * allocation class, the live counts and the step times. Exits 0 on a failure
 * too: the failure is the result. --budget-s stops stepping before the
 * hold's limit (counted from the hold, or from construction without --hold).
 * env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-scale-ladder.ts
 *   --cell=0.0025 [--width=2] [--depth=0.5] [--pool=<blocks>] [--steps=10] [--budget-s=150]
 *   [--hold=/private/tmp/claude-501/fluid-gpu-run.sh [--hold-mode=timing]] [--values={...}] [--out=artifacts/scale-ladder/X.json]
 * --hold builds the solver outside the shared GPU runner and takes it (check, or timing) for the
 * steps; without it, run the whole probe under the runner. --plan evaluates the host-side
 * limits (page window, residency, lattice address and block grid, block count, memory)
 * without a GPU.
 * --pool replaces the scene's page pool (fluid.pagePool.blocks); without it
 * the scene's own pool is used, and --max-gib (default 6, the shared runner's check-mode cap) refuses a pool
 * whose blocks would exceed it at --block-mib (default 16, measured at 5 mm) a block before
 * any GPU allocation. A row per step: fenced ms, h tiles, resident pages,
 * particles, regions COLD/WARM/HOT, the max speed in h cells a step, and
 * (every --every steps and the last) the catch chamber's h tiles: on its
 * floor (the bottom 8 cells) and in all. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues, type MethodParamValues } from "../lib/core/method-contract";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { sceneFluidLatticeDimensions, sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { takeGpuHold } from "./gpu-hold";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { UNIFORM_PAGE_TABLE_WORDS, uniformLatticeAddress, uniformLatticeWindowBox } from "../lib/methods/uniform/uniform-lattice-address";
import { uniformPageSources, type WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformLatticeInitialVolume } from "../lib/methods/uniform/uniform-volume-initial";
import { UNIFORM_PAGE_POOL_STATUS as S, uniformPagePoolFatalMessage, uniformPagePoolResidency, type UniformPagePool } from "../lib/methods/uniform/uniform-page-pool";
import { auditUniformGPUAllocations } from "./uniform-gpu-allocation-audit";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const cell = Number(option("cell") ?? 0.005), widthScale = Number(option("width") ?? 1), depthScale = Number(option("depth") ?? 1);
const holdFile = option("hold"), holdMode = option("hold-mode") === "timing" ? "timing" as const : "check" as const;
const steps = Number(option("steps") ?? 10), budget_s = Number(option("budget-s") ?? 150), every = Number(option("every") ?? 5);
const maxGiB = Number(option("max-gib") ?? 6), blockMiB = Number(option("block-mib") ?? 16);
const pool = option("pool") === undefined ? undefined : Number(option("pool"));
const extra: MethodParamValues = JSON.parse(option("values") ?? "{}");
assert.ok(cell > 0 && widthScale > 0 && depthScale > 0 && Number.isInteger(steps) && steps >= 0);
const tag = `seep-${cell * 1000}mm${widthScale === 1 ? "" : `-w${widthScale}`}${depthScale === 1 ? "" : `-d${depthScale}`}`;
const out = resolve(option("out") ?? `artifacts/scale-ladder/${tag}.json`);

const base = getSceneDefinition(RESERVOIR_SEEP_ID);
const definition = { ...base, buildAt: (l: { cellSize_m: number }) => createReservoirSeepScene(l.cellSize_m, { widthScale, depthScale }) };
const scene = structuredClone(sceneDocumentAtLattice(definition, { cellSize_m: cell }).scene);
scene.duration_s = Math.max(scene.duration_s, (steps + 1) * scene.numerics.fixedDt_s);
if (pool !== undefined) scene.fluid.pagePool = { ...scene.fluid.pagePool, blocks: pool };
const quality = base.methodProfile!.quality;
const values = resolveMethodValues(method, quality, { ...method.appDefaults, timeStep: "scene", ...base.methodProfile!.overrides, ...extra });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
const lattice = sceneFluidLatticeDimensions(scene, Number.MAX_SAFE_INTEGER), wallHigh = 16 * Math.floor(sceneLatticeDimensions(scene)[0] / 32);
const blocks = scene.fluid.pagePool?.blocks ?? 0;
const header = { tag, cell_m: cell, widthScale, depthScale, lattice, boxCells: lattice[0] * lattice[1] * lattice[2],
  boxPages: lattice.map(n => Math.ceil(n / 32)), poolBlocks: blocks, dt, values: extra };
console.log(JSON.stringify(header));

interface Failure { phase: string; message: string; where?: string; numbers: number[]; stack?: string }
const failureOf = (phase: string, error: unknown): Failure => {
  const e = error as Error, cause = e?.cause instanceof Error ? `: ${e.cause.message}` : "";
  const message = `${e?.message ?? String(error)}${cause}`, stack = e?.stack ?? "";
  const where = /(?:lib|tools)\/[\w/.-]+\.ts:\d+/.exec(stack)?.[0];
  return { phase, message, where, numbers: [...message.matchAll(/\d+(?:\.\d+)?(?:e[+-]?\d+)?/g)].map(m => Number(m[0])), stack: stack.split("\n").slice(0, 8).join("\n") };
};
// Allocation classes for the byte report, by resource label.
const classes: [string, RegExp][] = [
  ["particles", /particle|FLIP (cell starts|bins|next|slab)/i],
  ["pressure", /pressure|arena|multigrid|brick/i],
  ["h detail fields", /velocity|volume|surface|gamma|phi|detail|unit tap/i],
  ["ownership/layout", /owner|layout|tile|hanging|seam|census|mixed/i],
  ["page pool", /page|pool|block/i],
  ["scratch", /scratch|stage|conditioning/i],
];
const classify = (label: string) => classes.find(([, r]) => r.test(label))?.[0] ?? "other";

const rows: Record<string, unknown>[] = [];
let failure: Failure | undefined, construction_ms = NaN, allocation: ReturnType<ReturnType<typeof auditUniformGPUAllocations>["snapshot"]> | undefined;
const gpuErrors: string[] = [];
const report = () => {
  const byClass: Record<string, number> = {};
  for (const r of allocation?.resources ?? []) byClass[classify(r.label)] = (byClass[classify(r.label)] ?? 0) + r.bytes;
  const mib = (n: number) => Math.round(n / 1048576 * 10) / 10;
  const ms = rows.map(r => r.wall_ms as number).filter(n => n > 0), steady = ms.slice(Math.min(3, ms.length - 1)).sort((a, b) => a - b);
  const summary = { ...header, construction_ms: Math.round(construction_ms), stepsRun: rows.length, failure: failure && { ...failure, stack: undefined },
    gpuErrors: gpuErrors.slice(0, 5),
    liveMiB: allocation && mib(allocation.liveBytes), peakMiB: allocation && mib(allocation.peakBytes),
    perBlockMiB: allocation && blocks ? mib(allocation.liveBytes / blocks) : undefined,
    byClassMiB: Object.fromEntries(Object.entries(byClass).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, mib(v)])),
    topResourcesMiB: (allocation?.resources ?? []).slice(0, 16).map(r => [r.label, r.count, mib(r.bytes)]),
    medianStep_ms: steady.length ? Math.round(steady[Math.floor(steady.length / 2)]! * 100) / 100 : undefined,
    last: rows.at(-1) };
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify({ date: new Date().toISOString(), ...summary, rows }, null, 1) + "\n");
  console.log(JSON.stringify({ ...summary, last: undefined, out }));
  if (failure) console.log(`FIRST FAILURE (${failure.phase}) at ${failure.where ?? "?"}: ${failure.message}\n${failure.stack}`);
};

// --plan: no GPU. The host-side checks the constructor makes, each evaluated
// on its own so one run lists the whole static chain, with the seed's
// resident pages from the production residency pass (--no-seed skips it).
if (process.argv.includes("--plan")) {
  const limits = { maxBufferSize: 4294967296, maxStorageBufferBindingSize: 4294967292, maxTextureDimension3D: 2048,
    ...JSON.parse(option("limits") ?? "{}") as Record<string, number> };
  const checks: { check: string; owner: string; ok: boolean; detail: string }[] = [];
  const check = (name: string, owner: string, run: () => string) => {
    try { checks.push({ check: name, owner, ok: true, detail: run() }); }
    catch (error) { checks.push({ check: name, owner, ok: false, detail: (error as Error).message }); }
  };
  check("page window, per axis", "W7", () => { uniformLatticeWindowBox(lattice); return `${header.boxPages.join("x")} pages`; });
  let air: [number, number, number][] | undefined, resident = NaN;
  if (!process.argv.includes("--no-seed")) check("seed residency (host pass over every cell)", "W1", () => {
    const t = performance.now(), residency = uniformPagePoolResidency(lattice, scene.fluid.pagePool,
      uniformPageSources(scene, undefined, dt), values.detailPolicy === "dynamic", () => uniformLatticeInitialVolume(scene).volume, [0, 0, 0]);
    air = residency.air.map(p => [...p] as [number, number, number]);
    resident = header.boxPages.reduce((a, b) => a * b, 1) - residency.air.length;
    if (resident > blocks - 2) throw new Error(`seed holds ${resident} resident pages; the pool stores ${blocks - 2}`);
    return `${resident} resident pages of ${blocks - 2} slots, ${Math.round(performance.now() - t)} ms host`;
  });
  check("lattice address: window y, block grid (<= 3968 blocks)", "W7", () => {
    const a = uniformLatticeAddress(lattice, scene.container.top === "open", { ...scene.fluid.pagePool, air: air ?? [] });
    return `${a.blocks} blocks, block grid ${a.blockGrid.join("x")}`;
  });
  // The address checks the block grid too, but after the window; this one stands alone. The box-sized
  // device checks are gone (W1, 8bf2039e) and the pressure tail takes super-blocks past 4096 pages (W8, 51f3b614).
  check("block grid: 33 vertex texels a block in a 2048 texture (<= 3968 blocks)", "W6/W7", () => {
    const bits = Math.min(Math.floor(blocks / 8), 1) + Math.min(Math.floor(blocks / 64), 1) + Math.min(Math.floor(blocks / 512), 1), z = Math.ceil(blocks / 4 ** bits);
    if (z > 62) throw new Error(`${blocks} blocks need a block grid z of ${z} > 62`); return `z ${z}`; });
  check("memory at today's block cost", "W6", () => { const gib = blocks * blockMiB / 1024;
    if (gib > 16) throw new Error(`${blocks} blocks x ${blockMiB} MiB = ${gib.toFixed(1)} GiB, past 16 GiB of a 32 GiB machine`); return `${gib.toFixed(1)} GiB`; });
  const plan = { ...header, plan: true, residentPages: resident, checks, blockers: checks.filter(c => !c.ok).map(c => `${c.owner}: ${c.check}`) };
  mkdirSync(dirname(out), { recursive: true }); writeFileSync(out.replace(/\.json$/, "-plan.json"), JSON.stringify(plan, null, 1) + "\n");
  for (const c of checks) console.log(`${c.ok ? "ok  " : "FAIL"} ${c.owner} ${c.check}: ${c.detail}`);
  process.exit(0);
}
if (blocks * blockMiB / 1024 > maxGiB) {
  failure = { phase: "budget", message: `pool of ${blocks} blocks is ~${(blocks * blockMiB / 1024).toFixed(1)} GiB at ${blockMiB} MiB a block, over --max-gib=${maxGiB}`, numbers: [blocks] };
  report(); process.exit(0);
}
paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter(); assert.ok(adapter);
const raw = await adapter.requestDevice({ requiredFeatures: (["float32-filterable", "subgroups"] as GPUFeatureName[]).filter(f => adapter.features.has(f)),
  requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const limits = Object.fromEntries((["maxBufferSize", "maxStorageBufferBindingSize", "maxTextureDimension2D", "maxTextureDimension3D"] as const)
  .map(k => [k, raw.limits[k]]));
console.log(JSON.stringify({ limits }));
Object.assign(header, { limits });
const audit = auditUniformGPUAllocations(raw);
const device = managedGPUDevice(audit.device, { requireWorkerRealm: false });
raw.addEventListener("uncapturederror", e => { e.preventDefault(); gpuErrors.push((e as GPUUncapturedErrorEvent).error.message); });
let solver: WebGPUUniformReferenceSolver | undefined;
let started = performance.now(), release: (() => void) | undefined;
try {
  const t0 = performance.now();
  try { solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver; }
  catch (error) { failure = failureOf("construction", error); }
  construction_ms = performance.now() - t0; allocation = audit.snapshot();
  if (gpuErrors.length && !failure) failure = { phase: "construction", message: `GPU validation: ${gpuErrors[0]}`, numbers: [] };
  if (solver && !failure && holdFile) {
    // Built outside the lock (shader compilation and host passes are CPU); the steps run inside it.
    console.log(JSON.stringify({ built_ms: Math.round(construction_ms), waiting: holdMode }));
    const t = performance.now(); release = await takeGpuHold(holdFile, holdMode); started = performance.now();
    console.log(JSON.stringify({ held: holdMode, waited_ms: Math.round(started - t) }));
  }
  if (solver && !failure) {
    const frame = (solver as unknown as { mixedFrame: { pagePool: UniformPagePool; heat: { tiers: GPUBuffer; regions: number }; setLayoutViews(on: boolean): void;
      ownership: { presentation: GPUBufferBinding; capacity: { tiles: number; fineTiles: number } } } }).mixedFrame;
    const l = frame.pagePool.layout, status = new Uint32Array(S.words);
    console.log(JSON.stringify({ constructed_ms: Math.round(construction_ms), liveMiB: Math.round(allocation.liveBytes / 1048576), poolSlots: l.slots,
      blocks: l.blocks, tileCapacity: frame.ownership.capacity.tiles, fineCapacity: frame.ownership.capacity.fineTiles }));
    const read = async (buffer: GPUBuffer, offset: number, size: number) => {
      const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, offset, staging, 0, size); device.queue.submit([e.finish()]);
      await staging.mapAsync(GPUMapMode.READ); const data = new Uint32Array(staging.getMappedRange().slice(0)); staging.destroy(); return data;
    };
    // h tiles of the catch chamber: tile words' fine bit over each block's page.
    const catchTiles = async () => {
      const n = frame.ownership.capacity.tiles, nb = n / 512, p = frame.ownership.presentation;
      const words = await read(p.buffer, p.offset ?? 0, 4 * (5 * n + UNIFORM_PAGE_TABLE_WORDS + 3 + nb));
      const recordBase = 5 * n + UNIFORM_PAGE_TABLE_WORDS, origin = [0, 1, 2].map(k => words[5 * n + 64 ** 3 + k]! | 0);
      let floor = 0, chamber = 0, all = 0;
      for (let b = 1; b < nb - 1; b++) {
        const r = words[recordBase + b]!; if (r === 0xffffffff) continue;
        const page = [r & 63, (r >>> 6) & 63, (r >>> 12) & 63].map((v, k) => ((v - origin[k]!) % 64 + 64) % 64);
        for (let t = 0; t < 512; t++) {
          if (!(words[512 * b + t]! & 0x80000000)) continue;
          all++;
          const tx = page[0]! * 8 + (t & 7), ty = page[1]! * 8 + ((t >> 3) & 7);
          if (4 * tx >= wallHigh) { chamber++; if (ty < 2) floor++; }
        }
      }
      return { catchFloorH: floor, catchChamberH: chamber, presentedH: all, origin };
    };
    for (let step = 1; step <= steps; step++) {
      if ((performance.now() - started) / 1000 > budget_s) { console.log(`budget ${budget_s} s reached after ${step - 1} steps`); break; }
      const census = step % every === 0 || step === steps;
      if (census) frame.setLayoutViews(true);
      const t = performance.now();
      try { assert.ok(solver.advanceTo(step * dt, []), "advance admitted"); await solver.awaitFrameCompletion(); }
      catch (error) { failure = failureOf(`step ${step}`, error); }
      const wall_ms = performance.now() - t;
      if (!failure && solver.info.simulationPipelineError) failure = { phase: `step ${step}`, message: String(solver.info.simulationPipelineError), numbers: [] };
      const receipt = device.createBuffer({ size: 4 * S.words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const e = device.createCommandEncoder(); frame.pagePool.encodeReceipt(e, receipt, 0); device.queue.submit([e.finish()]);
      await receipt.mapAsync(GPUMapMode.READ); status.set(new Uint32Array(receipt.getMappedRange())); receipt.destroy();
      const fatal = uniformPagePoolFatalMessage(l, status);
      if (!failure && fatal) failure = { phase: `step ${step}`, message: `page pool: ${fatal}`, numbers: [...fatal.matchAll(/\d+/g)].map(m => Number(m[0])) };
      if (!failure && gpuErrors.length) failure = { phase: `step ${step}`, message: `GPU validation: ${gpuErrors[0]}`, numbers: [] };
      const tiers = failure ? [] : [...(await read(frame.heat.tiers, 4 * frame.heat.regions, 16))].slice(0, 3);
      const stats = failure ? undefined : await solver.readStats();
      const row: Record<string, unknown> = { step, wall_ms: Math.round(wall_ms * 100) / 100, hTiles: solver.info.uniformMixedFineTiles,
        resident: status[S.resident], free: status[S.free], particles: solver.narrowBandFlipInfo?.particles,
        regionsColdWarmHot: tiers, maxSpeed_m_s: stats?.maxSpeed_m_s, maxSpeed_hPerStep: stats && Math.round((stats.maxSpeed_m_s ?? NaN) * dt / cell * 100) / 100,
        drift: stats?.volumeDrift };
      if (census && !failure) { Object.assign(row, await catchTiles()); frame.setLayoutViews(false); }
      rows.push(row); console.log(JSON.stringify(row));
      if (failure) break;
    }
    allocation = audit.snapshot();
  }
} catch (error) { failure ??= failureOf("probe", error); }
finally {
  report();
  try { solver?.destroy(); } catch { /* reported */ }
  await raw.queue.onSubmittedWorkDone(); raw.destroy(); release?.();
}

/**
 * Detection cost attribution (GPU, opt in with CLOTH_CONTACT_COST=sphere): the rotating-sphere sheet
 * through ClothSystem (its scene contact and Newton hooks, the moving sphere a rigid body unless
 * CLOTH_COST_HELD=1), per tick the wall time, Newton iterations, the rounds that ran (mandatory,
 * redos, re-detections, levelled and their fallbacks) and each round's broad phase (the round log:
 * scanned, collisions, box-rejected, candidates, DAT pairs). After the last tick, at its state:
 * each round kernel timed alone (a mandatory round armed by encodeStepBegin), the redo decision, an
 * unarmed mandatory and later detection call (their indirect dispatches with zero workgroups), the
 * truncation and the filter, so a step's detection is
 *   armed rounds x their kernels + every encoded dispatch's floor.
 *
 *   CLOTH_COST_NODES  nodes across the sheet (default 16)
 *   CLOTH_COST_TICKS  ticks (default 60)
 *   CLOTH_COST_EVERY  print every n-th tick (default 1)
 */
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { CLOTH_SCENE_STEP_S, ClothSystem } from "../lib/cloth/system";
import { OGC_ROUND_KERNELS, OgcContactDetector } from "../lib/cloth/contact/ogc-detector";
import { bridson2002RotatingSphere } from "../lib/cloth/scenes/bridson-2002";
import { bridson2002SceneDescription } from "../lib/cloth/scenes/bridson-2002/scene-description";

const want = !!process.env.WEBGPU_NODE_MODULE && process.env.CLOTH_CONTACT_COST === "sphere";
const NODES = Number(process.env.CLOTH_COST_NODES ?? 16), TICKS = Number(process.env.CLOTH_COST_TICKS ?? 60);
const EVERY = Number(process.env.CLOTH_COST_EVERY ?? 1), HELD = process.env.CLOTH_COST_HELD === "1";

async function withDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const features = (["timestamp-query", "subgroups"] as GPUFeatureName[]).filter(f => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); }
  finally { device.destroy(); }
}

/** Each named encode in its own timestamped pass, median of 5 (ms); `before` runs in a pass of its own first. */
async function timePasses(device: GPUDevice, names: string[], encode: (pass: GPUComputePassEncoder, name: string) => void, before?: (pass: GPUComputePassEncoder) => void): Promise<number[]> {
  const set = device.createQuerySet({ type: "timestamp", count: 2 * names.length });
  const resolve = device.createBuffer({ size: 16 * names.length, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 16 * names.length, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const runs: number[][] = [];
  for (let rep = 0; rep < 5; rep++) {
    const enc = device.createCommandEncoder();
    if (before) { const p = enc.beginComputePass(); before(p); p.end(); }
    names.forEach((n, i) => {
      const pass = enc.beginComputePass({ timestampWrites: { querySet: set, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      encode(pass, n); pass.end();
    });
    enc.resolveQuerySet(set, 0, 2 * names.length, resolve, 0);
    enc.copyBufferToBuffer(resolve, 0, read, 0, 16 * names.length);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(read.getMappedRange().slice(0));
    read.unmap();
    runs.push(names.map((_, i) => Number(t[2 * i + 1]! - t[2 * i]!) * 1e-6));
  }
  set.destroy(); resolve.destroy(); read.destroy();
  return names.map((_, i) => runs.map(r => r[i]!).sort((a, b) => a - b)[2]!);
}

test("cost: the rotating-sphere sheet's detection, round by round and kernel by kernel", { skip: !want }, async t => withDevice(async device => {
  const log = (s: string) => { t.diagnostic(s); console.log(s); };
  const scene = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: NODES }), HELD ? { movingSolids: "held" } : {});
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 });
  const doc = scene.cloth![0]!, det = cloth.contactFor(doc.id)!, solver = cloth.solverFor(doc.id)!;
  const N = doc.geometry.data.restPositions_m.length / 3;
  log(`cost: sphere ${NODES}^2 (${N} vertices), ${HELD ? "held" : "moving"} sphere, ${TICKS} ticks`);
  const sum = { ms: 0, newton: 0, rounds: 0, re: 0, redos: 0, levelled: 0, fallback: 0, scanned: 0, collisions: 0, rejected: 0, candidates: 0, dat: 0 };
  let failure: string | undefined, maxRedos = 0, maxRounds = 0;
  try {
    for (let k = 1; k <= TICKS; k++) {
      const t0 = performance.now();
      try { await cloth.capture([k * CLOTH_SCENE_STEP_S]); } catch (e) { failure = `tick ${k}: ${String(e).slice(0, 240)}`; break; }
      const ms = performance.now() - t0;
      const c = await det.readCounters(), ctl = Array.from(await solver.readNewtonControl() as ArrayLike<number>), rl = await det.readRoundLog();
      sum.ms += ms; sum.newton += ctl[3]!; sum.rounds += c.detections; sum.re += c.redetections; sum.redos += c.redos;
      sum.levelled += c.levelled; sum.fallback += c.levelFallback; sum.scanned += c.scanned; sum.collisions += c.collisions;
      sum.rejected += c.boxRejected; sum.candidates += c.candidates; sum.dat += c.datPairs;
      maxRedos = Math.max(maxRedos, c.redos); maxRounds = Math.max(maxRounds, c.detections);
      if (k % EVERY === 0 || k === TICKS) {
        log(`  tick ${k}: ${ms.toFixed(1)} ms, newton ${ctl[3]}; rounds ${c.detections} (re ${c.redetections}, redos ${c.redos}, levelled ${c.levelled}, fallback ${c.levelFallback}); `
          + `scan ${c.scanned}/${c.collisions}/${c.boxRejected}/${c.candidates} dat ${c.datPairs}; status [${OgcContactDetector.describeStatus(c.status).join(",")}]`);
        for (const [i, r] of rl.entries()) log(`    round ${i}: |w| ${r.maxPrediction.toExponential(2)} env ${r.maxEnvelope.toExponential(2)} scan ${r.scanned}/${r.collisions}/${r.boxRejected}/${r.candidates} dat ${r.datPairs}`);
      }
      if (cloth.failure) { failure = `tick ${k}: ${cloth.failure.message}`; break; }
    }
    const n = TICKS, per = (x: number) => (x / n).toFixed(1);
    log(`cost per tick (mean of ${n}): ${per(sum.ms)} ms wall, newton ${per(sum.newton)}, rounds ${per(sum.rounds)} (re ${per(sum.re)}, redos ${per(sum.redos)}, levelled ${per(sum.levelled)}, fallback ${per(sum.fallback)}); `
      + `scanned ${per(sum.scanned)}, collisions ${per(sum.collisions)}, box-rejected ${per(sum.rejected)}, candidates ${per(sum.candidates)}, DAT pairs ${per(sum.dat)}; most in one tick: rounds ${maxRounds}, redos ${maxRedos}`);
    if (device.features.has("timestamp-query")) {
      const names = [...OGC_ROUND_KERNELS, "redo"] as const;
      const round = await timePasses(device, [...names], (p, nm) => det.encodeRoundKernel(p, nm as (typeof names)[number]), p => det.encodeStepBegin(p));
      const rest = await timePasses(device, ["unarmed mandatory detection", "unarmed later detection", "decide", "truncate", "filter"], (p, nm) => {
        if (nm === "unarmed mandatory detection") det.encodeDetection(p, true);
        else if (nm === "unarmed later detection") det.encodeDetection(p, false);
        else if (nm === "decide") det.encodeDecide(p);
        else if (nm === "truncate") det.encodeTruncate(p);
        else det.encodeFilter(p);
      });
      const roundMs = round.reduce((a, b) => a + b, 0);
      log(`cost kernels at tick ${n}'s state: an armed mandatory round ${roundMs.toFixed(3)} ms (${names.map((nm, i) => `${nm} ${round[i]!.toFixed(3)}`).join(", ")}); `
        + `unarmed mandatory detection ${rest[0]!.toFixed(3)}, unarmed later detection ${rest[1]!.toFixed(3)}, decide ${rest[2]!.toFixed(3)}, truncate ${rest[3]!.toFixed(3)}, filter ${rest[4]!.toFixed(3)} ms`);
      const newton = sum.newton / n;
      log(`cost model per tick: rounds x armed round ${(sum.rounds / n * roundMs).toFixed(2)} ms; unarmed floor ${(rest[0]! + newton * rest[1]!).toFixed(2)} ms `
        + `(mandatory ${rest[0]!.toFixed(2)} + ${newton.toFixed(1)} x later ${rest[1]!.toFixed(3)}); filter + truncate + decide ${(newton * (rest[4]! + rest[3]! + rest[2]!) + rest[3]!).toFixed(2)} ms`);
    }
  } finally { cloth.destroy(); }
  assert.equal(failure, undefined, failure);
}));

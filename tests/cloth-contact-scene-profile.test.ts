/**
 * Scene-contact profile (GPU, opt in with CLOTH_CONTACT_PROFILE=rotating): the
 * Bridson rotating-sphere sheet built as ClothSystem builds it (scene field,
 * createClothSceneContact, K iterations), run once per truncation schedule.
 *
 *   CLOTH_PROFILE_TRUNCATE  comma list of iteration (every iteration, the
 *                           detector's hooks) and colour (also after every colour
 *                           sweep, the schedule before 2026-11-09); default iteration
 *   CLOTH_PROFILE_TICKS     ticks per schedule, default 20
 *
 * Per schedule: each tick fenced alone (the step only), then the intersection
 * count and the counters, checked as the gates check them (no failing status, no
 * intersecting face pair, no face reaching into the solid, no vertex inside it).
 * After the ticks: each detection kernel, the truncation's two kernels, the decide
 * and an unarmed detection timed in their own passes, and the candidate diagnosis
 * (r_q, rho and held shares, the broad phase's entries per occupied cell and the
 * entries a vertex query scans, rebuilt on the CPU from the round's own state).
 */
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { clothVertexNeighbours, colourClothVertices, prepareClothTopology } from "../lib/cloth/topology";
import { buildClothSolverModel } from "../lib/cloth/oracle";
import { ClothGPUSolver } from "../lib/cloth/gpu-solver";
import { CLOTH_BINDING } from "../lib/cloth/gpu-solver-wgsl";
import { CLOTH_SCENE_STEP_S } from "../lib/cloth/system";
import { CLOTH_CONTACT_FAILING, buildClothSceneField, clothContactRadius, createClothSceneContact } from "../lib/cloth/scene-contact";
import { OGC_ROUND_KERNELS, OgcContactDetector } from "../lib/cloth/contact/ogc-detector";
import { sampleClothSolidField, type ClothSolidField, type V3 } from "../lib/cloth/contact/solid-field";
import { bridson2002RotatingSphere } from "../lib/cloth/scenes/bridson-2002";
import { bridson2002SceneDescription } from "../lib/cloth/scenes/bridson-2002/scene-description";
import type { ClothVector } from "../lib/cloth/types";

const want = !!process.env.WEBGPU_NODE_MODULE && process.env.CLOTH_CONTACT_PROFILE === "rotating";
const SCHEDULES = (process.env.CLOTH_PROFILE_TRUNCATE ?? "iteration").split(",").filter(Boolean);
const TICKS = Number(process.env.CLOTH_PROFILE_TICKS ?? 20);
const ITERATIONS = 64;

async function withDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    requiredFeatures: adapter.features.has("timestamp-query") ? ["timestamp-query"] : [] });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); }
  finally { device.destroy(); }
}

/** Each named encode in its own timestamped pass, median of 5 (ms). */
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

const quantiles = (a: ArrayLike<number>, qs = [0, 0.5, 0.9, 0.99, 1]) => {
  const s = Float64Array.from(a).sort();
  return qs.map(q => s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))]!);
};
const fmt = (xs: number[], d = 2) => xs.map(x => x.toExponential(d)).join(" / ");

/**
 * The round's broad phase rebuilt on the CPU from its own state (x_det, w, r_q,
 * cell): each face's grid box (triBox), its cells, and per vertex the entries its
 * query scans (every entry of every bucket its box covers, hash collisions
 * included) and the candidates it tests (box overlap, owned cell, not incident).
 */
function brokenDownGrid(d: Awaited<ReturnType<OgcContactDetector["readDiagnosis"]>>, faces: Uint32Array, N: number, H: number) {
  const inv = 1 / d.cell, T = faces.length / 3;
  const X = (i: number, c: number) => d.xdet[4 * i + c]!, W = (i: number, c: number) => d.pred[4 * i + c]!;
  const boxes = new Float64Array(6 * T);
  const key = (x: number, y: number, z: number) => `${x},${y},${z}`;
  const hash = (x: number, y: number, z: number) => ((Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0) & (H - 1);
  const cells = new Map<string, number[]>(), buckets = new Map<number, number>();
  for (let t = 0; t < T; t++) {
    const v = [faces[3 * t]!, faces[3 * t + 1]!, faces[3 * t + 2]!], e = Math.max(...v.map(i => d.env[i]!));
    for (let c = 0; c < 3; c++) {
      const ps = v.flatMap(i => [X(i, c), X(i, c) + W(i, c)]);
      boxes[6 * t + c] = Math.min(...ps) - e; boxes[6 * t + 3 + c] = Math.max(...ps) + e;
    }
    const c0 = [0, 1, 2].map(c => Math.floor(boxes[6 * t + c]! * inv)), c1 = [0, 1, 2].map(c => Math.floor(boxes[6 * t + 3 + c]! * inv));
    for (let z = c0[2]!; z <= c1[2]!; z++) for (let y = c0[1]!; y <= c1[1]!; y++) for (let x = c0[0]!; x <= c1[0]!; x++) {
      const k = key(x, y, z); let l = cells.get(k); if (!l) cells.set(k, l = []); l.push(t);
      const h = hash(x, y, z); buckets.set(h, (buckets.get(h) ?? 0) + 1);
    }
  }
  const perCell = [...cells.values()].map(l => l.length);
  const scanned = new Float64Array(N), tested = new Float64Array(N), cellsPerQuery = new Float64Array(N);
  for (let v = 0; v < N; v++) {
    const r = d.env[v]!, lo = [0, 1, 2].map(c => Math.min(X(v, c), X(v, c) + W(v, c)) - r), hi = [0, 1, 2].map(c => Math.max(X(v, c), X(v, c) + W(v, c)) + r);
    const c0 = lo.map(x => Math.floor(x * inv)), c1 = hi.map(x => Math.floor(x * inv));
    for (let z = c0[2]!; z <= c1[2]!; z++) for (let y = c0[1]!; y <= c1[1]!; y++) for (let x = c0[0]!; x <= c1[0]!; x++) {
      cellsPerQuery[v]!++;
      scanned[v]! += buckets.get(hash(x, y, z)) ?? 0;
      for (const t of cells.get(key(x, y, z)) ?? []) {
        if (faces[3 * t] === v || faces[3 * t + 1] === v || faces[3 * t + 2] === v) continue;
        let overlap = true, owned = true;
        for (let c = 0; c < 3; c++) {
          if (lo[c]! > boxes[6 * t + 3 + c]! || boxes[6 * t + c]! > hi[c]!) overlap = false;
          if (Math.floor(Math.max(lo[c]!, boxes[6 * t + c]!) * inv) !== [x, y, z][c]) owned = false;
        }
        if (overlap && owned) tested[v]!++;
      }
    }
  }
  return { cells: cells.size, perCell, scanned, tested, cellsPerQuery, entries: perCell.reduce((a, b) => a + b, 0) };
}

/** The faces below phi 0 at a 16-step barycentric lattice (Float64 on the stored field), worst first. */
function solidDiagnosis(field: ClothSolidField, x: Float64Array, origin: ClothVector, faces: Uint32Array): string {
  const P = (v: number): V3 => [x[3 * v]! - origin[0], x[3 * v + 1]! - origin[1], x[3 * v + 2]! - origin[2]];
  const rows: { depth: number; text: string }[] = [];
  for (let t = 0; t < faces.length / 3; t++) {
    const A = [0, 1, 2].map(k => P(faces[3 * t + k]!)), n = 16;
    let worst = Infinity, at: V3 = [0, 0, 0];
    for (let i = 0; i <= n; i++) for (let j = 0; j <= n - i; j++) {
      const u = i / n, w = j / n, p = [0, 1, 2].map(a => (1 - u - w) * A[0]![a]! + u * A[1]![a]! + w * A[2]![a]!) as unknown as V3;
      const q = sampleClothSolidField(field, p);
      if (q.stored && q.phi < worst) { worst = q.phi; at = p; }
    }
    if (!(worst < 0)) continue;
    const g = sampleClothSolidField(field, at).grad, edge = Math.max(...[0, 1, 2].map(k => Math.hypot(...[0, 1, 2].map(a => A[k]![a]! - A[(k + 1) % 3]![a]!))));
    rows.push({ depth: worst, text: `face ${t} depth ${worst.toExponential(2)} at (${at.map(v => v.toFixed(4)).join(",")}) normal (${g.map(v => v.toFixed(2)).join(",")}), `
      + `vertex phi ${A.map(p => sampleClothSolidField(field, p).phi.toExponential(2)).join("/")}, longest edge ${edge.toExponential(2)}` });
  }
  rows.sort((a, b) => a.depth - b.depth);
  return `${rows.length} faces below phi 0 (Float64, cell ${field.cell}): ${rows.slice(0, 5).map(r => r.text).join(" | ")}`;
}

test("profile: the rotating-sphere sheet through scene contact, per truncation schedule", { skip: !want }, async t => withDevice(async device => {
  const log = (s: string) => { t.diagnostic(s); console.log(s); };
  const scene = bridson2002SceneDescription(bridson2002RotatingSphere()), document = scene.cloth![0]!;
  const data = document.geometry.kind === "inline" ? document.geometry.data : assert.fail("inline geometry");
  const topology = prepareClothTopology(data, document.materials);
  const model = buildClothSolverModel(topology, document.materials, colourClothVertices(clothVertexNeighbours(topology)), document.scripts ?? []);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  data.restPositions_m.forEach((v, i) => { lo[i % 3] = Math.min(lo[i % 3]!, v); hi[i % 3] = Math.max(hi[i % 3]!, v); });
  const origin = [0, 1, 2].map(a => (lo[a]! + hi[a]!) / 2) as unknown as ClothVector;
  const g = scene.fluid.gravity_m_s2, gravity: [number, number, number] = [g.x, g.y, g.z], r = clothContactRadius(document);
  const t0 = performance.now(), field = buildClothSceneField(scene, origin, r);
  const N = topology.vertexCount, colours = model.colourCount;
  log(`rotating sphere: ${N} vertices, ${topology.faceCount} faces, ${topology.edgeCount} edges, ${colours} colours, K = ${ITERATIONS}, r ${r}; field ${field.bricks.size} bricks, built in ${(performance.now() - t0).toFixed(0)} ms`);

  const failures: string[] = [], diagnosed = new Set<string>();
  for (const schedule of SCHEDULES) {
    if (schedule !== "colour" && schedule !== "iteration") throw new Error(`Unknown truncation schedule "${schedule}"`);
    const det = await createClothSceneContact(device, model, document, origin, field, CLOTH_SCENE_STEP_S, gravity);
    const hooks = schedule === "colour" ? { ...det.stepHooks, afterColour: (p: GPUComputePassEncoder) => det.encodeTruncate(p) } : det.stepHooks;
    const solver = await ClothGPUSolver.create(device, model, {
      origin_m: origin, dt_s: CLOTH_SCENE_STEP_S, gravity_m_s2: gravity, integrator: "bdf2", iterations: ITERATIONS, chebyshevRho: 0,
      positions_m: data.initialPositions_m, velocities_m_s: data.initialVelocities_m_s, label: `profile ${schedule}`,
      contact: det.solverContact(CLOTH_BINDING.contact), hooks,
    });
    det.bind({ positions: solver.positions, velocities: solver.velocities, stepStart: solver.stepStart });
    det.setSolidField(field);
    const truncations = schedule === "iteration" ? ITERATIONS + 1 : ITERATIONS * (colours + 1) + 1;
    const inside = async () => {
      const x = await solver.readPositions();
      let n = 0, minPhi = Infinity;
      for (let v = 0; v < N; v++) {
        const phi = sampleClothSolidField(field, [x[3 * v]! - origin[0], x[3 * v + 1]! - origin[1], x[3 * v + 2]! - origin[2]]).phi;
        minPhi = Math.min(minPhi, phi); if (phi <= 0) n++;
      }
      return { n, minPhi };
    };
    const tickMs: number[] = [];
    let worstIntersections = 0, worstSolid = 0, minPhi = Infinity;
    for (let s = 0; s < TICKS; s++) {
      await device.queue.onSubmittedWorkDone();
      const ts = performance.now();
      let enc = device.createCommandEncoder(), pass = enc.beginComputePass();
      det.encodeFrameBegin(pass); solver.encodeStep(pass); pass.end(); device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      tickMs.push(performance.now() - ts);
      enc = device.createCommandEncoder(); pass = enc.beginComputePass(); det.encodeIntersectionCount(pass); pass.end(); device.queue.submit([enc.finish()]);
      const c = await det.readCounters(), ins = await inside();
      worstIntersections = Math.max(worstIntersections, c.intersections); worstSolid = Math.max(worstSolid, c.solidIntersections); minPhi = Math.min(minPhi, ins.minPhi);
      log(`${schedule} tick ${s}: ${tickMs[s]!.toFixed(1)} ms; rounds ${c.detections} (re ${c.redetections}, redos ${c.redos}), decides ${c.decides}, truncated/N ${(c.truncated / N).toFixed(3)}, `
        + `candidates/N per round ${(c.candidates / Math.max(1, c.detections) / N).toFixed(1)}, pairs/N per round vf ${(c.vfPairs / Math.max(1, c.detections) / N).toFixed(3)} ee ${(c.eePairs / Math.max(1, c.detections) / N).toFixed(3)} `
        + `dat ${(c.datPairs / Math.max(1, c.detections) / N).toFixed(2)} sf ${c.sfPairs}, held/N ${(c.held / N).toFixed(3)}, conflicts ${c.conflicts}, solid ${c.solidContacts} cut ${c.solidCut}, `
        + `capped ${c.capped} uncertified ${c.uncertified}; intersections ${c.intersections}, faces in solid ${c.solidIntersections}, vertices inside ${ins.n}, min phi ${ins.minPhi.toExponential(2)}, `
        + `status [${OgcContactDetector.describeStatus(c.status).join(", ")}]`);
      if (c.solidIntersections && !diagnosed.has(schedule)) { diagnosed.add(schedule); log(`${schedule} tick ${s}: ${solidDiagnosis(field, await solver.readPositions(), origin, topology.faces)}`); }
      // Recorded, not thrown: every schedule runs and profiles; the test fails at the end.
      if (c.status & CLOTH_CONTACT_FAILING) failures.push(`${schedule}: status ${OgcContactDetector.describeStatus(c.status)} at tick ${s}`);
      if (c.intersections) failures.push(`${schedule}: ${c.intersections} intersecting face pairs at tick ${s}`);
      if (c.solidIntersections) failures.push(`${schedule}: ${c.solidIntersections} faces reach into the solid at tick ${s}`);
      if (ins.n) failures.push(`${schedule}: ${ins.n} vertices inside the solid at tick ${s}`);
    }
    const med = [...tickMs].sort((a, b) => a - b)[tickMs.length >> 1]!;
    log(`${schedule} summary: ${truncations} truncations and ${ITERATIONS} decides per step; median tick ${med.toFixed(1)} ms (first ${tickMs[0]!.toFixed(1)}, max ${Math.max(...tickMs).toFixed(1)}); `
      + `max intersections ${worstIntersections}, max faces in solid ${worstSolid}, min vertex phi ${minPhi.toExponential(2)}`);

    // Candidate diagnosis on the step's last round (before the kernel timings re-run a round).
    const d = await det.readDiagnosis();
    const grid = brokenDownGrid(d, topology.faces, N, det.config.bucketCount);
    const heldShare = d.held.reduce((a, f) => a + (f !== 0 ? 1 : 0), 0) / N;
    // The hold rule's residual half: rho > |w_step| / 2 (pred[v].w is the step's predicted length).
    const plen = Float64Array.from({ length: N }, (_, v) => d.pred[4 * v + 3]!);
    const rhoHeld = plen.reduce((a, l, v) => a + (2 * d.rho[v]! > l ? 1 : 0), 0) / N;
    log(`${schedule} diagnosis: cell ${d.cell.toExponential(3)} m; r_q quantiles (0/50/90/99/100) ${fmt(quantiles(d.env))} m; rho ${fmt(quantiles(d.rho))} m; `
      + `|w| ${fmt(quantiles(Float64Array.from({ length: N }, (_, v) => Math.hypot(d.pred[4 * v]!, d.pred[4 * v + 1]!, d.pred[4 * v + 2]!))))} m; step length ${fmt(quantiles(plen))} m; held share ${heldShare.toFixed(3)} (rho > |w|/2: ${rhoHeld.toFixed(3)}); `
      + `grid ${grid.entries} entries in ${grid.cells} cells, faces per occupied cell ${quantiles(grid.perCell).join(" / ")}; `
      + `per vertex query: cells ${quantiles(grid.cellsPerQuery).join(" / ")}, entries scanned ${quantiles(grid.scanned).join(" / ")} (mean ${(grid.scanned.reduce((a, b) => a + b, 0) / N).toFixed(1)}), `
      + `candidates ${quantiles(grid.tested).join(" / ")} (mean ${(grid.tested.reduce((a, b) => a + b, 0) / N).toFixed(1)})`);
    // Kernel costs at the last state.
    if (device.features.has("timestamp-query")) {
      const roundNames: ((typeof OGC_ROUND_KERNELS)[number] | "redo")[] = [...OGC_ROUND_KERNELS, "redo"];
      const round = await timePasses(device, roundNames, (p, n) => det.encodeRoundKernel(p, n as (typeof roundNames)[number]), p => det.encodeStepBegin(p));
      const trunc = await timePasses(device, ["planes", "apply", "decide", "unarmed detection"], (p, n) => {
        if (n === "planes" || n === "apply") det.encodeTruncateKernel(p, n);
        else if (n === "decide") det.encodeDecide(p);
        else det.encodeDetection(p);
      });
      const roundMs = round.reduce((a, b) => a + b, 0);
      log(`${schedule} kernels: round ${roundMs.toFixed(3)} ms (${roundNames.map((n, i) => `${n} ${round[i]!.toFixed(3)}`).filter((_, i) => round[i]! >= 0.005).join(", ")}); `
        + `planes ${trunc[0]!.toFixed(3)}, apply ${trunc[1]!.toFixed(3)}, decide ${trunc[2]!.toFixed(3)}, unarmed detection ${trunc[3]!.toFixed(3)} ms; `
        + `truncation per step ${(truncations * (trunc[0]! + trunc[1]!)).toFixed(1)} ms, decides + unarmed detections ${(ITERATIONS * (trunc[2]! + trunc[3]!)).toFixed(1)} ms`);
    }

    solver.destroy(); det.destroy();
  }
  assert.deepEqual(failures, []);
}));

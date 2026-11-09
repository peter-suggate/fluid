/**
 * Milestone gates of the cloth contact (lane C2) inside C1's Tier A GPU
 * solver: the OGC detector drives the solver through its step hooks, and the
 * solver's vertex kernel and line search carry the contact energy.
 *
 *   drop   ten 31 x 31 sheets, each turned a few degrees, fall onto a voxel
 *          pedestal (a sharp box on a floor slab, both from the static solid
 *          field: no analytic collider) and pile over its edges
 *   drape  a 61 x 61 sheet falls onto a smooth sphere (an sdf part: no surface
 *          samples) resting on a floor slab, and drapes to the floor
 *   twist  a 100 x 100 sheet whose two opposite edges turn a quarter turn per
 *          second each in opposite directions while moving together (OGC Fig. 12)
 *
 * Every frame: no intersecting face pair, no vertex inside a solid, no cloth
 * face reaching into a solid (the sharp edges are held off the faces by their
 * surface samples), no overflowing capacity. Solid friction is the pedestal's own (0.3); cloth on
 * cloth uses the detector's mu (0.2). The frame's rounds, truncations, pairs, envelope and
 * residual are reported; then one timed batch of steps without readback.
 *
 * GPU only and opt-in: CLOTH_CONTACT_GATES=drop,drape,twist (or "all");
 * CLOTH_GATE_STEPS (default 60); CLOTH_GATE_VARIANTS=all adds the isotropic
 * truncation and the fixed envelope to the drop.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { clothGridData } from "../lib/cloth/grid";
import { prepareClothTopology } from "../lib/cloth/topology";
import { CLOTH_NEWTON_SCENE_DEFAULTS } from "../lib/cloth/system";
import type { ClothAttachmentScript, ClothMaterial, ClothTopologyData } from "../lib/cloth/types";
import { buildClothSolverModel, type ClothSolverModel } from "../lib/cloth/oracle";
import { ClothGPUSolver } from "../lib/cloth/gpu-solver";
import { CLOTH_BINDING, CLOTH_STATE } from "../lib/cloth/gpu-solver-wgsl";
import { OGC_ROUND_KERNELS, OgcContactDetector, type OgcContactConfig, type OgcCellRule, type OgcFrameCounters, type OgcPrediction } from "../lib/cloth/contact/ogc-detector";
import { OGC_STATUS } from "../lib/cloth/contact/ogc-wgsl";
import { buildClothSolidField, clothSolidSamples, sampleClothSolidField, type ClothSolidField } from "../lib/cloth/contact/solid-field";
import { facesInsideSolid, trianglesIntersect, type V3 } from "./cloth-contact-oracle";

const gates = new Set((process.env.CLOTH_CONTACT_GATES ?? "").split(",").filter(Boolean));
const want = (name: string) => !!process.env.WEBGPU_NODE_MODULE && (gates.has(name) || gates.has("all"));
const STEPS = Number(process.env.CLOTH_GATE_STEPS ?? 60);
const TIMED_STEPS = 30;
/** DAT pair slots per vertex, authored per scene from the measured peak. */
const PAIRS_PER_VERTEX = Number(process.env.CLOTH_GATE_PAIRS ?? 840);

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

const cotton: ClothMaterial = { id: "cotton", arealDensity_kg_m2: 0.2, stretchWarp_N_m: 400, stretchWeft_N_m: 250, shear_N_m: 40, poisson: 0.3,
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0.002, thickness_m: 5e-4, strainLimit: 0.02 };
const g: [number, number, number] = [0, -9.81, 0];
/**
 * Envelope safety factor s. A round lets a vertex move 0.5 gamma_r r_q =
 * 0.45 (r + s d) of its inertial displacement d, so s > 2 / gamma_r = 2.22 is
 * needed for an unobstructed vertex to finish its step in one round.
 */
const ENVELOPE_S = Number(process.env.CLOTH_GATE_S ?? 2.5);
const DT = 1 / 60, NEWTON = CLOTH_NEWTON_SCENE_DEFAULTS, KC = Number(process.env.CLOTH_GATE_KC ?? 100);
/**
 * CLOTH_GATE_PREDICT: guess (default; the solver's initial guess), inertial, or
 * none (Planar-DAT with every prediction held, the detector before predicted
 * motion). CLOTH_GATE_CELL (m): the broad-phase cell's floor; CLOTH_GATE_CELL_RULE:
 * largest (default) or cubicMean.
 */
const PREDICTION = (process.env.CLOTH_GATE_PREDICT ?? "guess") as OgcPrediction;
const CELL = process.env.CLOTH_GATE_CELL ? Number(process.env.CLOTH_GATE_CELL) : undefined;
const CELL_RULE = (process.env.CLOTH_GATE_CELL_RULE ?? "largest") as OgcCellRule;
const knobs = { prediction: PREDICTION, cellRule: CELL_RULE, ...(CELL ? { cellSize: CELL } : {}) };

/**
 * One mandatory round at the current state, each kernel in its own timestamped
 * pass (median of 5): the detection's per-pass breakdown. It re-runs a step's
 * round and truncation without the solver, so it runs after the gate's checks.
 */
async function profileRound(device: GPUDevice, det: OgcContactDetector): Promise<string> {
  if (!device.features.has("timestamp-query")) return "no timestamp-query";
  const names = [...OGC_ROUND_KERNELS.map(n => `${n}`), "redo", ...OGC_ROUND_KERNELS.map(n => `held ${n}`), "planes", "apply"];
  const set = device.createQuerySet({ type: "timestamp", count: 2 * names.length });
  const resolve = device.createBuffer({ size: 16 * names.length, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 16 * names.length, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const runs: number[][] = [];
  for (let rep = 0; rep < 5; rep++) {
    const enc = device.createCommandEncoder();
    let p = enc.beginComputePass(); det.encodeStepBegin(p); p.end();
    names.forEach((n, i) => {
      const pass = enc.beginComputePass({ timestampWrites: { querySet: set, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      if (n === "planes" || n === "apply") det.encodeTruncateKernel(pass, n);
      else det.encodeRoundKernel(pass, n.replace("held ", "") as (typeof OGC_ROUND_KERNELS)[number] | "redo");
      pass.end();
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
  const med = names.map((_, i) => runs.map(r => r[i]!).sort((a, b) => a - b)[2]!);
  const total = med.reduce((a, b) => a + b, 0);
  return `round ${total.toFixed(3)} ms: ` + names.map((n, i) => `${n} ${med[i]!.toFixed(3)}`).filter((_, i) => med[i]! >= 0.005).join(", ");
}

/** Status bits that fail a gate (the round and envelope caps are reported, not failed). */
const FAILING = OGC_STATUS.contactOverflow | OGC_STATUS.gridOverflow | OGC_STATUS.pairOverflow | OGC_STATUS.nonFinite | OGC_STATUS.solidPenetration | OGC_STATUS.solidBounds;

function concat(sheets: ClothTopologyData[]): ClothTopologyData {
  const rest: number[] = [], triangles: number[] = [], uv: number[] = [], groups: number[] = [];
  for (const s of sheets) {
    const base = rest.length / 3;
    rest.push(...s.restPositions_m); triangles.push(...s.triangles.map(i => i + base)); uv.push(...(s.uv ?? [])); groups.push(...(s.groups ?? []));
  }
  return { restPositions_m: rest, triangles, uv, groups };
}

function model(data: ClothTopologyData, scripts: ClothAttachmentScript[] = []): ClothSolverModel {
  const topology = prepareClothTopology(data, [cotton]);
  return buildClothSolverModel(topology, [cotton], scripts);
}

/** The detector's mesh and rest pose from the solver model (cloth-local = world here: origin 0). */
function meshOf(m: ClothSolverModel) {
  const t = m.topology, T = t.faceCount;
  const triangles = new Uint32Array(4 * T);
  for (let f = 0; f < T; f++) { triangles.set(t.faces.subarray(3 * f, 3 * f + 3), 4 * f); triangles[4 * f + 3] = t.faceMaterial[f]!; }
  const rest = new Float32Array(4 * t.vertexCount);
  for (let v = 0; v < t.vertexCount; v++) for (let c = 0; c < 3; c++) rest[4 * v + c] = t.data.restPositions_m[3 * v + c]!;
  return { mesh: { vertexCount: t.vertexCount, triangles, edges: t.edges }, rest };
}

/** Float64 count of intersecting vertex-disjoint face pairs (gate diagnosis only). */
function cpuIntersections(x: Float64Array, faces: Uint32Array, cell: number): { count: number; pairs: [number, number][] } {
  const T = faces.length / 3, grid = new Map<string, number[]>(), pairs: [number, number][] = [];
  const P = (v: number): V3 => [x[3 * v]!, x[3 * v + 1]!, x[3 * v + 2]!];
  const span = (t: number) => {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < 3; k++) for (let a = 0; a < 3; a++) { const c = x[3 * faces[3 * t + k]! + a]!; lo[a] = Math.min(lo[a]!, c); hi[a] = Math.max(hi[a]!, c); }
    return [lo.map(v => Math.floor(v / cell)), hi.map(v => Math.floor(v / cell))];
  };
  for (let t = 0; t < T; t++) {
    const [lo, hi] = span(t);
    for (let i = lo[0]!; i <= hi[0]!; i++) for (let j = lo[1]!; j <= hi[1]!; j++) for (let k = lo[2]!; k <= hi[2]!; k++) {
      const key = `${i},${j},${k}`; let l = grid.get(key); if (!l) grid.set(key, l = []); l.push(t);
    }
  }
  const seen = new Set<string>();
  for (const l of grid.values()) for (let a = 0; a < l.length; a++) for (let b = a + 1; b < l.length; b++) {
    const s = l[a]!, t = l[b]!, key = `${Math.min(s, t)},${Math.max(s, t)}`;
    if (seen.has(key)) continue; seen.add(key);
    const A = [0, 1, 2].map(k => faces[3 * s + k]!), B = [0, 1, 2].map(k => faces[3 * t + k]!);
    if (A.some(v => B.includes(v))) continue;
    if (trianglesIntersect(A.map(P), B.map(P))) pairs.push([s, t]);
  }
  return { count: pairs.length, pairs };
}

interface Frame { step: number; c: OgcFrameCounters }

/** Debug: the faces reaching phi < 0 (Float64 over the field), their depth, vertex phi and nearest sample. */
async function solidDiagnosis(field: ClothSolidField, x: Float64Array, faces: Uint32Array, det: OgcContactDetector): Promise<string> {
  const S = clothSolidSamples(field), P = (v: number): V3 => [x[3 * v]!, x[3 * v + 1]!, x[3 * v + 2]!];
  const rows: string[] = [];
  for (let t = 0; t < faces.length / 3; t++) {
    const A = [0, 1, 2].map(k => P(faces[3 * t + k]!)), n = 16;
    let worst = Infinity, at: V3 = [0, 0, 0];
    for (let i = 0; i <= n; i++) for (let j = 0; j <= n - i; j++) {
      const u = i / n, w = j / n, p = [0, 1, 2].map(a => (1 - u - w) * A[0]![a]! + u * A[1]![a]! + w * A[2]![a]!) as V3;
      const q = sampleClothSolidField(field, p);
      if (q.stored && q.phi < worst) { worst = q.phi; at = p; }
    }
    if (!(worst < 0)) continue;
    let near = Infinity, nearPhi = 0;
    for (let k = 0; k < S.length / 4; k++) {
      const d = Math.hypot(S[4 * k]! - at[0], S[4 * k + 1]! - at[1], S[4 * k + 2]! - at[2]);
      if (d < near) { near = d; nearPhi = sampleClothSolidField(field, [S[4 * k]!, S[4 * k + 1]!, S[4 * k + 2]!]).phi; }
    }
    // Distance from the nearest sample to the face's plane region (its closest point).
    rows.push(`face ${t} depth ${worst.toExponential(2)} at (${at.map(v => v.toFixed(4)).join(",")}), vertex phi ${A.map(p => sampleClothSolidField(field, p).phi.toExponential(2)).join("/")}, nearest sample ${near.toExponential(2)} (lift ${nearPhi.toExponential(2)})`);
  }
  void det;
  return `${rows.length} faces below phi 0 (Float64): ${rows.slice(0, 6).join(" | ")}`;
}

async function runGate(device: GPUDevice, name: string, m: ClothSolverModel, config: OgcContactConfig, field: ClothSolidField | undefined, log: (s: string) => void) {
  const { mesh, rest } = meshOf(m), N = mesh.vertexCount;
  const det = await OgcContactDetector.create(device, mesh, rest, { ...config, newton: true });
  const solver = await ClothGPUSolver.create(device, m, {
    origin_m: [0, 0, 0], dt_s: DT, gravity_m_s2: g, integrator: "bdf2", newton: NEWTON, diagnostics: true,
    label: name,
    // CLOTH_GATE_NO_CONTACT: the solver alone (diagnosis baseline; the intersection count still runs).
    ...(process.env.CLOTH_GATE_NO_CONTACT ? {} : {
      contact: { ...det.solverContact(CLOTH_BINDING.contact, true), alpha: det.alpha, state: det.contactState },
      hooks: { afterFrame: det.stepHooks.afterFrame, filter: det.stepHooks.filter, afterNewtonStep: det.stepHooks.afterNewtonStep, stepEnd: det.stepHooks.stepEnd },
    }),
  });
  // x_t: the solver's step-start segment (C1 to expose it as a getter; see COORDINATION 2026-11-09).
  const stride = solver.tables.stride;
  det.bind({ positions: solver.positions, velocities: solver.velocities, stepStart: { buffer: solver.state, offset: 16 * CLOTH_STATE.xt * stride, size: 16 * stride }, ...(det.config.prediction === "target" ? { target: solver.inertialTarget } : {}), direction: solver.newtonDirection });
  if (field) { det.setSolidField(field); log(`${name}: ${det.solidSamples} surface samples at the field's sharp features (capacity ${det.solidSampleCapacity})`); }

  const frame = (count: number, intersections: boolean) => {
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    det.encodeFrameBegin(pass);
    for (let s = 0; s < count; s++) solver.encodeStep(pass);
    if (intersections) det.encodeIntersectionCount(pass);
    pass.end(); device.queue.submit([enc.finish()]);
  };
  const inside = async () => {
    if (!field) return 0;
    const x = await solver.readPositions();
    let n = 0;
    for (let v = 0; v < N; v++) if (sampleClothSolidField(field, [x[3 * v]!, x[3 * v + 1]!, x[3 * v + 2]!]).phi <= 0) n++;
    return n;
  };

  // Every face against the trilinear surface, exactly (Float64 Bernstein bounds), not at a lattice.
  let faceClearance = Infinity;
  const exactFaces = (x: Float64Array, when: string) => {
    if (!field) return;
    const r = facesInsideSolid(field, x, m.topology.faces);
    faceClearance = Math.min(faceClearance, r.lower);
    assert.equal(r.inside.length, 0, `${name}: ${r.inside.length} faces inside the solid (exact) ${when}: ${r.inside.slice(0, 8).join(", ")}`);
  };
  frame(0, true);
  const c0 = await det.readCounters();
  assert.equal(c0.intersections, 0, `${name}: the authored pose intersects`);
  const frames: Frame[] = [];
  for (let s = 0; s < STEPS; s++) {
    frame(1, true);
    const c = await det.readCounters();
    frames.push({ step: s, c });
    if (process.env.CLOTH_GATE_DEBUG) {
      const x = await solver.readPositions();
      let ylo = Infinity, yhi = -Infinity, bad = 0, minPhi = Infinity;
      for (let v = 0; v < N; v++) {
        const y = x[3 * v + 1]!; ylo = Math.min(ylo, y); yhi = Math.max(yhi, y); if (!Number.isFinite(y)) bad++;
        if (field) minPhi = Math.min(minPhi, sampleClothSolidField(field, [x[3 * v]!, y, x[3 * v + 2]!]).phi);
      }
      log(`${name} debug step ${s}: y [${ylo.toFixed(4)}, ${yhi.toFixed(4)}], non-finite ${bad}, min phi ${minPhi.toExponential(3)}, status [${OgcContactDetector.describeStatus(c.status)}], `
        + `rounds ${c.detections}, truncated ${c.truncated}, vf ${c.vfPairs} ee ${c.eePairs} dat ${c.datPairs} peakSlots ${c.framePeakPairSlots}, solid ${c.solidContacts}, envelope ${c.peakEnvelope.toExponential(2)}, cell ${c.cell.toExponential(2)}, candidates ${c.candidates}, held ${c.held} conflicts ${c.conflicts} redos ${c.redos} capped ${c.capped} uncertified ${c.uncertified} envCap ${c.envelopeCapHits}`);
    }
    if (c.solidIntersections && field && process.env.CLOTH_GATE_DEBUG) log(`${name} step ${s}: ${await solidDiagnosis(field, await solver.readPositions(), m.topology.faces, det)}`);
    if (c.intersections) {
      const x = await solver.readPositions(), cpu = cpuIntersections(x, m.topology.faces, 0.02);
      log(`${name} step ${s}: GPU counts ${c.intersections} intersecting pairs, Float64 ${cpu.count}: ${JSON.stringify(cpu.pairs.slice(0, 4))} `
        + cpu.pairs.slice(0, 2).map(([a, b]) => [a, b].map(t => [0, 1, 2].map(k => { const v = m.topology.faces[3 * t + k]!; return `${v}:(${[0, 1, 2].map(i => x[3 * v + i]!.toFixed(6)).join(",")})`; }).join(" ")).join(" | ")).join(" || "));
    }
    assert.equal(c.status & FAILING, 0, `${name}: status ${OgcContactDetector.describeStatus(c.status)} at step ${s}`);
    assert.equal(c.intersections, 0, `${name}: ${c.intersections} intersecting face pairs at step ${s}`);
    assert.equal(c.solidIntersections, 0, `${name}: ${c.solidIntersections} faces reach into the solid at step ${s}`);
    if (s % 10 === 9) {
      assert.equal(await inside(), 0, `${name}: vertices inside the solid at step ${s}`);
      exactFaces(await solver.readPositions(), `step ${s}`);
    }
  }
  const report = await solver.readStatus();
  const residual = new Map(report.slots.map(r => [r.step, r.residual]));

  // One timed batch, no readback inside it.
  await device.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  frame(TIMED_STEPS, true);
  await device.queue.onSubmittedWorkDone();
  const ms = (performance.now() - t0) / TIMED_STEPS;
  const cT = await det.readCounters();
  assert.equal(cT.intersections, 0, `${name}: intersections after the timed batch`);
  assert.equal(cT.solidIntersections, 0, `${name}: faces in the solid after the timed batch`);
  assert.equal(cT.status & FAILING, 0, `${name}: status ${OgcContactDetector.describeStatus(cT.status)} after the timed batch`);
  assert.equal(await inside(), 0, `${name}: vertices inside the solid after the timed batch`);
  exactFaces(await solver.readPositions(), "after the timed batch");
  if (field) log(`${name}: faces (exact, Float64 Bernstein on the trilinear field) keep phi >= ${faceClearance.toExponential(2)} m (a lower bound)`);

  const sum = (f: (c: OgcFrameCounters) => number) => frames.reduce((a, fr) => a + f(fr.c), 0);
  const max = (f: (c: OgcFrameCounters) => number) => frames.reduce((a, fr) => Math.max(a, f(fr.c)), 0);
  for (const fr of frames) if (fr.step % 10 === 0 || fr.step === STEPS - 1) {
    const c = fr.c;
    log(`${name} step ${fr.step}: rounds ${c.detections} (re ${c.redetections}), truncated/N ${(c.truncated / N).toFixed(3)} (max ${(c.truncatedMax / N).toFixed(3)}), `
      + `pairs/N vf ${(c.vfPairs / N).toFixed(2)} ee ${(c.eePairs / N).toFixed(2)} dat ${(c.datPairs / N).toFixed(2)}, solid ${c.solidContacts} cut ${c.solidCut} sample-facet ${c.sfPairs}, `
      + `capped ${c.capped}, uncertified ${c.uncertified}, envelope ${c.peakEnvelope.toExponential(2)}, candidates/N ${(c.candidates / N).toFixed(1)}, `
      + `held/N ${(c.held / N).toFixed(3)} conflicts ${c.conflicts} redos ${c.redos}, residual ${residual.get(fr.step)?.toExponential(2) ?? "-"}`);
  }
  log(`${name} [kc ${config.kc}, s ${config.envelopeScale}, prediction ${config.prediction ?? "inertial"}, cell ${cT.cell.toExponential(2)}, truncate per ${process.env.CLOTH_GATE_TRUNCATE ?? "iteration"}] summary (${N} vertices, ${STEPS} steps): `
    + `rounds/step ${(sum(c => c.detections) / STEPS).toFixed(2)}, round-cap steps ${sum(c => c.roundCapHits)}, redone steps ${sum(c => c.redos)}, `
    + `candidates/N per step ${(sum(c => c.candidates) / STEPS / N).toFixed(1)}, dat pairs/N per step ${(sum(c => c.datPairs) / STEPS / N).toFixed(2)}, `
    + `envelope-cap hits ${sum(c => c.envelopeCapHits)}, peak contacts/vertex ${cT.peakContacts}/${config.contactCapacity}, `
    + `peak pair slots ${cT.peakPairSlots}/${config.pairCapacity} (${(cT.peakPairSlots / N).toFixed(1)}N), peak grid ${cT.peakGrid}/${config.gridCapacity}, `
    + `max truncated/N ${(max(c => c.truncated) / N).toFixed(3)}, status [${OgcContactDetector.describeStatus(cT.status).join(", ")}], `
    + `${ms.toFixed(2)} ms/step over ${TIMED_STEPS} steps (diagnostics and intersection count included)`);
  log(`${name} profile: ${await profileRound(device, det)}`);
  solver.destroy(); det.destroy();
  // The same solver without contact, timed alike from the authored pose: the step's non-contact floor.
  const bare = await ClothGPUSolver.create(device, m, { origin_m: [0, 0, 0], dt_s: DT, gravity_m_s2: g, integrator: "bdf2", newton: NEWTON, diagnostics: true, label: `${name} bare` });
  const bareRun = (count: number) => { const enc = device.createCommandEncoder(), pass = enc.beginComputePass(); for (let s = 0; s < count; s++) bare.encodeStep(pass); pass.end(); device.queue.submit([enc.finish()]); };
  bareRun(1); await device.queue.onSubmittedWorkDone();
  const tb = performance.now(); bareRun(TIMED_STEPS); await device.queue.onSubmittedWorkDone();
  log(`${name} solver alone: ${((performance.now() - tb) / TIMED_STEPS).toFixed(2)} ms/step (no contact, no intersection count)`);
  bare.destroy();
}

// ---------------------------------------------------------------- drop

const R_C = 2e-3, RQ_MAX = 0.04;
function dropScene() {
  const sheets: ClothTopologyData[] = [];
  const W = 0.3, n = 31;
  for (let k = 0; k < 10; k++) {
    const a = (7 * k) * Math.PI / 180, warp: [number, number, number] = [Math.cos(a), 0, Math.sin(a)], weft: [number, number, number] = [-Math.sin(a), 0, Math.cos(a)];
    const y = 0.01 + 0.01 * k, c: [number, number, number] = [0.004 * k, y, -0.003 * k];
    sheets.push(clothGridData({ columns: n, rows: n, width_m: W, height_m: W, warp, weft,
      origin_m: [c[0] - 0.5 * W * (warp[0] + weft[0]), y, c[2] - 0.5 * W * (warp[2] + weft[2])] }));
  }
  return model(concat(sheets));
}

function pedestal(band: number): ClothSolidField {
  // Lattice boxes: faces on node planes, exact under the interpolant (sharp edges and corners kept).
  return buildClothSolidField({ parts: [
    { kind: "lattice-box", lo: [-0.1, -0.12, -0.1], hi: [0.1, 0, 0.1], mu: 0.3 },
    { kind: "lattice-box", lo: [-0.32, -0.16, -0.32], hi: [0.32, -0.12, 0.32], mu: 0.3 },
  ], lo: [-0.36, -0.2, -0.36], hi: [0.36, 0.22, 0.36], cell: 0.005, band });
}

/** Surface samples a gate field may hold, authored above the measured counts (pedestal 7,758; drape 13,766). */
const SOLID_SAMPLES = 16384;

function dropConfig(N: number, T: number, E: number, solidWords: number, variant: { truncation: "planar" | "isotropic"; adaptive: boolean }): OgcContactConfig {
  return {
    r: R_C, kc: KC, mu: 0.2, epsV: 1e-2, dt: DT, gravity: g,
    rqBase: variant.adaptive ? 3e-3 : 1e-2, rqMax: RQ_MAX, envelopeScale: variant.adaptive ? ENVELOPE_S : 0, truncation: variant.truncation,
    contactCapacity: 128, pairCapacity: PAIRS_PER_VERTEX * N, gridCapacity: 32 * (N + T + E), bucketCount: 1 << 18, roundCap: 4, solidCapacityWords: solidWords,
    solidSampleCapacity: SOLID_SAMPLES, ...knobs,
  };
}

test("gate: ten sheets drop onto a sharp voxel pedestal without intersecting or entering the solid", { skip: !want("drop"), todo: "C2 Newton item: on Newton-PCG (M4) this gate halts on roundCap, envelopeCap and pairOverflow (progress-lane-c1.md RESULT (28))" }, async t => withDevice(async device => {
  const m = dropScene(), N = m.topology.vertexCount, T = m.topology.faceCount, E = m.topology.edgeCount;
  const t0 = performance.now();
  // band >= r + 1.5 L_max rqMax with the trilinear bricks' L_max <= sqrt(3)
  const field = pedestal(R_C + 1.5 * Math.sqrt(3) * RQ_MAX + 0.005);
  t.diagnostic(`solid field: ${field.bricks.size} bricks, ${field.words.length} words, L_max ${field.maxLipschitz.toFixed(3)}, built in ${(performance.now() - t0).toFixed(0)} ms`);
  const variants = process.env.CLOTH_GATE_VARIANTS === "all"
    ? [{ truncation: "planar", adaptive: true }, { truncation: "isotropic", adaptive: true }, { truncation: "planar", adaptive: false }] as const
    : [{ truncation: "planar", adaptive: true }] as const;
  for (const v of variants)
    await runGate(device, `drop/${v.truncation}/${v.adaptive ? "adaptive" : "fixed"}`, m, dropConfig(N, T, E, field.words.length, v), field, s => t.diagnostic(s));
}));

// ---------------------------------------------------------------- drape

test("gate: a sheet drapes over a smooth sphere onto the floor without intersecting or entering the solid", { skip: !want("drape"), todo: "C2 Newton item: on Newton-PCG (M4) this gate halts on roundCap, envelopeCap and pairOverflow (progress-lane-c1.md RESULT (28))" }, async t => withDevice(async device => {
  const n = 61, W = 0.6;
  const m = model(clothGridData({ columns: n, rows: n, width_m: W, height_m: W, origin_m: [-0.5 * W, 0.1, -0.5 * W] }));
  const N = m.topology.vertexCount, T = m.topology.faceCount, E = m.topology.edgeCount;
  const t0 = performance.now(), R = 0.1, c: V3 = [0, -0.02, 0];
  const field = buildClothSolidField({ parts: [
    { kind: "sdf", sdf: p => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) - R, lo: [-R, c[1] - R, -R], hi: [R, c[1] + R, R], mu: 0.3 },
    { kind: "lattice-box", lo: [-0.32, -0.16, -0.32], hi: [0.32, -0.12, 0.32], mu: 0.3 },
  ], lo: [-0.36, -0.2, -0.36], hi: [0.36, 0.22, 0.36], cell: 0.005, band: R_C + 1.5 * Math.sqrt(3) * RQ_MAX + 0.005 });
  t.diagnostic(`solid field: ${field.bricks.size} bricks, ${field.words.length} words, L_max ${field.maxLipschitz.toFixed(3)}, built in ${(performance.now() - t0).toFixed(0)} ms`);
  await runGate(device, "drape", m, dropConfig(N, T, E, field.words.length, { truncation: "planar", adaptive: true }), field, s => t.diagnostic(s));
}));

// ---------------------------------------------------------------- twist

test("gate: a twisted sheet (OGC Fig. 12) stays free of intersections", { skip: !want("twist"), todo: "C2 Newton item: on Newton-PCG (M4) this gate halts on roundCap, envelopeCap and pairOverflow (progress-lane-c1.md RESULT (28))" }, async t => withDevice(async device => {
  const n = 100, W = 0.5;
  const data = clothGridData({ columns: n, rows: n, width_m: W, height_m: W, origin_m: [-0.5 * W, 0, -0.5 * W],
    group: i => i === 0 ? 1 : i === n - 1 ? 2 : 0 });
  // Each edge turns a quarter turn per second (the pins' speed stays below the
  // envelope cap's 0.45 rqMax per step); longer runs (CLOTH_GATE_STEPS) twist further.
  const TWIST = 0.5 * Math.PI * STEPS * DT;
  const turn = { start_s: 0, end_s: STEPS * DT, easing: "cosine" as const };
  const scripts: ClothAttachmentScript[] = [
    { kind: "rotate", group: 1, vector: [1, 0, 0], angle_rad: TWIST, origin_m: [0, 0, 0], ...turn },
    { kind: "rotate", group: 2, vector: [1, 0, 0], angle_rad: -TWIST, origin_m: [0, 0, 0], ...turn },
    { kind: "translate", group: 1, vector: [0.05, 0, 0], ...turn },
    { kind: "translate", group: 2, vector: [-0.05, 0, 0], ...turn },
  ];
  const m = model(data, scripts), N = m.topology.vertexCount, T = m.topology.faceCount, E = m.topology.edgeCount;
  const config: OgcContactConfig = {
    r: 1.5e-3, kc: KC, mu: 0.2, epsV: 1e-2, dt: DT, gravity: g, rqBase: 3e-3, rqMax: 0.04, envelopeScale: ENVELOPE_S,
    contactCapacity: 128, pairCapacity: PAIRS_PER_VERTEX * N, gridCapacity: 32 * (N + T + E), bucketCount: 1 << 18, roundCap: 4,
    ...knobs,
  };
  await runGate(device, "twist", m, config, undefined, s => t.diagnostic(s));
}));

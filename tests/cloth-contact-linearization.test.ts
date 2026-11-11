/**
 * Lane L1 (plan cloth-m1-max-locality-2026-11-11, experiment 1): the compiled contact linearization
 * (newton.contactLinearization) is the matrix-free contact product.
 *
 *   switch    'off' is the module as it was (no kernel, no layout change); unknown values fail; the pool sits before
 *             the control block
 *   operator  GPU: two sheets, one on the other, over a static box and a moving, sliding plate, with friction. At one
 *             fixed state the same pcgStep dispatch 0 (w = P (H u), u = M^-1 r_0) runs three times: through the
 *             compiled records, with every row sent back to the matrix-free product (the header's fallback mark), and
 *             with every row's contact emptied. The first two must agree per vertex to f32 rounding of contact's
 *             share (the second less the third); a pool too small for the demand must agree as well, part compiled
 *             and part matrix-free.
 *   forms     the trust region's model (nwModel: s.H s per patch, the same stack under trustRegion 'hybrid') and the
 *             one-patch solve (pcgSolve: a 45-vertex sheet across the off-lattice edge of a turned box, which holds
 *             hull-facet entries) through the records and through the matrix-free product.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { OgcContactDetector, type OgcBodyPose, type OgcContactConfig } from "../lib/cloth/contact/ogc-detector";
import { OGC_STATUS } from "../lib/cloth/contact/ogc-wgsl";
import { buildClothSolidField } from "../lib/cloth/contact/solid-field";
import { clothGridData } from "../lib/cloth/grid";
import { prepareClothTopology, validateClothDocument } from "../lib/cloth/topology";
import type { ClothDocument, ClothMaterial, ClothNewtonSettings, ClothTopologyData } from "../lib/cloth/types";
import { buildClothSolverModel } from "../lib/cloth/oracle";
import { ClothGPUSolver, CLOTH_PARAM_NEWTON, packClothGPUTables } from "../lib/cloth/gpu-solver";
import { CLOTH_BINDING, CLOTH_SEGMENT_ALIGN, CLOTH_STATE } from "../lib/cloth/gpu-solver-wgsl";
import { CLOTH_NEWTON_SEGMENTS, clothNewtonWGSL } from "../lib/cloth/gpu-newton-wgsl";
import { bridson2002RotatingSphere } from "../lib/cloth/scenes/bridson-2002";
import { bridson2002SceneDescription } from "../lib/cloth/scenes/bridson-2002/scene-description";
import { boxSdf, type V3 } from "./cloth-contact-oracle";

const gpu = !!process.env.WEBGPU_NODE_MODULE;
const cotton: ClothMaterial = { id: "cotton", arealDensity_kg_m2: 0.2, stretchWarp_N_m: 400, stretchWeft_N_m: 250, shear_N_m: 40, poisson: 0.3,
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0.002, thickness_m: 5e-4, strainLimit: 0.02 };
const NEWTON: ClothNewtonSettings = { iterations: 8, pcgIterations: 100, pcgTolerance: 1e-4, residualTolerance: 1e-5 };
const DT = 1 / 60, R_C = 2e-3, RQ_MAX = 0.04;

function concat(sheets: ClothTopologyData[]): ClothTopologyData {
  const rest: number[] = [], triangles: number[] = [], uv: number[] = [], groups: number[] = [];
  for (const s of sheets) {
    const base = rest.length / 3;
    rest.push(...s.restPositions_m); triangles.push(...s.triangles.map(i => i + base)); uv.push(...(s.uv ?? [])); groups.push(...(s.groups ?? []));
  }
  return { restPositions_m: rest, triangles, uv, groups };
}
/** Two sheets of different lattices, the upper 0.8 r over the lower, both overhanging the two solids' sharp edges. */
function twoSheets() {
  const lower = clothGridData({ columns: 27, rows: 17, width_m: 0.26, height_m: 0.16, origin_m: [-0.13, 1.5 * R_C, -0.08] });
  const upper = clothGridData({ columns: 24, rows: 15, width_m: 0.25, height_m: 0.15, origin_m: [-0.125, 2.3 * R_C, -0.075] });
  return buildClothSolverModel(prepareClothTopology(concat([lower, upper]), [cotton]), [cotton], []);
}

test("newton.contactLinearization: 'off' is the module and the layout as they were; the pool sits before the control block; unknown values fail", () => {
  const m = twoSheets(), N = m.topology.vertexCount;
  const base = { dt_s: DT, gravity_m_s2: [0, -9.81, 0] as const, integrator: "bdf2" as const, contact: {} };
  const off = packClothGPUTables(m, [0, 0, 0], base), named = packClothGPUTables(m, [0, 0, 0], { ...base, newton: { contactLinearization: "off" } });
  assert.equal(off.pcg.contactLin, undefined);
  assert.deepEqual(named, off);
  // Without contact the switch has nothing to compile.
  assert.deepEqual(packClothGPUTables(m, [0, 0, 0], { ...base, contact: undefined, newton: { contactLinearization: "blocks" } }), packClothGPUTables(m, [0, 0, 0], { ...base, contact: undefined }));
  for (const mode of ["factorized", "blocks"] as const) {
    const on = packClothGPUTables(m, [0, 0, 0], { ...base, newton: { contactLinearization: mode, contactLinearizationRecords: 7 } }), L = on.pcg.contactLin!;
    assert.equal(L.mode, mode);
    assert.equal(L.records, 7 * N);
    // Headers, patch totals and the pool follow the PCG scalars; the control block is still the last CLOTH_SEGMENT_ALIGN records.
    assert.ok(L.head >= off.newtonRecords - CLOTH_SEGMENT_ALIGN && L.patch >= L.head + N && L.pool >= L.patch + on.pcg.patches);
    assert.ok(on.newtonRecords - CLOTH_SEGMENT_ALIGN >= L.pool + L.records);
    assert.deepEqual({ ...on, params: 0, topo: 0, newtonRecords: 0, pcg: { ...on.pcg, contactLin: undefined } }, { ...off, params: 0, topo: 0, newtonRecords: 0, pcg: { ...off.pcg, contactLin: undefined } });
    const wgslOff = clothNewtonWGSL(CLOTH_BINDING.newton, 64, true, off.pcg), wgslOn = clothNewtonWGSL(CLOTH_BINDING.newton, 64, true, on.pcg);
    assert.doesNotMatch(wgslOff, /nwLin|LIN_/);
    assert.match(wgslOn, /fn nwLinCount\(/); assert.match(wgslOn, /fn nwLinBuild\(/);
    // The three products (pcgStep, pcgSolve, nwModel) go through the records; the matrix-free product remains only as the fallback.
    assert.equal(wgslOn.split("nwLinHv(v)").length - 1, 3);
    assert.equal(wgslOn.split("ogcVertexContactHvLanes(v, 0u, 1u)").length - 1, 1);
    assert.equal(wgslOff.split("ogcVertexContactHvLanes(v, 0u, 1u)").length - 1, 3);
    // No contact: nothing of it, whatever the layout says.
    assert.equal(clothNewtonWGSL(CLOTH_BINDING.newton, 64, false, on.pcg), clothNewtonWGSL(CLOTH_BINDING.newton, 64, false, off.pcg));
  }
  assert.throws(() => packClothGPUTables(m, [0, 0, 0], { ...base, newton: { contactLinearization: "dense" as "off" } }), /Unknown cloth Newton contactLinearization dense/);
  for (const bad of [0, 1.5, -3, NaN])
    assert.throws(() => packClothGPUTables(m, [0, 0, 0], { ...base, newton: { contactLinearization: "blocks", contactLinearizationRecords: bad } }), /contactLinearizationRecords/);
  const document: ClothDocument = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 8 })).cloth![0]!, newton = document.solver!.newton!;
  const withNewton = (n: object) => validateClothDocument({ ...document, solver: { ...document.solver, newton: { ...newton, ...n } } });
  for (const ok of ["off", "factorized", "blocks"]) assert.deepEqual(withNewton({ contactLinearization: ok }), []);
  assert.deepEqual(withNewton({ contactLinearization: "blocks", contactLinearizationRecords: 64 }), []);
  assert.ok(withNewton({ contactLinearization: "dense" }).some(e => e.includes("contactLinearization dense")));
  assert.ok(withNewton({ contactLinearizationRecords: 0 }).some(e => e.includes("contactLinearizationRecords")));
});

async function withDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has("subgroups") ? ["subgroups" as GPUFeatureName] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); }
  finally { device.destroy(); }
}
async function readBuffer(device: GPUDevice, buffer: GPUBuffer, offset: number, bytes: number): Promise<ArrayBuffer> {
  const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(buffer, offset, read, 0, bytes); device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = read.getMappedRange().slice(0); read.destroy();
  return out;
}
const quat = (axis: V3, angle: number): [number, number, number, number] => { const s = Math.sin(angle / 2); return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)]; };

type Internals = { newtonWork: GPUBuffer; run(pass: GPUComputePassEncoder, name: string, workgroups: number, record?: number): void; threads(n: number): number };
const LIN_NONE = 0x7f000000;
type Mode = "factorized" | "blocks";
type Kind = "solid" | "body" | "vt" | "ee" | "sample" | "facet";
interface Scenario {
  label: string; model: ReturnType<typeof twoSheets>; floor: ReturnType<typeof buildClothSolidField>; plate?: ReturnType<typeof buildClothSolidField>;
  /** Real steps before the fixed state; 0: the step's begin and its mandatory detection only (the entries at the authored pose). */
  steps: number; config?: Partial<OgcContactConfig>; newton?: Partial<ClothNewtonSettings>; kinds: readonly Kind[];
}
const band = R_C + 1.5 * Math.sqrt(3) * RQ_MAX + 0.005;
const box = (lo: V3, hi: V3, flo: V3, fhi: V3) => buildClothSolidField({ parts: [{ kind: "sdf", sdf: boxSdf(lo, hi), lo, hi, mu: 0.5 }], lo: flo, hi: fhi, cell: 0.01, band });
const STILL = 8;
const poseAt = (step: number): OgcBodyPose => { const tm = Math.max(0, step - STILL) * DT; return { position: [0.06, -0.01, 0.05 * tm], orientation: quat([0, 1, 0], 0.3 * tm) }; };
/** The static box under the sheets' left half (top at y = 0), the plate under their right half: still, then sliding in z and turning. */
function stack(label: string, newton?: Partial<ClothNewtonSettings>): Scenario {
  const half: V3 = [0.04, 0.01, 0.06];
  return { label, model: twoSheets(), steps: 14, newton, kinds: ["solid", "body", "vt", "ee", "sample"],
    floor: box([-0.10, -0.02, -0.06], [-0.02, 0, 0.06], [-0.22, -0.14, -0.18], [0.10, 0.12, 0.18]),
    plate: box([-half[0], -half[1], -half[2]], half, [-0.16, -0.13, -0.18], [0.16, 0.13, 0.18]) };
}
/**
 * Hull facets, in one PCG patch (tests/cloth-contact-gpu.test.ts, "GPU hull facets"): a box turned 30 degrees about y, so
 * the field (cell 5 mm) rounds its edges along y in and the samples' hull facets span them; a 9 x 5 sheet across the
 * top-front edge's bisector, 1.2 mm off it (r = 2 mm), its rows 1.25 mm apart across the edge.
 */
function facetSheet(): Scenario {
  const th = Math.PI / 6, C = Math.cos(th), Sn = Math.sin(th), ctr: V3 = [0.05, 0.04, 0.02], hb: V3 = [0.03, 0.025, 0.015];
  const local = boxSdf([-hb[0], -hb[1], -hb[2]], hb);
  const sdf = (q: readonly number[]) => { const d = [q[0]! - ctr[0], q[1]! - ctr[1], q[2]! - ctr[2]]; return local([C * d[0]! - Sn * d[2]!, d[1]!, Sn * d[0]! + C * d[2]!]); };
  const floor = buildClothSolidField({ parts: [{ kind: "sdf", sdf, lo: [0.0, 0.01, -0.02], hi: [0.09, 0.07, 0.06], mu: 0.3 }], lo: [-0.02, -0.02, -0.03], hi: [0.18, 0.12, 0.1], cell: 0.005, band: 0.1 });
  const nb: V3 = [(C + Sn) / Math.SQRT2, 0, (C - Sn) / Math.SQRT2], tb: V3 = [(C - Sn) / Math.SQRT2, 0, (-Sn - C) / Math.SQRT2];
  const e: V3 = [ctr[0] + C * hb[0] + Sn * hb[2], ctr[1] - 0.01, ctr[2] - Sn * hb[0] + C * hb[2]];
  const rest: number[] = [], triangles: number[] = [];
  // Flat (the bending model's rest shape), 1.2 mm off the edge.
  for (let j = 0; j < 5; j++) for (let i = 0; i < 9; i++) {
    const off = 1.2e-3, sAcross = (j - 2) * 1.25e-3;
    for (let a = 0; a < 3; a++) rest.push(e[a]! + (a === 1 ? 0.0025 * i : 0) + off * nb[a]! + sAcross * tb[a]!);
  }
  for (let j = 0; j < 4; j++) for (let i = 0; i < 8; i++) { const a = 9 * j + i; triangles.push(a, a + 1, a + 10, a, a + 10, a + 9); }
  const model = buildClothSolverModel(prepareClothTopology({ restPositions_m: rest, triangles }, [cotton]), [cotton], []), N = model.topology.vertexCount;
  return { label: "facet sheet", model, floor, steps: 0, kinds: ["facet"],
    config: { rqBase: 0.004, pairCapacity: 4096 * N, gridCapacity: 256 * (N + model.topology.faceCount + model.topology.edgeCount), solidCapacityWords: floor.words.length + 64 } };
}

async function operatorCheck(device: GPUDevice, sc: Scenario, mode: Mode, recordsPerVertex: number, log: (s: string) => void) {
  const m = sc.model, t = m.topology, N = t.vertexCount, T = t.faceCount, E = t.edgeCount;
  const triangles = new Uint32Array(4 * T);
  for (let f = 0; f < T; f++) triangles.set(t.faces.subarray(3 * f, 3 * f + 3), 4 * f);
  const rest = new Float32Array(4 * N);
  for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) rest[4 * i + k] = t.data.restPositions_m[3 * i + k]!;
  const config: OgcContactConfig = { r: R_C, kc: 100, mu: 0.3, epsV: 1e-2, dt: DT, gravity: [0, -9.81, 0], rqBase: 3e-3, rqMax: RQ_MAX, envelopeScale: 2.5,
    contactCapacity: 128, pairCapacity: 840 * N, gridCapacity: 32 * (N + T + E), bucketCount: 1 << 16, roundCap: 8, prediction: "guess", newton: true, planeRecordCapacity: 64 * N,
    solidCapacityWords: sc.floor.words.length, solidSampleCapacity: 8192, ...(sc.plate ? { body: { capacityWords: sc.plate.words.length, sampleCapacity: 8192, poseCapacity: 2 } } : {}), ...sc.config };
  const det = await OgcContactDetector.create(device, { vertexCount: N, triangles, edges: t.edges }, rest, config);
  const solver = await ClothGPUSolver.create(device, m, { origin_m: [0, 0, 0], dt_s: DT, gravity_m_s2: [0, -9.81, 0], integrator: "bdf2",
    newton: { ...NEWTON, ...sc.newton, contactLinearization: mode, contactLinearizationRecords: recordsPerVertex }, label: `lin ${mode}`,
    contact: { ...det.solverContact(CLOTH_BINDING.contact, true), alpha: det.alpha, state: det.contactState },
    hooks: { afterFrame: det.stepHooks.afterFrame, filter: det.stepHooks.filter, afterNewtonStep: det.stepHooks.afterNewtonStep, stepEnd: det.stepHooks.stepEnd } });
  const stride = solver.tables.stride, L = solver.tables.pcg.contactLin!, patches = solver.tables.pcg.patches;
  const S = solver as unknown as Internals, W = S.threads(N), control = 16 * (solver.tables.newtonRecords - CLOTH_SEGMENT_ALIGN);
  det.bind({ positions: solver.positions, velocities: solver.velocities, stepStart: { buffer: solver.state, offset: 16 * CLOTH_STATE.xt * stride, size: 16 * stride }, direction: solver.newtonDirection });
  det.setSolidField(sc.floor); if (sc.plate) det.setBodyField(sc.plate);
  const FAILING = OGC_STATUS.contactOverflow | OGC_STATUS.gridOverflow | OGC_STATUS.pairOverflow | OGC_STATUS.nonFinite | OGC_STATUS.solidPenetration
    | OGC_STATUS.bodySqueeze | OGC_STATUS.heldConflict | OGC_STATUS.bodyPose;
  for (let step = 0; step < sc.steps; step++) {
    if (sc.plate) det.setBodyPoses([poseAt(step), poseAt(step + 1)]);
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    det.encodeFrameBegin(pass); solver.encodeStep(pass); pass.end(); device.queue.submit([enc.finish()]);
    const c = await det.readCounters();
    assert.equal(c.status & FAILING, 0, `step ${step}: status ${OgcContactDetector.describeStatus(c.status)}`);
  }
  if (!sc.steps) {
    // A step's begin and its mandatory detection (encodeStep to nwBeginStep): the entries of the authored pose.
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    det.encodeFrameBegin(pass);
    S.run(pass, "restartScan", W); S.run(pass, "beginStep", 1); S.run(pass, "frameElements", S.threads(Math.max(E, T))); S.run(pass, "frame", W);
    det.stepHooks.afterFrame!(pass); S.run(pass, "nwBeginStep", 1);
    pass.end(); device.queue.submit([enc.finish()]);
    const c = await det.readCounters();
    assert.equal(c.status & FAILING, 0, `status ${OgcContactDetector.describeStatus(c.status)}`);
  }
  // The entries the fixed state holds, by kind (the contact buffer's header: N, capacity, counts, entries, in vec4 units).
  const words = new Uint32Array(await readBuffer(device, det.solverBuffer, 0, det.solverBuffer.size)), fl = new Float32Array(words.buffer);
  const cap = words[1]!, countsAt = 4 * words[2]!, entriesAt = 4 * words[3]!;
  const kinds: Record<Kind, number> = { solid: 0, body: 0, vt: 0, ee: 0, sample: 0, facet: 0 }, friction: Record<Kind, number> = { solid: 0, body: 0, vt: 0, ee: 0, sample: 0, facet: 0 };
  let entries = 0, rowsWith = 0;
  for (let v = 0; v < N; v++) {
    const n = Math.min(words[countsAt + v]!, cap);
    entries += n; if (n) rowsWith++;
    for (let i = 0; i < n; i++) {
      const b = entriesAt + 4 * (v * cap + i), y = words[b + 1]!, w = words[b + 3]!;
      const kind: Kind = y === 0x70000000 ? "solid" : y === 0x70000001 ? "body" : (y & 0x80000000) !== 0 ? "ee" : w === 1 ? "sample" : w === 2 ? "facet" : "vt";
      kinds[kind]++; if (fl[b + 2]! > 0) friction[kind]++;
    }
  }
  log(`${sc.label}, ${mode} (pool ${recordsPerVertex} records a vertex): ${N} vertices, ${patches} patches, ${entries} entries on ${rowsWith} rows: ${JSON.stringify(kinds)}; with lagged friction ${JSON.stringify(friction)}`);
  for (const kind of sc.kinds) {
    assert.ok(kinds[kind] > 0, `the state holds ${kind} entries`);
    assert.ok(friction[kind] > 0, `${kind} entries carry friction`);
  }

  // One encoded sequence at the fixed state: Newton re-armed at its first iteration (no active set: P = I), so every
  // run assembles the same g, H, M^-1 and r_0; `build` compiles the records, else `headers` replaces them.
  const sequence = async (build: boolean, headers: Uint32Array<ArrayBuffer> | undefined, then: (pass: GPUComputePassEncoder) => void) => {
    device.queue.writeBuffer(S.newtonWork, control, new Float32Array(4));
    if (headers) device.queue.writeBuffer(S.newtonWork, 16 * L.head, headers);
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    S.run(pass, "nwFaces", S.threads(T)); S.run(pass, "nwGradient", W);
    if (build) { S.run(pass, "nwLinCount", patches); S.run(pass, "nwLinBuild", patches); }
    then(pass);
    pass.end(); device.queue.submit([enc.finish()]);
  };
  // The product: pcgStep dispatch 0 (w0 = P H M^-1 r_0), or with one patch pcgSolve capped at 3 iterations (d).
  const one = patches === 1, out = one ? CLOTH_NEWTON_SEGMENTS.d : CLOTH_NEWTON_SEGMENTS.w0;
  const product = async (build: boolean, headers?: Uint32Array<ArrayBuffer>): Promise<Float32Array> => {
    await sequence(build, headers, pass => { if (one) S.run(pass, "pcgSolve", 1, 3); else S.run(pass, "pcgStep", patches, 0); });
    return new Float32Array(await readBuffer(device, S.newtonWork, 16 * out * stride, 16 * N));
  };
  const compiled = await product(true);
  const stats = (await solver.readContactLinearization())!;
  const head = new Uint32Array(await readBuffer(device, S.newtonWork, 16 * L.head, 16 * N));
  const pool = new Uint32Array(await readBuffer(device, S.newtonWork, 16 * L.pool, 16 * L.records));
  const allFree = head.slice(), none = head.slice();
  let compiledRows = 0, globalColumns = 0, localColumns = 0;
  for (let v = 0; v < N; v++) {
    if (head[4 * v + 2] === 0) continue;
    if (head[4 * v] !== LIN_NONE) {
      compiledRows++;
      const first = head[4 * v]! - L.pool;
      assert.ok(first >= 0 && first + head[4 * v + 2]! <= L.records, `row ${v}'s records lie in the pool`);
    }
    // Back to the matrix-free product: the mark, with the entry count nwLinCount found.
    allFree[4 * v] = LIN_NONE; allFree[4 * v + 1] = Math.min(words[countsAt + v]!, cap);
    none[4 * v] = 0; none[4 * v + 1] = 0;
  }
  if (mode === "blocks") for (let v = 0; v < N; v++) {
    if (head[4 * v + 2] === 0 || head[4 * v] === LIN_NONE) continue;
    for (let i = 0; i < head[4 * v + 1]!; i++) { const c = pool[4 * (head[4 * v]! - L.pool + 2 * i) + 3]!; if (c & 0x80000000) globalColumns++; else localColumns++; }
  }
  const free = await product(false, allFree), empty = await product(false, none);
  // The f32 floor: rounding of contact's share and of the row's whole product it is added into. Three iterations of
  // the one-patch solve carry three products and two reductions over every row into each row: its bound is looser.
  const bound = one ? 4096 : 64;
  let rows = 0, worstRel = 0, worstAbs = 0, maxContact = 0, sumContact2 = 0, sumErr2 = 0, bad = 0;
  for (let v = 0; v < N; v++) {
    let err = 0, contact = 0, whole = 0;
    for (let c = 0; c < 3; c++) {
      assert.ok(Number.isFinite(compiled[4 * v + c]!) && Number.isFinite(free[4 * v + c]!), `row ${v} is finite`);
      err += (compiled[4 * v + c]! - free[4 * v + c]!) ** 2; contact += (free[4 * v + c]! - empty[4 * v + c]!) ** 2; whole += free[4 * v + c]! ** 2;
    }
    err = Math.sqrt(err); contact = Math.sqrt(contact); whole = Math.sqrt(whole);
    if (head[4 * v + 2] === 0 && !one) { assert.equal(err, 0, `row ${v} has no contact entry: the products are the same bits`); continue; }
    if (head[4 * v + 2] !== 0) rows++;
    maxContact = Math.max(maxContact, contact); sumContact2 += contact ** 2; sumErr2 += err ** 2; worstAbs = Math.max(worstAbs, err);
    if (contact > 0) worstRel = Math.max(worstRel, err / (contact + whole));
    if (!(err <= bound * 1.1920929e-7 * (contact + whole))) bad++;
  }
  log(`  ${stats.items} items in ${stats.usedRecords} records (${stats.usedBytes} B) of ${stats.poolRecords}, demand ${stats.demandRecords}; rows compiled ${compiledRows}, left matrix-free ${stats.fallbackRows}`
    + `${mode === "blocks" ? `; columns ${localColumns} in the row's patch, ${globalColumns} global` : ""}`);
  log(`  ${one ? "pcgSolve (3 iterations), d" : "pcgStep 0, w"}: compiled vs matrix-free over ${rows} contact rows: max |dw| ${worstAbs.toExponential(2)}, rms ${Math.sqrt(sumErr2 / Math.max(rows, 1)).toExponential(2)}; contact's share max ${maxContact.toExponential(2)}, rms ${Math.sqrt(sumContact2 / Math.max(rows, 1)).toExponential(2)}; `
    + `worst |dw| / (|contact| + |w|) ${worstRel.toExponential(2)} (eps32 1.19e-7); rows past ${bound} eps ${bad}`);
  assert.ok(rows >= (one ? 5 : N / 4) && maxContact > 0, `contact acts on ${rows} rows`);
  assert.equal(stats.rows, rows);
  assert.equal(stats.fallbackRows + compiledRows, rows);
  if (recordsPerVertex === 1) assert.ok(stats.fallbackRows > 0 && compiledRows > 0, `a short pool compiles some rows (${compiledRows}) and leaves others matrix-free (${stats.fallbackRows})`);
  else assert.equal(stats.fallbackRows, 0, "the authored pool holds this scene");
  if (mode === "blocks" && !one) assert.ok(globalColumns > 0 && localColumns > 0, "partners in the row's patch and outside it");
  assert.equal(bad, 0, `every row within ${bound} eps32 of (|contact share| + |row product|); worst ${worstRel.toExponential(2)}`);

  if (sc.newton?.trustRegion === "hybrid") {
    // The trust region's model at a step s (nwMove's trial 0: the direction d of eight PCG iterations from here, run
    // once through the records): s.H s and contact's share of it, per patch into PCG partial array 2. Then nwModel
    // alone per variant, on that one d, with the trust record set so the kernel takes the iteration as a trust-region one.
    const partials = new Uint32Array(solver.tables.params.buffer, solver.tables.params.byteOffset)[CLOTH_PARAM_NEWTON + 12]!;
    await sequence(true, undefined, pass => { for (let i = 0; i <= 8; i++) S.run(pass, "pcgStep", patches, i); });
    const model = async (headers?: Uint32Array<ArrayBuffer>): Promise<Float32Array> => {
      device.queue.writeBuffer(S.newtonWork, control, new Float32Array(4));
      device.queue.writeBuffer(S.newtonWork, control + 16 * 3, new Float32Array([1, 1e3, 0, 0]));
      device.queue.writeBuffer(S.newtonWork, 16 * L.head, headers ?? head);
      const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
      S.run(pass, "nwModel", patches);
      pass.end(); device.queue.submit([enc.finish()]);
      return new Float32Array(await readBuffer(device, S.newtonWork, 16 * (partials + 2 * patches), 16 * patches));
    };
    const a = await model(), b = await model(allFree), c = await model(none);
    let scale = 0, worst = 0, share = 0;
    for (let q = 0; q < patches; q++) { scale = Math.max(scale, Math.abs(b[4 * q]!)); share = Math.max(share, Math.abs(b[4 * q + 1]!)); worst = Math.max(worst, Math.abs(a[4 * q]! - b[4 * q]!), Math.abs(a[4 * q + 1]! - b[4 * q + 1]!)); }
    log(`  nwModel: s.H s per patch compiled ${Array.from({ length: patches }, (_, q) => a[4 * q]!.toExponential(6)).join(" ")} vs matrix-free ${Array.from({ length: patches }, (_, q) => b[4 * q]!.toExponential(6)).join(" ")}; `
      + `contact's share ${Array.from({ length: patches }, (_, q) => a[4 * q + 1]!.toExponential(6)).join(" ")} vs ${Array.from({ length: patches }, (_, q) => b[4 * q + 1]!.toExponential(6)).join(" ")}; worst difference ${worst.toExponential(2)} (${(worst / scale).toExponential(2)} of the largest s.H s)`);
    assert.ok(share > 0 && scale > 0, "the model's step has a contact share");
    for (let q = 0; q < patches; q++) assert.equal(c[4 * q + 1], 0, "no records, no contact share");
    assert.ok(worst <= 1e-5 * scale, `the model through the records within 1e-5 of the largest patch's s.H s: ${worst} of ${scale}`);
  }
  solver.destroy(); det.destroy();
}

test("GPU compiled contact linearization: the PCG product through the records is the matrix-free product to f32 rounding (solid, moving solid, VT, EE and sample entries, friction on)", { skip: !gpu }, async t => {
  await withDevice(async device => {
    // This stack is dense in pair entries (about 5 a vertex, 8 on the rows that have any): a pool authored for it.
    await operatorCheck(device, stack("stack"), "factorized", 48, s => t.diagnostic(s));
    await operatorCheck(device, stack("stack"), "blocks", 48, s => t.diagnostic(s));
    // A pool of one record a vertex: the first patches' rows compile, the rest keep the matrix-free product.
    await operatorCheck(device, stack("stack"), "factorized", 1, s => t.diagnostic(s));
  });
});

test("GPU compiled contact linearization: the trust region's model (nwModel) and the one-patch solve (pcgSolve, hull-facet entries) through the records", { skip: !gpu }, async t => {
  await withDevice(async device => {
    await operatorCheck(device, stack("stack, hybrid", { trustRegion: "hybrid" }), "factorized", 48, s => t.diagnostic(s));
    await operatorCheck(device, stack("stack, hybrid", { trustRegion: "hybrid" }), "blocks", 48, s => t.diagnostic(s));
    await operatorCheck(device, facetSheet(), "factorized", 48, s => t.diagnostic(s));
    await operatorCheck(device, facetSheet(), "blocks", 48, s => t.diagnostic(s));
  });
});

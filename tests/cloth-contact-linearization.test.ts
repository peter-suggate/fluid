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
 *   pieces    (experiment 4) the compiled modes' indirect overflow and staged partners (their own, always on) against
 *             their baselines through the A/B knob (ClothGPUSolverOptions.contactLinearizationAB: the grid overflow,
 *             global partners); staged partners with a table too small for the patch's (4 slots), so the rest stay
 *             global; and the indirect overflow's coverage with the pool's end forced into the first, a middle and the
 *             last patch, and past the last (no workgroup): pcgStepOverflow alone, over a poisoned w, writes exactly
 *             the marked patches' rows, from the first marked patch on.
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
import { ClothGPUSolver, CLOTH_CONTACT_LIN_AB_DEFAULTS, CLOTH_NEWTON_DEFAULTS, CLOTH_PARAM_NEWTON, packClothGPUTables, type ClothContactLinAB } from "../lib/cloth/gpu-solver";
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

test("newton.contactLinearization: 'off' is the module and the layout as they were; the default is CLOTH_NEWTON_DEFAULTS.contactLinearization; the pool sits before the control block; unknown values fail", () => {
  const m = twoSheets(), N = m.topology.vertexCount;
  const base = { dt_s: DT, gravity_m_s2: [0, -9.81, 0] as const, integrator: "bdf2" as const, contact: {} };
  const off = packClothGPUTables(m, [0, 0, 0], { ...base, newton: { contactLinearization: "off" } });
  assert.equal(off.pcg.contactLin, undefined);
  // The default is CLOTH_NEWTON_DEFAULTS' mode at its pool.
  const { contactLinearization, contactLinearizationRecords } = CLOTH_NEWTON_DEFAULTS;
  assert.deepEqual(packClothGPUTables(m, [0, 0, 0], base), packClothGPUTables(m, [0, 0, 0], { ...base, newton: { contactLinearization, contactLinearizationRecords } }));
  assert.equal(packClothGPUTables(m, [0, 0, 0], base).pcg.contactLin?.mode, contactLinearization === "off" ? undefined : contactLinearization);
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
    // The three products (pcgStep, pcgSolve, nwModel) go through the records. pcgStep's has no matrix-free branch: the
    // patches holding a row the pool could not hold are pcgStepOverflow's; pcgSolve and nwModel keep a per-row fallback.
    assert.equal(wgslOn.split("nwLinRecordsHv(v)").length - 1, 1);
    assert.equal(wgslOn.split("nwLinHv(v)").length - 1, 2);
    assert.equal(wgslOn.split("ogcVertexContactHvLanes(v, 0u, 1u)").length - 1, 2);
    assert.match(wgslOn, /fn pcgStepOverflow\(/); assert.doesNotMatch(wgslOff, /pcgStepOverflow|pcgLinOverflow/);
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

  // Experiment 4's pieces, the compiled modes' own (the indirect overflow, staged partners at 256 slots); the A/B
  // knob's baseline (the grid overflow, global partners) is the layout and the module without them; each is validated.
  const fact = { ...base, newton: { contactLinearization: "factorized" as const, contactLinearizationRecords: 7 } };
  const own = packClothGPUTables(m, [0, 0, 0], fact), A = own.pcg.contactLin!;
  assert.deepEqual(own, packClothGPUTables(m, [0, 0, 0], { ...fact, contactLinearizationAB: { ...CLOTH_CONTACT_LIN_AB_DEFAULTS } }));
  assert.deepEqual(CLOTH_CONTACT_LIN_AB_DEFAULTS, { overflow: "indirect", partners: "staged", stageSlots: 256 });
  // The slot keys (slots / 4 records a patch) between the patch totals and the pool; nothing else moves.
  assert.equal(A.overflow, "indirect");
  assert.deepEqual(A.staged, { keys: A.patch + own.pcg.patches, slots: 256 });
  assert.ok(A.pool >= A.staged!.keys + 64 * own.pcg.patches && own.newtonRecords - CLOTH_SEGMENT_ALIGN >= A.pool + A.records);
  const baseline = packClothGPUTables(m, [0, 0, 0], { ...fact, contactLinearizationAB: { overflow: "grid", partners: "global" } }), B = baseline.pcg.contactLin!;
  assert.deepEqual({ overflow: B.overflow, staged: B.staged }, { overflow: "grid", staged: undefined });
  assert.equal(B.pool, packClothGPUTables(m, [0, 0, 0], { ...fact, contactLinearizationAB: { partners: "global" } }).pcg.contactLin!.pool);
  assert.deepEqual({ ...own, params: 0, topo: 0, newtonRecords: 0, pcg: { ...own.pcg, contactLin: undefined } }, { ...baseline, params: 0, topo: 0, newtonRecords: 0, pcg: { ...baseline.pcg, contactLin: undefined } });
  const wgslOwn = clothNewtonWGSL(CLOTH_BINDING.newton, 64, true, own.pcg), wgslBase = clothNewtonWGSL(CLOTH_BINDING.newton, 64, true, baseline.pcg);
  assert.doesNotMatch(wgslBase, /linArgs|@group\(1\)|LIN_STAGED|linStageKeys|linU\(/);
  // Group 1 is the args alone, bound by nwLinCount (the reset) and nwLinBuild (the crossing patch's count); the overflow
  // kernel's patch is counted from the end of the grid.
  assert.equal(wgslOwn.split("@group(1)").length - 1, 1);
  assert.match(wgslOwn, /@group\(1\) @binding\(0\) var<storage, read_write> linArgs: array<u32, 4>;/);
  assert.equal(wgslOwn.split("linArgs[").length - 1, 5);
  assert.match(wgslOwn, /fn pcgStepOverflow\([^]*?pcgLinOverflow\(lane, P\.newton3\.y - nwg\.x \+ wg\.x\)/);
  assert.match(wgslBase, /fn pcgStepOverflow\([^]*?pcgLinOverflow\(lane, wgIndex\(wg, nwg\)\)/);
  // pcgStep and nwModel stage the slots; the records read them through linU.
  assert.equal(wgslOwn.split("linStageKeys(lane, p);").length - 1, 2);
  assert.match(wgslOwn, /linU\(ids\[k\]\)/); assert.match(wgslBase, /pcgU\(ids\[k\]\)/);
  const bad = (ab: object) => () => packClothGPUTables(m, [0, 0, 0], { ...fact, contactLinearizationAB: ab as ClothContactLinAB });
  assert.throws(bad({ overflow: "auto" }), /Unknown cloth contactLinearizationAB\.overflow auto/);
  assert.throws(bad({ partners: "shared" }), /Unknown cloth contactLinearizationAB\.partners shared/);
  assert.throws(bad({ layout: "lanes" }), /Unknown cloth contactLinearizationAB key layout/);
  for (const slots of [0, 2, 3, 48, 1024, 1.5, NaN]) assert.throws(bad({ stageSlots: slots }), /contactLinearizationAB\.stageSlots/);
  // Off, or no contact: the knob has nothing to switch.
  assert.deepEqual(packClothGPUTables(m, [0, 0, 0], { ...base, newton: { contactLinearization: "off" }, contactLinearizationAB: { overflow: "grid", partners: "global", stageSlots: 4 } }), off);
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

type Internals = { newtonWork: GPUBuffer; linArgs?: GPUBuffer; run(pass: GPUComputePassEncoder, name: string, workgroups: number, record?: number): void; threads(n: number): number };
const LIN_NONE = 0x7f000000;
type Mode = "factorized" | "blocks";
type Kind = "solid" | "body" | "vt" | "ee" | "sample" | "facet";
interface Scenario {
  label: string; model: ReturnType<typeof twoSheets>; floor: ReturnType<typeof buildClothSolidField>; plate?: ReturnType<typeof buildClothSolidField>;
  /** Real steps before the fixed state; 0: the step's begin and its mandatory detection only (the entries at the authored pose). */
  steps: number; config?: Partial<OgcContactConfig>; newton?: Partial<ClothNewtonSettings>; kinds: readonly Kind[];
  /** The A/B knob (default the compiled modes' own pieces). */
  ab?: ClothContactLinAB;
}
const band = R_C + 1.5 * Math.sqrt(3) * RQ_MAX + 0.005;
const box = (lo: V3, hi: V3, flo: V3, fhi: V3) => buildClothSolidField({ parts: [{ kind: "sdf", sdf: boxSdf(lo, hi), lo, hi, mu: 0.5 }], lo: flo, hi: fhi, cell: 0.01, band });
const STILL = 8;
const poseAt = (step: number): OgcBodyPose => { const tm = Math.max(0, step - STILL) * DT; return { position: [0.06, -0.01, 0.05 * tm], orientation: quat([0, 1, 0], 0.3 * tm) }; };
/** The static box under the sheets' left half (top at y = 0), the plate under their right half: still, then sliding in z and turning. */
function stack(label: string, newton?: Partial<ClothNewtonSettings>, ab?: ClothContactLinAB): Scenario {
  const half: V3 = [0.04, 0.01, 0.06];
  return { label, model: twoSheets(), steps: 14, newton, ...(ab ? { ab } : {}), kinds: ["solid", "body", "vt", "ee", "sample"],
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
    newton: { ...NEWTON, ...sc.newton, contactLinearization: mode, contactLinearizationRecords: recordsPerVertex }, label: `lin ${mode}`, ...(sc.ab ? { contactLinearizationAB: sc.ab } : {}),
    contact: { ...det.solverContact(CLOTH_BINDING.contact, true), alpha: det.alpha, state: det.contactState },
    hooks: { afterFrame: det.stepHooks.afterFrame, filter: det.stepHooks.filter, afterNewtonStep: det.stepHooks.afterNewtonStep, stepEnd: det.stepHooks.stepEnd } });
  const stride0 = solver.tables.stride, L = solver.tables.pcg.contactLin!, patches = solver.tables.pcg.patches;
  const S = solver as unknown as Internals, W = S.threads(N), control = 16 * (solver.tables.newtonRecords - CLOTH_SEGMENT_ALIGN);
  det.bind({ positions: solver.positions, velocities: solver.velocities, stepStart: { buffer: solver.state, offset: 16 * CLOTH_STATE.xt * stride0, size: 16 * stride0 }, direction: solver.newtonDirection });
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
  // With headers, every patch's mark too (flag: LIN_NONE, the patch's rows to pcgStepOverflow; 0, to pcgStep's records).
  const sequence = async (build: boolean, headers: Uint32Array<ArrayBuffer> | undefined, then: (pass: GPUComputePassEncoder) => void, flag = 0) => {
    device.queue.writeBuffer(S.newtonWork, control, new Float32Array(4));
    if (headers) {
      device.queue.writeBuffer(S.newtonWork, 16 * L.head, headers);
      const marks = new Uint32Array(4 * patches); for (let p = 0; p < patches; p++) marks[4 * p + 1] = flag;
      device.queue.writeBuffer(S.newtonWork, 16 * L.patch, marks);
      // The indirect overflow's count to match: every patch, or none.
      if (L.overflow === "indirect") device.queue.writeBuffer(S.linArgs!, 0, new Uint32Array([flag === LIN_NONE ? patches : 0, 1, 1, 0]));
    }
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    S.run(pass, "nwFaces", S.threads(T)); S.run(pass, "nwGradient", W);
    if (build) { S.run(pass, "nwLinCount", patches); S.run(pass, "nwLinBuild", patches); }
    then(pass);
    pass.end(); device.queue.submit([enc.finish()]);
  };
  // The product: pcgStep dispatch 0 (w0 = P H M^-1 r_0), or with one patch pcgSolve capped at 3 iterations (d).
  const one = patches === 1, out = one ? CLOTH_NEWTON_SEGMENTS.d : CLOTH_NEWTON_SEGMENTS.w0;
  const pcg = (pass: GPUComputePassEncoder, i: number) => { S.run(pass, "pcgStep", patches, i); S.run(pass, "pcgStepOverflow", patches, i); };
  const product = async (build: boolean, headers?: Uint32Array<ArrayBuffer>, flag = 0): Promise<Float32Array> => {
    await sequence(build, headers, pass => { if (one) S.run(pass, "pcgSolve", 1, 3); else pcg(pass, 0); }, flag);
    return new Float32Array(await readBuffer(device, S.newtonWork, 16 * out * stride0, 16 * N));
  };
  const compiled = await product(true);
  const stats = (await solver.readContactLinearization())!;
  const head = new Uint32Array(await readBuffer(device, S.newtonWork, 16 * L.head, 16 * N));
  const marks = new Uint32Array(await readBuffer(device, S.newtonWork, 16 * L.patch, 16 * patches));
  const pool = new Uint32Array(await readBuffer(device, S.newtonWork, 16 * L.pool, 16 * L.records));
  const allFree = head.slice(), none = head.slice();
  // Record c of item i of a compiled row, and the patch of each vertex (the PCG rows).
  const per = mode === "factorized" ? 6 : 2;
  const recordAt = (v: number, i: number, c: number) => head[4 * v]! - L.pool + per * i + c;
  const P32 = new Uint32Array(solver.tables.params.buffer, solver.tables.params.byteOffset), rowsWord = P32[CLOTH_PARAM_NEWTON + 16]!;
  const patchOf = new Uint32Array(N);
  for (let slot = 0; slot < N; slot++) patchOf[solver.tables.topo[rowsWord + slot]!] = Math.floor(slot / 256);
  let compiledRows = 0, globalColumns = 0, localColumns = 0, stagedColumns = 0;
  for (let v = 0; v < N; v++) {
    if (head[4 * v + 2] === 0) continue;
    if (head[4 * v] !== LIN_NONE) {
      compiledRows++;
      const first = head[4 * v]! - L.pool;
      assert.ok(first >= 0 && first + head[4 * v + 2]! <= L.records, `row ${v}'s records lie in the pool`);
      // Its partner columns: a lane of its patch, a staged slot, or global.
      for (let i = 0; i < head[4 * v + 1]!; i++) {
        const cols = mode === "blocks" ? [pool[4 * recordAt(v, i, 0) + 3]!] : Array.from({ length: 4 }, (_, k) => pool[4 * recordAt(v, i, 0) + k]!)
          .filter((_, k) => new Float32Array(pool.buffer, 16 * recordAt(v, i, 1), 4)[k] !== 0 || new Float32Array(pool.buffer, 16 * recordAt(v, i, 2), 4)[k] !== 0);
        for (const c of cols) { if (c & 0x80000000) globalColumns++; else if (c & 0x40000000) { stagedColumns++; assert.ok((c & 0x3fffffff) < L.staged!.slots, `slot ${c & 0x3fffffff} of ${L.staged!.slots}`); } else localColumns++; }
      }
    }
    // Back to the matrix-free product: the mark, with the entry count nwLinCount found.
    allFree[4 * v] = LIN_NONE; allFree[4 * v + 1] = Math.min(words[countsAt + v]!, cap);
    none[4 * v] = 0; none[4 * v + 1] = 0;
  }
  // The marked patches: a suffix from the first marked one, k0 (patches that took no record stay unmarked), so the
  // indirect overflow dispatches patches - k0 workgroups.
  const totals = Array.from({ length: patches }, (_, q) => marks[4 * q]!), marked = Array.from({ length: patches }, (_, q) => marks[4 * q + 1] === LIN_NONE);
  const k0 = marked.indexOf(true) < 0 ? patches : marked.indexOf(true);
  for (let q = k0; q < patches; q++) assert.equal(marked[q], totals[q]! > 0, `patch ${q} (from k0 ${k0}) is marked iff it took records`);
  if (L.overflow === "indirect") {
    assert.equal(stats.overflowWorkgroups, patches - k0, `pcgStepOverflow's indirect count: the patches from k0 = ${k0} on`);
    // Coverage: pcgStepOverflow alone at dispatch 0 over a poisoned w writes exactly the marked patches' rows.
    device.queue.writeBuffer(S.newtonWork, 16 * CLOTH_NEWTON_SEGMENTS.w0 * stride0, new Float32Array(4 * N).fill(NaN));
    await sequence(true, undefined, pass => S.run(pass, "pcgStepOverflow", patches, 0));
    const w = new Float32Array(await readBuffer(device, S.newtonWork, 16 * CLOTH_NEWTON_SEGMENTS.w0 * stride0, 16 * N));
    let written = 0;
    for (let v = 0; v < N; v++) {
      const q = patchOf[v]!, finite = Number.isFinite(w[4 * v]!);
      assert.equal(finite, marked[q]!, `vertex ${v} of patch ${q}: pcgStepOverflow ${marked[q] ? "writes" : "leaves"} it (k0 ${k0})`);
      if (finite) written++;
    }
    log(`  indirect overflow: k0 ${k0} of ${patches} patches (totals ${totals.join(" ")}, pool ${L.records}), ${stats.overflowWorkgroups} workgroups, ${written} rows written by pcgStepOverflow alone`);
  }
  // The reference: every row matrix-free (with several patches, every patch through pcgStepOverflow); no contact.
  const free = await product(false, allFree, LIN_NONE), empty = await product(false, none);
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
    // A row the pool could not hold: its patch went through pcgStepOverflow, the reference's kernel on the same inputs.
    if (head[4 * v] === LIN_NONE && !one) assert.equal(err, 0, `row ${v} overflowed the pool: its patch took pcgStepOverflow (the matrix-free bits)`);
    if (head[4 * v + 2] !== 0) rows++;
    maxContact = Math.max(maxContact, contact); sumContact2 += contact ** 2; sumErr2 += err ** 2; worstAbs = Math.max(worstAbs, err);
    if (contact > 0) worstRel = Math.max(worstRel, err / (contact + whole));
    if (!(err <= bound * 1.1920929e-7 * (contact + whole))) bad++;
  }
  log(`  ${stats.items} items in ${stats.usedRecords} records (${stats.usedBytes} B) of ${stats.poolRecords}, demand ${stats.demandRecords}; rows compiled ${compiledRows}, left matrix-free ${stats.fallbackRows} (patches through pcgStepOverflow ${stats.fallbackPatches} of ${patches})`
    + `; columns ${localColumns} in the row's patch, ${stagedColumns} staged, ${globalColumns} global (overflow ${L.overflow}${L.staged ? `, ${L.staged.slots} slots` : ""})`);
  log(`  ${one ? "pcgSolve (3 iterations), d" : "pcgStep 0, w"}: compiled vs matrix-free over ${rows} contact rows: max |dw| ${worstAbs.toExponential(2)}, rms ${Math.sqrt(sumErr2 / Math.max(rows, 1)).toExponential(2)}; contact's share max ${maxContact.toExponential(2)}, rms ${Math.sqrt(sumContact2 / Math.max(rows, 1)).toExponential(2)}; `
    + `worst |dw| / (|contact| + |w|) ${worstRel.toExponential(2)} (eps32 1.19e-7); rows past ${bound} eps ${bad}`);
  assert.ok(rows >= (one ? 5 : N / 4) && maxContact > 0, `contact acts on ${rows} rows`);
  assert.equal(stats.rows, rows);
  assert.equal(stats.fallbackRows + compiledRows, rows);
  if (recordsPerVertex < 32) {
    assert.ok(stats.fallbackRows > 0 && compiledRows > 0, `a short pool compiles some rows (${compiledRows}) and leaves others matrix-free (${stats.fallbackRows})`);
    assert.ok(k0 < patches && stats.fallbackPatches === patches - k0, `patches through pcgStepOverflow (${stats.fallbackPatches} of ${patches}, from k0 ${k0})`);
  } else assert.ok(stats.fallbackRows === 0 && stats.fallbackPatches === 0, "the authored pool holds this scene");
  if (!one) assert.ok(globalColumns + stagedColumns > 0 && localColumns > 0, "partners in the row's patch and outside it");
  if (L.staged && !one) {
    assert.ok(stagedColumns > 0, "outside partners staged");
    // A table too small for the patch's outside partners: the rest stay global (the full table's fallback).
    if (L.staged.slots <= 4) assert.ok(globalColumns > 0, `a full ${L.staged.slots}-slot table leaves partners global`);
  } else if (!L.staged) assert.equal(stagedColumns, 0, "no staged partners with the knob's global partners");
  assert.equal(bad, 0, `every row within ${bound} eps32 of (|contact share| + |row product|); worst ${worstRel.toExponential(2)}`);

  if (sc.newton?.trustRegion === "hybrid") {
    // The trust region's model at a step s (nwMove's trial 0: the direction d of eight PCG iterations from here, run
    // once through the records): s.H s and contact's share of it, per patch into PCG partial array 2. Then nwModel
    // alone per variant, on that one d, with the trust record set so the kernel takes the iteration as a trust-region one.
    const partials = new Uint32Array(solver.tables.params.buffer, solver.tables.params.byteOffset)[CLOTH_PARAM_NEWTON + 12]!;
    await sequence(true, undefined, pass => { for (let i = 0; i <= 8; i++) pcg(pass, i); });
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
  return { totals, k0, N };
}

test("GPU compiled contact linearization: the PCG product through the records is the matrix-free product to f32 rounding (solid, moving solid, VT, EE and sample entries, friction on)", { skip: !gpu }, async t => {
  await withDevice(async device => {
    // This stack is dense in pair entries (about 5 a vertex, 8 on the rows that have any): a pool authored for it.
    await operatorCheck(device, stack("stack"), "factorized", 48, s => t.diagnostic(s));
    await operatorCheck(device, stack("stack"), "blocks", 48, s => t.diagnostic(s));
    // A short pool (12 records a vertex against a demand near 31): the first patches compile, the rest go through
    // pcgStepOverflow; and one record a vertex, where hardly a row compiles.
    const short = await operatorCheck(device, stack("stack"), "factorized", 12, s => t.diagnostic(s));
    assert.ok(short.k0 > 0, "some patches through pcgStep's records, the rest through pcgStepOverflow");
    await operatorCheck(device, stack("stack"), "factorized", 1, s => t.diagnostic(s));
  });
});

/** The A/B knob's baseline: the grid overflow, global partners. */
const BASELINE: ClothContactLinAB = { overflow: "grid", partners: "global" };
test("GPU compiled contact linearization, experiment 4: the A/B knob's baseline (grid overflow, global partners) and a full staged-slot table are the matrix-free product to f32 rounding", { skip: !gpu }, async t => {
  await withDevice(async device => {
    await operatorCheck(device, stack("stack, baseline", undefined, BASELINE), "factorized", 48, s => t.diagnostic(s));
    await operatorCheck(device, stack("stack, baseline", undefined, BASELINE), "blocks", 48, s => t.diagnostic(s));
    const short = await operatorCheck(device, stack("stack, baseline", undefined, BASELINE), "factorized", 12, s => t.diagnostic(s));
    assert.ok(short.k0 > 0, "some patches through pcgStep's records, the rest through pcgStepOverflow");
    await operatorCheck(device, stack("stack, hybrid, baseline", { trustRegion: "hybrid" }, BASELINE), "factorized", 48, s => t.diagnostic(s));
    // Four slots a patch: the table fills, the rest of the patch's outside partners stay global.
    await operatorCheck(device, stack("stack, 4 slots", undefined, { stageSlots: 4 }), "factorized", 48, s => t.diagnostic(s));
    await operatorCheck(device, stack("stack, 4 slots", undefined, { stageSlots: 4 }), "blocks", 48, s => t.diagnostic(s));
    await operatorCheck(device, stack("stack, hybrid, 4 slots", { trustRegion: "hybrid" }, { stageSlots: 4 }), "blocks", 48, s => t.diagnostic(s));
  });
});

test("GPU compiled contact linearization: the indirect overflow covers exactly the patches from the first the pool could not hold (k0 forced to the first, a middle and the last patch; none: no workgroup)", { skip: !gpu }, async t => {
  await withDevice(async device => {
    const sc = () => stack("stack, indirect");
    // The authored pool holds every row: k0 past the last patch, a zero count.
    const all = await operatorCheck(device, sc(), "factorized", 48, s => t.diagnostic(s));
    const patches = all.totals.length;
    assert.ok(patches >= 3 && all.k0 === patches, `the authored pool holds the scene (k0 ${all.k0} of ${patches})`);
    // The pool's end in the middle of patch k's records (records = a whole number a vertex): k0 = k.
    for (const k of [0, Math.floor(patches / 2), patches - 1]) {
      const base = all.totals.slice(0, k).reduce((a, b) => a + b, 0), total = all.totals[k]!;
      const perVertex = Math.max(1, Math.round((base + total / 2) / all.N));
      assert.ok(total > 0 && perVertex * all.N >= base && perVertex * all.N < base + total, `a pool of ${perVertex} records a vertex ends inside patch ${k} [${base}, ${base + total})`);
      const run = await operatorCheck(device, sc(), "factorized", perVertex, s => t.diagnostic(s));
      assert.equal(run.k0, k, `the pool's end forced into patch ${k}`);
    }
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

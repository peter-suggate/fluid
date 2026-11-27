/**
 * I-Cloth (lib/cloth/icloth; Tang et al. 2018): the Float64 references, and on the GPU (WEBGPU_NODE_MODULE)
 *   1. the continuous collision kernels against the Float64 oracle on constructed vertex-face and edge-edge crossings
 *      and near misses (and the proximity records of a pair at rest);
 *   2. the impact-zone solve against the Float64 implementation of the same augmented-Lagrangian problem;
 *   3. end to end through ClothSystem (method 'icloth'): a sheet thrown at a moving solid, and a sheet thrown through
 *      another, end penetration-free where the same run without the collision rounds does not;
 *   4. a pose saved mid-run steps on as the straight run does; a 'newton' document's solver module has no I-Cloth terms;
 *   5. an unknown method fails explicitly.
 *
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js tools/gpu/fluid-gpu-run.sh check \
 *     node --import tsx --test --test-concurrency=1 tests/cloth-icloth.test.ts
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { createEmptyScene } from '../lib/core/empty-scene';
import { validateScene, type SceneDescription } from '../lib/core/model';
import { clothGridData } from '../lib/cloth/grid';
import { prepareClothTopology, validateClothDocument, type PreparedClothTopology } from '../lib/cloth/topology';
import { buildClothSolverModel } from '../lib/cloth/oracle';
import { publishSceneClothSurfaces, snapshotSceneClothPose } from '../lib/cloth/scene-geometry';
import { CLOTH_SCENE_STEP_S, ClothSystem, type ClothSystemOptions } from '../lib/cloth/system';
import { packClothGPUTables } from '../lib/cloth/gpu-solver';
import { CLOTH_BINDING, CLOTH_STATE, clothSolverWGSL } from '../lib/cloth/gpu-solver-wgsl';
import { buildClothBodyField, clothBodyPoses, clothContactRadius } from '../lib/cloth/scene-contact';
import { sampleClothSolidField } from '../lib/cloth/contact/solid-field';
import { clothStageOf } from '../lib/cloth/stage-timing';
import { clothExecutionSettings } from '../lib/cloth/execution-settings';
import type { ClothDocument, ClothMaterial, ClothTopologyData } from '../lib/cloth/types';
import { ICLOTH_FAILING, ICLOTH_STATUS, IClothCollision, iclothOmega, iclothOwnershipMasks, iclothTolerances, type IClothConfig } from '../lib/cloth/icloth/icloth';
import { ICLOTH_KIND, iclothProximityBlockWGSL } from '../lib/cloth/icloth/icloth-wgsl';
import { icCcd, iclothSelfIntersections, iclothZoneSolveOracle, type IcImpact, type IcQuad, type IcV3 } from '../lib/cloth/icloth/oracle';

const gpu = !!process.env.WEBGPU_NODE_MODULE;
const R = 1e-3, TOL = iclothTolerances(R);

// ------------------------------------------------------------------ constructed pairs
type P3 = [number, number, number];
interface Case { name: string; start: P3[]; end: P3[]; expect: { kind: 0 | 1; hit: 1 | 2; n: P3 }[] }
const moved = (p: P3[], which: number[], d: P3): P3[] => p.map((x, i) => which.includes(i) ? [x[0] + d[0], x[1] + d[1], x[2] + d[2]] as P3 : x);
/** Triangle B flat in y = 0; triangle A a tent whose corner 0 points at it from `h` (below when up = -1), over (px, pz). */
function vertexFace(name: string, h0: number, h1: number, px: number, up: 1 | -1, expect: Case['expect']): Case {
  const away = px < 0 ? -1 : 1;
  const A = (h: number): P3[] => [[px, up * h, 0.005], [px + away * 0.02, up * (h + 0.05), 0.005], [px, up * (h + 0.05), 0.025]];
  const B: P3[] = [[0, 0, 0], [0.02, 0, 0], [0, 0, 0.02]];
  return { name, start: [...A(h0), ...B], end: [...A(h1), ...B], expect };
}
/** Edge 0 of A along x at height h over B's edge 0 along z (B hangs below); A's edge crosses z = `pz`. */
function edgeEdge(name: string, h0: number, h1: number, pz: number, expect: Case['expect']): Case {
  const A = (h: number): P3[] => [[-0.01, h, pz], [0.01, h, pz], [0.003, h + 0.05, pz]];
  const B: P3[] = [[0, 0, -0.01], [0, 0, 0.02], [0, -0.05, 0.006]];
  return { name, start: [...A(h0), ...B], end: [...A(h1), ...B], expect };
}
const CASES: Case[] = [
  vertexFace('a vertex through a face from above', 0.003, -0.003, 0.005, 1, [{ kind: 0, hit: 1, n: [0, 1, 0] }]),
  vertexFace('a vertex through a face from below', 0.003, -0.003, 0.005, -1, [{ kind: 0, hit: 1, n: [0, -1, 0] }]),
  vertexFace('a vertex that stops 0.5 mm short of a face', 0.003, 0.0005, 0.005, 1, []),
  vertexFace('a vertex that ends 20 um from a face (inside the end tolerance)', 0.003, 0.00002, 0.005, 1, [{ kind: 0, hit: 2, n: [0, 1, 0] }]),
  vertexFace('a vertex through the face\'s plane 5 mm outside it', 0.003, -0.003, -0.005, 1, []),
  edgeEdge('an edge through an edge', 0.003, -0.003, 0.005, [{ kind: 1, hit: 1, n: [0, -1, 0] }]),
  edgeEdge('an edge that stops 0.5 mm short of an edge', 0.003, 0.0005, 0.005, []),
  edgeEdge('an edge past the end of an edge by 5 mm', 0.003, -0.003, 0.025, []),
  edgeEdge('an edge that ends 30 um from an edge', 0.003, 0.00003, 0.005, [{ kind: 1, hit: 2, n: [0, -1, 0] }]),
];
/** Every elementary pair between a case's two triangles, as the kernel enumerates them. */
function casePairs(base: number): { kind: 0 | 1; ids: [number, number, number, number] }[] {
  const A = [base, base + 1, base + 2], B = [base + 3, base + 4, base + 5], out: { kind: 0 | 1; ids: [number, number, number, number] }[] = [];
  for (const v of A) out.push({ kind: 0, ids: [v, B[0]!, B[1]!, B[2]!] });
  for (const v of B) out.push({ kind: 0, ids: [v, A[0]!, A[1]!, A[2]!] });
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out.push({ kind: 1, ids: [A[i]!, A[(i + 1) % 3]!, B[j]!, B[(j + 1) % 3]!] });
  return out;
}
/** The cases side by side (0.1 m apart) as one mesh of disjoint triangles; positions rounded to f32 as the GPU holds them. */
function layout(cases: Case[]) {
  const start: number[] = [], end: number[] = [], faces: number[] = [];
  cases.forEach((c, i) => {
    const dx = 0.1 * (i - (cases.length - 1) / 2);
    for (const p of c.start) start.push(Math.fround(p[0] + dx), Math.fround(p[1]), Math.fround(p[2]));
    for (const p of c.end) end.push(Math.fround(p[0] + dx), Math.fround(p[1]), Math.fround(p[2]));
    faces.push(6 * i, 6 * i + 1, 6 * i + 2, 6 * i + 3, 6 * i + 4, 6 * i + 5);
  });
  return { start, end, faces };
}
const quad = (x: number[], ids: readonly number[]): IcQuad => ids.map(v => [x[3 * v]!, x[3 * v + 1]!, x[3 * v + 2]!] as IcV3) as unknown as IcQuad;
/** The oracle's impacts of a layout (target separation delta). */
function oracleImpacts(start: number[], end: number[], cases: number): (IcImpact & { kind: number; hit: number })[] {
  const out: (IcImpact & { kind: number; hit: number })[] = [];
  for (let i = 0; i < cases; i++) for (const p of casePairs(6 * i)) {
    const h = icCcd(p.kind, quad(start, p.ids), quad(end, p.ids), TOL.eta, TOL.dTol);
    if (h.hit) out.push({ kind: p.kind, hit: h.hit, ids: p.ids, n: h.n, d0: TOL.delta, w: h.w });
  }
  return out;
}
/** What IClothCollision reads of a topology. */
function triangleSoup(rest: number[], faces: number[], mass: number[], groups: number[]): PreparedClothTopology {
  const P = (v: number) => [rest[3 * v]!, rest[3 * v + 1]!, rest[3 * v + 2]!], edge: number[] = [], area: number[] = [];
  for (let t = 0; t < faces.length / 3; t++) {
    const [a, b, c] = [P(faces[3 * t]!), P(faces[3 * t + 1]!), P(faces[3 * t + 2]!)];
    const l = (p: number[], q: number[]) => Math.hypot(p[0]! - q[0]!, p[1]! - q[1]!, p[2]! - q[2]!);
    edge.push(l(a!, b!), l(b!, c!), l(c!, a!));
    const u = [b![0]! - a![0]!, b![1]! - a![1]!, b![2]! - a![2]!], w = [c![0]! - a![0]!, c![1]! - a![1]!, c![2]! - a![2]!];
    area.push(0.5 * Math.hypot(u[1]! * w[2]! - u[2]! * w[1]!, u[2]! * w[0]! - u[0]! * w[2]!, u[0]! * w[1]! - u[1]! * w[0]!));
  }
  return { data: { restPositions_m: rest, triangles: faces }, vertexCount: rest.length / 3, faceCount: faces.length / 3, faces: Uint32Array.from(faces),
    edgeRestLength_m: Float64Array.from(edge), faceArea_m2: Float64Array.from(area), mass_kg: Float64Array.from(mass), groups: Uint32Array.from(groups) } as unknown as PreparedClothTopology;
}

// ------------------------------------------------------------------ Float64 references (no GPU)
test('the Float64 collision test finds the constructed crossings and end contacts, with the normal on the side the pair came from, and no near miss', () => {
  const { start, end } = layout(CASES), found = oracleImpacts(start, end, CASES.length);
  CASES.forEach((c, i) => {
    const mine = found.filter(h => h.ids[0] >= 6 * i && h.ids[0] < 6 * i + 6);
    assert.equal(mine.length, c.expect.length, `${c.name}: ${mine.length} impacts`);
    c.expect.forEach((e, k) => {
      const h = mine[k]!;
      assert.equal(h.kind, e.kind, c.name); assert.equal(h.hit, e.hit, c.name);
      for (let a = 0; a < 3; a++) assert.ok(Math.abs(h.n[a]! - e.n[a]!) < 1e-6, `${c.name}: normal ${h.n.join(', ')}`);
      // The weights are a partition: the vertex against the face's point, or the two edges' points.
      assert.ok(Math.abs(h.w[0] + h.w[1] + h.w[2] + h.w[3]) < 1e-9, `${c.name}: weights ${h.w.join(', ')}`);
    });
  });
  // The vertex pierces B at (0.005, 0, 0.005): barycentric (0.5, 0.25, 0.25); the edges cross at their s = t = 0.5.
  assert.deepEqual(found[0]!.w.map(x => Math.round(1e6 * x) / 1e6), [1, -0.5, -0.25, -0.25]);
  assert.deepEqual(found.find(h => h.kind === 1 && h.hit === 1)!.w.map(x => Math.round(1e6 * x) / 1e6), [-0.5, -0.5, 0.5, 0.5]);
});

test('the Float64 impact-zone solve reaches the constrained minimum: a plane on one vertex, and a vertex-face pair shared by mass', () => {
  // One vertex 0.3 mm under a plane y >= 1e-4: it moves to the plane along the normal and no further.
  // The solve stops once its step and every violation are under the feasibility tolerance (0.1 delta); a multiplier
  // may then hold a vertex up to that much beyond its plane, so the minimum is met to 0.2 delta (2e-5 m).
  const one = iclothZoneSolveOracle(Float64Array.from([0.2, -2e-4, 0.1]), [{ ids: [0, -1, -1, -1], n: [0, 1, 0], d0: 1e-4, w: [1, 0, 0, 0] }],
    { mass: [1e-4], pinned: [0], iterations: 100, stepThreshold: TOL.stepThreshold, feasible: 0.1 * TOL.delta });
  assert.ok(Math.abs(one.positions[1]! - 1e-4) < 0.2 * TOL.delta && one.positions[0] === 0.2 && one.positions[2] === 0.1, `plane: y = ${one.positions[1]}`);
  // Vertex 0 (mass 3 m) 0.3 mm through the face of 1, 2, 3 (mass m each) at its centroid: the closed form of
  // min sum m_i |d_i|^2 s.t. n . sum w_i d_i = g is d_i = g (w_i / m_i) / sum_j (w_j^2 / m_j) n.
  const m = 1e-4, x0 = Float64Array.from([0.01, -3e-4, 0.01, 0, 0, 0, 0.03, 0, 0, 0, 0, 0.03]), w = [1, -1 / 3, -1 / 3, -1 / 3] as const, masses = [3 * m, m, m, m];
  const pair = iclothZoneSolveOracle(x0, [{ ids: [0, 1, 2, 3], n: [0, 1, 0], d0: TOL.delta, w }], { mass: masses, pinned: [0, 0, 0, 0], iterations: 200, stepThreshold: TOL.stepThreshold, feasible: 0.1 * TOL.delta });
  const gap = TOL.delta + 3e-4, denom = w.reduce((s, wi, i) => s + wi * wi / masses[i]!, 0);
  for (let v = 0; v < 4; v++) {
    const want = x0[3 * v + 1]! + gap * (w[v]! / masses[v]!) / denom;
    assert.ok(Math.abs(pair.positions[3 * v + 1]! - want) < 0.2 * TOL.delta, `vertex ${v}: y ${pair.positions[3 * v + 1]} against ${want} (${pair.iterations} iterations)`);
  }
  assert.ok(pair.maxViolation < 0.1 * TOL.delta && !pair.capped, `violation ${pair.maxViolation} after ${pair.iterations} iterations`);
  // A pinned face: the vertex alone closes the gap.
  const held = iclothZoneSolveOracle(x0, [{ ids: [0, 1, 2, 3], n: [0, 1, 0], d0: TOL.delta, w }], { mass: masses, pinned: [0, 1, 1, 1], iterations: 200, stepThreshold: TOL.stepThreshold, feasible: 0.1 * TOL.delta });
  assert.ok(Math.abs(held.positions[1]! - TOL.delta) < 0.2 * TOL.delta && held.positions[4] === 0, `pinned face: vertex y ${held.positions[1]}`);
});

test('ownership masks give every vertex and every edge to one triangle; Chebyshev is off; kernels land on the I-Cloth stages', () => {
  const data = clothGridData({ columns: 5, rows: 4, width_m: 0.4, height_m: 0.3, origin_m: [0, 0, 0] });
  const faces = Uint32Array.from(data.triangles), masks = iclothOwnershipMasks(faces, 20);
  const vertices = new Map<number, number>(), edges = new Map<string, number>(), all = new Set<string>();
  for (let t = 0; t < faces.length / 3; t++) for (let i = 0; i < 3; i++) {
    const a = faces[3 * t + i]!, b = faces[3 * t + (i + 1) % 3]!, key = `${Math.min(a, b)}-${Math.max(a, b)}`;
    all.add(key);
    if (masks[t]! & (1 << i)) vertices.set(a, (vertices.get(a) ?? 0) + 1);
    if (masks[t]! & (8 << i)) edges.set(key, (edges.get(key) ?? 0) + 1);
  }
  assert.equal(vertices.size, 20); assert.ok([...vertices.values()].every(n => n === 1));
  assert.equal(edges.size, all.size); assert.ok([...edges.values()].every(n => n === 1));
  // Chebyshev is off (ICLOTH_SOLVER.rho 0: it diverges with the per-iteration multiplier update at this scale).
  assert.equal(iclothOmega(0), 1); assert.equal(iclothOmega(9), 1); assert.equal(iclothOmega(10), 1); assert.equal(iclothOmega(99), 1);
  assert.equal(clothStageOf('icloth prox icPairs'), 'icloth-proximity');
  assert.equal(clothStageOf('icloth icPairs'), 'icloth-ccd');
  assert.equal(clothStageOf('icloth icGrad'), 'icloth-impact');
  assert.equal(clothStageOf('Cloth pcgStep'), 'pcg');
});

const cotton: ClothMaterial = { id: 'cotton', arealDensity_kg_m2: 0.2, stretchWarp_N_m: 400, stretchWeft_N_m: 250, shear_N_m: 40, poisson: 0.3,
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0.002, thickness_m: 5e-4, strainLimit: 0.02 };

test("a 'newton' solver module carries no I-Cloth term; the proximity block adds them; the method defaults to 'newton' and an unknown one is refused", () => {
  const data = clothGridData({ columns: 6, rows: 6, width_m: 0.3, height_m: 0.3, origin_m: [0, 1, 0] });
  const model = buildClothSolverModel(prepareClothTopology(data, [cotton]), [cotton], []);
  const pcg = packClothGPUTables(model, [0, 0, 0], { dt_s: 1 / 60, gravity_m_s2: [0, -9.81, 0], integrator: 'implicit-euler' }).pcg;
  const plain = clothSolverWGSL({ pcg });
  assert.equal(plain, clothSolverWGSL({ pcg, proximityBlock: undefined }));
  for (const term of ['pxVertexTerms', 'pxVertexHv', 'pxOwnedEnergy', 'nwTrialMove', `@binding(${CLOTH_BINDING.proximity})`]) assert.ok(!plain.includes(term), `the plain module names ${term}`);
  const block = iclothProximityBlockWGSL({ binding: CLOTH_BINDING.proximity, lists: 400, records: 4000, entriesPerVertex: 48, r: 5e-4, kc: 100, epsU: 1e-5 });
  const withTerms = clothSolverWGSL({ pcg, proximityBlock: block });
  for (const term of ['pxVertexTerms(v, xv', 'pxVertexHv(v)', 'pxOwnedEnergy(v)', `@binding(${CLOTH_BINDING.proximity})`]) assert.ok(withTerms.includes(term), `the proximity module lacks ${term}`);
  assert.throws(() => clothSolverWGSL({ pcg, proximityBlock: 'fn pxVertexTerms() {}' }), /pxVertexHv/);
  assert.equal(clothExecutionSettings({}).method, 'newton');
  assert.deepEqual(clothExecutionSettings({ method: 'icloth' }).icloth, { rounds: 8, impactIterations: 100 });
  const doc = { schemaVersion: '2.0.0', id: 'sheet', geometry: { kind: 'inline', data }, materials: [cotton], solver: { method: 'verlet' } } as unknown as ClothDocument;
  assert.ok(validateClothDocument(doc).some(e => /Unknown cloth method verlet/.test(e)));
});

// ------------------------------------------------------------------ GPU
async function withDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('subgroups') ? ['subgroups'] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener('uncapturederror', (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); }
  finally { device.destroy(); }
}

/** The collision stage alone on a triangle soup: the test owns the solver's two buffers (positions; x_t and y segments). */
async function standalone(device: GPUDevice, start: number[], end: number[], faces: number[], mass: number[], groups: number[], config: Partial<IClothConfig>) {
  const N = start.length / 3, stride = Math.ceil(N / 16) * 16, segments = Math.max(CLOTH_STATE.xt, CLOTH_STATE.y) + 1;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const positions = device.createBuffer({ size: 16 * stride, usage }), state = device.createBuffer({ size: 16 * segments * stride, usage });
  const pack = (x: number[]) => { const f = new Float32Array(4 * stride); for (let v = 0; v < N; v++) f.set([x[3 * v]!, x[3 * v + 1]!, x[3 * v + 2]!, 0], 4 * v); return f; };
  device.queue.writeBuffer(positions, 0, pack(end));
  device.queue.writeBuffer(state, 16 * CLOTH_STATE.xt * stride, pack(start));
  device.queue.writeBuffer(state, 16 * CLOTH_STATE.y * stride, pack(end));
  const icloth = await IClothCollision.create(device, triangleSoup(start, faces, mass, groups),
    { r: R, kc: 100, selfFriction: 0.3, dt_s: 1 / 60, epsV_m_s: 1e-3, rounds: 1, impactIterations: 200, stepsPerFrame: 1, body: false, ...config });
  icloth.bind({ positions, state, stride, xtSegment: CLOTH_STATE.xt, ySegment: CLOTH_STATE.y });
  const submit = (encode: (pass: GPUComputePassEncoder) => void) => {
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    encode(pass); pass.end();
    device.queue.submit([encoder.finish()]);
  };
  const read = async () => {
    const staging = device.createBuffer({ size: 16 * stride, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(positions, 0, staging, 0, 16 * stride);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(staging.getMappedRange().slice(0)), out = new Float64Array(3 * N);
    staging.destroy();
    for (let v = 0; v < N; v++) out.set([f[4 * v]!, f[4 * v + 1]!, f[4 * v + 2]!], 3 * v);
    return out;
  };
  return { icloth, submit, read, destroy: () => { icloth.destroy(); positions.destroy(); state.destroy(); } };
}

test('1. the GPU collision test agrees with the Float64 oracle on constructed crossings and near misses; a pair at rest inside the range is a proximity record', { skip: !gpu }, async () => withDevice(async device => {
  // The cases, and one pair at rest 1.5 mm apart (inside the 2 mm proximity range, outside every collision tolerance).
  const cases = [...CASES, vertexFace('a vertex at rest 1.5 mm over a face', 0.0015, 0.0015, 0.005, 1, [])];
  const { start, end, faces } = layout(cases), N = start.length / 3;
  const h = await standalone(device, start, end, faces, new Array(N).fill(1e-4), new Array(N).fill(0), {});
  h.icloth.encodeFrameBegin();
  h.submit(pass => h.icloth.encodeDetectOnly(pass));
  const got = await h.icloth.readImpacts(), want = oracleImpacts(start, end, cases.length), counters = await h.icloth.readCounters();
  assert.equal(counters.status, 0, IClothCollision.describe(counters.status).join('; '));
  console.log(`CCD: ${got.length} GPU impacts, ${want.length} oracle impacts over ${cases.length} constructed pairs of triangles (${18 * cases.length} elementary pairs)`);
  assert.equal(got.length, want.length);
  let worstN = 0, worstW = 0, worstEnd = 0;
  for (const w of want) {
    const g = got.find(x => x.kind === w.kind && x.ids.join() === w.ids.join());
    assert.ok(g, `the GPU lacks the oracle's ${w.kind ? 'edge-edge' : 'vertex-face'} impact of ${w.ids.join(', ')}`);
    // An end contact's normal is the direction of a 20-30 um separation of f32 positions (ulp 3e-8 m at 0.4 m).
    for (let a = 0; a < 3; a++) { const dn = Math.abs(g.n[a]! - w.n[a]!); if (w.hit === 2) worstEnd = Math.max(worstEnd, dn); else worstN = Math.max(worstN, dn); }
    for (let a = 0; a < 4; a++) worstW = Math.max(worstW, Math.abs(g.w[a]! - w.w[a]!));
    assert.ok(Math.abs(g.d0 - TOL.delta) < 1e-9);
  }
  console.log(`CCD: largest normal component difference ${worstN.toExponential(2)} at a root, ${worstEnd.toExponential(2)} at an end contact; largest weight difference ${worstW.toExponential(2)}`);
  assert.ok(worstN < 1e-4 && worstW < 1e-4 && worstEnd < 5e-3);
  // The proximity pass at x_t: the one pair inside the range, its normal from the face to the vertex.
  h.submit(pass => { h.icloth.stepHooks.afterFrame(pass); });
  const prox = await h.icloth.readProximity(), last = 6 * (cases.length - 1);
  assert.equal(prox.length, 1, `${prox.length} proximity records`);
  assert.deepEqual(prox[0]!.ids, [last, last + 3, last + 4, last + 5]);
  assert.ok(Math.abs(prox[0]!.n[1] - 1) < 1e-5 && prox[0]!.muN === 0, `proximity normal ${prox[0]!.n.join(', ')}, friction force ${prox[0]!.muN}`);
  h.destroy();
}));

test('2. the GPU impact-zone solve lands on the Float64 solve of the same problem: free pairs, unequal masses and a pinned face', { skip: !gpu }, async () => withDevice(async device => {
  const cases = [CASES[0]!, CASES[5]!, CASES[0]!, CASES[1]!];
  const { start, end, faces } = layout(cases), N = start.length / 3;
  // Case 2: the vertex three times as heavy as its face's corners. Case 3: the face pinned.
  const mass = new Array<number>(N).fill(1e-4), groups = new Array<number>(N).fill(0);
  mass[12] = 3e-4; groups[21] = groups[22] = groups[23] = 1;
  const h = await standalone(device, start, end, faces, mass, groups, { rounds: 1, impactIterations: 200 });
  h.icloth.encodeFrameBegin();
  h.submit(pass => { h.icloth.stepHooks.afterFrame(pass); h.icloth.stepHooks.beforeFinish(pass); });
  const got = await h.read(), counters = await h.icloth.readCounters(), impacts = oracleImpacts(start, end, cases.length);
  const want = iclothZoneSolveOracle(Float64Array.from(end), impacts, { mass, pinned: groups, iterations: 200, stepThreshold: TOL.stepThreshold, feasible: 0.1 * TOL.delta });
  let worst = 0, moved = 0;
  for (let i = 0; i < 3 * N; i++) { worst = Math.max(worst, Math.abs(got[i]! - want.positions[i]!)); moved = Math.max(moved, Math.abs(want.positions[i]! - end[i]!)); }
  const C = (x: Float64Array, c: IcImpact) => c.ids.reduce((s, v, i) => s + c.w[i]! * (c.n[0] * x[3 * v]! + c.n[1] * x[3 * v + 1]! + c.n[2] * x[3 * v + 2]!), 0) - c.d0;
  const violation = Math.max(...impacts.map(c => -C(got, c)));
  console.log(`impact zones: ${impacts.length} impacts, GPU ${counters.impactIterations} iterations, oracle ${want.iterations}; largest move ${moved.toExponential(3)} m; `
    + `GPU against oracle ${worst.toExponential(3)} m; GPU violation ${violation.toExponential(3)} m, oracle ${want.maxViolation.toExponential(3)} m; left by the closing detection: ${counters.remainingPairs}`);
  assert.equal(counters.status & ~ICLOTH_STATUS.iterationCap, 0, IClothCollision.describe(counters.status).join('; '));
  assert.deepEqual(counters.impactsPerRound, [impacts.length]);
  // Tolerance: the solve's feasibility tolerance, 10% of the target separation delta (1e-5 m at r = 1 mm): each stops
  // once its descent stalls with every impact within that of its plane, on f32 against f64 sums. The moves are 3 mm.
  assert.ok(worst < 0.1 * TOL.delta, `GPU against oracle ${worst} m`);
  assert.ok(violation < 0.1 * TOL.delta, `GPU violation ${violation} m`);
  assert.ok(counters.cappedSolves === 0 && !want.capped, `${counters.cappedSolves} GPU solves at the cap; oracle capped ${want.capped}`);
  // The pinned face has not moved.
  for (let v = 21; v < 24; v++) for (let a = 0; a < 3; a++) assert.equal(got[3 * v + a], end[3 * v + a]);
  assert.equal(counters.remainingPairs, 0, 'the closing detection finds the solved pairs clear');
  h.destroy();
}));

// ------------------------------------------------------------------ end to end
const rotateInverse = (q: readonly [number, number, number, number], v: [number, number, number]): [number, number, number] => {
  // v' = q* v q for the unit quaternion (x, y, z, w).
  const [x, y, z, w] = [-q[0], -q[1], -q[2], q[3]];
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
};
/** A 16 x 16 sheet (2.7 cm edges) thrown down at 2 m/s from 4 cm above a sphere that slides and turns under it. */
function thrownAtSphere(solver: ClothDocument['solver'], valid = true) {
  const Rs = 0.1, y = 0.76, ticks = 12, end_s = ticks * CLOTH_SCENE_STEP_S;
  const scene = createEmptyScene({ sceneId: 'icloth-solid-test', finestCellSize_m: 0.01, extents_m: { x: 1.2, y: 1.6, z: 1.2 } });
  const grid = clothGridData({ columns: 16, rows: 16, width_m: 0.4, height_m: 0.4, origin_m: [-0.2, 0.9, -0.2] });
  const data: ClothTopologyData = { ...grid, initialVelocities_m_s: Array.from({ length: grid.restPositions_m.length }, (_, i) => i % 3 === 1 ? -2 : 0) };
  scene.cloth = [{ schemaVersion: '2.0.0', id: 'sheet', geometry: { kind: 'inline', data }, materials: [cotton], solver, pendingSolids: [{ id: 'turner', start_s: 0 }] }];
  const at = (x: number) => ({ x, y, z: 0 }), zero = { x: 0, y: 0, z: 0 }, q = (a: number) => ({ w: Math.cos(a / 2), x: 0, y: Math.sin(a / 2), z: 0 });
  const body: SceneDescription['rigidBodies'][number] = { id: 'turner', name: 'turner', shape: 'sphere', dimensions_m: { x: Rs, y: Rs, z: Rs }, density_kg_m3: 1000, position_m: at(0), orientation: q(0),
    linearVelocity_m_s: zero, angularVelocity_rad_s: zero, restitution: 0, friction: 0.5, motion: 'static',
    poseTrack: { keyframes: [0, 0.5, 1].map(f => ({ time_s: f * end_s, position_m: at(f * 0.03), orientation: q(f * 0.6) })) } };
  scene.rigidBodies = [body];
  if (valid) assert.deepEqual(validateScene(scene), []);
  const X = grid.restPositions_m, origin = [0, 1, 2].map(a => { let lo = Infinity, hi = -Infinity; for (let i = a; i < X.length; i += 3) { lo = Math.min(lo, X[i]!); hi = Math.max(hi, X[i]!); } return (lo + hi) / 2; }) as [number, number, number];
  const field = buildClothBodyField(scene, body, clothContactRadius(scene.cloth[0]!));
  /** Vertices inside the moving solid at scene time t, and the least field value (m), by the Float64 field sampler. */
  const inside = (x: ArrayLike<number>, t: number) => {
    const pose = clothBodyPoses(body, origin, t, 0, CLOTH_SCENE_STEP_S)[0]!;
    let count = 0, least = Infinity;
    for (let v = 0; v < x.length / 3; v++) {
      const p = rotateInverse(pose.orientation, [x[3 * v]! - origin[0] - pose.position[0], x[3 * v + 1]! - origin[1] - pose.position[1], x[3 * v + 2]! - origin[2] - pose.position[2]]);
      const phi = sampleClothSolidField(field, p).phi;
      least = Math.min(least, phi); if (phi < 0) count++;
    }
    return { count, least };
  };
  return { scene, ticks, inside };
}
async function run(device: GPUDevice, scene: SceneDescription, ticks: number, options: ClothSystemOptions, each?: (tick: number, x: Float64Array, cloth: ClothSystem) => Promise<void> | void) {
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1, ...options }), id = scene.cloth![0]!.id, solver = cloth.solverFor(id)!;
  for (let tick = 1; tick <= ticks; tick++) {
    assert.ok(cloth.stepTo(cloth.time_s + CLOTH_SCENE_STEP_S), `step ${tick} refused: ${cloth.failure?.message ?? 'pending'}`);
    await device.queue.onSubmittedWorkDone();
    if (each) await each(tick, await solver.readPositions(), cloth);
  }
  await device.queue.onSubmittedWorkDone();
  return cloth;
}

test("3a. method 'icloth': a sheet thrown at the moving solid ends every step outside it; the same run without the collision handling does not", { skip: !gpu }, async () => withDevice(async device => {
  // The whole method against none of it: with the repulsion and no rounds the solve alone holds most of the solid,
  // and with the rounds and no repulsion the stopped patch stretches the falling sheet by 38%, 75%, 104% a step
  // (measured; nothing in an impact-zone solve resists stretch), which is not the method.
  const report = async (collisions: boolean) => {
    const { scene, ticks, inside } = thrownAtSphere({ method: 'icloth' });
    let worst = 0, least = Infinity, impacts = 0, rounds = 0, left = 0;
    const cloth = await run(device, scene, ticks, { iclothStages: { proximity: collisions, collisions } }, async (tick, x, c) => {
      const at = inside(x, tick * CLOTH_SCENE_STEP_S), n = await c.iclothFor('sheet')!.readCounters();
      worst = Math.max(worst, at.count); least = Math.min(least, at.least);
      impacts += n.impactsPerRound.reduce((a, b) => a + b, 0); rounds = Math.max(rounds, n.rounds); left += n.remainingSolid + n.remainingPairs;
    });
    const failure = cloth.failure?.message, notes = [...cloth.notes.keys()];
    cloth.destroy();
    return { worst, least, impacts, rounds, left, failure, notes };
  };
  const on = await report(true), off = await report(false);
  console.log(`thrown at the sphere, collision handling on: most vertices inside ${on.worst}, least field ${on.least.toExponential(3)} m, ${on.impacts} impacts, at most ${on.rounds} rounds, ${on.left} left; notes ${JSON.stringify(on.notes)}`);
  console.log(`thrown at the sphere, collision handling off: most vertices inside ${off.worst}, least field ${off.least.toExponential(3)} m, ${off.left} impacts left by the closing detection`);
  assert.equal(on.failure, undefined, on.failure);
  assert.equal(on.worst, 0, 'no vertex inside the solid after any step');
  assert.ok(on.impacts > 0 && on.left === 0, 'the rounds found impacts and left none');
  assert.ok(off.worst > 0 && off.left > 0, 'without the rounds the sheet passes into the solid, and the closing detection says so');
  assert.equal(off.failure, undefined, 'with the rounds switched off the closing detection only counts');
}));

/** One document of two 12 x 12 sheets: the lower flat and pinned at its corners, the upper tilted 30 degrees, its low edge 1 cm above, thrown down at 3 m/s. */
function thrownThroughSheet(solver: ClothDocument['solver']): SceneDescription {
  const scene = createEmptyScene({ sceneId: 'icloth-self-test', finestCellSize_m: 0.01, extents_m: { x: 1.2, y: 1.6, z: 1.2 } });
  const lower = clothGridData({ columns: 12, rows: 12, width_m: 0.3, height_m: 0.3, origin_m: [-0.15, 0.8, -0.15], group: (i, j) => (i === 0 || i === 11) && (j === 0 || j === 11) ? 1 : 0 });
  const upper = clothGridData({ columns: 12, rows: 12, width_m: 0.2, height_m: 0.2, origin_m: [-0.1, 0.81, -0.1], weft: [0, 0.5, Math.sqrt(0.75)] });
  const base = lower.restPositions_m.length / 3;
  const data: ClothTopologyData = { restPositions_m: [...lower.restPositions_m, ...upper.restPositions_m], triangles: [...lower.triangles, ...upper.triangles.map(i => i + base)],
    uv: [...(lower.uv ?? []), ...(upper.uv ?? [])], groups: [...(lower.groups ?? []), ...(upper.groups ?? [])],
    initialVelocities_m_s: [...lower.restPositions_m.map(() => 0), ...upper.restPositions_m.map((_, i) => i % 3 === 1 ? -3 : 0)] };
  scene.cloth = [{ schemaVersion: '2.0.0', id: 'sheets', geometry: { kind: 'inline', data }, materials: [cotton], solver }];
  assert.deepEqual(validateScene(scene), []);
  return scene;
}

test("3b. method 'icloth': a sheet thrown through another is stopped with no self-intersection after any step; without the collision rounds it cuts through", { skip: !gpu }, async () => withDevice(async device => {
  const report = async (collisions: boolean) => {
    const scene = thrownThroughSheet({ method: 'icloth' }), faces = scene.cloth![0]!.geometry.data.triangles;
    const crossings: number[] = [];
    let impacts = 0, rounds = 0, iterations = 0, left = 0;
    const cloth = await run(device, scene, 8, { iclothStages: { collisions } }, async (_tick, x, c) => {
      crossings.push(iclothSelfIntersections(x, faces));
      const n = await c.iclothFor('sheets')!.readCounters();
      impacts += n.impactsPerRound.reduce((a, b) => a + b, 0); rounds = Math.max(rounds, n.rounds); iterations = Math.max(iterations, n.impactIterations); left += n.remainingPairs;
    });
    const failure = cloth.failure?.message, notes = [...cloth.notes.keys()];
    cloth.destroy();
    return { crossings, impacts, rounds, iterations, left, failure, notes };
  };
  const on = await report(true), off = await report(false);
  console.log(`thrown through a sheet, collision rounds on: self-intersections per step ${on.crossings.join(' ')}; ${on.impacts} impacts, at most ${on.rounds} rounds and ${on.iterations} impact-zone iterations a step, ${on.left} left; notes ${JSON.stringify(on.notes)}`);
  console.log(`thrown through a sheet, collision rounds off: self-intersections per step ${off.crossings.join(' ')}; ${off.left} impacts left by the closing detection`);
  assert.equal(on.failure, undefined, on.failure);
  assert.deepEqual(on.crossings, on.crossings.map(() => 0), 'no self-intersection after any step');
  assert.ok(on.impacts > 0 && on.left === 0, 'the rounds found impacts and left none');
  assert.ok(off.crossings.some(n => n > 0) && off.left > 0, 'without the rounds the sheets cross, and the closing detection says so');
}));

test("3c. the guard: a step whose closing detection still finds impacts halts the stage on the GPU, and the scene when its status is read", { skip: !gpu }, async () => withDevice(async device => {
  // One round of one iteration cannot stop the thrown sheet. The host reads the status only every 6 steps here, so
  // steps after the failing one are encoded: they must do nothing (the step counter, bumped by the first kernel of a
  // step, stands still), and the failing step's counts stay readable.
  const scene = thrownThroughSheet({ method: 'icloth', icloth: { rounds: 1, impactIterations: 1 } });
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 6 }), stage = cloth.iclothFor('sheets')!;
  const seen: { tick: number; steps: number; halted: boolean; left: number; status: number }[] = [];
  let tick = 0;
  for (; tick < 14 && cloth.stepTo(cloth.time_s + CLOTH_SCENE_STEP_S); tick++) {
    await device.queue.onSubmittedWorkDone();
    const n = await stage.readCounters();
    seen.push({ tick: tick + 1, steps: n.steps, halted: n.halted, left: n.remainingPairs + n.remainingSolid, status: n.status });
  }
  await device.queue.onSubmittedWorkDone();
  const first = seen.find(s => s.halted);
  console.log(`guard: ${seen.map(s => `${s.tick}:${s.steps}${s.halted ? 'H' : ''}/${s.left}`).join(' ')}; failure: ${cloth.failure?.message}`);
  assert.ok(first, 'the stage halted');
  assert.ok(first.left > 0 && (first.status & ICLOTH_STATUS.roundCap) !== 0, 'at a step whose closing detection found impacts');
  assert.equal(first.steps, first.tick, 'in that very step');
  for (const s of seen) {
    if (s.tick < first.tick) assert.ok(!s.halted && s.left === 0);
    else assert.deepEqual([s.steps, s.left, s.halted], [first.steps, first.left, true], 'nothing ran after the halt');
  }
  assert.match(cloth.failure?.message ?? '', /impacts remained after the last collision round/);
  assert.ok(seen.length < 14, 'the scene stopped at the next status read');
  cloth.destroy();
}));

test("4. method 'icloth': a pose saved mid-run steps on as the straight run does; a 'newton' document runs the OGC detector and no I-Cloth stage", { skip: !gpu }, async () => withDevice(async device => {
  const { scene } = thrownAtSphere({ method: 'icloth' }), dt = CLOTH_SCENE_STEP_S;
  const straight = await ClothSystem.create(device, scene, { statusInterval: 1 });
  assert.ok(straight.iclothFor('sheet') && !straight.contactFor('sheet'));
  await straight.capture([5 * dt]);
  publishSceneClothSurfaces(scene, straight.surfaces);
  const saved = snapshotSceneClothPose(scene);
  assert.deepEqual(validateScene(saved), []);
  const [a1, a] = await straight.capture([6 * dt, 9 * dt]);
  assert.equal(straight.failure, undefined, straight.failure?.message);
  // The document's integrator (BDF2 by default) is the method's too; the stage's contact state restarts its history
  // at a contact onset as the OGC detector's does (some steps of the nine, not all: the sheet starts in free flight).
  const restarts = (await straight.solverFor('sheet')!.readStatus()).restarts;
  console.log(`BDF2 restarts over 9 steps onto the sphere: ${restarts}`);
  assert.ok(restarts >= 1 && restarts < 9, `BDF2 restarts ${restarts}`);
  const resumed = await ClothSystem.create(device, saved, { statusInterval: 1 });
  assert.ok(Math.abs(resumed.time_s - 5 * dt) < 1e-9, `the resumed clock starts at ${resumed.time_s} s`);
  const [b1, b] = await resumed.capture([6 * dt, 9 * dt]);
  assert.equal(resumed.failure, undefined, resumed.failure?.message);
  let worst = 0;
  const x = a![0]!.positions_m, y = b![0]!.positions_m;
  for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(x[i]! - y[i]!));
  let first = 0;
  const x1 = a1![0]!.positions_m, y1 = b1![0]!.positions_m;
  for (let i = 0; i < x1.length; i++) first = Math.max(first, Math.abs(x1[i]! - y1[i]!));
  console.log(`save/load: largest position difference ${first.toExponential(3)} m after 1 step, ${worst.toExponential(3)} m after 4 steps in contact from the saved pose`);
  // The saved pose carries the whole state (the solver's BDF2 history and duals, the stage's contact state): the next
  // step is the straight run's to f32 rounding. Later steps append their records in the GPU's scheduling order, so f32
  // sums differ in their last bits between any two runs, and a vertex at the edge of contact can then flip a BDF2
  // restart (two straight runs of this scene differ by 8e-6 m or 2.9e-4 m after these steps, measured): the bound
  // there is the cloth's thickness r.
  assert.ok(first < 1e-6, `resumed against straight after one step ${first} m`);
  assert.ok(worst < 5e-4, `resumed against straight ${worst} m`);
  straight.destroy(); resumed.destroy();
  const plain = thrownAtSphere({ integrator: 'bdf2', contactPlaneRecordsPerVertex: 200 }).scene;
  const newton = await ClothSystem.create(device, plain, { statusInterval: 1 });
  assert.ok(newton.contactFor('sheet') && !newton.iclothFor('sheet'));
  newton.destroy();
}));

test("5. an unknown cloth method fails explicitly when the document is built; 'icloth' refuses the hybrid trust region", { skip: !gpu }, async () => withDevice(async device => {
  const unknown = thrownAtSphere({ method: 'verlet' as unknown as 'icloth' }, false).scene;
  await assert.rejects(ClothSystem.create(device, unknown), /unknown cloth method 'verlet'/);
  const hybrid = thrownAtSphere({ method: 'icloth', newton: { iterations: 4, pcgIterations: 50, trustRegion: 'hybrid' } }, false).scene;
  assert.ok(validateScene(hybrid).some(e => /icloth.*hybrid trust region/.test(e)), 'scene validation names it too');
  await assert.rejects(ClothSystem.create(device, hybrid), /method 'icloth' takes the line-search Newton/);
  assert.equal(ICLOTH_FAILING & ICLOTH_STATUS.roundCap, ICLOTH_STATUS.roundCap, 'a step that ends with impacts left halts');
  assert.equal(ICLOTH_KIND.body, 3);
}));

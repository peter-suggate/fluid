/**
 * Two-level correction (lib/cloth/coarse.ts, gpu-coarse-wgsl.ts): the topology coarse space (connected
 * components, cuts, partition of unity, explicit failures), its Float64 reference against plain VBD, and the GPU
 * path against the reference. GPU tests skip without WEBGPU_NODE_MODULE.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { clothGridData } from '../lib/cloth/grid';
import { clothVertexNeighbours, colourClothVertices, prepareClothTopology } from '../lib/cloth/topology';
import type { ClothMaterial, ClothTopologyData } from '../lib/cloth/types';
import { buildClothSolverModel, clothMechanicalEnergy, clothOracleStep, createClothOracleState, type ClothStepOptions } from '../lib/cloth/oracle';
import { buildClothCoarseSpace, ClothCoarseCorrection, packClothCoarseTables } from '../lib/cloth/coarse';
import { ClothGPUSolver } from '../lib/cloth/gpu-solver';
import { CLOTH_BINDING, CLOTH_SEGMENT_ALIGN } from '../lib/cloth/gpu-solver-wgsl';
import { CLOTH_COARSE_MISC } from '../lib/cloth/gpu-coarse-wgsl';
import { herringBoneSheet } from '../lib/cloth/scenes/bridson-2002/mesh';
import { CLOTH } from '../lib/cloth/scenes/bridson-2002/parameters';

const gpu = !!process.env.WEBGPU_NODE_MODULE;

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

const cotton: ClothMaterial = { id: 'cotton', arealDensity_kg_m2: 0.2, stretchWarp_N_m: 400, stretchWeft_N_m: 250, shear_N_m: 40, poisson: 0.3,
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0.002, thickness_m: 5e-4, strainLimit: 0.02 };
const g: [number, number, number] = [0, -9.81, 0];
const origin: [number, number, number] = [0.25, 1, 0.25];

/** A sheet hanging from its two near corners; `slit` removes the faces across the middle column below row `slit`. */
function sheet(columns: number, slit?: number): ClothTopologyData {
  const data = clothGridData({ columns, rows: columns, width_m: 0.5, height_m: 0.5, origin_m: [0, 1, 0], group: (i, j) => j === 0 && (i === 0 || i === columns - 1) ? 1 : 0 });
  if (slit === undefined) return data;
  const f = data.triangles, keep: number[] = [], mid = Math.floor(columns / 2);
  for (let k = 0; k < f.length; k += 3) {
    const vs = [f[k]!, f[k + 1]!, f[k + 2]!], cols = vs.map(v => v % columns), rows = vs.map(v => Math.floor(v / columns));
    if (!(Math.min(...cols) < mid && Math.max(...cols) >= mid && Math.min(...rows) >= slit)) keep.push(...vs);
  }
  return { ...data, triangles: keep };
}
const model = (data: ClothTopologyData) => {
  const t = prepareClothTopology(data, [cotton]);
  return buildClothSolverModel(t, [cotton], colourClothVertices(clothVertexNeighbours(t)));
};

test('the coarse space follows the topology: a partition of unity per vertex, and every row inside its piece', () => {
  const whole = prepareClothTopology(sheet(12), [cotton]), space = buildClothCoarseSpace(whole, 9);
  assert.equal(space.nodeCount, 9); assert.equal(space.componentCount, 1);
  for (let v = 0; v < whole.vertexCount; v++) {
    let sum = 0;
    for (let p = space.rowOffsets[v]!; p < space.rowOffsets[v + 1]!; p++) { assert.ok(space.rowWeights[p]! > 0); sum += space.rowWeights[p]!; }
    assert.ok(Math.abs(sum - 1) < 1e-12, `vertex ${v} weights sum to ${sum}`);
  }
  // A slit through the whole sheet makes two pieces: each gets nodes and no row crosses the cut.
  const cut = prepareClothTopology(sheet(12, 0), [cotton]), pieces = buildClothCoarseSpace(cut, 9);
  assert.equal(pieces.componentCount, 2);
  const nodeComponent = Array.from(pieces.seeds, s => pieces.component[s]!);
  assert.deepEqual([...new Set(nodeComponent)].sort(), [0, 1]);
  for (let v = 0; v < cut.vertexCount; v++)
    for (let p = pieces.rowOffsets[v]!; p < pieces.rowOffsets[v + 1]!; p++) assert.equal(nodeComponent[pieces.rowNodes[p]!], pieces.component[v]);
  assert.throws(() => buildClothCoarseSpace(cut, 1), /2 connected pieces; its coarse space holds 1 nodes/);
  assert.throws(() => buildClothCoarseSpace(cut, 0), /at least one node/);
  // The GPU tables sum each A_c term once: every vertex and every edge has exactly one owning face.
  const tables = packClothCoarseTables(cut, pieces);
  let owned = 0, ownedEdges = 0;
  for (let f = 0; f < cut.faceCount; f++) {
    owned += [0, 1, 2].filter(k => tables.faceOwn[4 * f]! & (1 << k)).length;
    ownedEdges += [1, 2, 3].filter(k => tables.faceOwn[4 * f + k] !== 0xffffffff).length;
  }
  assert.equal(owned, cut.vertexCount); assert.equal(ownedEdges, cut.edgeCount);
});

test('free fall is exact in one iteration with the coarse step: the partition of unity carries the translation', () => {
  const data = clothGridData({ columns: 10, rows: 10, width_m: 0.5, height_m: 0.5, origin_m: [0, 1, 0] });
  const m = model(data), space = buildClothCoarseSpace(m.topology, 6);
  const run = (initialGuess: 'previous' | 'inertial', coarse?: ClothCoarseCorrection) => {
    const st = createClothOracleState(m, data.restPositions_m);
    for (let s = 0; s < 10; s++) clothOracleStep(m, st, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 1, chebyshevRho: 0, initialGuess }, false, undefined, coarse);
    return st.x;
  };
  const exact = run('inertial'), lagging = run('previous'), twoLevel = run('previous', new ClothCoarseCorrection(m, space));
  const diff = (a: Float64Array, b: Float64Array) => a.reduce((w, x, i) => Math.max(w, Math.abs(x - b[i]!)), 0);
  assert.ok(diff(lagging, exact) > 1e-3, `plain VBD from the previous pose lags free fall by only ${diff(lagging, exact)} m`);
  assert.ok(diff(twoLevel, exact) < 1e-9, `two-level free fall is off by ${diff(twoLevel, exact)} m`);
});

test('a cut sheet keeps its energy bounded, its strain near the limit and its swing on a converged run with the two-level correction at K = 8', () => {
  // Plain VBD at K = 8 fails one way or the other here: from the adaptive guess the energy grows (5.7 % strain), from
  // the previous pose the sheet hangs back (mean height 13 cm above the converged run's after 20 steps).
  const m = model(sheet(16, 4)), space = buildClothCoarseSpace(m.topology, 12), coarse = new ClothCoarseCorrection(m, space);
  const st = createClothOracleState(m, m.topology.data.restPositions_m), converged = createClothOracleState(m, m.topology.data.restPositions_m);
  const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 8, chebyshevRho: 0, initialGuess: 'previous' };
  const reference = new ClothCoarseCorrection(m, space);
  const meanY = (x: Float64Array) => x.reduce((sum, v, i) => i % 3 === 1 ? sum + v : sum, 0) / m.topology.vertexCount;
  const H0 = clothMechanicalEnergy(m, st.x, st.v, g).total;
  let worst = 0, Hmax = -Infinity, lag = 0;
  for (let s = 0; s < 60; s++) {
    worst = Math.max(worst, clothOracleStep(m, st, options, false, undefined, coarse).maxStrain);
    clothOracleStep(m, converged, { ...options, iterations: 32 }, false, undefined, reference);
    Hmax = Math.max(Hmax, clothMechanicalEnergy(m, st.x, st.v, g).total);
    lag = Math.max(lag, Math.abs(meanY(st.x) - meanY(converged.x)));
  }
  console.log(`cut sheet 16^2, K = 8: max strain ${(100 * worst).toFixed(2)} %, H0 ${H0.toFixed(5)} J, max H ${Hmax.toFixed(5)} J, mean height off K = 32 by ${(1000 * lag).toFixed(1)} mm, ${JSON.stringify(coarse.stats)}`);
  assert.ok(lag < 0.02, `mean height off the converged run by ${lag} m`);
  assert.ok(Hmax <= H0 + 1e-9 * Math.abs(H0), `energy grew from ${H0} to ${Hmax} J`);
  assert.ok(worst < 0.04, `strain ${worst}`);
  assert.equal(coarse.stats.indefinite, 0);
});

test('one GPU step with the two-level correction matches its Float64 reference', { skip: !gpu }, async () => withDevice(async device => {
  const m = model(sheet(8)), space = buildClothCoarseSpace(m.topology, 6);
  let seed = 12345;
  const r = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32 - 0.5);
  // A 4 cm perturbation is far enough from equilibrium that the full coarse step is rejected: the shorter trials run.
  for (const [amplitude, iterations, every] of [[4e-3, 1, 1], [4e-3, 4, 1], [4e-3, 4, 2], [4e-2, 4, 1]] as const) {
    seed = 12345;
    const start = Float64Array.from(m.topology.data.restPositions_m, (p, i) => Math.fround(p + amplitude * r() - origin[i % 3]!) + origin[i % 3]!);
    const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations, chebyshevRho: 0 };
    const solver = await ClothGPUSolver.create(device, m, { ...options, origin_m: origin, positions_m: start, coarse: space, coarseEvery: every });
    const oracle = createClothOracleState(m, start), coarse = new ClothCoarseCorrection(m, space, 2 ** -23, false, every);
    clothOracleStep(m, oracle, options, false, undefined, coarse);
    solver.step();
    let d = 0;
    const x = await solver.readPositions();
    for (let i = 0; i < x.length; i++) d = Math.max(d, Math.abs(x[i]! - oracle.x[i]!));
    const stats = await solver.readCoarseStats();
    console.log(`perturbation ${amplitude} m, K=${iterations} every ${every}: GPU vs reference ${d.toExponential(2)} m; GPU ${JSON.stringify(stats)}; reference ${JSON.stringify(coarse.stats)}`);
    assert.equal(stats!.corrections, Math.ceil(iterations / every));
    assert.deepEqual(stats.accepted, coarse.stats.accepted);
    if (amplitude > 1e-2) assert.ok(stats.accepted[0]! < stats.corrections, 'some full step is rejected');
    assert.ok(d < 1e-5, `K=${iterations}: GPU vs reference ${d} m`);
    solver.destroy();
  }
}));

test('the GPU coarse factor inverts blocks of one stiff contact direction over soft ones (k / s = 1e5 off the axes)', { skip: !gpu }, async () => withDevice(async device => {
  // A contact Hessian puts k n n^T on a node block over its soft s I. The block's determinant s^2 (s + k) cancels
  // from terms of k^3 in f32, so a determinant test called this positive definite block indefinite.
  const m = model(sheet(8)), space = buildClothCoarseSpace(m.topology, 6), M = 3 * space.nodeCount;
  const solver = await ClothGPUSolver.create(device, m, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 1, chebyshevRho: 0, origin_m: origin, coarse: space });
  const raw = solver as unknown as { coarse: GPUBuffer; T: number; tables: { coarse: { miscRecord: number } }; run(pass: GPUComputePassEncoder, name: string, workgroups: number): void; read(b: GPUBuffer, size: number): Promise<ArrayBuffer> };
  const align = (x: number) => Math.ceil(x / CLOTH_SEGMENT_ALIGN) * CLOTH_SEGMENT_ALIGN, aRecord = align(6 * raw.T), wRecord = align(aRecord + align(Math.ceil(M * M / 4)));
  const A = new Float64Array(M * M), s0 = 1, k = 1e5;
  for (let c = 0; c < space.nodeCount; c++) {
    const n = [1, (-1) ** c, 1 + 0.1 * c], l = Math.hypot(...n);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) A[(3 * c + i) * M + 3 * c + j] = (i === j ? s0 : 0) + k * n[i]! * n[j]! / (l * l);
    // A soft coupling to the next node keeps the sweep's Schur updates in play.
    if (c + 1 < space.nodeCount) for (let i = 0; i < 3; i++) { A[(3 * c + i) * M + 3 * (c + 1) + i] = -0.25 * s0; A[(3 * (c + 1) + i) * M + 3 * c + i] = -0.25 * s0; }
  }
  device.queue.writeBuffer(raw.coarse, 16 * aRecord, Float32Array.from(A));
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  raw.run(pass, 'coarseFactor', 1);
  pass.end(); device.queue.submit([encoder.finish()]);
  const words = new Float32Array(await raw.read(raw.coarse, 16 * (raw.tables.coarse.miscRecord + 1)));
  assert.equal(words[4 * raw.tables.coarse.miscRecord + CLOTH_COARSE_MISC.ok], 1, 'the factor called a positive definite A_c indefinite');
  assert.equal((await solver.readStatus()).status, 0);
  // A^-1 A = I in its soft directions too (columns of size 1 / s).
  const W = words.subarray(4 * wRecord, 4 * wRecord + M * M);
  let worst = 0;
  for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) {
    let p = 0;
    for (let q = 0; q < M; q++) p += W[i * M + q]! * A[q * M + j]!;
    worst = Math.max(worst, Math.abs(p - (i === j ? 1 : 0)));
  }
  console.log(`stiff blocks k / s = ${k / s0}: |A^-1 A - I| ${worst.toExponential(2)}`);
  assert.ok(worst < 0.05, `A^-1 A off the identity by ${worst}`);
  solver.destroy();
}));

// A stand-in for C2's vertex contact block: vertex v with k_v > 0 carries the spring (k_v / 2) (n_v . (x - x0_v))^2,
// one stiff direction per vertex (the Gauss-Newton form of a barrier spike). Records per vertex: (n, k), (x0, 0).
const STIFF_SPRING_CONTACT_WGSL = /* wgsl */`
@group(0) @binding(${CLOTH_BINDING.contact}) var<storage, read> springs: array<vec4f>;
fn ogcContactCount(v: u32) -> u32 { return select(0u, 1u, springs[2u * v].w > 0.0); }
fn ogcVertexContactLanes(v: u32, xv: vec3f, first: u32, stride: u32, f: ptr<function, vec3f>, H: ptr<function, mat3x3f>) -> f32 {
  let s = springs[2u * v];
  if (first != 0u || !(s.w > 0.0)) { return 0.0; }
  let c = dot(s.xyz, xv - springs[2u * v + 1u].xyz);
  *f -= s.w * c * s.xyz; *H += s.w * mat3x3f(s.xyz * s.x, s.xyz * s.y, s.xyz * s.z);
  return 0.5 * s.w * c * c;
}
fn ogcVertexContactEnergyLanes(v: u32, xv: vec3f, first: u32, stride: u32) -> f32 {
  var f = vec3f(0.0); var H = mat3x3f(); return ogcVertexContactLanes(v, xv, first, stride, &f, &H);
}
fn ogcAccumulateVertexContact(v: u32, xv: vec3f, f: ptr<function, vec3f>, H: ptr<function, mat3x3f>) { _ = ogcVertexContactLanes(v, xv, 0u, 1u, f, H); }
fn ogcVertexContactEnergy(v: u32, xv: vec3f) -> f32 { return ogcVertexContactEnergyLanes(v, xv, 0u, 1u); }
`;

test('a contact spike far past f32\'s range (k / s = 1e10, one off-axis direction per coarse node) neither halts the step nor stops its coarse corrections', { skip: !gpu }, async () => withDevice(async device => {
  // The curtain halted at a contact spike of k / s 4e9 at one vertex (coarseIndefinite, RESULT 18): A_c held, per
  // node, one direction 1e9 stiffer than the soft ones, which no f32 sum or sweep resolves. The fine solve survives it.
  // The coarse model now takes such a vertex's contact scaled into f32's range, Hessian and force alike (its minimizer
  // kept), so the factor runs and its step along the stiff direction stays the Newton one: corrections are accepted.
  const m = model(sheet(8)), space = buildClothCoarseSpace(m.topology, 6), n = m.topology.vertexCount, h = (2 / 3) / 60;
  const springs = new Float32Array(8 * n), rest = m.topology.data.restPositions_m;
  for (let c = 0; c < space.nodeCount; c++) {
    // The spike sits on the node's seed (its largest weight), across the sheet at a different slant per node.
    const v = space.seeds[c]!, d = [1, 0.3 * (c - 2), 0.7], l = Math.hypot(...d);
    springs.set([d[0]! / l, d[1]! / l, d[2]! / l, 1e10 * m.topology.mass_kg[v]! / (h * h)], 8 * v);
    springs.set([rest[3 * v]! - origin[0], rest[3 * v + 1]! - origin[1], rest[3 * v + 2]! - origin[2], 0], 8 * v + 4);
  }
  const buffer = device.createBuffer({ size: springs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, springs);
  const solver = await ClothGPUSolver.create(device, m, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 4, chebyshevRho: 0, origin_m: origin,
    coarse: space, contact: { wgsl: STIFF_SPRING_CONTACT_WGSL, buffer }, hooks: { afterCoarse: () => {} } });
  solver.step(6);
  const status = await solver.readStatus(), stats = await solver.readCoarseStats(), x = await solver.readPositions();
  assert.deepEqual(ClothGPUSolver.describeStatus(status.status), [], 'the spike halted the step');
  assert.equal(stats.corrections, 6 * 2, 'every coarse slot ran');
  const accepted = stats.accepted.reduce((s, a) => s + a, 0);
  console.log(`spike k / s = 1e10: ${stats.corrections} corrections, accepted at 1, 1/2, 1/4, 1/8: ${stats.accepted.join(', ')}`);
  assert.ok(accepted > 0, 'no coarse correction was accepted');
  // The springs hold their vertices along n (to the f32 resolution of 1e10 s against gravity's pull).
  let slip = 0;
  for (let c = 0; c < space.nodeCount; c++) {
    const v = space.seeds[c]!, s = springs.subarray(8 * v, 8 * v + 8);
    slip = Math.max(slip, Math.abs(s[0]! * (x[3 * v]! - rest[3 * v]!) + s[1]! * (x[3 * v + 1]! - rest[3 * v + 1]!) + s[2]! * (x[3 * v + 2]! - rest[3 * v + 2]!)));
  }
  assert.ok(x.every(Number.isFinite) && slip < 1e-4, `spiked vertices slipped ${slip} m along their normals`);
  solver.destroy(); buffer.destroy();
}));

test('a GPU two-level run tracks its reference over many steps, and its saved state replays', { skip: !gpu }, async () => withDevice(async device => {
  const m = model(sheet(8, 3)), space = buildClothCoarseSpace(m.topology, 8);
  const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 6, chebyshevRho: 0, initialGuess: 'previous' };
  const solver = await ClothGPUSolver.create(device, m, { ...options, origin_m: origin, coarse: space });
  const oracle = createClothOracleState(m, m.topology.data.restPositions_m), coarse = new ClothCoarseCorrection(m, space, 2 ** -23);
  const steps = 16;
  for (let s = 0; s < steps; s++) clothOracleStep(m, oracle, options, false, undefined, coarse);
  solver.step(steps);
  const x = await solver.readPositions();
  let d = 0;
  for (let i = 0; i < x.length; i++) d = Math.max(d, Math.abs(x[i]! - oracle.x[i]!));
  console.log(`${steps} steps: GPU vs reference ${d.toExponential(2)} m (sheet drop ${(0.5 - Math.min(...Array.from(x).filter((_, i) => i % 3 === 1)) + 0.5).toFixed(3)} m)`);
  assert.ok(d < 5e-4, `trajectory ${d} m`);
  assert.equal((await solver.readStatus()).status, 0);
  const saved = await solver.readState();
  solver.step(5);
  const reference = await solver.readPositions();
  solver.uploadState(saved.positions_m, saved.velocities_m_s, saved.solverState);
  solver.step(5);
  const replay = await solver.readPositions();
  assert.ok(replay.every((v, i) => Math.abs(v - reference[i]!) < 1e-6), 'replay from the saved state differs');
  solver.destroy();
}));

// Bulk strain must match the static equilibrium, not merely sit under the strain limit. The Bridson sheet (150 nodes
// across, 1 m by 1.7 m, Bridson material) hangs from its whole top edge. With Poisson 0 and area-lumped mass, the
// exact equilibrium of band j (rows j, j + 1) is uniaxial StVK: k lambda (lambda^2 - 1) / 2 = rho g d, where d is
// the band middle's height above the bottom row. Top band 7.46%, middle 3.90%. That is rho g L / k = 4.9% per metre
// of hanging cloth below. At K = 8 with the two-level correction the sheet reaches it in 10 s.
test('a hanging Bridson sheet comes to rest at its exact uniaxial equilibrium at K = 8 with the two-level correction', { skip: !gpu }, async () => withDevice(async device => {
  const m = CLOTH.nodesAcross.value, h = 1 / (m - 1), n = Math.floor(1.7 * (m - 1) + 1e-9) + 1, top = 2;
  const flat = herringBoneSheet({ columns: m, rows: n, width_m: 1, depth_m: (n - 1) * h, origin_m: [0, 0, 0], group: (_i, j) => j === 0 ? 1 : 0 });
  const rest: number[] = [];
  for (let v = 0; v < m * n; v++) rest.push(flat.restPositions_m[3 * v]!, top - flat.restPositions_m[3 * v + 2]!, 0);
  const material: ClothMaterial = { id: 'bridson', arealDensity_kg_m2: CLOTH.arealDensity.value, stretchWarp_N_m: CLOTH.stretch.value,
    stretchWeft_N_m: CLOTH.stretch.value, shear_N_m: CLOTH.shear.value, poisson: CLOTH.poisson.value, compressionRatio: CLOTH.compressionRatio.value,
    bending_N_m: CLOTH.bending.value, damping_s: CLOTH.damping.value, thickness_m: CLOTH.thickness.value, strainLimit: CLOTH.strainLimit.value };
  const topology = prepareClothTopology({ ...flat, restPositions_m: rest }, [material]);
  const hanging = buildClothSolverModel(topology, [material], colourClothVertices(clothVertexNeighbours(topology)));
  const solver = await ClothGPUSolver.create(device, hanging, { origin_m: [0.5, 1.15, 0], dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2',
    // K = 8, the gate's own setting: the scene's K went to 128 (44f02436), where the step is near converged and the
    // weakly damped sheet (0.002 s) still swings at 10 s (KE 8.9e-4 J, bending or not); at K = 8 it rests (5e-9 J).
    iterations: 8, chebyshevRho: 0, initialGuess: 'previous', coarse: buildClothCoarseSpace(topology) });
  for (let s = 0; s < 10; s++) solver.step(60);
  const { positions_m: x, velocities_m_s: v } = await solver.readState();
  const k = material.stretchWeft_N_m, rho = material.arealDensity_kg_m2;
  const exact = (j: number) => {
    const load = rho * 9.81 * (n - 1.5 - j) * h;
    let l = 1;
    for (let it = 0; it < 50; it++) l -= (k * l * (l * l - 1) / 2 - load) / (k * (3 * l * l - 1) / 2);
    return l - 1;
  };
  let worst = 0, drop = 0, maxEdge = 0;
  for (let j = 0; j + 1 < n; j++) {
    drop += exact(j) * h;
    let sum = 0;
    for (let i = 1; i < m - 1; i++) {
      const a = i + m * j, b = a + m;
      const e = Math.hypot(x[3 * a]! - x[3 * b]!, x[3 * a + 1]! - x[3 * b + 1]!, x[3 * a + 2]! - x[3 * b + 2]!) / h - 1;
      sum += e; maxEdge = Math.max(maxEdge, e);
    }
    if (j >= 2 && j < n - 4) worst = Math.max(worst, Math.abs(sum / (m - 2) - exact(j)) / exact(j));
  }
  let ke = 0;
  for (let i = 0; i < v.length; i++) ke += 0.5 * rho * h * h * v[i]! ** 2;
  const bottom = Math.floor(m / 2) + m * (n - 1), stretch = top - x[3 * bottom + 1]! - (n - 1) * h;
  console.log(`hanging sheet, 600 steps: worst band strain error ${(100 * worst).toFixed(2)}%, bottom stretch ${stretch.toFixed(4)} m (exact ${drop.toFixed(4)}), max edge ${(100 * maxEdge).toFixed(2)}%, KE ${ke.toExponential(2)} J`);
  assert.ok(worst < 0.01, `band strain off its equilibrium by ${worst}`);
  assert.ok(Math.abs(stretch - drop) < 0.01 * drop, `stretch ${stretch} m, exact ${drop} m`);
  assert.ok(maxEdge < CLOTH.strainLimit.value, `max edge strain ${maxEdge}: the limit should be inactive at rest`);
  assert.ok(ke < 1e-6, `kinetic energy ${ke} J`);
  assert.equal((await solver.readStatus()).status, 0);
  solver.destroy();
}));

/**
 * Tier A GPU cloth solver against the Float64 oracle: one step, a scripted
 * multi-step run with BDF2, strain limits and damping (Chebyshev refused), the
 * diagnostics ring, save/load of the solver state, and C2's contact block
 * plugged into the vertex kernel. Every GPU step takes the two-level
 * correction (there is no one-level path), so the oracle runs it too, from
 * the same coarse space. Skips without WEBGPU_NODE_MODULE.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { clothGridData } from '../lib/cloth/grid';
import { clothVertexNeighbours, colourClothVertices, prepareClothTopology } from '../lib/cloth/topology';
import type { ClothAttachmentScript, ClothMaterial } from '../lib/cloth/types';
import { buildClothSolverModel, clothHingeVector, clothOracleStep, createClothOracleState, type ClothStepOptions } from '../lib/cloth/oracle';
import { ClothGPUSolver, clothBendingStencil } from '../lib/cloth/gpu-solver';
import { buildClothCoarseSpace, ClothCoarseCorrection } from '../lib/cloth/coarse';
import { CLOTH_BINDING } from '../lib/cloth/gpu-solver-wgsl';
import { ogcVertexBlockWGSL } from '../lib/cloth/contact/ogc-wgsl';

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

function setup(columns: number, scripts: ClothAttachmentScript[] = []) {
  const data = clothGridData({ columns, rows: columns, width_m: 0.5, height_m: 0.5, origin_m: [0, 1, 0],
    group: (i, j) => j === 0 && (i === 0 || i === columns - 1) ? 1 : 0 });
  const topology = prepareClothTopology(data, [cotton]);
  return buildClothSolverModel(topology, [cotton], colourClothVertices(clothVertexNeighbours(topology)), scripts);
}

/** A pose with definite strains (the rest pose sits on the compression switch, where f32 and f64 pick different branches). */
function perturbed(model: ReturnType<typeof setup>, amplitude: number): Float64Array {
  let seed = 12345;
  const r = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32 - 0.5);
  const x = Float64Array.from(model.topology.data.restPositions_m, (p, i) => p + amplitude * r());
  // Start the oracle from the f32 pose the GPU holds.
  for (let i = 0; i < x.length; i++) x[i] = Math.fround(x[i]! - origin[i % 3]!) + origin[i % 3]!;
  return x;
}

/** The oracle's two-level correction on the coarse space the GPU solver builds (f32 line-search slack). */
const coarseOf = (model: ReturnType<typeof setup>) => new ClothCoarseCorrection(model, buildClothCoarseSpace(model.topology), 2 ** -23);

function maxDifference(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
  return worst;
}

test('one GPU step matches the oracle, for every initial guess', { skip: !gpu }, async () => withDevice(async device => {
  const model = setup(6), start = perturbed(model, 4e-3);
  for (const [iterations, initialGuess] of [[1, 'adaptive'], [4, 'adaptive'], [4, 'previous'], [4, 'inertial']] as const) {
    const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations, chebyshevRho: 0, initialGuess };
    const solver = await ClothGPUSolver.create(device, model, { ...options, origin_m: origin, positions_m: start });
    const oracle = createClothOracleState(model, start);
    clothOracleStep(model, oracle, options, false, undefined, coarseOf(model));
    solver.step();
    const d = maxDifference(await solver.readPositions(), oracle.x);
    assert.ok(d < 1e-6, `K=${iterations} ${initialGuess}: GPU vs oracle ${d} m`);
    solver.destroy();
  }
}));

test('every lanes-per-vertex layout of the colour solve matches the oracle, with subgroup or workgroup sums; invalid layouts fail', { skip: !gpu }, async () => withDevice(async device => {
  const model = setup(7), start = perturbed(model, 4e-3);
  const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 4, chebyshevRho: 0, initialGuess: 'adaptive' };
  const oracle = createClothOracleState(model, start), coarse = coarseOf(model);
  for (let s = 0; s < 3; s++) clothOracleStep(model, oracle, options, false, undefined, coarse);
  // Subgroup shuffles where the device has them, and the workgroup-memory sums always.
  for (const subgroups of device.features.has('subgroups') ? [undefined, false] : [false]) for (const vertexLanes of [1, 2, 4, 8, 32]) {
    const solver = await ClothGPUSolver.create(device, model, { ...options, origin_m: origin, positions_m: start, vertexLanes, subgroups });
    assert.equal(solver.subgroups, subgroups !== false && vertexLanes > 1);
    solver.step(3);
    const d = maxDifference(await solver.readPositions(), oracle.x);
    assert.ok(d < 2e-6, `${vertexLanes} lanes per vertex (${solver.subgroups ? 'subgroup' : 'workgroup'} sums): GPU vs oracle ${d} m`);
    solver.destroy();
  }
  await assert.rejects(ClothGPUSolver.create(device, model, { ...options, origin_m: origin, vertexLanes: 3 }), /power of two/);
}));

test('the bending stencil is the per-hinge quadratic bending: gradient, zero row sums, and a stiff fold on both colour kernels', async t => {
  const stiff: ClothMaterial = { ...cotton, id: 'stiff', bending_N_m: 2e-3, damping_s: 0.01 };
  const data = clothGridData({ columns: 7, rows: 7, width_m: 0.5, height_m: 0.5, origin_m: [0, 1, 0], group: (i, j) => j === 0 && (i === 0 || i === 6) ? 1 : 0 });
  const topology = prepareClothTopology(data, [stiff]);
  const model = buildClothSolverModel(topology, [stiff], colourClothVertices(clothVertexNeighbours(topology)));
  const x = perturbed(model, 2e-2), stencil = clothBendingStencil(model), N = topology.vertexCount;
  // Float64: the stencil's sum_j Q_vj (x_j - x_v) is sum_h k w_v s_h over v's hinges, and each row of Q sums to zero.
  const perHinge = new Float64Array(3 * N);
  for (let h = 0; h < topology.hingeCount; h++) {
    const s = clothHingeVector(topology, h, i => [x[3 * i]!, x[3 * i + 1]!, x[3 * i + 2]!]), k = topology.hingeStiffness_N_m[h]!;
    for (let a = 0; a < 4; a++) for (let c = 0; c < 3; c++) perHinge[3 * topology.hinges[4 * h + a]! + c] += k * topology.hingeWeights[4 * h + a]! * s[c]!;
  }
  let worst = 0, scale = 0, rowSum = 0;
  for (let v = 0; v < N; v++) {
    let sum = stencil.diagonal[2 * v]!;
    for (let c = 0; c < 3; c++) {
      let gs = 0;
      for (let o = stencil.offsets[v]!; o < stencil.offsets[v + 1]!; o++) gs += stencil.q[o]! * (x[3 * stencil.columns[o]! + c]! - x[3 * v + c]!);
      worst = Math.max(worst, Math.abs(gs - perHinge[3 * v + c]!)); scale = Math.max(scale, Math.abs(perHinge[3 * v + c]!));
    }
    for (let o = stencil.offsets[v]!; o < stencil.offsets[v + 1]!; o++) sum += stencil.q[o]!;
    rowSum = Math.max(rowSum, Math.abs(sum) / stencil.diagonal[2 * v]!);
  }
  assert.ok(scale > 0 && worst < 1e-12 * scale, `stencil vs per-hinge bending gradient: ${worst} of ${scale}`);
  assert.ok(rowSum < 1e-12, `stencil row sums: ${rowSum}`);
  if (!gpu) { t.skip('no GPU'); return; }
  await withDevice(async device => {
    const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 4, chebyshevRho: 0, initialGuess: 'adaptive' };
    const oracle = createClothOracleState(model, x), coarse = coarseOf(model);
    for (let s = 0; s < 3; s++) clothOracleStep(model, oracle, options, false, undefined, coarse);
    for (const vertexLanes of [1, 4]) {
      const solver = await ClothGPUSolver.create(device, model, { ...options, origin_m: origin, positions_m: x, vertexLanes });
      solver.step(3);
      const d = maxDifference(await solver.readPositions(), oracle.x);
      t.diagnostic(`stiff fold, ${vertexLanes} lanes per vertex: GPU vs oracle ${d.toExponential(2)} m`);
      assert.ok(d < 1e-5, `stiff fold, ${vertexLanes} lanes per vertex: GPU vs oracle ${d} m`);
      solver.destroy();
    }
  });
});

test('a scripted BDF2 run with strain limits tracks the oracle; the ring reports its residual; Chebyshev acceleration fails explicitly', { skip: !gpu }, async t => withDevice(async device => {
  const scripts: ClothAttachmentScript[] = [{ kind: 'translate', group: 1, start_s: 0, end_s: 0.2, vector: [0.05, 0.02, 0], easing: 'cosine' }];
  const model = setup(8, scripts), start = perturbed(model, 4e-3);
  const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 48, chebyshevRho: 0 };
  // Chebyshev on top of the two-level correction explodes this run (the oracle: residual 3e10 at rho 0.9), so the solver refuses it.
  await assert.rejects(ClothGPUSolver.create(device, model, { ...options, chebyshevRho: 0.9, origin_m: origin }), /Chebyshev/);
  const solver = await ClothGPUSolver.create(device, model, { ...options, origin_m: origin, diagnostics: true, positions_m: start });
  const oracle = createClothOracleState(model, start);
  const steps = 20, residuals: number[] = [];
  const coarse = coarseOf(model);
  for (let s = 0; s < steps; s++) residuals.push(clothOracleStep(model, oracle, options, true, undefined, coarse).residuals.at(-1)!);
  solver.step(steps);
  const x = await solver.readPositions(), d = maxDifference(x, oracle.x);
  const report = await solver.readStatus();
  t.diagnostic(`max |x_gpu - x_oracle| after ${steps} steps: ${d.toExponential(2)} m; residual GPU ${report.slots.at(-1)!.residual.toExponential(2)}, oracle ${residuals.at(-1)!.toExponential(2)}`);
  assert.equal(report.status, 0);
  assert.equal(report.steps, steps);
  assert.equal(report.slots.length, steps);
  assert.ok(report.slots.every((s, i) => s.step === i && s.bdf2 === (i > 0)));
  assert.ok(d < 5e-4, `trajectory ${d} m`);
  const gpuResidual = report.slots.at(-1)!.residual, cpuResidual = residuals.at(-1)!;
  assert.ok(gpuResidual < 3 * cpuResidual + 1e-3 && cpuResidual < 3 * gpuResidual + 1e-3, `residual ${gpuResidual} vs ${cpuResidual}`);
  solver.destroy();
}));

test('saved solver state reloads to the same trajectory', { skip: !gpu }, async () => withDevice(async device => {
  const scripts: ClothAttachmentScript[] = [{ kind: 'oscillate', group: 1, start_s: 0, end_s: 10, vector: [0.03, 0, 0], period_s: 0.5 }];
  const model = setup(7, scripts);
  const options = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2' as const, iterations: 8, chebyshevRho: 0, origin_m: origin };
  const solver = await ClothGPUSolver.create(device, model, options);
  solver.step(6);
  const saved = await solver.readState();
  assert.ok(Math.abs(saved.solverState.time_s - 0.1) < 1e-6);
  solver.step(6);
  const reference = await solver.readPositions();
  solver.uploadState(saved.positions_m, saved.velocities_m_s, saved.solverState);
  solver.step(6);
  const replay = await solver.readPositions();
  assert.ok(maxDifference(replay, reference) < 1e-6, `replay ${maxDifference(replay, reference)} m`);
  const fresh = await ClothGPUSolver.create(device, model, { ...options, positions_m: saved.positions_m, velocities_m_s: saved.velocities_m_s, solverState: saved.solverState });
  fresh.step(6);
  assert.ok(maxDifference(await fresh.readPositions(), reference) < 1e-6);
  solver.destroy(); fresh.destroy();
}));

test("C2's OGC vertex block compiles into the vertex kernel; an empty contact list changes nothing beyond f32 rounding; contact needs the coarse move's truncation", { skip: !gpu }, async () => withDevice(async device => {
  const model = setup(6), start = perturbed(model, 4e-3);
  const options = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2' as const, iterations: 6, chebyshevRho: 0, origin_m: origin, positions_m: start };
  const plain = await ClothGPUSolver.create(device, model, options);
  // Header (0, capacity 0, counts at word-vec 2, entries at 2): every vertex has no contacts.
  const contacts = device.createBuffer({ size: 16 * (4 + Math.ceil(model.topology.vertexCount / 4)), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(contacts, 0, new Uint32Array([0, 0, 2, 2]));
  const contact = { wgsl: ogcVertexBlockWGSL({ group: 0, binding: CLOTH_BINDING.contact }), buffer: contacts };
  // The two-level correction's move is a move like a sweep's: a solver with contact refuses to start without its truncation.
  await assert.rejects(ClothGPUSolver.create(device, model, { ...options, contact }), /afterCoarse/);
  const withContact = await ClothGPUSolver.create(device, model, { ...options, contact, hooks: { afterCoarse: () => {} } });
  let rounds = 0;
  withContact.hooks = { afterIteration: () => { rounds++; } };
  plain.step(3); withContact.step(3);
  assert.equal(rounds, 18);
  const d = maxDifference(await plain.readPositions(), await withContact.readPositions());
  // The two-level kernels compile differently with the contact block (its zero terms change the shader's FMA
  // contraction), so the runs agree to f32 rounding, not bitwise: measured 9.3e-8 m, one ulp at 1 m.
  assert.ok(d < 5e-7, `empty contact list moved the cloth by ${d} m`);
  plain.destroy(); withContact.destroy(); contacts.destroy();
}));

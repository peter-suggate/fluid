/**
 * The GPU cloth solver (Newton-PCG) against the Float64 oracle run to convergence: one step from every initial guess,
 * the bending stencil on a stiff fold, a scripted multi-step run with BDF2, strain limits and damping, the diagnostics
 * ring, save/load of the solver state, and C2's contact block compiled into the Newton kernels. The oracle is
 * Gauss-Seidel with its two-level correction (oracle-coarse.ts) at 250 sweeps a step, so both land on the step's
 * minimizer. Skips without WEBGPU_NODE_MODULE.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { clothGridData } from '../lib/cloth/grid';
import { prepareClothTopology } from '../lib/cloth/topology';
import type { ClothAttachmentScript, ClothMaterial } from '../lib/cloth/types';
import { buildClothSolverModel, clothBendVector, clothOracleStep, createClothOracleState, type ClothStepOptions } from '../lib/cloth/oracle';
import { ClothGPUSolver, clothBendingStencil, type ClothNewtonOptions } from '../lib/cloth/gpu-solver';
import { buildClothCoarseSpace, ClothCoarseCorrection } from '../lib/cloth/oracle-coarse';
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
  return buildClothSolverModel(topology, [cotton], scripts);
}

/** A noisy pose with definite strains for the CPU stencil check (pins at their targets). */
function perturbed(model: ReturnType<typeof setup>, amplitude: number): Float64Array {
  let seed = 12345;
  const r = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32 - 0.5), groups = model.topology.groups;
  const x = Float64Array.from(model.topology.data.restPositions_m, (p, i) => groups[Math.floor(i / 3)]! > 0 ? p : p + amplitude * r());
  // Start the oracle from the f32 pose the GPU holds.
  for (let i = 0; i < x.length; i++) x[i] = Math.fround(x[i]! - origin[i % 3]!) + origin[i % 3]!;
  return x;
}

/**
 * A smooth bump of height A over the sheet (zero on its border, so the pins stay at their targets): a strained pose
 * whose step has one minimizer. Per-vertex noise is not one: the compression-softened membrane (compressionRatio 0.05)
 * buckles either way, and Newton and the oracle, each converged to the f32 floor, land on stationary points mm to cm
 * apart. From the bump they agree to ~1e-8 m (one step, the fold and the scripted run alike) at 12 x 200.
 */
function bumped(model: ReturnType<typeof setup>, A: number): Float64Array {
  const x = Float64Array.from(model.topology.data.restPositions_m), n = Math.round(Math.sqrt(x.length / 3));
  for (let v = 0; v < x.length / 3; v++) x[3 * v + 1] += A * Math.sin(Math.PI * (v % n) / (n - 1)) * Math.sin(Math.PI * Math.floor(v / n) / (n - 1));
  for (let i = 0; i < x.length; i++) x[i] = Math.fround(x[i]! - origin[i % 3]!) + origin[i % 3]!;
  return x;
}

/** The oracle's two-level correction, which carries its Gauss-Seidel sweeps to convergence in ORACLE sweeps a step. */
const coarseOf = (model: ReturnType<typeof setup>) => new ClothCoarseCorrection(model, buildClothCoarseSpace(model.topology));
const ORACLE = 250;
/**
 * Converged Newton-PCG (cloth-newton.test.ts). The f32 residual floors near 3e-4 |m g| on these poses, so the
 * tolerance never stops a step and every step runs its caps (the stop criterion is part of the performance work).
 */
const converged: ClothNewtonOptions = { iterations: 12, pcgIterations: 200, pcgTolerance: 1e-4, residualTolerance: 1e-5 };

function maxDifference(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i]! - b[i]!));
  return worst;
}

test('one converged GPU step matches the converged oracle from every initial guess', { skip: !gpu }, async t => withDevice(async device => {
  const model = setup(6), start = bumped(model, 4e-3);
  const oracle = createClothOracleState(model, start);
  clothOracleStep(model, oracle, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: ORACLE }, false, coarseOf(model));
  for (const initialGuess of ['previous', 'adaptive', 'inertial'] as const) {
    const solver = await ClothGPUSolver.create(device, model, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', newton: converged, initialGuess, origin_m: origin, positions_m: start });
    solver.step();
    const d = maxDifference(await solver.readPositions(), oracle.x);
    t.diagnostic(`${initialGuess}: GPU vs oracle ${d.toExponential(2)} m`);
    assert.ok(d < 2e-6, `${initialGuess}: GPU vs oracle ${d} m`);
    solver.destroy();
  }
}));

test('the bending stencil is the per-row bending: gradient, zero row sums, and a stiff fold against the oracle', async t => {
  const stiff: ClothMaterial = { ...cotton, id: 'stiff', bending_N_m: 2e-3, damping_s: 0.01 };
  const data = clothGridData({ columns: 7, rows: 7, width_m: 0.5, height_m: 0.5, origin_m: [0, 1, 0], group: (i, j) => j === 0 && (i === 0 || i === 6) ? 1 : 0 });
  const topology = prepareClothTopology(data, [stiff]);
  const model = buildClothSolverModel(topology, [stiff]);
  const x = perturbed(model, 2e-2), stencil = clothBendingStencil(model), N = topology.vertexCount;
  // Float64: the stencil's sum_j Q_vj (x_j - x_v) is sum_r k_r w_rv L_r x over the rows holding v, and each row of Q sums to zero.
  const perRow = new Float64Array(3 * N);
  for (let r = 0; r < topology.bendRowCount; r++) {
    const s = clothBendVector(topology, r, i => [x[3 * i]!, x[3 * i + 1]!, x[3 * i + 2]!]), k = topology.bendStiffness_N_m3[r]!;
    for (let o = topology.bendOffsets[r]!; o < topology.bendOffsets[r + 1]!; o++) for (let c = 0; c < 3; c++) perRow[3 * topology.bendVertices[o]! + c] += k * topology.bendWeights[o]! * s[c]!;
  }
  let worst = 0, scale = 0, rowSum = 0;
  for (let v = 0; v < N; v++) {
    let sum = stencil.diagonal[2 * v]!;
    for (let c = 0; c < 3; c++) {
      let gs = 0;
      for (let o = stencil.offsets[v]!; o < stencil.offsets[v + 1]!; o++) gs += stencil.q[o]! * (x[3 * stencil.columns[o]! + c]! - x[3 * v + c]!);
      worst = Math.max(worst, Math.abs(gs - perRow[3 * v + c]!)); scale = Math.max(scale, Math.abs(perRow[3 * v + c]!));
    }
    for (let o = stencil.offsets[v]!; o < stencil.offsets[v + 1]!; o++) sum += stencil.q[o]!;
    rowSum = Math.max(rowSum, Math.abs(sum) / stencil.diagonal[2 * v]!);
  }
  assert.ok(scale > 0 && worst < 1e-12 * scale, `stencil vs per-row bending gradient: ${worst} of ${scale}`);
  assert.ok(rowSum < 1e-12, `stencil row sums: ${rowSum}`);
  if (!gpu) { t.skip('no GPU'); return; }
  await withDevice(async device => {
    const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: ORACLE };
    const fold = bumped(model, 2e-2);
    const oracle = createClothOracleState(model, fold), coarse = new ClothCoarseCorrection(model, buildClothCoarseSpace(model.topology));
    for (let s = 0; s < 3; s++) clothOracleStep(model, oracle, options, false, coarse);
    const solver = await ClothGPUSolver.create(device, model, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', newton: converged, origin_m: origin, positions_m: fold });
    solver.step(3);
    const d = maxDifference(await solver.readPositions(), oracle.x);
    t.diagnostic(`stiff fold: GPU vs oracle ${d.toExponential(2)} m`);
    assert.ok(d < 5e-6, `stiff fold: GPU vs oracle ${d} m`);
    solver.destroy();
  });
});

test('a scripted BDF2 run with strain limits tracks the oracle; the ring reports its residual', { skip: !gpu }, async t => withDevice(async device => {
  const scripts: ClothAttachmentScript[] = [{ kind: 'translate', group: 1, start_s: 0, end_s: 0.2, vector: [0.05, 0.02, 0], easing: 'cosine' }];
  const model = setup(8, scripts), start = bumped(model, 4e-3);
  const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: ORACLE };
  // The strain limit's multipliers converge over Newton iterations (one dual update after each; cloth-newton.test.ts).
  const solver = await ClothGPUSolver.create(device, model, { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', newton: { ...converged, iterations: 32 },
    origin_m: origin, diagnostics: true, positions_m: start });
  const oracle = createClothOracleState(model, start);
  const steps = 20, residuals: number[] = [];
  const coarse = coarseOf(model);
  for (let s = 0; s < steps; s++) residuals.push(clothOracleStep(model, oracle, options, true, coarse).residuals.at(-1)!);
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
  const options = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2' as const, newton: { iterations: 4, pcgIterations: 50 }, origin_m: origin };
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

test("C2's OGC vertex block compiles into the Newton kernels; an empty contact list changes nothing beyond f32 rounding; contact needs the filter and its state", { skip: !gpu }, async () => withDevice(async device => {
  const model = setup(6), start = bumped(model, 4e-3), N = model.topology.vertexCount;
  const newton = { iterations: 6, pcgIterations: 50 };
  const options = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2' as const, newton, origin_m: origin, positions_m: start };
  const plain = await ClothGPUSolver.create(device, model, options);
  // Header (0, capacity 0, counts at word-vec 2, entries at 2): every vertex has no contacts.
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const contacts = device.createBuffer({ size: 16 * (4 + Math.ceil(N / 4)), usage });
  device.queue.writeBuffer(contacts, 0, new Uint32Array([0, 0, 2, 2]));
  // The filter allows the full step (alpha_v = alpha_min = 1); no vertex is in contact.
  const alpha = device.createBuffer({ size: 4 * (N + 1), usage }), state = device.createBuffer({ size: 4 * Math.max(N, 4), usage });
  device.queue.writeBuffer(alpha, 0, new Float32Array(N + 1).fill(1));
  const contact = { wgsl: ogcVertexBlockWGSL({ group: 0, binding: CLOTH_BINDING.contact, newton: true }), buffer: contacts, alpha, state };
  // Contact enters Newton through the filter and re-detection: a solver with contact refuses to start without those hooks.
  await assert.rejects(ClothGPUSolver.create(device, model, { ...options, contact }), /hooks\.filter/);
  let rounds = 0;
  const withContact = await ClothGPUSolver.create(device, model, { ...options, contact, hooks: { filter: () => {}, afterNewtonStep: () => { rounds++; }, stepEnd: () => {} } });
  plain.step(3); withContact.step(3);
  assert.equal(rounds, 3 * newton.iterations);
  assert.equal((await withContact.readStatus()).restarts, 0);
  const d = maxDifference(await plain.readPositions(), await withContact.readPositions());
  // The contact block's zero terms change the shader's FMA contraction, so the runs agree to f32 rounding, not bitwise.
  assert.ok(d < 5e-7, `empty contact list moved the cloth by ${d} m`);
  plain.destroy(); withContact.destroy(); contacts.destroy(); alpha.destroy(); state.destroy();
}));

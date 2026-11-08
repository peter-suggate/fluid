/**
 * Tier A GPU cloth solver against the Float64 oracle: one step, a scripted
 * multi-step run with BDF2, strain limits, damping and Chebyshev, the
 * diagnostics ring, save/load of the solver state, and C2's contact block
 * plugged into the vertex kernel. Skips without
 * WEBGPU_NODE_MODULE.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { clothGridData } from '../lib/cloth/grid';
import { clothVertexNeighbours, colourClothVertices, prepareClothTopology } from '../lib/cloth/topology';
import type { ClothAttachmentScript, ClothMaterial } from '../lib/cloth/types';
import { buildClothSolverModel, clothOracleStep, createClothOracleState, type ClothStepOptions } from '../lib/cloth/oracle';
import { ClothGPUSolver } from '../lib/cloth/gpu-solver';
import { CLOTH_BINDING } from '../lib/cloth/gpu-solver-wgsl';
import { ogcVertexBlockWGSL } from '../lib/cloth/contact/ogc-wgsl';

const gpu = !!process.env.WEBGPU_NODE_MODULE;

async function withDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
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
    clothOracleStep(model, oracle, options);
    solver.step();
    const d = maxDifference(await solver.readPositions(), oracle.x);
    assert.ok(d < 1e-6, `K=${iterations} ${initialGuess}: GPU vs oracle ${d} m`);
    solver.destroy();
  }
}));

test('every lanes-per-vertex layout of the colour solve matches the oracle; invalid layouts fail', { skip: !gpu }, async () => withDevice(async device => {
  const model = setup(7), start = perturbed(model, 4e-3);
  const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 4, chebyshevRho: 0, initialGuess: 'adaptive' };
  const oracle = createClothOracleState(model, start);
  for (let s = 0; s < 3; s++) clothOracleStep(model, oracle, options);
  for (const vertexLanes of [1, 2, 4, 8, 32]) {
    const solver = await ClothGPUSolver.create(device, model, { ...options, origin_m: origin, positions_m: start, vertexLanes });
    solver.step(3);
    const d = maxDifference(await solver.readPositions(), oracle.x);
    assert.ok(d < 2e-6, `${vertexLanes} lanes per vertex: GPU vs oracle ${d} m`);
    solver.destroy();
  }
  await assert.rejects(ClothGPUSolver.create(device, model, { ...options, origin_m: origin, vertexLanes: 3 }), /power of two/);
}));

test('a scripted BDF2 run with strain limits and Chebyshev tracks the oracle; the ring reports its residual', { skip: !gpu }, async t => withDevice(async device => {
  const scripts: ClothAttachmentScript[] = [{ kind: 'translate', group: 1, start_s: 0, end_s: 0.2, vector: [0.05, 0.02, 0], easing: 'cosine' }];
  const model = setup(8, scripts), start = perturbed(model, 4e-3);
  const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 48, chebyshevRho: 0.9 };
  const solver = await ClothGPUSolver.create(device, model, { ...options, origin_m: origin, diagnostics: true, positions_m: start });
  const oracle = createClothOracleState(model, start);
  const steps = 20, residuals: number[] = [];
  for (let s = 0; s < steps; s++) residuals.push(clothOracleStep(model, oracle, options, true).residuals.at(-1)!);
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
  const options = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2' as const, iterations: 8, chebyshevRho: 0.8, origin_m: origin };
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

test("C2's OGC vertex block compiles into the vertex kernel; an empty contact list changes nothing", { skip: !gpu }, async () => withDevice(async device => {
  const model = setup(6), start = perturbed(model, 4e-3);
  const options = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2' as const, iterations: 6, chebyshevRho: 0.8, origin_m: origin, positions_m: start };
  const plain = await ClothGPUSolver.create(device, model, options);
  // Header (0, capacity 0, counts at word-vec 2, entries at 2): every vertex has no contacts.
  const contacts = device.createBuffer({ size: 16 * (4 + Math.ceil(model.topology.vertexCount / 4)), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(contacts, 0, new Uint32Array([0, 0, 2, 2]));
  const withContact = await ClothGPUSolver.create(device, model, { ...options, contact: { wgsl: ogcVertexBlockWGSL({ group: 0, binding: CLOTH_BINDING.contact }), buffer: contacts } });
  let rounds = 0;
  withContact.hooks = { afterIteration: () => { rounds++; } };
  plain.step(3); withContact.step(3);
  assert.equal(rounds, 18);
  assert.ok(maxDifference(await plain.readPositions(), await withContact.readPositions()) === 0);
  plain.destroy(); withContact.destroy(); contacts.destroy();
}));

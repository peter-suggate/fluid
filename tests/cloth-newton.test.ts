/**
 * The Newton-PCG GPU step (method 'newton') against the Float64 oracle run to convergence: the same objective
 * (BDF2 inertia, membrane, bending, damping), so a converged Newton step and a converged oracle step land on the same
 * minimizer (the oracle runs Gauss-Seidel sweeps with its two-level correction, oracle-coarse.ts). Pins are Dirichlet DOFs in Newton and AVBD terms in the oracle, which agree at convergence; strain limits are
 * augmented-Lagrangian terms in both, whose duals reach the same KKT point. Also the refusals: unknown methods and caps. Skips without WEBGPU_NODE_MODULE.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { clothGridData } from '../lib/cloth/grid';
import { prepareClothTopology } from '../lib/cloth/topology';
import type { ClothMaterial } from '../lib/cloth/types';
import { buildClothSolverModel, clothOracleStep, createClothOracleState, type ClothStepOptions } from '../lib/cloth/oracle';
import { ClothGPUSolver } from '../lib/cloth/gpu-solver';
import { buildClothCoarseSpace, ClothCoarseCorrection } from '../lib/cloth/oracle-coarse';

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
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0.002, thickness_m: 5e-4 };
const g: [number, number, number] = [0, -9.81, 0];
const origin: [number, number, number] = [0.25, 1, 0.25];

function setup(columns: number, material = cotton) {
  const data = clothGridData({ columns, rows: columns, width_m: 0.5, height_m: 0.5, origin_m: [0, 1, 0],
    group: (i, j) => j === 0 && (i === 0 || i === columns - 1) ? 1 : 0 });
  const topology = prepareClothTopology(data, [material]);
  return buildClothSolverModel(topology, [material]);
}

const base = { origin_m: origin, dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2' as const, initialGuess: 'previous' as const };

test('Newton-PCG steps match the converged Float64 oracle over a BDF2 run of a corner-held sheet', { skip: !gpu }, async () => withDevice(async device => {
  const model = setup(10), steps = 15;
  const solver = await ClothGPUSolver.create(device, model, { ...base, method: 'newton', newton: { iterations: 12, pcgIterations: 200, pcgTolerance: 1e-4, residualTolerance: 1e-5 } });
  const st = createClothOracleState(model, model.topology.data.restPositions_m);
  const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 250, initialGuess: 'previous' };
  const coarse = new ClothCoarseCorrection(model, buildClothCoarseSpace(model.topology));
  for (let s = 0; s < steps; s++) clothOracleStep(model, st, options, false, coarse);
  solver.step(steps);
  const x = await solver.readPositions(), status = await solver.readStatus(), ctl = await solver.readNewtonControl();
  assert.equal(status.status, 0); assert.equal(status.steps, steps);
  let worst = 0, fall = 0;
  for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(x[i]! - st.x[i]!));
  for (let v = 0; v < model.topology.vertexCount; v++) fall = Math.max(fall, model.topology.data.restPositions_m[3 * v + 1]! - x[3 * v + 1]!);
  assert.ok(fall > 0.05, `the sheet swings down (fall ${fall})`);
  assert.ok(worst < 2e-5, `Newton GPU vs converged oracle: ${worst} m`);
  // The last step's free-DOF residual |g| / |m g| reached the tolerance or the f32 floor.
  assert.ok(Math.sqrt(ctl[5]! / ctl[6]!) < 1e-2, `residual ${Math.sqrt(ctl[5]! / ctl[6]!)}`);
  solver.destroy();
}));

test('Newton-PCG with augmented-Lagrangian strain limits matches the converged Float64 oracle with the limit active', { skip: !gpu }, async () => withDevice(async device => {
  const limit = 0.01, model = setup(10, { ...cotton, strainLimit: limit }), steps = 15;
  // The multipliers converge over Newton iterations (one AVBD dual update after each): 16 leave 3.4e-5 m, 32 reach 3.1e-6.
  const solver = await ClothGPUSolver.create(device, model, { ...base, method: 'newton', newton: { iterations: 32, pcgIterations: 200, pcgTolerance: 1e-4, residualTolerance: 1e-5 } });
  const st = createClothOracleState(model, model.topology.data.restPositions_m);
  const options: ClothStepOptions = { dt_s: 1 / 60, gravity_m_s2: g, integrator: 'bdf2', iterations: 250, initialGuess: 'previous' };
  const coarse = new ClothCoarseCorrection(model, buildClothCoarseSpace(model.topology));
  for (let s = 0; s < steps; s++) clothOracleStep(model, st, options, false, coarse);
  solver.step(steps);
  const x = await solver.readPositions(), status = await solver.readStatus(), ctl = await solver.readNewtonControl();
  assert.equal(status.status, 0); assert.equal(status.steps, steps);
  const t = model.topology, strain = (p: ArrayLike<number>) => {
    let worst = -Infinity;
    for (let e = 0; e < t.edgeCount; e++) {
      const a = t.edges[4 * e]!, b = t.edges[4 * e + 1]!;
      worst = Math.max(worst, Math.hypot(p[3 * a]! - p[3 * b]!, p[3 * a + 1]! - p[3 * b + 1]!, p[3 * a + 2]! - p[3 * b + 2]!) / t.edgeRestLength_m[e]! - 1);
    }
    return worst;
  };
  let active = 0, worst = 0;
  for (let e = 0; e < t.edgeCount; e++) if (st.edgeDuals[2 * e]! > 0) active++;
  for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(x[i]! - st.x[i]!));
  assert.ok(active > 0, 'the limit is active in the oracle');
  assert.ok(strain(x) < limit * 1.05, `GPU strain ${strain(x)} within the limit ${limit}`);
  assert.ok(worst < 1e-5, `Newton GPU vs converged oracle: ${worst} m`);
  assert.ok(Math.sqrt(ctl[5]! / ctl[6]!) < 1e-2, `residual ${Math.sqrt(ctl[5]! / ctl[6]!)}`);
  solver.destroy();
}));

test('the Newton method refuses unknown methods and bad caps', { skip: !gpu }, async () => withDevice(async device => {
  const model = setup(6);
  const newton = { iterations: 2, pcgIterations: 4 };
  // VBD was removed (M4): its method name is an unknown method like any other.
  for (const method of ['jacobi', 'vbd'])
    await assert.rejects(ClothGPUSolver.create(device, model, { ...base, newton, method: method as never }), /Unknown cloth solver method/);
  await assert.rejects(ClothGPUSolver.create(device, model, { ...base } as never), /needs integer caps/);
  await assert.rejects(ClothGPUSolver.create(device, model, { ...base, newton: { iterations: 2, pcgIterations: 0 } }), /needs integer caps/);
  await assert.rejects(ClothGPUSolver.create(device, model, { ...base, newton: { iterations: 2.5, pcgIterations: 4 } }), /needs integer caps/);
}));

import test from 'node:test';
import assert from 'node:assert/strict';
import {GPUClothNewton, gpuClothNewtonSupported} from '../lib/cloth/gpu-newton';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {createClothState, prepareClothTopology} from '../lib/cloth/topology';
import {solveClothIncrement} from '../lib/cloth/nonlinear-solve';
import type {ClothMaterial, ClothState, ClothTopologyData} from '../lib/cloth/types';
import type {PreparedClothTopology} from '../lib/cloth/topology';
import type {ClothStepOptions} from '../lib/cloth/dynamics';
import {withUniformDevice} from './helpers/uniform-geometric';

/** An n x n sheet of spacing h in the y = 1 plane, two corners pinned. */
function sheet(n: number, h: number, y = 1): ClothTopologyData {
  const restPositions_m: number[] = [], triangles: number[] = [], fixed: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {restPositions_m.push(i*h, y, j*h); fixed.push(j === 0 && (i === 0 || i === n - 1) ? 1 : 0);}
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {const a = j*n + i, b = a + 1, c = a + n, d = c + 1; triangles.push(a, c, b, b, c, d);}
  return {restPositions_m, triangles, rodOffsets: [0], rodIndices: [], fixed};
}
const maximumDifference = (a: ArrayLike<number>, b: ArrayLike<number>) => {let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k]! - b[k]!)); return m;};
const gpu = {skip: !process.env.WEBGPU_NODE_MODULE};
/** The Float64 solve's own acceptance at the GPU result: evaluated from the
 * GPU iterate, it must stop before taking a Newton iteration. */
function acceptedInFloat64(topology: PreparedClothTopology, before: ClothState, after: ClothState, dt: number, options: ClothStepOptions): number {
  const check = solveClothIncrement(topology, structuredClone(before), dt, options, {positions_m: after.positions_m, twist_rad: after.twist_rad});
  assert.ok(check.converged && check.newtonIterations === 0, `Float64 measure ${check.maximumUpdate_m} at the GPU result`);
  return check.maximumUpdate_m;
}

test('rod joints are explicitly outside the GPU Newton solve', () => {
  const data: ClothTopologyData = {restPositions_m: [0, 0, 0, .01, 0, 0, .02, 0, 0], triangles: [], rodOffsets: [0, 3], rodIndices: [0, 1, 2], fixed: [3, 0, 0]};
  const topology = prepareClothTopology(data, [DEFAULT_CLOTH_MATERIAL]);
  assert.equal(gpuClothNewtonSupported(topology), false);
  assert.throws(() => new GPUClothNewton(undefined as unknown as GPUDevice, topology), /rod joints/);
});

test('GPU Newton-PCG reaches the Float64 Newton minimizer on a falling sheet', gpu, async () => {
  await withUniformDevice('cloth GPU Newton sheet', async device => {
    const topology = prepareClothTopology(sheet(8, .01), [DEFAULT_CLOTH_MATERIAL]), a = createClothState(topology), b = createClothState(topology), solver = new GPUClothNewton(device, topology);
    try {
      await solver.initialize();
      for (let step = 0; step < 8; step++) {
        const options = {iterations: 64, tolerance_m: 1e-8}, before = structuredClone(b);
        const cpu = solveClothIncrement(topology, a, .001, options), gpuReport = await solver.step(b, .001, options);
        assert.ok(cpu.converged && gpuReport.converged, `step ${step}: ${gpuReport.maximumUpdate_m}`);
        assert.equal(gpuReport.refinementIterations, 0, `step ${step} needed the CPU refinement`);
        assert.ok(gpuReport.newtonIterations <= 8, `step ${step}: ${gpuReport.newtonIterations} Newton iterations`);
        acceptedInFloat64(topology, before, b, .001, options);
        assert.ok(maximumDifference(a.positions_m, b.positions_m) < 1e-6, `step ${step}: positions differ by ${maximumDifference(a.positions_m, b.positions_m)} m`);
      }
      assert.ok(b.positions_m[3*20 + 1]! < 1, 'the sheet falls');
    } finally {solver.destroy();}
  });
});

test('GPU Newton-PCG converges a stiff plastic sheet and a soft-fixed, cohesive, rod-stretch mix', gpu, async () => {
  await withUniformDevice('cloth GPU Newton stiff', async device => {
    const stiff: ClothMaterial = {...DEFAULT_CLOTH_MATERIAL, id: 'plastic', youngModulus_Pa: 3e9, thickness_m: .0005};
    const topology = prepareClothTopology(sheet(6, .01), [stiff]), a = createClothState(topology), b = createClothState(topology), solver = new GPUClothNewton(device, topology);
    try {
      await solver.initialize();
      for (let step = 0; step < 4; step++) {
        const options = {iterations: 64, tolerance_m: 1e-8}, before = structuredClone(b);
        const cpu = solveClothIncrement(topology, a, .001, options), g = await solver.step(b, .001, options);
        assert.ok(cpu.converged && g.converged && g.refinementIterations === 0, `step ${step}: ${g.maximumUpdate_m} m, refined ${g.refinementIterations}`);
        acceptedInFloat64(topology, before, b, .001, options);
        assert.ok(maximumDifference(a.positions_m, b.positions_m) < 1e-6, `step ${step}: ${maximumDifference(a.positions_m, b.positions_m)} m`);
      }
    } finally {solver.destroy();}
    // Two sheets joined by a cohesive pair, plus a two-vertex rod hanging off a corner.
    const base = sheet(4, .01), second = sheet(4, .01, 1.02), offset = 16;
    const data: ClothTopologyData = {
      restPositions_m: [...base.restPositions_m, ...second.restPositions_m, 0, .99, 0],
      triangles: [...base.triangles, ...second.triangles.map(v => v + offset)],
      rodOffsets: [0, 2], rodIndices: [0, 32], fixed: [...base.fixed!, ...second.fixed!.map(() => 0), 0],
    };
    const mixed = prepareClothTopology(data, [{...DEFAULT_CLOTH_MATERIAL, rodRadius_m: .0005}]), c = createClothState(mixed), d = createClothState(mixed), pairSolver = new GPUClothNewton(device, mixed);
    const pair = {vertices: [5, 6, 9, offset + 5, offset + 6, offset + 9], weights: [1/3, 1/3, 1/3, -1/3, -1/3, -1/3], force_N: 1e-4, startElement: 0, endElement: 1};
    try {
      await pairSolver.initialize();
      for (let step = 0; step < 4; step++) {
        const options = {iterations: 64, tolerance_m: 1e-8, cohesionPairs: [pair]}, before = structuredClone(d);
        const cpu = solveClothIncrement(mixed, c, .001, options), g = await pairSolver.step(d, .001, options);
        assert.ok(cpu.converged && g.converged, `step ${step}: ${g.maximumUpdate_m}`);
        acceptedInFloat64(mixed, before, d, .001, options);
        assert.ok(maximumDifference(c.positions_m, d.positions_m) < 1e-6, `step ${step}: ${maximumDifference(c.positions_m, d.positions_m)} m (refined ${g.refinementIterations})`);
        assert.ok(maximumDifference(c.twist_rad, d.twist_rad) < 1e-9);
      }
      assert.ok(d.positions_m[3*offset + 16]! < 1.02, 'cohesion pulls the upper sheet toward the lower one');
    } finally {pairSolver.destroy();}
  });
});

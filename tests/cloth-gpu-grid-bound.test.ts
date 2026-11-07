import test from 'node:test';
import assert from 'node:assert/strict';
import {GPUClothNewton} from '../lib/cloth/gpu-newton';
import {clothContactStatic} from '../lib/cloth/gpu-contact';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {createClothState, prepareClothTopology, type PreparedClothTopology} from '../lib/cloth/topology';
import {ClothContactSet} from '../lib/cloth/contact-constraints';
import type {ClothState, ClothTopologyData} from '../lib/cloth/types';
import {withUniformDevice} from './helpers/uniform-geometric';

/** A 5 x 5 sheet of spacing 5 mm, two corners pinned. */
function sheet(): ClothTopologyData {
  const n = 5, h = .005, restPositions_m: number[] = [], triangles: number[] = [], fixed: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {restPositions_m.push(i*h, 1, j*h); fixed.push(j === 0 && (i === 0 || i === n - 1) ? 1 : 0);}
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {const a = j*n + i, b = a + 1, c = a + n, d = c + 1; triangles.push(a, c, b, b, c, d);}
  return {restPositions_m, triangles, rodOffsets: [0], rodIndices: [], fixed};
}
const gpu = {skip: !process.env.WEBGPU_NODE_MODULE};
const options = {iterations: 64, tolerance_m: 1e-8, selfCollision: true};

test('An unknown authored contact grid bound fails explicitly', () => {
  const topology = prepareClothTopology(sheet(), [DEFAULT_CLOTH_MATERIAL]);
  assert.throws(() => new GPUClothNewton({} as GPUDevice, topology, {contactGridEntries: 'three-cells-per-axis' as never}), /Invalid authored cloth contactGridEntries three-cells-per-axis/);
});

/** The end-of-step positions every first detection uses: the prediction, fixed vertices at their targets. */
function predicted(topology: PreparedClothTopology, state: ClothState, dt: number): Float64Array {
  const x = state.positions_m.slice(), target = topology.data.restPositions_m;
  for (let i = 0; i < topology.vertexCount; i++) for (let a = 0; a < 3; a++) {const k = 3*i + a; x[k] = (topology.kinematicFixed[i] ?? 0) & 1 ? target[k]! : x[k]! + dt*state.velocities_m_s[k]! + dt*dt*(a === 1 ? -9.81 : 0);}
  return x;
}
/** The GPU's first detection is the Float64 ClothContactSet's, key for key, inside 8 grid entries a
 * primitive, in a cell wider than every swept, padded box; returns the set's size and that cell. */
async function sameFirstDetection(solver: GPUClothNewton, topology: PreparedClothTopology, state: ClothState, dt: number): Promise<{detected: number; cell: number}> {
  const reference = new ClothContactSet(topology, state.positions_m.slice(), dt, [], true, options.tolerance_m);
  reference.detect(predicted(topology, state, dt));
  const first = await solver.detectContacts(state, dt, options), primitives = topology.triangles.length + clothContactStatic(topology).edges;
  assert.equal(first.errors, 0, `detection error bits ${first.errors}`);
  assert.deepEqual(first.contacts.map(c => c.key).sort(), reference.constraints.map(c => c.key).sort());
  assert.ok(first.gridEntries > 0 && first.gridEntries <= 8 * primitives, `${first.gridEntries} grid entries for ${primitives} primitives`);
  return {detected: reference.constraints.length, cell: first.gridCell_m};
}

test('The two-cells-per-axis grid bound (the default) holds 8 entries a primitive however far a primitive sweeps: the cell follows the widest box', gpu, async () => {
  await withUniformDevice('cloth grid bound', async device => {
    const topology = prepareClothTopology(sheet(), [DEFAULT_CLOTH_MATERIAL]), statics = clothContactStatic(topology);
    const primitives = topology.triangles.length + statics.edges, rest = Math.fround(statics.cellSize);
    // Falling under gravity: every swept box is far narrower than the rest cell, which is the grid's cell.
    const solver = new GPUClothNewton(device, topology, {contactGridEntries: 'two-cells-per-axis'}), state = createClothState(topology);
    try {
      await solver.initialize();
      for (let step = 0; step < 3; step++) {
        const report = await solver.step(state, .001, options);
        assert.ok(report.converged, `step ${step}: ${report.maximumUpdate_m} m`);
        assert.ok(report.contactGridFill! > 0 && report.contactGridFill! <= 8 * primitives, `step ${step}: ${report.contactGridFill} grid entries for ${primitives} primitives`);
        assert.equal(report.contactGridCell_m, rest, `step ${step}: the cell grew`);
      }
    } finally {solver.destroy();}
    // One vertex travels 30 mm in the predicted step, more than the 11.44 mm rest cell: the first
    // detection's cell grows past its incident faces' swept boxes, so the step holds the bound and runs.
    // No authored grid size: the bound is the default.
    const fast = new GPUClothNewton(device, topology), moving = createClothState(topology);
    moving.velocities_m_s[3*12 + 1] = -30;
    try {
      await fast.initialize();
      const first = await sameFirstDetection(fast, topology, moving, .001);
      assert.equal(first.detected, 0);
      assert.ok(first.cell > .03 && first.cell > rest, `cell ${first.cell} m against the ${rest} m rest cell`);
      const report = await fast.step(moving, .001, options);
      assert.ok(report.converged, `${report.maximumUpdate_m} m`);
      assert.ok(report.contactGridFill! > 0 && report.contactGridFill! <= 8 * primitives, `${report.contactGridFill} grid entries for ${primitives} primitives`);
      assert.ok(moving.positions_m.every(Number.isFinite));
    } finally {fast.destroy();}
  });
});

test('A grown grid cell detects the Float64 contact set: a fast vertex thrown through a fixed sheet below', gpu, async () => {
  await withUniformDevice('cloth grid bound parity', async device => {
    const h = .005, lower = sheet(), upper = sheet();
    const data: ClothTopologyData = {
      restPositions_m: [...lower.restPositions_m, ...upper.restPositions_m.map((v, k) => v + (k % 3 === 0 ? .4*h : k % 3 === 1 ? .002 : .3*h))],
      triangles: [...lower.triangles, ...upper.triangles.map(v => v + 25)], rodOffsets: [0], rodIndices: [], fixed: [...Array(25).fill(1), ...Array(25).fill(0)],
    };
    const topology = prepareClothTopology(data, [DEFAULT_CLOTH_MATERIAL]), statics = clothContactStatic(topology), state = createClothState(topology);
    const primitives = topology.triangles.length + statics.edges;
    // The upper sheet's centre vertex at 30 m/s down: 30 mm in the step, through the lower sheet 2 mm below.
    state.velocities_m_s[3*(25 + 12) + 1] = -30;
    const solver = new GPUClothNewton(device, topology);
    try {
      await solver.initialize();
      const {detected, cell} = await sameFirstDetection(solver, topology, state, .001);
      assert.ok(detected > 0, 'the fast vertex reached the lower sheet');
      assert.ok(cell > .03 && cell > Math.fround(statics.cellSize), `cell ${cell} m against the ${statics.cellSize} m rest cell`);
      const report = await solver.step(state, .001, options);
      assert.ok(report.contactGridFill! > 0 && report.contactGridFill! <= 8 * primitives, `${report.contactGridFill} grid entries for ${primitives} primitives`);
      assert.ok(state.positions_m[3*(25 + 12) + 1]! > 1, `the fast vertex crossed the lower sheet: ${state.positions_m[3*(25 + 12) + 1]} m`);
      console.log(`grown cell parity: ${detected} detected in a ${cell} m cell, converged ${report.converged}, last cell ${report.contactGridCell_m} m, ${report.contactGridFill} grid entries for ${primitives} primitives`);
    } finally {solver.destroy();}
  });
});

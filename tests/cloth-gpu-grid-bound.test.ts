import test from 'node:test';
import assert from 'node:assert/strict';
import {GPUClothNewton} from '../lib/cloth/gpu-newton';
import {clothContactStatic} from '../lib/cloth/gpu-contact';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {createClothState, prepareClothTopology} from '../lib/cloth/topology';
import type {ClothTopologyData} from '../lib/cloth/types';
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

test('The two-cells-per-axis grid bound holds 8 entries a primitive and fails a box as wide as the cell, naming it', gpu, async () => {
  await withUniformDevice('cloth grid bound', async device => {
    const topology = prepareClothTopology(sheet(), [DEFAULT_CLOTH_MATERIAL]), statics = clothContactStatic(topology);
    const primitives = topology.triangles.length + statics.edges;
    // Falling under gravity: every swept box is far narrower than the cell, so the fill stays inside 8 per primitive.
    const solver = new GPUClothNewton(device, topology, {contactGridEntries: 'two-cells-per-axis'}), state = createClothState(topology);
    try {
      await solver.initialize();
      for (let step = 0; step < 3; step++) {
        const report = await solver.step(state, .001, options);
        assert.ok(report.converged, `step ${step}: ${report.maximumUpdate_m} m`);
        assert.ok(report.contactGridFill! > 0 && report.contactGridFill! <= 8 * primitives, `step ${step}: ${report.contactGridFill} grid entries for ${primitives} primitives`);
      }
    } finally {solver.destroy();}
    // One vertex travels 30 mm in the step, more than the 11.44 mm cell: the premise fails, and the step names it.
    const fast = new GPUClothNewton(device, topology, {contactGridEntries: 'two-cells-per-axis'}), moving = createClothState(topology);
    moving.velocities_m_s[3*12 + 1] = -30;
    try {
      await fast.initialize();
      await assert.rejects(fast.step(moving, .001, options), (e: Error) => {
        assert.match(e.message, /a swept primitive box at least the grid cell wide \(the two-cells-per-axis grid bound\)/);
        const m = /; (triangle|edge) (\d+) has a swept box ([\d.e+-]+) m wide against the ([\d.e+-]+) m cell/.exec(e.message);
        assert.ok(m, e.message);
        assert.ok(Number(m[3]) >= Number(m[4]) && Math.abs(Number(m[4]) - statics.cellSize) < 1e-12, e.message);
        if (m[1] === 'triangle') assert.ok(topology.triangles[Number(m[2])]!.vertices.includes(12), `triangle ${m[2]} does not hold the moving vertex`);
        return true;
      });
    } finally {fast.destroy();}
  });
});

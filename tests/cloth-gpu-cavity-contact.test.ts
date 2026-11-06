import test from 'node:test';
import assert from 'node:assert/strict';
import {GPUClothNewton} from '../lib/cloth/gpu-newton';
import {clothStaticColliderRecords} from '../lib/cloth/gpu-contact';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {createClothState, prepareClothTopology} from '../lib/cloth/topology';
import {solveClothIncrement} from '../lib/cloth/nonlinear-solve';
import type {ClothCollider} from '../lib/cloth/contact';
import type {ClothTopologyData, ClothVector} from '../lib/cloth/types';
import {withUniformDevice} from './helpers/uniform-geometric';

/** A thin cavity in the GPU's static solid records: the representation of the
 * dam-break tank (clothSceneStaticColliders), whose medial plane once read as
 * a crossing. One brick of 8 cells: walls of 2 cells at |z| = 2h, the medial
 * plane z = 0 on a node row, so the trilinear field is exactly 2h - |z|. A path
 * into the solid (the crossing test) is the oracle's, in cloth-contact-constraints.test.ts:
 * no GPU solve drives a free vertex through a wall, and a fixed one has no contact. */

const gpu = {skip: !process.env.WEBGPU_NODE_MODULE};
const h = .0018, wall = 2*h, r = .5*DEFAULT_CLOTH_MATERIAL.thickness_m;
function cavity(): ClothCollider {
  const values: number[] = [];
  for (let z = 0; z < 9; z++) for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) values.push(2 - Math.abs(z - 4));
  // The same field for the Float64 oracle: inside the brick exactly the trilinear values, past
  // its z faces the record's max(box, value - box) (a 2-cell wall slab, then free space).
  const distance = (p: ClothVector) => {const a = Math.abs(p[2]); return a <= 4*h ? wall - a : Math.max(a - 4*h, -2*h - (a - 4*h));};
  const collider: ClothCollider = {kind: 'sampled', maximumSpeed_m_s: 0, spatialLipschitz: 1, velocityAt: () => [0, 0, 0], distanceAt: distance,
    sample: p => ({distance_m: distance(p), normal: [0, 0, Math.abs(p[2]) <= 4*h ? (p[2] > 0 ? -1 : 1) : Math.sign(p[2])]})};
  clothStaticColliderRecords.set(collider, {h, records: [{min: [0, 0, -4*h], max: [8*h, 8*h, 4*h], values, lipschitz: 1}], cuts: 0});
  return collider;
}
/** A 3 x 3 sheet in the plane z = z0, inside the brick in x and y. */
function sheet(z0: number): ClothTopologyData {
  const restPositions_m: number[] = [], triangles: number[] = [];
  for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) restPositions_m.push((3 + i)*h, (3 + j)*h, z0);
  for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {const a = 3*j + i, b = a + 1, c = a + 3, d = c + 1; triangles.push(a, b, c, b, d, c);}
  return {restPositions_m, triangles, rodOffsets: [0], rodIndices: [], fixed: Array(9).fill(0)};
}

test('GPU contact: a sheet leaving a thin cavity\'s wall past its medial plane steps clean, as in the oracle', gpu, async () => {
  // Touching the +z wall and leaving at 5 m/s: 1.6 mm past the medial plane at the step's end,
  // where the nearest wall (and normal) is the opposite one, the path in free space throughout.
  const dt = .001, speed = 5, collider = cavity(), options = {iterations: 64, tolerance_m: 1e-9, colliders: [collider], gravity_m_s2: [0, 0, 0] as ClothVector};
  const topology = prepareClothTopology(sheet(wall - r), [DEFAULT_CLOTH_MATERIAL]), a = createClothState(topology), b = createClothState(topology);
  for (const s of [a, b]) for (let i = 0; i < 9; i++) s.velocities_m_s[3*i + 2] = -speed;
  const cpu = solveClothIncrement(topology, a, dt, options);
  assert.ok(cpu.converged);
  await withUniformDevice('cloth GPU cavity contact', async device => {
    const solver = new GPUClothNewton(device, topology);
    try {
      await solver.initialize();
      const report = await solver.step(b, dt, options);
      assert.ok(report.converged);
      assert.equal(report.contactModel, 'in-solve');
      assert.equal(report.contactDetection, 'gpu', report.contactFallback);
      for (let i = 0; i < 9; i++) {
        const z = b.positions_m[3*i + 2]!;
        assert.ok(Math.abs(z - (wall - r - speed*dt)) < 1e-6, `vertex ${i} at z ${z}`);
        assert.ok(Math.abs(z - a.positions_m[3*i + 2]!) < 1e-6);
      }
    } finally {solver.destroy();}
  });
});

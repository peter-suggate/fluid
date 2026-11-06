import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_CLOTH_MATERIAL, rodSegmentInertia_kg_m2} from '../lib/cloth/material';
import {createClothState, prepareClothTopology, type PreparedClothTopology} from '../lib/cloth/topology';
import {solveClothIncrement} from '../lib/cloth/nonlinear-solve';
import {closestSegmentPoints, type ClothCollider} from '../lib/cloth/contact';
import {sampleClippedRodCollider} from '../lib/cloth/clipped-contact';
import {sampleClothCollider} from '../lib/cloth/contact';
import type {ClothMaterial, ClothState, ClothTopologyData, ClothVector} from '../lib/cloth/types';

/** Round, uncut DER rods inside the Float64 Newton minimization: rod-vertex
 * spheres and segment interiors against colliders and each other, with the
 * friction slip of the rod's material point (endpoint couple and twist). */

const point = (x: ArrayLike<number>, i: number): ClothVector => [x[3*i]!, x[3*i + 1]!, x[3*i + 2]!];
/** Straight rods along `direction` from each origin, n vertices of spacing h. */
function rods(n: number, h: number, ...lines: readonly {origin: ClothVector; direction: ClothVector; fixed?: number}[]): ClothTopologyData {
  const restPositions_m: number[] = [], fixed: number[] = [], rodOffsets = [0];
  for (const line of lines) {
    for (let i = 0; i < n; i++) {for (let a = 0; a < 3; a++) restPositions_m.push(line.origin[a]! + i*h*line.direction[a]!); fixed.push(line.fixed ?? 0);}
    rodOffsets.push(rodOffsets.at(-1)! + n);
  }
  return {restPositions_m, triangles: [], rodOffsets, rodIndices: [...Array(rodOffsets.at(-1)!).keys()], fixed};
}
const material = (patch: Partial<ClothMaterial>): ClothMaterial => ({...DEFAULT_CLOTH_MATERIAL, rodRadius_m: .002, ...patch});
function run(topology: PreparedClothTopology, state: ClothState, steps: number, options: Parameters<typeof solveClothIncrement>[3], each?: (step: number) => void) {
  for (let step = 0; step < steps; step++) {
    const report = solveClothIncrement(topology, state, .001, {...options, rodContactOracle: true});
    assert.equal(report.contactModel, 'in-solve');
    assert.ok(report.converged, `step ${step}: residual ${report.maximumUpdate_m} m after ${report.contactIterations} contact rounds`);
    each?.(step);
  }
}

test('a round rod lands on a plane and rests on its radius, in the solve', () => {
  const r = .002, topology = prepareClothTopology(rods(6, .01, {origin: [0, r + .001, 0], direction: [1, 0, 0]}), [material({})]), state = createClothState(topology);
  for (let i = 0; i < topology.vertexCount; i++) state.velocities_m_s[3*i + 1] = -1;
  let contacts = 0;
  run(topology, state, 30, {iterations: 64, tolerance_m: 1e-9, colliders: [{kind: 'plane', normal: [0, 1, 0], offset_m: 0}]}, () => {
    for (let i = 0; i < topology.vertexCount; i++) assert.ok(state.positions_m[3*i + 1]! > r - 1e-7, `vertex ${i} at y = ${state.positions_m[3*i + 1]}`);
    contacts++;
  });
  for (let i = 0; i < topology.vertexCount; i++) assert.ok(Math.abs(state.positions_m[3*i + 1]! - r) < 1e-5, `vertex ${i} rests at y = ${state.positions_m[3*i + 1]}`);
  assert.equal(contacts, 30);
});

test('a long segment draped over a sphere is held by its interior, away from both rod-vertex spheres', () => {
  // One 20 cm segment over a 5 cm sphere: its ends never come near the
  // sphere, so only the segment-interior pair (rc:) can hold it.
  const r = .002, R = .05, topology = prepareClothTopology(rods(2, .2, {origin: [-.1, R + r + .0005, 0], direction: [1, 0, 0]}), [material({})]), state = createClothState(topology);
  const sphere: ClothCollider = {kind: 'sphere', center_m: [0, 0, 0], radius_m: R};
  run(topology, state, 40, {iterations: 64, tolerance_m: 1e-9, colliders: [sphere]}, step => {
    const gap = sampleClippedRodCollider({a: point(state.positions_m, 0), b: point(state.positions_m, 1), radius_m: r, planes: []}, sphere, 0, sampleClothCollider).distance_m;
    assert.ok(gap > -1e-7, `step ${step}: capsule penetrates the sphere by ${-gap} m`);
  });
  const middle = (state.positions_m[1]! + state.positions_m[4]!)/2;
  assert.ok(Math.abs(middle - (R + r)) < 1e-5, `the segment's middle rests at y = ${middle}, on the sphere`);
});

test('a spinning rod on a frictional floor rolls: its material point sticks, and angular momentum about the contact line holds', () => {
  const r = .002, omega = 50, steps = 30;
  const make = (friction: number) => {
    const topology = prepareClothTopology(rods(6, .01, {origin: [0, r, 0], direction: [1, 0, 0]}), [material({})]), state = createClothState(topology);
    state.angularVelocity_rad_s.fill(omega);
    const options = {iterations: 64, tolerance_m: 1e-9, colliders: [{kind: 'plane', normal: [0, 1, 0], offset_m: 0, friction}] as ClothCollider[]};
    return {topology, state, options};
  };
  const momentum = (topology: PreparedClothTopology, state: ClothState) => {
    // About the floor line under the axis: spin I omega plus m r v_z of the centreline.
    let spin = 0, orbital = 0, mass = 0, vz = 0;
    for (let s = 0; s < topology.segmentCount; s++) spin += rodSegmentInertia_kg_m2(topology, state, 0, s)*state.angularVelocity_rad_s[s]!;
    for (let i = 0; i < topology.vertexCount; i++) {const m = state.dryMass_kg[i]!; orbital += m*state.positions_m[3*i + 1]!*state.velocities_m_s[3*i + 2]!; mass += m; vz += m*state.velocities_m_s[3*i + 2]!;}
    return {total: spin + orbital, vz: vz/mass};
  };
  const rough = make(.8), before = momentum(rough.topology, rough.state);
  run(rough.topology, rough.state, steps, rough.options);
  const after = momentum(rough.topology, rough.state), spin = rough.state.angularVelocity_rad_s.reduce((a, b) => a + b, 0)/rough.topology.segmentCount;
  // Rolling without slip: v = r omega along t x n = +z; the solid section keeps omega/3.
  assert.ok(after.vz > 0, `rolls toward +z (v = ${after.vz} m/s)`);
  assert.ok(Math.abs(after.vz - r*spin) < .05*r*spin, `material point sticks: v = ${after.vz}, r omega = ${r*spin}`);
  assert.ok(Math.abs(spin - omega/3) < .05*omega/3, `spin ${spin} rad/s, rolling cylinder ${omega/3}`);
  assert.ok(Math.abs(after.total - before.total) < .02*Math.abs(before.total), `angular momentum about the contact line ${before.total} -> ${after.total}`);
  const smooth = make(0);
  run(smooth.topology, smooth.state, steps, smooth.options);
  const free = momentum(smooth.topology, smooth.state);
  assert.ok(Math.abs(free.vz) < 1e-9, `frictionless floor: no rolling (v = ${free.vz})`);
  for (const w of smooth.state.angularVelocity_rad_s) assert.ok(Math.abs(w - omega) < 1e-6*omega, `frictionless spin ${w}`);
});

test('a rod falling across a fixed rod is stopped by the segment-segment pair between their vertices', () => {
  const r = .002, h = .025, topology = prepareClothTopology(rods(5, h,
    {origin: [-.05, 0, .0125], direction: [1, 0, 0], fixed: 3},
    {origin: [.0125, 2*r + .002, -.05], direction: [0, 0, 1]}), [material({})]), state = createClothState(topology);
  for (let i = 5; i < 10; i++) state.velocities_m_s[3*i + 1] = -1;
  let lowest = Infinity;
  run(topology, state, 15, {iterations: 64, tolerance_m: 1e-9, selfCollision: true}, step => {
    for (let s = 0; s < 4; s++) for (let u = 0; u < 4; u++) {
      const q = closestSegmentPoints(point(state.positions_m, s), point(state.positions_m, s + 1), point(state.positions_m, 5 + u), point(state.positions_m, 6 + u));
      const d = Math.hypot(q.a[0] - q.b[0], q.a[1] - q.b[1], q.a[2] - q.b[2]);
      lowest = Math.min(lowest, d);
      assert.ok(d > 2*r - 1e-7, `step ${step}: segments ${s}, ${u} overlap (${d} m apart)`);
    }
  });
  assert.ok(lowest < 2*r + 1e-5, `the rods met (closest ${lowest} m)`);
});

test('a rod finer than its radius has no self contact with its own rest neighbourhood', () => {
  const r = .002, topology = prepareClothTopology(rods(12, .001, {origin: [0, .1, 0], direction: [1, 0, 0]}), [material({})]), state = createClothState(topology);
  const report = solveClothIncrement(topology, state, .001, {iterations: 64, tolerance_m: 1e-9, selfCollision: true, rodContactOracle: true});
  assert.equal(report.contactModel, 'in-solve');
  assert.ok(report.converged);
  assert.equal(report.contacts.selfContacts, 0);
  assert.ok(r > .001);
});

test('a rod rolling as it strikes a pinned sheet converges quadratically: its twist and positions are one CG system', () => {
  // Friction ties the twist to the positions through the rolling slip; stepped
  // as separate blocks (the twist after the position CG) this impact took 258
  // Newton iterations at 1e-8 m, and 128 without converging at 1e-7 m the step after.
  const n = 6, hs = .02, restPositions_m: number[] = [], triangles: number[] = [], fixed: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {restPositions_m.push(i*hs, 0, j*hs); fixed.push(1);}
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {const q = j*n + i; triangles.push(q, q + n, q + 1, q + 1, q + n, q + n + 1);}
  const r = .002, rodStart = n*n, count = 9, friction = Math.tan(10*Math.PI/180);
  for (let k = 0; k < count; k++) {restPositions_m.push(.013 + .0095*k, r + .0035, .017 + .0071*k); fixed.push(0);}
  const data: ClothTopologyData = {restPositions_m, triangles, triangleMaterials: Array(triangles.length/3).fill(0), rodOffsets: [0, count], rodIndices: Array.from({length: count}, (_, k) => rodStart + k), rodMaterials: [1], fixed};
  const topology = prepareClothTopology(data, [{...DEFAULT_CLOTH_MATERIAL, thickness_m: .001, friction}, material({friction})]), state = createClothState(topology);
  for (let k = 0; k < count; k++) state.velocities_m_s[3*(rodStart + k) + 1] = -2;
  let newton = 0;
  for (let step = 0; step < 4; step++) {
    const report = solveClothIncrement(topology, state, .001, {iterations: 64, tolerance_m: 1e-8, selfCollision: true, rodContactOracle: true});
    assert.ok(report.converged && report.contactModel === 'in-solve', `step ${step}: residual ${report.maximumUpdate_m} m`);
    newton += report.iterations;
  }
  for (let k = 0; k < count; k++) assert.ok(state.positions_m[3*(rodStart + k) + 1]! > r + .0005 - 1e-6, `rod vertex ${k} at y = ${state.positions_m[3*(rodStart + k) + 1]}`);
  assert.ok(newton < 100, `${newton} Newton iterations over the impact`);
});

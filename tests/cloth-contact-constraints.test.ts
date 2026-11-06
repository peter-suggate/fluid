import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import { createClothState, prepareClothTopology, type PreparedClothTopology } from '../lib/cloth/topology';
import { clothElasticEnergy } from '../lib/cloth/dynamics';
import { solveClothIncrement } from '../lib/cloth/nonlinear-solve';
import { ClothContactSet, inSolveContactUnsupported } from '../lib/cloth/contact-constraints';
import { sourceClothColliders } from '../lib/cloth/source-contact';
import type { WetClothSourceField } from '../lib/cloth/source-types';
import { sampleClothCollider, type ClothCollider } from '../lib/cloth/contact';
import type { ClothState, ClothTopologyData } from '../lib/cloth/types';

/** n x n sheets of spacing h; each sheet is [origin, fixed mask]. */
function sheets(n: number, h: number, ...parts: readonly [readonly [number, number, number], number][]): ClothTopologyData {
  const restPositions_m: number[] = [], triangles: number[] = [], fixed: number[] = [];
  for (const [origin, mask] of parts) {
    const base = restPositions_m.length/3;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {restPositions_m.push(origin[0] + i*h, origin[1], origin[2] + j*h); fixed.push(mask);}
    for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {const a = base + j*n + i, b = a + 1, c = a + n, d = c + 1; triangles.push(a, c, b, b, c, d);}
  }
  return {restPositions_m, triangles, rodOffsets: [0], rodIndices: [], fixed};
}
/** Kinetic + gravitational + elastic energy (J). */
function energy(topology: PreparedClothTopology, state: ClothState): number {
  let e = clothElasticEnergy(topology, state);
  for (let i = 0; i < topology.vertexCount; i++) {
    const m = state.dryMass_kg[i]!, v = state.velocities_m_s;
    e += .5*m*(v[3*i]!**2 + v[3*i+1]!**2 + v[3*i+2]!**2) + m*9.81*state.positions_m[3*i+1]!;
  }
  return e;
}

test('in-solve contact drapes a sheet on a sphere: no penetration, velocity from final positions, energy bounded', () => {
  const topology = prepareClothTopology(sheets(7, .01, [[0, 1, 0], 0]), [DEFAULT_CLOTH_MATERIAL]), state = createClothState(topology);
  const sphere: ClothCollider = {kind: 'sphere', center_m: [.03, .97, .03], radius_m: .02}, radius = .5*DEFAULT_CLOTH_MATERIAL.thickness_m, dt = .001;
  const initial = energy(topology, state);
  let contacts = 0;
  for (let step = 0; step < 120; step++) {
    const old = state.positions_m.slice();
    const report = solveClothIncrement(topology, state, dt, {iterations: 64, tolerance_m: 1e-9, colliders: [sphere]});
    assert.ok(report.converged, `step ${step}: ${report.maximumUpdate_m} m`);
    assert.equal(report.contactModel, 'in-solve');
    contacts = Math.max(contacts, report.contacts.contacts);
    for (let i = 0; i < topology.vertexCount; i++) {
      const p = [0, 1, 2].map(a => state.positions_m[3*i+a]! - sphere.center_m[a]!);
      assert.ok(Math.hypot(...p) - sphere.radius_m >= radius - 1e-8, `step ${step}: vertex ${i} penetrates`);
      // No projection after the solve: the stored velocity is that of the final positions.
      for (let a = 0; a < 3; a++) assert.ok(Math.abs(state.velocities_m_s[3*i+a]! - (state.positions_m[3*i+a]! - old[3*i+a]!)/dt) < 1e-9);
    }
    assert.ok(energy(topology, state) <= initial + 1e-9, `step ${step}: energy rose by ${energy(topology, state) - initial} J`);
  }
  assert.ok(contacts > 0, 'the sheet reached the sphere');
  const centre = 3*24;
  assert.ok(Math.abs(state.positions_m[centre + 1]! - (sphere.center_m[1] + sphere.radius_m + radius)) < 1e-6, 'the centre rests on the sphere top');
});

test('in-solve self contact stops a falling sheet on a fixed sheet at the shell thickness', () => {
  const h = .005, gap = .002, thickness = DEFAULT_CLOTH_MATERIAL.thickness_m;
  const topology = prepareClothTopology(sheets(5, h, [[0, 1, 0], 1], [[.4*h, 1 + gap, .3*h], 0]), [DEFAULT_CLOTH_MATERIAL]), state = createClothState(topology);
  let selfContacts = 0, lowest = Infinity;
  for (let step = 0; step < 60; step++) {
    const report = solveClothIncrement(topology, state, .001, {iterations: 64, tolerance_m: 1e-9, selfCollision: true});
    assert.ok(report.converged, `step ${step}: ${report.maximumUpdate_m} m`);
    assert.equal(report.contactModel, 'in-solve');
    selfContacts = Math.max(selfContacts, report.contacts.selfContacts);
    // Upper-sheet vertices over the lower sheet's interior never cross it.
    for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) lowest = Math.min(lowest, state.positions_m[3*(25 + 5*j + i) + 1]!);
  }
  assert.ok(selfContacts > 0, 'the sheets met');
  assert.ok(lowest > 1 + thickness - 1e-7, `upper sheet reached y = ${lowest}`);
  assert.ok(lowest < 1 + thickness + 1e-5, `upper sheet rests at y = ${lowest}, not on the lower sheet`);
});

test('elliptical rods keep the sequential contact projection and say so; round rods are in-solve only in the oracle', () => {
  const n = 5, restPositions_m: number[] = [];
  for (let i = 0; i < n; i++) restPositions_m.push(i*.005, .001, 0);
  const data = {restPositions_m, triangles: [], rodOffsets: [0, n], rodIndices: [...Array(n).keys()], fixed: Array(n).fill(0)}, colliders: ClothCollider[] = [{kind: 'plane', normal: [0, 1, 0], offset_m: 0}];
  const elliptical = prepareClothTopology(data, [{...DEFAULT_CLOTH_MATERIAL, rodRadiusB_m: 2*DEFAULT_CLOTH_MATERIAL.rodRadius_m}]);
  assert.match(inSolveContactUnsupported(elliptical, colliders)!, /elliptical/);
  assert.equal(solveClothIncrement(elliptical, createClothState(elliptical), .001, {iterations: 64, tolerance_m: 1e-9, colliders}).contactModel, 'post-projection');
  const round = prepareClothTopology(data, [DEFAULT_CLOTH_MATERIAL]);
  assert.equal(solveClothIncrement(round, createClothState(round), .001, {iterations: 64, tolerance_m: 1e-9, colliders}).contactModel, 'post-projection');
  assert.equal(solveClothIncrement(round, createClothState(round), .001, {iterations: 64, tolerance_m: 1e-9, colliders, rodContactOracle: true}).contactModel, 'in-solve');
});

test('an unknown collider kind fails explicitly in the in-solve contact path', () => {
  const topology = prepareClothTopology(sheets(3, .01, [[0, 1, 0], 0]), [DEFAULT_CLOTH_MATERIAL]);
  assert.throws(() => solveClothIncrement(topology, createClothState(topology), .001, {colliders: [{kind: 'torus'} as unknown as ClothCollider]}), /Unsupported cloth collider/);
});

/** The dam-break tank of libWetCloth's drag scene (assets/drag): a rounded box
 * cavity 2.88 cm deep in z, its walls at |z| = 1.44 cm and medial plane z = 0. */
const thinCavity: WetClothSourceField = {index: 0, kind: 'box', usage: 'solid', group: 2, inside: true, sampled: false, root: true,
  center_m: [0, .19728, 0], axis: [0, 1, 0], angle_rad: 0, halfExtents_m: [.144, .144, .0072], roundingRadius_m: .0072, sourceAttributes: {}};
/** An n x n sheet in the plane z = z0, spacing h, its corner at (x0, y0). */
function wallSheet(n: number, h: number, x0: number, y0: number, z0: number): ClothTopologyData {
  const restPositions_m: number[] = [], triangles: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) restPositions_m.push(x0 + i*h, y0 + j*h, z0);
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {const a = j*n + i, b = a + 1, c = a + n, d = c + 1; triangles.push(a, b, c, b, d, c);}
  return {restPositions_m, triangles, rodOffsets: [0], rodIndices: [], fixed: restPositions_m.filter((_, k) => k % 3 === 0).map(() => 0)};
}

test('a thin cavity\'s medial plane is no crossing: a sheet leaving one wall past it keeps its contact', () => {
  // The sheet touches the +z wall and leaves it at 2 m/s, ending the step 5.8 mm past the medial
  // plane. Its contacts were detected against the +z wall, and the nearest wall at the end is -z:
  // the nearest normal turns over with the path in free space all the way.
  const r = .5*DEFAULT_CLOTH_MATERIAL.thickness_m, z0 = .0144 - r, dt = .01, speed = 2;
  const topology = prepareClothTopology(wallSheet(3, .002, .05, .15, z0), [DEFAULT_CLOTH_MATERIAL]), state = createClothState(topology);
  for (let i = 0; i < topology.vertexCount; i++) state.velocities_m_s[3*i + 2] = -speed;
  const [cavity] = sourceClothColliders([thinCavity], [thinCavity], dt);
  const report = solveClothIncrement(topology, state, dt, {iterations: 64, tolerance_m: 1e-9, colliders: [cavity!], gravity_m_s2: [0, 0, 0]});
  assert.ok(report.converged);
  assert.equal(report.contactModel, 'in-solve');
  for (let i = 0; i < topology.vertexCount; i++) {
    const z = state.positions_m[3*i + 2]!;
    // Free flight: nothing touched, so the rigid translation is the whole motion.
    assert.ok(Math.abs(z - (z0 - speed*dt)) < 1e-9, `vertex ${i} at z ${z}`);
    assert.ok(z < 0 && sampleClothCollider(cavity!, [state.positions_m[3*i]!, state.positions_m[3*i + 1]!, z]).distance_m > r);
  }
});

test('the same cavity with the cloth driven into its wall fails explicitly, as does a crossing of a thin plate', () => {
  const r = .5*DEFAULT_CLOTH_MATERIAL.thickness_m, z0 = .0144 - r, dt = .001;
  const topology = prepareClothTopology(wallSheet(3, .002, .05, .15, z0), [DEFAULT_CLOTH_MATERIAL]), old = Float64Array.from(topology.data.restPositions_m);
  const [cavity] = sourceClothColliders([thinCavity], [thinCavity], dt), set = new ClothContactSet(topology, old, dt, [cavity!], false, 1e-9);
  assert.equal(set.detect(old), topology.vertexCount, 'each vertex touches the +z wall');
  // 3 mm into the wall: the nearest normal has not turned, but the path entered the solid.
  assert.throws(() => set.relinearize(old.map((v, k) => k % 3 === 2 ? .0174 : v)), /Cloth vertex 0 crossed collider 0 within the step/);
  // Through a thin plate the nearest normal does turn; the path is what fails it.
  const plate: ClothCollider = {kind: 'box', minimum_m: [-1, -1e-4, -1], maximum_m: [1, 1e-4, 1]};
  const sheet = prepareClothTopology(sheets(2, .002, [[0, 1e-4 + r, 0], 0]), [DEFAULT_CLOTH_MATERIAL]), above = Float64Array.from(sheet.data.restPositions_m);
  const through = new ClothContactSet(sheet, above, dt, [plate], false, 1e-9);
  assert.equal(through.detect(above), sheet.vertexCount);
  assert.throws(() => through.relinearize(above.map((v, k) => k % 3 === 1 ? -.003 : v)), /Cloth vertex 0 crossed collider 0 within the step/);
});

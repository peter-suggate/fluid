import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import { createClothState, prepareClothTopology, type PreparedClothTopology } from '../lib/cloth/topology';
import { clothElasticEnergy } from '../lib/cloth/dynamics';
import { solveClothIncrement } from '../lib/cloth/nonlinear-solve';
import { ClothContactSet, inSolveContactUnsupported } from '../lib/cloth/contact-constraints';
import { gpuContactUnsupported } from '../lib/cloth/gpu-contact';
import { clothRodCapsuleGeometry } from '../lib/cloth/rod-geometry';
import { supportClippedRod } from '../lib/cloth/rod-convex';
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

test('elliptical rods are in-solve in the oracle, but not against sampled solids; round rods are in-solve only in the oracle', () => {
  const n = 5, restPositions_m: number[] = [];
  for (let i = 0; i < n; i++) restPositions_m.push(i*.005, .001, 0);
  const data = {restPositions_m, triangles: [], rodOffsets: [0, n], rodIndices: [...Array(n).keys()], fixed: Array(n).fill(0)}, colliders: ClothCollider[] = [{kind: 'plane', normal: [0, 1, 0], offset_m: 0}];
  const elliptical = prepareClothTopology(data, [{...DEFAULT_CLOTH_MATERIAL, rodRadiusB_m: 2*DEFAULT_CLOTH_MATERIAL.rodRadius_m}]);
  assert.equal(inSolveContactUnsupported(elliptical, colliders), undefined);
  assert.equal(solveClothIncrement(elliptical, createClothState(elliptical), .001, {iterations: 64, tolerance_m: 1e-9, colliders}).contactModel, 'post-projection');
  assert.equal(solveClothIncrement(elliptical, createClothState(elliptical), .001, {iterations: 64, tolerance_m: 1e-9, colliders, rodContactOracle: true}).contactModel, 'in-solve');
  const sampled = {kind: 'sampled', maximumSpeed_m_s: 0, velocityAt: () => [0, 0, 0], sample: () => ({distance_m: 1, normal: [0, 1, 0]})} as unknown as ClothCollider;
  assert.match(inSolveContactUnsupported(elliptical, [sampled])!, /elliptical rods against sampled/);
  const round = prepareClothTopology(data, [DEFAULT_CLOTH_MATERIAL]);
  assert.equal(inSolveContactUnsupported(round, [sampled]), undefined);
  assert.equal(solveClothIncrement(round, createClothState(round), .001, {iterations: 64, tolerance_m: 1e-9, colliders}).contactModel, 'post-projection');
  assert.equal(solveClothIncrement(round, createClothState(round), .001, {iterations: 64, tolerance_m: 1e-9, colliders, rodContactOracle: true}).contactModel, 'in-solve');
});

/** A free rod (radii A = r, B = 2r for an elliptical section: the director
 * starts vertical, so twist 0 stands on A), stepped in-solve with the oracle;
 * returns the state and the most self contacts. The oracle's Newton model has
 * only the diagonal of rod bending, which dominates a light yarn's inertia,
 * so a rolling or sliding rod converges linearly: 1e-8 m within 200 iterations. */
const ROD_R = DEFAULT_CLOTH_MATERIAL.rodRadius_m;
function settleRod(points: readonly number[], options: {radiusB?: number; twist?: number; fixed?: readonly number[]; triangles?: readonly number[]; rodOffsets?: readonly number[]; rodIndices?: readonly number[]; colliders?: ClothCollider[]; selfCollision?: boolean; steps: number;
  /** Per-rod material-frame cut normals (d1, d2, t; zero: uncut). */
  startClip?: readonly number[]; endClip?: readonly number[]}) {
  const count = points.length/3, rodIndices = options.rodIndices ?? [...Array(count).keys()], rodOffsets = options.rodOffsets ?? [0, rodIndices.length];
  const data: ClothTopologyData = {restPositions_m: [...points], triangles: [...(options.triangles ?? [])], rodOffsets: [...rodOffsets], rodIndices: [...rodIndices], fixed: [...(options.fixed ?? Array(count).fill(0))], initialTwist_rad: Array(rodIndices.length - rodOffsets.length + 1).fill(options.twist ?? 0),
    ...(options.startClip ? {rodStartClipNormals: [...options.startClip]} : {}), ...(options.endClip ? {rodEndClipNormals: [...options.endClip]} : {})};
  const topology = prepareClothTopology(data, [{...DEFAULT_CLOTH_MATERIAL, rodRadiusB_m: options.radiusB ?? ROD_R}]), state = createClothState(topology);
  let selfContacts = 0, newton = 0;
  for (let step = 0; step < options.steps; step++) {
    const report = solveClothIncrement(topology, state, .001, {iterations: 200, tolerance_m: 1e-8, colliders: options.colliders, selfCollision: options.selfCollision, rodContactOracle: true});
    assert.ok(report.converged, `step ${step}: ${report.maximumUpdate_m} m`);
    assert.equal(report.contactModel, 'in-solve');
    selfContacts = Math.max(selfContacts, report.contacts.selfContacts);
    newton += report.newtonIterations;
  }
  return {state, selfContacts, topology, newton};
}
const lowestY = (state: ClothState, vertices: readonly number[]) => Math.min(...vertices.map(i => state.positions_m[3*i + 1]!));

test('an elliptical rod rolls onto its minor radius on a plane, a box and a fixed sheet', () => {
  const rod = (y: number, z = 0) => [-.0020, y, z, .0003, y, z, .0026, y, z];
  // A plane and a box top at y = 1 cm: the convex distance agrees with the closed form.
  for (const colliders of [[{kind: 'plane', normal: [0, 1, 0], offset_m: .01}], [{kind: 'box', minimum_m: [-.02, -.01, -.02], maximum_m: [.02, .01, .02]}]] as ClothCollider[][]) {
    const {state} = settleRod(rod(.01 + 4*ROD_R), {radiusB: 2*ROD_R, twist: Math.PI/4, colliders, steps: 200});
    assert.ok(Math.abs(lowestY(state, [0, 1, 2]) - .01 - ROD_R) < 3e-8, `${colliders[0]!.kind}: rests at ${(lowestY(state, [0, 1, 2]) - .01)/ROD_R} r`);
    assert.ok(Math.abs(state.twist_rad[0]!) < 2e-3, `${colliders[0]!.kind}: twist ${state.twist_rad[0]}`);
  }
  // A fixed sheet of shell thickness h: the caps and interiors meet its faces.
  const m = 6, s = .002, sheet: number[] = [], triangles: number[] = [];
  for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) sheet.push((i - 2.5)*s, 0, (j - 2.5)*s);
  for (let j = 0; j + 1 < m; j++) for (let i = 0; i + 1 < m; i++) {const a = j*m + i; triangles.push(a, a + m, a + 1, a + 1, a + m, a + m + 1);}
  const v0 = m*m, {state, selfContacts} = settleRod([...sheet, ...rod(6*ROD_R, .0007)], {radiusB: 2*ROD_R, twist: Math.PI/4, fixed: [...Array(v0).fill(1), 0, 0, 0], triangles, rodIndices: [v0, v0 + 1, v0 + 2], selfCollision: true, steps: 200});
  assert.ok(selfContacts > 0, 'the rod met the sheet');
  assert.ok(Math.abs(lowestY(state, [v0, v0 + 1, v0 + 2]) - .5*DEFAULT_CLOTH_MATERIAL.thickness_m - ROD_R) < 3e-8, `sheet: rests at ${lowestY(state, [v0, v0 + 1, v0 + 2])}`);
  assert.ok(Math.abs(state.twist_rad[0]!) < 2e-3, `sheet: twist ${state.twist_rad[0]}`);
});

test('crossing elliptical segments rest at their vertical radii, and a cap on a sphere at its axial radius', () => {
  // A fixed rod along x and a free single segment along z dropped on its interior.
  const lower = [-.003, 0, 0, -.001, 0, 0, .001, 0, 0, .003, 0, 0], {state, selfContacts} = settleRod([...lower, 0, 4*ROD_R, -.002, 0, 4*ROD_R, .002], {radiusB: 2*ROD_R, fixed: [1, 1, 1, 1, 0, 0], rodOffsets: [0, 4, 6], selfCollision: true, steps: 60});
  assert.ok(selfContacts > 0, 'the segments met');
  const middle = (state.positions_m[13]! + state.positions_m[16]!)/2;
  assert.ok(Math.abs(middle - 2*ROD_R) < 3e-8, `upper segment rests at ${middle/ROD_R} r`);
  // An upright rod falls end first on a sphere: the cap's axial radius is sqrt(AB).
  const upright = settleRod([0, .01 + 3*ROD_R, 0, 0, .012 + 3*ROD_R, 0, 0, .014 + 3*ROD_R, 0], {radiusB: 2*ROD_R, colliders: [{kind: 'sphere', center_m: [0, 0, 0], radius_m: .01}], steps: 60}).state;
  assert.ok(Math.abs(upright.positions_m[1]! - .01 - Math.SQRT2*ROD_R) < 3e-8, `cap rests at ${(upright.positions_m[1]! - .01)/ROD_R} r`);
});

test('a round rod segment lying on a box face rests on its radius', () => {
  // Its interior meets the box by the convex distance; a sampled lower bound
  // degenerates for a segment parallel to the face.
  const {state} = settleRod([-.002, .01 + 4*ROD_R, 0, 0, .01 + 4*ROD_R, 0, .002, .01 + 4*ROD_R, 0], {colliders: [{kind: 'box', minimum_m: [-.02, -.01, -.02], maximum_m: [.02, .01, .02]}], steps: 60});
  assert.ok(Math.abs(lowestY(state, [0, 1, 2]) - .01 - ROD_R) < 3e-8, `rests at ${(lowestY(state, [0, 1, 2]) - .01)/ROD_R} r`);
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

/** The lowest point of a rod's segment as rendering and picking clip it. */
function renderedLowest(topology: PreparedClothTopology, state: ClothState, rod: number, segment: number): number {
  const shape = clothRodCapsuleGeometry(topology, topology.data, state.positions_m, rod, segment, {rodTwist_rad: state.twist_rad, rodDirectors: state.rodDirectors, rodTangents: state.rodTangents});
  return supportClippedRod(shape, [0, -1, 0])[1];
}

test('a straight-cut rod end stands flush on a plane, a box and a sphere, round or elliptical', () => {
  // An upright rod cut square at its foot (start plane normal -t): its face, not a cap, meets the solid at y = 1 cm.
  const upright = [0, .01 + ROD_R, 0, 0, .012 + ROD_R, 0, 0, .014 + ROD_R, 0];
  const solids: ClothCollider[] = [{kind: 'plane', normal: [0, 1, 0], offset_m: .01}, {kind: 'box', minimum_m: [-.02, -.01, -.02], maximum_m: [.02, .01, .02]}, {kind: 'sphere', center_m: [0, 0, 0], radius_m: .01}];
  for (const radiusB of [ROD_R, 2*ROD_R]) {
    let onPlane = 0;
    for (const solid of solids) {
      const {state, topology, newton} = settleRod(upright, {radiusB, colliders: [solid], startClip: [0, 0, -1], steps: 60});
      assert.ok(Math.abs(state.positions_m[1]! - .01) < 3e-8, `${solid.kind}, B ${radiusB/ROD_R} r: the cut foot rests at ${(state.positions_m[1]! - .01)/ROD_R} r`);
      assert.ok(Math.abs(renderedLowest(topology, state, 0, 0) - .01) < 3e-8, `${solid.kind}: the rendered foot at ${renderedLowest(topology, state, 0, 0)}`);
      // A face flush on a box face is the plane's contact (one support point), not GJK's
      // arbitrary closest pair on two parallel faces: the same Newton work as on the plane.
      if (solid.kind === 'plane') onPlane = newton;
      else assert.ok(newton <= 1.1*onPlane, `${solid.kind}, B ${radiusB/ROD_R} r: ${newton} Newton iterations, ${onPlane} on the plane`);
    }
  }
});

test('a cut rod face stands on another cut face as on a plane', () => {
  // A fixed upright rod cut square at its top, and a free one cut square at its foot dropped on it, centred or off centre.
  for (const radiusB of [ROD_R, 2*ROD_R]) for (const dx of [0, 1e-4]) {
    const points = [0, 0, 0, 0, .002, 0, dx, .0022, 0, dx, .0042, 0, dx, .0062, 0];
    const {state, selfContacts, newton} = settleRod(points, {radiusB, fixed: [1, 1, 0, 0, 0], rodOffsets: [0, 2, 5], selfCollision: true, startClip: [0, 0, 0, 0, 0, -1], endClip: [0, 0, 1, 0, 0, 0], steps: 60});
    const plane = settleRod(points.slice(6).map((v, i) => i % 3 === 1 ? v + .008 : v), {radiusB, colliders: [{kind: 'plane', normal: [0, 1, 0], offset_m: .01}], startClip: [0, 0, -1], steps: 60});
    assert.ok(selfContacts > 0, 'the faces met');
    assert.ok(Math.abs(state.positions_m[7]! - .002) < 3e-8, `B ${radiusB/ROD_R} r, dx ${dx}: the foot rests at ${(state.positions_m[7]! - .002)/ROD_R} r`);
    assert.ok(Math.abs(state.positions_m[12]! - dx) < 1e-9, `B ${radiusB/ROD_R} r, dx ${dx}: the top sways ${state.positions_m[12]! - dx} m`);
    assert.ok(newton <= 1.1*plane.newton, `B ${radiusB/ROD_R} r, dx ${dx}: ${newton} Newton iterations, ${plane.newton} on a plane`);
  }
});

test('a cut face rests on another rod, and an oblique cut tips onto its rim as rendering clips it', () => {
  // A fixed rod along x and an upright cut rod dropped on it: the face is tangent to the tube (one radius, not two).
  const lower = [-.003, 0, 0, -.001, 0, 0, .001, 0, 0, .003, 0, 0], upper = [0, ROD_R + 1e-4, 0, 0, ROD_R + .0021, 0, 0, ROD_R + .0041, 0];
  const {state, selfContacts} = settleRod([...lower, ...upper], {fixed: [1, 1, 1, 1, 0, 0, 0], rodOffsets: [0, 4, 7], selfCollision: true, startClip: [0, 0, 0, 0, 0, -1], steps: 60});
  assert.ok(selfContacts > 0, 'the rods met');
  assert.ok(Math.abs(state.positions_m[13]! - ROD_R) < 3e-8, `the cut foot rests at ${state.positions_m[13]!/ROD_R} r`);
  // Cut 40 degrees from square in the d1-t plane: the rim's low point lands first and the rod tips on it, never into the plane.
  const angle = 40*Math.PI/180, plane: ClothCollider = {kind: 'plane', normal: [0, 1, 0], offset_m: .01};
  const {state: tipped, topology} = settleRod([0, .01 + 2*ROD_R, 0, 0, .012 + 2*ROD_R, 0, 0, .014 + 2*ROD_R, 0], {colliders: [plane], startClip: [Math.sin(angle), 0, -Math.cos(angle)], steps: 40});
  assert.ok(Math.abs(renderedLowest(topology, tipped, 0, 0) - .01) < 3e-8, `the rim rests at ${(renderedLowest(topology, tipped, 0, 0) - .01)/ROD_R} r`);
  assert.ok(tipped.positions_m[1]! > .01 + 1e-5, `the cut foot's centre stays above the plane: ${(tipped.positions_m[1]! - .01)/ROD_R} r`);
});

test('a cut end is not on the GPU, and a cut rod against a sampled solid keeps the sequential projection', () => {
  const data: ClothTopologyData = {restPositions_m: [0, 0, 0, .002, 0, 0], triangles: [], rodOffsets: [0, 2], rodIndices: [0, 1], fixed: [0, 0], rodEndClipNormals: [0, 0, 1]};
  const topology = prepareClothTopology(data, [DEFAULT_CLOTH_MATERIAL]);
  assert.equal(inSolveContactUnsupported(topology, [{kind: 'plane', normal: [0, 1, 0], offset_m: 0}]), undefined);
  assert.match(inSolveContactUnsupported(topology, [{kind: 'sampled'} as unknown as ClothCollider])!, /clipped rod end caps against sampled/);
  assert.match(gpuContactUnsupported(topology, [{kind: 'plane', normal: [0, 1, 0], offset_m: 0}])!, /clipped rod caps are not on the GPU/);
});

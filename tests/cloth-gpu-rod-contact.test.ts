import test from 'node:test';
import assert from 'node:assert/strict';
import {GPUClothNewton} from '../lib/cloth/gpu-newton';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {createClothState, prepareClothTopology, type PreparedClothTopology} from '../lib/cloth/topology';
import {solveClothIncrement} from '../lib/cloth/nonlinear-solve';
import {ClothContactSet, clothContactWarmStart} from '../lib/cloth/contact-constraints';
import {closestSegmentPoints, sampleClothCollider, type ClothCollider} from '../lib/cloth/contact';
import {sampleClippedRodCollider} from '../lib/cloth/clipped-contact';
import type {ClothStepOptions} from '../lib/cloth/dynamics';
import type {ClothMaterial, ClothState, ClothTopologyData, ClothVector} from '../lib/cloth/types';
import {withUniformDevice} from './helpers/uniform-geometric';

/** Rod contact on the GPU (lib/cloth/gpu-contact.ts, lib/cloth/gpu-newton.ts)
 * against the Float64 oracle (ClothContactSet with rodContactOracle): the same
 * pairs (rod-vertex spheres, segment interiors, rolling endpoints), normals and
 * weights at each step's first detection, the same positions, twist and
 * multipliers after it, and the oracle tests' physics on the GPU result. */

const gpu = {skip: !process.env.WEBGPU_NODE_MODULE};
const point = (x: ArrayLike<number>, i: number): ClothVector => [x[3*i]!, x[3*i + 1]!, x[3*i + 2]!];
const maximumDifference = (a: ArrayLike<number>, b: ArrayLike<number>) => {let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k]! - b[k]!)); return m;};
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
function predicted(topology: PreparedClothTopology, state: ClothState, dt: number, options: ClothStepOptions): Float64Array {
  const x = state.positions_m.slice(), g = options.gravity_m_s2 ?? [0, -9.81, 0], target = options.supportPositions_m ?? topology.data.restPositions_m;
  for (let i = 0; i < topology.vertexCount; i++) {
    const mass = state.dryMass_kg[i]! + state.waterVolume_m3[i]!*topology.materials[topology.vertexMaterial[i]!]!.waterDensity_kg_m3;
    for (let a = 0; a < 3; a++) {const k = 3*i + a; x[k] = (topology.kinematicFixed[i] ?? 0) & 1 ? target[k]! : x[k]! + dt*state.velocities_m_s[k]! + dt*dt*(g[a]! + (options.externalForces_N?.[k] ?? 0)/mass);}
  }
  return x;
}

/** Steps the oracle (a) and the GPU (b) side by side, b reset to a before each step. */
async function rodParity(topology: PreparedClothTopology, a: ClothState, steps: number, options: ClothStepOptions, each?: (step: number, b: ClothState) => void, dt = .001): Promise<{kinds: Set<string>; active: number; positions: number; twist: number}> {
  const kinds = new Set<string>();
  let active = 0, positions = 0, twist = 0;
  await withUniformDevice('cloth GPU rod contact', async device => {
    const b = createClothState(topology), solver = new GPUClothNewton(device, topology);
    try {
      await solver.initialize();
      for (let step = 0; step < steps; step++) {
        for (const [key, value] of Object.entries(a)) if (value instanceof Float64Array) (b as unknown as Record<string, Float64Array>)[key]!.set(value);
        b.time_s = a.time_s;
        const reference = new ClothContactSet(topology, b.positions_m.slice(), dt, options.colliders ?? [], options.selfCollision ?? false, options.tolerance_m ?? 1e-8, undefined, {current: b.twist_rad, old: b.twist_rad});
        reference.detect(predicted(topology, b, dt, options));
        const first = await solver.detectContacts(b, dt, options);
        assert.equal(first.errors, 0, `step ${step}: detection error bits ${first.errors}`);
        const cpuKeys = reference.constraints.map(c => c.key).sort(), gpuKeys = first.contacts.map(c => c.key).sort();
        assert.deepEqual(gpuKeys, cpuKeys, `step ${step}: first detection differs`);
        const byKey = new Map(first.contacts.map(c => [c.key, c]));
        for (const c of reference.constraints) {
          const g = byKey.get(c.key)!;
          kinds.add(c.key.split(':')[0]!);
          assert.deepEqual(g.vertices, c.vertices, `step ${step} ${c.key}: vertices (rolling endpoints included)`);
          for (let k = 0; k < 3; k++) assert.ok(Math.abs(g.normal[k]! - c.normal[k]!) < 1e-4, `step ${step} ${c.key}: normal ${g.normal} vs ${c.normal}`);
          c.weights.forEach((w, j) => assert.ok(Math.abs(g.weights[j]! - w) < 1e-4, `step ${step} ${c.key}: weights ${g.weights} vs ${c.weights}`));
        }
        const cpu = solveClothIncrement(topology, a, dt, {...options, rodContactOracle: true}), report = await solver.step(b, dt, options);
        assert.ok(cpu.converged && report.converged, `step ${step}: ${report.maximumUpdate_m} m`);
        assert.equal(cpu.contactModel, 'in-solve');
        assert.equal(report.contactModel, 'in-solve');
        assert.equal(report.contactDetection, 'gpu', `step ${step}: ${report.contactFallback}`);
        const dp = maximumDifference(a.positions_m, b.positions_m), dtw = maximumDifference(a.twist_rad, b.twist_rad);
        positions = Math.max(positions, dp); twist = Math.max(twist, dtw);
        assert.ok(dp < 1e-6, `step ${step}: positions differ by ${dp} m`);
        assert.ok(dtw < 1e-4, `step ${step}: twist differs by ${dtw} rad`);
        const cpuLambda = clothContactWarmStart(a) ?? new Map<string, number>(), gpuContacts = await solver.contacts(), largest = Math.max(0, ...cpuLambda.values());
        const gpuLambda = new Map(gpuContacts.filter(c => c.lambda > 0).map(c => [c.key, c.lambda])), rho = new Map(gpuContacts.map(c => [c.key, c.rho]));
        assert.deepEqual([...gpuLambda.keys()].sort(), [...cpuLambda.keys()].sort(), `step ${step}: the pairs carrying force differ`);
        for (const [key, lambda] of cpuLambda) {
          const error = Math.abs(gpuLambda.get(key)! - lambda), bound = Math.max(1e-3*largest, rho.get(key)!*(options.tolerance_m ?? 1e-8));
          assert.ok(error < bound, `step ${step} ${key}: multiplier ${gpuLambda.get(key)} vs ${lambda} (rho ${rho.get(key)})`);
        }
        active = Math.max(active, cpuLambda.size);
        each?.(step, b);
      }
    } finally {solver.destroy();}
  });
  return {kinds, active, positions, twist};
}

test('GPU rod contact: a round rod lands on a plane and rests on its radius, as the oracle', gpu, async () => {
  const r = .002, topology = prepareClothTopology(rods(6, .01, {origin: [0, r + .001, 0], direction: [1, 0, 0]}), [material({})]), a = createClothState(topology);
  for (let i = 0; i < topology.vertexCount; i++) a.velocities_m_s[3*i + 1] = -1;
  const result = await rodParity(topology, a, 30, {iterations: 64, tolerance_m: 1e-8, colliders: [{kind: 'plane', normal: [0, 1, 0], offset_m: 0}]}, (step, b) => {
    for (let i = 0; i < topology.vertexCount; i++) assert.ok(b.positions_m[3*i + 1]! > r - 1e-6, `step ${step}: vertex ${i} at y = ${b.positions_m[3*i + 1]}`);
  });
  assert.ok(result.kinds.has('c') && result.active > 0, `pairs ${[...result.kinds]}`);
});

test('GPU rod contact: a long segment over a sphere is held by its interior (rc), as the oracle', gpu, async () => {
  const r = .002, R = .05, topology = prepareClothTopology(rods(2, .2, {origin: [-.1, R + r + .0005, 0], direction: [1, 0, 0]}), [material({})]), a = createClothState(topology);
  const sphere: ClothCollider = {kind: 'sphere', center_m: [0, 0, 0], radius_m: R};
  const result = await rodParity(topology, a, 40, {iterations: 64, tolerance_m: 1e-8, colliders: [sphere]}, (step, b) => {
    const gap = sampleClippedRodCollider({a: point(b.positions_m, 0), b: point(b.positions_m, 1), radius_m: r, planes: []}, sphere, 0, sampleClothCollider).distance_m;
    assert.ok(gap > -1e-6, `step ${step}: capsule penetrates the sphere by ${-gap} m`);
  });
  assert.ok(result.kinds.has('rc'), `pairs ${[...result.kinds]}`);
});

test('GPU rod contact: a spinning rod on a frictional floor rolls (endpoint couple and twist in the slip), as the oracle', gpu, async () => {
  const r = .002, omega = 50, topology = prepareClothTopology(rods(6, .01, {origin: [0, r, 0], direction: [1, 0, 0]}), [material({})]), a = createClothState(topology);
  a.angularVelocity_rad_s.fill(omega);
  let vz = 0, spin = 0;
  const result = await rodParity(topology, a, 30, {iterations: 64, tolerance_m: 1e-8, colliders: [{kind: 'plane', normal: [0, 1, 0], offset_m: 0, friction: .8}]}, (_, b) => {
    let m = 0, p = 0; for (let i = 0; i < topology.vertexCount; i++) {m += b.dryMass_kg[i]!; p += b.dryMass_kg[i]!*b.velocities_m_s[3*i + 2]!;} vz = p/m;
    spin = b.angularVelocity_rad_s.reduce((s, w) => s + w, 0)/topology.segmentCount;
  });
  // Rolling without slip (the oracle test's bounds): v = r omega toward t x n = +z, the solid section at omega/3.
  assert.ok(vz > 0, `rolls toward +z (v = ${vz} m/s)`);
  assert.ok(Math.abs(vz - r*spin) < .05*r*spin, `material point sticks: v = ${vz}, r omega = ${r*spin}`);
  assert.ok(Math.abs(spin - omega/3) < .05*omega/3, `spin ${spin} rad/s, rolling cylinder ${omega/3}`);
  assert.ok(result.kinds.has('c') && result.twist < 1e-4);
});

test('GPU rod contact: a rod falling across a fixed rod is stopped by the segment-segment pair (rs), as the oracle', gpu, async () => {
  const r = .002, h = .025, topology = prepareClothTopology(rods(5, h,
    {origin: [-.05, 0, .0125], direction: [1, 0, 0], fixed: 3},
    {origin: [.0125, 2*r + .002, -.05], direction: [0, 0, 1]}), [material({})]), a = createClothState(topology);
  for (let i = 5; i < 10; i++) a.velocities_m_s[3*i + 1] = -1;
  const result = await rodParity(topology, a, 15, {iterations: 64, tolerance_m: 1e-8, selfCollision: true}, (step, b) => {
    for (let s = 0; s < 4; s++) for (let u = 0; u < 4; u++) {
      const q = closestSegmentPoints(point(b.positions_m, s), point(b.positions_m, s + 1), point(b.positions_m, 5 + u), point(b.positions_m, 6 + u));
      assert.ok(Math.hypot(q.a[0] - q.b[0], q.a[1] - q.b[1], q.a[2] - q.b[2]) > 2*r - 1e-6, `step ${step}: segments ${s}, ${u} overlap`);
    }
  });
  assert.ok(result.kinds.has('rs'), `pairs ${[...result.kinds]}`);
});

test('GPU rod contact: a rod finer than its radius has no self contact with its own rest neighbourhood', gpu, async () => {
  const topology = prepareClothTopology(rods(12, .001, {origin: [0, .1, 0], direction: [1, 0, 0]}), [material({})]), a = createClothState(topology);
  const result = await rodParity(topology, a, 2, {iterations: 64, tolerance_m: 1e-8, selfCollision: true});
  assert.equal(result.kinds.size, 0);
});

/** A 6 x 6 sheet (2 cm spacing, all pinned) in y = 0 and a 9-vertex rod across it at an angle, falling at `speed`. */
function rodOverSheet(friction: number, speed: number): {topology: PreparedClothTopology; a: ClothState; check: (step: number, b: ClothState) => void} {
  const n = 6, hs = .02, restPositions_m: number[] = [], triangles: number[] = [], fixed: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {restPositions_m.push(i*hs, 0, j*hs); fixed.push(1);}
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {const q = j*n + i; triangles.push(q, q + n, q + 1, q + 1, q + n, q + n + 1);}
  const r = .002, rodStart = n*n, count = 9;
  for (let k = 0; k < count; k++) {restPositions_m.push(.013 + .0095*k, r + .0035, .017 + .0071*k); fixed.push(0);}
  const data: ClothTopologyData = {restPositions_m, triangles, triangleMaterials: Array(triangles.length/3).fill(0), rodOffsets: [0, count], rodIndices: Array.from({length: count}, (_, k) => rodStart + k), rodMaterials: [1], fixed};
  const topology = prepareClothTopology(data, [{...DEFAULT_CLOTH_MATERIAL, thickness_m: .001, friction}, material({friction})]), a = createClothState(topology);
  for (let k = 0; k < count; k++) a.velocities_m_s[3*(rodStart + k) + 1] = -speed;
  const check = (step: number, b: ClothState) => {for (let k = 0; k < count; k++) assert.ok(b.positions_m[3*(rodStart + k) + 1]! > r + .0005 - 1e-6, `step ${step}: rod vertex ${k} at y = ${b.positions_m[3*(rodStart + k) + 1]}`);};
  return {topology, a, check};
}

test('GPU rod contact: a rod falling onto a pinned sheet is stopped by its vertices against faces (rv) and its segments against edges (rx), as the oracle', gpu, async () => {
  const {topology, a, check} = rodOverSheet(0, 2);
  const result = await rodParity(topology, a, 12, {iterations: 64, tolerance_m: 1e-8, selfCollision: true}, check);
  for (const kind of ['rv', 'rx']) assert.ok(result.kinds.has(kind), `pairs ${[...result.kinds]}`);
});

test('GPU rod contact: the same impact with rolling friction, as the oracle', gpu, async () => {
  // Friction rolls the rod: its twist and positions are one CG system (as
  // separate block steps this impact converged linearly and capped Newton).
  const {topology, a, check} = rodOverSheet(Math.tan(10*Math.PI/180), 2);
  const result = await rodParity(topology, a, 12, {iterations: 64, tolerance_m: 1e-8, selfCollision: true}, check);
  for (const kind of ['rv', 'rx']) assert.ok(result.kinds.has(kind), `pairs ${[...result.kinds]}`);
});

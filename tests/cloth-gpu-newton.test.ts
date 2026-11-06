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
import {ClothContactSet, clothContactWarmStart} from '../lib/cloth/contact-constraints';
import type {ClothCollider} from '../lib/cloth/contact';

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

test('GPU in-solve contact matches the Float64 solve: a sheet draped on a sphere', gpu, async () => {
  await withUniformDevice('cloth GPU Newton sphere contact', async device => {
    const topology = prepareClothTopology(sheet(7, .01), [DEFAULT_CLOTH_MATERIAL]), a = createClothState(topology), b = createClothState(topology), solver = new GPUClothNewton(device, topology);
    const sphere = {kind: 'sphere' as const, center_m: [.03, .978, .03] as const, radius_m: .02}, radius = .5*DEFAULT_CLOTH_MATERIAL.thickness_m;
    try {
      await solver.initialize();
      let contacts = 0;
      for (let step = 0; step < 50; step++) {
        const options = {iterations: 64, tolerance_m: 1e-8, colliders: [sphere]}, old = b.positions_m.slice();
        const cpu = solveClothIncrement(topology, a, .001, options), g = await solver.step(b, .001, options);
        assert.ok(cpu.converged && g.converged, `step ${step}: ${g.maximumUpdate_m} m`);
        assert.equal(g.contactModel, 'in-solve');
        assert.equal(g.refinementIterations, 0, `step ${step} needed the CPU refinement`);
        contacts = Math.max(contacts, g.contacts.contacts);
        assert.ok(maximumDifference(a.positions_m, b.positions_m) < 1e-6, `step ${step}: positions differ by ${maximumDifference(a.positions_m, b.positions_m)} m`);
        for (let i = 0; i < topology.vertexCount; i++) {
          assert.ok(Math.hypot(...[0, 1, 2].map(k => b.positions_m[3*i + k]! - sphere.center_m[k]!)) - sphere.radius_m >= radius - 1e-7, `step ${step}: vertex ${i} penetrates`);
          for (let k = 0; k < 3; k++) assert.ok(Math.abs(b.velocities_m_s[3*i + k]! - (b.positions_m[3*i + k]! - old[3*i + k]!)/.001) < 1e-9, 'velocity is that of the final positions');
        }
      }
      assert.ok(contacts > 0, 'the sheet reached the sphere');
    } finally {solver.destroy();}
  });
});

test('GPU in-solve self contact matches the Float64 solve: a sheet falling on a fixed sheet', gpu, async () => {
  await withUniformDevice('cloth GPU Newton self contact', async device => {
    const h = .005, lower = sheet(5, h), upper = sheet(5, h, 1.002);
    const data: ClothTopologyData = {
      restPositions_m: [...lower.restPositions_m, ...upper.restPositions_m.map((v, k) => v + (k % 3 === 0 ? .4*h : k % 3 === 2 ? .3*h : 0))],
      triangles: [...lower.triangles, ...upper.triangles.map(v => v + 25)], rodOffsets: [0], rodIndices: [], fixed: [...Array(25).fill(1), ...Array(25).fill(0)],
    };
    const topology = prepareClothTopology(data, [DEFAULT_CLOTH_MATERIAL]), a = createClothState(topology), b = createClothState(topology), solver = new GPUClothNewton(device, topology);
    try {
      await solver.initialize();
      let selfContacts = 0;
      for (let step = 0; step < 40; step++) {
        const options = {iterations: 64, tolerance_m: 1e-8, selfCollision: true};
        const cpu = solveClothIncrement(topology, a, .001, options), g = await solver.step(b, .001, options);
        assert.ok(cpu.converged && g.converged, `step ${step}: ${g.maximumUpdate_m} m`);
        assert.equal(g.contactModel, 'in-solve');
        assert.equal(g.refinementIterations, 0, `step ${step} needed the CPU refinement`);
        selfContacts = Math.max(selfContacts, g.contacts.selfContacts);
        assert.ok(maximumDifference(a.positions_m, b.positions_m) < 1e-6, `step ${step}: positions differ by ${maximumDifference(a.positions_m, b.positions_m)} m`);
        for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) assert.ok(b.positions_m[3*(25 + 5*j + i) + 1]! > 1 + DEFAULT_CLOTH_MATERIAL.thickness_m - 1e-7, `step ${step}: the upper sheet crossed`);
      }
      assert.ok(selfContacts > 0, 'the sheets met');
    } finally {solver.destroy();}
  });
});

/** The end-of-step positions every first detection uses: the prediction, fixed vertices at their targets. */
function predicted(topology: PreparedClothTopology, state: ClothState, dt: number, options: ClothStepOptions): Float64Array {
  const x = state.positions_m.slice(), g = options.gravity_m_s2 ?? [0, -9.81, 0], target = options.supportPositions_m ?? topology.data.restPositions_m;
  for (let i = 0; i < topology.vertexCount; i++) {
    const mass = state.dryMass_kg[i]! + state.waterVolume_m3[i]!*topology.materials[topology.vertexMaterial[i]!]!.waterDensity_kg_m3;
    for (let a = 0; a < 3; a++) {const k = 3*i + a; x[k] = (topology.kinematicFixed[i] ?? 0) & 1 ? target[k]! : x[k]! + dt*state.velocities_m_s[k]! + dt*dt*(g[a]! + (options.externalForces_N?.[k] ?? 0)/mass);}
  }
  return x;
}
/** Steps the Float64 solve (a) and the GPU solve with GPU detection (b) side by side. Every step:
 * the GPU's first detection is the Float64 ClothContactSet's, key for key, with the same normals
 * and weights; after the step the pairs carrying force and their multipliers match the Float64 ones. */
async function contactParity(device: GPUDevice, topology: PreparedClothTopology, a: ClothState, b: ClothState, steps: number, options: ClothStepOptions, dt = .001): Promise<{detected: number; active: number; lambda: number; ofBound: number}> {
  const solver = new GPUClothNewton(device, topology);
  let detected = 0, active = 0, lambdaError = 0, ofBound = 0;
  try {
    await solver.initialize();
    for (let step = 0; step < steps; step++) {
      // Each step starts both solves from the Float64 state: the Newton tolerance bounds the
      // scaled residual, not the position error, so free trajectories drift apart by more than
      // the contact parity being measured (1.2e-6 m by step 3 of the fold with contact off).
      for (const [key, value] of Object.entries(a)) if (value instanceof Float64Array) (b as unknown as Record<string, Float64Array>)[key]!.set(value);
      b.time_s = a.time_s;
      const reference = new ClothContactSet(topology, b.positions_m.slice(), dt, options.colliders ?? [], options.selfCollision ?? false, options.tolerance_m ?? 1e-8);
      reference.detect(predicted(topology, b, dt, options));
      const first = await solver.detectContacts(b, dt, options);
      assert.equal(first.errors, 0, `step ${step}: detection error bits ${first.errors}`);
      const cpuKeys = reference.constraints.map(c => c.key).sort(), gpuKeys = first.contacts.map(c => c.key).sort();
      assert.deepEqual(gpuKeys, cpuKeys, `step ${step}: first detection differs`);
      const byKey = new Map(first.contacts.map(c => [c.key, c]));
      for (const c of reference.constraints) {
        const g = byKey.get(c.key)!;
        for (let k = 0; k < 3; k++) assert.ok(Math.abs(g.normal[k]! - c.normal[k]!) < 1e-4, `step ${step} ${c.key}: normal ${g.normal} vs ${c.normal}`);
        c.weights.forEach((w, j) => assert.ok(Math.abs(g.weights[j]! - w) < 1e-4, `step ${step} ${c.key}: weights ${g.weights} vs ${c.weights}`));
      }
      detected = Math.max(detected, cpuKeys.length);
      const cpu = solveClothIncrement(topology, a, dt, options), report = await solver.step(b, dt, options);
      assert.ok(cpu.converged && report.converged, `step ${step}: ${report.maximumUpdate_m} m`);
      assert.equal(report.contactDetection, 'gpu', `step ${step}: ${report.contactFallback}`);
      assert.equal(report.refinementIterations, 0, `step ${step} needed the CPU refinement`);
      assert.ok(maximumDifference(a.positions_m, b.positions_m) < 1e-6, `step ${step}: positions differ by ${maximumDifference(a.positions_m, b.positions_m)} m`);
      const cpuLambda = clothContactWarmStart(a) ?? new Map<string, number>(), gpuContacts = await solver.contacts(), largest = Math.max(0, ...cpuLambda.values());
      const gpuLambda = new Map(gpuContacts.filter(c => c.lambda > 0).map(c => [c.key, c.lambda])), rho = new Map(gpuContacts.map(c => [c.key, c.rho]));
      assert.deepEqual([...gpuLambda.keys()].sort(), [...cpuLambda.keys()].sort(), `step ${step}: the pairs carrying force differ`);
      // Both solves stop once |min(g, lambda/rho)| < tolerance, which fixes lambda/rho, and so
      // lambda to rho*tolerance; within that, the multipliers agree to 1e-3 of the largest.
      for (const [key, lambda] of cpuLambda) {
        const error = Math.abs(gpuLambda.get(key)! - lambda), bound = Math.max(1e-3*largest, rho.get(key)!*(options.tolerance_m ?? 1e-8));
        lambdaError = Math.max(lambdaError, error/largest); ofBound = Math.max(ofBound, error/bound);
        assert.ok(error < bound, `step ${step} ${key}: multiplier ${gpuLambda.get(key)} vs ${lambda} (rho ${rho.get(key)})`);
      }
      assert.equal(report.contacts.contacts + report.contacts.selfContacts, cpuLambda.size, `step ${step}: reported contacts`);
      active = Math.max(active, cpuLambda.size);
    }
  } finally {solver.destroy();}
  return {detected, active, lambda: lambdaError, ofBound};
}

test('GPU contact detection is the Float64 contact set: sphere drape', gpu, async () => {
  await withUniformDevice('cloth GPU contact parity sphere', async device => {
    const topology = prepareClothTopology(sheet(7, .01), [DEFAULT_CLOTH_MATERIAL]), a = createClothState(topology), b = createClothState(topology);
    const sphere: ClothCollider = {kind: 'sphere', center_m: [.03, .978, .03], radius_m: .02};
    const result = await contactParity(device, topology, a, b, 40, {iterations: 64, tolerance_m: 1e-8, colliders: [sphere]});
    assert.ok(result.active > 0, 'the sheet reached the sphere');
    console.log(`sphere drape parity: ${result.detected} detected, ${result.active} active, multipliers within ${result.lambda.toExponential(2)} of the largest`);
  });
});

/** A flat strip folded in its initial state around a crease of radius R (isometric),
 * the lower layer fixed and the upper layer thrown down onto it. */
function foldedStrip(): {topology: PreparedClothTopology; a: ClothState; b: ClothState; supportPositions_m: Float64Array} {
  const nx = 24, nz = 4, h = .002, R = .003, crease = 8*h, restPositions_m: number[] = [], triangles: number[] = [], fixed: number[] = [], folded: number[] = [];
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const x = i*h, z = j*h, s = x - crease;
    restPositions_m.push(x, 1, z); fixed.push(i <= 8 ? 1 : 0);
    if (s <= 0) folded.push(x, 1, z); else if (s <= Math.PI*R) folded.push(crease + R*Math.sin(s/R), 1 + R*(1 - Math.cos(s/R)), z); else folded.push(crease - (s - Math.PI*R), 1 + 2*R, z);
  }
  for (let j = 0; j + 1 < nz; j++) for (let i = 0; i + 1 < nx; i++) {const p = j*nx + i, q = p + 1, r = p + nx, t = r + 1; triangles.push(p, r, q, q, r, t);}
  const data: ClothTopologyData = {restPositions_m, triangles, rodOffsets: [0], rodIndices: [], fixed};
  const topology = prepareClothTopology(data, [DEFAULT_CLOTH_MATERIAL]), a = createClothState(topology), b = createClothState(topology);
  for (const state of [a, b]) {
    state.positions_m.set(folded);
    for (let i = 0; i < topology.vertexCount; i++) if (i % nx > 8 + Math.PI*R/h) state.velocities_m_s[3*i + 1] = -.3;
  }
  return {topology, a, b, supportPositions_m: Float64Array.from(folded)};
}

test('GPU contact detection is the Float64 contact set: a sheet folded onto itself', gpu, async () => {
  await withUniformDevice('cloth GPU contact parity fold', async device => {
    const {topology, a, b, supportPositions_m} = foldedStrip();
    const result = await contactParity(device, topology, a, b, 30, {iterations: 64, tolerance_m: 1e-8, selfCollision: true, supportPositions_m});
    assert.ok(result.active > 0, 'the layers met');
    console.log(`self fold parity: ${result.detected} detected, ${result.active} active, multipliers within ${result.lambda.toExponential(2)} of the largest, ${result.ofBound.toFixed(2)} of the rho*tolerance bound`);
  });
});

// Open (todo): the trajectories agree to 6e-10 m through step 12, then part by 1.0e-5 m at the
// first contact (step 13). The GPU path stops at the Float32 floor and the Float64 refinement
// restarts its contact set from the GPU iterate; self contacts keep the normal and weights of
// their detection (ClothContactSet.relinearize updates collider contacts only), so the converged
// state depends on where the pair was detected. The Float64 solve from the prediction ends with
// e:93:228 at a fresh gap of -1.8e-6 m; from the GPU iterate, v:0:89 at -1.6e-7 m.
test('GPU contact on the folded sheet, free-running at a 1e-12 m tolerance, stays within 1e-8 m of the Float64 trajectory', {...gpu, todo: 'self contacts are not relinearized: the converged state depends on the detection point'}, async () => {
  await withUniformDevice('cloth GPU contact fold free run', async device => {
    // No resync: each solve starts from its own previous state, so drift cannot hide. At a
    // 1e-10 m tolerance the contact-free trajectories already part by 1.06e-8 m at step 7 (the
    // tolerance bounds the scaled residual, not the position error), hence 1e-12 m. Float32
    // alone stops near 1e-9 m; the Float64 refinement finishes each step.
    const {topology, a, b, supportPositions_m} = foldedStrip(), options: ClothStepOptions = {iterations: 64, tolerance_m: 1e-12, selfCollision: true, supportPositions_m};
    const solver = new GPUClothNewton(device, topology);
    let worst = 0, active = 0;
    try {
      await solver.initialize();
      for (let step = 0; step < 30; step++) {
        const cpu = solveClothIncrement(topology, a, .001, options), report = await solver.step(b, .001, options);
        assert.ok(cpu.converged && report.converged, `step ${step}: ${report.maximumUpdate_m} m`);
        assert.equal(report.contactDetection, 'gpu', `step ${step}: ${report.contactFallback}`);
        const difference = maximumDifference(a.positions_m, b.positions_m); worst = Math.max(worst, difference);
        assert.ok(difference < 1e-8, `step ${step}: positions differ by ${difference} m`);
        const cpuKeys = [...(clothContactWarmStart(a) ?? new Map()).keys()].sort(), gpuKeys = (await solver.contacts()).filter(c => c.lambda > 0).map(c => c.key).sort();
        assert.deepEqual(gpuKeys, cpuKeys, `step ${step}: the pairs carrying force differ`);
        active = Math.max(active, cpuKeys.length);
      }
    } finally {solver.destroy();}
    assert.ok(active > 0, 'the layers met');
    console.log(`self fold free run: ${active} active, positions within ${worst.toExponential(2)} m over 30 steps`);
  });
});

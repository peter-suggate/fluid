import test from 'node:test';
import assert from 'node:assert/strict';
import { withUniformDevice } from './helpers/uniform-geometric';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import { createClothState, prepareClothTopology } from '../lib/cloth/topology';
import type { ClothMaterial, ClothPhysicsOptions, ClothTopologyData } from '../lib/cloth/types';
import { advanceClothPoreTransport } from '../lib/cloth/porous-flow';
import { ClothGPUPoreTransport, clothPoreQuantum, clothPoreTopology } from '../lib/cloth/gpu-pore-transport';

function lcg(seed: number) { let s = seed >>> 0; return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32; }

// A deformed two-material sheet beside free rods: wet, damp and dry
// controls (suction fronts and donor limiting), fixed vertices, oblique
// gravity-frame and pressure accelerations.
function fixture(seed: number) {
  const random = lcg(seed), n = 24, positions: number[] = [], triangles: number[] = [], triangleMaterials: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) positions.push(.4 + .01 * i + .2 * .01 * j, .9 + .004 * i, .3 + .01 * j);
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {
    const a = i + n * j, m = (i + j) % 3 === 0 ? 1 : 0;
    triangles.push(a, a + 1, a + n + 1, a, a + n + 1, a + n); triangleMaterials.push(m, m);
  }
  const rodOffsets = [0], rodIndices: number[] = [];
  for (let rod = 0; rod < 2; rod++) {
    for (let k = 0; k < 30; k++) { rodIndices.push(positions.length / 3); positions.push(.4 + .008 * k, 1.05 + .02 * rod, .3 + .003 * k * (rod + 1)); }
    rodOffsets.push(rodIndices.length);
  }
  const V = positions.length / 3, fixed = Array.from({ length: V }, (_, v) => v % 89 === 5 ? 1 : 0);
  const data: ClothTopologyData = { restPositions_m: positions, triangles, triangleMaterials, rodOffsets, rodIndices, rodMaterials: [0, 1], fixed };
  const materials: ClothMaterial[] = [DEFAULT_CLOTH_MATERIAL, { ...DEFAULT_CLOTH_MATERIAL, id: 'wool', thickness_m: .0007, rodRadius_m: .0003, rodRadiusB_m: .0002, fiberDiameter_m: .00021, contactAngle_rad: 1.1, waterViscosity_Pa_s: .0011 }];
  const state = createClothState(prepareClothTopology(data, materials));
  for (let v = 0; v < V; v++) {
    for (let a = 0; a < 3; a++) { state.positions_m[3 * v + a]! += .0008 * (random() - .5); state.velocities_m_s[3 * v + a] = .2 * (random() - .5); }
    const capacity = state.controlVolume_m3[v]! - state.solidVolume_m3[v]!, band = (v % n) / n;
    state.waterVolume_m3[v] = v % 7 === 0 ? 0 : capacity * (band < .3 ? .95 : band < .6 ? .3 * random() : .02 * random());
  }
  const pressure = Float64Array.from({ length: 3 * V }, () => 4 * (random() - .5)), solid = Float64Array.from({ length: 3 * V }, () => 2 * (random() - .5));
  return { data, materials, state, fixed, pressure, solid, V };
}

// f32 against the f64 oracle after every substep of one outer step. Each
// Krylov solve stops at a 16u normwise backward error and the Picard fixed
// point at 128u of the saturation scale; per substep the f32 edge
// coefficients carry a few ulps. Saturation and per-edge flux errors stay
// within 2^-12 of their scale over the step (a 1/4096 bound). Each substep
// also rounds every accepted edge flux to the water quantum (2^-26 of the
// largest rest control volume): at most half a quantum an edge a substep.
const BOUND = 2 ** -12;

for (const [label, physics, dt] of [
  ['source nonlinear drag, 36 substeps', { nonlinearDrag: true, manifoldSubsteps: 36, manifoldDragFloorReferenceVolume_m3: 1e-6 }, .01],
  ['linear drag, rest solid fraction, fixed solid, 8 substeps', { nonlinearDrag: false, varyingSolidFraction: false, solveSolid: false, manifoldSubsteps: 8, pressureOnManifold: false }, .01],
] as [string, ClothPhysicsOptions, number][]) for (const features of [['subgroups'], []] as GPUFeatureName[][]) {
  test(`GPU pore transport matches the f64 oracle: ${label}, ${features.length ? 'subgroup' : 'workgroup'} reductions`, { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
    await withUniformDevice('gpu pore transport', async device => {
      const f = fixture(7), s = f.state, gpu = new ClothGPUPoreTransport(device, clothPoreTopology(f.data, f.materials, f.V), s.restControlVolume_m3);
      await gpu.initialize();
      // Timing is profiling only (ClothGPUPoreTransportOptions.timing): an opt-in without the feature fails explicitly.
      if (!device.features.has('timestamp-query')) assert.throws(() => new ClothGPUPoreTransport(device, gpu.topology, s.restControlVolume_m3, { timing: true }), /timestamp-query/);
      if (features.length && device.features.has('subgroups')) assert.ok(gpu.subgroups, 'subgroup reductions on a 32-lane device');
      const q = clothPoreQuantum(s.restControlVolume_m3), gravity = [.3, -9.80665, .1] as const;
      // Both start from the same whole-quantum inventory.
      for (let v = 0; v < f.V; v++) s.waterVolume_m3[v] = Math.round(s.waterVolume_m3[v]! / q) * q;
      const initial = s.waterVolume_m3.reduce((a, b) => a + b, 0);
      const result = await gpu.advance({ positions_m: s.positions_m, velocities_m_s: s.velocities_m_s, controlVolume_m3: s.controlVolume_m3, solidVolume_m3: s.solidVolume_m3, solidFraction: s.solidFraction, dryMass_kg: s.dryMass_kg,
        waterVolume_m3: s.waterVolume_m3, fixed: f.fixed, materials: f.materials, physics, dt_s: dt, gravity_m_s2: gravity, pressureAcceleration_m_s2: f.pressure, solidAcceleration_m_s2: f.solid });
      assert.ok(result, 'wet cloth transports');
      assert.equal(result.gpu_ms, undefined, 'a production transport carries no timestamp writes');
      const fluxes = new Map<number, number>();
      const oracle = advanceClothPoreTransport({ state: s, data: f.data, materials: f.materials, physics, dt_s: dt, gravity_m_s2: gravity, pressureAcceleration_m_s2: f.pressure, solidAcceleration_m_s2: f.solid, edgeFlux_m3: fluxes });
      // Exact conservation in whole quanta; never negative.
      let total = 0; for (let v = 0; v < f.V; v++) { const w = result.waterVolume_m3[v]!; assert.ok(w >= 0); total += w; }
      assert.equal(total / q, Math.round(initial / q), 'GPU pore water is conserved to the quantum');
      let saturationError = 0, wet = 0;
      for (let v = 0; v < f.V; v++) {
        const capacity = Math.max(0, s.controlVolume_m3[v]! - s.solidVolume_m3[v]!);
        if (capacity > 0) { saturationError = Math.max(saturationError, Math.abs(result.waterVolume_m3[v]! - s.waterVolume_m3[v]!) / capacity); wet = Math.max(wet, s.waterVolume_m3[v]! / capacity); }
      }
      const slots = gpu.topology.slots, gpuFlux = await gpu.readEdgeFluxes();
      let fluxScale = 0, fluxError = 0; for (const x of fluxes.values()) fluxScale = Math.max(fluxScale, Math.abs(x));
      for (let e = 0; e < gpu.topology.E; e++) fluxError = Math.max(fluxError, Math.abs(gpuFlux[e]! - (fluxes.get(slots[2 * e]! * f.V + slots[2 * e + 1]!) ?? 0)));
      let speed = 0, velocityError = 0;
      for (let k = 0; k < 3 * f.V; k++) { speed = Math.max(speed, Math.abs(s.velocities_m_s[k]!)); velocityError = Math.max(velocityError, Math.abs(result.velocities_m_s[k]! - s.velocities_m_s[k]!)); }
      console.log(`${label}: saturation ${saturationError.toExponential(2)} flux ${(fluxError / fluxScale).toExponential(2)} velocity ${(velocityError / speed).toExponential(2)} initial ${initial.toExponential(3)} quantum ${q.toExponential(2)} flux scale ${fluxScale.toExponential(2)} moved ${oracle.movedVolume_m3.toExponential(3)}/${result.movedVolume_m3.toExponential(3)} limited ${oracle.limitedVolume_m3.toExponential(2)}/${result.limitedVolume_m3.toExponential(2)} iterations ${oracle.iterations}/${result.iterations} picard ${result.picardIterations}`);
      // Nonlinear drag and the source regularizer slow the front: still ~1e3 quanta an edge.
      assert.ok(oracle.movedVolume_m3 > 1e-4 * initial, 'the fixture moves water');
      assert.ok(saturationError <= BOUND * Math.max(1, wet), `saturation ${saturationError}`);
      const substeps = physics.manifoldSubsteps ?? 1;
      assert.ok(fluxError <= BOUND * fluxScale + substeps * q / 2, `edge flux ${fluxError / fluxScale}`);
      assert.ok(velocityError <= BOUND * speed, `velocity ${velocityError / speed}`);
      gpu.destroy();
    }, features);
  });
}

// Round-off magnitudes: a transverse velocity near 1e-22 (a momentum
// right-hand side near 1e-25 on that axis, about 0.1 on the others) and a
// nearly dry cloth (saturations near 1e-5). The raw systems' BiCGSTAB inner
// products square below f32's range: rho flushed to zero at iteration 0
// (the free-fall-with-absorption breakdown). Solved at unit scale, each axis
// and the dry mass solve keep the unit system's backward error: errors stay
// at the 2^-12 bound of their own axis or saturation scale, plus the
// half-quantum flux rounding each substep.
for (const variant of ['transverse momentum at round-off scale', 'nearly dry cloth'] as const) for (const features of [['subgroups'], []] as GPUFeatureName[][]) {
  test(`GPU pore transport matches the f64 oracle at round-off magnitudes: ${variant}, ${features.length ? 'subgroup' : 'workgroup'} reductions`, { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
    await withUniformDevice('gpu pore transport round-off scale', async device => {
      const f = fixture(7), s = f.state, random = lcg(11), physics: ClothPhysicsOptions = { nonlinearDrag: false, manifoldSubsteps: 8 }, dt = .01, substeps = 8;
      const gpu = new ClothGPUPoreTransport(device, clothPoreTopology(f.data, f.materials, f.V), s.restControlVolume_m3);
      await gpu.initialize();
      const q = clothPoreQuantum(s.restControlVolume_m3), gravity = [0, -9.80665, 0] as const;
      for (let v = 0; v < f.V; v++) {
        if (variant === 'nearly dry cloth') s.waterVolume_m3[v]! *= 1e-5;
        else s.velocities_m_s[3 * v] = 1e-22 * (random() - .5);
        s.waterVolume_m3[v] = Math.round(s.waterVolume_m3[v]! / q) * q;
      }
      const initial = s.waterVolume_m3.reduce((a, b) => a + b, 0);
      assert.ok(initial > 1000 * q, 'the dry fixture still holds water to move');
      const result = await gpu.advance({ positions_m: s.positions_m, velocities_m_s: s.velocities_m_s, controlVolume_m3: s.controlVolume_m3, solidVolume_m3: s.solidVolume_m3, solidFraction: s.solidFraction, dryMass_kg: s.dryMass_kg,
        waterVolume_m3: s.waterVolume_m3, fixed: f.fixed, materials: f.materials, physics, dt_s: dt, gravity_m_s2: gravity, pressureAcceleration_m_s2: f.pressure, solidAcceleration_m_s2: f.solid });
      assert.ok(result, 'wet cloth transports');
      const fluxes = new Map<number, number>();
      const oracle = advanceClothPoreTransport({ state: s, data: f.data, materials: f.materials, physics, dt_s: dt, gravity_m_s2: gravity, pressureAcceleration_m_s2: f.pressure, solidAcceleration_m_s2: f.solid, edgeFlux_m3: fluxes });
      assert.ok(oracle.movedVolume_m3 > 0, 'the fixture moves water');
      const slots = gpu.topology.slots, degree = new Uint32Array(f.V);
      for (let e = 0; e < gpu.topology.E; e++) { degree[slots[2 * e]!]!++; degree[slots[2 * e + 1]!]!++; }
      let total = 0; for (let v = 0; v < f.V; v++) { assert.ok(result.waterVolume_m3[v]! >= 0); total += result.waterVolume_m3[v]!; }
      assert.equal(total / q, Math.round(initial / q), 'GPU pore water is conserved to the quantum');
      let wet = 0; for (let v = 0; v < f.V; v++) { const c = s.controlVolume_m3[v]! - s.solidVolume_m3[v]!; if (c > 0) wet = Math.max(wet, s.waterVolume_m3[v]! / c); }
      for (let v = 0; v < f.V; v++) {
        const c = Math.max(0, s.controlVolume_m3[v]! - s.solidVolume_m3[v]!);
        const error = Math.abs(result.waterVolume_m3[v]! - s.waterVolume_m3[v]!);
        assert.ok(error <= BOUND * wet * c + substeps * degree[v]! * q / 2, `vertex ${v}: water ${error} at saturation scale ${wet}`);
      }
      const gpuFlux = await gpu.readEdgeFluxes();
      let fluxScale = 0, fluxError = 0; for (const x of fluxes.values()) fluxScale = Math.max(fluxScale, Math.abs(x));
      for (let e = 0; e < gpu.topology.E; e++) fluxError = Math.max(fluxError, Math.abs(gpuFlux[e]! - (fluxes.get(slots[2 * e]! * f.V + slots[2 * e + 1]!) ?? 0)));
      assert.ok(fluxError <= BOUND * fluxScale + substeps * q / 2, `edge flux ${fluxError / fluxScale}`);
      const speed = [0, 0, 0], velocityError = [0, 0, 0];
      for (let k = 0; k < 3 * f.V; k++) { speed[k % 3] = Math.max(speed[k % 3]!, Math.abs(s.velocities_m_s[k]!)); velocityError[k % 3] = Math.max(velocityError[k % 3]!, Math.abs(result.velocities_m_s[k]! - s.velocities_m_s[k]!)); }
      console.log(`${variant}: wet ${wet.toExponential(2)} initial ${(initial / q).toFixed(0)} quanta, flux ${(fluxError / fluxScale).toExponential(2)}, axis speeds ${speed.map(x => x.toExponential(2))}, velocity errors ${velocityError.map((x, a) => (x / speed[a]!).toExponential(2))}`);
      for (let a = 0; a < 3; a++) assert.ok(velocityError[a]! <= BOUND * speed[a]!, `axis ${'xyz'[a]}: velocity ${velocityError[a]! / speed[a]!} of its own scale ${speed[a]}`);
      gpu.destroy();
    }, features);
  });
}

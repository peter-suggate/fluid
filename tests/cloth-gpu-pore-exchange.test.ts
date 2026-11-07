import test from 'node:test';
import assert from 'node:assert/strict';
import { createRigidCouplingOracleScene } from '../lib/core/scenes';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import { createClothState, prepareClothTopology } from '../lib/cloth/topology';
import type { ClothMaterial, ClothPhysicsOptions, ClothState, ClothTopologyData } from '../lib/cloth/types';
import { advanceClothPoreTransport } from '../lib/cloth/porous-flow';
import { ClothGPUPoreTransport, clothPoreQuantum, clothPoreTopology } from '../lib/cloth/gpu-pore-transport';
import type { UniformMixedFrame } from '../lib/methods/uniform/uniform-mixed-frame';
import { UniformClothWaterExchange } from '../lib/methods/uniform/uniform-cloth-water-exchange';
import { UniformClothGPUPoreExchange } from '../lib/methods/uniform/uniform-cloth-gpu-pore-exchange';
import { UniformGPUOrderingKernels } from '../lib/methods/uniform/uniform-gpu-radix-sort';
import { prepareClothPorousCapacity } from '../lib/methods/uniform/uniform-porous-capacity';
import { NB_VOLUME_COMPENSATION_FLOAT as compensation } from '../lib/methods/uniform/uniform-mixed-surface-volume';
import { createUniformSolver, withUniformDevice } from './helpers/uniform-geometric';
import { readMixedBuffer } from './helpers/uniform-mixed-native-fields';

function lcg(seed: number) { let s = seed >>> 0; return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32; }
const clone = (s: ClothState): ClothState => ({ ...s, positions_m: s.positions_m.slice(), velocities_m_s: s.velocities_m_s.slice(), waterVolume_m3: s.waterVolume_m3.slice() });

// A two-material sheet tilted through the free surface of a still tank
// (y = 0.4): submerged controls capture from full cells, the crossing row
// from partly filled ones, and over-full controls release; dry, damp and
// wet bands, fixed vertices, oblique accelerations.
function fixture(seed: number) {
  const random = lcg(seed), n = 13, positions: number[] = [], triangles: number[] = [], triangleMaterials: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const x = -.12 + .02 * i, z = -.12 + .02 * j; positions.push(x, .4 + .25 * x + .03 * z, z); }
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {
    const a = i + n * j, m = (i + j) % 3 === 0 ? 1 : 0;
    triangles.push(a, a + 1, a + n + 1, a, a + n + 1, a + n); triangleMaterials.push(m, m);
  }
  const V = positions.length / 3, fixed = Array.from({ length: V }, (_, v) => v % 41 === 3 ? 1 : 0);
  const data: ClothTopologyData = { restPositions_m: positions, triangles, triangleMaterials, rodOffsets: [0], rodIndices: [], fixed };
  const materials: ClothMaterial[] = [DEFAULT_CLOTH_MATERIAL, { ...DEFAULT_CLOTH_MATERIAL, id: 'wool', thickness_m: .0007, fiberDiameter_m: .00021, contactAngle_rad: 1.1, waterViscosity_Pa_s: .0011 }];
  const prepared = prepareClothTopology(data, materials), state = createClothState(prepared);
  for (let v = 0; v < V; v++) {
    for (let a = 0; a < 3; a++) { state.positions_m[3 * v + a]! += .002 * (random() - .5); state.velocities_m_s[3 * v + a] = .2 * (random() - .5); }
    const capacity = state.controlVolume_m3[v]! - state.solidVolume_m3[v]!, band = (v % n) / n;
    state.waterVolume_m3[v] = v % 5 === 0 ? 0 : capacity * (v % 11 === 1 ? 1.3 : band < .3 ? .95 : band < .6 ? .3 * random() : .02 * random());
  }
  const pressure = Float64Array.from({ length: 3 * V }, () => 4 * (random() - .5)), solid = Float64Array.from({ length: 3 * V }, () => 2 * (random() - .5));
  return { data, materials, prepared, state, fixed, pressure, solid, V };
}

// f32 against the f64 oracles. Capture and release move whole quanta
// (2^-26 of the largest rest control volume; at most one a vertex each);
// their f32 rates and fills carry a few ulps. The transport keeps its own
// 2^-12 saturation bound (tests/cloth-gpu-pore-transport.test.ts).
const BOUND = 2 ** -12;

for (const [label, physics] of [
  ['nonlinear drag, 12 substeps', { nonlinearDrag: true, manifoldSubsteps: 12, manifoldDragFloorReferenceVolume_m3: 1e-6 }],
  ['no transport, rest solid fraction', { manifoldTransport: false, varyingSolidFraction: false }],
] as [string, ClothPhysicsOptions][]) test(`GPU pore exchange matches the f64 oracles at a real NB-FLIP surface: ${label}`, { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('gpu pore exchange', async device => {
    const scene = createRigidCouplingOracleScene('rigid-hydrostatic'); scene.rigidBodies = [];
    scene.systems = { ...scene.systems, porous: true };
    scene.fluid.density_kg_m3 = 1000; scene.numerics.fixedDt_s = scene.numerics.maxDt_s = .001;
    scene.fluid.refinementRegions = [{ id: 'cloth-support', min_m: { x: -.3, y: .1, z: -.3 }, max_m: { x: .3, y: .65, z: .3 }, rule: 'minimum-cell-size', minimumCellSize_cells: 1, maximumCellSize_cells: 1 }];
    const solver = await createUniformSolver(device, scene), frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame, lattice = frame.ownership.capacity.lattice;
    const fields = frame.clothCouplingFields([lattice.origin_m.x, lattice.origin_m.y, lattice.origin_m.z], lattice.cellSize_m);
    const f = fixture(11), dt = .01, gravity = [0, -9.80665, 0] as const, identity = { generation: f.state.generation, topologyRevision: f.state.topologyRevision, tick: 1, time_s: dt };
    const cpu = new UniformClothWaterExchange(fields), transport = new ClothGPUPoreTransport(device, clothPoreTopology(f.data, f.materials, f.V), f.state.restControlVolume_m3);
    const q = clothPoreQuantum(f.state.restControlVolume_m3);
    for (let v = 0; v < f.V; v++) f.state.waterVolume_m3[v] = Math.round(f.state.waterVolume_m3[v]! / q) * q;
    let gpu: UniformClothGPUPoreExchange | undefined;
    try {
      await cpu.initialize(); await cpu.initializeCapacity({ state: f.state }); await frame.initializePhysicalWaterBudget();
      assert.ok(solver.advanceTo(.001, [])); await solver.awaitFrameCompletion(); assert.equal(solver.info.simulationPipelineError, undefined);
      await transport.initialize(); gpu = new UniformClothGPUPoreExchange(fields, transport, await UniformGPUOrderingKernels.create(device)); await gpu.initialize();
      // The oracle chain of finishClothStep, on a copy.
      const o = clone(f.state), step = { state: o, data: f.data, kinematicFixed: f.fixed, materials: f.materials, physics, identity, dt_s: dt };
      const sample = await cpu.sampleSupport(o.positions_m), beforeCapture = o.velocities_m_s.slice();
      const captured = cpu.captureFromSample(step, sample, f.prepared.vertexMaterial).reduce((a, b) => a + b, 0);
      const solidAcceleration = Float64Array.from(f.solid, (a, k) => a + (o.velocities_m_s[k]! - beforeCapture[k]!) / dt);
      advanceClothPoreTransport({ ...step, gravity_m_s2: gravity, pressureAcceleration_m_s2: f.pressure, solidAcceleration_m_s2: solidAcceleration });
      const released = cpu.releaseFromSample(step, sample).reduce((a, b) => a + b, 0), table = prepareClothPorousCapacity(fields, { state: o });
      assert.ok(captured > 1e-3 * f.state.waterVolume_m3.reduce((a, b) => a + b, 0) && released > 0, `the fixture captures (${captured}) and releases (${released})`);
      const s = f.state, budgetBefore = await readMixedBuffer(device, fields.waterBudget);
      const result = await gpu.step({ positions_m: s.positions_m, velocities_m_s: s.velocities_m_s, controlVolume_m3: s.controlVolume_m3, solidVolume_m3: s.solidVolume_m3, solidFraction: s.solidFraction,
        dryMass_kg: s.dryMass_kg, waterVolume_m3: s.waterVolume_m3, fixed: f.fixed, materials: f.materials, physics, dt_s: dt, gravity_m_s2: gravity,
        pressureAcceleration_m_s2: f.pressure, solidAcceleration_m_s2: f.solid, identity, vertexMaterial: f.prepared.vertexMaterial });
      const budgetAfter = await readMixedBuffer(device, fields.waterBudget), cellVolume = fields.cellSize_m.reduce((a, b) => a * b, 1);
      // Exact inventory: bound water changes by captured minus released whole quanta.
      const total = (w: ArrayLike<number>) => Array.from(w).reduce((a, b) => a + b, 0), before = total(s.waterVolume_m3), after = total(result.waterVolume_m3);
      assert.ok(Math.abs(after - before - (result.captured_m3 - result.released_m3)) <= 1e-12 * before, 'GPU exchange conserves the bound inventory exactly');
      assert.equal(result.ledger.freeDelta_m3, result.released_m3 - result.captured_m3); assert.equal(result.ledger.boundDelta_m3, -result.ledger.freeDelta_m3);
      const committed = ((budgetAfter[0]! + budgetAfter[compensation]!) - (budgetBefore[0]! + budgetBefore[compensation]!)) * cellVolume;
      assert.ok(Math.abs(committed - result.ledger.freeDelta_m3) <= 1e-6 * Math.abs(result.ledger.freeDelta_m3) + 1e-12, `paired NB budget commit ${committed} versus ${result.ledger.freeDelta_m3}: before ${budgetBefore[0]}+${budgetBefore[compensation]}, after ${budgetAfter[0]}+${budgetAfter[compensation]}, cell ${cellVolume}`);
      const capturedError = Math.abs(result.captured_m3 - captured), releasedError = Math.abs(result.released_m3 - released);
      assert.ok(capturedError <= 1e-4 * captured + f.V * q, `captured ${result.captured_m3} versus ${captured}`);
      assert.ok(releasedError <= 1e-4 * released + f.V * q + BOUND * total(s.controlVolume_m3) * (physics.manifoldTransport === false ? 0 : 1), `released ${result.released_m3} versus ${released}`);
      let water = 0, speed = 0, velocity = 0;
      for (let v = 0; v < f.V; v++) {
        const capacity = s.controlVolume_m3[v]! - s.solidVolume_m3[v]!;
        water = Math.max(water, Math.abs(result.waterVolume_m3[v]! - o.waterVolume_m3[v]!) / capacity);
        for (let a = 0; a < 3; a++) { speed = Math.max(speed, Math.abs(o.velocities_m_s[3 * v + a]!)); velocity = Math.max(velocity, Math.abs(result.velocities_m_s[3 * v + a]! - o.velocities_m_s[3 * v + a]!)); }
      }
      console.log(`# ${label}: captured ${result.captured_m3.toExponential(4)}/${captured.toExponential(4)} released ${result.released_m3.toExponential(4)}/${released.toExponential(4)} water ${water.toExponential(2)} velocity ${(velocity / speed).toExponential(2)} cells ${table.cells.length} gpu ${result.gpu_ms?.toFixed(3)} ms`);
      assert.ok(water <= BOUND, `saturation ${water}`);
      assert.ok(velocity <= BOUND * speed, `velocity ${velocity / speed}`);
      // The published partition: the oracle's cells, in its order.
      const rows = await gpu.readCapacityTable();
      assert.equal(rows.length / 5, table.cells.length, 'capacity cell count');
      table.cells.forEach((cell, i) => {
        assert.deepEqual([...rows.subarray(5 * i, 5 * i + 3)], [...cell.p], `capacity cell ${i}`);
        assert.ok(Math.abs(rows[5 * i + 3]! - cell.solidFraction) <= 1e-5 * cell.solidFraction + 1e-9, `fiber fraction ${i}`);
        assert.ok(Math.abs(rows[5 * i + 4]! - cell.boundFraction) <= BOUND * cell.boundFraction + 1e-9, `bound fraction ${i}`);
      });
      assert.equal(fields.porousCapacity, gpu.capacityTexture, 'NB-FLIP reads the GPU partition');
      // A submission that never runs (Metal aborts are silent) leaves the previous
      // stamp in the readback: the exchange refuses it and publishes nothing.
      const queue = device.queue, submit = queue.submit, dropped = () => undefined;
      (queue as unknown as { submit: () => undefined }).submit = dropped;
      assert.equal(device.queue.submit, dropped, 'the fixture drops the next queue submission');
      try {
        await assert.rejects(() => gpu!.step({ positions_m: s.positions_m, velocities_m_s: s.velocities_m_s, controlVolume_m3: s.controlVolume_m3, solidVolume_m3: s.solidVolume_m3, solidFraction: s.solidFraction,
          dryMass_kg: s.dryMass_kg, waterVolume_m3: s.waterVolume_m3, fixed: f.fixed, materials: f.materials, physics, dt_s: dt, gravity_m_s2: gravity,
          pressureAcceleration_m_s2: f.pressure, solidAcceleration_m_s2: f.solid, identity: { ...identity, tick: 2 }, vertexMaterial: f.prepared.vertexMaterial }),
          /^Error: Cloth GPU pore exchange: stale GPU readback \(stamp 1, expected 2\)$/);
      } finally { queue.submit = submit; }
      assert.equal(fields.porousCapacity, gpu.capacityTexture, 'a refused readback leaves the published partition in place');
    } finally { gpu?.destroy(); transport.destroy(); cpu.destroy(); solver.destroy(); }
  });
});

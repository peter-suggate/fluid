import assert from "node:assert/strict";
import test from "node:test";
import {
  measureFluidSymmetry,
  measureHorizontalFrontCircularity,
} from "../lib/fluid-symmetry-diagnostic";

function symmetricState() {
  const grid = [6, 3, 6] as const;
  const [nx, ny, nz] = grid;
  const cells = nx * ny * nz;
  const volume = new Float32Array(cells);
  const velocity = new Float32Array(3 * cells);
  const pressure = new Float32Array(cells);
  const topology = new Uint32Array(cells);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const cell = x + nx * (y + ny * z);
    const dx = x - 0.5 * (nx - 1), dz = z - 0.5 * (nz - 1);
    volume[cell] = x === 0 || x === nx - 1 || z === 0 || z === nz - 1 ? 0.25 : 1;
    pressure[cell] = dx * dx + dz * dz + y;
    topology[cell] = Math.abs(dx) > 1 || Math.abs(dz) > 1 ? 2 : 1;
    velocity[3 * cell] = dx;
    velocity[3 * cell + 1] = -y;
    velocity[3 * cell + 2] = dz;
  }
  return { time_s: 0.1, grid, volume, velocity, pressure, rhs: pressure.slice(),
    diagonal: pressure.slice(), topology, wallLiquidThreshold: 0.1 };
}

test("D4 diagnostic accepts exact scalar, vector, topology, and wall symmetry", () => {
  const observation = measureFluidSymmetry(symmetricState());
  for (const metrics of [observation.volume, observation.velocity, observation.pressure,
    observation.rhs, observation.diagonal, observation.topology]) {
    assert.equal(metrics.maximumAbsoluteError, 0);
    assert.equal(metrics.exactMismatchCount, 0);
    assert.equal(metrics.nonFiniteCount, 0);
  }
  assert.deepEqual(Object.values(observation.walls).map(({ touched }) => touched), [true, true, true, true]);
});

test("D4 diagnostic locates the first scalar and signed-vector discrepancy", () => {
  const state = symmetricState();
  state.volume[0] += 0.125;
  state.velocity[0] += 0.5;
  const observation = measureFluidSymmetry(state);
  assert.ok(observation.volume.maximumAbsoluteError > 0);
  assert.ok(observation.volume.first);
  assert.ok(observation.velocity.maximumAbsoluteError > 0);
  assert.ok(observation.velocity.first?.component);
});

test("D4 diagnostic fails closed on incomplete adaptive fields", () => {
  const state = symmetricState();
  state.pressure[5] = Number.NaN;
  assert.ok(measureFluidSymmetry(state).pressure.nonFiniteCount > 0);
});

test("radial diagnostic distinguishes circular and axis-leading D4 fronts", () => {
  const grid = [32, 1, 32] as const;
  const circle = new Float32Array(32 * 32), axisBiased = new Float32Array(32 * 32);
  const centre = 15.5;
  for (let z = 0; z < 32; z += 1) for (let x = 0; x < 32; x += 1) {
    const dx = x - centre, dz = z - centre;
    circle[x + 32 * z] = Math.hypot(dx, dz) <= 10 ? 1 : 0;
    axisBiased[x + 32 * z] = Math.abs(dx) + 1.5 * Math.abs(dz) <= 12 ? 1 : 0;
  }
  const round = measureHorizontalFrontCircularity(circle, grid);
  const biased = measureHorizontalFrontCircularity(axisBiased, grid);
  assert.ok(Math.abs(round.axisLead_cells) < 0.2, JSON.stringify(round));
  assert.ok(biased.axisLead_cells > 2, JSON.stringify(biased));
});

import assert from "node:assert/strict";
import test from "node:test";
import {
  advectAndRedistanceLevelSet,
  applyVariationalMatrix,
  buildQuadtree,
  buildVariationalSystem,
  populateTallPressureGrid,
  quadtreeSizingFromVelocityAndSurface,
  signedDistanceFromVolume
} from "../lib/quadtree-tall-cell-grid";

const index3 = (x: number, y: number, z: number, nx: number, ny: number) => x + nx * (y + ny * z);

test("VOF reconstruction produces an anisotropic signed-distance field", () => {
  const volume = new Float32Array([1, 1, 0, 0]);
  const phi = signedDistanceFromVolume(volume, 1, 4, 1, { x: 2, y: 0.5, z: 3 });
  assert.deepEqual(Array.from(phi), [-0.75, -0.25, 0.25, 0.75]);
});

test("the independently transported level set remains finite and translates a planar interface", () => {
  const nx = 7, ny = 3, nz = 3, h = { x: 1, y: 1, z: 1 };
  const phi = new Float32Array(nx * ny * nz);
  const velocity = Array.from({ length: phi.length }, () => ({ x: 1, y: 0, z: 0 }));
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    phi[index3(x, y, z, nx, ny)] = x - 2.5;
  }
  const transported = advectAndRedistanceLevelSet(phi, velocity, nx, ny, nz, h, 1);
  assert.ok(transported.every(Number.isFinite));
  assert.ok(transported[index3(3, 1, 1, nx, ny)] < 0, "interface must move one cell with the flow");
  assert.ok(transported[index3(4, 1, 1, nx, ny)] > 0, "translated interface must retain its sign change");
  assert.ok(Math.abs(Math.abs(transported[index3(5, 1, 1, nx, ny)] - transported[index3(4, 1, 1, nx, ny)]) - 1) < 1e-6, "redistancing must restore a unit gradient");
});

test("quadtree subdivision evaluates cell centres coarse-to-fine and remains 2:1", () => {
  const nx = 16, nz = 16, sizing = new Float32Array(nx * nz);
  sizing[8 + nx * 8] = 100; // root centre
  sizing[4 + nx * 4] = 100; // refine one child
  sizing[2 + nx * 2] = 100; // and one grandchild
  sizing[1 + nx] = 100; // reach the finest level
  const grid = buildQuadtree(sizing, nx, nz, { h: 1, maximumLeafSize: 16, smoothingDilations: 3 });
  assert.ok(grid.leaves.some((leaf) => leaf.size === 1));
  assert.ok(grid.leaves.some((leaf) => leaf.size === 2));
  assert.equal(grid.maximumNeighborRatio, 2);
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) assert.ok(grid.leafAt[x + nx * z] >= 0);
});

test("adaptivity-strength zero produces the ordinary-grid limit", () => {
  const sizing = new Float32Array(64).fill(1e6);
  const grid = buildQuadtree(sizing, 8, 8, { h: 0.1, maximumLeafSize: 8, adaptivityStrength: 0 });
  assert.equal(grid.leaves.length, 64);
  assert.ok(grid.leaves.every((leaf) => leaf.size === 1));
});

test("dyadic roots preserve adaptivity on non-power-of-two matched grids", () => {
  const nx = 61, nz = 41;
  const grid = buildQuadtree(new Float32Array(nx * nz), nx, nz, { h: 0.02, maximumLeafSize: 8 });
  assert.ok(grid.leaves.length < nx * nz / 2);
  assert.ok(grid.leaves.some((leaf) => leaf.size === 8));
  assert.ok(grid.leafAt.every((leaf) => leaf >= 0));
  assert.ok(grid.maximumNeighborRatio <= 2);
});

test("three-pass adaptivity smoothing expands refinement beyond strict balancing", () => {
  const nx = 16, sizing = new Float32Array(nx * nx);
  for (const [x, z] of [[8, 8], [4, 4], [2, 2], [1, 1]] as const) sizing[x + nx * z] = 100;
  const balanced = buildQuadtree(sizing, nx, nx, { h: 1, maximumLeafSize: 16, smoothingDilations: 0 });
  const smoothed = buildQuadtree(sizing, nx, nx, { h: 1, maximumLeafSize: 16, smoothingDilations: 3 });
  assert.ok(smoothed.leaves.length > balanced.leaves.length);
  assert.ok(smoothed.maximumNeighborRatio <= 2);
});

test("vertical population creates multiple tall cells for disconnected liquid", () => {
  const nx = 2, ny = 12, nz = 2, h = { x: 1, y: 1, z: 1 };
  const quadtree = buildQuadtree(new Float32Array(nx * nz), nx, nz, { h: 1, maximumLeafSize: 2 });
  const phi = new Float32Array(nx * ny * nz).fill(10);
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    for (let y = 0; y <= 4; y += 1) phi[index3(x, y, z, nx, ny)] = -10;
    for (let y = 7; y <= 10; y += 1) phi[index3(x, y, z, nx, ny)] = -10;
  }
  const grid = populateTallPressureGrid(quadtree, phi, ny, h, 0);
  const liquidTall = grid.segments.filter((segment) => segment.tall && grid.samples[segment.bottomSample].liquid);
  assert.ok(liquidTall.length >= 2, "ground-connected and airborne liquid must not be merged");
  for (const segment of grid.segments.filter((candidate) => candidate.tall)) {
    assert.equal(grid.samples[segment.bottomSample].y, segment.firstY);
    assert.equal(grid.samples[segment.topSample].y, segment.lastY);
  }
});

test("T-junction gradient uses vertically interpolated pressures and 1.5 dx", () => {
  const nx = 4, ny = 3, nz = 4, h = { x: 1, y: 1, z: 1 };
  const sizing = new Float32Array(nx * nz); sizing[2 + nx * 2] = 100; sizing[1 + nx] = 100;
  const quadtree = buildQuadtree(sizing, nx, nz, { h: 1, maximumLeafSize: 4, smoothingDilations: 0 });
  const phi = new Float32Array(nx * ny * nz).fill(-10);
  const velocity = Array.from({ length: phi.length }, () => ({ x: 0, y: 0, z: 0 }));
  const pressureGrid = populateTallPressureGrid(quadtree, phi, ny, h, ny);
  const system = buildVariationalSystem(pressureGrid, { velocity });
  const transition = system.faces.find((face) => face.axis === 0 && Math.abs(face.coefficients.filter((value) => value < 0).reduce((sum, value) => sum + value, 0) + 2 / 3) < 1e-12);
  assert.ok(transition, "a 2:1 face must divide its pressure difference by 1.5 finest cells");
  assert.equal(transition.bounds.y1 - transition.bounds.y0, ny, "a tall shared face must be integrated rather than expanded into background cubes");
});

test("variational matrix is symmetric positive semidefinite and uses ghost volumes", () => {
  const nx = 2, ny = 10, nz = 2, h = { x: 0.5, y: 0.25, z: 0.5 };
  const quadtree = buildQuadtree(new Float32Array(nx * nz), nx, nz, { h: h.x, maximumLeafSize: 2 });
  const phi = new Float32Array(nx * ny * nz).fill(-2);
  // An air cap supplies a Dirichlet free surface while the deep run remains tall.
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) for (let y = 8; y < ny; y += 1) phi[index3(x, y, z, nx, ny)] = 2;
  const velocity = Array.from({ length: phi.length }, (_, index) => ({ x: Math.sin(index), y: Math.cos(index), z: 0.1 * index }));
  const grid = populateTallPressureGrid(quadtree, phi, ny, h, 1);
  const system = buildVariationalSystem(grid, { velocity });
  const n = system.liquidSampleIds.length;
  assert.ok(system.faces.some((face) => face.axis === 1 && face.ghost && face.volume > h.x * h.y * h.z));
  for (let row = 0; row < n; row += 1) for (let column = 0; column < n; column += 1) {
    assert.ok(Math.abs(system.matrix[row * n + column] - system.matrix[column * n + row]) < 1e-12);
  }
  const vector = Float64Array.from({ length: n }, (_, index) => Math.sin(1.7 * index));
  const applied = applyVariationalMatrix(system, vector);
  const energy = vector.reduce((sum, value, index) => sum + value * applied[index], 0);
  assert.ok(energy >= -1e-11, `G^T V A F G energy was ${energy}`);
});

test("matrix pressure correction drives the variational divergence residual down", () => {
  const nx = 2, ny = 8, nz = 2, h = { x: 0.5, y: 0.25, z: 0.5 };
  const quadtree = buildQuadtree(new Float32Array(nx * nz), nx, nz, { h: h.x, maximumLeafSize: 2 });
  const phi = new Float32Array(nx * ny * nz).fill(-1);
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) phi[index3(x, ny - 1, z, nx, ny)] = 1;
  const velocity = Array.from({ length: phi.length }, (_, index) => ({ x: 0.02 * index, y: Math.sin(index) * 0.1, z: -0.01 * index }));
  const system = buildVariationalSystem(populateTallPressureGrid(quadtree, phi, ny, h, 1), { velocity });
  const n = system.rhs.length, pressure = new Float64Array(n), residual = Float64Array.from(system.rhs), direction = Float64Array.from(residual);
  let rr = residual.reduce((sum, value) => sum + value * value, 0); const initial = Math.sqrt(rr);
  for (let iteration = 0; iteration < Math.max(8, n * 2) && Math.sqrt(rr) > 1e-10 * Math.max(1, initial); iteration += 1) {
    const ad = applyVariationalMatrix(system, direction);
    const denominator = direction.reduce((sum, value, index) => sum + value * ad[index], 0);
    if (Math.abs(denominator) < 1e-20) break;
    const alpha = rr / denominator;
    for (let i = 0; i < n; i += 1) { pressure[i] += alpha * direction[i]; residual[i] -= alpha * ad[i]; }
    const next = residual.reduce((sum, value) => sum + value * value, 0), beta = next / rr;
    for (let i = 0; i < n; i += 1) direction[i] = residual[i] + beta * direction[i];
    rr = next;
  }
  assert.ok(Math.sqrt(rr) <= Math.max(1e-8, initial * 1e-6), `${Math.sqrt(rr)} did not sufficiently reduce ${initial}`);
});

test("paper sizing function flattens curvature and non-translation velocity maxima", () => {
  const nx = 3, ny = 5, nz = 3, h = { x: 1, y: 1, z: 1 }, count = nx * ny * nz;
  const phi = new Float32Array(count).fill(4), velocity = Array.from({ length: count }, () => ({ x: 1, y: 2, z: 3 }));
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) phi[index3(x, 2, z, nx, ny)] = 0;
  const translation = quadtreeSizingFromVelocityAndSurface(phi, velocity, nx, ny, nz, h);
  velocity[index3(1, 2, 1, nx, ny)] = { x: 20, y: -10, z: 7 };
  const disturbed = quadtreeSizingFromVelocityAndSurface(phi, velocity, nx, ny, nz, h);
  assert.ok(disturbed.some((value, index) => value > translation[index]));
});

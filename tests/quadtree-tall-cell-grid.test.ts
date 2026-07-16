import assert from "node:assert/strict";
import test from "node:test";
import {
  adaptivePressureCellIds,
  adaptivePressureCellTopology,
  advectAndRedistanceLevelSet,
  applyVariationalMatrix,
  buildMlsProjectionRows,
  buildQuadtree,
  buildVariationalSystem,
  maximumFluidScale,
  populateTallPressureGrid,
  populateTallPressureGridFromLeafProfiles,
  quadtreeFromPackedCells,
  quadtreeSizingFromVelocityAndSurface,
  reconcileLevelSetWithVolume,
  signedDistanceFromVolume
} from "../lib/quadtree-tall-cell-grid";

const index3 = (x: number, y: number, z: number, nx: number, ny: number) => x + nx * (y + ny * z);

test("packed GPU leaf maps reconstruct the CPU quadtree exactly", () => {
  const nx = 16, nz = 8, sizing = new Float32Array(nx * nz);
  for (let z = 2; z < 5; z += 1) for (let x = 3; x < 7; x += 1) sizing[x + nx * z] = 100;
  const reference = buildQuadtree(sizing, nx, nz, { h: 1, maximumLeafSize: 8, smoothingDilations: 1 });
  const packed = new Uint32Array(nx * nz);
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const leaf = reference.leaves[reference.leafAt[x + nx * z]];
    packed[x + nx * z] = leaf.x | (leaf.z << 10) | (leaf.size << 20);
  }
  const rebuilt = quadtreeFromPackedCells(packed, nx, nz);
  assert.deepEqual(
    rebuilt.leaves.map(({ x, z, size }) => ({ x, z, size })),
    reference.leaves.map(({ x, z, size }) => ({ x, z, size })).sort((a, b) => a.z - b.z || a.x - b.x || b.size - a.size)
  );
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const leaf = rebuilt.leaves[rebuilt.leafAt[x + nx * z]];
    assert.equal(leaf.x | (leaf.z << 10) | (leaf.size << 20), packed[x + nx * z]);
  }
  assert.ok(rebuilt.maximumNeighborRatio <= 2);
});

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

test("VOF reconciliation reseeds liquid the advected level set never saw", () => {
  const nx = 9, ny = 5, nz = 5, h = { x: 1, y: 1, z: 1 };
  const phi = new Float32Array(nx * ny * nz), volume = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const index = index3(x, y, z, nx, ny);
    phi[index] = y - 1.5; // pool occupying the two bottom layers
    volume[index] = y < 2 ? 1 : 0;
    // An inflow blob the transported level set knows nothing about.
    if (x >= 6 && x <= 7 && y === 3 && z >= 2 && z <= 3) volume[index] = 1;
  }
  const { phi: reconciled, mismatchFraction } = reconcileLevelSetWithVolume(phi, volume, nx, ny, nz, h);
  assert.ok(mismatchFraction > 0, "the blob must register as a sign mismatch");
  assert.ok(reconciled[index3(6, 3, 2, nx, ny)] < 0, "VOF liquid must become level-set liquid");
  assert.ok(reconciled[index3(2, 3, 2, nx, ny)] > 0, "air away from the blob must stay air");
  assert.ok(reconciled[index3(2, 1, 2, nx, ny)] < 0, "the original pool must remain liquid");
  assert.ok(reconciled.every(Number.isFinite));
  const gradient = Math.abs(reconciled[index3(2, 4, 2, nx, ny)] - reconciled[index3(2, 3, 2, nx, ny)]);
  assert.ok(Math.abs(gradient - 1) < 0.35, `redistancing must restore a near-unit gradient, got ${gradient}`);
});

test("VOF reconciliation preserves agreeing sub-cell distances", () => {
  const nx = 5, ny = 6, nz = 5, h = { x: 1, y: 1, z: 1 };
  const phi = new Float32Array(nx * ny * nz), volume = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const index = index3(x, y, z, nx, ny);
    phi[index] = y - 2.25; // interface a quarter-cell above the y=2 centre
    volume[index] = y < 2 ? 1 : y === 2 ? 0.75 : 0;
  }
  const { phi: reconciled, mismatchFraction } = reconcileLevelSetWithVolume(phi, volume, nx, ny, nz, h);
  assert.equal(mismatchFraction, 0, "matching wet/dry signs must not count as drift");
  assert.ok(Math.abs(reconciled[index3(2, 2, 2, nx, ny)] + 0.25) < 1e-6, "the advected sub-cell offset must survive reconciliation");
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

test("adaptive pressure-cell ids collapse quadtree leaves and tall segments", () => {
  const nx = 4, ny = 8, nz = 4, h = { x: 1, y: 1, z: 1 };
  const quadtree = buildQuadtree(new Float32Array(nx * nz), nx, nz, { h: 1, maximumLeafSize: 4 });
  const phi = new Float32Array(nx * ny * nz).fill(-1);
  const grid = populateTallPressureGrid(quadtree, phi, ny, h, 0), ids = adaptivePressureCellIds(grid);
  const first = ids[index3(0, 0, 0, nx, ny)];
  assert.ok(first > 0);
  assert.equal(ids[index3(3, 7, 3, nx, ny)], first, "one coarse tall cell must own its entire dense backing region");
  assert.equal(new Set(ids).size, grid.segments.length, "every adaptive segment must have one debug-cell id");
});

test("adaptive debug topology carries the complete represented-cell bounds", () => {
  const nx = 4, ny = 8, nz = 4, h = { x: 1, y: 1, z: 1 };
  const quadtree = buildQuadtree(new Float32Array(nx * nz), nx, nz, { h: 1, maximumLeafSize: 4 });
  const grid = populateTallPressureGrid(quadtree, new Float32Array(nx * ny * nz).fill(-1), ny, h, 0);
  const topology = adaptivePressureCellTopology(grid);
  const offset = 2 * index3(3, 7, 3, nx, ny), horizontal = topology[offset], vertical = topology[offset + 1];
  assert.deepEqual({ x: horizontal & 1023, z: (horizontal >>> 10) & 1023, size: (horizontal >>> 20) & 1023 }, { x: 0, z: 0, size: 4 });
  assert.deepEqual({ firstY: vertical & 1023, lastYExclusive: (vertical >>> 10) & 1023 }, { firstY: 0, lastYExclusive: 8 });
});

test("every retained cube is its own single-cell cubic segment", () => {
  const nx = 2, ny = 12, nz = 2, h = { x: 1, y: 1, z: 1 };
  const quadtree = buildQuadtree(new Float32Array(nx * nz), nx, nz, { h: 1, maximumLeafSize: 2 });
  const phi = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    phi[index3(x, y, z, nx, ny)] = y - 8.5; // deep pool, surface between y=8 and 9
  }
  const optical = 3;
  const grid = populateTallPressureGrid(quadtree, phi, ny, h, optical);
  for (const segment of grid.segments.filter((candidate) => !candidate.tall)) {
    assert.equal(segment.firstY, segment.lastY, `cubic segment [${segment.firstY}, ${segment.lastY}] must span exactly one cube`);
    assert.equal(grid.samples[segment.bottomSample].kind, "cubic");
  }
  for (const column of grid.samplesByLeaf) {
    // The optical band below each interface must resolve every cube separately.
    const cubicSamples = column.filter((sample) => sample.kind === "cubic");
    assert.ok(cubicSamples.length >= optical, `expected at least ${optical} cubic samples, found ${cubicSamples.length}`);
    const ys = cubicSamples.map((sample) => sample.y);
    for (let index = 1; index < ys.length; index += 1) assert.equal(ys[index], ys[index - 1] + 1, "cubic band samples must sit on consecutive cells");
  }
});

test("optical layers use each connected column's local liquid depth", () => {
  const nx = 2, ny = 12, nz = 1, h = { x: 1, y: 1, z: 1 };
  const quadtree = buildQuadtree(new Float32Array(nx * nz).fill(1e6), nx, nz, { h: 1, maximumLeafSize: 1 });
  const profiles = new Float32Array(quadtree.leaves.length * ny).fill(10);
  for (const leaf of quadtree.leaves) {
    const depth = leaf.x === 0 ? 4 : 8;
    for (let y = 0; y < depth; y += 1) profiles[leaf.id * ny + y] = -10;
  }
  const grid = populateTallPressureGridFromLeafProfiles(quadtree, profiles, ny, h, 0.25);
  const cubicCounts = grid.samplesByLeaf.map((samples) => samples.filter((sample) => sample.kind === "cubic").length);
  assert.ok(cubicCounts[1] > cubicCounts[0], `deep local column should retain a thicker optical band (${cubicCounts})`);
});

test("leaf-centre profile rebuilds reproduce dense phi tall-cell segmentation", () => {
  const nx = 4, ny = 9, nz = 4, h = { x: 1, y: 1, z: 1 };
  const sizing = new Float32Array(nx * nz); sizing[3 + nx * 3] = 100;
  const quadtree = buildQuadtree(sizing, nx, nz, { h: 1, maximumLeafSize: 4 });
  const phi = Float32Array.from({ length: nx * ny * nz }, (_, index) => {
    const x = index % nx, y = Math.floor(index / nx) % ny, z = Math.floor(index / (nx * ny));
    return y - 4.25 + 0.1 * x - 0.05 * z;
  });
  const dense = populateTallPressureGrid(quadtree, phi, ny, h, 1, 0.25);
  const profiles = new Float32Array(quadtree.leaves.length * ny);
  for (const leaf of quadtree.leaves) for (let y = 0; y < ny; y += 1) profiles[leaf.id * ny + y] = y - 4.25 + 0.1 * (leaf.x + leaf.size / 2 - 0.5) - 0.05 * (leaf.z + leaf.size / 2 - 0.5);
  const compact = populateTallPressureGridFromLeafProfiles(quadtree, profiles, ny, h, 0.25);
  assert.deepEqual(compact.segments, dense.segments);
  assert.deepEqual(compact.samples.map((sample) => ({ leaf: sample.leaf, y: sample.y, liquid: sample.liquid, kind: sample.kind })), dense.samples.map((sample) => ({ leaf: sample.leaf, y: sample.y, liquid: sample.liquid, kind: sample.kind })));
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

test("free-surface scale is clamped for nearly-emptied liquid samples", () => {
  const nx = 2, ny = 4, nz = 2, h = { x: 0.5, y: 0.25, z: 0.5 };
  const quadtree = buildQuadtree(new Float32Array(nx * nz), nx, nz, { h: h.x, maximumLeafSize: 1 });
  // A liquid sample with phi barely below zero across from full-depth air makes
  // Ando--Batty's ratio (air + liquid contributions) / (liquid contribution)
  // arbitrarily large without the ghost-fluid interface floor.
  const phi = new Float32Array(nx * ny * nz).fill(1);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) phi[index3(0, y, z, nx, ny)] = -1e-9;
  const velocity = Array.from({ length: phi.length }, (_, index) => ({ x: 0.1 * Math.sin(index), y: 0, z: 0 }));
  const system = buildVariationalSystem(populateTallPressureGrid(quadtree, phi, ny, h, 1), { velocity });
  assert.ok(system.faces.length > 0);
  for (const face of system.faces) {
    assert.ok(face.fluidScale >= 0 && face.fluidScale <= maximumFluidScale, `face scale ${face.fluidScale} escaped [0, ${maximumFluidScale}]`);
  }
  assert.ok(system.faces.some((face) => face.fluidScale === maximumFluidScale), "the degenerate interface must reach the ceiling");
  const n = system.liquidSampleIds.length;
  for (let row = 0; row < n; row += 1) for (let column = 0; column < n; column += 1) {
    assert.ok(Math.abs(system.matrix[row * n + column] - system.matrix[column * n + row]) < 1e-12);
  }
  const vector = Float64Array.from({ length: n }, (_, index) => Math.cos(2.3 * index));
  const applied = applyVariationalMatrix(system, vector);
  const energy = vector.reduce((sum, value, index) => sum + value * applied[index], 0);
  assert.ok(energy >= -1e-11, `clamped G^T V A F G energy was ${energy}`);
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

function submergedBoxFixture(bodyVelocity: { x: number; y: number; z: number }, inverseMass: number, inverseInertia: number[]) {
  // A closed 6x10x6 tank, liquid to y=8, with a 2x2x2 solid box centred inside.
  const nx = 6, ny = 10, nz = 6, h = { x: 0.1, y: 0.1, z: 0.1 };
  const quadtree = buildQuadtree(new Float32Array(nx * nz), nx, nz, { h: h.x, maximumLeafSize: 1 });
  const phi = new Float32Array(nx * ny * nz), solidFraction = new Float32Array(nx * ny * nz), solidOwner = new Int32Array(nx * ny * nz).fill(-1);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const index = index3(x, y, z, nx, ny);
    phi[index] = (y - 7.5) * h.y;
    if (x >= 2 && x <= 3 && y >= 3 && y <= 4 && z >= 2 && z <= 3) { solidFraction[index] = 1; solidOwner[index] = 0; }
  }
  const g = 9.81, dt = 0.01;
  const velocity = Array.from({ length: nx * ny * nz }, () => ({ x: 0, y: -g * dt, z: 0 }));
  const body = {
    position: { x: 3 * h.x, y: 4 * h.y, z: 3 * h.z },
    linearVelocity: bodyVelocity, angularVelocity: { x: 0, y: 0, z: 0 },
    inverseMass, inverseInertia
  };
  // A deep optical band keeps every liquid cell cubic so the box boundary is
  // resolved by per-cell faces.
  const grid = populateTallPressureGrid(quadtree, phi, ny, h, 8);
  const system = buildVariationalSystem(grid, { velocity, solidFraction, solidOwner, bodies: [body] });
  return { system, body, g, dt, h };
}

function solveDense(system: ReturnType<typeof buildVariationalSystem>) {
  const n = system.liquidSampleIds.length;
  const pressure = new Float64Array(n), residual = Float64Array.from(system.rhs), direction = Float64Array.from(residual);
  let rr = residual.reduce((sum, value) => sum + value * value, 0);
  const initial = Math.sqrt(rr);
  for (let iteration = 0; iteration < 6 * n && Math.sqrt(rr) > 1e-12 * Math.max(1, initial); iteration += 1) {
    const ad = applyVariationalMatrix(system, direction);
    const denominator = direction.reduce((sum, value, index) => sum + value * ad[index], 0);
    if (Math.abs(denominator) < 1e-25) break;
    const alpha = rr / denominator;
    for (let i = 0; i < n; i += 1) { pressure[i] += alpha * direction[i]; residual[i] -= alpha * ad[i]; }
    const next = residual.reduce((sum, value) => sum + value * value, 0), beta = next / rr;
    for (let i = 0; i < n; i += 1) direction[i] = residual[i] + beta * direction[i];
    rr = next;
  }
  return pressure;
}

function bodyPressureDelta(system: ReturnType<typeof buildVariationalSystem>, body: { inverseMass: number; inverseInertia: number[] }, pressure: Float64Array) {
  const generalized = new Float64Array(6);
  for (const [dof, row] of system.couplings[0].rows) for (let component = 0; component < 6; component += 1) generalized[component] += row[component] * pressure[dof];
  const inertia = body.inverseInertia;
  return {
    linear: { x: -body.inverseMass * generalized[0], y: -body.inverseMass * generalized[1], z: -body.inverseMass * generalized[2] },
    angular: {
      x: -(inertia[0] * generalized[3] + inertia[1] * generalized[4] + inertia[2] * generalized[5]),
      y: -(inertia[3] * generalized[3] + inertia[4] * generalized[4] + inertia[5] * generalized[5]),
      z: -(inertia[6] * generalized[3] + inertia[7] * generalized[4] + inertia[8] * generalized[5])
    }
  };
}

test("solid area fractions close faces and keep the coupled matrix SPD", () => {
  const { system } = submergedBoxFixture({ x: 0, y: 0, z: 0 }, 0.5, [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1]);
  assert.ok(system.faces.some((face) => face.openFraction === 0), "faces fully inside the box must be closed");
  assert.ok(system.couplings.length === 1 && system.couplings[0].rows.size > 0, "the box must produce coupling rows");
  const n = system.liquidSampleIds.length;
  for (let row = 0; row < n; row += 1) for (let column = 0; column < n; column += 1) {
    assert.ok(Math.abs(system.matrix[row * n + column] - system.matrix[column * n + row]) < 1e-12, "coupled matrix must stay symmetric");
  }
  const vector = Float64Array.from({ length: n }, (_, index) => Math.sin(3.1 * index));
  const applied = applyVariationalMatrix(system, vector);
  const energy = vector.reduce((sum, value, index) => sum + value * applied[index], 0);
  assert.ok(energy >= -1e-11, `coupled G^T V A F G + K M^-1 K^T energy was ${energy}`);
});

test("a kinematically driven solid forces fluid through the coupled right-hand side", () => {
  const still = submergedBoxFixture({ x: 0, y: 0, z: 0 }, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const rising = submergedBoxFixture({ x: 0, y: 1, z: 0 }, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const difference = rising.system.rhs.map((value, index) => value - still.system.rhs[index]);
  assert.ok(difference.some((value) => Math.abs(value) > 1e-9), "solid motion must appear in the divergence right-hand side");
});

test("a neutrally buoyant submerged body is held against gravity by the monolithic solve", () => {
  // rho_body = rho_fluid: the pressure impulse must cancel the gravity kick.
  const boxVolume = 2 * 2 * 2 * 0.1 * 0.1 * 0.1 * 1.0; // slightly enlarged by cell sampling; exact value below from fractions
  void boxVolume;
  const mass_over_rho = 8 * 0.001; // 8 cells of 0.001 m^3 at rho_body = rho_fluid
  const inverseMass = 1 / mass_over_rho; // rho/m with rho folded in
  const { system, body, g, dt } = submergedBoxFixture({ x: 0, y: -9.81 * 0.01, z: 0 }, inverseMass, [0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const pressure = solveDense(system);
  const delta = bodyPressureDelta(system, body, pressure);
  const final = body.linearVelocity.y + delta.linear.y;
  assert.ok(delta.linear.y > 0, `pressure impulse must push the submerged body up, got ${delta.linear.y}`);
  assert.ok(Math.abs(final) < 0.35 * g * dt, `neutral body must be nearly held: kick ${-g * dt}, final ${final}`);
});

function tJunctionSystem(fullyLiquid = false) {
  // One coarse column against fine columns, deep liquid: T-junction faces with
  // multiple sub-faces exist and everything below the cap is liquid. The
  // fully-liquid variant removes Dirichlet air samples so that MLS linear
  // reproduction can be asserted without surface truncation.
  const nx = 4, ny = 6, nz = 4, h = { x: 1, y: 1, z: 1 };
  const sizing = new Float32Array(nx * nz); sizing[1 + nx] = 100;
  const quadtree = buildQuadtree(sizing, nx, nz, { h: 1, maximumLeafSize: 4, smoothingDilations: 0 });
  const phi = new Float32Array(nx * ny * nz).fill(-10);
  if (!fullyLiquid) for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) phi[index3(x, ny - 1, z, nx, ny)] = 10;
  const velocity = Array.from({ length: phi.length }, () => ({ x: 0, y: 0, z: 0 }));
  const grid = populateTallPressureGrid(quadtree, phi, ny, h, ny);
  return buildVariationalSystem(grid, { velocity });
}

test("MLS pressure mapping is exact for linear fields: sub-face corrections collapse to the face value", () => {
  // Under a linear pressure field the MLS sub-face gradients are exact and
  // identical, so after the conservation shift every sub-face carries exactly
  // the solved variational face value. Any MLS inexactness would appear as
  // variation across a face's sub-faces.
  const system = tJunctionSystem(true);
  const rows = buildMlsProjectionRows(system);
  assert.ok(rows.length > 0, "T-junction faces must produce MLS sub-face rows");
  const gradient = { x: 0.7, y: -0.4, z: 1.3 };
  const pressure = new Float64Array(system.liquidSampleIds.length);
  system.liquidSampleIds.forEach((sampleId, dof) => {
    const position = system.grid.samples[sampleId].position;
    pressure[dof] = gradient.x * position.x + gradient.y * position.y + gradient.z * position.z;
  });
  const byKey = new Map<string, number>();
  rows.forEach((row, index) => byKey.set(`${row.cell}:${row.axis}`, index));
  const { grid } = system, { quadtree, ny } = grid;
  let checkedFaces = 0;
  for (const face of system.faces) {
    if (face.axis === 1) continue;
    const values: number[] = [];
    for (let y = face.bounds.y0; y < face.bounds.y1 && y < ny; y += 1) for (let transverse = 0; transverse < face.bounds.span; transverse += 1) {
      const x = face.axis === 0 ? face.bounds.x : face.bounds.x + transverse;
      const z = face.axis === 2 ? face.bounds.z : face.bounds.z + transverse;
      const index = byKey.get(`${index3(x, y, z, quadtree.nx, ny)}:${face.axis}`);
      if (index !== undefined) values.push(rows[index].entries.reduce((sum, [dof, weight]) => sum + weight * pressure[dof], 0));
    }
    if (values.length <= 1) continue;
    const solved = face.nodes.reduce((sum, node, slot) => {
      const dof = system.dofBySample[node];
      return dof < 0 ? sum : sum + face.coefficients[slot] * pressure[dof];
    }, 0);
    for (const value of values) assert.ok(Math.abs(value - solved) < 1e-9, `sub-face ${value} must equal the exact face value ${solved}`);
    checkedFaces += 1;
  }
  assert.ok(checkedFaces > 0, "at least one multi-sub-face horizontal face must be checked");
});

test("MLS sub-face corrections average exactly to the solved face value", () => {
  const system = tJunctionSystem();
  const rows = buildMlsProjectionRows(system);
  const pressure = Float64Array.from({ length: system.liquidSampleIds.length }, (_, index) => Math.sin(2.9 * index) + 0.3 * index);
  // Group rows back to their variational faces via cellProjection-equivalent
  // enumeration: rebuild the same grouping from the faces.
  const { grid } = system, { quadtree, ny } = grid;
  const byKey = new Map<string, number>();
  rows.forEach((row, index) => byKey.set(`${row.cell}:${row.axis}`, index));
  for (const face of system.faces) {
    const subCells: number[] = [];
    if (face.axis === 1) {
      const leaf = quadtree.leaves[grid.samples[face.nodes[0]].leaf];
      for (let z = leaf.z; z < leaf.z + leaf.size; z += 1) for (let x = leaf.x; x < leaf.x + leaf.size; x += 1) for (let y = face.bounds.y0; y < face.bounds.y1 && y < ny; y += 1) subCells.push(index3(x, y, z, quadtree.nx, ny));
    } else {
      for (let y = face.bounds.y0; y < face.bounds.y1 && y < ny; y += 1) for (let transverse = 0; transverse < face.bounds.span; transverse += 1) {
        const x = face.axis === 0 ? face.bounds.x : face.bounds.x + transverse;
        const z = face.axis === 2 ? face.bounds.z : face.bounds.z + transverse;
        if (x < quadtree.nx && z < quadtree.nz) subCells.push(index3(x, y, z, quadtree.nx, ny));
      }
    }
    const mapped = subCells.map((cell) => byKey.get(`${cell}:${face.axis}`)).filter((value) => value !== undefined) as number[];
    if (mapped.length !== subCells.length || mapped.length <= 1) continue;
    const solved = face.nodes.reduce((sum, node, slot) => {
      const dof = system.dofBySample[node];
      return dof < 0 ? sum : sum + face.coefficients[slot] * pressure[dof];
    }, 0);
    const average = mapped.reduce((sum, index) => sum + rows[index].entries.reduce((inner, [dof, weight]) => inner + weight * pressure[dof], 0), 0) / mapped.length;
    assert.ok(Math.abs(average - solved) < 1e-9, `face average ${average} must equal solved ${solved}`);
  }
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

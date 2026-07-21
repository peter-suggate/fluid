import assert from "node:assert/strict";
import test from "node:test";
import {
  FINE_LEVELSET_SAMPLE_FLAGS,
  FineLevelSetBrickOracle,
  packFineLevelSetBrickKey,
  planFineLevelSetBricks,
  unpackFineLevelSetBrickKey,
} from "../lib/octree-fine-levelset-bricks";
import {
  advectFineLevelSet,
  redistanceFineLevelSet,
  traceFineLevelSetDeparture,
} from "../lib/octree-fine-levelset-transport";

function oracle(fineFactor: 4 | 8 = 4) {
  const plan = planFineLevelSetBricks({
    domainOrigin: [0, 0, 0], finestCellDimensions: [2, 2, 2], finestCellWidth: 1,
    fineFactor, brickResolution: 4, maximumResidentBricks: 8,
  });
  const result = new FineLevelSetBrickOracle(plan);
  result.publishInterfaceAndRing([packFineLevelSetBrickKey(plan, [0, 0, 0])], ([x]) => x - 0.9);
  return result;
}

function fullyResidentOracle(fineFactor: 4 | 8, dimensions: [number, number, number],
  phi: (position: readonly [number, number, number]) => number) {
  const plan = planFineLevelSetBricks({
    domainOrigin: [0, 0, 0], finestCellDimensions: dimensions, finestCellWidth: 1,
    fineFactor, brickResolution: 4,
    maximumResidentBricks: dimensions.reduce((product, value) => product * value, 1)
      * (fineFactor / 4) ** 3,
  });
  const result = new FineLevelSetBrickOracle(plan);
  const keys: number[] = [];
  for (let z = 0; z < plan.brickDimensions[2]; z += 1) for (let y = 0; y < plan.brickDimensions[1]; y += 1) {
    for (let x = 0; x < plan.brickDimensions[0]; x += 1) keys.push(packFineLevelSetBrickKey(plan, [x, y, z]));
  }
  result.publishInterfaceAndRing(keys, phi);
  return result;
}

function forEachFineSample(field: FineLevelSetBrickOracle,
  visit: (position: [number, number, number], value: number, pageKey: number, localIndex: number) => void) {
  const r = field.plan.brickResolution;
  for (const page of field.residentPages()) {
    const brick = unpackFineLevelSetBrickKey(field.plan, page.key);
    for (let z = 0; z < r; z += 1) for (let y = 0; y < r; y += 1) for (let x = 0; x < r; x += 1) {
      const local = x + r * (y + r * z);
      const q = [brick[0] * r + x, brick[1] * r + y, brick[2] * r + z];
      const position = q.map((value, axis) => field.plan.domainOrigin[axis]
        + (value + 0.5) * field.plan.fineCellWidth) as [number, number, number];
      visit(position, page.phi[local], page.key, local);
    }
  }
}

test("piecewise trace resamples injected octree velocity once per fine-ratio segment", () => {
  let samples = 0;
  const trace = traceFineLevelSetDeparture([1, 0, 0], 1, 4, ([x]) => { samples += 1; return [x, 0, 0]; });
  assert.equal(samples, 4);
  assert.equal(trace.segments, 4);
  assert.ok(Math.abs(trace.departure[0] - 0.31640625) < 1e-12);
});

test("fine advection uses at least factor segments and commits atomically", () => {
  const field = oracle();
  const before = field.residentPages().map((page) => page.phi.slice());
  const stationary = advectFineLevelSet(field, 0.25, () => [0, 0, 0]);
  assert.equal(stationary.segmentsPerSample, 4);
  assert.equal(stationary.departureOutsideResidentBand, 0);
  assert.equal(stationary.committed, true);
  field.residentPages().forEach((page, index) => assert.deepEqual(page.phi, before[index]));
  assert.throws(() => advectFineLevelSet(field, 0.1, () => [0, 0, 0], 3), /fewer segments/);
});

test("factor-8 advection performs eight injected velocity samples per active fine sample", () => {
  const field = oracle(8);
  let velocityCalls = 0;
  const diagnostics = advectFineLevelSet(field, 0, () => { velocityCalls += 1; return [0, 0, 0]; });
  assert.equal(diagnostics.segmentsPerSample, 8);
  assert.equal(velocityCalls, diagnostics.samples * 8);
  assert.equal(diagnostics.committed, true);
});

test("departure outside the sparse band invalidates the update without coarse fallback or partial commit", () => {
  const field = oracle();
  const before = field.residentPages().map((page) => page.phi.slice());
  const diagnostics = advectFineLevelSet(field, 1, () => [100, 0, 0]);
  assert.ok(diagnostics.departureOutsideResidentBand > 0);
  assert.equal(diagnostics.committed, false);
  field.residentPages().forEach((page, index) => assert.deepEqual(page.phi, before[index]));
});

test("fine advection uses a constant half-cell ghost at solid domain walls", () => {
  const strictField = fullyResidentOracle(4, [2, 1, 1], ([x]) => x - 0.75);
  const strict = advectFineLevelSet(strictField, 0.25, () => [0.5, 0, 0]);
  assert.ok(strict.departureOutsideResidentBand > 0,
    "strict sparse sampling must not silently manufacture a wall ghost");
  const field = fullyResidentOracle(4, [2, 1, 1], ([x]) => x - 0.75);
  const diagnostics = advectFineLevelSet(field, 0.25, () => [0.5, 0, 0], 4, "closed-neumann");
  assert.equal(diagnostics.departureOutsideResidentBand, 0,
    "an in-domain backtrace between the wall and first sample centre must clamp to the boundary sample");
  assert.equal(diagnostics.committed, true);
});

test("uniform sparse redistance seeds subcell crossings, exchanges page boundaries, and preserves sign", () => {
  const field = oracle();
  const signs = field.residentPages().map((page) => Array.from(page.phi, (value) => value < 0));
  const diagnostics = redistanceFineLevelSet(field, {
    physicalBandWidth: 2,
    residualTolerance: 0.35,
    maximumSweeps: 64,
  });
  assert.ok(diagnostics.seeds > 0);
  assert.equal(diagnostics.converged, true);
  assert.equal(diagnostics.unresolvedCells, 0);
  assert.equal(diagnostics.committed, true);
  field.residentPages().forEach((page, pageIndex) => page.phi.forEach((value, sampleIndex) => {
    assert.equal(value < 0, signs[pageIndex][sampleIndex]);
    assert.ok(Number.isFinite(value));
  }));
});

test("redistance saturates topology guard pages outside its physical band", () => {
  // Twenty factor-4 bricks are deliberately much wider than the eight-sweep
  // budget. Only the two-cell signed-distance band needs propagation; the
  // remaining publication guard is a finite saturated narrow band.
  const field = fullyResidentOracle(4, [20, 1, 1], ([x]) => x - 0.5);
  const diagnostics = redistanceFineLevelSet(field, {
    physicalBandWidth: 0.5,
    residualTolerance: 1,
    maximumSweeps: 8,
  });
  assert.ok(diagnostics.seeds > 0);
  assert.equal(diagnostics.unresolvedCells, 0);
  assert.equal(diagnostics.converged, true);
  assert.equal(diagnostics.committed, true);
  forEachFineSample(field, ([x], value, pageKey, localIndex) => {
    assert.ok(Number.isFinite(value));
    const flags = field.pageForKey(pageKey)!.flags[localIndex];
    if (x >= 1) {
      assert.equal(value, 0.5);
      assert.equal(flags & FINE_LEVELSET_SAMPLE_FLAGS.valid, 0,
        "allocated guard samples outside the narrow band use coarse fallback");
    } else assert.notEqual(flags & FINE_LEVELSET_SAMPLE_FLAGS.valid, 0);
  });
});

for (const fineFactor of [4, 8] as const) test(`factor-${fineFactor} translating plane transports phi across pages before redistance`, () => {
  const initialPlane = 1.7;
  const displacement = 0.4;
  const field = fullyResidentOracle(fineFactor, [4, 2, 2], ([x]) => x - initialPlane);
  const generationBefore = field.generation;
  const brickWidth = field.plan.brickResolution * field.plan.fineCellWidth;
  const transport = advectFineLevelSet(field, displacement, ([x]) => [x >= 1 ? 1 : 0, 0, 0]);
  assert.equal(transport.committed, true);
  assert.equal(transport.departureOutsideResidentBand, 0);
  assert.equal(transport.segmentsPerSample, fineFactor);

  let checked = 0; let signChanges = 0; let crossPageSamples = 0;
  forEachFineSample(field, ([x], value) => {
    // Stay inside the constant-velocity part of the deliberately tapered
    // boundary field for the complete backtrace.
    if (x < 1.5 || x > 2.75) return;
    const expected = x - displacement - initialPlane;
    assert.ok(Math.abs(value - expected) < 2e-6,
      `factor ${fineFactor} transported plane mismatch at x=${x}: ${value} versus ${expected}`);
    assert.equal(value < 0, expected < 0);
    if ((x - initialPlane < 0) !== (expected < 0)) signChanges += 1;
    if (Math.floor(x / brickWidth) !== Math.floor((x - displacement) / brickWidth)) crossPageSamples += 1;
    checked += 1;
  });
  assert.ok(checked > 0 && signChanges > 0, "the payload must move rather than merely retain topology seeds");
  assert.ok(crossPageSamples > 0, "at least one trilinear departure must cross a brick boundary");

  const signs = new Map<string, boolean>();
  forEachFineSample(field, (position, value) => signs.set(position.join(","), value < 0));
  const redistance = redistanceFineLevelSet(field, {
    physicalBandWidth: 2, residualTolerance: 0.4, maximumSweeps: 96,
  });
  assert.equal(redistance.converged, true);
  assert.equal(redistance.unresolvedCells, 0);
  assert.equal(redistance.committed, true);
  forEachFineSample(field, (position, value) => {
    assert.equal(value < 0, signs.get(position.join(",")), "redistance changed a transported plane sign");
    assert.ok(Number.isFinite(value));
  });

  const beforeChurn = new Map(field.residentPages().map((page) => [page.key, page.phi.slice()]));
  const publication = field.publishInterfaceAndRing(field.detectInterfaceKeys(), ([x]) => x - initialPlane - displacement, 2);
  assert.equal(publication.generation, generationBefore + 1);
  assert.ok(publication.reusedPages > 0);
  for (const page of field.residentPages()) {
    const previous = beforeChurn.get(page.key);
    if (previous) assert.deepEqual(page.phi, previous, "retained generation pages must preserve transported phi");
  }
  const beforeReturnToA = new Map(field.residentPages().map((page) => [page.key, page.phi.slice()]));
  const returnToA = field.publishInterfaceAndRing(field.detectInterfaceKeys(), ([x]) => x - initialPlane - displacement, 2);
  assert.equal(returnToA.generation, generationBefore + 2);
  assert.deepEqual([generationBefore & 1, publication.generation & 1, returnToA.generation & 1], [1, 0, 1],
    "successive publications must exercise both A/B generation parities");
  for (const page of field.residentPages()) {
    const previous = beforeReturnToA.get(page.key);
    if (previous) assert.deepEqual(page.phi, previous, "A/B churn must not replace retained transported payloads");
  }
});

for (const fineFactor of [4, 8] as const) test(`factor-${fineFactor} rotating/deforming sphere preserves transported sign through redistance`, () => {
  const centre = [2, 2, 2] as const;
  const radius = 0.8;
  const timestep = 0.35;
  const sphere = ([x, y, z]: readonly [number, number, number]) =>
    Math.hypot(x - centre[0], y - centre[1], z - centre[2]) - radius;
  const velocity = ([x, y, z]: readonly [number, number, number]) => {
    const dx = x - centre[0], dy = y - centre[1], dz = z - centre[2];
    if (Math.hypot(dx, dy, dz) > 1.35) return [0, 0, 0] as [number, number, number];
    const angularSpeed = 0.7, strainRate = 0.65;
    return [-angularSpeed * dy + strainRate * dx,
      angularSpeed * dx - strainRate * dy, 0] as [number, number, number];
  };
  const field = fullyResidentOracle(fineFactor, [4, 4, 4], sphere);
  const before = new Map<string, boolean>();
  forEachFineSample(field, (position, value) => before.set(position.join(","), value < 0));
  const transport = advectFineLevelSet(field, timestep, velocity);
  assert.equal(transport.committed, true);
  assert.equal(transport.departureOutsideResidentBand, 0);
  assert.equal(transport.segmentsPerSample, fineFactor);

  const brickWidth = field.plan.brickResolution * field.plan.fineCellWidth;
  let compared = 0; let changedSigns = 0; let crossPageSamples = 0;
  forEachFineSample(field, (position, value) => {
    const distance = Math.hypot(position[0] - centre[0], position[1] - centre[1], position[2] - centre[2]);
    if (distance < 0.4 || distance > 1.2) return;
    const trace = traceFineLevelSetDeparture(position, timestep, fineFactor, velocity);
    assert.equal(trace.finite, true);
    const expected = sphere(trace.departure);
    // Trilinear interpolation of a curved signed-distance field has O(h^2)
    // value error.  Away from that narrow uncertainty interval, its phase
    // classification must match the analytically backtraced sphere.
    if (Math.abs(expected) > 1.25 * field.plan.fineCellWidth ** 2 / radius) {
      assert.equal(value < 0, expected < 0,
        `factor ${fineFactor} transported sphere sign mismatch at ${position.join(",")}`);
      compared += 1;
    }
    if (before.get(position.join(",")) !== (value < 0)) changedSigns += 1;
    if (position.some((coordinate, axis) => Math.floor(coordinate / brickWidth)
      !== Math.floor(trace.departure[axis] / brickWidth))) crossPageSamples += 1;
  });
  assert.ok(compared > 0 && changedSigns > 0, "strain must deform the sampled sphere phase field");
  assert.ok(crossPageSamples > 0, "rotation/deformation must cross resident brick boundaries");

  const transportedSigns = new Map<string, boolean>();
  forEachFineSample(field, (position, value) => transportedSigns.set(position.join(","), value < 0));
  const redistance = redistanceFineLevelSet(field, {
    physicalBandWidth: 1.5, residualTolerance: 0.45, maximumSweeps: 128,
  });
  assert.equal(redistance.converged, true);
  assert.equal(redistance.unresolvedCells, 0);
  assert.equal(redistance.committed, true);
  forEachFineSample(field, (position, value) => {
    assert.equal(value < 0, transportedSigns.get(position.join(",")), "redistance changed a transported sphere sign");
    assert.ok(Number.isFinite(value));
  });
});

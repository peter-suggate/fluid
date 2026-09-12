import assert from "node:assert/strict";
import test from "node:test";
import { productionSceneSliceSeedById } from "./production-scene-slice";
import { advanceSlice, createAdvanceSlice } from "./slice-solver";
import { createSlicePresentation, publishSlicePresentation, rdfAffineAt,
  reconstructSliceSharedRdf, type SlicePublishedRigidBody } from
  "./slice-presentation-publication";
import { compileSliceTopology } from "./slice-topology";

test("2D boundary interpolation preserves its observable oblique slope", () => {
  const direction = [1 / Math.sqrt(5), 2 / Math.sqrt(5)] as const;
  const samples = [-2, -1, 1, 2].map(parameter => ({
    point: [parameter * direction[0], parameter * direction[1]] as const,
    value: 4 + 2.5 * parameter,
  }));
  const parameter = 3;
  const value = rdfAffineAt(samples,
    [parameter * direction[0], parameter * direction[1]]);
  assert.ok(value !== null && Math.abs(value - (4 + 2.5 * parameter)) < 1e-12);
});

function halfPlaneFraction(minimum: readonly [number, number],
  maximum: readonly [number, number], normal: readonly [number, number],
  constant: number): number {
  const input: [number, number][] = [[minimum[0], minimum[1]], [maximum[0], minimum[1]],
    [maximum[0], maximum[1]], [minimum[0], maximum[1]]];
  const polygon: [number, number][] = [];
  for (let i = 0; i < input.length; i += 1) {
    const a = input[i]!, b = input[(i + 1) % input.length]!;
    const da = normal[0] * a[0] + normal[1] * a[1] - constant;
    const db = normal[0] * b[0] + normal[1] * b[1] - constant;
    if (da <= 0) polygon.push(a);
    if ((da < 0) !== (db < 0)) {
      const t = da / (da - db);
      polygon.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
  }
  let twiceArea = 0;
  for (let i = 0; i < polygon.length; i += 1) {
    const a = polygon[i]!, b = polygon[(i + 1) % polygon.length]!;
    twiceArea += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(twiceArea) / (2 * (maximum[0] - minimum[0])
    * (maximum[1] - minimum[1]));
}

test("native sparse-air support preserves several global oblique planes", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("cm12-figure-3"));
  for (const fixture of [
    { rawNormal: [-1, 8] as const, rawConstant: 124 },
    { rawNormal: [1, 16] as const, rawConstant: 248 },
    { rawNormal: [1, 32] as const, rawConstant: 496 },
  ]) {
    const length = Math.hypot(...fixture.rawNormal);
    const normal = [fixture.rawNormal[0] / length,
      fixture.rawNormal[1] / length] as const;
    const constant = fixture.rawConstant / length;
    const density = new Float32Array(slice.fields.density.length);
    const capacity = new Float32Array(slice.fields.capacity.length).fill(1);
    const interfaceNormal = new Float32Array(slice.fields.interfaceNormal.length);
    const interfaceOffset = new Float32Array(slice.fields.interfaceOffset.length);
    const partial = new Set<number>();
    for (const cell of slice.topology.accepted.cells) {
      const fill = halfPlaneFraction(cell.minimumFine, cell.maximumFine, normal, constant);
      density[cell.id] = Math.fround(fill);
      if (fill > 1e-6 && fill < 1 - 1e-6) {
        partial.add(cell.id);
        interfaceNormal[2 * cell.id] = Math.fround(normal[0]);
        interfaceNormal[2 * cell.id + 1] = Math.fround(normal[1]);
        interfaceOffset[cell.id] = Math.fround(constant
          - normal[0] * cell.centerFine[0] - normal[1] * cell.centerFine[1]);
      }
    }
    const result = reconstructSliceSharedRdf(slice.topology.accepted,
      { ...slice.fields, density, capacity, interfaceNormal, interfaceOffset,
        solidMotionActive: false }, slice.numericalTopology);
    assert.ok(result.receipt.inactiveAirGhostSamples > 0);
    const stride = result.dimensions[0] + 1;
    for (const id of partial) {
      const cell = slice.topology.accepted.cells[id]!;
      for (const x of [cell.minimumFine[0], cell.maximumFine[0]])
        for (const y of [cell.minimumFine[1], cell.maximumFine[1]]) {
          if (x === 0 || y === 0 || x === result.dimensions[0]
            || y === result.dimensions[1]) continue;
          const expected = normal[0] * x + normal[1] * y - constant;
          assert.ok(Math.abs(result.vertexPhiFine[x + stride * y]! - expected) < 1e-5,
            `oblique plane ${fixture.rawNormal} changed at ${x},${y}`);
        }
    }
  }
});

test("presentation publication preserves scene policy and accepted frame metrics", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("garden-hose"));
  const body: SlicePublishedRigidBody = Object.freeze({ id: "fixture",
    position_m: [1, 2, 3] as const, orientation: [0, 0, 0, 1] as const,
    linearVelocity_m_s: [4, 5, 6] as const, angularVelocity_rad_s: [7, 8, 9] as const,
    load_N: [10, 11, 12] as const, torque_Nm: [13, 14, 15] as const });
  const initial = createSlicePresentation(slice.scene, slice.topology.accepted,
    slice.fields, { rigidBodies: [body] });
  assert.equal(initial.cellSize, slice.scene.viewport.sourceCellSize);
  assert.equal(initial.hasSource, true);
  assert.equal(initial.receipt.columnPageCount, 0,
    "production disables the single-height publication in source scenes");
  assert.deepEqual(initial.receipt.rigidBodies, [body]);
  assert.ok(initial.receipt.liquidAreaFine > 0);
  assert.ok(initial.receipt.liquidCentroidFine);
  assert.ok(initial.receipt.liquidBoundsFine);

  const next = publishSlicePresentation(initial, slice.topology.accepted,
    slice.fields, slice.scene.sourceAtlas, { rigidBodies: [body] });
  assert.equal(next.receipt.accepted, true);
  assert.equal(next.state.publicationGeneration, 2);
  assert.equal(next.state.cellSize, initial.cellSize);
  assert.equal(next.state.hasSource, true);
  assert.equal(next.receipt.columnPageCount, 0);
  assert.deepEqual(next.receipt.rigidBodies, [body]);
});

function assertNoInteriorDanglingEndpoints(result: ReturnType<typeof reconstructSliceSharedRdf>): void {
  const counts = new Map<string, number>();
  for (let i = 0; i < result.segmentsFine.length; i += 4) for (const offset of [0, 2]) {
    const x = result.segmentsFine[i + offset]!, y = result.segmentsFine[i + offset + 1]!;
    const key = `${x.toFixed(5)}:${y.toFixed(5)}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const [nx, ny] = result.dimensions;
  for (const [key, count] of counts) {
    const [x, y] = key.split(":").map(Number);
    if (x! > 1e-5 && x! < nx - 1e-5 && y! > 1e-5 && y! < ny - 1e-5) {
      assert.equal(count % 2, 0, `interior RDF contour endpoint ${key} dangles`);
    }
  }
}

test("shared RDF removes sphere seams without replacing VOF authority", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-first-pool-impact-half"));
  for (let frame = 0; frame < 2; frame += 1) {
    const before = Float32Array.from(slice.fields.density);
    const result = reconstructSliceSharedRdf(slice.topology.accepted, slice.fields,
      slice.numericalTopology);
    assert.equal(result.receipt.unsupportedCutPartialCells, 0);
    assert.equal(result.receipt.ambiguousFineCells, 0);
    assert.equal(result.receipt.unresolvedFineCells, 0);
    assert.ok(result.receipt.interfaceCells >= 68);
    assert.ok(result.segmentsFine.length > 0);
    assert.ok(Math.abs(result.receipt.signedAreaErrorFine)
      / result.receipt.exactAreaFine < 0.002);
    assert.deepEqual(slice.fields.density, before,
      "the shared contour must be a mutation-free publication view");
    if (frame === 0) {
      const stride = result.dimensions[0] + 1;
      for (let x = 0; x <= result.dimensions[0]; x += 1)
        assert.equal(result.vertexPhiFine[x + stride * 16], 0,
          "the exact full/empty pool face must remain flat across AMR stencils");
    }
    assertNoInteriorDanglingEndpoints(result);
    advanceSlice(slice, { pressureIterations: 4 });
  }
});

test("shared RDF publication commutes with reflection after nonlinear transport", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-first-pool-impact-half"));
  for (let frame = 0; frame < 2; frame += 1) advanceSlice(slice, { pressureIterations: 4 });
  const topology = slice.topology.accepted, [nx, ny] = topology.dimensions;
  const byBounds = new Map(topology.cells.map(cell => [
    `${cell.minimumFine[0]}:${cell.minimumFine[1]}:${cell.maximumFine[0]}:${cell.maximumFine[1]}`,
    cell,
  ]));
  const reflectedScalar = (source: ArrayLike<number>) => {
    const result = new Float32Array(source.length);
    for (const cell of topology.cells) {
      const mirror = byBounds.get(`${nx - cell.maximumFine[0]}:${cell.minimumFine[1]}:${
        nx - cell.minimumFine[0]}:${cell.maximumFine[1]}`)!;
      assert.ok(mirror, `missing reflected owner for cell ${cell.id}`);
      result[mirror.id] = source[cell.id]!;
    }
    return result;
  };
  const reflectedVector = (source: ArrayLike<number>) => {
    const result = new Float32Array(source.length);
    for (const cell of topology.cells) {
      const mirror = byBounds.get(`${nx - cell.maximumFine[0]}:${cell.minimumFine[1]}:${
        nx - cell.minimumFine[0]}:${cell.maximumFine[1]}`)!;
      result[2 * mirror.id] = -source[2 * cell.id]!;
      result[2 * mirror.id + 1] = source[2 * cell.id + 1]!;
    }
    return result;
  };
  const reflectedFields = { ...slice.fields,
    capacity: reflectedScalar(slice.fields.capacity),
    density: reflectedScalar(slice.fields.density),
    cellVelocity: reflectedVector(slice.fields.cellVelocity),
    interfaceNormal: reflectedVector(slice.fields.interfaceNormal),
    interfaceOffset: reflectedScalar(slice.fields.interfaceOffset) };
  const original = reconstructSliceSharedRdf(topology, slice.fields, slice.numericalTopology);
  const reflected = reconstructSliceSharedRdf(topology, reflectedFields, slice.numericalTopology);
  const stride = nx + 1;
  let maximumPhiError = 0;
  for (let y = 0; y <= ny; y += 1) for (let x = 0; x <= nx; x += 1) {
    const a = original.vertexPhiFine[x + stride * y]!;
    const b = reflected.vertexPhiFine[nx - x + stride * y]!;
    if (Number.isFinite(a) && Number.isFinite(b)) {
      maximumPhiError = Math.max(maximumPhiError, Math.abs(a - b));
    } else assert.equal(Number.isFinite(a), Number.isFinite(b));
  }
  assert.ok(maximumPhiError <= 4 * 1.1920928955078125e-7,
    `reflected shared RDF differs by ${maximumPhiError}`);
  assert.ok(Math.abs(original.receipt.representedAreaFine
    - reflected.receipt.representedAreaFine) <= 1e-9);
  assert.equal(original.segmentsFine.length, reflected.segmentsFine.length);
});

test("impact RDF cannot turn accepted bulk liquid into an air cell", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-first-pool-impact-half"));
  // The roundoff-consistent ELVIRA path reaches the same transported
  // near-full coverage one frame later than the former asymmetric branch.
  for (let frame = 0; frame < 5; frame += 1) advanceSlice(slice, { pressureIterations: 4 });
  const result = reconstructSliceSharedRdf(slice.topology.accepted, slice.fields,
    slice.numericalTopology);
  const stride = result.dimensions[0] + 1;
  let nearFullCells = 0;
  for (const cell of slice.topology.accepted.cells) {
    const capacity = slice.fields.capacity[cell.id]!;
    if (!(capacity > 1e-8)) continue;
    const fill = slice.fields.density[cell.id]! / capacity;
    if (fill > 0.99 && fill < 1 - 1e-6) nearFullCells += 1;
    for (let y = cell.minimumFine[1]; y < cell.maximumFine[1]; y += 1)
      for (let x = cell.minimumFine[0]; x < cell.maximumFine[0]; x += 1) {
        const phi = [result.vertexPhiFine[x + stride * y]!,
          result.vertexPhiFine[x + 1 + stride * y]!,
          result.vertexPhiFine[x + stride * (y + 1)]!,
          result.vertexPhiFine[x + 1 + stride * (y + 1)]!];
        if (!phi.every(Number.isFinite)) continue;
        if (fill >= 1 - 1e-6) assert.ok(!phi.every(value => value >= 0),
          `full fine cell ${x},${y} was reconstructed entirely as air`);
        if (fill <= 1e-6) assert.ok(!phi.every(value => value <= 0),
          `empty fine cell ${x},${y} was reconstructed entirely as liquid`);
        if (!(fill > 0.99 && fill < 1 - 1e-6)) continue;
        const nx = slice.fields.interfaceNormal[2 * cell.id]!;
        const ny = slice.fields.interfaceNormal[2 * cell.id + 1]!;
        const accepted = nx * (x + 0.5 - cell.centerFine[0])
          + ny * (y + 0.5 - cell.centerFine[1])
          - slice.fields.interfaceOffset[cell.id]!;
        if (accepted < 0) assert.ok(!phi.every(value => value >= 0),
          `near-full PLIC liquid at ${x},${y} was reconstructed entirely as air`);
      }
  }
  assert.ok(nearFullCells >= 50, "the regression frame must exercise transported near-full cells");
  assert.ok(Math.abs(result.receipt.signedAreaErrorFine) / result.receipt.exactAreaFine < 0.002,
    "phase consistency must not trade the hole for a global area shift");
});

test("shared RDF handles disconnected AMR interfaces and reports cut-cell limits", () => {
  const drops = createAdvanceSlice(productionSceneSliceSeedById("cm12-figure-3"));
  const [width, height] = drops.topology.accepted.dimensions;
  let outsideSolidQueries = 0;
  const sourceFraction = drops.numericalTopology.solidVoxelFractionAt;
  const sourceSolid = drops.numericalTopology.solidVoxelAt;
  const result = reconstructSliceSharedRdf(drops.topology.accepted, drops.fields, {
    solidVoxelFractionAt: (x, y) => {
      if (x < 0 || y < 0 || x >= width || y >= height) outsideSolidQueries += 1;
      return sourceFraction?.(x, y) ?? (sourceSolid?.(x, y) ? 1 : 0);
    },
  });
  assert.equal(outsideSolidQueries, 0,
    "physical-domain boundaries must not create or query reflected air support");
  assert.equal(result.receipt.inactiveAirGhostSamples, 52,
    "native inactive sparse rungs complete the rank-deficient air stencil");
  assert.equal(result.receipt.reflectedAirGhostSamples, 0,
    "allocated inactive pages take precedence over reflected fallback geometry");
  assert.equal(result.receipt.ambiguousFineCells, 0);
  assert.equal(result.receipt.unresolvedFineCells, 0);
  assert.ok(result.receipt.interfaceCells >= 150);
  assertNoInteriorDanglingEndpoints(result);

  const activeOnly = compileSliceTopology(drops.topology.accepted.bricks
    .filter(brick => brick.active !== false), drops.topology.accepted.dimensions,
  drops.topology.accepted.generation, 0.5, drops.topology.accepted.boundaryModes);
  const reflected = reconstructSliceSharedRdf(activeOnly, drops.fields,
    drops.numericalTopology);
  assert.equal(reflected.receipt.inactiveAirGhostSamples, 0);
  assert.equal(reflected.receipt.reflectedAirGhostSamples, 60,
    "static open geometry supplies a deterministic face-connected fallback when no inactive rung exists");
  assert.equal(reflected.receipt.unresolvedFineCells, 0);

  const solidSuppressed = reconstructSliceSharedRdf(drops.topology.accepted,
    drops.fields, { solidVoxelFractionAt: () => 1 });
  assert.equal(solidSuppressed.receipt.inactiveAirGhostSamples, 0,
    "a missing numerical owner cannot be treated as air through static solid geometry");
  assert.equal(solidSuppressed.receipt.reflectedAirGhostSamples, 0);
  assert.ok(solidSuppressed.receipt.unresolvedFineCells > 0,
    "unsupported solid-side vertices must remain visible in the receipt");

  const cut = createAdvanceSlice(productionSceneSliceSeedById("garden-pond"));
  const cutResult = reconstructSliceSharedRdf(cut.topology.accepted, cut.fields,
    cut.numericalTopology);
  assert.equal(cutResult.receipt.unsupportedCutPartialCells, 2,
    "immersed partial liquid cells remain an explicit preview limitation");

  assert.ok(Math.abs(result.receipt.signedAreaErrorFine)
    / result.receipt.exactAreaFine < 0.002,
  "coarse-cell RDF prolongation must not shift the pool by a fine cell");
});

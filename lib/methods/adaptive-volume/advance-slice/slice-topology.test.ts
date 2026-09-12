import assert from "node:assert/strict";
import test from "node:test";
import { buildSparseAtlasCompositeGrid,
  type SparseAtlasGradientRow } from "../sparse-atlas-composite-projection";
import {
  createSparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
} from "../sparse-brick-atlas";
import {
  SLICE_TOPOLOGY_INVALID,
  SLICE_TOPOLOGY_NEW_AIR,
  SliceTopologyGenerationCapacityDeferred,
  cancelSliceTopologyCandidate,
  commitSliceTopologyCandidate,
  compileSliceTopology,
  createSliceTopologyAuthority,
  stageSliceTopologyCandidate,
  transferSliceTopology,
  transferSliceTopologyFields,
  type SliceTopology,
  type SliceTopologyBrick,
} from "./slice-topology";

const brick = (id: number, x: number, resolution: 1 | 2 | 4 | 8,
  active = true): SliceTopologyBrick => ({ id, key: id, coordinate: [x, 0],
    resolution, active,
    density: new Float32Array(resolution ** 2).fill(0.25 + 0.1 * id),
    gamma: new Float32Array(resolution ** 2).fill(1 + 0.2 * id) });

const mixed = (left = 8 as 4 | 8, right = 4 as 4 | 8) =>
  compileSliceTopology([brick(0, 0, left), brick(1, 1, right)], [16, 8]);

test("B8/B4 pressure ports and physical subfaces use production coefficients", () => {
  const topology = mixed();
  assert.equal(topology.cells.length, 8 ** 2 + 4 ** 2);
  assert.equal(topology.mixedSeamRowCount, 4);
  const rows = topology.rows.filter(row => row.kind === "mixed-seam");
  for (const row of rows) {
    assert.equal(row.axis, 0);
    assert.equal(row.areaFineCells, 2);
    assert.equal(row.centerDistanceFine, 1.5);
    assert.equal(row.dualWeight, 3);
    assert.equal(row.terms.length, 3);
    const negative = row.terms.filter(term => term.coefficient < 0);
    const positive = row.terms.filter(term => term.coefficient > 0);
    assert.equal(negative.length, 2);
    assert.equal(positive.length, 1);
    assert.ok(negative.every(term => Math.abs(term.coefficient + 1 / 3) < 3e-8));
    assert.ok(Math.abs(positive[0]!.coefficient - 2 / 3) < 3e-8);
    assert.ok(Math.abs(row.terms.reduce((sum, term) => sum + term.coefficient, 0)) < 6e-8);
    const subfaces = topology.subfaces.filter(face => face.row === row.id);
    assert.equal(subfaces.length, 2);
    assert.ok(subfaces.every(face => face.areaFineCells === 1
      && face.negativeCell !== SLICE_TOPOLOGY_INVALID
      && face.positiveCell !== SLICE_TOPOLOGY_INVALID));
  }
});

test("the 2-D mixed port is the unit-depth reduction of production HTP geometry", () => {
  const density8 = new Float64Array(8 ** 3).fill(0.25);
  const density4 = new Float64Array(4 ** 3).fill(0.35);
  const productionBricks: SparseAdaptiveMassBrick[] = [
    { key: 0, coordinate: [0, 0, 0], resolution: 8,
      density: density8, gamma: new Float64Array(8 ** 3).fill(1) },
    { key: 1, coordinate: [1, 0, 0], resolution: 4,
      density: density4, gamma: new Float64Array(4 ** 3).fill(1.2) },
  ];
  const production = buildSparseAtlasCompositeGrid(
    createSparseAdaptiveMassAtlas([16, 8, 8], productionBricks));
  const slice = mixed();
  const sliceRows = new Map(slice.rows.filter(row => row.kind === "mixed-seam")
    .map(row => [row.centerFine[1], row]));
  const productionRows = production.gradientRows.filter(row => row.kind === "mixed-seam");
  assert.equal(productionRows.length, 16);
  for (const row of productionRows) {
    const reduced = sliceRows.get(row.centerFine[1]);
    assert.ok(reduced, `missing reduced port at y=${row.centerFine[1]}`);
    // A 3-D coarse port is two finest cells deep here. Dividing its face and
    // dual measures by that extrusion produces the exact 2-D records.
    assert.equal(row.areaFineCells2 / 2, reduced.areaFineCells);
    assert.equal(row.centerDistanceFine, reduced.centerDistanceFine);
    assert.equal(row.dualWeight / 2, reduced.dualWeight);
    const collapsed = new Map<string, number>();
    for (const term of row.terms) {
      const cell = production.cells[term.cellId]!;
      const key = `${cell.brickKey}/${cell.local[0]}/${cell.local[1]}`;
      collapsed.set(key, (collapsed.get(key) ?? 0) + term.coefficient);
    }
    for (const term of reduced.terms) {
      const cell = slice.cells[term.cellId]!;
      const key = `${cell.brickKey}/${cell.local[0]}/${cell.local[1]}`;
      assert.ok(Math.abs((collapsed.get(key) ?? NaN) - term.coefficient) < 3e-8,
        `${key}: ${collapsed.get(key)} versus ${term.coefficient}`);
    }
  }
});

test("the complete X/Y row stream keeps production order and unit-depth algebra", () => {
  const productionBricks: SparseAdaptiveMassBrick[] = [
    { key: 0, coordinate: [0, 0, 0], resolution: 8,
      density: new Float64Array(8 ** 3), gamma: new Float64Array(8 ** 3).fill(1) },
    { key: 1, coordinate: [1, 0, 0], resolution: 4,
      density: new Float64Array(4 ** 3), gamma: new Float64Array(4 ** 3).fill(1) },
  ];
  const production = buildSparseAtlasCompositeGrid(
    createSparseAdaptiveMassAtlas([16, 8, 8], productionBricks));
  const slice = mixed();
  const z = 4.5;
  const signature3d = (row: SparseAtlasGradientRow) => {
    const rowCells = row.terms.map(term => production.cells[term.cellId]!);
    const lowerZ = Math.min(...rowCells.map(cell => cell.minimumFine[2]));
    const upperZ = Math.max(...rowCells.map(cell => cell.maximumFine[2]));
    if (row.axis === 2 || z < lowerZ || z >= upperZ) return undefined;
    const depth = upperZ - lowerZ;
    const terms = new Map<string, number>();
    for (const term of row.terms) {
      const cell = production.cells[term.cellId]!;
      const key = `${cell.brickKey}/${cell.local[0]}/${cell.local[1]}`;
      terms.set(key, (terms.get(key) ?? 0) + term.coefficient);
    }
    return {
      kind: row.kind, axis: row.axis, center: row.centerFine.slice(0, 2),
      area: row.areaFineCells2 / depth, distance: row.centerDistanceFine,
      dual: row.dualWeight / depth,
      terms: [...terms].sort(([a], [b]) => a.localeCompare(b))
        .map(([key, value]) => [key, Math.fround(value)]),
    };
  };
  const expected = production.gradientRows.map(signature3d).filter(value => value !== undefined);
  const actual = slice.rows.map(row => ({
    kind: row.kind, axis: row.axis, center: [...row.centerFine],
    area: row.areaFineCells, distance: row.centerDistanceFine, dual: row.dualWeight,
    terms: row.terms.map(term => {
      const cell = slice.cells[term.cellId]!;
      return [`${cell.brickKey}/${cell.local[0]}/${cell.local[1]}`, term.coefficient] as const;
    }).sort(([a], [b]) => a.localeCompare(b)),
  }));
  assert.deepEqual(actual, expected);
});

test("cell incidence follows production template owner buckets before row id", () => {
  const topology = mixed();
  const coarse = topology.cells.find(cell => cell.brickKey === 1
    && cell.local[0] === 0 && cell.local[1] === 0)!;
  const begin = topology.incidenceOffsets[coarse.id]!;
  const end = topology.incidenceOffsets[coarse.id + 1]!;
  const incident = topology.incidences.slice(begin, end).map(value => topology.rows[value.row]!);
  const owners = incident.map(row => Math.min(...row.terms.map(term => {
    const cell = topology.cells[term.cellId]!;
    return topology.bricks.findIndex(brick => brick.key === cell.brickKey);
  })));
  for (let at = 1; at < owners.length; at++) {
    assert.ok(owners[at - 1]! <= owners[at]!,
      `owner ${owners[at - 1]} must precede ${owners[at]}`);
  }
  const ownRows = incident.filter((_, at) => owners[at] === 1).map(row => row.id);
  assert.deepEqual(ownRows, [...ownRows].sort((a, b) => a - b),
    "source row order must remain stable inside an owner/rung bucket");
});

function gradient(topology: SliceTopology, values: ArrayLike<number>): Float64Array {
  return Float64Array.from(topology.rows, row => row.terms.reduce((sum, term) =>
    sum + term.coefficient * values[term.cellId]!, 0));
}

function divergence(topology: SliceTopology, velocity: ArrayLike<number>): Float64Array {
  const result = new Float64Array(topology.cells.length);
  for (const row of topology.rows) for (const term of row.terms) {
    result[term.cellId] -= row.dualWeight * term.coefficient * velocity[row.id]!
      / topology.cells[term.cellId]!.volumeFineCells;
  }
  return result;
}

test("the same adaptive rows make gradient and divergence negative transposes", () => {
  for (const topology of [mixed(8, 4), mixed(4, 8)]) {
    const pressure = Float64Array.from(topology.cells, cell =>
      Math.sin(0.37 * (cell.stableLeafId + 1)));
    const velocity = Float64Array.from(topology.rows, row =>
      Math.cos(0.23 * (row.id + 1)));
    const g = gradient(topology, pressure), d = divergence(topology, velocity);
    const facePairing = topology.rows.reduce((sum, row) =>
      sum + row.dualWeight * g[row.id]! * velocity[row.id]!, 0);
    const cellPairing = topology.cells.reduce((sum, cell) =>
      sum + cell.volumeFineCells * pressure[cell.id]! * d[cell.id]!, 0);
    assert.ok(Math.abs(facePairing + cellPairing) < 2e-12,
      `${facePairing} + ${cellPairing}`);
  }
});

test("candidate lifecycle is contiguous and transfer covers split and merge", () => {
  const coarse = compileSliceTopology([brick(0, 0, 4), brick(1, 1, 4)], [16, 8], 7);
  const refinedBricks = [brick(0, 0, 8), brick(1, 1, 4)];
  let authority = createSliceTopologyAuthority(coarse);
  authority = stageSliceTopologyCandidate(authority, refinedBricks);
  assert.equal(authority.accepted.generation, 7);
  assert.equal(authority.candidate?.generation, 8);
  assert.throws(() => stageSliceTopologyCandidate(authority, refinedBricks), /already has/);
  const forward = transferSliceTopology(authority.accepted, authority.candidate!);
  for (const cell of authority.candidate!.cells) {
    let area = 0;
    for (let at = forward.cellOffsets[cell.id]!; at < forward.cellOffsets[cell.id + 1]!; at++) {
      area += forward.cellAreas[at]!;
    }
    assert.ok(Math.abs(area - cell.volumeFineCells) < 1e-6);
  }
  authority = commitSliceTopologyCandidate(authority);
  assert.equal(authority.accepted.generation, 8);
  authority = stageSliceTopologyCandidate(authority, [brick(0, 0, 4), brick(1, 1, 4)]);
  const reverse = transferSliceTopology(authority.accepted, authority.candidate!);
  for (const cell of authority.candidate!.cells) {
    let area = 0;
    for (let at = reverse.cellOffsets[cell.id]!; at < reverse.cellOffsets[cell.id + 1]!; at++) {
      area += reverse.cellAreas[at]!;
    }
    assert.ok(Math.abs(area - cell.volumeFineCells) < 1e-6);
  }
  authority = cancelSliceTopologyCandidate(authority);
  assert.equal(authority.accepted.generation, 8);
  assert.equal(authority.candidate, undefined);
});

test("partial edge cells, inactive leaves and physical boundary modes stay explicit", () => {
  const topology = compileSliceTopology([
    brick(0, 0, 4),
    { ...brick(1, 1, 4, false), coordinate: [1, 0] },
    { ...brick(2, 2, 4), coordinate: [2, 0] },
  ], [21, 7], 1, 0.5, {
    negativeX: "closed", positiveX: "open", negativeY: "closed", positiveY: "open",
  });
  assert.equal(topology.brickByKey.size, 3);
  assert.ok(topology.cells.every(cell => cell.brickKey !== 1));
  assert.ok(topology.cells.some(cell => cell.widthsFine[0] === 1));
  assert.ok(topology.cells.some(cell => cell.widthsFine[1] === 1));
  assert.ok(topology.rows.some(row => row.boundaryMode === "closed"));
  assert.ok(topology.rows.some(row => row.boundaryMode === "open"));
  assert.ok(topology.rows.some(row => row.kind === "sparse-air"
    && row.boundaryMode === undefined), "inactive coverage must remain a sparse-air frontier");
  const incidenceCount = topology.rows.reduce((sum, row) => sum + row.terms.length, 0);
  assert.equal(topology.incidences.length, incidenceCount);
  assert.equal(topology.incidenceOffsets.at(-1), incidenceCount);
});

function transferFields(topology: SliceTopology, values?: Partial<{
  density: readonly number[]; gamma: readonly number[]; pressure: readonly number[];
  cellVelocity: readonly number[]; faceVelocity: readonly number[];
  capacity: readonly number[]; interfaceNormal: readonly number[];
}>) {
  const cells = topology.cells.length, rows = topology.rows.length;
  return {
    density: Float32Array.from(values?.density ?? new Array(cells).fill(0)),
    gamma: Float32Array.from(values?.gamma ?? new Array(cells).fill(1)),
    pressure: Float32Array.from(values?.pressure ?? new Array(cells).fill(0)),
    cellVelocity: Float32Array.from(values?.cellVelocity ?? new Array(2 * cells).fill(0)),
    faceVelocity: Float32Array.from(values?.faceVelocity ?? new Array(rows).fill(0)),
    capacity: Float32Array.from(values?.capacity ?? new Array(cells).fill(1)),
    interfaceNormal: values?.interfaceNormal
      ? Float32Array.from(values.interfaceNormal) : undefined,
  };
}

test("source-owned PLIC split uses accepted plane and invalidates its candidate cache", () => {
  const source = compileSliceTopology([brick(0, 0, 1)], [8, 8], 1);
  const target = compileSliceTopology([brick(0, 0, 2)], [8, 8], 2);
  const fields = transferFields(source, {
    density: [0.5], gamma: [1.25], pressure: [3], cellVelocity: [2, -1],
    faceVelocity: source.rows.map(row => 10 + row.id), capacity: [1],
    interfaceNormal: [1, 0],
  });
  const moved = transferSliceTopologyFields(source, target, fields,
    new Float32Array(target.cells.length).fill(1));
  for (const cell of target.cells) {
    assert.equal(moved.density[cell.id], cell.local[0] === 0 ? 1 : 0);
    assert.equal(moved.gamma[cell.id], 1.25);
    assert.equal(moved.pressure[cell.id], 3);
    assert.deepEqual([...moved.cellVelocity.slice(2 * cell.id, 2 * cell.id + 2)], [2, -1]);
  }
  assert.deepEqual([...moved.interfaceNormal], new Array(2 * target.cells.length).fill(0));
  assert.deepEqual([...moved.interfaceOffset], new Array(target.cells.length).fill(0));
  const internalX = target.rows.find(row => row.axis === 0 && row.centerFine[0] === 4);
  assert.ok(internalX);
  assert.equal(moved.plan.faceOffsets[internalX.id], moved.plan.faceOffsets[internalX.id + 1]);
  assert.equal(moved.faceVelocity[internalX.id], 2,
    "new face authority must use target collocated velocity");
});

test("cut-cell split deliberately uses production capacity fallback instead of PLIC", () => {
  const source = compileSliceTopology([brick(0, 0, 1)], [8, 8], 1);
  const target = compileSliceTopology([brick(0, 0, 2)], [8, 8], 2);
  const fields = transferFields(source, {
    density: [0.25], capacity: [0.5], interfaceNormal: [1, 0],
  });
  const moved = transferSliceTopologyFields(source, target, fields,
    new Float32Array(target.cells.length).fill(0.5));
  assert.deepEqual([...moved.density], [0.25, 0.25, 0.25, 0.25]);
  assert.deepEqual([...moved.targetAmounts], [4, 4, 4, 4]);
  assert.equal(moved.sourceAmounts[0], 16);
  assert.equal(moved.sourceCapacities[0], 32);
});

test("coarsening transports mass/momentum but volume-weights gamma and pressure", () => {
  const source = compileSliceTopology([brick(0, 0, 2)], [8, 8], 1);
  const target = compileSliceTopology([brick(0, 0, 1)], [8, 8], 2);
  const moved = transferSliceTopologyFields(source, target, transferFields(source, {
    density: [1, 0, 0.5, 0], gamma: [1, 2, 3, 4], pressure: [2, 4, 6, 8],
    cellVelocity: [1, 10, 20, 200, 3, 30, 40, 400],
  }), new Float32Array([1]));
  assert.equal(moved.density[0], 0.375);
  assert.equal(moved.gamma[0], 2.5);
  assert.equal(moved.pressure[0], 5);
  assert.deepEqual([...moved.cellVelocity], [5 / 3, 50 / 3].map(Math.fround));
});

test("new-air coverage is explicit and initializes production gamma-one air", () => {
  const source = compileSliceTopology([brick(0, 0, 1)], [16, 8], 1);
  const target = compileSliceTopology([brick(0, 0, 1), brick(1, 1, 1)], [16, 8], 2);
  assert.throws(() => transferSliceTopology(source, target), /complete accepted coverage/);
  const coverage = [{ minimumFine: [8, 0] as const,
    maximumExclusiveFine: [16, 8] as const }];
  const moved = transferSliceTopologyFields(source, target, transferFields(source, {
    density: [0.5], gamma: [0.75], pressure: [2], cellVelocity: [4, 5],
  }), new Float32Array([1, 1]), coverage);
  const second = moved.plan.cellOffsets[1]!;
  assert.equal(moved.plan.cellSources[second], SLICE_TOPOLOGY_NEW_AIR);
  assert.equal(moved.density[1], 0);
  assert.equal(moved.gamma[1], 1);
  assert.equal(moved.pressure[1], 0);
  assert.deepEqual([...moved.cellVelocity.slice(2, 4)], [0, 0]);
});

test("generation transfer preserves accepted f32 endpoint tolerance and defers overflow", () => {
  const topology = compileSliceTopology([brick(0, 0, 1)], [8, 8], 1);
  const epsilon8 = Math.fround(8 * 1.1920928955078125e-7);
  const accepted = transferFields(topology, { density: [1 + epsilon8], capacity: [1] });
  const moved = transferSliceTopologyFields(topology, topology, accepted, new Float32Array([1]));
  assert.equal(moved.density[0], Math.fround(1 + epsilon8));
  const rejected = transferFields(topology, {
    density: [1 + Math.fround(16 * 1.1920928955078125e-7)], capacity: [1],
  });
  assert.throws(() => transferSliceTopologyFields(topology, topology, rejected,
    new Float32Array([1])), SliceTopologyGenerationCapacityDeferred);
});

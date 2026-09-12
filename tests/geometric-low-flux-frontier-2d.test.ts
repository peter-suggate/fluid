import assert from "node:assert/strict";
import test from "node:test";
import { productionSceneSliceSeedById } from
  "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { createAdvanceSlice } from
  "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import {
  lowFluxLimiterExecution2D,
  solveStaticLowFluxDenseCopy2D,
  solveStaticLowFluxDensePingPong2D,
  solveStaticLowFluxFrontier2D,
  staticLowFluxProblem2D,
  type StaticLowFluxProblem2D,
  type StaticLowFluxResult2D,
} from "../lib/core/geometric-low-flux-frontier/reference-2d";

function words(values: Float32Array): number[] {
  return [...new Uint32Array(values.buffer, values.byteOffset, values.length)];
}

function comparablePasses(result: StaticLowFluxResult2D) {
  return result.passReceipts.map(({ pass, invalidCount, firstInvalid, changedCells }) =>
    ({ pass, invalidCount, firstInvalid, changedCells }));
}

function assertExact(problem: StaticLowFluxProblem2D): readonly [StaticLowFluxResult2D,
  StaticLowFluxResult2D, StaticLowFluxResult2D] {
  const copy = solveStaticLowFluxDenseCopy2D(problem);
  const pingPong = solveStaticLowFluxDensePingPong2D(problem);
  const frontier = solveStaticLowFluxFrontier2D(problem);
  for (const candidate of [pingPong, frontier]) {
    assert.equal(candidate.converged, copy.converged);
    assert.equal(candidate.passes, copy.passes);
    assert.deepEqual(comparablePasses(candidate), comparablePasses(copy));
    assert.deepEqual(words(candidate.factors), words(copy.factors));
    assert.deepEqual(words(candidate.lowFlux), words(copy.lowFlux));
    assert.deepEqual(words(candidate.lowStateVolume), words(copy.lowStateVolume));
  }
  return [copy, pingPong, frontier];
}

function assertBounds(problem: StaticLowFluxProblem2D, result: StaticLowFluxResult2D): void {
  for (let cell = 0; cell < problem.capacities.length; cell += 1) {
    const capacity = problem.capacities[cell]!;
    const margin = Math.fround(Math.fround(9.5367431640625e-7) * Math.fround(capacity));
    assert.ok(result.lowStateVolume[cell]! >= -margin,
      `cell ${cell} below its lower volume bound`);
    assert.ok(result.lowStateVolume[cell]! <= Math.fround(capacity + margin),
      `cell ${cell} above its upper volume bound`);
  }
}

test("ping-pong and frontier preserve a valid closed through-flow in one pass", () => {
  const problem = staticLowFluxProblem2D({ volumes: [1, 1], capacities: [1, 1],
    faces: [
      { negativeCell: 0, positiveCell: 1, lowFlux: 0.25 },
      { negativeCell: 0, positiveCell: 1, lowFlux: -0.25 },
    ] });
  const [copy, pingPong, frontier] = assertExact(problem);
  assert.equal(copy.passes, 1);
  assert.equal(copy.updateCellVisits, 2);
  assert.equal(pingPong.bankCopyCellVisits, 0);
  assert.equal(frontier.updateCellVisits, 2);
  assertBounds(problem, frontier);
});

test("frontier follows a receiver-factor change backward through a directed chain", () => {
  const problem = staticLowFluxProblem2D({
    volumes: [0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    capacities: [1, 1, 1, 1, 1, 1],
    faces: Array.from({ length: 5 }, (_, cell) =>
      ({ negativeCell: cell, positiveCell: cell + 1, lowFlux: 0.2 })),
  });
  const [copy, , frontier] = assertExact(problem);
  assert.ok(copy.passes > 2);
  assert.ok(frontier.passReceipts.some(pass => pass.evaluatedCells.length < 6));
  assert.ok(frontier.updateCellVisits < copy.updateCellVisits);
  assertBounds(problem, frontier);
});

test("frontier remains exact through a forty-pass reverse dependency wave", () => {
  const cells = 64;
  const problem = staticLowFluxProblem2D({
    volumes: new Array<number>(cells).fill(0.995),
    capacities: new Array<number>(cells).fill(1),
    faces: Array.from({ length: cells - 1 }, (_, cell) =>
      ({ negativeCell: cell, positiveCell: cell + 1, lowFlux: 0.2 })),
  });
  const [copy, , frontier] = assertExact(problem);
  assert.equal(copy.passes, 40);
  assert.deepEqual(frontier.passReceipts.map(pass => pass.evaluatedCells.length),
    [64, ...new Array<number>(39).fill(2)]);
  assert.equal(copy.updateCellVisits, 2_560);
  assert.equal(frontier.updateCellVisits, 142);
  assertBounds(problem, frontier);
});

test("branch, cycle, exterior and source terms retain dense pass semantics", () => {
  const problem = staticLowFluxProblem2D({
    volumes: [0.8, 0.85, 0.9, 0.75, 0.9], capacities: [1, 1, 1, 1, 1],
    sourceRates: [0, 0.02, 0, 0, 0], dt: 0.5,
    faces: [
      { negativeCell: 0, positiveCell: 1, lowFlux: 0.23 },
      { negativeCell: 1, positiveCell: 2, lowFlux: 0.19 },
      { negativeCell: 1, positiveCell: 3, lowFlux: 0.17 },
      { negativeCell: 3, positiveCell: 4, lowFlux: 0.21 },
      { negativeCell: 4, positiveCell: 1, lowFlux: 0.07 },
      { negativeCell: -1, positiveCell: 0, lowFlux: 0 },
      { negativeCell: 2, positiveCell: -1, lowFlux: 0.04 },
    ],
  });
  const [copy, , frontier] = assertExact(problem);
  assert.equal(copy.converged, true);
  assert.ok(frontier.updateCellVisits <= copy.updateCellVisits);
  assertBounds(problem, frontier);
});

test("static partial capacity stays bounded without changing the dependency proof", () => {
  const problem = staticLowFluxProblem2D({
    volumes: [0.2, 0.48, 0.7, 0.15], capacities: [0.25, 0.5, 0.8, 0.2],
    faces: [
      { negativeCell: 0, positiveCell: 1, lowFlux: 0.08 },
      { negativeCell: 1, positiveCell: 2, lowFlux: 0.11 },
      { negativeCell: 2, positiveCell: 3, lowFlux: 0.13 },
    ],
  });
  const [, , frontier] = assertExact(problem);
  assertBounds(problem, frontier);
});

test("a reflected dependency graph retains exact factors after unpermuting", () => {
  const forward = staticLowFluxProblem2D({ volumes: [0.95, 0.95, 0.95, 0.95],
    capacities: [1, 1, 1, 1], faces: [
      { negativeCell: 0, positiveCell: 1, lowFlux: 0.2 },
      { negativeCell: 1, positiveCell: 2, lowFlux: 0.2 },
      { negativeCell: 2, positiveCell: 3, lowFlux: 0.2 },
    ] });
  const reflected = staticLowFluxProblem2D({ volumes: [0.95, 0.95, 0.95, 0.95],
    capacities: [1, 1, 1, 1], faces: [
      { negativeCell: 2, positiveCell: 3, lowFlux: -0.2 },
      { negativeCell: 1, positiveCell: 2, lowFlux: -0.2 },
      { negativeCell: 0, positiveCell: 1, lowFlux: -0.2 },
    ] });
  const a = solveStaticLowFluxFrontier2D(forward);
  const b = solveStaticLowFluxFrontier2D(reflected);
  assert.deepEqual(words(a.factors), words(Float32Array.from(b.factors).reverse()));
  assert.deepEqual(words(a.lowStateVolume), words(Float32Array.from(b.lowStateVolume).reverse()));
});

test("topology-derived synthetic B8:B4 seam gives the same low state with fewer frontier visits", () => {
  const slice = createAdvanceSlice(productionSceneSliceSeedById("coarse-surface-translation"));
  const topology = slice.numericalTopology;
  assert.deepEqual([...new Set(slice.topology.accepted.bricks
    .filter(brick => brick.active !== false).map(brick => brick.resolution))].sort(), [1, 2]);
  const seam = topology.subfaces.find(face => face.negativeCell >= 0 && face.positiveCell >= 0
    && topology.rows[face.rowId]!.kind === "mixed-seam");
  assert.ok(seam, "fixture must expose an internal B8:B4 subface");
  const capacities = Float32Array.from(topology.cells,
    cell => Math.fround(slice.fields.capacity[cell.id]! * cell.area));
  const volumes = Float32Array.from(capacities,
    capacity => Math.fround(0.5 * capacity));
  const receiver = seam.positiveCell;
  volumes[receiver] = Math.fround(0.95 * capacities[receiver]!);
  const seamFlux = Math.fround(0.1 * Math.min(
    capacities[seam.negativeCell]!, capacities[seam.positiveCell]!));
  const faces = topology.subfaces.map(face => ({ negativeCell: face.negativeCell,
    positiveCell: face.positiveCell, lowFlux: face.id === seam.id ? seamFlux : 0 }));
  const problem: StaticLowFluxProblem2D = {
    volumes, capacities,
    sourceRates: new Float32Array(topology.cells.length), dt: 0, faces,
    incidences: topology.cells.map(cell => {
      const entries: { face: number; negative: boolean }[] = [];
      const source = topology.subfaceIncidences?.[cell.id];
      if (source) for (const entry of source)
        entries.push({ face: entry.subfaceId, negative: entry.negative });
      else for (const face of topology.subfaces) {
        if (face.negativeCell === cell.id) entries.push({ face: face.id, negative: true });
        else if (face.positiveCell === cell.id) entries.push({ face: face.id, negative: false });
      }
      return entries;
    }),
  };
  const [copy, , frontier] = assertExact(problem);
  assert.equal(copy.converged, true);
  assert.ok(topology.rows.some(row => row.kind === "mixed-seam"));
  assert.ok(frontier.updateCellVisits < copy.updateCellVisits);
  assertBounds(problem, frontier);
});

test("moving-solid inputs are explicitly routed to the unchanged dense algorithm", () => {
  assert.equal(lowFluxLimiterExecution2D({ solidMotionActive: true }), "dense-moving-solid");
  assert.equal(lowFluxLimiterExecution2D({ solidMotionActive: false }), "active-frontier");
  const moving = { ...staticLowFluxProblem2D({ volumes: [0.5], capacities: [1], faces: [] }),
    solidMotionActive: true };
  assert.throws(() => solveStaticLowFluxFrontier2D(moving), /static-only/);
});

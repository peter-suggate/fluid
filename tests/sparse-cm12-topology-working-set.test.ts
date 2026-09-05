import assert from "node:assert/strict";
import test from "node:test";
import { createOceanSeicheScene } from "../lib/core/scenes";
import { adaptiveMassPresentationDimensionsForScene } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { buildSparseAtlasCompositeGrid, type SparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { createSparseAdaptiveMassAtlas, initializeSparseBrickAtlasFromScene, sparseBrickSpan,
  sparseBrickKey, type SparseAdaptiveMassBrick,
  type SparseBrickResolution } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { prepareSparseCM12TopologyWorkingSet, type SparseCM12TopologyPreparation,
  type SparseCM12TopologyPreparationBudget } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const budget: SparseCM12TopologyPreparationBudget = {
  maximumCells: 100_000, maximumRows: 500_000, maximumBytes: 64 * 1024 ** 2,
};
type Ready = Extract<SparseCM12TopologyPreparation, { status: "ready" }>;

function fixture(dimensions: readonly [number, number, number],
  specs: readonly { q: readonly [number, number, number]; r: SparseBrickResolution; span?: number }[]) {
  const dims = dimensions.map((n) => Math.ceil(n / 8)) as [number, number, number];
  const bricks: SparseAdaptiveMassBrick[] = specs.map(({ q, r, span }) => ({
    key: sparseBrickKey(q, dims), coordinate: q, resolution: r, spanBricks: span,
    density: Float64Array.from({ length: r ** 3 }, (_, i) => 0.25 + (i % 4) / 8),
    gamma: new Float64Array(r ** 3).fill(1),
  }));
  return buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas(dimensions, bricks, 7, 8));
}

function assertReady(result: SparseCM12TopologyPreparation): asserts result is Ready {
  if (result.status !== "ready") assert.fail(JSON.stringify(result));
}

const cellKey = (leaf: number, center: readonly number[], widths: readonly number[]) =>
  `${leaf}:${center.join(",")}:${widths.join(",")}`;
function packed(packet: Ready) {
  const words = packet.words, floats = new Float32Array(words.buffer);
  const cellBase = words[6]!, rowBase = words[7]!, termBase = words[8]!;
  const cell = (id: number) => cellKey(words[cellBase + 8 * id + 7]! >>> 5,
    Array.from(floats.slice(cellBase + 8 * id, cellBase + 8 * id + 3)),
    Array.from(floats.slice(cellBase + 8 * id + 4, cellBase + 8 * id + 7)));
  const row = (id: number) => {
    const at = (plane: number) => rowBase + plane * packet.rowCount + id;
    const termBits = words[at(0)]!, first = termBits & 0x007fffff, count = termBits >>> 23;
    return JSON.stringify([words[at(1)]! >>> 30, (words[at(1)]! >>> 28) & 3,
      ...[2, 3, 4, 5, 6, 7, 8].map((plane) => floats[at(plane)]!),
      Array.from({ length: count }, (_, i) => {
        const term = termBase + 2 * (first + i);
        return [cell(words[term]!), floats[term + 1]!];
      }).sort((a, b) => String(a[0]).localeCompare(String(b[0])))]);
  };
  return { cell, row };
}

function checkGraph(packet: Ready, graph: SparseAtlasCompositeGrid, candidate: boolean) {
  const read = packed(packet), byKey = new Map(graph.atlas.bricks.map((b, i) => [b.key, i]));
  const cell = (id: number) => {
    const c = graph.cells[id]!;
    return cellKey(byKey.get(c.brickKey)!, c.centerFine, c.widthsFine);
  };
  const cells = candidate ? packet.candidateCellWorklist : packet.acceptedCellWorklist;
  const rows = candidate ? packet.candidateRowWorklist : packet.acceptedRowWorklist;
  assert.deepEqual(Array.from(cells, read.cell).sort(), graph.cells.map((c) => cell(c.id)).sort());
  const kind = { "intra-brick": 0, "brick-face": 1, "mixed-seam": 2, "sparse-air": 3 };
  const expected = graph.gradientRows.map((r) => JSON.stringify([
    r.axis, kind[r.kind], ...[r.dualWeight, r.area, r.distance, r.exteriorPhi ?? 0.5].map(Math.fround),
    ...r.centerFine.map(Math.fround), r.terms.map((t) => [cell(t.cellId), Math.fround(t.coefficient)])
      .sort((a, b) => String(a[0]).localeCompare(String(b[0]))),
  ])).sort();
  assert.deepEqual(Array.from(rows, read.row).sort(), expected,
    "the selected rows must equal a separately compiled complete graph, including sparse-air faces");

  // Every row term must occur exactly once in its endpoint's incidence list.
  const w = packet.words, offsets = w[9]!, records = w[10]!, terms = w[8]!;
  const seen = new Set<number>();
  for (let c = 0; c < packet.cellCount; c++) {
    for (let at = w[offsets + c]!; at < w[offsets + c + 1]!; at++) {
      const term = w[records + 2 * at + 1]!;
      assert.equal(w[terms + 2 * term], c);
      assert.equal(seen.has(term), false); seen.add(term);
    }
  }
  assert.equal(seen.size, w[4]);

  // Pressure adjacency must preserve every off-diagonal pair from each row,
  // including multi-term coarse/fine seams, with no extra edges.
  const f = new Float32Array(w.buffer), edgeOffsets = w[15]!;
  const edgeRecords = edgeOffsets + packet.cellCount + 1;
  for (let c = 0; c < packet.cellCount; c++) {
    const expectedEdges = new Map<string, number>();
    for (let at = w[offsets + c]!; at < w[offsets + c + 1]!; at++) {
      const row = w[records + 2 * at]!, own = w[records + 2 * at + 1]!;
      const bits = w[w[7]! + row]!, first = bits & 0x007fffff, count = bits >>> 23;
      const weight = f[w[7]! + 2 * packet.rowCount + row]!;
      for (let term = first; term < first + count; term++) {
        const other = w[terms + 2 * term]!;
        if (other !== c) expectedEdges.set(`${row}/${other}`,
          f[terms + 2 * own + 1]! * weight * f[terms + 2 * term + 1]!);
      }
    }
    for (let edge = w[edgeOffsets + c]!; edge < w[edgeOffsets + c + 1]!; edge++) {
      const at = edgeRecords + 3 * edge;
      const key = `${w[at]}/${w[at + 1]}`, expected = expectedEdges.get(key);
      assert.notEqual(expected, undefined);
      assert.ok(Math.abs(f[at + 2]! - expected!) <= 1e-6 * Math.max(1, Math.abs(expected!)));
      expectedEdges.delete(key);
    }
    assert.equal(expectedEdges.size, 0);
  }
}

const candidateGraph = (grid: SparseAtlasCompositeGrid, requests: ReadonlyMap<number, SparseBrickResolution>) =>
  buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas(grid.atlas.dimensions,
    grid.atlas.bricks.map((b) => {
      const resolution = requests.get(b.key) ?? b.resolution;
      return { ...b, resolution, density: new Float64Array(resolution ** 3).fill(0.5),
        gamma: new Float64Array(resolution ** 3).fill(1) };
    }), grid.atlas.generation + 1, 8));

test("requested topology includes exactly the accepted and candidate graph, not all rungs", () => {
  const grid = fixture([32, 16, 16], Array.from({ length: 16 }, (_, i) => ({
    q: [i % 4, Math.floor(i / 4) % 2, Math.floor(i / 8)] as const, r: 2 as const,
  })));
  const requests = new Map(grid.atlas.bricks.filter((b) => b.coordinate[0] === 1)
    .map((b) => [b.key, 4 as const]));
  const packet = prepareSparseCM12TopologyWorkingSet(grid, requests, budget);
  assertReady(packet);
  assert.equal(packet.acceptedGeneration, 7); assert.equal(packet.candidateGeneration, 8);
  assert.equal(packet.cellCount, grid.cells.length + requests.size * 4 ** 3);
  checkGraph(packet, grid, false); checkGraph(packet, candidateGraph(grid, requests), true);
  const rangeBase = packet.words[11]!;
  for (let leaf = 0; leaf < grid.atlas.bricks.length; leaf++) {
    assert.equal(packet.words[rangeBase + 2 * (4 * leaf + 3) + 1], 0,
      "an unrequested B8 rung must have no cell range, not alias accepted cells");
  }
});

test("candidate preparation discovers every partial face of a macro leaf", () => {
  const grid = fixture([32, 16, 16], [{ q: [0, 0, 0], r: 2, span: 2 },
    ...Array.from({ length: 4 }, (_, i) => ({
      q: [2, i % 2, Math.floor(i / 2)] as const, r: 1 as const,
    }))]);
  const requests = new Map([[grid.atlas.bricks[0]!.key, 4 as const]]);
  const packet = prepareSparseCM12TopologyWorkingSet(grid, requests, budget);
  assertReady(packet);
  checkGraph(packet, grid, false); checkGraph(packet, candidateGraph(grid, requests), true);
});

test("macro re-rung preparation supports physical cells beyond 16h", () => {
  const grid = fixture([512, 256, 256], [
    { q: [0, 0, 0], r: 2, span: 32 }, { q: [32, 0, 0], r: 2, span: 32 },
  ]);
  const requests = new Map([[grid.atlas.bricks[0]!.key, 4 as const]]);
  const packet = prepareSparseCM12TopologyWorkingSet(grid, requests, budget);
  assertReady(packet);
  assert.equal(packet.cellCount, 16 + 64);
  checkGraph(packet, grid, false); checkGraph(packet, candidateGraph(grid, requests), true);
});

test("candidate chunks agree on shared halos without borrowing reused workspace cells", () => {
  const grid = fixture([96, 48, 32], Array.from({ length: 288 }, (_, i) => ({
    q: [i % 12, Math.floor(i / 12) % 6, Math.floor(i / 72)] as const, r: 2 as const,
  })));
  const requests = new Map(grid.atlas.bricks.filter((b) => b.coordinate[0] < 9)
    .map((b) => [b.key, 1 as const]));
  assert.ok(requests.size > 128);
  const packet = prepareSparseCM12TopologyWorkingSet(grid, requests, budget);
  assertReady(packet);
  assert.equal(packet.cellCount, grid.cells.length + requests.size);
  checkGraph(packet, grid, false); checkGraph(packet, candidateGraph(grid, requests), true);
});

test("clipped-cell restriction preserves represented mass and complete boundary rows", () => {
  const grid = fixture([13, 10, 9], Array.from({ length: 8 }, (_, i) => ({
    q: [i % 2, Math.floor(i / 2) % 2, Math.floor(i / 4)] as const, r: 4 as const,
  })));
  const requests = new Map(grid.atlas.bricks.map((b) => [b.key, 2 as const]));
  const packet = prepareSparseCM12TopologyWorkingSet(grid, requests, budget);
  assertReady(packet);
  checkGraph(packet, candidateGraph(grid, requests), true);
  const f = new Float32Array(packet.words.buffer), base = packet.words[6]!;
  const mass = (cells: Uint32Array) => cells.reduce((sum, c) =>
    sum + packet.initialDensity[c]! * f[base + 8 * c + 3]!, 0);
  assert.ok(Math.abs(mass(packet.acceptedCellWorklist) - mass(packet.candidateCellWorklist)) < 1e-4);
});

test("every preparation budget defers atomically and can be retried", () => {
  const grid = fixture([16, 8, 8], [{ q: [0, 0, 0], r: 2 }, { q: [1, 0, 0], r: 2 }]);
  const before = structuredClone(grid);
  const requests = new Map(grid.atlas.bricks.map((b) => [b.key, 4 as const]));
  const ready = prepareSparseCM12TopologyWorkingSet(grid, requests, budget);
  assertReady(ready);
  for (const [resource, limits] of [
    ["cells", { ...budget, maximumCells: ready.cellCount - 1 }],
    ["rows", { ...budget, maximumRows: ready.rowCount - 1 }],
    ["bytes", { ...budget, maximumBytes: ready.words.byteLength - 4 }],
  ] as const) {
    const deferred = prepareSparseCM12TopologyWorkingSet(grid, requests, limits);
    assert.equal(deferred.status, "deferred");
    if (deferred.status === "deferred") assert.equal(deferred.resource, resource);
    assert.deepEqual(grid, before);
  }
  const retried = prepareSparseCM12TopologyWorkingSet(grid, requests, budget);
  assertReady(retried); assert.deepEqual(retried.words, ready.words);
  assert.throws(() => prepareSparseCM12TopologyWorkingSet(grid,
    new Map([[grid.atlas.bricks[0]!.key, 8]]), budget), /2:1 closure/);
  assert.throws(() => prepareSparseCM12TopologyWorkingSet(grid,
    new Map([[999, 4]]), budget), /invalid/);
});

test("full ocean prepares a changing region beyond the legacy mutable-leaf ceiling", (t) => {
  const scene = createOceanSeicheScene();
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
    initialSurfaceCoarseningBiasRings: 1,
  });
  const mutable = atlas.bricks.filter((b) => sparseBrickSpan(b) === 1);
  assert.ok(mutable.length > 2_048);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  // Coarsening the finest rung preserves the already graded physical 2:1 envelope.
  const requests = new Map(mutable.filter((b) => b.resolution === 8)
    .map((b) => [b.key, 4 as const]));
  assert.ok(requests.size > 0);
  const packet = prepareSparseCM12TopologyWorkingSet(grid, requests, {
    maximumCells: grid.cells.length + requests.size * 64,
    maximumRows: 1_000_000, maximumBytes: 256 * 1024 ** 2,
  });
  assertReady(packet);
  assert.equal(packet.cellCount, grid.cells.length + requests.size * 64);
  assert.equal(packet.changedBrickKeys.length, requests.size);
  const candidate = candidateGraph(grid, requests);
  assert.equal(packet.candidateCellWorklist.length, candidate.cells.length);
  assert.equal(packet.candidateRowWorklist.length, candidate.gradientRows.length);
  assert.equal(packet.acceptedCellWorklist.length, grid.cells.length);
  assert.equal(packet.acceptedRowWorklist.length, grid.gradientRows.length);
  t.diagnostic(JSON.stringify({ mutableLeaves: mutable.length, changedLeaves: requests.size,
    acceptedCells: grid.cells.length, preparedCells: packet.cellCount,
    preparedRows: packet.rowCount, packetBytes: packet.words.byteLength }));
});

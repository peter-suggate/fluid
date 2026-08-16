#!/usr/bin/env node
/** Frozen manufactured-snapshot benchmark for the Sparse CM12 pressure oracle. */

import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  buildSparseAtlasCompositeGrid,
  projectSparseAtlasVelocity,
  type SparseAtlasCompositeGrid,
  type SparseAtlasProjectionResult,
  type SparseAtlasProjectionStageId,
} from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
  type SparseBrickVec3,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { CM12_LIQUID_ISOVALUE, cm12GhostFluidTheta } from
  "../lib/core/cm12-numerics";
import {
  SPARSE_CM12_PRESSURE_RECEIPT_VERSION,
  assertSparseCM12PressureReceipt,
  sparseCM12PressureResidualDrift,
  type SparseCM12PressureReceipt,
} from "../lib/methods/adaptive-mass/sparse-cm12-pressure-receipt";

type Fixture = {
  readonly name: string;
  readonly atlas: SparseAdaptiveMassAtlas;
  readonly velocityScale?: number;
  readonly targetDivergence?: (grid: SparseAtlasCompositeGrid) => Float64Array;
  readonly warm?: boolean;
};

const args = new Map(process.argv.slice(2).map((argument) => {
  const [key, value = "true"] = argument.replace(/^--/, "").split("=", 2);
  return [key, value] as const;
}));
const maximumIterations = Math.max(1, Number(args.get("iterations") ?? 256));
const relativeTolerance = Number(args.get("relative-tolerance") ?? 1e-8);
const requestedFixture = args.get("fixture");

function brick(
  dimensions: SparseBrickVec3,
  coordinate: SparseBrickVec3,
  resolution: SparseBrickResolution,
  density: number | ((x: number, y: number, z: number) => number) = 1,
): SparseAdaptiveMassBrick {
  const count = resolution ** 3;
  const values = Float64Array.from({ length: count }, (_, index) => {
    const x = index % resolution;
    const y = Math.floor(index / resolution) % resolution;
    const z = Math.floor(index / resolution ** 2);
    return typeof density === "number" ? density : density(x, y, z);
  });
  const brickDimensions = dimensions.map((value) => Math.ceil(value / 8)) as
    [number, number, number];
  return {
    key: sparseBrickKey(coordinate, brickDimensions), coordinate, resolution,
    density: values, gamma: new Float64Array(count).fill(1),
  };
}

function atlas(
  dimensions: SparseBrickVec3,
  definitions: readonly [SparseBrickVec3, SparseBrickResolution,
    number | ((x: number, y: number, z: number) => number)][],
): SparseAdaptiveMassAtlas {
  return createSparseAdaptiveMassAtlas(dimensions, definitions.map((definition) =>
    brick(dimensions, ...definition)));
}

function fixtures(): readonly Fixture[] {
  const fineDimensions: SparseBrickVec3 = [16, 8, 8];
  const longDimensions: SparseBrickVec3 = [64, 8, 8];
  const longBricks: [SparseBrickVec3, SparseBrickResolution,
    (x: number, y: number, z: number) => number][] = [];
  for (let x = 0; x < 8; x += 1) {
    longBricks.push([[x, 0, 0], 4,
      (_lx, ly, lz) => ly === 1 && lz === 1 ? 1 : 0]);
  }
  return [
    { name: "complete-regular-all-fine", atlas: atlas(fineDimensions, [
      [[0, 0, 0], 8, 1], [[1, 0, 0], 8, 1],
    ]) },
    { name: "complete-regular-all-coarse", atlas: atlas(fineDimensions, [
      [[0, 0, 0], 4, 1], [[1, 0, 0], 4, 1],
    ]) },
    { name: "localized-liquid-island", atlas: atlas([32, 16, 16], [
      [[1, 0, 0], 8, (x, y, z) => x >= 2 && x <= 5 && y >= 2 && y <= 5
        && z >= 2 && z <= 5 ? 1 : 0],
    ]) },
    { name: "long-thin-component", atlas: atlas(longDimensions, longBricks) },
    { name: "planar-free-surface", atlas: atlas(fineDimensions, [
      [[0, 0, 0], 8, (_x, y) => y < 3 ? 1 : y === 3 ? 0.55 : 0],
      [[1, 0, 0], 8, (_x, y) => y < 3 ? 1 : y === 3 ? 0.55 : 0],
    ]) },
    { name: "single-2-to-1-seam", atlas: atlas(fineDimensions, [
      [[0, 0, 0], 4, 1], [[1, 0, 0], 8, 1],
    ]) },
    { name: "deep-wet-bulk-fine-surface", atlas: atlas([16, 16, 8], [
      [[0, 0, 0], 4, 1], [[1, 0, 0], 4, 1],
      [[0, 1, 0], 8, (_x, y) => y < 5 ? 1 : 0],
      [[1, 1, 0], 8, (_x, y) => y < 5 ? 1 : 0],
    ]) },
    { name: "all-neumann-compatible", atlas: atlas(fineDimensions, [
      [[0, 0, 0], 4, 1], [[1, 0, 0], 4, 1],
    ]), targetDivergence: (grid) => Float64Array.from(grid.cells,
      (cell) => cell.centerFine[0] < 8 ? -1e-3 : 1e-3) },
    { name: "all-neumann-incompatible", atlas: atlas(fineDimensions, [
      [[0, 0, 0], 4, 1], [[1, 0, 0], 4, 1],
    ]), targetDivergence: (grid) => new Float64Array(grid.cells.length).fill(1e-3) },
    { name: "topology-transition-warm-start", warm: true, atlas: atlas(
      fineDimensions, [[[0, 0, 0], 4, 1], [[1, 0, 0], 8, 1]],
    ) },
  ];
}

function classifyTopology(grid: SparseAtlasCompositeGrid): {
  rows: SparseCM12PressureReceipt["topology"]["rows"];
  rowDegreeHistogram: Record<string, number>;
  maximumRowDegree: number;
  activeRows: number;
  edges: number;
  minimumTheta: number;
  thetaHistogram: Record<string, number>;
} {
  const liquid = Uint8Array.from(grid.cells, (cell) =>
    cell.density >= CM12_LIQUID_ISOVALUE ? 1 : 0);
  const rows = { regular: 0, "brick-face": 0, "mixed-seam": 0, "sparse-air": 0 };
  const rowDegreeHistogram: Record<string, number> = {};
  const thetaHistogram: Record<string, number> = {};
  let maximumRowDegree = 0;
  let activeRows = 0;
  let edges = 0;
  let minimumTheta = 1;
  for (const row of grid.gradientRows) {
    const liquidTerms = row.terms.filter((term) => liquid[term.cellId] !== 0);
    if (row.dualWeight <= 0 || liquidTerms.length === 0) continue;
    activeRows += 1;
    const rowClass = row.kind === "intra-brick" ? "regular" : row.kind;
    rows[rowClass] += 1;
    const degree = liquidTerms.length;
    rowDegreeHistogram[String(degree)] = (rowDegreeHistogram[String(degree)] ?? 0) + 1;
    maximumRowDegree = Math.max(maximumRowDegree, degree);
    edges += degree * Math.max(0, degree - 1);
    const airTerms = row.terms.filter((term) => liquid[term.cellId] === 0);
    if (airTerms.length === 0 && row.kind !== "sparse-air") continue;
    const liquidPhi = liquidTerms.reduce((sum, term) => sum
      + Math.abs(term.coefficient)
        * (CM12_LIQUID_ISOVALUE - grid.cells[term.cellId]!.density), 0)
      / liquidTerms.reduce((sum, term) => sum + Math.abs(term.coefficient), 0);
    const airWeight = airTerms.reduce((sum, term) => sum + Math.abs(term.coefficient), 0);
    const exteriorWeight = row.kind === "sparse-air"
      ? liquidTerms.reduce((sum, term) => sum + Math.abs(term.coefficient), 0) : 0;
    const airPhi = (airTerms.reduce((sum, term) => sum + Math.abs(term.coefficient)
      * (CM12_LIQUID_ISOVALUE - grid.cells[term.cellId]!.density), 0)
      + exteriorWeight * (row.exteriorPhi ?? 0.5)) / Math.max(airWeight + exteriorWeight, 1e-30);
    const theta = cm12GhostFluidTheta(liquidPhi, airPhi, 1e-12);
    minimumTheta = Math.min(minimumTheta, theta);
    const bucket = String(Math.floor(Math.log2(Math.max(theta, 2 ** -32))));
    thetaHistogram[bucket] = (thetaHistogram[bucket] ?? 0) + 1;
  }
  return { rows, rowDegreeHistogram, maximumRowDegree, activeRows, edges,
    minimumTheta, thetaHistogram };
}

function makeReceipt(
  fixture: Fixture,
  grid: SparseAtlasCompositeGrid,
  result: SparseAtlasProjectionResult,
  stageTimes: Record<SparseAtlasProjectionStageId, number>,
  totalTime: number,
  seedKind: SparseCM12PressureReceipt["seed"]["kind"],
): SparseCM12PressureReceipt {
  const topology = classifyTopology(grid);
  const recursiveRelativeL2 = result.receipt.relativeResidualL2;
  const trueRelativeL2 = result.receipt.relativeResidualL2;
  const executed = result.receipt.iterations;
  const pressureState = 7 * grid.cells.length * 8 + grid.gradientRows.length * 8;
  const effectiveEdges = topology.edges * 8;
  const reductionPartials = Math.max(1, Math.ceil(grid.cells.length / 64)) * 16;
  const incompatible = result.receipt.rhsCompatibilityMaxAbs
    > 1e-10 * Math.max(1, Math.sqrt(result.leafRhs.reduce((sum, value) =>
      sum + value * value, 0)));
  const receipt: SparseCM12PressureReceipt = {
    version: SPARSE_CM12_PRESSURE_RECEIPT_VERSION,
    snapshot: fixture.name,
    solver: "cpu-composite-oracle",
    topology: {
      denseCellCount: grid.atlas.dimensions.reduce((product, value) => product * value, 1),
      acceptedCellCount: grid.cells.length,
      acceptedRowCount: grid.gradientRows.length,
      liquidCellCount: result.receipt.liquidCellCount,
      activePressureRowCount: topology.activeRows,
      effectiveOffDiagonalCount: topology.edges,
      rows: topology.rows,
      rowDegreeHistogram: topology.rowDegreeHistogram,
      maximumRowDegree: topology.maximumRowDegree,
    },
    theta: {
      minimum: topology.minimumTheta,
      logarithmicHistogram: topology.thetaHistogram,
      clampCount: result.receipt.thetaClampCount,
    },
    components: {
      count: result.receipt.componentCount,
      anchored: result.receipt.anchoredComponentCount,
      unanchored: result.receipt.unanchoredComponentCount,
      compatibilityCorrectionMaximum: result.receipt.rhsCompatibilityMaxAbs,
      incompatible,
    },
    seed: {
      kind: seedKind,
      trueInitial: {
        relativeL2: result.receipt.initialRelativeResidualL2,
        maximum: result.receipt.initialMaximumResidual,
        maximumDivergenceEquivalent_s: result.receipt.preDivergenceMaximum,
      },
      zeroSeedTrueInitial: {
        relativeL2: result.receipt.zeroSeedInitialRelativeResidualL2,
        maximum: Math.max(...result.leafRhs.map(Math.abs), 0),
        maximumDivergenceEquivalent_s: result.receipt.preDivergenceMaximum,
      },
      newlyLiquidCellCount: 0,
      fallbackToZeroCellCount: 0,
      physicalScaleRatio: 1,
      rescaled: false,
    },
    convergence: {
      converged: result.receipt.converged,
      reason: result.receipt.converged ? "tolerance" : "iteration-cap",
      encodedIterationCeiling: maximumIterations,
      executedIterations: executed,
      firstToleranceCrossingIteration: result.receipt.converged ? executed : null,
      gatedTailIterations: maximumIterations - executed,
      spmvApplications: executed + 2,
      recursiveRelativeL2,
      trueFinal: {
        relativeL2: trueRelativeL2,
        maximum: result.receipt.maximumResidual,
        maximumDivergenceEquivalent_s: result.receipt.postDivergenceMaximum,
      },
      recursiveToTrueRatio: trueRelativeL2 > 0
        ? recursiveRelativeL2 / trueRelativeL2 : 1,
      residualDrift: sparseCM12PressureResidualDrift(
        recursiveRelativeL2, trueRelativeL2,
      ),
    },
    projection: {
      divergenceVolumeL2_s: result.receipt.postDivergenceVolumeL2,
      divergenceMaximum_s: result.receipt.postDivergenceMaximum,
    },
    timing_ms: {
      topology: stageTimes.topology,
      rhs: stageTimes.rhs,
      solve: stageTimes.solve,
      projection: stageTimes.projection,
      diagnostics: stageTimes.diagnostics,
      total: totalTime,
      method: "cpu-wall",
    },
    memoryBytes: {
      pressureState, effectiveEdges, reductionPartials, hierarchy: 0,
      total: pressureState + effectiveEdges + reductionPartials,
    },
  };
  assertSparseCM12PressureReceipt(receipt);
  return receipt;
}

function run(fixture: Fixture): SparseCM12PressureReceipt {
  const grid = buildSparseAtlasCompositeGrid(fixture.atlas);
  const velocity = Float64Array.from(grid.gradientRows, (row) =>
    (fixture.velocityScale ?? 0.01) * Math.sin(0.173 * (row.id + 1)));
  const targetDivergence = fixture.targetDivergence?.(grid);
  let initialPressure: Float64Array | undefined;
  if (fixture.warm) {
    const previousVelocity = Float64Array.from(velocity, (value) => 0.98 * value);
    initialPressure = projectSparseAtlasVelocity(grid, {
      normalVelocity: previousVelocity, targetDivergence,
      maximumIterations, relativeTolerance,
    }).leafPressure.slice();
  }
  const stageTimes = { topology: 0, rhs: 0, solve: 0, projection: 0, diagnostics: 0 };
  let previous = performance.now();
  const started = previous;
  const result = projectSparseAtlasVelocity(grid, {
    normalVelocity: velocity, targetDivergence, initialPressure,
    maximumIterations, relativeTolerance,
    onStageComplete: (stage) => {
      const now = performance.now();
      stageTimes[stage] += now - previous;
      previous = now;
    },
  });
  return makeReceipt(fixture, grid, result, stageTimes,
    performance.now() - started, fixture.warm ? "transferred" : "zero");
}

const selected = fixtures().filter((fixture) =>
  requestedFixture === undefined || fixture.name === requestedFixture);
if (selected.length === 0) throw new Error(`unknown fixture ${requestedFixture}`);
const receipts = selected.map(run);
const document = {
  schema: "sparse-cm12-pressure-benchmark/v1",
  generatedAt: new Date().toISOString(),
  command: process.argv.join(" "),
  receipts,
};
const json = `${JSON.stringify(document, null, 2)}\n`;
const output = args.get("output");
if (output) await writeFile(resolve(output), json);
else process.stdout.write(json);

import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH,
  OCTREE_SECTION43_MINI_FINEST_CELL_CAPACITY,
  OCTREE_SECTION43_MINI_SHELL_DEPTH,
  OCTREE_SECTION43_SHELL_DEPTH_ENVIRONMENT,
  OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL_ENVIRONMENT,
  OCTREE_SOLVE_TAIL_RELATIVE_TOLERANCE,
  OCTREE_VCYCLE_MINIMUM_PARALLEL_LEVELS,
  OCTREE_VCYCLE_TAIL_WORKGROUP_CELLS,
  countOctreePressureCommands,
  octreeFactorOnePredictedSolveTailEnabled,
  planOctreeSolveTail,
  planOctreeVCycleParallelLevels,
  selectOctreeFactorOneEncodedSolveTail,
  type OctreeSolveTailSceneProfile,
} from "../lib/octree-solve-tail-policy";
import { solveOctreePipelinedPCG } from "../lib/octree-pipelined-pcg";

const PROFILES = Object.freeze({
  miniDam: {
    finestDimensions: [16, 16, 16], maximumLeafSize: 32,
    initialCondition: "dam-break", hasInflow: false, hasTerrain: false,
    movingRigidBodyCount: 0, closedTop: false, requestedRelativeTolerance: 1e-4,
  },
  ceilingDrop: {
    finestDimensions: [24, 16, 24], maximumLeafSize: 32,
    initialCondition: "dam-break", hasInflow: false, hasTerrain: false,
    movingRigidBodyCount: 0, closedTop: true, requestedRelativeTolerance: 1e-4,
  },
  largerTwoLevel: {
    finestDimensions: [32, 24, 16], maximumLeafSize: 32,
    initialCondition: "tank-fill", hasInflow: false, hasTerrain: false,
    movingRigidBodyCount: 0, closedTop: true, requestedRelativeTolerance: 1e-4,
  },
  uiDam: {
    finestDimensions: [24, 18, 16], maximumLeafSize: 32,
    initialCondition: "dam-break", hasInflow: false, hasTerrain: false,
    movingRigidBodyCount: 0, closedTop: false, requestedRelativeTolerance: 1e-4,
  },
  quiescent: {
    finestDimensions: [48, 24, 24], maximumLeafSize: 32,
    initialCondition: "tank-fill", hasInflow: false, hasTerrain: false,
    movingRigidBodyCount: 0, closedTop: true, requestedRelativeTolerance: 1e-6,
  },
  river: {
    finestDimensions: [96, 16, 16], maximumLeafSize: 32,
    initialCondition: "tank-fill", hasInflow: true, hasTerrain: true,
    movingRigidBodyCount: 0, closedTop: false, requestedRelativeTolerance: 1e-4,
  },
  ocean: {
    finestDimensions: [320, 96, 80], maximumLeafSize: 16,
    initialCondition: "tank-fill", hasInflow: false, hasTerrain: false,
    movingRigidBodyCount: 0, closedTop: true, requestedRelativeTolerance: 1e-4,
  },
} satisfies Record<string, OctreeSolveTailSceneProfile>);

test("factor-one predicted tail is explicit-on and fails to the full envelope", () => {
  assert.equal(octreeFactorOnePredictedSolveTailEnabled({}), false);
  assert.equal(octreeFactorOnePredictedSolveTailEnabled({
    [OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL_ENVIRONMENT]: "0",
  }), false);
  assert.equal(octreeFactorOnePredictedSolveTailEnabled({
    [OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL_ENVIRONMENT]: "1",
  }), true);
  const common = {
    enabled: true, factorOne: true, nextStep: 42,
    fullEncodedOuterIterations: 10,
  } as const;
  assert.deepEqual(selectOctreeFactorOneEncodedSolveTail(common), {
    encodedOuterIterations: 10, usedHistory: false, reason: "missing-history",
  });
  assert.equal(selectOctreeFactorOneEncodedSolveTail({
    ...common,
    observation: {
      step: 40, publishedIterationCount: 4, converged: true,
      acceptedTopologyEpoch: 7, topologyHash: 70,
      topologyFlipReady: false, topologyEpochError: 0,
    },
    precedingTopologyHash: 70,
  }).reason, "non-adjacent-history");
  for (const observation of [
    { step: 41, publishedIterationCount: 4, converged: false,
      acceptedTopologyEpoch: 7, topologyHash: 70,
      topologyFlipReady: false, topologyEpochError: 0 },
    { step: 41, publishedIterationCount: 4, converged: true,
      acceptedTopologyEpoch: 7, topologyHash: 70,
      topologyFlipReady: false, topologyEpochError: 1 },
  ]) {
    assert.equal(selectOctreeFactorOneEncodedSolveTail({
      ...common, observation, precedingTopologyHash: 70,
    }).reason, "invalid-history");
  }
  for (const selection of [
    selectOctreeFactorOneEncodedSolveTail({
      ...common,
      observation: {
        step: 41, publishedIterationCount: 4, converged: true,
        acceptedTopologyEpoch: 8, topologyHash: 81,
        topologyFlipReady: true, topologyEpochError: 0,
      },
      precedingTopologyHash: 80,
    }),
    selectOctreeFactorOneEncodedSolveTail({
      ...common,
      observation: {
        step: 41, publishedIterationCount: 4, converged: true,
        acceptedTopologyEpoch: 8, topologyHash: 81,
        topologyFlipReady: false, topologyEpochError: 0,
      },
      precedingTopologyHash: 80,
    }),
  ]) {
    assert.deepEqual(selection, {
      encodedOuterIterations: 10, usedHistory: false, reason: "topology-change",
    });
  }
  assert.deepEqual(selectOctreeFactorOneEncodedSolveTail({
    ...common,
    observation: {
      step: 41, publishedIterationCount: 4, converged: true,
      acceptedTopologyEpoch: 7, topologyHash: 70,
      topologyFlipReady: true, topologyEpochError: 0,
    },
    precedingTopologyHash: 70,
  }), {
    encodedOuterIterations: 6, usedHistory: true, reason: "predicted",
  });
  assert.deepEqual(selectOctreeFactorOneEncodedSolveTail({
    ...common,
    observation: {
      step: 41, publishedIterationCount: 9, converged: true,
      acceptedTopologyEpoch: 7, topologyHash: 70,
      topologyFlipReady: false, topologyEpochError: 0,
    },
    precedingTopologyHash: 70,
  }), {
    encodedOuterIterations: 10, usedHistory: true, reason: "predicted",
  });
  assert.equal(selectOctreeFactorOneEncodedSolveTail({
    ...common, factorOne: false,
  }).reason, "not-factor-one");
  assert.equal(selectOctreeFactorOneEncodedSolveTail({
    ...common, enabled: false,
  }).reason, "disabled");
});

test("solve-tail policy encodes the paper upper envelope and keeps scene score as telemetry", () => {
  const mini = planOctreeSolveTail(PROFILES.miniDam);
  const quiescent = planOctreeSolveTail(PROFILES.quiescent);
  const river = planOctreeSolveTail(PROFILES.river);
  assert.deepEqual([
    mini.encodedOuterIterations,
    quiescent.encodedOuterIterations,
    river.encodedOuterIterations,
  ], [10, 10, 10]);
  assert.equal(mini.boundarySmoothingIterations,
    OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH);
  assert.equal(OCTREE_SECTION43_MINI_FINEST_CELL_CAPACITY, 9_216);
  assert.equal(planOctreeSolveTail(PROFILES.ceilingDrop).boundarySmoothingIterations,
    OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH,
    "the 24x16x24 ceiling shares the production k=8 shell at leaf 32");
  for (const policy of [quiescent, river, planOctreeSolveTail(PROFILES.uiDam),
    planOctreeSolveTail(PROFILES.largerTwoLevel), planOctreeSolveTail(PROFILES.ocean)]) {
    assert.ok(policy.encodedOuterIterations >= 4
      && policy.encodedOuterIterations <= 10);
    assert.equal(policy.hardOuterIterationCeiling, 16);
    assert.equal(policy.boundarySmoothingIterations,
      OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH);
    assert.ok(policy.relativeTolerance >= OCTREE_SOLVE_TAIL_RELATIVE_TOLERANCE);
  }
  assert.equal(quiescent.relativeTolerance, 1e-4,
    "production retains the established f32 residual acceptance floor");
});

test("solve-tail policy admits an explicit symmetric Section 4.3 shell-depth experiment", () => {
  for (const depth of [2, 4, 6, 8, 10, 12, 14, 16]) {
    const policy = planOctreeSolveTail(PROFILES.miniDam, {
      [OCTREE_SECTION43_SHELL_DEPTH_ENVIRONMENT]: String(depth),
    });
    assert.equal(policy.boundarySmoothingIterations, depth);
  }
  for (const value of ["", "0", "3", "18", "four"]) {
    assert.throws(() => planOctreeSolveTail(PROFILES.miniDam, {
      [OCTREE_SECTION43_SHELL_DEPTH_ENVIRONMENT]: value,
    }), /even integer in \[2,16\]/);
  }
});

test("selected Section 4.3 shell has deterministic five-level command counts", () => {
  const expectedCounts = new Map<OctreeSolveTailSceneProfile, number>([
    [PROFILES.miniDam, 872],
    [PROFILES.ceilingDrop, 872],
    [PROFILES.largerTwoLevel, 872],
    [PROFILES.uiDam, 872],
    [PROFILES.quiescent, 872],
    [PROFILES.river, 872],
    [PROFILES.ocean, 872],
  ]);
  for (const profile of Object.values(PROFILES)) {
    const policy = planOctreeSolveTail(profile);
    const commands = countOctreePressureCommands({
      encodedOuterIterations: policy.encodedOuterIterations,
      fullOperatorDispatches: 5,
      mergedBandOperatorDispatches: 1,
      firstOrderSetupDispatches: 10,
      firstOrderCorrectionDispatches: 26,
      boundarySmoothingIterations: policy.boundarySmoothingIterations,
    });
    assert.equal(commands.encodedPressureDispatches, expectedCounts.get(profile),
      profile.finestDimensions.join("x"));
  }
  assert.equal(countOctreePressureCommands({
    encodedOuterIterations: 10,
    fullOperatorDispatches: 5,
    mergedBandOperatorDispatches: 1,
    firstOrderSetupDispatches: 10,
    firstOrderCorrectionDispatches: 26,
    boundarySmoothingIterations: OCTREE_SECTION43_MINI_SHELL_DEPTH,
  }).encodedPressureDispatches, 696,
  "the maximum encoded envelope includes the selected mini matching shell");
  assert.equal(countOctreePressureCommands({
    encodedOuterIterations: 10,
    fullOperatorDispatches: 5,
    mergedBandOperatorDispatches: 1,
    firstOrderSetupDispatches: 10,
    firstOrderCorrectionDispatches: 26,
    boundarySmoothingIterations: 8,
  }).encodedPressureDispatches, 872,
  "the paper k=8 reference remains available through the experiment override");
});

type Matrix = readonly (readonly number[])[];

function fixtureMatrix(length: number, coupling: (edge: number) => number,
  anchor: (row: number) => number): Matrix {
  const matrix = Array.from({ length }, () => new Array<number>(length).fill(0));
  for (let edge = 0; edge + 1 < length; edge += 1) {
    const weight = coupling(edge);
    matrix[edge]![edge] += weight;
    matrix[edge + 1]![edge + 1] += weight;
    matrix[edge]![edge + 1] -= weight;
    matrix[edge + 1]![edge] -= weight;
  }
  for (let row = 0; row < length; row += 1) matrix[row]![row] += anchor(row);
  return matrix;
}

function apply(matrix: Matrix, input: Readonly<Float64Array>): Float64Array {
  return Float64Array.from(matrix, (row) => row.reduce(
    (sum, coefficient, column) => sum + coefficient * input[column]!, 0));
}

function solve(matrix: Matrix, rightHandSide: Readonly<Float64Array>): Float64Array {
  const size = matrix.length;
  const rows = matrix.map((row, index) => [...row, rightHandSide[index]!]);
  for (let pivot = 0; pivot < size; pivot += 1) {
    let selected = pivot;
    for (let row = pivot + 1; row < size; row += 1) {
      if (Math.abs(rows[row]![pivot]!) > Math.abs(rows[selected]![pivot]!)) selected = row;
    }
    [rows[pivot], rows[selected]] = [rows[selected]!, rows[pivot]!];
    const diagonal = rows[pivot]![pivot]!;
    assert.ok(Math.abs(diagonal) > 1e-14);
    for (let column = pivot; column <= size; column += 1) rows[pivot]![column]! /= diagonal;
    for (let row = 0; row < size; row += 1) {
      if (row === pivot) continue;
      const factor = rows[row]![pivot]!;
      for (let column = pivot; column <= size; column += 1) {
        rows[row]![column]! -= factor * rows[pivot]![column]!;
      }
    }
  }
  return Float64Array.from(rows, (row) => row[size]!);
}

interface NumericalFixture {
  readonly name: "miniDam" | "quiescent" | "river";
  readonly accurate: Matrix;
  readonly firstOrder: Matrix;
  readonly rhs: Float64Array;
}

function fixtures(): readonly NumericalFixture[] {
  return [
    { name: "miniDam",
      accurate: fixtureMatrix(18, (edge) => 0.55 + 0.7 * ((edge * 7) % 5) / 4,
        (row) => 0.08 + 0.02 * (row % 3)),
      firstOrder: fixtureMatrix(18, () => 0.72, () => 0.22),
      rhs: Float64Array.from({ length: 18 }, (_, row) => Math.sin(0.71 * row) + 0.2),
    },
    { name: "quiescent",
      accurate: fixtureMatrix(24, () => 0.9, () => 0.16),
      firstOrder: fixtureMatrix(24, () => 0.86, () => 0.2),
      rhs: Float64Array.from({ length: 24 }, (_, row) => Math.cos(0.31 * row) * 0.1),
    },
    { name: "river",
      accurate: fixtureMatrix(32, (edge) => edge % 6 === 0 ? 1.5 : 0.48,
        (row) => row === 0 || row === 31 ? 0.5 : 0.055),
      firstOrder: fixtureMatrix(32, () => 0.62, (row) => row % 8 === 0 ? 0.35 : 0.12),
      rhs: Float64Array.from({ length: 32 }, (_, row) => 1 + 0.35 * Math.sin(row * 0.4)),
    },
  ];
}

/**
 * CPU-only shell model: each matched pair adds one accurate residual
 * correction around the fixed first-order solve. It is deliberately used
 * only to select the immutable command policy, never as runtime authority.
 */
function shellPreconditioner(fixture: NumericalFixture, depth: number) {
  return (rightHandSide: Readonly<Float64Array>) => {
    let result = solve(fixture.firstOrder, rightHandSide);
    const damping = 0.5;
    for (let sweep = 0; sweep < depth; sweep += 1) {
      const image = apply(fixture.accurate, result);
      const residual = Float64Array.from(rightHandSide,
        (value, row) => value - image[row]!);
      const correction = solve(fixture.firstOrder, residual);
      result = Float64Array.from(result,
        (value, row) => value + damping * correction[row]!);
    }
    return result;
  };
}

test("CPU shell-depth sweep keeps the selected production shell inside acceptance", () => {
  const depths = [2, 4, 6, 8] as const;
  for (const fixture of fixtures()) {
    const policy = planOctreeSolveTail(PROFILES[fixture.name]);
    const results = depths.map((depth) => solveOctreePipelinedPCG({
      rightHandSide: fixture.rhs,
      applyOperator: (input) => apply(fixture.accurate, input),
      applyPreconditioner: shellPreconditioner(fixture, depth),
    }, {
      maximumIterations: policy.encodedOuterIterations,
      relativeTolerance: OCTREE_SOLVE_TAIL_RELATIVE_TOLERANCE,
    }));
    const selected = results[depths.indexOf(
      policy.boundarySmoothingIterations as typeof depths[number])]!;
    assert.equal(selected.accepted, true,
      `${fixture.name} k=${policy.boundarySmoothingIterations} acceptance`);
    assert.ok(selected.diagnostics.relativeResidualNorm <= 1e-4);
    assert.ok(selected.diagnostics.iterations <= policy.encodedOuterIterations);
    const shallowCommands = countOctreePressureCommands({
      encodedOuterIterations: policy.encodedOuterIterations,
      fullOperatorDispatches: 5,
      mergedBandOperatorDispatches: 1,
      firstOrderSetupDispatches: 10,
      firstOrderCorrectionDispatches: 26,
      boundarySmoothingIterations: 2,
    }).encodedPressureDispatches;
    const selectedCommands = countOctreePressureCommands({
      encodedOuterIterations: policy.encodedOuterIterations,
      fullOperatorDispatches: 5,
      mergedBandOperatorDispatches: 1,
      firstOrderSetupDispatches: 10,
      firstOrderCorrectionDispatches: 26,
      boundarySmoothingIterations: policy.boundarySmoothingIterations,
    }).encodedPressureDispatches;
    assert.ok(selectedCommands > shallowCommands,
      "the paper shell must account for all additional fixed command bodies");
  }
});

test("solve-tail policy rejects malformed or out-of-envelope command inputs", () => {
  assert.throws(() => planOctreeSolveTail({ ...PROFILES.miniDam,
    finestDimensions: [24, 0, 16] }), /dimensions/);
  assert.throws(() => countOctreePressureCommands({
    encodedOuterIterations: 11, fullOperatorDispatches: 5,
    mergedBandOperatorDispatches: 1, firstOrderSetupDispatches: 10,
    firstOrderCorrectionDispatches: 26, boundarySmoothingIterations: 2,
  }), /outside the paper envelope/);
  assert.throws(() => countOctreePressureCommands({
    encodedOuterIterations: 4, fullOperatorDispatches: 5,
    mergedBandOperatorDispatches: 1, firstOrderSetupDispatches: 10,
    firstOrderCorrectionDispatches: 26, boundarySmoothingIterations: 3,
  }), /must be even/);
});

/** Dense cell cardinality of each V-cycle level, finest first. */
const vcycleLevelCells = (dimensions: readonly number[]): number[] => {
  const levels = Math.max(2, Math.ceil(Math.log2(Math.max(...dimensions))) + 1);
  return Array.from({ length: levels }, (_, level) => dimensions
    .map((value) => Math.ceil(value / 2 ** level))
    .reduce((product, value) => product * value, 1));
};

test("the V-cycle tail keeps only the levels one workgroup can own", () => {
  assert.equal(OCTREE_VCYCLE_TAIL_WORKGROUP_CELLS, 64);
  assert.equal(OCTREE_VCYCLE_MINIMUM_PARALLEL_LEVELS, 2);
  // Every domain whose level two already fitted the authored "<=63-cell tail"
  // assumption keeps its exact former command graph. This is what makes the
  // selection safe to apply to the shipping mini lane without a Gate-B run.
  for (const dimensions of [[8, 8, 8], [16, 16, 16], [12, 16, 16]]) {
    assert.equal(planOctreeVCycleParallelLevels(vcycleLevelCells(dimensions)),
      OCTREE_VCYCLE_MINIMUM_PARALLEL_LEVELS, dimensions.join("x"));
  }
  // Domains that never fitted it are widened by exactly the levels that do
  // not fit. The 32-cell symmetric-expansion oracle is the measured case: its
  // level two holds 256 cells, and leaving it in the one-workgroup tail cost
  // 29.4 ms/advance at 0.1% compute occupancy.
  assert.deepEqual(vcycleLevelCells([32, 16, 32]), [16384, 2048, 256, 32, 4, 1]);
  assert.equal(planOctreeVCycleParallelLevels(vcycleLevelCells([32, 16, 32])), 3);
  assert.equal(planOctreeVCycleParallelLevels(vcycleLevelCells([24, 16, 24])), 3);
  assert.equal(planOctreeVCycleParallelLevels(vcycleLevelCells([64, 20, 64])), 4);
  // The tail always retains at least the exact one-cell bottom it solves.
  for (const dimensions of [[8, 8, 8], [32, 16, 32], [64, 20, 64], [320, 96, 80]]) {
    const cells = vcycleLevelCells(dimensions);
    assert.ok(planOctreeVCycleParallelLevels(cells) <= cells.length - 1);
    assert.ok(planOctreeVCycleParallelLevels(cells)
      >= OCTREE_VCYCLE_MINIMUM_PARALLEL_LEVELS);
  }
});

test("V-cycle level selection rejects malformed hierarchies", () => {
  assert.throws(() => planOctreeVCycleParallelLevels([4096]), /at least two levels/);
  assert.throws(() => planOctreeVCycleParallelLevels([4096, 0]), /positive safe integers/);
  assert.throws(() => planOctreeVCycleParallelLevels([4096, 64], 0),
    /workgroup capacity must be a positive integer/);
  // The rollback lever reproduces any earlier cut, including the former two.
  assert.equal(planOctreeVCycleParallelLevels(vcycleLevelCells([32, 16, 32]), 256), 2);
});

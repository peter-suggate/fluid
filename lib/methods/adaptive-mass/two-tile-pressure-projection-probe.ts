/**
 * Pressure-only acceptance for the frozen two-tile composite grid.
 *
 * This is intentionally an executable probe, not a second pressure
 * implementation.  The solve calls the matrix-free operator from
 * two-tile-composite-grid.ts, and the projection calls its sole gradient and
 * divergence operators.  A zero-mean quotient solve removes the closed-domain
 * constant-pressure nullspace without adding a numerical diagonal shift.
 */

import {
  applyCompositeDivergence,
  applyCompositeGradient,
  applyCompositePressureOperator,
  buildTwoTileCompositeGrid,
  type CompositeAxis,
  type TwoTileCompositeGrid,
  type TwoTileResolution,
  type Vec3i,
} from "./two-tile-composite-grid";

export interface TwoTileProjectionVariantReceipt {
  readonly axis: CompositeAxis;
  readonly negativeResolution: TwoTileResolution;
  readonly positiveResolution: TwoTileResolution;
  readonly cellCount: number;
  readonly faceRowCount: number;
  readonly iterations: number;
  readonly rhsCompatibilityAbs: number;
  readonly solverRelativeResidualL2: number;
  readonly solverMaxAbsResidual: number;
  readonly preDivergenceVolumeL2: number;
  readonly postDivergenceVolumeL2: number;
  readonly preDivergenceVolumeRms: number;
  readonly postDivergenceVolumeRms: number;
  readonly postDivergenceMaxAbs: number;
  readonly divergenceReduction: number;
  readonly kineticEnergyBefore: number;
  readonly kineticEnergyAfter: number;
  readonly pressureCorrectionEnergy: number;
  readonly energyIdentityAbsError: number;
  readonly energyNonIncreasing: boolean;
}
export interface TwoTileProjectionReflectionReceipt {
  readonly axis: CompositeAxis;
  readonly preVelocityMaxAbsError: number;
  readonly pressureMaxAbsError: number;
  readonly projectedVelocityMaxAbsError: number;
  readonly preDivergenceMaxAbsError: number;
  readonly postDivergenceMaxAbsError: number;
}

export interface TwoTileProjectionThresholds {
  readonly compatibilityAbsolute: number;
  readonly solverRelative: number;
  readonly postDivergenceRelative: number;
  readonly postDivergenceAbsolute: number;
  readonly energyAbsolute: number;
  readonly reflectionAbsolute: number;
}

export interface TwoTileProjectionProbeReceipt {
  readonly schemaVersion: 1;
  readonly milestone: "M1-pressure-only-two-tile-projection";
  readonly limitation: string;
  readonly solver: {
    readonly kind: "matrix-free-zero-mean-conjugate-gradient";
    readonly operatorSource: "lib/methods/adaptive-mass/two-tile-composite-grid.ts";
    readonly relativeTolerance: number;
    readonly maximumIterations: number;
  };
  readonly thresholds: TwoTileProjectionThresholds;
  readonly variants: readonly TwoTileProjectionVariantReceipt[];
  readonly reflections: readonly TwoTileProjectionReflectionReceipt[];
  readonly failures: readonly string[];
  readonly passed: boolean;
}

interface ProjectionExecution {
  readonly grid: TwoTileCompositeGrid;
  readonly receipt: TwoTileProjectionVariantReceipt;
  readonly velocityBefore: Float64Array;
  readonly pressure: Float64Array;
  readonly velocityAfter: Float64Array;
  readonly divergenceBefore: Float64Array;
  readonly divergenceAfter: Float64Array;
}

interface QuotientCgResult {
  readonly solution: Float64Array;
  readonly iterations: number;
  readonly relativeResidualL2: number;
  readonly maxAbsResidual: number;
}

const DEFAULT_THRESHOLDS: TwoTileProjectionThresholds = {
  compatibilityAbsolute: 1e-12,
  solverRelative: 2e-10,
  postDivergenceRelative: 2e-10,
  postDivergenceAbsolute: 2e-10,
  energyAbsolute: 2e-11,
  reflectionAbsolute: 2e-10,
};

function dot(left: ArrayLike<number>, right: ArrayLike<number>): number {
  if (left.length !== right.length) throw new RangeError("dot vector lengths differ");
  let sum = 0;
  for (let i = 0; i < left.length; i += 1) sum += left[i] * right[i];
  return sum;
}

function l2(values: ArrayLike<number>): number {
  return Math.sqrt(dot(values, values));
}

function maxAbs(values: ArrayLike<number>): number {
  let maximum = 0;
  for (let i = 0; i < values.length; i += 1) {
    maximum = Math.max(maximum, Math.abs(values[i]));
  }
  return maximum;
}

function subtractArithmeticMean(values: Float64Array): void {
  let sum = 0;
  for (const value of values) sum += value;
  const mean = sum / values.length;
  for (let i = 0; i < values.length; i += 1) values[i] -= mean;
}

function solvePressureOnZeroMeanQuotient(
  grid: TwoTileCompositeGrid,
  rhs: Float64Array,
  relativeTolerance: number,
  maximumIterations: number,
): QuotientCgResult {
  const rhsNorm = l2(rhs);
  if (!(rhsNorm > 0)) {
    return {
      solution: new Float64Array(rhs.length),
      iterations: 0,
      relativeResidualL2: 0,
      maxAbsResidual: 0,
    };
  }

  const solution = new Float64Array(rhs.length);
  const residual = rhs.slice();
  subtractArithmeticMean(residual);
  const direction = residual.slice();
  let residualSquared = dot(residual, residual);
  let iterations = 0;

  while (iterations < maximumIterations
    && Math.sqrt(residualSquared) > relativeTolerance * rhsNorm) {
    const operatorDirection = applyCompositePressureOperator(grid, direction);
    // A maps the quotient to itself analytically. Removing roundoff in its
    // constant component prevents long solves from slowly leaking gauge.
    subtractArithmeticMean(operatorDirection);
    const denominator = dot(direction, operatorDirection);
    if (!(denominator > 0) || !Number.isFinite(denominator)) {
      throw new Error(`composite pressure CG lost positive curvature at iteration ${iterations}`);
    }
    const alpha = residualSquared / denominator;
    for (let i = 0; i < solution.length; i += 1) {
      solution[i] += alpha * direction[i];
      residual[i] -= alpha * operatorDirection[i];
    }
    subtractArithmeticMean(solution);
    subtractArithmeticMean(residual);
    const nextResidualSquared = dot(residual, residual);
    iterations += 1;
    if (Math.sqrt(nextResidualSquared) <= relativeTolerance * rhsNorm) {
      residualSquared = nextResidualSquared;
      break;
    }
    const beta = nextResidualSquared / residualSquared;
    for (let i = 0; i < direction.length; i += 1) {
      direction[i] = residual[i] + beta * direction[i];
    }
    subtractArithmeticMean(direction);
    residualSquared = nextResidualSquared;
  }

  const trueResidual = applyCompositePressureOperator(grid, solution);
  for (let i = 0; i < trueResidual.length; i += 1) trueResidual[i] -= rhs[i];
  subtractArithmeticMean(trueResidual);
  return {
    solution,
    iterations,
    relativeResidualL2: l2(trueResidual) / rhsNorm,
    maxAbsResidual: maxAbs(trueResidual),
  };
}

function manufacturedVelocity(
  grid: TwoTileCompositeGrid,
  reflectToFineNegative: boolean,
): Float64Array {
  return Float64Array.from(grid.gradientRows, (row) => {
    const position = [...row.center] as [number, number, number];
    let componentSign = 1;
    if (reflectToFineNegative) {
      position[grid.axis] = 2 * grid.tileWidth - position[grid.axis];
      if (row.axis === grid.axis) componentSign = -1;
    }
    const [x, y, z] = position;
    const component = row.axis;
    const vectorComponent = component === 0
      ? Math.sin(0.37 + 1.13 * x - 0.29 * y + 0.17 * z) + 0.11 * y * z
      : component === 1
        ? Math.cos(0.23 - 0.31 * x + 0.97 * y + 0.19 * z) - 0.07 * x * z
        : Math.sin(0.41 + 0.27 * x + 0.13 * y - 1.07 * z) + 0.09 * x * y;
    return componentSign * vectorComponent;
  });
}

function volumeDivergenceNorms(grid: TwoTileCompositeGrid, divergence: ArrayLike<number>): {
  l2: number;
  rms: number;
} {
  let weightedSquared = 0;
  let totalVolume = 0;
  for (const cell of grid.cells) {
    weightedSquared += cell.volume * divergence[cell.id] * divergence[cell.id];
    totalVolume += cell.volume;
  }
  return { l2: Math.sqrt(weightedSquared), rms: Math.sqrt(weightedSquared / totalVolume) };
}

function faceEnergy(grid: TwoTileCompositeGrid, velocity: ArrayLike<number>): number {
  let energy = 0;
  for (const row of grid.gradientRows) {
    energy += 0.5 * row.dualWeight * velocity[row.id] * velocity[row.id];
  }
  return energy;
}

function executeProjection(
  axis: CompositeAxis,
  negativeResolution: TwoTileResolution,
  positiveResolution: TwoTileResolution,
  relativeTolerance: number,
  maximumIterations: number,
): ProjectionExecution {
  const grid = buildTwoTileCompositeGrid({ axis, negativeResolution, positiveResolution });
  const reflectToFineNegative = negativeResolution === 4 && positiveResolution === 8;
  const velocityBefore = manufacturedVelocity(grid, reflectToFineNegative);
  const divergenceBefore = applyCompositeDivergence(grid, velocityBefore);

  // D u = -M^-1 G^T W u, therefore A p = G^T W u = -M D u.
  // Forming the RHS through the shared divergence makes compatibility an
  // observable invariant rather than maintaining a second transpose stencil.
  const rhs = Float64Array.from(
    divergenceBefore,
    (value, cellId) => -grid.cells[cellId].volume * value,
  );
  let rhsSum = 0;
  for (const value of rhs) rhsSum += value;
  const rhsCompatibilityAbs = Math.abs(rhsSum);
  subtractArithmeticMean(rhs);

  const solve = solvePressureOnZeroMeanQuotient(
    grid,
    rhs,
    relativeTolerance,
    maximumIterations,
  );
  const pressureGradient = applyCompositeGradient(grid, solve.solution);
  const velocityAfter = Float64Array.from(
    velocityBefore,
    (value, rowId) => value - pressureGradient[rowId],
  );
  const divergenceAfter = applyCompositeDivergence(grid, velocityAfter);
  const beforeNorms = volumeDivergenceNorms(grid, divergenceBefore);
  const afterNorms = volumeDivergenceNorms(grid, divergenceAfter);
  const kineticEnergyBefore = faceEnergy(grid, velocityBefore);
  const kineticEnergyAfter = faceEnergy(grid, velocityAfter);
  const pressureCorrectionEnergy = faceEnergy(grid, pressureGradient);

  return {
    grid,
    velocityBefore,
    pressure: solve.solution,
    velocityAfter,
    divergenceBefore,
    divergenceAfter,
    receipt: {
      axis,
      negativeResolution,
      positiveResolution,
      cellCount: grid.cells.length,
      faceRowCount: grid.gradientRows.length,
      iterations: solve.iterations,
      rhsCompatibilityAbs,
      solverRelativeResidualL2: solve.relativeResidualL2,
      solverMaxAbsResidual: solve.maxAbsResidual,
      preDivergenceVolumeL2: beforeNorms.l2,
      postDivergenceVolumeL2: afterNorms.l2,
      preDivergenceVolumeRms: beforeNorms.rms,
      postDivergenceVolumeRms: afterNorms.rms,
      postDivergenceMaxAbs: maxAbs(divergenceAfter),
      divergenceReduction: afterNorms.l2 / beforeNorms.l2,
      kineticEnergyBefore,
      kineticEnergyAfter,
      pressureCorrectionEnergy,
      energyIdentityAbsError: Math.abs(
        kineticEnergyBefore - kineticEnergyAfter - pressureCorrectionEnergy
      ),
      energyNonIncreasing: kineticEnergyAfter <= kineticEnergyBefore,
    },
  };
}

function reflectionCellMap(source: TwoTileCompositeGrid, target: TwoTileCompositeGrid): Int32Array {
  const key = (resolution: number, center: Vec3i): string => `${resolution}:${center.join(",")}`;
  const targetCells = new Map<string, number>();
  for (const cell of target.cells) targetCells.set(key(cell.resolution, cell.centerFineHalf), cell.id);
  const domainFineHalf = 4 * source.finestResolution;
  return Int32Array.from(source.cells, (cell) => {
    const center = [...cell.centerFineHalf] as [number, number, number];
    center[source.axis] = domainFineHalf - center[source.axis];
    const targetId = targetCells.get(key(cell.resolution, center));
    if (targetId === undefined) throw new Error(`missing reflected cell ${key(cell.resolution, center)}`);
    return targetId;
  });
}

function reflectionRowMap(source: TwoTileCompositeGrid, target: TwoTileCompositeGrid): Int32Array {
  const key = (axis: CompositeAxis, center: Vec3i): string => `${axis}:${center.join(",")}`;
  const targetRows = new Map<string, number>();
  for (const row of target.gradientRows) targetRows.set(key(row.axis, row.centerFineHalf), row.id);
  const domainFineHalf = 4 * source.finestResolution;
  return Int32Array.from(source.gradientRows, (row) => {
    const center = [...row.centerFineHalf] as [number, number, number];
    center[source.axis] = domainFineHalf - center[source.axis];
    const targetId = targetRows.get(key(row.axis, center));
    if (targetId === undefined) throw new Error(`missing reflected row ${key(row.axis, center)}`);
    return targetId;
  });
}

function compareReflections(
  source: ProjectionExecution,
  target: ProjectionExecution,
): TwoTileProjectionReflectionReceipt {
  const cellMap = reflectionCellMap(source.grid, target.grid);
  const rowMap = reflectionRowMap(source.grid, target.grid);
  let preVelocityMaxAbsError = 0;
  let pressureMaxAbsError = 0;
  let projectedVelocityMaxAbsError = 0;
  let preDivergenceMaxAbsError = 0;
  let postDivergenceMaxAbsError = 0;
  for (let sourceId = 0; sourceId < cellMap.length; sourceId += 1) {
    const targetId = cellMap[sourceId];
    pressureMaxAbsError = Math.max(
      pressureMaxAbsError,
      Math.abs(source.pressure[sourceId] - target.pressure[targetId]),
    );
    preDivergenceMaxAbsError = Math.max(
      preDivergenceMaxAbsError,
      Math.abs(source.divergenceBefore[sourceId] - target.divergenceBefore[targetId]),
    );
    postDivergenceMaxAbsError = Math.max(
      postDivergenceMaxAbsError,
      Math.abs(source.divergenceAfter[sourceId] - target.divergenceAfter[targetId]),
    );
  }
  for (let sourceId = 0; sourceId < rowMap.length; sourceId += 1) {
    const targetId = rowMap[sourceId];
    const sign = source.grid.gradientRows[sourceId].axis === source.grid.axis ? -1 : 1;
    preVelocityMaxAbsError = Math.max(
      preVelocityMaxAbsError,
      Math.abs(sign * source.velocityBefore[sourceId] - target.velocityBefore[targetId]),
    );
    projectedVelocityMaxAbsError = Math.max(
      projectedVelocityMaxAbsError,
      Math.abs(sign * source.velocityAfter[sourceId] - target.velocityAfter[targetId]),
    );
  }
  return {
    axis: source.grid.axis,
    preVelocityMaxAbsError,
    pressureMaxAbsError,
    projectedVelocityMaxAbsError,
    preDivergenceMaxAbsError,
    postDivergenceMaxAbsError,
  };
}

/** Execute a real manufactured pressure projection on every M1 topology. */
export function probeTwoTilePressureProjection(options: {
  readonly relativeTolerance?: number;
  readonly maximumIterations?: number;
  readonly thresholds?: Partial<TwoTileProjectionThresholds>;
} = {}): TwoTileProjectionProbeReceipt {
  const relativeTolerance = options.relativeTolerance ?? 2e-12;
  const maximumIterations = options.maximumIterations ?? 1024;
  const thresholds = { ...DEFAULT_THRESHOLDS, ...options.thresholds };
  if (!(relativeTolerance > 0) || !Number.isFinite(relativeTolerance)) {
    throw new RangeError("relativeTolerance must be finite and positive");
  }
  if (!Number.isInteger(maximumIterations) || maximumIterations <= 0) {
    throw new RangeError("maximumIterations must be a positive integer");
  }

  const executions: ProjectionExecution[] = [];
  const reflections: TwoTileProjectionReflectionReceipt[] = [];
  for (const axis of [0, 1, 2] as const) {
    executions.push(executeProjection(axis, 8, 8, relativeTolerance, maximumIterations));
    executions.push(executeProjection(axis, 4, 4, relativeTolerance, maximumIterations));
    const fineNegative = executeProjection(axis, 8, 4, relativeTolerance, maximumIterations);
    const finePositive = executeProjection(axis, 4, 8, relativeTolerance, maximumIterations);
    executions.push(fineNegative, finePositive);
    reflections.push(compareReflections(fineNegative, finePositive));
  }

  const failures: string[] = [];
  for (const { receipt } of executions) {
    const label = `axis=${receipt.axis} ${receipt.negativeResolution}+${receipt.positiveResolution}`;
    if (receipt.rhsCompatibilityAbs > thresholds.compatibilityAbsolute) {
      failures.push(`${label}: incompatible RHS ${receipt.rhsCompatibilityAbs}`);
    }
    if (receipt.solverRelativeResidualL2 > thresholds.solverRelative) {
      failures.push(`${label}: solver residual ${receipt.solverRelativeResidualL2}`);
    }
    if (receipt.postDivergenceVolumeL2 > thresholds.postDivergenceAbsolute
      && receipt.divergenceReduction > thresholds.postDivergenceRelative) {
      failures.push(`${label}: post divergence ${receipt.postDivergenceVolumeL2}`);
    }
    if (!receipt.energyNonIncreasing
      || receipt.kineticEnergyAfter - receipt.kineticEnergyBefore > thresholds.energyAbsolute) {
      failures.push(`${label}: projection increased face energy`);
    }
    if (receipt.energyIdentityAbsError > thresholds.energyAbsolute) {
      failures.push(`${label}: energy identity error ${receipt.energyIdentityAbsError}`);
    }
    if (receipt.iterations >= maximumIterations) {
      failures.push(`${label}: CG exhausted ${maximumIterations} iterations`);
    }
  }
  for (const reflection of reflections) {
    for (const [metric, value] of Object.entries(reflection)) {
      if (metric !== "axis" && value > thresholds.reflectionAbsolute) {
        failures.push(`axis=${reflection.axis} reflection ${metric} ${value}`);
      }
    }
  }

  return {
    schemaVersion: 1,
    milestone: "M1-pressure-only-two-tile-projection",
    limitation: "Manufactured closed-domain pressure projection only: no free surface, CM12 transport, time integration, GPU solve, or production multigrid is exercised.",
    solver: {
      kind: "matrix-free-zero-mean-conjugate-gradient",
      operatorSource: "lib/methods/adaptive-mass/two-tile-composite-grid.ts",
      relativeTolerance,
      maximumIterations,
    },
    thresholds,
    variants: executions.map(({ receipt }) => receipt),
    reflections,
    failures,
    passed: failures.length === 0,
  };
}

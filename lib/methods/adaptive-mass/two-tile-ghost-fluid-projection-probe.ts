/**
 * Free-surface/ghost-fluid pressure acceptance on the frozen two-tile grid.
 *
 * The base CompositeGradientRow topology remains authoritative. For a row cut
 * by the surface, this probe drops air-pressure terms (Dirichlet p_air = 0),
 * computes one coefficient-magnitude-weighted CM12 theta for the whole row,
 * and contributes (W/theta) b b^T for the remaining liquid coefficient vector
 * b. The projected row velocity is u - (b.p)/theta. This is SPD, is an
 * orthogonal projection in the W*theta face metric, and reduces to the usual
 * two-cell ghost-fluid diagonal W*c^2/theta.
 *
 * For a five-term coarse/fine seam row this is deliberately a single shared
 * interface fraction, not a claim that the four subface intersections have
 * been reconstructed. A production Ando--Batty/cut-cell formulation may need
 * per-subface geometry and a higher-rank local block.
 */

import { cm12GhostFluidTheta } from "../../core/cm12-numerics";
import {
  buildTwoTileCompositeGrid,
  type CompositeAxis,
  type TwoTileCompositeGrid,
  type TwoTileResolution,
  type Vec3i,
} from "./two-tile-composite-grid";

type GhostFluidCase = "manufactured" | "hydrostatic-like";

interface LiquidTerm {
  readonly unknownId: number;
  readonly coefficient: number;
}
interface LiquidRow {
  readonly rowId: number;
  readonly terms: readonly LiquidTerm[];
  readonly theta: number;
  readonly rawTheta: number;
  readonly cut: boolean;
  readonly thetaClamped: boolean;
}

interface LiquidSystem {
  readonly grid: TwoTileCompositeGrid;
  readonly phi: Float64Array;
  readonly liquidCellIds: Int32Array;
  readonly unknownByCell: Int32Array;
  readonly rows: readonly LiquidRow[];
  readonly cutRowCount: number;
  readonly cutSeamRowCount: number;
  readonly cutFiveTermSeamRowCount: number;
  readonly thetaClampCount: number;
  readonly minimumTheta: number;
  readonly maximumTheta: number;
}

export interface GhostFluidProjectionVariantReceipt {
  readonly case: GhostFluidCase;
  readonly axis: CompositeAxis;
  readonly negativeResolution: TwoTileResolution;
  readonly positiveResolution: TwoTileResolution;
  readonly liquidCellCount: number;
  readonly activeRowCount: number;
  readonly cutRowCount: number;
  readonly cutSeamRowCount: number;
  readonly cutFiveTermSeamRowCount: number;
  readonly thetaMinimum: number;
  readonly thetaMaximum: number;
  readonly thetaClampCount: number;
  readonly sampledMinimumRayleigh: number;
  readonly minimumCgRayleigh: number;
  readonly iterations: number;
  readonly solverRelativeResidualL2: number;
  readonly pressureEquationMaxAbsResidual: number;
  readonly preLiquidDivergenceVolumeL2: number;
  readonly postLiquidDivergenceVolumeL2: number;
  readonly divergenceReduction: number;
  readonly metricEnergyBefore: number;
  readonly metricEnergyAfter: number;
  readonly pressureCorrectionEnergy: number;
  readonly energyIdentityAbsError: number;
  readonly energyNonIncreasing: boolean;
  readonly hydrostaticPressureMaxAbsError?: number;
}

export interface GhostFluidProjectionReflectionReceipt {
  readonly case: GhostFluidCase;
  readonly axis: CompositeAxis;
  readonly phiMaxAbsError: number;
  readonly thetaMaxAbsError: number;
  readonly pressureMaxAbsError: number;
  readonly projectedVelocityMaxAbsError: number;
  readonly equationResidualMaxAbsError: number;
}

export interface GhostFluidProjectionProbeReceipt {
  readonly schemaVersion: 1;
  readonly milestone: "M1-two-tile-ghost-fluid-projection";
  readonly formulation: {
    readonly topologySource: "lib/methods/adaptive-mass/two-tile-composite-grid.ts";
    readonly thetaSource: "lib/core/cm12-numerics.ts#cm12GhostFluidTheta";
    readonly mixedRowWeighting: string;
    readonly limitation: string;
  };
  readonly thresholds: {
    readonly solverRelative: number;
    readonly postDivergenceRelative: number;
    readonly postDivergenceAbsolute: number;
    readonly energyAbsolute: number;
    readonly reflectionAbsolute: number;
    readonly minimumPositiveRayleigh: number;
  };
  readonly variants: readonly GhostFluidProjectionVariantReceipt[];
  readonly reflections: readonly GhostFluidProjectionReflectionReceipt[];
  readonly failures: readonly string[];
  readonly passed: boolean;
}

interface CgResult {
  readonly solution: Float64Array;
  readonly iterations: number;
  readonly relativeResidual: number;
  readonly maximumResidual: number;
  readonly minimumRayleigh: number;
}

interface Execution {
  readonly system: LiquidSystem;
  readonly receipt: GhostFluidProjectionVariantReceipt;
  readonly pressure: Float64Array;
  readonly velocityAfter: Float64Array;
  readonly equationResidual: Float64Array;
}

const DENOMINATOR_EPSILON = 1e-12;
const RELATIVE_SOLVE_TOLERANCE = 2e-12;
const MAXIMUM_ITERATIONS = 2048;

function dot(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < left.length; i += 1) sum += left[i] * right[i];
  return sum;
}

function l2(values: ArrayLike<number>): number {
  return Math.sqrt(dot(values, values));
}

function maxAbs(values: ArrayLike<number>): number {
  let maximum = 0;
  for (let i = 0; i < values.length; i += 1) maximum = Math.max(maximum, Math.abs(values[i]));
  return maximum;
}

function canonicalPosition(
  grid: TwoTileCompositeGrid,
  position: readonly [number, number, number],
): [number, number, number] {
  const result = [...position] as [number, number, number];
  if (grid.negativeResolution === 4 && grid.positiveResolution === 8) {
    result[grid.axis] = 2 * grid.tileWidth - result[grid.axis];
  }
  return result;
}

function surfacePhi(grid: TwoTileCompositeGrid, position: readonly [number, number, number]): number {
  const canonical = canonicalPosition(grid, position);
  const normal = canonical[grid.axis] / grid.tileWidth;
  const tangentU = canonical[grid.tangentialAxes[0]] / grid.tileWidth;
  const tangentV = canonical[grid.tangentialAxes[1]] / grid.tileWidth;
  return 0.82 * (normal - 1) + 0.56 * (tangentU - 0.5) - 0.34 * (tangentV - 0.5);
}

function buildLiquidSystem(grid: TwoTileCompositeGrid): LiquidSystem {
  const phi = Float64Array.from(grid.cells, (cell) => surfacePhi(grid, cell.center));
  const liquidCellIds = Int32Array.from(
    grid.cells.filter((cell) => phi[cell.id] <= 0),
    (cell) => cell.id,
  );
  const unknownByCell = new Int32Array(grid.cells.length).fill(-1);
  for (let unknownId = 0; unknownId < liquidCellIds.length; unknownId += 1) {
    unknownByCell[liquidCellIds[unknownId]] = unknownId;
  }

  const rows: LiquidRow[] = [];
  let cutRowCount = 0;
  let cutSeamRowCount = 0;
  let cutFiveTermSeamRowCount = 0;
  let thetaClampCount = 0;
  let minimumTheta = 1;
  let maximumTheta = 0;
  for (const row of grid.gradientRows) {
    const liquidTerms = row.terms.filter((term) => unknownByCell[term.cellId] >= 0);
    if (liquidTerms.length === 0) continue;
    const airTerms = row.terms.filter((term) => unknownByCell[term.cellId] < 0);
    const cut = airTerms.length > 0;
    let rawTheta = 1;
    let theta = 1;
    if (cut) {
      let liquidPhiWeighted = 0;
      let liquidWeight = 0;
      for (const term of liquidTerms) {
        const weight = Math.abs(term.coefficient);
        liquidPhiWeighted += weight * phi[term.cellId];
        liquidWeight += weight;
      }
      let airPhiWeighted = 0;
      let airWeight = 0;
      for (const term of airTerms) {
        const weight = Math.abs(term.coefficient);
        airPhiWeighted += weight * phi[term.cellId];
        airWeight += weight;
      }
      const liquidPhi = liquidPhiWeighted / liquidWeight;
      const airPhi = airPhiWeighted / airWeight;
      rawTheta = Math.abs(liquidPhi)
        / Math.max(Math.abs(liquidPhi) + Math.abs(airPhi), DENOMINATOR_EPSILON);
      theta = cm12GhostFluidTheta(liquidPhi, airPhi, DENOMINATOR_EPSILON);
      cutRowCount += 1;
      if (row.kind === "seam") cutSeamRowCount += 1;
      if (row.kind === "seam" && row.terms.length === 5) cutFiveTermSeamRowCount += 1;
      if (theta !== rawTheta) thetaClampCount += 1;
    }
    minimumTheta = Math.min(minimumTheta, theta);
    maximumTheta = Math.max(maximumTheta, theta);
    rows.push({
      rowId: row.id,
      terms: liquidTerms.map((term) => ({
        unknownId: unknownByCell[term.cellId],
        coefficient: term.coefficient,
      })),
      theta,
      rawTheta,
      cut,
      thetaClamped: theta !== rawTheta,
    });
  }
  return {
    grid,
    phi,
    liquidCellIds,
    unknownByCell,
    rows,
    cutRowCount,
    cutSeamRowCount,
    cutFiveTermSeamRowCount,
    thetaClampCount,
    minimumTheta,
    maximumTheta,
  };
}

function applyLiquidOperator(
  system: LiquidSystem,
  pressure: ArrayLike<number>,
): Float64Array {
  const output = new Float64Array(system.liquidCellIds.length);
  for (const row of system.rows) {
    let jump = 0;
    for (const term of row.terms) jump += term.coefficient * pressure[term.unknownId];
    const weightedJump = system.grid.gradientRows[row.rowId].dualWeight * jump / row.theta;
    for (const term of row.terms) output[term.unknownId] += term.coefficient * weightedJump;
  }
  return output;
}

function liquidPressureEnergy(system: LiquidSystem, pressure: ArrayLike<number>): number {
  return dot(pressure, applyLiquidOperator(system, pressure));
}

function sampledMinimumRayleigh(system: LiquidSystem): number {
  let minimum = Number.POSITIVE_INFINITY;
  for (let mode = 1; mode <= 5; mode += 1) {
    const sample = Float64Array.from(system.liquidCellIds, (cellId) => {
      const [x, y, z] = canonicalPosition(system.grid, system.grid.cells[cellId].center);
      return Math.sin(0.19 + mode * 0.47 * x + 0.31 * y - 0.23 * z)
        + 0.07 * mode * x * z;
    });
    minimum = Math.min(minimum, liquidPressureEnergy(system, sample) / dot(sample, sample));
  }
  return minimum;
}

function solveSpd(system: LiquidSystem, rhs: Float64Array): CgResult {
  const rhsNorm = l2(rhs);
  const solution = new Float64Array(rhs.length);
  const residual = rhs.slice();
  const direction = residual.slice();
  let residualSquared = dot(residual, residual);
  let iterations = 0;
  let minimumRayleigh = Number.POSITIVE_INFINITY;
  while (iterations < MAXIMUM_ITERATIONS
    && Math.sqrt(residualSquared) > RELATIVE_SOLVE_TOLERANCE * rhsNorm) {
    const applied = applyLiquidOperator(system, direction);
    const directionSquared = dot(direction, direction);
    const curvature = dot(direction, applied);
    minimumRayleigh = Math.min(minimumRayleigh, curvature / directionSquared);
    if (!(curvature > 0) || !Number.isFinite(curvature)) {
      throw new Error(`ghost-fluid CG lost SPD curvature at iteration ${iterations}`);
    }
    const alpha = residualSquared / curvature;
    for (let i = 0; i < solution.length; i += 1) {
      solution[i] += alpha * direction[i];
      residual[i] -= alpha * applied[i];
    }
    const nextResidualSquared = dot(residual, residual);
    iterations += 1;
    if (Math.sqrt(nextResidualSquared) <= RELATIVE_SOLVE_TOLERANCE * rhsNorm) {
      residualSquared = nextResidualSquared;
      break;
    }
    const beta = nextResidualSquared / residualSquared;
    for (let i = 0; i < direction.length; i += 1) {
      direction[i] = residual[i] + beta * direction[i];
    }
    residualSquared = nextResidualSquared;
  }
  const trueResidual = applyLiquidOperator(system, solution);
  for (let i = 0; i < trueResidual.length; i += 1) trueResidual[i] -= rhs[i];
  return {
    solution,
    iterations,
    relativeResidual: l2(trueResidual) / rhsNorm,
    maximumResidual: maxAbs(trueResidual),
    minimumRayleigh,
  };
}

function manufacturedVelocity(system: LiquidSystem): Float64Array {
  return Float64Array.from(system.grid.gradientRows, (row) => {
    const canonical = canonicalPosition(system.grid, row.center);
    const [x, y, z] = canonical;
    const sign = system.grid.negativeResolution === 4
      && system.grid.positiveResolution === 8
      && row.axis === system.grid.axis ? -1 : 1;
    const component = row.axis === 0
      ? Math.sin(0.37 + 1.13 * x - 0.29 * y + 0.17 * z)
      : row.axis === 1
        ? Math.cos(0.23 - 0.31 * x + 0.97 * y + 0.19 * z)
        : Math.sin(0.41 + 0.27 * x + 0.13 * y - 1.07 * z);
    return sign * component;
  });
}

function hydrostaticPressure(system: LiquidSystem): Float64Array {
  return Float64Array.from(system.liquidCellIds, (cellId) => Math.max(0, -system.phi[cellId]));
}

function pressureCorrection(system: LiquidSystem, pressure: ArrayLike<number>): Float64Array {
  const correction = new Float64Array(system.grid.gradientRows.length);
  for (const row of system.rows) {
    let jump = 0;
    for (const term of row.terms) jump += term.coefficient * pressure[term.unknownId];
    correction[row.rowId] = jump / row.theta;
  }
  return correction;
}

function assembleLiquidRhs(system: LiquidSystem, velocity: ArrayLike<number>): Float64Array {
  const rhs = new Float64Array(system.liquidCellIds.length);
  for (const row of system.rows) {
    const weightedVelocity = system.grid.gradientRows[row.rowId].dualWeight * velocity[row.rowId];
    for (const term of row.terms) rhs[term.unknownId] += term.coefficient * weightedVelocity;
  }
  return rhs;
}

function liquidDivergenceNorm(system: LiquidSystem, equationResidual: ArrayLike<number>): number {
  let weightedSquared = 0;
  for (let unknownId = 0; unknownId < system.liquidCellIds.length; unknownId += 1) {
    const cell = system.grid.cells[system.liquidCellIds[unknownId]];
    const divergence = -equationResidual[unknownId] / cell.volume;
    weightedSquared += cell.volume * divergence * divergence;
  }
  return Math.sqrt(weightedSquared);
}

function metricVelocityEnergy(system: LiquidSystem, velocity: ArrayLike<number>): number {
  let energy = 0;
  for (const row of system.rows) {
    energy += 0.5 * system.grid.gradientRows[row.rowId].dualWeight
      * row.theta * velocity[row.rowId] * velocity[row.rowId];
  }
  return energy;
}

function execute(
  axis: CompositeAxis,
  negativeResolution: TwoTileResolution,
  positiveResolution: TwoTileResolution,
  caseName: GhostFluidCase,
): Execution {
  const system = buildLiquidSystem(buildTwoTileCompositeGrid({
    axis,
    negativeResolution,
    positiveResolution,
  }));
  const referencePressure = caseName === "hydrostatic-like" ? hydrostaticPressure(system) : undefined;
  const velocityBefore = referencePressure
    ? pressureCorrection(system, referencePressure)
    : manufacturedVelocity(system);
  const rhs = assembleLiquidRhs(system, velocityBefore);
  const preLiquidDivergenceVolumeL2 = liquidDivergenceNorm(system, rhs);
  const solve = solveSpd(system, rhs);
  const correction = pressureCorrection(system, solve.solution);
  const velocityAfter = Float64Array.from(
    velocityBefore,
    (value, rowId) => value - correction[rowId],
  );
  const equationResidual = assembleLiquidRhs(system, velocityAfter);
  const postLiquidDivergenceVolumeL2 = liquidDivergenceNorm(system, equationResidual);
  const metricEnergyBefore = metricVelocityEnergy(system, velocityBefore);
  const metricEnergyAfter = metricVelocityEnergy(system, velocityAfter);
  const pressureCorrectionEnergy = metricVelocityEnergy(system, correction);
  return {
    system,
    pressure: solve.solution,
    velocityAfter,
    equationResidual,
    receipt: {
      case: caseName,
      axis,
      negativeResolution,
      positiveResolution,
      liquidCellCount: system.liquidCellIds.length,
      activeRowCount: system.rows.length,
      cutRowCount: system.cutRowCount,
      cutSeamRowCount: system.cutSeamRowCount,
      cutFiveTermSeamRowCount: system.cutFiveTermSeamRowCount,
      thetaMinimum: system.minimumTheta,
      thetaMaximum: system.maximumTheta,
      thetaClampCount: system.thetaClampCount,
      sampledMinimumRayleigh: sampledMinimumRayleigh(system),
      minimumCgRayleigh: solve.minimumRayleigh,
      iterations: solve.iterations,
      solverRelativeResidualL2: solve.relativeResidual,
      pressureEquationMaxAbsResidual: solve.maximumResidual,
      preLiquidDivergenceVolumeL2,
      postLiquidDivergenceVolumeL2,
      divergenceReduction: postLiquidDivergenceVolumeL2 / preLiquidDivergenceVolumeL2,
      metricEnergyBefore,
      metricEnergyAfter,
      pressureCorrectionEnergy,
      energyIdentityAbsError: Math.abs(
        metricEnergyBefore - metricEnergyAfter - pressureCorrectionEnergy
      ),
      energyNonIncreasing: metricEnergyAfter <= metricEnergyBefore,
      ...(referencePressure ? {
        hydrostaticPressureMaxAbsError: maxAbs(Float64Array.from(
          solve.solution,
          (value, unknownId) => value - referencePressure[unknownId],
        )),
      } : {}),
    },
  };
}

function reflectedCellMap(source: TwoTileCompositeGrid, target: TwoTileCompositeGrid): Int32Array {
  const key = (resolution: number, center: Vec3i): string => `${resolution}:${center.join(",")}`;
  const targetByPosition = new Map<string, number>();
  for (const cell of target.cells) targetByPosition.set(key(cell.resolution, cell.centerFineHalf), cell.id);
  const domainFineHalf = 4 * source.finestResolution;
  return Int32Array.from(source.cells, (cell) => {
    const center = [...cell.centerFineHalf] as [number, number, number];
    center[source.axis] = domainFineHalf - center[source.axis];
    const targetId = targetByPosition.get(key(cell.resolution, center));
    if (targetId === undefined) throw new Error("reflected liquid cell is missing");
    return targetId;
  });
}

function reflectedRowMap(source: TwoTileCompositeGrid, target: TwoTileCompositeGrid): Int32Array {
  const key = (axis: CompositeAxis, center: Vec3i): string => `${axis}:${center.join(",")}`;
  const targetByPosition = new Map<string, number>();
  for (const row of target.gradientRows) targetByPosition.set(key(row.axis, row.centerFineHalf), row.id);
  const domainFineHalf = 4 * source.finestResolution;
  return Int32Array.from(source.gradientRows, (row) => {
    const center = [...row.centerFineHalf] as [number, number, number];
    center[source.axis] = domainFineHalf - center[source.axis];
    const targetId = targetByPosition.get(key(row.axis, center));
    if (targetId === undefined) throw new Error("reflected ghost-fluid row is missing");
    return targetId;
  });
}

function compareReflection(source: Execution, target: Execution): GhostFluidProjectionReflectionReceipt {
  const cellMap = reflectedCellMap(source.system.grid, target.system.grid);
  const rowMap = reflectedRowMap(source.system.grid, target.system.grid);
  let phiMaxAbsError = 0;
  let thetaMaxAbsError = 0;
  let pressureMaxAbsError = 0;
  let projectedVelocityMaxAbsError = 0;
  let equationResidualMaxAbsError = 0;
  const targetRows = new Map(target.system.rows.map((row) => [row.rowId, row]));
  for (const sourceRow of source.system.rows) {
    const targetRow = targetRows.get(rowMap[sourceRow.rowId]);
    if (!targetRow) throw new Error("reflected active ghost-fluid row is missing");
    thetaMaxAbsError = Math.max(thetaMaxAbsError, Math.abs(sourceRow.theta - targetRow.theta));
  }
  for (let sourceCellId = 0; sourceCellId < cellMap.length; sourceCellId += 1) {
    const targetCellId = cellMap[sourceCellId];
    phiMaxAbsError = Math.max(
      phiMaxAbsError,
      Math.abs(source.system.phi[sourceCellId] - target.system.phi[targetCellId]),
    );
    const sourceUnknown = source.system.unknownByCell[sourceCellId];
    const targetUnknown = target.system.unknownByCell[targetCellId];
    if (sourceUnknown >= 0 && targetUnknown >= 0) {
      pressureMaxAbsError = Math.max(
        pressureMaxAbsError,
        Math.abs(source.pressure[sourceUnknown] - target.pressure[targetUnknown]),
      );
      equationResidualMaxAbsError = Math.max(
        equationResidualMaxAbsError,
        Math.abs(source.equationResidual[sourceUnknown] - target.equationResidual[targetUnknown]),
      );
    }
  }
  for (let sourceRowId = 0; sourceRowId < rowMap.length; sourceRowId += 1) {
    const targetRowId = rowMap[sourceRowId];
    const sign = source.system.grid.gradientRows[sourceRowId].axis === source.system.grid.axis
      ? -1 : 1;
    projectedVelocityMaxAbsError = Math.max(
      projectedVelocityMaxAbsError,
      Math.abs(sign * source.velocityAfter[sourceRowId] - target.velocityAfter[targetRowId]),
    );
  }
  return {
    case: source.receipt.case,
    axis: source.system.grid.axis,
    phiMaxAbsError,
    thetaMaxAbsError,
    pressureMaxAbsError,
    projectedVelocityMaxAbsError,
    equationResidualMaxAbsError,
  };
}

/** Run both free-surface cases across every M1 topology and seam axis. */
export function probeTwoTileGhostFluidProjection(): GhostFluidProjectionProbeReceipt {
  const variants: GhostFluidProjectionVariantReceipt[] = [];
  const reflections: GhostFluidProjectionReflectionReceipt[] = [];
  for (const caseName of ["manufactured", "hydrostatic-like"] as const) {
    for (const axis of [0, 1, 2] as const) {
      for (const [negativeResolution, positiveResolution] of [[8, 8], [4, 4]] as const) {
        variants.push(execute(axis, negativeResolution, positiveResolution, caseName).receipt);
      }
      const fineNegative = execute(axis, 8, 4, caseName);
      const finePositive = execute(axis, 4, 8, caseName);
      variants.push(fineNegative.receipt, finePositive.receipt);
      reflections.push(compareReflection(fineNegative, finePositive));
    }
  }

  const thresholds = {
    solverRelative: 2e-10,
    postDivergenceRelative: 2e-10,
    postDivergenceAbsolute: 2e-9,
    energyAbsolute: 2e-10,
    reflectionAbsolute: 2e-9,
    minimumPositiveRayleigh: 1e-12,
  };
  const failures: string[] = [];
  for (const variant of variants) {
    const label = `${variant.case} axis=${variant.axis} ${variant.negativeResolution}+${variant.positiveResolution}`;
    if (variant.sampledMinimumRayleigh <= thresholds.minimumPositiveRayleigh
      || variant.minimumCgRayleigh <= thresholds.minimumPositiveRayleigh) {
      failures.push(`${label}: non-positive sampled/CG Rayleigh receipt`);
    }
    if (variant.solverRelativeResidualL2 > thresholds.solverRelative) {
      failures.push(`${label}: solver residual ${variant.solverRelativeResidualL2}`);
    }
    if (variant.postLiquidDivergenceVolumeL2 > thresholds.postDivergenceAbsolute
      && variant.divergenceReduction > thresholds.postDivergenceRelative) {
      failures.push(`${label}: post liquid divergence ${variant.postLiquidDivergenceVolumeL2}`);
    }
    if (!variant.energyNonIncreasing || variant.energyIdentityAbsError > thresholds.energyAbsolute) {
      failures.push(`${label}: projection metric energy identity failed`);
    }
    if (variant.negativeResolution !== variant.positiveResolution
      && variant.cutFiveTermSeamRowCount === 0) {
      failures.push(`${label}: free surface did not cut a five-term seam row`);
    }
    if (variant.iterations >= MAXIMUM_ITERATIONS) failures.push(`${label}: CG iteration limit`);
  }
  for (const reflection of reflections) {
    for (const [metric, value] of Object.entries(reflection)) {
      if (metric !== "axis" && metric !== "case" && value > thresholds.reflectionAbsolute) {
        failures.push(`${reflection.case} axis=${reflection.axis}: reflection ${metric} ${value}`);
      }
    }
  }
  return {
    schemaVersion: 1,
    milestone: "M1-two-tile-ghost-fluid-projection",
    formulation: {
      topologySource: "lib/methods/adaptive-mass/two-tile-composite-grid.ts",
      thetaSource: "lib/core/cm12-numerics.ts#cm12GhostFluidTheta",
      mixedRowWeighting: "For each cut row, coefficient-magnitude-weighted liquid and air phi averages feed the shared CM12 theta. Air terms are fixed to pressure zero and the liquid row contributes (dualWeight/theta) b b^T; projection subtracts (b dot p)/theta.",
      limitation: "CPU pressure-only probe with cell-center classification and one theta per composite row. It does not reconstruct per-subface interface geometry, cut-cell volumes, solid fractions, Ando-Batty face blocks, transport, GPU kernels, or a complete liquid time step.",
    },
    thresholds,
    variants,
    reflections,
    failures,
    passed: failures.length === 0,
  };
}

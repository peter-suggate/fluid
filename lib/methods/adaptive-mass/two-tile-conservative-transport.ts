/**
 * CPU proof for conservative scalar transport across the frozen 8^3/4^3
 * two-tile seam.
 *
 * This is deliberately an algebra oracle, not the final characteristic or GPU
 * implementation. It uses straight translations and an explicit finest-lattice
 * bridge at the seam:
 *
 * - a coarse donor is piecewise-constantly prolonged to the bridge lattice;
 * - trilinear bridge weights are collapsed back to leaf donors/receivers;
 * - a coarse receiver is therefore an intensive leaf sample, not eight hidden
 *   fine-cell updates;
 * - CM12's backward clamp and forward remainder are generalized with physical
 *   cell volumes.
 *
 * All topology and sampling choices stay here. The actual CM12 formulas are
 * imported from lib/core so the uniform and adaptive methods have one numerical
 * source of truth.
 */

import {
  cm12ConditionedGamma,
  cm12ConditionedRowCoefficient,
  cm12VolumeScaledDeficitCoefficient,
  cm12VolumeWeightedBetaContribution,
} from "../../core/cm12-numerics";
import {
  buildTwoTileCompositeGrid,
  type CompositeAxis,
  type TwoTileCompositeGrid,
  type TwoTileResolution,
  type Vec3,
} from "./two-tile-composite-grid";

export interface TwoTileTranslationOptions {
  axis: CompositeAxis;
  negativeResolution: TwoTileResolution;
  positiveResolution: TwoTileResolution;
  /** Signed world-space translation along axis. */
  displacement: number;
  tileWidth?: number;
  /** Persistent CM12 gamma at the donors. Defaults to one for the first step. */
  sourceGamma?: ArrayLike<number>;
}

export interface SparseTransportCoefficient {
  donorCellId: number;
  coefficient: number;
}

export interface TwoTileConservativeOperator {
  grid: TwoTileCompositeGrid;
  displacement: Vec3;
  /** A[i,j], stored by receiver i. */
  rows: readonly (readonly SparseTransportCoefficient[])[];
  backwardBeta: Float64Array;
  conditionedBeta: Float64Array;
  finalBeta: Float64Array;
  deficits: Float64Array;
  /** Gamma snapshot from which the conditioned density rows were built. */
  sourceGamma: Float64Array;
  /** CM12 gamma-prime: conditioned row sum plus forward deficit return. */
  nextGamma: Float64Array;
}

export interface TwoTileTransportFields {
  density: Float64Array;
  gamma: Float64Array;
}

export interface TwoTileTransportRun {
  operator: TwoTileConservativeOperator;
  before: TwoTileTransportFields;
  after: TwoTileTransportFields;
}

export interface TwoTileTransportVariantReceipt {
  axis: CompositeAxis;
  negativeResolution: TwoTileResolution;
  positiveResolution: TwoTileResolution;
  direction: -1 | 1;
  cellCount: number;
  nonzeroCoefficientCount: number;
  nonzeroDeficitCount: number;
  maximumDeficit: number;
  finalBetaMaximumAbsoluteError: number;
  minimumCoefficient: number;
  constantMassAbsoluteError: number;
  constantNormalizedDensityMaximumAbsoluteError: number;
  pulseMassBefore: number;
  pulseMassAfter: number;
  pulseMassAbsoluteError: number;
  pulseRelativeMassError: number;
  pulseMinimumDensity: number;
  pulseMaximumDensity: number;
  pulseCrossSeamIntegratedMass: number;
}

export interface TwoTileTransportReflectionReceipt {
  axis: CompositeAxis;
  maximumDensityDifference: number;
  maximumGammaDifference: number;
}

export interface TwoTileTransportSoakReceipt {
  axis: CompositeAxis;
  negativeResolution: 8 | 4;
  positiveResolution: 8 | 4;
  steps: number;
  maximumMassAbsoluteError: number;
  minimumDensity: number;
  maximumDensity: number;
  minimumGamma: number;
  maximumGamma: number;
}

export interface TwoTileTransportProbeResult {
  passed: boolean;
  tolerance: number;
  variants: readonly TwoTileTransportVariantReceipt[];
  reflections: readonly TwoTileTransportReflectionReceipt[];
  soaks: readonly TwoTileTransportSoakReceipt[];
  failures: readonly string[];
}

type MutableRow = Map<number, number>;

function vectorAlong(axis: CompositeAxis, value: number): Vec3 {
  const result: [number, number, number] = [0, 0, 0];
  result[axis] = value;
  return result;
}

function linearLocalIndex(local: readonly number[], resolution: number): number {
  return local[0] + resolution * (local[1] + resolution * local[2]);
}

function add(row: MutableRow, donorCellId: number, coefficient: number): void {
  if (coefficient === 0) return;
  row.set(donorCellId, (row.get(donorCellId) ?? 0) + coefficient);
}

function compensatedSum(values: Iterable<number>): number {
  let sum = 0;
  let correction = 0;
  for (const value of values) {
    const adjusted = value - correction;
    const next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return sum;
}

/**
 * Nonnegative trilinear weights on a periodic finest-cell bridge.
 *
 * Periodicity isolates the internal seam from outer-boundary policy. It also
 * means the two logical tiles have a second, reflected seam at the wrap. The
 * production method will replace this lookup with brick-neighbour and boundary
 * descriptors; the coefficient algebra is unchanged.
 */
function bridgeWeights(
  grid: TwoTileCompositeGrid,
  position: Vec3,
): MutableRow {
  const bridgeResolution = Math.max(grid.negativeResolution, grid.positiveResolution);
  const dimensions: [number, number, number] = [
    bridgeResolution,
    bridgeResolution,
    bridgeResolution,
  ];
  dimensions[grid.axis] *= 2;
  const cellWidth = grid.tileWidth / bridgeResolution;
  const bases: [number, number] = [0, grid.negativeResolution ** 3];
  const base: [number, number, number] = [0, 0, 0];
  const fraction: [number, number, number] = [0, 0, 0];

  for (const component of [0, 1, 2] as const) {
    const extent = dimensions[component] * cellWidth;
    const wrapped = ((position[component] % extent) + extent) % extent;
    const coordinate = wrapped / cellWidth - 0.5;
    base[component] = Math.floor(coordinate);
    fraction[component] = coordinate - base[component];
  }

  const row: MutableRow = new Map();
  for (let dz = 0; dz <= 1; dz += 1) {
    for (let dy = 0; dy <= 1; dy += 1) {
      for (let dx = 0; dx <= 1; dx += 1) {
        const corner = [dx, dy, dz] as const;
        const bridgeLocal: [number, number, number] = [0, 0, 0];
        let weight = 1;
        for (const component of [0, 1, 2] as const) {
          const unwrapped = base[component] + corner[component];
          bridgeLocal[component] =
            ((unwrapped % dimensions[component]) + dimensions[component]) % dimensions[component];
          weight *= corner[component] === 0
            ? 1 - fraction[component]
            : fraction[component];
        }

        const normalBridge = bridgeLocal[grid.axis];
        const tile: 0 | 1 = normalBridge < bridgeResolution ? 0 : 1;
        bridgeLocal[grid.axis] = normalBridge - tile * bridgeResolution;
        const leafResolution = tile === 0
          ? grid.negativeResolution
          : grid.positiveResolution;
        const bridgePerLeaf = bridgeResolution / leafResolution;
        const leafLocal: [number, number, number] = [
          Math.floor(bridgeLocal[0] / bridgePerLeaf),
          Math.floor(bridgeLocal[1] / bridgePerLeaf),
          Math.floor(bridgeLocal[2] / bridgePerLeaf),
        ];
        const donorCellId = bases[tile] + linearLocalIndex(leafLocal, leafResolution);
        add(row, donorCellId, weight);
      }
    }
  }
  return row;
}

function betaForRows(
  grid: TwoTileCompositeGrid,
  rows: readonly MutableRow[],
): Float64Array {
  const beta = new Float64Array(grid.cells.length);
  for (let receiverCellId = 0; receiverCellId < rows.length; receiverCellId += 1) {
    const receiverVolume = grid.cells[receiverCellId].volume;
    for (const [donorCellId, coefficient] of rows[receiverCellId]) {
      beta[donorCellId] += cm12VolumeWeightedBetaContribution(
        receiverVolume,
        grid.cells[donorCellId].volume,
        coefficient,
      );
    }
  }
  return beta;
}

export function buildTwoTileConservativeTransportOperator(
  options: TwoTileTranslationOptions,
): TwoTileConservativeOperator {
  if (!Number.isFinite(options.displacement)) {
    throw new RangeError(`displacement must be finite; received ${options.displacement}`);
  }
  const grid = buildTwoTileCompositeGrid(options);
  const sourceGamma = options.sourceGamma === undefined
    ? new Float64Array(grid.cells.length).fill(1)
    : Float64Array.from(options.sourceGamma);
  if (sourceGamma.length !== grid.cells.length) {
    throw new RangeError(
      `sourceGamma has ${sourceGamma.length} cells; expected ${grid.cells.length}`,
    );
  }
  for (let cellId = 0; cellId < sourceGamma.length; cellId += 1) {
    if (!Number.isFinite(sourceGamma[cellId]) || sourceGamma[cellId] < 0) {
      throw new RangeError(`sourceGamma[${cellId}] must be finite and nonnegative`);
    }
  }
  const displacement = vectorAlong(options.axis, options.displacement);
  const backwardWeights: MutableRow[] = grid.cells.map((receiver) => {
    const departure: Vec3 = [
      receiver.center[0] - displacement[0],
      receiver.center[1] - displacement[1],
      receiver.center[2] - displacement[2],
    ];
    return bridgeWeights(grid, departure);
  });
  const advectedGamma = Float64Array.from(backwardWeights, (backwardRow) => {
    const visibleTotal = compensatedSum(backwardRow.values());
    const sampledGamma = compensatedSum([...backwardRow].map(
      ([donorCellId, weight]) => weight * sourceGamma[donorCellId],
    ));
    return cm12ConditionedGamma(sampledGamma, visibleTotal);
  });
  const unconditionedRows: MutableRow[] = backwardWeights.map((backwardRow, receiverCellId) => {
    const visibleTotal = compensatedSum(backwardRow.values());
    const row: MutableRow = new Map();
    if (visibleTotal <= 1e-15) return row;
    for (const [donorCellId, backwardWeight] of backwardRow) {
      add(row, donorCellId, advectedGamma[receiverCellId] * backwardWeight / visibleTotal);
    }
    return row;
  });
  const backwardBeta = betaForRows(grid, unconditionedRows);

  const rows: MutableRow[] = backwardWeights.map((backwardRow, receiverCellId) => {
    const visibleTotal = compensatedSum(backwardRow.values());
    const conditioned: MutableRow = new Map();
    if (visibleTotal <= 1e-15) return conditioned;
    for (const [donorCellId, backwardWeight] of backwardRow) {
      add(conditioned, donorCellId, cm12ConditionedRowCoefficient(
        advectedGamma[receiverCellId],
        backwardWeight / visibleTotal,
        backwardBeta[donorCellId],
      ));
    }
    return conditioned;
  });
  const nextGamma = Float64Array.from(rows, (row) => compensatedSum(row.values()));
  const conditionedBeta = betaForRows(grid, rows);
  const deficits = Float64Array.from(conditionedBeta, (value) => Math.max(0, 1 - value));

  for (const donor of grid.cells) {
    const deficit = deficits[donor.id];
    if (deficit <= 0) continue;
    const arrival: Vec3 = [
      donor.center[0] + displacement[0],
      donor.center[1] + displacement[1],
      donor.center[2] + displacement[2],
    ];
    const forwardWeights = bridgeWeights(grid, arrival);
    const total = compensatedSum(forwardWeights.values());
    for (const [receiverCellId, rawWeight] of forwardWeights) {
      const normalizedWeight = rawWeight / total;
      const deficitCoefficient = cm12VolumeScaledDeficitCoefficient(
        donor.volume,
        grid.cells[receiverCellId].volume,
        deficit,
        normalizedWeight,
      );
      add(rows[receiverCellId], donor.id, deficitCoefficient);
      nextGamma[receiverCellId] += deficitCoefficient * sourceGamma[donor.id];
    }
  }

  const finalBeta = betaForRows(grid, rows);
  return {
    grid,
    displacement,
    rows: rows.map((row) => [...row]
      .sort(([left], [right]) => left - right)
      .map(([donorCellId, coefficient]) => ({ donorCellId, coefficient }))),
    backwardBeta,
    conditionedBeta,
    finalBeta,
    deficits,
    sourceGamma,
    nextGamma,
  };
}

export function applyTwoTileConservativeTransport(
  operator: TwoTileConservativeOperator,
  fields: TwoTileTransportFields,
): TwoTileTransportFields {
  const count = operator.grid.cells.length;
  if (fields.density.length !== count || fields.gamma.length !== count) {
    throw new RangeError(`transport fields must contain ${count} cells`);
  }
  for (let cellId = 0; cellId < count; cellId += 1) {
    if (fields.gamma[cellId] !== operator.sourceGamma[cellId]) {
      throw new Error(
        `transport operator gamma snapshot differs at cell ${cellId}; rebuild it for this step`,
      );
    }
  }
  const density = new Float64Array(count);
  for (let receiverCellId = 0; receiverCellId < count; receiverCellId += 1) {
    for (const { donorCellId, coefficient } of operator.rows[receiverCellId]) {
      density[receiverCellId] += coefficient * fields.density[donorCellId];
    }
  }
  return { density, gamma: operator.nextGamma.slice() };
}

/** Build and apply one CM12 transport step from the supplied persistent gamma. */
export function advanceTwoTileConservativeTransport(
  options: Omit<TwoTileTranslationOptions, "sourceGamma">,
  fields: TwoTileTransportFields,
): TwoTileTransportRun {
  const operator = buildTwoTileConservativeTransportOperator({
    ...options,
    sourceGamma: fields.gamma,
  });
  return { operator, before: fields, after: applyTwoTileConservativeTransport(operator, fields) };
}

export function integratedScalar(
  grid: TwoTileCompositeGrid,
  values: ArrayLike<number>,
): number {
  return compensatedSum(grid.cells.map((cell) => cell.volume * values[cell.id]));
}

function pulseDensity(grid: TwoTileCompositeGrid, direction: -1 | 1): Float64Array {
  const centerNormal = grid.tileWidth + direction * -0.18 * grid.tileWidth;
  return Float64Array.from(grid.cells, (cell) => {
    const normal = (cell.center[grid.axis] - centerNormal) / (0.31 * grid.tileWidth);
    const tangent0 = (cell.center[grid.tangentialAxes[0]] - 0.5 * grid.tileWidth)
      / (0.29 * grid.tileWidth);
    const tangent1 = (cell.center[grid.tangentialAxes[1]] - 0.5 * grid.tileWidth)
      / (0.37 * grid.tileWidth);
    const radiusSquared = normal * normal + tangent0 * tangent0 + tangent1 * tangent1;
    return Math.max(0, 1 - radiusSquared) ** 2;
  });
}

function maximumAbsolute(values: ArrayLike<number>, target = 0): number {
  let maximum = 0;
  for (let index = 0; index < values.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(values[index] - target));
  }
  return maximum;
}

function minimum(values: ArrayLike<number>): number {
  let result = Number.POSITIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) result = Math.min(result, values[index]);
  return result;
}

function maximum(values: ArrayLike<number>): number {
  let result = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) result = Math.max(result, values[index]);
  return result;
}

function runVariant(
  axis: CompositeAxis,
  negativeResolution: TwoTileResolution,
  positiveResolution: TwoTileResolution,
  direction: -1 | 1,
): { receipt: TwoTileTransportVariantReceipt; pulse: TwoTileTransportRun } {
  const bridgeResolution = Math.max(negativeResolution, positiveResolution);
  const displacement = direction * 0.75 / bridgeResolution;
  const operator = buildTwoTileConservativeTransportOperator({
    axis,
    negativeResolution,
    positiveResolution,
    displacement,
  });
  const count = operator.grid.cells.length;
  const constantBefore = {
    density: new Float64Array(count).fill(1),
    gamma: new Float64Array(count).fill(1),
  };
  const constantAfter = applyTwoTileConservativeTransport(operator, constantBefore);
  const constantMassAbsoluteError = Math.abs(
    integratedScalar(operator.grid, constantAfter.density)
      - integratedScalar(operator.grid, constantBefore.density),
  );
  let constantNormalizedDensityMaximumAbsoluteError = 0;
  for (let index = 0; index < count; index += 1) {
    constantNormalizedDensityMaximumAbsoluteError = Math.max(
      constantNormalizedDensityMaximumAbsoluteError,
      Math.abs(constantAfter.density[index] / constantAfter.gamma[index] - 1),
    );
  }

  const pulseBefore = {
    density: pulseDensity(operator.grid, direction),
    gamma: new Float64Array(count).fill(1),
  };
  const pulseAfter = applyTwoTileConservativeTransport(operator, pulseBefore);
  const pulse: TwoTileTransportRun = { operator, before: pulseBefore, after: pulseAfter };
  const pulseMassBefore = integratedScalar(operator.grid, pulseBefore.density);
  const pulseMassAfter = integratedScalar(operator.grid, pulseAfter.density);
  const pulseMassAbsoluteError = Math.abs(pulseMassAfter - pulseMassBefore);
  let pulseCrossSeamIntegratedMass = 0;
  let nonzeroCoefficientCount = 0;
  let minimumCoefficient = Number.POSITIVE_INFINITY;
  for (let receiverCellId = 0; receiverCellId < operator.rows.length; receiverCellId += 1) {
    const receiver = operator.grid.cells[receiverCellId];
    for (const coefficient of operator.rows[receiverCellId]) {
      nonzeroCoefficientCount += 1;
      minimumCoefficient = Math.min(minimumCoefficient, coefficient.coefficient);
      const donor = operator.grid.cells[coefficient.donorCellId];
      if (receiver.tile !== donor.tile) {
        pulseCrossSeamIntegratedMass += receiver.volume * coefficient.coefficient
          * pulseBefore.density[donor.id];
      }
    }
  }

  return {
    receipt: {
      axis,
      negativeResolution,
      positiveResolution,
      direction,
      cellCount: count,
      nonzeroCoefficientCount,
      nonzeroDeficitCount: [...operator.deficits].filter((value) => value > 1e-15).length,
      maximumDeficit: maximum(operator.deficits),
      finalBetaMaximumAbsoluteError: maximumAbsolute(operator.finalBeta, 1),
      minimumCoefficient,
      constantMassAbsoluteError,
      constantNormalizedDensityMaximumAbsoluteError,
      pulseMassBefore,
      pulseMassAfter,
      pulseMassAbsoluteError,
      pulseRelativeMassError: pulseMassAbsoluteError / Math.max(pulseMassBefore, Number.EPSILON),
      pulseMinimumDensity: minimum(pulseAfter.density),
      pulseMaximumDensity: maximum(pulseAfter.density),
      pulseCrossSeamIntegratedMass,
    },
    pulse,
  };
}

function reflectedCellMap(
  source: TwoTileCompositeGrid,
  reflected: TwoTileCompositeGrid,
): Int32Array {
  const scale = 1e9;
  const key = (resolution: number, center: Vec3): string =>
    `${resolution}:${center.map((value) => Math.round(value * scale)).join(",")}`;
  const reflectedByKey = new Map<string, number>();
  for (const cell of reflected.cells) reflectedByKey.set(key(cell.resolution, cell.center), cell.id);
  return Int32Array.from(source.cells, (cell) => {
    const center = [...cell.center] as [number, number, number];
    center[source.axis] = 2 * source.tileWidth - center[source.axis];
    const match = reflectedByKey.get(key(cell.resolution, center));
    if (match === undefined) throw new Error("reflected cell lookup failed");
    return match;
  });
}

export function probeTwoTileConservativeTransport(
  tolerance = 1e-11,
): TwoTileTransportProbeResult {
  if (!Number.isFinite(tolerance) || tolerance <= 0) {
    throw new RangeError(`tolerance must be finite and positive; received ${tolerance}`);
  }
  const variants: TwoTileTransportVariantReceipt[] = [];
  const runs = new Map<string, TwoTileTransportRun>();
  for (const axis of [0, 1, 2] as const) {
    for (const [negativeResolution, positiveResolution] of [
      [8, 8],
      [4, 4],
      [8, 4],
      [4, 8],
    ] as const) {
      for (const direction of [-1, 1] as const) {
        const result = runVariant(axis, negativeResolution, positiveResolution, direction);
        variants.push(result.receipt);
        runs.set(`${axis}:${negativeResolution}:${positiveResolution}:${direction}`, result.pulse);
      }
    }
  }

  const reflections: TwoTileTransportReflectionReceipt[] = [];
  for (const axis of [0, 1, 2] as const) {
    const original = runs.get(`${axis}:8:4:1`)!;
    const reflected = runs.get(`${axis}:4:8:-1`)!;
    const mapping = reflectedCellMap(original.operator.grid, reflected.operator.grid);
    let maximumDensityDifference = 0;
    let maximumGammaDifference = 0;
    for (let source = 0; source < mapping.length; source += 1) {
      const target = mapping[source];
      maximumDensityDifference = Math.max(maximumDensityDifference,
        Math.abs(original.after.density[source] - reflected.after.density[target]));
      maximumGammaDifference = Math.max(maximumGammaDifference,
        Math.abs(original.after.gamma[source] - reflected.after.gamma[target]));
    }
    reflections.push({ axis, maximumDensityDifference, maximumGammaDifference });
  }

  // Persistent gamma is state, not another scalar multiplied by A. Exercise
  // that distinction over enough crossings to expose a one-step-only oracle.
  const soaks: TwoTileTransportSoakReceipt[] = [];
  const soakSteps = 128;
  for (const axis of [0, 1, 2] as const) {
    for (const [negativeResolution, positiveResolution] of [[8, 4], [4, 8]] as const) {
      const grid = buildTwoTileCompositeGrid({ axis, negativeResolution, positiveResolution });
      let fields: TwoTileTransportFields = {
        density: pulseDensity(grid, 1),
        gamma: new Float64Array(grid.cells.length).fill(1),
      };
      const initialMass = integratedScalar(grid, fields.density);
      let maximumMassAbsoluteError = 0;
      let minimumDensity = minimum(fields.density);
      let maximumDensity = maximum(fields.density);
      let minimumGamma = 1;
      let maximumGamma = 1;
      for (let step = 0; step < soakSteps; step += 1) {
        const direction = step % 2 === 0 ? 1 : -1;
        const run = advanceTwoTileConservativeTransport({
          axis,
          negativeResolution,
          positiveResolution,
          displacement: direction * 0.75 / 8,
        }, fields);
        fields = run.after;
        maximumMassAbsoluteError = Math.max(
          maximumMassAbsoluteError,
          Math.abs(integratedScalar(grid, fields.density) - initialMass),
        );
        minimumDensity = Math.min(minimumDensity, minimum(fields.density));
        maximumDensity = Math.max(maximumDensity, maximum(fields.density));
        minimumGamma = Math.min(minimumGamma, minimum(fields.gamma));
        maximumGamma = Math.max(maximumGamma, maximum(fields.gamma));
      }
      soaks.push({
        axis,
        negativeResolution,
        positiveResolution,
        steps: soakSteps,
        maximumMassAbsoluteError,
        minimumDensity,
        maximumDensity,
        minimumGamma,
        maximumGamma,
      });
    }
  }

  const failures: string[] = [];
  for (const variant of variants) {
    const name = `axis ${variant.axis} ${variant.negativeResolution}+${variant.positiveResolution}`
      + ` direction ${variant.direction}`;
    if (variant.finalBetaMaximumAbsoluteError > tolerance) {
      failures.push(`${name}: weighted beta error ${variant.finalBetaMaximumAbsoluteError}`);
    }
    if (variant.constantMassAbsoluteError > tolerance) {
      failures.push(`${name}: constant mass error ${variant.constantMassAbsoluteError}`);
    }
    if (variant.pulseMassAbsoluteError > tolerance) {
      failures.push(`${name}: pulse mass error ${variant.pulseMassAbsoluteError}`);
    }
    if (variant.constantNormalizedDensityMaximumAbsoluteError > tolerance) {
      failures.push(`${name}: normalized constant error ${variant.constantNormalizedDensityMaximumAbsoluteError}`);
    }
    if (variant.minimumCoefficient < -tolerance) {
      failures.push(`${name}: negative coefficient ${variant.minimumCoefficient}`);
    }
    if (!(variant.pulseCrossSeamIntegratedMass > tolerance)) {
      failures.push(`${name}: pulse did not cross a seam`);
    }
  }
  for (const reflection of reflections) {
    if (reflection.maximumDensityDifference > tolerance
      || reflection.maximumGammaDifference > tolerance) {
      failures.push(`axis ${reflection.axis}: reflected transport differs by density `
        + `${reflection.maximumDensityDifference}, gamma ${reflection.maximumGammaDifference}`);
    }
  }
  for (const soak of soaks) {
    const name = `axis ${soak.axis} ${soak.negativeResolution}+${soak.positiveResolution}`;
    if (soak.maximumMassAbsoluteError > tolerance) {
      failures.push(`${name}: ${soak.steps}-step mass error ${soak.maximumMassAbsoluteError}`);
    }
    if (!Number.isFinite(soak.minimumDensity) || soak.minimumDensity < -tolerance) {
      failures.push(`${name}: ${soak.steps}-step minimum density ${soak.minimumDensity}`);
    }
    if (!Number.isFinite(soak.minimumGamma) || soak.minimumGamma < -tolerance
      || !Number.isFinite(soak.maximumGamma)) {
      failures.push(`${name}: ${soak.steps}-step gamma range `
        + `${soak.minimumGamma}..${soak.maximumGamma}`);
    }
  }
  return { passed: failures.length === 0, tolerance, variants, reflections, soaks, failures };
}

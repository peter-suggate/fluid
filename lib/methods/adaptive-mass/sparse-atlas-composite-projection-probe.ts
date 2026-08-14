/** Executable symmetry/physics receipt for the arbitrary sparse-atlas projector. */

import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "./sparse-brick-atlas";
import {
  applySparseAtlasDivergence,
  applySparseAtlasGradient,
  applySparseAtlasPressureOperator,
  buildSparseAtlasCompositeGrid,
  projectSparseAtlasVelocity,
  type SparseAtlasAxis,
  type SparseAtlasCompositeGrid,
} from "./sparse-atlas-composite-projection";

export interface SparseAtlasD4Receipt {
  readonly name: string;
  readonly operatorMaximumError: number;
  readonly pressureMaximumError: number;
  readonly projectedVelocityMaximumError: number;
  readonly divergenceMaximumError: number;
}

export interface SparseAtlasCompositeProjectionProbeReceipt {
  readonly schemaVersion: 1;
  readonly milestone: "arbitrary-sparse-atlas-composite-projection";
  readonly cellCount: number;
  readonly rowCount: number;
  readonly mixedSeamRowCount: number;
  readonly mixedSeamTermCounts: readonly number[];
  readonly constantGradientMaximum: number;
  readonly linearGradientMaximum: number;
  readonly transposeError: number;
  readonly symmetryError: number;
  readonly sampledMinimumRayleigh: number;
  readonly projectionRelativeResidual: number;
  readonly projectionDivergenceReduction: number;
  readonly projectionEnergyIdentityError: number;
  readonly d4: readonly SparseAtlasD4Receipt[];
  readonly d4MaximumError: number;
  readonly failures: readonly string[];
  readonly passed: boolean;
}

interface D4Transform {
  readonly name: string;
  readonly point: (x: number, z: number, extent: number) => readonly [number, number];
  readonly axis: (axis: SparseAtlasAxis) => readonly [SparseAtlasAxis, -1 | 1];
}

const D4: readonly D4Transform[] = [
  { name: "identity", point: (x, z) => [x, z], axis: (axis) => [axis, 1] },
  { name: "rotate-90", point: (x, z, l) => [l - z, x], axis: (axis) =>
    axis === 0 ? [2, 1] : axis === 2 ? [0, -1] : [1, 1] },
  { name: "rotate-180", point: (x, z, l) => [l - x, l - z], axis: (axis) =>
    axis === 0 || axis === 2 ? [axis, -1] : [1, 1] },
  { name: "rotate-270", point: (x, z, l) => [z, l - x], axis: (axis) =>
    axis === 0 ? [2, -1] : axis === 2 ? [0, 1] : [1, 1] },
  { name: "reflect-x", point: (x, z, l) => [l - x, z], axis: (axis) =>
    axis === 0 ? [0, -1] : [axis, 1] },
  { name: "reflect-diagonal", point: (x, z) => [z, x], axis: (axis) =>
    axis === 0 ? [2, 1] : axis === 2 ? [0, 1] : [1, 1] },
  { name: "reflect-z", point: (x, z, l) => [x, l - z], axis: (axis) =>
    axis === 2 ? [2, -1] : [axis, 1] },
  { name: "reflect-antidiagonal", point: (x, z, l) => [l - z, l - x], axis: (axis) =>
    axis === 0 ? [2, -1] : axis === 2 ? [0, -1] : [1, 1] },
];

function createBrick(
  key: number,
  coordinate: readonly [number, number, number],
  resolution: SparseBrickResolution,
): SparseAdaptiveMassBrick {
  return {
    key,
    coordinate,
    resolution,
    density: new Float64Array(resolution ** 3).fill(1),
    gamma: new Float64Array(resolution ** 3).fill(1),
  };
}

function transformedAtlas(transform: D4Transform) {
  const dimensions = [16, 8, 16] as const;
  const brickDimensions = [2, 1, 2] as const;
  const pattern: readonly (readonly [number, number, SparseBrickResolution])[] = [
    [0, 0, 8], [1, 0, 4], [0, 1, 4], [1, 1, 8],
  ];
  const bricks = pattern.map(([x, z, resolution]) => {
    const [centerX, centerZ] = transform.point(8 * x + 4, 8 * z + 4, 16);
    const coordinate = [Math.floor(centerX / 8), 0, Math.floor(centerZ / 8)] as const;
    return createBrick(sparseBrickKey(coordinate, brickDimensions), coordinate, resolution);
  });
  return createSparseAdaptiveMassAtlas(dimensions, bricks);
}

function dot(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let sum = 0;
  for (let index = 0; index < left.length; index += 1) sum += left[index] * right[index];
  return sum;
}

function maximumAbsolute(values: ArrayLike<number>): number {
  let maximum = 0;
  for (let index = 0; index < values.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(values[index]));
  }
  return maximum;
}

function maximumMappedError(
  source: ArrayLike<number>,
  target: ArrayLike<number>,
  map: Int32Array,
  sign = 1,
): number {
  let maximum = 0;
  for (let index = 0; index < map.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(sign * source[index] - target[map[index]]));
  }
  return maximum;
}

function positionKey(
  resolution: SparseBrickResolution,
  center: readonly [number, number, number],
): string {
  return `${resolution}:${center.join(",")}`;
}

function rowKey(axis: SparseAtlasAxis, center: readonly [number, number, number]): string {
  return `${axis}:${center.join(",")}`;
}

function cellMap(
  source: SparseAtlasCompositeGrid,
  target: SparseAtlasCompositeGrid,
  transform: D4Transform,
): Int32Array {
  const targetByPosition = new Map<string, number>();
  for (const cell of target.cells) {
    targetByPosition.set(positionKey(cell.brickResolution, cell.centerFine), cell.id);
  }
  return Int32Array.from(source.cells, (cell) => {
    const [x, z] = transform.point(cell.centerFine[0], cell.centerFine[2], 16);
    const id = targetByPosition.get(positionKey(
      cell.brickResolution, [x, cell.centerFine[1], z],
    ));
    if (id === undefined) throw new Error(`D4 ${transform.name} lost a composite cell`);
    return id;
  });
}

function rowMap(
  source: SparseAtlasCompositeGrid,
  target: SparseAtlasCompositeGrid,
  transform: D4Transform,
): { readonly ids: Int32Array; readonly signs: Int8Array } {
  const targetByPosition = new Map<string, number>();
  for (const row of target.gradientRows) targetByPosition.set(rowKey(row.axis, row.centerFine), row.id);
  const signs = new Int8Array(source.gradientRows.length);
  const ids = Int32Array.from(source.gradientRows, (row) => {
    const [axis, sign] = transform.axis(row.axis);
    signs[row.id] = sign;
    const [x, z] = transform.point(row.centerFine[0], row.centerFine[2], 16);
    const id = targetByPosition.get(rowKey(axis, [x, row.centerFine[1], z]));
    if (id === undefined) throw new Error(`D4 ${transform.name} lost a composite row`);
    return id;
  });
  return { ids, signs };
}

function manufacturedPressure(grid: SparseAtlasCompositeGrid): Float64Array {
  return Float64Array.from(grid.cells, (cell) => {
    const [x, y, z] = cell.centerFine;
    return Math.sin(0.17 + 0.13 * x - 0.09 * y + 0.07 * z) + 0.002 * x * z;
  });
}

function manufacturedVelocity(grid: SparseAtlasCompositeGrid): Float64Array {
  return Float64Array.from(grid.gradientRows, (row) => {
    const [x, y, z] = row.centerFine;
    return row.axis === 0
      ? Math.sin(0.31 + 0.11 * x - 0.07 * y + 0.03 * z)
      : row.axis === 1
        ? Math.cos(0.23 - 0.05 * x + 0.13 * y + 0.04 * z)
        : Math.sin(0.41 + 0.06 * x - 0.02 * y - 0.1 * z);
  });
}

function mappedCellValues(
  source: ArrayLike<number>,
  map: Int32Array,
  targetLength: number,
): Float64Array {
  const result = new Float64Array(targetLength);
  for (let sourceId = 0; sourceId < map.length; sourceId += 1) result[map[sourceId]] = source[sourceId];
  return result;
}

function mappedRowValues(
  source: ArrayLike<number>,
  map: { readonly ids: Int32Array; readonly signs: Int8Array },
  targetLength: number,
): Float64Array {
  const result = new Float64Array(targetLength);
  for (let sourceId = 0; sourceId < map.ids.length; sourceId += 1) {
    result[map.ids[sourceId]] = map.signs[sourceId] * source[sourceId];
  }
  return result;
}

/** Run mixed-resolution algebra, projection, reflection and all eight D4 maps. */
export function probeSparseAtlasCompositeProjection(
  tolerance = 2e-8,
): SparseAtlasCompositeProjectionProbeReceipt {
  const base = buildSparseAtlasCompositeGrid(transformedAtlas(D4[0]));
  const constantGradientMaximum = maximumAbsolute(
    applySparseAtlasGradient(base, new Float64Array(base.cells.length).fill(2.5)),
  );
  let linearGradientMaximum = 0;
  for (const axis of [0, 1, 2] as const) {
    const linear = Float64Array.from(base.cells, (cell) => cell.centerFine[axis]);
    const gradient = applySparseAtlasGradient(base, linear);
    for (const row of base.gradientRows) {
      linearGradientMaximum = Math.max(
        linearGradientMaximum, Math.abs(gradient[row.id] - (row.axis === axis ? 1 : 0)),
      );
    }
  }
  const pressure = manufacturedPressure(base);
  const velocity = manufacturedVelocity(base);
  const gradient = applySparseAtlasGradient(base, pressure);
  const divergence = applySparseAtlasDivergence(base, velocity);
  let transposeLeft = 0;
  let transposeRight = 0;
  for (const row of base.gradientRows) {
    transposeLeft += row.dualWeight * gradient[row.id] * velocity[row.id];
  }
  for (const cell of base.cells) {
    transposeRight += cell.volume * pressure[cell.id] * divergence[cell.id];
  }
  const transposeError = Math.abs(transposeLeft + transposeRight);
  const secondPressure = Float64Array.from(base.cells, (cell) => {
    const [x, y, z] = cell.centerFine;
    return Math.cos(0.11 * x + 0.05 * y - 0.14 * z) + 0.003 * y * z;
  });
  const appliedPressure = applySparseAtlasPressureOperator(base, pressure);
  const appliedSecond = applySparseAtlasPressureOperator(base, secondPressure);
  const symmetryError = Math.abs(dot(pressure, appliedSecond) - dot(secondPressure, appliedPressure));
  let sampledMinimumRayleigh = Number.POSITIVE_INFINITY;
  for (let mode = 1; mode <= 5; mode += 1) {
    const sample = Float64Array.from(base.cells, (cell) => {
      const [x, y, z] = cell.centerFine;
      return Math.sin(mode * 0.09 * x + 0.07 * y - 0.04 * z);
    });
    const mean = sample.reduce((sum, value) => sum + value, 0) / sample.length;
    for (let index = 0; index < sample.length; index += 1) sample[index] -= mean;
    sampledMinimumRayleigh = Math.min(
      sampledMinimumRayleigh,
      dot(sample, applySparseAtlasPressureOperator(base, sample)) / dot(sample, sample),
    );
  }
  const sourceProjection = projectSparseAtlasVelocity(base, {
    normalVelocity: velocity,
    relativeTolerance: 2e-11,
    maximumIterations: 8192,
  });

  const d4: SparseAtlasD4Receipt[] = [];
  let d4MaximumError = 0;
  for (const transform of D4) {
    const target = buildSparseAtlasCompositeGrid(transformedAtlas(transform));
    const cells = cellMap(base, target, transform);
    const rows = rowMap(base, target, transform);
    const mappedPressure = mappedCellValues(pressure, cells, target.cells.length);
    const mappedVelocity = mappedRowValues(velocity, rows, target.gradientRows.length);
    const targetApplied = applySparseAtlasPressureOperator(target, mappedPressure);
    const targetProjection = projectSparseAtlasVelocity(target, {
      normalVelocity: mappedVelocity,
      relativeTolerance: 2e-11,
      maximumIterations: 8192,
    });
    const targetDivergence = targetProjection.leafDivergence;
    const operatorMaximumError = maximumMappedError(appliedPressure, targetApplied, cells);
    const pressureMaximumError = maximumMappedError(
      sourceProjection.leafPressure, targetProjection.leafPressure, cells,
    );
    let projectedVelocityMaximumError = 0;
    for (let sourceId = 0; sourceId < rows.ids.length; sourceId += 1) {
      projectedVelocityMaximumError = Math.max(projectedVelocityMaximumError, Math.abs(
        rows.signs[sourceId] * sourceProjection.projectedFaceVelocity[sourceId]
          - targetProjection.projectedFaceVelocity[rows.ids[sourceId]],
      ));
    }
    const divergenceMaximumError = maximumMappedError(
      sourceProjection.leafDivergence, targetDivergence, cells,
    );
    const receipt = {
      name: transform.name,
      operatorMaximumError,
      pressureMaximumError,
      projectedVelocityMaximumError,
      divergenceMaximumError,
    };
    d4.push(receipt);
    d4MaximumError = Math.max(d4MaximumError, ...Object.values(receipt)
      .filter((value): value is number => typeof value === "number"));
  }

  const failures: string[] = [];
  if (base.mixedSeamRowCount === 0) failures.push("mixed seam was not exercised");
  const termCounts = [...new Set(base.gradientRows
    .filter((row) => row.kind === "mixed-seam").map((row) => row.terms.length))].sort();
  if (termCounts.length !== 1 || termCounts[0] !== 5) failures.push("mixed seam is not 5-term");
  if (constantGradientMaximum > tolerance) failures.push("constant gradient");
  if (linearGradientMaximum > tolerance) failures.push("linear gradient");
  if (transposeError > tolerance) failures.push("weighted transpose identity");
  if (symmetryError > tolerance) failures.push("pressure symmetry");
  if (!(sampledMinimumRayleigh > 0)) failures.push("positive pressure energy");
  if (sourceProjection.receipt.relativeResidualL2 > tolerance) failures.push("pressure residual");
  if (sourceProjection.receipt.divergenceReduction > tolerance) failures.push("divergence reduction");
  if (sourceProjection.receipt.energyIdentityAbsError > tolerance) failures.push("energy identity");
  if (d4MaximumError > tolerance) failures.push("D4/reflection equivariance");
  return {
    schemaVersion: 1,
    milestone: "arbitrary-sparse-atlas-composite-projection",
    cellCount: base.cells.length,
    rowCount: base.gradientRows.length,
    mixedSeamRowCount: base.mixedSeamRowCount,
    mixedSeamTermCounts: termCounts,
    constantGradientMaximum,
    linearGradientMaximum,
    transposeError,
    symmetryError,
    sampledMinimumRayleigh,
    projectionRelativeResidual: sourceProjection.receipt.relativeResidualL2,
    projectionDivergenceReduction: sourceProjection.receipt.divergenceReduction,
    projectionEnergyIdentityError: sourceProjection.receipt.energyIdentityAbsError,
    d4,
    d4MaximumError,
    failures,
    passed: failures.length === 0,
  };
}

/**
 * CPU algebra oracle for the first adaptive-mass milestone.
 *
 * The grid consists of two equal-world-extent tiles sharing one complete face.
 * Each tile may contain either 4^3 or 8^3 cells.  All pressure operations are
 * derived from one list of oriented gradient rows.  In particular, divergence
 * is the volume-inner-product negative transpose of that same list; there is no
 * independently maintained coarse/fine divergence stencil.
 *
 * This module deliberately has no WebGPU or uniform-method dependency.  It is
 * small enough to be used by command-line probes and precise enough to remain
 * the CPU oracle when the WGSL implementation arrives.
 */

export type CompositeAxis = 0 | 1 | 2;
export type TwoTileResolution = 4 | 8;
export type Vec3 = readonly [number, number, number];
export type Vec3i = readonly [number, number, number];

export interface TwoTileCompositeGridOptions {
  /** Normal axis of the shared face. Rows are oriented in its positive direction. */
  axis: CompositeAxis;
  /** Resolution of the tile on the negative side of the shared face. */
  negativeResolution: TwoTileResolution;
  /** Resolution of the tile on the positive side of the shared face. */
  positiveResolution: TwoTileResolution;
  /** World-space width of each logical tile. Defaults to one. */
  tileWidth?: number;
}

export interface CompositeCell {
  id: number;
  /** Stable world-side slot: zero is negative, one is positive. */
  tile: 0 | 1;
  resolution: TwoTileResolution;
  local: Vec3i;
  /** Exact cell center in half-finest-cell integer units. */
  centerFineHalf: Vec3i;
  center: Vec3;
  width: number;
  volume: number;
  /** True for the one-cell layer touching the resolution seam. */
  touchesSeam: boolean;
}

export interface CompositeGradientTerm {
  cellId: number;
  coefficient: number;
}

export interface CompositeGradientRow {
  /** Stable global face/port ID and index into gradientRows. */
  id: number;
  kind: "regular" | "seam";
  axis: CompositeAxis;
  /** Stable world-side tile for regular rows; absent for shared seam rows. */
  tile?: 0 | 1;
  /** Local face coordinate for regular rows. */
  localFace?: Vec3i;
  /** Stable tangential port coordinates for seam rows. */
  port?: readonly [number, number];
  centerFineHalf: Vec3i;
  center: Vec3;
  area: number;
  distance: number;
  /** Finite-volume dual measure: face area times center distance. */
  dualWeight: number;
  /** Oriented coefficients of G. Negative-side coefficients precede positive. */
  terms: readonly CompositeGradientTerm[];
}

export interface TwoTileCompositeGrid {
  axis: CompositeAxis;
  tangentialAxes: readonly [CompositeAxis, CompositeAxis];
  tileWidth: number;
  finestResolution: 8;
  negativeResolution: TwoTileResolution;
  positiveResolution: TwoTileResolution;
  cells: readonly CompositeCell[];
  regularFaces: readonly CompositeGradientRow[];
  seamPorts: readonly CompositeGradientRow[];
  /** regularFaces followed by seamPorts; row.id equals its array index. */
  gradientRows: readonly CompositeGradientRow[];
  /** Cell IDs for the layers immediately adjacent to the seam. */
  seamCellIds: readonly number[];
}

export interface DensePressureMatrix {
  size: number;
  /** Row-major symmetric matrix for A = G^T W G. */
  values: Float64Array;
}

export interface TwoTileCompositeVariantProbe {
  axis: CompositeAxis;
  negativeResolution: TwoTileResolution;
  positiveResolution: TwoTileResolution;
  cellCount: number;
  regularFaceCount: number;
  seamPortCount: number;
  seamTermCountMin: number;
  seamTermCountMax: number;
  constantGradientMaxAbs: number;
  linearGradientMaxAbs: number;
  constantNormalSeamDivergenceMaxAbs: number;
  transposeError: number;
  symmetryMaxAbs: number;
  quadraticEnergy: number;
  /** Smallest Rayleigh quotient among deterministic non-constant probe fields. */
  minimumRayleigh: number;
}

export interface TwoTileSwapSymmetryProbe {
  axis: CompositeAxis;
  negativeResolution: 8;
  positiveResolution: 4;
  reflectedNegativeResolution: 4;
  reflectedPositiveResolution: 8;
  maximumMatrixDifference: number;
}

export interface TwoTileCompositeProbeResult {
  passed: boolean;
  tolerance: number;
  variants: readonly TwoTileCompositeVariantProbe[];
  swapSymmetryMaxAbs: number;
  swapSymmetryByAxis: readonly TwoTileSwapSymmetryProbe[];
  failures: readonly string[];
}

const FINEST_RESOLUTION = 8 as const;

function assertAxis(axis: number): asserts axis is CompositeAxis {
  if (axis !== 0 && axis !== 1 && axis !== 2) {
    throw new RangeError(`axis must be 0, 1, or 2; received ${axis}`);
  }
}

function assertResolution(value: number, label: string): asserts value is TwoTileResolution {
  if (value !== 4 && value !== 8) {
    throw new RangeError(`${label} must be 4 or 8; received ${value}`);
  }
}

function mutableVec3(x = 0, y = 0, z = 0): [number, number, number] {
  return [x, y, z];
}

function tangentialAxes(normal: CompositeAxis): [CompositeAxis, CompositeAxis] {
  if (normal === 0) return [1, 2];
  if (normal === 1) return [0, 2];
  return [0, 1];
}

function localLinearIndex(local: Vec3i, resolution: number): number {
  return local[0] + resolution * (local[1] + resolution * local[2]);
}

function maxAbs(values: ArrayLike<number>): number {
  let result = 0;
  for (let i = 0; i < values.length; i += 1) {
    result = Math.max(result, Math.abs(values[i]));
  }
  return result;
}

function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new RangeError("dot vector lengths differ");
  let sum = 0;
  let correction = 0;
  for (let i = 0; i < a.length; i += 1) {
    const product = a[i] * b[i];
    const adjusted = product - correction;
    const next = sum + adjusted;
    correction = next - sum - adjusted;
    sum = next;
  }
  return sum;
}

/**
 * Construct the frozen two-tile pressure graph.
 *
 * Global cell IDs are world-side stable: all negative-tile cells first, then
 * all positive-tile cells, each in x-major local order.  Regular face IDs are
 * emitted by tile, axis, and local face coordinate; seam IDs follow them.
 */
export function buildTwoTileCompositeGrid(
  options: TwoTileCompositeGridOptions,
): TwoTileCompositeGrid {
  assertAxis(options.axis);
  assertResolution(options.negativeResolution, "negativeResolution");
  assertResolution(options.positiveResolution, "positiveResolution");
  const tileWidth = options.tileWidth ?? 1;
  if (!Number.isFinite(tileWidth) || tileWidth <= 0) {
    throw new RangeError(`tileWidth must be finite and positive; received ${tileWidth}`);
  }

  const axis = options.axis;
  const tangents = tangentialAxes(axis);
  const resolutions = [options.negativeResolution, options.positiveResolution] as const;
  const cells: CompositeCell[] = [];
  const cellBases: [number, number] = [0, 0];

  for (const tile of [0, 1] as const) {
    const resolution = resolutions[tile];
    const strideFine = FINEST_RESOLUTION / resolution;
    const width = tileWidth / resolution;
    cellBases[tile] = cells.length;
    for (let z = 0; z < resolution; z += 1) {
      for (let y = 0; y < resolution; y += 1) {
        for (let x = 0; x < resolution; x += 1) {
          const local: Vec3i = [x, y, z];
          const centerFineHalf = mutableVec3();
          const center = mutableVec3();
          for (const component of [0, 1, 2] as const) {
            const tileOffsetFine = component === axis ? tile * FINEST_RESOLUTION : 0;
            centerFineHalf[component] =
              2 * tileOffsetFine + strideFine * (2 * local[component] + 1);
            center[component] =
              (centerFineHalf[component] * tileWidth) / (2 * FINEST_RESOLUTION);
          }
          const seamCoordinate = tile === 0 ? resolution - 1 : 0;
          cells.push({
            id: cells.length,
            tile,
            resolution,
            local,
            centerFineHalf,
            center,
            width,
            volume: width * width * width,
            touchesSeam: local[axis] === seamCoordinate,
          });
        }
      }
    }
  }

  const cellId = (tile: 0 | 1, local: Vec3i): number =>
    cellBases[tile] + localLinearIndex(local, resolutions[tile]);

  const regularFaces: CompositeGradientRow[] = [];
  for (const tile of [0, 1] as const) {
    const resolution = resolutions[tile];
    const strideFine = FINEST_RESOLUTION / resolution;
    const width = tileWidth / resolution;
    for (const faceAxis of [0, 1, 2] as const) {
      for (let z = 0; z < resolution; z += 1) {
        for (let y = 0; y < resolution; y += 1) {
          for (let x = 0; x < resolution; x += 1) {
            const localFace: Vec3i = [x, y, z];
            if (localFace[faceAxis] === 0) continue;
            const negative = [...localFace] as [number, number, number];
            negative[faceAxis] -= 1;
            const positive = localFace;
            const centerFineHalf = mutableVec3();
            const center = mutableVec3();
            for (const component of [0, 1, 2] as const) {
              const tileOffsetFine = component === axis ? tile * FINEST_RESOLUTION : 0;
              centerFineHalf[component] = component === faceAxis
                ? 2 * (tileOffsetFine + strideFine * localFace[component])
                : 2 * tileOffsetFine + strideFine * (2 * localFace[component] + 1);
              center[component] =
                (centerFineHalf[component] * tileWidth) / (2 * FINEST_RESOLUTION);
            }
            const area = width * width;
            regularFaces.push({
              id: regularFaces.length,
              kind: "regular",
              axis: faceAxis,
              tile,
              localFace,
              centerFineHalf,
              center,
              area,
              distance: width,
              dualWeight: area * width,
              terms: [
                { cellId: cellId(tile, negative), coefficient: -1 / width },
                { cellId: cellId(tile, positive), coefficient: 1 / width },
              ],
            });
          }
        }
      }
    }
  }

  const seamPorts: CompositeGradientRow[] = [];
  const negativeResolution = resolutions[0];
  const positiveResolution = resolutions[1];
  const seamResolution = Math.min(negativeResolution, positiveResolution) as TwoTileResolution;
  const negativeWidth = tileWidth / negativeResolution;
  const positiveWidth = tileWidth / positiveResolution;
  const distance = 0.5 * (negativeWidth + positiveWidth);
  const area = (tileWidth / seamResolution) ** 2;
  const seamFineHalf = 2 * FINEST_RESOLUTION;

  for (let portV = 0; portV < seamResolution; portV += 1) {
    for (let portU = 0; portU < seamResolution; portU += 1) {
      const centerFineHalf = mutableVec3();
      centerFineHalf[axis] = seamFineHalf;
      const coarseStrideFine = FINEST_RESOLUTION / seamResolution;
      centerFineHalf[tangents[0]] = coarseStrideFine * (2 * portU + 1);
      centerFineHalf[tangents[1]] = coarseStrideFine * (2 * portV + 1);
      const center: [number, number, number] = [
        (centerFineHalf[0] * tileWidth) / (2 * FINEST_RESOLUTION),
        (centerFineHalf[1] * tileWidth) / (2 * FINEST_RESOLUTION),
        (centerFineHalf[2] * tileWidth) / (2 * FINEST_RESOLUTION),
      ];

      const sideTerms = (tile: 0 | 1, sign: -1 | 1): CompositeGradientTerm[] => {
        const resolution = resolutions[tile];
        const ratio = resolution / seamResolution;
        const coefficient = sign / (distance * ratio * ratio);
        const result: CompositeGradientTerm[] = [];
        for (let childV = 0; childV < ratio; childV += 1) {
          for (let childU = 0; childU < ratio; childU += 1) {
            const local = mutableVec3();
            local[axis] = tile === 0 ? resolution - 1 : 0;
            local[tangents[0]] = ratio * portU + childU;
            local[tangents[1]] = ratio * portV + childV;
            result.push({ cellId: cellId(tile, local), coefficient });
          }
        }
        return result;
      };

      seamPorts.push({
        id: regularFaces.length + seamPorts.length,
        kind: "seam",
        axis,
        port: [portU, portV],
        centerFineHalf,
        center,
        area,
        distance,
        dualWeight: area * distance,
        terms: [...sideTerms(0, -1), ...sideTerms(1, 1)],
      });
    }
  }

  const gradientRows = [...regularFaces, ...seamPorts];
  const seamCellIds = cells.filter((cell) => cell.touchesSeam).map((cell) => cell.id);
  return {
    axis,
    tangentialAxes: tangents,
    tileWidth,
    finestResolution: FINEST_RESOLUTION,
    negativeResolution,
    positiveResolution,
    cells,
    regularFaces,
    seamPorts,
    gradientRows,
    seamCellIds,
  };
}

/** Apply the sole authoritative gradient G. */
export function applyCompositeGradient(
  grid: TwoTileCompositeGrid,
  pressure: ArrayLike<number>,
  output: Float64Array = new Float64Array(grid.gradientRows.length),
): Float64Array {
  if (pressure.length !== grid.cells.length) {
    throw new RangeError(`pressure has ${pressure.length} entries; expected ${grid.cells.length}`);
  }
  if (output.length !== grid.gradientRows.length) {
    throw new RangeError(`gradient output has ${output.length} entries; expected ${grid.gradientRows.length}`);
  }
  for (const row of grid.gradientRows) {
    let value = 0;
    for (const term of row.terms) value += term.coefficient * pressure[term.cellId];
    output[row.id] = value;
  }
  return output;
}

/**
 * Apply D = -M^-1 G^T W, where M is the diagonal cell-volume measure.
 * `normalVelocity[row.id]` is the one authoritative velocity for that face or
 * seam port.
 */
export function applyCompositeDivergence(
  grid: TwoTileCompositeGrid,
  normalVelocity: ArrayLike<number>,
  output: Float64Array = new Float64Array(grid.cells.length),
): Float64Array {
  if (normalVelocity.length !== grid.gradientRows.length) {
    throw new RangeError(
      `normalVelocity has ${normalVelocity.length} entries; expected ${grid.gradientRows.length}`,
    );
  }
  if (output.length !== grid.cells.length) {
    throw new RangeError(`divergence output has ${output.length} entries; expected ${grid.cells.length}`);
  }
  output.fill(0);
  for (const row of grid.gradientRows) {
    const weightedVelocity = row.dualWeight * normalVelocity[row.id];
    for (const term of row.terms) {
      output[term.cellId] -= term.coefficient * weightedVelocity;
    }
  }
  for (const cell of grid.cells) output[cell.id] /= cell.volume;
  return output;
}

/** Apply the symmetric positive-semidefinite pressure operator A = G^T W G. */
export function applyCompositePressureOperator(
  grid: TwoTileCompositeGrid,
  pressure: ArrayLike<number>,
  output: Float64Array = new Float64Array(grid.cells.length),
): Float64Array {
  if (output.length !== grid.cells.length) {
    throw new RangeError(`pressure output has ${output.length} entries; expected ${grid.cells.length}`);
  }
  const gradient = applyCompositeGradient(grid, pressure);
  output.fill(0);
  for (const row of grid.gradientRows) {
    const weightedGradient = row.dualWeight * gradient[row.id];
    for (const term of row.terms) {
      output[term.cellId] += term.coefficient * weightedGradient;
    }
  }
  return output;
}

/** Assemble a row-major dense oracle for the exact same gradient rows. */
export function assembleDensePressureMatrix(grid: TwoTileCompositeGrid): DensePressureMatrix {
  const size = grid.cells.length;
  const values = new Float64Array(size * size);
  for (const row of grid.gradientRows) {
    for (const left of row.terms) {
      const rowOffset = left.cellId * size;
      for (const right of row.terms) {
        values[rowOffset + right.cellId] +=
          row.dualWeight * left.coefficient * right.coefficient;
      }
    }
  }
  return { size, values };
}

/** Apply a row-major dense pressure matrix. Useful for GPU/readback comparison. */
export function applyDensePressureMatrix(
  matrix: DensePressureMatrix,
  pressure: ArrayLike<number>,
  output: Float64Array = new Float64Array(matrix.size),
): Float64Array {
  if (pressure.length !== matrix.size || output.length !== matrix.size) {
    throw new RangeError(`dense matrix/vector size mismatch for ${matrix.size} rows`);
  }
  for (let row = 0; row < matrix.size; row += 1) {
    let sum = 0;
    const offset = row * matrix.size;
    for (let column = 0; column < matrix.size; column += 1) {
      sum += matrix.values[offset + column] * pressure[column];
    }
    output[row] = sum;
  }
  return output;
}

/** Return x^T A x, evaluated through the row representation without assembly. */
export function compositePressureEnergy(
  grid: TwoTileCompositeGrid,
  pressure: ArrayLike<number>,
): number {
  const gradient = applyCompositeGradient(grid, pressure);
  let energy = 0;
  for (const row of grid.gradientRows) {
    energy += row.dualWeight * gradient[row.id] * gradient[row.id];
  }
  return energy;
}

/**
 * Copy and pin one pressure row/column, making the closed-domain Laplacian SPD.
 * This is intentionally explicit: silently adding a diagonal shift would hide
 * a null-space or connectivity error in the seam graph.
 */
export function pinDensePressureSystem(
  matrix: DensePressureMatrix,
  rhs: ArrayLike<number>,
  pinCellId = 0,
  pinValue = 0,
): { matrix: DensePressureMatrix; rhs: Float64Array } {
  if (rhs.length !== matrix.size) throw new RangeError("dense pressure RHS size mismatch");
  if (!Number.isInteger(pinCellId) || pinCellId < 0 || pinCellId >= matrix.size) {
    throw new RangeError(`pinCellId ${pinCellId} is outside [0, ${matrix.size})`);
  }
  const values = matrix.values.slice();
  const pinnedRhs = Float64Array.from(rhs);
  for (let row = 0; row < matrix.size; row += 1) {
    if (row !== pinCellId) {
      pinnedRhs[row] -= values[row * matrix.size + pinCellId] * pinValue;
    }
    values[row * matrix.size + pinCellId] = 0;
    values[pinCellId * matrix.size + row] = 0;
  }
  values[pinCellId * matrix.size + pinCellId] = 1;
  pinnedRhs[pinCellId] = pinValue;
  return { matrix: { size: matrix.size, values }, rhs: pinnedRhs };
}

/**
 * Dense Cholesky oracle. It is diagnostic code, not the planned runtime solver.
 * The caller must first remove the constant-pressure null space (for example
 * with pinDensePressureSystem) or provide a genuinely Dirichlet-constrained A.
 */
export function solveDenseSymmetricPositiveDefinite(
  matrix: DensePressureMatrix,
  rhs: ArrayLike<number>,
): Float64Array {
  const size = matrix.size;
  if (rhs.length !== size) throw new RangeError("dense pressure RHS size mismatch");
  const lower = new Float64Array(size * size);
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column <= row; column += 1) {
      let value = matrix.values[row * size + column];
      for (let k = 0; k < column; k += 1) {
        value -= lower[row * size + k] * lower[column * size + k];
      }
      if (row === column) {
        if (!(value > 0) || !Number.isFinite(value)) {
          throw new Error(`pressure matrix is not SPD at row ${row}; pivot=${value}`);
        }
        lower[row * size + column] = Math.sqrt(value);
      } else {
        lower[row * size + column] = value / lower[column * size + column];
      }
    }
  }

  const y = new Float64Array(size);
  for (let row = 0; row < size; row += 1) {
    let value = rhs[row];
    for (let column = 0; column < row; column += 1) {
      value -= lower[row * size + column] * y[column];
    }
    y[row] = value / lower[row * size + row];
  }
  const solution = new Float64Array(size);
  for (let row = size - 1; row >= 0; row -= 1) {
    let value = y[row];
    for (let column = row + 1; column < size; column += 1) {
      value -= lower[column * size + row] * solution[column];
    }
    solution[row] = value / lower[row * size + row];
  }
  return solution;
}

function probeVariant(
  axis: CompositeAxis,
  negativeResolution: TwoTileResolution,
  positiveResolution: TwoTileResolution,
): TwoTileCompositeVariantProbe {
  const grid = buildTwoTileCompositeGrid({ axis, negativeResolution, positiveResolution });
  const constant = new Float64Array(grid.cells.length).fill(3.25);
  const constantGradientMaxAbs = maxAbs(applyCompositeGradient(grid, constant));

  const linear = Float64Array.from(grid.cells, (cell) => cell.center[axis]);
  const linearGradient = applyCompositeGradient(grid, linear);
  let linearGradientMaxAbs = 0;
  for (const row of grid.gradientRows) {
    const expected = row.axis === axis ? 1 : 0;
    linearGradientMaxAbs = Math.max(
      linearGradientMaxAbs,
      Math.abs(linearGradient[row.id] - expected),
    );
  }

  const constantNormalVelocity = Float64Array.from(
    grid.gradientRows,
    (row) => (row.axis === axis ? 1 : 0),
  );
  const constantDivergence = applyCompositeDivergence(grid, constantNormalVelocity);
  let constantNormalSeamDivergenceMaxAbs = 0;
  for (const cellId of grid.seamCellIds) {
    constantNormalSeamDivergenceMaxAbs = Math.max(
      constantNormalSeamDivergenceMaxAbs,
      Math.abs(constantDivergence[cellId]),
    );
  }

  const pressure = Float64Array.from(grid.cells, (cell) => {
    const [x, y, z] = cell.center;
    return Math.sin(0.37 + 0.91 * x) + 0.31 * y * y - 0.17 * z + 0.07 * x * z;
  });
  const velocity = Float64Array.from(grid.gradientRows, (row) => {
    const [x, y, z] = row.center;
    return Math.cos(0.23 + 0.43 * x - 0.29 * y + 0.19 * z);
  });
  const gradient = applyCompositeGradient(grid, pressure);
  const divergence = applyCompositeDivergence(grid, velocity);
  const weightedFaceVelocity = Float64Array.from(
    velocity,
    (value, rowId) => value * grid.gradientRows[rowId].dualWeight,
  );
  const weightedCellDivergence = Float64Array.from(
    divergence,
    (value, cellId) => value * grid.cells[cellId].volume,
  );
  const transposeError = Math.abs(dot(gradient, weightedFaceVelocity) + dot(pressure, weightedCellDivergence));

  const dense = assembleDensePressureMatrix(grid);
  let symmetryMaxAbs = 0;
  for (let row = 0; row < dense.size; row += 1) {
    for (let column = 0; column < row; column += 1) {
      symmetryMaxAbs = Math.max(
        symmetryMaxAbs,
        Math.abs(dense.values[row * dense.size + column] - dense.values[column * dense.size + row]),
      );
    }
  }
  const matrixFree = applyCompositePressureOperator(grid, pressure);
  const denseApplied = applyDensePressureMatrix(dense, pressure);
  for (let i = 0; i < dense.size; i += 1) {
    symmetryMaxAbs = Math.max(symmetryMaxAbs, Math.abs(matrixFree[i] - denseApplied[i]));
  }

  let minimumRayleigh = Number.POSITIVE_INFINITY;
  for (let mode = 1; mode <= 4; mode += 1) {
    const sample = Float64Array.from(grid.cells, (cell) => {
      const [x, y, z] = cell.center;
      return Math.sin(mode * 0.61 * x + 0.17 * y)
        + Math.cos(mode * 0.43 * y - 0.11 * z)
        + 0.13 * mode * x * z;
    });
    const mean = sample.reduce((sum, value) => sum + value, 0) / sample.length;
    for (let i = 0; i < sample.length; i += 1) sample[i] -= mean;
    minimumRayleigh = Math.min(
      minimumRayleigh,
      compositePressureEnergy(grid, sample) / dot(sample, sample),
    );
  }

  const seamTermCounts = grid.seamPorts.map((port) => port.terms.length);
  return {
    axis,
    negativeResolution,
    positiveResolution,
    cellCount: grid.cells.length,
    regularFaceCount: grid.regularFaces.length,
    seamPortCount: grid.seamPorts.length,
    seamTermCountMin: Math.min(...seamTermCounts),
    seamTermCountMax: Math.max(...seamTermCounts),
    constantGradientMaxAbs,
    linearGradientMaxAbs,
    constantNormalSeamDivergenceMaxAbs,
    transposeError,
    symmetryMaxAbs,
    quadraticEnergy: compositePressureEnergy(grid, pressure),
    minimumRayleigh,
  };
}

function reflectedCellMap(
  source: TwoTileCompositeGrid,
  target: TwoTileCompositeGrid,
): Int32Array {
  const key = (resolution: number, center: Vec3i): string => `${resolution}:${center.join(",")}`;
  const targetByPosition = new Map<string, number>();
  for (const cell of target.cells) targetByPosition.set(key(cell.resolution, cell.centerFineHalf), cell.id);
  const domainFineHalf = 4 * FINEST_RESOLUTION;
  return Int32Array.from(source.cells, (cell) => {
    const reflected = [...cell.centerFineHalf] as [number, number, number];
    reflected[source.axis] = domainFineHalf - reflected[source.axis];
    const targetId = targetByPosition.get(key(cell.resolution, reflected));
    if (targetId === undefined) {
      throw new Error(`no reflected cell for ${key(cell.resolution, reflected)}`);
    }
    return targetId;
  });
}

function probeSwapSymmetry(axis: CompositeAxis): number {
  const fineNegative = buildTwoTileCompositeGrid({
    axis,
    negativeResolution: 8,
    positiveResolution: 4,
  });
  const finePositive = buildTwoTileCompositeGrid({
    axis,
    negativeResolution: 4,
    positiveResolution: 8,
  });
  const map = reflectedCellMap(fineNegative, finePositive);
  const sourceMatrix = assembleDensePressureMatrix(fineNegative);
  const targetMatrix = assembleDensePressureMatrix(finePositive);
  let maximum = 0;
  for (let sourceRow = 0; sourceRow < sourceMatrix.size; sourceRow += 1) {
    const targetRow = map[sourceRow];
    for (let sourceColumn = 0; sourceColumn < sourceMatrix.size; sourceColumn += 1) {
      const targetColumn = map[sourceColumn];
      maximum = Math.max(
        maximum,
        Math.abs(
          sourceMatrix.values[sourceRow * sourceMatrix.size + sourceColumn]
            - targetMatrix.values[targetRow * targetMatrix.size + targetColumn],
        ),
      );
    }
  }
  return maximum;
}

/**
 * Executable, non-test-runner invariant probe for all axes, both seam
 * orientations, and the equal-resolution adapter-collapse controls.
 */
export function probeTwoTileCompositeGrid(tolerance = 1e-11): TwoTileCompositeProbeResult {
  if (!Number.isFinite(tolerance) || tolerance <= 0) {
    throw new RangeError(`tolerance must be finite and positive; received ${tolerance}`);
  }
  const variants: TwoTileCompositeVariantProbe[] = [];
  let swapSymmetryMaxAbs = 0;
  const swapSymmetryByAxis: TwoTileSwapSymmetryProbe[] = [];
  for (const axis of [0, 1, 2] as const) {
    variants.push(probeVariant(axis, 8, 8));
    variants.push(probeVariant(axis, 4, 4));
    variants.push(probeVariant(axis, 8, 4));
    variants.push(probeVariant(axis, 4, 8));
    const maximumMatrixDifference = probeSwapSymmetry(axis);
    swapSymmetryMaxAbs = Math.max(swapSymmetryMaxAbs, maximumMatrixDifference);
    swapSymmetryByAxis.push({
      axis,
      negativeResolution: 8,
      positiveResolution: 4,
      reflectedNegativeResolution: 4,
      reflectedPositiveResolution: 8,
      maximumMatrixDifference,
    });
  }

  const failures: string[] = [];
  for (const variant of variants) {
    const label = `axis=${variant.axis} ${variant.negativeResolution}+${variant.positiveResolution}`;
    if (variant.constantGradientMaxAbs > tolerance) failures.push(`${label}: constant gradient`);
    if (variant.linearGradientMaxAbs > tolerance) failures.push(`${label}: linear gradient`);
    if (variant.constantNormalSeamDivergenceMaxAbs > tolerance) {
      failures.push(`${label}: constant normal seam divergence`);
    }
    if (variant.transposeError > tolerance) failures.push(`${label}: transpose identity`);
    if (variant.symmetryMaxAbs > tolerance) failures.push(`${label}: dense symmetry/apply`);
    if (variant.quadraticEnergy < -tolerance) failures.push(`${label}: negative pressure energy`);
    if (variant.minimumRayleigh < -tolerance) failures.push(`${label}: negative Rayleigh quotient`);
    const expectedSeamPorts = Math.min(
      variant.negativeResolution,
      variant.positiveResolution,
    ) ** 2;
    if (variant.seamPortCount !== expectedSeamPorts) failures.push(`${label}: seam port count`);
  }
  if (swapSymmetryMaxAbs > tolerance) failures.push("fine/coarse reflection symmetry");
  return {
    passed: failures.length === 0,
    tolerance,
    variants,
    swapSymmetryMaxAbs,
    swapSymmetryByAxis,
    failures,
  };
}

/** Throw with compact diagnostics if the executable oracle fails. */
export function assertTwoTileCompositeInvariants(tolerance = 1e-11): TwoTileCompositeProbeResult {
  const result = probeTwoTileCompositeGrid(tolerance);
  if (!result.passed) {
    throw new Error(`two-tile composite invariant failure: ${result.failures.join("; ")}`);
  }
  return result;
}

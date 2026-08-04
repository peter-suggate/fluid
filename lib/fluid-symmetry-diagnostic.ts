export type HorizontalSymmetryTransform = "reflect-x" | "reflect-z" | "swap-xz";

export interface SymmetryMismatchLocation {
  readonly transform: HorizontalSymmetryTransform;
  readonly source: readonly [number, number, number];
  readonly target: readonly [number, number, number];
  readonly component?: "x" | "y" | "z";
  readonly sourceValue: number;
  readonly expectedValue: number;
  readonly targetValue: number;
  readonly absoluteError: number;
}

export interface SymmetryFieldMetrics {
  readonly comparedValues: number;
  readonly exactMismatchCount: number;
  readonly nonFiniteCount: number;
  readonly maximumAbsoluteError: number;
  readonly worst?: SymmetryMismatchLocation;
  readonly first?: SymmetryMismatchLocation;
}

export interface SymmetryWallState {
  readonly maximumVolume: number;
  readonly wetCellCount: number;
  readonly touched: boolean;
}

export interface HorizontalFrontCircularity {
  readonly threshold: number;
  readonly angularSampleCount: number;
  readonly meanRadius_cells: number;
  readonly minimumRadius_cells: number;
  readonly maximumRadius_cells: number;
  /** RMS distance from the all-angle mean contour radius. */
  readonly radialRmsDeviation_cells: number;
  /** Worst absolute distance from the all-angle mean contour radius. */
  readonly radialMaximumDeviation_cells: number;
  readonly axisRadius_cells: number;
  readonly diagonalRadius_cells: number;
  /** Positive means the axis-aligned front leads the diagonal front. */
  readonly axisLead_cells: number;
  readonly diagonalToAxisRatio: number;
}

export interface FluidSymmetryObservation {
  readonly time_s: number;
  readonly grid: readonly [number, number, number];
  readonly volume: SymmetryFieldMetrics;
  readonly velocity: SymmetryFieldMetrics;
  readonly pressure: SymmetryFieldMetrics;
  /** Assembled Section 4.3 right-hand side, before MGPCG. */
  readonly rhs: SymmetryFieldMetrics;
  /** Assembled Section 4.3 operator diagonal, before MGPCG. */
  readonly diagonal: SymmetryFieldMetrics;
  readonly section63Diagonal?: SymmetryFieldMetrics;
  readonly section63CaseId?: SymmetryFieldMetrics;
  readonly initialResidual?: SymmetryFieldMetrics;
  readonly initialPreconditioned?: SymmetryFieldMetrics;
  readonly initialPreconditionedImage?: SymmetryFieldMetrics;
  readonly preconditionerPreSmoothed?: SymmetryFieldMetrics;
  readonly preconditionerZeroSmoothed?: SymmetryFieldMetrics;
  readonly preconditionerFirstOperatorImage?: SymmetryFieldMetrics;
  readonly preconditionerFirstSmoothed?: SymmetryFieldMetrics;
  readonly preconditionerInnerResidual?: SymmetryFieldMetrics;
  readonly preconditionerInnerCorrection?: SymmetryFieldMetrics;
  readonly preconditionerPostCorrected?: SymmetryFieldMetrics;
  readonly topology: SymmetryFieldMetrics;
  /** Bottom-layer half-volume contour sampled along the four axes/diagonals. */
  readonly frontCircularity: HorizontalFrontCircularity;
  readonly walls: Readonly<Record<"negativeX" | "positiveX" | "negativeZ" | "positiveZ", SymmetryWallState>>;
}

/**
 * Measure radial front shape independently of D4 symmetry. A square or
 * axis-biased field can be exactly D4 while still being far from circular.
 * Bilinear ray samples avoid making the result depend on whether an interface
 * happens to cross a cell centre.
 */
export function measureHorizontalFrontCircularity(
  volume: ArrayLike<number>,
  grid: readonly [number, number, number],
  threshold = 0.5,
  angularSampleCount = 64,
): HorizontalFrontCircularity {
  const [nx, ny, nz] = grid;
  if (!(threshold > 0 && threshold < 1) || nx < 2 || ny < 1 || nz < 2
    || !Number.isSafeInteger(angularSampleCount) || angularSampleCount < 16
    || angularSampleCount % 8 !== 0) {
    throw new RangeError("Horizontal front circularity needs a 2-D grid, threshold in (0,1), and an angular sample count divisible by eight and at least sixteen");
  }
  const cx = 0.5 * (nx - 1), cz = 0.5 * (nz - 1);
  const alpha = (x: number, z: number): number => {
    if (x < 0 || x > nx - 1 || z < 0 || z > nz - 1) return 0;
    const x0 = Math.floor(x), z0 = Math.floor(z);
    const x1 = Math.min(nx - 1, x0 + 1), z1 = Math.min(nz - 1, z0 + 1);
    const tx = x - x0, tz = z - z0;
    const at = (ix: number, iz: number) => Number(volume[ix + nx * ny * iz] ?? 0);
    return (1 - tz) * ((1 - tx) * at(x0, z0) + tx * at(x1, z0))
      + tz * ((1 - tx) * at(x0, z1) + tx * at(x1, z1));
  };
  const radius = (dx: number, dz: number): number => {
    const bound = Math.min(
      dx > 0 ? (nx - 1 - cx) / dx : dx < 0 ? cx / -dx : Infinity,
      dz > 0 ? (nz - 1 - cz) / dz : dz < 0 ? cz / -dz : Infinity,
    );
    const step = 1 / 32;
    let previousR = 0, previous = alpha(cx, cz), crossing = 0;
    for (let r = step; r <= bound + 1e-9; r += step) {
      const value = alpha(cx + r * dx, cz + r * dz);
      if (previous >= threshold && value < threshold) {
        const fraction = (previous - threshold) / Math.max(previous - value, 1e-12);
        crossing = previousR + fraction * (r - previousR);
      }
      previousR = r; previous = value;
    }
    if (previous >= threshold) crossing = bound;
    return crossing;
  };
  const invSqrt2 = Math.SQRT1_2;
  const axes = [[1, 0], [-1, 0], [0, 1], [0, -1]] as const;
  const diagonals = [[invSqrt2, invSqrt2], [invSqrt2, -invSqrt2],
    [-invSqrt2, invSqrt2], [-invSqrt2, -invSqrt2]] as const;
  const mean = (directions: readonly (readonly [number, number])[]) =>
    directions.reduce((sum, [dx, dz]) => sum + radius(dx, dz), 0) / directions.length;
  const axisRadius_cells = mean(axes), diagonalRadius_cells = mean(diagonals);
  const radii = Array.from({ length: angularSampleCount }, (_, sample) => {
    const angle = 2 * Math.PI * sample / angularSampleCount;
    return radius(Math.cos(angle), Math.sin(angle));
  });
  const meanRadius_cells = radii.reduce((sum, value) => sum + value, 0) / radii.length;
  const deviations = radii.map((value) => Math.abs(value - meanRadius_cells));
  const radialRmsDeviation_cells = Math.sqrt(radii.reduce(
    (sum, value) => sum + (value - meanRadius_cells) ** 2, 0,
  ) / radii.length);
  return Object.freeze({ threshold, angularSampleCount, meanRadius_cells,
    minimumRadius_cells: Math.min(...radii), maximumRadius_cells: Math.max(...radii),
    radialRmsDeviation_cells,
    radialMaximumDeviation_cells: Math.max(...deviations),
    axisRadius_cells, diagonalRadius_cells,
    axisLead_cells: axisRadius_cells - diagonalRadius_cells,
    diagonalToAxisRatio: axisRadius_cells > 0 ? diagonalRadius_cells / axisRadius_cells : 1 });
}

export interface FluidSymmetryState {
  readonly time_s: number;
  readonly grid: readonly [number, number, number];
  readonly volume: ArrayLike<number>;
  /** Collocated xyz components, three consecutive values per finest cell. */
  readonly velocity: ArrayLike<number>;
  /** Solved pressure potential expanded from adaptive rows to the finest lattice. */
  readonly pressure: ArrayLike<number>;
  readonly rhs: ArrayLike<number>;
  readonly diagonal: ArrayLike<number>;
  readonly section63Diagonal?: ArrayLike<number>;
  readonly section63CaseId?: ArrayLike<number>;
  readonly initialResidual?: ArrayLike<number>;
  readonly initialPreconditioned?: ArrayLike<number>;
  readonly initialPreconditionedImage?: ArrayLike<number>;
  readonly preconditionerPreSmoothed?: ArrayLike<number>;
  readonly preconditionerZeroSmoothed?: ArrayLike<number>;
  readonly preconditionerFirstOperatorImage?: ArrayLike<number>;
  readonly preconditionerFirstSmoothed?: ArrayLike<number>;
  readonly preconditionerInnerResidual?: ArrayLike<number>;
  readonly preconditionerInnerCorrection?: ArrayLike<number>;
  readonly preconditionerPostCorrected?: ArrayLike<number>;
  /** Adaptive leaf size expanded to every finest cell. */
  readonly topology: ArrayLike<number>;
  readonly wallLiquidThreshold: number;
}

const transforms: readonly HorizontalSymmetryTransform[] = ["reflect-x", "reflect-z", "swap-xz"];

function targetCoordinate(
  transform: HorizontalSymmetryTransform,
  x: number,
  y: number,
  z: number,
  nx: number,
  nz: number,
): readonly [number, number, number] {
  if (transform === "reflect-x") return [nx - 1 - x, y, z];
  if (transform === "reflect-z") return [x, y, nz - 1 - z];
  return [z, y, x];
}

function expectedVector(
  transform: HorizontalSymmetryTransform,
  x: number,
  y: number,
  z: number,
): readonly [number, number, number] {
  if (transform === "reflect-x") return [-x, y, z];
  if (transform === "reflect-z") return [x, y, -z];
  return [z, y, x];
}

function emptyMetrics(): {
  comparedValues: number;
  exactMismatchCount: number;
  nonFiniteCount: number;
  maximumAbsoluteError: number;
  worst?: SymmetryMismatchLocation;
  first?: SymmetryMismatchLocation;
} {
  return { comparedValues: 0, exactMismatchCount: 0, nonFiniteCount: 0, maximumAbsoluteError: 0 };
}

function compareScalarField(
  field: ArrayLike<number>,
  grid: readonly [number, number, number],
): SymmetryFieldMetrics {
  const [nx, ny, nz] = grid;
  const result = emptyMetrics();
  for (const transform of transforms) {
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const target = targetCoordinate(transform, x, y, z, nx, nz);
      const sourceIndex = x + nx * (y + ny * z);
      const targetIndex = target[0] + nx * (target[1] + ny * target[2]);
      const sourceValue = Number(field[sourceIndex]);
      const targetValue = Number(field[targetIndex]);
      const sourceFinite = Number.isFinite(sourceValue), targetFinite = Number.isFinite(targetValue);
      // Compact pressure/velocity fields are intentionally undefined outside
      // their adaptive support. A symmetric absent pair is not a bad value;
      // one-sided presence is the actual support-symmetry failure.
      if (!sourceFinite && !targetFinite) continue;
      result.comparedValues += 1;
      if (!sourceFinite || !targetFinite) {
        result.nonFiniteCount += 1;
        continue;
      }
      const error = Math.abs(targetValue - sourceValue);
      if (!Object.is(targetValue, sourceValue)) result.exactMismatchCount += 1;
      if (error > 0 && !result.first) result.first = {
        transform, source: [x, y, z], target,
        sourceValue, expectedValue: sourceValue, targetValue, absoluteError: error,
      };
      if (error > result.maximumAbsoluteError) {
        result.maximumAbsoluteError = error;
        result.worst = {
          transform, source: [x, y, z], target,
          sourceValue, expectedValue: sourceValue, targetValue, absoluteError: error,
        };
      }
    }
  }
  return result;
}

function compareVelocityField(
  field: ArrayLike<number>,
  grid: readonly [number, number, number],
): SymmetryFieldMetrics {
  const [nx, ny, nz] = grid;
  const result = emptyMetrics();
  const componentNames = ["x", "y", "z"] as const;
  for (const transform of transforms) {
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const target = targetCoordinate(transform, x, y, z, nx, nz);
      const sourceCell = x + nx * (y + ny * z);
      const targetCell = target[0] + nx * (target[1] + ny * target[2]);
      const source = [Number(field[3 * sourceCell]), Number(field[3 * sourceCell + 1]), Number(field[3 * sourceCell + 2])] as const;
      const expected = expectedVector(transform, ...source);
      for (let component = 0; component < 3; component += 1) {
        const sourceValue = source[component];
        const expectedValue = expected[component];
        const targetValue = Number(field[3 * targetCell + component]);
        const expectedFinite = Number.isFinite(expectedValue), targetFinite = Number.isFinite(targetValue);
        if (!expectedFinite && !targetFinite) continue;
        result.comparedValues += 1;
        if (!expectedFinite || !targetFinite) {
          result.nonFiniteCount += 1;
          result.first ??= {
            transform, source: [x, y, z], target,
            component: componentNames[component], sourceValue, expectedValue,
            targetValue, absoluteError: Number.POSITIVE_INFINITY,
          };
          continue;
        }
        const error = Math.abs(targetValue - expectedValue);
        if (!Object.is(targetValue, expectedValue)) result.exactMismatchCount += 1;
        const mismatch = {
          transform, source: [x, y, z] as const, target,
          component: componentNames[component], sourceValue, expectedValue, targetValue, absoluteError: error,
        };
        if (error > 0 && !result.first) result.first = mismatch;
        if (error > result.maximumAbsoluteError) {
          result.maximumAbsoluteError = error;
          result.worst = mismatch;
        }
      }
    }
  }
  return result;
}

function wallState(values: readonly number[], threshold: number): SymmetryWallState {
  let maximumVolume = Number.NEGATIVE_INFINITY, wetCellCount = 0;
  for (const value of values) {
    if (Number.isFinite(value)) maximumVolume = Math.max(maximumVolume, value);
    if (value >= threshold) wetCellCount += 1;
  }
  return { maximumVolume, wetCellCount, touched: wetCellCount > 0 };
}

/**
 * Measures the three generators of the horizontal D4 symmetry group. If these
 * comparisons pass, both axis reflections, both diagonal reflections, and all
 * four quarter turns pass. Gravity deliberately leaves y unchanged.
 */
export function measureFluidSymmetry(state: FluidSymmetryState): FluidSymmetryObservation {
  const [nx, ny, nz] = state.grid;
  if (![nx, ny, nz].every((value) => Number.isSafeInteger(value) && value > 0) || nx !== nz) {
    throw new RangeError("Fluid symmetry requires a positive grid with equal x/z dimensions");
  }
  const cellCount = nx * ny * nz;
  for (const [name, field, expectedLength] of [
    ["volume", state.volume, cellCount],
    ["velocity", state.velocity, 3 * cellCount],
    ["pressure", state.pressure, cellCount],
    ["rhs", state.rhs, cellCount],
    ["diagonal", state.diagonal, cellCount],
    ["topology", state.topology, cellCount],
  ] as const) {
    if (field.length !== expectedLength) throw new RangeError(`${name} symmetry field has ${field.length} values; expected ${expectedLength}`);
  }
  for (const [name, field] of [
    ["initialResidual", state.initialResidual],
    ["section63Diagonal", state.section63Diagonal],
    ["section63CaseId", state.section63CaseId],
    ["initialPreconditioned", state.initialPreconditioned],
    ["initialPreconditionedImage", state.initialPreconditionedImage],
    ["preconditionerPreSmoothed", state.preconditionerPreSmoothed],
    ["preconditionerZeroSmoothed", state.preconditionerZeroSmoothed],
    ["preconditionerFirstOperatorImage", state.preconditionerFirstOperatorImage],
    ["preconditionerFirstSmoothed", state.preconditionerFirstSmoothed],
    ["preconditionerInnerResidual", state.preconditionerInnerResidual],
    ["preconditionerInnerCorrection", state.preconditionerInnerCorrection],
    ["preconditionerPostCorrected", state.preconditionerPostCorrected],
  ] as const) {
    if (field && field.length !== cellCount) {
      throw new RangeError(`${name} symmetry field has ${field.length} values; expected ${cellCount}`);
    }
  }
  if (!(state.wallLiquidThreshold >= 0) || !Number.isFinite(state.wallLiquidThreshold)) {
    throw new RangeError("Wall liquid threshold must be finite and non-negative");
  }

  const negativeX: number[] = [], positiveX: number[] = [], negativeZ: number[] = [], positiveZ: number[] = [];
  for (let y = 0; y < ny; y += 1) for (let q = 0; q < nx; q += 1) {
    negativeX.push(Number(state.volume[nx * (y + ny * q)]));
    positiveX.push(Number(state.volume[(nx - 1) + nx * (y + ny * q)]));
    negativeZ.push(Number(state.volume[q + nx * y]));
    positiveZ.push(Number(state.volume[q + nx * (y + ny * (nz - 1))]));
  }

  return {
    time_s: state.time_s,
    grid: [...state.grid],
    volume: compareScalarField(state.volume, state.grid),
    velocity: compareVelocityField(state.velocity, state.grid),
    pressure: compareScalarField(state.pressure, state.grid),
    rhs: compareScalarField(state.rhs, state.grid),
    diagonal: compareScalarField(state.diagonal, state.grid),
    ...(state.section63Diagonal
      ? { section63Diagonal: compareScalarField(state.section63Diagonal, state.grid) } : {}),
    ...(state.section63CaseId
      ? { section63CaseId: compareScalarField(state.section63CaseId, state.grid) } : {}),
    ...(state.initialResidual ? { initialResidual: compareScalarField(state.initialResidual, state.grid) } : {}),
    ...(state.initialPreconditioned
      ? { initialPreconditioned: compareScalarField(state.initialPreconditioned, state.grid) } : {}),
    ...(state.initialPreconditionedImage
      ? { initialPreconditionedImage: compareScalarField(state.initialPreconditionedImage, state.grid) } : {}),
    ...(state.preconditionerPreSmoothed
      ? { preconditionerPreSmoothed: compareScalarField(state.preconditionerPreSmoothed, state.grid) } : {}),
    ...(state.preconditionerZeroSmoothed
      ? { preconditionerZeroSmoothed: compareScalarField(state.preconditionerZeroSmoothed, state.grid) } : {}),
    ...(state.preconditionerFirstOperatorImage
      ? { preconditionerFirstOperatorImage: compareScalarField(state.preconditionerFirstOperatorImage, state.grid) } : {}),
    ...(state.preconditionerFirstSmoothed
      ? { preconditionerFirstSmoothed: compareScalarField(state.preconditionerFirstSmoothed, state.grid) } : {}),
    ...(state.preconditionerInnerResidual
      ? { preconditionerInnerResidual: compareScalarField(state.preconditionerInnerResidual, state.grid) } : {}),
    ...(state.preconditionerInnerCorrection
      ? { preconditionerInnerCorrection: compareScalarField(state.preconditionerInnerCorrection, state.grid) } : {}),
    ...(state.preconditionerPostCorrected
      ? { preconditionerPostCorrected: compareScalarField(state.preconditionerPostCorrected, state.grid) } : {}),
    topology: compareScalarField(state.topology, state.grid),
    frontCircularity: measureHorizontalFrontCircularity(state.volume, state.grid),
    walls: {
      negativeX: wallState(negativeX, state.wallLiquidThreshold),
      positiveX: wallState(positiveX, state.wallLiquidThreshold),
      negativeZ: wallState(negativeZ, state.wallLiquidThreshold),
      positiveZ: wallState(positiveZ, state.wallLiquidThreshold),
    },
  };
}

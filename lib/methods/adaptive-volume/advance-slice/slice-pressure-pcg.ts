/** Production Sparse CM12 Chronopoulos-Gear pressure recurrence, rank-reduced to CPU f32. */

export const SLICE_PRESSURE_TRUE_RESIDUAL_CADENCE = 8;

export interface SlicePressurePCGInput {
  readonly diagonal: Float32Array;
  readonly rhs: Float32Array;
  readonly pressure: Float32Array;
  readonly member: Uint8Array;
  /** PEI compact-cell traversal order. Defaults to ascending member id. */
  readonly executionOrder?: Uint32Array;
  readonly maximumIterations: number;
  readonly relativeTolerance: number;
  /** Writes A*input. The caller owns the exact composite G^T W G operator. */
  readonly apply: (input: Float32Array, output: Float32Array) => void;
}

export interface SlicePressurePCGReceipt {
  readonly pressure: Float32Array;
  readonly residual: Float32Array;
  readonly iterations: number;
  readonly encodedIterations: number;
  readonly initialTrueResidualSquared: number;
  readonly finalTrueResidualSquared: number;
  readonly finalTrueResidualMaximum: number;
  readonly rhsSquared: number;
  readonly recursiveResidualSquared: number;
  readonly firstToleranceIteration: number | null;
  readonly curvatureRecoveries: number;
  readonly residualDrift: boolean;
  readonly converged: boolean;
  readonly records: readonly SlicePressurePCGIteration[];
}

export interface SlicePressurePCGIteration {
  readonly encodedIteration: number;
  readonly gateOpen: boolean;
  readonly gamma: number;
  readonly alpha: number;
  readonly beta: number;
  readonly recursiveResidualSquared: number;
  readonly guardedTrueResidualSquared: number;
  readonly executedIterations: number;
  readonly curvatureBreakdown: boolean;
  readonly curvatureRecoveries: number;
  readonly firstToleranceIteration: number | null;
}

const f = Math.fround;
const add = (a: number, b: number): number => f(f(a) + f(b));
const mul = (a: number, b: number): number => f(f(a) * f(b));
const div = (a: number, b: number): number => f(f(a) / f(b));

/** Mirrors reducePair + the single-workgroup reduction over its partials. */
function reduceProduction(values: Float32Array, active: readonly number[]): number {
  const groups = Math.ceil(active.length / 64);
  const partials = new Float32Array(groups);
  for (let group = 0; group < groups; group++) {
    const lanes = new Float32Array(64);
    for (let lane = 0; lane < 64; lane++) {
      const ordinal = 64 * group + lane;
      if (ordinal < active.length) lanes[lane] = values[active[ordinal]!]!;
    }
    for (let width = 32; width >= 1; width >>= 1) {
      for (let lane = 0; lane < width; lane++) lanes[lane] = add(lanes[lane]!, lanes[lane + width]!);
    }
    partials[group] = lanes[0]!;
  }
  const lanes = new Float32Array(64);
  for (let lane = 0; lane < 64; lane++) {
    let sum = 0;
    for (let at = lane; at < groups; at += 64) sum = add(sum, partials[at]!);
    lanes[lane] = sum;
  }
  for (let width = 32; width >= 1; width >>= 1) {
    for (let lane = 0; lane < width; lane++) lanes[lane] = add(lanes[lane]!, lanes[lane + width]!);
  }
  return lanes[0]!;
}

function reduceDot(a: Float32Array, b: Float32Array, active: readonly number[]): number {
  const products = new Float32Array(a.length);
  for (const id of active) products[id] = mul(a[id]!, b[id]!);
  return reduceProduction(products, active);
}

function measureTrueResidual(input: SlicePressurePCGInput, pressure: Float32Array,
  residual: Float32Array, active: readonly number[]): { squared: number; maximum: number } {
  const image = new Float32Array(pressure.length);
  input.apply(pressure, image);
  const squares = new Float32Array(pressure.length);
  let maximum = 0;
  for (const id of active) {
    const value = f(input.rhs[id]! - image[id]!);
    residual[id] = value;
    squares[id] = mul(value, value);
    maximum = Math.max(maximum, Math.abs(value));
  }
  return { squared: reduceProduction(squares, active), maximum: f(maximum) };
}

/**
 * Execute the same fixed-budget, device-gated recurrence as the resident WGSL.
 * Array traversal is the compact pressure-member order and reductions retain
 * the resident's 64-lane workgroup tree rather than using a JavaScript sum.
 */
export function solveSlicePressurePCG(input: SlicePressurePCGInput): SlicePressurePCGReceipt {
  const count = input.rhs.length;
  if (input.diagonal.length !== count || input.pressure.length !== count
    || input.member.length !== count) throw new RangeError("pressure PCG plane lengths differ");
  if (!Number.isSafeInteger(input.maximumIterations) || input.maximumIterations < 0) {
    throw new RangeError("pressure PCG maximumIterations must be a nonnegative integer");
  }
  if (!(Number.isFinite(input.relativeTolerance) && input.relativeTolerance >= 0)) {
    throw new RangeError("pressure PCG relativeTolerance must be finite and nonnegative");
  }
  const active = input.executionOrder
    ? Array.from(input.executionOrder)
    : Array.from({ length: count }, (_, id) => id).filter(id => input.member[id] !== 0);
  if (new Set(active).size !== active.length
    || active.some(id => id >= count || input.member[id] === 0)) {
    throw new RangeError("pressure PCG executionOrder must contain unique pressure members");
  }
  const pressure = Float32Array.from(input.pressure);
  const residual = new Float32Array(count), z = new Float32Array(count);
  const direction = new Float32Array(count), imageDirection = new Float32Array(count);
  const imageZ = new Float32Array(count), guardResidual = new Float32Array(count);

  // initializePCG then initializeJacobiDirection.
  const applied = new Float32Array(count);
  input.apply(pressure, applied);
  for (const id of active) {
    residual[id] = f(input.rhs[id]! - applied[id]!);
    z[id] = input.diagonal[id]! > 0 ? div(residual[id]!, input.diagonal[id]!) : 0;
    direction[id] = z[id]!;
  }
  let gamma = reduceDot(residual, z, active);
  const rhsSquared = reduceDot(input.rhs, input.rhs, active);
  let live = true, curvature = false, alpha = 0, beta = 0;
  let recursiveResidualSquared = 0, iterations = 0, firstTolerance: number | null = null;
  let curvatureRecoveries = 0;
  const initial = measureTrueResidual(input, pressure, residual, active);
  // The fresh residual is written back before the pipelined image starts.
  for (const id of active) z[id] = input.diagonal[id]! > 0
    ? div(residual[id]!, input.diagonal[id]!) : 0;
  gamma = reduceDot(residual, z, active);
  const toleranceSquared = mul(mul(input.relativeTolerance, input.relativeTolerance), rhsSquared);
  if (input.relativeTolerance > 0 && initial.squared <= toleranceSquared) {
    live = false; firstTolerance = 0;
  }
  if (live) {
    input.apply(z, imageZ);
    imageDirection.set(imageZ);
    const delta = reduceDot(z, imageZ, active);
    if (delta > 1e-20) alpha = div(gamma, delta);
    else { alpha = 0; curvature = true; curvatureRecoveries++; }
  }

  let lastTrue = initial;
  let guardedTrueResidualSquared = 0;
  const records: SlicePressurePCGIteration[] = [];
  const record = (encodedIteration: number) => records.push(Object.freeze({
    encodedIteration, gateOpen: live && !curvature, gamma, alpha, beta,
    recursiveResidualSquared, guardedTrueResidualSquared,
    executedIterations: iterations, curvatureBreakdown: curvature,
    curvatureRecoveries, firstToleranceIteration: firstTolerance,
  }));
  record(0);
  for (let iteration = 0; iteration < input.maximumIterations; iteration++) {
    if (live && !curvature) {
      // updatePipelinedState: recurrence directions precede x/r/z.
      if (iterations > 0) for (const id of active) {
        direction[id] = add(z[id]!, mul(beta, direction[id]!));
        imageDirection[id] = add(imageZ[id]!, mul(beta, imageDirection[id]!));
      }
      for (const id of active) {
        pressure[id] = add(pressure[id]!, mul(alpha, direction[id]!));
        residual[id] = add(residual[id]!, -mul(alpha, imageDirection[id]!));
        z[id] = input.diagonal[id]! > 0 ? div(residual[id]!, input.diagonal[id]!) : 0;
      }
      // applyPipelinedImage + one packed reduction.
      input.apply(z, imageZ);
      const nextGamma = reduceDot(residual, z, active);
      const delta = reduceDot(imageZ, z, active);
      recursiveResidualSquared = reduceDot(residual, residual, active);
      const previousGamma = gamma, previousAlpha = alpha;
      beta = previousGamma > 1e-20 ? div(nextGamma, previousGamma) : 0;
      const denominator = f(delta - mul(beta, div(nextGamma, Math.max(previousAlpha, 1e-20))));
      gamma = nextGamma;
      iterations++;
      if (denominator > 1e-20) alpha = div(nextGamma, denominator);
      else { alpha = 0; curvature = true; curvatureRecoveries++; }
    }

    if ((iteration + 1) % SLICE_PRESSURE_TRUE_RESIDUAL_CADENCE === 0
      && iteration + 1 < input.maximumIterations) {
      if (live) {
        lastTrue = measureTrueResidual(input, pressure, guardResidual, active);
        guardedTrueResidualSquared = lastTrue.squared;
        if (input.relativeTolerance > 0 && lastTrue.squared <= toleranceSquared) {
          if (firstTolerance === null) firstTolerance = iterations;
          live = false;
        } else if (curvature
          || lastTrue.squared > mul(16, Math.max(recursiveResidualSquared, 1e-30))) {
          if (!curvature) curvatureRecoveries++;
          curvature = true;
        }
      }
      if (live && curvature) {
        // restartPCGAfterCurvatureLoss + Jacobi direction publication.
        residual.set(guardResidual);
        for (const id of active) {
          z[id] = input.diagonal[id]! > 0 ? div(residual[id]!, input.diagonal[id]!) : 0;
          direction[id] = z[id]!;
        }
        gamma = reduceDot(residual, z, active); beta = 0;
        input.apply(z, imageZ); imageDirection.set(imageZ);
        const delta = reduceDot(z, imageZ, active);
        if (delta > 1e-20) { alpha = div(gamma, delta); curvature = false; }
        else live = false;
      }
    }
    record(iteration + 1);
  }

  // restore dispatch gate + unconditional final true b-Ax receipt.
  lastTrue = measureTrueResidual(input, pressure, residual, active);
  if (input.relativeTolerance > 0 && lastTrue.squared <= toleranceSquared
    && firstTolerance === null) firstTolerance = iterations;
  return Object.freeze({ pressure, residual, iterations,
    encodedIterations: input.maximumIterations,
    initialTrueResidualSquared: initial.squared,
    finalTrueResidualSquared: lastTrue.squared,
    finalTrueResidualMaximum: lastTrue.maximum, rhsSquared,
    recursiveResidualSquared, firstToleranceIteration: firstTolerance,
    curvatureRecoveries,
    residualDrift: mul(16, Math.max(0, recursiveResidualSquared)) < lastTrue.squared,
    converged: input.relativeTolerance > 0 && lastTrue.squared <= toleranceSquared,
    records: Object.freeze(records) });
}

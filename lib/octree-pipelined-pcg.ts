/**
 * Float64 algebra oracle for matrix-free preconditioned Conjugate Gradient.
 *
 * This module is test-only mathematical authority. It must not be used as a
 * CPU fallback by the WebGPU pressure path.
 *
 * Every updated search direction is mapped by A and its positive curvature is
 * reduced directly. This mirrors the production f32 solver without relying on
 * a cancellation identity derived from exact A-conjugacy.
 */

export type OctreePipelinedPCGLinearMap =
  (input: Readonly<Float64Array>) => ArrayLike<number>;

export interface OctreePipelinedPCGProblem {
  readonly rightHandSide: ArrayLike<number>;
  readonly applyOperator: OctreePipelinedPCGLinearMap;
  /** Fixed linear, symmetric-positive preconditioner M. */
  readonly applyPreconditioner?: OctreePipelinedPCGLinearMap;
  readonly initialGuess?: ArrayLike<number>;
}

export interface OctreePipelinedPCGOptions {
  readonly maximumIterations?: number;
  readonly relativeTolerance?: number;
  readonly absoluteTolerance?: number;
  /**
   * Audit A and M with deterministic probes before solving. This is a bounded
   * oracle check, not a proof for arbitrary callbacks.
   */
  readonly validateLinearMaps?: boolean;
  readonly validationTolerance?: number;
}

export type OctreePipelinedPCGFailure =
  | "operator-is-not-linear"
  | "operator-is-not-symmetric"
  | "operator-is-not-positive"
  | "preconditioner-is-not-linear"
  | "preconditioner-is-not-symmetric"
  | "preconditioner-is-not-positive"
  | "non-finite-map-output"
  | "non-positive-preconditioned-residual"
  | "non-positive-direction-curvature"
  | "non-finite-recurrence"
  | "maximum-iterations"
  | "true-residual-validation";

export interface OctreePipelinedPCGIteration {
  /** Zero is the initialized residual; positive values are completed updates. */
  readonly iteration: number;
  readonly reduction: number;
  readonly residualNorm: number;
  readonly relativeResidualNorm: number;
  readonly gamma: number;
  readonly delta: number;
  /** Step length selected after this reduction, when another update is needed. */
  readonly alpha?: number;
  /** Direction recurrence selected after this reduction. */
  readonly beta?: number;
}

export interface OctreePipelinedPCGDiagnostics {
  readonly converged: boolean;
  readonly iterations: number;
  /**
   * Conceptual global scalar reductions. Initialization uses one reduction;
   * each nonterminal update uses a residual reduction and a direct-curvature
   * reduction, while the terminal converged update needs only the residual.
   */
  readonly reductions: number;
  readonly initialResidualNorm: number;
  /** Norm of the recursively updated residual at termination. */
  readonly recursiveResidualNorm: number;
  /** Norm of b-Ax recomputed independently at termination. */
  readonly residualNorm: number;
  readonly relativeResidualNorm: number;
  readonly stoppingThreshold: number;
  readonly history: readonly OctreePipelinedPCGIteration[];
  readonly failure?: OctreePipelinedPCGFailure;
}

export interface OctreePipelinedPCGResult {
  /**
   * True only after both the recurrence and an independent b-Ax residual check
   * satisfy the requested tolerance.
   */
  readonly accepted: boolean;
  readonly solution: Float64Array;
  /** Independently recomputed b-Ax for the returned candidate solution. */
  readonly residual: Float64Array;
  readonly diagnostics: OctreePipelinedPCGDiagnostics;
}

interface ReductionScalars {
  readonly gamma: number;
  readonly delta: number;
  readonly residualSquared: number;
}

interface MapAudit {
  readonly accepted: boolean;
  readonly failure?: OctreePipelinedPCGFailure;
}

const DEFAULT_RELATIVE_TOLERANCE = 1e-10;
const DEFAULT_VALIDATION_TOLERANCE = 2e-12;

function finiteNonNegative(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${label} must be finite and non-negative`);
  }
  return value;
}

function copyFiniteVector(input: ArrayLike<number>, expectedLength: number, label: string): Float64Array {
  if (!Number.isSafeInteger(input.length) || input.length !== expectedLength) {
    throw new RangeError(`${label} must have length ${expectedLength}`);
  }
  const result = new Float64Array(expectedLength);
  for (let index = 0; index < expectedLength; index += 1) {
    const value = input[index];
    if (!Number.isFinite(value)) throw new RangeError(`${label} contains a non-finite value at ${index}`);
    result[index] = value;
  }
  return result;
}

function applyMap(
  map: OctreePipelinedPCGLinearMap,
  input: Float64Array,
): Float64Array | undefined {
  // A private copy prevents a callback from corrupting recurrence state.
  const mapped = map(new Float64Array(input));
  if (!mapped || !Number.isSafeInteger(mapped.length) || mapped.length !== input.length) return undefined;
  const result = new Float64Array(input.length);
  for (let index = 0; index < result.length; index += 1) {
    const value = mapped[index];
    if (!Number.isFinite(value)) return undefined;
    result[index] = value;
  }
  return result;
}

/** Neumaier accumulation keeps the float64 oracle insensitive to row order. */
function dot(left: Float64Array, right: Float64Array): number {
  let sum = 0, compensation = 0;
  for (let index = 0; index < left.length; index += 1) {
    const product = left[index] * right[index];
    const next = sum + product;
    compensation += Math.abs(sum) >= Math.abs(product)
      ? (sum - next) + product
      : (product - next) + sum;
    sum = next;
  }
  return sum + compensation;
}

function norm(vector: Float64Array): number {
  return Math.sqrt(Math.max(0, dot(vector, vector)));
}

function close(left: number, right: number, tolerance: number): boolean {
  return Math.abs(left - right) <= tolerance * Math.max(1, Math.abs(left), Math.abs(right));
}

function vectorsClose(left: Float64Array, right: Float64Array, tolerance: number): boolean {
  for (let index = 0; index < left.length; index += 1) {
    if (!close(left[index], right[index], tolerance)) return false;
  }
  return true;
}

function deterministicProbes(length: number): readonly [Float64Array, Float64Array] {
  const first = new Float64Array(length), second = new Float64Array(length);
  for (let index = 0; index < length; index += 1) {
    // Integer-valued, non-collinear probes avoid dependence on random state.
    first[index] = ((index * 17 + 5) % 23) - 11;
    second[index] = ((index * 29 + 3) % 19) - 9;
  }
  return [first, second];
}

function auditLinearSPDMap(
  map: OctreePipelinedPCGLinearMap,
  length: number,
  tolerance: number,
  role: "operator" | "preconditioner",
): MapAudit {
  const [first, second] = deterministicProbes(length);
  const combined = new Float64Array(length);
  for (let index = 0; index < length; index += 1) {
    combined[index] = first[index] - 0.375 * second[index];
  }
  const firstImage = applyMap(map, first);
  const repeatedFirstImage = applyMap(map, first);
  const secondImage = applyMap(map, second);
  const combinedImage = applyMap(map, combined);
  const zeroImage = applyMap(map, new Float64Array(length));
  if (!firstImage || !repeatedFirstImage || !secondImage || !combinedImage || !zeroImage) {
    return { accepted: false, failure: "non-finite-map-output" };
  }
  const expectedCombined = new Float64Array(length);
  for (let index = 0; index < length; index += 1) {
    expectedCombined[index] = firstImage[index] - 0.375 * secondImage[index];
  }
  const zero = new Float64Array(length);
  if (!vectorsClose(firstImage, repeatedFirstImage, tolerance)
    || !vectorsClose(combinedImage, expectedCombined, tolerance)
    || !vectorsClose(zeroImage, zero, tolerance)) {
    return {
      accepted: false,
      failure: role === "operator" ? "operator-is-not-linear" : "preconditioner-is-not-linear",
    };
  }
  const firstSecond = dot(first, secondImage), secondFirst = dot(second, firstImage);
  if (!close(firstSecond, secondFirst, tolerance)) {
    return {
      accepted: false,
      failure: role === "operator" ? "operator-is-not-symmetric" : "preconditioner-is-not-symmetric",
    };
  }
  if (!(dot(first, firstImage) > 0) || !(dot(second, secondImage) > 0)) {
    return {
      accepted: false,
      failure: role === "operator" ? "operator-is-not-positive" : "preconditioner-is-not-positive",
    };
  }
  return { accepted: true };
}

function mergeReduction(
  residual: Float64Array,
  preconditioned: Float64Array,
  image?: Float64Array,
): ReductionScalars {
  // This one loop is the CPU transcription of one combined GPU reduction.
  let gamma = 0, gammaCompensation = 0;
  let delta = 0, deltaCompensation = 0;
  let residualSquared = 0, residualCompensation = 0;
  const accumulate = (sum: number, compensation: number, value: number): readonly [number, number] => {
    const next = sum + value;
    return [next, compensation + (Math.abs(sum) >= Math.abs(value)
      ? (sum - next) + value
      : (value - next) + sum)];
  };
  for (let index = 0; index < residual.length; index += 1) {
    [gamma, gammaCompensation] = accumulate(
      gamma, gammaCompensation, residual[index] * preconditioned[index],
    );
    if (image) {
      [delta, deltaCompensation] = accumulate(
        delta, deltaCompensation, preconditioned[index] * image[index],
      );
    }
    [residualSquared, residualCompensation] = accumulate(
      residualSquared, residualCompensation, residual[index] * residual[index],
    );
  }
  return {
    gamma: gamma + gammaCompensation,
    delta: delta + deltaCompensation,
    residualSquared: residualSquared + residualCompensation,
  };
}

function rejectedResult(
  solution: Float64Array,
  residual: Float64Array,
  history: readonly OctreePipelinedPCGIteration[],
  iterations: number,
  reductions: number,
  initialResidualNorm: number,
  recursiveResidualNorm: number,
  stoppingThreshold: number,
  rightHandSideNorm: number,
  failure: OctreePipelinedPCGFailure,
): OctreePipelinedPCGResult {
  const residualNorm = norm(residual);
  return {
    accepted: false,
    solution,
    residual,
    diagnostics: {
      converged: false,
      iterations,
      reductions,
      initialResidualNorm,
      recursiveResidualNorm,
      residualNorm,
      relativeResidualNorm: residualNorm / Math.max(rightHandSideNorm, Number.MIN_VALUE),
      stoppingThreshold,
      history: Object.freeze([...history]),
      failure,
    },
  };
}

/**
 * Solve an SPD system with the matrix-free direct-curvature PCG recurrence.
 *
 * A rejected result always has `accepted === false`; this oracle never changes
 * algorithms, retries without M, or provides an unpreconditioned fallback.
 */
export function solveOctreePipelinedPCG(
  problem: OctreePipelinedPCGProblem,
  options: OctreePipelinedPCGOptions = {},
): OctreePipelinedPCGResult {
  const length = problem.rightHandSide.length;
  if (!Number.isSafeInteger(length) || length < 1) {
    throw new RangeError("Pipelined PCG requires at least one row");
  }
  const rightHandSide = copyFiniteVector(problem.rightHandSide, length, "Pipelined PCG right-hand side");
  const solution = problem.initialGuess
    ? copyFiniteVector(problem.initialGuess, length, "Pipelined PCG initial guess")
    : new Float64Array(length);
  const maximumIterations = options.maximumIterations ?? length;
  if (!Number.isSafeInteger(maximumIterations) || maximumIterations < 1) {
    throw new RangeError("Pipelined PCG maximumIterations must be a positive integer");
  }
  const relativeTolerance = finiteNonNegative(
    options.relativeTolerance ?? DEFAULT_RELATIVE_TOLERANCE,
    "Pipelined PCG relativeTolerance",
  );
  const absoluteTolerance = finiteNonNegative(
    options.absoluteTolerance ?? 0,
    "Pipelined PCG absoluteTolerance",
  );
  const validationTolerance = finiteNonNegative(
    options.validationTolerance ?? DEFAULT_VALIDATION_TOLERANCE,
    "Pipelined PCG validationTolerance",
  );
  if (validationTolerance === 0) {
    throw new RangeError("Pipelined PCG validationTolerance must be positive");
  }
  const identity: OctreePipelinedPCGLinearMap = (input) => input;
  const precondition = problem.applyPreconditioner ?? identity;

  if (options.validateLinearMaps ?? true) {
    const operatorAudit = auditLinearSPDMap(
      problem.applyOperator, length, validationTolerance, "operator",
    );
    if (!operatorAudit.accepted) {
      const residual = new Float64Array(rightHandSide);
      return rejectedResult(
        solution, residual, [], 0, 0, norm(residual), norm(residual), 0,
        norm(rightHandSide), operatorAudit.failure!,
      );
    }
    const preconditionerAudit = auditLinearSPDMap(
      precondition, length, validationTolerance, "preconditioner",
    );
    if (!preconditionerAudit.accepted) {
      const initialProduct = applyMap(problem.applyOperator, solution);
      const residual = new Float64Array(rightHandSide);
      if (initialProduct) for (let index = 0; index < length; index += 1) residual[index] -= initialProduct[index];
      return rejectedResult(
        solution, residual, [], 0, 0, norm(residual), norm(residual), 0,
        norm(rightHandSide), preconditionerAudit.failure!,
      );
    }
  }

  const initialProduct = applyMap(problem.applyOperator, solution);
  if (!initialProduct) {
    const residual = new Float64Array(rightHandSide);
    return rejectedResult(
      solution, residual, [], 0, 0, norm(residual), norm(residual), 0,
      norm(rightHandSide), "non-finite-map-output",
    );
  }
  const residual = new Float64Array(length);
  for (let index = 0; index < length; index += 1) {
    residual[index] = rightHandSide[index] - initialProduct[index];
  }
  const initialResidualNorm = norm(residual);
  const rightHandSideNorm = norm(rightHandSide);
  const stoppingThreshold = Math.max(
    absoluteTolerance,
    relativeTolerance * Math.max(rightHandSideNorm, Number.MIN_VALUE),
  );
  const history: OctreePipelinedPCGIteration[] = [];

  let preconditioned = applyMap(precondition, residual);
  const preconditionedImage = preconditioned
    ? applyMap(problem.applyOperator, preconditioned)
    : undefined;
  if (!preconditioned || !preconditionedImage) {
    return rejectedResult(
      solution, residual, history, 0, 0, initialResidualNorm, initialResidualNorm,
      stoppingThreshold, rightHandSideNorm, "non-finite-map-output",
    );
  }
  let reduction = mergeReduction(residual, preconditioned, preconditionedImage);
  let reductions = 1, iterations = 0;
  let recursiveResidualNorm = Math.sqrt(Math.max(0, reduction.residualSquared));

  if (recursiveResidualNorm <= stoppingThreshold) {
    const entry = {
      iteration: 0, reduction: reductions, residualNorm: recursiveResidualNorm,
      relativeResidualNorm: recursiveResidualNorm / Math.max(rightHandSideNorm, Number.MIN_VALUE),
      gamma: reduction.gamma, delta: reduction.delta,
    };
    history.push(entry);
    return {
      accepted: true,
      solution,
      residual,
      diagnostics: {
        converged: true, iterations, reductions, initialResidualNorm,
        recursiveResidualNorm, residualNorm: recursiveResidualNorm,
        relativeResidualNorm: entry.relativeResidualNorm, stoppingThreshold,
        history: Object.freeze(history),
      },
    };
  }
  if (!(reduction.gamma > 0) || !Number.isFinite(reduction.gamma)) {
    return rejectedResult(
      solution, residual, history, iterations, reductions, initialResidualNorm,
      recursiveResidualNorm, stoppingThreshold, rightHandSideNorm,
      "non-positive-preconditioned-residual",
    );
  }
  if (!(reduction.delta > 0) || !Number.isFinite(reduction.delta)) {
    return rejectedResult(
      solution, residual, history, iterations, reductions, initialResidualNorm,
      recursiveResidualNorm, stoppingThreshold, rightHandSideNorm,
      "non-positive-direction-curvature",
    );
  }

  let alpha = reduction.gamma / reduction.delta;
  if (!Number.isFinite(alpha) || !(alpha > 0)) {
    return rejectedResult(
      solution, residual, history, iterations, reductions, initialResidualNorm,
      recursiveResidualNorm, stoppingThreshold, rightHandSideNorm,
      "non-finite-recurrence",
    );
  }
  history.push({
    iteration: 0, reduction: reductions, residualNorm: recursiveResidualNorm,
    relativeResidualNorm: recursiveResidualNorm / Math.max(rightHandSideNorm, Number.MIN_VALUE),
    gamma: reduction.gamma, delta: reduction.delta, alpha,
  });
  const direction = new Float64Array(preconditioned);
  const directionImage = new Float64Array(preconditionedImage);

  while (iterations < maximumIterations) {
    for (let index = 0; index < length; index += 1) {
      solution[index] += alpha * direction[index];
      residual[index] -= alpha * directionImage[index];
    }
    iterations += 1;

    preconditioned = applyMap(precondition, residual);
    if (!preconditioned) {
      const verifiedResidual = recomputeResidual(
        problem.applyOperator, solution, rightHandSide,
      ) ?? new Float64Array(residual);
      return rejectedResult(
        solution, verifiedResidual, history, iterations, reductions,
        initialResidualNorm, norm(residual), stoppingThreshold, rightHandSideNorm,
        "non-finite-map-output",
      );
    }
    const next = mergeReduction(residual, preconditioned);
    reductions += 1;
    recursiveResidualNorm = Math.sqrt(Math.max(0, next.residualSquared));
    const relativeRecursive = recursiveResidualNorm
      / Math.max(rightHandSideNorm, Number.MIN_VALUE);

    if (recursiveResidualNorm <= stoppingThreshold) {
      history.push({
        iteration: iterations, reduction: reductions,
        residualNorm: recursiveResidualNorm,
        relativeResidualNorm: relativeRecursive,
        gamma: next.gamma, delta: next.delta,
      });
      const verifiedResidual = recomputeResidual(
        problem.applyOperator, solution, rightHandSide,
      );
      if (!verifiedResidual) {
        return rejectedResult(
          solution, new Float64Array(residual), history, iterations, reductions,
          initialResidualNorm, recursiveResidualNorm, stoppingThreshold,
          rightHandSideNorm, "non-finite-map-output",
        );
      }
      const verifiedNorm = norm(verifiedResidual);
      if (verifiedNorm > stoppingThreshold) {
        return rejectedResult(
          solution, verifiedResidual, history, iterations, reductions,
          initialResidualNorm, recursiveResidualNorm, stoppingThreshold,
          rightHandSideNorm, "true-residual-validation",
        );
      }
      return {
        accepted: true,
        solution,
        residual: verifiedResidual,
        diagnostics: {
          converged: true, iterations, reductions, initialResidualNorm,
          recursiveResidualNorm, residualNorm: verifiedNorm,
          relativeResidualNorm: verifiedNorm / Math.max(rightHandSideNorm, Number.MIN_VALUE),
          stoppingThreshold, history: Object.freeze(history),
        },
      };
    }
    if (!(next.gamma > 0) || !Number.isFinite(next.gamma)) {
      const verifiedResidual = recomputeResidual(
        problem.applyOperator, solution, rightHandSide,
      ) ?? new Float64Array(residual);
      return rejectedResult(
        solution, verifiedResidual, history, iterations, reductions,
        initialResidualNorm, recursiveResidualNorm, stoppingThreshold,
        rightHandSideNorm, "non-positive-preconditioned-residual",
      );
    }
    if (iterations >= maximumIterations) {
      const verifiedResidual = recomputeResidual(
        problem.applyOperator, solution, rightHandSide,
      ) ?? new Float64Array(residual);
      return rejectedResult(
        solution, verifiedResidual, history, iterations, reductions,
        initialResidualNorm, recursiveResidualNorm, stoppingThreshold,
        rightHandSideNorm, "maximum-iterations",
      );
    }
    const beta = next.gamma / reduction.gamma;
    if (!Number.isFinite(beta) || beta < 0) {
      const verifiedResidual = recomputeResidual(
        problem.applyOperator, solution, rightHandSide,
      ) ?? new Float64Array(residual);
      return rejectedResult(
        solution, verifiedResidual, history, iterations, reductions,
        initialResidualNorm, recursiveResidualNorm, stoppingThreshold,
        rightHandSideNorm, "non-finite-recurrence",
      );
    }
    for (let index = 0; index < length; index += 1) {
      direction[index] = preconditioned[index] + beta * direction[index];
    }
    const directImage = applyMap(problem.applyOperator, direction);
    if (!directImage) {
      const verifiedResidual = recomputeResidual(
        problem.applyOperator, solution, rightHandSide,
      ) ?? new Float64Array(residual);
      return rejectedResult(
        solution, verifiedResidual, history, iterations, reductions,
        initialResidualNorm, recursiveResidualNorm, stoppingThreshold,
        rightHandSideNorm, "non-finite-map-output",
      );
    }
    directionImage.set(directImage);
    const denominator = dot(direction, directionImage);
    reductions += 1;
    if (!(denominator > 0) || !Number.isFinite(denominator)) {
      const verifiedResidual = recomputeResidual(
        problem.applyOperator, solution, rightHandSide,
      ) ?? new Float64Array(residual);
      return rejectedResult(
        solution, verifiedResidual, history, iterations, reductions,
        initialResidualNorm, recursiveResidualNorm, stoppingThreshold,
        rightHandSideNorm, "non-positive-direction-curvature",
      );
    }
    alpha = next.gamma / denominator;
    if (!Number.isFinite(alpha) || !(alpha > 0)) {
      const verifiedResidual = recomputeResidual(
        problem.applyOperator, solution, rightHandSide,
      ) ?? new Float64Array(residual);
      return rejectedResult(
        solution, verifiedResidual, history, iterations, reductions,
        initialResidualNorm, recursiveResidualNorm, stoppingThreshold,
        rightHandSideNorm, "non-finite-recurrence",
      );
    }
    history.push({
      iteration: iterations, reduction: reductions,
      residualNorm: recursiveResidualNorm, relativeResidualNorm: relativeRecursive,
      gamma: next.gamma, delta: denominator, alpha, beta,
    });
    reduction = next;
  }

  const verifiedResidual = recomputeResidual(
    problem.applyOperator, solution, rightHandSide,
  ) ?? new Float64Array(residual);
  return rejectedResult(
    solution, verifiedResidual, history, iterations, reductions,
    initialResidualNorm, recursiveResidualNorm, stoppingThreshold,
    rightHandSideNorm, "maximum-iterations",
  );
}

function recomputeResidual(
  operator: OctreePipelinedPCGLinearMap,
  solution: Float64Array,
  rightHandSide: Float64Array,
): Float64Array | undefined {
  const product = applyMap(operator, solution);
  if (!product) return undefined;
  const residual = new Float64Array(rightHandSide.length);
  for (let index = 0; index < residual.length; index += 1) {
    residual[index] = rightHandSide[index] - product[index];
  }
  return residual;
}

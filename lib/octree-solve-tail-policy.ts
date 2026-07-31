/**
 * Immutable host-encoding policy for the production octree pressure tail.
 *
 * The policy selects command-graph size from authored scene facts only. GPU
 * residuals still own convergence and zero the indirect tail; no observation
 * from a previous step can change the next step's physics or command graph.
 */

export const OCTREE_SOLVE_TAIL_MINIMUM_OUTER_ITERATIONS = 4;
export const OCTREE_SOLVE_TAIL_MAXIMUM_ENCODED_OUTER_ITERATIONS = 10;
/** Validation/diagnostic ceiling. It is deliberately not host-encoded. */
export const OCTREE_SOLVE_TAIL_HARD_OUTER_ITERATION_CEILING = 16;
export const OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL_ENVIRONMENT =
  "FLUID_OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL";
/**
 * Two iterations of headroom covered every adjacent-step change in the
 * measured factor-one mini-dam trace (the steady solve uses four, while the
 * impact transient reaches six). The selector below still refuses stale,
 * topology-changing, failed, or non-adjacent evidence.
 */
export const OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL_SAFETY_MARGIN = 2;
export const OCTREE_SOLVE_TAIL_RELATIVE_TOLERANCE = 1e-4;
/** Section 4.3 reports k≈8 as a general choice. */
export const OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH = 8;
/** The parity topology's 16-cubed dam sweep selects the established k=4 shell. */
export const OCTREE_SECTION43_MINI_SHELL_DEPTH = 4;
/**
 * Validated finest-cell envelope for the small two-level k=4 formulation.
 *
 * This is deliberately capacity-shaped rather than an axis-length test: the
 * 24x16x24 ceiling drop is still a small 9,216-cell system. The envelope is
 * independent of which MGPCG executor is selected: ceiling-drop now uses the
 * row-parallel solver while retaining the separately validated k=4 shell. The
 * next larger authored two-level validation profile (32x24x16) remains on k=8.
 * Finest-cell count is an immutable conservative bound, so this selection
 * never depends on a readback or a previous advance.
 */
export const OCTREE_SECTION43_MINI_FINEST_CELL_CAPACITY = 24 * 16 * 24;
export const OCTREE_SECTION43_SHELL_DEPTH_ENVIRONMENT =
  "FLUID_OCTREE_SECTION43_SHELL_DEPTH";

export interface OctreeSolveTailSceneProfile {
  readonly finestDimensions: readonly [number, number, number];
  readonly maximumLeafSize: 2 | 4 | 8 | 16 | 32;
  readonly initialCondition: "dam-break" | "tank-fill";
  readonly hasInflow: boolean;
  readonly hasTerrain: boolean;
  readonly movingRigidBodyCount: number;
  readonly closedTop: boolean;
  readonly requestedRelativeTolerance: number;
}

export interface OctreeSolveTailPolicy {
  /** Paper-backed upper envelope; the GPU residual gate zeroes the live tail. */
  readonly encodedOuterIterations: number;
  /** Retained only for validation and diagnostics; no commands are emitted. */
  readonly hardOuterIterationCeiling: 16;
  readonly relativeTolerance: number;
  readonly boundarySmoothingIterations: number;
  readonly sceneComplexityScore: number;
  readonly reasons: readonly string[];
}

export interface OctreeFactorOneSolveTailObservation {
  /** Step whose end-of-step snapshot published this evidence. */
  readonly step: number;
  /** MGPCG control word 2 from that same step. */
  readonly publishedIterationCount: number;
  /** MGPCG control words 0/1 proved a successful converged solve. */
  readonly converged: boolean;
  /** Accepted structured-authority epoch in the same snapshot. */
  readonly acceptedTopologyEpoch: number;
  /** Hash of the candidate that the next step may commit. */
  readonly topologyHash: number;
  /**
   * End-of-step candidate verdict. Combined with `topologyHash`, this
   * distinguishes an identity publication from a real topology change.
   */
  readonly topologyFlipReady: boolean;
  readonly topologyEpochError: number;
}

export interface OctreeFactorOneSolveTailSelection {
  readonly encodedOuterIterations: number;
  readonly usedHistory: boolean;
  readonly reason:
    | "disabled"
    | "not-factor-one"
    | "missing-history"
    | "non-adjacent-history"
    | "invalid-history"
    | "topology-change"
    | "predicted";
}

export function octreeFactorOnePredictedSolveTailEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  // Explicit-on until the required impact-transient A/B has proved
  // `executedIterations < encodedIterations` on every shortened advance.
  return resolved?.[OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL_ENVIRONMENT] === "1";
}

/**
 * Select an encode-only outer schedule from a step-coherent previous result.
 *
 * This never changes either numerical ceiling: `fullEncodedOuterIterations`
 * remains the production envelope and the separate hard ceiling stays 16.
 * The same-step GPU residual remains convergence authority. If prediction
 * headroom were ever exhausted, MGPCG sees this selected encoded envelope,
 * marks the solve fatal, and publishes only the prior seed (fail closed).
 */
export function selectOctreeFactorOneEncodedSolveTail(
  options: {
    readonly enabled: boolean;
    readonly factorOne: boolean;
    readonly nextStep: number;
    readonly fullEncodedOuterIterations: number;
    readonly observation?: OctreeFactorOneSolveTailObservation;
    /** Prior snapshot's candidate hash; that candidate is active this step. */
    readonly precedingTopologyHash?: number;
    readonly safetyMargin?: number;
  },
): OctreeFactorOneSolveTailSelection {
  const full = options.fullEncodedOuterIterations;
  if (!Number.isSafeInteger(full)
    || full < OCTREE_SOLVE_TAIL_MINIMUM_OUTER_ITERATIONS
    || full > OCTREE_SOLVE_TAIL_MAXIMUM_ENCODED_OUTER_ITERATIONS) {
    throw new RangeError("Factor-one solve-tail full envelope is outside the paper bounds");
  }
  if (!Number.isSafeInteger(options.nextStep) || options.nextStep < 0) {
    throw new RangeError("Factor-one solve-tail next step must be a non-negative integer");
  }
  const fallback = (
    reason: Exclude<OctreeFactorOneSolveTailSelection["reason"], "predicted">,
  ): OctreeFactorOneSolveTailSelection => Object.freeze({
    encodedOuterIterations: full, usedHistory: false, reason,
  });
  if (!options.enabled) return fallback("disabled");
  if (!options.factorOne) return fallback("not-factor-one");
  const observation = options.observation;
  if (!observation) return fallback("missing-history");
  if (observation.step + 1 !== options.nextStep) {
    return fallback("non-adjacent-history");
  }
  if (options.precedingTopologyHash === undefined) {
    return fallback("missing-history");
  }
  const validInteger = (value: number) => Number.isSafeInteger(value) && value >= 0;
  if (!observation.converged || !validInteger(observation.publishedIterationCount)
    || observation.publishedIterationCount > full
    || !validInteger(observation.acceptedTopologyEpoch)
    || !validInteger(observation.topologyHash)
    || !validInteger(observation.topologyEpochError)
    || observation.topologyEpochError !== 0) {
    return fallback("invalid-history");
  }
  // At snapshot N-1, its candidate hash is the topology that step N commits.
  // Snapshot N's candidate is the topology step N+1 may commit. Equal hashes
  // prove the selected schedule does not cross a topology change even though
  // the monotonically stamped epoch itself ordinarily advances.
  if (options.precedingTopologyHash !== observation.topologyHash) {
    return fallback("topology-change");
  }
  const margin = options.safetyMargin
    ?? OCTREE_FACTOR1_PREDICTED_SOLVE_TAIL_SAFETY_MARGIN;
  if (!Number.isSafeInteger(margin) || margin < 1) {
    throw new RangeError("Factor-one solve-tail safety margin must be a positive integer");
  }
  return Object.freeze({
    encodedOuterIterations: Math.min(full, Math.max(
      OCTREE_SOLVE_TAIL_MINIMUM_OUTER_ITERATIONS,
      observation.publishedIterationCount + margin,
    )),
    usedHistory: true,
    reason: "predicted" as const,
  });
}

function validateProfile(profile: OctreeSolveTailSceneProfile): void {
  if (profile.finestDimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Octree solve-tail dimensions must be positive safe integers");
  }
  if (![2, 4, 8, 16, 32].includes(profile.maximumLeafSize)) {
    throw new RangeError("Octree solve-tail maximum leaf size must be dyadic in [2,32]");
  }
  if (!Number.isSafeInteger(profile.movingRigidBodyCount)
    || profile.movingRigidBodyCount < 0) {
    throw new RangeError("Octree solve-tail moving-body count must be a non-negative integer");
  }
  if (!Number.isFinite(profile.requestedRelativeTolerance)
    || profile.requestedRelativeTolerance <= 0) {
    throw new RangeError("Octree solve-tail relative tolerance must be finite and positive");
  }
}

/**
 * Select the bounded paper solve envelope from immutable scene geometry.
 * Refinement depth supplies the baseline conditioning allowance. Free-surface
 * motion, persistent inflow, moving immersed boundaries, terrain, and extreme
 * aspect ratios add at most one deterministic unit each.
 */
export function planOctreeSolveTail(
  profile: OctreeSolveTailSceneProfile,
  environment?: Readonly<Record<string, string | undefined>>,
): OctreeSolveTailPolicy {
  validateProfile(profile);
  const resolvedEnvironment = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  const shellDepthText = resolvedEnvironment?.[OCTREE_SECTION43_SHELL_DEPTH_ENVIRONMENT];
  const finestCellCapacity = profile.finestDimensions.reduce(
    (product, dimension) => product * dimension, 1);
  const smallTwoLevelProfile = profile.maximumLeafSize === 2
    && finestCellCapacity <= OCTREE_SECTION43_MINI_FINEST_CELL_CAPACITY;
  const boundarySmoothingIterations = shellDepthText === undefined
    ? (smallTwoLevelProfile
      ? OCTREE_SECTION43_MINI_SHELL_DEPTH
      : OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH)
    : Number(shellDepthText);
  if (!Number.isSafeInteger(boundarySmoothingIterations)
    || boundarySmoothingIterations < 2 || boundarySmoothingIterations > 16
    || (boundarySmoothingIterations & 1) !== 0) {
    throw new RangeError(`${OCTREE_SECTION43_SHELL_DEPTH_ENVIRONMENT} must be an even integer in [2,16]`);
  }
  const dimensions = profile.finestDimensions;
  const aspectRatio = Math.max(...dimensions) / Math.min(...dimensions);
  const refinementDepth = Math.log2(profile.maximumLeafSize);
  const reasons: string[] = [];
  let score = OCTREE_SOLVE_TAIL_MINIMUM_OUTER_ITERATIONS;

  const refinementAllowance = Math.min(2, Math.ceil(refinementDepth / 2));
  score += refinementAllowance;
  reasons.push(`dyadic-depth:${refinementAllowance}`);
  if (profile.initialCondition === "dam-break") {
    score += 1; reasons.push("moving-free-surface");
  }
  if (profile.hasInflow) { score += 1; reasons.push("inflow"); }
  if (profile.hasTerrain) { score += 1; reasons.push("terrain"); }
  if (profile.movingRigidBodyCount > 0) {
    score += 1; reasons.push("moving-immersed-boundary");
  }
  if (profile.closedTop && profile.initialCondition === "dam-break") {
    score += 1; reasons.push("closed-top-transient");
  }
  if (aspectRatio >= 4) { score += 1; reasons.push("aspect-ratio>=4"); }
  if (aspectRatio >= 8) { score += 1; reasons.push("aspect-ratio>=8"); }

  // Section 4.3 reports a 6--10 iteration range, not a scene-metadata formula
  // for predicting a safe numerical cap. Encode that upper envelope once and
  // let the same-step GPU residual gate eliminate every unused iteration.
  // The mini-dam impact is the important counterexample: its authored score is
  // only six, but some transient steps need the remaining tail to preserve the
  // projected pressure distribution and wall-climbing jet.
  const encodedOuterIterations = OCTREE_SOLVE_TAIL_MAXIMUM_ENCODED_OUTER_ITERATIONS;
  return Object.freeze({
    encodedOuterIterations,
    hardOuterIterationCeiling: OCTREE_SOLVE_TAIL_HARD_OUTER_ITERATION_CEILING,
    relativeTolerance: Math.max(
      profile.requestedRelativeTolerance,
      OCTREE_SOLVE_TAIL_RELATIVE_TOLERANCE,
    ),
    boundarySmoothingIterations,
    sceneComplexityScore: score,
    reasons: Object.freeze(reasons),
  });
}

export interface OctreePressureCommandShape {
  readonly encodedOuterIterations: number;
  readonly fullOperatorDispatches: number;
  readonly mergedBandOperatorDispatches: 1;
  readonly firstOrderSetupDispatches: number;
  readonly firstOrderCorrectionDispatches: number;
  readonly boundarySmoothingIterations: number;
}

export interface OctreePressureCommandCount {
  readonly preconditionerSetupDispatches: number;
  readonly preconditionerCorrectionDispatches: number;
  readonly encodedPressureDispatches: number;
}

/** CPU mirror of the production orchestration formulas. */
export function countOctreePressureCommands(
  shape: OctreePressureCommandShape,
): OctreePressureCommandCount {
  const integerFields = [shape.encodedOuterIterations, shape.fullOperatorDispatches,
    shape.mergedBandOperatorDispatches, shape.firstOrderSetupDispatches,
    shape.firstOrderCorrectionDispatches, shape.boundarySmoothingIterations];
  if (integerFields.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Octree pressure command shape must contain positive integers");
  }
  if (shape.encodedOuterIterations < OCTREE_SOLVE_TAIL_MINIMUM_OUTER_ITERATIONS
    || shape.encodedOuterIterations > OCTREE_SOLVE_TAIL_MAXIMUM_ENCODED_OUTER_ITERATIONS) {
    throw new RangeError("Octree encoded outer budget is outside the paper envelope");
  }
  if ((shape.boundarySmoothingIterations & 1) !== 0) {
    throw new RangeError("Section 4.3 shell depth must be even for matching halves");
  }
  // The encoder adds `5 + OCTREE_SECTION43_BOUNDARY_BAND_LAYERS` (= 8), and
  // `encodeSetup` emits exactly that many: resetBandWorksets,
  // prepareCorrectionDispatches, classifyBand, three dilate sweeps,
  // compactBandIntersections, finalizeBandWorksets. This mirror said 7.
  const preconditionerSetupDispatches = shape.firstOrderSetupDispatches + 8;
  const preconditionerCorrectionDispatches = shape.firstOrderCorrectionDispatches
    + 2 * shape.boundarySmoothingIterations + 4
    + shape.fullOperatorDispatches
    + (2 * shape.boundarySmoothingIterations - 1)
      * shape.mergedBandOperatorDispatches;
  const encodedPressureDispatches = 8 + 2 * shape.fullOperatorDispatches
    + preconditionerSetupDispatches + preconditionerCorrectionDispatches
    + shape.encodedOuterIterations
      * (6 + shape.fullOperatorDispatches + preconditionerCorrectionDispatches);
  return Object.freeze({ preconditionerSetupDispatches,
    preconditionerCorrectionDispatches, encodedPressureDispatches });
}

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
export const OCTREE_SOLVE_TAIL_RELATIVE_TOLERANCE = 1e-4;
/** Section 4.3 reports that about eight matching boundary Jacobi sweeps were
 * needed for satisfactory 6--10 iteration convergence. */
export const OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH = 8;

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
  readonly boundarySmoothingIterations: 8;
  readonly sceneComplexityScore: number;
  readonly reasons: readonly string[];
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
): OctreeSolveTailPolicy {
  validateProfile(profile);
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
    boundarySmoothingIterations: OCTREE_SECTION43_PRODUCTION_SHELL_DEPTH,
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

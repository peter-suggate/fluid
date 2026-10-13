/** Pure hierarchy planning, shared by method hosts and scene tools. */
/** Preferred coarse-grid size for hierarchy planning, not a solver limit. */
export const UNIFORM_CM11A_COARSEST_TARGET_CELLS = 256;

/** Physical cells on each axis, halo excluded. */
export type UniformCM11aLevelSize = readonly [number, number, number];

/** What a dense CM11a hierarchy over `dimensions` would look like. */
export interface UniformCM11aHierarchyPlan {
  readonly levelCount: number;
  /** Physical lattice of every level, finest first. */
  readonly levelDimensions: readonly UniformCM11aLevelSize[];
  /** Coarsest lattice cell count, including its one-cell halo. */
  readonly coarsestCells: number;
  /** True when each axis coarsens on its own schedule. */
  readonly semiCoarsened: boolean;
  /** Why the hierarchy is impossible, or `undefined` when it is buildable. */
  readonly rejection?: string;
}

const haloedCells = (size: UniformCM11aLevelSize): number =>
  size.reduce((cells, value) => cells * (value + 2), 1);

/**
 * The original hierarchy: halve all three axes together, stopping when the
 * thinnest reaches two cells. Returns undefined when it cannot be built.
 */
function lockstepLevels(dimensions: UniformCM11aLevelSize, dimension: 2 | 3 = 3): UniformCM11aLevelSize[] | undefined {
  const axes = dimension === 2 ? dimensions.slice(0, 2) : dimensions;
  const minimum = Math.min(...axes);
  if ((minimum & (minimum - 1)) !== 0) return undefined;
  const levelCount = Math.floor(Math.log2(minimum));
  const coarsening = 2 ** (levelCount - 1);
  if (!axes.every((value) => value % coarsening === 0)) return undefined;
  const levels: UniformCM11aLevelSize[] = [];
  for (let index = 0; index < levelCount; index += 1) {
    const step = 2 ** index;
    levels.push([dimensions[0] / step, dimensions[1] / step, dimension === 2 ? 1 : dimensions[2] / step]);
  }
  return haloedCells(levels[levels.length - 1]!) <= UNIFORM_CM11A_COARSEST_TARGET_CELLS ? levels : undefined;
}

/**
 * Semi-coarsening: halve each axis on its own schedule, while it is even and
 * still above two cells. An axis that has bottomed out simply stops, and the
 * others keep going.
 */
function semiCoarsenedLevels(dimensions: UniformCM11aLevelSize): UniformCM11aLevelSize[] {
  const levels: UniformCM11aLevelSize[] = [dimensions];
  for (;;) {
    const previous = levels[levels.length - 1]!;
    const next = previous.map((value) =>
      value % 2 === 0 && value > 2 ? value / 2 : value) as unknown as UniformCM11aLevelSize;
    if (next.every((value, axis) => value === previous[axis])) return levels;
    levels.push(next);
  }
}

/**
 * Whether a lattice can carry the dense hierarchy, without a GPU.
 *
 * This is a *constructor* precondition, not a scene-validity one: a scene can
 * satisfy every rule in `validateScene`, build, render its dry world and still
 * fail to load the instant the solver is created. Exposing it as arithmetic is
 * what lets the catalog be checked on the CPU.
 *
 * Two rules, tried in order, and the order is the whole point. The original
 * hierarchy coarsens all three axes in **lockstep** and stops when the thinnest
 * reaches two cells, so a thin axis caps how far the wide ones may coarsen: a
 * 128x128x8 lattice reaches only 32x32x2 and leaves 4624 coarsest cells for 256
 * lanes. Coarsening each axis on its own schedule reaches 2x2x2 instead --
 * fewer coarsest cells, not more, because the 256 lanes were never the real
 * limit. But semi-coarsening also builds a *different* hierarchy wherever both
 * rules apply, and the D4 folds downstream make that a change in rounding, so
 * the lockstep plan is preferred whenever it exists and semi-coarsening is
 * reached only by lattices that have no hierarchy at all today. Every scene
 * that loads now keeps the hierarchy, and the numbers, that it already has.
 * Odd terminal axes are retained too. The coarse solver uses strided storage
 * for every grid; the target above only preserves the hierarchy preference.
 */
export function planUniformCM11aHierarchy(
  dimensions: UniformCM11aLevelSize,
  dimension: 2 | 3 = 3,
): UniformCM11aHierarchyPlan {
  const reject = (rejection: string): UniformCM11aHierarchyPlan =>
    ({ levelCount: 0, levelDimensions: [], coarsestCells: 0, semiCoarsened: false, rejection });
  if (!dimensions.every((value, axis) => Number.isSafeInteger(value) && (dimension === 2 && axis === 2 ? value === 1 : value >= 2))) {
    return reject("CM11a dense hierarchy requires positive integral dimensions of at least two cells");
  }
  const lockstep = lockstepLevels(dimensions, dimension);
  const levels = lockstep ?? semiCoarsenedLevels(dimensions);
  const coarsestCells = haloedCells(levels[levels.length - 1]!);
  const plan = {
    levelCount: levels.length, levelDimensions: Object.freeze(levels),
    coarsestCells, semiCoarsened: lockstep === undefined,
  };
  return plan;
}

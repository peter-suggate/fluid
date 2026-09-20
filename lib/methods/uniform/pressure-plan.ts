/** Pure hierarchy and window planning, shared by method hosts and scene tools. */
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

/** Alignments a window capacity and origin may use, widest first. */
export const UNIFORM_CM11A_WINDOW_ALIGNMENTS = [32, 16] as const;

export interface UniformCM11aWindowPlan {
  readonly capacity: UniformCM11aLevelSize;
  readonly origin: readonly [number, number, number];
  readonly hierarchy: UniformCM11aHierarchyPlan;
  /** Alignment used per axis; a domain-wide axis reports its own length. */
  readonly alignment: readonly [number, number, number];
}

/**
 * Where the current capacity can sit so that it covers `[low, high)`.
 *
 * Returns the aligned origin, or undefined when the capacity is simply too
 * small. The caller uses this to keep an instance across a step in which the
 * liquid moved but did not grow -- which is most steps.
 */
export function seatUniformCM11aWindow(
  domain: UniformCM11aLevelSize,
  capacity: readonly [number, number, number],
  alignment: readonly [number, number, number],
  low: readonly [number, number, number],
  high: readonly [number, number, number],
): [number, number, number] | undefined {
  const origin: [number, number, number] = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const step = alignment[axis]!, size = domain[axis]!, width = capacity[axis]!;
    if (width > size) return undefined;
    const seat = Math.max(0, Math.min(Math.floor(Math.max(0, low[axis]!) / step) * step, size - width));
    if (seat > low[axis]! || seat + width < high[axis]!) return undefined;
    origin[axis] = seat;
  }
  return origin;
}

/**
 * Capacity and origin of a CM11a lattice covering `[low, high)`.
 *
 * Aligning both keeps every coarse grid registered to the domain, so the
 * origin only has to move when the liquid crosses an alignment boundary, and
 * the same capacity is reached again and again instead of drifting by a cell
 * a step. The lockstep hierarchy is preferred over semi-coarsening for the
 * same reason the domain planner prefers it, so a capacity one alignment step
 * wider, or a shortest axis lifted to a power of two, is tried before giving
 * up on it.
 */
export function planUniformCM11aWindow(
  domain: UniformCM11aLevelSize,
  low: readonly [number, number, number],
  high: readonly [number, number, number],
): UniformCM11aWindowPlan {
  const alignment = domain.map((size) =>
    UNIFORM_CM11A_WINDOW_ALIGNMENTS.find((step) => size % step === 0 && size > step) ?? size,
  ) as unknown as [number, number, number];
  const seat = (request: readonly number[]): { capacity: UniformCM11aLevelSize;
    origin: [number, number, number] } => {
    const capacity: number[] = []; const origin: number[] = [];
    for (let axis = 0; axis < 3; axis += 1) {
      const step = alignment[axis]!, size = domain[axis]!;
      const width = Math.min(size, Math.max(step, Math.ceil(request[axis]! / step) * step));
      capacity.push(width);
      origin.push(Math.max(0, Math.min(
        Math.floor(Math.max(0, low[axis]!) / step) * step, size - width)));
    }
    return { capacity: capacity as unknown as UniformCM11aLevelSize,
      origin: origin as [number, number, number] };
  };
  const requested = [0, 1, 2].map((axis) => {
    const step = alignment[axis]!;
    const start = Math.floor(Math.max(0, low[axis]!) / step) * step;
    return Math.max(2, Math.min(domain[axis]!, Math.max(high[axis]! - start, 0)));
  });
  const candidates: number[][] = [requested];
  for (let axis = 0; axis < 3; axis += 1) {
    const bumped = [...requested];
    bumped[axis] = Math.min(domain[axis]!, bumped[axis]! + alignment[axis]!);
    candidates.push(bumped);
  }
  const lifted = [...requested];
  const shortest = lifted.indexOf(Math.min(...lifted));
  lifted[shortest] = Math.min(domain[shortest]!,
    2 ** Math.ceil(Math.log2(Math.max(2, lifted[shortest]!))));
  candidates.push(lifted, [...domain]);
  let fallback: UniformCM11aWindowPlan | undefined;
  let largerFallback: UniformCM11aWindowPlan | undefined;
  for (const candidate of candidates) {
    const { capacity, origin } = seat(candidate);
    const hierarchy = planUniformCM11aHierarchy(capacity);
    if (hierarchy.rejection) continue;
    const plan: UniformCM11aWindowPlan = { capacity, origin, hierarchy, alignment };
    if (!hierarchy.semiCoarsened) return plan;
    if (hierarchy.coarsestCells <= UNIFORM_CM11A_COARSEST_TARGET_CELLS) fallback ??= plan;
    else largerFallback ??= plan;
  }
  return fallback ?? largerFallback ?? { capacity: domain, origin: [0, 0, 0],
    hierarchy: planUniformCM11aHierarchy(domain), alignment };
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

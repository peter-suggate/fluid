import {
  SPARSE_BRICK_INVALID_INDEX,
  SPARSE_BRICK_NUMBER_MORTON_BITS,
  mortonEncode3D,
  mortonNumber3D,
  mortonNumberCoordinate,
  type SparseBrickCoordinate,
  type SparseBrickLeafPlan,
  type SparseBrickNodePlan,
  type SparseBrickPlan,
  type SparseBrickSize,
  type SparseBrickLeafTerminal,
  SPARSE_BRICK_LEAF_TERMINAL,
  SPARSE_BRICK_VOXEL_TERMINAL,
} from "../svo/features/construction/sparse-brick-octree";
import { completeCooperativeBuild } from "./cooperative-build";
import { createContentMemo } from "./content-memo";

export interface AdaptiveSparseBrickPlanOptions {
  /** Batch geometric decisions on the device while the host assembles records. */
  classifyEnvironmentBatch?: (level: number, coordinates: readonly SparseBrickCoordinate[]) => Promise<readonly SparseBrickEnvironmentClassification[]>;
  /** GPU-produced occupancy, unioned with the explicit proxy claim. */
  proxyOccupancy?: SparseBrickProxyOccupancy;
  brickSize: SparseBrickSize;
  /** Coordinates of simulation bricks, expressed at {@link solverLevel}. */
  solverBricks: readonly SparseBrickCoordinate[];
  /** Bricks touched by raster/analytic scene proxy geometry, expressed at maximumDepth. */
  proxyBricks: readonly SparseBrickCoordinate[];
  /** Depth of the tree. Environment leaves may reach it; solver leaves do not. */
  maximumDepth: number;
  /** A power of two: 0 disables environment coarsening, 3 permits 8x coarser bricks. */
  maximumEnvironmentCoarseningPower: number;
  /**
   * The level the solver's own bricks occupy. Defaults to `maximumDepth`.
   *
   * Separating the two is what lets environment geometry be resolved *finer*
   * than the simulation without moving the simulation's lattice. Until this
   * existed the tree could only coarsen: environment leaves started at
   * `maximumDepth - coarseningPower` and descended solely where a solver brick
   * was in the way, so there was no way to spend resolution on detail — the
   * only knob was `finestCellSize_m`, and it applied to the whole domain.
   */
  solverLevel?: number;
  /**
   * Whether an environment leaf at this level should be split further.
   *
   * Consulted only below {@link solverLevel}, and only where no solver brick
   * already provides coverage. The intended criterion is candidate density: the
   * voxeliser bins primitives per *leaf* against a fixed budget and drops the
   * surplus silently, so a leaf holding more than the budget is one that stops
   * casting shadows and stops occluding indirect light without saying so.
   * Splitting it is the direct fix, and it is local — the leaves around it keep
   * their size.
   */
  refineEnvironmentLeaf?: (level: number, coordinate: SparseBrickCoordinate) => boolean;
  /**
   * Content the set's own claim does not reach (a backdrop's stored detail
   * rings), planned after the set by a descent from the root: a node under a
   * set leaf is skipped, a node above set leaves splits, and anywhere else this
   * answers empty (0), leaf (1) or split (2). The set's plan is unchanged;
   * these leaves take the voxel terminal.
   */
  classifySupplementalNode?: (level: number, coordinate: SparseBrickCoordinate) => 0 | 1 | 2;
  /**
   * A name for everything {@link classifySupplementalNode} answers from: two
   * classifiers under one name give one verdict at every node. With it, a plan
   * over the same set, solver bricks and levels takes its supplemental leaves
   * from the last descent that found them, and the classifier is not asked.
   * Left out, the descent always runs.
   */
  supplementalContentKey?: string;
  /** Representation selected after topology has admitted an environment leaf. */
  classifyEnvironmentLeaf?: (
    level: number,
    coordinate: SparseBrickCoordinate,
  ) => SparseBrickLeafTerminal;
}

/** A plan, and which of its leaves the supplemental descent placed. */
export interface AdaptiveSparseBrickPlan extends SparseBrickPlan {
  /** One byte a leaf, by leaf index: 1 where the leaf is a supplemental one. */
  readonly supplementalLeaf: Uint8Array;
}

/** A supplemental descent kept: what it ran against, and the leaves it found. */
interface SupplementalDescent {
  readonly setLeaves: readonly Float64Array[];
  readonly solverKeys: Float64Array;
  readonly leaves: readonly Float64Array[];
}
/**
 * Descents by the classifier's name and the tree's levels. A descent reads the
 * set's leaves, the solver's bricks and the classifier and nothing else, so one
 * that meets the same three again has the same leaves to give.
 */
const supplementalDescents = createContentMemo<SupplementalDescent>(2);

function sameKeys(a: Float64Array, b: Float64Array): boolean {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) if (a[index] !== b[index]) return false;
  return true;
}

export interface SparseBrickProxyOccupancy {
  keys(level: number): Iterable<bigint>;
  has(level: number, coordinate: SparseBrickCoordinate): boolean;
}
export interface SparseBrickEnvironmentClassification {
  refine: boolean;
  terminal: SparseBrickLeafTerminal;
}

export interface AdaptiveSparseBrickReductionReport {
  fineLeafCount: number;
  plannedLeafCount: number;
  savedLeafCount: number;
  fineVoxelCount: number;
  plannedVoxelCount: number;
  savedVoxelCount: number;
  reductionFraction: number;
  compressionRatio: number;
  solverLeafCount: number;
  environmentLeafCount: number;
  coarsenedEnvironmentLeafCount: number;
  maximumCoarseningPowerUsed: number;
}

function assertDepth(value: number): number {
  if (!Number.isInteger(value) || value < 0 || value > SPARSE_BRICK_NUMBER_MORTON_BITS) {
    throw new RangeError(`Maximum depth must be 0..${SPARSE_BRICK_NUMBER_MORTON_BITS}`);
  }
  return value;
}

function assertCoarseningPower(value: number, solverLevel: number): number {
  if (!Number.isInteger(value) || value < 0 || value > solverLevel) {
    throw new RangeError("Maximum environment coarsening power must be an integer from 0 to the solver level");
  }
  return value;
}

function assertSolverLevel(value: number, maximumDepth: number): number {
  if (!Number.isInteger(value) || value < 0 || value > maximumDepth) {
    throw new RangeError("Solver level must be an integer from 0 to maximumDepth");
  }
  return value;
}

/**
 * How many items of a flat pass run between yield offers.
 *
 * The planner's loops are millions of iterations of a few hundred nanoseconds,
 * so offering a yield on every one would be dominated by the offer. A batch of
 * this size is a few hundred microseconds of work — two orders of magnitude
 * below the driver's slice, so the slice budget rather than the batch is what
 * decides responsiveness, while the per-item cost of the offer disappears.
 */
const PLAN_YIELD_BATCH = 4096;

/**
 * The same, for the adaptive descent, whose per-item cost is not comparable.
 *
 * A flat pass moves a Morton key; a descent evaluates `refineEnvironmentLeaf`,
 * which on a refined garden asks an exact distance function or a ground-column
 * pyramid and is measured in tens of microseconds. Sized so a batch stays
 * inside the driver's slice even at that cost.
 */
const PLAN_DESCENT_YIELD_BATCH = 64;

/**
 * The bricks' Morton keys at `depth`, ascending and unique.
 *
 * Every table the planner builds is one of these: a level's keys in a
 * `Float64Array` (`mortonNumber3D`), so a set is a sorted run, membership is a
 * binary search, a node's subtree is a contiguous range of the finest keys,
 * and a level's parents are one pass over the level below.
 */
function* sortedMortonKeysSteps(
  bricks: readonly SparseBrickCoordinate[],
  depth: number,
  label: string,
): Generator<unknown, Float64Array, undefined> {
  const keys = new Float64Array(bricks.length);
  for (let first = 0; first < bricks.length; first += PLAN_YIELD_BATCH) {
    mortonKeysOf(bricks, keys, first, Math.min(bricks.length, first + PLAN_YIELD_BATCH), 2 ** depth, label);
    yield;
  }
  // The sort itself is not interruptible; it is one native pass over numbers.
  return uniqueAscending(keys.sort());
}

/**
 * One slice of the keys.
 *
 * A plain function, as every loop of the planner is: a generator's own body
 * is never optimised, so a loop written in one runs several times slower and
 * boxes every number it forms. The generators here only hand out slices.
 */
function mortonKeysOf(
  bricks: readonly SparseBrickCoordinate[],
  keys: Float64Array,
  first: number,
  end: number,
  limit: number,
  label: string,
): void {
  for (let index = first; index < end; index += 1) {
    const { x, y, z } = bricks[index]!;
    if (!Number.isSafeInteger(x) || x < 0 || x >= limit) throw new RangeError(`${label} x must be an integer in [0, ${limit})`);
    if (!Number.isSafeInteger(y) || y < 0 || y >= limit) throw new RangeError(`${label} y must be an integer in [0, ${limit})`);
    if (!Number.isSafeInteger(z) || z < 0 || z >= limit) throw new RangeError(`${label} z must be an integer in [0, ${limit})`);
    keys[index] = mortonNumber3D(x, y, z);
  }
}

/** An ascending run with its repeats dropped. */
function uniqueAscending(keys: Float64Array): Float64Array {
  let count = 0;
  for (let index = 0; index < keys.length; index += 1) {
    if (count === 0 || keys[index] !== keys[count - 1]) keys[count++] = keys[index]!;
  }
  return keys.subarray(0, count);
}

/** First index in `[low, high)` whose key is at least `key`. */
function lowerBound(keys: Float64Array, key: number, low: number, high: number): number {
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (keys[middle]! < key) low = middle + 1; else high = middle;
  }
  return low;
}

function hasKey(keys: Float64Array, key: number): boolean {
  const at = lowerBound(keys, key, 0, keys.length);
  return at < keys.length && keys[at] === key;
}

/** Two ascending, disjoint runs as one. */
function mergeAscending(a: Float64Array, b: Float64Array): Float64Array {
  if (a.length === 0) return b;
  if (b.length === 0) return a;
  const merged = new Float64Array(a.length + b.length);
  let i = 0, j = 0, count = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) throw new RangeError("Adaptive sparse brick plan holds one leaf twice");
    merged[count++] = a[i]! < b[j]! ? a[i++]! : b[j++]!;
  }
  while (i < a.length) merged[count++] = a[i++]!;
  while (j < b.length) merged[count++] = b[j++]!;
  return merged;
}

/** A descent's leaves at one level, which a Morton-ordered walk emits ascending. */
function ascendingLeaves(keys: readonly number[]): Float64Array {
  const leaves = Float64Array.from(keys);
  for (let index = 1; index < leaves.length; index += 1) {
    if (!(leaves[index - 1]! < leaves[index]!)) throw new RangeError("Adaptive sparse brick plan leaves left Morton order");
  }
  return leaves;
}

/**
 * Every node of the tree whose leaves these are, per level, ascending: a
 * level's nodes are its own leaves and the parents of the level below.
 */
function* nodeKeysSteps(leaves: readonly Float64Array[]): Generator<unknown, Float64Array[], undefined> {
  const nodes: Float64Array[] = new Array(leaves.length);
  let below: Float64Array = new Float64Array(0);
  for (let level = leaves.length - 1; level >= 0; level -= 1) {
    nodes[level] = below = levelNodeKeys(leaves[level]!, below);
    yield;
  }
  return nodes;
}

/** A level's nodes: its own leaves and the parents of the level below. */
function levelNodeKeys(own: Float64Array, below: Float64Array): Float64Array {
  // Children of one parent are adjacent, so the parents come out ascending.
  const parents = new Float64Array(below.length);
  let count = 0;
  for (let index = 0; index < below.length; index += 1) {
    const parent = Math.floor(below[index]! / 8);
    if (count === 0 || parents[count - 1] !== parent) parents[count++] = parent;
  }
  const ancestors = parents.subarray(0, count);
  const merged = new Float64Array(own.length + ancestors.length);
  let i = 0, j = 0, size = 0;
  while (i < own.length || j < ancestors.length) {
    const a = i < own.length ? own[i]! : Number.POSITIVE_INFINITY;
    const b = j < ancestors.length ? ancestors[j]! : Number.POSITIVE_INFINITY;
    if (a === b) throw new RangeError("Adaptive sparse brick plan holds a leaf above another leaf");
    merged[size++] = a < b ? own[i++]! : ancestors[j++]!;
  }
  return merged;
}

/** Where each root's run of the finest keys starts, and after the last, where the keys end. */
function rootRunStarts(keys: Float64Array, roots: Float64Array, scale: number): Int32Array {
  const starts = new Int32Array(roots.length + 1);
  for (let index = 0; index < roots.length; index += 1) {
    starts[index] = lowerBound(keys, roots[index]! * scale, index === 0 ? 0 : starts[index - 1]!, keys.length);
  }
  starts[roots.length] = keys.length;
  return starts;
}

/** The ascending, unique prefixes `floor(key / scale)` of an ascending run. */
function prefixesOf(keys: Float64Array, scale: number): Float64Array {
  const prefixes = new Float64Array(keys.length);
  let count = 0;
  for (let index = 0; index < keys.length; index += 1) {
    const prefix = Math.floor(keys[index]! / scale);
    if (count === 0 || prefixes[count - 1] !== prefix) prefixes[count++] = prefix;
  }
  return prefixes.subarray(0, count);
}

function popcount8(value: number): number {
  let count = 0;
  for (let word = value & 0xff; word !== 0; word >>>= 1) count += word & 1;
  return count;
}

/**
 * Plan one pointerless octree containing fine solver leaves and adaptively
 * coarsened environment leaves. A proxy leaf is accepted only when its entire
 * extent is free of solver bricks; otherwise proxy coverage descends locally.
 *
 * The one-shot form, for callers that are already synchronous. See
 * {@link planAdaptiveSparseBrickOctreeSteps} for the interruptible one; both
 * run the same body, so there is no second definition of the plan.
 */
export function planAdaptiveSparseBrickOctree(options: AdaptiveSparseBrickPlanOptions): AdaptiveSparseBrickPlan {
  return completeCooperativeBuild(planAdaptiveSparseBrickOctreeSteps(options));
}

/**
 * The same plan, offered as slices.
 *
 * At environment refinement depth 3 on `hero-garden-hose` this is the single
 * longest uninterrupted block in a scene build, and the reason the render
 * worker cannot service a `draw` — or the abort of the very request that
 * superseded it — while a refined scene is being planned. Every `yield` here is
 * at a point where the planner owns nothing but its own local state: no GPU
 * resource exists yet at any of them, so an abandoned plan is collected rather
 * than released.
 */
export function* planAdaptiveSparseBrickOctreeSteps(
  options: AdaptiveSparseBrickPlanOptions,
): Generator<unknown, AdaptiveSparseBrickPlan, undefined> {
  if (options.brickSize !== 4 && options.brickSize !== 8) throw new RangeError("Sparse brick size must be 4 or 8");
  const maximumDepth = assertDepth(options.maximumDepth);
  const solverLevel = assertSolverLevel(options.solverLevel ?? maximumDepth, maximumDepth);
  const coarseningPower = assertCoarseningPower(options.maximumEnvironmentCoarseningPower, solverLevel);
  // Solver bricks are addressed in their own level's domain, which is coarser
  // than the tree's whenever the environment is being resolved more finely.
  const solverKeys = yield* sortedMortonKeysSteps(options.solverBricks, solverLevel, "Solver brick");
  const proxyKeys = yield* sortedMortonKeysSteps(options.proxyBricks, maximumDepth, "Proxy brick");
  const minimumEnvironmentLevel = solverLevel - coarseningPower;
  const refine = options.refineEnvironmentLeaf;
  const occupancy = options.proxyOccupancy;

  /**
   * Whether a solver brick already owns this node — as an ancestor of one above
   * `solverLevel`, or as the brick containing it below.
   *
   * Below the solver level there are no solver *nodes*, so containment is the
   * only question that makes sense: the solver's leaf is the ancestor, and it
   * provides the coverage its whole subtree would have.
   */
  const solverCovers = (level: number, key: number): boolean => {
    if (solverKeys.length === 0) return false;
    if (level > solverLevel) return hasKey(solverKeys, Math.floor(key / 8 ** (level - solverLevel)));
    const scale = 8 ** (solverLevel - level);
    const at = lowerBound(solverKeys, key * scale, 0, solverKeys.length);
    return at < solverKeys.length && solverKeys[at]! < (key + 1) * scale;
  };

  const perLevel = <T>(make: () => T): T[] => Array.from({ length: maximumDepth + 1 }, make);
  // The set's environment leaves, per level, in the order the descent emits
  // them (ascending), and beside each the terminal a batch classified it with.
  const proxyLeaves = perLevel<number[]>(() => []);
  const proxyTerminals = perLevel<SparseBrickLeafTerminal[]>(() => []);

  // A node's proxy bricks are the run `[low, high)` of the finest keys under
  // it, so the children that hold any are found by splitting that run, and a
  // child holding none is asked of the device's occupancy alone.
  const childLow = new Int32Array(9);
  const occupiedChildren = (level: number, key: number, low: number, high: number): number => {
    const scale = 8 ** (maximumDepth - level - 1);
    let mask = 0;
    childLow[0] = low;
    for (let octant = 0; octant < 8; octant += 1) {
      const end = octant === 7 ? high : lowerBound(proxyKeys, (key * 8 + octant + 1) * scale, childLow[octant]!, high);
      childLow[octant + 1] = end;
      if (end > childLow[octant]! || occupancy?.has(level + 1, mortonNumberCoordinate(key * 8 + octant))) mask |= 1 << octant;
    }
    return mask;
  };

  // The roots: every node of the coarsest environment level that holds a proxy
  // brick, with the device's occupancy at that level.
  const rootScale = 8 ** (maximumDepth - minimumEnvironmentLevel);
  let roots: Float64Array = prefixesOf(proxyKeys, rootScale);
  if (occupancy) {
    const occupied: number[] = [];
    for (const key of occupancy.keys(minimumEnvironmentLevel)) occupied.push(Number(key));
    const union = new Float64Array(roots.length + occupied.length);
    union.set(roots); union.set(occupied, roots.length);
    roots = uniqueAscending(union.sort());
  }
  const rootLow = rootRunStarts(proxyKeys, roots, rootScale);
  const rootHigh = (index: number): number => lowerBound(proxyKeys, (roots[index]! + 1) * rootScale, rootLow[index]!, rootLow[index + 1]!);

  if (options.classifyEnvironmentBatch) {
    let frontier = Array.from(roots), frontierLow = Array.from(rootLow.subarray(0, roots.length));
    let frontierHigh = frontier.map((_, index) => rootHigh(index));
    let next: number[] = [], nextLow: number[] = [], nextHigh: number[] = [];
    const settle = (level: number, first: number, end: number, coordinates: readonly SparseBrickCoordinate[],
      classification: readonly SparseBrickEnvironmentClassification[]): void => {
      for (let index = first; index < end; index += 1) {
        const key = frontier[index]!, covered = solverCovers(level, key);
        if (covered && level >= solverLevel) continue;
        if (level < maximumDepth && (covered || classification[index]!.refine || refine?.(level, coordinates[index]!))) {
          const mask = occupiedChildren(level, key, frontierLow[index]!, frontierHigh[index]!);
          for (let octant = 0; octant < 8; octant += 1) {
            if ((mask & (1 << octant)) === 0) continue;
            next.push(key * 8 + octant); nextLow.push(childLow[octant]!); nextHigh.push(childLow[octant + 1]!);
          }
        } else {
          proxyLeaves[level]!.push(key); proxyTerminals[level]!.push(classification[index]!.terminal);
        }
      }
    };
    for (let level = minimumEnvironmentLevel; level <= maximumDepth && frontier.length; level += 1) {
      const coordinates = frontier.map(mortonNumberCoordinate);
      let classification: readonly SparseBrickEnvironmentClassification[] = [];
      yield options.classifyEnvironmentBatch(level, coordinates).then(result => { classification = result; });
      if (classification.length !== frontier.length) throw new Error("GPU environment classification count mismatch");
      for (let first = 0; first < frontier.length; first += PLAN_YIELD_BATCH) {
        settle(level, first, Math.min(frontier.length, first + PLAN_YIELD_BATCH), coordinates, classification);
        yield;
      }
      frontier = next; frontierLow = nextLow; frontierHigh = nextHigh;
      next = []; nextLow = []; nextHigh = [];
    }
  } else {
    // The descent, on an explicit stack so a yield offer reaches the driver
    // from any depth of it. Children are pushed last octant first, so nodes
    // are visited — and a level's leaves emitted — in Morton order.
    const stackLevel: number[] = [], stackKey: number[] = [], stackLow: number[] = [], stackHigh: number[] = [];
    let root = 0;
    /** Visit up to `budget` nodes; false once every root's subtree is planned. */
    const descend = (budget: number): boolean => {
      for (;;) {
        if (stackLevel.length === 0) {
          if (root === roots.length) return false;
          stackLevel.push(minimumEnvironmentLevel); stackKey.push(roots[root]!);
          stackLow.push(rootLow[root]!); stackHigh.push(rootHigh(root));
          root += 1;
        }
        while (stackLevel.length > 0) {
          if (budget === 0) return true;
          budget -= 1;
          const level = stackLevel.pop()!, key = stackKey.pop()!, low = stackLow.pop()!, high = stackHigh.pop()!;
          if (solverCovers(level, key)) {
            // The coincident solver leaf provides coverage; nothing below it is ours.
            if (level >= solverLevel) continue;
          } else if (!(level < maximumDepth && refine?.(level, mortonNumberCoordinate(key)))) {
            proxyLeaves[level]!.push(key);
            continue;
          }
          const mask = occupiedChildren(level, key, low, high);
          for (let octant = 7; octant >= 0; octant -= 1) {
            if ((mask & (1 << octant)) === 0) continue;
            stackLevel.push(level + 1); stackKey.push(key * 8 + octant);
            stackLow.push(childLow[octant]!); stackHigh.push(childLow[octant + 1]!);
          }
        }
      }
    };
    while (descend(PLAN_DESCENT_YIELD_BATCH)) yield;
  }
  const setLeaves = proxyLeaves.map(ascendingLeaves);
  const environmentLeaves = [...setLeaves];
  setLeaves[solverLevel] = mergeAscending(setLeaves[solverLevel]!, solverKeys);

  let leafKeys = setLeaves;
  let supplemental: readonly Float64Array[] = perLevel(() => new Float64Array(0));
  const descentKey = options.supplementalContentKey === undefined ? undefined
    : `${options.supplementalContentKey}|${maximumDepth}|${solverLevel}`;
  const heldDescent = descentKey === undefined || !options.classifySupplementalNode ? undefined
    : supplementalDescents.get(descentKey);
  if (heldDescent && sameKeys(heldDescent.solverKeys, solverKeys)
    && heldDescent.setLeaves.every((leaves, level) => sameKeys(leaves, setLeaves[level]!))) {
    supplemental = heldDescent.leaves;
    leafKeys = setLeaves.map((leaves, level) => mergeAscending(leaves, supplemental[level]!));
  } else if (options.classifySupplementalNode) {
    const classify = options.classifySupplementalNode;
    const supplementalLeaves = perLevel<number[]>(() => []);
    // The set's own tree: a supplemental leaf at one of its internal nodes
    // would overlap the set, so those split instead.
    const setNodes = yield* nodeKeysSteps(setLeaves);
    // Top-down, so a node is only reached through ancestors that are not
    // leaves: a set leaf *at* the node is the one coverage left to test. The
    // set's tree holds every ancestor of each of its nodes, so nothing under a
    // node outside it is in it, or under a solver brick: those nodes (`clear`)
    // go straight to the classifier.
    const stackLevel: number[] = [0], stackKey: number[] = [0], stackClear: boolean[] = [false];
    /** Visit up to `budget` nodes; false once the stack is empty. */
    const descend = (budget: number): boolean => {
      while (stackLevel.length > 0) {
        if (budget === 0) return true;
        budget -= 1;
        const level = stackLevel.pop()!, key = stackKey.pop()!;
        let clear = stackClear.pop()!;
        let verdict: 0 | 1 | 2;
        if (!clear && hasKey(setNodes[level]!, key)) {
          if (hasKey(setLeaves[level]!, key)) continue;
          verdict = 2;
        } else {
          const covered = !clear && solverCovers(level, key);
          if (covered && level >= solverLevel) continue;
          clear = true;
          verdict = covered ? 2 : classify(level, mortonNumberCoordinate(key));
        }
        if (verdict === 1 || (verdict === 2 && level === maximumDepth)) {
          supplementalLeaves[level]!.push(key);
        } else if (verdict === 2) {
          for (let octant = 7; octant >= 0; octant -= 1) {
            stackLevel.push(level + 1); stackKey.push(key * 8 + octant); stackClear.push(clear);
          }
        }
      }
      return false;
    };
    while (descend(PLAN_YIELD_BATCH)) yield;
    supplemental = supplementalLeaves.map(ascendingLeaves);
    leafKeys = setLeaves.map((leaves, level) => mergeAscending(leaves, supplemental[level]!));
    if (descentKey !== undefined) supplementalDescents.set(descentKey, { setLeaves, solverKeys, leaves: supplemental });
  }

  const nodeKeys = yield* nodeKeysSteps(leafKeys);
  const levelOffsets: number[] = [];
  let nodeCount = 0;
  for (let level = 0; level <= maximumDepth; level += 1) {
    levelOffsets.push(nodeCount);
    nodeCount += nodeKeys[level]!.length;
  }
  levelOffsets.push(nodeCount);

  // The child-range invariant, established here and asserted here.
  //
  // Traversal descends by `firstChild + popcountBefore(mask, octant)` and never
  // reads a child record to check it landed where it meant to. That is only
  // sound because this pass guarantees three things at once: the node array is
  // level-major (`levelOffsets`), a level-L node's children are taken
  // exclusively from level L+1, and one parent's children occupy a contiguous
  // run because a level is Morton-sorted and the eight child keys of a parent
  // are a contiguous Morton range. So `[firstChild, firstChild + childCount)`
  // lies inside level L+1 and every key in it has this node as its parent. One
  // cursor walks level L+1 as level L is emitted; a child the cursor reaches
  // out of turn — an orphan, or a level out of order — stops it, and the level
  // is refused unless the cursor consumed all of it.
  //
  // The GPU topology mutator (`webgpu-sparse-brick-topology-mutation.ts`) keeps
  // the same invariant by its own construction — `insertChild` writes level+1
  // into every node it initializes and relocates a parent's children into one
  // fresh contiguous run — and is not covered by this host-side check.
  const voxelsPerBrick = options.brickSize ** 3;
  const nodes: SparseBrickNodePlan[] = new Array(nodeCount);
  const leaves: SparseBrickLeafPlan[] = [];
  const supplementalLeaf = new Uint8Array(leafKeys.reduce((count, keys) => count + keys.length, 0));
  // The level's cursors, between slices: the next child, solver brick, set leaf
  // and supplemental leaf not yet taken.
  const cursors = new Int32Array(4);
  const emit = (level: number, first: number, end: number): void => {
    const keys = nodeKeys[level]!;
    const children = level < maximumDepth ? nodeKeys[level + 1]! : undefined;
    const environment = environmentLeaves[level]!, terminals = proxyTerminals[level]!, extra = supplemental[level]!;
    let child = cursors[0]!, solverAt = cursors[1]!, environmentAt = cursors[2]!, extraAt = cursors[3]!;
    for (let local = first; local < end; local += 1) {
      const key = keys[local]!, index = levelOffsets[level]! + local;
      let childMask = 0;
      const firstChild = child;
      if (children) {
        while (child < children.length && Math.floor(children[child]! / 8) === key) {
          childMask |= 1 << (children[child]! - key * 8);
          child += 1;
        }
      }
      const node: SparseBrickNodePlan = {
        index,
        level,
        morton: BigInt(key),
        coordinate: mortonNumberCoordinate(key),
        childMask,
        firstChild: childMask === 0 ? SPARSE_BRICK_INVALID_INDEX : levelOffsets[level + 1]! + firstChild,
        childCount: child - firstChild,
        leafIndex: SPARSE_BRICK_INVALID_INDEX,
      };
      nodes[index] = node;
      if (node.childCount !== popcount8(childMask)) {
        throw new RangeError("Adaptive sparse brick plan child range is not a contiguous run of this node's next-level children");
      }
      // Which producer's leaf this node is, if any: a solver brick, a set leaf
      // (with the terminal a batch gave it) or a supplemental one.
      let batchTerminal: SparseBrickLeafTerminal | undefined;
      if (level === solverLevel && solverAt < solverKeys.length && solverKeys[solverAt] === key) {
        solverAt += 1;
      } else if (environmentAt < environment.length && environment[environmentAt] === key) {
        batchTerminal = terminals[environmentAt];
        environmentAt += 1;
      } else if (extraAt < extra.length && extra[extraAt] === key) {
        batchTerminal = SPARSE_BRICK_VOXEL_TERMINAL;
        extraAt += 1;
        supplementalLeaf[leaves.length] = 1;
      } else {
        continue;
      }
      const leafIndex: number = leaves.length;
      const terminal = !solverCovers(level, key)
        ? batchTerminal ?? options.classifyEnvironmentLeaf?.(level, node.coordinate) ?? SPARSE_BRICK_VOXEL_TERMINAL
        : SPARSE_BRICK_VOXEL_TERMINAL;
      if (terminal.kind !== SPARSE_BRICK_LEAF_TERMINAL.voxels
        && terminal.kind !== SPARSE_BRICK_LEAF_TERMINAL.planarBoundary) {
        throw new RangeError(`Sparse brick leaf terminal kind ${terminal.kind} is unsupported`);
      }
      if (!Number.isSafeInteger(terminal.index) || terminal.index < 0
        || terminal.index > 0xffff_ffff) {
        throw new RangeError("Sparse brick leaf terminal index must fit uint32");
      }
      if ((terminal.kind === SPARSE_BRICK_LEAF_TERMINAL.voxels
        && terminal.index !== SPARSE_BRICK_INVALID_INDEX)
        || (terminal.kind === SPARSE_BRICK_LEAF_TERMINAL.planarBoundary
          && terminal.index === SPARSE_BRICK_INVALID_INDEX)) {
        throw new RangeError("Sparse brick leaf terminal index does not match its kind");
      }
      node.leafIndex = leafIndex;
      leaves.push({
        index: leafIndex,
        nodeIndex: index,
        morton: node.morton,
        coordinate: node.coordinate,
        voxelOffset: leafIndex * voxelsPerBrick,
        terminalKind: terminal.kind,
        terminalIndex: terminal.index,
      });
    }
    cursors[0] = child; cursors[1] = solverAt; cursors[2] = environmentAt; cursors[3] = extraAt;
  };
  for (let level = 0; level <= maximumDepth; level += 1) {
    const count = nodeKeys[level]!.length;
    cursors.fill(0);
    for (let first = 0; first < count; first += PLAN_YIELD_BATCH) {
      emit(level, first, Math.min(count, first + PLAN_YIELD_BATCH));
      yield;
    }
    if ((level < maximumDepth && cursors[0] !== nodeKeys[level + 1]!.length) || cursors[2] !== environmentLeaves[level]!.length
      || cursors[3] !== supplemental[level]!.length || (level === solverLevel && cursors[1] !== solverKeys.length)) {
      throw new RangeError("Adaptive sparse brick plan child range is not a contiguous run of this node's next-level children");
    }
  }

  return {
    brickSize: options.brickSize,
    maximumDepth,
    levelOffsets,
    nodes,
    leaves,
    voxelCount: leaves.length * voxelsPerBrick,
    supplementalLeaf,
  };
}

/** Return the inclusive-exclusive finest-brick bounds represented by a leaf node. */
export function adaptiveSparseBrickLeafBounds(
  plan: SparseBrickPlan,
  leafIndex: number,
): { minimum: SparseBrickCoordinate; maximum: SparseBrickCoordinate } {
  const leaf = plan.leaves[leafIndex];
  if (!leaf) throw new RangeError("Leaf index is outside the plan");
  const node = plan.nodes[leaf.nodeIndex];
  const scale = 2 ** (plan.maximumDepth - node.level);
  return {
    minimum: { x: node.coordinate.x * scale, y: node.coordinate.y * scale, z: node.coordinate.z * scale },
    maximum: {
      x: (node.coordinate.x + 1) * scale,
      y: (node.coordinate.y + 1) * scale,
      z: (node.coordinate.z + 1) * scale,
    },
  };
}

export function adaptiveSparseBrickLeafContains(
  plan: SparseBrickPlan,
  leafIndex: number,
  coordinate: SparseBrickCoordinate,
): boolean {
  const bounds = adaptiveSparseBrickLeafBounds(plan, leafIndex);
  return coordinate.x >= bounds.minimum.x && coordinate.x < bounds.maximum.x
    && coordinate.y >= bounds.minimum.y && coordinate.y < bounds.maximum.y
    && coordinate.z >= bounds.minimum.z && coordinate.z < bounds.maximum.z;
}

/** Estimate allocation savings against one finest-level brick per unique input candidate. */
export function reportAdaptiveSparseBrickReduction(
  plan: SparseBrickPlan,
  options: Pick<AdaptiveSparseBrickPlanOptions, "solverBricks" | "proxyBricks">,
): AdaptiveSparseBrickReductionReport {
  const fineKeys = new Set<bigint>();
  const solverKeys = new Set<bigint>();
  for (const coordinate of options.solverBricks) {
    const key = mortonEncode3D(coordinate.x, coordinate.y, coordinate.z);
    fineKeys.add(key);
    solverKeys.add(key);
  }
  for (const coordinate of options.proxyBricks) fineKeys.add(mortonEncode3D(coordinate.x, coordinate.y, coordinate.z));
  const fineLeafCount = fineKeys.size;
  const plannedLeafCount = plan.leaves.length;
  const voxelsPerBrick = plan.brickSize ** 3;
  const fineVoxelCount = fineLeafCount * voxelsPerBrick;
  const plannedVoxelCount = plan.voxelCount;
  let coarsenedEnvironmentLeafCount = 0;
  let maximumCoarseningPowerUsed = 0;
  for (const leaf of plan.leaves) {
    const node = plan.nodes[leaf.nodeIndex];
    if (node.level === plan.maximumDepth && solverKeys.has(node.morton)) continue;
    if (node.level < plan.maximumDepth) coarsenedEnvironmentLeafCount += 1;
    maximumCoarseningPowerUsed = Math.max(maximumCoarseningPowerUsed, plan.maximumDepth - node.level);
  }
  return {
    fineLeafCount,
    plannedLeafCount,
    savedLeafCount: fineLeafCount - plannedLeafCount,
    fineVoxelCount,
    plannedVoxelCount,
    savedVoxelCount: fineVoxelCount - plannedVoxelCount,
    reductionFraction: fineVoxelCount === 0 ? 0 : (fineVoxelCount - plannedVoxelCount) / fineVoxelCount,
    compressionRatio: plannedVoxelCount === 0 ? 1 : fineVoxelCount / plannedVoxelCount,
    solverLeafCount: solverKeys.size,
    environmentLeafCount: plannedLeafCount - solverKeys.size,
    coarsenedEnvironmentLeafCount,
    maximumCoarseningPowerUsed,
  };
}

import {
  SPARSE_BRICK_INVALID_INDEX,
  SPARSE_BRICK_MAX_MORTON_BITS,
  mortonChild,
  mortonEncode3D,
  type SparseBrickCoordinate,
  type SparseBrickLeafPlan,
  type SparseBrickNodePlan,
  type SparseBrickPlan,
  type SparseBrickSize,
  type SparseBrickLeafTerminal,
  SPARSE_BRICK_LEAF_TERMINAL,
  SPARSE_BRICK_VOXEL_TERMINAL,
} from "../svo/sparse-brick-octree";
import { completeCooperativeBuild } from "./cooperative-build";

export interface AdaptiveSparseBrickPlanOptions {
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
  /** Representation selected after topology has admitted an environment leaf. */
  classifyEnvironmentLeaf?: (
    level: number,
    coordinate: SparseBrickCoordinate,
  ) => SparseBrickLeafTerminal;
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

interface CanonicalInputs {
  solver: Map<bigint, SparseBrickCoordinate>;
  proxy: Map<bigint, SparseBrickCoordinate>;
}

function assertDepth(value: number): number {
  if (!Number.isInteger(value) || value < 0 || value > SPARSE_BRICK_MAX_MORTON_BITS) {
    throw new RangeError(`Maximum depth must be 0..${SPARSE_BRICK_MAX_MORTON_BITS}`);
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
 * The local coordinate a Morton key denotes at a level.
 *
 * At a fixed level the prefix *is* the Morton encoding of the coordinate, so
 * this is a de-interleave rather than a lookup.
 */
function coordinateForKey(key: bigint, level: number): SparseBrickCoordinate {
  let x = 0;
  let y = 0;
  let z = 0;
  for (let bit = 0; bit < level; bit += 1) {
    x += Number((key >> BigInt(3 * bit)) & 1n) * 2 ** bit;
    y += Number((key >> BigInt(3 * bit + 1)) & 1n) * 2 ** bit;
    z += Number((key >> BigInt(3 * bit + 2)) & 1n) * 2 ** bit;
  }
  return { x, y, z };
}

function canonicalCoordinate(
  coordinate: SparseBrickCoordinate,
  maximumDepth: number,
  label: string,
): SparseBrickCoordinate {
  const limit = 2 ** maximumDepth;
  for (const [axis, value] of Object.entries(coordinate)) {
    if (!Number.isSafeInteger(value) || value < 0 || value >= limit) {
      throw new RangeError(`${label} ${axis} must be an integer in [0, ${limit})`);
    }
  }
  return { x: coordinate.x, y: coordinate.y, z: coordinate.z };
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

function* canonicalizeSteps(
  options: AdaptiveSparseBrickPlanOptions,
  solverLevel: number,
): Generator<unknown, CanonicalInputs, undefined> {
  const solver = new Map<bigint, SparseBrickCoordinate>();
  const proxy = new Map<bigint, SparseBrickCoordinate>();
  let visited = 0;
  for (const input of options.solverBricks) {
    // Solver bricks are addressed in their own level's domain, which is coarser
    // than the tree's whenever the environment is being resolved more finely.
    const coordinate = canonicalCoordinate(input, solverLevel, "Solver brick");
    solver.set(mortonEncode3D(coordinate.x, coordinate.y, coordinate.z), coordinate);
    if ((visited += 1) % PLAN_YIELD_BATCH === 0) yield;
  }
  for (const input of options.proxyBricks) {
    const coordinate = canonicalCoordinate(input, options.maximumDepth, "Proxy brick");
    proxy.set(mortonEncode3D(coordinate.x, coordinate.y, coordinate.z), coordinate);
    if ((visited += 1) % PLAN_YIELD_BATCH === 0) yield;
  }
  return { solver, proxy };
}

function* prefixSetsSteps(
  finestKeys: Iterable<bigint>,
  maximumDepth: number,
): Generator<unknown, Set<bigint>[], undefined> {
  const levels = Array.from({ length: maximumDepth + 1 }, () => new Set<bigint>());
  let visited = 0;
  for (const finestKey of finestKeys) {
    let key = finestKey;
    for (let level = maximumDepth; level >= 0; level -= 1) {
      levels[level].add(key);
      key >>= 3n;
    }
    if ((visited += 1) % PLAN_YIELD_BATCH === 0) yield;
  }
  return levels;
}

function compareMorton(a: bigint, b: bigint): number {
  return a < b ? -1 : a > b ? 1 : 0;
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
export function planAdaptiveSparseBrickOctree(options: AdaptiveSparseBrickPlanOptions): SparseBrickPlan {
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
): Generator<unknown, SparseBrickPlan, undefined> {
  if (options.brickSize !== 4 && options.brickSize !== 8) throw new RangeError("Sparse brick size must be 4 or 8");
  const maximumDepth = assertDepth(options.maximumDepth);
  const solverLevel = assertSolverLevel(options.solverLevel ?? maximumDepth, maximumDepth);
  const coarseningPower = assertCoarseningPower(options.maximumEnvironmentCoarseningPower, solverLevel);
  const inputs = yield* canonicalizeSteps(options, solverLevel);
  const solverPrefixes = yield* prefixSetsSteps(inputs.solver.keys(), solverLevel);
  const proxyPrefixes = yield* prefixSetsSteps(inputs.proxy.keys(), maximumDepth);
  const minimumEnvironmentLevel = solverLevel - coarseningPower;
  const refine = options.refineEnvironmentLeaf;

  /**
   * Whether a solver brick already owns this node — as an ancestor of one above
   * `solverLevel`, or as the brick containing it below.
   *
   * Below the solver level there are no solver *nodes*, so containment is the
   * only question that makes sense: the solver's leaf is the ancestor, and it
   * provides the coverage its whole subtree would have.
   */
  const solverCovers = (level: number, key: bigint): boolean => (level <= solverLevel
    ? solverPrefixes[level].has(key)
    : solverPrefixes[solverLevel].has(key >> BigInt(3 * (level - solverLevel))));

  // `${level}:${morton}` is unambiguous because each level has its own Morton domain.
  const leafKeys = new Set<string>();
  for (const key of inputs.solver.keys()) leafKeys.add(`${solverLevel}:${key}`);

  /**
   * The descent, as a generator so a yield offer reaches the driver from any
   * depth of the recursion.
   *
   * `yield*` costs one generator object per visited node and nothing per
   * *offer*, because a delegation that never yields is never resumed. The
   * alternative — offering only at the roots — makes the interrupt granularity
   * a property of how the scene happens to be laid out, which on a garden whose
   * geometry sits under a handful of coarse nodes is no granularity at all.
   */
  let visited = 0;
  function* addProxyLeaves(level: number, key: bigint): Generator<unknown, void, undefined> {
    function* descend(): Generator<unknown, void, undefined> {
      for (let octant = 0; octant < 8; octant += 1) {
        const child = mortonChild(key, octant);
        if (proxyPrefixes[level + 1].has(child)) yield* addProxyLeaves(level + 1, child);
      }
    }
    if ((visited += 1) % PLAN_DESCENT_YIELD_BATCH === 0) yield;
    if (solverCovers(level, key)) {
      // The coincident solver leaf provides coverage; nothing below it is ours.
      if (level >= solverLevel) return;
      yield* descend();
      return;
    }
    if (level < maximumDepth && refine?.(level, coordinateForKey(key, level))) {
      yield* descend();
      return;
    }
    leafKeys.add(`${level}:${key}`);
  }
  for (const key of [...proxyPrefixes[minimumEnvironmentLevel]].sort(compareMorton)) {
    yield* addProxyLeaves(minimumEnvironmentLevel, key);
  }

  const nodesByLevel = Array.from({ length: maximumDepth + 1 }, () => new Map<bigint, SparseBrickCoordinate>());
  visited = 0;
  for (const leafKey of leafKeys) {
    const separator = leafKey.indexOf(":");
    const leafLevel = Number(leafKey.slice(0, separator));
    let key = BigInt(leafKey.slice(separator + 1));
    for (let level = leafLevel; level >= 0; level -= 1) {
      if (!nodesByLevel[level].has(key)) nodesByLevel[level].set(key, coordinateForKey(key, level));
      key >>= 3n;
    }
    if ((visited += 1) % PLAN_YIELD_BATCH === 0) yield;
  }

  const levelOffsets: number[] = [];
  const nodes: SparseBrickNodePlan[] = [];
  const nodeIndex = new Map<string, number>();
  visited = 0;
  for (let level = 0; level <= maximumDepth; level += 1) {
    levelOffsets.push(nodes.length);
    // The sort itself is not interruptible; the level boundary is the finest
    // grain this pass has, and one level's sort is the residual stall it costs.
    const ordered = [...nodesByLevel[level]].sort(([a], [b]) => compareMorton(a, b));
    yield;
    for (const [morton, coordinate] of ordered) {
      if ((visited += 1) % PLAN_YIELD_BATCH === 0) yield;
      const index = nodes.length;
      nodeIndex.set(`${level}:${morton}`, index);
      nodes.push({
        index,
        level,
        morton,
        coordinate,
        childMask: 0,
        firstChild: SPARSE_BRICK_INVALID_INDEX,
        childCount: 0,
        leafIndex: SPARSE_BRICK_INVALID_INDEX,
      });
    }
  }
  levelOffsets.push(nodes.length);

  visited = 0;
  for (let level = 0; level < maximumDepth; level += 1) {
    for (let index = levelOffsets[level]; index < levelOffsets[level + 1]; index += 1) {
      if ((visited += 1) % PLAN_YIELD_BATCH === 0) yield;
      const node = nodes[index];
      let firstChild = SPARSE_BRICK_INVALID_INDEX;
      let childMask = 0;
      for (let octant = 0; octant < 8; octant += 1) {
        const childIndex = nodeIndex.get(`${level + 1}:${mortonChild(node.morton, octant)}`);
        if (childIndex === undefined) continue;
        firstChild = Math.min(firstChild, childIndex);
        childMask |= 1 << octant;
      }
      node.firstChild = firstChild;
      node.childMask = childMask;
      node.childCount = popcount8(childMask);
    }
  }

  // The child-range invariant, established here and asserted here.
  //
  // Traversal descends by `firstChild + popcountBefore(mask, octant)` and never
  // reads a child record to check it landed where it meant to. That is only
  // sound because this pass guarantees three things at once: the node array is
  // level-major (`levelOffsets`), a level-L node's children are looked up
  // exclusively in the level-(L+1) map, and one parent's children occupy a
  // contiguous run because a level is Morton-sorted and the eight child keys of
  // a parent are a contiguous Morton range. So `[firstChild, firstChild +
  // childCount)` lies inside level L+1 — every node in it therefore has level
  // L+1 — and both ends share the parent's Morton prefix, which pins the run to
  // this parent alone. Checking the two ends is sufficient: the level is sorted,
  // so anything between two keys with the same 3-bit-shifted prefix shares it.
  //
  // The GPU topology mutator (`webgpu-sparse-brick-topology-mutation.ts`) keeps
  // the same invariant by its own construction — `insertChild` writes level+1
  // into every node it initializes and relocates a parent's children into one
  // fresh contiguous run — and is not covered by this host-side check.
  visited = 0;
  for (let level = 0; level < maximumDepth; level += 1) {
    for (let index = levelOffsets[level]; index < levelOffsets[level + 1]; index += 1) {
      if ((visited += 1) % PLAN_YIELD_BATCH === 0) yield;
      const node = nodes[index];
      if (node.childMask === 0) continue;
      const last = node.firstChild + node.childCount - 1;
      if (node.firstChild < levelOffsets[level + 1] || last >= levelOffsets[level + 2]
        || nodes[node.firstChild].morton >> 3n !== node.morton
        || nodes[last].morton >> 3n !== node.morton) {
        throw new RangeError("Adaptive sparse brick plan child range is not a contiguous run of this node's next-level children");
      }
    }
  }

  const voxelsPerBrick = options.brickSize ** 3;
  const leaves: SparseBrickLeafPlan[] = [];
  visited = 0;
  for (const node of nodes) {
    if ((visited += 1) % PLAN_YIELD_BATCH === 0) yield;
    if (!leafKeys.has(`${node.level}:${node.morton}`)) continue;
    const index: number = leaves.length;
    const terminal = !solverCovers(node.level, node.morton)
      ? options.classifyEnvironmentLeaf?.(node.level, node.coordinate)
        ?? SPARSE_BRICK_VOXEL_TERMINAL
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
    node.leafIndex = index;
    leaves.push({
      index,
      nodeIndex: node.index,
      morton: node.morton,
      coordinate: node.coordinate,
      voxelOffset: index * voxelsPerBrick,
      terminalKind: terminal.kind,
      terminalIndex: terminal.index,
    });
  }

  return {
    brickSize: options.brickSize,
    maximumDepth,
    levelOffsets,
    nodes,
    leaves,
    voxelCount: leaves.length * voxelsPerBrick,
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

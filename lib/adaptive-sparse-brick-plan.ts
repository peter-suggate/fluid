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
} from "./sparse-brick-octree";

export interface AdaptiveSparseBrickPlanOptions {
  brickSize: SparseBrickSize;
  /** Coordinates of simulation bricks. These always remain at maximumDepth. */
  solverBricks: readonly SparseBrickCoordinate[];
  /** Finest-level bricks touched by raster/analytic scene proxy geometry. */
  proxyBricks: readonly SparseBrickCoordinate[];
  maximumDepth: number;
  /** A power of two: 0 disables environment coarsening, 3 permits 8x coarser bricks. */
  maximumEnvironmentCoarseningPower: number;
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

function assertCoarseningPower(value: number, maximumDepth: number): number {
  if (!Number.isInteger(value) || value < 0 || value > maximumDepth) {
    throw new RangeError("Maximum environment coarsening power must be an integer from 0 to maximumDepth");
  }
  return value;
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

function canonicalize(options: AdaptiveSparseBrickPlanOptions): CanonicalInputs {
  const solver = new Map<bigint, SparseBrickCoordinate>();
  const proxy = new Map<bigint, SparseBrickCoordinate>();
  for (const input of options.solverBricks) {
    const coordinate = canonicalCoordinate(input, options.maximumDepth, "Solver brick");
    solver.set(mortonEncode3D(coordinate.x, coordinate.y, coordinate.z), coordinate);
  }
  for (const input of options.proxyBricks) {
    const coordinate = canonicalCoordinate(input, options.maximumDepth, "Proxy brick");
    proxy.set(mortonEncode3D(coordinate.x, coordinate.y, coordinate.z), coordinate);
  }
  return { solver, proxy };
}

function prefixSets(finestKeys: Iterable<bigint>, maximumDepth: number): Set<bigint>[] {
  const levels = Array.from({ length: maximumDepth + 1 }, () => new Set<bigint>());
  for (const finestKey of finestKeys) {
    let key = finestKey;
    for (let level = maximumDepth; level >= 0; level -= 1) {
      levels[level].add(key);
      key >>= 3n;
    }
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
 */
export function planAdaptiveSparseBrickOctree(options: AdaptiveSparseBrickPlanOptions): SparseBrickPlan {
  if (options.brickSize !== 4 && options.brickSize !== 8) throw new RangeError("Sparse brick size must be 4 or 8");
  const maximumDepth = assertDepth(options.maximumDepth);
  const coarseningPower = assertCoarseningPower(options.maximumEnvironmentCoarseningPower, maximumDepth);
  const inputs = canonicalize(options);
  const solverPrefixes = prefixSets(inputs.solver.keys(), maximumDepth);
  const proxyPrefixes = prefixSets(inputs.proxy.keys(), maximumDepth);
  const minimumEnvironmentLevel = maximumDepth - coarseningPower;

  // `${level}:${morton}` is unambiguous because each level has its own Morton domain.
  const leafKeys = new Set<string>();
  for (const key of inputs.solver.keys()) leafKeys.add(`${maximumDepth}:${key}`);

  const addProxyLeaves = (level: number, key: bigint): void => {
    if (!solverPrefixes[level].has(key)) {
      leafKeys.add(`${level}:${key}`);
      return;
    }
    if (level === maximumDepth) return; // The coincident solver leaf provides coverage.
    for (let octant = 0; octant < 8; octant += 1) {
      const child = mortonChild(key, octant);
      if (proxyPrefixes[level + 1].has(child)) addProxyLeaves(level + 1, child);
    }
  };
  for (const key of [...proxyPrefixes[minimumEnvironmentLevel]].sort(compareMorton)) {
    addProxyLeaves(minimumEnvironmentLevel, key);
  }

  const nodesByLevel = Array.from({ length: maximumDepth + 1 }, () => new Map<bigint, SparseBrickCoordinate>());
  for (const leafKey of leafKeys) {
    const separator = leafKey.indexOf(":");
    const leafLevel = Number(leafKey.slice(0, separator));
    let key = BigInt(leafKey.slice(separator + 1));
    for (let level = leafLevel; level >= 0; level -= 1) {
      if (!nodesByLevel[level].has(key)) {
        // At a fixed level the prefix is also the Morton encoding of the local coordinate.
        let x = 0;
        let y = 0;
        let z = 0;
        for (let bit = 0; bit < level; bit += 1) {
          x += Number((key >> BigInt(3 * bit)) & 1n) * 2 ** bit;
          y += Number((key >> BigInt(3 * bit + 1)) & 1n) * 2 ** bit;
          z += Number((key >> BigInt(3 * bit + 2)) & 1n) * 2 ** bit;
        }
        nodesByLevel[level].set(key, { x, y, z });
      }
      key >>= 3n;
    }
  }

  const levelOffsets: number[] = [];
  const nodes: SparseBrickNodePlan[] = [];
  const nodeIndex = new Map<string, number>();
  for (let level = 0; level <= maximumDepth; level += 1) {
    levelOffsets.push(nodes.length);
    for (const [morton, coordinate] of [...nodesByLevel[level]].sort(([a], [b]) => compareMorton(a, b))) {
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

  for (let level = 0; level < maximumDepth; level += 1) {
    for (let index = levelOffsets[level]; index < levelOffsets[level + 1]; index += 1) {
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

  const voxelsPerBrick = options.brickSize ** 3;
  const leaves: SparseBrickLeafPlan[] = [];
  for (const node of nodes) {
    if (!leafKeys.has(`${node.level}:${node.morton}`)) continue;
    const index: number = leaves.length;
    node.leafIndex = index;
    leaves.push({
      index,
      nodeIndex: node.index,
      morton: node.morton,
      coordinate: node.coordinate,
      voxelOffset: index * voxelsPerBrick,
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

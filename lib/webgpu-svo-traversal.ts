import { SPARSE_BRICK_INVALID_INDEX, type SparseBrickSize } from "./sparse-brick-octree";

/** Three-component world-space value. */
export type SvoVec3 = readonly [number, number, number];

export interface SvoRay {
  origin: SvoVec3;
  direction: SvoVec3;
  /** Inclusive ray interval. Defaults to [0, +infinity]. */
  tMin?: number;
  tMax?: number;
}

export interface SvoAabb {
  minimum: SvoVec3;
  maximum: SvoVec3;
}

export interface SvoRayInterval {
  tEnter: number;
  tExit: number;
}

/**
 * Direct view of the existing sparse-brick publication ABI.
 *
 * `nodes` contains eight u32 words per node:
 * `[mortonLo, mortonHi, level, childMask, firstChild, childCount, leafIndex, flags]`.
 * Children are contiguous in ascending octant order; the child for octant `o`
 * is `firstChild + popcount(childMask & ((1 << o) - 1))`.
 *
 * `leaves` contains four u32 words per leaf:
 * `[nodeIndex, voxelOffset, mortonLo, mortonHi]`. Payload voxels begin at
 * `voxelOffset` and retain the existing x-major brick-local addressing.
 * Published counts must be supplied when the arrays are capacity-sized rather
 * than tightly packed, so traversal never reads unpublished storage.
 */
export interface SvoPackedTopologyView {
  nodes: Uint32Array;
  leaves: Uint32Array;
  publishedNodeCount?: number;
  publishedLeafCount?: number;
  /** The existing control word 12. Non-zero publication overflow is fail-closed. */
  overflowFlags?: number;
}

export interface SvoWorldMapping {
  /** World position of finest-brick coordinate (0, 0, 0). */
  origin: SvoVec3;
  /** Per-axis world size of one simulation cell; anisotropy is preserved. */
  cellSize: SvoVec3;
  brickSize: SparseBrickSize;
  maximumDepth: number;
}

export interface SvoTraversalOptions {
  /** Hard bound on node reads. */
  maxNodeVisits?: number;
  /** Hard bound on the explicit near-to-far traversal stack. */
  stackCapacity?: number;
  /** Defaults to node zero, the canonical sparse-brick root. */
  rootNodeIndex?: number;
}

export interface SvoLeafHit extends SvoRayInterval {
  nodeIndex: number;
  leafIndex: number;
  voxelOffset: number;
  level: number;
  /** Node coordinate in its own level's brick lattice. */
  coordinate: readonly [number, number, number];
  bounds: SvoAabb;
}

/** Resolve x-major brick-local coordinates into the published payload arrays. */
export function svoBrickVoxelIndex(
  voxelOffset: number,
  local: readonly [number, number, number],
  brickSize: SparseBrickSize,
): number {
  if (!Number.isSafeInteger(voxelOffset) || voxelOffset < 0) throw new RangeError("Voxel offset must be a non-negative integer");
  if (brickSize !== 4 && brickSize !== 8) throw new RangeError("SVO brick size must be 4 or 8");
  if (local.some((value) => !Number.isInteger(value) || value < 0 || value >= brickSize)) {
    throw new RangeError("Brick-local coordinate is outside the brick");
  }
  return voxelOffset + local[0] + local[1] * brickSize + local[2] * brickSize * brickSize;
}

export type SvoTraversalResult =
  | { status: "hit"; visits: number; hit: SvoLeafHit }
  | { status: "miss"; visits: number }
  | { status: "work-exhausted"; visits: number }
  | { status: "stack-overflow"; visits: number }
  | { status: "source-overflow"; visits: 0; overflowFlags: number }
  | { status: "invalid-topology"; visits: number; reason: string };

const NODE_WORDS = 8;
const LEAF_WORDS = 4;
const DEFAULT_MAX_NODE_VISITS = 256;
const DEFAULT_STACK_CAPACITY = 32;

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function rayRange(ray: SvoRay): readonly [number, number] {
  finiteVec3(ray.origin, "Ray origin");
  finiteVec3(ray.direction, "Ray direction");
  if (ray.direction.every((component) => component === 0)) throw new RangeError("Ray direction must be non-zero");
  const minimum = ray.tMin ?? 0;
  const maximum = ray.tMax ?? Number.POSITIVE_INFINITY;
  if (!Number.isFinite(minimum)) throw new RangeError("Ray tMin must be finite");
  if (!(Number.isFinite(maximum) || maximum === Number.POSITIVE_INFINITY) || maximum < minimum) {
    throw new RangeError("Ray tMax must be at least tMin");
  }
  return [minimum, maximum];
}

/** Robust closed-box slab intersection, including rays parallel to box faces. */
export function intersectSvoRayAabb(ray: SvoRay, bounds: SvoAabb): SvoRayInterval | null {
  const [rayMinimum, rayMaximum] = rayRange(ray);
  finiteVec3(bounds.minimum, "AABB minimum");
  finiteVec3(bounds.maximum, "AABB maximum");
  let enter = rayMinimum;
  let exit = rayMaximum;
  for (let axis = 0; axis < 3; axis += 1) {
    const lower = bounds.minimum[axis];
    const upper = bounds.maximum[axis];
    if (upper < lower) throw new RangeError("AABB maximum must not be below its minimum");
    const origin = ray.origin[axis];
    const direction = ray.direction[axis];
    if (direction === 0) {
      if (origin < lower || origin > upper) return null;
      continue;
    }
    let near = (lower - origin) / direction;
    let far = (upper - origin) / direction;
    if (near > far) [near, far] = [far, near];
    enter = Math.max(enter, near);
    exit = Math.min(exit, far);
    if (exit < enter) return null;
  }
  return { tEnter: enter, tExit: exit };
}

function decodeMorton(low: number, high: number, level: number): readonly [number, number, number] {
  const result = [0, 0, 0];
  for (let bit = 0; bit < level; bit += 1) {
    const scale = 2 ** bit;
    for (let axis = 0; axis < 3; axis += 1) {
      const addressBit = 3 * bit + axis;
      const word = addressBit < 32 ? low : high;
      const shift = addressBit < 32 ? addressBit : addressBit - 32;
      result[axis] += ((word >>> shift) & 1) * scale;
    }
  }
  return result as unknown as readonly [number, number, number];
}

function nodeBounds(nodes: Uint32Array, nodeIndex: number, mapping: SvoWorldMapping): SvoAabb {
  const base = nodeIndex * NODE_WORDS;
  const level = nodes[base + 2];
  const coordinate = decodeMorton(nodes[base], nodes[base + 1], level);
  const scale = 2 ** (mapping.maximumDepth - level) * mapping.brickSize;
  const minimum = coordinate.map((value, axis) => mapping.origin[axis] + value * scale * mapping.cellSize[axis]) as [number, number, number];
  const maximum = coordinate.map((value, axis) => mapping.origin[axis] + (value + 1) * scale * mapping.cellSize[axis]) as [number, number, number];
  return { minimum, maximum };
}

function popcountBefore(mask: number, octant: number): number {
  const before = octant === 0 ? 0 : mask & ((1 << octant) - 1);
  let count = 0;
  for (let bits = before; bits !== 0; bits >>>= 1) count += bits & 1;
  return count;
}

interface StackEntry extends SvoRayInterval {
  nodeIndex: number;
  octant: number;
}

function invalid(visits: number, reason: string): SvoTraversalResult {
  return { status: "invalid-topology", visits, reason };
}

/**
 * CPU mirror of the WGSL traversal below. It returns the nearest terminal leaf
 * AABB, leaving brick payload/SDF intersection to the caller. All resource and
 * work failures are explicit and fail closed instead of silently becoming air.
 */
export function traversePackedSvo(
  ray: SvoRay,
  topology: SvoPackedTopologyView,
  mapping: SvoWorldMapping,
  options: SvoTraversalOptions = {},
): SvoTraversalResult {
  finiteVec3(mapping.origin, "SVO world origin");
  finiteVec3(mapping.cellSize, "SVO cell size");
  if (mapping.cellSize.some((component) => component <= 0)) throw new RangeError("SVO cell size must be positive");
  if (mapping.brickSize !== 4 && mapping.brickSize !== 8) throw new RangeError("SVO brick size must be 4 or 8");
  if (!Number.isInteger(mapping.maximumDepth) || mapping.maximumDepth < 0 || mapping.maximumDepth > 21) {
    throw new RangeError("SVO maximum depth must be an integer from 0 to 21");
  }
  const maxNodeVisits = positiveInteger(options.maxNodeVisits ?? DEFAULT_MAX_NODE_VISITS, "SVO node-visit budget");
  const stackCapacity = positiveInteger(options.stackCapacity ?? DEFAULT_STACK_CAPACITY, "SVO stack capacity");
  const overflowFlags = topology.overflowFlags ?? 0;
  if (overflowFlags !== 0) return { status: "source-overflow", visits: 0, overflowFlags: overflowFlags >>> 0 };
  if (topology.nodes.length % NODE_WORDS !== 0 || topology.leaves.length % LEAF_WORDS !== 0) {
    return invalid(0, "Packed node or leaf array has a partial record");
  }
  const nodeCount = topology.publishedNodeCount ?? topology.nodes.length / NODE_WORDS;
  const leafCount = topology.publishedLeafCount ?? topology.leaves.length / LEAF_WORDS;
  if (!Number.isInteger(nodeCount) || nodeCount < 0 || nodeCount > topology.nodes.length / NODE_WORDS
      || !Number.isInteger(leafCount) || leafCount < 0 || leafCount > topology.leaves.length / LEAF_WORDS) {
    return invalid(0, "Published counts exceed packed topology bounds");
  }
  if (nodeCount === 0) return { status: "miss", visits: 0 };
  const rootNodeIndex = options.rootNodeIndex ?? 0;
  if (!Number.isInteger(rootNodeIndex) || rootNodeIndex < 0 || rootNodeIndex >= nodeCount) {
    return invalid(0, "Root node index is outside the published topology");
  }
  const rootLevel = topology.nodes[rootNodeIndex * NODE_WORDS + 2];
  if (rootLevel > mapping.maximumDepth) return invalid(0, "Node level exceeds maximum depth");
  const rootInterval = intersectSvoRayAabb(ray, nodeBounds(topology.nodes, rootNodeIndex, mapping));
  if (!rootInterval) return { status: "miss", visits: 0 };

  const stack: StackEntry[] = [{ nodeIndex: rootNodeIndex, octant: 0, ...rootInterval }];
  let visits = 0;
  while (stack.length > 0) {
    if (visits >= maxNodeVisits) return { status: "work-exhausted", visits };
    const current = stack.pop() as StackEntry;
    if (current.nodeIndex >= nodeCount) return invalid(visits, "Child node index is outside the published topology");
    const base = current.nodeIndex * NODE_WORDS;
    const level = topology.nodes[base + 2];
    if (level > mapping.maximumDepth) return invalid(visits, "Node level exceeds maximum depth");
    visits += 1;
    const leafIndex = topology.nodes[base + 6];
    if (leafIndex !== SPARSE_BRICK_INVALID_INDEX) {
      if (leafIndex >= leafCount) return invalid(visits, "Leaf index is outside the published topology");
      const leafBase = leafIndex * LEAF_WORDS;
      if (topology.leaves[leafBase] !== current.nodeIndex) return invalid(visits, "Leaf backlink does not match its node");
      const bounds = nodeBounds(topology.nodes, current.nodeIndex, mapping);
      return {
        status: "hit",
        visits,
        hit: {
          nodeIndex: current.nodeIndex,
          leafIndex,
          voxelOffset: topology.leaves[leafBase + 1],
          level,
          coordinate: decodeMorton(topology.nodes[base], topology.nodes[base + 1], level),
          bounds,
          tEnter: current.tEnter,
          tExit: current.tExit,
        },
      };
    }

    const childMask = topology.nodes[base + 3] & 0xff;
    const firstChild = topology.nodes[base + 4];
    const recordedChildCount = topology.nodes[base + 5];
    if (childMask === 0) continue;
    if (firstChild === SPARSE_BRICK_INVALID_INDEX) return invalid(visits, "Non-empty child mask has no first child");
    const childHits: StackEntry[] = [];
    let discoveredChildCount = 0;
    for (let octant = 0; octant < 8; octant += 1) {
      if ((childMask & (1 << octant)) === 0) continue;
      discoveredChildCount += 1;
      const childIndex = firstChild + popcountBefore(childMask, octant);
      if (childIndex >= nodeCount) return invalid(visits, "Child range exceeds the published topology");
      const childBase = childIndex * NODE_WORDS;
      if (topology.nodes[childBase + 2] !== level + 1) return invalid(visits, "Child level is not parent level plus one");
      const interval = intersectSvoRayAabb(ray, nodeBounds(topology.nodes, childIndex, mapping));
      if (interval) childHits.push({ nodeIndex: childIndex, octant, ...interval });
    }
    if (discoveredChildCount !== recordedChildCount) return invalid(visits, "Child count does not match child mask");
    childHits.sort((left, right) => left.tEnter - right.tEnter || left.tExit - right.tExit || left.octant - right.octant);
    if (stack.length + childHits.length > stackCapacity) return { status: "stack-overflow", visits };
    // LIFO stack: farthest first makes the nearest child the next node read.
    for (let index = childHits.length - 1; index >= 0; index -= 1) stack.push(childHits[index]);
  }
  return { status: "miss", visits };
}

/** WGSL result codes mirrored by `SvoTraversalResult`. */
export const SVO_WGSL_STATUS = Object.freeze({
  miss: 0,
  hit: 1,
  workExhausted: 2,
  stackOverflow: 3,
  sourceOverflow: 4,
  invalidTopology: 5,
});

/**
 * Bindings expected by this integration-ready WGSL helper:
 * - `svoControl`: existing 32-word sparse control buffer (group 0, binding 0)
 * - `svoNodes`: topology buffer at `nodeOffsetBytes` (group 0, binding 1)
 * - `svoLeaves`: same topology buffer at `leafOffsetBytes` (group 0, binding 2)
 *
 * It traverses those buffers directly: no debug-record expansion is involved.
 * The fixed stack and visit cap make shader work bounded. A hit identifies a
 * leaf and ray interval; a later payload intersection stage resolves voxels or
 * an analytic field inside that brick.
 */
export const webgpuSvoTraversalWGSL = /* wgsl */ `
const SVO_INVALID: u32 = 0xffffffffu;
const SVO_STATUS_MISS: u32 = 0u;
const SVO_STATUS_HIT: u32 = 1u;
const SVO_STATUS_WORK_EXHAUSTED: u32 = 2u;
const SVO_STATUS_STACK_OVERFLOW: u32 = 3u;
const SVO_STATUS_SOURCE_OVERFLOW: u32 = 4u;
const SVO_STATUS_INVALID_TOPOLOGY: u32 = 5u;
const SVO_STACK_CAPACITY: u32 = 32u;
const SVO_MAX_VISITS: u32 = 256u;

struct SvoNode { address: vec4u, links: vec4u }
struct SvoLeaf { topology: vec4u }
struct SvoRay { origin: vec3f, tMin: f32, direction: vec3f, tMax: f32 }
struct SvoMapping {
  worldOrigin: vec3f,
  brickSize: u32,
  cellSize: vec3f,
  maximumDepth: u32,
  nodeCount: u32,
  leafCount: u32,
  maxVisits: u32,
  _padding: u32,
}
struct SvoStackEntry { nodeIndex: u32, octant: u32, tEnter: f32, tExit: f32 }
struct SvoTraversalHit {
  status: u32,
  visits: u32,
  nodeIndex: u32,
  leafIndex: u32,
  voxelOffset: u32,
  level: u32,
  tEnter: f32,
  tExit: f32,
}

@group(0) @binding(0) var<storage, read> svoControl: array<u32>;
@group(0) @binding(1) var<storage, read> svoNodes: array<SvoNode>;
@group(0) @binding(2) var<storage, read> svoLeaves: array<SvoLeaf>;

fn svoMiss(status: u32, visits: u32) -> SvoTraversalHit {
  return SvoTraversalHit(status, visits, SVO_INVALID, SVO_INVALID, 0u, 0u, 0.0, 0.0);
}

fn svoKeyBit(low: u32, high: u32, bit: u32) -> u32 {
  if (bit < 32u) { return (low >> bit) & 1u; }
  return (high >> (bit - 32u)) & 1u;
}

fn svoDecodeMorton(low: u32, high: u32, level: u32) -> vec3u {
  var result = vec3u(0u);
  for (var bit = 0u; bit < level; bit += 1u) {
    let scale = 1u << bit;
    result.x += svoKeyBit(low, high, 3u * bit) * scale;
    result.y += svoKeyBit(low, high, 3u * bit + 1u) * scale;
    result.z += svoKeyBit(low, high, 3u * bit + 2u) * scale;
  }
  return result;
}

fn svoNodeBounds(node: SvoNode, mapping: SvoMapping) -> mat2x3f {
  let coordinate = vec3f(svoDecodeMorton(node.address.x, node.address.y, node.address.z));
  let scale = f32((1u << (mapping.maximumDepth - node.address.z)) * mapping.brickSize);
  let minimum = mapping.worldOrigin + coordinate * scale * mapping.cellSize;
  return mat2x3f(minimum, minimum + scale * mapping.cellSize);
}

fn svoRayAabb(ray: SvoRay, bounds: mat2x3f) -> vec3f {
  var enter = ray.tMin;
  var exit = ray.tMax;
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let origin = ray.origin[axis];
    let direction = ray.direction[axis];
    let lower = bounds[0][axis];
    let upper = bounds[1][axis];
    if (direction == 0.0) {
      if (origin < lower || origin > upper) { return vec3f(0.0); }
    } else {
      let first = (lower - origin) / direction;
      let second = (upper - origin) / direction;
      enter = max(enter, min(first, second));
      exit = min(exit, max(first, second));
      if (exit < enter) { return vec3f(0.0); }
    }
  }
  return vec3f(1.0, enter, exit);
}

fn svoPopcountBefore(mask: u32, octant: u32) -> u32 {
  if (octant == 0u) { return 0u; }
  return countOneBits(mask & ((1u << octant) - 1u));
}

fn svoBrickVoxelIndex(voxelOffset: u32, local: vec3u, brickSize: u32) -> u32 {
  return voxelOffset + local.x + local.y * brickSize + local.z * brickSize * brickSize;
}

fn svoTraverse(ray: SvoRay, mapping: SvoMapping) -> SvoTraversalHit {
  if (svoControl[12] != 0u) { return svoMiss(SVO_STATUS_SOURCE_OVERFLOW, 0u); }
  if (mapping.nodeCount == 0u) { return svoMiss(SVO_STATUS_MISS, 0u); }
  let rootInterval = svoRayAabb(ray, svoNodeBounds(svoNodes[0], mapping));
  if (rootInterval.x == 0.0) { return svoMiss(SVO_STATUS_MISS, 0u); }
  var stack: array<SvoStackEntry, 32>;
  var stackSize = 1u;
  stack[0] = SvoStackEntry(0u, 0u, rootInterval.y, rootInterval.z);
  var visits = 0u;
  let visitLimit = min(max(mapping.maxVisits, 1u), SVO_MAX_VISITS);
  loop {
    if (stackSize == 0u) { return svoMiss(SVO_STATUS_MISS, visits); }
    if (visits >= visitLimit) { return svoMiss(SVO_STATUS_WORK_EXHAUSTED, visits); }
    stackSize -= 1u;
    let current = stack[stackSize];
    if (current.nodeIndex >= mapping.nodeCount) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
    let node = svoNodes[current.nodeIndex];
    visits += 1u;
    if (node.address.z > mapping.maximumDepth) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
    if (node.links.z != SVO_INVALID) {
      if (node.links.z >= mapping.leafCount) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
      let leaf = svoLeaves[node.links.z];
      if (leaf.topology.x != current.nodeIndex) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
      return SvoTraversalHit(SVO_STATUS_HIT, visits, current.nodeIndex, node.links.z,
        leaf.topology.y, node.address.z, current.tEnter, current.tExit);
    }
    let mask = node.address.w & 0xffu;
    if (mask == 0u) { continue; }
    if (node.links.x == SVO_INVALID || countOneBits(mask) != node.links.y) {
      return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
    }
    var candidates: array<SvoStackEntry, 8>;
    var candidateCount = 0u;
    for (var octant = 0u; octant < 8u; octant += 1u) {
      if ((mask & (1u << octant)) == 0u) { continue; }
      let childIndex = node.links.x + svoPopcountBefore(mask, octant);
      if (childIndex >= mapping.nodeCount) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
      let child = svoNodes[childIndex];
      if (child.address.z != node.address.z + 1u) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
      let interval = svoRayAabb(ray, svoNodeBounds(child, mapping));
      if (interval.x == 0.0) { continue; }
      var insertion = candidateCount;
      loop {
        if (insertion == 0u) { break; }
        let previous = candidates[insertion - 1u];
        let ordered = previous.tEnter < interval.y
          || (previous.tEnter == interval.y && (previous.tExit < interval.z
          || (previous.tExit == interval.z && previous.octant <= octant)));
        if (ordered) { break; }
        candidates[insertion] = previous;
        insertion -= 1u;
      }
      candidates[insertion] = SvoStackEntry(childIndex, octant, interval.y, interval.z);
      candidateCount += 1u;
    }
    if (stackSize + candidateCount > SVO_STACK_CAPACITY) {
      return svoMiss(SVO_STATUS_STACK_OVERFLOW, visits);
    }
    var remaining = candidateCount;
    loop {
      if (remaining == 0u) { break; }
      remaining -= 1u;
      stack[stackSize] = candidates[remaining];
      stackSize += 1u;
    }
  }
  return svoMiss(SVO_STATUS_MISS, visits);
}
`;

export interface WebgpuSvoTraversalBindings {
  group?: number;
  control?: number;
  nodes?: number;
  leaves?: number;
}

/** Remap the helper's three bindings when composing it into a larger renderer shader. */
export function createWebgpuSvoTraversalWGSL(bindings: WebgpuSvoTraversalBindings = {}): string {
  const group = bindings.group ?? 0;
  const control = bindings.control ?? 0;
  const nodes = bindings.nodes ?? 1;
  const leaves = bindings.leaves ?? 2;
  for (const [label, value] of Object.entries({ group, control, nodes, leaves })) {
    if (!Number.isInteger(value) || value < 0) throw new RangeError(`SVO WGSL ${label} must be a non-negative integer`);
  }
  if (new Set([control, nodes, leaves]).size !== 3) throw new RangeError("SVO WGSL bindings must be distinct");
  return webgpuSvoTraversalWGSL
    .replace("@group(0) @binding(0) var<storage, read> svoControl", `@group(${group}) @binding(${control}) var<storage, read> svoControl`)
    .replace("@group(0) @binding(1) var<storage, read> svoNodes", `@group(${group}) @binding(${nodes}) var<storage, read> svoNodes`)
    .replace("@group(0) @binding(2) var<storage, read> svoLeaves", `@group(${group}) @binding(${leaves}) var<storage, read> svoLeaves`);
}

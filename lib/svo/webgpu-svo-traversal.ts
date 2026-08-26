import {
  SPARSE_BRICK_INVALID_INDEX,
  type SparseBrickLeafTerminalKind,
  type SparseBrickSize,
} from "./sparse-brick-octree";

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
 * `[nodeIndex, voxelOffset, terminalKind, terminalIndex]`. Payload voxels begin
 * at `voxelOffset` for kind zero and retain x-major brick-local addressing;
 * other kinds resolve `terminalIndex` through their accepted scene record.
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
  /** Optional mutable counters for deterministic traversal-work diagnostics. */
  diagnostics?: SvoTraversalWorkDiagnostics;
  /** Experimental child enumerator. The default retains the production AABB path. */
  childEnumeration?: "aabb" | "parametric";
}

export interface SvoTraversalWorkDiagnostics {
  rootAabbTests: number;
  internalNodesExpanded: number;
  childAabbTests: number;
}

export interface SvoIntersectionArithmeticComparison {
  previous: { mortonBoundsDecodes: number; rayDirectionDivisions: number };
  optimized: { mortonBoundsDecodes: number; rayDirectionDivisions: number };
}

/**
 * Compare the canonical intersection arithmetic before and after parent-bound
 * derivation. `nonParallelAxes` is three for a general ray and may be lower for
 * axis-aligned diagnostic rays; the optimized path still forms one vec3
 * reciprocal because camera rays are ordinarily non-parallel on every axis.
 */
export function compareSvoIntersectionArithmetic(
  diagnostics: SvoTraversalWorkDiagnostics,
  nonParallelAxes = 3,
): SvoIntersectionArithmeticComparison {
  if (!Number.isInteger(nonParallelAxes) || nonParallelAxes < 1 || nonParallelAxes > 3) {
    throw new RangeError("Non-parallel SVO ray axes must be an integer from 1 to 3");
  }
  const intersectionTests = diagnostics.rootAabbTests + diagnostics.childAabbTests;
  return {
    previous: {
      mortonBoundsDecodes: intersectionTests,
      rayDirectionDivisions: intersectionTests * nonParallelAxes,
    },
    optimized: {
      mortonBoundsDecodes: diagnostics.rootAabbTests + diagnostics.internalNodesExpanded,
      rayDirectionDivisions: 3,
    },
  };
}

export interface SvoLeafHit extends SvoRayInterval {
  nodeIndex: number;
  leafIndex: number;
  voxelOffset: number;
  level: number;
  /** Node coordinate in its own level's brick lattice. */
  coordinate: readonly [number, number, number];
  terminalKind: SparseBrickLeafTerminalKind;
  terminalIndex: number;
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

export interface SvoChildIntersection extends SvoRayInterval {
  octant: number;
}

export interface SvoParametricChildIntersections {
  /** Degenerate midpoint-plane cases deliberately retain closed-AABB semantics. */
  mode: "parametric" | "aabb-fallback";
  intersections: readonly SvoChildIntersection[];
}

function childBounds(parent: SvoAabb, octant: number): SvoAabb {
  const middle = parent.minimum.map((value, axis) => (value + parent.maximum[axis]) * 0.5) as [number, number, number];
  return {
    minimum: middle.map((value, axis) => (octant & (1 << axis)) === 0 ? parent.minimum[axis] : value) as [number, number, number],
    maximum: middle.map((value, axis) => (octant & (1 << axis)) === 0 ? value : parent.maximum[axis]) as [number, number, number],
  };
}

function referenceChildIntersections(ray: SvoRay, parent: SvoAabb, childMask: number): SvoChildIntersection[] {
  const intersections: SvoChildIntersection[] = [];
  for (let octant = 0; octant < 8; octant += 1) {
    if ((childMask & (1 << octant)) === 0) continue;
    const interval = intersectSvoRayAabb(ray, childBounds(parent, octant));
    if (interval) intersections.push({ octant, ...interval });
  }
  intersections.sort((left, right) => left.tEnter - right.tEnter || left.tExit - right.tExit || left.octant - right.octant);
  return intersections;
}

/**
 * Enumerate the at-most-four open child cells crossed by a ray using the three
 * parent midpoint-plane times. Exact plane, edge, point, and zero-length ties
 * fall back to the closed-AABB oracle because those cases intentionally visit
 * every touching child in stable ascending-octant order.
 */
export function enumerateSvoChildIntersectionsParametric(
  ray: SvoRay,
  parent: SvoAabb,
  parentInterval: SvoRayInterval,
  childMask = 0xff,
): SvoParametricChildIntersections {
  if (!Number.isInteger(childMask) || childMask < 0 || childMask > 0xff) {
    throw new RangeError("SVO child mask must be an 8-bit unsigned integer");
  }
  const [rayMinimum, rayMaximum] = rayRange(ray);
  const tEnter = Math.max(parentInterval.tEnter, rayMinimum);
  const tExit = Math.min(parentInterval.tExit, rayMaximum);
  if (tExit < tEnter) return { mode: "parametric", intersections: [] };
  const fallback = (): SvoParametricChildIntersections => ({
    mode: "aabb-fallback",
    intersections: referenceChildIntersections(ray, parent, childMask),
  });
  if (tEnter === tExit) return fallback();

  const middle = parent.minimum.map((value, axis) => (value + parent.maximum[axis]) * 0.5) as [number, number, number];
  const crossings: number[] = [];
  for (let axis = 0; axis < 3; axis += 1) {
    const direction = ray.direction[axis];
    if (direction === 0) {
      if (ray.origin[axis] === middle[axis]) return fallback();
      continue;
    }
    const crossing = (middle[axis] - ray.origin[axis]) / direction;
    if (crossing === tEnter || crossing === tExit) return fallback();
    if (crossing > tEnter && crossing < tExit) {
      if (crossings.includes(crossing)) return fallback();
      crossings.push(crossing);
    }
  }
  crossings.sort((left, right) => left - right);

  const boundaries = [tEnter, ...crossings, tExit];
  const intersections: SvoChildIntersection[] = [];
  for (let segment = 0; segment + 1 < boundaries.length; segment += 1) {
    const lower = boundaries[segment], upper = boundaries[segment + 1];
    const probeT = lower + (upper - lower) * 0.5;
    let octant = 0;
    for (let axis = 0; axis < 3; axis += 1) {
      const coordinate = ray.origin[axis] + ray.direction[axis] * probeT;
      if (coordinate === middle[axis]) return fallback();
      if (coordinate > middle[axis]) octant |= 1 << axis;
    }
    if ((childMask & (1 << octant)) !== 0) intersections.push({ octant, tEnter: lower, tExit: upper });
  }
  return { mode: "parametric", intersections };
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
  const diagnostics = options.diagnostics;
  const childEnumeration = options.childEnumeration ?? "aabb";
  if (childEnumeration !== "aabb" && childEnumeration !== "parametric") {
    throw new RangeError("SVO child enumeration must be aabb or parametric");
  }
  if (diagnostics) {
    diagnostics.rootAabbTests = 0;
    diagnostics.internalNodesExpanded = 0;
    diagnostics.childAabbTests = 0;
  }
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
  if (diagnostics) diagnostics.rootAabbTests += 1;
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
          terminalKind: topology.leaves[leafBase + 2] as SparseBrickLeafTerminalKind,
          terminalIndex: topology.leaves[leafBase + 3],
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
    if (diagnostics) diagnostics.internalNodesExpanded += 1;
    let childHits: StackEntry[] = [];
    let discoveredChildCount = 0;
    for (let octant = 0; octant < 8; octant += 1) {
      if ((childMask & (1 << octant)) === 0) continue;
      discoveredChildCount += 1;
      const childIndex = firstChild + popcountBefore(childMask, octant);
      if (childIndex >= nodeCount) return invalid(visits, "Child range exceeds the published topology");
      const childBase = childIndex * NODE_WORDS;
      if (topology.nodes[childBase + 2] !== level + 1) return invalid(visits, "Child level is not parent level plus one");
      if (childEnumeration === "aabb") {
        if (diagnostics) diagnostics.childAabbTests += 1;
        const interval = intersectSvoRayAabb(ray, nodeBounds(topology.nodes, childIndex, mapping));
        if (interval) childHits.push({ nodeIndex: childIndex, octant, ...interval });
      }
    }
    if (discoveredChildCount !== recordedChildCount) return invalid(visits, "Child count does not match child mask");
    if (childEnumeration === "parametric") {
      const enumeration = enumerateSvoChildIntersectionsParametric(
        ray,
        nodeBounds(topology.nodes, current.nodeIndex, mapping),
        current,
        childMask,
      );
      if (enumeration.mode === "aabb-fallback" && diagnostics) diagnostics.childAabbTests += discoveredChildCount;
      childHits = enumeration.intersections.map((interval) => ({
        ...interval,
        nodeIndex: firstChild + popcountBefore(childMask, interval.octant),
      }));
    } else {
      childHits.sort((left, right) => left.tEnter - right.tEnter || left.tExit - right.tExit || left.octant - right.octant);
    }
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
const SVO_STATUS_CONTINUE: u32 = 6u;
const SVO_LEAF_TERMINAL_VOXELS: u32 = 0u;
const SVO_LEAF_TERMINAL_PLANAR_BOUNDARY: u32 = 1u;
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
struct SvoStackEntry { nodeIndex: u32, tEnter: f32, tExit: f32 }
struct SvoCandidate { nodeIndex: u32, octant: u32, tEnter: f32, tExit: f32 }
struct SvoTraversalContinuation {
  stack: array<SvoStackEntry, 32>,
  current: SvoStackEntry,
  currentBounds: mat2x3f,
  inverseDirection: vec3f,
  stackSize: u32,
  currentBoundsValid: u32,
  status: u32,
}
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
fn svoControlLoad(index:u32)->u32{return svoControl[index];}
fn svoNodeLoad(index:u32)->SvoNode{return svoNodes[index];}
fn svoLeafLoad(index:u32)->SvoLeaf{return svoLeaves[index];}

fn svoMiss(status: u32, visits: u32) -> SvoTraversalHit {
  return SvoTraversalHit(status, visits, SVO_INVALID, SVO_INVALID, 0u, 0u, 0.0, 0.0);
}

fn svoCompactMortonBits(value: vec3u) -> vec3u {
  var compact = value & vec3u(0x49249249u);
  compact = (compact ^ (compact >> vec3u(2u))) & vec3u(0xc30c30c3u);
  compact = (compact ^ (compact >> vec3u(4u))) & vec3u(0x0f00f00fu);
  compact = (compact ^ (compact >> vec3u(8u))) & vec3u(0xff0000ffu);
  return (compact ^ (compact >> vec3u(16u))) & vec3u(0x0000ffffu);
}

fn svoDecodeMorton(low: u32, high: u32, level: u32) -> vec3u {
  let levelMask = (1u << level) - 1u;
  let lowBits = svoCompactMortonBits(vec3u(low, low >> 1u, low >> 2u));
  let highBits = svoCompactMortonBits(vec3u(high >> 1u, high >> 2u, high));
  return (lowBits | (highBits << vec3u(11u, 11u, 10u))) & vec3u(levelMask);
}

fn svoNodeBounds(node: SvoNode, mapping: SvoMapping) -> mat2x3f {
  let coordinate = vec3f(svoDecodeMorton(node.address.x, node.address.y, node.address.z));
  let scale = f32((1u << (mapping.maximumDepth - node.address.z)) * mapping.brickSize);
  let minimum = mapping.worldOrigin + coordinate * scale * mapping.cellSize;
  return mat2x3f(minimum, minimum + scale * mapping.cellSize);
}

fn svoRootBounds(mapping: SvoMapping) -> mat2x3f {
  let scale = f32((1u << mapping.maximumDepth) * mapping.brickSize);
  return mat2x3f(mapping.worldOrigin, mapping.worldOrigin + scale * mapping.cellSize);
}

fn svoRayAabb(ray: SvoRay, bounds: mat2x3f) -> vec3f {
  return svoRayAabbWithInverse(ray, 1.0 / ray.direction, bounds);
}

fn svoRayAabbWithInverse(ray: SvoRay, inverseDirection: vec3f, bounds: mat2x3f) -> vec3f {
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
      let first = (lower - origin) * inverseDirection[axis];
      let second = (upper - origin) * inverseDirection[axis];
      enter = max(enter, min(first, second));
      exit = min(exit, max(first, second));
      if (exit < enter) { return vec3f(0.0); }
    }
  }
  return vec3f(1.0, enter, exit);
}

fn svoChildBounds(parent: mat2x3f, octant: u32) -> mat2x3f {
  let middle = (parent[0] + parent[1]) * 0.5;
  let high = vec3f(f32(octant & 1u), f32((octant >> 1u) & 1u), f32((octant >> 2u) & 1u));
  return mat2x3f(mix(parent[0], middle, high), mix(middle, parent[1], high));
}

fn svoPopcountBefore(mask: u32, octant: u32) -> u32 {
  if (octant == 0u) { return 0u; }
  return countOneBits(mask & ((1u << octant) - 1u));
}

fn svoBrickVoxelIndex(voxelOffset: u32, local: vec3u, brickSize: u32) -> u32 {
  return voxelOffset + local.x + local.y * brickSize + local.z * brickSize * brickSize;
}

fn svoTraversalContinuationAdvance(continuation: ptr<function, SvoTraversalContinuation>) {
  if ((*continuation).stackSize == 0u) {
    (*continuation).status = SVO_STATUS_MISS;
    return;
  }
  (*continuation).stackSize -= 1u;
  (*continuation).current = (*continuation).stack[(*continuation).stackSize];
  (*continuation).currentBoundsValid = 0u;
}

// Initialize a near-to-far traversal cursor once. svoTraversalContinuationNext
// then yields successive leaves without re-reading the root and common ancestors.
fn svoTraversalContinuationBegin(
  ray: SvoRay,
  mapping: SvoMapping,
  continuation: ptr<function, SvoTraversalContinuation>,
) {
  (*continuation).stackSize = 0u;
  (*continuation).currentBoundsValid = 0u;
  (*continuation).status = SVO_STATUS_CONTINUE;
  if (svoControlLoad(12u) != 0u) {
    (*continuation).status = SVO_STATUS_SOURCE_OVERFLOW;
    return;
  }
  if (mapping.nodeCount == 0u) {
    (*continuation).status = SVO_STATUS_MISS;
    return;
  }
  let root = svoNodeLoad(0u);
  if (root.address.x != 0u || root.address.y != 0u || root.address.z != 0u) {
    (*continuation).status = SVO_STATUS_INVALID_TOPOLOGY;
    return;
  }
  (*continuation).inverseDirection = 1.0 / ray.direction;
  let rootBounds = svoRootBounds(mapping);
  let rootInterval = svoRayAabbWithInverse(ray, (*continuation).inverseDirection, rootBounds);
  if (rootInterval.x == 0.0) {
    (*continuation).status = SVO_STATUS_MISS;
    return;
  }
  (*continuation).current = SvoStackEntry(0u, rootInterval.y, rootInterval.z);
  (*continuation).currentBounds = rootBounds;
  (*continuation).currentBoundsValid = 1u;
}

// Resume the cursor until the next terminal leaf or a terminal traversal status.
// ray.tMin may advance between calls; deferred entries behind it are discarded,
// matching a fresh traversal over the narrower interval without restarting at root.
fn svoTraversalContinuationNext(
  ray: SvoRay,
  mapping: SvoMapping,
  maximumTraversalDepth: u32,
  continuation: ptr<function, SvoTraversalContinuation>,
) -> SvoTraversalHit {
  if ((*continuation).status != SVO_STATUS_CONTINUE) {
    return svoMiss((*continuation).status, 0u);
  }
  var visits = 0u;
  let visitLimit = min(max(mapping.maxVisits, 1u), SVO_MAX_VISITS);
  var traversalGuard = 0u;
  while (traversalGuard <= SVO_MAX_VISITS) {
    traversalGuard += 1u;
    let current = (*continuation).current;
    if (current.tExit < ray.tMin || current.tEnter > ray.tMax) {
      svoTraversalContinuationAdvance(continuation);
      if ((*continuation).status != SVO_STATUS_CONTINUE) {
        return svoMiss((*continuation).status, visits);
      }
      continue;
    }
    if (visits >= visitLimit) {
      (*continuation).status = SVO_STATUS_WORK_EXHAUSTED;
      return svoMiss(SVO_STATUS_WORK_EXHAUSTED, visits);
    }
    if (current.nodeIndex >= mapping.nodeCount) {
      (*continuation).status = SVO_STATUS_INVALID_TOPOLOGY;
      return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
    }
    let node = svoNodeLoad(current.nodeIndex);
    visits += 1u;
    if (node.address.z > mapping.maximumDepth) {
      (*continuation).status = SVO_STATUS_INVALID_TOPOLOGY;
      return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
    }
    if (node.address.z > maximumTraversalDepth) {
      (*continuation).status = SVO_STATUS_WORK_EXHAUSTED;
      return svoMiss(SVO_STATUS_WORK_EXHAUSTED, visits);
    }
    if (node.links.z != SVO_INVALID) {
      if (node.links.z >= mapping.leafCount) {
        (*continuation).status = SVO_STATUS_INVALID_TOPOLOGY;
        return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
      }
      let leaf = svoLeafLoad(node.links.z);
      if (leaf.topology.x != current.nodeIndex) {
        (*continuation).status = SVO_STATUS_INVALID_TOPOLOGY;
        return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
      }
      let hit = SvoTraversalHit(SVO_STATUS_HIT, visits, current.nodeIndex, node.links.z,
        leaf.topology.y, node.address.z, max(current.tEnter, ray.tMin), min(current.tExit, ray.tMax));
      svoTraversalContinuationAdvance(continuation);
      return hit;
    }
    let mask = node.address.w & 0xffu;
    if (mask == 0u) {
      svoTraversalContinuationAdvance(continuation);
      if ((*continuation).status != SVO_STATUS_CONTINUE) {
        return svoMiss((*continuation).status, visits);
      }
      continue;
    }
    if (node.links.x == SVO_INVALID || countOneBits(mask) != node.links.y) {
      (*continuation).status = SVO_STATUS_INVALID_TOPOLOGY;
      return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
    }
    var parentBounds = (*continuation).currentBounds;
    if ((*continuation).currentBoundsValid == 0u) { parentBounds = svoNodeBounds(node, mapping); }
    // SVO_CONTINUATION_CHILD_EXPANSION_BEGIN
    // Streaming expansion: the nearest candidate lives in registers while every
    // deferred sibling is insertion-sorted directly into the traversal stack
    // region [expansionBase, stackSize), kept farthest-at-bottom so pops stay
    // nearest-first. This retires the eight-entry scratch array while keeping
    // the exact (tEnter, tExit, octant) comparison sequence of the sorted-array
    // expansion; sibling nodeIndex order equals octant order because children
    // are firstChild + popcountBefore(mask, octant).
    let expansionBase = (*continuation).stackSize;
    var nearest = SvoCandidate(0u, 0u, 0.0, 0.0);
    var candidateCount = 0u;
    var overflowPending = false;
    for (var octant = 0u; octant < 8u; octant += 1u) {
      if ((mask & (1u << octant)) == 0u) { continue; }
      let childIndex = node.links.x + svoPopcountBefore(mask, octant);
      if (childIndex >= mapping.nodeCount) {
        (*continuation).status = SVO_STATUS_INVALID_TOPOLOGY;
        return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
      }
      // No child record is read here. Child bounds are halved out of the
      // parent's, and the child's own level is a build-time invariant of the
      // planner (adaptive-sparse-brick-plan.ts), not a per-ray property.
      if (overflowPending) { continue; }
      let childBounds = svoChildBounds(parentBounds, octant);
      let interval = svoRayAabbWithInverse(ray, (*continuation).inverseDirection, childBounds);
      if (interval.x == 0.0) { continue; }
      candidateCount += 1u;
      if (candidateCount == 1u) {
        nearest = SvoCandidate(childIndex, octant, interval.y, interval.z);
        continue;
      }
      var deferred = SvoStackEntry(childIndex, interval.y, interval.z);
      let nearestOrdered = nearest.tEnter < interval.y
        || (nearest.tEnter == interval.y && (nearest.tExit < interval.z
        || (nearest.tExit == interval.z && nearest.nodeIndex <= childIndex)));
      if (!nearestOrdered) {
        deferred = SvoStackEntry(nearest.nodeIndex, nearest.tEnter, nearest.tExit);
        nearest = SvoCandidate(childIndex, octant, interval.y, interval.z);
      }
      if ((*continuation).stackSize == SVO_STACK_CAPACITY) {
        overflowPending = true;
        continue;
      }
      var insertion = (*continuation).stackSize;
      loop {
        if (insertion == expansionBase) { break; }
        let settled = (*continuation).stack[insertion - 1u];
        let ordered = deferred.tEnter < settled.tEnter
          || (deferred.tEnter == settled.tEnter && (deferred.tExit < settled.tExit
          || (deferred.tExit == settled.tExit && deferred.nodeIndex <= settled.nodeIndex)));
        if (ordered) { break; }
        (*continuation).stack[insertion] = settled;
        insertion -= 1u;
      }
      (*continuation).stack[insertion] = deferred;
      (*continuation).stackSize += 1u;
    }
    if (candidateCount == 0u) {
      svoTraversalContinuationAdvance(continuation);
      if ((*continuation).status != SVO_STATUS_CONTINUE) {
        return svoMiss((*continuation).status, visits);
      }
      continue;
    }
    if (overflowPending) {
      (*continuation).status = SVO_STATUS_STACK_OVERFLOW;
      return svoMiss(SVO_STATUS_STACK_OVERFLOW, visits);
    }
    (*continuation).current = SvoStackEntry(nearest.nodeIndex, nearest.tEnter, nearest.tExit);
    (*continuation).currentBounds = svoChildBounds(parentBounds, nearest.octant);
    (*continuation).currentBoundsValid = 1u;
    // SVO_CONTINUATION_CHILD_EXPANSION_END
  }
  (*continuation).status = SVO_STATUS_WORK_EXHAUSTED;
  return svoMiss(SVO_STATUS_WORK_EXHAUSTED, visits);
}

fn svoTraverseWithDepthLimit(ray: SvoRay, mapping: SvoMapping, maximumTraversalDepth: u32) -> SvoTraversalHit {
  if (svoControlLoad(12u) != 0u) { return svoMiss(SVO_STATUS_SOURCE_OVERFLOW, 0u); }
  if (mapping.nodeCount == 0u) { return svoMiss(SVO_STATUS_MISS, 0u); }
  let root = svoNodeLoad(0u);
  if (root.address.x != 0u || root.address.y != 0u || root.address.z != 0u) {
    return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, 0u);
  }
  let inverseDirection = 1.0 / ray.direction;
  let rootBounds = svoRootBounds(mapping);
  let rootInterval = svoRayAabbWithInverse(ray, inverseDirection, rootBounds);
  if (rootInterval.x == 0.0) { return svoMiss(SVO_STATUS_MISS, 0u); }
  var stack: array<SvoStackEntry, 32>;
  var stackSize = 0u;
  var current = SvoStackEntry(0u, rootInterval.y, rootInterval.z);
  var currentBounds = rootBounds;
  var currentBoundsValid = true;
  var visits = 0u;
  let visitLimit = min(max(mapping.maxVisits, 1u), SVO_MAX_VISITS);
  var traversalGuard = 0u;
  while (traversalGuard <= SVO_MAX_VISITS) {
    traversalGuard += 1u;
    if (visits >= visitLimit) { return svoMiss(SVO_STATUS_WORK_EXHAUSTED, visits); }
    if (current.nodeIndex >= mapping.nodeCount) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
    let node = svoNodeLoad(current.nodeIndex);
    visits += 1u;
    if (node.address.z > mapping.maximumDepth) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
    if (node.address.z > maximumTraversalDepth) { return svoMiss(SVO_STATUS_WORK_EXHAUSTED, visits); }
    if (node.links.z != SVO_INVALID) {
      if (node.links.z >= mapping.leafCount) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
      let leaf = svoLeafLoad(node.links.z);
      if (leaf.topology.x != current.nodeIndex) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
      return SvoTraversalHit(SVO_STATUS_HIT, visits, current.nodeIndex, node.links.z,
        leaf.topology.y, node.address.z, current.tEnter, current.tExit);
    }
    let mask = node.address.w & 0xffu;
    if (mask == 0u) {
      if (stackSize == 0u) { return svoMiss(SVO_STATUS_MISS, visits); }
      stackSize -= 1u;
      current = stack[stackSize];
      currentBoundsValid = false;
      continue;
    }
    if (node.links.x == SVO_INVALID || countOneBits(mask) != node.links.y) {
      return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
    }
    var parentBounds = currentBounds;
    if (!currentBoundsValid) { parentBounds = svoNodeBounds(node, mapping); }
    // SVO_RESTART_CHILD_EXPANSION_BEGIN
    // Streaming expansion: identical ordering contract to the continuation
    // variant above — nearest candidate in registers, deferred siblings
    // insertion-sorted into the stack region [expansionBase, stackSize).
    let expansionBase = stackSize;
    var nearest = SvoCandidate(0u, 0u, 0.0, 0.0);
    var candidateCount = 0u;
    var overflowPending = false;
    for (var octant = 0u; octant < 8u; octant += 1u) {
      if ((mask & (1u << octant)) == 0u) { continue; }
      let childIndex = node.links.x + svoPopcountBefore(mask, octant);
      if (childIndex >= mapping.nodeCount) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
      // See the continuation variant: the child record is not needed to bound or
      // to order a child, and its level is a build-time invariant.
      if (overflowPending) { continue; }
      let childBounds = svoChildBounds(parentBounds, octant);
      let interval = svoRayAabbWithInverse(ray, inverseDirection, childBounds);
      if (interval.x == 0.0) { continue; }
      candidateCount += 1u;
      if (candidateCount == 1u) {
        nearest = SvoCandidate(childIndex, octant, interval.y, interval.z);
        continue;
      }
      var deferred = SvoStackEntry(childIndex, interval.y, interval.z);
      let nearestOrdered = nearest.tEnter < interval.y
        || (nearest.tEnter == interval.y && (nearest.tExit < interval.z
        || (nearest.tExit == interval.z && nearest.nodeIndex <= childIndex)));
      if (!nearestOrdered) {
        deferred = SvoStackEntry(nearest.nodeIndex, nearest.tEnter, nearest.tExit);
        nearest = SvoCandidate(childIndex, octant, interval.y, interval.z);
      }
      if (stackSize == SVO_STACK_CAPACITY) {
        overflowPending = true;
        continue;
      }
      var insertion = stackSize;
      loop {
        if (insertion == expansionBase) { break; }
        let settled = stack[insertion - 1u];
        let ordered = deferred.tEnter < settled.tEnter
          || (deferred.tEnter == settled.tEnter && (deferred.tExit < settled.tExit
          || (deferred.tExit == settled.tExit && deferred.nodeIndex <= settled.nodeIndex)));
        if (ordered) { break; }
        stack[insertion] = settled;
        insertion -= 1u;
      }
      stack[insertion] = deferred;
      stackSize += 1u;
    }
    if (candidateCount == 0u) {
      if (stackSize == 0u) { return svoMiss(SVO_STATUS_MISS, visits); }
      stackSize -= 1u;
      current = stack[stackSize];
      currentBoundsValid = false;
      continue;
    }
    if (overflowPending) {
      return svoMiss(SVO_STATUS_STACK_OVERFLOW, visits);
    }
    current = SvoStackEntry(nearest.nodeIndex, nearest.tEnter, nearest.tExit);
    currentBounds = svoChildBounds(parentBounds, nearest.octant);
    currentBoundsValid = true;
    // SVO_RESTART_CHILD_EXPANSION_END
  }
  return svoMiss(SVO_STATUS_WORK_EXHAUSTED, visits);
}
fn svoTraverse(ray: SvoRay, mapping: SvoMapping) -> SvoTraversalHit {
  return svoTraverseWithDepthLimit(ray,mapping,mapping.maximumDepth);
}
`;

const SVO_PARAMETRIC_CHILD_HELPERS_WGSL = /* wgsl */ `
struct SvoParametricSegments {
  crossings: vec4f,
  tExit: f32,
  crossingCount: u32,
  valid: u32,
  _padding: u32,
}

fn svoParametricSegmentInterval(segments: SvoParametricSegments, segment: u32) -> vec2f {
  let lower = segments.crossings[segment];
  var upper = segments.tExit;
  if (segment < segments.crossingCount) { upper = segments.crossings[segment + 1u]; }
  return vec2f(lower, upper);
}

fn svoParametricSegmentOctant(ray: SvoRay, parentBounds: mat2x3f, interval: vec2f) -> u32 {
  let probeT = interval.x + (interval.y - interval.x) * 0.5;
  let point = ray.origin + ray.direction * probeT;
  let middle = (parentBounds[0] + parentBounds[1]) * 0.5;
  return select(0u, 1u, point.x > middle.x)
    | select(0u, 2u, point.y > middle.y)
    | select(0u, 4u, point.z > middle.z);
}

// A line crosses at most three midpoint planes and therefore at most four open
// octree children. Exact plane/edge/point ties return valid=0 so the caller can
// preserve the reference traversal's inclusive touching-child behavior.
fn svoParametricSegments(ray: SvoRay, parentBounds: mat2x3f, current: SvoStackEntry) -> SvoParametricSegments {
  let tEnter = max(current.tEnter, ray.tMin);
  let tExit = min(current.tExit, ray.tMax);
  var crossings = vec4f(tEnter, 0.0, 0.0, 0.0);
  if (tExit <= tEnter) { return SvoParametricSegments(crossings, tExit, 0u, 0u, 0u); }
  let middle = (parentBounds[0] + parentBounds[1]) * 0.5;
  var crossingCount = 0u;
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let direction = ray.direction[axis];
    if (direction == 0.0) {
      if (ray.origin[axis] == middle[axis]) { return SvoParametricSegments(crossings, tExit, crossingCount, 0u, 0u); }
      continue;
    }
    let crossing = (middle[axis] - ray.origin[axis]) / direction;
    if (crossing == tEnter || crossing == tExit) {
      return SvoParametricSegments(crossings, tExit, crossingCount, 0u, 0u);
    }
    if (crossing <= tEnter || crossing >= tExit) { continue; }
    for (var previous = 1u; previous <= crossingCount; previous += 1u) {
      if (crossings[previous] == crossing) {
        return SvoParametricSegments(crossings, tExit, crossingCount, 0u, 0u);
      }
    }
    var insertion = crossingCount + 1u;
    loop {
      if (insertion == 1u || crossings[insertion - 1u] < crossing) { break; }
      crossings[insertion] = crossings[insertion - 1u];
      insertion -= 1u;
    }
    crossings[insertion] = crossing;
    crossingCount += 1u;
  }
  let segments = SvoParametricSegments(crossings, tExit, crossingCount, 1u, 0u);
  for (var segment = 0u; segment <= crossingCount; segment += 1u) {
    let interval = svoParametricSegmentInterval(segments, segment);
    let probeT = interval.x + (interval.y - interval.x) * 0.5;
    let point = ray.origin + ray.direction * probeT;
    if (any(point == middle)) { return SvoParametricSegments(crossings, tExit, crossingCount, 0u, 0u); }
  }
  return segments;
}
`;

function sectionBetween(source: string, begin: string, end: string): string {
  const beginIndex = source.indexOf(begin), endIndex = source.indexOf(end, beginIndex + begin.length);
  if (beginIndex < 0 || endIndex < 0) throw new Error(`Missing WGSL section ${begin}`);
  return source.slice(beginIndex + begin.length, endIndex);
}

function replaceSection(source: string, begin: string, end: string, body: string): string {
  const beginIndex = source.indexOf(begin), endIndex = source.indexOf(end, beginIndex + begin.length);
  if (beginIndex < 0 || endIndex < 0) throw new Error(`Missing WGSL section ${begin}`);
  return source.slice(0, beginIndex + begin.length) + body + source.slice(endIndex);
}

function parametricContinuationExpansion(aabbFallback: string): string {
  return /* wgsl */ `
    let parametricSegments = svoParametricSegments(ray, parentBounds, current);
    if (parametricSegments.valid == 0u) {
${aabbFallback}
    } else {
      // Children are one contiguous run, so the whole sparse child range is
      // bounds-checked by its last element and no child record is read at all.
      // countOneBits(mask) equals node.links.y, already agreed above, and the
      // lowest present octant addresses node.links.x itself, so this is the
      // same acceptance set the per-octant loop had — without its up to eight
      // dependent 32-byte loads, every one of which was discarded after
      // asserting a level the node packer establishes once per world build.
      if (node.links.x + countOneBits(mask) > mapping.nodeCount) {
        (*continuation).status = SVO_STATUS_INVALID_TOPOLOGY;
        return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
      }
      var nearest = SvoCandidate(0u, 0u, 0.0, 0.0);
      var nearestSegment = 0u;
      var candidateCount = 0u;
      let segmentCount = parametricSegments.crossingCount + 1u;
      for (var segment = 0u; segment < segmentCount; segment += 1u) {
        let interval = svoParametricSegmentInterval(parametricSegments, segment);
        let octant = svoParametricSegmentOctant(ray, parentBounds, interval);
        if ((mask & (1u << octant)) == 0u) { continue; }
        let childIndex = node.links.x + svoPopcountBefore(mask, octant);
        if (candidateCount == 0u) {
          nearest = SvoCandidate(childIndex, octant, interval.x, interval.y);
          nearestSegment = segment;
        }
        candidateCount += 1u;
      }
      if (candidateCount == 0u) {
        svoTraversalContinuationAdvance(continuation);
        if ((*continuation).status != SVO_STATUS_CONTINUE) {
          return svoMiss((*continuation).status, visits);
        }
        continue;
      }
      if ((*continuation).stackSize + candidateCount - 1u > SVO_STACK_CAPACITY) {
        (*continuation).status = SVO_STATUS_STACK_OVERFLOW;
        return svoMiss(SVO_STATUS_STACK_OVERFLOW, visits);
      }
      var reverseSegment = segmentCount;
      loop {
        if (reverseSegment == 0u) { break; }
        reverseSegment -= 1u;
        if (reverseSegment == nearestSegment) { continue; }
        let interval = svoParametricSegmentInterval(parametricSegments, reverseSegment);
        let octant = svoParametricSegmentOctant(ray, parentBounds, interval);
        if ((mask & (1u << octant)) == 0u) { continue; }
        let childIndex = node.links.x + svoPopcountBefore(mask, octant);
        (*continuation).stack[(*continuation).stackSize] = SvoStackEntry(childIndex, interval.x, interval.y);
        (*continuation).stackSize += 1u;
      }
      (*continuation).current = SvoStackEntry(nearest.nodeIndex, nearest.tEnter, nearest.tExit);
      (*continuation).currentBounds = svoChildBounds(parentBounds, nearest.octant);
      (*continuation).currentBoundsValid = 1u;
    }
`;
}

function parametricRestartExpansion(aabbFallback: string): string {
  return /* wgsl */ `
    let parametricSegments = svoParametricSegments(ray, parentBounds, current);
    if (parametricSegments.valid == 0u) {
${aabbFallback}
    } else {
      // Same collapse as the continuation variant above.
      if (node.links.x + countOneBits(mask) > mapping.nodeCount) {
        return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
      }
      var nearest = SvoCandidate(0u, 0u, 0.0, 0.0);
      var nearestSegment = 0u;
      var candidateCount = 0u;
      let segmentCount = parametricSegments.crossingCount + 1u;
      for (var segment = 0u; segment < segmentCount; segment += 1u) {
        let interval = svoParametricSegmentInterval(parametricSegments, segment);
        let octant = svoParametricSegmentOctant(ray, parentBounds, interval);
        if ((mask & (1u << octant)) == 0u) { continue; }
        let childIndex = node.links.x + svoPopcountBefore(mask, octant);
        if (candidateCount == 0u) {
          nearest = SvoCandidate(childIndex, octant, interval.x, interval.y);
          nearestSegment = segment;
        }
        candidateCount += 1u;
      }
      if (candidateCount == 0u) {
        if (stackSize == 0u) { return svoMiss(SVO_STATUS_MISS, visits); }
        stackSize -= 1u;
        current = stack[stackSize];
        currentBoundsValid = false;
        continue;
      }
      if (stackSize + candidateCount - 1u > SVO_STACK_CAPACITY) {
        return svoMiss(SVO_STATUS_STACK_OVERFLOW, visits);
      }
      var reverseSegment = segmentCount;
      loop {
        if (reverseSegment == 0u) { break; }
        reverseSegment -= 1u;
        if (reverseSegment == nearestSegment) { continue; }
        let interval = svoParametricSegmentInterval(parametricSegments, reverseSegment);
        let octant = svoParametricSegmentOctant(ray, parentBounds, interval);
        if ((mask & (1u << octant)) == 0u) { continue; }
        let childIndex = node.links.x + svoPopcountBefore(mask, octant);
        stack[stackSize] = SvoStackEntry(childIndex, interval.x, interval.y);
        stackSize += 1u;
      }
      current = SvoStackEntry(nearest.nodeIndex, nearest.tEnter, nearest.tExit);
      currentBounds = svoChildBounds(parentBounds, nearest.octant);
      currentBoundsValid = true;
    }
`;
}

export interface WebgpuSvoTraversalBindings {
  group?: number;
  control?: number;
  nodes?: number;
  leaves?: number;
  /** Experimental internal-node expansion; omitted keeps the production shader byte-identical. */
  childEnumeration?: "aabb" | "parametric";
  /** Experimental private stack capacity; omitted keeps the 32-entry production stack. */
  stackCapacity?: 8 | 16 | 32;
  /** One raw structural arena. Offsets are WGSL u32 word-offset expressions. */
  arena?: Readonly<{
    binding: number;
    controlOffset: string;
    nodeOffset: string;
    leafOffset: string;
  }>;
}

/** Remap the helper's three bindings when composing it into a larger renderer shader. */
export function createWebgpuSvoTraversalWGSL(bindings: WebgpuSvoTraversalBindings = {}): string {
  const group = bindings.group ?? 0;
  const control = bindings.control ?? 0;
  const nodes = bindings.nodes ?? 1;
  const leaves = bindings.leaves ?? 2;
  const childEnumeration = bindings.childEnumeration ?? "aabb";
  const stackCapacity = bindings.stackCapacity ?? 32;
  const arena = bindings.arena;
  for (const [label, value] of Object.entries({ group, control, nodes, leaves })) {
    if (!Number.isInteger(value) || value < 0) throw new RangeError(`SVO WGSL ${label} must be a non-negative integer`);
  }
  if (!arena && new Set([control, nodes, leaves]).size !== 3) throw new RangeError("SVO WGSL bindings must be distinct");
  if (arena && (!Number.isInteger(arena.binding) || arena.binding < 0
    || !arena.controlOffset || !arena.nodeOffset || !arena.leafOffset)) {
    throw new RangeError("SVO WGSL structural arena requires one binding and three word-offset expressions");
  }
  if (childEnumeration !== "aabb" && childEnumeration !== "parametric") {
    throw new RangeError("SVO WGSL child enumeration must be aabb or parametric");
  }
  if (stackCapacity !== 8 && stackCapacity !== 16 && stackCapacity !== 32) {
    throw new RangeError("SVO WGSL stack capacity must be 8, 16, or 32");
  }
  let source = webgpuSvoTraversalWGSL;
  if (childEnumeration === "parametric") {
    const continuationBegin = "// SVO_CONTINUATION_CHILD_EXPANSION_BEGIN";
    const continuationEnd = "// SVO_CONTINUATION_CHILD_EXPANSION_END";
    const restartBegin = "// SVO_RESTART_CHILD_EXPANSION_BEGIN";
    const restartEnd = "// SVO_RESTART_CHILD_EXPANSION_END";
    source = source.replace("fn svoTraversalContinuationAdvance", `${SVO_PARAMETRIC_CHILD_HELPERS_WGSL}\nfn svoTraversalContinuationAdvance`);
    source = replaceSection(source, continuationBegin, continuationEnd,
      parametricContinuationExpansion(sectionBetween(source, continuationBegin, continuationEnd)));
    source = replaceSection(source, restartBegin, restartEnd,
      parametricRestartExpansion(sectionBetween(source, restartBegin, restartEnd)));
  }
  if (stackCapacity !== 32) {
    source = source
      .replace("const SVO_STACK_CAPACITY: u32 = 32u;", `const SVO_STACK_CAPACITY: u32 = ${stackCapacity}u;`)
      .replaceAll("array<SvoStackEntry, 32>", `array<SvoStackEntry, ${stackCapacity}>`);
  }
  if (arena) {
    const declarations = `@group(0) @binding(0) var<storage, read> svoControl: array<u32>;
@group(0) @binding(1) var<storage, read> svoNodes: array<SvoNode>;
@group(0) @binding(2) var<storage, read> svoLeaves: array<SvoLeaf>;
fn svoControlLoad(index:u32)->u32{return svoControl[index];}
fn svoNodeLoad(index:u32)->SvoNode{return svoNodes[index];}
fn svoLeafLoad(index:u32)->SvoLeaf{return svoLeaves[index];}`;
    const arenaDeclarations = `@group(${group}) @binding(${arena.binding}) var<storage,read> svoStructure:array<u32>;
fn svoStructureWords4(offset:u32)->vec4u{return vec4u(svoStructure[offset],svoStructure[offset+1u],svoStructure[offset+2u],svoStructure[offset+3u]);}
fn svoControlLoad(index:u32)->u32{return svoStructure[${arena.controlOffset}+index];}
fn svoNodeLoad(index:u32)->SvoNode{let base=${arena.nodeOffset}+index*8u;return SvoNode(svoStructureWords4(base),svoStructureWords4(base+4u));}
fn svoLeafLoad(index:u32)->SvoLeaf{let base=${arena.leafOffset}+index*4u;return SvoLeaf(svoStructureWords4(base));}`;
    return source.replace(declarations, arenaDeclarations);
  }
  return source
    .replace("@group(0) @binding(0) var<storage, read> svoControl", `@group(${group}) @binding(${control}) var<storage, read> svoControl`)
    .replace("@group(0) @binding(1) var<storage, read> svoNodes", `@group(${group}) @binding(${nodes}) var<storage, read> svoNodes`)
    .replace("@group(0) @binding(2) var<storage, read> svoLeaves", `@group(${group}) @binding(${leaves}) var<storage, read> svoLeaves`);
}

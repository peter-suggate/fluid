/**
 * Immutable renderer-facing packing of the canonical binary SVO.
 *
 * The simulation/publication ABI remains eight words per node. Rendering only
 * needs five: Morton lo/hi, packed level/mask/terminal, link-or-leaf, and flags.
 * Child count is exactly popcount(mask), so storing it in the hot view would be
 * redundant. Children and leaf records retain their canonical indices.
 */

import { SPARSE_BRICK_INVALID_INDEX } from "./sparse-brick-octree";
import {
  intersectSvoRayAabb,
  type SvoAabb,
  type SvoLeafHit,
  type SvoPackedTopologyView,
  type SvoRay,
  type SvoTraversalOptions,
  type SvoTraversalResult,
  type SvoWorldMapping,
} from "./webgpu-svo-traversal";

export const SVO_COMPACT_NODE_WORDS = 5;
export const SVO_COMPACT_NODE_STRIDE_BYTES = SVO_COMPACT_NODE_WORDS * Uint32Array.BYTES_PER_ELEMENT;
export const SVO_CANONICAL_NODE_WORDS = 8;
export const SVO_COMPACT_LEVEL_MASK = 0x1f;
export const SVO_COMPACT_CHILD_MASK_SHIFT = 8;
export const SVO_COMPACT_TERMINAL_BIT = 1 << 16;

export interface SvoCompactHierarchy {
  /** Five u32 words per node. */
  nodes: Uint32Array<ArrayBuffer>;
  nodeCount: number;
  leafCount: number;
  sourceGeneration: number;
  residentBytes: number;
  canonicalNodeBytes: number;
}

export interface SvoCompactNode {
  mortonLow: number;
  mortonHigh: number;
  level: number;
  childMask: number;
  terminal: boolean;
  linkOrLeaf: number;
  flags: number;
}

function publishedCount(value: number | undefined, capacity: number, label: string): number {
  const result = value ?? capacity;
  if (!Number.isInteger(result) || result < 0 || result > capacity) {
    throw new RangeError(`${label} exceeds the canonical packed topology`);
  }
  return result;
}

/** Pack a validated canonical publication without mutating or replacing it. */
export function packSvoCompactHierarchy(
  topology: SvoPackedTopologyView,
  sourceGeneration = 0,
): SvoCompactHierarchy {
  if (topology.nodes.length % SVO_CANONICAL_NODE_WORDS !== 0 || topology.leaves.length % 4 !== 0) {
    throw new RangeError("Canonical SVO contains a partial node or leaf record");
  }
  if (!Number.isInteger(sourceGeneration) || sourceGeneration < 0 || sourceGeneration > 0xffff_ffff) {
    throw new RangeError("Compact SVO source generation must fit uint32");
  }
  if ((topology.overflowFlags ?? 0) !== 0) {
    throw new RangeError("Cannot derive a compact SVO from an overflowed canonical publication");
  }
  const nodeCapacity = topology.nodes.length / SVO_CANONICAL_NODE_WORDS;
  const leafCapacity = topology.leaves.length / 4;
  const nodeCount = publishedCount(topology.publishedNodeCount, nodeCapacity, "Published node count");
  const leafCount = publishedCount(topology.publishedLeafCount, leafCapacity, "Published leaf count");
  const nodes = new Uint32Array(nodeCount * SVO_COMPACT_NODE_WORDS);
  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex += 1) {
    const source = nodeIndex * SVO_CANONICAL_NODE_WORDS;
    const target = nodeIndex * SVO_COMPACT_NODE_WORDS;
    const level = topology.nodes[source + 2];
    const childMask = topology.nodes[source + 3] & 0xff;
    const firstChild = topology.nodes[source + 4];
    const childCount = topology.nodes[source + 5];
    const leafIndex = topology.nodes[source + 6];
    if (level > 21) throw new RangeError(`Canonical node ${nodeIndex} exceeds the supported SVO depth`);
    if (childCount !== countOneBits8(childMask)) {
      throw new RangeError(`Canonical node ${nodeIndex} child count does not match its mask`);
    }
    const terminal = leafIndex !== SPARSE_BRICK_INVALID_INDEX;
    if (terminal && leafIndex >= leafCount) throw new RangeError(`Canonical node ${nodeIndex} leaf is unpublished`);
    if (!terminal && childMask !== 0 && firstChild === SPARSE_BRICK_INVALID_INDEX) {
      throw new RangeError(`Canonical node ${nodeIndex} has children but no first-child index`);
    }
    nodes[target] = topology.nodes[source];
    nodes[target + 1] = topology.nodes[source + 1];
    nodes[target + 2] = (level | (childMask << SVO_COMPACT_CHILD_MASK_SHIFT)
      | (terminal ? SVO_COMPACT_TERMINAL_BIT : 0)) >>> 0;
    nodes[target + 3] = terminal ? leafIndex : firstChild;
    nodes[target + 4] = topology.nodes[source + 7];
  }
  return {
    nodes,
    nodeCount,
    leafCount,
    sourceGeneration: sourceGeneration >>> 0,
    residentBytes: nodes.byteLength,
    canonicalNodeBytes: nodeCount * SVO_CANONICAL_NODE_WORDS * Uint32Array.BYTES_PER_ELEMENT,
  };
}

export function decodeSvoCompactNode(nodes: Uint32Array, nodeIndex: number): SvoCompactNode {
  if (!Number.isInteger(nodeIndex) || nodeIndex < 0 || (nodeIndex + 1) * SVO_COMPACT_NODE_WORDS > nodes.length) {
    throw new RangeError("Compact SVO node index is outside the packed view");
  }
  const base = nodeIndex * SVO_COMPACT_NODE_WORDS;
  const packed = nodes[base + 2];
  return {
    mortonLow: nodes[base], mortonHigh: nodes[base + 1],
    level: packed & SVO_COMPACT_LEVEL_MASK,
    childMask: (packed >>> SVO_COMPACT_CHILD_MASK_SHIFT) & 0xff,
    terminal: (packed & SVO_COMPACT_TERMINAL_BIT) !== 0,
    linkOrLeaf: nodes[base + 3], flags: nodes[base + 4],
  };
}

function countOneBits8(value: number): number {
  let count = 0;
  for (let bits = value & 0xff; bits !== 0; bits >>>= 1) count += bits & 1;
  return count;
}

function popcountBefore(mask: number, octant: number): number {
  return countOneBits8(mask & ((1 << octant) - 1));
}

function decodeMorton(low: number, high: number, level: number): readonly [number, number, number] {
  const coordinate = [0, 0, 0];
  for (let bit = 0; bit < level; bit += 1) for (let axis = 0; axis < 3; axis += 1) {
    const addressBit = 3 * bit + axis;
    coordinate[axis] += (((addressBit < 32 ? low : high) >>> (addressBit < 32 ? addressBit : addressBit - 32)) & 1) * 2 ** bit;
  }
  return coordinate as unknown as readonly [number, number, number];
}

function boundsFor(node: SvoCompactNode, mapping: SvoWorldMapping): SvoAabb {
  const coordinate = decodeMorton(node.mortonLow, node.mortonHigh, node.level);
  const scale = 2 ** (mapping.maximumDepth - node.level) * mapping.brickSize;
  const minimum = coordinate.map((value, axis) => mapping.origin[axis] + value * scale * mapping.cellSize[axis]) as [number, number, number];
  const maximum = coordinate.map((value, axis) => mapping.origin[axis] + (value + 1) * scale * mapping.cellSize[axis]) as [number, number, number];
  return { minimum, maximum };
}

interface CompactStackEntry { nodeIndex: number; octant: number; tEnter: number; tExit: number; bounds: SvoAabb }

/** CPU differential oracle for the compact WGSL continuation path. */
export function traverseCompactSvo(
  ray: SvoRay,
  hierarchy: SvoCompactHierarchy,
  leaves: Uint32Array,
  mapping: SvoWorldMapping,
  options: SvoTraversalOptions = {},
): SvoTraversalResult {
  if (hierarchy.nodes.length !== hierarchy.nodeCount * SVO_COMPACT_NODE_WORDS || leaves.length % 4 !== 0) {
    return { status: "invalid-topology", visits: 0, reason: "Compact node or leaf extent is invalid" };
  }
  if (hierarchy.nodeCount === 0) return { status: "miss", visits: 0 };
  const maximumVisits = options.maxNodeVisits ?? 256;
  const stackCapacity = options.stackCapacity ?? 32;
  const rootIndex = options.rootNodeIndex ?? 0;
  if (!Number.isInteger(rootIndex) || rootIndex < 0 || rootIndex >= hierarchy.nodeCount) {
    return { status: "invalid-topology", visits: 0, reason: "Root node index is outside the compact topology" };
  }
  const root = decodeSvoCompactNode(hierarchy.nodes, rootIndex);
  const rootBounds = boundsFor(root, mapping);
  const rootInterval = intersectSvoRayAabb(ray, rootBounds);
  if (!rootInterval) return { status: "miss", visits: 0 };
  const stack: CompactStackEntry[] = [{ nodeIndex: rootIndex, octant: 0, ...rootInterval, bounds: rootBounds }];
  let visits = 0;
  while (stack.length > 0) {
    if (visits >= maximumVisits) return { status: "work-exhausted", visits };
    const current = stack.pop()!;
    if (current.nodeIndex >= hierarchy.nodeCount) return { status: "invalid-topology", visits, reason: "Compact child index is unpublished" };
    const node = decodeSvoCompactNode(hierarchy.nodes, current.nodeIndex);
    visits += 1;
    if (node.level > mapping.maximumDepth) return { status: "invalid-topology", visits, reason: "Compact node level exceeds maximum depth" };
    if (node.terminal) {
      const leafIndex = node.linkOrLeaf;
      if (leafIndex >= hierarchy.leafCount || leafIndex * 4 + 3 >= leaves.length || leaves[leafIndex * 4] !== current.nodeIndex) {
        return { status: "invalid-topology", visits, reason: "Compact leaf backlink is invalid" };
      }
      const coordinate = decodeMorton(node.mortonLow, node.mortonHigh, node.level);
      const hit: SvoLeafHit = { nodeIndex: current.nodeIndex, leafIndex, voxelOffset: leaves[leafIndex * 4 + 1],
        terminalKind: leaves[leafIndex * 4 + 2] as SvoLeafHit["terminalKind"],
        terminalIndex: leaves[leafIndex * 4 + 3], level: node.level, coordinate,
        bounds: boundsFor(node, mapping), tEnter: current.tEnter, tExit: current.tExit };
      return { status: "hit", visits, hit };
    }
    if (node.childMask === 0) continue;
    if (node.linkOrLeaf === SPARSE_BRICK_INVALID_INDEX) {
      return { status: "invalid-topology", visits, reason: "Compact internal node has no first child" };
    }
    const hits: CompactStackEntry[] = [];
    for (let octant = 0; octant < 8; octant += 1) {
      if ((node.childMask & (1 << octant)) === 0) continue;
      const childIndex = node.linkOrLeaf + popcountBefore(node.childMask, octant);
      if (childIndex >= hierarchy.nodeCount) return { status: "invalid-topology", visits, reason: "Compact child range is unpublished" };
      const child = decodeSvoCompactNode(hierarchy.nodes, childIndex);
      if (child.level !== node.level + 1) return { status: "invalid-topology", visits, reason: "Compact child level is invalid" };
      const bounds = boundsFor(child, mapping);
      const interval = intersectSvoRayAabb(ray, bounds);
      if (interval) hits.push({ nodeIndex: childIndex, octant, ...interval, bounds });
    }
    hits.sort((a, b) => a.tEnter - b.tEnter || a.tExit - b.tExit || a.octant - b.octant);
    if (stack.length + hits.length > stackCapacity) return { status: "stack-overflow", visits };
    for (let index = hits.length - 1; index >= 0; index -= 1) stack.push(hits[index]);
  }
  return { status: "miss", visits };
}

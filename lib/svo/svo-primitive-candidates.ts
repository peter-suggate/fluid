import type { Quaternion, Vec3 } from "../core/model";
import {
  SVO_PRIMITIVE_RECORD_STRIDE_BYTES,
  SVO_PRIMITIVE_RECORD_WORDS,
  canonicalSvoPrimitive,
  intersectSvoPrimitive,
  svoPrimitiveLocalExtent_m,
  type SvoFinitePrimitiveDescriptor,
  type SvoPrimitiveRay,
  type SvoPrimitiveRayHit,
} from "./svo-primitive-abi";

/** Nodes intentionally share the 64-byte primitive-record stride and binding. */
export const SVO_PRIMITIVE_CANDIDATE_VERSION = 1;
/**
 * The live scene contract, not a shipped-catalog observation. Render updates
 * must never fall off the acceleration structure merely because an editor
 * grows the scene beyond today's authored examples.
 */
/**
 * Raised from 4 096 for the 10x acceptance scene (`hero-garden-hose-x10`,
 * `docs/svo-raster-visibility-handoff.md` W0). The hero publishes 501 records
 * and its 10x densification publishes 5 039, so 4 096 stopped that scene
 * drawing at all — loudly, from `buildSvoPrimitiveCandidates` below.
 *
 * **What 16 384 costs, in bytes.** The arena is
 * `(leaves + nodes) x 64 B` with `nodes = 2 x leaves - 1`, so it is linear:
 * 4 096 leaves reserved 786 368 B (0.75 MiB) and 16 384 reserve 3 145 664 B
 * (3.00 MiB). That allocation is per dry-scene renderer, once, as one region of
 * `SVO_DRY_SCENE_ARENA_LAYOUT`; only the records a scene actually publishes are
 * ever uploaded into it. Against the ~190 MB per-pixel coverage buffer the same
 * program is sizing (handoff §8), 2.25 MiB is not the memory worth arguing
 * about — but it is linear in this constant, so the next raise costs the same
 * again and the number belongs beside it.
 *
 * Four times the hero's 10x need rather than exactly it: the surrounding
 * capacities are cliffs, and a ceiling that a scene sits just under is a ceiling
 * someone hits while authoring.
 */
export const SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES = 16_384;
export const SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES = 2 * SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES - 1;
/**
 * Balanced median construction needs ceil(log2(leaves)) + the current node.
 *
 * Derived rather than written down, because it stops being right the moment the
 * leaf count moves and the failure is not loud: the shader's traversal drops
 * children it has no stack room for (`stackSize+2u<=MAX`, the WGSL in
 * `lib/webgpu-svo-dry-scene.ts`), so an undersized stack is *missing geometry*
 * rather than an error. The three-entry margin is the one 16 carried against
 * 4 096's twelve levels, kept so the value is unchanged in spirit; at 16 384 it
 * resolves to 18, which is two more u32 of shader stack per traversal.
 */
export const SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK =
  Math.ceil(Math.log2(SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES)) + 1 + 3;
export const SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL = 0xffff_ffff;

export interface SvoPrimitiveCandidateBounds {
  readonly minimum_m: Readonly<Vec3>;
  readonly maximum_m: Readonly<Vec3>;
}

export interface SvoPrimitiveCandidateNode extends SvoPrimitiveCandidateBounds {
  /** Internal nodes reference two node indices. Leaves store a primitive index in `leftOrPrimitiveIndex`. */
  readonly leftOrPrimitiveIndex: number;
  readonly rightChildIndex: number;
}

export interface SvoPrimitiveCandidatePublication {
  readonly version: typeof SVO_PRIMITIVE_CANDIDATE_VERSION;
  readonly primitiveCount: number;
  readonly rootNodeIndex: number;
  readonly nodes: readonly SvoPrimitiveCandidateNode[];
  readonly packedRecords: Uint32Array<ArrayBuffer>;
  readonly cacheKey: string;
}

export interface SvoPrimitiveCandidateArena {
  readonly primitiveCount: number;
  readonly candidateRecordOffset: number;
  readonly candidateNodeCount: number;
  readonly candidateRootNodeIndex: number;
  readonly candidateVersion: typeof SVO_PRIMITIVE_CANDIDATE_VERSION;
  readonly packedRecords: Uint32Array<ArrayBuffer>;
  readonly cacheKey: string;
}

/**
 * Fixed GPU arena capacity: live primitives followed by their complete BVH.
 *
 * 3 145 664 B at 16 384 leaves, against 786 368 B at the 4 096 this held before
 * W0. Linear in the leaf count — see the note on
 * `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES` — and consumed by
 * `SVO_DRY_SCENE_ARENA_LAYOUT`, which is where the buffer is actually created.
 */
export const SVO_PRIMITIVE_CANDIDATE_ARENA_RECORD_CAPACITY =
  SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES + SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES;
export const SVO_PRIMITIVE_CANDIDATE_ARENA_SIZE_BYTES =
  SVO_PRIMITIVE_CANDIDATE_ARENA_RECORD_CAPACITY * SVO_PRIMITIVE_RECORD_STRIDE_BYTES;

function worldExtents(local: Vec3, orientation: Quaternion): Vec3 {
  const { w, x, y, z } = orientation;
  const rows = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
  return {
    x: Math.abs(rows[0][0]) * local.x + Math.abs(rows[0][1]) * local.y + Math.abs(rows[0][2]) * local.z,
    y: Math.abs(rows[1][0]) * local.x + Math.abs(rows[1][1]) * local.y + Math.abs(rows[1][2]) * local.z,
    z: Math.abs(rows[2][0]) * local.x + Math.abs(rows[2][1]) * local.y + Math.abs(rows[2][2]) * local.z,
  };
}

export function svoPrimitiveCandidateBounds(descriptorInput: SvoFinitePrimitiveDescriptor): SvoPrimitiveCandidateBounds {
  const descriptor = canonicalSvoPrimitive(descriptorInput) as SvoFinitePrimitiveDescriptor;
  // One formula, in the kind table, rather than the sixth copy of it. A leaf
  // bound that is too small does not fail loudly: the BVH simply stops offering
  // the primitive to rays that do hit it, and the shape loses an edge.
  const local = svoPrimitiveLocalExtent_m(descriptor);
  const orientation = descriptor.kind === "sphere" ? { w: 1, x: 0, y: 0, z: 0 } : descriptor.orientation!;
  const extent = worldExtents(local, orientation);
  // Float32 upload plus a small absolute pad keeps tangencies and subcell props
  // conservative without changing their exact intersection or normal.
  const padding = Math.max(1e-6, Math.max(extent.x, extent.y, extent.z) * 1e-6);
  return Object.freeze({
    minimum_m: Object.freeze({
      x: descriptor.center_m.x - extent.x - padding,
      y: descriptor.center_m.y - extent.y - padding,
      z: descriptor.center_m.z - extent.z - padding,
    }),
    maximum_m: Object.freeze({
      x: descriptor.center_m.x + extent.x + padding,
      y: descriptor.center_m.y + extent.y + padding,
      z: descriptor.center_m.z + extent.z + padding,
    }),
  });
}

function union(entries: readonly SvoPrimitiveCandidateBounds[]): SvoPrimitiveCandidateBounds {
  return {
    minimum_m: {
      x: Math.min(...entries.map(({ minimum_m }) => minimum_m.x)),
      y: Math.min(...entries.map(({ minimum_m }) => minimum_m.y)),
      z: Math.min(...entries.map(({ minimum_m }) => minimum_m.z)),
    },
    maximum_m: {
      x: Math.max(...entries.map(({ maximum_m }) => maximum_m.x)),
      y: Math.max(...entries.map(({ maximum_m }) => maximum_m.y)),
      z: Math.max(...entries.map(({ maximum_m }) => maximum_m.z)),
    },
  };
}

function packNodes(nodes: readonly SvoPrimitiveCandidateNode[]): Uint32Array<ArrayBuffer> {
  const buffer = new ArrayBuffer(nodes.length * SVO_PRIMITIVE_RECORD_STRIDE_BYTES);
  const words = new Uint32Array(buffer), floats = new Float32Array(buffer);
  nodes.forEach((node, index) => writePackedNode(words, floats, index, node));
  return words;
}

function writePackedNode(
  words: Uint32Array,
  floats: Float32Array,
  index: number,
  node: SvoPrimitiveCandidateNode,
): void {
  const base = index * SVO_PRIMITIVE_RECORD_WORDS;
  floats.set([node.minimum_m.x, node.minimum_m.y, node.minimum_m.z], base);
  words[base + 3] = node.leftOrPrimitiveIndex;
  floats.set([node.maximum_m.x, node.maximum_m.y, node.maximum_m.z], base + 4);
  words[base + 7] = node.rightChildIndex;
}

function hashPacked(words: Uint32Array): string {
  let hash = 0x811c9dc5;
  for (const word of words) for (const byte of [word & 0xff, (word >>> 8) & 0xff, (word >>> 16) & 0xff, word >>> 24]) {
    hash = Math.imul((hash ^ byte) >>> 0, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

export function buildSvoPrimitiveCandidates(
  descriptors: readonly SvoFinitePrimitiveDescriptor[],
  options: { readonly skippedOwnerId?: number } = {},
): SvoPrimitiveCandidatePublication {
  if (descriptors.length < 1 || descriptors.length > SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES) {
    throw new RangeError(`SVO primitive candidate index needs 1-${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES} primitives`);
  }
  const entries = descriptors.map((descriptor, primitiveIndex) => ({
    primitiveIndex,
    descriptor: canonicalSvoPrimitive(descriptor) as SvoFinitePrimitiveDescriptor,
    bounds: svoPrimitiveCandidateBounds(descriptor),
  })).filter(({ descriptor }) => descriptor.ownerId !== options.skippedOwnerId);
  if (entries.length === 0) throw new Error("SVO primitive candidate index cannot omit every primitive");
  const nodes: SvoPrimitiveCandidateNode[] = [];
  const build = (subset: typeof entries): number => {
    const nodeIndex = nodes.length;
    nodes.push(undefined as unknown as SvoPrimitiveCandidateNode);
    const bounds = union(subset.map(({ bounds: entryBounds }) => entryBounds));
    if (subset.length === 1) {
      nodes[nodeIndex] = Object.freeze({ ...bounds, leftOrPrimitiveIndex: subset[0].primitiveIndex, rightChildIndex: SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL });
      return nodeIndex;
    }
    const extent = {
      x: bounds.maximum_m.x - bounds.minimum_m.x,
      y: bounds.maximum_m.y - bounds.minimum_m.y,
      z: bounds.maximum_m.z - bounds.minimum_m.z,
    };
    const axis: keyof Vec3 = extent.x >= extent.y && extent.x >= extent.z ? "x" : extent.y >= extent.z ? "y" : "z";
    const sorted = [...subset].sort((left, right) => {
      const leftCenter = left.bounds.minimum_m[axis] + left.bounds.maximum_m[axis];
      const rightCenter = right.bounds.minimum_m[axis] + right.bounds.maximum_m[axis];
      return leftCenter - rightCenter || left.primitiveIndex - right.primitiveIndex;
    });
    const middle = Math.floor(sorted.length / 2);
    const leftChild = build(sorted.slice(0, middle));
    const rightChild = build(sorted.slice(middle));
    nodes[nodeIndex] = Object.freeze({ ...bounds, leftOrPrimitiveIndex: leftChild, rightChildIndex: rightChild });
    return nodeIndex;
  };
  const rootNodeIndex = build(entries);
  if (nodes.length > SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES) throw new Error("SVO primitive candidate node bound exceeded");
  const packedRecords = packNodes(nodes);
  return Object.freeze({
    version: SVO_PRIMITIVE_CANDIDATE_VERSION,
    primitiveCount: descriptors.length,
    rootNodeIndex,
    nodes: Object.freeze(nodes),
    packedRecords,
    cacheKey: `svo-primitive-candidates-v${SVO_PRIMITIVE_CANDIDATE_VERSION}:${hashPacked(packedRecords)}`,
  });
}

/** Linear-time live refit preserving the publication's balanced topology. */
export function refitSvoPrimitiveCandidates(
  descriptors: readonly SvoFinitePrimitiveDescriptor[],
  publication: SvoPrimitiveCandidatePublication,
): SvoPrimitiveCandidatePublication {
  validateCandidatePublication(publication);
  if (descriptors.length !== publication.primitiveCount) throw new Error("SVO primitive candidate refit count mismatch");
  const nodes = new Array<SvoPrimitiveCandidateNode>(publication.nodes.length);
  const refit = (nodeIndex: number): SvoPrimitiveCandidateBounds => {
    const node = publication.nodes[nodeIndex];
    let bounds: SvoPrimitiveCandidateBounds;
    if (node.rightChildIndex === SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL) {
      const descriptor = descriptors[node.leftOrPrimitiveIndex];
      if (!descriptor) throw new Error("SVO primitive candidate refit leaf is invalid");
      bounds = svoPrimitiveCandidateBounds(descriptor);
    } else {
      bounds = union([refit(node.leftOrPrimitiveIndex), refit(node.rightChildIndex)]);
    }
    nodes[nodeIndex] = Object.freeze({
      ...bounds,
      leftOrPrimitiveIndex: node.leftOrPrimitiveIndex,
      rightChildIndex: node.rightChildIndex,
    });
    return bounds;
  };
  refit(publication.rootNodeIndex);
  const packedRecords = packNodes(nodes);
  return Object.freeze({
    version: SVO_PRIMITIVE_CANDIDATE_VERSION,
    primitiveCount: publication.primitiveCount,
    rootNodeIndex: publication.rootNodeIndex,
    nodes: Object.freeze(nodes),
    packedRecords,
    cacheKey: `svo-primitive-candidates-v${SVO_PRIMITIVE_CANDIDATE_VERSION}:refit-${hashPacked(packedRecords)}`,
  });
}

/** Topology-only state built once for O(dirty * depth) live refits. */
export interface SvoPrimitiveCandidateRefitPlan {
  readonly publication: SvoPrimitiveCandidatePublication;
  readonly parentByNode: Int32Array<ArrayBuffer>;
  readonly depthByNode: Uint16Array<ArrayBuffer>;
  readonly leafByPrimitive: Int32Array<ArrayBuffer>;
}

/** Validate once, then detach mutable node/packing mirrors for animation. */
export function createSvoPrimitiveCandidateRefitPlan(
  source: SvoPrimitiveCandidatePublication,
): SvoPrimitiveCandidateRefitPlan {
  validateCandidatePublication(source);
  const nodes = source.nodes.map((node) => ({
    minimum_m: { ...node.minimum_m }, maximum_m: { ...node.maximum_m },
    leftOrPrimitiveIndex: node.leftOrPrimitiveIndex,
    rightChildIndex: node.rightChildIndex,
  }));
  const parentByNode = new Int32Array(nodes.length); parentByNode.fill(-1);
  const depthByNode = new Uint16Array(nodes.length);
  const leafByPrimitive = new Int32Array(source.primitiveCount); leafByPrimitive.fill(-1);
  const stack: { index: number; depth: number }[] = [{ index: source.rootNodeIndex, depth: 0 }];
  while (stack.length) {
    const { index, depth } = stack.pop()!;
    depthByNode[index] = depth;
    const node = nodes[index];
    if (node.rightChildIndex === SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL) {
      leafByPrimitive[node.leftOrPrimitiveIndex] = index;
    } else {
      parentByNode[node.leftOrPrimitiveIndex] = index;
      parentByNode[node.rightChildIndex] = index;
      stack.push({ index: node.leftOrPrimitiveIndex, depth: depth + 1 },
        { index: node.rightChildIndex, depth: depth + 1 });
    }
  }
  const publication: SvoPrimitiveCandidatePublication = Object.freeze({
    version: source.version,
    primitiveCount: source.primitiveCount,
    rootNodeIndex: source.rootNodeIndex,
    nodes,
    packedRecords: new Uint32Array(source.packedRecords),
    // A live refit intentionally has no per-frame content hash.
    cacheKey: `${source.cacheKey}:incremental`,
  });
  return Object.freeze({ publication, parentByNode, depthByNode, leafByPrimitive });
}

/** Refit dirty leaves and their ancestor closure, updating only their packed records. */
export function refitSvoPrimitiveCandidatesIncremental(
  descriptors: readonly SvoFinitePrimitiveDescriptor[],
  plan: SvoPrimitiveCandidateRefitPlan,
  dirtyPrimitiveIndices: readonly number[],
): { readonly publication: SvoPrimitiveCandidatePublication; readonly dirtyNodeIndices: readonly number[] } {
  const publication = plan.publication;
  if (descriptors.length !== publication.primitiveCount) throw new Error("SVO primitive candidate refit count mismatch");
  const dirtyNodes = new Set<number>();
  for (const primitiveIndex of dirtyPrimitiveIndices) {
    if (!Number.isSafeInteger(primitiveIndex) || primitiveIndex < 0 || primitiveIndex >= descriptors.length) {
      throw new RangeError("SVO primitive candidate dirty index is invalid");
    }
    let nodeIndex = plan.leafByPrimitive[primitiveIndex];
    if (nodeIndex < 0) continue; // Deliberately omitted owner (for example an open shell).
    while (nodeIndex >= 0 && !dirtyNodes.has(nodeIndex)) {
      dirtyNodes.add(nodeIndex);
      nodeIndex = plan.parentByNode[nodeIndex];
    }
  }
  const ordered = [...dirtyNodes].sort((left, right) => plan.depthByNode[right] - plan.depthByNode[left]);
  const nodes = publication.nodes as SvoPrimitiveCandidateNode[];
  const words = publication.packedRecords;
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  for (const nodeIndex of ordered) {
    const topology = nodes[nodeIndex];
    const bounds = topology.rightChildIndex === SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL
      ? svoPrimitiveCandidateBounds(descriptors[topology.leftOrPrimitiveIndex])
      : union([nodes[topology.leftOrPrimitiveIndex], nodes[topology.rightChildIndex]]);
    const node = {
      ...bounds,
      leftOrPrimitiveIndex: topology.leftOrPrimitiveIndex,
      rightChildIndex: topology.rightChildIndex,
    };
    nodes[nodeIndex] = node;
    writePackedNode(words, floats, nodeIndex, node);
  }
  return { publication, dirtyNodeIndices: ordered };
}

function validateCandidatePublication(publication: SvoPrimitiveCandidatePublication): void {
  if (publication.version !== SVO_PRIMITIVE_CANDIDATE_VERSION) throw new Error("SVO primitive candidate version mismatch");
  if (!Number.isSafeInteger(publication.primitiveCount) || publication.primitiveCount < 1
    || publication.primitiveCount > SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES) throw new Error("SVO primitive candidate count is invalid");
  if (publication.nodes.length < 1 || publication.nodes.length > SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES
    || publication.packedRecords.byteLength !== publication.nodes.length * SVO_PRIMITIVE_RECORD_STRIDE_BYTES) {
    throw new Error("SVO primitive candidate node publication is incomplete");
  }
  const visited = new Set<number>(), primitiveIndices = new Set<number>(), stack = [publication.rootNodeIndex];
  while (stack.length > 0) {
    const nodeIndex = stack.pop()!;
    if (!Number.isSafeInteger(nodeIndex) || nodeIndex < 0 || nodeIndex >= publication.nodes.length || visited.has(nodeIndex)) {
      throw new Error("SVO primitive candidate tree is invalid or cyclic");
    }
    visited.add(nodeIndex);
    const node = publication.nodes[nodeIndex];
    if (![node.minimum_m.x, node.minimum_m.y, node.minimum_m.z, node.maximum_m.x, node.maximum_m.y, node.maximum_m.z].every(Number.isFinite)
      || node.minimum_m.x > node.maximum_m.x || node.minimum_m.y > node.maximum_m.y || node.minimum_m.z > node.maximum_m.z) {
      throw new Error("SVO primitive candidate bounds are invalid");
    }
    if (node.rightChildIndex === SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL) {
      if (!Number.isSafeInteger(node.leftOrPrimitiveIndex) || node.leftOrPrimitiveIndex < 0
        || node.leftOrPrimitiveIndex >= publication.primitiveCount || primitiveIndices.has(node.leftOrPrimitiveIndex)) {
        throw new Error("SVO primitive candidate leaf is invalid or duplicated");
      }
      primitiveIndices.add(node.leftOrPrimitiveIndex);
    } else stack.push(node.leftOrPrimitiveIndex, node.rightChildIndex);
  }
  if (visited.size !== publication.nodes.length) throw new Error("SVO primitive candidate publication contains unreachable nodes");
  const canonicalPackedRecords = packNodes(publication.nodes);
  if (!canonicalPackedRecords.every((word, index) => word === publication.packedRecords[index])) {
    throw new Error("SVO primitive candidate packed records do not match their validated nodes");
  }
}

/** Append candidate records after primitives so no additional GPU binding is required. */
export function packSvoPrimitiveCandidateArena(
  primitiveRecords: Uint32Array<ArrayBuffer>,
  publication: SvoPrimitiveCandidatePublication,
): SvoPrimitiveCandidateArena {
  validateCandidatePublication(publication);
  if (primitiveRecords.byteLength !== publication.primitiveCount * SVO_PRIMITIVE_RECORD_STRIDE_BYTES) {
    throw new Error("SVO primitive candidate publication does not match primitive records");
  }
  const packedRecords = new Uint32Array(new ArrayBuffer(primitiveRecords.byteLength + publication.packedRecords.byteLength));
  packedRecords.set(primitiveRecords);
  packedRecords.set(publication.packedRecords, primitiveRecords.length);
  return Object.freeze({
    primitiveCount: publication.primitiveCount,
    candidateRecordOffset: publication.primitiveCount,
    candidateNodeCount: publication.nodes.length,
    candidateRootNodeIndex: publication.rootNodeIndex,
    candidateVersion: publication.version,
    packedRecords,
    cacheKey: `${publication.cacheKey}:arena-${primitiveRecords.byteLength.toString(16)}`,
  });
}

function rayBounds(ray: SvoPrimitiveRay, bounds: SvoPrimitiveCandidateBounds): number | undefined {
  const tMin = ray.tMin_m ?? 0, tMax = ray.tMax_m ?? Number.POSITIVE_INFINITY;
  let near = tMin, far = tMax;
  for (const axis of ["x", "y", "z"] as const) {
    const origin = ray.origin_m[axis], direction = ray.direction[axis];
    if (!Number.isFinite(origin) || !Number.isFinite(direction)) throw new RangeError("Candidate ray must be finite");
    if (Math.abs(direction) <= 1e-15) {
      if (origin < bounds.minimum_m[axis] || origin > bounds.maximum_m[axis]) return undefined;
      continue;
    }
    const a = (bounds.minimum_m[axis] - origin) / direction;
    const b = (bounds.maximum_m[axis] - origin) / direction;
    near = Math.max(near, Math.min(a, b));
    far = Math.min(far, Math.max(a, b));
    if (near > far) return undefined;
  }
  return near;
}

/** CPU oracle returning only conservative candidates; exact intersections remain a second phase. */
export function querySvoPrimitiveCandidates(
  publication: SvoPrimitiveCandidatePublication,
  ray: SvoPrimitiveRay,
): Readonly<{ primitiveIndices: readonly number[]; nodeVisits: number; maximumStackDepth: number }> {
  const stack = [publication.rootNodeIndex], candidates: number[] = [];
  let nodeVisits = 0, maximumStackDepth = 1;
  while (stack.length > 0) {
    if (nodeVisits >= SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES) throw new Error("SVO primitive candidate oracle exhausted node bound");
    const nodeIndex = stack.pop()!;
    const node = publication.nodes[nodeIndex];
    if (!node) throw new Error("SVO primitive candidate oracle encountered an invalid node");
    nodeVisits += 1;
    if (rayBounds(ray, node) === undefined) continue;
    if (node.rightChildIndex === SVO_PRIMITIVE_CANDIDATE_LEAF_SENTINEL) candidates.push(node.leftOrPrimitiveIndex);
    else {
      const left = publication.nodes[node.leftOrPrimitiveIndex], right = publication.nodes[node.rightChildIndex];
      if (!left || !right) throw new Error("SVO primitive candidate oracle encountered an invalid child");
      const leftNear = rayBounds(ray, left), rightNear = rayBounds(ray, right);
      // Push farther first so nearer candidates are exact-tested first. The
      // primitive index tie-break remains explicit in the consumer.
      if (leftNear !== undefined && rightNear !== undefined) {
        if (leftNear <= rightNear) stack.push(node.rightChildIndex, node.leftOrPrimitiveIndex);
        else stack.push(node.leftOrPrimitiveIndex, node.rightChildIndex);
      } else if (leftNear !== undefined) stack.push(node.leftOrPrimitiveIndex);
      else if (rightNear !== undefined) stack.push(node.rightChildIndex);
      maximumStackDepth = Math.max(maximumStackDepth, stack.length);
      if (stack.length > SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK) throw new Error("SVO primitive candidate oracle exceeded stack bound");
    }
  }
  return Object.freeze({ primitiveIndices: Object.freeze(candidates), nodeVisits, maximumStackDepth });
}

/** Exact second phase used as the CPU oracle for primary/picking determinism. */
export function traceSvoPrimitiveCandidates(
  publication: SvoPrimitiveCandidatePublication,
  descriptors: readonly SvoFinitePrimitiveDescriptor[],
  ray: SvoPrimitiveRay,
): Readonly<{
  hit: SvoPrimitiveRayHit | null;
  primitiveIndex: number | null;
  nodeVisits: number;
  candidateIntersections: number;
}> {
  if (descriptors.length !== publication.primitiveCount) throw new Error("SVO primitive candidate descriptor count mismatch");
  const query = querySvoPrimitiveCandidates(publication, ray);
  let hit: SvoPrimitiveRayHit | null = null, primitiveIndex: number | null = null;
  for (const candidateIndex of query.primitiveIndices) {
    const candidate = descriptors[candidateIndex] ? intersectSvoPrimitive(descriptors[candidateIndex], ray) : null;
    if (!candidate) continue;
    const tolerance = 1e-6 * Math.max(1, candidate.t_m, hit?.t_m ?? 0);
    if (!hit || candidate.t_m < hit.t_m - tolerance
      || (Math.abs(candidate.t_m - hit.t_m) <= tolerance && candidateIndex < primitiveIndex!)) {
      hit = candidate;
      primitiveIndex = candidateIndex;
    }
  }
  return Object.freeze({
    hit,
    primitiveIndex,
    nodeVisits: query.nodeVisits,
    candidateIntersections: query.primitiveIndices.length,
  });
}

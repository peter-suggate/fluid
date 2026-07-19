import type { Quaternion, Vec3 } from "./model";
import {
  SVO_PRIMITIVE_RECORD_STRIDE_BYTES,
  SVO_PRIMITIVE_RECORD_WORDS,
  canonicalSvoPrimitive,
  intersectSvoPrimitive,
  type SvoFinitePrimitiveDescriptor,
  type SvoPrimitiveRay,
  type SvoPrimitiveRayHit,
} from "./svo-primitive-abi";

/** Nodes intentionally share the 64-byte primitive-record stride and binding. */
export const SVO_PRIMITIVE_CANDIDATE_VERSION = 1;
export const SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES = 64;
export const SVO_PRIMITIVE_CANDIDATE_MAXIMUM_NODES = 2 * SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES - 1;
export const SVO_PRIMITIVE_CANDIDATE_MAXIMUM_STACK = 16;
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
  let local: Vec3;
  if (descriptor.kind === "sphere") local = { x: descriptor.radius_m, y: descriptor.radius_m, z: descriptor.radius_m };
  else if (descriptor.kind === "box") local = descriptor.halfExtents_m;
  else if (descriptor.kind === "capsule") local = { x: descriptor.radius_m, y: descriptor.segmentHalfLength_m + descriptor.radius_m, z: descriptor.radius_m };
  else if (descriptor.kind === "cylinder") local = { x: descriptor.radius_m, y: descriptor.halfHeight_m, z: descriptor.radius_m };
  else local = descriptor.radii_m;
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
  nodes.forEach((node, index) => {
    const base = index * SVO_PRIMITIVE_RECORD_WORDS;
    floats.set([node.minimum_m.x, node.minimum_m.y, node.minimum_m.z], base);
    words[base + 3] = node.leftOrPrimitiveIndex;
    floats.set([node.maximum_m.x, node.maximum_m.y, node.maximum_m.z], base + 4);
    words[base + 7] = node.rightChildIndex;
  });
  return words;
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

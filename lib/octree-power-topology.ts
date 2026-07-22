/** CPU topology audit and cube-symmetry helpers for power catalog generation. */

import type { PowerVec3 } from "./octree-power-geometry";

/** @deprecated Every graded same/coarser mask now has a strict acute catalog triangulation. */
export const OCTREE_POWER_STRICTLY_OBTUSE_COARSE_MASKS: readonly number[] = Object.freeze([]);

export function octreePowerCoarseMaskNeedsAcuteRepair(mask: number): boolean {
  return false;
}

export interface OctreeTopologyLeaf {
  readonly key: string;
  readonly origin: PowerVec3;
  readonly size: number;
}

export type OctreeNeighborKind = "face" | "edge";

export interface OctreeTopologyNeighbor {
  readonly a: number;
  readonly b: number;
  readonly kind: OctreeNeighborKind;
  readonly levelDifference: number;
}

export type OctreeGradingCase = "same-only" | "same-finer" | "same-coarser" | "mixed";

export interface OctreeTopologyLeafAudit {
  readonly key: string;
  readonly size: number;
  readonly gradingCase: OctreeGradingCase;
  readonly faceNeighbors: number;
  readonly edgeNeighbors: number;
  readonly maximumNeighborLevelDifference: number;
}

export interface OctreePowerTopologyAudit {
  readonly liveLeafCount: number;
  readonly sameOnlyLeaves: number;
  readonly sameOrFinerLeaves: number;
  readonly sameOrCoarserLeaves: number;
  readonly mixedFinerAndCoarserLeaves: number;
  readonly strictlyObtuseSameOrCoarserLeaves: number;
  /** Coarse face leaves that must split to satisfy the Section 5 simplex gate. */
  readonly acuteRepairCoarseLeaves: readonly string[];
  readonly maximumFaceNeighborLevelDifference: number;
  readonly maximumEdgeNeighborLevelDifference: number;
  readonly countsBySize: Readonly<Record<string, number>>;
  readonly paperCompatible: boolean;
  readonly ordinaryTwoToOne: boolean;
  readonly neighbors: readonly OctreeTopologyNeighbor[];
  readonly leaves: readonly OctreeTopologyLeafAudit[];
}

export interface OctreePaperGradingResult {
  readonly leaves: readonly OctreeTopologyLeaf[];
  readonly audit: OctreePowerTopologyAudit;
  readonly iterations: number;
  readonly refinedParents: number;
  readonly leafIncreaseFraction: number;
  readonly accepted: boolean;
  readonly rejectionReason?: string;
}

function validateLeaf(leaf: OctreeTopologyLeaf): void {
  if (!leaf.key || !Number.isSafeInteger(leaf.size) || leaf.size < 1 || (leaf.size & (leaf.size - 1)) !== 0) {
    throw new RangeError("Topology leaves need a stable key and positive dyadic size");
  }
  if (leaf.origin.length !== 3 || leaf.origin.some((value) => !Number.isSafeInteger(value))) {
    throw new RangeError(`Topology leaf ${leaf.key} origin must contain safe integers`);
  }
  if (leaf.origin.some((value) => value % leaf.size !== 0)) {
    throw new RangeError(`Topology leaf ${leaf.key} is not aligned to its dyadic size`);
  }
}

function contactKind(a: OctreeTopologyLeaf, b: OctreeTopologyLeaf): OctreeNeighborKind | undefined {
  let touching = 0;
  let overlapping = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const low = Math.max(a.origin[axis], b.origin[axis]);
    const high = Math.min(a.origin[axis] + a.size, b.origin[axis] + b.size);
    if (high < low) return undefined;
    if (high === low) touching += 1;
    else overlapping += 1;
  }
  if (touching === 1 && overlapping === 2) return "face";
  if (touching === 2 && overlapping === 1) return "edge";
  return undefined;
}

/** Read-only audit required before selecting the paper-compatible catalog. */
export function auditOctreePowerTopology(input: readonly OctreeTopologyLeaf[]): OctreePowerTopologyAudit {
  if (input.length === 0) throw new RangeError("Topology audit requires at least one live leaf");
  const leaves = [...input].sort((a, b) => a.key.localeCompare(b.key));
  const keys = new Set<string>();
  for (const leaf of leaves) {
    validateLeaf(leaf);
    if (keys.has(leaf.key)) throw new RangeError(`Duplicate topology leaf key ${leaf.key}`);
    keys.add(leaf.key);
  }
  // An octree frontier is a disjoint tiling. Touching is valid; positive
  // overlap on all axes means corrupt owner publication and is never audited
  // as an innocent neighbor.
  for (let i = 0; i < leaves.length - 1; i += 1) for (let j = i + 1; j < leaves.length; j += 1) {
    if ([0, 1, 2].every((axis) => Math.min(leaves[i].origin[axis] + leaves[i].size, leaves[j].origin[axis] + leaves[j].size)
      > Math.max(leaves[i].origin[axis], leaves[j].origin[axis]))) {
      throw new Error(`Topology leaves ${leaves[i].key} and ${leaves[j].key} overlap`);
    }
  }

  const neighbors: OctreeTopologyNeighbor[] = [];
  const adjacency = leaves.map(() => [] as OctreeTopologyNeighbor[]);
  let maximumFaceNeighborLevelDifference = 0;
  let maximumEdgeNeighborLevelDifference = 0;
  for (let a = 0; a < leaves.length - 1; a += 1) for (let b = a + 1; b < leaves.length; b += 1) {
    const kind = contactKind(leaves[a], leaves[b]);
    if (!kind) continue;
    const levelDifference = Math.abs(Math.log2(leaves[a].size) - Math.log2(leaves[b].size));
    const neighbor = { a, b, kind, levelDifference } as const;
    neighbors.push(neighbor); adjacency[a].push(neighbor); adjacency[b].push(neighbor);
    if (kind === "face") maximumFaceNeighborLevelDifference = Math.max(maximumFaceNeighborLevelDifference, levelDifference);
    else maximumEdgeNeighborLevelDifference = Math.max(maximumEdgeNeighborLevelDifference, levelDifference);
  }
  const leafAudit = leaves.map((leaf, index): OctreeTopologyLeafAudit => {
    let finer = false, coarser = false, maximumNeighborLevelDifference = 0, faceNeighbors = 0, edgeNeighbors = 0;
    for (const neighbor of adjacency[index]) {
      const other = leaves[neighbor.a === index ? neighbor.b : neighbor.a];
      finer ||= other.size < leaf.size;
      coarser ||= other.size > leaf.size;
      maximumNeighborLevelDifference = Math.max(maximumNeighborLevelDifference, neighbor.levelDifference);
      if (neighbor.kind === "face") faceNeighbors += 1; else edgeNeighbors += 1;
    }
    const gradingCase: OctreeGradingCase = finer && coarser ? "mixed"
      : finer ? "same-finer" : coarser ? "same-coarser" : "same-only";
    return { key: leaf.key, size: leaf.size, gradingCase, faceNeighbors, edgeNeighbors, maximumNeighborLevelDifference };
  });
  const countsBySize: Record<string, number> = {};
  for (const leaf of leaves) countsBySize[String(leaf.size)] = (countsBySize[String(leaf.size)] ?? 0) + 1;
  const mixedFinerAndCoarserLeaves = leafAudit.filter((leaf) => leaf.gradingCase === "mixed").length;
  return {
    liveLeafCount: leaves.length,
    sameOnlyLeaves: leafAudit.filter((leaf) => leaf.gradingCase === "same-only").length,
    sameOrFinerLeaves: leafAudit.filter((leaf) => leaf.gradingCase === "same-finer" || leaf.gradingCase === "same-only").length,
    sameOrCoarserLeaves: leafAudit.filter((leaf) => leaf.gradingCase === "same-coarser" || leaf.gradingCase === "same-only").length,
    mixedFinerAndCoarserLeaves,
    strictlyObtuseSameOrCoarserLeaves: 0,
    acuteRepairCoarseLeaves: [],
    maximumFaceNeighborLevelDifference,
    maximumEdgeNeighborLevelDifference,
    countsBySize,
    paperCompatible: mixedFinerAndCoarserLeaves === 0,
    ordinaryTwoToOne: maximumFaceNeighborLevelDifference <= 1 && maximumEdgeNeighborLevelDifference <= 1,
    neighbors,
    leaves: leafAudit,
  };
}

/**
 * Deterministic CPU oracle for the paper's stronger local grading rule
 * (Section 6.1). For every mixed leaf, its coarser face/edge neighbors are
 * split once; the audit is repeated until every 1-ring is exclusively
 * same/finer or same/coarser. The caller may reject the proposed generation
 * without publishing it when the configured growth gate is exceeded.
 */
export function enforcePaperCompatibleOctreeGrading(
  input: readonly OctreeTopologyLeaf[],
  maximumLeafIncreaseFraction = 0.15,
  maximumIterations = 32,
): OctreePaperGradingResult {
  if (!Number.isFinite(maximumLeafIncreaseFraction) || maximumLeafIncreaseFraction < 0) {
    throw new RangeError("Paper grading leaf-growth gate must be non-negative");
  }
  if (!Number.isSafeInteger(maximumIterations) || maximumIterations < 1) {
    throw new RangeError("Paper grading maximum iterations must be a positive integer");
  }
  const initial = auditOctreePowerTopology(input);
  if (!initial.ordinaryTwoToOne) throw new Error("Paper grading requires an ordinary face/edge 2:1-balanced frontier");
  let leaves = [...input].map((leaf) => ({ ...leaf, origin: [...leaf.origin] as PowerVec3 }))
    .sort((a, b) => a.key.localeCompare(b.key));
  let audit = initial, iterations = 0, refinedParents = 0;
  const refineFrontier = (parents: ReadonlySet<string>): void => {
    const existingKeys = new Set(leaves.map((leaf) => leaf.key));
    const next: OctreeTopologyLeaf[] = [];
    for (const leaf of leaves) {
      if (!parents.has(leaf.key)) { next.push(leaf); continue; }
      if (leaf.size === 1) throw new Error(`Paper grading cannot refine unit leaf ${leaf.key}`);
      const childSize = leaf.size / 2;
      for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
        const key = `${leaf.key}/${x}${y}${z}`;
        if (existingKeys.has(key)) throw new Error(`Paper grading child key collision ${key}`);
        next.push({ key, size: childSize, origin: [
          leaf.origin[0] + x * childSize, leaf.origin[1] + y * childSize, leaf.origin[2] + z * childSize,
        ] });
      }
      refinedParents += 1;
    }
    leaves = next.sort((a, b) => a.key.localeCompare(b.key));
  };
  while (!audit.paperCompatible) {
    if (iterations >= maximumIterations) throw new Error("Paper grading did not stabilize within its bounded iteration budget");
    const refine = new Set<string>();
    for (const mixed of audit.leaves.filter((leaf) => leaf.gradingCase === "mixed")) {
      const anchorIndex = audit.leaves.findIndex((leaf) => leaf.key === mixed.key);
      for (const neighbor of audit.neighbors) {
        if (neighbor.a !== anchorIndex && neighbor.b !== anchorIndex) continue;
        const otherAudit = audit.leaves[neighbor.a === anchorIndex ? neighbor.b : neighbor.a];
        if (otherAudit.size > mixed.size) refine.add(otherAudit.key);
      }
    }
    if (refine.size === 0) throw new Error("Paper grading audit reported mixed neighborhoods without a refinable coarse neighbor");
    refineFrontier(refine);
    iterations += 1;
    audit = auditOctreePowerTopology(leaves);
    // Section 7 requires re-running ordinary face/edge 2:1 balancing after
    // every stronger-grading refinement. Refining a former coarse neighbor
    // can otherwise make its children two levels smaller than a leaf on the
    // opposite side.
    while (!audit.ordinaryTwoToOne) {
      if (iterations >= maximumIterations) throw new Error("Paper grading did not stabilize within its bounded iteration budget");
      const balance = new Set<string>();
      for (const neighbor of audit.neighbors.filter((candidate) => candidate.levelDifference > 1)) {
        const a = audit.leaves[neighbor.a], b = audit.leaves[neighbor.b];
        balance.add(a.size > b.size ? a.key : b.key);
      }
      if (balance.size === 0) throw new Error("Ordinary 2:1 audit failed without a refinable imbalance");
      refineFrontier(balance);
      iterations += 1;
      audit = auditOctreePowerTopology(leaves);
    }
  }
  const leafIncreaseFraction = (leaves.length - input.length) / input.length;
  const accepted = leafIncreaseFraction <= maximumLeafIncreaseFraction;
  return {
    leaves, audit, iterations, refinedParents, leafIncreaseFraction, accepted,
    ...(!accepted ? { rejectionReason: `leaf growth ${(100 * leafIncreaseFraction).toFixed(2)}% exceeds ${(100 * maximumLeafIncreaseFraction).toFixed(2)}%` } : {}),
  };
}

export interface CubeTransform {
  /** World component i reads source component permutation[i] with sign[i]. */
  readonly permutation: readonly [number, number, number];
  readonly sign: readonly [1 | -1, 1 | -1, 1 | -1];
  readonly determinant: 1 | -1;
  readonly code: number;
}

function permutationParity(p: readonly number[]): 1 | -1 {
  let inversions = 0;
  for (let i = 0; i < p.length; i += 1) for (let j = i + 1; j < p.length; j += 1) inversions += Number(p[i] > p[j]);
  return inversions % 2 === 0 ? 1 : -1;
}

/** All 24 rotations and 24 safe reflections in a stable packed order. */
export const OCTREE_CUBE_TRANSFORMS: readonly CubeTransform[] = (() => {
  const permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] as const;
  const result: CubeTransform[] = [];
  for (const permutation of permutations) for (let bits = 0; bits < 8; bits += 1) {
    const sign: [1 | -1, 1 | -1, 1 | -1] = [0, 1, 2].map((axis) => (bits & (1 << axis)) ? -1 : 1) as typeof sign;
    const determinant = (permutationParity(permutation) * sign[0] * sign[1] * sign[2]) as 1 | -1;
    result.push({ permutation, sign, determinant, code: result.length });
  }
  return Object.freeze(result);
})();

export function transformPowerVector(value: PowerVec3, transform: CubeTransform): PowerVec3 {
  return [0, 1, 2].map((axis) => transform.sign[axis] * value[transform.permutation[axis]]) as [number, number, number];
}

export function inverseCubeTransform(transform: CubeTransform): CubeTransform {
  const permutation: [number, number, number] = [0, 0, 0];
  const sign: [1 | -1, 1 | -1, 1 | -1] = [1, 1, 1];
  for (let axis = 0; axis < 3; axis += 1) {
    permutation[transform.permutation[axis]] = axis;
    sign[transform.permutation[axis]] = transform.sign[axis];
  }
  const found = OCTREE_CUBE_TRANSFORMS.find((candidate) => candidate.permutation.every((value, axis) => value === permutation[axis])
    && candidate.sign.every((value, axis) => value === sign[axis]));
  if (!found) throw new Error("Cube transform table is incomplete");
  return found;
}

/**
 * Returns the table transform equivalent to applying `first` and then
 * `second`. Keeping composition inside the finite 48-element group lets the
 * generated catalog resolve raw descriptors without storing matrices.
 */
export function composeCubeTransforms(first: CubeTransform, second: CubeTransform): CubeTransform {
  const permutation = [0, 1, 2].map((axis) => first.permutation[second.permutation[axis]]) as [number, number, number];
  const sign = [0, 1, 2].map((axis) => second.sign[axis] * first.sign[second.permutation[axis]]) as [1 | -1, 1 | -1, 1 | -1];
  const found = OCTREE_CUBE_TRANSFORMS.find((candidate) => candidate.permutation.every((value, axis) => value === permutation[axis])
    && candidate.sign.every((value, axis) => value === sign[axis]));
  if (!found) throw new Error("Cube transform table is not closed under composition");
  return found;
}

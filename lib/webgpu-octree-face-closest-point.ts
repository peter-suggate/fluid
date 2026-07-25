/**
 * GPU-resident Section 5 regular-face velocity extrapolation and power-face
 * publication for factor-4 and factor-8 global fine level sets.
 *
 * The fine SPGrid contributes the row-discovery frontier and authoritative
 * signed distance sampled at actual regular-face centroids; velocity remains
 * on compact octree rows/faces. Faces are emitted only after both endpoint
 * rows resolve, and each dry regular face samples the liquid vector field at
 * its physical closest point. An unresolved sample rejects the candidate.
 */
import { OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW, OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW,
  planOctreeRegularFaceBand, type OctreeRegularFaceBandPlan } from "./octree-face-band";
import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import {
  OCTREE_POWER_NEIGHBOR_DIRECTIONS,
  OCTREE_POWER_SAME_OR_COARSER_FLAG,
} from "./octree-power-descriptor";
import {
  OCTREE_POWER_ROW_DELTA_VALID,
  type OctreePowerRowDeltaSource,
} from "./webgpu-octree-power-descriptor";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import type { OctreePowerFaceSource } from "./webgpu-octree-power-faces";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import { PassBroker } from "./webgpu-pass-broker";

export const OCTREE_FACE_BAND_VALID = 0x8000_0000;
export const OCTREE_FACE_BAND_EXTRAPOLATED = 0x1000_0000;
export const OCTREE_FACE_BAND_CONTROL_BYTES = 128;
export const OCTREE_FACE_BAND_FACE_BYTES = 64;
export const OCTREE_FACE_BAND_ROW_BYTES = 32;
export const OCTREE_FACE_BAND_TRANSITION_ADJACENCY_BYTES = 16;
/** Existing 16-word gate followed by a 16-word first-owner-mismatch record,
 * then the appended S4-node prefix. The first 128 bytes remain ABI-stable. */
export const OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES = 160;
export const OCTREE_FACE_BAND_POWER_PUBLICATION_CONTROL_BYTES = 64;
export const OCTREE_FACE_BAND_POINT_FIELD_CONTROL_BYTES = 32;
export const OCTREE_FACE_BAND_TRANSIENT_POWER_FACE_BYTES = 64;
export const OCTREE_FACE_BAND_TRANSIENT_INCIDENCE_BYTES = 8;
export const OCTREE_FACE_BAND_TRANSIENT_ROW_BYTES = 16;
export const OCTREE_FACE_BAND_TRANSIENT_CONTROL_BYTES = 64;
/** The transient graph uses one first-error word for both row-producing
 * stages and physical-face sampling. Tagging face slots keeps those address
 * spaces unambiguous without widening the fixed control ABI. */
export const OCTREE_FACE_BAND_TRANSIENT_FACE_FAILURE = 0x8000_0000;
export const OCTREE_FACE_BAND_ACUTE_TETRA_FAILURE = 0x8000_0000;
/** A uniform trilinear stencil can touch every cell in the surrounding 3x3x3
 * owner block (the anchor itself is already present).  The generated power
 * catalog has only the 18 face/edge neighbours because body diagonals are not
 * Delaunay edges; Section 5 interpolation nevertheless needs the remaining
 * eight exact same-size corner owners. */
export const OCTREE_FACE_BAND_UNIFORM_SUPPORT_REQUESTS = 26;

/** CPU oracle for the uniform-catalog support request. "Uniform" describes
 * the anchor's six ordinary Delaunay faces; Section 6.1 can still encode a
 * parity-allowed coarse edge neighbor which does not alter those six faces.
 * The support graph must request that actual coarse owner, not a virtual
 * same-resolution cell at the descriptor direction. */
export function resolveOctreeFaceBandUniformSupportRequest(
  rowOrigin: readonly [number, number, number],
  rowSize: number,
  request: number,
  descriptor: number,
): { readonly origin: readonly [number, number, number]; readonly size: number } {
  if (rowOrigin.some((value) => !Number.isSafeInteger(value) || value < 0)
    || !Number.isSafeInteger(rowSize) || rowSize < 1
    || !Number.isSafeInteger(request) || request < 0 || request >= 18
    || !Number.isSafeInteger(descriptor) || descriptor < 0 || descriptor > 0xffff_ffff) {
    throw new RangeError("Uniform face-band support request is malformed");
  }
  const direction = OCTREE_POWER_NEIGHBOR_DIRECTIONS[request];
  let size = rowSize;
  if ((descriptor & OCTREE_POWER_SAME_OR_COARSER_FLAG) !== 0) {
    const child = [descriptor & 1, (descriptor >>> 1) & 1, (descriptor >>> 2) & 1];
    const outward = child.map((bit) => bit === 0 ? -1 : 1);
    const coarseDirections = [
      [outward[0], 0, 0], [0, outward[1], 0], [0, 0, outward[2]],
      [outward[0], outward[1], 0], [outward[0], 0, outward[2]], [0, outward[1], outward[2]],
    ];
    const sameOrigin = rowOrigin.map((value, axis) => value + direction[axis] * rowSize);
    for (let coarse = 0; coarse < coarseDirections.length; coarse += 1) {
      if ((descriptor & (1 << (coarse + 3))) === 0) continue;
      const coarseDirection = coarseDirections[coarse];
      const coarseSize = rowSize * 2;
      const probe = rowOrigin.map((value, axis) => coarseDirection[axis] < 0 ? value - 1
        : coarseDirection[axis] > 0 ? value + rowSize : value + Math.floor(rowSize / 2));
      const coarseOrigin = probe.map((value) => Math.floor(value / coarseSize) * coarseSize);
      const overlaps = coarseOrigin.every((value, axis) =>
        Math.min(value + coarseSize, sameOrigin[axis] + rowSize) > Math.max(value, sameOrigin[axis]));
      if (overlaps) { size = coarseSize; break; }
    }
  }
  if (size === rowSize) return {
    origin: rowOrigin.map((value, axis) => value + direction[axis] * rowSize) as [number, number, number],
    size,
  };
  const probe = rowOrigin.map((value, axis) => direction[axis] < 0 ? value - 1
    : direction[axis] > 0 ? value + rowSize : value + Math.floor(rowSize / 2));
  return {
    origin: probe.map((value) => Math.floor(value / size) * size) as [number, number, number],
    size,
  };
}
export const OCTREE_FACE_BAND_ERROR = Object.freeze({
  capacity: 1, invalidDirectory: 2, invalidSource: 4, invalidRow: 8,
  invalidFace: 16, invalidPhi: 32, unresolved: 64, incompleteVector: 128,
  outsideFineBand: 256,
} as const);
export const OCTREE_FACE_BAND_TRANSITION_ERROR = Object.freeze({
  invalidSource: 1, capacity: 2, unresolvedAdjacency: 4, invalidBandDescriptor: 8,
} as const);
export const OCTREE_FACE_BAND_TRANSITION_DETAIL = Object.freeze({
  malformedGeometry: 1, belowDomain: 2, aboveDomain: 4, misalignedGeometry: 8,
  ownerMismatch: 16, missingBandRow: 32, rowOutOfRange: 64, ownerSizeMismatch: 128,
} as const);
export const OCTREE_FACE_BAND_OWNER_FAILURE_STAGE = Object.freeze({
  support1: 1, support2: 2, support3: 3, transitionAdjacency: 4, endpoint: 5,
  supportEdge: 7, support4: 8, descriptor: 9, bandPhi: 10,
} as const);
export const OCTREE_FACE_BAND_POWER_PUBLICATION_ERROR = Object.freeze({
  invalidSource: 1, capacity: 2, missingRow: 4, invalidFace: 8,
  invalidNormal: 16, nonfinite: 32, incomplete: 64,
} as const);
export const OCTREE_FACE_BAND_POINT_ERROR = Object.freeze({
  invalidSource: 1, capacity: 2, invalidFace: 4, missingSample: 8,
  invalidNormal: 16, nonfinite: 32, singular: 64, illConditioned: 128,
} as const);

/** CPU mirror of the Section 5 fine/coarse publication gate. A restored A/B
 * scratch slot is not a paper publication and cannot seed a new face band. */
export function octreeFaceBandCoarseGenerationPairIsValid(
  coarseGeneration: number,
  fineGeneration: number,
  topologyControl: ArrayLike<number>,
): boolean {
  if (topologyControl.length < 8) return false;
  const mask = 0x3fff_ffff;
  const coarse = coarseGeneration & mask, fine = fineGeneration & mask;
  const clean = topologyControl[0] === 0 && topologyControl[4] === 1
    && topologyControl[5] === 0 && topologyControl[7] === 0;
  return clean && coarse === fine;
}

export interface OctreeFaceBandControlSnapshot {
  readonly flags: number;
  readonly firstError: number;
  readonly rowCount: number;
  readonly faceCount: number;
  readonly incidenceCount: number;
  readonly generation: number;
  readonly valid: boolean;
  readonly seedCount: number;
  readonly acceptedCount: number;
  readonly unresolvedCount: number;
  readonly sampleFailures: number;
  /** Faces whose current fine-centroid stencil was absent and were evaluated
   * from the redistanced transient owner field with the same cube/tetra interpolant. */
  readonly coarsePhiSamples: number;
  /** Owner rows or faces for which current fine, exact compact coarse, and
   * band-local redistance could not provide the Section 5 ordering scalar. */
  readonly coarsePhiFailures: number;
  /** Dry owner rows whose signed distance was extended from the current
   * narrow-band field over the transient Section 5 owner graph. */
  readonly bandPhiExtensions: number;
  /** Air faces resolved from a finite closest point and a liquid-only interpolant. */
  readonly closestPointFaces: number;
  /** Faces rejected because the signed-distance gradient could not define a closest point. */
  readonly closestPointFailures: number;
  /** Closest points whose inward sample had no complete pre-extrapolation liquid stencil. */
  readonly liquidInterpolationFailures: number;
  /** CPT queries whose inward ownership probe did not resolve a canonical owner. */
  readonly cptNoOwnerFailures: number;
  /** CPT queries whose exact owner is support-only and has no liquid power-row vector. */
  readonly cptSupportOwnerFailures: number;
  /** CPT queries for which the selected owner's local cube/Delaunay star contains no simplex. */
  readonly cptNoContainingSimplexFailures: number;
  /** CPT queries whose containing cube/tetrahedron has an unpublished liquid vertex vector. */
  readonly cptMissingLiquidVertexFailures: number;
  /** Stable, stage-local first failing identities. These are kept separate
   * from `firstError` so the final transition gate cannot hide the producer
   * that actually rejected the Section 5 publication. */
  readonly stageFirstFailures: {
    /** Cell/global-face identity rejected while emitting the regular-face graph. */
    readonly faceEmission: number;
    /** Row cell or global-face identity rejected while constructing signed distance. */
    readonly phi: number;
    /** Global-face identity rejected by closest-point velocity extension. */
    readonly closestPoint: number;
    /** Band-row identity rejected while reconstructing a complete vector. */
    readonly vectorReconstruction: number;
  };
  /** Direct face-arena slot for a phi/closest-point construction failure. */
  readonly firstPhiFailureSlot: number;
  /** First face-arena slot for each closest-point interpolation rejection.
   * The slot permits a direct QA read of the full Face record, including its
   * stable global-face identity, without scanning the arena. */
  readonly firstClosestPointFailureSlotByCause: {
    readonly noOwner: number;
    readonly supportOwner: number;
    readonly noContainingSimplex: number;
    readonly missingLiquidVertex: number;
  };
  readonly capacityFailure: boolean;
  readonly invalidDirectory: boolean;
  readonly invalidSource: boolean;
  readonly invalidRow: boolean;
  readonly invalidFace: boolean;
  readonly invalidPhi: boolean;
  readonly unresolved: boolean;
  readonly incompleteVector: boolean;
  readonly outsideFineBand: boolean;
}

export interface OctreeFaceBandPowerPublicationSnapshot {
  readonly flags: number;
  readonly firstError: number;
  readonly faceCount: number;
  readonly targetCount: number;
  readonly interpolatedCount: number;
  readonly committedCount: number;
  readonly fineGeneration: number;
  readonly powerGeneration: number;
  readonly valid: boolean;
}

export interface OctreeFaceBandTransitionControlSnapshot {
  readonly flags: number;
  readonly firstError: number;
  readonly rowCount: number;
  readonly transitionRows: number;
  readonly adjacencyCount: number;
  readonly ready: boolean;
  readonly transferReady: boolean;
  /** The catalog-Delaunay hierarchy committed before regular-face work began.
   * `ready` can remain false later when face/phi publication fails. */
  readonly hierarchyReady: boolean;
  readonly detailFlags: number;
  readonly malformedGeometry: boolean;
  readonly belowDomain: boolean;
  readonly aboveDomain: boolean;
  readonly misalignedGeometry: boolean;
  readonly ownerMismatch: boolean;
  readonly missingBandRow: boolean;
  readonly rowOutOfRange: boolean;
  readonly ownerSizeMismatch: boolean;
  /** Immutable prefix ends for the exact support hierarchy. */
  readonly coreRowCount: number;
  readonly support1RowCount: number;
  readonly support2RowCount: number;
  readonly support3NodeRowCount: number;
  readonly support4NodeRowCount?: number;
  readonly support5NodeRowCount?: number;
  readonly support6NodeRowCount?: number;
  readonly support7NodeRowCount?: number;
  readonly endpointRowCount: number;
  readonly boundaryGhostRequests: number;
  readonly phiFailureCounts: {
    readonly missingRow: number;
    readonly exactCoarseMiss: number;
    readonly invalidMetric: number;
    readonly invalidSelector: number;
  };
  readonly invalidSource: boolean;
  readonly capacityFailure: boolean;
  readonly unresolvedAdjacency: boolean;
  readonly invalidBandDescriptor: boolean;
  /** Readback-only first exact-owner mismatch. The selector reconstructs its
   * canonical xyzw from the immutable generated catalog. */
  readonly ownerFailure?: {
    readonly band: number;
    readonly stage: number;
    readonly rowCell: number;
    readonly rowSize: number;
    readonly descriptor: number;
    readonly topology: number;
    readonly transformFlags: number;
    readonly selector: number;
    readonly rawOrigin: readonly [number, number, number];
    readonly requestedSize: number;
    readonly resolvedOriginCell: number;
    readonly boundaryFlips: number;
    readonly actualOwnerCell: number;
    readonly actualOwnerSize: number;
    readonly actualOwnerValid: boolean;
  };
  readonly phiFailure?: {
    readonly cause: number;
    readonly faceIndex: number;
    readonly globalFace: number;
    readonly negativeRow: number;
    readonly positiveRow: number;
    readonly anchorRow: number;
    readonly centroid: readonly [number, number, number];
    readonly interpolantPath: number;
    readonly missingOrigin: readonly [number, number, number];
    readonly missingSize: number;
    readonly selectorOrCorner: number;
    readonly detail: number;
  };
}

export interface OctreeFaceBandPointFieldControlSnapshot {
  readonly flags: number;
  readonly firstError: number;
  readonly rowCount: number;
  readonly generation: number;
  readonly solvedCount: number;
  readonly valid: boolean;
  readonly wallContributions: number;
  readonly coreRowCount: number;
}

export interface OctreeFaceBandTransientPowerControlSnapshot {
  readonly flags: number;
  readonly firstError: number;
  readonly failureDomain?: "row" | "face";
  readonly rowCount: number;
  readonly faceSlots: number;
  readonly emittedCount: number;
  readonly sampledCount: number;
  readonly validatedCount: number;
  readonly generation: number;
  readonly valid: boolean;
  readonly diagnostic?: readonly number[];
}

/** CPU mirror of the live-prefix indirect schedule used by the GPU closure. */
export function planOctreeFaceBandLiveSupportDispatch(prefixes: {
  readonly coreEnd: number;
  readonly support1End: number;
  readonly support2End: number;
  readonly support3NodeEnd: number;
  readonly support4NodeEnd?: number;
}): readonly [number, number, number, number, number] {
  const values = [prefixes.coreEnd, prefixes.support1End, prefixes.support2End,
    prefixes.support3NodeEnd, prefixes.support4NodeEnd ?? prefixes.support3NodeEnd];
  if (values.some((value) => !Number.isSafeInteger(value) || value < 0)
    || values.some((value, index) => index > 0 && value < values[index - 1])) {
    throw new RangeError("Face-band live support prefixes must be monotone non-negative safe integers");
  }
  return [values[0], values[1] - values[0], values[2] - values[1], values[3] - values[2],
    values[4] - values[3]].map((count) => Math.ceil(count / 64)) as
    [number, number, number, number, number];
}

/** CPU mirror of the composed one-layer boundary classification used by the
 * Section 5 generalized-face LS producer. Plane bits follow the paper/catalog
 * descriptor order x-, y-, z-, z+, y+, x+. */
export function classifyOctreeFaceBandBoundaryCrossing(
  origin: readonly [number, number, number],
  size: number,
  dimensions: readonly [number, number, number],
  closedBoundaryMask: number,
): { readonly valid: boolean; readonly closedComponents: number; readonly openPlanes: number } {
  positive(size, "Face-band boundary neighbor size");
  const negativeBits = [1, 2, 4] as const, positiveBits = [32, 16, 8] as const;
  let closedComponents = 0, openPlanes = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const low = origin[axis], high = low + size;
    let plane = 0;
    if (low < 0) {
      if (low < -size) return { valid: false, closedComponents: 0, openPlanes: 0 };
      plane = negativeBits[axis];
    } else if (high > dimensions[axis]) {
      if (high > dimensions[axis] + size) return { valid: false, closedComponents: 0, openPlanes: 0 };
      plane = positiveBits[axis];
    }
    if (plane === 0) continue;
    if ((closedBoundaryMask & plane) !== 0) closedComponents |= 1 << axis;
    else openPlanes |= plane;
  }
  return { valid: true, closedComponents, openPlanes };
}

/** Decode the catalog-Delaunay transition gate that precedes face emission. */
export function unpackOctreeFaceBandTransitionControl(
  words: ArrayLike<number>,
): OctreeFaceBandTransitionControlSnapshot {
  if (words.length < 8) throw new RangeError("Face-band transition control needs eight u32 words");
  const flags = Number(words[0]) >>> 0;
  const detailFlags = Number(words[7]) >>> 0;
  const failureBand = Number(words[16] ?? 0xffff_ffff) >>> 0;
  const phiFailure = (failureBand & 0x8000_0000) !== 0 && failureBand !== 0xffff_ffff;
  const packedOwner = Number(words[31] ?? 0) >>> 0;
  const centroidBuffer = new ArrayBuffer(12);
  const centroidWords = new Uint32Array(centroidBuffer);
  centroidWords.set([Number(words[22] ?? 0), Number(words[23] ?? 0), Number(words[24] ?? 0)]);
  const centroid = Array.from(new Float32Array(centroidBuffer)) as [number, number, number];
  const phiCounts = Number(words[15] ?? 0) >>> 0;
  return {
    flags,
    firstError: Number(words[1]) >>> 0,
    rowCount: Number(words[2]) >>> 0,
    transitionRows: Number(words[3]) >>> 0,
    adjacencyCount: Number(words[4]) >>> 0,
    ready: (Number(words[5]) >>> 0) === OCTREE_FACE_BAND_VALID,
    transferReady: (Number(words[6]) >>> 0) === OCTREE_FACE_BAND_VALID,
    hierarchyReady: (Number(words[14] ?? 0) >>> 0) === OCTREE_FACE_BAND_VALID,
    detailFlags,
    malformedGeometry: (detailFlags & OCTREE_FACE_BAND_TRANSITION_DETAIL.malformedGeometry) !== 0,
    belowDomain: (detailFlags & OCTREE_FACE_BAND_TRANSITION_DETAIL.belowDomain) !== 0,
    aboveDomain: (detailFlags & OCTREE_FACE_BAND_TRANSITION_DETAIL.aboveDomain) !== 0,
    misalignedGeometry: (detailFlags & OCTREE_FACE_BAND_TRANSITION_DETAIL.misalignedGeometry) !== 0,
    ownerMismatch: (detailFlags & OCTREE_FACE_BAND_TRANSITION_DETAIL.ownerMismatch) !== 0,
    missingBandRow: (detailFlags & OCTREE_FACE_BAND_TRANSITION_DETAIL.missingBandRow) !== 0,
    rowOutOfRange: (detailFlags & OCTREE_FACE_BAND_TRANSITION_DETAIL.rowOutOfRange) !== 0,
    ownerSizeMismatch: (detailFlags & OCTREE_FACE_BAND_TRANSITION_DETAIL.ownerSizeMismatch) !== 0,
    coreRowCount: Number(words[8] ?? words[2]) >>> 0,
    support1RowCount: Number(words[9] ?? words[2]) >>> 0,
    support2RowCount: Number(words[10] ?? words[2]) >>> 0,
    support3NodeRowCount: Number(words[11] ?? words[2]) >>> 0,
    ...(words.length > 32 ? { support4NodeRowCount: Number(words[32]) >>> 0 } : {}),
    ...(words.length > 33 ? { support5NodeRowCount: Number(words[33]) >>> 0 } : {}),
    ...(words.length > 34 ? { support6NodeRowCount: Number(words[34]) >>> 0 } : {}),
    ...(words.length > 35 ? { support7NodeRowCount: Number(words[35]) >>> 0 } : {}),
    endpointRowCount: Number(words[12] ?? words[2]) >>> 0,
    boundaryGhostRequests: Number(words[13] ?? 0) >>> 0,
    phiFailureCounts: {
      missingRow: phiCounts & 0xff,
      exactCoarseMiss: (phiCounts >>> 8) & 0xff,
      invalidMetric: (phiCounts >>> 16) & 0xff,
      invalidSelector: (phiCounts >>> 24) & 0xff,
    },
    invalidSource: (flags & OCTREE_FACE_BAND_TRANSITION_ERROR.invalidSource) !== 0,
    capacityFailure: (flags & OCTREE_FACE_BAND_TRANSITION_ERROR.capacity) !== 0,
    unresolvedAdjacency: (flags & OCTREE_FACE_BAND_TRANSITION_ERROR.unresolvedAdjacency) !== 0,
    invalidBandDescriptor: (flags & OCTREE_FACE_BAND_TRANSITION_ERROR.invalidBandDescriptor) !== 0,
    ...(phiFailure ? { phiFailure: {
      cause: packedOwner & 0xff,
      faceIndex: Number(words[17]) >>> 0,
      globalFace: Number(words[18]) >>> 0,
      negativeRow: Number(words[19]) >>> 0,
      positiveRow: Number(words[20]) >>> 0,
      anchorRow: Number(words[21]) >>> 0,
      centroid,
      interpolantPath: Number(words[25]) >>> 0,
      missingOrigin: [Number(words[26]) | 0, Number(words[27]) | 0, Number(words[28]) | 0] as const,
      missingSize: Number(words[29]) >>> 0,
      selectorOrCorner: Number(words[30]) >>> 0,
      detail: packedOwner >>> 8,
    } } : failureBand !== 0xffff_ffff ? { ownerFailure: {
      band: failureBand,
      stage: Number(words[17]) >>> 0,
      rowCell: Number(words[18]) >>> 0,
      rowSize: Number(words[19]) >>> 0,
      descriptor: Number(words[20]) >>> 0,
      topology: Number(words[21]) >>> 0,
      transformFlags: Number(words[22]) >>> 0,
      selector: Number(words[23]) >>> 0,
      rawOrigin: [Number(words[24]) | 0, Number(words[25]) | 0, Number(words[26]) | 0] as const,
      requestedSize: Number(words[27]) >>> 0,
      resolvedOriginCell: Number(words[28]) >>> 0,
      boundaryFlips: Number(words[29]) >>> 0,
      actualOwnerCell: Number(words[30]) >>> 0,
      actualOwnerSize: packedOwner & 0xffff,
      actualOwnerValid: (packedOwner >>> 16) !== 0,
    } } : {}),
  };
}

/** Decode the fixed regular-to-power transaction header. */
export function unpackOctreeFaceBandPowerPublication(
  words: ArrayLike<number>,
): OctreeFaceBandPowerPublicationSnapshot {
  if (words.length < 9) throw new RangeError("Face-band power publication needs at least nine u32 words");
  const flags = Number(words[0]) >>> 0;
  const targetCount = Number(words[3]) >>> 0;
  const interpolatedCount = Number(words[4]) >>> 0;
  const committedCount = Number(words[5]) >>> 0;
  return {
    flags,
    firstError: Number(words[1]) >>> 0,
    faceCount: Number(words[2]) >>> 0,
    targetCount,
    interpolatedCount,
    committedCount,
    fineGeneration: Number(words[6]) >>> 0,
    powerGeneration: Number(words[7]) >>> 0,
    valid: flags === 0 && (Number(words[8]) >>> 0) === OCTREE_FACE_BAND_VALID
      && targetCount > 0 && interpolatedCount === targetCount && committedCount === targetCount,
  };
}

/** Decode the fixed 32-byte final cell-centre least-squares transaction. */
export function unpackOctreeFaceBandPointFieldControl(
  words: ArrayLike<number>,
): OctreeFaceBandPointFieldControlSnapshot {
  if (words.length < 8) throw new RangeError("Face-band point-field control needs eight u32 words");
  const flags = Number(words[0]) >>> 0;
  const rowCount = Number(words[2]) >>> 0;
  const solvedCount = Number(words[4]) >>> 0;
  const coreRowCount = Number(words[7]) >>> 0;
  return {
    flags,
    firstError: Number(words[1]) >>> 0,
    rowCount,
    generation: Number(words[3]) >>> 0,
    solvedCount,
    valid: flags === 0 && (Number(words[5]) >>> 0) === OCTREE_FACE_BAND_VALID
      && solvedCount === rowCount && coreRowCount <= rowCount,
    wallContributions: Number(words[6]) >>> 0,
    coreRowCount,
  };
}

/** Decode the fixed 64-byte all-band transient physical-face graph transaction. */
export function unpackOctreeFaceBandTransientPowerControl(
  words: ArrayLike<number>,
): OctreeFaceBandTransientPowerControlSnapshot {
  if (words.length < 9) throw new RangeError("Face-band transient power control needs at least nine u32 words");
  const flags = Number(words[0]) >>> 0;
  const rawFirstError = Number(words[1]) >>> 0;
  const hasFailure = rawFirstError !== 0xffff_ffff;
  const faceFailure = hasFailure && (rawFirstError & OCTREE_FACE_BAND_TRANSIENT_FACE_FAILURE) !== 0;
  const rowCount = Number(words[2]) >>> 0;
  const emittedCount = Number(words[4]) >>> 0;
  const sampledCount = Number(words[5]) >>> 0;
  const validatedCount = Number(words[6]) >>> 0;
  return {
    flags,
    firstError: faceFailure ? rawFirstError & ~OCTREE_FACE_BAND_TRANSIENT_FACE_FAILURE : rawFirstError,
    ...(hasFailure ? { failureDomain: faceFailure ? "face" as const : "row" as const } : {}),
    rowCount,
    faceSlots: Number(words[3]) >>> 0,
    emittedCount,
    sampledCount,
    validatedCount,
    generation: Number(words[7]) >>> 0,
    valid: flags === 0 && (Number(words[8]) >>> 0) === OCTREE_FACE_BAND_VALID
      && emittedCount > 0 && sampledCount === emittedCount && validatedCount === rowCount,
    ...(flags !== 0 && !faceFailure && words.length >= 16 && (Number(words[9]) >>> 0) !== 0xffff_ffff
      ? { diagnostic: Array.from({ length: 7 }, (_, index) => Number(words[9 + index]) >>> 0) }
      : {}),
  };
}

/** Decode the stable control header and optional direct-extension diagnostics. */
export function unpackOctreeFaceBandControl(words: ArrayLike<number>): OctreeFaceBandControlSnapshot {
  if (words.length < 13) throw new RangeError("Face-band control needs at least 13 u32 words");
  const flags = words[0] >>> 0;
  const faceCount = words[3] >>> 0, acceptedCount = words[9] >>> 0, unresolvedCount = words[10] >>> 0;
  return {
    flags, firstError: words[1] >>> 0, rowCount: words[2] >>> 0, faceCount,
    incidenceCount: words[4] >>> 0, generation: words[5] >>> 0,
    valid: (words[6] >>> 0) === OCTREE_FACE_BAND_VALID && flags === 0
      && unresolvedCount === 0 && acceptedCount === faceCount,
    seedCount: words[8] >>> 0, acceptedCount, unresolvedCount,
    sampleFailures: words[12] >>> 0,
    coarsePhiSamples: words[13] >>> 0,
    coarsePhiFailures: words[14] >>> 0,
    bandPhiExtensions: words[15] >>> 0,
    closestPointFaces: Number(words[16] ?? 0) >>> 0,
    closestPointFailures: Number(words[17] ?? 0) >>> 0,
    liquidInterpolationFailures: Number(words[18] ?? 0) >>> 0,
    cptNoOwnerFailures: Number(words[24] ?? 0) >>> 0,
    cptSupportOwnerFailures: Number(words[25] ?? 0) >>> 0,
    cptNoContainingSimplexFailures: Number(words[26] ?? 0) >>> 0,
    cptMissingLiquidVertexFailures: Number(words[27] ?? 0) >>> 0,
    stageFirstFailures: {
      faceEmission: Number(words[19] ?? 0xffff_ffff) >>> 0,
      phi: Number(words[21] ?? 0xffff_ffff) >>> 0,
      closestPoint: Number(words[22] ?? 0xffff_ffff) >>> 0,
      vectorReconstruction: Number(words[23] ?? 0xffff_ffff) >>> 0,
    },
    firstPhiFailureSlot: Number(words[20] ?? 0xffff_ffff) >>> 0,
    firstClosestPointFailureSlotByCause: {
      noOwner: Number(words[28] ?? 0xffff_ffff) >>> 0,
      supportOwner: Number(words[29] ?? 0xffff_ffff) >>> 0,
      noContainingSimplex: Number(words[30] ?? 0xffff_ffff) >>> 0,
      missingLiquidVertex: Number(words[31] ?? 0xffff_ffff) >>> 0,
    },
    capacityFailure: (flags & OCTREE_FACE_BAND_ERROR.capacity) !== 0,
    invalidDirectory: (flags & OCTREE_FACE_BAND_ERROR.invalidDirectory) !== 0,
    invalidSource: (flags & OCTREE_FACE_BAND_ERROR.invalidSource) !== 0,
    invalidRow: (flags & OCTREE_FACE_BAND_ERROR.invalidRow) !== 0,
    invalidFace: (flags & OCTREE_FACE_BAND_ERROR.invalidFace) !== 0,
    invalidPhi: (flags & OCTREE_FACE_BAND_ERROR.invalidPhi) !== 0,
    unresolved: (flags & OCTREE_FACE_BAND_ERROR.unresolved) !== 0,
    incompleteVector: (flags & OCTREE_FACE_BAND_ERROR.incompleteVector) !== 0,
    outsideFineBand: (flags & OCTREE_FACE_BAND_ERROR.outsideFineBand) !== 0,
  };
}

export interface OctreeFaceBandGPUPlan extends OctreeRegularFaceBandPlan {
  readonly metricRowCapacity: number;
  /** Padded count of immutable sorted `(cell,row)` directory records. */
  readonly rowDirectoryCapacity: number;
  /** Word offset of the level-addressed exact `(cell,size) -> row+1` table. */
  readonly identityToRowOffsetWords: number;
  /** Total number of exact dyadic identity slots across sizes 1..32. */
  readonly identityToRowCapacity: number;
  readonly rowDirectoryBytes: number;
  readonly rowDirectoryScratchBytes: number;
  readonly rowBytes: number;
  readonly bandFaceBytes: number;
  /** Live indirect-dispatch schedule. */
  readonly indirectDispatchBytes: number;
  readonly velocityBytes: number;
  readonly provisionalVelocityBytes: number;
  readonly pointStatusBytes: number;
  readonly transientPowerFaceCapacity: number;
  readonly transientPowerFaceBytes: number;
  readonly transientPowerIncidenceBytes: number;
  readonly transientPowerRowBytes: number;
  /** One exact fixed slot per catalog-neighbor request. The first arena
   * aliases transient incidence before that later phase begins. */
  readonly catalogSupportCandidateCapacity: number;
  readonly catalogSupportCandidateBytes: number;
  readonly catalogSupportScratchBytes: number;
  readonly transitionAdjacencyCapacity: number;
  readonly transitionAdjacencyBytes: number;
  readonly transitionMetricBytes: number;
  /** One bounded reverse-edge append counter per immutable row identity. */
  readonly endpointIncomingCountBytes: number;
  /** One count followed by the compact set of rows whose band phi is mutable. */
  readonly bandPhiFrontierBytes: number;
  /** One count followed by canonical immutable LIVE face-slot identities. */
  readonly liveFaceWorklistBytes: number;
  readonly powerFaceCapacity: number;
  readonly powerVelocityScratchBytes: number;
  readonly fusedAuthorityRegions: Readonly<Record<"control" | "rows" | "rowDirectory"
    | "velocities" | "metrics" | "transition" | "point",
    { readonly offsetBytes: number; readonly sizeBytes: number }>>;
  readonly fusedAuthorityBytes: number;
  /** Largest one-dimensional dispatch encoded by this plan. */
  readonly maximumDirectWorkgroups: number;
  readonly gpuAllocatedBytes: number;
}

export interface OctreeFaceBandInput {
  readonly fine: WebGPUFineLevelSetBrickSource;
  /** Transaction that produced `fine`; must prove a clean current publication. */
  readonly fineTopologyControl: GPUBuffer;
  /** Live octree owner authority. Fine bricks bound work but never define cell topology. */
  readonly owners: GPUBuffer;
  /** Published compact coarse phi supplies real sign/distance outside fine validity. */
  readonly coarsePhiDirectory: GPUBuffer;
  /** Canonical power-row directory sorted by `(level, Morton)`. */
  readonly powerRowDirectory: GPUBuffer;
  readonly powerRowDirectoryCapacity: number;
  /** Section 5 Stage-A least-squares vectors reconstructed from power faces. */
  readonly powerRowVelocities: GPUBuffer;
  readonly powerVelocityControl: GPUBuffer;
  readonly powerVelocityGeneration: number;
  readonly powerTopology: OctreePowerTopologySource;
  /**
   * Exact current/previous pressure-row identity transaction. Transition
   * adjacency has no whole-generation mode: an invalid or missing delta
   * rejects the candidate publication.
   */
  readonly rowDelta: OctreePowerRowDeltaSource;
  /** Live generalized faces that receive the paper's regular-to-power result transactionally. */
  readonly powerFaces: OctreePowerFaceSource;
  readonly dimensions: readonly [number, number, number];
  /** Finest-level liquid sites are exact; coarser sites remain valid seeds at transitions. */
  readonly maximumLeafSize: number;
  readonly generation: number;
  /** The dam-break top is open; all other container planes are closed. */
  readonly closedTop?: boolean;
}

export interface OctreeFaceBandSource {
  readonly plan: OctreeFaceBandGPUPlan;
  readonly control: GPUBuffer;
  readonly rows: GPUBuffer;
  readonly rowDirectory: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly incidence: GPUBuffer;
  readonly velocities: GPUBuffer;
  /**
   * Catalog-Delaunay `(anchor,a,b,c)` band-row records resolved from the live
   * owner neighborhood for every wet or dry band row. Publication consumes
   * them only after the transition and power-face transaction gates validate.
   */
  readonly transitionAdjacency: GPUBuffer;
  readonly transitionControl: GPUBuffer;
  readonly transitionMetrics: GPUBuffer;
  /** Diagnostic transaction header for regular-face -> power-face publication. */
  readonly powerPublicationControl: GPUBuffer;
  readonly pointFieldControl: GPUBuffer;
  readonly transientPowerControl: GPUBuffer;
}

export interface OctreeFaceBandSampleOptions {
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  readonly queryCount: number;
  readonly physicalCellSize: number;
  /** Adaptive octree owner map; band rows are keyed by owner origin. */
  readonly owners: GPUBuffer;
  /** Fine source generation whose face band must already be published. */
  readonly fineGeneration: number;
  /** Explicit current catalog authority for strict local point evaluation. */
  readonly powerTopology: OctreePowerTopologySource;
}

/** Immutable published inputs consumed by the fused fine characteristic. */
export interface OctreeFaceBandFusedSamplerSource {
  readonly params: GPUBuffer;
  readonly sampleParams: GPUBuffer;
  readonly authority: GPUBuffer;
  readonly regions: OctreeFaceBandGPUPlan["fusedAuthorityRegions"];
}

function positive(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}
export function planOctreeFaceBandGPU(
  wetRowCapacity: number,
  maximumFineBricks: number,
  brickResolution: number,
  fineFactor: number,
  finestCellDimensions: readonly [number, number, number],
  powerFaceCapacityValue?: number,
): OctreeFaceBandGPUPlan {
  const base = planOctreeRegularFaceBand(wetRowCapacity, maximumFineBricks, brickResolution, fineFactor);
  const domainOwnerCapacity = finestCellDimensions
    .map((value) => positive(value, "Face-band finest-cell dimension"))
    .reduce((product, value) => product * value, 1);
  const maximumSupport = OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW + Math.max(
    OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows,
    OCTREE_FACE_BAND_UNIFORM_SUPPORT_REQUESTS,
  );
  // The pressure capacity is a shared dense owner-row arena. Live power rows
  // retain their immutable prefix ids; exact support-only owners are appended
  // in canonical 4^3-brick/bit order by the topology producer. No ring arena,
  // atomic append order, or alternate authority survives this cut-over.
  const rowCapacity = Math.min(wetRowCapacity, domainOwnerCapacity);
  // Size every row-indexed arena once over the shared physical owner bound.
  const faceCapacity = rowCapacity * OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW;
  const incidenceBytes = rowCapacity * (1 + OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW) * 4;
  if (![rowCapacity, faceCapacity, incidenceBytes].every(Number.isSafeInteger)) {
    throw new RangeError("Face-band owner directory exceeds the exact integer range");
  }
  const rowDirectoryCapacity = Math.ceil(rowCapacity / 256) * 256;
  // The committed A-side directory owns its four-word support epoch in the
  // same allocation: final end, S1 end, S2 end, and S3-node end. Grouping this
  // immutable metadata with the directory keeps delta consumers within
  // WebGPU's portable ten-storage-buffer limit.
  const identityToRowOffsetWords = rowDirectoryCapacity * 2 + 4;
  // Six dense, level-addressed planes are collision free even when valid
  // dyadic identities share an origin. The small fixed schedule trades unused
  // unaligned cells for a single shift-level + linear-cell lookup in WGSL:
  // no per-query dimension schedule, prefix sum, hash, or binary search.
  const identityToRowCapacity = domainOwnerCapacity * 6;
  const rowDirectoryBytes = (identityToRowOffsetWords + identityToRowCapacity) * 4;
  const rowDirectoryScratchBytes = rowDirectoryCapacity * 12;
  const rowBytes = rowCapacity * OCTREE_FACE_BAND_ROW_BYTES;
  const bandFaceBytes = faceCapacity * OCTREE_FACE_BAND_FACE_BYTES;
  // Twenty-two three-word records cover the existing row/face schedule plus
  // exact S1/S2/S3-node catalog support and terminal endpoint emission. S3 and
  // the endpoint pass reuse consumed dispatch records, so candidate dispatches
  // still derive from GPU-published live prefixes without capacity-sized scans.
  const indirectDispatchBytes = 264;
  const velocityBytes = rowCapacity * 16;
  const provisionalVelocityBytes = rowCapacity * 16;
  const pointStatusBytes = rowCapacity * 4;
  const transientPowerFaceCapacity = rowCapacity
    * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence;
  const transientPowerFaceBytes = transientPowerFaceCapacity
    * OCTREE_FACE_BAND_TRANSIENT_POWER_FACE_BYTES;
  // Reused before transient-power publication as the complete incoming
  // Section 5 scalar-support edge list. Total support requests are bounded by
  // rows*maximumSupport even when one row's reverse degree is larger.
  const transientPowerIncidenceBytes = rowCapacity * Math.max(
    OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence,
    maximumSupport,
  ) * OCTREE_FACE_BAND_TRANSIENT_INCIDENCE_BYTES;
  const transientPowerRowBytes = (rowCapacity + 1) * OCTREE_FACE_BAND_TRANSIENT_ROW_BYTES;
  const catalogSupportCandidateCapacity = rowCapacity * maximumSupport;
  if (catalogSupportCandidateCapacity < faceCapacity) {
    throw new RangeError("Face-band catalog scratch cannot stage one repair carrier per face");
  }
  const catalogSupportCandidateBytes = catalogSupportCandidateCapacity * 8;
  // Support publication uses a collision-free (cell, dyadic-level) identity
  // plane.  Two dense planes retain the mark and block-local exclusive
  // prefix; the small tail stores one total per 256-entry block plus the
  // global total.  The same allocation is reused later by dry-face repair.
  const supportIdentityCount = domainOwnerCapacity * 6;
  const supportIdentityBlockCount = Math.ceil(supportIdentityCount / 256);
  const catalogSupportScatterBytes =
    (supportIdentityCount * 2 + supportIdentityBlockCount + 1) * 4;
  const catalogSupportScratchBytes = Math.max(
    catalogSupportCandidateBytes,
    catalogSupportScatterBytes,
  );
  const metricRowCapacity = rowCapacity;
  const transitionAdjacencyCapacity = metricRowCapacity
    * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra;
  const transitionAdjacencyBytes = transitionAdjacencyCapacity * OCTREE_FACE_BAND_TRANSITION_ADJACENCY_BYTES;
  const transitionMetricBytes = rowCapacity * 16;
  const endpointIncomingCountBytes = rowCapacity * 4;
  const bandPhiFrontierBytes = (rowCapacity + 1) * 4;
  const liveFaceWorklistBytes = (faceCapacity + 1) * 4;
  const powerFaceCapacity = positive(powerFaceCapacityValue
    ?? base.wetRowCapacity * Math.ceil(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence / 2),
  "Face-band power-face capacity");
  // Scratch changes representation across the publication transaction:
  // (0, 0, negative band, positive band) -> (vector xyz bits, valid) ->
  // (normal velocity bits, targeted marker, 0, 0). Splitting catalog
  // interpolation from normal projection keeps every compute stage within
  // WebGPU's portable ten-storage-buffer limit.
  const powerVelocityScratchBytes = powerFaceCapacity * 16;
  const fusedAuthoritySizes = {
    control: OCTREE_FACE_BAND_CONTROL_BYTES,
    rows: rowBytes,
    rowDirectory: rowDirectoryBytes,
    velocities: velocityBytes,
    metrics: transitionMetricBytes,
    transition: OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES,
    point: OCTREE_FACE_BAND_POINT_FIELD_CONTROL_BYTES,
  };
  let fusedAuthorityBytes = 0;
  const fusedAuthorityRegions = {} as Record<keyof typeof fusedAuthoritySizes,
    { offsetBytes: number; sizeBytes: number }>;
  for (const key of Object.keys(fusedAuthoritySizes) as (keyof typeof fusedAuthoritySizes)[]) {
    const sizeBytes = fusedAuthoritySizes[key];
    fusedAuthorityRegions[key] = { offsetBytes: fusedAuthorityBytes, sizeBytes };
    fusedAuthorityBytes += sizeBytes;
  }
  const maximumDirectWorkgroups = Math.ceil(Math.max(rowCapacity, faceCapacity,
    powerFaceCapacity, transientPowerFaceCapacity, domainOwnerCapacity) / 64);
  if (![transitionAdjacencyCapacity, transitionAdjacencyBytes,
    transientPowerFaceCapacity, catalogSupportCandidateCapacity,
    transientPowerFaceBytes, transientPowerIncidenceBytes, transientPowerRowBytes,
    catalogSupportCandidateBytes, catalogSupportScratchBytes,
    identityToRowOffsetWords, identityToRowCapacity, rowDirectoryBytes,
    endpointIncomingCountBytes, bandPhiFrontierBytes, liveFaceWorklistBytes,
    maximumDirectWorkgroups].every(Number.isSafeInteger)) {
    throw new RangeError("Face-band transition adjacency exceeds the exact integer range");
  }
  return { ...base, rowCapacity, faceCapacity, incidenceBytes,
    allocatedBytes: bandFaceBytes + incidenceBytes,
    metricRowCapacity,
    rowDirectoryCapacity, identityToRowOffsetWords, identityToRowCapacity,
    rowDirectoryBytes, rowDirectoryScratchBytes,
    rowBytes, bandFaceBytes,
    indirectDispatchBytes, velocityBytes, provisionalVelocityBytes, pointStatusBytes,
    transientPowerFaceCapacity, transientPowerFaceBytes, transientPowerIncidenceBytes, transientPowerRowBytes,
    catalogSupportCandidateCapacity, catalogSupportCandidateBytes, catalogSupportScratchBytes,
    transitionAdjacencyCapacity, transitionAdjacencyBytes, transitionMetricBytes,
    endpointIncomingCountBytes, bandPhiFrontierBytes, liveFaceWorklistBytes,
    fusedAuthorityRegions: Object.freeze(fusedAuthorityRegions), fusedAuthorityBytes,
    powerFaceCapacity, powerVelocityScratchBytes, maximumDirectWorkgroups,
    gpuAllocatedBytes: rowBytes + bandFaceBytes + incidenceBytes
      + rowDirectoryBytes + rowDirectoryScratchBytes
      + indirectDispatchBytes + velocityBytes + provisionalVelocityBytes
      + pointStatusBytes + OCTREE_FACE_BAND_POINT_FIELD_CONTROL_BYTES
      + transientPowerFaceBytes + transientPowerIncidenceBytes + transientPowerRowBytes
      + catalogSupportScratchBytes
      + OCTREE_FACE_BAND_TRANSIENT_CONTROL_BYTES
      + transitionAdjacencyBytes + transitionMetricBytes
      // Exact delta carry snapshots replace the deleted whole-generation
      // adjacency reconstruction. They are topology state, not a fallback.
      + rowBytes + rowDirectoryBytes + bandFaceBytes + incidenceBytes
      + transitionAdjacencyBytes + transitionMetricBytes
      + OCTREE_FACE_BAND_CONTROL_BYTES + OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES
      + rowCapacity * 4 + endpointIncomingCountBytes + bandPhiFrontierBytes
      + liveFaceWorklistBytes
      + powerVelocityScratchBytes
      + OCTREE_FACE_BAND_CONTROL_BYTES + OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES
      + OCTREE_FACE_BAND_POWER_PUBLICATION_CONTROL_BYTES
      + fusedAuthorityBytes + 64
      // Topology P (112), sample/repair parameters (48 each), and retained
      // topology P (112). Count the actual buffers, not the old 96-byte P ABI.
      + 112 + 48 + 48 + 112 };
}

function storage(device: GPUDevice, size: number, label: string, extra = 0): GPUBuffer {
  return device.createBuffer({ label, size: Math.max(4, Math.ceil(size / 4) * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra });
}

/** Dependency-ordered Section 5 checkpoints. Keeping each checkpoint in one
 * compute pass preserves the paper's publication order while avoiding a
 * driver/command-buffer transition around every small dispatch. */
export const OCTREE_FACE_BAND_ENCODE_PHASES = [
  "topology-build",
  "transition-adjacency",
  "closest-point-extension",
  "power-publication",
] as const;
export type OctreeFaceBandEncodePhase = typeof OCTREE_FACE_BAND_ENCODE_PHASES[number];

interface FaceBandBindGroupCacheNode {
  readonly buffers: WeakMap<GPUBuffer, FaceBandBindGroupCacheNode>;
  bindGroup?: GPUBindGroup;
}

/**
 * Every LIVE face is resolved independently from its physical closest point.
 * No face is committed while any requested closest point or liquid-only
 * interpolation stencil is invalid.
 * Fine bricks provide the bounded row-discovery frontier and current physical
 * signed distance wherever their narrow band contains a regular-face centroid.
 * Outside that finite stencil, coarse signed distance is extended across
 * the transient owner graph and evaluated with the paper's cube/local-Delaunay
 * interpolant; row topology still comes from the owner map. This increment accepts uniform owners;
 * transition owners fail closed until their regular-face adjacency is emitted
 * from the catalog Delaunay tetrahedra already used by the Stage-B sampler.
 */
export class WebGPUOctreeFaceClosestPointExtension {
  readonly plan: OctreeFaceBandGPUPlan;
  /** Last completely validated immutable Section 5 authority. */
  readonly control: GPUBuffer;
  readonly rows: GPUBuffer;
  readonly rowDirectory: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly incidence: GPUBuffer;
  readonly transitionAdjacency: GPUBuffer;
  readonly transitionControl: GPUBuffer;
  readonly transitionMetrics: GPUBuffer;
  /** Unpublished B-side topology. It may be overwritten freely because no
   * consumer can observe it before the final closest-point commit gate. */
  private readonly candidateControl: GPUBuffer;
  private readonly candidateRows: GPUBuffer;
  private readonly candidateRowDirectory: GPUBuffer;
  private readonly candidateFaces: GPUBuffer;
  private readonly candidateIncidence: GPUBuffer;
  private readonly candidateTransitionAdjacency: GPUBuffer;
  private readonly candidateTransitionControl: GPUBuffer;
  private readonly candidateTransitionMetrics: GPUBuffer;
  private readonly rowDirectoryScratch: GPUBuffer;
  private readonly catalogSupportScratch: GPUBuffer;
  readonly velocities: GPUBuffer;
  /** Per-row exclusive prefix for the exact affected transition-row compaction. */
  private readonly transitionDeltaScan: GPUBuffer;
  /** Direct reverse-edge append count for each terminal support identity. */
  private readonly endpointIncomingCounts: GPUBuffer;
  /** Dedicated recurring mutable-phi frontier; never aliases immutable graph state. */
  private readonly bandPhiFrontier: GPUBuffer;
  /** Canonical dense list of immutable LIVE candidate-face slots. */
  private readonly liveFaceWorklist: GPUBuffer;
  /** Atomic partials used only by parallel publication diagnostics. */
  private readonly publicationReductions: GPUBuffer;
  readonly powerPublicationControl: GPUBuffer;
  readonly pointFieldControl: GPUBuffer;
  private readonly powerVelocityScratch: GPUBuffer;
  private readonly provisionalVelocities: GPUBuffer;
  private readonly pointStatus: GPUBuffer;
  private readonly transientPowerFaces: GPUBuffer;
  private readonly transientPowerIncidence: GPUBuffer;
  private readonly transientPowerRows: GPUBuffer;
  readonly transientPowerControl: GPUBuffer;
  private readonly indirect: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly sampleParams: GPUBuffer;
  private readonly completionParams: GPUBuffer;
  private readonly completionTopologyParams: GPUBuffer;
  private readonly fusedAuthority: GPUBuffer;
  private readonly fusedAuthorityParams: GPUBuffer;
  private readonly fusedAuthorityPipeline: GPUComputePipeline;
  private readonly fusedAuthorityBindGroup: GPUBindGroup;
  private readonly fusedAuthorityWorkgroups: number;
  private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;
  private readonly preparePowerAdvectionCompletionPipeline: GPUComputePipeline;
  private readonly completePowerAdvectionPipeline: GPUComputePipeline;
  private readonly publishPowerAdvectionCompletionPipeline: GPUComputePipeline;
  /**
   * Bind groups are immutable and depend only on the pipeline layout, binding
   * numbers, and GPUBuffer identities. A weak identity trie retains no
   * otherwise-dead external scene generation while making the steady-state
   * encoder allocation-free.
   */
  private readonly bindGroupCache =
    new Map<GPUComputePipeline, Map<string, FaceBandBindGroupCacheNode>>();
  private destroyed = false;

  constructor(private readonly device: GPUDevice, fine: WebGPUFineLevelSetBrickSource,
    wetRowCapacity: number, private readonly bandPhiRelaxationRounds: number,
    powerFaceCapacity?: number) {
    positive(bandPhiRelaxationRounds, "Band-phi relaxation round count");
    this.plan = planOctreeFaceBandGPU(wetRowCapacity, fine.plan.maximumResidentBricks,
      fine.plan.brickResolution, fine.plan.fineFactor, fine.plan.finestCellDimensions, powerFaceCapacity);
    const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    for (const [label, bytes] of [["row directory", this.plan.rowDirectoryBytes],
      ["row directory scratch", this.plan.rowDirectoryScratchBytes], ["faces", this.plan.bandFaceBytes],
      ["candidate rows", this.plan.rowBytes],
      ["candidate row directory", this.plan.rowDirectoryBytes],
      ["candidate faces", this.plan.bandFaceBytes],
      ["candidate incidence", this.plan.incidenceBytes],
      ["transition adjacency", this.plan.transitionAdjacencyBytes],
      ["candidate transition adjacency", this.plan.transitionAdjacencyBytes],
      ["transition metrics", this.plan.transitionMetricBytes],
      ["candidate transition metrics", this.plan.transitionMetricBytes],
      ["catalog support scratch", this.plan.catalogSupportScratchBytes],
      ["committed transition state", 16],
      ["transition delta scan", this.plan.rowCapacity * 4],
      ["endpoint incoming counts", this.plan.endpointIncomingCountBytes],
      ["band phi frontier", this.plan.bandPhiFrontierBytes],
      ["live face worklist", this.plan.liveFaceWorklistBytes],
      ["publication reductions", 32],
      ["provisional velocities", this.plan.provisionalVelocityBytes],
      ["point status", this.plan.pointStatusBytes],
      ["transient power faces", this.plan.transientPowerFaceBytes],
      ["transient power incidence", this.plan.transientPowerIncidenceBytes],
      ["transient power rows", this.plan.transientPowerRowBytes],
      ["power velocity scratch", this.plan.powerVelocityScratchBytes],
      ["grouped transport authority", this.plan.fusedAuthorityBytes]] as const) {
      if (bytes > maximumBinding) throw new RangeError(`Face-band ${label} exceeds the adapter storage binding limit`);
    }
    if (this.plan.maximumDirectWorkgroups > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new RangeError("Face-band direct dispatch exceeds the adapter workgroup dimension limit");
    }
    this.control = storage(device, OCTREE_FACE_BAND_CONTROL_BYTES, "octree face-band control");
    this.rows = storage(device, this.plan.rowBytes, "octree face-band rows");
    this.rowDirectory = storage(device, this.plan.rowDirectoryBytes,
      "octree face-band sorted row directory");
    this.faces = storage(device, this.plan.bandFaceBytes, "octree face-band faces");
    this.incidence = storage(device, this.plan.incidenceBytes, "octree face-band incidence");
    this.transitionAdjacency = storage(device, this.plan.transitionAdjacencyBytes,
      "octree catalog-Delaunay face-band adjacency");
    this.transitionControl = storage(device, OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES,
      "octree face-band transition control");
    this.transitionMetrics = storage(device, this.plan.transitionMetricBytes,
      "octree all-band catalog metrics");
    this.candidateControl = storage(device, OCTREE_FACE_BAND_CONTROL_BYTES,
      "octree face-band candidate control");
    this.candidateRows = storage(device, this.plan.rowBytes, "octree face-band candidate rows");
    this.candidateRowDirectory = storage(device, this.plan.rowDirectoryBytes,
      "octree face-band candidate sorted row directory");
    this.candidateFaces = storage(device, this.plan.bandFaceBytes, "octree face-band candidate faces");
    this.candidateIncidence = storage(device, this.plan.incidenceBytes,
      "octree face-band candidate incidence");
    this.candidateTransitionAdjacency = storage(device, this.plan.transitionAdjacencyBytes,
      "octree candidate catalog-Delaunay face-band adjacency");
    this.candidateTransitionControl = storage(device, OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES,
      "octree face-band candidate transition control");
    this.candidateTransitionMetrics = storage(device, this.plan.transitionMetricBytes,
      "octree candidate all-band catalog metrics");
    this.rowDirectoryScratch = storage(device, this.plan.rowDirectoryScratchBytes,
      "octree face-band row-directory radix scratch");
    this.catalogSupportScratch = storage(device, this.plan.catalogSupportScratchBytes,
      "octree face-band catalog-support radix scratch");
    this.velocities = storage(device, this.plan.velocityBytes, "octree face-band row velocity");
    this.provisionalVelocities = storage(device, this.plan.provisionalVelocityBytes,
      "octree face-band provisional carrier velocity");
    this.pointStatus = storage(device, this.plan.pointStatusBytes, "octree face-band point-field row status");
    this.pointFieldControl = storage(device, OCTREE_FACE_BAND_POINT_FIELD_CONTROL_BYTES,
      "octree face-band point-field control");
    this.transientPowerFaces = storage(device, this.plan.transientPowerFaceBytes,
      "octree face-band transient physical power faces");
    this.transientPowerIncidence = storage(device, this.plan.transientPowerIncidenceBytes,
      "octree face-band transient physical power incidence");
    this.transientPowerRows = storage(device, this.plan.transientPowerRowBytes,
      "octree face-band transient physical power CSR rows");
    this.transientPowerControl = storage(device, OCTREE_FACE_BAND_TRANSIENT_CONTROL_BYTES,
      "octree face-band transient physical power graph control");
    this.transitionDeltaScan = storage(device, this.plan.rowCapacity * 4,
      "octree face-band transition delta scan");
    this.endpointIncomingCounts = storage(device, this.plan.endpointIncomingCountBytes,
      "octree face-band endpoint incoming counts");
    this.bandPhiFrontier = storage(device, this.plan.bandPhiFrontierBytes,
      "octree face-band mutable phi frontier");
    this.liveFaceWorklist = storage(device, this.plan.liveFaceWorklistBytes,
      "octree face-band live face worklist");
    this.publicationReductions = storage(device, 128,
      "octree face-band parallel publication reductions");
    this.powerVelocityScratch = storage(device, this.plan.powerVelocityScratchBytes,
      "octree regular-to-power velocity scratch");
    this.powerPublicationControl = storage(device, OCTREE_FACE_BAND_POWER_PUBLICATION_CONTROL_BYTES,
      "octree regular-to-power publication control");
    this.fusedAuthority = storage(device, this.plan.fusedAuthorityBytes,
      "grouped face-band transport authority");
    this.fusedAuthorityParams = device.createBuffer({
      label: "grouped face-band transport authority parameters", size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const fused = this.plan.fusedAuthorityRegions;
    device.queue.writeBuffer(this.fusedAuthorityParams, 0, new Uint32Array([
      fused.control.offsetBytes / 4, fused.control.sizeBytes / 4,
      fused.rows.offsetBytes / 4, fused.rows.sizeBytes / 4,
      fused.rowDirectory.offsetBytes / 4, fused.rowDirectory.sizeBytes / 4,
      fused.velocities.offsetBytes / 4, fused.velocities.sizeBytes / 4,
      fused.metrics.offsetBytes / 4, fused.metrics.sizeBytes / 4,
      fused.transition.offsetBytes / 4, fused.transition.sizeBytes / 4,
      fused.point.offsetBytes / 4, fused.point.sizeBytes / 4, 0, 0,
    ]));
    this.indirect = storage(device, this.plan.indirectDispatchBytes,
      "octree face-band indirect dispatch", GPUBufferUsage.INDIRECT);
    this.params = device.createBuffer({ label: "octree face-band parameters", size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampleParams = device.createBuffer({ label: "octree face-band sample parameters", size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.completionParams = device.createBuffer({ label: "regular-face completion parameters", size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.completionTopologyParams = device.createBuffer({ label: "regular-face completion topology parameters", size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const shaderModule = device.createShaderModule({
      label: "octree face-band topology and closest-point extension",
      code: octreeFaceBandWGSL,
    });
    const supportScatterModule = device.createShaderModule({
      label: "octree face-band parallel identity support scatter",
      code: octreeFaceBandSupportScatterWGSL,
    });
    const pipeline = (entryPoint: string, constants?: Record<string, number>) => device.createComputePipeline({ label: entryPoint, layout: "auto",
      compute: { module: shaderModule, entryPoint, ...(constants ? { constants } : {}) } });
    const supportPipeline = (entryPoint: string) => device.createComputePipeline({
      label: entryPoint,
      layout: "auto",
      compute: { module: supportScatterModule, entryPoint },
    });
    this.fusedAuthorityPipeline = device.createComputePipeline({
      label: "Publish grouped face-band transport authority", layout: "auto",
      compute: { module: device.createShaderModule({ code: octreeFaceBandFusedAuthorityWGSL }),
        entryPoint: "publishFaceBandFusedAuthority" },
    });
    const pipelines: Record<string, GPUComputePipeline> = {
      prepare: pipeline("prepareFaceBandDelta"),
      buildTopologyDelta: pipeline("buildFaceBandTopologyDelta"),
      emitCatalogSupport1: pipeline("emitFaceBandCatalogSupport1"),
      emitCatalogSupport2: pipeline("emitFaceBandCatalogSupport2"),
      emitCatalogSupport3: pipeline("emitFaceBandCatalogSupport3"),
      emitEndpointSupport4: pipeline("emitFaceBandEndpointSupport4"),
      clearSupportIdentityMarks: supportPipeline("clearSupportIdentityMarks"),
      markSupport1: supportPipeline("markSupport1"),
      markSupport2: supportPipeline("markSupport2"),
      markSupport3: supportPipeline("markSupport3"),
      markSupport4: supportPipeline("markSupport4"),
      markPublishedSupportRows: supportPipeline("markPublishedSupportRows"),
      scanSupportIdentityBlocks: supportPipeline("scanSupportIdentityBlocks"),
      prefixSupportIdentityBlocks: supportPipeline("prefixSupportIdentityBlocks"),
      finalizeSupport1: supportPipeline("finalizeSupport1"),
      finalizeSupport2: supportPipeline("finalizeSupport2"),
      finalizeSupport3: supportPipeline("finalizeSupport3"),
      finalizeSupport4: supportPipeline("finalizeSupport4"),
      scatterSupport1: supportPipeline("scatterSupport1"),
      scatterSupport2: supportPipeline("scatterSupport2"),
      scatterSupport3: supportPipeline("scatterSupport3"),
      scatterSupport4: supportPipeline("scatterSupport4"),
      scatterCanonicalRowDirectory: supportPipeline("scatterCanonicalRowDirectory"),
      validateRowDirectory: pipeline("validateFaceBandRowDirectory"),
      classifyRowSigns: pipeline("classifyFaceBandRowSigns"),
      carryFaceTopology: pipeline("carryFaceBandTopology"),
      carryFaceIncidence: pipeline("carryFaceBandIncidence"),
      rebuildIncidence: pipeline("rebuildFaceBandIncidence"),
      publishFaceCounts: pipeline("publishFaceBandCounts"),
      commitRows: pipeline("commitFaceBandRows"),
      commitFaces: pipeline("commitFaceBandFaces"),
      commitTransition: pipeline("commitFaceBandTransition"),
      commitControls: pipeline("commitFaceBandControls"),
      emit: pipeline("emitBandFaces"),
      sampleFacePhi: pipeline("sampleBandFacePhi"),
      initializeBandPhi: pipeline("initializeBandRowPhi"),
      seedBandPhiFaces: pipeline("seedBandRowPhiFromFaces"),
      collectBandPhiFrontier: pipeline("collectBandPhiActiveRows"),
      extendBandPhi: pipeline("extendBandRowPhiParallel"),
      commitBandPhi: pipeline("commitBandRowPhi"),
      sampleFaceCoarsePhi: pipeline("sampleBandFaceCoarsePhi"),
      reducePhiFailure: pipeline("reduceBandPhiFailure"),
      summarizeRowPhi: pipeline("summarizeBandRowPhi"), seedCentroids: pipeline("seedFaceCentroids"),
      extendClosestPoints: pipeline("extendFaceClosestPoints"),
      gatherClosestPointRepairs: pipeline("gatherDryFaceClosestPointRepairs"),
      commitClosestPointRepairs: pipeline("commitDryFaceClosestPointRepairs"),
      reconstruct: pipeline("reconstructBandRowVelocity"),
      seedMappedPowerRows: pipeline("seedMappedPowerRowVelocity"),
      finalizeClosestPoint: pipeline("finalizeFaceBandClosestPointDiagnostics"),
      preparePointField: pipeline("prepareBandPointField"),
      reducePointField: pipeline("reduceBandPointField"),
      finalizePointField: pipeline("finalizeBandPointField"),
      prepareTransientPower: pipeline("prepareTransientBandPowerGraph"),
      emitTransientPower: pipeline("emitTransientBandPowerGraph"),
      sampleTransientPower: pipeline("sampleTransientBandPowerFaces"),
      validateTransientPower: pipeline("validateTransientBandPowerGraph"),
      reconstructPhysicalPoint: pipeline("reconstructBandTransientPowerPointField"),
      publishPoint: pipeline("publishBandPointField"),
      prepareTransition: pipeline("prepareCatalogTransitionAdjacency"),
      classifyTransitionDelta: pipeline("classifyCatalogTransitionDelta"),
      carryTransitionAdjacency: pipeline("carryCatalogTransitionAdjacency"),
      clearBandPhiEndpointEdges: pipeline("clearBandPhiEndpointIncomingEdges"),
      buildBandPhiEdges: pipeline("buildBandPhiIncomingEdges"),
      buildBandPhiEndpointEdges: pipeline("validateBandPhiEndpointIncomingEdges"),
      describeCatalogRows: pipeline("describeCatalogBandRows"),
      resolveCatalogAdjacency: pipeline("resolveCatalogTransitionAdjacency"),
      validateCatalogAdjacency: pipeline("validateCatalogTransitionAdjacency"),
      publishCatalogAdjacency: pipeline("publishCatalogTransitionAdjacency"),
      gateTransition: pipeline("gateTransitionTransfer"),
      preparePowerPublication: pipeline("preparePowerPublication"),
      mapPowerFaceBands: pipeline("mapPowerFaceBands"),
      interpolatePowerFaces: pipeline("interpolatePowerFaceVector"),
      projectPowerFaces: pipeline("projectPowerFaceVelocity"),
      commitPowerFaces: pipeline("commitPowerFaceVelocity") };
    this.pipelines = Object.freeze(pipelines);
    this.preparePowerAdvectionCompletionPipeline = pipeline("preparePowerFaceRegularCompletion");
    this.completePowerAdvectionPipeline = pipeline("completePowerFaceAdvectionFromRegularBand");
    this.publishPowerAdvectionCompletionPipeline = pipeline("publishCompletedPowerFaceAdvection");
    this.fusedAuthorityWorkgroups = Math.ceil(Math.max(
      ...Object.values(this.plan.fusedAuthorityRegions).map(region => region.sizeBytes),
    ) / 256);
    this.fusedAuthorityBindGroup = this.cachedBindGroup(
      "grouped face-band transport authority",
      this.fusedAuthorityPipeline,
      [[0, this.fusedAuthorityParams], [1, this.control], [2, this.rows],
        [3, this.rowDirectory], [4, this.velocities], [5, this.transitionMetrics],
        [6, this.transitionControl], [7, this.pointFieldControl], [8, this.fusedAuthority]],
    );
  }

  private cachedBindGroup(label: string, pipeline: GPUComputePipeline,
    entries: readonly (readonly [number, GPUBuffer])[]): GPUBindGroup {
    const signature = entries.map(([binding]) => binding).join(",");
    let bySignature = this.bindGroupCache.get(pipeline);
    if (!bySignature) {
      bySignature = new Map();
      this.bindGroupCache.set(pipeline, bySignature);
    }
    let node: FaceBandBindGroupCacheNode;
    const cachedRoot = bySignature.get(signature);
    if (cachedRoot) node = cachedRoot;
    else {
      node = { buffers: new WeakMap() };
      bySignature.set(signature, node);
    }
    for (const [, buffer] of entries) {
      let next: FaceBandBindGroupCacheNode | undefined = node.buffers.get(buffer);
      if (!next) {
        next = { buffers: new WeakMap() };
        node.buffers.set(buffer, next);
      }
      node = next;
    }
    if (!node.bindGroup) {
      node.bindGroup = this.device.createBindGroup({
        label,
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
      });
    }
    return node.bindGroup;
  }

  encode(broker: PassBroker, input: OctreeFaceBandInput): void {
    for (const phase of OCTREE_FACE_BAND_ENCODE_PHASES) this.encodePhase(broker, input, phase);
  }

  encodePhase(broker: PassBroker, input: OctreeFaceBandInput,
    phase: OctreeFaceBandEncodePhase): void {
    this.assertLive();
    if (input.fine.plan.fineFactor !== 4 && input.fine.plan.fineFactor !== 8) {
      throw new RangeError("GPU face-band topology supports global fine factors 4 and 8");
    }
    if ((input.generation >>> 0) !== (input.fine.generation >>> 0)) {
      throw new RangeError("Face-band signed distance requires the exact current fine generation");
    }
    const dims = input.dimensions.map((value) => positive(value, "Face-band dimension")) as [number, number, number];
    const maximumLeaf = positive(input.maximumLeafSize, "Face-band maximum leaf size");
    const powerDirectoryCapacity = positive(input.powerRowDirectoryCapacity,
      "Face-band power-row directory capacity");
    const powerVelocityGeneration = input.powerVelocityGeneration >>> 0;
    const tetrahedronHeaders = input.powerTopology.catalogTetrahedronHeaders;
    const tetrahedra = input.powerTopology.catalogTetrahedra;
    const tetrahedronVertices = input.powerTopology.catalogTetrahedronVertices;
    if (!tetrahedronHeaders || !tetrahedra || !tetrahedronVertices) {
      throw new RangeError("Face-band transitions require the catalog Delaunay buffers");
    }
    if (input.powerFaces.plan.faceCapacity > this.plan.powerFaceCapacity) {
      throw new RangeError("Face-band power-face publication capacity is smaller than its live source");
    }
    if (input.rowDelta.rowCapacity !== this.plan.wetRowCapacity
      || [input.rowDelta.controlOffsetWords, input.rowDelta.newToOldOffsetWords,
        input.rowDelta.oldToNewOffsetWords, input.rowDelta.dirtyRowsOffsetWords,
        input.rowDelta.affectedRowsOffsetWords]
        .some((value) => !Number.isSafeInteger(value) || value < 0)) {
      throw new RangeError("Face-band transition adjacency requires the exact power-row delta capacity");
    }
    const bind = (name: string, pipeline: GPUComputePipeline,
      entries: readonly (readonly [number, GPUBuffer])[]) =>
      this.cachedBindGroup(`Section 5 ${name} bindings`, pipeline, entries);
    const run = (name: keyof typeof this.pipelines, entries: readonly (readonly [number, GPUBuffer])[],
      workgroups: number, pass: GPUComputePassEncoder, indirectOffset?: number) => {
      const pipeline = this.pipelines[name];
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind(name, pipeline, entries));
      if (indirectOffset === undefined) pass.dispatchWorkgroups(workgroups);
      else pass.dispatchWorkgroupsIndirect(this.indirect, indirectOffset);
    };
    const computePass = (label: string, encode: (pass: GPUComputePassEncoder) => void) => {
      const pass = broker.compute({ label }); encode(pass);
    };

    switch (phase) {
      case "topology-build": {
        const words = new Uint32Array(24);
        words.set([...dims, maximumLeaf, this.plan.rowCapacity, this.plan.faceCapacity,
          this.plan.rowDirectoryCapacity, this.plan.identityToRowOffsetWords,
          powerDirectoryCapacity, this.plan.wetRowCapacity,
          input.generation >>> 0, 0, this.plan.ownerCandidatesPerBrick,
          powerVelocityGeneration, OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW, OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW]);
        // Descriptor direction bits are x-, y-, z-, z+, y+, x+. The standard
        // container closes every plane except the optional open ceiling.
        words[16] = 0b10_1111 | (input.closedTop ? 0b01_0000 : 0);
        words.set([input.rowDelta.controlOffsetWords, input.rowDelta.newToOldOffsetWords,
          input.rowDelta.affectedRowsOffsetWords, 1, input.rowDelta.oldToNewOffsetWords,
          input.rowDelta.dirtyRowsOffsetWords], 17);
        this.device.queue.writeBuffer(this.params, 0, words);
        computePass("Build exact Section 5 support-row delta", (pass) => {
          run("prepare", [[0, this.params], [1, input.fine.params], [2, input.fine.metadata],
            [3, input.fine.worklist], [4, input.powerRowDirectory], [5, this.candidateControl],
            [7, this.candidateRowDirectory], [8, input.fine.flags],
            [9, input.powerVelocityControl], [18, this.indirect],
            [61, input.rowDelta.rows]], 1, pass);
          const identityCount = dims[0] * dims[1] * dims[2] * 6;
          const identityWorkgroups = Math.ceil(identityCount / 256);
          run("clearSupportIdentityMarks", [[0, this.params],
            [67, this.catalogSupportScratch]], identityWorkgroups, pass);
          run("buildTopologyDelta", [[0, this.params], [4, input.powerRowDirectory],
            [5, this.candidateControl], [6, this.candidateRows],
            [7, this.candidateRowDirectory], [26, input.owners],
            [61, input.rowDelta.rows]], 0, pass, 240);
          run("emitCatalogSupport1", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [26, input.owners],
            [28, tetrahedronHeaders], [29, tetrahedra], [30, tetrahedronVertices],
            [33, input.powerTopology.sameOrFinerDirect],
            [34, input.powerTopology.sameOrCoarserDirect],
            [66, this.transientPowerIncidence]], 0, pass, 240);
          run("markSupport1", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [66, this.transientPowerIncidence],
            [67, this.catalogSupportScratch]], 0, pass, 240);
          run("scanSupportIdentityBlocks", [[0, this.params],
            [67, this.catalogSupportScratch]], identityWorkgroups, pass);
          run("prefixSupportIdentityBlocks", [[0, this.params],
            [67, this.catalogSupportScratch]], 1, pass);
          run("finalizeSupport1", [[0, this.params], [5, this.candidateControl],
            [7, this.candidateRowDirectory], [18, this.indirect],
            [67, this.catalogSupportScratch]], 1, pass);
          run("scatterSupport1", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [13, this.rowDirectoryScratch], [62, this.rows],
            [67, this.catalogSupportScratch], [68, this.rowDirectory]],
          identityWorkgroups, pass);
          run("emitCatalogSupport2", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [26, input.owners],
            [28, tetrahedronHeaders], [29, tetrahedra], [30, tetrahedronVertices],
            [33, input.powerTopology.sameOrFinerDirect],
            [34, input.powerTopology.sameOrCoarserDirect],
            [66, this.transientPowerIncidence]], 0, pass, 252);
          run("markSupport2", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [66, this.transientPowerIncidence],
            [67, this.catalogSupportScratch]], 0, pass, 252);
          run("scanSupportIdentityBlocks", [[0, this.params],
            [67, this.catalogSupportScratch]], identityWorkgroups, pass);
          run("prefixSupportIdentityBlocks", [[0, this.params],
            [67, this.catalogSupportScratch]], 1, pass);
          run("finalizeSupport2", [[0, this.params], [5, this.candidateControl],
            [7, this.candidateRowDirectory], [18, this.indirect],
            [67, this.catalogSupportScratch]], 1, pass);
          run("scatterSupport2", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [13, this.rowDirectoryScratch], [62, this.rows],
            [67, this.catalogSupportScratch], [68, this.rowDirectory]],
          identityWorkgroups, pass);
          run("emitCatalogSupport3", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [26, input.owners],
            [28, tetrahedronHeaders], [29, tetrahedra], [30, tetrahedronVertices],
            [33, input.powerTopology.sameOrFinerDirect],
            [34, input.powerTopology.sameOrCoarserDirect],
            [66, this.transientPowerIncidence]], 0, pass, 240);
          run("markSupport3", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [66, this.transientPowerIncidence],
            [67, this.catalogSupportScratch]], 0, pass, 240);
          run("scanSupportIdentityBlocks", [[0, this.params],
            [67, this.catalogSupportScratch]], identityWorkgroups, pass);
          run("prefixSupportIdentityBlocks", [[0, this.params],
            [67, this.catalogSupportScratch]], 1, pass);
          run("finalizeSupport3", [[0, this.params], [5, this.candidateControl],
            [7, this.candidateRowDirectory], [18, this.indirect],
            [67, this.catalogSupportScratch]], 1, pass);
          run("scatterSupport3", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [13, this.rowDirectoryScratch], [62, this.rows],
            [67, this.catalogSupportScratch], [68, this.rowDirectory]],
          identityWorkgroups, pass);
          run("emitEndpointSupport4", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [26, input.owners],
            [28, tetrahedronHeaders], [29, tetrahedra], [30, tetrahedronVertices],
            [33, input.powerTopology.sameOrFinerDirect],
            [34, input.powerTopology.sameOrCoarserDirect],
            [66, this.transientPowerIncidence]], 0, pass, 252);
          run("markSupport4", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [66, this.transientPowerIncidence],
            [67, this.catalogSupportScratch]], 0, pass, 252);
          run("scanSupportIdentityBlocks", [[0, this.params],
            [67, this.catalogSupportScratch]], identityWorkgroups, pass);
          run("prefixSupportIdentityBlocks", [[0, this.params],
            [67, this.catalogSupportScratch]], 1, pass);
          run("finalizeSupport4", [[0, this.params], [5, this.candidateControl],
            [7, this.candidateRowDirectory], [18, this.indirect],
            [67, this.catalogSupportScratch]], 1, pass);
          run("scatterSupport4", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [13, this.rowDirectoryScratch], [62, this.rows],
            [67, this.catalogSupportScratch], [68, this.rowDirectory]],
          identityWorkgroups, pass);
          run("markPublishedSupportRows", [[0, this.params],
            [5, this.candidateControl], [6, this.candidateRows],
            [67, this.catalogSupportScratch]], 0, pass, 12);
          run("scanSupportIdentityBlocks", [[0, this.params],
            [67, this.catalogSupportScratch]], identityWorkgroups, pass);
          run("prefixSupportIdentityBlocks", [[0, this.params],
            [67, this.catalogSupportScratch]], 1, pass);
          run("scatterCanonicalRowDirectory", [[0, this.params],
            [5, this.candidateControl], [6, this.candidateRows],
            [7, this.candidateRowDirectory], [67, this.catalogSupportScratch]],
          identityWorkgroups, pass);
          run("validateRowDirectory", [[5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory]], 0, pass, 12);
          run("classifyRowSigns", [[0, this.params],
            [5, this.candidateControl], [6, this.candidateRows],
            [25, input.coarsePhiDirectory], [26, input.owners]], 0, pass, 12);
        });
        return;
      }
      case "transition-adjacency": {
        // The B-side row candidate is complete, but the prior A-side authority
        // remains immutable until faces, incidence, phi, and adjacency all
        // validate. Resolution never appends rows or selects a full rebuild.
        computePass("Classify exact Section 5 catalog-adjacency delta", (pass) => {
          run("prepareTransition", [[0, this.params], [5, this.candidateControl],
            [7, this.candidateRowDirectory],
            [18, this.indirect], [32, this.candidateTransitionControl], [61, input.rowDelta.rows]], 1, pass);
          run("classifyTransitionDelta", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows],
            [7, this.candidateRowDirectory], [26, input.owners], [32, this.candidateTransitionControl],
            [47, this.pointStatus], [61, input.rowDelta.rows], [62, this.rows],
            [65, this.transitionDeltaScan], [68, this.rowDirectory]],
          0, pass, 12);
          run("carryTransitionAdjacency", [[0, this.params], [6, this.candidateRows],
            [7, this.candidateRowDirectory], [27, this.candidateTransitionMetrics],
            [31, this.candidateTransitionAdjacency], [32, this.candidateTransitionControl],
            [61, input.rowDelta.rows], [62, this.rows], [63, this.transitionMetrics],
            [64, this.transitionAdjacency], [68, this.rowDirectory]],
          0, pass, 12);
        });
        computePass("Resolve exact affected Section 5 catalog adjacency", (pass) => {
          run("describeCatalogRows", [[0, this.params], [6, this.candidateRows],
            [26, input.owners],
            [27, this.candidateTransitionMetrics], [32, this.candidateTransitionControl],
            [33, input.powerTopology.sameOrFinerDirect],
            [34, input.powerTopology.sameOrCoarserDirect]],
          0, pass, 204);
          run("resolveCatalogAdjacency", [[0, this.params], [6, this.candidateRows],
            [7, this.candidateRowDirectory], [26, input.owners],
            [27, this.candidateTransitionMetrics],
            [28, tetrahedronHeaders], [29, tetrahedra],
            [30, tetrahedronVertices], [31, this.candidateTransitionAdjacency],
            [32, this.candidateTransitionControl]],
          0, pass, 204);
        });
        computePass("Validate and publish Section 5 catalog adjacency", (pass) => {
          run("validateCatalogAdjacency", [[0, this.params], [6, this.candidateRows],
            [7, this.candidateRowDirectory], [27, this.candidateTransitionMetrics],
            [28, tetrahedronHeaders], [31, this.candidateTransitionAdjacency],
            [32, this.candidateTransitionControl]],
          0, pass, 12);
          run("publishCatalogAdjacency", [[0, this.params], [6, this.candidateRows],
            [18, this.indirect], [27, this.candidateTransitionMetrics],
            [32, this.candidateTransitionControl], [65, this.transitionDeltaScan]], 1, pass);
        });
        computePass("Publish Section 5 regular-face topology", (pass) => {
          run("clearBandPhiEndpointEdges", [[32, this.candidateTransitionControl],
            [53, this.transientPowerIncidence], [73, this.endpointIncomingCounts]],
          0, pass, 216);
          run("buildBandPhiEdges", [[0, this.params], [6, this.candidateRows],
            [7, this.candidateRowDirectory], [26, input.owners],
            [27, this.candidateTransitionMetrics], [31, this.candidateTransitionAdjacency],
            [32, this.candidateTransitionControl],
            [53, this.transientPowerIncidence], [73, this.endpointIncomingCounts]], 0, pass, 216);
          run("buildBandPhiEndpointEdges", [[6, this.candidateRows],
            [32, this.candidateTransitionControl],
            [53, this.transientPowerIncidence], [73, this.endpointIncomingCounts]], 0, pass, 216);
          run("carryFaceTopology", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [12, this.candidateFaces],
            [32, this.candidateTransitionControl], [62, this.rows],
            [65, this.transitionDeltaScan], [69, this.faces],
          ], 0, pass, 216);
          // Publish every affected owner slot before validating incidence
          // carried by an unchanged endpoint. A shared face is owned by one
          // row but occurs in both endpoint lists, so the endpoint may be
          // unchanged while its owner is in the exact face delta.
          run("emit", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [12, this.candidateFaces], [26, input.owners],
            [32, this.candidateTransitionControl], [65, this.transitionDeltaScan]],
          0, pass, 204);
          run("carryFaceIncidence", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [12, this.candidateFaces], [14, this.candidateIncidence],
            [32, this.candidateTransitionControl],
            [62, this.rows], [65, this.transitionDeltaScan], [70, this.incidence]], 0, pass, 216);
          run("rebuildIncidence", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [12, this.candidateFaces], [14, this.candidateIncidence],
            [26, input.owners], [32, this.candidateTransitionControl],
            [65, this.transitionDeltaScan]], 0, pass, 204);
          run("publishFaceCounts", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [12, this.candidateFaces], [14, this.candidateIncidence],
            [18, this.indirect], [32, this.candidateTransitionControl],
            [75, this.liveFaceWorklist]], 1, pass);
          // Paper Section 5 orders regular faces by closest-interface distance.
          // Give every emitted live face its signed-distance value at the
          // actual face centroid from the same current fine SPGrid generation.
          run("sampleFacePhi", [[0, this.params], [1, input.fine.params], [2, input.fine.metadata],
            [3, input.fine.worklist],
            [5, this.candidateControl], [8, input.fine.flags],
            [12, this.candidateFaces], [24, input.fine.phi], [75, this.liveFaceWorklist],
          ], 0, pass, 228);
          // The fine SPGrid is intentionally narrow. Build the paper's coarse
          // signed-distance field on the complete transient owner graph before
          // evaluating dry regular-face centroids. Existing row-velocity
          // scratch is reused here and cleared before closest-point extension.
          run("initializeBandPhi", [[0, this.params], [1, input.fine.params], [2, input.fine.metadata],
            [3, input.fine.worklist],
            [5, this.candidateControl], [6, this.candidateRows],
            [8, input.fine.flags], [19, this.velocities],
            [24, input.fine.phi], [25, input.coarsePhiDirectory], [42, input.fineTopologyControl]],
          0, pass, 216);
          run("seedBandPhiFaces", [[0, this.params], [1, input.fine.params],
            [5, this.candidateControl], [6, this.candidateRows],
            [12, this.candidateFaces], [14, this.candidateIncidence], [19, this.velocities]],
          0, pass, 216);
          run("collectBandPhiFrontier", [[5, this.candidateControl],
            [19, this.velocities], [44, this.provisionalVelocities],
            [18, this.indirect], [74, this.bandPhiFrontier]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          let currentPhi = this.velocities, nextPhi = this.provisionalVelocities;
          for (let round = 0; round < this.bandPhiRelaxationRounds; round += 1) {
            run("extendBandPhi", [[0, this.params], [1, input.fine.params],
              [5, this.candidateControl], [6, this.candidateRows],
              [12, this.candidateFaces], [14, this.candidateIncidence],
              [19, currentPhi], [27, this.candidateTransitionMetrics],
              [31, this.candidateTransitionAdjacency], [44, nextPhi],
              [53, this.transientPowerIncidence], [74, this.bandPhiFrontier]],
            0, pass, 252);
            [currentPhi, nextPhi] = [nextPhi, currentPhi];
          }
          run("commitBandPhi", [[5, this.candidateControl], [6, this.candidateRows],
            [19, currentPhi], [32, this.candidateTransitionControl]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("sampleFaceCoarsePhi", [[0, this.params], [1, input.fine.params],
            [6, this.candidateRows],
            [7, this.candidateRowDirectory], [12, this.candidateFaces],
            [27, this.candidateTransitionMetrics], [28, tetrahedronHeaders], [29, tetrahedra],
            [30, tetrahedronVertices], [31, this.candidateTransitionAdjacency],
            [32, this.candidateTransitionControl], [75, this.liveFaceWorklist]],
          0, pass, 228);
          run("reducePhiFailure", [[12, this.candidateFaces], [32, this.candidateTransitionControl],
            [75, this.liveFaceWorklist]],
            1, pass);
          run("summarizeRowPhi", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [12, this.candidateFaces],
            [14, this.candidateIncidence], [32, this.candidateTransitionControl]],
          0, pass, 216);
          // Face emission can still discover a missing endpoint/capacity
          // fault. Publish transition readiness only after that final writer.
          run("gateTransition", [[5, this.candidateControl],
            [32, this.candidateTransitionControl]], 1, pass);
        });
        return;
      }
      case "closest-point-extension":
        computePass("Extrapolate Section 5 regular-face velocities", (pass) => {
          run("seedCentroids", [[0, this.params], [6, this.candidateRows],
            [7, this.candidateRowDirectory], [10, input.powerRowVelocities],
            [12, this.candidateFaces], [27, this.candidateTransitionMetrics],
            [28, tetrahedronHeaders], [29, tetrahedra], [30, tetrahedronVertices],
            [75, this.liveFaceWorklist]],
          0, pass, 228);
          run("extendClosestPoints", [[0, this.params],
            [6, this.candidateRows], [7, this.candidateRowDirectory],
            [12, this.candidateFaces], [14, this.candidateIncidence],
            [26, input.owners], [27, this.candidateTransitionMetrics], [28, tetrahedronHeaders],
            [29, tetrahedra], [30, tetrahedronVertices], [75, this.liveFaceWorklist]],
            0, pass, 228);
          // The paper's face marcher copies the velocity of the face closest
          // to the free surface. Eight fixed dependency waves close the
          // catalog's bounded support radius: each wave publishes immutable
          // inner carriers for the next while moving strictly outward in
          // positive phi. This is neither a convergence loop nor an
          // arena-wide repair.
          for (let wave = 0; wave < 8; wave += 1) {
            run("gatherClosestPointRepairs", [[0, this.params], [1, input.fine.params],
              [6, this.candidateRows],
              [12, this.candidateFaces], [14, this.candidateIncidence],
              [32, this.candidateTransitionControl], [67, this.catalogSupportScratch],
              [75, this.liveFaceWorklist]],
            0, pass, 228);
            run("commitClosestPointRepairs", [[0, this.params], [1, input.fine.params],
              [6, this.candidateRows],
              [12, this.candidateFaces], [14, this.candidateIncidence],
              [32, this.candidateTransitionControl], [67, this.catalogSupportScratch],
              [75, this.liveFaceWorklist]],
            0, pass, 228);
          }
          run("reconstruct", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows], [12, this.candidateFaces],
            [14, this.candidateIncidence], [19, this.velocities],
            [32, this.candidateTransitionControl],
            [44, this.provisionalVelocities]],
          0, pass, 216);
          // Aanjaneya et al. (2017), Section 5 first constructs a full vector
          // at every liquid octree-cell centre from the generalized power-face
          // normal velocities. Closest-point extension only extrapolates that
          // field outside the liquid.  Restore those authoritative vectors
          // after support closure so an auxiliary zero-incidence support row
          // cannot replace a real power-cell value.
          run("seedMappedPowerRows", [[0, this.params], [5, this.candidateControl],
            [6, this.candidateRows],
            [9, input.powerVelocityControl], [10, input.powerRowVelocities],
            [19, this.velocities], [32, this.candidateTransitionControl],
            [44, this.provisionalVelocities]],
          0, pass, 216);
          run("finalizeClosestPoint", [[5, this.candidateControl],
            [6, this.candidateRows], [12, this.candidateFaces],
            [32, this.candidateTransitionControl], [44, this.provisionalVelocities],
            [75, this.liveFaceWorklist], [76, this.publicationReductions]],
          1, pass);
          run("commitRows", [[0, this.params], [5, this.candidateControl], [6, this.candidateRows],
            [7, this.candidateRowDirectory], [32, this.candidateTransitionControl],
            [62, this.rows], [68, this.rowDirectory]],
          0, pass, 216);
          run("commitFaces", [[0, this.params], [5, this.candidateControl],
            [12, this.candidateFaces], [14, this.candidateIncidence],
            [32, this.candidateTransitionControl], [69, this.faces],
            [70, this.incidence]], 0, pass, 216);
          run("commitTransition", [[0, this.params], [5, this.candidateControl],
            [27, this.candidateTransitionMetrics], [31, this.candidateTransitionAdjacency],
            [32, this.candidateTransitionControl], [63, this.transitionMetrics],
            [64, this.transitionAdjacency]], 0, pass, 216);
          run("commitControls", [[0, this.params], [5, this.candidateControl],
            [32, this.candidateTransitionControl], [68, this.rowDirectory], [71, this.control],
            [72, this.transitionControl]], 1, pass);
        });
        return;
      case "power-publication":
        // This graph is a fixed-address, counted publication. Each live row
        // overwrites its row descriptor and every catalog incidence slot has
        // exactly one current writer. Count-delimited authority makes stale
        // capacity tails unreachable without arena clears.
        computePass("Publish Section 5 regular velocities to power faces", (pass) => {
          run("prepareTransientPower", [[0, this.params], [5, this.control], [18, this.indirect], [32, this.transitionControl],
            [52, this.transientPowerFaces], [53, this.transientPowerIncidence],
            [54, this.transientPowerRows], [55, this.transientPowerControl]], 1, pass);
          run("emitTransientPower", [[0, this.params], [6, this.rows], [7, this.rowDirectory],
            [11, input.powerTopology.catalogEntryHeaders], [27, this.transitionMetrics],
            [32, this.transitionControl], [45, input.powerTopology.catalogFaces], [52, this.transientPowerFaces],
            [53, this.transientPowerIncidence], [54, this.transientPowerRows],
            [55, this.transientPowerControl]], 0, pass, 216);
          run("sampleTransientPower", [[0, this.params], [6, this.rows],
            [7, this.rowDirectory], [27, this.transitionMetrics], [28, tetrahedronHeaders], [29, tetrahedra],
            [30, tetrahedronVertices], [44, this.provisionalVelocities],
            [52, this.transientPowerFaces], [54, this.transientPowerRows],
            [55, this.transientPowerControl]], 0, pass, 216);
          run("validateTransientPower", [[52, this.transientPowerFaces], [53, this.transientPowerIncidence],
            [54, this.transientPowerRows], [55, this.transientPowerControl]],
          0, pass, 216);
          // The paper's final cell-centre field is reconstructed only from the
          // complete transient all-band physical generalized-face graph.
          run("preparePointField", [[0, this.params], [18, this.indirect],
            [32, this.transitionControl], [48, this.pointFieldControl],
            [55, this.transientPowerControl], [76, this.publicationReductions]], 1, pass);
          run("reducePointField", [[52, this.transientPowerFaces], [54, this.transientPowerRows],
            [55, this.transientPowerControl], [76, this.publicationReductions]],
          this.plan.maximumDirectWorkgroups, pass);
          run("finalizePointField", [[0, this.params], [5, this.control], [18, this.indirect],
            [32, this.transitionControl], [48, this.pointFieldControl],
            [52, this.transientPowerFaces], [55, this.transientPowerControl],
            [76, this.publicationReductions]], 1, pass);
          run("reconstructPhysicalPoint", [[0, this.params], [19, this.velocities],
            [47, this.pointStatus], [48, this.pointFieldControl], [52, this.transientPowerFaces],
            [53, this.transientPowerIncidence], [54, this.transientPowerRows],
            [55, this.transientPowerControl]],
          0, pass, 216);
          run("publishPoint", [[0, this.params], [47, this.pointStatus], [48, this.pointFieldControl],
            [55, this.transientPowerControl]], 1, pass);
          // Only the completed point field is allowed to return to the
          // production power faces. This breaks the former circular dependency
          // where those faces were required before the point field existed.
          run("preparePowerPublication", [[0, this.params], [5, this.control], [32, this.transitionControl],
            [36, input.powerFaces.control], [37, input.powerFaces.faces], [38, input.powerFaces.faceNormals],
            [39, input.powerFaces.faceCentroids], [18, this.indirect], [40, this.powerVelocityScratch],
            [41, this.powerPublicationControl], [48, this.pointFieldControl]], 1, pass);
          run("mapPowerFaceBands", [[5, this.control], [6, this.rows],
            [37, input.powerFaces.faces], [40, this.powerVelocityScratch], [41, this.powerPublicationControl]],
          0, pass, 216);
          run("interpolatePowerFaces", [[0, this.params], [1, input.fine.params], [6, this.rows], [7, this.rowDirectory],
            [19, this.velocities], [27, this.transitionMetrics], [28, tetrahedronHeaders], [29, tetrahedra],
            [30, tetrahedronVertices], [39, input.powerFaces.faceCentroids], [40, this.powerVelocityScratch],
            [41, this.powerPublicationControl]], 0, pass, 216);
          run("projectPowerFaces", [[0, this.params], [37, input.powerFaces.faces], [38, input.powerFaces.faceNormals],
            [40, this.powerVelocityScratch], [41, this.powerPublicationControl]],
          0, pass, 216);
          run("commitPowerFaces", [[5, this.control], [37, input.powerFaces.faces], [40, this.powerVelocityScratch],
            [41, this.powerPublicationControl]], 1, pass);
        });
        return;
      default: phase satisfies never;
    }
  }

  /** Publish the exact immutable air-sampler parameters for an enclosing fused
   * characteristic kernel. No recurrent counters or topology bytes change. */
  prepareFusedAirSampling(options: OctreeFaceBandSampleOptions): number {
    const count = positive(options.queryCount, "Face-band sample count");
    if (!Number.isFinite(options.physicalCellSize) || options.physicalCellSize <= 0) {
      throw new RangeError("Face-band sample physical cell size must be finite and positive");
    }
    const data = new ArrayBuffer(48), words = new Uint32Array(data), floats = new Float32Array(data);
    words.set([...options.dimensions, positive(options.maximumLeafSize, "Face-band sample maximum leaf"), 0,
      count, this.plan.rowDirectoryCapacity, this.plan.rowCapacity]); floats[8] = options.physicalCellSize;
    words[9] = options.fineGeneration >>> 0;
    this.device.queue.writeBuffer(this.sampleParams, 0, data);
    return count;
  }

  encodeFusedAuthority(broker: PassBroker): void {
    const pipeline = this.fusedAuthorityPipeline;
    const pass = broker.compute({ label: "Publish grouped face-band transport authority" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.fusedAuthorityBindGroup);
    pass.dispatchWorkgroups(this.fusedAuthorityWorkgroups);
  }

  get fusedSamplerSource(): OctreeFaceBandFusedSamplerSource {
    return {
      params: this.params, sampleParams: this.sampleParams,
      authority: this.fusedAuthority, regions: this.plan.fusedAuthorityRegions,
    };
  }

  /** Complete every recurrent Section 5 characteristic left pending by the
   * transition-specialized old-power interpolant. The retained regular-face
   * band is the paper's mandatory regular/air-side interpolation domain. */
  encodeCompletePowerFaceAdvectionFromRegularBand(broker: PassBroker, input: {
    faces: OctreePowerFaceSource;
    advectionControl: GPUBuffer;
    seedControl: GPUBuffer;
    dimensions: readonly [number, number, number];
    maximumLeafSize: number;
    physicalCellSize: number;
    timestep: number;
    fineGeneration: number;
    powerGeneration: number;
    powerTopology: OctreePowerTopologySource;
    closedTop: boolean;
  }): void {
    this.assertLive();
    if (input.powerGeneration <= 1) {
      throw new RangeError("Regular-face completion requires a recurring power generation");
    }
    const data = new ArrayBuffer(48), words = new Uint32Array(data), floats = new Float32Array(data);
    words.set([...input.dimensions, positive(input.maximumLeafSize, "Old face-band maximum leaf"),
      0, 0, this.plan.rowDirectoryCapacity, this.plan.rowCapacity]);
    floats[8] = input.physicalCellSize;
    words[9] = input.fineGeneration >>> 0;
    floats[10] = input.timestep;
    words[11] = input.powerGeneration >>> 0;
    this.device.queue.writeBuffer(this.completionParams, 0, data);
    const topologyWords = new Uint32Array(24);
    topologyWords.set([...input.dimensions, input.maximumLeafSize, this.plan.rowCapacity,
      this.plan.faceCapacity, this.plan.rowDirectoryCapacity, this.plan.identityToRowOffsetWords,
      input.faces.plan.rowDirectoryCapacity, this.plan.wetRowCapacity, input.fineGeneration >>> 0,
      0, this.plan.ownerCandidatesPerBrick,
      input.powerGeneration >>> 0, OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW,
      OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW]);
    topologyWords[16] = 0b10_1111 | (input.closedTop ? 0b01_0000 : 0);
    this.device.queue.writeBuffer(this.completionTopologyParams, 0, topologyWords);
    const common: (readonly [number, GPUBuffer])[] = [
      [0, this.completionTopologyParams],
      [5, this.control],
      [6, this.rows],
      [7, this.rowDirectory],
      [19, this.velocities],
      [20, this.completionParams],
      [27, this.transitionMetrics],
    ];
    // Catalog buffers are owned by the same immutable topology authority used
    // when this old band was published.
    const topology = input.powerTopology;
    if (!topology?.catalogTetrahedronHeaders || !topology.catalogTetrahedra
      || !topology.catalogTetrahedronVertices) {
      throw new RangeError("Regular-face completion requires the Section 5 Delaunay catalog");
    }
    common.push(
      [28, topology.catalogTetrahedronHeaders],
      [29, topology.catalogTetrahedra],
      [30, topology.catalogTetrahedronVertices],
      [32, this.transitionControl],
      [36, input.faces.control],
      [37, input.faces.faces],
      [38, input.faces.faceNormals],
      [39, input.faces.faceCentroids],
      [48, this.pointFieldControl],
      [58, input.advectionControl],
      [59, input.seedControl],
    );
    const bind = (pipeline: GPUComputePipeline, bindings: readonly number[]) =>
      this.cachedBindGroup("Section 5 regular-face completion bindings", pipeline,
        common.filter(([binding]) => bindings.includes(binding)));
    const prepare = broker.compute({ label: "Prepare mandatory Section 5 regular-face completion" });
    prepare.setPipeline(this.preparePowerAdvectionCompletionPipeline);
    prepare.setBindGroup(0, bind(this.preparePowerAdvectionCompletionPipeline,
      [5, 20, 36, 37, 38, 39, 48, 58, 59]));
    prepare.dispatchWorkgroups(1);
    const complete = broker.compute({ label: "Complete power-face advection from Section 5 regular band" });
    complete.setPipeline(this.completePowerAdvectionPipeline);
    complete.setBindGroup(0, bind(this.completePowerAdvectionPipeline,
      [0, 6, 7, 19, 20, 27, 28, 29, 30, 36, 38, 39]));
    complete.dispatchWorkgroupsIndirect(input.faces.liveFaceDispatch, 0);
    const publish = broker.compute({ label: "Publish completed Section 5 power-face advection" });
    publish.setPipeline(this.publishPowerAdvectionCompletionPipeline);
    publish.setBindGroup(0, bind(this.publishPowerAdvectionCompletionPipeline, [37, 38, 39, 58, 59]));
    publish.dispatchWorkgroups(1);
  }

  get source(): OctreeFaceBandSource { return { plan: this.plan, control: this.control, rows: this.rows,
    rowDirectory: this.rowDirectory, faces: this.faces, incidence: this.incidence, velocities: this.velocities,
    transitionAdjacency: this.transitionAdjacency, transitionControl: this.transitionControl,
    transitionMetrics: this.transitionMetrics, powerPublicationControl: this.powerPublicationControl,
    pointFieldControl: this.pointFieldControl, transientPowerControl: this.transientPowerControl }; }
  /** Rejection-only visibility into the unpublished current Section 5
   * transaction. These controls never participate in authority selection. */
  get candidateControlForDiagnostics(): GPUBuffer { return this.candidateControl; }
  get candidateTransitionControlForDiagnostics(): GPUBuffer { return this.candidateTransitionControl; }
  /** Rejection-only audit for the exact `sum(row incidence) == 2 * live faces`
   * publication invariant. No simulation decision consumes this readback. */
  async readCandidateBandIncidenceFailure(rowCount: number) {
    if (!Number.isSafeInteger(rowCount) || rowCount <= 0 || rowCount > this.plan.rowCapacity) {
      return undefined;
    }
    const owned = OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW;
    const stride = OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW;
    const faceSlots = rowCount * owned;
    const faceBytes = faceSlots * OCTREE_FACE_BAND_FACE_BYTES;
    const countBytes = rowCount * 4;
    const itemBytes = rowCount * stride * 4;
    const readback = this.device.createBuffer({
      label: "Rejected face-band incidence invariant",
      size: faceBytes + countBytes + itemBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read rejected face-band incidence invariant",
    });
    encoder.copyBufferToBuffer(this.candidateFaces, 0, readback, 0, faceBytes);
    encoder.copyBufferToBuffer(this.candidateIncidence, 0, readback, faceBytes, countBytes);
    encoder.copyBufferToBuffer(this.candidateIncidence, this.plan.rowCapacity * 4,
      readback, faceBytes + countBytes, itemBytes);
    this.device.queue.submit([encoder.finish()]);
    let snapshot: ArrayBuffer;
    try {
      await readback.mapAsync(GPUMapMode.READ);
      snapshot = readback.getMappedRange().slice(0);
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
    const words = new Uint32Array(snapshot);
    const faceWords = OCTREE_FACE_BAND_FACE_BYTES / 4;
    const countsBase = faceBytes / 4;
    const itemsBase = countsBase + rowCount;
    const occurrences = new Uint32Array(faceSlots);
    let incidenceCount = 0;
    let firstInvalidIncidence: { row: number; local: number; face: number } | undefined;
    for (let row = 0; row < rowCount; row += 1) {
      const count = Math.min(words[countsBase + row] ?? 0, stride);
      incidenceCount += count;
      for (let local = 0; local < count; local += 1) {
        const face = words[itemsBase + row * stride + local] >>> 0;
        if (face < faceSlots) occurrences[face] += 1;
        else if (!firstInvalidIncidence) firstInvalidIncidence = { row, local, face };
      }
    }
    let topologyFaceCount = 0;
    let firstFace: number | undefined;
    for (let face = 0; face < faceSlots; face += 1) {
      const live = ((words[face * faceWords + 14] ?? 0) & 1) !== 0;
      if (!live) continue;
      topologyFaceCount += 1;
      if (occurrences[face] !== 2 && firstFace === undefined) firstFace = face;
    }
    const detail = firstFace === undefined
      ? undefined : await this.readBandFaceFailure(firstFace, true);
    return {
      rowCount, topologyFaceCount, incidenceCount,
      expectedIncidenceCount: 2 * topologyFaceCount,
      firstInvalidIncidence,
      firstFace: firstFace === undefined ? undefined : {
        slot: firstFace, occurrences: occurrences[firstFace], detail,
      },
    };
  }
  /** Bounded QA readback for the retained-band row named by an advection failure. */
  async readBandRowFailure(index: number, candidate = false) {
    if (!Number.isSafeInteger(index) || index < 0 || index >= this.plan.rowCapacity) return undefined;
    const rowSource = candidate ? this.candidateRows : this.rows;
    const metricSource = candidate ? this.candidateTransitionMetrics : this.transitionMetrics;
    const incidenceSource = candidate ? this.candidateIncidence : this.incidence;
    const faceSource = candidate ? this.candidateFaces : this.faces;
    const incidenceWords = 1 + OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW;
    const fixedBytes = OCTREE_FACE_BAND_ROW_BYTES + 16 + 16 + 16 + incidenceWords * 4;
    const fixed = this.device.createBuffer({ label: "Face-band retained-row failure", size: fixedBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read face-band retained row" });
    let offset = 0;
    encoder.copyBufferToBuffer(rowSource, index * OCTREE_FACE_BAND_ROW_BYTES,
      fixed, offset, OCTREE_FACE_BAND_ROW_BYTES); offset += OCTREE_FACE_BAND_ROW_BYTES;
    encoder.copyBufferToBuffer(metricSource, index * 16, fixed, offset, 16); offset += 16;
    encoder.copyBufferToBuffer(this.velocities, index * 16, fixed, offset, 16); offset += 16;
    encoder.copyBufferToBuffer(this.provisionalVelocities, index * 16, fixed, offset, 16); offset += 16;
    encoder.copyBufferToBuffer(incidenceSource, index * 4, fixed, offset, 4);
    encoder.copyBufferToBuffer(incidenceSource,
      (this.plan.rowCapacity + index * OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW) * 4,
      fixed, offset + 4, (incidenceWords - 1) * 4);
    this.device.queue.submit([encoder.finish()]);
    let snapshot: ArrayBuffer;
    try {
      await fixed.mapAsync(GPUMapMode.READ);
      snapshot = fixed.getMappedRange().slice(0);
    } finally {
      if (fixed.mapState === "mapped") fixed.unmap();
      fixed.destroy();
    }
    const words = new Uint32Array(snapshot), floats = new Float32Array(snapshot);
    const rowWords = OCTREE_FACE_BAND_ROW_BYTES / 4;
    const metricBase = rowWords, velocityBase = metricBase + 4;
    const provisionalBase = velocityBase + 4, incidenceBase = provisionalBase + 4;
    const count = Math.min(words[incidenceBase], OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW);
    const faceIndices = Array.from(words.slice(incidenceBase + 1, incidenceBase + 1 + count));
    const liveFaceIndices = faceIndices.filter((face) => face < this.plan.faceCapacity);
    const faceBytes = liveFaceIndices.length * OCTREE_FACE_BAND_FACE_BYTES;
    let faces: unknown[] = [];
    if (faceBytes > 0) {
      const faceReadback = this.device.createBuffer({ label: "Face-band retained-row incident faces",
        size: faceBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const faceEncoder = this.device.createCommandEncoder({ label: "Read retained-row incident faces" });
      liveFaceIndices.forEach((face, local) => faceEncoder.copyBufferToBuffer(faceSource,
        face * OCTREE_FACE_BAND_FACE_BYTES, faceReadback, local * OCTREE_FACE_BAND_FACE_BYTES,
        OCTREE_FACE_BAND_FACE_BYTES));
      this.device.queue.submit([faceEncoder.finish()]);
      try {
        await faceReadback.mapAsync(GPUMapMode.READ);
        const bytes = faceReadback.getMappedRange().slice(0);
        const faceWords = new Uint32Array(bytes), faceFloats = new Float32Array(bytes);
        const stride = OCTREE_FACE_BAND_FACE_BYTES / 4;
        faces = liveFaceIndices.map((face, local) => { const base = local * stride; return { face,
          negativeRow: faceWords[base], positiveRow: faceWords[base + 1], axisSpan: faceWords[base + 2],
          globalFace: faceWords[base + 3], velocity: Array.from(faceFloats.slice(base + 4, base + 8)),
          centroid: Array.from(faceFloats.slice(base + 8, base + 12)), phi: faceFloats[base + 12],
          area: faceFloats[base + 13], flags: faceWords[base + 14] }; });
      } finally {
        if (faceReadback.mapState === "mapped") faceReadback.unmap();
        faceReadback.destroy();
      }
    }
    return { index,
      row: { cell: words[0], globalRow: words[1], flags: words[2], size: words[3],
        representativePhi: floats[4], minimumPhi: floats[5], maximumPhi: floats[6], padf: floats[7] },
      metric: { topology: words[metricBase], transformFlags: words[metricBase + 1],
        volume: floats[metricBase + 2], reserved: words[metricBase + 3] },
      velocity: Array.from(floats.slice(velocityBase, velocityBase + 4)),
      provisionalVelocity: Array.from(floats.slice(provisionalBase, provisionalBase + 4)),
      incidenceCount: words[incidenceBase], faceIndices, faces };
  }
  /** Decode the tagged Eikonal failure id and read its realized row-local
   * tetrahedron. This proves whether the immutable catalog geometry was
   * mis-realized by the current compact row graph. */
  async readBandAcuteTetraFailure(tagged: number, candidate = false) {
    const packed = (tagged >>> 0) & ~OCTREE_FACE_BAND_ACUTE_TETRA_FAILURE;
    const stride = OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra;
    const row = Math.floor(packed / stride), local = packed % stride;
    if ((tagged >>> 0 & OCTREE_FACE_BAND_ACUTE_TETRA_FAILURE) === 0
      || row >= this.plan.rowCapacity) return undefined;
    const readback = this.device.createBuffer({ label: "Face-band acute tetra failure", size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read face-band acute tetra" });
    const adjacencySource = candidate ? this.candidateTransitionAdjacency : this.transitionAdjacency;
    encoder.copyBufferToBuffer(adjacencySource, (row * stride + local) * 16,
      readback, 0, 16);
    this.device.queue.submit([encoder.finish()]);
    let snapshot: ArrayBuffer;
    try {
      await readback.mapAsync(GPUMapMode.READ);
      snapshot = readback.getMappedRange().slice(0);
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
    const record = new Uint32Array(snapshot);
    const ids = [...new Set([row, record[1], record[2], record[3]]
      .filter((value) => value !== 0xffff_ffff && value < this.plan.rowCapacity))];
    const rows: unknown[] = [];
    for (const id of ids) rows.push(await this.readBandRowFailure(id, candidate));
    return { tagged: tagged >>> 0, row, local,
      adjacency: { band: record[0], a: record[1], b: record[2], c: record[3] }, rows };
  }
  /** Direct, bounded QA readback for the face-arena slot published by the
   * stage-local CPT diagnostics. The arena slot is authoritative for lookup;
   * the record itself carries the stable global-face identity. */
  async readBandFaceFailure(slot: number, candidate = false) {
    if (!Number.isSafeInteger(slot) || slot < 0 || slot >= this.plan.faceCapacity) return undefined;
    const readback = this.device.createBuffer({ label: "Face-band CPT failure face",
      size: OCTREE_FACE_BAND_FACE_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read face-band CPT failure face" });
    const faceSource = candidate ? this.candidateFaces : this.faces;
    encoder.copyBufferToBuffer(faceSource, slot * OCTREE_FACE_BAND_FACE_BYTES,
      readback, 0, OCTREE_FACE_BAND_FACE_BYTES);
    this.device.queue.submit([encoder.finish()]);
    let snapshot: ArrayBuffer;
    try {
      await readback.mapAsync(GPUMapMode.READ);
      snapshot = readback.getMappedRange().slice(0);
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
    const words = new Uint32Array(snapshot), floats = new Float32Array(snapshot);
    const negativeRow = words[0] >>> 0, positiveRow = words[1] >>> 0;
    const rowIds = [...new Set([negativeRow, positiveRow]
      .filter((row) => row !== 0xffff_ffff && row < this.plan.rowCapacity))];
    const rows: unknown[] = [];
    for (const row of rowIds) rows.push(await this.readBandRowFailure(row, candidate));
    const encodedAnchor = floats[7], decodedAnchor = -encodedAnchor;
    const roundedAnchor = Math.round(decodedAnchor);
    const closestPointAnchor = Number.isFinite(decodedAnchor) && decodedAnchor >= 1
      && decodedAnchor <= 16_777_216 && Math.abs(decodedAnchor - roundedAnchor) <= 1e-5
      ? roundedAnchor - 1 : undefined;
    const closestPointAnchorDetail = closestPointAnchor !== undefined
      && closestPointAnchor < this.plan.rowCapacity
      ? await this.readBandRowFailure(closestPointAnchor, candidate) : undefined;
    const anchorFaces = (closestPointAnchorDetail as {
      faces?: readonly { negativeRow?: number; positiveRow?: number }[];
    } | undefined)?.faces ?? [];
    const closestPointAnchorOppositeRowIds: number[] = [];
    for (const face of anchorFaces) {
      const row = face.negativeRow === closestPointAnchor ? face.positiveRow
        : face.positiveRow === closestPointAnchor ? face.negativeRow : undefined;
      if (row === undefined || !Number.isSafeInteger(row) || row === 0xffff_ffff
        || row < 0 || row >= this.plan.rowCapacity
        || closestPointAnchorOppositeRowIds.includes(row)) continue;
      closestPointAnchorOppositeRowIds.push(row);
    }
    const closestPointAnchorOppositeRows: unknown[] = [];
    for (const row of closestPointAnchorOppositeRowIds) {
      closestPointAnchorOppositeRows.push(await this.readBandRowFailure(row, candidate));
    }
    return {
      slot,
      face: {
        negativeRow, positiveRow, axisSpan: words[2] >>> 0, globalFace: words[3] >>> 0,
        velocity: Array.from(floats.slice(4, 8)), centroid: Array.from(floats.slice(8, 12)),
        phi: floats[12], area: floats[13], flags: words[14] >>> 0, pad: words[15] >>> 0,
      },
      rows, closestPointAnchor, closestPointAnchorDetail, closestPointAnchorOppositeRows,
    };
  }
  /** Exact failure-only readback for a physical face in the transient
   * Section 5 generalized-face graph. Unlike the compact regular-face arena,
   * the transient graph is addressed by fixed row-local catalog slots. */
  async readTransientPowerFaceFailure(slot: number) {
    if (!Number.isSafeInteger(slot) || slot < 0 || slot >= this.plan.transientPowerFaceCapacity) {
      return undefined;
    }
    const readback = this.device.createBuffer({
      label: "Face-band transient power failure face",
      size: OCTREE_FACE_BAND_TRANSIENT_POWER_FACE_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read transient physical power face" });
    encoder.copyBufferToBuffer(this.transientPowerFaces,
      slot * OCTREE_FACE_BAND_TRANSIENT_POWER_FACE_BYTES, readback, 0,
      OCTREE_FACE_BAND_TRANSIENT_POWER_FACE_BYTES);
    this.device.queue.submit([encoder.finish()]);
    let snapshot: ArrayBuffer;
    try {
      await readback.mapAsync(GPUMapMode.READ);
      snapshot = readback.getMappedRange().slice(0);
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
    const words = new Uint32Array(snapshot), floats = new Float32Array(snapshot);
    const negativeRow = words[0] >>> 0, positiveRow = words[1] >>> 0;
    const endpointIds = [...new Set([negativeRow, positiveRow]
      .filter((row) => row !== 0xffff_ffff && row < this.plan.rowCapacity))];
    const endpoints: unknown[] = [];
    for (const row of endpointIds) {
      const endpointReadback = this.device.createBuffer({
        label: "Face-band transient power failure endpoint",
        size: 96,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const endpointEncoder = this.device.createCommandEncoder({
        label: "Read transient physical power endpoint",
      });
      endpointEncoder.copyBufferToBuffer(this.rows, row * OCTREE_FACE_BAND_ROW_BYTES,
        endpointReadback, 0, OCTREE_FACE_BAND_ROW_BYTES);
      endpointEncoder.copyBufferToBuffer(this.transitionMetrics, row * 16, endpointReadback, 32, 16);
      endpointEncoder.copyBufferToBuffer(this.velocities, row * 16, endpointReadback, 48, 16);
      endpointEncoder.copyBufferToBuffer(this.provisionalVelocities, row * 16, endpointReadback, 64, 16);
      endpointEncoder.copyBufferToBuffer(this.transientPowerRows,
        row * OCTREE_FACE_BAND_TRANSIENT_ROW_BYTES, endpointReadback, 80,
        OCTREE_FACE_BAND_TRANSIENT_ROW_BYTES);
      this.device.queue.submit([endpointEncoder.finish()]);
      try {
        await endpointReadback.mapAsync(GPUMapMode.READ);
        const endpointBytes = endpointReadback.getMappedRange().slice(0);
        const endpointWords = new Uint32Array(endpointBytes);
        const endpointFloats = new Float32Array(endpointBytes);
        endpoints.push({
          index: row,
          row: {
            cell: endpointWords[0], globalRow: endpointWords[1],
            flags: endpointWords[2], size: endpointWords[3],
            representativePhi: endpointFloats[4], minimumPhi: endpointFloats[5],
            maximumPhi: endpointFloats[6], padf: endpointFloats[7],
          },
          metric: {
            topology: endpointWords[8], transformFlags: endpointWords[9],
            volume: endpointFloats[10], reserved: endpointWords[11],
          },
          velocity: Array.from(endpointFloats.slice(12, 16)),
          provisionalVelocity: Array.from(endpointFloats.slice(16, 20)),
          transientRow: {
            faceCount: endpointWords[20], incidenceCount: endpointWords[21],
            faceOffset: endpointWords[22], incidenceOffset: endpointWords[23],
          },
        });
      } finally {
        if (endpointReadback.mapState === "mapped") endpointReadback.unmap();
        endpointReadback.destroy();
      }
    }
    const diagnostic = words[3] >>> 0;
    const stageCode = (diagnostic >>> 8) & 0xff;
    const stage = stageCode === 1 ? "boundary" : stageCode === 2 ? "centroidInterpolation"
      : stageCode === 3 ? "nonfiniteProjection" : "unrecorded";
    return {
      slot,
      owner: {
        row: Math.floor(slot / OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence),
        localFace: slot % OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence,
      },
      face: {
        negativeRow, positiveRow, flags: words[2] >>> 0, diagnostic,
        normal: Array.from(floats.slice(4, 8)),
        centroid: Array.from(floats.slice(8, 12)),
        normalVelocity: floats[12], area: floats[13],
        inverseDistance: floats[14], reserved: floats[15],
      },
      sampleDiagnostic: {
        stage,
        negativeInterpolantValid: (diagnostic & 1) !== 0,
        positiveInterpolantValid: (diagnostic & 2) !== 0,
        negativeReason: (diagnostic >>> 16) & 0xff,
        positiveReason: diagnostic >>> 24,
      },
      endpoints,
    };
  }
  /** Bounded failure-only readback for a rejected regular-to-power transfer. */
  async readPowerPublicationFailure(index: number, powerFaces: {
    readonly faces: GPUBuffer; readonly faceNormals: GPUBuffer; readonly faceCentroids: GPUBuffer;
  }) {
    if (!Number.isSafeInteger(index) || index < 0 || index >= this.plan.powerFaceCapacity) return undefined;
    const faceReadback = this.device.createBuffer({ label: "Face-band power-publication failure face", size: 80,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read face-band power-publication failure" });
    encoder.copyBufferToBuffer(powerFaces.faces, index * 32, faceReadback, 0, 32);
    encoder.copyBufferToBuffer(powerFaces.faceNormals, index * 16, faceReadback, 32, 16);
    encoder.copyBufferToBuffer(powerFaces.faceCentroids, index * 16, faceReadback, 48, 16);
    encoder.copyBufferToBuffer(this.powerVelocityScratch, index * 16, faceReadback, 64, 16);
    this.device.queue.submit([encoder.finish()]);
    let snapshot: ArrayBuffer;
    try {
      await faceReadback.mapAsync(GPUMapMode.READ);
      snapshot = faceReadback.getMappedRange().slice(0);
    } finally {
      if (faceReadback.mapState === "mapped") faceReadback.unmap();
      faceReadback.destroy();
    }
    const words = new Uint32Array(snapshot), floats = new Float32Array(snapshot);
    const bands = [words[18], words[19]].filter((row) => row !== 0xffff_ffff && row < this.plan.rowCapacity);
    const uniqueBands = [...new Set(bands)];
    const bandDetails: unknown[] = [];
    for (const row of uniqueBands) {
      const rowReadback = this.device.createBuffer({ label: "Face-band power-publication failure row", size: 64,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const rowEncoder = this.device.createCommandEncoder({ label: "Read face-band power-publication mapped row" });
      rowEncoder.copyBufferToBuffer(this.rows, row * 32, rowReadback, 0, 32);
      rowEncoder.copyBufferToBuffer(this.transitionMetrics, row * 16, rowReadback, 32, 16);
      rowEncoder.copyBufferToBuffer(this.provisionalVelocities, row * 16, rowReadback, 48, 16);
      this.device.queue.submit([rowEncoder.finish()]);
      try {
        await rowReadback.mapAsync(GPUMapMode.READ);
        const rowBytes = rowReadback.getMappedRange().slice(0), rowWords = new Uint32Array(rowBytes);
        const rowFloats = new Float32Array(rowBytes);
        bandDetails.push({ band: row,
          row: { cell: rowWords[0], globalRow: rowWords[1], flags: rowWords[2], size: rowWords[3],
            representativePhi: rowFloats[4], minimumPhi: rowFloats[5], maximumPhi: rowFloats[6], padf: rowFloats[7] },
          metric: { topology: rowWords[8], transformFlags: rowWords[9], volume: rowFloats[10] },
          provisionalVelocity: Array.from(rowFloats.slice(12, 16)) });
      } finally {
        if (rowReadback.mapState === "mapped") rowReadback.unmap();
        rowReadback.destroy();
      }
    }
    return { index,
      face: { negativeRow: words[0], positiveRow: words[1], geometryCode: words[2], flags: words[3],
        normalVelocity: floats[4], area: floats[5], inverseDistance: floats[6], openFraction: floats[7] },
      normal: Array.from(floats.slice(8, 12)), centroid: Array.from(floats.slice(12, 16)),
      mapping: { x: words[16], y: words[17], negativeBand: words[18], positiveBand: words[19] },
      bandDetails };
  }
  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    for (const buffer of [this.control, this.rows, this.rowDirectory,
      this.faces, this.candidateControl, this.candidateRows, this.candidateRowDirectory,
      this.candidateFaces, this.candidateIncidence, this.candidateTransitionAdjacency,
      this.candidateTransitionControl, this.candidateTransitionMetrics,
      this.rowDirectoryScratch, this.catalogSupportScratch, this.incidence,
      this.velocities, this.provisionalVelocities, this.pointStatus, this.pointFieldControl,
      this.transientPowerFaces, this.transientPowerIncidence, this.transientPowerRows, this.transientPowerControl,
      this.transitionAdjacency, this.transitionControl, this.transitionMetrics,
      this.transitionDeltaScan, this.endpointIncomingCounts, this.bandPhiFrontier,
      this.liveFaceWorklist, this.publicationReductions,
      this.powerVelocityScratch, this.powerPublicationControl,
      this.fusedAuthority, this.fusedAuthorityParams,
      this.indirect, this.params, this.sampleParams,
      this.completionParams, this.completionTopologyParams]) buffer.destroy();
    this.bindGroupCache.clear();
  }
  private assertLive(): void { if (this.destroyed) throw new Error("Octree face-band extension is destroyed"); }
}

export const octreeFaceBandFusedAuthorityWGSL = /* wgsl */ `
struct P{controlOffset:u32,controlWords:u32,rowOffset:u32,rowWords:u32,directoryOffset:u32,directoryWords:u32,
 velocityOffset:u32,velocityWords:u32,metricOffset:u32,metricWords:u32,transitionOffset:u32,transitionWords:u32,
 pointOffset:u32,pointWords:u32,p0:u32,p1:u32}
@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(1)var<storage,read>control:array<u32>;
@group(0)@binding(2)var<storage,read>rows:array<u32>;
@group(0)@binding(3)var<storage,read>directory:array<u32>;
@group(0)@binding(4)var<storage,read>velocities:array<u32>;
@group(0)@binding(5)var<storage,read>metrics:array<u32>;
@group(0)@binding(6)var<storage,read>transition:array<u32>;
@group(0)@binding(7)var<storage,read>point:array<u32>;
@group(0)@binding(8)var<storage,read_write>authority:array<u32>;
fn publish(i:u32,count:u32,offset:u32,value:u32){if(i<count&&offset+i<arrayLength(&authority)){authority[offset+i]=value;}}
@compute @workgroup_size(64)fn publishFaceBandFusedAuthority(@builtin(global_invocation_id)id:vec3u){
 let i=id.x;
 if(i<arrayLength(&control)){publish(i,p.controlWords,p.controlOffset,control[i]);}
 if(i<arrayLength(&rows)){publish(i,p.rowWords,p.rowOffset,rows[i]);}
 if(i<arrayLength(&directory)){publish(i,p.directoryWords,p.directoryOffset,directory[i]);}
 if(i<arrayLength(&velocities)){publish(i,p.velocityWords,p.velocityOffset,velocities[i]);}
 if(i<arrayLength(&metrics)){publish(i,p.metricWords,p.metricOffset,metrics[i]);}
 if(i<arrayLength(&transition)){publish(i,p.transitionWords,p.transitionOffset,transition[i]);}
 if(i<arrayLength(&point)){publish(i,p.pointWords,p.pointOffset,point[i]);}
}`;

function wgslFunctionDeclaration(source: string, name: string): string {
  const match = new RegExp(`\\bfn\\s+${name}\\s*\\(`).exec(source);
  if (!match) throw new Error(`Missing face-band WGSL function ${name}`);
  const open = source.indexOf("{", match.index);
  if (open < 0) throw new Error(`Malformed face-band WGSL function ${name}`);
  let depth = 0;
  for (let cursor = open; cursor < source.length; cursor += 1) {
    if (source[cursor] === "{") depth += 1;
    else if (source[cursor] === "}" && --depth === 0) return source.slice(match.index, cursor + 1);
  }
  throw new Error(`Unterminated face-band WGSL function ${name}`);
}

/**
 * Recurring Section 5 air completion over an immutable face-band publication.
 * The topology shader below publishes fixed-address row/face records and
 * reduces them deterministically at transaction boundaries. Once published,
 * this dedicated module binds the identical bytes as
 * ordinary read-only structures and gives every query exclusive ownership of
 * its result/status slot. No atomic operation is reachable from these entries.
 */
export function makeOctreeFaceBandAirSamplerFunctionsWGSL(): string {
  const functions = [
    "finite", "cell", "supportedOwnerSize", "ownerContains",
    "invalidOwner", "decodePagedOwner", "residentCanonicalOwner",
    "pagedOwnerPublicationValid", "ownerAt", "ownerAtCached", "coord",
    "negativeBoundaryBit", "positiveBoundaryBit", "velocityValid", "powerTransform",
    "inversePowerTransform", "velocityExtendedOrigin", "reflectComponents", "finalCellVector", "supportCellVector",
    "supportSelectorCellVector", "reflectedBoundarySelectorVector", "finalSignedVector", "finalSelectorVector",
    "tetraWeights", "contained", "invalidPointVector", "invalidPointVectorAt", "uniformCubeContains",
    "finalTetraPointVector", "finalPointVector", "containingPublishedRow",
    "resolveFinalPointVectorMeasured",
    "airSampleGrid",
  ].map(name => wgslFunctionDeclaration(octreeFaceBandWGSL, name)).join("\n");
  return functions;
}

export function makeOctreeFaceBandAirSampleWGSL(): string {
  const functions = makeOctreeFaceBandAirSamplerFunctionsWGSL();
  const source = /* wgsl */ `
struct P{dims:vec3u,maximumLeaf:u32,rowCapacity:u32,faceCapacity:u32,rowDirectoryCapacity:u32,reservedDirectory:u32,powerDirectoryCapacity:u32,powerRowCapacity:u32,generation:u32,reserved0:u32,ownersPerBrick:u32,powerGeneration:u32,axisStride:u32,ownedFacesPerRow:u32,closedBoundaryMask:u32,rowDeltaControl:u32,rowDeltaNewToOld:u32,rowDeltaAffected:u32,rowDeltaEnabled:u32,rowDeltaOldToNew:u32,rowDeltaDirty:u32,pad2:u32}
struct Row{cell:u32,globalRow:u32,flags:u32,size:u32,representativePhi:f32,minimumPhi:f32,maximumPhi:f32,padf:f32}
struct Owner{origin:vec3u,size:u32,valid:u32}
struct Metric{topology:u32,transformFlags:u32,volume:f32,reserved:u32}
struct TetraHeader{first:u32,count:u32,flags:u32}struct TetraVertex{v:vec4f}
struct ResolvedPointVectorMeasurement{value:vec4f,directSuccess:u32,boundedCatalogSearch:u32,candidateRowsTested:u32}
struct C{flags:u32,firstError:u32,rowCount:u32,faceCount:u32,incidenceCount:u32,generation:u32,valid:u32,reserved0:u32,seedCount:u32,acceptedCount:u32,unresolvedCount:u32,initialRows:u32,sampleFailures:u32,coarsePhiSamples:u32,coarsePhiFailures:u32,bandPhiExtensions:u32,closestPointFaces:u32,closestPointFailures:u32,liquidInterpolationFailures:u32,firstFaceEmissionFailure:u32,firstPhiFailureSlot:u32,firstPhiFailure:u32,firstClosestPointFailure:u32,firstVectorFailure:u32,cptNoOwnerFailures:u32,cptSupportOwnerFailures:u32,cptNoContainingSimplexFailures:u32,cptMissingLiquidVertexFailures:u32,firstCptNoOwnerFailure:u32,firstCptSupportOwnerFailure:u32,firstCptNoSimplexFailure:u32,firstCptMissingVertexFailure:u32}
struct PointControl{flags:u32,firstError:u32,rowCount:u32,generation:u32,solved:u32,valid:u32,wallContributions:u32,pad:u32}
struct TransitionControl{flags:u32,firstError:u32,rowCount:u32,transitionRows:u32,adjacencyCount:u32,ready:u32,transferReady:u32,detailFlags:u32,coreEnd:u32,support1End:u32,support2End:u32,support3NodeEnd:u32,endpointEnd:u32,boundaryGhostRequests:u32,hierarchyReady:u32,phiFailureCounts:u32,failureBand:u32,failureStage:u32,failureRowCell:u32,failureRowSize:u32,failureDescriptor:u32,failureTopology:u32,failureTransformFlags:u32,failureSelector:u32,failureRawX:u32,failureRawY:u32,failureRawZ:u32,failureRequestedSize:u32,failureResolvedCell:u32,failureBoundaryFlips:u32,failureOwnerCell:u32,failureOwnerSizeValid:u32,support4NodeEnd:u32,support5NodeEnd:u32,support6NodeEnd:u32,support7NodeEnd:u32,pad35:u32,pad36:u32,pad37:u32,pad38:u32}
struct SampleP{dims:vec3u,maximumLeaf:u32,powerDirectoryCapacity:u32,count:u32,rowDirectoryCapacity:u32,rowCapacity:u32,cellSize:f32,fineGeneration:u32,p1:u32,p2:u32}
@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(5)var<storage,read>control:C;
@group(0)@binding(6)var<storage,read>rows:array<Row>;
@group(0)@binding(7)var<storage,read>rowDirectory:array<u32>;
@group(0)@binding(19)var<storage,read>rowVelocities:array<vec4f>;
@group(0)@binding(20)var<uniform>sp:SampleP;
@group(0)@binding(21)var<storage,read>positions:array<vec4f>;
@group(0)@binding(22)var<storage,read_write>sampleResults:array<vec4f>;
@group(0)@binding(23)var<storage,read_write>sampleStatus:array<u32>;
@group(0)@binding(26)var<storage,read>owners:array<u32>;
@group(0)@binding(27)var<storage,read>metrics:array<Metric>;
@group(0)@binding(28)var<storage,read>tetraHeaders:array<TetraHeader>;
@group(0)@binding(29)var<storage,read>tetrahedra:array<u32>;
@group(0)@binding(30)var<storage,read>tetraVertices:array<TetraVertex>;
@group(0)@binding(32)var<storage,read>transitionControl:TransitionControl;
@group(0)@binding(48)var<storage,read>pointControl:PointControl;
const INVALID:u32=0xffffffffu;const VALID:u32=0x80000000u;const EXTRAPOLATED:u32=0x10000000u;const FACE_BAND_UNAVAILABLE:u32=0x08000000u;
const ROW_PHI:u32=1u;const ROW_SUPPORT2:u32=16u;const ROW_SUPPORT3_NODE:u32=32u;const ROW_SUPPORT3_ENDPOINT:u32=64u;
const SAMPLE_EVALUATE:u32=0x04000000u;const SAMPLE_EVALUATED:u32=0x02000000u;const SAMPLE_FAILED:u32=0x01000000u;
const SAMPLE_FAIL_DOMAIN:u32=1u;const SAMPLE_FAIL_OWNER:u32=2u;const SAMPLE_FAIL_ROW:u32=3u;const SAMPLE_FAIL_GENERATION:u32=4u;
${functions}
fn rowIdentityLess(cellA:u32,sizeA:u32,cellB:u32,sizeB:u32)->bool{return cellA<cellB||(cellA==cellB&&sizeA<sizeB);}
fn rowIdentitySlot(cellKey:u32,size:u32)->u32{let domain=p.dims.x*p.dims.y*p.dims.z;if(!supportedOwnerSize(size)||cellKey>=domain){return INVALID;}let origin=coord(cellKey);if(any(origin%vec3u(size)!=vec3u(0u))||any(origin+vec3u(size)>p.dims)){return INVALID;}let level=31u-countLeadingZeros(size);if(p.reservedDirectory>arrayLength(&rowDirectory)||level>5u||level*domain>arrayLength(&rowDirectory)-p.reservedDirectory){return INVALID;}let base=p.reservedDirectory+level*domain;if(base>=arrayLength(&rowDirectory)||cellKey>=arrayLength(&rowDirectory)-base){return INVALID;}return base+cellKey;}
fn rowOfIdentity(cellKey:u32,size:u32)->u32{let state=p.rowDirectoryCapacity*2u;if(state>=arrayLength(&rowDirectory)){return INVALID;}let committed=state+3u<arrayLength(&rowDirectory)&&rowDirectory[state+3u]==0x54524e53u;let count=min(p.rowDirectoryCapacity,select(rowDirectory[state],rowDirectory[state+1u],committed));let slot=rowIdentitySlot(cellKey,size);if(slot==INVALID){return INVALID;}let encoded=rowDirectory[slot];if(encoded==0u){return INVALID;}let row=encoded-1u;if(row>=count||row>=arrayLength(&rows)){return INVALID;}let candidate=rows[row];return select(INVALID,row,candidate.cell==cellKey&&candidate.size==size);}
fn sampledBandGenerationValid()->bool{let mask=0x3fffffffu;let fine=sp.fineGeneration&mask;let paramsFine=p.generation&mask;let band=control.generation&mask;let power=p.powerGeneration&mask;let predecessor=(fine+mask)&mask;return paramsFine==fine&&(band==fine||(power==fine&&band==predecessor));}
fn retainedBandAnchor(pointGrid:vec3f)->u32{if(any(pointGrid<vec3f(0))||any(pointGrid>=vec3f(sp.dims))){return INVALID;}let q=vec3u(floor(pointGrid));var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);let band=rowOfIdentity(cell(origin),size);if(band!=INVALID&&band<sp.rowCapacity&&band<transitionControl.support7NodeEnd&&band<arrayLength(&rows)){let row=rows[band];if(row.cell==cell(origin)&&(row.flags&ROW_SUPPORT3_ENDPOINT)==0u){return band;}}if(size>=sp.maximumLeaf){break;}size<<=1u;}return INVALID;}
@compute @workgroup_size(64)fn classifyAirBandVelocity(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=sp.count||i>=arrayLength(&positions)||i>=arrayLength(&sampleResults)||i>=arrayLength(&sampleStatus)||positions[i].w<=0.){return;}let sample=airSampleGrid(positions[i].xyz);if(sample.w==0.){sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_DOMAIN;return;}let grid=sample.xyz;let q=vec3u(floor(grid));let owner=ownerAt(q);if(owner.valid==0u){sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_OWNER;return;}let band=retainedBandAnchor(grid);if(band==INVALID||band>=sp.rowCapacity||band>=transitionControl.support7NodeEnd||band>=arrayLength(&rows)){sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_ROW;return;}if(!sampledBandGenerationValid()||pointControl.generation!=sp.fineGeneration){sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_GENERATION;return;}if((rows[band].flags&ROW_PHI)==0u){sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_ROW;return;}let stageBStatus=sampleStatus[i];let stageBReason=stageBStatus&255u;let needsDualCompletion=(stageBStatus&VALID)==0u&&(stageBReason==4u||stageBReason==8u);if(rows[band].minimumPhi<0.&&!needsDualCompletion){return;}sampleResults[i].w=bitcast<f32>(band);sampleStatus[i]=SAMPLE_EVALUATE;}
@compute @workgroup_size(64)fn evaluateAirBandVelocity(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=sp.count||i>=arrayLength(&positions)||i>=arrayLength(&sampleResults)||i>=arrayLength(&sampleStatus)||sampleStatus[i]!=SAMPLE_EVALUATE){return;}let band=bitcast<u32>(sampleResults[i].w);let sample=airSampleGrid(positions[i].xyz);if(sample.w==0.){sampleResults[i]=invalidPointVector(SAMPLE_FAIL_DOMAIN);sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_DOMAIN;return;}let velocity=resolveFinalPointVectorMeasured(band,sample.xyz).value;if(!velocityValid(velocity)){let reason=u32(round(max(1.,-velocity.w)));sampleResults[i]=velocity;sampleStatus[i]=SAMPLE_FAILED|((band&0xffffu)<<8u)|((16u+min(reason,11u))&255u);return;}sampleResults[i]=velocity;sampleStatus[i]=SAMPLE_EVALUATED;}
@compute @workgroup_size(64)fn finalizeAirBandVelocity(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=sp.count||i>=arrayLength(&sampleStatus)){return;}let status=sampleStatus[i];if(status!=SAMPLE_EVALUATE&&status!=SAMPLE_EVALUATED&&(status&SAMPLE_FAILED)==0u){return;}if(status==SAMPLE_EVALUATED&&control.valid==VALID&&sampledBandGenerationValid()&&pointControl.valid==VALID&&pointControl.flags==0u&&pointControl.generation==sp.fineGeneration){sampleStatus[i]=VALID|EXTRAPOLATED;return;}sampleStatus[i]=FACE_BAND_UNAVAILABLE|(status&0x00ffffffu);}
`;
  if (/\\batomic(?:Add|And|CompareExchangeWeak|Exchange|Load|Max|Min|Or|Store|Sub|Xor)?\\b/.test(source)) {
    throw new Error("Recurring face-band air sampler must not contain atomic operations");
  }
  return source;
}

/**
 * Deterministic parallel support-row compaction.
 *
 * Catalog requests map to one collision-free `cell * 6 + log2(size)` bit.
 * Marking is therefore insensitive to duplicate emission order.  A
 * workgroup-local scan plus a compact scan of block totals publishes rows in
 * the exact cell/size order formerly obtained by four whole-stream radix
 * sorts, without assigning one workgroup all candidate records.
 */
export const octreeFaceBandSupportScatterWGSL = /* wgsl */ `
struct P{dims:vec3u,maximumLeaf:u32,rowCapacity:u32,faceCapacity:u32,rowDirectoryCapacity:u32,reservedDirectory:u32,powerDirectoryCapacity:u32,powerRowCapacity:u32,generation:u32,reserved0:u32,ownersPerBrick:u32,powerGeneration:u32,axisStride:u32,ownedFacesPerRow:u32,closedBoundaryMask:u32,rowDeltaControl:u32,rowDeltaNewToOld:u32,rowDeltaAffected:u32,rowDeltaEnabled:u32,rowDeltaOldToNew:u32,rowDeltaDirty:u32,pad2:u32}
struct Row{cell:u32,globalRow:u32,flags:u32,size:u32,representativePhi:f32,minimumPhi:f32,maximumPhi:f32,padf:f32}
struct SupportCandidate{cellPlusOne:u32,size:u32}
@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(5)var<storage,read_write>controlWords:array<u32>;
@group(0)@binding(6)var<storage,read_write>rows:array<Row>;
@group(0)@binding(7)var<storage,read_write>rowDirectory:array<u32>;
@group(0)@binding(13)var<storage,read_write>rowDirectoryScratch:array<u32>;
@group(0)@binding(18)var<storage,read_write>indirect:array<u32>;
@group(0)@binding(62)var<storage,read>committedRows:array<Row>;
@group(0)@binding(66)var<storage,read>supportCandidates:array<SupportCandidate>;
@group(0)@binding(67)var<storage,read_write>supportScratch:array<atomic<u32>>;
@group(0)@binding(68)var<storage,read>committedRowDirectory:array<u32>;
const INVALID:u32=0xffffffffu;
const CAPACITY:u32=1u;
const BAD_DIRECTORY:u32=2u;
const ROW_COARSE:u32=2u;
const ROW_SUPPORT1:u32=8u;
const ROW_SUPPORT2:u32=16u;
const ROW_SUPPORT3_NODE:u32=32u;
const ROW_SUPPORT3_ENDPOINT:u32=64u;
const COMMITTED_TRANSITION_VALID:u32=0x54524e53u;
const MAX_GUARDS:u32=${OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW
    + Math.max(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows,
      OCTREE_FACE_BAND_UNIFORM_SUPPORT_REQUESTS)}u;
var<workgroup>scanValues:array<u32,256>;

fn domainCount()->u32{return p.dims.x*p.dims.y*p.dims.z;}
fn identityCount()->u32{return domainCount()*6u;}
fn prefixBase()->u32{return identityCount();}
fn blockTotalBase()->u32{return identityCount()*2u;}
fn blockCount()->u32{return (identityCount()+255u)/256u;}
fn coord(cellKey:u32)->vec3u{
 return vec3u(cellKey%p.dims.x,(cellKey/p.dims.x)%p.dims.y,
   cellKey/(p.dims.x*p.dims.y));
}
fn supportedOwnerSize(size:u32)->bool{
 return size!=0u&&(size&(size-1u))==0u&&size<=p.maximumLeaf;
}
fn identityRank(cellKey:u32,size:u32)->u32{
 let domain=domainCount();
 if(!supportedOwnerSize(size)||cellKey>=domain){return INVALID;}
 let origin=coord(cellKey);
 if(any(origin%vec3u(size)!=vec3u(0u))||any(origin+vec3u(size)>p.dims)){return INVALID;}
 let level=31u-countLeadingZeros(size);
 if(level>5u){return INVALID;}
 return cellKey*6u+level;
}
fn identitySlot(cellKey:u32,size:u32)->u32{
 let rank=identityRank(cellKey,size);
 if(rank==INVALID){return INVALID;}
 let domain=domainCount();
 let level=rank%6u;
 let slot=p.reservedDirectory+level*domain+cellKey;
 return select(slot,INVALID,slot>=arrayLength(&rowDirectory));
}
fn currentRowCount()->u32{
 let state=p.rowDirectoryCapacity*2u;
 if(state>=arrayLength(&rowDirectory)){return 0u;}
 return min(p.rowCapacity,rowDirectory[state]);
}
fn alreadyPublished(cellKey:u32,size:u32)->bool{
 let slot=identitySlot(cellKey,size);
 if(slot==INVALID){return true;}
 let encoded=rowDirectory[slot];
 if(encoded==0u){return false;}
 let row=encoded-1u;
 let count=currentRowCount();
 return row<count&&row<arrayLength(&rows)
   &&rows[row].cell==cellKey&&rows[row].size==size;
}
fn committedState(word:u32)->u32{
 let at=p.rowDirectoryCapacity*2u+word;
 if(at>=arrayLength(&committedRowDirectory)){return 0u;}
 return committedRowDirectory[at];
}
fn committedRowOfIdentity(cellKey:u32,size:u32)->u32{
 if(committedState(3u)!=COMMITTED_TRANSITION_VALID){return INVALID;}
 let slot=identitySlot(cellKey,size);
 if(slot==INVALID||slot>=arrayLength(&committedRowDirectory)){return INVALID;}
 let encoded=committedRowDirectory[slot];
 if(encoded==0u){return INVALID;}
 let row=encoded-1u;
 let count=min(p.rowCapacity,committedState(1u));
 if(row>=count||row>=arrayLength(&committedRows)){return INVALID;}
 let candidate=committedRows[row];
 return select(INVALID,row,candidate.cell==cellKey&&candidate.size==size);
}
fn tierBegin(tier:u32)->u32{
 let state=p.rowDirectoryCapacity*2u;
 if(tier==1u){return controlWords[11u];}
 if(tier==2u){return rowDirectory[state+1u];}
 if(tier==3u){return rowDirectory[state+2u];}
 return rowDirectory[state+3u];
}
fn tierFlag(tier:u32)->u32{
 if(tier==1u){return ROW_SUPPORT1;}
 if(tier==2u){return ROW_SUPPORT2;}
 if(tier==3u){return ROW_SUPPORT3_NODE;}
 return ROW_SUPPORT3_ENDPOINT;
}
fn fail(code:u32,detail:u32){
 controlWords[0u]|=code;
 controlWords[1u]=min(controlWords[1u],detail);
}

@compute @workgroup_size(256)fn clearSupportIdentityMarks(
 @builtin(global_invocation_id)g:vec3u){
 let rank=g.x;
 if(rank<identityCount()&&rank<arrayLength(&supportScratch)){
   atomicStore(&supportScratch[rank],0u);
 }
}
fn markSupportSource(source:u32){
 if(controlWords[0u]!=0u){return;}
 let base=source*MAX_GUARDS;
 if(base>arrayLength(&supportCandidates)
    ||MAX_GUARDS>arrayLength(&supportCandidates)-base){
   return;
 }
 for(var slot=0u;slot<MAX_GUARDS;slot+=1u){
   let value=supportCandidates[base+slot];
   if(value.cellPlusOne==INVALID||value.cellPlusOne==0u||value.size==0u){continue;}
   let cellKey=value.cellPlusOne-1u;
   let rank=identityRank(cellKey,value.size);
   if(rank==INVALID||rank>=arrayLength(&supportScratch)
      ||alreadyPublished(cellKey,value.size)){continue;}
   atomicStore(&supportScratch[rank],1u);
 }
}
@compute @workgroup_size(64)fn markSupport1(@builtin(global_invocation_id)g:vec3u){
 markSupportSource(g.x);
}
@compute @workgroup_size(64)fn markSupport2(@builtin(global_invocation_id)g:vec3u){
 markSupportSource(g.x);
}
@compute @workgroup_size(64)fn markSupport3(@builtin(global_invocation_id)g:vec3u){
 markSupportSource(g.x);
}
@compute @workgroup_size(64)fn markSupport4(@builtin(global_invocation_id)g:vec3u){
 markSupportSource(g.x);
}
@compute @workgroup_size(64)fn markPublishedSupportRows(
 @builtin(global_invocation_id)g:vec3u){
 let row=g.x;
 if(controlWords[0u]!=0u||row>=controlWords[2u]||row>=arrayLength(&rows)){return;}
 let candidate=rows[row];
 let rank=identityRank(candidate.cell,candidate.size);
 if(rank<identityCount()&&rank<arrayLength(&supportScratch)){
   atomicStore(&supportScratch[rank],1u);
 }
}

@compute @workgroup_size(256)fn scanSupportIdentityBlocks(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
 let rank=wid.x*256u+lane;
 var value=0u;
 if(rank<identityCount()&&rank<arrayLength(&supportScratch)){
   value=atomicLoad(&supportScratch[rank]);
 }
 scanValues[lane]=value;
 workgroupBarrier();
 for(var offset=1u;offset<256u;offset<<=1u){
   var addend=0u;
   if(lane>=offset){addend=scanValues[lane-offset];}
   workgroupBarrier();
   scanValues[lane]+=addend;
   workgroupBarrier();
 }
 let prefixAt=prefixBase()+rank;
 if(rank<identityCount()&&prefixAt<arrayLength(&supportScratch)){
   atomicStore(&supportScratch[prefixAt],scanValues[lane]-value);
 }
 if(lane==255u){
   let totalAt=blockTotalBase()+wid.x;
   if(totalAt<arrayLength(&supportScratch)){
     atomicStore(&supportScratch[totalAt],scanValues[lane]);
   }
 }
}
fn laneBlockRange(lane:u32)->vec2u{
 let count=blockCount();
 let width=count/256u;
 let remainder=count%256u;
 let begin=lane*width+min(lane,remainder);
 return vec2u(begin,begin+width+select(0u,1u,lane<remainder));
}
@compute @workgroup_size(256)fn prefixSupportIdentityBlocks(
 @builtin(local_invocation_index)lane:u32){
 let range=laneBlockRange(lane);
 var localTotal=0u;
 for(var block=range.x;block<range.y;block+=1u){
   localTotal+=atomicLoad(&supportScratch[blockTotalBase()+block]);
 }
 scanValues[lane]=localTotal;
 workgroupBarrier();
 for(var offset=1u;offset<256u;offset<<=1u){
   var addend=0u;
   if(lane>=offset){addend=scanValues[lane-offset];}
   workgroupBarrier();
   scanValues[lane]+=addend;
   workgroupBarrier();
 }
 var running=scanValues[lane]-localTotal;
 for(var block=range.x;block<range.y;block+=1u){
   let at=blockTotalBase()+block;
   let count=atomicLoad(&supportScratch[at]);
   atomicStore(&supportScratch[at],running);
   running+=count;
 }
 if(lane==255u){
   atomicStore(&supportScratch[blockTotalBase()+blockCount()],scanValues[lane]);
 }
}
fn finalizeTier(tier:u32){
 if(controlWords[0u]!=0u){return;}
 let total=atomicLoad(&supportScratch[blockTotalBase()+blockCount()]);
 let begin=tierBegin(tier);
 if(begin>p.rowCapacity||total>p.rowCapacity-begin){
   fail(CAPACITY,total);
   return;
 }
 let state=p.rowDirectoryCapacity*2u;
 let end=begin+total;
 rowDirectory[state]=end;
 if(tier==1u){rowDirectory[state+1u]=end;indirect[63u]=(total+63u)/64u;}
 else if(tier==2u){rowDirectory[state+2u]=end;indirect[60u]=(total+63u)/64u;}
 else if(tier==3u){rowDirectory[state+3u]=end;indirect[63u]=(total+63u)/64u;}
 else{controlWords[2u]=end;indirect[0u]=1u;indirect[3u]=(end+63u)/64u;}
}
@compute @workgroup_size(1)fn finalizeSupport1(){finalizeTier(1u);}
@compute @workgroup_size(1)fn finalizeSupport2(){finalizeTier(2u);}
@compute @workgroup_size(1)fn finalizeSupport3(){finalizeTier(3u);}
@compute @workgroup_size(1)fn finalizeSupport4(){finalizeTier(4u);}

fn scatterTier(rank:u32,tier:u32){
 if(rank>=identityCount()||rank>=arrayLength(&supportScratch)){return;}
 let marked=atomicLoad(&supportScratch[rank]);
 if(marked==0u){return;}
 atomicStore(&supportScratch[rank],0u);
 if(controlWords[0u]!=0u){return;}
 let block=rank/256u;
 let position=atomicLoad(&supportScratch[prefixBase()+rank])
   +atomicLoad(&supportScratch[blockTotalBase()+block]);
 let band=tierBegin(tier)+position;
 let cellKey=rank/6u;
 let level=rank%6u;
 let size=1u<<level;
 let identity=identitySlot(cellKey,size);
 if(band>=p.rowCapacity||band>=arrayLength(&rows)
    ||band*2u+1u>=arrayLength(&rowDirectory)||identity==INVALID){
   return;
 }
 let old=committedRowOfIdentity(cellKey,size);
 let priorEncoded=select(0u,old+1u,old!=INVALID);
 rows[band]=Row(cellKey,band+1u,ROW_COARSE|tierFlag(tier),size,
   0.,0.,0.,bitcast<f32>(priorEncoded));
 rowDirectory[band*2u]=cellKey;
 rowDirectory[band*2u+1u]=band;
 rowDirectory[identity]=band+1u;
 // Preserve the existing diagnostic S1 tuple without making it authority.
 if(tier==1u&&position*3u+2u<arrayLength(&rowDirectoryScratch)){
   rowDirectoryScratch[position*3u]=cellKey+1u;
   rowDirectoryScratch[position*3u+1u]=size;
   rowDirectoryScratch[position*3u+2u]=tier;
 }
}
@compute @workgroup_size(256)fn scatterSupport1(@builtin(global_invocation_id)g:vec3u){
 scatterTier(g.x,1u);
}
@compute @workgroup_size(256)fn scatterSupport2(@builtin(global_invocation_id)g:vec3u){
 scatterTier(g.x,2u);
}
@compute @workgroup_size(256)fn scatterSupport3(@builtin(global_invocation_id)g:vec3u){
 scatterTier(g.x,3u);
}
@compute @workgroup_size(256)fn scatterSupport4(@builtin(global_invocation_id)g:vec3u){
 scatterTier(g.x,4u);
}
@compute @workgroup_size(256)fn scatterCanonicalRowDirectory(
 @builtin(global_invocation_id)g:vec3u){
 let rank=g.x;
 if(rank>=identityCount()||rank>=arrayLength(&supportScratch)
    ||atomicLoad(&supportScratch[rank])==0u){return;}
 atomicStore(&supportScratch[rank],0u);
 if(controlWords[0u]!=0u){return;}
 let block=rank/256u;
 let position=atomicLoad(&supportScratch[prefixBase()+rank])
   +atomicLoad(&supportScratch[blockTotalBase()+block]);
 let cellKey=rank/6u;
 let size=1u<<(rank%6u);
 let slot=identitySlot(cellKey,size);
 if(position>=controlWords[2u]||position*2u+1u>=arrayLength(&rowDirectory)
    ||slot==INVALID){return;}
 let encoded=rowDirectory[slot];
 if(encoded==0u){return;}
 let row=encoded-1u;
 if(row>=controlWords[2u]||row>=arrayLength(&rows)
    ||rows[row].cell!=cellKey||rows[row].size!=size){return;}
 rowDirectory[position*2u]=cellKey;
 rowDirectory[position*2u+1u]=row;
}
`;

export const octreeFaceBandWGSL = /* wgsl */ `
struct P{dims:vec3u,maximumLeaf:u32,rowCapacity:u32,faceCapacity:u32,rowDirectoryCapacity:u32,reservedDirectory:u32,powerDirectoryCapacity:u32,powerRowCapacity:u32,generation:u32,reserved0:u32,ownersPerBrick:u32,powerGeneration:u32,axisStride:u32,ownedFacesPerRow:u32,closedBoundaryMask:u32,rowDeltaControl:u32,rowDeltaNewToOld:u32,rowDeltaAffected:u32,rowDeltaEnabled:u32,rowDeltaOldToNew:u32,rowDeltaDirty:u32,pad2:u32}
struct FineP{brickDims:vec3u,brickResolution:u32,sampleDims:vec3u,samplesPerBrick:u32,domainOrigin:vec3f,fineWidth:f32,worklistCapacity:u32,worklistHeaderWords:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32}
struct PowerRowDirectoryEntry{cellPlusOne:u32,size:u32,row:u32,morton:u32}
alias Site=PowerRowDirectoryEntry;
struct Row{cell:u32,globalRow:u32,flags:u32,size:u32,representativePhi:f32,minimumPhi:f32,maximumPhi:f32,padf:f32}
struct CoarseEntry{cellPlusOne:u32,size:u32,phi:f32,minimumPhi:f32,maximumPhi:f32,flags:u32,row:u32,physicalVolume:f32}struct CoarseDirectory{state:u32,generation:u32,rowCount:u32,maximumLeafSize:u32,dimensions:vec3u,physicalCellSize:f32,entries:array<CoarseEntry>}
struct Owner{origin:vec3u,size:u32,valid:u32}
struct CatalogEntry{firstFace:u32,faceCount:u32}struct PowerCatalogFace{neighborOffsetSize:vec4f,areaCentroid:vec4f,normalInverseDistance:vec4f}struct BandLS{a0:vec4f,a1:vec4f,a2:vec4f}struct PointControl{flags:u32,firstError:u32,rowCount:u32,generation:u32,solved:u32,valid:u32,wallContributions:u32,pad:u32}
struct Metric{topology:u32,transformFlags:u32,volume:f32,reserved:u32}struct TetraHeader{first:u32,count:u32,flags:u32}struct TetraVertex{v:vec4f}struct TransitionAdjacency{band:u32,a:u32,b:u32,c:u32}struct TransitionControl{flags:u32,firstError:u32,rowCount:u32,transitionRows:u32,adjacencyCount:u32,ready:u32,transferReady:u32,detailFlags:u32,coreEnd:u32,support1End:u32,support2End:u32,support3NodeEnd:u32,endpointEnd:u32,boundaryGhostRequests:u32,hierarchyReady:u32,phiFailureCounts:u32,failureBand:u32,failureStage:u32,failureRowCell:u32,failureRowSize:u32,failureDescriptor:u32,failureTopology:u32,failureTransformFlags:u32,failureSelector:u32,failureRawX:u32,failureRawY:u32,failureRawZ:u32,failureRequestedSize:u32,failureResolvedCell:u32,failureBoundaryFlips:u32,failureOwnerCell:u32,failureOwnerSizeValid:u32,support4NodeEnd:u32,support5NodeEnd:u32,support6NodeEnd:u32,support7NodeEnd:u32,pad35:u32,pad36:u32,pad37:u32,pad38:u32}
struct Face{negativeRow:u32,positiveRow:u32,axisSpan:u32,globalFace:u32,velocity:vec4f,centroid:vec4f,phi:f32,area:f32,flags:u32,pad:u32}
struct PhiDiagnostic{origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32,cause:u32,detail:u32}
struct PhiCPT{phi:f32,gradient:vec3f,valid:u32}
struct LiquidInterpolation{value:vec4f,reason:u32}
struct C{flags:u32,firstError:u32,rowCount:u32,faceCount:u32,incidenceCount:u32,generation:u32,valid:u32,reserved0:u32,seedCount:u32,acceptedCount:u32,unresolvedCount:u32,initialRows:u32,sampleFailures:u32,coarsePhiSamples:u32,coarsePhiFailures:u32,bandPhiExtensions:u32,closestPointFaces:u32,closestPointFailures:u32,liquidInterpolationFailures:u32,firstFaceEmissionFailure:u32,firstPhiFailureSlot:u32,firstPhiFailure:u32,firstClosestPointFailure:u32,firstVectorFailure:u32,cptNoOwnerFailures:u32,cptSupportOwnerFailures:u32,cptNoContainingSimplexFailures:u32,cptMissingLiquidVertexFailures:u32,firstCptNoOwnerFailure:u32,firstCptSupportOwnerFailure:u32,firstCptNoSimplexFailure:u32,firstCptMissingVertexFailure:u32}
struct PowerFace{negativeRow:u32,positiveRow:u32,geometryCode:u32,flags:u32,normalVelocity:f32,area:f32,inverseDistance:f32,openFraction:f32}struct PowerPublication{flags:u32,firstError:u32,faceCount:u32,targetCount:u32,interpolatedCount:u32,committedCount:u32,fineGeneration:u32,powerGeneration:u32,valid:u32,p0:u32,p1:u32,p2:u32,p3:u32,p4:u32,p5:u32,p6:u32}struct PowerRowWork{faceCount:u32,incidenceCount:u32,faceOffset:u32,incidenceOffset:u32}struct PowerIncidence{face:u32,sign:i32}
struct TransientPowerFace{negativeRow:u32,positiveRow:u32,flags:u32,pad:u32,normal:vec4f,centroid:vec4f,normalVelocity:f32,area:f32,inverseDistance:f32,padf:f32}struct TransientPowerControl{flags:u32,firstError:u32,rowCount:u32,faceSlots:u32,emitted:u32,sampled:u32,validated:u32,generation:u32,valid:u32,p0:u32,p1:u32,p2:u32,p3:u32,p4:u32,p5:u32,p6:u32}
struct OldAdvectionControl{flags:u32,firstError:u32,faceCount:u32,advected:u32,generation:u32,oldGeneration:u32,valid:u32,mode:u32,requestedFaces:u32,priorFlags:u32,priorFirstError:u32,priorAdvected:u32,priorValid:u32,firstInterpolationStage:u32,firstInterpolationReason:u32,firstInterpolationDetail:u32}
struct OldAdvectionSeed{flags:u32,firstError:u32,rowCount:u32,faceCount:u32,seededCount:u32,generation:u32,valid:u32}
struct SampleP{dims:vec3u,maximumLeaf:u32,powerDirectoryCapacity:u32,count:u32,rowDirectoryCapacity:u32,rowCapacity:u32,cellSize:f32,fineGeneration:u32,p1:u32,p2:u32}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<uniform>fp:FineP;@group(0)@binding(2)var<storage,read>metadata:array<u32>;@group(0)@binding(3)var<storage,read>worklist:array<u32>;@group(0)@binding(4)var<storage,read>sites:array<Site>;@group(0)@binding(5)var<storage,read_write>control:C;@group(0)@binding(6)var<storage,read_write>rows:array<Row>;@group(0)@binding(7)var<storage,read_write>rowDirectory:array<u32>;@group(0)@binding(8)var<storage,read>sampleFlags:array<u32>;@group(0)@binding(9)var<storage,read>powerVelocityControl:array<u32>;@group(0)@binding(10)var<storage,read>powerRowVelocities:array<vec4f>;@group(0)@binding(11)var<storage,read>catalogEntries:array<CatalogEntry>;@group(0)@binding(12)var<storage,read_write>faces:array<Face>;@group(0)@binding(13)var<storage,read_write>rowDirectoryScratch:array<u32>;@group(0)@binding(14)var<storage,read_write>incidence:array<u32>;@group(0)@binding(18)var<storage,read_write>indirect:array<u32>;@group(0)@binding(19)var<storage,read_write>rowVelocities:array<vec4f>;@group(0)@binding(20)var<uniform>sp:SampleP;@group(0)@binding(21)var<storage,read>positions:array<vec4f>;@group(0)@binding(22)var<storage,read_write>sampleResults:array<vec4f>;@group(0)@binding(23)var<storage,read_write>sampleStatus:array<u32>;@group(0)@binding(24)var<storage,read>finePhi:array<f32>;@group(0)@binding(25)var<storage,read>coarsePhi:CoarseDirectory;@group(0)@binding(26)var<storage,read>owners:array<u32>;@group(0)@binding(27)var<storage,read_write>metrics:array<Metric>;@group(0)@binding(28)var<storage,read>tetraHeaders:array<TetraHeader>;@group(0)@binding(29)var<storage,read>tetrahedra:array<u32>;@group(0)@binding(30)var<storage,read>tetraVertices:array<TetraVertex>;@group(0)@binding(31)var<storage,read_write>transitionAdjacency:array<TransitionAdjacency>;@group(0)@binding(32)var<storage,read_write>transitionControl:TransitionControl;@group(0)@binding(33)var<storage,read>sameOrFinerDirect:array<u32>;@group(0)@binding(34)var<storage,read>sameOrCoarserDirect:array<u32>;@group(0)@binding(36)var<storage,read>powerFaceControl:array<u32>;@group(0)@binding(37)var<storage,read_write>powerFaces:array<PowerFace>;@group(0)@binding(38)var<storage,read_write>powerFaceNormals:array<vec4f>;@group(0)@binding(39)var<storage,read_write>powerFaceCentroids:array<vec4f>;@group(0)@binding(40)var<storage,read_write>powerVelocityScratch:array<vec4u>;@group(0)@binding(41)var<storage,read_write>powerPublication:PowerPublication;@group(0)@binding(42)var<storage,read>fineTopologyControl:array<u32>;@group(0)@binding(44)var<storage,read_write>provisionalVelocities:array<vec4f>;@group(0)@binding(45)var<storage,read>catalogFaces:array<PowerCatalogFace>;@group(0)@binding(47)var<storage,read_write>pointStatus:array<u32>;@group(0)@binding(48)var<storage,read_write>pointControl:PointControl;@group(0)@binding(49)var<storage,read>powerIncidenceRows:array<PowerRowWork>;@group(0)@binding(50)var<storage,read>powerIncidences:array<PowerIncidence>;@group(0)@binding(52)var<storage,read_write>transientPowerFaces:array<TransientPowerFace>;@group(0)@binding(53)var<storage,read_write>transientPowerIncidences:array<PowerIncidence>;@group(0)@binding(54)var<storage,read_write>transientPowerRows:array<PowerRowWork>;@group(0)@binding(55)var<storage,read_write>transientPowerControl:TransientPowerControl;
@group(0)@binding(58)var<storage,read_write>oldAdvectionControl:OldAdvectionControl;@group(0)@binding(59)var<storage,read_write>oldAdvectionSeed:OldAdvectionSeed;
@group(0)@binding(61)var<storage,read>rowDelta:array<u32>;@group(0)@binding(62)var<storage,read_write>committedRows:array<Row>;@group(0)@binding(63)var<storage,read_write>committedMetrics:array<Metric>;@group(0)@binding(64)var<storage,read_write>committedAdjacency:array<TransitionAdjacency>;
@group(0)@binding(65)var<storage,read_write>transitionDeltaScan:array<u32>;
struct SupportCandidate{cellPlusOne:u32,size:u32}
@group(0)@binding(66)var<storage,read_write>supportCandidates:array<SupportCandidate>;
@group(0)@binding(67)var<storage,read_write>supportCandidateScratch:array<SupportCandidate>;
@group(0)@binding(68)var<storage,read_write>committedRowDirectory:array<u32>;
@group(0)@binding(69)var<storage,read_write>committedFaces:array<Face>;
@group(0)@binding(70)var<storage,read_write>committedIncidence:array<u32>;
@group(0)@binding(71)var<storage,read_write>committedControl:C;
@group(0)@binding(72)var<storage,read_write>committedTransitionControl:TransitionControl;
@group(0)@binding(73)var<storage,read_write>endpointIncomingCounts:array<atomic<u32>>;
@group(0)@binding(74)var<storage,read_write>bandPhiFrontier:array<atomic<u32>>;
@group(0)@binding(75)var<storage,read_write>liveFaceWorklist:array<u32>;
@group(0)@binding(76)var<storage,read_write>publicationReductions:array<atomic<u32>>;
const INVALID:u32=0xffffffffu;const VALID:u32=0x80000000u;const ACUTE_TETRA_FAILURE:u32=0x80000000u;const STATUS_VALID:u32=0x3f800000u;const EXTRAPOLATED:u32=0x10000000u;const FACE_BAND_UNAVAILABLE:u32=0x08000000u;const LIVE:u32=1u;const SEED:u32=2u;const PHI_VALID:u32=4u;const PHI_DIAGNOSTIC:u32=8u;const CLOSEST_POINT_VALID:u32=16u;const FACE_VELOCITY_VALID:u32=32u;const VELOCITY_TARGET:u32=64u;const COARSE_PHI:u32=128u;const PRIMARY_EXTENSION:u32=256u;const CPT_NO_OWNER:u32=1u;const CPT_SUPPORT_OWNER:u32=2u;const CPT_NO_SIMPLEX:u32=3u;const CPT_MISSING_VERTEX:u32=4u;const CPT_OWNER_ROW_MISMATCH:u32=7u;const ROW_PHI:u32=1u;const ROW_COARSE:u32=2u;const ROW_CORE:u32=4u;const ROW_SUPPORT1:u32=8u;const ROW_SUPPORT2:u32=16u;const ROW_SUPPORT3_NODE:u32=32u;const ROW_SUPPORT3_ENDPOINT:u32=64u;const ROW_COARSE_AIR:u32=128u;const ROW_COARSE_LIQUID:u32=256u;const ROW_COARSE_MIXED:u32=512u;const ROW_SUPPORT4_NODE:u32=1024u;const ROW_SUPPORT5_NODE:u32=2048u;const ROW_SUPPORT6_NODE:u32=4096u;const ROW_CENTER_WET:u32=8192u;const ROW_PHI_EXTENDED:u32=0x10000000u;const ROW_BOUNDARY_GHOST:u32=0x20000000u;const ROW_TRANSITION_VALIDATED:u32=0x40000000u;const ROW_GUARD:u32=ROW_SUPPORT1|ROW_SUPPORT2|ROW_SUPPORT3_NODE|ROW_SUPPORT4_NODE|ROW_SUPPORT5_NODE|ROW_SUPPORT6_NODE|ROW_SUPPORT3_ENDPOINT;
const CAPACITY:u32=1u;const BAD_DIRECTORY:u32=2u;const SOURCE:u32=4u;const BAD_ROW:u32=8u;const BAD_FACE:u32=16u;const BAD_PHI:u32=32u;const UNRESOLVED:u32=64u;const INCOMPLETE:u32=128u;const OUTSIDE_FINE_BAND:u32=256u;
const TRANSITION_SOURCE:u32=1u;const TRANSITION_CAPACITY:u32=2u;const TRANSITION_ADJACENCY:u32=4u;const TRANSITION_DESCRIPTOR:u32=8u;const MAX_TETRA:u32=${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra}u;const MAX_CATALOG_GUARDS:u32=${Math.max(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows, OCTREE_FACE_BAND_UNIFORM_SUPPORT_REQUESTS)}u;const MAX_ENDPOINTS:u32=${OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW}u;const MAX_GUARDS:u32=MAX_ENDPOINTS+MAX_CATALOG_GUARDS;const UNIFORM_GUARDS:u32=${OCTREE_FACE_BAND_UNIFORM_SUPPORT_REQUESTS}u;const UNIFORM_REQUEST:u32=0x80000000u;const COARSER_DESCRIPTOR:u32=0x80000000u;
const ROW_DELTA_VALID:u32=${OCTREE_POWER_ROW_DELTA_VALID}u;const ROW_DELTA_AFFECTED:u32=0x80000000u;const ROW_TRANSITION_AFFECTED:u32=0x80000000u;const TRANSITION_CARRIED:u32=0x40000000u;const COMMITTED_TRANSITION_VALID:u32=0x54524e53u;
const PHI_CAUSE_MISSING_ROW:u32=0u;const PHI_CAUSE_EXACT_COARSE_MISS:u32=1u;const PHI_CAUSE_INVALID_METRIC:u32=2u;const PHI_CAUSE_INVALID_SELECTOR:u32=3u;const PHI_PATH_CUBE:u32=1u;const PHI_PATH_DELAUNAY:u32=2u;const PHI_PATH_ANCHOR:u32=3u;const PHI_FAILURE_TAG:u32=0x80000000u;
const DETAIL_GEOMETRY:u32=1u;const DETAIL_BELOW_DOMAIN:u32=2u;const DETAIL_ABOVE_DOMAIN:u32=4u;const DETAIL_ALIGNMENT:u32=8u;const DETAIL_OWNER:u32=16u;const DETAIL_MISSING_ROW:u32=32u;const DETAIL_ROW_RANGE:u32=64u;const DETAIL_SIZE:u32=128u;
const OWNER_FAILURE_SUPPORT1:u32=1u;const OWNER_FAILURE_SUPPORT2:u32=2u;const OWNER_FAILURE_SUPPORT3:u32=3u;const OWNER_FAILURE_TRANSITION:u32=4u;const OWNER_FAILURE_ENDPOINT:u32=5u;const OWNER_FAILURE_ACUTE_GRADING:u32=6u;const OWNER_FAILURE_SUPPORT4:u32=8u;const OWNER_FAILURE_DESCRIPTOR:u32=9u;const OWNER_FAILURE_BAND_PHI:u32=10u;
const POWER_SOURCE:u32=1u;const POWER_CAPACITY:u32=2u;const POWER_MISSING_ROW:u32=4u;const POWER_FACE:u32=8u;const POWER_NORMAL:u32=16u;const POWER_NONFINITE:u32=32u;const POWER_INCOMPLETE:u32=64u;const POWER_FAILURE_MARKER:u32=0x80000000u;
var<workgroup>supportSortBins:array<u32,4096>;
// The two terminal topology publishers share one bounded reduction package.
// Every lane owns a contiguous row interval, preserving canonical row/face
// order while replacing the former capacity-sized singleton walks.
var<workgroup>topologyPublishSums:array<vec4u,256>;
var<workgroup>topologyPublishOffsets:array<u32,256>;
fn topologyPublishLaneRange(count:u32,lane:u32)->vec2u{
 let base=count/256u;let extra=count%256u;
 let begin=lane*base+min(lane,extra);
 return vec2u(begin,begin+base+select(0u,1u,lane<extra));
}
const POWER_FACE_VALID:u32=0x80000000u;const POWER_FACE_BOUNDARY:u32=1u;const POWER_FACE_OPEN_BOUNDARY:u32=2u;
const POINT_SOURCE:u32=1u;const POINT_CAPACITY:u32=2u;const POINT_FACE:u32=4u;const POINT_SAMPLE:u32=8u;const POINT_NORMAL:u32=16u;const POINT_NONFINITE:u32=32u;const POINT_SINGULAR:u32=64u;const POINT_CONDITION:u32=128u;const POINT_MAX_FACES:u32=${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u;const TRANSIENT_FACE_FAILURE:u32=${OCTREE_FACE_BAND_TRANSIENT_FACE_FAILURE}u;
const TRANSIENT_ROW_VALIDATED:u32=0x80000000u;const TRANSIENT_ROW_ERROR:u32=0x40000000u;const TRANSIENT_FACE_ERROR:u32=4u;
// Dispatches are dependency-ordered by the Section 5 transaction. Plain
// storage access is sufficient for single-writer records and for immutable
// publication words produced by an earlier dispatch; synchronization atomics
// would only serialize unrelated rows/faces.
const DESCRIPTOR_DIRECTIONS:array<vec3i,18>=array<vec3i,18>(vec3i(-1,0,0),vec3i(0,-1,0),vec3i(0,0,-1),vec3i(0,0,1),vec3i(0,1,0),vec3i(1,0,0),vec3i(-1,-1,0),vec3i(-1,0,-1),vec3i(-1,0,1),vec3i(-1,1,0),vec3i(0,-1,-1),vec3i(0,-1,1),vec3i(0,1,-1),vec3i(0,1,1),vec3i(1,-1,0),vec3i(1,0,-1),vec3i(1,0,1),vec3i(1,1,0));
const BODY_DIAGONAL_DIRECTIONS:array<vec3i,8>=array<vec3i,8>(vec3i(-1,-1,-1),vec3i(-1,-1,1),vec3i(-1,1,-1),vec3i(-1,1,1),vec3i(1,-1,-1),vec3i(1,-1,1),vec3i(1,1,-1),vec3i(1,1,1));
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn fail(code:u32,index:u32){control.flags|=code;control.firstError=min(control.firstError,index);}fn catalogRowFail(code:u32,index:u32,reason:u32){control.reserved0=min(control.reserved0,(index<<3u)|(reason&7u));fail(code,index);}fn faceEmissionFail(code:u32,index:u32,reason:u32){control.firstFaceEmissionFailure=min(control.firstFaceEmissionFailure,index);control.reserved0=min(control.reserved0,(min(index,0xfffffu)<<12u)|(reason&0xfffu));fail(code,index);}fn phiFail(code:u32,index:u32){control.firstPhiFailure=min(control.firstPhiFailure,index);fail(code,index);}fn closestPointFail(code:u32,index:u32){control.firstClosestPointFailure=min(control.firstClosestPointFailure,index);fail(code,index);}fn vectorFail(code:u32,index:u32){control.firstVectorFailure=min(control.firstVectorFailure,index);fail(code,index);}fn unpackBrick(key:u32)->vec3u{let xy=fp.brickDims.x*fp.brickDims.y;let z=key/xy;let r=key-z*xy;let y=r/fp.brickDims.x;return vec3u(r-y*fp.brickDims.x,y,z);}fn cell(q:vec3u)->u32{return q.x+p.dims.x*(q.y+p.dims.y*q.z);}
fn invalidOwner()->Owner{return Owner(vec3u(0u),0u,0u);}fn decodePagedOwner(word:u32,q:vec3u)->Owner{if((word&0x80000000u)==0u){return invalidOwner();}let exponent=(word>>18u)&7u;if(exponent>5u){return invalidOwner();}let brickOrigin=vec3i((q/vec3u(8u))*vec3u(8u));let delta=vec3i(i32(word&63u)-32,i32((word>>6u)&63u)-32,i32((word>>12u)&63u)-32);let signedOrigin=brickOrigin+delta;if(any(signedOrigin<vec3i(0))){return invalidOwner();}let owner=Owner(vec3u(signedOrigin),1u<<exponent,1u);if(!ownerContains(owner,q)){return invalidOwner();}return owner;}fn residentCanonicalOwner(q:vec3u)->Owner{var size=min(p.maximumLeaf,8u);var origin=(q/vec3u(size))*vec3u(size);loop{if(all(origin+vec3u(size)<=p.dims)||size==1u){break;}size>>=1u;origin=(q/vec3u(size))*vec3u(size);}return Owner(origin,size,1u);}
fn pagedOwnerPublicationValid()->bool{if(arrayLength(&owners)<16u||owners[15]!=0x4f574e52u){return false;}let capacity=owners[3];let resident=owners[1];let accepted=owners[7];let recordPageOffset=owners[5];let payloadOffset=owners[6];return accepted!=0u&&owners[10]==1u&&owners[11]==accepted&&owners[2]==0u&&owners[12]==0u&&resident<=capacity&&owners[0]==capacity-resident&&capacity>0u&&recordPageOffset==16u+capacity&&payloadOffset==recordPageOffset+capacity&&16u<=arrayLength(&owners)&&capacity<=arrayLength(&owners)-16u&&recordPageOffset<=arrayLength(&owners)&&capacity<=arrayLength(&owners)-recordPageOffset&&payloadOffset<=arrayLength(&owners)&&capacity<=(arrayLength(&owners)-payloadOffset)/512u;}
fn ownerAt(q:vec3u)->Owner{if(!pagedOwnerPublicationValid()){return invalidOwner();}let capacity=owners[3];let resident=owners[1];let recordKeyOffset=16u;let recordPageOffset=owners[5];let payloadOffset=owners[6];let bd=(p.dims+vec3u(7u))/8u;let b=q/8u;if(any(b>=bd)||bd.x>0xffffffffu/bd.y){return invalidOwner();}let layer=bd.x*bd.y;if(b.z>0xffffffffu/layer){return invalidOwner();}let logical=b.x+b.y*bd.x+b.z*layer;let key=logical+1u;var low=0u;var high=resident;while(low<high){let middle=low+(high-low)/2u;let observed=owners[recordKeyOffset+middle];if(observed<key){low=middle+1u;}else{high=middle;}}if(low>=resident||owners[recordKeyOffset+low]!=key){return residentCanonicalOwner(q);}let encoded=owners[recordPageOffset+low];if(encoded==0u||encoded==INVALID||encoded>capacity){return invalidOwner();}let local=q%vec3u(8u);let physical=encoded-1u;let at=payloadOffset+physical*512u+local.x+local.y*8u+local.z*64u;let word=owners[at];if(word==0u){return residentCanonicalOwner(q);}return decodePagedOwner(word,q);}
// One support-row invocation probes the same 8^3 owner page dozens of times
// while enumerating its endpoint star, descriptor, and catalog selectors.
// Cache the exact last logical page and its immutable physical-page word.
// cache.x is 0 before validation, 1 for a valid publication, and 2 for a
// rejected publication; cache.y/cache.z are logical key/encoded page.
fn ownerAtCached(q:vec3u,cache:ptr<function,vec3u>)->Owner{
 if((*cache).x==0u){(*cache).x=select(2u,1u,pagedOwnerPublicationValid());}
 if((*cache).x!=1u){return invalidOwner();}
 let capacity=owners[3];let resident=owners[1];let recordKeyOffset=16u;
 let recordPageOffset=owners[5];let payloadOffset=owners[6];
 let bd=(p.dims+vec3u(7u))/8u;let b=q/8u;
 if(any(b>=bd)||bd.x>0xffffffffu/bd.y){return invalidOwner();}
 let layer=bd.x*bd.y;if(b.z>0xffffffffu/layer){return invalidOwner();}
 let key=b.x+b.y*bd.x+b.z*layer+1u;var encoded=0u;
 if((*cache).y==key){encoded=(*cache).z;}
 else{
  var low=0u;var high=resident;
  while(low<high){let middle=low+(high-low)/2u;
   let observed=owners[recordKeyOffset+middle];
   if(observed<key){low=middle+1u;}else{high=middle;}}
  if(low<resident&&owners[recordKeyOffset+low]==key){encoded=owners[recordPageOffset+low];}
  (*cache).y=key;(*cache).z=encoded;
 }
 if(encoded==0u){return residentCanonicalOwner(q);}
 if(encoded==INVALID||encoded>capacity){return invalidOwner();}
 let local=q%vec3u(8u);let physical=encoded-1u;
 let at=payloadOffset+physical*512u+local.x+local.y*8u+local.z*64u;
 let word=owners[at];
 if(word==0u){return residentCanonicalOwner(q);}
 return decodePagedOwner(word,q);
}
fn validCatalogCoarse()->bool{let rowCount=min(coarsePhi.rowCount,arrayLength(&coarsePhi.entries));let mask=0x3fffffffu;return coarsePhi.state==VALID&&(coarsePhi.generation&mask)==(p.generation&mask)&&coarsePhi.rowCount==rowCount&&rowCount>0u&&coarsePhi.maximumLeafSize==p.maximumLeaf&&all(coarsePhi.dimensions==p.dims)&&finite(coarsePhi.physicalCellSize)&&coarsePhi.physicalCellSize>0.0;}fn validCoarseGeneration()->bool{if(arrayLength(&fineTopologyControl)<8u){return false;}let mask=0x3fffffffu;let coarseGeneration=coarsePhi.generation&mask;let fineGeneration=p.generation&mask;let clean=fineTopologyControl[0]==0u&&fineTopologyControl[4]==1u&&fineTopologyControl[5]==0u&&fineTopologyControl[7]==0u;return clean&&coarseGeneration==fineGeneration;}fn validCoarse()->bool{let expectedWidth=fp.fineWidth*f32(fp.fineFactor);return validCatalogCoarse()&&validCoarseGeneration()&&abs(coarsePhi.physicalCellSize-expectedWidth)<=1e-5*max(coarsePhi.physicalCellSize,expectedWidth);}
fn coarseSlot(cellKey:u32,size:u32)->u32{let count=min(coarsePhi.rowCount,arrayLength(&coarsePhi.entries));let wantedLevel=31u-countLeadingZeros(size);let wantedMorton=powerMorton(cellKey);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let entry=coarsePhi.entries[middle];let entryLevel=31u-countLeadingZeros(entry.size);let entryMorton=powerMorton(entry.cellPlusOne-1u);if(entryLevel<wantedLevel||(entryLevel==wantedLevel&&entryMorton<wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let entry=coarsePhi.entries[low];if(entry.cellPlusOne==cellKey+1u&&entry.size==size){return low;}}return INVALID;}
fn coarseEntryRecord(slot:u32)->vec4f{if(slot==INVALID||slot>=arrayLength(&coarsePhi.entries)){return vec4f(0.);}let entry=coarsePhi.entries[slot];if((entry.flags&9u)!=9u||!finite(entry.phi)||!finite(entry.minimumPhi)||!finite(entry.maximumPhi)||entry.minimumPhi>entry.phi||entry.phi>entry.maximumPhi){return vec4f(0.);}return vec4f(entry.phi,entry.minimumPhi,entry.maximumPhi,1.);}
// Aanjaneya et al. 2017 Section 5 keeps a coarse octree level set specifically
// to carry signed-distance authority beyond the narrow fine SPGrid.  A pressure
// rebuild may change the leaf containing a point before that same-time coarse
// field is republished on the new leaves.  Sample the previous spatial
// publication at the new leaf centre; the current owner graph is redistanced
// below by the paper's local cube/Delaunay closest-point extension.
fn coarseCellSeedRecordValid(origin:vec3u,size:u32)->vec4f{if(size==0u||any(origin+vec3u(size)>p.dims)){return vec4f(0.);}let exact=coarseEntryRecord(coarseSlot(cell(origin),size));if(exact.w!=0.){return exact;}let q=min(origin+vec3u(size/2u),p.dims-vec3u(1u));var scale=1u;loop{let priorOrigin=(q/vec3u(scale))*vec3u(scale);let prior=coarseEntryRecord(coarseSlot(cell(priorOrigin),scale));if(prior.w!=0.){return prior;}if(scale>=coarsePhi.maximumLeafSize){break;}scale*=2u;}return vec4f(0.);}
fn coarseCellSeedRecord(origin:vec3u,size:u32)->vec4f{if(!validCoarse()){return vec4f(0.);}return coarseCellSeedRecordValid(origin,size);}
fn coarseSignFlag(owner:Owner)->u32{if(!validCatalogCoarse()||owner.valid==0u){return 0u;}let entry=coarseCellSeedRecordValid(owner.origin,owner.size);if(entry.w==0.){return ROW_COARSE_MIXED;}if(entry.z<0.){return ROW_COARSE_LIQUID;}if(entry.y>0.){return ROW_COARSE_AIR;}return ROW_COARSE_MIXED;}
fn committedStateBase()->u32{return p.rowDirectoryCapacity*2u;}
fn committedStateWord(word:u32)->u32{let at=committedStateBase()+word;if(at>=arrayLength(&committedRowDirectory)){return 0u;}return committedRowDirectory[at];}
fn committedCoreEnd()->u32{return committedStateWord(0u);}
fn committedRowCount()->u32{return committedStateWord(1u);}
fn committedGeneration()->u32{return committedStateWord(2u);}
fn committedValid()->u32{return committedStateWord(3u);}
fn mortonPart10(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn powerMorton(c:u32)->u32{let q=coord(c);return mortonPart10(q.x)|(mortonPart10(q.y)<<1u)|(mortonPart10(q.z)<<2u);}
fn powerLevel(size:u32)->u32{return 31u-countLeadingZeros(size);}
fn powerKeyLess(levelA:u32,mortonA:u32,levelB:u32,mortonB:u32)->bool{return levelA<levelB||(levelA==levelB&&mortonA<mortonB);}
fn powerDirectoryCount()->u32{return min(powerFaceControl[0],min(p.powerDirectoryCapacity,arrayLength(&sites)));}
fn findSiteInCount(c:u32,s:u32,count:u32)->u32{if(s==0u){return INVALID;}let bounded=min(count,min(p.powerDirectoryCapacity,arrayLength(&sites)));let level=powerLevel(s);let morton=powerMorton(c);
 var low=0u;var high=bounded;while(low<high){let middle=low+(high-low)/2u;let candidate=sites[middle];
  if(powerKeyLess(powerLevel(candidate.size),candidate.morton,level,morton)){low=middle+1u;}else{high=middle;}}
 if(low<bounded){let candidate=sites[low];if(candidate.cellPlusOne==c+1u&&candidate.size==s){return candidate.row;}}return INVALID;}
fn findSite(c:u32,s:u32)->u32{return findSiteInCount(c,s,powerDirectoryCount());}
fn containing(q:vec3u)->u32{var size=1u;loop{let o=(q/vec3u(size))*vec3u(size);let found=findSite(cell(o),size);if(found!=INVALID){return found;}if(size>=p.maximumLeaf){break;}size*=2u;}return INVALID;}
fn rowDirectoryCount()->u32{let state=p.rowDirectoryCapacity*2u;if(state>=arrayLength(&rowDirectory)){return 0u;}if(state+3u<arrayLength(&rowDirectory)&&rowDirectory[state+3u]==COMMITTED_TRANSITION_VALID){return rowDirectory[state+1u];}return rowDirectory[state];}
fn rowIdentityLess(cellA:u32,sizeA:u32,cellB:u32,sizeB:u32)->bool{return cellA<cellB||(cellA==cellB&&sizeA<sizeB);}
fn rowIdentitySlot(cellKey:u32,size:u32)->u32{let domain=p.dims.x*p.dims.y*p.dims.z;if(!supportedOwnerSize(size)||cellKey>=domain){return INVALID;}let origin=coord(cellKey);if(any(origin%vec3u(size)!=vec3u(0u))||any(origin+vec3u(size)>p.dims)){return INVALID;}let level=31u-countLeadingZeros(size);if(p.reservedDirectory>arrayLength(&rowDirectory)||level>5u||level*domain>arrayLength(&rowDirectory)-p.reservedDirectory){return INVALID;}let base=p.reservedDirectory+level*domain;if(base>=arrayLength(&rowDirectory)||cellKey>=arrayLength(&rowDirectory)-base){return INVALID;}return base+cellKey;}
fn rowOfIdentity(cellKey:u32,size:u32)->u32{let count=min(rowDirectoryCount(),p.rowDirectoryCapacity);let slot=rowIdentitySlot(cellKey,size);if(slot==INVALID){return INVALID;}let encoded=rowDirectory[slot];if(encoded==0u){return INVALID;}let row=encoded-1u;if(row>=count||row>=arrayLength(&rows)){return INVALID;}let candidate=rows[row];return select(INVALID,row,candidate.cell==cellKey&&candidate.size==size);}
fn supportedOwnerSize(size:u32)->bool{return size==1u||size==2u||size==4u||size==8u||size==16u||size==32u;}fn ownerContains(owner:Owner,q:vec3u)->bool{return owner.valid!=0u&&supportedOwnerSize(owner.size)&&owner.size<=p.maximumLeaf&&all(owner.origin%vec3u(owner.size)==vec3u(0u))&&all(owner.origin+vec3u(owner.size)<=p.dims)&&all(q>=owner.origin)&&all(q<owner.origin+vec3u(owner.size));}
fn recordDescriptorFailure(band:u32,row:Row,descriptor:u32,site:u32){if(band>=transitionControl.failureBand){return;}transitionControl.failureBand=band;let origin=coord(row.cell);transitionControl.failureStage=OWNER_FAILURE_DESCRIPTOR;transitionControl.failureRowCell=row.cell;transitionControl.failureRowSize=row.size;transitionControl.failureDescriptor=descriptor;transitionControl.failureTopology=INVALID;transitionControl.failureTransformFlags=0u;transitionControl.failureSelector=site;transitionControl.failureRawX=origin.x;transitionControl.failureRawY=origin.y;transitionControl.failureRawZ=origin.z;transitionControl.failureRequestedSize=row.size;transitionControl.failureResolvedCell=INVALID;transitionControl.failureBoundaryFlips=0u;transitionControl.failureOwnerCell=INVALID;transitionControl.failureOwnerSizeValid=0u;}
fn resolveBandDescriptor(descriptor:u32,band:u32)->Metric{var packed=INVALID;if((descriptor&COARSER_DESCRIPTOR)!=0u){let index=descriptor&0x1ffu;if((descriptor&0x40fffe00u)==0u&&index<arrayLength(&sameOrCoarserDirect)){packed=sameOrCoarserDirect[index];}}else{let index=descriptor&0x3ffffu;if((descriptor&0x40fc0000u)==0u&&index<arrayLength(&sameOrFinerDirect)){packed=sameOrFinerDirect[index];}}if(packed==INVALID){recordDescriptorFailure(band,rows[band],descriptor,6u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,descriptor);}let topology=packed&0xffffu;let transform=packed>>16u;let boundary=(descriptor>>16u)&0x3f00u;return Metric(topology,transform|boundary|VALID,0.,descriptor);}
fn describeBandRow(band:u32)->Metric{let row=rows[band];if(!supportedOwnerSize(row.size)||row.size>p.maximumLeaf||row.cell>=p.dims.x*p.dims.y*p.dims.z){recordDescriptorFailure(band,row,0u,1u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,TRANSITION_DESCRIPTOR);}let origin=vec3u(row.cell%p.dims.x,(row.cell/p.dims.x)%p.dims.y,row.cell/(p.dims.x*p.dims.y));if(any(origin%vec3u(row.size)!=vec3u(0u))||any(origin+vec3u(row.size)>p.dims)){recordDescriptorFailure(band,row,0u,2u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,TRANSITION_DESCRIPTOR);}let anchor=ownerAt(origin);if(!ownerContains(anchor,origin)||anchor.size!=row.size||any(anchor.origin!=origin)){recordDescriptorFailure(band,row,anchor.size|(anchor.origin.x<<8u)|(anchor.origin.y<<16u)|(anchor.origin.z<<24u),3u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,TRANSITION_DESCRIPTOR);}var sizes:array<u32,18>;var boundaryMask=0u;var finer=false;var coarser=false;for(var bit=0u;bit<18u;bit+=1u){let direction=DESCRIPTOR_DIRECTIONS[bit];var probe=vec3i(0);for(var axis=0u;axis<3u;axis+=1u){probe[axis]=select(select(i32(origin[axis]+row.size/2u),i32(origin[axis]+row.size),direction[axis]>0),i32(origin[axis])-1,direction[axis]<0);}if(any(probe<vec3i(0))||any(probe>=vec3i(p.dims))){if(bit<6u){boundaryMask|=1u<<bit;}sizes[bit]=row.size;continue;}let neighbor=ownerAt(vec3u(probe));if(!ownerContains(neighbor,vec3u(probe))||all(min(neighbor.origin+vec3u(neighbor.size),origin+vec3u(row.size))>max(neighbor.origin,origin))||neighbor.size*2u<row.size||neighbor.size>row.size*2u){recordDescriptorFailure(band,row,(bit<<26u)|(neighbor.size<<20u)|cell(vec3u(max(probe,vec3i(0)))),4u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,TRANSITION_DESCRIPTOR);}sizes[bit]=neighbor.size;finer=finer||neighbor.size<row.size;coarser=coarser||neighbor.size>row.size;}if(finer&&coarser){var packedSizes=0u;for(var bit=0u;bit<16u;bit+=1u){packedSizes|=(sizes[bit]&3u)<<(bit*2u);}recordDescriptorFailure(band,row,packedSizes,5u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,TRANSITION_DESCRIPTOR);}var descriptor=boundaryMask<<24u;if(!coarser){for(var bit=0u;bit<18u;bit+=1u){if(sizes[bit]==row.size){descriptor|=1u<<bit;}}}else{let child=(origin/vec3u(row.size))&vec3u(1u);descriptor|=COARSER_DESCRIPTOR|child.x|(child.y<<1u)|(child.z<<2u);let outward=vec3i(select(-1,1,child.x==1u),select(-1,1,child.y==1u),select(-1,1,child.z==1u));let wanted=array<vec3i,6>(vec3i(outward.x,0,0),vec3i(0,outward.y,0),vec3i(0,0,outward.z),vec3i(outward.x,outward.y,0),vec3i(outward.x,0,outward.z),vec3i(0,outward.y,outward.z));for(var coarseBit=0u;coarseBit<6u;coarseBit+=1u){for(var bit=0u;bit<18u;bit+=1u){if(all(DESCRIPTOR_DIRECTIONS[bit]==wanted[coarseBit])&&sizes[bit]==row.size*2u){descriptor|=1u<<(coarseBit+3u);}}}}return resolveBandDescriptor(descriptor,band);}
fn validPowerVelocity()->bool{return arrayLength(&powerVelocityControl)>=8u&&powerVelocityControl[0]==VALID&&powerVelocityControl[2]<=p.powerRowCapacity&&powerVelocityControl[7]==p.powerGeneration;}
fn validPowerDirectory()->bool{let count=powerVelocityControl[2];
 return count<=p.powerDirectoryCapacity&&count<=arrayLength(&sites);}
// Aanjaneya et al. 2017 Section 5 computes the velocity interpolant used at
// every point reached by fine-level-set advection.  The interface prefix is
// the allocation seed, not the velocity domain: all cells in the published
// fine narrow band require regular-face extrapolation and a full centre
// vector.  Rows added after this core exist only to close local cube/Delaunay
// stencils.
fn fineBandBrickCount()->u32{if(arrayLength(&worklist)<5u
 ||arrayLength(&metadata)<fp.pageCapacity*10u
 ||arrayLength(&sampleFlags)<fp.pageCapacity*fp.samplesPerBrick){return INVALID;}
 let residentCount=worklist[0];if(residentCount>fp.pageCapacity||worklist[1]!=fp.generation
 ||worklist[3]!=1u||worklist[4]!=1u){return INVALID;}return residentCount;}
fn topologyDeltaAccepted()->bool{
 if(p.rowDeltaEnabled==0u||p.rowDeltaControl>arrayLength(&rowDelta)
    ||arrayLength(&rowDelta)-p.rowDeltaControl<16u){return false;}
 let base=p.rowDeltaControl;let current=rowDelta[base];let previous=rowDelta[base+1u];
 let carried=rowDelta[base+2u];let added=rowDelta[base+3u];let retired=rowDelta[base+4u];
 let dirty=rowDelta[base+5u];let affected=rowDelta[base+6u];
 let currentMapFits=p.rowDeltaNewToOld<=arrayLength(&rowDelta)
   &&current<=arrayLength(&rowDelta)-p.rowDeltaNewToOld;
 let previousMapFits=p.rowDeltaOldToNew<=arrayLength(&rowDelta)
   &&previous<=arrayLength(&rowDelta)-p.rowDeltaOldToNew;
 let dirtyFits=p.rowDeltaDirty<=arrayLength(&rowDelta)
   &&dirty<=arrayLength(&rowDelta)-p.rowDeltaDirty;
 let affectedFits=p.rowDeltaAffected<=arrayLength(&rowDelta)
   &&affected<=arrayLength(&rowDelta)-p.rowDeltaAffected;
 // The producer delta is sufficient to rebuild the complete candidate graph.
 // An exact committed predecessor is only an optimization input: when it is
 // absent, committedTransitionValid() is false and transition classification
 // marks every row affected, bypassing every carry path. Requiring the
 // predecessor here made one rejected publication permanently self-lock.
 return currentMapFits&&previousMapFits&&dirtyFits&&affectedFits
   &&rowDelta[base+7u]==p.powerGeneration&&rowDelta[base+8u]==ROW_DELTA_VALID
   &&current==powerVelocityControl[2]&&(previous!=0u||affected==current)
   &&dirty<=affected&&affected<=current&&carried<=previous
   &&current==carried+added&&current==previous+added-retired;
}
fn committedRowOfIdentity(cellKey:u32,size:u32)->u32{
 let count=min(committedRowCount(),p.rowDirectoryCapacity);
 let slot=rowIdentitySlot(cellKey,size);
 if(slot==INVALID||slot>=arrayLength(&committedRowDirectory)){return INVALID;}
 let encoded=committedRowDirectory[slot];if(encoded==0u){return INVALID;}
 let row=encoded-1u;if(row>=count||row>=arrayLength(&committedRows)){return INVALID;}
 let candidate=committedRows[row];
 return select(INVALID,row,candidate.cell==cellKey&&candidate.size==size);
}
fn committedContainingRow(q:vec3u)->u32{
 var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);
   let row=committedRowOfIdentity(cell(origin),size);if(row!=INVALID&&row<arrayLength(&committedRows)
      &&all(q>=origin)&&all(q<origin+vec3u(size))){return row;}
   if(size>=p.maximumLeaf){break;}size*=2u;}return INVALID;
}
fn topologyDescriptor(row:Row,ownerCache:ptr<function,vec3u>)->vec2u{
 if(!supportedOwnerSize(row.size)||row.size>p.maximumLeaf){fail(BAD_ROW,row.cell);return vec2u(INVALID);}
 let origin=coord(row.cell);let anchor=ownerAtCached(origin,ownerCache);
 if(!ownerContains(anchor,origin)||anchor.size!=row.size||any(anchor.origin!=origin)){
   fail(BAD_ROW,row.cell);return vec2u(INVALID);}
 var sizes:array<u32,18>;var boundaryMask=0u;var finer=false;var coarser=false;
 for(var bit=0u;bit<18u;bit+=1u){let direction=DESCRIPTOR_DIRECTIONS[bit];var probe=vec3i(0);
   for(var axis=0u;axis<3u;axis+=1u){
     probe[axis]=select(select(i32(origin[axis]+row.size/2u),i32(origin[axis]+row.size),
       direction[axis]>0),i32(origin[axis])-1,direction[axis]<0);}
   if(any(probe<vec3i(0))||any(probe>=vec3i(p.dims))){
     if(bit<6u){boundaryMask|=1u<<bit;}sizes[bit]=row.size;continue;}
   let neighbor=ownerAtCached(vec3u(probe),ownerCache);
   if(!ownerContains(neighbor,vec3u(probe))||neighbor.size*2u<row.size
      ||neighbor.size>row.size*2u){fail(BAD_ROW,row.cell);return vec2u(INVALID);}
   sizes[bit]=neighbor.size;finer=finer||neighbor.size<row.size;coarser=coarser||neighbor.size>row.size;}
 if(finer&&coarser){fail(BAD_ROW,row.cell);return vec2u(INVALID);}
 var descriptor=boundaryMask<<24u;
 if(!coarser){for(var bit=0u;bit<18u;bit+=1u){
   if(sizes[bit]==row.size){descriptor|=1u<<bit;}}}
 else{let child=(origin/vec3u(row.size))&vec3u(1u);
   descriptor|=COARSER_DESCRIPTOR|child.x|(child.y<<1u)|(child.z<<2u);
   let outward=vec3i(select(-1,1,child.x==1u),select(-1,1,child.y==1u),
     select(-1,1,child.z==1u));
   let wanted=array<vec3i,6>(vec3i(outward.x,0,0),vec3i(0,outward.y,0),
     vec3i(0,0,outward.z),vec3i(outward.x,outward.y,0),
     vec3i(outward.x,0,outward.z),vec3i(0,outward.y,outward.z));
   for(var coarseBit=0u;coarseBit<6u;coarseBit+=1u){for(var bit=0u;bit<18u;bit+=1u){
     if(all(DESCRIPTOR_DIRECTIONS[bit]==wanted[coarseBit])&&sizes[bit]==row.size*2u){
       descriptor|=1u<<(coarseBit+3u);}}}}
 var packed=INVALID;if((descriptor&COARSER_DESCRIPTOR)!=0u){
   let index=descriptor&0x1ffu;if((descriptor&0x40fffe00u)==0u
     &&index<arrayLength(&sameOrCoarserDirect)){packed=sameOrCoarserDirect[index];}}
 else{let index=descriptor&0x3ffffu;if((descriptor&0x40fc0000u)==0u
   &&index<arrayLength(&sameOrFinerDirect)){packed=sameOrFinerDirect[index];}}
 if(packed==INVALID){fail(SOURCE,row.cell);return vec2u(INVALID);}
 return vec2u(packed&0xffffu,(packed>>16u)|((descriptor>>16u)&0x3f00u));
}
fn invalidSupportCandidate()->SupportCandidate{return SupportCandidate(INVALID,0u);}
fn supportCandidateBase(source:u32)->u32{return source*MAX_GUARDS;}
fn writeSupportCandidate(base:u32,slot:u32,value:SupportCandidate){
 let at=base+slot;if(at>=arrayLength(&supportCandidates)){fail(CAPACITY,at);return;}
 supportCandidates[at]=value;
}
fn candidateFromOwner(owner:Owner,row:Row)->SupportCandidate{
 if(!ownerContains(owner,owner.origin)||owner.size==0u||any(owner.origin+vec3u(owner.size)>p.dims)){
   fail(BAD_ROW,row.cell);return invalidSupportCandidate();}
 return SupportCandidate(cell(owner.origin)+1u,owner.size);
}
fn selectorSupportCandidate(row:Row,selector:u32,transform:u32,
 ownerCache:ptr<function,vec3u>)->SupportCandidate{
 if(selector>=arrayLength(&tetraVertices)){fail(SOURCE,row.cell);return invalidSupportCandidate();}
 let vertex=tetraVertices[selector].v;
 if(!finite(vertex.x)||!finite(vertex.y)||!finite(vertex.z)||!finite(vertex.w)||vertex.w<=0.){
   fail(SOURCE,row.cell);return invalidSupportCandidate();}
 let center=vec3f(coord(row.cell))+.5*f32(row.size);
 let point=center+f32(row.size)*inversePowerTransform(vertex.xyz,transform&63u);
 let sizeFloat=f32(row.size)*vertex.w;let size=u32(round(sizeFloat));
 let origin=vec3i(round(point-.5*f32(size)));
 if(size==0u||abs(sizeFloat-f32(size))>1e-4){
   fail(BAD_ROW,row.cell);return invalidSupportCandidate();}
 // The Section 5 interpolation mesh contains virtual cell centres across a
 // container wall.  Publish the exact reflected in-domain selector that
 // finalSignedVector will read; dropping the virtual selector here leaves a
 // valid boundary tetrahedron with a missing velocity vertex.
 let extended=velocityExtendedOrigin(origin,size);
 if(extended.w<0){return invalidSupportCandidate();}
 let resolved=vec3u(extended.xyz);
 let owner=ownerAtCached(resolved,ownerCache);
 if(!ownerContains(owner,resolved)){
   fail(BAD_ROW,row.cell);return invalidSupportCandidate();}
 let outside=any(origin<vec3i(0))||any(origin+vec3i(i32(size))>vec3i(p.dims));
 // Interior catalog vertices are exact identities. A wall ghost instead
 // copies the full vector of the exact reflected adaptive owner; the octree
 // is not required to be mirror-symmetric across the container boundary.
 if(!outside&&(owner.size!=size||any(owner.origin!=resolved))){
   fail(BAD_ROW,row.cell);return invalidSupportCandidate();}
 return candidateFromOwner(owner,row);
}
fn emitRegularFaceEndpointSupport(row:Row,base:u32,
 ownerCache:ptr<function,vec3u>){
 let origin=coord(row.cell);let half=max(1u,row.size/2u);
 let sampleCount=select(1u,4u,row.size>1u);var request=0u;
 for(var axis=0u;axis<3u;axis+=1u){let ta=(axis+1u)%3u;let tb=(axis+2u)%3u;
   for(var side=0u;side<2u;side+=1u){for(var sample=0u;sample<4u;sample+=1u){
     var candidate=invalidSupportCandidate();let isActive=sample<sampleCount
       &&!(side==0u&&origin[axis]==0u)
       &&!(side==1u&&origin[axis]+row.size>=p.dims[axis]);
     if(isActive){var probe=origin;
       probe[axis]=select(origin[axis]-1u,origin[axis]+row.size,side==1u);
       probe[ta]+=select(0u,half,(sample&1u)!=0u);
       probe[tb]+=select(0u,half,(sample&2u)!=0u);
       let owner=ownerAtCached(probe,ownerCache);if(owner.valid==0u){fail(BAD_ROW,row.cell);}
       else{candidate=candidateFromOwner(owner,row);}}
     writeSupportCandidate(base,request,candidate);request+=1u;
   }}
 }
 if(request!=MAX_ENDPOINTS){fail(SOURCE,row.cell);}
}
fn emitUniformCatalogSupport(row:Row,base:u32,
 ownerCache:ptr<function,vec3u>){
 let origin=coord(row.cell);var request=MAX_ENDPOINTS;
 for(var z=-1;z<=1;z+=1){for(var y=-1;y<=1;y+=1){for(var x=-1;x<=1;x+=1){
   if(x==0&&y==0&&z==0){continue;}var candidate=invalidSupportCandidate();
   var probe=vec3i(0);let direction=vec3i(x,y,z);
   for(var axis=0u;axis<3u;axis+=1u){
     probe[axis]=select(select(i32(origin[axis]+row.size/2u),i32(origin[axis]+row.size),
       direction[axis]>0),i32(origin[axis])-1,direction[axis]<0);}
   if(all(probe>=vec3i(0))&&all(probe<vec3i(p.dims))){
     let owner=ownerAtCached(vec3u(probe),ownerCache);if(owner.valid==0u){fail(BAD_ROW,row.cell);}
     else{candidate=candidateFromOwner(owner,row);}}
   writeSupportCandidate(base,request,candidate);request+=1u;
 }}}
 if(request!=MAX_ENDPOINTS+UNIFORM_GUARDS){fail(SOURCE,row.cell);}
}
fn emitDelaunayCatalogSupport(row:Row,base:u32,header:TetraHeader,transform:u32,
 ownerCache:ptr<function,vec3u>){
 var selectors:array<u32,36>;var selectorCount=0u;
 for(var local=0u;local<header.count;local+=1u){
   let packed=tetrahedra[header.first+local];
   let requested=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);
   for(var lane=0u;lane<3u;lane+=1u){let selector=requested[lane];var duplicate=false;
     for(var prior=0u;prior<selectorCount;prior+=1u){duplicate=duplicate||selectors[prior]==selector;}
     if(duplicate){continue;}if(selectorCount>=MAX_CATALOG_GUARDS){fail(CAPACITY,row.cell);return;}
     selectors[selectorCount]=selector;selectorCount+=1u;
   }
 }
 for(var slot=0u;slot<selectorCount;slot+=1u){
   writeSupportCandidate(base,MAX_ENDPOINTS+slot,
     selectorSupportCandidate(row,selectors[slot],transform,ownerCache));}
}
fn emitCatalogNeighborhood(source:u32,tier:u32){
 let core=control.initialRows;let state=p.rowDirectoryCapacity*2u;
 let support1End=select(core,rowDirectory[state+1u],state+1u<arrayLength(&rowDirectory));
 let support2End=select(support1End,rowDirectory[state+2u],state+2u<arrayLength(&rowDirectory));
 let support3End=select(support2End,rowDirectory[state+3u],state+3u<arrayLength(&rowDirectory));
 let sourceBegin=select(select(select(0u,core,tier==2u),support1End,tier==3u),
   support2End,tier==4u);
 let sourceEnd=select(select(select(core,support1End,tier==2u),support2End,tier==3u),
   support3End,tier==4u);
 if(sourceEnd<sourceBegin||source>=sourceEnd-sourceBegin){return;}
 let rowIndex=sourceBegin+source;let base=supportCandidateBase(source);
 if(base>arrayLength(&supportCandidates)||MAX_GUARDS>arrayLength(&supportCandidates)-base){
   fail(CAPACITY,base);return;}
 for(var slot=0u;slot<MAX_GUARDS;slot+=1u){
   supportCandidates[base+slot]=invalidSupportCandidate();}
 if(control.flags!=0u||rowIndex>=arrayLength(&rows)){return;}
 var ownerCache=vec3u(0u,INVALID,0u);
 let row=rows[rowIndex];emitRegularFaceEndpointSupport(row,base,&ownerCache);
 let metric=topologyDescriptor(row,&ownerCache);
 if(metric.x==INVALID||metric.x>=arrayLength(&tetraHeaders)){fail(SOURCE,row.cell);return;}
 let header=tetraHeaders[metric.x];
 if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first
    ||header.count>MAX_TETRA){fail(SOURCE,row.cell);return;}
 if((header.flags&1u)!=0u){emitUniformCatalogSupport(row,base,&ownerCache);return;}
 if(header.count==0u){fail(SOURCE,row.cell);return;}
 emitDelaunayCatalogSupport(row,base,header,metric.y,&ownerCache);
}
@compute @workgroup_size(1)fn prepareFaceBandDelta(){
 control.generation=p.generation;control.flags=0u;control.firstError=INVALID;
 control.reserved0=INVALID;control.rowCount=0u;control.faceCount=0u;
 control.incidenceCount=0u;control.initialRows=0u;
 control.seedCount=0u;control.acceptedCount=0u;
 control.unresolvedCount=0u;control.sampleFailures=0u;
 control.coarsePhiSamples=0u;control.coarsePhiFailures=0u;
 control.bandPhiExtensions=0u;control.closestPointFaces=0u;
 control.closestPointFailures=0u;control.liquidInterpolationFailures=0u;
 control.cptNoOwnerFailures=0u;control.cptSupportOwnerFailures=0u;
 control.cptNoContainingSimplexFailures=0u;
 control.cptMissingLiquidVertexFailures=0u;
 control.firstFaceEmissionFailure=INVALID;control.firstPhiFailureSlot=INVALID;
 control.firstPhiFailure=INVALID;control.firstClosestPointFailure=INVALID;
 control.firstVectorFailure=INVALID;control.firstCptNoOwnerFailure=INVALID;
 control.firstCptSupportOwnerFailure=INVALID;control.firstCptNoSimplexFailure=INVALID;
 control.firstCptMissingVertexFailure=INVALID;control.valid=0u;
 indirect[0]=0u;indirect[1]=1u;indirect[2]=1u;indirect[3]=0u;indirect[4]=1u;indirect[5]=1u;
 indirect[60]=0u;indirect[61]=1u;indirect[62]=1u;
 indirect[63]=0u;indirect[64]=1u;indirect[65]=1u;
 if(fineBandBrickCount()==INVALID||!validPowerVelocity()||!validPowerDirectory()
   ||!topologyDeltaAccepted()){fail(SOURCE,0u);return;}
 let core=powerVelocityControl[2];if(core==0u||core>p.rowCapacity||core>p.powerRowCapacity
   ||core>p.rowDirectoryCapacity){fail(CAPACITY,core);return;}
 let candidateState=p.rowDirectoryCapacity*2u;
 if(candidateState+3u>=arrayLength(&rowDirectory)){fail(CAPACITY,candidateState);return;}
 // Candidate and scratch arenas are constructor-sized from the same immutable
 // row plan. Initialize the exact three-tier publication epoch here instead
 // of launching a second singleton solely to recheck those host invariants.
 rowDirectory[candidateState]=core;rowDirectory[candidateState+1u]=core;
 rowDirectory[candidateState+2u]=core;rowDirectory[candidateState+3u]=0u;
 control.initialRows=core;control.rowCount=core;
 indirect[60]=(core+63u)/64u;
}
@compute @workgroup_size(64)fn buildFaceBandTopologyDelta(@builtin(global_invocation_id)g:vec3u){
 if(control.flags!=0u){return;}
 let band=g.x;let core=control.initialRows;if(band>=core){return;}
 // Publish the immutable power prefix directly from its canonical directory.
 if(band>=arrayLength(&sites)||band>=arrayLength(&rows)
    ||band*2u+1u>=arrayLength(&rowDirectory)){fail(CAPACITY,band);return;}
 let entry=sites[band];if(entry.cellPlusOne==0u||entry.row!=band||entry.size==0u){
   fail(BAD_ROW,band);return;}
 let cellKey=entry.cellPlusOne-1u;let origin=coord(cellKey);let owner=ownerAt(origin);
 if(entry.morton!=powerMorton(cellKey)||!ownerContains(owner,origin)
    ||owner.size!=entry.size||any(owner.origin!=origin)){fail(BAD_ROW,band);return;}
 let priorEncoded=rowDelta[p.rowDeltaNewToOld+band]&0x7fffffffu;
 rows[band]=Row(cellKey,band,ROW_COARSE|ROW_CORE,entry.size,0.,0.,0.,bitcast<f32>(priorEncoded));
 rowDirectory[band*2u]=cellKey;rowDirectory[band*2u+1u]=band;
 let identitySlot=rowIdentitySlot(cellKey,entry.size);
 if(identitySlot==INVALID){fail(BAD_DIRECTORY,band);return;}
 rowDirectory[identitySlot]=band+1u;
}
@compute @workgroup_size(64)fn emitFaceBandCatalogSupport1(
 @builtin(global_invocation_id)g:vec3u){
 emitCatalogNeighborhood(g.x,1u);
}
fn supportRead(position:u32,side:u32)->SupportCandidate{
 if(side==1u){return supportCandidateScratch[position];}
 return supportCandidates[position];
}
fn supportWrite(position:u32,value:SupportCandidate,side:u32){
 if(side==0u){supportCandidateScratch[position]=value;}
 else{supportCandidates[position]=value;}
}
fn supportLaneRange(count:u32,lane:u32)->vec2u{
 let width=count/256u;let remainder=count%256u;
 let begin=lane*width+min(lane,remainder);
 return vec2u(begin,begin+width+select(0u,1u,lane<remainder));
}
fn supportDigit(value:SupportCandidate,digit:u32)->u32{
 if(digit<2u){return (value.size>>(digit*4u))&15u;}
 return (value.cellPlusOne>>((digit-2u)*4u))&15u;
}
fn faceBandRadixDigits()->u32{
 // Size occupies two nibbles. Cell identities need only the nibbles capable
 // of representing this immutable domain, rather than all eight u32 nibbles.
 // Round to an even pass count so the canonical result remains on side zero
 // of both ping-pong arenas without a copy dispatch.
 let domain=p.dims.x*p.dims.y*p.dims.z;var cellDigits=1u;var limit=15u;
 while(domain>limit&&cellDigits<8u){cellDigits+=1u;limit=(limit<<4u)|15u;}
 let digits=2u+cellDigits;return digits+(digits&1u);
}
fn supportIdentityLess(cellPlusOneA:u32,sizeA:u32,cellPlusOneB:u32,sizeB:u32)->bool{
 return cellPlusOneA<cellPlusOneB||(cellPlusOneA==cellPlusOneB&&sizeA<sizeB);
}
fn publishedSupportRecord(cellPlusOne:u32,size:u32,tier:u32,core:u32)->bool{
 // Every completed tier publishes the collision-free identity slot before
 // the next tier emits. rowOfIdentity validates both the live prefix bound
 // and the pointed-to Row, so stale words from an older candidate epoch
 // remain unreachable without rescanning any sorted prefix.
 return rowOfIdentity(cellPlusOne-1u,size)!=INVALID;
}
fn sortUniqueFaceBandCatalogSupport(tier:u32,lane:u32){
 let candidateState=p.rowDirectoryCapacity*2u;
 // Storage and atomic loads are dynamically non-uniform to Tint even though
 // this dispatch reads one immutable transaction. Lane zero snapshots the
 // transaction and workgroupUniformLoad makes every barrier predecessor
 // formally uniform.
 if(lane==0u){
   let core=control.initialRows;
   let support1End=rowDirectory[candidateState+1u];
   let support2End=rowDirectory[candidateState+2u];
   let support3End=rowDirectory[candidateState+3u];
   let clean=control.flags==0u;
   let prefixValid=support1End>=core&&support1End<=p.rowCapacity
     &&(tier<3u||(support2End>=support1End&&support2End<=p.rowCapacity))
     &&(tier<4u||(support3End>=support2End&&support3End<=p.rowCapacity));
   if(clean&&!prefixValid){fail(BAD_DIRECTORY,
     select(select(support1End,support2End,tier==3u),support3End,tier==4u));}
   let valid=clean&&prefixValid;
   var sourceCount=core;var stride=MAX_GUARDS;
   if(tier==2u){sourceCount=select(0u,support1End-core,prefixValid);}
   else if(tier==3u){sourceCount=select(0u,support2End-support1End,prefixValid);}
   else if(tier==4u){sourceCount=select(0u,support3End-support2End,prefixValid);}
   supportSortBins[0u]=select(0u,1u,valid);
   supportSortBins[1u]=core;supportSortBins[2u]=support1End;
   supportSortBins[3u]=sourceCount*stride;supportSortBins[258u]=support2End;
   supportSortBins[259u]=support3End;
 }
 let enabled=workgroupUniformLoad(&supportSortBins[0u]);
 let core=workgroupUniformLoad(&supportSortBins[1u]);
 let support1End=workgroupUniformLoad(&supportSortBins[2u]);
 let support2End=workgroupUniformLoad(&supportSortBins[258u]);
 let support3End=workgroupUniformLoad(&supportSortBins[259u]);
 let count=workgroupUniformLoad(&supportSortBins[3u]);
 if(enabled==0u){return;}
 let range=supportLaneRange(count,lane);var side=0u;
 for(var digit=0u;digit<faceBandRadixDigits();digit+=1u){
   var localCounts:array<u32,16>;for(var bin=0u;bin<16u;bin+=1u){localCounts[bin]=0u;}
   for(var position=range.x;position<range.y;position+=1u){
     localCounts[supportDigit(supportRead(position,side),digit)]+=1u;}
   for(var bin=0u;bin<16u;bin+=1u){
     supportSortBins[lane*16u+bin]=localCounts[bin];}
   workgroupBarrier();
   if(lane==0u){var binBase=0u;for(var bin=0u;bin<16u;bin+=1u){
     var total=0u;for(var sourceLane=0u;sourceLane<256u;sourceLane+=1u){
       let record=sourceLane*16u+bin;let laneCount=supportSortBins[record];
       supportSortBins[record]=binBase+total;total+=laneCount;}binBase+=total;}}
   workgroupBarrier();var cursors:array<u32,16>;
   for(var bin=0u;bin<16u;bin+=1u){cursors[bin]=supportSortBins[lane*16u+bin];}
   for(var position=range.x;position<range.y;position+=1u){
     let value=supportRead(position,side);let bin=supportDigit(value,digit);
     supportWrite(cursors[bin],value,side);cursors[bin]+=1u;}
   storageBarrier();workgroupBarrier();side=1u-side;storageBarrier();workgroupBarrier();
 }
 var localUnique=0u;
 for(var position=range.x;position<range.y;position+=1u){
   let value=supportCandidates[position];
   if(value.cellPlusOne==INVALID||value.cellPlusOne==0u||value.size==0u){continue;}
   if(position>0u){let prior=supportCandidates[position-1u];
     if(prior.cellPlusOne==value.cellPlusOne&&prior.size==value.size){continue;}}
   if(!publishedSupportRecord(value.cellPlusOne,value.size,tier,core)){localUnique+=1u;}
 }
 supportSortBins[lane]=localUnique;workgroupBarrier();
 if(lane==0u){var total=0u;for(var sourceLane=0u;sourceLane<256u;sourceLane+=1u){
   let laneCount=supportSortBins[sourceLane];supportSortBins[sourceLane]=total;total+=laneCount;}
   supportSortBins[256u]=total;
   let begin=select(select(select(core,support1End,tier==2u),support2End,tier==3u),
     support3End,tier==4u);
   let capacityAccepted=begin<=p.rowCapacity&&total<=p.rowCapacity-begin;
   if(!capacityAccepted){fail(CAPACITY,total);}
   let accepted=capacityAccepted&&control.flags==0u;
   supportSortBins[257u]=select(0u,1u,accepted);
 }
 let publishEnabled=workgroupUniformLoad(&supportSortBins[257u]);
 if(publishEnabled==0u){return;}
 let begin=select(select(select(core,support1End,tier==2u),support2End,tier==3u),
   support3End,tier==4u);
 var output=supportSortBins[lane];
 for(var position=range.x;position<range.y;position+=1u){
   let value=supportCandidates[position];
   if(value.cellPlusOne==INVALID||value.cellPlusOne==0u||value.size==0u){continue;}
   if(position>0u){let prior=supportCandidates[position-1u];
     if(prior.cellPlusOne==value.cellPlusOne&&prior.size==value.size){continue;}}
   if(publishedSupportRecord(value.cellPlusOne,value.size,tier,core)){continue;}
   let band=begin+output;let cellKey=value.cellPlusOne-1u;
   if(band>=arrayLength(&rows)||band*2u+1u>=arrayLength(&rowDirectory)
      ||(tier==1u&&output*3u+2u>=arrayLength(&rowDirectoryScratch))){
     fail(BAD_ROW,cellKey);continue;}
   let old=committedRowOfIdentity(cellKey,value.size);
   let priorEncoded=select(0u,old+1u,old!=INVALID);
   var flag=ROW_SUPPORT1;if(tier==2u){flag=ROW_SUPPORT2;}
   else if(tier==3u){flag=ROW_SUPPORT3_NODE;}
   else if(tier==4u){flag=ROW_SUPPORT3_ENDPOINT;}
   rows[band]=Row(cellKey,band+1u,ROW_COARSE|flag,value.size,0.,0.,0.,
     bitcast<f32>(priorEncoded));
   rowDirectory[band*2u]=cellKey;rowDirectory[band*2u+1u]=band;
   let identitySlot=rowIdentitySlot(cellKey,value.size);
   if(identitySlot==INVALID){fail(BAD_DIRECTORY,band);continue;}
   rowDirectory[identitySlot]=band+1u;
   if(tier==1u){rowDirectoryScratch[output*3u]=value.cellPlusOne;
     rowDirectoryScratch[output*3u+1u]=value.size;
     rowDirectoryScratch[output*3u+2u]=tier;}
   output+=1u;
 }
 storageBarrier();workgroupBarrier();
 if(lane==0u&&control.flags==0u){
   let uniqueTotal=supportSortBins[256u];let end=begin+uniqueTotal;
   // rowDirectoryCount() exposes only fully published tier prefixes. This
   // makes the direct identity slots above an exact O(1) membership table
   // for the following tier while retaining the canonical sorted stream.
   rowDirectory[candidateState]=end;
   if(tier==1u){rowDirectory[candidateState+1u]=end;
     indirect[63]=(uniqueTotal+63u)/64u;}
   else if(tier==2u){rowDirectory[candidateState+2u]=end;
     indirect[60]=(uniqueTotal+63u)/64u;}
   else if(tier==3u){rowDirectory[candidateState+3u]=end;
     indirect[63]=(uniqueTotal+63u)/64u;}
   else{rowDirectory[candidateState]=end;
     control.rowCount=end;indirect[0]=1u;indirect[3]=(end+63u)/64u;}
 }
}
@compute @workgroup_size(256)fn sortUniqueFaceBandCatalogSupport1(
 @builtin(local_invocation_index)lane:u32){
 sortUniqueFaceBandCatalogSupport(1u,lane);
}
@compute @workgroup_size(64)fn emitFaceBandCatalogSupport2(
 @builtin(global_invocation_id)g:vec3u){
 emitCatalogNeighborhood(g.x,2u);
}
@compute @workgroup_size(256)fn sortUniqueFaceBandCatalogSupport2(
 @builtin(local_invocation_index)lane:u32){
 sortUniqueFaceBandCatalogSupport(2u,lane);
}
@compute @workgroup_size(64)fn emitFaceBandCatalogSupport3(
 @builtin(global_invocation_id)g:vec3u){
 emitCatalogNeighborhood(g.x,3u);
}
@compute @workgroup_size(256)fn sortUniqueFaceBandCatalogSupport3(
 @builtin(local_invocation_index)lane:u32){
 sortUniqueFaceBandCatalogSupport(3u,lane);
}
@compute @workgroup_size(64)fn emitFaceBandEndpointSupport4(
 @builtin(global_invocation_id)g:vec3u){
 emitCatalogNeighborhood(g.x,4u);
}
@compute @workgroup_size(256)fn sortUniqueFaceBandEndpointSupport4(
 @builtin(local_invocation_index)lane:u32){
 sortUniqueFaceBandCatalogSupport(4u,lane);
 // The terminal tier already owns the one 256-lane deterministic workgroup
 // and binds both directory arenas. Complete the canonical directory radix
 // and validation before returning instead of relaunching the same lanes.
 storageBarrier();workgroupBarrier();
 sortFaceBandRowDirectoryInWorkgroup(lane);
 storageBarrier();workgroupBarrier();
 let count=control.rowCount;
 for(var index=lane;index<count;index+=256u){
   validateFaceBandRowDirectoryIndex(index,count);}
}
@compute @workgroup_size(256)fn sortValidateFaceBandRowDirectory(
 @builtin(local_invocation_index)lane:u32){
 sortFaceBandRowDirectoryInWorkgroup(lane);
 storageBarrier();workgroupBarrier();
 let count=control.rowCount;
 for(var index=lane;index<count;index+=256u){
   validateFaceBandRowDirectoryIndex(index,count);
 }
}
@compute @workgroup_size(64)fn validateFaceBandRowDirectory(
 @builtin(global_invocation_id)g:vec3u){
 let index=g.x;
 let count=control.rowCount;
 if(index<count){validateFaceBandRowDirectoryIndex(index,count);}
}
@compute @workgroup_size(64)fn classifyFaceBandRowSigns(@builtin(global_invocation_id)g:vec3u){
 let band=g.x;let count=control.rowCount;if(band>=count||band>=arrayLength(&rows)){return;}
 var row=rows[band];let origin=coord(row.cell);let owner=ownerAt(origin);
 if(!ownerContains(owner,origin)||owner.size!=row.size||any(owner.origin!=origin)){
   fail(BAD_ROW,row.cell);return;}
 let signFlag=coarseSignFlag(owner);if(signFlag==0u){fail(SOURCE,band);return;}
 row.flags|=signFlag;rows[band]=row;
}
fn rowSortRead(position:u32,side:u32)->vec2u{
 return select(vec2u(rowDirectory[position*2u],rowDirectory[position*2u+1u]),
   vec2u(rowDirectoryScratch[position*2u],rowDirectoryScratch[position*2u+1u]),side==1u);
}
fn rowSortWrite(position:u32,value:vec2u,side:u32){
 if(side==0u){rowDirectoryScratch[position*2u]=value.x;rowDirectoryScratch[position*2u+1u]=value.y;}
 else{rowDirectory[position*2u]=value.x;rowDirectory[position*2u+1u]=value.y;}
}
fn rowSortDigit(value:vec2u,digit:u32)->u32{
 if(value.y>=arrayLength(&rows)){return 0u;}let size=rows[value.y].size;
 if(digit<2u){return (size>>(digit*4u))&15u;}
 return (value.x>>((digit-2u)*4u))&15u;
}
fn rowSortLaneRange(count:u32,lane:u32)->vec2u{
 let width=count/256u;let remainder=count%256u;let begin=lane*width+min(lane,remainder);
 return vec2u(begin,begin+width+select(0u,1u,lane<remainder));
}
fn sortFaceBandRowDirectoryInWorkgroup(lane:u32){
 let count=control.rowCount;let range=rowSortLaneRange(count,lane);var side=0u;
 for(var digit=0u;digit<faceBandRadixDigits();digit+=1u){var localCounts:array<u32,16>;
   for(var bin=0u;bin<16u;bin+=1u){localCounts[bin]=0u;}
   for(var position=range.x;position<range.y;position+=1u){
     let bin=rowSortDigit(rowSortRead(position,side),digit);localCounts[bin]+=1u;}
   for(var bin=0u;bin<16u;bin+=1u){supportSortBins[lane*16u+bin]=localCounts[bin];}
   workgroupBarrier();if(lane==0u){var binBase=0u;for(var bin=0u;bin<16u;bin+=1u){
       var total=0u;for(var sourceLane=0u;sourceLane<256u;sourceLane+=1u){
         let record=sourceLane*16u+bin;let laneCount=supportSortBins[record];
         supportSortBins[record]=binBase+total;total+=laneCount;}binBase+=total;}}
   workgroupBarrier();var cursors:array<u32,16>;
   for(var bin=0u;bin<16u;bin+=1u){cursors[bin]=supportSortBins[lane*16u+bin];}
   for(var position=range.x;position<range.y;position+=1u){let value=rowSortRead(position,side);
     let bin=rowSortDigit(value,digit);rowSortWrite(cursors[bin],value,side);cursors[bin]+=1u;}
   storageBarrier();workgroupBarrier();side=1u-side;storageBarrier();workgroupBarrier();
 }
}
fn validateFaceBandRowDirectoryIndex(index:u32,count:u32){
 let key=rowDirectory[index*2u];let row=rowDirectory[index*2u+1u];
 if(row>=count||row>=arrayLength(&rows)){fail(BAD_DIRECTORY,index);return;}
 var ordered=true;if(index>0u){let prior=rowDirectory[(index-1u)*2u+1u];
   ordered=prior<arrayLength(&rows)
     &&rowIdentityLess(rowDirectory[(index-1u)*2u],rows[prior].size,key,rows[row].size);}
 if(rows[row].cell!=key||!ordered){
   fail(BAD_DIRECTORY,index);}
}
fn bandForGlobalRow(globalRow:u32)->u32{if(globalRow==INVALID||globalRow>=control.initialRows||globalRow>=arrayLength(&rows)||rows[globalRow].globalRow!=globalRow){return INVALID;}return globalRow;}
fn coord(cellKey:u32)->vec3u{return vec3u(cellKey%p.dims.x,(cellKey/p.dims.x)%p.dims.y,cellKey/(p.dims.x*p.dims.y));}
fn transitionFail(code:u32,index:u32){transitionControl.flags|=code;transitionControl.firstError=min(transitionControl.firstError,index);}fn adjacencyFail(index:u32,detail:u32){transitionControl.detailFlags|=detail;transitionFail(TRANSITION_ADJACENCY,index);}fn recordOwnerFailure(stage:u32,band:u32,row:Row,metric:Metric,selector:u32,rawOrigin:vec3i,requestedSize:u32,resolved:vec4i,owner:Owner){if(band>=transitionControl.failureBand){return;}transitionControl.failureBand=band;transitionControl.failureStage=stage;transitionControl.failureRowCell=row.cell;transitionControl.failureRowSize=row.size;transitionControl.failureDescriptor=metric.reserved;transitionControl.failureTopology=metric.topology;transitionControl.failureTransformFlags=metric.transformFlags;transitionControl.failureSelector=selector;transitionControl.failureRawX=bitcast<u32>(rawOrigin.x);transitionControl.failureRawY=bitcast<u32>(rawOrigin.y);transitionControl.failureRawZ=bitcast<u32>(rawOrigin.z);transitionControl.failureRequestedSize=requestedSize;transitionControl.failureResolvedCell=select(INVALID,cell(vec3u(resolved.xyz)),resolved.w>=0);transitionControl.failureBoundaryFlips=bitcast<u32>(resolved.w);transitionControl.failureOwnerCell=cell(owner.origin);transitionControl.failureOwnerSizeValid=(owner.size&0xffffu)|((owner.valid&0xffffu)<<16u);}fn inversePowerTransform(x:vec3f,code:u32)->vec3f{let bits=code&7u;let q=x*vec3f(select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));let k=(code/8u)%6u;if(k==0u){return q;}if(k==1u){return q.xzy;}if(k==2u){return q.yxz;}if(k==3u){return q.zxy;}if(k==4u){return q.yzx;}return q.zyx;}
fn reflectedOrigin(origin:vec3i,size:u32)->vec4i{if(size==0u){return vec4i(0,0,0,-1);}var resolved=origin;var flips=0;let high=origin+vec3i(i32(size));for(var axis=0u;axis<3u;axis+=1u){if(origin[axis]<0){if(origin[axis]<-i32(size)||(p.closedBoundaryMask&negativeBoundaryBit(axis))==0u){return vec4i(0,0,0,-1);}resolved[axis]=-origin[axis]-i32(size);flips|=1<<axis;}else if(high[axis]>i32(p.dims[axis])){if(high[axis]>i32(p.dims[axis]+size)||(p.closedBoundaryMask&positiveBoundaryBit(axis))==0u){return vec4i(0,0,0,-1);}resolved[axis]=2*i32(p.dims[axis])-origin[axis]-i32(size);flips|=1<<axis;}}if(any(resolved<vec3i(0))||any(resolved+vec3i(i32(size))>vec3i(p.dims))){return vec4i(0,0,0,-1);}return vec4i(resolved,flips);}
// Product boundary policy layered around the paper's in-domain interpolant:
// closed planes reflect the wall-normal component, while open planes use a
// one-layer zero-gradient vector extension. Geometry/weights remain at the
// original virtual point and no exterior row is ever published.
fn velocityExtendedOrigin(origin:vec3i,size:u32)->vec4i{if(size==0u){return vec4i(0,0,0,-1);}var resolved=origin;var flips=0;let high=origin+vec3i(i32(size));for(var axis=0u;axis<3u;axis+=1u){if(origin[axis]<0){if(origin[axis]<-i32(size)){return vec4i(0,0,0,-1);}resolved[axis]=-origin[axis]-i32(size);if((p.closedBoundaryMask&negativeBoundaryBit(axis))!=0u){flips|=1<<axis;}}else if(high[axis]>i32(p.dims[axis])){if(high[axis]>i32(p.dims[axis]+size)){return vec4i(0,0,0,-1);}resolved[axis]=2*i32(p.dims[axis])-origin[axis]-i32(size);if((p.closedBoundaryMask&positiveBoundaryBit(axis))!=0u){flips|=1<<axis;}}}if(any(resolved<vec3i(0))||any(resolved+vec3i(i32(size))>vec3i(p.dims))){return vec4i(0,0,0,-1);}return vec4i(resolved,flips);}
fn hasOpenBoundaryCrossing(origin:vec3i,size:u32)->bool{let high=origin+vec3i(i32(size));for(var axis=0u;axis<3u;axis+=1u){if(origin[axis]<0&&(p.closedBoundaryMask&negativeBoundaryBit(axis))==0u){return true;}if(high[axis]>i32(p.dims[axis])&&(p.closedBoundaryMask&positiveBoundaryBit(axis))==0u){return true;}}return false;}
fn recordRowDirectoryFailure(band:u32,selector:u32,cellKey:u32,size:u32){if(transitionControl.failureBand!=band||transitionControl.failureSelector!=selector){return;}let count=min(rowDirectoryCount(),min(p.rowDirectoryCapacity,arrayLength(&rowDirectory)/2u));var low=0u;var high=count;while(low<high){let mid=low+(high-low)/2u;let row=rowDirectory[mid*2u+1u];if(row>=arrayLength(&rows)){break;}if(rowIdentityLess(rowDirectory[mid*2u],rows[row].size,cellKey,size)){low=mid+1u;}else{high=mid;}}transitionControl.pad35=low;transitionControl.pad36=count;if(low<count){transitionControl.pad37=rowDirectory[low*2u];transitionControl.pad38=rowDirectory[low*2u+1u];}else{transitionControl.pad37=INVALID;transitionControl.pad38=INVALID;}}
fn transitionNeighbor(band:u32,selector:u32,metric:Metric)->u32{if(band>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){transitionFail(TRANSITION_SOURCE,band);return INVALID;}let row=rows[band];let vertex=tetraVertices[selector].v;if(row.size==0u||!finite(vertex.x)||!finite(vertex.y)||!finite(vertex.z)||!finite(vertex.w)||vertex.w<=0.){transitionFail(TRANSITION_SOURCE,band);return INVALID;}let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(vertex.xyz,metric.transformFlags&63u);let neighborSizeFloat=f32(row.size)*vertex.w;let neighborSize=u32(round(neighborSizeFloat));let rawOrigin=vec3i(round(point-.5*f32(neighborSize)));if(neighborSize==0u||abs(neighborSizeFloat-f32(neighborSize))>1e-4){adjacencyFail(band,DETAIL_GEOMETRY);return INVALID;}let rawHigh=rawOrigin+vec3i(i32(neighborSize));let outside=any(rawOrigin<vec3i(0))||any(rawHigh>vec3i(p.dims));if(outside){let extended=velocityExtendedOrigin(rawOrigin,neighborSize);if(extended.w<0){adjacencyFail(band,select(DETAIL_ABOVE_DOMAIN,DETAIL_BELOW_DOMAIN,any(rawOrigin<vec3i(0))));return INVALID;}rows[band].flags|=ROW_BOUNDARY_GHOST;return INVALID;}let reflected=reflectedOrigin(rawOrigin,neighborSize);if(reflected.w<0){adjacencyFail(band,select(DETAIL_ABOVE_DOMAIN,DETAIL_BELOW_DOMAIN,any(rawOrigin<vec3i(0))));return INVALID;}let neighborOrigin=vec3u(reflected.xyz);if(any(neighborOrigin%vec3u(neighborSize)!=vec3u(0u))){adjacencyFail(band,DETAIL_ALIGNMENT);return INVALID;}let expectedOwner=ownerAt(neighborOrigin);if(!ownerContains(expectedOwner,neighborOrigin)||expectedOwner.size!=neighborSize||any(expectedOwner.origin!=neighborOrigin)){recordOwnerFailure(OWNER_FAILURE_TRANSITION,band,row,metric,selector,rawOrigin,neighborSize,reflected,expectedOwner);adjacencyFail(band,DETAIL_OWNER);return INVALID;}let neighborCell=cell(neighborOrigin);let neighbor=rowOfIdentity(neighborCell,neighborSize);if(neighbor==INVALID){recordOwnerFailure(OWNER_FAILURE_TRANSITION,band,row,metric,selector,rawOrigin,neighborSize,reflected,expectedOwner);recordRowDirectoryFailure(band,selector,neighborCell,neighborSize);adjacencyFail(band,DETAIL_MISSING_ROW);return INVALID;}if(neighbor>=transitionControl.rowCount||neighbor>=arrayLength(&rows)){recordOwnerFailure(OWNER_FAILURE_TRANSITION,band,row,metric,selector,rawOrigin,neighborSize,reflected,expectedOwner);adjacencyFail(band,DETAIL_ROW_RANGE);return INVALID;}return neighbor;}
// A catalog contains many tetrahedra but only a small immutable selector set.
// Earlier records in this row are therefore an exact generation-local selector
// cache: reuse the already resolved lane instead of repeating owner-page and
// row-identity resolution for every tetrahedron occurrence.
fn cachedTransitionNeighbor(band:u32,first:u32,local:u32,selector:u32,metric:Metric)->u32{
 for(var prior=0u;prior<local;prior+=1u){
  let packed=tetrahedra[first+prior];
  let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);
  let record=transitionAdjacency[band*MAX_TETRA+prior];
  if(record.band==band){
   if(selectors.x==selector){return record.a;}
   if(selectors.y==selector){return record.b;}
   if(selectors.z==selector){return record.c;}
  }
 }
 return transitionNeighbor(band,selector,metric);
}
fn transitionDeltaAccepted(requested:u32)->bool{if(p.rowDeltaEnabled==0u||p.rowDeltaControl>arrayLength(&rowDelta)||arrayLength(&rowDelta)-p.rowDeltaControl<16u){return false;}let base=p.rowDeltaControl;let previous=rowDelta[base+1u];let carried=rowDelta[base+2u];let added=rowDelta[base+3u];let retired=rowDelta[base+4u];let affected=rowDelta[base+6u];let mapsFit=p.rowDeltaNewToOld<=arrayLength(&rowDelta)&&requested<=arrayLength(&rowDelta)-p.rowDeltaNewToOld;let affectedFits=p.rowDeltaAffected<=arrayLength(&rowDelta)&&affected<=arrayLength(&rowDelta)-p.rowDeltaAffected;return mapsFit&&affectedFits&&rowDelta[base]==requested&&rowDelta[base+7u]==p.powerGeneration&&rowDelta[base+8u]==ROW_DELTA_VALID&&carried<=previous&&requested==carried+added&&requested==previous+added-retired;}
fn committedTransitionValid()->bool{return committedValid()==COMMITTED_TRANSITION_VALID
 &&p.powerGeneration!=0u&&committedGeneration()==p.powerGeneration-1u
 &&committedCoreEnd()<=committedRowCount()
 &&committedRowCount()<=p.rowCapacity;}
fn transitionPriorRow(band:u32)->u32{
 if(!committedTransitionValid()||band>=transitionControl.rowCount||band>=arrayLength(&rows)){return INVALID;}
 if(band<transitionControl.coreEnd){if(p.rowDeltaNewToOld+band>=arrayLength(&rowDelta)){return INVALID;}
   let encoded=rowDelta[p.rowDeltaNewToOld+band]&0x7fffffffu;
   let old=select(INVALID,encoded-1u,encoded!=0u);
   return select(INVALID,old,old<committedCoreEnd());}
 let old=committedRowOfIdentity(rows[band].cell,rows[band].size);
 return select(INVALID,old,old>=committedCoreEnd()
   &&old<committedRowCount()&&committedRows[old].size==rows[band].size);
}
fn transitionIdentityAffected(band:u32)->bool{if(!committedTransitionValid()||band>=transitionControl.rowCount||band>=arrayLength(&rows)){return true;}if(band<transitionControl.coreEnd){if(p.rowDeltaNewToOld+band>=arrayLength(&rowDelta)){return true;}let encoded=rowDelta[p.rowDeltaNewToOld+band];if((encoded&ROW_DELTA_AFFECTED)!=0u||(encoded&0x7fffffffu)==0u){return true;}}let old=transitionPriorRow(band);if(old==INVALID||old>=arrayLength(&committedRows)){return true;}let current=rows[band];let prior=committedRows[old];let tierMask=ROW_CORE|ROW_SUPPORT1|ROW_SUPPORT3_NODE|ROW_SUPPORT3_ENDPOINT;return current.cell!=prior.cell||current.size!=prior.size||(current.flags&tierMask)!=(prior.flags&tierMask);}
fn transitionSupportNeighborhoodAffected(band:u32)->bool{let row=rows[band];let origin=coord(row.cell);for(var bit=0u;bit<18u;bit+=1u){let direction=DESCRIPTOR_DIRECTIONS[bit];var probe=vec3i(0);for(var axis=0u;axis<3u;axis+=1u){probe[axis]=select(select(i32(origin[axis]+row.size/2u),i32(origin[axis]+row.size),direction[axis]>0),i32(origin[axis])-1,direction[axis]<0);}if(any(probe<vec3i(0))||any(probe>=vec3i(p.dims))){continue;}let owner=ownerAt(vec3u(probe));if(!ownerContains(owner,vec3u(probe))){return true;}let neighbor=rowOfIdentity(cell(owner.origin),owner.size);if(neighbor==INVALID||neighbor>=transitionControl.rowCount||neighbor>=arrayLength(&rows)||rows[neighbor].cell!=cell(owner.origin)||transitionIdentityAffected(neighbor)){return true;}}return false;}
// A face slot is owned by one row but appears in both endpoint incidence
// lists.  Therefore every row in the axial 1-ring of an identity change must
// join the exact face delta even when its own identity is carried unchanged.
// This is the plan's dirty-row union 1-ring closure: it prevents an incidence
// carry from observing a face slot that the owning neighbor is rebuilding.
fn transitionFaceNeighborhoodAffected(band:u32)->bool{let row=rows[band];let origin=coord(row.cell);let half=max(1u,row.size/2u);let sampleCount=select(1u,4u,row.size>1u);for(var axis=0u;axis<3u;axis+=1u){let ta=(axis+1u)%3u;let tb=(axis+2u)%3u;for(var positive=0u;positive<2u;positive+=1u){for(var sample=0u;sample<sampleCount;sample+=1u){var probe=vec3i(origin);probe[axis]=select(i32(origin[axis])-1,i32(origin[axis]+row.size),positive!=0u);probe[ta]+=select(0i,i32(half),(sample&1u)!=0u);probe[tb]+=select(0i,i32(half),(sample&2u)!=0u);if(any(probe<vec3i(0))||any(probe>=vec3i(p.dims))){continue;}let q=vec3u(probe);let owner=ownerAt(q);if(!ownerContains(owner,q)){return true;}let neighbor=rowOfIdentity(cell(owner.origin),owner.size);let priorNeighbor=committedContainingRow(q);if(neighbor==INVALID||neighbor>=transitionControl.rowCount||neighbor>=arrayLength(&rows)||priorNeighbor==INVALID||rows[neighbor].cell!=cell(owner.origin)||transitionPriorRow(neighbor)!=priorNeighbor||transitionIdentityAffected(neighbor)){return true;}}}}return false;}
fn transitionRowNeedsResolution(band:u32)->bool{return band>=arrayLength(&rows)||(rows[band].flags&ROW_TRANSITION_AFFECTED)!=0u;}
fn transitionDeltaWorkCount()->u32{return transitionControl.rowCount;}
fn transitionDeltaRow(item:u32)->u32{return select(INVALID,item,item<transitionControl.rowCount);}
@compute @workgroup_size(1)fn prepareCatalogTransitionAdjacency(){transitionControl.flags=0u;transitionControl.firstError=INVALID;transitionControl.rowCount=min(control.rowCount,p.rowCapacity);transitionControl.coreEnd=min(control.initialRows,transitionControl.rowCount);let state=p.rowDirectoryCapacity*2u;if(state+3u>=arrayLength(&rowDirectory)){transitionFail(TRANSITION_CAPACITY,state);transitionControl.support1End=0u;transitionControl.support2End=0u;transitionControl.support3NodeEnd=0u;}else{transitionControl.support1End=rowDirectory[state+1u];transitionControl.support2End=rowDirectory[state+2u];transitionControl.support3NodeEnd=rowDirectory[state+3u];if(transitionControl.coreEnd>transitionControl.support1End||transitionControl.support1End>transitionControl.support2End||transitionControl.support2End>transitionControl.support3NodeEnd||transitionControl.support3NodeEnd>transitionControl.rowCount){transitionFail(TRANSITION_CAPACITY,transitionControl.rowCount);}}transitionControl.support4NodeEnd=transitionControl.support3NodeEnd;transitionControl.support5NodeEnd=transitionControl.support3NodeEnd;transitionControl.support6NodeEnd=transitionControl.support3NodeEnd;transitionControl.support7NodeEnd=transitionControl.support3NodeEnd;transitionControl.endpointEnd=transitionControl.rowCount;transitionControl.transitionRows=0u;transitionControl.adjacencyCount=0u;transitionControl.ready=0u;transitionControl.transferReady=0u;transitionControl.detailFlags=0u;transitionControl.boundaryGhostRequests=0u;transitionControl.hierarchyReady=0u;transitionControl.phiFailureCounts=0u;transitionControl.failureBand=INVALID;if(!transitionDeltaAccepted(transitionControl.coreEnd)){transitionFail(TRANSITION_CAPACITY,transitionControl.coreEnd);}indirect[51]=(transitionControl.rowCount+63u)/64u;indirect[52]=1u;indirect[53]=1u;}
@compute @workgroup_size(64)fn classifyCatalogTransitionDelta(@builtin(global_invocation_id)g:vec3u){let band=g.x;if(band>=transitionControl.rowCount||band>=arrayLength(&rows)||band>=arrayLength(&transitionDeltaScan)){return;}if(band==0u){control.bandPhiExtensions=0u;}if(band<arrayLength(&pointStatus)){pointStatus[band]=INVALID;}var affected=transitionIdentityAffected(band);if(!affected&&band>=transitionControl.coreEnd&&band<transitionControl.support3NodeEnd){affected=transitionSupportNeighborhoodAffected(band);}if(!affected){affected=transitionFaceNeighborhoodAffected(band);}rows[band].flags=(rows[band].flags&~ROW_TRANSITION_AFFECTED)|select(0u,ROW_TRANSITION_AFFECTED,affected);transitionDeltaScan[band]=select(0u,1u,affected);}
fn writeSupportDispatch(word:u32,count:u32){indirect[word]=(count+63u)/64u;indirect[word+1u]=1u;indirect[word+2u]=1u;}
fn appendBandPhiNeighbor(parent:u32,neighbor:u32,seen:ptr<function,array<u32,36>>,count:ptr<function,u32>){
 if(neighbor==INVALID||neighbor==parent){return;}
 for(var prior=0u;prior<*count;prior+=1u){if((*seen)[prior]==neighbor){return;}}
 if(*count>=MAX_CATALOG_GUARDS||neighbor>=transitionControl.rowCount
    ||neighbor>=arrayLength(&rows)){
   transitionFail(TRANSITION_CAPACITY,neighbor);return;}
 let edge=parent*MAX_CATALOG_GUARDS+*count;
 if(edge>=arrayLength(&transientPowerIncidences)){
   transitionFail(TRANSITION_CAPACITY,edge);return;}
 (*seen)[*count]=neighbor;*count+=1u;
 // Fixed, source-owned adjacency replaces the scheduler-ordered incoming
 // linked list. Exact support closure makes the local cube/Delaunay neighbor
 // graph symmetric, so marching consumes this row's bounded records directly.
 transientPowerIncidences[edge]=PowerIncidence(neighbor,0);
 if(neighbor>=transitionControl.support3NodeEnd&&neighbor<transitionControl.endpointEnd){
  if(neighbor>=arrayLength(&endpointIncomingCounts)){transitionFail(TRANSITION_CAPACITY,neighbor);return;}
  let incoming=atomicAdd(&endpointIncomingCounts[neighbor],1u);
  if(incoming>=MAX_CATALOG_GUARDS){transitionFail(TRANSITION_CAPACITY,neighbor);return;}
  let reverseEdge=neighbor*MAX_CATALOG_GUARDS+incoming;
  if(reverseEdge>=arrayLength(&transientPowerIncidences)){
   transitionFail(TRANSITION_CAPACITY,reverseEdge);return;}
  transientPowerIncidences[reverseEdge]=PowerIncidence(parent,0);
 }
}
@compute @workgroup_size(64)fn clearBandPhiEndpointIncomingEdges(
 @builtin(global_invocation_id)g:vec3u){
 let endpoint=g.x;if(endpoint<transitionControl.support3NodeEnd
   ||endpoint>=transitionControl.endpointEnd
   ||endpoint>=arrayLength(&endpointIncomingCounts)){return;}
 atomicStore(&endpointIncomingCounts[endpoint],0u);
 let edgeBase=endpoint*MAX_CATALOG_GUARDS;
 if(edgeBase>arrayLength(&transientPowerIncidences)
   ||MAX_CATALOG_GUARDS>arrayLength(&transientPowerIncidences)-edgeBase){
   transitionFail(TRANSITION_CAPACITY,endpoint);return;}
 for(var slot=0u;slot<MAX_CATALOG_GUARDS;slot+=1u){
  transientPowerIncidences[edgeBase+slot]=PowerIncidence(INVALID,0);}
}
// Section 5 fast marching uses each cell's local cube/Delaunay mesh. Support
// endpoints do not own another local mesh, so publish every validated local
// neighbor before Jacobi propagation. Each parent exclusively owns exactly
// MAX_CATALOG_GUARDS records; no target head or linked-list claim exists.
@compute @workgroup_size(64)fn buildBandPhiIncomingEdges(@builtin(global_invocation_id)g:vec3u){
 let parent=g.x;if(parent>=transitionControl.support3NodeEnd
   ||parent>=arrayLength(&rows)||parent>=arrayLength(&metrics)
   ||transitionControl.flags!=0u){return;}
 let edgeBase=parent*MAX_CATALOG_GUARDS;
 if(edgeBase>arrayLength(&transientPowerIncidences)
   ||MAX_CATALOG_GUARDS>arrayLength(&transientPowerIncidences)-edgeBase){
   transitionFail(TRANSITION_CAPACITY,parent);return;}
 for(var slot=0u;slot<MAX_CATALOG_GUARDS;slot+=1u){
   transientPowerIncidences[edgeBase+slot]=PowerIncidence(INVALID,0);}
 var seen:array<u32,36>;var count=0u;let row=rows[parent];let metric=metrics[parent];
 if((metric.transformFlags&VALID)==0u){transitionFail(TRANSITION_ADJACENCY,parent);return;}
 if(metric.reserved==0u){
   let origin=coord(row.cell);
   for(var z=-1;z<=1;z+=1){for(var y=-1;y<=1;y+=1){for(var x=-1;x<=1;x+=1){
     if(x==0&&y==0&&z==0){continue;}let direction=vec3i(x,y,z);var probe=vec3i(0);
     for(var axis=0u;axis<3u;axis+=1u){
       probe[axis]=select(select(i32(origin[axis]+row.size/2u),i32(origin[axis]+row.size),
         direction[axis]>0),i32(origin[axis])-1,direction[axis]<0);}
     if(any(probe<vec3i(0))||any(probe>=vec3i(p.dims))){continue;}
     let owner=ownerAt(vec3u(probe));
     if(!ownerContains(owner,vec3u(probe))){transitionFail(TRANSITION_ADJACENCY,parent);return;}
     let neighbor=rowOfIdentity(cell(owner.origin),owner.size);
     if(neighbor==INVALID){transitionFail(TRANSITION_ADJACENCY,parent);return;}
       appendBandPhiNeighbor(parent,neighbor,&seen,&count);
   }}}
   return;
 }
 let tetraCount=min(metric.reserved,MAX_TETRA);
 for(var local=0u;local<tetraCount;local+=1u){
   let at=parent*MAX_TETRA+local;
   if(at>=arrayLength(&transitionAdjacency)){transitionFail(TRANSITION_CAPACITY,at);return;}
   let adjacency=transitionAdjacency[at];
   if(adjacency.band!=parent){transitionFail(TRANSITION_ADJACENCY,parent);return;}
   appendBandPhiNeighbor(parent,adjacency.a,&seen,&count);
   appendBandPhiNeighbor(parent,adjacency.b,&seen,&count);
   appendBandPhiNeighbor(parent,adjacency.c,&seen,&count);
 }
}
// Terminal S3 rows are vertices of an S0-S2 cell-local cube/tetrahedron but
// own no local mesh. Sources publish these reverse edges directly while their
// bounded forward adjacency is resident in registers. This validator is O(E)
// publication accounting; it never scans the source-row cross product.
@compute @workgroup_size(64)fn validateBandPhiEndpointIncomingEdges(
 @builtin(global_invocation_id)g:vec3u){
 let endpoint=g.x;if(endpoint<transitionControl.support3NodeEnd
   ||endpoint>=transitionControl.endpointEnd||endpoint>=arrayLength(&rows)
   ||transitionControl.flags!=0u){return;}
 if(endpoint>=arrayLength(&endpointIncomingCounts)){transitionFail(TRANSITION_CAPACITY,endpoint);return;}
 let count=atomicLoad(&endpointIncomingCounts[endpoint]);
 if(count>MAX_CATALOG_GUARDS){transitionFail(TRANSITION_CAPACITY,endpoint);return;}
 let base=endpoint*MAX_CATALOG_GUARDS;
 // Atomic append order is intentionally irrelevant to construction. Restore
 // canonical source-row order in the endpoint's exclusive validator so the
 // published graph remains bit-deterministic across GPU schedules.
 for(var item=1u;item<count;item+=1u){
  let value=transientPowerIncidences[base+item];var cursor=item;
  loop{if(cursor==0u){break;}let prior=transientPowerIncidences[base+cursor-1u];
   if(prior.face<=value.face){break;}
   transientPowerIncidences[base+cursor]=prior;cursor-=1u;}
  transientPowerIncidences[base+cursor]=value;
 }
}
fn negativeBoundaryBit(axis:u32)->u32{return select(select(4u,2u,axis==1u),1u,axis==0u);}fn positiveBoundaryBit(axis:u32)->u32{return select(select(8u,16u,axis==1u),32u,axis==0u);}fn closedBoundaryException(low:vec3i,high:vec3i)->bool{var crossed=false;for(var axis=0u;axis<3u;axis+=1u){if(low[axis]<0){crossed=true;if((p.closedBoundaryMask&negativeBoundaryBit(axis))==0u){return false;}}if(high[axis]>i32(p.dims[axis])){crossed=true;if((p.closedBoundaryMask&positiveBoundaryBit(axis))==0u){return false;}}}return crossed;}
fn faceDeltaCount()->u32{return transitionControl.rowCount;}
fn faceDeltaRow(item:u32)->u32{return select(INVALID,item,item<transitionControl.rowCount);}
fn faceRowAffected(band:u32)->bool{return band<arrayLength(&transitionDeltaScan)
  &&transitionDeltaScan[band]!=0u;}
fn incidenceRowAffected(band:u32)->bool{return faceRowAffected(band);}
fn facePriorRow(band:u32)->u32{
 if(band>=arrayLength(&rows)){return INVALID;}let encoded=bitcast<u32>(rows[band].padf);
 return select(INVALID,encoded-1u,encoded!=0u);
}
fn remapCommittedBand(old:u32)->u32{
 if(old==INVALID||old>=arrayLength(&committedRows)){return INVALID;}
 let prior=committedRows[old];let current=rowOfIdentity(prior.cell,prior.size);
 return select(INVALID,current,current<transitionControl.rowCount&&current<arrayLength(&rows)
   &&rows[current].cell==prior.cell&&rows[current].size==prior.size);
}
@compute @workgroup_size(64)fn carryFaceBandTopology(@builtin(global_invocation_id)g:vec3u){
 let band=g.x;if(band>=transitionControl.rowCount||band>=arrayLength(&rows)
   ||transitionControl.flags!=0u){return;}
 if(faceRowAffected(band)){return;}
 let old=facePriorRow(band);if(old==INVALID||old>=arrayLength(&committedRows)){
   faceEmissionFail(BAD_ROW,band,2u);return;}
 let oldFaceBase=old*p.ownedFacesPerRow;let newFaceBase=band*p.ownedFacesPerRow;
 if(oldFaceBase>arrayLength(&committedFaces)||p.ownedFacesPerRow>arrayLength(&committedFaces)-oldFaceBase
   ||newFaceBase>arrayLength(&faces)||p.ownedFacesPerRow>arrayLength(&faces)-newFaceBase){
   faceEmissionFail(CAPACITY,band,3u);return;}
 for(var local=0u;local<p.ownedFacesPerRow;local+=1u){var face=committedFaces[oldFaceBase+local];
   if((face.flags&LIVE)!=0u){let negative=remapCommittedBand(face.negativeRow);
     let positive=remapCommittedBand(face.positiveRow);
     if(negative==INVALID||positive==INVALID){faceEmissionFail(BAD_FACE,face.globalFace,4u);return;}
     if(negative!=band||(band>=transitionControl.support3NodeEnd
       &&(positive<transitionControl.support1End||positive>=transitionControl.support3NodeEnd))){
       faceEmissionFail(BAD_FACE,face.globalFace,1u);return;}
     face.negativeRow=negative;face.positiveRow=positive;
     // Topology may be carried, but Section 5 phi, closest points, and
     // velocities are current-generation publications.  Retaining any of
     // these bits would let publishClosestPoint reinterpret an old velocity
     // as the new CPT, or let extension consume that CPT as an old seed.
     face.flags&=~(SEED|PHI_VALID|PHI_DIAGNOSTIC|CLOSEST_POINT_VALID
       |FACE_VELOCITY_VALID|COARSE_PHI|PRIMARY_EXTENSION);
     face.phi=0.;face.velocity=vec4f(0.);face.pad=0u;}
   faces[newFaceBase+local]=face;
 }
}
@compute @workgroup_size(64)fn carryFaceBandIncidence(@builtin(global_invocation_id)g:vec3u){
 let band=g.x;if(band>=transitionControl.rowCount||band>=arrayLength(&rows)
   ||incidenceRowAffected(band)||transitionControl.flags!=0u){return;}
 let old=facePriorRow(band);if(old==INVALID||old>=arrayLength(&committedRows)){
   faceEmissionFail(BAD_ROW,band,5u);return;}
 if(old>=arrayLength(&committedIncidence)||band>=arrayLength(&incidence)){
   faceEmissionFail(CAPACITY,band,6u);return;}
 let count=min(committedIncidence[old],p.axisStride);
 incidence[band]=count;
 for(var local=0u;local<count;local+=1u){
   let oldAt=p.rowCapacity+old*p.axisStride+local;
   let newAt=p.rowCapacity+band*p.axisStride+local;
   if(oldAt>=arrayLength(&committedIncidence)||newAt>=arrayLength(&incidence)){
     faceEmissionFail(CAPACITY,band,7u);return;}
   let oldFace=committedIncidence[oldAt];let oldOwner=oldFace/p.ownedFacesPerRow;
   let mapped=remapCommittedBand(oldOwner);if(mapped==INVALID){faceEmissionFail(BAD_FACE,oldFace,8u);return;}
   let newFace=mapped*p.ownedFacesPerRow+oldFace%p.ownedFacesPerRow;
   if(newFace>=arrayLength(&faces)||(faces[newFace].flags&LIVE)==0u
     ||(faces[newFace].negativeRow!=band&&faces[newFace].positiveRow!=band)){
     faceEmissionFail(BAD_FACE,newFace,9u);return;}
   incidence[newAt]=newFace;
 }
}
fn carryCatalogTransitionMetricAt(band:u32){if(band>=transitionControl.support3NodeEnd||transitionRowNeedsResolution(band)||transitionControl.flags!=0u){return;}let old=transitionPriorRow(band);if(old==INVALID||old>=arrayLength(&committedMetrics)||band>=arrayLength(&metrics)){transitionFail(TRANSITION_CAPACITY,band);return;}var metric=committedMetrics[old];if((metric.transformFlags&VALID)==0u){transitionFail(TRANSITION_ADJACENCY,band);return;}metric.transformFlags|=TRANSITION_CARRIED;metrics[band]=metric;}
@compute @workgroup_size(64)fn carryCatalogTransitionMetrics(@builtin(global_invocation_id)g:vec3u){carryCatalogTransitionMetricAt(g.x);}
fn remapCommittedTransitionNeighbor(oldNeighbor:u32,count:u32)->u32{if(oldNeighbor==INVALID){return INVALID;}if(oldNeighbor>=arrayLength(&committedRows)){return INVALID;}let prior=committedRows[oldNeighbor];let current=rowOfIdentity(prior.cell,prior.size);if(current>=count||current>=arrayLength(&rows)){return INVALID;}let row=rows[current];return select(INVALID,current,row.cell==prior.cell&&row.size==prior.size);}
fn carryCatalogTransitionAdjacencyAt(band:u32){if(band>=transitionControl.support3NodeEnd||transitionRowNeedsResolution(band)||transitionControl.flags!=0u){return;}let old=transitionPriorRow(band);if(old==INVALID||old>=arrayLength(&committedRows)||band>=arrayLength(&rows)||band>=arrayLength(&metrics)){transitionFail(TRANSITION_CAPACITY,band);return;}let priorRow=committedRows[old];let currentRow=rows[band];if(priorRow.cell!=currentRow.cell||priorRow.size!=currentRow.size){transitionFail(TRANSITION_ADJACENCY,band);return;}let count=metrics[band].reserved;if(count==0u){return;}if(count>MAX_TETRA||old>arrayLength(&committedAdjacency)/MAX_TETRA||old*MAX_TETRA>arrayLength(&committedAdjacency)-count||band>arrayLength(&transitionAdjacency)/MAX_TETRA||band*MAX_TETRA>arrayLength(&transitionAdjacency)-count){transitionFail(TRANSITION_CAPACITY,band);return;}for(var local=0u;local<count;local+=1u){let prior=committedAdjacency[old*MAX_TETRA+local];if(prior.band!=old){transitionFail(TRANSITION_ADJACENCY,band);return;}let a=remapCommittedTransitionNeighbor(prior.a,transitionControl.endpointEnd);let b=remapCommittedTransitionNeighbor(prior.b,transitionControl.endpointEnd);let c=remapCommittedTransitionNeighbor(prior.c,transitionControl.endpointEnd);if((prior.a!=INVALID&&a==INVALID)||(prior.b!=INVALID&&b==INVALID)||(prior.c!=INVALID&&c==INVALID)){transitionFail(TRANSITION_ADJACENCY,band);return;}transitionAdjacency[band*MAX_TETRA+local]=TransitionAdjacency(band,a,b,c);}}
@compute @workgroup_size(64)fn carryCatalogTransitionAdjacency(@builtin(global_invocation_id)g:vec3u){let band=g.x;carryCatalogTransitionMetricAt(band);carryCatalogTransitionAdjacencyAt(band);}
@compute @workgroup_size(64)fn describeCatalogBandRows(@builtin(global_invocation_id)g:vec3u){let item=g.x;if(item>=transitionDeltaWorkCount()||transitionControl.flags!=0u){return;}let band=transitionDeltaRow(item);if(band>=transitionControl.support3NodeEnd||!transitionRowNeedsResolution(band)){return;}if(band>=arrayLength(&rows)||band>=arrayLength(&metrics)){transitionFail(TRANSITION_ADJACENCY,band);return;}metrics[band]=describeBandRow(band);}
fn exactCatalogRowIdentity(band:u32,row:Row)->bool{
 let exactGlobal=select(row.globalRow>0u,row.globalRow==band,band<transitionControl.coreEnd);
 return exactGlobal&&rowOfIdentity(row.cell,row.size)==band;
}
fn exactCatalogNeighborIdentity(neighbor:u32,count:u32)->bool{if(neighbor==INVALID){return true;}return neighbor<count&&exactCatalogRowIdentity(neighbor,rows[neighbor]);}
@compute @workgroup_size(64)fn resolveCatalogTransitionAdjacency(@builtin(global_invocation_id)g:vec3u){let item=g.x;if(item>=transitionDeltaWorkCount()||transitionControl.flags!=0u){return;}let band=transitionDeltaRow(item);if(band>=transitionControl.support3NodeEnd){return;}if(band>=arrayLength(&rows)||band>=arrayLength(&metrics)||(metrics[band].transformFlags&TRANSITION_CARRIED)!=0u){return;}let row=rows[band];if(!exactCatalogRowIdentity(band,row)){transitionFail(TRANSITION_ADJACENCY,band);return;}var metric=metrics[band];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){transitionFail(TRANSITION_DESCRIPTOR,band);return;}let header=tetraHeaders[metric.topology];if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first||header.count>MAX_TETRA){transitionFail(TRANSITION_SOURCE,band);return;}if((header.flags&1u)!=0u){metric.reserved=0u;metrics[band]=metric;return;}let descriptor=metric.reserved;metric.reserved=header.count;metrics[band]=metric;metric.reserved=descriptor;if(header.count==0u||band>arrayLength(&transitionAdjacency)/MAX_TETRA||band*MAX_TETRA>arrayLength(&transitionAdjacency)-header.count){transitionFail(TRANSITION_CAPACITY,band);return;}for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){transitionFail(TRANSITION_SOURCE,band);return;}let a=cachedTransitionNeighbor(band,header.first,local,selectors.x,metric);var b=a;if(selectors.y!=selectors.x){b=cachedTransitionNeighbor(band,header.first,local,selectors.y,metric);}var c=a;if(selectors.z==selectors.y){c=b;}else if(selectors.z!=selectors.x){c=cachedTransitionNeighbor(band,header.first,local,selectors.z,metric);}if(transitionControl.flags!=0u){return;}transitionAdjacency[band*MAX_TETRA+local]=TransitionAdjacency(band,a,b,c);}}
@compute @workgroup_size(64)fn validateCatalogTransitionAdjacency(@builtin(global_invocation_id)g:vec3u){let band=g.x;if(band>=transitionControl.rowCount||band>=arrayLength(&rows)){return;}var row=rows[band];row.flags&=~(ROW_TRANSITION_AFFECTED|ROW_TRANSITION_VALIDATED);rows[band]=row;if(band>=transitionControl.support3NodeEnd){return;}if(band>=arrayLength(&metrics)){transitionFail(TRANSITION_ADJACENCY,band);return;}var metric=metrics[band];metric.transformFlags&=~TRANSITION_CARRIED;metrics[band]=metric;if(!exactCatalogRowIdentity(band,row)||(metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){transitionFail(TRANSITION_ADJACENCY,band);return;}let header=tetraHeaders[metric.topology];if((header.flags&1u)!=0u){if(metric.reserved!=0u){transitionFail(TRANSITION_ADJACENCY,band);return;}rows[band].flags|=ROW_TRANSITION_VALIDATED;return;}if(metric.reserved!=header.count||header.count==0u||header.count>MAX_TETRA||band>arrayLength(&transitionAdjacency)/MAX_TETRA||band*MAX_TETRA>arrayLength(&transitionAdjacency)-header.count){transitionFail(TRANSITION_ADJACENCY,band);return;}for(var local=0u;local<header.count;local+=1u){let record=transitionAdjacency[band*MAX_TETRA+local];if(record.band!=band||!exactCatalogNeighborIdentity(record.a,transitionControl.endpointEnd)||!exactCatalogNeighborIdentity(record.b,transitionControl.endpointEnd)||!exactCatalogNeighborIdentity(record.c,transitionControl.endpointEnd)){transitionFail(TRANSITION_ADJACENCY,band);return;}}rows[band].flags|=ROW_TRANSITION_VALIDATED;}
@compute @workgroup_size(256)fn publishCatalogTransitionAdjacency(@builtin(local_invocation_index)lane:u32){
 let count=transitionControl.rowCount;
 var validatedRows=0u;var adjacencyCount=0u;var boundaryGhostRequests=0u;var affectedRows=0u;
 let range=topologyPublishLaneRange(min(count,arrayLength(&rows)),lane);
 for(var band=range.x;band<range.y;band+=1u){
   let flags=rows[band].flags;
   if(band<transitionControl.support3NodeEnd&&(flags&ROW_TRANSITION_VALIDATED)!=0u){
     validatedRows+=1u;
     if(band<arrayLength(&metrics)){adjacencyCount+=metrics[band].reserved;}
   }
   boundaryGhostRequests+=select(0u,1u,(flags&ROW_BOUNDARY_GHOST)!=0u);
   affectedRows+=select(0u,1u,band<arrayLength(&transitionDeltaScan)
     &&transitionDeltaScan[band]!=0u);
 }
 topologyPublishSums[lane]=vec4u(validatedRows,adjacencyCount,boundaryGhostRequests,affectedRows);
 for(var width=128u;width>0u;width>>=1u){
  workgroupBarrier();if(lane<width){topologyPublishSums[lane]+=topologyPublishSums[lane+width];}
 }
 workgroupBarrier();if(lane!=0u){return;}
 let totals=topologyPublishSums[0];
 transitionControl.transitionRows=totals.x;
 transitionControl.adjacencyCount=totals.y;
 transitionControl.boundaryGhostRequests=totals.z;
 if(count==0u||count>p.rowCapacity||count>arrayLength(&rows)
   ||transitionControl.coreEnd>transitionControl.support1End
   ||transitionControl.support1End>transitionControl.support2End
   ||transitionControl.support2End>count
   ||transitionControl.support2End>transitionControl.support3NodeEnd
   ||transitionControl.support3NodeEnd>count
   ||transitionControl.endpointEnd!=count
   ||transitionControl.transitionRows!=transitionControl.support3NodeEnd
   ||transitionControl.adjacencyCount>transitionControl.support3NodeEnd*MAX_TETRA){
   transitionFail(TRANSITION_CAPACITY,count);}
 transitionControl.support4NodeEnd=transitionControl.support3NodeEnd;
 transitionControl.support5NodeEnd=transitionControl.support3NodeEnd;
 transitionControl.support6NodeEnd=transitionControl.support3NodeEnd;
 transitionControl.support7NodeEnd=transitionControl.support3NodeEnd;
 // The dense row mask remains live through face carry/emission. This keeps
 // the GPU-computed neighborhood closure without a global scan/prefix/scatter.
 transitionControl.pad35=totals.w;
 let clean=transitionControl.flags==0u;
 writeSupportDispatch(54u,select(0u,count,clean));
 writeSupportDispatch(57u,select(0u,count*p.ownedFacesPerRow,clean));
 transitionControl.hierarchyReady=select(0u,VALID,clean);
}
fn candidateTopologyReady()->bool{return control.flags==0u
 &&control.unresolvedCount==0u
 &&control.acceptedCount==control.faceCount
 &&control.generation==p.generation
 &&transitionControl.ready==VALID&&transitionControl.transferReady==VALID
 &&transitionControl.hierarchyReady==VALID
 &&transitionControl.flags==0u
 &&transitionControl.rowCount==control.rowCount;}
fn candidateTopologyAccepted()->bool{return control.valid==VALID
 &&candidateTopologyReady();}
@compute @workgroup_size(64)fn commitFaceBandRows(@builtin(global_invocation_id)g:vec3u){
 let index=g.x;let count=control.rowCount;let ready=candidateTopologyReady();
 if(index==0u){control.valid=select(0u,VALID,ready);}
 if(!ready||index>=count){return;}
 if(index>=arrayLength(&rows)||index>=arrayLength(&committedRows)
   ||index*2u+1u>=arrayLength(&rowDirectory)
   ||index*2u+1u>=arrayLength(&committedRowDirectory)){return;}
 committedRows[index]=rows[index];committedRowDirectory[index*2u]=rowDirectory[index*2u];
 committedRowDirectory[index*2u+1u]=rowDirectory[index*2u+1u];
 let identitySlot=rowIdentitySlot(rows[index].cell,rows[index].size);
 if(identitySlot==INVALID||identitySlot>=arrayLength(&committedRowDirectory)){
   fail(BAD_DIRECTORY,index);return;}
 committedRowDirectory[identitySlot]=index+1u;
}
@compute @workgroup_size(64)fn commitFaceBandFaces(@builtin(global_invocation_id)g:vec3u){
 let band=g.x;let count=control.rowCount;if(!candidateTopologyAccepted()||band>=count){return;}
 let base=band*p.ownedFacesPerRow;if(base<=arrayLength(&faces)
   &&p.ownedFacesPerRow<=arrayLength(&faces)-base&&base<=arrayLength(&committedFaces)
   &&p.ownedFacesPerRow<=arrayLength(&committedFaces)-base){
   for(var local=0u;local<p.ownedFacesPerRow;local+=1u){committedFaces[base+local]=faces[base+local];}}
 if(band>=arrayLength(&incidence)||band>=arrayLength(&committedIncidence)){return;}
 let incidenceCount=min(incidence[band],p.axisStride);
 committedIncidence[band]=incidenceCount;
 for(var local=0u;local<incidenceCount;local+=1u){
   let at=p.rowCapacity+band*p.axisStride+local;
   if(at<arrayLength(&incidence)&&at<arrayLength(&committedIncidence)){
     committedIncidence[at]=incidence[at];}}
}
@compute @workgroup_size(64)fn commitFaceBandTransition(@builtin(global_invocation_id)g:vec3u){
 let band=g.x;if(!candidateTopologyAccepted()||band>=transitionControl.support3NodeEnd
   ||band>=arrayLength(&metrics)||band>=arrayLength(&committedMetrics)){return;}
 committedMetrics[band]=metrics[band];let count=metrics[band].reserved;
 if(count>MAX_TETRA){return;}let base=band*MAX_TETRA;
 if(base<=arrayLength(&transitionAdjacency)&&count<=arrayLength(&transitionAdjacency)-base
   &&base<=arrayLength(&committedAdjacency)&&count<=arrayLength(&committedAdjacency)-base){
   for(var local=0u;local<count;local+=1u){committedAdjacency[base+local]=transitionAdjacency[base+local];}}
}
@compute @workgroup_size(1)fn commitFaceBandControls(){
 if(!candidateTopologyAccepted()){return;}
 let base=committedStateBase();
 if(base+3u>=arrayLength(&committedRowDirectory)){fail(CAPACITY,base);return;}
 committedRowDirectory[base]=transitionControl.coreEnd;
 committedRowDirectory[base+1u]=transitionControl.rowCount;
 committedRowDirectory[base+2u]=p.powerGeneration;
 committedRowDirectory[base+3u]=COMMITTED_TRANSITION_VALID;
 committedControl.flags=control.flags;
 committedControl.firstError=control.firstError;
 committedControl.rowCount=control.rowCount;
 committedControl.faceCount=control.faceCount;
 committedControl.incidenceCount=control.incidenceCount;
 committedControl.generation=control.generation;
 committedControl.reserved0=control.reserved0;
 committedControl.seedCount=control.seedCount;
 committedControl.acceptedCount=control.acceptedCount;
 committedControl.unresolvedCount=control.unresolvedCount;
 committedControl.initialRows=control.initialRows;
 committedControl.sampleFailures=control.sampleFailures;
 committedControl.coarsePhiSamples=control.coarsePhiSamples;
 committedControl.coarsePhiFailures=control.coarsePhiFailures;
 committedControl.bandPhiExtensions=control.bandPhiExtensions;
 committedControl.closestPointFaces=control.closestPointFaces;
 committedControl.closestPointFailures=control.closestPointFailures;
 committedControl.liquidInterpolationFailures=control.liquidInterpolationFailures;
 committedControl.firstFaceEmissionFailure=control.firstFaceEmissionFailure;
 committedControl.firstPhiFailureSlot=control.firstPhiFailureSlot;
 committedControl.firstPhiFailure=control.firstPhiFailure;
 committedControl.firstClosestPointFailure=control.firstClosestPointFailure;
 committedControl.firstVectorFailure=control.firstVectorFailure;
 committedControl.cptNoOwnerFailures=control.cptNoOwnerFailures;
 committedControl.cptSupportOwnerFailures=control.cptSupportOwnerFailures;
 committedControl.cptNoContainingSimplexFailures=control.cptNoContainingSimplexFailures;
 committedControl.cptMissingLiquidVertexFailures=control.cptMissingLiquidVertexFailures;
 committedControl.firstCptNoOwnerFailure=control.firstCptNoOwnerFailure;
 committedControl.firstCptSupportOwnerFailure=control.firstCptSupportOwnerFailure;
 committedControl.firstCptNoSimplexFailure=control.firstCptNoSimplexFailure;
 committedControl.firstCptMissingVertexFailure=control.firstCptMissingVertexFailure;
 committedTransitionControl.flags=transitionControl.flags;
 committedTransitionControl.firstError=transitionControl.firstError;
 committedTransitionControl.rowCount=transitionControl.rowCount;
 committedTransitionControl.transitionRows=transitionControl.transitionRows;
 committedTransitionControl.adjacencyCount=transitionControl.adjacencyCount;
 committedTransitionControl.transferReady=transitionControl.transferReady;
 committedTransitionControl.detailFlags=transitionControl.detailFlags;
 committedTransitionControl.coreEnd=transitionControl.coreEnd;
 committedTransitionControl.support1End=transitionControl.support1End;
 committedTransitionControl.support2End=transitionControl.support2End;
 committedTransitionControl.support3NodeEnd=transitionControl.support3NodeEnd;
 committedTransitionControl.endpointEnd=transitionControl.endpointEnd;
 committedTransitionControl.boundaryGhostRequests=
   transitionControl.boundaryGhostRequests;
 committedTransitionControl.hierarchyReady=transitionControl.hierarchyReady;
 committedTransitionControl.phiFailureCounts=transitionControl.phiFailureCounts;
 committedTransitionControl.failureBand=transitionControl.failureBand;
 committedTransitionControl.failureStage=transitionControl.failureStage;
 committedTransitionControl.failureRowCell=transitionControl.failureRowCell;
 committedTransitionControl.failureRowSize=transitionControl.failureRowSize;
 committedTransitionControl.failureDescriptor=transitionControl.failureDescriptor;
 committedTransitionControl.failureTopology=transitionControl.failureTopology;
 committedTransitionControl.failureTransformFlags=transitionControl.failureTransformFlags;
 committedTransitionControl.failureSelector=transitionControl.failureSelector;
 committedTransitionControl.failureRawX=transitionControl.failureRawX;
 committedTransitionControl.failureRawY=transitionControl.failureRawY;
 committedTransitionControl.failureRawZ=transitionControl.failureRawZ;
 committedTransitionControl.failureRequestedSize=transitionControl.failureRequestedSize;
 committedTransitionControl.failureResolvedCell=transitionControl.failureResolvedCell;
 committedTransitionControl.failureBoundaryFlips=transitionControl.failureBoundaryFlips;
 committedTransitionControl.failureOwnerCell=transitionControl.failureOwnerCell;
 committedTransitionControl.failureOwnerSizeValid=transitionControl.failureOwnerSizeValid;
 committedTransitionControl.support4NodeEnd=transitionControl.support4NodeEnd;
 committedTransitionControl.support5NodeEnd=transitionControl.support5NodeEnd;
 committedTransitionControl.support6NodeEnd=transitionControl.support6NodeEnd;
 committedTransitionControl.support7NodeEnd=transitionControl.support7NodeEnd;
 committedTransitionControl.pad35=transitionControl.pad35;
 committedTransitionControl.pad36=transitionControl.pad36;
 committedTransitionControl.pad37=transitionControl.pad37;
 committedTransitionControl.pad38=transitionControl.pad38;
 committedTransitionControl.ready=transitionControl.ready;
 committedControl.valid=VALID;
}
@compute @workgroup_size(1)fn gateTransitionTransfer(){transitionControl.ready=0u;transitionControl.transferReady=0u;let transitionFlags=transitionControl.flags;if(transitionFlags==0u&&transitionControl.hierarchyReady==VALID&&control.flags==0u){transitionControl.ready=VALID;transitionControl.transferReady=VALID;return;}control.flags|=BAD_ROW;let first=transitionControl.firstError;if(control.firstError==INVALID&&first!=INVALID){control.firstError=first;}}
@compute @workgroup_size(64)fn emitBandFaces(@builtin(global_invocation_id)g:vec3u){
 let item=g.x;if(item>=faceDeltaCount()||control.flags!=0u){return;}
 let band=faceDeltaRow(item);if(!faceRowAffected(band)){return;}if(band>=transitionControl.endpointEnd||band>=arrayLength(&rows)){faceEmissionFail(BAD_ROW,band,16u);return;}
 let faceBase=band*p.ownedFacesPerRow;if(faceBase>arrayLength(&faces)
   ||p.ownedFacesPerRow>arrayLength(&faces)-faceBase){faceEmissionFail(CAPACITY,band,17u);return;}
 for(var local=0u;local<p.ownedFacesPerRow;local+=1u){faces[faceBase+local].flags=0u;}
 let row=rows[band];let origin=coord(row.cell);
 for(var axis=0u;axis<3u;axis+=1u){let boundary=origin[axis]+row.size;
   if(boundary>=p.dims[axis]){continue;}let ta=(axis+1u)%3u;let tb=(axis+2u)%3u;
   let sampleCount=select(1u,4u,row.size>1u);let half=max(1u,row.size/2u);
   var emitted:array<u32,4>;for(var local=0u;local<sampleCount;local+=1u){
     var probe=origin;probe[axis]=boundary;probe[ta]+=select(0u,half,(local&1u)!=0u);
     probe[tb]+=select(0u,half,(local&2u)!=0u);let neighborOwner=ownerAt(probe);
     if(neighborOwner.valid==0u){faceEmissionFail(BAD_ROW,cell(probe),18u);return;}
     let neighbor=rowOfIdentity(cell(neighborOwner.origin),neighborOwner.size);
     if(neighbor==INVALID||neighbor>=transitionControl.endpointEnd||neighbor>=arrayLength(&rows)){
       if(band>=transitionControl.support3NodeEnd){continue;}
       adjacencyFail(band,DETAIL_MISSING_ROW);faceEmissionFail(BAD_ROW,row.cell,19u);return;}
     // S2 node rows own their complete positive half-star. Terminal S3
     // endpoints own only the canonical incoming half needed to close an S2
     // node; S3-to-S3 faces are outside the minimal interpolation topology.
     if(band>=transitionControl.support3NodeEnd){
       let s3ToS2=neighbor>=transitionControl.support1End
         &&neighbor<transitionControl.support3NodeEnd;
       if(!s3ToS2){if(neighbor<transitionControl.support1End){
           faceEmissionFail(BAD_ROW,band,21u);return;}continue;}
     }
     let other=rows[neighbor];var duplicate=false;for(var prior=0u;prior<local;prior+=1u){
       duplicate=duplicate||emitted[prior]==neighbor+1u;}if(duplicate){continue;}
     emitted[local]=neighbor+1u;let otherOrigin=coord(other.cell);let low=max(origin,otherOrigin);
     let high=min(origin+vec3u(row.size),otherOrigin+vec3u(other.size));
     if(high[ta]<=low[ta]||high[tb]<=low[tb]){faceEmissionFail(BAD_FACE,row.cell,20u);return;}
     let area=f32(high[ta]-low[ta])*f32(high[tb]-low[tb]);
     var centroid=.5*(vec3f(low)+vec3f(high));centroid[axis]=f32(boundary);
     var keyCell=low;keyCell[axis]=boundary-1u;let stable=3u*cell(keyCell)+axis;
     let slot=faceBase+axis*4u+local;
     let velocityTarget=band<transitionControl.support3NodeEnd&&neighbor<transitionControl.support3NodeEnd;
     faces[slot]=Face(band,neighbor,axis|(min(row.size,other.size)<<2u),stable,
       vec4f(0),vec4f(centroid,1.),0.,area,LIVE|select(0u,VELOCITY_TARGET,velocityTarget),0u);
   }
 }
}
fn appendFixedIncidence(items:ptr<function,array<u32,24>>,count:ptr<function,u32>,face:u32){
 for(var i=0u;i<*count;i+=1u){if((*items)[i]==face){return;}}
 if(*count>=p.axisStride){faceEmissionFail(CAPACITY,face,24u);return;}
 (*items)[*count]=face;*count+=1u;
}
@compute @workgroup_size(64)fn rebuildFaceBandIncidence(@builtin(global_invocation_id)g:vec3u){
 let item=g.x;if(item>=faceDeltaCount()||control.flags!=0u){return;}
 // Incidence belongs to every S0/S1/S2 node and terminal S3 endpoint touched
 // by the delta. S3 owns only canonical positive-axis faces incident to S2,
 // but both endpoints publish the shared incidence.
 let band=faceDeltaRow(item);if(!incidenceRowAffected(band)){return;}if(band>=transitionControl.rowCount||band>=arrayLength(&rows)
   ||band>=arrayLength(&incidence)){faceEmissionFail(CAPACITY,band,25u);return;}
 var items:array<u32,24>;var count=0u;for(var i=0u;i<24u;i+=1u){items[i]=INVALID;}
 let ownBase=band*p.ownedFacesPerRow;for(var local=0u;local<p.ownedFacesPerRow;local+=1u){
   let faceIndex=ownBase+local;if(faceIndex<arrayLength(&faces)&&(faces[faceIndex].flags&LIVE)!=0u){
     appendFixedIncidence(&items,&count,faceIndex);}}
 let row=rows[band];let origin=coord(row.cell);let half=max(1u,row.size/2u);
 for(var axis=0u;axis<3u;axis+=1u){if(origin[axis]==0u){continue;}
   let ta=(axis+1u)%3u;let tb=(axis+2u)%3u;let sampleCount=select(1u,4u,row.size>1u);
   var visited:array<u32,4>;
   for(var sample=0u;sample<sampleCount;sample+=1u){var probe=origin;probe[axis]-=1u;
     probe[ta]+=select(0u,half,(sample&1u)!=0u);probe[tb]+=select(0u,half,(sample&2u)!=0u);
     let owner=ownerAt(probe);if(owner.valid==0u){faceEmissionFail(BAD_ROW,row.cell,26u);return;}
     let neighbor=rowOfIdentity(cell(owner.origin),owner.size);if(neighbor==INVALID){
       // S3 is terminal endpoint closure and never requests an S4 row merely
       // to inspect a face slot. Every S0/S1/S2 node is fully closed.
       if(band>=transitionControl.support3NodeEnd){continue;}
       faceEmissionFail(BAD_ROW,band,27u);return;}
     if(neighbor>=transitionControl.endpointEnd){
       if(band>=transitionControl.support3NodeEnd){continue;}
       faceEmissionFail(BAD_ROW,band,29u);return;}
     var duplicate=false;for(var prior=0u;prior<sample;prior+=1u){duplicate=duplicate||visited[prior]==neighbor+1u;}
     if(duplicate){continue;}visited[sample]=neighbor+1u;let base=neighbor*p.ownedFacesPerRow+axis*4u;
     for(var local=0u;local<4u;local+=1u){let faceIndex=base+local;
       if(faceIndex<arrayLength(&faces)){let face=faces[faceIndex];
         if((face.flags&LIVE)!=0u&&(face.negativeRow==band||face.positiveRow==band)){
           appendFixedIncidence(&items,&count,faceIndex);}}}
   }
 }
 if(control.flags!=0u){return;}incidence[band]=count;
 for(var local=0u;local<count;local+=1u){let at=p.rowCapacity+band*p.axisStride+local;
   if(at>=arrayLength(&incidence)){faceEmissionFail(CAPACITY,band,28u);return;}
   incidence[at]=items[local];}
}
@compute @workgroup_size(256)fn publishFaceBandCounts(@builtin(local_invocation_index)lane:u32){
 if(lane==0u){control.bandPhiExtensions=0u;if(arrayLength(&liveFaceWorklist)==0u||arrayLength(&indirect)<60u){faceEmissionFail(CAPACITY,0u,35u);}else{liveFaceWorklist[0]=0u;indirect[57]=0u;indirect[58]=1u;indirect[59]=1u;}}
 storageBarrier();workgroupBarrier();
 let count=select(0u,transitionControl.rowCount,control.flags==0u&&transitionControl.flags==0u);
 let range=topologyPublishLaneRange(count,lane);
 var topologyFaceCount=0u;var velocityTargetCount=0u;var incidenceCount=0u;var firstCapacity=INVALID;
 for(var band=range.x;band<range.y;band+=1u){
   if(band>=arrayLength(&incidence)){firstCapacity=min(firstCapacity,band*2u);continue;}
   if(band>=arrayLength(&rows)){firstCapacity=min(firstCapacity,band*2u+1u);continue;}
   rows[band].padf=0.;incidenceCount+=min(incidence[band],p.axisStride);
   let base=band*p.ownedFacesPerRow;for(var local=0u;local<p.ownedFacesPerRow;local+=1u){
     let faceIndex=base+local;if(faceIndex<arrayLength(&faces)&&(faces[faceIndex].flags&LIVE)!=0u){
       topologyFaceCount+=1u;velocityTargetCount+=select(0u,1u,faceVelocityTarget(faces[faceIndex]));}}
 }
 topologyPublishOffsets[lane]=topologyFaceCount;
 topologyPublishSums[lane]=vec4u(topologyFaceCount,velocityTargetCount,incidenceCount,firstCapacity);
 for(var width=128u;width>0u;width>>=1u){workgroupBarrier();if(lane<width){let right=topologyPublishSums[lane+width];topologyPublishSums[lane]=vec4u(topologyPublishSums[lane].xyz+right.xyz,min(topologyPublishSums[lane].w,right.w));}}
 workgroupBarrier();if(lane==0u){
   let failure=topologyPublishSums[0].w;if(failure!=INVALID){let band=failure/2u;faceEmissionFail(CAPACITY,band,select(32u,33u,(failure&1u)!=0u));}
   var offset=0u;for(var sourceLane=0u;sourceLane<256u;sourceLane+=1u){let laneCount=topologyPublishOffsets[sourceLane];topologyPublishOffsets[sourceLane]=offset;offset+=laneCount;}
   if(offset+1u>arrayLength(&liveFaceWorklist)){faceEmissionFail(CAPACITY,offset,35u);}
 }
 storageBarrier();workgroupBarrier();
 var output=topologyPublishOffsets[lane];
 let publishEnd=select(range.x,range.y,control.flags==0u);
 for(var band=range.x;band<publishEnd;band+=1u){let base=band*p.ownedFacesPerRow;
   for(var local=0u;local<p.ownedFacesPerRow;local+=1u){let faceIndex=base+local;
     if(faceIndex<arrayLength(&faces)&&(faces[faceIndex].flags&LIVE)!=0u){liveFaceWorklist[output+1u]=faceIndex;output+=1u;}}
 }
 storageBarrier();workgroupBarrier();if(lane==0u&&control.flags==0u){let totals=topologyPublishSums[0];
   if(totals.z!=2u*totals.x){faceEmissionFail(BAD_FACE,totals.z,34u);return;}
   liveFaceWorklist[0]=totals.x;indirect[57]=(totals.x+63u)/64u;
   control.faceCount=totals.y;control.incidenceCount=totals.z;
 }
}
fn liveFaceIndex(item:u32)->u32{if(arrayLength(&liveFaceWorklist)==0u||item>=liveFaceWorklist[0]||item+1u>=arrayLength(&liveFaceWorklist)){return INVALID;}return liveFaceWorklist[item+1u];}
fn fineBrickKey(q:vec3u)->u32{return q.x+fp.brickDims.x*(q.y+fp.brickDims.y*q.z);}
fn finePage(key:u32)->u32{
 if(fp.worklistHeaderWords!=5u||arrayLength(&worklist)<5u||worklist[1]!=fp.generation
   ||worklist[3]!=1u||worklist[4]!=1u){return INVALID;}
 let count=min(worklist[0],min(fp.worklistCapacity,fp.pageCapacity));
 let logicalCount=fp.brickDims.x*fp.brickDims.y*fp.brickDims.z;
 let directoryBase=5u+fp.worklistCapacity;
 if(5u+count>arrayLength(&worklist)||key>=logicalCount
   ||directoryBase+key>=arrayLength(&worklist)){return INVALID;}
 let id=worklist[directoryBase+key];let base=id*10u;
 return select(INVALID,id,id<fp.pageCapacity&&base+2u<arrayLength(&metadata)
   &&metadata[base]==id&&metadata[base+1u]==key&&metadata[base+2u]==fp.generation);
}
// At a world-plane face centroid the centered trilinear product stencil has
// one virtual sample center. Preserve its original product weight and mirror
// only that one layer to the interior scalar (homogeneous-Neumann/even phi).
// Open/closed velocity behavior is deliberately irrelevant to scalar phi.
fn loadFineScalarExtended(virtualIndex:vec3i)->vec2f{if(any(fp.sampleDims==vec3u(0u))){return vec2f(0.);}var q=virtualIndex;for(var axis=0u;axis<3u;axis+=1u){let limit=i32(fp.sampleDims[axis]);if(q[axis]<0){if(q[axis]!=-1){return vec2f(0.);}q[axis]=0;}else if(q[axis]>=limit){if(q[axis]!=limit){return vec2f(0.);}q[axis]=limit-1;}}let uq=vec3u(q);let brick=uq/fp.brickResolution;if(any(brick>=fp.brickDims)){return vec2f(0.);}let local=uq-brick*fp.brickResolution;let id=finePage(fineBrickKey(brick));if(id==INVALID){return vec2f(0.);}let index=id*fp.samplesPerBrick+local.x+fp.brickResolution*(local.y+fp.brickResolution*local.z);if(index>=arrayLength(&sampleFlags)||index>=arrayLength(&finePhi)||(sampleFlags[index]&1u)==0u){return vec2f(0.);}let value=finePhi[index];if(!finite(value)){return vec2f(0.);}return vec2f(value,1.);}
fn finePhiAtFaceCentroid(pointGrid:vec3f)->PhiCPT{if(fp.generation!=p.generation||fp.fineFactor==0u||fp.brickResolution==0u||fp.samplesPerBrick!=fp.brickResolution*fp.brickResolution*fp.brickResolution||any(fp.sampleDims!=p.dims*fp.fineFactor)){return PhiCPT(0.,vec3f(0.),0u);}let coarseWidth=fp.fineWidth*f32(fp.fineFactor);if(!finite(coarseWidth)||coarseWidth<=0.){return PhiCPT(0.,vec3f(0.),0u);}let world=fp.domainOrigin+pointGrid*coarseWidth;let raw=(world-fp.domainOrigin)/fp.fineWidth-vec3f(.5);let wall=vec3f(fp.sampleDims)-vec3f(.5);if(any(raw<vec3f(-.5))||any(raw>wall)){return PhiCPT(0.,vec3f(0.),0u);}let base=vec3i(floor(raw));let fraction=fract(raw);var result=0.;var gradient=vec3f(0.);for(var z=0;z<2;z+=1){for(var y=0;y<2;y+=1){for(var x=0;x<2;x+=1){let wx=select(1.-fraction.x,fraction.x,x==1);let wy=select(1.-fraction.y,fraction.y,y==1);let wz=select(1.-fraction.z,fraction.z,z==1);let sample=loadFineScalarExtended(base+vec3i(x,y,z));if(sample.y==0.){return PhiCPT(0.,vec3f(0.),0u);}result+=wx*wy*wz*sample.x;gradient+=sample.x*vec3f(select(-1.,1.,x==1)*wy*wz,select(-1.,1.,y==1)*wx*wz,select(-1.,1.,z==1)*wx*wy);}}}if(!finite(result)||!finite(gradient.x)||!finite(gradient.y)||!finite(gradient.z)){return PhiCPT(0.,vec3f(0.),0u);}return PhiCPT(result,gradient,1u);}
fn halfOpenDomainPoint(point:vec3f)->vec3f{return clamp(point,vec3f(0.),max(vec3f(0.),vec3f(p.dims)-vec3f(1e-4)));}
fn publishClosestPoint(face:Face,sampled:PhiCPT,width:f32)->Face{var result=face;result.phi=sampled.phi;if(sampled.valid==0u||!finite(width)||width<=0.){return result;}result.flags|=PHI_VALID;let magnitude=length(sampled.gradient);if(!finite(magnitude)||magnitude<=1e-8){return result;}let normal=sampled.gradient/magnitude;let rawClosest=face.centroid.xyz-(sampled.phi/width)*normal;if(!finite(rawClosest.x)||!finite(rawClosest.y)||!finite(rawClosest.z)){return result;}
 let closest=halfOpenDomainPoint(rawClosest);
 result.velocity=vec4f(closest,1.);result.flags|=CLOSEST_POINT_VALID;return result;}
// Section 5 stores phi at octree cell centers and Section 6.2 supplies the
// local cube/Delaunay interpolant. The fine SPGrid is a narrow band, so a
// missing fine stencil is resolved from that current compact octree field;
// it is never replaced by a fabricated sign or distance.
fn exactCoarseCellScalar(origin:vec3u,size:u32)->vec2f{if(!validCoarse()||size==0u||any(origin+vec3u(size)>p.dims)){return vec2f(0.);}let entry=coarseEntryRecord(coarseSlot(cell(origin),size));return vec2f(entry.x,entry.w);}
fn coarseCellSeedScalar(origin:vec3u,size:u32)->vec2f{let entry=coarseCellSeedRecord(origin,size);return vec2f(entry.x,entry.w);}
fn coarseCellScalar(origin:vec3u,size:u32)->vec2f{if(size==0u||any(origin+vec3u(size)>p.dims)){return vec2f(0.);}let band=rowOfIdentity(cell(origin),size);if(band!=INVALID&&band<arrayLength(&rows)){let row=rows[band];if((row.flags&ROW_PHI)!=0u&&finite(row.representativePhi)){return vec2f(row.representativePhi,1.);}}return exactCoarseCellScalar(origin,size);}
fn compactScalar(origin:vec3i,size:u32)->vec2f{let extended=velocityExtendedOrigin(origin,size);if(extended.w<0){return vec2f(0.);}return coarseCellScalar(vec3u(extended.xyz),size);}
fn selectorScalar(anchor:u32,selector:u32,transform:u32)->vec2f{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return vec2f(0.);}let row=rows[anchor];let v=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(v.xyz,transform);let sizeFloat=f32(row.size)*v.w;let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return vec2f(0.);}return compactScalar(vec3i(round(point-.5*f32(size))),size);}
fn publishedBandScalar(rowIndex:u32)->vec2f{if(rowIndex==INVALID||rowIndex>=arrayLength(&rows)){return vec2f(0.);}let row=rows[rowIndex];if((row.flags&ROW_PHI)==0u||!finite(row.representativePhi)){return vec2f(0.);}return vec2f(row.representativePhi,1.);}
fn exactRegularFaceEdgeRows(face:Face)->vec2u{if(face.negativeRow>=arrayLength(&rows)||face.positiveRow>=arrayLength(&rows)){return vec2u(INVALID);}let negative=rows[face.negativeRow];let positive=rows[face.positiveRow];if(negative.size==0u||negative.size!=positive.size){return vec2u(INVALID);}let negativeCenter=vec3f(coord(negative.cell))+.5*f32(negative.size);let positiveCenter=vec3f(coord(positive.cell))+.5*f32(positive.size);let midpoint=.5*(negativeCenter+positiveCenter);let tolerance=1e-5*max(1.,f32(negative.size));if(any(abs(face.centroid.xyz-midpoint)>vec3f(tolerance))){return vec2u(INVALID);}return vec2u(face.negativeRow,face.positiveRow);}
fn exactRegularFaceEdgePhi(face:Face)->vec2f{let edge=exactRegularFaceEdgeRows(face);if(edge.x==INVALID){return vec2f(0.);}let negative=publishedBandScalar(edge.x);let positive=publishedBandScalar(edge.y);if(negative.y==0.||positive.y==0.){return vec2f(0.);}let phi=.5*(negative.x+positive.x);return select(vec2f(0.),vec2f(phi,1.),finite(phi));}
fn compactPublishedBandScalar(origin:vec3i,size:u32)->vec2f{let high=origin+vec3i(i32(size));let boundary=any(origin<vec3i(0))||any(high>vec3i(p.dims));let extended=velocityExtendedOrigin(origin,size);if(extended.w<0){return vec2f(0.);}let q=vec3u(extended.xyz);var candidateSize=size;loop{let candidateOrigin=(q/vec3u(candidateSize))*vec3u(candidateSize);let band=rowOfIdentity(cell(candidateOrigin),candidateSize);if(band!=INVALID&&band<arrayLength(&rows)){return publishedBandScalar(band);}if(!boundary||candidateSize>=p.maximumLeaf){break;}candidateSize*=2u;}return vec2f(0.);}
fn localTetraBandScalar(anchor:u32,local:u32,lane:u32,selector:u32,transform:u32)->vec2f{let at=anchor*MAX_TETRA+local;if(at<arrayLength(&transitionAdjacency)){let adjacency=transitionAdjacency[at];if(adjacency.band==anchor){let neighbor=select(select(adjacency.c,adjacency.b,lane==1u),adjacency.a,lane==0u);if(neighbor!=INVALID){return publishedBandScalar(neighbor);}}}if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return vec2f(0.);}let row=rows[anchor];let v=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(v.xyz,transform);let sizeFloat=f32(row.size)*v.w;let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return vec2f(0.);}let origin=vec3i(round(point-.5*f32(size)));if(all(origin>=vec3i(0))&&all(origin+vec3i(i32(size))<=vec3i(p.dims))){return vec2f(0.);}return compactPublishedBandScalar(origin,size);}
fn coarsePhiAtPoint(anchor:u32,pointGrid:vec3f)->vec2f{if(anchor==INVALID||anchor>=arrayLength(&rows)||anchor>=arrayLength(&metrics)){return vec2f(0.);}let row=rows[anchor];let origin=coord(row.cell);let anchorPhi=publishedBandScalar(anchor);if(anchorPhi.y==0.){return vec2f(0.);}let metric=metrics[anchor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){return vec2f(0.);}let header=tetraHeaders[metric.topology];if((header.flags&1u)!=0u){let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);if(any(t<vec3f(-2e-6))||any(t>vec3f(1.000002))){return vec2f(0.);}var result=0.;for(var corner=0u;corner<8u;corner+=1u){let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);if(weight==0.){continue;}let cornerOrigin=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u))*i32(row.size);let value=compactPublishedBandScalar(cornerOrigin,row.size);if(value.y==0.){return vec2f(0.);}result+=weight*value.x;}return select(vec2f(0.),vec2f(result,1.),finite(result));}if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return vec2f(0.);}let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return vec2f(0.);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}let a=localTetraBandScalar(anchor,local,0u,selectors.x,metric.transformFlags&63u);let b=localTetraBandScalar(anchor,local,1u,selectors.y,metric.transformFlags&63u);let c=localTetraBandScalar(anchor,local,2u,selectors.z,metric.transformFlags&63u);if(a.y==0.||b.y==0.||c.y==0.){return vec2f(0.);}let result=weights.x*anchorPhi.x+weights.y*a.x+weights.z*b.x+weights.w*c.x;return select(vec2f(0.),vec2f(result,1.),finite(result));}return vec2f(0.);}
fn noPhiDiagnostic(anchor:u32,path:u32)->PhiDiagnostic{return PhiDiagnostic(vec3i(0),0u,anchor,path,INVALID,INVALID,0u);}
fn phiDiagnostic(origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32,cause:u32,detail:u32)->PhiDiagnostic{return PhiDiagnostic(origin,size,anchor,path,selector,cause,detail);}
fn diagnoseCellScalar(origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32)->PhiDiagnostic{if(size==0u||any(origin<vec3i(0))||any(origin+vec3i(i32(size))>vec3i(p.dims))){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_INVALID_METRIC,1u);}let unsignedOrigin=vec3u(origin);let encodedCell=cell(unsignedOrigin);let band=rowOfIdentity(encodedCell,size);if(band!=INVALID&&band<arrayLength(&rows)){let row=rows[band];if((row.flags&ROW_PHI)!=0u&&finite(row.representativePhi)){return noPhiDiagnostic(anchor,path);}}if(exactCoarseCellScalar(unsignedOrigin,size).y!=0.){return noPhiDiagnostic(anchor,path);}if(band==INVALID||band>=arrayLength(&rows)){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_MISSING_ROW,0u);}return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_EXACT_COARSE_MISS,1u);}
fn diagnoseCompactScalar(origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32)->PhiDiagnostic{let extended=velocityExtendedOrigin(origin,size);if(extended.w<0){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_INVALID_METRIC,2u);}return diagnoseCellScalar(extended.xyz,size,anchor,path,selector);}
fn diagnoseSelectorScalar(anchor:u32,selector:u32,transform:u32,detail:u32)->PhiDiagnostic{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return phiDiagnostic(vec3i(0),0u,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail);}let row=rows[anchor];let vertex=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(vertex.xyz,transform);let sizeFloat=f32(row.size)*vertex.w;if(!finite(sizeFloat)||sizeFloat<=0.||!finite(point.x)||!finite(point.y)||!finite(point.z)){return phiDiagnostic(vec3i(0),0u,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail|0x10000u);}let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return phiDiagnostic(vec3i(round(point)),size,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail|0x20000u);}return diagnoseCompactScalar(vec3i(round(point-.5*f32(size))),size,anchor,PHI_PATH_DELAUNAY,selector);}
fn diagnosePublishedBandRow(rowIndex:u32,origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32)->PhiDiagnostic{if(rowIndex==INVALID){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_MISSING_ROW,0x6000u);}if(rowIndex>=arrayLength(&rows)){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_MISSING_ROW,0x7000u|(rowIndex&0xfffu));}let row=rows[rowIndex];if(row.cell!=cell(vec3u(origin))||row.size!=size){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_MISSING_ROW,0x8000u|(rowIndex&0xfffu));}if((row.flags&ROW_PHI)==0u||!finite(row.representativePhi)){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_EXACT_COARSE_MISS,0x9000u|(rowIndex&0xfffu));}return noPhiDiagnostic(anchor,path);}
fn diagnoseCompactPublishedBandScalar(origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32)->PhiDiagnostic{let high=origin+vec3i(i32(size));let boundary=any(origin<vec3i(0))||any(high>vec3i(p.dims));let extended=velocityExtendedOrigin(origin,size);if(extended.w<0){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_INVALID_METRIC,2u);}let q=vec3u(extended.xyz);var candidateSize=size;var record=phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_INVALID_METRIC,3u);loop{let candidateOrigin=(q/vec3u(candidateSize))*vec3u(candidateSize);record=diagnosePublishedBandRow(rowOfIdentity(cell(candidateOrigin),candidateSize),vec3i(candidateOrigin),candidateSize,anchor,path,selector);if(record.cause==INVALID||!boundary||candidateSize>=p.maximumLeaf){break;}candidateSize*=2u;}return record;}
fn diagnoseLocalTetraBandScalar(anchor:u32,local:u32,lane:u32,selector:u32,transform:u32,detail:u32)->PhiDiagnostic{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return phiDiagnostic(vec3i(0),0u,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail);}let row=rows[anchor];let vertex=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(vertex.xyz,transform);let sizeFloat=f32(row.size)*vertex.w;if(!finite(sizeFloat)||sizeFloat<=0.||!finite(point.x)||!finite(point.y)||!finite(point.z)){return phiDiagnostic(vec3i(0),0u,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail|0x1000u);}let size=u32(round(sizeFloat));let origin=vec3i(round(point-.5*f32(size)));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return phiDiagnostic(origin,size,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail|0x2000u);}let at=anchor*MAX_TETRA+local;if(at>=arrayLength(&transitionAdjacency)){return phiDiagnostic(origin,size,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_METRIC,detail|0x3000u);}let adjacency=transitionAdjacency[at];if(adjacency.band!=anchor){return phiDiagnostic(origin,size,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_METRIC,detail|0x4000u);}let neighbor=select(select(adjacency.c,adjacency.b,lane==1u),adjacency.a,lane==0u);if(neighbor!=INVALID){return diagnosePublishedBandRow(neighbor,origin,size,anchor,PHI_PATH_DELAUNAY,selector);}if(all(origin>=vec3i(0))&&all(origin+vec3i(i32(size))<=vec3i(p.dims))){return phiDiagnostic(origin,size,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_MISSING_ROW,detail|0x5000u);}return diagnoseCompactPublishedBandScalar(origin,size,anchor,PHI_PATH_DELAUNAY,selector);}
fn diagnoseCoarsePhiAtPoint(anchor:u32,pointGrid:vec3f)->PhiDiagnostic{if(anchor==INVALID||anchor>=arrayLength(&rows)||anchor>=arrayLength(&metrics)){return phiDiagnostic(vec3i(0),0u,anchor,PHI_PATH_ANCHOR,INVALID,PHI_CAUSE_INVALID_METRIC,3u);}let row=rows[anchor];let origin=coord(row.cell);let anchorDiagnostic=diagnosePublishedBandRow(anchor,vec3i(origin),row.size,anchor,PHI_PATH_ANCHOR,INVALID);if(anchorDiagnostic.cause!=INVALID){return anchorDiagnostic;}let metric=metrics[anchor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){return phiDiagnostic(vec3i(origin),row.size,anchor,PHI_PATH_ANCHOR,INVALID,PHI_CAUSE_INVALID_METRIC,4u);}let header=tetraHeaders[metric.topology];if((header.flags&1u)!=0u){let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);if(any(t<vec3f(-2e-6))||any(t>vec3f(1.000002))){return phiDiagnostic(low,row.size,anchor,PHI_PATH_CUBE,INVALID,PHI_CAUSE_INVALID_METRIC,5u);}for(var corner=0u;corner<8u;corner+=1u){let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);if(weight==0.){continue;}let cornerOrigin=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u))*i32(row.size);let phiRecord=diagnoseCompactPublishedBandScalar(cornerOrigin,row.size,anchor,PHI_PATH_CUBE,corner);if(phiRecord.cause!=INVALID){return phiRecord;}}return phiDiagnostic(low,row.size,anchor,PHI_PATH_CUBE,INVALID,PHI_CAUSE_INVALID_SELECTOR,6u);}if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return phiDiagnostic(vec3i(origin),row.size,anchor,PHI_PATH_DELAUNAY,INVALID,PHI_CAUSE_INVALID_SELECTOR,7u);}let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return phiDiagnostic(vec3i(origin),row.size,anchor,PHI_PATH_DELAUNAY,INVALID,PHI_CAUSE_INVALID_SELECTOR,0x80000u|local);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}var phiRecord=diagnoseLocalTetraBandScalar(anchor,local,0u,selectors.x,metric.transformFlags&63u,local|(0u<<8u));if(phiRecord.cause!=INVALID){return phiRecord;}phiRecord=diagnoseLocalTetraBandScalar(anchor,local,1u,selectors.y,metric.transformFlags&63u,local|(1u<<8u));if(phiRecord.cause!=INVALID){return phiRecord;}phiRecord=diagnoseLocalTetraBandScalar(anchor,local,2u,selectors.z,metric.transformFlags&63u,local|(2u<<8u));if(phiRecord.cause!=INVALID){return phiRecord;}return phiDiagnostic(vec3i(origin),row.size,anchor,PHI_PATH_DELAUNAY,INVALID,PHI_CAUSE_INVALID_SELECTOR,0x90000u|local);}return phiDiagnostic(vec3i(origin),row.size,anchor,PHI_PATH_DELAUNAY,INVALID,PHI_CAUSE_INVALID_SELECTOR,0xa0000u);}
@compute @workgroup_size(64)fn sampleBandFacePhi(@builtin(global_invocation_id)g:vec3u){let index=liveFaceIndex(g.x);if(index==INVALID||index>=p.faceCapacity||index>=arrayLength(&faces)||control.flags!=0u){return;}let face=faces[index];if((face.flags&LIVE)==0u){return;}let sampled=finePhiAtFaceCentroid(face.centroid.xyz);if(sampled.valid==0u){return;}let published=publishClosestPoint(face,sampled,fp.fineWidth*f32(fp.fineFactor));if((published.flags&PHI_VALID)==0u){return;}faces[index]=published;}
fn rowMarchSign(row:Row)->f32{let air=(row.flags&ROW_COARSE_AIR)!=0u;let liquid=(row.flags&ROW_COARSE_LIQUID)!=0u;return select(select(0.,-1.,liquid&&!air),1.,air&&!liquid);}
@compute @workgroup_size(64)fn initializeBandRowPhi(@builtin(global_invocation_id)g:vec3u){let rowIndex=g.x;if(rowIndex>=p.rowCapacity||rowIndex>=arrayLength(&rowVelocities)){return;}let liveRow=rowIndex<control.rowCount&&rowIndex<arrayLength(&rows);if(!liveRow){rowVelocities[rowIndex]=vec4f(0.);return;}let row=rows[rowIndex];let center=vec3f(coord(row.cell))+.5*f32(row.size);let fineSample=finePhiAtFaceCentroid(center);var sampled=vec2f(fineSample.phi,f32(fineSample.valid));if(sampled.y==0.){sampled=coarseCellSeedScalar(coord(row.cell),row.size);}rowVelocities[rowIndex]=select(vec4f(0.,0.,0.,1.),vec4f(sampled.x,1.,1.,1.),sampled.y!=0.);}
@compute @workgroup_size(64)fn seedBandRowPhiFromFaces(@builtin(global_invocation_id)g:vec3u){let rowIndex=g.x;if(rowIndex>=control.rowCount||rowIndex>=arrayLength(&rows)||rowIndex>=arrayLength(&rowVelocities)){return;}var state=rowVelocities[rowIndex];if(state.y!=0.){return;}let row=rows[rowIndex];let sign=rowMarchSign(row);let center=vec3f(coord(row.cell))+.5*f32(row.size);var best=3.402823e38;var signedBest=0.;let count=min(incidence[rowIndex],p.axisStride);for(var local=0u;local<count;local+=1u){let faceIndex=incidence[p.rowCapacity+rowIndex*p.axisStride+local];if(faceIndex>=arrayLength(&faces)){continue;}let face=faces[faceIndex];if((face.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||!finite(face.phi)){continue;}let distance=length(center-face.centroid.xyz)*fp.fineWidth*f32(fp.fineFactor);if(sign==0.){let candidate=abs(face.phi)+distance;if(candidate<best){best=candidate;signedBest=select(-candidate,candidate,face.phi>=0.);}}else{let candidate=max(0.,sign*face.phi+distance);if(candidate<best){best=candidate;signedBest=sign*candidate;}}}if(best<3.402823e38){rowVelocities[rowIndex]=vec4f(signedBest,1.,0.,1.);}}
@compute @workgroup_size(64)fn collectBandPhiActiveRows(@builtin(global_invocation_id)g:vec3u){
 let rowIndex=g.x;
 if(rowIndex==0u&&arrayLength(&bandPhiFrontier)>0u&&arrayLength(&indirect)>=66u){
  atomicStore(&bandPhiFrontier[0],control.rowCount);
  indirect[63]=(control.rowCount+63u)/64u;indirect[64]=1u;indirect[65]=1u;
 }
 if(rowIndex>=control.rowCount||rowIndex>=arrayLength(&rowVelocities)
   ||rowIndex>=arrayLength(&provisionalVelocities)
   ||rowIndex+1u>=arrayLength(&bandPhiFrontier)){return;}
 let state=rowVelocities[rowIndex];provisionalVelocities[rowIndex]=state;
 atomicStore(&bandPhiFrontier[rowIndex+1u],
   select(rowIndex,INVALID,state.w==0.||state.z>0.));
}
fn bandCenter(row:u32)->vec3f{return vec3f(coord(rows[row].cell))+.5*f32(rows[row].size);}
fn sameMarchSide(value:vec4f,sign:f32)->bool{return value.y!=0.&&finite(value.x)&&sign*value.x>=-1e-5;}
fn solveColumns3(a:vec3f,b:vec3f,c:vec3f,rhs:vec3f)->vec4f{let determinant=dot(a,cross(b,c));if(!finite(determinant)||abs(determinant)<=1e-10){return vec4f(0.);}return vec4f(dot(rhs,cross(b,c)),dot(a,cross(rhs,c)),dot(a,cross(b,rhs)),determinant);}
fn solveTranspose3(a:vec3f,b:vec3f,c:vec3f,rhs:vec3f)->vec4f{let determinant=dot(a,cross(b,c));if(!finite(determinant)||abs(determinant)<=1e-10){return vec4f(0.);}let value=(rhs.x*cross(b,c)+rhs.y*cross(c,a)+rhs.z*cross(a,b))/determinant;return vec4f(value,1.);}
// Section 5 requires the solid angle incident at the current cell centre to
// be at most pi/2. For Omega=2 atan2(D,N), this is the stable N>=D test.
fn nonobtuseIncidentSolidAngle(a:vec3f,b:vec3f,c:vec3f)->bool{let determinant=abs(dot(a,cross(b,c)));let la=length(a);let lb=length(b);let lc=length(c);let denominator=la*lb*lc+dot(a,b)*lc+dot(a,c)*lb+dot(b,c)*la;let scale=max(1.,max(determinant,abs(denominator)));return finite(determinant)&&finite(denominator)&&determinant>1e-10&&denominator+2e-5*scale>=determinant;}
fn localTetraEikonal(rowIndex:u32,sign:f32)->f32{var best=3.402823e38;if(rowIndex>=arrayLength(&metrics)){return best;}let count=min(metrics[rowIndex].reserved,MAX_TETRA);let center=bandCenter(rowIndex);let unit=fp.fineWidth*f32(fp.fineFactor);for(var local=0u;local<count;local+=1u){let at=rowIndex*MAX_TETRA+local;if(at>=arrayLength(&transitionAdjacency)){break;}let adjacency=transitionAdjacency[at];if(adjacency.band!=rowIndex||adjacency.a>=arrayLength(&rows)||adjacency.b>=arrayLength(&rows)||adjacency.c>=arrayLength(&rows)||adjacency.a>=arrayLength(&rowVelocities)||adjacency.b>=arrayLength(&rowVelocities)||adjacency.c>=arrayLength(&rowVelocities)){continue;}let va=rowVelocities[adjacency.a];let vb=rowVelocities[adjacency.b];let vc=rowVelocities[adjacency.c];if(!sameMarchSide(va,sign)||!sameMarchSide(vb,sign)||!sameMarchSide(vc,sign)){continue;}let a=(bandCenter(adjacency.a)-center)*unit;let b=(bandCenter(adjacency.b)-center)*unit;let c=(bandCenter(adjacency.c)-center)*unit;if(!nonobtuseIncidentSolidAngle(a,b,c)){phiFail(BAD_PHI,ACUTE_TETRA_FAILURE|(rowIndex*MAX_TETRA+local));return best;}let known=vec3f(abs(va.x),abs(vb.x),abs(vc.x));let av=solveTranspose3(a,b,c,known);let bv=solveTranspose3(a,b,c,vec3f(1.));if(av.w==0.||bv.w==0.){continue;}let aa=dot(bv.xyz,bv.xyz);let bb=dot(av.xyz,bv.xyz);let cc=dot(av.xyz,av.xyz)-1.;let discriminant=bb*bb-aa*cc;if(!finite(aa)||aa<=1e-12||!finite(discriminant)||discriminant<0.){continue;}let candidate=(bb+sqrt(max(0.,discriminant)))/aa;let lower=max(known.x,max(known.y,known.z));if(!finite(candidate)||candidate+1e-5<lower){continue;}let ray=solveColumns3(a,b,c,-(av.xyz-candidate*bv.xyz));if(ray.w==0.){continue;}let weights=ray.xyz/ray.w;let sum=weights.x+weights.y+weights.z;if(!all(vec3<bool>(finite(weights.x),finite(weights.y),finite(weights.z)))||sum<=1e-8||any(weights/sum<vec3f(-2e-5))){continue;}best=min(best,candidate);}return best;}
fn uniformEikonal(rowIndex:u32,sign:f32)->f32{var axisValues=vec3f(3.402823e38);let count=min(incidence[rowIndex],p.axisStride);for(var local=0u;local<count;local+=1u){let faceIndex=incidence[p.rowCapacity+rowIndex*p.axisStride+local];if(faceIndex>=arrayLength(&faces)){continue;}let face=faces[faceIndex];let axis=face.axisSpan&3u;let neighbor=select(face.negativeRow,face.positiveRow,face.negativeRow==rowIndex);if(axis>=3u||neighbor==INVALID||neighbor>=arrayLength(&rowVelocities)){continue;}let source=rowVelocities[neighbor];if(sameMarchSide(source,sign)){axisValues[axis]=min(axisValues[axis],abs(source.x));}}var a=min(axisValues.x,min(axisValues.y,axisValues.z));var c=max(axisValues.x,max(axisValues.y,axisValues.z));var b=axisValues.x+axisValues.y+axisValues.z-a-c;if(a>=3.402823e38){return a;}let h=fp.fineWidth*f32(fp.fineFactor*rows[rowIndex].size);var result=a+h;if(b<3.402823e38&&result>b){let disc=2.*h*h-(a-b)*(a-b);if(disc>=0.){result=.5*(a+b+sqrt(disc));}}if(c<3.402823e38&&result>c){let disc=3.*h*h-(a-b)*(a-b)-(a-c)*(a-c)-(b-c)*(b-c);if(disc>=0.){result=(a+b+c+sqrt(disc))/3.;}}return result;}
fn lowerSimplexCandidate(rowIndex:u32,sign:f32)->f32{var best=3.402823e38;let center=bandCenter(rowIndex);let unit=fp.fineWidth*f32(fp.fineFactor);let edgeBase=rowIndex*MAX_CATALOG_GUARDS;for(var edgeLocal=0u;edgeLocal<MAX_CATALOG_GUARDS;edgeLocal+=1u){let edge=edgeBase+edgeLocal;if(edge>=arrayLength(&transientPowerIncidences)){break;}let parent=transientPowerIncidences[edge].face;if(parent==INVALID){break;}if(parent<arrayLength(&rowVelocities)&&parent<arrayLength(&rows)){let source=rowVelocities[parent];if(sameMarchSide(source,sign)){best=min(best,abs(source.x)+length(center-bandCenter(parent))*unit);}}}let count=min(incidence[rowIndex],p.axisStride);for(var local=0u;local<count;local+=1u){let faceIndex=incidence[p.rowCapacity+rowIndex*p.axisStride+local];if(faceIndex>=arrayLength(&faces)){continue;}let face=faces[faceIndex];let neighbor=select(face.negativeRow,face.positiveRow,face.negativeRow==rowIndex);if(neighbor==INVALID||neighbor>=arrayLength(&rowVelocities)||neighbor>=arrayLength(&rows)){continue;}let source=rowVelocities[neighbor];if(sameMarchSide(source,sign)){best=min(best,abs(source.x)+length(center-bandCenter(neighbor))*unit);}}return best;}
// Aanjaneya et al. (2017), Section 5 stores signed distance outside the fine
// interface band on the coarse octree. Mixed deep-closure rows own no LIVE
// regular faces, so they inherit a measured sign through the bounded support
// edges captured while that coarse interpolation topology is constructed.
fn signedDistanceAxisDerivative(center:vec2f,plus:vec2f,minus:vec2f,h:f32)->vec2f{if(center.y==0.||!finite(center.x)||!finite(h)||h<=0.){return vec2f(0.);}if(plus.y!=0.&&minus.y!=0.){let central=(plus.x-minus.x)/(2.*h);if(abs(central)>1e-8){return vec2f(central,1.);}let forward=(plus.x-center.x)/h;let backward=(center.x-minus.x)/h;let oneSided=select(backward,forward,abs(forward)>abs(backward));return select(vec2f(0.),vec2f(oneSided,1.),finite(oneSided));}if(plus.y!=0.){let forward=(plus.x-center.x)/h;return select(vec2f(0.),vec2f(forward,1.),finite(forward));}if(minus.y!=0.){let backward=(center.x-minus.x)/h;return select(vec2f(0.),vec2f(backward,1.),finite(backward));}return vec2f(0.);}
fn coarsePhiCptAtPoint(anchor:u32,pointGrid:vec3f)->PhiCPT{let center=coarsePhiAtPoint(anchor,pointGrid);if(center.y==0.||anchor>=arrayLength(&rows)){return PhiCPT(0.,vec3f(0.),0u);}let rowScale=f32(rows[anchor].size);for(var attempt=0u;attempt<4u;attempt+=1u){let h=min(.5*rowScale,max(.03125,.0625*rowScale*f32(1u<<attempt)));var gradient=vec3f(0.);var complete=true;for(var axis=0u;axis<3u;axis+=1u){var delta=vec3f(0.);delta[axis]=h;let plus=coarsePhiAtPoint(anchor,pointGrid+delta);let minus=coarsePhiAtPoint(anchor,pointGrid-delta);let derivative=signedDistanceAxisDerivative(center,plus,minus,h);if(derivative.y==0.){complete=false;break;}gradient[axis]=derivative.x;}let magnitude=length(gradient);if(complete&&finite(magnitude)&&magnitude>1e-8){return PhiCPT(center.x,gradient,1u);}}return PhiCPT(0.,vec3f(0.),0u);}
fn nearestSignedNeighbor(rowIndex:u32,current:vec4f)->vec2f{var best=select(3.402823e38,abs(current.x),current.y!=0.&&finite(current.x));var signedBest=current.x;let center=bandCenter(rowIndex);let unit=fp.fineWidth*f32(fp.fineFactor);let edgeBase=rowIndex*MAX_CATALOG_GUARDS;for(var edgeLocal=0u;edgeLocal<MAX_CATALOG_GUARDS;edgeLocal+=1u){let edge=edgeBase+edgeLocal;if(edge>=arrayLength(&transientPowerIncidences)){break;}let parent=transientPowerIncidences[edge].face;if(parent==INVALID){break;}if(parent<arrayLength(&rowVelocities)&&parent<arrayLength(&rows)){let source=rowVelocities[parent];if(source.y!=0.&&finite(source.x)){let candidate=abs(source.x)+length(center-bandCenter(parent))*unit;if(candidate<best){best=candidate;signedBest=select(-candidate,candidate,source.x>=0.);}}}}let count=min(incidence[rowIndex],p.axisStride);for(var local=0u;local<count;local+=1u){let faceIndex=incidence[p.rowCapacity+rowIndex*p.axisStride+local];if(faceIndex>=arrayLength(&faces)){continue;}let face=faces[faceIndex];let neighbor=select(face.negativeRow,face.positiveRow,face.negativeRow==rowIndex);if(neighbor==INVALID||neighbor>=arrayLength(&rowVelocities)||neighbor>=arrayLength(&rows)){continue;}let source=rowVelocities[neighbor];if(source.y==0.||!finite(source.x)){continue;}let candidate=abs(source.x)+length(center-bandCenter(neighbor))*unit;if(candidate<best){best=candidate;signedBest=select(-candidate,candidate,source.x>=0.);}}return vec2f(signedBest,select(0.,1.,best<3.402823e38));}
fn extendBandRowPhi(rowIndex:u32)->vec4f{
  if(rowIndex>=p.rowCapacity||rowIndex>=arrayLength(&rows)
    ||rowIndex>=arrayLength(&rowVelocities)||rowIndex>=arrayLength(&provisionalVelocities)){
    return vec4f(0.);
  }
  let current=rowVelocities[rowIndex];
  if(current.w==0.){return vec4f(0.);}
  if(current.z>0.){return current;}
  let row=rows[rowIndex];
  let sign=rowMarchSign(row);
  if(sign==0.){
    let candidate=nearestSignedNeighbor(rowIndex,current);
    return select(vec4f(0.,0.,0.,1.),vec4f(candidate.x,1.,0.,1.),candidate.y!=0.);
  }
  var best=select(3.402823e38,abs(current.x),sameMarchSide(current,sign));
  best=min(best,lowerSimplexCandidate(rowIndex,sign));
  let closureOnly=(row.flags&(ROW_SUPPORT3_NODE|ROW_SUPPORT4_NODE|ROW_SUPPORT5_NODE
    |ROW_SUPPORT6_NODE|ROW_SUPPORT3_ENDPOINT))!=0u;
  if(!closureOnly&&rowIndex<arrayLength(&metrics)&&(metrics[rowIndex].transformFlags&VALID)!=0u){
    if(metrics[rowIndex].reserved==0u){best=min(best,uniformEikonal(rowIndex,sign));}
    else{best=min(best,localTetraEikonal(rowIndex,sign));}
  }
  return select(vec4f(0.,0.,0.,1.),vec4f(sign*best,1.,0.,1.),best<3.402823e38);
}
@compute @workgroup_size(64)fn extendBandRowPhiParallel(
  @builtin(global_invocation_id)g:vec3u
){
  let item=g.x;let count=atomicLoad(&bandPhiFrontier[0]);
  if(item>=count||item+1u>=arrayLength(&bandPhiFrontier)){return;}
  let rowIndex=atomicLoad(&bandPhiFrontier[item+1u]);
  if(rowIndex!=INVALID&&rowIndex<p.rowCapacity&&rowIndex<arrayLength(&provisionalVelocities)){
    provisionalVelocities[rowIndex]=extendBandRowPhi(rowIndex);
  }
}
fn recordBandPhiRowFailure(rowIndex:u32,row:Row,state:vec4f){if(rowIndex>=transitionControl.failureBand){return;}transitionControl.failureBand=rowIndex;transitionControl.failureStage=OWNER_FAILURE_BAND_PHI;transitionControl.failureRowCell=row.cell;transitionControl.failureRowSize=row.size;transitionControl.failureDescriptor=row.flags;transitionControl.failureTopology=row.globalRow;transitionControl.failureTransformFlags=bitcast<u32>(state.x);transitionControl.failureSelector=bitcast<u32>(state.y);transitionControl.failureRawX=bitcast<u32>(state.z);transitionControl.failureRawY=bitcast<u32>(state.w);transitionControl.failureRawZ=rowIndex;transitionControl.failureRequestedSize=0u;transitionControl.failureResolvedCell=0u;transitionControl.failureBoundaryFlips=0u;transitionControl.failureOwnerCell=0u;transitionControl.failureOwnerSizeValid=0u;}
@compute @workgroup_size(64)fn commitBandRowPhi(@builtin(global_invocation_id)g:vec3u){let rowIndex=g.x;if(rowIndex>=transitionControl.endpointEnd||rowIndex>=arrayLength(&rows)||rowIndex>=arrayLength(&rowVelocities)){return;}let state=rowVelocities[rowIndex];if(state.w==0.||state.y==0.||!finite(state.x)){recordBandPhiRowFailure(rowIndex,rows[rowIndex],state);phiFail(OUTSIDE_FINE_BAND,rows[rowIndex].cell);return;}var row=rows[rowIndex];row.representativePhi=state.x;row.minimumPhi=state.x;row.maximumPhi=state.x;row.flags=(row.flags&~(ROW_CENTER_WET|ROW_PHI_EXTENDED))|ROW_PHI|select(0u,ROW_CENTER_WET,state.x<0.)|select(0u,ROW_PHI_EXTENDED,state.z==0.);rows[rowIndex]=row;}
@compute @workgroup_size(64)fn sampleBandFaceCoarsePhi(@builtin(global_invocation_id)g:vec3u){let index=liveFaceIndex(g.x);if(index==INVALID||index>=p.faceCapacity||index>=arrayLength(&faces)){return;}var face=faces[index];if((face.flags&(LIVE|PHI_VALID))!=LIVE){return;}var anchor=face.negativeRow;var sampled=coarsePhiCptAtPoint(anchor,face.centroid.xyz);if(sampled.valid==0u&&face.positiveRow<transitionControl.support3NodeEnd){anchor=face.positiveRow;sampled=coarsePhiCptAtPoint(anchor,face.centroid.xyz);}if(sampled.valid==0u){let edge=exactRegularFaceEdgePhi(face);if(edge.y!=0.&&edge.x<=0.){sampled=PhiCPT(edge.x,vec3f(0.),1u);}}if(sampled.valid==0u){var phiRecord=diagnoseCoarsePhiAtPoint(face.negativeRow,face.centroid.xyz);if(phiRecord.cause==INVALID&&face.positiveRow<transitionControl.support3NodeEnd){phiRecord=diagnoseCoarsePhiAtPoint(face.positiveRow,face.centroid.xyz);}if(phiRecord.cause==INVALID){phiRecord=phiDiagnostic(vec3i(0),0u,face.negativeRow,PHI_PATH_ANCHOR,INVALID,PHI_CAUSE_INVALID_SELECTOR,0xb0000u);}face.velocity=bitcast<vec4f>(vec4u(bitcast<u32>(phiRecord.origin.x),bitcast<u32>(phiRecord.origin.y),bitcast<u32>(phiRecord.origin.z),phiRecord.size));face.phi=bitcast<f32>(phiRecord.anchor);face.area=bitcast<f32>(phiRecord.selector);face.pad=(phiRecord.path&255u)|((phiRecord.cause&255u)<<8u)|((phiRecord.detail&65535u)<<16u);face.flags|=PHI_DIAGNOSTIC;faces[index]=face;return;}let published=publishClosestPoint(face,sampled,fp.fineWidth*f32(fp.fineFactor));if((published.flags&PHI_VALID)==0u){face.flags|=PHI_DIAGNOSTIC;face.pad=(PHI_CAUSE_INVALID_METRIC<<8u);faces[index]=face;return;}faces[index]=published;faces[index].flags|=COARSE_PHI;}
@compute @workgroup_size(256)fn reduceBandPhiFailure(@builtin(local_invocation_index)lane:u32){
 let liveCount=select(0u,liveFaceWorklist[0],arrayLength(&liveFaceWorklist)>0u);
 let range=topologyPublishLaneRange(liveCount,lane);var counts=0u;var first=INVALID;
 for(var item=range.x;item<range.y;item+=1u){
  let index=liveFaceIndex(item);if(index==INVALID||index>=arrayLength(&faces)){continue;}
  let face=faces[index];if((face.flags&PHI_DIAGNOSTIC)==0u){continue;}
  let cause=(face.pad>>8u)&255u;if(cause<4u){counts+=1u<<(cause*8u);}first=min(first,index);
 }
 topologyPublishSums[lane]=vec4u(counts,first,0u,0u);
 for(var width=128u;width>0u;width>>=1u){
  workgroupBarrier();if(lane<width){let right=topologyPublishSums[lane+width];
   topologyPublishSums[lane].x+=right.x;topologyPublishSums[lane].y=min(topologyPublishSums[lane].y,right.y);}
 }
 workgroupBarrier();if(lane!=0u){return;}
 let reduced=topologyPublishSums[0];transitionControl.phiFailureCounts=reduced.x;
 let diagnosticFirst=reduced.y;if(diagnosticFirst==INVALID){return;}
 transitionControl.failureBand=PHI_FAILURE_TAG|diagnosticFirst;let face=faces[diagnosticFirst];
 let packedOrigin=bitcast<vec4u>(face.velocity);let cause=(face.pad>>8u)&255u;let detail=(face.pad>>16u)&65535u;
 transitionControl.failureStage=diagnosticFirst;transitionControl.failureRowCell=face.globalFace;
 transitionControl.failureRowSize=face.negativeRow;transitionControl.failureDescriptor=face.positiveRow;
 transitionControl.failureTopology=bitcast<u32>(face.phi);
 transitionControl.failureTransformFlags=bitcast<u32>(face.centroid.x);
 transitionControl.failureSelector=bitcast<u32>(face.centroid.y);
 transitionControl.failureRawX=bitcast<u32>(face.centroid.z);transitionControl.failureRawY=face.pad&255u;
 transitionControl.failureRawZ=packedOrigin.x;transitionControl.failureRequestedSize=packedOrigin.y;
 transitionControl.failureResolvedCell=packedOrigin.z;transitionControl.failureBoundaryFlips=packedOrigin.w;
 transitionControl.failureOwnerCell=bitcast<u32>(face.area);
 transitionControl.failureOwnerSizeValid=cause|(detail<<8u);
}
@compute @workgroup_size(64)fn summarizeBandRowPhi(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=transitionControl.support1End||row>=arrayLength(&rows)||control.flags!=0u){return;}let count=min(incidence[row],p.axisStride);var minimum=3.402823e38;var maximum=-3.402823e38;var representative=0.;var representativeAbs=3.402823e38;var representativeFace=INVALID;var found=0u;for(var local=0u;local<count;local+=1u){let index=incidence[p.rowCapacity+row*p.axisStride+local];if(index>=p.faceCapacity||index>=arrayLength(&faces)){phiFail(BAD_PHI,row);return;}let face=faces[index];if((face.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||(face.negativeRow!=row&&face.positiveRow!=row)||!finite(face.phi)){phiFail(BAD_PHI,row);return;}minimum=min(minimum,face.phi);maximum=max(maximum,face.phi);let magnitude=abs(face.phi);if(magnitude<representativeAbs||(magnitude==representativeAbs&&face.globalFace<representativeFace)){representative=face.phi;representativeAbs=magnitude;representativeFace=face.globalFace;}found+=1u;}if(found==0u){phiFail(BAD_PHI,row);return;}rows[row].representativePhi=representative;rows[row].minimumPhi=minimum;rows[row].maximumPhi=maximum;rows[row].flags|=ROW_PHI;}
fn velocityValid(v:vec4f)->bool{return v.w>0.&&finite(v.x)&&finite(v.y)&&finite(v.z);}fn cellVector(origin:vec3u,size:u32)->vec4f{let band=rowOfIdentity(cell(origin),size);if(band==INVALID||band>=arrayLength(&rows)){return vec4f(0);}let row=rows[band];if((row.flags&ROW_CORE)==0u||row.globalRow>=p.powerRowCapacity||row.globalRow>=arrayLength(&powerRowVelocities)){return vec4f(0);}let velocity=powerRowVelocities[row.globalRow];return select(vec4f(0),velocity,velocityValid(velocity));}fn powerTransform(value:vec3f,code:u32)->vec3f{let permutation=(code/8u)%6u;var result=value;if(permutation==1u){result=value.xzy;}else if(permutation==2u){result=value.yxz;}else if(permutation==3u){result=value.yzx;}else if(permutation==4u){result=value.zxy;}else if(permutation==5u){result=value.zyx;}let bits=code&7u;return result*vec3f(select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));}fn tetraWeights(point:vec3f,a:vec3f,b:vec3f,c:vec3f)->vec4f{let determinant=dot(a,cross(b,c));if(!finite(determinant)||abs(determinant)<=1e-10){return vec4f(-2.);}let wa=dot(point,cross(b,c))/determinant;let wb=dot(a,cross(point,c))/determinant;let wc=dot(a,cross(b,point))/determinant;return vec4f(1.-wa-wb-wc,wa,wb,wc);}fn contained(weights:vec4f)->bool{return all(weights>=vec4f(-2e-6))&&all(weights<=vec4f(1.000002));}
fn reflectComponents(value:vec4f,mask:i32)->vec4f{if(!velocityValid(value)){return vec4f(0);}var result=value;for(var axis=0u;axis<3u;axis+=1u){if((mask&(1i<<axis))!=0){result[axis]=-result[axis];}}return result;}
fn compactSignedVector(origin:vec3i,size:u32)->vec4f{let reflected=velocityExtendedOrigin(origin,size);if(reflected.w<0){return vec4f(0);}return reflectComponents(cellVector(vec3u(reflected.xyz),size),reflected.w);}
fn selectorVector(anchor:u32,selector:u32,transform:u32)->vec4f{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return vec4f(0);}let row=rows[anchor];let v=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(v.xyz,transform);let sizeFloat=f32(row.size)*v.w;let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return vec4f(0);}return compactSignedVector(vec3i(round(point-.5*f32(size))),size);}
fn liquidCentroidVector(anchor:u32,pointGrid:vec3f)->LiquidInterpolation{if(anchor>=arrayLength(&rows)||anchor>=arrayLength(&metrics)){return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);}let row=rows[anchor];let origin=coord(row.cell);let anchorVelocity=cellVector(origin,row.size);let metric=metrics[anchor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);}let header=tetraHeaders[metric.topology];if((header.flags&1u)!=0u){let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);if(any(t<vec3f(-2e-6))||any(t>vec3f(1.000002))){return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);}var result=vec3f(0);var validWeight=0.;for(var corner=0u;corner<8u;corner+=1u){let weight=max(0.,select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u));if(weight==0.){continue;}let cornerOrigin=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u))*i32(row.size);let v=compactSignedVector(cornerOrigin,row.size);if(!velocityValid(v)){continue;}result+=weight*v.xyz;validWeight+=weight;}if(validWeight<=1e-6){return LiquidInterpolation(vec4f(0),CPT_MISSING_VERTEX);}result/=validWeight;let value=select(vec4f(0),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));return LiquidInterpolation(value,select(CPT_MISSING_VERTEX,0u,velocityValid(value)));}if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);}let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}let wetWeights=max(vec4f(0),weights);let va=selectorVector(anchor,selectors.x,metric.transformFlags&63u);let vb=selectorVector(anchor,selectors.y,metric.transformFlags&63u);let vc=selectorVector(anchor,selectors.z,metric.transformFlags&63u);var result=vec3f(0);var validWeight=0.;if(velocityValid(anchorVelocity)){result+=wetWeights.x*anchorVelocity.xyz;validWeight+=wetWeights.x;}if(velocityValid(va)){result+=wetWeights.y*va.xyz;validWeight+=wetWeights.y;}if(velocityValid(vb)){result+=wetWeights.z*vb.xyz;validWeight+=wetWeights.z;}if(velocityValid(vc)){result+=wetWeights.w*vc.xyz;validWeight+=wetWeights.w;}if(validWeight<=1e-6){return LiquidInterpolation(vec4f(0),CPT_MISSING_VERTEX);}result/=validWeight;let value=select(vec4f(0),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));return LiquidInterpolation(value,select(CPT_MISSING_VERTEX,0u,velocityValid(value)));}return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);}
fn centroidVector(anchor:u32,pointGrid:vec3f)->vec4f{return liquidCentroidVector(anchor,pointGrid).value;}
fn exactRegularFaceCoreVector(face:Face)->vec4f{let edge=exactRegularFaceEdgeRows(face);if(edge.x==INVALID){return vec4f(0);}let negativeRow=rows[edge.x];let positiveRow=rows[edge.y];let negative=cellVector(coord(negativeRow.cell),negativeRow.size);let positive=cellVector(coord(positiveRow.cell),positiveRow.size);if(!velocityValid(negative)||!velocityValid(positive)){return vec4f(0);}let value=.5*(negative.xyz+positive.xyz);return select(vec4f(0),vec4f(value,1.),finite(value.x)&&finite(value.y)&&finite(value.z));}
// Aanjaneya et al. Section 5 defines the known regular-face field on the
// liquid cells before extending it into air.  For an adaptive cell, a
// negative face-centroid sample is sufficient but not necessary: the
// interface can cut a corner while every subdivided face centroid lies in
// air.  The current coarse volume classification is the exact authority for
// whether a cell contains liquid, so publish the complete incident face star
// for liquid and mixed cells before the independent extension dispatch.
fn rowCarriesLiquid(rowIndex:u32)->bool{
  if(rowIndex==INVALID||rowIndex>=arrayLength(&rows)){return false;}
  let flags=rows[rowIndex].flags;
  // The centre sign is sufficient for uniform wet cells.  Adaptive cells can
  // contain a corner of liquid while their centre and every subdivided face
  // centroid remain in air; the current coarse min/max classification is the
  // conservative volume authority for that case.
  return (flags&(ROW_PHI|ROW_CENTER_WET))==(ROW_PHI|ROW_CENTER_WET)
    ||(flags&(ROW_COARSE_LIQUID|ROW_COARSE_MIXED))!=0u;
}
fn faceCarriesLiquid(face:Face)->bool{return face.phi<=0.||rowCarriesLiquid(face.negativeRow)||rowCarriesLiquid(face.positiveRow);}
fn residualPhiEpsilon(scale:f32)->f32{return max(1e-8,1e-5*scale);}
fn residualEndpointIsDry(rowIndex:u32)->bool{
  if(rowIndex==INVALID||rowIndex>=p.rowCapacity
    ||rowIndex>=arrayLength(&rows)||rowIndex>=arrayLength(&incidence)){return false;}
  let row=rows[rowIndex];
  // S0/S1 extrema are the exact current incident-face reduction.
  if(rowIndex<transitionControl.support1End){
    if((row.flags&ROW_PHI)==0u||!finite(row.minimumPhi)
      ||!finite(row.representativePhi)||!finite(row.maximumPhi)
      ||row.minimumPhi>row.representativePhi
      ||row.representativePhi>row.maximumPhi){return false;}
    let scale=max(abs(row.minimumPhi),max(abs(row.representativePhi),abs(row.maximumPhi)));
    return row.minimumPhi>residualPhiEpsilon(scale);
  }
  // S2 phi is marched and min=max there, so prove dryness from the complete
  // fixed incident star instead of mistaking the marched scalar for a star
  // minimum. This remains O(axisStride) and never scans the row/face arena.
  let count=min(incidence[rowIndex],p.axisStride);
  let base=p.rowCapacity+rowIndex*p.axisStride;
  if(count==0u||base>arrayLength(&incidence)||count>arrayLength(&incidence)-base){return false;}
  var minimum=3.402823e38;
  var scale=0.;
  for(var local=0u;local<count;local+=1u){
    let slot=incidence[base+local];
    if(slot>=p.faceCapacity||slot>=arrayLength(&faces)){return false;}
    let incident=faces[slot];
    if((incident.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)
      ||(incident.negativeRow!=rowIndex&&incident.positiveRow!=rowIndex)
      ||!finite(incident.phi)){return false;}
    minimum=min(minimum,incident.phi);
    scale=max(scale,abs(incident.phi));
  }
  return minimum>residualPhiEpsilon(scale);
}
// Seed eligibility is deliberately conservative: ROW_COARSE_MIXED preserves
// a corner-cut liquid cell even when its current face samples are positive.
// A failed non-seed face has already missed that known-liquid interpolation
// path, so its bounded closest-face repair instead uses current Section 5 phi
// authority.  Every present endpoint must have a strictly positive minimum;
// missing endpoints remain fail-closed because they have no complete star.
fn residualFaceIsDry(face:Face)->bool{
  return finite(face.phi)&&face.phi>0.
    &&residualEndpointIsDry(face.negativeRow)
    &&face.positiveRow!=INVALID&&residualEndpointIsDry(face.positiveRow);
}
// A face whose closest point reached a real local cube/Delaunay fan can still
// have a wet incident endpoint: that is precisely the interface neighborhood
// in which a containing fan may lack one velocity vertex. This is not the
// support-owner outward march, so requiring both complete endpoint stars to
// be dry would strand the face beside an already-valid interface carrier.
// Keep this exception limited to interpolation-domain gaps and to an ordinary
// finite two-ended face. A support-owner result means ownerAt found a real
// geometric owner just beyond the bounded interpolation publication; it is
// the same local extension case, not a corrupt or ownerless sample. Commit
// applies a physical signed-distance reach bound before accepting its carrier.
fn localInterpolationRepairEligible(face:Face)->bool{
  return (face.pad==CPT_MISSING_VERTEX||face.pad==CPT_NO_SIMPLEX
      ||face.pad==CPT_SUPPORT_OWNER)
    &&finite(face.phi)
    &&face.negativeRow!=INVALID&&face.positiveRow!=INVALID;
}
// Terminal S3 endpoints exist only to close scalar and local-Delaunay
// topology. They are deliberately rejected by finalPointVector and are
// outside the S0/S1 transient physical point field. Their incident faces
// remain LIVE for phi/incidence closure but are not velocity unknowns.
fn faceVelocityTarget(face:Face)->bool{return (face.flags&(LIVE|VELOCITY_TARGET))==(LIVE|VELOCITY_TARGET);}
@compute @workgroup_size(64)fn seedFaceCentroids(@builtin(global_invocation_id)g:vec3u){let face=liveFaceIndex(g.x);if(face==INVALID||face>=p.faceCapacity||face>=arrayLength(&faces)){return;}var f=faces[face];if((f.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||f.negativeRow>=arrayLength(&rows)||!finite(f.phi)||!faceVelocityTarget(f)){return;}if(!faceCarriesLiquid(f)){f.pad=1u;faces[face]=f;return;}let exact=exactRegularFaceCoreVector(f);if(velocityValid(exact)){f.velocity=exact;f.pad=0u;f.flags|=SEED|FACE_VELOCITY_VALID;faces[face]=f;return;}var reason=0u;var velocity=centroidVector(f.negativeRow,f.centroid.xyz);if(!velocityValid(velocity)){reason|=2u;}if(!velocityValid(velocity)&&f.positiveRow!=INVALID){velocity=centroidVector(f.positiveRow,f.centroid.xyz);if(!velocityValid(velocity)){reason|=4u;}}if(!velocityValid(velocity)){f.pad=reason;faces[face]=f;return;}f.velocity=velocity;f.pad=0u;f.flags|=SEED|FACE_VELOCITY_VALID;faces[face]=f;}
fn seededIncidentVector(rowIndex:u32)->vec4f{
  if(rowIndex>=arrayLength(&rows)){return vec4f(0);}
  let count=min(incidence[rowIndex],p.axisStride);
  var weighted=vec3f(0);
  var weightSum=0.;
  for(var local=0u;local<count;local+=1u){
    let faceIndex=incidence[p.rowCapacity+rowIndex*p.axisStride+local];
    if(faceIndex>=p.faceCapacity||faceIndex>=arrayLength(&faces)){continue;}
    let face=faces[faceIndex];
    if((face.flags&(LIVE|SEED))!=(LIVE|SEED)||(face.negativeRow!=rowIndex&&face.positiveRow!=rowIndex)
      ||!velocityValid(face.velocity)||!finite(face.area)||face.area<=0.){continue;}
    weighted+=face.area*face.velocity.xyz;
    weightSum+=face.area;
  }
  if(weightSum<=1e-8){return vec4f(0);}
  let value=weighted/weightSum;
  return select(vec4f(0),vec4f(value,1.),finite(value.x)&&finite(value.y)&&finite(value.z));
}
// S2/S3 rows close interpolation topology but are not pressure unknowns.
// Their regular-face carriers already contain complete vectors after the
// closest-point pass, so a rank-deficient axis star is completed by the
// deterministic area-weighted constant extension prescribed by Section 5.
fn completedIncidentVector(rowIndex:u32)->vec4f{
  if(rowIndex>=arrayLength(&rows)){return vec4f(0);}
  let count=min(incidence[rowIndex],p.axisStride);
  var weighted=vec3f(0);
  var weightSum=0.;
  for(var local=0u;local<count;local+=1u){
    let faceIndex=incidence[p.rowCapacity+rowIndex*p.axisStride+local];
    if(faceIndex>=p.faceCapacity||faceIndex>=arrayLength(&faces)){continue;}
    let face=faces[faceIndex];
    if(!faceVelocityTarget(face)||(face.flags&FACE_VELOCITY_VALID)==0u
      ||(face.negativeRow!=rowIndex&&face.positiveRow!=rowIndex)
      ||!velocityValid(face.velocity)||!finite(face.area)||face.area<=0.){continue;}
    weighted+=face.area*face.velocity.xyz;
    weightSum+=face.area;
  }
  if(weightSum<=1e-8){return vec4f(0);}
  let value=weighted/weightSum;
  return select(vec4f(0),vec4f(value,1.),finite(value.x)&&finite(value.y)&&finite(value.z));
}
fn seededSignedIncidentVector(origin:vec3i,size:u32)->vec4f{
  let reflected=velocityExtendedOrigin(origin,size);
  if(reflected.w<0){return vec4f(0);}
  let resolved=vec3u(reflected.xyz);
  let band=rowOfIdentity(cell(resolved),size);
  if(band==INVALID||band>=arrayLength(&rows)){return vec4f(0);}
  let row=rows[band];
  if(row.cell!=cell(resolved)||row.size!=size){return vec4f(0);}
  return reflectComponents(seededIncidentVector(band),reflected.w);
}
fn seededSelectorVector(anchor:u32,selector:u32,transform:u32)->vec4f{
  if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return vec4f(0);}
  let row=rows[anchor];
  let vertex=tetraVertices[selector].v;
  let center=vec3f(coord(row.cell))+.5*f32(row.size);
  let point=center+f32(row.size)*inversePowerTransform(vertex.xyz,transform);
  let sizeFloat=f32(row.size)*vertex.w;
  let size=u32(round(sizeFloat));
  if(size==0u||abs(sizeFloat-f32(size))>1e-4){return vec4f(0);}
  return seededSignedIncidentVector(vec3i(round(point-.5*f32(size))),size);
}
fn wetFaceVectorAtPoint(anchor:u32,pointGrid:vec3f)->LiquidInterpolation{
  if(anchor>=arrayLength(&rows)||anchor>=arrayLength(&metrics)){return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);}
  let row=rows[anchor];
  let origin=coord(row.cell);
  let metric=metrics[anchor];
  if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);}
  let header=tetraHeaders[metric.topology];
  if((header.flags&1u)!=0u){
    let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));
    let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);
    if(any(t<vec3f(-2e-6))||any(t>vec3f(1.000002))){return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);}
    var result=vec3f(0);
    var validWeight=0.;
    for(var corner=0u;corner<8u;corner+=1u){
      let weight=max(0.,select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u));
      if(weight==0.){continue;}
      let cornerOrigin=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u))*i32(row.size);
      let value=seededSignedIncidentVector(cornerOrigin,row.size);
      if(!velocityValid(value)){continue;}
      result+=weight*value.xyz;
      validWeight+=weight;
    }
    if(validWeight<=1e-6){return LiquidInterpolation(vec4f(0),CPT_MISSING_VERTEX);}
    result/=validWeight;
    let value=select(vec4f(0),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));
    return LiquidInterpolation(value,select(CPT_MISSING_VERTEX,0u,velocityValid(value)));
  }
  if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);}
  let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);
  for(var local=0u;local<header.count;local+=1u){
    let packed=tetrahedra[header.first+local];
    let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);
    if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);}
    let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);
    if(!contained(weights)){continue;}
    let wetWeights=max(vec4f(0),weights);
    let anchorValue=seededIncidentVector(anchor);
    let a=seededSelectorVector(anchor,selectors.x,metric.transformFlags&63u);
    let b=seededSelectorVector(anchor,selectors.y,metric.transformFlags&63u);
    let c=seededSelectorVector(anchor,selectors.z,metric.transformFlags&63u);
    var result=vec3f(0);
    var validWeight=0.;
    if(velocityValid(anchorValue)){result+=wetWeights.x*anchorValue.xyz;validWeight+=wetWeights.x;}
    if(velocityValid(a)){result+=wetWeights.y*a.xyz;validWeight+=wetWeights.y;}
    if(velocityValid(b)){result+=wetWeights.z*b.xyz;validWeight+=wetWeights.z;}
    if(velocityValid(c)){result+=wetWeights.w*c.xyz;validWeight+=wetWeights.w;}
    if(validWeight<=1e-6){return LiquidInterpolation(vec4f(0),CPT_MISSING_VERTEX);}
    result/=validWeight;
    let value=select(vec4f(0),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));
    return LiquidInterpolation(value,select(CPT_MISSING_VERTEX,0u,velocityValid(value)));
  }
  return LiquidInterpolation(vec4f(0),CPT_NO_SIMPLEX);
}
struct SeededFaceCarrier{value:vec4f,distanceSquared:f32,face:u32}
fn closestSeededFaceCarrier(rowIndex:u32,pointGrid:vec3f)->SeededFaceCarrier{
  var best=SeededFaceCarrier(vec4f(0),3.402823e38,INVALID);
  if(rowIndex>=arrayLength(&rows)||(rows[rowIndex].flags&ROW_SUPPORT3_ENDPOINT)!=0u){return best;}
  let count=min(incidence[rowIndex],p.axisStride);
  for(var local=0u;local<count;local+=1u){
    let faceIndex=incidence[p.rowCapacity+rowIndex*p.axisStride+local];
    if(faceIndex>=p.faceCapacity||faceIndex>=arrayLength(&faces)){continue;}
    let face=faces[faceIndex];
    if((face.flags&(LIVE|SEED|PHI_VALID|FACE_VELOCITY_VALID))
        !=(LIVE|SEED|PHI_VALID|FACE_VELOCITY_VALID)
      ||(face.negativeRow!=rowIndex&&face.positiveRow!=rowIndex)
      ||!velocityValid(face.velocity)
      ||!finite(face.centroid.x)||!finite(face.centroid.y)||!finite(face.centroid.z)){continue;}
    let delta=face.centroid.xyz-pointGrid;
    let distanceSquared=dot(delta,delta);
    if(!finite(distanceSquared)){continue;}
    if(distanceSquared<best.distanceSquared
      ||(distanceSquared==best.distanceSquared&&faceIndex<best.face)){
      best=SeededFaceCarrier(face.velocity,distanceSquared,faceIndex);
    }
  }
  return best;
}
// A closest point can be geometrically contained by a neighboring catalog
// fan rather than the half-open owner fan (notably for co-spherical adaptive
// sites). Section 5 still requires a true cube/tetra interpolant on the liquid
// side, so search only the proven fixed selector-radius box and accept only a
// geometrically containing interpolant. This is independent of row capacity
// and is not the later air-side constant extension.
fn resolveWetFaceVectorAtPoint(initialAnchor:u32,pointGrid:vec3f)->LiquidInterpolation{
  let direct=wetFaceVectorAtPoint(initialAnchor,pointGrid);
  let localFanGap=direct.reason==CPT_NO_SIMPLEX||direct.reason==CPT_MISSING_VERTEX;
  if(velocityValid(direct.value)||!localFanGap){return direct;}
  // Section 5 extends the regular-face field by copying the velocity of the
  // face closest to the free surface. A co-spherical catalog can leave either
  // no selected containing simplex or a containing simplex whose
  // positive-weight vertices lack an immutable wet carrier. In both local-fan
  // cases, first exhaust the complete fixed catalog-radius interpolation
  // search, then use only the nearest immutable seeded face. A valid
  // cube/tetra interpolation below always wins.
  let carrierEligible=localFanGap;
  var bestCarrier=SeededFaceCarrier(vec4f(0),3.402823e38,INVALID);
  if(carrierEligible){bestCarrier=closestSeededFaceCarrier(initialAnchor,pointGrid);}
  var bestRow=INVALID;
  var best=direct;
  var size=1u;
  loop{
    let base=vec3i(floor(pointGrid/f32(size)));
    for(var dz=-2i;dz<=2i;dz+=1i){for(var dy=-2i;dy<=2i;dy+=1i){for(var dx=-2i;dx<=2i;dx+=1i){
      let originIndex=base+vec3i(dx,dy,dz);
      if(any(originIndex<vec3i(0))){continue;}
      let origin=vec3u(originIndex)*size;
      if(any(origin+vec3u(size)>p.dims)){continue;}
      let candidate=rowOfIdentity(cell(origin),size);
      if(candidate==INVALID||candidate==initialAnchor
        ||candidate>=arrayLength(&rows)||candidate>=arrayLength(&metrics)
        ||(rows[candidate].flags&ROW_SUPPORT3_ENDPOINT)!=0u){continue;}
      if(carrierEligible&&bestRow==INVALID){
        let carrier=closestSeededFaceCarrier(candidate,pointGrid);
        if(carrier.face!=INVALID&&(carrier.distanceSquared<bestCarrier.distanceSquared
          ||(carrier.distanceSquared==bestCarrier.distanceSquared&&carrier.face<bestCarrier.face))){
          bestCarrier=carrier;
        }
      }
      if(candidate>=bestRow){continue;}
      let sampled=wetFaceVectorAtPoint(candidate,pointGrid);
      if(velocityValid(sampled.value)){bestRow=candidate;best=sampled;}
    }}}
    if(size>=p.maximumLeaf){break;}
    size<<=1u;
  }
  if(bestRow!=INVALID){return best;}
  if(bestCarrier.face!=INVALID){return LiquidInterpolation(bestCarrier.value,0u);}
  return direct;
}
fn closestPointLiquidVector(face:Face)->LiquidInterpolation{
  let point=halfOpenDomainPoint(face.velocity.xyz);
  // Resolve a half-open owner on the liquid side, but evaluate the paper's
  // interpolant at the physical closest point itself.
  let inward=point-face.centroid.xyz;let inwardLength=length(inward);
  let ownerPoint=select(point,halfOpenDomainPoint(point+1e-4*inward/inwardLength),inwardLength>1e-8);
  let owner=ownerAt(vec3u(floor(ownerPoint)));
  if(owner.valid==0u||owner.size==0u){return LiquidInterpolation(vec4f(0),CPT_NO_OWNER);}
  let band=rowOfIdentity(cell(owner.origin),owner.size);
  // A valid geometric owner that is intentionally absent from the bounded
  // liquid-interpolation publication is a support-only gap. A directory
  // result that points outside the publication is corruption, not a gap that
  // Section 5 face extrapolation may conceal.
  if(band==INVALID){return LiquidInterpolation(vec4f(0),CPT_SUPPORT_OWNER);}
  if(band>=arrayLength(&rows)||band>=arrayLength(&metrics)){
    return LiquidInterpolation(vec4f(0),CPT_OWNER_ROW_MISMATCH);
  }
  let row=rows[band];
  if(row.cell!=cell(owner.origin)||row.size!=owner.size){
    return LiquidInterpolation(vec4f(0),CPT_OWNER_ROW_MISMATCH);
  }
  var sampled=resolveWetFaceVectorAtPoint(band,point);
  if(!velocityValid(sampled.value)){sampled.value.w=-f32(band+1u);}
  return sampled;
}
const CPT_BAD_FACE:u32=5u;
const CPT_NO_CLOSEST_POINT:u32=6u;
fn rejectClosestPointFace(index:u32,face:Face){
  // The face slot is the immutable diagnostic record. The terminal reduction
  // derives flags, counts, and stable first identities in face-index order.
  faces[index]=face;
}
@compute @workgroup_size(64)fn extendFaceClosestPoints(@builtin(global_invocation_id)g:vec3u){
  let index=liveFaceIndex(g.x);
  if(index==INVALID||index>=p.faceCapacity||index>=arrayLength(&faces)){return;}
  var face=faces[index];
  if((face.flags&LIVE)==0u){return;}
  if(!faceVelocityTarget(face)){return;}
  if((face.flags&PHI_VALID)==0u||!finite(face.phi)||!finite(face.area)||face.area<=0.){
    face.pad=CPT_BAD_FACE;
    faces[index]=face;
    rejectClosestPointFace(index,face);
    return;
  }
  if((face.flags&SEED)!=0u){
    if((face.flags&FACE_VELOCITY_VALID)==0u||!velocityValid(face.velocity)){
      face.pad=CPT_MISSING_VERTEX;
      faces[index]=face;
      rejectClosestPointFace(index,face);
      return;
    }
    return;
  }
  if((face.flags&CLOSEST_POINT_VALID)==0u){
    face.pad=CPT_NO_CLOSEST_POINT;
    faces[index]=face;
    rejectClosestPointFace(index,face);
    return;
  }
  let sampled=closestPointLiquidVector(face);
  if(!velocityValid(sampled.value)){
    face.pad=sampled.reason;
    // Failure-only telemetry: retain the physical closest point in xyz and
    // encode its exact containing row in w. This never reaches publication.
    face.velocity.w=sampled.value.w;
    faces[index]=face;
    rejectClosestPointFace(index,face);
    return;
  }
  face.velocity=sampled.value;
  face.pad=0u;
  face.flags|=FACE_VELOCITY_VALID|PRIMARY_EXTENSION;
  faces[index]=face;
}
struct ClosestFaceCarrier{
  absolutePhi:f32,distanceSquared:f32,seedRank:u32,globalFace:u32,slot:u32
}
fn invalidClosestFaceCarrier()->ClosestFaceCarrier{
  return ClosestFaceCarrier(3.402823e38,3.402823e38,INVALID,INVALID,INVALID);
}
fn closerFaceCarrier(candidate:ClosestFaceCarrier,best:ClosestFaceCarrier)->bool{
  return candidate.slot!=INVALID
    &&(candidate.absolutePhi<best.absolutePhi
      ||(candidate.absolutePhi==best.absolutePhi
        &&(candidate.seedRank<best.seedRank
          ||(candidate.seedRank==best.seedRank
            &&(candidate.distanceSquared<best.distanceSquared
              ||(candidate.distanceSquared==best.distanceSquared
                &&candidate.globalFace<best.globalFace))))));
}
fn closestValidTargetCarrier(rowIndex:u32,excluded:u32,point:vec3f,
 targetPhi:f32,positivePath:bool,requirePrimary:bool,
 allowUnorderedPositive:bool)->ClosestFaceCarrier{
  var best=invalidClosestFaceCarrier();
  if(rowIndex==INVALID||rowIndex>=p.rowCapacity||rowIndex>=arrayLength(&rows)
    ||rowIndex>=arrayLength(&incidence)){return best;}
  let count=min(incidence[rowIndex],p.axisStride);
  for(var local=0u;local<count;local+=1u){
    let at=p.rowCapacity+rowIndex*p.axisStride+local;
    if(at>=arrayLength(&incidence)){continue;}
    let slot=incidence[at];
    if(slot==excluded||slot>=p.faceCapacity||slot>=arrayLength(&faces)){continue;}
    let candidate=faces[slot];
    if(!faceVelocityTarget(candidate)
      ||(candidate.flags&(PHI_VALID|FACE_VELOCITY_VALID))!=(PHI_VALID|FACE_VELOCITY_VALID)
      ||(candidate.negativeRow!=rowIndex&&candidate.positiveRow!=rowIndex)
      ||!velocityValid(candidate.velocity)
      ||!finite(candidate.phi)||!finite(candidate.area)||candidate.area<=0.
      ||!finite(candidate.centroid.x)||!finite(candidate.centroid.y)||!finite(candidate.centroid.z)){continue;}
    let primary=(candidate.flags&PRIMARY_EXTENSION)!=0u;
    if(requirePrimary&&!primary){continue;}
    if(allowUnorderedPositive){
      let epsilon=residualPhiEpsilon(max(abs(targetPhi),abs(candidate.phi)));
      if(candidate.phi<=epsilon){continue;}
    }else if(positivePath){
      let epsilon=residualPhiEpsilon(max(abs(targetPhi),abs(candidate.phi)));
      if(candidate.phi<=epsilon||candidate.phi+epsilon>=targetPhi){continue;}
    }else if(primary&&(candidate.phi<0.||candidate.phi>=targetPhi)){continue;}
    let delta=candidate.centroid.xyz-point;
    let distanceSquared=dot(delta,delta);
    if(!finite(distanceSquared)){continue;}
    let carrier=ClosestFaceCarrier(abs(candidate.phi),distanceSquared,
      select(0u,1u,primary),candidate.globalFace,slot);
    if(closerFaceCarrier(carrier,best)){best=carrier;}
  }
  return best;
}
// A terminal S3 anchor may be the final positive closure cell before the
// interface. In that one exact case, Section 5's face marcher must be able to
// terminate at the immutable wet regular-face seed on the reached S2 star.
// This is not another graph step: the anchor bridge stays strictly positive,
// and a local signed-distance Lipschitz check proves that the seed is the
// adjacent interface crossing rather than an unrelated liquid face.
fn closestWetSeedAcrossAnchor(rowIndex:u32,excluded:u32,point:vec3f,
 bridge:Face)->ClosestFaceCarrier{
  var best=invalidClosestFaceCarrier();
  if(rowIndex==INVALID||rowIndex<transitionControl.support1End
    ||rowIndex>=transitionControl.support3NodeEnd
    ||rowIndex>=p.rowCapacity||rowIndex>=arrayLength(&rows)
    ||rowIndex>=arrayLength(&incidence)){return best;}
  let coarseWidth=fp.fineWidth*f32(fp.fineFactor);
  if(!finite(coarseWidth)||coarseWidth<=0.){return best;}
  let count=min(incidence[rowIndex],p.axisStride);
  for(var local=0u;local<count;local+=1u){
    let at=p.rowCapacity+rowIndex*p.axisStride+local;
    if(at>=arrayLength(&incidence)){continue;}
    let slot=incidence[at];
    if(slot==excluded||slot>=p.faceCapacity||slot>=arrayLength(&faces)){continue;}
    let candidate=faces[slot];
    let required=LIVE|VELOCITY_TARGET|PHI_VALID|SEED|FACE_VELOCITY_VALID;
    if((candidate.flags&required)!=required
      ||(candidate.flags&PRIMARY_EXTENSION)!=0u
      ||(candidate.negativeRow!=rowIndex&&candidate.positiveRow!=rowIndex)
      ||!velocityValid(candidate.velocity)
      ||!finite(candidate.phi)||!finite(candidate.area)||candidate.area<=0.
      ||!finite(candidate.centroid.x)||!finite(candidate.centroid.y)
      ||!finite(candidate.centroid.z)){continue;}
    let phiScale=max(abs(bridge.phi),abs(candidate.phi));
    let epsilon=residualPhiEpsilon(phiScale);
    if(candidate.phi>epsilon){continue;}
    let edge=bridge.centroid.xyz-candidate.centroid.xyz;
    let edgeDistance=coarseWidth*length(edge);
    let phiDistance=abs(bridge.phi-candidate.phi);
    let tolerance=max(4.*residualPhiEpsilon(max(phiScale,edgeDistance)),
      1e-3*coarseWidth);
    if(!finite(edgeDistance)||!finite(phiDistance)
      ||phiDistance>edgeDistance+tolerance){continue;}
    let delta=candidate.centroid.xyz-point;
    let distanceSquared=dot(delta,delta);
    if(!finite(distanceSquared)){continue;}
    let carrier=ClosestFaceCarrier(abs(candidate.phi),distanceSquared,
      0u,candidate.globalFace,slot);
    if(closerFaceCarrier(carrier,best)){best=carrier;}
  }
  return best;
}
fn closestCarrierThroughEndpoint(rowIndex:u32,excluded:u32,point:vec3f,
 targetPhi:f32,current:ClosestFaceCarrier)->ClosestFaceCarrier{
  var best=current;
  let immediate=closestValidTargetCarrier(rowIndex,excluded,point,targetPhi,true,false,false);
  if(closerFaceCarrier(immediate,best)){best=immediate;}
  if(rowIndex==INVALID||rowIndex>=p.rowCapacity
    ||rowIndex>=arrayLength(&rows)||rowIndex>=arrayLength(&incidence)){return best;}
  let count=min(incidence[rowIndex],p.axisStride);
  for(var local=0u;local<count;local+=1u){
    let at=p.rowCapacity+rowIndex*p.axisStride+local;
    if(at>=arrayLength(&incidence)){continue;}
    let bridgeSlot=incidence[at];
    if(bridgeSlot==excluded||bridgeSlot>=p.faceCapacity||bridgeSlot>=arrayLength(&faces)){continue;}
    let bridge=faces[bridgeSlot];
    if(!faceVelocityTarget(bridge)||(bridge.flags&PHI_VALID)==0u
      ||(bridge.negativeRow!=rowIndex&&bridge.positiveRow!=rowIndex)
      ||!finite(bridge.phi)||!finite(bridge.area)||bridge.area<=0.){continue;}
    let bridgeEpsilon=residualPhiEpsilon(max(abs(targetPhi),abs(bridge.phi)));
    if(bridge.phi<=bridgeEpsilon||bridge.phi+bridgeEpsilon>=targetPhi){continue;}
    let opposite=select(bridge.negativeRow,bridge.positiveRow,bridge.negativeRow==rowIndex);
    if(opposite==INVALID||opposite==rowIndex
      ||opposite>=transitionControl.support3NodeEnd){continue;}
    let secondRing=closestValidTargetCarrier(opposite,excluded,point,bridge.phi,true,false,false);
    if(closerFaceCarrier(secondRing,best)){best=secondRing;}
  }
  return best;
}
fn failedClosestPointAnchor(face:Face)->u32{
  if(face.pad!=CPT_NO_SIMPLEX||!finite(face.velocity.w)){return INVALID;}
  let encoded=-face.velocity.w;
  let rounded=round(encoded);
  if(encoded<1.||encoded>16777216.||abs(encoded-rounded)>1e-5){return INVALID;}
  return u32(rounded)-1u;
}
fn closestCarrierThroughFailedAnchor(excluded:u32,face:Face,point:vec3f,
 current:ClosestFaceCarrier)->ClosestFaceCarrier{
  let anchor=failedClosestPointAnchor(face);
  if(anchor==INVALID||anchor<transitionControl.support3NodeEnd
    ||anchor>=transitionControl.endpointEnd||anchor>=p.rowCapacity
    ||anchor>=arrayLength(&rows)||anchor>=arrayLength(&incidence)){return current;}
  let anchorRow=rows[anchor];
  if((anchorRow.flags&(ROW_PHI|ROW_SUPPORT3_ENDPOINT))
    !=(ROW_PHI|ROW_SUPPORT3_ENDPOINT)){return current;}
  let count=min(incidence[anchor],p.axisStride);
  let base=p.rowCapacity+anchor*p.axisStride;
  if(count==0u||base>arrayLength(&incidence)
    ||count>arrayLength(&incidence)-base){return current;}
  var minimum=3.402823e38;
  var scale=abs(face.phi);
  for(var local=0u;local<count;local+=1u){
    let bridgeSlot=incidence[base+local];
    if(bridgeSlot>=p.faceCapacity||bridgeSlot>=arrayLength(&faces)){return current;}
    let bridge=faces[bridgeSlot];
    if((bridge.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)
      ||(bridge.negativeRow!=anchor&&bridge.positiveRow!=anchor)
      ||!finite(bridge.phi)||!finite(bridge.area)||bridge.area<=0.
      ||!finite(bridge.centroid.x)||!finite(bridge.centroid.y)
      ||!finite(bridge.centroid.z)){return current;}
    minimum=min(minimum,bridge.phi);
    scale=max(scale,abs(bridge.phi));
  }
  if(minimum<=residualPhiEpsilon(scale)){return current;}
  var best=current;
  for(var local=0u;local<count;local+=1u){
    let bridgeSlot=incidence[base+local];
    let bridge=faces[bridgeSlot];
    let bridgeEpsilon=residualPhiEpsilon(max(abs(face.phi),abs(bridge.phi)));
    if(bridge.phi<=bridgeEpsilon||bridge.phi+bridgeEpsilon>=face.phi){continue;}
    let opposite=select(bridge.negativeRow,bridge.positiveRow,bridge.negativeRow==anchor);
    if(opposite<transitionControl.support1End
      ||opposite>=transitionControl.support3NodeEnd){continue;}
    let carrier=closestValidTargetCarrier(opposite,excluded,point,bridge.phi,true,true,false);
    if(closerFaceCarrier(carrier,best)){best=carrier;}
    let wetSeed=closestWetSeedAcrossAnchor(opposite,excluded,point,bridge);
    if(closerFaceCarrier(wetSeed,best)){best=wetSeed;}
  }
  return best;
}
// Aanjaneya et al. Section 5 extends the regular-face field into air by
// copying the face velocity closest to the free surface. The direct CPT
// interpolant above remains authoritative. Eight bounded follow-up
// gather/commit waves handle only residual interpolation-domain gaps and read
// only the two fixed endpoint incidence stars. Separate dispatches make every
// already-valid face publication immutable while candidates are ranked.
// CPT_SUPPORT_OWNER is also an interpolation-domain gap: ownerAt succeeded
// with a nonzero owner, but its exact support row was not published. The same
// positive endpoint-star proof may close it; CPT_NO_OWNER remains fail-closed.
// Each gather ranks immutable carriers in the two endpoint stars and their
// fixed one-hop neighbors. Each wave is one bounded two-ring search (at most
// 2*axisStride^2 candidates), not an iterative marcher. Every dry path and
// every later wave is strictly monotone in current positive phi:
// target > physical bridge/carrier > scale-aware epsilon.
// NO_SIMPLEX retains the exact containing support anchor as -(band+1) in
// velocity.w. One additional anchor-star branch may cross a
// positive scalar-closure face back to a nonterminal row, with the identical
// monotone primary rule or terminate at an adjacent immutable wet seed.
@compute @workgroup_size(64)fn gatherDryFaceClosestPointRepairs(@builtin(global_invocation_id)g:vec3u){
  let index=liveFaceIndex(g.x);
  if(index==INVALID||index>=p.faceCapacity||index>=arrayLength(&faces)||index>=arrayLength(&supportCandidateScratch)){return;}
  supportCandidateScratch[index]=SupportCandidate(0u,0u);
  let face=faces[index];
  if(!faceVelocityTarget(face)||(face.flags&FACE_VELOCITY_VALID)!=0u
    ||(face.flags&(PHI_VALID|CLOSEST_POINT_VALID))!=(PHI_VALID|CLOSEST_POINT_VALID)
    ||(face.pad!=CPT_SUPPORT_OWNER
      &&face.pad!=CPT_NO_SIMPLEX&&face.pad!=CPT_MISSING_VERTEX)
    ||!finite(face.area)||face.area<=0.
    ||!finite(face.velocity.x)||!finite(face.velocity.y)||!finite(face.velocity.z)){return;}
  let localInterpolationGap=localInterpolationRepairEligible(face);
  if(!residualFaceIsDry(face)&&!localInterpolationGap){return;}
  let point=face.velocity.xyz;
  var best=invalidClosestFaceCarrier();
  best=closestCarrierThroughEndpoint(face.negativeRow,index,point,face.phi,best);
  best=closestCarrierThroughEndpoint(face.positiveRow,index,point,face.phi,best);
  best=closestCarrierThroughFailedAnchor(index,face,point,best);
  // MISSING_VERTEX and NO_SIMPLEX are local interpolation-domain holes, not
  // another outward-march step. If the monotone/anchor searches found
  // nothing, close either from the nearest already-valid positive endpoint
  // face. The fixed waves may consume one such immutable local fallback;
  // support-owner gaps remain strictly phi-monotone and ownerless points stay
  // fail-closed.
  if(best.slot==INVALID&&localInterpolationGap){
    best=closestValidTargetCarrier(
      face.negativeRow,index,point,face.phi,false,false,true);
    let positive=closestValidTargetCarrier(
      face.positiveRow,index,point,face.phi,false,false,true);
    if(closerFaceCarrier(positive,best)){best=positive;}
  }
  supportCandidateScratch[index]=SupportCandidate(
    select(0u,best.slot+1u,best.slot!=INVALID),0u);
}
@compute @workgroup_size(64)fn commitDryFaceClosestPointRepairs(@builtin(global_invocation_id)g:vec3u){
  let index=liveFaceIndex(g.x);
  if(index==INVALID||index>=p.faceCapacity||index>=arrayLength(&faces)||index>=arrayLength(&supportCandidateScratch)){return;}
  let carrierPlusOne=supportCandidateScratch[index].cellPlusOne;
  if(carrierPlusOne==0u){return;}
  let carrierSlot=carrierPlusOne-1u;
  if(carrierSlot==INVALID||carrierSlot>=p.faceCapacity||carrierSlot>=arrayLength(&faces)){return;}
  var face=faces[index];
  let carrier=faces[carrierSlot];
  let localInterpolationGap=localInterpolationRepairEligible(face);
  if(!faceVelocityTarget(face)||(face.flags&FACE_VELOCITY_VALID)!=0u
    ||(face.flags&(PHI_VALID|CLOSEST_POINT_VALID))!=(PHI_VALID|CLOSEST_POINT_VALID)
    ||(face.pad!=CPT_SUPPORT_OWNER
      &&face.pad!=CPT_NO_SIMPLEX&&face.pad!=CPT_MISSING_VERTEX)
    ||(!residualFaceIsDry(face)&&!localInterpolationGap)
    ||!finite(face.area)||face.area<=0.
    ||!faceVelocityTarget(carrier)
    ||(carrier.flags&(PHI_VALID|FACE_VELOCITY_VALID))!=(PHI_VALID|FACE_VELOCITY_VALID)
    ||!velocityValid(carrier.velocity)||!finite(carrier.phi)
    ||!finite(carrier.area)||carrier.area<=0.){return;}
  let epsilon=residualPhiEpsilon(max(abs(face.phi),abs(carrier.phi)));
  let wetSeed=(carrier.flags&SEED)!=0u&&carrier.phi<=epsilon;
  let outwardAir=carrier.phi>epsilon&&carrier.phi+epsilon<face.phi;
  let carrierDistance=length(carrier.centroid.xyz-face.centroid.xyz)
    *fp.fineWidth*f32(fp.fineFactor);
  let localInterpolationCarrier=localInterpolationGap&&carrier.phi>epsilon
    &&abs(face.phi-carrier.phi)<=carrierDistance+epsilon;
  if(!wetSeed&&!outwardAir&&!localInterpolationCarrier){return;}
  face.velocity=vec4f(carrier.velocity.xyz,1.);
  face.pad=0u;
  face.flags=(face.flags&~PRIMARY_EXTENSION)|FACE_VELOCITY_VALID;
  faces[index]=face;
}
fn publishSupportClosureVector(row:u32,r:Row,origin:vec3u)->bool{if(row<transitionControl.support1End){return false;}let value=completedIncidentVector(row);if(!velocityValid(value)){return false;}var openScalars=0u;for(var axis=0u;axis<3u;axis+=1u){if(origin[axis]+r.size==p.dims[axis]&&(p.closedBoundaryMask&positiveBoundaryBit(axis))==0u){rows[row].padf=value[axis];openScalars+=1u;}}if(openScalars>1u){return false;}provisionalVelocities[row]=value;rowVelocities[row]=value;return true;}
@compute @workgroup_size(64)fn reconstructBandRowVelocity(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=transitionControl.endpointEnd||row>=arrayLength(&rows)||row>=arrayLength(&rowVelocities)||row>=arrayLength(&provisionalVelocities)){return;}provisionalVelocities[row]=vec4f(0.);if(row>=transitionControl.support3NodeEnd){rowVelocities[row]=vec4f(0.);return;}let r=rows[row];let origin=coord(r.cell);let count=min(incidence[row],p.axisStride);var ls=BandLS(vec4f(0.),vec4f(0.),vec4f(0.));for(var local=0u;local<count;local+=1u){let fi=incidence[p.rowCapacity+row*p.axisStride+local];if(fi>=p.faceCapacity||fi>=arrayLength(&faces)){vectorFail(INCOMPLETE,row);return;}let f=faces[fi];if(!faceVelocityTarget(f)){continue;}if((f.flags&FACE_VELOCITY_VALID)==0u){vectorFail(INCOMPLETE,row);return;}let axis=f.axisSpan&3u;if(axis>=3u||!finite(f.area)||f.area<=0.){vectorFail(INCOMPLETE,row);return;}var normal=vec3f(0.);let orientation=select(-1.,1.,f.negativeRow==row);if(f.negativeRow!=row&&f.positiveRow!=row){vectorFail(INCOMPLETE,row);return;}normal[axis]=orientation;let u=orientation*f.velocity[axis];let weight=f.area;if(!finite(u)){vectorFail(INCOMPLETE,row);return;}ls.a0.x+=weight*normal.x*normal.x;ls.a0.y+=weight*normal.x*normal.y;ls.a0.z+=weight*normal.x*normal.z;ls.a0.w+=weight*normal.y*normal.y;ls.a1.x+=weight*normal.y*normal.z;ls.a1.y+=weight*normal.z*normal.z;ls.a1.z+=weight*normal.x*u;ls.a1.w+=weight*normal.y*u;ls.a2.x+=weight*normal.z*u;}let xx=ls.a0.x;let xy=ls.a0.y;let xz=ls.a0.z;let yy=ls.a0.w;let yz=ls.a1.x;let zz=ls.a1.y;let b=vec3f(ls.a1.z,ls.a1.w,ls.a2.x);let c00=yy*zz-yz*yz;let c01=xz*yz-xy*zz;let c02=xy*yz-xz*yy;let c11=xx*zz-xz*xz;let c12=xy*xz-xx*yz;let c22=xx*yy-xy*xy;let determinant=xx*c00+xy*c01+xz*c02;let trace=xx+yy+zz;let matrixNorm2=xx*xx+yy*yy+zz*zz+2.*(xy*xy+xz*xz+yz*yz);let adjugateNorm2=c00*c00+c11*c11+c22*c22+2.*(c01*c01+c02*c02+c12*c12);if(!finite(determinant)||!finite(trace)||determinant<=1e-7*trace*trace*trace){if(publishSupportClosureVector(row,r,origin)){return;}vectorFail(INCOMPLETE,row);return;}let condition=sqrt(matrixNorm2*adjugateNorm2)/determinant;if(!finite(condition)||condition>1e5){if(publishSupportClosureVector(row,r,origin)){return;}vectorFail(INCOMPLETE,row);return;}let result=vec3f(c00*b.x+c01*b.y+c02*b.z,c01*b.x+c11*b.y+c12*b.z,c02*b.x+c12*b.y+c22*b.z)/determinant;var openScalars=0u;for(var axis=0u;axis<3u;axis+=1u){if(origin[axis]+r.size==p.dims[axis]&&(p.closedBoundaryMask&positiveBoundaryBit(axis))==0u){rows[row].padf=result[axis];openScalars+=1u;}}if(openScalars>1u||!finite(result.x)||!finite(result.y)||!finite(result.z)){if(publishSupportClosureVector(row,r,origin)){return;}vectorFail(INCOMPLETE,row);return;}let value=vec4f(result,1.);provisionalVelocities[row]=value;if(row>=transitionControl.support1End){rowVelocities[row]=value;}}
// S2/S3 rows are interpolation closure, not pressure unknowns.  Section 5's
// outside-liquid operation copies the closest-interface full face velocity;
// average those already accepted face carriers to publish one complete vector
// at every closure cell centre used by the cube/Delaunay interpolant.
// Aanjaneya et al. (2017), Section 5: liquid power cells retain the full
// centre velocity obtained by least-squares fitting their generalized-face
// normal components. Closest-point extension supplies vectors only to the outside-
// liquid support cells that have no corresponding power row.
@compute @workgroup_size(64)fn seedMappedPowerRowVelocity(@builtin(global_invocation_id)g:vec3u){let band=g.x;if(band>=transitionControl.endpointEnd||band>=arrayLength(&rows)||band>=arrayLength(&rowVelocities)||band>=arrayLength(&provisionalVelocities)){return;}let row=rows[band];if((row.flags&ROW_CORE)==0u){return;}if(row.globalRow>=p.powerRowCapacity||row.globalRow>=powerVelocityControl[2]||row.globalRow>=arrayLength(&powerRowVelocities)){vectorFail(INCOMPLETE,band);return;}let value=powerRowVelocities[row.globalRow];if(!velocityValid(value)){vectorFail(INCOMPLETE,band);return;}rowVelocities[band]=value;provisionalVelocities[band]=value;}
@compute @workgroup_size(256)fn finalizeFaceBandClosestPointDiagnostics(
 @builtin(local_invocation_index)lane:u32
){
 // One workgroup covers the exact published row and face prefixes. Each lane
 // reduces a deterministic strided subset, then contributes once to the
 // shared storage package; the control remains solely owned by lane zero.
 if(arrayLength(&publicationReductions)<23u){
  if(lane==0u){control.flags|=CAPACITY;control.firstError=min(control.firstError,0u);}
  return;
 }
 if(lane==0u){
  atomicStore(&publicationReductions[0],control.flags);
  atomicStore(&publicationReductions[1],control.firstError);
  for(var word=2u;word<=10u;word+=1u){atomicStore(&publicationReductions[word],0u);}
  atomicStore(&publicationReductions[11],INVALID);
  atomicStore(&publicationReductions[12],control.firstPhiFailure);
  atomicStore(&publicationReductions[13],control.firstClosestPointFailure);
  atomicStore(&publicationReductions[14],control.firstVectorFailure);
  for(var word=15u;word<=18u;word+=1u){atomicStore(&publicationReductions[word],0u);}
  for(var word=19u;word<=22u;word+=1u){atomicStore(&publicationReductions[word],INVALID);}
 }
 storageBarrier();workgroupBarrier();
 var flags=0u;var first=INVALID;
 var seeds=0u;var accepted=0u;var unresolved=0u;var coarseSamples=0u;
 var coarseFailures=0u;var extendedRows=0u;var closestFaces=0u;
 var closestFailures=0u;var liquidFailures=0u;
 var firstPhiSlot=INVALID;var firstPhi=INVALID;
 var firstClosest=INVALID;var firstVector=INVALID;
 var noOwner=0u;var supportOwner=0u;var noSimplex=0u;var missingVertex=0u;
 var firstNoOwner=INVALID;var firstSupportOwner=INVALID;
 var firstNoSimplex=INVALID;var firstMissingVertex=INVALID;
 let rowCount=min(transitionControl.endpointEnd,arrayLength(&rows));
 for(var row=lane;row<rowCount;row+=256u){
  let rowFlags=rows[row].flags;
  extendedRows+=select(0u,1u,(rowFlags&ROW_PHI_EXTENDED)!=0u);
  if((rowFlags&ROW_PHI)==0u){
   coarseFailures+=1u;flags|=OUTSIDE_FINE_BAND;
   firstPhi=min(firstPhi,rows[row].cell);first=min(first,rows[row].cell);
  }
  if(row<transitionControl.support3NodeEnd
    &&(row>=arrayLength(&provisionalVelocities)||!velocityValid(provisionalVelocities[row]))){
   flags|=INCOMPLETE;firstVector=min(firstVector,row);first=min(first,row);
  }
 }
 let faceCount=select(0u,liveFaceWorklist[0],arrayLength(&liveFaceWorklist)>0u);
 for(var item=lane;item<faceCount;item+=256u){
  let slot=liveFaceIndex(item);if(slot==INVALID||slot>=arrayLength(&faces)){continue;}
  let face=faces[slot];
  if((face.flags&PHI_DIAGNOSTIC)!=0u){
   coarseFailures+=1u;closestFailures+=1u;flags|=OUTSIDE_FINE_BAND;
   firstPhiSlot=min(firstPhiSlot,slot);firstPhi=min(firstPhi,face.globalFace);
   first=min(first,face.globalFace);
  }
  // Scalar phi closes the S0-S3 interpolation topology, including terminal
  // faces that are deliberately not physical velocity targets.
  if(!faceVelocityTarget(face)){continue;}
  if((face.flags&COARSE_PHI)!=0u){coarseSamples+=1u;}
  if((face.flags&FACE_VELOCITY_VALID)!=0u){
   accepted+=1u;if((face.flags&SEED)!=0u){seeds+=1u;}else{closestFaces+=1u;}
   continue;
  }
  unresolved+=1u;firstClosest=min(firstClosest,face.globalFace);first=min(first,face.globalFace);
  if(face.pad==CPT_BAD_FACE){closestFailures+=1u;flags|=BAD_FACE;continue;}
  if(face.pad==CPT_NO_CLOSEST_POINT){closestFailures+=1u;flags|=UNRESOLVED;continue;}
  if(face.pad==CPT_OWNER_ROW_MISMATCH){closestFailures+=1u;flags|=BAD_DIRECTORY;continue;}
  liquidFailures+=1u;flags|=UNRESOLVED;
  let reason=select(CPT_MISSING_VERTEX,face.pad,face.pad>=CPT_NO_OWNER&&face.pad<=CPT_MISSING_VERTEX);
  if(reason==CPT_NO_OWNER){noOwner+=1u;firstNoOwner=min(firstNoOwner,slot);}
  else if(reason==CPT_SUPPORT_OWNER){supportOwner+=1u;firstSupportOwner=min(firstSupportOwner,slot);}
  else if(reason==CPT_NO_SIMPLEX){noSimplex+=1u;firstNoSimplex=min(firstNoSimplex,slot);}
  else{missingVertex+=1u;firstMissingVertex=min(firstMissingVertex,slot);}
 }
 atomicOr(&publicationReductions[0],flags);atomicMin(&publicationReductions[1],first);
 atomicAdd(&publicationReductions[2],seeds);atomicAdd(&publicationReductions[3],accepted);
 atomicAdd(&publicationReductions[4],unresolved);atomicAdd(&publicationReductions[5],coarseSamples);
 atomicAdd(&publicationReductions[6],coarseFailures);atomicAdd(&publicationReductions[7],extendedRows);
 atomicAdd(&publicationReductions[8],closestFaces);atomicAdd(&publicationReductions[9],closestFailures);
 atomicAdd(&publicationReductions[10],liquidFailures);atomicMin(&publicationReductions[11],firstPhiSlot);
 atomicMin(&publicationReductions[12],firstPhi);atomicMin(&publicationReductions[13],firstClosest);
 atomicMin(&publicationReductions[14],firstVector);atomicAdd(&publicationReductions[15],noOwner);
 atomicAdd(&publicationReductions[16],supportOwner);atomicAdd(&publicationReductions[17],noSimplex);
 atomicAdd(&publicationReductions[18],missingVertex);atomicMin(&publicationReductions[19],firstNoOwner);
 atomicMin(&publicationReductions[20],firstSupportOwner);
 atomicMin(&publicationReductions[21],firstNoSimplex);
 atomicMin(&publicationReductions[22],firstMissingVertex);
 storageBarrier();workgroupBarrier();if(lane!=0u){return;}
 control.flags=atomicLoad(&publicationReductions[0]);
 control.firstError=atomicLoad(&publicationReductions[1]);
 control.seedCount=atomicLoad(&publicationReductions[2]);
 control.acceptedCount=atomicLoad(&publicationReductions[3]);
 control.unresolvedCount=atomicLoad(&publicationReductions[4]);
 control.coarsePhiSamples=atomicLoad(&publicationReductions[5]);
 control.coarsePhiFailures=atomicLoad(&publicationReductions[6]);
 control.bandPhiExtensions=atomicLoad(&publicationReductions[7]);
 control.closestPointFaces=atomicLoad(&publicationReductions[8]);
 control.closestPointFailures=atomicLoad(&publicationReductions[9]);
 control.liquidInterpolationFailures=atomicLoad(&publicationReductions[10]);
 control.firstPhiFailureSlot=atomicLoad(&publicationReductions[11]);
 control.firstPhiFailure=atomicLoad(&publicationReductions[12]);
 control.firstClosestPointFailure=atomicLoad(&publicationReductions[13]);
 control.firstVectorFailure=atomicLoad(&publicationReductions[14]);
 control.cptNoOwnerFailures=atomicLoad(&publicationReductions[15]);
 control.cptSupportOwnerFailures=atomicLoad(&publicationReductions[16]);
 control.cptNoContainingSimplexFailures=atomicLoad(&publicationReductions[17]);
 control.cptMissingLiquidVertexFailures=atomicLoad(&publicationReductions[18]);
 control.firstCptNoOwnerFailure=atomicLoad(&publicationReductions[19]);
 control.firstCptSupportOwnerFailure=atomicLoad(&publicationReductions[20]);
 control.firstCptNoSimplexFailure=atomicLoad(&publicationReductions[21]);
 control.firstCptMissingVertexFailure=atomicLoad(&publicationReductions[22]);
}
fn pointFail(code:u32,index:u32){pointControl.flags|=code;pointControl.firstError=min(pointControl.firstError,index);}
fn storePointError(row:u32,code:u32){if(row<arrayLength(&pointStatus)){pointStatus[row]=code;}}
struct BandPowerGeometry{neighborCenter:vec3f,neighborSize:f32,normal:vec3f,inverseDistance:f32,centroid:vec3f,area:f32}struct BandPowerPolygon{vertices:array<vec3f,16>,count:u32}
fn transientFail(code:u32,index:u32){transientPowerControl.flags|=code;transientPowerControl.firstError=min(transientPowerControl.firstError,index);}
fn storeTransientRowFailure(code:u32,row:u32){if(row<arrayLength(&transientPowerRows)){let work=transientPowerRows[row];transientPowerRows[row]=PowerRowWork(TRANSIENT_ROW_ERROR|((code&255u)<<16u)|(work.faceCount&65535u),work.incidenceCount,work.faceOffset,work.incidenceOffset);}}
fn transientFaceFail(code:u32,faceIndex:u32){if(faceIndex<arrayLength(&transientPowerFaces)){var face=transientPowerFaces[faceIndex];face.flags|=TRANSIENT_FACE_ERROR;face.pad=(face.pad&0xffffff00u)|(code&255u);transientPowerFaces[faceIndex]=face;}}
fn transientFailGeometry(code:u32,index:u32,slot:u32,geometry:BandPowerGeometry,neighbor:u32,stage:u32){let faceIndex=index*POINT_MAX_FACES+slot;if(faceIndex<arrayLength(&transientPowerFaces)){transientPowerFaces[faceIndex]=TransientPowerFace(index,neighbor,TRANSIENT_FACE_ERROR,(code&255u)|((stage&0xffffffu)<<8u),vec4f(geometry.neighborCenter,geometry.neighborSize),vec4f(geometry.centroid,1.),0.,geometry.area,geometry.inverseDistance,0.);}storeTransientRowFailure(code,index);}
fn transientFailGeometryStage(code:u32,index:u32,slot:u32,geometry:BandPowerGeometry,neighbor:u32,stage:u32){transientFailGeometry(code,index,slot,geometry,neighbor,stage);}
fn transientRowCenter(row:u32)->vec3f{return vec3f(coord(rows[row].cell))+.5*f32(rows[row].size);}
fn transientCatalogGeometry(row:u32,slot:u32)->BandPowerGeometry{let r=rows[row];let metric=metrics[row];let entry=catalogEntries[metric.topology];let f=catalogFaces[entry.firstFace+slot];let center=transientRowCenter(row);let size=f32(r.size);return BandPowerGeometry(center+size*inversePowerTransform(f.neighborOffsetSize.xyz,metric.transformFlags&63u),size*f.neighborOffsetSize.w,inversePowerTransform(f.normalInverseDistance.xyz,metric.transformFlags&63u),f.normalInverseDistance.w/size,center+size*inversePowerTransform(f.areaCentroid.yzw,metric.transformFlags&63u),size*size*f.areaCentroid.x);}
fn transientWorldBoundaryBit(g:BandPowerGeometry)->u32{if(g.neighborSize==0.){if(g.normal.x<-.9999){return 1u;}if(g.normal.y<-.9999){return 2u;}if(g.normal.z<-.9999){return 4u;}if(g.normal.z>.9999){return 8u;}if(g.normal.y>.9999){return 16u;}if(g.normal.x>.9999){return 32u;}return 0u;}if(g.neighborCenter.x<0.){return 1u;}if(g.neighborCenter.y<0.){return 2u;}if(g.neighborCenter.z<0.){return 4u;}if(g.neighborCenter.z>f32(p.dims.z)){return 8u;}if(g.neighborCenter.y>f32(p.dims.y)){return 16u;}if(g.neighborCenter.x>f32(p.dims.x)){return 32u;}return 0u;}
fn validTransientGeometry(g:BandPowerGeometry,world:bool)->bool{let n2=dot(g.normal,g.normal);return finite(g.neighborSize)&&select(g.neighborSize>0.,g.neighborSize>=0.,world)&&finite(g.neighborCenter.x)&&finite(g.neighborCenter.y)&&finite(g.neighborCenter.z)&&finite(n2)&&abs(n2-1.)<=4e-4&&finite(g.inverseDistance)&&g.inverseDistance>0.&&finite(g.area)&&g.area>0.&&finite(g.centroid.x)&&finite(g.centroid.y)&&finite(g.centroid.z);}
fn transientNeighborMeasurement(geometry:BandPowerGeometry)->vec2u{if(geometry.neighborSize<=0.){return vec2u(INVALID,1u);}let size=u32(round(geometry.neighborSize));let originValue=geometry.neighborCenter-.5*f32(size);let origin=round(originValue);if(size==0u||abs(geometry.neighborSize-f32(size))>2e-4||any(abs(originValue-origin)>vec3f(2e-4))||any(origin<vec3f(0.))||any(origin+vec3f(f32(size))>vec3f(p.dims))){return vec2u(INVALID,1u);}let expectedCell=cell(vec3u(origin));let row=rowOfIdentity(expectedCell,size);if(row==INVALID){return vec2u(INVALID,2u);}if(row>=arrayLength(&rows)){return vec2u(row,3u);}return vec2u(row,0u);}
fn transientNeighbor(geometry:BandPowerGeometry)->u32{let measured=transientNeighborMeasurement(geometry);return select(measured.x,INVALID,measured.y!=0u);}
fn transientReciprocalSlot(row:u32,neighbor:u32)->u32{if(neighbor>=arrayLength(&rows)||neighbor>=arrayLength(&metrics)){return INVALID;}let metric=metrics[neighbor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&catalogEntries)){return INVALID;}let entry=catalogEntries[metric.topology];if(entry.firstFace>arrayLength(&catalogFaces)||entry.faceCount>arrayLength(&catalogFaces)-entry.firstFace){return INVALID;}let wantedCenter=transientRowCenter(row);let wantedSize=f32(rows[row].size);let tolerance=max(1e-5,wantedSize*2e-5);for(var slot=0u;slot<entry.faceCount;slot+=1u){let g=transientCatalogGeometry(neighbor,slot);if(g.neighborSize>0.&&abs(g.neighborSize-wantedSize)<=tolerance&&all(abs(g.neighborCenter-wantedCenter)<=vec3f(tolerance))){return slot;}}return INVALID;}
fn clipTransientPolygon(value:BandPowerPolygon,plane:BandPowerGeometry,epsilon:f32)->BandPowerPolygon{if(value.count<3u){return value;}var clipped:array<vec3f,16>;var output=0u;var previous=value.vertices[value.count-1u];var previousSide=dot(plane.normal,previous-plane.centroid);var previousInside=previousSide<=epsilon;for(var i=0u;i<value.count;i+=1u){let current=value.vertices[i];let currentSide=dot(plane.normal,current-plane.centroid);let currentInside=currentSide<=epsilon;if(currentInside!=previousInside&&output<16u){let denominator=previousSide-currentSide;let t=select(.5,clamp(previousSide/denominator,0.,1.),abs(denominator)>1e-12);clipped[output]=mix(previous,current,t);output+=1u;}if(currentInside&&output<16u){clipped[output]=current;output+=1u;}previous=current;previousSide=currentSide;previousInside=currentInside;}return BandPowerPolygon(clipped,output);}
fn clipTransientByCell(value:BandPowerPolygon,row:u32,epsilon:f32)->BandPowerPolygon{var polygon=value;let metric=metrics[row];let entry=catalogEntries[metric.topology];for(var slot=0u;slot<entry.faceCount&&polygon.count>=3u;slot+=1u){polygon=clipTransientPolygon(polygon,transientCatalogGeometry(row,slot),epsilon);}return polygon;}
fn transientFacePolygon(row:u32,slot:u32,neighbor:u32,reverseSlot:u32)->BandPowerPolygon{let geometry=transientCatalogGeometry(row,slot);let normal=normalize(geometry.normal);let reference=select(vec3f(0.,1.,0.),vec3f(1.,0.,0.),abs(normal.x)<.75);let tangent=normalize(cross(reference,normal));let bitangent=cross(normal,tangent);var neighborScale=f32(rows[row].size);var center=geometry.centroid;if(neighbor!=INVALID){neighborScale=f32(rows[neighbor].size);let reverseGeometry=transientCatalogGeometry(neighbor,reverseSlot);center=.5*(geometry.centroid+reverseGeometry.centroid);}let scale=max(f32(rows[row].size),neighborScale);let extent=8.*scale;var vertices:array<vec3f,16>;vertices[0]=center-extent*tangent-extent*bitangent;vertices[1]=center+extent*tangent-extent*bitangent;vertices[2]=center+extent*tangent+extent*bitangent;vertices[3]=center-extent*tangent+extent*bitangent;let epsilon=max(1e-6,1e-5*scale);var polygon=clipTransientByCell(BandPowerPolygon(vertices,4u),row,epsilon);if(neighbor!=INVALID){polygon=clipTransientByCell(polygon,neighbor,epsilon);}return polygon;}
fn exactTransientGeometry(row:u32,slot:u32,neighbor:u32,reverseSlot:u32)->BandPowerGeometry{var geometry=transientCatalogGeometry(row,slot);let polygon=transientFacePolygon(row,slot,neighbor,reverseSlot);if(polygon.count<3u){geometry.area=0.;return geometry;}var area=0.;var centroid=vec3f(0.);for(var i=1u;i+1u<polygon.count;i+=1u){let triangle=.5*max(0.,dot(cross(polygon.vertices[i]-polygon.vertices[0],polygon.vertices[i+1u]-polygon.vertices[0]),geometry.normal));area+=triangle;centroid+=triangle*(polygon.vertices[0]+polygon.vertices[i]+polygon.vertices[i+1u])/3.;}geometry.area=area;if(area>1e-12){geometry.centroid=centroid/area;}return geometry;}
fn committedBandGenerationValid()->bool{let mask=0x3fffffffu;let fine=p.generation&mask;let band=control.generation&mask;let power=p.powerGeneration&mask;let predecessor=(fine+mask)&mask;return band==fine||(power==fine&&band==predecessor);}
@compute @workgroup_size(1)fn prepareTransientBandPowerGraph(){transientPowerControl.flags=0u;transientPowerControl.firstError=INVALID;transientPowerControl.rowCount=transitionControl.support1End;transientPowerControl.faceSlots=transientPowerControl.rowCount*POINT_MAX_FACES;transientPowerControl.emitted=0u;transientPowerControl.sampled=0u;transientPowerControl.validated=0u;transientPowerControl.generation=p.generation;transientPowerControl.valid=0u;transientPowerControl.p0=INVALID;transientPowerControl.p1=INVALID;transientPowerControl.p2=INVALID;transientPowerControl.p3=INVALID;transientPowerControl.p4=INVALID;transientPowerControl.p5=INVALID;transientPowerControl.p6=INVALID;writeSupportDispatch(54u,0u);if(control.valid!=VALID||control.flags!=0u||!committedBandGenerationValid()||transitionControl.ready!=VALID||transitionControl.transferReady!=VALID||transitionControl.hierarchyReady!=VALID||transitionControl.flags!=0u){transientFail(POINT_SOURCE,0u);return;}if(transitionControl.support1End>transitionControl.support2End||transientPowerControl.rowCount>p.rowCapacity||transientPowerControl.faceSlots>arrayLength(&transientPowerFaces)||transientPowerControl.faceSlots>arrayLength(&transientPowerIncidences)||transientPowerControl.rowCount>arrayLength(&transientPowerRows)){transientFail(POINT_CAPACITY,0u);return;}writeSupportDispatch(54u,transientPowerControl.rowCount);}
@compute @workgroup_size(64)fn emitTransientBandPowerGraph(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=transientPowerControl.rowCount||row>=arrayLength(&rows)||row>=arrayLength(&metrics)||transientPowerControl.flags!=0u){return;}let base=row*POINT_MAX_FACES;if(row>=arrayLength(&transientPowerRows)||base>arrayLength(&transientPowerFaces)||POINT_MAX_FACES>arrayLength(&transientPowerFaces)-base){storeTransientRowFailure(POINT_CAPACITY,row);return;}transientPowerRows[row]=PowerRowWork(0u,0u,base,base);for(var retired=0u;retired<POINT_MAX_FACES;retired+=1u){transientPowerFaces[base+retired].flags=0u;}let metric=metrics[row];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&catalogEntries)){storeTransientRowFailure(POINT_SOURCE,row);return;}let entry=catalogEntries[metric.topology];if(entry.faceCount==0u||entry.faceCount>POINT_MAX_FACES||entry.firstFace>arrayLength(&catalogFaces)||entry.faceCount>arrayLength(&catalogFaces)-entry.firstFace){storeTransientRowFailure(POINT_CAPACITY,row);return;}transientPowerRows[row]=PowerRowWork(entry.faceCount,entry.faceCount,base,base);for(var slot=0u;slot<entry.faceCount;slot+=1u){let geometry=transientCatalogGeometry(row,slot);let declared=(metric.transformFlags>>8u)&63u;let boundaryBit=transientWorldBoundaryBit(geometry);let world=geometry.neighborSize==0.||(boundaryBit&declared)!=0u;if(!validTransientGeometry(geometry,world)){transientFailGeometryStage(POINT_FACE,row,slot,geometry,INVALID,1u);return;}var neighbor=INVALID;var reverseSlot=INVALID;if(!world){let measured=transientNeighborMeasurement(geometry);neighbor=measured.x;if(measured.y!=0u||neighbor>=transitionControl.support2End){transientFailGeometryStage(POINT_SAMPLE,row,slot,geometry,neighbor,2u|(measured.y<<8u));return;}reverseSlot=transientReciprocalSlot(row,neighbor);if(reverseSlot==INVALID){transientFailGeometryStage(POINT_FACE,row,slot,geometry,neighbor,3u);return;}let reverse=transientCatalogGeometry(neighbor,reverseSlot);if(dot(geometry.normal,reverse.normal)>-.999||abs(geometry.inverseDistance-reverse.inverseDistance)>max(1e-5,geometry.inverseDistance*2e-4)){transientFailGeometryStage(POINT_FACE,row,slot,geometry,neighbor,4u);return;}if(row>neighbor){continue;}}let exact=exactTransientGeometry(row,slot,neighbor,reverseSlot);if(!validTransientGeometry(exact,world)){transientFailGeometryStage(POINT_FACE,row,slot,exact,neighbor,5u);return;}let faceIndex=base+slot;var plane=0u;if(world){plane=boundaryBit&declared;if(plane==0u){transientFailGeometryStage(POINT_FACE,row,slot,exact,neighbor,7u);return;}}transientPowerFaces[faceIndex]=TransientPowerFace(row,neighbor,1u|(plane<<8u),0u,vec4f(exact.normal,0.),vec4f(exact.centroid,1.),0.,exact.area,exact.inverseDistance,0.);transientPowerIncidences[base+slot]=PowerIncidence(faceIndex,1);if(neighbor!=INVALID){transientPowerIncidences[neighbor*POINT_MAX_FACES+reverseSlot]=PowerIncidence(faceIndex,-1);}}}
fn provisionalS1ToS3Carrier(s1Index:u32,s3Index:u32,s1Reason:u32,s3Reason:u32)->vec4f{if(s1Reason!=7u||s3Reason!=1u||s1Index==INVALID||s3Index==INVALID||s1Index>=arrayLength(&rows)||s3Index>=arrayLength(&rows)){return invalidPointVector(12u);}let s1=rows[s1Index];let s3=rows[s3Index];if((s1.flags&ROW_SUPPORT1)==0u||(s1.flags&(ROW_SUPPORT3_NODE|ROW_SUPPORT3_ENDPOINT))!=0u||(s3.flags&ROW_SUPPORT3_NODE)==0u){return invalidPointVector(12u);}return provisionalCellVector(coord(s1.cell),s1.size);}
@compute @workgroup_size(64)fn sampleTransientBandPowerFaces(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=transientPowerControl.rowCount||transientPowerControl.flags!=0u){return;}let base=row*POINT_MAX_FACES;for(var slot=0u;slot<POINT_MAX_FACES;slot+=1u){let faceIndex=base+slot;if(faceIndex>=transientPowerControl.faceSlots||faceIndex>=arrayLength(&transientPowerFaces)){storeTransientRowFailure(POINT_CAPACITY,row);return;}var face=transientPowerFaces[faceIndex];if((face.flags&1u)==0u){continue;}let plane=(face.flags>>8u)&63u;var scalar=0.;if(plane!=0u){if((p.closedBoundaryMask&plane)==0u){if(plane!=positiveBoundaryBit(1u)||face.negativeRow>=arrayLength(&rows)||!finite(rows[face.negativeRow].padf)){face.pad=1u<<8u;transientPowerFaces[faceIndex]=face;transientFaceFail(POINT_FACE,faceIndex);return;}scalar=face.normal.y*rows[face.negativeRow].padf;}}else{let negative=marchedCentroidVector(face.negativeRow,face.centroid.xyz);let positive=marchedCentroidVector(face.positiveRow,face.centroid.xyz);let negativeValid=velocityValid(negative);let positiveValid=velocityValid(positive);var full=vec3f(0.);if(negativeValid&&positiveValid){full=.5*(negative.xyz+positive.xyz);}else if(negativeValid){full=negative.xyz;}else if(positiveValid){full=positive.xyz;}else{let negativeReason=u32(round(max(0.,-negative.w)));let positiveReason=u32(round(max(0.,-positive.w)));var carrier=provisionalS1ToS3Carrier(face.negativeRow,face.positiveRow,negativeReason,positiveReason);if(!velocityValid(carrier)){carrier=provisionalS1ToS3Carrier(face.positiveRow,face.negativeRow,positiveReason,negativeReason);}if(velocityValid(carrier)){full=carrier.xyz;}else{face.pad=(2u<<8u)|((negativeReason&255u)<<16u)|((positiveReason&255u)<<24u);transientPowerFaces[faceIndex]=face;transientFaceFail(POINT_SAMPLE,faceIndex);return;}}scalar=dot(full,face.normal.xyz);}if(!finite(scalar)){face.pad=3u<<8u;transientPowerFaces[faceIndex]=face;transientFaceFail(POINT_NONFINITE,faceIndex);return;}face.normalVelocity=scalar;face.flags|=2u;transientPowerFaces[faceIndex]=face;}}
@compute @workgroup_size(64)fn validateTransientBandPowerGraph(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=transientPowerControl.rowCount||row>=arrayLength(&transientPowerRows)||transientPowerControl.flags!=0u){return;}let work=transientPowerRows[row];if((work.faceCount&TRANSIENT_ROW_ERROR)!=0u){return;}if(work.incidenceCount==0u||work.incidenceCount>POINT_MAX_FACES||work.incidenceOffset>arrayLength(&transientPowerIncidences)||work.incidenceCount>arrayLength(&transientPowerIncidences)-work.incidenceOffset){storeTransientRowFailure(POINT_CAPACITY,row);return;}for(var local=0u;local<work.incidenceCount;local+=1u){let incidence=transientPowerIncidences[work.incidenceOffset+local];if((incidence.sign!=1&&incidence.sign!=-1)||incidence.face>=arrayLength(&transientPowerFaces)){storeTransientRowFailure(POINT_FACE,row);return;}let face=transientPowerFaces[incidence.face];let signMatches=select(face.positiveRow==row,face.negativeRow==row,incidence.sign==1);if((face.flags&3u)!=3u||!signMatches){storeTransientRowFailure(POINT_FACE,row);return;}}transientPowerRows[row].faceCount|=TRANSIENT_ROW_VALIDATED;}
@compute @workgroup_size(1)fn prepareBandPointField(){for(var i=0u;i<8u;i+=1u){atomicStore(&publicationReductions[i],select(0u,INVALID,i==3u||i==4u));}transientPowerControl.valid=0u;pointControl.flags=0u;pointControl.firstError=INVALID;pointControl.rowCount=transitionControl.support1End;pointControl.generation=p.generation;pointControl.solved=0u;pointControl.valid=0u;pointControl.wallContributions=0u;pointControl.pad=transitionControl.coreEnd;writeSupportDispatch(54u,0u);}
@compute @workgroup_size(64)fn reduceBandPointField(@builtin(global_invocation_id)g:vec3u){let item=g.x;if(item<min(transientPowerControl.faceSlots,arrayLength(&transientPowerFaces))){let face=transientPowerFaces[item];let flags=face.flags;if((flags&1u)!=0u){atomicAdd(&publicationReductions[0],1u);}if((flags&3u)==3u){atomicAdd(&publicationReductions[1],1u);}if((flags&TRANSIENT_FACE_ERROR)!=0u){atomicOr(&publicationReductions[2],face.pad&255u);atomicMin(&publicationReductions[3],select(face.negativeRow,TRANSIENT_FACE_FAILURE|item,(flags&1u)!=0u));atomicMin(&publicationReductions[4],item);}}if(item<min(transientPowerControl.rowCount,arrayLength(&transientPowerRows))){let marker=transientPowerRows[item].faceCount;if((marker&TRANSIENT_ROW_ERROR)!=0u){atomicOr(&publicationReductions[2],(marker>>16u)&255u);atomicMin(&publicationReductions[3],item);}if((marker&TRANSIENT_ROW_VALIDATED)!=0u){atomicAdd(&publicationReductions[5],1u);}}}
@compute @workgroup_size(1)fn finalizeBandPointField(){let emittedCount=atomicLoad(&publicationReductions[0]);let sampledCount=atomicLoad(&publicationReductions[1]);let failureFlags=atomicLoad(&publicationReductions[2]);let firstFailure=min(transientPowerControl.firstError,atomicLoad(&publicationReductions[3]));let diagnosticIndex=atomicLoad(&publicationReductions[4]);let validatedRows=atomicLoad(&publicationReductions[5]);let combinedFlags=transientPowerControl.flags|failureFlags;transientPowerControl.flags=combinedFlags;transientPowerControl.firstError=firstFailure;transientPowerControl.emitted=emittedCount;transientPowerControl.sampled=sampledCount;transientPowerControl.validated=validatedRows;if(diagnosticIndex!=INVALID&&diagnosticIndex<arrayLength(&transientPowerFaces)){let failedFace=transientPowerFaces[diagnosticIndex];transientPowerControl.p0=diagnosticIndex%POINT_MAX_FACES;transientPowerControl.p1=bitcast<u32>(failedFace.normal.x);transientPowerControl.p2=bitcast<u32>(failedFace.normal.y);transientPowerControl.p3=bitcast<u32>(failedFace.normal.z);transientPowerControl.p4=bitcast<u32>(failedFace.normal.w);transientPowerControl.p5=failedFace.positiveRow;transientPowerControl.p6=failedFace.pad>>8u;}let readyForPointField=combinedFlags==0u&&transientPowerControl.generation==p.generation&&emittedCount>0u&&sampledCount==emittedCount&&validatedRows==transientPowerControl.rowCount;transientPowerControl.valid=select(0u,VALID,readyForPointField);if(control.valid!=VALID||control.flags!=0u||!committedBandGenerationValid()||transitionControl.ready!=VALID||transitionControl.transferReady!=VALID||transitionControl.hierarchyReady!=VALID||transitionControl.flags!=0u||transitionControl.support1End>transitionControl.support2End||transientPowerControl.valid!=VALID||combinedFlags!=0u||transientPowerControl.generation!=p.generation||transientPowerControl.rowCount!=pointControl.rowCount){pointFail(POINT_SOURCE,0u);return;}writeSupportDispatch(54u,pointControl.rowCount);}
@compute @workgroup_size(64)fn reconstructBandTransientPowerPointField(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=pointControl.rowCount){return;}if(row>=arrayLength(&pointStatus)||row>=arrayLength(&rowVelocities)){pointFail(POINT_CAPACITY,row);return;}pointStatus[row]=0u;if(transientPowerControl.valid!=VALID||transientPowerControl.generation!=p.generation||row>=transientPowerControl.rowCount||row>=arrayLength(&transientPowerRows)){storePointError(row,POINT_SOURCE);return;}let work=transientPowerRows[row];if(work.incidenceCount==0u||work.incidenceCount>POINT_MAX_FACES||work.incidenceOffset>arrayLength(&transientPowerIncidences)||work.incidenceCount>arrayLength(&transientPowerIncidences)-work.incidenceOffset){storePointError(row,POINT_CAPACITY);return;}var ls=BandLS(vec4f(0.),vec4f(0.),vec4f(0.));for(var local=0u;local<work.incidenceCount;local+=1u){let item=transientPowerIncidences[work.incidenceOffset+local];if((item.sign!=1&&item.sign!=-1)||item.face>=arrayLength(&transientPowerFaces)){storePointError(row,POINT_FACE);return;}let face=transientPowerFaces[item.face];let signMatches=select(face.positiveRow==row,face.negativeRow==row,item.sign==1);if((face.flags&3u)!=3u||!signMatches){storePointError(row,POINT_FACE);return;}let orientation=f32(item.sign);let normal=orientation*face.normal.xyz;let u=orientation*face.normalVelocity;let n2=dot(normal,normal);let weight=face.area;if(!finite(normal.x)||!finite(normal.y)||!finite(normal.z)||!finite(n2)||abs(n2-1.)>4e-4||!finite(u)||!finite(weight)||weight<=0.){storePointError(row,select(POINT_FACE,POINT_NORMAL,!finite(n2)||abs(n2-1.)>4e-4));return;}ls.a0.x+=weight*normal.x*normal.x;ls.a0.y+=weight*normal.x*normal.y;ls.a0.z+=weight*normal.x*normal.z;ls.a0.w+=weight*normal.y*normal.y;ls.a1.x+=weight*normal.y*normal.z;ls.a1.y+=weight*normal.z*normal.z;ls.a1.z+=weight*normal.x*u;ls.a1.w+=weight*normal.y*u;ls.a2.x+=weight*normal.z*u;}let xx=ls.a0.x;let xy=ls.a0.y;let xz=ls.a0.z;let yy=ls.a0.w;let yz=ls.a1.x;let zz=ls.a1.y;let b=vec3f(ls.a1.z,ls.a1.w,ls.a2.x);let c00=yy*zz-yz*yz;let c01=xz*yz-xy*zz;let c02=xy*yz-xz*yy;let c11=xx*zz-xz*xz;let c12=xy*xz-xx*yz;let c22=xx*yy-xy*xy;let determinant=xx*c00+xy*c01+xz*c02;let trace=xx+yy+zz;let matrixNorm2=xx*xx+yy*yy+zz*zz+2.*(xy*xy+xz*xz+yz*yz);let adjugateNorm2=c00*c00+c11*c11+c22*c22+2.*(c01*c01+c02*c02+c12*c12);if(!finite(determinant)||!finite(trace)||determinant<=1e-7*trace*trace*trace){storePointError(row,POINT_SINGULAR);return;}let condition=sqrt(matrixNorm2*adjugateNorm2)/determinant;if(!finite(condition)||condition>1e5){storePointError(row,POINT_CONDITION);return;}let velocity=vec3f(c00*b.x+c01*b.y+c02*b.z,c01*b.x+c11*b.y+c12*b.z,c02*b.x+c12*b.y+c22*b.z)/determinant;if(!finite(velocity.x)||!finite(velocity.y)||!finite(velocity.z)){storePointError(row,POINT_NONFINITE);return;}rowVelocities[row]=vec4f(velocity,1.);}
var<workgroup> pointReduceSolved:array<u32,256>;var<workgroup> pointReduceFlags:array<u32,256>;var<workgroup> pointReduceFirst:array<u32,256>;
@compute @workgroup_size(256)fn publishBandPointField(@builtin(local_invocation_index)lid:u32){var flags=0u;var first=INVALID;var solved=0u;for(var row=lid;row<pointControl.rowCount;row+=256u){if(row>=arrayLength(&pointStatus)){flags|=POINT_CAPACITY;first=min(first,row);continue;}let status=pointStatus[row];if(status==0u){solved+=1u;}else{flags|=status;first=min(first,row);}}pointReduceSolved[lid]=solved;pointReduceFlags[lid]=flags;pointReduceFirst[lid]=first;workgroupBarrier();for(var width=128u;width>0u;width>>=1u){if(lid<width){pointReduceSolved[lid]+=pointReduceSolved[lid+width];pointReduceFlags[lid]|=pointReduceFlags[lid+width];pointReduceFirst[lid]=min(pointReduceFirst[lid],pointReduceFirst[lid+width]);}workgroupBarrier();}if(lid==0u){pointControl.flags|=pointReduceFlags[0];pointControl.firstError=min(pointControl.firstError,pointReduceFirst[0]);pointControl.solved=pointReduceSolved[0];pointControl.valid=select(0u,VALID,pointControl.flags==0u&&pointReduceSolved[0]==pointControl.rowCount&&pointControl.generation==p.generation&&transientPowerControl.valid==VALID&&transientPowerControl.flags==0u&&transientPowerControl.generation==p.generation&&transientPowerControl.rowCount==pointControl.rowCount);}}
// Singleton preparation/commit may publish directly to the control. Parallel
// face stages own one scratch word each and publish failures there; the terminal
// singleton reduces those records in stable face order.
fn powerFail(code:u32,index:u32){powerPublication.flags|=code;powerPublication.firstError=min(powerPublication.firstError,index);}
fn storePowerFaceFailure(index:u32,code:u32){if(index<arrayLength(&powerVelocityScratch)){powerVelocityScratch[index]=vec4u(0u,POWER_FAILURE_MARKER|code,0u,0u);}}
@compute @workgroup_size(1)fn preparePowerPublication(){powerPublication.flags=0u;powerPublication.firstError=INVALID;powerPublication.faceCount=0u;powerPublication.targetCount=0u;powerPublication.interpolatedCount=0u;powerPublication.committedCount=0u;powerPublication.fineGeneration=p.generation;powerPublication.powerGeneration=p.powerGeneration;powerPublication.valid=0u;writeSupportDispatch(54u,0u);if(arrayLength(&powerFaceControl)<9u){powerFail(POWER_CAPACITY,0u);return;}powerPublication.faceCount=powerFaceControl[1];if(control.valid!=VALID||control.flags!=0u||!committedBandGenerationValid()||transitionControl.ready!=VALID||transitionControl.transferReady!=VALID||transitionControl.flags!=0u||pointControl.valid!=VALID||pointControl.flags!=0u||pointControl.generation!=p.generation||powerFaceControl[3]!=0u||powerFaceControl[8]!=VALID||powerFaceControl[7]!=p.powerGeneration){powerFail(POWER_SOURCE,0u);return;}if(powerFaceControl[0]>p.powerRowCapacity||powerFaceControl[1]>arrayLength(&powerFaces)||powerFaceControl[1]>arrayLength(&powerFaceNormals)||powerFaceControl[1]>arrayLength(&powerFaceCentroids)||powerFaceControl[1]>arrayLength(&powerVelocityScratch)){powerFail(POWER_CAPACITY,powerFaceControl[1]);return;}writeSupportDispatch(54u,powerPublication.faceCount);}
// Paper Section 5 extrapolation is one-sided: it assigns velocities outside
// the liquid and must not overwrite a projected face incident to liquid. A
// band row is wet when its extended centre phi is negative — the same current
// centre-sign convention the pressure solve uses. Preserve both liquid-liquid
// and liquid-air/interface faces; only wholly air-side targets consume the
// extrapolated full-vector field. This also prevents cell-centre reconstruction
// from re-smoothing the pressure-projected free-surface normal velocity.
fn bandRowIsWet(band:u32)->bool{return band!=INVALID&&band<arrayLength(&rows)&&(rows[band].flags&ROW_PHI)!=0u&&finite(rows[band].representativePhi)&&rows[band].representativePhi<0.0;}
@compute @workgroup_size(64)fn mapPowerFaceBands(@builtin(global_invocation_id)g:vec3u){let index=g.x;if(index>=powerPublication.faceCount||powerPublication.flags!=0u){return;}if(index>=arrayLength(&powerFaces)||index>=arrayLength(&powerVelocityScratch)){storePowerFaceFailure(index,POWER_CAPACITY);return;}let powerFace=powerFaces[index];let negativeBand=bandForGlobalRow(powerFace.negativeRow);let positiveBand=bandForGlobalRow(powerFace.positiveRow);powerVelocityScratch[index]=vec4u(0u,0u,negativeBand,positiveBand);}
fn provisionalCellVector(origin:vec3u,size:u32)->vec4f{if(size==0u||any(origin+vec3u(size)>p.dims)){return vec4f(0);}let band=rowOfIdentity(cell(origin),size);if(band==INVALID||band>=arrayLength(&rows)||band>=arrayLength(&provisionalVelocities)){return vec4f(0);}let row=rows[band];let value=provisionalVelocities[band];if((row.flags&ROW_SUPPORT3_ENDPOINT)!=0u||!velocityValid(value)){return vec4f(0);}return value;}
fn finalCellVector(origin:vec3u,size:u32)->vec4f{if(size==0u||any(origin+vec3u(size)>p.dims)){return vec4f(0);}let band=rowOfIdentity(cell(origin),size);if(band==INVALID||band>=arrayLength(&rows)||band>=arrayLength(&rowVelocities)){return vec4f(0);}let row=rows[band];let value=rowVelocities[band];if((row.flags&(ROW_SUPPORT2|ROW_SUPPORT3_NODE|ROW_SUPPORT3_ENDPOINT))!=0u||!velocityValid(value)){return vec4f(0);}return value;}
fn supportCellVector(origin:vec3u,size:u32)->vec4f{if(size==0u||any(origin+vec3u(size)>p.dims)){return vec4f(0);}let band=rowOfIdentity(cell(origin),size);if(band==INVALID||band>=arrayLength(&rows)||band>=arrayLength(&rowVelocities)){return vec4f(0);}let row=rows[band];let value=rowVelocities[band];if((row.flags&ROW_SUPPORT3_ENDPOINT)!=0u||!velocityValid(value)){return vec4f(0);}return value;}
fn supportSelectorCellVector(origin:vec3u,size:u32)->vec4f{if(size==0u||any(origin+vec3u(size)>p.dims)){return vec4f(0);}let band=rowOfIdentity(cell(origin),size);if(band==INVALID||band>=arrayLength(&rows)||band>=arrayLength(&rowVelocities)){return vec4f(0);}let row=rows[band];let value=rowVelocities[band];if((row.flags&ROW_SUPPORT3_ENDPOINT)!=0u||!velocityValid(value)){return vec4f(0);}return value;}
fn provisionalContainingVector(q:vec3u)->vec4f{var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);if(all(origin+vec3u(size)<=p.dims)){let value=provisionalCellVector(origin,size);if(velocityValid(value)){return value;}}if(size>=p.maximumLeaf){break;}size<<=1u;}return vec4f(0);}
fn finalContainingVector(q:vec3u)->vec4f{var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);if(all(origin+vec3u(size)<=p.dims)){let value=finalCellVector(origin,size);if(velocityValid(value)){return value;}}if(size>=p.maximumLeaf){break;}size<<=1u;}return vec4f(0);}
fn provisionalSignedVector(origin:vec3i,size:u32)->vec4f{let reflected=velocityExtendedOrigin(origin,size);if(reflected.w<0){return vec4f(0);}var value=provisionalCellVector(vec3u(reflected.xyz),size);let outside=any(origin<vec3i(0))||any(origin+vec3i(i32(size))>vec3i(p.dims));if(!velocityValid(value)&&outside){value=provisionalContainingVector(vec3u(reflected.xyz));}return reflectComponents(value,reflected.w);}
fn reflectedBoundarySelectorVector(resolved:vec3u)->vec4f{var size=1u;loop{let origin=(resolved/vec3u(size))*vec3u(size);if(all(origin+vec3u(size)<=p.dims)){let value=supportSelectorCellVector(origin,size);if(velocityValid(value)){return value;}}if(size>=p.maximumLeaf){break;}size<<=1u;}return vec4f(0.);}
fn finalSignedVector(origin:vec3i,size:u32)->vec4f{let reflected=velocityExtendedOrigin(origin,size);if(reflected.w<0){return vec4f(0);}let resolved=vec3u(reflected.xyz);let outside=any(origin<vec3i(0))||any(origin+vec3i(i32(size))>vec3i(p.dims));let value=select(supportSelectorCellVector(resolved,size),reflectedBoundarySelectorVector(resolved),outside);return reflectComponents(value,reflected.w);}
fn provisionalSelectorVector(anchor:u32,selector:u32,transform:u32)->vec4f{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return vec4f(0);}let row=rows[anchor];let v=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(v.xyz,transform);let sizeFloat=f32(row.size)*v.w;let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return vec4f(0);}return provisionalSignedVector(vec3i(round(point-.5*f32(size))),size);}
fn finalSelectorVector(anchor:u32,selector:u32,transform:u32)->vec4f{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return vec4f(0);}let row=rows[anchor];let v=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(v.xyz,transform);let sizeFloat=f32(row.size)*v.w;let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return vec4f(0);}return finalSignedVector(vec3i(round(point-.5*f32(size))),size);}
fn finalTetraPointVector(anchor:u32,row:Row,anchorVelocity:vec4f,metric:Metric,header:TetraHeader,pointGrid:vec3f)->vec4f{if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return invalidPointVector(7u);}let origin=coord(row.cell);let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return invalidPointVector(8u);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}let va=finalSelectorVector(anchor,selectors.x,metric.transformFlags&63u);let vb=finalSelectorVector(anchor,selectors.y,metric.transformFlags&63u);let vc=finalSelectorVector(anchor,selectors.z,metric.transformFlags&63u);var result=weights.x*anchorVelocity.xyz;var validWeight=weights.x;if(velocityValid(va)){result+=weights.y*va.xyz;validWeight+=weights.y;}if(velocityValid(vb)){result+=weights.z*vb.xyz;validWeight+=weights.z;}if(velocityValid(vc)){result+=weights.w*vc.xyz;validWeight+=weights.w;}if(validWeight<=1e-6){return invalidPointVector(9u);}result/=validWeight;return select(invalidPointVector(10u),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}return invalidPointVectorAt(11u,vec3f(origin));}
fn marchedCentroidVector(anchor:u32,pointGrid:vec3f)->vec4f{if(anchor==INVALID||anchor>=arrayLength(&rows)||anchor>=arrayLength(&metrics)||(rows[anchor].flags&(ROW_SUPPORT3_NODE|ROW_SUPPORT3_ENDPOINT))!=0u){return invalidPointVector(1u);}let row=rows[anchor];let origin=coord(row.cell);let anchorVelocity=provisionalCellVector(origin,row.size);if(!velocityValid(anchorVelocity)){return invalidPointVector(2u);}let metric=metrics[anchor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){return invalidPointVector(3u);}let header=tetraHeaders[metric.topology];var cubeSupportMissing=false;if((header.flags&1u)!=0u){let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);if(all(t>=vec3f(-2e-6))&&all(t<=vec3f(1.000002))){var result=vec3f(0);var cubeValid=true;for(var corner=0u;corner<8u;corner+=1u){let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);if(weight==0.){continue;}let cornerOrigin=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u))*i32(row.size);let value=provisionalSignedVector(cornerOrigin,row.size);if(!velocityValid(value)){cubeValid=false;cubeSupportMissing=true;break;}result+=weight*value.xyz;}if(cubeValid){return select(invalidPointVector(8u),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}}}if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return invalidPointVector(5u);}let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return invalidPointVector(6u);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}let va=provisionalSelectorVector(anchor,selectors.x,metric.transformFlags&63u);let vb=provisionalSelectorVector(anchor,selectors.y,metric.transformFlags&63u);let vc=provisionalSelectorVector(anchor,selectors.z,metric.transformFlags&63u);if(!velocityValid(va)||!velocityValid(vb)||!velocityValid(vc)){return invalidPointVector(7u);}let result=weights.x*anchorVelocity.xyz+weights.y*va.xyz+weights.z*vb.xyz+weights.w*vc.xyz;return select(invalidPointVector(8u),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}return invalidPointVector(select(9u,4u,cubeSupportMissing));}
fn invalidPointVector(reason:u32)->vec4f{return vec4f(0.,0.,0.,-f32(reason));}
fn invalidPointVectorAt(reason:u32,detail:vec3f)->vec4f{return vec4f(detail,-f32(reason));}
fn uniformCubeContains(row:Row,pointGrid:vec3f)->bool{let origin=coord(row.cell);let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);return all(t>=vec3f(-2e-6))&&all(t<=vec3f(1.000002));}
fn finalPointVector(anchor:u32,pointGrid:vec3f)->vec4f{if(anchor==INVALID||anchor>=arrayLength(&rows)||anchor>=arrayLength(&metrics)||(rows[anchor].flags&ROW_SUPPORT3_ENDPOINT)!=0u){return invalidPointVector(1u);}let row=rows[anchor];let origin=coord(row.cell);var anchorVelocity=finalCellVector(origin,row.size);if(!velocityValid(anchorVelocity)&&(row.flags&(ROW_SUPPORT2|ROW_SUPPORT3_NODE))!=0u){anchorVelocity=supportCellVector(origin,row.size);}if(!velocityValid(anchorVelocity)){return invalidPointVector(2u);}let metric=metrics[anchor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){return invalidPointVector(3u);}let header=tetraHeaders[metric.topology];if((header.flags&1u)!=0u){let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);if(uniformCubeContains(row,pointGrid)){var result=vec3f(0);var cubeValid=true;for(var corner=0u;corner<8u;corner+=1u){let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);if(weight==0.){continue;}let cornerOrigin=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u))*i32(row.size);let value=finalSignedVector(cornerOrigin,row.size);if(!velocityValid(value)){cubeValid=false;break;}result+=weight*value.xyz;}if(cubeValid){return select(invalidPointVector(6u),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}}return finalTetraPointVector(anchor,row,anchorVelocity,metric,header,pointGrid);}if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return invalidPointVector(7u);}let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return invalidPointVector(8u);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}let va=finalSelectorVector(anchor,selectors.x,metric.transformFlags&63u);let vb=finalSelectorVector(anchor,selectors.y,metric.transformFlags&63u);let vc=finalSelectorVector(anchor,selectors.z,metric.transformFlags&63u);var result=weights.x*anchorVelocity.xyz;var validWeight=weights.x;if(velocityValid(va)){result+=weights.y*va.xyz;validWeight+=weights.y;}if(velocityValid(vb)){result+=weights.z*vb.xyz;validWeight+=weights.z;}if(velocityValid(vc)){result+=weights.w*vc.xyz;validWeight+=weights.w;}if(validWeight<=1e-6){return invalidPointVector(9u);}result/=validWeight;return select(invalidPointVector(10u),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}return invalidPointVectorAt(11u,vec3f(origin));}
// Resolve the published row containing an integer probe without scanning the
// row arena. Multiple support levels can cover the same probe; selecting the
// lowest row id matches the publication-prefix order used by the former scan.
fn containingPublishedRow(probe:vec3i)->u32{if(any(probe<vec3i(0))||any(probe>=vec3i(p.dims))){return INVALID;}let q=vec3u(probe);var size=1u;var selected=INVALID;loop{let origin=(q/vec3u(size))*vec3u(size);let candidate=rowOfIdentity(cell(origin),size);if(candidate!=INVALID&&candidate<arrayLength(&rows)&&candidate<arrayLength(&rowVelocities)){let r=rows[candidate];if((r.flags&ROW_SUPPORT3_ENDPOINT)==0u&&all(q>=origin)&&all(q<origin+vec3u(size))){selected=min(selected,candidate);}}if(size>=p.maximumLeaf){break;}size<<=1u;}return selected;}
// Co-spherical Cartesian sites admit several ordinary-Delaunay
// tetrahedralizations. A static anchor fan can legitimately choose the
// axial-star triangulation while a trajectory lies in the complementary
// central octahedron. Section 6.2 permits any deterministic ordinary-Delaunay
// triangulation, so resolve that degeneracy from the actual eight surrounding
// The owner containing a trajectory sample need not anchor the catalog fan
// that contains it. Catalog selectors are bounded by 1.5 anchor widths in
// every axis, so a containing cube/tetra anchor of size s must lie in the 5^3
// aligned-origin box around floor(point/s). Enumerate that exact bounded box
// for every dyadic size and select the lowest valid row id, reproducing the
// former ascending-row result without work proportional to row capacity.
struct ResolvedPointVectorMeasurement{value:vec4f,directSuccess:u32,boundedCatalogSearch:u32,candidateRowsTested:u32}
fn resolveFinalPointVectorMeasured(initialAnchor:u32,pointGrid:vec3f)->ResolvedPointVectorMeasurement{let direct=finalPointVector(initialAnchor,pointGrid);if(velocityValid(direct)){return ResolvedPointVectorMeasurement(direct,1u,0u,0u);}let directReason=u32(round(max(0.,-direct.w)));let interpolationGap=directReason==2u||directReason==9u||directReason==11u;let carrierEligible=interpolationGap&&initialAnchor<arrayLength(&rows)&&rows[initialAnchor].minimumPhi>=0.;var bestRow=INVALID;var bestValue=vec4f(0.);var bestCarrierRow=INVALID;var bestCarrierDistance=3.402823e38;var bestCarrierValue=vec4f(0.);var containedInvalidRow=INVALID;var containedInvalid=vec4f(0.);var candidateRowsTested=0u;if(carrierEligible){let initial=rows[initialAnchor];let carrier=finalCellVector(coord(initial.cell),initial.size);if(velocityValid(carrier)){bestCarrierRow=initialAnchor;bestCarrierDistance=0.;bestCarrierValue=carrier;}}var size=1u;loop{let base=vec3i(floor(pointGrid/f32(size)));for(var dz=-2i;dz<=2i;dz+=1i){for(var dy=-2i;dy<=2i;dy+=1i){for(var dx=-2i;dx<=2i;dx+=1i){let originIndex=base+vec3i(dx,dy,dz);if(any(originIndex<vec3i(0))){continue;}let origin=vec3u(originIndex)*size;if(any(origin+vec3u(size)>p.dims)){continue;}let candidate=rowOfIdentity(cell(origin),size);if(candidate==INVALID||candidate==initialAnchor||candidate>=arrayLength(&rows)||candidate>=arrayLength(&metrics)){continue;}let row=rows[candidate];if((row.flags&ROW_SUPPORT3_ENDPOINT)!=0u){continue;}candidateRowsTested+=1u;if(carrierEligible){let carrier=finalCellVector(origin,size);if(velocityValid(carrier)){let low=vec3f(origin);let high=low+f32(size);let delta=max(max(low-pointGrid,vec3f(0.)),pointGrid-high);let distance=dot(delta,delta);if(distance<bestCarrierDistance||(distance==bestCarrierDistance&&candidate<bestCarrierRow)){bestCarrierRow=candidate;bestCarrierDistance=distance;bestCarrierValue=carrier;}}}if(candidate>=bestRow){continue;}let value=finalPointVector(candidate,pointGrid);if(velocityValid(value)){bestRow=candidate;bestValue=value;continue;}if(u32(round(max(0.,-value.w)))==9u&&candidate<containedInvalidRow){containedInvalidRow=candidate;containedInvalid=value;}}}}if(size>=p.maximumLeaf){break;}size<<=1u;}if(bestRow!=INVALID){return ResolvedPointVectorMeasurement(bestValue,0u,1u,candidateRowsTested);}if(bestCarrierRow!=INVALID){return ResolvedPointVectorMeasurement(bestCarrierValue,0u,1u,candidateRowsTested);}
  // A co-spherical local catalog can leave a trajectory exactly in the gap
  // between otherwise valid local fans.  Section 5's outside-liquid operation
  // is constant extension from the closest regular-face carrier, so retain the
  // already-published full vector of the row that actually owns this point.
  // This is bounded by the dyadic depth and remains fail-closed when either the
  // containing row or its completed vector is absent.
  let carrierRow=select(INVALID,containingPublishedRow(vec3i(floor(pointGrid))),directReason==11u);
  if(carrierRow!=INVALID&&carrierRow<arrayLength(&rows)){
    let carrierRecord=rows[carrierRow];
    let carrier=supportCellVector(coord(carrierRecord.cell),carrierRecord.size);
    if(velocityValid(carrier)){return ResolvedPointVectorMeasurement(carrier,0u,1u,candidateRowsTested);}
  }
  return ResolvedPointVectorMeasurement(select(direct,containedInvalid,containedInvalidRow!=INVALID),0u,1u,candidateRowsTested);}
fn resolveFinalPointVector(initialAnchor:u32,pointGrid:vec3f)->vec4f{return resolveFinalPointVectorMeasured(initialAnchor,pointGrid).value;}
@compute @workgroup_size(64)fn interpolatePowerFaceVector(@builtin(global_invocation_id)g:vec3u){let index=g.x;if(index>=powerPublication.faceCount||powerPublication.flags!=0u||index>=arrayLength(&powerVelocityScratch)){return;}if(index>=arrayLength(&powerFaceCentroids)){storePowerFaceFailure(index,POWER_CAPACITY);return;}let mapping=powerVelocityScratch[index];let negativeBand=mapping.z;let positiveBand=mapping.w;if(negativeBand==INVALID&&positiveBand==INVALID){return;}
  if(bandRowIsWet(negativeBand)||bandRowIsWet(positiveBand)){powerVelocityScratch[index]=vec4u(0u,0u,0u,2u);return;}
  let centroid=powerFaceCentroids[index].xyz;let h=fp.fineWidth*f32(fp.fineFactor);if(!finite(centroid.x)||!finite(centroid.y)||!finite(centroid.z)||!finite(h)||h<=0.){storePowerFaceFailure(index,POWER_NONFINITE);return;}let pointGrid=centroid/h;for(var axis=0u;axis<3u;axis+=1u){let atNegative=abs(pointGrid[axis])<=1e-5;let atPositive=abs(pointGrid[axis]-f32(p.dims[axis]))<=1e-5;if(!atNegative&&!atPositive){continue;}let closedBit=select(positiveBoundaryBit(axis),negativeBoundaryBit(axis),atNegative);if((p.closedBoundaryMask&closedBit)!=0u){powerVelocityScratch[index]=vec4u(0u,0u,0u,1u);return;}let band=select(positiveBand,negativeBand,atPositive);if(!atPositive||band==INVALID||band>=arrayLength(&rows)||!finite(rows[band].padf)){storePowerFaceFailure(index,POWER_MISSING_ROW);return;}var boundaryVector=vec3f(0.);boundaryVector[axis]=rows[band].padf;powerVelocityScratch[index]=vec4u(bitcast<u32>(boundaryVector.x),bitcast<u32>(boundaryVector.y),bitcast<u32>(boundaryVector.z),1u);return;}let negative=resolveFinalPointVector(negativeBand,pointGrid);let positive=resolveFinalPointVector(positiveBand,pointGrid);var full=vec4f(0);if(velocityValid(negative)&&velocityValid(positive)){full=vec4f(.5*(negative.xyz+positive.xyz),1.);}else if(velocityValid(negative)){full=negative;}else if(velocityValid(positive)){full=positive;}else{storePowerFaceFailure(index,POWER_MISSING_ROW);return;}powerVelocityScratch[index]=vec4u(bitcast<u32>(full.x),bitcast<u32>(full.y),bitcast<u32>(full.z),1u);}
@compute @workgroup_size(64)fn projectPowerFaceVelocity(@builtin(global_invocation_id)g:vec3u){let index=g.x;if(index>=powerPublication.faceCount||powerPublication.flags!=0u||index>=arrayLength(&powerVelocityScratch)){return;}if(index>=arrayLength(&powerFaces)||index>=arrayLength(&powerFaceNormals)){storePowerFaceFailure(index,POWER_CAPACITY);return;}let candidate=powerVelocityScratch[index];
  // Sentinel 2 marks a liquid/interface face whose projected velocity must
  // survive; it completes the publication tally without becoming committable.
  if(candidate.w==2u){powerVelocityScratch[index]=vec4u(0u,3u,0u,0u);return;}
  if(candidate.w!=1u){return;}let powerFace=powerFaces[index];if((powerFace.flags&VALID)==0u||powerFace.negativeRow>=p.powerRowCapacity||(powerFace.positiveRow!=INVALID&&powerFace.positiveRow>=p.powerRowCapacity)||!finite(powerFace.area)||powerFace.area<=0.){storePowerFaceFailure(index,POWER_FACE);return;}let normal=powerFaceNormals[index].xyz;let n2=dot(normal,normal);if(!finite(normal.x)||!finite(normal.y)||!finite(normal.z)||!finite(n2)||abs(n2-1.)>4e-4){storePowerFaceFailure(index,POWER_NORMAL);return;}let full=vec3f(bitcast<f32>(candidate.x),bitcast<f32>(candidate.y),bitcast<f32>(candidate.z));let value=dot(full,normal);if(!finite(value)){storePowerFaceFailure(index,POWER_NONFINITE);return;}powerVelocityScratch[index]=vec4u(bitcast<u32>(value),1u,0u,0u);}
var<workgroup> powerReduceCount0:array<u32,256>;var<workgroup> powerReduceCount1:array<u32,256>;var<workgroup> powerReduceFlags:array<u32,256>;var<workgroup> powerReduceFirst:array<u32,256>;
@compute @workgroup_size(256)fn commitPowerFaceVelocity(@builtin(local_invocation_index)lid:u32){var targets=0u;var interpolated=0u;var flags=0u;var first=INVALID;for(var index=lid;index<powerPublication.faceCount;index+=256u){if(index>=arrayLength(&powerVelocityScratch)){flags|=POWER_CAPACITY;first=min(first,index);continue;}let marker=powerVelocityScratch[index].y;if((marker&POWER_FAILURE_MARKER)!=0u){flags|=marker&~POWER_FAILURE_MARKER;first=min(first,index);continue;}if(marker==1u||marker==3u){targets+=1u;interpolated+=1u;}}powerReduceCount0[lid]=targets;powerReduceCount1[lid]=interpolated;powerReduceFlags[lid]=flags;powerReduceFirst[lid]=first;workgroupBarrier();for(var width=128u;width>0u;width>>=1u){if(lid<width){powerReduceCount0[lid]+=powerReduceCount0[lid+width];powerReduceCount1[lid]+=powerReduceCount1[lid+width];powerReduceFlags[lid]|=powerReduceFlags[lid+width];powerReduceFirst[lid]=min(powerReduceFirst[lid],powerReduceFirst[lid+width]);}workgroupBarrier();}if(lid==0u){powerPublication.flags|=powerReduceFlags[0];powerPublication.firstError=min(powerPublication.firstError,powerReduceFirst[0]);powerPublication.targetCount=powerReduceCount0[0];powerPublication.interpolatedCount=powerReduceCount1[0];let ready=control.valid==VALID&&powerPublication.flags==0u&&powerReduceCount0[0]>0u&&powerReduceCount1[0]==powerReduceCount0[0];if(!ready&&powerReduceCount0[0]==0u&&powerPublication.faceCount>0u){powerPublication.flags|=POWER_INCOMPLETE;powerPublication.firstError=min(powerPublication.firstError,0u);}powerPublication.valid=select(0u,VALID,ready);powerPublication.committedCount=0u;}storageBarrier();workgroupBarrier();var committed=0u;if(powerPublication.valid==VALID){for(var index=lid;index<powerPublication.faceCount;index+=256u){if(index>=arrayLength(&powerFaces)||index>=arrayLength(&powerVelocityScratch)){flags|=POWER_CAPACITY;first=min(first,index);continue;}let candidate=powerVelocityScratch[index];if(candidate.y==1u){powerFaces[index].normalVelocity=bitcast<f32>(candidate.x);committed+=1u;}else if(candidate.y==3u){committed+=1u;}}}powerReduceCount0[lid]=committed;powerReduceFlags[lid]=flags;powerReduceFirst[lid]=first;workgroupBarrier();for(var width=128u;width>0u;width>>=1u){if(lid<width){powerReduceCount0[lid]+=powerReduceCount0[lid+width];powerReduceFlags[lid]|=powerReduceFlags[lid+width];powerReduceFirst[lid]=min(powerReduceFirst[lid],powerReduceFirst[lid+width]);}workgroupBarrier();}if(lid==0u){powerPublication.flags|=powerReduceFlags[0];powerPublication.firstError=min(powerPublication.firstError,powerReduceFirst[0]);powerPublication.committedCount=powerReduceCount0[0];if(powerPublication.flags!=0u||powerReduceCount0[0]!=powerPublication.targetCount){powerPublication.valid=0u;}}}
fn retainedBandAnchor(pointGrid:vec3f)->u32{if(any(pointGrid<vec3f(0))||any(pointGrid>=vec3f(sp.dims))){return INVALID;}let q=vec3u(floor(pointGrid));var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);let band=rowOfIdentity(cell(origin),size);if(band!=INVALID&&band<sp.rowCapacity&&band<arrayLength(&rows)&&(rows[band].flags&ROW_SUPPORT3_ENDPOINT)==0u){return band;}if(size>=sp.maximumLeaf){break;}size<<=1u;}return INVALID;}
// Section 5 first extrapolates velocity on regular octree faces, then reuses
// the standard per-axis staggered interpolant away from level transitions.
// Query the retained closest-point-extended face carriers directly: reconstructing cell
// centres first would insert an avoidable face->cell->face low-pass filter.
fn retainedBandVector(world:vec3f)->vec4f{let pointGrid=world/sp.cellSize;let anchor=retainedBandAnchor(pointGrid);if(anchor==INVALID){return vec4f(bitcast<f32>(0x1ffffu),0.,0.,-20.);}let value=resolveFinalPointVector(anchor,pointGrid);if(velocityValid(value)){return value;}return vec4f(bitcast<f32>(anchor),value.yz,value.w);}
// A generalized face centroid is shared by its liquid and air interpolation
// elements.  The integer owner directory is half-open, so an exact dual-face
// point may select the air support row even though the continuous Section 5
// field includes the incident liquid element.  Bias only that measure-zero
// case toward the face's negative (incident) cell; points genuinely in air
// still have to resolve through the closest-point-extended band.
fn retainedBandIncidentVector(world:vec3f,normal:vec3f)->vec4f{
  let direct=retainedBandVector(world);
  if(velocityValid(direct)){return direct;}
  let incident=retainedBandVector(world-sp.cellSize*1e-4*normal);
  if(velocityValid(incident)){return incident;}
  // Apply the same product boundary policy as velocityExtendedOrigin to a
  // continuous RK departure: closed planes mirror the query and odd-reflect
  // only their wall-normal velocity component, while the authored open +Y
  // ceiling has a zero-gradient extension. The fixed maximum-leaf bound
  // admits one CFL support crossing without publishing exterior rows.
  var grid=world/sp.cellSize;
  var flips=0i;
  var extended=false;
  for(var axis=0u;axis<3u;axis+=1u){
    let extent=f32(sp.dims[axis]);
    if(grid[axis]<0.){
      let closed=(p.closedBoundaryMask&negativeBoundaryBit(axis))!=0u;
      if(!closed||grid[axis]<-f32(sp.maximumLeaf)){return direct;}
      grid[axis]=-grid[axis];
      flips|=1i<<axis;
      extended=true;
    }else if(grid[axis]>=extent){
      let closed=(p.closedBoundaryMask&positiveBoundaryBit(axis))!=0u;
      let openCeiling=axis==1u&&!closed;
      if((!closed&&!openCeiling)||grid[axis]>extent+f32(sp.maximumLeaf)){return direct;}
      grid[axis]=select(extent-1e-5,2.*extent-grid[axis],closed);
      if(closed){flips|=1i<<axis;}
      extended=true;
    }
    grid[axis]=clamp(grid[axis],0.,extent-1e-5);
  }
  if(!extended){return direct;}
  let boundary=retainedBandVector(grid*sp.cellSize);
  return select(direct,reflectComponents(boundary,flips),velocityValid(boundary));
}
fn failPowerFaceRegularCompletion(index:u32,stage:u32,detail:u32){let reason=detail&255u;if(index<oldAdvectionControl.firstError){oldAdvectionControl.firstInterpolationDetail=detail;}oldAdvectionControl.flags|=8u;oldAdvectionControl.firstError=min(oldAdvectionControl.firstError,index);oldAdvectionControl.firstInterpolationStage=min(oldAdvectionControl.firstInterpolationStage,index*4u+stage);oldAdvectionControl.firstInterpolationReason=min(oldAdvectionControl.firstInterpolationReason,index*16u+min(reason,15u));oldAdvectionControl.valid=0u;oldAdvectionSeed.flags|=8u;oldAdvectionSeed.firstError=min(oldAdvectionSeed.firstError,index);oldAdvectionSeed.valid=0u;}
fn regularCompletionAuthorityFailures()->u32{var failures=0u;if(control.valid!=VALID){failures|=1u;}if(control.generation!=sp.fineGeneration){failures|=2u;}if(pointControl.valid!=VALID){failures|=4u;}if(pointControl.flags!=0u){failures|=8u;}if(pointControl.generation!=sp.fineGeneration){failures|=16u;}if(powerFaceControl[8]!=VALID){failures|=32u;}if(powerFaceControl[7]!=sp.p2){failures|=64u;}return failures;}
@compute @workgroup_size(1)fn preparePowerFaceRegularCompletion(){oldAdvectionSeed.valid=0u;let requested=oldAdvectionControl.requestedFaces;if(oldAdvectionControl.flags!=0u){oldAdvectionSeed.flags|=oldAdvectionControl.flags;oldAdvectionSeed.firstError=min(oldAdvectionSeed.firstError,oldAdvectionControl.firstError);return;}if(requested==0u||arrayLength(&powerFaceControl)<9u||requested!=powerFaceControl[1]||requested>arrayLength(&powerFaceCentroids)||requested>arrayLength(&powerFaceNormals)||requested>arrayLength(&powerFaces)){failPowerFaceRegularCompletion(0u,1u,(requested<<8u)|27u);return;}let authorityFailures=regularCompletionAuthorityFailures();if(authorityFailures!=0u){failPowerFaceRegularCompletion(0u,1u,(authorityFailures<<8u)|21u);}}
const STATUS_REGULAR_COMPLETED:u32=0x40000000u;
fn storeBandCompletionFailure(index:u32,stage:u32,value:vec4f){let reason=u32(max(1.,-value.w));let anchor=bitcast<u32>(value.x);powerFaceCentroids[index].w=bitcast<f32>((min(anchor,0x1ffffu)<<10u)|((stage&3u)<<8u)|min(reason,255u));}
// The retained regular field is already the paper's closest-point constant
// extension. A newly exposed power face can start inside that field while an
// RK midpoint lands just beyond its terminal row set. Continue the last valid
// extended vector only for that exact missing-band sentinel; malformed local
// interpolation and every other failure remain fail-closed.
fn continueMissingBandVector(sampled:vec4f,previous:vec4f)->vec4f{
  return select(sampled,previous,!velocityValid(sampled)&&sampled.w==-20.&&velocityValid(previous));
}
@compute @workgroup_size(64)fn completePowerFaceAdvectionFromRegularBand(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(arrayLength(&powerFaceControl)<9u||i>=powerFaceControl[1]||i>=arrayLength(&powerFaceNormals)||i>=arrayLength(&powerFaceCentroids)){return;}let priorStatus=bitcast<u32>(powerFaceCentroids[i].w);if(priorStatus==STATUS_VALID||priorStatus==STATUS_REGULAR_COMPLETED){return;}let n=powerFaceNormals[i].xyz;let x=powerFaceCentroids[i].xyz;let v0=retainedBandIncidentVector(x,n);if(!velocityValid(v0)){storeBandCompletionFailure(i,1u,v0);return;}let dt=bitcast<f32>(sp.p1);var vm=retainedBandIncidentVector(x-.5*dt*v0.xyz,n);vm=continueMissingBandVector(vm,v0);if(!velocityValid(vm)){storeBandCompletionFailure(i,2u,vm);return;}var va=retainedBandIncidentVector(x-dt*vm.xyz,n);va=continueMissingBandVector(va,vm);if(!velocityValid(va)){storeBandCompletionFailure(i,3u,va);return;}let value=dot(va.xyz,n);if(!finite(value)){powerFaceCentroids[i].w=bitcast<f32>((3u<<8u)|22u);return;}powerFaceNormals[i].w=value;powerFaceCentroids[i].w=bitcast<f32>(STATUS_REGULAR_COMPLETED);}
var<workgroup>powerCompletionReduce:array<vec4u,256>;var<workgroup>powerCompletionReason:array<u32,256>;var<workgroup>powerCompletionDetail:array<u32,256>;
@compute @workgroup_size(256)fn publishCompletedPowerFaceAdvection(@builtin(local_invocation_index)lid:u32){let requested=oldAdvectionControl.requestedFaces;var flags=0u;var first=INVALID;var completed=0u;var firstStage=INVALID;var firstReason=INVALID;var firstDetail=INVALID;if(oldAdvectionControl.flags==0u){for(var i=lid;i<requested;i+=256u){let status=bitcast<u32>(powerFaceCentroids[i].w);if(status==STATUS_VALID){completed+=1u;continue;}if(status==STATUS_REGULAR_COMPLETED){let value=powerFaceNormals[i].w;if(finite(value)&&(powerFaces[i].flags&POWER_FACE_VALID)!=0u){powerFaces[i].normalVelocity=value;powerFaceNormals[i].w=0.;powerFaceCentroids[i].w=bitcast<f32>(STATUS_VALID);completed+=1u;continue;}}let effective=select((1u<<8u)|26u,status,status!=0u);let stage=(effective>>8u)&3u;let reason=effective&255u;flags|=8u;if(i<first){first=i;firstDetail=((effective>>10u)<<8u)|reason;}firstStage=min(firstStage,i*4u+stage);firstReason=min(firstReason,i*16u+min(reason,15u));}}powerCompletionReduce[lid]=vec4u(flags,first,completed,firstStage);powerCompletionReason[lid]=firstReason;powerCompletionDetail[lid]=firstDetail;for(var width=128u;width>0u;width>>=1u){workgroupBarrier();if(lid<width){let right=powerCompletionReduce[lid+width];if(right.y<powerCompletionReduce[lid].y){powerCompletionDetail[lid]=powerCompletionDetail[lid+width];}powerCompletionReduce[lid].x|=right.x;powerCompletionReduce[lid].y=min(powerCompletionReduce[lid].y,right.y);powerCompletionReduce[lid].z+=right.z;powerCompletionReduce[lid].w=min(powerCompletionReduce[lid].w,right.w);powerCompletionReason[lid]=min(powerCompletionReason[lid],powerCompletionReason[lid+width]);}}workgroupBarrier();if(lid==0u){let result=powerCompletionReduce[0];oldAdvectionControl.flags|=result.x;oldAdvectionControl.firstError=min(oldAdvectionControl.firstError,result.y);oldAdvectionControl.advected=result.z;oldAdvectionControl.firstInterpolationStage=min(oldAdvectionControl.firstInterpolationStage,result.w);oldAdvectionControl.firstInterpolationReason=min(oldAdvectionControl.firstInterpolationReason,powerCompletionReason[0]);if(result.y!=INVALID){oldAdvectionControl.firstInterpolationDetail=powerCompletionDetail[0];}let complete=oldAdvectionControl.flags==0u&&requested>0u&&result.z==requested;oldAdvectionControl.valid=select(0u,VALID,complete);oldAdvectionSeed.flags|=result.x;oldAdvectionSeed.firstError=min(oldAdvectionSeed.firstError,result.y);oldAdvectionSeed.valid=select(0u,VALID,complete);}}
fn airSampleGrid(world:vec3f)->vec4f{var grid=world/sp.cellSize;if(!finite(grid.x)||!finite(grid.y)||!finite(grid.z)){return vec4f(0);}for(var axis=0u;axis<3u;axis+=1u){if(grid[axis]<0.){if((p.closedBoundaryMask&negativeBoundaryBit(axis))==0u){return vec4f(0);}grid[axis]=0.;}if(grid[axis]>=f32(sp.dims[axis])){let closed=(p.closedBoundaryMask&positiveBoundaryBit(axis))!=0u;let openCeiling=axis==1u&&!closed;if(!closed&&!openCeiling){return vec4f(0);}grid[axis]=f32(sp.dims[axis])-1e-5;}}return vec4f(grid,1.);}
`;

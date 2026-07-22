/**
 * GPU-resident Section 5 regular-face velocity extrapolation and power-face
 * publication for factor-4 and factor-8 global fine level sets.
 *
 * The fine SPGrid contributes the row-discovery frontier and authoritative
 * signed distance sampled at actual regular-face centroids; velocity remains
 * on compact octree rows/faces. Faces are emitted only after both endpoint
 * rows resolve, and a deterministic parallel closest-point transform copies
 * the wet incident-face carrier reached by a strictly decreasing phi path.
 */
import { OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW, OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW,
  planOctreeRegularFaceBand, type OctreeRegularFaceBandPlan } from "./octree-face-fast-march";
import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import {
  OCTREE_POWER_NEIGHBOR_DIRECTIONS,
  OCTREE_POWER_SAME_OR_COARSER_FLAG,
} from "./octree-power-descriptor";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import type { OctreePowerFaceSource } from "./webgpu-octree-power-faces";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";

export const OCTREE_FACE_BAND_VALID = 0x8000_0000;
export const OCTREE_FACE_BAND_EXTRAPOLATED = 0x1000_0000;
export const OCTREE_FACE_BAND_CONTROL_BYTES = 128;
const OCTREE_FACE_BAND_HEAP_CHUNK = 1024;
export const OCTREE_FACE_BAND_FACE_BYTES = 64;
export const OCTREE_FACE_BAND_ROW_BYTES = 32;
export const OCTREE_FACE_BAND_STATE_BYTES = 32;
export const OCTREE_FACE_BAND_TRANSITION_ADJACENCY_BYTES = 16;
/** Existing 16-word gate followed by a 16-word first-owner-mismatch record,
 * then the appended S4-node prefix. The first 128 bytes remain ABI-stable. */
export const OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES = 160;
export const OCTREE_FACE_BAND_POWER_PUBLICATION_CONTROL_BYTES = 64;
export const OCTREE_FACE_BAND_POINT_FIELD_CONTROL_BYTES = 32;
export const OCTREE_FACE_BAND_POINT_LS_BYTES = 48;
export const OCTREE_FACE_BAND_TRANSIENT_POWER_FACE_BYTES = 64;
export const OCTREE_FACE_BAND_TRANSIENT_INCIDENCE_BYTES = 8;
export const OCTREE_FACE_BAND_TRANSIENT_ROW_BYTES = 16;
export const OCTREE_FACE_BAND_TRANSIENT_CONTROL_BYTES = 64;
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
  capacity: 1, hashProbe: 2, invalidSource: 4, invalidRow: 8,
  invalidFace: 16, invalidPhi: 32, unresolved: 64, incompleteVector: 128,
  outsideFineBand: 256,
} as const);
export const OCTREE_FACE_BAND_TRANSITION_ERROR = Object.freeze({
  invalidSource: 1, capacity: 2, unresolvedAdjacency: 4, invalidBandDescriptor: 8,
  acuteGrading: 16,
} as const);
export const OCTREE_FACE_BAND_TRANSITION_DETAIL = Object.freeze({
  malformedGeometry: 1, belowDomain: 2, aboveDomain: 4, misalignedGeometry: 8,
  ownerMismatch: 16, missingBandRow: 32, rowOutOfRange: 64, ownerSizeMismatch: 128,
} as const);
export const OCTREE_FACE_BAND_OWNER_FAILURE_STAGE = Object.freeze({
  support1: 1, support2: 2, support3: 3, transitionAdjacency: 4, endpoint: 5,
  acuteGrading: 6, supportEdge: 7, support4: 8, descriptor: 9,
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
  readonly maximumDepth: number;
  readonly seedCount: number;
  readonly acceptedCount: number;
  readonly unresolvedCount: number;
  readonly sampleFailures: number;
  /** Faces whose current fine-centroid stencil was absent and were evaluated
   * from the redistanced transient owner field with the same cube/tetra interpolant. */
  readonly coarsePhiFallbacks: number;
  /** Owner rows or faces for which current fine, exact compact coarse, and
   * band-local redistance could not provide the Section 5 ordering scalar. */
  readonly coarsePhiFailures: number;
  /** Dry owner rows whose signed distance was extended from the current
   * narrow-band field over the transient Section 5 owner graph. */
  readonly bandPhiExtensions: number;
  /** Maximum number of non-seed LIVE faces resident in the GPU min-heap. */
  readonly marchHeapHighWater: number;
  /** Candidates removed from the heap; must equal `marchTrials` at completion. */
  readonly marchPops: number;
  /** LIVE non-seed faces inserted exactly once into the heap. */
  readonly marchTrials: number;
  /** Chunk dispatches that performed at least one bounded heap pop. */
  readonly marchChunks: number;
  /** Topology-derived pre-encoded chunk bound, ceil(faceCapacity / 1024). */
  readonly marchChunkBound: number;
  /** Faces still TRIAL after the exact candidate-pop bound: scheduler exhaustion. */
  readonly marchCapExhausted: number;
  /** Popped UNKNOWN faces that later expose an accepted causal predecessor. */
  readonly marchUnresolvedWithAcceptedPredecessor: number;
  /** Popped UNKNOWN faces with no accepted causal predecessor: disconnected component/sink. */
  readonly marchDisconnected: number;
  /** Stage-B air queries satisfied by their retained owner anchor without a row scan. */
  readonly directAnchorSuccess: number;
  /** Stage-B air queries whose retained owner anchor failed and entered the full-row fallback. */
  readonly fullRowFallbackInvocations: number;
  /** Candidate-row loop iterations performed by the full-row fallback, including skipped rows. */
  readonly fullRowCandidateRowsTested: number;
  /** Stage-B air queries that exhausted the full-row fallback and entered local-owner repair. */
  readonly surroundingOwnerFallbackInvocations: number;
  /** Candidate-row loop iterations performed while resolving the eight surrounding owners. */
  readonly surroundingOwnerRowsTested: number;
  /** Air samples classified for exact Stage-B face-band evaluation. */
  readonly airSamplesSelected: number;
  /** Classified air samples actually entered by the Stage-B evaluation dispatch. */
  readonly airSamplesEvaluated: number;
  /** Faces completed by the bounded graph-connectivity fallback after the
   * signed-distance ordering reached a discrete positive-air local minimum. */
  readonly connectivityFallbacks: number;
  readonly capacityFailure: boolean;
  readonly hashProbeFailure: boolean;
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
  readonly acuteGrading: boolean;
  /** An excluded same/coarser mask survived grading into the dry-band graph. */
  readonly acuteGradingFailure?: {
    readonly band: number;
    readonly rowCell: number;
    readonly rowSize: number;
    readonly descriptor: number;
    readonly coarseMask: number;
  };
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

/** Exact host encoding bound for the GPU-resident Section 5 min-heap. Every
 * LIVE non-seed face is appended once, so `faceCapacity` pops are sufficient
 * without observing a GPU counter on the CPU. Empty tail chunks are no-ops. */
export function planOctreeFaceBandMarchHeap(faceCapacity: number): {
  readonly popBound: number; readonly chunkSize: number; readonly chunkBound: number;
} {
  positive(faceCapacity, "Face-march heap capacity");
  return { popBound: faceCapacity, chunkSize: OCTREE_FACE_BAND_HEAP_CHUNK,
    chunkBound: Math.ceil(faceCapacity / OCTREE_FACE_BAND_HEAP_CHUNK) };
}

/** Fixed parallel pointer-jump budget for the face closest-point transform.
 * The predecessor graph is strictly ordered by (arrival, stable face, slot),
 * so its longest possible path is bounded by the adaptive-domain diameter.
 * Pointer jumping halves that path on every dispatch. */
export function planOctreeFaceBandCPT(maximumGraphDepth: number): {
  readonly maximumGraphDepth: number; readonly jumpRounds: number;
} {
  positive(maximumGraphDepth, "Face-band CPT graph depth");
  return { maximumGraphDepth, jumpRounds: Math.ceil(Math.log2(maximumGraphDepth)) };
}

/** CPU mirror of the deterministic GPU heap key used by tests/diagnostics. */
export function octreeFaceBandMarchKeyBefore(
  a: { readonly phi: number; readonly globalFace: number; readonly slot: number },
  b: { readonly phi: number; readonly globalFace: number; readonly slot: number },
): boolean {
  const am = Math.abs(a.phi), bm = Math.abs(b.phi);
  return am < bm || (am === bm && (a.globalFace < b.globalFace
    || (a.globalFace === b.globalFace && a.slot < b.slot)));
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
    acuteGrading: (flags & OCTREE_FACE_BAND_TRANSITION_ERROR.acuteGrading) !== 0,
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
    } } : failureBand !== 0xffff_ffff && Number(words[17]) === OCTREE_FACE_BAND_OWNER_FAILURE_STAGE.acuteGrading
      ? { acuteGradingFailure: {
        band: failureBand,
        rowCell: Number(words[18]) >>> 0,
        rowSize: Number(words[19]) >>> 0,
        descriptor: Number(words[20]) >>> 0,
        coarseMask: Number(words[23]) >>> 0,
      } }
      : failureBand !== 0xffff_ffff ? { ownerFailure: {
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
  const rowCount = Number(words[2]) >>> 0;
  const emittedCount = Number(words[4]) >>> 0;
  const sampledCount = Number(words[5]) >>> 0;
  const validatedCount = Number(words[6]) >>> 0;
  return {
    flags,
    firstError: Number(words[1]) >>> 0,
    rowCount,
    faceSlots: Number(words[3]) >>> 0,
    emittedCount,
    sampledCount,
    validatedCount,
    generation: Number(words[7]) >>> 0,
    valid: flags === 0 && (Number(words[8]) >>> 0) === OCTREE_FACE_BAND_VALID
      && emittedCount > 0 && sampledCount === emittedCount && validatedCount === rowCount,
    ...(flags !== 0 && words.length >= 16 && (Number(words[9]) >>> 0) !== 0xffff_ffff
      ? { diagnostic: Array.from({ length: 7 }, (_, index) => Number(words[9 + index]) >>> 0) }
      : {}),
  };
}

/** Decode the stable 64-byte header plus optional 32-byte heap diagnostics
 * without granting either simulation authority. */
export function unpackOctreeFaceBandControl(words: ArrayLike<number>): OctreeFaceBandControlSnapshot {
  if (words.length < 13) throw new RangeError("Face-band control needs at least 13 u32 words");
  const flags = words[0] >>> 0;
  const faceCount = words[3] >>> 0, acceptedCount = words[9] >>> 0, unresolvedCount = words[10] >>> 0;
  return {
    flags, firstError: words[1] >>> 0, rowCount: words[2] >>> 0, faceCount,
    incidenceCount: words[4] >>> 0, generation: words[5] >>> 0,
    valid: (words[6] >>> 0) === OCTREE_FACE_BAND_VALID && flags === 0
      && unresolvedCount === 0 && acceptedCount === faceCount,
    maximumDepth: words[7] >>> 0,
    seedCount: words[8] >>> 0, acceptedCount, unresolvedCount,
    sampleFailures: words[12] >>> 0,
    coarsePhiFallbacks: words[13] >>> 0,
    coarsePhiFailures: words[14] >>> 0,
    bandPhiExtensions: words[15] >>> 0,
    marchHeapHighWater: Number(words[16] ?? 0) >>> 0,
    marchPops: Number(words[17] ?? 0) >>> 0,
    marchTrials: Number(words[18] ?? 0) >>> 0,
    marchChunks: Number(words[19] ?? 0) >>> 0,
    marchChunkBound: Number(words[20] ?? 0) >>> 0,
    marchCapExhausted: Number(words[21] ?? 0) >>> 0,
    marchUnresolvedWithAcceptedPredecessor: Number(words[22] ?? 0) >>> 0,
    marchDisconnected: Number(words[23] ?? 0) >>> 0,
    directAnchorSuccess: Number(words[24] ?? 0) >>> 0,
    fullRowFallbackInvocations: Number(words[25] ?? 0) >>> 0,
    fullRowCandidateRowsTested: Number(words[26] ?? 0) >>> 0,
    surroundingOwnerFallbackInvocations: Number(words[27] ?? 0) >>> 0,
    surroundingOwnerRowsTested: Number(words[28] ?? 0) >>> 0,
    airSamplesSelected: Number(words[29] ?? 0) >>> 0,
    airSamplesEvaluated: Number(words[30] ?? 0) >>> 0,
    connectivityFallbacks: Number(words[31] ?? 0) >>> 0,
    capacityFailure: (flags & OCTREE_FACE_BAND_ERROR.capacity) !== 0,
    hashProbeFailure: (flags & OCTREE_FACE_BAND_ERROR.hashProbe) !== 0,
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
  /** Fine-bounded rows that own descriptors and transition adjacency. */
  readonly coreRowCapacity: number;
  /** All exact support-only owner rows after domain-key deduplication. */
  readonly guardRowCapacity: number;
  readonly support0RowCapacity: number;
  readonly support1RowCapacity: number;
  readonly support2RowCapacity: number;
  readonly support3NodeRowCapacity: number;
  readonly support4NodeRowCapacity: number;
  readonly support5NodeRowCapacity: number;
  readonly support6NodeRowCapacity: number;
  readonly endpointRowCapacity: number;
  readonly metricRowCapacity: number;
  readonly guardCandidateCapacity: number;
  readonly guardCandidateBytes: number;
  readonly rowHashCapacity: number;
  readonly faceHashCapacity: number;
  readonly rowBytes: number;
  readonly bandFaceBytes: number;
  readonly stateBytes: number;
  /** Two immutable-snapshot buffers used by logarithmic CPT pointer jumping. */
  readonly cptParentBytes: number;
  readonly hashBytes: number;
  /** Live indirect-dispatch schedule; the retired serial heap has no arena. */
  readonly frontierBytes: number;
  readonly velocityBytes: number;
  readonly provisionalVelocityBytes: number;
  readonly pointAccumulatorBytes: number;
  readonly pointStatusBytes: number;
  readonly transientPowerFaceCapacity: number;
  readonly transientPowerFaceBytes: number;
  readonly transientPowerIncidenceBytes: number;
  readonly transientPowerRowBytes: number;
  readonly transitionAdjacencyCapacity: number;
  readonly transitionAdjacencyBytes: number;
  readonly transitionMetricBytes: number;
  readonly powerFaceCapacity: number;
  readonly powerVelocityScratchBytes: number;
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
  readonly siteIndex: GPUBuffer;
  readonly siteHashCapacity: number;
  /** Section 5 Stage-A least-squares vectors reconstructed from power faces. */
  readonly powerRowVelocities: GPUBuffer;
  readonly powerVelocityControl: GPUBuffer;
  readonly powerVelocityGeneration: number;
  readonly powerTopology: OctreePowerTopologySource;
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
  readonly rowHash: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly incidence: GPUBuffer;
  readonly velocities: GPUBuffer;
  readonly state: GPUBuffer;
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

/** Semantic stages exposed only to the intrusive host queue-fence profiler. */
export type OctreeFaceBandAirSampleStage = "classifyAirBandVelocity"
  | "evaluateAirBandVelocity" | "finalizeAirBandVelocity";
export type OctreeFaceBandAirSampleBoundary = (
  stage: OctreeFaceBandAirSampleStage,
  encoder: GPUCommandEncoder,
) => GPUCommandEncoder;

function positive(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}
function powerOfTwoAtLeast(value: number): number {
  let result = 1; while (result < value) result *= 2;
  if (!Number.isSafeInteger(result)) throw new RangeError("Face-band hash capacity exceeds the exact integer range");
  return result;
}

export function planOctreeFaceBandGPU(
  wetRowCapacity: number,
  maximumFineBricks: number,
  brickResolution: number,
  fineFactor: number,
  powerFaceCapacityValue?: number,
  finestCellDimensions?: readonly [number, number, number],
): OctreeFaceBandGPUPlan {
  const base = planOctreeRegularFaceBand(wetRowCapacity, maximumFineBricks, brickResolution, fineFactor);
  const coreRowCapacity = base.maximumFineBricks * base.ownerCandidatesPerBrick;
  const domainOwnerCapacity = finestCellDimensions
    ? finestCellDimensions.map((value) => positive(value, "Face-band finest-cell dimension"))
      .reduce((product, value) => product * value, 1)
    : Number.POSITIVE_INFINITY;
  const maximumSupport = Math.max(
    OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows,
    OCTREE_FACE_BAND_UNIFORM_SUPPORT_REQUESTS,
  );
  // S0 is the deduplicated set of rows mapped from fine bricks, never the
  // legacy W+C regular-band planning bound.  Each later role consumes only
  // the remaining physical owner-key budget; this keeps production memory
  // proportional to the actual closure instead of repeatedly reserving the
  // entire finest logical box.
  const support0RowCapacity = Math.min(coreRowCapacity, domainOwnerCapacity);
  let remainingOwners = domainOwnerCapacity - support0RowCapacity;
  const support1RowCapacity = Math.min(remainingOwners, support0RowCapacity * maximumSupport);
  remainingOwners -= support1RowCapacity;
  const support2RowCapacity = Math.min(remainingOwners, support1RowCapacity * maximumSupport);
  remainingOwners -= support2RowCapacity;
  const support3NodeRowCapacity = Math.min(remainingOwners, support2RowCapacity * maximumSupport);
  remainingOwners -= support3NodeRowCapacity;
  const support4NodeRowCapacity = Math.min(remainingOwners, support3NodeRowCapacity * maximumSupport);
  remainingOwners -= support4NodeRowCapacity;
  const support5NodeRowCapacity = Math.min(remainingOwners, support4NodeRowCapacity * maximumSupport);
  remainingOwners -= support5NodeRowCapacity;
  const support6NodeRowCapacity = Math.min(remainingOwners, support5NodeRowCapacity * maximumSupport);
  remainingOwners -= support6NodeRowCapacity;
  const endpointRowCapacity = Math.min(remainingOwners,
    support6NodeRowCapacity * OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW);
  const rowCapacity = support0RowCapacity + support1RowCapacity + support2RowCapacity
    + support3NodeRowCapacity + support4NodeRowCapacity + support5NodeRowCapacity
    + support6NodeRowCapacity + endpointRowCapacity;
  const guardRowCapacity = Math.max(0, rowCapacity - support0RowCapacity);
  // The closure tiers are live prefixes, not fixed partitions of the static
  // reservation above.  Rows can therefore migrate into a tier whose planned
  // role capacity was zero (for example, one S0 row can append three S1 rows
  // in a four-row domain).  Size the shared exact-request arena for every
  // physical row at the largest Section 5 enumeration fanout.
  const maximumCandidateFanout = Math.max(
    maximumSupport,
    OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW,
  );
  const guardCandidateCapacity = rowCapacity * maximumCandidateFanout;
  const faceCapacity = rowCapacity * OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW;
  const incidenceBytes = rowCapacity * (1 + OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW) * 4;
  const guardCandidateBytes = guardCandidateCapacity * 16;
  if (![coreRowCapacity, guardRowCapacity, support0RowCapacity, support1RowCapacity,
    support2RowCapacity, support3NodeRowCapacity, support4NodeRowCapacity, support5NodeRowCapacity,
    support6NodeRowCapacity, endpointRowCapacity,
    guardCandidateCapacity, rowCapacity, faceCapacity,
    incidenceBytes, guardCandidateBytes].every(Number.isSafeInteger)) {
    throw new RangeError("Face-band guard closure exceeds the exact integer range");
  }
  const rowHashCapacity = powerOfTwoAtLeast(rowCapacity * 2);
  const faceHashCapacity = powerOfTwoAtLeast(faceCapacity * 2);
  const rowBytes = rowCapacity * OCTREE_FACE_BAND_ROW_BYTES;
  const bandFaceBytes = faceCapacity * OCTREE_FACE_BAND_FACE_BYTES;
  const stateBytes = faceCapacity * OCTREE_FACE_BAND_STATE_BYTES;
  const cptParentBytes = faceCapacity * 4;
  const hashBytes = (rowHashCapacity + faceHashCapacity) * 8;
  // One map record plus sixteen row/candidate records for the live closure,
  // followed by final row and row-local face-slot prefix records.
  // tiers and endpoint expansion. Candidate dispatches derive from the GPU-published
  // live row prefix, so no tier scans or clears the capacity-sized arena.
  // The records are reused by later phases; the retired serial heap has no
  // allocation in the parallel CPT implementation.
  const frontierBytes = 240;
  const velocityBytes = rowCapacity * 16;
  const provisionalVelocityBytes = rowCapacity * 16;
  const pointAccumulatorBytes = rowCapacity * OCTREE_FACE_BAND_POINT_LS_BYTES;
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
  const metricRowCapacity = Math.min(rowCapacity,
    support0RowCapacity + support1RowCapacity + support2RowCapacity);
  const transitionAdjacencyCapacity = metricRowCapacity
    * OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra;
  const transitionAdjacencyBytes = transitionAdjacencyCapacity * OCTREE_FACE_BAND_TRANSITION_ADJACENCY_BYTES;
  const transitionMetricBytes = rowCapacity * 16;
  const powerFaceCapacity = positive(powerFaceCapacityValue
    ?? base.wetRowCapacity * Math.ceil(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence / 2),
  "Face-band power-face capacity");
  // Scratch changes representation across the publication transaction:
  // (0, 0, negative band, positive band) -> (vector xyz bits, valid) ->
  // (normal velocity bits, targeted marker, 0, 0). Splitting catalog
  // interpolation from normal projection keeps every compute stage within
  // WebGPU's portable ten-storage-buffer limit.
  const powerVelocityScratchBytes = powerFaceCapacity * 16;
  const maximumDirectWorkgroups = Math.ceil(Math.max(rowCapacity, faceCapacity, guardCandidateCapacity,
    powerFaceCapacity, transientPowerFaceCapacity) / 64);
  if (![transitionAdjacencyCapacity, transitionAdjacencyBytes, transientPowerFaceCapacity,
    transientPowerFaceBytes, transientPowerIncidenceBytes, transientPowerRowBytes,
    maximumDirectWorkgroups].every(Number.isSafeInteger)) {
    throw new RangeError("Face-band transition adjacency exceeds the exact integer range");
  }
  return { ...base, rowCapacity, faceCapacity, incidenceBytes,
    allocatedBytes: bandFaceBytes + incidenceBytes + stateBytes + 2 * cptParentBytes,
    coreRowCapacity, guardRowCapacity, support0RowCapacity, support1RowCapacity,
    support2RowCapacity, support3NodeRowCapacity, support4NodeRowCapacity, support5NodeRowCapacity,
    support6NodeRowCapacity, endpointRowCapacity, metricRowCapacity,
    guardCandidateCapacity, guardCandidateBytes,
    rowHashCapacity, faceHashCapacity, rowBytes, bandFaceBytes, stateBytes, cptParentBytes, hashBytes,
    frontierBytes, velocityBytes, provisionalVelocityBytes, pointAccumulatorBytes, pointStatusBytes,
    transientPowerFaceCapacity, transientPowerFaceBytes, transientPowerIncidenceBytes, transientPowerRowBytes,
    transitionAdjacencyCapacity, transitionAdjacencyBytes, transitionMetricBytes,
    powerFaceCapacity, powerVelocityScratchBytes, maximumDirectWorkgroups,
    gpuAllocatedBytes: rowBytes + bandFaceBytes + incidenceBytes + stateBytes + 2 * cptParentBytes + hashBytes
      + rowHashCapacity * 8 + frontierBytes + velocityBytes + provisionalVelocityBytes
      + pointAccumulatorBytes + pointStatusBytes + OCTREE_FACE_BAND_POINT_FIELD_CONTROL_BYTES
      + transientPowerFaceBytes + transientPowerIncidenceBytes + transientPowerRowBytes
      + OCTREE_FACE_BAND_TRANSIENT_CONTROL_BYTES
      + transitionAdjacencyBytes + transitionMetricBytes
      + guardCandidateBytes
      + powerVelocityScratchBytes + OCTREE_FACE_BAND_CONTROL_BYTES + OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES
      + OCTREE_FACE_BAND_POWER_PUBLICATION_CONTROL_BYTES + 96 };
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
  "fast-march",
  "power-publication",
] as const;
export type OctreeFaceBandEncodePhase = typeof OCTREE_FACE_BAND_ENCODE_PHASES[number];

/**
 * The encoded heap-pop maximum is the exact face-capacity bound: every LIVE
 * non-seed face is appended once and empty tail chunks become no-ops. A
 * post-bound GPU scan distinguishes scheduler exhaustion from disconnected
 * causal components. No face is committed while any requested face is unknown.
 * Fine bricks provide the bounded row-discovery frontier and current physical
 * signed distance wherever their narrow band contains a regular-face centroid.
 * Outside that finite stencil, coarse signed distance is fast-marched across
 * the transient owner graph and evaluated with the paper's cube/local-Delaunay
 * interpolant; row topology still comes from the owner map. This increment accepts uniform owners;
 * transition owners fail closed until their regular-face adjacency is emitted
 * from the catalog Delaunay tetrahedra already used by the Stage-B sampler.
 */
export class WebGPUOctreeFaceFastMarch {
  readonly plan: OctreeFaceBandGPUPlan;
  readonly control: GPUBuffer;
  readonly rows: GPUBuffer;
  readonly rowHash: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly faceHash: GPUBuffer;
  readonly incidence: GPUBuffer;
  readonly velocities: GPUBuffer;
  readonly state: GPUBuffer;
  readonly transitionAdjacency: GPUBuffer;
  readonly transitionControl: GPUBuffer;
  readonly transitionMetrics: GPUBuffer;
  readonly powerPublicationControl: GPUBuffer;
  readonly pointFieldControl: GPUBuffer;
  private readonly guardCandidates: GPUBuffer;
  private readonly globalRowHash: GPUBuffer;
  private readonly powerVelocityScratch: GPUBuffer;
  private readonly provisionalVelocities: GPUBuffer;
  private readonly pointAccumulator: GPUBuffer;
  private readonly pointStatus: GPUBuffer;
  private readonly transientPowerFaces: GPUBuffer;
  private readonly transientPowerIncidence: GPUBuffer;
  private readonly transientPowerRows: GPUBuffer;
  readonly transientPowerControl: GPUBuffer;
  private readonly cptParentsA: GPUBuffer;
  private readonly cptParentsB: GPUBuffer;
  private readonly indirect: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly sampleParams: GPUBuffer;
  private readonly repairParams: GPUBuffer;
  private readonly repairTopologyParams: GPUBuffer;
  private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;
  private readonly sampleClassifyPipeline: GPUComputePipeline;
  private readonly sampleEvaluatePipeline: GPUComputePipeline;
  private readonly sampleFinalizePipeline: GPUComputePipeline;
  private readonly preparePowerAdvectionRepairPipeline: GPUComputePipeline;
  private readonly repairPowerAdvectionPipeline: GPUComputePipeline;
  private readonly finalizePowerAdvectionPipeline: GPUComputePipeline;
  private readonly coldPowerAdvectionPipeline: GPUComputePipeline;
  private readonly faceBfsLayers: number;
  private readonly cptPlan: ReturnType<typeof planOctreeFaceBandCPT>;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, fine: WebGPUFineLevelSetBrickSource,
    wetRowCapacity: number, private readonly bandPhiRelaxationRounds: number,
    maximumCptGraphDepth: number, powerFaceCapacity?: number) {
    positive(bandPhiRelaxationRounds, "Band-phi relaxation round count");
    this.cptPlan = planOctreeFaceBandCPT(maximumCptGraphDepth);
    this.plan = planOctreeFaceBandGPU(wetRowCapacity, fine.plan.maximumResidentBricks,
      fine.plan.brickResolution, fine.plan.fineFactor, powerFaceCapacity, fine.plan.finestCellDimensions);
    this.faceBfsLayers = Math.min(8, bandPhiRelaxationRounds);
    const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    for (const [label, bytes] of [["row hash", this.plan.rowHashCapacity * 8],
      ["face hash", this.plan.faceHashCapacity * 8], ["faces", this.plan.bandFaceBytes],
      ["state", this.plan.stateBytes], ["transition adjacency", this.plan.transitionAdjacencyBytes],
      ["transition metrics", this.plan.transitionMetricBytes],
      ["transition guard candidates", this.plan.guardCandidateBytes],
      ["provisional velocities", this.plan.provisionalVelocityBytes],
      ["point accumulator", this.plan.pointAccumulatorBytes], ["point status", this.plan.pointStatusBytes],
      ["transient power faces", this.plan.transientPowerFaceBytes],
      ["transient power incidence", this.plan.transientPowerIncidenceBytes],
      ["transient power rows", this.plan.transientPowerRowBytes],
      ["power velocity scratch", this.plan.powerVelocityScratchBytes]] as const) {
      if (bytes > maximumBinding) throw new RangeError(`Face-band ${label} exceeds the adapter storage binding limit`);
    }
    if (this.plan.maximumDirectWorkgroups > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new RangeError("Face-band direct dispatch exceeds the adapter workgroup dimension limit");
    }
    this.control = storage(device, OCTREE_FACE_BAND_CONTROL_BYTES, "octree face-band control");
    this.rows = storage(device, this.plan.rowBytes, "octree face-band rows");
    this.rowHash = storage(device, this.plan.rowHashCapacity * 8, "octree face-band row hash");
    this.faces = storage(device, this.plan.bandFaceBytes, "octree face-band faces");
    this.faceHash = storage(device, this.plan.faceHashCapacity * 8, "octree face-band face hash");
    this.incidence = storage(device, this.plan.incidenceBytes, "octree face-band incidence");
    this.velocities = storage(device, this.plan.velocityBytes, "octree face-band row velocity");
    this.provisionalVelocities = storage(device, this.plan.provisionalVelocityBytes,
      "octree face-band provisional carrier velocity");
    this.pointAccumulator = storage(device, this.plan.pointAccumulatorBytes,
      "octree face-band generalized-face least-squares accumulator");
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
    this.state = storage(device, this.plan.stateBytes, "octree face-band march state");
    this.transitionAdjacency = storage(device, this.plan.transitionAdjacencyBytes,
      "octree catalog-Delaunay face-band adjacency");
    this.transitionControl = storage(device, OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES,
      "octree face-band transition control");
    this.transitionMetrics = storage(device, this.plan.transitionMetricBytes,
      "octree all-band catalog metrics");
    this.guardCandidates = storage(device, this.plan.guardCandidateBytes,
      "octree catalog-Delaunay guard candidates");
    this.globalRowHash = storage(device, this.plan.rowHashCapacity * 8, "octree face-band global-row hash");
    this.powerVelocityScratch = storage(device, this.plan.powerVelocityScratchBytes,
      "octree regular-to-power velocity scratch");
    this.powerPublicationControl = storage(device, OCTREE_FACE_BAND_POWER_PUBLICATION_CONTROL_BYTES,
      "octree regular-to-power publication control");
    this.cptParentsA = storage(device, this.plan.cptParentBytes, "octree face-band CPT parents A");
    this.cptParentsB = storage(device, this.plan.cptParentBytes, "octree face-band CPT parents B");
    this.indirect = storage(device, this.plan.frontierBytes,
      "octree face-band indirect dispatch", GPUBufferUsage.INDIRECT);
    this.params = device.createBuffer({ label: "octree face-band parameters", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampleParams = device.createBuffer({ label: "octree face-band sample parameters", size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.repairParams = device.createBuffer({ label: "retained face-band advection parameters", size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.repairTopologyParams = device.createBuffer({ label: "retained face-band topology parameters", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ label: "octree face-band topology and fast march", code: octreeFaceBandWGSL });
    const pipeline = (entryPoint: string, constants?: Record<string, number>) => device.createComputePipeline({ label: entryPoint, layout: "auto",
      compute: { module, entryPoint, ...(constants ? { constants } : {}) } });
    const pipelines: Record<string, GPUComputePipeline> = { prepare: pipeline("prepareFaceBand"), map: pipeline("mapFineBricksToBandRows"),
      emit: pipeline("emitBandFaces"), sampleFacePhi: pipeline("sampleBandFacePhi"),
      emitDeep: pipeline("emitDeepBandFaces"),
      initializeBandPhi: pipeline("initializeBandRowPhi"),
      seedBandPhiFaces: pipeline("seedBandRowPhiFromFaces"),
      extendBandPhi: pipeline("extendBandRowPhi"), commitBandPhi: pipeline("commitBandRowPhi"),
      sampleFaceCoarsePhi: pipeline("sampleBandFaceCoarsePhi"),
      reducePhiFailure: pipeline("reduceBandPhiFailure"),
      publishPhiFailure: pipeline("publishBandPhiFailure"),
      summarizeRowPhi: pipeline("summarizeBandRowPhi"), seedCentroids: pipeline("seedFaceCentroids"),
      retireFaceSlots: pipeline("retireBandFaceSlots"),
      initialize: pipeline("initializeFaceMarch"), linkCpt: pipeline("linkFaceClosestPoints"),
      jumpCpt: pipeline("jumpFaceClosestPoints"),
      resolveCpt: pipeline("resolveFaceClosestPoints"),
      prepareBfs: pipeline("prepareFaceBfsFallback"),
      validate: pipeline("validateFaceMarch"),
      reconstruct: pipeline("reconstructBandRowVelocity"),
      completeClosure: pipeline("completeClosureBandRowVelocity"), publish: pipeline("publishFaceBand"),
      completeDeepClosure: pipeline("completeDeepClosureBandRowVelocity"),
      seedMappedPowerRows: pipeline("seedMappedPowerRowVelocity"),
      initializeColdClosure: pipeline("initializeColdClosureBandVelocity"),
      initializeColdEndpoints: pipeline("initializeColdEndpointVelocity"),
      completeEndpoints: pipeline("completeEndpointBandVelocity"),
      reconstructDeep: pipeline("reconstructDeepBandRowVelocity"),
      reconstructSupport5: pipeline("reconstructSupport5BandRowVelocity"),
      reconstructSupport6: pipeline("reconstructSupport6BandRowVelocity"),
      seedOpenWorld: pipeline("seedOpenWorldNormal"),
      preparePointField: pipeline("prepareBandPointField"), preparePointRows: pipeline("prepareBandPointRows"),
      preparePointDispatch: pipeline("prepareBandPointDispatch"),
      prepareTransientPower: pipeline("prepareTransientBandPowerGraph"),
      emitTransientPower: pipeline("emitTransientBandPowerGraph"),
      sampleTransientPower: pipeline("sampleTransientBandPowerFaces"),
      validateTransientPower: pipeline("validateTransientBandPowerGraph"),
      publishTransientPower: pipeline("publishTransientBandPowerGraph"),
      accumulatePhysicalPoint: pipeline("accumulateBandTransientPowerLS"),
      solvePoint: pipeline("solveBandPowerLS"),
      validatePoint: pipeline("validateBandPointField"), publishPoint: pipeline("publishBandPointField"),
      prepareTransition: pipeline("prepareTransitionAdjacency"),
      clearBandPhiEdges: pipeline("clearBandPhiEdges"),
      recordBandPhiEdges: pipeline("recordBandPhiSupportEdges"),
      recordBandPhiEndpointEdges: pipeline("recordBandPhiEndpointEdges"),
      resetBandPhiCount: pipeline("resetBandPhiExtensionCount"),
      prepareSupport0Dispatch: pipeline("prepareSupport0Dispatch"),
      resolveTransition: pipeline("resolveCoreBandTopology"),
      clearSupportCandidates: pipeline("clearSupportCandidates"),
      enumerateSupport1: pipeline("enumerateSupport1Requests"),
      enumerateSupport2: pipeline("enumerateSupport2Requests"),
      enumerateSupport3: pipeline("enumerateSupport3Requests"),
      enumerateSupport4: pipeline("enumerateSupport4Requests"),
      enumerateCatalogEndpoints: pipeline("enumerateSupportEndpointRequests"),
      enumerateSupport6: pipeline("enumerateSupport6Requests"),
      resolveSupportOwners: pipeline("resolveSupportOwners"),
      insertSupport1: pipeline("insertSupport1Rows"),
      insertSupport2: pipeline("insertSupport2Rows"),
      insertSupport3: pipeline("insertSupport3NodeRows"),
      insertSupport4: pipeline("insertSupport4NodeRows"),
      insertSupport5: pipeline("insertSupport5NodeRows"),
      insertSupport6: pipeline("insertSupport6NodeRows"),
      auditNarrowBandOwners: pipeline("auditNarrowBandOwnerRows"),
      captureSupport1: pipeline("captureSupport1Boundary"),
      captureSupport2: pipeline("captureSupport2Boundary"),
      captureSupport3: pipeline("captureSupport3NodeBoundary"),
      captureSupport4: pipeline("captureSupport4NodeBoundary"),
      captureSupport5: pipeline("captureSupport5NodeBoundary"),
      captureSupport6: pipeline("captureSupport6NodeBoundary"),
      captureSupport7: pipeline("captureSupport7NodeBoundary"),
      resolveSupport1Topology: pipeline("resolveSupport1BandTopology"),
      resolveSupport2Topology: pipeline("resolveSupport2BandTopology"),
      resolveSupport3Topology: pipeline("resolveSupport3BandTopology"),
      resolveSupport4Topology: pipeline("resolveSupport4BandTopology"),
      resolveSupport5Topology: pipeline("resolveSupport5BandTopology"),
      resolveSupport6Topology: pipeline("resolveSupport6BandTopology"),
      resolveSupport3EndpointTopology: pipeline("resolveSupport3EndpointTopology"),
      enumerateEndpoints: pipeline("enumerateSupport3EndpointRequests"),
      insertEndpoints: pipeline("insertSupport3EndpointRows"),
      captureEndpoints: pipeline("captureSupport3EndpointBoundary"),
      transition: pipeline("buildTransitionAdjacency"), gateTransition: pipeline("gateTransitionTransfer"),
      transitionDeep: pipeline("buildDeepTransitionAdjacency"),
      transitionSupport5: pipeline("buildSupport5TransitionAdjacency"),
      transitionSupport6: pipeline("buildSupport6TransitionAdjacency"),
      indexGlobalRows: pipeline("indexBandGlobalRows"),
      preparePowerPublication: pipeline("preparePowerPublication"),
      mapPowerFaceBands: pipeline("mapPowerFaceBands"),
      interpolatePowerFaces: pipeline("interpolatePowerFaceVector"),
      projectPowerFaces: pipeline("projectPowerFaceVelocity"),
      publishPowerFaces: pipeline("publishPowerFaceVelocity"),
      commitPowerFaces: pipeline("commitPowerFaceVelocity") };
    for (let layer = 1; layer <= this.faceBfsLayers; layer += 1) {
      pipelines[`propagateBfs${layer}`] = pipeline("propagateFaceBfsLayer",
        { faceBfsLayer: layer, faceBfsCausal: 1 });
      pipelines[`propagateConnectivity${layer}`] = pipeline("propagateFaceBfsLayer",
        { faceBfsLayer: this.faceBfsLayers + layer, faceBfsCausal: 0 });
    }
    this.pipelines = Object.freeze(pipelines);
    // Recurring transport reads an immutable face-band publication. Compile it
    // separately so hashes, status words, and publication controls are ordinary
    // storage values; the mixed topology module necessarily declares those same
    // buffers atomic while it is constructing the publication.
    const sampleModule = device.createShaderModule({ label: "atomic-free octree face-band air sampler",
      code: makeOctreeFaceBandAirSampleWGSL() });
    const samplePipeline = (entryPoint: string) => device.createComputePipeline({ label: entryPoint,
      layout: "auto", compute: { module: sampleModule, entryPoint } });
    this.sampleClassifyPipeline = samplePipeline("classifyAirBandVelocity");
    this.sampleEvaluatePipeline = samplePipeline("evaluateAirBandVelocity");
    this.sampleFinalizePipeline = samplePipeline("finalizeAirBandVelocity");
    this.preparePowerAdvectionRepairPipeline = pipeline("preparePowerFaceAdvectionBandRepair");
    this.repairPowerAdvectionPipeline = pipeline("repairPowerFaceAdvectionFromBand");
    this.finalizePowerAdvectionPipeline = pipeline("finalizePowerFaceAdvectionFromBand");
    this.coldPowerAdvectionPipeline = pipeline("publishColdPowerFaceAdvection");
  }

  encode(encoder: GPUCommandEncoder, input: OctreeFaceBandInput): void {
    for (const phase of OCTREE_FACE_BAND_ENCODE_PHASES) this.encodePhase(encoder, input, phase);
  }

  encodePhase(encoder: GPUCommandEncoder, input: OctreeFaceBandInput,
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
    const siteHash = positive(input.siteHashCapacity, "Face-band site hash capacity");
    const powerVelocityGeneration = input.powerVelocityGeneration >>> 0;
    const tetrahedronHeaders = input.powerTopology.catalogTetrahedronHeaders;
    const tetrahedra = input.powerTopology.catalogTetrahedra;
    const tetrahedronVertices = input.powerTopology.catalogTetrahedronVertices;
    if (!tetrahedronHeaders || !tetrahedra || !tetrahedronVertices) {
      throw new RangeError("Face-band transitions require the catalog Delaunay buffers");
    }
    if ((siteHash & (siteHash - 1)) !== 0) throw new RangeError("Face-band site hash capacity must be a power of two");
    if (input.powerFaces.plan.faceCapacity > this.plan.powerFaceCapacity) {
      throw new RangeError("Face-band power-face publication capacity is smaller than its live source");
    }
    const bind = (pipeline: GPUComputePipeline, entries: readonly (readonly [number, GPUBuffer])[]) =>
      this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries.map(([binding, buffer]) =>
        ({ binding, resource: { buffer } })) });
    const run = (name: keyof typeof this.pipelines, entries: readonly (readonly [number, GPUBuffer])[],
      workgroups: number, pass: GPUComputePassEncoder, indirectOffset?: number) => {
      const pipeline = this.pipelines[name];
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind(pipeline, entries));
      if (indirectOffset === undefined) pass.dispatchWorkgroups(workgroups);
      else pass.dispatchWorkgroupsIndirect(this.indirect, indirectOffset);
    };
    const computePass = (label: string, encode: (pass: GPUComputePassEncoder) => void) => {
      const pass = encoder.beginComputePass({ label }); encode(pass); pass.end();
    };

    switch (phase) {
      case "topology-build": {
        const words = new Uint32Array(20);
        words.set([...dims, maximumLeaf, this.plan.rowCapacity, this.plan.faceCapacity,
          this.plan.rowHashCapacity, this.plan.faceHashCapacity, siteHash, this.plan.wetRowCapacity,
          input.generation >>> 0, this.cptPlan.maximumGraphDepth, this.plan.ownerCandidatesPerBrick,
          powerVelocityGeneration, OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW, OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW]);
        // Descriptor direction bits are x-, y-, z-, z+, y+, x+. The standard
        // container closes every plane except the optional open ceiling.
        words[16] = 0b10_1111 | (input.closedTop ? 0b01_0000 : 0);
        this.device.queue.writeBuffer(this.params, 0, words);
        // This checkpoint owns only row discovery. Clearing every later-phase
        // capacity here made the small sparse-t0 discovery pay for the full
        // face/march/publication allocation. Row payload is guarded by the
        // freshly reset count; only the two open-addressed indices need a
        // capacity reset before keys may be claimed again.
        encoder.clearBuffer(this.control); encoder.clearBuffer(this.rowHash);
        encoder.clearBuffer(this.globalRowHash);
        // These tiny downstream headers are diagnostics as well as gates.
        // Reset them now so a later fenced-phase failure cannot expose the
        // preceding generation's valid snapshot to release evidence.
        encoder.clearBuffer(this.transitionControl);
        encoder.clearBuffer(this.powerPublicationControl);
        computePass("Build Section 5 regular-face topology", (pass) => {
          run("prepare", [[0, this.params], [1, input.fine.params], [3, input.fine.worklist],
            [5, this.control], [9, input.powerVelocityControl], [18, this.indirect],
            [42, input.fineTopologyControl]], 1, pass);
          run("map", [[0, this.params], [1, input.fine.params], [2, input.fine.metadata], [3, input.fine.worklist],
            [4, input.siteIndex], [5, this.control], [6, this.rows], [7, this.rowHash],
            [25, input.coarsePhiDirectory], [26, input.owners], [42, input.fineTopologyControl]],
          0, pass, 0);
          run("indexGlobalRows", [[0, this.params], [5, this.control], [6, this.rows], [35, this.globalRowHash]],
            Math.ceil(this.plan.rowCapacity / 64), pass);
        });
        return;
      }
      case "transition-adjacency": {
        // Incidence payload is self-delimiting and overwritten on append; only
        // its per-row atomic counts require reset. Final endpoint capture
        // publishes a counted row-local face-slot prefix and retires only those
        // LIVE bits before emission, so the 64-byte face arena is never cleared.
        encoder.clearBuffer(this.incidence, 0, this.plan.rowCapacity * 4);
        let transitionPassIndex = 0;
        let pass = encoder.beginComputePass({ label: "Build Section 5 local Delaunay face adjacency · 0" });
        const synchronizeTransitionStorage = () => {
          pass.end(); transitionPassIndex += 1;
          pass = encoder.beginComputePass({
            label: `Build Section 5 local Delaunay face adjacency · ${transitionPassIndex}`,
          });
        };
          run("prepareTransition", [[0, this.params], [5, this.control], [32, this.transitionControl]], 1, pass);
          run("clearBandPhiEdges", [[5, this.control], [47, this.pointStatus]],
            Math.ceil(this.plan.rowCapacity / 64), pass);
          run("resolveTransition", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("prepareSupport0Dispatch", [[18, this.indirect], [32, this.transitionControl],
            [43, this.guardCandidates]], 1, pass);
          const supportBindings = [[0, this.params], [1, input.fine.params], [4, input.siteIndex], [5, this.control], [6, this.rows], [7, this.rowHash],
            [25, input.coarsePhiDirectory], [27, this.transitionMetrics], [32, this.transitionControl],
            [42, input.fineTopologyControl], [43, this.guardCandidates]] as const;
          const recordPhiEdgeBindings = [[0, this.params], [5, this.control], [7, this.rowHash],
            [32, this.transitionControl], [43, this.guardCandidates], [47, this.pointStatus],
            [53, this.transientPowerIncidence]] as const;
          const resolveOwnerBindings = [[0, this.params], [5, this.control], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [30, tetrahedronVertices], [32, this.transitionControl],
            [43, this.guardCandidates]] as const;
          run("clearSupportCandidates", [[43, this.guardCandidates]], 0, pass, 36);
          run("enumerateSupport1", [[6, this.rows], [27, this.transitionMetrics], [28, tetrahedronHeaders],
            [29, tetrahedra], [32, this.transitionControl], [43, this.guardCandidates]],
          0, pass, 24);
          synchronizeTransitionStorage();
          run("resolveSupportOwners", resolveOwnerBindings, 0, pass, 36);
          synchronizeTransitionStorage();
          run("insertSupport1", supportBindings, 0, pass, 36);
          synchronizeTransitionStorage();
          run("recordBandPhiEdges", recordPhiEdgeBindings, 0, pass, 36);
          run("captureSupport1", [[0, this.params], [5, this.control], [18, this.indirect],
            [32, this.transitionControl], [43, this.guardCandidates]], 1, pass);
          synchronizeTransitionStorage();
          run("resolveSupport1Topology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          0, pass, 48);
          run("clearSupportCandidates", [[43, this.guardCandidates]], 0, pass, 60);
          run("enumerateSupport2", [[6, this.rows], [27, this.transitionMetrics], [28, tetrahedronHeaders],
            [29, tetrahedra], [32, this.transitionControl], [43, this.guardCandidates]],
          0, pass, 48);
          synchronizeTransitionStorage();
          run("resolveSupportOwners", resolveOwnerBindings, 0, pass, 60);
          synchronizeTransitionStorage();
          run("insertSupport2", supportBindings, 0, pass, 60);
          synchronizeTransitionStorage();
          run("recordBandPhiEdges", recordPhiEdgeBindings, 0, pass, 60);
          run("captureSupport2", [[0, this.params], [5, this.control], [18, this.indirect],
            [32, this.transitionControl], [43, this.guardCandidates]], 1, pass);
          synchronizeTransitionStorage();
          run("resolveSupport2Topology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          0, pass, 72);
          run("clearSupportCandidates", [[43, this.guardCandidates]], 0, pass, 84);
          run("enumerateSupport3", [[6, this.rows], [27, this.transitionMetrics], [28, tetrahedronHeaders],
            [29, tetrahedra], [32, this.transitionControl], [43, this.guardCandidates]],
          0, pass, 72);
          synchronizeTransitionStorage();
          run("resolveSupportOwners", resolveOwnerBindings, 0, pass, 84);
          synchronizeTransitionStorage();
          run("insertSupport3", supportBindings, 0, pass, 84);
          synchronizeTransitionStorage();
          run("recordBandPhiEdges", recordPhiEdgeBindings, 0, pass, 84);
          run("captureSupport3", [[0, this.params], [5, this.control], [18, this.indirect],
            [32, this.transitionControl], [43, this.guardCandidates]], 1, pass);
          synchronizeTransitionStorage();
          run("resolveSupport3Topology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          0, pass, 96);
          run("clearSupportCandidates", [[43, this.guardCandidates]], 0, pass, 108);
          run("enumerateSupport4", [[6, this.rows], [27, this.transitionMetrics], [28, tetrahedronHeaders],
            [29, tetrahedra], [32, this.transitionControl], [43, this.guardCandidates]],
          0, pass, 96);
          synchronizeTransitionStorage();
          run("resolveSupportOwners", resolveOwnerBindings, 0, pass, 108);
          synchronizeTransitionStorage();
          run("insertSupport4", supportBindings, 0, pass, 108);
          synchronizeTransitionStorage();
          run("recordBandPhiEdges", recordPhiEdgeBindings, 0, pass, 108);
          run("captureSupport4", [[0, this.params], [5, this.control], [18, this.indirect],
            [32, this.transitionControl], [43, this.guardCandidates]], 1, pass);
          synchronizeTransitionStorage();
          run("resolveSupport4Topology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          0, pass, 120);
          run("clearSupportCandidates", [[43, this.guardCandidates]], 0, pass, 132);
          run("enumerateCatalogEndpoints", [[6, this.rows], [27, this.transitionMetrics],
            [28, tetrahedronHeaders], [29, tetrahedra], [32, this.transitionControl],
            [43, this.guardCandidates]], 0, pass, 120);
          synchronizeTransitionStorage();
          run("resolveSupportOwners", resolveOwnerBindings, 0, pass, 132);
          synchronizeTransitionStorage();
          run("insertSupport5", supportBindings, 0, pass, 132);
          synchronizeTransitionStorage();
          run("recordBandPhiEdges", recordPhiEdgeBindings, 0, pass, 132);
          run("captureSupport5", [[0, this.params], [5, this.control], [18, this.indirect],
            [32, this.transitionControl], [43, this.guardCandidates]], 1, pass);
          synchronizeTransitionStorage();
          run("resolveSupport5Topology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          0, pass, 144);
          run("clearSupportCandidates", [[43, this.guardCandidates]], 0, pass, 156);
          run("enumerateSupport6", [[6, this.rows], [27, this.transitionMetrics],
            [28, tetrahedronHeaders], [29, tetrahedra], [32, this.transitionControl],
            [43, this.guardCandidates]], 0, pass, 144);
          synchronizeTransitionStorage();
          run("resolveSupportOwners", resolveOwnerBindings, 0, pass, 156);
          synchronizeTransitionStorage();
          run("insertSupport6", supportBindings, 0, pass, 156);
          synchronizeTransitionStorage();
          run("recordBandPhiEdges", recordPhiEdgeBindings, 0, pass, 156);
          run("captureSupport6", [[0, this.params], [5, this.control], [18, this.indirect],
            [32, this.transitionControl], [43, this.guardCandidates]], 1, pass);
          synchronizeTransitionStorage();
          // Section 5 is a narrow-band construction. The six exact local
          // catalog-support tiers above are its bounded closure; promoting
          // every compact pressure owner here turns that band into the whole
          // liquid/air domain and exhausts the endpoint arena. Audit only the
          // owner rows actually requested by the interface closure.
          run("auditNarrowBandOwners", [[0, this.params], [6, this.rows], [7, this.rowHash],
            [26, input.owners], [32, this.transitionControl]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("captureSupport7", [[0, this.params], [5, this.control], [18, this.indirect],
            [32, this.transitionControl], [43, this.guardCandidates]], 1, pass);
          synchronizeTransitionStorage();
          run("resolveSupport6Topology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          0, pass, 168);
          run("clearSupportCandidates", [[43, this.guardCandidates]], 0, pass, 204);
          run("enumerateEndpoints", [[0, this.params], [6, this.rows], [26, input.owners],
            [32, this.transitionControl], [43, this.guardCandidates]],
          0, pass, 192);
          synchronizeTransitionStorage();
          run("insertEndpoints", supportBindings, 0, pass, 204);
          synchronizeTransitionStorage();
          run("recordBandPhiEndpointEdges", recordPhiEdgeBindings, 0, pass, 204);
          run("captureEndpoints", [[0, this.params], [5, this.control], [18, this.indirect],
            [32, this.transitionControl]], 1, pass);
          synchronizeTransitionStorage();
          run("retireFaceSlots", [[0, this.params], [12, this.faces], [32, this.transitionControl]],
            0, pass, 228);
          run("resolveSupport3EndpointTopology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("indexGlobalRows", [[0, this.params], [5, this.control], [6, this.rows],
            [35, this.globalRowHash]], Math.ceil(this.plan.rowCapacity / 64), pass);
          run("transition", [[0, this.params], [5, this.control], [6, this.rows], [7, this.rowHash],
            [26, input.owners],
            [27, this.transitionMetrics], [28, tetrahedronHeaders], [29, tetrahedra],
            [30, tetrahedronVertices], [31, this.transitionAdjacency], [32, this.transitionControl]],
          Math.ceil(this.plan.metricRowCapacity / 64), pass);
          // Aanjaneya et al. 2017 Section 5 keeps the fine SPGrid as a narrow
          // interface band and stores the distance outside it on the coarse
          // octree. S3-S6 and endpoint rows therefore close interpolation
          // stencils only: they must not become marched velocity/pressure
          // faces or recursively demand another ring of owner rows.
          run("emit", [[0, this.params], [5, this.control], [6, this.rows], [7, this.rowHash],
            [12, this.faces], [14, this.incidence], [26, input.owners], [32, this.transitionControl]],
          0, pass, 216);
          // Spatial face ownership can put an S0-S2 row's negative seam on
          // an S3+ closure row. Materialize only those inward seams so every
          // marched target has a complete regular-face stencil without
          // turning the deep support tiers into another marched graph.
          run("emitDeep", [[0, this.params], [5, this.control], [6, this.rows], [7, this.rowHash],
            [12, this.faces], [14, this.incidence], [26, input.owners], [32, this.transitionControl]],
          0, pass, 216);
          run("resetBandPhiCount", [[5, this.control]], 1, pass);
          // Paper Section 5 marches regular faces in closest-interface order.
          // Give every emitted live face its signed-distance value at the
          // actual face centroid from the same current fine SPGrid generation.
          run("sampleFacePhi", [[0, this.params], [1, input.fine.params], [2, input.fine.metadata],
            [5, this.control], [8, input.fine.flags], [12, this.faces], [24, input.fine.phi],
            [51, input.fine.hash]], 0, pass, 228);
          // The fine SPGrid is intentionally narrow. Build the paper's coarse
          // signed-distance field on the complete transient owner graph before
          // evaluating dry regular-face centroids. Existing row-velocity
          // scratch is reused here and cleared before the later velocity march.
          run("initializeBandPhi", [[0, this.params], [1, input.fine.params], [2, input.fine.metadata],
            [5, this.control], [6, this.rows], [8, input.fine.flags], [19, this.velocities],
            [24, input.fine.phi], [25, input.coarsePhiDirectory], [42, input.fineTopologyControl],
            [51, input.fine.hash]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("seedBandPhiFaces", [[0, this.params], [1, input.fine.params], [5, this.control],
            [6, this.rows], [12, this.faces], [14, this.incidence], [19, this.velocities]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          let currentPhi = this.velocities, nextPhi = this.provisionalVelocities;
          for (let round = 0; round < this.bandPhiRelaxationRounds; round += 1) {
            run("extendBandPhi", [[0, this.params], [1, input.fine.params], [5, this.control], [6, this.rows],
              [12, this.faces], [14, this.incidence], [19, currentPhi], [27, this.transitionMetrics],
              [31, this.transitionAdjacency], [44, nextPhi], [47, this.pointStatus],
              [53, this.transientPowerIncidence]], Math.ceil(this.plan.rowCapacity / 64), pass);
            [currentPhi, nextPhi] = [nextPhi, currentPhi];
          }
          run("commitBandPhi", [[5, this.control], [6, this.rows], [19, currentPhi],
            [32, this.transitionControl]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("sampleFaceCoarsePhi", [[0, this.params], [5, this.control], [6, this.rows], [7, this.rowHash], [12, this.faces],
            [27, this.transitionMetrics], [28, tetrahedronHeaders], [29, tetrahedra],
            [30, tetrahedronVertices], [31, this.transitionAdjacency], [32, this.transitionControl]],
          0, pass, 228);
          run("reducePhiFailure", [[12, this.faces], [32, this.transitionControl]],
            0, pass, 228);
          run("publishPhiFailure", [[12, this.faces], [32, this.transitionControl]],
            0, pass, 228);
          run("summarizeRowPhi", [[0, this.params], [5, this.control], [6, this.rows], [12, this.faces],
            [14, this.incidence], [32, this.transitionControl]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          // Face emission can still discover a missing endpoint/capacity
          // fault. Publish transition readiness only after that final writer.
          run("gateTransition", [[5, this.control], [32, this.transitionControl]], 1, pass);
        pass.end();
        return;
      }
      case "fast-march":
        // The initialize kernel rewrites state for every live face. Frontier
        // payload is likewise count-delimited, so reset only the queue heads.
        encoder.clearBuffer(this.provisionalVelocities);
        encoder.clearBuffer(this.velocities);
        encoder.clearBuffer(this.pointAccumulator);
        encoder.clearBuffer(this.pointStatus);
        encoder.clearBuffer(this.pointFieldControl);
        computePass("Extrapolate Section 5 regular-face velocities", (pass) => {
          run("seedCentroids", [[0, this.params], [4, input.siteIndex], [6, this.rows],
            [10, input.powerRowVelocities], [12, this.faces], [27, this.transitionMetrics],
            [28, tetrahedronHeaders], [29, tetrahedra], [30, tetrahedronVertices]],
          0, pass, 228);
          run("seedOpenWorld", [[5, this.control], [6, this.rows],
            [12, this.faces], [37, input.powerFaces.faces], [38, input.powerFaces.faceNormals],
            [49, input.powerFaces.incidenceRows], [50, input.powerFaces.incidence]],
          0, pass, 228);
          run("initialize", [[0, this.params], [5, this.control], [12, this.faces], [15, this.state]],
            0, pass, 228);
          run("linkCpt", [[0, this.params], [5, this.control], [6, this.rows],
            [12, this.faces], [14, this.incidence], [15, this.state],
            [27, this.transitionMetrics], [31, this.transitionAdjacency], [32, this.transitionControl],
            [56, this.cptParentsA]],
          0, pass, 228);
          let currentParents = this.cptParentsA;
          let nextParents = this.cptParentsB;
          for (let round = 0; round < this.cptPlan.jumpRounds; round += 1) {
            run("jumpCpt", [[0, this.params], [5, this.control], [12, this.faces], [15, this.state],
              [56, currentParents], [57, nextParents]], 0, pass, 228);
            [currentParents, nextParents] = [nextParents, currentParents];
          }
          run("resolveCpt", [[0, this.params], [5, this.control], [12, this.faces], [15, this.state],
            [56, currentParents]],
            0, pass, 228);
          run("prepareBfs", [[0, this.params], [5, this.control], [12, this.faces], [15, this.state]],
            0, pass, 228);
          for (let layer = 1; layer <= this.faceBfsLayers; layer += 1) {
            run(`propagateBfs${layer}`, [[0, this.params], [5, this.control], [6, this.rows], [12, this.faces],
              [14, this.incidence], [15, this.state], [27, this.transitionMetrics],
              [31, this.transitionAdjacency], [32, this.transitionControl]],
            0, pass, 228);
          }
          // Face-centred samples of an otherwise valid signed-distance field
          // can form a shallow positive-air local minimum on the mixed-axis
          // regular-face graph. The causal pass above cannot leave that basin,
          // even though its surrounding faces already carry seed-rooted
          // velocities. Finish only those still-UNKNOWN components with a
          // bounded, deterministic connectivity BFS. Validation below remains
          // strict: a genuinely disconnected component still rejects.
          for (let layer = 1; layer <= this.faceBfsLayers; layer += 1) {
            run(`propagateConnectivity${layer}`, [[0, this.params], [5, this.control], [6, this.rows],
              [12, this.faces], [14, this.incidence], [15, this.state], [27, this.transitionMetrics],
              [31, this.transitionAdjacency], [32, this.transitionControl]],
            0, pass, 228);
          }
          run("validate", [[0, this.params], [5, this.control], [6, this.rows], [12, this.faces],
            [14, this.incidence], [15, this.state], [27, this.transitionMetrics],
            [31, this.transitionAdjacency], [32, this.transitionControl]],
            0, pass, 228);
          run("reconstruct", [[0, this.params], [5, this.control], [6, this.rows], [12, this.faces],
            [14, this.incidence], [15, this.state], [19, this.velocities], [32, this.transitionControl],
            [44, this.provisionalVelocities], [48, this.pointFieldControl]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("reconstructDeep", [[0, this.params], [6, this.rows], [12, this.faces],
            [14, this.incidence], [15, this.state], [19, this.velocities], [32, this.transitionControl],
            [44, this.provisionalVelocities]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("reconstructSupport5", [[0, this.params], [6, this.rows], [12, this.faces],
            [14, this.incidence], [15, this.state], [19, this.velocities], [32, this.transitionControl],
            [44, this.provisionalVelocities]], Math.ceil(this.plan.rowCapacity / 64), pass);
          run("reconstructSupport6", [[0, this.params], [6, this.rows], [12, this.faces],
            [14, this.incidence], [15, this.state], [19, this.velocities], [32, this.transitionControl],
            [44, this.provisionalVelocities]], Math.ceil(this.plan.rowCapacity / 64), pass);
          run("completeClosure", [[0, this.params], [6, this.rows], [12, this.faces],
            [14, this.incidence], [15, this.state], [19, this.velocities], [32, this.transitionControl],
            [44, this.provisionalVelocities]], Math.ceil(this.plan.rowCapacity / 64), pass);
          run("completeDeepClosure", [[0, this.params], [6, this.rows], [12, this.faces],
            [14, this.incidence], [15, this.state], [19, this.velocities], [32, this.transitionControl],
            [44, this.provisionalVelocities]], Math.ceil(this.plan.rowCapacity / 64), pass);
          run("initializeColdClosure", [[0, this.params], [6, this.rows], [19, this.velocities],
            [32, this.transitionControl], [44, this.provisionalVelocities]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("completeEndpoints", [[0, this.params], [6, this.rows], [7, this.rowHash],
            [19, this.velocities], [32, this.transitionControl], [44, this.provisionalVelocities]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("initializeColdEndpoints", [[0, this.params], [6, this.rows], [19, this.velocities],
            [32, this.transitionControl], [44, this.provisionalVelocities]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          // Aanjaneya et al. (2017), Section 5 first constructs a full vector
          // at every liquid octree-cell centre from the generalized power-face
          // normal velocities.  The regular-face march only extrapolates that
          // field outside the liquid.  Restore those authoritative vectors
          // after support closure so an auxiliary zero-incidence support row
          // cannot replace a real power-cell value.
          run("seedMappedPowerRows", [[0, this.params], [5, this.control], [6, this.rows],
            [9, input.powerVelocityControl], [10, input.powerRowVelocities],
            [19, this.velocities], [32, this.transitionControl], [44, this.provisionalVelocities]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("publish", [[5, this.control]], 1, pass);
        });
        return;
      case "power-publication":
        // This graph is a fixed-address, counted publication. Each live row
        // overwrites its row descriptor and every catalog incidence slot has
        // exactly one current writer. The emitter retires only active face
        // flags, leaving stale capacity tails unreachable without arena clears.
        computePass("Publish Section 5 regular velocities to power faces", (pass) => {
          run("prepareTransientPower", [[0, this.params], [32, this.transitionControl],
            [52, this.transientPowerFaces], [53, this.transientPowerIncidence],
            [54, this.transientPowerRows], [55, this.transientPowerControl]], 1, pass);
          run("emitTransientPower", [[0, this.params], [6, this.rows], [7, this.rowHash],
            [11, input.powerTopology.catalogEntryHeaders], [27, this.transitionMetrics],
            [32, this.transitionControl], [45, input.powerTopology.catalogFaces], [52, this.transientPowerFaces],
            [53, this.transientPowerIncidence], [54, this.transientPowerRows],
            [55, this.transientPowerControl]], Math.ceil(this.plan.rowCapacity / 64), pass);
          run("sampleTransientPower", [[0, this.params], [6, this.rows],
            [7, this.rowHash], [27, this.transitionMetrics], [28, tetrahedronHeaders], [29, tetrahedra],
            [30, tetrahedronVertices], [44, this.provisionalVelocities], [52, this.transientPowerFaces],
            [55, this.transientPowerControl]], Math.ceil(this.plan.rowCapacity / 64), pass);
          run("validateTransientPower", [[52, this.transientPowerFaces], [53, this.transientPowerIncidence],
            [54, this.transientPowerRows], [55, this.transientPowerControl]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("publishTransientPower", [[0, this.params], [55, this.transientPowerControl]], 1, pass);
          // The paper's final cell-centre field is reconstructed only from the
          // complete transient all-band physical generalized-face graph.
          run("preparePointField", [[0, this.params], [5, this.control], [32, this.transitionControl],
            [48, this.pointFieldControl], [55, this.transientPowerControl]], 1, pass);
          run("preparePointDispatch", [[18, this.indirect], [48, this.pointFieldControl]], 1, pass);
          run("preparePointRows", [[6, this.rows], [46, this.pointAccumulator], [48, this.pointFieldControl]],
          0, pass, 24);
          run("accumulatePhysicalPoint", [[0, this.params], [46, this.pointAccumulator],
            [48, this.pointFieldControl], [52, this.transientPowerFaces],
            [53, this.transientPowerIncidence], [54, this.transientPowerRows],
            [55, this.transientPowerControl]],
          0, pass, 24);
          run("solvePoint", [[19, this.velocities], [46, this.pointAccumulator], [47, this.pointStatus],
            [48, this.pointFieldControl]], 0, pass, 24);
          run("validatePoint", [[47, this.pointStatus], [48, this.pointFieldControl]],
          0, pass, 24);
          run("publishPoint", [[0, this.params], [48, this.pointFieldControl],
            [55, this.transientPowerControl]], 1, pass);
          // Only the completed point field is allowed to return to the
          // production power faces. This breaks the former circular dependency
          // where those faces were required before the point field existed.
          run("preparePowerPublication", [[0, this.params], [5, this.control], [32, this.transitionControl],
            [36, input.powerFaces.control], [37, input.powerFaces.faces], [38, input.powerFaces.faceNormals],
            [39, input.powerFaces.faceCentroids], [40, this.powerVelocityScratch],
            [41, this.powerPublicationControl], [48, this.pointFieldControl]], 1, pass);
          run("mapPowerFaceBands", [[0, this.params], [35, this.globalRowHash],
            [37, input.powerFaces.faces], [40, this.powerVelocityScratch], [41, this.powerPublicationControl]],
          Math.ceil(this.plan.powerFaceCapacity / 64), pass);
          run("interpolatePowerFaces", [[0, this.params], [1, input.fine.params], [6, this.rows], [7, this.rowHash],
            [19, this.velocities], [27, this.transitionMetrics], [28, tetrahedronHeaders], [29, tetrahedra],
            [30, tetrahedronVertices], [39, input.powerFaces.faceCentroids], [40, this.powerVelocityScratch],
            [41, this.powerPublicationControl]], Math.ceil(this.plan.powerFaceCapacity / 64), pass);
          run("projectPowerFaces", [[0, this.params], [37, input.powerFaces.faces], [38, input.powerFaces.faceNormals],
            [40, this.powerVelocityScratch], [41, this.powerPublicationControl]],
          Math.ceil(this.plan.powerFaceCapacity / 64), pass);
          run("publishPowerFaces", [[5, this.control], [41, this.powerPublicationControl]], 1, pass);
          run("commitPowerFaces", [[37, input.powerFaces.faces], [40, this.powerVelocityScratch],
            [41, this.powerPublicationControl]], Math.ceil(this.plan.powerFaceCapacity / 64), pass);
          // S3 nodes are exact face-derived carrier vectors for trajectory
          // interpolation. Resolve their catalog descriptors only after every
          // scalar/face consumer has finished, so they can anchor a containing
          // Delaunay tetrahedron without entering the marched S0-S2 graph.
          run("resolveSupport3Topology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          0, pass, 60);
          run("resolveSupport4Topology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("resolveSupport5Topology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("resolveSupport6Topology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
          run("resolveSupport3EndpointTopology", [[0, this.params], [6, this.rows], [26, input.owners],
            [27, this.transitionMetrics], [32, this.transitionControl],
            [33, input.powerTopology.sameOrFinerDirect], [34, input.powerTopology.sameOrCoarserDirect]],
          Math.ceil(this.plan.rowCapacity / 64), pass);
        });
        return;
      default: phase satisfies never;
    }
  }

  /** Complete Stage-B for positive-air rows and exact catalog-simplex coverage
   * misses using the published Section 5 regular-face point field. */
  encodeAirSamples(encoder: GPUCommandEncoder, positions: GPUBuffer | GPUBufferBinding,
    results: GPUBuffer, statuses: GPUBuffer,
    options: OctreeFaceBandSampleOptions,
    boundary?: OctreeFaceBandAirSampleBoundary): GPUCommandEncoder {
    this.assertLive();
    const count = positive(options.queryCount, "Face-band sample count");
    if (statuses.size < count * 4) {
      throw new RangeError("Face-band sample status buffer is smaller than the query count");
    }
    if (!Number.isFinite(options.physicalCellSize) || options.physicalCellSize <= 0) {
      throw new RangeError("Face-band sample physical cell size must be finite and positive");
    }
    const data = new ArrayBuffer(48), words = new Uint32Array(data), floats = new Float32Array(data);
    words.set([...options.dimensions, positive(options.maximumLeafSize, "Face-band sample maximum leaf"), 0,
      count, this.plan.rowHashCapacity, this.plan.rowCapacity]); floats[8] = options.physicalCellSize;
    words[9] = options.fineGeneration >>> 0;
    this.device.queue.writeBuffer(this.sampleParams, 0, data);
    const tetrahedronHeaders = options.powerTopology.catalogTetrahedronHeaders;
    const tetrahedra = options.powerTopology.catalogTetrahedra;
    const tetrahedronVertices = options.powerTopology.catalogTetrahedronVertices;
    if (!tetrahedronHeaders || !tetrahedra || !tetrahedronVertices) {
      throw new RangeError("Face-band point sampling requires current catalog Delaunay buffers");
    }
    const binding = (buffer: GPUBuffer | GPUBufferBinding): GPUBufferBinding => "buffer" in buffer ? buffer : { buffer };
    const classifyEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 20, resource: { buffer: this.sampleParams } }, { binding: 21, resource: binding(positions) },
      { binding: 5, resource: { buffer: this.control } }, { binding: 6, resource: { buffer: this.rows } },
      { binding: 7, resource: { buffer: this.rowHash } },
      { binding: 22, resource: { buffer: results } }, { binding: 23, resource: { buffer: statuses } },
      { binding: 26, resource: { buffer: options.owners } },
      { binding: 32, resource: { buffer: this.transitionControl } },
      { binding: 48, resource: { buffer: this.pointFieldControl } },
    ];
    const evaluateEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.params } }, { binding: 6, resource: { buffer: this.rows } },
      { binding: 7, resource: { buffer: this.rowHash } }, { binding: 19, resource: { buffer: this.velocities } },
      { binding: 20, resource: { buffer: this.sampleParams } }, { binding: 21, resource: binding(positions) },
      { binding: 22, resource: { buffer: results } }, { binding: 23, resource: { buffer: statuses } },
      { binding: 27, resource: { buffer: this.transitionMetrics } },
      { binding: 28, resource: { buffer: tetrahedronHeaders } }, { binding: 29, resource: { buffer: tetrahedra } },
      { binding: 30, resource: { buffer: tetrahedronVertices } },
    ];
    const finalizeEntries: GPUBindGroupEntry[] = [
      { binding: 5, resource: { buffer: this.control } }, { binding: 20, resource: { buffer: this.sampleParams } },
      { binding: 23, resource: { buffer: statuses } }, { binding: 48, resource: { buffer: this.pointFieldControl } },
    ];
    const dispatch = (pass: GPUComputePassEncoder, pipeline: GPUComputePipeline,
      entries: GPUBindGroupEntry[], workgroups: number) => {
      pass.setPipeline(pipeline); pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0), entries })); pass.dispatchWorkgroups(workgroups);
    };
    const queryWorkgroups = Math.ceil(count / 64);
    if (!boundary) {
      const pass = encoder.beginComputePass({ label: "Sample extrapolated air-side octree velocity" });
      dispatch(pass, this.sampleClassifyPipeline, classifyEntries, queryWorkgroups);
      dispatch(pass, this.sampleEvaluatePipeline, evaluateEntries, queryWorkgroups);
      dispatch(pass, this.sampleFinalizePipeline, finalizeEntries, queryWorkgroups);
      pass.end();
      return encoder;
    }
    const splitStage = (stage: OctreeFaceBandAirSampleStage, pipeline: GPUComputePipeline,
      entries: GPUBindGroupEntry[]) => {
      const pass = encoder.beginComputePass({ label: `Profile ${stage}` });
      dispatch(pass, pipeline, entries, queryWorkgroups);
      pass.end();
      encoder = boundary(stage, encoder);
    };
    splitStage("classifyAirBandVelocity", this.sampleClassifyPipeline, classifyEntries);
    splitStage("evaluateAirBandVelocity", this.sampleEvaluatePipeline, evaluateEntries);
    splitStage("finalizeAirBandVelocity", this.sampleFinalizePipeline, finalizeEntries);
    return encoder;
  }

  /** Complete a recurrent Section 5 characteristic wherever the compact
   * liquid-row interpolant reported that its retained mesh had no owner.  The
   * old face band is still live at this scheduling point and supplies the
   * paper's extrapolated air-side cube/Delaunay field. */
  encodeRepairPowerFaceAdvection(encoder: GPUCommandEncoder, input: {
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
    const data = new ArrayBuffer(48), words = new Uint32Array(data), floats = new Float32Array(data);
    words.set([...input.dimensions, positive(input.maximumLeafSize, "Old face-band maximum leaf"),
      0, 0, this.plan.rowHashCapacity, this.plan.rowCapacity]);
    floats[8] = input.physicalCellSize;
    words[9] = input.fineGeneration >>> 0;
    floats[10] = input.timestep;
    words[11] = input.powerGeneration >>> 0;
    this.device.queue.writeBuffer(this.repairParams, 0, data);
    // The recurrent face count is produced by the GPU topology build in the
    // preceding pass.  Copy that exact transaction word into this invocation's
    // parameters; a host capacity is not a live Section 5 topology bound.
    encoder.copyBufferToBuffer(input.advectionControl, 8 * 4, this.repairParams, 5 * 4, 4);
    const topologyWords = new Uint32Array(20);
    topologyWords.set([...input.dimensions, input.maximumLeafSize, this.plan.rowCapacity,
      this.plan.faceCapacity, this.plan.rowHashCapacity, this.plan.faceHashCapacity,
      input.faces.plan.hashCapacity, this.plan.wetRowCapacity, input.fineGeneration >>> 0,
      this.cptPlan.maximumGraphDepth, this.plan.ownerCandidatesPerBrick,
      input.powerGeneration >>> 0, OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW,
      OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW]);
    topologyWords[16] = 0b10_1111 | (input.closedTop ? 0b01_0000 : 0);
    this.device.queue.writeBuffer(this.repairTopologyParams, 0, topologyWords);
    const common: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.repairTopologyParams } },
      { binding: 5, resource: { buffer: this.control } },
      { binding: 6, resource: { buffer: this.rows } },
      { binding: 7, resource: { buffer: this.rowHash } },
      { binding: 19, resource: { buffer: this.velocities } },
      { binding: 20, resource: { buffer: this.repairParams } },
      { binding: 27, resource: { buffer: this.transitionMetrics } },
    ];
    // Catalog buffers are owned by the same immutable topology authority used
    // when this old band was published.
    const topology = input.powerTopology;
    if (!topology?.catalogTetrahedronHeaders || !topology.catalogTetrahedra
      || !topology.catalogTetrahedronVertices) {
      throw new RangeError("Old face-band advection repair requires the Section 5 Delaunay catalog");
    }
    common.push(
      { binding: 28, resource: { buffer: topology.catalogTetrahedronHeaders } },
      { binding: 29, resource: { buffer: topology.catalogTetrahedra } },
      { binding: 30, resource: { buffer: topology.catalogTetrahedronVertices } },
      { binding: 32, resource: { buffer: this.transitionControl } },
      { binding: 36, resource: { buffer: input.faces.control } },
      { binding: 37, resource: { buffer: input.faces.faces } },
      { binding: 38, resource: { buffer: input.faces.faceNormals } },
      { binding: 39, resource: { buffer: input.faces.faceCentroids } },
      { binding: 48, resource: { buffer: this.pointFieldControl } },
      { binding: 58, resource: { buffer: input.advectionControl } },
      { binding: 59, resource: { buffer: input.seedControl } },
    );
    const bind = (pipeline: GPUComputePipeline, bindings: readonly number[]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: common.filter((entry) => bindings.includes(entry.binding)),
    });
    // Cold start is a property of the power mesh, not of the retained fine
    // band. The first recurring power rebuild legitimately traces through
    // fine-band generation 2, so keying this branch to `fineGeneration`
    // erased every advected generalized-face velocity during the first
    // physical steps. Section 5's characteristic is old power mesh -> new
    // power mesh; only power generation 1 has no predecessor.
    if (input.powerGeneration === 1) {
      const cold = encoder.beginComputePass({ label: "Publish exact cold-start power-face advection" });
      cold.setPipeline(this.coldPowerAdvectionPipeline);
      cold.setBindGroup(0, bind(this.coldPowerAdvectionPipeline, [37, 58, 59]));
      cold.dispatchWorkgroups(1);
      cold.end();
      return;
    }
    const prepare = encoder.beginComputePass({ label: "Prepare old-mesh face repair from Section 5 air band" });
    prepare.setPipeline(this.preparePowerAdvectionRepairPipeline);
    prepare.setBindGroup(0, bind(this.preparePowerAdvectionRepairPipeline, [5, 20, 36, 48, 58, 59]));
    prepare.dispatchWorkgroups(1);
    prepare.end();
    const repair = encoder.beginComputePass({ label: "Repair old-mesh face advection from Section 5 air band" });
    repair.setPipeline(this.repairPowerAdvectionPipeline);
    repair.setBindGroup(0, bind(this.repairPowerAdvectionPipeline,
      [0, 6, 7, 19, 20, 27, 28, 29, 30, 37, 38, 39]));
    repair.dispatchWorkgroups(Math.ceil(this.plan.powerFaceCapacity / 64));
    repair.end();
    const finalize = encoder.beginComputePass({ label: "Publish repaired old-mesh face advection" });
    finalize.setPipeline(this.finalizePowerAdvectionPipeline);
    finalize.setBindGroup(0, bind(this.finalizePowerAdvectionPipeline, [39, 58, 59]));
    finalize.dispatchWorkgroups(1);
    finalize.end();
  }

  get source(): OctreeFaceBandSource { return { plan: this.plan, control: this.control, rows: this.rows,
    rowHash: this.rowHash, faces: this.faces, incidence: this.incidence, velocities: this.velocities,
    state: this.state,
    transitionAdjacency: this.transitionAdjacency, transitionControl: this.transitionControl,
    transitionMetrics: this.transitionMetrics, powerPublicationControl: this.powerPublicationControl,
    pointFieldControl: this.pointFieldControl, transientPowerControl: this.transientPowerControl }; }
  /** Bounded failure-only readback for a disconnected regular face. */
  async readDisconnectedFaceFailure(index: number) {
    if (!Number.isSafeInteger(index) || index < 0 || index >= this.plan.faceCapacity) return undefined;
    const faceBytes = OCTREE_FACE_BAND_FACE_BYTES;
    const stateBytes = OCTREE_FACE_BAND_STATE_BYTES;
    const rowBytes = OCTREE_FACE_BAND_ROW_BYTES;
    const incidenceWords = 1 + OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW;
    const incidenceBytes = incidenceWords * 4;
    const totalBytes = faceBytes + stateBytes + 2 * rowBytes + 2 * incidenceBytes;
    const readback = this.device.createBuffer({ label: "Face-band disconnected-face failure", size: totalBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const faceProbe = this.device.createBuffer({ label: "Face-band disconnected-face row probe", size: 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let encoder = this.device.createCommandEncoder({ label: "Read face-band disconnected face" });
    encoder.copyBufferToBuffer(this.faces, index * faceBytes, readback, 0, faceBytes);
    encoder.copyBufferToBuffer(this.state, index * stateBytes, readback, faceBytes, stateBytes);
    encoder.copyBufferToBuffer(this.faces, index * faceBytes, faceProbe, 0, 8);
    this.device.queue.submit([encoder.finish()]);
    try {
      await faceProbe.mapAsync(GPUMapMode.READ);
      const rowIndices = new Uint32Array(faceProbe.getMappedRange().slice(0));
      faceProbe.unmap();
      encoder = this.device.createCommandEncoder({ label: "Read face-band disconnected rows" });
      for (let side = 0; side < 2; side += 1) {
        const row = rowIndices[side];
        if (row >= this.plan.rowCapacity) continue;
        const rowOffset = faceBytes + stateBytes + side * rowBytes;
        encoder.copyBufferToBuffer(this.rows, row * rowBytes, readback, rowOffset, rowBytes);
        const incidenceOffset = faceBytes + stateBytes + 2 * rowBytes + side * incidenceBytes;
        encoder.copyBufferToBuffer(this.incidence, row * 4, readback, incidenceOffset, 4);
        encoder.copyBufferToBuffer(this.incidence,
          (this.plan.rowCapacity + row * OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW) * 4,
          readback, incidenceOffset + 4, incidenceBytes - 4);
      }
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const bytes = readback.getMappedRange().slice(0);
      const words = new Uint32Array(bytes), floats = new Float32Array(bytes);
      const faceWordCount = faceBytes / 4, stateWordCount = stateBytes / 4, rowWordCount = rowBytes / 4;
      const rowBase = faceWordCount + stateWordCount;
      const incidenceBase = rowBase + 2 * rowWordCount;
      return {
        index,
        face: { words: Array.from(words.slice(0, faceWordCount)), phi: floats[12], area: floats[13] },
        state: Array.from(words.slice(faceWordCount, faceWordCount + stateWordCount)),
        rows: Array.from({ length: 2 }, (_, side) => ({ index: rowIndices[side],
          words: Array.from(words.slice(rowBase + side * rowWordCount, rowBase + (side + 1) * rowWordCount)),
          incidence: Array.from(words.slice(incidenceBase + side * incidenceWords,
            incidenceBase + (side + 1) * incidenceWords)) })),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      if (faceProbe.mapState === "mapped") faceProbe.unmap();
      readback.destroy(); faceProbe.destroy();
    }
  }
  /** Bounded QA readback for the retained-band row named by an advection failure. */
  async readBandRowFailure(index: number) {
    if (!Number.isSafeInteger(index) || index < 0 || index >= this.plan.rowCapacity) return undefined;
    const incidenceWords = 1 + OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW;
    const fixedBytes = OCTREE_FACE_BAND_ROW_BYTES + 16 + 16 + 16 + incidenceWords * 4;
    const fixed = this.device.createBuffer({ label: "Face-band retained-row failure", size: fixedBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder({ label: "Read face-band retained row" });
    let offset = 0;
    encoder.copyBufferToBuffer(this.rows, index * OCTREE_FACE_BAND_ROW_BYTES,
      fixed, offset, OCTREE_FACE_BAND_ROW_BYTES); offset += OCTREE_FACE_BAND_ROW_BYTES;
    encoder.copyBufferToBuffer(this.transitionMetrics, index * 16, fixed, offset, 16); offset += 16;
    encoder.copyBufferToBuffer(this.velocities, index * 16, fixed, offset, 16); offset += 16;
    encoder.copyBufferToBuffer(this.provisionalVelocities, index * 16, fixed, offset, 16); offset += 16;
    encoder.copyBufferToBuffer(this.incidence, index * 4, fixed, offset, 4);
    encoder.copyBufferToBuffer(this.incidence,
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
      liveFaceIndices.forEach((face, local) => faceEncoder.copyBufferToBuffer(this.faces,
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
    for (const buffer of [this.control, this.rows, this.rowHash, this.faces, this.faceHash, this.incidence,
      this.velocities, this.provisionalVelocities, this.pointAccumulator, this.pointStatus, this.pointFieldControl,
      this.transientPowerFaces, this.transientPowerIncidence, this.transientPowerRows, this.transientPowerControl,
      this.state, this.transitionAdjacency, this.transitionControl, this.transitionMetrics, this.guardCandidates,
      this.globalRowHash, this.powerVelocityScratch, this.powerPublicationControl,
      this.cptParentsA, this.cptParentsB, this.indirect, this.params, this.sampleParams,
      this.repairParams, this.repairTopologyParams]) buffer.destroy(); }
  private assertLive(): void { if (this.destroyed) throw new Error("Octree face-band marcher is destroyed"); }
}

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
 * The topology shader below uses atomics while constructing its hashes and
 * controls. Once published, this dedicated module binds the identical bytes as
 * ordinary read-only structures and gives every query exclusive ownership of
 * its result/status slot. No atomic operation is reachable from these entries.
 */
export function makeOctreeFaceBandAirSampleWGSL(): string {
  const functions = [
    "finite", "hash", "cell", "invalidOwner", "decodeOwner", "decodePagedOwner", "residentCanonicalOwner",
    "ownerAt", "coord", "negativeBoundaryBit", "positiveBoundaryBit", "velocityValid", "powerTransform",
    "inversePowerTransform", "velocityExtendedOrigin", "reflectComponents", "finalCellVector", "supportCellVector",
    "supportSelectorCellVector", "supportContainingVector", "finalSignedVector", "finalSelectorVector",
    "tetraWeights", "contained", "invalidPointVector", "invalidPointVectorAt", "uniformCubeContains",
    "finalTetraPointVector", "finalPointVector", "containingPublishedRow",
    "surroundingOwnerDelaunayVectorMeasured", "locateFinalPointVectorMeasured",
    "airSampleGrid",
  ].map(name => wgslFunctionDeclaration(octreeFaceBandWGSL, name)).join("\n");
  const source = /* wgsl */ `
struct P{dims:vec3u,maximumLeaf:u32,rowCapacity:u32,faceCapacity:u32,rowHashCapacity:u32,faceHashCapacity:u32,siteHashCapacity:u32,powerRowCapacity:u32,generation:u32,maximumRounds:u32,ownersPerBrick:u32,powerGeneration:u32,axisStride:u32,ownedFacesPerRow:u32,closedBoundaryMask:u32,pad0:u32,pad1:u32,pad2:u32}
struct Row{cell:u32,globalRow:u32,flags:u32,size:u32,representativePhi:f32,minimumPhi:f32,maximumPhi:f32,padf:f32}
struct Owner{origin:vec3u,size:u32,valid:u32}
struct Metric{topology:u32,transformFlags:u32,volume:f32,reserved:u32}
struct TetraHeader{first:u32,count:u32,flags:u32}struct TetraVertex{v:vec4f}
struct SurroundingOwnerVectorMeasurement{value:vec4f,rowsTested:u32}
struct LocatedFinalPointVector{value:vec4f,directSuccess:u32,fullRowFallback:u32,candidateRowsTested:u32,surroundingFallback:u32,surroundingRowsTested:u32}
struct C{flags:u32,firstError:u32,rowCount:u32,faceCount:u32,incidenceCount:u32,generation:u32,valid:u32,maximumDepth:u32,seedCount:u32,acceptedCount:u32,unresolvedCount:u32,initialRows:u32,sampleFailures:u32,coarsePhiFallbacks:u32,coarsePhiFailures:u32,bandPhiExtensions:u32,marchHeapHighWater:u32,marchPops:u32,marchTrials:u32,marchChunks:u32,marchChunkBound:u32,marchCapExhausted:u32,marchUnresolvedWithPredecessor:u32,marchDisconnected:u32,directAnchorSuccess:u32,fullRowFallbackInvocations:u32,fullRowCandidateRowsTested:u32,surroundingOwnerFallbackInvocations:u32,surroundingOwnerRowsTested:u32,airSamplesSelected:u32,airSamplesEvaluated:u32,pad31:u32}
struct PointControl{flags:u32,firstError:u32,rowCount:u32,generation:u32,solved:u32,valid:u32,wallContributions:u32,pad:u32}
struct TransitionControl{flags:u32,firstError:u32,rowCount:u32,transitionRows:u32,adjacencyCount:u32,ready:u32,transferReady:u32,detailFlags:u32,coreEnd:u32,support1End:u32,support2End:u32,support3NodeEnd:u32,endpointEnd:u32,boundaryGhostRequests:u32,hierarchyReady:u32,phiFailureCounts:u32,failureBand:u32,failureStage:u32,failureRowCell:u32,failureRowSize:u32,failureDescriptor:u32,failureTopology:u32,failureTransformFlags:u32,failureSelector:u32,failureRawX:u32,failureRawY:u32,failureRawZ:u32,failureRequestedSize:u32,failureResolvedCell:u32,failureBoundaryFlips:u32,failureOwnerCell:u32,failureOwnerSizeValid:u32,support4NodeEnd:u32,support5NodeEnd:u32,support6NodeEnd:u32,support7NodeEnd:u32,pad35:u32,pad36:u32,pad37:u32,pad38:u32}
struct SampleP{dims:vec3u,maximumLeaf:u32,siteHashCapacity:u32,count:u32,rowHashCapacity:u32,rowCapacity:u32,cellSize:f32,fineGeneration:u32,p1:u32,p2:u32}
@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(5)var<storage,read>control:C;
@group(0)@binding(6)var<storage,read>rows:array<Row>;
@group(0)@binding(7)var<storage,read>rowHash:array<u32>;
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
fn rowOf(cellKey:u32)->u32{let key=cellKey+1u;let start=hash(key)&(p.rowHashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(p.rowHashCapacity-1u);let observed=rowHash[slot*2u];if(observed==0u){return INVALID;}if(observed==key){let encoded=rowHash[slot*2u+1u];return select(INVALID,encoded-1u,encoded!=0u&&encoded!=INVALID);}}return INVALID;}
fn sampleBandRow(cellKey:u32)->u32{let start=hash(cellKey+1u)&(sp.rowHashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(sp.rowHashCapacity-1u);let observed=rowHash[slot*2u];if(observed==0u){return INVALID;}if(observed==cellKey+1u){let encoded=rowHash[slot*2u+1u];return select(INVALID,encoded-1u,encoded!=0u&&encoded!=INVALID);}}return INVALID;}
fn retainedBandAnchor(pointGrid:vec3f)->u32{if(any(pointGrid<vec3f(0))||any(pointGrid>=vec3f(sp.dims))){return INVALID;}let q=vec3u(floor(pointGrid));var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);let band=sampleBandRow(cell(origin));if(band!=INVALID&&band<sp.rowCapacity&&band<transitionControl.support7NodeEnd&&band<arrayLength(&rows)){let row=rows[band];if(row.cell==cell(origin)&&row.size==size&&(row.flags&ROW_SUPPORT3_ENDPOINT)==0u){return band;}}if(size>=sp.maximumLeaf){break;}size<<=1u;}return INVALID;}
@compute @workgroup_size(64)fn classifyAirBandVelocity(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=sp.count||i>=arrayLength(&positions)||i>=arrayLength(&sampleResults)||i>=arrayLength(&sampleStatus)||positions[i].w<=0.){return;}let sample=airSampleGrid(positions[i].xyz);if(sample.w==0.){sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_DOMAIN;return;}let grid=sample.xyz;let q=vec3u(floor(grid));let owner=ownerAt(q);if(owner.valid==0u){sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_OWNER;return;}let band=retainedBandAnchor(grid);if(band==INVALID||band>=sp.rowCapacity||band>=transitionControl.support7NodeEnd||band>=arrayLength(&rows)){sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_ROW;return;}if(control.generation!=sp.fineGeneration||pointControl.generation!=sp.fineGeneration){sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_GENERATION;return;}if((rows[band].flags&ROW_PHI)==0u){sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_ROW;return;}let stageBStatus=sampleStatus[i];let stageBReason=stageBStatus&255u;let needsDualCompletion=(stageBStatus&VALID)==0u&&(stageBReason==4u||stageBReason==8u);if(rows[band].minimumPhi<0.&&!needsDualCompletion){return;}sampleResults[i].w=bitcast<f32>(band);sampleStatus[i]=SAMPLE_EVALUATE;}
@compute @workgroup_size(64)fn evaluateAirBandVelocity(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=sp.count||i>=arrayLength(&positions)||i>=arrayLength(&sampleResults)||i>=arrayLength(&sampleStatus)||sampleStatus[i]!=SAMPLE_EVALUATE){return;}let band=bitcast<u32>(sampleResults[i].w);let sample=airSampleGrid(positions[i].xyz);if(sample.w==0.){sampleResults[i]=invalidPointVector(SAMPLE_FAIL_DOMAIN);sampleStatus[i]=SAMPLE_FAILED|SAMPLE_FAIL_DOMAIN;return;}let velocity=locateFinalPointVectorMeasured(band,sample.xyz).value;if(!velocityValid(velocity)){let reason=u32(round(max(1.,-velocity.w)));sampleResults[i]=velocity;sampleStatus[i]=SAMPLE_FAILED|((band&0xffffu)<<8u)|((16u+min(reason,11u))&255u);return;}sampleResults[i]=velocity;sampleStatus[i]=SAMPLE_EVALUATED;}
@compute @workgroup_size(64)fn finalizeAirBandVelocity(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=sp.count||i>=arrayLength(&sampleStatus)){return;}let status=sampleStatus[i];if(status!=SAMPLE_EVALUATE&&status!=SAMPLE_EVALUATED&&(status&SAMPLE_FAILED)==0u){return;}if(status==SAMPLE_EVALUATED&&control.valid==VALID&&control.generation==sp.fineGeneration&&pointControl.valid==VALID&&pointControl.flags==0u&&pointControl.generation==sp.fineGeneration){sampleStatus[i]=VALID|EXTRAPOLATED;return;}sampleStatus[i]=FACE_BAND_UNAVAILABLE|(status&0x00ffffffu);}
`;
  if (/\\batomic(?:Add|And|CompareExchangeWeak|Exchange|Load|Max|Min|Or|Store|Sub|Xor)?\\b/.test(source)) {
    throw new Error("Recurring face-band air sampler must not contain atomic operations");
  }
  return source;
}

export const octreeFaceBandWGSL = /* wgsl */ `
override faceBfsLayer:u32=1u;
override faceBfsCausal:u32=1u;
struct P{dims:vec3u,maximumLeaf:u32,rowCapacity:u32,faceCapacity:u32,rowHashCapacity:u32,faceHashCapacity:u32,siteHashCapacity:u32,powerRowCapacity:u32,generation:u32,maximumRounds:u32,ownersPerBrick:u32,powerGeneration:u32,axisStride:u32,ownedFacesPerRow:u32,closedBoundaryMask:u32,pad0:u32,pad1:u32,pad2:u32}
struct FineP{brickDims:vec3u,brickResolution:u32,sampleDims:vec3u,samplesPerBrick:u32,domainOrigin:vec3f,fineWidth:f32,hashCapacity:u32,maxProbes:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32}
struct Site{cellPlusOne:atomic<u32>,size:u32,row:u32,pad:u32}
struct Row{cell:u32,globalRow:u32,flags:u32,size:u32,representativePhi:f32,minimumPhi:f32,maximumPhi:f32,padf:f32}
struct CoarseEntry{cellPlusOne:u32,size:u32,phi:f32,minimumPhi:f32,maximumPhi:f32,flags:u32,row:u32,physicalVolume:f32}struct CoarseDirectory{state:u32,generation:u32,hashCapacity:u32,maximumLeafSize:u32,dimensions:vec3u,physicalCellSize:f32,entries:array<CoarseEntry>}
struct Owner{origin:vec3u,size:u32,valid:u32}
struct CatalogEntry{firstFace:u32,faceCount:u32}struct PowerCatalogFace{neighborOffsetSize:vec4f,areaCentroid:vec4f,normalInverseDistance:vec4f}struct BandLS{a0:vec4f,a1:vec4f,a2:vec4f}struct PointControl{flags:atomic<u32>,firstError:atomic<u32>,rowCount:u32,generation:u32,solved:atomic<u32>,valid:atomic<u32>,wallContributions:atomic<u32>,pad:u32}
struct Metric{topology:u32,transformFlags:u32,volume:f32,reserved:u32}struct TetraHeader{first:u32,count:u32,flags:u32}struct TetraVertex{v:vec4f}struct GuardCandidate{band:u32,selector:u32,cell:u32,size:u32}struct TransitionAdjacency{band:u32,a:u32,b:u32,c:u32}struct TransitionControl{flags:atomic<u32>,firstError:atomic<u32>,rowCount:u32,transitionRows:atomic<u32>,adjacencyCount:atomic<u32>,ready:atomic<u32>,transferReady:u32,detailFlags:atomic<u32>,coreEnd:u32,support1End:u32,support2End:u32,support3NodeEnd:u32,endpointEnd:u32,boundaryGhostRequests:atomic<u32>,hierarchyReady:atomic<u32>,phiFailureCounts:atomic<u32>,failureBand:atomic<u32>,failureStage:u32,failureRowCell:u32,failureRowSize:u32,failureDescriptor:u32,failureTopology:u32,failureTransformFlags:u32,failureSelector:u32,failureRawX:u32,failureRawY:u32,failureRawZ:u32,failureRequestedSize:u32,failureResolvedCell:u32,failureBoundaryFlips:u32,failureOwnerCell:u32,failureOwnerSizeValid:u32,support4NodeEnd:u32,support5NodeEnd:u32,support6NodeEnd:u32,support7NodeEnd:u32,pad35:u32,pad36:u32,pad37:u32,pad38:u32}
struct Face{negativeRow:u32,positiveRow:u32,axisSpan:u32,globalFace:u32,velocity:vec4f,centroid:vec4f,phi:f32,area:f32,flags:u32,pad:u32}
struct PhiDiagnostic{origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32,cause:u32,detail:u32}
struct State{velocity:vec4f,parent:u32,depth:u32,status:atomic<u32>,pad:u32}struct C{flags:atomic<u32>,firstError:atomic<u32>,rowCount:atomic<u32>,faceCount:atomic<u32>,incidenceCount:atomic<u32>,generation:u32,valid:atomic<u32>,maximumDepth:atomic<u32>,seedCount:atomic<u32>,acceptedCount:atomic<u32>,unresolvedCount:atomic<u32>,initialRows:atomic<u32>,sampleFailures:atomic<u32>,coarsePhiFallbacks:atomic<u32>,coarsePhiFailures:atomic<u32>,bandPhiExtensions:atomic<u32>,marchHeapHighWater:atomic<u32>,marchPops:atomic<u32>,marchTrials:atomic<u32>,marchChunks:atomic<u32>,marchChunkBound:u32,marchCapExhausted:atomic<u32>,marchUnresolvedWithPredecessor:atomic<u32>,marchDisconnected:atomic<u32>,directAnchorSuccess:atomic<u32>,fullRowFallbackInvocations:atomic<u32>,fullRowCandidateRowsTested:atomic<u32>,surroundingOwnerFallbackInvocations:atomic<u32>,surroundingOwnerRowsTested:atomic<u32>,airSamplesSelected:atomic<u32>,airSamplesEvaluated:atomic<u32>,connectivityFallbacks:atomic<u32>}
struct PowerFace{negativeRow:u32,positiveRow:u32,geometryCode:u32,flags:u32,normalVelocity:f32,area:f32,inverseDistance:f32,openFraction:f32}struct PowerPublication{flags:atomic<u32>,firstError:atomic<u32>,faceCount:u32,targetCount:atomic<u32>,interpolatedCount:atomic<u32>,committedCount:atomic<u32>,fineGeneration:u32,powerGeneration:u32,valid:atomic<u32>,p0:u32,p1:u32,p2:u32,p3:u32,p4:u32,p5:u32,p6:u32}struct PowerRowWork{faceCount:u32,incidenceCount:u32,faceOffset:u32,incidenceOffset:u32}struct PowerIncidence{face:u32,sign:i32}
struct TransientPowerFace{negativeRow:u32,positiveRow:u32,flags:u32,pad:u32,normal:vec4f,centroid:vec4f,normalVelocity:f32,area:f32,inverseDistance:f32,padf:f32}struct TransientPowerControl{flags:atomic<u32>,firstError:atomic<u32>,rowCount:u32,faceSlots:u32,emitted:atomic<u32>,sampled:atomic<u32>,validated:atomic<u32>,generation:u32,valid:atomic<u32>,p0:u32,p1:u32,p2:u32,p3:u32,p4:u32,p5:u32,p6:u32}
struct SampleP{dims:vec3u,maximumLeaf:u32,siteHashCapacity:u32,count:u32,rowHashCapacity:u32,rowCapacity:u32,cellSize:f32,fineGeneration:u32,p1:u32,p2:u32}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<uniform>fp:FineP;@group(0)@binding(2)var<storage,read>metadata:array<u32>;@group(0)@binding(3)var<storage,read>worklist:array<u32>;@group(0)@binding(4)var<storage,read_write>sites:array<Site>;@group(0)@binding(5)var<storage,read_write>control:C;@group(0)@binding(6)var<storage,read_write>rows:array<Row>;@group(0)@binding(7)var<storage,read_write>rowHash:array<atomic<u32>>;@group(0)@binding(8)var<storage,read>sampleFlags:array<u32>;@group(0)@binding(9)var<storage,read>powerVelocityControl:array<u32>;@group(0)@binding(10)var<storage,read>powerRowVelocities:array<vec4f>;@group(0)@binding(11)var<storage,read>catalogEntries:array<CatalogEntry>;@group(0)@binding(12)var<storage,read_write>faces:array<Face>;@group(0)@binding(13)var<storage,read_write>faceHash:array<atomic<u32>>;@group(0)@binding(14)var<storage,read_write>incidence:array<atomic<u32>>;@group(0)@binding(15)var<storage,read_write>states:array<State>;@group(0)@binding(18)var<storage,read_write>indirect:array<u32>;@group(0)@binding(19)var<storage,read_write>rowVelocities:array<vec4f>;@group(0)@binding(20)var<uniform>sp:SampleP;@group(0)@binding(21)var<storage,read>positions:array<vec4f>;@group(0)@binding(22)var<storage,read_write>sampleResults:array<vec4f>;@group(0)@binding(23)var<storage,read_write>sampleStatus:array<atomic<u32>>;@group(0)@binding(24)var<storage,read>finePhi:array<f32>;@group(0)@binding(25)var<storage,read>coarsePhi:CoarseDirectory;@group(0)@binding(26)var<storage,read>owners:array<u32>;@group(0)@binding(27)var<storage,read_write>metrics:array<Metric>;@group(0)@binding(28)var<storage,read>tetraHeaders:array<TetraHeader>;@group(0)@binding(29)var<storage,read>tetrahedra:array<u32>;@group(0)@binding(30)var<storage,read>tetraVertices:array<TetraVertex>;@group(0)@binding(31)var<storage,read_write>transitionAdjacency:array<TransitionAdjacency>;@group(0)@binding(32)var<storage,read_write>transitionControl:TransitionControl;@group(0)@binding(33)var<storage,read>sameOrFinerDirect:array<u32>;@group(0)@binding(34)var<storage,read>sameOrCoarserDirect:array<u32>;@group(0)@binding(35)var<storage,read_write>globalRowHash:array<atomic<u32>>;@group(0)@binding(36)var<storage,read>powerFaceControl:array<u32>;@group(0)@binding(37)var<storage,read_write>powerFaces:array<PowerFace>;@group(0)@binding(38)var<storage,read>powerFaceNormals:array<vec4f>;@group(0)@binding(39)var<storage,read_write>powerFaceCentroids:array<vec4f>;@group(0)@binding(40)var<storage,read_write>powerVelocityScratch:array<vec4u>;@group(0)@binding(41)var<storage,read_write>powerPublication:PowerPublication;@group(0)@binding(42)var<storage,read>fineTopologyControl:array<u32>;@group(0)@binding(43)var<storage,read_write>guardCandidates:array<GuardCandidate>;@group(0)@binding(44)var<storage,read_write>provisionalVelocities:array<vec4f>;@group(0)@binding(45)var<storage,read>catalogFaces:array<PowerCatalogFace>;@group(0)@binding(46)var<storage,read_write>pointAccumulator:array<BandLS>;@group(0)@binding(47)var<storage,read_write>pointStatus:array<atomic<u32>>;@group(0)@binding(48)var<storage,read_write>pointControl:PointControl;@group(0)@binding(49)var<storage,read>powerIncidenceRows:array<PowerRowWork>;@group(0)@binding(50)var<storage,read>powerIncidences:array<PowerIncidence>;@group(0)@binding(51)var<storage,read>fineHash:array<u32>;@group(0)@binding(52)var<storage,read_write>transientPowerFaces:array<TransientPowerFace>;@group(0)@binding(53)var<storage,read_write>transientPowerIncidences:array<PowerIncidence>;@group(0)@binding(54)var<storage,read_write>transientPowerRows:array<PowerRowWork>;@group(0)@binding(55)var<storage,read_write>transientPowerControl:TransientPowerControl;@group(0)@binding(56)var<storage,read_write>cptParentInput:array<u32>;@group(0)@binding(57)var<storage,read_write>cptParentOutput:array<u32>;
@group(0)@binding(58)var<storage,read_write>oldAdvectionControl:array<atomic<u32>>;@group(0)@binding(59)var<storage,read_write>oldAdvectionSeed:array<atomic<u32>>;
const INVALID:u32=0xffffffffu;const VALID:u32=0x80000000u;const STATUS_VALID:u32=0x3f800000u;const EXTRAPOLATED:u32=0x10000000u;const FACE_BAND_UNAVAILABLE:u32=0x08000000u;const LIVE:u32=1u;const SEED:u32=2u;const PHI_VALID:u32=4u;const PHI_DIAGNOSTIC:u32=8u;const ROW_PHI:u32=1u;const ROW_COARSE:u32=2u;const ROW_CORE:u32=4u;const ROW_SUPPORT1:u32=8u;const ROW_SUPPORT2:u32=16u;const ROW_SUPPORT3_NODE:u32=32u;const ROW_SUPPORT3_ENDPOINT:u32=64u;const ROW_COARSE_AIR:u32=128u;const ROW_COARSE_LIQUID:u32=256u;const ROW_COARSE_MIXED:u32=512u;const ROW_SUPPORT4_NODE:u32=1024u;const ROW_SUPPORT5_NODE:u32=2048u;const ROW_SUPPORT6_NODE:u32=4096u;const ROW_GUARD:u32=ROW_SUPPORT1|ROW_SUPPORT2|ROW_SUPPORT3_NODE|ROW_SUPPORT4_NODE|ROW_SUPPORT5_NODE|ROW_SUPPORT6_NODE|ROW_SUPPORT3_ENDPOINT;const UNKNOWN:u32=0u;const TRIAL:u32=1u;const ACCEPTED:u32=2u;const REJECTED:u32=3u;
const CAPACITY:u32=1u;const HASH:u32=2u;const SOURCE:u32=4u;const BAD_ROW:u32=8u;const BAD_FACE:u32=16u;const BAD_PHI:u32=32u;const UNRESOLVED:u32=64u;const INCOMPLETE:u32=128u;const OUTSIDE_FINE_BAND:u32=256u;
const TRANSITION_SOURCE:u32=1u;const TRANSITION_CAPACITY:u32=2u;const TRANSITION_ADJACENCY:u32=4u;const TRANSITION_DESCRIPTOR:u32=8u;const TRANSITION_ACUTE_GRADING:u32=16u;const MAX_TETRA:u32=${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra}u;const MAX_GUARDS:u32=${Math.max(OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows, OCTREE_FACE_BAND_UNIFORM_SUPPORT_REQUESTS)}u;const UNIFORM_GUARDS:u32=${OCTREE_FACE_BAND_UNIFORM_SUPPORT_REQUESTS}u;const MAX_ENDPOINTS:u32=${OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW}u;const UNIFORM_REQUEST:u32=0x80000000u;const COARSER_DESCRIPTOR:u32=0x80000000u;
const PHI_CAUSE_MISSING_ROW:u32=0u;const PHI_CAUSE_EXACT_COARSE_MISS:u32=1u;const PHI_CAUSE_INVALID_METRIC:u32=2u;const PHI_CAUSE_INVALID_SELECTOR:u32=3u;const PHI_PATH_CUBE:u32=1u;const PHI_PATH_DELAUNAY:u32=2u;const PHI_PATH_ANCHOR:u32=3u;const PHI_FAILURE_TAG:u32=0x80000000u;
const DETAIL_GEOMETRY:u32=1u;const DETAIL_BELOW_DOMAIN:u32=2u;const DETAIL_ABOVE_DOMAIN:u32=4u;const DETAIL_ALIGNMENT:u32=8u;const DETAIL_OWNER:u32=16u;const DETAIL_MISSING_ROW:u32=32u;const DETAIL_ROW_RANGE:u32=64u;const DETAIL_SIZE:u32=128u;
const OWNER_FAILURE_SUPPORT1:u32=1u;const OWNER_FAILURE_SUPPORT2:u32=2u;const OWNER_FAILURE_SUPPORT3:u32=3u;const OWNER_FAILURE_TRANSITION:u32=4u;const OWNER_FAILURE_ENDPOINT:u32=5u;const OWNER_FAILURE_ACUTE_GRADING:u32=6u;const OWNER_FAILURE_DISCONNECTED_FACE:u32=7u;const OWNER_FAILURE_SUPPORT4:u32=8u;const OWNER_FAILURE_DESCRIPTOR:u32=9u;const OWNER_FAILURE_BAND_PHI:u32=10u;
const POWER_SOURCE:u32=1u;const POWER_CAPACITY:u32=2u;const POWER_MISSING_ROW:u32=4u;const POWER_FACE:u32=8u;const POWER_NORMAL:u32=16u;const POWER_NONFINITE:u32=32u;const POWER_INCOMPLETE:u32=64u;
const POWER_FACE_VALID:u32=0x80000000u;const POWER_FACE_BOUNDARY:u32=1u;const POWER_FACE_OPEN_BOUNDARY:u32=2u;
const POINT_SOURCE:u32=1u;const POINT_CAPACITY:u32=2u;const POINT_FACE:u32=4u;const POINT_SAMPLE:u32=8u;const POINT_NORMAL:u32=16u;const POINT_NONFINITE:u32=32u;const POINT_SINGULAR:u32=64u;const POINT_CONDITION:u32=128u;const POINT_MAX_FACES:u32=${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u;
const SAMPLE_EVALUATE:u32=0x04000000u;const SAMPLE_EVALUATED:u32=0x02000000u;const SAMPLE_FAILED:u32=0x01000000u;
// Preserve the structural reason in the low byte when a Stage-B air query
// cannot be closed.  The authority predicate remains SAMPLE_FAILED and the
// final public status remains FACE_BAND_UNAVAILABLE; these tags are QA-only.
const SAMPLE_FAIL_DOMAIN:u32=1u;const SAMPLE_FAIL_OWNER:u32=2u;const SAMPLE_FAIL_ROW:u32=3u;
const SAMPLE_FAIL_GENERATION:u32=4u;const SAMPLE_FAIL_VECTOR:u32=5u;
const DESCRIPTOR_DIRECTIONS:array<vec3i,18>=array<vec3i,18>(vec3i(-1,0,0),vec3i(0,-1,0),vec3i(0,0,-1),vec3i(0,0,1),vec3i(0,1,0),vec3i(1,0,0),vec3i(-1,-1,0),vec3i(-1,0,-1),vec3i(-1,0,1),vec3i(-1,1,0),vec3i(0,-1,-1),vec3i(0,-1,1),vec3i(0,1,-1),vec3i(0,1,1),vec3i(1,-1,0),vec3i(1,0,-1),vec3i(1,0,1),vec3i(1,1,0));
const BODY_DIAGONAL_DIRECTIONS:array<vec3i,8>=array<vec3i,8>(vec3i(-1,-1,-1),vec3i(-1,-1,1),vec3i(-1,1,-1),vec3i(-1,1,1),vec3i(1,-1,-1),vec3i(1,-1,1),vec3i(1,1,-1),vec3i(1,1,1));
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn fail(code:u32,index:u32){atomicOr(&control.flags,code);atomicMin(&control.firstError,index);}fn hash(k:u32)->u32{var v=k*0x9e3779b1u;v=(v^(v>>16u))*0x7feb352du;return v^(v>>15u);}fn siteHash(c:u32,s:u32)->u32{var v=c^(s*0x9e3779b9u);v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);}fn unpackBrick(key:u32)->vec3u{let xy=fp.brickDims.x*fp.brickDims.y;let z=key/xy;let r=key-z*xy;let y=r/fp.brickDims.x;return vec3u(r-y*fp.brickDims.x,y,z);}fn cell(q:vec3u)->u32{return q.x+p.dims.x*(q.y+p.dims.y*q.z);}
fn invalidOwner()->Owner{return Owner(vec3u(0u),0u,0u);}fn decodeOwner(word:u32,q:vec3u)->Owner{if(word==0x80000000u){return Owner(q,1u,1u);}let exponent=word&7u;if(exponent==0u||exponent>5u){return invalidOwner();}let size=1u<<exponent;let origin=(q>>vec3u(exponent))<<vec3u(exponent);if(any(origin+vec3u(size)>p.dims)){return invalidOwner();}return Owner(origin,size,1u);}fn decodePagedOwner(word:u32,q:vec3u)->Owner{if(word==0x80000000u){return Owner(q,1u,1u);}if(word<1u||word>5u){return invalidOwner();}let size=1u<<word;let origin=(q>>vec3u(word))<<vec3u(word);if(any(origin+vec3u(size)>p.dims)){return invalidOwner();}return Owner(origin,size,1u);}fn residentCanonicalOwner(q:vec3u)->Owner{var size=min(p.maximumLeaf,8u);var origin=(q/vec3u(size))*vec3u(size);loop{if(all(origin+vec3u(size)<=p.dims)||size==1u){break;}size>>=1u;origin=(q/vec3u(size))*vec3u(size);}return Owner(origin,size,1u);}
fn ownerAt(q:vec3u)->Owner{let denseIndex=cell(q);if(arrayLength(&owners)<=15u||owners[15]!=0x4f574e52u){if(denseIndex>=arrayLength(&owners)){return invalidOwner();}return decodeOwner(owners[denseIndex],q);}if(owners[7]==0u||owners[7]!=p.powerGeneration){return invalidOwner();}let freeListOffset=owners[5];let payloadOffset=owners[6];let capacity=owners[3];if(capacity==0u||freeListOffset<=16u||((freeListOffset-16u)&1u)!=0u||payloadOffset>=arrayLength(&owners)){return invalidOwner();}let hashCapacity=(freeListOffset-16u)/2u;if(hashCapacity==0u||16u+2u*hashCapacity>arrayLength(&owners)){return invalidOwner();}let bd=(p.dims+vec3u(7u))/8u;let b=q/8u;if(any(b>=bd)){return invalidOwner();}let logical=b.x+b.y*bd.x+b.z*bd.x*bd.y;let key=logical+1u;var slot=(logical*0x9e3779b1u)%hashCapacity;var encoded=0u;var found=false;for(var probe=0u;probe<hashCapacity;probe+=1u){let observed=owners[16u+slot];if(observed==key){encoded=owners[16u+hashCapacity+slot];found=true;break;}if(observed==0u){break;}slot=select(slot+1u,0u,slot+1u==hashCapacity);}if(!found||encoded==0u||encoded==INVALID||encoded>capacity){return invalidOwner();}let local=q%vec3u(8u);let physical=encoded-1u;if(physical>(arrayLength(&owners)-payloadOffset-1u)/512u){return invalidOwner();}let at=payloadOffset+physical*512u+local.x+local.y*8u+local.z*64u;let word=owners[at];if(word==0u){return residentCanonicalOwner(q);}if(word==INVALID){return invalidOwner();}return decodePagedOwner(word,q);}
fn coarseHash(c:u32,s:u32)->u32{var v=c^(s*0x9e3779b9u);v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);}fn validCoarseGeneration()->bool{if(arrayLength(&fineTopologyControl)<8u){return false;}let mask=0x3fffffffu;let coarseGeneration=coarsePhi.generation&mask;let fineGeneration=p.generation&mask;let clean=fineTopologyControl[0]==0u&&fineTopologyControl[4]==1u&&fineTopologyControl[5]==0u&&fineTopologyControl[7]==0u;return clean&&coarseGeneration==fineGeneration;}fn validCoarse()->bool{let capacity=min(coarsePhi.hashCapacity,arrayLength(&coarsePhi.entries));let expectedWidth=fp.fineWidth*f32(fp.fineFactor);return coarsePhi.state==VALID&&validCoarseGeneration()&&coarsePhi.hashCapacity==capacity&&capacity>0u&&(capacity&(capacity-1u))==0u&&coarsePhi.maximumLeafSize==p.maximumLeaf&&all(coarsePhi.dimensions==p.dims)&&coarsePhi.physicalCellSize>0.0&&abs(coarsePhi.physicalCellSize-expectedWidth)<=1e-5*max(coarsePhi.physicalCellSize,expectedWidth);}
fn coarseSlot(cellKey:u32,size:u32)->u32{let capacity=min(coarsePhi.hashCapacity,arrayLength(&coarsePhi.entries));let start=coarseHash(cellKey,size)&(capacity-1u);for(var probe=0u;probe<min(32u,capacity);probe+=1u){let slot=(start+probe)&(capacity-1u);let entry=coarsePhi.entries[slot];if(entry.cellPlusOne==0u){return INVALID;}if(entry.cellPlusOne==cellKey+1u&&entry.size==size){return slot;}}return INVALID;}
fn coarseEntryRecord(slot:u32)->vec4f{if(slot==INVALID||slot>=arrayLength(&coarsePhi.entries)){return vec4f(0.);}let entry=coarsePhi.entries[slot];if((entry.flags&9u)!=9u||!finite(entry.phi)||!finite(entry.minimumPhi)||!finite(entry.maximumPhi)||entry.minimumPhi>entry.phi||entry.phi>entry.maximumPhi){return vec4f(0.);}return vec4f(entry.phi,entry.minimumPhi,entry.maximumPhi,1.);}
// Aanjaneya et al. 2017 Section 5 keeps a coarse octree level set specifically
// to carry signed-distance authority beyond the narrow fine SPGrid.  A pressure
// rebuild may change the leaf containing a point before that same-time coarse
// field is republished on the new leaves.  Sample the previous spatial
// publication at the new leaf centre; the current owner graph is redistanced
// below by the paper's local cube/Delaunay fast march.
fn coarseCellSeedRecord(origin:vec3u,size:u32)->vec4f{if(!validCoarse()||size==0u||any(origin+vec3u(size)>p.dims)){return vec4f(0.);}let exact=coarseEntryRecord(coarseSlot(cell(origin),size));if(exact.w!=0.){return exact;}let q=min(origin+vec3u(size/2u),p.dims-vec3u(1u));var scale=1u;loop{let priorOrigin=(q/vec3u(scale))*vec3u(scale);let prior=coarseEntryRecord(coarseSlot(cell(priorOrigin),scale));if(prior.w!=0.){return prior;}if(scale>=coarsePhi.maximumLeafSize){break;}scale*=2u;}return vec4f(0.);}
fn coarseSignFlag(owner:Owner)->u32{if(!validCoarse()||owner.valid==0u){return 0u;}let entry=coarseCellSeedRecord(owner.origin,owner.size);if(entry.w==0.){return ROW_COARSE_MIXED;}if(entry.z<0.){return ROW_COARSE_LIQUID;}if(entry.y>0.){return ROW_COARSE_AIR;}return ROW_COARSE_MIXED;}
fn findSite(c:u32,s:u32)->u32{let cap=min(p.siteHashCapacity,arrayLength(&sites));if(cap==0u){return INVALID;}let start=siteHash(c,s)&(cap-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(cap-1u);let observed=atomicLoad(&sites[slot].cellPlusOne);if(observed==0u){return INVALID;}if(observed==c+1u&&sites[slot].size==s){return sites[slot].row;}}return INVALID;}fn containing(q:vec3u)->u32{var size=1u;loop{let o=(q/vec3u(size))*vec3u(size);let found=findSite(cell(o),size);if(found!=INVALID){return found;}if(size>=p.maximumLeaf){break;}size*=2u;}return INVALID;}
fn publishedRow(slot:u32)->u32{let encoded=atomicLoad(&rowHash[slot*2u+1u]);return select(INVALID,encoded-1u,encoded!=0u&&encoded!=INVALID);}
fn insertRow(cellKey:u32,globalRow:u32,rowFlags:u32,ownerSize:u32)->u32{if(cellKey>=p.dims.x*p.dims.y*p.dims.z||ownerSize==0u){fail(BAD_ROW,cellKey);return INVALID;}let key=cellKey+1u;let start=hash(key)&(p.rowHashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(p.rowHashCapacity-1u);var occupied=0u;for(var retry=0u;retry<32u;retry+=1u){let result=atomicCompareExchangeWeak(&rowHash[slot*2u],0u,key);if(result.exchanged){let row=atomicAdd(&control.rowCount,1u);if(row>=p.rowCapacity||row>=arrayLength(&rows)){fail(CAPACITY,cellKey);atomicStore(&rowHash[slot*2u+1u],INVALID);return INVALID;}rows[row]=Row(cellKey,globalRow,rowFlags,ownerSize,0.,0.,0.,0.);atomicStore(&rowHash[slot*2u+1u],row+1u);return row;}if(result.old_value==0u){continue;}if(result.old_value==key){return INVALID;}occupied=result.old_value;break;}if(occupied==0u){fail(HASH,cellKey);return INVALID;}}fail(HASH,cellKey);return INVALID;}
fn rowOf(cellKey:u32)->u32{let key=cellKey+1u;let start=hash(key)&(p.rowHashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(p.rowHashCapacity-1u);let observed=atomicLoad(&rowHash[slot*2u]);if(observed==0u){return INVALID;}if(observed==key){return publishedRow(slot);}}return INVALID;}
fn supportedOwnerSize(size:u32)->bool{return size==1u||size==2u||size==4u||size==8u||size==16u||size==32u;}fn ownerContains(owner:Owner,q:vec3u)->bool{return owner.valid!=0u&&supportedOwnerSize(owner.size)&&owner.size<=p.maximumLeaf&&all(owner.origin%vec3u(owner.size)==vec3u(0u))&&all(owner.origin+vec3u(owner.size)<=p.dims)&&all(q>=owner.origin)&&all(q<owner.origin+vec3u(owner.size));}
// All graded same/coarser masks have catalog entries. Keep the old branch as
// a reserved diagnostic ABI, but never classify a topology as intrinsically
// obtuse: co-spherical vertex-only sites supply the strict-acute local link.
fn strictlyObtuseCoarseMask(mask:u32)->bool{return false;}
fn recordDescriptorFailure(band:u32,row:Row,descriptor:u32,site:u32){var claimed=false;loop{let result=atomicCompareExchangeWeak(&transitionControl.failureBand,INVALID,band);if(result.exchanged){claimed=true;break;}if(result.old_value!=INVALID){break;}}if(!claimed){return;}let origin=coord(row.cell);transitionControl.failureStage=OWNER_FAILURE_DESCRIPTOR;transitionControl.failureRowCell=row.cell;transitionControl.failureRowSize=row.size;transitionControl.failureDescriptor=descriptor;transitionControl.failureTopology=INVALID;transitionControl.failureTransformFlags=0u;transitionControl.failureSelector=site;transitionControl.failureRawX=origin.x;transitionControl.failureRawY=origin.y;transitionControl.failureRawZ=origin.z;transitionControl.failureRequestedSize=row.size;transitionControl.failureResolvedCell=INVALID;transitionControl.failureBoundaryFlips=0u;transitionControl.failureOwnerCell=INVALID;transitionControl.failureOwnerSizeValid=0u;}
fn recordAcuteGradingFailure(band:u32,row:Row,descriptor:u32,mask:u32){}
fn resolveBandDescriptor(descriptor:u32,band:u32)->Metric{var packed=INVALID;if((descriptor&COARSER_DESCRIPTOR)!=0u){let index=descriptor&0x1ffu;if((descriptor&0x40fffe00u)==0u&&index<arrayLength(&sameOrCoarserDirect)){packed=sameOrCoarserDirect[index];}}else{let index=descriptor&0x3ffffu;if((descriptor&0x40fc0000u)==0u&&index<arrayLength(&sameOrFinerDirect)){packed=sameOrFinerDirect[index];}}if(packed==INVALID){recordDescriptorFailure(band,rows[band],descriptor,6u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,descriptor);}let topology=packed&0xffffu;let transform=packed>>16u;let boundary=(descriptor>>16u)&0x3f00u;return Metric(topology,transform|boundary|VALID,0.,descriptor);}
fn describeBandRow(band:u32)->Metric{let row=rows[band];if(!supportedOwnerSize(row.size)||row.size>p.maximumLeaf||row.cell>=p.dims.x*p.dims.y*p.dims.z){recordDescriptorFailure(band,row,0u,1u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,TRANSITION_DESCRIPTOR);}let origin=vec3u(row.cell%p.dims.x,(row.cell/p.dims.x)%p.dims.y,row.cell/(p.dims.x*p.dims.y));if(any(origin%vec3u(row.size)!=vec3u(0u))||any(origin+vec3u(row.size)>p.dims)){recordDescriptorFailure(band,row,0u,2u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,TRANSITION_DESCRIPTOR);}let anchor=ownerAt(origin);if(!ownerContains(anchor,origin)||anchor.size!=row.size||any(anchor.origin!=origin)){recordDescriptorFailure(band,row,anchor.size|(anchor.origin.x<<8u)|(anchor.origin.y<<16u)|(anchor.origin.z<<24u),3u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,TRANSITION_DESCRIPTOR);}var sizes:array<u32,18>;var boundaryMask=0u;var finer=false;var coarser=false;for(var bit=0u;bit<18u;bit+=1u){let direction=DESCRIPTOR_DIRECTIONS[bit];var probe=vec3i(0);for(var axis=0u;axis<3u;axis+=1u){probe[axis]=select(select(i32(origin[axis]+row.size/2u),i32(origin[axis]+row.size),direction[axis]>0),i32(origin[axis])-1,direction[axis]<0);}if(any(probe<vec3i(0))||any(probe>=vec3i(p.dims))){if(bit<6u){boundaryMask|=1u<<bit;}sizes[bit]=row.size;continue;}let neighbor=ownerAt(vec3u(probe));if(!ownerContains(neighbor,vec3u(probe))||all(min(neighbor.origin+vec3u(neighbor.size),origin+vec3u(row.size))>max(neighbor.origin,origin))||neighbor.size*2u<row.size||neighbor.size>row.size*2u){recordDescriptorFailure(band,row,(bit<<26u)|(neighbor.size<<20u)|cell(vec3u(max(probe,vec3i(0)))),4u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,TRANSITION_DESCRIPTOR);}sizes[bit]=neighbor.size;finer=finer||neighbor.size<row.size;coarser=coarser||neighbor.size>row.size;}if(finer&&coarser){var packedSizes=0u;for(var bit=0u;bit<16u;bit+=1u){packedSizes|=(sizes[bit]&3u)<<(bit*2u);}recordDescriptorFailure(band,row,packedSizes,5u);transitionFail(TRANSITION_DESCRIPTOR,band);return Metric(INVALID,0u,0.,TRANSITION_DESCRIPTOR);}var descriptor=boundaryMask<<24u;if(!coarser){for(var bit=0u;bit<18u;bit+=1u){if(sizes[bit]==row.size){descriptor|=1u<<bit;}}}else{let child=(origin/vec3u(row.size))&vec3u(1u);descriptor|=COARSER_DESCRIPTOR|child.x|(child.y<<1u)|(child.z<<2u);let outward=vec3i(select(-1,1,child.x==1u),select(-1,1,child.y==1u),select(-1,1,child.z==1u));let wanted=array<vec3i,6>(vec3i(outward.x,0,0),vec3i(0,outward.y,0),vec3i(0,0,outward.z),vec3i(outward.x,outward.y,0),vec3i(outward.x,0,outward.z),vec3i(0,outward.y,outward.z));var coarseMask=0u;for(var coarseBit=0u;coarseBit<6u;coarseBit+=1u){for(var bit=0u;bit<18u;bit+=1u){if(all(DESCRIPTOR_DIRECTIONS[bit]==wanted[coarseBit])&&sizes[bit]==row.size*2u){descriptor|=1u<<(coarseBit+3u);coarseMask|=1u<<coarseBit;}}}if(strictlyObtuseCoarseMask(coarseMask)){recordAcuteGradingFailure(band,row,descriptor,coarseMask);transitionFail(TRANSITION_ACUTE_GRADING,band);return Metric(INVALID,0u,0.,descriptor);}}return resolveBandDescriptor(descriptor,band);}
fn resolveTopologyRange(local:u32,begin:u32,end:u32){let band=begin+local;if(band>=end||band>=arrayLength(&rows)||band>=arrayLength(&metrics)){return;}metrics[band]=describeBandRow(band);}
@compute @workgroup_size(64)fn resolveCoreBandTopology(@builtin(global_invocation_id)g:vec3u){resolveTopologyRange(g.x,0u,transitionControl.coreEnd);}
@compute @workgroup_size(64)fn resolveSupport1BandTopology(@builtin(global_invocation_id)g:vec3u){resolveTopologyRange(g.x,transitionControl.coreEnd,transitionControl.support1End);}
@compute @workgroup_size(64)fn resolveSupport2BandTopology(@builtin(global_invocation_id)g:vec3u){resolveTopologyRange(g.x,transitionControl.support1End,transitionControl.support2End);}
@compute @workgroup_size(64)fn resolveSupport3BandTopology(@builtin(global_invocation_id)g:vec3u){resolveTopologyRange(g.x,transitionControl.support2End,transitionControl.support3NodeEnd);}
@compute @workgroup_size(64)fn resolveSupport4BandTopology(@builtin(global_invocation_id)g:vec3u){resolveTopologyRange(g.x,transitionControl.support3NodeEnd,transitionControl.support4NodeEnd);}
@compute @workgroup_size(64)fn resolveSupport5BandTopology(@builtin(global_invocation_id)g:vec3u){resolveTopologyRange(g.x,transitionControl.support4NodeEnd,transitionControl.support5NodeEnd);}
@compute @workgroup_size(64)fn resolveSupport6BandTopology(@builtin(global_invocation_id)g:vec3u){resolveTopologyRange(g.x,transitionControl.support5NodeEnd,transitionControl.support7NodeEnd);}
@compute @workgroup_size(64)fn resolveSupport3EndpointTopology(@builtin(global_invocation_id)g:vec3u){resolveTopologyRange(g.x,transitionControl.support7NodeEnd,transitionControl.endpointEnd);}
fn faceOf(globalFace:u32)->u32{let key=globalFace+1u;let start=hash(key)&(p.faceHashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(p.faceHashCapacity-1u);let observed=atomicLoad(&faceHash[slot*2u]);if(observed==0u){return INVALID;}if(observed==key){let encoded=atomicLoad(&faceHash[slot*2u+1u]);return select(INVALID,encoded-1u,encoded!=0u);}}return INVALID;}
fn validPowerVelocity()->bool{return arrayLength(&powerVelocityControl)>=8u&&powerVelocityControl[0]==VALID&&powerVelocityControl[2]<=p.powerRowCapacity&&powerVelocityControl[7]==p.powerGeneration;}
// Aanjaneya et al. 2017 Section 5 computes the velocity interpolant used at
// every point reached by fine-level-set advection.  The interface prefix is
// the allocation seed, not the velocity domain: all cells in the published
// fine narrow band require regular-face extrapolation and a full centre
// vector.  Rows added after this core exist only to close local cube/Delaunay
// stencils.
fn fineBandBrickCount()->u32{if(arrayLength(&fineTopologyControl)<9u||arrayLength(&worklist)<5u){return INVALID;}let residentCount=worklist[0];let clean=fineTopologyControl[0]==0u&&fineTopologyControl[4]==1u&&fineTopologyControl[5]==0u&&fineTopologyControl[7]==0u;let seeds=fineTopologyControl[8];if(!clean||residentCount>fp.pageCapacity||worklist[1]!=fp.generation||worklist[3]!=1u||worklist[4]!=1u||fineTopologyControl[2]!=residentCount||seeds>residentCount){return INVALID;}return residentCount;}
@compute @workgroup_size(1)fn prepareFaceBand(){control.generation=p.generation;atomicStore(&control.firstError,INVALID);atomicStore(&control.valid,0u);let bandBricks=fineBandBrickCount();if(bandBricks==INVALID||!validPowerVelocity()){indirect[0]=0u;indirect[1]=1u;indirect[2]=1u;fail(SOURCE,0u);return;}indirect[0]=(bandBricks*p.ownersPerBrick+63u)/64u;indirect[1]=1u;indirect[2]=1u;}
@compute @workgroup_size(64)fn mapFineBricksToBandRows(@builtin(global_invocation_id)g:vec3u){let item=g.x;let bandBricks=fineBandBrickCount();if(bandBricks==INVALID){fail(SOURCE,item);return;}if(item>=bandBricks*p.ownersPerBrick||atomicLoad(&control.flags)!=0u){return;}let work=item/p.ownersPerBrick;let id=worklist[5u+work];if(id>=fp.pageCapacity||metadata[id*10u+2u]!=fp.generation){fail(SOURCE,item);return;}let key=metadata[id*10u+1u];let origin=(unpackBrick(key)*fp.brickResolution)/fp.fineFactor;let local=item-work*p.ownersPerBrick;let side=u32(ceil(f32(fp.brickResolution)/f32(fp.fineFactor)));let q=origin+vec3u(local%side,(local/side)%side,local/(side*side));if(!all(q<p.dims)){return;}let owner=ownerAt(q);if(owner.valid==0u){fail(SOURCE,cell(q));return;}let ownerCell=cell(owner.origin);let signFlag=coarseSignFlag(owner);if(signFlag==0u){fail(SOURCE,ownerCell);return;}let globalRow=containing(owner.origin);_ = insertRow(ownerCell,globalRow,ROW_COARSE|ROW_CORE|signFlag,owner.size);}
fn indexGlobalRow(globalRow:u32,band:u32){let key=globalRow+1u;let start=hash(key)&(p.rowHashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(p.rowHashCapacity-1u);let result=atomicCompareExchangeWeak(&globalRowHash[slot*2u],0u,key);if(result.exchanged){atomicStore(&globalRowHash[slot*2u+1u],band+1u);return;}if(result.old_value==0u){continue;}if(result.old_value==key){let encoded=atomicLoad(&globalRowHash[slot*2u+1u]);if(encoded!=0u&&encoded!=band+1u){fail(BAD_ROW,globalRow);}return;}}fail(HASH,globalRow);}
@compute @workgroup_size(64)fn indexBandGlobalRows(@builtin(global_invocation_id)g:vec3u){let band=g.x;if(band>=atomicLoad(&control.rowCount)||band>=arrayLength(&rows)||atomicLoad(&control.flags)!=0u){return;}let row=rows[band];let targetRoles=ROW_CORE|ROW_SUPPORT1|ROW_SUPPORT2;if((row.flags&targetRoles)==0u){return;}let globalRow=row.globalRow;if(globalRow!=INVALID){if(globalRow>=p.powerRowCapacity){fail(BAD_ROW,globalRow);return;}indexGlobalRow(globalRow,band);}}
fn bandForGlobalRow(globalRow:u32)->u32{if(globalRow==INVALID){return INVALID;}let key=globalRow+1u;let start=hash(key)&(p.rowHashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(p.rowHashCapacity-1u);let observed=atomicLoad(&globalRowHash[slot*2u]);if(observed==0u){return INVALID;}if(observed==key){let encoded=atomicLoad(&globalRowHash[slot*2u+1u]);return select(INVALID,encoded-1u,encoded!=0u);}}return INVALID;}
fn insertFace(globalFace:u32,slotValue:u32){let key=globalFace+1u;let start=hash(key)&(p.faceHashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(p.faceHashCapacity-1u);let result=atomicCompareExchangeWeak(&faceHash[slot*2u],0u,key);if(result.exchanged){atomicStore(&faceHash[slot*2u+1u],slotValue+1u);return;}if(result.old_value==0u){continue;}if(result.old_value==key){fail(BAD_FACE,globalFace);return;}}fail(HASH,globalFace);}
fn coord(cellKey:u32)->vec3u{return vec3u(cellKey%p.dims.x,(cellKey/p.dims.x)%p.dims.y,cellKey/(p.dims.x*p.dims.y));}
fn transitionFail(code:u32,index:u32){atomicOr(&transitionControl.flags,code);atomicMin(&transitionControl.firstError,index);}fn adjacencyFail(index:u32,detail:u32){atomicOr(&transitionControl.detailFlags,detail);transitionFail(TRANSITION_ADJACENCY,index);}fn recordOwnerFailure(stage:u32,band:u32,row:Row,metric:Metric,selector:u32,rawOrigin:vec3i,requestedSize:u32,resolved:vec4i,owner:Owner){var claimed=false;loop{let result=atomicCompareExchangeWeak(&transitionControl.failureBand,INVALID,band);if(result.exchanged){claimed=true;break;}if(result.old_value!=INVALID){break;}}if(!claimed){return;}transitionControl.failureStage=stage;transitionControl.failureRowCell=row.cell;transitionControl.failureRowSize=row.size;transitionControl.failureDescriptor=metric.reserved;transitionControl.failureTopology=metric.topology;transitionControl.failureTransformFlags=metric.transformFlags;transitionControl.failureSelector=selector;transitionControl.failureRawX=bitcast<u32>(rawOrigin.x);transitionControl.failureRawY=bitcast<u32>(rawOrigin.y);transitionControl.failureRawZ=bitcast<u32>(rawOrigin.z);transitionControl.failureRequestedSize=requestedSize;transitionControl.failureResolvedCell=select(INVALID,cell(vec3u(resolved.xyz)),resolved.w>=0);transitionControl.failureBoundaryFlips=bitcast<u32>(resolved.w);transitionControl.failureOwnerCell=cell(owner.origin);transitionControl.failureOwnerSizeValid=(owner.size&0xffffu)|((owner.valid&0xffffu)<<16u);}fn inversePowerTransform(x:vec3f,code:u32)->vec3f{let bits=code&7u;let q=x*vec3f(select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));let k=(code/8u)%6u;if(k==0u){return q;}if(k==1u){return q.xzy;}if(k==2u){return q.yxz;}if(k==3u){return q.zxy;}if(k==4u){return q.yzx;}return q.zyx;}
fn reflectedOrigin(origin:vec3i,size:u32)->vec4i{if(size==0u){return vec4i(0,0,0,-1);}var resolved=origin;var flips=0;let high=origin+vec3i(i32(size));for(var axis=0u;axis<3u;axis+=1u){if(origin[axis]<0){if(origin[axis]<-i32(size)||(p.closedBoundaryMask&negativeBoundaryBit(axis))==0u){return vec4i(0,0,0,-1);}resolved[axis]=-origin[axis]-i32(size);flips|=1<<axis;}else if(high[axis]>i32(p.dims[axis])){if(high[axis]>i32(p.dims[axis]+size)||(p.closedBoundaryMask&positiveBoundaryBit(axis))==0u){return vec4i(0,0,0,-1);}resolved[axis]=2*i32(p.dims[axis])-origin[axis]-i32(size);flips|=1<<axis;}}if(any(resolved<vec3i(0))||any(resolved+vec3i(i32(size))>vec3i(p.dims))){return vec4i(0,0,0,-1);}return vec4i(resolved,flips);}
// Product boundary policy layered around the paper's in-domain interpolant:
// closed planes reflect the wall-normal component, while open planes use a
// one-layer zero-gradient vector extension. Geometry/weights remain at the
// original virtual point and no exterior row is ever published.
fn velocityExtendedOrigin(origin:vec3i,size:u32)->vec4i{if(size==0u){return vec4i(0,0,0,-1);}var resolved=origin;var flips=0;let high=origin+vec3i(i32(size));for(var axis=0u;axis<3u;axis+=1u){if(origin[axis]<0){if(origin[axis]<-i32(size)){return vec4i(0,0,0,-1);}resolved[axis]=-origin[axis]-i32(size);if((p.closedBoundaryMask&negativeBoundaryBit(axis))!=0u){flips|=1<<axis;}}else if(high[axis]>i32(p.dims[axis])){if(high[axis]>i32(p.dims[axis]+size)){return vec4i(0,0,0,-1);}resolved[axis]=2*i32(p.dims[axis])-origin[axis]-i32(size);if((p.closedBoundaryMask&positiveBoundaryBit(axis))!=0u){flips|=1<<axis;}}}if(any(resolved<vec3i(0))||any(resolved+vec3i(i32(size))>vec3i(p.dims))){return vec4i(0,0,0,-1);}return vec4i(resolved,flips);}
fn hasOpenBoundaryCrossing(origin:vec3i,size:u32)->bool{let high=origin+vec3i(i32(size));for(var axis=0u;axis<3u;axis+=1u){if(origin[axis]<0&&(p.closedBoundaryMask&negativeBoundaryBit(axis))==0u){return true;}if(high[axis]>i32(p.dims[axis])&&(p.closedBoundaryMask&positiveBoundaryBit(axis))==0u){return true;}}return false;}
fn recordRowHashFailure(band:u32,selector:u32,cellKey:u32){if(atomicLoad(&transitionControl.failureBand)!=band||transitionControl.failureSelector!=selector){return;}let key=cellKey+1u;let start=hash(key)&(p.rowHashCapacity-1u);transitionControl.pad35=start;transitionControl.pad36=INVALID;transitionControl.pad37=INVALID;transitionControl.pad38=INVALID;for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(p.rowHashCapacity-1u);let observed=atomicLoad(&rowHash[slot*2u]);if(observed==0u&&transitionControl.pad36==INVALID){transitionControl.pad36=slot;}if(observed==key){transitionControl.pad37=slot;transitionControl.pad38=atomicLoad(&rowHash[slot*2u+1u]);return;}}}
fn transitionNeighbor(band:u32,selector:u32,metric:Metric)->u32{if(band>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){transitionFail(TRANSITION_SOURCE,band);return INVALID;}let row=rows[band];let vertex=tetraVertices[selector].v;if(row.size==0u||!finite(vertex.x)||!finite(vertex.y)||!finite(vertex.z)||!finite(vertex.w)||vertex.w<=0.){transitionFail(TRANSITION_SOURCE,band);return INVALID;}let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(vertex.xyz,metric.transformFlags&63u);let neighborSizeFloat=f32(row.size)*vertex.w;let neighborSize=u32(round(neighborSizeFloat));let rawOrigin=vec3i(round(point-.5*f32(neighborSize)));if(neighborSize==0u||abs(neighborSizeFloat-f32(neighborSize))>1e-4){adjacencyFail(band,DETAIL_GEOMETRY);return INVALID;}let rawHigh=rawOrigin+vec3i(i32(neighborSize));let outside=any(rawOrigin<vec3i(0))||any(rawHigh>vec3i(p.dims));if(outside){let extended=velocityExtendedOrigin(rawOrigin,neighborSize);if(extended.w<0){adjacencyFail(band,select(DETAIL_ABOVE_DOMAIN,DETAIL_BELOW_DOMAIN,any(rawOrigin<vec3i(0))));return INVALID;}atomicAdd(&transitionControl.boundaryGhostRequests,1u);return INVALID;}let reflected=reflectedOrigin(rawOrigin,neighborSize);if(reflected.w<0){adjacencyFail(band,select(DETAIL_ABOVE_DOMAIN,DETAIL_BELOW_DOMAIN,any(rawOrigin<vec3i(0))));return INVALID;}let neighborOrigin=vec3u(reflected.xyz);if(any(neighborOrigin%vec3u(neighborSize)!=vec3u(0u))){adjacencyFail(band,DETAIL_ALIGNMENT);return INVALID;}let expectedOwner=ownerAt(neighborOrigin);if(!ownerContains(expectedOwner,neighborOrigin)||expectedOwner.size!=neighborSize||any(expectedOwner.origin!=neighborOrigin)){recordOwnerFailure(OWNER_FAILURE_TRANSITION,band,row,metric,selector,rawOrigin,neighborSize,reflected,expectedOwner);adjacencyFail(band,DETAIL_OWNER);return INVALID;}let neighborCell=cell(neighborOrigin);let neighbor=rowOf(neighborCell);if(neighbor==INVALID){recordOwnerFailure(OWNER_FAILURE_TRANSITION,band,row,metric,selector,rawOrigin,neighborSize,reflected,expectedOwner);recordRowHashFailure(band,selector,neighborCell);adjacencyFail(band,DETAIL_MISSING_ROW);return INVALID;}if(neighbor>=atomicLoad(&control.rowCount)||neighbor>=arrayLength(&rows)){recordOwnerFailure(OWNER_FAILURE_TRANSITION,band,row,metric,selector,rawOrigin,neighborSize,reflected,expectedOwner);adjacencyFail(band,DETAIL_ROW_RANGE);return INVALID;}if(rows[neighbor].size!=neighborSize){recordOwnerFailure(OWNER_FAILURE_TRANSITION,band,row,metric,selector,rawOrigin,neighborSize,reflected,expectedOwner);adjacencyFail(band,DETAIL_SIZE);return INVALID;}return neighbor;}
@compute @workgroup_size(1)fn prepareTransitionAdjacency(){atomicStore(&transitionControl.flags,0u);atomicStore(&transitionControl.firstError,INVALID);transitionControl.rowCount=min(atomicLoad(&control.rowCount),p.rowCapacity);transitionControl.coreEnd=transitionControl.rowCount;transitionControl.support1End=transitionControl.rowCount;transitionControl.support2End=transitionControl.rowCount;transitionControl.support3NodeEnd=transitionControl.rowCount;transitionControl.support4NodeEnd=transitionControl.rowCount;transitionControl.support5NodeEnd=transitionControl.rowCount;transitionControl.support6NodeEnd=transitionControl.rowCount;transitionControl.support7NodeEnd=transitionControl.rowCount;transitionControl.endpointEnd=transitionControl.rowCount;atomicStore(&control.initialRows,transitionControl.rowCount);atomicStore(&transitionControl.transitionRows,0u);atomicStore(&transitionControl.adjacencyCount,0u);atomicStore(&transitionControl.ready,0u);transitionControl.transferReady=0u;atomicStore(&transitionControl.detailFlags,0u);atomicStore(&transitionControl.boundaryGhostRequests,0u);atomicStore(&transitionControl.hierarchyReady,0u);atomicStore(&transitionControl.phiFailureCounts,0u);atomicStore(&transitionControl.failureBand,INVALID);}
fn writeSupportDispatch(word:u32,count:u32){indirect[word]=(count+63u)/64u;indirect[word+1u]=1u;indirect[word+2u]=1u;}
fn writeSupportTierDispatch(rowWord:u32,candidateWord:u32,count:u32,fanout:u32){writeSupportDispatch(rowWord,count);if(fanout!=0u&&count>arrayLength(&guardCandidates)/fanout){transitionFail(TRANSITION_CAPACITY,count);writeSupportDispatch(candidateWord,0u);return;}writeSupportDispatch(candidateWord,count*fanout);}
@compute @workgroup_size(64)fn clearBandPhiEdges(@builtin(global_invocation_id)g:vec3u){if(g.x==0u){atomicStore(&control.bandPhiExtensions,0u);}if(g.x<arrayLength(&pointStatus)){atomicStore(&pointStatus[g.x],INVALID);}}
@compute @workgroup_size(1)fn prepareSupport0Dispatch(){writeSupportTierDispatch(6u,9u,transitionControl.coreEnd,MAX_GUARDS);}
@compute @workgroup_size(64)fn clearSupportCandidates(@builtin(global_invocation_id)g:vec3u){if(g.x<arrayLength(&guardCandidates)){guardCandidates[g.x]=GuardCandidate(INVALID,INVALID,INVALID,0u);}}
fn enumerateSupportRequests(local:u32,begin:u32,end:u32){let band=begin+local;if(band>=end||band>=arrayLength(&rows)||band>=arrayLength(&metrics)){return;}let base=local*MAX_GUARDS;if(base>arrayLength(&guardCandidates)||MAX_GUARDS>arrayLength(&guardCandidates)-base){transitionFail(TRANSITION_CAPACITY,band);return;}for(var request=0u;request<MAX_GUARDS;request+=1u){guardCandidates[base+request]=GuardCandidate(INVALID,INVALID,INVALID,0u);}let metric=metrics[band];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){transitionFail(TRANSITION_DESCRIPTOR,band);return;}let header=tetraHeaders[metric.topology];if((header.flags&1u)!=0u){for(var request=0u;request<UNIFORM_GUARDS;request+=1u){guardCandidates[base+request]=GuardCandidate(band,UNIFORM_REQUEST|request,INVALID,0u);}return;}if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first||header.count>MAX_TETRA){transitionFail(TRANSITION_SOURCE,band);return;}var count=0u;for(var tetra=0u;tetra<header.count;tetra+=1u){let packed=tetrahedra[header.first+tetra];let selectors=array<u32,3>(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);for(var corner=0u;corner<3u;corner+=1u){let selector=selectors[corner];var duplicate=false;for(var prior=0u;prior<count;prior+=1u){duplicate=duplicate||guardCandidates[base+prior].selector==selector;}if(duplicate){continue;}if(count>=MAX_GUARDS){transitionFail(TRANSITION_CAPACITY,band);return;}guardCandidates[base+count]=GuardCandidate(band,selector,INVALID,0u);count+=1u;}}}
@compute @workgroup_size(64)fn enumerateSupport1Requests(@builtin(global_invocation_id)g:vec3u){enumerateSupportRequests(g.x,0u,transitionControl.coreEnd);}
@compute @workgroup_size(64)fn enumerateSupport2Requests(@builtin(global_invocation_id)g:vec3u){enumerateSupportRequests(g.x,transitionControl.coreEnd,transitionControl.support1End);}
@compute @workgroup_size(64)fn enumerateSupport3Requests(@builtin(global_invocation_id)g:vec3u){enumerateSupportRequests(g.x,transitionControl.support1End,transitionControl.support2End);}
@compute @workgroup_size(64)fn enumerateSupport4Requests(@builtin(global_invocation_id)g:vec3u){enumerateSupportRequests(g.x,transitionControl.support2End,transitionControl.support3NodeEnd);}
@compute @workgroup_size(64)fn enumerateSupportEndpointRequests(@builtin(global_invocation_id)g:vec3u){enumerateSupportRequests(g.x,transitionControl.support3NodeEnd,transitionControl.support4NodeEnd);}
@compute @workgroup_size(64)fn enumerateSupport6Requests(@builtin(global_invocation_id)g:vec3u){enumerateSupportRequests(g.x,transitionControl.support4NodeEnd,transitionControl.support5NodeEnd);}
fn negativeBoundaryBit(axis:u32)->u32{return select(select(4u,2u,axis==1u),1u,axis==0u);}fn positiveBoundaryBit(axis:u32)->u32{return select(select(8u,16u,axis==1u),32u,axis==0u);}fn closedBoundaryException(low:vec3i,high:vec3i)->bool{var crossed=false;for(var axis=0u;axis<3u;axis+=1u){if(low[axis]<0){crossed=true;if((p.closedBoundaryMask&negativeBoundaryBit(axis))==0u){return false;}}if(high[axis]>i32(p.dims[axis])){crossed=true;if((p.closedBoundaryMask&positiveBoundaryBit(axis))==0u){return false;}}}return crossed;}
@compute @workgroup_size(64)fn resolveSupportOwners(@builtin(global_invocation_id)g:vec3u){let at=g.x;if(at>=arrayLength(&guardCandidates)){return;}var candidate=guardCandidates[at];let existingEnd=min(atomicLoad(&control.rowCount),p.rowCapacity);if(candidate.band==INVALID||candidate.band>=existingEnd||candidate.band>=arrayLength(&rows)){return;}let row=rows[candidate.band];let metric=metrics[candidate.band];var originI=vec3i(0);var size=row.size;var uniformBodyDiagonal=false;if((candidate.selector&UNIFORM_REQUEST)!=0u){let request=candidate.selector&~UNIFORM_REQUEST;if(request>=MAX_GUARDS||row.size==0u){transitionFail(TRANSITION_SOURCE,candidate.band);return;}uniformBodyDiagonal=request>=18u;var direction=vec3i(0);if(uniformBodyDiagonal){direction=BODY_DIAGONAL_DIRECTIONS[request-18u];}else{direction=DESCRIPTOR_DIRECTIONS[request];}let rowOrigin=vec3i(coord(row.cell));originI=rowOrigin+direction*i32(row.size);let descriptor=metric.reserved;if(!uniformBodyDiagonal&&(descriptor&COARSER_DESCRIPTOR)!=0u){let child=vec3u(descriptor&1u,(descriptor>>1u)&1u,(descriptor>>2u)&1u);let outward=vec3i(select(-1,1,child.x==1u),select(-1,1,child.y==1u),select(-1,1,child.z==1u));let coarseDirections=array<vec3i,6>(vec3i(outward.x,0,0),vec3i(0,outward.y,0),vec3i(0,0,outward.z),vec3i(outward.x,outward.y,0),vec3i(outward.x,0,outward.z),vec3i(0,outward.y,outward.z));let sameOrigin=originI;for(var coarse=0u;coarse<6u;coarse+=1u){if((descriptor&(1u<<(coarse+3u)))==0u){continue;}let coarseDirection=coarseDirections[coarse];let coarseSize=row.size*2u;var coarseProbe=rowOrigin;for(var axis=0u;axis<3u;axis+=1u){coarseProbe[axis]=select(select(rowOrigin[axis]+i32(row.size/2u),rowOrigin[axis]+i32(row.size),coarseDirection[axis]>0),rowOrigin[axis]-1,coarseDirection[axis]<0);}let coarseOrigin=(coarseProbe/i32(coarseSize))*i32(coarseSize);let overlaps=all(min(coarseOrigin+vec3i(i32(coarseSize)),sameOrigin+vec3i(i32(row.size)))>max(coarseOrigin,sameOrigin));if(overlaps){size=coarseSize;originI=coarseOrigin;break;}}}}else{if(candidate.selector>=arrayLength(&tetraVertices)){transitionFail(TRANSITION_SOURCE,candidate.band);return;}let vertex=tetraVertices[candidate.selector].v;if(row.size==0u||!finite(vertex.x)||!finite(vertex.y)||!finite(vertex.z)||!finite(vertex.w)||vertex.w<=0.){transitionFail(TRANSITION_SOURCE,candidate.band);return;}let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(vertex.xyz,metric.transformFlags&63u);let sizeFloat=f32(row.size)*vertex.w;size=u32(round(sizeFloat));let originFloat=round(point-.5*f32(size));if(size==0u||abs(sizeFloat-f32(size))>1e-4){adjacencyFail(candidate.band,DETAIL_GEOMETRY);return;}originI=vec3i(originFloat);}let high=originI+vec3i(i32(size));if(any(originI<vec3i(0))||any(high>vec3i(p.dims))){let extended=velocityExtendedOrigin(originI,size);if(extended.w<0){if(uniformBodyDiagonal){return;}adjacencyFail(candidate.band,select(DETAIL_ABOVE_DOMAIN,DETAIL_BELOW_DOMAIN,any(originI<vec3i(0))));return;}atomicAdd(&transitionControl.boundaryGhostRequests,1u);let boundaryOwner=ownerAt(vec3u(extended.xyz));if(!ownerContains(boundaryOwner,vec3u(extended.xyz))){if(uniformBodyDiagonal){return;}recordOwnerFailure(OWNER_FAILURE_SUPPORT1,candidate.band,row,metric,candidate.selector,originI,size,extended,boundaryOwner);adjacencyFail(candidate.band,DETAIL_OWNER);return;}originI=vec3i(boundaryOwner.origin);size=boundaryOwner.size;}let origin=vec3u(originI);if(size==0u||any(origin%vec3u(size)!=vec3u(0u))){if(uniformBodyDiagonal){return;}adjacencyFail(candidate.band,DETAIL_ALIGNMENT);return;}let exact=ownerAt(origin);if(!ownerContains(exact,origin)||exact.size!=size||any(exact.origin!=origin)){if(uniformBodyDiagonal){if(ownerContains(exact,origin)&&exact.size>size){candidate.cell=cell(exact.origin);candidate.size=exact.size;guardCandidates[at]=candidate;}return;}let stage=select(select(OWNER_FAILURE_SUPPORT3,OWNER_FAILURE_SUPPORT2,candidate.band<transitionControl.support1End),OWNER_FAILURE_SUPPORT1,candidate.band<transitionControl.coreEnd);recordOwnerFailure(stage,candidate.band,row,metric,candidate.selector,originI,size,vec4i(originI,0),exact);adjacencyFail(candidate.band,DETAIL_OWNER);return;}candidate.cell=cell(origin);candidate.size=size;guardCandidates[at]=candidate;}
fn insertSupportCandidate(at:u32,begin:u32,rowRole:u32){if(at>=arrayLength(&guardCandidates)||atomicLoad(&control.flags)!=0u){return;}let candidate=guardCandidates[at];if(candidate.band==INVALID||candidate.cell==INVALID||candidate.size==0u){return;}let origin=coord(candidate.cell);let owner=Owner(origin,candidate.size,1u);let signFlag=coarseSignFlag(owner);if(signFlag==0u){transitionFail(TRANSITION_SOURCE,candidate.band);fail(SOURCE,candidate.cell);return;}let globalRow=containing(origin);_ = insertRow(candidate.cell,globalRow,ROW_COARSE|rowRole|signFlag,candidate.size);let support=rowOf(candidate.cell);if(support!=INVALID&&support>=begin&&support<arrayLength(&metrics)){metrics[support]=Metric(INVALID,0u,0.0,0u);}}
fn recordSupportEdgeFailure(candidate:GuardCandidate,support:u32){let result=atomicCompareExchangeWeak(&transitionControl.failureBand,INVALID,candidate.band);if(!result.exchanged){return;}transitionControl.failureStage=7u;transitionControl.failureRowCell=candidate.cell;transitionControl.failureRowSize=candidate.size;transitionControl.failureDescriptor=support;transitionControl.failureTopology=INVALID;transitionControl.failureTransformFlags=0u;transitionControl.failureSelector=INVALID;transitionControl.failureRawX=coord(candidate.cell).x;transitionControl.failureRawY=coord(candidate.cell).y;transitionControl.failureRawZ=coord(candidate.cell).z;transitionControl.failureRequestedSize=candidate.size;transitionControl.failureResolvedCell=candidate.cell;transitionControl.failureBoundaryFlips=0u;transitionControl.failureOwnerCell=candidate.cell;transitionControl.failureOwnerSizeValid=candidate.size|(1u<<16u);let key=candidate.cell+1u;let start=hash(key)&(p.rowHashCapacity-1u);transitionControl.pad35=start;transitionControl.pad36=INVALID;transitionControl.pad37=INVALID;transitionControl.pad38=INVALID;for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(p.rowHashCapacity-1u);let observed=atomicLoad(&rowHash[slot*2u]);if(observed==0u&&transitionControl.pad36==INVALID){transitionControl.pad36=slot;}if(observed==key){transitionControl.pad37=slot;transitionControl.pad38=atomicLoad(&rowHash[slot*2u+1u]);return;}}}
fn recordBandPhiEdgeGroup(base:u32,stride:u32){for(var request=0u;request<stride;request+=1u){let at=base+request;if(at>=arrayLength(&guardCandidates)){break;}let candidate=guardCandidates[at];if(candidate.band==INVALID||candidate.cell==INVALID||candidate.size==0u){continue;}var duplicate=false;for(var prior=0u;prior<request;prior+=1u){let existing=guardCandidates[base+prior];duplicate=duplicate||(existing.band==candidate.band&&existing.cell==candidate.cell&&existing.size==candidate.size);}if(duplicate){continue;}let support=rowOf(candidate.cell);if(support==INVALID||support>=arrayLength(&pointStatus)){recordSupportEdgeFailure(candidate,support);fail(BAD_ROW,candidate.cell);return;}let edge=atomicAdd(&control.bandPhiExtensions,1u);if(edge>=arrayLength(&transientPowerIncidences)){fail(CAPACITY,candidate.cell);return;}let previous=atomicExchange(&pointStatus[support],edge);transientPowerIncidences[edge]=PowerIncidence(candidate.band,bitcast<i32>(previous));}}
@compute @workgroup_size(64)fn recordBandPhiSupportEdges(@builtin(global_invocation_id)g:vec3u){let base=g.x;if(base>=arrayLength(&guardCandidates)||base%MAX_GUARDS!=0u||atomicLoad(&control.flags)!=0u){return;}recordBandPhiEdgeGroup(base,MAX_GUARDS);}
@compute @workgroup_size(64)fn recordBandPhiEndpointEdges(@builtin(global_invocation_id)g:vec3u){let base=g.x;if(base>=arrayLength(&guardCandidates)||base%MAX_ENDPOINTS!=0u||atomicLoad(&control.flags)!=0u){return;}recordBandPhiEdgeGroup(base,MAX_ENDPOINTS);}
@compute @workgroup_size(1)fn resetBandPhiExtensionCount(){atomicStore(&control.bandPhiExtensions,0u);}
@compute @workgroup_size(64)fn insertSupport1Rows(@builtin(global_invocation_id)g:vec3u){insertSupportCandidate(g.x,transitionControl.coreEnd,ROW_SUPPORT1);}
@compute @workgroup_size(64)fn insertSupport2Rows(@builtin(global_invocation_id)g:vec3u){insertSupportCandidate(g.x,transitionControl.support1End,ROW_SUPPORT2);}
@compute @workgroup_size(64)fn insertSupport3NodeRows(@builtin(global_invocation_id)g:vec3u){insertSupportCandidate(g.x,transitionControl.support2End,ROW_SUPPORT3_NODE);}
@compute @workgroup_size(64)fn insertSupport4NodeRows(@builtin(global_invocation_id)g:vec3u){insertSupportCandidate(g.x,transitionControl.support3NodeEnd,ROW_SUPPORT4_NODE);}
@compute @workgroup_size(64)fn insertSupport5NodeRows(@builtin(global_invocation_id)g:vec3u){insertSupportCandidate(g.x,transitionControl.support4NodeEnd,ROW_SUPPORT5_NODE);}
@compute @workgroup_size(64)fn insertSupport6NodeRows(@builtin(global_invocation_id)g:vec3u){insertSupportCandidate(g.x,transitionControl.support5NodeEnd,ROW_SUPPORT6_NODE);}
// Audit the immutable S0..S6 prefix after all parallel insertions are visible.
// This verifies exact current owner identity without discovering or appending
// unrelated compact rows outside the fine narrow band.
@compute @workgroup_size(64)fn auditNarrowBandOwnerRows(@builtin(global_invocation_id)g:vec3u){let band=g.x;if(band>=transitionControl.support6NodeEnd||band>=arrayLength(&rows)){return;}let row=rows[band];let origin=coord(row.cell);let exact=ownerAt(origin);if(!ownerContains(exact,origin)||exact.size!=row.size||any(exact.origin!=origin)){transitionFail(TRANSITION_SOURCE,band);return;}if(rowOf(row.cell)!=band){transitionFail(TRANSITION_CAPACITY,band);}}
fn captureBoundary(previous:u32)->u32{let end=min(atomicLoad(&control.rowCount),p.rowCapacity);if(end<previous){transitionFail(TRANSITION_CAPACITY,end);return previous;}return end;}
@compute @workgroup_size(1)fn captureSupport1Boundary(){transitionControl.support1End=captureBoundary(transitionControl.coreEnd);writeSupportTierDispatch(12u,15u,transitionControl.support1End-transitionControl.coreEnd,MAX_GUARDS);}
@compute @workgroup_size(1)fn captureSupport2Boundary(){transitionControl.support2End=captureBoundary(transitionControl.support1End);writeSupportTierDispatch(18u,21u,transitionControl.support2End-transitionControl.support1End,MAX_GUARDS);}
@compute @workgroup_size(1)fn captureSupport3NodeBoundary(){transitionControl.support3NodeEnd=captureBoundary(transitionControl.support2End);writeSupportTierDispatch(24u,27u,transitionControl.support3NodeEnd-transitionControl.support2End,MAX_GUARDS);}
@compute @workgroup_size(1)fn captureSupport4NodeBoundary(){transitionControl.support4NodeEnd=captureBoundary(transitionControl.support3NodeEnd);writeSupportDispatch(30u,transitionControl.support4NodeEnd-transitionControl.support2End);let appended=transitionControl.support4NodeEnd-transitionControl.support3NodeEnd;if(appended>arrayLength(&guardCandidates)/MAX_GUARDS){transitionFail(TRANSITION_CAPACITY,appended);writeSupportDispatch(33u,0u);return;}writeSupportDispatch(33u,appended*MAX_GUARDS);}
@compute @workgroup_size(1)fn captureSupport5NodeBoundary(){transitionControl.support5NodeEnd=captureBoundary(transitionControl.support4NodeEnd);writeSupportTierDispatch(36u,39u,transitionControl.support5NodeEnd-transitionControl.support4NodeEnd,MAX_GUARDS);}
@compute @workgroup_size(1)fn captureSupport6NodeBoundary(){transitionControl.support6NodeEnd=captureBoundary(transitionControl.support5NodeEnd);writeSupportTierDispatch(42u,45u,transitionControl.support6NodeEnd-transitionControl.support5NodeEnd,MAX_GUARDS);}
@compute @workgroup_size(1)fn captureSupport7NodeBoundary(){transitionControl.support7NodeEnd=captureBoundary(transitionControl.support6NodeEnd);writeSupportTierDispatch(48u,51u,transitionControl.support7NodeEnd-transitionControl.support2End,MAX_ENDPOINTS);}
@compute @workgroup_size(64)fn enumerateSupport3EndpointRequests(@builtin(global_invocation_id)g:vec3u){let band=transitionControl.support2End+g.x;if(band>=transitionControl.support7NodeEnd||band>=arrayLength(&rows)){return;}let base=g.x*MAX_ENDPOINTS;if(base>arrayLength(&guardCandidates)||MAX_ENDPOINTS>arrayLength(&guardCandidates)-base){transitionFail(TRANSITION_CAPACITY,band);return;}for(var request=0u;request<MAX_ENDPOINTS;request+=1u){guardCandidates[base+request]=GuardCandidate(INVALID,INVALID,INVALID,0u);}let row=rows[band];let origin=coord(row.cell);let half=max(1u,row.size/2u);let sampleCount=select(1u,4u,row.size>1u);var out=0u;for(var axis=0u;axis<3u;axis+=1u){let ta=(axis+1u)%3u;let tb=(axis+2u)%3u;for(var side=0u;side<2u;side+=1u){for(var local=0u;local<sampleCount;local+=1u){var probe=vec3i(origin);probe[axis]+=select(-1,i32(row.size),side==1u);probe[ta]+=i32(select(0u,half,(local&1u)!=0u));probe[tb]+=i32(select(0u,half,(local&2u)!=0u));if(any(probe<vec3i(0))||any(probe>=vec3i(p.dims))){if(velocityExtendedOrigin(probe,1u).w>=0){atomicAdd(&transitionControl.boundaryGhostRequests,1u);out+=1u;continue;}adjacencyFail(band,select(DETAIL_ABOVE_DOMAIN,DETAIL_BELOW_DOMAIN,any(probe<vec3i(0))));out+=1u;continue;}let exact=ownerAt(vec3u(probe));if(!ownerContains(exact,vec3u(probe))){recordOwnerFailure(OWNER_FAILURE_ENDPOINT,band,row,Metric(INVALID,0u,0.,INVALID),INVALID,probe,0u,vec4i(probe,0),exact);adjacencyFail(band,DETAIL_OWNER);out+=1u;continue;}guardCandidates[base+out]=GuardCandidate(band,0u,cell(exact.origin),exact.size);out+=1u;}}}}
@compute @workgroup_size(64)fn insertSupport3EndpointRows(@builtin(global_invocation_id)g:vec3u){insertSupportCandidate(g.x,transitionControl.support7NodeEnd,ROW_SUPPORT3_ENDPOINT);}
@compute @workgroup_size(1)fn captureSupport3EndpointBoundary(){transitionControl.endpointEnd=captureBoundary(transitionControl.support7NodeEnd);writeSupportDispatch(54u,transitionControl.endpointEnd);writeSupportDispatch(57u,transitionControl.endpointEnd*p.ownedFacesPerRow);atomicStore(&transitionControl.hierarchyReady,select(0u,VALID,atomicLoad(&transitionControl.flags)==0u));}
@compute @workgroup_size(64)fn retireBandFaceSlots(@builtin(global_invocation_id)g:vec3u){let count=min(p.faceCapacity,transitionControl.endpointEnd*p.ownedFacesPerRow);if(g.x<count&&g.x<arrayLength(&faces)){faces[g.x].flags=0u;}}
@compute @workgroup_size(64)fn buildTransitionAdjacency(@builtin(global_invocation_id)g:vec3u){let band=g.x;if(band>=transitionControl.support2End||band>=arrayLength(&rows)){return;}if(band>=arrayLength(&metrics)){transitionFail(TRANSITION_CAPACITY,band);return;}var metric=metrics[band];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){transitionFail(TRANSITION_DESCRIPTOR,band);return;}let header=tetraHeaders[metric.topology];if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first||header.count>MAX_TETRA){transitionFail(TRANSITION_SOURCE,band);return;}let uniform=(header.flags&1u)!=0u;if(uniform){metric.reserved=0u;metrics[band]=metric;return;}atomicAdd(&transitionControl.transitionRows,1u);let descriptor=metric.reserved;metric.reserved=header.count;metrics[band]=metric;metric.reserved=descriptor;if(header.count==0u||band>arrayLength(&transitionAdjacency)/MAX_TETRA||band*MAX_TETRA>arrayLength(&transitionAdjacency)-header.count){transitionFail(TRANSITION_CAPACITY,band);return;}for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){transitionFail(TRANSITION_SOURCE,band);return;}let a=transitionNeighbor(band,selectors.x,metric);let b=transitionNeighbor(band,selectors.y,metric);let c=transitionNeighbor(band,selectors.z,metric);if(atomicLoad(&transitionControl.flags)!=0u){return;}transitionAdjacency[band*MAX_TETRA+local]=TransitionAdjacency(band,a,b,c);atomicAdd(&transitionControl.adjacencyCount,1u);}}
@compute @workgroup_size(64)fn buildDeepTransitionAdjacency(@builtin(global_invocation_id)g:vec3u){let band=transitionControl.support2End+g.x;if(band>=transitionControl.support4NodeEnd||band>=arrayLength(&rows)){return;}if(band>=arrayLength(&metrics)){transitionFail(TRANSITION_CAPACITY,band);return;}var metric=metrics[band];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){transitionFail(TRANSITION_DESCRIPTOR,band);return;}let header=tetraHeaders[metric.topology];if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first||header.count>MAX_TETRA){transitionFail(TRANSITION_SOURCE,band);return;}let uniform=(header.flags&1u)!=0u;if(uniform){metric.reserved=0u;metrics[band]=metric;return;}atomicAdd(&transitionControl.transitionRows,1u);let descriptor=metric.reserved;metric.reserved=header.count;metrics[band]=metric;metric.reserved=descriptor;if(header.count==0u||band>arrayLength(&transitionAdjacency)/MAX_TETRA||band*MAX_TETRA>arrayLength(&transitionAdjacency)-header.count){transitionFail(TRANSITION_CAPACITY,band);return;}for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){transitionFail(TRANSITION_SOURCE,band);return;}let a=transitionNeighbor(band,selectors.x,metric);let b=transitionNeighbor(band,selectors.y,metric);let c=transitionNeighbor(band,selectors.z,metric);if(atomicLoad(&transitionControl.flags)!=0u){return;}transitionAdjacency[band*MAX_TETRA+local]=TransitionAdjacency(band,a,b,c);atomicAdd(&transitionControl.adjacencyCount,1u);}}
@compute @workgroup_size(64)fn buildSupport5TransitionAdjacency(@builtin(global_invocation_id)g:vec3u){let band=transitionControl.support4NodeEnd+g.x;if(band>=transitionControl.support5NodeEnd||band>=arrayLength(&rows)){return;}if(band>=arrayLength(&metrics)){transitionFail(TRANSITION_CAPACITY,band);return;}var metric=metrics[band];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){transitionFail(TRANSITION_DESCRIPTOR,band);return;}let header=tetraHeaders[metric.topology];if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first||header.count>MAX_TETRA){transitionFail(TRANSITION_SOURCE,band);return;}let uniform=(header.flags&1u)!=0u;if(uniform){metric.reserved=0u;metrics[band]=metric;return;}atomicAdd(&transitionControl.transitionRows,1u);let descriptor=metric.reserved;metric.reserved=header.count;metrics[band]=metric;metric.reserved=descriptor;if(header.count==0u||band>arrayLength(&transitionAdjacency)/MAX_TETRA||band*MAX_TETRA>arrayLength(&transitionAdjacency)-header.count){transitionFail(TRANSITION_CAPACITY,band);return;}for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){transitionFail(TRANSITION_SOURCE,band);return;}let a=transitionNeighbor(band,selectors.x,metric);let b=transitionNeighbor(band,selectors.y,metric);let c=transitionNeighbor(band,selectors.z,metric);if(atomicLoad(&transitionControl.flags)!=0u){return;}transitionAdjacency[band*MAX_TETRA+local]=TransitionAdjacency(band,a,b,c);atomicAdd(&transitionControl.adjacencyCount,1u);}}
@compute @workgroup_size(64)fn buildSupport6TransitionAdjacency(@builtin(global_invocation_id)g:vec3u){let band=transitionControl.support5NodeEnd+g.x;if(band>=transitionControl.support7NodeEnd||band>=arrayLength(&rows)){return;}if(band>=arrayLength(&metrics)){transitionFail(TRANSITION_CAPACITY,band);return;}var metric=metrics[band];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){transitionFail(TRANSITION_DESCRIPTOR,band);return;}let header=tetraHeaders[metric.topology];if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first||header.count>MAX_TETRA){transitionFail(TRANSITION_SOURCE,band);return;}let uniform=(header.flags&1u)!=0u;if(uniform){metric.reserved=0u;metrics[band]=metric;return;}atomicAdd(&transitionControl.transitionRows,1u);let descriptor=metric.reserved;metric.reserved=header.count;metrics[band]=metric;metric.reserved=descriptor;if(header.count==0u||band>arrayLength(&transitionAdjacency)/MAX_TETRA||band*MAX_TETRA>arrayLength(&transitionAdjacency)-header.count){transitionFail(TRANSITION_CAPACITY,band);return;}for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){transitionFail(TRANSITION_SOURCE,band);return;}let a=transitionNeighbor(band,selectors.x,metric);let b=transitionNeighbor(band,selectors.y,metric);let c=transitionNeighbor(band,selectors.z,metric);if(atomicLoad(&transitionControl.flags)!=0u){return;}transitionAdjacency[band*MAX_TETRA+local]=TransitionAdjacency(band,a,b,c);atomicAdd(&transitionControl.adjacencyCount,1u);}}
@compute @workgroup_size(1)fn gateTransitionTransfer(){atomicStore(&transitionControl.ready,0u);transitionControl.transferReady=0u;let transitionFlags=atomicLoad(&transitionControl.flags);if(transitionFlags==0u&&atomicLoad(&transitionControl.hierarchyReady)==VALID&&atomicLoad(&control.flags)==0u){atomicStore(&transitionControl.ready,VALID);transitionControl.transferReady=VALID;return;}let first=atomicLoad(&transitionControl.firstError);fail(BAD_ROW,select(0u,first,first!=INVALID));}
fn appendIncidence(row:u32,face:u32){if(row>=p.rowCapacity||row>=arrayLength(&incidence)){fail(CAPACITY,row);return;}let local=atomicAdd(&incidence[row],1u);if(local>=p.axisStride){fail(CAPACITY,row);return;}let at=p.rowCapacity+row*p.axisStride+local;if(at>=arrayLength(&incidence)){fail(CAPACITY,row);return;}atomicStore(&incidence[at],face);atomicAdd(&control.incidenceCount,1u);}
@compute @workgroup_size(64)fn emitBandFaces(@builtin(global_invocation_id)g:vec3u){let band=g.x;if(band>=transitionControl.support2End||band>=arrayLength(&rows)||atomicLoad(&control.flags)!=0u){return;}let row=rows[band];let origin=coord(row.cell);for(var axis=0u;axis<3u;axis+=1u){let boundary=origin[axis]+row.size;if(boundary>=p.dims[axis]){if(boundary==p.dims[axis]&&(p.closedBoundaryMask&positiveBoundaryBit(axis))==0u){let slot=band*p.ownedFacesPerRow+axis*4u;if(slot>=p.faceCapacity||slot>=arrayLength(&faces)){fail(CAPACITY,row.cell);return;}var centroid=vec3f(origin)+.5*f32(row.size);centroid[axis]=f32(boundary);var keyCell=origin;keyCell[axis]=boundary-1u;let stable=3u*cell(keyCell)+axis;faces[slot]=Face(band,INVALID,axis|(row.size<<2u),stable,vec4f(0),vec4f(centroid,1.),0.,f32(row.size*row.size),LIVE,0u);appendIncidence(band,slot);atomicAdd(&control.faceCount,1u);}continue;}let ta=(axis+1u)%3u;let tb=(axis+2u)%3u;let sampleCount=select(1u,4u,row.size>1u);let half=max(1u,row.size/2u);var emitted:array<u32,4>;for(var local=0u;local<sampleCount;local+=1u){var probe=origin;probe[axis]=boundary;probe[ta]+=select(0u,half,(local&1u)!=0u);probe[tb]+=select(0u,half,(local&2u)!=0u);let neighborOwner=ownerAt(probe);if(neighborOwner.valid==0u){fail(BAD_ROW,cell(probe));return;}let neighbor=rowOf(cell(neighborOwner.origin));if(neighbor==INVALID||neighbor>=transitionControl.endpointEnd||neighbor>=arrayLength(&rows)){adjacencyFail(band,DETAIL_MISSING_ROW);fail(BAD_ROW,row.cell);continue;}let other=rows[neighbor];var duplicate=false;for(var prior=0u;prior<local;prior+=1u){duplicate=duplicate||emitted[prior]==neighbor+1u;}if(duplicate){continue;}emitted[local]=neighbor+1u;let otherOrigin=coord(other.cell);let low=max(origin,otherOrigin);let high=min(origin+vec3u(row.size),otherOrigin+vec3u(other.size));if(high[ta]<=low[ta]||high[tb]<=low[tb]){fail(BAD_FACE,row.cell);return;}let area=f32(high[ta]-low[ta])*f32(high[tb]-low[tb]);var centroid=.5*(vec3f(low)+vec3f(high));centroid[axis]=f32(boundary);var keyCell=low;keyCell[axis]=boundary-1u;let stable=3u*cell(keyCell)+axis;let slot=band*p.ownedFacesPerRow+axis*4u+local;if(slot>=p.faceCapacity||slot>=arrayLength(&faces)){fail(CAPACITY,stable);return;}faces[slot]=Face(band,neighbor,axis|(min(row.size,other.size)<<2u),stable,vec4f(0),vec4f(centroid,1.),0.,area,LIVE,0u);appendIncidence(band,slot);appendIncidence(neighbor,slot);atomicAdd(&control.faceCount,1u);}}}
// Regular faces have spatial ownership: the row on the negative coordinate
// side emits the shared face. S0-S2 emit their own positive sides above, but
// an S2 row can have an S3+ owner on its negative side. Emit exactly those
// inward seams here. Deep-deep and deep-world faces are outside the marched
// narrow band and must not recursively extend it.
@compute @workgroup_size(64)fn emitDeepBandFaces(@builtin(global_invocation_id)g:vec3u){
 let band=transitionControl.support2End+g.x;
 if(band>=transitionControl.endpointEnd||band>=arrayLength(&rows)||atomicLoad(&control.flags)!=0u){return;}
 let row=rows[band];let endpointOnly=(row.flags&ROW_SUPPORT3_ENDPOINT)!=0u;let origin=coord(row.cell);
 for(var axis=0u;axis<3u;axis+=1u){
  let boundary=origin[axis]+row.size;if(boundary>=p.dims[axis]){continue;}
  let ta=(axis+1u)%3u;let tb=(axis+2u)%3u;let sampleCount=select(1u,4u,row.size>1u);let half=max(1u,row.size/2u);var emitted:array<u32,4>;
  for(var local=0u;local<sampleCount;local+=1u){
   var probe=origin;probe[axis]=boundary;probe[ta]+=select(0u,half,(local&1u)!=0u);probe[tb]+=select(0u,half,(local&2u)!=0u);
   let neighborOwner=ownerAt(probe);if(neighborOwner.valid==0u){fail(BAD_ROW,cell(probe));return;}
   let neighbor=rowOf(cell(neighborOwner.origin));
   if(neighbor==INVALID||neighbor>=transitionControl.endpointEnd||neighbor>=arrayLength(&rows)){if(!endpointOnly){adjacencyFail(band,DETAIL_MISSING_ROW);fail(BAD_ROW,row.cell);}continue;}
   if(neighbor>=transitionControl.support2End){continue;}
   let other=rows[neighbor];var duplicate=false;for(var prior=0u;prior<local;prior+=1u){duplicate=duplicate||emitted[prior]==neighbor+1u;}if(duplicate){continue;}emitted[local]=neighbor+1u;
   let otherOrigin=coord(other.cell);let low=max(origin,otherOrigin);let high=min(origin+vec3u(row.size),otherOrigin+vec3u(other.size));
   if(high[ta]<=low[ta]||high[tb]<=low[tb]){fail(BAD_FACE,row.cell);return;}
   let area=f32(high[ta]-low[ta])*f32(high[tb]-low[tb]);var centroid=.5*(vec3f(low)+vec3f(high));centroid[axis]=f32(boundary);
   var keyCell=low;keyCell[axis]=boundary-1u;let stable=3u*cell(keyCell)+axis;let slot=band*p.ownedFacesPerRow+axis*4u+local;
   if(slot>=p.faceCapacity||slot>=arrayLength(&faces)){fail(CAPACITY,stable);return;}
   faces[slot]=Face(band,neighbor,axis|(min(row.size,other.size)<<2u),stable,vec4f(0),vec4f(centroid,1.),0.,area,LIVE,0u);
   appendIncidence(band,slot);appendIncidence(neighbor,slot);atomicAdd(&control.faceCount,1u);
  }
 }
}
fn fineHashKey(key:u32)->u32{return ((key^(key>>16u))*0x9e3779b1u)&(fp.hashCapacity-1u);}
fn fineBrickKey(q:vec3u)->u32{return q.x+fp.brickDims.x*(q.y+fp.brickDims.y*q.z);}
fn finePage(key:u32)->u32{if(fp.hashCapacity==0u||(fp.hashCapacity&(fp.hashCapacity-1u))!=0u||arrayLength(&fineHash)<fp.hashCapacity*2u){return INVALID;}let start=fineHashKey(key);for(var probe=0u;probe<32u;probe+=1u){if(probe>=fp.maxProbes){break;}let slot=(start+probe)&(fp.hashCapacity-1u);let stored=fineHash[slot*2u];if(stored==key){let id=fineHash[slot*2u+1u];if(id>=fp.pageCapacity){return INVALID;}let base=id*10u;if(base+2u>=arrayLength(&metadata)||metadata[base]!=id||metadata[base+1u]!=key||metadata[base+2u]!=fp.generation){return INVALID;}return id;}if(stored==INVALID){return INVALID;}}return INVALID;}
// At a world-plane face centroid the centered trilinear product stencil has
// one virtual sample center. Preserve its original product weight and mirror
// only that one layer to the interior scalar (homogeneous-Neumann/even phi).
// Open/closed velocity behavior is deliberately irrelevant to scalar phi.
fn loadFineScalarExtended(virtualIndex:vec3i)->vec2f{if(any(fp.sampleDims==vec3u(0u))){return vec2f(0.);}var q=virtualIndex;for(var axis=0u;axis<3u;axis+=1u){let limit=i32(fp.sampleDims[axis]);if(q[axis]<0){if(q[axis]!=-1){return vec2f(0.);}q[axis]=0;}else if(q[axis]>=limit){if(q[axis]!=limit){return vec2f(0.);}q[axis]=limit-1;}}let uq=vec3u(q);let brick=uq/fp.brickResolution;if(any(brick>=fp.brickDims)){return vec2f(0.);}let local=uq-brick*fp.brickResolution;let id=finePage(fineBrickKey(brick));if(id==INVALID){return vec2f(0.);}let index=id*fp.samplesPerBrick+local.x+fp.brickResolution*(local.y+fp.brickResolution*local.z);if(index>=arrayLength(&sampleFlags)||index>=arrayLength(&finePhi)||(sampleFlags[index]&1u)==0u){return vec2f(0.);}let value=finePhi[index];if(!finite(value)){return vec2f(0.);}return vec2f(value,1.);}
fn finePhiAtFaceCentroid(pointGrid:vec3f)->vec2f{if(fp.generation!=p.generation||fp.fineFactor==0u||fp.brickResolution==0u||fp.samplesPerBrick!=fp.brickResolution*fp.brickResolution*fp.brickResolution||any(fp.sampleDims!=p.dims*fp.fineFactor)){return vec2f(0.);}let coarseWidth=fp.fineWidth*f32(fp.fineFactor);if(!finite(coarseWidth)||coarseWidth<=0.){return vec2f(0.);}let world=fp.domainOrigin+pointGrid*coarseWidth;let raw=(world-fp.domainOrigin)/fp.fineWidth-vec3f(.5);let wall=vec3f(fp.sampleDims)-vec3f(.5);if(any(raw<vec3f(-.5))||any(raw>wall)){return vec2f(0.);}let base=vec3i(floor(raw));let fraction=fract(raw);var result=0.;for(var z=0;z<2;z+=1){for(var y=0;y<2;y+=1){for(var x=0;x<2;x+=1){let weight=select(1.-fraction.x,fraction.x,x==1)*select(1.-fraction.y,fraction.y,y==1)*select(1.-fraction.z,fraction.z,z==1);if(weight==0.){continue;}let sample=loadFineScalarExtended(base+vec3i(x,y,z));if(sample.y==0.){return vec2f(0.);}result+=weight*sample.x;}}}return select(vec2f(0.),vec2f(result,1.),finite(result));}
// Section 5 stores phi at octree cell centers and Section 6.2 supplies the
// local cube/Delaunay interpolant. The fine SPGrid is a narrow band, so a
// missing fine stencil is resolved from that current compact octree field;
// it is never replaced by a fabricated sign or distance.
fn exactCoarseCellScalar(origin:vec3u,size:u32)->vec2f{if(!validCoarse()||size==0u||any(origin+vec3u(size)>p.dims)){return vec2f(0.);}let entry=coarseEntryRecord(coarseSlot(cell(origin),size));return vec2f(entry.x,entry.w);}
fn coarseCellSeedScalar(origin:vec3u,size:u32)->vec2f{let entry=coarseCellSeedRecord(origin,size);return vec2f(entry.x,entry.w);}
fn coarseCellScalar(origin:vec3u,size:u32)->vec2f{if(size==0u||any(origin+vec3u(size)>p.dims)){return vec2f(0.);}let band=rowOf(cell(origin));if(band!=INVALID&&band<arrayLength(&rows)){let row=rows[band];if(row.cell==cell(origin)&&row.size==size&&(row.flags&ROW_PHI)!=0u&&finite(row.representativePhi)){return vec2f(row.representativePhi,1.);}}return exactCoarseCellScalar(origin,size);}
fn compactScalar(origin:vec3i,size:u32)->vec2f{let extended=velocityExtendedOrigin(origin,size);if(extended.w<0){return vec2f(0.);}return coarseCellScalar(vec3u(extended.xyz),size);}
fn selectorScalar(anchor:u32,selector:u32,transform:u32)->vec2f{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return vec2f(0.);}let row=rows[anchor];let v=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(v.xyz,transform);let sizeFloat=f32(row.size)*v.w;let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return vec2f(0.);}return compactScalar(vec3i(round(point-.5*f32(size))),size);}
fn publishedBandScalar(rowIndex:u32)->vec2f{if(rowIndex==INVALID||rowIndex>=arrayLength(&rows)){return vec2f(0.);}let row=rows[rowIndex];if((row.flags&ROW_PHI)==0u||!finite(row.representativePhi)){return vec2f(0.);}return vec2f(row.representativePhi,1.);}
fn compactPublishedBandScalar(origin:vec3i,size:u32)->vec2f{let high=origin+vec3i(i32(size));let boundary=any(origin<vec3i(0))||any(high>vec3i(p.dims));let extended=velocityExtendedOrigin(origin,size);if(extended.w<0){return vec2f(0.);}let q=vec3u(extended.xyz);var candidateSize=size;loop{let candidateOrigin=(q/vec3u(candidateSize))*vec3u(candidateSize);let band=rowOf(cell(candidateOrigin));if(band!=INVALID&&band<arrayLength(&rows)){let row=rows[band];if(row.cell==cell(candidateOrigin)&&row.size==candidateSize){return publishedBandScalar(band);}}if(!boundary||candidateSize>=p.maximumLeaf){break;}candidateSize*=2u;}return vec2f(0.);}
fn localTetraBandScalar(anchor:u32,local:u32,lane:u32,selector:u32,transform:u32)->vec2f{let at=anchor*MAX_TETRA+local;if(at<arrayLength(&transitionAdjacency)){let adjacency=transitionAdjacency[at];if(adjacency.band==anchor){let neighbor=select(select(adjacency.c,adjacency.b,lane==1u),adjacency.a,lane==0u);if(neighbor!=INVALID){return publishedBandScalar(neighbor);}}}if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return vec2f(0.);}let row=rows[anchor];let v=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(v.xyz,transform);let sizeFloat=f32(row.size)*v.w;let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return vec2f(0.);}let origin=vec3i(round(point-.5*f32(size)));if(all(origin>=vec3i(0))&&all(origin+vec3i(i32(size))<=vec3i(p.dims))){return vec2f(0.);}return compactPublishedBandScalar(origin,size);}
fn coarsePhiAtPoint(anchor:u32,pointGrid:vec3f)->vec2f{if(anchor==INVALID||anchor>=arrayLength(&rows)||anchor>=arrayLength(&metrics)){return vec2f(0.);}let row=rows[anchor];let origin=coord(row.cell);let anchorPhi=publishedBandScalar(anchor);if(anchorPhi.y==0.){return vec2f(0.);}let metric=metrics[anchor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){return vec2f(0.);}let header=tetraHeaders[metric.topology];if((header.flags&1u)!=0u){let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);if(any(t<vec3f(-2e-6))||any(t>vec3f(1.000002))){return vec2f(0.);}var result=0.;for(var corner=0u;corner<8u;corner+=1u){let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);if(weight==0.){continue;}let cornerOrigin=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u))*i32(row.size);let value=compactPublishedBandScalar(cornerOrigin,row.size);if(value.y==0.){return vec2f(0.);}result+=weight*value.x;}return select(vec2f(0.),vec2f(result,1.),finite(result));}if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return vec2f(0.);}let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return vec2f(0.);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}let a=localTetraBandScalar(anchor,local,0u,selectors.x,metric.transformFlags&63u);let b=localTetraBandScalar(anchor,local,1u,selectors.y,metric.transformFlags&63u);let c=localTetraBandScalar(anchor,local,2u,selectors.z,metric.transformFlags&63u);if(a.y==0.||b.y==0.||c.y==0.){return vec2f(0.);}let result=weights.x*anchorPhi.x+weights.y*a.x+weights.z*b.x+weights.w*c.x;return select(vec2f(0.),vec2f(result,1.),finite(result));}return vec2f(0.);}
fn noPhiDiagnostic(anchor:u32,path:u32)->PhiDiagnostic{return PhiDiagnostic(vec3i(0),0u,anchor,path,INVALID,INVALID,0u);}
fn phiDiagnostic(origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32,cause:u32,detail:u32)->PhiDiagnostic{return PhiDiagnostic(origin,size,anchor,path,selector,cause,detail);}
fn diagnoseCellScalar(origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32)->PhiDiagnostic{if(size==0u||any(origin<vec3i(0))||any(origin+vec3i(i32(size))>vec3i(p.dims))){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_INVALID_METRIC,1u);}let unsignedOrigin=vec3u(origin);let encodedCell=cell(unsignedOrigin);let band=rowOf(encodedCell);if(band!=INVALID&&band<arrayLength(&rows)){let row=rows[band];if(row.cell==encodedCell&&row.size==size&&(row.flags&ROW_PHI)!=0u&&finite(row.representativePhi)){return noPhiDiagnostic(anchor,path);}}if(exactCoarseCellScalar(unsignedOrigin,size).y!=0.){return noPhiDiagnostic(anchor,path);}if(band==INVALID||band>=arrayLength(&rows)){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_MISSING_ROW,0u);}return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_EXACT_COARSE_MISS,select(2u,1u,rows[band].cell==encodedCell&&rows[band].size==size));}
fn diagnoseCompactScalar(origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32)->PhiDiagnostic{let extended=velocityExtendedOrigin(origin,size);if(extended.w<0){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_INVALID_METRIC,2u);}return diagnoseCellScalar(extended.xyz,size,anchor,path,selector);}
fn diagnoseSelectorScalar(anchor:u32,selector:u32,transform:u32,detail:u32)->PhiDiagnostic{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return phiDiagnostic(vec3i(0),0u,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail);}let row=rows[anchor];let vertex=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(vertex.xyz,transform);let sizeFloat=f32(row.size)*vertex.w;if(!finite(sizeFloat)||sizeFloat<=0.||!finite(point.x)||!finite(point.y)||!finite(point.z)){return phiDiagnostic(vec3i(0),0u,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail|0x10000u);}let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return phiDiagnostic(vec3i(round(point)),size,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail|0x20000u);}return diagnoseCompactScalar(vec3i(round(point-.5*f32(size))),size,anchor,PHI_PATH_DELAUNAY,selector);}
fn diagnosePublishedBandRow(rowIndex:u32,origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32)->PhiDiagnostic{if(rowIndex==INVALID){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_MISSING_ROW,0x6000u);}if(rowIndex>=arrayLength(&rows)){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_MISSING_ROW,0x7000u|(rowIndex&0xfffu));}let row=rows[rowIndex];if(row.cell!=cell(vec3u(origin))||row.size!=size){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_MISSING_ROW,0x8000u|(rowIndex&0xfffu));}if((row.flags&ROW_PHI)==0u||!finite(row.representativePhi)){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_EXACT_COARSE_MISS,0x9000u|(rowIndex&0xfffu));}return noPhiDiagnostic(anchor,path);}
fn diagnoseCompactPublishedBandScalar(origin:vec3i,size:u32,anchor:u32,path:u32,selector:u32)->PhiDiagnostic{let extended=velocityExtendedOrigin(origin,size);if(extended.w<0){return phiDiagnostic(origin,size,anchor,path,selector,PHI_CAUSE_INVALID_METRIC,2u);}let q=vec3u(extended.xyz);return diagnosePublishedBandRow(rowOf(cell(q)),vec3i(q),size,anchor,path,selector);}
fn diagnoseLocalTetraBandScalar(anchor:u32,local:u32,lane:u32,selector:u32,transform:u32,detail:u32)->PhiDiagnostic{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return phiDiagnostic(vec3i(0),0u,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail);}let row=rows[anchor];let vertex=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(vertex.xyz,transform);let sizeFloat=f32(row.size)*vertex.w;if(!finite(sizeFloat)||sizeFloat<=0.||!finite(point.x)||!finite(point.y)||!finite(point.z)){return phiDiagnostic(vec3i(0),0u,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail|0x1000u);}let size=u32(round(sizeFloat));let origin=vec3i(round(point-.5*f32(size)));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return phiDiagnostic(origin,size,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_SELECTOR,detail|0x2000u);}let at=anchor*MAX_TETRA+local;if(at>=arrayLength(&transitionAdjacency)){return phiDiagnostic(origin,size,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_METRIC,detail|0x3000u);}let adjacency=transitionAdjacency[at];if(adjacency.band!=anchor){return phiDiagnostic(origin,size,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_INVALID_METRIC,detail|0x4000u);}let neighbor=select(select(adjacency.c,adjacency.b,lane==1u),adjacency.a,lane==0u);if(neighbor!=INVALID){return diagnosePublishedBandRow(neighbor,origin,size,anchor,PHI_PATH_DELAUNAY,selector);}if(all(origin>=vec3i(0))&&all(origin+vec3i(i32(size))<=vec3i(p.dims))){return phiDiagnostic(origin,size,anchor,PHI_PATH_DELAUNAY,selector,PHI_CAUSE_MISSING_ROW,detail|0x5000u);}return diagnoseCompactPublishedBandScalar(origin,size,anchor,PHI_PATH_DELAUNAY,selector);}
fn diagnoseCoarsePhiAtPoint(anchor:u32,pointGrid:vec3f)->PhiDiagnostic{if(anchor==INVALID||anchor>=arrayLength(&rows)||anchor>=arrayLength(&metrics)){return phiDiagnostic(vec3i(0),0u,anchor,PHI_PATH_ANCHOR,INVALID,PHI_CAUSE_INVALID_METRIC,3u);}let row=rows[anchor];let origin=coord(row.cell);let anchorDiagnostic=diagnosePublishedBandRow(anchor,vec3i(origin),row.size,anchor,PHI_PATH_ANCHOR,INVALID);if(anchorDiagnostic.cause!=INVALID){return anchorDiagnostic;}let metric=metrics[anchor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){return phiDiagnostic(vec3i(origin),row.size,anchor,PHI_PATH_ANCHOR,INVALID,PHI_CAUSE_INVALID_METRIC,4u);}let header=tetraHeaders[metric.topology];if((header.flags&1u)!=0u){let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);if(any(t<vec3f(-2e-6))||any(t>vec3f(1.000002))){return phiDiagnostic(low,row.size,anchor,PHI_PATH_CUBE,INVALID,PHI_CAUSE_INVALID_METRIC,5u);}for(var corner=0u;corner<8u;corner+=1u){let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);if(weight==0.){continue;}let cornerOrigin=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u))*i32(row.size);let phiRecord=diagnoseCompactPublishedBandScalar(cornerOrigin,row.size,anchor,PHI_PATH_CUBE,corner);if(phiRecord.cause!=INVALID){return phiRecord;}}return phiDiagnostic(low,row.size,anchor,PHI_PATH_CUBE,INVALID,PHI_CAUSE_INVALID_SELECTOR,6u);}if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return phiDiagnostic(vec3i(origin),row.size,anchor,PHI_PATH_DELAUNAY,INVALID,PHI_CAUSE_INVALID_SELECTOR,7u);}let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return phiDiagnostic(vec3i(origin),row.size,anchor,PHI_PATH_DELAUNAY,INVALID,PHI_CAUSE_INVALID_SELECTOR,0x80000u|local);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}var phiRecord=diagnoseLocalTetraBandScalar(anchor,local,0u,selectors.x,metric.transformFlags&63u,local|(0u<<8u));if(phiRecord.cause!=INVALID){return phiRecord;}phiRecord=diagnoseLocalTetraBandScalar(anchor,local,1u,selectors.y,metric.transformFlags&63u,local|(1u<<8u));if(phiRecord.cause!=INVALID){return phiRecord;}phiRecord=diagnoseLocalTetraBandScalar(anchor,local,2u,selectors.z,metric.transformFlags&63u,local|(2u<<8u));if(phiRecord.cause!=INVALID){return phiRecord;}return phiDiagnostic(vec3i(origin),row.size,anchor,PHI_PATH_DELAUNAY,INVALID,PHI_CAUSE_INVALID_SELECTOR,0x90000u|local);}return phiDiagnostic(vec3i(origin),row.size,anchor,PHI_PATH_DELAUNAY,INVALID,PHI_CAUSE_INVALID_SELECTOR,0xa0000u);}
@compute @workgroup_size(64)fn sampleBandFacePhi(@builtin(global_invocation_id)g:vec3u){let index=g.x;if(index>=p.faceCapacity||index>=arrayLength(&faces)||atomicLoad(&control.flags)!=0u){return;}var face=faces[index];if((face.flags&LIVE)==0u){return;}let sampled=finePhiAtFaceCentroid(face.centroid.xyz);if(sampled.y==0.){return;}face.phi=sampled.x;face.flags|=PHI_VALID;faces[index]=face;}
fn rowMarchSign(row:Row)->f32{let air=(row.flags&ROW_COARSE_AIR)!=0u;let liquid=(row.flags&ROW_COARSE_LIQUID)!=0u;return select(select(0.,-1.,liquid&&!air),1.,air&&!liquid);}
@compute @workgroup_size(64)fn initializeBandRowPhi(@builtin(global_invocation_id)g:vec3u){let rowIndex=g.x;if(rowIndex>=p.rowCapacity||rowIndex>=arrayLength(&rowVelocities)){return;}let liveRow=rowIndex<atomicLoad(&control.rowCount)&&rowIndex<arrayLength(&rows);if(!liveRow){rowVelocities[rowIndex]=vec4f(0.);return;}let row=rows[rowIndex];let center=vec3f(coord(row.cell))+.5*f32(row.size);var sampled=finePhiAtFaceCentroid(center);if(sampled.y==0.){sampled=coarseCellSeedScalar(coord(row.cell),row.size);}rowVelocities[rowIndex]=select(vec4f(0.,0.,0.,1.),vec4f(sampled.x,1.,1.,1.),sampled.y!=0.);}
@compute @workgroup_size(64)fn seedBandRowPhiFromFaces(@builtin(global_invocation_id)g:vec3u){let rowIndex=g.x;if(rowIndex>=atomicLoad(&control.rowCount)||rowIndex>=arrayLength(&rows)||rowIndex>=arrayLength(&rowVelocities)){return;}var state=rowVelocities[rowIndex];if(state.y!=0.){return;}let row=rows[rowIndex];let sign=rowMarchSign(row);let center=vec3f(coord(row.cell))+.5*f32(row.size);var best=3.402823e38;var signedBest=0.;let count=min(atomicLoad(&incidence[rowIndex]),p.axisStride);for(var local=0u;local<count;local+=1u){let faceIndex=atomicLoad(&incidence[p.rowCapacity+rowIndex*p.axisStride+local]);if(faceIndex>=arrayLength(&faces)){continue;}let face=faces[faceIndex];if((face.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||!finite(face.phi)){continue;}let distance=length(center-face.centroid.xyz)*fp.fineWidth*f32(fp.fineFactor);if(sign==0.){let candidate=abs(face.phi)+distance;if(candidate<best){best=candidate;signedBest=select(-candidate,candidate,face.phi>=0.);}}else{let candidate=max(0.,sign*face.phi+distance);if(candidate<best){best=candidate;signedBest=sign*candidate;}}}if(best<3.402823e38){rowVelocities[rowIndex]=vec4f(signedBest,1.,0.,1.);}}
fn bandCenter(row:u32)->vec3f{return vec3f(coord(rows[row].cell))+.5*f32(rows[row].size);}
fn sameMarchSide(value:vec4f,sign:f32)->bool{return value.y!=0.&&finite(value.x)&&sign*value.x>=-1e-5;}
fn solveColumns3(a:vec3f,b:vec3f,c:vec3f,rhs:vec3f)->vec4f{let determinant=dot(a,cross(b,c));if(!finite(determinant)||abs(determinant)<=1e-10){return vec4f(0.);}return vec4f(dot(rhs,cross(b,c)),dot(a,cross(rhs,c)),dot(a,cross(b,rhs)),determinant);}
fn solveTranspose3(a:vec3f,b:vec3f,c:vec3f,rhs:vec3f)->vec4f{let determinant=dot(a,cross(b,c));if(!finite(determinant)||abs(determinant)<=1e-10){return vec4f(0.);}let value=(rhs.x*cross(b,c)+rhs.y*cross(c,a)+rhs.z*cross(a,b))/determinant;return vec4f(value,1.);}
// Section 5 requires the solid angle incident at the current cell centre to
// be at most pi/2. For Omega=2 atan2(D,N), this is the stable N>=D test.
fn nonobtuseIncidentSolidAngle(a:vec3f,b:vec3f,c:vec3f)->bool{let determinant=abs(dot(a,cross(b,c)));let la=length(a);let lb=length(b);let lc=length(c);let denominator=la*lb*lc+dot(a,b)*lc+dot(a,c)*lb+dot(b,c)*la;let scale=max(1.,max(determinant,abs(denominator)));return finite(determinant)&&finite(denominator)&&determinant>1e-10&&denominator+2e-5*scale>=determinant;}
fn localTetraEikonal(rowIndex:u32,sign:f32)->f32{var best=3.402823e38;if(rowIndex>=arrayLength(&metrics)){return best;}let count=min(metrics[rowIndex].reserved,MAX_TETRA);let center=bandCenter(rowIndex);let unit=fp.fineWidth*f32(fp.fineFactor);for(var local=0u;local<count;local+=1u){let at=rowIndex*MAX_TETRA+local;if(at>=arrayLength(&transitionAdjacency)){break;}let adjacency=transitionAdjacency[at];if(adjacency.band!=rowIndex||adjacency.a>=arrayLength(&rows)||adjacency.b>=arrayLength(&rows)||adjacency.c>=arrayLength(&rows)||adjacency.a>=arrayLength(&rowVelocities)||adjacency.b>=arrayLength(&rowVelocities)||adjacency.c>=arrayLength(&rowVelocities)){continue;}let va=rowVelocities[adjacency.a];let vb=rowVelocities[adjacency.b];let vc=rowVelocities[adjacency.c];if(!sameMarchSide(va,sign)||!sameMarchSide(vb,sign)||!sameMarchSide(vc,sign)){continue;}let a=(bandCenter(adjacency.a)-center)*unit;let b=(bandCenter(adjacency.b)-center)*unit;let c=(bandCenter(adjacency.c)-center)*unit;if(!nonobtuseIncidentSolidAngle(a,b,c)){fail(BAD_PHI,rows[rowIndex].cell);return best;}let known=vec3f(abs(va.x),abs(vb.x),abs(vc.x));let av=solveTranspose3(a,b,c,known);let bv=solveTranspose3(a,b,c,vec3f(1.));if(av.w==0.||bv.w==0.){continue;}let aa=dot(bv.xyz,bv.xyz);let bb=dot(av.xyz,bv.xyz);let cc=dot(av.xyz,av.xyz)-1.;let discriminant=bb*bb-aa*cc;if(!finite(aa)||aa<=1e-12||!finite(discriminant)||discriminant<0.){continue;}let candidate=(bb+sqrt(max(0.,discriminant)))/aa;let lower=max(known.x,max(known.y,known.z));if(!finite(candidate)||candidate+1e-5<lower){continue;}let ray=solveColumns3(a,b,c,-(av.xyz-candidate*bv.xyz));if(ray.w==0.){continue;}let weights=ray.xyz/ray.w;let sum=weights.x+weights.y+weights.z;if(!all(vec3<bool>(finite(weights.x),finite(weights.y),finite(weights.z)))||sum<=1e-8||any(weights/sum<vec3f(-2e-5))){continue;}best=min(best,candidate);}return best;}
fn uniformEikonal(rowIndex:u32,sign:f32)->f32{var axisValues=vec3f(3.402823e38);let count=min(atomicLoad(&incidence[rowIndex]),p.axisStride);for(var local=0u;local<count;local+=1u){let faceIndex=atomicLoad(&incidence[p.rowCapacity+rowIndex*p.axisStride+local]);if(faceIndex>=arrayLength(&faces)){continue;}let face=faces[faceIndex];let axis=face.axisSpan&3u;let neighbor=select(face.negativeRow,face.positiveRow,face.negativeRow==rowIndex);if(axis>=3u||neighbor==INVALID||neighbor>=arrayLength(&rowVelocities)){continue;}let source=rowVelocities[neighbor];if(sameMarchSide(source,sign)){axisValues[axis]=min(axisValues[axis],abs(source.x));}}var a=min(axisValues.x,min(axisValues.y,axisValues.z));var c=max(axisValues.x,max(axisValues.y,axisValues.z));var b=axisValues.x+axisValues.y+axisValues.z-a-c;if(a>=3.402823e38){return a;}let h=fp.fineWidth*f32(fp.fineFactor*rows[rowIndex].size);var result=a+h;if(b<3.402823e38&&result>b){let disc=2.*h*h-(a-b)*(a-b);if(disc>=0.){result=.5*(a+b+sqrt(disc));}}if(c<3.402823e38&&result>c){let disc=3.*h*h-(a-b)*(a-b)-(a-c)*(a-c)-(b-c)*(b-c);if(disc>=0.){result=(a+b+c+sqrt(disc))/3.;}}return result;}
fn lowerSimplexCandidate(rowIndex:u32,sign:f32)->f32{var best=3.402823e38;let center=bandCenter(rowIndex);let unit=fp.fineWidth*f32(fp.fineFactor);var edge=INVALID;if(rowIndex<arrayLength(&pointStatus)){edge=atomicLoad(&pointStatus[rowIndex]);}var traversed=0u;loop{if(edge==INVALID||edge>=arrayLength(&transientPowerIncidences)){break;}let item=transientPowerIncidences[edge];let parent=item.face;if(parent<arrayLength(&rowVelocities)&&parent<arrayLength(&rows)){let source=rowVelocities[parent];if(sameMarchSide(source,sign)){best=min(best,abs(source.x)+length(center-bandCenter(parent))*unit);}}edge=bitcast<u32>(item.sign);traversed+=1u;if(traversed>p.rowCapacity*MAX_GUARDS){break;}}let count=min(atomicLoad(&incidence[rowIndex]),p.axisStride);for(var local=0u;local<count;local+=1u){let faceIndex=atomicLoad(&incidence[p.rowCapacity+rowIndex*p.axisStride+local]);if(faceIndex>=arrayLength(&faces)){continue;}let face=faces[faceIndex];let neighbor=select(face.negativeRow,face.positiveRow,face.negativeRow==rowIndex);if(neighbor==INVALID||neighbor>=arrayLength(&rowVelocities)||neighbor>=arrayLength(&rows)){continue;}let source=rowVelocities[neighbor];if(sameMarchSide(source,sign)){best=min(best,abs(source.x)+length(center-bandCenter(neighbor))*unit);}}return best;}
// Aanjaneya et al. (2017), Section 5 stores signed distance outside the fine
// interface band on the coarse octree. Mixed deep-closure rows own no LIVE
// regular faces, so they inherit a measured sign through the bounded support
// edges captured while that coarse interpolation topology is constructed.
fn nearestSignedNeighbor(rowIndex:u32,current:vec4f)->vec2f{var best=select(3.402823e38,abs(current.x),current.y!=0.&&finite(current.x));var signedBest=current.x;let center=bandCenter(rowIndex);let unit=fp.fineWidth*f32(fp.fineFactor);var edge=INVALID;if(rowIndex<arrayLength(&pointStatus)){edge=atomicLoad(&pointStatus[rowIndex]);}var traversed=0u;loop{if(edge==INVALID||edge>=arrayLength(&transientPowerIncidences)){break;}let item=transientPowerIncidences[edge];let parent=item.face;if(parent<arrayLength(&rowVelocities)&&parent<arrayLength(&rows)){let source=rowVelocities[parent];if(source.y!=0.&&finite(source.x)){let candidate=abs(source.x)+length(center-bandCenter(parent))*unit;if(candidate<best){best=candidate;signedBest=select(-candidate,candidate,source.x>=0.);}}}edge=bitcast<u32>(item.sign);traversed+=1u;if(traversed>p.rowCapacity*MAX_GUARDS){break;}}let count=min(atomicLoad(&incidence[rowIndex]),p.axisStride);for(var local=0u;local<count;local+=1u){let faceIndex=atomicLoad(&incidence[p.rowCapacity+rowIndex*p.axisStride+local]);if(faceIndex>=arrayLength(&faces)){continue;}let face=faces[faceIndex];let neighbor=select(face.negativeRow,face.positiveRow,face.negativeRow==rowIndex);if(neighbor==INVALID||neighbor>=arrayLength(&rowVelocities)||neighbor>=arrayLength(&rows)){continue;}let source=rowVelocities[neighbor];if(source.y==0.||!finite(source.x)){continue;}let candidate=abs(source.x)+length(center-bandCenter(neighbor))*unit;if(candidate<best){best=candidate;signedBest=select(-candidate,candidate,source.x>=0.);}}return vec2f(signedBest,select(0.,1.,best<3.402823e38));}
@compute @workgroup_size(64)fn extendBandRowPhi(@builtin(global_invocation_id)g:vec3u){let rowIndex=g.x;if(rowIndex>=p.rowCapacity||rowIndex>=arrayLength(&rows)||rowIndex>=arrayLength(&rowVelocities)||rowIndex>=arrayLength(&provisionalVelocities)){return;}let current=rowVelocities[rowIndex];if(current.w==0.){provisionalVelocities[rowIndex]=vec4f(0.);return;}if(current.z>0.){provisionalVelocities[rowIndex]=current;return;}let row=rows[rowIndex];let sign=rowMarchSign(row);if(sign==0.){let candidate=nearestSignedNeighbor(rowIndex,current);provisionalVelocities[rowIndex]=select(vec4f(0.,0.,0.,1.),vec4f(candidate.x,1.,0.,1.),candidate.y!=0.);return;}var best=select(3.402823e38,abs(current.x),sameMarchSide(current,sign));best=min(best,lowerSimplexCandidate(rowIndex,sign));let closureOnly=(row.flags&(ROW_SUPPORT3_NODE|ROW_SUPPORT4_NODE|ROW_SUPPORT5_NODE|ROW_SUPPORT6_NODE|ROW_SUPPORT3_ENDPOINT))!=0u;if(!closureOnly&&rowIndex<arrayLength(&metrics)&&(metrics[rowIndex].transformFlags&VALID)!=0u){if(metrics[rowIndex].reserved==0u){best=min(best,uniformEikonal(rowIndex,sign));}else{best=min(best,localTetraEikonal(rowIndex,sign));}}provisionalVelocities[rowIndex]=select(vec4f(0.,0.,0.,1.),vec4f(sign*best,1.,0.,1.),best<3.402823e38);}
fn recordBandPhiRowFailure(rowIndex:u32,row:Row,state:vec4f){var claimed=false;loop{let result=atomicCompareExchangeWeak(&transitionControl.failureBand,INVALID,rowIndex);if(result.exchanged){claimed=true;break;}if(result.old_value!=INVALID){break;}}if(!claimed){return;}transitionControl.failureStage=OWNER_FAILURE_BAND_PHI;transitionControl.failureRowCell=row.cell;transitionControl.failureRowSize=row.size;transitionControl.failureDescriptor=row.flags;transitionControl.failureTopology=row.globalRow;transitionControl.failureTransformFlags=bitcast<u32>(state.x);transitionControl.failureSelector=bitcast<u32>(state.y);transitionControl.failureRawX=bitcast<u32>(state.z);transitionControl.failureRawY=bitcast<u32>(state.w);transitionControl.failureRawZ=rowIndex;transitionControl.failureRequestedSize=0u;transitionControl.failureResolvedCell=0u;transitionControl.failureBoundaryFlips=0u;transitionControl.failureOwnerCell=0u;transitionControl.failureOwnerSizeValid=0u;}
@compute @workgroup_size(64)fn commitBandRowPhi(@builtin(global_invocation_id)g:vec3u){let rowIndex=g.x;if(rowIndex>=transitionControl.endpointEnd||rowIndex>=arrayLength(&rows)||rowIndex>=arrayLength(&rowVelocities)||atomicLoad(&control.flags)!=0u){return;}let state=rowVelocities[rowIndex];if(state.w==0.||state.y==0.||!finite(state.x)){recordBandPhiRowFailure(rowIndex,rows[rowIndex],state);atomicAdd(&control.coarsePhiFailures,1u);fail(OUTSIDE_FINE_BAND,rows[rowIndex].cell);return;}rows[rowIndex].representativePhi=state.x;rows[rowIndex].minimumPhi=state.x;rows[rowIndex].maximumPhi=state.x;rows[rowIndex].flags|=ROW_PHI;if(state.z==0.){atomicAdd(&control.bandPhiExtensions,1u);}}
@compute @workgroup_size(64)fn sampleBandFaceCoarsePhi(@builtin(global_invocation_id)g:vec3u){let index=g.x;if(index>=p.faceCapacity||index>=arrayLength(&faces)||atomicLoad(&control.flags)!=0u){return;}var face=faces[index];if((face.flags&(LIVE|PHI_VALID))!=LIVE){return;}var sampled=coarsePhiAtPoint(face.negativeRow,face.centroid.xyz);if(sampled.y==0.&&face.positiveRow<transitionControl.support2End){sampled=coarsePhiAtPoint(face.positiveRow,face.centroid.xyz);}if(sampled.y==0.){var phiRecord=diagnoseCoarsePhiAtPoint(face.negativeRow,face.centroid.xyz);if(phiRecord.cause==INVALID&&face.positiveRow<transitionControl.support2End){phiRecord=diagnoseCoarsePhiAtPoint(face.positiveRow,face.centroid.xyz);}if(phiRecord.cause==INVALID){phiRecord=phiDiagnostic(vec3i(0),0u,face.negativeRow,PHI_PATH_ANCHOR,INVALID,PHI_CAUSE_INVALID_SELECTOR,0xb0000u);}face.velocity=bitcast<vec4f>(vec4u(bitcast<u32>(phiRecord.origin.x),bitcast<u32>(phiRecord.origin.y),bitcast<u32>(phiRecord.origin.z),phiRecord.size));face.phi=bitcast<f32>(phiRecord.anchor);face.area=bitcast<f32>(phiRecord.selector);face.pad=(phiRecord.path&255u)|((phiRecord.cause&255u)<<8u)|((phiRecord.detail&65535u)<<16u);face.flags|=PHI_DIAGNOSTIC;faces[index]=face;atomicAdd(&control.coarsePhiFailures,1u);fail(OUTSIDE_FINE_BAND,face.globalFace);return;}atomicAdd(&control.coarsePhiFallbacks,1u);face.phi=sampled.x;face.flags|=PHI_VALID;faces[index]=face;}
@compute @workgroup_size(64)fn reduceBandPhiFailure(@builtin(global_invocation_id)g:vec3u){let index=g.x;if(index>=arrayLength(&faces)){return;}let face=faces[index];if((face.flags&PHI_DIAGNOSTIC)==0u){return;}let cause=(face.pad>>8u)&255u;if(cause<4u){atomicAdd(&transitionControl.phiFailureCounts,1u<<(cause*8u));}atomicMin(&transitionControl.failureBand,PHI_FAILURE_TAG|index);}
@compute @workgroup_size(64)fn publishBandPhiFailure(@builtin(global_invocation_id)g:vec3u){let index=g.x;if(index>=arrayLength(&faces)||(atomicLoad(&transitionControl.failureBand)!=(PHI_FAILURE_TAG|index))){return;}let face=faces[index];if((face.flags&PHI_DIAGNOSTIC)==0u){return;}let packedOrigin=bitcast<vec4u>(face.velocity);let cause=(face.pad>>8u)&255u;let detail=(face.pad>>16u)&65535u;transitionControl.failureStage=index;transitionControl.failureRowCell=face.globalFace;transitionControl.failureRowSize=face.negativeRow;transitionControl.failureDescriptor=face.positiveRow;transitionControl.failureTopology=bitcast<u32>(face.phi);transitionControl.failureTransformFlags=bitcast<u32>(face.centroid.x);transitionControl.failureSelector=bitcast<u32>(face.centroid.y);transitionControl.failureRawX=bitcast<u32>(face.centroid.z);transitionControl.failureRawY=face.pad&255u;transitionControl.failureRawZ=packedOrigin.x;transitionControl.failureRequestedSize=packedOrigin.y;transitionControl.failureResolvedCell=packedOrigin.z;transitionControl.failureBoundaryFlips=packedOrigin.w;transitionControl.failureOwnerCell=bitcast<u32>(face.area);transitionControl.failureOwnerSizeValid=cause|(detail<<8u);}
@compute @workgroup_size(64)fn summarizeBandRowPhi(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=transitionControl.support2End||row>=arrayLength(&rows)||atomicLoad(&control.flags)!=0u){return;}let count=min(atomicLoad(&incidence[row]),p.axisStride);var minimum=3.402823e38;var maximum=-3.402823e38;var representative=0.;var representativeAbs=3.402823e38;var representativeFace=INVALID;var found=0u;for(var local=0u;local<count;local+=1u){let index=atomicLoad(&incidence[p.rowCapacity+row*p.axisStride+local]);if(index>=p.faceCapacity||index>=arrayLength(&faces)){fail(BAD_PHI,row);return;}let face=faces[index];if((face.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||(face.negativeRow!=row&&face.positiveRow!=row)||!finite(face.phi)){fail(BAD_PHI,row);return;}minimum=min(minimum,face.phi);maximum=max(maximum,face.phi);let magnitude=abs(face.phi);if(magnitude<representativeAbs||(magnitude==representativeAbs&&face.globalFace<representativeFace)){representative=face.phi;representativeAbs=magnitude;representativeFace=face.globalFace;}found+=1u;}if(found==0u){fail(BAD_PHI,row);return;}rows[row].representativePhi=representative;rows[row].minimumPhi=minimum;rows[row].maximumPhi=maximum;rows[row].flags|=ROW_PHI;}
fn velocityValid(v:vec4f)->bool{return v.w>0.&&finite(v.x)&&finite(v.y)&&finite(v.z);}fn cellVector(origin:vec3u,size:u32)->vec4f{let row=findSite(cell(origin),size);if(row==INVALID||row>=p.powerRowCapacity||row>=arrayLength(&powerRowVelocities)){return vec4f(0);}let velocity=powerRowVelocities[row];return select(vec4f(0),velocity,velocityValid(velocity));}fn powerTransform(value:vec3f,code:u32)->vec3f{let permutation=(code/8u)%6u;var result=value;if(permutation==1u){result=value.xzy;}else if(permutation==2u){result=value.yxz;}else if(permutation==3u){result=value.yzx;}else if(permutation==4u){result=value.zxy;}else if(permutation==5u){result=value.zyx;}let bits=code&7u;return result*vec3f(select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));}fn tetraWeights(point:vec3f,a:vec3f,b:vec3f,c:vec3f)->vec4f{let determinant=dot(a,cross(b,c));if(!finite(determinant)||abs(determinant)<=1e-10){return vec4f(-2.);}let wa=dot(point,cross(b,c))/determinant;let wb=dot(a,cross(point,c))/determinant;let wc=dot(a,cross(b,point))/determinant;return vec4f(1.-wa-wb-wc,wa,wb,wc);}fn contained(weights:vec4f)->bool{return all(weights>=vec4f(-2e-6))&&all(weights<=vec4f(1.000002));}
fn reflectComponents(value:vec4f,mask:i32)->vec4f{if(!velocityValid(value)){return vec4f(0);}var result=value;for(var axis=0u;axis<3u;axis+=1u){if((mask&(1i<<axis))!=0){result[axis]=-result[axis];}}return result;}
fn compactSignedVector(origin:vec3i,size:u32)->vec4f{let high=origin+vec3i(i32(size));let boundary=any(origin<vec3i(0))||any(high>vec3i(p.dims));let reflected=velocityExtendedOrigin(origin,size);if(reflected.w<0){return vec4f(0);}let q=vec3u(reflected.xyz);var value=cellVector(q,size);if(!velocityValid(value)&&boundary){let globalRow=containing(q);if(globalRow!=INVALID&&globalRow<p.powerRowCapacity&&globalRow<arrayLength(&powerRowVelocities)){value=powerRowVelocities[globalRow];}}return reflectComponents(value,reflected.w);}
fn selectorVector(anchor:u32,selector:u32,transform:u32)->vec4f{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return vec4f(0);}let row=rows[anchor];let v=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(v.xyz,transform);let sizeFloat=f32(row.size)*v.w;let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return vec4f(0);}return compactSignedVector(vec3i(round(point-.5*f32(size))),size);}
fn centroidVector(anchor:u32,pointGrid:vec3f)->vec4f{if(anchor>=arrayLength(&rows)||anchor>=arrayLength(&metrics)){return vec4f(0);}let row=rows[anchor];let origin=coord(row.cell);let anchorVelocity=cellVector(origin,row.size);if(!velocityValid(anchorVelocity)){return vec4f(0);}let metric=metrics[anchor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){return vec4f(0);}let header=tetraHeaders[metric.topology];if((header.flags&1u)!=0u){let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);if(any(t<vec3f(-2e-6))||any(t>vec3f(1.000002))){return vec4f(0);}var result=vec3f(0);for(var corner=0u;corner<8u;corner+=1u){let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);if(weight==0.){continue;}let cornerOrigin=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u))*i32(row.size);let v=compactSignedVector(cornerOrigin,row.size);if(!velocityValid(v)){return vec4f(0);}result+=weight*v.xyz;}return select(vec4f(0),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return vec4f(0);}let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return vec4f(0);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}let va=selectorVector(anchor,selectors.x,metric.transformFlags&63u);let vb=selectorVector(anchor,selectors.y,metric.transformFlags&63u);let vc=selectorVector(anchor,selectors.z,metric.transformFlags&63u);if(!velocityValid(va)||!velocityValid(vb)||!velocityValid(vc)){return vec4f(0);}let result=weights.x*anchorVelocity.xyz+weights.y*va.xyz+weights.z*vb.xyz+weights.w*vc.xyz;return select(vec4f(0),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}return vec4f(0);}
@compute @workgroup_size(64)fn seedFaceCentroids(@builtin(global_invocation_id)g:vec3u){let face=g.x;if(face>=p.faceCapacity||face>=arrayLength(&faces)){return;}var f=faces[face];if((f.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||f.negativeRow>=arrayLength(&rows)||!finite(f.phi)){return;}if(f.phi>0.){f.pad=1u;faces[face]=f;return;}var reason=0u;var velocity=centroidVector(f.negativeRow,f.centroid.xyz);if(!velocityValid(velocity)){reason|=2u;}if(!velocityValid(velocity)&&f.positiveRow!=INVALID){velocity=centroidVector(f.positiveRow,f.centroid.xyz);if(!velocityValid(velocity)){reason|=4u;}}if(!velocityValid(velocity)){f.pad=reason;faces[face]=f;return;}f.velocity=velocity;f.pad=0u;f.flags|=SEED;faces[face]=f;}
@compute @workgroup_size(64)fn seedOpenWorldNormal(@builtin(global_invocation_id)g:vec3u){let face=g.x;if(face>=arrayLength(&faces)){return;}var f=faces[face];if((f.flags&(LIVE|SEED))!=(LIVE|SEED)||f.positiveRow!=INVALID||f.negativeRow>=arrayLength(&rows)){return;}let axis=f.axisSpan&3u;let row=rows[f.negativeRow];if(axis>=3u||row.globalRow==INVALID||row.globalRow+1u>=arrayLength(&powerIncidenceRows)){fail(INCOMPLETE,row.cell);return;}let begin=powerIncidenceRows[row.globalRow].incidenceOffset;let end=powerIncidenceRows[row.globalRow+1u].incidenceOffset;if(begin>end||end>arrayLength(&powerIncidences)){fail(INCOMPLETE,row.cell);return;}var found=false;var scalar=0.;for(var cursor=begin;cursor<end;cursor+=1u){let item=powerIncidences[cursor];if(item.face>=arrayLength(&powerFaces)||item.face>=arrayLength(&powerFaceNormals)){fail(INCOMPLETE,row.cell);return;}let pf=powerFaces[item.face];let world=(pf.flags>>8u)&63u;if((pf.flags&3u)!=3u||(world&positiveBoundaryBit(axis))==0u){continue;}let n=powerFaceNormals[item.face].xyz;if(!finite(pf.normalVelocity)||!finite(n[axis])||abs(abs(n[axis])-1.)>4e-4){fail(INCOMPLETE,row.cell);return;}let candidate=pf.normalVelocity/n[axis];if(found&&abs(candidate-scalar)>1e-5*max(1.,max(abs(candidate),abs(scalar)))){fail(INCOMPLETE,row.cell);return;}scalar=candidate;found=true;}if(!found){fail(INCOMPLETE,row.cell);return;}f.velocity[axis]=scalar;faces[face]=f;}
@compute @workgroup_size(64)fn initializeFaceMarch(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=p.faceCapacity||i>=arrayLength(&faces)||i>=arrayLength(&states)){return;}let f=faces[i];if((f.flags&LIVE)==0u){return;}states[i].parent=INVALID;states[i].depth=0u;states[i].pad=INVALID;atomicStore(&states[i].status,UNKNOWN);if((f.flags&PHI_VALID)==0u||!finite(f.phi)||!finite(f.area)||f.area<=0.){fail(BAD_FACE,f.globalFace);return;}if((f.flags&SEED)==0u){return;}if(!velocityValid(f.velocity)){fail(BAD_FACE,f.globalFace);return;}states[i].velocity=f.velocity;states[i].parent=i;atomicStore(&states[i].status,ACCEPTED);atomicAdd(&control.seedCount,1u);atomicAdd(&control.acceptedCount,1u);}
fn faceArrival(phi:f32)->f32{return max(phi,0.);}
fn faceHeapBefore(a:u32,b:u32)->bool{if(a>=p.faceCapacity||a>=arrayLength(&faces)){return false;}if(b>=p.faceCapacity||b>=arrayLength(&faces)){return true;}let af=faces[a];let bf=faces[b];let ap=faceArrival(af.phi);let bp=faceArrival(bf.phi);if(ap<bp){return true;}if(ap>bp){return false;}if(af.globalFace<bf.globalFace){return true;}if(af.globalFace>bf.globalFace){return false;}return a<b;}
fn consider(row:u32,targetFace:u32,best:ptr<function,u32>,bestPhi:ptr<function,f32>,bestGlobal:ptr<function,u32>){if(row==INVALID||row>=p.rowCapacity||row>=arrayLength(&incidence)){return;}let targetRecord=faces[targetFace];if((targetRecord.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||!finite(targetRecord.phi)){fail(BAD_PHI,targetRecord.globalFace);return;}let targetArrival=faceArrival(targetRecord.phi);let count=min(atomicLoad(&incidence[row]),p.axisStride);for(var local=0u;local<count;local+=1u){let candidate=atomicLoad(&incidence[p.rowCapacity+row*p.axisStride+local]);if(candidate==targetFace||candidate>=p.faceCapacity||candidate>=arrayLength(&faces)||candidate>=arrayLength(&states)||atomicLoad(&states[candidate].status)!=ACCEPTED){continue;}let accepted=faces[candidate];if((accepted.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||!finite(accepted.phi)){fail(BAD_PHI,accepted.globalFace);return;}let ap=abs(accepted.phi);let globalFace=accepted.globalFace;if(faceArrival(accepted.phi)<=targetArrival+1e-6&&(ap<(*bestPhi)||(ap==(*bestPhi)&&(globalFace<(*bestGlobal)||(globalFace==(*bestGlobal)&&candidate<(*best)))))){*best=candidate;*bestPhi=ap;*bestGlobal=globalFace;}}}
fn considerTopology(row:u32,targetFace:u32,best:ptr<function,u32>,bestPhi:ptr<function,f32>,bestGlobal:ptr<function,u32>){if(row==INVALID||row>=transitionControl.endpointEnd||row>=arrayLength(&rows)||(rows[row].flags&ROW_SUPPORT3_ENDPOINT)!=0u){return;}consider(row,targetFace,best,bestPhi,bestGlobal);if(row>=transitionControl.support2End||row>=arrayLength(&metrics)){return;}let count=min(metrics[row].reserved,MAX_TETRA);for(var local=0u;local<count;local+=1u){let at=row*MAX_TETRA+local;if(at>=arrayLength(&transitionAdjacency)){return;}let adjacency=transitionAdjacency[at];if(adjacency.band!=row){continue;}consider(adjacency.a,targetFace,best,bestPhi,bestGlobal);consider(adjacency.b,targetFace,best,bestPhi,bestGlobal);consider(adjacency.c,targetFace,best,bestPhi,bestGlobal);}}
fn acceptedFacePredecessor(targetFace:u32)->u32{let targetRecord=faces[targetFace];var best=INVALID;var bestPhi=3.402823e38;var bestGlobal=INVALID;considerTopology(targetRecord.negativeRow,targetFace,&best,&bestPhi,&bestGlobal);considerTopology(targetRecord.positiveRow,targetFace,&best,&bestPhi,&bestGlobal);return best;}
fn recordedAcceptedPredecessor(targetFace:u32)->u32{if(targetFace>=p.faceCapacity||targetFace>=arrayLength(&faces)||targetFace>=arrayLength(&states)){return INVALID;}let sourceFace=states[targetFace].pad;if(sourceFace==targetFace||sourceFace>=p.faceCapacity||sourceFace>=arrayLength(&faces)||sourceFace>=arrayLength(&states)||atomicLoad(&states[sourceFace].status)!=ACCEPTED){return INVALID;}let targetRecord=faces[targetFace];let sourceRecord=faces[sourceFace];if((targetRecord.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||(sourceRecord.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||!finite(targetRecord.phi)||!finite(sourceRecord.phi)||!velocityValid(states[sourceFace].velocity)||faceArrival(sourceRecord.phi)>faceArrival(targetRecord.phi)+1e-6){return INVALID;}return sourceFace;}
fn closestPredecessorRow(row:u32,targetFace:u32,best:ptr<function,u32>){if(row==INVALID||row>=p.rowCapacity||row>=arrayLength(&incidence)){return;}let count=min(atomicLoad(&incidence[row]),p.axisStride);for(var local=0u;local<count;local+=1u){let candidate=atomicLoad(&incidence[p.rowCapacity+row*p.axisStride+local]);if(candidate==targetFace||candidate>=p.faceCapacity||candidate>=arrayLength(&faces)||candidate>=arrayLength(&states)){continue;}let record=faces[candidate];if((record.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||!finite(record.phi)){continue;}if(faceHeapBefore(candidate,targetFace)&&((*best)==INVALID||faceHeapBefore(candidate,*best))){*best=candidate;}}}
fn closestPredecessorTopology(row:u32,targetFace:u32,best:ptr<function,u32>){if(row==INVALID||row>=transitionControl.endpointEnd||row>=arrayLength(&rows)||(rows[row].flags&ROW_SUPPORT3_ENDPOINT)!=0u){return;}closestPredecessorRow(row,targetFace,best);if(row>=transitionControl.support2End||row>=arrayLength(&metrics)){return;}let count=min(metrics[row].reserved,MAX_TETRA);for(var local=0u;local<count;local+=1u){let at=row*MAX_TETRA+local;if(at>=arrayLength(&transitionAdjacency)){return;}let adjacency=transitionAdjacency[at];if(adjacency.band!=row){continue;}closestPredecessorRow(adjacency.a,targetFace,best);closestPredecessorRow(adjacency.b,targetFace,best);closestPredecessorRow(adjacency.c,targetFace,best);}}
@compute @workgroup_size(64)fn linkFaceClosestPoints(@builtin(global_invocation_id)g:vec3u){let faceIndex=g.x;if(faceIndex>=p.faceCapacity||faceIndex>=arrayLength(&faces)||faceIndex>=arrayLength(&states)||faceIndex>=arrayLength(&cptParentInput)||(faces[faceIndex].flags&LIVE)==0u){return;}if(faceIndex==0u){control.marchChunkBound=p.maximumRounds;}if(atomicLoad(&states[faceIndex].status)==ACCEPTED){states[faceIndex].parent=faceIndex;cptParentInput[faceIndex]=faceIndex;return;}let face=faces[faceIndex];var best=INVALID;closestPredecessorTopology(face.negativeRow,faceIndex,&best);closestPredecessorTopology(face.positiveRow,faceIndex,&best);if(best==INVALID){cptParentInput[faceIndex]=INVALID;atomicStore(&states[faceIndex].status,REJECTED);return;}states[faceIndex].parent=best;states[faceIndex].pad=best;states[faceIndex].depth=1u;cptParentInput[faceIndex]=best;atomicStore(&states[faceIndex].status,TRIAL);atomicAdd(&control.marchTrials,1u);}
// Every jump reads an immutable parent snapshot and writes a distinct buffer.
// The host swaps those buffers only between dispatches, avoiding the racy
// in-place parent reads that WebGPU's storage memory model cannot order.
@compute @workgroup_size(64)fn jumpFaceClosestPoints(@builtin(global_invocation_id)g:vec3u){let faceIndex=g.x;if(faceIndex==0u){atomicAdd(&control.marchChunks,1u);}if(faceIndex>=p.faceCapacity||faceIndex>=arrayLength(&faces)||faceIndex>=arrayLength(&states)||faceIndex>=arrayLength(&cptParentInput)||faceIndex>=arrayLength(&cptParentOutput)||(faces[faceIndex].flags&LIVE)==0u){return;}let parent=cptParentInput[faceIndex];if(parent>=p.faceCapacity||parent>=arrayLength(&cptParentInput)){cptParentOutput[faceIndex]=INVALID;return;}let ancestor=cptParentInput[parent];cptParentOutput[faceIndex]=select(parent,ancestor,ancestor<p.faceCapacity&&ancestor<arrayLength(&cptParentInput));}
@compute @workgroup_size(64)fn resolveFaceClosestPoints(@builtin(global_invocation_id)g:vec3u){let faceIndex=g.x;if(faceIndex>=p.faceCapacity||faceIndex>=arrayLength(&faces)||faceIndex>=arrayLength(&states)||faceIndex>=arrayLength(&cptParentInput)||(faces[faceIndex].flags&LIVE)==0u||atomicLoad(&states[faceIndex].status)!=TRIAL){return;}let root=cptParentInput[faceIndex];if(root>=p.faceCapacity||root>=arrayLength(&faces)||root>=arrayLength(&states)||atomicLoad(&states[root].status)!=ACCEPTED||(faces[root].flags&SEED)==0u||!velocityValid(states[root].velocity)){atomicStore(&states[faceIndex].status,REJECTED);return;}states[faceIndex].velocity=states[root].velocity;states[faceIndex].parent=faces[root].globalFace;atomicStore(&states[faceIndex].status,ACCEPTED);atomicAdd(&control.marchPops,1u);atomicAdd(&control.acceptedCount,1u);atomicMax(&control.maximumDepth,states[faceIndex].depth);}
@compute @workgroup_size(64)fn prepareFaceBfsFallback(@builtin(global_invocation_id)g:vec3u){let faceIndex=g.x;if(faceIndex==0u){atomicStore(&control.marchChunks,0u);}if(faceIndex>=p.faceCapacity||faceIndex>=arrayLength(&faces)||faceIndex>=arrayLength(&states)||(faces[faceIndex].flags&LIVE)==0u){return;}if(atomicLoad(&states[faceIndex].status)==REJECTED){states[faceIndex].parent=INVALID;states[faceIndex].depth=INVALID;states[faceIndex].pad=INVALID;atomicStore(&states[faceIndex].status,UNKNOWN);}}
fn considerBfsRow(row:u32,targetFace:u32,layer:u32,best:ptr<function,u32>,bestPhi:ptr<function,f32>,bestGlobal:ptr<function,u32>){if(row==INVALID||row>=p.rowCapacity||row>=arrayLength(&incidence)){return;}let targetRecord=faces[targetFace];let targetArrival=faceArrival(targetRecord.phi);let count=min(atomicLoad(&incidence[row]),p.axisStride);for(var local=0u;local<count;local+=1u){let candidate=atomicLoad(&incidence[p.rowCapacity+row*p.axisStride+local]);if(candidate==targetFace||candidate>=p.faceCapacity||candidate>=arrayLength(&faces)||candidate>=arrayLength(&states)||atomicLoad(&states[candidate].status)!=ACCEPTED||states[candidate].depth>=layer){continue;}let accepted=faces[candidate];if((accepted.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||!finite(accepted.phi)||(faceBfsCausal!=0u&&faceArrival(accepted.phi)>targetArrival+1e-6)){continue;}let ap=abs(accepted.phi);let globalFace=accepted.globalFace;if(ap<(*bestPhi)||(ap==(*bestPhi)&&(globalFace<(*bestGlobal)||(globalFace==(*bestGlobal)&&candidate<(*best))))){*best=candidate;*bestPhi=ap;*bestGlobal=globalFace;}}}
fn considerBfsTopology(row:u32,targetFace:u32,layer:u32,best:ptr<function,u32>,bestPhi:ptr<function,f32>,bestGlobal:ptr<function,u32>){if(row==INVALID||row>=transitionControl.endpointEnd||row>=arrayLength(&rows)||(rows[row].flags&ROW_SUPPORT3_ENDPOINT)!=0u){return;}considerBfsRow(row,targetFace,layer,best,bestPhi,bestGlobal);if(row>=transitionControl.support2End||row>=arrayLength(&metrics)){return;}let count=min(metrics[row].reserved,MAX_TETRA);for(var local=0u;local<count;local+=1u){let at=row*MAX_TETRA+local;if(at>=arrayLength(&transitionAdjacency)){return;}let adjacency=transitionAdjacency[at];if(adjacency.band!=row){continue;}considerBfsRow(adjacency.a,targetFace,layer,best,bestPhi,bestGlobal);considerBfsRow(adjacency.b,targetFace,layer,best,bestPhi,bestGlobal);considerBfsRow(adjacency.c,targetFace,layer,best,bestPhi,bestGlobal);}}
@compute @workgroup_size(64)fn propagateFaceBfsLayer(@builtin(global_invocation_id)g:vec3u){let faceIndex=g.x;if(faceIndex>=p.faceCapacity||faceIndex>=arrayLength(&faces)||faceIndex>=arrayLength(&states)||(faces[faceIndex].flags&LIVE)==0u||atomicLoad(&states[faceIndex].status)!=UNKNOWN){return;}let layer=faceBfsLayer;let face=faces[faceIndex];var best=INVALID;var bestPhi=3.402823e38;var bestGlobal=INVALID;considerBfsTopology(face.negativeRow,faceIndex,layer,&best,&bestPhi,&bestGlobal);considerBfsTopology(face.positiveRow,faceIndex,layer,&best,&bestPhi,&bestGlobal);if(best==INVALID){return;}states[faceIndex].velocity=states[best].velocity;states[faceIndex].parent=faces[best].globalFace;states[faceIndex].depth=layer;atomicStore(&states[faceIndex].status,ACCEPTED);atomicAdd(&control.acceptedCount,1u);if(faceBfsCausal==0u){atomicAdd(&control.connectivityFallbacks,1u);}atomicMax(&control.maximumDepth,layer);atomicMax(&control.marchChunks,layer);}
fn auditDisconnectedRow(row:u32,targetFace:u32,best:ptr<function,u32>,bestPhi:ptr<function,f32>,neighborCount:ptr<function,u32>,nonincreasingCount:ptr<function,u32>,acceptedCount:ptr<function,u32>){if(row==INVALID||row>=p.rowCapacity||row>=arrayLength(&incidence)){return;}let targetArrival=faceArrival(faces[targetFace].phi);let count=min(atomicLoad(&incidence[row]),p.axisStride);for(var local=0u;local<count;local+=1u){let candidate=atomicLoad(&incidence[p.rowCapacity+row*p.axisStride+local]);if(candidate==targetFace||candidate>=p.faceCapacity||candidate>=arrayLength(&faces)||candidate>=arrayLength(&states)){continue;}let record=faces[candidate];if((record.flags&(LIVE|PHI_VALID))!=(LIVE|PHI_VALID)||!finite(record.phi)){continue;}*neighborCount+=1u;let candidatePhi=abs(record.phi);if(faceArrival(record.phi)<=targetArrival+1e-6){*nonincreasingCount+=1u;}if(atomicLoad(&states[candidate].status)==ACCEPTED){*acceptedCount+=1u;}if(candidatePhi<(*bestPhi)||(candidatePhi==(*bestPhi)&&record.globalFace<faces[*best].globalFace)){*best=candidate;*bestPhi=candidatePhi;}}}
fn auditDisconnectedTopology(row:u32,targetFace:u32,best:ptr<function,u32>,bestPhi:ptr<function,f32>,neighborCount:ptr<function,u32>,nonincreasingCount:ptr<function,u32>,acceptedCount:ptr<function,u32>){if(row==INVALID||row>=transitionControl.endpointEnd||row>=arrayLength(&rows)||(rows[row].flags&ROW_SUPPORT3_ENDPOINT)!=0u){return;}auditDisconnectedRow(row,targetFace,best,bestPhi,neighborCount,nonincreasingCount,acceptedCount);if(row>=transitionControl.support2End||row>=arrayLength(&metrics)){return;}let count=min(metrics[row].reserved,MAX_TETRA);for(var local=0u;local<count;local+=1u){let at=row*MAX_TETRA+local;if(at>=arrayLength(&transitionAdjacency)){return;}let adjacency=transitionAdjacency[at];if(adjacency.band!=row){continue;}auditDisconnectedRow(adjacency.a,targetFace,best,bestPhi,neighborCount,nonincreasingCount,acceptedCount);auditDisconnectedRow(adjacency.b,targetFace,best,bestPhi,neighborCount,nonincreasingCount,acceptedCount);auditDisconnectedRow(adjacency.c,targetFace,best,bestPhi,neighborCount,nonincreasingCount,acceptedCount);}}
fn recordDisconnectedFace(i:u32,face:Face){var claimed=false;loop{let result=atomicCompareExchangeWeak(&transitionControl.failureBand,INVALID,i);if(result.exchanged){claimed=true;break;}if(result.old_value!=INVALID){break;}}if(!claimed){return;}var best=INVALID;var bestPhi=3.402823e38;var neighborCount=0u;var nonincreasingCount=0u;var acceptedCount=0u;auditDisconnectedTopology(face.negativeRow,i,&best,&bestPhi,&neighborCount,&nonincreasingCount,&acceptedCount);auditDisconnectedTopology(face.positiveRow,i,&best,&bestPhi,&neighborCount,&nonincreasingCount,&acceptedCount);var bestGlobal=INVALID;var bestPhiBits=0u;var bestStatus=INVALID;var bestFlags=INVALID;var bestParent=INVALID;if(best!=INVALID){bestGlobal=faces[best].globalFace;bestPhiBits=bitcast<u32>(faces[best].phi);bestStatus=atomicLoad(&states[best].status);bestFlags=faces[best].flags;bestParent=states[best].parent;}transitionControl.failureStage=OWNER_FAILURE_DISCONNECTED_FACE;transitionControl.failureRowCell=face.globalFace;transitionControl.failureRowSize=face.negativeRow;transitionControl.failureDescriptor=face.positiveRow;transitionControl.failureTopology=face.flags;transitionControl.failureTransformFlags=bitcast<u32>(face.phi);transitionControl.failureSelector=best;transitionControl.failureRawX=bestGlobal;transitionControl.failureRawY=bestPhiBits;transitionControl.failureRawZ=bestStatus;transitionControl.failureRequestedSize=neighborCount;transitionControl.failureResolvedCell=nonincreasingCount;transitionControl.failureBoundaryFlips=acceptedCount;transitionControl.failureOwnerCell=bestFlags;transitionControl.failureOwnerSizeValid=bestParent;}
@compute @workgroup_size(64)fn validateFaceMarch(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=p.faceCapacity||i>=arrayLength(&faces)||i>=arrayLength(&states)||(faces[i].flags&LIVE)==0u){return;}let status=atomicLoad(&states[i].status);if(status==ACCEPTED){return;}atomicAdd(&control.unresolvedCount,1u);if(status==TRIAL){atomicAdd(&control.marchCapExhausted,1u);}else if(recordedAcceptedPredecessor(i)!=INVALID||acceptedFacePredecessor(i)!=INVALID){atomicAdd(&control.marchUnresolvedWithPredecessor,1u);}else{atomicAdd(&control.marchDisconnected,1u);recordDisconnectedFace(i,faces[i]);}fail(UNRESOLVED,faces[i].globalFace);}
@compute @workgroup_size(64)fn reconstructBandRowVelocity(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=transitionControl.support3NodeEnd||row>=arrayLength(&rows)||row>=arrayLength(&provisionalVelocities)){return;}let r=rows[row];let origin=coord(r.cell);let targetArea=f32(r.size*r.size);let count=min(atomicLoad(&incidence[row]),p.axisStride);let closure=(r.flags&ROW_SUPPORT3_NODE)!=0u;var negativeSum=vec3f(0);var positiveSum=vec3f(0);var negativeArea=vec3f(0);var positiveArea=vec3f(0);for(var local=0u;local<count;local+=1u){let fi=atomicLoad(&incidence[p.rowCapacity+row*p.axisStride+local]);if(fi>=p.faceCapacity||fi>=arrayLength(&faces)||fi>=arrayLength(&states)||atomicLoad(&states[fi].status)!=ACCEPTED){fail(INCOMPLETE,row);return;}let f=faces[fi];let axis=f.axisSpan&3u;if(axis>=3u||!finite(f.area)||f.area<=0.){fail(INCOMPLETE,row);return;}let scalar=states[fi].velocity[axis];if(!finite(scalar)){fail(INCOMPLETE,row);return;}if(f.negativeRow==row){positiveSum[axis]+=f.area*scalar;positiveArea[axis]+=f.area;}else if(f.positiveRow==row){negativeSum[axis]+=f.area*scalar;negativeArea[axis]+=f.area;}else{fail(INCOMPLETE,row);return;}}var result=vec3f(0);var openScalars=0u;for(var axis=0u;axis<3u;axis+=1u){if(negativeArea[axis]<targetArea){if(closure&&negativeArea[axis]==0.&&positiveArea[axis]>=targetArea){negativeArea[axis]=positiveArea[axis];negativeSum[axis]=positiveSum[axis];}else if(origin[axis]==0u&&(p.closedBoundaryMask&negativeBoundaryBit(axis))!=0u){negativeArea[axis]=targetArea;negativeSum[axis]=0.;atomicAdd(&pointControl.wallContributions,1u);}else{if(closure){return;}fail(INCOMPLETE,row);return;}}if(positiveArea[axis]<targetArea){if(closure&&positiveArea[axis]==0.&&negativeArea[axis]>=targetArea){positiveArea[axis]=negativeArea[axis];positiveSum[axis]=negativeSum[axis];}else if(origin[axis]+r.size==p.dims[axis]&&(p.closedBoundaryMask&positiveBoundaryBit(axis))!=0u){positiveArea[axis]=targetArea;positiveSum[axis]=0.;atomicAdd(&pointControl.wallContributions,1u);}else{if(closure){return;}fail(INCOMPLETE,row);return;}}let tolerance=1e-4*max(1.,targetArea);if(abs(negativeArea[axis]-targetArea)>tolerance||abs(positiveArea[axis]-targetArea)>tolerance){if(closure){return;}fail(INCOMPLETE,row);return;}result[axis]=.5*(negativeSum[axis]/negativeArea[axis]+positiveSum[axis]/positiveArea[axis]);if(origin[axis]+r.size==p.dims[axis]&&(p.closedBoundaryMask&positiveBoundaryBit(axis))==0u){rows[row].padf=positiveSum[axis]/positiveArea[axis];openScalars+=1u;}}if(openScalars>1u||!finite(result.x)||!finite(result.y)||!finite(result.z)){fail(INCOMPLETE,row);return;}let value=vec4f(result,1.);provisionalVelocities[row]=value;if(row>=transitionControl.support1End&&row<arrayLength(&rowVelocities)){rowVelocities[row]=value;}}
@compute @workgroup_size(64)fn reconstructDeepBandRowVelocity(@builtin(global_invocation_id)g:vec3u){let row=transitionControl.support3NodeEnd+g.x;if(row>=transitionControl.support4NodeEnd||row>=arrayLength(&rows)||row>=arrayLength(&provisionalVelocities)){return;}let r=rows[row];let origin=coord(r.cell);let targetArea=f32(r.size*r.size);let count=min(atomicLoad(&incidence[row]),p.axisStride);var negativeSum=vec3f(0);var positiveSum=vec3f(0);var negativeArea=vec3f(0);var positiveArea=vec3f(0);for(var local=0u;local<count;local+=1u){let fi=atomicLoad(&incidence[p.rowCapacity+row*p.axisStride+local]);if(fi>=p.faceCapacity||fi>=arrayLength(&faces)||fi>=arrayLength(&states)||atomicLoad(&states[fi].status)!=ACCEPTED){return;}let f=faces[fi];let axis=f.axisSpan&3u;if(axis>=3u||!finite(f.area)||f.area<=0.){return;}let scalar=states[fi].velocity[axis];if(!finite(scalar)){return;}if(f.negativeRow==row){positiveSum[axis]+=f.area*scalar;positiveArea[axis]+=f.area;}else if(f.positiveRow==row){negativeSum[axis]+=f.area*scalar;negativeArea[axis]+=f.area;}else{return;}}var result=vec3f(0);for(var axis=0u;axis<3u;axis+=1u){if(negativeArea[axis]<targetArea){if(negativeArea[axis]==0.&&positiveArea[axis]>=targetArea){negativeArea[axis]=positiveArea[axis];negativeSum[axis]=positiveSum[axis];}else if(origin[axis]==0u&&(p.closedBoundaryMask&negativeBoundaryBit(axis))!=0u){negativeArea[axis]=targetArea;negativeSum[axis]=0.;}else{return;}}if(positiveArea[axis]<targetArea){if(positiveArea[axis]==0.&&negativeArea[axis]>=targetArea){positiveArea[axis]=negativeArea[axis];positiveSum[axis]=negativeSum[axis];}else if(origin[axis]+r.size==p.dims[axis]&&(p.closedBoundaryMask&positiveBoundaryBit(axis))!=0u){positiveArea[axis]=targetArea;positiveSum[axis]=0.;}else{return;}}let tolerance=1e-4*max(1.,targetArea);if(abs(negativeArea[axis]-targetArea)>tolerance||abs(positiveArea[axis]-targetArea)>tolerance){return;}result[axis]=.5*(negativeSum[axis]/negativeArea[axis]+positiveSum[axis]/positiveArea[axis]);}if(!finite(result.x)||!finite(result.y)||!finite(result.z)){return;}let value=vec4f(result,1.);provisionalVelocities[row]=value;if(row<arrayLength(&rowVelocities)){rowVelocities[row]=value;}}
@compute @workgroup_size(64)fn reconstructSupport5BandRowVelocity(@builtin(global_invocation_id)g:vec3u){let row=transitionControl.support4NodeEnd+g.x;if(row>=transitionControl.support5NodeEnd||row>=arrayLength(&rows)||row>=arrayLength(&provisionalVelocities)){return;}let r=rows[row];let origin=coord(r.cell);let targetArea=f32(r.size*r.size);let count=min(atomicLoad(&incidence[row]),p.axisStride);var negativeSum=vec3f(0);var positiveSum=vec3f(0);var negativeArea=vec3f(0);var positiveArea=vec3f(0);for(var local=0u;local<count;local+=1u){let fi=atomicLoad(&incidence[p.rowCapacity+row*p.axisStride+local]);if(fi>=p.faceCapacity||fi>=arrayLength(&faces)||fi>=arrayLength(&states)||atomicLoad(&states[fi].status)!=ACCEPTED){return;}let f=faces[fi];let axis=f.axisSpan&3u;if(axis>=3u||!finite(f.area)||f.area<=0.){return;}let scalar=states[fi].velocity[axis];if(!finite(scalar)){return;}if(f.negativeRow==row){positiveSum[axis]+=f.area*scalar;positiveArea[axis]+=f.area;}else if(f.positiveRow==row){negativeSum[axis]+=f.area*scalar;negativeArea[axis]+=f.area;}else{return;}}var result=vec3f(0);for(var axis=0u;axis<3u;axis+=1u){if(negativeArea[axis]<targetArea){if(negativeArea[axis]==0.&&positiveArea[axis]>=targetArea){negativeArea[axis]=positiveArea[axis];negativeSum[axis]=positiveSum[axis];}else if(origin[axis]==0u&&(p.closedBoundaryMask&negativeBoundaryBit(axis))!=0u){negativeArea[axis]=targetArea;negativeSum[axis]=0.;}else{return;}}if(positiveArea[axis]<targetArea){if(positiveArea[axis]==0.&&negativeArea[axis]>=targetArea){positiveArea[axis]=negativeArea[axis];positiveSum[axis]=negativeSum[axis];}else if(origin[axis]+r.size==p.dims[axis]&&(p.closedBoundaryMask&positiveBoundaryBit(axis))!=0u){positiveArea[axis]=targetArea;positiveSum[axis]=0.;}else{return;}}let tolerance=1e-4*max(1.,targetArea);if(abs(negativeArea[axis]-targetArea)>tolerance||abs(positiveArea[axis]-targetArea)>tolerance){return;}result[axis]=.5*(negativeSum[axis]/negativeArea[axis]+positiveSum[axis]/positiveArea[axis]);}if(!finite(result.x)||!finite(result.y)||!finite(result.z)){return;}let value=vec4f(result,1.);provisionalVelocities[row]=value;if(row<arrayLength(&rowVelocities)){rowVelocities[row]=value;}}
@compute @workgroup_size(64)fn reconstructSupport6BandRowVelocity(@builtin(global_invocation_id)g:vec3u){let row=transitionControl.support5NodeEnd+g.x;if(row>=transitionControl.support7NodeEnd||row>=arrayLength(&rows)||row>=arrayLength(&provisionalVelocities)){return;}let r=rows[row];let origin=coord(r.cell);let targetArea=f32(r.size*r.size);let count=min(atomicLoad(&incidence[row]),p.axisStride);var negativeSum=vec3f(0);var positiveSum=vec3f(0);var negativeArea=vec3f(0);var positiveArea=vec3f(0);for(var local=0u;local<count;local+=1u){let fi=atomicLoad(&incidence[p.rowCapacity+row*p.axisStride+local]);if(fi>=p.faceCapacity||fi>=arrayLength(&faces)||fi>=arrayLength(&states)||atomicLoad(&states[fi].status)!=ACCEPTED){return;}let f=faces[fi];let axis=f.axisSpan&3u;if(axis>=3u||!finite(f.area)||f.area<=0.){return;}let scalar=states[fi].velocity[axis];if(!finite(scalar)){return;}if(f.negativeRow==row){positiveSum[axis]+=f.area*scalar;positiveArea[axis]+=f.area;}else if(f.positiveRow==row){negativeSum[axis]+=f.area*scalar;negativeArea[axis]+=f.area;}else{return;}}var result=vec3f(0);for(var axis=0u;axis<3u;axis+=1u){if(negativeArea[axis]<targetArea){if(negativeArea[axis]==0.&&positiveArea[axis]>=targetArea){negativeArea[axis]=positiveArea[axis];negativeSum[axis]=positiveSum[axis];}else if(origin[axis]==0u&&(p.closedBoundaryMask&negativeBoundaryBit(axis))!=0u){negativeArea[axis]=targetArea;negativeSum[axis]=0.;}else{return;}}if(positiveArea[axis]<targetArea){if(positiveArea[axis]==0.&&negativeArea[axis]>=targetArea){positiveArea[axis]=negativeArea[axis];positiveSum[axis]=negativeSum[axis];}else if(origin[axis]+r.size==p.dims[axis]&&(p.closedBoundaryMask&positiveBoundaryBit(axis))!=0u){positiveArea[axis]=targetArea;positiveSum[axis]=0.;}else{return;}}let tolerance=1e-4*max(1.,targetArea);if(abs(negativeArea[axis]-targetArea)>tolerance||abs(positiveArea[axis]-targetArea)>tolerance){return;}result[axis]=.5*(negativeSum[axis]/negativeArea[axis]+positiveSum[axis]/positiveArea[axis]);}if(!finite(result.x)||!finite(result.y)||!finite(result.z)){return;}let value=vec4f(result,1.);provisionalVelocities[row]=value;if(row<arrayLength(&rowVelocities)){rowVelocities[row]=value;}}
// S2/S3 rows are interpolation closure, not pressure unknowns.  Section 5's
// outside-liquid operation copies the closest-interface full face velocity;
// average those already accepted face carriers to publish one complete vector
// at every closure cell centre used by the cube/Delaunay interpolant.
@compute @workgroup_size(64)fn completeClosureBandRowVelocity(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row<transitionControl.support1End||row>=transitionControl.support3NodeEnd||row>=arrayLength(&rows)||row>=arrayLength(&provisionalVelocities)||row>=arrayLength(&rowVelocities)){return;}let flags=rows[row].flags;if((flags&(ROW_SUPPORT2|ROW_SUPPORT3_NODE))==0u){return;}let count=min(atomicLoad(&incidence[row]),p.axisStride);var weighted=vec3f(0.);var area=0.;for(var local=0u;local<count;local+=1u){let fi=atomicLoad(&incidence[p.rowCapacity+row*p.axisStride+local]);if(fi>=p.faceCapacity||fi>=arrayLength(&faces)||fi>=arrayLength(&states)||atomicLoad(&states[fi].status)!=ACCEPTED){return;}let face=faces[fi];let carrier=states[fi].velocity;if(!finite(face.area)||face.area<=0.||!velocityValid(carrier)){return;}weighted+=face.area*carrier.xyz;area+=face.area;}if(!finite(area)||area<=0.){return;}let result=weighted/area;if(!finite(result.x)||!finite(result.y)||!finite(result.z)){return;}let value=vec4f(result,1.);provisionalVelocities[row]=value;rowVelocities[row]=value;}
@compute @workgroup_size(64)fn completeDeepClosureBandRowVelocity(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row<transitionControl.support3NodeEnd||row>=transitionControl.support7NodeEnd||row>=arrayLength(&rows)||row>=arrayLength(&provisionalVelocities)||row>=arrayLength(&rowVelocities)){return;}if((rows[row].flags&ROW_SUPPORT3_ENDPOINT)!=0u){return;}let count=min(atomicLoad(&incidence[row]),p.axisStride);var weighted=vec3f(0.);var area=0.;for(var local=0u;local<count;local+=1u){let fi=atomicLoad(&incidence[p.rowCapacity+row*p.axisStride+local]);if(fi>=p.faceCapacity||fi>=arrayLength(&faces)||fi>=arrayLength(&states)||atomicLoad(&states[fi].status)!=ACCEPTED){return;}let face=faces[fi];let carrier=states[fi].velocity;if(!finite(face.area)||face.area<=0.||!velocityValid(carrier)){return;}weighted+=face.area*carrier.xyz;area+=face.area;}if(!finite(area)||area<=0.){return;}let result=weighted/area;if(!finite(result.x)||!finite(result.y)||!finite(result.z)){return;}let value=vec4f(result,1.);provisionalVelocities[row]=value;rowVelocities[row]=value;}
@compute @workgroup_size(64)fn initializeColdClosureBandVelocity(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(p.generation!=2u||row<transitionControl.support1End||row>=transitionControl.support7NodeEnd||row>=arrayLength(&rows)||row>=arrayLength(&provisionalVelocities)||row>=arrayLength(&rowVelocities)||(rows[row].flags&ROW_SUPPORT3_ENDPOINT)!=0u||velocityValid(rowVelocities[row])){return;}let rest=vec4f(0.,0.,0.,1.);provisionalVelocities[row]=rest;rowVelocities[row]=rest;}
@compute @workgroup_size(64)fn completeEndpointBandVelocity(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row<transitionControl.support7NodeEnd||row>=transitionControl.endpointEnd||row>=arrayLength(&rows)||row>=arrayLength(&provisionalVelocities)||row>=arrayLength(&rowVelocities)||(rows[row].flags&ROW_SUPPORT3_ENDPOINT)==0u){return;}let value=supportContainingVector(coord(rows[row].cell));if(!velocityValid(value)){return;}provisionalVelocities[row]=value;rowVelocities[row]=value;}
@compute @workgroup_size(64)fn initializeColdEndpointVelocity(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(p.generation!=2u||row<transitionControl.support7NodeEnd||row>=transitionControl.endpointEnd||row>=arrayLength(&rows)||row>=arrayLength(&provisionalVelocities)||row>=arrayLength(&rowVelocities)||(rows[row].flags&ROW_SUPPORT3_ENDPOINT)==0u){return;}let rest=vec4f(0.,0.,0.,1.);provisionalVelocities[row]=rest;rowVelocities[row]=rest;}
// Aanjaneya et al. (2017), Section 5: liquid power cells retain the full
// centre velocity obtained by least-squares fitting their generalized-face
// normal components.  Fast marching supplies vectors only to the outside-
// liquid support cells that have no corresponding power row.
@compute @workgroup_size(64)fn seedMappedPowerRowVelocity(@builtin(global_invocation_id)g:vec3u){let band=g.x;if(band>=transitionControl.endpointEnd||band>=arrayLength(&rows)||band>=arrayLength(&rowVelocities)||band>=arrayLength(&provisionalVelocities)){return;}let row=rows[band];if(row.globalRow==INVALID){return;}if(row.globalRow>=p.powerRowCapacity||row.globalRow>=powerVelocityControl[2]||row.globalRow>=arrayLength(&powerRowVelocities)){fail(INCOMPLETE,band);return;}let value=powerRowVelocities[row.globalRow];if(!velocityValid(value)){fail(INCOMPLETE,band);return;}rowVelocities[band]=value;provisionalVelocities[band]=value;}
@compute @workgroup_size(1)fn publishFaceBand(){if(atomicLoad(&control.flags)==0u&&atomicLoad(&control.unresolvedCount)==0u&&atomicLoad(&control.acceptedCount)==atomicLoad(&control.faceCount)){atomicStore(&control.valid,VALID);}}
fn pointFail(code:u32,index:u32){atomicOr(&pointControl.flags,code);atomicMin(&pointControl.firstError,index);}
fn storePointError(row:u32,code:u32){var ls=pointAccumulator[row];ls.a2.w=bitcast<f32>(bitcast<u32>(ls.a2.w)|code);pointAccumulator[row]=ls;pointFail(code,row);}
struct BandPowerGeometry{neighborCenter:vec3f,neighborSize:f32,normal:vec3f,inverseDistance:f32,centroid:vec3f,area:f32}struct BandPowerPolygon{vertices:array<vec3f,16>,count:u32}
fn transientFail(code:u32,index:u32){atomicOr(&transientPowerControl.flags,code);atomicMin(&transientPowerControl.firstError,index);}
fn transientFailGeometry(code:u32,index:u32,slot:u32,geometry:BandPowerGeometry,neighbor:u32,reverseSlot:u32){atomicOr(&transientPowerControl.flags,code);let prior=atomicMin(&transientPowerControl.firstError,index);if(index<prior){transientPowerControl.p0=slot;transientPowerControl.p1=bitcast<u32>(geometry.neighborCenter.x);transientPowerControl.p2=bitcast<u32>(geometry.neighborCenter.y);transientPowerControl.p3=bitcast<u32>(geometry.neighborCenter.z);transientPowerControl.p4=bitcast<u32>(geometry.neighborSize);transientPowerControl.p5=neighbor;transientPowerControl.p6=reverseSlot;}}
fn transientFailGeometryStage(code:u32,index:u32,slot:u32,geometry:BandPowerGeometry,neighbor:u32,stage:u32){transientFailGeometry(code,index,slot,geometry,neighbor,stage);}
fn transientRowCenter(row:u32)->vec3f{return vec3f(coord(rows[row].cell))+.5*f32(rows[row].size);}
fn transientCatalogGeometry(row:u32,slot:u32)->BandPowerGeometry{let r=rows[row];let metric=metrics[row];let entry=catalogEntries[metric.topology];let f=catalogFaces[entry.firstFace+slot];let center=transientRowCenter(row);let size=f32(r.size);return BandPowerGeometry(center+size*inversePowerTransform(f.neighborOffsetSize.xyz,metric.transformFlags&63u),size*f.neighborOffsetSize.w,inversePowerTransform(f.normalInverseDistance.xyz,metric.transformFlags&63u),f.normalInverseDistance.w/size,center+size*inversePowerTransform(f.areaCentroid.yzw,metric.transformFlags&63u),size*size*f.areaCentroid.x);}
fn transientWorldBoundaryBit(g:BandPowerGeometry)->u32{if(g.neighborSize==0.){if(g.normal.x<-.9999){return 1u;}if(g.normal.y<-.9999){return 2u;}if(g.normal.z<-.9999){return 4u;}if(g.normal.z>.9999){return 8u;}if(g.normal.y>.9999){return 16u;}if(g.normal.x>.9999){return 32u;}return 0u;}if(g.neighborCenter.x<0.){return 1u;}if(g.neighborCenter.y<0.){return 2u;}if(g.neighborCenter.z<0.){return 4u;}if(g.neighborCenter.z>f32(p.dims.z)){return 8u;}if(g.neighborCenter.y>f32(p.dims.y)){return 16u;}if(g.neighborCenter.x>f32(p.dims.x)){return 32u;}return 0u;}
fn validTransientGeometry(g:BandPowerGeometry,world:bool)->bool{let n2=dot(g.normal,g.normal);return finite(g.neighborSize)&&select(g.neighborSize>0.,g.neighborSize>=0.,world)&&finite(g.neighborCenter.x)&&finite(g.neighborCenter.y)&&finite(g.neighborCenter.z)&&finite(n2)&&abs(n2-1.)<=4e-4&&finite(g.inverseDistance)&&g.inverseDistance>0.&&finite(g.area)&&g.area>0.&&finite(g.centroid.x)&&finite(g.centroid.y)&&finite(g.centroid.z);}
fn transientNeighbor(geometry:BandPowerGeometry)->u32{if(geometry.neighborSize<=0.){return INVALID;}let size=u32(round(geometry.neighborSize));let originValue=geometry.neighborCenter-.5*f32(size);let origin=round(originValue);if(size==0u||abs(geometry.neighborSize-f32(size))>2e-4||any(abs(originValue-origin)>vec3f(2e-4))||any(origin<vec3f(0.))||any(origin+vec3f(f32(size))>vec3f(p.dims))){return INVALID;}let row=rowOf(cell(vec3u(origin)));if(row==INVALID||row>=arrayLength(&rows)||rows[row].cell!=cell(vec3u(origin))||rows[row].size!=size){return INVALID;}return row;}
fn transientReciprocalSlot(row:u32,neighbor:u32)->u32{if(neighbor>=arrayLength(&rows)||neighbor>=arrayLength(&metrics)){return INVALID;}let metric=metrics[neighbor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&catalogEntries)){return INVALID;}let entry=catalogEntries[metric.topology];if(entry.firstFace>arrayLength(&catalogFaces)||entry.faceCount>arrayLength(&catalogFaces)-entry.firstFace){return INVALID;}let wantedCenter=transientRowCenter(row);let wantedSize=f32(rows[row].size);let tolerance=max(1e-5,wantedSize*2e-5);for(var slot=0u;slot<entry.faceCount;slot+=1u){let g=transientCatalogGeometry(neighbor,slot);if(g.neighborSize>0.&&abs(g.neighborSize-wantedSize)<=tolerance&&all(abs(g.neighborCenter-wantedCenter)<=vec3f(tolerance))){return slot;}}return INVALID;}
fn clipTransientPolygon(value:BandPowerPolygon,plane:BandPowerGeometry,epsilon:f32)->BandPowerPolygon{if(value.count<3u){return value;}var clipped:array<vec3f,16>;var output=0u;var previous=value.vertices[value.count-1u];var previousSide=dot(plane.normal,previous-plane.centroid);var previousInside=previousSide<=epsilon;for(var i=0u;i<value.count;i+=1u){let current=value.vertices[i];let currentSide=dot(plane.normal,current-plane.centroid);let currentInside=currentSide<=epsilon;if(currentInside!=previousInside&&output<16u){let denominator=previousSide-currentSide;let t=select(.5,clamp(previousSide/denominator,0.,1.),abs(denominator)>1e-12);clipped[output]=mix(previous,current,t);output+=1u;}if(currentInside&&output<16u){clipped[output]=current;output+=1u;}previous=current;previousSide=currentSide;previousInside=currentInside;}return BandPowerPolygon(clipped,output);}
fn clipTransientByCell(value:BandPowerPolygon,row:u32,epsilon:f32)->BandPowerPolygon{var polygon=value;let metric=metrics[row];let entry=catalogEntries[metric.topology];for(var slot=0u;slot<entry.faceCount&&polygon.count>=3u;slot+=1u){polygon=clipTransientPolygon(polygon,transientCatalogGeometry(row,slot),epsilon);}return polygon;}
fn transientFacePolygon(row:u32,slot:u32,neighbor:u32,reverseSlot:u32)->BandPowerPolygon{let geometry=transientCatalogGeometry(row,slot);let normal=normalize(geometry.normal);let reference=select(vec3f(0.,1.,0.),vec3f(1.,0.,0.),abs(normal.x)<.75);let tangent=normalize(cross(reference,normal));let bitangent=cross(normal,tangent);var neighborScale=f32(rows[row].size);var center=geometry.centroid;if(neighbor!=INVALID){neighborScale=f32(rows[neighbor].size);let reverseGeometry=transientCatalogGeometry(neighbor,reverseSlot);center=.5*(geometry.centroid+reverseGeometry.centroid);}let scale=max(f32(rows[row].size),neighborScale);let extent=8.*scale;var vertices:array<vec3f,16>;vertices[0]=center-extent*tangent-extent*bitangent;vertices[1]=center+extent*tangent-extent*bitangent;vertices[2]=center+extent*tangent+extent*bitangent;vertices[3]=center-extent*tangent+extent*bitangent;let epsilon=max(1e-6,1e-5*scale);var polygon=clipTransientByCell(BandPowerPolygon(vertices,4u),row,epsilon);if(neighbor!=INVALID){polygon=clipTransientByCell(polygon,neighbor,epsilon);}return polygon;}
fn exactTransientGeometry(row:u32,slot:u32,neighbor:u32,reverseSlot:u32)->BandPowerGeometry{var geometry=transientCatalogGeometry(row,slot);let polygon=transientFacePolygon(row,slot,neighbor,reverseSlot);if(polygon.count<3u){geometry.area=0.;return geometry;}var area=0.;var centroid=vec3f(0.);for(var i=1u;i+1u<polygon.count;i+=1u){let triangle=.5*max(0.,dot(cross(polygon.vertices[i]-polygon.vertices[0],polygon.vertices[i+1u]-polygon.vertices[0]),geometry.normal));area+=triangle;centroid+=triangle*(polygon.vertices[0]+polygon.vertices[i]+polygon.vertices[i+1u])/3.;}geometry.area=area;if(area>1e-12){geometry.centroid=centroid/area;}return geometry;}
@compute @workgroup_size(1)fn prepareTransientBandPowerGraph(){atomicStore(&transientPowerControl.flags,0u);atomicStore(&transientPowerControl.firstError,INVALID);transientPowerControl.rowCount=transitionControl.support1End;transientPowerControl.faceSlots=transientPowerControl.rowCount*POINT_MAX_FACES;atomicStore(&transientPowerControl.emitted,0u);atomicStore(&transientPowerControl.sampled,0u);atomicStore(&transientPowerControl.validated,0u);transientPowerControl.generation=p.generation;atomicStore(&transientPowerControl.valid,0u);transientPowerControl.p0=INVALID;transientPowerControl.p1=INVALID;transientPowerControl.p2=INVALID;transientPowerControl.p3=INVALID;transientPowerControl.p4=INVALID;transientPowerControl.p5=INVALID;transientPowerControl.p6=INVALID;if(transitionControl.support1End>transitionControl.support2End||transientPowerControl.rowCount>p.rowCapacity||transientPowerControl.faceSlots>arrayLength(&transientPowerFaces)||transientPowerControl.faceSlots>arrayLength(&transientPowerIncidences)||transientPowerControl.rowCount>arrayLength(&transientPowerRows)){transientFail(POINT_CAPACITY,0u);}}
@compute @workgroup_size(64)fn emitTransientBandPowerGraph(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=transientPowerControl.rowCount||row>=arrayLength(&rows)||row>=arrayLength(&metrics)||atomicLoad(&transientPowerControl.flags)!=0u){return;}let base=row*POINT_MAX_FACES;if(base>arrayLength(&transientPowerFaces)||POINT_MAX_FACES>arrayLength(&transientPowerFaces)-base){transientFail(POINT_CAPACITY,row);return;}for(var retired=0u;retired<POINT_MAX_FACES;retired+=1u){transientPowerFaces[base+retired].flags=0u;}let metric=metrics[row];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&catalogEntries)){transientFail(POINT_SOURCE,row);return;}let entry=catalogEntries[metric.topology];if(entry.faceCount==0u||entry.faceCount>POINT_MAX_FACES||entry.firstFace>arrayLength(&catalogFaces)||entry.faceCount>arrayLength(&catalogFaces)-entry.firstFace){transientFail(POINT_CAPACITY,row);return;}transientPowerRows[row]=PowerRowWork(entry.faceCount,entry.faceCount,base,base);for(var slot=0u;slot<entry.faceCount;slot+=1u){let geometry=transientCatalogGeometry(row,slot);let declared=(metric.transformFlags>>8u)&63u;let boundaryBit=transientWorldBoundaryBit(geometry);let world=geometry.neighborSize==0.||(boundaryBit&declared)!=0u;if(!validTransientGeometry(geometry,world)){transientFailGeometryStage(POINT_FACE,row,slot,geometry,INVALID,1u);return;}var neighbor=INVALID;var reverseSlot=INVALID;if(!world){neighbor=transientNeighbor(geometry);if(neighbor==INVALID||neighbor>=transitionControl.support2End){transientFailGeometryStage(POINT_SAMPLE,row,slot,geometry,neighbor,2u);return;}reverseSlot=transientReciprocalSlot(row,neighbor);if(reverseSlot==INVALID){transientFailGeometryStage(POINT_FACE,row,slot,geometry,neighbor,3u);return;}let reverse=transientCatalogGeometry(neighbor,reverseSlot);if(dot(geometry.normal,reverse.normal)>-.999||abs(geometry.inverseDistance-reverse.inverseDistance)>max(1e-5,geometry.inverseDistance*2e-4)){transientFailGeometryStage(POINT_FACE,row,slot,geometry,neighbor,4u);return;}if(row>neighbor){continue;}}let exact=exactTransientGeometry(row,slot,neighbor,reverseSlot);if(!validTransientGeometry(exact,world)){transientFailGeometryStage(POINT_FACE,row,slot,exact,neighbor,5u);return;}let faceIndex=base+slot;var plane=0u;if(world){plane=boundaryBit&declared;if(plane==0u){transientFailGeometryStage(POINT_FACE,row,slot,exact,neighbor,7u);return;}}transientPowerFaces[faceIndex]=TransientPowerFace(row,neighbor,1u|(plane<<8u),0u,vec4f(exact.normal,0.),vec4f(exact.centroid,1.),0.,exact.area,exact.inverseDistance,0.);transientPowerIncidences[base+slot]=PowerIncidence(faceIndex,1);if(neighbor!=INVALID){transientPowerIncidences[neighbor*POINT_MAX_FACES+reverseSlot]=PowerIncidence(faceIndex,-1);}atomicAdd(&transientPowerControl.emitted,1u);}}
@compute @workgroup_size(64)fn sampleTransientBandPowerFaces(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=transientPowerControl.rowCount||atomicLoad(&transientPowerControl.flags)!=0u){return;}let base=row*POINT_MAX_FACES;for(var slot=0u;slot<POINT_MAX_FACES;slot+=1u){let faceIndex=base+slot;if(faceIndex>=transientPowerControl.faceSlots||faceIndex>=arrayLength(&transientPowerFaces)){transientFail(POINT_CAPACITY,row);return;}var face=transientPowerFaces[faceIndex];if((face.flags&1u)==0u){continue;}let plane=(face.flags>>8u)&63u;var scalar=0.;if(plane!=0u){if((p.closedBoundaryMask&plane)==0u){if(plane!=positiveBoundaryBit(1u)||face.negativeRow>=arrayLength(&rows)||!finite(rows[face.negativeRow].padf)){transientFail(POINT_FACE,faceIndex);return;}scalar=face.normal.y*rows[face.negativeRow].padf;}}else{let negative=marchedCentroidVector(face.negativeRow,face.centroid.xyz);let positive=marchedCentroidVector(face.positiveRow,face.centroid.xyz);let negativeValid=velocityValid(negative);let positiveValid=velocityValid(positive);var full=vec3f(0.);if(negativeValid&&positiveValid){full=.5*(negative.xyz+positive.xyz);}else if(negativeValid){full=negative.xyz;}else if(positiveValid){full=positive.xyz;}else{transientFail(POINT_SAMPLE,faceIndex);return;}scalar=dot(full,face.normal.xyz);}if(!finite(scalar)){transientFail(POINT_NONFINITE,faceIndex);return;}face.normalVelocity=scalar;face.flags|=2u;transientPowerFaces[faceIndex]=face;atomicAdd(&transientPowerControl.sampled,1u);}}
@compute @workgroup_size(64)fn validateTransientBandPowerGraph(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=transientPowerControl.rowCount||row>=arrayLength(&transientPowerRows)||atomicLoad(&transientPowerControl.flags)!=0u){return;}let work=transientPowerRows[row];if(work.incidenceCount==0u||work.incidenceCount>POINT_MAX_FACES||work.incidenceOffset>arrayLength(&transientPowerIncidences)||work.incidenceCount>arrayLength(&transientPowerIncidences)-work.incidenceOffset){transientFail(POINT_CAPACITY,row);return;}for(var local=0u;local<work.incidenceCount;local+=1u){let incidence=transientPowerIncidences[work.incidenceOffset+local];if((incidence.sign!=1&&incidence.sign!=-1)||incidence.face>=arrayLength(&transientPowerFaces)){transientFail(POINT_FACE,row);return;}let face=transientPowerFaces[incidence.face];let signMatches=select(face.positiveRow==row,face.negativeRow==row,incidence.sign==1);if((face.flags&3u)!=3u||!signMatches){transientFail(POINT_FACE,row);return;}}atomicAdd(&transientPowerControl.validated,1u);}
@compute @workgroup_size(1)fn publishTransientBandPowerGraph(){let emitted=atomicLoad(&transientPowerControl.emitted);if(atomicLoad(&transientPowerControl.flags)==0u&&transientPowerControl.generation==p.generation&&emitted>0u&&atomicLoad(&transientPowerControl.sampled)==emitted&&atomicLoad(&transientPowerControl.validated)==transientPowerControl.rowCount){atomicStore(&transientPowerControl.valid,VALID);}else{atomicStore(&transientPowerControl.valid,0u);}}
@compute @workgroup_size(1)fn prepareBandPointField(){atomicStore(&pointControl.flags,0u);atomicStore(&pointControl.firstError,INVALID);pointControl.rowCount=transitionControl.support1End;pointControl.generation=p.generation;atomicStore(&pointControl.solved,0u);atomicStore(&pointControl.valid,0u);pointControl.pad=transitionControl.coreEnd;if(atomicLoad(&control.valid)!=VALID||atomicLoad(&control.flags)!=0u||control.generation!=p.generation||atomicLoad(&transitionControl.ready)!=VALID||transitionControl.transferReady!=VALID||atomicLoad(&transitionControl.hierarchyReady)!=VALID||atomicLoad(&transitionControl.flags)!=0u||transitionControl.support1End>transitionControl.support2End||atomicLoad(&transientPowerControl.valid)!=VALID||atomicLoad(&transientPowerControl.flags)!=0u||transientPowerControl.generation!=p.generation||transientPowerControl.rowCount!=pointControl.rowCount){pointFail(POINT_SOURCE,0u);}}
@compute @workgroup_size(1)fn prepareBandPointDispatch(){writeSupportDispatch(6u,pointControl.rowCount);}
@compute @workgroup_size(64)fn prepareBandPointRows(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=pointControl.rowCount||row>=arrayLength(&rows)||row>=arrayLength(&pointAccumulator)){return;}pointAccumulator[row]=BandLS(vec4f(0.),vec4f(0.),vec4f(0.));}
@compute @workgroup_size(64)fn accumulateBandTransientPowerLS(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=pointControl.rowCount||row>=arrayLength(&pointAccumulator)){return;}if(atomicLoad(&transientPowerControl.valid)!=VALID||transientPowerControl.generation!=p.generation||row>=transientPowerControl.rowCount||row>=arrayLength(&transientPowerRows)){storePointError(row,POINT_SOURCE);return;}let work=transientPowerRows[row];if(work.incidenceCount==0u||work.incidenceCount>POINT_MAX_FACES||work.incidenceOffset>arrayLength(&transientPowerIncidences)||work.incidenceCount>arrayLength(&transientPowerIncidences)-work.incidenceOffset){storePointError(row,POINT_CAPACITY);return;}var ls=pointAccumulator[row];for(var local=0u;local<work.incidenceCount;local+=1u){let item=transientPowerIncidences[work.incidenceOffset+local];if((item.sign!=1&&item.sign!=-1)||item.face>=arrayLength(&transientPowerFaces)){storePointError(row,POINT_FACE);return;}let face=transientPowerFaces[item.face];let signMatches=select(face.positiveRow==row,face.negativeRow==row,item.sign==1);if((face.flags&3u)!=3u||!signMatches){storePointError(row,POINT_FACE);return;}let orientation=f32(item.sign);let normal=orientation*face.normal.xyz;let u=orientation*face.normalVelocity;let n2=dot(normal,normal);let weight=face.area;if(!finite(normal.x)||!finite(normal.y)||!finite(normal.z)||!finite(n2)||abs(n2-1.)>4e-4||!finite(u)||!finite(weight)||weight<=0.){storePointError(row,select(POINT_FACE,POINT_NORMAL,!finite(n2)||abs(n2-1.)>4e-4));return;}ls.a0.x+=weight*normal.x*normal.x;ls.a0.y+=weight*normal.x*normal.y;ls.a0.z+=weight*normal.x*normal.z;ls.a0.w+=weight*normal.y*normal.y;ls.a1.x+=weight*normal.y*normal.z;ls.a1.y+=weight*normal.z*normal.z;ls.a1.z+=weight*normal.x*u;ls.a1.w+=weight*normal.y*u;ls.a2.x+=weight*normal.z*u;}pointAccumulator[row]=ls;}
@compute @workgroup_size(64)fn solveBandPowerLS(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=pointControl.rowCount||row>=arrayLength(&pointAccumulator)||row>=arrayLength(&pointStatus)||row>=arrayLength(&rowVelocities)){return;}let ls=pointAccumulator[row];let prior=bitcast<u32>(ls.a2.w);if(prior!=0u){atomicStore(&pointStatus[row],prior);return;}let xx=ls.a0.x;let xy=ls.a0.y;let xz=ls.a0.z;let yy=ls.a0.w;let yz=ls.a1.x;let zz=ls.a1.y;let b=vec3f(ls.a1.z,ls.a1.w,ls.a2.x);let c00=yy*zz-yz*yz;let c01=xz*yz-xy*zz;let c02=xy*yz-xz*yy;let c11=xx*zz-xz*xz;let c12=xy*xz-xx*yz;let c22=xx*yy-xy*xy;let determinant=xx*c00+xy*c01+xz*c02;let trace=xx+yy+zz;let matrixNorm2=xx*xx+yy*yy+zz*zz+2.*(xy*xy+xz*xz+yz*yz);let adjugateNorm2=c00*c00+c11*c11+c22*c22+2.*(c01*c01+c02*c02+c12*c12);if(!finite(determinant)||!finite(trace)||determinant<=1e-7*trace*trace*trace){atomicStore(&pointStatus[row],POINT_SINGULAR);return;}let condition=sqrt(matrixNorm2*adjugateNorm2)/determinant;if(!finite(condition)||condition>1e5){atomicStore(&pointStatus[row],POINT_CONDITION);return;}let velocity=vec3f(c00*b.x+c01*b.y+c02*b.z,c01*b.x+c11*b.y+c12*b.z,c02*b.x+c12*b.y+c22*b.z)/determinant;if(!finite(velocity.x)||!finite(velocity.y)||!finite(velocity.z)){atomicStore(&pointStatus[row],POINT_NONFINITE);return;}rowVelocities[row]=vec4f(velocity,1.);atomicStore(&pointStatus[row],0u);atomicAdd(&pointControl.solved,1u);}
@compute @workgroup_size(64)fn validateBandPointField(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=pointControl.rowCount){return;}if(row>=arrayLength(&pointStatus)){pointFail(POINT_CAPACITY,row);return;}let status=atomicLoad(&pointStatus[row]);if(status!=0u){pointFail(status,row);}}
@compute @workgroup_size(1)fn publishBandPointField(){if(atomicLoad(&pointControl.flags)==0u&&atomicLoad(&pointControl.solved)==pointControl.rowCount&&pointControl.generation==p.generation&&atomicLoad(&transientPowerControl.valid)==VALID&&atomicLoad(&transientPowerControl.flags)==0u&&transientPowerControl.generation==p.generation&&transientPowerControl.rowCount==pointControl.rowCount){atomicStore(&pointControl.valid,VALID);}else{atomicStore(&pointControl.valid,0u);}}
fn powerFail(code:u32,index:u32){atomicOr(&powerPublication.flags,code);atomicMin(&powerPublication.firstError,index);}
@compute @workgroup_size(1)fn preparePowerPublication(){atomicStore(&powerPublication.flags,0u);atomicStore(&powerPublication.firstError,INVALID);powerPublication.faceCount=0u;atomicStore(&powerPublication.targetCount,0u);atomicStore(&powerPublication.interpolatedCount,0u);atomicStore(&powerPublication.committedCount,0u);powerPublication.fineGeneration=p.generation;powerPublication.powerGeneration=p.powerGeneration;atomicStore(&powerPublication.valid,0u);if(arrayLength(&powerFaceControl)<9u){powerFail(POWER_CAPACITY,0u);return;}powerPublication.faceCount=powerFaceControl[1];if(atomicLoad(&control.valid)!=VALID||atomicLoad(&control.flags)!=0u||control.generation!=p.generation||atomicLoad(&transitionControl.ready)!=VALID||transitionControl.transferReady!=VALID||atomicLoad(&transitionControl.flags)!=0u||atomicLoad(&pointControl.valid)!=VALID||atomicLoad(&pointControl.flags)!=0u||pointControl.generation!=p.generation||powerFaceControl[3]!=0u||powerFaceControl[8]!=VALID||powerFaceControl[7]!=p.powerGeneration){powerFail(POWER_SOURCE,0u);return;}if(powerFaceControl[0]>p.powerRowCapacity||powerFaceControl[1]>arrayLength(&powerFaces)||powerFaceControl[1]>arrayLength(&powerFaceNormals)||powerFaceControl[1]>arrayLength(&powerFaceCentroids)||powerFaceControl[1]>arrayLength(&powerVelocityScratch)){powerFail(POWER_CAPACITY,powerFaceControl[1]);}}
// Paper Section 5 extrapolation is one-sided: it assigns velocities outside
// the liquid and must not overwrite a projected face incident to liquid. A
// band row is wet when its marched centre phi is negative — the same current
// centre-sign convention the pressure solve uses. Preserve both liquid-liquid
// and liquid-air/interface faces; only wholly air-side targets consume the
// extrapolated full-vector field. This also prevents cell-centre reconstruction
// from re-smoothing the pressure-projected free-surface normal velocity.
fn bandRowIsWet(band:u32)->bool{return band!=INVALID&&band<arrayLength(&rows)&&(rows[band].flags&ROW_PHI)!=0u&&finite(rows[band].representativePhi)&&rows[band].representativePhi<0.0;}
@compute @workgroup_size(64)fn mapPowerFaceBands(@builtin(global_invocation_id)g:vec3u){let index=g.x;if(index>=powerPublication.faceCount||atomicLoad(&powerPublication.flags)!=0u){return;}if(index>=arrayLength(&powerFaces)||index>=arrayLength(&powerVelocityScratch)){powerFail(POWER_CAPACITY,index);return;}let powerFace=powerFaces[index];let negativeBand=bandForGlobalRow(powerFace.negativeRow);let positiveBand=bandForGlobalRow(powerFace.positiveRow);powerVelocityScratch[index]=vec4u(0u,0u,negativeBand,positiveBand);if(negativeBand!=INVALID||positiveBand!=INVALID){atomicAdd(&powerPublication.targetCount,1u);}}
fn provisionalCellVector(origin:vec3u,size:u32)->vec4f{if(size==0u||any(origin+vec3u(size)>p.dims)){return vec4f(0);}let band=rowOf(cell(origin));if(band==INVALID||band>=arrayLength(&rows)||band>=arrayLength(&provisionalVelocities)){return vec4f(0);}let row=rows[band];let value=provisionalVelocities[band];if(row.cell!=cell(origin)||row.size!=size||(row.flags&ROW_SUPPORT3_ENDPOINT)!=0u||!velocityValid(value)){return vec4f(0);}return value;}
fn finalCellVector(origin:vec3u,size:u32)->vec4f{if(size==0u||any(origin+vec3u(size)>p.dims)){return vec4f(0);}let band=rowOf(cell(origin));if(band==INVALID||band>=arrayLength(&rows)||band>=arrayLength(&rowVelocities)){return vec4f(0);}let row=rows[band];let value=rowVelocities[band];if(row.cell!=cell(origin)||row.size!=size||(row.flags&(ROW_SUPPORT2|ROW_SUPPORT3_NODE|ROW_SUPPORT3_ENDPOINT))!=0u||!velocityValid(value)){return vec4f(0);}return value;}
fn supportCellVector(origin:vec3u,size:u32)->vec4f{if(size==0u||any(origin+vec3u(size)>p.dims)){return vec4f(0);}let band=rowOf(cell(origin));if(band==INVALID||band>=arrayLength(&rows)||band>=arrayLength(&rowVelocities)){return vec4f(0);}let row=rows[band];let value=rowVelocities[band];if(row.cell!=cell(origin)||row.size!=size||(row.flags&ROW_SUPPORT3_ENDPOINT)!=0u||!velocityValid(value)){return vec4f(0);}return value;}
fn supportSelectorCellVector(origin:vec3u,size:u32)->vec4f{if(size==0u||any(origin+vec3u(size)>p.dims)){return vec4f(0);}let band=rowOf(cell(origin));if(band==INVALID||band>=arrayLength(&rows)||band>=arrayLength(&rowVelocities)){return vec4f(0);}let row=rows[band];let value=rowVelocities[band];let coldEndpoint=p.generation==2u&&(row.flags&ROW_SUPPORT3_ENDPOINT)!=0u;if(row.cell!=cell(origin)||row.size!=size||(!coldEndpoint&&(row.flags&ROW_SUPPORT3_ENDPOINT)!=0u)||!velocityValid(value)){return vec4f(0);}return value;}
fn provisionalContainingVector(q:vec3u)->vec4f{var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);if(all(origin+vec3u(size)<=p.dims)){let value=provisionalCellVector(origin,size);if(velocityValid(value)){return value;}}if(size>=p.maximumLeaf){break;}size<<=1u;}return vec4f(0);}
fn finalContainingVector(q:vec3u)->vec4f{var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);if(all(origin+vec3u(size)<=p.dims)){let value=finalCellVector(origin,size);if(velocityValid(value)){return value;}}if(size>=p.maximumLeaf){break;}size<<=1u;}return vec4f(0);}
fn supportContainingVector(q:vec3u)->vec4f{var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);if(all(origin+vec3u(size)<=p.dims)){let value=supportCellVector(origin,size);if(velocityValid(value)){return value;}}if(size>=p.maximumLeaf){break;}size<<=1u;}return vec4f(0);}
fn provisionalSignedVector(origin:vec3i,size:u32)->vec4f{let reflected=velocityExtendedOrigin(origin,size);if(reflected.w<0){return vec4f(0);}var value=provisionalCellVector(vec3u(reflected.xyz),size);let outside=any(origin<vec3i(0))||any(origin+vec3i(i32(size))>vec3i(p.dims));if(!velocityValid(value)&&outside){value=provisionalContainingVector(vec3u(reflected.xyz));}return reflectComponents(value,reflected.w);}
fn finalSignedVector(origin:vec3i,size:u32)->vec4f{let reflected=velocityExtendedOrigin(origin,size);if(reflected.w<0){return vec4f(0);}let q=vec3u(reflected.xyz);var value=supportSelectorCellVector(q,size);
 // S2/S3-node rows are support for interpolation, not point-field anchors.
 // Their exact Section 5 carrier is reconstructed directly from the accepted
 // fast-marched faces. Prefer the final S0/S1 point LS, then consume that
 // face-derived carrier only when a selector/corner requires deeper support.
 if(!velocityValid(value)){value=supportContainingVector(q);}return reflectComponents(value,reflected.w);}
fn provisionalSelectorVector(anchor:u32,selector:u32,transform:u32)->vec4f{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return vec4f(0);}let row=rows[anchor];let v=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(v.xyz,transform);let sizeFloat=f32(row.size)*v.w;let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return vec4f(0);}return provisionalSignedVector(vec3i(round(point-.5*f32(size))),size);}
fn finalSelectorVector(anchor:u32,selector:u32,transform:u32)->vec4f{if(anchor>=arrayLength(&rows)||selector>=arrayLength(&tetraVertices)){return vec4f(0);}let row=rows[anchor];let v=tetraVertices[selector].v;let center=vec3f(coord(row.cell))+.5*f32(row.size);let point=center+f32(row.size)*inversePowerTransform(v.xyz,transform);let sizeFloat=f32(row.size)*v.w;let size=u32(round(sizeFloat));if(size==0u||abs(sizeFloat-f32(size))>1e-4){return vec4f(0);}return finalSignedVector(vec3i(round(point-.5*f32(size))),size);}
fn finalTetraPointVector(anchor:u32,row:Row,anchorVelocity:vec4f,metric:Metric,header:TetraHeader,pointGrid:vec3f)->vec4f{if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return invalidPointVector(7u);}let origin=coord(row.cell);let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return invalidPointVector(8u);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}let va=finalSelectorVector(anchor,selectors.x,metric.transformFlags&63u);let vb=finalSelectorVector(anchor,selectors.y,metric.transformFlags&63u);let vc=finalSelectorVector(anchor,selectors.z,metric.transformFlags&63u);if(!velocityValid(va)||!velocityValid(vb)||!velocityValid(vc)){return invalidPointVector(9u);}let result=weights.x*anchorVelocity.xyz+weights.y*va.xyz+weights.z*vb.xyz+weights.w*vc.xyz;return select(invalidPointVector(10u),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}return invalidPointVectorAt(11u,vec3f(origin));}
fn marchedCentroidVector(anchor:u32,pointGrid:vec3f)->vec4f{if(anchor==INVALID||anchor>=arrayLength(&rows)||anchor>=arrayLength(&metrics)||(rows[anchor].flags&(ROW_SUPPORT3_NODE|ROW_SUPPORT3_ENDPOINT))!=0u){return vec4f(0);}let row=rows[anchor];let origin=coord(row.cell);let anchorVelocity=provisionalCellVector(origin,row.size);if(!velocityValid(anchorVelocity)){return vec4f(0);}let metric=metrics[anchor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){return vec4f(0);}let header=tetraHeaders[metric.topology];if((header.flags&1u)!=0u){let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);if(all(t>=vec3f(-2e-6))&&all(t<=vec3f(1.000002))){var result=vec3f(0);var cubeValid=true;for(var corner=0u;corner<8u;corner+=1u){let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);if(weight==0.){continue;}let cornerOrigin=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u))*i32(row.size);let value=provisionalSignedVector(cornerOrigin,row.size);if(!velocityValid(value)){cubeValid=false;break;}result+=weight*value.xyz;}if(cubeValid){return select(vec4f(0),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}}}if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return vec4f(0);}let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return vec4f(0);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}let va=provisionalSelectorVector(anchor,selectors.x,metric.transformFlags&63u);let vb=provisionalSelectorVector(anchor,selectors.y,metric.transformFlags&63u);let vc=provisionalSelectorVector(anchor,selectors.z,metric.transformFlags&63u);if(!velocityValid(va)||!velocityValid(vb)||!velocityValid(vc)){return vec4f(0);}let result=weights.x*anchorVelocity.xyz+weights.y*va.xyz+weights.z*vb.xyz+weights.w*vc.xyz;return select(vec4f(0),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}return vec4f(0);}
fn invalidPointVector(reason:u32)->vec4f{return vec4f(0.,0.,0.,-f32(reason));}
fn invalidPointVectorAt(reason:u32,detail:vec3f)->vec4f{return vec4f(detail,-f32(reason));}
fn uniformCubeContains(row:Row,pointGrid:vec3f)->bool{let origin=coord(row.cell);let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);return all(t>=vec3f(-2e-6))&&all(t<=vec3f(1.000002));}
fn finalPointVector(anchor:u32,pointGrid:vec3f)->vec4f{if(anchor==INVALID||anchor>=arrayLength(&rows)||anchor>=arrayLength(&metrics)||(rows[anchor].flags&ROW_SUPPORT3_ENDPOINT)!=0u){return invalidPointVector(1u);}let row=rows[anchor];let origin=coord(row.cell);var anchorVelocity=finalCellVector(origin,row.size);if(!velocityValid(anchorVelocity)&&(row.flags&(ROW_SUPPORT2|ROW_SUPPORT3_NODE))!=0u){anchorVelocity=supportCellVector(origin,row.size);}if(!velocityValid(anchorVelocity)){return invalidPointVector(2u);}let metric=metrics[anchor];if((metric.transformFlags&VALID)==0u||metric.topology>=arrayLength(&tetraHeaders)){return invalidPointVector(3u);}let header=tetraHeaders[metric.topology];if((header.flags&1u)!=0u){let low=select(vec3i(origin)-vec3i(i32(row.size)),vec3i(origin),pointGrid>=vec3f(origin)+.5*f32(row.size));let t=(pointGrid-(vec3f(low)+.5*f32(row.size)))/f32(row.size);if(uniformCubeContains(row,pointGrid)){var result=vec3f(0);var cubeValid=true;for(var corner=0u;corner<8u;corner+=1u){let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);if(weight==0.){continue;}let cornerOrigin=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u))*i32(row.size);let value=finalSignedVector(cornerOrigin,row.size);if(!velocityValid(value)){cubeValid=false;break;}result+=weight*value.xyz;}if(cubeValid){return select(invalidPointVector(6u),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}}return finalTetraPointVector(anchor,row,anchorVelocity,metric,header,pointGrid);}if(header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){return invalidPointVector(7u);}let point=powerTransform((pointGrid-(vec3f(origin)+.5*f32(row.size)))/f32(row.size),metric.transformFlags&63u);for(var local=0u;local<header.count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return invalidPointVector(8u);}let weights=tetraWeights(point,tetraVertices[selectors.x].v.xyz,tetraVertices[selectors.y].v.xyz,tetraVertices[selectors.z].v.xyz);if(!contained(weights)){continue;}let va=finalSelectorVector(anchor,selectors.x,metric.transformFlags&63u);let vb=finalSelectorVector(anchor,selectors.y,metric.transformFlags&63u);let vc=finalSelectorVector(anchor,selectors.z,metric.transformFlags&63u);if(!velocityValid(va)||!velocityValid(vb)||!velocityValid(vc)){return invalidPointVector(9u);}let result=weights.x*anchorVelocity.xyz+weights.y*va.xyz+weights.z*vb.xyz+weights.w*vc.xyz;return select(invalidPointVector(10u),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z));}return invalidPointVectorAt(11u,vec3f(origin));}
// Resolve the published row containing an integer probe without scanning the
// row arena. Multiple support levels can cover the same probe; selecting the
// lowest row id matches the publication-prefix order used by the former scan.
fn containingPublishedRow(probe:vec3i)->u32{if(any(probe<vec3i(0))||any(probe>=vec3i(p.dims))){return INVALID;}let q=vec3u(probe);var size=1u;var selected=INVALID;loop{let origin=(q/vec3u(size))*vec3u(size);let candidate=rowOf(cell(origin));if(candidate!=INVALID&&candidate<arrayLength(&rows)&&candidate<arrayLength(&rowVelocities)){let r=rows[candidate];if(r.cell==cell(origin)&&r.size==size&&(r.flags&ROW_SUPPORT3_ENDPOINT)==0u&&all(q>=origin)&&all(q<origin+vec3u(size))){selected=min(selected,candidate);}}if(size>=p.maximumLeaf){break;}size<<=1u;}return selected;}
// Co-spherical Cartesian sites admit several ordinary-Delaunay
// tetrahedralizations. A static anchor fan can legitimately choose the
// axial-star triangulation while a trajectory lies in the complementary
// central octahedron. Section 6.2 permits any deterministic ordinary-Delaunay
// triangulation, so resolve that degeneracy from the actual eight surrounding
// adaptive owners. No projected/nearest value is accepted: a candidate must
// contain the query and have an empty circumsphere against the complete,
// deduplicated local owner set.
struct SurroundingOwnerVectorMeasurement{value:vec4f,rowsTested:u32}
fn surroundingOwnerDelaunayVectorMeasured(pointGrid:vec3f)->SurroundingOwnerVectorMeasurement{
 let low=vec3i(floor(pointGrid-vec3f(.5)));var ids:array<u32,8>;for(var i=0u;i<8u;i+=1u){ids[i]=INVALID;}var count=0u;var rowsTested=0u;
 for(var corner=0u;corner<8u;corner+=1u){let probe=low+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));if(any(probe<vec3i(0))||any(probe>=vec3i(p.dims))){continue;}let row=containingPublishedRow(probe);rowsTested+=1u;if(row==INVALID||!velocityValid(rowVelocities[row])){return SurroundingOwnerVectorMeasurement(invalidPointVectorAt(15u,vec3f(bitcast<f32>(row),0.,0.)),rowsTested);}var duplicate=false;var insert=count;for(var prior=0u;prior<count;prior+=1u){if(ids[prior]==row){duplicate=true;break;}if(insert==count&&row<ids[prior]){insert=prior;}}if(duplicate){continue;}if(count>=8u){return SurroundingOwnerVectorMeasurement(invalidPointVector(16u),rowsTested);}for(var shift=count;shift>insert;shift-=1u){ids[shift]=ids[shift-1u];}ids[insert]=row;count+=1u;}
 if(count<4u){return SurroundingOwnerVectorMeasurement(invalidPointVector(17u),rowsTested);}for(var ia=0u;ia+3u<count;ia+=1u){for(var ib=ia+1u;ib+2u<count;ib+=1u){for(var ic=ib+1u;ic+1u<count;ic+=1u){for(var id=ic+1u;id<count;id+=1u){let ra=rows[ids[ia]];let rb=rows[ids[ib]];let rc=rows[ids[ic]];let rd=rows[ids[id]];let a=vec3f(coord(ra.cell))+.5*f32(ra.size);let b=vec3f(coord(rb.cell))+.5*f32(rb.size);let c=vec3f(coord(rc.cell))+.5*f32(rc.size);let d=vec3f(coord(rd.cell))+.5*f32(rd.size);let u=b-a;let v=c-a;let w=d-a;let determinant=dot(u,cross(v,w));if(!finite(determinant)||abs(determinant)<=1e-8){continue;}let weights=tetraWeights(pointGrid-a,u,v,w);if(!contained(weights)){continue;}let rhs=vec3f(.5*dot(u,u),.5*dot(v,v),.5*dot(w,w));let center=(rhs.x*cross(v,w)+rhs.y*cross(w,u)+rhs.z*cross(u,v))/determinant;let radius2=dot(center,center);if(!finite(radius2)){continue;}let tolerance=2e-5*max(1.,radius2);var empty=true;for(var site=0u;site<count;site+=1u){let sr=rows[ids[site]];let delta=(vec3f(coord(sr.cell))+.5*f32(sr.size))-(a+center);if(dot(delta,delta)<radius2-tolerance){empty=false;break;}}if(!empty){continue;}let va=rowVelocities[ids[ia]];let vb=rowVelocities[ids[ib]];let vc=rowVelocities[ids[ic]];let vd=rowVelocities[ids[id]];let result=weights.x*va.xyz+weights.y*vb.xyz+weights.z*vc.xyz+weights.w*vd.xyz;return SurroundingOwnerVectorMeasurement(select(invalidPointVector(18u),vec4f(result,1.),finite(result.x)&&finite(result.y)&&finite(result.z)),rowsTested);}}}}
 return SurroundingOwnerVectorMeasurement(invalidPointVector(19u),rowsTested);
}
fn surroundingOwnerDelaunayVector(pointGrid:vec3f)->vec4f{return surroundingOwnerDelaunayVectorMeasured(pointGrid).value;}
// The owner containing a trajectory sample need not anchor the catalog fan
// that contains it. Catalog selectors are bounded by 1.5 anchor widths in
// every axis, so a containing cube/tetra anchor of size s must lie in the 5^3
// aligned-origin box around floor(point/s). Enumerate that exact bounded box
// for every dyadic size and select the lowest valid row id, reproducing the
// former ascending-row result without work proportional to row capacity.
struct LocatedFinalPointVector{value:vec4f,directSuccess:u32,fullRowFallback:u32,candidateRowsTested:u32,surroundingFallback:u32,surroundingRowsTested:u32}
fn locateFinalPointVectorMeasured(initialAnchor:u32,pointGrid:vec3f)->LocatedFinalPointVector{let direct=finalPointVector(initialAnchor,pointGrid);if(velocityValid(direct)){return LocatedFinalPointVector(direct,1u,0u,0u,0u,0u);}var bestRow=INVALID;var bestValue=vec4f(0.);var containedInvalidRow=INVALID;var containedInvalid=vec4f(0.);var candidateRowsTested=0u;var size=1u;loop{let base=vec3i(floor(pointGrid/f32(size)));for(var dz=-2i;dz<=2i;dz+=1i){for(var dy=-2i;dy<=2i;dy+=1i){for(var dx=-2i;dx<=2i;dx+=1i){let originIndex=base+vec3i(dx,dy,dz);if(any(originIndex<vec3i(0))){continue;}let origin=vec3u(originIndex)*size;if(any(origin+vec3u(size)>p.dims)){continue;}let candidate=rowOf(cell(origin));if(candidate==INVALID||candidate==initialAnchor||candidate>=arrayLength(&rows)||candidate>=arrayLength(&metrics)){continue;}let row=rows[candidate];if(row.cell!=cell(origin)||row.size!=size||(row.flags&ROW_SUPPORT3_ENDPOINT)!=0u){continue;}candidateRowsTested+=1u;if(candidate>=bestRow){continue;}let value=finalPointVector(candidate,pointGrid);if(velocityValid(value)){bestRow=candidate;bestValue=value;continue;}if(u32(round(max(0.,-value.w)))==9u&&candidate<containedInvalidRow){containedInvalidRow=candidate;containedInvalid=value;}}}}if(size>=p.maximumLeaf){break;}size<<=1u;}if(bestRow!=INVALID){return LocatedFinalPointVector(bestValue,0u,1u,candidateRowsTested,0u,0u);}let local=surroundingOwnerDelaunayVectorMeasured(pointGrid);if(velocityValid(local.value)){return LocatedFinalPointVector(local.value,0u,1u,candidateRowsTested,1u,local.rowsTested);}return LocatedFinalPointVector(select(direct,containedInvalid,containedInvalidRow!=INVALID),0u,1u,candidateRowsTested,1u,local.rowsTested);}
fn locateFinalPointVector(initialAnchor:u32,pointGrid:vec3f)->vec4f{return locateFinalPointVectorMeasured(initialAnchor,pointGrid).value;}
@compute @workgroup_size(64)fn interpolatePowerFaceVector(@builtin(global_invocation_id)g:vec3u){let index=g.x;if(index>=powerPublication.faceCount||atomicLoad(&powerPublication.flags)!=0u||index>=arrayLength(&powerFaceCentroids)||index>=arrayLength(&powerVelocityScratch)){return;}let mapping=powerVelocityScratch[index];let negativeBand=mapping.z;let positiveBand=mapping.w;if(negativeBand==INVALID&&positiveBand==INVALID){return;}
  if(bandRowIsWet(negativeBand)||bandRowIsWet(positiveBand)){powerVelocityScratch[index]=vec4u(0u,0u,0u,2u);return;}
  let centroid=powerFaceCentroids[index].xyz;let h=fp.fineWidth*f32(fp.fineFactor);if(!finite(centroid.x)||!finite(centroid.y)||!finite(centroid.z)||!finite(h)||h<=0.){powerFail(POWER_NONFINITE,index);return;}let pointGrid=centroid/h;for(var axis=0u;axis<3u;axis+=1u){let atNegative=abs(pointGrid[axis])<=1e-5;let atPositive=abs(pointGrid[axis]-f32(p.dims[axis]))<=1e-5;if(!atNegative&&!atPositive){continue;}let closedBit=select(positiveBoundaryBit(axis),negativeBoundaryBit(axis),atNegative);if((p.closedBoundaryMask&closedBit)!=0u){powerVelocityScratch[index]=vec4u(0u,0u,0u,1u);return;}let band=select(positiveBand,negativeBand,atPositive);if(!atPositive||band==INVALID||band>=arrayLength(&rows)||!finite(rows[band].padf)){powerFail(POWER_MISSING_ROW,index);return;}var boundaryVector=vec3f(0.);boundaryVector[axis]=rows[band].padf;powerVelocityScratch[index]=vec4u(bitcast<u32>(boundaryVector.x),bitcast<u32>(boundaryVector.y),bitcast<u32>(boundaryVector.z),1u);return;}let negative=locateFinalPointVector(negativeBand,pointGrid);let positive=locateFinalPointVector(positiveBand,pointGrid);var full=vec4f(0);if(velocityValid(negative)&&velocityValid(positive)){full=vec4f(.5*(negative.xyz+positive.xyz),1.);}else if(velocityValid(negative)){full=negative;}else if(velocityValid(positive)){full=positive;}else{powerFail(POWER_MISSING_ROW,index);return;}powerVelocityScratch[index]=vec4u(bitcast<u32>(full.x),bitcast<u32>(full.y),bitcast<u32>(full.z),1u);}
@compute @workgroup_size(64)fn projectPowerFaceVelocity(@builtin(global_invocation_id)g:vec3u){let index=g.x;if(index>=powerPublication.faceCount||atomicLoad(&powerPublication.flags)!=0u||index>=arrayLength(&powerFaces)||index>=arrayLength(&powerFaceNormals)||index>=arrayLength(&powerVelocityScratch)){return;}let candidate=powerVelocityScratch[index];
  // Sentinel 2 marks a liquid/interface face whose projected velocity must
  // survive; it completes the publication tally without becoming committable.
  if(candidate.w==2u){powerVelocityScratch[index]=vec4u(0u,3u,0u,0u);atomicAdd(&powerPublication.interpolatedCount,1u);return;}
  if(candidate.w!=1u){return;}let powerFace=powerFaces[index];if((powerFace.flags&VALID)==0u||powerFace.negativeRow>=p.powerRowCapacity||(powerFace.positiveRow!=INVALID&&powerFace.positiveRow>=p.powerRowCapacity)||!finite(powerFace.area)||powerFace.area<=0.){powerFail(POWER_FACE,index);return;}let normal=powerFaceNormals[index].xyz;let n2=dot(normal,normal);if(!finite(normal.x)||!finite(normal.y)||!finite(normal.z)||!finite(n2)||abs(n2-1.)>4e-4){powerFail(POWER_NORMAL,index);return;}let full=vec3f(bitcast<f32>(candidate.x),bitcast<f32>(candidate.y),bitcast<f32>(candidate.z));let value=dot(full,normal);if(!finite(value)){powerFail(POWER_NONFINITE,index);return;}powerVelocityScratch[index]=vec4u(bitcast<u32>(value),1u,0u,0u);atomicAdd(&powerPublication.interpolatedCount,1u);}
@compute @workgroup_size(1)fn publishPowerFaceVelocity(){let targets=atomicLoad(&powerPublication.targetCount);if(atomicLoad(&control.valid)==VALID&&atomicLoad(&powerPublication.flags)==0u&&targets>0u&&atomicLoad(&powerPublication.interpolatedCount)==targets){atomicStore(&powerPublication.valid,VALID);}else{if(targets==0u&&powerPublication.faceCount>0u){powerFail(POWER_INCOMPLETE,0u);}atomicStore(&powerPublication.valid,0u);}}
@compute @workgroup_size(64)fn commitPowerFaceVelocity(@builtin(global_invocation_id)g:vec3u){let index=g.x;if(atomicLoad(&powerPublication.valid)!=VALID||index>=powerPublication.faceCount||index>=arrayLength(&powerFaces)||index>=arrayLength(&powerVelocityScratch)){return;}let candidate=powerVelocityScratch[index];if(candidate.y==1u){powerFaces[index].normalVelocity=bitcast<f32>(candidate.x);atomicAdd(&powerPublication.committedCount,1u);}else if(candidate.y==3u){atomicAdd(&powerPublication.committedCount,1u);}}
fn sampleBandRow(cellKey:u32)->u32{let start=hash(cellKey+1u)&(sp.rowHashCapacity-1u);for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(sp.rowHashCapacity-1u);let observed=atomicLoad(&rowHash[slot*2u]);if(observed==0u){return INVALID;}if(observed==cellKey+1u){let encoded=atomicLoad(&rowHash[slot*2u+1u]);return select(INVALID,encoded-1u,encoded!=0u&&encoded!=INVALID);}}return INVALID;}
fn retainedBandAnchor(pointGrid:vec3f)->u32{if(any(pointGrid<vec3f(0))||any(pointGrid>=vec3f(sp.dims))){return INVALID;}let q=vec3u(floor(pointGrid));var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);let band=sampleBandRow(cell(origin));if(band!=INVALID&&band<sp.rowCapacity&&band<arrayLength(&rows)){let row=rows[band];if(row.cell==cell(origin)&&row.size==size&&(row.flags&ROW_SUPPORT3_ENDPOINT)==0u){return band;}}if(size>=sp.maximumLeaf){break;}size<<=1u;}return INVALID;}
// Section 5 first extrapolates velocity on regular octree faces, then reuses
// the standard per-axis staggered interpolant away from level transitions.
// Query the retained marched face carriers directly: reconstructing cell
// centres first would insert an avoidable face->cell->face low-pass filter.
fn retainedBandVector(world:vec3f)->vec4f{let pointGrid=world/sp.cellSize;let anchor=retainedBandAnchor(pointGrid);if(anchor==INVALID){return vec4f(bitcast<f32>(0x1ffffu),0.,0.,-20.);}let value=locateFinalPointVector(anchor,pointGrid);if(velocityValid(value)){return value;}return vec4f(bitcast<f32>(anchor),value.yz,value.w);}
// A generalized face centroid is shared by its liquid and air interpolation
// elements.  The integer owner directory is half-open, so an exact dual-face
// point may select the air support row even though the continuous Section 5
// field includes the incident liquid element.  Bias only that measure-zero
// case toward the face's negative (incident) cell; points genuinely in air
// still have to resolve through the fast-marched band.
fn retainedBandIncidentVector(world:vec3f,normal:vec3f)->vec4f{let direct=retainedBandVector(world);if(velocityValid(direct)){return direct;}let incident=retainedBandVector(world-sp.cellSize*1e-4*normal);return select(direct,incident,velocityValid(incident));}
fn oldAdvectionFail(index:u32,stage:u32,detail:u32){let reason=detail&255u;atomicOr(&oldAdvectionControl[0],8u);atomicMin(&oldAdvectionControl[1],index);atomicMin(&oldAdvectionControl[13],index*4u+stage);atomicMin(&oldAdvectionControl[14],index*16u+min(reason,15u));if(atomicLoad(&oldAdvectionControl[15])==INVALID){atomicStore(&oldAdvectionControl[15],detail);}atomicOr(&oldAdvectionSeed[0],8u);atomicMin(&oldAdvectionSeed[1],index);atomicStore(&oldAdvectionSeed[6],0u);}
@compute @workgroup_size(1)fn publishColdPowerFaceAdvection(){if(atomicLoad(&oldAdvectionControl[0])!=0u){return;}let requested=atomicLoad(&oldAdvectionControl[8]);if(requested==0u||requested>arrayLength(&powerFaces)){oldAdvectionFail(0u,1u,24u);return;}for(var i=0u;i<requested;i+=1u){var face=powerFaces[i];if((face.flags&POWER_FACE_VALID)==0u){oldAdvectionFail(i,1u,25u);return;}face.normalVelocity=0.;powerFaces[i]=face;}atomicStore(&oldAdvectionControl[3],requested);atomicStore(&oldAdvectionControl[6],VALID);atomicStore(&oldAdvectionSeed[6],VALID);atomicStore(&oldAdvectionControl[9],0u);atomicStore(&oldAdvectionControl[10],INVALID);atomicStore(&oldAdvectionControl[11],requested);atomicStore(&oldAdvectionControl[12],VALID);}
@compute @workgroup_size(1)fn preparePowerFaceAdvectionBandRepair(){atomicStore(&oldAdvectionControl[9],atomicLoad(&oldAdvectionControl[0]));atomicStore(&oldAdvectionControl[10],atomicLoad(&oldAdvectionControl[1]));atomicStore(&oldAdvectionControl[11],atomicLoad(&oldAdvectionControl[3]));atomicStore(&oldAdvectionControl[12],atomicLoad(&oldAdvectionControl[6]));atomicStore(&oldAdvectionControl[0],0u);atomicStore(&oldAdvectionControl[1],INVALID);atomicStore(&oldAdvectionControl[3],0u);atomicStore(&oldAdvectionControl[6],0u);atomicStore(&oldAdvectionControl[13],INVALID);atomicStore(&oldAdvectionControl[14],INVALID);atomicStore(&oldAdvectionControl[15],INVALID);atomicStore(&oldAdvectionSeed[0],0u);atomicStore(&oldAdvectionSeed[1],INVALID);atomicStore(&oldAdvectionSeed[6],0u);let requested=atomicLoad(&oldAdvectionControl[8]);if(sp.count!=requested||arrayLength(&powerFaceControl)<9u||sp.count!=powerFaceControl[1]){oldAdvectionFail(0u,1u,(sp.count<<8u)|27u);return;}if(atomicLoad(&control.valid)!=VALID||control.generation!=sp.fineGeneration||atomicLoad(&pointControl.valid)!=VALID||atomicLoad(&pointControl.flags)!=0u||pointControl.generation!=sp.fineGeneration||powerFaceControl[8]!=VALID||powerFaceControl[7]!=sp.p2){oldAdvectionFail(0u,1u,21u);}}
fn storeBandRepairFailure(index:u32,stage:u32,value:vec4f){let reason=u32(max(1.,-value.w));let anchor=bitcast<u32>(value.x);powerFaceCentroids[index].w=bitcast<f32>((min(anchor,0x1ffffu)<<10u)|((stage&3u)<<8u)|min(reason,255u));}
@compute @workgroup_size(64)fn repairPowerFaceAdvectionFromBand(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=sp.count||i>=arrayLength(&powerFaces)||i>=arrayLength(&powerFaceNormals)||i>=arrayLength(&powerFaceCentroids)){return;}let priorStatus=bitcast<u32>(powerFaceCentroids[i].w);if(priorStatus==STATUS_VALID){return;}var face=powerFaces[i];if((face.flags&POWER_FACE_VALID)==0u){powerFaceCentroids[i].w=bitcast<f32>((1u<<8u)|25u);return;}if(face.positiveRow==INVALID&&(face.flags&POWER_FACE_BOUNDARY)!=0u&&(face.flags&POWER_FACE_OPEN_BOUNDARY)==0u){face.normalVelocity=0.;powerFaces[i]=face;powerFaceCentroids[i].w=bitcast<f32>(STATUS_VALID);return;}let n=powerFaceNormals[i].xyz;let x=powerFaceCentroids[i].xyz;let v0=retainedBandIncidentVector(x,n);if(!velocityValid(v0)){storeBandRepairFailure(i,1u,v0);return;}let dt=bitcast<f32>(sp.p1);let vm=retainedBandIncidentVector(x-.5*dt*v0.xyz,n);if(!velocityValid(vm)){storeBandRepairFailure(i,2u,vm);return;}let va=retainedBandIncidentVector(x-dt*vm.xyz,n);if(!velocityValid(va)){storeBandRepairFailure(i,3u,va);return;}let value=dot(va.xyz,n);if(!finite(value)){powerFaceCentroids[i].w=bitcast<f32>((3u<<8u)|22u);return;}face.normalVelocity=value;powerFaces[i]=face;powerFaceCentroids[i].w=bitcast<f32>(STATUS_VALID);}
@compute @workgroup_size(1)fn finalizePowerFaceAdvectionFromBand(){let requestedFaces=atomicLoad(&oldAdvectionControl[8]);var completed=0u;for(var i=0u;i<requestedFaces&&i<arrayLength(&powerFaceCentroids);i+=1u){let status=bitcast<u32>(powerFaceCentroids[i].w);if(status==STATUS_VALID){completed+=1u;}else if(status!=0u){let stage=(status>>8u)&3u;let detail=((status>>10u)<<8u)|(status&255u);oldAdvectionFail(i,stage,detail);}else{oldAdvectionFail(i,1u,(i<<8u)|26u);}}atomicStore(&oldAdvectionControl[3],completed);if(atomicLoad(&oldAdvectionControl[0])==0u&&requestedFaces>0u&&completed==requestedFaces){atomicStore(&oldAdvectionControl[6],VALID);atomicStore(&oldAdvectionSeed[6],VALID);}else{if(atomicLoad(&oldAdvectionControl[0])==0u){oldAdvectionFail(0u,1u,23u);}atomicStore(&oldAdvectionControl[6],0u);atomicStore(&oldAdvectionSeed[6],0u);}}
fn loadSampleStatus(i:u32)->u32{return atomicLoad(&sampleStatus[i]);}fn storeSampleStatus(i:u32,value:u32){atomicStore(&sampleStatus[i],value);}fn airSearchCounterBase()->u32{return arrayLength(&sampleStatus)-8u;}fn addAirSearchCounter(slot:u32,value:u32){atomicAdd(&sampleStatus[airSearchCounterBase()+slot],value);}
// Section 5 extends characteristics constantly through solid container walls,
// just as transported phi uses a Neumann ghost there. The authored open
// ceiling also uses the limiting interior vector for outflow. A coordinate
// outside any genuinely open non-ceiling boundary remains fail-closed.
fn airSampleGrid(world:vec3f)->vec4f{var grid=world/sp.cellSize;if(!finite(grid.x)||!finite(grid.y)||!finite(grid.z)){return vec4f(0);}for(var axis=0u;axis<3u;axis+=1u){if(grid[axis]<0.){if((p.closedBoundaryMask&negativeBoundaryBit(axis))==0u){return vec4f(0);}grid[axis]=0.;}if(grid[axis]>=f32(sp.dims[axis])){let closed=(p.closedBoundaryMask&positiveBoundaryBit(axis))!=0u;let openCeiling=axis==1u&&!closed;if(!closed&&!openCeiling){return vec4f(0);}grid[axis]=f32(sp.dims[axis])-1e-5;}}return vec4f(grid,1.);}
@compute @workgroup_size(64)fn classifyAirBandVelocity(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=sp.count||i>=arrayLength(&positions)||i>=arrayLength(&sampleResults)||i>=arrayLength(&sampleStatus)||positions[i].w<=0.){return;}let sample=airSampleGrid(positions[i].xyz);if(sample.w==0.){storeSampleStatus(i,SAMPLE_FAILED|SAMPLE_FAIL_DOMAIN);return;}let grid=sample.xyz;let q=vec3u(floor(grid));let owner=ownerAt(q);if(owner.valid==0u){storeSampleStatus(i,SAMPLE_FAILED|SAMPLE_FAIL_OWNER);return;}
 // Section 5 extrapolates on adaptive octree faces. Band rows are keyed by
 // the containing owner's origin, not by every finest cell inside that owner.
 // Only the exact current owner row can prove that Stage B is sampling liquid.
 // Missing classification remains structural failure; definitely-positive air
 // requires a valid fast-marched regular-face vector.
 let band=retainedBandAnchor(grid);if(band==INVALID||band>=sp.rowCapacity||band>=transitionControl.support7NodeEnd||band>=arrayLength(&rows)){storeSampleStatus(i,SAMPLE_FAILED|SAMPLE_FAIL_ROW);return;}if(control.generation!=sp.fineGeneration||pointControl.generation!=sp.fineGeneration){storeSampleStatus(i,SAMPLE_FAILED|SAMPLE_FAIL_GENERATION);return;}if((rows[band].flags&ROW_PHI)==0u){storeSampleStatus(i,SAMPLE_FAILED|SAMPLE_FAIL_ROW);return;}let stageBStatus=loadSampleStatus(i);let stageBReason=stageBStatus&255u;let needsDualCompletion=(stageBStatus&VALID)==0u&&(stageBReason==4u||stageBReason==8u);if(rows[band].minimumPhi<0.&&!needsDualCompletion){return;}sampleResults[i].w=bitcast<f32>(band);storeSampleStatus(i,SAMPLE_EVALUATE);addAirSearchCounter(5u,1u);}
@compute @workgroup_size(64)fn evaluateAirBandVelocity(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=sp.count||i>=arrayLength(&positions)||i>=arrayLength(&sampleResults)||i>=arrayLength(&sampleStatus)||loadSampleStatus(i)!=SAMPLE_EVALUATE){return;}addAirSearchCounter(6u,1u);let band=bitcast<u32>(sampleResults[i].w);let sample=airSampleGrid(positions[i].xyz);if(sample.w==0.){sampleResults[i]=invalidPointVector(SAMPLE_FAIL_DOMAIN);storeSampleStatus(i,SAMPLE_FAILED|SAMPLE_FAIL_DOMAIN);return;}let measured=locateFinalPointVectorMeasured(band,sample.xyz);addAirSearchCounter(0u,measured.directSuccess);addAirSearchCounter(1u,measured.fullRowFallback);addAirSearchCounter(2u,measured.candidateRowsTested);addAirSearchCounter(3u,measured.surroundingFallback);addAirSearchCounter(4u,measured.surroundingRowsTested);let velocity=measured.value;if(!velocityValid(velocity)){let reason=u32(round(max(1.,-velocity.w)));sampleResults[i]=velocity;
  // QA payload only: bits 8..23 identify the exact retained Delaunay anchor.
  // The low byte remains the structural reason and SAMPLE_FAILED remains the
  // authority bit, so this cannot admit or alter a velocity sample.
  storeSampleStatus(i,SAMPLE_FAILED|((band&0xffffu)<<8u)|((16u+min(reason,11u))&255u));return;}sampleResults[i]=velocity;storeSampleStatus(i,SAMPLE_EVALUATED);}
@compute @workgroup_size(1)fn aggregateAirBandSearchCounters(){let base=airSearchCounterBase();atomicAdd(&control.directAnchorSuccess,atomicLoad(&sampleStatus[base]));atomicAdd(&control.fullRowFallbackInvocations,atomicLoad(&sampleStatus[base+1u]));atomicAdd(&control.fullRowCandidateRowsTested,atomicLoad(&sampleStatus[base+2u]));atomicAdd(&control.surroundingOwnerFallbackInvocations,atomicLoad(&sampleStatus[base+3u]));atomicAdd(&control.surroundingOwnerRowsTested,atomicLoad(&sampleStatus[base+4u]));atomicAdd(&control.airSamplesSelected,atomicLoad(&sampleStatus[base+5u]));atomicAdd(&control.airSamplesEvaluated,atomicLoad(&sampleStatus[base+6u]));}
@compute @workgroup_size(64)fn finalizeAirBandVelocity(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=sp.count||i>=arrayLength(&sampleStatus)){return;}let status=loadSampleStatus(i);if(status!=SAMPLE_EVALUATE&&status!=SAMPLE_EVALUATED&&(status&SAMPLE_FAILED)==0u){return;}if(status==SAMPLE_EVALUATED&&atomicLoad(&control.valid)==VALID&&control.generation==sp.fineGeneration&&atomicLoad(&pointControl.valid)==VALID&&atomicLoad(&pointControl.flags)==0u&&pointControl.generation==sp.fineGeneration){storeSampleStatus(i,VALID|EXTRAPOLATED);return;}storeSampleStatus(i,FACE_BAND_UNAVAILABLE|(status&0x00ffffffu));atomicAdd(&control.sampleFailures,1u);}
`;

/** Deterministic compact GPU construction of generalized power faces. */

import { OCTREE_POWER_FACE_RECORD_BYTES, OCTREE_POWER_INVALID_ROW } from "./octree-power-operator";
import { octreePowerCatalogWGSL } from "./octree-power-wgsl";
import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import {
  OCTREE_POWER_TOPOLOGY_VALID,
  type OctreePowerTopologySource,
} from "./webgpu-octree-power-topology";
import {
  makeFineLevelSetSortedWorklistLookupWGSL,
  type WebGPUFineLevelSetBrickSource,
} from "./webgpu-octree-fine-levelset-bricks";
import { makeOctreePowerCoarseLevelSetSampleWGSL } from "./webgpu-octree-power-coarse-levelset";
import { PassBroker } from "./webgpu-pass-broker";
import {
  OCTREE_POWER_ROW_DELTA_VALID,
  assertOctreePowerRowDeltaLayout,
  type OctreePowerRowDeltaSource,
} from "./webgpu-octree-power-descriptor";

export const OCTREE_POWER_FACE_INCIDENCE_BYTES = 8;
export const OCTREE_POWER_FACE_CONTROL_BYTES = 64;
export const OCTREE_POWER_FACE_PARAMETER_BYTES = 112;
export const OCTREE_POWER_FACE_BOUNDARY_QUERY_BYTES = 32;
export const OCTREE_POWER_FACE_EMPTY_FINE_BINDINGS_BYTES = 160;
/** Byte offset of the GPU-authored `(ceil(liveRows / 64), 1, 1)` record. */
export const OCTREE_POWER_FACE_LIVE_ROW_DISPATCH_OFFSET_BYTES = 24;
export const OCTREE_POWER_FACE_QUADRATURE_SAMPLES = 16;
const OCTREE_POWER_FACE_ROW_MAX = OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence;
// Exact centroid plus sixteen packed f16 tangent-plane coordinates. The
// samples are equal-area strata of the clipped world-space power polygon.
export const OCTREE_POWER_FACE_QUADRATURE_BYTES = 16 + OCTREE_POWER_FACE_QUADRATURE_SAMPLES * 4;
export const OCTREE_POWER_FACE_VALID = 0x8000_0000;
export const OCTREE_POWER_FACE_BOUNDARY = 1;
export const OCTREE_POWER_FACE_OPEN_BOUNDARY = 1 << 1;
export const OCTREE_POWER_FACE_WORLD_BOUNDARY_SHIFT = 8;

/**
 * Power-face world-boundary bits follow the catalog/descriptor order
 * `x-, y-, z-, z+, y+, x+`.  All container walls except the ceiling are
 * currently closed by the scene model.
 */
export const OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_OPEN_TOP = 47;
export const OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_CLOSED_TOP = 63;

export function octreePowerClosedBoundaryMask(closedTop: boolean): number {
  return OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_OPEN_TOP | (closedTop ? 16 : 0);
}

/**
 * Transient face-record mark: fine boundary sampling deferred this face to
 * the coarse-octree resolve pass.  Set and cleared inside one face-build
 * encoder; never present on a published face.
 */
export const OCTREE_POWER_FACE_COARSE_PENDING = 1 << 30;
/**
 * Transient exact-delta mark. The face builder sets it only when the sorted
 * old/new identity merge proves that a current face is the same degree of
 * freedom as its predecessor. Section 5 advection consumes and clears it
 * before any downstream face publication.
 */
export const OCTREE_POWER_FACE_DELTA_CARRIED = 1 << 28;

export const OCTREE_POWER_FACE_ERROR = Object.freeze({
  invalidMetric: 1 << 0,
  invalidHeader: 1 << 1,
  invalidCatalog: 1 << 2,
  invalidGeometry: 1 << 3,
  asymmetricFace: 1 << 4,
  capacity: 1 << 5,
  rowDirectory: 1 << 6,
} as const);

export interface OctreePowerFacePlan {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly incidenceCapacity: number;
  readonly faceBytes: number;
  readonly normalBytes: number;
  readonly centroidBytes: number;
  readonly quadratureBytes: number;
  readonly incidenceBytes: number;
  readonly workspaceBytes: number;
  /** One canonical `(level, Morton) -> row` record per possible row. */
  readonly rowDirectoryCapacity: number;
  readonly rowDirectoryBytes: number;
  /** One deterministic fail-closed diagnostic record per possible row. */
  readonly rowDiagnosticBytes: number;
  /** Exact affected-face staging; one record per possible public face. */
  readonly deltaFaceBytes: number;
  readonly boundaryQueryBytes: number;
  readonly liveFaceDispatchBytes: number;
  /** Four copied indirect records; this buffer is never storage-bound. */
  readonly workDispatchBytes: number;
  readonly allocatedBytes: number;
}

const OCTREE_POWER_FACE_DISPATCH_BYTES = 4 * 12;

function powerFacePublicationArenaBytes(rowCapacity: number): number {
  const blocks = Math.ceil(rowCapacity / 256);
  // Four indirect records followed by exact per-block
  // {faceTotal, incidenceTotal, faceBase, incidenceBase}.
  return (12 + 4 * blocks) * 4;
}

export interface OctreePowerFaceEncodeOptions {
  readonly dimensions: readonly [number, number, number];
  /** COPY_SRC GPU buffer whose first u32 is the live compact-row count. */
  readonly rowCount: GPUBuffer;
  /** Physical width of one finest-grid cell. The topology stage requires it to be isotropic. */
  readonly physicalCellSize?: number;
  readonly generation?: number;
  /** Exact old/new row mapping and dirty plus one-ring work publication. */
  readonly rowDelta: OctreePowerRowDeltaSource;
  /** Closed world faces in `x-, y-, z-, z+, y+, x+` bit order. */
  readonly closedBoundaryMask?: number;
  /**
   * Signed-distance authority for internal liquid/air pressure faces.  The
   * analytic mode is the authored t=0 field; fine mode is the current
   * published two-sided Section 5 band.  No estimate is substituted when a
   * requested sample is unavailable.
   */
  readonly boundaryPhi?: {
    readonly mode: "analytic" | "fine" | "coarse";
    /** Fine storage is absent in coarse-only mode. The face builder binds
     * inert private buffers while the coarse resolve pass owns every query. */
    readonly fine?: Pick<WebGPUFineLevelSetBrickSource, "params" | "metadata" | "worklist" | "flags" | "phi">;
    /**
     * Published coarse-octree signed-distance directory (paper Section 5).
     * Outside the resident fine band the coarse level set is the cell-centre
     * authority, so fine mode falls back to it instead of failing the
     * generation when a query centre has no fine sample.
     */
    readonly coarse?: { readonly directory: GPUBuffer };
    readonly container: readonly [number, number, number];
    readonly fillFraction: number;
    readonly initialCondition: "dam-break" | "tank-fill";
  };
}

export interface OctreePowerFaceControl {
  readonly rowCount: number;
  readonly faceCount: number;
  readonly incidenceCount: number;
  readonly flags: number;
  readonly firstInvalid: number;
  readonly invalidCount: number;
  readonly boundaryCount: number;
  readonly generation: number;
  readonly valid: number;
  readonly lookupMissCount: number;
  readonly maximumLookupSteps: number;
  readonly worldBoundaryCount: number;
  readonly firstInvalidSlot: number;
  readonly firstInvalidNeighbor: number;
  readonly firstInvalidDetail: number;
  readonly firstInvalidRow: number;
}

export interface OctreePowerFaceSource {
  readonly plan: OctreePowerFacePlan;
  readonly faces: GPUBuffer;
  /** Transient unit normals, oriented from negativeRow to positiveRow. */
  readonly faceNormals: GPUBuffer;
  /** Physical centroid xyz and zero padding for each public face. */
  readonly faceCentroids: GPUBuffer;
  /**
   * Exact centroid and bounded equal-area samples of the actual clipped power
   * polygon. This SDF integration rule is an engineering extension consistent
   * with the paper's Eqs. (3)-(4), not a quadrature prescribed by the paper.
   */
  readonly faceQuadrature: GPUBuffer;
  /** RowWork records; the CSR incidence offset is the fourth u32 of each 16-byte row. */
  readonly incidenceRows: GPUBuffer;
  /** Compact `(face:u32, sign:i32)` pairs. */
  readonly incidence: GPUBuffer;
  readonly control: GPUBuffer;
  readonly rowDirectory: GPUBuffer;
  /** Exact `{carried, added, retired, valid}` face-identity delta. */
  readonly deltaStats: GPUBuffer;
  /**
   * Device-authored indirect-only arena. Its first record covers live faces;
   * `OCTREE_POWER_FACE_LIVE_ROW_DISPATCH_OFFSET_BYTES` covers live rows.
   */
  readonly liveFaceDispatch: GPUBuffer;
  /**
   * Exact liquid/air cell-centre pairs for internal open power faces.  These
   * drive an implementation support rule motivated by Section 5's wide-band
   * requirement and Section 4.1's cell-centre phi consumers: the next fine
   * generation keeps every trilinear contributor needed at both endpoints
   * resident. The paper does not prescribe this exact seeding mechanism.
   */
  readonly boundaryPhiQueries: GPUBuffer;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

export function planOctreePowerFaces(
  rowCapacityValue: number,
  faceCapacityValue: number,
  incidenceCapacityValue = faceCapacityValue * 2,
): OctreePowerFacePlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Power-face row capacity");
  const faceCapacity = positiveInteger(faceCapacityValue, "Power-face capacity");
  const incidenceCapacity = positiveInteger(incidenceCapacityValue, "Power-face incidence capacity");
  if (incidenceCapacity > faceCapacity * 2) {
    throw new RangeError("Power-face incidence capacity may not exceed two incidences per physical face");
  }
  const faceBytes = faceCapacity * OCTREE_POWER_FACE_RECORD_BYTES;
  const normalBytes = faceCapacity * 16;
  const centroidBytes = faceCapacity * 16;
  const quadratureBytes = faceCapacity * OCTREE_POWER_FACE_QUADRATURE_BYTES;
  const incidenceBytes = incidenceCapacity * OCTREE_POWER_FACE_INCIDENCE_BYTES;
  // Four u32 values per live row, plus one terminal offset row.
  const workspaceBytes = (rowCapacity + 1) * 16;
  const rowDirectoryCapacity = rowCapacity;
  const rowDirectoryBytes = rowDirectoryCapacity * 16;
  const rowDiagnosticBytes = rowCapacity * 16;
  const deltaFaceBytes = faceBytes;
  const deltaOldFaceBytes = faceBytes;
  const deltaOldSourceBytes = faceCapacity * 4;
  const deltaCandidateSourceBytes = faceCapacity * 4;
  const boundaryQueryBytes = faceCapacity * OCTREE_POWER_FACE_BOUNDARY_QUERY_BYTES;
  const liveFaceDispatchBytes = powerFacePublicationArenaBytes(rowCapacity);
  const workDispatchBytes = OCTREE_POWER_FACE_DISPATCH_BYTES;
  return {
    rowCapacity,
    faceCapacity,
    incidenceCapacity,
    faceBytes, normalBytes, centroidBytes, quadratureBytes,
    incidenceBytes,
    workspaceBytes, boundaryQueryBytes, liveFaceDispatchBytes, workDispatchBytes,
    rowDirectoryCapacity,
    rowDirectoryBytes,
    rowDiagnosticBytes,
    deltaFaceBytes,
    allocatedBytes: 2 * (faceBytes + normalBytes + centroidBytes + quadratureBytes + incidenceBytes + workspaceBytes
      + boundaryQueryBytes + OCTREE_POWER_FACE_CONTROL_BYTES)
      + 2 * rowDirectoryBytes + rowDiagnosticBytes + deltaFaceBytes + deltaOldFaceBytes
      + deltaOldSourceBytes + deltaCandidateSourceBytes
      + liveFaceDispatchBytes + workDispatchBytes + OCTREE_POWER_FACE_PARAMETER_BYTES + 16 + 12
      + OCTREE_POWER_FACE_EMPTY_FINE_BINDINGS_BYTES,
  };
}

export function unpackOctreePowerFaceControl(words: ArrayLike<number>): OctreePowerFaceControl {
  if (words.length < 12) throw new RangeError("Power-face control needs at least twelve words");
  return {
    rowCount: Number(words[0]) >>> 0,
    faceCount: Number(words[1]) >>> 0,
    incidenceCount: Number(words[2]) >>> 0,
    flags: Number(words[3]) >>> 0,
    firstInvalid: Number(words[4]) >>> 0,
    invalidCount: Number(words[5]) >>> 0,
    boundaryCount: Number(words[6]) >>> 0,
    generation: Number(words[7]) >>> 0,
    valid: Number(words[8]) >>> 0,
    lookupMissCount: Number(words[9]) >>> 0,
    maximumLookupSteps: Number(words[10]) >>> 0,
    worldBoundaryCount: Number(words[11]) >>> 0,
    firstInvalidSlot: Number(words[12]) >>> 0,
    firstInvalidNeighbor: Number(words[13]) >>> 0,
    firstInvalidDetail: Number(words[14]) >>> 0,
    firstInvalidRow: Number(words[15]) >>> 0,
  };
}

export interface OctreePowerFaceIdentity {
  readonly negativeRow: number;
  readonly positiveRow: number;
  readonly geometryCode: number;
}

export interface OctreePowerFaceIdentityDelta {
  readonly carried: number;
  readonly added: number;
  readonly retired: number;
}

function compareOctreePowerFaceIdentity(a: OctreePowerFaceIdentity, b: OctreePowerFaceIdentity): number {
  return a.negativeRow - b.negativeRow
    || a.geometryCode - b.geometryCode
    || a.positiveRow - b.positiveRow;
}

/**
 * CPU oracle for the GPU's exact sorted face-identity merge. Previous row IDs
 * are first translated through the frontier's old-to-new publication; a face
 * whose owner or internal neighbour retired is itself retired.
 */
export function mergeOctreePowerFaceIdentities(
  previous: readonly OctreePowerFaceIdentity[],
  current: readonly OctreePowerFaceIdentity[],
  oldToNew: readonly number[],
): OctreePowerFaceIdentityDelta {
  for (let index = 1; index < current.length; index += 1) {
    if (compareOctreePowerFaceIdentity(current[index - 1], current[index]) >= 0) {
      throw new RangeError("Current power-face identities must be strictly sorted");
    }
  }
  const remapped: OctreePowerFaceIdentity[] = [];
  let retired = 0;
  for (const face of previous) {
    const negativeRow = oldToNew[face.negativeRow] ?? -1;
    const positiveRow = face.positiveRow === OCTREE_POWER_INVALID_ROW
      ? OCTREE_POWER_INVALID_ROW
      : oldToNew[face.positiveRow] ?? -1;
    if (negativeRow < 0 || positiveRow < 0) {
      retired += 1;
      continue;
    }
    remapped.push({ negativeRow, positiveRow, geometryCode: face.geometryCode >>> 0 });
  }
  for (let index = 1; index < remapped.length; index += 1) {
    if (compareOctreePowerFaceIdentity(remapped[index - 1], remapped[index]) >= 0) {
      throw new RangeError("Remapped previous power-face identities must be strictly sorted");
    }
  }
  let previousIndex = 0;
  let currentIndex = 0;
  let carried = 0;
  let added = 0;
  while (previousIndex < remapped.length && currentIndex < current.length) {
    const comparison = compareOctreePowerFaceIdentity(remapped[previousIndex], current[currentIndex]);
    if (comparison === 0) {
      carried += 1; previousIndex += 1; currentIndex += 1;
    } else if (comparison < 0) {
      retired += 1; previousIndex += 1;
    } else {
      added += 1; currentIndex += 1;
    }
  }
  retired += remapped.length - previousIndex;
  added += current.length - currentIndex;
  if (current.length !== carried + added || current.length + retired !== previous.length + added) {
    throw new Error("Power-face identity delta failed its exact count identities");
  }
  return { carried, added, retired };
}

/**
 * Standalone transactional builder. It consumes resolved PowerRowMetric records,
 * never descriptor words, so descriptor canonicalization remains owned by WP3.
 * Public face IDs are a row-major prefix sum followed by catalog-local order.
 * The canonical leaf ordering supplies an immutable binary-search directory;
 * fixed row records and deterministic reductions replace every allocator and
 * synchronization atomic.
 */
export class WebGPUOctreePowerFaces {
  readonly plan: OctreePowerFacePlan;
  readonly faces: GPUBuffer;
  private readonly candidateFaces: GPUBuffer;
  readonly faceNormals: GPUBuffer;
  private readonly candidateFaceNormals: GPUBuffer;
  readonly faceCentroids: GPUBuffer;
  private readonly candidateFaceCentroids: GPUBuffer;
  readonly faceQuadrature: GPUBuffer;
  private readonly candidateFaceQuadrature: GPUBuffer;
  readonly incidence: GPUBuffer;
  private readonly candidateIncidence: GPUBuffer;
  readonly control: GPUBuffer;
  private readonly candidateControl: GPUBuffer;
  private readonly deltaStats: GPUBuffer;
  readonly rowDirectory: GPUBuffer;
  private readonly candidateRowDirectory: GPUBuffer;
  private readonly rowDiagnostics: GPUBuffer;
  private readonly workspace: GPUBuffer;
  private readonly candidateRows: GPUBuffer;
  private readonly boundaryQueries: GPUBuffer;
  private readonly candidateBoundaryQueries: GPUBuffer;
  private readonly deltaFaces: GPUBuffer;
  private readonly deltaOldFaces: GPUBuffer;
  private readonly deltaSources: GPUBuffer;
  private readonly liveFaceDispatch: GPUBuffer;
  private readonly workDispatch: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly buildAffectedRowsPipeline: GPUComputePipeline;
  private readonly compactIdentityDeltaPipeline: GPUComputePipeline;
  private readonly mergeIdentityDeltaPipeline: GPUComputePipeline;
  private readonly prepareCandidateCSRPipeline: GPUComputePipeline;
  private readonly scanCandidateCSRPipeline: GPUComputePipeline;
  private readonly prefixCandidateCSRPipeline: GPUComputePipeline;
  private readonly offsetCandidateCSRPipeline: GPUComputePipeline;
  private readonly finalizeCandidateCSRPipeline: GPUComputePipeline;
  private readonly publishCandidateCSRPipeline: GPUComputePipeline;
  private readonly carryCandidateCSRPipeline: GPUComputePipeline;
  private readonly carryPayloadPipeline: GPUComputePipeline;
  private readonly geometryPipeline: GPUComputePipeline;
  private readonly boundaryPhiPipeline: GPUComputePipeline;
  private readonly boundaryCoarsePipeline: GPUComputePipeline;
  private readonly boundaryPhiEmptyParams: GPUBuffer;
  private readonly boundaryPhiEmptyStorage: GPUBuffer;
  private readonly finalizePublishPipeline: GPUComputePipeline;
  private readonly commitPipeline: GPUComputePipeline;
  private readonly commitIncidencePipeline: GPUComputePipeline;
  private readonly commitControlPipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    rowCapacity: number,
    faceCapacity: number,
    private readonly topology: OctreePowerTopologySource,
    incidenceCapacity = faceCapacity * 2,
  ) {
    this.plan = planOctreePowerFaces(rowCapacity, faceCapacity, incidenceCapacity);
    if (rowCapacity > topology.plan.rowCapacity) {
      throw new RangeError("Power-face row capacity exceeds the topology metric capacity");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.faces = device.createBuffer({ label: "Octree power faces", size: this.plan.faceBytes, usage: storage });
    this.candidateFaces = device.createBuffer({ label: "Octree power candidate faces", size: this.plan.faceBytes, usage: storage });
    this.faceNormals = device.createBuffer({ label: "Octree power face normals", size: this.plan.normalBytes, usage: storage });
    this.candidateFaceNormals = device.createBuffer({ label: "Octree power candidate face normals", size: this.plan.normalBytes, usage: storage });
    this.faceCentroids = device.createBuffer({ label: "Octree power face centroids", size: this.plan.centroidBytes, usage: storage });
    this.candidateFaceCentroids = device.createBuffer({ label: "Octree power candidate face centroids", size: this.plan.centroidBytes, usage: storage });
    this.faceQuadrature = device.createBuffer({ label: "Octree power face polygon quadrature", size: this.plan.quadratureBytes, usage: storage });
    this.candidateFaceQuadrature = device.createBuffer({ label: "Octree power candidate face polygon quadrature", size: this.plan.quadratureBytes, usage: storage });
    this.incidence = device.createBuffer({ label: "Octree power face incidence", size: this.plan.incidenceBytes, usage: storage });
    this.candidateIncidence = device.createBuffer({ label: "Octree power candidate face incidence", size: this.plan.incidenceBytes, usage: storage });
    this.workspace = device.createBuffer({ label: "Octree power face row workspace", size: this.plan.workspaceBytes, usage: storage });
    this.candidateRows = device.createBuffer({ label: "Octree power candidate face row workspace", size: this.plan.workspaceBytes, usage: storage });
    this.rowDirectory = device.createBuffer({
      label: "Octree power committed sorted row directory", size: this.plan.rowDirectoryBytes, usage: storage,
    });
    this.candidateRowDirectory = device.createBuffer({
      label: "Octree power candidate sorted row directory", size: this.plan.rowDirectoryBytes, usage: storage,
    });
    this.rowDiagnostics = device.createBuffer({
      label: "Octree power face row diagnostics", size: this.plan.rowDiagnosticBytes, usage: storage,
    });
    this.boundaryQueries = device.createBuffer({ label: "Octree power face boundary phi queries", size: this.plan.boundaryQueryBytes, usage: storage });
    this.candidateBoundaryQueries = device.createBuffer({ label: "Octree power candidate face boundary phi queries", size: this.plan.boundaryQueryBytes, usage: storage });
    this.deltaFaces = device.createBuffer({
      label: "Octree power exact affected-face staging", size: this.plan.deltaFaceBytes, usage: storage,
    });
    this.deltaOldFaces = device.createBuffer({
      label: "Octree power compact remapped old-face staging", size: this.plan.deltaFaceBytes, usage: storage,
    });
    this.deltaSources = device.createBuffer({
      label: "Octree power compact old and exact candidate source indices",
      size: this.plan.faceCapacity * 8, usage: storage,
    });
    this.liveFaceDispatch = device.createBuffer({
      label: "Octree power live-face dispatch arena", size: this.plan.liveFaceDispatchBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.workDispatch = device.createBuffer({
      label: "Octree power indirect-only work dispatch", size: this.plan.workDispatchBytes,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    this.control = device.createBuffer({ label: "Octree power face control", size: OCTREE_POWER_FACE_CONTROL_BYTES, usage: storage });
    this.candidateControl = device.createBuffer({ label: "Octree power candidate face control", size: OCTREE_POWER_FACE_CONTROL_BYTES, usage: storage });
    this.deltaStats = device.createBuffer({ label: "Octree power face delta statistics", size: 16, usage: storage });
    this.params = device.createBuffer({ label: "Octree power face parameters", size: OCTREE_POWER_FACE_PARAMETER_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const shaderModule = device.createShaderModule({ label: "Octree power face construction", code: octreePowerFaceShader });
    const pipeline = (label: string, entryPoint: string) => device.createComputePipeline({
      label, layout: "auto", compute: { module: shaderModule, entryPoint },
    });
    this.preparePipeline = pipeline("Prepare octree power faces", "preparePowerFaces");
    this.buildAffectedRowsPipeline = pipeline("Build affected octree power-face rows in parallel", "buildAffectedPowerFaceRows");
    this.compactIdentityDeltaPipeline = pipeline("Compact octree power-face identity delta in parallel", "compactPowerFaceIdentityDelta");
    this.mergeIdentityDeltaPipeline = pipeline("Merge octree power-face identity delta in parallel", "mergePowerFaceIdentityDelta");
    this.prepareCandidateCSRPipeline = pipeline("Prepare octree power-face candidate CSR in parallel", "preparePowerFaceCandidateCSR");
    this.scanCandidateCSRPipeline = pipeline("Scan octree power-face candidate CSR blocks", "scanPowerFaceCandidateCSR");
    this.prefixCandidateCSRPipeline = pipeline("Prefix octree power-face candidate CSR blocks", "prefixPowerFaceCandidateCSR");
    this.offsetCandidateCSRPipeline = pipeline("Offset octree power-face candidate CSR rows", "offsetPowerFaceCandidateCSR");
    this.finalizeCandidateCSRPipeline = pipeline("Finalize octree power-face candidate CSR", "finalizePowerFaceCandidateCSR");
    this.publishCandidateCSRPipeline = pipeline("Publish affected octree power-face candidate CSR", "publishAffectedPowerFaceCandidateCSR");
    this.carryCandidateCSRPipeline = pipeline("Carry clean octree power-face candidate CSR", "publishCarriedPowerFaceCandidateCSR");
    this.carryPayloadPipeline = pipeline("Carry immutable octree power face payload", "carryPowerFacePayload");
    this.geometryPipeline = pipeline("Publish added octree power face geometry", "publishAddedPowerFaceGeometry");
    const boundaryPhiModule = device.createShaderModule({
      label: "Octree power boundary phi sampling",
      code: octreePowerBoundaryPhiShader,
    });
    this.boundaryPhiPipeline = device.createComputePipeline({
      label: "Sample octree power boundary phi", layout: "auto",
      compute: { module: boundaryPhiModule, entryPoint: "samplePowerBoundaryPhi" },
    });
    const boundaryCoarseModule = device.createShaderModule({
      label: "Octree power boundary coarse phi resolve",
      code: octreePowerBoundaryCoarsePhiShader,
    });
    this.boundaryCoarsePipeline = device.createComputePipeline({
      label: "Resolve octree power boundary phi from coarse octree", layout: "auto",
      compute: { module: boundaryCoarseModule, entryPoint: "resolveCoarsePowerBoundaryPhi" },
    });
    this.boundaryPhiEmptyParams = device.createBuffer({
      label: "Inert power boundary fine-phi parameters", size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.boundaryPhiEmptyStorage = device.createBuffer({
      label: "Inert power boundary fine-phi storage", size: 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.finalizePublishPipeline = pipeline("Finalize and publish exact octree power faces", "finalizeAndPublishPowerFaces");
    this.commitPipeline = pipeline("Commit octree power face geometry", "commitPowerFaceGeometry");
    this.commitIncidencePipeline = pipeline("Commit octree power face incidence", "commitPowerFaceRowsAndIncidence");
    this.commitControlPipeline = pipeline("Commit octree power face control", "commitPowerFaceControl");
  }

  private writeParams(broker: PassBroker, options: OctreePowerFaceEncodeOptions): number {
    if (this.destroyed) throw new Error("Octree power-face builder is destroyed");
    options.dimensions.forEach((value, axis) => positiveInteger(value, `Power-face dimension ${axis}`));
    const volume = options.dimensions[0] * options.dimensions[1] * options.dimensions[2];
    if (!Number.isSafeInteger(volume) || volume > 0xffff_ffff) throw new RangeError("Power-face dimensions must fit the u32 cell ABI");
    const generation = options.generation ?? 0;
    if (!Number.isSafeInteger(generation) || generation < 0 || generation > 0xffff_ffff) throw new RangeError("Power-face generation must be unsigned");
    const physicalCellSize = options.physicalCellSize ?? 1;
    if (!Number.isFinite(physicalCellSize) || physicalCellSize <= 0) throw new RangeError("Power-face physical cell size must be finite and positive");
    const closedBoundaryMask = options.closedBoundaryMask ?? OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_CLOSED_TOP;
    if (!Number.isSafeInteger(closedBoundaryMask) || closedBoundaryMask < 0 || closedBoundaryMask > 63) {
      throw new RangeError("Power-face closed-boundary mask must be a six-bit unsigned integer");
    }
    assertOctreePowerRowDeltaLayout(options.rowDelta, this.plan.rowCapacity, "Power face");
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([
      ...options.dimensions, this.plan.rowCapacity,
      this.plan.rowCapacity, this.plan.faceCapacity, this.plan.incidenceCapacity, generation,
    ]));
    this.device.queue.writeBuffer(this.params, 32, new Float32Array([
      physicalCellSize, 0, 0, 0,
    ]));
    this.device.queue.writeBuffer(this.params, 48, new Uint32Array([closedBoundaryMask, 0, 0, 0]));
    const boundaryPhi = options.boundaryPhi;
    const container = boundaryPhi?.container ?? [0, 0, 0];
    if (boundaryPhi && (!container.every((value) => Number.isFinite(value) && value > 0)
      || !Number.isFinite(boundaryPhi.fillFraction) || boundaryPhi.fillFraction < 0 || boundaryPhi.fillFraction > 1)) {
      throw new RangeError("Power-face boundary phi scene parameters are invalid");
    }
    this.device.queue.writeBuffer(this.params, 64, new Float32Array([
      ...container, boundaryPhi?.fillFraction ?? 0,
    ]));
    this.device.queue.writeBuffer(this.params, 80, new Uint32Array([
      boundaryPhi?.mode === "analytic" ? 1 : boundaryPhi?.mode === "fine" ? 2 : boundaryPhi?.mode === "coarse" ? 3 : 0,
      boundaryPhi?.initialCondition === "dam-break" ? 2 : boundaryPhi ? 1 : 0,
      (boundaryPhi?.mode === "fine" || boundaryPhi?.mode === "coarse") && boundaryPhi.coarse ? 1 : 0,
      options.rowDelta.dirtyRowsOffsetWords,
    ]));
    this.device.queue.writeBuffer(this.params, 96, new Uint32Array([
      options.rowDelta.controlOffsetWords,
      options.rowDelta.newToOldOffsetWords,
      options.rowDelta.oldToNewOffsetWords,
      options.rowDelta.affectedRowsOffsetWords,
    ]));
    broker.copyBufferToBuffer(options.rowCount, 0, this.params, 12, 4);
    return this.plan.rowCapacity;
  }

  encode(broker: PassBroker, leafHeaders: GPUBuffer, options: OctreePowerFaceEncodeOptions): void {
    this.writeParams(broker, options);
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const group = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries,
    });
    const run = (label: string, pipeline: GPUComputePipeline, groups: number, bindings: GPUBindGroupEntry[]) => {
      const pass = broker.compute({ label });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group(pipeline, bindings)); pass.dispatchWorkgroups(groups);
    };
    const params = { binding: 0, resource: resource(this.params) };
    const headers = { binding: 1, resource: resource(leafHeaders) };
    const metrics = { binding: 2, resource: resource(this.topology.metrics) };
    const entries = { binding: 3, resource: resource(this.topology.catalogEntryHeaders) };
    const catalog = { binding: 4, resource: resource(this.topology.catalogFaces) };
    const rows = { binding: 5, resource: resource(this.candidateRows) };
    const faces = { binding: 6, resource: resource(this.candidateFaces) };
    const incidence = { binding: 7, resource: resource(this.candidateIncidence) };
    const control = { binding: 8, resource: resource(this.candidateControl) };
    const directory = { binding: 9, resource: resource(this.candidateRowDirectory) };
    const diagnostics = { binding: 16, resource: resource(this.rowDiagnostics) };
    const normals = { binding: 11, resource: resource(this.candidateFaceNormals) };
    const centroids = { binding: 12, resource: resource(this.candidateFaceCentroids) };
    const quadrature = { binding: 13, resource: resource(this.candidateFaceQuadrature) };
    const boundaryQueries = { binding: 14, resource: resource(this.candidateBoundaryQueries) };
    const deltaFaceScratch = { binding: 30, resource: resource(this.deltaFaces) };
    const deltaOldFaceScratch = { binding: 27, resource: resource(this.deltaOldFaces) };
    const deltaSources = { binding: 28, resource: resource(this.deltaSources) };
    const committedDirectory = { binding: 31, resource: resource(this.rowDirectory) };
    const liveFaceDispatch = { binding: 15, resource: resource(this.liveFaceDispatch) };
    const rowDelta = { binding: 18, resource: resource(options.rowDelta.rows) };
    const committedRows = { binding: 19, resource: resource(this.workspace) };
    const committedIncidence = { binding: 20, resource: resource(this.incidence) };
    const committedNormals = { binding: 21, resource: resource(this.faceNormals) };
    const committedCentroids = { binding: 22, resource: resource(this.faceCentroids) };
    const committedQuadrature = { binding: 23, resource: resource(this.faceQuadrature) };
    const committedBoundaryQueries = { binding: 24, resource: resource(this.boundaryQueries) };
    const committedControl = { binding: 25, resource: resource(this.control) };
    const deltaStats = { binding: 26, resource: resource(this.deltaStats) };
    const committedFaces = { binding: 17, resource: resource(this.faces) };
    const runIndirectFaces = (label: string, pipeline: GPUComputePipeline, bindings: GPUBindGroupEntry[], offset = 0) => {
      const pass = broker.compute({ label });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group(pipeline, bindings));
      pass.dispatchWorkgroupsIndirect(this.workDispatch, offset);
    };
    run("Prepare exact affected power-face publication", this.preparePipeline, 1,
      [params, control, liveFaceDispatch, diagnostics, rowDelta, committedControl, deltaStats]);
    run("Build affected power-face rows in parallel", this.buildAffectedRowsPipeline, 1,
      [params, headers, metrics, entries, catalog, rows, control, rowDelta, diagnostics, deltaFaceScratch]);
    run("Compact immutable power-face identity delta in parallel", this.compactIdentityDeltaPipeline, 1,
      [params, control, diagnostics, committedFaces, rowDelta, committedControl, deltaStats,
        deltaOldFaceScratch, deltaSources, deltaFaceScratch]);
    run("Merge immutable power-face identity delta in parallel", this.mergeIdentityDeltaPipeline, 1,
      [params, faces, control, diagnostics, deltaStats, boundaryQueries,
        deltaOldFaceScratch, deltaSources, deltaFaceScratch]);
    broker.fence("power face CSR work dispatch publication");
    broker.updateIndirectBuffer(
      this.liveFaceDispatch,
      OCTREE_POWER_FACE_LIVE_ROW_DISPATCH_OFFSET_BYTES,
      this.workDispatch,
      OCTREE_POWER_FACE_LIVE_ROW_DISPATCH_OFFSET_BYTES,
      24,
    );
    runIndirectFaces("Prepare exact affected and carried power-face CSR rows", this.prepareCandidateCSRPipeline,
      [params, headers, metrics, entries, rows, control, directory, diagnostics, rowDelta], 24);
    runIndirectFaces("Scan power-face candidate CSR blocks", this.scanCandidateCSRPipeline,
      [params, rows, faces, control, diagnostics, liveFaceDispatch, rowDelta, committedRows, committedControl], 36);
    run("Prefix power-face candidate CSR blocks", this.prefixCandidateCSRPipeline, 1,
      [params, rows, control, liveFaceDispatch, diagnostics]);
    runIndirectFaces("Offset power-face candidate CSR rows", this.offsetCandidateCSRPipeline,
      [params, rows, control, liveFaceDispatch], 36);
    run("Finalize power-face candidate CSR", this.finalizeCandidateCSRPipeline, 1,
      [params, faces, control, diagnostics]);
    runIndirectFaces("Publish affected power-face candidate CSR incidence", this.publishCandidateCSRPipeline,
      [params, headers, metrics, entries, catalog, rows, faces, incidence, control, diagnostics, rowDelta], 24);
    runIndirectFaces("Carry clean power-face candidate CSR incidence", this.carryCandidateCSRPipeline,
      [params, rows, faces, incidence, control, diagnostics, rowDelta, committedRows,
        committedIncidence, committedFaces, committedControl], 24);
    broker.fence("power face exact identity publication");
    broker.updateIndirectBuffer(this.liveFaceDispatch, 0, this.workDispatch, 0, 12);
    {
      // The generation-95 failure established this as a semantic publication
      // fence: geometry consumers must not observe partially emitted faces.
      runIndirectFaces("Carry immutable power face payload", this.carryPayloadPipeline,
        [faces, control, normals, centroids, quadrature, boundaryQueries,
          committedNormals, committedCentroids, committedQuadrature, committedBoundaryQueries]);
      runIndirectFaces("Publish added power face geometry", this.geometryPipeline,
        [params, headers, metrics, entries, catalog, faces, control, normals, centroids, quadrature, boundaryQueries]);
      const boundaryPhi = options.boundaryPhi;
      if (boundaryPhi) {
        const fine = boundaryPhi.fine;
        runIndirectFaces("Sample deterministic power boundary phi", this.boundaryPhiPipeline, [
            { binding: 0, resource: resource(this.params) },
            { binding: 1, resource: resource(fine?.params ?? this.boundaryPhiEmptyParams) },
            { binding: 2, resource: resource(fine?.metadata ?? this.boundaryPhiEmptyStorage) },
            { binding: 3, resource: resource(fine?.worklist ?? this.boundaryPhiEmptyStorage) },
            { binding: 4, resource: resource(fine?.flags ?? this.boundaryPhiEmptyStorage) },
            { binding: 5, resource: resource(fine?.phi ?? this.boundaryPhiEmptyStorage) },
            { binding: 6, resource: resource(this.candidateFaces) },
            { binding: 7, resource: resource(this.candidateBoundaryQueries) },
            { binding: 8, resource: resource(this.candidateControl) },
          ]);
        if ((boundaryPhi.mode === "fine" || boundaryPhi.mode === "coarse") && boundaryPhi.coarse) {
          runIndirectFaces("Resolve power boundary phi from coarse octree", this.boundaryCoarsePipeline, [
              { binding: 0, resource: resource(this.candidateFaces) },
              { binding: 1, resource: resource(this.candidateBoundaryQueries) },
              { binding: 2, resource: resource(this.candidateControl) },
              { binding: 3, resource: resource(boundaryPhi.coarse.directory) },
            ]);
        }
      }
    }
    broker.fence("power face geometry publication");
    run("Finalize and publish exact power faces", this.finalizePublishPipeline, 1,
      [rows, faces, control, liveFaceDispatch, diagnostics, deltaStats]);
    broker.fence("power face candidate publication");
    broker.updateIndirectBuffer(this.liveFaceDispatch, 12, this.workDispatch, 12, 12);
    runIndirectFaces("Commit power face candidate geometry", this.commitPipeline,
      [faces, control, normals, centroids, quadrature, committedFaces, committedNormals,
        committedCentroids, committedQuadrature], 12);
    runIndirectFaces("Commit power face candidate incidence", this.commitIncidencePipeline,
      [rows, incidence, control, boundaryQueries, committedRows, committedIncidence,
        committedBoundaryQueries, directory, committedDirectory], 12);
    broker.fence("power face payload commit");
    run("Commit power face publication header", this.commitControlPipeline, 1,
      [control, committedControl]);
  }

  get source(): OctreePowerFaceSource {
    return { plan: this.plan, faces: this.faces, faceNormals: this.faceNormals, faceCentroids: this.faceCentroids,
      faceQuadrature: this.faceQuadrature,
      incidenceRows: this.workspace,
      incidence: this.incidence, control: this.control, rowDirectory: this.rowDirectory,
      deltaStats: this.deltaStats, liveFaceDispatch: this.workDispatch,
      boundaryPhiQueries: this.boundaryQueries };
  }

  /** Current topology's sorted row identities for the pre-commit fine-to-coarse
   * transaction. Admitted consumers must continue to use `source.rowDirectory`. */
  get candidateRowDirectoryForCoarse(): GPUBuffer { return this.candidateRowDirectory; }

  /** QA-only transactional header. Production consumers must use `source.control`,
   * which remains the last immutable committed publication. */
  get candidateControlForDiagnostics(): GPUBuffer { return this.candidateControl; }

  /** Failure-only readback for a rejected candidate face. Boundary-phi
   * failures repurpose area/inverseDistance for the two sampled endpoint
   * values, so retaining the immutable query beside that payload makes an
   * exact Dawn rejection reproducible without admitting CPU authority. */
  async readCandidateFailure(faceIndex: number) {
    if (!Number.isSafeInteger(faceIndex) || faceIndex < 0
      || faceIndex >= this.plan.faceCapacity) return undefined;
    const readback = this.device.createBuffer({
      label: "Rejected power-face candidate QA",
      size: 96,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read rejected power-face candidate",
    });
    encoder.copyBufferToBuffer(this.candidateFaces, faceIndex * 32, readback, 0, 32);
    encoder.copyBufferToBuffer(this.candidateFaceNormals, faceIndex * 16, readback, 32, 16);
    encoder.copyBufferToBuffer(this.candidateFaceCentroids, faceIndex * 16, readback, 48, 16);
    encoder.copyBufferToBuffer(this.candidateBoundaryQueries, faceIndex * 32, readback, 64, 32);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const bytes = readback.getMappedRange().slice(0);
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      return {
        faceIndex,
        face: {
          negativeRow: words[0], positiveRow: words[1], geometryCode: words[2],
          flags: words[3], normalVelocity: floats[4], area: floats[5],
          inverseDistance: floats[6], openFraction: floats[7],
        },
        normal: Array.from(floats.slice(8, 12)),
        centroid: Array.from(floats.slice(12, 16)),
        query: {
          liquidCenter: Array.from(floats.slice(16, 20)),
          airCenter: Array.from(floats.slice(20, 24)),
        },
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.faces.destroy(); this.candidateFaces.destroy();
    this.faceNormals.destroy(); this.candidateFaceNormals.destroy();
    this.faceCentroids.destroy(); this.candidateFaceCentroids.destroy();
    this.faceQuadrature.destroy(); this.candidateFaceQuadrature.destroy();
    this.incidence.destroy(); this.candidateIncidence.destroy();
    this.workspace.destroy(); this.candidateRows.destroy();
    this.rowDirectory.destroy(); this.candidateRowDirectory.destroy();
    this.rowDiagnostics.destroy();
    this.boundaryQueries.destroy(); this.candidateBoundaryQueries.destroy();
    this.deltaFaces.destroy(); this.deltaOldFaces.destroy(); this.deltaSources.destroy();
    this.liveFaceDispatch.destroy(); this.workDispatch.destroy(); this.deltaStats.destroy();
    this.boundaryPhiEmptyParams.destroy(); this.boundaryPhiEmptyStorage.destroy();
    this.control.destroy(); this.candidateControl.destroy(); this.params.destroy();
  }
}

export const octreePowerFaceShader = /* wgsl */ `
${octreePowerCatalogWGSL}
struct Params { dimensionsRowCount:vec4u, capacitiesGeneration:vec4u, physical:vec4f, boundaryPolicy:vec4u, containerFill:vec4f, phiPolicy:vec4u, delta:vec4u }
struct LeafHeader { cell:u32, entryStart:u32, entryCount:u32, size:u32, diagonal:f32, rhs:f32, pad0:u32, pad1:u32, gradient:vec4f }
struct PowerRowMetric { topologyCode:u32, transformAndFlags:u32, volume:f32, reserved:u32 }
struct CatalogEntry { firstFace:u32, faceCount:u32 }
struct RowWork { faceCount:u32, incidenceCount:u32, faceOffset:u32, incidenceOffset:u32 }
struct PowerIncidence { face:u32, sign:i32 }
struct RowDirectoryEntry { cellPlusOne:u32, size:u32, row:u32, morton:u32 }
struct RowDiagnostic { flag:u32, slot:u32, neighbor:u32, detail:u32 }
struct FaceQuadrature { centroidArea:vec4f, sampleUV:array<u32,${OCTREE_POWER_FACE_QUADRATURE_SAMPLES}> }
struct BoundaryPhiQuery { liquidCenter:vec4f, airCenter:vec4f }
struct Control { rowCount:u32, faceCount:u32, incidenceCount:u32, flags:u32, firstInvalid:u32, invalidCount:u32, boundaryCount:u32, generation:u32, valid:u32, lookupMissCount:u32, maximumLookupSteps:u32, worldBoundaryCount:u32, pad0:u32, pad1:u32, pad2:u32, pad3:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(2) var<storage,read> metrics:array<PowerRowMetric>;
@group(0) @binding(3) var<storage,read> entries:array<CatalogEntry>;
@group(0) @binding(4) var<storage,read> catalogFaces:array<PowerCatalogFace>;
@group(0) @binding(5) var<storage,read_write> rows:array<RowWork>;
@group(0) @binding(6) var<storage,read_write> powerFaces:array<PowerFaceRecord>;
@group(0) @binding(7) var<storage,read_write> incidence:array<PowerIncidence>;
@group(0) @binding(8) var<storage,read_write> control:Control;
@group(0) @binding(9) var<storage,read_write> rowDirectory:array<RowDirectoryEntry>;
@group(0) @binding(11) var<storage,read_write> faceNormals:array<vec4f>;
@group(0) @binding(12) var<storage,read_write> faceCentroids:array<vec4f>;
@group(0) @binding(13) var<storage,read_write> faceQuadrature:array<FaceQuadrature>;
@group(0) @binding(14) var<storage,read_write> boundaryPhiQueries:array<BoundaryPhiQuery>;
@group(0) @binding(15) var<storage,read_write> liveFaceDispatch:array<u32>;
@group(0) @binding(16) var<storage,read_write> rowDiagnostics:array<RowDiagnostic>;
@group(0) @binding(17) var<storage,read_write> committedFaces:array<PowerFaceRecord>;
@group(0) @binding(18) var<storage,read> rowDelta:array<u32>;
@group(0) @binding(19) var<storage,read_write> committedRows:array<RowWork>;
@group(0) @binding(20) var<storage,read_write> committedIncidence:array<PowerIncidence>;
@group(0) @binding(21) var<storage,read_write> committedNormals:array<vec4f>;
@group(0) @binding(22) var<storage,read_write> committedCentroids:array<vec4f>;
@group(0) @binding(23) var<storage,read_write> committedQuadrature:array<FaceQuadrature>;
@group(0) @binding(24) var<storage,read_write> committedBoundaryQueries:array<BoundaryPhiQuery>;
@group(0) @binding(25) var<storage,read_write> committedControl:Control;
@group(0) @binding(26) var<storage,read_write> deltaStats:array<u32>;
@group(0) @binding(27) var<storage,read_write> deltaOldFaces:array<PowerFaceRecord>;
@group(0) @binding(28) var<storage,read_write> deltaSources:array<u32>;
@group(0) @binding(30) var<storage,read_write> deltaFaces:array<PowerFaceRecord>;
@group(0) @binding(31) var<storage,read_write> committedRowDirectory:array<RowDirectoryEntry>;
var<workgroup> parallelScan:array<u32,256>;
var<workgroup> parallelScanAux:array<u32,256>;
var<workgroup> parallelFlags:array<u32,256>;
var<workgroup> parallelRows:array<u32,256>;
var<workgroup> parallelCounts:array<u32,256>;
var<workgroup> parallelBase:u32;
var<workgroup> parallelAuxBase:u32;
var<workgroup> parallelChunkBase:u32;
var<workgroup> parallelChunkAuxBase:u32;
var<workgroup> parallelEnabled:u32;
var<workgroup> parallelInputA:u32;
var<workgroup> parallelInputB:u32;
const INVALID:u32=${OCTREE_POWER_INVALID_ROW}u;
const TOPOLOGY_VALID:u32=${OCTREE_POWER_TOPOLOGY_VALID}u;
const FACE_VALID:u32=${OCTREE_POWER_FACE_VALID}u;
const ROW_DELTA_VALID:u32=${OCTREE_POWER_ROW_DELTA_VALID}u;
const ROW_DELTA_AFFECTED:u32=0x80000000u;
const FACE_DELTA_VALID:u32=0x46444c54u;
const FACE_DELTA_COMPACT_VALID:u32=0x43444c54u;
const FACE_DELTA_CARRIED:u32=${OCTREE_POWER_FACE_DELTA_CARRIED}u;
const FACE_FAILURE:u32=0x20000000u;
const BOUNDARY:u32=${OCTREE_POWER_FACE_BOUNDARY}u;
const OPEN_BOUNDARY:u32=${OCTREE_POWER_FACE_OPEN_BOUNDARY}u;
const WORLD_BOUNDARY_SHIFT:u32=${OCTREE_POWER_FACE_WORLD_BOUNDARY_SHIFT}u;
const INVALID_METRIC:u32=${OCTREE_POWER_FACE_ERROR.invalidMetric}u;
const INVALID_HEADER:u32=${OCTREE_POWER_FACE_ERROR.invalidHeader}u;
const INVALID_CATALOG:u32=${OCTREE_POWER_FACE_ERROR.invalidCatalog}u;
const INVALID_GEOMETRY:u32=${OCTREE_POWER_FACE_ERROR.invalidGeometry}u;
const ASYMMETRIC_FACE:u32=${OCTREE_POWER_FACE_ERROR.asymmetricFace}u;
const CAPACITY:u32=${OCTREE_POWER_FACE_ERROR.capacity}u;
const ROW_DIRECTORY:u32=${OCTREE_POWER_FACE_ERROR.rowDirectory}u;
fn finite(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn dims()->vec3u{return params.dimensionsRowCount.xyz;}
fn deltaAccepted(requested:u32)->bool{
  if(params.delta.x+15u>=arrayLength(&rowDelta)){return false;}
  let base=params.delta.x;let previous=rowDelta[base+1u];let carried=rowDelta[base+2u];
  let added=rowDelta[base+3u];let retired=rowDelta[base+4u];let dirty=rowDelta[base+5u];let affected=rowDelta[base+6u];
  let newToOldFits=params.delta.y<=arrayLength(&rowDelta)&&requested<=arrayLength(&rowDelta)-params.delta.y;
  let oldToNewFits=params.delta.z<=arrayLength(&rowDelta)&&previous<=arrayLength(&rowDelta)-params.delta.z;
  let affectedFits=params.delta.w<=arrayLength(&rowDelta)&&affected<=arrayLength(&rowDelta)-params.delta.w;
  let dirtyFits=params.phiPolicy.w<=arrayLength(&rowDelta)&&dirty<=arrayLength(&rowDelta)-params.phiPolicy.w;
  return rowDelta[base]==requested&&rowDelta[base+7u]==params.capacitiesGeneration.w
    &&rowDelta[base+8u]==ROW_DELTA_VALID&&carried<=previous&&affected<=requested
    &&newToOldFits&&oldToNewFits&&affectedFits&&dirtyFits&&dirty<=affected
    &&requested==carried+added&&requested==previous+added-retired;
}
fn affectedCount()->u32{let at=params.delta.x+6u;if(at>=arrayLength(&rowDelta)){return 0u;}return rowDelta[at];}
fn affectedRow(item:u32)->u32{let at=params.delta.w+item;if(at>=arrayLength(&rowDelta)){return INVALID;}return rowDelta[at];}
fn oldToNew(row:u32)->u32{
  if(params.delta.z+row>=arrayLength(&rowDelta)){return INVALID;}let encoded=rowDelta[params.delta.z+row];
  return select(INVALID,encoded-1u,encoded!=0u);
}
fn newToOld(row:u32)->u32{
  if(params.delta.y+row>=arrayLength(&rowDelta)){return INVALID;}let encoded=rowDelta[params.delta.y+row]&0x7fffffffu;
  return select(INVALID,encoded-1u,encoded!=0u);
}
fn rowAffectedFlag(row:u32)->bool{
  return params.delta.y+row<arrayLength(&rowDelta)&&(rowDelta[params.delta.y+row]&ROW_DELTA_AFFECTED)!=0u;
}
fn rowAffected(row:u32)->bool{
  // The exact new-to-old publication already carries the affected bit for
  // every row. The prepare transaction validates the compact affected list
  // against these bits before any consumer runs, so repeated binary searches
  // of that same immutable list add no authority.
  return rowAffectedFlag(row);
}
fn cellCoord(cell:u32)->vec3u{let d=dims();return vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));}
fn rowHeaderValid(row:u32)->bool{if(row>=arrayLength(&headers)){return false;}let h=headers[row];let d=dims();if(h.size==0u||h.cell>=d.x*d.y*d.z){return false;}let o=cellCoord(h.cell);return all(o%vec3u(h.size)==vec3u(0u))&&all(vec3u(h.size)<=d)&&all(o<=d-vec3u(h.size));}
fn rowCenter(row:u32)->vec3f{let h=headers[row];return (vec3f(cellCoord(h.cell))+vec3f(0.5*f32(h.size)))*params.physical.x;}
fn rowSize(row:u32)->f32{return f32(headers[row].size)*params.physical.x;}
fn closeVector(a:vec3f,b:vec3f,scale:f32)->bool{return all(abs(a-b)<=vec3f(max(1e-5,scale*2e-5)));}
// Keep these checks scalar and reason-coded.  A single deeply nested boolean
// predicate was observed to intermittently miscompile on Dawn's Metal path:
// it rejected finite catalog reconstructions while the same values passed
// when read back and evaluated independently.  This form preserves the strict
// fail-closed contract and reports which invariant actually failed.
fn reconstructionError(face:ReconstructedPowerFace)->u32{
  var error=0u;
  if(!finite(face.neighborSize)||face.neighborSize<0.0){error|=1u;}
  if(!finite(face.area)||face.area<=0.0){error|=2u;}
  if(!finite(face.inverseDistance)||face.inverseDistance<=0.0){error|=4u;}
  if(!finite(face.neighborCenter.x)){error|=8u;}
  if(!finite(face.neighborCenter.y)){error|=8u;}
  if(!finite(face.neighborCenter.z)){error|=8u;}
  if(!finite(face.centroid.x)){error|=16u;}
  if(!finite(face.centroid.y)){error|=16u;}
  if(!finite(face.centroid.z)){error|=16u;}
  var normalFinite=true;
  if(!finite(face.normal.x)){normalFinite=false;}
  if(!finite(face.normal.y)){normalFinite=false;}
  if(!finite(face.normal.z)){normalFinite=false;}
  if(!normalFinite){error|=32u;}
  if(normalFinite){let normalLength=length(face.normal);if(!finite(normalLength)||abs(normalLength-1.0)>2e-4){error|=64u;}}
  return error;
}
fn validReconstruction(face:ReconstructedPowerFace)->bool{return reconstructionError(face)==0u;}
fn mortonPart10(value:u32)->u32{
  var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;
  x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;
}
fn rowMorton(cell:u32)->u32{let p=cellCoord(cell);return mortonPart10(p.x)|(mortonPart10(p.y)<<1u)|(mortonPart10(p.z)<<2u);}
fn rowLevel(size:u32)->u32{return 31u-countLeadingZeros(size);}
fn directoryKeyLess(levelA:u32,mortonA:u32,levelB:u32,mortonB:u32)->bool{
  return levelA<levelB||(levelA==levelB&&mortonA<mortonB);
}
fn neighborKey(center:vec3f,size:f32)->vec2u{let gridSize=size/params.physical.x;let origin=center/params.physical.x-vec3f(0.5*gridSize);let rounded=round(origin);if(abs(gridSize-round(gridSize))>2e-4||any(abs(origin-rounded)>vec3f(2e-4))||any(rounded<vec3f(0.0))||any(rounded>=vec3f(dims()))){return vec2u(INVALID);}let o=vec3u(rounded);return vec2u(o.x+dims().x*(o.y+dims().y*o.z),u32(round(gridSize)));}
fn findNeighbor(center:vec3f,size:f32)->vec2u{
  let key=neighborKey(center,size);if(key.x==INVALID){return vec2u(INVALID,0u);}
  let count=min(params.dimensionsRowCount.w,arrayLength(&headers));
  let level=rowLevel(key.y);let morton=rowMorton(key.x);var low=0u;var high=count;var steps=0u;
  while(low<high){steps+=1u;let middle=low+(high-low)/2u;let candidate=headers[middle];
    if(directoryKeyLess(rowLevel(candidate.size),rowMorton(candidate.cell),level,morton)){low=middle+1u;}else{high=middle;}}
  if(low<count){let candidate=headers[low];if(candidate.cell==key.x&&candidate.size==key.y){return vec2u(low,steps);}}
  return vec2u(INVALID,steps);
}
fn reconstructSlot(row:u32,slot:u32)->ReconstructedPowerFace{let metric=metrics[row];let entry=entries[metric.topologyCode];return reconstructPowerCatalogFace(rowCenter(row),rowSize(row),catalogFaces[entry.firstFace+slot],metric.transformAndFlags&63u);}
fn reciprocalSlot(row:u32,neighbor:u32)->u32{if(neighbor>=arrayLength(&metrics)){return INVALID;}let metric=metrics[neighbor];if((metric.transformAndFlags&TOPOLOGY_VALID)==0u||metric.topologyCode>=arrayLength(&entries)){return INVALID;}let entry=entries[metric.topologyCode];if(entry.firstFace>arrayLength(&catalogFaces)||entry.faceCount>arrayLength(&catalogFaces)-entry.firstFace){return INVALID;}let wantedCenter=rowCenter(row);let wantedSize=rowSize(row);for(var slot=0u;slot<entry.faceCount;slot+=1u){let face=reconstructSlot(neighbor,slot);if(validReconstruction(face)&&abs(face.neighborSize-wantedSize)<=max(1e-5,wantedSize*2e-5)&&closeVector(face.neighborCenter,wantedCenter,wantedSize)){return slot;}}return INVALID;}
struct SharedPolygon{vertices:array<vec3f,16>,count:u32}
fn catalogFace(row:u32,slot:u32)->PowerCatalogFace{let metric=metrics[row];let entry=entries[metric.topologyCode];return catalogFaces[entry.firstFace+slot];}
fn clipPowerPolygon(polygon:SharedPolygon,plane:ReconstructedPowerFace,epsilon:f32)->SharedPolygon{
  if(polygon.count<3u){return polygon;}var clipped:array<vec3f,16>;var output=0u;var previous=polygon.vertices[polygon.count-1u];var previousSide=dot(plane.normal,previous-plane.centroid);var previousInside=previousSide<=epsilon;
  for(var i=0u;i<polygon.count;i+=1u){let current=polygon.vertices[i];let currentSide=dot(plane.normal,current-plane.centroid);let currentInside=currentSide<=epsilon;
    if(currentInside!=previousInside&&output<16u){let denominator=previousSide-currentSide;let t=select(0.5,clamp(previousSide/denominator,0.0,1.0),abs(denominator)>1e-12);clipped[output]=mix(previous,current,t);output+=1u;}
    if(currentInside&&output<16u){clipped[output]=current;output+=1u;}previous=current;previousSide=currentSide;previousInside=currentInside;}
  return SharedPolygon(clipped,output);
}
fn clipPowerPolygonByCell(polygonValue:SharedPolygon,cellRow:u32,epsilon:f32)->SharedPolygon{
  var polygon=polygonValue;let metric=metrics[cellRow];let entry=entries[metric.topologyCode];
  for(var planeSlot=0u;planeSlot<entry.faceCount&&polygon.count>=3u;planeSlot+=1u){polygon=clipPowerPolygon(polygon,reconstructSlot(cellRow,planeSlot),epsilon);}return polygon;
}
fn reciprocalIntersection(row:u32,slot:u32,neighbor:u32,reverseSlot:u32)->SharedPolygon{
  let geometry=reconstructSlot(row,slot);let reverseGeometry=reconstructSlot(neighbor,reverseSlot);let normal=normalize(geometry.normal);let reference=select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(normal.x)<0.75);let tangent=normalize(cross(reference,normal));let bitangent=cross(normal,tangent);let extent=8.0*max(rowSize(row),rowSize(neighbor));let center=0.5*(geometry.centroid+reverseGeometry.centroid);var vertices:array<vec3f,16>;
  vertices[0]=center-extent*tangent-extent*bitangent;vertices[1]=center+extent*tangent-extent*bitangent;vertices[2]=center+extent*tangent+extent*bitangent;vertices[3]=center-extent*tangent+extent*bitangent;
  let epsilon=max(1e-6,1e-5*max(rowSize(row),rowSize(neighbor)));var polygon=SharedPolygon(vertices,4u);polygon=clipPowerPolygonByCell(polygon,row,epsilon);polygon=clipPowerPolygonByCell(polygon,neighbor,epsilon);return polygon;
}
fn boundaryIntersection(row:u32,slot:u32)->SharedPolygon{
  let geometry=reconstructSlot(row,slot);let normal=normalize(geometry.normal);
  let reference=select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(normal.x)<0.75);
  let tangent=normalize(cross(reference,normal));let bitangent=cross(normal,tangent);let extent=8.0*rowSize(row);
  var vertices:array<vec3f,16>;vertices[0]=geometry.centroid-extent*tangent-extent*bitangent;
  vertices[1]=geometry.centroid+extent*tangent-extent*bitangent;vertices[2]=geometry.centroid+extent*tangent+extent*bitangent;
  vertices[3]=geometry.centroid-extent*tangent+extent*bitangent;
  return clipPowerPolygonByCell(SharedPolygon(vertices,4u),row,max(1e-6,1e-5*rowSize(row)));
}
fn physicalFacePolygon(face:PowerFaceRecord)->SharedPolygon{
  let slot=face.geometryCode&0xffffu;if(face.positiveRow==INVALID){return boundaryIntersection(face.negativeRow,slot);}
  let reciprocal=reciprocalSlot(face.negativeRow,face.positiveRow);
  if(reciprocal==INVALID){var empty:array<vec3f,16>;return SharedPolygon(empty,0u);}
  return reciprocalIntersection(face.negativeRow,slot,face.positiveRow,reciprocal);
}
fn sharedGeometry(row:u32,slot:u32,neighbor:u32,reverseSlot:u32)->ReconstructedPowerFace{var geometry=reconstructSlot(row,slot);let polygon=reciprocalIntersection(row,slot,neighbor,reverseSlot);if(polygon.count<3u){geometry.area=0.0;return geometry;}var area=0.0;var centroid=vec3f(0.0);for(var i=1u;i+1u<polygon.count;i+=1u){let crossValue=cross(polygon.vertices[i]-polygon.vertices[0],polygon.vertices[i+1u]-polygon.vertices[0]);let triangleArea=max(0.0,0.5*dot(crossValue,geometry.normal));area+=triangleArea;centroid+=triangleArea*(polygon.vertices[0]+polygon.vertices[i]+polygon.vertices[i+1u])/3.0;}geometry.area=area;if(area>1e-12){geometry.centroid=centroid/area;}return geometry;}
fn worldBoundaryBit(center:vec3f)->u32{let maximum=vec3f(dims())*params.physical.x;if(center.x<0.0){return 1u;}if(center.y<0.0){return 2u;}if(center.z<0.0){return 4u;}if(center.z>maximum.z){return 8u;}if(center.y>maximum.y){return 16u;}if(center.x>maximum.x){return 32u;}return 0u;}
fn worldBoundaryBitFromNormal(normal:vec3f)->u32{if(normal.x< -0.9999){return 1u;}if(normal.y< -0.9999){return 2u;}if(normal.z< -0.9999){return 4u;}if(normal.z>0.9999){return 8u;}if(normal.y>0.9999){return 16u;}if(normal.x>0.9999){return 32u;}return 0u;}
fn boundaryFlags(metric:PowerRowMetric,geometry:ReconstructedPowerFace)->u32{let world=select(worldBoundaryBit(geometry.neighborCenter),worldBoundaryBitFromNormal(geometry.normal),geometry.neighborSize==0.0);let declared=(metric.transformAndFlags>>8u)&63u;let geometricWorld=world&declared;if(geometricWorld!=0u){let open=select(OPEN_BOUNDARY,0u,(params.boundaryPolicy.x&geometricWorld)!=0u);return BOUNDARY|open|(geometricWorld<<WORLD_BOUNDARY_SHIFT);}return BOUNDARY|OPEN_BOUNDARY;}
fn recordRowFailure(row:u32,flag:u32,slot:u32,neighbor:u32,detail:u32){
  if(row>=arrayLength(&rowDiagnostics)){return;}
  if(rowDiagnostics[row].flag==0u){rowDiagnostics[row]=RowDiagnostic(flag,slot,neighbor,detail);}
}
fn fail(row:u32,flag:u32){recordRowFailure(row,flag,INVALID,INVALID,0u);}
fn failGeometry(row:u32,slot:u32,detail:u32){recordRowFailure(row,INVALID_GEOMETRY,slot,INVALID,detail);}
fn failGeometryTopology(row:u32,slot:u32,detail:u32,topologyCode:u32,transformAndFlags:u32){
  recordRowFailure(row,INVALID_GEOMETRY,slot,topologyCode,detail|((transformAndFlags&63u)<<24u));
}
fn failSite(row:u32,slot:u32,neighbor:u32,detail:u32){recordRowFailure(row,ROW_DIRECTORY,slot,neighbor,detail);}
fn failFace(row:u32,slot:u32,neighbor:u32,detail:u32){recordRowFailure(row,ASYMMETRIC_FACE,slot,neighbor,detail);}
fn invalidatePowerFace(faceIndex:u32,flag:u32,detail:u32){
  if(faceIndex>=arrayLength(&powerFaces)){return;}var face=powerFaces[faceIndex];
  face.flags|=FACE_FAILURE;face.normalVelocity=bitcast<f32>(detail);
  face.openFraction=bitcast<f32>(flag);powerFaces[faceIndex]=face;
}
fn reduceRowFailures(count:u32){
  for(var row=0u;row<count&&row<arrayLength(&rowDiagnostics);row+=1u){let failure=rowDiagnostics[row];if(failure.flag==0u){continue;}
    control.flags|=failure.flag;control.invalidCount+=1u;
    if(control.firstInvalid==INVALID){control.firstInvalid=row;control.pad0=failure.slot;control.pad1=failure.neighbor;
      control.pad2=failure.detail;control.pad3=row;}}
}
fn parallelInclusiveScan(lane:u32,value:u32)->u32{
  parallelScan[lane]=value;workgroupBarrier();
  for(var offset=1u;offset<256u;offset*=2u){
    var add=0u;if(lane>=offset){add=parallelScan[lane-offset];}workgroupBarrier();
    if(lane>=offset){parallelScan[lane]+=add;}workgroupBarrier();
  }
  return parallelScan[lane]-value;
}
fn uniformParallelEnable(lane:u32,enabled:bool)->u32{
  if(lane==0u){parallelEnabled=select(0u,1u,enabled);}
  return workgroupUniformLoad(&parallelEnabled);
}
fn uniformParallelInputA(lane:u32,value:u32)->u32{
  if(lane==0u){parallelInputA=value;}return workgroupUniformLoad(&parallelInputA);
}
fn uniformParallelInputB(lane:u32,value:u32)->u32{
  if(lane==0u){parallelInputB=value;}return workgroupUniformLoad(&parallelInputB);
}
fn parallelInclusiveScanPair(lane:u32,value:u32,auxValue:u32)->vec2u{
  parallelScan[lane]=value;parallelScanAux[lane]=auxValue;workgroupBarrier();
  for(var offset=1u;offset<256u;offset*=2u){
    var add=0u;var auxAdd=0u;if(lane>=offset){add=parallelScan[lane-offset];auxAdd=parallelScanAux[lane-offset];}workgroupBarrier();
    if(lane>=offset){parallelScan[lane]+=add;parallelScanAux[lane]+=auxAdd;}workgroupBarrier();
  }
  return vec2u(parallelScan[lane]-value,parallelScanAux[lane]-auxValue);
}
fn reduceRowFailuresParallel(lane:u32,count:u32){
  var flags=0u;var first=INVALID;var failures=0u;
  for(var row=lane;row<count&&row<arrayLength(&rowDiagnostics);row+=256u){let failure=rowDiagnostics[row];
    if(failure.flag!=0u){flags|=failure.flag;first=min(first,row);failures+=1u;}}
  parallelFlags[lane]=flags;parallelRows[lane]=first;parallelCounts[lane]=failures;workgroupBarrier();
  for(var stride=128u;stride>0u;stride/=2u){if(lane<stride){
      parallelFlags[lane]|=parallelFlags[lane+stride];parallelRows[lane]=min(parallelRows[lane],parallelRows[lane+stride]);
      parallelCounts[lane]+=parallelCounts[lane+stride];}workgroupBarrier();}
  if(lane==0u&&parallelCounts[0]>0u){let firstRow=parallelRows[0];let failure=rowDiagnostics[firstRow];
    control.flags|=parallelFlags[0];control.invalidCount+=parallelCounts[0];
    if(control.firstInvalid==INVALID){control.firstInvalid=firstRow;control.pad0=failure.slot;control.pad1=failure.neighbor;
      control.pad2=failure.detail;control.pad3=firstRow;}}
  storageBarrier();workgroupBarrier();
}
fn faceKeyLess(aNegative:u32,aPositive:u32,aGeometry:u32,bNegative:u32,bPositive:u32,bGeometry:u32)->bool{
  if(aNegative!=bNegative){return aNegative<bNegative;}
  if(aGeometry!=bGeometry){return aGeometry<bGeometry;}
  return aPositive<bPositive;
}
fn faceKeyEqual(a:PowerFaceRecord,b:PowerFaceRecord)->bool{
  return a.negativeRow==b.negativeRow&&a.positiveRow==b.positiveRow&&a.geometryCode==b.geometryCode;
}
fn deltaFaceLowerBound(face:PowerFaceRecord,count:u32)->u32{
  var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let candidate=deltaFaces[middle];
    if(faceKeyLess(candidate.negativeRow,candidate.positiveRow,candidate.geometryCode,face.negativeRow,face.positiveRow,face.geometryCode)){low=middle+1u;}else{high=middle;}}
  return low;
}
fn oldFaceLowerBound(face:PowerFaceRecord,count:u32)->u32{
  var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let candidate=deltaOldFaces[middle];
    if(faceKeyLess(candidate.negativeRow,candidate.positiveRow,candidate.geometryCode,face.negativeRow,face.positiveRow,face.geometryCode)){low=middle+1u;}else{high=middle;}}
  return low;
}
fn powerFaceLowerBound(face:PowerFaceRecord,count:u32)->u32{
  var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let candidate=powerFaces[middle];
    if(faceKeyLess(candidate.negativeRow,candidate.positiveRow,candidate.geometryCode,face.negativeRow,face.positiveRow,face.geometryCode)){low=middle+1u;}else{high=middle;}}
  return low;
}
fn negativeRowLowerBound(row:u32,count:u32)->u32{
  var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;
    if(powerFaces[middle].negativeRow<row){low=middle+1u;}else{high=middle;}}return low;
}
fn stagePayloadSource(output:u32,sourcePlusOne:u32)->bool{
  if(output>=arrayLength(&powerFaces)||output>=arrayLength(&boundaryPhiQueries)){return false;}
  boundaryPhiQueries[output]=BoundaryPhiQuery(vec4f(0.0),vec4f(0.0,0.0,0.0,bitcast<f32>(sourcePlusOne)));
  return true;
}
@compute @workgroup_size(256) fn preparePowerFaces(@builtin(local_invocation_index) lane:u32){
  // Diagnostics are generation-local transactional state. Reset them in the
  // already-required prepare dispatch so construction does not split its
  // compute pass with a recurring full-buffer clear command.
  for(var row=lane;row<arrayLength(&rowDiagnostics);row+=256u){
    rowDiagnostics[row]=RowDiagnostic(0u,INVALID,INVALID,0u);
  }
  workgroupBarrier();
  if(lane!=0u){return;}
  let requested=params.dimensionsRowCount.w;
  let lookupSteps=select(0u,32u-countLeadingZeros(requested),requested>0u);
  control=Control(requested,0u,0u,0u,INVALID,0u,0u,params.capacitiesGeneration.w,
    0u,0u,lookupSteps,0u,INVALID,INVALID,0u,INVALID);
  if(arrayLength(&deltaStats)>=4u){deltaStats[0]=0u;deltaStats[1]=0u;deltaStats[2]=0u;deltaStats[3]=0u;}
  if(arrayLength(&liveFaceDispatch)>=6u){
    for(var slot=0u;slot<4u;slot+=1u){liveFaceDispatch[slot*3u]=0u;liveFaceDispatch[slot*3u+1u]=1u;liveFaceDispatch[slot*3u+2u]=1u;}
    let rowGroups=(requested+63u)/64u;liveFaceDispatch[6]=rowGroups;liveFaceDispatch[7]=1u;
    let blockGroups=(requested+255u)/256u;liveFaceDispatch[9]=blockGroups;liveFaceDispatch[10]=1u;
  }
  var accepted=deltaAccepted(requested);
  if(accepted){
    let base=params.delta.x;let previous=rowDelta[base+1u];let dirty=rowDelta[base+5u];let affected=rowDelta[base+6u];
    if(previous>0u&&(committedControl.valid!=FACE_VALID||committedControl.flags!=0u||committedControl.rowCount!=previous)){accepted=false;}
    var prior=INVALID;
    for(var item=0u;item<affected;item+=1u){let row=affectedRow(item);
      if(row>=requested||(item>0u&&row<=prior)||!rowAffectedFlag(row)){accepted=false;break;}prior=row;}
    prior=INVALID;
    for(var item=0u;item<dirty;item+=1u){let row=rowDelta[params.phiPolicy.w+item];
      if(row>=requested||(item>0u&&row<=prior)||!rowAffected(row)){accepted=false;break;}prior=row;}
  }
  if(!accepted){control.flags=CAPACITY;control.firstInvalid=0u;control.invalidCount=1u;
    if(arrayLength(&liveFaceDispatch)>=6u){liveFaceDispatch[0]=0u;liveFaceDispatch[3]=0u;}}
}
struct AffectedFaceResult { face:PowerFaceRecord, emit:u32 }
fn affectedFace(row:u32,slot:u32)->AffectedFaceResult{
  let empty=PowerFaceRecord(0u,0u,0u,0u,0.0,0.0,0.0,0.0);
  if(row>=params.dimensionsRowCount.w||!rowHeaderValid(row)){fail(row,INVALID_HEADER);return AffectedFaceResult(empty,0u);}
  if(row>=arrayLength(&metrics)){fail(row,INVALID_METRIC);return AffectedFaceResult(empty,0u);}let metric=metrics[row];
  if((metric.transformAndFlags&TOPOLOGY_VALID)==0u||metric.topologyCode>=arrayLength(&entries)||!finite(metric.volume)||metric.volume<=0.0){
    fail(row,INVALID_METRIC);return AffectedFaceResult(empty,0u);}
  let entry=entries[metric.topologyCode];
  if(entry.firstFace>arrayLength(&catalogFaces)||entry.faceCount>arrayLength(&catalogFaces)-entry.firstFace||entry.faceCount>${OCTREE_POWER_FACE_ROW_MAX}u){
    fail(row,INVALID_CATALOG);return AffectedFaceResult(empty,0u);}
  var geometry=reconstructSlot(row,slot);let reconstructionFailure=reconstructionError(geometry);
  if(reconstructionFailure!=0u){failGeometryTopology(row,slot,4096u|reconstructionFailure,metric.topologyCode,metric.transformAndFlags);return AffectedFaceResult(empty,0u);}
  let worldPlane=geometry.neighborSize==0.0;var neighbor=INVALID;
  if(worldPlane){let world=worldBoundaryBitFromNormal(geometry.normal);
    if(world==0u||((((metric.transformAndFlags>>8u)&63u)&world)==0u)){failGeometry(row,slot,8192u);return AffectedFaceResult(empty,0u);}}
  else{neighbor=findNeighbor(geometry.neighborCenter,geometry.neighborSize).x;}
  if(neighbor!=INVALID){if(neighbor==row){failSite(row,slot,neighbor,16u);return AffectedFaceResult(empty,0u);}
    let reciprocal=reciprocalSlot(row,neighbor);if(reciprocal==INVALID){failFace(row,slot,neighbor,1u);return AffectedFaceResult(empty,0u);}
    let reverse=reconstructSlot(neighbor,reciprocal);var detail=0u;
    if(abs(geometry.inverseDistance-reverse.inverseDistance)>max(1e-5,geometry.inverseDistance*2e-4)){detail|=4u;}
    if(dot(geometry.normal,reverse.normal)>-0.999){detail|=16u;}
    let sharedFace=sharedGeometry(row,slot,neighbor,reciprocal);
    if(!finite(sharedFace.area)||sharedFace.area<=1e-10||!all(vec3<bool>(finite(sharedFace.centroid.x),finite(sharedFace.centroid.y),finite(sharedFace.centroid.z)))){detail|=32u;}
    if(detail!=0u){failFace(row,slot,neighbor,detail);return AffectedFaceResult(empty,0u);}
    if(row>neighbor){return AffectedFaceResult(empty,0u);}geometry=sharedFace;
  }
  let boundary=neighbor==INVALID;let flags=FACE_VALID|select(0u,boundaryFlags(metric,geometry),boundary);
  return AffectedFaceResult(PowerFaceRecord(row,neighbor,(slot&0xffffu)|((metric.transformAndFlags&63u)<<16u),
    flags,0.0,geometry.area,geometry.inverseDistance,1.0),1u);
}
@compute @workgroup_size(256) fn buildAffectedPowerFaceRows(@builtin(local_invocation_index) lane:u32){
  if(uniformParallelEnable(lane,control.flags==0u)==0u){return;}
  let count=uniformParallelInputA(lane,affectedCount());
  if(lane==0u){parallelBase=0u;}workgroupBarrier();
  for(var chunk=0u;chunk<count;chunk+=256u){let item=chunk+lane;var emitted=0u;var row=INVALID;
    if(item<count){row=affectedRow(item);if(row<arrayLength(&rows)&&row<arrayLength(&metrics)){let metric=metrics[row];
        if((metric.transformAndFlags&TOPOLOGY_VALID)!=0u&&metric.topologyCode<arrayLength(&entries)){
          let entry=entries[metric.topologyCode];for(var slot=0u;slot<entry.faceCount&&slot<${OCTREE_POWER_FACE_ROW_MAX}u;slot+=1u){emitted+=affectedFace(row,slot).emit;}}
        else{fail(row,INVALID_METRIC);}}else{fail(min(row,params.dimensionsRowCount.w-1u),CAPACITY);}}
    let localOffset=parallelInclusiveScan(lane,emitted);let chunkItems=min(256u,count-chunk);
    if(lane==0u){parallelChunkBase=parallelBase;parallelBase+=parallelScan[chunkItems-1u];}workgroupBarrier();
    if(item<count&&row<arrayLength(&rows)){rows[row]=RowWork(emitted,0u,parallelChunkBase+localOffset,0u);
      if(parallelBase<=params.capacitiesGeneration.y&&parallelBase<=arrayLength(&deltaFaces)){var output=parallelChunkBase+localOffset;
        let entry=entries[metrics[row].topologyCode];for(var slot=0u;slot<entry.faceCount&&slot<${OCTREE_POWER_FACE_ROW_MAX}u;slot+=1u){
          let result=affectedFace(row,slot);if(result.emit!=0u){deltaFaces[output]=result.face;output+=1u;}}}}
    workgroupBarrier();
  }
  if(lane==0u){control.faceCount=parallelBase;if(parallelBase>params.capacitiesGeneration.y||parallelBase>arrayLength(&deltaFaces)){fail(0u,CAPACITY);}}
}
fn mapOldFace(source:u32)->AffectedFaceResult{
  var face=committedFaces[source];let oldNegative=oldToNew(face.negativeRow);if(oldNegative==INVALID){return AffectedFaceResult(face,0u);}
  var oldPositive=INVALID;if(face.positiveRow!=INVALID){oldPositive=oldToNew(face.positiveRow);if(oldPositive==INVALID){return AffectedFaceResult(face,0u);}}
  face.negativeRow=oldNegative;face.positiveRow=oldPositive;return AffectedFaceResult(face,1u);
}
fn remappedOldFace(source:u32,candidateCount:u32)->AffectedFaceResult{
  let mapped=mapOldFace(source);if(mapped.emit==0u){return mapped;}let face=mapped.face;
  var priorSource=source;while(priorSource>0u){priorSource-=1u;let prior=mapOldFace(priorSource);if(prior.emit==0u){continue;}
    if(!faceKeyLess(prior.face.negativeRow,prior.face.positiveRow,prior.face.geometryCode,
        face.negativeRow,face.positiveRow,face.geometryCode)){return AffectedFaceResult(face,4u);}break;}
  let candidateAt=deltaFaceLowerBound(face,candidateCount);let exact=candidateAt<candidateCount&&faceKeyEqual(deltaFaces[candidateAt],face);
  let touched=rowAffected(face.negativeRow)||(face.positiveRow!=INVALID&&rowAffected(face.positiveRow));
  if(touched&&!rowAffected(face.negativeRow)&&!exact){return AffectedFaceResult(face,2u);}
  if(exact){return AffectedFaceResult(face,3u);}
  return AffectedFaceResult(face,select(0u,1u,!touched));
}
@compute @workgroup_size(256) fn compactPowerFaceIdentityDelta(@builtin(local_invocation_index) lane:u32){
  if(uniformParallelEnable(lane,arrayLength(&deltaStats)>=4u)==0u){return;}storageBarrier();workgroupBarrier();
  reduceRowFailuresParallel(lane,min(params.dimensionsRowCount.w,arrayLength(&rowDiagnostics)));
  if(uniformParallelEnable(lane,control.flags==0u)==0u){return;}
  let candidateCount=uniformParallelInputA(lane,control.faceCount);
  let previousCount=uniformParallelInputB(lane,select(0u,committedControl.faceCount,committedControl.valid==FACE_VALID));
  let candidateSourceBase=params.capacitiesGeneration.y;
  for(var candidate=lane;candidate<candidateCount&&candidateSourceBase+candidate<arrayLength(&deltaSources);candidate+=256u){
    deltaSources[candidateSourceBase+candidate]=0u;}
  storageBarrier();workgroupBarrier();
  if(lane==0u){parallelBase=0u;parallelAuxBase=0u;}workgroupBarrier();
  for(var chunk=0u;chunk<previousCount;chunk+=256u){let source=chunk+lane;var keep=0u;var exact=0u;var closureError=INVALID;
    var remapped=PowerFaceRecord(0u,0u,0u,0u,0.0,0.0,0.0,0.0);
    if(source<previousCount&&source<arrayLength(&committedFaces)){let result=remappedOldFace(source,candidateCount);
      remapped=result.face;keep=select(0u,1u,result.emit==1u);exact=select(0u,1u,result.emit==3u);
      if(result.emit==2u||result.emit==4u){closureError=source;}
      if(result.emit==3u){let candidateAt=deltaFaceLowerBound(remapped,candidateCount);var candidateFace=deltaFaces[candidateAt];
        candidateFace.normalVelocity=remapped.normalVelocity;candidateFace.flags|=FACE_DELTA_CARRIED;deltaFaces[candidateAt]=candidateFace;
        deltaSources[candidateSourceBase+candidateAt]=source+1u;}}
    parallelRows[lane]=closureError;workgroupBarrier();
    for(var stride=128u;stride>0u;stride/=2u){if(lane<stride){parallelRows[lane]=min(parallelRows[lane],parallelRows[lane+stride]);}workgroupBarrier();}
    if(lane==0u&&parallelRows[0]!=INVALID){let bad=remappedOldFace(parallelRows[0],candidateCount).face;fail(bad.negativeRow,ASYMMETRIC_FACE);}
    let offsets=parallelInclusiveScanPair(lane,keep,exact);let chunkItems=min(256u,previousCount-chunk);
    if(lane==0u){parallelChunkBase=parallelBase;parallelBase+=parallelScan[chunkItems-1u];
      parallelAuxBase+=parallelScanAux[chunkItems-1u];}workgroupBarrier();
    if(keep!=0u){let output=parallelChunkBase+offsets.x;if(output<arrayLength(&deltaOldFaces)&&output<candidateSourceBase){
        deltaOldFaces[output]=remapped;deltaSources[output]=source;}}
    workgroupBarrier();
  }
  let survivorCount=uniformParallelInputA(lane,parallelBase);
  let exactCount=uniformParallelInputB(lane,parallelAuxBase);
  if(lane==0u&&(candidateCount>arrayLength(&deltaFaces)||candidateSourceBase+candidateCount>arrayLength(&deltaSources)||
      previousCount>arrayLength(&committedFaces)||survivorCount>arrayLength(&deltaOldFaces)||survivorCount>candidateSourceBase)){fail(0u,CAPACITY);}
  storageBarrier();workgroupBarrier();
  var firstBad=INVALID;
  for(var index=lane;index<candidateCount;index+=256u){let face=deltaFaces[index];
    if((face.flags&FACE_VALID)==0u||face.negativeRow>=params.dimensionsRowCount.w||
        (index>0u&&!faceKeyLess(deltaFaces[index-1u].negativeRow,deltaFaces[index-1u].positiveRow,deltaFaces[index-1u].geometryCode,
          face.negativeRow,face.positiveRow,face.geometryCode))){firstBad=min(firstBad,index);}}
  for(var index=lane;index<survivorCount;index+=256u){let face=deltaOldFaces[index];
    if((face.flags&FACE_VALID)==0u||face.negativeRow>=params.dimensionsRowCount.w||
        (index>0u&&!faceKeyLess(deltaOldFaces[index-1u].negativeRow,deltaOldFaces[index-1u].positiveRow,deltaOldFaces[index-1u].geometryCode,
          face.negativeRow,face.positiveRow,face.geometryCode))){firstBad=min(firstBad,candidateCount+index);}
    let candidateAt=deltaFaceLowerBound(face,candidateCount);
    if(candidateAt<candidateCount&&faceKeyEqual(deltaFaces[candidateAt],face)){firstBad=min(firstBad,candidateCount+index);}}
  parallelRows[lane]=firstBad;workgroupBarrier();
  for(var stride=128u;stride>0u;stride/=2u){if(lane<stride){parallelRows[lane]=min(parallelRows[lane],parallelRows[lane+stride]);}workgroupBarrier();}
  if(lane==0u&&parallelRows[0]!=INVALID){var bad=deltaFaces[parallelRows[0]];
    if(parallelRows[0]>=candidateCount){bad=deltaOldFaces[parallelRows[0]-candidateCount];}
    fail(min(bad.negativeRow,params.dimensionsRowCount.w-1u),ASYMMETRIC_FACE);}
  storageBarrier();workgroupBarrier();reduceRowFailuresParallel(lane,min(params.dimensionsRowCount.w,arrayLength(&rowDiagnostics)));
  if(uniformParallelEnable(lane,control.flags==0u)==0u){return;}
  if(lane==0u){deltaStats[0]=survivorCount;deltaStats[1]=exactCount;deltaStats[2]=previousCount;deltaStats[3]=FACE_DELTA_COMPACT_VALID;}
}
@compute @workgroup_size(256) fn mergePowerFaceIdentityDelta(@builtin(local_invocation_index) lane:u32){
  if(uniformParallelEnable(lane,control.flags==0u&&arrayLength(&deltaStats)>=4u&&deltaStats[3]==FACE_DELTA_COMPACT_VALID)==0u){return;}
  let candidateCount=uniformParallelInputA(lane,control.faceCount);
  let survivorCount=uniformParallelInputB(lane,deltaStats[0]);
  let exactCount=uniformParallelInputA(lane,deltaStats[1]);
  let previousCount=uniformParallelInputB(lane,deltaStats[2]);
  let candidateSourceBase=params.capacitiesGeneration.y;
  let outputCount=candidateCount+survivorCount;
  if(lane==0u&&outputCount>min(params.capacitiesGeneration.y,arrayLength(&powerFaces))){fail(0u,CAPACITY);}
  storageBarrier();workgroupBarrier();reduceRowFailuresParallel(lane,min(params.dimensionsRowCount.w,arrayLength(&rowDiagnostics)));
  if(uniformParallelEnable(lane,control.flags==0u)==0u){return;}
  for(var candidate=lane;candidate<candidateCount;candidate+=256u){var face=deltaFaces[candidate];let oldAt=oldFaceLowerBound(face,survivorCount);
    let output=candidate+oldAt;powerFaces[output]=face;_ = stagePayloadSource(output,deltaSources[candidateSourceBase+candidate]);}
  for(var oldAt=lane;oldAt<survivorCount;oldAt+=256u){var face=deltaOldFaces[oldAt];let candidateAt=deltaFaceLowerBound(face,candidateCount);
    let output=oldAt+candidateAt;face.flags|=FACE_DELTA_CARRIED;powerFaces[output]=face;_ = stagePayloadSource(output,deltaSources[oldAt]+1u);}
  storageBarrier();workgroupBarrier();
  if(lane==0u){let carried=survivorCount+exactCount;let added=candidateCount-exactCount;let retired=previousCount-carried;
    if(outputCount!=carried+added||outputCount+retired!=previousCount+added){fail(0u,ASYMMETRIC_FACE);}
    else{control.faceCount=outputCount;deltaStats[0]=carried;deltaStats[1]=added;deltaStats[2]=retired;deltaStats[3]=FACE_DELTA_VALID;}}
}
struct RowIncidenceResult { identity:PowerFaceRecord, sign:i32, valid:u32 }
fn rowIncidenceIdentity(row:u32,slot:u32)->RowIncidenceResult{
  let empty=PowerFaceRecord(0u,0u,0u,0u,0.0,0.0,0.0,0.0);let metric=metrics[row];let geometry=reconstructSlot(row,slot);
  var neighbor=INVALID;if(geometry.neighborSize!=0.0){neighbor=findNeighbor(geometry.neighborCenter,geometry.neighborSize).x;}
  if(neighbor==row){failSite(row,slot,neighbor,16u);return RowIncidenceResult(empty,0,0u);}
  if(neighbor==INVALID||row<neighbor){return RowIncidenceResult(PowerFaceRecord(row,neighbor,
      (slot&0xffffu)|((metric.transformAndFlags&63u)<<16u),FACE_VALID,0.0,0.0,0.0,0.0),1,1u);}
  let reciprocal=reciprocalSlot(row,neighbor);if(reciprocal==INVALID){failFace(row,slot,neighbor,1u);return RowIncidenceResult(empty,0,0u);}
  let ownerMetric=metrics[neighbor];return RowIncidenceResult(PowerFaceRecord(neighbor,row,
    (reciprocal&0xffffu)|((ownerMetric.transformAndFlags&63u)<<16u),FACE_VALID,0.0,0.0,0.0,0.0),-1,1u);
}
@compute @workgroup_size(64) fn preparePowerFaceCandidateCSR(@builtin(global_invocation_id) gid:vec3u){
  if(control.flags!=0u){return;}let row=gid.x;let requested=params.dimensionsRowCount.w;if(row>=requested){return;}
  if(row>=arrayLength(&rows)||row>=arrayLength(&rowDirectory)||row>=arrayLength(&headers)||row>=arrayLength(&metrics)){fail(min(row,requested-1u),CAPACITY);return;}
  let metric=metrics[row];if(!rowHeaderValid(row)){fail(row,INVALID_HEADER);return;}
  if((metric.transformAndFlags&TOPOLOGY_VALID)==0u||metric.topologyCode>=arrayLength(&entries)){fail(row,INVALID_METRIC);return;}
  let entry=entries[metric.topologyCode];if(entry.faceCount>${OCTREE_POWER_FACE_ROW_MAX}u){fail(row,INVALID_CATALOG);return;}
  let incidenceCount=select(INVALID,entry.faceCount,rowAffected(row));rows[row]=RowWork(0u,incidenceCount,0u,0u);
  let header=headers[row];let morton=rowMorton(header.cell);
  if(row>0u){let prior=headers[row-1u];if(!directoryKeyLess(rowLevel(prior.size),rowMorton(prior.cell),rowLevel(header.size),morton)){failSite(row,row-1u,row,32u);}}
  rowDirectory[row]=RowDirectoryEntry(header.cell+1u,header.size,row,morton);
}
@compute @workgroup_size(256) fn scanPowerFaceCandidateCSR(
    @builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) wid:vec3u,
    @builtin(num_workgroups) workgroups:vec3u){
  let enabled=uniformParallelEnable(lane,control.flags==0u);
  let block=wid.x+wid.y*workgroups.x;let row=block*256u+lane;
  let requested=params.dimensionsRowCount.w;var faceCount=0u;var incidenceCount=0u;
  if(enabled!=0u&&row<requested&&row<arrayLength(&rows)){var work=rows[row];
    faceCount=negativeRowLowerBound(row+1u,control.faceCount)-negativeRowLowerBound(row,control.faceCount);
    if(work.incidenceCount==INVALID){let old=newToOld(row);
      if(old==INVALID||old>=committedControl.rowCount||old>=arrayLength(&committedRows)){fail(row,CAPACITY);}
      else{incidenceCount=committedRows[old].incidenceCount;}}
    else{incidenceCount=work.incidenceCount;}
    work.faceCount=faceCount;work.incidenceCount=incidenceCount;rows[row]=work;
  }
  let offsets=parallelInclusiveScanPair(lane,faceCount,incidenceCount);
  if(enabled!=0u&&row<requested&&row<arrayLength(&rows)){rows[row]=RowWork(faceCount,incidenceCount,offsets.x,offsets.y);}
  if(enabled!=0u&&lane==0u){let at=12u+4u*block;if(at+3u<arrayLength(&liveFaceDispatch)){
      liveFaceDispatch[at]=parallelScan[255];liveFaceDispatch[at+1u]=parallelScanAux[255];
      liveFaceDispatch[at+2u]=0u;liveFaceDispatch[at+3u]=0u;}else{fail(0u,CAPACITY);}}
}
@compute @workgroup_size(256) fn prefixPowerFaceCandidateCSR(@builtin(local_invocation_index) lane:u32){
  let enabled=uniformParallelEnable(lane,control.flags==0u);
  let requested=params.dimensionsRowCount.w;let blocks=(requested+255u)/256u;
  let chunk=(blocks+255u)/256u;let begin=min(blocks,lane*chunk);let end=min(blocks,begin+chunk);
  var faceSubtotal=0u;var incidenceSubtotal=0u;
  if(enabled!=0u){for(var block=begin;block<end;block+=1u){let at=12u+4u*block;
      faceSubtotal+=liveFaceDispatch[at];incidenceSubtotal+=liveFaceDispatch[at+1u];}}
  let offsets=parallelInclusiveScanPair(lane,faceSubtotal,incidenceSubtotal);
  var faceCursor=offsets.x;var incidenceCursor=offsets.y;
  if(enabled!=0u){for(var block=begin;block<end;block+=1u){let at=12u+4u*block;liveFaceDispatch[at+2u]=faceCursor;
      liveFaceDispatch[at+3u]=incidenceCursor;faceCursor+=liveFaceDispatch[at];incidenceCursor+=liveFaceDispatch[at+1u];}}
  if(enabled!=0u&&lane==0u){let faceTotal=parallelScan[255];let incidenceTotal=parallelScanAux[255];
    if(faceTotal!=control.faceCount||incidenceTotal>params.capacitiesGeneration.z||requested>=arrayLength(&rows)){fail(0u,CAPACITY);return;}
    control.incidenceCount=incidenceTotal;rows[requested]=RowWork(FACE_VALID,0u,faceTotal,incidenceTotal);
    let faceGroups=(faceTotal+63u)/64u;liveFaceDispatch[0]=faceGroups;liveFaceDispatch[1]=1u;
    liveFaceDispatch[3]=faceGroups;liveFaceDispatch[4]=1u;}
}
@compute @workgroup_size(256) fn offsetPowerFaceCandidateCSR(
    @builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) wid:vec3u,
    @builtin(num_workgroups) workgroups:vec3u){
  if(control.flags!=0u){return;}let block=wid.x+wid.y*workgroups.x;let row=block*256u+lane;
  if(row>=params.dimensionsRowCount.w||row>=arrayLength(&rows)){return;}let at=12u+4u*block;var work=rows[row];
  work.faceOffset+=liveFaceDispatch[at+2u];work.incidenceOffset+=liveFaceDispatch[at+3u];rows[row]=work;
}
@compute @workgroup_size(256) fn finalizePowerFaceCandidateCSR(@builtin(local_invocation_index) lane:u32){
  let requested=params.dimensionsRowCount.w;reduceRowFailuresParallel(lane,min(requested,arrayLength(&rowDiagnostics)));
  if(uniformParallelEnable(lane,control.flags==0u)==0u){return;}let faceCountTotal=uniformParallelInputA(lane,control.faceCount);
  var boundaryCount=0u;var worldCount=0u;var lookupMissCount=0u;
  for(var faceIndex=lane;faceIndex<faceCountTotal;faceIndex+=256u){let face=powerFaces[faceIndex];let world=(face.flags>>WORLD_BOUNDARY_SHIFT)&63u;
    if((face.flags&BOUNDARY)!=0u){boundaryCount+=1u;if(world!=0u){worldCount+=1u;}else if(face.positiveRow==INVALID){lookupMissCount+=1u;}}}
  parallelFlags[lane]=boundaryCount;parallelRows[lane]=worldCount;parallelCounts[lane]=lookupMissCount;workgroupBarrier();
  for(var stride=128u;stride>0u;stride/=2u){if(lane<stride){parallelFlags[lane]+=parallelFlags[lane+stride];
      parallelRows[lane]+=parallelRows[lane+stride];parallelCounts[lane]+=parallelCounts[lane+stride];}workgroupBarrier();}
  if(lane==0u){control.boundaryCount=parallelFlags[0];control.worldBoundaryCount=parallelRows[0];control.lookupMissCount=parallelCounts[0];}
}
@compute @workgroup_size(64) fn publishAffectedPowerFaceCandidateCSR(@builtin(global_invocation_id) gid:vec3u){
  if(control.flags!=0u){return;}
  let row=gid.x;let requested=params.dimensionsRowCount.w;if(row>=requested||row>=arrayLength(&rows)){return;}
  if(!rowAffected(row)){return;}let faceCountTotal=control.faceCount;var local:array<PowerIncidence,${OCTREE_POWER_FACE_ROW_MAX}>;var localCount=0u;
  let metric=metrics[row];let entry=entries[metric.topologyCode];
  for(var slot=0u;slot<entry.faceCount;slot+=1u){let result=rowIncidenceIdentity(row,slot);if(result.valid==0u){continue;}
    let faceIndex=powerFaceLowerBound(result.identity,faceCountTotal);
    if(faceIndex>=faceCountTotal||!faceKeyEqual(powerFaces[faceIndex],result.identity)){failFace(row,slot,result.identity.negativeRow,64u);continue;}
    var insertion=localCount;while(insertion>0u&&local[insertion-1u].face>faceIndex){local[insertion]=local[insertion-1u];insertion-=1u;}
    local[insertion]=PowerIncidence(faceIndex,result.sign);localCount+=1u;}
  if(localCount!=rows[row].incidenceCount){fail(row,ASYMMETRIC_FACE);return;}
  for(var item=0u;item<localCount;item+=1u){let output=rows[row].incidenceOffset+item;if(output>=arrayLength(&incidence)){fail(row,CAPACITY);break;}
    incidence[output]=local[item];}
}
@compute @workgroup_size(64) fn publishCarriedPowerFaceCandidateCSR(@builtin(global_invocation_id) gid:vec3u){
  if(control.flags!=0u){return;}let row=gid.x;let requested=params.dimensionsRowCount.w;
  if(row>=requested||row>=arrayLength(&rows)||rowAffected(row)){return;}
  let old=newToOld(row);if(old==INVALID||old>=arrayLength(&committedRows)){fail(row,CAPACITY);return;}
  let prior=committedRows[old];var localCount=0u;
  for(var item=0u;item<prior.incidenceCount;item+=1u){let source=prior.incidenceOffset+item;
    if(source>=arrayLength(&committedIncidence)){fail(row,CAPACITY);break;}let oldIncidence=committedIncidence[source];
    if(oldIncidence.face>=committedControl.faceCount||oldIncidence.face>=arrayLength(&committedFaces)){fail(row,CAPACITY);break;}
    let mapped=mapOldFace(oldIncidence.face);if(mapped.emit==0u){fail(row,ASYMMETRIC_FACE);break;}
    let faceIndex=powerFaceLowerBound(mapped.face,control.faceCount);
    if(faceIndex>=control.faceCount||!faceKeyEqual(powerFaces[faceIndex],mapped.face)){fail(row,ASYMMETRIC_FACE);break;}
    let output=rows[row].incidenceOffset+localCount;if(output>=arrayLength(&incidence)){fail(row,CAPACITY);break;}
    incidence[output]=PowerIncidence(faceIndex,oldIncidence.sign);localCount+=1u;
  }
  if(localCount!=rows[row].incidenceCount){fail(row,ASYMMETRIC_FACE);}
}
@compute @workgroup_size(64) fn carryPowerFacePayload(@builtin(global_invocation_id) gid:vec3u){let faceIndex=gid.x;
  if(faceIndex>=control.faceCount||faceIndex>=arrayLength(&powerFaces)){return;}let face=powerFaces[faceIndex];
  if((face.flags&FACE_DELTA_CARRIED)==0u){return;}let sourcePlusOne=bitcast<u32>(boundaryPhiQueries[faceIndex].airCenter.w);
  if(sourcePlusOne==0u){invalidatePowerFace(faceIndex,CAPACITY,32u);return;}let source=sourcePlusOne-1u;
  if(faceIndex>=arrayLength(&faceNormals)||faceIndex>=arrayLength(&faceCentroids)||faceIndex>=arrayLength(&faceQuadrature)
      ||faceIndex>=arrayLength(&boundaryPhiQueries)||source>=arrayLength(&committedNormals)
      ||source>=arrayLength(&committedCentroids)||source>=arrayLength(&committedQuadrature)
      ||source>=arrayLength(&committedBoundaryQueries)){invalidatePowerFace(faceIndex,CAPACITY,33u);return;}
  faceNormals[faceIndex]=committedNormals[source];faceCentroids[faceIndex]=committedCentroids[source];
  faceQuadrature[faceIndex]=committedQuadrature[source];boundaryPhiQueries[faceIndex]=committedBoundaryQueries[source];
}
@compute @workgroup_size(64) fn publishAddedPowerFaceGeometry(@builtin(global_invocation_id) gid:vec3u){let faceIndex=gid.x;
  if(faceIndex>=control.faceCount||faceIndex>=arrayLength(&powerFaces)||faceIndex>=arrayLength(&faceNormals)
      ||faceIndex>=arrayLength(&faceCentroids)||faceIndex>=arrayLength(&faceQuadrature)
      ||faceIndex>=arrayLength(&boundaryPhiQueries)){return;}
  var face=powerFaces[faceIndex];if((face.flags&FACE_DELTA_CARRIED)!=0u){return;}
  boundaryPhiQueries[faceIndex]=BoundaryPhiQuery(vec4f(0.0),vec4f(0.0));
  let slot=face.geometryCode&0xffffu;var geometry=reconstructSlot(face.negativeRow,slot);
  if(face.positiveRow!=INVALID){let reciprocal=reciprocalSlot(face.negativeRow,face.positiveRow);if(reciprocal!=INVALID){geometry=sharedGeometry(face.negativeRow,slot,face.positiveRow,reciprocal);}}
  faceNormals[faceIndex]=vec4f(geometry.normal,0.0);faceCentroids[faceIndex]=vec4f(geometry.centroid,0.0);
  let polygon=physicalFacePolygon(face);
  if(polygon.count<3u||!finite(geometry.area)||geometry.area<=1e-12){invalidatePowerFace(faceIndex,INVALID_GEOMETRY,64u);return;}
  var triangleAreas:array<f32,14>;var totalArea=0.0;
  for(var triangle=0u;triangle+2u<polygon.count;triangle+=1u){let area=0.5*abs(dot(cross(polygon.vertices[triangle+1u]-polygon.vertices[0],polygon.vertices[triangle+2u]-polygon.vertices[0]),geometry.normal));triangleAreas[triangle]=area;totalArea+=area;}
  if(!finite(totalArea)||totalArea<=1e-12||abs(totalArea-face.area)>max(2e-5,face.area*5e-4)){invalidatePowerFace(faceIndex,INVALID_GEOMETRY,128u);return;}
  let normal=normalize(geometry.normal);let reference=select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(normal.x)<0.75);
  let tangent=normalize(cross(reference,normal));let bitangent=cross(normal,tangent);let scale=sqrt(totalArea);
  faceQuadrature[faceIndex].centroidArea=vec4f(geometry.centroid,totalArea);
  for(var sample=0u;sample<${OCTREE_POWER_FACE_QUADRATURE_SAMPLES}u;sample+=1u){
    let targetArea=(f32(sample)+0.5)*totalArea/f32(${OCTREE_POWER_FACE_QUADRATURE_SAMPLES}u);var cumulative=0.0;var selected=0u;
    for(var triangle=0u;triangle+2u<polygon.count;triangle+=1u){let next=cumulative+triangleAreas[triangle];if(targetArea<=next||triangle+3u>=polygon.count){selected=triangle;break;}cumulative=next;}
    let area=max(triangleAreas[selected],1e-20);let u=clamp((targetArea-cumulative)/area,0.0,1.0);
    let v=fract((f32(sample)+0.5)*0.61803398875);let root=sqrt(u);
    let point=(1.0-root)*polygon.vertices[0]+root*(1.0-v)*polygon.vertices[selected+1u]+root*v*polygon.vertices[selected+2u];
    let offset=(point-geometry.centroid)/scale;let uv=vec2f(dot(offset,tangent),dot(offset,bitangent));
    if(any(abs(uv)>vec2f(65500.0))||!all(vec2<bool>(finite(uv.x),finite(uv.y)))){invalidatePowerFace(faceIndex,INVALID_GEOMETRY,256u);return;}
    faceQuadrature[faceIndex].sampleUV[sample]=pack2x16float(uv);
  }
  if((face.flags&OPEN_BOUNDARY)==0u){return;}
    let world=(face.flags>>WORLD_BOUNDARY_SHIFT)&63u;
    if(world!=0u){
      // World-open pressure is Dirichlet at the physical boundary plane.  All
      // points on the clipped polygon have the same signed normal distance,
      // so its reconstructed plane centroid is exact for this coefficient.
      let boundaryDistance=dot(geometry.centroid-rowCenter(face.negativeRow),geometry.normal);
      if(!finite(boundaryDistance)||boundaryDistance<=1e-12){invalidatePowerFace(faceIndex,INVALID_GEOMETRY,512u);return;}
      face.inverseDistance=1.0/boundaryDistance;powerFaces[faceIndex]=face;return;
    }
    // Paper Sections 4.1 and 5 require phi at the two actual cell centres.
    // Query construction is separate from sparse sampling to keep both
    // portable WebGPU layouts below ten bindings.  A missing authority is a
    // publication error, never an affine or sign-repaired substitute.
    if(params.phiPolicy.x==0u||geometry.neighborSize<=0.0){invalidatePowerFace(faceIndex,INVALID_GEOMETRY,1024u);return;}
    // liquidCenter.w is the immutable geometric inverse distance. Dynamic
    // generations always divide this base by their current theta, never the
    // previously published coefficient.
    boundaryPhiQueries[faceIndex]=BoundaryPhiQuery(vec4f(rowCenter(face.negativeRow),geometry.inverseDistance),vec4f(geometry.neighborCenter,1.0));
}
@compute @workgroup_size(1) fn finalizeAndPublishPowerFaces(){
  reduceRowFailures(min(control.rowCount,arrayLength(&rowDiagnostics)));
  if(control.flags==0u){
    if(arrayLength(&deltaStats)<4u||deltaStats[3]!=FACE_DELTA_VALID||control.faceCount>arrayLength(&powerFaces)){fail(0u,CAPACITY);}
    else{for(var faceIndex=0u;faceIndex<control.faceCount;faceIndex+=1u){let face=powerFaces[faceIndex];
      if((face.flags&FACE_FAILURE)!=0u){let failureFlag=bitcast<u32>(face.openFraction);
        control.flags|=failureFlag;control.invalidCount+=1u;
        if(control.firstInvalid==INVALID){control.firstInvalid=face.negativeRow;control.pad0=face.geometryCode&0xffffu;
          control.pad1=face.positiveRow;control.pad2=bitcast<u32>(face.normalVelocity);control.pad3=faceIndex;}break;}
      if((face.flags&FACE_VALID)==0u){fail(face.negativeRow,ASYMMETRIC_FACE);break;}
    }}
  }
  reduceRowFailures(min(control.rowCount,arrayLength(&rowDiagnostics)));
  let terminal=control.rowCount;let terminalValid=terminal<arrayLength(&rows)&&rows[terminal].faceCount==FACE_VALID;
  let nonempty=control.rowCount>0u&&control.faceCount>0u&&control.incidenceCount>0u;
  if(control.flags==0u&&control.invalidCount==0u&&terminalValid&&nonempty&&
      arrayLength(&deltaStats)>=4u&&deltaStats[3]==FACE_DELTA_VALID){control.valid=FACE_VALID;control.pad2=0u;}
  else{if(!nonempty){control.flags|=INVALID_HEADER;control.firstInvalid=min(control.firstInvalid,0u);control.invalidCount+=1u;}
    control.faceCount=0u;control.incidenceCount=0u;control.valid=0u;if(arrayLength(&liveFaceDispatch)>=6u){liveFaceDispatch[3]=0u;}}
}
@compute @workgroup_size(64) fn commitPowerFaceGeometry(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) workgroups:vec3u){
  if(control.valid!=FACE_VALID||control.flags!=0u){return;}
  let index=gid.x;let stride=max(1u,workgroups.x*64u);let faceCount=control.faceCount;
  for(var face=index;face<faceCount;face+=stride){
    if(face<arrayLength(&committedFaces)&&face<arrayLength(&powerFaces)){committedFaces[face]=powerFaces[face];}
    if(face<arrayLength(&committedNormals)&&face<arrayLength(&faceNormals)){committedNormals[face]=faceNormals[face];}
    if(face<arrayLength(&committedCentroids)&&face<arrayLength(&faceCentroids)){committedCentroids[face]=faceCentroids[face];}
    if(face<arrayLength(&committedQuadrature)&&face<arrayLength(&faceQuadrature)){committedQuadrature[face]=faceQuadrature[face];}
  }
}
@compute @workgroup_size(64) fn commitPowerFaceRowsAndIncidence(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) workgroups:vec3u){
  if(control.valid!=FACE_VALID||control.flags!=0u){return;}
  let index=gid.x;let stride=max(1u,workgroups.x*64u);let faceCount=control.faceCount;
  for(var face=index;face<faceCount;face+=stride){
    if(face<arrayLength(&committedBoundaryQueries)&&face<arrayLength(&boundaryPhiQueries)){committedBoundaryQueries[face]=boundaryPhiQueries[face];}
  }
  let incidenceCount=control.incidenceCount;
  for(var item=index;item<incidenceCount;item+=stride){if(item<arrayLength(&incidence)&&item<arrayLength(&committedIncidence)){committedIncidence[item]=incidence[item];}}
  let rowCount=control.rowCount;
  for(var row=index;row<=rowCount&&row<arrayLength(&rows)&&row<arrayLength(&committedRows);row+=stride){
    committedRows[row]=rows[row];
    if(row<rowCount&&row<arrayLength(&rowDirectory)&&row<arrayLength(&committedRowDirectory)){
      committedRowDirectory[row]=rowDirectory[row];
    }
  }
}
@compute @workgroup_size(1) fn commitPowerFaceControl(){
  if(control.valid!=FACE_VALID||control.flags!=0u){return;}
  committedControl=control;
}
`;

/**
 * Paper-facing free-surface pressure coefficient publication.  The liquid and
 * air query positions are the power-cell centres emitted by the geometry
 * kernel above.  Analytic sampling is used only for the authored cold field;
 * every recurring generation samples the current signed fine lattice.
 */
export const octreePowerBoundaryPhiShader = /* wgsl */ `
${octreePowerCatalogWGSL}
struct FaceParams { dimensionsRowCount:vec4u, capacitiesGeneration:vec4u, physical:vec4f, boundaryPolicy:vec4u, containerFill:vec4f, phiPolicy:vec4u }
struct FineParams {
  brickDimensions:vec3u, brickResolution:u32,
  sampleDimensions:vec3u, samplesPerBrick:u32,
  domainOrigin:vec3f, fineCellWidth:f32,
  worklistCapacity:u32, worklistHeaderWords:u32, pageCapacity:u32, generation:u32,
  activeCount:u32, invalid:u32, fineFactor:u32, timestep:f32,
}
struct BoundaryPhiQuery { liquidCenter:vec4f, airCenter:vec4f }
struct Control { rowCount:u32, faceCount:u32, incidenceCount:u32, flags:u32, firstInvalid:u32, invalidCount:u32, boundaryCount:u32, generation:u32, valid:u32, lookupMissCount:u32, maximumLookupSteps:u32, worldBoundaryCount:u32, pad0:u32, pad1:u32, pad2:u32, pad3:u32 }
@group(0) @binding(0) var<uniform> faceParams:FaceParams;
@group(0) @binding(1) var<uniform> fineParams:FineParams;
@group(0) @binding(2) var<storage,read> metadata:array<u32>;
@group(0) @binding(3) var<storage,read> worklist:array<u32>;
@group(0) @binding(4) var<storage,read> sampleFlags:array<u32>;
@group(0) @binding(5) var<storage,read> signedPhi:array<f32>;
@group(0) @binding(6) var<storage,read_write> powerFaces:array<PowerFaceRecord>;
@group(0) @binding(7) var<storage,read> queries:array<BoundaryPhiQuery>;
@group(0) @binding(8) var<storage,read_write> control:Control;
const INVALID:u32=0xffffffffu;
const SAMPLE_VALID:u32=1u;
const COARSE_PENDING:u32=${OCTREE_POWER_FACE_COARSE_PENDING}u;
const FACE_FAILURE:u32=0x20000000u;
const INVALID_GEOMETRY:u32=${OCTREE_POWER_FACE_ERROR.invalidGeometry}u;
fn finite(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn failBoundary(faceIndex:u32,detail:u32,liquid:f32,air:f32){
  var face=powerFaces[faceIndex];face.flags|=FACE_FAILURE;face.normalVelocity=bitcast<f32>(detail);
  face.area=liquid;face.inverseDistance=air;face.openFraction=bitcast<f32>(INVALID_GEOMETRY);powerFaces[faceIndex]=face;
}
fn packBrick(coord:vec3u)->u32{return coord.x+fineParams.brickDimensions.x*(coord.y+fineParams.brickDimensions.y*coord.z);}
${makeFineLevelSetSortedWorklistLookupWGSL("fineParams", "metadata", "worklist", "lookupBrick")}
fn loadFine(coord:vec3i)->vec2f{
  if(any(coord<vec3i(0))||any(coord>=vec3i(fineParams.sampleDimensions))){return vec2f(0.0);}
  let q=vec3u(coord);let brick=q/fineParams.brickResolution;let local=q-brick*fineParams.brickResolution;
  let id=lookupBrick(packBrick(brick));if(id==INVALID){return vec2f(0.0);}
  let localIndex=local.x+fineParams.brickResolution*(local.y+fineParams.brickResolution*local.z);
  let index=id*fineParams.samplesPerBrick+localIndex;
  if(index>=arrayLength(&sampleFlags)||index>=arrayLength(&signedPhi)||(sampleFlags[index]&SAMPLE_VALID)==0u){return vec2f(0.0);}
  let value=signedPhi[index];if(!finite(value)){return vec2f(0.0);}
  return vec2f(value,1.0);
}
fn sampleFine(position:vec3f)->vec2f{
  if(!(fineParams.fineCellWidth>0.0)||arrayLength(&worklist)<5u||worklist[0]==0u
    ||worklist[0]>fineParams.pageCapacity||worklist[1]!=fineParams.generation
    ||worklist[3]!=1u||worklist[4]!=1u){return vec2f(0.0);}
  let lattice=(position-fineParams.domainOrigin)/fineParams.fineCellWidth-vec3f(0.5);
  let maximum=vec3f(fineParams.sampleDimensions)-vec3f(1.0);
  if(any(lattice<vec3f(0.0))||any(lattice>maximum)){return vec2f(0.0);}
  let base=vec3i(floor(lattice));let fraction=fract(lattice);var value=0.0;
  for(var z=0;z<2;z+=1){for(var y=0;y<2;y+=1){for(var x=0;x<2;x+=1){
    let weight=select(1.0-fraction.x,fraction.x,x==1)*select(1.0-fraction.y,fraction.y,y==1)*select(1.0-fraction.z,fraction.z,z==1);
    if(weight==0.0){continue;}
    let sample=loadFine(base+vec3i(x,y,z));if(sample.y==0.0){return vec2f(0.0);}
    value+=weight*sample.x;
  }}}
  return vec2f(value,1.0);
}
fn boxDistance(point:vec3f,centre:vec3f,halfExtent:vec3f)->f32{
  let q=abs(point-centre)-halfExtent;
  return length(max(q,vec3f(0.0)))+min(max(q.x,max(q.y,q.z)),0.0);
}
fn sampleAnalytic(position:vec3f)->vec2f{
  let container=faceParams.containerFill.xyz;let fill=faceParams.containerFill.w;
  if(any(container<=vec3f(0.0))||fill<0.0||fill>1.0){return vec2f(0.0);}
  if(faceParams.phiPolicy.y==1u){return vec2f(position.y-fill*container.y,1.0);}
  if(faceParams.phiPolicy.y!=2u){return vec2f(0.0);}
  let heightFraction=max(0.92,fill);let footprintFraction=sqrt(fill/max(heightFraction,1e-9));
  let halfExtent=0.5*vec3f(footprintFraction*container.x,heightFraction*container.y,footprintFraction*container.z);
  return vec2f(boxDistance(position,halfExtent,halfExtent),1.0);
}
fn sampleAuthority(position:vec3f)->vec2f{
  if(faceParams.phiPolicy.x==1u){return sampleAnalytic(position);}
  if(faceParams.phiPolicy.x==2u){return sampleFine(position);}
  return vec2f(0.0);
}
@compute @workgroup_size(64)
fn samplePowerBoundaryPhi(@builtin(global_invocation_id) gid:vec3u){
  let faceIndex=gid.x;
  if(faceIndex>=control.faceCount||faceIndex>=arrayLength(&powerFaces)||faceIndex>=arrayLength(&queries)){return;}
  let query=queries[faceIndex];if(query.liquidCenter.w==0.0){return;}
  var face=powerFaces[faceIndex];let slot=face.geometryCode&0xffffu;
  let liquid=sampleAuthority(query.liquidCenter.xyz);let air=sampleAuthority(query.airCenter.xyz);
  if(liquid.y==0.0||air.y==0.0){
    // Paper Section 5: the fine band is undefined beyond a narrow band of the
    // interface, and the coarse octree level set is the cell-centre authority
    // there.  Defer this face to the coarse resolve pass when that authority
    // is bound; only its absence is a publication error.
    if(faceParams.phiPolicy.z==1u){powerFaces[faceIndex].flags=face.flags|COARSE_PENDING;return;}
    failBoundary(faceIndex,2048u,liquid.x,air.x);return;
  }
  // A pressure row and its signed-distance authority must describe the same
  // publication. Do not conceal a crossed row centre with a bounded
  // degenerate coefficient: a stale row set is a failed 2017 generation.
  // Pressure-frontier membership is phi < 0.  Its absent neighbour is
  // therefore the exact complementary phase phi >= 0, including the valid
  // endpoint where the free surface passes through the air cell centre.
  // That endpoint gives theta == 1 below and needs no sign repair or clamp.
  if(!finite(liquid.x)||!finite(air.x)||!(liquid.x<0.0)||air.x<0.0){
    failBoundary(faceIndex,4096u,liquid.x,air.x);return;
  }
  // Ghost-fluid zero crossing from the two current cell-centre samples used
  // by the embedded-boundary treatment described in Sections 4.1 and 5.
  let theta=(-liquid.x)/(air.x-liquid.x);
  if(!finite(theta)||!(theta>0.0)||theta>1.0){failBoundary(faceIndex,8192u,liquid.x,air.x);return;}
  face.inverseDistance=query.liquidCenter.w/theta;powerFaces[faceIndex]=face;
}
`;

/**
 * Coarse-octree resolve pass for boundary faces the fine band could not
 * sample (paper Section 5: "we also store and evolve a separate (coarse)
 * level set on the octree because our fine representation does not carry
 * inside/outside information beyond a narrow band of the free surface").
 * Both dual-edge endpoints are re-sampled from the published coarse
 * directory so one face never mixes fine and coarse authorities.
 */
export const octreePowerBoundaryCoarsePhiShader = /* wgsl */ `
struct BoundaryPhiQuery { liquidCenter:vec4f, airCenter:vec4f }
struct PowerFaceRecord { negativeRow:u32, positiveRow:u32, geometryCode:u32, flags:u32, normalVelocity:f32, area:f32, inverseDistance:f32, openFraction:f32 }
struct Control { rowCount:u32, faceCount:u32, incidenceCount:u32, flags:u32, firstInvalid:u32, invalidCount:u32, boundaryCount:u32, generation:u32, valid:u32, lookupMissCount:u32, maximumLookupSteps:u32, worldBoundaryCount:u32, pad0:u32, pad1:u32, pad2:u32, pad3:u32 }
@group(0) @binding(0) var<storage,read_write> powerFaces:array<PowerFaceRecord>;
@group(0) @binding(1) var<storage,read> queries:array<BoundaryPhiQuery>;
@group(0) @binding(2) var<storage,read_write> control:Control;
${makeOctreePowerCoarseLevelSetSampleWGSL(3)}
const COARSE_PENDING:u32=${OCTREE_POWER_FACE_COARSE_PENDING}u;
const FACE_FAILURE:u32=0x20000000u;
const INVALID_GEOMETRY:u32=${OCTREE_POWER_FACE_ERROR.invalidGeometry}u;
fn finite(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn coarseAvailable(value:f32)->bool{return finite(value)&&abs(value)<1e30;}
fn failBoundary(faceIndex:u32,detail:u32,liquid:f32,air:f32){
  var face=powerFaces[faceIndex];face.flags|=FACE_FAILURE;face.normalVelocity=bitcast<f32>(detail);
  face.area=liquid;face.inverseDistance=air;face.openFraction=bitcast<f32>(INVALID_GEOMETRY);powerFaces[faceIndex]=face;
}
@compute @workgroup_size(64)
fn resolveCoarsePowerBoundaryPhi(@builtin(global_invocation_id) gid:vec3u){
  let faceIndex=gid.x;
  if(faceIndex>=control.faceCount||faceIndex>=arrayLength(&powerFaces)||faceIndex>=arrayLength(&queries)){return;}
  var face=powerFaces[faceIndex];if((face.flags&COARSE_PENDING)==0u){return;}
  face.flags&=~COARSE_PENDING;powerFaces[faceIndex].flags=face.flags;
  let query=queries[faceIndex];let slot=face.geometryCode&0xffffu;
  if(query.liquidCenter.w==0.0){return;}
  let liquid=sampleCoarseOctreePhi(query.liquidCenter.xyz);
  let air=sampleCoarseOctreePhi(query.airCenter.xyz);
  if(!coarseAvailable(liquid)||!coarseAvailable(air)){failBoundary(faceIndex,2050u,liquid,air);return;}
  // Match the pressure-frontier predicate exactly: liquid rows satisfy
  // phi < 0 and an absent air row satisfies phi >= 0.  In particular,
  // air == 0 is the well-defined theta == 1 embedded-boundary endpoint.
  if(!(liquid<0.0)||air<0.0){failBoundary(faceIndex,4098u,liquid,air);return;}
  let theta=(-liquid)/(air-liquid);
  if(!finite(theta)||!(theta>0.0)||theta>1.0){failBoundary(faceIndex,8194u,liquid,air);return;}
  face.inverseDistance=query.liquidCenter.w/theta;powerFaces[faceIndex]=face;
}
`;

/** Deterministic compact GPU construction of generalized power faces. */

import { OCTREE_POWER_FACE_RECORD_BYTES, OCTREE_POWER_INVALID_ROW } from "./octree-power-operator";
import { octreePowerCatalogWGSL } from "./octree-power-wgsl";
import {
  OCTREE_POWER_TOPOLOGY_VALID,
  type OctreePowerTopologySource,
} from "./webgpu-octree-power-topology";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";

export const OCTREE_POWER_FACE_INCIDENCE_BYTES = 8;
export const OCTREE_POWER_FACE_CONTROL_BYTES = 64;
export const OCTREE_POWER_FACE_PARAMETER_BYTES = 96;
export const OCTREE_POWER_FACE_BOUNDARY_QUERY_BYTES = 32;
export const OCTREE_POWER_FACE_QUADRATURE_SAMPLES = 16;
// Exact centroid plus sixteen packed f16 tangent-plane coordinates. The
// samples are equal-area strata of the clipped world-space power polygon.
export const OCTREE_POWER_FACE_QUADRATURE_BYTES = 16 + OCTREE_POWER_FACE_QUADRATURE_SAMPLES * 4;
export const OCTREE_POWER_FACE_VALID = 0x8000_0000;
export const OCTREE_POWER_FACE_BOUNDARY = 1;
export const OCTREE_POWER_FACE_OPEN_BOUNDARY = 1 << 1;
export const OCTREE_POWER_FACE_WORLD_BOUNDARY_SHIFT = 8;
export const OCTREE_POWER_FACE_SCAN_BLOCK_SIZE = 256;
export const OCTREE_POWER_FACE_MAX_HASH_PROBES = 32;

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

export const OCTREE_POWER_FACE_ERROR = Object.freeze({
  invalidMetric: 1 << 0,
  invalidHeader: 1 << 1,
  invalidCatalog: 1 << 2,
  invalidGeometry: 1 << 3,
  asymmetricFace: 1 << 4,
  capacity: 1 << 5,
  siteIndex: 1 << 6,
  lookupMiss: 1 << 7,
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
  readonly hashCapacity: number;
  readonly hashBytes: number;
  readonly scanBlockCount: number;
  readonly scanBytes: number;
  readonly boundaryQueryBytes: number;
  readonly maximumHashProbes: number;
  readonly allocatedBytes: number;
}

export interface OctreePowerFaceEncodeOptions {
  readonly dimensions: readonly [number, number, number];
  /** CPU count or a COPY_SRC GPU buffer whose first u32 is the live compact-row count. */
  readonly rowCount: number | GPUBuffer;
  /** Physical width of one finest-grid cell. The topology stage requires it to be isotropic. */
  readonly physicalCellSize?: number;
  readonly generation?: number;
  /** Closed world faces in `x-, y-, z-, z+, y+, x+` bit order. */
  readonly closedBoundaryMask?: number;
  /**
   * Signed-distance authority for internal liquid/air pressure faces.  The
   * analytic mode is the authored t=0 field; fine mode is the current
   * published two-sided Section 5 band.  No estimate is substituted when a
   * requested sample is unavailable.
   */
  readonly boundaryPhi?: {
    readonly mode: "analytic" | "fine";
    readonly fine: Pick<WebGPUFineLevelSetBrickSource, "params" | "hash" | "metadata" | "worklist" | "flags" | "phi">;
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
  readonly maximumObservedProbe: number;
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
  /** Backward-compatible alias for incidenceRows; offsets are 16-byte-strided, not a packed u32 array. */
  readonly incidenceOffsets: GPUBuffer;
  /** Compact `(face:u32, sign:i32)` pairs. */
  readonly incidence: GPUBuffer;
  readonly control: GPUBuffer;
  readonly siteIndex: GPUBuffer;
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
  let hashCapacity = 1;
  while (hashCapacity < rowCapacity * 2) hashCapacity *= 2;
  const hashBytes = hashCapacity * 16;
  const scanBlockCount = Math.ceil(rowCapacity / OCTREE_POWER_FACE_SCAN_BLOCK_SIZE);
  const scanBytes = scanBlockCount * 16;
  const boundaryQueryBytes = faceCapacity * OCTREE_POWER_FACE_BOUNDARY_QUERY_BYTES;
  return {
    rowCapacity,
    faceCapacity,
    incidenceCapacity,
    faceBytes, normalBytes, centroidBytes, quadratureBytes,
    incidenceBytes,
    workspaceBytes, boundaryQueryBytes,
    hashCapacity,
    hashBytes,
    scanBlockCount,
    scanBytes,
    maximumHashProbes: OCTREE_POWER_FACE_MAX_HASH_PROBES,
    allocatedBytes: faceBytes + normalBytes + centroidBytes + quadratureBytes + incidenceBytes + workspaceBytes + hashBytes + scanBytes
      + boundaryQueryBytes
      + OCTREE_POWER_FACE_CONTROL_BYTES + OCTREE_POWER_FACE_PARAMETER_BYTES,
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
    maximumObservedProbe: Number(words[10]) >>> 0,
    worldBoundaryCount: Number(words[11]) >>> 0,
    firstInvalidSlot: Number(words[12]) >>> 0,
    firstInvalidNeighbor: Number(words[13]) >>> 0,
    firstInvalidDetail: Number(words[14]) >>> 0,
    firstInvalidRow: Number(words[15]) >>> 0,
  };
}

/**
 * Standalone dual-path builder. It consumes resolved PowerRowMetric records,
 * never descriptor words, so descriptor canonicalization remains owned by WP3.
 * Public face IDs are a row-major prefix sum followed by catalog-local order;
 * atomics are used only for fail-closed diagnostics.
 */
export class WebGPUOctreePowerFaces {
  readonly plan: OctreePowerFacePlan;
  readonly faces: GPUBuffer;
  readonly faceNormals: GPUBuffer;
  readonly faceCentroids: GPUBuffer;
  readonly faceQuadrature: GPUBuffer;
  readonly incidenceOffsets: GPUBuffer;
  readonly incidence: GPUBuffer;
  readonly control: GPUBuffer;
  readonly siteIndex: GPUBuffer;
  private readonly workspace: GPUBuffer;
  private readonly scanBlocks: GPUBuffer;
  private readonly boundaryQueries: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly clearIndexPipeline: GPUComputePipeline;
  private readonly buildIndexPipeline: GPUComputePipeline;
  private readonly countPipeline: GPUComputePipeline;
  private readonly scanRowsPipeline: GPUComputePipeline;
  private readonly scanBlocksPipeline: GPUComputePipeline;
  private readonly addBlockOffsetsPipeline: GPUComputePipeline;
  private readonly emitPipeline: GPUComputePipeline;
  private readonly sortIncidencePipeline: GPUComputePipeline;
  private readonly normalPipeline: GPUComputePipeline;
  private readonly quadraturePipeline: GPUComputePipeline;
  private readonly boundaryQueryPipeline: GPUComputePipeline;
  private readonly boundaryPhiPipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
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
    this.faceNormals = device.createBuffer({ label: "Octree power face normals", size: this.plan.normalBytes, usage: storage });
    this.faceCentroids = device.createBuffer({ label: "Octree power face centroids", size: this.plan.centroidBytes, usage: storage });
    this.faceQuadrature = device.createBuffer({ label: "Octree power face polygon quadrature", size: this.plan.quadratureBytes, usage: storage });
    this.incidence = device.createBuffer({ label: "Octree power face incidence", size: this.plan.incidenceBytes, usage: storage });
    this.workspace = device.createBuffer({ label: "Octree power face row workspace", size: this.plan.workspaceBytes, usage: storage });
    this.siteIndex = device.createBuffer({ label: "Octree power face site index", size: this.plan.hashBytes, usage: storage });
    this.scanBlocks = device.createBuffer({ label: "Octree power face scan blocks", size: this.plan.scanBytes, usage: storage });
    this.boundaryQueries = device.createBuffer({ label: "Octree power face boundary phi queries", size: this.plan.boundaryQueryBytes, usage: storage });
    // Offsets alias the workspace: RowWork.incidenceOffset is word 3 of each 16-byte row.
    // A separate compact copy would add memory and a sixth pass; consumers bind this
    // range with a 16-byte stride through the shared RowWork ABI.
    this.incidenceOffsets = this.workspace;
    this.control = device.createBuffer({ label: "Octree power face control", size: OCTREE_POWER_FACE_CONTROL_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Octree power face parameters", size: OCTREE_POWER_FACE_PARAMETER_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const shaderModule = device.createShaderModule({ label: "Octree power face construction", code: octreePowerFaceShader });
    const pipeline = (label: string, entryPoint: string) => device.createComputePipeline({
      label, layout: "auto", compute: { module: shaderModule, entryPoint },
    });
    this.preparePipeline = pipeline("Prepare octree power faces", "preparePowerFaces");
    this.clearIndexPipeline = pipeline("Clear octree power site index", "clearPowerSiteIndex");
    this.buildIndexPipeline = pipeline("Build octree power site index", "buildPowerSiteIndex");
    this.countPipeline = pipeline("Count octree power faces", "countPowerFaces");
    this.scanRowsPipeline = pipeline("Block scan octree power face rows", "scanPowerFaceRows");
    this.scanBlocksPipeline = pipeline("Scan octree power face block sums", "scanPowerFaceBlocks");
    this.addBlockOffsetsPipeline = pipeline("Add octree power face block offsets", "addPowerFaceBlockOffsets");
    this.emitPipeline = pipeline("Emit octree power faces", "emitPowerFaces");
    this.sortIncidencePipeline = pipeline("Sort octree power face incidence", "sortPowerIncidenceRows");
    this.normalPipeline = pipeline("Publish octree power face normals", "publishPowerFaceNormals");
    this.quadraturePipeline = pipeline("Publish octree power face polygon quadrature", "publishPowerFaceQuadrature");
    this.boundaryQueryPipeline = pipeline("Build octree power boundary phi queries", "buildPowerBoundaryPhiQueries");
    const boundaryPhiModule = device.createShaderModule({
      label: "Octree power boundary phi sampling",
      code: octreePowerBoundaryPhiShader,
    });
    this.boundaryPhiPipeline = device.createComputePipeline({
      label: "Sample octree power boundary phi", layout: "auto",
      compute: { module: boundaryPhiModule, entryPoint: "samplePowerBoundaryPhi" },
    });
    this.publishPipeline = pipeline("Publish octree power faces", "publishPowerFaces");
  }

  private writeParams(encoder: GPUCommandEncoder, options: OctreePowerFaceEncodeOptions): number {
    if (this.destroyed) throw new Error("Octree power-face builder is destroyed");
    options.dimensions.forEach((value, axis) => positiveInteger(value, `Power-face dimension ${axis}`));
    const volume = options.dimensions[0] * options.dimensions[1] * options.dimensions[2];
    if (!Number.isSafeInteger(volume) || volume > 0xffff_ffff) throw new RangeError("Power-face dimensions must fit the u32 cell ABI");
    if (typeof options.rowCount === "number"
      && (!Number.isSafeInteger(options.rowCount) || options.rowCount < 0 || options.rowCount > 0xffff_ffff)) {
      throw new RangeError("Power-face row count must be unsigned");
    }
    const generation = options.generation ?? 0;
    if (!Number.isSafeInteger(generation) || generation < 0 || generation > 0xffff_ffff) throw new RangeError("Power-face generation must be unsigned");
    const physicalCellSize = options.physicalCellSize ?? 1;
    if (!Number.isFinite(physicalCellSize) || physicalCellSize <= 0) throw new RangeError("Power-face physical cell size must be finite and positive");
    const closedBoundaryMask = options.closedBoundaryMask ?? OCTREE_POWER_CLOSED_BOUNDARY_MASK_WITH_CLOSED_TOP;
    if (!Number.isSafeInteger(closedBoundaryMask) || closedBoundaryMask < 0 || closedBoundaryMask > 63) {
      throw new RangeError("Power-face closed-boundary mask must be a six-bit unsigned integer");
    }
    const cpuRowCount = typeof options.rowCount === "number" ? options.rowCount : 0;
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([
      ...options.dimensions, cpuRowCount,
      this.plan.rowCapacity, this.plan.faceCapacity, this.plan.incidenceCapacity, generation,
    ]));
    this.device.queue.writeBuffer(this.params, 32, new Float32Array([
      physicalCellSize, this.plan.hashCapacity, this.plan.scanBlockCount, this.plan.maximumHashProbes,
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
      boundaryPhi?.mode === "analytic" ? 1 : boundaryPhi?.mode === "fine" ? 2 : 0,
      boundaryPhi?.initialCondition === "dam-break" ? 2 : boundaryPhi ? 1 : 0,
      0, 0,
    ]));
    if (typeof options.rowCount !== "number") encoder.copyBufferToBuffer(options.rowCount, 0, this.params, 12, 4);
    return typeof options.rowCount === "number"
      ? Math.min(options.rowCount, this.plan.rowCapacity)
      : this.plan.rowCapacity;
  }

  /** Build the metric-independent live `(origin,size)->row` index. */
  encodeSiteIndex(encoder: GPUCommandEncoder, leafHeaders: GPUBuffer, options: OctreePowerFaceEncodeOptions): void {
    const available = this.writeParams(encoder, options);
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const group = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries,
    });
    const run = (label: string, pipeline: GPUComputePipeline, groups: number, bindings: GPUBindGroupEntry[]) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group(pipeline, bindings)); pass.dispatchWorkgroups(groups); pass.end();
    };
    const params = { binding: 0, resource: resource(this.params) };
    const headers = { binding: 1, resource: resource(leafHeaders) };
    const control = { binding: 8, resource: resource(this.control) };
    const index = { binding: 9, resource: resource(this.siteIndex) };
    run("Prepare power face control", this.preparePipeline, 1, [params, control]);
    run("Clear power site index", this.clearIndexPipeline, Math.ceil(this.plan.hashCapacity / 64), [index]);
    if (available > 0) {
      const groups = Math.ceil(available / 64);
      run("Build power site index", this.buildIndexPipeline, groups, [params, headers, control, index]);
    }
  }

  encode(encoder: GPUCommandEncoder, leafHeaders: GPUBuffer, options: OctreePowerFaceEncodeOptions, siteIndexPrepared = false): void {
    const available = this.writeParams(encoder, options);
    if (!siteIndexPrepared) this.encodeSiteIndex(encoder, leafHeaders, options);
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const group = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries,
    });
    const run = (label: string, pipeline: GPUComputePipeline, groups: number, bindings: GPUBindGroupEntry[]) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group(pipeline, bindings)); pass.dispatchWorkgroups(groups); pass.end();
    };
    const params = { binding: 0, resource: resource(this.params) };
    const headers = { binding: 1, resource: resource(leafHeaders) };
    const metrics = { binding: 2, resource: resource(this.topology.metrics) };
    const entries = { binding: 3, resource: resource(this.topology.catalogEntryHeaders) };
    const catalog = { binding: 4, resource: resource(this.topology.catalogFaces) };
    const rows = { binding: 5, resource: resource(this.workspace) };
    const faces = { binding: 6, resource: resource(this.faces) };
    const incidence = { binding: 7, resource: resource(this.incidence) };
    const control = { binding: 8, resource: resource(this.control) };
    const index = { binding: 9, resource: resource(this.siteIndex) };
    const blocks = { binding: 10, resource: resource(this.scanBlocks) };
    const normals = { binding: 11, resource: resource(this.faceNormals) };
    const centroids = { binding: 12, resource: resource(this.faceCentroids) };
    const quadrature = { binding: 13, resource: resource(this.faceQuadrature) };
    const boundaryQueries = { binding: 14, resource: resource(this.boundaryQueries) };
    if (available > 0) {
      const groups = Math.ceil(available / 64);
      run("Count deterministic power faces", this.countPipeline, groups,
        [params, headers, metrics, entries, catalog, rows, control, index]);
    }
    run("Block scan deterministic power faces", this.scanRowsPipeline, this.plan.scanBlockCount, [params, rows, blocks]);
    run("Scan deterministic power face block sums", this.scanBlocksPipeline, 1,
      [params, rows, faces, incidence, control, blocks]);
    if (available > 0) {
      run("Add deterministic power face block offsets", this.addBlockOffsetsPipeline, Math.ceil(available / 64), [params, rows, blocks]);
      run("Emit deterministic power faces", this.emitPipeline, Math.ceil(available / 64),
        [params, headers, metrics, entries, catalog, rows, faces, incidence, index]);
      run("Sort deterministic power face incidence", this.sortIncidencePipeline, Math.ceil(available / 64),
        [params, rows, incidence, control]);
      run("Publish deterministic power face normals", this.normalPipeline,
        Math.ceil(this.plan.faceCapacity / 64), [params, headers, metrics, entries, catalog, faces, control, normals, centroids]);
      run("Publish deterministic power face polygon quadrature", this.quadraturePipeline,
        Math.ceil(this.plan.faceCapacity / 64), [params, headers, metrics, entries, catalog, faces, control, quadrature]);
      run("Build deterministic power boundary phi queries", this.boundaryQueryPipeline,
        Math.ceil(this.plan.faceCapacity / 64), [params, headers, metrics, entries, catalog, faces, control, boundaryQueries]);
      const boundaryPhi = options.boundaryPhi;
      if (boundaryPhi) {
        run("Sample deterministic power boundary phi", this.boundaryPhiPipeline,
          Math.ceil(this.plan.faceCapacity / 64), [
            { binding: 0, resource: resource(this.params) },
            { binding: 1, resource: resource(boundaryPhi.fine.params) },
            { binding: 2, resource: resource(boundaryPhi.fine.hash) },
            { binding: 3, resource: resource(boundaryPhi.fine.metadata) },
            { binding: 4, resource: resource(boundaryPhi.fine.worklist) },
            { binding: 5, resource: resource(boundaryPhi.fine.flags) },
            { binding: 6, resource: resource(boundaryPhi.fine.phi) },
            { binding: 7, resource: resource(this.faces) },
            { binding: 8, resource: resource(this.boundaryQueries) },
            { binding: 9, resource: resource(this.control) },
          ]);
      }
    }
    run("Publish power face control", this.publishPipeline, 1, [rows, control]);
  }

  get source(): OctreePowerFaceSource {
    return { plan: this.plan, faces: this.faces, faceNormals: this.faceNormals, faceCentroids: this.faceCentroids,
      faceQuadrature: this.faceQuadrature,
      incidenceRows: this.incidenceOffsets, incidenceOffsets: this.incidenceOffsets,
      incidence: this.incidence, control: this.control, siteIndex: this.siteIndex };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.faces.destroy(); this.faceNormals.destroy(); this.faceCentroids.destroy(); this.faceQuadrature.destroy(); this.incidence.destroy(); this.workspace.destroy(); this.siteIndex.destroy(); this.scanBlocks.destroy(); this.boundaryQueries.destroy();
    this.control.destroy(); this.params.destroy();
  }
}

export const octreePowerFaceShader = /* wgsl */ `
${octreePowerCatalogWGSL}
struct Params { dimensionsRowCount:vec4u, capacitiesGeneration:vec4u, physical:vec4f, boundaryPolicy:vec4u, containerFill:vec4f, phiPolicy:vec4u }
struct LeafHeader { cell:u32, entryStart:u32, entryCount:u32, size:u32, diagonal:f32, rhs:f32, pad0:u32, pad1:u32, gradient:vec4f }
struct PowerRowMetric { topologyCode:u32, transformAndFlags:u32, volume:f32, reserved:u32 }
struct CatalogEntry { firstFace:u32, faceCount:u32 }
struct RowWork { faceCount:u32, incidenceCount:u32, faceOffset:u32, incidenceOffset:u32 }
struct PowerIncidence { face:u32, sign:i32 }
struct SiteIndexEntry { cellPlusOne:atomic<u32>, size:u32, row:u32, published:atomic<u32> }
struct FaceQuadrature { centroidArea:vec4f, sampleUV:array<u32,${OCTREE_POWER_FACE_QUADRATURE_SAMPLES}> }
struct BoundaryPhiQuery { liquidCenter:vec4f, airCenter:vec4f }
struct Control { rowCount:atomic<u32>, faceCount:atomic<u32>, incidenceCount:atomic<u32>, flags:atomic<u32>, firstInvalid:atomic<u32>, invalidCount:atomic<u32>, boundaryCount:atomic<u32>, generation:atomic<u32>, valid:atomic<u32>, lookupMissCount:atomic<u32>, maximumObservedProbe:atomic<u32>, worldBoundaryCount:atomic<u32>, pad0:atomic<u32>, pad1:atomic<u32>, pad2:atomic<u32>, pad3:atomic<u32> }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(2) var<storage,read> metrics:array<PowerRowMetric>;
@group(0) @binding(3) var<storage,read> entries:array<CatalogEntry>;
@group(0) @binding(4) var<storage,read> catalogFaces:array<PowerCatalogFace>;
@group(0) @binding(5) var<storage,read_write> rows:array<RowWork>;
@group(0) @binding(6) var<storage,read_write> powerFaces:array<PowerFaceRecord>;
@group(0) @binding(7) var<storage,read_write> incidence:array<PowerIncidence>;
@group(0) @binding(8) var<storage,read_write> control:Control;
@group(0) @binding(9) var<storage,read_write> siteIndex:array<SiteIndexEntry>;
@group(0) @binding(10) var<storage,read_write> scanBlocks:array<RowWork>;
@group(0) @binding(11) var<storage,read_write> faceNormals:array<vec4f>;
@group(0) @binding(12) var<storage,read_write> faceCentroids:array<vec4f>;
@group(0) @binding(13) var<storage,read_write> faceQuadrature:array<FaceQuadrature>;
@group(0) @binding(14) var<storage,read_write> boundaryPhiQueries:array<BoundaryPhiQuery>;
const INVALID:u32=${OCTREE_POWER_INVALID_ROW}u;
const TOPOLOGY_VALID:u32=${OCTREE_POWER_TOPOLOGY_VALID}u;
const FACE_VALID:u32=${OCTREE_POWER_FACE_VALID}u;
const BOUNDARY:u32=${OCTREE_POWER_FACE_BOUNDARY}u;
const OPEN_BOUNDARY:u32=${OCTREE_POWER_FACE_OPEN_BOUNDARY}u;
const WORLD_BOUNDARY_SHIFT:u32=${OCTREE_POWER_FACE_WORLD_BOUNDARY_SHIFT}u;
const SCAN_BLOCK_SIZE:u32=${OCTREE_POWER_FACE_SCAN_BLOCK_SIZE}u;
const INVALID_METRIC:u32=${OCTREE_POWER_FACE_ERROR.invalidMetric}u;
const INVALID_HEADER:u32=${OCTREE_POWER_FACE_ERROR.invalidHeader}u;
const INVALID_CATALOG:u32=${OCTREE_POWER_FACE_ERROR.invalidCatalog}u;
const INVALID_GEOMETRY:u32=${OCTREE_POWER_FACE_ERROR.invalidGeometry}u;
const ASYMMETRIC_FACE:u32=${OCTREE_POWER_FACE_ERROR.asymmetricFace}u;
const CAPACITY:u32=${OCTREE_POWER_FACE_ERROR.capacity}u;
const SITE_INDEX:u32=${OCTREE_POWER_FACE_ERROR.siteIndex}u;
const LOOKUP_MISS:u32=${OCTREE_POWER_FACE_ERROR.lookupMiss}u;
fn finite(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn dims()->vec3u{return params.dimensionsRowCount.xyz;}
fn hashCapacity()->u32{return u32(params.physical.y);}
fn scanBlockCount()->u32{return u32(params.physical.z);}
fn maximumHashProbes()->u32{return u32(params.physical.w);}
fn cellCoord(cell:u32)->vec3u{let d=dims();return vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));}
fn rowHeaderValid(row:u32)->bool{if(row>=arrayLength(&headers)){return false;}let h=headers[row];let d=dims();if(h.size==0u||h.cell>=d.x*d.y*d.z){return false;}let o=cellCoord(h.cell);return all(o%vec3u(h.size)==vec3u(0u))&&all(vec3u(h.size)<=d)&&all(o<=d-vec3u(h.size));}
fn rowCenter(row:u32)->vec3f{let h=headers[row];return (vec3f(cellCoord(h.cell))+vec3f(0.5*f32(h.size)))*params.physical.x;}
fn rowSize(row:u32)->f32{return f32(headers[row].size)*params.physical.x;}
fn closeVector(a:vec3f,b:vec3f,scale:f32)->bool{return all(abs(a-b)<=vec3f(max(1e-5,scale*2e-5)));}
fn validReconstruction(face:ReconstructedPowerFace)->bool{return finite(face.neighborSize)&&face.neighborSize>=0.0&&finite(face.area)&&face.area>0.0&&finite(face.inverseDistance)&&face.inverseDistance>0.0&&all(vec3<bool>(finite(face.neighborCenter.x),finite(face.neighborCenter.y),finite(face.neighborCenter.z)))&&all(vec3<bool>(finite(face.centroid.x),finite(face.centroid.y),finite(face.centroid.z)))&&all(vec3<bool>(finite(face.normal.x),finite(face.normal.y),finite(face.normal.z)))&&abs(length(face.normal)-1.0)<=2e-4;}
fn hashSite(cell:u32,size:u32)->u32{var value=cell^(size*0x9e3779b9u);value=(value^(value>>16u))*0x7feb352du;value=(value^(value>>15u))*0x846ca68bu;return value^(value>>16u);}
fn neighborKey(center:vec3f,size:f32)->vec2u{let gridSize=size/params.physical.x;let origin=center/params.physical.x-vec3f(0.5*gridSize);let rounded=round(origin);if(abs(gridSize-round(gridSize))>2e-4||any(abs(origin-rounded)>vec3f(2e-4))||any(rounded<vec3f(0.0))||any(rounded>=vec3f(dims()))){return vec2u(INVALID);}let o=vec3u(rounded);return vec2u(o.x+dims().x*(o.y+dims().y*o.z),u32(round(gridSize)));}
fn findNeighbor(center:vec3f,size:f32)->vec2u{let key=neighborKey(center,size);if(key.x==INVALID){return vec2u(INVALID,0u);}let capacity=min(hashCapacity(),arrayLength(&siteIndex));if(capacity==0u){return vec2u(INVALID,0u);}let mask=capacity-1u;let bound=min(maximumHashProbes(),capacity);let base=hashSite(key.x,key.y)&mask;for(var probe=0u;probe<bound;probe+=1u){let slot=(base+probe)&mask;let observed=atomicLoad(&siteIndex[slot].cellPlusOne);if(observed==0u){return vec2u(INVALID,probe+1u);}if(observed==key.x+1u&&siteIndex[slot].size==key.y){return vec2u(siteIndex[slot].row,probe+1u);}}return vec2u(INVALID,bound+1u);}
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
fn fail(row:u32,flag:u32){atomicOr(&control.flags,flag);atomicMin(&control.firstInvalid,row);atomicAdd(&control.invalidCount,1u);}
fn failGeometry(row:u32,slot:u32,detail:u32){atomicOr(&control.flags,INVALID_GEOMETRY);let previous=atomicMin(&control.firstInvalid,row);if(row<previous){atomicStore(&control.pad0,slot);atomicStore(&control.pad1,INVALID);atomicStore(&control.pad2,detail);}atomicAdd(&control.invalidCount,1u);}
fn failSite(row:u32,slot:u32,neighbor:u32,detail:u32){atomicOr(&control.flags,SITE_INDEX);let previous=atomicMin(&control.firstInvalid,row);if(row<previous){atomicStore(&control.pad0,slot);atomicStore(&control.pad1,neighbor);atomicStore(&control.pad2,detail);atomicStore(&control.pad3,row);}atomicAdd(&control.invalidCount,1u);}
fn failFace(row:u32,slot:u32,neighbor:u32,detail:u32){atomicOr(&control.flags,ASYMMETRIC_FACE);let previous=atomicMin(&control.firstInvalid,row);if(row<previous){atomicStore(&control.pad0,slot);atomicStore(&control.pad1,neighbor);atomicStore(&control.pad2,detail);atomicStore(&control.pad3,row);}atomicAdd(&control.invalidCount,1u);}
@compute @workgroup_size(1) fn preparePowerFaces(){atomicStore(&control.rowCount,params.dimensionsRowCount.w);atomicStore(&control.faceCount,0u);atomicStore(&control.incidenceCount,0u);atomicStore(&control.flags,0u);atomicStore(&control.firstInvalid,INVALID);atomicStore(&control.invalidCount,0u);atomicStore(&control.boundaryCount,0u);atomicStore(&control.generation,params.capacitiesGeneration.w);atomicStore(&control.valid,0u);atomicStore(&control.lookupMissCount,0u);atomicStore(&control.maximumObservedProbe,0u);atomicStore(&control.worldBoundaryCount,0u);atomicStore(&control.pad0,INVALID);atomicStore(&control.pad1,INVALID);atomicStore(&control.pad2,0u);atomicStore(&control.pad3,INVALID);}
@compute @workgroup_size(64) fn clearPowerSiteIndex(@builtin(global_invocation_id) gid:vec3u){let slot=gid.x;if(slot>=arrayLength(&siteIndex)){return;}atomicStore(&siteIndex[slot].cellPlusOne,0u);siteIndex[slot].size=0u;siteIndex[slot].row=INVALID;atomicStore(&siteIndex[slot].published,0u);}
@compute @workgroup_size(64) fn buildPowerSiteIndex(@builtin(global_invocation_id) gid:vec3u){
  let row=gid.x;if(row>=params.dimensionsRowCount.w||row>=params.capacitiesGeneration.x){return;}if(!rowHeaderValid(row)){fail(row,INVALID_HEADER);return;}let header=headers[row];let capacity=min(hashCapacity(),arrayLength(&siteIndex));if(capacity==0u||(capacity&(capacity-1u))!=0u){failSite(row,INVALID,INVALID,1u);return;}let base=hashSite(header.cell,header.size)&(capacity-1u);var probe=0u;var attempts=0u;
  loop{if(probe>=min(maximumHashProbes(),capacity)||attempts>=maximumHashProbes()*4u){failSite(row,(base+probe)&(capacity-1u),INVALID,2u);return;}let slot=(base+probe)&(capacity-1u);let result=atomicCompareExchangeWeak(&siteIndex[slot].cellPlusOne,0u,header.cell+1u);attempts+=1u;if(result.exchanged){siteIndex[slot].size=header.size;siteIndex[slot].row=row;atomicStore(&siteIndex[slot].published,1u);atomicMax(&control.maximumObservedProbe,probe+1u);return;}if(result.old_value==0u){continue;}if(result.old_value==header.cell+1u){if(atomicLoad(&siteIndex[slot].published)==0u){continue;}let otherSize=siteIndex[slot].size;let otherRow=siteIndex[slot].row;if(otherSize!=header.size||otherRow!=row){failSite(row,slot,otherRow,select(8u,4u,otherSize!=header.size));}return;}probe+=1u;}}
@compute @workgroup_size(64) fn countPowerFaces(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;if(row>=params.dimensionsRowCount.w||row>=params.capacitiesGeneration.x||row>=arrayLength(&rows)){return;}rows[row]=RowWork(0u,0u,0u,0u);if(!rowHeaderValid(row)){fail(row,INVALID_HEADER);return;}if(row>=arrayLength(&metrics)){fail(row,INVALID_METRIC);return;}let metric=metrics[row];if((metric.transformAndFlags&TOPOLOGY_VALID)==0u||metric.topologyCode>=arrayLength(&entries)||!finite(metric.volume)||metric.volume<=0.0){fail(row,INVALID_METRIC);return;}let entry=entries[metric.topologyCode];if(entry.firstFace>arrayLength(&catalogFaces)||entry.faceCount>arrayLength(&catalogFaces)-entry.firstFace){fail(row,INVALID_CATALOG);return;}var owned=0u;for(var slot=0u;slot<entry.faceCount;slot+=1u){let face=reconstructSlot(row,slot);if(!validReconstruction(face)){fail(row,INVALID_GEOMETRY);return;}if(face.neighborSize==0.0){let world=worldBoundaryBitFromNormal(face.normal);if(world==0u||((((metric.transformAndFlags>>8u)&63u)&world)==0u)){fail(row,INVALID_GEOMETRY);return;}owned+=1u;atomicAdd(&control.boundaryCount,1u);atomicAdd(&control.worldBoundaryCount,1u);continue;}let found=findNeighbor(face.neighborCenter,face.neighborSize);atomicMax(&control.maximumObservedProbe,found.y);if(found.y>maximumHashProbes()){fail(row,LOOKUP_MISS);return;}let neighbor=found.x;if(neighbor==INVALID){owned+=1u;atomicAdd(&control.boundaryCount,1u);atomicAdd(&control.lookupMissCount,1u);let boundary=boundaryFlags(metric,face);if(((boundary>>WORLD_BOUNDARY_SHIFT)&63u)!=0u){atomicAdd(&control.worldBoundaryCount,1u);}}else{if(neighbor==row){fail(row,SITE_INDEX);return;}let reciprocal=reciprocalSlot(row,neighbor);if(reciprocal==INVALID){failFace(row,slot,neighbor,1u);return;}let reverse=reconstructSlot(neighbor,reciprocal);var detail=0u;if(abs(face.inverseDistance-reverse.inverseDistance)>max(1e-5,face.inverseDistance*2e-4)){detail|=4u;}if(dot(face.normal,reverse.normal)>-0.999){detail|=16u;}let intersectionGeometry=sharedGeometry(row,slot,neighbor,reciprocal);if(!finite(intersectionGeometry.area)||intersectionGeometry.area<=1e-10||!all(vec3<bool>(finite(intersectionGeometry.centroid.x),finite(intersectionGeometry.centroid.y),finite(intersectionGeometry.centroid.z)))){detail|=32u;}if(detail!=0u){failFace(row,slot,neighbor,detail);return;}if(row<neighbor){owned+=1u;}}}rows[row].faceCount=owned;rows[row].incidenceCount=entry.faceCount;}
var<workgroup> faceScan:array<u32,${OCTREE_POWER_FACE_SCAN_BLOCK_SIZE}>;
var<workgroup> incidenceScan:array<u32,${OCTREE_POWER_FACE_SCAN_BLOCK_SIZE}>;
@compute @workgroup_size(${OCTREE_POWER_FACE_SCAN_BLOCK_SIZE}) fn scanPowerFaceRows(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){let row=wid.x*SCAN_BLOCK_SIZE+lid;let available=min(params.dimensionsRowCount.w,min(params.capacitiesGeneration.x,arrayLength(&rows)-1u));var faceValue=0u;var incidenceValue=0u;if(row<available){faceValue=rows[row].faceCount;incidenceValue=rows[row].incidenceCount;}faceScan[lid]=faceValue;incidenceScan[lid]=incidenceValue;workgroupBarrier();for(var offset=1u;offset<SCAN_BLOCK_SIZE;offset<<=1u){var addFace=0u;var addIncidence=0u;if(lid>=offset){addFace=faceScan[lid-offset];addIncidence=incidenceScan[lid-offset];}workgroupBarrier();faceScan[lid]+=addFace;incidenceScan[lid]+=addIncidence;workgroupBarrier();}if(row<available){rows[row].faceOffset=faceScan[lid]-faceValue;rows[row].incidenceOffset=incidenceScan[lid]-incidenceValue;}if(lid==SCAN_BLOCK_SIZE-1u&&wid.x<arrayLength(&scanBlocks)){scanBlocks[wid.x]=RowWork(faceScan[lid],incidenceScan[lid],0u,0u);}}
@compute @workgroup_size(1) fn scanPowerFaceBlocks(){let requested=params.dimensionsRowCount.w;let terminal=arrayLength(&rows)-1u;let available=min(requested,min(params.capacitiesGeneration.x,terminal));rows[terminal].faceCount=0u;let blockCount=min(scanBlockCount(),arrayLength(&scanBlocks));var faceCursor=0u;var incidenceCursor=0u;for(var block=0u;block<blockCount;block+=1u){let faceCount=scanBlocks[block].faceCount;let incidenceCount=scanBlocks[block].incidenceCount;scanBlocks[block].faceOffset=faceCursor;scanBlocks[block].incidenceOffset=incidenceCursor;faceCursor+=faceCount;incidenceCursor+=incidenceCount;}rows[available]=RowWork(0u,0u,faceCursor,incidenceCursor);if(available<requested||faceCursor>params.capacitiesGeneration.y||faceCursor>arrayLength(&powerFaces)||incidenceCursor>params.capacitiesGeneration.z||incidenceCursor>arrayLength(&incidence)){atomicOr(&control.flags,CAPACITY);atomicMin(&control.firstInvalid,available);atomicAdd(&control.invalidCount,1u);return;}if(atomicLoad(&control.flags)==0u){rows[available].faceCount=FACE_VALID;rows[terminal].faceCount=FACE_VALID;atomicStore(&control.faceCount,faceCursor);atomicStore(&control.incidenceCount,incidenceCursor);}}
@compute @workgroup_size(64) fn addPowerFaceBlockOffsets(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;let available=min(params.dimensionsRowCount.w,min(params.capacitiesGeneration.x,arrayLength(&rows)-1u));if(row>=available){return;}let block=row/SCAN_BLOCK_SIZE;rows[row].faceOffset+=scanBlocks[block].faceOffset;rows[row].incidenceOffset+=scanBlocks[block].incidenceOffset;}
@compute @workgroup_size(64) fn emitPowerFaces(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;let available=min(params.dimensionsRowCount.w,min(params.capacitiesGeneration.x,arrayLength(&rows)-1u));if(row>=available||rows[available].faceCount!=FACE_VALID){return;}let metric=metrics[row];let entry=entries[metric.topologyCode];var localFace=0u;for(var slot=0u;slot<entry.faceCount;slot+=1u){var geometry=reconstructSlot(row,slot);let worldPlane=geometry.neighborSize==0.0;var neighbor=INVALID;if(!worldPlane){neighbor=findNeighbor(geometry.neighborCenter,geometry.neighborSize).x;}if(neighbor!=INVALID&&row>neighbor){continue;}let faceIndex=rows[row].faceOffset+localFace;let boundary=neighbor==INVALID;if(!boundary){let reciprocal=reciprocalSlot(row,neighbor);geometry=sharedGeometry(row,slot,neighbor,reciprocal);}let flags=FACE_VALID|select(0u,boundaryFlags(metric,geometry),boundary);let geometryCode=(slot&0xffffu)|((metric.transformAndFlags&63u)<<16u);powerFaces[faceIndex]=PowerFaceRecord(row,neighbor,geometryCode,flags,0.0,geometry.area,geometry.inverseDistance,1.0);incidence[rows[row].incidenceOffset+slot]=PowerIncidence(faceIndex,1);if(!boundary){let reciprocal=reciprocalSlot(row,neighbor);incidence[rows[neighbor].incidenceOffset+reciprocal]=PowerIncidence(faceIndex,-1);}localFace+=1u;}}
@compute @workgroup_size(64) fn sortPowerIncidenceRows(@builtin(global_invocation_id) gid:vec3u){let row=gid.x;let available=min(params.dimensionsRowCount.w,min(params.capacitiesGeneration.x,arrayLength(&rows)-1u));if(row>=available||rows[available].faceCount!=FACE_VALID){return;}let begin=rows[row].incidenceOffset;let count=rows[row].incidenceCount;if(begin>arrayLength(&incidence)||count>arrayLength(&incidence)-begin){fail(row,CAPACITY);return;}for(var local=1u;local<count;local+=1u){let value=incidence[begin+local];var cursor=local;while(cursor>0u&&incidence[begin+cursor-1u].face>value.face){incidence[begin+cursor]=incidence[begin+cursor-1u];cursor-=1u;}incidence[begin+cursor]=value;}}
@compute @workgroup_size(64) fn publishPowerFaceNormals(@builtin(global_invocation_id) gid:vec3u){let faceIndex=gid.x;
  if(faceIndex>=atomicLoad(&control.faceCount)||faceIndex>=arrayLength(&powerFaces)||faceIndex>=arrayLength(&faceNormals)){return;}
  var face=powerFaces[faceIndex];let slot=face.geometryCode&0xffffu;var geometry=reconstructSlot(face.negativeRow,slot);if(face.positiveRow!=INVALID){let reciprocal=reciprocalSlot(face.negativeRow,face.positiveRow);geometry=sharedGeometry(face.negativeRow,slot,face.positiveRow,reciprocal);}
  faceNormals[faceIndex]=vec4f(geometry.normal,0.0);faceCentroids[faceIndex]=vec4f(geometry.centroid,0.0);}
@compute @workgroup_size(64) fn publishPowerFaceQuadrature(@builtin(global_invocation_id) gid:vec3u){let faceIndex=gid.x;
  if(faceIndex>=atomicLoad(&control.faceCount)||faceIndex>=arrayLength(&powerFaces)||faceIndex>=arrayLength(&faceQuadrature)){return;}
  let face=powerFaces[faceIndex];let polygon=physicalFacePolygon(face);let slot=face.geometryCode&0xffffu;
  var geometry=reconstructSlot(face.negativeRow,slot);if(face.positiveRow!=INVALID){let reciprocal=reciprocalSlot(face.negativeRow,face.positiveRow);if(reciprocal!=INVALID){geometry=sharedGeometry(face.negativeRow,slot,face.positiveRow,reciprocal);}}
  if(polygon.count<3u||!finite(geometry.area)||geometry.area<=1e-12){failGeometry(face.negativeRow,slot,64u);return;}
  var triangleAreas:array<f32,14>;var totalArea=0.0;
  for(var triangle=0u;triangle+2u<polygon.count;triangle+=1u){let area=0.5*abs(dot(cross(polygon.vertices[triangle+1u]-polygon.vertices[0],polygon.vertices[triangle+2u]-polygon.vertices[0]),geometry.normal));triangleAreas[triangle]=area;totalArea+=area;}
  if(!finite(totalArea)||totalArea<=1e-12||abs(totalArea-face.area)>max(2e-5,face.area*5e-4)){failGeometry(face.negativeRow,slot,128u);return;}
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
    if(any(abs(uv)>vec2f(65500.0))||!all(vec2<bool>(finite(uv.x),finite(uv.y)))){failGeometry(face.negativeRow,slot,256u);return;}
    faceQuadrature[faceIndex].sampleUV[sample]=pack2x16float(uv);
  }}
@compute @workgroup_size(64) fn buildPowerBoundaryPhiQueries(@builtin(global_invocation_id) gid:vec3u){let faceIndex=gid.x;
  if(faceIndex>=atomicLoad(&control.faceCount)||faceIndex>=arrayLength(&powerFaces)||faceIndex>=arrayLength(&boundaryPhiQueries)){return;}
  boundaryPhiQueries[faceIndex]=BoundaryPhiQuery(vec4f(0.0),vec4f(0.0));var face=powerFaces[faceIndex];if((face.flags&OPEN_BOUNDARY)==0u){return;}let slot=face.geometryCode&0xffffu;let geometry=reconstructSlot(face.negativeRow,slot);
    let world=(face.flags>>WORLD_BOUNDARY_SHIFT)&63u;
    if(world!=0u){
      // World-open pressure is Dirichlet at the physical boundary plane.  All
      // points on the clipped polygon have the same signed normal distance,
      // so its reconstructed plane centroid is exact for this coefficient.
      let boundaryDistance=dot(geometry.centroid-rowCenter(face.negativeRow),geometry.normal);
      if(!finite(boundaryDistance)||boundaryDistance<=1e-12){failGeometry(face.negativeRow,slot,512u);return;}
      face.inverseDistance=1.0/boundaryDistance;powerFaces[faceIndex]=face;return;
    }
    // Paper Sections 4.1 and 5 require phi at the two actual cell centres.
    // Query construction is separate from sparse sampling to keep both
    // portable WebGPU layouts below ten bindings.  A missing authority is a
    // publication error, never an affine or sign-repaired substitute.
    if(params.phiPolicy.x==0u||geometry.neighborSize<=0.0){failGeometry(face.negativeRow,slot,1024u);return;}
    boundaryPhiQueries[faceIndex]=BoundaryPhiQuery(vec4f(rowCenter(face.negativeRow),1.0),vec4f(geometry.neighborCenter,1.0));}
@compute @workgroup_size(1) fn publishPowerFaces(){let terminal=arrayLength(&rows)-1u;let nonempty=atomicLoad(&control.rowCount)>0u&&atomicLoad(&control.faceCount)>0u&&atomicLoad(&control.incidenceCount)>0u;if(atomicLoad(&control.flags)==0u&&atomicLoad(&control.invalidCount)==0u&&rows[terminal].faceCount==FACE_VALID&&nonempty){atomicStore(&control.valid,FACE_VALID);}else{if(!nonempty){atomicOr(&control.flags,INVALID_HEADER);atomicMin(&control.firstInvalid,0u);atomicAdd(&control.invalidCount,1u);}atomicStore(&control.faceCount,0u);atomicStore(&control.incidenceCount,0u);atomicStore(&control.valid,0u);}}
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
  hashCapacity:u32, maximumHashProbes:u32, pageCapacity:u32, generation:u32,
  activeCount:u32, invalid:u32, fineFactor:u32, timestep:f32,
}
struct BoundaryPhiQuery { liquidCenter:vec4f, airCenter:vec4f }
struct Control { rowCount:atomic<u32>, faceCount:atomic<u32>, incidenceCount:atomic<u32>, flags:atomic<u32>, firstInvalid:atomic<u32>, invalidCount:atomic<u32>, boundaryCount:atomic<u32>, generation:atomic<u32>, valid:atomic<u32>, lookupMissCount:atomic<u32>, maximumObservedProbe:atomic<u32>, worldBoundaryCount:atomic<u32>, pad0:atomic<u32>, pad1:atomic<u32>, pad2:atomic<u32>, pad3:atomic<u32> }
@group(0) @binding(0) var<uniform> faceParams:FaceParams;
@group(0) @binding(1) var<uniform> fineParams:FineParams;
@group(0) @binding(2) var<storage,read> pageHash:array<u32>;
@group(0) @binding(3) var<storage,read> metadata:array<u32>;
@group(0) @binding(4) var<storage,read> worklist:array<u32>;
@group(0) @binding(5) var<storage,read> sampleFlags:array<u32>;
@group(0) @binding(6) var<storage,read> signedPhi:array<f32>;
@group(0) @binding(7) var<storage,read_write> powerFaces:array<PowerFaceRecord>;
@group(0) @binding(8) var<storage,read> queries:array<BoundaryPhiQuery>;
@group(0) @binding(9) var<storage,read_write> control:Control;
const INVALID:u32=0xffffffffu;
const SAMPLE_VALID:u32=1u;
const INVALID_GEOMETRY:u32=${OCTREE_POWER_FACE_ERROR.invalidGeometry}u;
fn finite(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn failBoundary(row:u32,slot:u32,detail:u32,liquid:f32,air:f32){
  atomicOr(&control.flags,INVALID_GEOMETRY);let previous=atomicMin(&control.firstInvalid,row);
  if(row<previous){atomicStore(&control.pad0,slot);atomicStore(&control.pad1,bitcast<u32>(liquid));atomicStore(&control.pad2,detail);atomicStore(&control.pad3,bitcast<u32>(air));}
  atomicAdd(&control.invalidCount,1u);
}
fn hashKey(key:u32)->u32{return ((key^(key>>16u))*0x9e3779b1u)&(fineParams.hashCapacity-1u);}
fn packBrick(coord:vec3u)->u32{return coord.x+fineParams.brickDimensions.x*(coord.y+fineParams.brickDimensions.y*coord.z);}
fn lookupBrick(key:u32)->u32{
  if(fineParams.hashCapacity==0u||(fineParams.hashCapacity&(fineParams.hashCapacity-1u))!=0u){return INVALID;}
  let start=hashKey(key);
  for(var probe=0u;probe<32u;probe+=1u){
    if(probe>=fineParams.maximumHashProbes){break;}
    let slot=(start+probe)&(fineParams.hashCapacity-1u);let stored=pageHash[slot*2u];
    if(stored==key){
      let id=pageHash[slot*2u+1u];if(id>=fineParams.pageCapacity){return INVALID;}
      let base=id*10u;
      if(metadata[base]!=id||metadata[base+1u]!=key||metadata[base+2u]!=fineParams.generation){return INVALID;}
      return id;
    }
    if(stored==INVALID){return INVALID;}
  }
  return INVALID;
}
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
  let faceIndex=gid.x;if(faceIndex>=atomicLoad(&control.faceCount)||faceIndex>=arrayLength(&powerFaces)||faceIndex>=arrayLength(&queries)){return;}
  let query=queries[faceIndex];if(query.liquidCenter.w==0.0){return;}
  var face=powerFaces[faceIndex];let slot=face.geometryCode&0xffffu;
  let liquid=sampleAuthority(query.liquidCenter.xyz);let air=sampleAuthority(query.airCenter.xyz);
  if(liquid.y==0.0||air.y==0.0){failBoundary(face.negativeRow,slot,2048u,liquid.x,air.x);return;}
  // Ghost Fluid Method interface fraction from the two signed cell-centre
  // samples required by paper Sections 4.1 and 5.  No abs repair or floor.
  if(!finite(liquid.x)||!finite(air.x)||!(liquid.x<0.0)||!(air.x>0.0)){
    failBoundary(face.negativeRow,slot,4096u,liquid.x,air.x);return;
  }
  let theta=(-liquid.x)/(air.x-liquid.x);
  if(!finite(theta)||!(theta>0.0)||theta>1.0){failBoundary(face.negativeRow,slot,8192u,liquid.x,air.x);return;}
  face.inverseDistance=face.inverseDistance/theta;powerFaces[faceIndex]=face;
}
`;

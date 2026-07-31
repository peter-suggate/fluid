/** Direct GPU publication of the six structured velocity families.
 *
 * This publisher resolves immutable catalog slots against the canonical leaf
 * headers. It never creates a general face record or an incidence stream.
 */

import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import type { OctreePowerRowDeltaSource } from "./webgpu-octree-power-descriptor";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import type { PassBroker } from "./webgpu-pass-broker";
import {
  createGPULogicalActivityAdoptionContext,
  type GPULogicalActivityAdoptionContext,
} from "./gpu-logical-activity-adoption";
import { performanceShaderVariant } from "./stores/performance-instrumentation-store";

export const OCTREE_SECTION63_CHANNELS = 19 as const;

export const OCTREE_STRUCTURED_GPU_VALID = 0x5356_454c;
export const OCTREE_STRUCTURED_GPU_CONTROL_BYTES = 128;
export const OCTREE_STRUCTURED_GPU_TRANSFER_LIST_OFFSET_WORDS =
  OCTREE_STRUCTURED_GPU_CONTROL_BYTES / 4;
export const OCTREE_STRUCTURED_GPU_WORKSET_HEADER_WORDS = 7;
/** Candidate-only indirect record for the compact changed-face transfer set. */
export const OCTREE_STRUCTURED_TOPOLOGY_TRANSFER_DISPATCH_OFFSET_BYTES = 72;
/** One minimum-portable WebGPU workgroup cooperatively finalizes all nine classes. */
export const OCTREE_STRUCTURED_FINALIZE_LANES = 256;
export const OCTREE_STRUCTURED_FINALIZE_WORKGROUP_BYTES =
  9 * OCTREE_STRUCTURED_FINALIZE_LANES * Uint32Array.BYTES_PER_ELEMENT;
const OCTREE_STRUCTURED_GPU_DISPATCH_BYTES =
  OCTREE_STRUCTURED_TOPOLOGY_TRANSFER_DISPATCH_OFFSET_BYTES + 36;

export const OCTREE_STRUCTURED_GPU_ERROR = Object.freeze({
  source: 1 << 0,
  capacity: 1 << 1,
  catalog: 1 << 2,
  neighbor: 1 << 3,
  reciprocity: 1 << 4,
  geometry: 1 << 5,
  carry: 1 << 6,
} as const);

export interface StructuredVelocityGPUPlan {
  readonly rowCapacity: number;
  readonly maximumCaseSlots: number;
  readonly slotCapacity: number;
  readonly authorityWords: number;
  readonly authorityBytes: number;
  readonly worksetStrideWords: number;
  readonly worksetBytes: number;
  readonly allocatedBytes: number;
  readonly offsets: Readonly<{
    values: number;
    ownerRows: number;
    neighborRows: number;
    metadata: number;
    areas: number;
    inverseDistances: number;
    fractions: number;
    pressureScales: number;
    normals: number;
    centroids: number;
    rowNeighbors: number;
    rowReciprocals: number;
    rowOwnerMetadata: number;
    rowSlotHandles: number;
    rowSlotSigns: number;
    rowCatalogSlots: number;
    rowAxisNeighbors: number;
    rowFamilyPrefix: number;
    rowFamilyHandles: number;
    rowFamilySlots: number;
  }>;
}

const positive = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be positive`);
  return value;
};

/**
 * Diagnostic only: encode the whole candidate publication N times instead of
 * once, so its marginal cost can be read off the wall clock.
 *
 * The chain is idempotent by construction -- `beginStructuredPublication`
 * recomputes `activeBank` from the *accepted* control, which no pass here
 * writes, and every later pass is a pure function of the same inputs -- so
 * repeating it leaves exactly the state one pass would have left. That makes
 * `(wall(N) - wall(1)) / (N - 1)` a direct wall-clock measurement of the
 * publication, which per-pass GPU timestamps alone cannot supply: a merged
 * Metal encoder charges inter-pass drain to whichever encoder it merged into.
 * Never set in production; ignored unless it parses to 2..8.
 */
export function structuredPublicationRepeatCount(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): number {
  const raw = environment?.FLUID_PROBE_STRUCTURED_PUBLICATION_REPEAT;
  if (raw === undefined) return 1;
  const value = Number(raw);
  return Number.isInteger(value) && value >= 2 && value <= 8 ? value : 1;
}

export function planStructuredVelocityGPU(rowCapacityValue: number,
  maximumCaseSlotsValue: number = OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence,
  storageAlignment = 256): StructuredVelocityGPUPlan {
  const rowCapacity = positive(rowCapacityValue, "Structured velocity row capacity");
  const maximumCaseSlots = positive(maximumCaseSlotsValue, "Structured velocity case-slot bound");
  if (maximumCaseSlots > 31) throw new RangeError("Structured velocity packed case slots exceed five bits");
  const slotCapacity = rowCapacity * maximumCaseSlots;
  const offsets = {} as Record<string, number>;
  let words = 0;
  const region = (name: string, count: number) => { offsets[name] = words; words += count; };
  region("values", slotCapacity);
  region("ownerRows", slotCapacity);
  region("neighborRows", slotCapacity);
  region("metadata", slotCapacity);
  region("areas", slotCapacity);
  region("inverseDistances", slotCapacity);
  region("fractions", slotCapacity);
  region("pressureScales", slotCapacity);
  region("normals", 4 * slotCapacity);
  region("centroids", 4 * slotCapacity);
  region("rowNeighbors", slotCapacity);
  region("rowReciprocals", slotCapacity);
  region("rowOwnerMetadata", slotCapacity);
  region("rowSlotHandles", slotCapacity);
  region("rowSlotSigns", slotCapacity);
  region("rowCatalogSlots", slotCapacity);
  region("rowAxisNeighbors", 6 * rowCapacity);
  region("rowFamilyPrefix", 6 * rowCapacity);
  region("rowFamilyHandles", 6 * rowCapacity);
  region("rowFamilySlots", 48 * rowCapacity);
  const alignmentWords = positive(storageAlignment, "Structured velocity storage alignment") / 4;
  const worksetStrideWords = Math.ceil((OCTREE_STRUCTURED_GPU_WORKSET_HEADER_WORDS
    + Math.max(rowCapacity, slotCapacity)) / alignmentWords) * alignmentWords;
  const worksetBytes = 9 * worksetStrideWords * 4;
  const authorityBytes = words * 4;
  return Object.freeze({
    rowCapacity, maximumCaseSlots, slotCapacity, authorityWords: words, authorityBytes,
    worksetStrideWords, worksetBytes,
    allocatedBytes: 2 * authorityBytes + 2 * worksetBytes + 2 * OCTREE_STRUCTURED_GPU_CONTROL_BYTES
      + slotCapacity * 4
      + 4 * rowCapacity * 16 + 2 * rowCapacity * OCTREE_SECTION63_CHANNELS * 4 + 256
      + OCTREE_STRUCTURED_GPU_DISPATCH_BYTES,
    offsets: Object.freeze(offsets) as StructuredVelocityGPUPlan["offsets"],
  });
}

/**
 * Largest 256-row arena whose packed A/B authority remains one valid storage
 * binding.  The structured publisher deliberately binds both banks together
 * so candidate shaders can carry accepted values without another storage
 * binding; the pressure capacity therefore has to be negotiated before any of
 * the row-indexed solver resources are allocated.
 */
export function structuredVelocityRowCapacityForBindingLimit(
  maximumBindingBytesValue: number,
  maximumCaseSlotsValue: number = OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence,
  rowAlignment = 256,
): number {
  const maximumBindingBytes = positive(maximumBindingBytesValue, "Structured velocity storage binding limit");
  const alignment = positive(rowAlignment, "Structured velocity row alignment");
  const bytesPerBankRow = planStructuredVelocityGPU(1, maximumCaseSlotsValue).authorityBytes;
  return Math.floor(Math.floor(maximumBindingBytes / (2 * bytesPerBankRow)) / alignment) * alignment;
}

export interface DirectStructuredVelocitySource {
  readonly plan: StructuredVelocityGPUPlan;
  readonly params: GPUBuffer;
  readonly authority: GPUBuffer;
  readonly control: GPUBuffer;
  /** Inactive publication control. Candidate-only consumers may populate the
   * inactive value bank before the coupled epoch validator accepts it. */
  readonly candidateControl: GPUBuffer;
  readonly rowVelocities: GPUBuffer;
  /** Fixed vec4u: cell-linear, size-in-finest-cells, physical page, local cell. */
  readonly rowGeometry: GPUBuffer;
  readonly authorityBankStrideWords: number;
  readonly rowBankStrideWords: number;
  readonly worksetBankStrideWords: number;
  readonly section63: Readonly<{
    rowCapacity: number;
    coefficients: GPUBuffer;
    coefficientBankStrideWords: number;
    worksetStrideWords: number;
    worksetBankStrideWords: number;
    classDispatch: GPUBuffer;
    classDispatchOffsetBytes: number;
    topologyMetrics: GPUBuffer;
    catalogCoefficients: GPUBuffer;
    catalogFaces: GPUBuffer;
    control: GPUBuffer;
    worksets: Readonly<Record<
      "regularInterior" | "transitionInterior" | "physicalBoundary" | "transitionBoundary",
      { readonly buffer: GPUBuffer; readonly offset: number; readonly size: number }
    >>;
  }>;
  readonly familyWorksets: Readonly<Record<
    "regularInterior" | "transitionInterior" | "physicalBoundary" | "transitionBoundary",
    { readonly buffer: GPUBuffer; readonly offset: number; readonly size: number }
  >>;
  readonly liveRowDispatch: GPUBuffer;
}

export interface DirectStructuredVelocityInputs {
  readonly leafHeaders: GPUBuffer;
  readonly topology: OctreePowerTopologySource;
  readonly rowDelta: OctreePowerRowDeltaSource;
  readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: number;
  readonly closedBoundaryMask: number;
}

/**
 * A/B GPU authority. Candidate metadata and carried values are complete before
 * `finalizeStructuredPublication` changes the active bank bit.
 */
export class WebGPUDirectStructuredVelocityAuthority {
  readonly plan: StructuredVelocityGPUPlan;
  readonly control: GPUBuffer;
  readonly rowVelocitiesA: GPUBuffer;
  readonly rowVelocitiesB: GPUBuffer;
  readonly section63Coefficients: GPUBuffer;
  readonly worksets: GPUBuffer;
  readonly rowGeometry: GPUBuffer;
  readonly params: GPUBuffer;
  readonly liveRowDispatch: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly authorityA: GPUBuffer;
  /** Inactive publication report consumed by the coupled epoch validator. */
  readonly candidateControl: GPUBuffer;
  private readonly beginPipeline: GPUComputePipeline;
  private readonly classifyPipeline: GPUComputePipeline;
  private readonly countFamiliesPipeline: GPUComputePipeline;
  private readonly prefixPipeline: GPUComputePipeline;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly section63Pipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private readonly reconstructPipeline: GPUComputePipeline;
  private readonly acceptPipeline: GPUComputePipeline;
  private readonly groupCache = new Map<GPUComputePipeline,
    Array<{ readonly bindings: readonly number[]; readonly buffers: readonly GPUBuffer[]; readonly group: GPUBindGroup }>>();
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly inputs: DirectStructuredVelocityInputs) {
    this.plan = planStructuredVelocityGPU(inputs.topology.plan.rowCapacity,
      OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence,
      device.limits.minStorageBufferOffsetAlignment);
    if (!Number.isFinite(inputs.physicalCellSize) || inputs.physicalCellSize <= 0
      || inputs.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
      || inputs.leafHeaders.size < this.plan.rowCapacity * 48) {
      throw new RangeError("Direct structured velocity inputs are invalid or undersized");
    }
    const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    if (2 * this.plan.authorityBytes > maximumBinding) {
      throw new RangeError("Direct structured velocity authority exceeds the storage binding limit");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.authorityA = device.createBuffer({ label: "Packed A/B structured velocity authority", size: 2 * this.plan.authorityBytes, usage: storage });
    this.control = device.createBuffer({ label: "Accepted structured velocity publication control", size: OCTREE_STRUCTURED_GPU_CONTROL_BYTES, usage: storage });
    // The trailing runtime array mirrors the compact class-4 transfer list.
    // Dynamics already binds this fail-closed transaction, keeping transfer
    // within Metal's ten-storage-buffer limit without a capacity dispatch.
    this.candidateControl = device.createBuffer({
      label: "Candidate structured velocity publication control and changed-face list",
      size: OCTREE_STRUCTURED_GPU_CONTROL_BYTES + this.plan.slotCapacity * 4,
      usage: storage,
    });
    this.rowVelocitiesA = device.createBuffer({ label: "Packed A/B structured cell velocities", size: 2 * this.plan.rowCapacity * 16, usage: storage });
    this.rowVelocitiesB = this.rowVelocitiesA;
    this.section63Coefficients = device.createBuffer({
      label: "Packed A/B Section 6.3 row coefficients",
      size: 2 * this.plan.rowCapacity * OCTREE_SECTION63_CHANNELS * 4,
      usage: storage,
    });
    this.worksets = device.createBuffer({ label: "Packed A/B structured row and family worksets",
      size: 2 * this.plan.worksetBytes, usage: storage | GPUBufferUsage.INDIRECT });
    this.rowGeometry = device.createBuffer({ label: "Packed A/B fixed structured row geometry",
      size: 2 * this.plan.rowCapacity * 16, usage: storage });
    this.params = device.createBuffer({ label: "Direct structured velocity parameters", size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.liveRowDispatch = device.createBuffer({ label: "Structured publication and accepted class dispatches",
      size: OCTREE_STRUCTURED_GPU_DISPATCH_BYTES,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE
        | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const publicationProfile = performanceShaderVariant();
    const publicationActivity = createGPULogicalActivityAdoptionContext({
      moduleId: "octree/structured-publication",
      profile: publicationProfile,
    });
    const publicationVariant = publicationActivity.module(
      directStructuredVelocityPublicationActivityShader(publicationActivity),
      `octree/structured-publication/${publicationProfile.cacheKey}`,
    );
    const activityModule = device.createShaderModule({ label: "Direct six-family structured velocity publication",
      code: publicationVariant.code });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: entryPoint,
      layout: "auto", compute: { module: activityModule, entryPoint } });
    this.beginPipeline = pipeline("beginStructuredPublication");
    this.classifyPipeline = publicationActivity.registerPipeline(
      pipeline("classifyStructuredCatalogSlots"),
    );
    this.countFamiliesPipeline = pipeline("countStructuredRowFamilies");
    this.prefixPipeline = pipeline("prefixStructuredFamilies");
    this.scatterPipeline = pipeline("scatterStructuredFamilySlots");
    this.section63Pipeline = pipeline("publishSection63Rows");
    this.finalizePipeline = pipeline("finalizeStructuredPublication");
    this.reconstructPipeline = pipeline("reconstructStructuredCellVelocity");
    this.acceptPipeline = pipeline("acceptStructuredPublication");
    const unpublishedBytes = new ArrayBuffer(OCTREE_STRUCTURED_GPU_CONTROL_BYTES);
    const unpublished = new Uint32Array(unpublishedBytes);
    unpublished[0] = OCTREE_STRUCTURED_GPU_ERROR.source;
    unpublished[1] = 0xffff_ffff;
    this.device.queue.writeBuffer(this.control, 0, unpublishedBytes);
    this.allocatedBytes = this.plan.allocatedBytes;
  }

  private parameterData(epoch: number, injectedFailure: number): ArrayBuffer {
    const bytes = new ArrayBuffer(256), words = new Uint32Array(bytes), floats = new Float32Array(bytes);
    words.set([
      this.plan.rowCapacity, this.plan.maximumCaseSlots, this.plan.slotCapacity, epoch >>> 0,
      ...this.inputs.dimensions, this.inputs.closedBoundaryMask >>> 0,
      this.inputs.topology.rowTemplateHeaderOffsetBytes / 4,
      this.inputs.topology.rowTemplateSlotOffsetBytes / 4,
      this.inputs.topology.rowTemplateDataOffsetBytes / 4,
      this.inputs.topology.reconstructionDataOffsetBytes / 4,
      this.inputs.topology.plan.entryCount,
      this.inputs.topology.catalogFaces.size / 48,
      this.plan.worksetStrideWords,
      this.plan.authorityWords,
    ], 0);
    words.set(Object.values(this.plan.offsets), 16);
    words.set([
      this.inputs.rowDelta.controlOffsetWords,
      this.inputs.rowDelta.newToOldOffsetWords,
      this.inputs.rowDelta.oldToNewOffsetWords,
      this.inputs.rowDelta.affectedRowsOffsetWords,
    ], 36);
    floats[40] = this.inputs.physicalCellSize;
    words[41] = injectedFailure >>> 0;
    return bytes;
  }

  private group(pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]): GPUBindGroup {
    const bindings: number[] = [], buffers: GPUBuffer[] = [];
    for (const entry of entries) {
      const resource = entry.resource as GPUBufferBinding;
      if (!("buffer" in resource)) throw new Error("Structured publication bindings must be buffers");
      bindings.push(entry.binding); buffers.push(resource.buffer);
    }
    let cached = this.groupCache.get(pipeline);
    if (!cached) { cached = []; this.groupCache.set(pipeline, cached); }
    const hit = cached.find((candidate) => candidate.bindings.length === bindings.length
      && candidate.bindings.every((binding, index) => binding === bindings[index]
        && candidate.buffers[index] === buffers[index]));
    if (hit) return hit.group;
    const group = this.device.createBindGroup({ label: `${pipeline.label} direct structured publication bindings`,
      layout: pipeline.getBindGroupLayout(0), entries });
    cached.push({ bindings, buffers, group });
    return group;
  }

  encodeCandidate(broker: PassBroker, epoch: number, injectedFailure = 0,
    topologyMetrics: GPUBuffer = this.inputs.topology.metrics,
    leafHeaders: GPUBuffer = this.inputs.leafHeaders): void {
    if (this.destroyed) throw new Error("Direct structured velocity authority is destroyed");
    if (!Number.isSafeInteger(epoch) || epoch <= 0 || epoch > 0xffff_ffff) {
      throw new RangeError("Structured velocity epoch must be a published uint32 generation");
    }
    for (let extra = structuredPublicationRepeatCount(); extra > 1; extra -= 1) {
      this.encodeCandidatePasses(broker, epoch, injectedFailure, topologyMetrics, leafHeaders);
    }
    this.encodeCandidatePasses(broker, epoch, injectedFailure, topologyMetrics, leafHeaders);
  }

  private encodeCandidatePasses(broker: PassBroker, epoch: number, injectedFailure: number,
    topologyMetrics: GPUBuffer, leafHeaders: GPUBuffer): void {
    const authority = this.authorityA;
    this.device.queue.writeBuffer(this.params, 0, this.parameterData(epoch, injectedFailure));
    const resource = (buffer: GPUBuffer) => ({ buffer });
    let pass = broker.compute({ label: "Begin direct structured publication" });
    pass.setPipeline(this.beginPipeline);
    pass.setBindGroup(0, this.group(this.beginPipeline, [
      { binding: 0, resource: resource(this.params) },
      { binding: 9, resource: resource(this.candidateControl) },
      { binding: 15, resource: resource(this.control) },
      // The row-delta control is the immutable candidate-row authority. The
      // general compaction arena is concurrently reused for dirty-tile work.
      { binding: 24, resource: resource(this.inputs.rowDelta.rows) },
      { binding: 25, resource: resource(this.liveRowDispatch) },
    ]));
    pass.dispatchWorkgroups(1);
    broker.fence("structured publication initialized");
    pass = broker.compute({ label: "Classify direct structured catalog slots" });
    pass.setPipeline(this.classifyPipeline);
    pass.setBindGroup(0, this.group(this.classifyPipeline, [
      { binding: 0, resource: resource(this.params) },
      { binding: 1, resource: resource(leafHeaders) },
      { binding: 2, resource: resource(topologyMetrics) },
      { binding: 4, resource: resource(this.inputs.topology.catalogFaces) },
      { binding: 5, resource: resource(this.inputs.topology.catalogVolumes) },
      { binding: 8, resource: resource(authority) },
      { binding: 9, resource: resource(this.candidateControl) },
    ]));
    // Byte offset 12 is the (row x catalog slot) record `begin` publishes, one
    // invocation per slot instead of one per row. It was computed and never
    // read until this call site existed.
    pass.dispatchWorkgroupsIndirect(this.liveRowDispatch, 12);
    // Per-row fold of the per-slot ownership metadata the classifier just
    // wrote. Same pass, so WebGPU already orders the two dispatches.
    pass.setPipeline(this.countFamiliesPipeline);
    pass.setBindGroup(0, this.group(this.countFamiliesPipeline, [
      { binding: 0, resource: resource(this.params) },
      { binding: 1, resource: resource(leafHeaders) },
      { binding: 2, resource: resource(topologyMetrics) },
      { binding: 5, resource: resource(this.inputs.topology.catalogVolumes) },
      { binding: 8, resource: resource(authority) },
      { binding: 9, resource: resource(this.candidateControl) },
    ]));
    pass.dispatchWorkgroupsIndirect(this.liveRowDispatch, 0);
    // No fence: `classify` -> `prefix` -> `scatter` hand off only `authority`
    // (binding 8) and `candidateControl` (binding 9). Both are bound at the
    // same binding index to the same buffer in all three bind groups, neither
    // is ever read as indirect arguments, and WebGPU orders dispatches within
    // one pass. `liveRowDispatch` -- the sole indirect source here -- is bound
    // (binding 25) only by `begin` and `finalize`, so nothing between them
    // feeds an indirect read. The previous note kept these two fences after a
    // collapse attempt "coincided with" an in-app publication rejection; that
    // was a coincidence, and the attempt measured under 1 ms on a 96 ms frame.
    // The `scatter` -> `section63` fence stays so the Section 6.3 publication
    // keeps its own pass label (PassBroker drops labels on an open pass).
    pass = broker.compute({ label: "Prefix six structured velocity families" });
    pass.setPipeline(this.prefixPipeline);
    pass.setBindGroup(0, this.group(this.prefixPipeline, [
      { binding: 0, resource: resource(this.params) },
      { binding: 8, resource: resource(authority) },
      { binding: 9, resource: resource(this.candidateControl) },
    ]));
    pass.dispatchWorkgroups(1);
    pass = broker.compute({ label: "Scatter direct structured velocity slots" });
    pass.setPipeline(this.scatterPipeline);
    pass.setBindGroup(0, this.group(this.scatterPipeline, [
      { binding: 0, resource: resource(this.params) },
      { binding: 1, resource: resource(leafHeaders) },
      { binding: 2, resource: resource(topologyMetrics) },
      { binding: 4, resource: resource(this.inputs.topology.catalogFaces) },
      { binding: 5, resource: resource(this.inputs.topology.catalogVolumes) },
      { binding: 6, resource: resource(this.inputs.rowDelta.rows) },
      { binding: 8, resource: resource(authority) },
      { binding: 9, resource: resource(this.candidateControl) },
    ]));
    pass.dispatchWorkgroupsIndirect(this.liveRowDispatch, 12);
    const compactPublicationPass = typeof process !== "undefined"
      && process.env.FLUID_STRUCTURED_PUBLICATION_COMPACT_PASS === "1";
    if (!compactPublicationPass) broker.fence("structured family values scattered");
    pass = broker.compute({ label: "Publish direct Section 6.3 rows and worksets" });
    pass.setPipeline(this.section63Pipeline);
    pass.setBindGroup(0, this.group(this.section63Pipeline, [
      { binding: 0, resource: resource(this.params) },
      { binding: 1, resource: resource(leafHeaders) },
      { binding: 2, resource: resource(topologyMetrics) },
      { binding: 4, resource: resource(this.inputs.topology.catalogFaces) },
      { binding: 5, resource: resource(this.inputs.topology.catalogVolumes) },
      { binding: 8, resource: resource(authority) },
      { binding: 9, resource: resource(this.candidateControl) },
      { binding: 10, resource: resource(this.section63Coefficients) },
      { binding: 11, resource: resource(this.inputs.topology.catalogCoefficients) },
    ]));
    pass.dispatchWorkgroupsIndirect(this.liveRowDispatch, 0);
    // Required: `finalize` rebinds `liveRowDispatch` as storage (binding 25)
    // and rewrites the class arguments the dispatches above are still reading
    // as indirect. Keep the write-after-read out of a shared pass.
    broker.fence("direct Section 6.3 rows published");
    pass = broker.compute({ label: "Finalize direct structured publication" });
    pass.setPipeline(this.finalizePipeline);
    pass.setBindGroup(0, this.group(this.finalizePipeline, [
      { binding: 0, resource: resource(this.params) },
      { binding: 2, resource: resource(topologyMetrics) },
      { binding: 6, resource: resource(this.inputs.rowDelta.rows) },
      { binding: 8, resource: resource(authority) },
      { binding: 9, resource: resource(this.candidateControl) },
      { binding: 12, resource: resource(this.worksets) },
      { binding: 25, resource: resource(this.liveRowDispatch) },
    ]));
    pass.dispatchWorkgroups(1);
    // Required: `finalize` writes the class arguments that the reconstruction
    // below consumes with `dispatchWorkgroupsIndirect`. A storage write feeding
    // an indirect read is the one ordering WebGPU does not give us for free.
    broker.fence("direct structured publication finalized");
    this.encodeCandidateReconstruction(broker, topologyMetrics, leafHeaders);
  }

  /** Rebuild the inactive row-vector field after a candidate-only face-value
   * transfer. Calling this twice is intentional: the first publication makes
   * candidate geometry available, while the second observes transferred
   * values before the coupled epoch can be accepted. */
  encodeCandidateReconstruction(broker: PassBroker,
    topologyMetrics: GPUBuffer = this.inputs.topology.metrics,
    leafHeaders: GPUBuffer = this.inputs.leafHeaders): void {
    if (this.destroyed) throw new Error("Direct structured velocity authority is destroyed");
    const authority = this.authorityA;
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const pass = broker.compute({ label: "Reconstruct direct structured cell velocities" });
    pass.setPipeline(this.reconstructPipeline);
    pass.setBindGroup(0, this.group(this.reconstructPipeline, [
      { binding: 0, resource: resource(this.params) },
      { binding: 2, resource: resource(topologyMetrics) },
      { binding: 5, resource: resource(this.inputs.topology.catalogVolumes) },
      { binding: 8, resource: resource(authority) },
      { binding: 9, resource: resource(this.candidateControl) },
      { binding: 13, resource: resource(this.rowVelocitiesA) },
      { binding: 20, resource: resource(this.rowGeometry) },
      { binding: 1, resource: resource(leafHeaders) },
    ]));
    pass.dispatchWorkgroupsIndirect(this.liveRowDispatch, 0);
    const compactReconstructionPass = typeof process !== "undefined"
      && process.env.FLUID_STRUCTURED_RECONSTRUCTION_COMPACT_PASS === "1";
    if (!compactReconstructionPass) broker.fence("structured candidate reconstruction complete");
  }

  encodeReadyCommit(broker: PassBroker): void {
    if (this.destroyed) throw new Error("Direct structured velocity authority is destroyed");
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const pass = broker.compute({ label: "Accept structured publication" });
    pass.setPipeline(this.acceptPipeline);
    pass.setBindGroup(0, this.group(this.acceptPipeline, [
      { binding: 9, resource: resource(this.candidateControl) }, { binding: 15, resource: resource(this.control) },
    ]));
    pass.dispatchWorkgroups(1);
  }

  encode(broker: PassBroker, epoch: number, injectedFailure = 0): void {
    this.encodeCandidate(broker, epoch, injectedFailure);
    this.encodeReadyCommit(broker);
  }

  get source(): DirectStructuredVelocitySource {
    const authority = this.authorityA;
    const rowVelocities = this.rowVelocitiesA;
    const binding = (index: number) => ({ buffer: this.worksets,
      offset: index * this.plan.worksetStrideWords * 4,
      size: this.plan.worksetStrideWords * 4 });
    return {
      plan: this.plan, params: this.params, authority, control: this.control,
      candidateControl: this.candidateControl, rowVelocities,
      rowGeometry: this.rowGeometry,
      authorityBankStrideWords: this.plan.authorityWords,
      rowBankStrideWords: this.plan.rowCapacity,
      worksetBankStrideWords: this.plan.worksetBytes / 4,
      section63: {
        rowCapacity: this.plan.rowCapacity,
        coefficients: this.section63Coefficients,
        coefficientBankStrideWords: this.plan.rowCapacity * OCTREE_SECTION63_CHANNELS,
        worksetStrideWords: this.plan.worksetStrideWords,
        worksetBankStrideWords: this.plan.worksetBytes / 4,
        classDispatch: this.liveRowDispatch,
        classDispatchOffsetBytes: 24,
        topologyMetrics: this.inputs.topology.metrics,
        catalogCoefficients: this.inputs.topology.catalogCoefficients,
        catalogFaces: this.inputs.topology.catalogFaces,
        worksets: {
          regularInterior: binding(0), transitionInterior: binding(1),
          physicalBoundary: binding(2), transitionBoundary: binding(3),
        },
        control: this.control,
      },
      familyWorksets: {
        regularInterior: binding(5), transitionInterior: binding(6),
        physicalBoundary: binding(7), transitionBoundary: binding(8),
      },
      liveRowDispatch: this.liveRowDispatch,
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of [this.authorityA, this.control, this.candidateControl,
      this.rowVelocitiesA, this.section63Coefficients, this.worksets,
      this.rowGeometry, this.params, this.liveRowDispatch]) buffer.destroy();
  }
}

export const directStructuredVelocityPublicationWGSL = /* wgsl */ `
struct CatalogSlotGeometry {
  neighborOffsetSize:vec4f,
  areaCentroid:vec4f,
  normalInverseDistance:vec4f,
}
struct ResolvedSlotGeometry {
  neighborCenter:vec3f,
  neighborSize:f32,
  centroid:vec3f,
  area:f32,
  normal:vec3f,
  inverseDistance:f32,
}
fn structuredTransformSigns(code:u32)->vec3f{
  let bits=code&7u;
  return vec3f(select(1.0,-1.0,(bits&1u)!=0u),select(1.0,-1.0,(bits&2u)!=0u),select(1.0,-1.0,(bits&4u)!=0u));
}
fn inverseStructuredTransform(value:vec3f,code:u32)->vec3f{
  let q=value*structuredTransformSigns(code);let permutation=(code/8u)%6u;
  if(permutation==0u){return q.xyz;}if(permutation==1u){return q.xzy;}
  if(permutation==2u){return q.yxz;}if(permutation==3u){return q.zxy;}
  if(permutation==4u){return q.yzx;}return q.zyx;
}
fn reconstructStructuredCatalogSlot(anchorCenter:vec3f,anchorSize:f32,slot:CatalogSlotGeometry,transform:u32)->ResolvedSlotGeometry{
  let neighborOffset=inverseStructuredTransform(slot.neighborOffsetSize.xyz,transform);
  let centroidOffset=inverseStructuredTransform(slot.areaCentroid.yzw,transform);
  let normal=normalize(inverseStructuredTransform(slot.normalInverseDistance.xyz,transform));
  return ResolvedSlotGeometry(anchorCenter+anchorSize*neighborOffset,anchorSize*slot.neighborOffsetSize.w,
    anchorCenter+anchorSize*centroidOffset,anchorSize*anchorSize*slot.areaCentroid.x,normal,
    slot.normalInverseDistance.w/anchorSize);
}
struct Params {
  rowCapacity:u32,maxSlots:u32,slotCapacity:u32,epoch:u32,
  dimensions:vec3u,closedMask:u32,
  templateHeaderOffset:u32,templateSlotOffset:u32,templateDataOffset:u32,reconstructionOffset:u32,
  catalogEntryCount:u32,catalogFaceCount:u32,worksetStride:u32,authorityWords:u32,
  valuesOffset:u32,ownerOffset:u32,neighborOffset:u32,metadataOffset:u32,
  areaOffset:u32,inverseOffset:u32,fractionOffset:u32,pressureScaleOffset:u32,normalOffset:u32,
  centroidOffset:u32,rowNeighborOffset:u32,rowReciprocalOffset:u32,rowOwnerMetadataOffset:u32,
  rowHandleOffset:u32,rowSignOffset:u32,rowCatalogOffset:u32,rowAxisOffset:u32,rowFamilyPrefixOffset:u32,
  rowFamilyHandleOffset:u32,rowFamilySlotOffset:u32,
  deltaControlOffset:u32,newToOldOffset:u32,oldToNewOffset:u32,affectedRowsOffset:u32,
  cellSize:f32,injectedFailure:u32,pad1:u32,pad2:u32,
}
struct LeafHeader {cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,flags:u32,reserved:u32,gradient:vec4f}
struct Metric {caseId:u32,transformAndFlags:u32,volume:f32,error:u32}
struct Control {flags:atomic<u32>,firstError:atomic<u32>,rowCount:u32,slotCount:u32,epoch:u32,activeBank:u32,projected:u32,entryCount:atomic<u32>,familyOffsets:array<u32,7>,reserved:array<u32,17>,transferHandles:array<u32>}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>headers:array<LeafHeader>;
@group(0)@binding(2)var<storage,read>metrics:array<Metric>;
@group(0)@binding(4)var<storage,read>catalogSlots:array<CatalogSlotGeometry>;
@group(0)@binding(5)var<storage,read>dense:array<u32>;
@group(0)@binding(6)var<storage,read>rowDelta:array<u32>;
@group(0)@binding(8)var<storage,read_write>authority:array<u32>;
@group(0)@binding(9)var<storage,read_write>control:Control;
@group(0)@binding(10)var<storage,read_write>section63Coefficients:array<f32>;
@group(0)@binding(11)var<storage,read>catalogCoefficients:array<f32>;
@group(0)@binding(12)var<storage,read_write>worksets:array<u32>;
@group(0)@binding(13)var<storage,read_write>rowVelocities:array<vec4f>;
@group(0)@binding(15)var<storage,read_write>acceptedControl:array<atomic<u32>>;
@group(0)@binding(20)var<storage,read_write>rowGeometry:array<vec4u>;
@group(0)@binding(24)var<storage,read>sourceRowDelta:array<u32>;
@group(0)@binding(25)var<storage,read_write>publicationDispatch:array<u32>;
const INVALID:u32=0xffffffffu;const VALID:u32=0x80000000u;const ERROR_SOURCE:u32=${OCTREE_STRUCTURED_GPU_ERROR.source}u;
const ERROR_CAPACITY:u32=${OCTREE_STRUCTURED_GPU_ERROR.capacity}u;const ERROR_CATALOG:u32=${OCTREE_STRUCTURED_GPU_ERROR.catalog}u;
const ERROR_NEIGHBOR:u32=${OCTREE_STRUCTURED_GPU_ERROR.neighbor}u;const ERROR_RECIPROCITY:u32=${OCTREE_STRUCTURED_GPU_ERROR.reciprocity}u;
const ERROR_GEOMETRY:u32=${OCTREE_STRUCTURED_GPU_ERROR.geometry}u;const ERROR_CARRY:u32=${OCTREE_STRUCTURED_GPU_ERROR.carry}u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn fail(row:u32,flag:u32){atomicOr(&control.flags,flag);atomicMin(&control.firstError,row);}
fn coord(cell:u32)->vec3u{return vec3u(cell%p.dimensions.x,(cell/p.dimensions.x)%p.dimensions.y,cell/(p.dimensions.x*p.dimensions.y));}
fn center(row:u32)->vec3f{let h=headers[row];return(vec3f(coord(h.cell))+.5*f32(h.size))*p.cellSize;}
fn size(row:u32)->f32{return f32(headers[row].size)*p.cellSize;}
fn mortonPart10(v:u32)->u32{var x=v&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn morton(cell:u32)->u32{let q=coord(cell);return mortonPart10(q.x)|(mortonPart10(q.y)<<1u)|(mortonPart10(q.z)<<2u);}
fn level(s:u32)->u32{return 31u-countLeadingZeros(s);}
fn less(la:u32,ma:u32,lb:u32,mb:u32)->bool{return la<lb||(la==lb&&ma<mb);}
fn findRow(point:vec3f,s:f32)->u32{let grid=s/p.cellSize;let o=round(point/p.cellSize-.5*grid);
  if(!finite(grid)||abs(grid-round(grid))>2e-4||any(o<vec3f(0.0))||any(o>=vec3f(p.dimensions))){return INVALID;}
  let u=vec3u(o);let cell=u.x+p.dimensions.x*(u.y+p.dimensions.y*u.z);let wanted=u32(round(grid));
  var low=0u;var high=min(control.rowCount,arrayLength(&headers));let l=level(wanted);let m=morton(cell);
  while(low<high){let mid=low+(high-low)/2u;let h=headers[mid];if(less(level(h.size),morton(h.cell),l,m)){low=mid+1u;}else{high=mid;}}
  if(low<control.rowCount){let h=headers[low];if(h.cell==cell&&h.size==wanted){return low;}}return INVALID;}
fn caseHeader(caseId:u32)->vec2u{let at=p.templateHeaderOffset+4u*caseId;return vec2u(dense[at],dense[at+1u]);}
fn templatePacked(globalSlot:u32)->u32{return dense[p.templateSlotOffset+globalSlot];}
fn geometry(row:u32,local:u32)->ResolvedSlotGeometry{let m=metrics[row];let h=caseHeader(m.caseId);return reconstructStructuredCatalogSlot(center(row),size(row),catalogSlots[h.x+local],m.transformAndFlags&63u);}
fn reciprocal(row:u32,neighbor:u32)->u32{let m=metrics[neighbor];if((m.transformAndFlags&VALID)==0u||m.caseId>=p.catalogEntryCount){return INVALID;}let h=caseHeader(m.caseId);let wanted=center(row);let wantedSize=size(row);
  for(var local=0u;local<h.y;local+=1u){let g=geometry(neighbor,local);if(abs(g.neighborSize-wantedSize)<=max(1e-5,wantedSize*2e-5)&&all(abs(g.neighborCenter-wanted)<=vec3f(max(1e-5,wantedSize*2e-5)))){return local;}}return INVALID;}
fn rowBase(row:u32)->u32{return row*p.maxSlots;}
fn familyIndex(row:u32,family:u32)->u32{return row*6u+family;}
fn worksetBase(c:u32)->u32{return c*p.worksetStride;}
fn candidateAuthorityBase()->u32{return control.activeBank*p.authorityWords;}
fn acceptedAuthorityBase()->u32{return (1u-control.activeBank)*p.authorityWords;}
fn section63BankBase()->u32{return control.activeBank*p.rowCapacity*${OCTREE_SECTION63_CHANNELS}u;}
fn rowBankBase()->u32{return control.activeBank*p.rowCapacity;}
fn worksetBankBase()->u32{return control.activeBank*9u*p.worksetStride;}
fn rowClass(row:u32)->u32{let m=metrics[row];let transition=m.caseId!=0u;let boundary=(m.transformAndFlags&0x3f00u)!=0u;return select(select(0u,2u,boundary),select(1u,3u,boundary),transition);}

// Capacity-shaped dispatch records are two-dimensional. X is pinned at exactly
// 65,535 workgroups once the block count saturates one dimension, which is
// what lets a kernel recover its linear item from a constant stride instead of
// a uniform; below saturation Y is 1 and the stride never applies. The
// slot-shaped records need this: slots are rows * maximumFaceIncidence, so at
// the catalog's 30 they pass the one-dimensional limit near 140,000 rows.
fn publishBlockDispatch(at:u32,blocks:u32){let x=max(1u,min(65535u,blocks));
 publicationDispatch[at]=x;publicationDispatch[at+1u]=(blocks+x-1u)/x;publicationDispatch[at+2u]=1u;}
fn publishExactRowDispatch(at:u32,rows:u32){if(rows==0u){publicationDispatch[at]=0u;
 publicationDispatch[at+1u]=1u;publicationDispatch[at+2u]=1u;return;}publishBlockDispatch(at,rows);}
fn foldedItem(g:vec3u)->u32{return g.x+g.y*65535u*64u;}
@compute @workgroup_size(1)fn beginStructuredPublication(){let rows=min(sourceRowDelta[p.deltaControlOffset],p.rowCapacity);let hasAccepted=arrayLength(&acceptedControl)>=6u&&atomicLoad(&acceptedControl[0])==0u&&atomicLoad(&acceptedControl[3])!=0u;atomicStore(&control.flags,p.injectedFailure);atomicStore(&control.firstError,select(INVALID,0u,p.injectedFailure!=0u));control.rowCount=rows;control.slotCount=0u;control.epoch=0u;control.activeBank=select(0u,1u-(atomicLoad(&acceptedControl[4])&1u),hasAccepted);control.projected=select(0u,1u,hasAccepted);atomicStore(&control.entryCount,0u);for(var family=0u;family<7u;family+=1u){control.familyOffsets[family]=0u;}publishBlockDispatch(0u,(rows+63u)/64u);publishBlockDispatch(3u,(rows*p.maxSlots+63u)/64u);}

// One invocation per (row, catalog slot). The row-level guards are re-evaluated
// per slot rather than hoisted: \`fail\` is an atomicOr plus an atomicMin on the
// same row index, so N slots reporting one row's failure leave exactly the
// flags and first-error one row-owning thread left.
@compute @workgroup_size(64)fn classifyStructuredCatalogSlots(@builtin(global_invocation_id)g:vec3u){let rows=min(control.rowCount,p.rowCapacity);let slot=foldedItem(g);let row=slot/p.maxSlots;let local=slot%p.maxSlots;if(row>=rows){return;}
  if(row>=arrayLength(&headers)||row>=arrayLength(&metrics)){fail(row,ERROR_CAPACITY);return;}let m=metrics[row];if((m.transformAndFlags&VALID)==0u||m.error!=0u||m.caseId>=p.catalogEntryCount){fail(row,ERROR_SOURCE);return;}
  let h=caseHeader(m.caseId);if(h.y==0u||h.y>p.maxSlots||h.x>p.catalogFaceCount||h.y>p.catalogFaceCount-h.x){fail(row,ERROR_CATALOG);return;}
  // \`control\` is read_write with atomic members, so a compiler may not lift
  // \`candidateAuthorityBase()\` out of the body by itself. Nothing here writes
  // \`activeBank\`, so lifting it by hand is value-identical.
  let base=candidateAuthorityBase();let at=rowBase(row)+local;
  authority[base+p.rowNeighborOffset+at]=INVALID;authority[base+p.rowReciprocalOffset+at]=INVALID;authority[base+p.rowOwnerMetadataOffset+at]=0u;authority[base+p.rowHandleOffset+at]=INVALID;authority[base+p.rowCatalogOffset+at]=INVALID;if(local>=h.y){return;}
  let global=h.x+local;let packed=templatePacked(global);let family=(packed>>8u)&7u;let orientation=(packed>>11u)&7u;if(family>=6u){fail(row,ERROR_CATALOG);return;}let geo=geometry(row,local);if(!finite(geo.area)||geo.area<=0.0||!finite(geo.inverseDistance)||geo.inverseDistance<=0.0||abs(length(geo.normal)-1.0)>4e-4){fail(row,ERROR_GEOMETRY);return;}
  var neighbor=INVALID;var reverse=INVALID;if(geo.neighborSize>0.0){neighbor=findRow(geo.neighborCenter,geo.neighborSize);if(neighbor!=INVALID){if(neighbor==row){fail(row,ERROR_NEIGHBOR);return;}reverse=reciprocal(row,neighbor);if(reverse==INVALID){fail(row,ERROR_RECIPROCITY);return;}}}
  let owner=neighbor==INVALID||row<neighbor;authority[base+p.rowNeighborOffset+at]=neighbor;authority[base+p.rowReciprocalOffset+at]=reverse;authority[base+p.rowOwnerMetadataOffset+at]=family|(orientation<<3u)|(select(0u,1u,owner)<<6u)|(select(0u,1u,neighbor==INVALID)<<7u);authority[base+p.rowCatalogOffset+at]=global;}

// The per-row reduction the classifier used to fold in place. One thread still
// owns each row's six family counters, so the accumulation order is the slot
// order it always was -- no atomics, and therefore no atomics-as-ordering.
@compute @workgroup_size(64)fn countStructuredRowFamilies(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=control.rowCount||row>=p.rowCapacity){return;}
  if(row>=arrayLength(&headers)||row>=arrayLength(&metrics)){return;}let m=metrics[row];if((m.transformAndFlags&VALID)==0u||m.error!=0u||m.caseId>=p.catalogEntryCount){return;}
  let h=caseHeader(m.caseId);if(h.y==0u||h.y>p.maxSlots||h.x>p.catalogFaceCount||h.y>p.catalogFaceCount-h.x){return;}
  let base=candidateAuthorityBase();var counts:array<u32,6>;
  for(var local=0u;local<h.y;local+=1u){let slotMeta=authority[base+p.rowOwnerMetadataOffset+rowBase(row)+local];let family=slotMeta&7u;if(((slotMeta>>6u)&1u)!=0u&&family<6u){counts[family]+=1u;}}
  for(var family=0u;family<6u;family+=1u){authority[base+p.rowFamilyPrefixOffset+familyIndex(row,family)]=counts[family];authority[base+p.rowFamilyHandleOffset+familyIndex(row,family)]=(familyIndex(row,family)*8u);for(var o=0u;o<8u;o+=1u){authority[base+p.rowFamilySlotOffset+(familyIndex(row,family)*8u+o)]=INVALID;}}}

var<workgroup> familyLane:array<u32,384>;
@compute @workgroup_size(64)fn prefixStructuredFamilies(@builtin(local_invocation_index)lane:u32){let rows=min(control.rowCount,p.rowCapacity);let chunk=(rows+63u)/64u;let first=lane*chunk;let last=min(first+chunk,rows);
  for(var family=0u;family<6u;family+=1u){var total=0u;for(var row=first;row<last;row+=1u){total+=authority[candidateAuthorityBase()+p.rowFamilyPrefixOffset+familyIndex(row,family)];}familyLane[family*64u+lane]=total;}workgroupBarrier();
  for(var width=1u;width<64u;width<<=1u){for(var family=0u;family<6u;family+=1u){var add=0u;if(lane>=width){add=familyLane[family*64u+lane-width];}workgroupBarrier();familyLane[family*64u+lane]+=add;}workgroupBarrier();}
  if(lane==0u){control.familyOffsets[0]=0u;for(var family=0u;family<6u;family+=1u){control.familyOffsets[family+1u]=control.familyOffsets[family]+familyLane[family*64u+63u];}control.slotCount=control.familyOffsets[6];if(control.slotCount>p.slotCapacity){fail(0u,ERROR_CAPACITY);}}
  workgroupBarrier();for(var family=0u;family<6u;family+=1u){var prefix=select(0u,familyLane[family*64u+lane-1u],lane>0u);for(var row=first;row<last;row+=1u){let index=p.rowFamilyPrefixOffset+familyIndex(row,family);let count=authority[candidateAuthorityBase()+index];authority[candidateAuthorityBase()+index]=prefix;prefix+=count;}}}

fn oldRow(newRow:u32)->u32{if(p.newToOldOffset+newRow>=arrayLength(&rowDelta)){return INVALID;}let encoded=rowDelta[p.newToOldOffset+newRow]&0x3fffffffu;return select(INVALID,encoded-1u,encoded!=0u);}fn candidateEpoch()->u32{return select(0u,rowDelta[p.deltaControlOffset+7u],p.deltaControlOffset+7u<arrayLength(&rowDelta));}
// x is the carried value; y marks an exact face identity. Changed topology
// faces remain pending for the old-field transfer instead of becoming zero.
fn carryValue(row:u32,neighbor:u32,family:u32,orientation:u32)->vec2f{if(control.projected==0u){return vec2f(0.0);}let oldOwner=oldRow(row);if(oldOwner==INVALID){return vec2f(0.0);}let oldNeighbor=select(INVALID,oldRow(neighbor),neighbor!=INVALID);if(neighbor!=INVALID&&oldNeighbor==INVALID){return vec2f(0.0);}
  let accepted=acceptedAuthorityBase();
  for(var local=0u;local<p.maxSlots;local+=1u){let handle=authority[accepted+p.rowHandleOffset+rowBase(oldOwner)+local];if(handle==INVALID||handle>=p.slotCapacity){continue;}let slotMetadata=authority[accepted+p.metadataOffset+handle];if((slotMetadata&7u)!=family||((slotMetadata>>3u)&7u)!=orientation){continue;}let other=authority[accepted+p.neighborOffset+handle];if(other==oldNeighbor){return vec2f(bitcast<f32>(authority[accepted+p.valuesOffset+handle]),1.0);}}return vec2f(0.0);}

// One invocation per (row, catalog slot). The rank a slot used to inherit from
// the running \`ranks[family]\` counter is recomputed here as "owner slots of the
// same family strictly earlier in this row", which is the same number by
// construction and needs no atomic and no cross-thread ordering. A slot that
// overflows \`slotCount\` still consumes its rank, exactly as the serial loop's
// pre-check increment did.
@compute @workgroup_size(64)fn scatterStructuredFamilySlots(@builtin(global_invocation_id)g:vec3u){let rows=min(control.rowCount,p.rowCapacity);let slot=foldedItem(g);let row=slot/p.maxSlots;let local=slot%p.maxSlots;if(row>=rows||atomicLoad(&control.flags)!=0u){return;}let m=metrics[row];let h=caseHeader(m.caseId);if(local>=h.y){return;}
  let base=candidateAuthorityBase();let at=rowBase(row)+local;let ownerMeta=authority[base+p.rowOwnerMetadataOffset+at];if(((ownerMeta>>6u)&1u)==0u){return;}let family=ownerMeta&7u;let orientation=(ownerMeta>>3u)&7u;
  var rank=0u;for(var earlier=0u;earlier<local;earlier+=1u){let earlierMeta=authority[base+p.rowOwnerMetadataOffset+rowBase(row)+earlier];if(((earlierMeta>>6u)&1u)!=0u&&(earlierMeta&7u)==family){rank+=1u;}}
  let handle=control.familyOffsets[family]+authority[base+p.rowFamilyPrefixOffset+familyIndex(row,family)]+rank;if(handle>=control.slotCount){fail(row,ERROR_CAPACITY);return;}let neighbor=authority[base+p.rowNeighborOffset+at];let geo=geometry(row,local);let global=authority[base+p.rowCatalogOffset+at];
  let carried=carryValue(row,neighbor,family,orientation);authority[base+p.rowHandleOffset+at]=handle;authority[base+p.valuesOffset+handle]=bitcast<u32>(carried.x);authority[base+p.ownerOffset+handle]=row;authority[base+p.neighborOffset+handle]=neighbor;authority[base+p.metadataOffset+handle]=family|(orientation<<3u)|(global<<6u)|(((ownerMeta>>7u)&1u)<<30u);authority[base+p.areaOffset+handle]=bitcast<u32>(geo.area);authority[base+p.inverseOffset+handle]=bitcast<u32>(geo.inverseDistance);authority[base+p.fractionOffset+handle]=bitcast<u32>(1.0);authority[base+p.pressureScaleOffset+handle]=bitcast<u32>(1.0);authority[base+p.normalOffset+4u*handle]=bitcast<u32>(geo.normal.x);authority[base+p.normalOffset+4u*handle+1u]=bitcast<u32>(geo.normal.y);authority[base+p.normalOffset+4u*handle+2u]=bitcast<u32>(geo.normal.z);authority[base+p.normalOffset+4u*handle+3u]=0u;authority[base+p.centroidOffset+4u*handle]=bitcast<u32>(geo.centroid.x);authority[base+p.centroidOffset+4u*handle+1u]=bitcast<u32>(geo.centroid.y);authority[base+p.centroidOffset+4u*handle+2u]=bitcast<u32>(geo.centroid.z);authority[base+p.centroidOffset+4u*handle+3u]=bitcast<u32>(carried.y);}

@compute @workgroup_size(64)fn publishSection63Rows(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=control.rowCount||row>=p.rowCapacity||atomicLoad(&control.flags)!=0u){return;}let m=metrics[row];let h=caseHeader(m.caseId);
  for(var axisSlot=0u;axisSlot<6u;axisSlot+=1u){authority[candidateAuthorityBase()+p.rowAxisOffset+6u*row+axisSlot]=INVALID;}for(var local=0u;local<h.y;local+=1u){let at=rowBase(row)+local;let neighbor=authority[candidateAuthorityBase()+p.rowNeighborOffset+at];let ownerMeta=authority[candidateAuthorityBase()+p.rowOwnerMetadataOffset+at];var handle=authority[candidateAuthorityBase()+p.rowHandleOffset+at];var sign=1;if(((ownerMeta>>6u)&1u)==0u){let reverse=authority[candidateAuthorityBase()+p.rowReciprocalOffset+at];if(neighbor==INVALID||reverse==INVALID){atomicOr(&control.flags,ERROR_RECIPROCITY);continue;}handle=authority[candidateAuthorityBase()+p.rowHandleOffset+rowBase(neighbor)+reverse];sign=-1;}if(handle==INVALID||handle>=control.slotCount){atomicOr(&control.flags,ERROR_CAPACITY);continue;}authority[candidateAuthorityBase()+p.rowHandleOffset+at]=handle;authority[candidateAuthorityBase()+p.rowSignOffset+at]=bitcast<u32>(sign);let family=ownerMeta&7u;let orientation=(ownerMeta>>3u)&7u;authority[candidateAuthorityBase()+p.rowFamilySlotOffset+authority[candidateAuthorityBase()+p.rowFamilyHandleOffset+familyIndex(row,family)]+orientation]=handle;let localGeometry=geometry(row,local);let worldNormal=localGeometry.normal;if(neighbor!=INVALID&&headers[neighbor].size==headers[row].size){let absoluteNormal=abs(worldNormal);let axis=select(select(2u,1u,absoluteNormal.y>absoluteNormal.z),0u,absoluteNormal.x>max(absoluteNormal.y,absoluteNormal.z));let transverse=dot(absoluteNormal,vec3f(1.0))-absoluteNormal[axis];if(absoluteNormal[axis]>.9999&&transverse<1e-4){let direction=2u*axis+select(0u,1u,worldNormal[axis]>0.0);authority[candidateAuthorityBase()+p.rowAxisOffset+6u*row+direction]=neighbor;}}}
  let source=m.caseId*${OCTREE_SECTION63_CHANNELS}u;let destination=section63BankBase()+row*${OCTREE_SECTION63_CHANNELS}u;let scale=size(row);if(source+${OCTREE_SECTION63_CHANNELS}u>arrayLength(&catalogCoefficients)||destination+${OCTREE_SECTION63_CHANNELS}u>arrayLength(&section63Coefficients)){fail(row,ERROR_CAPACITY);return;}for(var channel=0u;channel<${OCTREE_SECTION63_CHANNELS}u;channel+=1u){section63Coefficients[destination+channel]=catalogCoefficients[source+channel]*scale;}
}

fn familyClass(handle:u32)->u32{let owner=authority[candidateAuthorityBase()+p.ownerOffset+handle];let neighbor=authority[candidateAuthorityBase()+p.neighborOffset+handle];var transition=metrics[owner].caseId!=0u;var boundary=neighbor==INVALID||rowClass(owner)>=2u;if(neighbor!=INVALID){transition=transition||metrics[neighbor].caseId!=0u;boundary=boundary||rowClass(neighbor)>=2u;}return select(select(5u,7u,boundary),select(6u,8u,boundary),transition);}
var<workgroup> finalCounts:array<u32,${9 * OCTREE_STRUCTURED_FINALIZE_LANES}>;
@compute @workgroup_size(${OCTREE_STRUCTURED_FINALIZE_LANES})fn finalizeStructuredPublication(@builtin(local_invocation_index)lane:u32){let rows=min(control.rowCount,p.rowCapacity);let slots=min(control.slotCount,p.slotCapacity);let rowChunk=(rows+${OCTREE_STRUCTURED_FINALIZE_LANES - 1}u)/${OCTREE_STRUCTURED_FINALIZE_LANES}u;let rowFirst=lane*rowChunk;let rowLast=min(rowFirst+rowChunk,rows);let slotChunk=(slots+${OCTREE_STRUCTURED_FINALIZE_LANES - 1}u)/${OCTREE_STRUCTURED_FINALIZE_LANES}u;let slotFirst=lane*slotChunk;let slotLast=min(slotFirst+slotChunk,slots);
  var local:array<u32,9>;for(var row=rowFirst;row<rowLast;row+=1u){local[rowClass(row)]+=1u;}for(var handle=slotFirst;handle<slotLast;handle+=1u){local[familyClass(handle)]+=1u;let marker=bitcast<f32>(authority[candidateAuthorityBase()+p.centroidOffset+4u*handle+3u]);if(marker<=.5){local[4u]+=1u;}}for(var cls=0u;cls<9u;cls+=1u){finalCounts[cls*${OCTREE_STRUCTURED_FINALIZE_LANES}u+lane]=local[cls];}workgroupBarrier();
  for(var width=1u;width<${OCTREE_STRUCTURED_FINALIZE_LANES}u;width<<=1u){for(var cls=0u;cls<9u;cls+=1u){var add=0u;if(lane>=width){add=finalCounts[cls*${OCTREE_STRUCTURED_FINALIZE_LANES}u+lane-width];}workgroupBarrier();finalCounts[cls*${OCTREE_STRUCTURED_FINALIZE_LANES}u+lane]+=add;}workgroupBarrier();}
  var prefix:array<u32,9>;if(lane>0u){for(var cls=0u;cls<9u;cls+=1u){prefix[cls]=finalCounts[cls*${OCTREE_STRUCTURED_FINALIZE_LANES}u+lane-1u];}}
  for(var row=rowFirst;row<rowLast;row+=1u){let cls=rowClass(row);worksets[worksetBankBase()+worksetBase(cls)+7u+prefix[cls]]=row;prefix[cls]+=1u;}for(var handle=slotFirst;handle<slotLast;handle+=1u){let cls=familyClass(handle);worksets[worksetBankBase()+worksetBase(cls)+7u+prefix[cls]]=handle;prefix[cls]+=1u;let marker=bitcast<f32>(authority[candidateAuthorityBase()+p.centroidOffset+4u*handle+3u]);if(marker<=.5){let transferRank=prefix[4u];worksets[worksetBankBase()+worksetBase(4u)+7u+transferRank]=handle;control.transferHandles[transferRank]=handle;prefix[4u]+=1u;}}workgroupBarrier();
  if(lane==0u){let clean=atomicLoad(&control.flags)==0u&&control.slotCount<=p.slotCapacity;let epoch=candidateEpoch();publicationDispatch[18u]=0u;publicationDispatch[19u]=1u;publicationDispatch[20u]=1u;publicationDispatch[21u]=0u;publicationDispatch[22u]=1u;publicationDispatch[23u]=1u;publicationDispatch[24u]=0u;publicationDispatch[25u]=1u;publicationDispatch[26u]=1u;for(var cls=0u;cls<9u;cls+=1u){let count=finalCounts[cls*${OCTREE_STRUCTURED_FINALIZE_LANES}u+${OCTREE_STRUCTURED_FINALIZE_LANES - 1}u];let groups=select((count+63u)/64u,count,cls==4u);let groupsX=max(1u,min(65535u,groups));let groupsY=(groups+groupsX-1u)/groupsX;let base=worksetBase(cls);worksets[worksetBankBase()+base]=epoch;worksets[worksetBankBase()+base+1u]=count;worksets[worksetBankBase()+base+2u]=select(p.rowCapacity,p.slotCapacity,cls>=4u);worksets[worksetBankBase()+base+3u]=3u;worksets[worksetBankBase()+base+4u]=groupsX;worksets[worksetBankBase()+base+5u]=groupsY;worksets[worksetBankBase()+base+6u]=1u;if(clean&&cls<=4u){let dispatch=6u+3u*cls;publicationDispatch[dispatch]=groupsX;publicationDispatch[dispatch+1u]=groupsY;publicationDispatch[dispatch+2u]=1u;}}if(clean){publishExactRowDispatch(21u,finalCounts[255u]+finalCounts[511u]+finalCounts[767u]+finalCounts[1023u]);publishExactRowDispatch(24u,finalCounts[511u]+finalCounts[1023u]);}if(clean&&epoch!=0u){control.epoch=epoch;atomicStore(&control.flags,${OCTREE_STRUCTURED_GPU_VALID}u);}}}

@compute @workgroup_size(64)fn reconstructStructuredCellVelocity(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=control.rowCount||row>=p.rowCapacity||atomicLoad(&control.flags)!=${OCTREE_STRUCTURED_GPU_VALID}u){return;}let m=metrics[row];let h=caseHeader(m.caseId);var canonical=vec3f(0.0);for(var local=0u;local<h.y;local+=1u){let at=rowBase(row)+local;let handle=authority[candidateAuthorityBase()+p.rowHandleOffset+at];if(handle==INVALID||handle>=control.slotCount){rowVelocities[rowBankBase()+row]=vec4f(0.0);return;}let sample=f32(bitcast<i32>(authority[candidateAuthorityBase()+p.rowSignOffset+at]))*bitcast<f32>(authority[candidateAuthorityBase()+p.valuesOffset+handle]);let global=authority[candidateAuthorityBase()+p.rowCatalogOffset+at];let r=p.reconstructionOffset+3u*global;canonical+=vec3f(bitcast<f32>(dense[r]),bitcast<f32>(dense[r+1u]),bitcast<f32>(dense[r+2u]))*sample;}let hrow=headers[row];rowVelocities[rowBankBase()+row]=vec4f(inverseStructuredTransform(canonical,m.transformAndFlags&63u),1.0);rowGeometry[rowBankBase()+row]=vec4u(hrow.cell,hrow.size,m.caseId,m.transformAndFlags);}

@compute @workgroup_size(1)fn acceptStructuredPublication(){if(atomicLoad(&control.flags)!=${OCTREE_STRUCTURED_GPU_VALID}u){return;}atomicStore(&acceptedControl[1],INVALID);atomicStore(&acceptedControl[2],control.rowCount);atomicStore(&acceptedControl[3],control.epoch);atomicStore(&acceptedControl[4],control.activeBank);atomicStore(&acceptedControl[5],control.slotCount);atomicStore(&acceptedControl[0],0u);}
`;

/** Activity-only classification variant; disabled mode returns the exact production bytes. */
export function directStructuredVelocityPublicationActivityShader(
  activity: GPULogicalActivityAdoptionContext,
): string {
  const entry = activity.workgroup("classify-structured-catalog-slots", "enter", {
    workgroupId: "activityWorkgroupId",
    numWorkgroups: "activityNumWorkgroups",
    localInvocationIndex: "activityLocalInvocationIndex",
    workgroupLaneCount: 64,
  });
  const exit = activity.workgroup("classify-structured-catalog-slots", "exit", {
    workgroupId: "activityWorkgroupId",
    numWorkgroups: "activityNumWorkgroups",
    localInvocationIndex: "activityLocalInvocationIndex",
    workgroupLaneCount: 64,
  });
  if (!entry && !exit) return directStructuredVelocityPublicationWGSL;
  const signature = "fn classifyStructuredCatalogSlots(@builtin(global_invocation_id)g:vec3u)";
  const instrumentedSignature = "fn classifyStructuredCatalogSlots(@builtin(global_invocation_id)g:vec3u,@builtin(workgroup_id)activityWorkgroupId:vec3u,@builtin(local_invocation_index)activityLocalInvocationIndex:u32,@builtin(num_workgroups)activityNumWorkgroups:vec3u)";
  const start = directStructuredVelocityPublicationWGSL.indexOf(signature);
  if (start < 0) throw new Error("Structured-classification activity entry point is missing");
  const bodyStart = directStructuredVelocityPublicationWGSL.indexOf("{", start + signature.length);
  let depth = 0, bodyEnd = -1;
  for (let index = bodyStart; index < directStructuredVelocityPublicationWGSL.length; index += 1) {
    if (directStructuredVelocityPublicationWGSL[index] === "{") depth += 1;
    else if (directStructuredVelocityPublicationWGSL[index] === "}" && --depth === 0) { bodyEnd = index; break; }
  }
  if (bodyStart < 0 || bodyEnd < 0) throw new Error("Structured-classification activity body is malformed");
  const body = directStructuredVelocityPublicationWGSL.slice(bodyStart + 1, bodyEnd)
    .replace(/\breturn;/g, `${exit}return;`);
  return `${directStructuredVelocityPublicationWGSL.slice(0, start)}${instrumentedSignature}{${entry}${body}${exit}${directStructuredVelocityPublicationWGSL.slice(bodyEnd)}`;
}

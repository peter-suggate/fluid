import type { OctreeFaceMirrorSource } from "./webgpu-octree-face-mirror";
import {
  OCTREE_FACE_TRANSFER_RECORD_BYTES,
} from "./webgpu-octree-face-transport";

export const OCTREE_FACE_TRANSFER_DIAGNOSTIC_BYTES = 32;
const SORT_PARAMETER_STRIDE = 256;
const RADIX_BITS = 4;
const RADIX_BINS = 1 << RADIX_BITS;
// Stable LSD order: axis/span, z, y, then x.  The resulting order is the
// lexicographic full-width (x,y,z,axisSpan) canonical face key.
const FULL_WIDTH_RADIX_PASSES = (32 / RADIX_BITS) * 4;
const RADIX_BLOCK_SIZE = 256;
export const OCTREE_FACE_PREVIOUS_CONTROL_BYTES = 64;
export const OCTREE_FACE_PREVIOUS_RECORD_BYTES = 20;
export const OCTREE_FACE_PREVIOUS_GENERATION_OFFSET_BYTES = 52;
export const OCTREE_FACE_PREVIOUS_VALID_OFFSET_BYTES = 56;
export const OCTREE_FACE_PREVIOUS_PUBLICATION_HEADER_BYTES = 64;
const SORT_DISPATCH_BYTES = 36;
const VALIDATE_DISPATCH_OFFSET = 12;
const TRANSFER_DISPATCH_OFFSET = 24;

export interface OctreeFaceTransferPlan {
  readonly faceCapacity: number;
  readonly sortCapacity: number;
  readonly sortPasses: number;
  readonly previousFaceBytes: number;
  readonly indexBytes: number;
  readonly recordBytes: number;
  readonly publicationBytes: number;
  readonly transferRecordOffsetBytes: number;
  readonly scratchBytes: number;
  readonly histogramBytes: number;
  readonly parameterBytes: number;
  readonly dispatchBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreeFaceTransferOptions {
  /** Full source maps are an inspection product and are not needed to transfer velocity. */
  readonly retainRecords?: boolean;
  /** Immutable finest-grid dimensions. Larger/ocean domains retain every
   * exact key nibble they require; only provably-zero high nibbles are gone. */
  readonly keyDimensions?: readonly [number, number, number];
}

export interface OctreeFacePreviousGenerationSource {
  readonly buffer: GPUBuffer;
  readonly offsetBytes: number;
}

/**
 * Compact, sorted generation-N Cartesian faces retained during the generation
 * N+1 topology rebuild. The first 64 bytes are a self-describing header:
 * live count, capacity, generation, validity, record offset in words, and
 * record stride in words. It is followed by the exact-key-sorted live prefix.
 */
export interface OctreeFacePreviousPublication {
  readonly buffer: GPUBuffer;
  readonly faceCapacity: number;
  readonly byteLength: number;
}

function radixDigitsForMaximum(maximumInclusive: number): number {
  if (!Number.isSafeInteger(maximumInclusive) || maximumInclusive < 0 || maximumInclusive > 0xffff_ffff) {
    throw new RangeError("Octree face radix key bound must fit in u32");
  }
  return Math.max(1, Math.ceil(Math.log2(maximumInclusive + 1) / RADIX_BITS));
}

export function octreeFaceTransferRadixFields(
  dimensions?: readonly [number, number, number],
): readonly { field: number; digits: number }[] {
  if (!dimensions) return [0, 1, 2, 3].map((field) => ({ field, digits: 32 / RADIX_BITS }));
  dimensions.forEach((value) => {
    if (!Number.isSafeInteger(value) || value < 1 || value > 0x3fff_ffff) {
      throw new RangeError("Octree face radix dimensions must be positive and leave two axis/span bits");
    }
  });
  const maximumSpan = Math.max(...dimensions);
  return [
    { field: 0, digits: radixDigitsForMaximum(maximumSpan * 4 + 2) },
    { field: 1, digits: radixDigitsForMaximum(dimensions[2]) },
    { field: 2, digits: radixDigitsForMaximum(dimensions[1]) },
    { field: 3, digits: radixDigitsForMaximum(dimensions[0]) },
  ];
}

export function planOctreeFaceTopologyTransfer(
  faceCapacity: number,
  options: OctreeFaceTransferOptions = {},
): OctreeFaceTransferPlan {
  if (!Number.isInteger(faceCapacity) || faceCapacity < 1) {
    throw new RangeError("Octree face transfer capacity must be positive");
  }
  // Full 256-lane radix blocks keep both workgroup barriers uniform. Padding
  // slots carry INVALID and sort after every live descriptor.
  let sortCapacity = RADIX_BLOCK_SIZE;
  while (sortCapacity < faceCapacity) sortCapacity *= 2;
  const sortPasses = octreeFaceTransferRadixFields(options.keyDimensions)
    .reduce((sum, field) => sum + field.digits, 0);
  const previousFaceBytes = faceCapacity * OCTREE_FACE_PREVIOUS_RECORD_BYTES;
  const indexBytes = sortCapacity * 4;
  const recordBytes = options.retainRecords ? faceCapacity * OCTREE_FACE_TRANSFER_RECORD_BYTES : 0;
  const publicationBytes = OCTREE_FACE_PREVIOUS_PUBLICATION_HEADER_BYTES + previousFaceBytes;
  const transferRecordOffsetBytes = publicationBytes;
  // Once radix sorting finishes, validation replaces the scratch indices with
  // directly consumable sorted faces. Optional audit records follow that live
  // publication and never alias it.
  const scratchBytes = Math.max(indexBytes, publicationBytes + recordBytes);
  const histogramBytes = Math.ceil(sortCapacity / RADIX_BLOCK_SIZE) * RADIX_BINS * 4;
  const parameterBytes = Math.max(256, sortPasses * SORT_PARAMETER_STRIDE);
  return {
    faceCapacity,
    sortCapacity,
    sortPasses,
    previousFaceBytes, indexBytes, recordBytes, publicationBytes, transferRecordOffsetBytes,
    scratchBytes, histogramBytes,
    parameterBytes, dispatchBytes: SORT_DISPATCH_BYTES,
    allocatedBytes: OCTREE_FACE_PREVIOUS_CONTROL_BYTES + previousFaceBytes + indexBytes + scratchBytes + histogramBytes
      + parameterBytes + OCTREE_FACE_TRANSFER_DIAGNOSTIC_BYTES + SORT_DISPATCH_BYTES,
  };
}

/**
 * Preserves adaptive face velocities across a deterministic topology rebuild.
 *
 * `encodeCapture` must be recorded before the octree/face rebuild and
 * `encodeTransfer` after it. Old face IDs are sorted by their canonical
 * `(originX, originY, originZ, axisSpan)` descriptor. Each new face performs bounded
 * exact, one-parent, or four-child lookups. This is compact and deterministic;
 * it deliberately allocates no box-sized spatial hash.
 */
export class WebGPUOctreeFaceTopologyTransfer {
  readonly plan: OctreeFaceTransferPlan;
  readonly records: GPUBuffer;
  readonly diagnostics: GPUBuffer;
  private readonly source: OctreeFaceMirrorSource;
  private readonly previousControl: GPUBuffer;
  private readonly previousFaces: GPUBuffer;
  private readonly sortedIndices: GPUBuffer;
  private readonly radixHistograms: GPUBuffer;
  private readonly sortParams: GPUBuffer;
  private readonly sortDispatch: GPUBuffer;
  private readonly prepareDispatchPipeline: GPUComputePipeline;
  private readonly prepareTransferDispatchPipeline: GPUComputePipeline;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly capturePipeline: GPUComputePipeline;
  private readonly histogramPipeline: GPUComputePipeline;
  private readonly prefixPipeline: GPUComputePipeline;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly validatePipeline: GPUComputePipeline;
  private readonly publishPreviousPipeline: GPUComputePipeline;
  private readonly transferPipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly swappedBindGroup: GPUBindGroup;

  constructor(device: GPUDevice, source: OctreeFaceMirrorSource, options: OctreeFaceTransferOptions = {}) {
    this.source = source;
    this.plan = planOctreeFaceTopologyTransfer(source.plan.faceCapacity, options);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.previousControl = device.createBuffer({ label: "Previous octree face control", size: OCTREE_FACE_PREVIOUS_CONTROL_BYTES, usage: storage });
    this.previousFaces = device.createBuffer({ label: "Previous canonical octree faces", size: this.plan.previousFaceBytes, usage: storage });
    this.sortedIndices = device.createBuffer({ label: "Sorted previous octree face IDs", size: this.plan.indexBytes, usage: storage });
    this.records = device.createBuffer({ label: "Octree face radix scratch and optional transfer records", size: this.plan.scratchBytes, usage: storage });
    this.radixHistograms = device.createBuffer({ label: "Octree face radix block histograms", size: this.plan.histogramBytes, usage: storage });
    this.diagnostics = device.createBuffer({ label: "Octree face topology transfer diagnostics", size: OCTREE_FACE_TRANSFER_DIAGNOSTIC_BYTES, usage: storage });
    this.sortParams = device.createBuffer({
      label: "Octree face transfer sort parameters",
      size: this.plan.parameterBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.sortDispatch = device.createBuffer({ label: "Previous octree face live radix dispatch", size: SORT_DISPATCH_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    const words = new Uint32Array(this.plan.parameterBytes / 4);
    let passIndex = 0;
    for (const field of octreeFaceTransferRadixFields(options.keyDimensions)) {
      for (let digit = 0; digit < field.digits; digit += 1) {
        const shift = digit * RADIX_BITS;
        const offset = (passIndex * SORT_PARAMETER_STRIDE) / 4;
        words.set([shift, field.field, this.plan.sortCapacity, Math.ceil(this.plan.sortCapacity / RADIX_BLOCK_SIZE)], offset);
        passIndex += 1;
      }
    }
    if (passIndex !== this.plan.sortPasses || passIndex > FULL_WIDTH_RADIX_PASSES) {
      throw new Error("Octree face radix parameter plan drifted from its exact key width");
    }
    device.queue.writeBuffer(this.sortParams, 0, words);

    const layout = device.createBindGroupLayout({ label: "Octree face topology transfer layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const shaderModule = device.createShaderModule({ label: "Octree face topology transfer", code: octreeFaceTopologyTransferShader });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (label: string, entryPoint: string): GPUComputePipeline => device.createComputePipeline({
      label,
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint, constants: {
        retainTransferRecords: this.plan.recordBytes > 0 ? 1 : 0,
        transferRecordBaseWords: this.plan.transferRecordOffsetBytes / 4,
      } },
    });
    this.preparePipeline = pipeline("Prepare previous octree face sort", "prepareSort");
    this.prepareDispatchPipeline = pipeline("Publish previous octree face live radix dispatch", "publishSortDispatch");
    this.prepareTransferDispatchPipeline = pipeline("Publish octree face validation and transfer dispatches", "publishTransferDispatches");
    this.capturePipeline = pipeline("Capture compact previous octree faces", "captureFaces");
    this.histogramPipeline = pipeline("Histogram previous octree face keys", "radixHistogram");
    this.prefixPipeline = pipeline("Prefix octree face radix blocks", "radixPrefix");
    this.scatterPipeline = pipeline("Scatter previous octree face keys", "radixScatter");
    this.validatePipeline = pipeline("Validate octree face topology transfer", "validateTopology");
    this.publishPreviousPipeline = pipeline("Publish compact previous octree faces", "publishPreviousFaces");
    this.transferPipeline = pipeline("Transfer canonical octree face velocities", "transferFaces");
    const group = (input: GPUBuffer, output: GPUBuffer, label: string) => device.createBindGroup({ label, layout, entries: [
      { binding: 0, resource: { buffer: this.previousControl } },
      { binding: 1, resource: { buffer: this.previousFaces } },
      { binding: 2, resource: { buffer: source.control } },
      { binding: 3, resource: { buffer: source.faces } },
      { binding: 4, resource: { buffer: input } },
      { binding: 5, resource: { buffer: output } },
      { binding: 6, resource: { buffer: this.diagnostics } },
      { binding: 7, resource: { buffer: this.sortParams, size: 16 } },
      { binding: 8, resource: { buffer: this.radixHistograms } },
    ] });
    this.bindGroup = group(this.sortedIndices, this.records, "Octree face topology transfer bindings");
    this.swappedBindGroup = group(this.records, this.sortedIndices, "Octree face topology transfer swapped radix bindings");
  }

  encodeCapture(encoder: GPUCommandEncoder, generation?: OctreeFacePreviousGenerationSource): void {
    encoder.copyBufferToBuffer(this.source.control, 0, this.previousControl, 0, 16);
    if (generation) {
      if (!Number.isSafeInteger(generation.offsetBytes) || generation.offsetBytes < 0
        || generation.offsetBytes % 4 !== 0) throw new RangeError("Previous face generation offset must be u32 aligned");
      encoder.copyBufferToBuffer(generation.buffer, generation.offsetBytes, this.previousControl,
        OCTREE_FACE_PREVIOUS_GENERATION_OFFSET_BYTES, 4);
    } else {
      encoder.clearBuffer(this.previousControl, OCTREE_FACE_PREVIOUS_GENERATION_OFFSET_BYTES, 4);
    }
    let capture = encoder.beginComputePass({ label: "Prepare compact previous octree face capture" });
    capture.setPipeline(this.prepareDispatchPipeline); capture.setBindGroup(0, this.bindGroup, [0]); capture.dispatchWorkgroups(1);
    capture.end();
    encoder.copyBufferToBuffer(this.previousControl, 16, this.sortDispatch, 0, 12);
    capture = encoder.beginComputePass({ label: "Capture compact previous octree faces" });
    capture.setPipeline(this.capturePipeline); capture.setBindGroup(0, this.bindGroup, [0]);
    capture.dispatchWorkgroupsIndirect(this.sortDispatch, 0); capture.end();
  }

  encodeTransfer(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.diagnostics);
    const dispatches = encoder.beginComputePass({ label: "Prepare octree face transfer live dispatches" });
    dispatches.setPipeline(this.prepareTransferDispatchPipeline); dispatches.setBindGroup(0, this.bindGroup, [0]);
    dispatches.dispatchWorkgroups(1); dispatches.end();
    encoder.copyBufferToBuffer(this.previousControl, 28, this.sortDispatch, VALIDATE_DISPATCH_OFFSET, 24);
    const prepare = encoder.beginComputePass({ label: "Prepare previous octree face topology" });
    prepare.setPipeline(this.preparePipeline);
    // Choose the initial side from radix parity so every plan ends in the
    // canonical sortedIndices buffer. Vast domains can require an odd number
    // of exact key nibbles; starting those in scratch avoids both a terminal
    // full-prefix copy and a parity-dependent validation/record ABI.
    const startsSwapped = this.plan.sortPasses % 2 === 1;
    prepare.setBindGroup(0, startsSwapped ? this.swappedBindGroup : this.bindGroup, [0]);
    prepare.dispatchWorkgroupsIndirect(this.sortDispatch, 0);
    prepare.end();
    for (let passIndex = 0; passIndex < this.plan.sortPasses; passIndex += 1) {
      const group = (passIndex % 2 === 0) === startsSwapped ? this.swappedBindGroup : this.bindGroup;
      encoder.clearBuffer(this.radixHistograms);
      const histogram = encoder.beginComputePass({ label: "Histogram previous octree face topology" });
      histogram.setPipeline(this.histogramPipeline); histogram.setBindGroup(0, group, [passIndex * SORT_PARAMETER_STRIDE]);
      histogram.dispatchWorkgroupsIndirect(this.sortDispatch, 0); histogram.end();
      const prefix = encoder.beginComputePass({ label: "Prefix previous octree face topology" });
      prefix.setPipeline(this.prefixPipeline); prefix.setBindGroup(0, group, [passIndex * SORT_PARAMETER_STRIDE]);
      prefix.dispatchWorkgroups(1); prefix.end();
      const scatter = encoder.beginComputePass({ label: "Scatter previous octree face topology" });
      scatter.setPipeline(this.scatterPipeline); scatter.setBindGroup(0, group, [passIndex * SORT_PARAMETER_STRIDE]);
      scatter.dispatchWorkgroupsIndirect(this.sortDispatch, 0); scatter.end();
    }
    const validate = encoder.beginComputePass({ label: "Validate octree face topology transfer" });
    validate.setPipeline(this.validatePipeline);
    validate.setBindGroup(0, this.bindGroup, [0]);
    validate.dispatchWorkgroupsIndirect(this.sortDispatch, VALIDATE_DISPATCH_OFFSET);
    validate.end();
    const publishPrevious = encoder.beginComputePass({ label: "Publish compact previous octree faces" });
    publishPrevious.setPipeline(this.publishPreviousPipeline);
    publishPrevious.setBindGroup(0, this.bindGroup, [0]);
    publishPrevious.dispatchWorkgroups(1);
    publishPrevious.end();
    const transfer = encoder.beginComputePass({ label: "Transfer canonical octree face velocities" });
    transfer.setPipeline(this.transferPipeline);
    transfer.setBindGroup(0, this.bindGroup, [0]);
    transfer.dispatchWorkgroupsIndirect(this.sortDispatch, TRANSFER_DISPATCH_OFFSET);
    transfer.end();
  }

  destroy(): void {
    this.previousControl.destroy();
    this.previousFaces.destroy();
    this.sortedIndices.destroy();
    this.radixHistograms.destroy();
    this.records.destroy();
    this.diagnostics.destroy();
    this.sortParams.destroy();
    this.sortDispatch.destroy();
  }

  get previousPublication(): OctreeFacePreviousPublication {
    return { buffer: this.records, faceCapacity: this.plan.faceCapacity,
      byteLength: this.plan.publicationBytes };
  }
}

export const octreeFaceTopologyTransferShader = /* wgsl */ `
struct FaceRecord { negativeRow: u32, positiveRow: u32, originX: u32, originY: u32, originZ: u32, axisSpan: u32, normalVelocity: f32, area: f32 }
struct PreviousFace { originX: u32, originY: u32, originZ: u32, axisSpan: u32, normalVelocity: f32 }
struct TransferRecord { newFace: u32, sourceCount: u32, old0: u32, old1: u32, old2: u32, old3: u32 }
struct SortParams { shift: u32, field: u32, capacity: u32, blockCapacity: u32 }
@group(0) @binding(0) var<storage, read_write> previousControl: array<u32>;
@group(0) @binding(1) var<storage, read_write> previousFaces: array<PreviousFace>;
@group(0) @binding(2) var<storage, read_write> nextControl: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> nextFaces: array<FaceRecord>;
@group(0) @binding(4) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> sortScratch: array<u32>;
// transferred, initialized, invalid topology, fail-closed
@group(0) @binding(6) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> sortParams: SortParams;
@group(0) @binding(8) var<storage, read_write> radixHistograms: array<atomic<u32>>;
override retainTransferRecords: bool = false;
override transferRecordBaseWords: u32 = 0u;
const INVALID = 0xffffffffu;
const VALID = 0x80000000u;
const RADIX_MASK = 15u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}

fn publishTransfer(record: TransferRecord) {
  if (retainTransferRecords) {
    let base = transferRecordBaseWords + record.newFace * 6u;
    sortScratch[base] = record.newFace; sortScratch[base + 1u] = record.sourceCount;
    sortScratch[base + 2u] = record.old0; sortScratch[base + 3u] = record.old1;
    sortScratch[base + 4u] = record.old2; sortScratch[base + 5u] = record.old3;
  }
}

fn keyLess(a: u32, b: u32) -> bool {
  if (a == INVALID) { return false; } if (b == INVALID) { return true; }
  let fa = previousFaces[a]; let fb = previousFaces[b];
  if (fa.originX != fb.originX) { return fa.originX < fb.originX; }
  if (fa.originY != fb.originY) { return fa.originY < fb.originY; }
  if (fa.originZ != fb.originZ) { return fa.originZ < fb.originZ; }
  return fa.axisSpan < fb.axisSpan;
}
fn compareKey(face: PreviousFace, origin: vec3u, axisSpan: u32) -> i32 {
  if (face.originX < origin.x) { return -1; } if (face.originX > origin.x) { return 1; }
  if (face.originY < origin.y) { return -1; } if (face.originY > origin.y) { return 1; }
  if (face.originZ < origin.z) { return -1; } if (face.originZ > origin.z) { return 1; }
  if (face.axisSpan < axisSpan) { return -1; }
  if (face.axisSpan == axisSpan) { return 0; }
  return 1;
}
fn findFace(origin: vec3u, axisSpan: u32) -> u32 {
  var low = 0u; var high = min(previousControl[0], previousControl[2]);
  while (low < high) { let middle = low + (high - low) / 2u; let index = sortedIndices[middle];
    if (compareKey(previousFaces[index], origin, axisSpan) < 0) { low = middle + 1u; } else { high = middle; }
  }
  if (low < min(previousControl[0], previousControl[2])) { let index = sortedIndices[low];
    if (compareKey(previousFaces[index], origin, axisSpan) == 0) { return index; }
  }
  return INVALID;
}
fn faceOrigin(face: FaceRecord) -> vec3u { return vec3u(face.originX, face.originY, face.originZ); }

@compute @workgroup_size(256)
fn captureFaces(@builtin(global_invocation_id) gid: vec3u) {
  let count = min(previousControl[0], previousControl[2]);
  if (gid.x >= count) { return; }
  let source = nextFaces[gid.x];
  let validArea=finite(source.area)&&source.area>0.;
  if(!validArea){atomicAdd(&diagnostics[2],1u);atomicStore(&diagnostics[3],1u);}
  previousFaces[gid.x] = PreviousFace(source.originX, source.originY, source.originZ, source.axisSpan,
    source.normalVelocity);
}

@compute @workgroup_size(1)
fn publishSortDispatch() {
  let count = min(previousControl[0], previousControl[2]);
  previousControl[14] = 0u;
  previousControl[4] = (count + 255u) / 256u;
  previousControl[5] = 1u;
  previousControl[6] = 1u;
}

@compute @workgroup_size(1)
fn publishTransferDispatches() {
  let oldCount = min(previousControl[0], previousControl[2]);
  let nextCount = min(atomicLoad(&nextControl[0]), atomicLoad(&nextControl[2]));
  previousControl[7] = (max(oldCount, nextCount) + 255u) / 256u;
  previousControl[8] = 1u; previousControl[9] = 1u;
  previousControl[10] = (nextCount + 255u) / 256u;
  previousControl[11] = 1u; previousControl[12] = 1u;
}

@compute @workgroup_size(256)
fn prepareSort(@builtin(global_invocation_id) gid: vec3u) {
  let count = min(previousControl[0], previousControl[2]);
  let rounded = previousControl[4] * 256u;
  if (gid.x >= rounded || gid.x >= sortParams.capacity) { return; }
  sortedIndices[gid.x] = select(INVALID, gid.x, gid.x < count);
  if (gid.x < count) { let face = previousFaces[gid.x]; let axis = face.axisSpan & 3u; let span = face.axisSpan >> 2u;
    if (axis > 2u || span == 0u || (span & (span - 1u)) != 0u) { atomicAdd(&diagnostics[2], 1u); atomicStore(&diagnostics[3], 1u); }
  }
}

fn radixKey(index: u32) -> u32 {
  if (index == INVALID) { return INVALID; }
  let face = previousFaces[index];
  switch sortParams.field {
    case 0u: { return face.axisSpan; }
    case 1u: { return face.originZ; }
    case 2u: { return face.originY; }
    default: { return face.originX; }
  }
}
fn radixBin(index: u32) -> u32 { return (radixKey(index) >> sortParams.shift) & RADIX_MASK; }

@compute @workgroup_size(256)
fn radixHistogram(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let bin = radixBin(sortedIndices[gid.x]);
  atomicAdd(&radixHistograms[bin * sortParams.blockCapacity + wid.x], 1u);
}

var<workgroup> radixTotals: array<u32, 16>;
@compute @workgroup_size(16)
fn radixPrefix(@builtin(local_invocation_id) lid: vec3u) {
  let bin = lid.x; var total = 0u;
  for (var block = 0u; block < previousControl[4]; block += 1u) {
    total += atomicLoad(&radixHistograms[bin * sortParams.blockCapacity + block]);
  }
  radixTotals[bin] = total;
  workgroupBarrier();
  var cursor = 0u;
  for (var prior = 0u; prior < bin; prior += 1u) { cursor += radixTotals[prior]; }
  for (var block = 0u; block < previousControl[4]; block += 1u) {
    let address = bin * sortParams.blockCapacity + block;
    let count = atomicLoad(&radixHistograms[address]);
    atomicStore(&radixHistograms[address], cursor); cursor += count;
  }
}

var<workgroup> blockBins: array<u32, 256>;
var<workgroup> blockRanks: array<u32, 256>;
@compute @workgroup_size(256)
fn radixScatter(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let index = sortedIndices[gid.x]; let bin = radixBin(index); blockBins[lid.x] = bin;
  workgroupBarrier();
  if (lid.x < 16u) {
    var rank = 0u;
    for (var lane = 0u; lane < 256u; lane += 1u) {
      if (blockBins[lane] == lid.x) { blockRanks[lane] = rank; rank += 1u; }
    }
  }
  workgroupBarrier();
  let destination = atomicLoad(&radixHistograms[bin * sortParams.blockCapacity + wid.x]) + blockRanks[lid.x];
  sortScratch[destination] = index;
}

@compute @workgroup_size(256)
fn validateTopology(@builtin(global_invocation_id) gid: vec3u) {
  let previousCount=min(previousControl[0],previousControl[2]);
  if(gid.x<previousCount){let index=sortedIndices[gid.x];let old=previousFaces[index];
    let base=16u+gid.x*5u;sortScratch[base]=old.originX;sortScratch[base+1u]=old.originY;
    sortScratch[base+2u]=old.originZ;sortScratch[base+3u]=old.axisSpan;sortScratch[base+4u]=bitcast<u32>(old.normalVelocity);
    if(!finite(old.normalVelocity)){atomicAdd(&diagnostics[2],1u);atomicStore(&diagnostics[3],1u);}
    if(gid.x>0u){let prior=previousFaces[sortedIndices[gid.x-1u]];if(prior.originX==old.originX&&prior.originY==old.originY&&prior.originZ==old.originZ&&prior.axisSpan==old.axisSpan){atomicAdd(&diagnostics[2],1u);atomicStore(&diagnostics[3],1u);}}
  }
  let nextCount = min(atomicLoad(&nextControl[0]), atomicLoad(&nextControl[2])); if (gid.x >= nextCount) { return; }
  let face = nextFaces[gid.x]; let axis = face.axisSpan & 3u; let span = face.axisSpan >> 2u;
  if (axis > 2u || span == 0u || (span & (span - 1u)) != 0u) { atomicAdd(&diagnostics[2], 1u); atomicStore(&diagnostics[3], 1u); }
}

@compute @workgroup_size(1)
fn publishPreviousFaces() {
  let count = min(previousControl[0], previousControl[2]);
  let valid=select(0u, VALID,
    previousControl[13] != 0u && previousControl[1] == 0u
      && previousControl[0] <= previousControl[2] && atomicLoad(&diagnostics[3]) == 0u
      && (count == 0u || previousControl[4] > 0u));
  previousControl[14]=valid;sortScratch[0]=count;sortScratch[1]=previousControl[2];
  sortScratch[2]=previousControl[13];sortScratch[3]=valid;sortScratch[4]=16u;sortScratch[5]=5u;
}

fn publishHash(face:FaceRecord){let h=(face.originX*0x9e3779b9u)^(face.originY*0x85ebca6bu)^(face.originZ*0xc2b2ae35u)^face.axisSpan;atomicXor(&diagnostics[4],h);atomicXor(&diagnostics[5],bitcast<u32>(face.normalVelocity)*(h|1u));atomicAdd(&diagnostics[6],1u);}

@compute @workgroup_size(256)
fn transferFaces(@builtin(global_invocation_id) gid: vec3u) {
  let nextCount = min(atomicLoad(&nextControl[0]), atomicLoad(&nextControl[2]));
  if (gid.x >= nextCount) { return; }
  if (previousControl[1] != 0u || atomicLoad(&nextControl[1]) != 0u || previousControl[0] > previousControl[2] || atomicLoad(&diagnostics[3]) != 0u) {
    atomicStore(&diagnostics[3], 1u); return;
  }
  var next = nextFaces[gid.x]; let axis = next.axisSpan & 3u; let span = next.axisSpan >> 2u;
  let origin = faceOrigin(next); let exact = findFace(origin, next.axisSpan);
  if (exact != INVALID) {
    next.normalVelocity = previousFaces[exact].normalVelocity; nextFaces[gid.x] = next;
    publishTransfer(TransferRecord(gid.x, 1u, exact, INVALID, INVALID, INVALID));publishHash(next); atomicAdd(&diagnostics[0], 1u); return;
  }
  let tangentA = (axis + 1u) % 3u; let tangentB = (axis + 2u) % 3u;
  if (span > 1u) {
    let half = span / 2u; var children: array<u32, 4>; var childCount = 0u;
    for (var quadrant = 0u; quadrant < 4u; quadrant += 1u) { var childOrigin = origin;
      childOrigin[tangentA] += select(0u, half, (quadrant & 1u) != 0u); childOrigin[tangentB] += select(0u, half, (quadrant & 2u) != 0u);
      let child = findFace(childOrigin, axis | (half << 2u)); children[quadrant] = child; childCount += select(0u, 1u, child != INVALID);
    }
    if (childCount == 4u) { next.normalVelocity = 0.25 * (previousFaces[children[0]].normalVelocity + previousFaces[children[1]].normalVelocity + previousFaces[children[2]].normalVelocity + previousFaces[children[3]].normalVelocity);
      nextFaces[gid.x] = next; publishTransfer(TransferRecord(gid.x, 4u, children[0], children[1], children[2], children[3]));publishHash(next); atomicAdd(&diagnostics[0], 1u); return;
    }
  }
  let parentSpan = span * 2u; var parentOrigin = origin;
  parentOrigin[tangentA] = (parentOrigin[tangentA] / parentSpan) * parentSpan; parentOrigin[tangentB] = (parentOrigin[tangentB] / parentSpan) * parentSpan;
  let parent = findFace(parentOrigin, axis | (parentSpan << 2u));
  if (parent != INVALID) { next.normalVelocity = previousFaces[parent].normalVelocity; nextFaces[gid.x] = next;
    publishTransfer(TransferRecord(gid.x, 1u, parent, INVALID, INVALID, INVALID));publishHash(next); atomicAdd(&diagnostics[0], 1u); return;
  }
  next.normalVelocity = 0.0; nextFaces[gid.x] = next; publishTransfer(TransferRecord(gid.x, 0u, INVALID, INVALID, INVALID, INVALID));publishHash(next); atomicAdd(&diagnostics[1], 1u);
}
`;

import type { OctreeFaceMirrorSource } from "./webgpu-octree-face-mirror";
import {
  OCTREE_FACE_TRANSFER_RECORD_BYTES,
} from "./webgpu-octree-face-transport";

export const OCTREE_FACE_TRANSFER_DIAGNOSTIC_BYTES = 32;
const SORT_PARAMETER_STRIDE = 256;
const PREVIOUS_FACE_RECORD_BYTES = 20;
const RADIX_BITS = 4;
const RADIX_BINS = 1 << RADIX_BITS;
// Stable LSD order: axis/span, z, y, then x.  The resulting order is the
// lexicographic full-width (x,y,z,axisSpan) canonical face key.
const RADIX_PASSES = (32 / RADIX_BITS) * 4;
const RADIX_BLOCK_SIZE = 256;

export interface OctreeFaceTransferPlan {
  readonly faceCapacity: number;
  readonly sortCapacity: number;
  readonly sortPasses: number;
  readonly previousFaceBytes: number;
  readonly indexBytes: number;
  readonly recordBytes: number;
  readonly scratchBytes: number;
  readonly histogramBytes: number;
  readonly parameterBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreeFaceTransferOptions {
  /** Full source maps are an inspection product and are not needed to transfer velocity. */
  readonly retainRecords?: boolean;
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
  const sortPasses = RADIX_PASSES;
  const previousFaceBytes = faceCapacity * PREVIOUS_FACE_RECORD_BYTES;
  const indexBytes = sortCapacity * 4;
  const recordBytes = options.retainRecords ? faceCapacity * OCTREE_FACE_TRANSFER_RECORD_BYTES : 0;
  // Audit records are published only after sorting, so their arena can also
  // serve as the radix ping-pong index. This keeps the shader at WebGPU's
  // minimum eight storage-buffer bindings.
  const scratchBytes = Math.max(OCTREE_FACE_TRANSFER_RECORD_BYTES, indexBytes, recordBytes);
  const histogramBytes = Math.ceil(sortCapacity / RADIX_BLOCK_SIZE) * RADIX_BINS * 4;
  const parameterBytes = Math.max(256, sortPasses * SORT_PARAMETER_STRIDE);
  return {
    faceCapacity,
    sortCapacity,
    sortPasses,
    previousFaceBytes, indexBytes, recordBytes, scratchBytes, histogramBytes,
    parameterBytes,
    allocatedBytes: 16 + previousFaceBytes + indexBytes + scratchBytes + histogramBytes
      + parameterBytes + OCTREE_FACE_TRANSFER_DIAGNOSTIC_BYTES,
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
  private readonly preparePipeline: GPUComputePipeline;
  private readonly capturePipeline: GPUComputePipeline;
  private readonly histogramPipeline: GPUComputePipeline;
  private readonly prefixPipeline: GPUComputePipeline;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly validatePipeline: GPUComputePipeline;
  private readonly transferPipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly swappedBindGroup: GPUBindGroup;

  constructor(device: GPUDevice, source: OctreeFaceMirrorSource, options: OctreeFaceTransferOptions = {}) {
    this.source = source;
    this.plan = planOctreeFaceTopologyTransfer(source.plan.faceCapacity, options);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.previousControl = device.createBuffer({ label: "Previous octree face control", size: 16, usage: storage });
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
    const words = new Uint32Array(this.plan.parameterBytes / 4);
    let passIndex = 0;
    for (let field = 0; field < 4; field += 1) {
      for (let shift = 0; shift < 32; shift += RADIX_BITS) {
        const offset = (passIndex * SORT_PARAMETER_STRIDE) / 4;
        words.set([shift, field, this.plan.sortCapacity, Math.ceil(this.plan.sortCapacity / RADIX_BLOCK_SIZE)], offset);
        passIndex += 1;
      }
    }
    device.queue.writeBuffer(this.sortParams, 0, words);

    const layout = device.createBindGroupLayout({ label: "Octree face topology transfer layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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
      compute: { module: shaderModule, entryPoint, constants: { retainTransferRecords: this.plan.recordBytes > 0 ? 1 : 0 } },
    });
    this.preparePipeline = pipeline("Prepare previous octree face sort", "prepareSort");
    this.capturePipeline = pipeline("Capture compact previous octree faces", "captureFaces");
    this.histogramPipeline = pipeline("Histogram previous octree face keys", "radixHistogram");
    this.prefixPipeline = pipeline("Prefix octree face radix blocks", "radixPrefix");
    this.scatterPipeline = pipeline("Scatter previous octree face keys", "radixScatter");
    this.validatePipeline = pipeline("Validate octree face topology transfer", "validateTopology");
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

  encodeCapture(encoder: GPUCommandEncoder): void {
    encoder.copyBufferToBuffer(this.source.control, 0, this.previousControl, 0, 16);
    const capture = encoder.beginComputePass({ label: "Capture compact previous octree faces" });
    capture.setPipeline(this.capturePipeline);
    capture.setBindGroup(0, this.bindGroup, [0]);
    capture.dispatchWorkgroups(Math.ceil(this.plan.faceCapacity / RADIX_BLOCK_SIZE));
    capture.end();
  }

  encodeTransfer(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.diagnostics);
    const groups = Math.ceil(this.plan.sortCapacity / RADIX_BLOCK_SIZE);
    const prepare = encoder.beginComputePass({ label: "Prepare previous octree face topology" });
    prepare.setPipeline(this.preparePipeline);
    prepare.setBindGroup(0, this.bindGroup, [0]);
    prepare.dispatchWorkgroups(groups);
    prepare.end();
    for (let passIndex = 0; passIndex < this.plan.sortPasses; passIndex += 1) {
      const group = passIndex % 2 === 0 ? this.bindGroup : this.swappedBindGroup;
      encoder.clearBuffer(this.radixHistograms);
      const histogram = encoder.beginComputePass({ label: "Histogram previous octree face topology" });
      histogram.setPipeline(this.histogramPipeline); histogram.setBindGroup(0, group, [passIndex * SORT_PARAMETER_STRIDE]);
      histogram.dispatchWorkgroups(groups); histogram.end();
      const prefix = encoder.beginComputePass({ label: "Prefix previous octree face topology" });
      prefix.setPipeline(this.prefixPipeline); prefix.setBindGroup(0, group, [passIndex * SORT_PARAMETER_STRIDE]);
      prefix.dispatchWorkgroups(1); prefix.end();
      const scatter = encoder.beginComputePass({ label: "Scatter previous octree face topology" });
      scatter.setPipeline(this.scatterPipeline); scatter.setBindGroup(0, group, [passIndex * SORT_PARAMETER_STRIDE]);
      scatter.dispatchWorkgroups(groups); scatter.end();
    }
    const validate = encoder.beginComputePass({ label: "Validate octree face topology transfer" });
    validate.setPipeline(this.validatePipeline);
    validate.setBindGroup(0, this.bindGroup, [0]);
    validate.dispatchWorkgroups(Math.ceil(this.plan.faceCapacity / 256));
    validate.end();
    const transfer = encoder.beginComputePass({ label: "Transfer canonical octree face velocities" });
    transfer.setPipeline(this.transferPipeline);
    transfer.setBindGroup(0, this.bindGroup, [0]);
    transfer.dispatchWorkgroups(Math.ceil(this.plan.faceCapacity / 256));
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
  }
}

export const octreeFaceTopologyTransferShader = /* wgsl */ `
struct FaceRecord { negativeRow: u32, positiveRow: u32, originX: u32, originY: u32, originZ: u32, axisSpan: u32, normalVelocity: f32, area: f32 }
struct PreviousFace { originX: u32, originY: u32, originZ: u32, axisSpan: u32, normalVelocity: f32 }
struct TransferRecord { newFace: u32, sourceCount: u32, old0: u32, old1: u32, old2: u32, old3: u32 }
struct SortParams { shift: u32, field: u32, capacity: u32, blockCapacity: u32 }
@group(0) @binding(0) var<storage, read> previousControl: array<u32>;
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
const INVALID = 0xffffffffu;
const RADIX_MASK = 15u;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}

fn publishTransfer(record: TransferRecord) {
  if (retainTransferRecords) {
    let base = record.newFace * 6u;
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
  previousFaces[gid.x] = PreviousFace(source.originX, source.originY, source.originZ, source.axisSpan, source.normalVelocity);
}

@compute @workgroup_size(256)
fn prepareSort(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= sortParams.capacity) { return; }
  sortedIndices[gid.x] = select(INVALID, gid.x, gid.x < min(previousControl[0], previousControl[2]));
  if (gid.x < min(previousControl[0], previousControl[2])) { let face = previousFaces[gid.x]; let axis = face.axisSpan & 3u; let span = face.axisSpan >> 2u;
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
  if (gid.x >= sortParams.capacity) { return; }
  let bin = radixBin(sortedIndices[gid.x]);
  atomicAdd(&radixHistograms[bin * sortParams.blockCapacity + wid.x], 1u);
}

var<workgroup> radixTotals: array<u32, 16>;
@compute @workgroup_size(16)
fn radixPrefix(@builtin(local_invocation_id) lid: vec3u) {
  let bin = lid.x; var total = 0u;
  for (var block = 0u; block < sortParams.blockCapacity; block += 1u) {
    total += atomicLoad(&radixHistograms[bin * sortParams.blockCapacity + block]);
  }
  radixTotals[bin] = total;
  workgroupBarrier();
  var cursor = 0u;
  for (var prior = 0u; prior < bin; prior += 1u) { cursor += radixTotals[prior]; }
  for (var block = 0u; block < sortParams.blockCapacity; block += 1u) {
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
    if(!finite(old.normalVelocity)){atomicAdd(&diagnostics[2],1u);atomicStore(&diagnostics[3],1u);}
    if(gid.x>0u){let prior=previousFaces[sortedIndices[gid.x-1u]];if(prior.originX==old.originX&&prior.originY==old.originY&&prior.originZ==old.originZ&&prior.axisSpan==old.axisSpan){atomicAdd(&diagnostics[2],1u);atomicStore(&diagnostics[3],1u);}}
  }
  let nextCount = min(atomicLoad(&nextControl[0]), atomicLoad(&nextControl[2])); if (gid.x >= nextCount) { return; }
  let face = nextFaces[gid.x]; let axis = face.axisSpan & 3u; let span = face.axisSpan >> 2u;
  if (axis > 2u || span == 0u || (span & (span - 1u)) != 0u) { atomicAdd(&diagnostics[2], 1u); atomicStore(&diagnostics[3], 1u); }
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

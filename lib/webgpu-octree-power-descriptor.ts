/**
 * Standalone GPU producer for paper-compatible octree power descriptors.
 *
 * This mirror consumes the live compact LeafHeader and owner-buffer ABIs but
 * deliberately has no production cutover. Invalid topology publishes a zero
 * indirect dispatch, so a later power-face stage cannot consume partial data.
 * The current immutable catalog contains interior configurations only;
 * consequently any face/edge probe outside the domain is reported as
 * `boundary` and invalidates publication. Boundary power planes need a catalog
 * extension before this mirror can become authoritative for boundary rows.
 */

import {
  OCTREE_POWER_NEIGHBOR_DIRECTIONS,
  encodeSameOrCoarserPowerDescriptor,
  encodeSameOrFinerPowerDescriptor,
} from "./octree-power-descriptor";
import { octreePowerCoarseMaskNeedsAcuteRepair } from "./octree-power-topology";

export const OCTREE_POWER_LEAF_HEADER_BYTES = 48;
export const OCTREE_POWER_DESCRIPTOR_CONTROL_BYTES = 32;
export const OCTREE_POWER_DESCRIPTOR_DISPATCH_BYTES = 12;
export const OCTREE_POWER_DESCRIPTOR_INVALID = 0xffff_ffff;
export const OCTREE_POWER_OWNER_ARENA_MAGIC = 0x4f57_4e52;
/** Six world-space face directions in OCTREE_POWER_NEIGHBOR_DIRECTIONS order. */
export const OCTREE_POWER_DESCRIPTOR_BOUNDARY_SHIFT = 24;
export const OCTREE_POWER_DESCRIPTOR_BOUNDARY_MASK = 0x3f00_0000;

export const OCTREE_POWER_DESCRIPTOR_ERROR = Object.freeze({
  malformedHeader: 1 << 0,
  malformedOwner: 1 << 1,
  /** Interior-only catalog limitation: at least one of 18 probes left the domain. */
  /** Reserved ABI bit; domain boundaries are valid descriptor metadata. */
  boundary: 1 << 2,
  mixedGrading: 1 << 3,
  gradingRatio: 1 << 4,
  capacity: 1 << 5,
  /** Strictly-obtuse same/coarser simplex escaped the topology repair pass. */
  acuteGrading: 1 << 6,
} as const);

export interface OctreePowerDescriptorControl {
  readonly rowCount: number;
  readonly validCount: number;
  readonly errorCount: number;
  readonly firstInvalid: number;
  readonly flags: number;
  readonly sameOrFinerCount: number;
  readonly sameOrCoarserCount: number;
  readonly generation: number;
}

export interface OctreePowerDescriptorPlan {
  readonly rowCapacity: number;
  readonly descriptorBytes: number;
  readonly controlBytes: number;
  readonly dispatchBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreePowerLeafRow {
  /** Linear finest-cell index of the leaf origin. */
  readonly cell: number;
  readonly size: number;
}

export interface OctreePowerOwner {
  readonly origin: readonly [number, number, number];
  readonly size: number;
  /** Malformed data, as distinct from a valid canonical missing-air owner. */
  readonly invalid?: boolean;
}

export type OctreePowerOwnerLookup = (cell: readonly [number, number, number]) => OctreePowerOwner;

export interface OctreePowerRowDescriptor {
  readonly descriptor: number;
  readonly flags: number;
  readonly kind: "same-or-finer" | "same-or-coarser" | "invalid";
}

export type OctreePowerOwnerMode = "auto" | "dense" | "paged" | "live-index";

export interface OctreePowerDescriptorEncodeOptions {
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  /** Host count for tests/bootstrap. Prefer rowCountBuffer for the normal GPU-resident path. */
  readonly rowCount?: number;
  /** First u32 is the live compact row count (for example compaction[0]). */
  readonly rowCountBuffer?: GPUBuffer;
  readonly generation?: number;
  readonly ownerMode?: OctreePowerOwnerMode;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function supportedLeafSize(size: number): boolean {
  return size === 1 || size === 2 || size === 4 || size === 8 || size === 16 || size === 32;
}

export function planOctreePowerDescriptors(rowCapacityValue: number): OctreePowerDescriptorPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Power descriptor row capacity");
  const descriptorBytes = rowCapacity * 4;
  return {
    rowCapacity,
    descriptorBytes,
    controlBytes: OCTREE_POWER_DESCRIPTOR_CONTROL_BYTES,
    dispatchBytes: OCTREE_POWER_DESCRIPTOR_DISPATCH_BYTES,
    allocatedBytes: descriptorBytes + OCTREE_POWER_DESCRIPTOR_CONTROL_BYTES + OCTREE_POWER_DESCRIPTOR_DISPATCH_BYTES + 32 + 4,
  };
}

export function unpackOctreePowerDescriptorControl(words: ArrayLike<number>): OctreePowerDescriptorControl {
  if (words.length < 8) throw new RangeError("Power descriptor control needs eight words");
  return {
    rowCount: Number(words[0]) >>> 0,
    validCount: Number(words[1]) >>> 0,
    errorCount: Number(words[2]) >>> 0,
    firstInvalid: Number(words[3]) >>> 0,
    flags: Number(words[4]) >>> 0,
    sameOrFinerCount: Number(words[5]) >>> 0,
    sameOrCoarserCount: Number(words[6]) >>> 0,
    generation: Number(words[7]) >>> 0,
  };
}

function contains(owner: OctreePowerOwner, cell: readonly [number, number, number], dimensions: readonly [number, number, number]): boolean {
  return supportedLeafSize(owner.size)
    && owner.origin.every((value, axis) => Number.isSafeInteger(value) && value >= 0
      && value % owner.size === 0 && value + owner.size <= dimensions[axis]
      && cell[axis] >= value && cell[axis] < value + owner.size);
}

/** Decode the dense u32 owner word used by webgpu-octree.ts. */
export function decodeDenseOctreePowerOwner(
  wordValue: number,
  cell: readonly [number, number, number],
  dimensions: readonly [number, number, number],
  maximumLeafSize: number,
): OctreePowerOwner {
  const word = wordValue >>> 0;
  let owner: OctreePowerOwner;
  if (word === 0x8000_0000) {
    owner = { origin: [...cell] as [number, number, number], size: 1 };
  } else {
    const exponent = word & 7;
    if ((word & 0xc000_0000) !== 0 || exponent < 1 || exponent > 5) {
      return { origin: [...cell] as [number, number, number], size: 1, invalid: true };
    }
    const size = 1 << exponent;
    owner = {
      origin: [((word >>> 3) & 511) << exponent, ((word >>> 12) & 511) << exponent, ((word >>> 21) & 511) << exponent],
      size,
    };
  }
  return owner.size <= maximumLeafSize && contains(owner, cell, dimensions)
    ? owner
    : { ...owner, invalid: true };
}

/**
 * Decode the simulation owner-page payload with the exact semantics used by
 * the projection shader.  Owner pages are generation-published: zero and the
 * in-flight/retired sentinel both mean that the shared canonical coarse owner
 * is visible for this query.  This deliberately differs from the strict dense
 * diagnostic decoder above, which reports malformed storage.
 */
export function decodePagedOctreePowerOwner(
  wordValue: number,
  cell: readonly [number, number, number],
  dimensions: readonly [number, number, number],
  maximumLeafSize: number,
): OctreePowerOwner {
  const canonical = (): OctreePowerOwner => {
    let size = Math.min(maximumLeafSize, 8);
    for (;;) {
      const origin = cell.map((value) => Math.floor(value / size) * size) as [number, number, number];
      if (origin.every((value, axis) => value + size <= dimensions[axis]) || size === 1) return { origin, size };
      size >>= 1;
    }
  };
  const word = wordValue >>> 0;
  if (word === 0 || word === 0xffff_ffff) return canonical();
  if ((word & 0x8000_0000) !== 0) return { origin: [...cell] as [number, number, number], size: 1 };
  const exponent = word & 7;
  if (exponent < 1 || exponent > 5) return { origin: [...cell] as [number, number, number], size: 1 };
  const size = 1 << exponent;
  return {
    origin: [((word >>> 3) & 511) << exponent, ((word >>> 12) & 511) << exponent, ((word >>> 21) & 511) << exponent],
    size,
  };
}

/** Pack the existing dense owner ABI; useful for parity fixtures. */
export function packDenseOctreePowerOwner(origin: readonly [number, number, number], size: number): number {
  if (!supportedLeafSize(size)) throw new RangeError("Dense octree owner size is unsupported");
  if (origin.some((value) => !Number.isSafeInteger(value) || value < 0 || value % size !== 0)) {
    throw new RangeError("Dense octree owner origin must be non-negative and size-aligned");
  }
  if (size === 1) return 0x8000_0000;
  const exponent = Math.log2(size);
  const aligned = origin.map((value) => value >>> exponent);
  if (aligned.some((value) => value > 511)) throw new RangeError("Dense octree owner origin exceeds its packed field");
  return (exponent | (aligned[0] << 3) | (aligned[1] << 12) | (aligned[2] << 21)) >>> 0;
}

/** CPU oracle for one 48-byte LeafHeader row. */
export function describeOctreePowerRow(
  row: OctreePowerLeafRow,
  dimensions: readonly [number, number, number],
  maximumLeafSize: number,
  ownerAt: OctreePowerOwnerLookup,
): OctreePowerRowDescriptor {
  let flags = 0;
  let boundaryMask = 0;
  const volume = dimensions[0] * dimensions[1] * dimensions[2];
  if (!dimensions.every((value) => Number.isSafeInteger(value) && value > 0)
    || !Number.isSafeInteger(volume) || volume > 0xffff_ffff
    || !supportedLeafSize(maximumLeafSize) || !Number.isSafeInteger(row.cell) || row.cell < 0 || row.cell >= volume
    || !supportedLeafSize(row.size) || row.size > maximumLeafSize) {
    return { descriptor: OCTREE_POWER_DESCRIPTOR_INVALID, flags: OCTREE_POWER_DESCRIPTOR_ERROR.malformedHeader, kind: "invalid" };
  }
  const origin: [number, number, number] = [
    row.cell % dimensions[0],
    Math.floor(row.cell / dimensions[0]) % dimensions[1],
    Math.floor(row.cell / (dimensions[0] * dimensions[1])),
  ];
  if (origin.some((value, axis) => value % row.size !== 0 || value + row.size > dimensions[axis])) {
    return { descriptor: OCTREE_POWER_DESCRIPTOR_INVALID, flags: OCTREE_POWER_DESCRIPTOR_ERROR.malformedHeader, kind: "invalid" };
  }
  const anchor = ownerAt(origin);
  if (anchor.invalid || !contains(anchor, origin, dimensions)
    || anchor.size !== row.size || anchor.origin.some((value, axis) => value !== origin[axis])) {
    return { descriptor: OCTREE_POWER_DESCRIPTOR_INVALID, flags: OCTREE_POWER_DESCRIPTOR_ERROR.malformedOwner, kind: "invalid" };
  }
  const sizes: number[] = [];
  for (const direction of OCTREE_POWER_NEIGHBOR_DIRECTIONS) {
    const probe = direction.map((component, axis) => component < 0
      ? origin[axis] - 1
      : component > 0 ? origin[axis] + row.size : origin[axis] + Math.floor(row.size / 2)) as [number, number, number];
    if (probe.some((value, axis) => value < 0 || value >= dimensions[axis])) {
      if (sizes.length < 6) boundaryMask |= 1 << sizes.length;
      sizes.push(row.size);
      continue;
    }
    const owner = ownerAt(probe);
    if (owner.invalid || !contains(owner, probe, dimensions)) flags |= OCTREE_POWER_DESCRIPTOR_ERROR.malformedOwner;
    const overlapsAnchor = owner.origin.every((value, axis) =>
      Math.min(value + owner.size, origin[axis] + row.size) > Math.max(value, origin[axis]));
    if (overlapsAnchor) flags |= OCTREE_POWER_DESCRIPTOR_ERROR.malformedOwner;
    if (owner.size * 2 < row.size || owner.size > row.size * 2) flags |= OCTREE_POWER_DESCRIPTOR_ERROR.gradingRatio;
    sizes.push(owner.size);
  }
  const finer = sizes.some((size) => size < row.size);
  const coarser = sizes.some((size) => size > row.size);
  if (finer && coarser) flags |= OCTREE_POWER_DESCRIPTOR_ERROR.mixedGrading;
  if (flags !== 0) return { descriptor: OCTREE_POWER_DESCRIPTOR_INVALID, flags, kind: "invalid" };
  if (!coarser) {
    return {
      descriptor: (encodeSameOrFinerPowerDescriptor(sizes.map((size) => size === row.size))
        | (boundaryMask << OCTREE_POWER_DESCRIPTOR_BOUNDARY_SHIFT)) >>> 0,
      flags: 0,
      kind: "same-or-finer",
    };
  }
  const child = origin.map((value) => ((value / row.size) & 1) as 0 | 1) as [0 | 1, 0 | 1, 0 | 1];
  const outward = child.map((bit) => bit === 0 ? -1 : 1);
  const coarseDirections = [
    [outward[0], 0, 0], [0, outward[1], 0], [0, 0, outward[2]],
    [outward[0], outward[1], 0], [outward[0], 0, outward[2]], [0, outward[1], outward[2]],
  ];
  const coarseNeighbors = coarseDirections.map((direction) => {
    const index = OCTREE_POWER_NEIGHBOR_DIRECTIONS.findIndex((candidate) => candidate.every((value, axis) => value === direction[axis]));
    return index >= 0 && sizes[index] === row.size * 2;
  }) as [boolean, boolean, boolean, boolean, boolean, boolean];
  const coarseMask = coarseNeighbors.reduce((word, coarse, bit) => coarse ? word | (1 << bit) : word, 0);
  if (octreePowerCoarseMaskNeedsAcuteRepair(coarseMask)) {
    return { descriptor: OCTREE_POWER_DESCRIPTOR_INVALID,
      flags: OCTREE_POWER_DESCRIPTOR_ERROR.acuteGrading, kind: "invalid" };
  }
  return { descriptor: (encodeSameOrCoarserPowerDescriptor({ child, coarseNeighbors })
    | (boundaryMask << OCTREE_POWER_DESCRIPTOR_BOUNDARY_SHIFT)) >>> 0, flags: 0, kind: "same-or-coarser" };
}

export class WebGPUOctreePowerDescriptor {
  readonly plan: OctreePowerDescriptorPlan;
  readonly descriptors: GPUBuffer;
  readonly control: GPUBuffer;
  readonly dispatch: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly hostRowCount: GPUBuffer;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly generatePipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
  private readonly layout: GPUBindGroupLayout;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, rowCapacity: number) {
    this.plan = planOctreePowerDescriptors(rowCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.descriptors = device.createBuffer({ label: "Octree power descriptors", size: this.plan.descriptorBytes, usage: storage });
    this.control = device.createBuffer({ label: "Octree power descriptor control", size: this.plan.controlBytes, usage: storage });
    this.dispatch = device.createBuffer({ label: "Octree power descriptor dispatch", size: this.plan.dispatchBytes, usage: storage | GPUBufferUsage.INDIRECT });
    this.params = device.createBuffer({ label: "Octree power descriptor parameters", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.hostRowCount = device.createBuffer({ label: "Octree power descriptor host row count", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.layout = device.createBindGroupLayout({ label: "Octree power descriptor layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ] });
    const shaderModule = device.createShaderModule({ label: "Octree power descriptor generation", code: octreePowerDescriptorShader });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.preparePipeline = device.createComputePipeline({ label: "Prepare octree power descriptors", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "preparePowerDescriptors" } });
    this.generatePipeline = device.createComputePipeline({ label: "Generate octree power descriptors", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "generatePowerDescriptors" } });
    this.publishPipeline = device.createComputePipeline({ label: "Publish octree power descriptor dispatch", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "publishPowerDescriptors" } });
  }

  encode(encoder: GPUCommandEncoder, leafHeaders: GPUBuffer, owners: GPUBuffer, options: OctreePowerDescriptorEncodeOptions): void {
    if (this.destroyed) throw new Error("Octree power descriptor generator is destroyed");
    options.dimensions.forEach((value, axis) => positiveInteger(value, `Power descriptor dimension ${axis}`));
    if (options.dimensions.some((value) => value > 0x7fff_ffff)
      || options.dimensions[0] * options.dimensions[1] * options.dimensions[2] > 0xffff_ffff) {
      throw new RangeError("Power descriptor dimensions must fit the u32 linear-cell ABI");
    }
    positiveInteger(options.maximumLeafSize, "Power descriptor maximum leaf size");
    if (!supportedLeafSize(options.maximumLeafSize)) throw new RangeError("Power descriptor maximum leaf size is unsupported");
    if ((options.rowCount === undefined) === (options.rowCountBuffer === undefined)) {
      throw new RangeError("Power descriptor encode needs exactly one host or GPU row count source");
    }
    const hostRowCount = options.rowCount ?? this.plan.rowCapacity;
    if (!Number.isSafeInteger(hostRowCount) || hostRowCount < 0 || hostRowCount > 0xffff_ffff) throw new RangeError("Power descriptor row count must be unsigned");
    if (options.rowCount !== undefined) this.device.queue.writeBuffer(this.hostRowCount, 0, new Uint32Array([options.rowCount]));
    const generation = options.generation ?? 0;
    if (!Number.isSafeInteger(generation) || generation < 0 || generation > 0xffff_ffff) throw new RangeError("Power descriptor generation must be unsigned");
    const ownerMode = options.ownerMode === "dense" ? 1 : options.ownerMode === "paged" ? 2
      : options.ownerMode === "live-index" ? 3 : 0;
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([
      ...options.dimensions, options.maximumLeafSize, hostRowCount, generation, ownerMode, this.plan.rowCapacity,
    ]));
    const group = this.device.createBindGroup({ label: "Octree power descriptor bindings", layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: { buffer: leafHeaders } },
      { binding: 2, resource: { buffer: owners } }, { binding: 3, resource: { buffer: this.descriptors } },
      { binding: 4, resource: { buffer: this.control } }, { binding: 5, resource: { buffer: this.dispatch } },
      { binding: 6, resource: { buffer: options.rowCountBuffer ?? this.hostRowCount } },
    ] });
    const prepare = encoder.beginComputePass({ label: "Prepare power descriptor control" });
    prepare.setPipeline(this.preparePipeline); prepare.setBindGroup(0, group); prepare.dispatchWorkgroups(1); prepare.end();
    const available = Math.min(hostRowCount, this.plan.rowCapacity);
    if (available > 0) {
      const groups = Math.ceil(available / 64);
      const x = Math.min(groups, 65_535);
      const generate = encoder.beginComputePass({ label: "Generate power topology descriptors" });
      generate.setPipeline(this.generatePipeline); generate.setBindGroup(0, group);
      generate.dispatchWorkgroups(x, Math.ceil(groups / x)); generate.end();
    }
    const publish = encoder.beginComputePass({ label: "Publish power descriptor dispatch" });
    publish.setPipeline(this.publishPipeline); publish.setBindGroup(0, group); publish.dispatchWorkgroups(1); publish.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.descriptors.destroy(); this.control.destroy(); this.dispatch.destroy(); this.params.destroy(); this.hostRowCount.destroy();
  }
}

export const octreePowerDescriptorShader = /* wgsl */ `
struct Params { dimensionsMaximumLeaf:vec4u, rowCountGenerationModeCapacity:vec4u }
struct LeafHeader { cell:u32, entryStart:u32, entryCount:u32, size:u32, diagonal:f32, rhs:f32, pad0:u32, pad1:u32, gradient:vec4f }
struct Owner { origin:vec3u, size:u32, invalid:u32 }
struct Control { rowCount:atomic<u32>, validCount:atomic<u32>, errorCount:atomic<u32>, firstInvalid:atomic<u32>, flags:atomic<u32>, sameOrFinerCount:atomic<u32>, sameOrCoarserCount:atomic<u32>, generation:atomic<u32> }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(2) var<storage,read> owners:array<u32>;
@group(0) @binding(3) var<storage,read_write> descriptors:array<u32>;
@group(0) @binding(4) var<storage,read_write> control:Control;
@group(0) @binding(5) var<storage,read_write> indirectDispatch:array<u32>;
@group(0) @binding(6) var<storage,read> rowCountSource:array<u32>;
const INVALID:u32=0xffffffffu;
const OWNER_MAGIC:u32=0x4f574e52u;
const COARSER_FLAG:u32=0x80000000u;
const MALFORMED_HEADER:u32=1u;
const MALFORMED_OWNER:u32=2u;
const MIXED_GRADING:u32=8u;const ACUTE_GRADING:u32=64u;
const GRADING_RATIO:u32=16u;
const CAPACITY:u32=32u;
const DIRECTIONS:array<vec3i,18>=array<vec3i,18>(
  vec3i(-1,0,0),vec3i(0,-1,0),vec3i(0,0,-1),vec3i(0,0,1),vec3i(0,1,0),vec3i(1,0,0),
  vec3i(-1,-1,0),vec3i(-1,0,-1),vec3i(-1,0,1),vec3i(-1,1,0),vec3i(0,-1,-1),vec3i(0,-1,1),
  vec3i(0,1,-1),vec3i(0,1,1),vec3i(1,-1,0),vec3i(1,0,-1),vec3i(1,0,1),vec3i(1,1,0));
fn dims()->vec3u{return params.dimensionsMaximumLeaf.xyz;}
fn supportedSize(size:u32)->bool{return size==1u||size==2u||size==4u||size==8u||size==16u||size==32u;}
fn volumeFits()->bool{let d=dims();return d.x!=0u&&d.y!=0u&&d.z!=0u&&d.x<=0xffffffffu/d.y&&d.x*d.y<=0xffffffffu/d.z;}
fn cellCoord(cell:u32)->vec3u{let d=dims();return vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));}
fn canonicalOwner(cell:vec3u)->Owner{var size=min(params.dimensionsMaximumLeaf.w,8u);loop{let origin=(cell/vec3u(size))*vec3u(size);if(all(vec3u(size)<=dims())&&all(origin<=dims()-vec3u(size))){return Owner(origin,size,0u);}if(size==1u){return Owner(cell,1u,1u);}size>>=1u;}}
fn ownerValid(owner:Owner,cell:vec3u)->bool{return owner.invalid==0u&&supportedSize(owner.size)&&owner.size<=params.dimensionsMaximumLeaf.w&&all(vec3u(owner.size)<=dims())&&all(owner.origin<=dims()-vec3u(owner.size))&&all(owner.origin%vec3u(owner.size)==vec3u(0u))&&all(cell>=owner.origin)&&all(cell<owner.origin+vec3u(owner.size));}
fn decodeDenseOwner(word:u32,cell:vec3u)->Owner{if(word==0x80000000u){return Owner(cell,1u,0u);}let exponent=word&7u;if((word&0xc0000000u)!=0u||exponent==0u||exponent>5u){return Owner(cell,1u,1u);}let size=1u<<exponent;let origin=(cell>>vec3u(exponent))<<vec3u(exponent);var result=Owner(origin,size,0u);result.invalid=select(1u,0u,ownerValid(result,cell));return result;}
fn decodePagedOwner(word:u32,cell:vec3u)->Owner{if((word&0x80000000u)!=0u){return Owner(cell,1u,0u);}let exponent=word&7u;if(exponent==0u||exponent>5u){return Owner(cell,1u,0u);}let size=1u<<exponent;let origin=(cell>>vec3u(exponent))<<vec3u(exponent);return Owner(origin,size,0u);}
fn pagedOwners()->bool{let mode=params.rowCountGenerationModeCapacity.z;if(mode==1u){return false;}if(mode==2u){return true;}return arrayLength(&owners)>15u&&owners[15]==OWNER_MAGIC;}
fn pagedOwner(cell:vec3u)->Owner{if(arrayLength(&owners)<=15u||owners[15]!=OWNER_MAGIC){return Owner(cell,1u,1u);}let freeOffset=owners[5];let payloadOffset=owners[6];let capacity=owners[3];if(capacity==0u||freeOffset<=16u||((freeOffset-16u)&1u)!=0u||payloadOffset>=arrayLength(&owners)){return Owner(cell,1u,1u);}let hashCapacity=(freeOffset-16u)/2u;if(hashCapacity==0u||16u+2u*hashCapacity>arrayLength(&owners)){return Owner(cell,1u,1u);}let bd=(dims()+vec3u(7u))/8u;let brick=cell/8u;if(any(brick>=bd)||bd.x>0xffffffffu/bd.y){return Owner(cell,1u,1u);}let logical=brick.x+brick.y*bd.x+brick.z*bd.x*bd.y;let key=logical+1u;var slot=(logical*0x9e3779b1u)%hashCapacity;var encoded=0u;var found=false;for(var probe=0u;probe<hashCapacity;probe+=1u){let observed=owners[16u+slot];if(observed==key){encoded=owners[16u+hashCapacity+slot];found=true;break;}if(observed==0u){break;}slot=select(slot+1u,0u,slot+1u==hashCapacity);}if(!found||encoded==0u||encoded==INVALID||encoded>capacity){return canonicalOwner(cell);}let local=cell%vec3u(8u);let localIndex=local.x+local.y*8u+local.z*64u;let physical=encoded-1u;if(payloadOffset>=arrayLength(&owners)||physical>(arrayLength(&owners)-payloadOffset-1u)/512u){return canonicalOwner(cell);}let word=owners[payloadOffset+physical*512u+localIndex];if(word==0u||word==INVALID){return canonicalOwner(cell);}return decodePagedOwner(word,cell);}
fn hashSite(cell:u32,size:u32)->u32{var value=cell^(size*0x9e3779b9u);value=(value^(value>>16u))*0x7feb352du;value=(value^(value>>15u))*0x846ca68bu;return value^(value>>16u);}
fn indexedOwner(cell:vec3u,preferredSize:u32)->Owner{let capacity=arrayLength(&owners)/4u;if(capacity==0u||(capacity&(capacity-1u))!=0u){return Owner(cell,1u,1u);}let mask=capacity-1u;for(var size=1u;size<=params.dimensionsMaximumLeaf.w;size<<=1u){let origin=(cell/vec3u(size))*vec3u(size);let linear=origin.x+dims().x*(origin.y+dims().y*origin.z);let base=hashSite(linear,size)&mask;for(var probe=0u;probe<32u;probe+=1u){let slot=(base+probe)&mask;let observed=owners[slot*4u];if(observed==0u){break;}if(observed==linear+1u&&owners[slot*4u+1u]==size){return Owner(origin,size,0u);}}}var size=preferredSize;loop{let origin=(cell/vec3u(size))*vec3u(size);if(all(origin<=dims()-vec3u(size))){return Owner(origin,size,0u);}if(size==1u){return Owner(cell,1u,0u);}size>>=1u;}}
fn ownerAt(cell:vec3u,preferredSize:u32)->Owner{if(params.rowCountGenerationModeCapacity.z==3u){return indexedOwner(cell,preferredSize);}if(pagedOwners()){return pagedOwner(cell);}if(!volumeFits()){return Owner(cell,1u,1u);}let index=cell.x+dims().x*(cell.y+dims().y*cell.z);if(index>=arrayLength(&owners)){return Owner(cell,1u,1u);}return decodeDenseOwner(owners[index],cell);}
fn failRow(row:u32,flags:u32,detail:u32){descriptors[row]=0x40000000u|(flags&127u)|(detail<<7u);atomicAdd(&control.errorCount,1u);atomicMin(&control.firstInvalid,row);atomicOr(&control.flags,flags);}
@compute @workgroup_size(1) fn preparePowerDescriptors(){let requested=select(0u,rowCountSource[0],arrayLength(&rowCountSource)>0u);atomicStore(&control.rowCount,requested);atomicStore(&control.validCount,0u);atomicStore(&control.errorCount,0u);atomicStore(&control.firstInvalid,INVALID);atomicStore(&control.flags,0u);atomicStore(&control.sameOrFinerCount,0u);atomicStore(&control.sameOrCoarserCount,0u);atomicStore(&control.generation,params.rowCountGenerationModeCapacity.y);if(arrayLength(&indirectDispatch)>=3u){indirectDispatch[0]=0u;indirectDispatch[1]=1u;indirectDispatch[2]=1u;}}
@compute @workgroup_size(64) fn generatePowerDescriptors(@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) workgroups:vec3u,@builtin(local_invocation_index) lid:u32){
  let row=(wid.x+wid.y*workgroups.x)*64u+lid;let requested=atomicLoad(&control.rowCount);
  if(row>=requested||row>=arrayLength(&headers)||row>=arrayLength(&descriptors)){return;}
  let header=headers[row];
  if(!volumeFits()||!supportedSize(params.dimensionsMaximumLeaf.w)||!supportedSize(header.size)||header.size>params.dimensionsMaximumLeaf.w){failRow(row,MALFORMED_HEADER,1u);return;}
  let volume=dims().x*dims().y*dims().z;if(header.cell>=volume){failRow(row,MALFORMED_HEADER,1u);return;}
  let origin=cellCoord(header.cell);
  if(any(origin%vec3u(header.size)!=vec3u(0u))||any(vec3u(header.size)>dims())||any(origin>dims()-vec3u(header.size))){failRow(row,MALFORMED_HEADER,1u);return;}
  let anchor=ownerAt(origin,header.size);
  if(!ownerValid(anchor,origin)||anchor.size!=header.size||any(anchor.origin!=origin)){
    let reasons=select(0u,1u,anchor.invalid!=0u)|select(0u,2u,anchor.size!=header.size)|select(0u,4u,any(anchor.origin!=origin));
    failRow(row,MALFORMED_OWNER,2u|(min(anchor.size,63u)<<5u)|(reasons<<11u));return;
  }
  var sizes:array<u32,18>;var flags=0u;var firstDetail=0u;var boundaryMask=0u;var finer=false;var coarser=false;
  for(var bit=0u;bit<18u;bit+=1u){
    let direction=DIRECTIONS[bit];var probe=vec3i(0);
    for(var axis=0u;axis<3u;axis+=1u){probe[axis]=select(select(i32(origin[axis]+header.size/2u),i32(origin[axis]+header.size),direction[axis]>0),i32(origin[axis])-1,direction[axis]<0);}
    if(any(probe<vec3i(0))||any(probe>=vec3i(dims()))){if(bit<6u){boundaryMask|=1u<<bit;}sizes[bit]=header.size;continue;}
    let owner=ownerAt(vec3u(probe),header.size);var localFlags=0u;var reasons=0u;
    if(!ownerValid(owner,vec3u(probe))){localFlags|=MALFORMED_OWNER;reasons|=1u;}
    if(all(min(owner.origin+vec3u(owner.size),origin+vec3u(header.size))>max(owner.origin,origin))){localFlags|=MALFORMED_OWNER;reasons|=2u;}
    if(owner.size*2u<header.size||owner.size>header.size*2u){localFlags|=GRADING_RATIO;reasons|=4u;}
    if(localFlags!=0u&&firstDetail==0u){firstDetail=(bit+3u)|(min(owner.size,63u)<<5u)|(reasons<<11u);}
    flags|=localFlags;sizes[bit]=owner.size;finer=finer||owner.size<header.size;coarser=coarser||owner.size>header.size;
  }
  if(finer&&coarser){flags|=MIXED_GRADING;if(firstDetail==0u){firstDetail=31u;}}
  if(flags!=0u){failRow(row,flags,firstDetail);return;}
  var descriptor=boundaryMask<<24u;
  if(!coarser){for(var bit=0u;bit<18u;bit+=1u){if(sizes[bit]==header.size){descriptor|=1u<<bit;}}atomicAdd(&control.sameOrFinerCount,1u);}
  else{let child=(origin/vec3u(header.size))&vec3u(1u);descriptor|=COARSER_FLAG|child.x|(child.y<<1u)|(child.z<<2u);let outward=vec3i(select(-1,1,child.x==1u),select(-1,1,child.y==1u),select(-1,1,child.z==1u));let wanted=array<vec3i,6>(vec3i(outward.x,0,0),vec3i(0,outward.y,0),vec3i(0,0,outward.z),vec3i(outward.x,outward.y,0),vec3i(outward.x,0,outward.z),vec3i(0,outward.y,outward.z));var coarseMask=0u;for(var coarseBit=0u;coarseBit<6u;coarseBit+=1u){for(var bit=0u;bit<18u;bit+=1u){if(all(DIRECTIONS[bit]==wanted[coarseBit])&&sizes[bit]==header.size*2u){descriptor|=1u<<(coarseBit+3u);coarseMask|=1u<<coarseBit;}}}if(coarseMask==25u||coarseMask==42u||coarseMask==52u||coarseMask==57u||coarseMask==58u||coarseMask==60u){failRow(row,ACUTE_GRADING,coarseMask);return;}atomicAdd(&control.sameOrCoarserCount,1u);}
  descriptors[row]=descriptor;atomicAdd(&control.validCount,1u);
}
@compute @workgroup_size(1) fn publishPowerDescriptors(){let requested=atomicLoad(&control.rowCount);let available=min(requested,min(arrayLength(&headers),arrayLength(&descriptors)));if(available<requested){atomicAdd(&control.errorCount,requested-available);atomicMin(&control.firstInvalid,available);atomicOr(&control.flags,CAPACITY);}if(arrayLength(&indirectDispatch)<3u){atomicOr(&control.flags,CAPACITY);return;}if(atomicLoad(&control.errorCount)!=0u||atomicLoad(&control.validCount)!=requested){indirectDispatch[0]=0u;indirectDispatch[1]=1u;indirectDispatch[2]=1u;return;}let groups=(requested+63u)/64u;let x=min(groups,65535u);indirectDispatch[0]=x;indirectDispatch[1]=select(1u,(groups+x-1u)/x,x>0u);indirectDispatch[2]=1u;}
`;

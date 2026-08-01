/**
 * Authoritative GPU producer for paper-compatible octree power descriptors.
 * It consumes the compact LeafHeader and immutable owner-page publications.
 * Invalid topology publishes zero indirect work, and domain boundaries remain
 * explicit descriptor metadata for the immutable catalog lookup.
 */

import {
  OCTREE_POWER_NEIGHBOR_DIRECTIONS,
  encodeSameOrCoarserPowerDescriptor,
  encodeSameOrFinerPowerDescriptor,
} from "./octree-power-descriptor";
import {
  OCTREE_OWNER_ARENA_MAGIC,
  OCTREE_OWNER_PAGE_CONTROL_WORDS,
  OCTREE_OWNER_PAGE_PUBLICATION_STATUS,
  unpackOctreeOwnerPageControl,
} from "./webgpu-octree-owner-pages";
import { PassBroker } from "./webgpu-pass-broker";

export const OCTREE_POWER_LEAF_HEADER_BYTES = 48;
export const OCTREE_POWER_DESCRIPTOR_CONTROL_BYTES = 32;
export const OCTREE_POWER_DESCRIPTOR_DISPATCH_BYTES = 12;
// WGSL ControlArena: two 32-byte controls followed by two independently
// 16-byte-aligned vec3u dispatches and two scalar counters.
const OCTREE_POWER_DESCRIPTOR_CONTROL_ARENA_BYTES = 112;
export const OCTREE_POWER_DESCRIPTOR_INVALID = 0xffff_ffff;
export const OCTREE_POWER_ROW_DELTA_VALID = 0x5244_4c54;
export const OCTREE_POWER_ROW_DELTA_CONTROL_WORDS = 16;
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
  /** Reserved ABI bit retained for decoding older diagnostic captures. */
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

function descriptorPublicationArenaBytes(rowCapacity: number): number {
  const blocks = Math.ceil(rowCapacity / 256);
  // Public dispatch, row status/prefix/publication lists, block offsets, and
  // eight-word exact block summaries. All regions are u32-addressed in WGSL.
  return (4 + 3 * rowCapacity + blocks + 1 + 9 * blocks) * 4;
}

/**
 * Exact immutable old/new row transaction produced with the compact frontier.
 *
 * `rows` contains a 16-word control header at `controlOffsetWords`, followed
 * by the total maps and compact lists at the supplied offsets.  Header words:
 * current, previous, carried, added, retired, dirty, affected, generation,
 * valid, dirty-dispatch xyz, affected-dispatch xyz, reserved.
 */
export interface OctreePowerRowDeltaSource {
  readonly rows: GPUBuffer;
  readonly rowCapacity: number;
  readonly controlOffsetWords: number;
  readonly newToOldOffsetWords: number;
  readonly oldToNewOffsetWords: number;
  readonly dirtyRowsOffsetWords: number;
  readonly affectedRowsOffsetWords: number;
}

/** Assert the complete fixed-capacity row-delta buffer ABI before encoding copies or passes. */
export function assertOctreePowerRowDeltaLayout(
  source: OctreePowerRowDeltaSource,
  rowCapacity: number,
  label: string,
): void {
  const spans = [
    [source.controlOffsetWords, OCTREE_POWER_ROW_DELTA_CONTROL_WORDS],
    [source.newToOldOffsetWords, rowCapacity],
    [source.oldToNewOffsetWords, rowCapacity],
    [source.dirtyRowsOffsetWords, rowCapacity],
    [source.affectedRowsOffsetWords, rowCapacity],
  ] as const;
  const invalidSpan = spans.some(([offset, count]) =>
    !Number.isSafeInteger(offset) || offset < 0 || offset > 0xffff_ffff
    || offset + count > 0x1_0000_0000);
  const requiredWords = invalidSpan ? Number.POSITIVE_INFINITY
    : Math.max(...spans.map(([offset, count]) => offset + count));
  if (source.rowCapacity !== rowCapacity || invalidSpan
    || !Number.isSafeInteger(source.rows.size) || source.rows.size < requiredWords * 4) {
    throw new RangeError(`${label} row-delta layout is invalid`);
  }
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

export interface OctreePowerDescriptorEncodeOptions {
  readonly dimensions: readonly [number, number, number];
  readonly maximumLeafSize: number;
  /** Owner lifecycle scratch naming the complete inactive page table. */
  readonly ownerCandidateControl?: GPUBuffer;
  readonly generation?: number;
  /** Exact old/new row mapping and dirty plus one-ring work publication. */
  readonly rowDelta: OctreePowerRowDeltaSource;
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
    allocatedBytes: 2 * descriptorBytes + OCTREE_POWER_DESCRIPTOR_CONTROL_ARENA_BYTES
      + descriptorPublicationArenaBytes(rowCapacity) + OCTREE_POWER_DESCRIPTOR_DISPATCH_BYTES + 48,
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

/**
 * Validate the owner arena's own immutable publication transaction.
 *
 * Owner pages are topology state, so their accepted generation is not in the
 * power-descriptor generation namespace. A descriptor generation may advance
 * while the exact same topology publication remains current. Admission is
 * therefore self-contained: the arena must report one clean committed
 * generation, no rejected candidate, exact free/resident accounting, and no
 * invalid entry. There is no previous-generation fallback.
 */
export function octreePowerOwnerArenaPublicationIsValid(words: ArrayLike<number>): boolean {
  if (words.length < 16) return false;
  const control = unpackOctreeOwnerPageControl(words);
  return control.magic === OCTREE_OWNER_ARENA_MAGIC
    && control.acceptedGeneration !== 0
    && (control.status & 0x7fff_ffff) === OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready
    && control.observedGeneration === control.acceptedGeneration
    && control.candidateError === 0
    && control.invalidEntryCount === 0
    && control.residentCount <= control.capacity
    && control.freeCount === control.capacity - control.residentCount;
}

function contains(owner: OctreePowerOwner, cell: readonly [number, number, number], dimensions: readonly [number, number, number]): boolean {
  return supportedLeafSize(owner.size)
    && owner.origin.every((value, axis) => Number.isSafeInteger(value) && value >= 0
      && value % owner.size === 0 && value + owner.size <= dimensions[axis]
      && cell[axis] >= value && cell[axis] < value + owner.size);
}

/**
 * Strictly decode one simulation owner-page payload word for descriptor
 * construction. Required face/edge neighbours are materialized by the
 * topology-tile owner transaction; a zero or malformed word is therefore an
 * incomplete publication and must fail closed.
 */
export function decodePagedOctreePowerOwner(
  wordValue: number,
  cell: readonly [number, number, number],
  dimensions: readonly [number, number, number],
  maximumLeafSize: number,
): OctreePowerOwner {
  const invalid = (): OctreePowerOwner => ({ origin: [...cell] as [number, number, number], size: 1, invalid: true });
  const word = wordValue >>> 0;
  if ((word & 0x8000_0000) === 0) return invalid();
  const exponent = (word >>> 18) & 7;
  if (exponent > 5) return invalid();
  const size = 1 << exponent;
  const brickOrigin = cell.map((value) => Math.floor(value / 8) * 8);
  const delta = [word & 63, (word >>> 6) & 63, (word >>> 12) & 63].map((value) => value - 32);
  const owner: OctreePowerOwner = {
    origin: brickOrigin.map((value, axis) => value + delta[axis]) as [number, number, number],
    size,
  };
  return size <= maximumLeafSize && contains(owner, cell, dimensions) ? owner : invalid();
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
  return { descriptor: (encodeSameOrCoarserPowerDescriptor({ child, coarseNeighbors })
    | (boundaryMask << OCTREE_POWER_DESCRIPTOR_BOUNDARY_SHIFT)) >>> 0, flags: 0, kind: "same-or-coarser" };
}

interface OctreePowerDescriptorPipelineBundle {
  readonly layout: GPUBindGroupLayout;
  readonly preparePipeline: GPUComputePipeline;
  readonly generatePipeline: GPUComputePipeline;
  readonly stagePipeline: GPUComputePipeline;
  readonly prefixPipeline: GPUComputePipeline;
  readonly scatterPipeline: GPUComputePipeline;
  readonly commitPipeline: GPUComputePipeline;
  readonly finalizePipeline: GPUComputePipeline;
}

const octreePowerDescriptorPipelineCache = new WeakMap<GPUDevice,
  OctreePowerDescriptorPipelineBundle>();

export class WebGPUOctreePowerDescriptor {
  readonly plan: OctreePowerDescriptorPlan;
  readonly descriptors: GPUBuffer;
  /** Complete inactive candidate; never visible to active consumers before commit. */
  readonly candidateDescriptors: GPUBuffer;
  readonly control: GPUBuffer;
  readonly dispatch: GPUBuffer;
  private readonly workDispatch: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly generatePipeline: GPUComputePipeline;
  private readonly stagePipeline: GPUComputePipeline;
  private readonly prefixPipeline: GPUComputePipeline;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly commitPipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private readonly layout: GPUBindGroupLayout;
  private cachedGroup?: {
    leafHeaders: GPUBuffer;
    owners: GPUBuffer;
    rowDelta: GPUBuffer;
    ownerCandidate: GPUBuffer;
    group: GPUBindGroup;
  };
  private destroyed = false;

  constructor(private readonly device: GPUDevice, rowCapacity: number) {
    this.plan = planOctreePowerDescriptors(rowCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.descriptors = device.createBuffer({ label: "Octree power descriptors", size: this.plan.descriptorBytes, usage: storage });
    this.candidateDescriptors = device.createBuffer({ label: "Octree power descriptor candidates", size: this.plan.descriptorBytes, usage: storage });
    this.control = device.createBuffer({ label: "Octree power descriptor control transaction",
      size: OCTREE_POWER_DESCRIPTOR_CONTROL_ARENA_BYTES, usage: storage });
    this.dispatch = device.createBuffer({
      label: "Octree power descriptor publication arena",
      size: descriptorPublicationArenaBytes(this.plan.rowCapacity),
      usage: storage | GPUBufferUsage.INDIRECT,
    });
    this.workDispatch = device.createBuffer({ label: "Octree power descriptor work dispatch", size: this.plan.dispatchBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.params = device.createBuffer({ label: "Octree power descriptor parameters", size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    let pipelines = octreePowerDescriptorPipelineCache.get(device);
    if (!pipelines) {
      const layout = device.createBindGroupLayout({ label: "Octree power descriptor layout", entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ] });
      const shaderModule = device.createShaderModule({
        label: "Octree power descriptor generation", code: octreePowerDescriptorShader });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
      pipelines = {
        layout,
        preparePipeline: device.createComputePipeline({ label: "Prepare octree power descriptors", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "preparePowerDescriptors" } }),
        generatePipeline: device.createComputePipeline({ label: "Generate octree power descriptors", layout: pipelineLayout, compute: { module: shaderModule, entryPoint: "generatePowerDescriptors" } }),
        stagePipeline: device.createComputePipeline({ label: "Stage exact octree power descriptor delta", layout: pipelineLayout,
          compute: { module: shaderModule, entryPoint: "stagePowerDescriptorDelta" } }),
        prefixPipeline: device.createComputePipeline({ label: "Prefix exact octree power descriptor blocks", layout: pipelineLayout,
          compute: { module: shaderModule, entryPoint: "prefixPowerDescriptorDelta" } }),
        scatterPipeline: device.createComputePipeline({ label: "Scatter exact octree power descriptor publication rows", layout: pipelineLayout,
          compute: { module: shaderModule, entryPoint: "scatterPowerDescriptorDelta" } }),
        commitPipeline: device.createComputePipeline({ label: "Commit changed octree power descriptors", layout: pipelineLayout,
          compute: { module: shaderModule, entryPoint: "commitPowerDescriptors" } }),
        finalizePipeline: device.createComputePipeline({ label: "Finalize octree power descriptors", layout: pipelineLayout,
          compute: { module: shaderModule, entryPoint: "finalizePowerDescriptors" } }),
      };
      octreePowerDescriptorPipelineCache.set(device, pipelines);
    }
    this.layout = pipelines.layout;
    this.preparePipeline = pipelines.preparePipeline;
    this.generatePipeline = pipelines.generatePipeline;
    this.stagePipeline = pipelines.stagePipeline;
    this.prefixPipeline = pipelines.prefixPipeline;
    this.scatterPipeline = pipelines.scatterPipeline;
    this.commitPipeline = pipelines.commitPipeline;
    this.finalizePipeline = pipelines.finalizePipeline;
  }

  encodeCandidate(broker: PassBroker, leafHeaders: GPUBuffer, owners: GPUBuffer,
    options: OctreePowerDescriptorEncodeOptions, candidateOwnerView = true): void {
    if (this.destroyed) throw new Error("Octree power descriptor generator is destroyed");
    options.dimensions.forEach((value, axis) => positiveInteger(value, `Power descriptor dimension ${axis}`));
    if (options.dimensions.some((value) => value > 0x7fff_ffff)
      || options.dimensions[0] * options.dimensions[1] * options.dimensions[2] > 0xffff_ffff) {
      throw new RangeError("Power descriptor dimensions must fit the u32 linear-cell ABI");
    }
    positiveInteger(options.maximumLeafSize, "Power descriptor maximum leaf size");
    if (!supportedLeafSize(options.maximumLeafSize)) throw new RangeError("Power descriptor maximum leaf size is unsupported");
    const generation = options.generation ?? 0;
    if (!Number.isSafeInteger(generation) || generation < 0 || generation > 0xffff_ffff) throw new RangeError("Power descriptor generation must be unsigned");
    const delta = options.rowDelta;
    assertOctreePowerRowDeltaLayout(delta, this.plan.rowCapacity, "Power descriptor");
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([
      ...options.dimensions, options.maximumLeafSize, this.plan.rowCapacity, generation, this.plan.rowCapacity, 0,
      delta.controlOffsetWords, delta.newToOldOffsetWords, delta.dirtyRowsOffsetWords,
      candidateOwnerView ? 1 : 0,
    ]));
    let cached = this.cachedGroup;
    if (!cached || cached.leafHeaders !== leafHeaders || cached.owners !== owners
      || cached.rowDelta !== delta.rows
      || cached.ownerCandidate !== (options.ownerCandidateControl ?? owners)) {
      cached = { leafHeaders, owners, rowDelta: delta.rows,
        ownerCandidate: options.ownerCandidateControl ?? owners,
        group: this.device.createBindGroup({ label: "Octree power descriptor bindings", layout: this.layout, entries: [
          { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: { buffer: leafHeaders } },
          { binding: 2, resource: { buffer: owners } }, { binding: 3, resource: { buffer: this.candidateDescriptors } },
          { binding: 4, resource: { buffer: this.control } }, { binding: 5, resource: { buffer: this.dispatch } },
          { binding: 7, resource: { buffer: delta.rows } },
          { binding: 8, resource: { buffer: this.descriptors } },
          { binding: 9, resource: { buffer: options.ownerCandidateControl ?? owners } },
        ] }) };
      this.cachedGroup = cached;
    }
    const group = cached.group;
    let pass = broker.compute({ label: "Prepare power descriptor control" });
    pass.setPipeline(this.preparePipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1);
    broker.copyBufferToBuffer(delta.rows, (delta.controlOffsetWords + 9) * 4,
      this.workDispatch, 0, this.plan.dispatchBytes);
    pass = broker.compute({ label: "Resolve structurally dirty power descriptors" });
    pass.setPipeline(this.generatePipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(this.workDispatch, 0);
    broker.copyBufferToBuffer(this.control, 64, this.workDispatch, 0, this.plan.dispatchBytes);
    pass = broker.compute({ label: "Stage exact power descriptor carry and affected rows" });
    pass.setPipeline(this.stagePipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(this.workDispatch, 0);
    pass.setPipeline(this.prefixPipeline); pass.dispatchWorkgroups(1);
    pass.setPipeline(this.scatterPipeline);
    pass.dispatchWorkgroupsIndirect(this.workDispatch, 0);
    broker.copyBufferToBuffer(this.control, 80, this.workDispatch, 0, this.plan.dispatchBytes);
  }

  /** Publish the previously validated inactive descriptor candidate. */
  encodeReadyCommit(broker: PassBroker): void {
    if (this.destroyed) throw new Error("Octree power descriptor generator is destroyed");
    const group = this.cachedGroup?.group;
    if (!group) throw new Error("Power descriptor candidate has not been encoded");
    const pass = broker.compute({ label: "Commit power descriptor publication" });
    pass.setPipeline(this.commitPipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(this.workDispatch, 0);
    pass.setPipeline(this.finalizePipeline); pass.dispatchWorkgroups(1);
  }

  encode(broker: PassBroker, leafHeaders: GPUBuffer, owners: GPUBuffer,
    options: OctreePowerDescriptorEncodeOptions): void {
    this.encodeCandidate(broker, leafHeaders, owners, options, false);
    this.encodeReadyCommit(broker);
  }

  /** Rejection-only readback for the exact candidate descriptor/status named
   * by the control transaction. It never participates in publication. */
  async readCandidateFailure(row: number) {
    if (!Number.isSafeInteger(row) || row < 0 || row >= this.plan.rowCapacity) return undefined;
    const readback = this.device.createBuffer({
      label: "Rejected power descriptor candidate",
      size: 12,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({
      label: "Read rejected power descriptor candidate",
    });
    encoder.copyBufferToBuffer(this.candidateDescriptors, row * 4, readback, 0, 4);
    encoder.copyBufferToBuffer(this.descriptors, row * 4, readback, 4, 4);
    encoder.copyBufferToBuffer(this.dispatch, (4 + row) * 4, readback, 8, 4);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const encoded = words[0] >>> 0;
      return {
        row,
        candidate: encoded,
        committed: words[1] >>> 0,
        status: words[2] >>> 0,
        flags: (encoded & 0x4000_0000) !== 0 ? encoded & 127 : 0,
        detail: (encoded & 0x4000_0000) !== 0 ? (encoded >>> 7) & 0x007f_ffff : 0,
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.descriptors.destroy(); this.candidateDescriptors.destroy(); this.control.destroy(); this.dispatch.destroy(); this.workDispatch.destroy();
    this.params.destroy();
  }
}

/**
 * Descriptor resolution follows the paper's compact 1-ring topology key.
 * Only affected rows are resolved. Clean rows retain their immutable prior
 * descriptors by exact row identity. Sorted insertion/removal can move a clean
 * identity; that representation-only relocation is detected from newToOld and
 * committed without re-resolving topology. The unchanged transaction publishes
 * zero descriptor work.
 */
export const octreePowerDescriptorShader = /* wgsl */ `
struct Params { dimensionsMaximumLeaf:vec4u, rowCountGenerationCapacityReserved:vec4u, delta:vec4u }
struct LeafHeader { cell:u32, entryStart:u32, entryCount:u32, size:u32, diagonal:f32, rhs:f32, pad0:u32, pad1:u32, gradient:vec4f }
struct Owner { origin:vec3u, size:u32, invalid:u32 }
struct Control { rowCount:u32, validCount:u32, errorCount:u32, firstInvalid:u32, flags:u32, sameOrFinerCount:u32, sameOrCoarserCount:u32, generation:u32 }
// The first control is the public result of the current attempt. The hidden
// authority retains the last accepted counters so a rejected attempt can be
// diagnosed without permitting stale descriptors to look current.
struct ControlArena {
  candidate:Control,
  authority:Control,
  blockDispatch:vec3u,
  commitDispatch:vec3u,
  publicationCount:u32,
  blockCount:u32,
}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(2) var<storage,read> owners:array<u32>;
@group(0) @binding(3) var<storage,read_write> descriptors:array<u32>;
@group(0) @binding(4) var<storage,read_write> controlArena:ControlArena;
@group(0) @binding(5) var<storage,read_write> indirectDispatch:array<u32>;
@group(0) @binding(7) var<storage,read> rowDelta:array<u32>;
@group(0) @binding(8) var<storage,read_write> committedDescriptors:array<u32>;
@group(0) @binding(9) var<storage,read> ownerCandidate:array<u32>;
const INVALID:u32=0xffffffffu;
const ROW_DELTA_VALID:u32=${OCTREE_POWER_ROW_DELTA_VALID}u;
const OWNER_MAGIC:u32=${OCTREE_OWNER_ARENA_MAGIC}u;
const OWNER_PUBLICATION_READY:u32=${OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready}u;
const OWNER_VALID:u32=0x80000000u;
const OWNER_CONTROL_WORDS:u32=16u;
const OWNER_FREE_COUNT:u32=${OCTREE_OWNER_PAGE_CONTROL_WORDS.freeCount}u;
const OWNER_RESIDENT_COUNT:u32=${OCTREE_OWNER_PAGE_CONTROL_WORDS.residentCount}u;
const OWNER_CANDIDATE_ERROR:u32=${OCTREE_OWNER_PAGE_CONTROL_WORDS.candidateError}u;
const OWNER_CAPACITY:u32=${OCTREE_OWNER_PAGE_CONTROL_WORDS.capacity}u;
const OWNER_RECORD_PAGE_OFFSET:u32=${OCTREE_OWNER_PAGE_CONTROL_WORDS.ownerRecordPageOffsetWords}u;
const OWNER_PAGES_OFFSET:u32=${OCTREE_OWNER_PAGE_CONTROL_WORDS.ownerPagesOffsetWords}u;
const OWNER_ACCEPTED_GENERATION:u32=${OCTREE_OWNER_PAGE_CONTROL_WORDS.acceptedGeneration}u;
const OWNER_PUBLICATION_STATUS:u32=${OCTREE_OWNER_PAGE_CONTROL_WORDS.status}u;
const OWNER_OBSERVED_GENERATION:u32=${OCTREE_OWNER_PAGE_CONTROL_WORDS.observedGeneration}u;
const OWNER_INVALID_ENTRY_COUNT:u32=${OCTREE_OWNER_PAGE_CONTROL_WORDS.invalidEntryCount}u;
const OWNER_ARENA_MAGIC:u32=${OCTREE_OWNER_PAGE_CONTROL_WORDS.magic}u;
const OWNER_PAGE_VOXELS:u32=512u;
const COARSER_FLAG:u32=0x80000000u;
const MALFORMED_HEADER:u32=1u;
const MALFORMED_OWNER:u32=2u;
const MIXED_GRADING:u32=8u;
const GRADING_RATIO:u32=16u;
const CAPACITY:u32=32u;
const STATUS_VALID:u32=0x80000000u;
const STATUS_COARSER:u32=0x40000000u;
const STATUS_PUBLISH:u32=0x20000000u;
const STATUS_AFFECTED:u32=0x10000000u;
const STATUS_LISTED:u32=0x08000000u;
const DIRECTIONS:array<vec3i,18>=array<vec3i,18>(
  vec3i(-1,0,0),vec3i(0,-1,0),vec3i(0,0,-1),vec3i(0,0,1),vec3i(0,1,0),vec3i(1,0,0),
  vec3i(-1,-1,0),vec3i(-1,0,-1),vec3i(-1,0,1),vec3i(-1,1,0),vec3i(0,-1,-1),vec3i(0,-1,1),
  vec3i(0,1,-1),vec3i(0,1,1),vec3i(1,-1,0),vec3i(1,0,-1),vec3i(1,0,1),vec3i(1,1,0));
fn dims()->vec3u{return params.dimensionsMaximumLeaf.xyz;}
fn generation()->u32{let base=params.delta.x;return select(0u,rowDelta[base+7u],base+7u<arrayLength(&rowDelta));}
fn deltaAccepted(requested:u32)->bool{
  if(params.delta.x>arrayLength(&rowDelta)||arrayLength(&rowDelta)-params.delta.x<16u){return false;}
  let base=params.delta.x;let previous=rowDelta[base+1u];let carried=rowDelta[base+2u];
  let added=rowDelta[base+3u];let retired=rowDelta[base+4u];let dirty=rowDelta[base+5u];
  let mapsFit=params.delta.y<=arrayLength(&rowDelta)&&requested<=arrayLength(&rowDelta)-params.delta.y;
  let dirtyFits=params.delta.z<=arrayLength(&rowDelta)&&dirty<=arrayLength(&rowDelta)-params.delta.z;
  return mapsFit&&dirtyFits&&rowDelta[base]==requested
    &&rowDelta[base+7u]==generation()&&rowDelta[base+8u]==ROW_DELTA_VALID
    &&carried<=previous&&dirty<=requested
    &&requested==carried+added&&requested==previous+added-retired;
}
fn dirtyRow(item:u32)->u32{return rowDelta[params.delta.z+item];}
fn supportedSize(size:u32)->bool{return size==1u||size==2u||size==4u||size==8u||size==16u||size==32u;}
fn volumeFits()->bool{let d=dims();return d.x!=0u&&d.y!=0u&&d.z!=0u&&d.x<=0xffffffffu/d.y&&d.x*d.y<=0xffffffffu/d.z;}
fn cellCoord(cell:u32)->vec3u{let d=dims();return vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));}
fn ownerValid(owner:Owner,cell:vec3u)->bool{return owner.invalid==0u&&supportedSize(owner.size)&&owner.size<=params.dimensionsMaximumLeaf.w&&all(vec3u(owner.size)<=dims())&&all(owner.origin<=dims()-vec3u(owner.size))&&all(owner.origin%vec3u(owner.size)==vec3u(0u))&&all(cell>=owner.origin)&&all(cell<owner.origin+vec3u(owner.size));}
fn decodePagedOwner(word:u32,cell:vec3u)->Owner{
  if((word&OWNER_VALID)==0u){return Owner(cell,1u,1u);}
  let exponent=(word>>18u)&7u;if(exponent>5u){return Owner(cell,1u,1u);}
  let brickOrigin=vec3i((cell/vec3u(8u))*vec3u(8u));
  let delta=vec3i(i32(word&63u)-32,i32((word>>6u)&63u)-32,i32((word>>12u)&63u)-32);
  let signedOrigin=brickOrigin+delta;if(any(signedOrigin<vec3i(0))){return Owner(cell,1u,1u);}
  let result=Owner(vec3u(signedOrigin),1u<<exponent,0u);
  return Owner(result.origin,result.size,select(1u,0u,ownerValid(result,cell)));
}
fn pagedOwnerPublicationValid()->bool{
  if(arrayLength(&owners)<OWNER_CONTROL_WORDS||owners[OWNER_ARENA_MAGIC]!=OWNER_MAGIC){return false;}
  let capacity=owners[OWNER_CAPACITY];let resident=owners[OWNER_RESIDENT_COUNT];
  if(params.delta.w!=0u){return arrayLength(&ownerCandidate)>28u&&ownerCandidate[28u]==OWNER_VALID
    &&ownerCandidate[23u]==0u&&ownerCandidate[24u]==generation()&&ownerCandidate[27u]<=capacity;}
  let accepted=owners[OWNER_ACCEPTED_GENERATION];
  return accepted!=0u&&(owners[OWNER_PUBLICATION_STATUS]&0x7fffffffu)==OWNER_PUBLICATION_READY
    &&owners[OWNER_OBSERVED_GENERATION]==accepted&&owners[OWNER_CANDIDATE_ERROR]==0u
    &&owners[OWNER_INVALID_ENTRY_COUNT]==0u&&resident<=capacity
    &&owners[OWNER_FREE_COUNT]==capacity-resident;
}
fn pagedOwner(cell:vec3u)->Owner{
  if(!pagedOwnerPublicationValid()){return Owner(cell,1u,1u);}
  let capacity=owners[OWNER_CAPACITY];let logicalCount=owners[4u];
  let recordKeyOffset=OWNER_CONTROL_WORDS;let recordPageOffset=owners[OWNER_RECORD_PAGE_OFFSET];
  let table=select(owners[OWNER_PUBLICATION_STATUS]>>31u,1u-(owners[OWNER_PUBLICATION_STATUS]>>31u),params.delta.w!=0u);
  let directoryOffset=recordPageOffset+3u*capacity+table*logicalCount;
  let payloadOffset=owners[OWNER_PAGES_OFFSET]+table*capacity*OWNER_PAGE_VOXELS;
  if(capacity==0u||recordPageOffset!=recordKeyOffset+capacity
      ||owners[OWNER_PAGES_OFFSET]!=recordPageOffset+3u*capacity+2u*logicalCount
      ||recordKeyOffset>arrayLength(&owners)||capacity>arrayLength(&owners)-recordKeyOffset
      ||recordPageOffset>arrayLength(&owners)||capacity>arrayLength(&owners)-recordPageOffset
      ||directoryOffset>arrayLength(&owners)||logicalCount>arrayLength(&owners)-directoryOffset
      ||payloadOffset>=arrayLength(&owners)){return Owner(cell,1u,1u);}
  let bd=(dims()+vec3u(7u))/8u;let brick=cell/8u;
  if(any(brick>=bd)||bd.x>0xffffffffu/bd.y){return Owner(cell,1u,1u);}
  let layer=bd.x*bd.y;if(brick.z>0xffffffffu/layer){return Owner(cell,1u,1u);}
  let logical=brick.x+brick.y*bd.x+brick.z*layer;
  if(logical>=logicalCount){return Owner(cell,1u,1u);}
  let encoded=owners[directoryOffset+logical];
  if(encoded==0u||encoded==INVALID||encoded>capacity){return Owner(cell,1u,1u);}
  let physical=encoded-1u;let local=cell%vec3u(8u);let localIndex=local.x+local.y*8u+local.z*64u;
  if(physical>(arrayLength(&owners)-payloadOffset-1u)/OWNER_PAGE_VOXELS){return Owner(cell,1u,1u);}
  let at=payloadOffset+physical*OWNER_PAGE_VOXELS+localIndex;
  if(at>=arrayLength(&owners)){return Owner(cell,1u,1u);}
  return decodePagedOwner(owners[at],cell);
}
fn ownerAt(cell:vec3u)->Owner{
  return pagedOwner(cell);
}
fn failRow(row:u32,flags:u32,detail:u32){
  descriptors[row]=0x40000000u|(flags&127u)|(detail<<7u);
}
fn succeedRow(row:u32,descriptor:u32){
  descriptors[row]=descriptor;
}
fn descriptorValid(descriptor:u32)->bool{return descriptor!=INVALID&&(descriptor&0x40000000u)==0u;}
fn descriptorFlags(descriptor:u32)->u32{return select(CAPACITY,descriptor&127u,(descriptor&0x40000000u)!=0u);}
fn dirtyCount()->u32{return rowDelta[params.delta.x+5u];}
fn dispatchFor(count:u32)->vec3u{
  let groups=(count+63u)/64u;let x=min(groups,65535u);
  return vec3u(x,select(1u,(groups+x-1u)/x,x>0u),1u);
}
fn blockDispatchFor(count:u32)->vec3u{
  let groups=(count+255u)/256u;let x=min(groups,65535u);
  return vec3u(x,select(1u,(groups+x-1u)/x,x>0u),1u);
}
fn blockCount()->u32{return (controlArena.candidate.rowCount+255u)/256u;}
fn statusBase()->u32{return 4u;}
fn prefixBase()->u32{return statusBase()+params.rowCountGenerationCapacityReserved.z;}
fn publicationBase()->u32{return prefixBase()+params.rowCountGenerationCapacityReserved.z;}
fn blockOffsetBase()->u32{return publicationBase()+params.rowCountGenerationCapacityReserved.z;}
fn blockSummaryBase()->u32{return blockOffsetBase()+blockCount()+1u;}
fn rejectCandidate(row:u32,flags:u32){
  controlArena.candidate.errorCount+=1u;
  controlArena.candidate.firstInvalid=min(controlArena.candidate.firstInvalid,row);
  controlArena.candidate.flags|=flags;
}
var<workgroup> publicationScan:array<u32,256>;
var<workgroup> publicationCounts:array<vec4u,256>;
var<workgroup> publicationFailures:array<vec4u,256>;
@compute @workgroup_size(1) fn preparePowerDescriptors(){
  let requested=rowDelta[params.delta.x];
  controlArena.candidate=Control(requested,0u,0u,INVALID,0u,0u,0u,generation());
  controlArena.blockDispatch=vec3u(0u,1u,1u);
  controlArena.commitDispatch=vec3u(0u,1u,1u);
  controlArena.publicationCount=0u;controlArena.blockCount=0u;
  if(arrayLength(&indirectDispatch)>=3u){indirectDispatch[0]=0u;indirectDispatch[1]=1u;indirectDispatch[2]=1u;}
  let available=min(params.rowCountGenerationCapacityReserved.z,min(arrayLength(&headers),min(arrayLength(&descriptors),arrayLength(&committedDescriptors))));
  if(available<requested){controlArena.candidate.errorCount=requested-available;controlArena.candidate.firstInvalid=available;controlArena.candidate.flags=CAPACITY;}
  if(!deltaAccepted(requested)){rejectCandidate(0u,CAPACITY);return;}
  let previous=rowDelta[params.delta.x+1u];
  if(previous>arrayLength(&committedDescriptors)
      ||(previous!=0u&&(controlArena.authority.rowCount!=previous
        ||controlArena.authority.validCount!=previous||controlArena.authority.errorCount!=0u
        ||controlArena.authority.flags!=0u))){
    rejectCandidate(0u,CAPACITY);return;
  }
  let blocks=(requested+255u)/256u;
  let required=blockSummaryBase()+9u*blocks;
  if(required>arrayLength(&indirectDispatch)){rejectCandidate(0u,CAPACITY);return;}
  controlArena.blockCount=blocks;controlArena.blockDispatch=blockDispatchFor(requested);
}
@compute @workgroup_size(64) fn generatePowerDescriptors(
    @builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) workgroups:vec3u,@builtin(local_invocation_index) lid:u32){
  if(controlArena.candidate.flags!=0u){return;}
  let item=(wid.x+wid.y*workgroups.x)*64u+lid;if(item>=dirtyCount()){return;}
  let row=dirtyRow(item);let requested=controlArena.candidate.rowCount;
  if(row>=requested||row>=arrayLength(&headers)||row>=arrayLength(&descriptors)){return;}
  indirectDispatch[statusBase()+row]=STATUS_LISTED;
  if(item>0u&&dirtyRow(item-1u)>=row){
    failRow(row,CAPACITY,0u);return;
  }
  let header=headers[row];
  if(!volumeFits()||!supportedSize(params.dimensionsMaximumLeaf.w)||!supportedSize(header.size)||header.size>params.dimensionsMaximumLeaf.w){failRow(row,MALFORMED_HEADER,1u);return;}
  let volume=dims().x*dims().y*dims().z;if(header.cell>=volume){failRow(row,MALFORMED_HEADER,1u);return;}
  let origin=cellCoord(header.cell);
  if(any(origin%vec3u(header.size)!=vec3u(0u))||any(vec3u(header.size)>dims())||any(origin>dims()-vec3u(header.size))){failRow(row,MALFORMED_HEADER,1u);return;}
  let anchor=ownerAt(origin);
  if(!ownerValid(anchor,origin)||anchor.size!=header.size||any(anchor.origin!=origin)){
    let reasons=select(0u,1u,anchor.invalid!=0u)|select(0u,2u,anchor.size!=header.size)|select(0u,4u,any(anchor.origin!=origin));
    failRow(row,MALFORMED_OWNER,2u|(min(anchor.size,63u)<<5u)|(reasons<<11u));return;
  }
  var sizes:array<u32,18>;var flags=0u;var firstDetail=0u;var boundaryMask=0u;var finer=false;var coarser=false;
  for(var bit=0u;bit<18u;bit+=1u){
    let direction=DIRECTIONS[bit];var probe=vec3i(0);
    for(var axis=0u;axis<3u;axis+=1u){probe[axis]=select(select(i32(origin[axis]+header.size/2u),i32(origin[axis]+header.size),direction[axis]>0),i32(origin[axis])-1,direction[axis]<0);}
    if(any(probe<vec3i(0))||any(probe>=vec3i(dims()))){if(bit<6u){boundaryMask|=1u<<bit;}sizes[bit]=header.size;continue;}
    let owner=ownerAt(vec3u(probe));var localFlags=0u;var reasons=0u;
    if(!ownerValid(owner,vec3u(probe))){localFlags|=MALFORMED_OWNER;reasons|=1u;}
    if(all(min(owner.origin+vec3u(owner.size),origin+vec3u(header.size))>max(owner.origin,origin))){localFlags|=MALFORMED_OWNER;reasons|=2u;}
    if(owner.size*2u<header.size||owner.size>header.size*2u){localFlags|=GRADING_RATIO;reasons|=4u;}
    if(localFlags!=0u&&firstDetail==0u){firstDetail=(bit+3u)|(min(owner.size,63u)<<5u)|(reasons<<11u);}
    flags|=localFlags;sizes[bit]=owner.size;finer=finer||owner.size<header.size;coarser=coarser||owner.size>header.size;
  }
  if(finer&&coarser){flags|=MIXED_GRADING;if(firstDetail==0u){firstDetail=31u;}}
  if(flags!=0u){failRow(row,flags,firstDetail);return;}
  var descriptor=boundaryMask<<24u;
  if(!coarser){
    for(var bit=0u;bit<18u;bit+=1u){if(sizes[bit]==header.size){descriptor|=1u<<bit;}}
  }else{
    let child=(origin/vec3u(header.size))&vec3u(1u);
    descriptor|=COARSER_FLAG|child.x|(child.y<<1u)|(child.z<<2u);
    let outward=vec3i(select(-1,1,child.x==1u),select(-1,1,child.y==1u),select(-1,1,child.z==1u));
    let wanted=array<vec3i,6>(vec3i(outward.x,0,0),vec3i(0,outward.y,0),vec3i(0,0,outward.z),vec3i(outward.x,outward.y,0),vec3i(outward.x,0,outward.z),vec3i(0,outward.y,outward.z));
    for(var coarseBit=0u;coarseBit<6u;coarseBit+=1u){for(var bit=0u;bit<18u;bit+=1u){if(all(DIRECTIONS[bit]==wanted[coarseBit])&&sizes[bit]==header.size*2u){descriptor|=1u<<(coarseBit+3u);}}}
  }
  succeedRow(row,descriptor);
}
@compute @workgroup_size(256) fn stagePowerDescriptorDelta(
    @builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) wid:vec3u,
    @builtin(num_workgroups) workgroups:vec3u){
  let block=wid.x+wid.y*workgroups.x;let row=block*256u+lane;let requested=controlArena.candidate.rowCount;
  var status=0u;
  if(row<requested&&controlArena.candidate.flags==0u){
    let encoded=rowDelta[params.delta.y+row];
    let listed=(indirectDispatch[statusBase()+row]&STATUS_LISTED)!=0u;
    let oldPlusOne=encoded&0x3fffffffu;var descriptor=INVALID;var flags=0u;
    if(listed){descriptor=descriptors[row];status|=STATUS_AFFECTED|STATUS_PUBLISH;}
    else if(oldPlusOne==0u){flags|=CAPACITY;}
    else{
      let old=oldPlusOne-1u;
      if(old>=controlArena.authority.rowCount||old>=arrayLength(&committedDescriptors)){flags|=CAPACITY;}
      else{
        descriptor=committedDescriptors[old];
        // Section 4 stores one normal-velocity degree of freedom on every
        // power-cell face (paper lines 305-307). Candidate consumers require
        // that complete face geometry even when an immutable row retains its
        // numeric index; a prior rejected attempt may have dirtied this
        // scratch slot, so same-index carry must materialize it as well.
        descriptors[row]=descriptor;
        if(old!=row){status|=STATUS_PUBLISH;}
      }
    }
    if(!descriptorValid(descriptor)){flags|=descriptorFlags(descriptor);}
    else{status|=STATUS_VALID;if((descriptor&COARSER_FLAG)!=0u){status|=STATUS_COARSER;}}
    status|=flags&127u;indirectDispatch[statusBase()+row]=status;
  }
  let publish=select(0u,1u,(status&STATUS_PUBLISH)!=0u);
  publicationScan[lane]=publish;workgroupBarrier();
  for(var offset=1u;offset<256u;offset*=2u){var add=0u;if(lane>=offset){add=publicationScan[lane-offset];}
    workgroupBarrier();publicationScan[lane]+=add;workgroupBarrier();}
  if(row<requested){indirectDispatch[prefixBase()+row]=publicationScan[lane]-publish;}
  let valid=select(0u,1u,(status&STATUS_VALID)!=0u);
  let coarser=select(0u,1u,(status&STATUS_COARSER)!=0u);
  let flags=status&127u;
  publicationCounts[lane]=vec4u(valid,valid-coarser,coarser,select(0u,1u,flags!=0u));
  publicationFailures[lane]=vec4u(flags,select(INVALID,row,flags!=0u),select(0u,1u,(status&STATUS_AFFECTED)!=0u),0u);
  workgroupBarrier();
  for(var stride=128u;stride>0u;stride/=2u){if(lane<stride){
      publicationCounts[lane]+=publicationCounts[lane+stride];
      publicationFailures[lane].x|=publicationFailures[lane+stride].x;
      publicationFailures[lane].y=min(publicationFailures[lane].y,publicationFailures[lane+stride].y);
      publicationFailures[lane].z+=publicationFailures[lane+stride].z;
    }workgroupBarrier();}
  if(lane==0u){let at=blockSummaryBase()+9u*block;let counts=publicationCounts[0];let failure=publicationFailures[0];
    indirectDispatch[at]=counts.x;indirectDispatch[at+1u]=counts.y;indirectDispatch[at+2u]=counts.z;
    indirectDispatch[at+3u]=counts.w;indirectDispatch[at+4u]=failure.x;indirectDispatch[at+5u]=failure.y;
    indirectDispatch[at+6u]=publicationScan[255];indirectDispatch[at+7u]=failure.z;indirectDispatch[at+8u]=0u;}
}
@compute @workgroup_size(256) fn prefixPowerDescriptorDelta(@builtin(local_invocation_index) lane:u32){
  let blocks=controlArena.blockCount;let chunk=(blocks+255u)/256u;let begin=min(blocks,lane*chunk);let end=min(blocks,begin+chunk);
  var counts=vec4u(0u);var failure=vec4u(0u,INVALID,0u,0u);var changed=0u;
  for(var block=begin;block<end;block+=1u){let at=blockSummaryBase()+9u*block;
    counts+=vec4u(indirectDispatch[at],indirectDispatch[at+1u],indirectDispatch[at+2u],indirectDispatch[at+3u]);
    failure.x|=indirectDispatch[at+4u];failure.y=min(failure.y,indirectDispatch[at+5u]);
    changed+=indirectDispatch[at+6u];failure.z+=indirectDispatch[at+7u];}
  publicationCounts[lane]=counts;publicationFailures[lane]=failure;publicationScan[lane]=changed;workgroupBarrier();
  for(var offset=1u;offset<256u;offset*=2u){var add=0u;if(lane>=offset){add=publicationScan[lane-offset];}
    workgroupBarrier();publicationScan[lane]+=add;workgroupBarrier();}
  var cursor=publicationScan[lane]-changed;
  for(var block=begin;block<end;block+=1u){let at=blockSummaryBase()+9u*block;
    indirectDispatch[blockOffsetBase()+block]=cursor;cursor+=indirectDispatch[at+6u];}
  for(var stride=128u;stride>0u;stride/=2u){if(lane<stride){
      publicationCounts[lane]+=publicationCounts[lane+stride];
      publicationFailures[lane].x|=publicationFailures[lane+stride].x;
      publicationFailures[lane].y=min(publicationFailures[lane].y,publicationFailures[lane+stride].y);
      publicationFailures[lane].z+=publicationFailures[lane+stride].z;
    }workgroupBarrier();}
  if(lane==0u){let total=publicationCounts[0];let failed=publicationFailures[0];let published=publicationScan[255];
    indirectDispatch[blockOffsetBase()+blocks]=published;controlArena.publicationCount=published;
    controlArena.candidate.validCount=total.x;controlArena.candidate.sameOrFinerCount=total.y;
    controlArena.candidate.sameOrCoarserCount=total.z;controlArena.candidate.errorCount+=total.w;
    controlArena.candidate.flags|=failed.x;controlArena.candidate.firstInvalid=min(controlArena.candidate.firstInvalid,failed.y);
    if(failed.z!=dirtyCount()||total.x!=controlArena.candidate.rowCount||total.y+total.z!=total.x){
      rejectCandidate(select(0u,failed.y,failed.y!=INVALID),CAPACITY);
    }
    if(controlArena.candidate.errorCount==0u&&controlArena.candidate.flags==0u){
      controlArena.commitDispatch=dispatchFor(published);
    }
  }
}
@compute @workgroup_size(256) fn scatterPowerDescriptorDelta(
    @builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) wid:vec3u,
    @builtin(num_workgroups) workgroups:vec3u){
  if(controlArena.candidate.errorCount!=0u||controlArena.candidate.flags!=0u){return;}
  let block=wid.x+wid.y*workgroups.x;let row=block*256u+lane;
  if(row>=controlArena.candidate.rowCount){return;}
  if((indirectDispatch[statusBase()+row]&STATUS_PUBLISH)==0u){return;}
  let output=indirectDispatch[blockOffsetBase()+block]+indirectDispatch[prefixBase()+row];
  if(output<controlArena.publicationCount){indirectDispatch[publicationBase()+output]=row;}
}
@compute @workgroup_size(64) fn commitPowerDescriptors(
    @builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) workgroups:vec3u,@builtin(local_invocation_index) lid:u32){
  if(controlArena.candidate.errorCount!=0u||controlArena.candidate.flags!=0u){return;}
  let item=(wid.x+wid.y*workgroups.x)*64u+lid;if(item>=controlArena.publicationCount){return;}
  let row=indirectDispatch[publicationBase()+item];
  if(row>=arrayLength(&descriptors)||row>=arrayLength(&committedDescriptors)){return;}
  committedDescriptors[row]=descriptors[row];
}
@compute @workgroup_size(1) fn finalizePowerDescriptors(){
  let candidate=controlArena.candidate;
  if(candidate.errorCount!=0u||candidate.flags!=0u||candidate.validCount!=candidate.rowCount){return;}
  controlArena.authority=candidate;
  if(arrayLength(&indirectDispatch)>=3u){
    let published=dispatchFor(candidate.rowCount);
    indirectDispatch[0]=published.x;indirectDispatch[1]=published.y;indirectDispatch[2]=published.z;
  }
}
`;
